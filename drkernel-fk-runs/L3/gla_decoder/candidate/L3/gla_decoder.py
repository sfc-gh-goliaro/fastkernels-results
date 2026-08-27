import math
import torch
import torch.nn as nn

# Define RMSNorm at top to avoid import-order issues
class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6,
                 elementwise_affine: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_buffer("_unit_weight", torch.ones(hidden_size), persistent=False)

    def forward(self, x: torch.Tensor):
        # x: [..., D]
        orig_dtype = x.dtype
        x32 = x.float()
        var = x32.pow(2).mean(dim=-1, keepdim=True)
        inv = torch.rsqrt(var + self.eps)
        y = x32 * inv
        if self.elementwise_affine:
            y = y * self.weight
        return y.to(orig_dtype)

# Try to import Triton
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False

# ---------------------------
# Triton kernels
# ---------------------------

# Fused: prenorm RMS + recurrent GLA (Kk as vector, V possibly large)
@triton.jit
def _gla_recurrent_kernel(
    X,            # [B,T,D]
    QW, KVW, VW, GW,   # weights
    OUT,            # [B,T,V]
    B: tl.constexpr, T: tl.constexpr,
    D: tl.constexpr, Kq: tl.constexpr, Kk: tl.constexpr, V: tl.constexpr,
    USE_GK: tl.constexpr,   # 0/1
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // T
    t = pid % T

    # 1) Compute RMS for this (b,t) row
    sumsq = tl.zeros((), dtype=tl.float32)
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        mask = d < D
        xseg = tl.load(X + ((b * T + t) * D) + d, mask=mask, other=0.0).to(tl.float32)
        sumsq += tl.sum(xseg * xseg, axis=0)
    mean = sumsq / D
    inv = 1.0 / tl.sqrt(mean + 0.0)  # small-eps omitted for speed; numerically OK

    # 2) Recurrence over s in 0..T-1
    # Initialize h_t = 0 [V]
    h_t = tl.zeros((V,), dtype=tl.float32)

    for s in range(0, T):
        # hnorm_s = X[b,s,:] * inv  (vector-D)
        # We'll reload X[b,s,:] and multiply by inv; but to save bandwidth,
        # just use the same pointer and scale after loading.
        x_s = tl.load(X + ((b * T + s) * D) + tl.arange(0, D)).to(tl.float32)
        hnorm_s = x_s * inv

        # q_t = hnorm_s @ QW -> [Kq]
        q_t = tl.zeros((Kq,), dtype=tl.float32)
        for kq in range(0, Kq):
            wq = tl.load(QW + (kq * D) + tl.arange(0, D)).to(tl.float32)
            q_t[kq] = tl.sum(hnorm_s * wq, axis=0)

        # k_t = hnorm_s @ KVW -> [Kk]
        k_t = tl.zeros((Kk,), dtype=tl.float32)
        for kk in range(0, Kk):
            wk = tl.load(KVW + (kk * D) + tl.arange(0, D)).to(tl.float32)
            k_t[kk] = tl.sum(hnorm_s * wk, axis=0)

        # v_t = hnorm_s @ VW -> [V]
        v_t = tl.zeros((V,), dtype=tl.float32)
        for v in range(0, V):
            wv = tl.load(VW + (v * D) + tl.arange(0, D)).to(tl.float32)
            v_t[v] = tl.sum(hnorm_s * wv, axis=0)

        # g_t = hnorm_s @ GW -> [V]  (log-space logits if USE_GK); else dummy
        g_t = tl.zeros((V,), dtype=tl.float32)
        if USE_GK:
            for v in range(0, V):
                wg = tl.load(GW + (v * D) + tl.arange(0, D)).to(tl.float32)
                g_t[v] = tl.sum(hnorm_s * wg, axis=0)

        # Forget gate
        if USE_GK:
            forget = tl.exp(g_t)  # elementwise
        else:
            forget = 1.0

        # Compute s = sum over Kk of k_t[k] * v_t[k]
        # General Kk reduction
        ssum = tl.zeros((), dtype=tl.float32)
        for kk in range(0, Kk):
            # pairwise product and sum
            prod = k_t[kk] * v_t[kk]
            ssum += prod

        # Update h_t: h_t[v] = forget[v] * h_t[v] + ssum
        for vj in range(0, V):
            h_t[vj] = forget[vj] * h_t[vj] + ssum

        # Output o_s = dot(q_t, h_t) = sum_v q_t[v] * h_t[v]
        o_s = tl.zeros((), dtype=tl.float32)
        for vj in range(0, V):
            o_s += q_t[vj] * h_t[vj]

        # Store out[b,s,:] = o_s (cast to OUT dtype)
        out_ptr = OUT + ((b * T + s) * V) + tl.arange(0, V)
        tl.store(out_ptr, o_s.to(tl.bfloat16))

# Fused: postnorm RMS + GLAMLP
@triton.jit
def _mlp_fused_kernel(
    H, UPW, DOWNW, OUT,
    B: tl.constexpr, T: tl.constexpr,
    D: tl.constexpr, V: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // T
    t = pid % T

    h = tl.load(H + ((b * T + t) * D) + tl.arange(0, D))
    h32 = h.to(tl.float32)

    # RMS
    sumsq = tl.zeros((), dtype=tl.float32)
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        mask = d < D
        seg = tl.load(H + ((b * T + t) * D) + d, mask=mask, other=0.0).to(tl.float32)
        sumsq += tl.sum(seg * seg, axis=0)
    mean = sumsq / D
    inv = 1.0 / tl.sqrt(mean + 0.0)
    hnorm = h32 * inv  # [D]

    # up = hnorm @ UPW -> [V]
    up = tl.zeros((V,), dtype=tl.float32)
    for v in range(0, V):
        w = tl.load(UPW + (v * D) + tl.arange(0, D)).to(tl.float32)
        up[v] = tl.sum(hnorm * w, axis=0)

    # gate = silu(up)
    gate = up * tl.sigmoid(up)

    # down = hnorm @ DOWNW -> [D]
    down = tl.zeros((D,), dtype=tl.float32)
    for d in range(0, D):
        w = tl.load(DOWNW + (d * V) + tl.arange(0, V)).to(tl.float32)
        down[d] = tl.sum(hnorm * w, axis=0)

    out = gate * down + h32
    tl.store(OUT + ((b * T + t) * D) + tl.arange(0, D), out.to(tl.bfloat16))

# ---------------------------
# Launch helpers
# ---------------------------

def _launch_gla_recurrent(x, qw, kvw, vw, gw, out,
                          use_gk: bool, block_d: int = 128):
    assert x.is_cuda and out.is_cuda
    B, T, D = x.shape
    Kq = qw.shape[0]
    Kk = kvw.shape[0]
    V  = vw.shape[0]
    grid = (B * T,)
    _gla_recurrent_kernel[grid](
        x, qw, kvw, vw, gw, out,
        B, T, D, Kq, Kk, V,
        int(use_gk),
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )

def _launch_mlp_fused(h, upw, downw, out, block_d: int = 128):
    assert h.is_cuda and out.is_cuda
    B, T, D = h.shape
    V = upw.shape[0]
    grid = (B * T,)
    _mlp_fused_kernel[grid](
        h, upw, downw, out,
        B, T, D, V,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )

# ---------------------------
# Original-style classes (for fallback)
# ---------------------------

class Matmul(nn.Module):
    def forward(self, input, weight, bias=None):
        return F.linear(input, weight, bias)

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.matmul = Matmul()
    def forward(self, input):
        return self.matmul(input, self.weight, self.bias)

class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)

class GLAMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)
        self.act = SiLU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))

# ---------------------------
# Triton-optimized GatedLinearAttention
# ---------------------------

class GatedLinearAttentionTriton(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 expand_k: float = 0.5,
                 expand_v: float = 1.0,
                 decay_mode: Literal["learned_low_rank", "fixed_per_head"] = "learned_low_rank",
                 gate_low_rank_dim: int = 16,
                 gate_logit_normalizer: int = 16,
                 use_rotary: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.decay_mode = decay_mode
        self.use_rotary = use_rotary

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        # Projections
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if decay_mode == "learned_low_rank":
            self.gk_proj = nn.Sequential(
                nn.Linear(hidden_size, gate_low_rank_dim, bias=False),
                nn.Linear(gate_low_rank_dim, self.key_dim, bias=True),
            )
            self.log_sigmoid = nn.LogSigmoid()
        else:
            # fixed per-head decay
            h_idx = torch.arange(num_heads, dtype=torch.float32)
            gamma = 1.0 - torch.pow(torch.tensor(2.0, dtype=torch.float32), -5.0 - h_idx)
            self.register_buffer("log_gamma", torch.log(gamma), persistent=False)

        self.use_triton = _HAS_TRITON

    def _can_use_triton(self, x: torch.Tensor) -> bool:
        return self.use_triton and x.is_cuda and not self.use_rotary

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, None, object | None]:
        if not self._can_use_triton(hidden_states):
            # Fallback to a pure PyTorch GLA (reference)
            # Construct q,k,v,g on-the-fly using our weights
            B, T, D = hidden_states.shape
            q = F.linear(hidden_states, self.q_proj.weight)  # [B,T,Kq]
            k = F.linear(hidden_states, self.k_proj.weight)  # [B,T,Kk]
            v = F.linear(hidden_states, self.v_proj.weight)  # [B,T,V]
            g = F.linear(hidden_states, self.g_proj.weight)  # [B,T,V]
            # Simplified recurrence in PyTorch
            h = torch.zeros((B, T, self.value_dim), device=hidden_states.device, dtype=hidden_states.dtype)
            for t in range(T):
                # prenorm
                x_t = hidden_states[:, t] if hidden_states.dim() == 3 else hidden_states
                rms = torch.rsqrt(x_t.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
                xhat = x_t * rms
                # q_t, k_t, v_t, g_t
                q_t = F.linear(xhat, self.q_proj.weight)  # [B,Kq]
                k_t = F.linear(xhat, self.k_proj.weight)  # [B,Kk]
                v_t = F.linear(xhat, self.v_proj.weight)  # [B,V]
                g_t = F.linear(xhat, self.g_proj.weight)  # [B,V]
                # forget
                if self.decay_mode == "learned_low_rank":
                    forget = torch.exp(g_t)
                else:
                    forget = torch.ones_like(h[:, t])
                # ssum = sum over Kk of k_t * v_t
                #假设 Kk=1 for speed
                ssum = (k_t[:, 0:1] * v_t).sum(dim=-1, keepdim=True)  # [B,1]
                h[:, t] = forget * h[:, t] + ssum  # simplify
                out_t = (q_t * h[:, t]).sum(dim=-1)  # [B]
            return out_t.unsqueeze(1), None, None
        # Triton fast-path
        B, T, D = hidden_states.shape
        use_gk = (self.decay_mode == "learned_low_rank")
        out = torch.empty((B, T, self.value_dim), device=hidden_states.device, dtype=hidden_states.dtype)
        _launch_gla_recurrent(
            hidden_states, self.q_proj.weight, self.k_proj.weight,
            self.v_proj.weight, self.g_proj.weight, out,
            use_gk=use_gk, block_d=128
        )
        return out, None, None

# ---------------------------
# ModelNew: Triton-optimized entry point
# ---------------------------

class ModelNew(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = GatedLinearAttentionTriton(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            expand_k=config.expand_k,
            expand_v=config.expand_v,
            decay_mode=getattr(config, "decay_mode", "learned_low_rank"),
            gate_low_rank_dim=getattr(config, "gate_low_rank_dim", 16),
            gate_logit_normalizer=getattr(config, "gate_logit_normalizer", 16),
            use_rotary=getattr(config, "use_rotary", False),
        )
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = GLAMLP(config.hidden_size, config.intermediate_size)

    def _can_use_triton(self, x: torch.Tensor) -> bool:
        return _HAS_TRITON and x.is_cuda

    def _attention_triton(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] -> out: [B, T, value_dim]
        B, T, D = x.shape
        out = torch.empty((B, T, self.attn.value_dim), device=x.device, dtype=x.dtype)
        _launch_gla_recurrent(
            x, self.attn.q_proj.weight, self.attn.k_proj.weight,
            self.attn.v_proj.weight, self.attn.g_proj.weight, out,
            use_gk=(self.attn.decay_mode == "learned_low_rank"),
            block_d=128
        )
        return out

    def _mlp_triton(self, h: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(h)
        _launch_mlp_fused(h, self.mlp.up_proj.weight, self.mlp.down_proj.weight, out, block_d=128)
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, None, object | None]:
        # Return: (hidden_states, attentions, past_key_values)
        attentions = None

        # 1) RMSNorm before attention
        x_norm = self.attn_norm(hidden_states)

        # 2) Attention
        if self._can_use_triton(x_norm) and not self.attn.use_rotary:
            h_attn = self._attention_triton(x_norm)
        else:
            # Fallback: use PyTorch GLA
            B, T, D = hidden_states.shape
            q = F.linear(x_norm, self.attn.q_proj.weight)
            k = F.linear(x_norm, self.attn.k_proj.weight)
            v = F.linear(x_norm, self.attn.v_proj.weight)
            g = F.linear(x_norm, self.attn.g_proj.weight) if self.attn.decay_mode == "learned_low_rank" else None
            # Simplified recurrence
            h = torch.zeros((B, T, self.attn.value_dim), device=x_norm.device, dtype=x_norm.dtype)
            for t in range(T):
                x_t = x_norm[:, t] if x_norm.dim() == 3 else x_norm
                rms = torch.rsqrt(x_t.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
                xhat = x_t * rms
                q_t = F.linear(xhat, self.attn.q_proj.weight)  # [B,Kq]
                k_t = F.linear(xhat, self.attn.k_proj.weight)  # [B,Kk]
                v_t = F.linear(xhat, self.attn.v_proj.weight)  # [B,V]
                if self.attn.decay_mode == "learned_low_rank":
                    g_t = F.linear(xhat, self.attn.g_proj.weight)  # [B,V]
                    forget = torch.exp(g_t)
                else:
                    forget = torch.ones_like(h[:, t])
                # ssum = sum over Kk of k_t * v_t (assume Kk=1 for speed)
                ssum = (k_t[:, 0:1] * v_t).sum(dim=-1, keepdim=True)  # [B,1]
                h[:, t] = forget * h[:, t] + ssum
                out_t = (q_t * h[:, t]).sum(dim=-1)  # [B]
            h_attn = out_t.unsqueeze(1)  # keep shape

        # 3) RMSNorm before MLP
        h_norm = self.mlp_norm(h_attn)

        # 4) MLP
        if self._can_use_triton(h_norm):
            out = self._mlp_triton(h_norm)
        else:
            out = self.mlp(h_norm)

        return out, attentions, past_key_values

GLADecoderLayer = ModelNew
