import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def _matmul_bias_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for the block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to blocks of A and B
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)
        # Advance by BLOCK_K along K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias if provided
    if Bias_ptr is not None:
        bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]

    # Store result
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _choose_blocks(M, N, K):
    # Simple heuristics for these problem sizes
    if K >= 1024:
        bk = 128
    elif K >= 256:
        bk = 64
    else:
        bk = 32
    if max(M, N) >= 1024:
        bm, bn = 128, 128
    else:
        bm, bn = 64, 64
    return bm, bn, bk


def _launch_matmul_bias(a, b, bias=None, out=None):
    """
    a: [M, K], b: [K, N], bias: [N] or None -> out: [M, N] float32
    """
    assert a.is_cuda and b.is_cuda, "Triton kernel requires CUDA tensors"
    # Enforce dtype and contiguity
    a_ = a.float().contiguous()
    b_ = b.float().contiguous()
    M, K = a_.shape
    Kb, N = b_.shape
    assert Kb == K, f"Incompatible shapes: a={a.shape}, b={b.shape}"
    if out is None:
        out = torch.empty((M, N), device=a_.device, dtype=torch.float32)
    else:
        assert out.shape == (M, N) and out.dtype == torch.float32 and out.device == a_.device

    # Strides in elements
    stride_am = a_.stride(0)
    stride_ak = a_.stride(1)
    stride_bk = b_.stride(0)
    stride_bn = b_.stride(1)
    stride_cm = out.stride(0)
    stride_cn = out.stride(1)

    bm, bn, bk = _choose_blocks(M, N, K)
    grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))

    _matmul_bias_kernel[grid](
        a_, b_, bias if bias is not None else None, out,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
        num_warps=4 if max(bm, bn) <= 64 else 8,
        num_stages=2,
    )
    return out


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class Matmul(nn.Module):
    """
    Functional linear that uses Triton GEMM when possible.
    Signature matches torch.nn.functional.linear:
      input:  [M, K]
      weight: [N, K]  (PyTorch stores as [out, in])
      bias:   [N] or None
    Returns: [M, N]
    """
    def forward(self, input, weight, bias=None):
        # If not CUDA, fallback
        if (not input.is_cuda) or (not weight.is_cuda):
            return F.linear(input, weight, bias)
        # Enforce float32, contiguous
        input_ = input.float().contiguous()
        weight_ = weight.float().contiguous()
        M, K = input_.shape
        N, Kw = weight_.shape
        assert Kw == K, f"Incompatible shapes: input={input.shape}, weight={weight.shape}"
        # Prepare B = weight^T as [K, N]
        b = weight_.transpose(0, 1).contiguous()
        # Launch
        out = _launch_matmul_bias(input_, b, bias=bias if bias is not None else None)
        return out


class Linear(nn.Module):
    """
    Parametric linear: stores weight and bias internally.
    Uses Triton matmul+bias when tensors are CUDA.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

    def forward(self, input):
        return Matmul()(input, self.weight, self.bias)


class CLIPMLP(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.fc1 = Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.activation_fn = QuickGELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class BMM(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)


class Softmax(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=self.dim)


class CLIPAttention(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections
        self.q_proj = Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = Linear(self.embed_dim, self.embed_dim, bias=True)

        self.bmm = BMM()
        self.softmax = Softmax(dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # hidden_states: [B, S, E]
        assert hidden_states.dim() == 3, f"Expected 3D tensor [B,S,E], got shape {hidden_states.shape}"
        batch_size, seq_length, _ = hidden_states.shape

        # Projections -> [B, S, E]
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        # Reshape to [B, S, H, Dh]
        H = self.num_heads
        Dh = self.head_dim
        queries = queries.view(batch_size, seq_length, H, Dh)
        keys = keys.view(batch_size, seq_length, H, Dh)
        values = values.view(batch_size, seq_length, H, Dh)

        # Concatenate heads into wide matrices to remove Python loops
        # Q_all: [B, S, H*Dh], K_all: [B, S, H*Dh]
        Q_all = queries.reshape(batch_size, seq_length, H * Dh).contiguous()
        K_all = keys.reshape(batch_size, seq_length, H * Dh).contiguous()

        # Compute scores = Q_all @ K_all^T -> [B, S, S]
        # Treat as many small GEMMs: A=[S, H*Dh], B=[H*Dh, S]
        B, S,wide = Q_all.shape
       宽 = wide
        scores = torch.empty((B, S, S), device=Q_all.device, dtype=torch.float32)
        for b in range(B):
            q_b = Q_all[b]     # [S, wide]
            k_b = K_all[b]     # [S, wide]
            kt = k_b.transpose(0, 1).contiguous()  # [wide, S]
            c = _launch_matmul_bias(q_b.float().contiguous(), kt, bias=None)  # [S, S]
            scores[b] = c * self.scale

        if attention_mask is not None:
            # attention_mask expected [B, 1, S, S] or [1,1,S,S]; broadcast to [B,S,S]
            scores = scores + attention_mask

        # Softmax over last dim (S): [B,S,S]
        attn = self.softmax(scores)

        # attn @ values: build V_all = [B, S, H*Dh]
        V_all = values.reshape(batch_size, seq_length, H * Dh).contiguous()
        # O_all = attn @ V_all^T -> [B, S, Dh]
        O_all = torch.empty((B, S, Dh), device=attn.device, dtype=torch.float32)
        for b in range(B):
            a_b = attn[b].float().contiguous()     # [S, S]
            v_b = V_all[b].float().contiguous()    # [S, Dh]
            vt = v_b.transpose(0, 1).contiguous()  # [Dh, S]
            c = _launch_matmul_bias(a_b, vt, bias=None)  # [S, Dh]
            O_all[b] = c

        # Restore to [B, S, E]
        attn_out = O_all.reshape(batch_size, seq_length, self.embed_dim)

        # Out projection (GEMM)
        attn_out = Matmul()(attn_out, self.out_proj.weight, self.out_proj.bias)
        return attn_out


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        create_scale: bool = True,
        create_offset: bool = True,
        promote_fp32: bool = True,
    ):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.promote_fp32 = promote_fp32

        if elementwise_affine and create_scale:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter("weight", None)

        if elementwise_affine and create_offset:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep PyTorch's layer_norm; it's already optimized
        return F.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps,
        )


class Model(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# Requested entry point
class ModelNew(Model):
    pass

CLIPEncoderLayer = ModelNew
