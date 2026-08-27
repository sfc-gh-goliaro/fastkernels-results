import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import Triton
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False

# =============================
# Triton kernels: elementwise LN, modulate, gate
# =============================
if _HAS_TRITON:
    @triton.jit
    def _layer_norm_forward_kernel(
        X_ptr,  # *f32 [N, D]
        W_ptr,  # *f32 [D] or dummy
        B_ptr,  # *f32 [D] or dummy
        Y_ptr,  # *f32 [N, D]
        stride_xn, stride_xd,
        stride_yn, stride_yd,
        stride_w,  # 0 if no affine
        stride_b,  # 0 if no bias
        N: tl.constexpr,
        D: tl.constexpr,
        EPS: tl.constexpr,
        HAS_AFFINE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= N:
            return
        x_row = X_ptr + pid * stride_xn
        y_row = Y_ptr + pid * stride_yn

        # mean
        sum_x = 0.0
        for k in range(0, D):
            xk = tl.load(x_row + k * stride_xd)
            sum_x += xk
        mean = sum_x / D

        # var
        sum_var = 0.0
        for k in range(0, D):
            xk = tl.load(x_row + k * stride_xd)
            diff = xk - mean
            sum_var += diff * diff
        var = sum_var / D
        rstd = 1.0 / tl.sqrt(var + EPS)

        # normalize + affine
        for k in range(0, D):
            xk = tl.load(x_row + k * stride_xd)
            y = (xk - mean) * rstd
            if HAS_AFFINE:
                wk = tl.load(W_ptr + k * stride_w)
                bk = tl.load(B_ptr + k * stride_b)
                y = y * wk + bk
            tl.store(y_row + k * stride_yd, y)

    @triton.jit
    def _modulate_forward_kernel(
        X_ptr,  # *f32 [N, D]
        S_ptr,  # *f32 [N, D]
        B_ptr,  # *f32 [N, D]
        Y_ptr,  # *f32 [N, D]
        stride_xn, stride_xd,
        stride_sn, stride_sd,
        stride_bn, stride_bd,
        stride_yn, stride_yd,
        N: tl.constexpr,
        D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= N:
            return
        x_row = X_ptr + pid * stride_xn
        s_row = S_ptr + pid * stride_sn
        b_row = B_ptr + pid * stride_bn
        y_row = Y_ptr + pid * stride_yn
        for k in range(0, D):
            xk = tl.load(x_row + k * stride_xd)
            sk = tl.load(s_row + k * stride_sd)
            bk = tl.load(b_row + k * stride_bd)
            y = xk * (1.0 + sk) + bk
            tl.store(y_row + k * stride_yd, y)

    @triton.jit
    def _gate_forward_kernel(
        X_ptr,  # *f32 [N, D]
        G_ptr,  # *f32 [N, D]
        Y_ptr,  # *f32 [N, D]
        stride_xn, stride_xd,
        stride_gn, stride_gd,
        stride_yn, stride_yd,
        N: tl.constexpr,
        D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= N:
            return
        x_row = X_ptr + pid * stride_xn
        g_row = G_ptr + pid * stride_gn
        y_row = Y_ptr + pid * stride_yn
        for k in range(0, D):
            xk = tl.load(x_row + k * stride_xd)
            gk = tl.load(g_row + k * stride_gd)
            y = gk * xk
            tl.store(y_row + k * stride_yd, y)

# =============================
# Triton kernel: GEMV Y[N, M] = X[N, D] @ W[M, D]^T + B[M]
# =============================
if _HAS_TRITON:
    @triton.jit
    def _linear_gemv_kernel(
        X_ptr,  # *f32 [N, D]
        W_ptr,  # *f32 [M, D]  (row-major)
        B_ptr,  # *f32 [M]    or dummy
        Y_ptr,  # *f32 [N, M]
        stride_xn, stride_xd,
        stride_wm, stride_wd,  # W strides
        stride_yn, stride_ym,
        N: tl.constexpr,
        D: tl.constexpr,
        M: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        # program ids
        pid_n = tl.program_id(0)  # row in N
        pid_m = tl.program_id(1)  # col in M tile
        if pid_n >= N:
            return

        # accumulator for this output row, over a tile of M
        acc = tl.zeros((M,), dtype=tl.float32)

        # loop over K=D
        for k in range(0, D):
            xk = tl.load(X_ptr + pid_n * stride_xn + k * stride_xd)
            for mm in range(0, M):
                wk = tl.load(W_ptr + mm * stride_wm + k * stride_wd)
                acc[mm] += xk * wk

        # add bias if present
        if HAS_BIAS:
            for mm in range(0, M):
                acc[mm] += tl.load(B_ptr + mm)

        # store
        for mm in range(0, M):
            tl.store(Y_ptr + pid_n * stride_yn + mm * stride_ym, acc[mm])


# =============================
# Utilities to dispatch Triton (force GPU when available)
# =============================
def _to_cuda_if_needed(t: torch.Tensor) -> torch.Tensor:
    return t if t.is_cuda else t.to("cuda")

def _to_cpu_if_needed(t: torch.Tensor, orig_device: torch.device) -> torch.Tensor:
    return t if t.device == orig_device else t.to(orig_device)

def _layer_norm_triton(x: torch.Tensor, weight: torch.Tensor | None, bias: torch.Tensor | None, eps: float = 1e-6) -> torch.Tensor:
    if not _HAS_TRITON:
        return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
    # Force CUDA
    x_cuda = _to_cuda_if_needed(x)
    D = x_cuda.shape[-1]
    N = x_cuda.numel() // D
    x2 = x_cuda.contiguous().view(N, D)
    y = torch.empty_like(x2)
    has_affine = (weight is not None) and (bias is not None)
    w = _to_cuda_if_needed(weight) if has_affine else torch.empty(1, device=x2.device, dtype=x2.dtype)
    b = _to_cuda_if_needed(bias)   if has_affine else torch.empty(1, device=x2.device, dtype=x2.dtype)
    grid = (N,)
    _layer_norm_forward_kernel[grid](
        x2, w, b, y,
        x2.stride(0), x2.stride(1),
        y.stride(0), y.stride(1),
        (w.stride(0) if has_affine else 0),
        (b.stride(0) if has_affine else 0),
        N, D, eps,
        has_affine,
        num_warps=4,
        num_stages=2,
    )
    y = y.view_as(x_cuda)
    return _to_cpu_if_needed(y, x.device)

def _modulate_triton(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if not _HAS_TRITON:
        return x * (1 + scale) + shift
    x_cuda = _to_cuda_if_needed(x)
    s_cuda = _to_cuda_if_needed(shift)
    b_cuda = _to_cuda_if_needed(scale)
    x2 = x_cuda.contiguous().view(-1, x_cuda.shape[-1])
    s2 = s_cuda.contiguous().view(-1, s_cuda.shape[-1])
    b2 = b_cuda.contiguous().view(-1, b_cuda.shape[-1])
    y = torch.empty_like(x2)
    N, D = x2.shape
    grid = (N,)
    _modulate_forward_kernel[grid](
        x2, s2, b2, y,
        x2.stride(0), x2.stride(1),
        s2.stride(0), s2.stride(1),
        b2.stride(0), b2.stride(1),
        y.stride(0), y.stride(1),
        N, D,
        num_warps=4,
        num_stages=2,
    )
    y = y.view_as(x_cuda)
    return _to_cpu_if_needed(y, x.device)

def _gate_triton(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    if not _HAS_TRITON:
        return g * x
    x_cuda = _to_cuda_if_needed(x)
    g_cuda = _to_cuda_if_needed(g)
    x2 = x_cuda.contiguous().view(-1, x_cuda.shape[-1])
    g2 = g_cuda.contiguous().view(-1, g_cuda.shape[-1])
    y = torch.empty_like(x2)
    N, D = x2.shape
    grid = (N,)
    _gate_forward_kernel[grid](
        x2, g2, y,
        x2.stride(0), x2.stride(1),
        g2.stride(0), g2.stride(1),
        y.stride(0), y.stride(1),
        N, D,
        num_warps=4,
        num_stages=2,
    )
    y = y.view_as(x_cuda)
    return _to_cpu_if_needed(y, x.device)

def _linear_gemv_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    # x: [N, D], weight: [M, D], bias: [M] or None -> y: [N, M]
    if not _HAS_TRITON:
        return x @ weight.t() + (bias if bias is not None else 0)
    x_cuda = _to_cuda_if_needed(x)
    w_cuda = _to_cuda_if_needed(weight)
    b_cuda = _to_cuda_if_needed(bias) if (bias is not None) else torch.empty(1, device=x_cuda.device, dtype=x_cuda.dtype)
    N, D = x_cuda.shape
    M = w_cuda.shape[0]
    y = torch.empty((N, M), device=x_cuda.device, dtype=x_cuda.dtype)
    has_bias = bias is not None
    grid = (N, triton.cdiv(M, 1))
    _linear_gemv_kernel[grid](
        x_cuda, w_cuda, b_cuda, y,
        x_cuda.stride(0), x_cuda.stride(1),
        w_cuda.stride(0), w_cuda.stride(1),
        y.stride(0), y.stride(1),
        N, D, M,
        has_bias,
        num_warps=4,
        num_stages=2,
    )
    return _to_cpu_if_needed(y, x.device)


# =============================
# Modules: keep PyTorch structure, use Triton where practical
# =============================
class OasisPatchEmbed(nn.Module):
    def __init__(self, img_height: int, img_width: int, patch_size: int, in_chans: int, embed_dim: int, flatten: bool = True):
        super().__init__()
        self.img_size = (img_height, img_width)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_height // patch_size, img_width // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6) if True else None

    def forward(self, x: torch.Tensor, random_sample: bool = False) -> torch.Tensor:
        _, _, height, width = x.shape
        if not random_sample and (height, width) != self.img_size:
            raise AssertionError(f"Input image size ({height}*{width}) doesn't match model {self.img_size}.")
        x = self.proj(x)  # [B*T, D, H', W']
        if self.norm is not None:
            x = _layer_norm_triton(x, self.norm.weight, self.norm.bias, eps=self.norm.eps)
        else:
            pass
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # [B*T, L, D]
        else:
            x = x.permute(0, 2, 3, 1)          # [B*T, H', W', D]
        return x

class OasisTimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.ModuleList(
            [
                nn.Linear(frequency_embedding_size, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
            ]
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half,
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = self.timestep_embedding(t, self.frequency_embedding_size)
        for layer in self.mlp:
            x = layer(x)
        return x

class SpatioTemporalDiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, is_causal: bool = True,
                 spatial_rotary_emb: OasisRotaryEmbedding = None, temporal_rotary_emb: OasisRotaryEmbedding = None):
        super().__init__()
        self.s_norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.s_attn = OasisSpatialAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            rotary_emb=spatial_rotary_emb,
        )
        self.s_norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.s_mlp = OasisMLP(
            hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            approximate_tanh=True,
        )
        self.s_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

        self.t_norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.t_attn = OasisTemporalAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            rotary_emb=temporal_rotary_emb,
            is_causal=is_causal,
        )
        self.t_norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.t_mlp = OasisMLP(
            hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            approximate_tanh=True,
        )
        self.t_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Spatial path
        s_shift_msa, s_scale_msa, s_gate_msa, s_shift_mlp, s_scale_mlp, s_gate_mlp = (
            self.s_adaLN_modulation(c).chunk(6, dim=-1)
        )
        x1 = _modulate_triton(self.s_norm1(x), s_shift_msa, s_scale_msa)
        x1 = self.s_attn(x1)  # SDPA
        x = x + _gate_triton(x1, s_gate_msa)

        # Spatial MLP
        x2 = _modulate_triton(self.s_norm2(x), s_shift_mlp, s_scale_mlp)
        x2 = self.s_mlp(x2)
        x = x + _gate_triton(x2, s_gate_mlp)

        # Temporal path
        t_shift_msa, t_scale_msa, t_gate_msa, t_shift_mlp, t_scale_mlp, t_gate_mlp = (
            self.t_adaLN_modulation(c).chunk(6, dim=-1)
        )
        x3 = _modulate_triton(self.t_norm1(x), t_shift_msa, t_scale_msa)
        x3 = self.t_attn(x3)  # SDPA
        x = x + _gate_triton(x3, t_gate_msa)

        # Temporal MLP
        x4 = _modulate_triton(self.t_norm2(x), t_shift_mlp, t_scale_mlp)
        x4 = self.t_mlp(x4)
        x = x + _gate_triton(x4, t_gate_mlp)
        return x

class OasisFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.ModuleList(
            [
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True),
            ]
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        modulation = c
        for layer in self.adaLN_modulation:
            modulation = layer(modulation)
        shift, scale = modulation.chunk(2, dim=-1)
        # Use Triton LN + modulate + GEMV
        x_norm = _layer_norm_triton(x, self.norm_final.weight, self.norm_final.bias, eps=self.norm_final.eps)
        x_mod = _modulate_triton(x_norm, shift, scale)  # [B*T, L, D]
        # Linear: y = x @ W^T + b
        D = x_mod.shape[-1]
        N = x_mod.numel() // D
        x_2d = x_mod.contiguous().view(N, D)
        W = self.linear.weight  # [M, D]
        B = self.linear.bias    # [M]
        y_2d = _linear_gemv_triton(x_2d, W, B)  # [N, M]
        y = y_2d.view(x_mod.shape[0], x_mod.shape[1], x_mod.shape[2])
        return y


# =============================
# Entry point: ModelNew
# =============================
class ModelNew(nn.Module):
    def __init__(
        self,
        input_h: int = 18,
        input_w: int = 32,
        patch_size: int = 2,
        in_channels: int = 16,
        hidden_size: int = 1024,
        depth: int = 16,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        external_cond_dim: int = 25,
        max_frames: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.max_frames = max_frames

        self.x_embedder = OasisPatchEmbed(input_h, input_w, patch_size, in_channels, hidden_size, flatten=False)
        self.t_embedder = OasisTimestepEmbedder(hidden_size)
        head_dim = hidden_size // num_heads
        self.spatial_rotary_emb = OasisRotaryEmbedding(dim=head_dim // 2, freqs_for="pixel", max_freq=256)
        self.temporal_rotary_emb = OasisRotaryEmbedding(dim=head_dim, freqs_for="lang")
        self.external_cond = nn.Linear(external_cond_dim, hidden_size, bias=True) if external_cond_dim > 0 else nn.Identity()
        self.blocks = nn.ModuleList(
            [
                SpatioTemporalDiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    is_causal=True,
                    spatial_rotary_emb=self.spatial_rotary_emb,
                    temporal_rotary_emb=self.temporal_rotary_emb,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = OasisFinalLayer(hidden_size, patch_size, self.out_channels)

    def initialize_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        weight = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(weight.view(weight.shape[0], -1))
        if self.x_embedder.proj.bias is not None:
            nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.s_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.s_adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.t_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.t_adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        if self.final_layer.linear.bias is not None:
            nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        c = self.out_channels
        p = self.patch_size
        h = x.shape[1]
        w = x.shape[2]
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward(self, x: torch.Tensor, t: torch.Tensor, external_cond: torch.Tensor | None = None) -> torch.Tensor:
        bsz, time, channels, height, width = x.shape
        x = x.reshape(bsz * time, channels, height, width)
        x = self.x_embedder(x)  # [B*T, D, H', W'] or [B*T, L, D]
        x = x.reshape(bsz, time, x.shape[1], x.shape[2], x.shape[3])
        t = t.reshape(bsz * time)
        c = self.t_embedder(t).reshape(bsz, time, -1)
        if torch.is_tensor(external_cond):
            c = c + self.external_cond(external_cond)
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        x = x.reshape(bsz * time, x.shape[2], x.shape[3], x.shape[4])
        x = self.unpatchify(x)
        return x.reshape(bsz, time, x.shape[1], x.shape[2], x.shape[3])


# =============================
# Keep helpers
# =============================
def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift

OasisDiT = ModelNew
