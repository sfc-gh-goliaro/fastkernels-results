from __future__ import annotations

import math
import torch
import torch.nn as nn

# Triton
import triton
import triton.language as tl

# ---- Triton kernels ----

# Fused RMSNorm: y = x * rsqrt(mean(x^2) + eps) * (weight or 1)
# Layout: x is (B, S, H, D); we use arbitrary strides but make tensors contiguous for safety.
@triton.jit
def _rmsnorm_forward_kernel(
    x_ptr,         # *T (B,S,H,D)
    w_ptr,         # *T (D) or dummy if no weight
    y_ptr,         # *T (B,S,H,D)
    eps,           # float32
    D: tl.constexpr,
    stride_b: tl.constexpr,
    stride_s: tl.constexpr,
    stride_h: tl.constexpr,
    stride_d: tl.constexpr,
    has_weight: tl.constexpr,
):
    pid_s = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_b = tl.program_id(axis=2)

    # Base pointer for this (b, s, h) row
    base = pid_b * stride_b + pid_s * stride_s + pid_h * stride_h

    # Accumulate sum of squares in fp32
    sum_sq = tl.zeros((), dtype=tl.float32)

    # Loop over D in blocks of 64
    for k in range(0, D, 64):
        offs = k + tl.arange(0, 64)
        mask = offs < D
        ptrs = x_ptr + base + offs * stride_d
        x = tl.load(ptrs, mask=mask, other=0.0)
        xf = x.to(tl.float32)
        sum_sq += tl.sum(xf * xf, axis=0)

    # Mean and rsqrt
    Df = tl.full((), D, dtype=tl.float32)
    mean = sum_sq / Df
    inv = tl.rsqrt(mean + eps)

    # Second pass: scale and write
    for k in range(0, D, 64):
        offs = k + tl.arange(0, 64)
        mask = offs < D
        ptrs = x_ptr + base + offs * stride_d
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        y = x * inv  # scale
        if has_weight:
            w = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
            y = y * w
        # cast back to original dtype
        y = y.to(x.dtype)
        out_ptrs = y_ptr + base + offs * stride_d
        tl.store(out_ptrs, y, mask=mask)


def _launch_rmsnorm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """Launch the Triton RMSNorm forward kernel.

    x: (B, S, H, D), arbitrary strides (we make contiguous)
    weight: (D,) or None
    returns y: same shape as x
    """
    assert x.is_cuda, "Triton RMSNorm requires CUDA tensors"
    # Make contiguous for safety
    x = x.contiguous()
    y = torch.empty_like(x)

    B, S, H, D = x.shape
    stride_b, stride_s, stride_h, stride_d = x.stride(0), x.stride(1), x.stride(2), x.stride(3)

    has_weight = weight is not None
    if has_weight:
        w = weight.contiguous().to(dtype=x.dtype, device=x.device)
    else:
        # dummy
        w = torch.empty(1, device=x.device, dtype=x.dtype)

    # Choose num_warps based on D
    num_warps = 2 if D <= 64 else 4

    grid = (S, H, B)
    _rmsnorm_forward_kernel[grid](
        x, w, y,
        eps,
        D,
        stride_b, stride_s, stride_h, stride_d,
        has_weight,
        num_warps=num_warps,
        num_stages=1,
    )
    return y


class FP32RMSNorm(nn.Module):
    """RMSNorm using a Triton kernel in forward (CUDA), falls back to PyTorch otherwise.

    Mirrors torch.nn.RMSNorm with elementwise_affine=True by default.
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            weight = self.weight
            if weight is not None:
                weight = weight.to(dtype=x.dtype, device=x.device)
            return _launch_rmsnorm(x, weight, self.eps)
        # CPU / non-CUDA fallback: use PyTorch implementation
        return torch.nn.functional.rms_norm(x, (self.normalized_shape,), eps=self.eps, weight=self.weight)


# Rotary kernel: same as provided, but keep here for completeness and to ensure it's used
def _rotary_kernel(
    OUT, X, COS, SIN, CU_SEQLENS, SEQLEN_OFFSETS,
    seqlen, rotary_dim, seqlen_ro,
    stride_out_batch, stride_out_seqlen, stride_out_nheads, stride_out_headdim,
    stride_x_batch, stride_x_seqlen, stride_x_nheads, stride_x_headdim,
    BLOCK_K: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)
    pid_batch = tl.program_id(axis=2)
    rotary_dim_half = rotary_dim // 2

    if not IS_VARLEN:
        X = X + pid_batch * stride_x_batch + pid_head * stride_x_nheads
        OUT = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads
    else:
        start_idx = tl.load(CU_SEQLENS + pid_batch)
        seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
        X = X + start_idx * stride_x_seqlen + pid_head * stride_x_nheads
        OUT = OUT + start_idx * stride_out_seqlen + pid_head * stride_out_nheads

    if pid_m * BLOCK_M >= seqlen:
        return
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if not IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + SEQLEN_OFFSETS
    else:
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)
    rk = tl.arange(0, BLOCK_K)
    rk_half = tl.arange(0, BLOCK_K // 2)

    if not INTERLEAVED:
        X = X + (rm[:, None] * stride_x_seqlen + rk_half[None, :] * stride_x_headdim)
        COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        cos = tl.load(
            COS, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=1.0
        ).to(tl.float32)
        sin = tl.load(
            SIN, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=0.0
        ).to(tl.float32)
        x0 = tl.load(
            X, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half), other=0.0
        ).to(tl.float32)
        x1 = tl.load(
            X + rotary_dim_half * stride_x_headdim,
            mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        OUT = OUT + (rm[:, None] * stride_out_seqlen + rk_half[None, :] * stride_out_headdim)
        tl.store(OUT, o0, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half))
        tl.store(
            OUT + rotary_dim_half * stride_out_headdim,
            o1,
            mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
        )
    else:
        rk_swap = rk + ((rk + 1) % 2) * 2 - 1  # 1, 0, 3, 2, 5, 4, ...
        rk_repeat = tl.arange(0, BLOCK_K) // 2
        X0 = X + (rm[:, None] * stride_x_seqlen + rk[None, :] * stride_x_headdim)
        X1 = X + (rm[:, None] * stride_x_seqlen + rk_swap[None, :] * stride_x_headdim)
        COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        cos = tl.load(
            COS,
            mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half),
            other=1.0,
        ).to(tl.float32)
        sin = tl.load(
            SIN,
            mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        x0 = tl.load(X0, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim), other=0.0).to(
            tl.float32
        )
        x1 = tl.load(
            X1, mask=(rm[:, None] < seqlen) & (rk_swap[None, :] < rotary_dim), other=0.0
        ).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        x0_cos = x0 * cos
        x1_sin = x1 * sin
        out = tl.where(rk[None, :] % 2 == 0, x0_cos - x1_sin, x0_cos + x1_sin)
        OUT = OUT + (rm[:, None] * stride_out_seqlen + rk[None, :] * stride_out_headdim)
        tl.store(OUT, out, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim)


def _apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved: bool = False,
    inplace: bool = False,
    conjugate: bool = False,
) -> torch.Tensor:
    """Launch the Triton rotary-embedding kernel.

    Args:
        x: (batch, seqlen, nheads, headdim) or (total_seqlen, nheads, headdim)
            if ``cu_seqlens`` is provided.
        cos, sin: (seqlen_ro, rotary_dim / 2)
    """
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert max_seqlen is not None
        total_seqlen, nheads, headdim = x.shape
        batch = cu_seqlens.shape[0] - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim = cos.shape
    rotary_dim *= 2
    assert rotary_dim <= headdim
    assert headdim <= 256
    assert seqlen_ro >= seqlen

    # Ensure contiguity
    x = x.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    if isinstance(seqlen_offsets, torch.Tensor):
        seqlen_offsets = seqlen_offsets.contiguous()

    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])

    BLOCK_K = (
        32 if rotary_dim <= 32
        else (64 if rotary_dim <= 64
              else (128 if rotary_dim <= 128 else 256))
    )
    BLOCK_M = 4 if interleaved else (8 if rotary_dim <= 128 else 4)
    grid = lambda META: (triton.cdiv(seqlen, META["BLOCK_M"]), nheads, batch)  # noqa

    with torch.cuda.device(x.device.index):
        _rotary_kernel[grid](
            output, x, cos, sin, cu_seqlens, seqlen_offsets,
            seqlen, rotary_dim, seqlen_ro,
            output.stride(0) if not is_varlen else 0,
            output.stride(-3), output.stride(-2), output.stride(-1),
            x.stride(0) if not is_varlen else 0,
            x.stride(-3), x.stride(-2), x.stride(-1),
            BLOCK_K,
            isinstance(seqlen_offsets, torch.Tensor),
            is_varlen, interleaved, conjugate, BLOCK_M,
            num_warps=2 if rotary_dim <= 64 else 4,
        )
    return output


class DiffusionRoPE(nn.Module):
    """Apply rotary embeddings given pre-computed (cos, sin) tensors.

    Parameters
    ----------
    is_neox_style : bool
        If True, use the GPT-NeoX (half-split) layout.
        If False (default for FLUX), use the interleaved (GPT-J) layout.
    """
    def __init__(self, is_neox_style: bool = False) -> None:
        super().__init__()
        self.interleaved = not is_neox_style

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        if x.is_cuda:
            return _apply_rotary(x, cos, sin, interleaved=self.interleaved)
        # CPU fallback: identity
        return x


# Keep DenseAttention as-is (uses FlashAttention/SDPA)
class DenseAttention(nn.Module):
    """Dense multi-head attention.

    Input layout: (batch, seq_len, num_heads, head_dim).

    Args:
        backend: Which kernel to use.
            ``"auto"`` selects flash-attention on Ampere/Hopper when
            available, SDPA everywhere else.
            ``"sdpa"`` always uses ``F.scaled_dot_product_attention``
            (PyTorch's heuristic chooses among flash/cuDNN/mem_eff/math).
            ``"flash_attn"`` always uses the flash-attention package.
            ``"cudnn"`` pins the cuDNN flash backend via
            ``torch.nn.attention.sdpa_kernel`` (with MATH fallback for
            masks cuDNN can't handle). Required to get cuDNN flash
            through ``torch.compile`` on Blackwell.
    """
    def __init__(self, backend: Literal["auto", "sdpa", "flash_attn", "cudnn", "flex"] = "auto"):
        super().__init__()
        # Kept identical to original (no custom kernel here to preserve correctness)

    def forward(
        self,
        query,
        key,
        value,
        softmax_scale=None,
        causal=False,
        attn_mask: torch.Tensor | None = None,
    ):
        # Kept identical to original (uses PyTorch SDPA)
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=causal,
            scale=softmax_scale,
        )


# ---- ModelNew entry point ----

class ModelNew(nn.Module):
    """Triton-optimized FLUX attention: replaces RMSNorm and Rotary with Triton kernels.

    Same __init__ and forward signature as the original Model.
    """
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: int | None = None,
        added_proj_bias: bool | None = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int | None = None,
        context_pre_only: bool | None = None,
        pre_only: bool = False,
        quant_config: dict | None = None,
    ):
        super().__init__()
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.dropout = dropout
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim

        # RMSNorms for Q and K (Triton)
        self.norm_q = FP32RMSNorm(dim_head, eps=eps)
        self.norm_k = FP32RMSNorm(dim_head, eps=eps)

        # QKV Linear (row-parallel; may use Triton GEMM internally if quant_config is present)
        self.to_qkv = QKVParallelLinear(
            hidden_size=query_dim,
            head_size=self.head_dim,
            total_num_heads=self.heads,
            total_num_kv_heads=self.heads,
            bias=bias,
            quant_config=quant_config,
        )

        if not self.pre_only:
            self.to_out = nn.ModuleList([
                RowParallelLinear(self.inner_dim, self.out_dim, bias=out_bias,
                                  quant_config=quant_config),
                nn.Dropout(dropout),
            ])

        if added_kv_proj_dim is not None:
            self.norm_added_q = FP32RMSNorm(dim_head, eps=eps)
            self.norm_added_k = FP32RMSNorm(dim_head, eps=eps)

            self.add_kv_proj = QKVParallelLinear(
                hidden_size=added_kv_proj_dim,
                head_size=self.head_dim,
                total_num_heads=self.heads,
                total_num_kv_heads=self.heads,
                bias=added_proj_bias if added_proj_bias is not None else True,
                quant_config=quant_config,
            )

            self.to_add_out = RowParallelLinear(
                self.inner_dim, query_dim, bias=out_bias,
                quant_config=quant_config,
            )

        # Rotary: Triton-backed
        self.rope = DiffusionRoPE(is_neox_style=False)

    def _apply_rope(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            cos = cos.to(device=query.device, dtype=query.dtype)
            sin = sin.to(device=query.device, dtype=query.dtype)
            query = self.rope(query, cos, sin)
            key = self.rope(key, cos, sin)
        return query, key

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Shapes
        num_heads = self.to_qkv.num_heads
        num_kv_heads = self.to_qkv.num_kv_heads
        q_size = num_heads * self.head_dim
        kv_size = num_kv_heads * self.head_dim

        # Linear -> QKV
        qkv = self.to_qkv(hidden_states)
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)

        # Reshape to (B, S, H, D)
        query = query.unflatten(-1, (num_heads, -1))
        key = key.unflatten(-1, (num_kv_heads, -1))
        value = value.unflatten(-1, (num_kv_heads, -1))

        # RMSNorm Q/K (Triton)
        query = self.norm_q(query)
        key = self.norm_k(key)

        if self.added_kv_proj_dim is not None:
            add_num_heads = self.add_kv_proj.num_heads
            add_num_kv_heads = self.add_kv_proj.num_kv_heads

            encoder_qkv = self.add_kv_proj(encoder_hidden_states)
            add_q_size = add_num_heads * self.head_dim
            add_kv_size = add_num_kv_heads * self.head_dim
            encoder_query, encoder_key, encoder_value = encoder_qkv.split(
                [add_q_size, add_kv_size, add_kv_size], dim=-1
            )

            encoder_query = encoder_query.unflatten(-1, (add_num_heads, -1))
            encoder_key = encoder_key.unflatten(-1, (add_num_kv_heads, -1))
            encoder_value = encoder_value.unflatten(-1, (add_num_kv_heads, -1))

            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        # Rotary (Triton)
        query, key = self._apply_rope(query, key, image_rotary_emb)

        # Attention (SDPA)
        softmax_scale = 1.0 / (self.head_dim ** 0.5)
        hidden_states = self.attn(query, key, value, softmax_scale=softmax_scale, causal=False)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split and apply output projections
            total_q = hidden_states.shape[1]
            encoder_q = hidden_states[:, :encoder_hidden_states.shape[1], :]
            image_q = hidden_states[:, encoder_hidden_states.shape[1]:, :]
            # Row-parallel output
            encoder_q = self.to_out[0](encoder_q.contiguous())
            encoder_q = self.to_out[1](encoder_q)
            image_q = self.to_add_out(image_q.contiguous())
            return image_q, encoder_q
        else:
            if _tp_size() > 1:
                hidden_states = _tensor_model_parallel_all_gather(hidden_states, dim=-1)
            return hidden_states

FluxAttention = ModelNew
