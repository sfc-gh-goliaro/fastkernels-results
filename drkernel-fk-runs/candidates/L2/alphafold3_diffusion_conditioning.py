import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


def _next_power_of_2(x: int) -> int:
    return 1 if x == 0 else 2 ** ((x - 1).bit_length())


@triton.jit
def _layernorm_forward_kernel(
    x_ptr,                  # *dtype
    y_ptr,                  # *dtype
    w_ptr,                  # *fp32 or dummy
    b_ptr,                  # *fp32 or dummy
    stride_row: tl.constexpr,
    C: tl.constexpr,        # int (last-dim size)
    eps,                    # fp32
    has_affine: tl.constexpr,  # 0/1
    BLOCK_SIZE: tl.constexpr,
):
    # One program per row
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < C

    x_row_ptr = x_ptr + row_id * stride_row + offs
    # Load and upcast to fp32 for math
    x = tl.load(x_row_ptr, mask=mask, other=0).to(tl.float32)

    # Two-moment variance: sum(x) and sum(x^2)
    sum_x = tl.sum(x, axis=0)
    sum_x2 = tl.sum(x * x, axis=0)

    C_f = tl.full((), C, dtype=tl.float32)
    mean = sum_x / C_f
    var = sum_x2 / C_f - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize
    y = (x - mean) * rstd  # fp32

    if has_affine:
        w = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = y * w + b

    # Store to y (cast to y dtype automatically)
    y_row_ptr = y_ptr + row_id * stride_row + offs
    tl.store(y_row_ptr, y, mask=mask)


def triton_layer_norm(
    x: torch.Tensor,
    normalized_shape: int,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Triton-backed layer_norm for last dimension only.
    - Computes in fp32 internally; returns in x.dtype.
    - Supports CUDA tensors; falls back to torch if not CUDA or Triton unavailable.
    """
    if (not TRITON_AVAILABLE) or (not x.is_cuda):
        # Fallback to torch
        return F.layer_norm(x, (normalized_shape,), weight, bias, eps)

    # Require at least 2D; if 1D, make it 2D by adding a leading dim
    if x.dim() == 1:
        x = x.view(1, -1)
    assert x.dim() >= 2, f"Expected at least 2D tensor, got shape {tuple(x.shape)}"

    # Normalize over the last dimension only
    C = int(normalized_shape)
    assert x.shape[-1] == C, f"Last dim {x.shape[-1]} != normalized_shape {C}"

    # Make contiguous in the last dimension for simple strides
    x_c = x.contiguous()
    # Prepare output
    y = torch.empty_like(x_c)

    # Flatten to 2D [M, C]: M = product of all but last
    M = x_c.numel() // C
    x_2d = x_c.view(M, C)
    y_2d = y.view(M, C)

    # Stride in elements between rows
    stride_row = x_2d.stride(0)

    # BLOCK_SIZE: next power-of-two of C, cap to 2048, ensure >= C
    BLOCK = _next_power_of_2(C)
    BLOCK = min(BLOCK, 2048)
    BLOCK = max(BLOCK, C)

    has_affine = int(weight is not None and bias is not None)

    # Weight and bias: use fp32 for math; if None, dummy
    w = None
    b = None
    if has_affine:
        # Ensure 1D over last dim
        w = weight.contiguous().view(C).to(dtype=torch.float32, device=x.device)
        b = bias.contiguous().view(C).to(dtype=torch.float32, device=x.device)

    # Choose num_warps heuristically
    if BLOCK <= 128:
        num_warps = 4
    elif BLOCK <= 512:
        num_warps = 8
    else:
        num_warps = 16

    grid = (M,)

    _layernorm_forward_kernel[grid](
        x_2d, y_2d,
        w if has_affine else x_2d,  # dummy ptr if not used
        b if has_affine else x_2d,  # dummy ptr if not used
        stride_row,
        C,
        eps,
        has_affine,
        BLOCK,
        num_warps=num_warps,
        num_stages=2,
    )

    return y


class LayerNorm(nn.Module):
    """
    Drop-in replacement for the original LayerNorm that uses Triton when available.

    Keeps the same API:
    - __init__(normalized_shape, eps=1e-5, elementwise_affine=True, create_scale=True, create_offset=True, promote_fp32=True)
    - forward(x): returns layer-normed tensor
    Notes:
    - We implement elementwise_affine and promote_fp32 behavior.
    - We compute reduction in fp32 internally for numerical stability.
    - If not CUDA or Triton not available, we fall back to F.layer_norm.
    """
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
            # Keep parameters in fp32 for stability; inputs may be bf16/fp16
            self.weight = nn.Parameter(torch.ones(normalized_shape, dtype=torch.float32))
        else:
            self.register_parameter("weight", None)

        if elementwise_affine and create_offset:
            self.bias = nn.Parameter(torch.zeros(normalized_shape, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Match original behavior: if promote_fp32, compute reduction in fp32
        # Our kernel already does fp32 math internally; output dtype == input dtype.
        # If not CUDA or Triton unavailable, fallback
        if not x.is_cuda or not TRITON_AVAILABLE:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        # Ensure weight/bias are on device
        w = self.weight
        b = self.bias
        if w is not None and w.device != x.device:
            w = w.to(device=x.device)
        if b is not None and b.device != x.device:
            b = b.to(device=x.device)

        return triton_layer_norm(x, self.normalized_shape[0], w, b, self.eps)


# The rest of the modules remain as before, using the new LayerNorm.
class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)

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

class SwiGLU(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.silu = SiLU()
        self.linear_a = Linear(c_in, c_out, bias=False)
        self.linear_b = Linear(c_in, c_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.silu(self.linear_a(x)) * self.linear_b(x)


class SwiGLUTransition(nn.Module):
    def __init__(self, c_in: int, n: int):
        super().__init__()
        self.c_in = c_in
        self.n = n

        self.layer_norm = LayerNorm(c_in)  # Triton-backed
        self.swiglu = SwiGLU(c_in, n * c_in)
        self.linear_out = Linear(n * c_in, c_in, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, chunk_size: int | None = None) -> torch.Tensor:
        if mask is None:
            mask = x.new_ones(x.shape[:-1])
        mask = mask.unsqueeze(-1)
        x = self.layer_norm(x)
        x = self.swiglu(x)
        x = self.linear_out(x)
        x = x * mask
        return x


def _binned_one_hot(x: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
    return (x[..., None] > boundaries).to(dtype=x.dtype)


def relpos_complex(batch: dict, max_relative_idx: int, max_relative_chain: int) -> torch.Tensor:
    res_idx = batch["residue_index"]
    asym_id = batch["asym_id"]
    entity_id = batch["entity_id"]
    same_chain = asym_id[..., None] == asym_id[..., None, :]
    same_res = res_idx[..., None] == res_idx[..., None, :]
    same_entity = entity_id[..., None] == entity_id[..., None, :]

    def _relpos(pos: torch.Tensor, condition: torch.BoolTensor, rel_clip_idx: int) -> torch.Tensor:
        offset = pos[..., None] - pos[..., None, :]
        clipped_offset = torch.clamp(offset + rel_clip_idx, min=0, max=2 * rel_clip_idx)
        final_offset = torch.where(
            condition,
            clipped_offset,
            (2 * rel_clip_idx + 1) * torch.ones_like(clipped_offset),
        )
        boundaries = torch.arange(
            start=0, end=2 * rel_clip_idx + 2, device=final_offset.device
        ).to(dtype=final_offset.dtype)
        return _binned_one_hot(final_offset, boundaries)

    rel_pos = _relpos(pos=res_idx, condition=same_chain, rel_clip_idx=max_relative_idx)
    rel_token = _relpos(
        pos=batch["token_index"],
        condition=same_chain & same_res,
        rel_clip_idx=max_relative_idx,
    )
    rel_chain = _relpos(
        pos=batch["sym_id"],
        condition=same_entity,
        rel_clip_idx=max_relative_chain,
    )
    same_entity_feat = same_entity[..., None].to(dtype=rel_pos.dtype)
    return torch.cat([rel_pos, rel_token, same_entity_feat, rel_chain], dim=-1)


class ModelNew(nn.Module):
    """
    Triton-optimized version of the original Model:
    - Replaces LayerNorm with Triton-backed implementation.
    - Keeps the rest unchanged.
    Forward signature preserved.
    """
    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        c_s_input: int = 449,
        sigma_data: float = 16.0,
        relpos_k: int = 32,
        max_relative_chain: int = 2,
        c_fourier_emb: int = 256,
        seed_fourier_emb: int = 42,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_s_input = c_s_input
        self.c_fourier_emb = c_fourier_emb
        self.sigma_data = sigma_data
        self.relpos_k = relpos_k
        self.max_relative_chain = max_relative_chain

        num_rel_pos_bins = 2 * relpos_k + 2
        num_rel_token_bins = 2 * relpos_k + 2
        num_rel_chain_bins = 2 * max_relative_chain + 2
        num_same_entity_features = 1
        num_relpos_dims = (
            num_rel_pos_bins + num_rel_token_bins
            + num_rel_chain_bins + num_same_entity_features
        )

        # Use Triton LayerNorm where possible
        self.layer_norm_z = LayerNorm(num_relpos_dims + c_z, create_offset=False)
        self.linear_z = Linear(num_relpos_dims + c_z, c_z, bias=False)

        self.transition_z = nn.ModuleList([
            SwiGLUTransition(c_in=c_z, n=2)
            for _ in range(2)
        ])

        self.layer_norm_s = LayerNorm(c_s + c_s_input, create_offset=False)
        self.linear_s = Linear(c_s + c_s_input, c_s, bias=False)

        # FourierEmbedding kept in PyTorch (tiny)
        class FourierEmbedding(nn.Module):
            def __init__(self, c: int = 256, seed: int = 42):
                super().__init__()
                self.c = c
                generator = torch.Generator()
                generator.manual_seed(seed)
                self.register_buffer("w", torch.randn(c, generator=generator))
                self.register_buffer("b", torch.randn(c, generator=generator))

            def forward(self, t: torch.Tensor) -> torch.Tensor:
                x = t * self.w + self.b
                return torch.cos(2 * math.pi * x)

        self.fourier_emb = FourierEmbedding(c=c_fourier_emb, seed=seed_fourier_emb)
        self.layer_norm_n = LayerNorm(c_fourier_emb, create_offset=False)
        self.linear_n = Linear(c_fourier_emb, c_s, bias=False)

        self.transition_s = nn.ModuleList([
            SwiGLUTransition(c_in=c_s, n=2)
            for _ in range(2)
        ])

    def forward(
        self,
        batch: dict,
        t: torch.Tensor,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
        use_conditioning: bool,
        chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Same signature as original Model:
        Returns:
            si:  [*, N_token, c_s]
            zij: [*, N_token, N_token, c_z]
        """
        if use_conditioning:
            # Pair conditioning: concat trunk pair with relpos features
            if "asym_id" in batch:
                relpos_zij = relpos_complex(
                    batch=batch,
                    max_relative_idx=self.relpos_k,
                    max_relative_chain=self.max_relative_chain,
                ).to(dtype=zij_trunk.dtype)
            else:
                relpos_dim = self.linear_z.weight.shape[-1] - self.c_z
                relpos_zij = zij_trunk.new_zeros(
                    zij_trunk.shape[:-1] + (relpos_dim,)
                )

            zij = torch.cat([zij_trunk, relpos_zij], dim=-1)
            zij = self.linear_z(self.layer_norm_z(zij))

            # Single conditioning: concat trunk single with input
            si = torch.cat([si_trunk, si_input], dim=-1)
            si = self.linear_s(self.layer_norm_s(si))
        else:
            zij = zij_trunk.new_zeros(zij_trunk.shape)
            si = si_trunk.new_zeros(si_trunk.shape[:-1] + (self.c_s,))

        # Fourier noise embedding
        n = 0.25 * torch.log(t / self.sigma_data)
        n_emb = self.fourier_emb(n.unsqueeze(-1) if n.dim() == 0 else n)
        si = si + self.linear_n(self.layer_norm_n(n_emb)).unsqueeze(-2)

        # Apply transition layers
        token_mask = batch.get("token_mask")
        if token_mask is not None:
            pair_mask = token_mask[..., :, None] * token_mask[..., None, :]
        else:
            pair_mask = None

        for layer in self.transition_z:
            zij = zij + layer(zij, mask=pair_mask)

        for layer in self.transition_s:
            si = si + layer(si, mask=token_mask)

        return si, zij

DiffusionConditioning = ModelNew
