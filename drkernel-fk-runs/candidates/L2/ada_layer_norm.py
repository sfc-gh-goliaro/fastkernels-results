import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _ln_fwd_kernel(
    X_ptr,         # *const T, shape [M, D] flattened
    Y_ptr,         # *T, shape [M, D] flattened
    D: tl.constexpr,
    stride_xm,     # int: stride between rows in X (elements)
    stride_xd,     # int: stride between features in X (elements, usually 1)
    stride_ym,     # int: stride between rows in Y (elements)
    stride_yd,     # int: stride between features in Y (elements, usually 1)
    eps,           # float
    BLOCK_D: tl.constexpr,
):
    # One program per row
    row = tl.program_id(0)
    x_row_ptr = X_ptr + row * stride_xm
    y_row_ptr = Y_ptr + row * stride_ym

    # Accumulators in fp32
    sum_x = tl.zeros((), dtype=tl.float32)
    sum_x2 = tl.zeros((), dtype=tl.float32)

    # Pass 1: reduction for mean and variance
    for start in range(0, D, BLOCK_D):
        offs = start + tl.arange(0, BLOCK_D)
        mask = offs < D
        x = tl.load(x_row_ptr + offs * stride_xd, mask=mask, other=0.0)
        xf = x.to(tl.float32)
        sum_x += tl.sum(xf, axis=0)
        sum_x2 += tl.sum(xf * xf, axis=0)

    mean = sum_x / D
    var = sum_x2 / D - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Pass 2: normalize and store
    for start in range(0, D, BLOCK_D):
        offs = start + tl.arange(0, BLOCK_D)
        mask = offs < D
        x = tl.load(x_row_ptr + offs * stride_xd, mask=mask, other=0.0)
        xf = x.to(tl.float32)
        y = (xf - mean) * inv_std
        y = y.to(x.dtype)
        tl.store(y_row_ptr + offs * stride_yd, y, mask=mask)


def _triton_layer_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    LayerNorm over the last dimension using Triton.
    View x as [M, D] where M = prod(shape[:-1]). Returns tensor of same shape/dtype.
    """
    assert x.is_cuda, "Triton kernel requires CUDA tensor"
    if not x.is_contiguous():
        x = x.contiguous()
    orig_shape = x.shape
    D = orig_shape[-1]
    M = int(x.numel() // D)
    x2d = x.view(M, D)
    y2d = torch.empty_like(x2d)

    # Strides in elements
    stride_xm = x2d.stride(0)
    stride_xd = x2d.stride(1)
    stride_ym = y2d.stride(0)
    stride_yd = y2d.stride(1)

    # BLOCK_D heuristic: power-of-two up to 2048
    block = 1
    while block < D and block < 2048:
        block <<= 1
    BLOCK_D = max(64, min(block, 2048))

    grid = (M,)
    _ln_fwd_kernel[grid](
        x2d, y2d,
        D,
        stride_xm, stride_xd,
        stride_ym, stride_yd,
        eps,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )
    return y2d.view(orig_shape)


class Model(nn.Module):
    r"""
    Triton-optimized dual-stream adaptive layer norm zero.

    Forward returns:
      (y, gate_msa, shift_mlp, scale_mlp, gate_mlp)

    Parameters:
        embedding_dim (`int`): D
        num_embeddings (`int` or None): ignored (API compatibility)
        norm_type (`str`): must be "layer_norm"
        bias (`bool`): whether Linear has bias
        promote_fp32 (`bool`): unused in kernel (we accumulate in fp32)
    """

    def __init__(self, embedding_dim: int, num_embeddings: int | None = None,
                 norm_type="layer_norm", bias=True, promote_fp32: bool = True):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=bias)
        if norm_type != "layer_norm":
            raise ValueError(f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm'.")
        self.eps = 1e-6

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor | None = None,
        class_labels: torch.LongTensor | None = None,
        hidden_dtype: torch.dtype | None = None,
        emb: torch.Tensor | None = None,
    ):
        if emb is None:
            raise ValueError("emb must be provided")
        if not x.is_cuda:
            raise RuntimeError("Model requires CUDA tensor for Triton kernel")

        # Coefficients
        emb2 = self.linear(self.silu(emb))  # [B, 6D]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb2.chunk(6, dim=1)

        # LayerNorm
        y = _triton_layer_norm(x, eps=self.eps)  # [B, N, D]

        B = emb2.shape[0]
        # Apply post-LN affine: y = y * (1 + scale_msa) + shift_msa
        scale = 1.0 + scale_msa.view(B, -1)[:, None, :]   # [B,1,D]
        y = y * scale + shift_msa.view(B, -1)[:, None, :] # [B,N,D]

        return y, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroSingle(nn.Module):
    r"""
    Triton-optimized single-stream adaptive layer norm zero.

    Forward returns:
      (y, gate_msa)

    Parameters:
        embedding_dim (`int`): D
        norm_type (`str`): must be "layer_norm"
        bias (`bool`): whether Linear has bias
        promote_fp32 (`bool`): unused
    """

    def __init__(self, embedding_dim: int, norm_type="layer_norm", bias=True, promote_fp32: bool = True):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        if norm_type != "layer_norm":
            raise ValueError(f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm'.")
        self.eps = 1e-6

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor | None = None,
    ):
        if emb is None:
            raise ValueError("emb must be provided")
        if not x.is_cuda:
            raise RuntimeError("AdaLayerNormZeroSingle requires CUDA tensor for Triton kernel")

        emb2 = self.linear(self.silu(emb))          # [B, 3D]
        shift_msa, scale_msa, gate_msa = emb2.chunk(3, dim=1)

        # LayerNorm
        y = _triton_layer_norm(x, eps=self.eps)      # [B, N, D]

        # Apply post-LN affine: y = y * (1 + scale_msa) + shift_msa
        B = emb2.shape[0]
        scale = 1.0 + scale_msa.view(B, -1)[:, None, :]          # [B,1,D]
        y = y * scale + shift_msa.view(B, -1)[:, None, :]        # [B,N,D]
        return y, gate_msa

AdaLayerNormZero = ModelNew
AdaLayerNormZeroSingle = ModelNew
