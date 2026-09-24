import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_D': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_D': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_D': 64},  num_warps=8, num_stages=2),

        triton.Config({'BLOCK_M': 16, 'BLOCK_D': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_D': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_D': 128}, num_warps=8, num_stages=2),

        triton.Config({'BLOCK_M': 16, 'BLOCK_D': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_D': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_D': 256}, num_warps=8, num_stages=3),
    ],
    key=['M', 'D', 'K'],
)
@triton.jit
def _moe_sum_kernel(
    x_ptr,                     # *T, shape [M, K, D]
    y_ptr,                     # *T, shape [M, D]
    M: tl.constexpr,           # int (rows = total // K)
    K: tl.constexpr,           # int (top-k)
    D: tl.constexpr,           # int (features)
    stride_x_m: tl.constexpr,  # int
    stride_x_k: tl.constexpr,  # int
    stride_x_d: tl.constexpr,  # int
    stride_y_m: tl.constexpr,  # int
    stride_y_d: tl.constexpr,  # int
    BLOCK_M: tl.constexpr,     # tile in M (autotuned)
    BLOCK_D: tl.constexpr,     # tile in D (autotuned)
):
    # Program IDs for 2D launch grid
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)

    # Compute offsets for this program
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    # Bounds masks
    mask_m = m_offsets < M
    mask_d = d_offsets < D

    # 2D index grid [BM, BD]
    mm = m_offsets[:, None]
    dd = d_offsets[None, :]

    # Accumulator in float32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # Loop over K and accumulate in float32
    for k in range(0, K):
        x_ptrs = x_ptr + mm * stride_x_m + k * stride_x_k + dd * stride_x_d
        x_val = tl.load(x_ptrs, mask=(mask_m[:, None] & mask_d[None, :]), other=0)
        x_val_f32 = x_val.to(tl.float32)
        acc += x_val_f32

    # Store result to y[m, d]; Triton will cast acc to y's element type
    y_ptrs = y_ptr + mm * stride_y_m + dd * stride_y_d
    tl.store(y_ptrs, acc, mask=(mask_m[:, None] & mask_d[None, :]))


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._output = None  # cached output buffer

    def forward(self, input: torch.Tensor, topk: int) -> torch.Tensor:
        """
        Sum over the top-k dimension for MoE outputs.

        Args:
            input: [M * topk, D] tensor
            topk: number of experts per token

        Returns:
            output: [M, D] tensor
        """
        if not isinstance(input, torch.Tensor):
            raise TypeError("input must be a torch.Tensor")
        if input.dim() != 2:
            raise ValueError(f"Expected a 2D tensor, got shape {tuple(input.shape)}")
        total, D = input.shape
        if total % topk != 0:
            raise ValueError(
                f"Input size0 ({total}) must be divisible by topk ({topk}); got remainder."
            )
        M = total // topk

        # CPU fallback
        if not input.is_cuda:
            return input.view(M, topk, D).sum(dim=1)

        # Ensure contiguous and get dtype
        x = input.contiguous()
        dtype = x.dtype

        # View as [M, K, D]
        x3 = x.view(M, topk, D)

        # Allocate or reuse output buffer
        if (
            self._output is None
            or self._output.shape != (M, D)
            or self._output.dtype != dtype
            or self._output.device != x.device
        ):
            self._output = torch.empty((M, D), device=x.device, dtype=dtype)
        y = self._output

        # Strides in elements
        stride_x_m, stride_x_k, stride_x_d = x3.stride()
        stride_y_m, stride_y_d = y.stride()

        # Launch grid: 2D over (M, D). BLOCK sizes are chosen by autotuner.
        # We still need to provide a grid; triton.cdiv uses the chosen config at runtime.
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(D, meta['BLOCK_D']))

        _moe_sum_kernel[grid](
            x3, y,
            M, topk, D,
            stride_x_m, stride_x_k, stride_x_d,
            stride_y_m, stride_y_d,
        )

        return y

MoeSum = ModelNew
