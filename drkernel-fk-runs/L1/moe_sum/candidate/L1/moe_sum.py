import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


@triton.jit
def _moe_sum_kernel(
    inp_ptr,                  # *T, shape [M, topk, D]
    out_ptr,                  # *T, shape [M, D]
    M: tl.constexpr,
    topk: tl.constexpr,       # constexpr to enable loop unrolling
    D: tl.constexpr,
    sM: tl.constexpr,         # stride for M dim (elements)
    sT: tl.constexpr,         # stride for topk dim (elements)
    sD: tl.constexpr,         # stride for D dim (elements)
    out_sM: tl.constexpr,     # output stride for M (elements)
    out_sD: tl.constexpr,     # output stride for D (elements)
    BLOCK_D: tl.constexpr,
):
    # Program ids
    pid_m = tl.program_id(0)   # which row m
    pid_db = tl.program_id(1)  # which block of D

    # Column offsets for this program
    d_offsets = pid_db * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D

    # Base pointer for this m in input and output
    base_in = inp_ptr + pid_m * sM
    base_out = out_ptr + pid_m * out_sM

    # Accumulator in float32 for numerical stability
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    # Unrolled loop over top-k (constexpr)
    for k in range(0, topk):
        ptr = base_in + k * sT + d_offsets * sD
        x = tl.load(ptr, mask=mask_d, other=0.0)
        acc += x.to(tl.float32)

    # Store result
    out_ptrs = base_out + d_offsets * out_sD
    tl.store(out_ptrs, acc, mask=mask_d)


class ModelNew(nn.Module):
    """Triton-optimized version of the MoE sum: sums over top-k experts per token.

    forward(input: [M*topk, D], topk: int) -> output: [M, D]
    """

    def __init__(self):
        super().__init__()
        self._output = None

    def forward(self, input: torch.Tensor, topk: int) -> torch.Tensor:
        """
        Args:
            input: Tensor of shape [M*topk, D], on CUDA.
            topk: int, number of experts per token.

        Returns:
            output: Tensor of shape [M, D].
        """
        # Fallback if Triton/CUDA not available
        if not _HAS_TRITON or not input.is_cuda:
            total, D = input.shape
            M = total // topk
            return input.view(M, topk, D).sum(dim=1)

        assert input.dim() == 2, f"Expected 2D input, got shape {tuple(input.shape)}"
        total, D = input.shape
        assert total % topk == 0, f"total ({total}) must be divisible by topk ({topk})"
        M = total // topk

        # Reshape to [M, topk, D]; no need to make contiguous
        inp = input.view(M, topk, D)

        # Prepare or reuse output buffer matching input dtype
        if (
            self._output is None
            or self._output.shape != (M, D)
            or self._output.dtype != input.dtype
            or self._output.device != input.device
        ):
            self._output = torch.empty((M, D), device=input.device, dtype=input.dtype)
        out = self._output

        # Get strides in elements
        sM, sT, sD = inp.stride(0), inp.stride(1), inp.stride(2)
        out_sM, out_sD = out.stride(0), out.stride(1)

        # Heuristic for block size and launch params
        # For these traces (D up to 4096), 256 is a good default.
        BLOCK_D = 256 if D >= 256 else 128 if D >= 128 else 64
        num_warps = 4 if BLOCK_D <= 256 else 8
        num_stages = 2

        # Grid depends on BLOCK_D
        grid = lambda META: (M, triton.cdiv(D, META['BLOCK_D']))

        # Launch kernel
        _moe_sum_kernel[grid](
            inp, out,
            M, topk, D,
            sM, sT, sD,
            out_sM, out_sD,
            BLOCK_D=BLOCK_D,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        return out

MoeSum = ModelNew
