import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


# Autotuned kernel: y = logsigmoid(x) = min(x, 0) - log(1 + exp(-|x|))
# Compute in float32 for numerical stability, then cast back to input dtype.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=3),
    ],
    key=['N'],
)
@triton.jit
def _logsigmoid_kernel(X_ptr, Y_ptr, N,
                       BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + offs, mask=mask, other=0)
    x32 = x.to(tl.float32)

    ax = tl.abs(x32)
    t = tl.exp(-ax)  # in [0,1], stable
    out32 = tl.minimum(x32, 0.0) - tl.log(1.0 + t)

    out = out32.to(x.dtype)
    tl.store(Y_ptr + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallbacks
        if (not _HAS_TRITON) or (not x.is_cuda):
            return torch.nn.functional.logsigmoid(x)

        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return torch.nn.functional.logsigmoid(x)

        # Ensure contiguous
        x_contig = x.contiguous()
        y = torch.empty_like(x_contig)

        n_elements = x_contig.numel()
        if n_elements == 0:
            return y  # nothing to do

        # Grid: 1D over flattened tensor
        # BLOCK is selected by autotune; grid only needs problem size.
        grid = (triton.cdiv(n_elements, 1024),)  # initial grid; autotune will adjust BLOCK internally

        _logsigmoid_kernel[grid](
            x_contig, y, n_elements,
        )

        return y

LogSigmoid = ModelNew
