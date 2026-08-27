import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


# Autotuned fused QuickGELU kernel: y = x * sigmoid(k * x)
# Computes in float32 for numerical stability.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 256},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK': 512},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=3),
    ],
    key=['n_elements'],
)
@triton.jit
def quickgelu_kernel(x_ptr, y_ptr, n_elements, k,
                     BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    # Load as float32
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # z = k * x
    z = x * k

    # s = 1 / (1 + exp(-z))
    s = 1.0 / (1.0 + tl.exp(-z))

    # y = x * s
    y = x * s

    tl.store(y_ptr + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, k: float = 1.702):
        super().__init__()
        self.k = float(k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback to PyTorch if Triton not available or tensor not on CUDA
        if (not _HAS_TRITON) or (not x.is_cuda):
            return x * torch.sigmoid(self.k * x)

        # Ensure dtype and layout
        if x.dtype != torch.float32:
            # For non-fp32, compute in fp32 then cast back
            x_fp32 = x.to(torch.float32).contiguous()
            y_fp32 = self._triton_quickgelu(x_fp32)
            return y_fp32.to(x.dtype)
        else:
            return self._triton_quickgelu(x.contiguous())

    def _triton_quickgelu(self, x: torch.Tensor) -> torch.Tensor:
        n = x.numel()
        y = torch.empty_like(x)

        # Grid: 1D over blocks
        # BLOCK is selected by autotune; grid depends on the chosen BLOCK,
        # but Triton will compute it using the meta at launch time.
        # We still need to supply a grid based on an assumed maximum block;
        # the canonical pattern is to use cdiv with the BLOCK from the chosen config.
        # Triton will bind the correct BLOCK; so we can compute grid with any BLOCK,
        # but to be safe, we use the largest BLOCK to slightly over-provision,
        # or just compute grid = (cdiv(n, typical_BLOCK),).
        # The recommended way is to use the BLOCK from the chosen config;
        # Triton allows grid as a lambda, but for simplicity we use cdiv with 1024.
        # However, to honor the autotuned BLOCK, we can pass a lambda that uses meta['BLOCK'].
        def grid(meta):
            return (triton.cdiv(n, meta['BLOCK']),)

        quickgelu_kernel[grid](
            x, y,
            n,
            self.k,
        )
        return y

QuickGELU = ModelNew
