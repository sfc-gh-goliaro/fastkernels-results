import math
import os
import torch
import torch.nn as nn

# Triton imports
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False

# ---------------------------
# Triton kernel: QuickGELU (with optional upcast)
# ---------------------------
if _HAS_TRITON:
    @triton.jit
    def _quick_gelu_kernel(x_ptr, y_ptr, n_elements: tl.constexpr,
                           BLOCK_SIZE: tl.constexpr,
                           UPCAST: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements

        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        if UPCAST:
            x32 = x.to(tl.float32)
            s = 1.702 * x32
            t = 1.0 / (1.0 + tl.exp(-s))
            y32 = x32 * t
            y = y32.to(x.dtype)
        else:
            # Compute in native dtype (float32) to avoid unnecessary casts
            s = 1.702 * x
            t = 1.0 / (1.0 + tl.exp(-s))
            y = x * t

        tl.store(y_ptr + offs, y, mask=mask)

    def _launch_quick_gelu(x: torch.Tensor) -> torch.Tensor:
        # Work on a contiguous view
        x_contig = x.contiguous()
        y = torch.empty_like(x_contig)

        n = x_contig.numel()
        # Tuned for large 1D elementwise workloads
        BLOCK = 4096
        grid = (triton.cdiv(n, BLOCK),)
        upcast = x_contig.dtype != torch.float32
        _quick_gelu_kernel[grid](
            x_contig, y, n, BLOCK_SIZE=BLOCK, UPCAST=upcast, num_warps=8,
        )
        return y.view_as(x)

class QuickGELUTrition(nn.Module):
    """QuickGELU activation, with Triton kernel on CUDA.

    Falls back to torch ops on CPU.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _HAS_TRITON and x.is_cuda:
            return _launch_quick_gelu(x)
        # CPU or no Triton: use torch ops
        return x * torch.sigmoid(1.702 * x)

# ---------------------------
# Clean ModelNew without TP dependencies
# ---------------------------

class ModelNew(nn.Module):
    """Vision encoder MLP with configurable activation.

    Qwen2-VL uses QuickGELU (default); Qwen3-VL uses F.silu.
    Here, QuickGELU is implemented with a Triton kernel on CUDA.
    """
    def __init__(self, in_features: int, hidden_features: int,
                 act_fn: nn.Module = QuickGELUTrition(),
                 bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act_fn(self.fc1(x)))

VisionMLP = ModelNew
