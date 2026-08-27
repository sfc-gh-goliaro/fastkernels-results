import math
import torch
import torch.nn as nn

# Try to import Triton; if not available, we'll fall back to PyTorch ops.
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


# -----------------------------
# Optimized Triton kernel: silu(gate) * up (1D, contiguous)
# -----------------------------
if _HAS_TRITON:
    @triton.jit
    def silu_and_mul_kernel_1d(
        X,  # [M, 2d], contiguous
        Y,  # [M, d], contiguous
        d: tl.int32,          # feature dimension
        M: tl.int32,          # batch dimension (rows)
        BLOCK: tl.constexpr,  # tile size over columns
    ):
        # 1D grid over all columns of Y: size = M * d
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        total = M * d
        mask = offs < total

        # Map linear index -> (row, col)
        rows = offs // d
        cols = offs % d

        # Base pointers (row start for X and Y)
        base_x = rows * (2 * d)
        base_y = rows * d

        # Compute element pointers
        ptr_x0 = X + base_x + cols                       # gate: first half
        ptr_x1 = X + base_x + cols + d                   # up: second half
        ptr_y  = Y + base_y + cols

        # Load
        x0 = tl.load(ptr_x0, mask=mask, other=0.0)
        x1 = tl.load(ptr_x1, mask=mask, other=0.0)

        # Compute in fp32
        x0f = x0.to(tl.float32)
        x1f = x1.to(tl.float32)

        # silu(x) = x * sigmoid(x); sigmoid(x) = 1 / (1 + exp(-x))
        sig = 1.0 / (1.0 + tl.exp(-x0f))
        silu = x0f * sig
        outf = silu * x1f

        out = outf.to(x0.dtype)
        tl.store(ptr_y, out, mask=mask)


class SiluAndMulTriton(nn.Module):
    def __init__(self, block_size: int = 4096, num_warps: int = 8):
        super().__init__()
        self.block_size = block_size
        self.num_warps = num_warps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [M, 2d]
        assert x.dim() == 2, f"Expected 2D tensor, got shape {tuple(x.shape)}"
        M, two_d = x.shape
        d = two_d // 2

        # Ensure contiguous for 1D kernel
        if not x.is_contiguous():
            x = x.contiguous()

        y = torch.empty((M, d), device=x.device, dtype=x.dtype)

        # Launch 1D grid over M*d elements
        grid = (triton.cdiv(M * d, self.block_size),)

        silu_and_mul_kernel_1d[grid](
            x, y,
            d, M,
            BLOCK=self.block_size,
            num_warps=self.num_warps,
        )
        return y


# ---------------------------------------
# Local definitions (self-contained)
# ---------------------------------------

class _LinearNoTP(nn.Module):
    """
    A simple linear layer without tensor-parallel sharding:
      y = x @ W.T + b
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None
        self.reset_parameters = self._reset_parameters

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.weight.shape[1]) if self.weight.shape[1] > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)


class ModelNew(nn.Module):
    """
    SwiGLU MLP block:
      gate_up: Linear(h -> 2d)
      act:     silu(gate) * up  -> [M, d] (Triton-optimized)
      down:    Linear(d -> h)
    Entry point must be named ModelNew.
    """
    def __init__(self, config, quant_config: dict | None = None,
                 hidden_size: int | None = None,
                 intermediate_size: int | None = None,
                 reduce_results: bool = True):
        super().__init__()
        h = hidden_size if hidden_size is not None else config.hidden_size
        i = intermediate_size if intermediate_size is not None else config.intermediate_size

        # Gate+Up projection (no tensor-parallel)
        self.gate_up_proj = _LinearNoTP(h, 2 * i, bias=True)
        self.gate_up_proj.reset_parameters()

        # Triton activation
        self.act = SiluAndMulTriton(block_size=4096, num_warps=8)

        # Down projection (no tensor-parallel)
        self.down_proj = _LinearNoTP(i, h, bias=True)
        self.down_proj.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [M, h] -> [M, 2d]
        x = self.gate_up_proj(x)
        # activation: [M, 2d] -> [M, d]
        if x.is_cuda and _HAS_TRITON:
            x = self.act(x)
        else:
            d = x.shape[-1] // 2
            x = torch.nn.functional.silu(x[..., :d]) * x[..., d:]
        # [M, d] -> [M, h]
        return self.down_proj(x)

LlamaMLP = ModelNew
