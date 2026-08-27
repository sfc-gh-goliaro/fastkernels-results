import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import Triton; provide fallback if not available.
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


# -----------------------------
# Triton kernel: SiLU (x * sigmoid(x))
# -----------------------------
if _HAS_TRITON:
    @triton.jit
    def silu_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + offs, mask=mask)
        # sigmoid(x) = 1 / (1 + exp(-x))
        sig = 1.0 / (1.0 + tl.exp(-x))
        y = x * sig
        tl.store(y_ptr + offs, y, mask=mask)

    def _choose_block_and_warps(N: int):
        # Simple heuristic: smaller blocks for small N to reduce overprovisioning.
        if N <= 2048:
            return 256, 2
        elif N <= 16384:
            return 1024, 4
        else:
            return 2048, 8

    def _silu_triton(x: torch.Tensor) -> torch.Tensor:
        # Fallback to PyTorch if no Triton or not CUDA
        if (not _HAS_TRITON) or (not x.is_cuda):
            return F.silu(x)
        # For very small tensors, PyTorch is faster due to lower overhead.
        N = x.numel()
        if N < 1024:
            return F.silu(x)

        y = torch.empty_like(x)
        BLOCK, num_warps = _choose_block_and_warps(N)
        grid = (triton.cdiv(N, BLOCK),)
        silu_kernel[grid](x, y, N, BLOCK=BLOCK, num_warps=num_warps)
        return y
else:
    def _silu_triton(x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)


# -----------------------------
# Keep timestep_embedding identical to original for bit-exactness
# -----------------------------
class ModelRef(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.ModuleList(
            [
                nn.Linear(frequency_embedding_size, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
            ]
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half,
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


# -----------------------------
# Original Model (kept as-is)
# -----------------------------
class Model(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.ModuleList(
            [
                nn.Linear(frequency_embedding_size, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
            ]
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half,
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


# -----------------------------
# Triton-optimized entry point: ModelNew
# - Uses Triton SiLU with heuristic fallback for small N
# - Keeps PyTorch timestep_embedding (bit-exact with reference)
# -----------------------------
class ModelNew(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.ModuleList(
            [
                nn.Linear(frequency_embedding_size, hidden_size, bias=True),
                SiLU(),          # will use Triton SiLU (with fallback)
                nn.Linear(hidden_size, hidden_size, bias=True),
            ]
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        # Match original PyTorch implementation exactly
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half,
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


# -----------------------------
# SiLU module using Triton (with small-N fallback)
# -----------------------------
class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _silu_triton(x)

OasisTimestepEmbedder = ModelNew
