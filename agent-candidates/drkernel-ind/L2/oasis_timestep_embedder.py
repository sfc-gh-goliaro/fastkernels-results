import math
import torch
import torch.nn as nn

# Try to import Triton; provide graceful fallback if unavailable.
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


# -----------------------------
# Triton kernel: 2D embedding from precomputed freq
# -----------------------------
if _HAS_TRITON:
    @triton.jit
    def _embed_sincos_2d_kernel(
        t_ptr,          # *int64 [B]
        freq_ptr,       # *float32 [half]
        out_ptr,        # *float32 [B, D]
        B: tl.constexpr,
        D: tl.constexpr,
        HALF: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        # 2D launch: pid_b over B, pid_d over D tiles
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)

        b = pid_b
        d_start = pid_d * BLOCK_D
        d = d_start + tl.arange(0, BLOCK_D)
        mask_d = d < D

        # load t[b]
        t_val = tl.load(t_ptr + b)
        t_f = tl.cast(t_val, tl.float32)

        # which entries are in [0, HALF) -> cos ; >=HALF -> sin
        use_cos = d < HALF
        use_sin = d >= HALF

        # indices into freq
        idx_cos = d
        idx_sin = d - HALF

        # load frequencies with masks
        freq_cos = tl.load(freq_ptr + idx_cos, mask=use_cos & mask_d, other=0.0)
        freq_sin = tl.load(freq_ptr + idx_sin, mask=use_sin & mask_d, other=0.0)

        # angles
        angle_cos = t_f * freq_cos
        angle_sin = t_f * freq_sin

        c = tl.cos(angle_cos)
        s = tl.sin(angle_sin)

        # select cos for <HALF, sin for >=HALF
        out_val = tl.where(use_cos, c, s)

        # store to out[b, d]
        row_base = b * D
        out_offsets = row_base + d
        tl.store(out_ptr + out_offsets, out_val, mask=mask_d)


def _timestep_embedding_triton_from_freq(t: torch.Tensor,
                                        dim: int,
                                        max_period: int = 10000) -> torch.Tensor:
    """
    Compute sine-cosine timestep embedding using Triton, given precomputed freq.
    x[b, d] = cos(t * freq[d]) if d < D/2
              sin(t * freq[d - D/2]) otherwise
    Shape: [B, dim], dtype=float32
    """
    assert _HAS_TRITON, "Triton is not available"
    assert t.is_cuda, "t must be on CUDA for Triton kernel"
    device = t.device
    B = t.shape[0]
    D = int(dim)
    assert D > 0, "dim must be positive"

    # Compute freq on device, exactly as reference:
    half = D // 2
    i = torch.arange(start=0, end=half, device=device, dtype=torch.float32)
    log_max = math.log(max_period)
    freq = torch.exp(-log_max * i / half)  # [half], float32

    # Output
    x = torch.empty((B, D), device=device, dtype=torch.float32)

    # Launch 2D grid over (B, cdiv(D, BLOCK_D))
    BLOCK_D = 256
    grid = (B, triton.cdiv(D, BLOCK_D))

    _embed_sincos_2d_kernel[grid](
        t, freq, x,
        B=B, D=D, HALF=half,
        BLOCK_D=BLOCK_D,
        num_warps=2,
        num_stages=2,
    )
    return x


def _timestep_embedding_torch(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Torch reference implementation of timestep embedding.
    """
    half = dim // 2
    i = torch.arange(start=0, end=half, device=t.device, dtype=torch.float32)
    log_max = math.log(max_period)
    freq = torch.exp(-log_max * i / half)  # [half]
    args = (t.float().view(-1, 1)) * freq.view(1, -1)  # [B, half]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, dim]
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# -----------------------------
# Modules (keep torch linears & silu for stability and speed)
# -----------------------------
class Matmul(nn.Module):
    """Pure functional linear: takes input, weight, and optional bias as forward args."""
    def forward(self, input, weight, bias=None):
        return torch.nn.functional.linear(input, weight, bias)

class Linear(nn.Module):
    """Parametric linear: stores weight and bias internally."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

    def forward(self, input):
        return torch.nn.functional.linear(input, self.weight, self.bias)

class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(x)

# -----------------------------
# Entry point: ModelNew
# -----------------------------

class ModelNew(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.ModuleList(
            [
                Linear(frequency_embedding_size, hidden_size, bias=True),
                SiLU(),
                Linear(hidden_size, hidden_size, bias=True),
            ]
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        # Use Triton if available + CUDA; else torch
        if _HAS_TRITON and t.is_cuda:
            return _timestep_embedding_triton_from_freq(t, dim, max_period)
        return _timestep_embedding_torch(t, dim, max_period)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = self.timestep_embedding(t, self.frequency_embedding_size)
        for layer in self.mlp:
            x = layer(x)
        return x

OasisTimestepEmbedder = ModelNew
