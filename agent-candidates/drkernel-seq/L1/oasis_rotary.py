import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


@triton.jit
def _outer_times_repeat2_kernel(
    pos_ptr,        # *T, shape [P]
    freq_ptr,       # *T, shape [D]
    out_ptr,        # *T, shape [P, 2*D]
    P: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # 2D program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]

    # Bounds masks
    mask_m = offs_m < P
    mask_n = offs_n < D

    # Load positions and frequencies (vectorized)
    pos = tl.load(pos_ptr + offs_m, mask=mask_m, other=0.0)    # [BM]
    freq = tl.load(freq_ptr + offs_n, mask=mask_n, other=0.0)  # [BN]

    # Compute outer product tile: [BM, BN]
    mat = pos[:, None] * freq[None, :]

    # Expand indices to [BM, BN] for pointer arithmetic
    row_ids = offs_m[:, None]          # [BM, 1]
    col_ids = offs_n[None, :]          # [1, BN]
    mask = mask_m[:, None] & mask_n[None, :]  # [BM, BN]

    # Column doubling: write to 2*col and 2*col+1
    col2 = 2 * col_ids                 # [1, BN]
    stride = 2 * D
    base = out_ptr + row_ids * stride + col2  # [BM, BN]

    # Store values; broadcast mat to [BM, BN]
    # First column (even): 2*col
    tl.store(base, mat, mask=mask)
    # Second column (odd): 2*col + 1
    tl.store(base + 1, mat, mask=mask)


class ModelNew(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        freqs_for: str = "lang",
        theta: float = 10000.0,
        max_freq: float = 10.0,
    ):
        super().__init__()
        self.dim = dim
        self.freqs_for = freqs_for
        if freqs_for == "lang":
            # freqs = 1 / (theta ** (k / dim)), k = 0..dim-1 step 2 => length D = dim//2
            k = torch.arange(0, dim, 2).float()
            freqs = 1.0 / (theta ** (k / dim))
        elif freqs_for == "pixel":
            # freqs = linspace(1, max_freq/2, dim//2) * pi
            n = dim // 2
            freqs = torch.linspace(1.0, max_freq / 2, steps=n) * math.pi
        else:
            raise ValueError(f"unsupported rotary mode: {freqs_for}")
        # Register as buffer so it moves with .to(device)
        self.register_buffer("freqs", freqs, persistent=False)

    @staticmethod
    def _choose_launch_config(P: int, D: int):
        # Simple heuristic: smaller blocks for tiny problems to reduce overprovisioning.
        if P <= 32 and D <= 32:
            return 32, 32, 2
        if P <= 64 and D <= 64:
            return 64, 64, 2
        # General case
        return 128, 128, 4

    def _triton_forward(self, positions: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """
        Compute out[p, 2*d + k] = positions[p] * freqs[d] for k in {0,1}, d in [0, D).
        Shape: [P, 2*D].
        """
        if not _HAS_TRITON:
            P = positions.shape[0]
            D = freqs.shape[0]
            mat = positions[:, None] * freqs[None, :]              # [P, D]
            out = mat.repeat_interleave(2, dim=1)                   # [P, 2D]
            return out

        if positions.device.type != "cuda" or freqs.device.type != "cuda":
            P = positions.shape[0]
            D = freqs.shape[0]
            mat = positions[:, None] * freqs[None, :]              # [P, D]
            out = mat.repeat_interleave(2, dim=1)                   # [P, 2D]
            return out

        P = positions.shape[0]
        D = freqs.shape[0]
        # Ensure dim == 2*D
        assert self.dim == 2 * D, f"Expected dim={2*D} but got dim={self.dim}"

        # Compute in freqs dtype to match original numerics
        compute_dtype = freqs.dtype
        pos = positions.to(compute_dtype)

        # Allocate output
        out = torch.empty((P, 2 * D), device=freqs.device, dtype=compute_dtype)

        # Launch config (shape-aware)
        BLOCK_M, BLOCK_N, num_warps = self._choose_launch_config(P, D)
        grid = (triton.cdiv(P, BLOCK_M), triton.cdiv(D, BLOCK_N))

        _outer_times_repeat2_kernel[grid](
            pos, freqs, out,
            P, D,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out

    def forward(
        self,
        t: torch.Tensor,
        freqs: torch.Tensor,
        seq_len: int | None = None,
        offset: int = 0,
    ) -> torch.Tensor:
        del seq_len, offset
        # Match original: ignore t, use self.freqs unless provided
        use_freqs = freqs if freqs is not None else self.freqs
        # Expect 1D positions
        if t.ndim != 1:
            raise ValueError("ModelNew.forward expects a 1D tensor for positions (t).")
        positions = t
        return self._triton_forward(positions, use_freqs)

OasisRotaryEmbedding = ModelNew
