import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def rotate_half_apply_1d_kernel(
    t_ptr,          # *const T, shape [B, L, D] (contiguous)
    freq_ptr,       # *const float32, shape [F]
    out_ptr,        # *T, shape [B, L, D] (contiguous)
    B: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    F: tl.constexpr,      # F = D // 2
    BLOCK_D: tl.constexpr,
):
    # 3D grid: (B, L, ceil_div(D, BLOCK_D))
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_db = tl.program_id(2)

    d_start = pid_db * BLOCK_D
    d = d_start + tl.arange(0, BLOCK_D)
    mask_d = d < D

    # Base offset for (b, l, 0)
    base = (((pid_b * L) + pid_l) * D)

    # Map d to frequency index f = d // 2
    f = d // 2  # vector length BLOCK_D
    # Load frequencies and compute cos/sin
    f32 = tl.load(freq_ptr + f, mask=mask_d, other=0.0)
    c = tl.cos(f32)
    s = tl.sin(f32)

    # Even/odd indices
    even = d
    odd  = d + 1

    # Compute input pointers
    ptr_even = t_ptr + base + even
    ptr_odd  = t_ptr + base + odd

    # Loads
    x1 = tl.load(ptr_even, mask=mask_d, other=0.0)
    x2 = tl.load(ptr_odd,  mask=mask_d, other=0.0)

    # Compute in float32
    x1f = x1.to(tl.float32)
    x2f = x2.to(tl.float32)
    cf  = c
    sf  = s

    # y1 = x1*c + x2*s
    # y2 = -x2*c + x1*s
    y1 = x1f * cf + x2f * sf
    y2 = -x2f * cf + x1f * sf

    # Store results (cast to input dtype)
    out_even = out_ptr + base + even
    out_odd  = out_ptr + base + odd

    tl.store(out_even, y1.to(x1.dtype), mask=mask_d)
    tl.store(out_odd,  y2.to(x2.dtype), mask=mask_d)


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
            # Classic RoPE: freqs = 1 / (theta^(2i / dim))
            arange = torch.arange(0, dim, 2).float()
            freqs = 1.0 / (theta ** (arange / dim))
        elif freqs_for == "pixel":
            # Pixel-wise: linspace [1, max_freq/2] * pi, length dim//2
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * math.pi
        else:
            raise ValueError(f"unsupported rotary mode: {freqs_for}")
        # Keep as buffer; no grad
        self.register_buffer("freqs", freqs, persistent=False)

    @property
    def device(self):
        return self.freqs.device

    def _forward_freqs(self, positions: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        # Same as original: einsum + repeat_interleave(2)
        pf = torch.einsum("..., f -> ... f", positions.to(freqs.dtype), freqs)
        return pf.repeat_interleave(2, dim=-1)

    def forward(
        self,
        t: torch.Tensor,
        freqs: torch.Tensor,
        seq_len: int | None = None,
        offset: int = 0,
    ) -> torch.Tensor:
        # Same contract as original: produce freqs tensor, do not apply to t
        del seq_len, offset
        return self._forward_freqs(t, freqs)

    def rotate_queries_or_keys(self, t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized rotate+mix for queries/keys.
        t: [B, L, D], float16/float32, CUDA
        freqs: [F], float* (will be upcast)
        Returns: [B, L, D]
        """
        assert t.is_cuda, "Triton kernel requires CUDA tensor"
        assert t.dim() == 3, f"Expected 3D tensor [B, L, D], got shape {tuple(t.shape)}"
        B, L, D = t.shape
        assert D % 2 == 0, f"Rotary embedding requires even dim, got D={D}"
        F = D // 2

        # Ensure contiguous
        t_c = t.contiguous()
        out = torch.empty_like(t_c)

        # Move freqs to device and upcast to float32
        freqs_dev = freqs
        if freqs_dev.device != t.device:
            freqs_dev = freqs_dev.to(t.device)
        freqs_f32 = freqs_dev.to(torch.float32).contiguous()

        # Launch configuration: 1D tiling over D for each (b,l)
        # Choose BLOCK_D close to D to minimize masks; cap at 128
        if D <= 32:
            BLOCK_D = 32
            num_warps = 1
        elif D <= 64:
            BLOCK_D = 64
            num_warps = 1
        else:
            BLOCK_D = 128
            num_warps = 2

        grid = (B, L, triton.cdiv(D, BLOCK_D))

        rotate_half_apply_1d_kernel[grid](
            t_c, freqs_f32, out,
            B, L, D, F,
            BLOCK_D=BLOCK_D,
            num_warps=num_warps,
            num_stages=1,
        )
        return out

    def get_axial_freqs(self, *dims: int) -> torch.Tensor:
        # Keep identical behavior: compute per-axis freqs and broadcast; does not apply to t
        colon = slice(None)
        all_freqs = []
        for index, dim in enumerate(dims):
            use_pixel = self.freqs_for == "pixel" and index >= len(dims) - 2
            if use_pixel:
                pos = torch.linspace(-1, 1, steps=dim, device=self.device)
            else:
                pos = torch.arange(dim, device=self.device)
            seq_freqs = self.forward(pos, self.freqs, seq_len=dim)
            axis = [None] * len(dims)
            axis[index] = colon
            all_freqs.append(seq_freqs[(Ellipsis, *axis, colon)])
        all_freqs = torch.broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim=-1)

OasisRotaryEmbedding = ModelNew
