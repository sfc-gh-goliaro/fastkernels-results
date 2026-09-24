import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def _t5_rmsnorm_kernel_2d(
    X,             # *ptr to input, shape [R, N], contiguous
    W,             # *ptr to weight, shape [N]
    Y,             # *ptr to output, shape [R, N], contiguous
    N,             # int: number of features (last dim)
    eps,           # float32 epsilon
    BLOCK_SIZE: tl.constexpr,
):
    # One program per row
    row = tl.program_id(0)

    # Base pointers for this row
    x_row = X + row * N
    y_row = Y + row * N

    offs = tl.arange(0, BLOCK_SIZE)

    # 1) Reduction: sum of squares in float32 over the row
    sum_x2 = tl.zeros((), dtype=tl.float32)

    col = 0
    while col < N:
        idx = col + offs
        mask = idx < N
        x = tl.load(x_row + idx, mask=mask, other=0.0)
        xf = x.to(tl.float32)
        sum_x2 += tl.sum(xf * xf, axis=0)
        col += BLOCK_SIZE

    # mean = E[x^2]; var = mean; inv_std = 1/sqrt(var + eps)
    mean = sum_x2 / N
    inv_std = tl.rsqrt(mean + eps)

    # 2) Scale by inv_std and multiply by weight; store as weight dtype
    col = 0
    while col < N:
        idx = col + offs
        mask = idx < N

        x = tl.load(x_row + idx, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + idx, mask=mask, other=0.0)  # weight in its native dtype

        y = x * inv_std                 # fp32
        out = y.to(w.dtype) * w         # cast y to w.dtype, then multiply in w.dtype

        tl.store(y_row + idx, out, mask=mask)
        col += BLOCK_SIZE


class ModelNew(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Fallback to PyTorch if Triton/GPU is not available
        if (not TRITON_AVAILABLE) or (not hidden_states.is_cuda):
            # Match original behavior exactly
            x32 = hidden_states.to(torch.float32)
            variance = x32.pow(2).mean(-1, keepdim=True)
            hidden_states_fp32 = x32 * torch.rsqrt(variance + self.variance_epsilon)
            # Cast to weight dtype before final multiply to match kernel behavior
            if self.weight.dtype in (torch.float16, torch.bfloat16):
                hidden_states = hidden_states_fp32.to(self.weight.dtype)
            else:
                hidden_states = hidden_states_fp32
            return self.weight * hidden_states

        # Expect shape [B, S, H]
        if hidden_states.dim() != 3:
            raise ValueError(f"Expected 3D tensor [B, S, H], got shape {tuple(hidden_states.shape)}")
        B, S, H = hidden_states.shape

        # Make contiguous and view as 2D [R, H] to simplify indexing
        x = hidden_states.contiguous()
        x2d = x.view(-1, H)
        w = self.weight.contiguous()
        y2d = torch.empty_like(x2d)

        R = x2d.shape[0]
        N = H

        # Grid: one program per row
        grid = (R,)

        eps_f32 = float(self.variance_epsilon)

        _t5_rmsnorm_kernel_2d[grid](
            x2d, w, y2d,
            N, eps_f32,
        )

        # Reshape back to [B, S, H]
        y = y2d.view(B, S, H)
        return y

T5LayerNorm = ModelNew
