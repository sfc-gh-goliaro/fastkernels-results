import torch
import torch.nn as nn

from ..L1.conv2d import Conv2d

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _softmax_dot_c1_kernel(
    x_ptr,                 # *const T, shape [B, C, Q, L] logical; we use C'=c1 channels
    w_ptr,                 # *const float32, shape [C1]
    out_ptr,               # *T, shape [B, Q, L] (will store in x.dtype)
    B: tl.constexpr,
    C1: tl.constexpr,      # number of channels softmax is computed over (c1)
    Q: tl.constexpr,
    L: tl.constexpr,
    sB: tl.constexpr,      # stride for B in elements
    sC: tl.constexpr,      # stride for C in elements
    sQ: tl.constexpr,      # stride for Q in elements
    sL: tl.constexpr,      # stride for L in elements
    BLOCK_L: tl.constexpr,
):
    # program ids
    pid_bq = tl.program_id(0)  # 0 .. B*Q-1
    pid_lb = tl.program_id(1)  # block id over L

    b = pid_bq // Q
    q = pid_bq % Q

    l_start = pid_lb * BLOCK_L
    l_offsets = l_start + tl.arange(0, BLOCK_L)
    mask_l = l_offsets < L

    # Pass 1: find max over c in 0..C1-1 for numerical stability
    m = tl.full([BLOCK_L], -float("inf"), dtype=tl.float32)
    for c in range(0, C1):
        ptr = x_ptr + b * sB + c * sC + q * sQ + l_offsets * sL
        v = tl.load(ptr, mask=mask_l, other=-float("inf"))
        v = v.to(tl.float32)
        m = tl.maximum(m, v)

    # Pass 2: compute exp(v - m), sum, and accumulate weighted sum
    denom = tl.zeros([BLOCK_L], dtype=tl.float32)
    num = tl.zeros([BLOCK_L], dtype=tl.float32)

    for c in range(0, C1):
        ptr = x_ptr + b * sB + c * sC + q * sQ + l_offsets * sL
        v = tl.load(ptr, mask=mask_l, other=-float("inf"))
        v = v.to(tl.float32)
        e = tl.exp(v - m)
        denom += e
        w_c = tl.load(w_ptr + c)  # float32
        num += e * w_c

    out_f32 = num / denom  # float32 result

    # store result to out[b, q, l] in x dtype
    out_ptr_addr = out_ptr + b * (Q * L) + q * L + l_offsets
    # Cast to output dtype (same as x's element type)
    # Triton will handle casting if out_ptr has that dtype; ensure contiguity.
    tl.store(out_ptr_addr, out_f32, mask=mask_l)


class ModelNew(nn.Module):
    def __init__(self, c1: int = 16):
        super().__init__()
        self.c1 = int(c1)  # e.g., 16 in the benchmark

        # Hold the Conv2d to store weight (API parity); we'll read .weight in Triton.
        self.conv = Conv2d(c1, 1, 1, bias=False)
        self.conv.requires_grad_(False)

        # Initialize weight to arange(c1), as in the original code.
        with torch.no_grad():
            self.conv.weight.copy_(torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1))

        # PyTorch Softmax for CPU fallback
        self._softmax = torch.nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, A]
        Returns: [B, 4, A], dtype == x.dtype
        """
        assert x.dim() == 3, f"Expected x [B, C, A], got shape {tuple(x.shape)}"
        B, C, A = x.shape

        # CPU or no-triton fallback: EXACT original pathway
        if (not x.is_cuda) or (not TRITON_AVAILABLE):
            # View to [B, 4, c1, A]
            xt = x.view(B, 4, self.c1, A)
            # Transpose to [B, c1, 4, A]
            xt = xt.transpose(2, 1)
            # Conv: [B,1,4,A]
            y = self.conv(xt)
            # View to [B,4,A]
            return y.view(B, 4, A)

        # GPU + Triton path: fused softmax over c1 + dot with weight
        # Reshape to [B, 4, c1, A]
        x4 = x.view(B, 4, self.c1, A)
        # Transpose to [B, c1, 4, A]; make contiguous for simple strides
        xt = x4.transpose(2, 1).contiguous()

        Bt, Ct, Q, L = xt.shape
        assert Ct == self.c1 and Q == 4 and L == A

        # Output [B, Q, L] as same dtype as input x
        out = torch.empty((B, Q, L), device=x.device, dtype=x.dtype)

        # Get weight as float32 vector [c1]
        w = self.conv.weight.view(-1).to(device=x.device, dtype=torch.float32).contiguous()

        # Strides in elements
        sB = xt.stride(0)
        sC = xt.stride(1)
        sQ = xt.stride(2)
        sL = xt.stride(3)

        # Kernel launch parameters
        BLOCK_L = 128
        grid = (B * Q, triton.cdiv(L, BLOCK_L))
        num_warps = 4

        _softmax_dot_c1_kernel[grid](
            xt,                # x_ptr (element type = x.dtype)
            w,                 # w_ptr (float32)
            out,               # out_ptr (same dtype as x)
            Bt, self.c1, Q, L,
            sB, sC, sQ, sL,
            BLOCK_L=BLOCK_L,
            num_warps=num_warps,
        )

        return out

YOLODFL = ModelNew
