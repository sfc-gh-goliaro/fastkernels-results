import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _softmax_rowwise_kernel(
    x_ptr,         # *const T, shape [M, N] contiguous over N
    y_ptr,         # *T, shape [M, N] contiguous over N
    stride_xm: tl.constexpr,  # row stride in elements
    stride_xn: tl.constexpr,  # col stride in elements (1 if contiguous)
    stride_ym: tl.constexpr,  # row stride in elements
    stride_yn: tl.constexpr,  # col stride in elements (1)
    N: tl.constexpr,          # number of columns
    BLOCK_SIZE: tl.constexpr, # block size (power-of-two >= N)
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    x_row_ptr = x_ptr + row * stride_xm + cols * stride_xn
    y_row_ptr = y_ptr + row * stride_ym + cols * stride_yn

    mask = cols < N

    # Load as float32 for stability; use -inf for masked lanes
    x = tl.load(x_row_ptr, mask=mask, other=-float('inf')).to(tl.float32)

    # Numerically stable softmax: subtract max
    x_max = tl.max(x, axis=0)
    x = x - x_max
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    softmax = num / den

    # Store; will cast to y dtype if needed
    tl.store(y_row_ptr, softmax, mask=mask)


def _next_power_of_two(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


class SoftmaxTriton(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback to torch if not CUDA or Triton not available
        if (not x.is_cuda) or (not TRITON_AVAILABLE):
            return torch.softmax(x, dim=self.dim)

        dim = self.dim
        if dim < 0:
            dim = dim + x.dim()

        # Move 'dim' to the last axis for 2D view [M, N]
        if dim != x.dim() - 1:
            order = [i for i in range(x.dim()) if i != dim] + [dim]
            x_moved = x.permute(order).contiguous()
            # Invert permutation to restore later
            restore_order = list(range(x.dim()))
            for i, p in enumerate(order):
                restore_order[p] = i
        else:
            x_moved = x.contiguous()
            restore_order = list(range(x.dim()))  # identity

        D = x_moved.shape[-1]
        M = x_moved.numel() // D
        x_2d = x_moved.view(M, D)

        # Allocate output
        y_2d = torch.empty_like(x_2d)

        # Strides in elements (contiguous after .contiguous())
        stride_xm = x_2d.stride(0)
        stride_xn = x_2d.stride(1)
        stride_ym = y_2d.stride(0)
        stride_yn = y_2d.stride(1)

        # Block size: power-of-two >= D, cap to 1024
        BLOCK = _next_power_of_two(D)
        BLOCK = min(BLOCK, 1024)

        # Launch
        grid = (M,)
        _softmax_rowwise_kernel[grid](
            x_2d, y_2d,
            stride_xm, stride_xn,
            stride_ym, stride_yn,
            D, BLOCK,
            num_warps=1,  # small vectors; 1 warp is sufficient
            num_stages=1,
        )

        # Restore shape and original dim order
        y_moved = y_2d.view(*x_moved.shape)
        if dim != x.dim() - 1:
            y = y_moved.permute(restore_order)
        else:
            y = y_moved

        return y


class Conv2d(nn.Module):
    """Parametric 2D convolution: stores weight and bias internally."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        groups: int = 1,
        dilation: int | tuple[int, int] = 1,
        bias: bool = True,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class ModelNew(nn.Module):
    def __init__(self, c1: int = 16):
        super().__init__()
        self.conv = Conv2d(c1, 1, 1, bias=False)
        self.conv.requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        # Initialize weights to a constant vector (value per in-channel)
        with torch.no_grad():
            self.conv.weight.data.copy_(x.view(1, c1, 1, 1).to(self.conv.weight.dtype))
        self.c1 = c1
        self._softmax = SoftmaxTriton(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, a = x.shape
        # Transform to (b, 4, c1, a)
        t = x.view(b, 4, self.c1, a)
        # Softmax over dim=1 (c1) using Triton when on CUDA
        s = self._softmax(t.transpose(2, 1))  # -> (b, c1, 4, a)
        # Conv: (b, c1, 4, a) -> (b, 1, 4, a)
        y = self.conv(s)
        return y.view(b, 4, a)

YOLODFL = ModelNew
