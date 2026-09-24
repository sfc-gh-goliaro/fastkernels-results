import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def max_pool2d_kernel(
    x_ptr,                         # *const T (float16)
    y_ptr,                         # *T (float16)
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    BLOCK_OW: tl.constexpr,
):
    # Program ids
    pid_nc = tl.program_id(0)  # over N*C
    pid_oh = tl.program_id(1)  # over OH
    pid_ow_blk = tl.program_id(2)  # over blocks of OW

    # Decode n and c from pid_nc
    n = pid_nc // C
    c = pid_nc % C

    # Vector of ow for this block
    ow = pid_ow_blk * BLOCK_OW + tl.arange(0, BLOCK_OW)
    mask_ow = ow < OW

    # Base linear offset for (n, c, 0, 0) in NCHW
    base_nc = ((n * C + c) * H) * W

    # Initialize max to -inf in fp32
    maxv = tl.full([BLOCK_OW], -float("inf"), dtype=tl.float32)

    # Loop over kernel height/width
    for kh in range(0, KH):
        top = pid_oh * SH - PH + kh
        # height in-bounds is guaranteed by output-size formula, but keep check for safety
        in_h = (top >= 0) & (top < H)

        for kw in range(0, KW):
            left = ow * SW - PW + kw  # vector

            # width in-bounds and ow tail
            in_w = (left >= 0) & (left < W)
            mask = mask_ow & in_h & in_w

            # Linear index: base + top*W + left
            ptr = x_ptr + base_nc + top * W + left

            # Load as fp16, cast to fp32
            val = tl.load(ptr, mask=mask, other=-float("inf"))
            val_f32 = val.to(tl.float32)

            # Update max
            maxv = tl.maximum(maxv, val_f32)

    # Store result to y: y is contiguous NCHW as (N,C,OH,OW)
    y_index = (((n * C + c) * OH + pid_oh) * OW) + ow
    out = maxv.to(tl.float16)
    tl.store(y_ptr + y_index, out, mask=mask_ow)


class ModelNew(nn.Module):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
        ceil_mode: bool = False,
    ):
        super().__init__()
        # Normalize parameters to tuples
        if isinstance(kernel_size, int):
            self.kh = self.kw = int(kernel_size)
        else:
            self.kh, self.kw = int(kernel_size[0]), int(kernel_size[1])

        if stride is None:
            self.sh = self.sw = self.kh  # default stride == kernel
        elif isinstance(stride, int):
            self.sh = self.sw = int(stride)
        else:
            self.sh, self.sw = int(stride[0]), int(stride[1])

        if isinstance(padding, int):
            self.ph = self.pw = int(padding)
        else:
            self.ph, self.pw = int(padding[0]), int(padding[1])

        self.ceil_mode = bool(ceil_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback to torch if not CUDA
        if not x.is_cuda:
            return torch.nn.functional.max_pool2d(
                x,
                (self.kh, self.kw),
                (self.sh, self.sw),
                (self.ph, self.pw),
                ceil_mode=self.ceil_mode,
            )

        assert x.dim() == 4, f"Expected NCHW tensor, got shape {tuple(x.shape)}"
        N, C, H, W = x.shape

        # Compute output dimensions
        if self.ceil_mode:
            def ceil_div(a, b): return math.ceil(a / b)
            OH = ceil_div(H + 2 * self.ph - self.kh, self.sh) + 1
            OW = ceil_div(W + 2 * self.pw - self.kw, self.sw) + 1
        else:
            OH = (H + 2 * self.ph - self.kh) // self.sh + 1
            OW = (W + 2 * self.pw - self.kw) // self.sw + 1

        OH = max(OH, 0)
        OW = max(OW, 0)

        # Make input contiguous and ensure dtype is float16 for kernel
        if not x.is_contiguous():
            x = x.contiguous()
        if x.dtype != torch.float16:
            x = x.to(torch.float16)

        # Allocate output
        y = torch.empty((N, C, OH, OW), device=x.device, dtype=torch.float16)

        # Launch configuration: restore the previously correct setup
        BLOCK_OW = 32
        grid = (N * C, OH, triton.cdiv(OW, BLOCK_OW))

        max_pool2d_kernel[grid](
            x, y,
            N, C, H, W,
            self.kh, self.kw,
            self.sh, self.sw,
            self.ph, self.pw,
            OH, OW,
            BLOCK_OW=BLOCK_OW,
            num_warps=2,
        )

        return y

MaxPool2d = ModelNew
