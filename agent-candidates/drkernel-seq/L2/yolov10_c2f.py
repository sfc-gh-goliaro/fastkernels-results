import math
import torch
import torch.nn as nn

from .yolov10_conv import YOLOConv  # keep cuDNN 1x1 convs

import triton
import triton.language as tl


@triton.jit
def depthwise_3x3_groups_kernel(
    x_ptr,         # *T, [B, C, H, W]
    w_ptr,         # *T, [C, 3, 3]
    b_ptr,         # *T, [C] (will be cast to fp32)
    y_ptr,         # *T, [B, C, H, W]
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    sN: tl.constexpr,
    sC: tl.constexpr,
    sH: tl.constexpr,
    sW: tl.constexpr,
    w_sC: tl.constexpr,   # weight stride over C (typically 9 for [C,3,3])
    has_bias: tl.constexpr,
    GROUPS: tl.constexpr,
    Cpg: tl.constexpr,    # channels per group
    BLOCK_C: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Program ids
    pid_nh = tl.program_id(0)  # over N*H
    pid_c  = tl.program_id(1)  # over C blocks
    pid_w  = tl.program_id(2)  # over W blocks

    n  = pid_nh // H
    h  = pid_nh % H

    c0 = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    w0 = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    mask_c = c0 < C
    mask_w = w0 < W

    # Accumulator
    acc = tl.zeros((BLOCK_C, BLOCK_W), dtype=tl.float32)

    # 3x3 loop (unrolled by Triton since constexpr)
    for kh in range(3):
        for kw in range(3):
            hi = h + kh
            wi = w0 + kw

            # group and local channel
            g  = c0 // Cpg
            loc= c0 % Cpg
            # input channel index
            c_in = g * Cpg + loc

            # x pointer block: [BLOCK_C, BLOCK_W]
            x_ptr_block = (
                x_ptr
                + n * sN
                + c_in[:, None] * sC
                + hi * sH
                + wi[None, :] * sW
            )
            # load and cast to fp32 with proper mask
            x_val = tl.load(x_ptr_block, mask=mask_c[:, None] & mask_w[None, :], other=0.0).to(tl.float32)

            # weight vector for outchannels c0 at (kh, kw)
            w_idx = (c0 * w_sC) + (kh * 3) + kw
            w_val = tl.load(w_ptr + w_idx, mask=mask_c, other=0.0).to(tl.float32)  # [BLOCK_C]

            # outer product accumulate
            acc += w_val[:, None] * x_val

    # Bias
    if has_bias:
        b_val = tl.load(b_ptr + c0, mask=mask_c, other=0.0).to(tl.float32)
        acc += b_val[:, None]

    # Store
    y_ptr_block = (
        y_ptr
        + n * sN
        + c0[:, None] * sC
        + h * sH
        + w0[None, :] * sW
    )
    store_mask = mask_c[:, None] & mask_w[None, :]
    tl.store(y_ptr_block, acc, mask=store_mask)  # acc is fp32; y matches x dtype


def _depthwise_3x3_groups(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None, groups: int) -> torch.Tensor:
    """
    x: [B, C, H, W]
    w: [C, 3, 3]
    b: [C] or None
    groups: int
    Returns y with same shape as x.
    """
    assert x.dim() == 4, f"Expected 4D NCHW, got {x.shape}"
    B, C, H, W = x.shape
    assert C % groups == 0, f"Channels {C} must be divisible by groups {groups}"
    assert w.shape == (C, 3, 3), f"Weight shape must be [C,3,3], got {w.shape}"
    device = x.device
    assert device.type == "cuda", "This Triton kernel requires CUDA device"
    assert w.device == device and (b is None or b.device == device), "Device mismatch"

    # Ensure contiguous
    x_ = x.contiguous()
    w_ = w.contiguous()
    y_ = torch.empty_like(x_)

    # Strides
    sN, sC, sH, sW = x_.stride()
    w_sC = w_.stride(0)  # should be 9

    # Heuristic block sizes
    if W <= 64:
        BLOCK_W = 64
        num_warps = 2
    elif W <= 128:
        BLOCK_W = 128
        num_warps = 4
    else:
        BLOCK_W = 256
        num_warps = 8

    BLOCK_C = 64 if C <= 64 else 128

    grid = (
        B * H,
        triton.cdiv(C, BLOCK_C),
        triton.cdiv(W, BLOCK_W),
    )

    has_bias = b is not None
    # Always pass a valid bias tensor; kernel gates usage with has_bias
    b_ptr = b if has_bias else y_  # dummy tensor on device (won't be read when has_bias=False)

    depthwise_3x3_groups_kernel[grid](
        x_, w_, b_ptr, y_,
        B, C, H, W,
        sN, sC, sH, sW,
        w_sC,
        has_bias,
        groups,
        C // groups,
        BLOCK_C=BLOCK_C,
        BLOCK_W=BLOCK_W,
        num_warps=num_warps,
        num_stages=3,
    )

    return y_


class YOLOBottleneckTriton(nn.Module):
    """
    Triton-optimized version of YOLOBottleneck with 3x3 depthwise, groups=g.
    Assumptions:
      - kH = kW = 3, stride=1, pad=0, dilation=1
      - shortcut can be True/False
    """
    def __init__(self, c: int, c_out: int, shortcut: bool, g: int, k: tuple = (3, 3), e: float = 1.0):
        super().__init__()
        assert c == c_out, "This Triton implementation assumes c_in == c_out for now"
        assert k == (3, 3), f"Only 3x3 is supported in this kernel, got {k}"
        assert e == 1.0, f"Expansion 1.0 expected; got {e}"
        self.c = c
        self.shortcut = shortcut
        self.groups = g

        # Parameters: depthwise conv weights [C,3,3] and bias [C]
        self.weight = nn.Parameter(torch.empty(c, 3, 3))
        self.bias = nn.Parameter(torch.empty(c))
        # Simple init
        bound = 1 / math.sqrt(c * 3 * 3)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = _depthwise_3x3_groups(x, self.weight, self.bias, groups=self.groups)
        if self.shortcut:
            y = y + x
        return y


class ModelNew(nn.Module):
    """
    Triton-optimized version of the provided Model:
      - Keep 1x1 convs as YOLOConv (cuDNN)
      - Replace each YOLOBottleneck with Triton depthwise 3x3 grouped kernel
    Entry point name: ModelNew
    Signature-compatible with YOLOC2f and YOLOC2fCIB (including 'lk' kw).
    """
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5, lk: bool = False):
        super().__init__()
        self.c = int(c2 * e)
        # First 1x1 conv: from c1 to 2*c
        self.cv1 = YOLOConv(c1, 2 * self.c, 1, 1)
        # Bottlenecks
        self.m = nn.ModuleList(
            YOLOBottleneckTriton(self.c, self.c, shortcut=shortcut, g=g, k=(3, 3), e=1.0)
            for _ in range(n)
        )
        # Second 1x1 conv: from (2+n)*c to c2
        self.cv2 = YOLOConv((2 + n) * self.c, c2, 1, 1)
        # Unused argument to match YOLOC2fCIB signature
        self.lk = lk

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) First conv
        t = self.cv1(x)  # [B, 2*c, H, W]
        # 2) Split
        y0, y1 = t.chunk(2, dim=1)  # each [B, c, H, W]
        # 3) Bottleneck chain
        t = y1
        y_list = [y0, y1]
        for i in range(len(self.m)):
            t = self.m[i](t)
            y_list.append(t)
        # 4) Concat
        y = torch.cat(y_list, dim=1)  # [B, (n+2)*c, H, W]
        # 5) Second conv
        out = self.cv2(y)
        return out

YOLOC2f = ModelNew
YOLOC2fCIB = ModelNew
