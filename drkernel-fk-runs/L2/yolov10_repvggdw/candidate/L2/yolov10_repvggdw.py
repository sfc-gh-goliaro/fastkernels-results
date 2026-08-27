import math
import torch
import torch.nn as nn

# Try to import triton; if unavailable, we'll fallback to torch.conv2d
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


# ---------------------------
# Required modules (local)
# ---------------------------

class Conv2d(nn.Module):
    """Parametric 2D convolution: stores weight and bias internally."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int],
                 stride: int | tuple[int, int] = 1, padding: int | tuple[int, int] = 0,
                 dilation: int | tuple[int, int] = 1, groups: int = 1, bias: bool = True):
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

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(
            x, self.weight, self.bias, stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups
        )


class BatchNorm2d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.batch_norm(
            x, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, self.momentum, self.eps
        )


class Pad(nn.Module):
    """Functional padding op."""
    def forward(self, x: torch.Tensor, pad: tuple[int, ...], value: float = 0.0) -> torch.Tensor:
        return torch.nn.functional.pad(x, pad, value=value)


def autopad(k: int | tuple[int, int], p=None, d: int = 1):
    if isinstance(k, tuple):
        if d > 1:
            k = tuple(d * (x - 1) + 1 for x in k)
        if p is None:
            return tuple(x // 2 for x in k)
        return p
    if d > 1:
        k = d * (k - 1) + 1
    return k // 2 if p is None else p


class YOLOConv(nn.Module):
    default_act = nn.SiLU  # callable module

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p=None,
                 g: int = 1, d: int = 1, act=True):
        super().__init__()
        self.conv = Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        if act is True:
            self.act = YOLOConv.default_act()
        elif isinstance(act, nn.Module):
            self.act = act
        else:
            self.act = nn.Identity()
        self._is_fused = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_fused:
            return self.act(self.conv(x))
        return self.act(self.bn(self.conv(x)))

    @torch.no_grad()
    def fuse(self):
        if self._is_fused:
            return self
        # Fuse Conv2d + BN into Conv2d using running stats
        conv = self.conv
        bn = self.bn
        w_conv = conv.weight.view(conv.weight.shape[0], -1)          # [OC, C*KH*KW]
        scale = bn.weight.to(w_conv.dtype).div((bn.running_var.to(w_conv.dtype) + bn.eps).sqrt())
        w_scaled = w_conv * scale.view(-1, 1)                         # [OC, C*KH*KW]
        fused_weight = w_scaled.view_as(conv.weight)                   # [OC, C//g, KH, KW]

        conv_bias = conv.bias
        if conv_bias is None:
            conv_bias = torch.zeros(conv.weight.shape[0], device=conv.weight.device, dtype=conv.weight.dtype)
        b_bn = bn.bias.to(w_conv.dtype) - bn.weight.to(w_conv.dtype) * bn.running_mean.to(w_conv.dtype) \
            / (bn.running_var.to(w_conv.dtype) + bn.eps).sqrt()
        b_fused = (conv_bias.view(1, -1) @ scale.view(-1, 1)).reshape(-1) + b_bn

        conv.weight.data.copy_(fused_weight)
        conv.bias = nn.Parameter(b_fused)
        delattr(self, "bn")
        self._is_fused = True
        return self


# ---------------------------
# Triton kernel & launcher
# ---------------------------

@triton.jit
def depthwise_conv2d_kernel(
    x_ptr, w_ptr, y_ptr,
    N: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr, P: tl.constexpr,
    sN: tl.constexpr, sC: tl.constexpr, sH: tl.constexpr, sW: tl.constexpr,
    wsC: tl.constexpr, wsKH: tl.constexpr, wsKW: tl.constexpr,
    ysN: tl.constexpr, ysC: tl.constexpr, ysH: tl.constexpr, ysW: tl.constexpr,
    BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr,
):
    # Program ids
    pid_c = tl.program_id(0)   # output channel index
    pid_nh = tl.program_id(1)  # flattened (n, oh)
    pid_w = tl.program_id(2)   # w tile id

    oh = pid_nh % H
    n  = pid_nh // H

    w_start = pid_w * BLOCK_W
    ow = w_start + tl.arange(0, BLOCK_W)
    mask_ow = ow < W

    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Unrolled small-kernel loops
    for kh in range(0, KH):
        ih = oh + kh - P
        in_bounds_h = (ih >= 0) & (ih < H)
        for kw in range(0, KW):
            iw = ow + kw - P
            in_bounds_w = (iw >= 0) & (iw < W)
            valid = mask_ow & in_bounds_w & in_bounds_h

            for c0 in range(0, C, BLOCK_C):
                c = c0 + tl.arange(0, BLOCK_C)
                mask_c = c < C

                # x[n, c, ih, iw]
                x_ptrs = x_ptr + n * sN + c[:, None] * sC + ih * sH + iw[None, :] * sW
                x_mask = mask_c[:, None] & valid[None, :]
                xv = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)  # [BC,BW]

                # w[c, kh, kw]
                w_ptrs = w_ptr + c * wsC + kh * wsKH + kw * wsKW
                wv = tl.load(w_ptrs, mask=mask_c, other=0.0).to(tl.float32)  # [BC]

                # FMA accumulate with small loop to reduce temporaries
                for ci in range(0, BLOCK_C):
                    if ci < C - c0:
                        acc += xv[ci, :] * wv[ci]

    y_ptrs = y_ptr + n * ysN + pid_c * ysC + oh * ysH + ow * ysW
    tl.store(y_ptrs, acc, mask=mask_ow)


def _choose_blocks(W: int, C: int):
    # Choose BLOCK_W to match W and be a small power-of-two or close
    if W >= 128:
        bw = 128
    elif W >= 64:
        bw = 64
    elif W >= 32:
        bw = 32
    elif W >= 16:
        bw = 16
    else:
        bw = 8
    # Choose BLOCK_C as 32 or 64
    bc = 64 if C >= 128 else 32
    return bw, bc


def _triton_depthwise_conv2d(x: torch.Tensor, w: torch.Tensor, padding: int) -> torch.Tensor:
    """
    Launches the Triton depthwise conv2d kernel.
    x: (N,C,H,W) float16/float32 on CUDA
    w: (C,KH,KW) on CUDA
    Returns y: (N,C,H,W) same dtype as x
    """
    assert x.is_cuda, "Triton kernel requires CUDA tensor"
    assert x.ndim == 4, f"Expected 4D NCHW, got shape {x.shape}"
    N, C, H, W = x.shape
    KH, KW = w.shape[-2], w.shape[-1]
    assert w.shape[0] == C, f"Depthwise: w.shape[0]={w.shape[0]} must equal C={C}"

    # Output (match input dtype)
    y = torch.empty_like(x)

    # Strides in elements
    sN, sC, sH, sW = x.stride(0), x.stride(1), x.stride(2), x.stride(3)
    wsC, wsKH, wsKW = w.stride(0), w.stride(1), w.stride(2)
    ysN, ysC, ysH, ysW = y.stride(0), y.stride(1), y.stride(2), y.stride(3)

    BLOCK_W, BLOCK_C = _choose_blocks(W, C)
    grid = (C, N * H, triton.cdiv(W, BLOCK_W))

    depthwise_conv2d_kernel[grid](
        x, w, y,
        N, C, H, W,
        KH, KW, padding,
        sN, sC, sH, sW,
        wsC, wsKH, wsKW,
        ysN, ysC, ysH, ysW,
        BLOCK_W=BLOCK_W, BLOCK_C=BLOCK_C,
        num_warps=8, num_stages=3,
    )
    return y


# ---------------------------
# Triton-optimized ModelNew
# ---------------------------

class ModelNew(nn.Module):
    def __init__(self, ed: int):
        super().__init__()
        # Build blocks locally (no external dependencies)
        self.conv = YOLOConv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = YOLOConv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.act = nn.SiLU()  # same as YOLOConv.default_act()
        self._is_fused = False
        self._conv_fused = None  # will hold lambda after fuse

    def _fused_conv(self):
        # Ensure fusion: Conv2d + BN -> Conv2d for both, then kernel add
        self.conv.fuse()
        self.conv1.fuse()

        w7 = self.conv.conv.weight.data        # (OC,1,7,7) -> view (OC,7,7)
        b7 = self.conv.conv.bias.data          # (OC,)
        w3 = self.conv1.conv.weight.data       # (OC,1,3,3)  -> view (OC,3,3)
        b3 = self.conv1.conv.bias.data         # (OC,)

        # Pad w3 to 7x7 and add
        C, KH3, KW3 = w3.shape
        pad = 2
        w3_padded = torch.zeros((C, 7, 7), device=w3.device, dtype=w3.dtype)
        w3_padded[:, pad:pad+KH3, pad:pad+KW3] = w3.view(C, KH3, KW3)
        w_fused = (w7.view(C, 7, 7) + w3_padded).contiguous()
        b_fused = (b7 + b3).contiguous()

        # Remove conv1 to save memory
        del self.conv1

        # Cache fused params
        self._w_fused = w_fused
        self._b_fused = b_fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._is_fused:
            # Unfused: two convs + add + act
            return self.act(self.conv(x) + self.conv1(x))

        # Fused: one conv + act
        if not self._conv_fused:
            self._fused_conv()
            # Choose kernel or fallback
            use_triton = _HAS_TRITON and x.is_cuda
            if use_triton:
                self._conv_fused = lambda t: _triton_depthwise_conv2d(t, self._w_fused, padding=3)
            else:
                # Fallback: torch conv (depthwise)
                self._conv_fused = lambda t: torch.nn.functional.conv2d(
                    t, self._w_fused.view(ed, 1, 7, 7), self._b_fused, stride=1, padding=3, groups=ed
                )

        y = self._conv_fused(x)
        return self.act(y)

    @torch.no_grad()
    def fuse(self):
        if self._is_fused:
            return self
        # Perform fusion and setup
        self._is_fused = True
        self._conv_fused = None  # will rebuild
        return self

YOLORepVGGDW = ModelNew
