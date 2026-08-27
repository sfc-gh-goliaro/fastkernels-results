import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: depthwise 2D convolution (groups == Cout)
@triton.jit
def _depthwise_conv2d_kernel(
    x_ptr, w_ptr, bias_ptr, y_ptr,
    # sizes (constexpr for loop unrolling)
    N: tl.constexpr, Cin: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    Cout: tl.constexpr, kH: tl.constexpr, kW: tl.constexpr,
    stride_h: tl.constexpr, stride_w: tl.constexpr,
    pad_h: tl.constexpr, pad_w: tl.constexpr,
    dil_h: tl.constexpr, dil_w: tl.constexpr,
    H_out: tl.constexpr, W_out: tl.constexpr,
    # strides in elements
    x_stride_n: tl.constexpr, x_stride_c: tl.constexpr, x_stride_h: tl.constexpr, x_stride_w: tl.constexpr,
    w_stride_c: tl.constexpr, w_stride_kh: tl.constexpr, w_stride_kw: tl.constexpr,
    y_stride_n: tl.constexpr, y_stride_c: tl.constexpr, y_stride_h: tl.constexpr, y_stride_w: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # program ids
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_tile = tl.program_id(2)

    # linear index over H_out*W_out
    offs = pid_tile * BLOCK + tl.arange(0, BLOCK)
    mask = offs < (H_out * W_out)
    ho = offs // W_out
    wo = offs % W_out

    # accumulator
    acc = tl.zeros([BLOCK], dtype=tl.float32)

    # loop over kernel height
    for kh in range(0, kH):
        in_h = ho * stride_h - pad_h + kh * dil_h
        # loop over kernel width (vectorized over kw)
        for kw in range(0, kW):
            in_w = wo * stride_w - pad_w + kw * dil_w

            # bounds for input
            in_bounds = (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W) & mask

            # depthwise: ci == co == pid_c
            ci = pid_c

            # x index: n, c=ci, h=in_h, w=in_w
            x_idx = (
                pid_n * x_stride_n
                + ci * x_stride_c
                + in_h * x_stride_h
                + in_w * x_stride_w
            )
            xv = tl.load(x_ptr + x_idx, mask=in_bounds, other=0.0).to(tl.float32)

            # w index: c=pid_c, kh, kw
            w_idx = (
                pid_c * w_stride_c
                + kh * w_stride_kh
                + kw * w_stride_kw
            )
            wv = tl.load(w_ptr + w_idx).to(tl.float32)

            acc += xv * wv

    if HAS_BIAS:
        bv = tl.load(bias_ptr + pid_c).to(tl.float32)
        acc += bv

    # store to y: [n, c, ho, wo]
    y_idx = (
        pid_n * y_stride_n
        + pid_c * y_stride_c
        + ho * y_stride_h
        + wo * y_stride_w
    )
    tl.store(y_ptr + y_idx, acc, mask=mask)


def _depthwise_conv2d_triton(x: torch.Tensor,
                             w: torch.Tensor,
                             bias: torch.Tensor | None,
                             stride: tuple[int, int] = (1, 1),
                             padding: tuple[int, int] = (0, 0),
                             dilation: tuple[int, int] = (1, 1)) -> torch.Tensor:
    """
    x: [N, C, H, W]
    w: [C, 1, kH, kW]  (groups == C, depthwise)
    bias: [C] or None
    Returns y: [N, C, H_out, W_out]
    """
    assert x.is_cuda, "Triton kernel requires CUDA tensor"
    assert x.ndim == 4, f"Expected NCHW, got shape {x.shape}"
    N, Cin, H, W = x.shape
    kH, kW = w.shape[-2], w.shape[-1]
    groups = w.shape[0]
    assert groups == Cin, f"Depthwise conv requires groups == Cin; got {groups} vs {Cin}"
    Cout = groups

    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    # output shape
    H_out = math.floor((H + 2 * ph - dh * (kH - 1) - 1) / sh) + 1
    W_out = math.floor((W + 2 * pw - dw * (kW - 1) - 1) / sw) + 1
    assert H_out > 0 and W_out > 0, "Invalid output shape; check stride/pad/dilation/kernel"

    # allocate
    y = torch.empty((N, Cout, H_out, W_out), device=x.device, dtype=x.dtype)

    # strides in elements
    x_strides = x.stride()  # (n, c, h, w)
    w_strides = w.stride()  # (co, ci, kh, kw) -> ci=0
    y_strides = y.stride()

    # tiling
    BLOCK = 256
    grid = (N, Cout, triton.cdiv(H_out * W_out, BLOCK))

    _depthwise_conv2d_kernel[grid](
        x, w, bias if bias is not None else w,  # valid ptr; checked by HAS_BIAS
        y,
        N, Cin, H, W,
        Cout, kH, kW,
        sh, sw,
        ph, pw,
        dh, dw,
        H_out, W_out,
        # strides
        x_strides[0], x_strides[1], x_strides[2], x_strides[3],
        w_strides[0], w_strides[2], w_strides[3],
        y_strides[0], y_strides[1], y_strides[2], y_strides[3],
        BLOCK,
        bias is not None,
        num_warps=4,
        num_stages=2,
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, c1: int, c2: int, k: int, s: int):
        super().__init__()
        # cv1: keep as original YOLOConv (1x1 + BN + act)
        self.cv1 = YOLOConv(c1, c2, 1, 1)
        # cv2: original YOLOConv(k, s, g=c2, act=False)
        self.cv2 = YOLOConv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1: cv1 -> use PyTorch conv2d (1x1 is fast and simple)
        x = self.cv1(x)

        # Stage 2: cv2 depthwise conv -> use Triton if available & CUDA, else fallback
        if x.is_cuda and TRITON_AVAILABLE:
            conv_mod: Conv2d = self.cv2.conv
            w = conv_mod.weight  # [C,1,k,k]
            b = conv_mod.bias    # [C] or None
            return _depthwise_conv2d_triton(
                x,
                w,
                b,
                stride=(self.cv2.conv.stride if isinstance(self.cv2.conv.stride, tuple) else (self.cv2.conv.stride, self.cv2.conv.stride)),
                padding=(self.cv2.conv.padding if isinstance(self.cv2.conv.padding, tuple) else (self.cv2.conv.padding, self.cv2.conv.padding)),
                dilation=(self.cv2.conv.dilation if isinstance(self.cv2.conv.dilation, tuple) else (self.cv2.conv.dilation, self.cv2.conv.dilation)),
            )
        else:
            # Fallback: PyTorch conv2d depthwise
            return torch.nn.functional.conv2d(
                x,
                self.cv2.conv.weight,
                self.cv2.conv.bias,
                stride=self.cv2.conv.stride,
                padding=self.cv2.conv.padding,
                dilation=self.cv2.conv.dilation,
                groups=x.shape[1],  # depthwise
            )


# Keep the original helper definitions
class Conv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int],
                 stride: int | tuple[int, int] = 1, padding: int | tuple[int, int] = 0,
                 groups: int = 1, dilation: int | tuple[int, int] = 1, bias: bool = True):
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
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
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
        if self.training and self.track_running_stats and self.num_batches_tracked is not None:
            self.num_batches_tracked.add_(1)
        return torch.nn.functional.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            self.momentum,
            self.eps,
        )


def _fuse_conv_bn(conv: Conv2d, bn: BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    w_conv = conv.weight.clone().view(conv.weight.shape[0], -1)
    w_bn = torch.diag(
        bn.weight.to(dtype=conv.weight.dtype).div(
            torch.sqrt(bn.eps + bn.running_var.to(dtype=conv.weight.dtype))
        )
    )
    fused_weight = torch.mm(w_bn, w_conv).view_as(conv.weight)

    conv_bias = conv.bias
    if conv_bias is None:
        conv_bias = torch.zeros(conv.weight.shape[0], device=conv.weight.device, dtype=conv.weight.dtype)
    b_bn = (
        bn.bias.to(dtype=conv.weight.dtype)
        - bn.weight.to(dtype=conv.weight.dtype)
        .mul(bn.running_mean.to(dtype=conv.weight.dtype))
        .div(torch.sqrt(bn.running_var.to(dtype=conv.weight.dtype) + bn.eps))
    )
    fused_bias = torch.mm(w_bn, conv_bias.reshape(-1, 1)).reshape(-1) + b_bn
    return fused_weight, fused_bias


class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(x)


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
    default_act = SiLU()

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p=None, g: int = 1, d: int = 1, act=True):
        super().__init__()
        self.conv = Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self._is_fused = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_fused:
            return self.act(self.conv(x))
        return self.act(self.bn(self.conv(x)))

    @torch.no_grad()
    def fuse(self):
        if self._is_fused:
            return self
        fused_weight, fused_bias = _fuse_conv_bn(self.conv, self.bn)
        self.conv.weight.data.copy_(fused_weight)
        self.conv.bias = nn.Parameter(fused_bias)
        delattr(self, "bn")
        self._is_fused = True
        return self

YOLOSCDown = ModelNew
