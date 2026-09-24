import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# ------------------------------
# Triton kernels
# ------------------------------

# 3x3 depthwise conv (stride=1, pad=1), NCHW, fp16 I/O, compute in fp32
@triton.jit
def _depthwise_conv3x3_stride1_pad1_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # program ids
    pid_nc = tl.program_id(0)   # over N*C
    pid_pos = tl.program_id(1)  # over spatial blocks

    n = pid_nc // C
    c = pid_nc % C

    offs = pid_pos * BLOCK + tl.arange(0, BLOCK)
    mask = offs < (H * W)

    # decode spatial indices
    hh = offs // W
    ww = offs % W

    # base for x at (n, c, 0, 0)
    HW = H * W
    base_nc = (n * C + c) * HW

    acc = tl.zeros([BLOCK], dtype=tl.float32)

    # loop 3x3, pad=1 => only central 3x3 valid; bounds check anyway
    for dh in range(3):
        for dw in range(3):
            h = hh + dh - 1
            w = ww + dw - 1
            in_bounds = mask & (h >= 0) & (h < H) & (w >= 0) & (w < W)
            # x index = ((n*C + c)*HW) + h*W + w
            ptr = x_ptr + base_nc + h * W + w
            val = tl.load(ptr, mask=in_bounds, other=0.0).to(tl.float32)
            # weight index: c * 9 + dh*3 + dw  (flattened 3x3)
            wval = tl.load(w_ptr + (c * 9 + dh * 3 + dw)).to(tl.float32)
            acc += val * wval

    # add bias
    bias = tl.load(b_ptr + c).to(tl.float32)
    acc = acc + bias

    # silu activation
    sig = 1.0 / (1.0 + tl.exp(-acc))
    out = acc * sig

    # store
    y_ptr_pos = y_ptr + base_nc + offs
    tl.store(y_ptr_pos, out.to(tl.float16), mask=mask)


# 1x1 convolution (any IC, OC), NCHW, fp16 I/O, compute in fp32, fused bias+silu
@triton.jit
def _conv1x1_fused_bias_silu_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N: tl.constexpr, C_in: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    OC: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_IC: tl.constexpr,
):
    pid_pos = tl.program_id(0)   # over N*H*W
    pid_oc_blk = tl.program_id(1)  # over OC blocks

    HW = H * W
    n = pid_pos // HW
    hw = pid_pos % HW
    h = hw // W
    w = hw % W

    oc_start = pid_oc_blk * BLOCK_OC
    offs_oc = oc_start + tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC

    acc = tl.zeros([BLOCK_OC], dtype=tl.float32)

    # loop over input channels in BLOCK_IC chunks
    for ic_start in range(0, C_in, BLOCK_IC):
        offs_ic = ic_start + tl.arange(0, BLOCK_IC)
        mask_ic = offs_ic < C_in

        # unrolled loop over BLOCK_IC
        for i in range(BLOCK_IC):
            ic = offs_ic[i]
            if not mask_ic[i]:
                continue
            # w[oc, ic] -> linear index: oc*C_in + ic
            w_vec = tl.load(w_ptr + (offs_oc * C_in + ic), mask=mask_oc, other=0.0).to(tl.float32)
            # x[n, ic, h, w] -> linear: ((n*C_in + ic)*HW + hw)
            x_val = tl.load(x_ptr + ((n * C_in + ic) * HW + hw)).to(tl.float32)
            acc += w_vec * x_val

    # add bias and silu
    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0).to(tl.float32)
    acc = acc + bias
    sig = 1.0 / (1.0 + tl.exp(-acc))
    out = acc * sig

    # store to y: [N, OC, H, W]
    base_n = n * OC * HW
    y_idx = base_n + offs_oc * HW + hw
    tl.store(y_ptr + y_idx, out.to(tl.float16), mask=mask_oc)


# ------------------------------
# Python wrappers around kernels
# ------------------------------

def _triton_depthwise_conv3x3(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and w.is_cuda and b.is_cuda, "Triton kernels require CUDA tensors"
    assert x.dtype == torch.float16 and w.dtype == torch.float16 and b.dtype == torch.float16, "Use fp16"
    N, C, H, W = x.shape
    # w shape expected [C, 3, 3] -> view as [C, 9]
    w_flat = w.view(C, 9).contiguous()
    y = torch.empty_like(x)

    # Tuned params: larger block, more warps/stages
    BLOCK = 512
    grid = (N * C, triton.cdiv(H * W, BLOCK))
    _depthwise_conv3x3_stride1_pad1_kernel[grid](
        x, w_flat, b, y,
        N, C, H, W,
        BLOCK=BLOCK,
        num_warps=8,
        num_stages=3,
    )
    return y


def _triton_conv1x1_fused(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and w.is_cuda and b.is_cuda, "Triton kernels require CUDA tensors"
    assert x.dtype == torch.float16 and w.dtype == torch.float16 and b.dtype == torch.float16, "Use fp16"
    N, C_in, H, W = x.shape
    OC = w.shape[0]
    assert w.shape == (OC, C_in), f"weight shape must be [OC, IC]; got {w.shape}"
    y = torch.empty((N, OC, H, W), device=x.device, dtype=x.dtype)

    # Tuned params
    BLOCK_OC = 128
    BLOCK_IC = 32
    grid = (N * H * W, triton.cdiv(OC, BLOCK_OC))
    _conv1x1_fused_bias_silu_kernel[grid](
        x, w, b, y,
        N, C_in, H, W,
        OC,
        BLOCK_OC=BLOCK_OC, BLOCK_IC=BLOCK_IC,
        num_warps=8,
        num_stages=3,
    )
    return y


# ------------------------------
# Replacements for original modules
# ------------------------------

class _TritonYOLOConv(nn.Module):
    """
    Drop-in replacement for YOLOConv that uses Triton kernels for 3x3 depthwise and 1x1 convs.
    Falls back to PyTorch if Triton not available or tensor not CUDA.
    """
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p=None, g: int = 1, d: int = 1, bias=True, act=True):
        super().__init__()
        self.k = k
        self.s = s
        self.g = g
        self.d = d
        self.padding = p if p is not None else (k // 2 if isinstance(k, int) and k > 1 else 0)
        self.stride = s
        self.dilation = d
        self.groups = g
        self.use_triton = True

        # weight layout: [Cout, Cin/groups, kH, kW]
        if isinstance(k, int):
            kH = kW = k
        else:
            kH, kW = int(k[0]), int(k[1])
        self.weight = nn.Parameter(torch.empty(c2, c1 // g, kH, kW))
        if bias:
            self.bias = nn.Parameter(torch.empty(c2))
        else:
            self.register_parameter("bias", None)
        # activation policy
        self.act = True if act is True else isinstance(act, nn.Module)

        # init
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = (c1 // g) * kH * kW
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def _fallback_conv(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use_triton = (
            self.use_triton
            and TRITON_AVAILABLE
            and x.is_cuda
            and x.dtype == torch.float16
        )
        if not use_triton:
            y = self._fallback_conv(x)
            if self.act:
                y = F.silu(y)
            return y

        N, C_in, H, W = x.shape
        Cout = self.weight.shape[0]
        kH = self.weight.shape[-2]
        kW = self.weight.shape[-1]
        Cin_g = self.weight.shape[1]  # per-group input channels
        # bias
        b = self.bias
        if b is None:
            b = torch.zeros(Cout, device=x.device, dtype=x.dtype)

        if self.k == 3 and self.s == 1 and self.padding == 1 and self.d == 1 and self.groups == Cout:
            # depthwise 3x3, NCHW
            # ensure weight view [C,3,3]
            w_d = self.weight.view(Cout, 3, 3).contiguous()
            y = _triton_depthwise_conv3x3(x, w_d, b)
        elif self.k == 1 and self.s == 1 and self.padding == 0 and self.d == 1:
            # 1x1 conv, fused bias+silu
            w_1 = self.weight.view(Cout, C_in).contiguous()
            y = _triton_conv1x1_fused(x, w_1, b)
        else:
            y = self._fallback_conv(x)

        return y


# ------------------------------
# Other modules (drop-in)
# ------------------------------

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
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class BatchNorm2d(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
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
        return F.batch_norm(
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
        return F.silu(x)


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

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p=None,
        g: int = 1,
        d: int = 1,
        act=True,
    ):
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


class MaxPool2d(nn.Module):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
        ceil_mode: bool = False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            ceil_mode=self.ceil_mode,
        )


class YOLOSPPF(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = YOLOConv(c1, c_, 1, 1)
        self.cv2 = YOLOConv(c_ * 4, c2, 1, 1)
        self.m = MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class YOLOSCDown(nn.Module):
    def __init__(self, c1: int, c2: int, k: int, s: int):
        super().__init__()
        self.cv1 = YOLOConv(c1, c2, 1, 1)
        self.cv2 = YOLOConv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv2(self.cv1(x))


class Softmax(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=self.dim)


class YOLOAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = YOLOConv(dim, h, 1, act=False)
        self.proj = YOLOConv(dim, dim, 1, act=False)
        self.pe = YOLOConv(dim, dim, 3, 1, g=dim, act=False)
        self._softmax = Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        qkv = self.qkv(x)
        q, k, v = qkv.view(b, self.num_heads, self.key_dim * 2 + self.head_dim, n).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = self._softmax(attn)
        x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.pe(v.reshape(b, c, h, w))
        return self.proj(x)


class YOLOPSA(nn.Module):
    def __init__(self, c1: int, c2: int, e: float = 0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = YOLOConv(c1, 2 * self.c, 1, 1)
        self.cv2 = YOLOConv(2 * self.c, c1, 1, 1)
        self.attn = YOLOAttention(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
        self.ffn = nn.Sequential(
            YOLOConv(self.c, self.c * 2, 1, 1),
            YOLOConv(self.c * 2, self.c, 1, 1, act=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class YOLOBottleneck(nn.Module):
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k=(3, 3), e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = YOLOConv(c1, c_, k[0], 1)
        self.cv2 = YOLOConv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class YOLOC2f(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = YOLOConv(c1, 2 * self.c, 1, 1)
        self.cv2 = YOLOConv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(
            YOLOBottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ------------------------------
# New model: ModelNew
# ------------------------------

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem1 = _TritonYOLOConv(3, 16, 3, 2)
        self.stem2 = _TritonYOLOConv(16, 32, 3, 2)
        self.stage2 = YOLOC2f(32, 32, n=1, shortcut=True)
        self.down3 = _TritonYOLOConv(32, 64, 3, 2)
        self.stage3 = YOLOC2f(64, 64, n=2, shortcut=True)
        self.down4 = YOLOSCDown(64, 128, 3, 2)
        self.stage4 = YOLOC2f(128, 128, n=2, shortcut=True)
        self.down5 = YOLOSCDown(128, 256, 3, 2)
        self.stage5 = YOLOC2f(256, 256, n=1, shortcut=True)
        self.sppf = YOLOSPPF(256, 256, 5)
        self.psa = YOLOPSA(256, 256)

    def forward(self, x: torch.Tensor):
        x = self.stem1(x)
        x = self.stem2(x)
        p2 = self.stage2(x)
        x = self.down3(p2)
        p3 = self.stage3(x)
        x = self.down4(p3)
        p4 = self.stage4(x)
        x = self.down5(p4)
        p5 = self.stage5(x)
        p5 = self.sppf(p5)
        p5 = self.psa(p5)
        return {"p3_backbone": p3, "p4_backbone": p4, "p5_backbone": p5}

YOLOv10Backbone = ModelNew
