import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import Triton; if not available, we'll fallback gracefully.
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


# -------------------------
# Triton kernel: 1x1 conv (NCHW)
# -------------------------

# Kernel: compute y[b, oc, :, :] = sum_ic W[oc, ic] * x[b, ic, :, :] + bias[oc]
# Assumptions:
#   - K=1, stride=1, padding=0, dilation=1, groups=1
#   - x: [B, Cin, H, W] (contiguous NCHW)
#   - w: [Cout, Cin] (contiguous)
#   - bias: [Cout] (contiguous)
#   - y: [B, Cout, H, W] (contiguous)
@triton.jit
def _conv1x1_nchw_kernel(
    x_ptr, w_ptr, bias_ptr, y_ptr,
    B: tl.constexpr, Cin: tl.constexpr, H: tl.constexpr, W: tl.constexpr, Cout: tl.constexpr,
    stride_x_b, stride_x_c, stride_x_h, stride_x_w,
    stride_w_oc, stride_w_ic,
    stride_y_b, stride_y_c, stride_y_h, stride_y_w,
    BLOCK: tl.constexpr,
):
    # Program ids
    b = tl.program_id(0)          # batch
    oc = tl.program_id(1)         # output channel
    tile = tl.program_id(2)       # tile over spatial

    L = H * W
    start = tile * BLOCK
    idx = start + tl.arange(0, BLOCK)
    mask = idx < L

    # h, w from linear idx
    w = idx % W
    h = idx // W

    # Accumulator
    acc = tl.zeros([BLOCK], dtype=tl.float32)

    # Loop over input channels
    for ic in range(0, Cin):
        # Load weight scalar W[oc, ic]
        w_val = tl.load(w_ptr + oc * stride_w_oc + ic * stride_w_ic)
        w_val = w_val.to(tl.float32)

        # Build pointer to X[b, ic, h, w] for vector idx
        x_offsets = (
            b * stride_x_b
            + ic * stride_x_c
            + h * stride_x_h
            + w * stride_x_w
        )
        x_vec = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        x_vec = x_vec.to(tl.float32)

        acc += w_val * x_vec

    # Add bias
    bias_val = tl.load(bias_ptr + oc)
    bias_val = bias_val.to(tl.float32)
    acc += bias_val

    # Store to Y[b, oc, h, w]
    y_offsets = b * stride_y_b + oc * stride_y_c + h * stride_y_h + w * stride_y_w
    tl.store(y_ptr + y_offsets, acc, mask=mask)


def _conv1x1_nchw(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """
    x: [B, Cin, H, W] (float16/float32), contiguous
    weight: [Cout, Cin] (same dtype as x), contiguous
    bias: [Cout] or None
    Returns y: [B, Cout, H, W] with same dtype as x
    """
    assert x.is_cuda, "Triton conv1x1 requires CUDA tensor"
    assert x.dim() == 4, f"Expected 4D NCHW, got shape {tuple(x.shape)}"
    B, Cin, H, W = x.shape
    Cout = weight.shape[0]
    assert weight.shape[1] == Cin, f"Weight shape mismatch: {weight.shape} vs Cin={Cin}"
    if bias is not None:
        assert bias.shape[0] == Cout, f"Bias shape mismatch: {bias.shape} vs Cout={Cout}"

    # Ensure contiguous
    x_c = x.contiguous()
    w_c = weight.contiguous()
    if bias is not None:
        bias_c = bias.contiguous()
    else:
        bias_c = torch.zeros(Cout, device=x.device, dtype=x.dtype)

    # Allocate output
    y = torch.empty((B, Cout, H, W), device=x.device, dtype=x.dtype)

    # Strides
    sx_b, sx_c, sx_h, sx_w = x_c.stride()
    sw_oc, sw_ic = w_c.stride()
    sy_b, sy_c, sy_h, sy_w = y.stride()

    # BLOCK over spatial
    L = H * W
    def _next_pow2(n):
        return 1 if n <= 1 else 1 << (int(math.ceil(math.log2(n))))
    BLOCK = min(_next_pow2(L), 1024)

    grid = (B, Cout, triton.cdiv(L, BLOCK))

    _conv1x1_nchw_kernel[grid](
        x_c, w_c, bias_c, y,
        B, Cin, H, W, Cout,
        sx_b, sx_c, sx_h, sx_w,
        sw_oc, sw_ic,
        sy_b, sy_c, sy_h, sy_w,
        BLOCK=BLOCK,
        num_warps=4,
        num_stages=2,
    )
    return y


# -------------------------
# Original modules (unchanged)
# -------------------------

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


class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)


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
        self.qkv = YOLOConv(dim, h, 1, 1, act=False)
        self.proj = YOLOConv(dim, dim, 1, 1, act=False)
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


# -------------------------
# Triton-optimized ModelNew (safer)
# -------------------------

class ModelNew(nn.Module):
    def __init__(self, c1: int, c2: int, e: float = 0.5):
        super().__init__()
        assert c1 == c2, f"ModelNew expects c1 == c2, got {c1} and {c2}"
        self.c = int(c1 * e)
        # Same structure
        self.cv1 = YOLOConv(c1, 2 * self.c, 1, 1)          # conv + bn + act
        self.cv2 = YOLOConv(2 * self.c, c1, 1, 1)           # conv + bn + act
        self.attn = YOLOAttention(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
        self.ffn = nn.Sequential(
            YOLOConv(self.c, self.c * 2, 1, 1),             # conv + bn + act
            YOLOConv(self.c * 2, self.c, 1, 1, act=False),   # conv (no act)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use_triton = _HAS_TRITON and x.is_cuda

        # 1) cv1: keep as-is (conv + bn + act)
        a, b = self.cv1(x).split((self.c, self.c), dim=1)

        # 2) attention on b: keep as-is
        b = b + self.attn(b)

        # 3) ffn:
        #    first YOLOConv includes BN+act: keep as-is
        b = self.ffn[0](b)
        #    second YOLOConv is 1x1 conv, no act: use Triton if CUDA and shapes match
        if use_triton:
            conv = self.ffn[1].conv          # type: Conv2d (1x1, groups=1)
            # Check shape compatibility
            if conv.weight.is_cuda and b.is_cuda:
                # Weight expected shape [Cout, Cin, 1, 1]; flatten to [Cout, Cin]
                w = conv.weight.view(conv.weight.shape[0], -1).contiguous()
                bias = conv.bias
                if bias is None:
                    bias = torch.zeros(conv.weight.shape[0], device=b.device, dtype=b.dtype)
                # Input b shape [B, Cin, H, W]
                y = _conv1x1_nchw(b, w, bias)
                b = y
        else:
            b = self.ffn[1](b)

        # 4) cv2: keep as-is (conv + bn + act)
        out = self.cv2(torch.cat((a, b), 1))
        return out

YOLOPSA = ModelNew
