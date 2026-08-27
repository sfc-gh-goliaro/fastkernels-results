import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try import Triton
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


# ---------------------------
# Triton kernels
# ---------------------------

if _TRITON_AVAILABLE:
    @triton.jit
    def silu_bias_kernel(x_ptr, bias_ptr, y_ptr,
                         E: tl.constexpr,  # flattened spatial+batches size = N*H*W
                         C: tl.constexpr,  # channels
                         stride_x_e: tl.constexpr, stride_x_c: tl.constexpr,
                         stride_y_e: tl.constexpr, stride_y_c: tl.constexpr,
                         BLOCK_E: tl.constexpr):
        # program id over (channel, e-block)
        pid_c = tl.program_id(0)
        pid_e = tl.program_id(1)
        c = pid_c

        e = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
        mask = e < E

        # Load x[e, c]
        x = tl.load(x_ptr + e * stride_x_e + c * stride_x_c, mask=mask, other=0.0)
        # Load bias[c]
        b = tl.load(bias_ptr + c)
        # Compute in fp32
        xf = x.to(tl.float32)
        bf = b.to(tl.float32)
        z = xf + bf
        # SiLU: z * sigmoid(z) = z / (1 + exp(-z))
        s = 1.0 / (1.0 + tl.exp(-z))
        y = (z * s).to(x.dtype)

        # Store y[e, c]
        tl.store(y_ptr + e * stride_y_e + c * stride_y_c, y, mask=mask)


    @triton.jit
    def linear1x1_kernel(x_ptr,       # [E, CI]
                         w_ptr,       # [CO, CI] (note: not transposed; we index as w[co, ci])
                         b_ptr,       # [CO]
                         y_ptr,       # [E, CO]
                         E: tl.constexpr,
                         CI: tl.constexpr,
                         CO: tl.constexpr,
                         stride_x_e: tl.constexpr, stride_x_c: tl.constexpr,
                         stride_w_co: tl.constexpr, stride_w_ci: tl.constexpr,
                         stride_y_e: tl.constexpr, stride_y_c: tl.constexpr,
                         BLOCK_E: tl.constexpr,
                         BLOCK_CO: tl.constexpr):
        # 2D launch: over (co-block, e-block)
        pid_co = tl.program_id(0)
        pid_e = tl.program_id(1)

        co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)   # [BLOCK_CO]
        e = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)       # [BLOCK_E]

        mask_co = co < CO
        mask_e = e < E

        # Accumulator [BLOCK_E, BLOCK_CO]
        acc = tl.zeros((BLOCK_E, BLOCK_CO), dtype=tl.float32)

        # Reduction loop over CI
        for ci in range(0, CI):
            # x[e, ci] -> [BLOCK_E]
            x_vec = tl.load(x_ptr + e * stride_x_e + ci * stride_x_c,
                            mask=mask_e, other=0.0).to(tl.float32)  # [BE]
            # w[co, ci] -> [BLOCK_CO]
            w_vec = tl.load(w_ptr + co * stride_w_co + ci * stride_w_ci,
                            mask=mask_co, other=0.0).to(tl.float32)  # [BCO]
            # Outer product accumulate: acc += x[:, None] * w[None, :]
            acc += x_vec[:, None] * w_vec[None, :]

        # Add bias: broadcast over e
        bias = tl.load(b_ptr + co, mask=mask_co, other=0.0).to(tl.float32)  # [BCO]
        acc = acc + bias[None, :]

        # Apply SiLU activation (on the linear output)
        z = acc
        s = 1.0 / (1.0 + tl.exp(-z))
        y = (z * s)  # keep float32

        # Store to y[e, co] with 2D masked store
        ptrs = y_ptr + (e[:, None] * stride_y_e + co[None, :] * stride_y_c)
        mask2d = mask_e[:, None] & mask_co[None, :]
        tl.store(ptrs, y, mask=mask2d)


# ---------------------------
# Helpers for launching kernels
# ---------------------------

def _launch_silu_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x: [N, C, H, W] (contiguous)
    bias: [C]
    Returns y with same shape as x.
    """
    assert x.is_cuda, "Expected CUDA tensor for Triton kernel"
    assert _TRITON_AVAILABLE, "Triton not available"

    assert x.dim() == 4, f"Expected 4D tensor, got {x.shape}"
    N, C, H, W = x.shape
    E = N * H * W
    x_ = x.contiguous()
    y = torch.empty_like(x_)

    # View as matrix [E, C] to get simple strides
    x_view = x_.view(E, C)
    y_view = y.view(E, C)
    stride_x_e = x_view.stride(0)
    stride_x_c = x_view.stride(1)
    stride_y_e = y_view.stride(0)
    stride_y_c = y_view.stride(1)

    BLOCK_E = 2048
    grid = (C, triton.cdiv(E, BLOCK_E))

    silu_bias_kernel[grid](
        x_view, bias, y_view,
        E, C,
        stride_x_e, stride_x_c,
        stride_y_e, stride_y_c,
        BLOCK_E=BLOCK_E,
        num_warps=4,
        num_stages=2,
    )
    return y


def _launch_linear1x1(x: torch.Tensor,         # [N, CI, H, W]
                      w: torch.Tensor,         # [CO, CI]
                      b: torch.Tensor) -> torch.Tensor:
    """
    Compute y = silu(x @ w^T + b) with 1x1, using Triton.
    x: [N, CI, H, W] (NCHW)
    w: [CO, CI]
    b: [CO]
    Returns y: [N, CO, H, W]
    """
    assert x.is_cuda and _TRITON_AVAILABLE

    assert x.dim() == 4, f"Expected 4D tensor, got {x.shape}"
    N, CI, H, W = x.shape
    CO = w.shape[0]
    assert w.shape[1] == CI, f"Weight CI mismatch: {w.shape} vs x CI={CI}"
    assert b is not None and b.shape[0] == CO, f"bias shape mismatch: {b.shape} vs CO={CO}"

    x_ = x.contiguous()
    w_ = w.contiguous()
    b_ = b.contiguous()

    E = N * H * W
    x_view = x_.view(E, CI)         # [E, CI]
    # Output in float32 for numeric stability
    y = torch.empty((N, CO, H, W), device=x.device, dtype=torch.float32)
    y_view = y.view(E, CO)          # [E, CO]

    # Strides
    stride_x_e = x_view.stride(0)
    stride_x_c = x_view.stride(1)
    stride_w_co = w_.stride(0)
    stride_w_ci = w_.stride(1)
    stride_y_e = y_view.stride(0)
    stride_y_c = y_view.stride(1)

    # Tune blocks
    BLOCK_E = 256
    BLOCK_CO = 64
    grid = (triton.cdiv(CO, BLOCK_CO), triton.cdiv(E, BLOCK_E))

    linear1x1_kernel[grid](
        x_view, w_, b_, y_view,
        E, CI, CO,
        stride_x_e, stride_x_c,
        stride_w_co, stride_w_ci,
        stride_y_e, stride_y_c,
        BLOCK_E=BLOCK_E,
        BLOCK_CO=BLOCK_CO,
        num_warps=4,
        num_stages=2,
    )
    # Cast back to input dtype if needed
    if x.dtype != torch.float32:
        y = y.to(x.dtype)
    return y


# ---------------------------
# YOLOConv with fused SiLU+Bias (Triton, forced on CUDA)
# ---------------------------

class YOLOConv(nn.Module):
    default_act = SiLU()

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
        # Conv via cuDNN
        y = F.conv2d(
            x,
            self.weight,
            self.bias,              # this is the conv bias (if any)
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        # On CUDA with Triton: apply fused bias+SiLU
        if _TRITON_AVAILABLE and y.is_cuda:
            conv_bias = self.bias if self.bias is not None else torch.zeros(
                self.weight.shape[0], device=y.device, dtype=y.dtype
            )
            y = _launch_silu_bias(y, conv_bias)
            return y
        else:
            # Fallback: pure SiLU
            return F.silu(y)


# ---------------------------
# Replace 1x1 convs with Triton linear kernels (forced on CUDA)
# ---------------------------

def _yoloconv_1x1_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Wrapper to use Triton linear1x1 kernel for 1x1 conv.
    x: [N, CI, H, W]
    weight: [CO, CI, 1, 1] -> view as [CO, CI]
    bias: [CO]
    """
    w_ = weight.view(weight.shape[0], weight.shape[1]).contiguous()
    return _launch_linear1x1(x, w_, bias)


class _YOLOConv1x1Triton(YOLOConv):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight.shape[2] == 1 and self.weight.shape[3] == 1:
            # 1x1 conv -> use Triton linear
            w = self.weight
            b = self.bias
            return _yoloconv_1x1_triton(x, w, b)
        else:
            # Fallback to cuDNN conv
            y = F.conv2d(
                x,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            if _TRITON_AVAILABLE and y.is_cuda:
                conv_bias = self.bias if self.bias is not None else torch.zeros(
                    self.weight.shape[0], device=y.device, dtype=y.dtype
                )
                y = _launch_silu_bias(y, conv_bias)
            else:
                y = F.silu(y)
            return y


# Replace YOLOConv with the 1x1-Triton version
YOLOConv = _YOLOConv1x1Triton


# ---------------------------
# Other modules: keep as-is, SiLU is handled by YOLOConv
# ---------------------------

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


# ---------------------------
# Entry point: ModelNew
# ---------------------------

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem1 = YOLOConv(3, 16, 3, 2)
        self.stem2 = YOLOConv(16, 32, 3, 2)
        self.stage2 = YOLOC2f(32, 32, n=1, shortcut=True)
        self.down3 = YOLOConv(32, 64, 3, 2)
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
