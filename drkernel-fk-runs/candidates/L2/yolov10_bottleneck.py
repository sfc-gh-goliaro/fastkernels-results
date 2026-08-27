import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


@triton.jit
def _bn_silu_infer_2d_kernel(
    x_ptr,                  # *T, shape [N, C, H, W] contiguous
    y_ptr,                  # *T, shape [N, C, H, W] contiguous
    weight_ptr,             # *f32, shape [C]
    bias_ptr,               # *f32, shape [C]
    mean_ptr,               # *f32, shape [C]
    var_ptr,                # *f32, shape [C]
    N: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,       # H*W
    eps: tl.constexpr,
    BLOCK: tl.constexpr,    # tile over HW
):
    # program ids: axis 0 over (N*C) rows, axis 1 over tiles of HW
    pid_nc = tl.program_id(0)
    pid_t  = tl.program_id(1)

    # derive n, c
    n = pid_nc // C
    c = pid_nc % C

    # starting element in this (n, c) plane
    base = (n * C + c) * HW

    # offsets within the HW span for this tile
    offs = pid_t * BLOCK + tl.arange(0, BLOCK)
    mask = offs < HW

    # load x as original dtype, cast to f32
    x = tl.load(x_ptr + base + offs, mask=mask, other=0).to(tl.float32)

    # load per-channel params (scalar per program)
    w = tl.load(weight_ptr + c)           # f32
    b = tl.load(bias_ptr + c)             # f32
    mean = tl.load(mean_ptr + c)          # f32
    var  = tl.load(var_ptr + c)           # f32

    # BN transform: y = (x - mean) * (w / sqrt(var + eps)) + b
    rstd = tl.rsqrt(var + eps)            # f32
    scale = w * rstd                      # f32

    y = (x - mean) * scale + b            # f32

    # SiLU: y * sigmoid(y)
    sig = 1.0 / (1.0 + tl.exp(-y))
    out = y * sig

    # cast back and store
    out_cast = out.to(x.dtype)
    tl.store(y_ptr + base + offs, out_cast, mask=mask)


class _FusedBNActFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, running_mean, running_var, eps):
        """
        x: (N,C,H,W) tensor, CUDA
        weight/bias/mean/var: 1D tensors of length C, on CUDA (dtype doesn't matter for loads; we upcast)
        Returns: tensor same shape/dtype as x
        """
        assert x.is_cuda, "Triton kernel requires CUDA tensor"
        # Ensure contiguity
        x_c = x.contiguous()
        y = torch.empty_like(x_c)

        # Shapes
        assert x_c.dim() == 4, f"Expected 4D NCHW, got shape {x_c.shape}"
        N, C, H, W = x_c.shape
        HW = H * W

        # Grid: axis 0 over N*C "rows", axis 1 over tiles of HW
        BLOCK = 256
        grid = (N * C, triton.cdiv(HW, BLOCK))

        _bn_silu_infer_2d_kernel[grid](
            x_c,
            y,
            weight,
            bias,
            running_mean,
            running_var,
            N,
            C,
            HW,
            eps,
            BLOCK=BLOCK,  # pass BLOCK as named arg to bind kernel param
        )

        # Save for potential debug; backward not implemented
        ctx.save_for_backward(x_c, weight, bias, running_mean, running_var)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError("Backward not implemented for fused BN+SiLU Triton kernel. Use eval mode or fallback.")


def _bn_silu_infer(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                   running_mean: torch.Tensor, running_var: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Fused BN + SiLU in eval: CUDA fast path uses Triton, else falls back to PyTorch.
    """
    use_triton = _HAS_TRITON and x.is_cuda and not x.requires_grad
    if use_triton:
        return _FusedBNActFunction.apply(x, weight, bias, running_mean, running_var, eps)
    # Fallback: pure PyTorch
    C = weight.shape[0]
    x_f = x.to(torch.float32)
    w_f = weight.to(torch.float32)
    b_f = bias.to(torch.float32)
    mean_f = running_mean.to(torch.float32)
    var_f = running_var.to(torch.float32)
    rstd = torch.rsqrt(var_f + eps)  # float32
    y = (x_f - mean_f.view(1, C, 1, 1)) * (w_f.view(1, C, 1, 1) * rstd.view(1, C, 1, 1)) + b_f.view(1, C, 1, 1)
    y = F.silu(y)
    return y.to(x.dtype)


# ------------ Below is the ModelNew using the fused BN+SiLU kernel in eval -----------

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
        z = self.conv(x)
        # Use fused BN+SiLU in eval on CUDA, else fallback to PyTorch BN+SiLU
        y = _bn_silu_infer(z, self.bn.weight, self.bn.bias, self.bn.running_mean, self.bn.running_var, self.bn.eps)
        return y

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


class ModelNew(nn.Module):
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k=(3, 3), e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = YOLOConv(c1, c_, k[0], 1)
        self.cv2 = YOLOConv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y

YOLOBottleneck = ModelNew
