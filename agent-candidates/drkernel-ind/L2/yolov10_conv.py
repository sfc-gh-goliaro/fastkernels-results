import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import Triton; if unavailable, we'll fall back to PyTorch ops.
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False

class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)

class Pad(nn.Module):
    """Functional padding op."""
    def forward(self, x: torch.Tensor, pad: tuple[int, ...], value: float = 0.0) -> torch.Tensor:
        return F.pad(x, pad, value=value)

def _ceil_div(a, b):
    return (a + b - 1) // b

# Triton kernel: out = silu(x + y) elementwise (out-of-place)
if _HAS_TRITON:
    @triton.jit
    def add_silu_kernel(x_ptr, y_ptr, out_ptr, n_elements: tl.int32, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements

        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        y = tl.load(y_ptr + offs, mask=mask, other=0.0)
        x32 = x.to(tl.float32)
        y32 = y.to(tl.float32)

        t = x32 + y32
        s = 1.0 / (1.0 + tl.exp(-t))
        out32 = t * s

        out = out32.to(x.dtype)
        tl.store(out_ptr + offs, out, mask=mask)

def _choose_block_and_warps(n):
    # Simple heuristic: larger blocks for larger tensors
    if n >= (1 << 20):       # >= ~1M elements
        return 4096, 8
    elif n >= (1 << 18):     # >= ~262k
        return 2048, 4
    else:
        return 1024, 4

def _triton_add_silu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Fused add + silu using Triton (out-of-place).
    Falls back to torch if Triton is not available or tensors are not CUDA.
    """
    if (not _HAS_TRITON) or (not x.is_cuda) or (not y.is_cuda):
        return torch.silu(x + y)

    assert x.shape == y.shape, f"Shapes must match for add+silu, got {x.shape} vs {y.shape}"
    assert x.dtype == y.dtype, f"Dtypes must match, got {x.dtype} vs {y.dtype}"

    # Ensure contiguous for simple 1D kernel
    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()

    n = x.numel()
    BLOCK, num_warps = _choose_block_and_warps(n)
    grid = (_ceil_div(n, BLOCK),)

    out = torch.empty_like(x)
    add_silu_kernel[grid](x, y, out, n, BLOCK=BLOCK, num_warps=num_warps)
    return out

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

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size))
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

class ModelNew(nn.Module):
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
        # Keep activation as a module to match original API
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self._is_fused = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv + BN
        z = self.bn(self.conv(x))
        # Apply activation
        return self.act(z)

    @torch.no_grad()
    def fuse(self):
        if self._is_fused:
            return self
        # Fuse BN into conv weights/bias (inference-style fusion)
        fused_weight, fused_bias = _fuse_conv_bn(self.conv, self.bn)
        self.conv.weight.data.copy_(fused_weight)
        self.conv.bias = nn.Parameter(fused_bias)
        delattr(self, "bn")
        self._is_fused = True
        return self

# The following helpers mirror the original for completeness.

class YOLOConv(nn.Module):
    def __init__(self, in_c, out_c, k, s, p, g=1, act=True):
        super().__init__()
        self.conv = Conv2d(in_c, out_c, k, s, p, groups=g, bias=not act)
        self.bn = BatchNorm2d(out_c)
        self.act = SiLU() if act else nn.Identity()
        self._is_fused = False

    def forward(self, x):
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

class YOLORepVGGDW(nn.Module):
    def __init__(self, ed: int):
        super().__init__()
        self.conv = YOLOConv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = YOLOConv(ed, ed, 3, 1, 0, g=ed, act=False)  # no padding
        self.act = SiLU()
        self._pad = Pad()
        self._is_fused = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_fused:
            return self.act(self.conv(x))
        # z = conv(x), z1 = conv1(x), out = silu(z + z1)
        z = self.conv(x)
        z1 = self.conv1(x)
        # Use Triton fused add+silu if possible (out-of-place)
        if _HAS_TRITON and z.is_cuda and z1.is_cuda and z.dtype == z1.dtype:
            return _triton_add_silu(z, z1)
        return self.act(z + z1)

    @torch.no_grad()
    def fuse(self):
        if self._is_fused:
            return self
        self.conv.fuse()
        self.conv1.fuse()
        # Algebraic parameter fusion: conv.weight += conv1.weight and sum biases
        w0 = self.conv.conv.weight
        w1 = self.conv1.conv.weight
        b0 = self.conv.conv.bias
        b1 = self.conv1.conv.bias
        final_conv_w = w0 + w1
        final_conv_b = b0 + b1
        self.conv.conv.weight.data.copy_(final_conv_w)
        self.conv.conv.bias.data.copy_(final_conv_b)
        delattr(self, "conv1")
        self._is_fused = True
        return self

def fuse_module(module: nn.Module) -> nn.Module:
    for child in module.children():
        fuse_module(child)
    if isinstance(module, YOLOConv):
        module.fuse()
    elif isinstance(module, YOLORepVGGDW):
        module.fuse()
    return module

YOLOConv = ModelNew
