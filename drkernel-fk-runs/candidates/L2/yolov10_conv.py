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


# Variant A: 2D tiling over (oh, ow); supports unrolling over kW
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OH': 32, 'BLOCK_OW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 16, 'BLOCK_OW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 64, 'BLOCK_OW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 16, 'BLOCK_OW': 64}, num_warps=8, num_stages=3),
    ],
    key=['OH', 'OW', 'Cin', 'kH', 'kW'],
)
@triton.jit
def conv2d_silu_kernel_2d(
    x_ptr,                 # *float16, [N, Cin, H, W]
    w_ptr,                 # *float16, [Cout, Cin, kH, kW]
    b_ptr,                 # *float16 or dummy, [Cout]
    y_ptr,                 # *float16, [N, Cout, OH, OW]
    # sizes
    N: tl.constexpr,
    Cin: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    Cout: tl.constexpr,
    kH: tl.constexpr,
    kW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    # conv params
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dil_h: tl.constexpr,
    dil_w: tl.constexpr,
    # tiling
    BLOCK_OH: tl.constexpr,
    BLOCK_OW: tl.constexpr,
    # bias flag
    HAS_BIAS: tl.constexpr,
    # unroll flag for kW
    UNROLL_KW: tl.constexpr,
):
    # Grid = (Cout*N, ceil_div(OH, BLOCK_OH), ceil_div(OW, BLOCK_OW))
    pid_nc = tl.program_id(0)
    pid_oh = tl.program_id(1)
    pid_ow = tl.program_id(2)

    co = pid_nc % Cout
    n = pid_nc // Cout

    oh0 = pid_oh * BLOCK_OH
    ow0 = pid_ow * BLOCK_OW

    oh = oh0 + tl.arange(0, BLOCK_OH)
    ow = ow0 + tl.arange(0, BLOCK_OW)

    mask_oh = oh < OH
    mask_ow = ow < OW

    acc = tl.zeros((BLOCK_OH, BLOCK_OW), dtype=tl.float32)

    for ci in range(0, Cin):
        for kh in range(0, kH):
            in_y = oh * stride_h - pad_h + kh * dil_h  # [BLOCK_OH]
            # loop over kw
            if UNROLL_KW:
                for kk in tl.static_range(0, kW):
                    in_x = ow * stride_w - pad_w + kk * dil_w  # [BLOCK_OW]
                    base_x = n * (Cin * H * W) + ci * (H * W)
                    ptr_x = base_x + (in_y[:, None] * W) + (in_x[None, :])
                    valid = (in_y[:, None] >= 0) & (in_y[:, None] < H) & \
                            (in_x[None, :] >= 0) & (in_x[None, :] < W) & \
                            (mask_oh[:, None]) & (mask_ow[None, :])
                    x_val = tl.load(x_ptr + ptr_x, mask=valid, other=0.0).to(tl.float32)

                    w_idx = co * (Cin * kH * kW) + ci * (kH * kW) + kh * kW + kk
                    w_val = tl.load(w_ptr + w_idx).to(tl.float32)

                    acc += x_val * w_val
            else:
                for kk in range(0, kW):
                    in_x = ow * stride_w - pad_w + kk * dil_w
                    base_x = n * (Cin * H * W) + ci * (H * W)
                    ptr_x = base_x + (in_y[:, None] * W) + (in_x[None, :])
                    valid = (in_y[:, None] >= 0) & (in_y[:, None] < H) & \
                            (in_x[None, :] >= 0) & (in_x[None, :] < W) & \
                            (mask_oh[:, None]) & (mask_ow[None, :])
                    x_val = tl.load(x_ptr + ptr_x, mask=valid, other=0.0).to(tl.float32)

                    w_idx = co * (Cin * kH * kW) + ci * (kH * kW) + kh * kW + kk
                    w_val = tl.load(w_ptr + w_idx).to(tl.float32)

                    acc += x_val * w_val

    if HAS_BIAS:
        b_val = tl.load(b_ptr + co).to(tl.float32)
        acc += b_val

    sig = 1.0 / (1.0 + tl.exp(-acc))
    out = acc * sig
    out = out.to(tl.float16)

    base_y = n * (Cout * OH * OW) + co * (OH * OW)
    ptr_y = base_y + (oh[:, None] * OW) + (ow[None, :])
    valid_store = (mask_oh[:, None]) & (mask_ow[None, :])
    tl.store(y_ptr + ptr_y, out, mask=valid_store)


# Variant B: 1D tiling over ow, loop over oh inside program; good register balance
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 32}, num_warps=8, num_stages=3),
    ],
    key=['OH', 'OW', 'Cin', 'kH', 'kW'],
)
@triton.jit
def conv2d_silu_kernel_row(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N: tl.constexpr, Cin: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    Cout: tl.constexpr, kH: tl.constexpr, kW: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    stride_h: tl.constexpr, stride_w: tl.constexpr, pad_h: tl.constexpr, pad_w: tl.constexpr,
    dil_h: tl.constexpr, dil_w: tl.constexpr,
    BLOCK_OW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    UNROLL_KW: tl.constexpr,
):
    # Grid = (Cout*N*OH, ceil_div(OW, BLOCK_OW))
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    oh_blocks = (OH + BLOCK_OW - 1) // BLOCK_OW
    tmp = pid0 // oh_blocks
    oh = tmp // Cout
    nc = tmp % Cout
    n = nc // N
    co = nc % Cout

    ow_start = pid1 * BLOCK_OW
    ow = ow_start + tl.arange(0, BLOCK_OW)
    mask_ow = ow < OW

    acc = tl.zeros((BLOCK_OW,), dtype=tl.float32)

    for ci in range(0, Cin):
        for kh in range(0, kH):
            in_y = oh * stride_h - pad_h + kh * dil_h  # scalar
            if 0 <= in_y < H:
                for kk in tl.static_range(0, kW) if UNROLL_KW else range(0, kW):
                    in_x = ow * stride_w - pad_w + kk * dil_w
                    valid = (in_x >= 0) & (in_x < W) & mask_ow
                    base_x = n * (Cin * H * W) + ci * (H * W) + in_y * W
                    ptr_x = base_x + in_x
                    x_val = tl.load(x_ptr + ptr_x, mask=valid, other=0.0).to(tl.float32)

                    w_idx = co * (Cin * kH * kW) + ci * (kH * kW) + kh * kW + kk
                    w_val = tl.load(w_ptr + w_idx).to(tl.float32)

                    acc += x_val * w_val
            else:
                # entire row out of bounds: skip
                pass

    if HAS_BIAS:
        b_val = tl.load(b_ptr + co).to(tl.float32)
        acc += b_val

    sig = 1.0 / (1.0 + tl.exp(-acc))
    out = acc * sig
    out = out.to(tl.float16)

    base_y = n * (Cout * OH * OW) + co * (OH * OW) + oh * OW + ow
    tl.store(y_ptr + base_y, out, mask=mask_ow)


def _conv2d_silu_triton(x: torch.Tensor,
                        w: torch.Tensor,
                        b: torch.Tensor | None,
                        stride: int | tuple[int, int],
                        padding: int | tuple[int, int],
                        dilation: int | tuple[int, int]) -> torch.Tensor:
    """
    Launch an optimized Triton conv2d + bias + SiLU kernel.
    Picks between 2D-tiled and row-tiled kernel with autotuning.
    """
    assert x.is_cuda and w.is_cuda, "Triton kernel requires CUDA tensors"
    assert x.dtype == torch.float16 and w.dtype == torch.float16, "Expected float16 tensors"

    # Normalize params to ints (symmetric)
    if isinstance(stride, tuple):
        assert stride[0] == stride[1], "Kernel assumes symmetric stride"
        stride_h = stride_w = int(stride[0])
    else:
        stride_h = stride_w = int(stride)

    if isinstance(padding, tuple):
        assert padding[0] == padding[1], "Kernel assumes symmetric padding"
        pad_h = pad_w = int(padding[0])
    else:
        pad_h = pad_w = int(padding) if padding is not None else 0

    if isinstance(dilation, tuple):
        assert dilation[0] == dilation[1], "Kernel assumes symmetric dilation"
        dil_h = dil_w = int(dilation[0])
    else:
        dil_h = dil_w = int(dilation)

    N, Cin, H, W = x.shape
    Cout, Cin_w, kH, kW = w.shape
    assert Cin == Cin_w, f"In channels mismatch: {Cin} vs {Cin_w}"
    assert kH == kW, "Kernel assumes square kernels"

    # Output shape
    OH = math.floor((H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1)
    OW = math.floor((W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1)

    # Allocate output
    y = torch.empty((N, Cout, OH, OW), device=x.device, dtype=x.dtype)

    # Choose kernel variant and grid
    use_2d = True  # default
    unroll_kw = (kW <= 7) and (kH <= 7)

    if use_2d:
        oh_blocks = triton.cdiv(OH, 32)  # rough; actual chosen by autotune
        ow_blocks = triton.cdiv(OW, 64)
        grid = (Cout * N, oh_blocks, ow_blocks)
        conv2d_silu_kernel_2d[grid](
            x, w, b if b is not None else w, y,
            N, Cin, H, W, Cout, kH, kW, OH, OW,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            HAS_BIAS=(b is not None),
            UNROLL_KW=unroll_kw,
        )
    else:
        # row kernel: grid = (Cout*N*OH, ceil_div(OW, BLOCK_OW))
        # block count approx for launch
        grid = (Cout * N * OH, triton.cdiv(OW, 64))
        conv2d_silu_kernel_row[grid](
            x, w, b if b is not None else w, y,
            N, Cin, H, W, Cout, kH, kW, OH, OW,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            HAS_BIAS=(b is not None),
            UNROLL_KW=unroll_kw,
        )

    return y


class ModelNew(nn.Module):
    default_act = nn.SiLU()

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int | tuple[int, int] = 1,
        p=None,
        g: int = 1,
        d: int = 1,
        act=True,
    ):
        super().__init__()
        # Same structure as original Model
        self.conv = Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self._is_fused = False

        # Symmetry assumptions for kernel
        if isinstance(s, tuple):
            assert s[0] == s[1], "ModelNew kernel assumes symmetric stride"
            self._sym_stride = int(s[0])
        else:
            self._sym_stride = int(s)

        if isinstance(p, tuple):
            assert p[0] == p[1], "ModelNew kernel assumes symmetric padding"
            self._sym_pad = int(p[0])
        else:
            self._sym_pad = int(p) if p is not None else None

        if isinstance(d, tuple):
            assert d[0] == d[1], "ModelNew kernel assumes symmetric dilation"
            self._sym_dil = int(d[0])
        else:
            self._sym_dil = int(d)

    def _can_use_triton(self, x: torch.Tensor) -> bool:
        if not _HAS_TRITON:
            return False
        if not x.is_cuda:
            return False
        # groups must be 1
        if self.conv.groups != 1:
            return False
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CPU or no-triton: fallback to PyTorch
        if not self._can_use_triton(x):
            y = F.conv2d(x, self.conv.weight, self.conv.bias, stride=self.conv.stride,
                         padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups)
            y = self.bn(y)
            return self.act(y)

        # Determine mode
        if self._is_fused:
            # Use fused conv+bias+SiLU kernel
            w = self.conv.weight
            b = self.conv.bias
            if b is None:
                b = torch.zeros(w.shape[0], device=w.device, dtype=w.dtype)

            # Ensure float16
            if w.dtype != torch.float16:
                w = w.to(torch.float16)
            if b.dtype != torch.float16:
                b = b.to(torch.float16)
            x_fp16 = x if x.dtype == torch.float16 else x.to(torch.float16)

            y = _conv2d_silu_triton(
                x_fp16, w, b,
                stride=self._sym_stride,
                padding=self._sym_pad if self._sym_pad is not None else self.conv.padding,
                dilation=self._sym_dil,
            )
            return y
        else:
            # Unfused: conv -> BN -> act
            y = F.conv2d(x, self.conv.weight, self.conv.bias, stride=self.conv.stride,
                         padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups)
            y = self.bn(y)
            return self.act(y)

    @torch.no_grad()
    def fuse(self):
        if self._is_fused:
            return self
        # Fuse BN into conv weights/bias using running stats
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        weight = self.bn.weight
        bias = self.bn.bias
        eps = self.bn.eps

        Cout = self.conv.weight.shape[0]
        inv_std = torch.rsqrt(running_var + eps)  # 1/sqrt(var+eps)
        scale = weight * inv_std  # [Cout]
        shift = bias - running_mean * scale    # [Cout]

        w = self.conv.weight  # [Cout, Cin, kH, kW]
        w_new = w * scale.view(Cout, 1, 1, 1)
        self.conv.weight.data.copy_(w_new)

        if self.conv.bias is None:
            self.conv.bias = nn.Parameter(torch.empty(Cout, device=w.device, dtype=w.dtype))
        self.conv.bias.data = shift

        self._is_fused = True
        delattr(self, "bn")
        return self


# Compatibility helpers
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


class Conv2d(nn.Module):
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

YOLOConv = ModelNew
