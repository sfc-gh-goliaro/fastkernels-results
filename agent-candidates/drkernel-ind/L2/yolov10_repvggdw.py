import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try import triton; if not available, we'll fallback to PyTorch
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


# -----------------------------
# Triton kernel: depthwise conv (N, C, H, W) -> (N, C, H_out, W_out)
# assumptions:
# - groups == C  (depthwise)
# - stride=1, dilation=1
# - no zero padding inside kernel; output size computed as H_out = H - kH + 1, W_out = W - kW + 1
# - weight shape [C, 1, kH, kW]  (Cin=1 per group)
# -----------------------------
if _TRITON_AVAILABLE:
    @triton.jit
    def conv_dw_fwd_nopad(
        x_ptr,                  # *T  [N, C, H, W]
        w_ptr,                  # *T    [C, 1, kH, kW]
        bias_ptr,               # *T    [C]
        out_ptr,                # *T    [N, C, H_out, W_out]
        # sizes
        N: tl.constexpr,
        C: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        kH: tl.constexpr,
        kW: tl.constexpr,
        H_out: tl.constexpr,
        W_out: tl.constexpr,
        # strides (in elements)
        x_sN: tl.constexpr, x_sC: tl.constexpr, x_sH: tl.constexpr, x_sW: tl.constexpr,
        w_sC: tl.constexpr, w_sCi: tl.constexpr, w_sKH: tl.constexpr, w_sKW: tl.constexpr,
        out_sN: tl.constexpr, out_sC: tl.constexpr, out_sH: tl.constexpr, out_sW: tl.constexpr,
        # tiling
        BLOCK_L: tl.constexpr,
    ):
        # program id: over (n, c)
        pid = tl.program_id(0)
        n = pid // C
        c = pid % C

        # flattened spatial length of output
        L_out = H_out * W_out

        # output base pointer for (n, c, 0, 0)
        out_base = n * out_sN + c * out_sC

        # accumulate in float32 for numeric stability
        acc = tl.zeros((BLOCK_L,), dtype=tl.float32)

        # loop over kernel height/width
        for kh in range(0, kH):
            for kw in range(0, kW):
                # weight load: w[c, 0, kh, kw]
                w_off = c * w_sC  + 0 * w_sCi  + kh * w_sKH  + kw * w_sKW
                w_val = tl.load(w_ptr + w_off).to(tl.float32)

                # input start (top-left) for this (kh, kw)
                ih0 = kh
                iw0 = kw

                # spatial base (vector) over L_out
                for l0 in range(0, L_out, BLOCK_L):
                    l = l0 + tl.arange(0, BLOCK_L)
                    mask = l < L_out

                    # convert l -> (oh, ow)
                    ow = l % W_out
                    oh = l // W_out

                    # map to input h,w
                    h = oh + ih0
                    w = ow + iw0

                    # input pointer offsets: x[n, c, h, w]
                    x_off = n * x_sN + c * x_sC + h * x_sH + w * x_sW
                    xv = tl.load(x_ptr + x_off, mask=mask, other=0.0).to(tl.float32)

                    # FMA accumulate
                    acc += w_val * xv

        # add bias
        b = tl.load(bias_ptr + c).to(tl.float32)
        acc = acc + b

        # store to out (cast to output dtype)
        for l0 in range(0, L_out, BLOCK_L):
            l = l0 + tl.arange(0, BLOCK_L)
            mask = l < L_out
            out_off = out_base + l
            tl.store(out_ptr + out_off, acc.to(tl.float16), mask=mask)


def _conv_bn_fused_weight_bias(conv: nn.Module, bn: nn.Module):
    """
    Fused Conv+BN parameters in eval mode:
    Returns (weight_fused, bias_fused) where weight has shape [C,1,kH,kW] and bias [C].
    Assumes:
    - conv.weight is [C,kH,kW]
    - bn is in eval: uses running_var, running_mean, weight, bias
    """
    assert not bn.training, "Conv+BN fusion requires BN in eval mode (self.training=False)."
    w = conv.weight  # [C, kH, kW]
    # reshape to [C,1,kH,kW]
    wv = w.view(w.shape[0], 1, w.shape[1], w.shape[2])
    # bn params
    rv = bn.running_var
    rm = bn.running_mean
    bw = bn.weight
    bb = bn.bias
    eps = bn.eps
    # scale per channel
    scale = bw / torch.sqrt(rv + eps)  # [C]
    # fused weight: W'[c] = W[c] * scale[c]
    wfv = wv * scale.view(-1, 1, 1, 1)
    # fused bias: b' = (W*rm*scale) sum over Cin + bb - rm*bn_weight[c]/sqrt(…)
    # Here Cin=1 per group:
    wrm = wv * rm.view(-1, 1, 1, 1)
    conv_b = conv.bias
    if conv_b is None:
        conv_b = torch.zeros(w.shape[0], device=w.device, dtype=w.dtype)
    term1 = (wrm * conv_b.view(-1, 1, 1, 1)).sum(dim=(1, 2, 3))  # [C]
    term2 = bb - rm * scale
    bf = term1 + term2  # [C]
    return wfv, bf


class ModelNew(nn.Module):
    def __init__(self, ed: int):
        super().__init__()
        self.conv = YOLOConv(ed, ed, 7, 1, 3, g=ed, act=False)  # k=7, pad=3
        self.conv1 = YOLOConv(ed, ed, 3, 1, 1, g=ed, act=False) # k=3, pad=1
        self._is_fused = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._is_fused:
            return F.silu(self.conv(x) + self.conv1(x))
        # Fused path: two Triton convs + sum + silu
        return self._fused_triton(x)

    @torch.no_grad()
    def fuse(self):
        if self._is_fused:
            return self

        # put to eval to use running stats
        self.eval()
        self.conv.eval()
        self.conv1.eval()

        # Fuse Conv+BN for each
        w0_3d, b0 = _conv_bn_fused_weight_bias(self.conv.conv, self.conv.bn)  # k=7
        w1_3d, b1 = _conv_bn_fused_weight_bias(self.conv1.conv, self.conv1.bn) # k=3

        # Views to [C, kH, kW]
        w0 = w0_3d.view(w0_3d.shape[0], w0_3d.shape[2], w0_3d.shape[3])
        w1 = w1_3d.view(w1_3d.shape[0], w1_3d.shape[2], w1_3d.shape[3])

        device = x.device
        dtype = x.dtype

        N, C, H, W = x.shape
        assert C == w0.shape[0] == w1.shape[0] == ed, f"Expected C==ed ({ed}), got {C}"

        # Kernel sizes
        kH0, kW0 = w0.shape[-2], w0.shape[-1]
        kH1, kW1 = w1.shape[-2], w1.shape[-1]
        assert kH0 == kH1 and kW0 == kW1, "Expect same kernel shapes for both convs"
        kH, kW = kH0, kW0

        # Output sizes (no padding in kernel; pad is external but output size is H - k + 1)
        H_out = H - kH + 1
        W_out = W - kW + 1
        if H_out <= 0 or W_out <= 0:
            raise ValueError(f"Invalid shapes: H={H}, W={W}, kH={kH}, kW={kW}")

        # Convert weights to [C,1,kH,kW]
        w0_c = w0.view(C, 1, kH, kW).contiguous().to(device=device, dtype=dtype)
        w1_c = w1.view(C, 1, kH, kW).contiguous().to(device=device, dtype=dtype)
        b0_c = b0.contiguous().to(device=device, dtype=dtype)
        b1_c = b1.contiguous().to(device=device, dtype=dtype)

        # Allocate outputs
        out0 = torch.empty((N, C, H_out, W_out), device=device, dtype=dtype)
        out1 = torch.empty((N, C, H_out, W_out), device=device, dtype=dtype)

        # Strides
        x_sN, x_sC, x_sH, x_sW = x.stride()
        w_sC, w_sCi, w_sKH, w_sKW = w0_c.stride()
        out_sN, out_sC, out_sH, out_sW = out0.stride()

        BLOCK_L = 128
        grid = (N * C,)

        # Kernel 0
        conv_dw_fwd_nopad[grid](
            x, w0_c, b0_c, out0,
            N, C, H, W,
            kH, kW,
            H_out, W_out,
            x_sN, x_sC, x_sH, x_sW,
            w_sC, w_sCi, w_sKH, w_sKW,
            out_sN, out_sC, out_sH, out_sW,
            BLOCK_L=BLOCK_L,
            num_warps=4,
            num_stages=2,
        )
        # Kernel 1
        conv_dw_fwd_nopad[grid](
            x, w1_c, b1_c, out1,
            N, C, H, W,
            kH, kW,
            H_out, W_out,
            x_sN, x_sC, x_sH, x_sW,
            w_sC, w_sCi, w_sKH, w_sKW,
            out_sN, out_sC, out_sH, out_sW,
            BLOCK_L=BLOCK_L,
            num_warps=4,
            num_stages=2,
        )

        # Sum and activate
        out = out0 + out1
        out = F.silu(out)

        self._is_fused = True
        return out

    def _fused_triton(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse()(x)


# YOLOConv definition
class YOLOConv(nn.Module):
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
        self.conv = Conv2d(c1, c2, k, s, self.autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self._is_fused = False

    @staticmethod
    def autopad(k: int | tuple[int, int], p=None, d: int = 1):
        # Symmetric padding: p = (k_dilated - 1) // 2
        if isinstance(k, tuple):
            return ((d * (k[0] - 1) + 1) - 1) // 2, ((d * (k[1] - 1) + 1) - 1) // 2
        return (d * (k - 1) + 1 - 1) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_fused:
            return self.act(self.conv(x))
        return self.act(self.bn(self.conv(x)))

    @torch.no_grad()
    def fuse(self):
        if self._is_fused:
            return self
        # Fused Conv+BN parameters in eval
        wfv, bf = _conv_bn_fused_weight_bias(self.conv, self.bn)
        self.conv.weight.data.copy_(wfv.view(self.conv.weight.shape))
        self.conv.bias = nn.Parameter(bf)
        delattr(self, "bn")
        self._is_fused = True
        return self


# Conv2d and BatchNorm2d as given
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

YOLORepVGGDW = ModelNew
