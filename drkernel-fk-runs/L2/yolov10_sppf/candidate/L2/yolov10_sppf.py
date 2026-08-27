import math
import torch
import torch.nn as nn

# Try to import triton; if unavailable, we'll fall back to PyTorch ops
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# -----------------------------
# Triton kernel: 1x1 Conv2d (NCHW)
# Y[n, co, h, w] = bias[co] + sum_{ci} X[n, ci, h, w] * W[co, ci]
# No padding, stride=1, kernel=1x1
# -----------------------------
if TRITON_AVAILABLE:
    @triton.jit
    def conv1x1_nchw_kernel(
        X, W, BIAS, Y,
        N: tl.constexpr, C_IN: tl.constexpr, H: tl.constexpr, WID: tl.constexpr, C_OUT: tl.constexpr,
        stride_xn: tl.constexpr, stride_xc: tl.constexpr, stride_xh: tl.constexpr, stride_xw: tl.constexpr,
        stride_wco: tl.constexpr, stride_wci: tl.constexpr,
        stride_yn: tl.constexpr, stride_yc: tl.constexpr, stride_yh: tl.constexpr, stride_yw: tl.constexpr,
        BLOCK_CO: tl.constexpr, BLOCK_CI: tl.constexpr,
    ):
        # program ids
        pid_p = tl.program_id(0)  # over N*H*W
        pid_co = tl.program_id(1)  # over C_OUT blocks

        # decode pixel index
        HW = H * WID
        n = pid_p // HW
        rem = pid_p % HW
        h = rem // WID
        w = rem % WID

        # vector of output channel indices this program computes
        co_start = pid_co * BLOCK_CO
        co = co_start + tl.arange(0, BLOCK_CO)
        mask_co = co < C_OUT

        # accumulator in fp32
        acc = tl.zeros([BLOCK_CO], dtype=tl.float32)

        # loop over input channels in BLOCK_CI chunks
        for ci_start in range(0, C_IN, BLOCK_CI):
            ci = ci_start + tl.arange(0, BLOCK_CI)
            mask_ci = ci < C_IN

            # load X[n, ci, h, w] -> shape [BLOCK_CI]
            x_ptr = X + n * stride_xn + ci * stride_xc + h * stride_xh + w * stride_xw
            x = tl.load(x_ptr, mask=mask_ci, other=0.0).to(tl.float32)  # [BLOCK_CI]

            # load W[co, ci] -> shape [BLOCK_CO, BLOCK_CI]
            w_ptr = W + (co[:, None] * stride_wco) + (ci[None, :] * stride_wci)
            wv = tl.load(w_ptr, mask=(mask_co[:, None] & mask_ci[None, :]), other=0.0).to(tl.float32)  # [BLOCK_CO, BLOCK_CI]

            # acc += sum over ci of wv[co, ci] * x[ci]
            acc += tl.sum(wv * x[None, :], axis=1)

        # add bias
        b = tl.load(BIAS + co, mask=mask_co, other=0.0).to(tl.float32)
        acc = acc + b

        # store to Y[n, co, h, w]
        y_ptr = Y + n * stride_yn + co * stride_yc + h * stride_yh + w * stride_yw
        tl.store(y_ptr, acc, mask=mask_co)


    def triton_conv1x1_nchw(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        """
        Launches the Triton 1x1 conv kernel.
        x: [N, C_in, H, W] (contiguous NCHW)
        w: [C_out, C_in] (contiguous)
        bias: [C_out] or None
        returns y: [N, C_out, H, W]
        """
        assert x.is_cuda, "Triton kernel requires CUDA tensor"
        assert x.dtype in (torch.float16, torch.bfloat16, torch.float32), "Supported dtypes: fp16/bf16/fp32"
        assert w.is_cuda and w.dtype == x.dtype, "Weight must be on CUDA and same dtype as input"

        N, C_in, H, W = x.shape
        C_out = w.shape[0]
        assert w.shape[1] == C_in, f"Weight shape mismatch: expected ({C_out}, {C_in}), got {w.shape}"
        if bias is not None:
            assert bias.is_cuda and bias.dtype == x.dtype and bias.shape[0] == C_out

        # Make sure tensors are contiguous
        x_c = x.contiguous()
        w_c = w.contiguous()
        if bias is not None:
            bias_c = bias.contiguous()
        else:
            # create a zero bias to simplify kernel
            bias_c = torch.zeros(C_out, device=x.device, dtype=x.dtype)

        # Output in fp32 for accumulation; cast later
        y = torch.empty((N, C_out, H, W), device=x.device, dtype=torch.float32)

        # Strides (in elements)
        sxn, sxc, sxh, sxw = x_c.stride()
        swco, swci = w_c.stride()
        syn, syc, syh, syw = y.stride()

        # Block sizes (heuristics); can be autotuned for more speed
        BLOCK_CO = 64
        BLOCK_CI = 128

        grid = (N * H * W, triton.cdiv(C_out, BLOCK_CO))

        conv1x1_nchw_kernel[grid](
            x_c, w_c, bias_c, y,
            N, C_in, H, W, C_out,
            sxn, sxc, sxh, sxw,
            swco, swci,
            syn, syc, syh, syw,
            BLOCK_CO=BLOCK_CO, BLOCK_CI=BLOCK_CI,
            num_warps=4, num_stages=2,
        )

        # Cast back to input dtype
        if x.dtype == torch.float32:
            return y
        else:
            return y.to(dtype=x.dtype)


class TritonConv2d(nn.Module):
    """
    A thin wrapper that uses a Triton 1x1 conv when possible, otherwise falls back to torch.nn.functional.conv2d.
    Only supports 1x1, stride=1, padding=0, dilation=1, groups=1.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1,
                 stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True):
        super().__init__()
        assert kernel_size == 1 and stride == 1 and padding == 0 and dilation == 1 and groups == 1, \
            "TritonConv2d currently supports only 1x1, stride=1, padding=0, dilation=1, groups=1"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        # init like torch defaults
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback if CPU or Triton not available
        if (not x.is_cuda) or (not TRITON_AVAILABLE):
            return torch.nn.functional.conv2d(x, self.weight.unsqueeze(-1).unsqueeze(-1), self.bias,
                                              stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        # Use Triton
        return triton_conv1x1_nchw(x, self.weight, self.bias)


# -----------------------------
# Replace YOLOConv with TritonConv2d where appropriate
# -----------------------------
class YOLOConvT(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p=0, g: int = 1, d: int = 1, act=True):
        super().__init__()
        # Only 1x1 is used in this model; keep general signature but enforce k=1
        assert k == 1, f"YOLOConvT currently supports only 1x1; got k={k}"
        self.conv = TritonConv2d(c1, c2, kernel_size=1, stride=s, padding=p, dilation=d, groups=g, bias=True)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


# -----------------------------
# ModelNew: Triton-optimized version (top-level definition)
# -----------------------------
class ModelNew(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = YOLOConvT(c1, c_, 1, 1)           # 1x1 conv -> 128
        self.cv2 = YOLOConvT(c_ * 4, c2, 1, 1)        # 1x1 conv -> 256
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


# -----------------------------
# Original helpers remain valid
# -----------------------------
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
        return torch.nn.functional.max_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            ceil_mode=self.ceil_mode,
        )

YOLOSPPF = ModelNew
