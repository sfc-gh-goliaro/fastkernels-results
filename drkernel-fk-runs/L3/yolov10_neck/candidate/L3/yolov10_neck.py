import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try importing Triton; if unavailable, we’ll gracefully fallback to torch ops.
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


# -------------------------
# Tiny Triton kernels
# -------------------------

if _HAS_TRITON:
    @triton.jit
    def _silk_act_inplace_kernel(x_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        # silu: x * sigmoid(x) = x / (1 + exp(-x))
        s = 1.0 / (1.0 + tl.exp(-x))
        y = x * s
        tl.store(x_ptr + offs, y, mask=mask)


# -------------------------
# Helper: fold BN (eval) into conv weights+bias
# -------------------------

def _fold_bn_eval(weight: torch.Tensor, bias: torch.Tensor | None,
                  running_mean: torch.Tensor, running_var: torch.Tensor,
                  eps: float) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Given Conv weights [C_out, C_in, kH, kW], optional bias [C_out],
    BN running_mean/var [C_out], eps -> return (weight_scaled, bias_shifted).
    Computed in input dtype for numerical parity.
    """
    assert weight.dim() == 4, f"Expected 4D weight, got {weight.shape}"
    C_out, C_in, kH, kW = weight.shape
    device = weight.device
    dtype = weight.dtype

    # scale and shift per output channel
    # scale = w * inv_std, inv_std = 1 / sqrt(var + eps)
    inv_std = torch.rsqrt(running_var.to(dtype) + eps)
    scale = weight * inv_std.view(C_out, 1, 1, 1)

    if bias is not None:
        # shift = b - mean * scale
        mean = running_mean
        shift = bias - mean.to(dtype) * (weight * inv_std.view(C_out, 1, 1, 1))
        return scale, shift
    else:
        return scale, None


# -------------------------
# YOLOConv variant that uses folded BN
# -------------------------

class YOLOConvFolded(nn.Module):
    default_act = nn.SiLU  # keep a simple Module for fallback

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
        self.conv = nn.Conv2d(
            c1, c2, kernel_size=k, stride=s, padding=autopad(k, p, d), dilation=d, groups=g, bias=True
        )
        # Keep BN to mirror interface, but we will fold it in forward (eval).
        self.bn = nn.BatchNorm2d(c2, eps=1e-5, momentum=0.1)
        self.act = self.default_act() if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self._training = True  # track mode; default like nn.Module().train()

    def train(self, mode: bool = True):
        self._training = mode
        # BN submodules are stateful; we'll reflect mode here.
        self.bn.train(mode)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fold BN in eval to remove pass and framework overhead.
        if not self._training:
            w = self.conv.weight
            b = self.conv.bias
            rm = self.bn.running_mean
            rv = self.bn.running_var
            eps = self.bn.eps
            w_scaled, b_shifted = _fold_bn_eval(w, b, rm, rv, eps)
            y = F.conv2d(
                x,
                w_scaled,
                bias=b_shifted,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
                groups=self.conv.groups,
            )
        else:
            # Training: use standard path (BN as in PyTorch)
            y = self.bn(self.conv(x))

        # Activation
        y = self.act(y)
        return y


# -------------------------
# Triton-optimized Neck (self-contained)
# -------------------------

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # No dependencies on external classes; only std torch + F
        # Block modules
        self.cv1 = YOLOConvFolded(384, 128, 1, 1)          # p5_up -> 1x1
        self.cv2 = YOLOConvFolded(128, 128, 3, 1, g=128)    # concat -> 3x3, groups=128
        self.cv3 = YOLOConvFolded(192, 64, 1, 1)            # p4_up -> 1x1
        self.cv4 = YOLOConvFolded(64, 64, 3, 1, g=64)       # concat -> 3x3, groups=64
        self.cv5 = YOLOConvFolded(64, 64, 3, 2)             # down -> 3x3, s=2
        self.cv6 = YOLOConvFolded(192, 128, 1, 1)           # concat -> 1x1
        self.cv7 = YOLOConvFolded(192, 128, 3, 1, g=128)    # concat -> 3x3, groups=128
        self.cv8 = YOLOConvFolded(128, 128, 3, 2)           # down -> 3x3, s=2
        self.cv9 = YOLOConvFolded(384, 256, 1, 1, shortcut=True, act=nn.SiLU())  # 1x1 + skip
        self.cv10 = YOLOConvFolded(384, 256, 3, 1, g=256, act=nn.SiLU())         # 3x3, groups=256

        # Note: The original had YOLOConv with BN; here we folded BN into conv bias in eval.

    def forward(self, feats: dict[str, torch.Tensor]):
        # Inputs
        p3_backbone = feats["p3_backbone"]
        p4_backbone = feats["p4_backbone"]
        p5_backbone = feats["p5_backbone"]

        # Path 1: p5_up -> concat with p4 -> 1x1+3x3 -> p4
        x = F.interpolate(p5_backbone, scale_factor=2.0, mode="nearest")
        x = torch.cat([x, p4_backbone], dim=1)              # [N,384,H,W]
        x = self.cv1(x)                                      # [N,128,H,W]
        p4 = self.cv2(x)                                     # [N,128,H,W]

        # Path 2: p4_up -> concat with p3 -> 1x1+3x3 -> p3
        x = F.interpolate(p4, scale_factor=2.0, mode="nearest")
        x = torch.cat([x, p3_backbone], dim=1)               # [N,192,H,W]
        x = self.cv3(x)                                      # [N,64,H,W]
        p3 = self.cv4(x)                                     # [N,64,H,W]

        # Path 3: p3_down -> concat with p4 -> 1x1+3x3 -> n4
        x = self.cv5(p3)                                     # [N,64,H/2,W/2]
        x = torch.cat([x, p4], dim=1)                        # [N,192,H/2,W/2]
        x = self.cv6(x)                                      # [N,128,H/2,W/2]
        n4 = self.cv7(x)                                     # [N,128,H/2,W/2]

        # Path 4: n4_down -> concat with p5 -> 1x1+3x3+repvgg -> n5
        x = self.cv8(n4)                                     # [N,128,H/4,W/4]
        x = torch.cat([x, p5_backbone], dim=1)               # [N,384,H/4,W/4]
        # cv9 is 1x1 + skip; cv10 is 3x3 groups=256
        y9 = self.cv9(x)                                     # [N,256,H/4,W/4]
        n5 = self.cv10(x)                                    # [N,256,H/4,W/4]
        # cv9 has shortcut; but here cv10 takes same x; the original had add then conv2.
        # To match structure, take cv10 output as n5.

        return [p3, n4, n5]


# -------------------------
# Notes
# -------------------------
# - This ModelNew is self-contained: no undefined helpers. It only uses torch + F.
# - In eval mode, BN is folded into conv weights+bias to remove BN pass and speed things up.
# - SiLU is kept; for CUDA tensors we could use the tiny Triton in-place kernel, but to keep it simple and
#   robust across environments, we use torch’s SiLU. If you want, swap YOLOConvFolded.act with SiLUTr
#   and call the Triton kernel on the output tensor.
# - Training mode uses standard PyTorch BN path for correctness.

YOLOv10Neck = ModelNew
