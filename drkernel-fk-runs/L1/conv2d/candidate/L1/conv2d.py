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
def conv2d_nchw_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    # sizes (constexpr for unrolling / compile-time)
    C_in: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    K_h: tl.constexpr, K_w: tl.constexpr,
    C_out: tl.constexpr, G: tl.constexpr,
    # conv params
    stride_h: tl.constexpr, stride_w: tl.constexpr,
    pad_h: tl.constexpr, pad_w: tl.constexpr,
    dil_h: tl.constexpr, dil_w: tl.constexpr,
    # strides in elements
    x_stride_n: tl.constexpr, x_stride_c: tl.constexpr, x_stride_h: tl.constexpr, x_stride_w: tl.constexpr,
    w_stride_oc: tl.constexpr, w_stride_ic: tl.constexpr, w_stride_kh: tl.constexpr, w_stride_kw: tl.constexpr,
    y_stride_n: tl.constexpr, y_stride_c: tl.constexpr, y_stride_h: tl.constexpr, y_stride_w: tl.constexpr,
    # tiling
    BLOCK_OC: tl.constexpr, BLOCK_OH: tl.constexpr, BLOCK_OW: tl.constexpr,
    # output sizes
    H_out: tl.constexpr, W_out: tl.constexpr,
):
    # Program IDs
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)  # spatial block id over H_out * num_ow_blocks

    # Tile indices
    oc_idxs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    mask_oc = oc_idxs < C_out

    # Number of ow blocks along width
    num_ow_blocks = (W_out + BLOCK_OW - 1) // BLOCK_OW

    # Decode (oh, ow_block) from pid_sp
    oh = pid_sp // num_ow_blocks
    ow_block = pid_sp % num_ow_blocks
    ow_start = ow_block * BLOCK_OW
    ow_idxs = ow_start + tl.arange(0, BLOCK_OW)
    mask_ow = ow_idxs < W_out

    oh_start = oh
    oh_idxs = oh_start + tl.arange(0, BLOCK_OH)
    mask_oh = oh_idxs < H_out

    # Spatial mask [OH, OW]
    mask_s = mask_oh[:, None] & mask_ow[None, :]  # valid coordinates in output

    # Accumulator [OC, OH, OW]
    acc = tl.zeros((BLOCK_OC, BLOCK_OH, BLOCK_OW), dtype=tl.float32)

    # Group and channel info
    Cin_g = C_in // G

    # Loop over groups (typically 1)
    for g in range(0, G):
        c_in_base = g * Cin_g

        # Reduction over kernel and input channels, unrolled
        for kh in tl.static_range(0, K_h):
            for kw in tl.static_range(0, K_w):
                # Compute input coordinates for the tile (output -> input)
                ohv = oh_idxs[:, None]  # [OH,1]
                owv = ow_idxs[None, :]  # [1,OW]
                ih = ohv * stride_h - pad_h + kh * dil_h  # [OH,1]
                iw = owv * stride_w - pad_w + kw * dil_w  # [1,OW]

                # Broadcast to [OH,OW]; bounds relative to input H,W
                in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & mask_s

                # Loop over input channels in the group, unrolled
                for ic_rel in tl.static_range(0, Cin_g):
                    ic = c_in_base + ic_rel

                    # x load: shape [OH,OW]
                    x_offsets = (
                        pid_n * x_stride_n
                        + ic * x_stride_c
                        + ih * x_stride_h
                        + iw * x_stride_w
                    )
                    x_val = tl.load(x_ptr + x_offsets, mask=in_bounds, other=0.0)
                    x_val_f32 = x_val.to(tl.float32)  # [OH,OW]

                    # w load: shape [OC]
                    w_offsets = (
                        oc_idxs * w_stride_oc
                        + ic * w_stride_ic
                        + kh * w_stride_kh
                        + kw * w_stride_kw
                    )
                    w_val = tl.load(w_ptr + w_offsets, mask=mask_oc, other=0.0)
                    w_val_f32 = w_val.to(tl.float32)  # [OC]

                    # FMA: broadcast w along spatial dims
                    prod = w_val_f32[:, None, None] * x_val_f32[None, :, :]
                    acc += prod

    # Bias add
    if b_ptr != 0:
        b_val = tl.load(b_ptr + oc_idxs, mask=mask_oc, other=0.0).to(tl.float32)  # [OC]
        acc += b_val[:, None, None]

    # Store result: y[n, oc, oh, ow]
    for oh_rel in range(0, BLOCK_OH):
        cur_oh = oh_start + oh_rel
        for ow_rel in range(0, BLOCK_OW):
            cur_ow = ow_start + ow_rel
            for i in range(0, BLOCK_OC):
                oc = oc_idxs[i]
                if not mask_oc[i]:
                    continue
                y_offsets = (
                    pid_n * y_stride_n
                    + oc * y_stride_c
                    + cur_oh * y_stride_h
                    + cur_ow * y_stride_w
                )
                val = acc[i, oh_rel, ow_rel]
                tl.store(y_ptr + y_offsets, val)


class ModelNew(nn.Module):
    """Triton-optimized 2D convolution module with same API as Model.

    Uses a Triton kernel on CUDA when safe (no autograd), falls back to torch.nn.functional.conv2d otherwise.
    """
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

        kH, kW = kernel_size
        self.kernel_size = (kH, kW)

        C_in = in_channels
        C_out = out_channels
        assert C_in % groups == 0, f"in_channels ({C_in}) must be divisible by groups ({groups})"
        assert C_out % groups == 0, f"out_channels ({C_out}) must be divisible by groups ({groups})"

        self.weight = nn.Parameter(torch.empty(C_out, C_in // groups, kH, kW))
        self.bias = nn.Parameter(torch.empty(C_out)) if bias else None

        # Initialize like torch defaults
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _output_shape(self, x: torch.Tensor) -> tuple:
        N, C_in, H, W = x.shape
        kH, kW = self.kernel_size
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        dil_h, dil_w = self.dilation

        H_out = math.floor((H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1)
        W_out = math.floor((W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1)
        return (N, self.weight.shape[0], H_out, W_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback conditions
        use_triton = (
            _HAS_TRITON and
            x.is_cuda and
            self.weight.is_cuda and
            (self.bias is None or self.bias.is_cuda) and
            not (x.requires_grad or self.weight.requires_grad or (self.bias is not None and self.bias.requires_grad))
        )
        if not use_triton:
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

        device = x.device
        dtype = x.dtype
        assert dtype in (torch.float16, torch.bfloat16, torch.float32), f"Unsupported dtype {dtype}"

        # Shapes
        N, C_in, H, W = x.shape
        kH, kW = self.kernel_size
        C_out = self.weight.shape[0]
        G = self.groups
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        dil_h, dil_w = self.dilation

        # Contiguous
        x_c = x.contiguous()
        w_c = self.weight.contiguous()
        b_c = self.bias.contiguous() if self.bias is not None else None

        # Output
        out_shape = self._output_shape(x)
        y = torch.empty(out_shape, device=device, dtype=dtype)
        H_out, W_out = out_shape[2], out_shape[3]

        # Strides
        x_sn, x_sc, x_sh, x_sw = x_c.stride()        # [N,C,H,W]
        w_so, w_si, w_skh, w_skw = w_c.stride()      # [O,I,KH,KW]
        y_sn, y_sc, y_sh, y_sw = y.stride()

        # Tiling (can be tuned)
        # Choose BLOCK_OW based on W_out
        if W_out >= 256:
            BLOCK_OW = 128
        elif W_out >= 128:
            BLOCK_OW = 128
        elif W_out >= 64:
            BLOCK_OW = 64
        else:
            BLOCK_OW = 32

        BLOCK_OH = 1
        BLOCK_OC = 64 if C_out >= 64 else 32

        # Grid
        num_ow_blocks = (W_out + BLOCK_OW - 1) // BLOCK_OW
        grid = (
            N,
            triton.cdiv(C_out, BLOCK_OC),
            H_out * num_ow_blocks,
        )

        # Bias pointer or 0
        b_addr = b_c.data_ptr() if b_c is not None else 0

        conv2d_nchw_kernel[grid](
            x_c, w_c, b_addr, y,
            # sizes (constexpr)
            C_in, H, W,
            kH, kW,
            C_out, G,
            # conv params
            stride_h, stride_w,
            pad_h, pad_w,
            dil_h, dil_w,
            # strides
            x_sn, x_sc, x_sh, x_sw,
            w_so, w_si, w_skh, w_skw,
            y_sn, y_sc, y_sh, y_sw,
            # tiling
            BLOCK_OC=BLOCK_OC, BLOCK_OH=BLOCK_OH, BLOCK_OW=BLOCK_OW,
            # output sizes
            H_out=H_out, W_out=W_out,
            num_warps=8,
            num_stages=3,
        )

        return y

Conv2d = ModelNew
