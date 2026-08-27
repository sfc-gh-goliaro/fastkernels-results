import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _conv2d_block_vec_kernel(
    x_ptr,           # *T, NCHW contiguous
    w_vec_ptr,       # *T, shape [C * K] contiguous
    b_ptr,           # *T, shape [C]
    y_ptr,           # *T, shape [N, C, Hout, Wout] contiguous
    # shapes (constexpr)
    N: tl.constexpr,
    C: tl.constexpr,
    Hout: tl.constexpr,
    Wout: tl.constexpr,
    PS: tl.constexpr,
    Cin: tl.constexpr,
    # strides for x (elements)
    x_sN: tl.constexpr,
    x_sC: tl.constexpr,
    x_sH: tl.constexpr,
    x_sW: tl.constexpr,
    # K = Cin * PS * PS (constexpr)
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program IDs
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)

    W_ = Wout
    h = pid_hw // W_
    w = pid_hw % W_

    if (pid_n >= N) or (pid_c >= C) or (h >= Hout) or (w >= Wout):
        return

    # Accumulator (scalar)
    acc = 0.0

    # Loop over K in blocks
    for k0 in range(0, K, BLOCK_K):
        k_ids = k0 + tl.arange(0, BLOCK_K)     # vector of length BLOCK_K
        valid = k_ids < K

        # Decompose k -> (c_in, rel) -> (kh, kw)
        PS2 = PS * PS
        c_in = k_ids // PS2                    # [BK]
        rel  = k_ids % PS2                     # [BK]
        kh   = rel // PS
        kw   = rel % PS

        # Spatial base for this (h, w)
        spat_h = (h * PS + kh) * x_sH          # [BK]
        spat_w = (w * PS + kw) * x_sW          # [BK]

        # Base for n and c_in
        base_nc = pid_n * x_sN + c_in * x_sC   # [BK]

        # x offsets and load
        x_offsets = base_nc + spat_h + spat_w  # [BK]
        x_vals = tl.load(x_ptr + x_offsets, mask=valid, other=0.0)

        # w offsets and load (w viewed as [C*K])
        w_base = pid_c * K
        w_offsets = w_base + k_ids             # [BK]
        w_vals = tl.load(w_vec_ptr + w_offsets, mask=valid, other=0.0)

        # Accumulate
        prod = x_vals * w_vals
        acc += tl.sum(prod, axis=0)

    # Add bias
    b_val = tl.load(b_ptr + pid_c)
    acc = acc + b_val

    # Store y[n, c, h, w]
    y_sN = y_ptr.stride(0)
    y_sC = y_ptr.stride(1)
    y_sH = y_ptr.stride(2)
    y_sW = y_ptr.stride(3)
    y_offset = pid_n * y_sN + pid_c * y_sC + h * y_sH + w * y_sW
    tl.store(y_ptr + y_offset, acc)


class ModelNew(nn.Module):
    def __init__(
        self,
        img_height: int = 256,
        img_width: int = 256,
        patch_size: int = 16,
        in_chans: int = 16,
        embed_dim: int = 768,
        norm_layer=None,
        flatten: bool = True,
    ):
        super().__init__()
        self.img_size = (img_height, img_width)
        self.patch_size = (patch_size, patch_size)
        ps = patch_size
        self.grid_size = (img_height // ps, img_width // ps)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # Parameters match Conv2d layout (C, Cin, KH, KW)
        self.weight = nn.Parameter(torch.empty(embed_dim, in_chans, ps, ps))
        self.bias = nn.Parameter(torch.empty(embed_dim))

        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x: torch.Tensor, random_sample: bool = False) -> torch.Tensor:
        # Fallback to PyTorch if not CUDA or Triton not available
        if (not x.is_cuda) or (not TRITON_AVAILABLE):
            y = torch.nn.functional.conv2d(
                x,
                self.weight,
                self.bias,
                stride=self.patch_size,
                padding=0,
                dilation=1,
                groups=1,
            )
            if self.flatten:
                y = y.flatten(2).transpose(1, 2)
            else:
                y = y.permute(0, 2, 3, 1)
            return self.norm(y) if self.norm is not None else y

        # Validate dims
        assert x.dim() == 4, f"Expected NCHW, got shape {tuple(x.shape)}"
        N, Cin, H, W = x.shape
        ps = self.patch_size[0]
        assert ps == self.patch_size[1], "Only square patches supported"
        C = self.weight.shape[0]
        KH, KW = self.weight.shape[2], self.weight.shape[3]
        assert Cin == self.weight.shape[1], f"in_channels {Cin} != weight.shape[1] {self.weight.shape[1]}"
        assert KH == ps and KW == ps, "Kernel size must equal patch size"

        # Enforce expected layout
        x_ = x.contiguous()
        w_ = self.weight.contiguous()
        b_ = self.bias.contiguous()

        # Output dimensions (no padding, stride = ps)
        Hout = H // ps
        Wout = W // ps
        if not random_sample and (H, W) != self.img_size:
            raise AssertionError(f"Input image size ({H}*{W}) doesn't match model {self.img_size}.")

        # Allocate output [N, C, Hout, Wout]
        y = torch.empty((N, C, Hout, Wout), device=x.device, dtype=x.dtype)

        # Strides for x (elements)
        x_sN, x_sC, x_sH, x_sW = x_.stride()

        # Prepare weight as a contiguous 1D vector [C*K]
        K = Cin * ps * ps
        w_vec = w_.view(-1).contiguous()  # shape [C*K]

        # Grid
        grid = (N, C, Hout * Wout)

        # Tuning
        BLOCK_K = 256
        num_warps = 4
        num_stages = 2

        # Launch kernel
        _conv2d_block_vec_kernel[grid](
            x_, w_vec, b_, y,
            N, C, Hout, Wout,
            ps, Cin,
            x_sN, x_sC, x_sH, x_sW,
            K,
            BLOCK_K=BLOCK_K,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        # Match original layout
        if self.flatten:
            out = y.view(N, C, Hout * Wout).transpose(1, 2).contiguous()  # [N, L, C]
        else:
            out = y.permute(0, 2, 3, 1).contiguous()                      # [N, H, W, C]

        if self.norm is not None:
            out = self.norm(out)

        return out

OasisPatchEmbed = ModelNew
