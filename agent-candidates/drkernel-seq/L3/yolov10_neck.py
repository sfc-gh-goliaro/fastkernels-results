from __future__ import annotations

import math
import torch
import torch.nn as nn

import triton
import triton.language as tl

# Keep these imports as in the original to respect the frozen lower-level winners
from ..L1.interpolate import Interpolate
from ..L2.yolov10_c2f import YOLOC2f, YOLOC2fCIB
from ..L2.yolov10_concat import YOLOConcat
from ..L2.yolov10_conv import YOLOConv
from ..L2.yolov10_scdown import YOLOSCDown


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_H": 128, "BLOCK_W": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 128}, num_warps=8, num_stages=2),
    ],
    key=["H_out", "W_out"],
)
@triton.jit
def nearest_upsample_2d_kernel(
    x_ptr,                      # *T
    out_ptr,                    # *T
    B: tl.constexpr,
    C: tl.constexpr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    scale: tl.constexpr,        # float scale_factor
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # program ids
    bc = tl.program_id(0)       # ranges over B*C
    y_block = tl.program_id(1)  # blocks over H_out
    x_block = tl.program_id(2)  # blocks over W_out

    # derive b and c from bc
    c = bc % C
    b = bc // C

    # create coordinates within the tile
    offs_y = y_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_x = x_block * BLOCK_W + tl.arange(0, BLOCK_W)

    Y = offs_y[:, None]
    X = offs_x[None, :]

    # mask for bounds
    mask = (Y < H_out) & (X < W_out)

    # General path: float division + floor for nearest with scale
    fy = Y.to(tl.float32)
    fx = X.to(tl.float32)
    in_y = tl.floor(fy / scale).to(tl.int32)
    in_x = tl.floor(fx / scale).to(tl.int32)

    # base linear index for (b, c) in input and output (64-bit to be safe)
    bc_in_base = ((b * C + c) * H_in) * W_in
    bc_out_base = ((b * C + c) * H_out) * W_out

    # flat indices (64-bit)
    in_idx = bc_in_base + (in_y * W_in) + in_x
    out_idx = bc_out_base + (Y * W_out) + X

    # load and store
    vals = tl.load(x_ptr + in_idx, mask=mask, other=0)
    tl.store(out_ptr + out_idx, vals, mask=mask)


def _triton_upsample_nearest(x: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """
    Triton implementation of nearest-neighbor upsample (2D) for NCHW tensors.
    Assumes:
      - x is CUDA and contiguous
      - mode='nearest', align_corners=False
    """
    assert x.is_cuda, "Triton kernel requires CUDA tensor"
    assert x.dim() == 4, f"Expected 4D NCHW, got shape {tuple(x.shape)}"
    if not x.is_contiguous():
        x = x.contiguous()

    B, C, H_in, W_in = x.shape
    # PyTorch's output size when scale_factor is provided: int(H*sf), int(W*sf)
    H_out = int(math.floor(H_in * float(scale_factor)))
    W_out = int(math.floor(W_in * float(scale_factor)))

    out = torch.empty((B, C, H_out, W_out), device=x.device, dtype=x.dtype)

    # Grid
    def grid(meta):
        BH = meta["BLOCK_H"]
        BW = meta["BLOCK_W"]
        return (
            B * C,
            triton.cdiv(H_out, BH),
            triton.cdiv(W_out, BW),
        )

    nearest_upsample_2d_kernel[grid](
        x, out,
        B, C, H_in, W_in, H_out, W_out,
        float(scale_factor),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # Keep the original Interpolate to mirror init, but we won't use it in forward
        self._upsample = Interpolate()
        self.cat1 = YOLOConcat(1)
        self.c2f_p4 = YOLOC2f(384, 128, n=1, shortcut=False)
        self.cat2 = YOLOConcat(1)
        self.c2f_p3 = YOLOC2f(192, 64, n=1, shortcut=False)
        self.down_p3 = YOLOConv(64, 64, 3, 2)
        self.cat3 = YOLOConcat(1)
        self.c2f_n4 = YOLOC2f(192, 128, n=1, shortcut=False)
        self.down_n4 = YOLOSCDown(128, 128, 3, 2)
        self.cat4 = YOLOConcat(1)
        self.c2fcib_n5 = YOLOC2fCIB(384, 256, n=1, shortcut=True, lk=True)

    def forward(self, feats: dict[str, torch.Tensor]):
        p3_backbone = feats["p3_backbone"]
        p4_backbone = feats["p4_backbone"]
        p5_backbone = feats["p5_backbone"]

        # Use Triton nearest upsample for CUDA tensors; fallback to PyTorch otherwise
        if p5_backbone.is_cuda:
            up5 = _triton_upsample_nearest(p5_backbone, scale_factor=2.0)
        else:
            up5 = torch.nn.functional.interpolate(
                p5_backbone, scale_factor=2.0, mode="nearest"
            )

        x = self.cat1([up5, p4_backbone])
        p4 = self.c2f_p4(x)

        if p4.is_cuda:
            up4 = _triton_upsample_nearest(p4, scale_factor=2.0)
        else:
            up4 = torch.nn.functional.interpolate(
                p4, scale_factor=2.0, mode="nearest"
            )

        x = self.cat2([up4, p3_backbone])
        p3 = self.c2f_p3(x)

        x = self.down_p3(p3)
        x = self.cat3([x, p4])
        n4 = self.c2f_n4(x)

        x = self.down_n4(n4)
        x = self.cat4([x, p5_backbone])
        n5 = self.c2fcib_n5(x)
        return [p3, n4, n5]

YOLOv10Neck = ModelNew
