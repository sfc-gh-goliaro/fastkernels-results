"""Interpolate: specialised Triton 2x nearest-neighbour upsample for the
captured regime, with F.interpolate as the fallback for everything else.

Layout fact the kernel is built on.  For a contiguous NCHW input flattened to
(R, W) with R = N*C*H, the 2x nearest output row-pair for input row r occupies
exactly [r*4W, (r+1)*4W) of the flat output: output rows 2h and 2h+1 are
adjacent in memory AND bit-identical, and plane boundaries need no special
case because the base offset r*4W is linear in the *global* row index.  So the
whole operator is

    out[4*r*W + 2*c + {0, 1, 2W, 2W+1}] = in[r*W + c]

The W-duplicated pair (out[2c], out[2c+1]) = (a, a) is a single 32-bit word, so
viewing the output as int32 turns the op into a permutation-free, fully
contiguous store of one 32-bit word per output word:

    out32[r*2W + u] = dup16(in[r*W + (u mod W)])       u in [0, 2W)

Parallelising over *output* 32-bit words (rather than input elements) is what
makes the store one flat contiguous run per program: the alternative
input-parallel tiling has a row stride of 4W elements that no power-of-two tile
width can divide, which fragments the stores into 2W-element pieces.  The input
is read twice (once per output row of the pair), but that is 1/8 of the write
volume and L1-resident, and it buys 100% lane utilisation with no padding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def _upsample2x_nearest_2byte(in_ptr, out_ptr, NQ, W: tl.constexpr,
                              W2: tl.constexpr, BLOCK: tl.constexpr,
                              MASK: tl.constexpr):
    """2x nearest upsample of a contiguous (R, W) plane stack of 2-byte values.

    ``in_ptr`` is an int16 view of the input, ``out_ptr`` an int32 view of the
    output; the kernel only moves bits, so it serves every 2-byte dtype.
    """
    q = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = q < NQ if MASK else None
    r = q // W2                        # global input row index
    u = q - r * W2                     # 32-bit word within the output row pair
    col = tl.where(u >= W, u - W, u)   # both rows of the pair read the same row
    v = tl.load(in_ptr + r * W + col, mask=m)
    v = v.to(tl.uint16, bitcast=True).to(tl.uint32)
    tl.store(out_ptr + q, v | (v << 16), mask=m)


_BLOCK = 1024
_NUM_WARPS = 4
# Index math inside the kernel is 32-bit; keep well clear of the boundary.
_MAX_OUT_WORDS = 1 << 30


def _is_two(v) -> bool:
    """True for a scale_factor meaning 'exactly 2x in H and W'."""
    if isinstance(v, (int, float)):
        return v == 2
    if isinstance(v, (tuple, list)):
        return len(v) == 2 and all(isinstance(s, (int, float)) and s == 2 for s in v)
    return False


class Interpolate(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        size: int | tuple[int, ...] | None = None,
        scale_factor: float | tuple[float, ...] | None = None,
        mode: str = "nearest",
        align_corners: bool | None = None,
    ) -> torch.Tensor:
        # Fast-path gate: cheap python/int checks only, no host<->device sync.
        if (
            size is None
            and align_corners is None
            and mode == "nearest"
            and _is_two(scale_factor)
            and x.dim() == 4
            and x.is_cuda
            and x.element_size() == 2
            and x.is_contiguous()
        ):
            n, c, h, w = x.shape
            nq = 2 * n * c * h * w          # 32-bit words in the output
            if 0 < nq <= _MAX_OUT_WORDS:
                out = x.new_empty((n, c, 2 * h, 2 * w))
                grid = ((nq + _BLOCK - 1) // _BLOCK,)
                _upsample2x_nearest_2byte[grid](
                    x.view(torch.int16), out.view(torch.int32), nq,
                    W=w, W2=2 * w, BLOCK=_BLOCK, MASK=(nq % _BLOCK != 0),
                    num_warps=_NUM_WARPS,
                )
                return out

        return F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
