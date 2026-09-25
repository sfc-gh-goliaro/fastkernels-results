"""SwiGLU MLP for GLA / RetNet decoder layers.

Three-projection variant matching FLA's checkpoint format:
  ``gate_proj.weight`` / ``up_proj.weight`` / ``down_proj.weight``

The existing ``L2.swiglu_mlp.SwiGLUMlp`` uses a different parameter
naming scheme (``fc1_g`` / ``fc1_x`` / ``fc2``), so we keep this thin
FLA-named variant rather than remapping checkpoint keys at load time.

Built exclusively from L1 ops.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.linear import Linear
from ..L1.silu_and_mul import SiluAndMul


@triton.jit
def _silu_mul_kernel(
    gate_up_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = (
        tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ).to(tl.int64)
    mask = offsets < n_elements
    row = offsets // N
    col = offsets - row * N
    base = row * (2 * N) + col
    gate = tl.load(gate_up_ptr + base, mask=mask).to(tl.float32)
    up = tl.load(gate_up_ptr + base + N, mask=mask)
    activated = (gate * tl.sigmoid(gate)).to(tl.bfloat16)
    tl.store(out_ptr + offsets, activated * up, mask=mask)


def _silu_mul_triton(gate_up: torch.Tensor) -> torch.Tensor:
    intermediate_size = gate_up.shape[-1] // 2
    out = torch.empty(
        (*gate_up.shape[:-1], intermediate_size),
        device=gate_up.device,
        dtype=gate_up.dtype,
    )
    n_elements = out.numel()
    _silu_mul_kernel[(triton.cdiv(n_elements, 8192),)](
        gate_up,
        out,
        n_elements,
        N=intermediate_size,
        BLOCK_SIZE=8192,
        num_warps=8,
    )
    return out


class GLAMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)
        self.act = SiluAndMul()
        self._merged_weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._merged_weight is None:
            self._merged_weight = torch.cat(
                (self.gate_proj.weight, self.up_proj.weight), dim=0
            )
        x_2d = x.view(-1, x.shape[-1])
        gate_up = torch.mm(x_2d, self._merged_weight.t())
        rows = gate_up.numel() // gate_up.shape[-1]
        hidden = _silu_mul_triton(gate_up) if rows > 128 else self.act(gate_up)
        out = torch.mm(hidden, self.down_proj.weight.t())
        return out.view(*x.shape[:-1], out.shape[-1])
