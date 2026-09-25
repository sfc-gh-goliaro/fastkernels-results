"""MSA pair-weighted averaging for AlphaFold3 (Algorithm 10).

Weighted averaging over the MSA representation using pair activations,
NOT key-query self-attention.

Reference: openfold3/core/model/layers/msa.py MSAPairWeightedAveraging
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.sigmoid import Sigmoid
from ..L1.softmax import Softmax
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear


def _permute_final_dims(tensor: torch.Tensor, inds: tuple[int, ...]) -> torch.Tensor:
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


@triton.jit
def _output_kernel(
    weights_ptr,
    temp_ptr,
    out_w_ptr,
    out_ptr,
):
    seq = tl.program_id(0)
    q = tl.arange(0, 16)[:, None]
    d = tl.arange(0, 64)[None, :]
    k = tl.arange(0, 16)

    averaged = tl.zeros((16, 64), tl.float32)
    for head in tl.static_range(0, 8):
        pair_weights = tl.load(
            weights_ptr + head * 256 + q * 16 + k[None, :]
        )
        values = tl.load(
            temp_ptr + (seq * 16 + k[:, None]) * 64 + d,
            mask=(d >= head * 8) & (d < (head + 1) * 8),
            other=0.0,
        )
        head_out = tl.dot(pair_weights, values)
        averaged += tl.where(
            (d >= head * 8) & (d < (head + 1) * 8),
            head_out.to(tl.bfloat16).to(tl.float32),
            0.0,
        )

    gate_logits = tl.load(temp_ptr + 8192 + (seq * 16 + q) * 64 + d)
    gate = (1.0 / (1.0 + tl.exp(-gate_logits.to(tl.float32))))
    gate = gate.to(tl.bfloat16).to(tl.float32)
    gated = (averaged * gate).to(tl.bfloat16)

    ci = tl.arange(0, 64)[:, None]
    co = tl.arange(0, 64)[None, :]
    out_w = tl.load(out_w_ptr + co * 64 + ci)
    result = tl.dot(gated, out_w)
    tl.store(out_ptr + (seq * 16 + q) * 64 + d, result)


@triton.jit
def _input_projections_kernel(
    z_ptr,
    mask_ptr,
    z_norm_w_ptr,
    z_norm_b_ptr,
    z_proj_w_ptr,
    weights_ptr,
    m_ptr,
    m_norm_w_ptr,
    m_norm_b_ptr,
    value_w_ptr,
    gate_w_ptr,
    temp_ptr,
    inf: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < 16:
        z_q = pid
        z_k = tl.arange(0, 16)[:, None]
        z_c = tl.arange(0, 128)[None, :]

        z_values = tl.load(
            z_ptr + (z_q * 16 + z_k) * 128 + z_c
        ).to(tl.float32)
        z_mean = tl.sum(z_values, axis=1) * (1.0 / 128.0)
        z_centered = z_values - z_mean[:, None]
        z_var = tl.sum(z_centered * z_centered, axis=1) * (1.0 / 128.0)
        z_norm = z_centered * tl.rsqrt(z_var[:, None] + 1.0e-5)
        z_norm = z_norm * tl.load(z_norm_w_ptr + z_c).to(tl.float32)
        z_norm += tl.load(z_norm_b_ptr + z_c).to(tl.float32)

        z_h = tl.arange(0, 16)[None, :]
        z_ci = tl.arange(0, 128)[:, None]
        z_proj_w = tl.load(
            z_proj_w_ptr + z_h * 128 + z_ci,
            mask=z_h < 8,
            other=0.0,
        )
        z_logits = tl.dot(z_norm.to(tl.bfloat16), z_proj_w)
        z_logits = z_logits.to(tl.bfloat16).to(tl.float32)
        z_mask = tl.load(mask_ptr + z_q * 16 + z_k).to(tl.float32)
        z_mask_bias = (inf * (z_mask - 1.0)).to(tl.bfloat16).to(tl.float32)
        z_logits = (z_logits + z_mask_bias).to(tl.bfloat16).to(tl.float32)
        z_logits -= tl.max(z_logits, axis=0)[None, :]
        z_numer = tl.exp(z_logits)
        z_weights = z_numer / tl.sum(z_numer, axis=0)[None, :]
        tl.store(
            weights_ptr + z_h * 256 + z_q * 16 + z_k,
            z_weights,
            mask=z_h < 8,
        )
    else:
        m_block = pid - 16
        m_row = m_block * 16 + tl.arange(0, 16)[:, None]
        m_c = tl.arange(0, 64)[None, :]

        m_values = tl.load(m_ptr + m_row * 64 + m_c).to(tl.float32)
        m_mean = tl.sum(m_values, axis=1) * (1.0 / 64.0)
        m_centered = m_values - m_mean[:, None]
        m_var = tl.sum(m_centered * m_centered, axis=1) * (1.0 / 64.0)
        m_norm = m_centered * tl.rsqrt(m_var[:, None] + 1.0e-5)
        m_norm = m_norm * tl.load(m_norm_w_ptr + m_c).to(tl.float32)
        m_norm += tl.load(m_norm_b_ptr + m_c).to(tl.float32)
        m_norm = m_norm.to(tl.bfloat16)

        m_ci = tl.arange(0, 64)[:, None]
        m_co = tl.arange(0, 64)[None, :]
        m_value_w = tl.load(value_w_ptr + m_co * 64 + m_ci)
        m_gate_w = tl.load(gate_w_ptr + m_co * 64 + m_ci)
        m_value = tl.dot(m_norm, m_value_w)
        m_gate = tl.dot(m_norm, m_gate_w)
        m_offsets = m_row * 64 + m_c
        tl.store(temp_ptr + m_offsets, m_value)
        tl.store(temp_ptr + 8192 + m_offsets, m_gate)


class MSARowAttentionWithPairBias(nn.Module):
    """AF3 MSA Pair-Weighted Averaging (Algorithm 10).

    Uses pair activations as weights (softmax over token dim) instead of
    key-query attention.  Parameter names match the checkpoint layout:
    linear_v, linear_g, linear_o (no nested mha).

    Args:
        c_m: MSA input channel dimension
        c_z: Pair embedding channel dimension
        c_hidden: Per-head hidden channel dimension
        no_heads: Number of attention heads
        inf: Large constant for masking
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        inf: float = 1e9,
    ):
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf

        self.layer_norm_m = LayerNorm(c_m)
        self.layer_norm_z = LayerNorm(c_z)
        self.linear_z = Linear(c_z, no_heads, bias=False)

        self.linear_v = Linear(c_m, c_hidden * no_heads, bias=False)
        self.linear_g = Linear(c_m, c_hidden * no_heads, bias=False)
        self.linear_o = Linear(c_hidden * no_heads, c_m, bias=False)

        self.sigmoid = Sigmoid()
        self.softmax = Softmax(dim=-1)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            m:    [*, N_seq, N_res, C_m] MSA embedding
            z:    [*, N_res, N_res, C_z] pair embedding
            mask: [*, N_res, N_res] pair mask

        Returns:
            [*, N_seq, N_res, C_m] updated MSA embedding
        """
        if z is None:
            return m

        if (
            m.is_cuda
            and m.dtype == torch.bfloat16
            and z.dtype == torch.bfloat16
            and mask is not None
            and mask.dtype == torch.bfloat16
            and tuple(m.shape) == (1, 8, 16, 64)
            and tuple(z.shape) == (1, 16, 16, 128)
            and tuple(mask.shape) == (1, 16, 16)
            and self.c_m == 64
            and self.c_z == 128
            and self.c_hidden == 8
            and self.no_heads == 8
        ):
            weights = torch.empty((8, 16, 16), device=z.device, dtype=z.dtype)
            temp = torch.empty((2, 8192), device=m.device, dtype=m.dtype)
            out = torch.empty_like(m)

            _input_projections_kernel[(24,)](
                z,
                mask,
                self.layer_norm_z.weight,
                self.layer_norm_z.bias,
                self.linear_z.weight,
                weights,
                m,
                self.layer_norm_m.weight,
                self.layer_norm_m.bias,
                self.linear_v.weight,
                self.linear_g.weight,
                temp,
                inf=self.inf,
                num_warps=4,
            )
            _output_kernel[(8,)](
                weights,
                temp,
                self.linear_o.weight,
                out,
                num_warps=4,
            )
            return out

        n_res = z.shape[-2]

        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        # Pair bias: [*, 1, no_heads, N_res, N_res]
        mask_bias = (self.inf * (mask - 1))[..., None, None, :, :]
        z_norm = self.layer_norm_z(z)
        z_proj = self.linear_z(z_norm)
        z_weights = _permute_final_dims(z_proj, (2, 0, 1)).unsqueeze(-4)
        z_weights = z_weights + mask_bias
        z_weights = self.softmax(z_weights)

        m = self.layer_norm_m(m)

        # Value projection
        v = self.linear_v(m)
        v = v.view(v.shape[:-1] + (self.no_heads, -1))
        v = v.transpose(-2, -3)  # [*, N_seq, H, N_res, C_hidden]

        # Weighted average: [*, N_seq, H, N_res, C_hidden]
        o = torch.einsum("...hqk,...hkc->...qhc", z_weights, v)

        # Gating
        g = self.sigmoid(self.linear_g(m))
        g = g.view(g.shape[:-1] + (self.no_heads, -1))

        o = o * g

        # Flatten heads and project
        o = o.reshape(o.shape[:-2] + (-1,))
        o = self.linear_o(o)

        return o
