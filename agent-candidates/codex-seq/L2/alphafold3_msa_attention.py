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
def _pair_weights_16x128_kernel(
    z,
    mask,
    ln_weight,
    ln_bias,
    proj_weight,
    weights,
    LN_EPS: tl.constexpr,
):
    q = tl.program_id(0)
    k = tl.arange(0, 16)
    c = tl.arange(0, 128)

    z_offsets = (q * 16 + k[:, None]) * 128 + c[None, :]
    x = tl.load(z + z_offsets).to(tl.float32)
    mean = tl.sum(x, axis=1) * (1.0 / 128.0)
    centered = x - mean[:, None]
    variance = tl.sum(centered * centered, axis=1) * (1.0 / 128.0)
    scale = tl.load(ln_weight + c).to(tl.float32)
    bias = tl.load(ln_bias + c).to(tl.float32)
    x = (centered * tl.rsqrt(variance[:, None] + LN_EPS) * scale + bias).to(
        tl.bfloat16
    )

    h16 = tl.arange(0, 16)
    proj_offsets = h16[None, :] * 128 + c[:, None]
    proj = tl.load(
        proj_weight + proj_offsets,
        mask=h16[None, :] < 8,
        other=0.0,
    )
    scores = tl.dot(x, proj).to(tl.bfloat16).to(tl.float32)
    pair_mask = tl.load(mask + q * 16 + k).to(tl.float32)
    scores += (pair_mask[:, None] - 1.0) * 1.0e9

    scores -= tl.max(scores, axis=0)[None, :]
    numerator = tl.exp(scores)
    probabilities = numerator / tl.sum(numerator, axis=0)[None, :]

    out_offsets = (
        h16[None, :] * 256
        + q * 16
        + k[:, None]
    )
    tl.store(
        weights + out_offsets,
        probabilities,
        mask=h16[None, :] < 8,
    )


@triton.jit
def _msa_fused_8x16x64_kernel(
    m,
    pair_weights,
    ln_weight,
    ln_bias,
    value_weight,
    gate_weight,
    output_weight,
    output,
    LN_EPS: tl.constexpr,
):
    seq = tl.program_id(0)
    row = tl.arange(0, 16)
    c = tl.arange(0, 64)

    m_offsets = (seq * 16 + row[:, None]) * 64 + c[None, :]
    x = tl.load(m + m_offsets).to(tl.float32)
    mean = tl.sum(x, axis=1) * (1.0 / 64.0)
    centered = x - mean[:, None]
    variance = tl.sum(centered * centered, axis=1) * (1.0 / 64.0)
    scale = tl.load(ln_weight + c).to(tl.float32)
    bias = tl.load(ln_bias + c).to(tl.float32)
    x = (centered * tl.rsqrt(variance[:, None] + LN_EPS) * scale + bias).to(
        tl.bfloat16
    )

    d = tl.arange(0, 32)
    out_c = tl.arange(0, 64)
    result = tl.zeros((16, 64), dtype=tl.float32)

    for head_group in tl.static_range(2):
        hidden = head_group * 32 + d
        linear_offsets = hidden[None, :] * 64 + c[:, None]

        wv = tl.load(value_weight + linear_offsets)
        value = tl.dot(x, wv).to(tl.bfloat16)

        wg = tl.load(gate_weight + linear_offsets)
        gate_linear = tl.dot(x, wg).to(tl.bfloat16).to(tl.float32)
        gate = 0.5 + gate_linear * (
            0.256 - 0.034 * tl.abs(gate_linear)
        )
        gate = tl.where(
            gate_linear > 4.0,
            0.9905,
            tl.where(gate_linear < -4.0, 0.0095, gate),
        ).to(tl.bfloat16)

        k = tl.arange(0, 16)
        attn_offsets = (
            (head_group * 4) * 256 + row[:, None] * 16 + k[None, :]
        )
        attn0 = tl.load(pair_weights + attn_offsets)
        attn1 = tl.load(pair_weights + attn_offsets + 256)
        attn2 = tl.load(pair_weights + attn_offsets + 512)
        attn3 = tl.load(pair_weights + attn_offsets + 768)
        averaged = (
            tl.dot(attn0, tl.where(d[None, :] < 8, value, 0.0)).to(tl.bfloat16)
            + tl.dot(
                attn1,
                tl.where((d[None, :] >= 8) & (d[None, :] < 16), value, 0.0),
            ).to(tl.bfloat16)
            + tl.dot(
                attn2,
                tl.where((d[None, :] >= 16) & (d[None, :] < 24), value, 0.0),
            ).to(tl.bfloat16)
            + tl.dot(attn3, tl.where(d[None, :] >= 24, value, 0.0)).to(
                tl.bfloat16
            )
        ).to(tl.bfloat16)
        gated = (averaged * gate).to(tl.bfloat16)

        wo_offsets = out_c[None, :] * 64 + hidden[:, None]
        wo = tl.load(output_weight + wo_offsets)
        result += tl.dot(gated, wo)

    tl.store(output + m_offsets, result)


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

        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        if (
            m.is_cuda
            and m.dtype == torch.bfloat16
            and m.shape == (1, 8, 16, 64)
            and z.shape == (1, 16, 16, 128)
            and mask.shape == (1, 16, 16)
            and self.c_m == 64
            and self.c_z == 128
            and self.c_hidden == 8
            and self.no_heads == 8
        ):
            pair_weights = torch.empty(
                (8, 16, 16), device=z.device, dtype=z.dtype
            )
            output = torch.empty_like(m)
            _pair_weights_16x128_kernel[(16,)](
                z,
                mask,
                self.layer_norm_z.weight,
                self.layer_norm_z.bias,
                self.linear_z.weight,
                pair_weights,
                LN_EPS=self.layer_norm_z.eps,
                num_warps=8,
                launch_pdl=True,
            )
            _msa_fused_8x16x64_kernel[(8,)](
                m,
                pair_weights,
                self.layer_norm_m.weight,
                self.layer_norm_m.bias,
                self.linear_v.weight,
                self.linear_g.weight,
                self.linear_o.weight,
                output,
                LN_EPS=self.layer_norm_m.eps,
                num_warps=4,
                launch_pdl=True,
            )
            return output

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
