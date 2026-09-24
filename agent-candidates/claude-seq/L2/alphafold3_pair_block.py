"""PairBlock for AlphaFold3 -- single fused CUDA kernel.

The captured workload is N_res=16, c_z=128: ~120 eager kernels whose total GPU
work is a few microseconds, so the baseline is entirely launch bound. The whole
block (TriMulOut -> TriMulIn -> TriAttStart -> TriAttEnd -> SwiGLUTransition)
runs here as ONE kernel launch on a 16-CTA thread-block cluster; see
``alphafold3_pair_block.cu`` for the parallel decomposition.

Submodule names/signatures match the baseline so weight sharing works, and any
configuration the kernel does not cover falls back to the baseline sequence.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .alphafold3_triangle_multiplication import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from .alphafold3_triangle_attention import TriangleAttention
from .alphafold3_swiglu_transition import SwiGLUTransition

from ....infra.cuda_ext import lazy_op

_C = lazy_op("fk_l2_af3_pair_block_fused", "alphafold3_pair_block.cu",
             extra_cuda_cflags=["-arch=sm_100a", "--use_fast_math"])

_TILE_BF16 = 32768 // 2   # bf16 elements in one 128x128 weight tile


def _pack_frag(w: torch.Tensor) -> torch.Tensor:
    """[N, K] -> mma m16n8k16 B-fragment order, tile-major (n-chunk, k-chunk).

    A 128x128 tile becomes ((n_tile*8 + k_tile)*32 + lane)*4 bf16, which is the
    exact 8-byte operand each lane feeds to one mma, so the kernel reads a
    weight fragment with a single conflict-free shared load.
    """
    n, k = w.shape
    return (w.reshape(n // 128, 16, 8, k // 128, 8, 2, 4, 2)
             .permute(0, 3, 1, 4, 2, 6, 5, 7)
             .reshape(-1))


class PairBlock(nn.Module):
    """Shared pair stack block for AF3 PairFormer / MSA module / template.

    Args:
        c_z: Pair embedding channel dimension
        c_hidden_mul: Hidden dim for triangle multiplication
        c_hidden_pair_att: Per-head hidden dim for triangle attention
        no_heads_pair: Number of heads in triangle attention
        transition_n: Scale of pair transition hidden dimension
        pair_dropout: Dropout rate (unused in inference baseline)
        inf: Large constant for masking
    """

    def __init__(
        self,
        c_z: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_pair: int,
        transition_n: int,
        pair_dropout: float = 0.0,
        fuse_projection_weights: bool = False,
        inf: float = 1e9,
    ):
        super().__init__()

        self.tri_mul_out = TriangleMultiplicationOutgoing(c_z, c_hidden_mul)
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z, c_hidden_mul)

        self.tri_att_start = TriangleAttention(
            c_z, c_hidden_pair_att, no_heads_pair, starting=True, inf=inf,
        )
        self.tri_att_end = TriangleAttention(
            c_z, c_hidden_pair_att, no_heads_pair, starting=False, inf=inf,
        )

        self.pair_transition = SwiGLUTransition(c_in=c_z, n=transition_n)

        # the fused kernel is specialised for the captured AF3 pair stack
        self._fusable = (
            c_z == 128 and c_hidden_mul == 128 and c_hidden_pair_att == 32
            and no_heads_pair == 4 and transition_n == 4 and inf == 1e9
        )
        self._packed = None
        # Packing the weights is done once. Rather than re-scanning ~40
        # parameters on every call (which would cost more CPU time than the
        # kernel itself at this size), drop the cache whenever weights could
        # have changed: a state-dict load or any ``_apply`` (``to``, ``cuda``).
        self.register_load_state_dict_post_hook(PairBlock._drop_cache)

    @staticmethod
    def _drop_cache(module, incompatible_keys=None):
        module._packed = None

    def _apply(self, *args, **kwargs):
        self._packed = None
        return super()._apply(*args, **kwargs)

    # -- weight packing ----------------------------------------------------
    def _norm_params(self):
        return [
            self.tri_mul_out.layer_norm_in, self.tri_mul_out.layer_norm_out,
            self.tri_mul_in.layer_norm_in, self.tri_mul_in.layer_norm_out,
            self.tri_att_start.layer_norm, self.tri_att_end.layer_norm,
            self.pair_transition.layer_norm,
        ]

    @torch.no_grad()
    def _build_packed(self, device, dtype):
        # the kernel reads the packed weights as bf16; if the module is not in
        # that dtype, stop claiming the fast path and let the baseline run
        if self.tri_mul_out.linear_a_p.weight.dtype is not dtype:
            self._fusable = False
            return None
        mo, mi = self.tri_mul_out, self.tri_mul_in
        a, b = self.tri_att_start, self.tri_att_end
        t = self.pair_transition

        def cat(mods):
            return torch.cat([m.weight for m in mods], 0)

        tiles = [
            _pack_frag(cat([mo.linear_a_p, mo.linear_a_g, mo.linear_b_p,
                            mo.linear_b_g, mo.linear_g])),
            _pack_frag(mo.linear_z.weight),
            _pack_frag(cat([mi.linear_a_p, mi.linear_a_g, mi.linear_b_p,
                            mi.linear_b_g, mi.linear_g])),
            _pack_frag(mi.linear_z.weight),
            _pack_frag(cat([a.mha.linear_q, a.mha.linear_k, a.mha.linear_v,
                            a.mha.linear_g])),
            _pack_frag(a.mha.linear_o.weight),
            _pack_frag(cat([b.mha.linear_q, b.mha.linear_k, b.mha.linear_v,
                            b.mha.linear_g])),
            _pack_frag(b.mha.linear_o.weight),
            # linear_a / linear_b interleaved in 128-row blocks: the kernel
            # consumes (a_i, b_i) as one pair so silu(a_i) * b_i stays in
            # registers instead of round-tripping shared memory
            _pack_frag(torch.cat(
                [w for i in range(4)
                 for w in (t.swiglu.linear_a.weight[i * 128:(i + 1) * 128],
                           t.swiglu.linear_b.weight[i * 128:(i + 1) * 128])], 0)),
            _pack_frag(t.linear_out.weight),
            # unswizzled tail: the two 4x128 triangle-bias projections
            a.linear_z.weight.reshape(-1), b.linear_z.weight.reshape(-1),
        ]
        w = torch.cat(tiles).contiguous()
        assert w.numel() == 34 * _TILE_BF16 + 2 * 4 * 128, w.numel()

        ln = torch.cat([torch.cat([n.weight.float().reshape(-1),
                                   n.bias.float().reshape(-1)])
                        for n in self._norm_params()]).contiguous()
        ws = torch.empty(4 * 256 * 128 + 2 * 256 * 4, dtype=dtype, device=device)
        self._packed = (w, ln, ws)
        return self._packed

    # -- forward -----------------------------------------------------------
    def _baseline(self, z, pair_mask, _mask_trans):
        pair_trans_mask = pair_mask if _mask_trans else None
        z = z + self.tri_mul_out(z, mask=pair_mask)
        z = z + self.tri_mul_in(z, mask=pair_mask)
        z = z + self.tri_att_start(z, mask=pair_mask)
        z = z + self.tri_att_end(z, mask=pair_mask)
        z = z + self.pair_transition(z, mask=pair_trans_mask)
        return z

    def forward(
        self,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            z:         [*, N, N, C_z] pair embedding
            pair_mask: [*, N, N] pair mask

        Returns:
            [*, N, N, C_z] updated pair embedding
        """
        if (self._fusable and z.numel() == 32768 and z.dtype is torch.bfloat16
                and z.shape[-3:] == (16, 16, 128) and z.is_contiguous()
                and pair_mask is not None and pair_mask.numel() == 256
                and pair_mask.dtype is torch.bfloat16 and pair_mask.is_contiguous()
                and z.is_cuda):
            pk = self._packed or self._build_packed(z.device, z.dtype)
            if pk is not None:
                out = torch.empty_like(z)
                _C.pair_block(z, pair_mask, out, pk[0], pk[1], pk[2],
                              1 if _mask_trans else 0)
                return out

        return self._baseline(z, pair_mask, _mask_trans)
