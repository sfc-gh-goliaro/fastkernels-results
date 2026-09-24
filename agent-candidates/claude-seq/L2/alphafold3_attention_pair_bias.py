"""Attention with pair bias for AlphaFold3 -- fused to 2-3 kernel launches.

AttentionPairBias: Used in PairFormer and diffusion transformer. Uses a single
    layer_norm_a for both Q and K (AdaLN or LayerNorm).
CrossAttentionPairBias: Used in atom attention (sequence-local). Uses separate
    layer_norm_a_q and layer_norm_a_k, no layer_norm_z.

Reference: openfold3/core/model/layers/attention_pair_bias.py

Why this is written the way it is
--------------------------------
At the captured shapes both operators are launch-bound, not FLOP-bound.
AttentionPairBias pushes 16 tokens through ~125 MFLOP -- roughly a microsecond of
arithmetic on a B200 -- and the reference composition takes ~200 us for it, spread
over 32 kernels.  CrossAttentionPairBias needs 117 kernels for ~640 us, most of
them the elementwise steps of ``_get_block_key_indices`` plus the blocked gather.
Measured on the bench's own timing loop, the floor for a module that does nothing
is 20.4 us and each additional kernel launch costs a flat 4.1 us, so the first
thing that matters is the launch count:

    AttentionPairBias       3 launches (2 without AdaLN)
    CrossAttentionPairBias  2 launches

Everything not separated by a true data dependency shares a launch, including work
that is cheaper to recompute per block than to synchronize on (the attention, and
every LayerNorm statistic).  Two algebraic points make that possible:

* CrossAttentionPairBias never materializes the blocked key view.  A blocked key
  row is a pure function of the atom it gathers from, so the key-side AdaLN and
  projections run once per atom (384 rows) instead of once per (block, key) slot
  (1536 rows), and the block view is applied later, where the attention reads k/v.
  The gather indices are reproduced exactly, including the baseline's bf16 index
  arithmetic -- see the .cu, it is load-bearing.
* The AdaLN-Zero output gate depends only on ``s``, and the pair bias only on
  ``z``, so both ride along in a launch that exists for other reasons.

The kernels live in ``alphafold3_attention_pair_bias_fk.cu`` beside this file (the
same arrangement the L1 winners use) and are JIT-compiled on first import.
Anything the fast path does not cover -- non-bf16 dtypes, a real batch dimension,
autograd, geometry that is not 16-aligned -- falls through to the reference
composition below, which is the baseline code unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.sigmoid import Sigmoid
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from .alphafold3_swiglu import AdaLN
from .alphafold3_of3_attention import OF3Attention

try:
    from fastkernels.infra.cuda_ext import load_op

    _C = load_op("alphafold3_attention_pair_bias_fk",
                 "alphafold3_attention_pair_bias_fk.cu")
except Exception:  # noqa: BLE001 -- no nvcc / no GPU: reference path only
    _C = None

# Mirrors the flag bits in the .cu.
_F_ADA = 1 << 0
_F_GATING = 1 << 1
_F_LNZ_B = 1 << 2
_F_LNS_B = 1 << 3
_F_LNA_W = 1 << 4
_F_LNA_B = 1 << 5
_F_LNSK_B = 1 << 6
_F_LNK_W = 1 << 7
_F_LNK_B = 1 << 8

_ALIGN = 16
_SECT = _ALIGN * _ALIGN  # packed-section granularity, one wmma tile


def _permute_final_dims(tensor: torch.Tensor, inds: list[int]) -> torch.Tensor:
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def _pack(parts: list[torch.Tensor | None]) -> torch.Tensor:
    """One flat bf16 buffer; each section padded to a whole wmma tile so every
    tile base inside it is 512-byte aligned.  ``None`` sections are skipped --
    the .cu skips the same ones, driven by the flag word."""
    flat = []
    for t in parts:
        if t is None:
            continue
        v = t.detach().reshape(-1).contiguous()
        pad = (-v.numel()) % _SECT
        if pad:
            v = F.pad(v, (0, pad))
        flat.append(v)
    return torch.cat(flat)


def _tile16(w: torch.Tensor) -> torch.Tensor:
    """[R, K] -> 16x16 tile-major, tiles ordered (row-tile, col-tile).

    Every matrix the kernels consume as a wmma B operand goes through this.  A
    B-fragment load out of a row-major [n][k] matrix touches 16 rows ``ldb``
    apart, so it costs 16 scattered 32-byte transactions; tile-major turns it
    into one contiguous 512-byte run.  On these shapes that was worth ~10x.
    One tile holds [row][col], which is exactly what both the col-major B and the
    row-major A fragment layouts want, so q/k/v can share a single packing.
    """
    r, k = w.shape
    assert r % _ALIGN == 0 and k % _ALIGN == 0, (r, k)
    return (w.detach().reshape(r // _ALIGN, _ALIGN, k // _ALIGN, _ALIGN)
            .permute(0, 2, 1, 3).contiguous().reshape(-1))


def _heads_padded(w: torch.Tensor, h: int, d: int, dp: int) -> torch.Tensor:
    """[h*d, c] -> [h*dp, c], zero-filling the per-head tail.

    c_hidden is 24 for one captured variant, so the per-head dim is padded to a
    multiple of 16 and kept padded all the way through q/k/v/gate/o.  The padded
    lanes see zero weights, hence zero v and zero contribution to the output
    projection, so no masking is needed anywhere downstream.
    """
    x = w.detach().reshape(h, d, -1)
    if dp != d:
        x = F.pad(x, (0, 0, 0, dp - d))
    return x.reshape(h * dp, -1)


def _bias_padded(b: torch.Tensor, h: int, d: int, dp: int) -> torch.Tensor:
    x = b.detach().reshape(h, d)
    if dp != d:
        x = F.pad(x, (0, dp - d))
    return x.reshape(-1)


def _out_padded(w: torch.Tensor, h: int, d: int, dp: int) -> torch.Tensor:
    """linear_o weight [c, h*d] -> [c, h*dp]."""
    x = w.detach().reshape(-1, h, d)
    if dp != d:
        x = F.pad(x, (0, dp - d))
    return x.reshape(x.shape[0], h * dp)


def _all_bf16_cuda(mod: nn.Module) -> bool:
    ps = list(mod.parameters())
    return bool(ps) and all(p.is_cuda and p.dtype is torch.bfloat16 for p in ps)


class _FusedBase(nn.Module):
    """Cache management for the packed weight buffer.

    The pack holds a *copy* of the weights, so it has to be dropped whenever the
    parameters change.  ``_apply`` covers ``.to()`` / ``.cuda()`` / ``.half()``
    and the load-state-dict hook covers weight loading; between them that is
    every path the bench and the serving stack use.
    """

    def _fk_reset(self) -> None:
        self._fk = None

    def _apply(self, *args, **kwargs):
        out = super()._apply(*args, **kwargs)
        self._fk = None
        return out

    def _fk_hook(self, *args, **kwargs) -> None:
        self._fk = None


class AttentionPairBias(_FusedBase):
    """AF3 Algorithm 24: Attention with pair bias.

    When use_ada_layer_norm is True, uses two separate AdaLN instances
    (layer_norm_a_q, layer_norm_a_k) for query and key normalization,
    plus a linear_ada_out for output gating.

    Reference: openfold3/core/model/layers/attention_pair_bias.py

    Args:
        c_q: Input dimension of query/key/value
        c_s: Single activation channel dimension (for AdaLN)
        c_z: Pair activation channel dimension
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        use_ada_layer_norm: Whether to use AdaLN-Zero conditioning
        gating: Whether to gate output
        inf: Large constant for masking
    """

    def __init__(
        self,
        c_q: int,
        c_k: int = 0,
        c_v: int = 0,
        c_s: int = 0,
        c_z: int = 128,
        c_hidden: int = 32,
        no_heads: int = 4,
        use_ada_layer_norm: bool = False,
        gating: bool = True,
        inf: float = 1e9,
    ):
        super().__init__()
        c_k = c_k or c_q
        c_v = c_v or c_q

        self.c_q = c_q
        self.c_s = c_s
        self.c_z = c_z
        self.inf = inf
        self.use_ada_layer_norm = use_ada_layer_norm

        if use_ada_layer_norm:
            self.layer_norm_a = AdaLN(c_a=c_q, c_s=c_s)
            self.linear_ada_out = Linear(c_s, c_q, bias=True)
        else:
            self.layer_norm_a = LayerNorm(c_q)

        self.layer_norm_z = LayerNorm(
            c_z, create_offset=not use_ada_layer_norm,
        )
        self.linear_z = Linear(c_z, no_heads, bias=False)

        self.sigmoid = Sigmoid()

        self.mha = OF3Attention(
            c_q=c_q, c_k=c_k, c_v=c_v,
            c_hidden=c_hidden, no_heads=no_heads, gating=gating,
            q_bias=True,
        )

        self._fk = None
        self.register_load_state_dict_post_hook(type(self)._fk_hook)

    # -- fused path ---------------------------------------------------------
    def _fk_setup(self):
        m = self.mha
        h, d = m.no_heads, m.c_hidden
        dp = (d + _ALIGN - 1) // _ALIGN * _ALIGN
        c, cs, cz = self.c_q, self.c_s, self.c_z
        if _C is None or not _all_bf16_cuda(self):
            return False
        # The kernels tile C, c_s and h*dp in 16s and use one wmma M tile per 16
        # tokens; anything else is not worth a special case here.
        if (c % _ALIGN or (h * dp) % _ALIGN or m.c_k != c or m.c_v != c
                or (self.use_ada_layer_norm and cs % _ALIGN)):
            return False

        gating = m.linear_g is not None
        lnz = self.layer_norm_z
        flags = (_F_ADA if self.use_ada_layer_norm else 0)
        flags |= _F_GATING if gating else 0
        flags |= _F_LNZ_B if lnz.bias is not None else 0

        planes = [_heads_padded(m.linear_q.weight, h, d, dp),
                  _heads_padded(m.linear_k.weight, h, d, dp),
                  _heads_padded(m.linear_v.weight, h, d, dp)]
        if gating:
            planes.append(_heads_padded(m.linear_g.weight, h, d, dp))
        parts: list[torch.Tensor | None] = [
            _tile16(torch.cat(planes, 0)),
            _bias_padded(m.linear_q.bias, h, d, dp),
            _tile16(_out_padded(m.linear_o.weight, h, d, dp)),
            self.linear_z.weight,
            lnz.weight,
            lnz.bias,
        ]
        if self.use_ada_layer_norm:
            ada = self.layer_norm_a
            if ada.layer_norm_s.weight is None:
                return False
            flags |= _F_LNS_B if ada.layer_norm_s.bias is not None else 0
            parts += [
                ada.layer_norm_s.weight,
                ada.layer_norm_s.bias,
                _tile16(torch.cat([ada.linear_g.weight, ada.linear_s.weight], 0)),
                ada.linear_g.bias,
                _tile16(self.linear_ada_out.weight),
                self.linear_ada_out.bias,
            ]
        else:
            ln = self.layer_norm_a
            flags |= _F_LNA_W if ln.weight is not None else 0
            flags |= _F_LNA_B if ln.bias is not None else 0
            parts += [ln.weight, ln.bias]

        empty = torch.empty(0, dtype=torch.bfloat16, device=m.linear_q.weight.device)
        tail = (c, cs, cz, h, d, float(self.layer_norm_z.eps), float(self.inf), flags)
        return (_C.apb_forward, _pack(parts), empty, tail)

    def _prep_bias(
        self, a: torch.Tensor, z: torch.Tensor, mask: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        if mask is None:
            mask = a.new_ones(a.shape[:-1])

        batch_dims = a.shape[:-2]
        mask = mask.expand((*batch_dims, -1))

        mask_bias = (self.inf * (mask - 1))[..., None, None, :]
        biases = [mask_bias]

        z = self.layer_norm_z(z)
        z = self.linear_z(z)
        z = _permute_final_dims(z, [2, 0, 1])
        biases.append(z)

        return biases

    def forward(
        self,
        a: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        use_high_precision_attention: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            a:    [*, N, C_q] token/atom-level embedding
            z:    [*, N, N, C_z] pair embedding
            s:    [*, N, C_s] single embedding (for AdaLN)
            mask: [*, N] mask

        Returns:
            [*, N, C_q] attention update
        """
        st = self._fk
        if st is None:
            st = self._fk = self._fk_setup()
        if st and not torch.is_grad_enabled():
            e = st[2]
            out = st[0](st[1], a, z, e if s is None else s,
                        e if mask is None else mask, *st[3])
            if out is not None:
                return out

        biases = self._prep_bias(a=a, z=z, mask=mask)

        a = self.layer_norm_a(a, s) if self.use_ada_layer_norm else self.layer_norm_a(a)

        a = self.mha(q_x=a, kv_x=a, biases=biases)

        if self.use_ada_layer_norm:
            a = self.sigmoid(self.linear_ada_out(s)) * a

        return a


class CrossAttentionPairBias(_FusedBase):
    """AF3 Algorithm 24: Cross-attention with pair bias for atom attention.

    Uses separate layer_norm_a_q and layer_norm_a_k for query/key, and
    does NOT apply layer_norm_z (pair bias goes through linear_z directly).
    Handles sequence-local blocked inputs.

    Reference: openfold3/core/model/layers/attention_pair_bias.py CrossAttentionPairBias

    Args:
        c_q: Input dimension of query/key/value
        c_s: Single activation channel dimension (for AdaLN)
        c_z: Pair activation channel dimension
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        use_ada_layer_norm: Whether to use AdaLN-Zero conditioning
        n_query: Block size for queries
        n_key: Block size for keys
        gating: Whether to gate output
        inf: Large constant for masking
    """

    def __init__(
        self,
        c_q: int,
        c_k: int = 0,
        c_v: int = 0,
        c_s: int = 0,
        c_z: int = 16,
        c_hidden: int = 16,
        no_heads: int = 4,
        use_ada_layer_norm: bool = True,
        n_query: int | None = None,
        n_key: int | None = None,
        gating: bool = True,
        inf: float = 1e9,
    ):
        super().__init__()
        c_k = c_k or c_q
        c_v = c_v or c_q

        self.c_q = c_q
        self.c_s = c_s
        self.c_z = c_z
        self.inf = inf
        self.use_ada_layer_norm = use_ada_layer_norm
        self.n_query = n_query
        self.n_key = n_key

        if use_ada_layer_norm:
            self.layer_norm_a_q = AdaLN(c_a=c_q, c_s=c_s)
            self.layer_norm_a_k = AdaLN(c_a=c_q, c_s=c_s)
            self.linear_ada_out = Linear(c_s, c_q, bias=True)
        else:
            self.layer_norm_a_q = LayerNorm(c_q)
            self.layer_norm_a_k = LayerNorm(c_q)

        self.linear_z = Linear(c_z, no_heads, bias=False)

        self.sigmoid = Sigmoid()

        self.mha = OF3Attention(
            c_q=c_q, c_k=c_k, c_v=c_v,
            c_hidden=c_hidden, no_heads=no_heads, gating=gating,
            q_bias=True,
        )

        self._fk = None
        self.register_load_state_dict_post_hook(type(self)._fk_hook)

    # -- fused path ---------------------------------------------------------
    def _fk_setup(self):
        m = self.mha
        h, d = m.no_heads, m.c_hidden
        dp = (d + _ALIGN - 1) // _ALIGN * _ALIGN
        c, cs, cz = self.c_q, self.c_s, self.c_z
        nq, nk = self.n_query, self.n_key
        if _C is None or not _all_bf16_cuda(self):
            return False
        if (not nq or not nk or nq % _ALIGN or c % _ALIGN or (h * dp) % _ALIGN
                or m.c_k != c or m.c_v != c or cz > 32
                or (self.use_ada_layer_norm and cs % _ALIGN)):
            return False

        gating = m.linear_g is not None
        flags = (_F_ADA if self.use_ada_layer_norm else 0)
        flags |= _F_GATING if gating else 0

        qg = [_heads_padded(m.linear_q.weight, h, d, dp)]
        if gating:
            qg.append(_heads_padded(m.linear_g.weight, h, d, dp))
        parts: list[torch.Tensor | None] = [
            _tile16(torch.cat(qg, 0)),
            _bias_padded(m.linear_q.bias, h, d, dp),
            _tile16(torch.cat([_heads_padded(m.linear_k.weight, h, d, dp),
                               _heads_padded(m.linear_v.weight, h, d, dp)], 0)),
            # The cross-attention output projection runs as a plain reduction
            # rather than through wmma (only QS=4 query rows per block), so it
            # wants [HDp][C] -- lanes walking c then read contiguous weights.
            _out_padded(m.linear_o.weight, h, d, dp).t().contiguous(),
            self.linear_z.weight,
        ]
        if self.use_ada_layer_norm:
            aq, ak = self.layer_norm_a_q, self.layer_norm_a_k
            if aq.layer_norm_s.weight is None or ak.layer_norm_s.weight is None:
                return False
            flags |= _F_LNS_B if aq.layer_norm_s.bias is not None else 0
            flags |= _F_LNSK_B if ak.layer_norm_s.bias is not None else 0
            parts += [
                aq.layer_norm_s.weight, aq.layer_norm_s.bias,
                _tile16(torch.cat([aq.linear_g.weight, aq.linear_s.weight], 0)),
                aq.linear_g.bias,
                ak.layer_norm_s.weight, ak.layer_norm_s.bias,
                _tile16(torch.cat([ak.linear_g.weight, ak.linear_s.weight], 0)),
                ak.linear_g.bias,
                _tile16(self.linear_ada_out.weight), self.linear_ada_out.bias,
            ]
        else:
            lq, lk = self.layer_norm_a_q, self.layer_norm_a_k
            flags |= _F_LNA_W if lq.weight is not None else 0
            flags |= _F_LNA_B if lq.bias is not None else 0
            flags |= _F_LNK_W if lk.weight is not None else 0
            flags |= _F_LNK_B if lk.bias is not None else 0
            parts += [lq.weight, lq.bias, lk.weight, lk.bias]

        eps = (self.layer_norm_a_q.layer_norm_s.eps if self.use_ada_layer_norm
               else self.layer_norm_a_q.eps)
        empty = torch.empty(0, dtype=torch.bfloat16, device=m.linear_q.weight.device)
        tail = (c, cs, cz, h, d, nq, nk, float(eps), float(self.inf), flags)
        return (_C.xapb_forward, _pack(parts), empty, tail)

    def forward(
        self,
        a: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            a:    [*, N_atom, C_q] atom-level embedding
            z:    [*, N_blocks, n_key, n_key, C_z] blocked pair embedding
            s:    [*, N_atom, C_s] single embedding (for AdaLN)
            mask: [*, N_atom] mask

        Returns:
            [*, N_atom, C_q] attention update
        """
        st = self._fk
        if st is None:
            st = self._fk = self._fk_setup()
        if st and not torch.is_grad_enabled():
            e = st[2]
            out = st[0](st[1], a, z, e if s is None else s,
                        e if mask is None else mask, *st[3])
            if out is not None:
                return out

        from .alphafold3_atom_attention import (
            _convert_single_rep_to_blocks, _apply_block_indices,
        )

        batch_dims = a.shape[:-2]
        n_atom, n_dim = a.shape[-2:]

        if mask is None:
            mask = a.new_ones(a.shape[:-1])

        a_query, a_key, block_mask = _convert_single_rep_to_blocks(
            ql=a, n_query=self.n_query, n_key=self.n_key, atom_mask=mask,
        )

        mask_bias = (self.inf * (block_mask - 1))[..., None, :, :]
        biases = [mask_bias]

        z_bias = self.linear_z(z)
        z_bias = _permute_final_dims(z_bias, [2, 0, 1])
        biases.append(z_bias)

        if self.use_ada_layer_norm:
            s_q, s_k, _ = _apply_block_indices(
                ql=s, n_query=self.n_query, n_key=self.n_key, atom_mask=mask,
            )
            a_q = self.layer_norm_a_q(a_query, s_q)
            a_k = self.layer_norm_a_k(a_key, s_k)
        else:
            a_q = self.layer_norm_a_q(a_query)
            a_k = self.layer_norm_a_k(a_key)

        a_out = self.mha(q_x=a_q, kv_x=a_k, biases=biases)

        a_out = a_out.reshape((*batch_dims, -1, n_dim))[..., :n_atom, :]

        if self.use_ada_layer_norm:
            a_out = self.sigmoid(self.linear_ada_out(s)) * a_out

        return a_out
