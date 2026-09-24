"""Diffusion transformer for AlphaFold3 -- one kernel launch per geometry.

24-block transformer used inside the diffusion module. Each block:
AttentionPairBias + ConditionedTransitionBlock (AdaLN-Zero).

Reference: openfold3/core/model/layers/diffusion_transformer.py

Why this is written the way it is
--------------------------------
Two geometries are captured and they are different problems, so there are two
kernels.  What they share is that neither is anywhere near a FLOP limit: the
token-level stack is ~0.9 GFLOP and the atom-level one ~0.7 GFLOP, microseconds
of arithmetic each.  Composed out of the frozen L2 winners they cost 2.6 ms and
0.24 ms, spread over 144 and 16 launches.  Both are therefore launch- and
synchronization-bound, and both collapse to a single persistent launch.

Three measurements on this B200 shape every decision below:

* a kernel launch costs ~4.5 us in the bench's own timing loop, a device-wide
  barrier ~1.0 us, and a *cluster* barrier ~0.25 us;
* one SM sustains only ~22 GB/s of streaming global reads at 8 bytes per lane,
  ~45 GB/s at 16 bytes -- so how wide the weight stream is, and how much of it
  is in flight, matters more than anything about the arithmetic;
* left alone, ptxas will sink a batch of hoisted B-fragment loads back next to
  the mmas that consume them, which leaves one load in flight per warp and turns
  a GEMM into a chain of L2 round trips.  The empty ``asm volatile`` with a
  memory clobber after each load batch is what stops it: worth 3x on the
  atom-level conditioners (where ptxas did sink them) and neutral in the
  token-level kernel (where it already did not) -- kept in both, since which way
  it goes is a scheduling decision that can change under any edit.

``af3_dit_fk.cu`` -- token level (``n_query is None``: 24 blocks, 16 tokens,
c_a=768).  16 tokens of activation are 24 KB against 397 MB of weights, so this
is pure weight streaming wrapped around a strictly sequential chain.  128
persistent CTAs:

* everything that depends only on ``s`` / ``z`` / ``mask`` -- both AdaLN
  conditioners, both output gates, the pair bias -- is hoisted into a prologue
  that computes all 24 blocks at once with no synchronization at all.  That is
  21% of the weights moved off the dependency chain.
* the remaining per-block chain has exactly four points where a reduction spans
  the channel dimension (q/k/v/g, linear_o, the SwiGLU hidden layer,
  linear_out), so it costs four grid barriers per block.
* activations are small enough to replicate, so every CTA keeps the token tensor
  in shared memory and *recomputes* every LayerNorm and elementwise step instead
  of synchronizing on it.  One warp owns one token row, which makes each
  LayerNorm statistic a warp shuffle, and every elementwise step the reference
  performs on bf16 tensors is one ``mul.rn.bf16x2`` / ``add.rn.bf16x2`` -- which
  rounds once, exactly where the reference rounds, at a quarter of the
  instructions.  That last point alone was 1.8x on this geometry.
* linear_o and linear_out close with fp32 atomics into a small accumulator, so
  splitting their reductions across CTAs costs no extra barrier.

``af3_dit_cross.cu`` -- atom level (``n_query=32``: 3 blocks, 368 atoms,
c_a=128, plus a 1.5 MB blocked pair tensor).  Here the weights are only 1.8 MB,
so the per-SM read limit above says a design where every CTA reads all of them
costs ~35 us on the weight stream alone -- they have to be split by output
column, which means the split has to be sewn back together wherever the next
GEMM needs a full reduction dimension.  Cluster barriers and distributed shared
memory make that nearly free: 96 CTAs in 24 clusters of 4, cluster ``mt`` owning
atom rows ``[16mt, 16mt+16)`` (always inside one key block, since n_query=32)
and rank ``cs`` owning channels ``[32cs, 32cs+32)`` -- which, for c_hidden=32, is
exactly attention head ``cs``.  Only the k/v projections cross cluster
boundaries, because a key block gathers atoms from anywhere, so there is exactly
one device-wide barrier per block against five cluster barriers.

Anything the fast paths do not cover -- other geometries, non-bf16, a real batch
dimension, autograd -- falls through to the reference composition below, which is
the baseline code unchanged.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..L2.alphafold3_attention_pair_bias import AttentionPairBias, CrossAttentionPairBias
from ..L2.alphafold3_swiglu_transition import ConditionedTransitionBlock


__targets__ = ["DiffusionTransformer"]

try:
    from fastkernels.infra.cuda_ext import load_op

    _C = load_op("af3_dit_fk", "af3_dit_fk.cu")
except Exception:  # noqa: BLE001 -- no nvcc / no GPU: reference path only
    _C = None

try:
    from fastkernels.infra.cuda_ext import load_op as _load_op_x

    _X = _load_op_x("af3_dit_cross", "af3_dit_cross.cu")
except Exception:  # noqa: BLE001
    _X = None

# Weight sections, in the order ``af3_dit_fk.cu``'s ``W_*`` enum expects them.
_SECT = ("LN1", "GS1", "BG1", "AO", "BAO", "LN2", "GS2", "BG2", "GC", "BGC",
         "LNZ", "Z", "QKVG", "BQ", "O", "SG", "OUT")

# The one self-attention geometry in the captures; the kernel is a compile-time
# template on it (a runtime ``/ c_a`` in an elementwise loop costs more than the
# memory it addresses at these sizes).
_GEO = (16, 768, 384, 16, 48, 1536, 128)   # N, c_a, c_s, heads, c_hidden, n*c_a, c_z
_PAD = 8

# Weight sections of ``af3_dit_cross.cu``'s ``W_*`` enum, and the one cross-
# attention geometry it is templated on:
#   n_atom, n_pad, c_a, c_s, heads, c_hidden, n*c_a, c_z, n_query, n_key,
#   n_key_blocks, n_blocks
_XSECT = ("LNQ", "GSQ", "BGQ", "LNK", "GSK", "BGK", "LN2", "GS2", "BG2",
          "AO", "BAO", "GC", "BGC", "Z", "QKVG", "BQ", "O", "SG", "OUT")
_XGEO = (368, 384, 128, 128, 4, 32, 256, 16, 32, 128, 12, 3)


def _pack_b(w: torch.Tensor) -> torch.Tensor:
    """[Nout, Kin] -> ``mma.m16n8k16`` B-fragment order.

    ``pack[nt][kt][lane*4 + i]`` holds, for output column ``nt*8 + lane/4``, the
    reduction elements ``kt*16 + (lane%4)*2 + (0, 1, 8, 9)`` -- exactly the two
    B registers of one mma.  A warp's operand load is then one contiguous 256 B
    run instead of eight scattered 32 B sectors out of a row-major matrix, which
    is worth ~10x on these shapes.
    """
    n, k = w.shape
    assert n % 8 == 0 and k % 16 == 0, (n, k)
    return (w.detach().reshape(n // 8, 8, k // 16, 2, 4, 2)
            .permute(0, 2, 1, 4, 3, 5).contiguous().reshape(-1))


def _pack_b2(w: torch.Tensor) -> torch.Tensor:
    """[Nout, Kin] -> ``mma.m16n8k16`` B-fragment order, two k-tiles per lane slot.

    The same fragment order as :func:`_pack_b`, except that the fragments of
    k-tiles ``2t`` and ``2t+1`` sit adjacently inside each lane's slot, so one
    16-byte load feeds two mmas.  ``af3_dit_cross.cu`` streams its weights that
    way; the token-level kernel keeps the 8-byte layout.
    """
    n, k = w.shape
    assert n % 8 == 0 and k % 32 == 0, (n, k)
    cur = (w.detach().reshape(n // 8, 8, k // 16, 2, 4, 2)
           .permute(0, 2, 1, 4, 3, 5).contiguous()
           .reshape(n // 8, k // 32, 2, 32, 4))
    return cur.permute(0, 1, 3, 2, 4).contiguous().reshape(-1)


class DiffusionTransformerBlock(nn.Module):
    """AF3 Algorithm 23: Diffusion transformer block.

    Args:
        c_a: Token activation channel dimension
        c_s: Single activation channel dimension
        c_z: Pair activation channel dimension
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        n_transition: Transition layer scale
        use_ada_layer_norm: Whether to use AdaLN-Zero
        inf: Large masking constant
    """

    def __init__(
        self,
        c_a: int,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        n_transition: int,
        use_ada_layer_norm: bool = True,
        n_query: int | None = None,
        n_key: int | None = None,
        inf: float = 1e9,
    ):
        super().__init__()
        self.use_cross_attention = n_query is not None

        if not self.use_cross_attention:
            self.attention_pair_bias = AttentionPairBias(
                c_q=c_a, c_k=c_a, c_v=c_a,
                c_s=c_s, c_z=c_z,
                c_hidden=c_hidden,
                no_heads=no_heads,
                use_ada_layer_norm=use_ada_layer_norm,
                gating=True,
                inf=inf,
            )
        else:
            self.attention_pair_bias = CrossAttentionPairBias(
                c_q=c_a, c_k=c_a, c_v=c_a,
                c_s=c_s, c_z=c_z,
                c_hidden=c_hidden,
                no_heads=no_heads,
                use_ada_layer_norm=use_ada_layer_norm,
                n_query=n_query,
                n_key=n_key,
                gating=True,
                inf=inf,
            )

        self.conditioned_transition = ConditionedTransitionBlock(
            c_a=c_a, c_s=c_s, n=n_transition,
        )

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        use_high_precision_attention: bool = False,
        _mask_trans: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            a:    [*, N, C_token] token-level embedding
            s:    [*, N, C_s] single embedding
            z:    [*, N, N, C_z] pair embedding
            mask: [*, N] mask

        Returns:
            [*, N, C_token] updated token embedding
        """
        a = a + self.attention_pair_bias(a=a, z=z, s=s, mask=mask)

        trans_mask = mask if _mask_trans else None
        a = a + self.conditioned_transition(a=a, s=s, mask=trans_mask)

        return a


class DiffusionTransformer(nn.Module):
    """AF3 Algorithm 23: Diffusion transformer stack.

    Args:
        c_a: Token activation channel dimension
        c_s: Single activation channel dimension
        c_z: Pair activation channel dimension
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        no_blocks: Number of transformer blocks
        n_transition: Transition layer scale
        use_ada_layer_norm: Whether to use AdaLN-Zero
        inf: Large masking constant
    """

    def __init__(
        self,
        c_a: int,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_blocks: int,
        n_transition: int,
        use_ada_layer_norm: bool = True,
        n_query: int | None = None,
        n_key: int | None = None,
        inf: float = 1e9,
        blocks_per_ckpt: int | None = None,
        **kwargs,
    ):
        super().__init__()
        from ..L1.layer_norm import LayerNorm

        self.use_cross_attention = n_query is not None
        if self.use_cross_attention:
            self.layer_norm_z = LayerNorm(c_z, create_offset=False)

        self.c_a, self.c_s, self.c_z = c_a, c_s, c_z
        self.c_hidden, self.no_heads = c_hidden, no_heads
        self.n_transition, self.no_blocks = n_transition, no_blocks
        self.use_ada_layer_norm = use_ada_layer_norm
        self.inf = inf

        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(
                c_a=c_a, c_s=c_s, c_z=c_z,
                c_hidden=c_hidden, no_heads=no_heads,
                n_transition=n_transition,
                use_ada_layer_norm=use_ada_layer_norm,
                n_query=n_query,
                n_key=n_key,
                inf=inf,
            )
            for _ in range(no_blocks)
        ])

        self._fk = None
        self.register_load_state_dict_post_hook(type(self)._fk_hook)

    # -- fused-path cache management ---------------------------------------
    # The packed buffer holds a *copy* of the weights, so it has to be dropped
    # whenever the parameters change: ``_apply`` covers ``.to()`` / ``.cuda()`` /
    # ``.half()`` and the load-state-dict hook covers weight loading.
    def _apply(self, *args, **kwargs):
        out = super()._apply(*args, **kwargs)
        self._fk = None
        return out

    def _fk_hook(self, *args, **kwargs) -> None:
        self._fk = None

    def _fk_setup(self):
        """Pack the token-level stack for ``af3_dit_fk.cu``."""
        if _C is None or self.use_cross_attention or not self.use_ada_layer_norm:
            return False
        ps = list(self.parameters())
        if not ps or not all(p.is_cuda and p.dtype is torch.bfloat16 for p in ps):
            return False
        c_a, c_s, c_hidden, h = self.c_a, self.c_s, self.c_hidden, self.no_heads
        geo = (16, c_a, c_s, h, c_hidden, self.n_transition * c_a, self.c_z)
        if geo != _GEO or h * c_hidden != c_a:
            return False
        dev = ps[0].device

        eps = self.blocks[0].attention_pair_bias.layer_norm_z.eps
        secs = []
        for blk in self.blocks:
            apb, ctb = blk.attention_pair_bias, blk.conditioned_transition
            ada, ada2, m = apb.layer_norm_a, ctb.layer_norm, apb.mha
            if (ada.layer_norm_s.weight is None or ada.layer_norm_s.bias is not None
                    or ada2.layer_norm_s.weight is None
                    or ada2.layer_norm_s.bias is not None
                    or apb.layer_norm_z.weight is None
                    or apb.layer_norm_z.bias is not None
                    or m.linear_g is None or m.linear_q.bias is None):
                return False
            if any(x.eps != eps for x in (apb.layer_norm_z, ada.layer_norm_s,
                                          ada2.layer_norm_s, ada.layer_norm_a,
                                          ada2.layer_norm_a)):
                return False
            ab = torch.stack([ctb.swiglu.linear_a.weight,
                              ctb.swiglu.linear_b.weight], 1).reshape(-1, c_a)
            secs.append({
                "LN1": ada.layer_norm_s.weight,
                "GS1": _pack_b(torch.cat([ada.linear_g.weight,
                                          ada.linear_s.weight], 0)),
                "BG1": ada.linear_g.bias,
                "AO": _pack_b(apb.linear_ada_out.weight),
                "BAO": apb.linear_ada_out.bias,
                "LN2": ada2.layer_norm_s.weight,
                "GS2": _pack_b(torch.cat([ada2.linear_g.weight,
                                          ada2.linear_s.weight], 0)),
                "BG2": ada2.linear_g.bias,
                "GC": _pack_b(ctb.linear_g.weight),
                "BGC": ctb.linear_g.bias,
                "LNZ": apb.layer_norm_z.weight,
                "Z": _pack_b(apb.linear_z.weight),
                "QKVG": _pack_b(torch.cat([m.linear_q.weight, m.linear_k.weight,
                                           m.linear_v.weight,
                                           m.linear_g.weight], 0)),
                "BQ": m.linear_q.bias,
                "O": _pack_b(m.linear_o.weight),
                "SG": _pack_b(ab),
                "OUT": _pack_b(ctb.linear_out.weight),
            })

        offs, off = [], 0
        for name in _SECT:
            offs.append(off)
            off += secs[0][name].numel()
        stride = (off + 127) // 128 * 128
        nb = len(secs)
        wp = torch.zeros(nb * stride, dtype=torch.bfloat16, device=dev)
        for i, sd in enumerate(secs):
            for o, name in zip(offs, _SECT):
                v = sd[name].detach().reshape(-1)
                wp[i * stride + o: i * stride + o + v.numel()] = v

        n, _, _, _, c_hid, f, c_z = _GEO
        z16 = lambda k: torch.zeros(k, dtype=torch.bfloat16, device=dev)  # noqa: E731
        state = dict(
            w=wp,
            empty=torch.empty(0, dtype=torch.bfloat16, device=dev),
            offs=torch.tensor(offs, dtype=torch.int64),
            # One spare block of conditioners: the last block's residual pass
            # reads the next block's AdaLN slot unconditionally.
            cond=z16((nb + 1) * 6 * n * c_a),
            zb=z16(nb * h * n * n),
            qkvg=z16(_C.dit_self_qkvg(n, h, c_hid)),
            hid=z16(n * f),
            acc=torch.zeros(4 * n * c_a, dtype=torch.float32, device=dev),
            sync=torch.zeros(2, dtype=torch.int32, device=dev),
            stride=stride,
            eps=float(eps),
            qdiv=math.sqrt(c_hid),
            inf=float(self.inf),
        )
        return state

    def _fk_forward(self, st, a, s, z, mask):
        n, c_a, c_s, h, c_hid, f, c_z = _GEO
        if (a.shape[-2:] != (n, c_a) or s.shape[-2:] != (n, c_s)
                or z.shape[-3:] != (n, n, c_z)
                or a.numel() != n * c_a or s.numel() != n * c_s
                or z.numel() != n * n * c_z
                or a.dtype is not torch.bfloat16 or s.dtype is not torch.bfloat16
                or z.dtype is not torch.bfloat16
                or not (a.is_contiguous() and s.is_contiguous()
                        and z.is_contiguous())):
            return None
        if mask is None:
            mk = st["empty"]
        elif (mask.dtype is not torch.bfloat16 or mask.numel() != n
              or not mask.is_contiguous()):
            return None
        else:
            mk = mask
        out = _C.dit_self_fwd(
            a, s, z, mk,
            st["w"], st["offs"], st["cond"], st["zb"], st["qkvg"], st["hid"],
            st["acc"], st["sync"], n, c_a, c_s, h, c_hid, f, c_z,
            self.no_blocks, st["stride"], st["eps"], st["qdiv"], st["inf"],
            0 if mask is None else 1, 0)
        return out.reshape(a.shape)

    # -- cross-attention (atom-level) fast path -----------------------------
    def _xk_setup(self):
        """Pack the atom-level stack for ``af3_dit_cross.cu``."""
        if _X is None or not self.use_cross_attention or not self.use_ada_layer_norm:
            return False
        ps = list(self.parameters())
        if not ps or not all(p.is_cuda and p.dtype is torch.bfloat16 for p in ps):
            return False
        na, npad, c_a, c_s, h, c_hid, f, c_z, nq, nk, nbk, nb = _XGEO
        if ((c_a, c_s, c_z, c_hid, h, self.n_transition * c_a, self.no_blocks)
                != (self.c_a, self.c_s, self.c_z, self.c_hidden, self.no_heads,
                    f, nb)):
            return False
        blk0 = self.blocks[0].attention_pair_bias
        if getattr(blk0, "n_query", None) != nq or getattr(blk0, "n_key", None) != nk:
            return False
        dev = ps[0].device

        eps = self.layer_norm_z.eps
        if self.layer_norm_z.weight is None or self.layer_norm_z.bias is not None:
            return False
        secs = []
        for blk in self.blocks:
            apb, ctb = blk.attention_pair_bias, blk.conditioned_transition
            aq, ak, ad2, m = (apb.layer_norm_a_q, apb.layer_norm_a_k,
                              ctb.layer_norm, apb.mha)
            for ada in (aq, ak, ad2):
                if (ada.layer_norm_s.weight is None
                        or ada.layer_norm_s.bias is not None
                        or ada.layer_norm_a.weight is not None):
                    return False
            if (m.linear_g is None or m.linear_q.bias is None
                    or m.linear_k.bias is not None or m.linear_v.bias is not None
                    or m.linear_o.bias is not None or ctb.linear_out.bias is not None):
                return False
            if any(x.eps != eps for x in (aq.layer_norm_s, ak.layer_norm_s,
                                          ad2.layer_norm_s, aq.layer_norm_a,
                                          ad2.layer_norm_a)):
                return False
            ab = torch.stack([ctb.swiglu.linear_a.weight,
                              ctb.swiglu.linear_b.weight], 1).reshape(-1, c_a)
            secs.append({
                "LNQ": aq.layer_norm_s.weight,
                "GSQ": _pack_b2(torch.cat([aq.linear_g.weight,
                                          aq.linear_s.weight], 0)),
                "BGQ": aq.linear_g.bias,
                "LNK": ak.layer_norm_s.weight,
                "GSK": _pack_b2(torch.cat([ak.linear_g.weight,
                                          ak.linear_s.weight], 0)),
                "BGK": ak.linear_g.bias,
                "LN2": ad2.layer_norm_s.weight,
                "GS2": _pack_b2(torch.cat([ad2.linear_g.weight,
                                          ad2.linear_s.weight], 0)),
                "BG2": ad2.linear_g.bias,
                "AO": _pack_b2(apb.linear_ada_out.weight),
                "BAO": apb.linear_ada_out.bias,
                "GC": _pack_b2(ctb.linear_g.weight),
                "BGC": ctb.linear_g.bias,
                # linear_z stays row-major: c_z = 16 is one short fp32 dot.
                "Z": apb.linear_z.weight,
                "QKVG": _pack_b2(torch.cat([m.linear_q.weight, m.linear_k.weight,
                                           m.linear_v.weight,
                                           m.linear_g.weight], 0)),
                "BQ": m.linear_q.bias,
                "O": _pack_b2(m.linear_o.weight),
                "SG": _pack_b2(ab),
                "OUT": _pack_b2(ctb.linear_out.weight),
            })

        offs, off = [], 0
        for name in _XSECT:
            offs.append(off)
            off += secs[0][name].numel()
        stride = (off + 127) // 128 * 128
        wp = torch.zeros(len(secs) * stride, dtype=torch.bfloat16, device=dev)
        for i, sd in enumerate(secs):
            for o, name in zip(offs, _XSECT):
                v = sd[name].detach().reshape(-1)
                wp[i * stride + o: i * stride + o + v.numel()] = v
        return dict(
            w=wp,
            empty=torch.empty(0, dtype=torch.bfloat16, device=dev),
            offs=torch.tensor(offs, dtype=torch.int64),
            lnz=self.layer_norm_z.weight.detach().reshape(-1).contiguous(),
            # k / v are the only tensors a key block reads across cluster
            # boundaries; double-buffered so the next block's projections cannot
            # race the current block's gather.
            kv=torch.zeros(2 * 2 * npad * c_a, dtype=torch.bfloat16, device=dev),
            sync=torch.zeros(2, dtype=torch.int32, device=dev),
            stride=stride,
            eps=float(eps),
            qdiv=math.sqrt(c_hid),
            inf=float(self.inf),
        )

    def _xk_forward(self, st, a, s, z, mask):
        # The bench times the module, not the kernel, and this stack is ~80 us of
        # device time -- so the guard is written to cost as little Python as it
        # can: no ``math.prod``, no ``reshape`` (the kernel only wants
        # ``data_ptr``), and a cached empty tensor for the no-mask case.
        na, npad, c_a, c_s, h, c_hid, f, c_z, nq, nk, nbk, nb = _XGEO
        if (a.shape[-2:] != (na, c_a) or s.shape[-2:] != (na, c_s)
                or z.shape[-4:] != (nbk, nq, nk, c_z)
                or a.numel() != na * c_a or s.numel() != na * c_s
                or z.numel() != nbk * nq * nk * c_z
                or a.dtype is not torch.bfloat16 or s.dtype is not torch.bfloat16
                or z.dtype is not torch.bfloat16
                or not (a.is_contiguous() and s.is_contiguous()
                        and z.is_contiguous())):
            return None
        if mask is None:
            mk = st["empty"]
        elif (mask.dtype is not torch.bfloat16 or mask.numel() != na
              or not mask.is_contiguous()):
            return None
        else:
            mk = mask
        out = _X.dit_cross_fwd(
            a, s, z, mk, st["lnz"], st["w"], st["offs"], st["kv"], st["sync"],
            self.no_blocks, st["stride"], st["eps"], st["qdiv"], st["inf"],
            0 if mask is None else 1, 0)
        return out.reshape(a.shape)

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        use_high_precision_attention: bool = False,
        _mask_trans: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            a:    [*, N, C_token] token-level embedding
            s:    [*, N, C_s] single embedding
            z:    [*, N, N, C_z] pair embedding
            mask: [*, N] mask

        Returns:
            [*, N, C_token] updated token embedding
        """
        if _mask_trans and not torch.is_grad_enabled():
            st = self._fk
            if st is None:
                st = self._fk = (self._xk_setup() if self.use_cross_attention
                                 else self._fk_setup())
            if st:
                out = (self._xk_forward(st, a, s, z, mask)
                       if self.use_cross_attention
                       else self._fk_forward(st, a, s, z, mask))
                if out is not None:
                    return out

        if self.use_cross_attention:
            z = self.layer_norm_z(z)

        for block in self.blocks:
            a = block(a=a, s=s, z=z, mask=mask)

        return a
