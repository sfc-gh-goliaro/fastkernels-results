"""Input embedder for AlphaFold3.

Produces initial single (s) and pair (z) representations from token and
atom features.

Reference: openfold3/core/model/feature_embedders/input_embedders.py
           InputEmbedderAllAtom
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.tensor_ops import OneHot, Pad
from .alphafold3_atom_attention import (
    AtomAttentionEncoder,
    _aggregate_atom_feat_to_tokens,
    _get_block_key_indices,
)
from .alphafold3_attention_pair_bias import _permute_final_dims


_RELPOS_KEYS = frozenset(
    {"residue_index", "token_index", "asym_id", "entity_id", "sym_id"},
)

# Rounding mode for the relative-offset arithmetic, so the fused kernel
# reproduces the reference's *intermediate* narrow-dtype rounding bit for bit
# (PyTorch computes `a - b` on bf16 by promoting to fp32 and rounding the
# result back down; a fused fp32 chain would occasionally land on the other
# side of an integer boundary and pick a different thermometer bin).
_ROUND_NONE, _ROUND_BF16, _ROUND_FP16 = 0, 1, 2
_ROUND_BY_DTYPE = {torch.bfloat16: _ROUND_BF16, torch.float16: _ROUND_FP16}

# CUDA-graph capture of the whole forward (see InputEmbedder._replay).
_GRAPHS_ENABLED = os.environ.get("AF3_IE_NO_CUDA_GRAPH", "") != "1"
# Eager calls to let through before capturing, so weight loading and the lazy
# weight-derived caches have certainly settled.
_GRAPH_WARMUP_CALLS = 2
# Sentinels naming the two positional forward arguments in the graph's copy-in
# slot list (`batch` keys are named by their own string, and a batch cannot
# contain these because they are not strings).
_TF, _RI = object(), object()


class _ReadRecorder(dict):
    """A `batch` wrapper that remembers whose *values* the forward consumed.

    The graph only has to refresh the inputs the traced forward actually reads;
    the rest are dead weight in the copy-in. Which ones those are is decided
    mechanically here rather than hardcoded, so the set stays correct if the
    encoder path changes.

    Presence-only accessors (`in`, `keys()`, `len()`) deliberately do *not*
    record -- `_RELPOS_KEYS.issubset(batch.keys())` and `"ref_pos" in batch`
    only branch on which keys exist, and the key set is validated separately on
    every replay. Anything that hands out a value records, and the bulk
    accessors record everything. Overriding `__iter__` also forces
    `dict(batch)` / `{**batch}` off CPython's fast path and through
    `keys()` + `__getitem__`, so those get recorded too.
    """

    __slots__ = ("read",)

    def __init__(self, base):
        super().__init__(base)
        self.read = set()

    def __getitem__(self, k):
        self.read.add(k)
        return super().__getitem__(k)

    def get(self, k, default=None):
        self.read.add(k)
        return super().get(k, default)

    def pop(self, k, *a):
        self.read.add(k)
        return super().pop(k, *a)

    def setdefault(self, k, default=None):
        self.read.add(k)
        return super().setdefault(k, default)

    def values(self):
        self.read.update(super().keys())
        return super().values()

    def items(self):
        self.read.update(super().keys())
        return super().items()

    def __iter__(self):
        # Iteration cannot tell us what happens to the values afterwards.
        self.read.update(super().keys())
        return super().__iter__()


class _GraphPlan:
    """A captured graph plus everything `_replay` needs to drive it.

    One pass per call: `gather` walks the copy-in destinations once, validating
    each source against the static buffer it is about to be copied into *and*
    collecting it, so a graph hit costs one traversal and never materializes a
    key tuple. Copies are issued only after the whole pass validates.
    """

    __slots__ = ("graph", "groups", "static_out", "out_meta", "out_total",
                 "out_dtype", "out_device", "tf_sig", "ri_sig", "keys", "others")

    def __init__(self, graph, groups, outputs, tf_sig, ri_sig, keys, others):
        self.graph = graph
        self.groups = groups
        self.static_out = list(outputs)
        self.tf_sig = tf_sig
        self.ri_sig = ri_sig
        self.keys = keys          # frozenset(batch) at capture, or None
        self.others = others      # ((key, repr(non-tensor value)), ...)
        dt = outputs[0].dtype
        dev = outputs[0].device
        if all(o.dtype is dt and o.device == dev and o.is_contiguous()
               for o in outputs):
            self.out_dtype, self.out_device = dt, dev
            self.out_meta = tuple((tuple(o.shape), o.numel()) for o in outputs)
            self.out_total = sum(n for _, n in self.out_meta)
        else:
            self.out_total = None

    def copied_names(self):
        return {n for _, names in self.groups for n in names
                if isinstance(n, str)}

    def gather(self, token_features, residue_index, batch):
        """One validating pass. Returns the per-group source lists, or None."""
        if _sig(token_features) != self.tf_sig or _sig(residue_index) != self.ri_sig:
            return None
        if batch is None:
            if self.keys is not None:
                return None
        else:
            if self.keys is None or batch.keys() != self.keys:
                return None
            for k, r in self.others:
                if repr(batch[k]) != r:
                    return None
        out = []
        for dsts, names in self.groups:
            srcs = []
            for dst, n in zip(dsts, names):
                src = (token_features if n is _TF
                       else residue_index if n is _RI else batch[n])
                if not isinstance(src, torch.Tensor):
                    return None
                if (src.shape != dst.shape or src.dtype is not dst.dtype
                        or src.device != dst.device or not src.is_contiguous()):
                    return None
                srcs.append(src)
            out.append(srcs)
        return out

    def replay(self, srcs):
        for (dsts, _), s in zip(self.groups, srcs):
            torch._foreach_copy_(dsts, s)
        self.graph.replay()
        return self.results()

    def results(self):
        """Fresh copies of the outputs -- the graph's own buffers are overwritten
        by the next replay, and callers may hold a result across a forward.

        The three outputs share a dtype and device, so one fresh allocation plus
        one `_foreach_copy_` replaces three `clone()`s (three dispatches, three
        allocations and three D2D memcpys). The profiler makes this look like a
        loss -- `multi_tensor_apply_kernel` is 6.5 µs against 3 x 1.5 µs of
        `Memcpy DtoD` -- but end to end it measured 4 µs *better* (iter09), which
        is what happens when the memcpys serialize against the replay while the
        one launch does not.
        """
        if self.out_total is None:
            return tuple(o.clone() for o in self.static_out)
        flat = torch.empty(self.out_total, dtype=self.out_dtype,
                           device=self.out_device)
        outs = []
        off = 0
        for shape, n in self.out_meta:
            outs.append(flat.narrow(0, off, n).view(shape))
            off += n
        torch._foreach_copy_(outs, self.static_out)
        return tuple(outs)


def _sig(t):
    return (tuple(t.shape), t.dtype, t.device, t.is_contiguous())


@triton.jit
def _narrow(x, ROUND: tl.constexpr):
    """Round `x` (fp32) through the reference's storage dtype and back."""
    if ROUND == 1:
        return x.to(tl.bfloat16).to(tl.float32)
    if ROUND == 2:
        return x.to(tl.float16).to(tl.float32)
    return x


@triton.jit
def _rnd(x, DT: tl.constexpr):
    """Round an fp32 value through the output storage dtype and back."""
    return x.to(DT).to(tl.float32)


@triton.jit
def _rnd_hard(x, DT: tl.constexpr):
    """Same as `_rnd`, but the rounding survives optimization.

    `_rnd`'s `truncf`/`extf` pair is *not* always kept: when its result feeds an
    fp32 add the pair is folded away (and the surrounding mul+add then contracts
    into an FMA), so the intermediate bf16 value the reference materializes
    silently disappears. `tests/diag_res.py` pins it -- with
    `y = _rnd(g*x); y = _rnd(y + r)` the kernel returns
    `bf16(fma(g, x, r))` instead of `bf16(bf16(g*x) + r)`, which is one ulp off
    on ~20% of elements.

    Doing the round-to-nearest-even in integer arithmetic on the bit pattern
    cannot be folded. This is the same expression as PyTorch's own
    `c10::detail::round_to_nearest_even`:
    `(u + ((u >> 16) & 1) + 0x7FFF) & 0xFFFF0000`. Values here are finite (the
    only infinities in this file live in `_sdpa_kernel`, which does not use
    this), so the NaN/overflow special cases are not needed.
    """
    if DT == tl.bfloat16:
        u = x.to(tl.uint32, bitcast=True)
        u = u + (((u >> 16) & 1) + 0x7FFF)
        return (u & 0xFFFF0000).to(tl.float32, bitcast=True)
    return x.to(DT).to(tl.float32)


@triton.jit
def _relpos_bin(pos_i, pos_j, cond, CLIP: tl.constexpr, ROUND: tl.constexpr):
    """`ceil(final_offset)` -- the number of thermometer bits `_binned_one_hot`
    would set. Mirrors `relpos_complex._relpos` op for op."""
    off = _narrow(pos_i - pos_j, ROUND)
    f = _narrow(off + CLIP, ROUND)
    f = tl.minimum(tl.maximum(f, 0.0), 2.0 * CLIP)
    f = tl.where(cond, f, 2.0 * CLIP + 1.0)
    v = tl.ceil(f).to(tl.int32)
    return tl.minimum(tl.maximum(v, 0), 2 * CLIP + 1)


@triton.jit
def _z_epilogue_kernel(
    ZI, ZJ, Z,
    RES, TOK, ASYM, ENT, SYM,
    CUMP, CUMT, CUMC, WSE,
    TB, WTB,
    N, CZ,
    K: tl.constexpr, KC: tl.constexpr,
    HAS_TB: tl.constexpr, ROUND: tl.constexpr,
    BLOCK_J: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """Whole pair-representation epilogue in one pass.

        z[b,i,j,:] = z_i[b,i,:] + z_j[b,j,:]
                   + cum_pos[v_pos(i,j)] + cum_tok[v_tok(i,j)]
                   + cum_chain[v_chain(i,j)] + w_se * same_entity(i,j)
                   + w_tb * token_bonds[b,i,j]

    One program per (row, j-tile). Replaces the reference's broadcast add, the
    three thermometer one-hot materializations, the 139-wide cat, the relpos
    GEMM and the token-bond GEMM with a single kernel.
    """
    row = tl.program_id(0)          # b * N + i
    jt = tl.program_id(1)
    b = row // N
    base = b * N

    j = jt * BLOCK_J + tl.arange(0, BLOCK_J)
    jm = j < N
    c = tl.arange(0, BLOCK_C)
    cm = c < CZ

    # --- projections ---
    # Each `_rnd` marks a point where the reference materializes a bf16 tensor.
    # Keeping those rounding steps (rather than one fp32 chain) matters: where
    # z_i + z_j nearly cancels, an fp32 chain drifts several bf16 ulps away from
    # the reference's already-rounded partial sum, which shows up as >1% relative
    # error on those elements.
    zi = tl.load(ZI + row * CZ + c, mask=cm, other=0.0).to(tl.float32)
    zj = tl.load(
        ZJ + (base + j)[:, None] * CZ + c[None, :],
        mask=jm[:, None] & cm[None, :], other=0.0,
    ).to(tl.float32)
    acc = _rnd(zi[None, :] + zj, Z.dtype.element_ty)

    # --- relative-position bins ---
    res_i = tl.load(RES + row).to(tl.float32)
    res_j = tl.load(RES + base + j, mask=jm, other=0.0).to(tl.float32)
    tok_i = tl.load(TOK + row).to(tl.float32)
    tok_j = tl.load(TOK + base + j, mask=jm, other=0.0).to(tl.float32)
    asym_i = tl.load(ASYM + row).to(tl.float32)
    asym_j = tl.load(ASYM + base + j, mask=jm, other=0.0).to(tl.float32)
    ent_i = tl.load(ENT + row).to(tl.float32)
    ent_j = tl.load(ENT + base + j, mask=jm, other=0.0).to(tl.float32)
    sym_i = tl.load(SYM + row).to(tl.float32)
    sym_j = tl.load(SYM + base + j, mask=jm, other=0.0).to(tl.float32)

    same_chain = asym_i == asym_j
    same_entity = ent_i == ent_j
    same_res = res_i == res_j

    v_pos = _relpos_bin(res_i, res_j, same_chain, K, ROUND)
    v_tok = _relpos_bin(tok_i, tok_j, same_chain & same_res, K, ROUND)
    v_ch = _relpos_bin(sym_i, sym_j, same_entity, KC, ROUND)

    # `linear_relpos` is one GEMM, so its four contributions accumulate in fp32
    # and get rounded once -- mirror that here, then add as a single bf16 term.
    tm = jm[:, None] & cm[None, :]
    rel = tl.load(CUMP + v_pos[:, None] * CZ + c[None, :], mask=tm, other=0.0)
    rel += tl.load(CUMT + v_tok[:, None] * CZ + c[None, :], mask=tm, other=0.0)
    rel += tl.load(CUMC + v_ch[:, None] * CZ + c[None, :], mask=tm, other=0.0)
    w_se = tl.load(WSE + c, mask=cm, other=0.0)
    rel += tl.where(same_entity, 1.0, 0.0)[:, None] * w_se[None, :]
    acc = _rnd(acc + _rnd(rel, Z.dtype.element_ty), Z.dtype.element_ty)

    # --- token bonds: a K=1 GEMM, i.e. an exact product rounded to bf16 ---
    if HAS_TB:
        tb = tl.load(TB + row * N + j, mask=jm, other=0.0).to(tl.float32)
        w_tb = tl.load(WTB + c, mask=cm, other=0.0)
        emb = _rnd(tb[:, None] * w_tb[None, :], Z.dtype.element_ty)
        acc = acc + emb

    tl.store(
        Z + (row * N + j)[:, None] * CZ + c[None, :],
        acc.to(Z.dtype.element_ty), mask=tm,
    )


@triton.jit
def _layer_norm_kernel(
    X, W, B, OUT, n_rows, C, EPS,
    HAS_W: tl.constexpr, HAS_B: tl.constexpr,
    BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """`L1.layer_norm.LayerNorm.forward` with `promote_fp32=True`, in one kernel.

    The reference spells that as `x.float()` -> `F.layer_norm` -> `.to(bf16)`:
    three launches, the middle one an fp32 kernel reading and writing an fp32
    copy of the input. Here the fp32 promotion is just the accumulator type, so
    there is one launch and no fp32 traffic.
    """
    pid = tl.program_id(0)
    r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    c = tl.arange(0, BLOCK_C)
    rm = r < n_rows
    cm = c < C
    msk = rm[:, None] & cm[None, :]

    x = tl.load(X + r[:, None] * C + c[None, :], mask=msk, other=0.0).to(tl.float32)
    mean = tl.sum(x, 1) / C
    d = tl.where(cm[None, :], x - mean[:, None], 0.0)
    var = tl.sum(d * d, 1) / C
    y = d * tl.rsqrt(var + EPS)[:, None]
    if HAS_W:
        y = y * tl.load(W + c, mask=cm, other=0.0)[None, :]
    if HAS_B:
        y = y + tl.load(B + c, mask=cm, other=0.0)[None, :]
    tl.store(OUT + r[:, None] * C + c[None, :],
             y.to(OUT.dtype.element_ty), mask=msk)


@triton.jit
def _adaln_kernel(
    A, GS, OUT, n_rows, C, EPS,
    BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """The whole tail of `AdaLN.forward`:

        out = sigmoid(g) * (layer_norm_a(a) + s_part)

    where `GS` is one `[n_rows, 2*C]` GEMM output holding `linear_g(s_norm)` in
    columns [0, C) and `linear_s(s_norm)` in [C, 2C). `layer_norm_a` is built
    with create_scale=create_offset=False, so it is a bare normalize.

    Each `_rnd` is a point where the reference materializes a bf16 tensor
    (LN output, the add, the sigmoid, the multiply); keeping them makes this
    bit-identical to the reference rather than merely more accurate.
    """
    pid = tl.program_id(0)
    r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    c = tl.arange(0, BLOCK_C)
    rm = r < n_rows
    cm = c < C
    msk = rm[:, None] & cm[None, :]
    dt = OUT.dtype.element_ty

    x = tl.load(A + r[:, None] * C + c[None, :], mask=msk, other=0.0).to(tl.float32)
    mean = tl.sum(x, 1) / C
    d = tl.where(cm[None, :], x - mean[:, None], 0.0)
    var = tl.sum(d * d, 1) / C
    a_norm = _rnd(d * tl.rsqrt(var + EPS)[:, None], dt)

    gs_row = r[:, None] * (2 * C)
    g_pre = tl.load(GS + gs_row + c[None, :], mask=msk, other=0.0).to(tl.float32)
    s_part = tl.load(GS + gs_row + C + c[None, :], mask=msk, other=0.0).to(tl.float32)

    y = _rnd(_rnd(tl.sigmoid(g_pre), dt) * _rnd(a_norm + s_part, dt), dt)
    tl.store(OUT + r[:, None] * C + c[None, :], y.to(dt), mask=msk)


@triton.jit
def _gate_mul_kernel(
    GPRE, X, MASK, RES, OUT, n, C,
    HAS_MASK: tl.constexpr, HAS_RES: tl.constexpr, BLOCK: tl.constexpr,
):
    """`sigmoid(g_pre) * x`, optionally `* mask` broadcast over the last dim,
    optionally `+ res`.

    Both `CrossAttentionPairBias` and `ConditionedTransition` end in this gate
    and the transformer immediately adds the result to the residual stream, so
    folding that add in here removes one launch per sublayer -- 6 of the 144
    kernels left in the graph. `RES` is applied last and through its own `_rnd`,
    which is exactly what `a = a + <bf16 tensor>` does.
    """
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    dt = OUT.dtype.element_ty
    g = _rnd_hard(tl.sigmoid(tl.load(GPRE + i, mask=m, other=0.0).to(tl.float32)),
                  dt)
    x = tl.load(X + i, mask=m, other=0.0).to(tl.float32)
    y = _rnd_hard(g * x, dt)
    if HAS_MASK:
        y = _rnd_hard(
            y * tl.load(MASK + i // C, mask=m, other=0.0).to(tl.float32), dt)
    if HAS_RES:
        y = _rnd_hard(
            y + tl.load(RES + i, mask=m, other=0.0).to(tl.float32), dt)
    tl.store(OUT + i, y.to(dt), mask=m)


@triton.jit
def _swiglu_kernel(
    AB, OUT, n_rows, C, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """`silu(linear_a(x)) * linear_b(x)` from one `[n_rows, 2*C]` GEMM output."""
    pid = tl.program_id(0)
    r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    c = tl.arange(0, BLOCK_C)
    msk = (r < n_rows)[:, None] & (c < C)[None, :]
    dt = OUT.dtype.element_ty
    row = r[:, None] * (2 * C)
    av = tl.load(AB + row + c[None, :], mask=msk, other=0.0).to(tl.float32)
    bv = tl.load(AB + row + C + c[None, :], mask=msk, other=0.0).to(tl.float32)
    y = _rnd(_rnd(av * tl.sigmoid(av), dt) * bv, dt)
    tl.store(OUT + r[:, None] * C + c[None, :], y.to(dt), mask=msk)


@triton.jit
def _transition_kernel(
    X, S, WA, WB, WO, WG, BG, MASK, RES, OUT,
    n_rows, C, CH, CO,
    HAS_BG: tl.constexpr, HAS_MASK: tl.constexpr, HAS_RES: tl.constexpr,
    BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_H: tl.constexpr,
    BLOCK_O: tl.constexpr,
):
    """Everything `ConditionedTransitionBlock.forward` does after its AdaLN:

        b   = silu(linear_a(x)) * linear_b(x)
        out = res + sigmoid(linear_g(s)) * linear_out(b) * mask

    That is five launches in the split form -- the stacked `linear_a|linear_b`
    GEMM, the silu-multiply, `linear_out`, `linear_g`, and the gate -- times
    three blocks. Here it is one, and `b` (256 wide) never reaches memory.

    All four weight tiles stay resident for the whole program, so each row tile
    is two K=128 dots into the hidden width, one K=256 dot back out, and one
    K=128 dot for the gate. Every `_rnd_hard` is a bf16 tensor the reference
    materializes; `linear_a`/`linear_b` are kept as two dots rather than one
    stacked GEMM to match the reference's two separate GEMMs (per-column K
    accumulation is identical either way).
    """
    pid = tl.program_id(0)
    r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    c = tl.arange(0, BLOCK_C)
    hd = tl.arange(0, BLOCK_H)
    o = tl.arange(0, BLOCK_O)
    rm = r < n_rows
    cm = c < C
    hm = hd < CH
    om = o < CO
    dt = OUT.dtype.element_ty

    x = tl.load(X + r[:, None] * C + c[None, :],
                mask=rm[:, None] & cm[None, :], other=0.0)
    wmsk = cm[:, None] & hm[None, :]
    av = _rnd_hard(tl.dot(x, tl.load(WA + c[:, None] * CH + hd[None, :],
                                     mask=wmsk, other=0.0)), dt)
    bv = _rnd_hard(tl.dot(x, tl.load(WB + c[:, None] * CH + hd[None, :],
                                     mask=wmsk, other=0.0)), dt)
    h = tl.where(rm[:, None] & hm[None, :],
                 _rnd_hard(_rnd_hard(av * tl.sigmoid(av), dt) * bv, dt),
                 0.0).to(dt)

    y = _rnd_hard(tl.dot(h, tl.load(WO + hd[:, None] * CO + o[None, :],
                                    mask=hm[:, None] & om[None, :],
                                    other=0.0)), dt)

    s = tl.load(S + r[:, None] * C + c[None, :],
                mask=rm[:, None] & cm[None, :], other=0.0)
    g = tl.dot(s, tl.load(WG + c[:, None] * CO + o[None, :],
                          mask=cm[:, None] & om[None, :], other=0.0))
    if HAS_BG:
        g = g + tl.load(BG + o, mask=om, other=0.0).to(tl.float32)[None, :]
    g = _rnd_hard(g, dt)

    y = _rnd_hard(_rnd_hard(tl.sigmoid(g), dt) * y, dt)
    if HAS_MASK:
        y = _rnd_hard(
            y * tl.load(MASK + r, mask=rm, other=0.0).to(tl.float32)[:, None], dt)
    if HAS_RES:
        y = _rnd_hard(y + tl.load(RES + r[:, None] * CO + o[None, :],
                                  mask=rm[:, None] & om[None, :], other=0.0
                                  ).to(tl.float32), dt)
    tl.store(OUT + r[:, None] * CO + o[None, :], y.to(dt),
             mask=rm[:, None] & om[None, :])


@triton.jit
def _adaln_fused_kernel(
    S, LNW, LNB, WG, WS, BG, A, OUT,
    n_rows, C, EPS_S, EPS_A,
    HAS_LNW: tl.constexpr, HAS_LNB: tl.constexpr, HAS_BG: tl.constexpr,
    BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """All of `AdaLN.forward` in one launch:

        s_norm = layer_norm_s(s)
        out    = sigmoid(linear_g(s_norm)) * (layer_norm_a(a) + linear_s(s_norm))

    The split version costs three launches per call -- the `layer_norm_s`
    kernel, one GEMM for `linear_g|linear_s`, and the epilogue -- and there are
    nine `AdaLN` calls per forward, so 27 of the graph's launches and ~65 µs of
    device time. `c_s == c_a == 128` here, so `s_norm` is a full row in
    registers and both projections are one `tl.dot` each with K=128 against a
    weight tile that stays resident; nothing needs to round-trip through memory.

    Rounding: `s_norm` is a bf16 tensor in the reference, so it is rounded
    *before* the projections (`snb`) rather than kept in fp32 -- that rounding is
    load-bearing, and the whole point of r1's finding that collapsing bf16 steps
    into one fp32 chain is more accurate and therefore wrong. `_rnd_hard` is used
    for every intermediate because the plain `.to(bf16).to(fp32)` round-trip is
    silently folded away when its result feeds fp32 arithmetic (see
    `_rnd_hard`'s docstring).

    `linear_g` and `linear_s` are kept as two separate `tl.dot`s, matching the
    reference's two separate GEMMs. Each output column is its own K=128
    accumulation either way, so this is also exactly what the stacked GEMM it
    replaces produced.
    """
    pid = tl.program_id(0)
    r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    c = tl.arange(0, BLOCK_C)
    rm = r < n_rows
    cm = c < C
    msk = rm[:, None] & cm[None, :]
    dt = OUT.dtype.element_ty

    # --- s_norm = layer_norm_s(s), rounded to bf16 like the reference stores it
    s = tl.load(S + r[:, None] * C + c[None, :], mask=msk, other=0.0).to(tl.float32)
    mean = tl.sum(s, 1) / C
    d = tl.where(cm[None, :], s - mean[:, None], 0.0)
    var = tl.sum(d * d, 1) / C
    sn = d * tl.rsqrt(var + EPS_S)[:, None]
    if HAS_LNW:
        sn = sn * tl.load(LNW + c, mask=cm, other=0.0)[None, :]
    if HAS_LNB:
        sn = sn + tl.load(LNB + c, mask=cm, other=0.0)[None, :]
    snb = tl.where(msk, _rnd_hard(sn, dt), 0.0).to(dt)

    # --- the two projections (weights arrive K-major, i.e. [C_in, C_out])
    wmsk = cm[:, None] & cm[None, :]
    g_pre = tl.dot(snb, tl.load(WG + c[:, None] * C + c[None, :], mask=wmsk,
                                other=0.0))
    if HAS_BG:
        g_pre = g_pre + tl.load(BG + c, mask=cm, other=0.0).to(tl.float32)[None, :]
    g_pre = _rnd_hard(g_pre, dt)
    s_part = _rnd_hard(
        tl.dot(snb, tl.load(WS + c[:, None] * C + c[None, :], mask=wmsk,
                            other=0.0)), dt)

    # --- layer_norm_a(a): built with create_scale=create_offset=False
    x = tl.load(A + r[:, None] * C + c[None, :], mask=msk, other=0.0).to(tl.float32)
    mean = tl.sum(x, 1) / C
    d = tl.where(cm[None, :], x - mean[:, None], 0.0)
    var = tl.sum(d * d, 1) / C
    a_norm = _rnd_hard(d * tl.rsqrt(var + EPS_A)[:, None], dt)

    y = _rnd_hard(_rnd_hard(tl.sigmoid(g_pre), dt)
                  * _rnd_hard(a_norm + s_part, dt), dt)
    tl.store(OUT + r[:, None] * C + c[None, :], y.to(dt), mask=msk)


@triton.jit
def _proj_kernel(
    A, B, C, M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """`A[M, K] @ B[K, N]` with one fp32 accumulation per output element.

    For the three token projections (`linear_s`, `linear_z_i`, `linear_z_j`)
    cuBLAS picks `gemmSN_TN_kernel<float, ...>` and spends 7.3 µs each -- 22 µs
    in the graph, 6% of the call, for 9 MFLOP. Their K is 449 and their M is
    N_token = 16, which is a degenerate GEMM shape, not a compute problem.

    Numerically this is *not* interchangeable with cuBLAS in general. With
    `allow_bf16_reduced_precision_reduction` on (torch's default) cuBLAS reduces
    split-K partials in bf16, and whether it splits K depends on the shape:
    `tests/diag_proj.py` measures Triton against cuBLAS against an fp64 ground
    truth and finds them bit-identical at M=16 (1 element in 6144) but far apart
    at M=7, where cuBLAS's own error is 4.7x larger. This kernel is therefore
    gated on a one-time measured comparison per shape (see `_proj`), not on an
    assumption -- which is the same hazard r1 hit by stacking these weights,
    seen from the other side.
    """
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    mm = rm < M
    nm = rn < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BK)):
        rk = k0 * BK + tl.arange(0, BK)
        km = rk < K
        a = tl.load(A + rm[:, None] * K + rk[None, :],
                    mask=mm[:, None] & km[None, :], other=0.0)
        b = tl.load(B + rk[:, None] * N + rn[None, :],
                    mask=km[:, None] & nm[None, :], other=0.0)
        acc += tl.dot(a, b)
    tl.store(C + rm[:, None] * N + rn[None, :],
             acc.to(C.dtype.element_ty), mask=mm[:, None] & nm[None, :])


@triton.jit
def _sdpa_kernel(
    QG, KV, MB, ZB, OUT,
    NQ, NK, H: tl.constexpr, CH: tl.constexpr, SCALE,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """The whole sequence-local attention core for one (block, head), in one pass.

        q      = rnd(qg[..., :HC] / sqrt(CH))     -> [NQ, CH]
        scores = rnd(q @ k^T) ; += mask_bias ; += z_bias   (rounding after each)
        p      = rnd(softmax(scores))
        o      = rnd(p @ v)
        out    = rnd(rnd(sigmoid(gate)) * o)

    One program per (block, head): `NK` is 128 here, so the whole key block fits
    in one tile and no online (flash) rescaling is needed -- the loop this
    replaces is not tiled over K at all.

    Replaces, per transformer block: the `/ sqrt(c_hidden)`, the two `einsum`s
    and the four `copy_`s einsum needs to make its operands contiguous, both
    bias adds, the softmax, the `o.transpose().reshape().contiguous()`, the
    `qg[..., hc:].contiguous()` the flat gate kernel used to need, and the
    gate multiply -- 11 launches down to 1.

    Every `_rnd` is a point where the reference materializes a bf16 tensor.
    Dropping any of them would make this *more* accurate than the reference and
    therefore wrong: the residual stream downstream is bf16 and the differences
    do not stay small.

    Layout (all contiguous, leading batch folded into the block index):
        QG  [NBLK, NQ, 2*H*CH]   q in the first H*CH, output gate in the second
        KV  [NBLK, NK, 2*H*CH]   k then v
        MB  [NBLK, NQ, NK]       mask bias, shared by all heads
        ZB  [NBLK, H, NQ, NK]    pair bias, per head
        OUT [NBLK, NQ, H*CH]
    """
    blk = tl.program_id(0)
    head = tl.program_id(1)
    dt = OUT.dtype.element_ty
    hc = H * CH

    q = tl.arange(0, BLOCK_Q)
    k = tl.arange(0, BLOCK_K)
    c = tl.arange(0, BLOCK_C)
    qm = q < NQ
    km = k < NK
    cm = c < CH

    qrow = (blk * NQ + q[:, None]) * (2 * hc) + head * CH
    krow = (blk * NK + k[:, None]) * (2 * hc) + head * CH

    qt = tl.load(QG + qrow + c[None, :], mask=qm[:, None] & cm[None, :],
                 other=0.0).to(tl.float32)
    qt = _rnd(qt / SCALE, dt)
    kt = tl.load(KV + krow + c[None, :], mask=km[:, None] & cm[None, :],
                 other=0.0)
    vt = tl.load(KV + krow + hc + c[None, :], mask=km[:, None] & cm[None, :],
                 other=0.0)

    s = _rnd(tl.dot(qt.to(dt), tl.trans(kt)), dt)
    s = _rnd(s + tl.load(MB + (blk * NQ + q[:, None]) * NK + k[None, :],
                         mask=qm[:, None] & km[None, :], other=0.0
                         ).to(tl.float32), dt)
    s = _rnd(s + tl.load(ZB + ((blk * H + head) * NQ + q[:, None]) * NK
                         + k[None, :],
                         mask=qm[:, None] & km[None, :], other=0.0
                         ).to(tl.float32), dt)

    s = tl.where(km[None, :], s, float("-inf"))
    p = tl.exp(s - tl.max(s, 1)[:, None])
    p = _rnd(p / tl.sum(p, 1)[:, None], dt)

    o = _rnd(tl.dot(p.to(dt), vt), dt)
    g = _rnd(tl.sigmoid(tl.load(QG + qrow + hc + c[None, :],
                                mask=qm[:, None] & cm[None, :], other=0.0
                                ).to(tl.float32)), dt)
    tl.store(OUT + (blk * NQ + q[:, None]) * hc + head * CH + c[None, :],
             _rnd(g * o, dt).to(dt), mask=qm[:, None] & cm[None, :])


@triton.jit
def _z_bias_kernel(
    Z, W, OUT, NROW, CZ,
    H: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """`_permute_final_dims(linear_z(z), [2, 0, 1])` in one pass.

        z      [NBLK, NQ, NK, CZ]        (the blocked, layer-normed pair rep)
        W      [H, CZ]                   (`linear_z.weight`, no bias)
        out    [NBLK, H, NQ, NK]         (the attention bias, contiguous)

    The reference runs this as a real GEMM with M = NBLK*NQ*NK = 49152 but
    K = CZ = 16 and N = H = 4, which on B200 costs 9.7 µs per block -- ~14x
    what the 2 MB of traffic needs -- and then hands the attention a
    *permuted view*, so the bias add reads it strided. This writes the permuted
    layout directly.

    `BLOCK_H` is padded to tl.dot's minimum N of 16 with masked-out (zero) rows
    of W. Padding N cannot change the K reduction, so each live output column is
    still one 16-deep fp32 accumulation -- the same single `mma` shape cuBLAS
    issues for K=16 -- and `acc.to(dt)` is the single rounding to bf16 that the
    reference's GEMM epilogue does.

    `(q, k)` is flattened into one row index of length `NROW = NQ*NK`, which the
    permuted output happens to index the same way (`q*NK + k == row`), so the
    store needs no divide and the tile can be as wide as registers allow.
    """
    blk = tl.program_id(0)
    rt = tl.program_id(1)

    r = rt * BLOCK_R + tl.arange(0, BLOCK_R)
    c = tl.arange(0, BLOCK_C)
    h = tl.arange(0, BLOCK_H)
    rm = r < NROW
    cm = c < CZ

    zt = tl.load(Z + (blk * NROW + r[:, None]) * CZ + c[None, :],
                 mask=rm[:, None] & cm[None, :], other=0.0)
    wt = tl.load(W + h[:, None] * CZ + c[None, :],
                 mask=(h < H)[:, None] & cm[None, :], other=0.0)
    acc = tl.dot(zt, tl.trans(wt))

    tl.store(OUT + (blk * H + h[None, :]) * NROW + r[:, None],
             acc.to(OUT.dtype.element_ty),
             mask=rm[:, None] & (h < H)[None, :])


@triton.jit
def _plm_kernel(
    DL, DM, VL, VM, BM, W_OFF, W_INV, W_VAL, PLM,
    NQ, NK, CP,
    BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """`RefAtomFeatureEmbedder`'s pair-feature block, in one kernel.

        dlm = (d_l[:, None] - d_m[None, :]) * block_mask
        vlm = (v_l[:, None] == v_m[None, :]) * block_mask
        plm = linear_ref_offset(dlm) * vlm
            + linear_inv_sq_dists(1 / (1 + sum(dlm^2))) * vlm
            + linear_valid_mask(vlm) * vlm

    The reference builds this as ~15 launches over `[nb, n_query, n_key, *]`
    tensors -- the largest in the module (786k elements at the benchmarked
    shape) -- so it is read and rewritten a dozen times. Here it is one pass.
    `d_l`/`d_m` are 3-vectors (positions), so the ref_offset GEMM is three
    multiply-adds per output channel and needs no `tl.dot`.

    Every `_rnd` is a bf16 materialization in the reference; keeping them makes
    the result bit-identical rather than merely more accurate.
    """
    row = tl.program_id(0)          # block * NQ + q
    kt = tl.program_id(1)
    blk = row // NQ

    k = kt * BLOCK_K + tl.arange(0, BLOCK_K)
    km = k < NK
    c = tl.arange(0, BLOCK_C)
    cm = c < CP
    dt = PLM.dtype.element_ty

    m = tl.load(BM + row * NK + k, mask=km, other=0.0).to(tl.float32)

    # dlm: three channels, each (d_l - d_m) * mask with the reference's rounding.
    dm_base = (blk * NK + k) * 3
    d0 = _rnd(_rnd(tl.load(DL + row * 3 + 0).to(tl.float32)
                   - tl.load(DM + dm_base + 0, mask=km, other=0.0).to(tl.float32),
                   dt) * m, dt)
    d1 = _rnd(_rnd(tl.load(DL + row * 3 + 1).to(tl.float32)
                   - tl.load(DM + dm_base + 1, mask=km, other=0.0).to(tl.float32),
                   dt) * m, dt)
    d2 = _rnd(_rnd(tl.load(DL + row * 3 + 2).to(tl.float32)
                   - tl.load(DM + dm_base + 2, mask=km, other=0.0).to(tl.float32),
                   dt) * m, dt)

    v_l = tl.load(VL + row).to(tl.float32)
    v_m = tl.load(VM + blk * NK + k, mask=km, other=0.0).to(tl.float32)
    vlm = _rnd(tl.where(v_l == v_m, 1.0, 0.0) * m, dt)

    # `sum(dlm**2, -1)`: squares round to bf16, the reduction accumulates in fp32.
    sq = _rnd(_rnd(d0 * d0, dt) + _rnd(d1 * d1, dt) + _rnd(d2 * d2, dt), dt)
    inv = _rnd(1.0 / _rnd(1.0 + sq, dt), dt)

    w0 = tl.load(W_OFF + c * 3 + 0, mask=cm, other=0.0).to(tl.float32)
    w1 = tl.load(W_OFF + c * 3 + 1, mask=cm, other=0.0).to(tl.float32)
    w2 = tl.load(W_OFF + c * 3 + 2, mask=cm, other=0.0).to(tl.float32)
    w_inv = tl.load(W_INV + c, mask=cm, other=0.0).to(tl.float32)
    w_val = tl.load(W_VAL + c, mask=cm, other=0.0).to(tl.float32)

    off = _rnd(d0[:, None] * w0[None, :] + d1[:, None] * w1[None, :]
               + d2[:, None] * w2[None, :], dt)
    acc = _rnd(off * vlm[:, None], dt)
    acc = _rnd(acc + _rnd(_rnd(inv[:, None] * w_inv[None, :], dt) * vlm[:, None], dt), dt)
    acc = _rnd(acc + _rnd(_rnd(vlm[:, None] * w_val[None, :], dt) * vlm[:, None], dt), dt)

    msk = km[:, None] & cm[None, :]
    tl.store(PLM + (row * NK + k)[:, None] * CP + c[None, :],
             acc.to(dt), mask=msk)


@triton.jit
def _pair_update_kernel(
    PLM, LQ, LM, BM, W1, W2, W3, OUT,
    NQ, NK, CP,
    BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """`AtomAttentionEncoder`'s pair update, in one kernel:

        p   = plm + (linear_l(relu(cl_l))[:, None] + linear_m(relu(cl_m))[None, :]) * bm
        out = (p + pair_mlp(p)) * bm

    `pair_mlp` is ReLU -> Linear -> ReLU -> Linear -> ReLU -> Linear, all
    c_atom_pair -> c_atom_pair (16 wide here). As three separate GEMMs those are
    M=nb*n_query*n_key, K=N=16 -- cuBLAS picks a 128x128 tile and spends ~10 us
    per launch moving 786k elements in and out for 16 MACs of work each. Held in
    registers instead, the three matmuls are `tl.dot`s on a tile already resident.

    `W*` are `[CP, CP]` in `[out, in]` order, so they are loaded transposed to
    give `tl.dot` its `[in, out]` operand. Masked lanes load zero, which is a
    no-op through both the ReLUs and the accumulations.
    """
    row = tl.program_id(0)          # block * NQ + q
    kt = tl.program_id(1)
    blk = row // NQ

    k = kt * BLOCK_K + tl.arange(0, BLOCK_K)
    km = k < NK
    c = tl.arange(0, BLOCK_C)
    cm = c < CP
    msk = km[:, None] & cm[None, :]
    dt = OUT.dtype.element_ty

    bm = tl.load(BM + row * NK + k, mask=km, other=0.0).to(tl.float32)
    lq = tl.load(LQ + row * CP + c, mask=cm, other=0.0).to(tl.float32)
    lm = tl.load(LM + (blk * NK + k)[:, None] * CP + c[None, :],
                 mask=msk, other=0.0).to(tl.float32)
    plm = tl.load(PLM + (row * NK + k)[:, None] * CP + c[None, :],
                  mask=msk, other=0.0).to(tl.float32)

    cl_lm = _rnd(_rnd(lq[None, :] + lm, dt) * bm[:, None], dt)
    p = _rnd(plm + cl_lm, dt)

    wmsk = cm[None, :] & cm[:, None]
    w1 = tl.load(W1 + c[None, :] * CP + c[:, None], mask=wmsk, other=0.0)
    w2 = tl.load(W2 + c[None, :] * CP + c[:, None], mask=wmsk, other=0.0)
    w3 = tl.load(W3 + c[None, :] * CP + c[:, None], mask=wmsk, other=0.0)

    t = _rnd(tl.dot(tl.maximum(p, 0.0).to(dt), w1), dt)
    t = _rnd(tl.dot(tl.maximum(t, 0.0).to(dt), w2), dt)
    t = _rnd(tl.dot(tl.maximum(t, 0.0).to(dt), w3), dt)

    out = _rnd(_rnd(p + t, dt) * bm[:, None], dt)
    tl.store(OUT + (row * NK + k)[:, None] * CP + c[None, :],
             out.to(dt), mask=msk)


@triton.jit
def _block_split_kernel(
    QL, IDX, INV, QUERY, KEY, n_atom, NRQ, NRK, C,
    BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """Both halves of `_convert_single_rep_to_blocks`'s data movement at once.

    The reference zero-pads `ql` to a whole number of query blocks (a fill plus a
    device-to-device copy), reshapes that for the query view, gathers the key
    view out of it, then zeroes the invalid key rows in place -- four launches per
    conversion, five conversions per forward. Here the padding is expressed as a
    load mask (padded rows read as zero, which is exactly what the pad wrote), so
    one launch produces both views and nothing is materialized twice.

    `program_id(0)` selects the output: 0 = query rows, 1 = key rows.
    """
    which = tl.program_id(0)
    r = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
    c = tl.arange(0, BLOCK_C)
    cm = c < C

    if which == 0:
        rm = r < NRQ
        keep = rm & (r < n_atom)
        v = tl.load(QL + r[:, None] * C + c[None, :],
                    mask=keep[:, None] & cm[None, :], other=0.0)
        tl.store(QUERY + r[:, None] * C + c[None, :], v,
                 mask=rm[:, None] & cm[None, :])
    else:
        rm = r < NRK
        idx = tl.load(IDX + r, mask=rm, other=0)
        inv = tl.load(INV + r, mask=rm, other=1)
        keep = rm & (inv == 0) & (idx < n_atom)
        v = tl.load(QL + idx[:, None] * C + c[None, :],
                    mask=keep[:, None] & cm[None, :], other=0.0)
        tl.store(KEY + r[:, None] * C + c[None, :], v,
                 mask=rm[:, None] & cm[None, :])


class _BlockCtx:
    """The (atom_mask, n_atom, n_query, n_key)-only half of
    `_convert_single_rep_to_blocks`, computed once per forward.

    `key_indices` / `invalid_mask` / `mask_blocks` depend on nothing but those
    four, yet the reference recomputes them at all nine call sites. `blocks(ql)`
    then costs one pad + one gather + one masked_fill instead of ~25 kernels.

    `_get_block_key_indices` is reused verbatim rather than reimplemented: it
    mixes int32 `initial` with a bf16 `n_real`, so `final` comes out in bf16 and
    key indices above 256 are rounded to even. Reproducing that faithfully is
    not worth the risk when the win here is deduplication, not arithmetic.
    """

    __slots__ = ("atom_mask", "padded_mask", "key_indices", "invalid_mask",
                 "invalid_flat", "mask_blocks", "idx_flat", "n_atom", "n_pad",
                 "n_query", "n_key", "num_blocks", "lead", "_cache")

    def __init__(self, atom_mask, n_atom, n_query, n_key):
        self.atom_mask = atom_mask
        self.n_atom = n_atom
        self.n_query = n_query
        self.n_key = n_key
        self.n_pad = (-n_atom) % n_query
        self.num_blocks = -(-n_atom // n_query)
        self.lead = atom_mask.shape[:-1]
        self._cache = {}

        am = atom_mask
        if self.n_pad:
            am = torch.nn.functional.pad(am, (0, self.n_pad))
        self.padded_mask = am

        key_indices, invalid_mask = _get_block_key_indices(am, n_query, n_key)
        self.key_indices = key_indices
        self.invalid_mask = invalid_mask
        self.idx_flat = key_indices.reshape(-1, self.num_blocks * n_key)
        # int8 view of invalid_mask for `_block_split_kernel` (one launch here,
        # instead of a bool-tensor load in each of the five conversions).
        self.invalid_flat = invalid_mask.reshape(-1).to(torch.int8)

        nb, nq, nk = self.num_blocks, n_query, n_key
        mask_q = am.reshape(*self.lead, nb, nq)
        mask_k = (~invalid_mask).to(am.dtype) * torch.gather(
            am.reshape(self.idx_flat.shape[0], -1), 1, self.idx_flat,
        ).reshape(*self.lead, nb, nk)
        self.mask_blocks = mask_q.unsqueeze(-1) * mask_k.unsqueeze(-2)

    def mask_bias(self, inf: float):
        """`(inf * (mask_blocks - 1))[..., None, :, :]`, computed once."""
        hit = self._cache.get(("mask_bias", inf))
        if hit is None:
            hit = (inf * (self.mask_blocks - 1))[..., None, :, :]
            self._cache[("mask_bias", inf)] = hit
        return hit

    def blocks(self, ql):
        """`(ql_query, ql_key)` for this block layout. Memoized on tensor
        identity: the transformer feeds the same `s` to all three blocks."""
        hit = self._cache.get(id(ql))
        if hit is not None and hit[0] is ql:
            return hit[1]

        c = ql.shape[-1]
        nb, nq, nk = self.num_blocks, self.n_query, self.n_key
        fused = (len(self.lead) == 1 and self.lead[0] == 1
                 and ql.is_contiguous() and ql.is_cuda
                 and triton.next_power_of_2(c) <= 1024)
        if fused:
            block_c = triton.next_power_of_2(c)
            block_r = max(1, min(triton.next_power_of_2(max(nb * nq, nb * nk)),
                                 2048 // block_c))
            query = torch.empty((1, nb, nq, c), dtype=ql.dtype, device=ql.device)
            key = torch.empty((1, nb, nk, c), dtype=ql.dtype, device=ql.device)
            grid = (2, triton.cdiv(max(nb * nq, nb * nk), block_r))
            _block_split_kernel[grid](
                ql, self.idx_flat[0], self.invalid_flat, query, key,
                self.n_atom, nb * nq, nb * nk, c,
                BLOCK_R=block_r, BLOCK_C=block_c,
            )
            out = (query, key)
        else:
            padded = ql
            if self.n_pad:
                padded = torch.nn.functional.pad(ql, (0, 0, 0, self.n_pad))
            ql_query = padded.reshape(*self.lead, nb, nq, c)
            flat = self.idx_flat.shape[0]
            ql_key = torch.gather(
                padded.reshape(flat, self.n_atom + self.n_pad, c), 1,
                self.idx_flat.unsqueeze(-1).expand(-1, -1, c),
            )
            ql_key.masked_fill_(
                self.invalid_mask.reshape(flat, nb * nk)
                .unsqueeze(-1).expand(-1, -1, c),
                0.0,
            )
            out = (ql_query, ql_key.reshape(*self.lead, nb, nk, c))
        self._cache[id(ql)] = (ql, out)
        return out


def _binned_one_hot(
    x: torch.Tensor, boundaries: torch.Tensor,
) -> torch.Tensor:
    """One-hot encoding with bin boundaries (matches reference binned_one_hot)."""
    return (x[..., None] > boundaries).to(dtype=x.dtype)


def relpos_complex(
    batch: dict,
    max_relative_idx: int,
    max_relative_chain: int,
) -> torch.Tensor:
    """Build relative position features matching the reference implementation.

    Produces 139 features when max_relative_idx=32, max_relative_chain=2:
      66 (rel_pos) + 66 (rel_token) + 1 (same_entity) + 6 (rel_chain)

    Reference: openfold3/core/utils/relpos.py relpos_complex
    """
    res_idx = batch["residue_index"]
    asym_id = batch["asym_id"]
    entity_id = batch["entity_id"]
    same_chain = asym_id[..., None] == asym_id[..., None, :]
    same_res = res_idx[..., None] == res_idx[..., None, :]
    same_entity = entity_id[..., None] == entity_id[..., None, :]

    def _relpos(
        pos: torch.Tensor, condition: torch.BoolTensor, rel_clip_idx: int,
    ) -> torch.Tensor:
        offset = pos[..., None] - pos[..., None, :]
        clipped_offset = torch.clamp(offset + rel_clip_idx, min=0, max=2 * rel_clip_idx)
        final_offset = torch.where(
            condition,
            clipped_offset,
            (2 * rel_clip_idx + 1) * torch.ones_like(clipped_offset),
        )
        boundaries = torch.arange(
            start=0, end=2 * rel_clip_idx + 2, device=final_offset.device,
        ).to(dtype=final_offset.dtype)
        return _binned_one_hot(final_offset, boundaries)

    rel_pos = _relpos(pos=res_idx, condition=same_chain, rel_clip_idx=max_relative_idx)
    rel_token = _relpos(
        pos=batch["token_index"],
        condition=same_chain & same_res,
        rel_clip_idx=max_relative_idx,
    )
    rel_chain = _relpos(
        pos=batch["sym_id"],
        condition=same_entity,
        rel_clip_idx=max_relative_chain,
    )

    same_entity_feat = same_entity[..., None].to(dtype=rel_pos.dtype)

    return torch.cat([rel_pos, rel_token, same_entity_feat, rel_chain], dim=-1)


class InputEmbedder(nn.Module):
    """Produces initial single and pair representations from token features.

    Matches InputEmbedderAllAtom: runs AtomAttentionEncoder to get a
    token-level representation, concatenates with restype/profile/deletion_mean
    to form s_input (449 dims), then projects to s and z.

    Args:
        c_s_input: Input single representation dimension (449 for all-atom)
        c_s: Single representation dimension
        c_z: Pair representation dimension
        relpos_k: Maximum relative residue position
        max_relative_chain: Maximum relative chain index
        c_atom: Atom single representation dim
        c_atom_pair: Atom pair representation dim
        c_token: Token dim for atom attention encoder output
    """

    def __init__(
        self,
        c_s_input: int,
        c_s: int,
        c_z: int,
        relpos_k: int = 32,
        max_relative_chain: int = 2,
        c_atom: int = 128,
        c_atom_pair: int = 16,
        c_token: int | None = None,
    ):
        super().__init__()
        self.c_s_input = c_s_input
        self.c_s = c_s
        self.c_z = c_z
        self.relpos_k = relpos_k
        self.max_relative_chain = max_relative_chain
        self._one_hot = OneHot()
        self._pad = Pad()

        if c_token is None:
            c_token = c_s

        self.atom_attn_enc = AtomAttentionEncoder(
            c_atom=c_atom,
            c_atom_pair=c_atom_pair,
            c_token=c_token,
            add_noisy_pos=False,
        )

        self.linear_s = Linear(c_s_input, c_s, bias=False)
        self.linear_z_i = Linear(c_s_input, c_z, bias=False)
        self.linear_z_j = Linear(c_s_input, c_z, bias=False)

        num_rel_pos_bins = 2 * relpos_k + 2
        num_rel_token_bins = 2 * relpos_k + 2
        num_rel_chain_bins = 2 * max_relative_chain + 2
        num_same_entity_features = 1
        n_relpos_features = (
            num_rel_pos_bins + num_rel_token_bins
            + num_rel_chain_bins + num_same_entity_features
        )
        self.linear_relpos = Linear(n_relpos_features, c_z, bias=False)

        self.linear_token_bonds = Linear(1, c_z, bias=False)

        # --- Weight-derived caches. Not parameters/buffers: the harness shares
        # weights via ``load_state_dict(strict=False)`` *after* construction, so
        # anything derived from a weight has to be built lazily on the first
        # forward and dropped whenever a parameter can have changed
        # (see _drop_derived).
        self._relpos_src: torch.Tensor | None = None
        self._relpos_tab: tuple | None = None
        self._tb_src: torch.Tensor | None = None
        self._tb_w: torch.Tensor | None = None
        self._ln_cache: dict = {}
        self._stack_cache: dict = {}
        self._proj_cache: dict = {}
        self._proj_ok: dict = {}
        # --- CUDA-graph state (see _replay) ---
        self._graph: tuple | None = None
        self._graph_warm = 0
        self._graph_off = False

    # ------------------------------------------------------------------
    # relpos: thermometer features x weight == gather from prefix sums.
    #
    # ``_binned_one_hot(x, boundaries) = (x[..., None] > boundaries)`` with
    # ``boundaries = arange(0, 2k+2)`` is a *thermometer* code, not a one-hot:
    # for an offset ``f in [0, 2k+1]`` it sets exactly the first ``ceil(f)``
    # entries. Hence
    #
    #     sum_b W[:, b] * feats[b] == sum_{b < ceil(f)} W[:, b] == cumW[ceil(f)]
    #
    # with ``cumW[v] = sum_{b<v} W[:, b]`` (``cumW[0] = 0``). ``relpos_feats`` is
    # ``cat([rel_pos(2k+2), rel_token(2k+2), same_entity(1), rel_chain(2kc+2)])``,
    # so ``linear_relpos(relpos_feats)`` is three table lookups plus one scaled
    # 0/1 column -- no one-hot materialization, no cat, no 139-wide GEMM.
    # ------------------------------------------------------------------
    def _relpos_tables(self):
        """``(cum_pos, cum_token, w_same_entity, cum_chain, w_onehot)``, fp32.

        Cached, keyed on ``linear_relpos.weight`` identity (the harness shares
        weights via ``load_state_dict`` after construction).
        """
        w = self.linear_relpos.weight
        if self._relpos_tab is not None and self._relpos_src is w:
            return self._relpos_tab

        n_idx = 2 * self.relpos_k + 2
        n_chain = 2 * self.max_relative_chain + 2
        wf = w.float()

        def _prefix(block: torch.Tensor) -> torch.Tensor:
            # block: [n_bins, c_z] rows b=0..n_bins-1 -> [n_bins, c_z] prefix
            # sums with a leading zero row, so row v == sum of rows < v.
            cs = block.cumsum(0)
            return torch.cat([torch.zeros_like(cs[:1]), cs[:-1]], dim=0).contiguous()

        o = 0
        cum_pos = _prefix(wf[:, o:o + n_idx].t())
        o += n_idx
        cum_tok = _prefix(wf[:, o:o + n_idx].t())
        o += n_idx
        w_se = wf[:, o].contiguous()
        o += 1
        cum_chain = _prefix(wf[:, o:o + n_chain].t())
        # The batch=None path builds a true one-hot over the first n_idx bins
        # (then zero-pads to 139), so there the answer is a plain row gather.
        w_onehot = wf[:, :n_idx].t().contiguous()

        self._relpos_src = w
        self._relpos_tab = (cum_pos, cum_tok, w_se, cum_chain, w_onehot)
        return self._relpos_tab

    def _relpos_bin(
        self, pos: torch.Tensor, condition: torch.Tensor, clip: int,
    ) -> torch.Tensor:
        """``ceil(final_offset)`` -- the number of thermometer bits set.

        Mirrors ``relpos_complex._relpos`` op-for-op (same dtype, same clamp,
        same out-of-condition sentinel) so the bin index is bit-identical.
        """
        offset = pos[..., :, None] - pos[..., None, :]
        clipped = torch.clamp(offset + clip, min=0, max=2 * clip)
        final = torch.where(
            condition, clipped, torch.full_like(clipped, 2 * clip + 1),
        )
        return final.ceil().long()

    def _relpos_emb(self, batch: dict, dtype: torch.dtype) -> torch.Tensor:
        cum_pos, cum_tok, w_se, cum_chain, _ = self._relpos_tables()

        res_idx = batch["residue_index"]
        asym_id = batch["asym_id"]
        entity_id = batch["entity_id"]
        same_chain = asym_id[..., :, None] == asym_id[..., None, :]
        same_entity = entity_id[..., :, None] == entity_id[..., None, :]
        same_res = res_idx[..., :, None] == res_idx[..., None, :]

        k, kc = self.relpos_k, self.max_relative_chain
        v_pos = self._relpos_bin(res_idx, same_chain, k)
        v_tok = self._relpos_bin(batch["token_index"], same_chain & same_res, k)
        v_chain = self._relpos_bin(batch["sym_id"], same_entity, kc)

        out = cum_pos[v_pos] + cum_tok[v_tok] + cum_chain[v_chain]
        out = out + w_se * same_entity[..., None].float()
        return out.to(dtype=dtype)

    def forward(
        self,
        token_features: torch.Tensor,
        residue_index: torch.Tensor,
        batch: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            token_features: [*, N_token, c_s_input] per-token features.
                If batch contains ref_pos (atom features), only restype/profile/deletion_mean
                are expected here and atom_attn_enc produces the remaining features.
                Otherwise, treated as pre-built s_input.
            residue_index:  [*, N_token] residue indices
            batch: Feature dict for relpos and atom attention.

        Returns:
            s_input: [*, N_token, c_s_input] input single representation
            s: [*, N_token, C_s] single representation
            z: [*, N_token, N_token, C_z] pair representation
        """
        out = self._replay(token_features, residue_index, batch)
        if out is not None:
            return out
        return self._forward_impl(token_features, residue_index, batch)

    # ------------------------------------------------------------------
    # CUDA-graph capture / replay.
    #
    # At the shapes this module is actually used at (N_token=16, N_atom=368) the
    # forward issues ~1900 kernel launches for ~1.5 ms of GPU work, so ~80% of
    # the wall time is CPU dispatch gap, not compute. Capturing the whole forward
    # collapses that to one graph launch; the only per-call CPU work left is
    # copying the inputs into the static buffers the graph reads, batched into
    # one `_foreach_copy_` per dtype.
    # ------------------------------------------------------------------
    def _copied_names(self):
        """The `batch` keys the captured graph refreshes (for tests)."""
        return None if self._graph is None else self._graph.copied_names()

    def _drop_derived(self):
        """Forget the graph and every weight-derived cache.

        A captured graph holds the *addresses* of the derived weights, so once a
        cache is rebuilt the old graph is reading a stale buffer. Called from
        every hook that can change a parameter.
        """
        self._graph = None
        self._graph_warm = 0
        self._relpos_tab = self._relpos_src = None
        self._tb_w = self._tb_src = None
        self._ln_cache = {}
        self._stack_cache = {}
        self._proj_cache = {}
        self._proj_ok = {}

    def _apply(self, *args, **kwargs):
        out = super()._apply(*args, **kwargs)
        self._drop_derived()
        return out

    def load_state_dict(self, *args, **kwargs):
        out = super().load_state_dict(*args, **kwargs)
        self._drop_derived()
        return out

    def _load_from_state_dict(self, *args, **kwargs):
        out = super()._load_from_state_dict(*args, **kwargs)
        self._drop_derived()
        return out

    def _replay(self, token_features, residue_index, batch):
        """Run the captured graph, or return ``None`` to fall back to eager."""
        if not _GRAPHS_ENABLED or self._graph_off or not token_features.is_cuda:
            return None
        if torch.is_grad_enabled() or torch.cuda.is_current_stream_capturing():
            return None

        plan = self._graph
        if plan is not None:
            # One pass: validate this call's inputs against the static buffers
            # they are about to be copied into, and collect them while doing it.
            srcs = plan.gather(token_features, residue_index, batch)
            if srcs is not None:
                return plan.replay(srcs)
            self._graph = plan = None

        # Let a few eager calls go through first: weight loading and the lazy
        # weight-derived caches must settle before anything is baked in.
        self._graph_warm += 1
        if self._graph_warm <= _GRAPH_WARMUP_CALLS:
            return None
        plan = self._capture(token_features, residue_index, batch)
        if plan is None:
            self._graph_off = True
            return None
        self._graph = plan
        srcs = plan.gather(token_features, residue_index, batch)
        if srcs is None:      # cannot happen; stay safe rather than guess
            self._graph = None
            return None
        return plan.replay(srcs)

    def _capture(self, token_features, residue_index, batch):
        try:
            static_tf = token_features.clone()
            static_ri = residue_index.clone()
            static_batch = None
            read = None
            if batch is not None:
                static_batch = dict(batch)
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        static_batch[k] = v.clone()

            # Warm up on a side stream: allocator pools, cuBLAS workspaces, the
            # Triton JIT and the lazy weight caches must all exist *before*
            # capture or they get baked in (or abort it). The warmup also runs
            # through `_ReadRecorder`, which is how the copy-in set is decided:
            # a tensor the traced forward never reads does not need refreshing,
            # and this batch carries seven such keys (`msa`, `msa_mask`,
            # `has_deletion`, `deletion_value` and a duplicate
            # `token_features`/`residue_index`/... the encoder path ignores).
            traced = static_batch
            if static_batch is not None:
                traced = _ReadRecorder(static_batch)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.no_grad():
                for _ in range(3):
                    self._forward_impl(static_tf, static_ri, traced)
            torch.cuda.current_stream().wait_stream(stream)
            if static_batch is not None:
                read = set(traced.read)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph), torch.no_grad():
                outputs = self._forward_impl(static_tf, static_ri, static_batch)
        except Exception:
            # Any host sync / unsupported op in the traced region: stay eager.
            torch.cuda.synchronize()
            return None

        # The two positional arguments are always refreshed: the recorder only
        # sees `batch`, and they are two tensors either way.
        slots = [(static_tf, _TF), (static_ri, _RI)]
        if static_batch is not None:
            for k in sorted(static_batch):
                v = static_batch[k]
                if isinstance(v, torch.Tensor) and (read is None or k in read):
                    slots.append((v, k))
        # One `_foreach_copy_` per dtype instead of one `copy_` per tensor.
        by_dtype: dict = {}
        for dst, name in slots:
            by_dtype.setdefault(dst.dtype, ([], []))
            by_dtype[dst.dtype][0].append(dst)
            by_dtype[dst.dtype][1].append(name)
        others = () if batch is None else tuple(
            (k, repr(v)) for k, v in sorted(batch.items())
            if not isinstance(v, torch.Tensor))
        return _GraphPlan(
            graph, list(by_dtype.values()), tuple(outputs),
            _sig(token_features), _sig(residue_index),
            None if batch is None else frozenset(batch), others,
        )

    # ------------------------------------------------------------------
    # Atom-attention encoder, re-expressed with the redundant work hoisted.
    #
    # `AtomAttentionEncoder` reaches `_convert_single_rep_to_blocks` nine times
    # per forward (ref_pos, ref_space_uid, cl, then `a` and `s` inside each of
    # the three `CrossAttentionPairBias` blocks). Every one of those calls
    # recomputes `_get_block_key_indices` and the block mask from the *same*
    # (atom_mask, n_query, n_key) -- ~17 kernels of pure duplicate work each --
    # and `s` is `cl` in all three blocks, so three of the nine conversions are
    # byte-identical to a fourth. Hoisting the shared parts out is ~180 of the
    # forward's 627 kernels.
    #
    # Everything below reads its parameters straight off the baseline submodules,
    # so the state dict (and therefore the harness's weight sharing) is untouched.
    # `None` is returned whenever the layout differs from what this path assumes,
    # and the caller falls back to the reference encoder.
    # ------------------------------------------------------------------
    def _encode_atoms(self, batch: dict):
        enc = self.atom_attn_enc
        if enc.noisy_position_embedder is not None:
            return None
        ref_pos = batch.get("ref_pos")
        atom_mask = batch.get("atom_mask")
        if ref_pos is None or atom_mask is None or "atom_to_token_index" not in batch:
            return None
        n_query, n_key = enc.n_query, enc.n_key
        n_atom = ref_pos.shape[-2]
        if (atom_mask.shape[-1] != n_atom
                or atom_mask.shape[:-1] != ref_pos.shape[:-2]
                or len(ref_pos.shape) != 3 or ref_pos.shape[0] != 1
                or n_key % 2 or n_query % 2):
            return None
        tr = enc.atom_transformer
        if not all(getattr(b.attention_pair_bias, "use_cross_attention", False)
                   or hasattr(b.attention_pair_bias, "layer_norm_a_q")
                   for b in tr.blocks):
            return None
        if not all(b.attention_pair_bias.use_ada_layer_norm for b in tr.blocks):
            return None

        ctx = _BlockCtx(atom_mask, n_atom, n_query, n_key)
        cl, plm = self._ref_atom_features(batch, ctx)

        cl_l, cl_m = ctx.blocks(cl)
        plm = self._pair_update(enc, plm, cl_l, cl_m, ctx)

        ql = self._atom_transformer(enc.atom_transformer, cl, cl, plm, ctx,
                                    s_blocks=(cl_l, cl_m))
        ql = ql * atom_mask.unsqueeze(-1)

        return _aggregate_atom_feat_to_tokens(
            token_mask=batch["token_mask"],
            atom_to_token_index=batch["atom_to_token_index"],
            atom_mask=atom_mask,
            atom_feat=enc.linear_q(ql),
            mode="mean",
        )

    def _ref_atom_features(self, batch: dict, ctx: "_BlockCtx"):
        """`RefAtomFeatureEmbedder.forward` with the block conversions hoisted.

        The five input projections of `cl` are deliberately *not* collapsed into
        one GEMM over concatenated inputs. That fusion is only 4 kernels, and it
        replaces five bf16-rounded partial sums with a single fp32 accumulation
        -- more accurate, but several bf16 ulps away from the reference. `cl`
        feeds the whole atom transformer and then a K=449 GEMM, which amplified
        that into ~8% relative error on `s`. Not worth 4 kernels.
        """
        emb = self.atom_attn_enc.ref_atom_feature_embedder
        ref_pos = batch["ref_pos"]
        dtype = ref_pos.dtype

        cl = emb.linear_ref_pos(ref_pos)
        cl = cl + emb.linear_ref_charge(
            torch.arcsinh(batch["ref_charge"].unsqueeze(-1)))
        cl = cl + emb.linear_ref_mask(batch["ref_mask"].unsqueeze(-1).to(dtype=dtype))
        cl = cl + emb.linear_ref_element(batch["ref_element"].to(dtype=dtype))
        cl = cl + emb.linear_ref_atom_chars(
            batch["ref_atom_name_chars"].flatten(start_dim=-2).to(dtype=dtype))

        d_l, d_m = ctx.blocks(ref_pos)
        v_l, v_m = ctx.blocks(batch["ref_space_uid"].unsqueeze(-1))

        plm = self._pair_features(emb, d_l, d_m, v_l, v_m, ctx)
        if plm is None:
            bm = ctx.mask_blocks.unsqueeze(-1)
            dlm = (d_l.unsqueeze(-2) - d_m.unsqueeze(-3)) * bm
            vlm = (v_l.unsqueeze(-2) == v_m.unsqueeze(-3)).to(dtype=dlm.dtype) * bm
            plm = emb.linear_ref_offset(dlm) * vlm
            inv_sq_dists = 1.0 / (1 + torch.sum(dlm ** 2, dim=-1, keepdim=True))
            plm = plm + emb.linear_inv_sq_dists(inv_sq_dists) * vlm
            plm = plm + emb.linear_valid_mask(vlm) * vlm
        return cl, plm

    def _pair_update(self, enc, plm, cl_l, cl_m, ctx: "_BlockCtx"):
        """`_pair_update_kernel` driver, with the reference chain as fallback.

        `linear_l` / `linear_m` stay real GEMMs (they reduce over c_atom=128 from
        only nb*n_query and nb*n_key rows); it is the three 16-wide `pair_mlp`
        matmuls over the full nb*n_query*n_key tile that are worth folding in.
        """
        lq = enc.linear_l(torch.relu(cl_l))
        lm = enc.linear_m(torch.relu(cl_m))
        c_p = lq.shape[-1]
        mlp = [m for m in enc.pair_mlp if isinstance(m, Linear)]
        block_c = triton.next_power_of_2(c_p)
        fusable = (
            len(mlp) == 3
            and 16 <= block_c <= 128
            and len(ctx.lead) == 1 and ctx.lead[0] == 1
            and all(m.weight.shape == (c_p, c_p) and m.bias is None for m in mlp)
            and plm.is_contiguous() and lq.is_contiguous() and lm.is_contiguous()
            and ctx.mask_blocks.is_contiguous()
        )
        if not fusable:
            bm = ctx.mask_blocks.unsqueeze(-1)
            p = plm + (lq.unsqueeze(-2) + lm.unsqueeze(-3)) * bm
            return (p + enc.pair_mlp(p)) * bm

        nb, nq, nk = ctx.num_blocks, ctx.n_query, ctx.n_key
        out = torch.empty_like(plm)
        block_k = min(triton.next_power_of_2(nk), 64)
        _pair_update_kernel[(nb * nq, triton.cdiv(nk, block_k))](
            plm, lq, lm, ctx.mask_blocks,
            mlp[0].weight, mlp[1].weight, mlp[2].weight, out,
            nq, nk, c_p,
            BLOCK_K=max(block_k, 16), BLOCK_C=block_c,
        )
        return out

    def _pair_features(self, emb, d_l, d_m, v_l, v_m, ctx: "_BlockCtx"):
        """`_plm_kernel` driver; ``None`` if the layout is not the 3-D-position,
        single-batch, contiguous case the kernel handles."""
        c_p = emb.linear_ref_offset.weight.shape[0]
        if (d_l.shape[-1] != 3 or v_l.shape[-1] != 1 or len(ctx.lead) != 1
                or ctx.lead[0] != 1 or triton.next_power_of_2(c_p) > 1024):
            return None
        for t in (d_l, d_m, v_l, v_m, ctx.mask_blocks):
            if not t.is_contiguous():
                return None

        nb, nq, nk = ctx.num_blocks, ctx.n_query, ctx.n_key
        plm = torch.empty((1, nb, nq, nk, c_p), dtype=d_l.dtype, device=d_l.device)
        block_k = min(triton.next_power_of_2(nk), 64)
        _plm_kernel[(nb * nq, triton.cdiv(nk, block_k))](
            d_l, d_m, v_l, v_m, ctx.mask_blocks,
            emb.linear_ref_offset.weight, emb.linear_inv_sq_dists.weight,
            emb.linear_valid_mask.weight, plm,
            nq, nk, c_p,
            BLOCK_K=block_k, BLOCK_C=triton.next_power_of_2(c_p),
        )
        return plm

    def _atom_transformer(self, tr, a, s, z, ctx: "_BlockCtx", s_blocks):
        """`DiffusionTransformer.forward` for the cross-attention (atom) case.

        `s` is the same tensor for every block, so its block conversion is done
        once by the caller and threaded in via *s_blocks*.
        """
        if tr.use_cross_attention:
            z = self._ln(tr.layer_norm_z, z)
        for block in tr.blocks:
            # `res=a` folds the residual add into each sublayer's gate kernel.
            a = self._cross_attn_pair_bias(
                block.attention_pair_bias, a, z, s, ctx, s_blocks, res=a)
            a = self._conditioned_transition(
                block.conditioned_transition, a, s, ctx.atom_mask, res=a)
        return a

    def _cross_attn_pair_bias(self, mod, a, z, s, ctx: "_BlockCtx", s_blocks,
                              res=None):
        a_query, a_key = ctx.blocks(a)
        s_q, s_k = s_blocks

        # `inf * (mask_blocks - 1)` depends only on the block mask, so it is the
        # same tensor in all three blocks; cache it on the context.
        mask_bias = ctx.mask_bias(mod.inf)
        z_bias = self._z_bias(mod, z)
        if z_bias is None:
            z_bias = _permute_final_dims(mod.linear_z(z), [2, 0, 1])

        a_q = self._ada_ln(mod.layer_norm_a_q, a_query, s_q)
        a_k = self._ada_ln(mod.layer_norm_a_k, a_key, s_k)

        out = self._attention(mod.mha, a_q, a_k, [mask_bias, z_bias])
        out = out.reshape((*a.shape[:-2], -1, a.shape[-1]))[..., :a.shape[-2], :]
        return self._gate_mul(mod.linear_ada_out(s), out.contiguous(), res=res)

    def _proj(self, mod, x):
        """`mod(x)` via `_proj_kernel`, or ``None`` to leave it to cuBLAS.

        Gated on a one-time *measured* comparison against `F.linear` for this
        (module, shape): cuBLAS's split-K choice -- and therefore whether its
        partials reduce in bf16 -- depends on the shape, so equivalence has to be
        checked, not assumed. The check needs a host sync, so it declines while a
        graph is being captured; by then the warmup calls have already filled the
        cache.
        """
        w = mod.weight
        if (getattr(mod, "bias", None) is not None or x.dim() < 2
                or not x.is_contiguous() or not x.is_cuda
                or not w.is_contiguous() or w.dtype is not x.dtype
                or x.shape[-1] != w.shape[1]):
            return None
        key = (id(mod), tuple(x.shape))
        ok = self._proj_ok.get(key)
        if ok is False:
            return None
        if ok is None:
            if torch.cuda.is_current_stream_capturing():
                return None
            ref = torch.nn.functional.linear(x, w)
            got = self._proj_run(x, w)
            if got is None:
                self._proj_ok[key] = False
                return None
            nd = int((got != ref).sum())
            ok = nd * 1000 <= ref.numel()
            self._proj_ok[key] = ok
            return got if ok else None
        return self._proj_run(x, w)

    def _proj_run(self, x, w):
        n_out, k = w.shape
        m = x.numel() // k
        b = self._proj_cache.get(id(w))
        if b is None or b[0] is not w:
            b = (w, w.t().contiguous())
            self._proj_cache[id(w)] = b
        bn = 64 if n_out >= 64 else triton.next_power_of_2(n_out)
        bm = max(16, min(triton.next_power_of_2(m), 64))
        bk = min(triton.next_power_of_2(k), 128)
        if bn < 16 or bk < 16:
            return None
        out = torch.empty((*x.shape[:-1], n_out), dtype=x.dtype, device=x.device)
        _proj_kernel[(triton.cdiv(m, bm), triton.cdiv(n_out, bn))](
            x, b[1], out, m, n_out, k, BM=bm, BN=bn, BK=bk, num_warps=4,
        )
        return out

    def _sdpa(self, qg, kv, biases, h, ch):
        """`_sdpa_kernel` driver. ``None`` unless everything is the blocked,
        contiguous, single-tile-over-K case the kernel handles.

        Takes the *stacked* projection outputs (`q|g` and `k|v`) and reads each
        head's slice with a stride, so neither the gate nor the head-major
        reshape needs a `.contiguous()` copy.
        """
        if len(biases) != 2 or qg.dim() < 3:
            return None
        mb, zb = biases
        if not isinstance(mb, torch.Tensor) or not isinstance(zb, torch.Tensor):
            return None
        hc = h * ch
        lead = qg.shape[:-2]
        nq, nk = qg.shape[-2], kv.shape[-2]
        if (qg.shape[-1] != 2 * hc or kv.shape[-1] != 2 * hc
                or kv.shape[:-2] != lead or len(lead) < 1):
            return None
        if (mb.shape != (*lead, 1, nq, nk) or zb.shape != (*lead, h, nq, nk)
                or mb.dtype is not qg.dtype or zb.dtype is not qg.dtype
                or kv.dtype is not qg.dtype
                or not (qg.is_contiguous() and kv.is_contiguous()
                        and mb.is_contiguous() and zb.is_contiguous())):
            return None
        bq = triton.next_power_of_2(nq)
        bk = triton.next_power_of_2(nk)
        bc = triton.next_power_of_2(ch)
        # tl.dot needs each dim >= 16, and the whole [q, k] score tile has to fit
        # in registers because there is no online-softmax loop here.
        if (bq < 16 or bk < 16 or bc < 16 or bc > 128 or bk > 256
                or bq * bk > 8192):
            return None
        nblk = 1
        for d in lead:
            nblk *= d
        out = torch.empty((*lead, nq, hc), dtype=qg.dtype, device=qg.device)
        _sdpa_kernel[(nblk, h)](
            qg, kv, mb, zb, out, nq, nk,
            H=h, CH=ch, SCALE=math.sqrt(ch),
            BLOCK_Q=bq, BLOCK_K=bk, BLOCK_C=bc, num_warps=4,
        )
        return out

    def _z_bias(self, mod, z):
        """`_z_bias_kernel` driver; ``None`` when the layout is not the blocked
        `[1, NBLK, NQ, NK, CZ]` contiguous case the kernel handles."""
        w = mod.linear_z.weight
        if (getattr(mod.linear_z, "bias", None) is not None
                or z.dim() != 5 or z.shape[0] != 1 or not z.is_contiguous()
                or not z.is_cuda or not w.is_contiguous()
                or w.dtype is not z.dtype):
            return None
        h, cz = w.shape
        nblk, nq, nk = z.shape[1], z.shape[2], z.shape[3]
        if z.shape[4] != cz or h > 16 or triton.next_power_of_2(cz) > 128:
            return None
        nrow = nq * nk
        block_r = min(triton.next_power_of_2(nrow), 256)
        if block_r < 16:
            return None
        out = torch.empty((1, nblk, h, nq, nk), dtype=z.dtype, device=z.device)
        _z_bias_kernel[(nblk, triton.cdiv(nrow, block_r))](
            z, w, out, nrow, cz,
            H=h, BLOCK_R=block_r, BLOCK_C=triton.next_power_of_2(cz),
            BLOCK_H=16, num_warps=4,
        )
        return out

    def _attention(self, mha, q_x, kv_x, biases):
        """`OF3Attention.forward` with linear_q|linear_g and linear_k|linear_v
        each stacked into one GEMM (they share an input), and the output gate
        folded into the fused sigmoid-multiply."""
        h, ch = mha.no_heads, mha.c_hidden
        hc = h * ch
        if mha.linear_g is None or not q_x.is_contiguous() or not kv_x.is_contiguous():
            return mha(q_x=q_x, kv_x=kv_x, biases=biases)

        qg = torch.nn.functional.linear(
            q_x, *self._stack(("qg", id(mha)), (mha.linear_q, mha.linear_g)))
        kv = torch.nn.functional.linear(
            kv_x, *self._stack(("kv", id(mha)), (mha.linear_k, mha.linear_v)))

        fused = self._sdpa(qg, kv, biases, h, ch)
        if fused is not None:
            return mha.linear_o(fused)

        def heads(t):
            return t.unflatten(-1, (h, ch)).transpose(-2, -3)

        q = heads(qg[..., :hc]) / math.sqrt(ch)
        k = heads(kv[..., :hc])
        v = heads(kv[..., hc:])

        scores = torch.einsum("...qc,...kc->...qk", q, k)
        for b in biases:
            scores = scores + b
        scores = torch.softmax(scores, dim=-1)
        o = torch.einsum("...qk,...kc->...qc", scores.to(dtype=v.dtype), v)
        o = o.transpose(-2, -3)

        o = self._gate_mul(
            qg[..., hc:].contiguous(),
            o.reshape(*o.shape[:-2], -1).contiguous(),
        )
        return mha.linear_o(o)

    def _ln(self, mod, x: torch.Tensor) -> torch.Tensor:
        """`mod(x)` via `_layer_norm_kernel`, or the module itself if the layout
        is not the plain contiguous last-dim case the kernel handles."""
        c = mod.normalized_shape[0]
        if (not mod.promote_fp32 or not x.is_cuda or x.shape[-1] != c
                or not x.is_contiguous() or x.numel() == 0):
            return mod(x)
        block_c = triton.next_power_of_2(c)
        if block_c > 1024:
            return mod(x)

        w, b = self._ln_affine(mod)
        n_rows = x.numel() // c
        out = torch.empty_like(x)
        block_r = max(1, min(triton.next_power_of_2(n_rows), 2048 // block_c))
        _layer_norm_kernel[(triton.cdiv(n_rows, block_r),)](
            x, w if w is not None else x, b if b is not None else x, out,
            n_rows, c, mod.eps,
            HAS_W=w is not None, HAS_B=b is not None,
            BLOCK_R=block_r, BLOCK_C=block_c,
        )
        return out

    def _ln_affine(self, mod):
        """fp32 copies of `mod.weight` / `mod.bias`, cached like the reference's
        own `_w32` / `_b32` (the reference promotes them once, not per call)."""
        key = id(mod)
        parts = (mod.weight, mod.bias)
        hit = self._ln_cache.get(key)
        if hit is None or hit[0][0] is not parts[0] or hit[0][1] is not parts[1]:
            vals = tuple(
                None if p is None else (p if p.dtype is torch.float32 else p.float())
                for p in parts
            )
            self._ln_cache[key] = (parts, vals)
        return self._ln_cache[key][1]

    def _stack(self, cache_key, mods):
        """One `(weight, bias)` for a list of Linears that all read the same
        input, stacked along the output dim. Bias-free members contribute zeros;
        adding an exact zero is exact, so `addmm` still matches `mm`.

        Only used where K is small (128 here). Verified bit-exact at every
        (M, K, N) this module uses -- see tests/diag_fusable.py. The same trick
        is *not* safe on the K=449 input projection, where widening N changes
        cuBLAS's split-K choice (tests/diag_gemm.py).
        """
        parts = tuple(m.weight for m in mods)
        hit = self._stack_cache.get(cache_key)
        if hit is None or any(a is not b for a, b in zip(hit[0], parts)):
            w = torch.cat(parts, dim=0).contiguous()
            if any(m.bias is not None for m in mods):
                b = torch.cat([
                    m.bias if m.bias is not None
                    else parts[0].new_zeros(m.weight.shape[0])
                    for m in mods
                ]).contiguous()
            else:
                b = None
            self._stack_cache[cache_key] = (parts, (w, b))
        return self._stack_cache[cache_key][1]

    def _wt(self, w):
        """`w.t().contiguous()`, cached on weight identity (K-major for tl.dot)."""
        hit = self._proj_cache.get(id(w))
        if hit is None or hit[0] is not w:
            hit = (w, w.t().contiguous())
            self._proj_cache[id(w)] = hit
        return hit[1]

    def _ada_ln_fused(self, mod, a, s):
        """`_adaln_fused_kernel` driver; ``None`` unless c_s == c_a, both inputs
        are contiguous with matching leading dims, and a row fits in one tile."""
        c = mod.c_a
        lns, lna = mod.layer_norm_s, mod.layer_norm_a
        wg, ws = mod.linear_g.weight, mod.linear_s.weight
        if (mod.c_s != c or not lns.promote_fp32 or not lna.promote_fp32
                or lna.weight is not None or lna.bias is not None
                or mod.linear_s.bias is not None
                or wg.shape != (c, c) or ws.shape != (c, c)
                or not a.is_cuda or not a.is_contiguous() or not s.is_contiguous()
                or a.shape != s.shape or a.dtype is not s.dtype
                or wg.dtype is not a.dtype or ws.dtype is not a.dtype
                or a.shape[-1] != c):
            return None
        block_c = triton.next_power_of_2(c)
        if block_c != c or block_c < 16 or block_c > 128:
            return None
        lnw, lnb = self._ln_affine(lns)
        n_rows = a.numel() // c
        block_r = max(16, min(triton.next_power_of_2(n_rows), 32))
        out = torch.empty_like(a)
        _adaln_fused_kernel[(triton.cdiv(n_rows, block_r),)](
            s, lnw if lnw is not None else s, lnb if lnb is not None else s,
            self._wt(wg), self._wt(ws),
            mod.linear_g.bias if mod.linear_g.bias is not None else s,
            a, out, n_rows, c, lns.eps, lna.eps,
            HAS_LNW=lnw is not None, HAS_LNB=lnb is not None,
            HAS_BG=mod.linear_g.bias is not None,
            BLOCK_R=block_r, BLOCK_C=block_c, num_warps=4,
        )
        return out

    def _ada_ln(self, mod, a, s):
        """`AdaLN.forward` as norm -> one GEMM -> one fused epilogue (3 kernels,
        against the reference's 7 once its LayerNorm casts are counted)."""
        fused = self._ada_ln_fused(mod, a, s)
        if fused is not None:
            return fused
        s_norm = self._ln(mod.layer_norm_s, s)
        c = mod.c_a
        if (mod.layer_norm_a.weight is None and mod.layer_norm_a.bias is None
                and a.is_contiguous() and s_norm.is_contiguous()
                and a.shape[-1] == c and a.shape[:-1] == s_norm.shape[:-1]
                and triton.next_power_of_2(c) <= 1024):
            gs = torch.nn.functional.linear(
                s_norm, *self._stack(("gs", id(mod)), (mod.linear_g, mod.linear_s)))
            out = torch.empty_like(a)
            n_rows = a.numel() // c
            block_c = triton.next_power_of_2(c)
            block_r = max(1, min(triton.next_power_of_2(n_rows), 2048 // block_c))
            _adaln_kernel[(triton.cdiv(n_rows, block_r),)](
                a, gs, out, n_rows, c, mod.layer_norm_a.eps,
                BLOCK_R=block_r, BLOCK_C=block_c,
            )
            return out

        g = torch.sigmoid(mod.linear_g(s_norm))
        return g * (self._ln(mod.layer_norm_a, a) + mod.linear_s(s_norm))

    def _gate_mul(self, g_pre, x, mask=None, res=None):
        """`res + sigmoid(g_pre) * x * mask[..., None]`, in one kernel.

        `mask` and `res` are optional; `res` is the residual stream the caller
        would otherwise add in a separate launch.
        """
        if (not g_pre.is_contiguous() or not x.is_contiguous()
                or g_pre.shape != x.shape
                or (mask is not None and (not mask.is_contiguous()
                                          or mask.shape != x.shape[:-1]))
                or (res is not None and (not res.is_contiguous()
                                         or res.shape != x.shape
                                         or res.dtype is not x.dtype))):
            out = torch.sigmoid(g_pre) * x
            if mask is not None:
                out = out * mask.unsqueeze(-1)
            return out if res is None else res + out
        out = torch.empty_like(x)
        n = x.numel()
        _gate_mul_kernel[(triton.cdiv(n, 1024),)](
            g_pre, x, mask if mask is not None else x,
            res if res is not None else x, out,
            n, x.shape[-1], HAS_MASK=mask is not None, HAS_RES=res is not None,
            BLOCK=1024,
        )
        return out

    def _swiglu(self, mod, x):
        """`SwiGLU.forward`: one stacked GEMM plus one fused silu-multiply."""
        c_out = mod.linear_a.weight.shape[0]
        if not x.is_contiguous() or triton.next_power_of_2(c_out) > 1024:
            return mod(x)
        ab = torch.nn.functional.linear(
            x, *self._stack(("ab", id(mod)), (mod.linear_a, mod.linear_b)))
        out = torch.empty((*x.shape[:-1], c_out), dtype=x.dtype, device=x.device)
        n_rows = out.numel() // c_out
        block_c = triton.next_power_of_2(c_out)
        block_r = max(1, min(triton.next_power_of_2(n_rows), 2048 // block_c))
        _swiglu_kernel[(triton.cdiv(n_rows, block_r),)](
            ab, out, n_rows, c_out, BLOCK_R=block_r, BLOCK_C=block_c,
        )
        return out

    def _transition_fused(self, mod, x, s, mask, res):
        """`_transition_kernel` driver; ``None`` unless every width fits one tile
        and all operands are contiguous with the same row count."""
        wa, wb = mod.swiglu.linear_a.weight, mod.swiglu.linear_b.weight
        wo, wg = mod.linear_out.weight, mod.linear_g.weight
        if (mod.swiglu.linear_a.bias is not None
                or mod.swiglu.linear_b.bias is not None
                or mod.linear_out.bias is not None):
            return None
        ch, c = wa.shape
        co = wo.shape[0]
        if (wb.shape != (ch, c) or wo.shape != (co, ch) or wg.shape[0] != co
                or wg.shape[1] != s.shape[-1] or x.shape[-1] != c
                or not x.is_cuda or not x.is_contiguous() or not s.is_contiguous()
                or x.shape[:-1] != s.shape[:-1]
                or x.dtype is not s.dtype or wa.dtype is not x.dtype
                or wb.dtype is not x.dtype or wo.dtype is not x.dtype
                or wg.dtype is not x.dtype or s.shape[-1] != c):
            return None
        if mask is not None and (not mask.is_contiguous()
                                 or mask.shape != x.shape[:-1]
                                 or mask.dtype is not x.dtype):
            return None
        if res is not None and (not res.is_contiguous()
                                or res.shape != (*x.shape[:-1], co)
                                or res.dtype is not x.dtype):
            return None
        bc, bh, bo = (triton.next_power_of_2(c), triton.next_power_of_2(ch),
                      triton.next_power_of_2(co))
        if bc != c or bh != ch or bo != co or bc > 128 or bh > 256 or bo > 128:
            return None
        if bc < 16 or bh < 16 or bo < 16:
            return None
        n_rows = x.numel() // c
        block_r = max(16, min(triton.next_power_of_2(n_rows), 16))
        out = torch.empty((*x.shape[:-1], co), dtype=x.dtype, device=x.device)
        _transition_kernel[(triton.cdiv(n_rows, block_r),)](
            x, s, self._wt(wa), self._wt(wb), self._wt(wo), self._wt(wg),
            mod.linear_g.bias if mod.linear_g.bias is not None else s,
            mask if mask is not None else s, res if res is not None else s,
            out, n_rows, c, ch, co,
            HAS_BG=mod.linear_g.bias is not None, HAS_MASK=mask is not None,
            HAS_RES=res is not None,
            BLOCK_R=block_r, BLOCK_C=bc, BLOCK_H=bh, BLOCK_O=bo, num_warps=8,
        )
        return out

    def _conditioned_transition(self, mod, a, s, mask, res=None):
        an = self._ada_ln(mod.layer_norm, a, s)
        fused = self._transition_fused(mod, an, s, mask, res)
        if fused is not None:
            return fused
        b = self._swiglu(mod.swiglu, an)
        return self._gate_mul(mod.linear_g(s), mod.linear_out(b), mask, res=res)

    def _forward_impl(
        self,
        token_features: torch.Tensor,
        residue_index: torch.Tensor,
        batch: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if batch is not None and "ref_pos" in batch:
            a = self._encode_atoms(batch)
            if a is None:
                a, _, _, _ = self.atom_attn_enc(batch=batch)
            s_input = torch.cat(
                [
                    a,
                    batch.get("restype", token_features[..., :32]),
                    batch.get("profile", token_features[..., 32:64]),
                    batch.get("deletion_mean", token_features[..., -1:]).unsqueeze(-1)
                    if batch.get("deletion_mean") is not None and batch["deletion_mean"].dim() == token_features.dim() - 1
                    else batch.get("deletion_mean", token_features[..., -1:]),
                ],
                dim=-1,
            )
        else:
            s_input = token_features

        # linear_s / linear_z_i / linear_z_j are deliberately left as three
        # GEMMs. Stacking their weights into one [c_s+2*c_z, c_s_input] GEMM does
        # collapse three launches into one, but it measured 1.00x (the region is
        # 0.6% of the forward) and it is a real accuracy hazard: widening N from
        # c_s to c_s+2*c_z changes cuBLAS's split-K choice, and with
        # `allow_bf16_reduced_precision_reduction` on (torch's default) those
        # partials reduce in bf16. See tests/diag_gemm.py -- at some shapes the
        # fused and unfused `s` disagree on >25% of elements. No time for real risk.
        s = self._proj(self.linear_s, s_input)
        if s is None:
            s = self.linear_s(s_input)
        z_i = self._proj(self.linear_z_i, s_input)
        if z_i is None:
            z_i = self.linear_z_i(s_input)
        z_j = self._proj(self.linear_z_j, s_input)
        if z_j is None:
            z_j = self.linear_z_j(s_input)

        z = self._fused_epilogue(z_i, z_j, batch)
        if z is None:
            z = self._epilogue(z_i, z_j, residue_index, batch)

        return s_input, s, z

    # ------------------------------------------------------------------
    # Pair-representation epilogue.
    # ------------------------------------------------------------------
    def _fused_epilogue(self, z_i: torch.Tensor, z_j: torch.Tensor,
                        batch: dict | None):
        """Single-kernel z. Returns ``None`` when the fast path does not apply."""
        if batch is None or not _RELPOS_KEYS.issubset(batch.keys()):
            return None
        if (z_i.dim() < 2 or not z_i.is_cuda or not z_i.is_contiguous()
                or not z_j.is_contiguous() or z_j.shape != z_i.shape
                or z_j.dtype is not z_i.dtype or z_i.shape[-1] != self.c_z):
            return None

        pos = [batch[k] for k in ("residue_index", "token_index",
                                  "asym_id", "entity_id", "sym_id")]
        lead = z_i.shape[:-2]
        n = z_i.shape[-2]
        dt = pos[0].dtype
        round_mode = _ROUND_BY_DTYPE.get(dt, _ROUND_NONE)
        for p in pos:
            if (p.dtype is not dt or p.shape[:-1] != lead or p.shape[-1] != n
                    or not p.is_contiguous()):
                return None

        c_z = self.c_z
        block_c = triton.next_power_of_2(c_z)
        if block_c > 1024:
            return None

        tb = batch.get("token_bonds")
        if tb is not None:
            if (tb.shape[:-2] != lead or tb.shape[-2:] != (n, n)
                    or not tb.is_contiguous()):
                return None
            # The reference casts token_bonds to the output dtype *before* the
            # projection, so match that rounding here.
            if tb.dtype is not z_i.dtype:
                tb = tb.to(dtype=z_i.dtype)

        cum_pos, cum_tok, w_se, cum_chain, _ = self._relpos_tables()
        w_tb = self._token_bond_weight()

        n_rows = z_i.numel() // c_z
        z = torch.empty((*lead, n, n, c_z), dtype=z_i.dtype, device=z_i.device)
        block_j = min(triton.next_power_of_2(n), 64)
        grid = (n_rows, triton.cdiv(n, block_j))
        _z_epilogue_kernel[grid](
            z_i, z_j, z,
            pos[0], pos[1], pos[2], pos[3], pos[4],
            cum_pos, cum_tok, cum_chain, w_se,
            tb if tb is not None else z_i, w_tb,
            n, c_z,
            K=self.relpos_k, KC=self.max_relative_chain,
            HAS_TB=tb is not None, ROUND=round_mode,
            BLOCK_J=block_j, BLOCK_C=block_c,
        )
        return z

    def _token_bond_weight(self) -> torch.Tensor:
        """``linear_token_bonds.weight[:, 0]`` as fp32, cached."""
        w = self.linear_token_bonds.weight
        if self._tb_w is None or self._tb_src is not w:
            self._tb_src = w
            self._tb_w = w[:, 0].float().contiguous()
        return self._tb_w

    def _epilogue(
        self, z_i: torch.Tensor, z_j: torch.Tensor,
        residue_index: torch.Tensor, batch: dict | None,
    ) -> torch.Tensor:
        """Reference-order epilogue, used when the fused kernel does not apply."""
        z = z_i[..., :, None, :] + z_j[..., None, :, :]

        if batch is not None and _RELPOS_KEYS.issubset(batch.keys()):
            z = z + self._relpos_emb(batch, z.dtype)
        elif batch is not None and "asym_id" in batch:
            # Some relpos feature is missing -- keep the reference path.
            relpos_feats = relpos_complex(
                batch=batch,
                max_relative_idx=self.relpos_k,
                max_relative_chain=self.max_relative_chain,
            ).to(dtype=z.dtype)
            z = z + self.linear_relpos(relpos_feats)
        else:
            # True one-hot over the first 2k+2 bins, zero-padded to the full
            # feature width, so the GEMM degenerates to a row gather.
            d = residue_index[..., :, None] - residue_index[..., None, :]
            d = d.clamp(-self.relpos_k, self.relpos_k) + self.relpos_k
            z = z + self._relpos_tables()[4][d.long()].to(dtype=z.dtype)

        if batch is not None and "token_bonds" in batch:
            token_bonds_emb = self.linear_token_bonds(
                batch["token_bonds"].unsqueeze(-1).to(dtype=z.dtype)
            )
            z = z + token_bonds_emb

        return z
