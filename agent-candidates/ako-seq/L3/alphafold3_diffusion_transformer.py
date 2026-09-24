"""Diffusion transformer for AlphaFold3.

24-block transformer used inside the diffusion module. Each block:
AttentionPairBias + ConditionedTransitionBlock (AdaLN-Zero).

Reference: openfold3/core/model/layers/diffusion_transformer.py

The three captured workloads are two different operators wearing one class name,
and the measured cost model says they want opposite things. Four measurements set
the design for both (all under the harness's own conditions -- it enqueues a
253 MiB L2 flush before its start event, so every weight is cold; see ``dev/``
and ITERATIONS.md):

* A kernel costs a **fixed ~4.5 us eagerly**, or ~1.5 us as a CUDA-graph node,
  *plus* bytes/5.7 TB/s. Reading 1, 2 or 4 MiB with >=64 programs all measured
  the same 4.47 us; bandwidth only starts to bind above ~16 MiB.
* Launch floors, back to back: **2.76 us** eager, **1.26 us** with PDL (same host
  cost), **1.01 us** as a graph node with ~2.5 us of host time for the whole
  replay. But node cost is not the whole story: capturing the cross path's chain
  measured **72.9 us against 69.2 eager with PDL**, because PDL's overlap (each
  kernel's weight and table loads issue while its producer drains) is worth more
  than the 0.25 us/node the graph saves. The self path's 121 nodes are captured
  because there the host, not the overlap, is what binds.
* **Cold HBM is not what either path is waiting on.** The self path's whole
  forward measures 672.8 us with the L2 flushed and 662.9 us with no flush at all
  -- 1.5% -- and one block's five kernels go 37.9 -> 32.1 us with that block's
  entire 12.4 MiB weight pack already resident. So prefetching, second streams
  and any other latency-hiding scheme are bounded by a prize that is already
  nearly zero. What the kernels are short of is **programs and warps**, not
  bytes in flight: a pure streaming read of ``_s_tab``'s 81 MiB in its own layout
  hits 5.82 TB/s, and it is the ``M=16`` dot on top that costs the other 17 us.
  Every kernel here is therefore sized by measured occupancy, and the row/query
  splits (``_AT_BQ``, ``_LO_BM``, ``_OTX_BN``, ...) exist for that reason alone.
* Rearranging a dot to avoid ``M=16`` does **not** help: computing
  ``out^T[BN,16] = w^T[BN,K] @ y^T[K,16]``, so that M is the output-column block
  and a full MMA tile, measured 33.0 us against the natural form's 31.8.

* **Cross-attention** (``n_query=32``/``n_key=128``, ``c_a=128 c_s=128 c_z=16``,
  ``no_heads=4``, ``no_blocks=3``, ``N=368``): ~500 MMAC and 1.4 MiB of weights
  for the whole stack, so the reference's 5.2-5.8 ms is ~1500 dispatches and
  nothing else. **Sixteen PDL-chained kernels**, adapted from the frozen L2
  ``alphafold3_atom_attention`` winner -- whose ``atom_transformer`` *is* this
  class at exactly these shapes -- with its structural probe and its numerics
  helpers imported rather than re-derived: one merged prologue (``_c_pr``) and
  five per block. Nothing here is weight-bound (the whole arena is L2-resident),
  so every kernel is cut until it fills the machine: the winner's ``_k_xa``
  became a row-wise projection (``_c_pq``) plus a pure attention (``_c_at``,
  split by query strip as well as head), and its ``_k_xb`` became
  ``_c_lo``/``_c_sw``/``_c_ot``, both of the latter column-split.
  5.2 ms -> 67 us.

* **Self-attention** (``c_a=768 c_s=384 c_z=128``, ``no_heads=16``,
  ``c_hidden=48``, ``no_blocks=24``, only ``N=16`` rows): the one variant with
  real bytes to move -- 12.4 MiB of bf16 weights per block, 378 MiB for the stack
  -- though not enough for bandwidth to be the binding constraint. With ``M=16``
  there is no row axis to split, so every GEMM is split along output columns; two
  prologue kernels plus five per block is 123 dependent launches, which at 3.1 us
  of host time each would be 380 us of host against ~66 us of streaming -- so the
  chain is **captured as a CUDA graph**, with the input staging launched eagerly
  in front of it and the last block's output projection eagerly behind it
  (writing straight into the caller's output, so nothing is copied back).
  16.3 ms -> 632 us.

Everything block-invariant is hoisted out of the block loop in both paths. ``s``
and ``z`` do not change across blocks, so all blocks' AdaLN scale/shift rows,
both output gates per block and every block's ``linear_z`` pair bias are computed
once in a prologue -- all blocks' ``linear_z`` as a single
``[c_z, no_blocks*no_heads]`` matmul, the AdaLN/gate tables as one row-wise
kernel emitting ``6*no_blocks`` (self) or ``8*no_blocks`` (cross) per-row tables
that the block kernels only gather rows from.

Numerics are the real risk, not speed: 24 sequential residual blocks compound
bf16 error. So both paths round to bf16 at exactly the points torch does rather
than keeping fp32 (``_rb``), and reproduce the reference's primitives rather than
approximating them -- ``F.layer_norm``'s two-pass moments (``_lnf``/``_mom``),
``torch.sigmoid`` (``_sig``), ``F.silu`` (``_silu``), ``std::exp`` in the softmax
(``_expf``) and a single K=no_heads*c_hidden ``linear_o``. Those are the
*baseline* L1 ops, not the candidate L1 winners: a baseline module's relative
import resolves inside the baseline tree, so that is what the reference runs.
Measured worst case over activation scales 0.02 to 20 and partial masks: 1.0000
matched, 1-2 bf16 ulps of deviation.

Anything the fast paths do not cover -- a non-bf16 dtype, a batch that does not
collapse to one row, a missing mask, ``_mask_trans=False``, grad enabled, a
misaligned pointer, a structural variation the probes decline, or a Triton that
refuses launcher memoization -- falls back to the reference composition below,
which is kept intact.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._C import _cuda_getCurrentRawStream as _raw_stream
from torch._C import _cuda_getDevice as _cur_device

try:                                    # bit-exact fp32 exp (see _expf)
    from triton.language.extra import libdevice as _libdev
except Exception:                       # noqa: BLE001 - binding moved
    _libdev = None

try:  # Triton 3.6+; probed, not assumed
    from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait
    _HAS_PDL = True
except Exception:                       # noqa: BLE001 - older Triton
    _HAS_PDL = False

from ..L2.alphafold3_attention_pair_bias import AttentionPairBias, CrossAttentionPairBias
from ..L2.alphafold3_swiglu_transition import ConditionedTransitionBlock

try:
    from ..L2.alphafold3_atom_attention import (
        _adaln,
        _align,
        _bias_of,
        _expf,
        _geom,
        _n_real,
        _lin_ok,
        _lnf,
        _rb,
        _sig,
        _silu,
        _t2,
        _xf_probe,
    )
    _HAS_XF = True
except Exception:                       # noqa: BLE001 - frozen winner moved
    _HAS_XF = False


__targets__ = ["DiffusionTransformer"]


class _Unsupported(Exception):
    """This (module, shape) pair is not something a fused path reproduces."""


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


def _drop_plan(module, incompatible_keys):
    """Invalidate the cached weight pack after a state-dict load: the pack is a
    transposed *copy*, so an in-place weight update must rebuild it."""
    module._plan = None
    module._no_fast = False


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

        # Fused-path state: derived on the first supported call, invalidated by
        # a weight reload (the packed weights are transposed copies).
        self._n_query = n_query
        self._n_key = n_key
        self._plan = None
        self._no_fast = not _HAS_XF
        self.register_load_state_dict_post_hook(_drop_plan)

    # -- fast path ---------------------------------------------------------
    def _build_plan(self, a, s, z, mask):
        """Compile the fused stack for this (module, input-shape) pair. Raises
        ``_Unsupported`` (caught by the caller) for anything the fast paths do not
        reproduce, which then means "always use the reference"."""
        if not _HAS_XF:
            return None
        if self.use_cross_attention:
            return _CxPlan(self, a, s, z, mask)
        return _SfPlan(self, a, s, z, mask)

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
        # The dtype test is here rather than only in ``matches`` so an
        # unsupported dtype does not pay for a plan build (which copies every
        # weight) that could never be used.
        if (not self._no_fast and _mask_trans and mask is not None
                and a.is_cuda and not torch.is_grad_enabled()
                and a.dtype is torch.bfloat16 and s.dtype is torch.bfloat16
                and z.dtype is torch.bfloat16 and mask.dtype is torch.bfloat16
                and a.dim() >= 2 and s.dim() >= 2
                and _prod(a.shape[:-2]) == 1 and _prod(s.shape[:-2]) == 1):
            p = self._plan
            if p is None:
                try:
                    p = self._build_plan(a, s, z, mask)
                except Exception:  # noqa: BLE001 - unsupported: use the reference
                    p = None
                self._plan = p
                if p is None:
                    self._no_fast = True
            if (p is not None and _cur_device() == p.dix
                    and p.matches(a, s, z, mask)):
                out = torch.empty(a.shape, device=a.device, dtype=a.dtype)
                p.run(a, s, z, mask, out)
                return out

        if self.use_cross_attention:
            z = self.layer_norm_z(z)

        for block in self.blocks:
            a = block(a=a, s=s, z=z, mask=mask)

        return a


def _prod(shape) -> int:
    out = 1
    for s in shape:
        out *= int(s)
    return out


# ###########################################################################
# Fused self-attention stack (the 24-block variant)
# ###########################################################################
# 16 rows, 768 channels, 24 blocks: 16.5 MiB of bf16 weights per block and
# 396 MiB for the stack, against ~150 MMAC of arithmetic per block. So this path
# is a *weight stream* with a launch chain wrapped around it, and the three
# costs are attacked separately.
#
# **Bytes in flight.** With M=16 there is no row axis to split, so output columns
# are the only axis there is, and every GEMM is split along it: per-program
# weight bytes fall by the split factor and each program's loads are independent,
# so a kernel's *whole* weight slab can be in flight at once (the
# ``alphafold3_pair_block`` lesson: 80 programs loading a 32 KiB column slice
# beat 8 loading all 160 KiB, 6.5 us -> 2.15 us).
#
# **Coalescing.** A column strip of a row-major ``[K, N]`` weight is a stride-N
# gather of BN*2 bytes per row -- at BN=32 that is 64 B out of every 128 B line,
# so half the bandwidth is thrown away before the split factor is even chosen.
# Every weight here is therefore *pre-tiled* at plan time into
# ``[column_block, K, BN]`` (``_tile``), which makes each program's whole
# ``[BK, BN]`` tile one contiguous run and decouples the choice of BN from the
# line size. Measured: this plus pipelining took the four block kernels from
# 10.6 / 24.2 / 10.9 / 8.6 us to a fraction of that (see ITERATIONS.md).
#
# **Dispatch.** Five kernels per block plus two prologue kernels is 123 dependent
# launches, 121 of them captured. Measured launch floors on this box (``dev/floor2.py``): eagerly
# 2.76 us of device serialization and 3.1 us of host time each; with PDL 1.26 us
# of device time but the same host cost; as CUDA-graph nodes **1.01 us of device
# time and ~2.5 us of host time for the whole replay**. At >100 launches the host
# is what binds, so the chain is captured. PDL *inside* the graph measured worse
# (1.33 us/node), so the captured kernels do not use it.
#
# Capture needs constant node arguments, so the caller's four inputs are staged
# into plan-owned buffers by one eager kernel in front of the graph, and the last
# block's output projection is launched eagerly *behind* it, writing straight
# into the caller's freshly allocated output -- no copy back, and only two
# launches pay eager cost.
#
# Block-invariant work is hoisted exactly as on the cross path: ``s`` and ``z``
# never change across blocks, so ``_stab_body`` emits all 24 blocks' six per-row
# tables (both AdaLNs' gate and shift, plus the two output gates) and ``_szb_body``
# all 24 blocks' pair bias, once, before the block loop -- and since the second
# needs only the staging kernel, the two run on **one grid** (``_s_tz``) rather
# than as two serialized nodes. The block kernels only
# gather rows from them -- the alternative, recomputing the AdaLN conditioning
# per block, would read the same 85 MiB of ``[c_s, c_a]`` weight four times.

# Tile shapes. The column block is what the pre-tiled layout makes free to
# choose; ``num_warps`` and the reduction chunk were swept in-plan (dev/sweep.py).
_TAB_BN = 64     # _stab_body output columns per program
_Q_BN, _Q_BK = 32, 256      # _s_qkvg output columns / reduction chunk
_LO_BN, _LO_BK = 8, 128     # _s_lo   output columns / reduction chunk
_SW_BN, _SW_BK = 32, 256    # _s_sw   hidden columns / reduction chunk
_OT_BN, _OT_BK = 16, 512    # _s_out  output columns / reduction chunk
_ZB_BM = 64      # _szb_body pair rows per program
_IN_BL = 1024    # _s_in   elements per program
_W_TAB, _W_Q, _W_LO, _W_SW, _W_OT = 8, 4, 1, 8, 2
_ST_Q, _ST_LO, _ST_SW, _ST_OT = 3, 8, 3, 4
_W_AT_S = 4      # _s_at warps
_AT_DS = 1       # _s_at: split a head's output channels this many ways


@triton.jit
def _mom(X, r, rm, C: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
         NK: tl.constexpr, EPS: tl.constexpr):
    """``F.layer_norm``'s two-pass moments of rows ``r`` of ``X`` -- the
    ``(mean, rstd)`` pair ``_lnf`` computes internally -- read in ``BK``-wide
    chunks so a column-chunked consumer can normalize the row a chunk at a time
    without ever holding a ``[BM, next_pow2(C)]`` tile.

    Reading the row twice more out of L1 is *much* cheaper than holding it: on the
    projection shape ([16,768] x [768,3072]) the whole-row form costs 4.1 us over
    the bare GEMM against 2.3 us for this one and 2.0 us for moments handed in
    already computed, so the chunked read is within 0.3 us of free
    (``dev/ada.py``). Two passes rather than ``E[x^2]-E[x]^2`` because that is
    what reproduces ``F.layer_norm`` -- see ``_lnf``."""
    s1 = tl.zeros((BM,), tl.float32)
    for ci in tl.range(0, NK):
        j = ci * BK + tl.arange(0, BK)
        m = rm[:, None] & (j < C)[None, :]
        x = tl.load(X + r[:, None] * C + j[None, :], mask=m, other=0.0).to(tl.float32)
        s1 += tl.sum(tl.where(m, x, 0.0), 1)
    mu = s1 / C
    s2 = tl.zeros((BM,), tl.float32)
    for ci in tl.range(0, NK):
        j = ci * BK + tl.arange(0, BK)
        m = rm[:, None] & (j < C)[None, :]
        x = tl.load(X + r[:, None] * C + j[None, :], mask=m, other=0.0).to(tl.float32)
        d = tl.where(m, x - mu[:, None], 0.0)
        s2 += tl.sum(d * d, 1)
    return mu, tl.rsqrt(s2 / C + EPS)


@triton.jit
def _adc(X, GP, SP, r, j, msk, C: tl.constexpr, mu, rstd):
    """One reduction chunk of ``AdaLN(x, s)`` for rows ``r``, columns ``j``:
    ``g * (layer_norm(x) + linear_s(layer_norm_s(s)))`` with the two ``s``-side
    terms read from the precomputed per-row tables and the row's moments passed
    in. Rounds to bf16 at the three points the reference rounds."""
    xk = tl.load(X + r[:, None] * C + j[None, :], mask=msk, other=0.0).to(tl.float32)
    sa = tl.load(SP + r[:, None] * C + j[None, :], mask=msk, other=0.0)
    ga = tl.load(GP + r[:, None] * C + j[None, :], mask=msk, other=0.0)
    y = ((xk - mu[:, None]) * rstd[:, None]).to(tl.bfloat16)
    t = (y.to(tl.float32) + sa.to(tl.float32)).to(tl.bfloat16)
    return tl.where(msk, ga.to(tl.float32) * t.to(tl.float32), 0.0).to(tl.bfloat16)


@triton.jit
def _s_in(A, S, Z, MK, SA, SS, SZ, SM,
          NA: tl.constexpr, NS: tl.constexpr, NZ: tl.constexpr,
          NM: tl.constexpr, BL: tl.constexpr):
    """Stage the caller's four inputs into the plan's fixed buffers.

    The only kernel that touches a pointer the plan does not own, which is what
    lets every node behind it be captured once. Launched eagerly in front of the
    graph."""
    o = tl.program_id(0) * BL + tl.arange(0, BL)
    ma = o < NA
    tl.store(SA + o, tl.load(A + o, mask=ma, other=0), mask=ma)
    ms = o < NS
    tl.store(SS + o, tl.load(S + o, mask=ms, other=0), mask=ms)
    mz = o < NZ
    tl.store(SZ + o, tl.load(Z + o, mask=mz, other=0), mask=mz)
    mm = o < NM
    tl.store(SM + o, tl.load(MK + o, mask=mm, other=0), mask=mm)


@triton.jit
def _s_sn(S, Y, TG,
          N: tl.constexpr, CS: tl.constexpr, BM: tl.constexpr,
          BS: tl.constexpr, EPS: tl.constexpr):
    """``layer_norm_s(s)`` scaled by each table's own gamma, one program per
    table.

    Split out of ``_s_tab`` rather than recomputed there. ``_s_tab`` has 1728
    programs (the finest split in the operator, because its 144 ``[c_s, c_a]``
    weights are 85 MiB -- a fifth of the whole stack), and each one recomputing
    this norm read ``s`` four more times and put two cross-lane reductions in
    front of its first dot: 110 MiB of L2 traffic and a serial prologue, to
    produce 1.7 MiB. Hoisted, ``_s_tab`` becomes a single-round-trip GEMM and
    this costs one launch and 144 programs. Measured 37.8 us -> 4.0 + 16.6."""
    g = tl.program_id(0)
    r = tl.arange(0, BM)
    j = tl.arange(0, BS)
    jm = j < CS
    msk = (r < N)[:, None] & jm[None, :]
    x = tl.load(S + r[:, None] * CS + j[None, :], mask=msk, other=0.0).to(tl.float32)
    gam = tl.load(TG + g * CS + j, mask=jm, other=0.0).to(tl.float32)
    # Tables 0..3 consume layer_norm_s(s); tables 4 and 5 consume raw s.
    y = tl.where((g % 6) < 4, _lnf(x, j, CS, EPS) * gam[None, :], x)
    tl.store(Y + g * (N * CS) + r[:, None] * CS + j[None, :],
             y.to(tl.bfloat16), mask=msk)


@triton.jit
def _stab_body(pid, Y, ST, TW, TB,
              N: tl.constexpr, CS: tl.constexpr, C: tl.constexpr,
              NCS: tl.constexpr, BM: tl.constexpr, BS: tl.constexpr,
              BN: tl.constexpr):
    """Every ``s``-dependent quantity of the whole stack: one program per
    (output table, column strip), one dot each.

    Per block the tables are the attention AdaLN's gate and additive shift, the
    transition AdaLN's, and the two output gates (``linear_ada_out`` and the
    transition's ``linear_g``) -- ``6 * no_blocks`` tables of ``[N, c_a]``, the
    ``ST`` layout every block kernel gathers rows from. Their weights are 85 MiB
    between them, a fifth of the stack, so this gets the finest column split:
    1728 programs of 49 KiB rather than 144 of 590 KiB. Every load is issued
    before the dot, so the program is one memory round trip deep."""
    g = pid // NCS
    r = tl.arange(0, BM)
    rm = r < N
    j = tl.arange(0, BS)
    jm = j < CS
    o = (pid % NCS) * BN + tl.arange(0, BN)
    om = o < C
    y = tl.load(Y + g * (N * CS) + r[:, None] * CS + j[None, :],
                mask=rm[:, None] & jm[None, :], other=0.0)
    w = tl.load(TW + (g * NCS + pid % NCS) * (CS * BN) + j[:, None] * BN
                + tl.arange(0, BN)[None, :], mask=jm[:, None], other=0.0)
    b = tl.load(TB + g * C + o, mask=om, other=0.0).to(tl.float32)
    acc = _rb(tl.dot(y, w) + b[None, :])
    # The gates are sigmoid'd; the additive shifts (odd tables below 4) are not.
    out = tl.where(((g % 6) >= 4) | ((g % 2) == 0), _sig(acc), acc)
    tl.store(ST + g * (N * C) + r[:, None] * C + o[None, :],
             out.to(tl.bfloat16), mask=rm[:, None] & om[None, :])


@triton.jit
def _szb_body(pid, Z, ZB, GZ, WZ,
              NN: tl.constexpr, P: tl.constexpr, H: tl.constexpr,
              HP: tl.constexpr, NRS: tl.constexpr, BM: tl.constexpr,
              BP: tl.constexpr, EPS: tl.constexpr):
    """All blocks' attention pair biases, one program per (block, pair-row
    strip).

    Unlike the cross-attention stack, ``layer_norm_z`` lives *inside* each block
    here -- but ``z`` does not change across blocks, so the whole thing is still
    block-invariant: 24 weight-only norms of the same ``[N*N, c_z]`` tile, each
    followed by its own ``c_z -> no_heads`` projection (the norm's gamma is
    rounded to bf16 before the projection, so the 24 cannot be folded into one
    matmul). Stored transposed to ``[block, head, q, k]`` so the attention
    kernel's read walks the key axis contiguously."""
    b = pid // NRS
    r = (pid % NRS) * BM + tl.arange(0, BM)
    rm = r < NN
    j = tl.arange(0, BP)
    jm = j < P
    o = tl.arange(0, HP)
    x = tl.load(Z + r[:, None] * P + j[None, :],
                mask=rm[:, None] & jm[None, :], other=0.0).to(tl.float32)
    gam = tl.load(GZ + b * P + j, mask=jm, other=0.0).to(tl.float32)
    y = (_lnf(x, j, P, EPS) * gam[None, :]).to(tl.bfloat16)
    w = tl.load(WZ + b * (P * HP) + j[:, None] * HP + o[None, :],
                mask=jm[:, None] & (o < H)[None, :], other=0.0)
    tl.store(ZB + b * (HP * NN) + o[:, None] * NN + r[None, :],
             tl.trans(tl.dot(y, w).to(tl.bfloat16)), mask=rm[None, :])


@triton.jit
def _s_tz(Y, ST, TW, TB, Z, ZB, GZ, WZ,
          N: tl.constexpr, CS: tl.constexpr, C: tl.constexpr,
          NCS: tl.constexpr, BM: tl.constexpr, BS: tl.constexpr,
          BN: tl.constexpr, NTAB: tl.constexpr, NN: tl.constexpr,
          P: tl.constexpr, H: tl.constexpr, HP: tl.constexpr,
          NRS: tl.constexpr, ZBM: tl.constexpr, BP: tl.constexpr,
          EPS: tl.constexpr):
    """``_s_tab`` and ``_s_zb`` on one grid.

    ``_s_zb`` depends only on the staging kernel, not on ``_s_sn``, so it is free
    to run alongside ``_s_tab`` -- but on one stream (and as two graph nodes) it
    cannot. ``_s_tab`` is 1728 programs of ~48 KiB streaming 81 MiB in ~12 waves;
    ``_s_zb``'s 96 programs are a rounding error on that, against 10.2 us of its
    own node otherwise. Same argument as ``_c_pr`` on the cross path."""
    pid = tl.program_id(0)
    if pid < NTAB:
        _stab_body(pid, Y, ST, TW, TB, N, CS, C, NCS, BM, BS, BN)
    else:
        _szb_body(pid - NTAB, Z, ZB, GZ, WZ, NN, P, H, HP, NRS, ZBM, BP, EPS)


@triton.jit
def _s_qkvg(A, ST, QKVG, W, BQ,
            N: tl.constexpr, C: tl.constexpr, CH: tl.constexpr,
            NCH: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
            NK: tl.constexpr, BN: tl.constexpr, EPS: tl.constexpr,
            RSQ: tl.constexpr):
    """The block's AdaLN and all four attention projections, one program per
    (projection, output column strip).

    ``q``/``k``/``v``/``g`` share one pre-tiled weight, so the projection index
    is just the high bits of the column task and the reduction loop has no
    branch. The AdaLN is row-complete work, so each program redoes it for its own
    16 rows -- 72 KiB of L2-resident activation and table against the 4.7 MiB of
    weight the split exists to spread, which is what lets all four projections be
    one launch.

    ``q`` is stored already scaled by ``1/sqrt(c_hidden)`` and ``g`` already
    through its sigmoid (both with the reference's roundings), so the attention
    kernel is pure attention."""
    pid = tl.program_id(0)
    task = pid // NCH
    r = tl.arange(0, BM)
    rm = r < N
    o = (pid % NCH) * BN + tl.arange(0, BN)
    om = o < CH
    mu, rstd = _mom(A, r, rm, C, BM, BK, NK, EPS)
    wb = W + pid * (C * BN)
    tj = tl.arange(0, BK)[:, None] * BN + tl.arange(0, BN)[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for ci in tl.range(0, NK):
        j = ci * BK + tl.arange(0, BK)
        jm = (j < C)[:, None]
        x1 = _adc(A, ST, ST + N * C, r, j, rm[:, None] & (j < C)[None, :],
                  C, mu, rstd)
        w = tl.load(wb + ci * (BK * BN) + tj, mask=jm, other=0.0)
        acc = tl.dot(x1, w, acc)
    # q also picks up ``linear_q``'s bias and the 1/sqrt(c_hidden) scale, g goes
    # through the sigmoid; k and v are the bare projection. Selected rather than
    # branched -- the extra work is a 64-lane bias load and one sigmoid.
    qq = (_rb(acc + tl.load(BQ + o, mask=om, other=0.0).to(tl.float32)[None, :])
          / RSQ).to(tl.bfloat16)
    gg = _sig(_rb(acc)).to(tl.bfloat16)
    out = tl.where(task == 0, qq, tl.where(task == 3, gg, _rb(acc).to(tl.bfloat16)))
    tl.store(QKVG + task * (N * CH) + r[:, None] * CH + o[None, :], out,
             mask=rm[:, None] & om[None, :])


@triton.jit
def _s_at(QKVG, ZB, MK, OG,
          N: tl.constexpr, CH: tl.constexpr, D: tl.constexpr,
          BM: tl.constexpr, BD: tl.constexpr, DS: tl.constexpr,
          DSZ: tl.constexpr, BDS: tl.constexpr, INF: tl.constexpr):
    """The gated attention output of one head.

    One program per head rather than a head loop inside ``linear_o``: as a loop
    each head is a chain of ~4 dependent memory round trips, and sixteen of those
    in series measured **24 us** against 4.7 + 6.8 for this kernel plus a plain
    ``linear_o``. Unrolling the loop with ``static_range`` does not fix it (also
    23.7 us) -- the scheduler will not hoist a load above the softmax reduction
    that precedes it. Run as 16 parallel programs the whole thing is one round
    trip: every load is issued before the first reduction.

    There are no weights here; the projections are already in ``QKVG``.
    ``K <= 32``, so the whole score row lives in registers -- no online softmax
    and no ``[H, Q, K]`` score tensor in HBM. Padded key lanes read a zero mask,
    so their bias is ``-inf`` and they drop out of the softmax on their own.

    ``DS`` splits a head's ``c_hidden`` output channels across programs. The
    scores have to be recomputed by each piece (``q``/``k`` are read in full), but
    they come out of an L2-resident arena and the second dot and the store shrink
    with the split -- the same trade that took the cross path's ``_c_at`` from
    6.16 to 2.94 us. ``DS=1`` is one program per head."""
    pid = tl.program_id(0)
    h = pid // DS
    sd = pid % DS
    r = tl.arange(0, BM)
    rm = r < N
    dd = tl.arange(0, BD)
    hd = h * D + dd
    hm = rm[:, None] & (dd < D)[None, :]
    ss = sd * DSZ + tl.arange(0, BDS)
    hs = h * D + ss
    sm = rm[:, None] & (ss < D)[None, :]
    nch: tl.constexpr = N * CH
    qh = tl.load(QKVG + r[:, None] * CH + hd[None, :], mask=hm, other=0.0)
    kh = tl.load(QKVG + nch + r[:, None] * CH + hd[None, :], mask=hm, other=0.0)
    vh = tl.load(QKVG + 2 * nch + r[:, None] * CH + hs[None, :], mask=sm, other=0.0)
    gh = tl.load(QKVG + 3 * nch + r[:, None] * CH + hs[None, :], mask=sm, other=0.0)
    zb = tl.load(ZB + h * (N * N) + r[:, None] * N + r[None, :],
                 mask=rm[:, None] & rm[None, :], other=0.0)
    mk = tl.load(MK + r, mask=rm, other=0.0).to(tl.float32)
    sc = _rb(_rb(_rb(tl.dot(qh, tl.trans(kh))) + _rb(INF * _rb(mk - 1.0))[None, :])
             + zb.to(tl.float32))
    e = _expf(sc - tl.max(sc, 1)[:, None])
    pr = (e / tl.sum(e, 1)[:, None]).to(tl.bfloat16)
    tl.store(OG + r[:, None] * CH + hs[None, :],
             (_rb(tl.dot(pr, vh)) * gh.to(tl.float32)).to(tl.bfloat16), mask=sm)


@triton.jit
def _s_lo(A, OG, ST, A1, W,
          N: tl.constexpr, C: tl.constexpr, CH: tl.constexpr,
          BM: tl.constexpr, BK: tl.constexpr, NK: tl.constexpr,
          BN: tl.constexpr):
    """``linear_o``, the ``linear_ada_out`` gate and the block's first residual
    add, one program per output column strip. One K=no_heads*c_hidden fp32
    reduction, which is the GEMM the reference issues."""
    r = tl.arange(0, BM)
    rm = r < N
    o = tl.program_id(0) * BN + tl.arange(0, BN)
    om = o < C
    wb = W + tl.program_id(0) * (CH * BN)
    tj = tl.arange(0, BK)[:, None] * BN + tl.arange(0, BN)[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for ci in tl.range(0, NK):
        j = ci * BK + tl.arange(0, BK)
        jm = (j < CH)[:, None]
        g = tl.load(OG + r[:, None] * CH + j[None, :],
                    mask=rm[:, None] & jm.trans(), other=0.0)
        acc = tl.dot(g, tl.load(wb + ci * (BK * BN) + tj, mask=jm, other=0.0), acc)
    msk = rm[:, None] & om[None, :]
    ga = tl.load(ST + 4 * (N * C) + r[:, None] * C + o[None, :], mask=msk, other=0.0)
    upd = (_rb(acc) * ga.to(tl.float32)).to(tl.bfloat16)
    a0 = tl.load(A + r[:, None] * C + o[None, :], mask=msk, other=0.0).to(tl.float32)
    tl.store(A1 + r[:, None] * C + o[None, :],
             (a0 + upd.to(tl.float32)).to(tl.bfloat16), mask=msk)


@triton.jit
def _s_sw(A1, ST, B, W,
          N: tl.constexpr, C: tl.constexpr, NT: tl.constexpr,
          NSW: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
          NK: tl.constexpr, BN: tl.constexpr, EPS: tl.constexpr):
    """The transition's AdaLN and its SwiGLU hidden layer, one program per hidden
    column strip.

    A program takes the *same* column strip of ``linear_a`` and ``linear_b``,
    because the two have to meet in one register tile for ``SiLU(a) * b``;
    splitting them apart would cost a launch and a round trip through memory for
    no reduction in weight bytes."""
    pid = tl.program_id(0)
    r = tl.arange(0, BM)
    rm = r < N
    o = pid * BN + tl.arange(0, BN)
    om = o < NT
    mu, rstd = _mom(A1, r, rm, C, BM, BK, NK, EPS)
    wa = W + pid * (C * BN)
    wbb = W + (NSW + pid) * (C * BN)
    h1 = tl.zeros((BM, BN), tl.float32)
    h2 = tl.zeros((BM, BN), tl.float32)
    tj = tl.arange(0, BK)[:, None] * BN + tl.arange(0, BN)[None, :]
    for ci in tl.range(0, NK):
        j = ci * BK + tl.arange(0, BK)
        jm = (j < C)[:, None]
        xx = _adc(A1, ST + 2 * N * C, ST + 3 * N * C, r, j,
                  rm[:, None] & (j < C)[None, :], C, mu, rstd)
        h1 = tl.dot(xx, tl.load(wa + ci * (BK * BN) + tj, mask=jm, other=0.0), h1)
        h2 = tl.dot(xx, tl.load(wbb + ci * (BK * BN) + tj, mask=jm, other=0.0), h2)
    tl.store(B + r[:, None] * NT + o[None, :],
             (_silu(_rb(h1)).to(tl.float32) * _rb(h2)).to(tl.bfloat16),
             mask=rm[:, None] & om[None, :])


@triton.jit
def _s_out(A1, B, ST, MK, OUT, W,
           N: tl.constexpr, C: tl.constexpr, NT: tl.constexpr,
           BM: tl.constexpr, BK: tl.constexpr, NK: tl.constexpr,
           BN: tl.constexpr):
    """The transition's output projection, its gate, the ``_mask_trans`` mask and
    the block's second residual add, one program per output column strip. For the
    last block this is the kernel launched eagerly behind the graph, writing
    straight into the caller's output."""
    r = tl.arange(0, BM)
    rm = r < N
    o = tl.program_id(0) * BN + tl.arange(0, BN)
    om = o < C
    wb = W + tl.program_id(0) * (NT * BN)
    tj = tl.arange(0, BK)[:, None] * BN + tl.arange(0, BN)[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for ci in tl.range(0, NK):
        j = ci * BK + tl.arange(0, BK)
        jm = (j < NT)[:, None]
        b = tl.load(B + r[:, None] * NT + j[None, :],
                    mask=rm[:, None] & jm.trans(), other=0.0)
        w = tl.load(wb + ci * (BK * BN) + tj, mask=jm, other=0.0)
        acc = tl.dot(b, w, acc)
    msk = rm[:, None] & om[None, :]
    gt = tl.load(ST + 5 * (N * C) + r[:, None] * C + o[None, :], mask=msk, other=0.0)
    mq = tl.load(MK + r, mask=rm, other=0.0).to(tl.float32)
    u2 = (gt.to(tl.float32) * _rb(acc)).to(tl.bfloat16)
    u2 = (u2.to(tl.float32) * mq[:, None]).to(tl.bfloat16)
    a1 = tl.load(A1 + r[:, None] * C + o[None, :], mask=msk, other=0.0).to(tl.float32)
    tl.store(OUT + r[:, None] * C + o[None, :],
             (a1 + u2.to(tl.float32)).to(tl.bfloat16), mask=msk)


class _Call:
    """One kernel, launched through the compiled kernel's own C entry point.

    ``kernel[grid](...)`` re-binds, re-specializes and re-hashes every argument on
    each call (~9 us of Python); with every argument here either a raw pointer or
    a constexpr there is nothing for it to derive that is not already known.
    ``argv`` is the complete argument list, so a launch is one Python call with no
    tuple building, and the handful of entries that change per forward are patched
    in place by index.
    """

    __slots__ = ("run", "argv")

    def __init__(self, kern, grid, args, device):
        launcher = kern.run
        self.run = launcher.launch
        self.argv = [grid, 1, 1, _raw_stream(device), kern.function,
                     launcher.launch_cooperative_grid, launcher.launch_pdl,
                     None, None, kern.packed_metadata, None, None, None, *args]

    @staticmethod
    def usable(kern) -> bool:
        launcher = getattr(kern, "run", None)
        return (launcher is not None
                and getattr(launcher, "launch", None) is not None
                and getattr(launcher, "global_scratch_size", None) == 0
                and getattr(launcher, "profile_scratch_size", None) == 0)


_ARGV0 = 13  # index of the first kernel argument inside _Call.argv


def _tile(w, bn):
    """``[K, N]`` -> a flat ``[ceil(N/bn), K, bn]`` copy, zero-padded in N.

    Each program's ``[K, bn]`` weight slab becomes one contiguous run of memory,
    so the choice of column block stops interacting with the 128 B line size: a
    stride-N column strip of the row-major original reads ``bn*2`` bytes out of
    every line, which at bn=32 throws away half the bandwidth before the split
    factor is even chosen. K is left unpadded -- a consumer that wants a
    ``[next_pow2(K), bn]`` register tile masks the tail, which fetches nothing."""
    k, n = w.shape
    nb = -(-n // bn)
    out = w.new_zeros(k, nb * bn)
    out[:, :n] = w
    return out.reshape(k, nb, bn).permute(1, 0, 2).contiguous().reshape(-1)


def _sf_probe(mod, c, cs, pz):
    """Structural check of the self-attention stack. Returns
    ``(nblk, h, d, ch, nt, eps_a, eps_s, eps_z, inf)`` or None.

    Deliberately strict, like the cross path's ``_xf_probe``: anything the fused
    stack does not reproduce bit for bit -- a different AdaLN wiring, a missing
    gate, an offset on a weight-only norm, a non-uniform eps -- has to reach the
    real modules instead."""
    if getattr(mod, "use_cross_attention", True):
        return None
    blocks = getattr(mod, "blocks", None)
    if not blocks or len(blocks) < 1:
        return None
    ref = None
    for blk in blocks:
        pb = getattr(blk, "attention_pair_bias", None)
        ct = getattr(blk, "conditioned_transition", None)
        if pb is None or ct is None or not getattr(pb, "use_ada_layer_norm", False):
            return None
        if getattr(pb, "n_query", None) is not None:
            return None
        lnz = getattr(pb, "layer_norm_z", None)
        if (lnz is None or getattr(lnz, "weight", None) is None
                or getattr(lnz, "bias", None) is not None
                or tuple(lnz.normalized_shape) != (pz,)):
            return None
        mha = getattr(pb, "mha", None)
        if mha is None or mha.linear_g is None or not mha.gating:
            return None
        h, d = int(mha.no_heads), int(mha.c_hidden)
        ch = h * d
        if not (_lin_ok(mha.linear_q, (ch, c), True)
                and _lin_ok(mha.linear_k, (ch, c), False)
                and _lin_ok(mha.linear_v, (ch, c), False)
                and _lin_ok(mha.linear_o, (c, ch), False)
                and _lin_ok(mha.linear_g, (ch, c), False)
                and _lin_ok(pb.linear_z, (h, pz), False)
                and _lin_ok(pb.linear_ada_out, (c, cs), True)
                and _lin_ok(ct.linear_g, (c, cs), True)):
            return None
        sw = getattr(ct, "swiglu", None)
        if sw is None or getattr(sw, "linear_a", None) is None:
            return None
        nt = int(sw.linear_a.weight.shape[0])
        if not (_lin_ok(sw.linear_a, (nt, c), False)
                and _lin_ok(sw.linear_b, (nt, c), False)
                and _lin_ok(ct.linear_out, (c, nt), False)):
            return None
        eps_a = eps_s = None
        for ad in (pb.layer_norm_a, ct.layer_norm):
            la = getattr(ad, "layer_norm_a", None)
            ls = getattr(ad, "layer_norm_s", None)
            if (la is None or ls is None
                    or getattr(la, "weight", None) is not None
                    or getattr(la, "bias", None) is not None
                    or getattr(ls, "weight", None) is None
                    or getattr(ls, "bias", None) is not None
                    or tuple(la.normalized_shape) != (c,)
                    or tuple(ls.normalized_shape) != (cs,)
                    or not _lin_ok(ad.linear_g, (c, cs), True)
                    or not _lin_ok(ad.linear_s, (c, cs), False)):
                return None
            if eps_a is None:
                eps_a, eps_s = float(la.eps), float(ls.eps)
            elif (float(la.eps), float(ls.eps)) != (eps_a, eps_s):
                return None
        sig = (h, d, ch, nt, eps_a, eps_s, float(lnz.eps),
               float(getattr(pb, "inf", 0.0)))
        if ref is None:
            ref = sig
        elif sig != ref:
            return None
    h, d, ch, nt, eps_a, eps_s, eps_z, inf = ref
    if not (h >= 1 and d >= 8 and nt >= 16 and inf > 0.0):
        return None
    return len(blocks), h, d, ch, nt, eps_a, eps_s, eps_z, inf


class _SfPlan:
    """Compiled kernels, pre-tiled weights, staging arena, pre-built launch
    arguments and the captured graph for one (module, input-shape) pair.

    Built on the first supported forward rather than in ``__init__`` so
    externally loaded weights are what gets folded into the tiled copies, and
    invalidated by a state-dict load on the owning module."""

    def __init__(self, mod, a, s, z, mask):
        bf = torch.bfloat16
        dev = mask.device
        n, c = int(a.shape[-2]), int(a.shape[-1])
        cs, pz = int(s.shape[-1]), int(z.shape[-1])
        probe = _sf_probe(mod, c, cs, pz)
        if probe is None:
            raise _Unsupported
        nblk, h, d, ch, nt, eps_a, eps_s, eps_z, inf = probe
        bm = max(16, triton.next_power_of_2(n))
        if bm > 32 or z.numel() != n * n * pz or n <= 0:
            raise _Unsupported
        hp = max(16, triton.next_power_of_2(h))
        bs = max(16, triton.next_power_of_2(cs))
        bp = max(16, triton.next_power_of_2(pz))
        bd = max(16, triton.next_power_of_2(d))
        # ``_s_sn`` and ``_s_zb`` hold a whole row in registers, so their padded
        # width is what bounds the shapes this path will take on.
        if max(bs, bp) > 2048:
            raise _Unsupported
        self.n, self.c, self.cs, self.p = n, c, cs, pz
        self.nblk, self.dtype, self.device = nblk, bf, dev
        self.na, self.ns, self.nz, self.nm = n * c, n * cs, n * n * pz, n

        ncs = -(-c // _TAB_BN)
        nrs = -(-(n * n) // _ZB_BM)
        nch = -(-ch // _Q_BN)
        nlo = -(-c // _LO_BN)
        nsw = -(-nt // _SW_BN)
        nou = -(-c // _OT_BN)
        ds = _AT_DS if (_AT_DS > 0 and d % _AT_DS == 0) else 1
        dsz = d // ds
        bds = max(16, triton.next_power_of_2(dsz))
        nkq, nkl = -(-c // _Q_BK), -(-ch // _LO_BK)
        nks, nko = -(-c // _SW_BK), -(-nt // _OT_BK)
        ntab = 6 * nblk

        # --- packed weights (tiled once; kernels read contiguous tiles) ------
        ones = torch.ones(cs, device=dev, dtype=bf)
        zero = torch.zeros(c, device=dev, dtype=bf)
        gamt, wt, bt, gz, wz, packs = [], [], [], [], [], []
        for blk in mod.blocks:
            pb, ct = blk.attention_pair_bias, blk.conditioned_transition
            # Tables 0..3: (gate, shift) for the attention and transition AdaLNs.
            for ad in (pb.layer_norm_a, ct.layer_norm):
                gw = ad.layer_norm_s.weight.detach().contiguous().to(bf)
                gamt += [gw, gw]
                wt += [_t2(ad.linear_g.weight), _t2(ad.linear_s.weight)]
                bt += [_bias_of(ad.linear_g, c, dev), zero]
            # Tables 4, 5: the two output gates, over raw s (gamma unused).
            gamt += [ones, ones]
            wt += [_t2(pb.linear_ada_out.weight), _t2(ct.linear_g.weight)]
            bt += [_bias_of(pb.linear_ada_out, c, dev), _bias_of(ct.linear_g, c, dev)]
            gz.append(pb.layer_norm_z.weight.detach().contiguous().to(bf))
            wzb = torch.zeros(pz, hp, device=dev, dtype=bf)
            wzb[:, :h].copy_(_t2(pb.linear_z.weight))
            wz.append(wzb.reshape(-1))
            mha = pb.mha
            packs.append(torch.cat([
                torch.cat([_tile(_t2(w.weight), _Q_BN) for w in
                           (mha.linear_q, mha.linear_k, mha.linear_v, mha.linear_g)]),
                _bias_of(mha.linear_q, ch, dev),
                _tile(_t2(mha.linear_o.weight), _LO_BN),
                _tile(_t2(ct.swiglu.linear_a.weight), _SW_BN),
                _tile(_t2(ct.swiglu.linear_b.weight), _SW_BN),
                _tile(_t2(ct.linear_out.weight), _OT_BN),
            ]).contiguous())
        self.w = (torch.cat(gamt).contiguous(),
                  torch.cat([_tile(t, _TAB_BN) for t in wt]).contiguous(),
                  torch.cat(bt).contiguous(),
                  torch.cat(gz).contiguous(),
                  torch.cat(wz).contiguous(),
                  tuple(packs))
        o_bq = 4 * nch * c * _Q_BN
        o_wo = o_bq + ch
        o_ab = o_wo + nlo * ch * _LO_BN
        o_ot = o_ab + 2 * nsw * c * _SW_BN
        n_pk = o_ot + nou * nt * _OT_BN

        # --- staging + scratch arena (one allocation, held by the plan) ------
        o_sa = 0
        o_ss = o_sa + _align(n * c)
        o_sz = o_ss + _align(n * cs)
        o_sm = o_sz + _align(n * n * pz)
        o_sy = o_sm + _align(n)
        o_st = o_sy + _align(ntab * n * cs)
        st_i = 6 * n * c
        zb_i = hp * n * n
        o_zb = o_st + _align(nblk * st_i)
        o_qk = o_zb + _align(nblk * zb_i)
        o_og = o_qk + _align(4 * n * ch)
        o_a1 = o_og + _align(n * ch)
        o_bh = o_a1 + _align(n * c)
        size = o_bh + _align(n * nt)
        ar = torch.zeros(size, device=dev, dtype=bf)
        out0 = torch.zeros(n * c, device=dev, dtype=bf)
        self.arena, self.out0 = ar, out0

        def sl(t, nel, off):
            """A flat view of ``nel`` elements at ``off``. Triton infers the
            pointer type -- and its 16-byte alignment class, which the compiled
            kernel is specialized on -- from the object it is handed, so every
            pointer argument has to be a real tensor at compile time."""
            return t.as_strided((nel,), (1,), off)

        v_sa = sl(ar, n * c, o_sa)
        v_ss = sl(ar, n * cs, o_ss)
        v_sz = sl(ar, n * n * pz, o_sz)
        v_sm = sl(ar, n, o_sm)
        v_sy = sl(ar, ntab * n * cs, o_sy)
        v_zb = sl(ar, nblk * zb_i, o_zb)
        v_qk = sl(ar, 4 * n * ch, o_qk)
        v_og = sl(ar, n * ch, o_og)
        v_a1 = sl(ar, n * c, o_a1)
        v_bh = sl(ar, n * nt, o_bh)

        # --- launch plan ----------------------------------------------------
        nin = -(-max(self.na, self.ns, self.nz, self.nm) // _IN_BL)
        gam, twt, tbt, gzt, wzt = self.w[:5]
        K, G, A, NW, NS = [], [], [], [], []

        def add(kern, grid, args, warps=8, stages=1):
            K.append(kern)
            G.append(grid)
            A.append(args)
            NW.append(warps)
            NS.append(stages)

        add(_s_in, nin, [a, s, z, mask, v_sa, v_ss, v_sz, v_sm,
                         self.na, self.ns, self.nz, self.nm, _IN_BL], 4)
        add(_s_sn, ntab, [v_ss, v_sy, gam, n, cs, bm, bs, eps_s], 4)
        ntb = ntab * ncs
        add(_s_tz, ntb + nblk * nrs,
            [v_sy, sl(ar, nblk * st_i, o_st), twt, tbt,
             v_sz, v_zb, gzt, wzt,
             n, cs, c, ncs, bm, bs, _TAB_BN,
             ntb, n * n, pz, h, hp, nrs, _ZB_BM, bp, eps_z], _W_TAB)
        for i in range(nblk):
            pk = packs[i]
            vst = sl(ar, st_i, o_st + i * st_i)
            add(_s_qkvg, 4 * nch,
                [v_sa, vst, v_qk, pk, sl(pk, ch, o_bq), n, c, ch, nch, bm,
                 _Q_BK, nkq, _Q_BN, eps_a, float(math.sqrt(d))], _W_Q, _ST_Q)
            add(_s_at, h * ds,
                [v_qk, sl(ar, zb_i, o_zb + i * zb_i), v_sm, v_og,
                 n, ch, d, bm, bd, ds, dsz, bds, float(inf)], _W_AT_S)
            add(_s_lo, nlo,
                [v_sa, v_og, vst, v_a1, sl(pk, o_ab - o_wo, o_wo),
                 n, c, ch, bm, _LO_BK, nkl, _LO_BN], _W_LO, _ST_LO)
            add(_s_sw, nsw,
                [v_a1, vst, v_bh, sl(pk, o_ot - o_ab, o_ab),
                 n, c, nt, nsw, bm, _SW_BK, nks, _SW_BN, eps_a],
                _W_SW, _ST_SW)
            add(_s_out, nou,
                [v_a1, v_bh, vst, v_sm, v_sa if i + 1 < nblk else out0,
                 sl(pk, n_pk - o_ot, o_ot),
                 n, c, nt, bm, _OT_BK, nko, _OT_BN], _W_OT, _ST_OT)

        # Compile with the real tensors -- Triton infers pointer types from the
        # objects it is handed -- then bake the resulting addresses in. Every
        # block's kernels share their constexpr tail and pointer alignment class,
        # so this is one compile per distinct kernel, not one per block.
        compiled = [k[(g,)](*x, num_warps=w, num_stages=st)
                    for k, g, x, w, st in zip(K, G, A, NW, NS)]
        if not all(_Call.usable(k) for k in compiled):
            raise _Unsupported
        self.compiled = compiled          # keeps the CUfunction handles alive
        dix = mask.get_device()
        raw = [[t.data_ptr() if torch.is_tensor(t) else t for t in x] for x in A]
        self.calls = [_Call(k, g, x, dix) for k, g, x in zip(compiled, G, raw)]
        self.dix = dix
        self.stream = self.calls[0].argv[3]

        # Every slot holding a tensor the plan does not own has to be re-pointed
        # per call: the caller's a/s/z/mask in the staging kernel, its output in
        # the final projection. Found by identity, not written down by hand -- a
        # missed slot is a read of freed memory, silent and only when the caller
        # happens to pass a fresh tensor.
        def slots(t):
            return tuple((ci, _ARGV0 + ai)
                         for ci, x in enumerate(A) for ai, v in enumerate(x)
                         if v is t)

        self.in_slots = tuple(slots(t) for t in (a, s, z, mask))
        self.out_slots = slots(out0)
        # Capture needs constant node arguments: only the first node may touch
        # the caller's inputs and only the last its output.
        if not (all(self.in_slots) and self.out_slots
                and all(ci == 0 for sl_ in self.in_slots for ci, _ in sl_)
                and all(ci == len(A) - 1 for ci, _ in self.out_slots)):
            raise _Unsupported
        self.graph = None
        self._capture()

    def _capture(self):
        """Capture the fixed-pointer middle of the chain into a CUDA graph.

        The cost being attacked is dispatch. Measured back to back on this box, an
        eagerly launched kernel costs 2.76 us of device serialization and 3.1 us of
        host time; as a graph node, 1.01 us and ~0.03 us. At >100 dependent
        launches the host binds, so capture is the lever and PDL is not. Falls
        back to eager launches if capture fails."""
        inner = self.calls[1:-1]
        try:
            g = torch.cuda.CUDAGraph()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                st = _raw_stream(self.dix)
                for c in inner:
                    c.argv[3] = st
                for _ in range(2):
                    for c in inner:
                        c.run(*c.argv)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            with torch.cuda.graph(g):
                st = _raw_stream(self.dix)
                for c in inner:
                    c.argv[3] = st
                    c.run(*c.argv)
            torch.cuda.synchronize()
        except Exception:  # noqa: BLE001 - no capture: eager launches still work
            self.graph = None
        else:
            self.graph = g
        # The captured launchers must never be invoked directly again -- their
        # stream slot now names the (dead) capture stream.
        self.inner = inner

    def matches(self, a, s, z, mask) -> bool:
        # The alignment test is not optional: Triton specialized the compiled
        # kernels on the 16-byte alignment of the pointers they were built with,
        # so a misaligned buffer would silently miscompute.
        bf = self.dtype
        return (a.dtype is bf and s.dtype is bf and z.dtype is bf
                and mask.dtype is bf
                and a.device == self.device and a.shape[-1] == self.c
                and a.shape[-2] == self.n and a.numel() == self.na
                and s.shape[-1] == self.cs and s.numel() == self.ns
                and z.shape[-1] == self.p and z.numel() == self.nz
                and mask.numel() == self.nm
                and a.is_contiguous() and s.is_contiguous()
                and z.is_contiguous() and mask.is_contiguous()
                and not ((a.data_ptr() | s.data_ptr() | z.data_ptr()
                          | mask.data_ptr()) & 15))

    def run(self, a, s, z, mask, out):
        calls = self.calls
        for sl_, t in zip(self.in_slots, (a, s, z, mask)):
            ptr = t.data_ptr()
            for ci, ai in sl_:
                calls[ci].argv[ai] = ptr
        op = out.data_ptr()
        for ci, ai in self.out_slots:
            calls[ci].argv[ai] = op
        stream = _raw_stream(self.dix)
        g = self.graph
        if stream != self.stream:
            self.stream = stream
            for c in (calls if g is None else (calls[0], calls[-1])):
                c.argv[3] = stream
        if g is None:
            for c in calls:
                c.run(*c.argv)
        else:
            first, last = calls[0], calls[-1]
            first.run(*first.argv)
            g.replay()
            last.run(*last.argv)


# ###########################################################################
# Fused cross-attention stack (the 3-block, n_query=32/n_key=128 variant)
# ###########################################################################
# ~500 MMAC and 1.4 MiB of weights for the whole stack, so the reference's
# 5.4-5.8 ms is ~1500 kernel dispatches and nothing else. The frozen L2
# ``alphafold3_atom_attention`` winner already fuses this exact stack at these
# exact shapes (its ``atom_transformer`` *is* this class) into five kernels;
# these are that design, with its probe and its numerics helpers imported rather
# than re-derived, plus four changes measured here:
#
# * **PDL.** The five kernels are strictly dependent and every weight and table
#   load is independent of the previous kernel's output, so each ``gdc_wait()``
#   sits after that prologue and the consumer's loads issue while the producer
#   drains. Measured launch floor on this box (``dev/floor2.py``): 1.26 us with
#   PDL against 2.76 us eager. The frozen winner's launcher has no PDL binding,
#   which is why these are copies rather than calls. Capture is *not* the answer
#   here even at sixteen launches: the same chain measured 72.9 us captured
#   against 69.2 eager with PDL, because the overlap is worth more than the
#   0.25 us/node a graph saves. What that buys is room to spend launches on
#   parallelism -- 29 us of host time against 58 us of device leaves ~10 free.
# * **The attention is split off the projections.** The winner's ``_k_xa`` does
#   AdaLN(query rows) + AdaLN(key window) + four projections + attention in one
#   program per (block, head): six dependent stages behind 82 KiB of gathered
#   table rows, 9 us of work at 48 programs. Split into a row-wise
#   projection kernel (``_c_pq``, no gathers) and a pure attention kernel
#   (``_c_at``, which now gathers ``[*, c_hidden]`` slices instead of ``[*, c_a]``
#   table rows -- a quarter of the bytes), the pair costs less than the one did.
#   Both AdaLNs are row-wise functions of the block input, so hoisting them out of
#   the (block, head) grid also removes the 4x redundancy across heads and the
#   overlap between key windows.
# * **Every kernel is cut until it fills the machine.** With 148 SMs and an arena
#   that is entirely L2-resident, the only thing left to buy is programs. ``_c_at``
#   splits each query block's rows as well as its heads (``_AT_BQ=16``: 48 -> 96
#   programs, 6.16 -> 2.94 us of marginal cost) even though every piece re-reads
#   the same key window. The winner's ``_k_xb`` epilogue -- a seven-stage chain
#   holding 224 KiB of live weight tile, which capped it at 23 programs -- becomes
#   ``_c_lo`` (linear_o, gate, residual, transition AdaLN), ``_c_sw`` (the SwiGLU
#   pair, split by hidden column) and ``_c_ot`` (output projection, gate, mask,
#   residual, split by output column). Its three stage boundaries were already
#   bf16 values, so the split is bit-identical.
# * **One persistent arena and no per-call allocation.** The plan owns its
#   scratch, the last block's epilogue writes straight into the caller's output,
#   and launches go through pre-built argument lists with only the caller's
#   pointers patched.
#
# Everything block-invariant is still hoisted into a prologue: ``_csad_body`` emits
# all blocks' 8 per-row tables (three AdaLNs' gate and shift plus the two output
# gates) and ``_czb_body`` all blocks' pair bias as a single
# ``[c_z, no_blocks*no_heads]`` matmul after the stack's one ``layer_norm_z``.
# Neither needs anything from the other, so they share **one grid** (``_c_pr``):
# as two nodes on one stream PDL overlapped only the first's tail, and the first
# had idle SMs throughout. Worth 2.5 us of the chain's 67.

_PQ_BM, _PQ_BN = 16, 64   # _c_pq rows / output columns per program
_SAD_BM = 32              # _c_pr rows per program (its _csad_body half)
_AT_BQ = 16               # _c_at query rows per program (<= n_query)
_LO_BM = 16               # _c_lo rows per program
_SWX_BM, _SWX_BN = 8, 64    # _c_sw rows / hidden columns per program
_OTX_BM, _OTX_BN = 8, 32    # _c_ot rows / output columns per program
_W_SAD, _W_PQ, _W_AT = 4, 8, 4
_W_LOX, _W_SWX, _W_OTX = 4, 8, 8
_PDL = _HAS_PDL


@triton.jit
def _csad_body(pid, S, ST, GAMT, WT, BS,
              N: tl.constexpr, C: tl.constexpr, NSTRIP: tl.constexpr,
              BM: tl.constexpr, BC: tl.constexpr, EPS: tl.constexpr):
    """Every ``s``-dependent quantity of the whole stack: one program per
    (row strip, output table).

    Per block the tables are the query AdaLN's gate and additive shift, the key
    AdaLN's, the transition AdaLN's, and the two output gates
    (``linear_ada_out`` and the transition's ``linear_g``) -- ``8 * no_blocks``
    tables of ``[N, c_a]``, the ``ST`` layout the block kernels gather rows from.

    ``layer_norm_s`` is weight-only, so the normalized row is computed once per
    program and each table only re-scales it by its own instance's weight
    (rounded to bf16 exactly where the reference rounds) before its own
    ``Linear``. The two output gates take *raw* ``s``, which is the only thing
    that differs between tables, so the gamma, the weight and the bias are packed
    per table and the kernel is one dot with no branches.

    One dot per program rather than one program per strip: the whole stack is
    ~500 MMAC, so what sets its runtime is how many SMs it can fill -- at the
    captured shape that is 12 strips x 24 tables = 288 programs instead of 12."""
    gg = pid // NSTRIP
    rows = (pid % NSTRIP) * BM + tl.arange(0, BM)
    rm = rows < N
    n = tl.arange(0, BC)
    nm = n < C
    msk = rm[:, None] & nm[None, :]

    g = tl.load(GAMT + gg * C + n, mask=nm, other=0.0).to(tl.float32)
    w = tl.load(WT + gg * (C * C) + n[:, None] * C + n[None, :],
                mask=nm[:, None] & nm[None, :], other=0.0)
    b = tl.load(BS + gg * C + n, mask=nm, other=0.0).to(tl.float32)
    x = tl.load(S + rows[:, None] * C + n[None, :], mask=msk, other=0.0).to(tl.float32)
    # Tables 0..5 of each block consume layer_norm_s(s); tables 6 and 7 consume
    # raw s. Both are cheap, so select rather than branch.
    y = tl.where((gg % 8) < 6, (_lnf(x, n, C, EPS) * g[None, :]).to(tl.bfloat16),
                 x.to(tl.bfloat16))
    acc = _rb(tl.dot(y, w) + b[None, :])
    # The gates are sigmoid'd; the additive shifts (odd tables below 6) are not.
    out = tl.where(((gg % 8) >= 6) | ((gg % 2) == 0), _sig(acc), acc)
    tl.store(ST + gg * (N * C) + rows[:, None] * C + n[None, :],
             out.to(tl.bfloat16), mask=msk)


@triton.jit
def _czb_body(qi, PLM, ZB, GZ, WZ,
             NK: tl.constexpr, P: tl.constexpr, CZP: tl.constexpr,
             BP: tl.constexpr, EPS: tl.constexpr):
    """All blocks' attention pair biases, one program per query row.

    ``layer_norm_z`` belongs to the stack rather than to a block here, so it runs
    once, and each block's ``linear_z`` is ``c_z -> no_heads``, so all blocks'
    biases are a single ``[c_z, no_blocks*no_heads]`` matmul. Stored transposed so
    the attention kernel's read walks the key axis contiguously."""
    kk = tl.arange(0, NK)
    j = tl.arange(0, BP)
    o = tl.arange(0, CZP)
    jm = j < P
    g = tl.load(GZ + j, mask=jm, other=0.0).to(tl.float32)
    w = tl.load(WZ + j[:, None] * CZP + o[None, :], mask=jm[:, None], other=0.0)
    x = tl.load(PLM + (qi * NK + kk)[:, None] * P + j[None, :],
                mask=jm[None, :], other=0.0).to(tl.float32)
    y = (_lnf(x, j, P, EPS) * g[None, :]).to(tl.bfloat16)
    tl.store(ZB + qi * (CZP * NK) + o[:, None] * NK + kk[None, :],
             tl.trans(tl.dot(y, w).to(tl.bfloat16)))


@triton.jit
def _c_pr(S, ST, GAMT, WT, BS, PLM, ZB, GZ, WZ,
          N: tl.constexpr, C: tl.constexpr, NSTRIP: tl.constexpr,
          BM: tl.constexpr, BC: tl.constexpr, EPS: tl.constexpr,
          NSAD: tl.constexpr, NK: tl.constexpr, P: tl.constexpr,
          CZP: tl.constexpr, BP: tl.constexpr, EPSZ: tl.constexpr,
          PDL: tl.constexpr):
    """Both prologues on one grid.

    ``_csad_body`` (all blocks' AdaLN and gate tables, from ``s``) and ``_czb_body``
    (all blocks' pair bias, from ``z``) share no data in either direction, but on
    one stream they serialize -- PDL lets the second overlap only the first's
    tail, and ``_c_sad`` measured 11.2 us against ``_c_zb``'s 4.2 with idle SMs
    throughout both. On one grid of ``NSAD + nb*n_query`` programs they interleave
    completely and cost one launch instead of two. The branch is uniform within a
    CTA so the divergence is free; the price is that every program is allocated
    the larger of the two bodies' registers."""
    pid = tl.program_id(0)
    if PDL:
        gdc_wait()
    if pid < NSAD:
        _csad_body(pid, S, ST, GAMT, WT, BS, N, C, NSTRIP, BM, BC, EPS)
    else:
        _czb_body(pid - NSAD, PLM, ZB, GZ, WZ, NK, P, CZP, BP, EPSZ)
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _c_pq(A, ST, QKVG, W, BQ,
          N: tl.constexpr, C: tl.constexpr, CH: tl.constexpr,
          NCH: tl.constexpr, NRS: tl.constexpr, BM: tl.constexpr,
          BC: tl.constexpr, BN: tl.constexpr, EPS: tl.constexpr,
          RSQ: tl.constexpr, PDL: tl.constexpr):
    """The block's two AdaLNs and all four attention projections, one program per
    (projection, row strip, output column strip).

    ``q`` and ``g`` come from the query AdaLN, ``k`` and ``v`` from the key
    AdaLN; the two differ only in which pair of ``ST`` tables they read, so the
    table base is selected and the kernel is one dot with no branch. Both are
    row-wise functions of the block input, so computing them here -- over the
    *unblocked* rows -- rather than inside the (block, head) attention grid
    removes the 4x redundancy across heads and the redundancy between overlapping
    key windows.

    ``q`` is stored already scaled by ``1/sqrt(c_hidden)`` and ``g`` already
    through its sigmoid, both with the reference's roundings, so the attention
    kernel is pure attention."""
    pid = tl.program_id(0)
    task = pid // (NRS * NCH)
    rest = pid % (NRS * NCH)
    rows = (rest // NCH) * BM + tl.arange(0, BM)
    rm = rows < N
    n = tl.arange(0, BC)
    nm = n < C
    o = (rest % NCH) * BN + tl.arange(0, BN)
    om = o < CH
    nc: tl.constexpr = N * C
    w = tl.load(W + n[:, None] * (4 * CH) + (task * CH + o)[None, :],
                mask=nm[:, None] & om[None, :], other=0.0)
    bq = tl.load(BQ + o, mask=om, other=0.0).to(tl.float32)
    qside = (task == 0) | (task == 3)
    tb = ST + tl.where(qside, 0, 2) * nc
    if PDL:
        gdc_wait()
    msk = rm[:, None] & nm[None, :]
    x = tl.load(A + rows[:, None] * C + n[None, :], mask=msk, other=0.0).to(tl.float32)
    a1 = _adaln(x, tb, tb + nc, rows, n, msk, C, EPS)
    acc = tl.dot(a1, w)
    out = tl.where(task == 0, (_rb(acc + bq[None, :]) / RSQ).to(tl.bfloat16),
                   tl.where(task == 3, _sig(_rb(acc)).to(tl.bfloat16),
                            _rb(acc).to(tl.bfloat16)))
    tl.store(QKVG + task * (N * CH) + rows[:, None] * CH + o[None, :], out,
             mask=rm[:, None] & om[None, :])
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _c_at(QKVG, ZB, AMSK, OG,
          N: tl.constexpr, CH: tl.constexpr, NQ: tl.constexpr,
          NK: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
          BD: tl.constexpr, BN: tl.constexpr, CZP: tl.constexpr,
          BQ: tl.constexpr, QS: tl.constexpr,
          RSQ: tl.constexpr, INF: tl.constexpr, PDL: tl.constexpr):
    """Sequence-local attention for one (query block, head), up to the gated
    output.

    One program per (block, head) rather than per block: at the captured shape
    that is 48 programs instead of 12, and the kernel is latency-bound, not
    FLOP-bound. ``n_key <= 128`` fits in registers, so there is no online softmax
    and no ``[H, Q, K]`` score tensor in HBM.

    The key window is *computed*, not gathered: ``_get_block_key_indices`` runs
    several times per reference forward over the same mask, so ``_geom``
    re-derives it inline from ``atom_mask`` -- including reproducing the
    reference's accidental bf16 index arithmetic, which quantizes key indices
    above 256 to even numbers and makes blocks 6..11 gather duplicated atoms.
    Getting that "right" instead of identical would gather different atoms.

    Invalid keys read a zero mask, so their bias is ``-inf`` and they drop out of
    the softmax; that is why the key-side AdaLN can be computed over the
    *unblocked* rows in ``_c_pq`` even though the reference zeroes the gathered
    key rows before normalizing them -- the rows where the two differ have
    probability zero.

    Writes its ``c_hidden`` slice of the ``[N, no_heads*c_hidden]`` gated output
    so that ``linear_o`` can be the single K=no_heads*c_hidden dot the reference's
    GEMM does."""
    pid = tl.program_id(0)
    # (query strip, head, query block): the key window is a function of the
    # block alone, so splitting the block's rows costs nothing but a second
    # L2-resident read of the same k/v slices.
    qs = pid % QS
    h = (pid // QS) % H
    b = pid // (QS * H)
    dd = tl.arange(0, BD)
    dm = dd < D
    kk = tl.arange(0, NK)
    hd = h * D + dd
    nch: tl.constexpr = N * CH
    qi = b * NQ + qs * BQ + tl.arange(0, BQ)
    qok = qi < N
    nreal = _n_real(AMSK, N, BN)
    safe, invalid = _geom(nreal, b, NQ, NK)
    kok = (safe < N) & ~invalid
    mq = tl.load(AMSK + qi, mask=qok, other=0.0).to(tl.float32)
    mk = tl.load(AMSK + safe, mask=kok, other=0.0).to(tl.float32)
    bm = (mq[:, None] * mk[None, :]).to(tl.bfloat16).to(tl.float32)
    mbias = (INF * (bm - 1.0)).to(tl.bfloat16).to(tl.float32)
    zb = tl.load(ZB + qi[:, None] * (CZP * NK) + h * NK + kk[None, :],
                 mask=qok[:, None], other=0.0)
    if PDL:
        gdc_wait()
    qmsk = qok[:, None] & dm[None, :]
    kmsk = kok[:, None] & dm[None, :]
    q = tl.load(QKVG + qi[:, None] * CH + hd[None, :], mask=qmsk, other=0.0)
    kt = tl.load(QKVG + nch + safe[:, None] * CH + hd[None, :], mask=kmsk, other=0.0)
    vt = tl.load(QKVG + 2 * nch + safe[:, None] * CH + hd[None, :], mask=kmsk, other=0.0)
    gh = tl.load(QKVG + 3 * nch + qi[:, None] * CH + hd[None, :], mask=qmsk, other=0.0)
    sc = (tl.dot(q, tl.trans(kt)).to(tl.bfloat16).to(tl.float32) + mbias).to(tl.bfloat16)
    sc = (sc.to(tl.float32) + zb.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    e = _expf(sc - tl.max(sc, 1)[:, None])
    pr = (e / tl.sum(e, 1)[:, None]).to(tl.bfloat16)
    oh = tl.dot(pr, vt).to(tl.bfloat16).to(tl.float32)
    tl.store(OG + qi[:, None] * CH + hd[None, :],
             (oh * gh.to(tl.float32)).to(tl.bfloat16), mask=qmsk)
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _c_lo(A, OG, ST, W, A1, X2,
          N: tl.constexpr, C: tl.constexpr, CH: tl.constexpr,
          BM: tl.constexpr, BC: tl.constexpr, BCH: tl.constexpr,
          EPS: tl.constexpr, PDL: tl.constexpr):
    """First third of the block epilogue: ``linear_o``, the ``linear_ada_out``
    gate, the first residual add, then the transition AdaLN.

    Split out of ``_c_xb`` because that kernel was one program per row strip
    holding a seven-stage dependent chain and 224 KiB of live weight tile, which
    capped it at 46 programs and 8.1 us. Every stage boundary here is already a
    bf16 value in the fused version (``a1``, ``x2``, ``bb``), so routing them
    through the arena is **bit-identical** -- no rounding point moves.

    Writes both ``a1`` (the second residual add needs it, two kernels later) and
    ``x2`` (the SwiGLU input)."""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    rm = rows < N
    n = tl.arange(0, BC)
    nm = n < C
    ch = tl.arange(0, BCH)
    chm = ch < CH
    msk = rm[:, None] & nm[None, :]
    nc: tl.constexpr = N * C
    wo = tl.load(W + ch[:, None] * C + n[None, :],
                 mask=chm[:, None] & nm[None, :], other=0.0)
    xq = tl.load(A + rows[:, None] * C + n[None, :], mask=msk, other=0.0).to(tl.float32)
    ga = tl.load(ST + 6 * nc + rows[:, None] * C + n[None, :], mask=msk, other=0.0)
    gp = tl.load(ST + 4 * nc + rows[:, None] * C + n[None, :], mask=msk, other=0.0)
    sp = tl.load(ST + 5 * nc + rows[:, None] * C + n[None, :], mask=msk, other=0.0)
    if PDL:
        gdc_wait()
    og = tl.load(OG + rows[:, None] * CH + ch[None, :],
                 mask=rm[:, None] & chm[None, :], other=0.0)
    upd = (_rb(tl.dot(og, wo)) * ga.to(tl.float32)).to(tl.bfloat16)
    a1 = (xq + upd.to(tl.float32)).to(tl.bfloat16)
    tl.store(A1 + rows[:, None] * C + n[None, :], a1, mask=msk)
    y = _lnf(a1.to(tl.float32), n, C, EPS).to(tl.bfloat16)
    t = (y.to(tl.float32) + sp.to(tl.float32)).to(tl.bfloat16)
    tl.store(X2 + rows[:, None] * C + n[None, :],
             (gp.to(tl.float32) * t.to(tl.float32)).to(tl.bfloat16), mask=msk)
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _c_sw(X2, W, B,
          N: tl.constexpr, C: tl.constexpr, NT: tl.constexpr,
          NSW: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr,
          BN: tl.constexpr, PDL: tl.constexpr):
    """The SwiGLU hidden layer, one program per (row strip, hidden column strip).

    A program takes the *same* column strip of ``linear_a`` and ``linear_b``:
    the two have to meet in one register tile for ``SiLU(a) * b``. The K
    reduction is the whole of ``C`` in a single dot, exactly as in the fused
    kernel, so the result is bit-identical."""
    pid = tl.program_id(0)
    rows = (pid // NSW) * BM + tl.arange(0, BM)
    rm = rows < N
    n = tl.arange(0, BC)
    nm = n < C
    o = (pid % NSW) * BN + tl.arange(0, BN)
    om = o < NT
    o_wa: tl.constexpr = 0
    o_wb: tl.constexpr = C * NT
    wnt = nm[:, None] & om[None, :]
    wa = tl.load(W + o_wa + n[:, None] * NT + o[None, :], mask=wnt, other=0.0)
    wb = tl.load(W + o_wb + n[:, None] * NT + o[None, :], mask=wnt, other=0.0)
    if PDL:
        gdc_wait()
    x2 = tl.load(X2 + rows[:, None] * C + n[None, :],
                 mask=rm[:, None] & nm[None, :], other=0.0)
    h1 = _rb(tl.dot(x2, wa))
    h2 = tl.dot(x2, wb).to(tl.bfloat16)
    tl.store(B + rows[:, None] * NT + o[None, :],
             (_silu(h1).to(tl.float32) * h2.to(tl.float32)).to(tl.bfloat16),
             mask=rm[:, None] & om[None, :])
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _c_ot(A1, B, ST, AMSK, OUT, W,
          N: tl.constexpr, C: tl.constexpr, NT: tl.constexpr,
          NOU: tl.constexpr, BM: tl.constexpr, BNT: tl.constexpr,
          BN: tl.constexpr, PDL: tl.constexpr):
    """The transition output projection, its gate, the ``_mask_trans`` mask and
    the block's second residual add, one program per (row strip, output column
    strip). ``a1`` comes from ``_c_lo``, two kernels back, so it is read above
    the wait along with the weight and the gate table."""
    pid = tl.program_id(0)
    rows = (pid // NOU) * BM + tl.arange(0, BM)
    rm = rows < N
    tt = tl.arange(0, BNT)
    tm = tt < NT
    o = (pid % NOU) * BN + tl.arange(0, BN)
    om = o < C
    msk = rm[:, None] & om[None, :]
    nc: tl.constexpr = N * C
    wt = tl.load(W + tt[:, None] * C + o[None, :],
                 mask=tm[:, None] & om[None, :], other=0.0)
    gt = tl.load(ST + 7 * nc + rows[:, None] * C + o[None, :], mask=msk, other=0.0)
    mq = tl.load(AMSK + rows, mask=rm, other=0.0).to(tl.float32)
    a1 = tl.load(A1 + rows[:, None] * C + o[None, :], mask=msk, other=0.0).to(tl.float32)
    if PDL:
        gdc_wait()
    bb = tl.load(B + rows[:, None] * NT + tt[None, :],
                 mask=rm[:, None] & tm[None, :], other=0.0)
    o2 = tl.dot(bb, wt).to(tl.bfloat16)
    u2 = (gt.to(tl.float32) * o2.to(tl.float32)).to(tl.bfloat16)
    u2 = (u2.to(tl.float32) * mq[:, None]).to(tl.bfloat16)
    tl.store(OUT + rows[:, None] * C + o[None, :],
             (a1 + u2.to(tl.float32)).to(tl.bfloat16), mask=msk)
    if PDL:
        gdc_launch_dependents()


class _CxPlan:
    """Compiled kernels, packed weights, persistent arena and pre-built launch
    arguments for one (module, input-shape) pair of the cross-attention variant.

    The structural probe is the frozen winner's ``_xf_probe``: it already encodes
    exactly which wirings this stack reproduces bit for bit, and anything it
    declines reaches the reference instead."""

    def __init__(self, mod, a, s, z, mask):
        bf = torch.bfloat16
        dev = mask.device
        n, c = int(a.shape[-2]), int(a.shape[-1])
        pz = int(z.shape[-1])
        nq, nk = mod._n_query, mod._n_key
        if not (isinstance(nq, int) and isinstance(nk, int) and nq > 0 and nk > 0):
            raise _Unsupported
        nb = -(-n // nq)
        if z.numel() != nb * nq * nk * pz or n <= 0 or c <= 0:
            raise _Unsupported
        probe = _xf_probe(mod, c, pz, nq, nk)
        if probe is None:
            raise _Unsupported
        nblk, h, d, ch, nt, eps_a, eps_s, eps_z, inf = probe
        czp = max(16, triton.next_power_of_2(nblk * h))
        if czp > 256:
            raise _Unsupported
        bc = max(16, triton.next_power_of_2(c))
        bp = max(16, triton.next_power_of_2(pz))
        bd = max(16, triton.next_power_of_2(d))
        bn = triton.next_power_of_2(n)
        self.n, self.c, self.p, self.nq, self.nk = n, c, pz, nq, nk
        self.dtype, self.device = bf, dev
        self.na, self.nz, self.nm = n * c, z.numel(), n

        # --- packed weights (pre-transposed once; kernels read columns) ------
        ones = torch.ones(c, device=dev, dtype=bf)
        zero = torch.zeros(c, device=dev, dtype=bf)
        gamt, wt, bs, packs = [], [], [], []
        wz = torch.zeros(pz, czp, device=dev, dtype=bf)
        for i, blk in enumerate(mod.blocks):
            pb, ct = blk.attention_pair_bias, blk.conditioned_transition
            # Tables 0..5: (gate, shift) for the query, key and transition AdaLNs.
            for ad in (pb.layer_norm_a_q, pb.layer_norm_a_k, ct.layer_norm):
                gw = ad.layer_norm_s.weight.detach().contiguous().to(bf)
                gamt += [gw, gw]
                wt += [_t2(ad.linear_g.weight), _t2(ad.linear_s.weight)]
                bs += [_bias_of(ad.linear_g, c, dev), zero]
            # Tables 6, 7: the two output gates, over raw s (gamma unused).
            gamt += [ones, ones]
            wt += [_t2(pb.linear_ada_out.weight), _t2(ct.linear_g.weight)]
            bs += [_bias_of(pb.linear_ada_out, c, dev), _bias_of(ct.linear_g, c, dev)]
            wz[:, i * h:(i + 1) * h].copy_(_t2(pb.linear_z.weight))
            mha = pb.mha
            packs.append(torch.cat([t.reshape(-1) for t in (
                torch.cat([_t2(mha.linear_q.weight), _t2(mha.linear_k.weight),
                           _t2(mha.linear_v.weight), _t2(mha.linear_g.weight)], 1),
                _bias_of(mha.linear_q, ch, dev),
                _t2(mha.linear_o.weight), _t2(ct.swiglu.linear_a.weight),
                _t2(ct.swiglu.linear_b.weight), _t2(ct.linear_out.weight))]
            ).contiguous())
        self.w = (torch.cat(gamt).contiguous(),
                  torch.cat([t.reshape(-1) for t in wt]).contiguous(),
                  torch.cat(bs).contiguous(),
                  mod.layer_norm_z.weight.detach().contiguous().to(bf),
                  wz.contiguous(), tuple(packs))
        o_bq = 4 * c * ch
        o_wo = o_bq + ch

        # --- persistent arena (one allocation, held by the plan) -------------
        st_n = nblk * 8 * n * c
        o_st = 0
        o_zb = o_st + _align(st_n)
        o_qk = o_zb + _align(nb * nq * czp * nk)
        o_og = o_qk + _align(4 * n * ch)
        o_x0 = o_og + _align(n * ch)
        o_x1 = o_x0 + _align(n * c)
        # Stage boundaries of the split epilogue. Reused across blocks -- the
        # blocks are strictly sequential -- and only allocated when it is on.
        o_a1 = o_x1 + _align(n * c)
        o_x2 = o_a1 + _align(n * c)
        o_bh = o_x2 + _align(n * c)
        size = o_bh + _align(n * nt)
        ar = torch.zeros(size, device=dev, dtype=bf)
        self.arena = ar

        def sl(t, nel, off):
            return t.as_strided((nel,), (1,), off)

        v_st = sl(ar, st_n, o_st)
        v_zb = sl(ar, nb * nq * czp * nk, o_zb)
        v_qk = sl(ar, 4 * n * ch, o_qk)
        v_og = sl(ar, n * ch, o_og)
        bufs = (sl(ar, n * c, o_x0), sl(ar, n * c, o_x1))
        v_a1 = sl(ar, n * c, o_a1)
        v_x2 = sl(ar, n * c, o_x2)
        v_bh = sl(ar, n * nt, o_bh)
        out0 = torch.zeros(n * c, device=dev, dtype=bf)
        self.out0 = out0

        # --- launch plan ----------------------------------------------------
        gam, twt, tbs, gz, wzt = self.w[:5]
        nstrip = -(-n // _SAD_BM)
        nch = -(-ch // _PQ_BN)
        nrs = -(-n // _PQ_BM)
        nswx = -(-nt // _SWX_BN)
        noux = -(-c // _OTX_BN)
        bnt = max(16, triton.next_power_of_2(nt))
        bch = max(16, triton.next_power_of_2(ch))
        K, G, A, NW = [], [], [], []

        def add(kern, grid, args, warps=8):
            K.append(kern)
            G.append(grid)
            A.append(args)
            NW.append(warps)

        bq = _AT_BQ if (_AT_BQ > 0 and nq % _AT_BQ == 0) else nq
        qsplit = nq // bq
        add(_c_pr, nstrip * 8 * nblk + nb * nq,
            [s, v_st, gam, twt, tbs, z, v_zb, gz, wzt,
             n, c, nstrip, _SAD_BM, bc, eps_s,
             nstrip * 8 * nblk, nk, pz, czp, bp, eps_z, _PDL], _W_SAD)
        for i in range(nblk):
            pk = packs[i]
            src = a if i == 0 else bufs[(i - 1) & 1]
            dst = out0 if i + 1 == nblk else bufs[i & 1]
            add(_c_pq, 4 * nrs * nch,
                [src, sl(ar, st_n, o_st + i * 8 * n * c), v_qk, pk,
                 sl(pk, ch, o_bq), n, c, ch, nch, nrs, _PQ_BM, bc, _PQ_BN,
                 eps_a, float(math.sqrt(d)), _PDL], _W_PQ)
            add(_c_at, nb * h * qsplit,
                [v_qk, sl(ar, nb * nq * czp * nk, o_zb + i * h * nk), mask, v_og,
                 n, ch, nq, nk, h, d, bd, bn, czp, bq, qsplit,
                 float(math.sqrt(d)), float(inf), _PDL], _W_AT)
            vst = sl(ar, st_n, o_st + i * 8 * n * c)
            # ``o_wo`` starts linear_o; the two SwiGLU matrices and the output
            # projection follow it in the pack, so each of the three epilogue
            # kernels gets its own view and none sees the others' bytes.
            o_wa = o_wo + ch * c
            add(_c_lo, -(-n // _LO_BM),
                [src, v_og, vst, sl(pk, ch * c, o_wo), v_a1, v_x2,
                 n, c, ch, _LO_BM, bc, bch, eps_a, _PDL], _W_LOX)
            add(_c_sw, -(-n // _SWX_BM) * nswx,
                [v_x2, sl(pk, 2 * c * nt, o_wa), v_bh,
                 n, c, nt, nswx, _SWX_BM, bc, _SWX_BN, _PDL], _W_SWX)
            add(_c_ot, -(-n // _OTX_BM) * noux,
                [v_a1, v_bh, vst, mask, dst,
                 sl(pk, nt * c, o_wa + 2 * c * nt),
                 n, c, nt, noux, _OTX_BM, bnt, _OTX_BN, _PDL], _W_OTX)

        compiled = [k[(g,)](*x, num_warps=w, num_stages=1,
                            **({"launch_pdl": True} if _PDL else {}))
                    for k, g, x, w in zip(K, G, A, NW)]
        if not all(_Call.usable(k) for k in compiled):
            raise _Unsupported
        self.compiled = compiled
        dix = mask.get_device()
        raw = [[t.data_ptr() if torch.is_tensor(t) else t for t in x] for x in A]
        self.calls = [_Call(k, g, x, dix) for k, g, x in zip(compiled, G, raw)]
        self.dix = dix
        self.stream = self.calls[0].argv[3]

        def slots(t):
            return tuple((ci, _ARGV0 + ai)
                         for ci, x in enumerate(A) for ai, v in enumerate(x)
                         if v is t)

        self.slots = tuple(slots(t) for t in (a, s, z, mask, out0))
        if not all(self.slots):
            raise _Unsupported

    def matches(self, a, s, z, mask) -> bool:
        # The alignment test is not optional: Triton specialized the compiled
        # kernels on the 16-byte alignment of the pointers they were built with.
        bf = self.dtype
        return (a.dtype is bf and s.dtype is bf and z.dtype is bf
                and mask.dtype is bf and a.device == self.device
                and a.shape[-1] == self.c and a.shape[-2] == self.n
                and a.numel() == self.na and s.shape[-1] == self.c
                and s.shape[-2] == self.n and s.numel() == self.na
                and z.shape[-1] == self.p and z.numel() == self.nz
                and mask.numel() == self.nm
                and a.is_contiguous() and s.is_contiguous()
                and z.is_contiguous() and mask.is_contiguous()
                and not ((a.data_ptr() | s.data_ptr() | z.data_ptr()
                          | mask.data_ptr()) & 15))

    def run(self, a, s, z, mask, out):
        calls = self.calls
        for sl_, t in zip(self.slots, (a, s, z, mask, out)):
            ptr = t.data_ptr()
            for ci, ai in sl_:
                calls[ci].argv[ai] = ptr
        stream = _raw_stream(self.dix)
        if stream != self.stream:
            self.stream = stream
            for c in calls:
                c.argv[3] = stream
        for c in calls:
            c.run(*c.argv)
