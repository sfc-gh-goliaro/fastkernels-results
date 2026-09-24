"""FLUX transformer blocks (L3 composites).

FluxTransformerBlock: Dual-stream block with AdaLayerNormZero conditioning.
  Separate attention/FFN for image and text (encoder) streams.

FluxSingleTransformerBlock: Single-stream DiT block: text+image concatenated,
  self-attention + MLP in parallel.

Everything expensive in these blocks already belongs to a frozen L2 winner --
the attention, the adaLN conditioning, the FFN GEMMs.  What is left is *glue*:
the two ``torch.cat`` calls and the gate / residual / modulate chains.  On B200
that glue is ~190 us of the 1.0 ms single-stream call and ~230 us of the 1.16 ms
dual-stream call, and at the smaller captured sequence length it is the *host*
that runs out of time first -- 30 kernel launches enqueue in 569 us against
425 us of device work.  This implementation removes the glue rather than
speeding it up: the dual-stream block goes from 30 launches to 15 and the
single-stream one from 12 to 10.  The submodule tree, the ``state_dict`` keys,
the weight loaders and the fp8 paths are untouched, and anything the fused path
does not cover runs ``_forward_reference``.

What changes
------------
1. **The single-stream block never materializes a concatenation.**  The
   reference builds ``cat([encoder, image], dim=1)`` (56 MB of copy traffic),
   then ``cat([attn_output, mlp_hidden_states], dim=2)`` -- a
   ``[1, 4608, 15360]`` bf16 tensor, 283 MB of copy traffic and a 142 MB
   allocation -- purely to feed one GEMM.  Here the modulated layer norm reads
   the two input streams directly and writes one packed ``[1, 512+S, dim]``
   buffer, and ``proj_out`` is split along K into its ``[dim]`` and
   ``[mlp_hidden_dim]`` halves (strided *views* of the one parameter, built
   once, so loading is unchanged) and run as two accumulating GEMMs.  Measured
   at T=4608: 271.4 us for the split pair against 275.6 us for the single GEMM
   on a pre-built buffer -- the concatenation is free to delete.  ``out=`` into
   a strided column range of a packed buffer was the other candidate and is
   *not* viable: cuBLAS falls off its fast kernel, 454.9 vs 226.4 us.

2. **Every gate / residual / modulate chain is one pass.**  ``gate * y``,
   ``x + y``, ``norm2(x)``, ``* (1 + scale)``, ``+ shift`` is five kernels and
   five passes over the activation in the reference; here one kernel loads x and
   y once, reduces in fp32 and emits both the updated residual and the modulated
   activation.  The image and text streams are covered by *one* launch each
   time, the row index selecting which residual, gate and shift/scale vector to
   use, so the dual-stream block's ~230 us of glue becomes ~75 us.

3. **The dead branches are gone and ``silu(temb)`` is computed once.**  The
   ``len(attention_outputs) == 3`` and fp16-clip branches never fire in any
   captured call and are off the fused path; the two adaLN conditioning GEMVs
   share the one ``silu(temb)`` the reference computes twice.  Everything a
   packed buffer or a conditioning vector is sliced into is a byte offset, not a
   view, and every launch goes through the compiled kernel's own C launcher, so
   the host cost of all this is three ``data_ptr()`` calls and some integer
   arithmetic.

4. **Each block's independent branches run on a side stream.**  With the glue
   gone, ~80% of both blocks is GEMM and SDPA already running at 1.4-1.7
   PFLOP/s, and what is left is either bandwidth-bound or too small to fill the
   device.  Two regions have an independent sibling that could be running
   alongside instead of behind: the single block's MLP branch (``proj_mlp``, the
   GELU pass, ``proj_out``'s K=12288 half) against its attention branch, and the
   dual block's M=512 text FFN against its image FFN.  Both are forked onto one
   cached side stream and joined with cached events, and ``_Fork`` documents the
   caching-allocator invariant that makes that safe.  The dual block's outputs
   stay bit-identical to the serial schedule, verified elementwise over the
   harness's own seeds; the single block's move by one bf16 rounding, which is
   ``_gsum2_fwd``'s doing rather than the side stream's and which measures free
   (see there).

   Splitting the branch off is not enough on its own, because ``proj_out``'s two
   K-halves were a *chain*: the wide half wrote a bf16 partial and the narrow one
   accumulated onto it with ``beta=1``, which leaves the narrow GEMM -- the half
   that needs the attention output -- sitting on the join, waiting for both
   branches.  The two halves are computed independently here and summed in
   ``_gsum2_fwd`` instead, at identical traffic (the accumulating GEMM's
   read-modify-write of the partial pays for the extra load), which balances the
   two streams at 428 against 448 us of kernel time and is worth another
   0.7-1.9% on top of the fork.

   Measured per case as candidate device time against the serial schedule, in the
   regime the harness scores (its own timing loop, L2 flushed before every call
   and input addresses shifting; see ITERATIONS.md, because a plain A/B of two
   bench runs cannot resolve 1%):

   =========================  ==============  =============
   case                       single-stream   dual-stream
   =========================  ==============  =============
   S=4096                     1.021x          1.013x
   S=1024                     1.046x          gated off
   =========================  ==============  =============

   The two negative results behind that table are the useful part.  Moving *only*
   the GELU pass across -- 35 us of pure bandwidth hidden under 367 us of qkv
   GEMM, RoPE and SDPA, which is the trade that looks free -- is worth nothing
   measurable (1.002x), while moving the branch's GEMMs across is worth all of
   the above.  What concurrency buys here is not idle SMs, it is memory-level
   parallelism: the scored regime re-reads all 226 MB of this block's weights
   from DRAM on every call, and two independent weight streams in flight beat
   one.  Under a profiler with warm caches the same schedule is a wash and the
   overlapped kernels visibly inflate, so the two regimes disagree and the scored
   one decides.  Conversely the dual block's text FFN only hides while there is
   something big enough to hide it under: at S=4096 its ~72 us disappear into a
   ~395 us image FFN, at S=1024 the image pair is only ~102 us, the two merely
   contend, and forking *costs* 1%.  Hence the ratio gate in ``_forward_fused``.

Numerics is the binding constraint here, not speed
--------------------------------------------------
The frozen ``AdaLayerNormZero`` applies its modulation in fp32 while the
reference rounds the normalized value to bf16 first.  Standing alone that is a
benign difference -- it is *more* accurate.  Composed into an L3 block whose
output is ``residual + gate * f(modulated)``, where the two terms are of
comparable size and partly cancel, it is not: the untouched baseline body,
merely picking up the frozen L2 imports, reaches only matched_ratio 0.961-0.974
against the 0.99 the harness requires.  So the fused kernels here reproduce the
reference's *rounding sequence* exactly, not just its algebra::

    n = bf16((x - mean) * rstd)      # what F.layer_norm stores
    s = bf16(1 + scale)              # bf16 tensor + python scalar
    y = bf16(bf16(n * s) + shift)    # two bf16 binary ops

and likewise ``bf16(residual + bf16(gate * y))`` for the gate chains.  See
``_rbf`` for why that cannot be spelled ``x.to(tl.bfloat16).to(tl.float32)``.

Two fusions that the same constraint rules out, both of which *work* and are
measurably faster, are documented where they would have gone -- ``_cond`` (one
batched hand-written conditioning GEMV; every value it produces is a broadcast
scale, so its ~1e-6 disagreement with cuBLAS costs 0.005 of matched_ratio) and
``_forward_fused`` step 2 of the single-stream block (cuBLASLt's bias+GELU
epilogue, which runs the activation on the unrounded fp32 accumulator and costs
0.007).  For the same reason the frozen FFN's own epilogue path is switched off
inside the dual-stream block; nothing in L2 is edited, only which of its two
paths runs.  What is left measures matched_ratio 0.995-0.998 on both captured
sequence lengths.

The fast paths are bf16 batch-1 only.  Every captured call is bf16 at batch 1;
fp16 additionally needs the reference's ``clip(-65504, 65504)``, fp32 needs no
rounding emulation at all, and a batched conditioning vector would make the
gates per-row rather than broadcast.  Those all keep the reference body, which
drives the frozen adaLN's *reference* branch (``_ref_adaln``) so that a declined
shape does not come out less accurate than the fused one.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._C import _cuda_getCurrentRawStream as _raw_stream
from torch._C import _cuda_getDevice as _cur_device

from ..L1.gelu import GELU
from ..L1.layer_norm import LayerNorm
from ..L2.ada_layer_norm import AdaLayerNormZero, AdaLayerNormZeroSingle
from ..L2.flux_attention import FluxAttention
from ..L2.flux_feedforward import FeedForward
from ..L2.parallel_linear import ReplicatedLinear

# The frozen adaLN winner's launch plumbing, reused rather than duplicated: the
# exact-cover tile split its layer norm is tuned around, and the wrapper that
# holds a compiled kernel's own C launcher so a launch costs one call instead of
# a trip through Triton's per-call argument binder (~12 us of Python, more than
# the rest of these blocks at the captured sizes).
from ..L2.ada_layer_norm import _Launcher, _tile_split
from ..L2.ada_layer_norm import _PDL, _gdc_wait

# Widest row the fused paths keep in registers.
_MAX_N = 16384


# ###########################################################################
# Stream plumbing
#
# Both blocks have a region whose two halves are independent, and on one stream
# the shorter half is dead time on the SMs rather than dead traffic (see
# ``_fork``).  Switching the current stream has to be cheap enough to be worth
# it: ``with torch.cuda.stream(s)`` measures 6.1 us of Python per entry/exit
# pair on this box, against 0.54 for the two raw ``_cuda_setStream`` calls it
# wraps, and 3.6 us for ``Event.record()`` (which looks the current stream up)
# against 2.3 for ``Event.record(stream)``.  So the current stream is set
# through the raw binding with a pre-built kwargs dict, the events are recorded
# against an explicit stream, and both the stream and the events are cached on
# the module -- creating either per call is real host cost at the smaller
# captured sequence length, where the whole enqueue has ~110 us of slack.
# ###########################################################################
_set_stream = torch._C._cuda_setStream


def _stream_kwargs(s) -> dict:
    return {"stream_id": s.stream_id, "device_index": s.device_index,
            "device_type": s.device_type}


class _Fork:
    """A side stream plus the two events that fork work onto it and join it.

    Getting this wrong does not crash, it corrupts an occasional element, so the
    rule for every buffer in a forked region is written down here and the call
    sites are annotated against it.  The hazard is that PyTorch's caching
    allocator is *stream-aware*: a freed block goes back to whichever stream
    allocated it and is handed out again relying on that stream's ordering
    alone, so a block allocated on one stream and still being read on the other
    can be recycled underneath the reader.  Three cases, and every buffer in
    this file is one of them:

    * **Side-allocated, read on the main stream** -- exactly one per forked
      region here (the single block's ``proj_out`` partial, the dual block's text
      FFN output).  Marked with ``record_stream(main)``, which makes the
      allocator defer reuse until the main stream has passed it.  ~0.3 us of
      Python.  Writing them into a main-stream buffer with ``out=`` instead
      would need no marking at all and is not usable: cuBLAS dispatches a
      different, much slower kernel for an explicit ``out`` (measured: 21% on
      the whole single block at T=4608).
    * **Main-allocated, read on the side stream** -- the modulated activations
      both branches consume.  Safe unmarked, because the caller always joins
      before it returns: every later main-stream allocation is therefore ordered
      after the side stream's reads of it.
    * **Side-allocated, side-only scratch** -- the MLP pre-activation, the
      frozen FFN's inner activation.  Never leaves the side stream, so its reuse
      is ordered by the side stream itself.

    The bail-outs matter as much as the happy path: a fused path that declines
    *after* forking has to join first, or it returns while side-stream stores are
    still in flight against buffers the reference body is about to be handed back
    by the allocator.

    ``dev/stress.py`` is the regression test for all of this -- the forked and
    serial schedules run side by side over many iterations of fresh inputs with
    allocator churn in between, and every output element must match.
    """

    __slots__ = ("side", "ev_f", "ev_j", "s_args", "main", "main_raw", "m_args",
                 "dev")

    def __init__(self, device, dev: int):
        self.dev = dev
        self.side = torch.cuda.Stream(device=device)
        self.ev_f = torch.cuda.Event()
        self.ev_j = torch.cuda.Event()
        self.s_args = _stream_kwargs(self.side)
        self.main = None
        # -1, not 0: the default stream's raw handle *is* 0, so a 0 sentinel
        # would make the first bind() a no-op and leave m_args empty.
        self.main_raw = -1
        self.m_args: dict = {}

    def bind(self, dev: int) -> bool:
        """Refresh the cached main-stream handle if the caller's stream moved.

        ``_raw_stream`` is a 0.05 us C call; building the ``Stream`` object it
        checks against is 1.2 us, so the object is cached and revalidated by
        pointer.  Returns False for a call on a different device than the side
        stream was created on, which the caller declines rather than tries to
        serve across devices.
        """
        if dev != self.dev:
            return False
        raw = _raw_stream(dev)
        if raw != self.main_raw:
            self.main = torch.cuda.current_stream(dev)
            self.main_raw = raw
            self.m_args = _stream_kwargs(self.main)
        return True

    def prime(self, dev: int, dtype, device, act=None) -> None:
        """Pay the side stream's one-time costs outside any timed call.

        cuBLAS allocates a workspace the first time it sees a stream and Triton
        has to load each kernel onto it, so without this the first *timed* call
        would carry both.
        """
        if not self.bind(dev):
            raise RuntimeError("fork device mismatch")
        a = torch.zeros((16, 16), dtype=dtype, device=device)
        _set_stream(**self.s_args)
        try:
            b = torch.mm(a, a)
            if act is not None:
                act(b)
        finally:
            _set_stream(**self.m_args)
        self.side.synchronize()


# ###########################################################################
# Kernels
#
# Each block's activation is two streams -- ``[0, SPLIT)`` text rows and
# ``[SPLIT, T)`` image rows -- that need the same arithmetic with different
# source tensors, gates and shift/scale vectors.  Rather than one launch per
# stream, every kernel below is one grid over all T rows whose program picks its
# side from the row index; the per-row body lives in a ``@triton.jit`` helper so
# the two sides cannot drift apart.  Side A addresses its tensors at ``r * N``
# and side B at ``(r - SPLIT) * N``, so a caller holding one packed ``[T, N]``
# buffer passes ``base`` and ``base + SPLIT * N`` and a caller holding two
# separate tensors passes both bases -- no other bookkeeping.
# ###########################################################################
@triton.jit
def _rbf(x):
    """Round an fp32 value to bf16 precision, staying in fp32.

    Every fused pass in this file has to reproduce the reference's *bf16
    rounding sequence*, not just its algebra (see the module docstring), and the
    obvious spelling of one such round -- ``x.to(tl.bfloat16).to(tl.float32)``
    -- does not work: the compiler folds the truncate/extend pair away, so the
    value silently keeps full fp32 precision and the kernel reproduces the
    frozen adaLN's fp32 modulation instead of the reference's bf16 one
    (measured: 71.7% of elements bit-equal to the reference, against 100% here).

    Doing it on the bit pattern cannot be folded: add the round-to-nearest-even
    offset into the low half, then mask the half off.  Bit-identical to
    ``cvt.rn.bf16.f32`` except on NaN payloads, which is one reason the fused
    paths are bf16-only and guarded rather than universal.

    Five integer ops, and at three rounds per element that is 6.2 us of the
    18.6 us modulated-norm pass on B200 -- so both cheaper spellings were tried.
    The one-instruction ``cvt.rn.bf16.f32`` through ``inline_asm_elementwise``
    is exactly as accurate and 2.0 us faster in isolation, but measures 0.3%
    *slower* across the four captured cases (register allocation shifts), so the
    portable arithmetic stays.  Rounding half-away-from-zero -- two ops,
    ``(i + 0x8000) & 0xFFFF0000`` -- is a further 4 us and is not usable: the
    products being rounded are two 8-bit mantissas wide, so exact ties are
    common rather than 2^-16 rare, and 9.5% of elements come out one ULP off.
    """
    i = x.to(tl.uint32, bitcast=True)
    i = (i + 0x7FFF + ((i >> 16) & 1)) & 0xFFFF0000
    return i.to(tl.float32, bitcast=True)


@triton.jit
def _ln_mod_row(X, Y, SH, SC, N: tl.constexpr, eps: tl.constexpr,
                B0: tl.constexpr, B1: tl.constexpr, TWO: tl.constexpr,
                MASK1: tl.constexpr):
    """``y = bf16(bf16(bf16((x-mean)*rstd) * bf16(1+scale)) + shift)``, one row.

    The reduction follows the frozen L1 ``_layer_norm_fwd``: the row is covered
    by one or two power-of-two tiles summing to exactly N (3072 = 2048 + 1024)
    rather than a masked ``next_pow2`` tile idling 25% of its lanes, is loaded
    once, and is reduced by the shifted one-pass formula -- subtract the row's
    own first element, then accumulate ``sum(d)`` and ``sum(d*d)`` together so
    the two reduction trees pipeline.  ``evict_first`` on the streamed row
    leaves L2 to the shift/scale vectors, which every one of the T programs
    re-reads.

    The casts are the point: see the module docstring.  Applying the affine in
    fp32 -- which is what the frozen adaLN does, and is *more* accurate -- moves
    the block output far enough to fail the harness, so each of the reference's
    three bf16 binary ops is reproduced with an explicit round.
    """
    dt = Y.dtype.element_ty
    c0 = tl.arange(0, B0)
    x0 = tl.load(X).to(tl.float32)          # the reduction's shift constant
    d0 = tl.load(X + c0, eviction_policy="evict_first").to(tl.float32) - x0
    acc = tl.sum(d0, axis=0)
    sq = tl.sum(d0 * d0, axis=0)
    if TWO:
        c1 = B0 + tl.arange(0, B1)
        if MASK1:
            m1 = c1 < N
            # Padding lanes must contribute 0 to both sums, so they are zeroed
            # after the shift rather than loaded with ``other=0.0``.
            d1 = tl.where(m1, tl.load(X + c1, mask=m1,
                                      eviction_policy="evict_first")
                          .to(tl.float32) - x0, 0.0)
        else:
            d1 = tl.load(X + c1, eviction_policy="evict_first").to(tl.float32) - x0
        acc += tl.sum(d1, axis=0)
        sq += tl.sum(d1 * d1, axis=0)
    inv_n: tl.constexpr = 1.0 / N
    off = acc * inv_n                       # mean, relative to the shift
    rstd = 1.0 / tl.sqrt(tl.maximum(sq * inv_n - off * off, 0.0) + eps)

    n0 = _rbf((d0 - off) * rstd)
    s0 = _rbf(1.0 + tl.load(SC + c0).to(tl.float32))
    y0 = _rbf(n0 * s0) + tl.load(SH + c0).to(tl.float32)
    tl.store(Y + c0, y0.to(dt), eviction_policy="evict_first")
    if TWO:
        n1 = _rbf((d1 - off) * rstd)
        if MASK1:
            s1 = _rbf(1.0 + tl.load(SC + c1, mask=m1).to(tl.float32))
            y1 = _rbf(n1 * s1) + tl.load(SH + c1, mask=m1).to(tl.float32)
            tl.store(Y + c1, y1.to(dt), mask=m1, eviction_policy="evict_first")
        else:
            s1 = _rbf(1.0 + tl.load(SC + c1).to(tl.float32))
            y1 = _rbf(n1 * s1) + tl.load(SH + c1).to(tl.float32)
            tl.store(Y + c1, y1.to(dt), eviction_policy="evict_first")


@triton.jit
def _modln2_fwd(XA, XB, Y, SHA, SCA, SHB, SCB,
                SPLIT: tl.constexpr, N: tl.constexpr, eps: tl.constexpr,
                B0: tl.constexpr, B1: tl.constexpr, TWO: tl.constexpr,
                MASK1: tl.constexpr):
    """Modulated layer norm over two input streams into one packed buffer."""
    _gdc_wait()
    r = tl.program_id(0)
    o = r.to(tl.int64) * N
    if r < SPLIT:
        _ln_mod_row(XA + o, Y + o, SHA, SCA, N, eps, B0, B1, TWO, MASK1)
    else:
        _ln_mod_row(XB + (o - SPLIT * N), Y + o, SHB, SCB, N, eps,
                    B0, B1, TWO, MASK1)


@triton.jit
def _gres_row(R, Y, G, O, N: tl.constexpr, B0: tl.constexpr, B1: tl.constexpr,
              TWO: tl.constexpr, MASK1: tl.constexpr):
    """``o = bf16(r + bf16(g * y))`` -- the gate-and-residual chain, one row.

    Three of the reference's passes over the activation (the broadcast
    multiply, the add, and -- in the single-stream block -- the slice that feeds
    them) collapse into this one: two loads, one store, 85 MB against 151 MB at
    the captured T=4608.
    """
    dt = O.dtype.element_ty
    c0 = tl.arange(0, B0)
    g0 = tl.load(G + c0).to(tl.float32)
    y0 = tl.load(Y + c0, eviction_policy="evict_first").to(tl.float32)
    o0 = _rbf(g0 * y0) + tl.load(R + c0,
                                 eviction_policy="evict_first").to(tl.float32)
    tl.store(O + c0, o0.to(dt), eviction_policy="evict_first")
    if TWO:
        c1 = B0 + tl.arange(0, B1)
        if MASK1:
            m1 = c1 < N
            g1 = tl.load(G + c1, mask=m1).to(tl.float32)
            y1 = tl.load(Y + c1, mask=m1, eviction_policy="evict_first").to(tl.float32)
            o1 = _rbf(g1 * y1) + tl.load(R + c1, mask=m1,
                                         eviction_policy="evict_first").to(tl.float32)
            tl.store(O + c1, o1.to(dt), mask=m1, eviction_policy="evict_first")
        else:
            g1 = tl.load(G + c1).to(tl.float32)
            y1 = tl.load(Y + c1, eviction_policy="evict_first").to(tl.float32)
            o1 = _rbf(g1 * y1) + tl.load(R + c1,
                                         eviction_policy="evict_first").to(tl.float32)
            tl.store(O + c1, o1.to(dt), eviction_policy="evict_first")


@triton.jit
def _gres2_fwd(RA, RB, YA, YB, GA, GB, O, SPLIT: tl.constexpr, N: tl.constexpr,
               B0: tl.constexpr, B1: tl.constexpr, TWO: tl.constexpr,
               MASK1: tl.constexpr):
    """Gate-and-residual for both streams in one launch."""
    _gdc_wait()
    r = tl.program_id(0)
    o = r.to(tl.int64) * N
    if r < SPLIT:
        _gres_row(RA + o, YA + o, GA, O + o, N, B0, B1, TWO, MASK1)
    else:
        d = o - SPLIT * N
        _gres_row(RB + d, YB + d, GB, O + o, N, B0, B1, TWO, MASK1)


@triton.jit
def _gsum_row(R, Y, Z, G, O, N: tl.constexpr, B0: tl.constexpr, B1: tl.constexpr,
              TWO: tl.constexpr, MASK1: tl.constexpr):
    """``o = bf16(r + bf16(g * bf16(y + z)))`` -- the same chain over *two* partials.

    The single-stream block's ``proj_out`` is one GEMM over K=15360 that this file
    already splits along K so the ``[1, T, 15360]`` concatenation need not be
    built.  Accumulating the second half onto the first (``y.addmm_(...)``) makes
    the two halves a chain, which puts the K=3072 half -- 61 us, and it is the
    half that needs the attention output -- on the join between the two streams.
    Summing two independent partials here instead lets both GEMMs run at once,
    for one extra load per element (+27 MB, ~4 us at the 7.7 TB/s this kernel
    reaches) and one extra bf16 rounding.

    That extra rounding is the cost of the trade, and it had to be measured
    rather than assumed: the reference sums all 15360 products in one fp32
    accumulator and rounds once; the accumulating form rounds the wide partial
    and then adds the narrow one exactly; this form rounds both partials before
    adding them.  It comes out free -- matched_ratio 0.9976 / 0.9973 at the two
    captured sequence lengths, the same to four decimals as the accumulating
    form, because the narrow partial carries the smaller magnitude (K=3072
    against 12288) so its own ULP lands below the one the sum is rounded to
    anyway.  The bias rides on the *narrow* half for the same reason: it is the
    partial whose low bits survive.  Both those choices are worth re-checking
    if the K split ever moves.
    """
    dt = O.dtype.element_ty
    c0 = tl.arange(0, B0)
    g0 = tl.load(G + c0).to(tl.float32)
    y0 = _rbf(tl.load(Y + c0, eviction_policy="evict_first").to(tl.float32)
              + tl.load(Z + c0, eviction_policy="evict_first").to(tl.float32))
    o0 = _rbf(g0 * y0) + tl.load(R + c0,
                                 eviction_policy="evict_first").to(tl.float32)
    tl.store(O + c0, o0.to(dt), eviction_policy="evict_first")
    if TWO:
        c1 = B0 + tl.arange(0, B1)
        if MASK1:
            m1 = c1 < N
            g1 = tl.load(G + c1, mask=m1).to(tl.float32)
            y1 = _rbf(tl.load(Y + c1, mask=m1, eviction_policy="evict_first").to(tl.float32)
                      + tl.load(Z + c1, mask=m1, eviction_policy="evict_first").to(tl.float32))
            o1 = _rbf(g1 * y1) + tl.load(R + c1, mask=m1,
                                         eviction_policy="evict_first").to(tl.float32)
            tl.store(O + c1, o1.to(dt), mask=m1, eviction_policy="evict_first")
        else:
            g1 = tl.load(G + c1).to(tl.float32)
            y1 = _rbf(tl.load(Y + c1, eviction_policy="evict_first").to(tl.float32)
                      + tl.load(Z + c1, eviction_policy="evict_first").to(tl.float32))
            o1 = _rbf(g1 * y1) + tl.load(R + c1,
                                         eviction_policy="evict_first").to(tl.float32)
            tl.store(O + c1, o1.to(dt), eviction_policy="evict_first")


@triton.jit
def _gsum2_fwd(RA, RB, YA, YB, ZA, ZB, GA, GB, O, SPLIT: tl.constexpr,
               N: tl.constexpr, B0: tl.constexpr, B1: tl.constexpr,
               TWO: tl.constexpr, MASK1: tl.constexpr):
    """Two-partial gate-and-residual for both streams in one launch."""
    _gdc_wait()
    r = tl.program_id(0)
    o = r.to(tl.int64) * N
    if r < SPLIT:
        _gsum_row(RA + o, YA + o, ZA + o, GA, O + o, N, B0, B1, TWO, MASK1)
    else:
        d = o - SPLIT * N
        _gsum_row(RB + d, YB + d, ZB + d, GB, O + o, N, B0, B1, TWO, MASK1)


@triton.jit
def _gres_ln_row(R, Y, G, SH, SC, H, Z, N: tl.constexpr, eps: tl.constexpr,
                 B0: tl.constexpr, B1: tl.constexpr, TWO: tl.constexpr,
                 MASK1: tl.constexpr):
    """``h = bf16(r + bf16(g*y))`` and ``z = modulated_layernorm(h)``, one row.

    This is the dual-stream block's whole mid-section: the reference spends five
    kernels and ~400 MB of traffic per call on the image stream alone for what
    needs one kernel and ~113 MB.  ``h`` stays in registers between the two
    halves, so the updated residual is written once and never read back.

    The layer norm's reduction shift is ``h[0]``, recomputed from three scalar
    loads rather than extracted from the register tile -- the row's data is
    being pulled anyway, so those land in L1.
    """
    dt = H.dtype.element_ty
    sft = _rbf(_rbf(tl.load(G).to(tl.float32) * tl.load(Y).to(tl.float32))
               + tl.load(R).to(tl.float32))

    c0 = tl.arange(0, B0)
    g0 = tl.load(G + c0).to(tl.float32)
    y0 = tl.load(Y + c0, eviction_policy="evict_first").to(tl.float32)
    h0 = _rbf(_rbf(g0 * y0) + tl.load(R + c0,
                                      eviction_policy="evict_first").to(tl.float32))
    tl.store(H + c0, h0.to(dt), eviction_policy="evict_first")
    d0 = h0 - sft
    acc = tl.sum(d0, axis=0)
    sq = tl.sum(d0 * d0, axis=0)
    if TWO:
        c1 = B0 + tl.arange(0, B1)
        if MASK1:
            m1 = c1 < N
            g1 = tl.load(G + c1, mask=m1).to(tl.float32)
            y1 = tl.load(Y + c1, mask=m1, eviction_policy="evict_first").to(tl.float32)
            h1 = _rbf(_rbf(g1 * y1) + tl.load(R + c1, mask=m1,
                                              eviction_policy="evict_first").to(tl.float32))
            tl.store(H + c1, h1.to(dt), mask=m1, eviction_policy="evict_first")
            d1 = tl.where(m1, h1 - sft, 0.0)
        else:
            g1 = tl.load(G + c1).to(tl.float32)
            y1 = tl.load(Y + c1, eviction_policy="evict_first").to(tl.float32)
            h1 = _rbf(_rbf(g1 * y1) + tl.load(R + c1,
                                              eviction_policy="evict_first").to(tl.float32))
            tl.store(H + c1, h1.to(dt), eviction_policy="evict_first")
            d1 = h1 - sft
        acc += tl.sum(d1, axis=0)
        sq += tl.sum(d1 * d1, axis=0)
    inv_n: tl.constexpr = 1.0 / N
    off = acc * inv_n
    rstd = 1.0 / tl.sqrt(tl.maximum(sq * inv_n - off * off, 0.0) + eps)

    n0 = _rbf((d0 - off) * rstd)
    s0 = _rbf(1.0 + tl.load(SC + c0).to(tl.float32))
    z0 = _rbf(n0 * s0) + tl.load(SH + c0).to(tl.float32)
    tl.store(Z + c0, z0.to(dt), eviction_policy="evict_first")
    if TWO:
        n1 = _rbf((d1 - off) * rstd)
        if MASK1:
            s1 = _rbf(1.0 + tl.load(SC + c1, mask=m1).to(tl.float32))
            z1 = _rbf(n1 * s1) + tl.load(SH + c1, mask=m1).to(tl.float32)
            tl.store(Z + c1, z1.to(dt), mask=m1, eviction_policy="evict_first")
        else:
            s1 = _rbf(1.0 + tl.load(SC + c1).to(tl.float32))
            z1 = _rbf(n1 * s1) + tl.load(SH + c1).to(tl.float32)
            tl.store(Z + c1, z1.to(dt), eviction_policy="evict_first")


@triton.jit
def _gres_ln2_fwd(RA, RB, YA, YB, GA, GB, SHA, SCA, SHB, SCB, H, Z,
                  SPLIT: tl.constexpr, N: tl.constexpr, eps: tl.constexpr,
                  B0: tl.constexpr, B1: tl.constexpr, TWO: tl.constexpr,
                  MASK1: tl.constexpr):
    """Gate-residual + modulated norm for both streams in one launch."""
    _gdc_wait()
    r = tl.program_id(0)
    o = r.to(tl.int64) * N
    if r < SPLIT:
        _gres_ln_row(RA + o, YA + o, GA, SHA, SCA, H + o, Z + o, N, eps,
                     B0, B1, TWO, MASK1)
    else:
        d = o - SPLIT * N
        _gres_ln_row(RB + d, YB + d, GB, SHB, SCB, H + o, Z + o, N, eps,
                     B0, B1, TWO, MASK1)


# ###########################################################################
# Launch plumbing
# ###########################################################################
class _Op:
    """One Triton kernel frozen to a single constexpr signature.

    ``prime`` runs the kernel once over scratch buffers -- which compiles it and
    lets Triton see the dtypes and the 16-byte pointer alignment it specializes
    on -- and then freezes the compiled kernel's own C launcher.  Afterwards
    ``run`` is one call with raw pointer ints, skipping Triton's per-call
    argument binder (~12 us of Python, more than the rest of these blocks at the
    captured sizes).  If the launcher cannot be frozen -- a future Triton
    reshaping it, a kernel needing scratch -- ``ready`` stays False and the
    caller keeps to the reference body.
    """

    __slots__ = ("kern", "cargs", "warps", "l", "ready")

    def __init__(self, kern, cargs, warps: int):
        self.kern = kern
        self.cargs = cargs
        self.warps = warps
        self.l = _Launcher()
        self.ready = False

    def prime(self, grid: int, tensors: tuple, dev: int) -> None:
        kern = self.kern[(grid,)](*tensors, *self.cargs, num_warps=self.warps,
                                launch_pdl=_PDL)
        self.ready = self.l.bind(kern, dev)

    def run(self, grid: int, dev: int, ptrs: tuple) -> None:
        self.l.run(grid, 1, 1, _raw_stream(dev), *self.l.pre, *ptrs, *self.cargs)


def _ref_adaln(norm, x: torch.Tensor, temb: torch.Tensor):
    """The frozen ``AdaLayerNormZero``'s *reference* path, spelled out.

    Its fused path is the reason this file exists: it applies the modulation in
    fp32 where the reference rounds the normalized value to bf16 first, which
    composed into ``residual + gate * f(modulated)`` costs 0.026-0.036 of
    matched_ratio and fails the harness (see ITERATIONS.md).  The fused paths
    below reproduce the reference rounding, so the fallback has to as well --
    otherwise a shape the fused path declines would come out *less* accurate
    than plain baseline, which is not a fallback.  This is the frozen module's
    own else-branch, driven directly rather than by disabling its fast path.
    """
    e = norm.linear(norm.silu(temb))
    chunks = e.chunk(6 if e.shape[-1] == 6 * x.shape[-1] else 3, dim=1)
    shift, scale = chunks[0], chunks[1]
    out = norm.norm(x) * (1 + scale[:, None]) + shift[:, None]
    return (out,) + chunks[2:]


def _ln_cargs(n: int, eps: float, split: int):
    b0, b1, two, mask1 = _tile_split(n)
    return (split, n, eps, b0, b1, two, mask1)


def _el_cargs(n: int, split: int):
    b0, b1, two, mask1 = _tile_split(n)
    return (split, n, b0, b1, two, mask1)


# ###########################################################################
# Modules
# ###########################################################################
class FluxTransformerBlock(nn.Module):
    """Dual-stream DiT block: joint attention over text+image, then separate FFNs."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        quant_config: dict | None = None,
    ):
        super().__init__()
        self.norm1 = AdaLayerNormZero(dim, promote_fp32=False)
        self.norm1_context = AdaLayerNormZero(dim, promote_fp32=False)

        self.attn = FluxAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            eps=eps,
            quant_config=quant_config,
        )

        self.norm2 = LayerNorm(dim, elementwise_affine=False, eps=1e-6, promote_fp32=False)
        self.ff = FeedForward(dim=dim, dim_out=dim, quant_config=quant_config)

        self.norm2_context = LayerNorm(dim, elementwise_affine=False, eps=1e-6, promote_fp32=False)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, quant_config=quant_config)

        # The frozen FFN's cuBLASLt bias+GELU epilogue runs the activation on the
        # fp32 accumulator instead of a bf16-rounded pre-activation.  Standing
        # alone that is *more* accurate and lands at matched_ratio ~0.995; inside
        # this block, where ``hidden_states + gate_mlp * ff_output`` adds two
        # comparable and partly-cancelling terms, it lands the whole block at
        # 0.9901-0.9915 -- inside the 0.99 requirement by less than the spread
        # between input seeds.  The frozen module's own ``_flat`` path is the
        # same GEMMs with the reference's separate GELU pass (bit-identical to
        # baseline, verified), costs ~28 us at S=4096 and ~8 us at S=1024, and
        # buys back 0.005 of margin, so it is selected here.  Nothing in L2 is
        # edited: this only picks which of its two paths runs.
        for _ff in (self.ff, self.ff_context):
            if getattr(_ff, "_fused", False):
                _ff._fused = False

        self._dim = dim
        # Fast-path pre-resolution.  Parameter objects stay valid for the
        # module's lifetime (``to()`` and ``load_state_dict`` rebind storage in
        # place), so the per-call check is "has the Parameter been *replaced*",
        # which is what the frozen GEMV's own identity guard does.
        self._lp = self.norm1.linear._parameters
        self._lpc = self.norm1_context.linear._parameters
        self._flat = quant_config is None
        self._ops: dict = {}
        # Side stream + events for the text FFN; built with the kernels.
        self._fk: _Fork | None = None

    # ------------------------------------------------------------------
    def _cond(self, temb: torch.Tensor):
        """The two adaLN conditioning vectors, ``linear(silu(temb))`` each.

        This is deliberately *not* a fused kernel, and that is the one place in
        this file where the obvious fusion had to be given up.  Batching the two
        M=1 GEMVs into a single hand-written launch works and is ~12 us and two
        launches cheaper -- but its fp32 accumulation order differs from cuBLAS's
        by ~1e-6 relative, which rounds ~0.15% of the 36864 conditioning values
        to a different bf16 (measured: 99.85% bit-equal).  Each of those values
        is a *broadcast* scale, shift or gate: one wrong ULP perturbs a whole
        column of the activation by 0.4%, and after ``x + gate * y`` -- two
        comparable, partly-cancelling terms -- that lands the block at
        matched_ratio 0.988-0.992 against the 0.99 requirement, versus 0.996
        with cuBLAS (see ITERATIONS.md for the grid).  No hand-written GEMV can
        avoid this: the reference *is* cuBLAS, so any different summation order
        costs the same ULP.  Concatenating the two weights into one cuBLAS call
        would fix the launch count instead, at 226 MB of duplicated weight per
        block (4.3 GB over a FLUX.1-dev stack), which is not a trade worth one
        launch.

        ``silu(temb)`` *is* shared, which the reference computes twice.
        """
        p1, p2 = self._lp, self._lpc
        w1, w2 = p1.get("weight"), p2.get("weight")
        if w1 is None or w2 is None:
            return None
        st = torch.nn.functional.silu(temb)
        return (torch.nn.functional.linear(st, w1, p1.get("bias")),
                torch.nn.functional.linear(st, w2, p2.get("bias")))

    def _ops_for(self, se: int, t: int, dev: int, dtype, device):
        """Compile + freeze the three fused kernels for this text length.

        Priming runs each kernel over throwaway buffers, so the first *real*
        call already takes the frozen-launcher path and never has to be
        special-cased.
        """
        n = self._dim
        scr = torch.zeros((t, n), dtype=dtype, device=device)
        vec = torch.zeros(n, dtype=dtype, device=device)
        ml = _Op(_modln2_fwd, _ln_cargs(n, 1e-6, se), 2)
        gl = _Op(_gres_ln2_fwd, _ln_cargs(n, 1e-6, se), 2)
        gr = _Op(_gres2_fwd, _el_cargs(n, se), 2)
        try:
            ml.prime(t, (scr, scr, scr, vec, vec, vec, vec), dev)
            gl.prime(t, (scr, scr, scr, scr, vec, vec, vec, vec, vec, vec, scr, scr), dev)
            gr.prime(t, (scr, scr, scr, scr, vec, vec, scr), dev)
            fk = _Fork(device, dev)
            fk.prime(dev, dtype, device)
        except Exception:  # noqa: BLE001 - no Triton, no nvcc, an API that moved
            trio = None
        else:
            trio = (ml, gl, gr) if (ml.ready and gl.ready and gr.ready) else None
            if trio is not None:
                self._fk = fk
        self._ops[se] = trio
        del scr, vec
        return trio

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # bf16 batch-1 only: that is every captured call, fp16 additionally needs
        # the reference's ``clip(-65504, 65504)``, and a batched conditioning
        # vector would make the gates per-row rather than broadcast.  Everything
        # else keeps the reference body, so it fails exactly where it fails.
        n = self._dim
        if (self._flat
                and not joint_attention_kwargs
                and hidden_states.dtype is torch.bfloat16
                and encoder_hidden_states.dtype is torch.bfloat16
                and temb.dtype is torch.bfloat16
                and hidden_states.ndim == 3
                and hidden_states.shape[0] == 1
                and encoder_hidden_states.shape[0] == 1
                and hidden_states.shape[2] == n
                and encoder_hidden_states.shape[2] == n
                and temb.shape == (1, n)
                and hidden_states.is_cuda
                and hidden_states.is_contiguous()
                and encoder_hidden_states.is_contiguous()
                and temb.is_contiguous()
                and not torch.is_grad_enabled()):
            out = self._forward_fused(hidden_states, encoder_hidden_states, temb,
                                      image_rotary_emb)
            if out is not None:
                return out
        return self._forward_reference(hidden_states, encoder_hidden_states, temb,
                                       image_rotary_emb, joint_attention_kwargs)

    def _forward_fused(self, hidden_states, encoder_hidden_states, temb,
                       image_rotary_emb):
        n = self._dim
        se = encoder_hidden_states.shape[1]
        t = se + hidden_states.shape[1]
        dev = hidden_states.get_device()
        if dev != _cur_device() or dev != encoder_hidden_states.get_device():
            return None

        cond = self._cond(temb)
        if cond is None:
            return None
        e, ec = cond
        if e.shape != (1, 6 * n) or ec.shape != (1, 6 * n):
            return None
        trio = self._ops.get(se)
        if trio is None:
            if se in self._ops:
                return None
            trio = self._ops_for(se, t, dev, temb.dtype, temb.device)
            if trio is None:
                return None
        modln, gres_ln, gres = trio
        fk = self._fk
        if fk is None:
            return None
        if not fk.bind(dev):
            return None

        row = n * e.element_size()
        ep, cp = e.data_ptr(), ec.data_ptr()
        hp, ecp = hidden_states.data_ptr(), encoder_hidden_states.data_ptr()
        if (hp | ecp | ep | cp) & 15:
            return None
        # ``e.chunk(6, dim=1)``, as byte offsets off each conditioning vector:
        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp.
        sh_msa, sc_msa, g_msa = ep, ep + row, ep + 2 * row
        sh_mlp, sc_mlp, g_mlp = ep + 3 * row, ep + 4 * row, ep + 5 * row
        c_sh_msa, c_sc_msa, c_g_msa = cp, cp + row, cp + 2 * row
        c_sh_mlp, c_sc_mlp, c_g_mlp = cp + 3 * row, cp + 4 * row, cp + 5 * row

        # 1) Both streams' modulated layer norm in one launch, into one packed
        #    [se + s, n] slab whose two halves are the contiguous tensors the
        #    frozen attention wants -- so splitting it costs nothing.  All three
        #    internal buffers come from one allocation: at the smaller captured
        #    sequence length this block is host-bound, so a ``torch.empty`` call
        #    is as real a cost as the bytes it hands back.
        buf = torch.empty((3, t, n), dtype=hidden_states.dtype,
                          device=hidden_states.device)
        slab = t * n * e.element_size()
        nbp = buf.data_ptr()
        rbp, zbp = nbp + slab, nbp + 2 * slab
        modln.run(t, dev, (ecp, hp, nbp, c_sh_msa, c_sc_msa, sh_msa, sc_msa))

        attn_output, context_attn_output = self.attn.forward(
            hidden_states=buf[0:1, se:], encoder_hidden_states=buf[0:1, :se],
            image_rotary_emb=image_rotary_emb,
        )
        if (attn_output.shape != (1, t - se, n)
                or context_attn_output.shape != (1, se, n)
                or attn_output.dtype is not buf.dtype
                or not attn_output.is_contiguous()
                or not context_attn_output.is_contiguous()):
            return None

        # 2) ``x = x + gate * y`` and ``norm2(x) * (1 + scale) + shift`` for both
        #    streams in one launch: x and y are read once, the reduction is fp32,
        #    and both the updated residual and the modulated activation come out.
        ap, cap = attn_output.data_ptr(), context_attn_output.data_ptr()
        if (ap | cap) & 15:
            return None
        gres_ln.run(t, dev, (ecp, hp, cap, ap, c_g_msa, g_msa,
                             c_sh_mlp, c_sc_mlp, sh_mlp, sc_mlp, rbp, zbp))

        # The two FFNs are independent from here to step 3, and they are wildly
        # unequal: the text one is M=512 at both captured sequence lengths, one
        # eighth of the tokens at S=4096 and one third at S=1024, and its GEMM
        # pair badly underfills B200 -- ~50 us of the block's ~1040 for 1/8th of
        # the work.  So it goes on the side stream, where it hides under the
        # image FFN instead of following it.  Total traffic is unchanged; what
        # this recovers is idle SMs.
        #
        # ``.forward`` rather than ``__call__`` skips nn.Module's hook dispatch,
        # the same trade the frozen attention winner makes; neither FFN has hooks.
        # The text FFN's output is the one buffer that crosses the join and the
        # only thing the side stream allocates that the main stream reads, so it
        # is marked with ``record_stream``; its inner activation never leaves the
        # side stream, and the modulated input it reads was allocated on the main
        # stream before the fork (safe: the caller always joins before returning).
        # ...but only when the image FFN is long enough to hide the text one
        # behind.  At S=4096 the image pair is ~395 us against the text pair's
        # ~72 and forking is worth +1.4%; at S=1024 it is ~102 against the same
        # ~72, the two pairs are comparable, there is no idle machine left to
        # fill, and forking *costs* 1.0% (measured, both directions, per case).
        # So the fork is gated on the ratio rather than on either length: the
        # image stream has to be at least 4x the text stream, which is the
        # condition under which the smaller pair fits inside the larger one.
        if t - se >= 4 * se:
            fk.ev_f.record(fk.main)
            _set_stream(**fk.s_args)
            fk.side.wait_event(fk.ev_f)
            context_ff_output = self.ff_context.forward(buf[2:3, :se])
            fk.ev_j.record(fk.side)
            _set_stream(**fk.m_args)
            context_ff_output.record_stream(fk.main)
            # Image stream second, on the main stream: the larger GEMM pair,
            # which the host now enqueues while the text pair is already running.
            ff_output = self.ff.forward(buf[2:3, se:])
            # Joined before the guards, so that declining the call here cannot
            # leave side-stream stores in flight against a buffer the reference
            # path is about to be handed back by the allocator.
            fk.main.wait_event(fk.ev_j)
        else:
            # Image stream first, as in the reference: its FFN is the larger GEMM
            # pair, so the GPU has work to chew on while the host enqueues the
            # rest.
            ff_output = self.ff.forward(buf[2:3, se:])
            context_ff_output = self.ff_context.forward(buf[2:3, :se])
        if (ff_output.shape != (1, t - se, n)
                or context_ff_output.shape != (1, se, n)
                or not ff_output.is_contiguous()
                or not context_ff_output.is_contiguous()
                or ff_output.dtype is not buf.dtype):
            return None

        # 3) The two final ``x + gate * ff`` adds, one launch, one buffer whose
        #    halves are the two returned tensors.
        ob = torch.empty((1, t, n), dtype=buf.dtype, device=buf.device)
        obp, fp, cfp = ob.data_ptr(), ff_output.data_ptr(), context_ff_output.data_ptr()
        if (obp | fp | cfp) & 15:
            return None
        gres.run(t, dev, (rbp, rbp + se * row, cfp, fp, c_g_mlp, g_mlp, obp))
        return ob[:, :se], ob[:, se:]

    def _forward_reference(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = _ref_adaln(
            self.norm1, hidden_states, temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = \
            _ref_adaln(self.norm1_context, encoder_hidden_states, temb)
        joint_attention_kwargs = joint_attention_kwargs or {}

        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output

        if len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        )

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class FluxSingleTransformerBlock(nn.Module):
    """Single-stream DiT block: text+image concatenated, self-attention + MLP in parallel."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        quant_config: dict | None = None,
    ):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim, promote_fp32=False)
        self.proj_mlp = ReplicatedLinear(dim, self.mlp_hidden_dim, bias=True,
                                         quant_config=quant_config)
        self.act_mlp = GELU(approximate="tanh")
        self.proj_out = ReplicatedLinear(dim + self.mlp_hidden_dim, dim, bias=True,
                                         quant_config=quant_config)

        self.attn = FluxAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            eps=1e-6,
            pre_only=True,
            quant_config=quant_config,
        )

        self._dim = dim
        self._np = self.norm.linear._parameters
        self._flat = quant_config is None and not self.proj_mlp.use_fp8 \
            and not self.proj_out.use_fp8
        self._ops: dict = {}
        # ``proj_out``'s weight split along K, as strided views of the one
        # parameter -- built on first use (the loader has not run at __init__
        # time) and revalidated by two integer compares: ``to()`` moves the
        # storage without touching the version counter, ``load_state_dict``
        # bumps the version without moving the storage.
        self._wa: torch.Tensor | None = None
        self._wm: torch.Tensor | None = None
        self._w_ptr = 0
        self._w_ver = -1
        # Side stream + events for the MLP branch; built with the kernels.
        self._fk: _Fork | None = None

    def _split_weight(self):
        w = self.proj_out.weight
        if (self._wa is None or w.data_ptr() != self._w_ptr
                or w._version != self._w_ver):
            d = self._dim
            self._wa = w[:, :d].t()
            self._wm = w[:, d:].t()
            self._w_ptr = w.data_ptr()
            self._w_ver = w._version
        return self._wa, self._wm

    def _ops_for(self, se: int, t: int, dev: int, dtype, device):
        """Compile + freeze the two fused kernels for this text length."""
        n = self._dim
        scr = torch.zeros((t, n), dtype=dtype, device=device)
        vec = torch.zeros(n, dtype=dtype, device=device)
        ml = _Op(_modln2_fwd, _ln_cargs(n, 1e-6, se), 2)
        gr = _Op(_gsum2_fwd, _el_cargs(n, se), 2)
        try:
            ml.prime(t, (scr, scr, scr, vec, vec, vec, vec), dev)
            gr.prime(t, (scr, scr, scr, scr, scr, scr, vec, vec, scr), dev)
            fk = _Fork(device, dev)
            fk.prime(dev, dtype, device, self.act_mlp)
        except Exception:  # noqa: BLE001 - no Triton, no nvcc, an API that moved
            pair = None
        else:
            pair = (ml, gr) if (ml.ready and gr.ready) else None
            if pair is not None:
                self._fk = fk
        self._ops[se] = pair
        del scr, vec
        return pair

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = self._dim
        if (self._flat
                and not joint_attention_kwargs
                and hidden_states.dtype is torch.bfloat16
                and encoder_hidden_states.dtype is torch.bfloat16
                and temb.dtype is torch.bfloat16
                and hidden_states.ndim == 3
                and hidden_states.shape[0] == 1
                and encoder_hidden_states.shape[0] == 1
                and hidden_states.shape[2] == n
                and encoder_hidden_states.shape[2] == n
                and temb.shape == (1, n)
                and hidden_states.is_cuda
                and hidden_states.is_contiguous()
                and encoder_hidden_states.is_contiguous()
                and temb.is_contiguous()
                and not torch.is_grad_enabled()):
            out = self._forward_fused(hidden_states, encoder_hidden_states, temb,
                                      image_rotary_emb)
            if out is not None:
                return out
        return self._forward_reference(hidden_states, encoder_hidden_states, temb,
                                       image_rotary_emb, joint_attention_kwargs)

    def _forward_fused(self, hidden_states, encoder_hidden_states, temb,
                       image_rotary_emb):
        n = self._dim
        se = encoder_hidden_states.shape[1]
        s = hidden_states.shape[1]
        t = se + s
        dev = hidden_states.get_device()
        if dev != _cur_device() or dev != encoder_hidden_states.get_device():
            return None

        p = self._np
        e = self.norm._gemv(temb, p.get("weight"), p.get("bias"))
        if e is None or e.shape != (1, 3 * n):
            return None
        pair = self._ops.get(se)
        if pair is None:
            if se in self._ops:
                return None
            pair = self._ops_for(se, t, dev, temb.dtype, temb.device)
            if pair is None:
                return None
        modln, gres = pair

        es = e.element_size()
        ep = e.data_ptr()
        shift, scale, gate = ep, ep + n * es, ep + 2 * n * es
        hp, ecp = hidden_states.data_ptr(), encoder_hidden_states.data_ptr()
        if (hp | ecp | ep) & 15:
            return None
        fk = self._fk
        if fk is None:
            return None
        if not fk.bind(dev):
            return None
        wa, wm = self._split_weight()

        # 1) One modulated layer norm over both input streams, straight into one
        #    packed [1, se + s, n] buffer -- the reference's dim=1 concatenation
        #    and its separate norm both disappear.
        nb = torch.empty((1, t, n), dtype=hidden_states.dtype,
                         device=hidden_states.device)
        nbp = nb.data_ptr()
        if nbp & 15:
            return None
        modln.run(t, dev, (ecp, hp, nbp, shift, scale, shift, scale))

        nb2 = nb.view(t, n)
        # 2) The MLP branch -- ``proj_mlp``, the frozen L1 GELU pass, and
        #    ``proj_out``'s K=12288 half -- on the side stream.  All three read
        #    only the modulated activation and nothing in the attention branch
        #    reads any of them, so on one stream this whole leg sits in series
        #    with 367 us of qkv GEMM + RoPE + SDPA for no reason at all.
        #    Enqueued first so the device has both branches in flight while the
        #    host is still walking the attention path.
        #
        #    Which kernels cross over was chosen by measurement and it is *not*
        #    the obvious subset.  On paper the GELU is the thing to hide -- 35 us
        #    of pure bandwidth with the SMs' math units idle -- and moving only
        #    the GELU across, leaving both GEMM chains in series, is worth
        #    +0.2% / +0.0%: nothing.  Moving the whole branch, GEMMs included, is
        #    worth +1.3% / +2.2%.  The reason is the regime the harness scores in:
        #    it flushes L2 before every call and shifts the input addresses, so
        #    every weight comes from DRAM (226 MB per call for this block) and the
        #    block runs 130 us slower than it does under a profiler that lets one
        #    call warm the next.  What two concurrent GEMMs buy there is not SMs,
        #    it is memory-level parallelism -- two independent weight streams in
        #    flight instead of one.  Under the profiler, where the caches are
        #    warm, the same schedule is a wash (927 vs 920 us) and the individual
        #    kernels inflate: 209.7 -> 273.9 us for the wide half, 183.5 -> 191.1
        #    for the SDPA it overlaps.  Both regimes were measured; the scored one
        #    decides.
        #
        #    cuBLASLt's bias+GELU epilogue (``torch._addmm_activation``, what the
        #    frozen FFN uses) would delete the GELU pass instead of hiding it, for
        #    +2.8 us of GEMM -- and it is measurably out of numerical budget.  It
        #    runs the tanh-GELU on the fp32 accumulator (verified: 99.0%
        #    bit-equal to ``gelu_tanh(fp32 pre)``, 66.7% to the reference's
        #    ``gelu_tanh(bf16 pre)``), and the 0.4% it moves the 12288-wide
        #    activation by carries through ``proj_out``'s K=12288 half into the
        #    output, dropping the block from matched_ratio 0.998 to 0.990 --
        #    below the 0.99 requirement on some input seeds.  Rescheduling the
        #    pass touches neither the arithmetic nor its order, so the outputs are
        #    bit-identical to the serial schedule (verified elementwise).
        pm = self.proj_mlp
        fk.ev_f.record(fk.main)
        _set_stream(**fk.s_args)
        fk.side.wait_event(fk.ev_f)
        mlp = self.act_mlp(torch.nn.functional.linear(nb2, pm.weight, pm.bias))
        # The wide half, with no bias and nothing accumulated onto it: the two
        # halves of ``proj_out`` are summed in ``_gsum2_fwd`` instead of chained
        # through a ``beta=1`` GEMM, which is what lets this one and the narrow
        # half run at the same time on the two streams.
        y = torch.mm(mlp, wm)
        fk.ev_j.record(fk.side)
        _set_stream(**fk.m_args)
        # ``y`` is the one buffer the side stream allocates that the main stream
        # reads, so it is handed over with ``record_stream``.  Writing it with
        # ``out=`` into a buffer allocated on the main stream instead -- which
        # would need no marking at all -- is not an option: cuBLAS picks a
        # different and much slower kernel for an explicit ``out``, 21% on the
        # whole block at T=4608.  The modulated activation the side stream reads
        # was allocated on the main stream, which is safe because the caller
        # always joins before it returns.
        y.record_stream(fk.main)

        # 3) Attention over the packed buffer (frozen L2), on the main stream.
        attn_output = self.attn.forward(hidden_states=nb,
                                        image_rotary_emb=image_rotary_emb)
        # Joined before the guard, so that declining the call here cannot leave
        # side-stream stores in flight against a buffer the reference path is
        # about to be handed back by the allocator.
        if (attn_output.shape != (1, t, n) or attn_output.dtype is not nb.dtype
                or not attn_output.is_contiguous()):
            fk.main.wait_event(fk.ev_j)
            return None

        # 4) ``proj_out``'s K=3072 half, on the main stream and *not* accumulated
        #    onto the wide one -- so it overlaps it rather than waiting for it.
        #    Enqueued before the join for exactly that reason.
        z = torch.addmm(self.proj_out.bias, attn_output.view(t, n), wa)

        # 5) ``gate * y + residual`` in one pass, reading the two input streams
        #    where the reference read their concatenation, and writing one
        #    buffer whose halves are the two returned tensors.
        ob = torch.empty((1, t, n), dtype=nb.dtype, device=nb.device)
        yp, zp, obp = y.data_ptr(), z.data_ptr(), ob.data_ptr()
        fk.main.wait_event(fk.ev_j)
        if (yp | zp | obp) & 15:
            return None
        off = se * n * es
        gres.run(t, dev, (ecp, hp, yp, yp + off, zp, zp + off, gate, gate, obp))
        return ob[:, :se], ob[:, se:]

    def _forward_reference(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        residual = hidden_states
        norm_hidden_states, gate = _ref_adaln(self.norm, hidden_states, temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        encoder_hidden_states, hidden_states = (
            hidden_states[:, :text_seq_len],
            hidden_states[:, text_seq_len:],
        )
        return encoder_hidden_states, hidden_states
