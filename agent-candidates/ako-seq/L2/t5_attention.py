"""T5 self-attention with TP-aware QKV projection and relative position bias (L2).

Mirrors vllm-omni's T5SelfAttention: QKVParallelLinear -> manual SDPA ->
RowParallelLinear, with T5-style relative position bias computed per-partition.

What is optimized here
----------------------
The reference attention core is five separate passes over a
``[b, n_heads, seq, seq]`` tensor -- ``bmm`` writes bf16 scores, ``+= bias``
reads and rewrites them, ``.float()`` doubles them, ``softmax`` round-trips the
fp32 copy, ``.type_as`` reads it back and writes bf16 -- for a problem whose
actual math is ~4.3 GFLOP.  At the captured shape (b=1, h=64, s=512, d=64) that
is ~370 MB of HBM traffic for ~3 us of tensor-core work, and it dominates the
operator.  ``_t5_attn_fwd`` below is a flash-attention forward that never
materializes the score matrix: it streams the bias tile straight into the score
accumulator, keeps the running softmax state in registers, and writes the
result already in ``[b, s, h, d]`` order so the output projection consumes it
without a transposing copy.  Only the position bias (an input), q/k/v and the
output cross HBM: ~50 MB instead of ~370 MB.

Reproducing the reference's numerics is the hard part, not the traffic
-------------------------------------------------------------------
T5 does not scale the queries, and the benchmark's synthetic weights
(``normal_(0, 0.02)`` over ``d_model=4096``) put the scores at std ~13.  A
softmax that peaked is *extremely* sensitive to the score values: the
reference's ``bmm`` rounds its fp32 accumulator to bf16 (1 ULP = 0.06 at that
magnitude) and ``scores += position_bias`` rounds again, which perturbs
individual probabilities by several percent relative to an unrounded fp32
score.  A stock flash kernel -- ``F.scaled_dot_product_attention`` on any
backend, cuDNN included -- keeps the scores in fp32 and therefore *disagrees
with the reference on 58% of output elements* at the benchmark's
atol/rtol = 1e-2 (measured; the MATH backend fails identically, so this is
rounding, not the flash algorithm).  So the kernel emulates both roundings
explicitly (``.to(bfloat16).to(float32)`` after the QK dot and after the bias
add) and feeds bf16 probabilities to the PV dot, exactly as
``softmax(...).type_as(scores)`` does.  This is a correctness requirement, not
a micro-optimization; without it the fast path cannot pass.

The position bias arrives head-innermost
---------------------------------------
``position_bias`` as captured is ``compute_bias``'s own output: a contiguous
``[q, k, heads]`` buffer viewed as ``[1, heads, q, k]`` (stride
``[64, 1, 32768, 64]``).  Its fastest axis is the head, so one per-head score
tile touches a 128-byte sector for every useful 2 bytes.  Reading it that way
inside the fused kernel measures 89 us against 42 us for repacking it head-major
first (``_bias_to_head_major``, ~11 us of streaming copy), so the fast path
repacks.  ``compute_bias`` itself is built head-major to begin with, which is
also why it no longer materializes the ``[S, S]`` bucket tensor: the bucket id
is recomputed inside the gather (bit-identically -- see ``_bias_gather``), and
that tensor's ~19 elementwise launches were 108 us on their own.  Only a
returned tensor's values/shape/dtype are contractual, never its strides.

Where the time goes now (captured shape, per forward)
----------------------------------------------------
qkv projection 43 us (nvjet), fused attention 23 us, output projection 15 us,
bias repack/gather 12 us, plus a 3.8 us device-to-device copy the harness itself
does inside the timed window.  That sums to ~94 us against a ~96 us measured
window, i.e. the operator is now ~97% occupied by kernel work with only ~2 us of
inter-kernel launch latency left; the remaining levers are all inside those four
kernels.

Three things beyond the fused core earn that:

* **Both projection weights are kept as a contiguous ``[K, N]`` copy.**
  ``nn.Linear`` stores ``[N, K]``, which makes the GEMM a TN product and gets
  ``nvjet_..._TNT``; the transposed copy gets ``nvjet_..._NNT``, which is 4 us
  faster on qkv and 2 us on o with a cold L2 and **bitwise identical** in
  output.  Both GEMMs re-stream 100 MB / 34 MB of weights for only 512 rows of
  activations, so the layout that streams better wins (the same reason time is
  so sublinear in M: 31.7 us at M=128, 50.2 at M=512, 125.1 at M=2048).
* **Nothing per-call is rebuilt on the host.**  Triton's ``jitfn[grid](...)``
  re-derives the argument binding, specialization and cache key every call
  (10-18 us each); ``_launch`` caches the ``CompiledKernel`` and hands the
  driver the argument tuple directly.  The three q/k/v ``[z, h, s, d]`` views
  are gone too -- the kernel takes the packed projection output three times with
  element offsets -- and the internal buffers are reused.  Host cost per forward
  fell from 101 us to 46 us, which matters because anything above the device
  time shows up as gaps *inside* the timed window.
* **The bias prep is PDL-launched** so its blocks start as the projection's last
  CTAs retire (2.7 us of measured overlap).  A second CUDA stream was tried for
  the same purpose and is worse -- see ``forward`` and ITERATIONS.md.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl
from transformers import T5Config
from triton.runtime import driver as _triton_driver

from ....infra.tp import _tp_size, _tp_rank
from ..L1.embedding import Embedding
from ..L1.linear import BMM
from ..L1.softmax import Softmax
from .parallel_linear import QKVParallelLinear, RowParallelLinear

_LOG2E = tl.constexpr(1.4426950408889634)


# ---------------------------------------------------------------------------
# Launching Triton kernels without paying for the JIT wrapper every call
# ---------------------------------------------------------------------------
_LAUNCHERS: dict = {}


def _launch(jitfn, key, grid, args, warps, stages=None, pdl=False):
    """Launch *jitfn* on *grid* with *args* in declaration order (constexprs
    included), bypassing ``JITFunction.run`` after the first call.

    ``jitfn[grid](...)`` re-derives the argument binding, the specialization,
    the cache key and the launch metadata on *every* call: measured 10.1 us of
    host time for ``_bias_to_head_major`` and ~18 us for ``_t5_attn_fwd``
    against 4.6 us for handing the driver a cached ``CompiledKernel``.  That
    matters here because enqueueing one forward costs ~100 us of Python against
    ~93 us of device work, so any host time the harness cannot hide behind its
    L2 flush turns straight into an inter-kernel gap inside the timed window
    (see ITERATIONS.md).

    *key* must pin down everything Triton specializes on -- every non-tensor
    argument (or the geometry all of them are derived from), the tensor dtypes,
    and the 16-byte alignment of the pointers -- because a cache hit skips the
    binder that would otherwise notice a change.
    """
    entry = _LAUNCHERS.get(key)
    if entry is None:
        opts = {"num_warps": warps}
        if stages is not None:
            opts["num_stages"] = stages
        if pdl:
            opts["launch_pdl"] = True
        kernel = jitfn[grid](*args, **opts)
        if kernel is None:
            return
        if hasattr(kernel, "result"):
            kernel = kernel.result()
        _LAUNCHERS[key] = (kernel.run, kernel.function, kernel.packed_metadata)
        return
    run, fn, meta = entry
    device = _triton_driver.active.get_current_device()
    stream = _triton_driver.active.get_current_stream(device)
    run(grid[0], grid[1], grid[2], stream, fn, meta, None, None, None, *args)


# ---------------------------------------------------------------------------
# Fused bias-aware attention forward
# ---------------------------------------------------------------------------
@triton.jit
def _t5_attn_fwd(
    Q, K, V, BIAS, OUT,
    OQ, OK, OV,
    sqz, sqh, sqm,
    skz, skh, skn,
    svz, svh, svn,
    sbz, sbh, sbm, sbn,
    soz, som, soh,
    M, N,
    H: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, HAS_BIAS: tl.constexpr,
    WS: tl.constexpr, SDT: tl.constexpr,
):
    """out[z, m, h, :] = softmax_n(bf16(bf16(q.k) + bias))[m, :] @ v.

    ``Q``/``K``/``V`` are pointer + element-offset pairs rather than three
    tensors so the caller can hand the same packed projection output three
    times instead of building three ``view``/``transpose`` objects (8.9 us of
    host time per forward).

    One program per (query tile, batch * head).  ``BIAS`` is an additive
    ``[z, h, M, N]`` tensor read through explicit strides (broadcast axes come
    in as stride 0; ``_prep_bias`` has already made the last one 1 wherever
    that is worth doing); ``OUT`` is indexed ``[z, m, h, d]`` so the caller can
    hand it to the output projection unpermuted.

    The two ``.to(bfloat16).to(float32)`` pairs are load-bearing: they
    reproduce the bf16 ``scores`` tensor of the eager chain (see the module
    docstring).  ``p`` is cast to the V dtype before the second dot for the
    same reason -- the reference multiplies bf16 probabilities by V.
    """
    pid_m = tl.program_id(0)
    pid_zh = tl.program_id(1)
    z = pid_zh // H
    h = pid_zh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    m_in = offs_m < M

    qp = Q + OQ + z * sqz + h * sqh + offs_m[:, None] * sqm + offs_d[None, :]
    q = tl.load(qp) if EVEN_M else tl.load(qp, mask=m_in[:, None], other=0.0)

    kb = K + OK + z * skz + h * skh
    vb = V + OV + z * svz + h * svh
    bb = BIAS + z * sbz + h * sbh + offs_m[:, None] * sbm

    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)

    for start_n in tl.range(0, N, BLOCK_N, warp_specialize=WS):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_in = offs_n < N
        # K is read straight into [D, BLOCK_N] (its d axis is the contiguous
        # one) so the MMA needs no register transpose of the tile.
        kp = kb + offs_d[:, None] + offs_n[None, :] * skn
        k = tl.load(kp) if EVEN_N else tl.load(kp, mask=n_in[None, :], other=0.0)
        s = tl.dot(q, k)
        s = s.to(SDT).to(tl.float32)
        if HAS_BIAS:
            bp = bb + offs_n[None, :] * sbn
            if EVEN_M and EVEN_N:
                bias = tl.load(bp)
            else:
                bias = tl.load(bp, mask=m_in[:, None] & n_in[None, :], other=0.0)
            s = (s + bias.to(tl.float32)).to(SDT).to(tl.float32)
        if not EVEN_N:
            s = tl.where(n_in[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2((m_i - m_new) * _LOG2E)
        p = tl.exp2((s - m_new[:, None]) * _LOG2E)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        vp = vb + offs_n[:, None] * svn + offs_d[None, :]
        v = tl.load(vp) if EVEN_N else tl.load(vp, mask=n_in[:, None], other=0.0)
        acc = tl.dot(p.to(V.dtype.element_ty), v, acc)
        m_i = m_new

    acc = acc / l_i[:, None]
    op = OUT + z * soz + offs_m[:, None] * som + h * soh + offs_d[None, :]
    o = acc.to(OUT.dtype.element_ty)
    if EVEN_M:
        tl.store(op, o)
    else:
        tl.store(op, o, mask=m_in[:, None])


# (BLOCK_M, BLOCK_N, num_warps, num_stages, warp_specialize) per head_dim.
# Picked by the sweep recorded in ITERATIONS.md; the fallback entry covers
# untuned head dims.
_ATTN_CFG: dict[int, tuple[int, int, int, int, bool]] = {
    64: (128, 32, 4, 3, False),
}
_ATTN_CFG_DEFAULT = (128, 64, 8, 3, False)


def _attn_cfg(d: int) -> tuple[int, int, int, int, bool]:
    return _ATTN_CFG.get(d, _ATTN_CFG_DEFAULT)


_SUPPORTED_D = (16, 32, 64, 128, 256)


def _bias_strides(bias, z, h, m, n):
    """Effective (z, h, m, n) strides of *bias* broadcast to [z, h, m, n], or None."""
    if bias.ndim != 4:
        return None
    bz, bh, bm, bn = bias.shape
    if bn != n or bm != m or bz not in (1, z) or bh not in (1, h):
        return None
    return (0 if bz == 1 else bias.stride(0),
            0 if bh == 1 else bias.stride(1),
            bias.stride(2), bias.stride(3))


@triton.jit
def _bias_to_head_major(SRC, OUT, TOT, SN, SZ, H,
                        HP: tl.constexpr, BLOCK: tl.constexpr):
    """OUT[z, h, t] = SRC[z * SZ + t * SN + h] -- head-innermost to head-major.

    The captured ``position_bias`` is ``compute_bias``'s output, i.e. a
    contiguous ``[q, k, heads]`` buffer viewed as ``[1, heads, q, k]``, so its
    fastest axis is the head.  Read that way a per-head score tile costs one
    128-byte transaction per useful 2 bytes, so the tile is instead read
    head-major (contiguous rows) and transposed in registers, exactly as
    ``_bias_gather`` does for the table gather.
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    z = tl.program_id(1)
    hs = tl.arange(0, HP)
    tm = offs < TOT
    hm = hs < H
    tile = tl.load(SRC + z * SZ + offs[:, None] * SN + hs[None, :],
                   mask=tm[:, None] & hm[None, :], other=0.0)
    tl.store(OUT + (z * H + hs)[:, None] * TOT + offs[None, :], tl.trans(tile),
             mask=hm[:, None] & tm[None, :])


def _head_major_bias(bias, out, pdl=False):
    """Repack a head-innermost *bias* into the contiguous *out*, or ``None``."""
    zb, hb, mb, nb = bias.shape
    sn = bias.stride(3)
    sz = bias.stride(0)
    if (bias.stride(1) != 1 or bias.stride(2) != nb * sn or hb < 2
            or not bias.is_cuda):
        return None
    tot = mb * nb
    hp = triton.next_power_of_2(hb)
    args = (bias, out, tot, sn, sz, hb, hp, 128)
    _launch(_bias_to_head_major,
            ("hm", bias.dtype, zb, hb, tot, sn, sz, hp, pdl,
             (bias.data_ptr() | out.data_ptr()) & 15),
            (-(-tot // 128), zb, 1), args, 4, pdl=pdl)
    return out


def _attn_launch(qkv, bias, z, s, h, d, out):
    """Fused attention over a packed ``[z * s, 3 * h * d]`` projection output.

    ``qkv`` is passed as Q, K and V with element offsets ``0 / h*d / 2*h*d``, so
    the three ``[z, h, s, d]`` views the kernel logically reads never exist as
    tensors.  Writes ``out``, viewed ``[z, s, h, d]``, so the output projection
    consumes it with no transposing copy.  Returns ``False`` when the bias
    layout falls outside what the kernel covers, so the caller can run the
    eager chain instead.
    """
    if bias is None:
        bstr = (0, 0, 0, 0)
        bias_t = qkv
    else:
        bstr = _bias_strides(bias, z, h, s, s)
        if bstr is None:
            return False
        bias_t = bias

    hd = h * d
    sqm = 3 * hd
    sqz = s * sqm
    bm, bn, warps, stages, ws = _attn_cfg(d)
    args = (qkv, qkv, qkv, bias_t, out,
            0, hd, 2 * hd,
            sqz, d, sqm,
            sqz, d, sqm,
            sqz, d, sqm,
            bstr[0], bstr[1], bstr[2], bstr[3],
            s * hd, hd, d,
            s, s,
            h, d, bm, bn, s % bm == 0, s % bn == 0, bias is not None, ws,
            tl.bfloat16 if qkv.dtype is torch.bfloat16 else tl.float16)
    _launch(_t5_attn_fwd,
            ("attn", qkv.dtype, bias_t.dtype, z, s, h, d, bstr,
             (qkv.data_ptr() | bias_t.data_ptr() | out.data_ptr()) & 15),
            (-(-s // bm), z * h, 1), args, warps, stages)
    return True


# ---------------------------------------------------------------------------
# Relative-position-bias table gather, straight into [h, q, k] order
# ---------------------------------------------------------------------------
@triton.jit
def _bias_gather(W, OUT, TOT, S2, wrow, hstart, H,
                 MAX_EXACT, LOG_DEN, MULT, NB2, CAP,
                 HP: tl.constexpr, BLOCK: tl.constexpr):
    """OUT[h, i * S2 + j] = W[bucket(j - i), hstart + h].

    The bucket id is recomputed here rather than read from a materialized
    ``[S, S]`` int64 tensor: the reference builds that tensor with ~19 separate
    elementwise launches over 262 k elements, which measures 113 us -- eight
    times the cost of the gather it feeds.  The float expression is a literal
    transcription of ``_relative_position_bucket`` (same ``logf``, same divisor
    rounded to fp32, same truncating int cast) so the ids are bit-identical;
    ``probe/`` checks that elementwise against the reference.

    One table row (``H`` contiguous values from a few-KB, L1-resident table) is
    gathered per bucket id and shared across all heads, then transposed in
    registers so each head's stores are contiguous runs of ``BLOCK`` elements.
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    hs = tl.arange(0, HP)
    tm = offs < TOT
    hm = hs < H

    i = offs // S2
    rp = (offs - i * S2) - i
    arp = tl.abs(rp)
    small = arp < MAX_EXACT
    large = MAX_EXACT + (tl.log(arp.to(tl.float32) / MAX_EXACT)
                         / LOG_DEN * MULT).to(tl.int32)
    bucket = tl.where(rp > 0, NB2, 0) + tl.where(small, arp,
                                                 tl.minimum(large, CAP))

    tile = tl.load(W + bucket[:, None] * wrow + (hstart + hs)[None, :],
                   mask=tm[:, None] & hm[None, :], other=0.0)
    tl.store(OUT + hs[:, None] * TOT + offs[None, :], tl.trans(tile),
             mask=hm[:, None] & tm[None, :])


def _gather_bias(weight, q_len, k_len, num_buckets, max_distance,
                 h_start, h_count, pdl=False):
    """[1, h_count, q_len, k_len] contiguous relative-position bias.

    Returns ``None`` if anything about the table rules out the fused path.
    Freshly allocated, never a reused scratch buffer: this tensor is what
    ``forward`` returns as ``position_bias``, so the caller owns it.
    """
    nb2 = num_buckets // 2
    max_exact = nb2 // 2
    if (weight.ndim != 2 or not weight.is_cuda or weight.stride(1) != 1
            or max_exact < 1 or max_distance <= max_exact
            or weight.shape[0] < num_buckets
            or weight.shape[1] < h_start + h_count
            or q_len <= 0 or k_len <= 0):
        return None
    tot = q_len * k_len
    out = torch.empty((1, h_count, q_len, k_len),
                      device=weight.device, dtype=weight.dtype)
    wrow = weight.stride(0)
    hp = triton.next_power_of_2(h_count)
    args = (weight, out, tot, k_len, wrow, h_start, h_count,
            max_exact, math.log(max_distance / max_exact), nb2 - max_exact,
            nb2, nb2 - 1, hp, 128)
    _launch(_bias_gather,
            ("gather", weight.dtype, tot, k_len, wrow, h_start, h_count,
             num_buckets, max_distance, hp, pdl,
             (weight.data_ptr() | out.data_ptr()) & 15),
            (-(-tot // 128), 1, 1), args, 4, pdl=pdl)
    return out


# ---------------------------------------------------------------------------
# Projections: hand cuBLAS the weight as [K, N] instead of [N, K]
# ---------------------------------------------------------------------------
def _linear2d(mod, x2, out=None):
    """``mod(x2)`` for a 2-D *x2*, computed through a ``[K, N]`` weight.

    ``nn.Linear``-style weights are ``[N, K]``, which makes the GEMM a TN
    product; cuBLAS answers that with ``nvjet_..._TNT``.  Handing it the same
    numbers stored as a contiguous ``[K, N]`` buffer instead selects
    ``nvjet_..._NNT``, which at these shapes (M=512, K=4096, N=12288 / 4096) is
    measurably faster with a cold L2 -- 46.1 vs 50.2 us for qkv and 21.5 vs
    23.5 us for o -- while producing a **bitwise identical** result (checked
    with ``torch.equal``).  Both GEMMs re-stream ~100 MB / ~34 MB of weights for
    only 512 rows of activations, so the layout that streams them better wins.

    The transposed copy is built once and cached on the submodule, keyed on the
    parameter's storage and version so a later ``load_state_dict`` invalidates
    it.  Anything unusual -- fp8, a bias, TP > 1, fp32, CPU, autograd -- falls
    through to the submodule's own ``forward``, which for a 2-D input returns
    the same 2-D shape.
    """
    w = getattr(mod, "weight", None)
    if (w is None or getattr(mod, "use_fp8", False)
            or getattr(mod, "bias", None) is not None
            or getattr(mod, "tp_size", 1) > 1
            or w.dtype not in (torch.bfloat16, torch.float16)
            or not w.is_cuda or w.ndim != 2
            or x2.dtype is not w.dtype or torch.is_grad_enabled()):
        return mod(x2)
    key = (w.data_ptr(), w._version)
    cached = getattr(mod, "_fk_wt", None)
    if cached is None or cached[0] != key:
        cached = (key, w.detach().t().contiguous())
        mod._fk_wt = cached
    if out is None:
        return torch.mm(x2, cached[1])
    return torch.mm(x2, cached[1], out=out)


class T5SelfAttention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.d_kv
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance

        tp_size = _tp_size()
        assert self.n_heads % tp_size == 0
        self.n_heads_per_partition = self.n_heads // tp_size

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.d_model,
            head_size=self.d_kv,
            total_num_heads=self.n_heads,
            total_num_kv_heads=self.n_heads,
            bias=False,
        )

        self.o = RowParallelLinear(self.inner_dim, self.d_model, bias=False)

        self.bmm = BMM()
        self.softmax = Softmax(dim=-1)
        # Reused internal buffers; not parameters, not part of state_dict.
        self._fk_scratch: dict = {}

        if has_relative_attention_bias:
            self.relative_attention_bias = Embedding(
                self.relative_attention_num_buckets, self.n_heads,
            )

    @staticmethod
    def _relative_position_bucket(
        relative_position: torch.Tensor,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> torch.Tensor:
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position),
            )
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )
        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large,
        )
        return relative_buckets

    def _bias_table(self) -> torch.Tensor:
        emb = self.relative_attention_bias
        w = getattr(emb, "weight", None)
        if w is None:
            w = emb.emb.weight
        return w

    def compute_bias(self, query_length: int, key_length: int, device: torch.device,
                     pdl: bool = False) -> torch.Tensor:
        tp_rank = _tp_rank()
        head_start = tp_rank * self.n_heads_per_partition
        head_end = head_start + self.n_heads_per_partition
        # Same values as the reference's ``emb(bucket)[:, :, hs:he].permute(2,0,1)``
        # but built contiguously and in one launch -- see the module docstring
        # and ``_bias_gather``.
        weight = self._bias_table()
        if weight.device.type == "cuda":
            out = _gather_bias(
                weight, query_length, key_length,
                self.relative_attention_num_buckets,
                self.relative_attention_max_distance,
                head_start, self.n_heads_per_partition, pdl,
            )
            if out is not None:
                return out
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position, bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        if relative_position_bucket.is_cuda:
            values = self.relative_attention_bias(relative_position_bucket)
        else:
            # The L1 embedding op is CUDA-only; the reference table gather is
            # exact either way, so CPU takes the aten path.
            values = torch.nn.functional.embedding(
                relative_position_bucket, self._bias_table())
        values = values[:, :, head_start:head_end]
        return values.permute(2, 0, 1).unsqueeze(0)

    def _eager_core(self, query_states, key_states, value_states, position_bias):
        """The reference bmm -> +bias -> fp32 softmax -> bmm chain.

        The frozen L1 ``Softmax`` this operator is built on is a CUDA-only
        Triton kernel, so on CPU it is called through ``F.softmax`` instead --
        which is exactly what the baseline's own L1 ``Softmax`` does.  Without
        that the whole forward raises on CPU tensors even though the baseline
        handles them.
        """
        scores = self.bmm(query_states, key_states.transpose(3, 2))
        scores += position_bias
        probs = scores.float()
        probs = (self.softmax(probs) if probs.is_cuda
                 else torch.nn.functional.softmax(probs, dim=-1))
        attn_output = self.bmm(probs.type_as(scores), value_states)
        return attn_output.transpose(1, 2).contiguous()

    def _scratch(self, key, shape, dtype, device):
        """A reused internal buffer.  Never anything ``forward`` returns.

        Every write to one of these is stream-ordered behind the previous
        call's reads of it, so reuse across calls is safe on a single stream;
        it saves a ~2.5 us ``torch.empty`` per 33.5 MB buffer per forward and
        keeps the caching allocator out of the timed window.
        """
        buf = self._fk_scratch.get(key)
        if buf is None:
            buf = torch.empty(shape, device=device, dtype=dtype)
            self._fk_scratch[key] = buf
        return buf

    def _prep_bias(self, bias, pdl=False):
        """Give the attention kernel a bias layout it can read coalesced.

        Reading the captured head-innermost layout through its own strides
        costs 89 us against 42 us for repack-then-read (measured; the repack
        itself is ~11 us of streaming copy), because a per-head score tile
        touches one 128-byte sector per useful 2 bytes.  Layouts that are
        neither last-dim-contiguous nor head-innermost are rare enough to hand
        to ``.contiguous()``.
        """
        if bias is None or bias.stride(-1) == 1:
            return bias
        if bias.ndim == 4:
            zb, hb, mb, nb = bias.shape
            out = self._scratch(("bias", zb, hb, mb, nb, bias.dtype, bias.device),
                                (zb, hb, mb, nb), bias.dtype, bias.device)
            packed = _head_major_bias(bias, out, pdl)
            if packed is not None:
                return packed
        return bias.contiguous()

    def _bias_branch(self, hidden_states, mask, position_bias, seq_length,
                     pdl=False):
        """The reference bias branch: ``(position_bias, bias_for_kernel)``.

        ``position_bias`` is what ``forward`` must return -- caller-supplied
        bias untouched, else ``compute_bias`` (or an explicit zero tensor) with
        ``mask`` added only when there is a mask.  ``bias_for_kernel`` is what
        the fused kernel should add, ``None`` when the bias is known to be
        exactly zero.
        """
        if position_bias is not None:
            return position_bias, position_bias
        if self.has_relative_attention_bias:
            position_bias = self.compute_bias(
                seq_length, seq_length, device=hidden_states.device, pdl=pdl,
            )
        else:
            position_bias = torch.zeros(
                (1, self.n_heads_per_partition, seq_length, seq_length),
                device=hidden_states.device, dtype=hidden_states.dtype,
            )
            if mask is None:
                # Adding an exact-zero bias is a no-op; skip the 33.5 MB read.
                return position_bias, None
        if mask is not None:
            position_bias = position_bias + mask
        return position_bias, position_bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length = hidden_states.shape[:2]
        n_heads = self.n_heads_per_partition
        d_kv = self.d_kv
        hd = n_heads * d_kv
        rows = batch_size * seq_length
        dtype = hidden_states.dtype
        device = hidden_states.device

        # Everything downstream stays 2-D: the projections are plain [rows, K]
        # x [K, N] GEMMs and the attention kernel indexes the packed buffer
        # through strides, so no reshape/transpose objects are built per call.
        x2 = hidden_states.reshape(rows, hidden_states.shape[-1])
        fast = (hidden_states.is_cuda and not torch.is_grad_enabled()
                and dtype in (torch.bfloat16, torch.float16)
                and d_kv in _SUPPORTED_D)
        qkv_buf = (self._scratch(("qkv", rows, 3 * hd, dtype, device), (rows, 3 * hd),
                                 dtype, device) if fast else None)
        qkv = _linear2d(self.qkv_proj, x2, qkv_buf)

        # The bias branch reads only ``position_bias`` / the embedding table,
        # never ``qkv``; only the attention kernel joins the two.  So it is
        # launched with PDL, which lets its blocks start as the projection's
        # last wave of CTAs retires instead of waiting for the whole GEMM
        # (measured: the repack begins 2.7 us before nvjet ends, worth ~3 us).
        #
        # Why that is safe: the immediately preceding kernel -- the primary that
        # PDL relaxes this launch against -- is always the qkv projection, which
        # reads only ``hidden_states`` and its weight and writes only the
        # long-lived qkv scratch buffer.  It can therefore neither read nor
        # write the bias prep's output, and everything enqueued before it (last
        # call's attention, which read the same bias scratch buffer) has already
        # completed.  ``pdl`` is passed down rather than assumed so that a
        # direct ``compute_bias`` call, whose predecessor is arbitrary, does not
        # inherit the relaxation.
        position_bias, bias_for_kernel = self._bias_branch(
            hidden_states, mask, position_bias, seq_length, fast)
        # Only worth (and only safe) to relayout the bias when the fused CUDA
        # kernel is the thing that will read it: the eager chain broadcasts the
        # bias through ATen, which is layout-agnostic and works on CPU.
        if (fast and bias_for_kernel is not None
                and bias_for_kernel.dtype == qkv.dtype
                and bias_for_kernel.ndim == 4):
            bias_for_kernel = self._prep_bias(bias_for_kernel, True)

        attn2 = None
        if (fast and qkv.dtype is dtype and qkv.stride(-1) == 1
                and qkv.stride(0) == 3 * hd
                and (bias_for_kernel is None
                     or (bias_for_kernel.dtype == dtype
                         and bias_for_kernel.ndim == 4))):
            buf = self._scratch(("attn", rows, hd, dtype, device), (rows, hd),
                                dtype, device)
            if _attn_launch(qkv, bias_for_kernel, batch_size, seq_length,
                            n_heads, d_kv, buf):
                attn2 = buf
        if attn2 is None:
            packed = qkv.view(batch_size, seq_length, 3, n_heads, d_kv)
            attn2 = self._eager_core(
                packed[:, :, 0].transpose(1, 2),
                packed[:, :, 1].transpose(1, 2),
                packed[:, :, 2].transpose(1, 2),
                position_bias,
            ).view(rows, hd)

        # The returned activation is freshly allocated -- it leaves this call.
        out = _linear2d(self.o, attn2)
        return out.view(batch_size, seq_length, out.shape[-1]), position_bias
