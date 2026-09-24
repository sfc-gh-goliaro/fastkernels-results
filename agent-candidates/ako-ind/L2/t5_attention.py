"""T5 self-attention with TP-aware QKV projection and relative position bias (L2).

Mirrors vllm-omni's T5SelfAttention: QKVParallelLinear -> manual SDPA ->
RowParallelLinear, with T5-style relative position bias computed per-partition.

The attention core is a fused Triton flash-attention kernel that streams the
additive per-head ``position_bias`` into the QK^T tile, so the [B, H, S, S]
scores tensor (33 MB bf16) and its fp32 softmax upcast (67 MB) are never
materialized.  Q/K/V are read straight out of the packed ``qkv_proj`` output
with strided pointer arithmetic and the result is written directly in
``[B, S, inner_dim]`` layout, removing the split/view/transpose and the
transpose/contiguous/view copies around the baseline's two bmms.  T5 applies no
1/sqrt(d_kv) scaling, so the logit scale is exactly 1.0.

Both bias cases feed that one kernel, and in both the preparation work is
independent of the qkv GEMM, so it runs on a side stream underneath it:

* a caller-supplied ``position_bias`` is *repacked* into a head-major
  contiguous buffer.  The captured tensor is head-**minor** (stride
  ``[64, 1, 32768, 64]``, the ``values.permute(2,0,1)`` layout that
  ``compute_bias`` returns), so a per-head [m, n] tile would touch one 128 B
  line per element;
* ``position_bias is None`` is served from a ``[heads, q+k-1]``
  relative-position table (the bucket ids depend only on the sequence lengths,
  so they are memoized) which a small kernel *expands* into the [1, H, S, S]
  tensor the ``(attn_output, position_bias)`` contract has to return.

Host dispatch is a co-bottleneck here, not a rounding error: a launch costs
9-16 us of Python + CUDA-API time on this box against ~85 us of GPU work for the
whole op, and the ``position_bias is None`` path (two extra kernels) needed
141 us of host time -- more than the GPU could consume, so the GPU idled.  The
fast path therefore also minimizes host work: Triton kernels are relaunched
through :class:`_Launcher` (``CompiledKernel.run`` directly, 4.4 us instead of
``JITFunction.run``'s 13.6 us), the two projections go straight to ``torch.mm``
on a cached transposed weight instead of through the parallel-linear modules, the
side stream is addressed by its raw handle rather than a ``torch.cuda.stream``
context manager, and cross-stream ordering uses two cached events instead of
``wait_stream``.  That is 106/141 us of host time down to 66/87 us.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from transformers import T5Config

from ....infra.tp import _tp_size, _tp_rank
from ..L1.embedding import Embedding
from ..L1.linear import BMM
from ..L1.softmax import Softmax
from .parallel_linear import QKVParallelLinear, RowParallelLinear

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - no Triton -> eager fallback
    _HAS_TRITON = False

# Raw cudaStream_t of the current stream: 0.05 us against 2.6 us for
# ``torch.cuda.current_stream().cuda_stream``.
_RAW_STREAM = getattr(torch._C, "_cuda_getCurrentRawStream", None)


# ---------------------------------------------------------------------------
# Tunables (the dev harness mutates these for sweeps; clear ``_PLANS`` after).
# ---------------------------------------------------------------------------
CFG = {
    "BLOCK_M": 128,          # flash: queries per CTA
    "BLOCK_N": 64,           # flash: keys per inner iteration
    "num_warps": 8,
    "num_stages": 3,
    "MATCH_BF16": True,      # round QK^T and QK^T+bias to bf16, as the
                             # materialized baseline chain does
    "H_FIRST": False,        # flash grid axis 0 = query block, so the CTAs
                             # sharing a head's K/V run concurrently (L2 reuse)
    "T_BLOCK": 64,           # bias-repack tile (flattened m*S+n)
    "T_WARPS": 4,
    "E_BLOCK_M": 64,         # bias-expand tile
    "E_BLOCK_N": 64,
    "E_WARPS": 8,
    "EXP2": True,            # ex2.approx softmax instead of libdevice expf
    "OVERLAP": True,         # bias prep on a side stream, under the qkv GEMM
    "FAST_LAUNCH": True,     # relaunch Triton kernels via CompiledKernel.run
    "FAST_GEMM": True,       # torch.mm + cached w.t() instead of the modules
    "WT_COPY": True,         # keep a contiguous [K, N] transpose of each weight
}


if _HAS_TRITON:

    @triton.jit
    def _bias_repack_kernel(
        SRC, DST,
        sb_b, sb_h, sb_m, sb_n,
        S, MN,
        H: tl.constexpr, BLOCK: tl.constexpr, BLOCK_H: tl.constexpr,
        EVEN_H: tl.constexpr,
    ):
        """Repack a [B, H, S, S] bias with arbitrary strides into a contiguous
        [B, H, S, S] buffer.  Reads a [BLOCK, H] (flattened-mn, h) tile --
        coalesced when the source is head-minor -- and stores it transposed."""
        pid = tl.program_id(0)
        b = tl.program_id(1)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs // S
        n = offs % S
        h = tl.arange(0, BLOCK_H)
        src = SRC + b * sb_b + h[None, :] * sb_h + m[:, None] * sb_m + n[:, None] * sb_n
        dst = DST + b * (H * MN) + h[None, :] * MN + offs[:, None]
        if EVEN_H:
            msk = offs[:, None] < MN
        else:
            msk = (offs[:, None] < MN) & (h[None, :] < H)
        x = tl.load(src, mask=msk, other=0.0, eviction_policy="evict_first")
        tl.store(dst, x, mask=msk)

    @triton.jit
    def _bias_table_kernel(
        BUCKET, W, OUT, R, sw_b, sw_h,
        BLOCK_R: tl.constexpr, EVEN_R: tl.constexpr,
    ):
        """``OUT[h, r] = W[BUCKET[r], h]`` -- the per-head bias value for every
        relative position, gathered from the tiny [num_buckets, H] embedding."""
        h = tl.program_id(0)
        r = tl.arange(0, BLOCK_R)
        if EVEN_R:
            bk = tl.load(BUCKET + r)
            tl.store(OUT + h * R + r, tl.load(W + bk * sw_b + h * sw_h))
        else:
            msk = r < R
            bk = tl.load(BUCKET + r, mask=msk, other=0)
            x = tl.load(W + bk * sw_b + h * sw_h, mask=msk, other=0.0)
            tl.store(OUT + h * R + r, x, mask=msk)

    @triton.jit
    def _bias_expand_kernel(
        TABLE, DST, S, R,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        EVEN_M: tl.constexpr, EVEN_N: tl.constexpr,
    ):
        """``DST[0, h, m, n] = TABLE[h, n - m + S - 1]`` over a [1, H, S, S]
        contiguous buffer (``R == 2 * S - 1`` is the table's row length)."""
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        h = tl.program_id(2)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        src = (TABLE + h * R + (S - 1) - offs_m[:, None] + offs_n[None, :])
        dst = DST + h * S * S + offs_m[:, None] * S + offs_n[None, :]
        if EVEN_M and EVEN_N:
            tl.store(dst, tl.load(src))
        else:
            msk = (offs_m[:, None] < S) & (offs_n[None, :] < S)
            tl.store(dst, tl.load(src, mask=msk, other=0.0), mask=msk)

    @triton.jit
    def _t5_flash_kernel(
        QKV, BIAS, OUT,
        sq_b, sq_s,
        sb_b, sb_h, sb_m, sb_n,
        so_b, so_s,
        S, HD,
        D: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        EVEN_M: tl.constexpr, EVEN_N: tl.constexpr,
        MATCH_BF16: tl.constexpr, H_FIRST: tl.constexpr, EXP2: tl.constexpr,
    ):
        """One (batch, head, BLOCK_M queries) tile of T5 attention.

        Q/K/V come from the packed ``[B, S, 3*H*D]`` projection output; the
        additive per-head bias tile is folded into QK^T inside the loop and the
        softmax accumulates in fp32.  Output is written straight to
        ``[B, S, H*D]``.
        """
        if H_FIRST:
            h = tl.program_id(0)
            pid_m = tl.program_id(1)
        else:
            pid_m = tl.program_id(0)
            h = tl.program_id(1)
        b = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        offs_n0 = tl.arange(0, BLOCK_N)

        q_base = QKV + b * sq_b + h * D
        k_base = q_base + HD
        v_base = q_base + 2 * HD
        if EVEN_M:
            q = tl.load(q_base + offs_m[:, None] * sq_s + offs_d[None, :])
        else:
            q = tl.load(q_base + offs_m[:, None] * sq_s + offs_d[None, :],
                        mask=offs_m[:, None] < S, other=0.0)
        b_base = BIAS + b * sb_b + h * sb_h + offs_m[:, None] * sb_m

        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, D], tl.float32)

        for start_n in range(0, S, BLOCK_N):
            offs_n = start_n + offs_n0
            if EVEN_N:
                k = tl.load(k_base + offs_n[:, None] * sq_s + offs_d[None, :])
                v = tl.load(v_base + offs_n[:, None] * sq_s + offs_d[None, :])
                bias = tl.load(b_base + offs_n[None, :] * sb_n)
            else:
                nm = offs_n < S
                k = tl.load(k_base + offs_n[:, None] * sq_s + offs_d[None, :],
                            mask=nm[:, None], other=0.0)
                v = tl.load(v_base + offs_n[:, None] * sq_s + offs_d[None, :],
                            mask=nm[:, None], other=0.0)
                bias = tl.load(b_base + offs_n[None, :] * sb_n,
                               mask=nm[None, :], other=0.0)

            qk = tl.dot(q, tl.trans(k))
            if MATCH_BF16:
                # The baseline rounds twice: the bmm rounds QK^T to the input
                # dtype, then ``scores += position_bias`` rounds again.
                qk = qk.to(bias.dtype).to(tl.float32)
                qk = (qk + bias.to(tl.float32)).to(bias.dtype).to(tl.float32)
            else:
                qk = qk + bias.to(tl.float32)
            if not EVEN_N:
                qk = qk + tl.where(offs_n < S, 0.0, float("-inf"))[None, :]

            if EXP2:
                qk = qk * 1.4426950408889634
                m_new = tl.maximum(m_i, tl.max(qk, 1))
                alpha = tl.math.exp2(m_i - m_new)
                p = tl.math.exp2(qk - m_new[:, None])
            else:
                m_new = tl.maximum(m_i, tl.max(qk, 1))
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        acc = acc / l_i[:, None]
        o_ptr = OUT + b * so_b + offs_m[:, None] * so_s + h * D + offs_d[None, :]
        o = acc.to(OUT.dtype.element_ty)
        if EVEN_M:
            tl.store(o_ptr, o)
        else:
            tl.store(o_ptr, o, mask=offs_m[:, None] < S)

def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


_SIDE_STREAMS: dict[int, tuple] = {}


def _side(device: torch.device):
    """(side stream, raw handle, event main->side, event side->main, spare).

    Cached because constructing a stream or an event costs more host time than
    the kernels they order.
    """
    idx = device.index if device.index is not None else torch.cuda.current_device()
    e = _SIDE_STREAMS.get(idx)
    if e is None:
        s = torch.cuda.Stream(device=idx)
        e = (s, s.cuda_stream, torch.cuda.Event(), torch.cuda.Event(),
             torch.cuda.Event())
        _SIDE_STREAMS[idx] = e
    return e


# ---------------------------------------------------------------------------
# Launch plans + scratch buffers.
#
# The op issues ~85 us of GPU work; ``JITFunction.run`` costs 13.6 us of host
# time per Triton launch and ``F.linear`` ~9 us, so the dispatch cost is the
# binding constraint unless it is cut.  Grids and constexpr tuples are memoized
# per shape, buffers are reused across calls, and each launch site keeps a
# ``_Launcher`` that relaunches the already-compiled kernel directly.
# ---------------------------------------------------------------------------
_PLANS: dict = {}
_SCRATCH: dict = {}


class _Launcher:
    """Relaunch one Triton kernel through ``CompiledKernel.run`` (4.4 us)
    instead of ``JITFunction.run`` (13.6 us), which re-binds the signature,
    recomputes the specialization and hashes a cache key every call.

    The compiled kernel is reused only while ``sig`` -- every scalar argument
    plus the low 4 address bits of every pointer argument, i.e. exactly the
    inputs Triton specializes on -- is unchanged; anything else falls back to
    the JIT path, which will look up or compile the right variant.
    """

    __slots__ = ("jit", "grid", "g", "warps", "stages", "ck", "fn", "pm", "sig")

    def __init__(self, jit, grid, warps, stages):
        n = len(grid)
        self.jit = jit
        self.grid = grid
        self.g = (grid[0], grid[1] if n > 1 else 1, grid[2] if n > 2 else 1)
        self.warps = warps
        self.stages = stages
        self.ck = None
        self.sig = None

    def _slow(self, args, sig, stream_obj):
        if stream_obj is None:
            ck = self.jit[self.grid](*args, num_warps=self.warps,
                                     num_stages=self.stages)
        else:
            with torch.cuda.stream(stream_obj):
                ck = self.jit[self.grid](*args, num_warps=self.warps,
                                         num_stages=self.stages)
        try:
            ck.run  # forces _init_handles(); populates .function/.packed_metadata
            self.fn = ck.function
            self.pm = ck.packed_metadata
            self.ck = ck
            self.sig = sig
        except Exception:  # pragma: no cover - unexpected Triton internals
            self.ck = None

    def __call__(self, args, sig, stream, stream_obj=None):
        ck = self.ck
        if ck is None or stream is None or sig != self.sig:
            self._slow(args, sig, stream_obj)
            return
        g = self.g
        ck.run(g[0], g[1], g[2], stream, self.fn, self.pm,
               None, None, None, *args)


def _guarded_loader(mod, loader):
    """``weight_loader`` wrapper that drops the owner's transposed-weight cache
    first.  ``param.data.copy_()`` -- which the parallel-linear loaders use --
    does not bump ``param._version``, so the version check alone would keep
    serving a stale copy.  Held weakly so the wrapper does not keep the module
    alive through its own parameter."""
    import weakref
    ref = weakref.ref(mod)

    def _load(param, *args, **kwargs):
        m = ref()
        if m is not None:
            m._wt.clear()
        return loader(param, *args, **kwargs)
    return _load


def _launcher(key, jit, grid, warps, stages):
    l = _PLANS.get(key)
    if l is None:
        l = _Launcher(jit, grid, warps, stages)
        _PLANS[key] = l
    return l


def _scratch(slot: int, shape: tuple, dtype: torch.dtype, device: torch.device):
    key = (slot, device.index, dtype)
    t = _SCRATCH.get(key)
    if t is None or t.shape != shape:
        t = torch.empty(shape, dtype=dtype, device=device)
        _SCRATCH[key] = t
    return t


def _scratch2(slot: int, shape: tuple, shape2: tuple, dtype: torch.dtype,
              device: torch.device):
    """A reused buffer plus a cached flat 2-D view of it (``Tensor.view`` is
    ~0.5 us, which is not negligible at this launch budget)."""
    key = (slot, device.index, dtype)
    e = _SCRATCH.get(key)
    if e is None or e[0].shape != shape:
        t = torch.empty(shape, dtype=dtype, device=device)
        e = (t, t.view(shape2))
        _SCRATCH[key] = e
    return e


def _repack_plan(S: int, H: int, nb: int):
    key = ("rp", S, H, nb)
    p = _PLANS.get(key)
    if p is None:
        blk = CFG["T_BLOCK"]
        bh = max(16, triton.next_power_of_2(H))
        p = ((-(-(S * S) // blk), nb), S * S, blk, bh, bh == H, CFG["T_WARPS"])
        _PLANS[key] = p
    return p


def _launch_repack(bias, packed, S: int, H: int, stream=None,
                   stream_obj=None) -> None:
    grid, MN, blk, bh, even_h, warps = _repack_plan(S, H, bias.shape[0])
    args = (bias, packed,
            bias.stride(0), bias.stride(1), bias.stride(2), bias.stride(3),
            S, MN, H, blk, bh, even_h)
    sig = (args[2:], bias.data_ptr() & 15, packed.data_ptr() & 15)
    _launcher(("r", S, H, bias.shape[0]), _bias_repack_kernel, grid,
              warps, 3)(args, sig, stream, stream_obj)


def _table_plan(R: int, H: int):
    key = ("tp", R, H)
    p = _PLANS.get(key)
    if p is None:
        br = triton.next_power_of_2(R)
        p = ((H,), br, br == R)
        _PLANS[key] = p
    return p


def _launch_table(bucket, w, out, R, stream=None, stream_obj=None) -> None:
    grid, br, even_r = _table_plan(R, out.shape[0])
    args = (bucket, w, out, R, w.stride(0), w.stride(1), br, even_r)
    sig = (args[3:], bucket.data_ptr() & 15, w.data_ptr() & 15,
           out.data_ptr() & 15)
    _launcher(("t", R, out.shape[0]), _bias_table_kernel, grid, 4, 3)(
        args, sig, stream, stream_obj)


def _expand_plan(S: int, H: int):
    key = ("ep", S, H)
    p = _PLANS.get(key)
    if p is None:
        bm, bn = CFG["E_BLOCK_M"], CFG["E_BLOCK_N"]
        p = ((-(-S // bn), -(-S // bm), H), bm, bn,
             S % bm == 0, S % bn == 0, CFG["E_WARPS"])
        _PLANS[key] = p
    return p


def _launch_expand(table, out, S: int, H: int, stream=None,
                   stream_obj=None) -> None:
    grid, bm, bn, em, en, warps = _expand_plan(S, H)
    args = (table, out, S, table.shape[1], bm, bn, em, en)
    sig = (args[2:], table.data_ptr() & 15, out.data_ptr() & 15)
    _launcher(("e", S, H), _bias_expand_kernel, grid, warps, 3)(
        args, sig, stream, stream_obj)


def _flash_plan(B: int, S: int, H: int, D: int):
    key = ("fp", B, S, H, D)
    p = _PLANS.get(key)
    if p is None:
        bm, bn = CFG["BLOCK_M"], CFG["BLOCK_N"]
        nm = -(-S // bm)
        h_first = CFG["H_FIRST"]
        p = ((H, nm, B) if h_first else (nm, H, B),
             H * D, D, bm, bn, S % bm == 0, S % bn == 0,
             CFG["MATCH_BF16"], h_first, CFG["EXP2"],
             CFG["num_warps"], CFG["num_stages"])
        _PLANS[key] = p
    return p


def _launch_flash(qkv, bias, out, H, D, stream=None, stream_obj=None) -> None:
    B, S, _ = qkv.shape
    (grid, hd, d, bm, bn, em, en, mbf, hf, e2,
     warps, stages) = _flash_plan(B, S, H, D)
    sb0 = bias.stride(0) if bias.shape[0] == B else 0
    args = (qkv, bias, out,
            qkv.stride(0), qkv.stride(1),
            sb0, bias.stride(1), bias.stride(2), bias.stride(3),
            out.stride(0), out.stride(1),
            S, hd, d, bm, bn, em, en, mbf, hf, e2)
    sig = (args[3:], qkv.data_ptr() & 15, bias.data_ptr() & 15,
           out.data_ptr() & 15)
    _launcher(("f", B, S, H, D), _t5_flash_kernel, grid, warps, stages)(
        args, sig, stream, stream_obj)


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

        if has_relative_attention_bias:
            self.relative_attention_bias = Embedding(
                self.relative_attention_num_buckets, self.n_heads,
            )

        # (query_length, key_length, device) -> relative-position bucket ids for
        # every rel in [-(q-1), k-1].  A pure function of the sequence lengths
        # and the two config constants, so it is safe to memoize.
        self._bucket_cache: dict[tuple, torch.Tensor] = {}
        # d_kv above 256 would overflow the fp32 accumulator's tensor-memory
        # budget at BLOCK_M=128; fall back to the eager core there.
        self._fast = (_HAS_TRITON and _is_pow2(self.d_kv)
                      and 16 <= self.d_kv <= 256)
        # Going straight to ``torch.mm`` for the two projections skips the
        # module call plus ``F.linear``'s reshape (15.7 -> 7.6 us of host time
        # each), but only reproduces the parallel-linear layers when they are
        # plain unbiased bf16/fp16 matmuls on one rank.
        _hd = self.n_heads_per_partition * self.d_kv
        self._mm_ok = (self.qkv_proj.bias is None and not self.qkv_proj.use_fp8
                       and self.o.bias is None and not self.o.use_fp8
                       and self.o.tp_size == 1
                       and self.qkv_proj.weight.shape[0] == 3 * _hd
                       and self.o.weight.shape[1] == _hd)
        # slot -> (key, transposed weight); see :meth:`_wt_of`.
        self._wt: dict[int, tuple] = {}
        # ``WT_COPY`` caches a *copy* of each projection weight, so every way a
        # weight can be rewritten has to drop it.  ``_version`` catches in-place
        # writes to the parameter (including ``load_state_dict``); the
        # parallel-linear ``weight_loader`` writes through ``param.data``, which
        # does not bump the version, so wrap it.
        try:
            self._register_load_state_dict_pre_hook(self._drop_wt_cache)
        except Exception:  # pragma: no cover - private API moved
            CFG["WT_COPY"] = False
        for _lin in (self.qkv_proj, self.o):
            _wl = getattr(_lin.weight, "weight_loader", None)
            if _wl is not None:
                _lin.weight.weight_loader = _guarded_loader(self, _wl)

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

    def compute_bias(self, query_length: int, key_length: int, device: torch.device) -> torch.Tensor:
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position, bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)
        tp_rank = _tp_rank()
        head_start = tp_rank * self.n_heads_per_partition
        head_end = head_start + self.n_heads_per_partition
        values = values[:, :, head_start:head_end]
        values = values.permute(2, 0, 1).unsqueeze(0)
        return values

    # ------------------------------------------------------------------
    # Fused-path helpers.
    # ------------------------------------------------------------------
    def _bucket_ids(self, query_length: int, key_length: int,
                    device: torch.device) -> torch.Tensor:
        """Bucket id per relative position ``n - m``, indexed by
        ``rel + query_length - 1``.  Same elementwise arithmetic (hence exactly
        the same values) as the 2-D form in :meth:`compute_bias`."""
        key = (query_length, key_length, str(device))
        t = self._bucket_cache.get(key)
        if t is None:
            rel = torch.arange(-(query_length - 1), key_length,
                               dtype=torch.long, device=device)
            t = self._relative_position_bucket(
                rel, bidirectional=True,
                num_buckets=self.relative_attention_num_buckets,
                max_distance=self.relative_attention_max_distance,
            )
            self._bucket_cache[key] = t
        return t

    def _drop_wt_cache(self, *args, **kwargs) -> None:
        self._wt.clear()

    def _emb_slice(self) -> torch.Tensor:
        """This rank's ``[num_buckets, H]`` columns of the bias embedding.
        Cached against the weight's address so the slice is not rebuilt (and
        ``_tp_rank()`` not re-queried) on every call."""
        w = self.relative_attention_bias.emb.weight
        dp = w.data_ptr()
        e = self._wt.get(-1)
        if e is None or e[0] != dp:
            H = self.n_heads_per_partition
            hs = _tp_rank() * H
            e = (dp, w.detach()[:, hs:hs + H])
            self._wt[-1] = e
        return e[1]

    def _bias_table(self, query_length: int, key_length: int,
                    device: torch.device, stream=None,
                    stream_obj=None) -> torch.Tensor:
        """[n_heads_per_partition, query_length + key_length - 1] table with
        ``table[h, rel + q - 1] == compute_bias(...)[0, h, m, n]`` for
        ``rel == n - m``.  Written into a reused scratch buffer (allocating on
        the side stream made the overlap with the qkv GEMM erratic)."""
        bucket = self._bucket_ids(query_length, key_length, device)
        w = self._emb_slice()
        R = query_length + key_length - 1
        table = _scratch(2, (self.n_heads_per_partition, R), w.dtype, device)
        _launch_table(bucket, w, table, R, stream, stream_obj)
        return table

    def _wt_of(self, slot: int, lin) -> torch.Tensor:
        """Cached detached transpose of a projection weight, keyed on the
        weight's address and version so an in-place load invalidates it.

        ``WT_COPY`` (default) materializes a contiguous [K, N] copy, which lets
        cuBLAS pick a faster algorithm for these skinny M=512 shapes -- 3 us off
        the pair, with bit-identical results -- at the cost of one extra copy of
        each projection weight.  Clearing it falls back to a plain transpose
        view: no extra memory, same numerics either way."""
        w = lin.weight
        key = (w.data_ptr(), w._version)
        e = self._wt.get(slot)
        if e is None or e[0] != key:
            t = w.detach().t()
            e = (key, t.contiguous() if CFG["WT_COPY"] else t)
            self._wt[slot] = e
        return e[1]

    def _eager_core(self, qkv, position_bias, batch_size, seq_length):
        """Reference (materialized) attention core -- the fallback path."""
        q_size = self.n_heads_per_partition * self.d_kv
        query_states, key_states, value_states = qkv.split(
            [q_size, q_size, q_size], dim=-1,
        )
        query_states = query_states.view(
            batch_size, seq_length, self.n_heads_per_partition, self.d_kv,
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_length, self.n_heads_per_partition, self.d_kv,
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_length, self.n_heads_per_partition, self.d_kv,
        ).transpose(1, 2)
        scores = self.bmm(query_states, key_states.transpose(3, 2))
        scores = scores + position_bias
        attn_weights = self.softmax(scores.float()).type_as(scores)
        attn_output = self.bmm(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output.view(batch_size, seq_length, -1)

    def _slow_path(self, qkv, mask, position_bias, batch_size, seq_length):
        """Baseline semantics (bias materialized, materialized scores)."""
        if position_bias is None:
            if self.has_relative_attention_bias:
                position_bias = self.compute_bias(
                    seq_length, seq_length, device=qkv.device,
                )
            else:
                position_bias = torch.zeros(
                    (1, self.n_heads_per_partition, seq_length, seq_length),
                    device=qkv.device, dtype=qkv.dtype,
                )
            if mask is not None:
                position_bias = position_bias + mask
        attn_output = self._eager_core(qkv, position_bias, batch_size, seq_length)
        return self.o(attn_output), position_bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = hidden_states.shape
        batch_size, seq_length = shape[0], shape[1]
        H, D = self.n_heads_per_partition, self.d_kv
        dev, dt = hidden_states.device, hidden_states.dtype

        if self._fast and hidden_states.is_cuda and dt in (torch.float16,
                                                           torch.bfloat16):
            # The bias prep (repack, or table build + expand) is independent of
            # the qkv GEMM, so issue it on a side stream and let them overlap.
            bias = None
            prep = None
            if position_bias is None:
                if (mask is None and self.has_relative_attention_bias
                        and self.relative_attention_bias.emb.weight.dtype == dt):
                    position_bias = torch.empty(
                        (1, H, seq_length, seq_length), device=dev, dtype=dt)
                    bias = position_bias
                    prep = 1
            elif (position_bias.dtype == dt and position_bias.dim() == 4
                    and position_bias.shape[1] == H
                    and position_bias.shape[2] == seq_length
                    and position_bias.shape[3] == seq_length
                    and position_bias.shape[0] in (1, batch_size)):
                bias = position_bias
                if bias.stride(-1) != 1:
                    bias = _scratch(
                        1, (bias.shape[0], H, seq_length, seq_length), dt, dev)
                    prep = 2

            if bias is not None:
                # ``torch.mm`` on a detached transposed view needs a contiguous
                # 2-D input, no autograd tape and single-rank unbiased layers;
                # otherwise fall back to the module calls.
                mm = (self._mm_ok and CFG["FAST_GEMM"]
                      and len(shape) == 3 and shape[2] == self.d_model
                      and hidden_states.is_contiguous()
                      and not torch.is_grad_enabled())
                rows = batch_size * seq_length
                stream = None
                if CFG["FAST_LAUNCH"] and _RAW_STREAM is not None:
                    idx = dev.index
                    stream = _RAW_STREAM(
                        torch.cuda.current_device() if idx is None else idx)

                side = sstream = None
                if prep is not None and CFG["OVERLAP"]:
                    side, sstream, ev_in, ev_out, _ = _side(dev)
                    if stream is None:
                        sstream = None
                if side is not None:
                    # The prep only reads inputs, but they may have been
                    # produced on this stream, so order side after it.
                    ev_in.record()
                    ev_in.wait(side)
                    if prep == 1:
                        _launch_expand(
                            self._bias_table(seq_length, seq_length, dev,
                                             sstream, side),
                            bias, seq_length, H, sstream, side)
                    else:
                        _launch_repack(position_bias, bias, seq_length, H,
                                       sstream, side)
                        position_bias.record_stream(side)
                    bias.record_stream(side)

                if mm:
                    qkv, qkv2 = _scratch2(
                        3, (batch_size, seq_length, 3 * H * D),
                        (rows, 3 * H * D), dt, dev)
                    torch.mm(hidden_states.view(rows, self.d_model),
                             self._wt_of(0, self.qkv_proj), out=qkv2)
                else:
                    qkv = self.qkv_proj(hidden_states)

                if side is not None:
                    ev_out.record(side)
                    ev_out.wait()
                elif prep is not None:
                    if prep == 1:
                        _launch_expand(
                            self._bias_table(seq_length, seq_length, dev,
                                             stream),
                            bias, seq_length, H, stream)
                    else:
                        _launch_repack(position_bias, bias, seq_length, H,
                                       stream)

                if qkv.is_contiguous():
                    if mm:
                        out, out2 = _scratch2(
                            0, (batch_size, seq_length, H * D),
                            (rows, H * D), dt, dev)
                        wo = self._wt_of(1, self.o)
                        _launch_flash(qkv, bias, out, H, D, stream)
                        attn = torch.mm(out2, wo)
                        return attn.view(batch_size, seq_length,
                                         self.d_model), position_bias
                    out = _scratch2(0, (batch_size, seq_length, H * D),
                                    (rows, H * D), dt, dev)[0]
                    _launch_flash(qkv, bias, out, H, D, stream)
                    return self.o(out), position_bias
                return self._slow_path(
                    qkv, mask, position_bias, batch_size, seq_length)

        qkv = self.qkv_proj(hidden_states)
        return self._slow_path(qkv, mask, position_bias, batch_size, seq_length)
