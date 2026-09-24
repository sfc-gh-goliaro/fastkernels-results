"""Vision MLP for Qwen vision transformer blocks.

Unified across Qwen2-VL (QuickGELU) and Qwen3-VL (SiLU) activations.

Optimized form of ``fc2(act(fc1(x)))``.  Two layers of optimization, in the
order they matter:

1. **The activation pass is deleted.**  The baseline runs three kernels over a
   ``[M, 1, hidden]`` intermediate (556 MB at M=64680); the standalone exact-erf
   GELU pass alone reads and writes 1.11 GB at only ~3.6 TB/s.  Folding bias +
   GELU into the fc1 epilogue removes that kernel outright.  Both GEMMs are
   already compute bound, so the hidden tensor's own HBM traffic overlaps with
   math and is not worth chasing.

2. **The two remaining GEMMs are dispatched by measurement, not by rule.**
   After (1) the call is exactly two cuBLAS GEMMs, and on this geometry cuBLAS's
   rate is strongly, non-monotonically M-dependent: measured on a B200 at boost
   clocks, ``fc2`` (M x 4304 x 1152) runs at 1575 TF/s at M=20680 but only
   ~1210 TF/s at M=23760 and ~1276 TF/s at M=25168, and both GEMMs collapse to
   ~680 TF/s at M=1760 where the tile grid is 1.6 waves deep.  Which *form* of
   the call is fastest therefore changes with M in a way no closed-form rule
   predicts, so this module keeps a small set of equivalent strategies --
   single-shot, M-chunked (which re-quantizes the tile grid), and a hand-written
   fused GEMM -- and times them once per distinct M on the real tensors, then
   caches the winner.  The single-shot form is the fallback, so the tuner can
   only improve on it.

All strategies are bit-identical to each other on the GELU path: chunking M
splits the output by rows, and every row of a GEMM is an independent reduction.

Other design points:

* The degenerate middle dim is collapsed once.  This is not itself a speedup --
  ``at::linear`` already folds a contiguous ``[M, 1, K]`` input to 2D internally
  -- it is here because ``_addmm_activation`` needs a 2D mat1.
* GELU (exact or tanh): bias + GELU ride along as a cuBLASLt epilogue via
  ``torch._addmm_activation``, so the hidden is written once, already activated.
* QuickGELU / SiLU: cuBLASLt has no epilogue for these, so a hand-written
  persistent warp-specialized TMA Triton GEMM applies bias + activation in its
  own epilogue instead.
* fc2 keeps ``RowParallelLinear`` semantics: the fast path is used only at
  tp_size 1 with a real bias; every other TP configuration falls through to the
  module's own forward so the rank-0-only bias and the all-reduce stay exactly
  as the baseline defines them.
* Every gate falls back to the eager baseline: fp8 / quantized fc1, dtypes
  outside {bfloat16, float16} (fp32's 1e-5 tolerance is tighter than the
  cuBLASLt GELU epilogue's approximation), unrecognized activations,
  non-contiguous or misaligned inputs, and any runtime failure in the fast path.

The activation is identified by *probing* ``self.act_fn`` on a fixed grid rather
than by class name, so a swapped-in candidate L1 activation, or a ``GELU``
carrying a non-default ``approximate=``, is classified on what it computes.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.quickgelu import QuickGELU
from .parallel_linear import ColumnParallelLinear, RowParallelLinear

# --- activation identity codes ------------------------------------------------
_ACT_GELU_ERF = 0     # 0.5x(1+erf(x/sqrt2))      -- Qwen3-VL vision (captured)
_ACT_GELU_TANH = 1    # tanh approximation of the above
_ACT_QUICKGELU = 2    # x*sigmoid(1.702x)         -- Qwen2-VL vision
_ACT_SILU = 3         # x*sigmoid(x)
_ACT_UNKNOWN = -1

# cuBLASLt's GELU epilogue matches either GELU spelling to well inside the
# bf16/fp16 tolerance (their worst-case gap is ~1e-3 on hidden values of O(1),
# and one bf16 ulp there is already 8e-3).
_GELU_CODES = (_ACT_GELU_ERF, _ACT_GELU_TANH)

_PROBE_LO, _PROBE_HI, _PROBE_N = -12.0, 12.0, 3073
# Measured max-abs distances over this grid: a candidate matches its own formula to
# <1e-6, exact vs tanh GELU are 4.7e-4 apart, GELU vs QuickGELU 2.0e-2, and anything
# further (SiLU, ReLU, tanh) >= 1.6e-1.  2e-3 therefore absorbs a differently-spelled
# GELU or QuickGELU implementation while staying an order of magnitude below the
# nearest *wrong* answer.  A residual 2e-3 on the hidden values would land ~2.6e-3 on
# the output against a ~1.5e-2 tolerance bound, so even a borderline match is safe.
_PROBE_TOL = 2e-3

_FAST_DTYPES = (torch.bfloat16, torch.float16)

_HAS_ADDMM_ACT = hasattr(torch, "_addmm_activation")

_DEBUG = bool(__import__("os").environ.get("VISION_MLP_DEBUG"))

# ---------------------------------------------------------------------------
# Hand-written fused GEMM + bias + activation.
#
# Persistent grid of one CTA per SM, host-built TMA descriptors, and
# ``warp_specialize=True`` on the outer tile loop -- the K loop is a pure
# ``acc += tl.dot`` accumulator, which is the only shape Triton 3.6 accepts for
# warp specialization, and it is worth ~1.4x here.  This is the only fc1 path
# for QuickGELU / SiLU (cuBLASLt has no epilogue for either); for GELU it is
# offered to the tuner as one more candidate and adopted only where it wins.
# ---------------------------------------------------------------------------
try:
    import triton
    import triton.language as tl
    from triton.tools.tensor_descriptor import TensorDescriptor

    _TRITON_OK = True
except Exception:  # pragma: no cover - Triton missing: eager path handles it
    _TRITON_OK = False

if _TRITON_OK:

    _SCRATCH: dict[int, torch.Tensor] = {}

    def _tl_alloc(size: int, alignment: int, stream):
        buf = _SCRATCH.get(alignment)
        if buf is None or buf.numel() < size:
            buf = torch.empty(size, dtype=torch.int8, device="cuda")
            _SCRATCH[alignment] = buf
        return buf

    triton.set_allocator(_tl_alloc)

    @triton.jit
    def _apply_act(x, ACT: tl.constexpr):
        if ACT == 0:      # exact erf GELU
            return x * 0.5 * (1.0 + tl.erf(x * 0.70710678118654752))
        elif ACT == 1:    # tanh-approximate GELU
            return x * 0.5 * (1.0 + tl.math.tanh(
                0.7978845608028654 * (x + 0.044715 * x * x * x)))
        elif ACT == 2:    # QuickGELU
            return x * tl.sigmoid(1.702 * x)
        elif ACT == 3:    # SiLU
            return x * tl.sigmoid(x)
        return x          # bias only

    @triton.jit
    def _gemm_bias_act(a_desc, b_desc, c_desc, bias_ptr, M, N, K,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                       BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
                       NUM_SMS: tl.constexpr, ACT: tl.constexpr,
                       WS: tl.constexpr, OUT_DTYPE: tl.constexpr):
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        k_tiles = tl.cdiv(K, BLOCK_K)
        num_tiles = num_pid_m * num_pid_n
        pid_in_group = GROUP_M * num_pid_n
        for tile in tl.range(tl.program_id(0), num_tiles, NUM_SMS,
                             flatten=True, warp_specialize=WS):
            group_id = tile // pid_in_group
            first_m = group_id * GROUP_M
            group_m = min(num_pid_m - first_m, GROUP_M)
            off_m = (first_m + (tile % group_m)) * BLOCK_M
            off_n = ((tile % pid_in_group) // group_m) * BLOCK_N
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for ki in range(k_tiles):
                off_k = ki * BLOCK_K
                a = a_desc.load([off_m, off_k])
                b = b_desc.load([off_n, off_k])
                acc = tl.dot(a, b.T, acc)
            cols = off_n + tl.arange(0, BLOCK_N)
            acc += tl.load(bias_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)[None, :]
            c_desc.store([off_m, off_n], _apply_act(acc, ACT).to(OUT_DTYPE))

    _BLOCK_K, _GROUP_M = 64, 8
    _NUM_WARPS, _NUM_STAGES = 8, 3
    # TMA needs the innermost extent 16B-aligned in both operands and the output.
    _TMA_ELEMS = 8

    def _fused_gemm_act(x2: torch.Tensor, w: torch.Tensor, bias: torch.Tensor,
                        act_code: int, block_m: int = 128, block_n: int = 256):
        """``act(x2 @ w.T + bias)``, or ``None`` if this shape is unsupported."""
        m, k = x2.shape
        n = w.shape[0]
        if (k % _TMA_ELEMS or n % _TMA_ELEMS
                or x2.data_ptr() % 16 or w.data_ptr() % 16
                or not w.is_contiguous()):
            return None
        out = torch.empty((m, n), dtype=x2.dtype, device=x2.device)
        bm = block_m if m >= block_m else max(16, triton.next_power_of_2(m))
        a_desc = TensorDescriptor.from_tensor(x2, [bm, _BLOCK_K])
        b_desc = TensorDescriptor.from_tensor(w, [block_n, _BLOCK_K])
        c_desc = TensorDescriptor.from_tensor(out, [bm, block_n])
        tiles = triton.cdiv(m, bm) * triton.cdiv(n, block_n)
        num_sms = torch.cuda.get_device_properties(x2.device).multi_processor_count
        grid = (min(num_sms, tiles),)
        out_dtype = tl.bfloat16 if x2.dtype is torch.bfloat16 else tl.float16
        _gemm_bias_act[grid](
            a_desc, b_desc, c_desc, bias, m, n, k,
            bm, block_n, _BLOCK_K, _GROUP_M, grid[0], act_code, True,
            out_dtype, num_warps=_NUM_WARPS, num_stages=_NUM_STAGES)
        return out

else:  # pragma: no cover
    _fused_gemm_act = None


def _act_reference(code: int, t: torch.Tensor) -> torch.Tensor:
    if code == _ACT_GELU_ERF:
        return F.gelu(t)
    if code == _ACT_GELU_TANH:
        return F.gelu(t, approximate="tanh")
    if code == _ACT_QUICKGELU:
        return t * torch.sigmoid(1.702 * t)
    return t * torch.sigmoid(t)


def _identify_act(act_fn: Callable[[torch.Tensor], torch.Tensor]) -> int:
    """Classify *act_fn* by what it computes on a fixed fp32 grid."""
    probe = torch.linspace(_PROBE_LO, _PROBE_HI, _PROBE_N, dtype=torch.float32)
    try:
        with torch.no_grad():
            got = act_fn(probe)
    except Exception:
        return _ACT_UNKNOWN
    if not isinstance(got, torch.Tensor) or got.shape != probe.shape:
        return _ACT_UNKNOWN
    got = got.float()
    best, best_err = _ACT_UNKNOWN, float("inf")
    for code in (_ACT_GELU_ERF, _ACT_GELU_TANH, _ACT_QUICKGELU, _ACT_SILU):
        err = (got - _act_reference(code, probe)).abs().max().item()
        if err < best_err:
            best, best_err = code, err
    return best if best_err <= _PROBE_TOL else _ACT_UNKNOWN


# ---------------------------------------------------------------------------
# Strategy set.  Each entry is a callable ``(state, x2) -> y2`` computing the
# whole ``fc2(act(fc1(x2)))`` on 2D tensors.  The axes are: how many row chunks
# to split M into (which re-quantizes the tile grid cuBLAS chooses against) and
# whether each weight is handed over as the transposed view of the stored
# ``[N, K]`` or as a row-major ``[K, N]`` copy.  Neither changes a single output
# value -- every row of a GEMM is an independent reduction, and both layouts
# hold the same data -- so the tuner is free to pick on speed alone.
# ---------------------------------------------------------------------------
# Distinct M values whose plan is cached before the tuner stops offering to
# re-tune; keeps a workload with a long tail of shapes from tuning forever.
_MAX_TUNED_SHAPES = 24
# Chunk counts offered to the tuner. 1 == single-shot, always present as the
# floor, so tuning can only pick something at least as fast.
_FC2_CHUNKS = (1, 2, 3, 4)
_BOTH_CHUNKS = (2, 3, 4, 6)


def _split(m: int, nc: int) -> list[tuple[int, int]]:
    """Row ranges for *nc* chunks, each start 8-row aligned (16B for bf16)."""
    edges = [min(m, (m * i // nc + 7) // 8 * 8) for i in range(nc)]
    edges.append(m)
    out = []
    for i in range(nc):
        if edges[i + 1] > edges[i]:
            out.append((edges[i], edges[i + 1]))
    return out


class _State:
    """Weights and shapes the strategies need, resolved once per forward.

    ``w1t`` / ``w2t`` are the transposed *views* of the stored ``[N, K]``
    weights, i.e. column-major ``[K, N]`` operands (a "TN" GEMM).  ``w1nn`` /
    ``w2nn`` are row-major ``[K, N]`` copies of the same data, which is a
    different cuBLAS problem and so can draw a different kernel from the
    heuristic; they are built on demand and cached on the module, keyed on the
    weight's identity and version so a later ``load_state_dict`` invalidates
    them.
    """

    __slots__ = ("w1t", "b1", "w2t", "b2", "act", "hid", "out", "dtype", "dev",
                 "_nn", "_fc1w", "_fc2w")

    def __init__(self, fc1, fc2, act, nn_cache):
        self.w1t = fc1.weight.t()
        self.b1 = fc1.bias
        self.w2t = fc2.weight.t()
        self.b2 = fc2.bias
        self.act = act
        self.hid = fc1.weight.shape[0]
        self.out = fc2.weight.shape[0]
        self.dtype = fc1.weight.dtype
        self.dev = fc1.weight.device
        self._nn = nn_cache
        self._fc1w = fc1.weight
        self._fc2w = fc2.weight

    def _nn_of(self, slot: str, w: torch.Tensor) -> torch.Tensor:
        key = (w.data_ptr(), w._version, w.shape)
        got = self._nn.get(slot)
        if got is None or got[0] != key:
            got = (key, w.t().contiguous())
            self._nn[slot] = got
        return got[1]

    @property
    def w1nn(self) -> torch.Tensor:
        return self._nn_of("w1", self._fc1w)

    @property
    def w2nn(self) -> torch.Tensor:
        return self._nn_of("w2", self._fc2w)


def _fc1(st: _State, x2: torch.Tensor, w=None) -> torch.Tensor:
    if st.act in _GELU_CODES:
        return torch._addmm_activation(st.b1, x2, st.w1t if w is None else w,
                                       use_gelu=True)
    return _fused_gemm_act(x2, st.w1t.t(), st.b1, st.act)


def _strategy(nc: int, fc2_nc: int, nn1: bool, nn2: bool):
    """One point in the strategy space.

    * ``nc``      -- split the whole pipeline into this many row chunks, so fc1
                     and fc2 alternate over sub-ranges of M.
    * ``fc2_nc``  -- with ``nc == 1``, split only fc2's M into this many chunks.
    * ``nn1/nn2`` -- feed the GEMM a row-major ``[K, N]`` copy of the weight
                     instead of the transposed view of the stored ``[N, K]``.

    Every point computes the same values: chunking M partitions the output by
    rows and each row of a GEMM is an independent reduction, and the two weight
    layouts hold identical data.  Splitting M or changing the operand layout
    only changes which cuBLAS kernel the heuristic picks and how its tile grid
    quantizes against the 148 SMs -- which on this geometry is worth up to 11%
    (M=1760, both weights row-major) and 10% (M=23760, fc2 split in two).
    """

    def run(st: _State, x2: torch.Tensor) -> torch.Tensor:
        m = x2.shape[0]
        w1 = st.w1nn if nn1 else None
        w2 = st.w2nn if nn2 else st.w2t
        if nc == 1 and fc2_nc == 1:
            return torch.addmm(st.b2, _fc1(st, x2, w1), w2)
        y = torch.empty((m, st.out), dtype=st.dtype, device=st.dev)
        if nc > 1:
            for a, b in _split(m, nc):
                torch.addmm(st.b2, _fc1(st, x2[a:b], w1), w2, out=y[a:b])
        else:
            h = _fc1(st, x2, w1)
            for a, b in _split(m, fc2_nc):
                torch.addmm(st.b2, h[a:b], w2, out=y[a:b])
        return y

    return run


def _strategy_triton_fc1(block_m: int, block_n: int):
    """Hand-written fused fc1 (bias + activation in the epilogue), plain fc2."""

    def run(st: _State, x2: torch.Tensor) -> torch.Tensor:
        h = _fused_gemm_act(x2, st.w1t.t(), st.b1, st.act, block_m, block_n)
        if h is None:
            raise RuntimeError("triton fc1 unsupported for this shape")
        return torch.addmm(st.b2, h, st.w2t)

    return run


def _build_strategies(act: int) -> list[tuple[str, Callable]]:
    """The single-shot all-transposed form is first; it is the tuner's floor."""
    gelu = act in _GELU_CODES
    # Row-major weight copies only reach fc1 through _addmm_activation, so the
    # nn1 axis is meaningless on the Triton (QuickGELU / SiLU) fc1 path.
    layouts = ((False, False), (True, True), (False, True), (True, False)) if gelu \
        else ((False, False), (False, True))
    out: list[tuple[str, Callable]] = []
    for nn1, nn2 in layouts:
        for fc2_nc in _FC2_CHUNKS:
            tag = f"{'N' if nn1 else 'T'}{'N' if nn2 else 'T'}"
            out.append((f"{tag}f{fc2_nc}", _strategy(1, fc2_nc, nn1, nn2)))
    for nc in _BOTH_CHUNKS:
        out.append((f"c{nc}", _strategy(nc, 1, False, False)))
    if gelu and _fused_gemm_act is not None:
        # Offered so the own-GEMM path is rejected on measurement rather than
        # assumption; it has yet to win at any benched shape.
        out.append(("t128x256", _strategy_triton_fc1(128, 256)))
        out.append(("t128x128", _strategy_triton_fc1(128, 128)))
    return out


class _Tuner:
    """Times the strategy set once per distinct M and caches the winner.

    Mirrors the benchmark's own timing conditions -- L2 flushed before each
    call, CUDA-event latency -- so the ranking reflects how the shape will
    actually be measured rather than a warm-cache best case.

    Two details of the measurement matter as much as what is measured:

    * The strategies are timed **round-robin** -- a burst each, repeated for
      several passes, keeping each strategy's best burst.  Timing each strategy
      to completion in turn instead lets clock and power drift over the tuning
      loop masquerade as a difference between strategies, which at this
      granularity is worth as much as the differences being measured (~5%).
    * Each burst is several calls enqueued **back to back with no host sync**,
      and the fastest is kept.  Synchronizing after every call instead leaves the
      GPU idle between them so every launch pays full latency, which is not the
      regime the benchmark measures in -- and it reorders the ranking: with a
      per-call sync, splitting fc2's M looks like a 6% loss at M=64680, while
      back to back it is a 4% win.
    """

    _WARMUP_PASSES, _PASSES, _BURST = 1, 3, 4
    # A strategy has to beat the single-shot form by more than the tuner's own
    # residual spread before it is adopted, so ranking noise cannot turn into a
    # regression.  The one reproducible win on this geometry (splitting fc2's M
    # in two at M=23760) is ~9%, comfortably clear of this bar.
    _MARGIN = 0.975

    def __init__(self):
        self.plan: dict[int, Callable] = {}
        self._flush: torch.Tensor | None = None

    def _flush_buf(self, dev) -> torch.Tensor:
        if self._flush is None:
            try:
                l2 = torch.cuda.get_device_properties(dev).L2_cache_size
            except Exception:
                l2 = 64 << 20
            self._flush = torch.empty(int(2 * l2), dtype=torch.int8, device=dev)
        return self._flush

    def select(self, strategies, st: _State, x2: torch.Tensor) -> Callable:
        m = x2.shape[0]
        got = self.plan.get(m)
        if got is not None:
            return got
        name0, default = strategies[0]
        if len(self.plan) >= _MAX_TUNED_SHAPES:
            return default
        flush = self._flush_buf(x2.device)
        live = list(strategies)
        best: dict[str, float] = {}
        ev = [(torch.cuda.Event(enable_timing=True),
               torch.cuda.Event(enable_timing=True)) for _ in range(self._BURST)]
        for p in range(self._WARMUP_PASSES + self._PASSES):
            timed = p >= self._WARMUP_PASSES
            survivors = []
            for name, fn in live:
                try:
                    for j in range(self._BURST if timed else 1):
                        flush.zero_()
                        if timed:
                            ev[j][0].record()
                        fn(st, x2)
                        if timed:
                            ev[j][1].record()
                    if timed:
                        torch.cuda.synchronize()
                        ms = min(a.elapsed_time(b) for a, b in ev)
                        best[name] = min(best.get(name, float("inf")), ms)
                except Exception:
                    continue
                survivors.append((name, fn))
            live = survivors
        by_name = dict(live)
        if not best:
            pick, why = default, "none-timed"
        else:
            win = min(best, key=best.get)
            base = best.get(name0)
            if base is not None and best[win] >= base * self._MARGIN:
                win = name0
            pick, why = by_name.get(win, default), win
        if _DEBUG:
            order = " ".join(f"{n}:{v * 1e3:.0f}" for n, v in
                             sorted(best.items(), key=lambda kv: kv[1]))
            print(f"[vision_mlp] M={m} -> {why}   {order}", flush=True)
        self.plan[m] = pick
        return pick


class VisionMLP(nn.Module):
    """Vision encoder MLP with configurable activation.

    Qwen2-VL uses QuickGELU (default); Qwen3-VL uses F.silu.
    """

    def __init__(self, in_features: int, hidden_features: int,
                 act_fn: Callable[[torch.Tensor], torch.Tensor] = QuickGELU(),
                 bias: bool = True):
        super().__init__()
        self.fc1 = ColumnParallelLinear(in_features, hidden_features, bias=bias)
        self.fc2 = RowParallelLinear(hidden_features, in_features, bias=bias)
        self.act_fn = act_fn
        self._act_code = _identify_act(act_fn)
        self._strategies = _build_strategies(self._act_code)
        self._tuner = _Tuner()
        self._nn_cache: dict[str, tuple] = {}
        # Set once a fast path proves unusable for this module, so a failure is
        # paid at most once instead of on every call.
        self._fast_off = False

    def _usable(self, x: torch.Tensor) -> bool:
        fc1, fc2 = self.fc1, self.fc2
        if self._fast_off or self._act_code == _ACT_UNKNOWN:
            return False
        if getattr(fc1, "use_fp8", False) or getattr(fc2, "use_fp8", False):
            return False
        if fc1.bias is None or fc2.bias is None:
            return False
        # RowParallelLinear applies its bias on rank 0 only and all-reduces
        # afterwards; leave anything but the single-rank case to the module.
        if getattr(fc2, "tp_size", 1) != 1:
            return False
        if self._act_code not in _GELU_CODES and _fused_gemm_act is None:
            return False
        if self._act_code in _GELU_CODES and not _HAS_ADDMM_ACT:
            return False
        w = fc1.weight
        if (x.dtype not in _FAST_DTYPES or x.dtype is not w.dtype
                or x.dtype is not fc2.weight.dtype
                or not x.is_cuda or x.dim() < 2 or not x.is_contiguous()):
            return False
        return x.shape[-1] == w.shape[1]

    def _fast(self, x: torch.Tensor) -> torch.Tensor | None:
        if not self._usable(x):
            return None
        k = self.fc1.weight.shape[1]
        x2 = x.view(-1, k)
        if x2.shape[0] == 0:
            return None
        try:
            st = _State(self.fc1, self.fc2, self._act_code, self._nn_cache)
            fn = self._tuner.select(self._strategies, st, x2)
            y2 = fn(st, x2)
        except Exception:
            self._fast_off = True
            return None
        return y2.view(*x.shape[:-1], self.fc2.weight.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._fast(x)
        if y is None:
            y = self.fc2(self.act_fn(self.fc1(x)))
        return y
