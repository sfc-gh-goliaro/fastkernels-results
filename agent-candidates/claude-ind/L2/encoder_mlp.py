"""Feed-forward blocks for encoder models.

The two halves of the encoder MLP are fused Triton kernels instead of the
baseline's ``matmul -> activation`` / ``matmul -> add -> LayerNorm`` chains:

* ``EncoderIntermediate`` -- one tcgen05 GEMM whose epilogue applies bias and
  GELU, so the 4096-wide activation never round-trips through HBM.
* ``EncoderOutput`` -- one kernel: a (K-split) GEMM, a grid-wide barrier, then a
  cooperative row-wise LayerNorm over the 1024-wide result.  The barrier is
  safe because the launch grid is sized to at most one CTA per SM, so every CTA
  is resident; a shape whose grid would exceed that takes the baseline path.

Four things dominate the measured latency here, in order:

1. **Kernel count.**  A launch costs a few us of GPU time and ~13 us of CPU
   dispatch, and both operators are small enough for that to matter, hence the
   fusions above.
2. **CPU dispatch.**  Triton's ``kernel[grid](...)`` binder costs ~20 us per
   launch, which on the M=64 cases is the whole runtime.  Kernels are compiled
   once and then launched through the cached ``CompiledKernel`` runner (~13 us),
   with scratch buffers and TMA descriptors built as cheaply as possible.
3. **Tile shape.**  Tiles are chosen so the grid lands in a single wave of CTAs
   (148 SMs on B200) with the largest tiles that still fill it -- K-splitting the
   1024-wide EncoderOutput GEMM to get there.
4. **Epilogue cost.**  ``tl.erf`` lowers to libdevice's ~140-instruction
   ``erff``; on a 2048x4096 activation that costs more than the GEMM it is fused
   into.  The tanh form via ``tanh.approx.f32`` (one MUFU instruction) is within
   5e-4 absolute of exact GELU -- far inside the fp16 tolerance.

The GEMM operands are read through host-built TMA descriptors: building them on
device (``tl.make_tensor_descriptor``) costs ~2 us of GPU time per descriptor
per launch, while a host build is CPU work that overlaps the GPU.  Epilogue
stores use plain vector stores -- a fourth descriptor is not worth its ~2.5 us
of dispatch.

Module structure, parameter names and dtypes match the baseline.  The fused
path covers fp16 2-D inputs whose tile grid fits the configured wave; anything
else (other dtypes, shapes the tuned configs do not divide, a grid larger than
the SM count) falls back to the baseline ops rather than to an untuned kernel.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from ..L1.gelu import GELU
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
@triton.jit
def _gelu(x):
    """tanh-form GELU.  ``tanh.approx.f32`` is a single MUFU instruction; the
    portable spellings (``tl.erf``, ``tl.sigmoid``, libdevice ``tanh``) each cost
    tens of instructions per element."""
    z = 0.7978845608028654 * x * (1.0 + 0.044715 * x * x)
    t = tl.inline_asm_elementwise("tanh.approx.f32 $0, $1;", "=r,r", [z],
                                  dtype=tl.float32, is_pure=True, pack=1)
    return x * 0.5 * (1.0 + t)


@triton.jit
def _gemm_gelu_kernel(a_desc, b_desc, C, Bias,
                      M, N: tl.constexpr, K: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                      GROUP_M: tl.constexpr, NSTAGES: tl.constexpr,
                      WS: tl.constexpr):
    """C = gelu(A @ B^T + bias); A[M,K], B[N,K], C[M,N], all row-major."""
    pid = tl.program_id(0)
    num_pid_n: tl.constexpr = N // BN
    num_pid_m = tl.cdiv(M, BM)
    # group-major (L2-friendly) tile order
    group = GROUP_M * num_pid_n
    first_m = (pid // group) * GROUP_M
    size_m = min(num_pid_m - first_m, GROUP_M)
    om = (first_m + ((pid % group) % size_m)) * BM
    on = ((pid % group) // size_m) * BN

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in tl.range(0, K, BK, num_stages=NSTAGES, warp_specialize=WS):
        acc = tl.dot(a_desc.load([om, k]), b_desc.load([on, k]).T, acc)
    acc += tl.load(Bias + on + tl.arange(0, BN)).to(tl.float32)[None, :]
    rows = om + tl.arange(0, BM)
    tl.store(C + rows[:, None] * N + (on + tl.arange(0, BN))[None, :],
             _gelu(acc).to(C.dtype.element_ty), mask=rows[:, None] < M)


@triton.jit
def _gemm_ln_kernel(a_desc, b_desc, P, R, Bias, Y, LnW, LnB, Counter,
                    M, eps, N: tl.constexpr, K: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                    SPLIT_K: tl.constexpr, NSTAGES: tl.constexpr,
                    WS: tl.constexpr, LN_ROWS: tl.constexpr,
                    NCTAS: tl.constexpr):
    """Y = LayerNorm(A @ B^T + bias + R) in one launch.

    Phase 1: CTA ``pid`` accumulates the (BM, BN) tile of the K-slice ``pid %
    SPLIT_K`` and writes it to the partial buffer ``P[sk]``.
    Phase 2: after a grid-wide barrier, CTA ``pid`` reduces / normalizes its own
    contiguous slice of rows.  Both phases use every CTA, so the LayerNorm is as
    parallel as a standalone kernel would be, minus the launch.
    """
    pid = tl.program_id(0)
    num_pid_n: tl.constexpr = N // BN
    sk = pid % SPLIT_K
    tile = pid // SPLIT_K
    om = (tile // num_pid_n) * BM
    on = (tile % num_pid_n) * BN
    KS: tl.constexpr = K // SPLIT_K

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in tl.range(sk * KS, sk * KS + KS, BK,
                      num_stages=NSTAGES, warp_specialize=WS):
        acc = tl.dot(a_desc.load([om, k]), b_desc.load([on, k]).T, acc)
    rows = om + tl.arange(0, BM)
    tl.store(P + (sk * M + rows)[:, None] * N + (on + tl.arange(0, BN))[None, :],
             acc.to(P.dtype.element_ty), mask=rows[:, None] < M,
             cache_modifier=".cg")

    # --- grid-wide barrier -------------------------------------------------
    # One arrival per CTA (a 1-element atomic runs on a single lane; the sum
    # broadcasts its result to the block), release-ordered after the partial
    # stores above, then spin on a volatile read and close with an acquire.
    #
    # The counter is never cleared: every launch on this counter adds exactly
    # NCTAS, so ``arrival // NCTAS`` is this launch's generation and the target
    # follows from it.  That keeps the barrier self-contained -- no host-side
    # bookkeeping to fall out of sync, and replaying the same launch (e.g. from
    # a CUDA graph) stays correct.
    tl.debug_barrier()
    arrival = tl.sum(tl.atomic_add(Counter + tl.arange(0, 1), 1,
                                   sem="release", scope="gpu"))
    target = (arrival // NCTAS + 1) * NCTAS
    while tl.load(Counter, volatile=True) < target:
        pass
    tl.atomic_add(Counter + tl.arange(0, 1), 0, sem="acquire", scope="gpu")
    tl.debug_barrier()

    # --- cooperative LayerNorm --------------------------------------------
    RPC: tl.constexpr = (M + NCTAS - 1) // NCTAS
    cols = tl.arange(0, N)
    w = tl.load(LnW + cols).to(tl.float32)
    b = tl.load(LnB + cols).to(tl.float32)
    bias = tl.load(Bias + cols).to(tl.float32)
    for r0 in tl.range(0, RPC, LN_ROWS):
        rr = pid * RPC + r0 + tl.arange(0, LN_ROWS)
        m2 = rr[:, None] < M
        off = rr[:, None] * N + cols[None, :]
        x = tl.load(P + off, mask=m2, other=0.0, cache_modifier=".cg").to(tl.float32)
        for s in tl.static_range(1, SPLIT_K):
            x += tl.load(P + s * M * N + off, mask=m2, other=0.0,
                         cache_modifier=".cg").to(tl.float32)
        x += bias[None, :] + tl.load(R + off, mask=m2, other=0.0).to(tl.float32)
        mean = tl.sum(x, 1) * (1.0 / N)
        xc = x - mean[:, None]
        var = tl.sum(xc * xc, 1) * (1.0 / N)
        rstd = tl.rsqrt(var + eps)
        tl.store(Y + off, (xc * rstd[:, None] * w[None, :] + b[None, :]).to(
            Y.dtype.element_ty), mask=m2)


# ---------------------------------------------------------------------------
# Launch configuration, tuned on the captured shapes (see docs/shapes.md).
# ---------------------------------------------------------------------------
# EncoderIntermediate GEMM: (BM, BN, BK, GROUP_M, NSTAGES, WARP_SPEC, num_warps)
_GELU_CFG = {
    64: (64, 32, 256, 8, 4, False, 4),
    512: (64, 128, 64, 8, 4, False, 8),
    2048: (256, 256, 64, 8, 3, False, 8),
}
# Fused EncoderOutput: (BM, BN, BK, SPLIT_K, NSTAGES, WARP_SPEC, num_warps, LN_ROWS)
_FUSED_CFG = {
    64: (64, 32, 256, 4, 4, False, 4, 1),
    512: (128, 128, 64, 4, 4, False, 8, 4),
    2048: (128, 256, 64, 2, 4, False, 8, 16),
}


# ---------------------------------------------------------------------------
# Launch helpers
# ---------------------------------------------------------------------------
_RUNNERS: dict = {}       # launch key -> cached CompiledKernel runner
_COUNTER: dict = {}       # (device, grid) -> barrier arrival counter
_NSM: dict = {}           # device -> SM count


try:
    _raw_stream = torch._C._cuda_getCurrentRawStream
except AttributeError:                                    # pragma: no cover
    def _raw_stream(index):
        return torch.cuda.current_stream(index).cuda_stream


def _weight_desc(cache: dict, w: torch.Tensor, block):
    """Cached TMA descriptor for a weight.

    Weights do not move between forwards, and building a descriptor costs ~1 us
    of the ~15 us CPU dispatch budget, so cache per (address, block shape) on the
    owning module.  A reload via ``load_state_dict`` writes in place (same
    address); a ``.to()`` that reallocates changes it and misses the cache.
    """
    key = (w.data_ptr(), block[0], block[1])
    d = cache.get(key)
    if d is None:
        if len(cache) > 4:      # weight moved (or many block shapes): start over
            cache.clear()
        d = TensorDescriptor(w, list(w.shape), [w.shape[1], 1], list(block))
        cache[key] = d
    return d


def _sm_count(t: torch.Tensor) -> int:
    idx = t.device.index
    n = _NSM.get(idx)
    if n is None:
        n = torch.cuda.get_device_properties(t.device).multi_processor_count
        _NSM[idx] = n
    return n


def _runner(kernel, key, args, grid, num_warps):
    """Compile *kernel* once for this key and return its cheap launcher.

    ``kernel[grid](...)`` re-does signature binding and specialization on every
    call (~20 us); the ``CompiledKernel`` runner skips all of it (~13 us).  The
    key carries every value the specialization depends on, so a cached runner is
    only ever reused for an identical launch signature.
    """
    r = _RUNNERS.get(key)
    if r is None:
        ck = kernel.warmup(*args, grid=(grid, 1, 1), num_warps=num_warps)
        ck._init_handles()
        r = ck[(grid, 1, 1)]
        _RUNNERS[key] = r
    return r


def _counter(device, grid: int):
    """One barrier counter per (device, grid); see the barrier in the kernel."""
    key = (device.index, grid)
    c = _COUNTER.get(key)
    if c is None:
        c = torch.zeros(1, dtype=torch.int64, device=device)
        _COUNTER[key] = c
    return c


def _bucket(M: int) -> int:
    if M <= 64:
        return 64
    if M <= 512:
        return 512
    return 2048


def _gemm_gelu(x: torch.Tensor, wdesc, bias: torch.Tensor, N: int, K: int):
    M = x.shape[0]
    BM, BN, BK, GM, NS, WS, NW = _GELU_CFG[_bucket(M)]
    if N % BN or K % BK or wdesc.block_shape != [BN, BK]:
        return None
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = -(-M // BM) * (N // BN)
    args = (TensorDescriptor(x, [M, K], [K, 1], [BM, BK]), wdesc,
            out, bias, M, N, K, BM, BN, BK, GM, NS, WS)
    key = ("gelu", M, N, K, BM, BN, BK, GM, NS, WS, NW)
    r = _RUNNERS.get(key) or _runner(_gemm_gelu_kernel, key, args, grid, NW)
    r(*args, stream=_raw_stream(x.device.index))
    return out


def _out_block(x, wdesc, bias, res, ln_w, ln_b, eps, N: int, K: int):
    M = x.shape[0]
    BM, BN, BK, SK, NS, WS, NW, LR = _FUSED_CFG[_bucket(M)]
    if N % BN or K % (BK * SK) or wdesc.block_shape != [BN, BK]:
        return None
    grid = -(-M // BM) * (N // BN) * SK
    if grid > _sm_count(x):
        return None
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    part = torch.empty((SK, M, N), device=x.device, dtype=x.dtype)
    args = (TensorDescriptor(x, [M, K], [K, 1], [BM, BK]), wdesc,
            part, res, bias, out, ln_w, ln_b, _counter(x.device, grid), M, eps,
            N, K, BM, BN, BK, SK, NS, WS, LR, grid)
    key = ("fused", M, N, K, BM, BN, BK, SK, NS, WS, NW, LR, grid)
    r = _RUNNERS.get(key) or _runner(_gemm_ln_kernel, key, args, grid, NW)
    r(*args, stream=_raw_stream(x.device.index))
    return out


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------
def _fusable(x: torch.Tensor, w: torch.Tensor, b) -> bool:
    return (x.dtype is torch.float16 and x.dim() == 2 and x.is_contiguous()
            and w.dtype is torch.float16 and w.is_contiguous()
            and b is not None and b.dtype is torch.float16
            and x.is_cuda and x.shape[1] == w.shape[1])


class EncoderIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.intermediate_act_fn = GELU()
        self._wdesc: dict = {}

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        w, b = self.dense.weight, self.dense.bias
        if _fusable(hidden_states, w, b):
            N, K = w.shape
            block = _GELU_CFG[_bucket(hidden_states.shape[0])][1:3]
            out = _gemm_gelu(hidden_states, _weight_desc(self._wdesc, w, block),
                             b, N, K)
            if out is not None:
                return out
        return self.intermediate_act_fn(self.dense(hidden_states))


class EncoderOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.intermediate_size, config.hidden_size, bias=True)
        # promote_fp32=False: vLLM's bert.py / roberta.py use a plain
        # nn.LayerNorm here (see encoder_embeddings for the full rationale).
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps,
                                   promote_fp32=False)
        self._wdesc: dict = {}

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        w, b = self.dense.weight, self.dense.bias
        ln = self.LayerNorm
        lw, lb = ln.weight, ln.bias
        if (_fusable(hidden_states, w, b) and lw is not None and lb is not None
                and lw.dtype is torch.float16 and lb.dtype is torch.float16
                and input_tensor.dtype is torch.float16
                and input_tensor.is_contiguous()
                and input_tensor.shape == (hidden_states.shape[0], w.shape[0])):
            N, K = w.shape
            block = _FUSED_CFG[_bucket(hidden_states.shape[0])][1:3]
            out = _out_block(hidden_states, _weight_desc(self._wdesc, w, block),
                             b, input_tensor, lw, lb, ln.eps, N, K)
            if out is not None:
                return out
        return self.LayerNorm(self.dense(hidden_states) + input_tensor)
