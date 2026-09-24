"""TP-aware embedding and LM head (L2 operators).

At the benchmarked configuration (single GPU, ``_tp_size() == 1``) this module
reduces to two things: ``VocabParallelEmbedding`` is the frozen L1 row-gather,
and ``ParallelLMHead.project`` is one ``[M, K] x [N, K]^T`` bf16 projection
against the full vocab table.  The vocab mask, the ``mask * y`` zeroing, the
all-reduce and the three ``gather_*`` helpers are all TP-only code paths that
never execute here; they are kept verbatim so the class contracts hold.

Hand-written kernel: ``fk_lmhead_gemv``
--------------------------------------
The captured LM-head shapes split cleanly by arithmetic intensity.  With
``N ~ 1.5e5`` and ``K ~ 2e3`` the weight table is 0.6-0.75 GB, so
``2*M*K*N / (K*N*2) = M`` flops per byte puts the crossover against B200's
~190 flop/byte ratio at ``M ~ 190``: the captured ``M = 16384`` and ``M = 494``
cases are compute-bound (cuBLAS gets 1.37-1.42 PFLOP/s on them, and the
Blackwell-native ``nvjet`` kernels it picks are not worth attacking), while
``M <= 88`` is a pure weight-bandwidth problem -- the kernel's whole job is to
read the table once, as fast as the HBM allows.

Measured on B200 with the benchmark's own timing loop (median CUDA-event latency
with a 2xL2 flush before every call, which costs a fixed ~8 us and ~126 MB of
dirty-L2 writeback that both candidate and baseline pay):

* A flat ``uint4`` streaming read of a 593 MB buffer -- no arithmetic, no store,
  just the fastest read this GPU can do at this kernel duration -- takes
  0.1014 ms (6.13 TB/s).  True peak is ~7.35 TB/s but only for multi-GB reads;
  a ~100 us kernel cannot amortise the DRAM ramp.
* cuBLAS' GEMV for ``M=1, K=2048, N=151936`` takes 0.1199 ms (5.19 TB/s).

So the real headroom on the tiny-M shapes is ~1.18x, and it is all in the load
path.  Three things get this kernel to 0.1056 ms (1.135x, i.e. ~96% of the
flat-read floor):

1. **One warp streams whole weight rows.**  Lane ``l`` reads bytes
   ``l*32 .. l*32+31`` of row ``n``, so a warp issues one 1 KB fully contiguous
   request per step and walks the row to its end before moving on.  Tiled
   ``[BN, BK]`` addressing (what a Triton GEMM emits) reads only ``BK*2`` bytes
   out of each 2*K-byte row before jumping to the next row, and a read-only
   probe with that access pattern tops out at 5.1-5.6 TB/s no matter how the
   tile is shaped -- the scattered instantaneous footprint is what costs the
   bandwidth, not the MMA.
2. **32-byte loads** (``ld.global.nc.v4.b64``, sm_100+).  Worth ~1% over 16 B
   ``uint4`` loads here, and it halves the instruction count per byte.
3. **``L2::evict_first`` on every weight load.**  The table is streamed exactly
   once, so allocating L2 lines for it only evicts the flush buffer's dirty
   lines into the critical path.  This single hint is worth ~10% and dominates
   every other knob (grid, block size, rows per warp, unroll) -- without it the
   same kernel measures 0.115-0.117 ms.  sm_100 accepts the inline modifier only
   on 32 B+ load forms ("requires '.v8.b32/.v4.b64' type"), which is a second
   reason for the 32 B load above; the pre-Blackwell fallback has to express the
   same hint through ``createpolicy.fractional`` + ``.L2::cache_hint``.

Accumulation is fp32 (matching cuBLAS) and the dot products are plain FFMA, no
tensor cores -- which is also why the gate stops at ``M == 1``.  Measured on the
two captured vocab tables, this kernel is 1.14x / 1.08x at M=1 but already
0.91x / 0.82x at M=2 and 0.45x / 0.41x at M=8: each extra row of x costs another
16 FFMA plus 8 bf16->fp32 conversions per lane per 32 B of weight, and that
non-tensor arithmetic stops hiding under the weight stream almost immediately.
Above M=1 the projection needs tensor cores, and there the tiled weight read
cannot reach the flat-read bandwidth (see the ITERATIONS.md roofline note), so
everything else defers.

``ParallelLMHead.project`` dispatches to this kernel only for
``(M, K, N)`` combinations measured faster than the reference on this GPU, and
falls back to the L1 ``Matmul`` (cuBLAS for bf16) otherwise -- the same
measure-then-gate rule ``L1/linear.py`` uses.  ``M`` in the 16..128 range is
deliberately *not* claimed: it needs tensor cores (FFMA is ~4x too slow by
M=60), and every Triton tiled-MMA variant tried there landed at 0.98-1.02x
because a tiled weight read cannot reach the flat-read bandwidth while the
A-operand's share of the shared-memory budget caps how many weight bytes stay
in flight.  See ITERATIONS.md for the full table.
"""

from __future__ import annotations

import hashlib
import os

import torch
import torch.distributed as dist
import torch.nn as nn

from ....infra.context import get_context
from ....infra.tp import _tp_size, _tp_rank
from ..L1.linear import Matmul
from ..L1.embedding import Embedding
from ..L1.allreduce import AllReduce

# ---------------------------------------------------------------------------
# Tiny-M vocab projection: y[m, n] = sum_k x[m, k] * w[n, k]
# ---------------------------------------------------------------------------
_CUDA = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <algorithm>

namespace {

struct alignas(32) v32 { uint4 a, b; };

// 32 B streaming load of read-only weight data, tagged so its L2 lines are the
// first evicted (the table is read exactly once per launch).  sm_100 only
// accepts the eviction-priority modifier on >=32 B forms, hence the policy
// register rather than a plain `.L2::evict_first` on a v4.b32 load.
__device__ __forceinline__ v32 ld_w(const v32* p) {
#if __CUDA_ARCH__ >= 1000
  union { v32 v; unsigned long long u[4]; } r;
  asm volatile("ld.global.nc.L2::evict_first.v4.b64 {%0,%1,%2,%3}, [%4];"
               : "=l"(r.u[0]), "=l"(r.u[1]), "=l"(r.u[2]), "=l"(r.u[3])
               : "l"(p));
  return r.v;
#elif __CUDA_ARCH__ >= 800
  v32 v;
  unsigned long long pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  asm volatile("ld.global.nc.L2::cache_hint.v4.b32 {%0,%1,%2,%3}, [%4], %5;"
               : "=r"(v.a.x), "=r"(v.a.y), "=r"(v.a.z), "=r"(v.a.w)
               : "l"(&p->a), "l"(pol));
  asm volatile("ld.global.nc.L2::cache_hint.v4.b32 {%0,%1,%2,%3}, [%4], %5;"
               : "=r"(v.b.x), "=r"(v.b.y), "=r"(v.b.z), "=r"(v.b.w)
               : "l"(&p->b), "l"(pol));
  return v;
#else
  return *p;
#endif
}

// 8 bf16 (one uint4) dotted against 8 fp32.
__device__ __forceinline__ float dot8(const uint4& wv, const float2* xf) {
  const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&wv);
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 wf = __bfloat1622float2(wb[j]);
    s = fmaf(wf.x, xf[j].x, s);
    s = fmaf(wf.y, xf[j].y, s);
  }
  return s;
}

// M rows of x (compile-time), NW warps per block, RPW weight rows per warp.
// Each warp owns RPW output rows at a time and streams them to the end of the
// row before advancing by the grid stride, so every request is a full 32 B per
// lane / 1 KB per warp contiguous run.
template <int M, int NW, int RPW>
__global__ __launch_bounds__(NW * 32) void fk_gemv_k(
    const v32* __restrict__ W, const uint4* __restrict__ X,
    __nv_bfloat16* __restrict__ Y, int N, int K, long long rowstride) {
  extern __shared__ uint4 xs[];  // M * K bf16, staged once per block
  const int nx = M * K / 8;
  for (int i = threadIdx.x; i < nx; i += NW * 32) xs[i] = X[i];
  __syncthreads();

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int NV = K / 16;  // 32 B steps per weight row
  const long long step = (long long)gridDim.x * NW * RPW;
  for (long long row0 = (long long)blockIdx.x * NW * RPW + (long long)warp * RPW;
       row0 < N; row0 += step) {
    float acc[RPW][M];
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
      for (int m = 0; m < M; ++m) acc[r][m] = 0.f;
    const v32* wp[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      // Clamp instead of predicate: the tail rows re-read row N-1 and their
      // results are dropped at the store, which keeps the inner loop branchless.
      const long long rr = (row0 + r < N) ? row0 + r : (long long)N - 1;
      wp[r] = (const v32*)((const char*)W + rr * rowstride * 2);
    }
    for (int i = lane; i < NV; i += 32) {
      v32 wv[RPW];
#pragma unroll
      for (int r = 0; r < RPW; ++r) wv[r] = ld_w(wp[r] + i);
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const uint4* xrow = xs + (long long)m * (K / 8);
        const uint4 x0 = xrow[2 * i], x1 = xrow[2 * i + 1];
        const __nv_bfloat162* xa = reinterpret_cast<const __nv_bfloat162*>(&x0);
        const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&x1);
        float2 xf[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          xf[j] = __bfloat1622float2(xa[j]);
          xf[4 + j] = __bfloat1622float2(xb[j]);
        }
#pragma unroll
        for (int r = 0; r < RPW; ++r)
          acc[r][m] += dot8(wv[r].a, xf) + dot8(wv[r].b, xf + 4);
      }
    }
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
      for (int m = 0; m < M; ++m) {
        float v = acc[r][m];
#pragma unroll
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if (lane == 0 && row0 + r < N)
          Y[(long long)m * N + row0 + r] = __float2bfloat16(v);
      }
  }
}

// Grid: 6 waves of 32-row blocks, capped by the row count.  Swept over
// {148, 296, 444, 592, 888, 1184, 2368} x {4, 8, 16} warps x {1, 2, 4} rows per
// warp; everything from 444 blocks up is within 1.5% of the best, so this picks
// the middle of the plateau.
constexpr int kNW = 16, kRPW = 2, kWaves = 6;

int sm_count() {
  static int n = [] {
    int dev = 0, sms = 148;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    return sms;
  }();
  return n;
}

}  // namespace

at::Tensor fk_lmhead_gemv(const at::Tensor& x, const at::Tensor& w) {
  const c10::cuda::CUDAGuard guard(w.device());
  const int M = (int)x.size(0), K = (int)x.size(1), N = (int)w.size(0);
  TORCH_CHECK(K % 16 == 0, "K must be a multiple of 16");
  TORCH_CHECK(M >= 1 && M <= 8, "M out of range");
  at::Tensor y = at::empty({M, N}, w.options());
  const int rows_per_block = kNW * kRPW;
  const int grid = (int)std::min<long long>(
      ((long long)N + rows_per_block - 1) / rows_per_block,
      (long long)sm_count() * kWaves);
  const size_t shm = (size_t)M * K * 2;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const v32* wp = (const v32*)w.const_data_ptr();
  const uint4* xp = (const uint4*)x.const_data_ptr();
  __nv_bfloat16* yp = (__nv_bfloat16*)y.data_ptr();
  const long long rs = w.stride(0);
#define FK_LAUNCH(MM)                                                       \
  fk_gemv_k<MM, kNW, kRPW><<<grid, kNW * 32, shm, stream>>>(wp, xp, yp, N, K, rs)
  switch (M) {
    case 1: FK_LAUNCH(1); break;
    case 2: FK_LAUNCH(2); break;
    case 3: FK_LAUNCH(3); break;
    case 4: FK_LAUNCH(4); break;
    case 5: FK_LAUNCH(5); break;
    case 6: FK_LAUNCH(6); break;
    case 7: FK_LAUNCH(7); break;
    default: FK_LAUNCH(8); break;
  }
#undef FK_LAUNCH
  C10_CUDA_CHECK(cudaGetLastError());
  return y;
}
"""

_DECL = ("#include <torch/extension.h>\n"
         "at::Tensor fk_lmhead_gemv(const at::Tensor&, const at::Tensor&);\n")

# Build for the present GPU only: the default arch list is 7 targets (~90 s of
# nvcc) versus ~15 s for one, and this compiles inside the bench worker.
if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
    try:
        _cc = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{_cc[0]}.{_cc[1]}"
    except Exception:  # noqa: BLE001 - fall back to torch's default list
        pass

_GEMV = None
try:
    from torch.utils.cpp_extension import load_inline

    _EXT = load_inline(
        name="fk_lmhead_" + hashlib.sha1(_CUDA.encode()).hexdigest()[:12],
        cpp_sources=_DECL,
        cuda_sources=_CUDA,
        functions=["fk_lmhead_gemv"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    _GEMV = _EXT.fk_lmhead_gemv
except Exception:  # noqa: BLE001 - no nvcc / unsupported arch: defer to cuBLAS
    _GEMV = None

# (K, N) -> largest M for which the kernel above was measured *faster than the
# reference on this GPU*, which is the whole dispatch gate.  Both entries are
# captured vocab projections (151936x2048 and 163840x2304) and both stop at
# M == 1: 1.135x / 1.081x there, 0.907x / 0.820x at M == 2.  Everything else --
# every M in the tensor-core regime, every other (K, N), and both compute-bound
# M=16384 shapes -- falls through to the L1 Matmul.
_TINY_MAX_M = {
    (2048, 151936): 1,
    (2304, 163840): 1,
}


def _tiny_projection(x, w):
    """The hand-written GEMV, or None when this shape must defer to cuBLAS."""
    if _GEMV is None or x.dim() != 2 or w.dim() != 2 or not x.is_cuda:
        return None
    if x.dtype is not torch.bfloat16 or w.dtype is not torch.bfloat16:
        return None
    M, K = x.shape
    N = w.shape[0]
    if M < 1 or w.shape[1] != K or M > _TINY_MAX_M.get((K, N), 0):
        return None
    # 32 B vector loads: row pitch and both base pointers must be 32 B aligned
    # (the caching allocator hands out 512 B-aligned blocks, and the benchmark's
    # shifting input pool steps by 256 B, so this holds in practice).
    if K % 16 or w.stride(1) != 1 or w.stride(0) % 16 or not x.is_contiguous():
        return None
    if M * K * 2 > 32768:  # x is staged in dynamic shared memory (48 KB w/o opt-in)
        return None
    if (w.data_ptr() | x.data_ptr()) % 32:
        return None
    return _GEMV(x, w)


class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 params_dtype: torch.dtype | None = None,
                 org_num_embeddings: int | None = None,
                 padding_size: int = 64):
        super().__init__()
        tp, rank = _tp_size(), _tp_rank()
        assert num_embeddings % tp == 0
        self.num_embeddings = num_embeddings
        self.org_vocab_size = org_num_embeddings or num_embeddings
        self.padding_size = padding_size
        self.embedding_dim = embedding_dim
        self.per_partition = num_embeddings // tp
        self.vocab_start = self.per_partition * rank
        self.vocab_end = self.vocab_start + self.per_partition
        self.tp_size = tp
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.embedding_op = Embedding(self.per_partition, embedding_dim)
        self.embedding_op.emb.weight.weight_loader = self._weight_loader
        self.allreduce = AllReduce()

    def _weight_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        shard = param.data.size(0)
        param.data.copy_(loaded_weight.narrow(0, rank * shard, shard))

    def forward(self, x):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start) & (x < self.vocab_end)
            x = mask * (x - self.vocab_start)
        y = self.embedding_op(x)
        if self.tp_size > 1:
            y = mask.unsqueeze(-1) * y
            y = self.allreduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 bias: bool = False,
                 params_dtype: torch.dtype | None = None,
                 org_num_embeddings: int | None = None,
                 padding_size: int = 64):
        super().__init__(num_embeddings, embedding_dim,
                         params_dtype=params_dtype,
                         org_num_embeddings=org_num_embeddings,
                         padding_size=padding_size)
        self.linear_op = Matmul()

    def project(self, x):
        """Linear projection only (no gather). Used inside CUDA graph."""
        ctx = get_context()
        if ctx.is_mixed:
            x = x[ctx.logit_indices].contiguous()
        elif ctx.is_prefill:
            last_indices = ctx.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        weight = self.embedding_op.emb.weight
        y = _tiny_projection(x, weight)
        if y is not None:
            return y
        return self.linear_op(x, weight)

    def gather_logits(self, logits):
        """Gather partial logits from all ranks. Used outside CUDA graph."""
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if _tp_rank() == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if _tp_rank() == 0 else logits
        return logits

    def gather_greedy(self, logits):
        """Fast path for greedy: local argmax + small allgather.

        Instead of gathering full vocab logits (~31MB/rank), gather only
        the (max_val, max_idx) per sequence (~2KB/rank).
        Returns token IDs directly on rank 0, None on other ranks.
        """
        if self.tp_size <= 1:
            return None

        rank = _tp_rank()
        local_max_vals, local_max_idxs = logits.max(dim=-1)
        local_max_idxs = local_max_idxs + self.vocab_start

        info = torch.stack([local_max_vals, local_max_idxs.float()], dim=-1)
        gathered = [torch.empty_like(info) for _ in range(self.tp_size)]
        dist.all_gather(gathered, info)
        if rank == 0:
            all_info = torch.stack(gathered, dim=0)
            all_vals = all_info[:, :, 0]
            all_idxs = all_info[:, :, 1].long()
            best_rank = all_vals.argmax(dim=0)
            bs = logits.size(0)
            token_ids = all_idxs[best_rank, torch.arange(bs, device=logits.device)]
            return token_ids
        return None

    def forward(self, x):
        logits = self.project(x)
        return self.gather_logits(logits)
