"""TP-aware embedding and LM head (L2 operators).

Both operators are HBM-bandwidth problems on this GPU, and both baselines leave
bandwidth on the table, so the two kernels here are written around 256-bit
global accesses plus explicit L2 eviction priorities (``ld.global.L2::evict_*``,
which sm_90+ only accepts on 32 B accesses).

``VocabParallelEmbedding`` -- ``weight[x]`` is a row gather, bound by the output
write plus the dependent ``idx[]`` load.  The table is the re-used side (repeated
tokens hit in L2), so it is read ``evict_last`` with 32 B accesses and a deeply
unrolled inner loop that keeps enough of them in flight to cover the index read.
A single pybind entry point also avoids the ``nn.Module`` / ``F.embedding``
dispatch chain, which is most of what a one-token lookup costs.

``ParallelLMHead`` -- ``x @ W^T`` over a 150 k-wide vocabulary.  For a single-row
``x`` the whole cost is streaming ``W`` once (622 MB at K=2048): ``F.linear``
runs it at ~5.1 TB/s where an ``evict_first`` 32 B stream reaches ~6.2 TB/s, and
the gemv below closes that gap on the fp32 FMA pipe.  Wider ``x`` needs tensor
cores, and there the reference GEMM is already at the hardware peak (measured
1.36 PFLOP/s at M=494, versus ~340 TFLOP/s for the legacy ``mma.sync`` path that
is reachable without tcgen05), so those shapes keep the reference path.
"""

from __future__ import annotations

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
# CUDA kernels.  JIT-compiled at import, i.e. never inside a timed region.
# ---------------------------------------------------------------------------
_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

namespace {

struct alignas(32) V32 { unsigned x[8]; };

// 256-bit global accesses.  The L2 eviction-priority modifiers are only legal
// on .v8.b32 / .v4.b64 (32 B) accesses on sm_90+.
#define LD32(NAME, MOD)                                                        \
  __device__ __forceinline__ V32 NAME(const V32* p) {                          \
    V32 v;                                                                     \
    asm volatile(MOD " {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"                       \
                 : "=r"(v.x[0]), "=r"(v.x[1]), "=r"(v.x[2]), "=r"(v.x[3]),     \
                   "=r"(v.x[4]), "=r"(v.x[5]), "=r"(v.x[6]), "=r"(v.x[7])      \
                 : "l"(p));                                                    \
    return v;                                                                  \
  }
LD32(ld_keep,   "ld.global.nc.L2::evict_last.v8.b32")   // re-used across CTAs
LD32(ld_stream, "ld.global.nc.L2::evict_first.v8.b32")  // touched once
#undef LD32

__device__ __forceinline__ void st32(V32* p, const V32& v) {
  asm volatile("st.global.v8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
               :: "l"(p), "r"(v.x[0]), "r"(v.x[1]), "r"(v.x[2]), "r"(v.x[3]),
                  "r"(v.x[4]), "r"(v.x[5]), "r"(v.x[6]), "r"(v.x[7]) : "memory");
}

// Two bf16 packed in a u32 -> two fp32.  Exact: bf16 *is* the fp32 top half.
__device__ __forceinline__ void unpack_bf16x2(unsigned u, float& lo, float& hi) {
  lo = __int_as_float(u << 16);
  hi = __int_as_float(u & 0xffff0000u);
}

// =========================================================================
// Row gather: out[r, :] = table[idx[r], :]
// =========================================================================
template <int TPB, int VEC, typename I>
__global__ __launch_bounds__(TPB) void gather32_k(
    const V32* __restrict__ W, const I* __restrict__ idx,
    V32* __restrict__ O, int Dv, int N) {
  const int c0 = blockIdx.x * (TPB * VEC) + threadIdx.x;
  for (long r = blockIdx.y; r < N; r += gridDim.y) {
    const V32* wp = W + (long)idx[r] * Dv;
    V32* op = O + r * (long)Dv;
    V32 t[VEC];
#pragma unroll
    for (int v = 0; v < VEC; ++v) { int c = c0 + v * TPB; if (c < Dv) t[v] = ld_keep(wp + c); }
#pragma unroll
    for (int v = 0; v < VEC; ++v) { int c = c0 + v * TPB; if (c < Dv) st32(op + c, t[v]); }
  }
}

// 2-byte-granular fallback (row byte length not a multiple of 32).
template <int TPB, typename I>
__global__ __launch_bounds__(TPB) void gather_any_k(
    const short* __restrict__ W, const I* __restrict__ idx,
    short* __restrict__ O, int D, int N) {
  for (long r = blockIdx.y; r < N; r += gridDim.y) {
    const short* wp = W + (long)idx[r] * D;
    short* op = O + r * (long)D;
    for (int c = blockIdx.x * TPB + threadIdx.x; c < D; c += gridDim.x * TPB)
      op[c] = wp[c];
  }
}

constexpr int MAX_GRID_Y = 32768;
constexpr int GTPB = 64;

template <typename I>
void launch_gather(const at::Tensor& W, const at::Tensor& idx, at::Tensor& O,
                   long N, long D) {
  auto s = at::cuda::getCurrentCUDAStream();
  const I* ix = reinterpret_cast<const I*>(idx.data_ptr());
  const unsigned gy = (unsigned)(N < MAX_GRID_Y ? N : MAX_GRID_Y);
  if ((D & 15) == 0) {
    const int Dv = (int)(D >> 4);
    const V32* w = reinterpret_cast<const V32*>(W.data_ptr());
    V32* o = reinterpret_cast<V32*>(O.data_ptr());
    // 64 threads x 8 x 32 B per block covers a 8 KiB row in one CTA; deep
    // unrolling keeps enough loads in flight to hide the dependent idx[] read,
    // which is what the small-token-count lookups are actually bound by.
    constexpr int GVEC = 8;
    dim3 g((unsigned)((Dv + GTPB*GVEC - 1)/(GTPB*GVEC)), gy);
    gather32_k<GTPB, GVEC, I><<<g, GTPB, 0, s>>>(w, ix, o, Dv, (int)N);
  } else {
    const short* w = reinterpret_cast<const short*>(W.data_ptr());
    short* o = reinterpret_cast<short*>(O.data_ptr());
    dim3 g((unsigned)((D + 255) / 256), gy);
    gather_any_k<256, I><<<g, 256, 0, s>>>(w, ix, o, (int)D, (int)N);
  }
}

// =========================================================================
// Skinny GEMM: C[M,N] = A[M,K] . B[N,K]^T, small M, fp32 accumulate.
// One warp owns NPW columns of the output; a lane holds 16 consecutive k of B
// (one 32 B access).  No shared memory: the M x K activation is tiny and stays
// resident in L2 for every CTA, so it is re-read with evict_last.
// =========================================================================
template <int MM, int TPB, int NPW, int U>
__global__ __launch_bounds__(TPB) void gemv_k(
    const V32* __restrict__ A, const V32* __restrict__ B,
    __nv_bfloat16* __restrict__ C, int M, int N, int Kv) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const long nBase = (long)blockIdx.x * ((TPB / 32) * NPW) + (long)warp * NPW;

  const V32* bp[NPW];
#pragma unroll
  for (int j = 0; j < NPW; ++j) {
    const long n = nBase + j;
    bp[j] = B + (n < N ? n : (long)N - 1) * (long)Kv;
  }

  float acc[NPW][MM];
#pragma unroll
  for (int j = 0; j < NPW; ++j)
#pragma unroll
    for (int m = 0; m < MM; ++m) acc[j][m] = 0.f;

  for (int kv0 = 0; kv0 < Kv; kv0 += 32 * U) {
    V32 bv[U][NPW], av[U][MM];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int kv = kv0 + u * 32 + lane;
      const int kc = kv < Kv ? kv : 0;
#pragma unroll
      for (int j = 0; j < NPW; ++j) bv[u][j] = ld_stream(bp[j] + kc);
#pragma unroll
      for (int m = 0; m < MM; ++m) {
        // clamp the row so m >= M never reads past A; the value is zeroed below
        av[u][m] = ld_keep(A + (long)(m < M ? m : 0) * Kv + kc);
        if (kv >= Kv || m >= M) {
#pragma unroll
          for (int e = 0; e < 8; ++e) av[u][m].x[e] = 0u;
        }
      }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        float alo[MM], ahi[MM];
#pragma unroll
        for (int m = 0; m < MM; ++m) unpack_bf16x2(av[u][m].x[e], alo[m], ahi[m]);
#pragma unroll
        for (int j = 0; j < NPW; ++j) {
          float blo, bhi;
          unpack_bf16x2(bv[u][j].x[e], blo, bhi);
#pragma unroll
          for (int m = 0; m < MM; ++m) {
            acc[j][m] = fmaf(alo[m], blo, acc[j][m]);
            acc[j][m] = fmaf(ahi[m], bhi, acc[j][m]);
          }
        }
      }
    }
  }
#pragma unroll
  for (int j = 0; j < NPW; ++j) {
    const long n = nBase + j;
#pragma unroll
    for (int m = 0; m < MM; ++m) {
      float v = acc[j][m];
#pragma unroll
      for (int off = 16; off; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
      if (lane == 0 && m < M && n < N) C[(long)m * N + n] = __float2bfloat16(v);
    }
  }
}

}  // namespace

at::Tensor gather_rows(const at::Tensor& W, const at::Tensor& idx) {
  const long D = W.size(1);
  const long N = idx.numel();
  at::Tensor O = at::empty({N, D}, W.options());
  if (N == 0) return O;
  if (idx.scalar_type() == at::kLong) launch_gather<long>(W, idx, O, N, D);
  else                                launch_gather<int>(W, idx, O, N, D);
  return O;
}

// Caller guarantees bf16, contiguous, K % 16 == 0 and a small M.
at::Tensor skinny_mm(const at::Tensor& A, const at::Tensor& B) {
  const int M = (int)A.size(0), K = (int)A.size(1), N = (int)B.size(0);
  at::Tensor C = at::empty({M, N}, A.options());
  const int Kv = K >> 4;
  auto s = at::cuda::getCurrentCUDAStream();
  const V32* a = (const V32*)A.data_ptr();
  const V32* b = (const V32*)B.data_ptr();
  __nv_bfloat16* c = (__nv_bfloat16*)C.data_ptr();
  constexpr int TPB = 256, NPW = 1, U = 2;
  const int cols = (TPB / 32) * NPW;
  const int grid = (N + cols - 1) / cols;
  if (M <= 1)      gemv_k<1, TPB, NPW, U><<<grid, TPB, 0, s>>>(a, b, c, M, N, Kv);
  else if (M <= 2) gemv_k<2, TPB, NPW, U><<<grid, TPB, 0, s>>>(a, b, c, M, N, Kv);
  else             gemv_k<4, TPB, NPW, U><<<grid, TPB, 0, s>>>(a, b, c, M, N, Kv);
  // M > 4 is never dispatched here (see _GEMV_MAX_M).
  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather_rows", &gather_rows);
  m.def("skinny_mm", &skinny_mm);
}
"""


def _build_ext():
    from torch.utils.cpp_extension import load_inline
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}.{minor}"
        if major in (9, 10, 12):
            arch += "a"       # L2::evict_* on 32 B accesses needs the 'a' variant
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    return load_inline(
        name="fk_parallel_embedding_v2",
        cpp_sources="",
        cuda_sources=_CUDA_SRC,
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


_EXT_ERR = None
try:
    _EXT = _build_ext() if torch.cuda.is_available() else None
except Exception as _exc:  # pragma: no cover -- keep the reference path
    _EXT, _EXT_ERR = None, _exc

# The gemv does M FLOP per byte of ``W`` on the fp32 FMA pipe, so it is only
# HBM-bound while M is tiny.  Measured against the reference GEMM (K=2048/2304/
# 4096, V=152k/164k): M=1 is 1.07-1.16x, M=2 is already mixed (0.91x at K=2304)
# and M>=3 is ~0.6x, so only single-row projections take this path.
_GEMV_MAX_M = 1


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

    def _lookup(self, x):
        """``weight[x]`` via the fused gather when the layout allows it."""
        emb = self.embedding_op.emb
        w = emb.weight.data
        if (_EXT is not None and emb.padding_idx is None and w.is_cuda
                and w.is_contiguous() and w.element_size() == 2
                and x.dtype in (torch.int64, torch.int32) and x.is_contiguous()):
            out = _EXT.gather_rows(w, x)
            return out if x.dim() == 1 else out.view(*x.shape, w.size(1))
        return self.embedding_op(x)

    def forward(self, x):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start) & (x < self.vocab_end)
            x = mask * (x - self.vocab_start)
            y = mask.unsqueeze(-1) * self._lookup(x)
            return self.allreduce(y)
        return self._lookup(x)


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
        w = self.embedding_op.emb.weight
        if (_EXT is not None and x.dim() == 2 and x.size(0) <= _GEMV_MAX_M
                and x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16
                and x.is_cuda and x.is_contiguous() and w.is_contiguous()
                and (x.size(1) & 15) == 0):
            return _EXT.skinny_mm(x, w.data)
        return self.linear_op(x, w)

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
