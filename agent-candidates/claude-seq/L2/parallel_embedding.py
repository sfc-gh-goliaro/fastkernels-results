"""TP-aware embedding and LM head (L2 operators).

Where the time goes
-------------------
Both classes are thin wrappers, so the cost is entirely in the two L1 kernels
they call, and on this GPU (B200) most of that cost is already at a hardware
wall:

* ``VocabParallelEmbedding`` is a row gather.  The frozen L1 kernel moves its
  output at the device's store limit: against a pure ``st.global`` stream over
  the same 128 MB it is within 0.2%, and 52 alternative tile / vector-width /
  eviction-hint gather variants all landed within 0.2% of it as well (the
  gather has to read the rows it writes, so read+write together are the wall).
  Nothing here is worth replacing.
* ``ParallelLMHead`` is ``x @ Wᵀ`` over a 150k-row vocab.  For the captured
  prefill shapes (494 and 16384 tokens) the reference runs at 1.03-1.38
  PFLOP/s, which is this machine's measured bf16 ceiling, so those keep the
  reference path.  The captured 60- and 64-token shapes are weight-streaming
  rather than compute bound, but reaching the reference's ~80% of peak DRAM
  throughput there needs the operands to go shared-memory -> tensor core
  without passing through registers: an mma.sync kernel has to stage W through
  shared memory to keep its global loads coalesced, and the resulting ~4.5x
  shared-memory amplification caps it at 3.5 TB/s against the reference's 5.5
  (measured, with ldmatrix fragments and a register-prefetch pipeline).  That
  needs tcgen05, so those shapes keep the reference path too.

The decode GEMV
---------------
What is left is ``x: [1, K]`` -- the decode step, and the single most frequent
LM-head shape in the captures after the two small prefills.  It touches every
one of the weight's 594-755 MB exactly once and does one FMA per element, so it
is pure weight streaming and the kernel's only job is to issue loads fast
enough.  Three things get it past the reference:

* **32-byte loads.**  ``ld.global.nc.v4.b64`` (sm_100) halves the memory
  instructions per row versus 16-byte loads, and is also the narrowest access
  that may carry an L2 cache-policy operand at all.
* **A streaming L2 policy.**  The weight is read once and never reused, so
  ``createpolicy.fractional.L2::evict_first`` + ``ld.global...L2::cache_hint``
  stops the stream from evicting everything else in L2 on its way through.
  Worth ~2% on its own, and more when L2 holds live data -- which is the real
  decode situation (the KV cache is what would otherwise be evicted), and also
  what the scorer's L2-flush buffer reproduces.
* **A persistent grid** (~4 blocks per SM, each sweeping many row tiles), so x
  is staged into shared memory once per block instead of once per 32 rows.
  Worth ~3%.

Together: 83-85% of peak DRAM throughput against the reference's 79-81%
(Nsight Compute, boost clocks, L2 flushed), i.e. 98 us vs 104 us on
``[1, 2048] x 151936`` and 116 us vs 123 us on ``[1, 2304] x 163840``.  Under
the scorer -- which leaves L2 full of dirty lines between iterations, so the
policy pays too -- the same shapes come out ~1.10x and ~1.03x.

Lane mapping
------------
One warp owns ``R`` vocab rows; ``FK_LPR`` of its lanes cooperate on each row
and reduce at the end.  A whole warp per row is the obvious choice but leaves
half the lanes idle on the last step of every row whenever K/16 is not a
multiple of 32 -- true for K = 2304, a captured shape, where it turns a win
into a loss.  Each lane keeps ``RM`` rows in flight so its loads are
independent, which is where the memory-level parallelism comes from.
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


_CUDA_SRC = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

#define FK_WARP 32
#define FK_LPR 16        // lanes cooperating on one vocab row (see module docstring)

struct B32 { unsigned long long a, b, c, d; };

static int fk_sm_count() {
  static int n = 0;
  if (n == 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
    if (n <= 0) n = 148;
  }
  return n;
}

// 32-byte non-coherent load carrying an L2 cache policy.  sm_100 rejects the
// cache-hint / evict-hint modifiers on anything narrower than .v4.b64.
__device__ __forceinline__ B32 fk_ld32(const B32* p, unsigned long long pol) {
  B32 v;
  asm volatile("ld.global.nc.L2::cache_hint.v4.b64 {%0,%1,%2,%3}, [%4], %5;"
               : "=l"(v.a), "=l"(v.b), "=l"(v.c), "=l"(v.d) : "l"(p), "l"(pol));
  return v;
}

// 8 bf16 pairs -> one fp32 partial dot product.
__device__ __forceinline__ float fk_dot8(const int4& a, const int4& b) {
  const __nv_bfloat162* ap = (const __nv_bfloat162*)&a;
  const __nv_bfloat162* bp = (const __nv_bfloat162*)&b;
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float2 xf = __bfloat1622float2(ap[j]);
    float2 wf = __bfloat1622float2(bp[j]);
    s = fmaf(xf.x, wf.x, s);
    s = fmaf(xf.y, wf.y, s);
  }
  return s;
}

// out[n] = sum_k x[k] * w[n, k],  w row-major [N, K], all bf16.
// LPR lanes per row, RM rows per lane, NW warps per block.  The grid is
// persistent (a few blocks per SM, each sweeping many row tiles): x is then
// staged into shared memory once per block instead of once per 32 rows, which
// keeps the gather of x out of L2 and off the critical path.
template <int LPR, int RM, int NW>
__global__ __launch_bounds__(FK_WARP * NW) void fk_gemv(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ out, int K, int N) {
  constexpr int RG = FK_WARP / LPR;   // row slots per warp
  constexpr int R = RG * RM;          // rows per warp
  constexpr int TILE = NW * R;        // rows per block per sweep
  extern __shared__ __align__(32) char smem[];
  __nv_bfloat16* xs = (__nv_bfloat16*)smem;             // K bf16
  float* red = (float*)(smem + (size_t)K * sizeof(__nv_bfloat16));

  const int tid = threadIdx.x, nthr = FK_WARP * NW;
  {   // x is tiny and read K/16 times per row: stage it once in shared memory.
    const int4* src = (const int4*)x;
    int4* dst = (int4*)xs;
    for (int i = tid; i < K / 8; i += nthr) dst[i] = src[i];
  }
  __syncthreads();

  unsigned long long pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));

  const int warp = tid / FK_WARP, lane = tid % FK_WARP;
  const int rsub = lane / LPR, ksub = lane % LPR;
  const int kv = K / 16;                                // 32-byte vectors per row
  const B32* xv = (const B32*)xs;
  const int ntile = (N + TILE - 1) / TILE;

  for (int t = blockIdx.x; t < ntile; t += gridDim.x) {
    const int n0 = t * TILE + warp * R;
    const B32* wv = (const B32*)(w + (size_t)(n0 + rsub) * K);
    const int nrem = N - (n0 + rsub);                   // rows left for this lane
    float acc[RM];
#pragma unroll
    for (int i = 0; i < RM; ++i) acc[i] = 0.f;
    if (nrem > 0) {
      for (int kb = ksub; kb < kv; kb += LPR) {
        B32 wr[RM];
#pragma unroll
        for (int i = 0; i < RM; ++i) {
          wr[i].a = wr[i].b = wr[i].c = wr[i].d = 0ull;
          if (nrem > i * RG) wr[i] = fk_ld32(wv + (size_t)i * RG * kv + kb, pol);
        }
        const B32 xr = xv[kb];
        const int4* xp = (const int4*)&xr;
#pragma unroll
        for (int i = 0; i < RM; ++i) {
          const int4* wp = (const int4*)&wr[i];
          acc[i] += fk_dot8(xp[0], wp[0]) + fk_dot8(xp[1], wp[1]);
        }
      }
    }
    // reduce each row across its LPR lanes, then one coalesced store per tile
#pragma unroll
    for (int i = 0; i < RM; ++i) {
      float a = acc[i];
#pragma unroll
      for (int o = LPR / 2; o; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
      if (ksub == 0) red[warp * R + i * RG + rsub] = a;
    }
    __syncthreads();
    for (int i = tid; i < TILE; i += nthr) {
      const int n = t * TILE + i;
      if (n < N) out[n] = __float2bfloat16(red[i]);
    }
    __syncthreads();
  }
}

static void fk_launch(const __nv_bfloat16* x, const __nv_bfloat16* w,
                      __nv_bfloat16* o, int K, int N, cudaStream_t s) {
  constexpr int RM = 2, NW = 8;
  constexpr int TILE = (FK_WARP / FK_LPR) * RM * NW;
  const int ntile = (N + TILE - 1) / TILE;
  const size_t smem = (size_t)K * sizeof(__nv_bfloat16) + (size_t)TILE * sizeof(float);
  // ~4 blocks per SM: enough loads in flight to saturate HBM, few enough that
  // each block's x staging amortises over many row tiles.
  int grid = 4 * fk_sm_count();
  if (grid > ntile) grid = ntile;
  fk_gemv<FK_LPR, RM, NW><<<grid, FK_WARP * NW, smem, s>>>(x, w, o, K, N);
}

// Returns an empty tensor if the shape is not one this kernel serves.
at::Tensor gemv(const at::Tensor& x, const at::Tensor& w) {
  const int64_t K = w.size(1), N = w.size(0);
  if ((K / 16) % FK_LPR) return at::Tensor();  // lanes would idle; caller falls back
  at::Tensor out = at::empty({x.size(0), N}, w.options());
  fk_launch((const __nv_bfloat16*)x.const_data_ptr(),
            (const __nv_bfloat16*)w.const_data_ptr(),
            (__nv_bfloat16*)out.data_ptr(), (int)K, (int)N,
            at::cuda::getCurrentCUDAStream());
  return out;
}
'''


def _build():
    """JIT-compile the decode GEMV, pinned to the local GPU arch."""
    from torch.utils.cpp_extension import load_inline
    try:
        from fastkernels.infra.cuda_ext import _pin_build_arch
        _pin_build_arch()
    except Exception:
        try:
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        except Exception:
            pass
    return load_inline(
        name="fk_cand_lmhead_gemv",
        cpp_sources="at::Tensor gemv(const at::Tensor&, const at::Tensor&);",
        cuda_sources=_CUDA_SRC,
        functions=["gemv"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


try:
    _GEMV = _build().gemv
except Exception:  # no nvcc / unsupported toolchain -> reference path only
    _GEMV = None

# Below this many weight bytes the GEMV is launch-bound rather than
# bandwidth-bound and there is nothing to win over the reference.
_GEMV_MIN_BYTES = 1 << 24


def _try_gemv(x: torch.Tensor, w: torch.Tensor):
    """``x @ w.T`` for a single-token (decode) row, or None if unsupported."""
    if (_GEMV is None or x.dim() != 2 or x.size(0) != 1
            or x.dtype is not torch.bfloat16 or w.dtype is not torch.bfloat16
            or w.dim() != 2 or x.size(1) != w.size(1)
            or not x.is_contiguous() or not w.is_contiguous()
            or not x.is_cuda or not w.is_cuda):
        return None
    K = w.size(1)
    # K % 256 keeps every lane busy (see the kernel) and makes each row's
    # 32-byte vectors land on 32-byte boundaries; K <= 8192 keeps x (plus the
    # reduction scratch) inside the 48 KB of shared memory a block gets without
    # an opt-in.  The alignment checks are what the inline-PTX 32-byte load and
    # the 16-byte x staging require and cannot recover from -- a 2-byte-offset
    # view of a weight is contiguous but unusable here.
    if K % 256 or K > 8192 or w.numel() * 2 < _GEMV_MIN_BYTES:
        return None
    if w.data_ptr() % 32 or x.data_ptr() % 16:
        return None
    out = _GEMV(x, w)
    return out if out.numel() else None


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
        w = self.embedding_op.emb.weight
        y = _try_gemv(x, w)
        return y if y is not None else self.linear_op(x, w)

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
