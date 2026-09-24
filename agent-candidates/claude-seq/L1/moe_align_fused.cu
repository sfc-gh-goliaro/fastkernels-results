// Fused single-launch MoE token-to-expert alignment.
//
// The whole alignment -- histogram, padded prefix sum, token scatter, padding
// fill and the block->expert table -- runs in ONE kernel launch instead of the
// usual count-then-scatter pair.  At these sizes a launch costs more than the
// work it carries, so the kernel count, not the arithmetic, sets the latency.
//
// Placing tokens needs the padded prefix sum, which needs every expert's count,
// so the counts have to be agreed on before anything can be written.  There are
// two ways to pay for that, and which is cheaper depends on the input size:
//
//   Large inputs (moe_align_fused_kernel): each block counts its own slice of
//   the tokens, and one thread-block *cluster* barrier later every block reads
//   its peers' histograms straight out of distributed shared memory.  No global
//   scratch, no global atomics, no second kernel.  Blocks with a heavy slice
//   also stage their tokens through shared memory first so each expert's run
//   reaches global memory as one contiguous burst rather than scattered words.
//
//   Small inputs (moe_align_small_kernel): counting a few thousand tokens costs
//   less than a cluster barrier plus a distributed-shared-memory round trip, so
//   every block just counts the *whole* input redundantly and needs no
//   cross-block communication at all.  Each block then owns a contiguous slice
//   of experts, which also confines its stores to a narrow output window.
//
// Both share two departures from the usual formulation: the input is read once
// into registers and reused for the scatter pass, and only genuine padding slots
// are stamped with the invalid-token marker (each expert's tail plus the region
// past the padded total) instead of pre-filling the whole buffer -- every slot
// is still written exactly once.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <torch/extension.h>

namespace cg = cooperative_groups;

#define WARP_SZ 32
#define CG 16  // cluster size == grid size (hardware maximum)

__device__ __forceinline__ int pad_up(int c, int bs, int bsl) {
  return (bsl >= 0) ? ((c + bs - 1) & ~(bs - 1)) : ((c + bs - 1) / bs * bs);
}

__device__ __forceinline__ int div_bs(int c, int bs, int bsl) {
  return (bsl >= 0) ? (c >> bsl) : (c / bs);
}

// Block-wide exclusive scan of in[0..E) into out[0..E); returns the total.
// With PAD, every element is first rounded up to a multiple of the block size.
template <int T, bool PAD>
__device__ __forceinline__ int blk_scan(
    const int* __restrict__ in,
    int* __restrict__ out,
    int E,
    int bs,
    int bsl,
    int* agg,
    int tid,
    int lane,
    int wid) {
  const int K = (E + T - 1) / T;  // experts per thread
  const int e0 = tid * K;
  int local = 0;
  for (int k = 0; k < K; ++k) {
    const int e = e0 + k;
    if (e < E) local += PAD ? pad_up(in[e], bs, bsl) : in[e];
  }
  int v = local;
#pragma unroll
  for (int o = 1; o < WARP_SZ; o <<= 1) {
    const int n = __shfl_up_sync(0xffffffffu, v, o);
    if (lane >= o) v += n;
  }
  if (lane == WARP_SZ - 1) agg[wid] = v;
  __syncthreads();
  if (wid == 0) {
    int w = (lane < T / WARP_SZ) ? agg[lane] : 0;
#pragma unroll
    for (int o = 1; o < WARP_SZ; o <<= 1) {
      const int n = __shfl_up_sync(0xffffffffu, w, o);
      if (lane >= o) w += n;
    }
    if (lane < T / WARP_SZ) agg[lane] = w;
  }
  __syncthreads();
  int acc = v - local + ((wid == 0) ? 0 : agg[wid - 1]);
  for (int k = 0; k < K; ++k) {
    const int e = e0 + k;
    if (e < E) {
      out[e] = acc;
      acc += PAD ? pad_up(in[e], bs, bsl) : in[e];
    }
  }
  return agg[T / WARP_SZ - 1];
}

// VPT = int4 loads per thread (0 = runtime-bounded loop, any input size).
// STAGE = route the scatter through shared memory so the global writes of each
// expert's run are contiguous; pays off once the scattered-store traffic
// outweighs the extra scan (large inputs only).
template <int T, int VPT, bool STAGE>
__global__ void moe_align_fused_kernel(
    const int* __restrict__ topk,
    int* __restrict__ sorted,
    int* __restrict__ eids,
    int* __restrict__ npp,
    int numel,
    int E,
    int bs,
    int bsl,
    int max_padded,
    int chunk) {
  extern __shared__ int sm[];
  int* hist = sm;            // [E]     this block's histogram, read over DSMEM
  int* pfx = hist + E;       // [E + 1] padded exclusive prefix, cluster-wide
  int* cnt = pfx + E + 1;    // [E]     cluster-wide count per expert
  int* cur = cnt + E;        // [E]     write cursor
  int* gbase = cur + E;      // [E]     global destination of this block's run
  int* stg = gbase + E;      // [chunk] staging buffer            (STAGE only)
  __shared__ int agg[T / WARP_SZ];

  cg::cluster_group cl = cg::this_cluster();
  const int tid = threadIdx.x;
  const int b = blockIdx.x;
  const int lane = tid & (WARP_SZ - 1);
  const int wid = tid >> 5;

  for (int e = tid; e < E; e += T) hist[e] = 0;
  __syncthreads();

  const int start = b * chunk;
  const int end = min(numel, start + chunk);
  const int end4 = start + ((end > start) ? ((end - start) & ~3) : 0);

  // ---- pass 1: this block's partial histogram ----------------------------
  // The loads are issued together so their latencies overlap, and the values
  // stay in registers for the scatter pass.  Idle lanes must skip the atomics
  // rather than fold into a shared discard bin: same-address shared atomics
  // serialize, and on a small input most lanes are idle.
  int4 buf[VPT > 0 ? VPT : 1];
  if (VPT > 0) {
#pragma unroll
    for (int k = 0; k < VPT; ++k) {
      const int i = start + (tid + k * T) * 4;
      buf[k] = (i < end4) ? *reinterpret_cast<const int4*>(topk + i)
                          : make_int4(0, 0, 0, 0);
    }
#pragma unroll
    for (int k = 0; k < VPT; ++k) {
      if (start + (tid + k * T) * 4 < end4) {
        atomicAdd(&hist[buf[k].x], 1);
        atomicAdd(&hist[buf[k].y], 1);
        atomicAdd(&hist[buf[k].z], 1);
        atomicAdd(&hist[buf[k].w], 1);
      }
    }
  } else {
    for (int i = start + tid * 4; i < end4; i += T * 4) {
      const int4 v = *reinterpret_cast<const int4*>(topk + i);
      atomicAdd(&hist[v.x], 1);
      atomicAdd(&hist[v.y], 1);
      atomicAdd(&hist[v.z], 1);
      atomicAdd(&hist[v.w], 1);
    }
  }
  for (int i = end4 + tid; i < end; i += T) atomicAdd(&hist[topk[i]], 1);

  cl.sync();

  // ---- cluster-wide counts + this block's base, via distributed shared mem -
  for (int e = tid; e < E; e += T) {
    int tot = 0, base = 0;
#pragma unroll
    for (int r = 0; r < CG; ++r) {
      const int v = cl.map_shared_rank(hist, r)[e];
      tot += v;
      base += (r < b) ? v : 0;
    }
    cnt[e] = tot;
    cur[e] = base;
  }
  // No block may exit (and release its shared memory) while a peer might still
  // be reading its hist[] over DSMEM, so the cluster meets again here.
  cl.sync();

  const int total = blk_scan<T, true>(cnt, pfx, E, bs, bsl, agg, tid, lane, wid);
  if (tid == 0) {
    pfx[E] = total;
    if (b == 0) *npp = total;
  }
  __syncthreads();
  if (STAGE) {
    for (int e = tid; e < E; e += T) gbase[e] = cur[e] + pfx[e];
    __syncthreads();
    blk_scan<T, false>(hist, cur, E, bs, bsl, agg, tid, lane, wid);
  } else {
    for (int e = tid; e < E; e += T) cur[e] += pfx[e];
  }
  __syncthreads();

  // ---- pass 2: place every token of this slice --------------------------
  int* dst = STAGE ? stg : sorted;
  if (VPT > 0) {
#pragma unroll
    for (int k = 0; k < VPT; ++k) {
      const int i = start + (tid + k * T) * 4;
      if (i < end4) {
        dst[atomicAdd(&cur[buf[k].x], 1)] = i;
        dst[atomicAdd(&cur[buf[k].y], 1)] = i + 1;
        dst[atomicAdd(&cur[buf[k].z], 1)] = i + 2;
        dst[atomicAdd(&cur[buf[k].w], 1)] = i + 3;
      }
    }
  } else {
    for (int i = start + tid * 4; i < end4; i += T * 4) {
      const int4 v = *reinterpret_cast<const int4*>(topk + i);
      dst[atomicAdd(&cur[v.x], 1)] = i;
      dst[atomicAdd(&cur[v.y], 1)] = i + 1;
      dst[atomicAdd(&cur[v.z], 1)] = i + 2;
      dst[atomicAdd(&cur[v.w], 1)] = i + 3;
    }
  }
  for (int i = end4 + tid; i < end; i += T) dst[atomicAdd(&cur[topk[i]], 1)] = i;

  const int nw = T / WARP_SZ;
  if (STAGE) {
    __syncthreads();
    // Each block copies out its own staged tokens, for every expert.
    for (int e = wid; e < E; e += nw) {
      const int lc = hist[e];
      const int lo = cur[e] - lc;  // cursor ended one past this block's run
      const int g0 = gbase[e];
      for (int j = lane; j < lc; j += WARP_SZ) sorted[g0 + j] = stg[lo + j];
    }
  }

  // ---- padding slots + block->expert table, one warp per expert ---------
  for (int e = b * nw + wid; e < E; e += CG * nw) {
    const int hi = pfx[e + 1];
    for (int p = pfx[e] + cnt[e] + lane; p < hi; p += WARP_SZ) sorted[p] = numel;
    const int qh = div_bs(hi, bs, bsl);
    for (int q = div_bs(pfx[e], bs, bsl) + lane; q < qh; q += WARP_SZ) eids[q] = e;
  }

  // ---- slots past the padded total --------------------------------------
  for (int p = total + b * T + tid; p < max_padded; p += CG * T) sorted[p] = numel;
}


// ---------------------------------------------------------------------------
// Small-input kernel: no cross-block communication at all.
//
// Every block histograms the *whole* input (a few thousand shared-memory
// atomics -- cheaper here than a cluster barrier plus a distributed-shared-
// memory round trip) so each block independently knows the exact per-expert
// counts and therefore the padded prefix.  Each block then owns a contiguous
// slice of experts and places every token belonging to them, which also keeps
// its stores inside a narrow window of the output.  The only synchronization
// is __syncthreads.
// ---------------------------------------------------------------------------
template <int T, int VPT, int G>
__global__ void moe_align_small_kernel(
    const int* __restrict__ topk,
    int* __restrict__ sorted,
    int* __restrict__ eids,
    int* __restrict__ npp,
    int numel,
    int E,
    int bs,
    int bsl,
    int max_padded) {
  extern __shared__ int sm[];
  int* hist = sm;          // [E]     global count per expert (computed here)
  int* pfx = hist + E;     // [E + 1] padded exclusive prefix
  int* cur = pfx + E + 1;  // [E]     write cursor for this block's experts
  __shared__ int agg[T / WARP_SZ];

  const int tid = threadIdx.x;
  const int b = blockIdx.x;
  const int lane = tid & (WARP_SZ - 1);
  const int wid = tid >> 5;
  const int n4 = numel & ~3;

  for (int e = tid; e < E; e += T) hist[e] = 0;
  __syncthreads();

  int4 buf[VPT];
#pragma unroll
  for (int k = 0; k < VPT; ++k) {
    const int i = (tid + k * T) * 4;
    buf[k] = (i < n4) ? *reinterpret_cast<const int4*>(topk + i)
                      : make_int4(0, 0, 0, 0);
  }
#pragma unroll
  for (int k = 0; k < VPT; ++k) {
    if ((tid + k * T) * 4 < n4) {
      atomicAdd(&hist[buf[k].x], 1);
      atomicAdd(&hist[buf[k].y], 1);
      atomicAdd(&hist[buf[k].z], 1);
      atomicAdd(&hist[buf[k].w], 1);
    }
  }
  for (int i = n4 + tid; i < numel; i += T) atomicAdd(&hist[topk[i]], 1);
  __syncthreads();

  const int total = blk_scan<T, true>(hist, pfx, E, bs, bsl, agg, tid, lane, wid);
  if (tid == 0) {
    pfx[E] = total;
    if (b == 0) *npp = total;
  }
  // This block's slice of experts.
  const int e0 = (int)(((long long)b * E) / G);
  const int e1 = (int)(((long long)(b + 1) * E) / G);
  __syncthreads();
  for (int e = e0 + tid; e < e1; e += T) cur[e] = pfx[e];
  __syncthreads();

#define PLACE(v, idx)                                     \
  {                                                       \
    const int e_ = (v);                                   \
    if (e_ >= e0 && e_ < e1) sorted[atomicAdd(&cur[e_], 1)] = (idx); \
  }
#pragma unroll
  for (int k = 0; k < VPT; ++k) {
    const int i = (tid + k * T) * 4;
    if (i < n4) {
      PLACE(buf[k].x, i)
      PLACE(buf[k].y, i + 1)
      PLACE(buf[k].z, i + 2)
      PLACE(buf[k].w, i + 3)
    }
  }
  for (int i = n4 + tid; i < numel; i += T) PLACE(topk[i], i)
#undef PLACE

  // Padding slots and the block->expert table for this block's experts.
  for (int e = e0 + wid; e < e1; e += T / WARP_SZ) {
    const int hi = pfx[e + 1];
    for (int p = pfx[e] + hist[e] + lane; p < hi; p += WARP_SZ) sorted[p] = numel;
    const int qh = div_bs(hi, bs, bsl);
    for (int q = div_bs(pfx[e], bs, bsl) + lane; q < qh; q += WARP_SZ) eids[q] = e;
  }
  for (int p = total + b * T + tid; p < max_padded; p += G * T) sorted[p] = numel;
}

template <int T, int VPT, int G>
static void launch_small(
    const int* topk, int* sorted, int* eids, int* npp, int numel, int E, int bs,
    int bsl, int max_padded, cudaStream_t stream) {
  const size_t smem = (size_t)(3 * E + 1) * sizeof(int);
  moe_align_small_kernel<T, VPT, G><<<G, T, smem, stream>>>(
      topk, sorted, eids, npp, numel, E, bs, bsl, max_padded);
}

template <int T, int VPT, bool STAGE>
static void launch(
    const int* topk,
    int* sorted,
    int* eids,
    int* npp,
    int numel,
    int E,
    int bs,
    int bsl,
    int max_padded,
    cudaStream_t stream) {
  int chunk = (numel + CG - 1) / CG;
  chunk = (chunk + 3) & ~3;  // every slice stays 16B-aligned
  if (chunk < 4) chunk = 4;
  size_t smem = (size_t)(5 * E + 1) * sizeof(int);
  if (STAGE) smem += (size_t)chunk * sizeof(int);

  auto kernel = moe_align_fused_kernel<T, VPT, STAGE>;
  static bool configured = false;
  if (!configured) {
    // A 16-block cluster is beyond the portable cluster-size limit.
    cudaFuncSetAttribute((const void*)kernel,
                         cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    configured = true;
  }

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(CG, 1, 1);
  cfg.blockDim = dim3(T, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = CG;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, kernel, topk, sorted, eids, npp, numel, E, bs, bsl,
                     max_padded, chunk);
}

void moe_align_fused(
    torch::Tensor topk_ids,
    int64_t num_experts,
    int64_t block_size,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_pad) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int numel = (int)topk_ids.numel();
  const int E = (int)num_experts;
  const int bs = (int)block_size;
  const int max_padded = (int)sorted_token_ids.size(0);
  int bsl = -1;  // log2(block_size), or -1 when it is not a power of two
  for (int s = 0; s < 31; ++s)
    if ((1 << s) == bs) bsl = s;

  const int* tp = topk_ids.data_ptr<int>();
  int* sp = sorted_token_ids.data_ptr<int>();
  int* ep = expert_ids.data_ptr<int>();
  int* np = num_tokens_post_pad.data_ptr<int>();
#define ARGS tp, sp, ep, np, numel, E, bs, bsl, max_padded, stream
  // Narrow blocks and a direct scatter for small inputs; wider blocks and
  // staged (coalesced) writes once the scatter traffic dominates.
  // Small inputs: the communication-free kernel, sized so the whole input fits
  // in registers.  Large inputs: one cluster, with staging capped at VPT = 4 so
  // the buffer stays inside the 48 KB of dynamic shared memory available
  // without an opt-in; past that the runtime loop scatters straight to global.
  constexpr int SG = 16;               // blocks for the small kernel
  constexpr int SPT = 256 * 4;         // elements per int4 round
  constexpr int SMALL = CG * 256 * 4;  // 16384 elements
  constexpr int WIDE = CG * 512 * 4;   // 32768 elements
  if (numel <= 4 * SPT) {
    launch_small<256, 4, SG>(ARGS);
  } else if (numel <= 8 * SPT) {
    launch_small<256, 8, SG>(ARGS);
  } else if (numel <= 16 * SPT) {
    launch_small<256, 16, SG>(ARGS);
  } else if (numel <= 2 * SMALL) {
    launch<256, 2, false>(ARGS);
  } else if (numel <= 2 * WIDE) {
    launch<512, 2, true>(ARGS);
  } else if (numel <= 4 * WIDE) {
    launch<512, 4, true>(ARGS);
  } else {
    launch<512, 0, false>(ARGS);
  }
#undef ARGS
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_align_fused", &moe_align_fused, "Fused MoE align block size (CUDA)");
}
