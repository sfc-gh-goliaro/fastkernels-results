// Fused top-k + softmax routing for MoE, specialised for the captured workload:
// bfloat16 router logits over 128 experts, top_k <= 8, renormalise.
//
// Differences from the vLLM/SGLang-style reference kernel (`topk_softmax.cu`),
// which spends ~550 instructions per row and is issue bound:
//
//  * With `renormalize=True` the full-row softmax denominator cancels out --
//    exp(x_i-m)/S divided by sum_{j in topk} exp(x_j-m)/S is just
//    exp(x_i-m)/sum_{j in topk} exp(x_j-m) -- so only the k selected logits
//    need exp(), 8 per row instead of 128, and no full-row sum reduction.
//  * Selection runs on one 32-bit sortable key per logit that packs an
//    order-preserving transform of the bfloat16 bit pattern together with
//    `127 - expert`. A single integer max therefore picks the winner *and*
//    carries its index, and because the low bits decrease with the expert index
//    ties break towards the lower expert exactly as the reference does.
//  * Each thread reduces its own slice of the row to a sorted top-8 with
//    sorting networks (19 comparators per 8 keys, 12 for a bitonic merge), then
//    the lanes covering a row merge their lists pairwise. Nothing is ever
//    indexed dynamically in registers -- the reference's
//    `row_chunk[expert % ELTS_PER_LDG] = -inf` lowers to a long compare/select
//    chain and dominates its instruction mix.
//  * The k results end up one per lane, so they are written with a single fully
//    coalesced store per warp, instead of one thread per row writing k scalars
//    and then reading them back to renormalise.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int kExperts = 128;
constexpr int kThreadsPerRow = 8;  // lanes cooperating on one row
constexpr int kWarpsPerCta = 2;

// Order-preserving map of two packed bfloat16 bit patterns to unsigned keys:
// positive -> bits | 0x8000, negative -> ~bits. Both halves at once.
__device__ __forceinline__ uint32_t mono2(uint32_t w) {
  const uint32_t s = (w >> 15) & 0x00010001u;
  return w ^ 0x80008000u ^ (s * 0x7FFFu);
}

// Inverse of mono2 for one 16-bit key, widened to float32.
__device__ __forceinline__ float key_to_float(uint32_t key16) {
  const uint32_t m = 0x8000u | (((key16 >> 15) ^ 1u) * 0x7FFFu);
  return __uint_as_float((key16 ^ m) << 16);
}

#define FK_CE(i, j)           \
  {                           \
    const uint32_t a_ = x[i]; \
    const uint32_t b_ = x[j]; \
    x[i] = ::max(a_, b_);     \
    x[j] = ::min(a_, b_);     \
  }

// Optimal 19-comparator 8-sorter, descending.
__device__ __forceinline__ void sort8_desc(uint32_t* x) {
  FK_CE(0, 1) FK_CE(2, 3) FK_CE(4, 5) FK_CE(6, 7)
  FK_CE(0, 2) FK_CE(1, 3) FK_CE(4, 6) FK_CE(5, 7)
  FK_CE(1, 2) FK_CE(5, 6) FK_CE(0, 4) FK_CE(3, 7)
  FK_CE(1, 5) FK_CE(2, 6)
  FK_CE(1, 4) FK_CE(3, 6)
  FK_CE(2, 4) FK_CE(3, 5)
  FK_CE(3, 4)
}

// x and y are each sorted descending; leave the 8 largest of the union in x,
// sorted descending: bitonic half-cleaner against the reversal of y, then a
// 12-comparator bitonic merge of the resulting bitonic sequence.
__device__ __forceinline__ void merge_top8(uint32_t* x, const uint32_t* y) {
#pragma unroll
  for (int i = 0; i < 8; ++i) x[i] = ::max(x[i], y[7 - i]);
  FK_CE(0, 4) FK_CE(1, 5) FK_CE(2, 6) FK_CE(3, 7)
  FK_CE(0, 2) FK_CE(1, 3) FK_CE(4, 6) FK_CE(5, 7)
  FK_CE(0, 1) FK_CE(2, 3) FK_CE(4, 5) FK_CE(6, 7)
}

#undef FK_CE

// Keys for the 8 logits of one 16B load. `pb` carries the thread/group part of
// (127 - expert); the per-element part is an immediate.
__device__ __forceinline__ void pack8(uint32_t* d, const uint4& v, uint32_t pb) {
#define FK_P(i, w)                                              \
  {                                                             \
    const uint32_t t_ = mono2(w);                               \
    d[i] = (t_ << 16) | pb | (uint32_t)(7 - (i));               \
    d[(i) + 1] = (t_ & 0xFFFF0000u) | pb | (uint32_t)(6 - (i)); \
  }
  FK_P(0, v.x) FK_P(2, v.y) FK_P(4, v.z) FK_P(6, v.w)
#undef FK_P
}

// x[j] for a runtime j in [0, 8): a select tree, so x stays in registers.
__device__ __forceinline__ uint32_t pick8(const uint32_t* x, int j) {
  uint32_t a0 = (j & 1) ? x[1] : x[0];
  uint32_t a1 = (j & 1) ? x[3] : x[2];
  uint32_t a2 = (j & 1) ? x[5] : x[4];
  uint32_t a3 = (j & 1) ? x[7] : x[6];
  a0 = (j & 2) ? a1 : a0;
  a2 = (j & 2) ? a3 : a2;
  return (j & 4) ? a2 : a0;
}

constexpr int ilog2(int n) { return n <= 1 ? 0 : 1 + ilog2(n / 2); }

template <int TOPK>
__launch_bounds__(32 * kWarpsPerCta) __global__ void topkSoftmax128(
    const uint4* __restrict__ input,  // [num_rows, 128] bf16
    float* __restrict__ out_w,        // [num_rows, TOPK]
    int* __restrict__ out_i,          // [num_rows, TOPK]
    int num_rows) {
  constexpr int TPR = kThreadsPerRow;
  constexpr int ROWS_PER_WARP = 32 / TPR;
  constexpr int GROUPS = kExperts / (8 * TPR);  // 16B loads per thread

  const int lane = threadIdx.x & 31;
  const int tid = lane & (TPR - 1);
  const int rw = lane / TPR;
  const int row = (blockIdx.x * kWarpsPerCta + (int)(threadIdx.x >> 5)) *
                      ROWS_PER_WARP + rw;
  if (row >= num_rows) return;
  // All TPR lanes of a row group share `row`, so they exit together.
  const unsigned gmask = (unsigned)((1ull << TPR) - 1ull) << (rw * TPR);

  // Interleaved 16B loads: the TPR lanes of a row sweep it GROUPS times, so
  // every load instruction touches whole 128B lines.
  const uint4* rp = input + row * (kExperts / 8);
  uint4 v[GROUPS];
#pragma unroll
  for (int g = 0; g < GROUPS; ++g) v[g] = rp[g * TPR + tid];

  // 127 - expert splits into disjoint bit fields:
  //   (GROUPS-1-g) << (3+log2 TPR) | (TPR-1-tid) << 3 | (7-u)
  const uint32_t tbase = (uint32_t)((TPR - 1 - tid) << 3);
  uint32_t x[8], y[8];
#pragma unroll
  for (int g = 0; g < GROUPS; ++g) {
    const uint32_t pb =
        tbase | ((uint32_t)(GROUPS - 1 - g) << (3 + ilog2(TPR)));
    if (g == 0) {
      pack8(x, v[0], pb);
      sort8_desc(x);
    } else {
      pack8(y, v[g], pb);
      sort8_desc(y);
      merge_top8(x, y);
    }
  }

  // Butterfly of "merge two sorted top-8 lists": after log2(TPR) levels every
  // lane of the group holds the row's sorted top-8. The 8 shuffles of a level
  // are independent, so the dependency chain is only log2(TPR) deep.
#pragma unroll
  for (int lvl = 1; lvl < TPR; lvl <<= 1) {
    uint32_t z[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) z[i] = __shfl_xor_sync(gmask, x[i], lvl);
    merge_top8(x, z);
  }

  // Softmax over the k selected logits only; lane j keeps the j-th largest.
  const uint32_t mine = pick8(x, tid);
  const float mx = key_to_float(x[0] >> 16);
  float e = (tid < TOPK) ? __expf(key_to_float(mine >> 16) - mx) : 0.0f;
  float s = e;
#pragma unroll
  for (int m = TPR / 2; m > 0; m >>= 1) s += __shfl_xor_sync(gmask, s, m);

  if (tid < TOPK) {
    const int o = row * TOPK + tid;
    out_w[o] = e * __frcp_rn(s);
    out_i[o] = 127 - (int)(mine & 0xFFu);
  }
}

template <int TOPK>
void launch(const void* in, float* w, int* ids, int num_rows, cudaStream_t s) {
  constexpr int ROWS_PER_CTA = (32 / kThreadsPerRow) * kWarpsPerCta;
  const int blocks = (num_rows + ROWS_PER_CTA - 1) / ROWS_PER_CTA;
  topkSoftmax128<TOPK><<<blocks, 32 * kWarpsPerCta, 0, s>>>(
      reinterpret_cast<const uint4*>(in), w, ids, num_rows);
}

}  // namespace

// Writes nothing and returns false when the inputs fall outside the fast path,
// leaving the caller to fall back.
bool topk_softmax_fast(
    torch::Tensor& topk_weights,
    torch::Tensor& topk_indices,
    const torch::Tensor& gating_output,
    bool renormalize) {
  if (!renormalize) return false;
  if (gating_output.scalar_type() != at::ScalarType::BFloat16) return false;
  if (gating_output.dim() != 2 || gating_output.size(1) != kExperts) return false;
  if (!gating_output.is_contiguous()) return false;
  const int topk = static_cast<int>(topk_weights.size(-1));
  if (topk < 1 || topk > kThreadsPerRow) return false;
  const int num_rows = static_cast<int>(gating_output.size(0));
  if (num_rows <= 0) return false;
  if (topk_weights.size(0) < num_rows || topk_indices.size(0) < num_rows) return false;
  if (topk_weights.scalar_type() != at::ScalarType::Float) return false;
  if (topk_indices.scalar_type() != at::ScalarType::Int) return false;
  if (!topk_weights.is_contiguous() || !topk_indices.is_contiguous()) return false;
  const void* in = gating_output.data_ptr();
  if ((reinterpret_cast<uintptr_t>(in) & 15u) != 0) return false;

  const at::cuda::OptionalCUDAGuard guard(device_of(gating_output));
  const cudaStream_t st = at::cuda::getCurrentCUDAStream();
  float* w = topk_weights.data_ptr<float>();
  int* ids = topk_indices.data_ptr<int>();
  switch (topk) {
    case 8: launch<8>(in, w, ids, num_rows, st); break;
    case 1: launch<1>(in, w, ids, num_rows, st); break;
    case 2: launch<2>(in, w, ids, num_rows, st); break;
    case 3: launch<3>(in, w, ids, num_rows, st); break;
    case 4: launch<4>(in, w, ids, num_rows, st); break;
    case 5: launch<5>(in, w, ids, num_rows, st); break;
    case 6: launch<6>(in, w, ids, num_rows, st); break;
    case 7: launch<7>(in, w, ids, num_rows, st); break;
    default: return false;
  }
  return true;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("topk_softmax_fast", &topk_softmax_fast,
        "Fused top-k softmax routing (bf16, 128 experts, k<=8, renormalised)");
}
