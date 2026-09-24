// Fast BF16 ("auto" layout) MLA KV-cache store / gather kernels.
//
// With ``kv_cache_dtype="auto"`` the MLA paged cache is BF16
// ``[num_blocks, block_size, 576]`` and both operations degenerate to a pure
// 1152-byte-per-token copy:
//
//   store : kv_c_normed[t, 0:512] ++ k_pe[t, 0:64]  ->  cache[slot(t), :]
//   gather: cache[slot(token), :]                   ->  workspace[token, :]
//
// The vendored vLLM kernels are latency- rather than bandwidth-bound on that
// path: ``concat_and_cache_mla_kernel`` moves *two bytes* per thread from a
// 512-thread CTA per token (and leaves 7/8 of those threads idle on the k_pe
// half), and ``gather_and_maybe_dequant_cache`` launches one 64-thread CTA per
// token, so a 65k-token gather costs 65k CTA dispatches.
//
// Here one warp owns a whole token: every metadata lookup (slot_mapping /
// token_to_seq / cu_seq_lens / block_table) is warp-uniform, all 72 accesses
// per token are full 128-bit transactions, and the grid is sized to the GPU
// with a grid-stride loop over tokens instead of to the token count.
// Measured on a B200 (13.5 GB cache, L2 flushed between iterations):
// store of 16384 tokens 83.0 -> 17.4 us, gather of 65536 tokens 66.5 -> 39.9 us.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace fk_mla {

using vec_t = uint4;  // 16B == 8 bf16 elements

constexpr int kStoreBlock = 128;   // 4 warps -> 4 tokens in flight per CTA
constexpr int kGatherBlock = 256;  // 8 warps -> 8 tokens in flight per CTA
constexpr int kMaxBlocksPerSM = 32;

// 576 BF16 elements per cache entry == 72 vectors == 64 (kv_c) + 8 (k_pe).
constexpr int kEntryVecs = 72;
constexpr int kNopeVecs = 64;
constexpr int kPeVecs = 8;

__device__ __forceinline__ int64_t slot_to_offset(int64_t slot, int block_size,
                                                 int bs_shift, int64_t blk_v,
                                                 int entry_v) {
  int64_t blk, off;
  if (bs_shift >= 0) {  // power-of-two block size (64 in practice)
    blk = slot >> bs_shift;
    off = slot & ((int64_t(1) << bs_shift) - 1);
  } else {
    blk = slot / block_size;
    off = slot - blk * block_size;
  }
  return blk * blk_v + off * entry_v;
}

// ---------------------------------------------------------------------------
// STORE: one warp per token.  NOPE_V vectors come from kv_c (contiguous rows),
// PE_V from k_pe (rows may be strided); both land in one cache entry.
// ---------------------------------------------------------------------------
template <int BLOCK, int NOPE_V, int PE_V>
__global__ __launch_bounds__(BLOCK) void store_wpt(
    const vec_t* __restrict__ kv_c, const vec_t* __restrict__ k_pe,
    vec_t* __restrict__ cache, const int64_t* __restrict__ slots, int n_tokens,
    int kvc_v, int kpe_v, int64_t blk_v, int entry_v, int block_size,
    int bs_shift) {
  constexpr int WPB = BLOCK / 32;
  constexpr int NR = NOPE_V / 32;  // 32-lane rounds over the kv_c half
  static_assert(NOPE_V % 32 == 0 && PE_V <= 32, "unsupported entry split");
  const int lane = threadIdx.x & 31;
  const bool pe_act = lane < PE_V;
  const int64_t step = (int64_t)gridDim.x * WPB;

  for (int64_t tok = (int64_t)blockIdx.x * WPB + (threadIdx.x >> 5);
       tok < n_tokens; tok += step) {
    const int64_t slot = slots[tok];
    if (slot < 0) continue;  // padded token
    const int64_t d = slot_to_offset(slot, block_size, bs_shift, blk_v, entry_v);
    const vec_t* sn = kv_c + tok * (int64_t)kvc_v + lane;
    vec_t a[NR];
#pragma unroll
    for (int k = 0; k < NR; ++k) a[k] = __ldg(sn + 32 * k);
    vec_t p;
    if (pe_act) p = __ldg(k_pe + tok * (int64_t)kpe_v + lane);
#pragma unroll
    for (int k = 0; k < NR; ++k) cache[d + lane + 32 * k] = a[k];
    if (pe_act) cache[d + NOPE_V + lane] = p;
  }
}

// Element-wise store fallback (entry sizes / alignments the vector path cannot
// take: fp32 caches, non-576 entries, unaligned strides).
template <typename T, int BLOCK>
__global__ __launch_bounds__(BLOCK) void store_generic(
    const T* __restrict__ kv_c, const T* __restrict__ k_pe,
    T* __restrict__ cache, const int64_t* __restrict__ slots, int n_tokens,
    int lora, int pe_dim, int kvc_s, int kpe_s, int64_t blk_s, int entry_s,
    int block_size) {
  constexpr int WPB = BLOCK / 32;
  const int lane = threadIdx.x & 31;
  for (int64_t tok = (int64_t)blockIdx.x * WPB + (threadIdx.x >> 5);
       tok < n_tokens; tok += (int64_t)gridDim.x * WPB) {
    const int64_t slot = slots[tok];
    if (slot < 0) continue;
    const int64_t d =
        (slot / block_size) * blk_s + (slot % block_size) * (int64_t)entry_s;
    for (int i = lane; i < lora; i += 32)
      cache[d + i] = kv_c[tok * (int64_t)kvc_s + i];
    for (int i = lane; i < pe_dim; i += 32)
      cache[d + lora + i] = k_pe[tok * (int64_t)kpe_s + i];
  }
}

// ---------------------------------------------------------------------------
// GATHER: one warp per token (ENTRY_V vectors each).
// ---------------------------------------------------------------------------
template <int BLOCK, int ENTRY_V>
__global__ __launch_bounds__(BLOCK) void gather_wpt(
    const vec_t* __restrict__ cache, vec_t* __restrict__ dst,
    const int32_t* __restrict__ block_table, const int32_t* __restrict__ cu,
    const int32_t* __restrict__ t2s, const int32_t* __restrict__ starts,
    int n_tokens, int bt_stride, int64_t blk_v, int entry_v, int dst_v,
    int block_size, int bs_shift) {
  constexpr int WPB = BLOCK / 32;
  constexpr int NR = ENTRY_V / 32;    // full 32-lane rounds
  constexpr int TAIL = ENTRY_V % 32;  // partial round
  const int lane = threadIdx.x & 31;
  const bool tail_act = lane < TAIL;
  const int64_t step = (int64_t)gridDim.x * WPB;

  for (int64_t tok = (int64_t)blockIdx.x * WPB + (threadIdx.x >> 5);
       tok < n_tokens; tok += step) {
    const int b = t2s[tok];
    if (tok >= cu[b + 1]) continue;  // token past its sequence
    const int off =
        (int)(tok - cu[b]) + (starts != nullptr ? starts[b] : 0);
    int bti, slot;
    if (bs_shift >= 0) {
      bti = off >> bs_shift;
      slot = off & ((1 << bs_shift) - 1);
    } else {
      bti = off / block_size;
      slot = off - bti * block_size;
    }
    const int blk = block_table[(int64_t)b * bt_stride + bti];
    const vec_t* s =
        cache + (int64_t)blk * blk_v + (int64_t)slot * entry_v + lane;
    vec_t* d = dst + tok * (int64_t)dst_v + lane;
    vec_t a[NR];
#pragma unroll
    for (int k = 0; k < NR; ++k) a[k] = __ldg(s + 32 * k);
    vec_t t;
    if (TAIL && tail_act) t = __ldg(s + 32 * NR);
#pragma unroll
    for (int k = 0; k < NR; ++k) d[32 * k] = a[k];
    if (TAIL && tail_act) d[32 * NR] = t;
  }
}

// Element-wise gather fallback (any entry size / alignment).
template <typename T, int BLOCK>
__global__ __launch_bounds__(BLOCK) void gather_generic(
    const T* __restrict__ cache, T* __restrict__ dst,
    const int32_t* __restrict__ block_table, const int32_t* __restrict__ cu,
    const int32_t* __restrict__ t2s, const int32_t* __restrict__ starts,
    int n_tokens, int entry_size, int bt_stride, int64_t blk_s, int entry_s,
    int dst_s, int block_size) {
  constexpr int WPB = BLOCK / 32;
  const int lane = threadIdx.x & 31;
  for (int64_t tok = (int64_t)blockIdx.x * WPB + (threadIdx.x >> 5);
       tok < n_tokens; tok += (int64_t)gridDim.x * WPB) {
    const int b = t2s[tok];
    if (tok >= cu[b + 1]) continue;
    const int off = (int)(tok - cu[b]) + (starts != nullptr ? starts[b] : 0);
    const int blk = block_table[(int64_t)b * bt_stride + off / block_size];
    const T* s =
        cache + (int64_t)blk * blk_s + (int64_t)(off % block_size) * entry_s;
    T* d = dst + tok * (int64_t)dst_s;
    for (int i = lane; i < entry_size; i += 32) d[i] = s[i];
  }
}

// ---------------------------------------------------------------------------
// host side
// ---------------------------------------------------------------------------
static int sm_count() {
  static int n = -1;
  if (n < 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
  }
  return n;
}

static inline int pow2_shift(int v) {
  if (v <= 0 || (v & (v - 1)) != 0) return -1;
  int s = 0;
  while ((1 << s) < v) ++s;
  return s;
}

// One warp per token, capped so the grid stays proportional to the GPU.
static inline int grid_for(int64_t n_tokens, int block) {
  const int64_t wpb = block / 32;
  const int64_t blocks = std::min<int64_t>((n_tokens + wpb - 1) / wpb,
                                          (int64_t)kMaxBlocksPerSM * sm_count());
  return (int)std::max<int64_t>(blocks, 1);
}

static inline bool aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15) == 0;
}

void store_mla(const at::Tensor& kv_c, const at::Tensor& k_pe,
               at::Tensor& cache, const at::Tensor& slots) {
  const int n = (int)slots.size(0);
  if (n == 0) return;
  TORCH_CHECK(kv_c.is_cuda() && k_pe.is_cuda() && cache.is_cuda() &&
              slots.is_cuda(), "store_mla: all tensors must be CUDA");
  TORCH_CHECK(slots.scalar_type() == at::kLong, "slot_mapping must be int64");
  TORCH_CHECK(kv_c.scalar_type() == k_pe.scalar_type() &&
              kv_c.scalar_type() == cache.scalar_type(),
              "store_mla: dtype mismatch");
  const int lora = (int)kv_c.size(1);
  const int pe_dim = (int)k_pe.size(1);
  const int block_size = (int)cache.size(1);
  TORCH_CHECK(cache.size(2) == lora + pe_dim, "store_mla: entry size mismatch");
  TORCH_CHECK(kv_c.stride(1) == 1 && k_pe.stride(1) == 1 && cache.stride(2) == 1,
              "store_mla: innermost stride must be 1");
  const int esz = (int)kv_c.element_size();
  const int64_t kvc_s = kv_c.stride(0), kpe_s = k_pe.stride(0);
  const int64_t blk_s = cache.stride(0), entry_s = cache.stride(1);

  const c10::cuda::CUDAGuard guard(kv_c.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int epv = 16 / esz;  // elements per 16B vector

  const bool vec_ok = esz == 2 && lora == kNopeVecs * 8 && pe_dim == kPeVecs * 8 &&
                      kvc_s % epv == 0 && kpe_s % epv == 0 &&
                      blk_s % epv == 0 && entry_s % epv == 0 &&
                      aligned16(kv_c.data_ptr()) && aligned16(k_pe.data_ptr()) &&
                      aligned16(cache.data_ptr());

  if (vec_ok) {
    store_wpt<kStoreBlock, kNopeVecs, kPeVecs>
        <<<grid_for(n, kStoreBlock), kStoreBlock, 0, stream>>>(
            (const vec_t*)kv_c.data_ptr(), (const vec_t*)k_pe.data_ptr(),
            (vec_t*)cache.data_ptr(), slots.data_ptr<int64_t>(), n,
            (int)(kvc_s / epv), (int)(kpe_s / epv), blk_s / epv,
            (int)(entry_s / epv), block_size, pow2_shift(block_size));
    return;
  }
  constexpr int BLOCK = 256;
  const int grid = grid_for(n, BLOCK);
  if (esz == 2) {
    store_generic<uint16_t, BLOCK><<<grid, BLOCK, 0, stream>>>(
        (const uint16_t*)kv_c.data_ptr(), (const uint16_t*)k_pe.data_ptr(),
        (uint16_t*)cache.data_ptr(), slots.data_ptr<int64_t>(), n, lora, pe_dim,
        (int)kvc_s, (int)kpe_s, blk_s, (int)entry_s, block_size);
  } else {
    TORCH_CHECK(esz == 4, "store_mla: unsupported element size ", esz);
    store_generic<uint32_t, BLOCK><<<grid, BLOCK, 0, stream>>>(
        (const uint32_t*)kv_c.data_ptr(), (const uint32_t*)k_pe.data_ptr(),
        (uint32_t*)cache.data_ptr(), slots.data_ptr<int64_t>(), n, lora, pe_dim,
        (int)kvc_s, (int)kpe_s, blk_s, (int)entry_s, block_size);
  }
}

void gather_mla(const at::Tensor& cache, at::Tensor& dst,
                const at::Tensor& block_table, const at::Tensor& cu_seq_lens,
                const at::Tensor& token_to_seq, int64_t num_tokens,
                const c10::optional<at::Tensor>& seq_starts) {
  const int n = (int)num_tokens;
  if (n == 0) return;
  TORCH_CHECK(cache.is_cuda() && dst.is_cuda(),
              "gather_mla: all tensors must be CUDA");
  TORCH_CHECK(block_table.scalar_type() == at::kInt &&
              cu_seq_lens.scalar_type() == at::kInt &&
              token_to_seq.scalar_type() == at::kInt,
              "gather_mla: metadata must be int32");
  TORCH_CHECK(cache.scalar_type() == dst.scalar_type(),
              "gather_mla: dtype mismatch");
  const int entry_size = (int)dst.size(-1);
  const int block_size = (int)cache.size(1);
  TORCH_CHECK(cache.size(2) == entry_size, "gather_mla: entry size mismatch");
  TORCH_CHECK(cache.stride(2) == 1 && dst.stride(1) == 1,
              "gather_mla: innermost stride must be 1");
  const int esz = (int)dst.element_size();
  const int64_t blk_s = cache.stride(0), entry_s = cache.stride(1);
  const int64_t dst_s = dst.stride(0);

  const c10::cuda::CUDAGuard guard(cache.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int32_t* starts =
      seq_starts.has_value() ? seq_starts->data_ptr<int32_t>() : nullptr;
  const int bt_stride = (int)block_table.stride(0);
  const int epv = 16 / esz;

  const bool vec_ok = esz == 2 && entry_size == kEntryVecs * 8 &&
                      blk_s % epv == 0 && entry_s % epv == 0 &&
                      dst_s % epv == 0 && aligned16(cache.data_ptr()) &&
                      aligned16(dst.data_ptr());

  if (vec_ok) {
    gather_wpt<kGatherBlock, kEntryVecs>
        <<<grid_for(n, kGatherBlock), kGatherBlock, 0, stream>>>(
            (const vec_t*)cache.data_ptr(), (vec_t*)dst.data_ptr(),
            block_table.data_ptr<int32_t>(), cu_seq_lens.data_ptr<int32_t>(),
            token_to_seq.data_ptr<int32_t>(), starts, n, bt_stride, blk_s / epv,
            (int)(entry_s / epv), (int)(dst_s / epv), block_size,
            pow2_shift(block_size));
    return;
  }
  constexpr int BLOCK = 256;
  const int grid = grid_for(n, BLOCK);
  TORCH_CHECK(esz == 2 || esz == 4, "gather_mla: unsupported element size ", esz);
  if (esz == 2) {
    gather_generic<uint16_t, BLOCK><<<grid, BLOCK, 0, stream>>>(
        (const uint16_t*)cache.data_ptr(), (uint16_t*)dst.data_ptr(),
        block_table.data_ptr<int32_t>(), cu_seq_lens.data_ptr<int32_t>(),
        token_to_seq.data_ptr<int32_t>(), starts, n, entry_size, bt_stride,
        blk_s, (int)entry_s, (int)dst_s, block_size);
  } else {
    gather_generic<uint32_t, BLOCK><<<grid, BLOCK, 0, stream>>>(
        (const uint32_t*)cache.data_ptr(), (uint32_t*)dst.data_ptr(),
        block_table.data_ptr<int32_t>(), cu_seq_lens.data_ptr<int32_t>(),
        token_to_seq.data_ptr<int32_t>(), starts, n, entry_size, bt_stride,
        blk_s, (int)entry_s, (int)dst_s, block_size);
  }
}

}  // namespace fk_mla

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("store_mla", &fk_mla::store_mla, "MLA KV store (BF16 cache)");
  m.def("gather_mla", &fk_mla::gather_mla, "MLA KV gather (BF16 cache)");
}
