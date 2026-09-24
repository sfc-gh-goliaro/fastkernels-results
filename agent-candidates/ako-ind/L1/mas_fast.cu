// Bandwidth-saturating streaming merge of two attention partitions.
//
// The workload is pure HBM streaming: for the hot shape (bf16 [16384, 16, 128])
// it reads 128 MiB of prefix/suffix output plus 2 MiB of LSE and writes 64 MiB,
// so the only thing that matters is getting close to achievable DRAM bandwidth.
// Against that, the vLLM kernel leaves three things on the table:
//
//   1. One 16-byte pack per thread (4.19M threads in 32K blocks) with runtime
//      integer div/mod for the index math and almost no memory-level
//      parallelism per thread.  Here a thread owns VEC 32-byte packs whose
//      loads are all issued before the first use, and the pack -> (token, head)
//      mapping is a shift because the packs-per-head count is a compile-time
//      power of two.
//   2. All 16 threads covering one (token, head) load prefix_lse/suffix_lse
//      independently, at a 64 KiB stride between heads.  Here a block covers a
//      whole TOK x num_heads tile, loads each LSE pair exactly once with the
//      token index varying fastest (so the reads are contiguous), and passes
//      the two merge weights to the data threads through shared memory.
//   3. Nothing this kernel reads is ever reused, so the 128 MiB input stream is
//      loaded with .L2::evict_first and leaves L2 to the output stream.
//
// Three further things matter once the kernel is timed the way the harness
// times it -- inside a window that also contains 194 MiB of input copies, so
// the inputs are partly L2-resident and memory is *faster* than cold, which
// leaves the arithmetic relatively more exposed:
//
//   4. The 32-byte output-stream loads are issued *before* the LSE phase, not
//      after it.  Otherwise every block serialises LSE load -> expf/logf ->
//      __syncthreads -> output loads, and that exposed prologue is worth 2.0 us
//      per call on the hot shape (in-window 113.7 -> 111.7 us, which is the
//      measured floor of a same-traffic kernel that does no merge at all).
//   5. The merge runs on sm_100's packed dual-fp32 ops (mul.f32x2 /
//      fma.rn.f32x2): each lane is an independent round-to-nearest fp32 op, so
//      the result is bit-identical, at 56 instead of 72 instructions per pack.
//   6. output_lse ownership is decoupled from the output tile.  A block owns
//      TILE_HEADS *flat* consecutive entries of the [head, token] LSE plane, so
//      every sector is written whole by exactly one block, instead of each block
//      writing a 16-byte fragment per head at 64 KiB stride and sharing every
//      sector with a neighbour.  Worth another 2.0 us on the output_lse shape.
//
// 128 threads x 4 packs beats 256 x 2 (same 512-pack tile) by ~2 us in-window
// on every shape, though the two are indistinguishable cold -- which is why r1
// picked 256 x 2 off a cold sweep.
//
// The tiled kernel covers the fp32/fp16/bf16, fully-contiguous, no-FP8,
// no-partial-prefix case; `merge_attn_states_fast` returns false for anything
// else and the caller falls back to the reference kernel.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <optional>

namespace mas {

// Blackwell can move 32 bytes per thread in a single instruction, so one warp
// covers a contiguous 1 KiB with one load.  Pre-sm_100 the same pack is two
// 128-bit accesses; the tiling and index math are unchanged.
struct alignas(32) Pack {
  uint64_t x0, x1, x2, x3;
};

__device__ __forceinline__ Pack ld_stream(const Pack* p) {
  Pack v;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  asm volatile("ld.global.nc.L2::evict_first.v4.b64 {%0,%1,%2,%3}, [%4];"
               : "=l"(v.x0), "=l"(v.x1), "=l"(v.x2), "=l"(v.x3)
               : "l"(p));
#else
  const uint4* q = reinterpret_cast<const uint4*>(p);
  uint4* d = reinterpret_cast<uint4*>(&v);
  d[0] = __ldg(q);
  d[1] = __ldg(q + 1);
#endif
  return v;
}

__device__ __forceinline__ void st_pack(Pack* p, const Pack& v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  asm volatile("st.global.v4.b64 [%0], {%1,%2,%3,%4};"
               :
               : "l"(p), "l"(v.x0), "l"(v.x1), "l"(v.x2), "l"(v.x3)
               : "memory");
#else
  uint4* q = reinterpret_cast<uint4*>(p);
  const uint4* s = reinterpret_cast<const uint4*>(&v);
  q[0] = s[0];
  q[1] = s[1];
#endif
}

// out = prefix * ps + suffix * ss, accumulated in fp32 exactly like the
// reference kernel, over the whole 32-byte pack.
template <typename scalar_t>
__device__ __forceinline__ Pack merge_pack(const Pack& a, const Pack& b,
                                           float ps, float ss);

template <>
__device__ __forceinline__ Pack merge_pack<__nv_bfloat16>(const Pack& a,
                                                          const Pack& b,
                                                          float ps, float ss) {
  Pack o;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  // bf16 -> fp32 is a shift/mask straight into a .b64 register pair (ptxas
  // turns the mov.b64 into register pairing, not a copy), then one mul.f32x2 +
  // one fma.rn.f32x2 per two elements.  Each lane of an f32x2 op is a separate
  // round-to-nearest fp32 op, so this is bit-identical to the scalar form
  // below: 16 LOP3 + 6 SHF + 10 IMAD + 8 FMUL2 + 8 FFMA2 + 8 F2FP per pack,
  // against 16 SHF + 16 PRMT + 16 FMUL + 16 FFMA + 8 F2FP.
  const uint32_t* A = reinterpret_cast<const uint32_t*>(&a);
  const uint32_t* B = reinterpret_cast<const uint32_t*>(&b);
  uint32_t* O = reinterpret_cast<uint32_t*>(&o);
  uint64_t ps2, ss2;
  asm("mov.b64 %0, {%1, %2};" : "=l"(ps2) : "f"(ps), "f"(ps));
  asm("mov.b64 %0, {%1, %2};" : "=l"(ss2) : "f"(ss), "f"(ss));
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    asm("{\n\t"
        ".reg .b32 al, ah, bl, bh, rl, rh;\n\t"
        ".reg .b64 fa, fb, t, r;\n\t"
        "shl.b32 al, %1, 16;\n\t"
        "and.b32 ah, %1, -65536;\n\t"
        "shl.b32 bl, %2, 16;\n\t"
        "and.b32 bh, %2, -65536;\n\t"
        "mov.b64 fa, {al, ah};\n\t"
        "mov.b64 fb, {bl, bh};\n\t"
        "mul.f32x2 t, fb, %4;\n\t"
        "fma.rn.f32x2 r, fa, %3, t;\n\t"
        "mov.b64 {rl, rh}, r;\n\t"
        "cvt.rn.bf16x2.f32 %0, rh, rl;\n\t"
        "}"
        : "=r"(O[i])
        : "r"(A[i]), "r"(B[i]), "l"(ps2), "l"(ss2));
  }
#else
  const __nv_bfloat162* A = reinterpret_cast<const __nv_bfloat162*>(&a);
  const __nv_bfloat162* B = reinterpret_cast<const __nv_bfloat162*>(&b);
  __nv_bfloat162* O = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float2 fa = __bfloat1622float2(A[i]);
    const float2 fb = __bfloat1622float2(B[i]);
    float2 r;
    r.x = fa.x * ps + fb.x * ss;
    r.y = fa.y * ps + fb.y * ss;
    O[i] = __float22bfloat162_rn(r);
  }
#endif
  return o;
}

template <>
__device__ __forceinline__ Pack merge_pack<__half>(const Pack& a, const Pack& b,
                                                   float ps, float ss) {
  Pack o;
  const __half2* A = reinterpret_cast<const __half2*>(&a);
  const __half2* B = reinterpret_cast<const __half2*>(&b);
  __half2* O = reinterpret_cast<__half2*>(&o);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float2 fa = __half22float2(A[i]);
    const float2 fb = __half22float2(B[i]);
    float2 r;
    r.x = fa.x * ps + fb.x * ss;
    r.y = fa.y * ps + fb.y * ss;
    O[i] = __float22half2_rn(r);
  }
  return o;
}

template <>
__device__ __forceinline__ Pack merge_pack<float>(const Pack& a, const Pack& b,
                                                  float ps, float ss) {
  Pack o;
  const float* A = reinterpret_cast<const float*>(&a);
  const float* B = reinterpret_cast<const float*>(&b);
  float* O = reinterpret_cast<float*>(&o);
#pragma unroll
  for (int i = 0; i < 8; ++i) O[i] = A[i] * ps + B[i] * ss;
  return o;
}

constexpr int BLOCK = 128;
constexpr int VEC = 4;  // 32-byte packs per thread; BLOCK * VEC = tile packs

// The merged log-sum-exp of one (token, head), plus the two weights that
// produce the merged output.  ss == 0 && ps == 1 also encodes "emit prefix
// unchanged", which is the both-LSE-are--inf case.
struct Weights {
  float ps, ss, lse;
};

__device__ __forceinline__ Weights merge_weights(float p, float s) {
  Weights w;
  w.ps = 1.0f;
  w.ss = 0.0f;
  p = isinf(p) ? -INFINITY : p;
  s = isinf(s) ? -INFINITY : s;
  const float m = fmaxf(p, s);
  w.lse = m;
  // isinf(m) here means both partitions are empty (MLA chunked prefill can
  // produce p == s == -inf); the reference emits prefix_output and -inf.
  if (!isinf(m)) {
    const float pe = expf(p - m);
    const float se = expf(s - m);
    const float tot = pe + se;
    w.ps = pe / tot;
    w.ss = se / tot;
    w.lse = logf(tot) + m;
  }
  return w;
}

// PACKS_PER_HEAD = head_size * sizeof(scalar_t) / 32, a compile-time power of
// two, so pack index -> head index is a shift.
template <typename scalar_t, int PACKS_PER_HEAD, bool HAS_OUTPUT_LSE>
__global__ __launch_bounds__(BLOCK) void merge_tiled_kernel(
    Pack* __restrict__ output, float* __restrict__ output_lse,
    const Pack* __restrict__ prefix_output, const float* __restrict__ prefix_lse,
    const Pack* __restrict__ suffix_output, const float* __restrict__ suffix_lse,
    const int num_tokens, const int num_heads, const int tokens_per_tile) {
  constexpr int TILE_PACKS = BLOCK * VEC;
  constexpr int TILE_HEADS = TILE_PACKS / PACKS_PER_HEAD;
  constexpr int LOG_PPH = PACKS_PER_HEAD == 1    ? 0
                          : PACKS_PER_HEAD == 2  ? 1
                          : PACKS_PER_HEAD == 4  ? 2
                          : PACKS_PER_HEAD == 8  ? 3
                                                 : 4;
  __shared__ float s_ps[TILE_HEADS];
  __shared__ float s_ss[TILE_HEADS];

  const int t0 = blockIdx.x * tokens_per_tile;
  const int64_t base = (int64_t)blockIdx.x * TILE_PACKS + threadIdx.x;
  const int64_t total = (int64_t)num_tokens * num_heads * PACKS_PER_HEAD;

  // Get the output stream in flight first.  Everything below it -- the LSE
  // loads, the transcendentals, the barrier -- is a dependent chain that would
  // otherwise delay these loads by its whole latency in every block.
  Pack pa[VEC], pb[VEC];
#pragma unroll
  for (int k = 0; k < VEC; ++k) {
    const int64_t i = base + (int64_t)k * BLOCK;
    if (i < total) {
      pa[k] = ld_stream(prefix_output + i);
      pb[k] = ld_stream(suffix_output + i);
    }
  }

  // One LSE pair per (token, head) of this tile, token index varying fastest so
  // the reads are contiguous instead of one 32-byte sector per scalar.
  for (int j = threadIdx.x; j < TILE_HEADS; j += BLOCK) {
    const int h = j / tokens_per_tile;
    const int dt = j - h * tokens_per_tile;
    const int t = t0 + dt;
    Weights w{1.0f, 0.0f, 0.0f};
    if (t < num_tokens) w = merge_weights(prefix_lse[h * num_tokens + t],
                                          suffix_lse[h * num_tokens + t]);
    s_ps[dt * num_heads + h] = w.ps;
    s_ss[dt * num_heads + h] = w.ss;
  }

  // output_lse ownership is *flat*, not tile-shaped: block b owns entries
  // [b * TILE_HEADS, (b+1) * TILE_HEADS) of the [head, token] plane, which is
  // TILE_HEADS contiguous floats no other block touches.  grid * TILE_HEADS >=
  // num_heads * num_tokens always, so every entry is written exactly once.
  // Costs one extra (contiguous) LSE pair load per entry and duplicates the
  // exp/log; buys full-sector single-owner stores.  Sits on the block's later
  // warps when it fits, so it overlaps the weight loop above.
  if (HAS_OUTPUT_LSE) {
    constexpr int OFF = (2 * TILE_HEADS <= BLOCK) ? TILE_HEADS : 0;
    const int64_t total_lse = (int64_t)num_heads * num_tokens;
    for (int j = (int)threadIdx.x - OFF; j >= 0 && j < TILE_HEADS; j += BLOCK) {
      const int64_t fi = (int64_t)blockIdx.x * TILE_HEADS + j;
      if (fi < total_lse) {
        output_lse[fi] = merge_weights(prefix_lse[fi], suffix_lse[fi]).lse;
      }
    }
  }

  __syncthreads();

#pragma unroll
  for (int k = 0; k < VEC; ++k) {
    const int64_t i = base + (int64_t)k * BLOCK;
    if (i < total) {
      const int hl = (threadIdx.x >> LOG_PPH) + (BLOCK >> LOG_PPH) * k;
      const float ps = s_ps[hl], ss = s_ss[hl];
      st_pack(output + i, (ss == 0.0f && ps == 1.0f)
                              ? pa[k]
                              : merge_pack<scalar_t>(pa[k], pb[k], ps, ss));
    }
  }
}

template <typename scalar_t, int PACKS_PER_HEAD>
void launch(torch::Tensor& output, float* output_lse_ptr,
            const torch::Tensor& prefix_output, const torch::Tensor& prefix_lse,
            const torch::Tensor& suffix_output, const torch::Tensor& suffix_lse,
            int num_tokens, int num_heads, int tokens_per_tile) {
  const int grid = (num_tokens + tokens_per_tile - 1) / tokens_per_tile;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* out = reinterpret_cast<Pack*>(output.data_ptr());
  const auto* po = reinterpret_cast<const Pack*>(prefix_output.data_ptr());
  const auto* so = reinterpret_cast<const Pack*>(suffix_output.data_ptr());
  const auto* pl = reinterpret_cast<const float*>(prefix_lse.data_ptr());
  const auto* sl = reinterpret_cast<const float*>(suffix_lse.data_ptr());
  if (output_lse_ptr != nullptr) {
    merge_tiled_kernel<scalar_t, PACKS_PER_HEAD, true><<<grid, BLOCK, 0, stream>>>(
        out, output_lse_ptr, po, pl, so, sl, num_tokens, num_heads,
        tokens_per_tile);
  } else {
    merge_tiled_kernel<scalar_t, PACKS_PER_HEAD, false><<<grid, BLOCK, 0, stream>>>(
        out, nullptr, po, pl, so, sl, num_tokens, num_heads, tokens_per_tile);
  }
}

template <typename scalar_t>
bool launch_by_packs(torch::Tensor& output, float* output_lse_ptr,
                     const torch::Tensor& prefix_output,
                     const torch::Tensor& prefix_lse,
                     const torch::Tensor& suffix_output,
                     const torch::Tensor& suffix_lse, int num_tokens,
                     int num_heads, int packs_per_head) {
#define MAS_CASE(PPH)                                                        \
  case PPH: {                                                                \
    constexpr int tile_heads = BLOCK * VEC / (PPH);                          \
    if (tile_heads % num_heads != 0) return false;                           \
    launch<scalar_t, PPH>(output, output_lse_ptr, prefix_output, prefix_lse,  \
                          suffix_output, suffix_lse, num_tokens, num_heads,   \
                          tile_heads / num_heads);                            \
    return true;                                                             \
  }
  switch (packs_per_head) {
    MAS_CASE(1)
    MAS_CASE(2)
    MAS_CASE(4)
    MAS_CASE(8)
    MAS_CASE(16)
    default:
      return false;
  }
#undef MAS_CASE
}

bool is_packed_3d(const torch::Tensor& t) {
  return t.dim() == 3 && t.is_contiguous();
}

bool is_lse(const std::optional<torch::Tensor>& t, int num_heads,
            int num_tokens) {
  if (!t.has_value()) return false;
  const torch::Tensor& x = t.value();
  return x.scalar_type() == torch::kFloat32 && x.dim() == 2 &&
         x.is_contiguous() && x.size(0) == num_heads && x.size(1) == num_tokens;
}

}  // namespace mas

// Returns true when the tiled kernel handled the call.  Returns false -- having
// launched nothing -- for FP8 output, non-contiguous or unaligned tensors,
// unsupported head sizes / head counts, or a partial-prefix split; the caller
// then dispatches the general reference path.
bool merge_attn_states_fast(
    torch::Tensor& output, std::optional<torch::Tensor> output_lse,
    const torch::Tensor& prefix_output, const torch::Tensor& prefix_lse,
    const torch::Tensor& suffix_output, const torch::Tensor& suffix_lse,
    std::optional<int64_t> prefill_tokens_with_context) {
  const auto dtype = prefix_output.scalar_type();
  if (dtype != torch::kBFloat16 && dtype != torch::kFloat16 &&
      dtype != torch::kFloat32)
    return false;
  if (output.scalar_type() != dtype || suffix_output.scalar_type() != dtype)
    return false;
  if (!prefix_output.is_cuda() || !mas::is_packed_3d(output) ||
      !mas::is_packed_3d(prefix_output) || !mas::is_packed_3d(suffix_output))
    return false;

  const int64_t num_tokens = output.size(0);
  const int64_t num_heads = output.size(1);
  const int64_t head_size = output.size(2);
  if (num_tokens <= 0 || num_heads <= 0 || num_heads > INT32_MAX ||
      num_tokens > INT32_MAX)
    return false;
  if (prefix_output.size(0) != num_tokens || prefix_output.size(1) != num_heads ||
      prefix_output.size(2) != head_size || suffix_output.size(0) != num_tokens ||
      suffix_output.size(1) != num_heads || suffix_output.size(2) != head_size)
    return false;

  // prefill_tokens_with_context splits the batch into a merged prefix and a
  // copy-from-suffix tail; only the all-merged case is specialized here.
  if (prefill_tokens_with_context.has_value() &&
      prefill_tokens_with_context.value() != num_tokens)
    return false;

  if (!mas::is_lse(prefix_lse, num_heads, num_tokens) ||
      !mas::is_lse(suffix_lse, num_heads, num_tokens))
    return false;
  float* output_lse_ptr = nullptr;
  if (output_lse.has_value()) {
    if (!mas::is_lse(output_lse, num_heads, num_tokens)) return false;
    output_lse_ptr = reinterpret_cast<float*>(output_lse.value().data_ptr());
  }

  const int64_t head_bytes = head_size * prefix_output.element_size();
  if (head_bytes % 32 != 0) return false;
  const int64_t packs_per_head = head_bytes / 32;
  // 32-byte vector accesses need 32-byte alignment.
  const torch::Tensor* packed[3] = {&output, &prefix_output, &suffix_output};
  for (const torch::Tensor* t : packed) {
    if (reinterpret_cast<uintptr_t>(t->data_ptr()) % 32 != 0) return false;
  }

  const c10::cuda::CUDAGuard guard(prefix_output.device());
  const int nt = static_cast<int>(num_tokens);
  const int nh = static_cast<int>(num_heads);
  const int pph = static_cast<int>(packs_per_head);
  switch (dtype) {
    case torch::kBFloat16:
      return mas::launch_by_packs<__nv_bfloat16>(output, output_lse_ptr,
                                                 prefix_output, prefix_lse,
                                                 suffix_output, suffix_lse, nt,
                                                 nh, pph);
    case torch::kFloat16:
      return mas::launch_by_packs<__half>(output, output_lse_ptr, prefix_output,
                                          prefix_lse, suffix_output, suffix_lse,
                                          nt, nh, pph);
    case torch::kFloat32:
      return mas::launch_by_packs<float>(output, output_lse_ptr, prefix_output,
                                         prefix_lse, suffix_output, suffix_lse,
                                         nt, nh, pph);
    default:
      return false;
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("merge_attn_states_fast", &merge_attn_states_fast,
        "Tiled streaming merge_attn_states fast path (returns false if the "
        "call is not supported and nothing was launched)",
        py::arg("output"), py::arg("output_lse"), py::arg("prefix_output"),
        py::arg("prefix_lse"), py::arg("suffix_output"), py::arg("suffix_lse"),
        py::arg("prefill_tokens_with_context"));
}
