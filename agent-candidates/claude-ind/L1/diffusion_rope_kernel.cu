// Interleaved (GPT-J style) rotary position embedding -- vectorized CUDA kernel.
//
// Fast-path layout (validated on the host side; anything else falls back to the
// baseline Triton kernel):
//   x        : (batch, seqlen, nheads, headdim) contiguous, bf16 / fp16 / fp32
//   cos, sin : (seqlen, rotary_dim/2) contiguous, same dtype as x
//
// One block owns a single sequence position and every (head, headdim) element of
// it.  A thread handles ITEMS packs of 16 bytes, spaced blockDim.x packs apart;
// blockDim.x is a multiple of headdim/PACK, so all ITEMS packs of a thread land
// on the *same* rotary-pair offsets and share one cos/sin fetch.  That keeps
// every global access 16 B wide and fully coalesced while cutting the cos/sin L1
// read traffic by ITEMS.  Stores are `st.global.cs` -- the output is never read
// again, so there is no reason to keep it resident in L2.
//
// Arithmetic is done in fp32 with round-to-nearest-even on the way back to the
// storage dtype, matching the Triton baseline bit for bit.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

// Per-dtype: 16-byte pack width, the 2-wide vector type holding one rotary
// pair, and the packed conversions to/from fp32.
template <typename T>
struct Traits;

template <>
struct Traits<__nv_bfloat16> {
  using Pair = __nv_bfloat162;
  static constexpr int kPack = 8;
  __device__ static float2 to_f2(Pair p) { return __bfloat1622float2(p); }
  __device__ static Pair from_f2(float2 f) { return __float22bfloat162_rn(f); }
};

template <>
struct Traits<__half> {
  using Pair = __half2;
  static constexpr int kPack = 8;
  __device__ static float2 to_f2(Pair p) { return __half22float2(p); }
  __device__ static Pair from_f2(float2 f) { return __float22half2_rn(f); }
};

template <>
struct Traits<float> {
  using Pair = float2;
  static constexpr int kPack = 4;
  __device__ static float2 to_f2(Pair p) { return p; }
  __device__ static Pair from_f2(float2 f) { return f; }
};

template <typename T, int N>
struct alignas(sizeof(T) * N) Vec {
  T v[N];
};

// ITEMS 16-byte packs per thread; FULL removes the rotary-tail check when the
// rotation covers the whole head dim.  vph_mask == headdim/PACK - 1.
template <typename T, int ITEMS, bool FULL>
__global__ void rope_interleaved_kernel(T *__restrict__ out, const T *__restrict__ x,
                                        const T *__restrict__ cosp,
                                        const T *__restrict__ sinp, int vec_per_row,
                                        int vph_mask, int rot_half) {
  using Tr = Traits<T>;
  using Pair = typename Tr::Pair;
  constexpr int PACK = Tr::kPack;
  constexpr int NPAIR = PACK / 2;
  using VecT = Vec<T, PACK>;

  const int tid = threadIdx.x;
  const int nt = blockDim.x;
  // First rotary-pair index this thread touches (identical for all its ITEMS,
  // because blockDim.x is a multiple of headdim/PACK).
  const int pair0 = (tid & vph_mask) * NPAIR;

  const size_t base = static_cast<size_t>(blockIdx.x) * vec_per_row + tid;
  const VecT *__restrict__ xv = reinterpret_cast<const VecT *>(x) + base;
  VecT *__restrict__ ov = reinterpret_cast<VecT *>(out) + base;

  VecT a[ITEMS];
#pragma unroll
  for (int u = 0; u < ITEMS; ++u) a[u] = xv[u * nt];

  if (!FULL && pair0 >= rot_half) {  // past rotary_dim: pass through
#pragma unroll
    for (int u = 0; u < ITEMS; ++u)
      __stcs(reinterpret_cast<float4 *>(ov + u * nt), *reinterpret_cast<float4 *>(&a[u]));
    return;
  }

  const size_t coff = static_cast<size_t>(blockIdx.x) * rot_half + pair0;
  const Pair *cp = reinterpret_cast<const Pair *>(cosp + coff);
  const Pair *sp = reinterpret_cast<const Pair *>(sinp + coff);
  float cf[NPAIR], sf[NPAIR];
#pragma unroll
  for (int j = 0; j < NPAIR; j += 2) {  // packed cvt: two scalars per instruction
    const float2 c2 = Tr::to_f2(cp[j / 2]);
    const float2 s2 = Tr::to_f2(sp[j / 2]);
    cf[j] = c2.x;
    cf[j + 1] = c2.y;
    sf[j] = s2.x;
    sf[j + 1] = s2.y;
  }

#pragma unroll
  for (int u = 0; u < ITEMS; ++u) {
    VecT o;
    const Pair *ap = reinterpret_cast<const Pair *>(a[u].v);
    Pair *op = reinterpret_cast<Pair *>(o.v);
#pragma unroll
    for (int j = 0; j < NPAIR; ++j) {
      const float2 p = Tr::to_f2(ap[j]);
      op[j] = Tr::from_f2(
          make_float2(p.x * cf[j] - p.y * sf[j], p.y * cf[j] + p.x * sf[j]));
    }
    __stcs(reinterpret_cast<float4 *>(ov + u * nt), *reinterpret_cast<float4 *>(&o));
  }
}

// blockDim closest to 128 that keeps every thread's packs on one rotary offset.
constexpr int kItemChoices[] = {1, 2, 3, 4, 6, 8, 12, 16, 24};

int pick_items(int vec_per_row, int vec_per_head) {
  int best = 0, best_score = 1 << 30;
  for (int it : kItemChoices) {
    if (vec_per_row % it) continue;
    const int bd = vec_per_row / it;
    if (bd > 1024 || bd % vec_per_head || bd % 32) continue;
    const int score = bd > 128 ? bd - 128 : 128 - bd;
    if (score < best_score) {
      best_score = score;
      best = it;
    }
  }
  return best;
}

template <typename T>
void launch(torch::Tensor &out, const torch::Tensor &x, const torch::Tensor &cos,
            const torch::Tensor &sin, int batch, int seqlen, int nheads, int headdim,
            int rot_half, int items) {
  constexpr int PACK = Traits<T>::kPack;
  const int vec_per_head = headdim / PACK;
  const int vec_per_row = nheads * vec_per_head;
  const int block = vec_per_row / items;
  const bool full = 2 * rot_half == headdim;
  auto stream = at::cuda::getCurrentCUDAStream();

  auto *op = reinterpret_cast<T *>(out.data_ptr());
  auto *xp = reinterpret_cast<const T *>(x.data_ptr());
  auto *cp = reinterpret_cast<const T *>(cos.data_ptr());
  auto *sp = reinterpret_cast<const T *>(sin.data_ptr());
  const size_t bstride = static_cast<size_t>(seqlen) * vec_per_row * PACK;

  // 1D grid (one block per sequence position) keeps the row index a bare
  // blockIdx.x; batch is looped on the host since it is 1 for every captured
  // shape and tiny otherwise.
  for (int b = 0; b < batch; ++b) {
#define LAUNCH(I, F)                                                              \
  rope_interleaved_kernel<T, I, F><<<seqlen, block, 0, stream>>>(                 \
      op + b * bstride, xp + b * bstride, cp, sp, vec_per_row, vec_per_head - 1,  \
      rot_half)
#define CASE(I)                                                                   \
  case I:                                                                         \
    if (full) LAUNCH(I, true);                                                     \
    else LAUNCH(I, false);                                                         \
    break;
    switch (items) {
      CASE(1) CASE(2) CASE(3) CASE(4) CASE(6) CASE(8) CASE(12) CASE(16) CASE(24)
      default:
        TORCH_CHECK(false, "rope: unreachable items=", items);
    }
#undef CASE
#undef LAUNCH
  }
}

}  // namespace

// Allocates and returns the output. Throws (-> Python RuntimeError) whenever the
// inputs fall outside the fast path so the caller can use the Triton fallback.
torch::Tensor rope_interleaved(const torch::Tensor &x, const torch::Tensor &cos,
                               const torch::Tensor &sin) {
  TORCH_CHECK(x.is_cuda() && cos.is_cuda() && sin.is_cuda(), "rope: cuda only");
  TORCH_CHECK(x.dim() == 4, "rope: need (b,s,h,d)");
  TORCH_CHECK(cos.dim() == 2 && sin.dim() == 2, "rope: need 2d cos/sin");
  TORCH_CHECK(x.is_contiguous() && cos.is_contiguous() && sin.is_contiguous(),
              "rope: contiguous only");
  TORCH_CHECK(cos.scalar_type() == x.scalar_type() && sin.scalar_type() == x.scalar_type(),
              "rope: dtype mismatch");

  const int batch = x.size(0);
  const int seqlen = x.size(1);
  const int nheads = x.size(2);
  const int headdim = x.size(3);
  const int rot_half = cos.size(1);
  // The kernel indexes cos/sin row s at s * rotary_dim/2, which is right for any
  // seqlen_ro >= seqlen (seqlen_offsets == 0 only).
  TORCH_CHECK(cos.size(0) >= seqlen && sin.size(0) >= seqlen, "rope: seqlen_ro < seqlen");
  TORCH_CHECK(sin.size(1) == rot_half, "rope: cos/sin shape mismatch");
  TORCH_CHECK(2 * rot_half <= headdim, "rope: rotary_dim > headdim");

  const int pack = x.scalar_type() == at::kFloat ? 4 : 8;
  TORCH_CHECK(headdim % pack == 0, "rope: headdim % pack");
  TORCH_CHECK(rot_half % (pack / 2) == 0, "rope: rotary_dim/2 % pairs");
  const int vec_per_head = headdim / pack;
  TORCH_CHECK((vec_per_head & (vec_per_head - 1)) == 0, "rope: headdim/pack not 2^k");
  const int items = pick_items(nheads * vec_per_head, vec_per_head);
  TORCH_CHECK(items > 0, "rope: no valid block shape");

  auto out = torch::empty_like(x);
  if (out.numel() == 0) return out;

  switch (x.scalar_type()) {
    case at::kBFloat16:
      launch<__nv_bfloat16>(out, x, cos, sin, batch, seqlen, nheads, headdim, rot_half,
                            items);
      break;
    case at::kHalf:
      launch<__half>(out, x, cos, sin, batch, seqlen, nheads, headdim, rot_half, items);
      break;
    case at::kFloat:
      launch<float>(out, x, cos, sin, batch, seqlen, nheads, headdim, rot_half, items);
      break;
    default:
      TORCH_CHECK(false, "rope: unsupported dtype");
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rope_interleaved", &rope_interleaved, "interleaved rotary embedding");
}
