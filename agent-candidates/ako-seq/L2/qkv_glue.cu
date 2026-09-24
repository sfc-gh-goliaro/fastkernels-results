// One fused pass over a packed QKV projection output: per-head RMSNorm, RoPE,
// weight-less QK norm and position-dependent temperature, in place, in the
// layout the downstream Attention call already views.
//
// Why CUDA and not Triton for this one: the shapes that matter most here are
// decode-sized (N == 1) and short prefills, where the whole LlamaAttention
// forward is *host*-bound -- ~226 us of wall time against ~45 us of device
// time -- so a launch is a first-class cost. Triton's per-launch specialization
// pass measured 13-15 us for this argument list; the same launch through a
// plain pybind entry point is ~4 us. The device-side structure is the same one
// a Triton version would emit: one block per token, cos/sin assembled once per
// token into shared memory, a fixed lane group per head row doing 128-bit
// vector loads and a shuffle-only norm reduction (no shared memory, no
// barrier).
//
// Rounding deliberately reproduces the reference stage-by-stage sequence rather
// than being as accurate as possible, so this is a drop-in replacement and not
// merely a close one:
//   * cos/sin are rounded to the activation dtype, because the reference casts
//     its whole fp32 table to that dtype and rotates with the result;
//   * the norm is fp32 with a single rounding, matching vLLM's
//     ``out = (scalar_t)(x * rstd * w)``, and that rounding lands here because
//     the reference writes the norm to memory before the rope launch reads it;
//   * M-RoPE products are rounded individually, because the reference M-RoPE is
//     a Triton kernel whose q/k and cos/sin tiles are both bf16 and Triton keeps
//     bf16 x bf16 in bf16, while the 1-D rope reference is a CUDA kernel that
//     computes in fp32 -- so the two rope families round differently and this
//     kernel follows each.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <algorithm>

namespace {

// Rotation modes, mirrored in the Python module.
constexpr int kRopeNone = 0;
constexpr int kRopeNeox = 1;       // pairs (i, i + head_dim/2)
constexpr int kRopeGptj = 2;       // pairs (2i, 2i + 1)
constexpr int kRopeMropeIl = 3;    // M-RoPE, interleaved sections
constexpr int kRopeMropeSec = 4;   // M-RoPE, contiguous sections
static_assert(kRopeNone == 0 && kRopeNeox < kRopeGptj
                  && kRopeMropeSec == kRopeMropeIl + 1,
              "mode encoding: the kernel tests `mode >= kRopeMropeIl` for the "
              "two M-RoPE forms and `mode != kRopeNone` for any rotation");

// Flag layout of the packed ``flags`` argument.
constexpr int kFlagNorm = 1 << 3;
constexpr int kFlagWlNorm = 1 << 4;
constexpr int kFlagTemp = 1 << 5;
constexpr int kFlagRopeRound = 1 << 6;
constexpr int kModeMask = 0x7;

// Which M-RoPE section (T/H/W) frequency index j belongs to. The interleaved
// form's ``j <= 3 * s`` (not ``<``) is the reference's, and it matters: at
// head_dim 128 with sections [24,20,20] it puts j = 61,62 back in T.
__device__ __forceinline__ int mrope_section(int j, int st, int sh, int sw,
                                             bool interleaved) {
  if (interleaved) {
    const int r = j % 3;
    if (r == 1 && j <= 3 * sh) return 1;
    if (r == 2 && j <= 3 * sw) return 2;
    return 0;
  }
  if (j < st) return 0;
  if (j < st + sh) return 1;
  return 2;
}

// A 16-byte load/store unit: VEC elements of T, moved as one 128-bit access.
template <typename T, int VEC>
struct alignas(16) Vec {
  T v[VEC];
};

template <typename T, int VEC>
__global__ void qkv_glue_kernel(
    T* __restrict__ qkv,
    const int64_t* __restrict__ pos,
    const float* __restrict__ cache,
    const T* __restrict__ qw,
    const T* __restrict__ kw,
    const int nq, const int nh, const int hd, const int half,
    const int64_t row_stride, const int64_t pos_s0, const int64_t pos_s1,
    const float eps, const float floor_scale, const float attn_scale,
    const int st, const int sh, const int sw,
    const int n_tok, const int flags, const int hpb, const int tpb) {
  using V = Vec<T, VEC>;
  extern __shared__ float smem[];  // [0, half) cos, [half, 2*half) sin

  const int mode = flags & kModeMask;
  const bool norm = flags & kFlagNorm;
  const bool wlnorm = flags & kFlagWlNorm;
  const bool temp = flags & kFlagTemp;

  const int t0 = blockIdx.y * tpb;
  const int tspan = min(tpb, n_tok - t0);
  const int h0 = blockIdx.x * hpb;
  const int hspan = min(hpb, nh - h0);

  // -- cos/sin for this block's tokens, once, shared by all their heads ------
  // Every head of a token wants the same head_dim/2 cos/sin pair, and with
  // M-RoPE assembling one costs three scattered table rows, so it is assembled
  // once per token here instead of once per (token, head) row.
  const bool mrope = mode >= kRopeMropeIl;
  const bool interleaved = mode == kRopeMropeIl;
  if (mode != kRopeNone) {
    for (int tt = 0; tt < tspan; ++tt) {
      const int64_t base = pos_s1 * (t0 + tt);
      const int64_t p_t = pos[base];
      const int64_t p_h = mrope ? pos[pos_s0 + base] : p_t;
      const int64_t p_w = mrope ? pos[2 * pos_s0 + base] : p_t;
      float* const dst = smem + 2 * half * tt;
      for (int j = threadIdx.x; j < half; j += blockDim.x) {
        int64_t p = p_t;
        if (mrope) {
          const int sec = mrope_section(j, st, sh, sw, interleaved);
          p = (sec == 0) ? p_t : ((sec == 1) ? p_h : p_w);
        }
        const float* row = cache + p * hd;
        dst[j] = static_cast<float>(static_cast<T>(row[j]));
        dst[half + j] = static_cast<float>(static_cast<T>(row[half + j]));
      }
    }
    __syncthreads();
  }

  // -- one lane group per head row ------------------------------------------
  const int lanes = half / VEC;             // power of two, <= 32 (host-checked)
  const int slots = blockDim.x / lanes;
  const int lane = threadIdx.x % lanes;
  const int slot = threadIdx.x / lanes;
  // Lane groups in one warp own different heads and so run different trip
  // counts; the norm's shuffle reduction must name only its own group, not the
  // whole warp, or it reads lanes that have already left the loop. ``lanes``
  // divides 32, so a group never straddles a warp.
  const unsigned grp_mask =
      (lanes >= 32) ? 0xffffffffu
                    : (((1u << lanes) - 1u) << ((threadIdx.x & 31) & ~(lanes - 1)));
  const int i0 = lane * VEC;                // first frequency index of this lane
  const bool gptj = mode == kRopeGptj;
  const bool round_rope = (flags & kFlagRopeRound) != 0;

  // One flat loop over the (token, head) rows this block owns: a slot walking
  // several rows keeps that many independent 128-bit accesses in flight, which
  // a slot-per-row block cannot do -- it has two loads and then stalls.
  const int nrows = tspan * hspan;
  for (int r = slot; r < nrows; r += slots) {
    const int tt = r / hspan;
    const int h = h0 + (r - tt * hspan);
    const int t = t0 + tt;
    const float* const cs = smem + 2 * half * tt;
    T* const base = qkv + static_cast<int64_t>(t) * row_stride
                    + static_cast<int64_t>(h) * hd;
    float a[VEC], b[VEC];

    // The rotation pair (i, i+half) is a pair of 16-byte vectors; the
    // interleaved pair (2i, 2i+1) is one 32-byte span de-interleaved in
    // registers. Both are fully coalesced.
    if (gptj) {
      const V v0 = *reinterpret_cast<const V*>(base + 2 * i0);
      const V v1 = *reinterpret_cast<const V*>(base + 2 * i0 + VEC);
#pragma unroll
      for (int k = 0; k < VEC; ++k) {
        const int m = 2 * k;
        a[k] = static_cast<float>(m < VEC ? v0.v[m] : v1.v[m - VEC]);
        b[k] = static_cast<float>((m + 1) < VEC ? v0.v[m + 1] : v1.v[m + 1 - VEC]);
      }
    } else {
      const V va = *reinterpret_cast<const V*>(base + i0);
      const V vb = *reinterpret_cast<const V*>(base + half + i0);
#pragma unroll
      for (int k = 0; k < VEC; ++k) {
        a[k] = static_cast<float>(va.v[k]);
        b[k] = static_cast<float>(vb.v[k]);
      }
    }

    if (norm) {
      float ss = 0.0f;
#pragma unroll
      for (int k = 0; k < VEC; ++k) ss += a[k] * a[k];
#pragma unroll
      for (int k = 0; k < VEC; ++k) ss += b[k] * b[k];
      for (int m = lanes >> 1; m; m >>= 1)
        ss += __shfl_xor_sync(grp_mask, ss, m, lanes);
      const float rstd = rsqrtf(ss / hd + eps);
      const T* const w = (h < nq) ? qw : kw;
      float wa[VEC], wb[VEC];
      if (gptj) {
        const V w0 = *reinterpret_cast<const V*>(w + 2 * i0);
        const V w1 = *reinterpret_cast<const V*>(w + 2 * i0 + VEC);
#pragma unroll
        for (int k = 0; k < VEC; ++k) {
          const int m = 2 * k;
          wa[k] = static_cast<float>(m < VEC ? w0.v[m] : w1.v[m - VEC]);
          wb[k] = static_cast<float>((m + 1) < VEC ? w0.v[m + 1]
                                                  : w1.v[m + 1 - VEC]);
        }
      } else {
        const V w0 = *reinterpret_cast<const V*>(w + i0);
        const V w1 = *reinterpret_cast<const V*>(w + half + i0);
#pragma unroll
        for (int k = 0; k < VEC; ++k) {
          wa[k] = static_cast<float>(w0.v[k]);
          wb[k] = static_cast<float>(w1.v[k]);
        }
      }
#pragma unroll
      for (int k = 0; k < VEC; ++k) {
        a[k] = static_cast<float>(static_cast<T>(a[k] * rstd * wa[k]));
        b[k] = static_cast<float>(static_cast<T>(b[k] * rstd * wb[k]));
      }
    }

    if (mode != kRopeNone) {
#pragma unroll
      for (int k = 0; k < VEC; ++k) {
        const float c = cs[i0 + k];
        const float s = cs[half + i0 + k];
        float na, nb;
        if (round_rope) {
          // The M-RoPE reference is a Triton kernel over bf16 tiles, and
          // Triton keeps bf16 x bf16 in bf16 -- but LLVM then contracts one of
          // the two products of each expression into the add/sub as an FMA,
          // so only the *other* product is materialised at bf16 precision. It
          // is the ``b`` product in both cases (``a*c - b*s`` contracts a*c,
          // ``b*c + a*s`` contracts a*s). Products of two bf16 values are
          // exact in fp32, so reproducing that is just rounding the b product:
          // this makes the fused pass bit-identical to the reference instead of
          // ~1 bf16 LSB away on 11% of q, which at the harness' 1e-2 tolerance
          // is the difference between a 0.7% and a 0.05% element mismatch.
          na = a[k] * c - static_cast<float>(static_cast<T>(b[k] * s));
          nb = a[k] * s + static_cast<float>(static_cast<T>(b[k] * c));
        } else {
          // The 1-D reference is the vendored rope CUDA kernel: fp32.
          na = a[k] * c - b[k] * s;
          nb = b[k] * c + a[k] * s;
        }
        a[k] = na;
        b[k] = nb;
      }
    }

    if (wlnorm) {
      float ss = 0.0f;
#pragma unroll
      for (int k = 0; k < VEC; ++k) ss += a[k] * a[k];
#pragma unroll
      for (int k = 0; k < VEC; ++k) ss += b[k] * b[k];
      for (int m = lanes >> 1; m; m >>= 1)
        ss += __shfl_xor_sync(grp_mask, ss, m, lanes);
      const float rstd = rsqrtf(ss / hd + eps);
#pragma unroll
      for (int k = 0; k < VEC; ++k) {
        a[k] = static_cast<float>(static_cast<T>(a[k] * rstd));
        b[k] = static_cast<float>(static_cast<T>(b[k] * rstd));
      }
    }

    if (temp && h < nq) {
      const float tscale =
          logf(floorf((static_cast<float>(pos[pos_s1 * t]) + 1.0f) / floor_scale)
               + 1.0f) * attn_scale + 1.0f;
#pragma unroll
      for (int k = 0; k < VEC; ++k) {
        a[k] *= tscale;
        b[k] *= tscale;
      }
    }

    if (gptj) {
      V v0, v1;
#pragma unroll
      for (int k = 0; k < VEC; ++k) {
        const int m = 2 * k;
        if (m < VEC) v0.v[m] = static_cast<T>(a[k]);
        else v1.v[m - VEC] = static_cast<T>(a[k]);
        if ((m + 1) < VEC) v0.v[m + 1] = static_cast<T>(b[k]);
        else v1.v[m + 1 - VEC] = static_cast<T>(b[k]);
      }
      *reinterpret_cast<V*>(base + 2 * i0) = v0;
      *reinterpret_cast<V*>(base + 2 * i0 + VEC) = v1;
    } else {
      V va, vb;
#pragma unroll
      for (int k = 0; k < VEC; ++k) {
        va.v[k] = static_cast<T>(a[k]);
        vb.v[k] = static_cast<T>(b[k]);
      }
      *reinterpret_cast<V*>(base + i0) = va;
      *reinterpret_cast<V*>(base + half + i0) = vb;
    }
  }
}

// Heads per block, and the block that covers them. The default is "one block
// per token, every head row in it", which reads each token's cos/sin exactly
// once -- what a long prefill wants. Short token counts halve the head span
// until the grid covers the device instead.
// Blocks own ``tpb`` tokens x ``hpb`` head rows and spread them over
// ``block / lanes`` slots. Two competing pressures: enough blocks to fill the
// device, and enough rows per slot to keep 128-bit accesses in flight.
void pick_geometry(int n_tok, int nh, int lanes, int hint, int* hpb, int* tpb,
                   int* block) {
  int h = nh, t = 1;
  int b = 32 * ((nh * lanes / 4 + 31) / 32);      // ~4 rows per slot
  b = std::min(std::max(b, 32), std::min(1024, nh * lanes));
  if (static_cast<int64_t>(n_tok) >= 4096) {
    // Long prefill: purely bandwidth-bound. Two tokens per block and ~12 slots
    // walking ~11 rows each measured best on B200 (4.0 TB/s of 570 MB at
    // n_tok=16384, against 3.2 for one row per slot); the sweep that produced
    // it is tools/geo_sweep.py driven through ``block_hint``.
    t = 2;
    b = 12 * lanes;
  } else {
    // Short token counts cannot fill the device with one block per token, so
    // the head span halves until the grid does.
    while (h > 4 && static_cast<int64_t>(n_tok) * ((nh + h - 1) / h) < 96) {
      h = (h + 1) / 2;
      b = std::min(std::max(32, h * lanes), b);
    }
  }
  if (hint > 0) {
    b = hint & 0xffff;
    const int th = hint >> 16;
    if (th > 0) t = th;
  }
  *hpb = h;
  *tpb = t;
  *block = std::max(32, (std::min(b, 1024) / lanes) * lanes);
}

template <typename T, int VEC>
void launch(torch::Tensor& qkv, const torch::Tensor& positions,
            const torch::Tensor& cache, const torch::Tensor& qweight,
            const torch::Tensor& kweight, int nq, int nh, int hd,
            double eps, double floor_scale, double attn_scale,
            int st, int sh, int sw, int flags, int hint) {
  const int n_tok = static_cast<int>(qkv.size(0));
  const int half = hd / 2;
  const int lanes = half / VEC;
  int hpb, tpb, block;
  pick_geometry(n_tok, nh, lanes, hint, &hpb, &tpb, &block);
  const dim3 grid((nh + hpb - 1) / hpb, (n_tok + tpb - 1) / tpb);
  const size_t shmem = (flags & 0x7)
                           ? static_cast<size_t>(2 * half) * tpb * sizeof(float)
                           : 0;

  const at::cuda::OptionalCUDAGuard guard(device_of(qkv));
  auto stream = at::cuda::getCurrentCUDAStream();
  qkv_glue_kernel<T, VEC><<<grid, block, shmem, stream>>>(
      reinterpret_cast<T*>(qkv.data_ptr()),
      positions.data_ptr<int64_t>(),
      // Unused, and deliberately untyped, when there is no rotation: the
      // caller hands us ``qkv`` itself as a valid-but-ignored pointer.
      reinterpret_cast<const float*>(cache.data_ptr()),
      reinterpret_cast<const T*>(qweight.defined() ? qweight.data_ptr()
                                                  : qkv.data_ptr()),
      reinterpret_cast<const T*>(kweight.defined() ? kweight.data_ptr()
                                                  : qkv.data_ptr()),
      nq, nh, hd, half, qkv.stride(0),
      positions.dim() == 2 ? positions.stride(0) : 0,
      positions.stride(-1),
      static_cast<float>(eps), static_cast<float>(floor_scale),
      static_cast<float>(attn_scale), st, sh, sw, n_tok, flags, hpb, tpb);
}

}  // namespace

bool qkv_glue_supported(int64_t head_dim, at::ScalarType dtype);

// ``sections`` packs (s_t, s_h, s_w) and ``flags`` packs
// (mode | norm<<3 | wlnorm<<4 | temp<<5 | round_rope<<6): the host side of this
// call is on the critical path for the decode shapes, so the argument list is
// kept short.
//
// ``block_hint`` overrides the launch geometry as ``block | (tokens << 16)``.
// It defaults to 0 (use the tuned heuristic) and nothing on the model path
// passes it; it exists so tools/geo_sweep.py can re-tune the table on a new
// device without rebuilding.
void qkv_glue(torch::Tensor qkv, torch::Tensor positions,
              torch::Tensor cos_sin_cache, torch::Tensor q_weight,
              torch::Tensor k_weight, int64_t num_q_heads,
              int64_t num_heads, int64_t head_dim, double eps,
              double floor_scale, double attn_scale, int64_t sections,
              int64_t flags, int64_t block_hint) {
  const int st = static_cast<int>(sections & 0x3ff);
  const int sh = static_cast<int>((sections >> 10) & 0x3ff);
  const int sw = static_cast<int>((sections >> 20) & 0x3ff);
  const int nq = static_cast<int>(num_q_heads);
  const int nh = static_cast<int>(num_heads);
  const int hd = static_cast<int>(head_dim);
  // The caller gates on ``supported`` at construction time, before the
  // activation dtype is known; re-checking against the dtype that actually
  // arrived turns an unaddressable geometry into an error rather than a wrong
  // answer.
  TORCH_CHECK(qkv_glue_supported(head_dim, qkv.scalar_type()),
              "qkv_glue: head_dim ", head_dim, " is not addressable for ",
              qkv.scalar_type());
  switch (qkv.scalar_type()) {
    case at::kBFloat16:
      launch<at::BFloat16, 8>(qkv, positions, cos_sin_cache, q_weight, k_weight,
                              nq, nh, hd, eps, floor_scale, attn_scale, st, sh,
                              sw, static_cast<int>(flags),
                              static_cast<int>(block_hint));
      break;
    case at::kHalf:
      launch<at::Half, 8>(qkv, positions, cos_sin_cache, q_weight, k_weight, nq,
                          nh, hd, eps, floor_scale, attn_scale, st, sh, sw,
                          static_cast<int>(flags),
                          static_cast<int>(block_hint));
      break;
    case at::kFloat:
      launch<float, 4>(qkv, positions, cos_sin_cache, q_weight, k_weight, nq,
                       nh, hd, eps, floor_scale, attn_scale, st, sh, sw,
                       static_cast<int>(flags),
                       static_cast<int>(block_hint));
      break;
    default:
      TORCH_CHECK(false, "qkv_glue: unsupported dtype ", qkv.scalar_type());
  }
}

// Whether the vectorized kernel can take this geometry: a 16-byte access must
// cover a whole number of frequency indices, and one head row must fit in a
// single lane group (so the norm reduction stays a shuffle).
bool qkv_glue_supported(int64_t head_dim, at::ScalarType dtype) {
  int vec;
  switch (dtype) {
    case at::kBFloat16:
    case at::kHalf: vec = 8; break;
    case at::kFloat: vec = 4; break;
    default: return false;
  }
  if (head_dim <= 0 || head_dim % 2) return false;
  const int64_t half = head_dim / 2;
  if (half % vec) return false;
  const int64_t lanes = half / vec;
  return lanes <= 32 && (lanes & (lanes - 1)) == 0;
}

#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qkv_glue", &qkv_glue, "Fused post-QKV norm + RoPE + scaling (CUDA)",
        py::arg("qkv"), py::arg("positions"), py::arg("cos_sin_cache"),
        py::arg("q_weight"), py::arg("k_weight"), py::arg("num_q_heads"),
        py::arg("num_heads"), py::arg("head_dim"), py::arg("eps"),
        py::arg("floor_scale"), py::arg("attn_scale"), py::arg("sections"),
        py::arg("flags"), py::arg("block_hint") = 0);
  m.def("supported", &qkv_glue_supported, "fast-path applicability");
}
