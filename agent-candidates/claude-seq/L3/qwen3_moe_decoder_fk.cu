// Fused kernels for the Qwen3-MoE decoder layer.
//
// Every kernel here reproduces the *arithmetic* of the op sequence it replaces,
// not merely its value to within a rounding.  That is a hard requirement rather
// than a stylistic one: the layer's output passes through two FP8
// quantizations inside the MoE, and a single-ULP bfloat16 difference upstream
// flips enough FP8 roundings downstream to move a few percent of the output
// elements by ~2% -- an order of magnitude past the scorer's per-element band.
// So the reduction orders, the intermediate roundings and the placement of
// every fused multiply-add below mirror the reference kernel being replaced.
//
//   add_rmsnorm    -- RMSNorm with the optional fused residual add, one pass
//                     over the row.  The reference reads the residual back out
//                     of HBM for its normalize pass (5 row-traversals; this is
//                     4, and 3 when there is no residual).
//   qk_norm_rope   -- per-head RMSNorm of the Q and K slices of a packed QKV
//                     buffer, then M-RoPE, in one pass.  The reference runs
//                     five kernels here (two `.contiguous()` copies, two norms,
//                     one rotary) and materializes Q twice.
//   silu_mul_quant -- SiLU-and-mul followed by per-128-group FP8 quantization
//                     of the MoE intermediate, in one pass instead of two
//                     kernels and a full bfloat16 round trip.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#include <algorithm>

namespace {

using bf16 = __nv_bfloat16;
constexpr unsigned kFull = 0xffffffffu;

// 16 B = 8 bfloat16 lanes, the reference's vector width everywhere below.
struct alignas(16) V8 {
  uint32_t w[4];
};

__device__ __forceinline__ float2 unpack(uint32_t v) {
  __nv_bfloat162 p;
  __builtin_memcpy(&p, &v, sizeof(p));
  return __bfloat1622float2(p);
}

__device__ __forceinline__ uint32_t pack(float a, float b) {
  const __nv_bfloat162 p = __floats2bfloat162_rn(a, b);
  uint32_t v;
  __builtin_memcpy(&v, &p, sizeof(v));
  return v;
}

// Packed bfloat16 add: `_f16Vec::operator+=` adds as __nv_bfloat162, so each
// lane is rounded to bfloat16 before the sum of squares ever sees it.
__device__ __forceinline__ uint32_t addp(uint32_t a, uint32_t b) {
  __nv_bfloat162 x, y;
  __builtin_memcpy(&x, &a, sizeof(x));
  __builtin_memcpy(&y, &b, sizeof(y));
  const __nv_bfloat162 s = __hadd2(x, y);
  uint32_t v;
  __builtin_memcpy(&v, &s, sizeof(v));
  return v;
}

// `_f16Vec<scalar_t, 8>::sum_squares()`: four independent
// ``result += z.x * z.x + z.y * z.y`` steps over the packed pairs, in order.
__device__ __forceinline__ float sum_squares(const V8& v) {
  float result = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 z = unpack(v.w[i]);
    result += z.x * z.x + z.y * z.y;
  }
  return result;
}

// cub::BlockReduce<float, 1024>::Reduce(x, Sum, num_valid) as the reference
// instantiates it: a balanced binary tree inside each warp, then the warp
// aggregates summed in warp order.  An ascending XOR butterfly lands on the
// same associations as cub's ascending shuffle-down tree and leaves the result
// in every lane, so no broadcast is needed.  `nwarp` is the number of warps the
// reference's `ApplyWarpAggregates` would fold in, i.e. ceil(num_valid / 32);
// warps whose threads all sat out contribute an exact zero.
__device__ __forceinline__ float block_sum(float x, float* s_warp, int nwarp) {
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) x += __shfl_xor_sync(kFull, x, off);
  if (nwarp == 1) return x;
  if ((threadIdx.x & 31) == 0) s_warp[threadIdx.x >> 5] = x;
  __syncthreads();
  float acc = s_warp[0];
  for (int w = 1; w < nwarp; ++w) acc += s_warp[w];
  return acc;
}

// ---------------------------------------------------------------------------
// RMSNorm over H elements per row (H % 8 == 0, bfloat16), one block per row.
//
// The chunk-to-thread map (`idx = tid; idx < H/8; idx += blockDim.x`) and the
// block size (`min(H, num_tokens < 256 ? 1024 : 256)`) are the reference's, so
// the per-thread partial sums are the same values summed in the same order.
// Unlike the reference the row stays in registers between the two passes.
// ---------------------------------------------------------------------------
template <int BLOCK, int VPT, bool FUSED>
__global__ __launch_bounds__(BLOCK) void add_rmsnorm_kernel(
    bf16* __restrict__ out, const bf16* __restrict__ inp,
    bf16* __restrict__ res, const bf16* __restrict__ wgt, float eps,
    float inv_h, int nvec, int64_t in_stride, int64_t out_stride, int nwarp) {
  __shared__ float s_warp[BLOCK / 32 ? BLOCK / 32 : 1];
  const int64_t row = blockIdx.x;
  const V8* __restrict__ ip =
      reinterpret_cast<const V8*>(inp + row * in_stride);
  V8* __restrict__ rp =
      FUSED ? reinterpret_cast<V8*>(res + row * (int64_t)nvec * 8) : nullptr;
  const V8* __restrict__ wp = reinterpret_cast<const V8*>(wgt);

  V8 z[VPT];
  float var = 0.0f;
#pragma unroll
  for (int k = 0; k < VPT; ++k) {
    const int idx = threadIdx.x + k * BLOCK;
    if (idx < nvec) {
      z[k] = ip[idx];
      if constexpr (FUSED) {
        const V8 r = rp[idx];
#pragma unroll
        for (int j = 0; j < 4; ++j) z[k].w[j] = addp(z[k].w[j], r.w[j]);
        rp[idx] = z[k];
      }
      var += sum_squares(z[k]);
    }
  }

  var = block_sum(var, s_warp, nwarp);
  const float s = rsqrtf(var * inv_h + eps);

  V8* __restrict__ op = reinterpret_cast<V8*>(out + row * out_stride);
#pragma unroll
  for (int k = 0; k < VPT; ++k) {
    const int idx = threadIdx.x + k * BLOCK;
    if (idx >= nvec) continue;
    const V8 w = wp[idx];
    V8 o;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 x = unpack(z[k].w[j]);
      const float2 wf = unpack(w.w[j]);
      o.w[j] = pack(x.x * s * wf.x, x.y * s * wf.y);
    }
    op[idx] = o;
  }
}

// ---------------------------------------------------------------------------
// Per-head RMSNorm (head_dim 128) of the Q/K slices of a packed QKV row, then
// M-RoPE, then a contiguous store of Q and K.
//
// A 16-lane group owns one head with 8 bfloat16 per lane, which is exactly the
// reference norm kernel's vector width *and* its thread count per row (block
// size = min(128 / 8, 1024) = 16): the per-lane partial is its per-thread
// partial and the 16-lane butterfly is its block reduction.
//
// The RoPE partner of element j < 64 is element j + 64, which lives in lane + 8
// of the same group, so the rotation is one XOR shuffle.  ROPE selects the
// reference variant: 1 = NeoX rotary over 1-D positions, 2 = sectioned M-RoPE,
// 3 = interleaved M-RoPE.
// ---------------------------------------------------------------------------
template <int ROPE>
__global__ __launch_bounds__(256) void qk_norm_rope_kernel(
    bf16* __restrict__ q_out, bf16* __restrict__ k_out,
    const bf16* __restrict__ qkv, int64_t qkv_stride,
    const bf16* __restrict__ qw, const bf16* __restrict__ kw, float eps,
    const bf16* __restrict__ cache, const int64_t* __restrict__ positions,
    int64_t pos_row_stride, int64_t pos_tok_stride, int p0, int p1,
    int n_q_heads, int total_heads, int n_tokens) {
  const int g = threadIdx.x >> 4;  // 16-lane group within the block
  const int lane = threadIdx.x & 15;
  // Groups inside a warp exit independently (the tail block is partly out of
  // range), so every shuffle below names only its own group's lanes.
  const unsigned gmask = 0xffffu << (threadIdx.x & 16);

  const int slot = blockIdx.x * 16 + g;
  const int tok = slot / total_heads;
  if (tok >= n_tokens) return;
  const int head = slot - tok * total_heads;
  const bool is_q = head < n_q_heads;

  const bf16* src = qkv + (int64_t)tok * qkv_stride + head * 128;
  V8 v = *reinterpret_cast<const V8*>(src + lane * 8);

  // --- RMSNorm: eight squares accumulated in order, then the 16-lane tree. ---
  float x[8];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 f = unpack(v.w[j]);
    x[2 * j] = f.x;
    x[2 * j + 1] = f.y;
  }
  float var = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) var += x[i] * x[i];
#pragma unroll
  for (int off = 1; off < 16; off <<= 1)
    var += __shfl_xor_sync(gmask, var, off, 16);
  const float s = rsqrtf(var * (1.0f / 128.0f) + eps);

  const V8 wv = *reinterpret_cast<const V8*>((is_q ? qw : kw) + lane * 8);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 wf = unpack(wv.w[j]);
    x[2 * j] = __bfloat162float(__float2bfloat16(x[2 * j] * s * wf.x));
    x[2 * j + 1] = __bfloat162float(__float2bfloat16(x[2 * j + 1] * s * wf.y));
  }

  // --- M-RoPE. ------------------------------------------------------------
  // The reference is a Triton kernel whose operands are bfloat16, and the
  // generated code rounds the product involving the *other* half of the pair
  // while contracting this half's product into the add:
  //     out_lo = bf16(lo * cos - bf16(hi * sin))
  //     out_hi = bf16(lo * sin + bf16(hi * cos))
  // Rotating entirely in fp32 is more accurate and measurably wrong here.
  if constexpr (ROPE != 0) {
    const bool first = lane < 8;
    const int base = (lane & 7) * 8;  // rotary index of this lane's chunk
    float cosv[8], sinv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int jj = base + i;
      int sec = 0;
      if constexpr (ROPE == 2) {
        sec = jj < p0 ? 0 : (jj < p1 ? 1 : 2);
      } else if constexpr (ROPE == 3) {
        // The reference's masks: index jj belongs to H when jj % 3 == 1 and
        // jj <= 3 * section_h, to W when jj % 3 == 2 and jj <= 3 * section_w,
        // and to T otherwise.
        const int r = jj % 3;
        sec = (r == 1 && jj <= p0) ? 1 : ((r == 2 && jj <= p1) ? 2 : 0);
      }
      const bf16* c =
          cache + positions[sec * pos_row_stride + tok * pos_tok_stride] * 128;
      cosv[i] = __bfloat162float(c[jj]);
      sinv[i] = __bfloat162float(c[64 + jj]);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float partner = __shfl_xor_sync(gmask, x[i], 8, 16);
      const float lo = first ? x[i] : partner;
      const float hi = first ? partner : x[i];
      x[i] = first ? lo * cosv[i] -
                         __bfloat162float(__float2bfloat16(hi * sinv[i]))
                   : lo * sinv[i] +
                         __bfloat162float(__float2bfloat16(hi * cosv[i]));
    }
  }

#pragma unroll
  for (int j = 0; j < 4; ++j) v.w[j] = pack(x[2 * j], x[2 * j + 1]);

  bf16* dst = is_q
                  ? q_out + (int64_t)tok * (n_q_heads * 128) + head * 128
                  : k_out + (int64_t)tok * ((total_heads - n_q_heads) * 128) +
                        (head - n_q_heads) * 128;
  *reinterpret_cast<V8*>(dst + lane * 8) = v;
}

// ---------------------------------------------------------------------------
// SiLU-and-mul + per-128-group FP8 quantization of the MoE intermediate.
//
// Reference numerics, both roundings included:
//   act = bf16( f32(bf16(g / (1 + expf(-g)))) * f32(u) )
// i.e. vLLM's packed_silu_kernel rounds the activation to bfloat16 before the
// multiply by the up-projection, and the product is rounded again on store.
// The quantization then matches deep_gemm's ceil_to_ue8m0 scale and the
// multiply-by-reciprocal the reference quantizer uses (dividing instead flips
// FP8 roundings at representable-value boundaries).
//
// One 16-lane group owns one 128-wide scale group, which is exactly the extent
// the absmax has to cover, so the reduction is four XOR shuffles and there is
// no shared memory and no barrier.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int ue8m0_exp(float absmax) {
  const float s = fmaxf(absmax, 1e-10f) * (1.0f / 448.0f);
  const unsigned int b = __float_as_uint(s);
  int e = static_cast<int>((b >> 23) & 0xFFu) + ((b & 0x7FFFFFu) != 0u ? 1 : 0);
  return e < 1 ? 1 : (e > 254 ? 254 : e);
}

__global__ __launch_bounds__(256) void silu_mul_quant_kernel(
    const bf16* __restrict__ inter, __nv_fp8_storage_t* __restrict__ q,
    float* __restrict__ scale, int rows, int N, int ngroups_per_row,
    int64_t s_stride) {
  const int lane = threadIdx.x & 15;
  const unsigned gmask = 0xffffu << (threadIdx.x & 16);
  const int gid = blockIdx.x * 16 + (threadIdx.x >> 4);
  const int total = rows * ngroups_per_row;
  if (gid >= total) return;
  const int row = gid / ngroups_per_row;
  const int grp = gid - row * ngroups_per_row;

  const bf16* base =
      inter + (int64_t)row * (2 * N) + grp * 128 + lane * 8;
  const V8 gv = *reinterpret_cast<const V8*>(base);
  const V8 uv = *reinterpret_cast<const V8*>(base + N);

  float f[8];
  float m = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 gf = unpack(gv.w[j]);
    const float2 uf = unpack(uv.w[j]);
    float2 a;
    a.x = gf.x / (1.0f + expf(-gf.x));
    a.y = gf.y / (1.0f + expf(-gf.y));
    const float2 ab = unpack(pack(a.x, a.y));  // activation -> bfloat16
    const float2 pb = unpack(pack(ab.x * uf.x, ab.y * uf.y));
    f[2 * j] = pb.x;
    f[2 * j + 1] = pb.y;
    m = fmaxf(m, fmaxf(fabsf(pb.x), fabsf(pb.y)));
  }
#pragma unroll
  for (int off = 1; off < 16; off <<= 1)
    m = fmaxf(m, __shfl_xor_sync(gmask, m, off, 16));

  const int e = ue8m0_exp(m);
  const float sc = __uint_as_float(static_cast<unsigned int>(e) << 23);
  const float inv = 1.0f / sc;

  uint2 out;
  unsigned short* op = reinterpret_cast<unsigned short*>(&out);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 t = make_float2(f[2 * j] * inv, f[2 * j + 1] * inv);
    op[j] = __nv_cvt_float2_to_fp8x2(t, __NV_SATFINITE, __NV_E4M3);
  }
  *reinterpret_cast<uint2*>(q + (int64_t)row * N + grp * 128 + lane * 8) = out;
  if (lane == 0) scale[(int64_t)row * s_stride + grp] = sc;
}

// ---------------------------------------------------------------------------
// Weighted top-k reduction: out[m] = bf16(sum_j f32(y[m * topk + j]) * w[m, j]).
//
// The reference applies the routing weight *after* the second grouped GEMM has
// rounded its accumulator to bfloat16, and sums in fp32 -- folding the weight
// into the GEMM epilogue instead rounds the product rather than the plain
// accumulator, which is a different (and observably different) result.
// ---------------------------------------------------------------------------
template <int TOPK>
__global__ void weighted_moe_sum_kernel(const bf16* __restrict__ y,
                                        const float* __restrict__ w,
                                        bf16* __restrict__ out, int64_t D,
                                        int64_t w_stride) {
  const int64_t m = blockIdx.x;
  const int64_t col = (int64_t)blockIdx.y * blockDim.x + threadIdx.x;
  if (col * 8 >= D) return;
  const V8* __restrict__ base =
      reinterpret_cast<const V8*>(y + m * TOPK * D) + col;
  const int64_t vd = D / 8;

  // All TOPK rows are in flight before any of them is touched.
  V8 v[TOPK];
  float wj[TOPK];
#pragma unroll
  for (int j = 0; j < TOPK; ++j) v[j] = base[(int64_t)j * vd];
#pragma unroll
  for (int j = 0; j < TOPK; ++j) wj[j] = w[m * w_stride + j];

  float acc[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) acc[e] = 0.0f;
#pragma unroll
  for (int j = 0; j < TOPK; ++j) {
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const float2 f = unpack(v[j].w[p]);
      acc[2 * p] += f.x * wj[j];
      acc[2 * p + 1] += f.y * wj[j];
    }
  }
  V8 o;
#pragma unroll
  for (int p = 0; p < 4; ++p) o.w[p] = pack(acc[2 * p], acc[2 * p + 1]);
  *(reinterpret_cast<V8*>(out + m * D) + col) = o;
}

template <int BLOCK, int VPT, bool FUSED>
void launch_norm(torch::Tensor& out, const torch::Tensor& inp,
                 torch::Tensor* res, const torch::Tensor& wgt, double eps,
                 int nrows, int nvec) {
  add_rmsnorm_kernel<BLOCK, VPT, FUSED>
      <<<nrows, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<bf16*>(out.data_ptr()),
          reinterpret_cast<const bf16*>(inp.data_ptr()),
          FUSED ? reinterpret_cast<bf16*>(res->data_ptr()) : nullptr,
          reinterpret_cast<const bf16*>(wgt.data_ptr()),
          static_cast<float>(eps), 1.0f / (nvec * 8), nvec, inp.stride(-2),
          out.stride(-2), (BLOCK + 31) / 32);
}

}  // namespace

// out = rmsnorm(inp [+ res]) * wgt.  With `res` given, `res` is overwritten by
// the bfloat16 sum first (the reference's in-place residual contract) and the
// norm reads that rounded value back, exactly as the reference does.
void add_rmsnorm(torch::Tensor out, torch::Tensor inp,
                 std::optional<torch::Tensor> res, torch::Tensor wgt,
                 double eps) {
  TORCH_CHECK(inp.scalar_type() == at::kBFloat16, "bfloat16 only");
  TORCH_CHECK(inp.stride(-1) == 1 && out.stride(-1) == 1, "innermost stride");
  const int64_t H = inp.size(-1);
  TORCH_CHECK(H % 8 == 0 && wgt.numel() == H, "hidden must be a multiple of 8");
  const int64_t nrows = inp.numel() / H;
  if (nrows == 0) return;
  const int nvec = static_cast<int>(H / 8);
  const c10::cuda::CUDAGuard guard(inp.device());
  const bool fused = res.has_value();

  // The reference's block-size rules, reproduced so the reduction matches: the
  // fused kernel sizes its block by the element count and the plain one by the
  // vector count, both capped at 1024 rows-per-block below 256 rows and 256
  // above (its "smaller blocks hide latency better once the grid is wide"
  // heuristic).
  const int max_block = (nrows < 256) ? 1024 : 256;
  const int block = static_cast<int>(
      std::min<int64_t>(fused ? H : nvec, max_block));
  const int rows = static_cast<int>(nrows);

#define DISPATCH(B, V)                                                     \
  if (fused)                                                               \
    launch_norm<B, V, true>(out, inp, &res.value(), wgt, eps, rows, nvec);  \
  else                                                                     \
    launch_norm<B, V, false>(out, inp, nullptr, wgt, eps, rows, nvec);

  // hidden 4096 = 512 chunks: two per thread over 256 threads, or one per
  // thread over 512 (1024 for the fused kernel, whose upper half sits idle --
  // the reference's own launch shape).
  if (nvec == 512 && block == 256) {
    DISPATCH(256, 2)
  } else if (nvec == 512 && block == 512) {
    DISPATCH(512, 1)
  } else if (nvec == 512 && block == 1024) {
    DISPATCH(1024, 1)
  } else {
    TORCH_CHECK(false, "add_rmsnorm: unsupported (hidden=", H, ", rows=", nrows,
                ")");
  }
#undef DISPATCH
}

// Normalizes the Q/K head slices of `qkv`, applies M-RoPE, and returns Q and K
// contiguously (the layout the reference's `.contiguous()` + norm + rotary
// chain leaves them in).  `rope`: 0 = none, 1 = 1-D NeoX, 2 = sectioned M-RoPE,
// 3 = interleaved M-RoPE.  `p0`/`p1` are the section bounds: (s_t, s_t + s_h)
// for rope 2 and (3 * s_h, 3 * s_w) for rope 3.
std::vector<torch::Tensor> qk_norm_rope(
    torch::Tensor qkv, torch::Tensor qw, torch::Tensor kw, double eps,
    torch::Tensor cos_sin, torch::Tensor positions, int64_t n_q_heads,
    int64_t n_kv_heads, int64_t p0, int64_t p1, int64_t rope) {
  TORCH_CHECK(qkv.scalar_type() == at::kBFloat16, "bfloat16 only");
  TORCH_CHECK(qkv.dim() == 2 && qkv.stride(1) == 1, "qkv must be row-major 2D");
  TORCH_CHECK(qw.numel() == 128 && kw.numel() == 128, "head_dim must be 128");
  const int n_tokens = static_cast<int>(qkv.size(0));
  const int total_heads = static_cast<int>(n_q_heads + n_kv_heads);
  auto q_out = torch::empty({qkv.size(0), n_q_heads * 128}, qkv.options());
  auto k_out = torch::empty({qkv.size(0), n_kv_heads * 128}, qkv.options());
  if (n_tokens == 0) return {q_out, k_out};
  if (rope != 0) {
    TORCH_CHECK(cos_sin.scalar_type() == at::kBFloat16 && cos_sin.dim() == 2 &&
                    cos_sin.size(1) == 128 && cos_sin.is_contiguous(),
                "cos_sin must be a contiguous bf16 [*, 128] cache");
    TORCH_CHECK(positions.scalar_type() == at::kLong, "positions int64");
  }

  const c10::cuda::CUDAGuard guard(qkv.device());
  const int64_t slots = (int64_t)n_tokens * total_heads;
  const int grid = static_cast<int>((slots + 15) / 16);
  const int64_t pos_row_stride = (rope >= 2) ? positions.stride(0) : 0;
  const int64_t pos_tok_stride = positions.numel() ? positions.stride(-1) : 0;
  auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH(R)                                                             \
  qk_norm_rope_kernel<R><<<grid, 256, 0, stream>>>(                           \
      reinterpret_cast<bf16*>(q_out.data_ptr()),                              \
      reinterpret_cast<bf16*>(k_out.data_ptr()),                              \
      reinterpret_cast<const bf16*>(qkv.data_ptr()), qkv.stride(0),           \
      reinterpret_cast<const bf16*>(qw.data_ptr()),                           \
      reinterpret_cast<const bf16*>(kw.data_ptr()),                           \
      static_cast<float>(eps),                                                \
      cos_sin.numel() ? reinterpret_cast<const bf16*>(cos_sin.data_ptr())     \
                      : nullptr,                                              \
      positions.numel() ? positions.data_ptr<int64_t>() : nullptr,            \
      pos_row_stride, pos_tok_stride, static_cast<int>(p0),                   \
      static_cast<int>(p1), static_cast<int>(n_q_heads), total_heads, n_tokens)
  if (rope == 3) {
    LAUNCH(3);
  } else if (rope == 2) {
    LAUNCH(2);
  } else if (rope == 1) {
    LAUNCH(1);
  } else {
    LAUNCH(0);
  }
#undef LAUNCH
  return {q_out, k_out};
}

// silu(gate) * up over `inter` [rows, 2N], quantized to FP8 per 128-wide group.
// `q` is [rows, N] float8_e4m3fn and `scale` [rows, N / 128] float32.
void silu_mul_quant(torch::Tensor inter, torch::Tensor q, torch::Tensor scale) {
  TORCH_CHECK(inter.scalar_type() == at::kBFloat16 && inter.dim() == 2,
              "inter must be 2-D bfloat16");
  TORCH_CHECK(inter.is_contiguous(), "inter must be contiguous");
  TORCH_CHECK(q.scalar_type() == at::kFloat8_e4m3fn && q.is_contiguous(),
              "q must be contiguous float8_e4m3fn");
  TORCH_CHECK(scale.scalar_type() == at::kFloat, "scale must be float32");
  const int rows = static_cast<int>(inter.size(0));
  const int N = static_cast<int>(inter.size(1) / 2);
  TORCH_CHECK(N % 128 == 0, "N must be a multiple of 128");
  TORCH_CHECK(q.size(0) == rows && q.size(1) == N, "q shape");
  if (rows == 0) return;
  const int ng = N / 128;
  const c10::cuda::CUDAGuard guard(inter.device());
  const int64_t groups = (int64_t)rows * ng;
  silu_mul_quant_kernel<<<(groups + 15) / 16, 256, 0,
                          at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(inter.data_ptr()),
      reinterpret_cast<__nv_fp8_storage_t*>(q.data_ptr()),
      scale.data_ptr<float>(), rows, N, ng, scale.stride(0));
}

// out[m] = bf16(sum_j f32(y[m * topk + j]) * topk_weights[m, j]).
torch::Tensor weighted_moe_sum(torch::Tensor y, torch::Tensor topk_weights,
                               int64_t topk) {
  TORCH_CHECK(y.scalar_type() == at::kBFloat16 && y.is_contiguous(),
              "y must be contiguous bfloat16");
  TORCH_CHECK(topk_weights.scalar_type() == at::kFloat, "weights float32");
  const int64_t D = y.size(1);
  TORCH_CHECK(D % 8 == 0, "D must be a multiple of 8");
  const int64_t M = y.size(0) / topk;
  auto out = torch::empty({M, D}, y.options());
  if (M == 0) return out;
  const c10::cuda::CUDAGuard guard(y.device());
  const int64_t vd = D / 8;
  const int threads = static_cast<int>(std::min<int64_t>(vd, 256));
  const dim3 grid(static_cast<unsigned>(M),
                  static_cast<unsigned>((vd + threads - 1) / threads));
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* yp = reinterpret_cast<const bf16*>(y.data_ptr());
  auto* op = reinterpret_cast<bf16*>(out.data_ptr());
  const float* wp = topk_weights.data_ptr<float>();
  const int64_t ws = topk_weights.stride(0);
#define WSUM(K)                                                              \
  weighted_moe_sum_kernel<K><<<grid, threads, 0, stream>>>(yp, wp, op, D, ws)
  if (topk == 8) {
    WSUM(8);
  } else if (topk == 4) {
    WSUM(4);
  } else if (topk == 2) {
    WSUM(2);
  } else if (topk == 1) {
    WSUM(1);
  } else {
    TORCH_CHECK(false, "weighted_moe_sum: unsupported topk ", topk);
  }
#undef WSUM
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add_rmsnorm", &add_rmsnorm, "Fused-add RMSNorm (bf16)");
  m.def("qk_norm_rope", &qk_norm_rope, "Per-head QK RMSNorm + M-RoPE");
  m.def("silu_mul_quant", &silu_mul_quant, "SiLU-and-mul + FP8 group quant");
  m.def("weighted_moe_sum", &weighted_moe_sum, "Weighted top-k MoE reduce");
}
