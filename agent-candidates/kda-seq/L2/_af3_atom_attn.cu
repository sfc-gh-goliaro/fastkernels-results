// Fused stages for AlphaFold3 sequence-local atom attention on sm_100.
//
// The operator is dispatch-bound: about 0.6 GFLOP per encoder forward against 618
// device kernels and 9.3 ms of wall time, with only 0.7 ms of that actually spent
// inside kernels. So these kernels exist to remove launches, not to run the
// arithmetic faster, and each one swallows a whole stage of the reference's
// elementwise tail.
//
// Precision rule, and it is the opposite of the usual one: reproduce the
// reference's rounding rather than improve on it. The comparison gate is
// atol=1e-2, rtol=1e-2 with 99% of elements required to pass, while the
// reference's own bf16 rounding of a cancelling sum can exceed that. Weight
// magnitudes are not bounded either -- the harness only re-randomizes torch.empty
// weights whose maximum falls outside [1e-6, 1e4], so surviving garbage reaches
// ~1e4 and intermediate magnitudes vary by orders of magnitude between runs. So
// every point where the reference stores a bf16 intermediate that a later stage
// re-reads is reproduced here by rounding through bf16, and reductions accumulate
// in fp32 exactly where the leaf ops do.

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace {

// The captured configuration. Compile-time so the inner loops unroll and the
// shared-memory footprint is known; the Python allow-list guarantees them and the
// entry points below re-check.
constexpr int kC = 128;    // c_atom
constexpr int kCP = 16;    // c_atom_pair
constexpr int kH = 4;      // no_heads
constexpr int kD = 32;     // c_hidden
constexpr int kQ = 32;     // n_query
constexpr int kK = 128;    // n_key
constexpr int kFF = 256;   // n_transition * c_atom

using bf16 = __nv_bfloat16;

// Round a value through bfloat16, which is what "the reference stored this and read
// it back" means numerically.
__device__ __forceinline__ float rbf(float x) {
  return __bfloat162float(__float2bfloat16(x));
}

__device__ __forceinline__ float ld(const bf16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ void st(bf16* p, float v) { *p = __float2bfloat16(v); }

__device__ __forceinline__ float sigmoidf_(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

// Mean and reciprocal standard deviation of one row of kC values spread across kC
// threads, accumulated in fp32 as the LayerNorm leaf does. `scratch` needs two
// floats per warp per row.
template <int WARPS>
__device__ __forceinline__ void row_moments(
    float v, float* scratch, int lane, int warp, float& mu, float& rstd) {
  float s = v, ss = v * v;
#pragma unroll
  for (int off = 16; off; off >>= 1) {
    s += __shfl_xor_sync(0xffffffffu, s, off);
    ss += __shfl_xor_sync(0xffffffffu, ss, off);
  }
  if (lane == 0) {
    scratch[warp] = s;
    scratch[WARPS + warp] = ss;
  }
  __syncthreads();
  float ts = 0.f, tss = 0.f;
#pragma unroll
  for (int w = 0; w < WARPS; ++w) {
    ts += scratch[w];
    tss += scratch[WARPS + w];
  }
  mu = ts / float(kC);
  float var = tss / float(kC) - mu * mu;
  rstd = rsqrtf(var + 1e-5f);
}

// ---------------------------------------------------------------------------
// The blocked layout: key-slot atom indices, per-slot key mask, padded query mask.
//
// This reproduces `_get_block_key_indices` bit for bit, and the bf16 arithmetic is
// the entire point rather than an artifact. The reference derives n_real from
// atom_mask.sum(), which is bfloat16, and every later quantity inherits that
// through int-to-bf16 promotion: PyTorch's type promotion converts the int32
// operand to bfloat16 first, computes in fp32, and rounds the result back. So
// n_real - 1 evaluates to 368.0 rather than 367 at the captured extent, and every
// window index above 256 snaps to an even atom -- which is why block 8 addresses
// 105 distinct keys rather than 128. A clean-integer window disagrees with the
// reference in 638 of the 1536 slots.
//
// One CTA: the whole table is 12x128 and every slot needs n_real first.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void layout_kernel(
    const bf16* __restrict__ mask,
    int n_atom, int n_padded, int num_blocks, int n_query, int n_key,
    int* __restrict__ idx_raw,        // [num_blocks, n_key] clamped, as the reference
    int* __restrict__ idx_gather,     // [num_blocks, n_key] invalid -> spare zero row
    bf16* __restrict__ mask_k,        // [num_blocks, n_key]
    bf16* __restrict__ mask_p) {      // [n_padded]
  const int t = threadIdx.x;
  __shared__ float red[256 / 32];
  __shared__ float sh_n_real;

  for (int i = t; i < n_padded; i += 256)
    mask_p[i] = i < n_atom ? mask[i] : __float2bfloat16(0.f);

  // torch.sum over a bf16 tensor accumulates in fp32 and rounds the result once.
  float s = 0.f;
  for (int i = t; i < n_atom; i += 256) s += ld(mask + i);
#pragma unroll
  for (int off = 16; off; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
  if ((t & 31) == 0) red[t >> 5] = s;
  __syncthreads();
  if (t == 0) {
    float tot = 0.f;
#pragma unroll
    for (int w = 0; w < 256 / 32; ++w) tot += red[w];
    sh_n_real = rbf(tot);
  }
  __syncthreads();

  const float n_real = sh_n_real;
  const float n_real_m1 = rbf(n_real - 1.0f);
  const float hi = fmaxf(n_real_m1, 0.f);

  for (int e = t; e < num_blocks * n_key; e += 256) {
    const int b = e / n_key, j = e % n_key;
    const int center = n_query / 2 + b * n_query;
    const int initial = center + (j - n_key / 2);
    const int init0 = center - n_key / 2;
    const int init_last = center + n_key / 2 - 1;

    const int underflow = max(0, -init0);
    const float overflow = fmaxf(0.f, rbf(rbf(float(init_last)) - n_real_m1));
    const float shift = underflow > 0 ? rbf(float(underflow)) : -overflow;
    const float fin = rbf(rbf(float(initial)) + shift);

    const bool invalid = (fin < 0.f) || (fin >= n_real);
    const int safe = int(fminf(fmaxf(fin, 0.f), hi));
    idx_raw[e] = safe;
    idx_gather[e] = invalid ? n_padded : safe;
    mask_k[e] = invalid ? __float2bfloat16(0.f) : mask_p[safe];
  }
}

// ---------------------------------------------------------------------------
// AdaLN for the query and key sides, then the Q / K / V / output-gate
// projections. Row-wise, so K and V are produced per *atom* here and gathered by
// the attention kernel: key rows are a gather of atom rows, which makes projecting
// 368 rows equivalent to the reference's 1536.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kC) void qkvg_kernel(
    const bf16* __restrict__ a,
    const bf16* __restrict__ gate_q, const bf16* __restrict__ add_q,
    const bf16* __restrict__ gate_k, const bf16* __restrict__ add_k,
    const bf16* __restrict__ qg_w,   // [kC, 2*kC] channel-major in the output dim
    const bf16* __restrict__ qg_b,   // [2*kC]
    const bf16* __restrict__ kv_w,   // [kC, 2*kC]
    bf16* __restrict__ qkv,          // [n_padded, 3*kC] -> Q, K, V
    bf16* __restrict__ g,            // [n_atom, kC]
    int n_atom, float root) {
  const int row = blockIdx.x;
  if (row >= n_atom) return;
  const int t = threadIdx.x;
  const int lane = t & 31, warp = t >> 5;

  __shared__ float scratch[2 * (kC / 32)];
  __shared__ bf16 aq[kC], ak[kC];

  const float av = ld(a + row * kC + t);
  float mu, rstd;
  row_moments<kC / 32>(av, scratch, lane, warp, mu, rstd);

  // layer_norm_a has neither scale nor offset; its bf16 result is what the two
  // AdaLN instances re-read, so it is rounded here.
  const float a_hat = rbf((av - mu) * rstd);
  aq[t] = __float2bfloat16(
      rbf(ld(gate_q + row * kC + t) * rbf(a_hat + ld(add_q + row * kC + t))));
  ak[t] = __float2bfloat16(
      rbf(ld(gate_k + row * kC + t) * rbf(a_hat + ld(add_k + row * kC + t))));
  __syncthreads();

  // Two doubled projections rather than one quadrupled one: Q and the output gate
  // read a_q, while K and V read a_k.
  float acc_q = ld(qg_b + t), acc_g = ld(qg_b + kC + t);
  float acc_k = 0.f, acc_v = 0.f;
#pragma unroll 8
  for (int k = 0; k < kC; ++k) {
    const float xq = __bfloat162float(aq[k]);
    const float xk = __bfloat162float(ak[k]);
    acc_q = fmaf(xq, ld(qg_w + k * 2 * kC + t), acc_q);
    acc_g = fmaf(xq, ld(qg_w + k * 2 * kC + kC + t), acc_g);
    acc_k = fmaf(xk, ld(kv_w + k * 2 * kC + t), acc_k);
    acc_v = fmaf(xk, ld(kv_w + k * 2 * kC + kC + t), acc_v);
  }

  // The reference scales the query *after* rounding the projection to bf16 and
  // rounds again, and it divides rather than multiplying by a reciprocal; scaling
  // before the product would additionally let the unscaled sum reach a different
  // exponent under a large weight draw.
  st(qkv + row * 3 * kC + t, rbf(rbf(acc_q) / root));
  st(qkv + row * 3 * kC + kC + t, rbf(acc_k));
  st(qkv + row * 3 * kC + 2 * kC + t, rbf(acc_v));
  st(g + row * kC + t, rbf(sigmoidf_(rbf(acc_g))));
}

// ---------------------------------------------------------------------------
// Sequence-local attention, split into a scoring pass and a row-wise epilogue.
//
// One kernel per query block was the obvious shape and the wrong one: grid=12 on a
// 148-SM device is 96 warps for the whole GPU, under one warp per SM, so every
// shared-memory latency is exposed with nothing to hide it behind. NCU measured
// that version at 2.0% of SM throughput and 0.08 waves per SM -- 128 us per launch
// for 1.6 MFLOP of work, a third of the whole candidate's device time. Splitting
// the cross-head reduction out lets the scoring pass run one CTA per
// (block, head, query tile), 192 CTAs rather than 12, and leaves the epilogue as a
// plain row-wise kernel over atoms -- the same shape as qkvg and trans, which both
// measure an order of magnitude better.
//
// The mask bias is formed here rather than passed in, which removes a 1.2 MB
// temporary and the launches that built it. It is rounded through bf16 for the same
// reason the reference's is bf16: at that precision -1e9 absorbs the score
// entirely, so a masked slot's logit becomes exactly -1e9 whatever its score. That
// is what drives its softmax weight to exactly zero, and what makes a fully masked
// query row (block 11's padding) come out uniform rather than scored. Those rows
// are discarded by the reference's [..., :n_atom, :] slice, reproduced here by
// writing only rows below n_atom.
//
// `zb` is the three blocks' stacked linear_z projection in its natural
// [num_blocks, kQ, kK, n_stack, kH] layout, read with a stride rather than permuted
// into a fresh contiguous tensor -- a permuted add in ATen preserves the input's
// dimension ordering, so materializing it would cost a copy of every element.
// ---------------------------------------------------------------------------
constexpr int kQT = 8;            // query rows per CTA
constexpr int kQG = kQ / kQT;     // query tiles per block

__global__ __launch_bounds__(256) void attn_pv_kernel(
    const bf16* __restrict__ qkv,      // [n_padded + 1, 3*kC]
    const int* __restrict__ idx,       // [num_blocks, kK], invalid -> spare zero row
    const bf16* __restrict__ zb,       // [num_blocks*kQ*kK, n_stack*kH]
    const bf16* __restrict__ mask_qp,  // [n_padded]
    const bf16* __restrict__ mask_k,   // [num_blocks, kK]
    bf16* __restrict__ o,              // [n_atom, kC], heads concatenated
    int n_atom, int blk, int n_stack) {
  const int t = threadIdx.x;
  const int qg = blockIdx.x % kQG;
  const int h = (blockIdx.x / kQG) % kH;
  const int b = blockIdx.x / (kQG * kH);
  const int q0 = qg * kQT;

  // 37 KB, inside the 48 KB static limit now that the head is a grid dimension
  // rather than a loop, so no dynamic-shared opt-in is needed.
  __shared__ float sh_k[kK][kD];
  __shared__ float sh_v[kK][kD];
  __shared__ float sh_q[kQT][kD];
  __shared__ float sh_p[kQT][kK];

  const int* my_idx = idx + b * kK;
  for (int e = t; e < kK * kD; e += 256) {
    const int j = e / kD, d = e % kD;
    const int row = my_idx[j];
    sh_k[j][d] = ld(qkv + row * 3 * kC + kC + h * kD + d);
    sh_v[j][d] = ld(qkv + row * 3 * kC + 2 * kC + h * kD + d);
  }
  for (int e = t; e < kQT * kD; e += 256) {
    const int q = e / kD, d = e % kD;
    const int row = b * kQ + q0 + q;
    sh_q[q][d] = row < n_atom ? ld(qkv + row * 3 * kC + h * kD + d) : 0.f;
  }
  __syncthreads();

  // Scores, then the mask bias and the pair bias as two separate bf16 adds, which
  // is the order and the precision the reference applies them in.
  for (int e = t; e < kQT * kK; e += 256) {
    const int q = e / kK, j = e % kK;
    float acc = 0.f;
#pragma unroll
    for (int d = 0; d < kD; ++d) acc = fmaf(sh_q[q][d], sh_k[j][d], acc);
    const float bm = rbf(ld(mask_qp + b * kQ + q0 + q) * ld(mask_k + b * kK + j));
    const float scored = rbf(rbf(acc) + rbf(1e9f * (bm - 1.0f)));
    const int zoff = ((b * kQ + q0 + q) * kK + j) * n_stack * kH + blk * kH + h;
    sh_p[q][j] = rbf(scored + ld(zb + zoff));
  }
  __syncthreads();

  // One warp per query row, max-subtracting. A row that is entirely -1e9 comes out
  // uniform, which is what the reference produces for a fully masked row and what
  // keeps it free of NaN.
  {
    const int w = t >> 5, lane = t & 31;
    if (w < kQT) {
      float m = -INFINITY;
      for (int j = lane; j < kK; j += 32) m = fmaxf(m, sh_p[w][j]);
#pragma unroll
      for (int off = 16; off; off >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
      float s = 0.f;
      for (int j = lane; j < kK; j += 32) {
        const float e_ = __expf(sh_p[w][j] - m);
        sh_p[w][j] = e_;
        s += e_;
      }
#pragma unroll
      for (int off = 16; off; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
      const float inv = 1.0f / s;
      for (int j = lane; j < kK; j += 32) sh_p[w][j] = rbf(sh_p[w][j] * inv);
    }
  }
  __syncthreads();

  for (int e = t; e < kQT * kD; e += 256) {
    const int q = e / kD, d = e % kD;
    const int row = b * kQ + q0 + q;
    if (row >= n_atom) continue;
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < kK; ++j) acc = fmaf(sh_p[q][j], sh_v[j][d], acc);
    st(o + row * kC + h * kD + d, rbf(acc));
  }
}

// Output gate, linear_o, AdaLN output gate, residual: row-wise over atoms.
__global__ __launch_bounds__(kC) void attn_epi_kernel(
    const bf16* __restrict__ o, const bf16* __restrict__ g,
    const bf16* __restrict__ o_w, const bf16* __restrict__ ada,
    bf16* __restrict__ a, int n_atom) {
  const int row = blockIdx.x;
  if (row >= n_atom) return;
  const int t = threadIdx.x;
  __shared__ bf16 og[kC];
  // The reference gates each head's output before concatenating the heads, which is
  // the same thing elementwise.
  og[t] = __float2bfloat16(rbf(ld(o + row * kC + t) * ld(g + row * kC + t)));
  __syncthreads();

  float acc = 0.f;
#pragma unroll 8
  for (int n = 0; n < kC; ++n)
    acc = fmaf(__bfloat162float(og[n]), ld(o_w + n * kC + t), acc);
  const float upd = rbf(ld(ada + row * kC + t) * rbf(acc));
  st(a + row * kC + t, rbf(ld(a + row * kC + t) + upd));
}


// ---------------------------------------------------------------------------
// The conditioned transition block: AdaLN, SwiGLU, output projection, output
// gate, mask, residual. The gate-up dual projection is one pass with two
// accumulators and the activation folded into the epilogue.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kC) void trans_kernel(
    const bf16* __restrict__ gate_t, const bf16* __restrict__ add_t,
    const bf16* __restrict__ sw_w,   // [kC, 2*kFF]
    const bf16* __restrict__ o_w,    // [kFF, kC]
    const bf16* __restrict__ raw_t,  // [n_atom, kC]
    const bf16* __restrict__ mask,   // [n_atom]
    bf16* __restrict__ a,            // [n_atom, kC], updated in place
    int n_atom) {
  const int row = blockIdx.x;
  if (row >= n_atom) return;
  const int t = threadIdx.x;
  const int lane = t & 31, warp = t >> 5;

  __shared__ float scratch[2 * (kC / 32)];
  __shared__ bf16 x[kC];
  __shared__ bf16 hid[kFF];

  const float av = ld(a + row * kC + t);
  float mu, rstd;
  row_moments<kC / 32>(av, scratch, lane, warp, mu, rstd);
  const float a_hat = rbf((av - mu) * rstd);
  x[t] = __float2bfloat16(
      rbf(ld(gate_t + row * kC + t) * rbf(a_hat + ld(add_t + row * kC + t))));
  __syncthreads();

  constexpr int kPer = kFF / kC;  // outputs of the doubled projection per thread
#pragma unroll
  for (int u = 0; u < kPer; ++u) {
    const int n = u * kC + t;
    float acc_a = 0.f, acc_b = 0.f;
#pragma unroll 8
    for (int k = 0; k < kC; ++k) {
      const float xv = __bfloat162float(x[k]);
      acc_a = fmaf(xv, ld(sw_w + k * 2 * kFF + n), acc_a);
      acc_b = fmaf(xv, ld(sw_w + k * 2 * kFF + kFF + n), acc_b);
    }
    const float ua = rbf(acc_a);
    hid[n] = __float2bfloat16(rbf(rbf(ua * sigmoidf_(ua)) * rbf(acc_b)));
  }
  __syncthreads();

  float acc = 0.f;
#pragma unroll 8
  for (int k = 0; k < kFF; ++k)
    acc = fmaf(__bfloat162float(hid[k]), ld(o_w + k * kC + t), acc);
  const float upd = rbf(rbf(ld(raw_t + row * kC + t) * rbf(acc)) * ld(mask + row));
  st(a + row * kC + t, rbf(ld(a + row * kC + t) + upd));
}

// ---------------------------------------------------------------------------
// Entry points.
// ---------------------------------------------------------------------------
#define CHECK_BF16_CUDA(x, name)                                             \
  TORCH_CHECK((x).is_cuda() && (x).scalar_type() == at::kBFloat16 &&          \
                  (x).is_contiguous(),                                        \
              name " must be contiguous bfloat16 on CUDA")

std::vector<at::Tensor> layout(const at::Tensor& mask, int64_t n_query,
                               int64_t n_key) {
  CHECK_BF16_CUDA(mask, "atom_mask");
  const int n_atom = mask.numel();
  const int pad_q = int((n_query - n_atom % n_query) % n_query);
  const int n_padded = n_atom + pad_q;
  const int num_blocks = n_padded / int(n_query);
  const c10::cuda::CUDAGuard guard(mask.device());
  const auto i32 = mask.options().dtype(at::kInt);
  // at::empty rather than empty_like: empty_like would inherit arbitrary strides
  // on a singleton dimension.
  auto idx_raw = at::empty({num_blocks, int(n_key)}, i32);
  auto idx_gather = at::empty({num_blocks, int(n_key)}, i32);
  auto mask_k = at::empty({num_blocks, int(n_key)}, mask.options());
  auto mask_p = at::empty({n_padded}, mask.options());
  layout_kernel<<<1, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(mask.data_ptr()),
      n_atom, n_padded, num_blocks, int(n_query), int(n_key),
      idx_raw.data_ptr<int>(), idx_gather.data_ptr<int>(),
      reinterpret_cast<bf16*>(mask_k.data_ptr()),
      reinterpret_cast<bf16*>(mask_p.data_ptr()));
  return {idx_raw, idx_gather, mask_k, mask_p};
}

void qkvg(const at::Tensor& a, const at::Tensor& gate_q, const at::Tensor& add_q,
          const at::Tensor& gate_k, const at::Tensor& add_k,
          const at::Tensor& qg_w, const at::Tensor& qg_b, const at::Tensor& kv_w,
          at::Tensor qkv, at::Tensor g, double root) {
  CHECK_BF16_CUDA(a, "a");
  CHECK_BF16_CUDA(qkv, "qkv");
  const int n_atom = a.size(0);
  TORCH_CHECK(a.size(1) == kC && qkv.size(1) == 3 * kC);
  TORCH_CHECK(qg_w.size(0) == kC && qg_w.size(1) == 2 * kC);
  const c10::cuda::CUDAGuard guard(a.device());
  qkvg_kernel<<<n_atom, kC, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(a.data_ptr()),
      reinterpret_cast<const bf16*>(gate_q.data_ptr()),
      reinterpret_cast<const bf16*>(add_q.data_ptr()),
      reinterpret_cast<const bf16*>(gate_k.data_ptr()),
      reinterpret_cast<const bf16*>(add_k.data_ptr()),
      reinterpret_cast<const bf16*>(qg_w.data_ptr()),
      reinterpret_cast<const bf16*>(qg_b.data_ptr()),
      reinterpret_cast<const bf16*>(kv_w.data_ptr()),
      reinterpret_cast<bf16*>(qkv.data_ptr()),
      reinterpret_cast<bf16*>(g.data_ptr()),
      n_atom, static_cast<float>(root));
}

void attn(const at::Tensor& qkv, const at::Tensor& idx, const at::Tensor& zb,
          const at::Tensor& mask_qp, const at::Tensor& mask_k,
          const at::Tensor& g, const at::Tensor& o_w, const at::Tensor& ada,
          at::Tensor o, at::Tensor a, int64_t blk, int64_t n_stack) {
  CHECK_BF16_CUDA(qkv, "qkv");
  CHECK_BF16_CUDA(zb, "zb");
  CHECK_BF16_CUDA(o, "o");
  CHECK_BF16_CUDA(a, "a");
  TORCH_CHECK(idx.scalar_type() == at::kInt && idx.is_contiguous(),
              "idx must be contiguous int32");
  const int num_blocks = idx.size(0);
  const int n_atom = a.size(0);
  TORCH_CHECK(idx.size(1) == kK && a.size(1) == kC && o.size(1) == kC);
  TORCH_CHECK(zb.size(0) == num_blocks * kQ * kK && zb.size(1) == n_stack * kH,
              "zb must be [num_blocks*n_query*n_key, n_stack*no_heads]");
  TORCH_CHECK(blk >= 0 && blk < n_stack, "block index out of range");
  TORCH_CHECK(mask_qp.size(0) >= num_blocks * kQ, "mask_qp must cover the pad");
  const c10::cuda::CUDAGuard guard(a.device());
  const auto stream = c10::cuda::getCurrentCUDAStream();
  attn_pv_kernel<<<num_blocks * kH * kQG, 256, 0, stream>>>(
      reinterpret_cast<const bf16*>(qkv.data_ptr()),
      idx.data_ptr<int>(),
      reinterpret_cast<const bf16*>(zb.data_ptr()),
      reinterpret_cast<const bf16*>(mask_qp.data_ptr()),
      reinterpret_cast<const bf16*>(mask_k.data_ptr()),
      reinterpret_cast<bf16*>(o.data_ptr()), n_atom,
      static_cast<int>(blk), static_cast<int>(n_stack));
  attn_epi_kernel<<<n_atom, kC, 0, stream>>>(
      reinterpret_cast<const bf16*>(o.data_ptr()),
      reinterpret_cast<const bf16*>(g.data_ptr()),
      reinterpret_cast<const bf16*>(o_w.data_ptr()),
      reinterpret_cast<const bf16*>(ada.data_ptr()),
      reinterpret_cast<bf16*>(a.data_ptr()), n_atom);
}

void trans(const at::Tensor& gate_t, const at::Tensor& add_t,
           const at::Tensor& sw_w, const at::Tensor& o_w,
           const at::Tensor& raw_t, const at::Tensor& mask, at::Tensor a) {
  CHECK_BF16_CUDA(a, "a");
  CHECK_BF16_CUDA(sw_w, "sw_w");
  const int n_atom = a.size(0);
  TORCH_CHECK(a.size(1) == kC && sw_w.size(1) == 2 * kFF && o_w.size(0) == kFF);
  const c10::cuda::CUDAGuard guard(a.device());
  trans_kernel<<<n_atom, kC, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(gate_t.data_ptr()),
      reinterpret_cast<const bf16*>(add_t.data_ptr()),
      reinterpret_cast<const bf16*>(sw_w.data_ptr()),
      reinterpret_cast<const bf16*>(o_w.data_ptr()),
      reinterpret_cast<const bf16*>(raw_t.data_ptr()),
      reinterpret_cast<const bf16*>(mask.data_ptr()),
      reinterpret_cast<bf16*>(a.data_ptr()), n_atom);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("layout", &layout, "blocked key-index table, bf16-faithful");
  m.def("qkvg", &qkvg, "AdaLN + Q/K/V/gate projections, per atom");
  m.def("attn", &attn, "sequence-local blocked attention + output projection");
  m.def("trans", &trans, "AdaLN + SwiGLU transition + gate + residual");
}
