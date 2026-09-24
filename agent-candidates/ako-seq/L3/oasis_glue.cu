// Bit-exact fused glue kernels for the Oasis DiT forward.
//
// Every arithmetic expression here reproduces, operation for operation, the
// rounding sequence of the ATen chain it replaces -- the DiT's output is
// pathologically sensitive (a 1-ULP change at the patch embedder puts 21% of
// the output outside the harness' fp32 tolerance), so these kernels are
// allowed to delete *dispatch*, never to reassociate arithmetic.  Hence
// __fmul_rn / __fadd_rn / __fsub_rn on every elementwise expression: a
// contracted FMA is a different number, and an intrinsic cannot be contracted.
//
// The intrinsics are what enforce that, NOT a compiler flag.  This file is
// compiled with nvcc's *default* contraction (fmad on) on purpose, because the
// LayerNorm statistics pass below is a transcription of an ATen kernel that is
// itself built that way: `mean + delta*(1/count)` and `sigma2 + delta*(val-mean)`
// are FMAs in ATen, and forcing them apart with -fmad=false makes the Welford
// miss by ~1 ULP -- which this operator cannot absorb.  Do not add -fmad=false
// back: it does not protect the elementwise kernels (they use intrinsics) and it
// breaks the reduction.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>
#include <cstdlib>

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float4 ld4(const float* p) {
  return *reinterpret_cast<const float4*>(p);
}
__device__ __forceinline__ void st4(float* p, const float4& v) {
  *reinterpret_cast<float4*>(p) = v;
}

// ---------------------------------------------------------------------------
// out = h * (1 + scale) + shift          (ATen: mul then add, scale bias first)
//
// h is [M, N] contiguous; `shift`/`scale` are rows of the packed modulation
// matrix (row stride sS, unit column stride) indexed by the frame m / P.
// ---------------------------------------------------------------------------
template <int U>
__global__ void k_modulate(float* __restrict__ out, const float* __restrict__ h,
                           const float* __restrict__ shift,
                           const float* __restrict__ scale,
                           long sS, int P, int Nv, long total) {
  const long base = (blockIdx.x * (long)blockDim.x) * U + threadIdx.x;
#pragma unroll
  for (int u = 0; u < U; ++u) {
    const long i = base + (long)u * blockDim.x;
    if (i >= total) return;
    long m = i / Nv;
    int nv = (int)(i - m * Nv);
    long r = m / P;
    const float4 hv = ld4(h + i * 4);
    const float4 sc = ld4(scale + r * sS + nv * 4);
    const float4 sh = ld4(shift + r * sS + nv * 4);
    float4 o;
    o.x = __fadd_rn(__fmul_rn(hv.x, __fadd_rn(1.0f, sc.x)), sh.x);
    o.y = __fadd_rn(__fmul_rn(hv.y, __fadd_rn(1.0f, sc.y)), sh.y);
    o.z = __fadd_rn(__fmul_rn(hv.z, __fadd_rn(1.0f, sc.z)), sh.z);
    o.w = __fadd_rn(__fmul_rn(hv.w, __fadd_rn(1.0f, sc.w)), sh.w);
    st4(out + i * 4, o);
  }
}

// ---------------------------------------------------------------------------
// x += gate * y                          (ATen: _gate's mul, then the residual add)
// ---------------------------------------------------------------------------
template <int U>
__global__ void k_gate_add(float* __restrict__ x, const float* __restrict__ y,
                           const float* __restrict__ gate,
                           long sS, int P, int Nv, long total) {
  const long base = (blockIdx.x * (long)blockDim.x) * U + threadIdx.x;
#pragma unroll
  for (int u = 0; u < U; ++u) {
    const long i = base + (long)u * blockDim.x;
    if (i >= total) return;
    long m = i / Nv;
    int nv = (int)(i - m * Nv);
    long r = m / P;
    const float4 xv = ld4(x + i * 4);
    const float4 yv = ld4(y + i * 4);
    const float4 g = ld4(gate + r * sS + nv * 4);
    float4 o;
    o.x = __fadd_rn(xv.x, __fmul_rn(g.x, yv.x));
    o.y = __fadd_rn(xv.y, __fmul_rn(g.y, yv.y));
    o.z = __fadd_rn(xv.z, __fmul_rn(g.z, yv.z));
    o.w = __fadd_rn(xv.w, __fmul_rn(g.w, yv.w));
    st4(x + i * 4, o);
  }
}

// ---------------------------------------------------------------------------
// x += gate*y ; hn = LayerNorm(x) * (1 + scale) + shift        -- all in one pass.
//
// Three launches (gate_add, ATen's layer_norm, modulate) at ~3.1-3.8 us each
// become one, and the HBM traffic drops from 16.5 to 9.4 MB per instance.  The
// LayerNorm half is a transcription of ATen's `vectorized_layer_norm_kernel`
// (Welford online sum, warp shuffle-down combine, then the blockDim.y tree
// through shared memory, sigma2/N before the rsqrt) -- the launch shape and the
// combine argument order are the only free parameters and `dev/lnx.py` pins
// them by `torch.equal`.  Anything else would not be the reference's number.
// ---------------------------------------------------------------------------
struct WD { float mean, sigma2, count; };

__device__ __forceinline__ WD wd_online(float val, WD c) {
  float delta = val - c.mean;
  float new_count = c.count + 1.f;
  float new_mean = c.mean + delta * (1.f / new_count);
  return {new_mean, c.sigma2 + delta * (val - new_mean), new_count};
}
__device__ __forceinline__ WD wd_combine(WD B, WD A) {
  float delta = B.mean - A.mean;
  float count = A.count + B.count;
  float mean = 0.f, sigma2 = 0.f;
  if (count > 0.f) {
    float coef = 1.f / count;
    float nA = A.count * coef;
    float nB = B.count * coef;
    mean = nA * A.mean + nB * B.mean;
    sigma2 = A.sigma2 + B.sigma2 + delta * delta * A.count * nB;
  }
  return {mean, sigma2, count};
}

// NVEC = 1024/4/numx float4s per thread, held in registers between the two passes.
template <int TY, int GATED>
__global__ void k_gate_ln_mod(float* __restrict__ hn, float* __restrict__ x,
                              const float* __restrict__ y,
                              const float* __restrict__ gate,
                              const float* __restrict__ shift,
                              const float* __restrict__ scale,
                              long sS, int P, int N, float eps) {
  extern __shared__ float buf[];
  const long row = blockIdx.x;
  const long r = row / P;
  float* xr = x + row * N;
  float* hr = hn + row * N;
  const float* yr = y + row * N;
  const int numx = 32 * TY;
  const int thrx = threadIdx.x + threadIdx.y * 32;
  const int nvec = N / 4;
  constexpr int NVEC = 8;                       // >= nvec/numx for N<=1024*... (checked host side)
  float4 keep[NVEC];
  WD wd{0.f, 0.f, 0.f};
  int c = 0;
  for (int i = thrx; i < nvec; i += numx, ++c) {
    float4 xv = ld4(xr + i * 4);
    if (GATED) {
      const float4 yv = ld4(yr + i * 4);
      const float4 g = ld4(gate + r * sS + i * 4);
      xv.x = __fadd_rn(xv.x, __fmul_rn(g.x, yv.x));
      xv.y = __fadd_rn(xv.y, __fmul_rn(g.y, yv.y));
      xv.z = __fadd_rn(xv.z, __fmul_rn(g.z, yv.z));
      xv.w = __fadd_rn(xv.w, __fmul_rn(g.w, yv.w));
      st4(xr + i * 4, xv);
    }
    keep[c] = xv;
    wd = wd_online(xv.x, wd);
    wd = wd_online(xv.y, wd);
    wd = wd_online(xv.z, wd);
    wd = wd_online(xv.w, wd);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    WD b{__shfl_down_sync(0xffffffff, wd.mean, off),
         __shfl_down_sync(0xffffffff, wd.sigma2, off),
         __shfl_down_sync(0xffffffff, wd.count, off)};
    wd = wd_combine(wd, b);
  }
  float mean, rstd;
  if (TY > 1) {
    float* ms = buf;
    float* cb = buf + TY * 2;
    for (int off = TY / 2; off > 0; off /= 2) {
      if (threadIdx.x == 0 && (int)threadIdx.y >= off && (int)threadIdx.y < 2 * off) {
        int wy = threadIdx.y - off;
        ms[2 * wy] = wd.mean; ms[2 * wy + 1] = wd.sigma2; cb[wy] = wd.count;
      }
      __syncthreads();
      if (threadIdx.x == 0 && (int)threadIdx.y < off) {
        WD b{ms[2 * threadIdx.y], ms[2 * threadIdx.y + 1], cb[threadIdx.y]};
        wd = wd_combine(wd, b);
      }
      __syncthreads();
    }
    if (thrx == 0) { ms[0] = wd.mean; ms[1] = wd.sigma2 / (float)N; }
    __syncthreads();
    mean = ms[0]; rstd = rsqrtf(ms[1] + eps);
  } else {
    mean = __shfl_sync(0xffffffff, wd.mean, 0);
    rstd = rsqrtf(__shfl_sync(0xffffffff, wd.sigma2, 0) / (float)N + eps);
  }
  c = 0;
  for (int i = thrx; i < nvec; i += numx, ++c) {
    const float4 xv = keep[c];
    const float4 sc = ld4(scale + r * sS + i * 4);
    const float4 sh = ld4(shift + r * sS + i * 4);
    float4 o;
    o.x = __fadd_rn(__fmul_rn(__fmul_rn(__fsub_rn(xv.x, mean), rstd), __fadd_rn(1.0f, sc.x)), sh.x);
    o.y = __fadd_rn(__fmul_rn(__fmul_rn(__fsub_rn(xv.y, mean), rstd), __fadd_rn(1.0f, sc.y)), sh.y);
    o.z = __fadd_rn(__fmul_rn(__fmul_rn(__fsub_rn(xv.z, mean), rstd), __fadd_rn(1.0f, sc.z)), sh.z);
    o.w = __fadd_rn(__fmul_rn(__fmul_rn(__fsub_rn(xv.w, mean), rstd), __fadd_rn(1.0f, sc.w)), sh.w);
    st4(hr + i * 4, o);
  }
}

// ---------------------------------------------------------------------------
// qkv [T*P, 3*C] -> Q, K, V in exactly the layout SDPA is handed by the
// reference chain, with the rotary folded into the Q/K loads.
//
//   t_transformed = (t * cos) + (rotate_half(t) * sin)
//   rotate_half: (x0,x1) -> (-x1, x0) per adjacent pair
//
// MODE 0 (spatial): Q/K/V are [T, H, P, D]; the (cos,sin) row index is the
//                   token p, tables are [P, D].
// MODE 1 (temporal): Q/K/V are [P, H, T, D]; the row index is the frame t,
//                   tables are [T, D].
// gridDim.y selects q (rope), k (rope), v (plain move).
// ---------------------------------------------------------------------------
template <int MODE>
__global__ void k_rope_split(float* __restrict__ q, float* __restrict__ k,
                             float* __restrict__ v,
                             const float* __restrict__ qkv,
                             const float* __restrict__ cosT,
                             const float* __restrict__ sinT,
                             int T, int P, int H, int Dv, long total) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int which = blockIdx.y;                 // 0=q 1=k 2=v
  // decompose i over the destination's own axis order
  int dv = (int)(i % Dv);
  long rest = i / Dv;
  int D = Dv * 4;
  int t, p, h;
  if (MODE == 0) {                              // [T, H, P, D]
    p = (int)(rest % P); rest /= P;
    h = (int)(rest % H); t = (int)(rest / H);
  } else {                                      // [P, H, T, D]
    t = (int)(rest % T); rest /= T;
    h = (int)(rest % H); p = (int)(rest / H);
  }
  const long src = (long)(t * P + p) * (3L * H * D) + (long)which * (H * D)
                 + (long)h * D + dv * 4;
  float* dst = (which == 0 ? q : (which == 1 ? k : v));
  const float4 in = ld4(qkv + src);
  if (which == 2) {
    st4(dst + i * 4, in);
    return;
  }
  const int row = (MODE == 0 ? p : t);
  const float4 c = ld4(cosT + (long)row * D + dv * 4);
  const float4 s = ld4(sinT + (long)row * D + dv * 4);
  float4 o;
  o.x = __fadd_rn(__fmul_rn(in.x, c.x), __fmul_rn(-in.y, s.x));
  o.y = __fadd_rn(__fmul_rn(in.y, c.y), __fmul_rn( in.x, s.y));
  o.z = __fadd_rn(__fmul_rn(in.z, c.z), __fmul_rn(-in.w, s.z));
  o.w = __fadd_rn(__fmul_rn(in.w, c.w), __fmul_rn( in.z, s.w));
  st4(dst + i * 4, o);
}

// ---------------------------------------------------------------------------
// SDPA output -> the [T*P, H*D] matrix `to_out` consumes.  The source's
// (t, p, h) strides are passed in, because SDPA hands back a permuted *view*
// whose layout depends on the backend it picked; the last axis must be dense.
// ---------------------------------------------------------------------------
__global__ void k_merge(float* __restrict__ out, const float* __restrict__ o,
                        int T, int P, int H, int Dv,
                        long st, long sp, long sh, long total) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= total) return;
  int dv = (int)(i % Dv);
  long rest = i / Dv;
  int D = Dv * 4;
  int h = (int)(rest % H); rest /= H;
  int p = (int)(rest % P);
  int t = (int)(rest / P);
  const long dstoff = (long)(t * P + p) * (H * D) + (long)h * D + dv * 4;
  st4(out + dstoff, ld4(o + t * st + p * sp + h * sh + dv * 4));
}

inline long nblocks(long total) { return (total + kThreads - 1) / kThreads; }
inline long nblocks_u(long total, int u) {
  const long per = (long)kThreads * u;
  return (total + per - 1) / per;
}

// Elements (float4s) per thread for the two elementwise glue kernels.  These
// kernels move 7 MB each and run in ~2.3 us, i.e. close to the grid-ramp floor
// rather than the bandwidth floor, so the useful knob is work per thread.
inline int glue_u() {
  static const int u = [] {
    const char* e = std::getenv("OASIS_DIT_GLUE_U");
    int v = e ? std::atoi(e) : 1;
    return (v == 2 || v == 4) ? v : 1;
  }();
  return u;
}

}  // namespace

// ---------------------------------------------------------------------------
// Host entry points.
// ---------------------------------------------------------------------------
at::Tensor oasis_modulate(const at::Tensor& h, const at::Tensor& mod,
                          int64_t shift_off, int64_t scale_off, int64_t P) {
  TORCH_CHECK(h.is_contiguous() && h.scalar_type() == at::kFloat);
  TORCH_CHECK(mod.is_contiguous() && mod.scalar_type() == at::kFloat);
  const long M = h.size(0), N = h.size(1);
  TORCH_CHECK(N % 4 == 0);
  auto out = at::empty_like(h);
  const int Nv = (int)(N / 4);
  const long total = M * Nv;
  const long sS = mod.size(1);
  const c10::cuda::CUDAGuard g(h.device());
  const int u = glue_u();
  auto st = at::cuda::getCurrentCUDAStream();
#define MODL(UU) k_modulate<UU><<<nblocks_u(total, UU), kThreads, 0, st>>>( \
      out.data_ptr<float>(), h.data_ptr<float>(), \
      mod.data_ptr<float>() + shift_off, mod.data_ptr<float>() + scale_off, \
      sS, (int)P, Nv, total)
  if (u == 4) { MODL(4); } else if (u == 2) { MODL(2); } else { MODL(1); }
  return out;
}

void oasis_gate_add(const at::Tensor& x, const at::Tensor& y, const at::Tensor& mod,
                    int64_t gate_off, int64_t P) {
  TORCH_CHECK(x.is_contiguous() && y.is_contiguous());
  const long M = x.size(0), N = x.size(1);
  const int Nv = (int)(N / 4);
  const long total = M * Nv;
  const c10::cuda::CUDAGuard g(x.device());
  const int u = glue_u();
  auto st = at::cuda::getCurrentCUDAStream();
#define GAL(UU) k_gate_add<UU><<<nblocks_u(total, UU), kThreads, 0, st>>>( \
      x.data_ptr<float>(), y.data_ptr<float>(), \
      mod.data_ptr<float>() + gate_off, mod.size(1), (int)P, Nv, total)
  if (u == 4) { GAL(4); } else if (u == 2) { GAL(2); } else { GAL(1); }
}

// emit_v=0: only Q and K are written.  V needs no rotation and no reordering --
// it is a plain strided *view* of the qkv GEMM output, and the mem-efficient FMHA
// reads it with those strides and returns bitwise the same output at the same cost
// (dev/vview.py checks both, on all 5 T x both axes).  So its copy is pure waste:
// the kernel's traffic drops from 21.2 to 14.2 MB at T=6.
std::vector<at::Tensor> oasis_rope_split(const at::Tensor& qkv,
                                         const at::Tensor& cosT,
                                         const at::Tensor& sinT,
                                         int64_t T, int64_t P, int64_t H,
                                         int64_t mode, int64_t emit_v) {
  TORCH_CHECK(qkv.is_contiguous() && qkv.scalar_type() == at::kFloat);
  const long C = qkv.size(1) / 3;
  const long D = C / H;
  TORCH_CHECK(D % 4 == 0 && qkv.size(1) == 3 * C);
  auto opts = qkv.options();
  std::vector<int64_t> shp = (mode == 0)
      ? std::vector<int64_t>{T, H, P, D} : std::vector<int64_t>{P, H, T, D};
  auto q = at::empty(shp, opts), k = at::empty(shp, opts);
  auto v = emit_v ? at::empty(shp, opts) : at::Tensor();
  const int Dv = (int)(D / 4);
  const long total = (long)T * P * H * Dv;
  dim3 grid((unsigned)nblocks(total), emit_v ? 3u : 2u);
  const c10::cuda::CUDAGuard g(qkv.device());
  auto st = at::cuda::getCurrentCUDAStream();
  float* vp = emit_v ? v.data_ptr<float>() : nullptr;
  if (mode == 0)
    k_rope_split<0><<<grid, kThreads, 0, st>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), vp,
        qkv.data_ptr<float>(), cosT.data_ptr<float>(), sinT.data_ptr<float>(),
        (int)T, (int)P, (int)H, Dv, total);
  else
    k_rope_split<1><<<grid, kThreads, 0, st>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), vp,
        qkv.data_ptr<float>(), cosT.data_ptr<float>(), sinT.data_ptr<float>(),
        (int)T, (int)P, (int)H, Dv, total);
  if (emit_v) return {q, k, v};
  return {q, k};
}

at::Tensor oasis_merge(const at::Tensor& o, int64_t T, int64_t P, int64_t H,
                       int64_t D, int64_t st, int64_t sp, int64_t sh) {
  TORCH_CHECK(o.scalar_type() == at::kFloat);
  TORCH_CHECK(D % 4 == 0);
  auto out = at::empty({T * P, H * D}, o.options());
  const int Dv = (int)(D / 4);
  const long total = (long)T * P * H * Dv;
  const c10::cuda::CUDAGuard g(o.device());
  k_merge<<<nblocks(total), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      out.data_ptr<float>(), o.data_ptr<float>(), (int)T, (int)P, (int)H, Dv,
      st, sp, sh, total);
  return out;
}

// One-pass residual + LayerNorm + modulate.  `y`/`gate_off` may be absent, in
// which case only the LayerNorm+modulate half runs (the first sub-block and the
// final layer have no residual to fold in).
at::Tensor oasis_gate_ln_mod(const at::Tensor& x, const c10::optional<at::Tensor>& y,
                             const at::Tensor& mod, int64_t gate_off,
                             int64_t shift_off, int64_t scale_off, int64_t P,
                             double eps, int64_t ty) {
  TORCH_CHECK(x.is_contiguous() && x.scalar_type() == at::kFloat && x.dim() == 2);
  const long M = x.size(0), N = x.size(1);
  TORCH_CHECK(N % 4 == 0);
  const int numx = 32 * (int)ty;
  TORCH_CHECK((N / 4 + numx - 1) / numx <= 8, "row too wide for the register cache");
  auto hn = at::empty_like(x);
  const long sS = mod.size(1);
  const float* mp = mod.data_ptr<float>();
  const bool gated = y.has_value();
  dim3 threads(32, (unsigned)ty);
  int nsh = ty > 1 ? (int)(ty * 3) * (int)sizeof(float) : 0;
  const c10::cuda::CUDAGuard g(x.device());
  auto st = at::cuda::getCurrentCUDAStream();
  const float* yp = gated ? y->data_ptr<float>() : nullptr;
  const float* gp = gated ? mp + gate_off : nullptr;
#define GLM(TYV, G)                                                            \
  k_gate_ln_mod<TYV, G><<<(unsigned)M, threads, nsh, st>>>(                     \
      hn.data_ptr<float>(), x.data_ptr<float>(), yp, gp, mp + shift_off,        \
      mp + scale_off, sS, (int)P, (int)N, (float)eps)
#define GLM_G(TYV) if (gated) { GLM(TYV, 1); } else { GLM(TYV, 0); }
  if (ty == 1)      { GLM_G(1); }
  else if (ty == 2) { GLM_G(2); }
  else if (ty == 4) { GLM_G(4); }
  else if (ty == 8) { GLM_G(8); }
  else              { TORCH_CHECK(false, "unsupported ty"); }
  return hn;
}
