// Fused MSAModuleEmbedder (AF3 Algorithm 8, lines 1-4) in one kernel launch.
//
//   out[s,t,c] = bf16( bf16(sum_k feat[s,t,k]*Wm[c,k]) + bf16(sum_j sin[t,j]*Ws[c,j]) )
//   feat[s,t,k] = msa[s,t,k]           k <  KM
//                 has_deletion[s,t]    k == KM
//                 deletion_value[s,t]  k == KM+1
//
// The captured shape is tiny (86 KB of traffic, ~0.7 MFMA), so this is pure
// launch/dispatch-latency work: one kernel, one wave, every load issued before
// any dependent use, and Programmatic Dependent Launch so the block prologue
// (the weight loads, which no predecessor kernel writes) runs while the
// producer kernel's tail is still draining.
//
// Block (t, cg) owns token t and channels [cg*CG, cg*CG+CG); warp w inside the
// block owns one channel. Grid = (T, C/CG) blocks of 32*CG threads.
//
// Two kernels: one templated on the captured shape (S, KM, C, J static -- no
// predicates, no tail loop, no 64-bit address math in the loops) and a
// runtime-shape fallback that keeps the op correct for anything else.
//
// Compile-time knobs (defaults are the shipped configuration):
//   MSA_CG     channels (= warps) per block
//   MSA_PDL    0 = plain launch, 1 = PDL + grid-dep sync after the weight
//              loads, 2 = PDL + grid-dep sync at kernel entry
//   MSA_SHM    1 = stage the Ws slice and the s_input row through shared memory
//              with 16-byte global loads (few outstanding global requests per
//              thread) instead of 2-byte per-lane loads
//   MSA_EMPTY  1 = empty kernel body, 2 = also skip the launch (host path only)
//   MSA_GEN    1 = always take the runtime-shape kernel (A/B reference)

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#ifndef MSA_CG
#define MSA_CG 8
#endif
#ifndef MSA_PDL
#define MSA_PDL 1
#endif
#ifndef MSA_SHM
#define MSA_SHM 0
#endif
#ifndef MSA_EMPTY
#define MSA_EMPTY 0
#endif
#ifndef MSA_GEN
#define MSA_GEN 0
#endif

// Register budget for the staged weight rows of the fallback kernel.
#define MSA_WSREG 16  // ceil(J / 32) <= 16  ->  J <= 512
#define MSA_WMREG 12  // ceil(K / 4)  <= 12  ->  K <= 48

using bf16 = __nv_bfloat16;

namespace {

__device__ __forceinline__ float ld(const bf16* p) { return __bfloat162float(*p); }

__device__ __forceinline__ void grid_dep_sync() {
#if MSA_PDL && __CUDA_ARCH__ >= 900
  cudaGridDependencySynchronize();
#endif
}

// Round-trip through bf16 exactly where the reference does: torch materializes
// linear_m(msa_feat) and linear_s_input(s_input) as bf16 tensors and then adds.
__device__ __forceinline__ bf16 combine(float q, float p) {
  return __float2bfloat16(__bfloat162float(__float2bfloat16(q)) +
                          __bfloat162float(__float2bfloat16(p)));
}

// ---------------------------------------------------------------------------
// Static fast path: S, KM, C, J known at compile time; only T stays runtime.
// ---------------------------------------------------------------------------
template <int S, int KM, int C, int J, int CG>
__global__ __launch_bounds__(32 * CG) void msa_embed_static(
    const bf16* __restrict__ msa, const bf16* __restrict__ hasdel,
    const bf16* __restrict__ delval, const bf16* __restrict__ sinp,
    const bf16* __restrict__ wm, const bf16* __restrict__ ws, bf16* __restrict__ out,
    int T) {
#if MSA_EMPTY
  grid_dep_sync();
  if (threadIdx.x == 1024) out[0] = msa[0];
#else
  constexpr int K = KM + 2;
  constexpr int NFULL = J / 32;  // fully-populated 32-lane rounds
  constexpr int JREM = J % 32;   // lanes active in the tail round
  constexpr int NK = KM / 4;     // msa elements per lane (4 lanes per dot)
  static_assert(S % 8 == 0, "S must be a multiple of the 8 sequence slots");
  constexpr int NS = S / 8;      // sequences per lane

  const int t = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int c = blockIdx.y * CG + (threadIdx.x >> 5);
  const int kq = lane & 3;
  const int sl = lane >> 2;

#if MSA_PDL == 2
  grid_dep_sync();
#endif

#if MSA_SHM
  // Ws rows [c0, c0+CG) are contiguous, and c0*J is a multiple of 8 elements
  // whenever CG is, so the whole slice is one 16-byte-aligned flat copy.
  static_assert(CG % 8 == 0, "shared-memory staging needs CG % 8 == 0");
  constexpr int WSN = CG * J;                  // elements of Ws this block needs
  constexpr int SINW = ((J + 7) / 8 + 1) * 8;  // s_input row + alignment slack
  __shared__ __align__(16) bf16 sh_ws[WSN];
  __shared__ __align__(16) bf16 sh_sin[SINW];
  {
    const int c0 = blockIdx.y * CG;
    const uint4* src = (const uint4*)(ws + c0 * J);
    uint4* dst = (uint4*)sh_ws;
    for (int i = threadIdx.x; i < WSN / 8; i += 32 * CG) dst[i] = src[i];
  }
  __syncthreads();

#if MSA_PDL == 1
  grid_dep_sync();
#endif

  // s_input row: copy the 16-byte-aligned window that covers it, so element j
  // of the row lands at sh_sin[m + j].
  const bf16* sinrow = sinp + t * J;
  const int m = (int)((((uintptr_t)sinrow) >> 1) & 7);
  {
    const uint4* src = (const uint4*)(sinrow - m);
    uint4* dst = (uint4*)sh_sin;
    const int n = (m + J + 7) / 8;
    for (int i = threadIdx.x; i < n; i += 32 * CG) dst[i] = src[i];
  }
#else
  // ---- group 1: weights (no predecessor kernel writes these) ----
  const bf16* wsrow = ws + c * J + lane;
  float wsv[NFULL + 1];
#pragma unroll
  for (int i = 0; i < NFULL; ++i) wsv[i] = ld(wsrow + 32 * i);
  wsv[NFULL] = (lane < JREM) ? ld(wsrow + 32 * NFULL) : 0.f;
#endif  // MSA_SHM

  const bf16* wmrow = wm + c * K + kq * NK;
  float wmv[NK + 2];
#pragma unroll
  for (int i = 0; i < NK; ++i) wmv[i] = ld(wmrow + i);
  wmv[NK] = (kq == 0) ? ld(wm + c * K + KM) : 0.f;
  wmv[NK + 1] = (kq == 0) ? ld(wm + c * K + KM + 1) : 0.f;

#if !MSA_SHM
#if MSA_PDL == 1
  grid_dep_sync();
#endif

  // ---- group 2: activations, all issued before the first dependent use ----
  const bf16* sinrow = sinp + t * J + lane;
  float sv[NFULL + 1];
#pragma unroll
  for (int i = 0; i < NFULL; ++i) sv[i] = ld(sinrow + 32 * i);
  sv[NFULL] = (lane < JREM) ? ld(sinrow + 32 * NFULL) : 0.f;
#endif

  // msa[s, t, kq*NK .. +NK) is 16-byte aligned for NK a multiple of 8.
  float fv[NK + 2];
  int st[NS];
#pragma unroll
  for (int n = 0; n < NS; ++n) st[n] = (sl + 8 * n) * T + t;
  {
    static_assert(NK % 8 == 0, "vector msa load needs NK % 8 == 0");
    const uint4 v = *(const uint4*)(msa + st[0] * KM + kq * NK);
    const bf16* vp = (const bf16*)&v;
#pragma unroll
    for (int i = 0; i < NK; ++i) fv[i] = ld(vp + i);
    fv[NK] = (kq == 0) ? ld(hasdel + st[0]) : 0.f;
    fv[NK + 1] = (kq == 0) ? ld(delval + st[0]) : 0.f;
  }

#if MSA_SHM
  __syncthreads();
  const bf16* wsr = sh_ws + (c - blockIdx.y * CG) * J + lane;
  const bf16* sir = sh_sin + m + lane;
#else
  const bf16* wsr = nullptr;
  const bf16* sir = nullptr;
  (void)wsr;
  (void)sir;
#endif

  // ---- s_input projection: p[c] = dot(sinrow, wsrow) over 32 lanes ----
  float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
#pragma unroll
  for (int i = 0; i < NFULL; i += 4) {
#if MSA_SHM
    p0 = fmaf(ld(wsr + 32 * i), ld(sir + 32 * i), p0);
    if (i + 1 < NFULL) p1 = fmaf(ld(wsr + 32 * (i + 1)), ld(sir + 32 * (i + 1)), p1);
    if (i + 2 < NFULL) p2 = fmaf(ld(wsr + 32 * (i + 2)), ld(sir + 32 * (i + 2)), p2);
    if (i + 3 < NFULL) p3 = fmaf(ld(wsr + 32 * (i + 3)), ld(sir + 32 * (i + 3)), p3);
#else
    p0 = fmaf(wsv[i], sv[i], p0);
    if (i + 1 < NFULL) p1 = fmaf(wsv[i + 1], sv[i + 1], p1);
    if (i + 2 < NFULL) p2 = fmaf(wsv[i + 2], sv[i + 2], p2);
    if (i + 3 < NFULL) p3 = fmaf(wsv[i + 3], sv[i + 3], p3);
#endif
  }
  if (JREM && lane < JREM) {
#if MSA_SHM
    p0 = fmaf(ld(wsr + 32 * NFULL), ld(sir + 32 * NFULL), p0);
#else
    p0 = fmaf(wsv[NFULL], sv[NFULL], p0);
#endif
  }
  float p = (p0 + p1) + (p2 + p3);
#pragma unroll
  for (int off = 16; off; off >>= 1) p += __shfl_xor_sync(0xffffffffu, p, off);

  // ---- msa projection: q[s,c] = dot(feat[s,t,:], wmrow) over 4 lanes ----
  float q = 0.f;
#pragma unroll
  for (int i = 0; i < NK + 2; ++i) q = fmaf(wmv[i], fv[i], q);
  q += __shfl_xor_sync(0xffffffffu, q, 1);
  q += __shfl_xor_sync(0xffffffffu, q, 2);
  if (kq == 0) out[st[0] * C + c] = combine(q, p);

  // ---- remaining sequence slots (NS == 1 for the captured shape) ----
#pragma unroll
  for (int n = 1; n < NS; ++n) {
    const uint4 v = *(const uint4*)(msa + st[n] * KM + kq * NK);
    const bf16* vp = (const bf16*)&v;
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < NK; ++i) acc = fmaf(wmv[i], ld(vp + i), acc);
    if (kq == 0) {
      acc = fmaf(wmv[NK], ld(hasdel + st[n]), acc);
      acc = fmaf(wmv[NK + 1], ld(delval + st[n]), acc);
    }
    acc += __shfl_xor_sync(0xffffffffu, acc, 1);
    acc += __shfl_xor_sync(0xffffffffu, acc, 2);
    if (kq == 0) out[st[n] * C + c] = combine(acc, p);
  }
#endif  // MSA_EMPTY
}

// ---------------------------------------------------------------------------
// Runtime-shape fallback: same decomposition, everything dynamic.
// ---------------------------------------------------------------------------
template <int CG>
__global__ __launch_bounds__(32 * CG) void msa_embed_generic(
    const bf16* __restrict__ msa, const bf16* __restrict__ hasdel,
    const bf16* __restrict__ delval, const bf16* __restrict__ sinp,
    const bf16* __restrict__ wm, const bf16* __restrict__ ws, bf16* __restrict__ out,
    int S, int T, int KM, int C, int J) {
  const int t = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int c = blockIdx.y * CG + (threadIdx.x >> 5);
  const int K = KM + 2;
  const int kq = lane & 3;
  const int sl = lane >> 2;
  if (c >= C) return;

  const bf16* wsrow = ws + (size_t)c * J;
  const bf16* wmrow = wm + (size_t)c * K;
  float wsv[MSA_WSREG];
#pragma unroll
  for (int i = 0; i < MSA_WSREG; ++i) {
    const int j = lane + 32 * i;
    wsv[i] = (j < J) ? ld(wsrow + j) : 0.f;
  }
  float wmv[MSA_WMREG];
#pragma unroll
  for (int i = 0; i < MSA_WMREG; ++i) {
    const int k = kq + 4 * i;
    wmv[i] = (k < K) ? ld(wmrow + k) : 0.f;
  }

  grid_dep_sync();

  const bf16* sinrow = sinp + (size_t)t * J;
  float p = 0.f;
#pragma unroll
  for (int i = 0; i < MSA_WSREG; ++i) {
    const int j = lane + 32 * i;
    if (j < J) p = fmaf(wsv[i], ld(sinrow + j), p);
  }
#pragma unroll
  for (int off = 16; off; off >>= 1) p += __shfl_xor_sync(0xffffffffu, p, off);

  for (int s = sl; s < S; s += 8) {
    const int st = s * T + t;
    const bf16* mrow = msa + (size_t)st * KM;
    float q = 0.f;
#pragma unroll
    for (int i = 0; i < MSA_WMREG; ++i) {
      const int k = kq + 4 * i;
      const bf16* src = (k < KM)        ? (mrow + k)
                        : (k == KM)     ? (hasdel + st)
                        : (k == KM + 1) ? (delval + st)
                                        : nullptr;
      if (src) q = fmaf(wmv[i], ld(src), q);
    }
    q += __shfl_xor_sync(0xffffffffu, q, 1);
    q += __shfl_xor_sync(0xffffffffu, q, 2);
    if (kq == 0) out[(size_t)st * C + c] = combine(q, p);
  }
}

struct Args {
  const bf16 *msa, *hasdel, *delval, *sinp, *wm, *ws;
  bf16* out;
  int S, T, KM, C, J;
};

template <typename F, typename... A>
inline void launch_pdl(F kernel, dim3 grid, int block, cudaStream_t stream, A... a) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(block);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = MSA_PDL ? 1 : 0;
  cudaLaunchKernelEx(&cfg, kernel, a...);
}

inline void dispatch(const Args& x, cudaStream_t stream) {
#if MSA_EMPTY == 2
  return;  // host path only: isolates the per-call CPU cost
#else
  const dim3 grid(x.T, x.C / MSA_CG);
  constexpr int BLK = 32 * MSA_CG;
#if !MSA_GEN
  if (x.S == 8 && x.KM == 32 && x.C == 64 && x.J == 449) {
    launch_pdl(msa_embed_static<8, 32, 64, 449, MSA_CG>, grid, BLK, stream, x.msa,
               x.hasdel, x.delval, x.sinp, x.wm, x.ws, x.out, x.T);
    return;
  }
#endif
  launch_pdl(msa_embed_generic<MSA_CG>, grid, BLK, stream, x.msa, x.hasdel, x.delval,
             x.sinp, x.wm, x.ws, x.out, x.S, x.T, x.KM, x.C, x.J);
#endif
}

}  // namespace

// ---------------------------------------------------------------------------
// Entry point. Raises for anything the kernel does not cover so the Python side
// can fall back to the reference implementation.
// ---------------------------------------------------------------------------

// Interned "msa" / "has_deletion" / "deletion_value" / "msa_mask", so the
// per-call batch lookups are a hash-cached PyDict_GetItem instead of building a
// fresh py::str each time.
static PyObject* g_keys[4] = {nullptr, nullptr, nullptr, nullptr};

static inline PyObject* dget(PyObject* d, int i) {
  PyObject* o = PyDict_GetItem(d, g_keys[i]);
  TORCH_CHECK(o != nullptr && THPVariable_Check(o), "msa_embed: bad batch entry");
  return o;  // borrowed
}

static py::object msa_embed(py::handle batch_h, const at::Tensor& s_input,
                            const at::Tensor& wm, const at::Tensor& ws) {
  PyObject* d = batch_h.ptr();
  TORCH_CHECK(PyDict_Check(d), "msa_embed: batch must be a dict");
  const at::Tensor& msa = THPVariable_Unpack(dget(d, 0));
  const at::Tensor& hasdel = THPVariable_Unpack(dget(d, 1));
  const at::Tensor& delval = THPVariable_Unpack(dget(d, 2));

  TORCH_CHECK(msa.scalar_type() == at::kBFloat16 &&
                  hasdel.scalar_type() == at::kBFloat16 &&
                  delval.scalar_type() == at::kBFloat16 &&
                  s_input.scalar_type() == at::kBFloat16 &&
                  wm.scalar_type() == at::kBFloat16 && ws.scalar_type() == at::kBFloat16,
              "msa_embed: bf16 only");
  TORCH_CHECK(msa.is_contiguous() && hasdel.is_contiguous() &&
                  delval.is_contiguous() && s_input.is_contiguous() &&
                  wm.is_contiguous() && ws.is_contiguous(),
              "msa_embed: contiguous only");
  TORCH_CHECK(msa.dim() >= 3 && msa.dim() <= 8 && s_input.dim() >= 2 &&
                  wm.dim() == 2 && ws.dim() == 2,
              "msa_embed: rank");

  const int KM = (int)msa.size(-1);
  const int T = (int)msa.size(-2);
  const int J = (int)s_input.size(-1);
  const int C = (int)wm.size(0);
  const int64_t S64 = msa.numel() / ((int64_t)T * KM);
  TORCH_CHECK(KM + 2 == wm.size(1) && ws.size(0) == C && ws.size(1) == J,
              "msa_embed: weight shape");
  TORCH_CHECK(s_input.numel() == (int64_t)T * J, "msa_embed: s_input batch");
  TORCH_CHECK(hasdel.numel() == S64 * T && delval.numel() == S64 * T,
              "msa_embed: deletion shape");
  TORCH_CHECK(J <= 32 * MSA_WSREG && KM + 2 <= 4 * MSA_WMREG && C % MSA_CG == 0 &&
                  KM % 4 == 0 && S64 > 0 && T > 0,
              "msa_embed: unsupported shape");

  const int nd = (int)msa.dim();
  int64_t osz[8];
  for (int i = 0; i < nd - 1; ++i) osz[i] = msa.size(i);
  osz[nd - 1] = C;
  at::Tensor out = at::detail::empty_cuda(at::IntArrayRef(osz, nd), at::kBFloat16,
                                          msa.device(), std::nullopt);

  Args x;
  x.msa = (const bf16*)msa.const_data_ptr();
  x.hasdel = (const bf16*)hasdel.const_data_ptr();
  x.delval = (const bf16*)delval.const_data_ptr();
  x.sinp = (const bf16*)s_input.const_data_ptr();
  x.wm = (const bf16*)wm.const_data_ptr();
  x.ws = (const bf16*)ws.const_data_ptr();
  x.out = (bf16*)out.mutable_data_ptr();
  x.S = (int)S64;
  x.T = T;
  x.KM = KM;
  x.C = C;
  x.J = J;
  dispatch(x, at::cuda::getCurrentCUDAStream());

  return py::make_tuple(std::move(out),
                        py::reinterpret_borrow<py::object>(dget(d, 3)));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  g_keys[0] = PyUnicode_InternFromString("msa");
  g_keys[1] = PyUnicode_InternFromString("has_deletion");
  g_keys[2] = PyUnicode_InternFromString("deletion_value");
  g_keys[3] = PyUnicode_InternFromString("msa_mask");
  m.def("msa_embed", &msa_embed, "fused AF3 MSA module embedder");
}
