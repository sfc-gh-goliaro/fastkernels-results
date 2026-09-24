// Fused Oasis timestep embedder: two kernels, one host call.
//
// Semantics reproduced from the baseline:
//   half      = dim // 2
//   freqs[j]  = exp(-log(max_period) * j / half)          (precomputed on host)
//   emb[b, k] = cos(t[b] * freqs[k])          k <  half
//               sin(t[b] * freqs[k - half])   half <= k < 2*half
//   h[b, n]   = silu( sum_k emb[b,k] * W1[n,k] + b1[n] )
//   y[b, m]   =        sum_n h[b,n]   * W2[m,n] + b2[m]
//
// Parallel decomposition.  With B <= 6 both GEMMs are skinny GEMVs, so the only
// axis with real width is the reduction axis K -- and the whole problem is 5MB,
// i.e. pure latency, not bandwidth or FLOPs.  A row-per-warp layout gives just
// H/NWARP blocks (128 for H=1024): 1.7 warps per scheduler, nothing to hide
// memory latency with, and ~13 stall cycles between issued instructions.
//
// So instead: a block owns `RPT * G` output rows, every thread owns one float4
// of the K axis, and the K axis is reduced across the threads that share a row
// (warp shuffles, then one shared-memory pass).  For H=K=1024 that is 512-1024
// blocks / 4096-8192 warps instead of 128 / 1024, which is what actually turns
// the launch from latency-bound into bandwidth-bound.  `RPT` (rows per thread)
// trades warp count against how many times the shared operand is re-read.
//
// Precision: cuBLAS rounds *both* fp32 GEMM operands to TF32 (round-to-nearest)
// before multiplying and accumulates in fp32 whenever torch's fp32 matmul
// policy allows it.  An exact-fp32 kernel is *too accurate* to sit inside the
// harness's fp32 tolerance against such a baseline (2.1e-4 vs 2.4e-5 max abs
// deviation at K=1024), so we emulate that rounding with `cvt.rna.tf32.f32`.
// Whether cuBLAS actually takes the TF32 path depends on its per-shape
// heuristics, so the Python side probes each GEMM shape once and passes the
// policy in; `tf=false` gives exact fp32 products.  The embedding is written to
// shared already rounded, and k1 stores `h` already rounded, so the rounding of
// each GEMM's activation operand is paid once rather than once per consumer.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ float to_tf32(float x) {
    unsigned int r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return __uint_as_float(r);
}

template <bool TF32>
__device__ __forceinline__ float rnd(float x) {
    return TF32 ? to_tf32(x) : x;
}

// sincos with the argument pre-reduced modulo 2pi.
//
// CUDA's `sincosf` switches to a Payne-Hanek reduction above |x| ~ 105, and the
// embedding's first few frequencies land exactly there (freqs[0] = 1, so the
// argument is t itself).  Those lanes live in warp 0, the slow path is taken for
// the whole warp, and every other warp in the block then waits on it at the
// `__syncthreads` -- ~1.5us of the kernel.  Reducing first with a two-term 2pi
// (error ~2.4e-7 rad, i.e. well under a TF32 ulp of the result, which is what
// the embedding is rounded to anyway) keeps every lane on the fast path.
__device__ __forceinline__ void sincos_red(float a, float* sn, float* cs) {
    constexpr float TWO_PI_HI = 6.28318548202514648e+00f;
    constexpr float TWO_PI_LO = -1.74845553146951715e-07f;
    constexpr float INV_TWO_PI = 1.59154936671257019e-01f;
    const float q = rintf(a * INV_TWO_PI);
    float r = fmaf(-q, TWO_PI_HI, a);
    r = fmaf(-q, TWO_PI_LO, r);
    sincosf(r, sn, cs);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

// K-axis geometry.  `nk` = K/4 float4 per row, required to be a power of two so
// a thread's (row-group, k-slot) split is two shifts instead of two runtime
// integer divisions -- those divisions cost ~100 instructions per thread, which
// on a kernel whose real work is 24 FMAs is the dominant cost.  Everything else
// (rows per block, k-passes, warps per row-group) is computed on the host and
// passed in.
//   nk <  TPB : TPB/nk row-groups per block, one k-pass
//   nk >= TPB : one row-group, nk/TPB k-passes
struct Geo {
    int rpb;      // output rows covered per block
    int passes;   // k-passes per thread
    int wpg;      // warps per row-group
    int log_nk;   // log2(K/4)
};

__host__ __forceinline__ Geo geo_of(int nk, int tpb, int rpt) {
    Geo q;
    int lg = 0;
    while ((1 << lg) < nk) ++lg;
    q.log_nk = lg;
    if (nk >= tpb) {
        q.rpb = rpt; q.passes = nk / tpb; q.wpg = tpb >> 5;
    } else {
        q.rpb = (tpb / nk) * rpt; q.passes = 1; q.wpg = nk >> 5;
    }
    return q;
}

// ---------------------------------------------------------------------------
// k1: sinusoidal embedding (shared, pre-rounded) then GEMV1 + bias + SiLU.
// A single sincosf serves both halves (cos at k, sin at k+half).
// Dynamic shared: B*dim floats for the embedding + TPB/32*RPT*B for reduction.
// ---------------------------------------------------------------------------
template <int B, int NWARP, int RPT, int MINB, bool TF1>
__global__ __launch_bounds__(NWARP * 32, MINB) void k1(
        const long long* __restrict__ t,
        const float* __restrict__ freqs,   // [half]
        const float* __restrict__ W1,      // [H, dim]
        const float* __restrict__ b1,      // [H]
        float* __restrict__ h,             // [B, H]
        int dim, int half_, int H, bool tf2,
        int rpb, int passes, int wpg, int log_nk, bool pdl) {
    constexpr int TPB = NWARP * 32;
    extern __shared__ float smem[];
    float* emb = smem;                     // [B][dim]
    float* red = smem + B * dim;           // [NWARP][RPT][B]
    float* bs = red + NWARP * RPT * B;     // [rpb]  bias, staged early

    const int tid = threadIdx.x;
    const int nk = 1 << log_nk;
    const int g = tid >> log_nk;           // row-group (0 when nk >= TPB)
    const int kc0 = tid & (nk - 1);
    const int m0 = blockIdx.x * rpb + g * RPT;

    // Issue every cold global load first: after the harness's 265MB L2 flush a
    // miss costs ~1us, so anything left at the end of the dependency chain (the
    // bias used to be) adds a full round trip to the kernel's runtime.
    if (tid < rpb) {
        const int m = blockIdx.x * rpb + tid;
        bs[tid] = b1[m < H ? m : H - 1];
    }
    float4 w0[RPT];
#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        const int m = m0 + r;
        w0[r] = reinterpret_cast<const float4*>(
                W1 + static_cast<size_t>(m < H ? m : H - 1) * dim)[kc0];
    }

    float acc[RPT][B];
#pragma unroll
    for (int r = 0; r < RPT; ++r)
#pragma unroll
        for (int b = 0; b < B; ++b) acc[r][b] = 0.0f;

    // Flatten (b, k) so all TPB threads share the transcendental work rather
    // than leaving the upper warps idle while the lower ones do B sincos each.
    // half_ is a power of two (dim/4 is, and half_ = 2*(dim/4)), so the split is
    // a shift and a mask.
    const int log_half = log_nk + 1;
    for (int idx = tid; idx < B * half_; idx += TPB) {
        const int b = idx >> log_half;
        const int k = idx & (half_ - 1);
        float s, c;
        sincos_red(static_cast<float>(t[b]) * freqs[k], &s, &c);
        float* row = emb + b * dim;
        row[k] = rnd<TF1>(c);
        row[half_ + k] = rnd<TF1>(s);
    }
    __syncthreads();

    const float4* __restrict__ e4 = reinterpret_cast<const float4*>(emb);
    for (int p = 0; p < passes; ++p) {
        const int kc = kc0 + p * TPB;
        float4 wv[RPT];
#pragma unroll
        for (int r = 0; r < RPT; ++r) {
            if (p == 0) {
                wv[r] = w0[r];
            } else {
                const int m = m0 + r;
                wv[r] = reinterpret_cast<const float4*>(
                        W1 + static_cast<size_t>(m < H ? m : H - 1) * dim)[kc];
            }
        }
#pragma unroll
        for (int r = 0; r < RPT; ++r) {
            const float wx = rnd<TF1>(wv[r].x), wy = rnd<TF1>(wv[r].y);
            const float wz = rnd<TF1>(wv[r].z), ww = rnd<TF1>(wv[r].w);
#pragma unroll
            for (int b = 0; b < B; ++b) {
                const float4 ev = e4[b * nk + kc];
                acc[r][b] = fmaf(wx, ev.x, acc[r][b]);
                acc[r][b] = fmaf(wy, ev.y, acc[r][b]);
                acc[r][b] = fmaf(wz, ev.z, acc[r][b]);
                acc[r][b] = fmaf(ww, ev.w, acc[r][b]);
            }
        }
    }

    const int warp = tid >> 5;
#pragma unroll
    for (int r = 0; r < RPT; ++r)
#pragma unroll
        for (int b = 0; b < B; ++b) {
            const float v = warp_sum(acc[r][b]);
            if ((tid & 31) == 0) red[(warp * RPT + r) * B + b] = v;
        }
    __syncthreads();

    if (tid < rpb * B) {
        const int b = tid % B;
        const int rr = tid / B;                 // gg * RPT + r
        const int gg = rr / RPT, r = rr - gg * RPT;
        const int m = blockIdx.x * rpb + rr;
        float sum = 0.0f;
        for (int w = 0; w < wpg; ++w) sum += red[((gg * wpg + w) * RPT + r) * B + b];
        if (m < H) {
            const float v = sum + bs[rr];
            const float hv = v / (1.0f + expf(-v));          // SiLU in exact fp32
            h[b * H + m] = tf2 ? to_tf32(hv) : hv;
        }
    }
    // Release k2's blocks as soon as this block's slice of h is visible.  ncu
    // puts k1 at 0.35 waves/SM, so there are idle SMs for k2 to start on while
    // k1's tail drains -- k2 spends that window fetching its own code, constant
    // bank, bias and W2 tile, none of which depend on h.
    if (pdl) {
        __syncthreads();
        cudaTriggerProgrammaticLaunchCompletion();
    }
}

// ---------------------------------------------------------------------------
// k2: GEMV2 + bias.  Same decomposition; `h` arrives pre-rounded from k1 and is
// read straight from global (it is 24KB, so it lives in L1/L2 across blocks).
// Dynamic shared: TPB/32*RPT*B floats for the reduction.
// ---------------------------------------------------------------------------
template <int B, int NWARP, int RPT, int MINB, bool TF2>
__global__ __launch_bounds__(NWARP * 32, MINB) void k2(
        const float* __restrict__ h,       // [B, K]  (pre-rounded)
        const float* __restrict__ W2,      // [M, K]
        const float* __restrict__ b2,      // [M]
        float* __restrict__ y,             // [B, M]
        int K, int M, int rpb, int passes, int wpg, int log_nk, bool pdl) {
    constexpr int TPB = NWARP * 32;
    extern __shared__ float smem[];
    float* red = smem;                     // [NWARP][RPT][B]
    float* bs = red + NWARP * RPT * B;     // [rpb]  bias, staged early

    const int tid = threadIdx.x;
    const int nk = 1 << log_nk;
    const int g = tid >> log_nk;
    const int kc0 = tid & (nk - 1);
    const int m0 = blockIdx.x * rpb + g * RPT;

    if (tid < rpb) {                       // cold load issued before the GEMV
        const int m = blockIdx.x * rpb + tid;
        bs[tid] = b2[m < M ? m : M - 1];
    }

    float acc[RPT][B];
#pragma unroll
    for (int r = 0; r < RPT; ++r)
#pragma unroll
        for (int b = 0; b < B; ++b) acc[r][b] = 0.0f;

    // Prefetch the first W2 tile (and the bias above) -- neither is written by
    // k1 -- then park on the grid dependency as late as possible.
    float4 w0[RPT];
#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        const int m = m0 + r;
        w0[r] = reinterpret_cast<const float4*>(
                W2 + static_cast<size_t>(m < M ? m : M - 1) * K)[kc0];
    }
    if (pdl) cudaGridDependencySynchronize();

    for (int p = 0; p < passes; ++p) {
        const int kc = kc0 + p * TPB;
        float4 wv[RPT];
#pragma unroll
        for (int r = 0; r < RPT; ++r) {
            if (p == 0) {
                wv[r] = w0[r];
            } else {
                const int m = m0 + r;
                wv[r] = reinterpret_cast<const float4*>(
                        W2 + static_cast<size_t>(m < M ? m : M - 1) * K)[kc];
            }
        }
        float4 hv[B];
#pragma unroll
        for (int b = 0; b < B; ++b)
            hv[b] = reinterpret_cast<const float4*>(h + b * K)[kc];
#pragma unroll
        for (int r = 0; r < RPT; ++r) {
            const float wx = rnd<TF2>(wv[r].x), wy = rnd<TF2>(wv[r].y);
            const float wz = rnd<TF2>(wv[r].z), ww = rnd<TF2>(wv[r].w);
#pragma unroll
            for (int b = 0; b < B; ++b) {
                acc[r][b] = fmaf(wx, hv[b].x, acc[r][b]);
                acc[r][b] = fmaf(wy, hv[b].y, acc[r][b]);
                acc[r][b] = fmaf(wz, hv[b].z, acc[r][b]);
                acc[r][b] = fmaf(ww, hv[b].w, acc[r][b]);
            }
        }
    }

    const int warp = tid >> 5;
#pragma unroll
    for (int r = 0; r < RPT; ++r)
#pragma unroll
        for (int b = 0; b < B; ++b) {
            const float v = warp_sum(acc[r][b]);
            if ((tid & 31) == 0) red[(warp * RPT + r) * B + b] = v;
        }
    __syncthreads();

    if (tid < rpb * B) {
        const int b = tid % B;
        const int rr = tid / B;
        const int gg = rr / RPT, r = rr - gg * RPT;
        const int m = blockIdx.x * rpb + rr;
        float sum = 0.0f;
        for (int w = 0; w < wpg; ++w) sum += red[((gg * wpg + w) * RPT + r) * B + b];
        if (m < M) y[b * M + m] = sum + bs[rr];
    }
}

// --- launch helpers ---------------------------------------------------------
// Launch `kernel` with `args...`, optionally marking it as programmatically
// dependent on the preceding kernel in the stream (PDL).
template <typename K, typename... Args>
void launch_ex(bool pdl, dim3 grid, dim3 block, size_t shmem, cudaStream_t s,
               K kernel, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = shmem;
    cfg.stream = s;
    cudaLaunchAttribute attr[1];
    if (pdl) {
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
    }
    cudaLaunchKernelEx(&cfg, kernel, args...);
}

template <int B, bool TF1>
void launch1(const void* t, const void* freqs, const void* w1, const void* b1,
             void* h, int dim, int half_, int H, bool tf2, bool pdl, cudaStream_t s) {
    const long long* tp = static_cast<const long long*>(t);
    const float* fp = static_cast<const float*>(freqs);
    const float* wp = static_cast<const float*>(w1);
    const float* bp = static_cast<const float*>(b1);
    float* hp = static_cast<float*>(h);
    const int nk = dim >> 2;
#define K1_CASE(NW, RPT, MINB)                                                       \
    do {                                                                             \
        const Geo q = geo_of(nk, (NW) * 32, RPT);                                    \
        const size_t sh = (static_cast<size_t>(B) * dim + (NW) * (RPT) * (B + 1))      \
                          * sizeof(float);                                            \
        k1<B, NW, RPT, MINB, TF1><<<(H + q.rpb - 1) / q.rpb, NW * 32, sh, s>>>(       \
                tp, fp, wp, bp, hp, dim, half_, H, tf2,                               \
                q.rpb, q.passes, q.wpg, q.log_nk, pdl);                               \
    } while (0)
    // Only the tuned geometry is instantiated -- every extra
    // (warps, rows/thread, min-blocks-per-SM) triple is 16 more template
    // instantiations of nvcc time, which the bench pays whenever the extension
    // cache is cold.  The tables that were swept are in ITERATIONS.md; add cases
    // back here plus a selector arg to sweep again.  Must match _K1_GEOM.
    K1_CASE(8, 2, 1);
#undef K1_CASE
}

template <int B, bool TF2>
void launch2(const void* h, const void* w2, const void* b2, void* y,
             int K, int M, bool pdl, cudaStream_t s) {
    const float* hp = static_cast<const float*>(h);
    const float* wp = static_cast<const float*>(w2);
    const float* bp = static_cast<const float*>(b2);
    float* yp = static_cast<float*>(y);
    const int nk = K >> 2;
#define K2_CASE(NW, RPT, MINB)                                                       \
    do {                                                                             \
        const Geo q = geo_of(nk, (NW) * 32, RPT);                                    \
        const size_t sh = static_cast<size_t>((NW) * (RPT) * (B + 1)) * sizeof(float); \
        launch_ex(pdl, dim3((M + q.rpb - 1) / q.rpb), dim3(NW * 32), sh, s,           \
                  k2<B, NW, RPT, MINB, TF2>,                                          \
                  hp, wp, bp, yp, K, M, q.rpb, q.passes, q.wpg, q.log_nk, pdl);        \
    } while (0)
    K2_CASE(8, 2, 4);   // must match _K2_GEOM
#undef K2_CASE
}

template <int B>
void launch(int tf1, int tf2, int pdl,
            const void* t, const void* freqs, const void* w1, const void* b1,
            const void* w2, const void* b2, void* h, void* y,
            int dim, int half_, int H, cudaStream_t s) {
    const bool p = pdl != 0;
    if (tf1) launch1<B, true>(t, freqs, w1, b1, h, dim, half_, H, tf2 != 0, p, s);
    else launch1<B, false>(t, freqs, w1, b1, h, dim, half_, H, tf2 != 0, p, s);
    if (tf2) launch2<B, true>(h, w2, b2, y, H, H, p, s);
    else launch2<B, false>(h, w2, b2, y, H, H, p, s);
}

}  // namespace

// Raw-pointer entry point: every argument is a plain integer, so the pybind
// trampoline is all the host pays (weight pointers are cached Python-side).
void oasis_fwd(int64_t t_ptr, int64_t freqs_ptr, int64_t w1_ptr, int64_t b1_ptr,
               int64_t w2_ptr, int64_t b2_ptr, int64_t h_ptr, int64_t y_ptr,
               int64_t B, int64_t dim, int64_t half_, int64_t H, int64_t flags) {
    cudaStream_t s = at::cuda::getCurrentCUDAStream();
    const void* t = reinterpret_cast<const void*>(t_ptr);
    const void* f = reinterpret_cast<const void*>(freqs_ptr);
    const void* w1 = reinterpret_cast<const void*>(w1_ptr);
    const void* c1 = reinterpret_cast<const void*>(b1_ptr);
    const void* w2 = reinterpret_cast<const void*>(w2_ptr);
    const void* c2 = reinterpret_cast<const void*>(b2_ptr);
    void* h = reinterpret_cast<void*>(h_ptr);
    void* y = reinterpret_cast<void*>(y_ptr);
    const int d = static_cast<int>(dim), hf = static_cast<int>(half_);
    const int HH = static_cast<int>(H);
    // flags: bit0 tf32(GEMV1), bit1 tf32(GEMV2), bit10 pdl
    const int tf1 = static_cast<int>(flags & 1);
    const int tf2 = static_cast<int>((flags >> 1) & 1);
    const int pdl = static_cast<int>((flags >> 10) & 1);
#define B_CASE(N)                                                                    \
    case N: launch<N>(tf1, tf2, pdl, t, f, w1, c1, w2, c2, h, y,                      \
                      d, hf, HH, s); break
    switch (B) {
        B_CASE(1); B_CASE(2); B_CASE(3); B_CASE(4);
        B_CASE(5); B_CASE(6); B_CASE(7); B_CASE(8);
        default: TORCH_CHECK(false, "oasis_fwd: batch ", B, " not specialized");
    }
#undef B_CASE
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("oasis_fwd", &oasis_fwd, "fused oasis timestep embedder");
}
