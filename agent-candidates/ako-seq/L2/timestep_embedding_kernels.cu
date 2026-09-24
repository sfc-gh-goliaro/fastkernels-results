// Fused kernels for the L2/timestep_embedding operators.
//
// Every scored shape is batch 1, so what costs time is the number of device ops,
// the per-call host dispatch cost, and -- once those are gone -- how the weight
// stream is scheduled.  All three are addressed here; see ITERATIONS.md for the
// measurements.  The whole per-call path lives in this file so a `forward` is one
// pybind call, and the op graph collapses to:
//
//   Timesteps                -> 1 kernel  (sinusoid written flip-permuted, no cat)
//   TimestepEmbedding        -> 1 kernel  (both layers, split-K rank update)
//   Combined*Embeddings      -> 1 kernel  (both layers, all branches, sinusoids)
//
// The two-layer MLP is one kernel because stage 2 is written as a sum of rank-1
// updates, `out[:] = sum_k h[k] * W2T[k,:]`, instead of a row-wise dot product.
// A block that owns a k-slice then needs only *its* slice of `h`, which it
// computes itself from a few KB of `W1`, so the stage-1 weight traffic rides
// along inside the stage-2 stream instead of in a serialized kernel in front of
// it.  See `fused_kernel`.  This is worth 1-2 of the benchmark's 2.048 us
// quantisation levels on a standalone `TimestepEmbedding`, and most of it on the
// case with the wider first layer (in_channels 768), whose stage 1 is exactly the
// part that disappears.
//
// The two-kernel path below it (`run_group`, round 1's design) is kept intact: it
// still serves every group the fused tiling cannot cover, and `FK_TSE_PATH=1`
// forces it for measurement.
//
// What limits the stream is *warps*, not payloads in flight.  A warp sustains
// ~1.2-1.8 GB/s of weight traffic however many 16-byte payloads it has
// outstanding, and the warp count is `kslices * N * wk / 256`, where `wk` is how
// many warp-rows of a block split its k-slice and reduce in shared memory.  The
// split-K reduction across *blocks* costs `kslices * N` atomics, so `wk` is what
// buys bandwidth without buying reduction.  Both that and the reduction's own scheduling (a
// vectorized L1-bypassing tail, one ticket per output tile, and `bias_2` seeded
// into the k-slice-0 accumulator rather than added in the tail) are what the
// round was actually spent on; the arithmetic was correct and fast from the start.

#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <c10/cuda/CUDAStream.h>

#define DEVINL __device__ __forceinline__

static constexpr int MAX_BRANCH = 3;

// ---------------------------------------------------------------------------
// dtype helpers: 16-byte payload = 8 half/bfloat16 or 4 float
// ---------------------------------------------------------------------------
template <typename T> struct Vec;
template <> struct Vec<__nv_bfloat16> { static constexpr int N = 8; };
template <> struct Vec<__half>        { static constexpr int N = 8; };
template <> struct Vec<float>         { static constexpr int N = 4; };

DEVINL float cvt(__nv_bfloat16 v) { return __bfloat162float(v); }
DEVINL float cvt(__half v)        { return __half2float(v); }
DEVINL float cvt(float v)         { return v; }

DEVINL void setv(__nv_bfloat16 &d, float v) { d = __float2bfloat16_rn(v); }
DEVINL void setv(__half &d, float v)        { d = __float2half_rn(v); }
DEVINL void setv(float &d, float v)         { d = v; }

template <typename T>
DEVINL float dot_payload(const uint4 &wv, const uint4 &xv) {
    const T *w = reinterpret_cast<const T *>(&wv);
    const T *x = reinterpret_cast<const T *>(&xv);
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < Vec<T>::N; ++i) s = fmaf(cvt(w[i]), cvt(x[i]), s);
    return s;
}

DEVINL float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// silu(v) = v * sigmoid(v).  Accurate form: the math is free at these sizes.
DEVINL float silu_f(float v) { return v / (1.0f + expf(-v)); }

// ---------------------------------------------------------------------------
// Sinusoidal timestep encoding.
//
// Reproduces the reference expression exactly, in the same fp32 order:
//     e_k   = (neg_log_period * k) / denom          denom = half - shift
//     ang_k = scale * (t * exp(e_k))
//     out   = [cos, sin] when flip_sin_to_cos else [sin, cos]
// ---------------------------------------------------------------------------
struct SinArgs {
    int half;
    float neg_log_period;
    float denom;
    float scale;
    int flip;
};

// Write `2 * half` values (+ zero pad to `n`) starting at dst, cooperatively.
template <typename T>
DEVINL void gen_sinusoid(T *dst, float t, int n, const SinArgs &sa,
                         int tid, int nthreads) {
    for (int k = tid; k < sa.half; k += nthreads) {
        const float ang = sa.scale * (t * expf((sa.neg_log_period * (float)k) / sa.denom));
        float s, c;
        sincosf(ang, &s, &c);
        if (sa.flip) { setv(dst[k], c); setv(dst[sa.half + k], s); }
        else         { setv(dst[k], s); setv(dst[sa.half + k], c); }
    }
    for (int k = 2 * sa.half + tid; k < n; k += nthreads) setv(dst[k], 0.f);
}

// Standalone `Timesteps` / `get_timestep_embedding`: one kernel, fp32 output,
// written straight into the flip_sin_to_cos-permuted layout (no cat, no pad op).
template <typename T, int TPB>
__global__ __launch_bounds__(TPB) void sinusoid_kernel(
        const T *__restrict__ tin, float *__restrict__ out,
        int nrow, int dim, SinArgs sa) {
    const int i = blockIdx.x * TPB + threadIdx.x;
    const int b = i / sa.half;
    if (b >= nrow) return;
    const int k = i - b * sa.half;
    const float t = cvt(tin[b]);
    const float ang = sa.scale * (t * expf((sa.neg_log_period * (float)k) / sa.denom));
    float s, c;
    sincosf(ang, &s, &c);
    float *o = out + (size_t)b * dim;
    if (sa.flip) { o[k] = c; o[sa.half + k] = s; }
    else         { o[k] = s; o[sa.half + k] = c; }
    if (k == 0)
        for (int j = 2 * sa.half; j < dim; ++j) o[j] = 0.f;   // odd embedding_dim
}

// ---------------------------------------------------------------------------
// Grouped batch-1 GEMV core: one warp owns RPW consecutive output rows and
// reduces over K with 16-byte loads; the row operand lives in shared memory so
// every warp in the block shares one trip to L2 for it.
// ---------------------------------------------------------------------------
// `U` payloads per row are loaded before any of them is consumed.  This is the
// whole performance story of these kernels: with one load in flight per row a
// warp's K-reduction is a chain of `K / (32 * NPV)` dependent HBM round trips,
// which measured 19.9 us for the 56.6 MB stage-2 GEMV against a ~8 us bandwidth
// bound.  Bytes in flight per SM are `warps_per_SM * U * RPW * 512`, and
// saturating B200 HBM needs ~5 MB in flight device-wide, so `U` (not the tile
// shape) is the knob that matters -- `warps_per_SM * RPW` is pinned to N/148
// however the rows are split up.
template <typename T, int RPW, int U, bool GUARD>
DEVINL void gemv_accum(const T *__restrict__ W, const T *xs, int K, int lane,
                       int nrow, float *acc) {
    constexpr int NPV = Vec<T>::N;
    constexpr int STRIDE = 32 * NPV;
    const int steps = K / STRIDE;
    int s = 0;
    for (; s + U <= steps; s += U) {
        uint4 wv[U][RPW];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int off = (s + u) * STRIDE + lane * NPV;
#pragma unroll
            for (int r = 0; r < RPW; ++r)
                if (!GUARD || r < nrow)
                    wv[u][r] = *reinterpret_cast<const uint4 *>(W + (size_t)r * K + off);
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const uint4 xv = *reinterpret_cast<const uint4 *>(xs + (s + u) * STRIDE + lane * NPV);
#pragma unroll
            for (int r = 0; r < RPW; ++r)
                if (!GUARD || r < nrow) acc[r] += dot_payload<T>(wv[u][r], xv);
        }
    }
    for (; s < steps; ++s) {
        const int off = s * STRIDE + lane * NPV;
        const uint4 xv = *reinterpret_cast<const uint4 *>(xs + off);
#pragma unroll
        for (int r = 0; r < RPW; ++r)
            if (!GUARD || r < nrow)
                acc[r] += dot_payload<T>(
                    *reinterpret_cast<const uint4 *>(W + (size_t)r * K + off), xv);
    }
}

struct GroupArgs {
    const void *w[MAX_BRANCH];
    const void *b[MAX_BRANCH];
    const void *x[MAX_BRANCH];      // branch input, or nullptr => generate sinusoid
    const void *tsrc[MAX_BRANCH];   // scalar timestep source for sinusoid branches
    int K[MAX_BRANCH];
    void *out;
    int N;
    int nbranch;
    SinArgs sa;
};

// Stage 1: h[branch, n] = silu(dot(W1_b[n, :], x_b) + b1_b[n]).
// grid = (row tiles, nbranch); the branch's input vector is staged in smem,
// generated on the fly for sinusoid branches.
template <typename T, int TPB, int RPW, int U>
__global__ __launch_bounds__(TPB) void stage1_kernel(GroupArgs a) {
    constexpr int WARPS = TPB / 32;
    const int br = blockIdx.y;
    const int K = a.K[br];
    extern __shared__ __align__(16) char sraw[];
    T *xs = reinterpret_cast<T *>(sraw);

    if (a.x[br] != nullptr) {
        const T *g = reinterpret_cast<const T *>(a.x[br]);
        for (int i = threadIdx.x; i < K; i += TPB) xs[i] = g[i];
    } else {
        gen_sinusoid<T>(xs, cvt(*reinterpret_cast<const T *>(a.tsrc[br])), K, a.sa,
                        threadIdx.x, TPB);
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int row0 = ((int)blockIdx.x * WARPS + (int)(threadIdx.x >> 5)) * RPW;
    if (row0 >= a.N) return;
    const int nrow = a.N - row0;
    const T *W = reinterpret_cast<const T *>(a.w[br]) + (size_t)row0 * K;

    float acc[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r) acc[r] = 0.f;
    if (nrow >= RPW) gemv_accum<T, RPW, U, false>(W, xs, K, lane, nrow, acc);
    else             gemv_accum<T, RPW, U, true>(W, xs, K, lane, nrow, acc);

    const T *bias = reinterpret_cast<const T *>(a.b[br]);
    T *h = reinterpret_cast<T *>(a.out) + (size_t)br * a.N;
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const float v = warp_sum(acc[r]);
        if (lane == r && r < nrow)
            setv(h[row0 + r], silu_f(v + (bias ? cvt(bias[row0 + r]) : 0.f)));
    }
}

// Stage 2: out[n] = sum_b (dot(W2_b[n, :], h_b) + b2_b[n]).
// One warp walks every branch for its rows and accumulates in one fp32
// register, so the cross-branch add costs no launch and no extra rounding.
template <typename T, int TPB, int RPW, int U>
__global__ __launch_bounds__(TPB) void stage2_kernel(GroupArgs a) {
    constexpr int WARPS = TPB / 32;
    constexpr int NPV = Vec<T>::N;
    const int K = a.K[0];
    extern __shared__ __align__(16) char sraw[];
    T *hs = reinterpret_cast<T *>(sraw);
    {   // h is our own 16B-aligned stage-1 buffer: copy it in vectorized.
        const uint4 *src = reinterpret_cast<const uint4 *>(a.x[0]);
        uint4 *dst = reinterpret_cast<uint4 *>(hs);
        const int nv = a.nbranch * K / NPV;
        for (int i = threadIdx.x; i < nv; i += TPB) dst[i] = src[i];
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int row0 = ((int)blockIdx.x * WARPS + (int)(threadIdx.x >> 5)) * RPW;
    if (row0 >= a.N) return;
    const int nrow = a.N - row0;

    float acc[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r) acc[r] = 0.f;
    for (int br = 0; br < a.nbranch; ++br) {
        const T *W = reinterpret_cast<const T *>(a.w[br]) + (size_t)row0 * K;
        const T *xb = hs + (size_t)br * K;
        if (nrow >= RPW) gemv_accum<T, RPW, U, false>(W, xb, K, lane, nrow, acc);
        else             gemv_accum<T, RPW, U, true>(W, xb, K, lane, nrow, acc);
    }

    T *out = reinterpret_cast<T *>(a.out);
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const float v = warp_sum(acc[r]);
        if (lane == r && r < nrow) {
            float z = v;
            for (int br = 0; br < a.nbranch; ++br) {
                const T *bias = reinterpret_cast<const T *>(a.b[br]);
                if (bias) z += cvt(bias[row0 + r]);
            }
            setv(out[row0 + r], z);
        }
    }
}

// ---------------------------------------------------------------------------
// Fused single-kernel path: split-K rank update.
//
// The two-stage form above is *serial*: stage 2 computes
// `out[n] = sum_k W2[n,k] h[k]`, so it cannot start until every element of `h`
// exists, and the measured window is `penalty + stage1 + stage2` (composite:
// 6.7 + 15.9 us; standalone TimestepEmbedding: 3.9 + 7.7 us).  Since stage 1 is
// nowhere near the bandwidth ceiling and stage 2 is at it, that sum is pure loss.
//
// Reformulating stage 2 as a sum of rank-1 updates removes the dependency:
//
//     out[:] = sum_k h[k] * W2T[k, :]
//
// Now a block that owns only the k-slice `[k0, k0+KT)` needs only *its* slice of
// `h` -- which it computes itself from a few KB of `W1` -- and can then stream
// `KT` rows of `W2T` straight into an fp32 register accumulator over the whole
// output.  One kernel, and the W1 traffic rides along inside the W2 stream
// instead of in front of it.  The cross-slice sum is closed with
// `red.global.add.v4.f32` into a persistent fp32 scratch plus the standard
// last-block-done pattern, which adds `bias_2` and writes the output dtype.
//
// Three things this hinges on, all measured (see ITERATIONS.md):
//
//   * `W2` must be **transposed** and cached.  A raw column read of the
//     row-major `[N, K]` weight gives every lane a different row, 6 KB apart;
//     `W2T[K, N]` makes a warp's 32 lanes read 512 contiguous bytes per k.
//     The transpose is built lazily, keyed on `data_ptr` + `_version()`, so it
//     lands in warmup and a mutated parameter invalidates it.
//   * The reduction must use **vector** atomics.  `kslices * N` scalar
//     `atomicAdd`s cost 6.9 us at kslices=96 -- more than the fusion saves --
//     against 0.8 us for the same reduction as `red.global.add.v4.f32`.
//   * `kslices` trades reduction cost (proportional to it) against bytes in
//     flight.  When one k-slice does not fill the machine, the *output* is
//     tiled instead (`ntiles`), which adds blocks without adding atomics: the
//     h-slice recompute that costs is an L2 hit, since consecutive blocks share
//     a k-slice by construction.
// ---------------------------------------------------------------------------

// 4-wide fp32 reduction-add.  `red` (not `atom`) because no return value is
// wanted; one instruction per 16 bytes instead of four.
DEVINL void red_add4(float *p, float a0, float a1, float a2, float a3) {
#if __CUDA_ARCH__ >= 900
    asm volatile("red.global.add.v4.f32 [%0], {%1,%2,%3,%4};"
                 :: "l"(p), "f"(a0), "f"(a1), "f"(a2), "f"(a3) : "memory");
#else
    atomicAdd(p + 0, a0); atomicAdd(p + 1, a1);
    atomicAdd(p + 2, a2); atomicAdd(p + 3, a3);
#endif
}

// 4-wide fp32 load that bypasses L1: the partial sums were written by other
// blocks' reductions, which land in L2, so an ordinary load could hit a stale
// line.  Vector form because the finish is one block's serial tail -- scalar
// `volatile` reads of 3072 floats measured 15.4 us on the 3-branch composite.
DEVINL float4 ld_cv4(const float *p) {
    float4 v;
    asm volatile("ld.global.cv.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p) : "memory");
    return v;
}

struct FusedArgs {
    const void *w1[MAX_BRANCH];
    const void *b1[MAX_BRANCH];
    const void *w2t[MAX_BRANCH];   // [K2, N] transposed stage-2 weight
    const void *b2[MAX_BRANCH];
    const void *x[MAX_BRANCH];     // branch input, or nullptr => generate sinusoid
    const void *tsrc[MAX_BRANCH];  // scalar timestep source for sinusoid branches
    int K1[MAX_BRANCH];
    int xoff[MAX_BRANCH];          // smem offset of each branch input, in elements
    float *scratch;                // [N] fp32, zero on entry, left zero on exit
    unsigned int *counter;         // [ntiles] uint, zero on entry, left zero on exit
    void *out;
    int N;
    int nbranch;
    int ntiles;
    int kslices;
    SinArgs sa;
};
// One block owns a k-slice `[k0, k0+KT)` of the hidden dimension and an n-tile of
// `NW` outputs, and its warps are arranged as a `WN x WK` grid: `WN` warp-columns
// each own 256 outputs, and `WK` warp-rows each own `KTW = KT/WK` of the block's
// k-rows.  Every tile bound is compile time, so all accumulators stay in registers.
//
//   WK  warp-rows over k.  This is the lever that separates bandwidth from
//       reduction cost.  A warp sustains ~1.2-1.8 GB/s of weight traffic however
//       many payloads it has outstanding, so the stream needs *warps*:
//       `kslices * N * WK / 256` of them.  The split-K reduction, in contrast,
//       costs `kslices * N` atomics.  With WK = 1 the only way to add warps is to
//       add k-slices, which adds atomics one for one; with WK > 1 the warp-rows
//       reduce their partials in shared memory first, so WK-fold more bandwidth
//       arrives at the same number of global atomics.
//   RH  rows of `h` per warp   (h-phase in-flight = RH * XP loads)
//   U   rows of W2T streamed before any is consumed
//   HB  W1 payload batch in the h-phase
template <typename T, int TPB, int WK, int RH, int U, int HB>
__global__ __launch_bounds__(TPB) void fused_kernel(FusedArgs a) {
    constexpr int NPV = Vec<T>::N;
    constexpr int WARPS = TPB / 32;
    constexpr int WN = WARPS / WK;          // warp-columns over n
    constexpr int KT = RH * WARPS;          // k-rows per block
    constexpr int KTW = KT / WK;            // k-rows per warp-row
    constexpr int NW = WN * 32 * NPV;       // outputs per block
    constexpr int STRIDE = 32 * NPV;
    // cross-warp-row partials live below the T region; WK*NW floats == 32*TPB bytes
    constexpr int RED_BYTES = (WK > 1) ? WK * NW * (int)sizeof(float) : 0;

    const int lane = threadIdx.x & 31;
    const int warp = (int)(threadIdx.x >> 5);
    const int wn = warp % WN;
    const int wk = warp / WN;
    const int ks = (int)blockIdx.x / a.ntiles;         // k-slice
    const int nt = (int)blockIdx.x - ks * a.ntiles;    // n-tile
    const int k0 = ks * KT;
    const int nbase = nt * NW;

    extern __shared__ __align__(16) char sraw[];
    T *sm = reinterpret_cast<T *>(sraw + RED_BYTES);
    T *hs = sm;                                        // [nbranch * KT]

    // --- branch inputs into shared memory (sinusoids generated here) --------
    for (int br = 0; br < a.nbranch; ++br) {
        T *dst = sm + a.xoff[br];
        if (a.x[br] != nullptr) {
            const T *g = reinterpret_cast<const T *>(a.x[br]);
            for (int i = threadIdx.x; i < a.K1[br]; i += TPB) dst[i] = g[i];
        } else {
            gen_sinusoid<T>(dst, cvt(*reinterpret_cast<const T *>(a.tsrc[br])),
                            a.K1[br], a.sa, threadIdx.x, TPB);
        }
    }
    __syncthreads();

    // --- this block's slice of h = silu(W1 x + b1) --------------------------
    // Identical arithmetic and summation order to `stage1_kernel`, so `h` is
    // bit-for-bit what the two-kernel path produces.
    for (int br = 0; br < a.nbranch; ++br) {
        const int K = a.K1[br];
        const int steps = K / STRIDE;
        const int r0 = k0 + warp * RH;
        const T *W = reinterpret_cast<const T *>(a.w1[br]) + (size_t)r0 * K;
        const T *xb = sm + a.xoff[br];
        float acc[RH];
#pragma unroll
        for (int r = 0; r < RH; ++r) acc[r] = 0.f;
        for (int s0 = 0; s0 < steps; s0 += HB) {
            uint4 wv[HB][RH];
#pragma unroll
            for (int p = 0; p < HB; ++p)
                if (s0 + p < steps) {
                    const int off = (s0 + p) * STRIDE + lane * NPV;
#pragma unroll
                    for (int r = 0; r < RH; ++r)
                        wv[p][r] = *reinterpret_cast<const uint4 *>(W + (size_t)r * K + off);
                }
#pragma unroll
            for (int p = 0; p < HB; ++p)
                if (s0 + p < steps) {
                    const uint4 xv = *reinterpret_cast<const uint4 *>(
                        xb + (s0 + p) * STRIDE + lane * NPV);
#pragma unroll
                    for (int r = 0; r < RH; ++r) acc[r] += dot_payload<T>(wv[p][r], xv);
                }
        }
        const T *bias = reinterpret_cast<const T *>(a.b1[br]);
#pragma unroll
        for (int r = 0; r < RH; ++r) {
            const float v = warp_sum(acc[r]);
            if (lane == r)
                setv(hs[br * KT + warp * RH + r],
                     silu_f(v + (bias ? cvt(bias[r0 + r]) : 0.f)));
        }
    }
    __syncthreads();

    // --- rank update: out_partial[:] += sum_{k in this warp-row} h[k]*W2T[k,:] ---
    // bias_2 is seeded into one contributor's accumulator rather than added in the
    // tail: it is 3 x 6 KB of cold HBM whichever block reads it, and here that read
    // overlaps the weight stream instead of serialising behind the whole grid.
    const int nloc = wn * 32 * NPV + lane * NPV;       // output offset within the tile
    float acc[NPV];
#pragma unroll
    for (int i = 0; i < NPV; ++i) acc[i] = 0.f;
    if (ks == 0 && wk == 0)
        for (int br = 0; br < a.nbranch; ++br) {
            const T *b2 = reinterpret_cast<const T *>(a.b2[br]);
            if (b2 == nullptr) continue;
            const uint4 bv = *reinterpret_cast<const uint4 *>(b2 + nbase + nloc);
            const T *bt = reinterpret_cast<const T *>(&bv);
#pragma unroll
            for (int i = 0; i < NPV; ++i) acc[i] += cvt(bt[i]);
        }
    for (int br = 0; br < a.nbranch; ++br) {
        const T *W2 = reinterpret_cast<const T *>(a.w2t[br])
                    + (size_t)(k0 + wk * KTW) * a.N + nbase + nloc;
        const T *hb = hs + br * KT + wk * KTW;
        for (int kb = 0; kb < KTW; kb += U) {
            uint4 wv[U];
#pragma unroll
            for (int u = 0; u < U; ++u)
                wv[u] = *reinterpret_cast<const uint4 *>(W2 + (size_t)(kb + u) * a.N);
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const float hv = cvt(hb[kb + u]);      // smem broadcast
                const T *w = reinterpret_cast<const T *>(&wv[u]);
#pragma unroll
                for (int i = 0; i < NPV; ++i) acc[i] = fmaf(cvt(w[i]), hv, acc[i]);
            }
        }
    }

    // --- close the split-K sum: warp-rows in shared memory, blocks in L2 ----
    if (WK > 1) {
        float *part = reinterpret_cast<float *>(sraw);
        __syncthreads();                                // hs is dead past here
#pragma unroll
        for (int i = 0; i < NPV; ++i) part[wk * NW + nloc + i] = acc[i];
        __syncthreads();
        const int p = (int)threadIdx.x;                 // one payload per thread
        if (p < NW / NPV) {
            float z[NPV];
#pragma unroll
            for (int i = 0; i < NPV; ++i) z[i] = part[p * NPV + i];
            for (int w = 1; w < WK; ++w)
#pragma unroll
                for (int i = 0; i < NPV; ++i) z[i] += part[w * NW + p * NPV + i];
            float *dst = a.scratch + nbase + p * NPV;
#pragma unroll
            for (int i = 0; i < NPV; i += 4)
                red_add4(dst + i, z[i + 0], z[i + 1], z[i + 2], z[i + 3]);
        }
    } else {
        float *dst = a.scratch + nbase + nloc;
#pragma unroll
        for (int i = 0; i < NPV; i += 4)
            red_add4(dst + i, acc[i + 0], acc[i + 1], acc[i + 2], acc[i + 3]);
    }

    // One ticket per n-tile rather than one for the whole grid: the blocks that
    // share a tile are exactly the `kslices` contributors to it, so the finish
    // runs on `ntiles` different blocks in parallel instead of serially on one.
    // The last one writes the output dtype and leaves both scratch and counter
    // zeroed for the next call, which is why no memset launch is ever needed.
    __threadfence();
    __syncthreads();
    __shared__ int s_last;
    if (threadIdx.x == 0)
        s_last = (atomicAdd(a.counter + nt, 1u) == (unsigned)(a.kslices - 1)) ? 1 : 0;
    __syncthreads();
    if (!s_last) return;

    // One vectorized pass: each thread owns NPV consecutive outputs, so the tail is
    // a handful of 16-byte accesses per thread instead of 3072 scalar ones.
    T *out = reinterpret_cast<T *>(a.out);
    for (int v = threadIdx.x; v < NW / NPV; v += TPB) {
        const int n = nbase + v * NPV;
        float z[NPV];
#pragma unroll
        for (int i = 0; i < NPV; i += 4) {
            const float4 q = ld_cv4(a.scratch + n + i);
            z[i + 0] = q.x; z[i + 1] = q.y; z[i + 2] = q.z; z[i + 3] = q.w;
            *reinterpret_cast<float4 *>(a.scratch + n + i) = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        uint4 ov;
        T *od = reinterpret_cast<T *>(&ov);
#pragma unroll
        for (int i = 0; i < NPV; ++i) setv(od[i], z[i]);
        *reinterpret_cast<uint4 *>(out + n) = ov;
    }
    if (threadIdx.x == 0) a.counter[nt] = 0u;
}

// ---------------------------------------------------------------------------
// Host side.  Tile shapes are compile-time; `FK_TSE_CFG=s1tpb,s1rpw,s1u,s2tpb,s2rpw,s2u`
// selects among the instantiated two-kernel set and
// `FK_TSE_FCFG=tpb,wk,rh,u,hb` among the fused set, so a sweep needs one build.
// `FK_TSE_PATH` picks the path: 0/unset = fused when eligible then two-kernel,
// 1 = two-kernel only, 2 = fused only (probes assert on the fallback).
// ---------------------------------------------------------------------------
#include <array>
#include <map>
#include <mutex>
#include <unordered_map>

struct Cfg { int s1_tpb, s1_rpw, s1_u, s2_tpb, s2_rpw, s2_u; };

static Cfg parse_cfg() {
    Cfg c{256, 2, 3, 128, 2, 12};
    const char *e = std::getenv("FK_TSE_CFG");
    if (e) std::sscanf(e, "%d,%d,%d,%d,%d,%d", &c.s1_tpb, &c.s1_rpw, &c.s1_u,
                       &c.s2_tpb, &c.s2_rpw, &c.s2_u);
    return c;
}
static const Cfg &cfg() { static Cfg c = parse_cfg(); return c; }

static int cfg_key(int tpb, int rpw, int u) { return (tpb * 100 + rpw) * 100 + u; }

#define S1_GRID(TPB, RPW) \
    dim3((a.N + (TPB / 32) * (RPW) - 1) / ((TPB / 32) * (RPW)), a.nbranch)
#define S2_GRID(TPB, RPW) \
    dim3((a.N + (TPB / 32) * (RPW) - 1) / ((TPB / 32) * (RPW)))

#define S1_LAUNCH(TPB, RPW, U) \
    stage1_kernel<T, TPB, RPW, U><<<S1_GRID(TPB, RPW), TPB, smem, stream>>>(a)
#define S2_LAUNCH(TPB, RPW, U) \
    stage2_kernel<T, TPB, RPW, U><<<S2_GRID(TPB, RPW), TPB, smem, stream>>>(a)

// Tile set kept small on purpose: these are the shapes that survived the sweep in
// ITERATIONS.md.  `FK_TSE_CFG=s1tpb,s1rpw,s1u,s2tpb,s2rpw,s2u` selects one.
#define S1_SET(X) X(256,2,3) X(128,1,4) X(128,2,3) X(64,4,2)
#define S2_SET(X) X(128,2,12) X(128,3,12) X(128,1,12) X(64,2,12)

template <typename T>
static void launch_stage1(const GroupArgs &a, size_t smem, cudaStream_t stream) {
    const int key = cfg_key(cfg().s1_tpb, cfg().s1_rpw, cfg().s1_u);
#define S1_CASE(TPB, RPW, U) \
    if (key == cfg_key(TPB, RPW, U)) { S1_LAUNCH(TPB, RPW, U); return; }
    S1_SET(S1_CASE)
#undef S1_CASE
    S1_LAUNCH(256, 2, 3);
}

template <typename T>
static void launch_stage2(const GroupArgs &a, size_t smem, cudaStream_t stream) {
    const int key = cfg_key(cfg().s2_tpb, cfg().s2_rpw, cfg().s2_u);
#define S2_CASE(TPB, RPW, U) \
    if (key == cfg_key(TPB, RPW, U)) { S2_LAUNCH(TPB, RPW, U); return; }
    S2_SET(S2_CASE)
#undef S2_CASE
    S2_LAUNCH(128, 2, 12);
}

// A vector length is kernel-eligible when a warp's 16-byte lanes tile it exactly.
static bool k_ok(int64_t k, at::ScalarType st) {
    const int npv = (st == at::kFloat) ? 4 : 8;
    return k > 0 && k % (32 * npv) == 0;
}

static bool aligned16(const void *p) {
    return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

static bool weight_ok(const at::Tensor &w, int64_t n, int64_t k, at::ScalarType st) {
    return w.defined() && w.is_cuda() && w.scalar_type() == st && w.dim() == 2 &&
           w.size(0) == n && w.size(1) == k && w.is_contiguous() &&
           aligned16(w.const_data_ptr());
}

static bool bias_ok(const at::Tensor &b, int64_t n, at::ScalarType st) {
    return !b.defined() ||
           (b.is_cuda() && b.scalar_type() == st && b.dim() == 1 && b.size(0) == n &&
            b.is_contiguous());
}

// One branch of the two-stage MLP: either a plain input vector, or a sinusoidal
// encoding generated inside the kernel from a scalar timestep.
struct Branch {
    at::Tensor x;      // undefined => sinusoid from `t`
    at::Tensor t;
    at::Tensor w1, b1, w2, b2;
    int64_t k1 = 0;
};

// Shared eligibility gate for both paths: returns N, or -1 when anything about
// the group is outside what the kernels cover and the reference must run.
static int64_t validate_group(Branch *br, int nb, const SinArgs &sa,
                             at::ScalarType st) {
    if (nb < 1 || nb > MAX_BRANCH) return -1;
    if (st != at::kBFloat16 && st != at::kHalf && st != at::kFloat) return -1;
    const int64_t N = br[0].w1.defined() ? br[0].w1.size(0) : 0;
    if (N <= 0 || !k_ok(N, st)) return -1;

    for (int i = 0; i < nb; ++i) {
        Branch &b = br[i];
        if (!k_ok(b.k1, st)) return -1;
        if (!weight_ok(b.w1, N, b.k1, st) || !weight_ok(b.w2, N, N, st)) return -1;
        if (!bias_ok(b.b1, N, st) || !bias_ok(b.b2, N, st)) return -1;
        if (b.x.defined()) {
            if (!b.x.is_cuda() || b.x.scalar_type() != st || b.x.numel() != b.k1 ||
                !b.x.is_contiguous() || !aligned16(b.x.const_data_ptr()))
                return -1;
        } else {
            // sinusoid branch: the reference emits exactly 2*half values here
            if (b.k1 != 2 * (int64_t)sa.half) return -1;
            if (!b.t.is_cuda() || b.t.scalar_type() != st || b.t.numel() != 1) return -1;
        }
    }
    return N;
}

// Two-kernel path (round 1): stage 1 for every branch, then stage 2 for every
// branch.  Kept intact as the fused path's fallback and its measurement control.
static at::Tensor run_group(Branch *br, int nb, const SinArgs &sa,
                            at::ScalarType st, const at::Device &dev) {
    const int64_t N = validate_group(br, nb, sa, st);
    if (N < 0) return at::Tensor();
    // stage 2 stages every branch's stage-1 vector in shared memory
    if ((size_t)nb * N * c10::elementSize(st) > 48u * 1024u) return at::Tensor();

    const auto opts = at::TensorOptions().dtype(st).device(dev);
    at::Tensor h = at::empty({nb, N}, opts);
    at::Tensor out = at::empty({1, N}, opts);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const size_t esz = c10::elementSize(st);

    GroupArgs a{};
    a.nbranch = nb;
    a.N = (int)N;
    a.sa = sa;
    a.out = h.data_ptr();
    int64_t kmax = 0;
    for (int i = 0; i < nb; ++i) {
        a.w[i] = br[i].w1.const_data_ptr();
        a.b[i] = br[i].b1.defined() ? br[i].b1.const_data_ptr() : nullptr;
        a.x[i] = br[i].x.defined() ? br[i].x.const_data_ptr() : nullptr;
        a.tsrc[i] = br[i].x.defined() ? nullptr : br[i].t.const_data_ptr();
        a.K[i] = (int)br[i].k1;
        kmax = std::max(kmax, br[i].k1);
    }
    switch (st) {
    case at::kBFloat16: launch_stage1<__nv_bfloat16>(a, kmax * esz, stream); break;
    case at::kHalf:     launch_stage1<__half>(a, kmax * esz, stream); break;
    default:            launch_stage1<float>(a, kmax * esz, stream); break;
    }

    for (int i = 0; i < nb; ++i) {
        a.w[i] = br[i].w2.const_data_ptr();
        a.b[i] = br[i].b2.defined() ? br[i].b2.const_data_ptr() : nullptr;
        a.x[i] = nullptr;
        a.K[i] = (int)N;
    }
    a.x[0] = h.const_data_ptr();
    a.out = out.data_ptr();
    switch (st) {
    case at::kBFloat16: launch_stage2<__nv_bfloat16>(a, nb * N * esz, stream); break;
    case at::kHalf:     launch_stage2<__half>(a, nb * N * esz, stream); break;
    default:            launch_stage2<float>(a, nb * N * esz, stream); break;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Fused path: host bookkeeping.
// ---------------------------------------------------------------------------

// Cached transpose of a stage-2 weight, keyed on `data_ptr` and invalidated by
// the tensor's version counter, so an in-place parameter update is picked up and
// a reassigned parameter lands on a different key.  Built on the first call,
// i.e. inside the benchmark's warmup, never in a timed window.
struct W2TEntry { at::Tensor t; uint64_t ver; int64_t n, k; };
static std::unordered_map<const void *, W2TEntry> g_w2t;
static std::mutex g_w2t_mu;

static at::Tensor w2_transposed(const at::Tensor &w2) {
    const void *key = w2.const_data_ptr();
    const uint64_t ver = (uint64_t)w2._version();
    std::lock_guard<std::mutex> lk(g_w2t_mu);
    auto it = g_w2t.find(key);
    if (it != g_w2t.end()) {
        const W2TEntry &e = it->second;
        if (e.ver == ver && e.n == w2.size(0) && e.k == w2.size(1) &&
            e.t.scalar_type() == w2.scalar_type() && e.t.device() == w2.device())
            return e.t;
        g_w2t.erase(it);
    }
    if (g_w2t.size() >= 64) g_w2t.clear();     // bound the footprint
    at::Tensor t = w2.t().contiguous();
    if (!t.is_contiguous() || !aligned16(t.const_data_ptr())) return at::Tensor();
    g_w2t.emplace(key, W2TEntry{t, ver, w2.size(0), w2.size(1)});
    return t;
}

// The split-K accumulator.  Persistent and self-resetting: the last block reads
// it, writes the output, and stores zeros back, so no memset launch is needed on
// any call.  Keyed by stream as well as by size, so two streams never share one.
struct ScratchEntry { at::Tensor buf, cnt; };
static std::map<std::array<uint64_t, 3>, ScratchEntry> g_scratch;
static std::mutex g_scratch_mu;

static ScratchEntry *scratch_for(int64_t N, const at::Device &dev,
                                 cudaStream_t stream) {
    const std::array<uint64_t, 3> key{(uint64_t)dev.index(), (uint64_t)N,
                                      (uint64_t)(uintptr_t)stream};
    std::lock_guard<std::mutex> lk(g_scratch_mu);
    auto it = g_scratch.find(key);
    if (it != g_scratch.end()) return &it->second;
    if (g_scratch.size() >= 32) return nullptr;
    ScratchEntry e;
    e.buf = at::zeros({N}, at::TensorOptions().dtype(at::kFloat).device(dev));
    e.cnt = at::zeros({64}, at::TensorOptions().dtype(at::kInt).device(dev));
    auto res = g_scratch.emplace(key, std::move(e));
    return &res.first->second;
}

struct FCfg { int tpb, wk, rh, u, hb; };

static FCfg parse_fcfg() {
    FCfg c{384, 1, 2, 24, 3};
    const char *e = std::getenv("FK_TSE_FCFG");
    if (e) std::sscanf(e, "%d,%d,%d,%d,%d", &c.tpb, &c.wk, &c.rh, &c.u, &c.hb);
    return c;
}
static const FCfg &fcfg() { static FCfg c = parse_fcfg(); return c; }

static int fpath() {
    static int p = []() {
        const char *e = std::getenv("FK_TSE_PATH");
        return e ? std::atoi(e) : 0;
    }();
    return p;
}

// Which path served the last call: 0 = none/reference, 1 = two-kernel, 2 = fused.
// Exposed so a probe can assert the fast path actually ran -- a silent fallback
// passes every correctness check while reporting the timings of another kernel.
static int g_path_used = 0;

// Derived geometry of a fused config.  `ntiles > 1` splits the output so more
// blocks are resident without more atomics; the h-slice recompute it costs is an
// L2 hit because consecutive blocks share a k-slice.
struct FGeom { int kt, ktw, nw, kslices, ntiles, nblocks, red_bytes; bool ok; };

static FGeom fgeom(const FCfg &c, int64_t N, int npv) {
    FGeom g{};
    g.ok = false;
    if (c.tpb < 32 || c.tpb > 1024 || (c.tpb & 31)) return g;
    if (c.rh < 1 || c.rh > 32 || c.u < 1 || c.hb < 1 || c.wk < 1) return g;
    const int warps = c.tpb / 32;
    if (warps % c.wk) return g;
    const int wn = warps / c.wk;
    g.kt = c.rh * warps;
    g.ktw = g.kt / c.wk;
    g.nw = wn * 32 * npv;
    if (g.ktw < 1 || g.ktw % c.u) return g;
    if (N % g.kt || N % g.nw) return g;
    g.kslices = (int)(N / g.kt);
    g.ntiles = (int)(N / g.nw);
    g.nblocks = g.kslices * g.ntiles;
    g.red_bytes = (c.wk > 1) ? c.wk * g.nw * (int)sizeof(float) : 0;
    if (g.nblocks < 1 || g.nblocks > 65535 || g.ntiles > 64) return g;
    g.ok = true;
    return g;
}

// Fused instantiation set, (tpb, wk, rh, u, hb).  KT = rh*tpb/32 rows of h per
// block, WN = tpb/32/wk warp-columns, NW = WN*32*NPV outputs, KTW = KT/wk rows per
// warp-row; so kslices = N/KT and ntiles = N/NW.
//
// The two quantities that matter pull against each other: the stream wants warps
// (`kslices * N * wk / 256` of them, since a warp sustains ~1.2-1.8 GB/s however
// many payloads it has in flight) and the reduction costs `kslices * N` atomics.
// `wk > 1` buys warps without buying atomics, at the price of a shared-memory
// reduction across warp-rows -- and **measured, it does not pay**: the extra
// barriers and smem round trip cost as much as the warps gain (wk=2 ties wk=1 at
// 33.9 us on the composite, wk=4 loses 4 us, wk=8 loses 14).  Two wk > 1 shapes
// are kept instantiated so the next round can re-probe cheaply, but the default
// is wk = 1.  See ITERATIONS.md.
#define F_SET(X) \
    X(384,1,2,24,3) X(384,1,2,8,3)  X(384,1,1,12,3) X(384,1,4,24,3) \
    X(192,1,4,24,3) X(128,1,6,24,3) X(128,1,8,8,3)  X(64,1,12,12,3) \
    X(384,2,2,12,3) X(512,4,6,12,3)

// Tried in order when the requested config does not tile the shape at hand; all
// are in F_SET, so a shape the first one cannot cover still gets a fused kernel
// rather than falling back to two.
static const FCfg F_CANDIDATES[] = {
    {384, 1, 2, 24, 3}, {384, 1, 1, 12, 3}, {128, 1, 6, 24, 3},
    {128, 1, 8, 8, 3},  {64, 1, 12, 12, 3},
};

template <typename T>
static bool launch_fused(const FCfg &c, const FusedArgs &a, int nblocks, size_t smem,
                         cudaStream_t stream) {
#define F_CASE(TPB, WK, RH, U, HB)                                             \
    if (c.tpb == TPB && c.wk == WK && c.rh == RH && c.u == U && c.hb == HB) {   \
        fused_kernel<T, TPB, WK, RH, U, HB><<<nblocks, TPB, smem, stream>>>(a); \
        return true;                                                           \
    }
    F_SET(F_CASE)
#undef F_CASE
    return false;
}

static at::Tensor run_fused(Branch *br, int nb, const SinArgs &sa,
                            at::ScalarType st, const at::Device &dev) {
    const int64_t N = validate_group(br, nb, sa, st);
    if (N < 0) return at::Tensor();
    const int npv = (st == at::kFloat) ? 4 : 8;
    FCfg c = fcfg();
    FGeom g = fgeom(c, N, npv);
    for (size_t i = 0; !g.ok && i < sizeof(F_CANDIDATES) / sizeof(FCfg); ++i) {
        c = F_CANDIDATES[i];
        g = fgeom(c, N, npv);
    }
    if (!g.ok) return at::Tensor();

    // shared memory: the h slices of every branch, then one copy of each input
    const size_t esz = c10::elementSize(st);
    const int align = (int)(16 / esz);
    int off = ((nb * g.kt + align - 1) / align) * align;
    int xoff[MAX_BRANCH] = {0, 0, 0};
    for (int i = 0; i < nb; ++i) {
        xoff[i] = off;
        off += (int)(((br[i].k1 + align - 1) / align) * align);
    }
    const size_t smem = (size_t)g.red_bytes + (size_t)off * esz;
    if (smem > 48u * 1024u) return at::Tensor();

    // both the accumulator seed and the finish read bias_2 16 bytes at a time
    for (int i = 0; i < nb; ++i)
        if (br[i].b2.defined() && !aligned16(br[i].b2.const_data_ptr()))
            return at::Tensor();

    at::Tensor w2t[MAX_BRANCH];
    for (int i = 0; i < nb; ++i) {
        w2t[i] = w2_transposed(br[i].w2);
        if (!w2t[i].defined()) return at::Tensor();
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    ScratchEntry *sc = scratch_for(N, dev, stream);
    if (sc == nullptr) return at::Tensor();

    at::Tensor out = at::empty({1, N}, at::TensorOptions().dtype(st).device(dev));

    FusedArgs a{};
    a.nbranch = nb;
    a.N = (int)N;
    a.ntiles = g.ntiles;
    a.kslices = g.kslices;
    a.sa = sa;
    a.out = out.data_ptr();
    a.scratch = sc->buf.data_ptr<float>();
    a.counter = reinterpret_cast<unsigned int *>(sc->cnt.data_ptr<int>());
    for (int i = 0; i < nb; ++i) {
        a.w1[i] = br[i].w1.const_data_ptr();
        a.b1[i] = br[i].b1.defined() ? br[i].b1.const_data_ptr() : nullptr;
        a.w2t[i] = w2t[i].const_data_ptr();
        a.b2[i] = br[i].b2.defined() ? br[i].b2.const_data_ptr() : nullptr;
        a.x[i] = br[i].x.defined() ? br[i].x.const_data_ptr() : nullptr;
        a.tsrc[i] = br[i].x.defined() ? nullptr : br[i].t.const_data_ptr();
        a.K1[i] = (int)br[i].k1;
        a.xoff[i] = xoff[i];
    }
    bool launched = false;
    switch (st) {
    case at::kBFloat16: launched = launch_fused<__nv_bfloat16>(c, a, g.nblocks, smem, stream); break;
    case at::kHalf:     launched = launch_fused<__half>(c, a, g.nblocks, smem, stream); break;
    default:            launched = launch_fused<float>(c, a, g.nblocks, smem, stream); break;
    }
    if (!launched) return at::Tensor();
    return out;
}

// Path dispatch.  The fused kernel is one device op instead of two serialized
// ones; the two-kernel path stays as the fallback for every group it cannot
// tile, and `FK_TSE_PATH` forces either for measurement.
static at::Tensor run_mlp(Branch *br, int nb, const SinArgs &sa,
                          at::ScalarType st, const at::Device &dev) {
    if (fpath() != 1) {
        at::Tensor r = run_fused(br, nb, sa, st, dev);
        if (r.defined()) { g_path_used = 2; return r; }
    }
    if (fpath() == 2) { g_path_used = 0; return at::Tensor(); }
    at::Tensor r = run_group(br, nb, sa, st, dev);
    g_path_used = r.defined() ? 1 : 0;
    return r;
}

int64_t fk_path_used() { return g_path_used; }

static SinArgs make_sin(int64_t dim, bool flip, double shift, double scale,
                        double max_period) {
    SinArgs sa{};
    sa.half = (int)(dim / 2);
    sa.neg_log_period = (float)(-std::log(max_period));
    sa.denom = (float)((double)(dim / 2) - shift);
    sa.scale = (float)scale;
    sa.flip = flip ? 1 : 0;
    return sa;
}

// --- exported entry points -------------------------------------------------

// `Timesteps` / `get_timestep_embedding`, fp32 output like the reference.
at::Tensor fk_sinusoid(const at::Tensor &t, int64_t dim, bool flip, double shift,
                       double scale, double max_period) {
    const auto st = t.scalar_type();
    if (!t.is_cuda() || t.dim() != 1 || !t.is_contiguous() || dim < 2 ||
        (st != at::kBFloat16 && st != at::kHalf && st != at::kFloat))
        return at::Tensor();
    const SinArgs sa = make_sin(dim, flip, shift, scale, max_period);
    if (sa.half < 1 || sa.denom == 0.f) return at::Tensor();
    const int64_t nrow = t.size(0);
    at::Tensor out = at::empty({nrow, dim},
                               at::TensorOptions().dtype(at::kFloat).device(t.device()));
    if (nrow == 0) return out;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int TPB = 128;
    const int64_t total = nrow * sa.half;
    const dim3 grid((unsigned)((total + TPB - 1) / TPB));
    float *o = out.data_ptr<float>();
    switch (st) {
    case at::kBFloat16:
        sinusoid_kernel<__nv_bfloat16, TPB><<<grid, TPB, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16 *>(t.const_data_ptr()), o,
            (int)nrow, (int)dim, sa);
        break;
    case at::kHalf:
        sinusoid_kernel<__half, TPB><<<grid, TPB, 0, stream>>>(
            reinterpret_cast<const __half *>(t.const_data_ptr()), o,
            (int)nrow, (int)dim, sa);
        break;
    default:
        sinusoid_kernel<float, TPB><<<grid, TPB, 0, stream>>>(
            t.const_data_ptr<float>(), o, (int)nrow, (int)dim, sa);
        break;
    }
    return out;
}

// Standalone `TimestepEmbedding`: one kernel for the whole MLP.
at::Tensor fk_mlp(const at::Tensor &x, const at::Tensor &w1, const at::Tensor &b1,
                  const at::Tensor &w2, const at::Tensor &b2) {
    if (!x.is_cuda() || x.dim() != 2 || x.size(0) != 1) return at::Tensor();
    Branch br[1];
    br[0].x = x;
    br[0].w1 = w1; br[0].b1 = b1; br[0].w2 = w2; br[0].b2 = b2;
    br[0].k1 = x.size(1);
    return run_mlp(br, 1, SinArgs{}, x.scalar_type(), x.device());
}

// `CombinedTimestepTextProjEmbeddings` (2 branches) and
// `CombinedTimestepGuidanceTextProjEmbeddings` (3 branches): one kernel for the
// whole module, sinusoids included.  Two entry points rather than optional args
// so every bound tensor is defined (Python falls back when a bias is absent).
static at::Tensor combined_impl(const at::Tensor &timestep, const at::Tensor *guidance,
                                const at::Tensor &pooled, const at::Tensor *w,
                                int64_t proj_dim, bool flip, double shift, double scale,
                                double max_period) {
    if (!pooled.is_cuda() || pooled.dim() != 2 || pooled.size(0) != 1) return at::Tensor();
    const auto st = pooled.scalar_type();
    if (!timestep.is_cuda() || timestep.dim() != 1 || timestep.numel() != 1 ||
        timestep.scalar_type() != st)
        return at::Tensor();
    if (guidance && (!guidance->is_cuda() || guidance->dim() != 1 ||
                     guidance->numel() != 1 || guidance->scalar_type() != st))
        return at::Tensor();
    if (proj_dim < 2 || (proj_dim & 1)) return at::Tensor();

    // `w` is a flat [branch][w1, b1, w2, b2] table in branch order.
    Branch br[MAX_BRANCH];
    int nb = 0;
    br[nb].t = timestep;
    br[nb].k1 = proj_dim;
    ++nb;
    if (guidance) {
        br[nb].t = *guidance;
        br[nb].k1 = proj_dim;
        ++nb;
    }
    br[nb].x = pooled;
    br[nb].k1 = pooled.size(1);
    ++nb;
    for (int i = 0; i < nb; ++i) {
        br[i].w1 = w[4 * i + 0];
        br[i].b1 = w[4 * i + 1];
        br[i].w2 = w[4 * i + 2];
        br[i].b2 = w[4 * i + 3];
    }
    return run_mlp(br, nb, make_sin(proj_dim, flip, shift, scale, max_period), st,
                   pooled.device());
}

at::Tensor fk_combined3(const at::Tensor &timestep, const at::Tensor &guidance,
                        const at::Tensor &pooled,
                        const at::Tensor &tw1, const at::Tensor &tb1,
                        const at::Tensor &tw2, const at::Tensor &tb2,
                        const at::Tensor &gw1, const at::Tensor &gb1,
                        const at::Tensor &gw2, const at::Tensor &gb2,
                        const at::Tensor &xw1, const at::Tensor &xb1,
                        const at::Tensor &xw2, const at::Tensor &xb2,
                        int64_t proj_dim, bool flip, double shift, double scale,
                        double max_period) {
    const at::Tensor w[12] = {tw1, tb1, tw2, tb2, gw1, gb1, gw2, gb2,
                              xw1, xb1, xw2, xb2};
    return combined_impl(timestep, &guidance, pooled, w, proj_dim, flip, shift, scale,
                         max_period);
}

at::Tensor fk_combined2(const at::Tensor &timestep, const at::Tensor &pooled,
                        const at::Tensor &tw1, const at::Tensor &tb1,
                        const at::Tensor &tw2, const at::Tensor &tb2,
                        const at::Tensor &xw1, const at::Tensor &xb1,
                        const at::Tensor &xw2, const at::Tensor &xb2,
                        int64_t proj_dim, bool flip, double shift, double scale,
                        double max_period) {
    const at::Tensor w[8] = {tw1, tb1, tw2, tb2, xw1, xb1, xw2, xb2};
    return combined_impl(timestep, nullptr, pooled, w, proj_dim, flip, shift, scale,
                         max_period);
}
