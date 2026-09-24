// Routed MXFP4 MoE experts for GPT-OSS.
//
// The trtllm-gen fused MoE streams all 128 expert weight matrices on every call
// -- ~1.9 GB, a flat ~450 us on this GPU -- no matter how many tokens the step
// carries.  At decode widths the router only touches 4..112 of them, so the win
// is not a faster GEMM but a *routed* one: read the experts the step actually
// selected and nothing else.
//
// With 1-4 tokens per expert there is no MMA worth issuing (tcgen05 wants M=128
// and would pad the token dimension ~30x), so the dot is a warp reduction over a
// weight row: `cvt.rn.f16x2.e2m1x2` turns the packed nibbles into half2 one
// instruction per pair -- the E8M0 block scale factors out of the 32-value block
// and is applied once to its partial sum -- and `__hfma2` folds them against the
// activations staged in shared memory.  The four independent half2 accumulators
// matter: at these tile sizes occupancy alone cannot cover the hfma2 latency.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define LANES 32
#define NWARP 4              // warps per expert-GEMM CTA
#define NTHREAD (LANES * NWARP)
#define RMAX 8               // max weight rows per warp (ROWS / NWARP)
#define RWARP 8              // warps in the (single-CTA) router
#define RNTHREAD (LANES * RWARP)
#define TMAX 4               // token slots per expert tile
#define MMAX 512             // max tokens the single-CTA router handles

// 8 packed nibbles -> 4 half2, one hardware conversion per pair.
__device__ __forceinline__ void cvt8(unsigned int w, __half2 *o) {
    unsigned int a, b, c, d;
    asm("{ .reg .b8 p0, p1, p2, p3;            \n"
        "  mov.b32 {p0, p1, p2, p3}, %4;       \n"
        "  cvt.rn.f16x2.e2m1x2 %0, p0;         \n"
        "  cvt.rn.f16x2.e2m1x2 %1, p1;         \n"
        "  cvt.rn.f16x2.e2m1x2 %2, p2;         \n"
        "  cvt.rn.f16x2.e2m1x2 %3, p3;       } \n"
        : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(w));
    o[0] = *reinterpret_cast<__half2 *>(&a);
    o[1] = *reinterpret_cast<__half2 *>(&b);
    o[2] = *reinterpret_cast<__half2 *>(&c);
    o[3] = *reinterpret_cast<__half2 *>(&d);
}

// E8M0 block scale -> float: the byte *is* the biased fp32 exponent.
__device__ __forceinline__ float e8m0(unsigned int s) {
    return __int_as_float(s << 23);
}

template <int NH2>
__device__ __forceinline__ float dot_h2(const __half2 *hv, const __half2 *xv) {
    __half2 a0 = __hmul2(hv[0], xv[0]);
    __half2 a1 = __hmul2(hv[1], xv[1]);
    __half2 a2 = __hmul2(hv[2], xv[2]);
    __half2 a3 = __hmul2(hv[3], xv[3]);
#pragma unroll
    for (int q = 4; q < NH2; q += 4) {
        a0 = __hfma2(hv[q], xv[q], a0);
        a1 = __hfma2(hv[q + 1], xv[q + 1], a1);
        a2 = __hfma2(hv[q + 2], xv[q + 2], a2);
        a3 = __hfma2(hv[q + 3], xv[q + 3], a3);
    }
    const __half2 s = __hadd2(__hadd2(a0, a1), __hadd2(a2, a3));
    return __half2float(__hadd(s.x, s.y));
}

// acc[r][t] += sum_k dequant(W[row0+r][k]) * xs[t][k], one warp, lanes strided
// over VPL-value chunks (VPL divides the 32-value MX block or is a multiple).
template <int NT, int VPL, int RPW>
__device__ __forceinline__ void row_dots(
    const uint8_t *__restrict__ wrow, const uint8_t *__restrict__ srow,
    long swr, long ssr, const __half *__restrict__ xs, int K, int lane,
    int ntok, float acc[RMAX][NT]) {
    constexpr int WORDS = VPL / 8;
    constexpr int NH2 = VPL / 2;
    constexpr int SHR = 32 / VPL;
    const int nchunk = K / VPL;
    const int steps = (nchunk + LANES - 1) / LANES;
#pragma unroll 1
    for (int s = 0; s < steps; ++s) {
        // The tail step leaves some lanes with no chunk.  They stay in the loop
        // (its trip count has to be warp-uniform for the shuffle reduction
        // below) and re-read chunk 0 with a zero scale, contributing nothing.
        const int hbr = s * LANES + lane;
        const bool ok = hbr < nchunk;
        const int hb = ok ? hbr : 0;
        __half2 xv[NT][NH2];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (t >= ntok) break;
            const uint4 *xp =
                reinterpret_cast<const uint4 *>(xs + t * K + hb * VPL);
#pragma unroll
            for (int q = 0; q < WORDS; ++q)
                *reinterpret_cast<uint4 *>(&xv[t][q * 4]) = xp[q];
        }
        unsigned int raw[RPW][WORDS];
        float sc[RPW];
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            const unsigned int *wp = reinterpret_cast<const unsigned int *>(
                wrow + i * swr + hb * (VPL / 2));
#pragma unroll
            for (int q = 0; q < WORDS; ++q) raw[i][q] = wp[q];
            sc[i] = ok ? e8m0(srow[i * ssr + hb / SHR]) : 0.f;
        }
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            __half2 hv[NH2];
#pragma unroll
            for (int q = 0; q < WORDS; ++q) cvt8(raw[i][q], hv + 4 * q);
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                if (t >= ntok) break;
                acc[i][t] = fmaf(sc[i], dot_h2<NH2>(hv, xv[t]), acc[i][t]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < RPW; ++i)
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (t >= ntok) break;
#pragma unroll
            for (int o = 16; o; o >>= 1)
                acc[i][t] += __shfl_xor_sync(0xffffffffu, acc[i][t], o);
        }
}

__device__ __forceinline__ int load_tile(const int *__restrict__ SORTED,
                                         const int *__restrict__ TILE_NV,
                                         int tile, int *ids) {
    const int n = TILE_NV[tile];
#pragma unroll
    for (int t = 0; t < TMAX; ++t) ids[t] = SORTED[tile * TMAX + (t < n ? t : 0)];
    return n;
}

// ---------------------------------------------------------------------------
// Routing: top-4 over the router logits, softmax over those four, then group the
// (token, expert) pairs by expert into TMAX-wide tiles.  One CTA: the whole
// thing is O(M * E) with M <= a few hundred, and a second launch would cost more
// than the work.
//
// The top-k ties break toward the *lower* expert index, which is what
// trtllm-gen's `TopKRedType` does (it packs 65535-idx into the low bits of the
// reduction key).  bf16 logits collide often enough at 128 experts that getting
// this wrong would change the selected expert on a few percent of tokens.
// ---------------------------------------------------------------------------
template <int EPL>
__global__ __launch_bounds__(RNTHREAD) void route_align(
    const __nv_bfloat16 *__restrict__ LOGITS, int ld, int M, int E,
    int *__restrict__ IDX, float *__restrict__ WGT, int *__restrict__ SORTED,
    int *__restrict__ TILE_E, int *__restrict__ TILE_NV,
    int *__restrict__ TOTAL) {
    __shared__ int cnt[EPL * LANES];
    __shared__ int tst[EPL * LANES];
    __shared__ int scan[EPL * LANES];
    __shared__ int pos_[4 * MMAX];
    __shared__ int eid_[4 * MMAX];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    for (int i = tid; i < E; i += RNTHREAD) cnt[i] = 0;
    __syncthreads();

    for (int t = warp; t < M; t += RWARP) {
        float v[EPL];
#pragma unroll
        for (int q = 0; q < EPL; ++q) {
            const int c = q * LANES + lane;
            v[q] = c < E ? __bfloat162float(LOGITS[(long)t * ld + c])
                         : -INFINITY;
        }
        float mv[4];
        int mi[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            float best = -INFINITY;
            int bi = E;
#pragma unroll
            for (int q = 0; q < EPL; ++q)
                if (v[q] > best) { best = v[q]; bi = q * LANES + lane; }
#pragma unroll
            for (int o = 16; o; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, best, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
            }
            mv[r] = best;
            mi[r] = bi;
#pragma unroll
            for (int q = 0; q < EPL; ++q)
                if (q * LANES + lane == bi) v[q] = -INFINITY;
        }
        if (lane == 0) {
            const float e1 = __expf(mv[1] - mv[0]);
            const float e2 = __expf(mv[2] - mv[0]);
            const float e3 = __expf(mv[3] - mv[0]);
            const float r = 1.f / (1.f + e1 + e2 + e3);
            const float w[4] = {r, e1 * r, e2 * r, e3 * r};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                IDX[t * 4 + k] = mi[k];
                WGT[t * 4 + k] = w[k];
                eid_[t * 4 + k] = mi[k];
                pos_[t * 4 + k] = atomicAdd(&cnt[mi[k]], 1);
            }
        }
    }
    __syncthreads();

    const int NE = EPL * LANES;
    int nt_e = 0;
    if (tid < NE) {
        nt_e = tid < E ? (cnt[tid] + TMAX - 1) / TMAX : 0;
        scan[tid] = nt_e;
    }
    __syncthreads();
    for (int off = 1; off < NE; off <<= 1) {
        int add = 0;
        if (tid < NE && tid >= off) add = scan[tid - off];
        __syncthreads();
        if (tid < NE) scan[tid] += add;
        __syncthreads();
    }
    if (tid < NE) tst[tid] = scan[tid] - nt_e;
    if (tid == 0) *TOTAL = scan[NE - 1];
    __syncthreads();

    if (tid < E) {
        const int c = cnt[tid], base = tst[tid];
        for (int j = 0; j * TMAX < c; ++j) {
            TILE_E[base + j] = tid;
            TILE_NV[base + j] = min(c - j * TMAX, TMAX);
        }
    }
    for (int p = tid; p < 4 * M; p += RNTHREAD)
        SORTED[tst[eid_[p]] * TMAX + pos_[p]] = p;
}

// ---------------------------------------------------------------------------

template <int NT, int VPL, int ROWS>
__global__ __launch_bounds__(NTHREAD) void gemm1(
    const __nv_bfloat16 *__restrict__ X, const uint8_t *__restrict__ W,
    const uint8_t *__restrict__ WS, const float *__restrict__ B,
    __half *__restrict__ Y1, const int *__restrict__ SORTED,
    const int *__restrict__ TILE_E, const int *__restrict__ TILE_NV,
    const int *__restrict__ TOTAL, int K, long ldx, long swe, long swr,
    long sse, long ssr, long sbe, long sy, float alpha, float beta, float lim) {
    if ((int)blockIdx.x >= *TOTAL) return;
    extern __shared__ __half xs[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int ids[TMAX];
    const int ntok = load_tile(SORTED, TILE_NV, blockIdx.x, ids);
    const int e = TILE_E[blockIdx.x];

    // Stage the activation rows, widening bf16 -> fp16 on the way in: the packed
    // dot needs fp16 operands and bf16 -> fp16 is exact for activations in range.
    const int NV8 = K / 8;
#pragma unroll 1
    for (int t = 0; t < ntok && t < NT; ++t) {
        const uint4 *src =
            reinterpret_cast<const uint4 *>(X + (long)(ids[t] >> 2) * ldx);
        uint4 *dst = reinterpret_cast<uint4 *>(xs + t * K);
        for (int i = tid; i < NV8; i += NTHREAD) {
            const uint4 v = src[i];
            const __nv_bfloat162 *b = reinterpret_cast<const __nv_bfloat162 *>(&v);
            __half2 h[4];
#pragma unroll
            for (int q = 0; q < 4; ++q)
                h[q] = __float22half2_rn(__bfloat1622float2(b[q]));
            dst[i] = *reinterpret_cast<uint4 *>(h);
        }
    }
    __syncthreads();

    constexpr int RPW = ROWS / NWARP;
    const int row0 = blockIdx.y * ROWS + warp * RPW;
    float acc[RMAX][NT];
#pragma unroll
    for (int i = 0; i < RPW; ++i)
#pragma unroll
        for (int t = 0; t < NT; ++t) acc[i][t] = 0.f;
    row_dots<NT, VPL, RPW>(W + (long)e * swe + (long)row0 * swr,
                           WS + (long)e * sse + (long)row0 * ssr, swr, ssr, xs,
                           K, lane, ntok, acc);
    if (lane) return;
    const float *bp = B + (long)e * sbe + row0;
#pragma unroll
    for (int i = 0; i < RPW; i += 2) {
        const float bg = bp[i], bu = bp[i + 1];
        const int col = (row0 + i) >> 1;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (t >= ntok) break;
            const float g = fminf(acc[i][t] + bg, lim);
            const float u = fminf(fmaxf(acc[i + 1][t] + bu, -lim), lim);
            const float y = (u + beta) * (g / (1.f + __expf(-alpha * g)));
            // trtllm-gen keeps the intermediate in bf16; round to it so the
            // second GEMM consumes the same operand (fp16 holds bf16 exactly).
            Y1[(long)ids[t] * sy + col] =
                __float2half(__bfloat162float(__float2bfloat16(y)));
        }
    }
}

template <int NT, int VPL, int ROWS>
__global__ __launch_bounds__(NTHREAD) void gemm2(
    const __half *__restrict__ Y1, const uint8_t *__restrict__ W,
    const uint8_t *__restrict__ WS, const float *__restrict__ B,
    const float *__restrict__ WGT, float *__restrict__ Y2,
    const int *__restrict__ SORTED, const int *__restrict__ TILE_E,
    const int *__restrict__ TILE_NV, const int *__restrict__ TOTAL, int K,
    long ldy, long swe, long swr, long sse, long ssr, long sbe, long sy2) {
    if ((int)blockIdx.x >= *TOTAL) return;
    extern __shared__ __half ys[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int ids[TMAX];
    const int ntok = load_tile(SORTED, TILE_NV, blockIdx.x, ids);
    const int e = TILE_E[blockIdx.x];

    const int NV8 = K / 8;
#pragma unroll 1
    for (int t = 0; t < ntok && t < NT; ++t) {
        const uint4 *src = reinterpret_cast<const uint4 *>(Y1 + (long)ids[t] * ldy);
        uint4 *dst = reinterpret_cast<uint4 *>(ys + t * K);
        for (int i = tid; i < NV8; i += NTHREAD) dst[i] = src[i];
    }
    __syncthreads();

    constexpr int RPW = ROWS / NWARP;
    const int row0 = blockIdx.y * ROWS + warp * RPW;
    float acc[RMAX][NT];
#pragma unroll
    for (int i = 0; i < RPW; ++i)
#pragma unroll
        for (int t = 0; t < NT; ++t) acc[i][t] = 0.f;
    row_dots<NT, VPL, RPW>(W + (long)e * swe + (long)row0 * swr,
                           WS + (long)e * sse + (long)row0 * ssr, swr, ssr, ys,
                           K, lane, ntok, acc);
    if (lane) return;
    const float *bp = B + (long)e * sbe + row0;
#pragma unroll
    for (int i = 0; i < RPW; ++i)
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (t >= ntok) break;
            Y2[(long)ids[t] * sy2 + row0 + i] = (acc[i][t] + bp[i]) * WGT[ids[t]];
        }
}

__global__ void combine(const float *__restrict__ Y2,
                        __nv_bfloat16 *__restrict__ OUT, int N, long sy2,
                        long so) {
    const int m = blockIdx.x;
    const float *p = Y2 + (long)4 * m * sy2;
    __nv_bfloat16 *o = OUT + (long)m * so;
    for (int n = blockIdx.y * blockDim.x + threadIdx.x; n < N;
         n += blockDim.x * gridDim.y)
        o[n] = __float2bfloat16(p[n] + p[n + sy2] + p[n + 2 * sy2] +
                                p[n + 3 * sy2]);
}

// ---------------------------------------------------------------------------

void moe_route(at::Tensor LOGITS, at::Tensor IDX, at::Tensor WGT,
               at::Tensor SORTED, at::Tensor TILE_E, at::Tensor TILE_NV,
               at::Tensor TOTAL) {
    const int M = LOGITS.size(0), E = LOGITS.size(1);
    TORCH_CHECK(M <= MMAX, "routed MoE: too many tokens");
    auto st = at::cuda::getCurrentCUDAStream();
#define LR(EPL)                                                                \
    route_align<EPL><<<1, NTHREAD, 0, st>>>(                                   \
        (const __nv_bfloat16 *)LOGITS.data_ptr(), LOGITS.stride(0), M, E,       \
        (int *)IDX.data_ptr(), (float *)WGT.data_ptr(),                        \
        (int *)SORTED.data_ptr(), (int *)TILE_E.data_ptr(),                     \
        (int *)TILE_NV.data_ptr(), (int *)TOTAL.data_ptr())
    if (E <= 32) { LR(1); }
    else if (E <= 64) { LR(2); }
    else if (E <= 128) { LR(4); }
    else { LR(8); }
#undef LR
}

// (token slots, values per lane, weight rows per CTA).  Narrow batches want
// fewer rows per CTA -- with only a handful of tiles the grid is what limits
// memory-level parallelism, not the per-CTA work.
#define DISPATCH(NT_, VPL_, ROWS_, LAUNCH)                                     \
    switch (((NT_) * 100 + (VPL_)) * 100 + (ROWS_)) {                          \
        case 11616: LAUNCH(1, 16, 16); break;                                  \
        case 11632: LAUNCH(1, 16, 32); break;                                  \
        case 21616: LAUNCH(2, 16, 16); break;                                  \
        case 21632: LAUNCH(2, 16, 32); break;                                  \
        case 43216: LAUNCH(4, 32, 16); break;                                  \
        case 43232: LAUNCH(4, 32, 32); break;                                  \
        case 41616: LAUNCH(4, 16, 16); break;                                  \
        default: LAUNCH(4, 16, 32); break;                                     \
    }

void moe_gemm1(at::Tensor X, at::Tensor W, at::Tensor WS, at::Tensor B,
               at::Tensor Y1, at::Tensor SORTED, at::Tensor TILE_E,
               at::Tensor TILE_NV, at::Tensor TOTAL, int64_t ntile, int64_t I,
               int64_t nt, int64_t vpl, int64_t rows, double alpha, double beta,
               double limit) {
    const int K = X.size(1);
    auto st = at::cuda::getCurrentCUDAStream();
    dim3 grid(ntile, I * 2 / rows);
    const int smem = nt * K * sizeof(__half);
#define LAUNCH1(NT, VPL, ROWS)                                                 \
    gemm1<NT, VPL, ROWS><<<grid, NTHREAD, smem, st>>>(                               \
        (const __nv_bfloat16 *)X.data_ptr(), (const uint8_t *)W.data_ptr(),    \
        (const uint8_t *)WS.data_ptr(), (const float *)B.data_ptr(),           \
        (__half *)Y1.data_ptr(), (const int *)SORTED.data_ptr(),               \
        (const int *)TILE_E.data_ptr(), (const int *)TILE_NV.data_ptr(),       \
        (const int *)TOTAL.data_ptr(), K, X.stride(0), W.stride(0),            \
        W.stride(1), WS.stride(0), WS.stride(1), B.stride(0), Y1.stride(0),    \
        alpha, beta, limit)
    DISPATCH(nt, vpl, rows, LAUNCH1);
#undef LAUNCH1
}

void moe_gemm2(at::Tensor Y1, at::Tensor W, at::Tensor WS, at::Tensor B,
               at::Tensor WGT, at::Tensor Y2, at::Tensor SORTED,
               at::Tensor TILE_E, at::Tensor TILE_NV, at::Tensor TOTAL,
               int64_t ntile, int64_t nt, int64_t vpl, int64_t rows) {
    const int K = Y1.size(1), N = Y2.size(1);
    auto st = at::cuda::getCurrentCUDAStream();
    dim3 grid(ntile, N / rows);
    const int smem = nt * K * sizeof(__half);
#define LAUNCH2(NT, VPL, ROWS)                                                 \
    gemm2<NT, VPL, ROWS><<<grid, NTHREAD, smem, st>>>(                               \
        (const __half *)Y1.data_ptr(), (const uint8_t *)W.data_ptr(),          \
        (const uint8_t *)WS.data_ptr(), (const float *)B.data_ptr(),           \
        (const float *)WGT.data_ptr(), (float *)Y2.data_ptr(),                 \
        (const int *)SORTED.data_ptr(), (const int *)TILE_E.data_ptr(),        \
        (const int *)TILE_NV.data_ptr(), (const int *)TOTAL.data_ptr(), K,     \
        Y1.stride(0), W.stride(0), W.stride(1), WS.stride(0), WS.stride(1),    \
        B.stride(0), Y2.stride(0))
    DISPATCH(nt, vpl, rows, LAUNCH2);
#undef LAUNCH2
}

void moe_combine(at::Tensor Y2, at::Tensor OUT) {
    const int M = OUT.size(0), N = OUT.size(1);
    auto st = at::cuda::getCurrentCUDAStream();
    dim3 grid(M, (N + 1023) / 1024);
    combine<<<grid, 256, 0, st>>>((const float *)Y2.data_ptr(),
                                  (__nv_bfloat16 *)OUT.data_ptr(), N,
                                  Y2.stride(0), OUT.stride(0));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moe_route", &moe_route);
    m.def("moe_gemm1", &moe_gemm1);
    m.def("moe_gemm2", &moe_gemm2);
    m.def("moe_combine", &moe_combine);
}
