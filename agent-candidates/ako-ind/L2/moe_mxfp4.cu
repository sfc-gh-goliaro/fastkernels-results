// Hand-written fused MXFP4 MoE for GPT-OSS (SM100 / B200).
//
// Everything here is local: no external MoE/GEMM operator is called.  Four
// launches per forward, no host<->device sync anywhere:
//
//   k_router     router GEMV -> bf16 logits, fp16 mirror of x, zeroed output
//   k_meta       top-k + softmax renormalize, per-expert token lists, work items
//   k_expert<0>  interleaved gate/up projection + OAI SwiGLU -> fp16 h
//   k_expert<1>  down projection, router-weight scale, atomic combine -> out
//
// Expert weights stay packed MXFP4 (2 values/byte) + E8M0 block scales all the
// way to the tensor core: every weight byte is read from HBM exactly once per
// forward and dequantized in-register directly into an mma.sync m16n8k16
// B-fragment.  process_weights_after_loading() pre-permutes the packed bytes so
// one 8-byte per-lane load covers the four B-fragments of a 64-wide k group --
// no shuffles, no ldmatrix, no SMEM staging of weights.  Block scales are stored
// as a *relative* fp16 exponent against a per-row reference, so folding them
// into the fp16 B operand is exact (power of two) and the inner loop never
// rescales the accumulator; the row reference is applied once in the epilogue.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define WARPS      8                      // warps per expert-GEMM CTA
#define CHUNK      16                     // max tokens per work chunk (mma M dim)
#define KG_W       64                     // k per weight group
#ifndef PF
#define PF         3                      // k-groups prefetched (wide tiling)
#endif
#ifndef PFSMALL
#define PFSMALL    3                      // narrow tiling: measured flat from 3..15
#endif
#ifndef CH_THRESH
// 0 disables the CHUNK=8 tiling entirely (M*K <= 0 is never true), so the CH=8
// instantiations below are compiled but never launched: measured, 3 CTAs/SM costs
// 30% rather than gaining anything (see ITERATIONS.md).  A build that raises this
// must re-check numerics on the CH=8 path and note that the `items` workspace is
// sized against CHUNK, not the runtime `ch`.
#define CH_THRESH  0
#endif
#ifndef ROUTER_FMA_MAX
// Router dispatch.  The mma path is much cheaper per MAC but tiles tokens 16 wide
// instead of 4, so it runs on ceil(M/16)*E/8 CTAs against ceil(M/4)*E/8: below
// ~M=64 the router is bound by how many CTAs are pulling the 737 KB of weights,
// not by arithmetic, and the FMA version's redundant reads are an advantage
// (measured 0.99x at M=26/60 for the mma path, 1.03x at M=398).
#define ROUTER_FMA_MAX 64
#endif
#ifndef MINCTA8
#define MINCTA8    3      // CTAs/SM the CH=8 (46 KB A tile) build must allow
#endif
#ifndef NTBIG
#define NTBIG 5            // 8-row N-tiles per warp for the wide (large-M) tiling
#endif
#ifndef MINCTA
#define MINCTA 2           // min CTAs/SM the register budget must allow
#endif
#define CDIV(a, b) (((a) + (b) - 1) / (b))

// ---------------------------------------------------------------- primitives
// Eight packed MXFP4 values (one 32-bit word) -> four fp16x2 registers.  ptxas
// turns the b8 sub-register moves into F2FP byte selectors (.B1/.B2/.B3), so the
// whole dequant of 8 values is 4 instructions with no shift/mask at all.
__device__ __forceinline__ void fp4x8_to_f16x2(unsigned w, unsigned* r) {
    asm("{ .reg .b8 b0, b1, b2, b3;      \n"
        "  mov.b32 {b0, b1, b2, b3}, %4; \n"
        "  cvt.rn.f16x2.e2m1x2 %0, b0;   \n"
        "  cvt.rn.f16x2.e2m1x2 %1, b1;   \n"
        "  cvt.rn.f16x2.e2m1x2 %2, b2;   \n"
        "  cvt.rn.f16x2.e2m1x2 %3, b3; }\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(w));
}

__device__ __forceinline__ unsigned mulf16x2(unsigned a, unsigned b) {
    unsigned r;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

__device__ __forceinline__ void mma16816bf(float* c, const unsigned* a, const unsigned* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void mma16816(float* c, const unsigned* a, const unsigned* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Scale bytes are stored pre-shifted (g = (15 + e_blk - e_row) << 2), so a
// single PRMT drops g into the high byte of both fp16 halves == 2^(e_blk-e_row).
__device__ __forceinline__ unsigned relscale_f16x2(unsigned sc, unsigned sel) {
    unsigned r;
    asm("prmt.b32 %0, %1, 0, %2;" : "=r"(r) : "r"(sc), "r"(sel));
    return r;
}

// ------------------------------------------------------------------- router
// grid (cdiv(M,TM), E/EGRP) x NTHR threads; one warp per expert, TM tokens held
// in shared memory.  Also emits the fp16 mirror of x (the expert GEMMs' A
// operand) and zeroes the output buffer that k_expert<1> accumulates into.
template <int TM, int EGRP, int NTHR>
__global__ __launch_bounds__(NTHR) void k_router(
        const __nv_bfloat16* __restrict__ x,
        const __nv_bfloat16* __restrict__ rw,
        const __nv_bfloat16* __restrict__ rbias,
        __nv_bfloat16* __restrict__ logits,
        __half* __restrict__ xh,
        __nv_bfloat16* __restrict__ outz,
        int M, int H, int E) {
    extern __shared__ char smem_raw[];
    __nv_bfloat16* sx = reinterpret_cast<__nv_bfloat16*>(smem_raw);

    const int t0 = blockIdx.x * TM;
    const int nrow = min(TM, M - t0);
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int H8 = H >> 3;

    for (int i = tid + nrow * H8; i < TM * H8; i += NTHR)      // zero pad rows so
        reinterpret_cast<uint4*>(sx)[i] = make_uint4(0, 0, 0, 0);  // the fixed-trip
    for (int i = tid; i < nrow * H8; i += NTHR) {                  // loop below is safe
        const int r = i / H8, c = i - r * H8;
        const uint4 v = reinterpret_cast<const uint4*>(x + (size_t)(t0 + r) * H)[c];
        reinterpret_cast<uint4*>(sx + r * H)[c] = v;
        if (blockIdx.y == 0) {
            const __nv_bfloat16* bv = reinterpret_cast<const __nv_bfloat16*>(&v);
            __half hv[8];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float f = __bfloat162float(bv[j]);
                hv[j] = __float2half(fminf(fmaxf(f, -60000.f), 60000.f));
            }
            reinterpret_cast<uint4*>(xh + (size_t)(t0 + r) * H)[c] =
                *reinterpret_cast<uint4*>(hv);
        } else if (blockIdx.y == 1) {
            reinterpret_cast<uint4*>(outz + (size_t)(t0 + r) * H)[c] =
                make_uint4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    const int eper = EGRP / (NTHR / 32);
    const int ebase = blockIdx.y * EGRP + warp * eper;
#pragma unroll 1
    for (int ei = 0; ei < eper; ei++) {
        const int e = ebase + ei;
        if (e >= E) break;
        float acc[TM];
#pragma unroll
        for (int t = 0; t < TM; t++) acc[t] = 0.f;
        const uint4* wrow = reinterpret_cast<const uint4*>(rw + (size_t)e * H);
        for (int c = lane; c < H8; c += 32) {
            const uint4 wv = wrow[c];
            const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv);
#pragma unroll
            for (int t = 0; t < TM; t++) {
                const uint4 xv = reinterpret_cast<const uint4*>(sx + t * H)[c];
                const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(&xv);
#pragma unroll
                for (int j = 0; j < 8; j++)
                    acc[t] = fmaf(__bfloat162float(wb[j]), __bfloat162float(xb[j]), acc[t]);
            }
        }
#pragma unroll
        for (int t = 0; t < TM; t++)
#pragma unroll
            for (int off = 16; off; off >>= 1)
                acc[t] += __shfl_down_sync(0xffffffff, acc[t], off);
        if (lane == 0) {
            const float bs = __bfloat162float(rbias[e]);
            for (int t = 0; t < nrow; t++)
                logits[(size_t)(t0 + t) * E + e] = __float2bfloat16(acc[t] + bs);
        }
    }
}

// -------------------------------------------------------- router (mma path)
// logits[M,E] = x[M,H] @ rw[E,H]^T + rb through the same tensor-core path as the
// experts.  The FMA version costs 2.4 instructions per MAC (two bf16->fp32
// converts per FMA); this is 6 instructions per k-tile of 16 -- 4 A loads, one
// 8-byte B load and one mma -- for 2048 MACs.
//
// grid (ceil(M/16), E/8): one CTA per (16 tokens, 8 experts), so the 737 KB
// router weight read stays spread over E/8 = 16 CTAs even at M=1 (one CTA per
// token tile would serialise it).  The CTA's warps split k and reduce through
// shared memory; the reduction order is fixed, so the bf16-rounded logits -- and
// therefore the top-k tie-breaks -- are deterministic.
//
// rwf holds rw in mma B-fragment order, [E/8][H/16][32][4] bf16, built once in
// process_weights_after_loading.
template <int NTHR>
__global__ __launch_bounds__(NTHR) void k_router_mma(
        const __nv_bfloat16* __restrict__ x,
        const __nv_bfloat16* __restrict__ rwf,
        const __nv_bfloat16* __restrict__ rbias,
        __nv_bfloat16* __restrict__ logits,
        __half* __restrict__ xh,
        __nv_bfloat16* __restrict__ outz,
        int M, int H, int E) {
    constexpr int WR = NTHR / 32;
    __shared__ float red[WR][128];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int t0 = blockIdx.x * 16;
    const int nrow = min(16, M - t0);
    const int NT = H >> 4;                       // k-tiles of 16
    const int gid = lane >> 2, tig = lane & 3;
    const int H8 = H >> 3;

    // Side work: the fp16 mirror of x that the expert GEMMs use as their A
    // operand, and the zeroing of the buffer k_expert<1> accumulates into.  Both
    // are (16 tokens x H) streams, so they are split across *all* the CTAs of the
    // expert-group dimension -- the first half writes xh, the second half zeroes
    // out.  Pinning them to blockIdx.y 0 and 1 left 2 CTAs doing 345 KB each at
    // M=26 and measured -2%.
    {
        const int half = gridDim.y >> 1;
        const int slot = (blockIdx.y < half) ? blockIdx.y : blockIdx.y - half;
        const int nsl = (blockIdx.y < half) ? half : gridDim.y - half;
        const int tot = nrow * H8;
        const int lo = (int)(((long long)slot * tot) / nsl);
        const int hi = (int)(((long long)(slot + 1) * tot) / nsl);
        if (blockIdx.y < half) {
            for (int i = lo + tid; i < hi; i += NTHR) {
                const int r = i / H8, c = i - r * H8;
                const uint4 v =
                    reinterpret_cast<const uint4*>(x + (size_t)(t0 + r) * H)[c];
                const __nv_bfloat16* bv = reinterpret_cast<const __nv_bfloat16*>(&v);
                __half hv[8];
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const float f = __bfloat162float(bv[j]);
                    hv[j] = __float2half(fminf(fmaxf(f, -60000.f), 60000.f));
                }
                reinterpret_cast<uint4*>(xh + (size_t)(t0 + r) * H)[c] =
                    *reinterpret_cast<uint4*>(hv);
            }
        } else {
            for (int i = lo + tid; i < hi; i += NTHR) {
                const int r = i / H8, c = i - r * H8;
                reinterpret_cast<uint4*>(outz + (size_t)(t0 + r) * H)[c] =
                    make_uint4(0, 0, 0, 0);
            }
        }
    }

    const int kt0 = (warp * NT) / WR, kt1 = ((warp + 1) * NT) / WR;
    const __nv_bfloat16* bp = rwf + ((size_t)blockIdx.y * NT + kt0) * 128 + lane * 4;
    const __nv_bfloat16* ap0 = x + (size_t)(t0 + gid) * H + tig * 2;
    const bool live0 = (t0 + gid) < M, live1 = (t0 + gid + 8) < M;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
    for (int T = kt0; T < kt1; T++) {
        const __nv_bfloat16* ap = ap0 + T * 16;
        unsigned a[4];
        a[0] = live0 ? *reinterpret_cast<const unsigned*>(ap) : 0u;
        a[1] = live1 ? *reinterpret_cast<const unsigned*>(ap + 8 * H) : 0u;
        a[2] = live0 ? *reinterpret_cast<const unsigned*>(ap + 8) : 0u;
        a[3] = live1 ? *reinterpret_cast<const unsigned*>(ap + 8 * H + 8) : 0u;
        const uint2 bv = *reinterpret_cast<const uint2*>(bp + (size_t)(T - kt0) * 128);
        unsigned b[2] = {bv.x, bv.y};
        mma16816bf(acc, a, b);
    }
#pragma unroll
    for (int j = 0; j < 4; j++) red[warp][lane * 4 + j] = acc[j];
    __syncthreads();
    if (tid < 128) {
        float sum = 0.f;
#pragma unroll
        for (int w = 0; w < WR; w++) sum += red[w][tid];
        const int l = tid >> 2, j = tid & 3;
        const int tok = (l >> 2) + ((j >> 1) ? 8 : 0);
        const int e = blockIdx.y * 8 + (l & 3) * 2 + (j & 1);
        if (tok < nrow)
            logits[(size_t)(t0 + tok) * E + e] =
                __float2bfloat16(sum + __bfloat162float(rbias[e]));
    }
}

// ------------------------------------------------------- routing metadata
// One CTA.  One lane owns a whole token: the top-k is a branch-predicated
// serial insert over the expert axis (no shuffles), fed by 16-byte loads that
// stay independent of the insert chain.  Ties go to the lower expert id (scan
// order), matching torch.topk and trtllm-gen.
//
// The insert is run over QSPL independent expert ranges and the partial top-4s
// merged afterwards: the arithmetic is identical (strict `>` in increasing expert
// order, and a global top-4 can take at most 4 entries from any range) but the
// dependent chain shrinks from E steps to E/QSPL + 4*QSPL, which is what this
// single-CTA kernel is limited by.
#ifndef QSPL
#define QSPL 4
#endif
#ifndef META512_MIN
#define META512_MIN 257    // token count above which k_meta gets 512 threads
#endif

// Insert (x, e) into a descending top-4 held in v/i.  Strict `>` keeps the lower
// expert id on ties.
#define TOPK_INS(v, i, x, e)                                                   \
    if ((x) > v[3]) {                                                          \
        if ((x) > v[1]) {                                                      \
            if ((x) > v[0]) { v[3]=v[2]; i[3]=i[2]; v[2]=v[1]; i[2]=i[1];       \
                              v[1]=v[0]; i[1]=i[0]; v[0]=(x);  i[0]=(e); }     \
            else            { v[3]=v[2]; i[3]=i[2]; v[2]=v[1]; i[2]=i[1];      \
                              v[1]=(x);  i[1]=(e); }                           \
        } else {                                                               \
            if ((x) > v[2]) { v[3]=v[2]; i[3]=i[2]; v[2]=(x); i[2]=(e); }       \
            else            { v[3]=(x);  i[3]=(e); }                            \
        }                                                                      \
    }

template <int K, int NTHR>
__global__ __launch_bounds__(NTHR) void k_meta(
        const __nv_bfloat16* __restrict__ logits,
        int* __restrict__ perm, float* __restrict__ pw,
        int* __restrict__ cnt_g, int* __restrict__ off_g,
        int* __restrict__ items, int* __restrict__ nwork,
        int M, int E, int chunk) {
    extern __shared__ char sm[];
    int* cnt = reinterpret_cast<int*>(sm);
    int* cur = cnt + E;
    int* offs = cur + E;
    short* ti = reinterpret_cast<short*>(offs + E);
    float* tw = reinterpret_cast<float*>(ti + M * K);

    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    for (int i = tid; i < E; i += NTHR) { cnt[i] = 0; cur[i] = 0; }
    __syncthreads();

    for (int t = tid; t < M; t += NTHR) {
        const int CQ = (E >> 3) / QSPL;             // 16-byte loads per range
        float pv[QSPL][4];
        int pi[QSPL][4];
#pragma unroll
        for (int q = 0; q < QSPL; q++)
#pragma unroll
            for (int r = 0; r < 4; r++) { pv[q][r] = -INFINITY; pi[q][r] = E; }
        const uint4* lg = reinterpret_cast<const uint4*>(logits + (size_t)t * E);
#pragma unroll
        for (int c = 0; c < CQ; c++) {
#pragma unroll
            for (int q = 0; q < QSPL; q++) {        // QSPL independent chains
                const uint4 raw = lg[q * CQ + c];
                const __nv_bfloat16* bv = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const float x = __bfloat162float(bv[j]);
                    const int e = (q * CQ + c) * 8 + j;
                    TOPK_INS(pv[q], pi[q], x, e)
                }
            }
        }
        float mv[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
        int mi[4] = {E, E, E, E};
#pragma unroll                       // ranges in increasing expert order, each
        for (int q = 0; q < QSPL; q++)          // already in descending value order
#pragma unroll
            for (int r = 0; r < 4; r++) TOPK_INS(mv, mi, pv[q][r], pi[q][r])
        const float v0 = mv[0], v1 = mv[1], v2 = mv[2], v3 = mv[3];
        const int i0 = mi[0], i1 = mi[1], i2 = mi[2], i3 = mi[3];
        const float e1 = __expf(v1 - v0), e2 = __expf(v2 - v0), e3 = __expf(v3 - v0);
        const float inv = 1.f / (1.f + e1 + e2 + e3);
        ti[t * K + 0] = (short)i0; tw[t * K + 0] = inv;
        ti[t * K + 1] = (short)i1; tw[t * K + 1] = e1 * inv;
        ti[t * K + 2] = (short)i2; tw[t * K + 2] = e2 * inv;
        ti[t * K + 3] = (short)i3; tw[t * K + 3] = e3 * inv;
        atomicAdd(&cnt[i0], 1); atomicAdd(&cnt[i1], 1);
        atomicAdd(&cnt[i2], 1); atomicAdd(&cnt[i3], 1);
    }
    __syncthreads();

    if (warp == 0) {                       // exclusive scan of the expert counts
        int total = 0;
        for (int base = 0; base < E; base += 32) {
            const int c = (base + lane < E) ? cnt[base + lane] : 0;
            int sc = c;
#pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const int o = __shfl_up_sync(0xffffffff, sc, off);
                if (lane >= off) sc += o;
            }
            if (base + lane < E) offs[base + lane] = sc - c + total;
            total += __shfl_sync(0xffffffff, sc, 31);
        }
    } else if (warp == 1) {                // work items: one per (expert, chunk)
        int nw = 0;
        for (int base = 0; base < E; base += 32) {
            const int c = (base + lane < E) ? CDIV(cnt[base + lane], chunk) : 0;
            int sc = c;
#pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const int o = __shfl_up_sync(0xffffffff, sc, off);
                if (lane >= off) sc += o;
            }
            const int excl = sc - c + nw;
            for (int ch = 0; ch < c; ch++) items[excl + ch] = (base + lane) | (ch << 16);
            nw += __shfl_sync(0xffffffff, sc, 31);
        }
        if (lane == 0) *nwork = nw;
    }
    __syncthreads();
    for (int i = tid; i < E; i += NTHR) { cnt_g[i] = cnt[i]; off_g[i] = offs[i]; }
    for (int i = tid; i < M * K; i += NTHR) {
        const int e = ti[i];
        const int slot = offs[e] + atomicAdd(&cur[e], 1);
        perm[slot] = i / K;
        pw[slot] = tw[i];
    }
}

// ------------------------------------------------------------ expert GEMMs
// EPI==0 : gate/up + SwiGLU -> h   (A = xh rows gathered by perm, N = 2I)
// EPI==1 : down + combine -> out   (A = h rows, N = H)
// KGN (k groups) is a template parameter so every address inside the k loop is a
// compile-time offset from one pointer.  NTT (8-row N-tiles per warp) trades
// A-fragment reuse against the number of CTAs: small token counts activate few
// experts, so they need the narrow tile to fill the machine.  HALF skips the
// A-fragments of token rows 8..15 when the chunk holds <= 8 tokens, halving
// shared-memory traffic for decode-sized batches.
template <int EPI, int KGN, int NTT, int CH>
__device__ __forceinline__ void expert_item(
        const char* __restrict__ A,
        const uint8_t* __restrict__ wp, const uint8_t* __restrict__ sp,
        const float* __restrict__ bias, const float* __restrict__ fac,
        const int* __restrict__ perm, const float* __restrict__ pw,
        __half* __restrict__ hout, __nv_bfloat16* __restrict__ out,
        int rbase, int n, int c0, int N, float alpha, float limit, bool half8) {
    constexpr int PFT = (NTT == 1) ? PFSMALL : PF;
    constexpr int AFT = (CH <= 8) ? 8 : 16;       // fragment bytes per lane per k-tile
    const int lane = threadIdx.x & 31;
    const int gid = lane >> 2, tig = lane & 3;
    // A is staged in mma-fragment order, so the whole A-fragment of a k-tile is
    // one 16-byte LDS in exactly the register order mma.sync wants -- no MOVs to
    // build the operand quad, and 4 LDS.128 per k-group instead of 16 LDS.32.
    const char* Abase = A + lane * AFT;

    float acc[NTT][4];
#pragma unroll
    for (int i = 0; i < NTT; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.f;

    unsigned wbuf[PFT][NTT][2];
    unsigned sbuf[PFT][NTT];
#pragma unroll
    for (int p = 0; p < PFT; p++)
#pragma unroll
        for (int t = 0; t < NTT; t++) {
            const uint2 v = *reinterpret_cast<const uint2*>(
                wp + (size_t)t * KGN * 256 + p * 256);
            wbuf[p][t][0] = v.x; wbuf[p][t][1] = v.y;
            sbuf[p][t] = *reinterpret_cast<const unsigned short*>(
                sp + (size_t)t * KGN * 16 + p * 16);
        }

#pragma unroll 1
    for (int kg0 = 0; kg0 < KGN; kg0 += PFT) {
#pragma unroll
        for (int p = 0; p < PFT; p++) {
            const int kg = kg0 + p;
            const char* ap = Abase + kg * (4 * 32 * AFT);   // 32 lanes per k-tile
            uint4 af[4];
#pragma unroll
            for (int c = 0; c < 4; c++) {
                if (CH <= 8) {              // only rows 0..7 exist in the tile
                    const uint2 v = *reinterpret_cast<const uint2*>(ap + c * 32 * AFT);
                    af[c].x = af[c].y = v.x;
                    af[c].z = af[c].w = v.y;
                } else {
                    af[c] = *reinterpret_cast<const uint4*>(ap + c * 32 * AFT);
                }
            }
#pragma unroll
            for (int t = 0; t < NTT; t++) {
                const unsigned w0 = wbuf[p][t][0], w1 = wbuf[p][t][1];
                const unsigned sc = sbuf[p][t];
                const unsigned s0 = relscale_f16x2(sc, 0x0404);
                const unsigned s1 = relscale_f16x2(sc, 0x1414);
                unsigned q0[4], q1[4], b[2];
                fp4x8_to_f16x2(w0, q0);
                fp4x8_to_f16x2(w1, q1);
                b[0] = mulf16x2(q0[0], s0); b[1] = mulf16x2(q0[1], s0);
                mma16816(acc[t], &af[0].x, b);
                b[0] = mulf16x2(q0[2], s0); b[1] = mulf16x2(q0[3], s0);
                mma16816(acc[t], &af[1].x, b);
                b[0] = mulf16x2(q1[0], s1); b[1] = mulf16x2(q1[1], s1);
                mma16816(acc[t], &af[2].x, b);
                b[0] = mulf16x2(q1[2], s1); b[1] = mulf16x2(q1[3], s1);
                mma16816(acc[t], &af[3].x, b);
                if (kg0 + PFT < KGN) {
                    const uint2 v = *reinterpret_cast<const uint2*>(
                        wp + (size_t)t * KGN * 256 + (kg + PFT) * 256);
                    wbuf[p][t][0] = v.x; wbuf[p][t][1] = v.y;
                    sbuf[p][t] = *reinterpret_cast<const unsigned short*>(
                        sp + (size_t)t * KGN * 16 + (kg + PFT) * 16);
                }
            }
        }
    }

#pragma unroll
    for (int t = 0; t < NTT; t++) {
        const int row = rbase + t * 8 + tig * 2;
        if (row >= N) continue;
        const float2 fa = *reinterpret_cast<const float2*>(fac + row);
        const float2 bi = *reinterpret_cast<const float2*>(bias + row);
        for (int m = 0; m < (half8 ? 1 : 2); m++) {
            const int tok = gid + m * 8;
            if (tok >= n) continue;
            if (EPI == 0) {
                float g = acc[t][m * 2 + 0] * fa.x + bi.x;
                float u = acc[t][m * 2 + 1] * fa.y + bi.y;
                g = fminf(g, limit);
                u = fminf(fmaxf(u, -limit), limit);
                const float hv = g * __frcp_rn(1.f + __expf(-alpha * g)) * (u + 1.f);
                hout[(size_t)(c0 + tok) * (N >> 1) + (row >> 1)] = __float2half(hv);
            } else {
                const float rwt = pw[c0 + tok];
                const float y0 = (acc[t][m * 2 + 0] * fa.x + bi.x) * rwt;
                const float y1 = (acc[t][m * 2 + 1] * fa.y + bi.y) * rwt;
                atomicAdd(reinterpret_cast<__nv_bfloat162*>(
                              out + (size_t)perm[c0 + tok] * N + row),
                          __floats2bfloat162_rn(y0, y1));
            }
        }
    }
}

template <int EPI, int KGN, int NTT, int CH>
__global__ __launch_bounds__(WARPS * 32, CH <= 8 ? MINCTA8 : MINCTA) void k_expert(
        const __half* __restrict__ Aglob,
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
        const float* __restrict__ bias, const float* __restrict__ fac,
        const int* __restrict__ perm, const float* __restrict__ pw,
        const int* __restrict__ cnt, const int* __restrict__ off,
        const int* __restrict__ items, const int* __restrict__ nwork,
        __half* __restrict__ hout, __nv_bfloat16* __restrict__ out,
        int N, int NROWS, int KKreal, float alpha, float limit) {
    constexpr int KK = KGN * KG_W;                // padded k extent
    constexpr int RCTA = WARPS * NTT * 8;
    extern __shared__ __align__(16) char sm[];
    char* A = sm;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int NTILES = NROWS / RCTA;
    const size_t wexp = (size_t)(NROWS / 8) * KGN * 256;
    const size_t sexp = (size_t)(NROWS / 8) * KGN * 16;
    const int total = (*nwork) * NTILES;

    // Contiguous item range per CTA (not grid-stride): consecutive items share
    // the same (expert, token chunk), so the A tile is staged once per CTA
    // instead of once per tile.
    // Balanced contiguous split: every CTA gets floor/ceil of total/grid items,
    // so a work count just above the grid size does not idle half the CTAs.
    const int ibeg = (int)(((long long)blockIdx.x * total) / gridDim.x);
    const int iend = (int)(((long long)(blockIdx.x + 1) * total) / gridDim.x);
    int cur_key = -1;
    for (int it = ibeg; it < iend; it++) {
        const int tile = it % NTILES;
        const int wi = items[it / NTILES];
        const int e = wi & 0xffff, ch = wi >> 16;
        const int c0 = off[e] + ch * CH;
        const int n = min(CH, cnt[e] - ch * CH);

        if (cur_key != wi) {
            if (cur_key >= 0) __syncthreads();
            // Scatter into mma-fragment order: 16 bytes per (k-tile of 16, lane)
            // holding the quad {(gid,k0), (gid+8,k0), (gid,k0+8), (gid+8,k0+8)}
            // for lane = gid*4 + tig, k0 = T*16 + 2*tig.  One thread owns one
            // (row pair, k-tile), so the two rows of a quad land in one STS.64.
            // When the chunk holds <= 8 tokens the low row is written into both
            // halves: the mma then computes it twice (the epilogue drops the
            // copy) which is cheaper than a second load path.
            constexpr int AFT = (CH <= 8) ? 8 : 16;
            const int NT16 = KKreal >> 4;
            const int nr = min(n, 8);           // live row groups (0..7)
            for (int i = tid; i < nr * NT16; i += WARPS * 32) {
                const int r = i / NT16, T = i - r * NT16;
                const size_t s0 = (EPI == 0) ? (size_t)perm[c0 + r] : (size_t)(c0 + r);
                const uint4* g0 = reinterpret_cast<const uint4*>(
                                      Aglob + s0 * KKreal) + T * 2;
                const uint4 a0 = g0[0], a1 = g0[1];        // row r, k' 0..7 / 8..15
                const unsigned* lw = reinterpret_cast<const unsigned*>(&a0);
                const unsigned* hw = reinterpret_cast<const unsigned*>(&a1);
                char* dst = A + ((size_t)T * 32 + r * 4) * AFT;
                if (CH <= 8) {
#pragma unroll
                    for (int tg = 0; tg < 4; tg++)
                        *reinterpret_cast<uint2*>(dst + tg * 8) =
                            make_uint2(lw[tg], hw[tg]);
                } else {
                    // second row of the quad; when it holds no token the low row
                    // is duplicated so the load path stays a single LDS.128.
                    const int r1 = (r + 8 < n) ? r + 8 : r;
                    const size_t s1 = (EPI == 0) ? (size_t)perm[c0 + r1]
                                                 : (size_t)(c0 + r1);
                    const uint4* g1 = reinterpret_cast<const uint4*>(
                                          Aglob + s1 * KKreal) + T * 2;
                    const uint4 b0 = g1[0], b1 = g1[1];
                    const unsigned* lw1 = reinterpret_cast<const unsigned*>(&b0);
                    const unsigned* hw1 = reinterpret_cast<const unsigned*>(&b1);
#pragma unroll
                    for (int tg = 0; tg < 4; tg++) {
                        *reinterpret_cast<uint2*>(dst + tg * 16 + 0) =
                            make_uint2(lw[tg], lw1[tg]);
                        *reinterpret_cast<uint2*>(dst + tg * 16 + 8) =
                            make_uint2(hw[tg], hw1[tg]);
                    }
                }
            }
            if (KK > KKreal) {                // weights are zero there, but the
                const int nz = ((KK - KKreal) >> 4) * (32 * AFT / 16);
                for (int i = tid; i < nz; i += WARPS * 32)   // mma would see 0*NaN
                    reinterpret_cast<uint4*>(A)[(KKreal >> 4) * (32 * AFT / 16) + i] =
                        make_uint4(0, 0, 0, 0);
            }
            cur_key = wi;
            __syncthreads();
        }

        const int rbase = tile * RCTA + warp * (NTT * 8);
        const uint8_t* wp = W + (size_t)e * wexp + (size_t)(rbase >> 3) * KGN * 256
                            + (size_t)(threadIdx.x & 31) * 8;
        const uint8_t* sp = S + (size_t)e * sexp + (size_t)(rbase >> 3) * KGN * 16
                            + (size_t)((threadIdx.x & 31) >> 2) * 2;
        const float* bp = bias + (size_t)e * NROWS;
        const float* fp = fac + (size_t)e * NROWS;
        expert_item<EPI, KGN, NTT, CH>(A, wp, sp, bp, fp, perm, pw, hout, out,
                                       rbase, n, c0, N, alpha, limit, CH <= 8 || n <= 8);
    }
}

// ------------------------------------------------------------------ host
struct Ctx {
    const uint8_t *w13, *s13, *w2, *s2;
    const float *b13, *f13, *b2, *f2;
    const __nv_bfloat16 *rw, *rwf, *rb;
    int H, I, E, K, maxM, nsm;
    int rows13, rows2;            // padded row counts of the packed layouts
    float alpha, limit;
    torch::Tensor ws;
    __nv_bfloat16* logits;
    __half *xh, *h;
    int *perm, *cnt, *off, *items, *nwork;
    float* pw;
};

static inline size_t align256(size_t v) { return (v + 255) & ~(size_t)255; }

int64_t moe_make_ctx(torch::Tensor w13, torch::Tensor s13, torch::Tensor b13,
                     torch::Tensor f13, torch::Tensor w2, torch::Tensor s2,
                     torch::Tensor b2, torch::Tensor f2, torch::Tensor rw,
                     torch::Tensor rwf, torch::Tensor rb, int64_t H, int64_t I, int64_t E,
                     int64_t K, int64_t maxM, int64_t rows13, int64_t rows2,
                     double alpha, double limit) {
    Ctx* c = new Ctx();
    c->w13 = w13.data_ptr<uint8_t>();  c->s13 = s13.data_ptr<uint8_t>();
    c->b13 = b13.data_ptr<float>();    c->f13 = f13.data_ptr<float>();
    c->w2 = w2.data_ptr<uint8_t>();    c->s2 = s2.data_ptr<uint8_t>();
    c->b2 = b2.data_ptr<float>();      c->f2 = f2.data_ptr<float>();
    c->rw = reinterpret_cast<const __nv_bfloat16*>(rw.data_ptr());
    c->rwf = reinterpret_cast<const __nv_bfloat16*>(rwf.data_ptr());
    c->rb = reinterpret_cast<const __nv_bfloat16*>(rb.data_ptr());
    c->H = H; c->I = I; c->E = E; c->K = K; c->maxM = maxM;
    c->rows13 = rows13; c->rows2 = rows2;
    // Both tilings must divide the padded row counts exactly, otherwise
    // NROWS/RCTA would silently drop the tail rows.  Returning 0 makes the
    // caller fall back to the delegate instead.
    for (int rows : {(int)rows13, (int)rows2})
        for (int rc : {WARPS * 8, WARPS * NTBIG * 8})
            if (rows % rc != 0) { delete c; return 0; }
    const int kg1 = CDIV(H, KG_W), kg2 = CDIV(I, KG_W);
    if (H % KG_W || I % KG_W || E % 8 || K != 4 || E < 16) { delete c; return 0; }
    if ((kg1 != 45 && kg1 != 23) || (kg2 != 45 && kg2 != 23)) { delete c; return 0; }
    c->alpha = (float)alpha; c->limit = (float)limit;
    cudaDeviceGetAttribute(&c->nsm, cudaDevAttrMultiProcessorCount, w13.get_device());

    const size_t nlog = align256((size_t)maxM * E * 2);
    const size_t nxh = align256((size_t)maxM * H * 2);
    const size_t nh = align256((size_t)maxM * K * I * 2);
    const size_t nperm = align256((size_t)maxM * K * 4);
    const size_t npw = align256((size_t)maxM * K * 4);
    const size_t nsmall = align256((size_t)(2 * E + 1) * 4
                                   + (size_t)(E + maxM * K / CHUNK + 8) * 4);
    auto opt = torch::TensorOptions().dtype(torch::kUInt8).device(w13.device());
    c->ws = torch::empty({(int64_t)(nlog + nxh + nh + nperm + npw + nsmall)}, opt);
    uint8_t* p = c->ws.data_ptr<uint8_t>();
    c->logits = reinterpret_cast<__nv_bfloat16*>(p); p += nlog;
    c->xh = reinterpret_cast<__half*>(p); p += nxh;
    c->h = reinterpret_cast<__half*>(p); p += nh;
    c->perm = reinterpret_cast<int*>(p); p += nperm;
    c->pw = reinterpret_cast<float*>(p); p += npw;
    c->cnt = reinterpret_cast<int*>(p);
    c->off = c->cnt + E;
    c->nwork = c->off + E;
    c->items = c->nwork + 1;
    return (int64_t)c;
}

template <int EPI, int KGN, int NTT, int CH>
static void launch_one(const Ctx* c, int grid, size_t smem, cudaStream_t st,
                       __nv_bfloat16* outp) {
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute((void*)k_expert<EPI, KGN, NTT, CH>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        attr = true;
    }
    if (EPI == 0)
        k_expert<EPI, KGN, NTT, CH><<<grid, WARPS * 32, smem, st>>>(
            c->xh, c->w13, c->s13, c->b13, c->f13, c->perm, c->pw, c->cnt, c->off,
            c->items, c->nwork, c->h, nullptr, 2 * c->I, c->rows13, c->H,
            c->alpha, c->limit);
    else
        k_expert<EPI, KGN, NTT, CH><<<grid, WARPS * 32, smem, st>>>(
            c->h, c->w2, c->s2, c->b2, c->f2, c->perm, c->pw, c->cnt, c->off,
            c->items, c->nwork, nullptr, outp, c->H, c->rows2, c->I,
            c->alpha, c->limit);
}

template <int EPI, int KGN>
static void launch_expert(const Ctx* c, int grid, size_t smem, cudaStream_t st,
                          __nv_bfloat16* outp, int ntt, int ch) {
    if (ch <= 8) {
        if (ntt == 1) launch_one<EPI, KGN, 1, 8>(c, grid, smem, st, outp);
        else          launch_one<EPI, KGN, NTBIG, 8>(c, grid, smem, st, outp);
    } else {
        if (ntt == 1) launch_one<EPI, KGN, 1, 16>(c, grid, smem, st, outp);
        else          launch_one<EPI, KGN, NTBIG, 16>(c, grid, smem, st, outp);
    }
}

torch::Tensor moe_forward(int64_t ctxp, torch::Tensor x) {
    Ctx* c = reinterpret_cast<Ctx*>(ctxp);
    const int M = (int)x.size(0);
    const int H = c->H, I = c->I, E = c->E, K = c->K;
    auto out = torch::empty({(int64_t)M, (int64_t)H}, x.options());
    cudaStream_t st = at::cuda::getCurrentCUDAStream();
    __nv_bfloat16* outp = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());

    // Tiny batches: one token per CTA row-slot and 4 experts per CTA so the
    // 737 KB router weight read is spread over enough CTAs.
    if (M <= ROUTER_FMA_MAX) {
        dim3 g1(M, E / 4);
        k_router<1, 4, 128><<<g1, 128, (size_t)1 * H * 2, st>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), c->rw, c->rb,
            c->logits, c->xh, outp, M, H, E);
    } else {
        dim3 g1(CDIV(M, 16), E / 8);
        k_router_mma<256><<<g1, 256, 0, st>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), c->rwf, c->rb,
            c->logits, c->xh, outp, M, H, E);
    }

    // A CHUNK of 8 halves the A tile, doubling resident warps; only worth it
    // while the expected tokens/expert stays under 8.
    const int ch = (M * K <= CH_THRESH * E) ? 8 : CHUNK;
    const size_t smem_meta = (size_t)(3 * E) * 4 + (size_t)M * K * 2 + (size_t)M * K * 4;
    if (M >= META512_MIN)
        k_meta<4, 512><<<1, 512, smem_meta, st>>>(c->logits, c->perm, c->pw, c->cnt,
                                                  c->off, c->items, c->nwork, M, E, ch);
    else
        k_meta<4, 256><<<1, 256, smem_meta, st>>>(c->logits, c->perm, c->pw, c->cnt,
                                                  c->off, c->items, c->nwork, M, E, ch);

    const int grid = c->nsm * 2;
    const int kg1 = CDIV(H, KG_W), kg2 = CDIV(I, KG_W);   // validated in make_ctx
    const size_t smem = (size_t)ch * ((kg1 > kg2 ? kg1 : kg2) * KG_W) * 2;
    // ch=16 -> 16 B per (k-tile, lane) = ch*KK*2; ch=8 -> 8 B, i.e. also ch*KK*2.
    // Few active experts (small M) -> narrow row tiles so the grid still fills
    // the machine; the estimate only has to be monotone in M.
    const int nw_est = min(M * K, E);
#ifdef FORCE_NTT
    const int ntt = FORCE_NTT;
#else
    const int ntt = (nw_est * (c->rows2 / (WARPS * NTBIG * 8)) >= grid) ? NTBIG : 1;
#endif
    switch (kg1) {
        case 45: launch_expert<0, 45>(c, grid, smem, st, outp, ntt, ch); break;
        case 23: launch_expert<0, 23>(c, grid, smem, st, outp, ntt, ch); break;
        default: TORCH_CHECK(false, "unsupported hidden size");
    }
    switch (kg2) {
        case 45: launch_expert<1, 45>(c, grid, smem, st, outp, ntt, ch); break;
        case 23: launch_expert<1, 23>(c, grid, smem, st, outp, ntt, ch); break;
        default: TORCH_CHECK(false, "unsupported intermediate size");
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("make_ctx", &moe_make_ctx);
    m.def("forward", &moe_forward);
}
