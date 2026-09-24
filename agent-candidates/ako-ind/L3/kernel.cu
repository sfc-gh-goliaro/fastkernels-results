// Fused elementwise / attention kernels for the AF3 diffusion transformer.
//
// At the captured shapes the operator is CPU-issue bound (see kernel.py), so
// each kernel here exists to collapse a run of tiny eager ops into one launch.
// All of them keep the reference's rounding sequence: LayerNorm statistics and
// every intermediate are computed in fp32 and rounded to bf16 exactly where
// the eager reference rounds (its LayerNorm returns bf16 before the AdaLN
// affine, its SiLU rounds before the SwiGLU product, attention rounds the
// scores, the biased scores and the softmax probabilities).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float b2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 f2b(float x) { return __float2bfloat16(x); }
// bf16 round-trip: the reference materializes a bf16 tensor at this point.
__device__ __forceinline__ float rnd(float x) { return b2f(f2b(x)); }

#define BLK 256
#define NWARP (BLK / 32)
// Head-dim row stride padding for the attention tiles.  In the score loop
// consecutive lanes walk the *key* index, so lane c reads shared word
// c*(SD/2) + d/2: the stride must be *odd in words* for all 32 lanes to land on
// distinct banks, i.e. SD = D + SMEM_PAD == 2 (mod 4).  D is always a multiple
// of 16 here, so a pad of 2 does it; the padding of 8 this replaces leaves
// gcd(SD/2, 32) == 4, i.e. a 4-way conflict at both captured head dims.
#define SMEM_PAD 2
#define RES_MAX 8      // per-thread row slots in k_res_adaln (C <= RES_MAX*BLK)

// One barrier, not two: instead of thread 0 combining the per-warp partials and
// broadcasting through shared memory, every thread re-sums them itself in the
// same order -- identical arithmetic, and these kernels are barrier-bound (the
// token stack's AdaLN rows are 16x768, i.e. 16 CTAs, so nothing hides latency).
// Callers pass disjoint scratch for consecutive reductions so no barrier is
// needed to protect the previous one's reads.
__device__ __forceinline__ float block_sum(float v, float* sh) {
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    float t = 0.f;
#pragma unroll
    for (int i = 0; i < NWARP; ++i) t += sh[i];
    return t;
}

// ---------------------------------------------------------------------------
// residual + AdaLN:  a += gate * delta ;  out = g * (LayerNorm(a) + shift)
//
// Covers three call sites with one kernel: the plain AdaLN before a block's
// attention (no residual), the post-attention residual followed by the
// transition's AdaLN, and the post-transition residual followed by the next
// block's AdaLN.  ``a`` is updated in place.
// ---------------------------------------------------------------------------
__global__ void k_res_adaln(bf16* __restrict__ a, const bf16* __restrict__ gate,
                            const bf16* __restrict__ delta,
                            const bf16* __restrict__ rmask,
                            const bf16* __restrict__ g, const bf16* __restrict__ shift,
                            bf16* __restrict__ out, bf16* __restrict__ nrm, int C,
                            float eps, int has_res, int has_out, int has_nrm,
                            int lda, int ldgate, int ldd, int ldg, int ldo) {
    __shared__ float sh[2][NWARP];
    // The precomputed gate / scale / shift tensors are column slices of the
    // stacked conditioning GEMMs, so their row stride is the full stacked
    // width, not C.
    bf16* ar = a + (size_t)blockIdx.x * lda;

    // Hold the row in registers: the LayerNorm needs three passes over it and
    // the rows are small (C <= RES_MAX * BLK at every captured config), so
    // re-reading global memory three times is pure latency.  The residual add
    // feeds those registers *directly* rather than storing and reading ``a``
    // back -- a thread only ever touches its own elements, so the round trip and
    // the barrier that ordered it were both unnecessary.
    float x[RES_MAX];
    int nv_ = 0;
    float s = 0.f;
    const bool want = (has_out | has_nrm) != 0;
    if (has_res) {
        const bf16* gr = gate + (size_t)blockIdx.x * ldgate;
        const bf16* dr = delta + (size_t)blockIdx.x * ldd;
        // The reference materializes gate*delta (and then *mask) as bf16 before
        // the residual add; keep both roundings or the error over 24 blocks
        // lands on the 1%-relative tolerance.
        if (rmask) {
            const float mv = b2f(rmask[blockIdx.x]);
            for (int i = threadIdx.x; i < C; i += BLK, ++nv_) {
                float v = rnd(b2f(ar[i]) + rnd(rnd(b2f(gr[i]) * b2f(dr[i])) * mv));
                ar[i] = f2b(v);
                x[nv_] = v;
                s += v;
            }
        } else {
            for (int i = threadIdx.x; i < C; i += BLK, ++nv_) {
                float v = rnd(b2f(ar[i]) + rnd(b2f(gr[i]) * b2f(dr[i])));
                ar[i] = f2b(v);
                x[nv_] = v;
                s += v;
            }
        }
        if (!want) return;
    } else {
        for (int i = threadIdx.x; i < C; i += BLK, ++nv_) {
            x[nv_] = b2f(ar[i]);
            s += x[nv_];
        }
    }
    float mean = block_sum(s, sh[0]) / (float)C;
    float q = 0.f;
    for (int j = 0; j < nv_; ++j) {
        float d = x[j] - mean;
        q += d * d;
    }
    float rstd = rsqrtf(block_sum(q, sh[1]) / (float)C + eps);

    const bf16* gr = g + (size_t)blockIdx.x * ldg;
    const bf16* sr = shift + (size_t)blockIdx.x * ldg;
    bf16* orow = out + (size_t)blockIdx.x * ldo;
    bf16* nrow = nrm + (size_t)blockIdx.x * ldo;
    int j = 0;
    for (int i = threadIdx.x; i < C; i += BLK, ++j) {
        float nv = rnd((x[j] - mean) * rstd);
        if (has_nrm) nrow[i] = f2b(nv);
        if (has_out) orow[i] = f2b(b2f(gr[i]) * rnd(nv + b2f(sr[i])));
    }
}

// ---------------------------------------------------------------------------
// gathered AdaLN (cross-attention key side):
//   out[r] = g[r] * (LayerNorm(a)[idx[r]] * valid[r] + shift[r])
//
// ``a_norm`` is already normalized (the query side needs the same rows), and
// the reference zeroes invalid key rows *before* normalizing -- LayerNorm of a
// zero row is zero, so masking after the fact is equivalent.
// ---------------------------------------------------------------------------
__global__ void k_gather_adaln(const bf16* __restrict__ an, const long* __restrict__ idx,
                               const bf16* __restrict__ valid, const bf16* __restrict__ g,
                               const bf16* __restrict__ shift, bf16* __restrict__ out,
                               int rows, int C, int ldan, int ldg, int ldo) {
    // One thread per output element rather than one CTA per row: the cross
    // stack's key rows are C=128 wide, so a CTA-per-row launch left half of
    // every 256-thread block idle and moved 2 bytes per active thread.
    int e = blockIdx.x * BLK + threadIdx.x;
    if (e >= rows * C) return;
    int row = e / C, i = e - row * C;
    float av = b2f(an[(size_t)idx[row] * ldan + i]);
    float vd = b2f(valid[row]);
    out[(size_t)row * ldo + i] =
        f2b(b2f(g[(size_t)row * ldg + i]) * rnd(av * vd + b2f(shift[(size_t)row * ldg + i])));
}

// ---------------------------------------------------------------------------
// LayerNorm over the last dim, bf16 in / bf16 out, fp32 statistics.
//
// Exists only because ``F.layer_norm`` is pathological at the cross stack's
// shape: normalizing z is [12*32*128, 16] -- 49152 rows of 16 -- and torch runs
// 62 us on 3 MB of traffic (48 GB/s) by giving each row its own block.  One
// thread per row instead measures 8 us.  The reference promotes to fp32 around
// this call (``LayerNorm(promote_fp32=True)``), but the promotion is a no-op
// here: ``F.layer_norm`` on a bf16 input already reduces in fp32, and the two
// forms were verified bit-identical, so the casts are dropped too.
//
// Statistics as sum then sum of squared deviations, which is the closest of the
// forms tried to torch's own (5 of 786432 elements differ by one bf16 ulp; a
// Welford pass or a fused ``x*(rstd*w) - mean*rstd*w`` epilogue is 4x further
// out).  This output feeds a pair bias that is added to attention scores holding
// a +-1e9 mask term, so a one-ulp residue on 6e-4% of it is immaterial.
// ---------------------------------------------------------------------------
__global__ void k_lnz(const bf16* __restrict__ x, const bf16* __restrict__ w,
                      bf16* __restrict__ y, int rows, int C, float eps) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const bf16* xr = x + (size_t)r * C;
    bf16* yr = y + (size_t)r * C;
    float s = 0.f;
    for (int c = 0; c < C; ++c) s += b2f(xr[c]);
    const float mean = s / (float)C;
    float ss = 0.f;
    for (int c = 0; c < C; ++c) { float d = b2f(xr[c]) - mean; ss += d * d; }
    const float rstd = rsqrtf(ss / (float)C + eps);
    for (int c = 0; c < C; ++c)
        yr[c] = f2b((b2f(xr[c]) - mean) * rstd * (w ? b2f(w[c]) : 1.f));
}

// ---------------------------------------------------------------------------
// Cross stack: key-block gather indices, block mask and mask bias.
//
// This replaces ~25 eager ops on 1536-element tensors, which cost 57 us per call
// -- not dispatch (the whole call is one graph replay) but 25 serialized tiny
// kernels.  ``n_real = mask.sum()`` stays a torch reduction, because its fp32
// reduction *order* is not reproducible here and a one-ulp difference in the
// bf16 result would move a gather index; everything downstream of it is
// deterministic integer / bf16 arithmetic and is reproduced exactly.
//
// The bf16 rounding is load-bearing and is why each step rounds explicitly.  In
// the reference the index arithmetic mixes int32 tensors with a bf16 ``n_real``,
// and PyTorch's type promotion makes the *result* bf16 -- so ``initial`` is cast
// to bf16 before the shift is applied, and an index like 431 is not
// representable and becomes 432.  Reproducing that (rather than staying in
// int32) is the whole point.
// ---------------------------------------------------------------------------
__global__ void k_blkidx(const bf16* __restrict__ mpad, const bf16* __restrict__ nreal,
                         long* __restrict__ idx, bf16* __restrict__ valid,
                         bf16* __restrict__ mk, int nblk, int nq, int nk) {
    int e = blockIdx.x * BLK + threadIdx.x;
    if (e >= nblk * nk) return;
    const int j = e / nk, tt = e - j * nk;
    const int half = nk / 2, off = nq / 2;
    const int base = off + j * nq;
    const int init0 = base - half, initl = base + half - 1, initt = base + tt - half;

    const float nr = b2f(nreal[0]);
    const int under = init0 < 0 ? -init0 : 0;            // relu, int32
    const float nrm1 = rnd(nr - 1.f);                    // bf16(n_real - 1)
    const float ovf = rnd(rnd((float)initl) - nrm1);     // int32 -> bf16, then -
    const float over = ovf > 0.f ? ovf : 0.f;            // relu
    const float shift = under > 0 ? rnd((float)under) : -over;
    const float fin = rnd(rnd((float)initt) + shift);

    const bool bad = (fin < 0.f) || (fin >= nr);
    const float hi = nrm1 > 0.f ? nrm1 : 0.f;
    float safe = fin > 0.f ? fin : 0.f;                  // clamp(fin, 0, hi)
    if (safe > hi) safe = hi;
    const long ix = (long)safe;
    idx[e] = ix;
    const float vv = bad ? 0.f : 1.f;
    valid[e] = f2b(vv);
    mk[e] = f2b(vv * b2f(mpad[ix]));
}

// mask_bias[j, q, t] = (mask[j*nq + q] * mk[j, t] - 1) * inf, with the two
// in-place eager steps' bf16 roundings kept.
__global__ void k_maskbias(const bf16* __restrict__ mpad, const bf16* __restrict__ mk,
                           bf16* __restrict__ out, int nblk, int nq, int nk,
                           float inf) {
    int e = blockIdx.x * BLK + threadIdx.x;
    if (e >= nblk * nq * nk) return;
    const int t = e % nk, q = (e / nk) % nq, j = e / (nq * nk);
    const float v = rnd(b2f(mpad[j * nq + q]) * b2f(mk[j * nk + t]));
    out[e] = f2b(rnd(v - 1.f) * inf);
}

// ---------------------------------------------------------------------------
// Fused attention: softmax(Q K^T + bias) V, gated by sigmoid(gate), written
// straight out in [row, head * D + d] layout.  One CTA per (block, head);
// everything lives in shared memory (the captured shapes are tiny).
//
// q / gate are addressed as [bb * Q + i, h * D + d] with row stride ldq / ldg,
// k / v as [bb * K + j, h * D + d] with row stride ldk / ldv, so the same
// kernel serves the self-attention stack (one fused q,k,v,gate projection:
// four column ranges of one tensor) and the cross-attention stack (a q,gate
// projection of the queries and a k,v projection of the gathered keys).
//
// The pair bias is read *in the layout the stacked bias GEMM produces* --
// ``zb[(bb*Q + i)*K + j][block*H + h]``, addressed as
// ``bb * zbb + h * zbh + i * zbs`` -- and the (block-invariant) mask bias is
// added here rather than beforehand.  Materializing a contiguous
// ``[block, bb, H, Q, K]`` plane instead cost 68 us per cross call: the source
// is a five-way permuted view of the GEMM output, so the eager add reads it at
// ~2% of peak.  The strided read itself is free: feeding this kernel a
// contiguous plane instead measures 4.83 vs 4.91 us on the token shape and 11.91
// vs 11.83 us on the cross shape.
//
// Summing bias and mask bias in fp32 and rounding once is exactly what the eager
// ``torch.add(bias, mask_bias, out=bf16)`` did.
//
// ``QS`` query rows per CTA: at the cross shape one CTA per (block, head) is
// only nblk*H = 48 CTAs on 148 SMs, so the kernel is a third of a GPU wide and
// latency-bound rather than throughput-bound.  Splitting the query tile
// re-loads the k/v tiles per split (L2 traffic, cheap) and buys the missing
// occupancy.  It changes no arithmetic: every score and every output element is
// still one thread's sequential ascending dot product.
// ---------------------------------------------------------------------------
__global__ void k_attn(const bf16* __restrict__ qp, int ldq, const bf16* __restrict__ kp,
                       int ldk, const bf16* __restrict__ vp, int ldv,
                       const bf16* __restrict__ gp, int ldg,
                       const bf16* __restrict__ zb, int zbs, int zbh, int zbb,
                       const bf16* __restrict__ mb, int mbrow, int mbblk,
                       bf16* __restrict__ out, int ldo,
                       int Q, int K, int D, int H, int QS) {
    extern __shared__ char smem[];
    const int nsp = Q / QS;                       // query splits per (block, head)
    const int qi = blockIdx.x % nsp, rest = blockIdx.x / nsp;
    const int bb = rest / H, h = rest - rest / H * H;
    const int qb = qi * QS;                       // first query row of this CTA
    const int SD = D + SMEM_PAD;      // see SMEM_PAD
    bf16* sq = (bf16*)smem;
    bf16* sk = sq + QS * SD;
    bf16* sv = sk + K * SD;
    float* sp = (float*)(sv + K * SD);

    const int tid = threadIdx.x;
    const bf16* q0 = qp + (size_t)(bb * Q + qb) * ldq + h * D;
    const bf16* k0 = kp + (size_t)(bb * K) * ldk + h * D;
    const bf16* v0 = vp + (size_t)(bb * K) * ldv + h * D;
    for (int i = tid; i < QS * D; i += BLK) {
        int r = i / D, c = i - r * D;
        sq[r * SD + c] = q0[(size_t)r * ldq + c];
    }
    for (int i = tid; i < K * D; i += BLK) {
        int r = i / D, c = i - r * D;
        sk[r * SD + c] = k0[(size_t)r * ldk + c];
        sv[r * SD + c] = v0[(size_t)r * ldv + c];
    }
    __syncthreads();

    const bf16* bs = zb + (size_t)bb * zbb + (size_t)h * zbh
                   + (size_t)qb * K * zbs;
    const bf16* ms = mb + (size_t)bb * mbblk + (size_t)qb * mbrow;
    for (int i = tid; i < QS * K; i += BLK) {
        int r = i / K, c = i - r * K;
        float acc = 0.f;
        for (int d = 0; d < D; ++d) acc += b2f(sq[r * SD + d]) * b2f(sk[c * SD + d]);
        sp[i] = rnd(rnd(acc) + rnd(b2f(bs[(size_t)i * zbs])
                                   + b2f(ms[r * mbrow + c])));
    }
    __syncthreads();

    int wid = tid >> 5, lane = tid & 31;
    for (int r = wid; r < QS; r += NWARP) {
        float* row = sp + r * K;
        float m = -INFINITY;
        for (int c = lane; c < K; c += 32) m = fmaxf(m, row[c]);
#pragma unroll
        for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        float s = 0.f;
        for (int c = lane; c < K; c += 32) {
            float e = expf(row[c] - m);
            row[c] = e;
            s += e;
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        float inv = 1.f / s;
        for (int c = lane; c < K; c += 32) row[c] = rnd(row[c] * inv);
    }
    __syncthreads();

    const bf16* g0 = gp + (size_t)(bb * Q + qb) * ldg + h * D;
    bf16* o0 = out + (size_t)(bb * Q + qb) * ldo + h * D;
    for (int i = tid; i < QS * D; i += BLK) {
        int r = i / D, c = i - r * D;
        float acc = 0.f;
        for (int j = 0; j < K; ++j) acc += sp[r * K + j] * b2f(sv[j * SD + c]);
        float gv = b2f(g0[(size_t)r * ldg + c]);
        o0[(size_t)r * ldo + c] = f2b(rnd(acc) * rnd(1.f / (1.f + expf(-gv))));
    }
}

// One warp per CTA.  The mma n-tile is 8 wide and M=16 is one tile tall, so a
// warp *is* the whole output tile; making the CTA any wider only concentrates
// the same 384 warps onto 96 of the 148 SMs (measured 12.5 -> 7.0 us on the
// 16x3072x768 projection).  A is re-read by every CTA as a result -- 9.4 MB
// against B's 4.7 MB -- but A is 24 KB and stays in L1, and staging it in shared
// memory instead measured *worse* (see the note on scheduling below).
#define GWARPS 1
#define GBN (GWARPS * 8)
__global__ void k_gemm(const bf16* __restrict__ A, int lda,
                       const bf16* __restrict__ B,
                       const bf16* __restrict__ bias, bf16* __restrict__ C, int ldc,
                       int M, int N, int K, int mode);

// ---------------------------------------------------------------------------
// launchers (pointer level, so the whole-stack driver below can reuse them)
// ---------------------------------------------------------------------------
static inline bf16* P(const at::Tensor& t) { return (bf16*)t.data_ptr(); }

static void launch_res_adaln(bf16* a, const bf16* gate, const bf16* delta,
                             const bf16* rmask, const bf16* g, const bf16* shift,
                             bf16* out, bf16* nrm, int rows, int C, float eps,
                             int lda, int ldgate, int ldd, int ldg, int ldo,
                             cudaStream_t st) {
    k_res_adaln<<<rows, BLK, 0, st>>>(a, gate, delta, rmask, g, shift, out, nrm, C,
                                      eps, gate != nullptr, out != nullptr,
                                      nrm != nullptr, lda, ldgate, ldd, ldg, ldo);
}

static void launch_gather_adaln(const bf16* an, const long* idx, const bf16* valid,
                                const bf16* g, const bf16* shift, bf16* out,
                                int rows, int C, int ldan, int ldg, int ldo,
                                cudaStream_t st) {
    k_gather_adaln<<<(rows * C + BLK - 1) / BLK, BLK, 0, st>>>(
        an, idx, valid, g, shift, out, rows, C, ldan, ldg, ldo);
}

static void launch_lnz(const bf16* x, const bf16* w, bf16* y, int rows, int C,
                       float eps, cudaStream_t st) {
    k_lnz<<<(rows + 255) / 256, 256, 0, st>>>(x, w, y, rows, C, eps);
}

static void launch_attn(const bf16* q, int ldq, const bf16* k, int ldk,
                        const bf16* v, int ldv, const bf16* g, int ldg,
                        const bf16* zb, int zbs, int zbh, int zbb,
                        const bf16* mb, int mbrow, int mbblk, bf16* out, int ldo,
                        int Q, int K, int D, int H, int nblk, int qsplit,
                        cudaStream_t st) {
    // Split the query tile until there are enough CTAs to fill the GPU, but
    // never below 8 query rows (below that the k/v re-load dominates).
    int QS = Q;
    if (qsplit > 0) {
        QS = Q / qsplit;
    } else {
        while (QS % 2 == 0 && QS > 8 && nblk * H * (Q / QS) < 2 * 148) QS >>= 1;
    }
    size_t sm = (size_t)((QS + 2 * K) * (D + SMEM_PAD)) * sizeof(bf16)
              + (size_t)(QS * K) * sizeof(float);
    static int cfg_sm = 0;
    if ((int)sm > 48 * 1024 && cfg_sm < (int)sm) {
        cudaFuncSetAttribute(k_attn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)sm);
        cfg_sm = (int)sm;
    }
    k_attn<<<nblk * H * (Q / QS), BLK, sm, st>>>(q, ldq, k, ldk, v, ldv, g, ldg, zb,
                                             zbs, zbh, zbb, mb, mbrow, mbblk, out,
                                             ldo, Q, K, D, H, QS);
}

static void launch_gemm(const bf16* A, int lda, const bf16* B,
                        const bf16* bias, bf16* C, int ldc, int M, int N, int K,
                        int mode, cudaStream_t st) {
    dim3 grid((N + GBN - 1) / GBN, (M + 15) / 16);
    k_gemm<<<grid, GWARPS * 32, 0, st>>>(A, lda, B, bias, C, ldc, M, N, K, mode);
}

// ---------------------------------------------------------------------------
// GEMM:  C[M,N] = A[M,K] @ B^T (+ bias[N]),  bf16 throughout, with A row-major
// [M,K] and B **[N,K]** -- i.e. the weight in its natural `nn.Linear` layout, so
// no transpose is needed when the plan is built.
//
// Why hand-written rather than cuBLAS through torch: in this CPU-bound regime a
// `torch.mm` dispatch costs ~8 us against ~4 us for an extension call, and the
// SwiGLU epilogue folds in for free.  M is 16 in the token stack, which is
// exactly the `mma.m16n8k16` tile height -- one warp covers the full M and an
// 8-wide n-slice, so parallelism is set purely by N (and by M/16 blocks for the
// atom stack) instead of being traded against per-thread register blocking.
// Without tensor cores this shape is hopeless: at M=16 an fp32-FMA kernel needs
// ~1 shared-memory load per FMA and measures ~4x slower than the launch it
// saves.
//
// Why B is [N,K] and not [K,N]: the mma B fragment wants lane (gid, tig) to hold
// B[n0+gid][kc], B[n0+gid][kc+1], B[n0+gid][kc+8], B[n0+gid][kc+9].  With k
// contiguous those are two aligned 32-bit loads, and the warp's eight distinct n
// rows each fetch one *fully used* 32 B sector.  With n contiguous ([K,N]) the
// same fragment costs four 16-bit loads whose warp-level requests are 16 B out
// of every 32 B sector.  Halving the instruction count and doubling sector
// utilisation measured 12.5 -> 7.0 us on 16x3072x768 -- and it changes nothing
// numerically, because each output is still one lane's single ascending mma
// chain (verified bit-identical to the [K,N] version on every captured shape).
//
// The `__launch_bounds__` + `#pragma unroll` pair is what sets how many k-steps
// of loads are in flight, and it is worth re-checking with `nvcc -Xptxas -v`
// after a toolchain change.  Without the launch bound ptxas caps itself at 32
// registers for unroll 4, 8, 16 and 32 -- one k-step in flight, fully exposed to
// load latency -- and happens to batch ~16 steps only at unroll 24 (130
// registers); neighbouring unroll factors then differ by 2x for no reason to do
// with the memory system.  With `__launch_bounds__(32, 1)` the register count
// rises monotonically with the unroll factor (60 / 92 / 126 / 148 / 156 at
// 4 / 8 / 12 / 16 / 24) and never spills, so the unroll means what it says; 48
// is another 1.15x over 24 and is the whole K at the token stack's projections.
//
// Writing the pipeline out by hand (explicit depth-D register queues, or a
// load-G-then-mma-G block) was tried and is slower: it stops ptxas from
// software-pipelining across the group boundary.  Staging A or B through shared
// memory with `cp.async` was also tried -- it puts far more bytes in flight (the
// whole [8,K] B slab per CTA) and is *still* slightly behind.
//
// mode 0: plain (+ optional bias).
// mode 1: SwiGLU -- B holds 2N rows, out = SiLU(A@B[:N]^T) * (A@B[N:]^T), which
//         fuses `linear_a`/`linear_b` and the activation into one launch and
//         halves the output traffic.
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned pack2(const bf16* p) {
    return *reinterpret_cast<const unsigned*>(p);  // {p[0], p[1]}, k-consecutive
}

__global__ __launch_bounds__(GWARPS * 32, 1)
void k_gemm(const bf16* __restrict__ A, int lda,
            const bf16* __restrict__ B,
            const bf16* __restrict__ bias, bf16* __restrict__ C, int ldc,
            int M, int N, int K, int mode) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int gid = lane >> 2, tig = lane & 3;
    const int m0 = blockIdx.y * 16;
    const int n0 = blockIdx.x * GBN + warp * 8;

    const int ra0 = m0 + gid, ra1 = ra0 + 8;
    const bf16* Ar0 = A + (size_t)ra0 * lda;
    const bf16* Ar1 = A + (size_t)ra1 * lda;
    const bool va0 = ra0 < M, va1 = ra1 < M;
    // Clamp instead of branching: a lane's B fragment only ever feeds output
    // column ``nb``, which the epilogue drops when it is out of range.
    const int nb = n0 + gid < N ? n0 + gid : N - 1;
    const bf16* Br = B + (size_t)nb * K;            // B is [N, K]
    const bf16* Br2 = B + (size_t)(nb + N) * K;     // mode 1: the `b` half

    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
#pragma unroll 48
    for (int k0 = 0; k0 < K; k0 += 16) {
        const int kc = k0 + tig * 2;
        unsigned a0 = va0 ? pack2(Ar0 + kc) : 0u;
        unsigned a1 = va1 ? pack2(Ar1 + kc) : 0u;
        unsigned a2 = va0 ? pack2(Ar0 + kc + 8) : 0u;
        unsigned a3 = va1 ? pack2(Ar1 + kc + 8) : 0u;
        unsigned b0 = pack2(Br + kc), b1 = pack2(Br + kc + 8);
        // NOT volatile: a volatile asm is a scheduling barrier, which pins every
        // k-step's loads behind the previous mma and leaves the kernel entirely
        // exposed to load latency (that alone cost 49 us vs ~3 us per call).
        asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        if (mode == 1) {
            unsigned e0 = pack2(Br2 + kc), e1 = pack2(Br2 + kc + 8);
            asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(e0), "r"(e1));
        }
    }

    // C fragment: rows {m0+gid, m0+gid+8}, cols {n0+2*tig, n0+2*tig+1}.
    const int nc = n0 + tig * 2;
    const float acc[4] = {c0, c1, c2, c3};
    const float acd[4] = {d0, d1, d2, d3};
    for (int e = 0; e < 4; ++e) {
        int r = m0 + gid + (e >> 1) * 8;
        int col = nc + (e & 1);
        if (r >= M || col >= N) continue;
        float v;
        if (mode == 1) {
            float x = rnd(acc[e]);
            v = rnd(x / (1.f + expf(-x))) * rnd(acd[e]);
        } else {
            v = acc[e] + (bias ? b2f(bias[col]) : 0.f);
        }
        C[(size_t)r * ldc + col] = f2b(v);
    }
}

// ---------------------------------------------------------------------------
// per-kernel pybind shims (used by the eager-driven path and the unit tests)
// ---------------------------------------------------------------------------
void res_adaln(at::Tensor a, at::Tensor gate, at::Tensor delta, at::Tensor rmask,
               at::Tensor g, at::Tensor shift, at::Tensor out, at::Tensor nrm,
               double eps) {
    int rows = (int)a.size(0), C = (int)a.size(1);
    bool hr = gate.numel() > 0, ho = out.numel() > 0, hn = nrm.numel() > 0;
    int ldo = ho ? (int)out.stride(0) : (hn ? (int)nrm.stride(0) : C);
    launch_res_adaln(P(a), hr ? P(gate) : nullptr, hr ? P(delta) : nullptr,
                     rmask.numel() ? P(rmask) : nullptr, ho ? P(g) : nullptr,
                     ho ? P(shift) : nullptr, ho ? P(out) : nullptr,
                     hn ? P(nrm) : nullptr, rows, C, (float)eps, (int)a.stride(0),
                     hr ? (int)gate.stride(0) : 0, hr ? (int)delta.stride(0) : 0,
                     ho ? (int)g.stride(0) : 0, ldo,
                     at::cuda::getCurrentCUDAStream());
}

void gather_adaln(at::Tensor an, at::Tensor idx, at::Tensor valid, at::Tensor g,
                  at::Tensor shift, at::Tensor out) {
    launch_gather_adaln(P(an), (const long*)idx.data_ptr(), P(valid), P(g), P(shift),
                        P(out), (int)out.size(0), (int)out.size(1),
                        (int)an.stride(0), (int)g.stride(0), (int)out.stride(0),
                        at::cuda::getCurrentCUDAStream());
}

void attn(at::Tensor q, int64_t ldq, at::Tensor k, int64_t ldk, at::Tensor v,
          int64_t ldv, at::Tensor g, int64_t ldg, at::Tensor zb, int64_t zbs,
          int64_t zbh, int64_t zbb, at::Tensor mb, int64_t mbrow, int64_t mbblk,
          at::Tensor out, int64_t Q, int64_t K, int64_t D, int64_t H,
          int64_t nblk, int64_t qsplit) {
    launch_attn(P(q), (int)ldq, P(k), (int)ldk, P(v), (int)ldv, P(g), (int)ldg,
                P(zb), (int)zbs, (int)zbh, (int)zbb,
                P(mb), (int)mbrow, (int)mbblk,
                P(out), (int)out.size(1), (int)Q, (int)K, (int)D, (int)H,
                (int)nblk, (int)qsplit, at::cuda::getCurrentCUDAStream());
}

// Cross-stack block index / mask preamble (see k_blkidx / k_maskbias).
void blkidx(at::Tensor mpad, at::Tensor nreal, at::Tensor idx, at::Tensor valid,
            at::Tensor mk, at::Tensor mbias, int64_t nblk, int64_t nq, int64_t nk,
            double inf) {
    auto st = at::cuda::getCurrentCUDAStream();
    k_blkidx<<<((int)(nblk * nk) + BLK - 1) / BLK, BLK, 0, st>>>(
        P(mpad), P(nreal), (long*)idx.data_ptr(), P(valid), P(mk),
        (int)nblk, (int)nq, (int)nk);
    k_maskbias<<<((int)(nblk * nq * nk) + BLK - 1) / BLK, BLK, 0, st>>>(
        P(mpad), P(mk), P(mbias), (int)nblk, (int)nq, (int)nk, (float)inf);
}

// Fused LayerNorm of the pair representation (see k_lnz).
void lnz(at::Tensor x, at::Tensor w, at::Tensor y, double eps) {
    launch_lnz(P(x), w.numel() ? P(w) : nullptr, P(y), (int)x.size(0),
               (int)x.size(1), (float)eps, at::cuda::getCurrentCUDAStream());
}

void gemm(at::Tensor A, at::Tensor B, at::Tensor bias, at::Tensor C, int64_t mode) {
    launch_gemm(P(A), (int)A.stride(0), P(B),
                bias.numel() ? P(bias) : nullptr, P(C), (int)C.stride(0),
                (int)A.size(0), (int)C.size(1), (int)A.size(1), (int)mode,
                at::cuda::getCurrentCUDAStream());
}

// ---------------------------------------------------------------------------
// Whole-stack drivers.
//
// Once the per-block work is 7 launches, the *dispatch* is what is left: 7
// pybind crossings x 24 blocks is ~750 us of Python at ~4.4 us per extension
// call.  Running the block loop here instead leaves one crossing per call, and
// all the per-block slicing (each block's gate / scale / shift columns of the
// stacked conditioning GEMMs, and its bias plane) becomes pointer arithmetic.
//
// ``cfg`` is a small int64 CPU tensor built once per shape, holding geometry
// followed by the stable scratch and per-block weight pointers -- see
// ``DiffusionTransformer._bufs`` for the layout.
// ---------------------------------------------------------------------------
#define CFG_W 22           // first weight-pointer slot
#define CFG_WPB 6          // weight pointers per block

void run_self(at::Tensor cfg, at::Tensor a, at::Tensor g1, at::Tensor g2,
              at::Tensor zb, at::Tensor mb, at::Tensor mcol, double eps) {
    const int64_t* c = cfg.data_ptr<int64_t>();
    const int nb = (int)c[0], rows = (int)c[1], c_a = (int)c[3], c_hid = (int)c[4];
    const int c_ff = (int)c[5], nh = (int)c[6], dh = (int)c[7];
    const int Q = (int)c[8], K = (int)c[9], ldq = (int)c[11];
    bf16* ax = (bf16*)c[12];
    bf16* tx = (bf16*)c[13];
    bf16* qkvg = (bf16*)c[14];
    bf16* og = (bf16*)c[16];
    bf16* ao = (bf16*)c[17];
    bf16* act = (bf16*)c[18];
    bf16* o2 = (bf16*)c[19];

    bf16* A = P(a);
    const bf16* G1 = P(g1);
    const bf16* G2 = P(g2);
    const bf16* ZB = P(zb);
    const bf16* MB = P(mb);
    const bf16* MC = P(mcol);
    const int zbs = (int)zb.size(1);     // = nb * nh, the stacked bias width
    const int ldg1 = (int)g1.stride(0), ldg2 = (int)g2.stride(0);
    const int blk = nb * c_a;            // one site's column block
    const float e = (float)eps;
    cudaStream_t st = at::cuda::getCurrentCUDAStream();

    const bf16* gate_p = nullptr;
    const bf16* delta_p = nullptr;
    const bf16* rm = nullptr;
    for (int b = 0; b < nb; ++b) {
        const int64_t* w = c + CFG_W + (int64_t)b * CFG_WPB;
        launch_res_adaln(A, gate_p, delta_p, rm, G1 + b * c_a, G1 + 2 * blk + b * c_a,
                         ax, nullptr, rows, c_a, e, c_a, ldg2, c_a, ldg1, c_a, st);
        launch_gemm(ax, c_a, (const bf16*)w[0], (const bf16*)w[1], qkvg,
                    4 * c_hid, rows, 4 * c_hid, c_a, 0, st);
        launch_attn(qkvg, ldq, qkvg + c_hid, ldq, qkvg + 2 * c_hid, ldq,
                    qkvg + 3 * c_hid, ldq, ZB + b * nh, zbs, 1, Q * K * zbs,
                    MB, 0, 0, og, c_hid, Q, K, dh, nh, 1, 0, st);
        launch_gemm(og, c_hid, (const bf16*)w[3], nullptr, ao, c_a, rows, c_a,
                    c_hid, 0, st);
        launch_res_adaln(A, G2 + b * c_a, ao, nullptr, G1 + blk + b * c_a,
                         G1 + 3 * blk + b * c_a, tx, nullptr, rows, c_a, e, c_a,
                         ldg2, c_a, ldg1, c_a, st);
        launch_gemm(tx, c_a, (const bf16*)w[4], nullptr, act, c_ff, rows,
                    c_ff, c_a, 1, st);
        launch_gemm(act, c_ff, (const bf16*)w[5], nullptr, o2, c_a, rows, c_a,
                    c_ff, 0, st);
        gate_p = G2 + blk + b * c_a;
        delta_p = o2;
        rm = MC;
    }
    launch_res_adaln(A, gate_p, delta_p, rm, nullptr, nullptr, nullptr, nullptr,
                     rows, c_a, e, c_a, ldg2, c_a, 0, c_a, st);
}

void run_cross(at::Tensor cfg, at::Tensor a, at::Tensor g1, at::Tensor g2,
               at::Tensor gk, at::Tensor zb, at::Tensor mb, at::Tensor mcol,
               at::Tensor idx, at::Tensor vcol, double eps) {
    const int64_t* c = cfg.data_ptr<int64_t>();
    const int nb = (int)c[0], rows = (int)c[1], keys = (int)c[2];
    const int c_a = (int)c[3], c_hid = (int)c[4], c_ff = (int)c[5];
    const int nh = (int)c[6], dh = (int)c[7];
    const int Q = (int)c[8], K = (int)c[9], nblk = (int)c[10], ldq = (int)c[11];
    bf16* aq = (bf16*)c[12];
    bf16* tx = (bf16*)c[13];
    bf16* qg = (bf16*)c[14];
    bf16* kv = (bf16*)c[15];
    bf16* og = (bf16*)c[16];
    bf16* ao = (bf16*)c[17];
    bf16* act = (bf16*)c[18];
    bf16* o2 = (bf16*)c[19];
    bf16* nrm = (bf16*)c[20];
    bf16* ak = (bf16*)c[21];

    bf16* A = P(a);
    const bf16* G1 = P(g1);
    const bf16* G2 = P(g2);
    const bf16* GK = P(gk);
    const bf16* ZB = P(zb);
    const bf16* MB = P(mb);
    const bf16* MC = P(mcol);
    const bf16* VC = P(vcol);
    const int zbs = (int)zb.size(1);
    const long* IDX = (const long*)idx.data_ptr();
    const int ldg1 = (int)g1.stride(0), ldg2 = (int)g2.stride(0);
    const int ldgk = (int)gk.stride(0);
    const int blk = nb * c_a;
    const float e = (float)eps;
    cudaStream_t st = at::cuda::getCurrentCUDAStream();

    const bf16* gate_p = nullptr;
    const bf16* delta_p = nullptr;
    const bf16* rm = nullptr;
    for (int b = 0; b < nb; ++b) {
        const int64_t* w = c + CFG_W + (int64_t)b * CFG_WPB;
        launch_res_adaln(A, gate_p, delta_p, rm, G1 + b * c_a, G1 + 2 * blk + b * c_a,
                         aq, nrm, rows, c_a, e, c_a, ldg2, c_a, ldg1, c_a, st);
        launch_gather_adaln(nrm, IDX, VC, GK + b * c_a, GK + blk + b * c_a, ak,
                            keys, c_a, c_a, ldgk, c_a, st);
        launch_gemm(aq, c_a, (const bf16*)w[0], (const bf16*)w[1], qg,
                    2 * c_hid, rows, 2 * c_hid, c_a, 0, st);
        launch_gemm(ak, c_a, (const bf16*)w[2], nullptr, kv, 2 * c_hid,
                    keys, 2 * c_hid, c_a, 0, st);
        // Cross stack: the mask bias is a [nblk, Q, K] plane, block-invariant.
        launch_attn(qg, ldq, kv, ldq, kv + c_hid, ldq, qg + c_hid, ldq,
                    ZB + b * nh, zbs, 1, Q * K * zbs, MB, K, Q * K, og, c_hid,
                    Q, K, dh, nh, nblk, 0, st);
        launch_gemm(og, c_hid, (const bf16*)w[3], nullptr, ao, c_a, rows, c_a,
                    c_hid, 0, st);
        launch_res_adaln(A, G2 + b * c_a, ao, nullptr, G1 + blk + b * c_a,
                         G1 + 3 * blk + b * c_a, tx, nullptr, rows, c_a, e, c_a,
                         ldg2, c_a, ldg1, c_a, st);
        launch_gemm(tx, c_a, (const bf16*)w[4], nullptr, act, c_ff, rows,
                    c_ff, c_a, 1, st);
        launch_gemm(act, c_ff, (const bf16*)w[5], nullptr, o2, c_a, rows, c_a,
                    c_ff, 0, st);
        gate_p = G2 + blk + b * c_a;
        delta_p = o2;
        rm = MC;
    }
    launch_res_adaln(A, gate_p, delta_p, rm, nullptr, nullptr, nullptr, nullptr,
                     rows, c_a, e, c_a, ldg2, c_a, 0, c_a, st);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("res_adaln", &res_adaln, "fused residual + AdaLN");
    m.def("gather_adaln", &gather_adaln, "fused gather + AdaLN");
    m.def("attn", &attn, "fused gated attention with pair bias");
    m.def("lnz", &lnz, "bf16 LayerNorm with fp32 statistics");
    m.def("blkidx", &blkidx, "cross-stack block indices / mask / mask bias");
    m.def("gemm", &gemm, "bf16 mma GEMM with optional bias / SwiGLU epilogue");
    m.def("run_self", &run_self, "whole self-attention stack, one crossing");
    m.def("run_cross", &run_cross, "whole cross-attention stack, one crossing");
}

