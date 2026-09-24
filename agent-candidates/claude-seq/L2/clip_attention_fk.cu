// Fused CLIP self-attention (fp32 in, TF32 on the tensor cores).
//
// The captured problem -- B=1, S=77, D=768, H=12, head_dim=64, fp32 -- is nowhere
// near compute bound: 396 MFLOP is a couple of microseconds of B200 tensor-core
// work against ~100-150 us for the baseline's eleven eager ops. Practically all of
// the baseline is per-op overhead, so the first job is to get the op count down,
// and the second is to keep what is left off the machine's latency limits.
//
// Structure. One cooperative kernel runs three stages separated by grid barriers:
//
//   1. q/k/v in a single GEMM (M=77, N=3*768, K=768) whose epilogue writes straight
//      into the [3, H, S, 64] layout attention wants, so three projections, three
//      ``view``s and three ``transpose``s cost nothing extra.
//   2. Attention -- QK^T, the scale, the mask add, the softmax and PV -- one block
//      per (head, 16-row band), its warps splitting the keys.
//   3. The output projection (M=77, N=768, K=768).
//
// One launch rather than three: a kernel launch costs ~3.8 us of *host* time at
// these shared-memory sizes and ~2.0 us of GPU-side gap, where a ``grid.sync()``
// costs ~1.1 us and no host time. Both matter -- the GPU work is ~25 us, and the
// scorer's L2 flush only just covers the host side, so on a quiet machine the host
// is the binding constraint.
//
// Numerics. ``torch.backends.cuda.matmul.fp32_precision`` is 'tf32' here, so every
// reference matmul rounds both operands to TF32 and accumulates in fp32; these
// kernels do the same, which leaves only accumulation order as a difference.
// Computing in full fp32 instead would be *more* accurate than the reference and
// still miss the comparison's 1e-5/1e-3 bound, because the reference's own TF32
// error is 20x that bound. The *tie* rule matters too, and it is not the obvious
// one -- see ``to_tf32``.
//
// Shapes, from four microbenchmarks of this GPU:
//
// * ``mma.sync.m16n8k8.f32.tf32`` peaks at ~85 TMAC/s, but only with >= 8 live
//   accumulator tiles per warp; with one or two it stalls on mma latency and loses
//   2-3x. (tcgen05 would go faster in principle, but at 77 rows its tile is >90%
//   pad, and the TMA/TMEM setup does not pay for itself in a 25 us op.)
// * Shared-memory fragment loads run at ~128 B/cycle/SM, so a warp tile has to
//   amortize (MT*16 + NT*8) operand floats over MT*NT*1024 MACs to stay off that.
// * An L2-resident stream reaches ~20 TB/s only with >= 16 loads in flight per
//   thread; with four it manages 10. So memory-level parallelism, not footprint, is
//   what bounds the staging.
// * A B-fragment read straight from global memory touches eight 32-byte sectors,
//   and at these block counts there are too few warps per scheduler to hide that:
//   an earlier attention kernel spent 11.4 cycles per issued instruction stalled on
//   long_scoreboard. Everything an inner loop touches is staged through shared
//   memory in one bulk pass, with every load issued before the first dependent
//   store.
//
// The two GEMM block shapes (48x32x256 for N=3D, 32x16x256 for N=D) and the
// attention warp count came out of the sweeps in ``dev/``. Both GEMM stages split K
// across the warps rather than N: that keeps the warp tile -- and so the accumulator
// count -- at its maximum for a given block tile, which has to stay small because
// 77 rows of output cannot otherwise fill 148 SMs.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace {

constexpr int kHeadDim = 64;
constexpr float kNegBig = -1e30f;  // "masked out"; never -inf, so exp() stays finite

// fp32 -> TF32, round to nearest with ties to even. ``mma.sync ... .tf32`` ignores
// the low 13 mantissa bits (it truncates), and truncation is *biased*, which
// drifts from the reference by several times the allowed tolerance.
//
// The tie rule matters, and it is not the obvious one. Probing cuBLAS with a
// one-term dot product (x[m,0]*W[n,0], everything else zero, so the result is
// exactly the rounded operand) shows it rounds ties to *even*: 0x3F801000 ->
// 0x3F800000 but 0x3F803000 -> 0x3F804000. Ties-away instead leaves 1.5% of the
// projection's outputs off by a TF32 ulp, which -- because the block's final
// output is ~0.04 and the comparison's atol is 1e-5 -- is enough on its own to
// fail the 99%-of-elements check. With ties-to-even the projection matches cuBLAS
// to 5e-6 (17x closer) and only fp32 accumulation order is left.
//
// ``cvt.rn.satfinite.tf32.f32`` does this in one instruction on sm_90+ and agrees
// bit-for-bit with the five-op emulation over 2^22 normals; the emulation was ~30%
// of all instructions the GEMM issued.
__device__ __forceinline__ unsigned to_tf32(float x) {
  unsigned r;
  asm("cvt.rn.satfinite.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
  return r;
}

// D[16x8] += A[16x8] * B[8x8], one warp, TF32 in / fp32 accumulate.
//
// Per-lane fragment layout (g = lane/4, t = lane%4):
//   a0=A[g][t]   a1=A[g+8][t]   a2=A[g][t+4]   a3=A[g+8][t+4]
//   b0=B[t][g]   b1=B[t+4][g]
//   d0=D[g][2t]  d1=D[g][2t+1]  d2=D[g+8][2t]  d3=D[g+8][2t+1]
__device__ __forceinline__ void mma_16x8x8(float *d, const unsigned *a, const unsigned *b) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ---------------------------------------------------------------------------
// GEMM:  Out[m, n] = sum_k X[m, k] * W[n, k] + bias[n]
//
// A block owns the BM x BN tile at (m0, n0); every warp computes the *whole* tile
// over a slice of K, so the warp tile is MT x NT = (BM/16) x (BN/8) -- the largest
// the block tile allows, which is what keeps the mma issue rate up at a block tile
// small enough for 77 rows of output to fill 148 SMs -- and the NWARP partial sums
// are reduced once at the end, through the same shared memory the operands used.
// The K extent of the staging buffer is padded by 4 so the 8 rows x 4 lanes an mma
// fragment reads land in 32 distinct banks.
//
// SPLIT_QKV routes the epilogue into the [3, H, S, 64] q/k/v layout: column n maps
// to (n / D, (n % D) / 64, n % 64). BN divides 64, so one (projection, head) pair
// covers the whole tile and the stores stay contiguous.
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int NWARP, bool SPLIT_QKV, bool ROUND_A, bool ROUND_OUT>
__device__ __forceinline__ void gemm_tile(
    const float *__restrict__ X, const float *__restrict__ W,
    const float *__restrict__ bias, float *__restrict__ Out, int M, int K, int N,
    int num_heads, int m0, int n0, unsigned *smem) {
  constexpr int MT = BM / 16;  // mma row tiles == warp row tiles
  constexpr int NT = BN / 8;   // mma column tiles == warp column tiles
  constexpr int KP = BK + 4;   // padded shared K extent (bank skew)
  constexpr int PERW = (BK / 8) / NWARP;  // mma k-steps per warp per staged block
  constexpr int NTHREAD = NWARP * 32;
  constexpr int BK4 = BK / 4;
  constexpr int NA4 = BM * BK4 / NTHREAD;  // float4 staged per thread, A
  constexpr int NB4 = BN * BK4 / NTHREAD;  // ... and B
  constexpr int MSTEP = NTHREAD / BK4;     // rows between a thread's staging slots
  static_assert(BM % 16 == 0 && BN % 8 == 0, "tile must be a whole number of mma tiles");
  static_assert(PERW * NWARP * 8 == BK, "BK must split evenly over the warps");
  static_assert(NTHREAD % BK4 == 0 && NA4 * MSTEP == BM && NB4 * MSTEP == BN,
                "staging must tile the block exactly");
  static_assert(!SPLIT_QKV || (BN <= kHeadDim && kHeadDim % BN == 0),
                "the q/k/v epilogue needs BN to divide the head dim");

  unsigned *const As = smem;                            // [BM][KP]
  unsigned *const Bs = smem + BM * KP;                  // [BN][KP]
  float *const red = reinterpret_cast<float *>(smem);    // [NWARP][BM][BN], later

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;

  const int K4 = K >> 2;

  float acc[MT][NT][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

  // Everything address-shaped is resolved once, before the K loop, and the NA4
  // staging slots of one thread are a fixed row stride apart -- so one base pointer
  // plus a compile-time multiple covers them all, instead of NA4 live 64-bit
  // pointers. Leaving the index arithmetic in the loop made integer/address
  // instructions outnumber the mmas ~15:1 in the SASS; keeping a pointer per slot
  // instead pushed the register count to 236 and squeezed the scheduler.
  const int mb = tid / BK4, kb0 = tid - mb * BK4;
  const float4 *abase = reinterpret_cast<const float4 *>(X) + (long)(m0 + mb) * K4 + kb0;
  const float4 *bbase = reinterpret_cast<const float4 *>(W) + (long)(n0 + mb) * K4 + kb0;
  unsigned *const ashb = As + mb * KP + kb0 * 4;
  unsigned *const bshb = Bs + mb * KP + kb0 * 4;
  bool alive[NA4];
#pragma unroll
  for (int u = 0; u < NA4; ++u) alive[u] = (m0 + mb + u * MSTEP) < M;
  const unsigned *ap[MT];
  const unsigned *bp[NT];
#pragma unroll
  for (int i = 0; i < MT; ++i) ap[i] = As + (i * 16 + g) * KP + warp * PERW * 8;
#pragma unroll
  for (int j = 0; j < NT; ++j) bp[j] = Bs + (j * 8 + g) * KP + warp * PERW * 8;

  // The K loop keeps the next block's global loads in flight while it works on the
  // staged one.
  float4 ra[NA4], rb[NB4];
#pragma unroll
  for (int u = 0; u < NA4; ++u)
    ra[u] = alive[u] ? abase[(long)u * MSTEP * K4] : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
  for (int u = 0; u < NB4; ++u) rb[u] = bbase[(long)u * MSTEP * K4];

  const int nkb = K / BK;
  for (int kb = 0; kb < nkb; ++kb) {
    __syncthreads();
#pragma unroll
    for (int u = 0; u < NA4; ++u) {
      unsigned *d = ashb + u * MSTEP * KP;
      if (ROUND_A) {
        d[0] = to_tf32(ra[u].x);
        d[1] = to_tf32(ra[u].y);
        d[2] = to_tf32(ra[u].z);
        d[3] = to_tf32(ra[u].w);
      } else {
        *reinterpret_cast<float4 *>(d) = ra[u];
      }
    }
    // W arrives already rounded to TF32 (done once, when the weights are packed).
#pragma unroll
    for (int u = 0; u < NB4; ++u)
      *reinterpret_cast<float4 *>(bshb + u * MSTEP * KP) = rb[u];
    __syncthreads();
    if (kb + 1 < nkb) {
      abase += BK4;
      bbase += BK4;
#pragma unroll
      for (int u = 0; u < NA4; ++u)
        ra[u] = alive[u] ? abase[(long)u * MSTEP * K4] : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int u = 0; u < NB4; ++u) rb[u] = bbase[(long)u * MSTEP * K4];
    }

#pragma unroll
    for (int p = 0; p < PERW; ++p) {
      const int ks = p * 8;
      unsigned a[MT][4], b[NT][2];
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        a[i][0] = ap[i][ks + t];
        a[i][2] = ap[i][ks + t + 4];
        a[i][1] = ap[i][ks + 8 * KP + t];
        a[i][3] = ap[i][ks + 8 * KP + t + 4];
      }
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        b[j][0] = bp[j][ks + t];
        b[j][1] = bp[j][ks + t + 4];
      }
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) mma_16x8x8(acc[i][j], a[i], b[j]);
    }
  }

  // ---- reduce the K split across warps, then epilogue ---------------------
  __syncthreads();
  {
    float *dst = red + warp * (BM * BN);
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          float2 v;
          v.x = acc[i][j][2 * h];
          v.y = acc[i][j][2 * h + 1];
          *reinterpret_cast<float2 *>(dst + (i * 16 + g + 8 * h) * BN + j * 8 + 2 * t) = v;
        }
  }
  __syncthreads();

  // BN divides 64, so the whole tile shares one (projection, head): resolve the
  // q/k/v destination once instead of dividing by a runtime D per element.
  long obase;
  long ostride;
  if (SPLIT_QKV) {
    const int D = num_heads * kHeadDim;
    const int proj = n0 / D;
    const int rem = n0 - proj * D;
    const int head = rem >> 6;
    obase = (long)(proj * num_heads + head) * M * kHeadDim + (rem & (kHeadDim - 1));
    ostride = kHeadDim;
  } else {
    obase = n0;
    ostride = N;
  }
  for (int idx = tid * 2; idx < BM * BN; idx += NTHREAD * 2) {
    const int row = idx / BN;
    if (m0 + row >= M) continue;
    const int col = idx - row * BN;
    float2 v = *reinterpret_cast<const float2 *>(red + idx);
#pragma unroll
    for (int w = 1; w < NWARP; ++w) {
      const float2 o = *reinterpret_cast<const float2 *>(red + w * (BM * BN) + idx);
      v.x += o.x;
      v.y += o.y;
    }
    const float2 bv = *reinterpret_cast<const float2 *>(bias + n0 + col);
    v.x += bv.x;
    v.y += bv.y;
    if (ROUND_OUT) {
      // q/k/v only ever feed an mma, so storing them pre-rounded saves the
      // attention kernel a conversion per operand register.
      v.x = __uint_as_float(to_tf32(v.x));
      v.y = __uint_as_float(to_tf32(v.y));
    }
    *reinterpret_cast<float2 *>(Out + obase + (long)(m0 + row) * ostride + col) = v;
  }
}

template <int BM, int BN, int BK, int NWARP, bool SPLIT_QKV, bool ROUND_A, bool ROUND_OUT>
__global__ __launch_bounds__(NWARP * 32) void gemm_kernel(
    const float *__restrict__ X, const float *__restrict__ W,
    const float *__restrict__ bias, float *__restrict__ Out, int M, int K, int N,
    int num_heads) {
  extern __shared__ unsigned smem[];
  const int nt = N / BN;
  const int tile = blockIdx.x;
  gemm_tile<BM, BN, BK, NWARP, SPLIT_QKV, ROUND_A, ROUND_OUT>(X, W, bias, Out, M, K, N, num_heads,
                                          (tile / nt) * BM, (tile - (tile / nt) * nt) * BN,
                                          smem);
}

// ---------------------------------------------------------------------------
// Attention.
//
//   logits = (Q K^T) * scale + mask  ->  softmax  ->  O = P V
//
// One block owns a 16-row band of one head; its NWARP warps split the *keys*, and
// the logits pass through shared memory so the softmax still sees whole rows.
// That matters twice over:
//
// * Parallelism. There are only ceil(S/16) * H = 60 (band, head) pairs at this
//   shape, so one warp per pair leaves 60 warps for 148 SMs -- one per SM
//   scheduler, with nothing to switch to while an mma or an LDS is in flight.
//   Splitting the keys NWARP ways multiplies the warp count and cuts each warp's
//   register footprint (KT/NWARP logit tiles instead of KT), which is what lets
//   the compiler keep several mmas in flight.
// * Numerics. Because the softmax reads complete rows out of shared memory, P is
//   exactly the reference's ``softmax(x)`` rounded to TF32 -- no flash-style
//   rescaling, whose independent rounding would cost ~2.4e-4 of relative error on
//   an output the comparison only allows 1e-5 of absolute slack on.
//
// It also removes work: reading the A operand of the PV mma straight out of shared
// memory replaces the lane permutation (lane t holds logit columns {2t, 2t+1} but
// needs {t, t+4}) that a register-resident P would need.
//
// Staging K/V is the other half. An mma B-fragment wants four consecutive floats
// from each of eight rows, which straight out of global memory is eight 32-byte
// sectors per instruction; with ~400 such loads per warp an earlier version spent
// 11.4 cycles per issued instruction stalled on long_scoreboard (measured), making
// attention the most expensive of the three kernels despite being 5% of the FLOPs.
// One bulk coalesced pass fixes that, and the fragment reads become conflict-free
// LDS: K is padded to 68 floats per row and V to 72, which is what makes the
// (8 rows x 4 lanes) and (4 rows x 8 lanes) patterns land in 32 distinct banks.
//
// q/k/v arrive already rounded to TF32 from the projection kernel, so nothing here
// converts an operand; they are read as raw mma words.
// ---------------------------------------------------------------------------
template <int KT, int NWARP>
__device__ __forceinline__ void attn_unit(
    const float *__restrict__ qkv, const float *__restrict__ mask,
    float *__restrict__ O, int S, int num_heads, int head, int band, float scale,
    long mask_hstride, unsigned *sh) {
  constexpr int SP = KT * 8;                    // padded key count
  constexpr int KPAD = kHeadDim + 4;            // 8 rows x 4 lanes -> 32 banks
  constexpr int VPAD = kHeadDim + 8;            // 4 rows x 8 lanes -> 32 banks
  constexpr int QPAD = kHeadDim + 4;
  constexpr int LPAD = ((SP + 27) & ~31) + 4;   // >= SP, == 4 (mod 32)
  constexpr int NTHREAD = NWARP * 32;
  constexpr int HD4 = kHeadDim / 4;
  constexpr int NTO = kHeadDim / 8;             // output column tiles
  constexpr int JW = (KT + NWARP - 1) / NWARP;  // key tiles per warp

  unsigned *const Ks = sh;                  // [SP][KPAD]
  unsigned *const Vs = Ks + SP * KPAD;      // [SP][VPAD]
  unsigned *const Qs = Vs + SP * VPAD;      // [16][QPAD]
  unsigned *const Lg = Qs + 16 * QPAD;      // [16][LPAD]  logits, then TF32 P
  float *const Ored = reinterpret_cast<float *>(Lg + 16 * LPAD);  // [NWARP][16][64]

  const int wid = (int)(threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int t = lane & 3;

  const long hoff = (long)head * S * kHeadDim;
  const long pstride = (long)num_heads * S * kHeadDim;
  const unsigned *__restrict__ qkvu = reinterpret_cast<const unsigned *>(qkv);
  const uint4 *__restrict__ Qg = reinterpret_cast<const uint4 *>(qkvu + hoff);
  const uint4 *__restrict__ Kg = reinterpret_cast<const uint4 *>(qkvu + pstride + hoff);
  const uint4 *__restrict__ Vg = reinterpret_cast<const uint4 *>(qkvu + 2 * pstride + hoff);

  // ---- stage Q (this band) and the head's K and V ---------------------------
  // Every load is issued before the first dependent store, so the block pays one
  // memory latency rather than one per staged vector -- with only two warps per
  // scheduler there is nothing else to hide them behind.
  constexpr int NKV = (SP * HD4 + NTHREAD - 1) / NTHREAD;
  constexpr int NQ = (16 * HD4 + NTHREAD - 1) / NTHREAD;
  const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
  uint4 kr[NKV], vr[NKV], qr[NQ];
#pragma unroll
  for (int u = 0; u < NKV; ++u) {
    const int idx = threadIdx.x + u * NTHREAD;
    const int n = idx / HD4, c4 = idx - n * HD4;
    const bool live = idx < SP * HD4 && n < S;
    kr[u] = live ? Kg[n * HD4 + c4] : z4;
    vr[u] = live ? Vg[n * HD4 + c4] : z4;
  }
#pragma unroll
  for (int u = 0; u < NQ; ++u) {
    const int idx = threadIdx.x + u * NTHREAD;
    const int r = idx / HD4, c4 = idx - r * HD4;
    const int row = band * 16 + r;
    qr[u] = (idx < 16 * HD4 && row < S) ? Qg[(long)row * HD4 + c4] : z4;
  }
#pragma unroll
  for (int u = 0; u < NKV; ++u) {
    const int idx = threadIdx.x + u * NTHREAD;
    if (idx >= SP * HD4) break;
    const int n = idx / HD4, c4 = idx - n * HD4;
    *reinterpret_cast<uint4 *>(Ks + n * KPAD + c4 * 4) = kr[u];
    *reinterpret_cast<uint4 *>(Vs + n * VPAD + c4 * 4) = vr[u];
  }
#pragma unroll
  for (int u = 0; u < NQ; ++u) {
    const int idx = threadIdx.x + u * NTHREAD;
    if (idx >= 16 * HD4) break;
    const int r = idx / HD4, c4 = idx - r * HD4;
    *reinterpret_cast<uint4 *>(Qs + r * QPAD + c4 * 4) = qr[u];
  }
  __syncthreads();

  // ---- phase A: this warp's slice of the logits, into shared --------------
  {
    // Mask first: independent of the mmas, so its latency hides under them.
    float mk[JW][2][2];
#pragma unroll
    for (int jj = 0; jj < JW; ++jj) {
      const int j = jj * NWARP + wid;
      const int c = j * 8 + 2 * t;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int row = band * 16 + g + 8 * h;
        mk[jj][h][0] = mk[jj][h][1] = 0.f;
        if (mask != nullptr && j < KT && row < S) {
          const float *mp = mask + head * mask_hstride + (long)row * S + c;
          if (c < S) mk[jj][h][0] = mp[0];
          if (c + 1 < S) mk[jj][h][1] = mp[1];
        }
      }
    }
    unsigned qa[kHeadDim / 8][4];
#pragma unroll
    for (int ks = 0; ks < kHeadDim / 8; ++ks) {
      const unsigned *q0 = Qs + g * QPAD + ks * 8;
      qa[ks][0] = q0[t];
      qa[ks][2] = q0[t + 4];
      qa[ks][1] = q0[8 * QPAD + t];
      qa[ks][3] = q0[8 * QPAD + t + 4];
    }
    float s[JW][4];
#pragma unroll
    for (int jj = 0; jj < JW; ++jj) s[jj][0] = s[jj][1] = s[jj][2] = s[jj][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < kHeadDim / 8; ++ks) {
#pragma unroll
      for (int jj = 0; jj < JW; ++jj) {
        const int j = jj * NWARP + wid;
        if (j >= KT) continue;
        const unsigned *kp = Ks + (j * 8 + g) * KPAD + ks * 8;
        unsigned b[2] = {kp[t], kp[t + 4]};
        mma_16x8x8(s[jj], qa[ks], b);
      }
    }
#pragma unroll
    for (int jj = 0; jj < JW; ++jj) {
      const int j = jj * NWARP + wid;
      if (j >= KT) continue;
      const int c = j * 8 + 2 * t;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int row = band * 16 + g + 8 * h;
        const float v0 = s[jj][2 * h] * scale + mk[jj][h][0];
        const float v1 = s[jj][2 * h + 1] * scale + mk[jj][h][1];
        float2 o;
        o.x = (row < S && c < S) ? v0 : kNegBig;
        o.y = (row < S && c + 1 < S) ? v1 : kNegBig;
        *reinterpret_cast<float2 *>(
            reinterpret_cast<float *>(Lg) + (g + 8 * h) * LPAD + c) = o;
      }
    }
  }
  __syncthreads();

  // ---- phase B: softmax over whole rows, in place, result left as TF32 ----
  {
    float *Lf = reinterpret_cast<float *>(Lg);
    constexpr int ROWS_PER_WARP = (16 + NWARP - 1) / NWARP;
#pragma unroll
    for (int rr = 0; rr < ROWS_PER_WARP; ++rr) {
      const int r = wid * ROWS_PER_WARP + rr;
      if (r >= 16) break;
      float *row = Lf + r * LPAD;
      float v[(SP + 31) / 32];
      float m = kNegBig;
#pragma unroll
      for (int u = 0; u < (SP + 31) / 32; ++u) {
        const int c = u * 32 + lane;
        v[u] = c < SP ? row[c] : kNegBig;
        m = fmaxf(m, v[u]);
      }
#pragma unroll
      for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
      float sum = 0.f;
#pragma unroll
      for (int u = 0; u < (SP + 31) / 32; ++u) {
        v[u] = __expf(v[u] - m);
        sum += v[u];
      }
#pragma unroll
      for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
      const float inv = 1.f / sum;
#pragma unroll
      for (int u = 0; u < (SP + 31) / 32; ++u) {
        const int c = u * 32 + lane;
        if (c < SP) Lg[r * LPAD + c] = to_tf32(v[u] * inv);
      }
    }
  }
  __syncthreads();

  // ---- phase C: O = P V over this warp's keys, then reduce over warps -----
  {
    float o[NTO][4];
#pragma unroll
    for (int jj = 0; jj < NTO; ++jj)
#pragma unroll
      for (int e = 0; e < 4; ++e) o[jj][e] = 0.f;
#pragma unroll
    for (int jj = 0; jj < JW; ++jj) {
      const int j = jj * NWARP + wid;
      if (j >= KT) continue;
      const unsigned *lp = Lg + g * LPAD + j * 8;
      unsigned pa[4] = {lp[t], lp[8 * LPAD + t], lp[t + 4], lp[8 * LPAD + t + 4]};
      const unsigned *v0 = Vs + (j * 8 + t) * VPAD;
#pragma unroll
      for (int nt = 0; nt < NTO; ++nt) {
        const int col = nt * 8 + g;
        unsigned b[2] = {v0[col], v0[4 * VPAD + col]};
        mma_16x8x8(o[nt], pa, b);
      }
    }
    float *dst = Ored + wid * 16 * kHeadDim;
#pragma unroll
    for (int nt = 0; nt < NTO; ++nt)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float2 v;
        v.x = o[nt][2 * h];
        v.y = o[nt][2 * h + 1];
        *reinterpret_cast<float2 *>(dst + (g + 8 * h) * kHeadDim + nt * 8 + 2 * t) = v;
      }
  }
  __syncthreads();

  // ---- store to [S, H*64], pre-rounded: the only consumer is the output
  //      projection's mma ---------------------------------------------------
  {
    const long ld = (long)num_heads * kHeadDim;
    for (int idx = threadIdx.x * 2; idx < 16 * kHeadDim; idx += NTHREAD * 2) {
      const int r = idx >> 6;
      const int row = band * 16 + r;
      if (row >= S) continue;
      float2 v = *reinterpret_cast<const float2 *>(Ored + idx);
#pragma unroll
      for (int w = 1; w < NWARP; ++w) {
        const float2 u = *reinterpret_cast<const float2 *>(Ored + w * 16 * kHeadDim + idx);
        v.x += u.x;
        v.y += u.y;
      }
      v.x = __uint_as_float(to_tf32(v.x));
      v.y = __uint_as_float(to_tf32(v.y));
      *reinterpret_cast<float2 *>(O + row * ld + head * kHeadDim + (idx & (kHeadDim - 1))) = v;
    }
  }
}

template <int KT, int NWARP>
__global__ __launch_bounds__(NWARP * 32) void attn_kernel(
    const float *__restrict__ qkv, const float *__restrict__ mask,
    float *__restrict__ O, int S, int num_heads, int num_bands, float scale,
    long mask_hstride) {
  extern __shared__ unsigned sh[];
  const int head = blockIdx.x / num_bands;
  attn_unit<KT, NWARP>(qkv, mask, O, S, num_heads, head,
                       blockIdx.x - head * num_bands, scale, mask_hstride, sh);
}

// ---------------------------------------------------------------------------
// All three stages in one cooperative kernel.
//
// A kernel launch costs ~3.8 us of *host* time here (large dynamic shared memory
// makes the driver reconfigure the carveout) and ~2.0 us of GPU-side gap; a
// grid-wide barrier costs ~1.1 us and no host time at all. At ~25 us of real work
// the three launches were a fifth of the wall clock on the GPU and, once the
// machine is quiet enough that the harness' L2 flush stops covering for it, the
// binding constraint on the host side too. Folding the stages into one launch with
// two ``grid.sync()``s removes both.
//
// The stages keep their own block shapes; what they have to agree on is the block
// size (8 warps) and the grid, which is why each stage walks its tiles in a
// persistent loop. The grid is clamped to what actually fits concurrently --
// cooperative launch requires every block resident -- and shared memory is sized
// for the hungriest stage and reused (``grid.sync()`` is a grid-wide barrier, so it
// also orders the block's own shared accesses between stages).
// ---------------------------------------------------------------------------
// One struct rather than fifteen scalars: the driver copies the parameter buffer
// per argument, and at ~0.2 us apiece that was ~3 us of host time on an op whose
// host side is close to binding.
struct FusedArgs {
  const float *x, *mask, *w_qkv, *b_qkv, *w_out, *b_out;
  float *qkv, *attn_out, *out;
  int S, D, num_heads, num_bands;
  float scale;
  long mask_hstride;
};

template <int KT, int NWARP, int QM, int QN, int QK, int OM, int ON, int OK>
__global__ __launch_bounds__(NWARP * 32) void fused_kernel(const FusedArgs a) {
  const float *__restrict__ x = a.x;
  const float *__restrict__ mask = a.mask;
  const int S = a.S, D = a.D, num_heads = a.num_heads, num_bands = a.num_bands;
  const float scale = a.scale;
  const long mask_hstride = a.mask_hstride;
  float *__restrict__ qkv = a.qkv;
  float *__restrict__ attn_out = a.attn_out;
  extern __shared__ unsigned sh[];
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();

  const int qn = (3 * D) / QN;
  for (int tile = blockIdx.x; tile < ((S + QM - 1) / QM) * qn; tile += gridDim.x)
    gemm_tile<QM, QN, QK, NWARP, true, true, true>(x, a.w_qkv, a.b_qkv, qkv, S, D, 3 * D,
                                                   num_heads, (tile / qn) * QM,
                                                   (tile % qn) * QN, sh);
  grid.sync();
  __syncthreads();

  for (int unit = blockIdx.x; unit < num_heads * num_bands; unit += gridDim.x)
    attn_unit<KT, NWARP>(qkv, mask, attn_out, S, num_heads, unit / num_bands,
                         unit % num_bands, scale, mask_hstride, sh);
  grid.sync();
  __syncthreads();

  const int on = D / ON;
  for (int tile = blockIdx.x; tile < ((S + OM - 1) / OM) * on; tile += gridDim.x)
    gemm_tile<OM, ON, OK, NWARP, false, false, false>(attn_out, a.w_out, a.b_out, a.out,
                                                      S, D, D, num_heads,
                                                      (tile / on) * OM,
                                                      (tile % on) * ON, sh);
}

template <int KT, int NWARP>
static int attn_smem() {
  constexpr int SP = KT * 8;
  constexpr int LPAD = ((SP + 27) & ~31) + 4;
  return 4 * (SP * (kHeadDim + 4) + SP * (kHeadDim + 8) + 16 * (kHeadDim + 4) +
              16 * LPAD + NWARP * 16 * kHeadDim);
}

// ---------------------------------------------------------------------------
// Launch helpers. The GEMM block shape is picked from a small table so that both
// call sites, which have very different N, can each use the tile that keeps the
// most SMs busy; ``cfg`` selects an entry (see kGemmCfgs).
// ---------------------------------------------------------------------------
struct Shape {
  int bm, bn, bk, nwarp;
};

// (BM, BN, BK, NWARP), swept against the scorer's own timer on this GPU. Valid
// entries satisfy: BM % 16 == 0, BN in {8,16,32,64} (the q/k/v epilogue needs BN to
// divide the head dim), (BK/8) % NWARP == 0, BK % 4 == 0 with NWARP*32 % (BK/4) == 0,
// and BM, BN both multiples of (NWARP*32)/(BK/4).
//
// Entry 0 is the q/k/v projection's shape (N = 3D: 144 tiles of 48x32 at D=768) and
// entry 1 the output projection's (N = D: 144 tiles of 32x16). Everything from
// 80x16 through 48x32 measured within noise of each other on the 2304-column GEMM;
// the narrow-N one is a clear 1.5x on the 768-column GEMM, where a wide tile leaves
// most of the machine idle.
constexpr Shape kGemmCfgs[] = {
    {48, 32, 256, 8},  // 0: q/k/v projection (N = 3D)
    {32, 16, 256, 8},  // 1: output projection (N = D)
    {48, 32, 128, 8},  // 2: same pair for a K that 256 does not divide
    {32, 16, 128, 8},  // 3
    {80, 16, 64, 8},   // 4: smallest BK, for a K that 128 does not divide
    {16, 16, 128, 8},  // 5
};
constexpr int kNumGemmCfgs = sizeof(kGemmCfgs) / sizeof(Shape);

template <int BM, int BN, int BK, int NWARP>
constexpr int smem_bytes() {
  constexpr int ops = (BM + BN) * (BK + 4);
  constexpr int red = NWARP * BM * BN;
  return 4 * (ops > red ? ops : red);
}

template <int BM, int BN, int BK, int NWARP, bool SPLIT>
static void launch_gemm(const float *X, const float *W, const float *bias, float *Out,
                        int M, int K, int N, int H, cudaStream_t stream) {
  constexpr int SM = smem_bytes<BM, BN, BK, NWARP>();
  auto kern = gemm_kernel<BM, BN, BK, NWARP, SPLIT, SPLIT, SPLIT>;
  if (SM > 48 * 1024) {
    static bool once = [&] {
      cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SM);
      return true;
    }();
    (void)once;
  }
  const int tiles = ((M + BM - 1) / BM) * (N / BN);
  kern<<<tiles, NWARP * 32, SM, stream>>>(X, W, bias, Out, M, K, N, H);
}

// The K loop stages BK at a time, so BK has to divide K; fall back to the first
// entry that does (the table always holds a BK=64 one).
static int pick_cfg(int cfg, int K) {
  if (cfg >= 0 && cfg < kNumGemmCfgs && K % kGemmCfgs[cfg].bk == 0) return cfg;
  for (int i = 0; i < kNumGemmCfgs; ++i)
    if (K % kGemmCfgs[i].bk == 0) return i;
  return -1;
}

template <bool SPLIT>
static void dispatch_gemm(int cfg, const float *X, const float *W, const float *bias,
                          float *Out, int M, int K, int N, int H, cudaStream_t stream) {
#define FK_CASE(i)                                                              \
  case i:                                                                       \
    launch_gemm<kGemmCfgs[i].bm, kGemmCfgs[i].bn, kGemmCfgs[i].bk,               \
                kGemmCfgs[i].nwarp, SPLIT>(X, W, bias, Out, M, K, N, H, stream); \
    return;
  switch (cfg) {
    FK_CASE(0) FK_CASE(1) FK_CASE(2)
    FK_CASE(3) FK_CASE(4) FK_CASE(5)
    default:
      return;
  }
#undef FK_CASE
}

template <int KT, int NW>
static void launch_attn_t(cudaStream_t stream, const float *qkv, const float *mask,
                          float *O, int S, int H, int bands, float scale, long mhs) {
  const int sm = attn_smem<KT, NW>();
  auto kern = attn_kernel<KT, NW>;
  if (sm > 48 * 1024) {
    static bool once = [&] {
      cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
      return true;
    }();
    (void)once;
  }
  const int blocks = H * bands;
  kern<<<blocks, NW * 32, sm, stream>>>(qkv, mask, O, S, H, bands, scale, mhs);
}

template <int KT, int QM, int QN, int QK, int OM, int ON, int OK, int NW>
static bool launch_fused(cudaStream_t stream, const float *x, const float *mask,
                         const float *w_qkv, const float *b_qkv, const float *w_out,
                         const float *b_out, float *qkv, float *attn_out, float *out,
                         int S, int D, int H, int bands, float scale, long mhs) {
  auto kern = fused_kernel<KT, NW, QM, QN, QK, OM, ON, OK>;
  constexpr int SMQ = smem_bytes<QM, QN, QK, NW>();
  constexpr int SMO = smem_bytes<OM, ON, OK, NW>();
  const int sm = max(max(SMQ, SMO), attn_smem<KT, NW>());

  // Resolved once: the opt-in for >48 KB of dynamic shared memory, and how many
  // blocks can actually be co-resident (a cooperative launch fails otherwise).
  static int grid_cap = [&] {
    if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, sm) !=
        cudaSuccess)
      return 0;
    int per_sm = 0, nsm = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, NW * 32, sm) !=
            cudaSuccess ||
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess)
      return 0;
    return per_sm * nsm;
  }();
  if (grid_cap <= 0) return false;

  const int want = max(((S + QM - 1) / QM) * ((3 * D) / QN),
                       ((S + OM - 1) / OM) * (D / ON));
  const int blocks = min(want, grid_cap);
  FusedArgs a{x,   mask,     w_qkv, b_qkv, w_out, b_out, qkv, attn_out,
              out, S,        D,     H,     bands, scale, mhs};
  void *args[] = {(void *)&a};
  return cudaLaunchCooperativeKernel((void *)kern, dim3(blocks), dim3(NW * 32), args,
                                     sm, stream) == cudaSuccess;
}

template <int QM, int QN, int QK, int OM, int ON, int OK, int NW>
static bool dispatch_kt(cudaStream_t stream, const float *x, const float *mask,
                        const float *w_qkv, const float *b_qkv, const float *w_out,
                        const float *b_out, float *qkv, float *attn_out, float *out,
                        int S, int D, int H, int bands, float scale, long mhs) {
  if (D % QK != 0 || D % OK != 0) return false;
  if ((3 * D) % QN != 0 || D % ON != 0) return false;
  const int kt = (S + 7) / 8;
#define FK_KT(N)                                                                  \
  if (kt <= N)                                                                     \
    return launch_fused<N, QM, QN, QK, OM, ON, OK, NW>(                            \
        stream, x, mask, w_qkv, b_qkv, w_out, b_out, qkv, attn_out, out, S, D, H,  \
        bands, scale, mhs);
  FK_KT(4) FK_KT(8) FK_KT(10) FK_KT(16)
#undef FK_KT
  return false;
}

// The fused path's two block shapes. A narrower pair (32x16x128 / 16x16x128), which
// fits two blocks per SM instead of one, measured 29.7 us against 23.6 us for these,
// so the wide q/k/v tile wins despite the lower occupancy. A K that 256 does not
// divide falls back to the three-kernel path and its table.
static bool dispatch_fused(cudaStream_t stream, const float *x, const float *mask,
                           const float *w_qkv, const float *b_qkv, const float *w_out,
                           const float *b_out, float *qkv, float *attn_out, float *out,
                           int S, int D, int H, int bands, float scale, long mhs) {
  return dispatch_kt<48, 32, 256, 32, 16, 256, 8>(stream, x, mask, w_qkv, b_qkv, w_out,
                                                  b_out, qkv, attn_out, out, S, D, H,
                                                  bands, scale, mhs);
}

template <int KT>
static void launch_attn(int nw, cudaStream_t stream, const float *qkv, const float *mask,
                        float *O, int S, int H, int bands, float scale, long mhs) {
  if (nw == 2)
    launch_attn_t<KT, 2>(stream, qkv, mask, O, S, H, bands, scale, mhs);
  else if (nw == 4)
    launch_attn_t<KT, 4>(stream, qkv, mask, O, S, H, bands, scale, mhs);
  else if (nw == 16)
    launch_attn_t<KT, 16>(stream, qkv, mask, O, S, H, bands, scale, mhs);
  else
    launch_attn_t<KT, 8>(stream, qkv, mask, O, S, H, bands, scale, mhs);
}

}  // namespace

// Largest sequence the register-resident logit tile is instantiated for.
constexpr int kMaxSeq = 128;

static bool plain_2d(const at::Tensor &t, long r, long c) {
  return t.dim() == 2 && t.size(0) == r && t.size(1) == c && t.is_contiguous() &&
         t.scalar_type() == at::kFloat;
}

static bool plain_1d(const at::Tensor &t, long n) {
  return t.dim() == 1 && t.size(0) == n && t.stride(0) == 1 &&
         t.scalar_type() == at::kFloat;
}

// ---------------------------------------------------------------------------
// Host entry point: one call, three launches, one scratch allocation.
//
// Every precondition is checked here rather than in Python -- the op runs in
// ~15 us, so a dozen interpreter-level guards per call would be a measurable
// fraction of it. An unsupported input returns nullopt and the module falls back
// to the eager formulation.
// ---------------------------------------------------------------------------
c10::optional<at::Tensor> clip_attention_tuned(
    const at::Tensor &x,  // [1, S, D] fp32
    const c10::optional<at::Tensor> &mask,
    const at::Tensor &w_qkv,  // [3D, D], pre-rounded to TF32
    const at::Tensor &b_qkv,  // [3D]
    const at::Tensor &w_out,  // [D, D], pre-rounded to TF32
    const at::Tensor &b_out,  // [D]
    int64_t num_heads, double scale, int64_t cfg_qkv, int64_t cfg_out,
    int64_t cfg_attn, int64_t phases) {
  if (!(x.is_cuda() && x.dim() == 3 && x.size(0) == 1 && x.is_contiguous() &&
        x.scalar_type() == at::kFloat))
    return c10::nullopt;
  const int S = (int)x.size(1);
  const int D = (int)x.size(2);
  const int H = (int)num_heads;
  // S == 1 is excluded deliberately: at M=1 the reference's projections land on an
  // exact fp32 GEMV rather than a TF32 GEMM -- measured, its error against float64
  // drops from ~2e-4 to 1.6e-7 -- so no TF32 kernel can track it there. From S >= 2
  // the reference is TF32 again and this kernel is as close to float64 as it is
  // (ratio 0.97-1.4 over S = 2..128, D = 448..1024).
  if (S <= 1 || S > kMaxSeq || H <= 0 || D != H * kHeadDim || D % 64 != 0)
    return c10::nullopt;
  if (!(plain_2d(w_qkv, 3l * D, D) && plain_1d(b_qkv, 3l * D) &&
        plain_2d(w_out, D, D) && plain_1d(b_out, D)))
    return c10::nullopt;
  long mhs = 0;
  const float *mp = nullptr;
  if (mask.has_value() && mask->defined()) {
    const at::Tensor &m = *mask;
    if (!(m.is_cuda() && m.dim() == 4 && m.size(0) == 1 && m.size(2) == S &&
          m.size(3) == S && m.stride(3) == 1 && m.stride(2) == S &&
          m.scalar_type() == at::kFloat && (m.size(1) == 1 || m.size(1) == H)))
      return c10::nullopt;
    mp = m.data_ptr<float>();
    mhs = m.size(1) == 1 ? 0l : m.stride(1);
  }

  const int cq = pick_cfg(cfg_qkv < 0 ? 0 : (int)cfg_qkv, D);
  const int co = pick_cfg(cfg_out < 0 ? 1 : (int)cfg_out, D);
  if (cq < 0 || co < 0) return c10::nullopt;

  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();

  // One allocation for q/k/v [3, H, S, 64], the attention output [S, D] and the
  // result: at ~20 us per call a second trip through the caching allocator is
  // measurable. The result is returned as a view of it.
  const long nqkv = (long)3 * H * S * kHeadDim;
  auto scratch = at::empty({nqkv + 2 * (long)S * D}, x.options());
  float *qkv = scratch.data_ptr<float>();
  float *attn_out = qkv + nqkv;
  auto out = scratch.narrow(0, nqkv + (long)S * D, (long)S * D).view({1, S, D});

  const int bands = (S + 15) / 16;
  if (phases == 7 && cfg_qkv < 0 &&
      dispatch_fused(stream, x.data_ptr<float>(), mp, w_qkv.data_ptr<float>(),
                     b_qkv.data_ptr<float>(), w_out.data_ptr<float>(),
                     b_out.data_ptr<float>(), qkv, attn_out, out.data_ptr<float>(), S, D,
                     H, bands, (float)scale, mhs))
    return out;

  if (phases & 1)
    dispatch_gemm<true>(cq, x.data_ptr<float>(), w_qkv.data_ptr<float>(),
                      b_qkv.data_ptr<float>(), qkv, S, D, 3 * D, H, stream);
  if (phases & 2) {
    const int kt = (S + 7) / 8;
    const int nw = (int)cfg_attn;
    if (kt <= 4)
      launch_attn<4>(nw, stream, qkv, mp, attn_out, S, H, bands, (float)scale, mhs);
    else if (kt <= 8)
      launch_attn<8>(nw, stream, qkv, mp, attn_out, S, H, bands, (float)scale, mhs);
    else if (kt <= 10)
      launch_attn<10>(nw, stream, qkv, mp, attn_out, S, H, bands, (float)scale, mhs);
    else
      launch_attn<16>(nw, stream, qkv, mp, attn_out, S, H, bands, (float)scale, mhs);
  }
  if (phases & 4)
    dispatch_gemm<false>(co, attn_out, w_out.data_ptr<float>(),
                       b_out.data_ptr<float>(), out.data_ptr<float>(), S, D, D, H, stream);
  return out;
}

// Production entry point: the tuned one with the swept launch shapes baked in.
c10::optional<at::Tensor> clip_attention(const at::Tensor &x,
                                         const c10::optional<at::Tensor> &mask,
                                         const at::Tensor &w_qkv,
                                         const at::Tensor &b_qkv,
                                         const at::Tensor &w_out,
                                         const at::Tensor &b_out,
                                         int64_t num_heads, double scale) {
  // cfg_qkv < 0 asks for the single-launch cooperative path, with the three-kernel
  // path (shapes 0 and 1) as the fallback when it cannot be used.
  return clip_attention_tuned(x, mask, w_qkv, b_qkv, w_out, b_out, num_heads, scale,
                              -1, -1, 8, 7);
}

#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("clip_attention", &clip_attention,
        "Fused CLIP self-attention (fp32/TF32); None if the input is unsupported");
  // Same op with the launch shapes forced, for the sweeps in dev/.
  m.def("clip_attention_tuned", &clip_attention_tuned,
        "clip_attention with forced launch shapes / phase mask");
  m.def("num_gemm_cfgs", []() { return (int64_t)kNumGemmCfgs; });
}
