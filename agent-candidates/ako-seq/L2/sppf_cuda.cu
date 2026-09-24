// Fused YOLOv10 SPPF for fp16 NCHW [N,256,20,20], k=5.
//
//   t   = silu(W1 x + b1)             (1x1 conv, K=256 -> 128 ch)
//   y5  = maxpool(t, 5,1,2)
//   y9  = maxpool(y5,5,1,2) == maxpool(t, 9,1,4)
//   y13 = maxpool(y9,5,1,2) == maxpool(t,13,1,6)
//   out = silu(W2 [t;y5;y9;y13] + b2)  (1x1 conv, K=512 -> 256 ch)
//
// Three PDL-chained launches issued from ONE pybind call, replacing the
// baseline's ~11 ATen ops.  The concatenation never exists: it is only where
// the k index lands in cv2's reduction.
//
// Both 1x1 convs are implicit GEMMs with m <-> output channel, n <-> spatial
// position, k <-> input channel:
//   * A (weights) is pre-swizzled on the host straight into mma.m16n8k16
//     fragment order, so a warp's whole operand for one k-step is one 16 B load
//     per lane, and all K/16 of those loads are issued *before* the
//     shared-memory staging and the mma loop.  Measured, that ordering alone is
//     worth 30% of cv2: with the load inside the k loop a warp serialises 32
//     ~500-cycle global round trips and at 8 warps/SM nothing is resident to
//     hide them behind.
//   * B (activations) is NCHW, i.e. k-strided, but mma needs both operands
//     k-major.  The whole (K x PP) tile is staged in shared memory in its
//     natural layout and read back with ldmatrix.trans, which does the 8x8
//     transpose for free.  The smem pitch is chosen so (pitch/2) % 32 == 4,
//     which makes each 8-lane phase of an ldmatrix hit 8 distinct bank quads.
//   * m <-> channel (not position) makes the D fragment two *contiguous*
//     positions at one channel, so the NCHW epilogue store is one 32-bit store
//     and bias + SiLU fold into it.  BatchNorm folds into (weight, bias) on the
//     host.
//
// Both convs are bound by how many SMs the tiling can light up, not by
// arithmetic (0.52 GFLOP is ~1us at the 546 TFLOPS mma.sync sustains here) and
// not by total traffic.  Since blocks = ceil(400/PP) * (CO_OUT/CO) * N, the
// position tile is deliberately allowed *not* to divide the 400-position plane:
// the last tile's origin is clamped to PLANE-PP so it overlaps its neighbour
// and recomputes a few positions with bit-identical results, which buys tile
// sizes that are otherwise unreachable.  That is what takes cv2 from 4.83us
// (PP=40, 80 blocks) to 3.87us (PP=24, 136 blocks) -- despite *more* traffic.
//
// The pool cascade is one shared-memory pass over each 20x20 plane with one
// thread per half2 element (columns packed two per lane, so a plane is 200
// half2 and blockDim is (200, planes-per-block) with no idle lanes), all three
// levels straight-line, and __hmax2_nan so the NaN semantics match ATen's
// `val > max || isnan(val)` exactly.  Giving each thread more than one element
// costs more than it saves: 2 elements/thread measured 3.33us and 8 measured
// 8.2us against 2.50us for one.  That is also why the three kernels are *not*
// fused into one launch behind software grid barriers: the convs want 136
// blocks of 8 warps and the pool wants 128 blocks of 25, and forcing the pool
// onto the convs' grid cost 8-11us in the pool phase -- far more than the ~3us of
// launch overhead fusion would have saved.  Measured, best of three attempts:
// one launch 21.5us and cv1+pool merged 21.5us, against 16.3us for three.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

#define PLANE 400   // 20x20
#define ROWS  20
#define COLS  20

// tile shapes (swept; see the header comment)
#define CV1_NT 4    // 32 positions x 64 channels per block, 4 warps
#define CV1_MTB 4
#define CV2_NT 3    // 24 positions x 128 channels per block, 8 warps
#define CV2_MTB 8
#define POOL_PPB 4  // planes per block; blockDim = (200, POOL_PPB)
#define POOL_BLK 256  // register-cascade block size (matches the convs')

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned h2u(__half2 v) { return *reinterpret_cast<unsigned *>(&v); }
__device__ __forceinline__ __half2 u2h(unsigned v) { return *reinterpret_cast<__half2 *>(&v); }
__device__ __forceinline__ __half2 neg2() { return u2h(0xFC00FC00u); }  // (-inf,-inf)
// Raw shfl.sync: the intrinsics wrap every SHFL in WARPSYNC.COLLECTIVE /
// ENDCOLLECTIVE, which is 108 of the pool's ~1000 SASS instructions per warp and
// buys nothing where control flow is already warp-uniform.  c = 0 for .up and
// 0x1f for .down are what __shfl_up_sync / __shfl_down_sync use at width 32,
// so out-of-range source lanes return the lane's own value, as before.
__device__ __forceinline__ unsigned shfl_up1(unsigned v) {
  unsigned r;
  asm("shfl.sync.up.b32 %0, %1, 1, 0, 0xffffffff;" : "=r"(r) : "r"(v));
  return r;
}
__device__ __forceinline__ unsigned shfl_down1(unsigned v) {
  unsigned r;
  asm("shfl.sync.down.b32 %0, %1, 1, 0x1f, 0xffffffff;" : "=r"(r) : "r"(v));
  return r;
}
__device__ __forceinline__ unsigned sa(const void *p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ float silu(float v) { return v / (1.0f + __expf(-v)); }

// D += A*B, m16n8k16, A row-major (k contiguous), B col-major (k contiguous)
#define MMA16816(d, a, b)                                                      \
  asm volatile(                                                                \
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "                      \
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"                 \
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])                          \
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]))
// four transposed 8x8 tiles -> B fragments for two consecutive n-tiles
#define LDSM4T(r0, r1, r2, r3, addr)                                           \
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "                \
               "{%0,%1,%2,%3}, [%4];\n"                                         \
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr))
#define LDSM2T(r0, r1, addr)                                                   \
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "                \
               "{%0,%1}, [%2];\n"                                               \
               : "=r"(r0), "=r"(r1) : "r"(addr))

// smem pitch (halves) for a PP-wide tile: >= PP, a multiple of 8 (16 B stores)
// and == 8 (mod 64) so ldmatrix is bank-conflict free.
template <int PP> struct Pitch { enum { V = ((PP + 63) / 64) * 64 + 8 }; };

// ---------------------------------------------------------------------------
// 1x1 conv + bias + SiLU, NCHW in / NCHW out, fp32 accumulation.
//
//   K       input channels (the GEMM k extent)
//   CO_OUT  output channels of this conv
//   MTB     m-tiles (== warps) per block; each warp owns 16 output channels
//   NT      n-tiles per warp; a block owns NT*8 positions
//   SRC_C / DST_C  channels per image in the source / destination tensor
//
// grid = (ceil(PLANE/PP), CO_OUT/(MTB*16), N), dynamic smem = K*Pitch<PP>*2 B
// ---------------------------------------------------------------------------
// One (position tile x channel group x image) tile of the conv.  Split out of
// the kernel so the single-launch fused kernel below can run the same code on a
// grid it does not own: `tile` is the linearised (x,y,z) block index the
// standalone launch would have had, `sm` is this warp group's staging buffer,
// and `valid` is false for warp groups that have no tile this iteration (they
// still have to reach the __syncthreads()).
//
// PRE says what has to happen between the address arithmetic and the staging
// loads: PDL waits for the producer kernel there (so the arithmetic overlaps the
// producer's tail -- worth 3% of the kernel), NONE is right after a grid barrier
// has already synchronised the block, and SYNC protects a staging buffer that a
// previous tile of this same block is still reading.
enum TilePre { TP_PDL, TP_NONE, TP_SYNC };

template <int K, int CO_OUT, int MTB, int NT, int SRC_C, int DST_C, TilePre PRE>
__device__ __forceinline__ void gemm_silu_tile(
    const __half *__restrict__ src, const uint4 *__restrict__ aw,
    const float *__restrict__ bias, __half *__restrict__ dst, __half *sm,
    int px, int cg, int n, bool valid) {
  constexpr int PP = NT * 8, LD = Pitch<PP>::V, KSTEPS = K / 16, NTH = MTB * 32;
  constexpr int PT = (PLANE + PP - 1) / PP, CG = CO_OUT / (MTB * 16);
  static_assert(PP <= PLANE, "position tile larger than the plane");
  static_assert(K * (PP / 8) % NTH == 0, "staging must divide evenly");

  const int gtid = threadIdx.x % NTH, lane = gtid & 31, warp = gtid >> 5;
  const int gid = lane >> 2, tig = lane & 3;
  // the last position tile is clamped so it overlaps its neighbour rather than
  // running off the plane -- the overlap recomputes bit-identical values
  int p0 = px * PP;
  if (p0 > PLANE - PP) p0 = PLANE - PP;
  const int co0 = (cg * MTB + warp) * 16;
  const __half *sbase = src + (size_t)n * SRC_C * PLANE + p0;
  const uint4 *awarp = aw + (size_t)(cg * MTB + warp) * KSTEPS * 32 + lane;
  const int lr = lane & 15, lc = 8 * (lane >> 4);

  float acc[NT][4];
#pragma unroll
  for (int i = 0; i < NT; ++i)
#pragma unroll
    for (int q = 0; q < 4; ++q) acc[i][q] = 0.0f;

  if (PRE == TP_PDL) {
    // Everything above is address arithmetic: keep it ahead of the wait so it
    // overlaps the producer kernel's tail.
#if __CUDA_ARCH__ >= 900
    asm volatile("" ::"l"(sbase), "l"(awarp), "r"(p0) : "memory");
    cudaGridDependencySynchronize();
#endif
  } else if (PRE == TP_SYNC) {
    __syncthreads();
  }

  // Every weight fragment first, so all KSTEPS round trips are in flight at once
  // and drain behind the activation staging below.  This loop must stay *here*,
  // not in a helper taking a `uint4 *`: through a pointer, ptxas sinks the loads
  // back into the mma loop and the warp serialises KSTEPS ~500-cycle round trips
  // with nothing resident to hide them behind (measured: 30% of cv2).
  uint4 af[KSTEPS];
  if (valid) {
#pragma unroll
    for (int i = 0; i < KSTEPS; ++i) af[i] = awarp[(size_t)i * 32];
#pragma unroll
    for (int idx = 0; idx < K * (PP / 8) / NTH; ++idx) {
      const int f = idx * NTH + gtid, kk = f / (PP / 8), c8 = (f - kk * (PP / 8)) * 8;
      *reinterpret_cast<uint4 *>(&sm[kk * LD + c8]) =
          *reinterpret_cast<const uint4 *>(&sbase[(size_t)kk * PLANE + c8]);
    }
  }
  __syncthreads();
  if (!valid) return;

#pragma unroll
  for (int ks = 0; ks < KSTEPS; ++ks) {
    unsigned a[4] = {af[ks].x, af[ks].y, af[ks].z, af[ks].w};
    unsigned b[NT][2];
    const __half *krow = &sm[(ks * 16 + lr) * LD];
#pragma unroll
    for (int q = 0; q < NT / 2; ++q)
      LDSM4T(b[2 * q][0], b[2 * q][1], b[2 * q + 1][0], b[2 * q + 1][1],
             sa(krow + 16 * q + lc));
    if (NT & 1) LDSM2T(b[NT - 1][0], b[NT - 1][1], sa(krow + 8 * (NT - 1)));
#pragma unroll
    for (int t2 = 0; t2 < NT; ++t2) MMA16816(acc[t2], a, b[t2]);
  }

  // epilogue: += bias, SiLU, fp16, NCHW store (two contiguous positions)
  const float bl = bias[co0 + gid], bh = bias[co0 + gid + 8];
  __half *dl = dst + (size_t)n * DST_C * PLANE + (size_t)(co0 + gid) * PLANE + p0 + 2 * tig;
  __half *dh = dl + 8 * PLANE;
#pragma unroll
  for (int t2 = 0; t2 < NT; ++t2) {
    *reinterpret_cast<__half2 *>(dl + t2 * 8) =
        __floats2half2_rn(silu(acc[t2][0] + bl), silu(acc[t2][1] + bl));
    *reinterpret_cast<__half2 *>(dh + t2 * 8) =
        __floats2half2_rn(silu(acc[t2][2] + bh), silu(acc[t2][3] + bh));
  }
}

// ---------------------------------------------------------------------------
// 1x1 conv + bias + SiLU, NCHW in / NCHW out, fp32 accumulation.
//
//   K       input channels (the GEMM k extent)
//   CO_OUT  output channels of this conv
//   MTB     m-tiles (== warps) per block; each warp owns 16 output channels
//   NT      n-tiles per warp; a block owns NT*8 positions
//   SRC_C / DST_C  channels per image in the source / destination tensor
//
// grid = (ceil(PLANE/PP), CO_OUT/(MTB*16), N), dynamic smem = K*Pitch<PP>*2 B
// ---------------------------------------------------------------------------
template <int K, int CO_OUT, int MTB, int NT, int SRC_C, int DST_C>
__global__ __launch_bounds__(MTB * 32) void gemm_silu_kernel(
    const __half *__restrict__ src, const uint4 *__restrict__ aw,
    const float *__restrict__ bias, __half *__restrict__ dst) {
  extern __shared__ __half sm[];
  gemm_silu_tile<K, CO_OUT, MTB, NT, SRC_C, DST_C, TP_PDL>(
      src, aw, bias, dst, sm, blockIdx.x, blockIdx.y, blockIdx.z, true);
}

// ---------------------------------------------------------------------------
// 5/9/13 max-pool cascade: all three levels in one shared-memory pass.
//
// Columns are packed two per half2, so a plane is 200 half2 and thread
// (threadIdx.x, threadIdx.y) owns exactly one element of one plane -- its row
// and column pair are loop invariants, so every level is straight-line: a
// separable 5-tap, vertical over rows then horizontal via the
// (c-2,c-1)/(c-1,c)/(c,c+1)/(c+1,c+2)/(c+2,c+3) half2 alignment trick.
//
// Plane p of image n is cat[n][p][:]; level L writes cat[n][128*L+p][:].
// ---------------------------------------------------------------------------
template <int PPB>
__global__ __launch_bounds__(200 * PPB) void pool_cascade_kernel(
    __half *__restrict__ cat, int nplanes) {
  constexpr int E = PLANE / 2;   // 200 half2 per plane
  __shared__ __half2 s0[PPB * E];
  __shared__ __half2 s1[PPB * E];

  const int e = threadIdx.x;                 // half2 element within the plane
  const int g = blockIdx.x * PPB + threadIdx.y;
  const int r = e / 10, j = e - r * 10;      // row, column pair
  const int si = threadIdx.y * E + e;
  const bool act = g < nplanes;
  const __half2 *gs = reinterpret_cast<const __half2 *>(
      cat + ((size_t)(g >> 7) * 512 + (g & 127)) * PLANE);
#if __CUDA_ARCH__ >= 900
  asm volatile("" ::"l"(gs), "r"(si) : "memory");
  cudaGridDependencySynchronize();
#endif
  s0[si] = act ? gs[e] : neg2();
  __syncthreads();

#pragma unroll
  for (int lvl = 1; lvl <= 3; ++lvl) {
    // vertical 5-tap; rows outside the plane are -inf, exactly pad=2
    __half2 v = s0[si];
    if (r >= 2) v = __hmax2_nan(v, s0[si - 20]);
    if (r >= 1) v = __hmax2_nan(v, s0[si - 10]);
    if (r <= ROWS - 2) v = __hmax2_nan(v, s0[si + 10]);
    if (r <= ROWS - 3) v = __hmax2_nan(v, s0[si + 20]);
    s1[si] = v;
    __syncthreads();
    // horizontal 5-tap
    const __half2 m = s1[si];
    const __half2 l = (j > 0) ? s1[si - 1] : neg2();
    const __half2 rr = (j < 9) ? s1[si + 1] : neg2();
    const __half2 m0 = u2h(__byte_perm(h2u(l), h2u(m), 0x5432));   // (c-1, c)
    const __half2 m1 = u2h(__byte_perm(h2u(m), h2u(rr), 0x5432));  // (c+1, c+2)
    __half2 res = __hmax2_nan(__hmax2_nan(l, m0), __hmax2_nan(m1, rr));
    res = __hmax2_nan(res, m);
    __syncthreads();
    s0[si] = res;
    if (act)
      reinterpret_cast<__half2 *>(
          cat + ((size_t)(g >> 7) * 512 + 128 * lvl + (g & 127)) * PLANE)[e] = res;
    if (lvl != 3) __syncthreads();
  }
}


// ---------------------------------------------------------------------------
// 5/9/13 max-pool cascade, register version: no shared memory, no barrier.
//
// Ported from the frozen L1 winner (candidate/L1/maxpool_cuda.cu) and cascaded
// three deep in registers.  16 lanes per plane, two planes per warp; lanes 1..10
// hold the 10 half2 column pairs of the 20 real columns and lanes 0 / 11..15
// hold (-inf,-inf), so pad=2 costs no instructions on the column axis.  Rows
// live in each lane's registers, so the vertical 5-tap is 4 __hmax2_nan and the
// horizontal one is 2 __shfl + 2 PRMT + 5 __hmax2_nan.
//
// The point of this version is that it is grid-shape agnostic: it needs ~1/4 of
// the threads the shared-memory version does (a whole plane row-block per warp
// instead of one thread per half2 element), so it can run on a grid sized for
// the two convs -- which is what makes single-launch fusion possible.
//
// A warp owns output rows [r0, r0+R) of two planes at ALL THREE levels.  The
// row cones are level3 <- level2 rows r0-2.., level2 <- level1 rows r0-4..,
// level1 <- input rows r0-6.., so it reads R+12 rows (clamped to the plane) and
// computes R+8 / R+4 / R rows.  Rows and columns outside the plane are -inf at
// *every* level, which is exactly what composing pad-2 max-pools means, and is
// why the second/third level may clamp its row cone to [0,20) rather than
// having to widen it.
// ---------------------------------------------------------------------------
template <bool MASK = true>
__device__ __forceinline__ __half2 pool5_reg(__half2 v0, __half2 v1, __half2 v2,
                                             __half2 v3, __half2 v4, bool keep) {
  // vertical: 5 rows of this lane's column pair
  __half2 a = __hmax2_nan(__hmax2_nan(v0, v1), __hmax2_nan(v2, v3));
  a = __hmax2_nan(a, v4);
  // horizontal: the 5 half2 terms aligned to this lane's column pair
  const unsigned au = h2u(a);
  const unsigned lm = shfl_up1(au);    // (c-2, c-1)
  const unsigned lp = shfl_down1(au);  // (c+2, c+3)
  const unsigned m0 = __byte_perm(lm, au, 0x5432);           // (c-1, c)
  const unsigned m1 = __byte_perm(au, lp, 0x5432);           // (c+1, c+2)
  __half2 r = __hmax2_nan(__hmax2_nan(u2h(lm), u2h(m0)),
                          __hmax2_nan(u2h(m1), u2h(lp)));
  r = __hmax2_nan(r, a);
  // Lanes outside the plane (column pads, spare planes, rows off the plane) must
  // hold -inf so the *next* level's shuffles and vertical taps see padding.  The
  // last level feeds nothing, so it skips the select.
  return (MASK && !keep) ? neg2() : r;
}

// wid0 / wstride let the caller map warps onto whatever grid it already has;
// wid enumerates (plane pair, row block) so consecutive warps of a block stay on
// consecutive planes.
template <int R>
__device__ __forceinline__ void pool_cascade_reg(__half *__restrict__ cat,
                                                int nplanes, int wid0, int wstride) {
  static_assert(ROWS % R == 0, "R must divide the plane height");
  constexpr int RB = ROWS / R, E = PLANE / 2, CH = COLS / 2, LVL = 128 * E;
  constexpr int NIN = R + 12, NL1 = R + 8, NL2 = R + 4;
  const int lane = threadIdx.x & 31;
  const int j = (lane & 15) - 1;            // half2 column pair, real for 0..9
  const bool lact = (unsigned)j < 10u;
  const int nw = ((nplanes + 1) >> 1) * RB;

  for (int wid = wid0; wid < nw; wid += wstride) {
    const int pg = wid / RB, rb = wid - pg * RB, r0 = rb * R;
    const int plane = pg * 2 + (lane >> 4);
    const bool act = lact && plane < nplanes;
    __half2 *base = reinterpret_cast<__half2 *>(cat)
                  + (size_t)(plane >> 7) * (512 * E) + (size_t)(plane & 127) * E + j;

    __half2 in[NIN];
#pragma unroll
    for (int i = 0; i < NIN; ++i) {
      const int row = r0 - 6 + i;
      in[i] = (act && (unsigned)row < (unsigned)ROWS) ? base[row * CH] : neg2();
    }
    __half2 l1[NL1];
#pragma unroll
    for (int i = 0; i < NL1; ++i) {
      const int row = r0 - 4 + i;
      const bool ok = act && (unsigned)row < (unsigned)ROWS;
      l1[i] = pool5_reg(in[i], in[i + 1], in[i + 2], in[i + 3], in[i + 4], ok);
      if (i >= 4 && i < 4 + R && ok) base[LVL + row * CH] = l1[i];
    }
    __half2 l2[NL2];
#pragma unroll
    for (int i = 0; i < NL2; ++i) {
      const int row = r0 - 2 + i;
      const bool ok = act && (unsigned)row < (unsigned)ROWS;
      l2[i] = pool5_reg(l1[i], l1[i + 1], l1[i + 2], l1[i + 3], l1[i + 4], ok);
      if (i >= 2 && i < 2 + R && ok) base[2 * LVL + row * CH] = l2[i];
    }
#pragma unroll
    for (int i = 0; i < R; ++i) {
      const int row = r0 + i;
      const bool ok = act && (unsigned)row < (unsigned)ROWS;
      __half2 v = pool5_reg<false>(l2[i], l2[i + 1], l2[i + 2], l2[i + 3], l2[i + 4],
                                   ok);
      if (ok) base[3 * LVL + row * CH] = v;
    }
  }
}

template <int R, int BLK>
__global__ __launch_bounds__(BLK) void pool_reg_kernel(__half *__restrict__ cat,
                                                      int nplanes) {
  const int wid0 = (threadIdx.x >> 5) * gridDim.x + blockIdx.x;
#if __CUDA_ARCH__ >= 900
  asm volatile("" ::"r"(wid0) : "memory");
  cudaGridDependencySynchronize();
#endif
  pool_cascade_reg<R>(cat, nplanes, wid0, (BLK / 32) * gridDim.x);
}



// ---------------------------------------------------------------------------
// Single-launch fused SPPF: cv1 | grid barrier | pool | grid barrier | cv2.
//
// The reason the parent shipped three launches is that its pool wanted one
// thread per half2 element (102400 threads) while the convs want ~104-136 blocks
// of 4-8 warps, and no single grid serves both.  `pool_cascade_reg` removes that
// constraint -- a warp does a whole row block of two planes in registers -- so a
// grid sized for the convs can run all three phases.
//
// Two things this must not do, both learned from the parent's failed attempts:
//   * blockDim is a uniform 256 (8 warps), so __launch_bounds__ leaves ptxas the
//     full 256 registers/thread.  The parent's 800-thread fused block capped
//     registers at 81 and cost ~4us.  Each phase tiles itself *inside* that
//     fixed blockDim: cv2's 8-warp tile is one block, cv1's 4-warp tile is one
//     of two warp groups, and the pool maps warps by (warp, block) so the busy
//     warps spread evenly over blocks rather than filling the low blocks.
//   * the grid is chosen deliberately (default 136, the count cv2 was tuned at)
//     rather than by cudaOccupancyMaxActiveBlocksPerMultiprocessor, so neither
//     conv is retuned by the fusion.  With <= 148 blocks of 256 threads and 72 KB
//     of smem every block is resident, which is what the software barrier needs.
//
// The barrier is a single monotone device counter that is never reset (resetting
// it would cost the memset launch fusion just saved): barrier b waits for
// `base + (b+1)*gridDim.x` arrivals, where `base` is the total number of
// arrivals from all previous launches, tracked on the host.  Deriving the target
// from a launch *index* instead would assume every launch has the same grid, and
// silently deadlock the first time the grid size changes.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void grid_sync(unsigned long long *bar,
                                          unsigned long long target, bool skip) {
  if (skip) { __syncthreads(); return; }
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();                       // release this block's stores to L2
    atomicAdd(bar, 1ull);
    // Poll with a plain volatile load, not atomicAdd(bar,0): 136 blocks
    // read-modify-writing one address serialises on a single L2 atomic unit.
    while (*reinterpret_cast<volatile unsigned long long *>(bar) < target)
      __nanosleep(32);
  }
  __syncthreads();
}

// One conv phase, one tile per warp group.  The host only takes the fused path
// when the grid covers every tile in one pass (GRP * gridDim.x >= tiles), so
// there is no tile loop and therefore only one copy of the tile code in the
// kernel -- which matters for a kernel this short, where the instruction
// footprint of all three phases has to stay in the I-cache.
template <int K, int CO_OUT, int MTB, int NT, int SRC_C, int DST_C, int BLK,
          TilePre PRE>
__device__ __forceinline__ void gemm_silu_phase(
    const __half *__restrict__ src, const uint4 *__restrict__ aw,
    const float *__restrict__ bias, __half *__restrict__ dst, __half *sm, int n) {
  constexpr int PP = NT * 8, LD = Pitch<PP>::V, NTH = MTB * 32;
  constexpr int GRP = BLK / NTH;                    // warp groups per block
  constexpr int PT = (PLANE + PP - 1) / PP, CG = CO_OUT / (MTB * 16);
  static_assert(GRP * NTH == BLK, "warp group must divide the block");
  const int grp = threadIdx.x / NTH;
  const int tile = grp * (int)gridDim.x + (int)blockIdx.x;
  const bool valid = tile < PT * CG * n;
  const int t = valid ? tile : 0;      // keep addresses in range for idle groups
  gemm_silu_tile<K, CO_OUT, MTB, NT, SRC_C, DST_C, PRE>(
      src, aw, bias, dst, sm + (size_t)grp * K * LD, t % PT, (t / PT) % CG,
      t / (PT * CG), valid);
}

template <int R, int BLK>
__global__ __launch_bounds__(BLK) void sppf_fused_kernel(
    const __half *__restrict__ x, __half *__restrict__ cat,
    __half *__restrict__ out, const uint4 *__restrict__ aw1,
    const float *__restrict__ b1, const uint4 *__restrict__ aw2,
    const float *__restrict__ b2, int n, unsigned long long *bar,
    unsigned long long base, int nobar) {
  extern __shared__ __half sm[];
  const unsigned long long G = gridDim.x;

  gemm_silu_phase<256, 128, CV1_MTB, CV1_NT, 256, 512, BLK, TP_PDL>(
      x, aw1, b1, cat, sm, n);
  grid_sync(bar, base + G, nobar);
  pool_cascade_reg<R>(cat, n * 128, (int)(threadIdx.x >> 5) * (int)G + (int)blockIdx.x,
                      (BLK / 32) * (int)G);
  grid_sync(bar, base + 2 * G, nobar);
  gemm_silu_phase<512, 256, CV2_MTB, CV2_NT, 512, 256, BLK, TP_NONE>(
      cat, aw2, b2, out, sm, n);
}

// ---------------------------------------------------------------------------
// host side
// ---------------------------------------------------------------------------
namespace {

constexpr int CV1_PP = CV1_NT * 8, CV2_PP = CV2_NT * 8;
constexpr int CV1_SMEM = 256 * Pitch<CV1_PP>::V * 2;
constexpr int CV2_SMEM = 512 * Pitch<CV2_PP>::V * 2;
constexpr int CV1_TILES = (PLANE + CV1_PP - 1) / CV1_PP;
constexpr int CV2_TILES = (PLANE + CV2_PP - 1) / CV2_PP;
// the fused kernel's staging buffer has to hold whichever phase needs more:
// cv1's two 4-warp groups (2 x K=256 x pitch) or cv2's single 8-warp group.
constexpr int FUSE_BLK = 256;
constexpr int FUSE_CV1_SMEM = (FUSE_BLK / (CV1_MTB * 32)) * CV1_SMEM;
constexpr int FUSE_CV2_SMEM = (FUSE_BLK / (CV2_MTB * 32)) * CV2_SMEM;
constexpr int FUSE_SMEM =
    FUSE_CV1_SMEM > FUSE_CV2_SMEM ? FUSE_CV1_SMEM : FUSE_CV2_SMEM;

struct Plan {
  const uint4 *aw1;
  const float *b1;
  const uint4 *aw2;
  const float *b2;
  __half *cat;
  int n;
  int c2;
  unsigned long long *bar;    // grid-barrier counter for the fused kernel
  unsigned long long arrived; // arrivals so far, so the counter never resets
};

template <typename K, typename... Args>
inline void launch_pdl(K kern, dim3 grid, dim3 blk, int smem, cudaStream_t s, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = blk;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = s;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, kern, args...);
}

// Dev knobs (env, read once).  POOL=0 keeps the parent's shared-memory pool,
// POOL=1 selects the register cascade; POOL_R is its rows-per-warp and
// POOL_BLOCKS forces a grid size (0 = exactly cover the warps that have work).
struct Cfg {
  int pool, pool_r, pool_blocks, fuse, fuse_blocks, fuse_r, fuse_nobar;
};
Cfg g_cfg{1, 10, 0, 1, 136, 5, 0};
int envi(const char *k, int d) { const char *v = getenv(k); return v ? atoi(v) : d; }

// Largest grid whose blocks are all simultaneously resident.  The software grid
// barrier deadlocks -- hangs, not slows -- if any block is not resident, so this
// is asked of the driver rather than assumed from "136 <= 148 SMs": a future
// driver, a register-count change or a different Blackwell part could all make
// one block per SM untrue.
int g_max_grid = 0;

bool g_setup = false;
void ensure_setup() {
  if (g_setup) return;
  g_cfg.pool = envi("SPPF_POOL", 1);
  g_cfg.pool_r = envi("SPPF_POOL_R", 10);
  g_cfg.pool_blocks = envi("SPPF_POOL_BLOCKS", 0);
  g_cfg.fuse = envi("SPPF_FUSE", 1);
  g_cfg.fuse_blocks = envi("SPPF_FUSE_BLOCKS", 136);
  g_cfg.fuse_r = envi("SPPF_FUSE_R", 5);
  // Diagnostic only: skips the two grid barriers, so results are wrong but the
  // barriers' cost is isolated.
  g_cfg.fuse_nobar = envi("SPPF_FUSE_NOBAR", 0);
  cudaFuncSetAttribute(gemm_silu_kernel<256, 128, CV1_MTB, CV1_NT, 256, 512>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, CV1_SMEM);
  cudaFuncSetAttribute(gemm_silu_kernel<512, 256, CV2_MTB, CV2_NT, 512, 256>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, CV2_SMEM);
#define FUSE_SETUP(R)                                                           \
  cudaFuncSetAttribute(sppf_fused_kernel<R, FUSE_BLK>,                           \
                       cudaFuncAttributeMaxDynamicSharedMemorySize, FUSE_SMEM);
  FUSE_SETUP(5) FUSE_SETUP(10)
#undef FUSE_SETUP

  int sms = 0, dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  int per_sm = 0, lo = 1 << 30;
#define FUSE_OCC(R)                                                             \
  per_sm = 0;                                                                   \
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(                                 \
      &per_sm, sppf_fused_kernel<R, FUSE_BLK>, FUSE_BLK, FUSE_SMEM);             \
  if (per_sm < lo) lo = per_sm;
  FUSE_OCC(5) FUSE_OCC(10)
#undef FUSE_OCC
  g_max_grid = (lo > 0 && sms > 0) ? lo * sms : 0;
  g_setup = true;
}

}  // namespace

int64_t sppf_make_plan(at::Tensor aw1, at::Tensor b1, at::Tensor aw2, at::Tensor b2,
                       at::Tensor cat, int64_t n, int64_t c2) {
  ensure_setup();
  unsigned long long *bar = nullptr;
  cudaMalloc(&bar, sizeof(unsigned long long));
  cudaMemset(bar, 0, sizeof(unsigned long long));
  Plan *p = new Plan{reinterpret_cast<const uint4 *>(aw1.data_ptr()),
                     reinterpret_cast<const float *>(b1.data_ptr()),
                     reinterpret_cast<const uint4 *>(aw2.data_ptr()),
                     reinterpret_cast<const float *>(b2.data_ptr()),
                     reinterpret_cast<__half *>(cat.data_ptr()),
                     (int)n, (int)c2, bar, 0ull};
  return reinterpret_cast<int64_t>(p);
}

// Dev hook: retune without a new process, so an A/B can interleave configs in
// one run and cross-process clock drift cannot masquerade as a speedup.
void sppf_set_cfg(int64_t fuse, int64_t fuse_blocks, int64_t fuse_r, int64_t pool,
                  int64_t pool_r, int64_t pool_blocks, int64_t nobar) {
  ensure_setup();
  g_cfg.fuse = (int)fuse;
  g_cfg.fuse_blocks = (int)fuse_blocks;
  g_cfg.fuse_r = (int)fuse_r;
  g_cfg.pool = (int)pool;
  g_cfg.pool_r = (int)pool_r;
  g_cfg.pool_blocks = (int)pool_blocks;
  g_cfg.fuse_nobar = (int)nobar;
}

void sppf_free_plan(int64_t h) {
  Plan *p = reinterpret_cast<Plan *>(h);
  if (p->bar) cudaFree(p->bar);
  delete p;
}

namespace {

// One phase each, so the dev bench hook below can replay a single phase without
// duplicating launch configuration.
inline void launch_cv1(const Plan *p, const __half *xp, cudaStream_t s) {
  launch_pdl(gemm_silu_kernel<256, 128, CV1_MTB, CV1_NT, 256, 512>,
             dim3(CV1_TILES, 128 / (CV1_MTB * 16), p->n), dim3(CV1_MTB * 32),
             CV1_SMEM, s, xp, p->aw1, p->b1, p->cat);
}

inline void launch_pool(const Plan *p, cudaStream_t s) {
  const int nplanes = p->n * 128;
  if (g_cfg.pool == 0) {
    launch_pdl(pool_cascade_kernel<POOL_PPB>,
               dim3((nplanes + POOL_PPB - 1) / POOL_PPB), dim3(PLANE / 2, POOL_PPB),
               0, s, p->cat, nplanes);
    return;
  }
#define POOL_REG_CASE(R)                                                         \
  case R: {                                                                      \
    const int warps = ((nplanes + 1) / 2) * (ROWS / (R));                         \
    const int g = g_cfg.pool_blocks ? g_cfg.pool_blocks                           \
                                    : (warps + POOL_BLK / 32 - 1) / (POOL_BLK / 32); \
    launch_pdl(pool_reg_kernel<R, POOL_BLK>, dim3(g), dim3(POOL_BLK), 0, s,       \
               p->cat, nplanes);                                                  \
    break;                                                                        \
  }
  switch (g_cfg.pool_r) {
    POOL_REG_CASE(5)
    POOL_REG_CASE(10)
    default: break;
  }
#undef POOL_REG_CASE
}

// The fused path needs every tile covered in one pass by a resident grid, i.e.
// cv1's tiles <= 2*G (two 4-warp groups per block), cv2's <= G, and G <= 148 SMs
// so all blocks are resident and the barrier cannot deadlock.
inline bool fused_ok(const Plan *p) {
  const int g = g_cfg.fuse_blocks;
  if (!g_cfg.fuse || g <= 0 || g > g_max_grid) return false;
  if (g_cfg.fuse_r != 5 && g_cfg.fuse_r != 10) return false;
  if (CV1_TILES * (128 / (CV1_MTB * 16)) * p->n > 2 * g) return false;
  return CV2_TILES * (256 / (CV2_MTB * 16)) * p->n <= g;
}

inline void launch_fused(Plan *p, const __half *xp, __half *op, cudaStream_t s) {
  const unsigned long long base = p->arrived;
  // With the barriers skipped (diagnostic) nothing arrives, so the base must not
  // move -- otherwise the next real launch waits for arrivals that never happened.
  if (!g_cfg.fuse_nobar)
    p->arrived += 2ull * (unsigned long long)g_cfg.fuse_blocks;
  const int g = g_cfg.fuse_blocks;
  // fused_ok() has already rejected any R that is not instantiated here, so this
  // can never fall through and silently launch nothing.
#define FUSE_CASE(R)                                                            \
  if (g_cfg.fuse_r == (R)) {                                                    \
    launch_pdl(sppf_fused_kernel<R, FUSE_BLK>, dim3(g), dim3(FUSE_BLK),          \
               FUSE_SMEM, s, xp, p->cat, op, p->aw1, p->b1, p->aw2, p->b2,       \
               p->n, p->bar, base, g_cfg.fuse_nobar);                            \
    return;                                                                     \
  }
  FUSE_CASE(5)
  FUSE_CASE(10)
#undef FUSE_CASE
}

inline void launch_cv2(const Plan *p, __half *op, cudaStream_t s) {
  launch_pdl(gemm_silu_kernel<512, 256, CV2_MTB, CV2_NT, 512, 256>,
             dim3(CV2_TILES, 256 / (CV2_MTB * 16), p->n), dim3(CV2_MTB * 32),
             CV2_SMEM, s, const_cast<const __half *>(p->cat), p->aw2, p->b2, op);
}

}  // namespace

at::Tensor sppf_run(int64_t h, at::Tensor x) {
  const Plan *p = reinterpret_cast<const Plan *>(h);
  at::Tensor out = at::empty({p->n, p->c2, ROWS, COLS}, x.options());
  const __half *xp = reinterpret_cast<const __half *>(x.data_ptr());
  __half *op = reinterpret_cast<__half *>(out.data_ptr());
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  if (fused_ok(p)) {
    launch_fused(const_cast<Plan *>(p), xp, op, s);
  } else {
    launch_cv1(p, xp, s);
    launch_pool(p, s);
    launch_cv2(p, op, s);
  }
  return out;
}

// Dev hook: replay one phase `reps` times so a sweep can read a per-phase
// launch-to-launch cost without the profiler's ~1.2x inflation.  Not on the hot
// path; `which` is 0=cv1 1=pool 2=cv2 3=all.
void sppf_bench_phase(int64_t h, at::Tensor x, at::Tensor y, int64_t which,
                      int64_t reps) {
  const Plan *p = reinterpret_cast<const Plan *>(h);
  const __half *xp = reinterpret_cast<const __half *>(x.data_ptr());
  __half *op = reinterpret_cast<__half *>(y.data_ptr());
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  for (int64_t i = 0; i < reps; ++i) {
    if (which == 4 && fused_ok(p)) { launch_fused(const_cast<Plan *>(p), xp, op, s); continue; }
    if (which == 0 || which == 3) launch_cv1(p, xp, s);
    if (which == 1 || which == 3) launch_pool(p, s);
    if (which == 2 || which == 3) launch_cv2(p, op, s);
  }
}
