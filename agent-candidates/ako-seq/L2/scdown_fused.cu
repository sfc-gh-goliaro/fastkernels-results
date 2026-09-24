// Fused YOLOv10 SCDown block: SiLU(BN(conv1x1(x))) + 3x3 stride-2 depthwise BN,
// in ONE kernel launch, never materializing the c2 x H x W intermediate.
//
// One CTA owns (image n, TC output channels, TOH output rows) across the full
// width.  It streams the 2*TOH+1 input rows it needs.  Per row it runs the 1x1
// GEMM on tensor cores (mma.sync.m16n8k16, A = the [TC,c1] folded w1 block held
// in registers for the CTA's whole lifetime, B = the x row via
// ldmatrix.x2.trans out of shared memory), applies SiLU exactly once per
// element, parks the fp16 row in shared memory, and folds it into the depthwise
// accumulators of the one or two output rows it feeds -- so only TOH
// accumulator sets are ever live.
//
// The row loop is software-pipelined at two depths, which is what makes the CTA
// critical path short: x rows arrive by double-buffered cp.async two rows
// ahead, and Ys is double-buffered so the GEMM of row r+1 is issued alongside
// the depthwise fold of row r.  That leaves ONE __syncthreads per row instead
// of three, and the fold's shared-memory latency overlaps the next GEMM.
//
// For the depthwise each thread owns ONE channel and DW contiguous output
// columns, so its 3x3 tap weights are a single shared-memory row, its Ys window
// is one pointer plus immediate offsets, and its output store is one 16B
// transaction.
//
// The host prepends a block of #defines:
//   C1 C2 H W OH OW TC TOH NWM NWN XSTRIDE YSTRIDE
// Compiled by NVRTC, so every shape constant is a compile-time literal.

#define NWARP    (NWM * NWN)
#define NTHREAD  (32 * NWARP)
#define MT       (TC / (16 * NWM))       /* m-tiles (of 16 channels) per warp  */
#define KT       (C1 / 16)               /* k-tiles of the 1x1 reduction       */
#define NTT      (W / 8)                 /* n-tiles (of 8 columns) per row     */
#define NTMAX    ((NTT + NWN - 1) / NWN) /* n-tiles per warp                   */
#define CPR      (W / 8)                 /* 16B chunks per channel row         */
#define NCHUNK   (C1 * CPR)
#define CPT      ((NCHUNK + NTHREAD - 1) / NTHREAD)
#define NROWS    (2 * TOH + 1)
#define TPC      (NTHREAD / TC)          /* depthwise threads per channel      */
#define DW       (OW / TPC)              /* depthwise columns per thread       */
#define YPAD     8                       /* zero column left of each Ys row    */
#define XBUF     (C1 * XSTRIDE)          /* halves per Xs buffer               */
#define YBUF     (TC * YSTRIDE)          /* halves per Ys buffer               */
/* XALL: all 2*TOH+1 x rows of one channel are CONTIGUOUS in global memory, so
   the whole patch can be staged in a single burst (CSTRIDE = padded halves per
   channel).  That takes every global load off the row loop's critical path --
   one wait and one barrier before it, after which the loop is pure shared
   memory with a single barrier per row. */
#define NCHUNKA  (C1 * NROWS * CPR)
#define CPTA     ((NCHUNKA + NTHREAD - 1) / NTHREAD)
#define WCH      (TC * WSTRIDE / 8)      /* 16B chunks of the w1 tile          */
#define WPT      ((WCH + NTHREAD - 1) / NTHREAD)
#define NCH      (1 + DW / 4)            /* 16B loads covering the fold window */
#ifndef XDEPTH
#define XDEPTH   3                       /* x row buffers (cp.async depth)     */
#endif
/* cp.async.wait_group is per-thread, so a __syncthreads must sit between the
   wait and the first read of the row.  With >=3 x buffers one barrier per row
   does both jobs -- publish row s+1 (issued XDEPTH-2 rows back, so already
   waited for) and free the buffer row s's GEMM just finished with.  With 2
   buffers those are the same buffer, so that variant needs two barriers; it is
   still worth having because it saves a whole x buffer of shared memory, which
   is what caps CTAs/SM. */
#if XDEPTH == 2
#define WAITN    1
#else
#define WAITN    (XDEPTH - 2)
#endif
#ifdef CPG
#define CPKIND   "cp.async.cg.shared.global"
#else
#define CPKIND   "cp.async.ca.shared.global"
#endif

/* Runtime-false, opaque to the compiler: keeps ablated stores from being DCE'd
   so an ablation measures only the work it removes. */
#define NEVER    (blockIdx.z >= gridDim.z)

typedef unsigned short u16;
typedef unsigned int u32;
struct __align__(16) U4 { u32 x, y, z, w; };

__device__ __forceinline__ u32 smem_u32(const void *p) {
  u32 a;
  asm("{ .reg .u64 u; cvta.to.shared.u64 u, %1; cvt.u32.u64 %0, u; }"
      : "=r"(a) : "l"(p));
  return a;
}

__device__ __forceinline__ float h2f(u16 h) {
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
  return f;
}

__device__ __forceinline__ void unpack2(u32 v, u16 &lo, u16 &hi) {
  asm("mov.b32 {%0,%1}, %2;" : "=h"(lo), "=h"(hi) : "r"(v));
}

__device__ __forceinline__ u16 f2h(float f) {
  u16 h;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
  return h;
}

// The first source of cvt.rn.f16x2.f32 lands in the HIGH half (measured).
// SiLU on a packed half2, in fp16 throughout: cvt, hmul2, tanh.approx.f16x2,
// hfma2 -- 4 instructions for two elements instead of 7.
__device__ __forceinline__ u32 silu_pack2_h(float lo, float hi);

__device__ __forceinline__ u32 pack2(float lo, float hi) {
  u32 d;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}

// x*sigmoid(x) = h*tanh(h) + h with h = x/2: one MUFU, two FMA-class ops.
// Max abs error vs double-precision silu over [-20,20] is 6.5e-6 (measured),
// two orders below the fp16 rounding that follows it.
__device__ __forceinline__ float silu(float v) {
#ifdef ABL_SILU
  return v;
#else
  float h = 0.5f * v, t;
  asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(h));
  return h * t + h;
#endif
}

__device__ __forceinline__ u32 silu_pack2_h(float lo, float hi) {
  u32 p = pack2(lo, hi), h, tv, r;
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(h) : "r"(p), "r"(0x38003800u));
  asm("tanh.approx.f16x2 %0, %1;" : "=r"(tv) : "r"(h));
  asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(h), "r"(tv), "r"(h));
  return r;
}

#ifdef MINBLK
#define BOUNDS __launch_bounds__(NTHREAD, MINBLK)
#else
#define BOUNDS __launch_bounds__(NTHREAD)
#endif

extern "C" __global__ BOUNDS void scdown_fused(
    const u16 *__restrict__ X,    // [N, C1, H, W]   fp16
    const u16 *__restrict__ W1,   // [C2, C1]        fp16, BN folded
    const float *__restrict__ B1, // [C2]
    const float *__restrict__ W2, // [C2, 9]         fp32, BN folded
    const float *__restrict__ B2, // [C2]
    u16 *__restrict__ OUT         // [N, C2, OH, OW] fp16
#ifdef TIMEIT
    , unsigned long long *__restrict__ TM
#endif
) {
#ifdef TIMEIT
  unsigned long long _t0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_t0));
#endif
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wid = tid >> 5;
  const int wm = wid / NWN;
  const int wn = wid - wm * NWN;
  const int gid = lane >> 2;
  const int tig = lane & 3;

  const int oh0 = blockIdx.x * TOH;
  const int cb = blockIdx.y * TC;
  const int nb = blockIdx.z;

  extern __shared__ __align__(16) u16 SM[];
  u16 *const Xs = SM;
#ifdef XALL
  u16 *const Ys = SM + C1 * CSTRIDE;
#else
  u16 *const Ys = SM + XDEPTH * XBUF;
#endif
  u16 *const Ws = Ys + 2 * YBUF;
  float *const Wd = (float *)(Ws + TC * WSTRIDE);

  // The [TC, C1] folded 1x1 weight block arrives by coalesced cp.async and is
  // pulled into the A fragments with one ldmatrix.x4 per 16x16 tile.  Reading
  // it straight from global instead costs MT*KT*4 scattered 4B loads per thread
  // on the CTA's critical path, which the profiler charges as long_scoreboard.
#pragma unroll
  for (int i = 0; i < WPT; ++i) {
    int q = tid + i * NTHREAD;
    if (q >= WCH) q = WCH - 1;
    const int r = q / (WSTRIDE / 8), j = q - r * (WSTRIDE / 8);
    if (j * 8 < C1) {
      const u32 d = smem_u32(Ws + r * WSTRIDE + j * 8);
      const u16 *s = W1 + (long)(cb + r) * C1 + j * 8;
      asm volatile(CPKIND " [%0], [%1], 16;" ::"r"(d), "l"(s));
    }
  }
  asm volatile("cp.async.commit_group;");

  for (int i = tid; i < TC * 9; i += NTHREAD) Wd[i] = W2[(cb + i / 9) * 9 + i % 9];
  for (int c = tid; c < 2 * TC; c += NTHREAD) Ys[c * YSTRIDE + YPAD - 1] = 0;

  u32 a[MT][KT][4];
  float bl[MT], bh[MT];
#pragma unroll
  for (int m = 0; m < MT; ++m) {
    const int cc = cb + wm * MT * 16 + m * 16 + gid;
    bl[m] = B1[cc];
    bh[m] = B1[cc + 8];
  }

  // ---- this thread's slice of the depthwise: one channel, DW columns -------
  const int dc = tid / TPC;
  const int ow0 = (tid - dc * TPC) * DW;
  const float *wp = Wd + dc * 9;
  const int yoff = dc * YSTRIDE + YPAD + 2 * ow0;   // fold window, per Ys buffer
  float acc[TOH][DW];
  {
    const float b = B2[cb + dc];
#pragma unroll
    for (int q = 0; q < TOH; ++q)
#pragma unroll
      for (int j = 0; j < DW; ++j) acc[q][j] = b;
  }

  // ---- cp.async plan: CPT fixed (channel, 16B chunk) slots per thread ------
  const int ih0 = 2 * oh0 - 1;
  const int r0 = (ih0 < 0) ? 1 : 0;
  const int RN = NROWS - r0;              // rows this CTA actually streams
#ifdef XALL
  {
    const u16 *xb = X + ((long)nb * C1) * H * W + (long)(ih0 + r0) * W;
    const int nchunk = C1 * RN * CPR;
#pragma unroll
    for (int i = 0; i < CPTA; ++i) {
      const int q = tid + i * NTHREAD;
      if (q < nchunk) {
        const int c = q / (RN * CPR), rj = q - c * (RN * CPR);
        const u32 d = smem_u32(Xs + c * CSTRIDE + rj * 8);
        const u16 *s = xb + (long)c * (H * W) + rj * 8;
        asm volatile(CPKIND " [%0], [%1], 16;" ::"r"(d), "l"(s));
      }
    }
  }
  asm volatile("cp.async.commit_group;");
#else
  const u16 *gp[CPT];
  u32 sa[CPT];
  {
    const u16 *xb = X + ((long)nb * C1) * H * W + (long)(ih0 + r0) * W;
#pragma unroll
    for (int i = 0; i < CPT; ++i) {
      int q = tid + i * NTHREAD;
      if (q >= NCHUNK) q = NCHUNK - 1;      // harmless duplicate copy
      const int c = q / CPR, j = q - c * CPR;
      gp[i] = xb + (long)c * (H * W) + j * 8;
      sa[i] = smem_u32(Xs + c * XSTRIDE + j * 8);
    }
  }
#endif
  const u32 XD = (u32)(XBUF * 2);           // bytes between the Xs buffers
  // ldmatrix bases: one per n-tile this warp owns, plus compile-time k offsets
#ifdef XALL
  const u32 xb0 = smem_u32(Xs + (lane & 15) * CSTRIDE);
#define KSTEP    (32 * CSTRIDE)
#else
  const u32 xb0 = smem_u32(Xs + (lane & 15) * XSTRIDE);
#define KSTEP    (32 * XSTRIDE)
#endif
#ifdef LDMX4
  const u32 xb1 = smem_u32(Xs + lane * XSTRIDE);   // ldmatrix.x4 uses all 32 lanes
#endif
  u32 xw[NTMAX];
#pragma unroll
  for (int u = 0; u < NTMAX; ++u) xw[u] = (u32)((wn + u * NWN) * 16);
  const int ysoff = (wm * MT * 16 + gid) * YSTRIDE + YPAD + 2 * tig;

#ifndef XALL
  // issue the cp.async group for the next x row into buffer `b`
  auto load_row = [&](int b) {
    const u32 off = (u32)b * XD;
#pragma unroll
    for (int i = 0; i < CPT; ++i) {
#ifndef ABL_CP
      asm volatile(CPKIND " [%0], [%1], 16;" ::"r"(sa[i] + off), "l"(gp[i]));
#endif
      gp[i] += W;
    }
    asm volatile("cp.async.commit_group;");
  };
#endif

  // 1x1 GEMM + SiLU for the x row in buffer `xbuf` -> Ys buffer `ybuf`
  auto gemm_row = [&](int xbuf, int ybuf) {
#ifdef XALL
    const u32 xoff = (u32)xbuf * (W * 2);      // xbuf is the row within the patch
#else
    const u32 xoff = (u32)xbuf * XD;
#endif
    u16 *ys1 = Ys + ybuf * YBUF + ysoff;
#pragma unroll
    for (int u = 0; u < NTMAX; ++u) {
      const int t = wn + u * NWN;
      if (t < NTT) {
        const u32 xa = xb0 + xoff + xw[u];
#ifdef LDMX4
        const u32 xa2 = xb1 + xoff + xw[u];
#endif
        float d[MT][4];
#pragma unroll
        for (int m = 0; m < MT; ++m) {
          d[m][0] = bl[m]; d[m][1] = bl[m]; d[m][2] = bh[m]; d[m][3] = bh[m];
        }
#if defined(LDMX4) && ((KT % 2) == 0)
#pragma unroll
        for (int kk = 0; kk < KT / 2; ++kk) {
          u32 bb[4] = {0, 0, 0, 0};
#ifndef ABL_LDM
          // one x4.trans covers two k-tiles: r0,r1 -> k-tile 2kk, r2,r3 -> 2kk+1
          asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                       "{%0,%1,%2,%3}, [%4];"
                       : "=r"(bb[0]), "=r"(bb[1]), "=r"(bb[2]), "=r"(bb[3])
                       : "r"(xa2 + kk * 2 * KSTEP));
#endif
#ifndef ABL_MMA
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int k = 2 * kk + h;
#pragma unroll
            for (int m = 0; m < MT; ++m)
              asm volatile(
                  "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                  : "+f"(d[m][0]), "+f"(d[m][1]), "+f"(d[m][2]), "+f"(d[m][3])
                  : "r"(a[m][k][0]), "r"(a[m][k][1]), "r"(a[m][k][2]), "r"(a[m][k][3]),
                    "r"(bb[2 * h]), "r"(bb[2 * h + 1]));
          }
#endif
        }
#else
#pragma unroll
        for (int k = 0; k < KT; ++k) {
          u32 b0 = 0, b1 = 0;
#ifndef ABL_LDM
          asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                       : "=r"(b0), "=r"(b1) : "r"(xa + k * KSTEP));
#endif
#ifndef ABL_MMA
#pragma unroll
          for (int m = 0; m < MT; ++m)
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(d[m][0]), "+f"(d[m][1]), "+f"(d[m][2]), "+f"(d[m][3])
                : "r"(a[m][k][0]), "r"(a[m][k][1]), "r"(a[m][k][2]), "r"(a[m][k][3]),
                  "r"(b0), "r"(b1));
#endif
        }
#endif
        u16 *ys = ys1 + t * 8;
#pragma unroll
        for (int m = 0; m < MT; ++m) {
#ifdef SILU_H2
          const u32 p0 = silu_pack2_h(d[m][0], d[m][1]);
          const u32 p1 = silu_pack2_h(d[m][2], d[m][3]);
#else
          const u32 p0 = pack2(silu(d[m][0]), silu(d[m][1]));
          const u32 p1 = pack2(silu(d[m][2]), silu(d[m][3]));
#endif
#ifdef ABL_YS
          if (NEVER) { *(u32 *)(ys + m * 16 * YSTRIDE) = p0;
                       *(u32 *)(ys + (m * 16 + 8) * YSTRIDE) = p1; }
#else
          *(u32 *)(ys + m * 16 * YSTRIDE) = p0;
          *(u32 *)(ys + (m * 16 + 8) * YSTRIDE) = p1;
#endif
        }
      }
    }
  };

  // ---- prologue: w1 tile (+ x, all at once under XALL) ----------------------
#ifdef XALL
  asm volatile("cp.async.wait_group 0;");
#else
#pragma unroll
  for (int b = 0; b < XDEPTH; ++b) {
    if (b < RN) load_row(b); else asm volatile("cp.async.commit_group;");
  }
  asm volatile("cp.async.wait_group %0;" ::"n"(WAITN));  // w1 tile + rows 0,1 in
#endif
  __syncthreads();

  {
    // ldmatrix.x4 row/col for this lane inside a 16x16 A tile
    const int ar = wm * MT * 16 + (lane & 7) + 8 * ((lane >> 3) & 1);
    const int ac = 8 * ((lane >> 4) & 1);
#pragma unroll
    for (int m = 0; m < MT; ++m) {
#pragma unroll
      for (int k = 0; k < KT; ++k) {
#ifdef ABL_A
        a[m][k][0] = a[m][k][1] = a[m][k][2] = a[m][k][3] = 0x3c003c00u;
#else
        const u32 wa = smem_u32(Ws + (ar + m * 16) * WSTRIDE + ac + k * 16);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(a[m][k][0]), "=r"(a[m][k][1]),
                       "=r"(a[m][k][2]), "=r"(a[m][k][3]) : "r"(wa));
#endif
      }
    }
  }
  gemm_row(0, 0);

  // Per row, in this order: barrier; GEMM of row s+1 (whose x buffer the
  // previous barrier published) into the other Ys buffer; issue row s+XDEPTH
  // into the buffer row s's GEMM has finished with; wait; fold row s.  The GEMM
  // and the fold are independent, so they interleave and the fold's
  // shared-memory latency hides under the next row's mma.
  for (int s = 0; s < RN; ++s) {
    __syncthreads();
#ifdef XALL
    if (s + 1 < RN) gemm_row(s + 1, (s + 1) & 1);
#elif XDEPTH == 2
    if (s + XDEPTH < RN) load_row(s % XDEPTH);
    else asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group %0;" ::"n"(WAITN));
    __syncthreads();
    if (s + 1 < RN) gemm_row((s + 1) % XDEPTH, (s + 1) & 1);
#else
    if (s + 1 < RN) gemm_row((s + 1) % XDEPTH, (s + 1) & 1);
    if (s + XDEPTH < RN) load_row(s % XDEPTH);
    else asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group %0;" ::"n"(WAITN));
#endif

    // ---- fold row s into the depthwise accumulators -----------------------
#ifndef ABL_DW
    const u16 *yp = Ys + (s & 1) * YBUF + yoff;
    const int ih = ih0 + r0 + s;
#if ((DW % 4) == 0) && !defined(ABL_FVEC)
    // The whole 2*DW+1 half window this thread needs comes in NCH 16B loads
    // instead of 2*DW+1 two-byte ones: same bytes, ~6x fewer LSU instructions,
    // and the fold competes with cp.async for exactly that pipe.
    float yv[2 * DW + 1];
    {
      u32 g[4 * NCH];
#pragma unroll
      for (int z = 0; z < NCH; ++z) {
        const U4 v = *(const U4 *)(yp - 8 + 8 * z);
        g[4 * z + 0] = v.x; g[4 * z + 1] = v.y;
        g[4 * z + 2] = v.z; g[4 * z + 3] = v.w;
      }
#pragma unroll
      for (int u = 0; u < 2 * DW + 1; ++u) {
        u16 lo, hi;
        unpack2(g[(u + 7) >> 1], lo, hi);
        yv[u] = h2f(((u + 7) & 1) ? hi : lo);
      }
    }
#pragma unroll
    for (int q = 0; q < TOH; ++q) {
      const int i = ih - 2 * (oh0 + q) + 1;
      if ((unsigned)i <= 2u) {
        const float *w = wp + i * 3;
        const float w0 = w[0], w1 = w[1], w2v = w[2];
#pragma unroll
        for (int j = 0; j < DW; ++j)
          acc[q][j] += w0 * yv[2 * j] + w1 * yv[2 * j + 1] + w2v * yv[2 * j + 2];
      }
    }
#else
#pragma unroll
    for (int q = 0; q < TOH; ++q) {
      const int i = ih - 2 * (oh0 + q) + 1;
      if ((unsigned)i <= 2u) {
        const float *w = wp + i * 3;
        const float w0 = w[0], w1 = w[1], w2v = w[2];
#pragma unroll
        for (int j = 0; j < DW; ++j)
          acc[q][j] += w0 * h2f(yp[2 * j - 1]) + w1 * h2f(yp[2 * j])
                     + w2v * h2f(yp[2 * j + 1]);
      }
    }
#endif
#endif
  }

  // ---- store -------------------------------------------------------------
#pragma unroll
  for (int q = 0; q < TOH; ++q) {
    u16 *o = OUT + ((long)(nb * C2 + cb + dc) * OH + oh0 + q) * OW + ow0;
#ifdef ABL_OUT
    if (NEVER)
#endif
    {
#if ((DW % 8) == 0) && ((OW % 8) == 0)
#pragma unroll
      for (int b = 0; b < DW / 8; ++b) {
        U4 v;
        v.x = pack2(acc[q][b * 8 + 0], acc[q][b * 8 + 1]);
        v.y = pack2(acc[q][b * 8 + 2], acc[q][b * 8 + 3]);
        v.z = pack2(acc[q][b * 8 + 4], acc[q][b * 8 + 5]);
        v.w = pack2(acc[q][b * 8 + 6], acc[q][b * 8 + 7]);
        *(U4 *)(o + b * 8) = v;
      }
#elif ((DW % 2) == 0) && ((OW % 2) == 0)
#pragma unroll
      for (int j = 0; j < DW / 2; ++j)
        *(u32 *)(o + 2 * j) = pack2(acc[q][2 * j], acc[q][2 * j + 1]);
#else
#pragma unroll
      for (int j = 0; j < DW; ++j) o[j] = f2h(acc[q][j]);
#endif
    }
  }
#ifdef TIMEIT
  if (tid == 0) {
    unsigned long long _t1;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_t1));
    const int bid = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    TM[2 * bid] = _t0;
    TM[2 * bid + 1] = _t1;
  }
#endif
}
