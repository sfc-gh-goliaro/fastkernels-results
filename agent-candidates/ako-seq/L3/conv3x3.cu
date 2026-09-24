// A 3x3 convolution that stages its input halo in shared memory ONCE and takes
// all nine taps from there, instead of re-gathering the input per tap.
//
// Why this file exists.  Every 3x3 in this backbone -- twelve dense convs inside
// the four C2f blocks and the three stride-2 stems -- runs on an implicit-GEMM
// Triton kernel that builds its A tile by gathering x once PER TAP, with the
// (ih, iw, in-bounds) arithmetic done per element per tap.  Measured with ncu on
// the four distinct dense 3x3s and their shipped tiles:
//
//   shape                 regs/thread  blocks/SM  warps active  L1    DRAM
//   [4, 16,160,160]->16        90          5         26.8%      70%   1.9%
//   [4, 32, 80, 80]->32       168          3         15.5%      68%   1.3%
//   [4, 64, 40, 40]->64       ~190         1         12.5%      33%   0.7%
//   [4,128, 20, 20]->128      188          1         12.1%      36%   0.5%
//
// i.e. the per-tap index tile (ih, iw and the four-way bounds mask, all
// [BLOCK_K, BLOCK_P] fp32-register-wide) costs so many registers that occupancy
// collapses to one or two CTAs per SM, and what is left is latency-bound at
// 0.3-2.2 waves with DRAM at one percent.  Both halves of that are fixed by the
// same change: bring the input patch into shared memory with one pass of
// straight vector loads, and let the nine taps be nine shared-memory offsets --
// the addressing becomes affine, the register file holds only accumulators, and
// occupancy goes back up.
//
// Two kernels, because the two regimes want different math.
//
//   KIND 0 -- direct FMA, no tensor cores.  For a tiny reduction depth, where
//     tl.dot's 16-row minimum is mostly padding: stem1 is C = 3, so 13 of every
//     16 rows of the frozen kernel's dots are zeros (19% MMA efficiency) and its
//     22 us against 23 MB of traffic is 6x off the bandwidth floor.  Here each
//     thread owns CO_T output channels x JW output columns and keeps them in
//     registers, so the nine taps of one input row feed 9 * CO_T * JW FMAs.
//
//   KIND 1 -- mma.sync.m16n8k16 against the same shared-memory patch.  For
//     C >= 16 the arithmetic is real (472 MFLOP per dense conv) and FFMA alone
//     tops out at ~13 us on the two 32-channel stems, so the tensor core has to
//     stay.  The patch is staged CHANNEL-MINOR (Xs[row][col][c]) so that the
//     m16n8k16 B fragment -- which wants two adjacent k (= two adjacent
//     channels) per lane -- is one 4-byte shared load per lane per fragment, and
//     the three horizontal taps are the same load at three column offsets.
//
// Both are CUDA-graph-capturable: no allocation, no stream sync, launched on
// whatever stream the caller passes.  The epilogue is the union of the two call
// sites' contracts (see FOLD/ACT/HAS_RES below), so a conv can read one channel
// slice of the shared C2f concat buffer and write another with no seam copy.
//
// The host prepends a block of #defines, so every problem constant below is a
// compile-time literal:
//
//   KIND C COUT IMH IMW OH OW SH SW PADH PADW
//   TH TW JW CO_T BCO NTHREAD XS IH_T NLD NQ
//   XSN XSC RSN RSC YSN YSC FOLD ACT HAS_RES A0F A1F A2F A3F
//   (KIND 1 also: BC BCS MT NT NWM NWN WS KSTAGE)

typedef unsigned short u16;
typedef unsigned int u32;
struct __align__(16) U4 { u32 x, y, z, w; };

#define CDIV(a, b) (((a) + (b) - 1) / (b))

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

__device__ __forceinline__ u32 pack2(float lo, float hi) {
  u32 d;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}

// x*sigmoid(x) = h*tanh(h) + h with h = x/2.  One MUFU and two FMA-class ops;
// max abs error against double-precision silu over [-20, 20] is 6.5e-6, two
// orders below the fp16 rounding that follows it.
__device__ __forceinline__ float silu(float v) {
  float h = 0.5f * v, t;
  asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(h));
  return h * t + h;
}

// The four epilogue pointers.  FOLD picks their meaning and the A*F flags their
// dtype, because the harness casts parameters to fp16 and leaves buffers fp32,
// so a folded BN arrives as two fp16 vectors and two fp32 ones.
//   FOLD 0: no bias at all.
//   FOLD 1: A0 is the (already folded) bias.
//   FOLD 2: A0..A3 are bn.weight, bn.bias, running_mean, running_var; the fold
//           happens here, in fp32, exactly as the frozen Triton epilogue does.
#define LDA(P, F, i) ((F) ? ((const float *)(P))[i] : h2f(((const u16 *)(P))[i]))

#if FOLD == 2
#define AFF_DECL(co)                                                          \
  const float _s = LDA(A0, A0F, co) * rsqrtf(LDA(A3, A3F, co) + eps);          \
  const float _b = LDA(A1, A1F, co) - LDA(A2, A2F, co) * _s;
#define AFF(v) ((v) * _s + _b)
#elif FOLD == 1
#define AFF_DECL(co) const float _b = LDA(A0, A0F, co);
#define AFF(v) ((v) + _b)
#else
#define AFF_DECL(co)
#define AFF(v) (v)
#endif

#if ACT
#define EPI(v) silu(AFF(v))
#else
#define EPI(v) AFF(v)
#endif

#define ARGS                                                                   \
  const u16 *__restrict__ X, const u16 *__restrict__ WT, const void *A0,        \
      const void *A1, const void *A2, const void *A3,                          \
      const u16 *__restrict__ RES, u16 *__restrict__ Y, float eps

// ===========================================================================
// Shared: stage the CTA's input patch.
//
// The patch spans input rows [oh0*SH - PADH, +IH_T) and, in columns, the
// 8-half-aligned window starting at ib = ow0*SW - 8.  Aligning the *global*
// base is what makes the staging a run of 16 B loads; the host guarantees
// TW*SW % 8 == 0, so ow0*SW is a multiple of 8 and the shared-memory index of
// input column g is exactly g - ib for every tile.  Output column ow, tap kx
// therefore reads index (ow - ow0)*SW + 7 + kx.
//
// PLANE(c) is the channel's base inside Xs; the two kinds lay the patch out
// differently (row-major per channel vs channel-minor), so each supplies its
// own.  Everything else -- the row/column bounds, the fast 16 B path and the
// edge path -- is shared.
// ===========================================================================
#define NCHUNK (C * IH_T * NQ)
#define CPT CDIV(NCHUNK, NTHREAD)


// One pass of CHW-half chunks.  A chunk is copied whole when its row is inside
// the image and its eight (or two) columns are, which is every chunk of every
// interior tile; the image-edge chunks take the per-element path and the
// row-edge ones store zeros, so the nine taps below never need a bounds test.
struct __align__(8) u32x2 { u32 x, y; };

#if KIND == 0

#if CHW == 8
#define CHUNK_T U4
#elif CHW == 4
#define CHUNK_T u32x2
#else
#define CHUNK_T u32
#endif

// A zero chunk as a *function*, not a braced literal: the literal's commas
// would be read as extra macro arguments by PSTORE below.
__device__ __forceinline__ CHUNK_T czero() {
  CHUNK_T z;
  u16 *p = (u16 *)&z;
#pragma unroll
  for (int i = 0; i < CHW; ++i) p[i] = 0;
  return z;
}

// The interior chunks -- every chunk of every tile that is not against an image
// edge -- go by cp.async, which is what keeps the staging off the warp's
// scoreboard: the data lands in shared memory without ever occupying a register
// or stalling the issuing warp, so the CTA's FMA work can start the moment the
// group is waited on rather than after a round trip per chunk.  ncu on stem1
// charges the synchronous form 2.57 warps stalled on long_scoreboard per issued
// instruction; that is the number this removes.
#if CHW == 8
#define CPY(d, s) asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(d), "l"(s))
#elif CHW == 4
#define CPY(d, s) asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(d), "l"(s))
#else
#define CPY(d, s) asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(d), "l"(s))
#endif

#define STAGE(XSBUF, PLANE, nb, oh0, ow0)                                      \
  {                                                                            \
    const int ib_ = (ow0) * SW - 8;                                            \
    const int ir_ = (oh0) * SH - PADH;                                         \
    _Pragma("unroll") for (int i_ = 0; i_ < CPT; ++i_) {                       \
      const int q_ = tid + i_ * NTHREAD;                                       \
      if (CPT * NTHREAD > NCHUNK && q_ >= NCHUNK) break;                       \
      const int c_ = q_ / (IH_T * NQ);                                         \
      const int rj_ = q_ - c_ * (IH_T * NQ);                                   \
      const int r_ = rj_ / NQ, j_ = rj_ - r_ * NQ;                             \
      const int ih_ = ir_ + r_, g0_ = ib_ + j_ * CHW;                          \
      u16 *dp_ = (XSBUF) + PLANE(c_, r_, j_);                                  \
      if (ih_ < 0 || ih_ >= IMH) {                                             \
        *(CHUNK_T *)dp_ = czero();                                             \
      } else {                                                                 \
        const u16 *sp_ =                                                       \
            X + (long)(nb) * XSN + (long)c_ * XSC + (long)ih_ * IMW;           \
        if (g0_ >= 0 && g0_ + CHW <= IMW) {                                    \
          CPY(smem_u32(dp_), sp_ + g0_);                                       \
        } else {                                                               \
          u16 t_[CHW];                                                         \
          _Pragma("unroll") for (int e_ = 0; e_ < CHW; ++e_) {                 \
            const int g_ = g0_ + e_;                                           \
            t_[e_] = (g_ >= 0 && g_ < IMW) ? sp_[g_] : (u16)0;                 \
          }                                                                    \
          *(CHUNK_T *)dp_ = *(const CHUNK_T *)t_;                              \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    asm volatile("cp.async.commit_group;");                                    \
    asm volatile("cp.async.wait_group 0;");                                    \
  }

#endif  // KIND == 0 staging

// ===========================================================================
// KIND 0: direct FMA.  Xs is [C][IH_T][XS], plain row-major per channel, so a
// thread's window for one (c, ky) is a contiguous run and loads as NLD 16 B
// shared loads.
// ===========================================================================
#if KIND == 0

#define XSPLANE (IH_T * XS)
#define XPLANE_K0(c, r, j) ((c) * XSPLANE + (r) * XS + (j) * CHW)
#define PLANE XPLANE_K0
#define NWH (BCO * 9 * C)
#define XVN ((JW - 1) * SW + 3)
#define NPJ (TW / JW)
#define NP (TH * NPJ)

#if WF32
typedef float wt_t;
#define WSTORE(h) h2f(h)      /* widen once, at staging time */
#define W2F(v) (v)
#else
typedef u16 wt_t;
#define WSTORE(h) (h)
#define W2F(v) h2f(v)
#endif

#ifdef MINBLK
#define BOUNDS __launch_bounds__(NTHREAD, MINBLK)
#else
#define BOUNDS __launch_bounds__(NTHREAD)
#endif

extern "C" __global__ BOUNDS void conv3x3(ARGS) {
  const int tid = threadIdx.x;
  const int pid = tid & (NP - 1);       // pixel slot; NP is a power of two
  const int cog = tid / NP;             // co group: warp-uniform (NP % 32 == 0)
  const int th = pid / NPJ;
  const int jg = pid - th * NPJ;

  const int owt = blockIdx.x % (OW / TW);
  const int oht = blockIdx.x / (OW / TW);
  const int oh0 = oht * TH, ow0 = owt * TW;
  const int co0 = blockIdx.y * BCO;
  const int nb = blockIdx.z;

  extern __shared__ __align__(16) u16 SM[];
  u16 *const Xs = SM;
#if WSMEM
  u16 *const Ws = SM + XSSZ;
#endif
  wt_t *const Ws = (wt_t *)(SM + C * XSPLANE);

  // [BCO, 9, C] straight out of the caller's [COUT, 9, C] weight, widened to
  // fp32 when WF32 so the inner loop's broadcast read is one LDS.32 with no cvt.
  for (int i = tid; i < NWH; i += NTHREAD)
    Ws[i] = WSTORE(WT[(long)co0 * 9 * C + i]);

  STAGE(Xs, PLANE, nb, oh0, ow0)
  __syncthreads();

  float acc[CO_T][JW];
#pragma unroll
  for (int m = 0; m < CO_T; ++m)
#pragma unroll
    for (int j = 0; j < JW; ++j) acc[m][j] = 0.f;

  const wt_t *wp = Ws + cog * CO_T * 9 * C;
  const int xbase = th * SH * XS + jg * JW * SW;

#pragma unroll CU_
  for (int c = 0; c < C; ++c) {
#pragma unroll
    for (int ky = 0; ky < 3; ++ky) {
      const u16 *xp = Xs + c * XSPLANE + xbase + ky * XS;
      u32 g[4 * NLD];
#pragma unroll
      for (int z = 0; z < NLD; ++z) {
        const U4 v = *(const U4 *)(xp + z * 8);
        g[4 * z] = v.x; g[4 * z + 1] = v.y; g[4 * z + 2] = v.z; g[4 * z + 3] = v.w;
      }
      float xv[XVN];
#pragma unroll
      for (int t = 0; t < XVN; ++t) {
        u16 lo, hi;
        unpack2(g[(t + 7) >> 1], lo, hi);
        xv[t] = h2f(((t + 7) & 1) ? hi : lo);
      }
#pragma unroll
      for (int m = 0; m < CO_T; ++m) {
        const wt_t *wq = wp + m * 9 * C + ky * 3 * C + c;
#pragma unroll
        for (int kx = 0; kx < 3; ++kx) {
          const float wv = W2F(wq[kx * C]);
#pragma unroll
          for (int j = 0; j < JW; ++j) acc[m][j] += wv * xv[j * SW + kx];
        }
      }
    }
  }

  // ---- epilogue + store ----------------------------------------------------
  const int oh = oh0 + th, ow = ow0 + jg * JW;
#pragma unroll
  for (int m = 0; m < CO_T; ++m) {
    const int co = co0 + cog * CO_T + m;
    AFF_DECL(co)
    float v[JW];
#pragma unroll
    for (int j = 0; j < JW; ++j) v[j] = EPI(acc[m][j]);
#if HAS_RES
    {
      const u16 *rp = RES + (long)nb * RSN + (long)co * RSC + (long)oh * OW + ow;
#pragma unroll
      for (int j = 0; j < JW; ++j) v[j] += h2f(rp[j]);
    }
#endif
    u16 *o = Y + (long)nb * YSN + (long)co * YSC + (long)oh * OW + ow;
#if (JW % 8) == 0
#pragma unroll
    for (int z = 0; z < JW / 8; ++z) {
      U4 q;
      q.x = pack2(v[z * 8 + 0], v[z * 8 + 1]);
      q.y = pack2(v[z * 8 + 2], v[z * 8 + 3]);
      q.z = pack2(v[z * 8 + 4], v[z * 8 + 5]);
      q.w = pack2(v[z * 8 + 6], v[z * 8 + 7]);
      *(U4 *)(o + z * 8) = q;
    }
#elif (JW % 2) == 0
#pragma unroll
    for (int j = 0; j < JW / 2; ++j)
      *(u32 *)(o + 2 * j) = pack2(v[2 * j], v[2 * j + 1]);
#else
#pragma unroll
    for (int j = 0; j < JW; ++j) o[j] = f2h(v[j]);
#endif
  }
}

#endif  // KIND == 0

// ===========================================================================
// KIND 1: mma.sync.m16n8k16 against a shared-memory patch.
//
// The layout is forced by the mma B fragment, and the constraint is worth
// stating because it rules out the obvious design.  For
// ``mma.m16n8k16.row.col`` each lane holds B[k = 2t, 2t+1][n = g] and
// B[k = 2t+8, 2t+9][n = g] with t = lane & 3 and g = lane >> 2 -- verified
// against a probe kernel, not assumed -- i.e. TWO ADJACENT REDUCTION INDICES at
// one output pixel.  Here k is the input channel, so a lane needs two adjacent
// *channels* at one pixel.
//
// The natural patch layout Xs[c][row][col] cannot serve that with one
// instruction: the two channels are a plane apart.  ``ldmatrix.x2.trans`` would
// fix it -- it reads 8 contiguous halves (the pixels) per lane and transposes,
// which is exactly how the frozen ``scdown_fused.cu`` feeds its 1x1 GEMM -- but
// ldmatrix requires each row address to be 16-byte aligned, and a 3x3's three
// horizontal taps sit at three CONSECUTIVE column offsets, so at most one of
// them can be aligned.  Measured on this device: offsets 0 and 8 load
// correctly, offsets 1, 2, 3, 7 and 9 fault.  A 1x1 can use ldmatrix.trans and
// a 3x3 cannot.
//
// So the patch is staged CHANNEL-PAIR INTERLEAVED: element (c, pixel) lives at
// (c >> 1) * CP + row * RS + slot * 2 + (c & 1).  A lane's two adjacent
// channels are then two adjacent halves -- ONE ld.shared.b32 -- and a tap shift
// moves the address by 2 halves, which keeps every tap 4-byte aligned and keeps
// the 32 lanes on 32 distinct banks (CP == 16 mod 32 is what makes that hold;
// with CP a multiple of 32 the four channel-pairs collide two ways).
//
// Interleaving costs one pass of prmt at staging time: two 16 B global loads
// (channels 2j and 2j+1 of one row) become eight prmt and two 16 B shared
// stores, so 12 instructions per 16 staged halves.
//
// The WEIGHTS have two paths, because which one wins is a traffic question that
// depends on the tile.  WSMEM 0 reads A fragments straight from global out of a
// layout the host has already permuted into mma A-fragment order
// (``FragW[co_tile][tap][k_tile][lane]``, four u32 per lane), so a fragment is
// 16 contiguous bytes and the warp's load is one coalesced 512 B ld.global.v4 --
// no shared memory, no barrier.  WSMEM 1 stages [BCO, 9, BC] once per CTA and
// reads it with ldmatrix.x4.
//
// Measured on [4,64,40,40] 64<-64 with ncu: the global path moves 15-71 MB
// through L1 for a conv whose real traffic is 1.6 MB, because an A fragment is
// re-fetched by every warp that owns its output-channel tile
// (``59 MB / NT``), and it stalls 5-12 warps per issue on long_scoreboard.
// Staging cuts that to one read per CTA, at the cost of BCO*9*BC*2 bytes of
// shared memory -- 18 KB at BCO 16, but 75 KB at BCO 64, which is what made the
// first version resident two CTAs per SM.  So the two paths trade off against
// BCO and the table picks per shape.
//
// Stride 2 gets one extra twist.  Eight consecutive output columns read input
// columns 2*ow - 1 + kx, which is stride 2, so a tap would no longer be a
// contiguous run of slots.  The patch is therefore stored PARITY-SPLIT -- even
// input columns in plane 0, odd in plane 1 -- after which output column ow at
// tap kx is slot (ow + 3) of plane 1, (ow + 4) of plane 0 and (ow + 4) of
// plane 1 for kx = 0, 1, 2, all unit-stride in ow.  The split is free: a
// staged 8-column chunk already splits into its four even and four odd halves
// inside the same prmt pass.
// ===========================================================================
#if KIND == 1

#define NWARP (NWM * NWN)
#define RS (P * XW * 2)               /* halves per patch row (pairs)        */
#define NPAIR (BC / 2)
#define XSSZ (NPAIR * CP)
#define KTC (C / 16)                  /* k-tiles in the whole reduction      */
#if WSMEM
#define WROW _PAD16_8(9 * BC)         /* halves per output channel in Ws     */
#define WCH (BCO * 9 * (BC / 8))      /* 16 B chunks of one c-block's weight */
#define WPT CDIV(WCH, NTHREAD)
#endif
#define NCB (C / BC)
#define KT (BC / 16)
#define NUNIT (NPAIR * IH_T * NCOL8)  /* interleave units of the patch       */
#define UPT CDIV(NUNIT, NTHREAD)
#define NT8 (NPIX / 8)
/* smallest value >= v congruent to 8 mod 16: an ldmatrix row stride that puts
   the 8 rows of a phase on 8 distinct bank groups. */
#define _PAD16_8(v) ((v) + ((8 - (v)) & 15))

// prmt selectors: {a.lo, b.lo} and {a.hi, b.hi} of two channel rows.
__device__ __forceinline__ u32 ilv_lo(u32 a, u32 b) {
  u32 d;
  asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
__device__ __forceinline__ u32 ilv_hi(u32 a, u32 b) {
  u32 d;
  asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

#define LDS32(dst, addr) \
  asm volatile("ld.shared.b32 %0, [%1];" : "=r"(dst) : "r"(addr))

#define MMA(d, a, b)                                                          \
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "            \
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"         \
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])                \
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),                   \
                 "r"(b[0]), "r"(b[1]))

#ifdef MINBLK
#define BOUNDS __launch_bounds__(NTHREAD, MINBLK)
#else
#define BOUNDS __launch_bounds__(NTHREAD)
#endif

extern "C" __global__ BOUNDS void conv3x3(ARGS) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wid = tid >> 5;
  const int wm = wid / NWN;
  const int wn = wid - wm * NWN;
  const int g = lane >> 2;          // mma n index within the 8-pixel tile
  const int t = lane & 3;           // mma k index / 2 within the 16-k tile

  const int oh0 = blockIdx.x * TH;
  const int co0 = blockIdx.y * BCO;
  const int nb = blockIdx.z;

  extern __shared__ __align__(16) u16 SM[];
  u16 *const Xs = SM;
#if WSMEM
  u16 *const Ws = SM + XSSZ;
#endif

  // ---- this warp's n-tiles: pixel row and column, resolved once ------------
  int trow[NT], tcol[NT];
#pragma unroll
  for (int u = 0; u < NT; ++u) {
    const int p = (wn + u * NWN) * 8 + g;
    trow[u] = p / OW;
    tcol[u] = p - trow[u] * OW;
  }
  // Base SHARED-WINDOW ADDRESS of (k-tile 0, this lane's channel pair, this
  // pixel), resolved once.  Everything the tap loop adds to it is a
  // compile-time constant, so the inner loop is ld.shared.b32 with an immediate
  // offset -- no cvta, no address arithmetic.  Doing the conversion inside the
  // loop instead cost 2.8x the designed instruction count (measured: 1.4 M
  // warp-instructions against a 0.24 M budget).
  u32 xa[NT];
#pragma unroll
  for (int u = 0; u < NT; ++u)
    xa[u] = smem_u32(Xs + t * CP + trow[u] * S * RS + tcol[u] * 2);
#if WSMEM
  // ldmatrix row/col of this lane inside a 16x16 A tile.
  const int ar = wm * MT * 16 + (lane & 7) + 8 * ((lane >> 3) & 1);
  const int ac = 8 * ((lane >> 4) & 1);
#else
  // This lane's A fragments live at a fixed 16 B slot per (co tile, tap, k
  // tile); the host has already put them in fragment order.
  const u16 *wp[MT];
#pragma unroll
  for (int m = 0; m < MT; ++m)
    wp[m] = WT + ((long)(blockIdx.y * (BCO / 16) + wm * MT + m) * 9 * KTC * 32
                  + lane) * 8;
#endif

  float acc[MT][NT][4];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int u = 0; u < NT; ++u)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[m][u][i] = 0.f;

  for (int cb = 0; cb < NCB; ++cb) {
    if (NCB > 1 && cb) __syncthreads();
#if WSMEM
    // ---- weights: [BCO, 9, BC] out of [COUT, 9, C], by cp.async ------------
#pragma unroll
    for (int i = 0; i < WPT; ++i) {
      int q = tid + i * NTHREAD;
      if (WPT * NTHREAD > WCH && q >= WCH) q = WCH - 1;
      const int co = q / (9 * (BC / 8));
      const int rem = q - co * (9 * (BC / 8));
      const int tp = rem / (BC / 8), e = rem - tp * (BC / 8);
      asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::
                       "r"(smem_u32(Ws + co * WROW + tp * BC + e * 8)),
                   "l"(WT + (long)(co0 + co) * 9 * C + tp * C + cb * BC + e * 8));
    }
    asm volatile("cp.async.commit_group;");
#endif
    // ---- patch: interleave two channels per unit --------------------------
    {
      const int ir = oh0 * S - PADH;
#pragma unroll
      for (int i = 0; i < UPT; ++i) {
        const int q = tid + i * NTHREAD;
        if (UPT * NTHREAD > NUNIT && q >= NUNIT) break;
        const int j = q / (IH_T * NCOL8);
        const int rj = q - j * (IH_T * NCOL8);
        const int r = rj / NCOL8, z = rj - r * NCOL8;
        const int ih = ir + r, g0 = -8 + 8 * z;
        u16 *dp = Xs + j * CP + r * RS;
        u32 e4[4] = {0, 0, 0, 0}, o4[4] = {0, 0, 0, 0};
        if (ih >= 0 && ih < IMH) {
          const u16 *sp = X + (long)nb * XSN
                          + (long)(cb * BC + 2 * j) * XSC + (long)ih * IMW;
          if (g0 >= 0 && g0 + 8 <= IMW) {
#if GVEC == 8
            *(U4 *)e4 = *(const U4 *)(sp + g0);
            *(U4 *)o4 = *(const U4 *)(sp + XSC + g0);
#elif GVEC == 4
            // A 20-wide image is only 4-half aligned (see the host's _c3_align),
            // so the chunk arrives as two 8 B loads instead of one 16 B one.
            *(u32x2 *)e4 = *(const u32x2 *)(sp + g0);
            *(u32x2 *)(e4 + 2) = *(const u32x2 *)(sp + g0 + 4);
            *(u32x2 *)o4 = *(const u32x2 *)(sp + XSC + g0);
            *(u32x2 *)(o4 + 2) = *(const u32x2 *)(sp + XSC + g0 + 4);
#else
#pragma unroll
            for (int z2 = 0; z2 < 4; ++z2) {
              e4[z2] = *(const u32 *)(sp + g0 + 2 * z2);
              o4[z2] = *(const u32 *)(sp + XSC + g0 + 2 * z2);
            }
#endif
          } else {
            u16 te[8], to[8];
#pragma unroll
            for (int q2 = 0; q2 < 8; ++q2) {
              const int gc = g0 + q2;
              const bool ok = gc >= 0 && gc < IMW;
              te[q2] = ok ? sp[gc] : (u16)0;
              to[q2] = ok ? sp[XSC + gc] : (u16)0;
            }
            *(U4 *)e4 = *(const U4 *)te;
            *(U4 *)o4 = *(const U4 *)to;
          }
        }
#if P == 1
        // slots g0+8 .. g0+15, two halves each: lo pair then hi pair
        u32 out[8];
#pragma unroll
        for (int z2 = 0; z2 < 4; ++z2) {
          out[2 * z2] = ilv_lo(e4[z2], o4[z2]);
          out[2 * z2 + 1] = ilv_hi(e4[z2], o4[z2]);
        }
        *(U4 *)(dp + 16 * z) = *(const U4 *)out;
        *(U4 *)(dp + 16 * z + 8) = *(const U4 *)(out + 4);
#else
        // parity 0 = even columns (the lo halves), parity 1 = odd (the hi)
        u32 p0[4], p1[4];
#pragma unroll
        for (int z2 = 0; z2 < 4; ++z2) {
          p0[z2] = ilv_lo(e4[z2], o4[z2]);
          p1[z2] = ilv_hi(e4[z2], o4[z2]);
        }
        *(U4 *)(dp + 8 * z) = *(const U4 *)p0;
        *(U4 *)(dp + XW * 2 + 8 * z) = *(const U4 *)p1;
#endif
      }
    }
#if WSMEM
    asm volatile("cp.async.wait_group 0;");
#endif
    __syncthreads();

    // ---- the nine taps, against the staged patch --------------------------
#pragma unroll
    for (int ck = 0; ck < KT; ++ck) {
#pragma unroll
      for (int ky = 0; ky < 3; ++ky) {
#pragma unroll
        for (int kx = 0; kx < 3; ++kx) {
          u32 a[MT][4];
#if WSMEM
#pragma unroll
          for (int m = 0; m < MT; ++m) {
            const u32 wa = smem_u32(Ws + (ar + m * 16) * WROW
                                    + (ky * 3 + kx) * BC + ck * 16 + ac);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                         "{%0,%1,%2,%3}, [%4];"
                         : "=r"(a[m][0]), "=r"(a[m][1]), "=r"(a[m][2]),
                           "=r"(a[m][3]) : "r"(wa));
          }
#else
#pragma unroll
          for (int m = 0; m < MT; ++m)
            *(U4 *)a[m] = *(const U4 *)(
                wp[m] + (long)(((ky * 3 + kx) * KTC + cb * KT + ck) * 32) * 8);
#endif
#pragma unroll
          for (int u = 0; u < NT; ++u) {
            u32 b[2];
            // Tap offset in halves: unit stride in the output column, because
            // parity is split at stride 2.  All compile-time.
#if P == 1
            const int toff = ky * RS + (7 + kx) * 2;
#else
            const int toff = ky * RS + (kx == 0 ? (XW + 3) * 2
                                       : kx == 1 ? 8 : (XW + 4) * 2);
#endif
            const u32 base = xa[u] + (u32)((ck * 8 * CP + toff) * 2);
            LDS32(b[0], base);
            LDS32(b[1], base + (u32)(4 * CP * 2));
#pragma unroll
            for (int m = 0; m < MT; ++m) MMA(acc[m][u], a[m], b);
          }
        }
      }
    }
  }

  // ---- epilogue + store ----------------------------------------------------
  // Lane l holds D[row = g, g+8][col = 2t, 2t+1] of each 16x8 fragment, so its
  // two pixels are adjacent (OW is even, so an 8-pixel tile never straddles a
  // row) and go out as one 4-byte store per output channel.
  int opix[NT];
#pragma unroll
  for (int u = 0; u < NT; ++u) {
    const int p = (wn + u * NWN) * 8 + 2 * t;
    const int r = p / OW;
    opix[u] = (oh0 + r) * OW + (p - r * OW);
  }
#pragma unroll
  for (int m = 0; m < MT; ++m) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int co = co0 + wm * MT * 16 + m * 16 + g + 8 * i;
      AFF_DECL(co)
      u16 *yp = Y + (long)nb * YSN + (long)co * YSC;
#if HAS_RES
      const u16 *rp = RES + (long)nb * RSN + (long)co * RSC;
#endif
#pragma unroll
      for (int u = 0; u < NT; ++u) {
        float v0 = EPI(acc[m][u][2 * i]);
        float v1 = EPI(acc[m][u][2 * i + 1]);
#if HAS_RES
        v0 += h2f(rp[opix[u]]);
        v1 += h2f(rp[opix[u] + 1]);
#endif
        *(u32 *)(yp + opix[u]) = pack2(v0, v1);
      }
    }
  }
}

#endif  // KIND == 1
