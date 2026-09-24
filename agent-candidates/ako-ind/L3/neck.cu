// YOLOv10 neck -- one fused static-shape inference path.
//
// The whole neck (22 folded convs, 2 nearest-2x upsamples, 4 concats, 4 chunks,
// 1 residual add) is issued from a single host call.  All BatchNorms are already
// folded into the conv weights on the Python side; SiLU, the bias add, the
// residual add and the concat placement are conv epilogues here, so no tensor is
// ever read or written just to move it.
//
// Layout is NCHW throughout (same as the harness inputs and outputs), so there
// is no layout conversion at either end.  A 1x1 conv is then a batched GEMM
//     C[Cout, HW] = W[Cout, Cin] * X[Cin, HW]
// and a KxK conv is the same GEMM with the K dimension extended to
// (kh, kw, ci) and the B-operand tile gathered at a spatial offset -- one
// templated kernel covers both.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <vector>

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t e_ = (x);                                                       \
    TORCH_CHECK(e_ == cudaSuccess, "CUDA error: ", cudaGetErrorString(e_));     \
  } while (0)

namespace {

// ===========================================================================
// The neck's static plan
// ===========================================================================
// Indices into the 6-entry device pointer array that the graph-capturable
// kernels read their harness-facing base addresses from.
enum IoId { IO_P3B = 0, IO_P4B, IO_P5B, IO_P3, IO_N4, IO_N5, NIO };

enum ConvId {
  A_CV1 = 0, A_B1, A_B2, A_CV2,
  B_CV1, B_B1, B_B2, B_CV2,
  DP3,
  D_CV1, D_B1, D_B2, D_CV2,
  S_CV1, S_CV2,
  E_CV1, I0, I1, I2, I3, I4, E_CV2,
  NCONV
};

struct ConvSpec {
  int cin, cout, k, stride, dw;  // dw != 0 => depthwise (groups == cin == cout)
};

constexpr ConvSpec kSpec[NCONV] = {
    {384, 128, 1, 1, 0},  // A_CV1  c2f_p4.cv1        40x40
    { 64,  64, 3, 1, 0},  // A_B1   c2f_p4.m0.cv1     40x40
    { 64,  64, 3, 1, 0},  // A_B2   c2f_p4.m0.cv2     40x40
    {192, 128, 1, 1, 0},  // A_CV2  c2f_p4.cv2        40x40
    {192,  64, 1, 1, 0},  // B_CV1  c2f_p3.cv1        80x80
    { 32,  32, 3, 1, 0},  // B_B1   c2f_p3.m0.cv1     80x80
    { 32,  32, 3, 1, 0},  // B_B2   c2f_p3.m0.cv2     80x80
    { 96,  64, 1, 1, 0},  // B_CV2  c2f_p3.cv2        80x80
    { 64,  64, 3, 2, 0},  // DP3    down_p3           80x80 -> 40x40
    {192, 128, 1, 1, 0},  // D_CV1  c2f_n4.cv1        40x40
    { 64,  64, 3, 1, 0},  // D_B1   c2f_n4.m0.cv1     40x40
    { 64,  64, 3, 1, 0},  // D_B2   c2f_n4.m0.cv2     40x40
    {192, 128, 1, 1, 0},  // D_CV2  c2f_n4.cv2        40x40
    {128, 128, 1, 1, 0},  // S_CV1  down_n4.cv1       40x40
    {128, 128, 3, 2, 1},  // S_CV2  down_n4.cv2 (dw)  40x40 -> 20x20
    {384, 256, 1, 1, 0},  // E_CV1  c2fcib_n5.cv1     20x20
    {128, 128, 3, 1, 1},  // I0     cib.cv1[0] (dw)   20x20
    {128, 256, 1, 1, 0},  // I1     cib.cv1[1]        20x20
    {256, 256, 7, 1, 1},  // I2     cib.cv1[2] repvgg 20x20
    {256, 128, 1, 1, 0},  // I3     cib.cv1[3]        20x20
    {128, 128, 3, 1, 1},  // I4     cib.cv1[4] (dw)   20x20
    {384, 256, 1, 1, 0},  // E_CV2  c2fcib_n5.cv2     20x20
};

// Output pixel count of each conv (needed at compile time: it selects BN).
constexpr int kN[NCONV] = {
    1600, 1600, 1600, 1600,          // c2f_p4   40x40
    6400, 6400, 6400, 6400,          // c2f_p3   80x80
    1600,                            // down_p3  -> 40x40
    1600, 1600, 1600, 1600,          // c2f_n4   40x40
    1600, 400,                       // SCDown   40x40 -> 20x20
    400, 400, 400, 400, 400, 400, 400,  // c2fcib_n5 20x20
};

// How many K-tiles' global loads to keep in flight.  These kernels run 25-200
// blocks on 148 SMs, so occupancy is bounded by available parallelism, never by
// registers or shared memory: depth is close to free here, which is the
// opposite of the trade a throughput-bound GEMM makes.  It only pays once the
// K loop is fully unrolled -- see the NTILES comment on conv_kernel.
#ifndef NECK_PF
#define NECK_PF 3
#endif
constexpr int pf_for(int ntile) {
  return NECK_PF < 1 ? 1 : (NECK_PF > ntile ? ntile : NECK_PF);
}

// Tile shape per conv.  Kept as r1 measured it; it is now a constexpr function
// of the spec so the conv id can carry Cin/Cout/N into the kernel as constants.
struct Cfg { int nt, bm, bn, bk, wm, wn; };

constexpr Cfg cfg_wide(int id) {
  const ConvSpec s = kSpec[id];
  if (s.cout % 64) return {128, 32, 64, 32, 1, 4};       // Cout == 32
  if (kN[id] < 512 && s.cin % 64 == 0)                   // 20x20
    return {256, 64, 32, 64, 2, 4};
  if (s.cin % 64 == 0) return {256, 64, 64, 64, 2, 4};
  return {256, 64, 64, 32, 2, 4};                        // Cin == 96
}

// The narrow shape halves BM and BN, so it quarters the output tile and roughly
// halves the shared-memory traffic a block moves (BM*Ktot + Ktot*BN) at the cost
// of re-reading the weights in more blocks.  It wins whenever the wide shape
// leaves the machine empty and loses once the batch already supplies blocks:
// measured over the whole neck, all-narrow is 0.1258 ms at B=1 (vs 0.1546 wide)
// and 0.1873 ms at B=4 (vs 0.1751 wide).
constexpr Cfg cfg_narrow(int id) {
  const ConvSpec s = kSpec[id];
  if (s.cout % 64) return {128, 32, 32, 32, 1, 4};       // Cout == 32
  if (s.cin % 64 == 0) return {256, 32, 32, 64, 2, 4};
  return {128, 32, 32, 32, 1, 4};                        // Cin == 96
}

// Between the two: halve only BN.  Useful when the batch supplies enough blocks
// for the wide tile to beat all-narrow but not enough to fill a wave.
constexpr Cfg cfg_mid(int id) {
  const ConvSpec s = kSpec[id];
  if (s.cout % 64) return {128, 32, 32, 32, 1, 4};
  if (s.cin % 64 == 0) return {256, 64, 32, 64, 2, 4};
  return cfg_wide(id);                                   // Cin == 96
}

constexpr int nblocks(int id, Cfg f, int bn) {
  return (kSpec[id].cout / f.bm) * ((kN[id] + f.bn - 1) / f.bn) * bn;
}

// Threshold between the two, in blocks of the wide shape.  148 SMs hold one
// block each at these register counts, and the crossover measured between
// B=1 (25-50 wide blocks -> narrow wins) and B=4 (100-200 -> wide wins) sits
// below one full wave, because a block's cost is a latency chain rather than
// throughput: extra blocks are free until they stop being free.
constexpr int kWideMinBlocks = 96;
constexpr int kFullWideMinBlocks = 256;

// The middle tier (BN halved, BM kept) only pays for the 80x80 stage, whose
// weight tile is small next to its spatial extent, so halving BN cuts most of a
// block's shared-memory traffic.  For the 40x40 convs at the same block count
// the weights dominate that traffic and the full-wide tile stays ahead.
// Measured (B=4 / B=1): wide-above-96 0.1753/0.1238, mid-above-96
// 0.1772/0.1199, this rule keeps the better of each.
constexpr bool mid_helps(int id) { return kN[id] >= 4096; }

// Autotune hook (dev): -1 keeps the block-count rule above, 0/1/2 force the
// narrow/mid/wide tile everywhere.  Forcing lets both candidates be timed in one
// process, which is the only reliable way to compare them: this machine does not
// permit locking the SM clock (`nvidia-smi -lgc` is denied, and the harness's
// --lock-clocks is off), so it floats 120-1965 MHz and two bench runs minutes
// apart can differ by 1.5x on identical code.
int g_tier[NCONV] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

constexpr int64_t round_up(int64_t v, int64_t a) { return (v + a - 1) / a * a; }

constexpr int64_t wnumel(const ConvSpec& s) {
  return (int64_t)s.cout * (s.dw ? 1 : s.cin) * s.k * s.k;
}

// Offsets, in fp16 elements, of every folded weight and bias inside the single
// packed parameter tensor.  Python fills the tensor using exactly this map.
struct WLayout {
  int64_t woff[NCONV], boff[NCONV], total;
};

constexpr WLayout make_wlayout() {
  WLayout L{};
  int64_t o = 0;
  for (int i = 0; i < NCONV; ++i) {
    L.woff[i] = o;
    o = round_up(o + wnumel(kSpec[i]), 8);
  }
  for (int i = 0; i < NCONV; ++i) {
    L.boff[i] = o;
    o = round_up(o + kSpec[i].cout, 8);
  }
  L.total = o;
  return L;
}
constexpr WLayout kW = make_wlayout();

// Intermediate buffers, in fp16 elements per batch item.
// Only the concats whose *producer* can place its own output need a buffer.
// cat1/cat2/cat4 have no such producer for one of their halves, so instead of
// materializing them their consuming 1x1 conv reads the two sources directly
// (see SPLIT/UP1 in conv_kernel) -- that removes 3 buffers, ~21 MB of
// write+read traffic, and the 2 fill kernels that used to build them.
enum BufId {
  Y_A = 0,   // (192, 40, 40)  c2f_p4:  cv1 out | bottleneck out
  T_A,       // ( 64, 40, 40)
  CAT3,      // (192, 40, 40)  dp3 out | p4      (both halves producer-written)
  Y_B,       // ( 96, 80, 80)
  T_B,       // ( 32, 80, 80)
  Y_D,       // (192, 40, 40)
  T_D,       // ( 64, 40, 40)
  T_S,       // (128, 40, 40)
  T_S2,      // (128, 20, 20)  down_n4 out (the cat4 half e_cv1 reads as src1)
  Y_E,       // (384, 20, 20)  cv1 out | cib out
  T0, T1, T2, T3,
  NBUF
};

constexpr int kBufCh[NBUF] = {192, 64, 192, 96, 32, 192, 64, 128,
                              128, 384, 128, 256, 256, 128};
constexpr int kBufHW[NBUF] = {1600, 1600, 1600, 6400, 6400, 1600, 1600, 1600,
                              400, 400, 400, 400, 400, 400};

struct BufLayout {
  int64_t off[NBUF], per_batch;
};

constexpr BufLayout make_buflayout() {
  BufLayout L{};
  int64_t o = 0;
  for (int i = 0; i < NBUF; ++i) {
    L.off[i] = o;
    o += (int64_t)kBufCh[i] * kBufHW[i];
  }
  L.per_batch = o;
  return L;
}
constexpr BufLayout kB = make_buflayout();

__device__ __forceinline__ void mma_m16n8k16(float (&d)[4], const uint32_t (&a)[4],
                                             const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// A is row-major (M, K) in shared memory -> the m16n8k16 A fragment is exactly
// four 8x8 tiles, so one ldmatrix.x4 replaces four ld.shared.
__device__ __forceinline__ void ld_a(uint32_t (&a)[4], const __half* p) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(s));
}

// B stays K-major in shared memory (a straight copy of the activation layout);
// the transposing ldmatrix produces the .col B fragment directly, so the tile
// never has to be transposed on the way in.
__device__ __forceinline__ void ld_b(uint32_t (&b)[2], const __half* p) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
               : "=r"(b[0]), "=r"(b[1]) : "r"(s));
}

__device__ __forceinline__ float silu(float v) {
  return v * __frcp_rn(1.0f + __expf(-v));
}

// The K loop is software-pipelined two tiles deep: the global loads for tile
// i+2 are issued right after tile i's barrier, so their latency hides behind two
// tiles' worth of tensor-core work.  These problems are far too small to fill the
// machine by occupancy (~1.4 warps/scheduler), so hiding latency *inside* each
// warp is the only lever; a one-deep pipeline left ~400 cycles/tile exposed.
// Two shared-memory buffers still suffice: a warp only reaches the barrier of
// tile i+1 after issuing tile i-1's mma, so smem[i&1] is free to refill at i+2.
// SPLIT: the K axis is served by two tensors (an un-materialized concat), the
// first `split` channels from `in` and the rest from `in2`.  UP1: source 1 is
// read through a nearest-2x upsample, so 8 adjacent output columns come from 4
// adjacent source halves -- the upsample becomes an addressing rule instead of a
// tensor.  Both are 1x1-only (every concat in this neck feeds a 1x1 conv).
// NTILES (= K*K * Cin/BK, the number of K-tiles) is a *template* parameter, not
// derived from the runtime Cin.  That is what makes the K loop fully unrollable,
// and unrolling it is the whole point: with a runtime trip count the register
// stage index `it % PF` is dynamic, so nvcc demotes the staging arrays to local
// memory and any prefetch depth above 1 costs ~1.8x instead of hiding latency.
// With NTILES known, every stage index is a constant, the arrays stay in
// registers, and the compiler is free to hoist the independent global loads of
// later tiles above earlier tiles' barriers on its own.
template <int NT, int BM, int BN, int BK, int NTILES, int PF, int WM, int WN,
          int K, int STRIDE, bool ACT, bool RESID, bool SPLIT = false,
          bool UP1 = false>
__global__ __launch_bounds__(NT) void conv_kernel(
    const __half* __restrict__ in, const __half* __restrict__ Wt,
    const __half* __restrict__ bias, __half* __restrict__ out,
    const __half* __restrict__ resid, int Hin, int Win,
    int Hout, int Wout, int in_bstride, int out_bstride, int resid_bstride,
    const __half* __restrict__ in2 = nullptr, int in2_bstride = 0,
    int split = 0, int src1_hw = 0, int src1_w = 0,
    const int64_t* __restrict__ io = nullptr, int in_ix = -1, int in2_ix = -1,
    int out_ix = -1) {
  constexpr int KK = K * K;
  constexpr int NTILE = NTILES;
  constexpr int NCB = NTILE / KK;      // K-tiles per (kh, kw) slice
  constexpr int CIN = NCB * BK;
  constexpr int Ksrc = KK * CIN;       // weight row stride
  constexpr int PAD = K / 2;
  constexpr int TM = BM / WM, TN = BN / WN;
  constexpr int MF = TM / 16, NF = TN / 8, KF = BK / 16;
  constexpr int ASTRIDE = BK + 8, BSTRIDE = BN + 8;
  constexpr int AV = BM * BK / 8 / NT;   // uint4 per thread, A tile
  constexpr int AROW = NT / (BK / 8);    // A rows covered per pass
  constexpr int NPG = BN / 8;            // 8-wide activation groups per tile
  constexpr int GV = BK * NPG / NT;      // uint4 per thread, vectorized B path
  constexpr int LKN = BK * BN / NT;      // halves per thread, scalar B path
  constexpr int VEC = (K == 1);          // 1x1 => the B window is 8-aligned
  static_assert(WM * WN * 32 == NT && AV >= 1 && LKN >= 1 && GV >= 1, "tile cfg");
  static_assert(!SPLIT || VEC, "split inputs are 1x1-only");

  __shared__ __half As[2][BM * ASTRIDE];
  __shared__ __half Bs[2][BK * BSTRIDE];

  const int tid = threadIdx.x;
  const int m0 = blockIdx.x * BM, p0 = blockIdx.y * BN, bi = blockIdx.z;
  const int N = Hout * Wout, HWin = Hin * Win;

  // The 3 harness inputs and the 3 returned outputs sit at a *different*
  // address every call (`_ShiftingPool` shifts the inputs, `at::empty` hands
  // out fresh outputs), so the kernels that touch them fetch their base
  // pointer from a 6-entry device array rather than taking it as a baked-in
  // launch argument.  That is the whole trick that makes this 22-kernel path
  // capturable into a CUDA graph *once* and replayable against arbitrary new
  // I/O buffers -- see `publish()` below.  The array is 48 bytes, so the first
  // kernel's load pulls it into L2 and the rest hit; `out` is fetched here
  // rather than in the epilogue so its latency hides behind the K loop.
  if (in_ix >= 0) in = (const __half*)io[in_ix];
  if (out_ix >= 0) out = (__half*)io[out_ix];
  if (SPLIT && in2_ix >= 0) in2 = (const __half*)io[in2_ix];
  in += (int64_t)bi * in_bstride;
  out += (int64_t)bi * out_bstride;
  if (SPLIT) in2 += (int64_t)bi * in2_bstride;
  if (RESID) resid += (int64_t)bi * resid_bstride;

  const int am = tid / (BK / 8), ak = (tid % (BK / 8)) * 8;   // A-tile slot
  const int ln = tid % BN, lk = (tid / BN) * LKN;             // scalar B slot
  const int p = p0 + ln;
  int ho = 0, wo = 0;
  if (K > 1) {
    ho = p / Wout;
    wo = p - ho * Wout;
  }

  const int warp = tid / 32, lane = tid % 32;
  const int wm = warp / WN, wn = warp % WN;
  const int gid = lane >> 2, tig = lane & 3;
  const int lr = (lane & 7) + ((lane & 8) ? 8 : 0), alc = (lane & 16) ? 8 : 0;

  float acc[MF][NF][4];
#pragma unroll
  for (int i = 0; i < MF; ++i)
#pragma unroll
    for (int j = 0; j < NF; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;

  uint4 areg[PF][AV];
  uint4 bvec[PF][VEC ? GV : 1];
  __half breg[PF][VEC ? 1 : LKN];
  int gbase = p;
  bool gok = p < N;

  auto prep = [&](int khw) {
    if (K > 1) {
      const int kh = khw / K, kw = khw - kh * K;
      const int hi = ho * STRIDE + kh - PAD;
      const int wi = wo * STRIDE + kw - PAD;
      gok = p < N && (unsigned)hi < (unsigned)Hin && (unsigned)wi < (unsigned)Win;
      gbase = hi * Win + wi;
    }
  };
  auto loadA = [&](int st, int khw, int cb) {
    const __half* wtb = Wt + (int64_t)m0 * Ksrc + khw * CIN + cb + ak;
#pragma unroll
    for (int v = 0; v < AV; ++v)
      areg[st][v] = *(const uint4*)(wtb + (int64_t)(am + v * AROW) * Ksrc);
  };
  auto loadB = [&](int st, int cb) {
    if (VEC) {
#pragma unroll
      for (int v = 0; v < GV; ++v) {
        const int cix = tid + v * NT;
        const int ps = p0 + (cix % NPG) * 8;
        const int ch = cb + cix / NPG;
        if (ps >= N) {
          bvec[st][v] = make_uint4(0, 0, 0, 0);
        } else if (SPLIT && ch >= split) {
          bvec[st][v] =
              *(const uint4*)(in2 + (int64_t)(ch - split) * HWin + ps);
        } else if (UP1) {
          // 8 adjacent outputs sit in one row (Wout % 8 == 0), so they read 4
          // adjacent source halves; duplicate each into the tile.
          const int hoq = ps / Wout, woq = ps - hoq * Wout;
          const uint2 q = *(const uint2*)(in + (int64_t)ch * src1_hw +
                                          (int64_t)(hoq >> 1) * src1_w +
                                          (woq >> 1));
          const __half* h = (const __half*)&q;
          __half o[8];
#pragma unroll
          for (int i = 0; i < 4; ++i) { o[2 * i] = h[i]; o[2 * i + 1] = h[i]; }
          bvec[st][v] = *(const uint4*)o;
        } else {
          bvec[st][v] = *(const uint4*)(in + (int64_t)ch * HWin + ps);
        }
      }
    } else if (gok) {
      const __half* ib = in + gbase + (int64_t)(cb + lk) * HWin;
#pragma unroll
      for (int q = 0; q < LKN; ++q) breg[st][q] = ib[(int64_t)q * HWin];
    } else {
#pragma unroll
      for (int q = 0; q < LKN; ++q) breg[st][q] = __float2half(0.f);
    }
  };
  auto store = [&](int st, int sm) {
#pragma unroll
    for (int v = 0; v < AV; ++v)
      *(uint4*)(As[sm] + (am + v * AROW) * ASTRIDE + ak) = areg[st][v];
    if (VEC) {
#pragma unroll
      for (int v = 0; v < GV; ++v) {
        const int cix = tid + v * NT;
        *(uint4*)(Bs[sm] + (cix / NPG) * BSTRIDE + (cix % NPG) * 8) = bvec[st][v];
      }
    } else {
      __half* bs = Bs[sm] + lk * BSTRIDE + ln;
#pragma unroll
      for (int q = 0; q < LKN; ++q) bs[q * BSTRIDE] = breg[st][q];
    }
  };

  auto fetch = [&](int st, int tile) {
    prep(tile / NCB);
    loadA(st, tile / NCB, (tile % NCB) * BK);
    loadB(st, (tile % NCB) * BK);
  };
#pragma unroll
  for (int s = 0; s < PF; ++s)
    if (s < NTILE) fetch(s, s);

#pragma unroll
  for (int it = 0; it < NTILE; ++it) {
    const int st = PF == 1 ? 0 : it % PF;  // register stage
    const int sm = it & 1;                 // shared-memory buffer
    store(st, sm);
    __syncthreads();
    if (it + PF < NTILE) fetch(st, it + PF);  // PF tiles of mma to hide behind

#pragma unroll
    for (int kf = 0; kf < KF; ++kf) {
      uint32_t af[MF][4], bf[NF][2];
#pragma unroll
      for (int i = 0; i < MF; ++i)
        ld_a(af[i], As[sm] + (wm * TM + i * 16 + lr) * ASTRIDE + kf * 16 + alc);
#pragma unroll
      for (int j = 0; j < NF; ++j)
        ld_b(bf[j], Bs[sm] + (kf * 16 + lr) * BSTRIDE + wn * TN + j * 8);
#pragma unroll
      for (int i = 0; i < MF; ++i)
#pragma unroll
        for (int j = 0; j < NF; ++j) mma_m16n8k16(acc[i][j], af[i], bf[j]);
    }
  }

  // --- epilogue: bias, SiLU, optional residual, direct write into the slice
  const int HWout = Hout * Wout;
#pragma unroll
  for (int i = 0; i < MF; ++i) {
#pragma unroll
    for (int j = 0; j < NF; ++j) {
      const int nn = p0 + wn * TN + j * 8 + tig * 2;
      if (nn >= N) continue;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int co = m0 + wm * TM + i * 16 + gid + h * 8;
        const float bv = __half2float(bias[co]);
        __half o[2];
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          float v = acc[i][j][h * 2 + q] + bv;
          if (ACT) v = silu(v);
          if (RESID && nn + q < N)
            v += __half2float(resid[(int64_t)co * HWout + nn + q]);
          o[q] = __float2half(v);
        }
        __half* dst = out + (int64_t)co * HWout + nn;
        if (nn + 1 < N) {
          *(uint32_t*)dst = *(const uint32_t*)o;
        } else {
          dst[0] = o[0];
        }
      }
    }
  }
}

// ===========================================================================
// Depthwise conv: one (batch, channel) plane per block, plane staged in shared
// memory.  Tiny FLOP count, so this is a plain scalar kernel with the epilogue
// fused in.
// ===========================================================================
// One (batch, channel) plane per block, staged in shared memory.  The FLOP count
// here is negligible, so this is a plain scalar kernel with the epilogue fused
// in.  (Measured dead ends: staging the plane as fp32 to skip the per-tap
// half->float converts cost +1.0us, and register-tiling 4 outputs per thread to
// share an input row cost +1.9us -- both kernels are already at the ~3.6us
// per-launch floor, so extra registers only hurt.)
template <int K, int STRIDE, bool ACT, bool RESID>
__global__ __launch_bounds__(256) void dw_kernel(
    const __half* __restrict__ in, const __half* __restrict__ Wt,
    const __half* __restrict__ bias, __half* __restrict__ out,
    const __half* __restrict__ resid, int Hin, int Win, int Hout, int Wout,
    int in_bstride, int out_bstride, int resid_bstride) {
  constexpr int PAD = K / 2;
  __shared__ __half plane[1600];
  __shared__ __half wsh[K * K];

  const int c = blockIdx.x, bi = blockIdx.y;
  const int HWin = Hin * Win, HWout = Hout * Wout;
  in += (int64_t)bi * in_bstride + (int64_t)c * HWin;
  out += (int64_t)bi * out_bstride + (int64_t)c * HWout;
  if (RESID) resid += (int64_t)bi * resid_bstride + (int64_t)c * HWout;

  for (int i = threadIdx.x; i < HWin; i += 256) plane[i] = in[i];
  for (int i = threadIdx.x; i < K * K; i += 256) wsh[i] = Wt[c * K * K + i];
  __syncthreads();

  const float bv = __half2float(bias[c]);
  for (int p = threadIdx.x; p < HWout; p += 256) {
    const int ho = p / Wout, wo = p - ho * Wout;
    float v = bv;
#pragma unroll
    for (int kh = 0; kh < K; ++kh) {
      const int hi = ho * STRIDE + kh - PAD;
      if ((unsigned)hi >= (unsigned)Hin) continue;
#pragma unroll
      for (int kw = 0; kw < K; ++kw) {
        const int wi = wo * STRIDE + kw - PAD;
        if ((unsigned)wi >= (unsigned)Win) continue;
        v = fmaf(__half2float(plane[hi * Win + wi]),
                 __half2float(wsh[kh * K + kw]), v);
      }
    }
    if (ACT) v = silu(v);
    if (RESID) v += __half2float(resid[p]);
    out[p] = __float2half(v);
  }
}

// ===========================================================================
// Host side
// ===========================================================================
struct Ctx {
  const __half* w;
  __half* ws;
  const int64_t* io;   // device array of the 6 harness-facing addresses
  int bn;
  cudaStream_t st;
  mutable int budget;  // debug: launch only the first N kernels (-1 = all)
};

// Returns false when the debug launch budget is exhausted.
bool take(const Ctx& c) {
  if (c.budget < 0) return true;
  if (c.budget == 0) return false;
  --c.budget;
  return true;
}


// Each buffer owns a contiguous [bn, ch, hw] region; kB.off is in per-batch
// elements, so the region base scales with bn.
__half* bufp(const Ctx& c, BufId id, int ch_off = 0) {
  return c.ws + kB.off[id] * c.bn + (int64_t)ch_off * kBufHW[id];
}
int bstride(BufId id) { return kBufCh[id] * kBufHW[id]; }

// Launch a dense conv.  ID is a template parameter so Cin, Cout, N and the
// whole tile shape reach the kernel as compile-time constants -- that is what
// makes NTILES (and therefore the K loop's trip count) constant.  BM always
// divides Cout exactly (every Cout here is a multiple of 32), so no M masking.
template <int ID, int K, int STRIDE, bool ACT, bool RESID, int NT, int BM,
          int BN, int BK, int WM, int WN>
void dense_impl(const Ctx& c, const __half* in, __half* out,
                const __half* resid, int Hin, int Win, int Hout, int Wout,
                int in_bs, int out_bs, int resid_bs, int in_ix, int out_ix) {
  constexpr ConvSpec s = kSpec[ID];
  constexpr int NTILES = K * K * s.cin / BK;
  static_assert(s.cin % BK == 0, "BK must divide Cin");
  dim3 g(s.cout / BM, (kN[ID] + BN - 1) / BN, c.bn);
  conv_kernel<NT, BM, BN, BK, NTILES, pf_for(NTILES), WM, WN, K, STRIDE, ACT,
              RESID><<<g, NT, 0, c.st>>>(
      in, c.w + kW.woff[ID], c.w + kW.boff[ID], out, resid, Hin, Win, Hout,
      Wout, in_bs, out_bs, resid_bs, nullptr, 0, 0, 0, 0, c.io, in_ix, -1,
      out_ix);
}

#define DENSE_CALL(F)                                                          \
  dense_impl<ID, K, STRIDE, ACT, RESID, (F).nt, (F).bm, (F).bn, (F).bk,         \
             (F).wm, (F).wn>(c, in, out, resid, Hin, Win, Hout, Wout, in_bs,    \
                             out_bs, resid_bs, in_ix, out_ix)

template <int ID, int K, int STRIDE, bool ACT, bool RESID>
void dense(const Ctx& c, const __half* in, __half* out, const __half* resid,
           int Hin, int Win, int Hout, int Wout, int in_bs, int out_bs,
           int resid_bs, int in_ix = -1, int out_ix = -1) {
  if (!take(c)) return;
  constexpr Cfg W = cfg_wide(ID), M_ = cfg_mid(ID), N_ = cfg_narrow(ID);
  const int nb = nblocks(ID, W, c.bn);
  const int tier =
      g_tier[ID] >= 0
          ? g_tier[ID]
          : (nb >= (mid_helps(ID) ? kFullWideMinBlocks : kWideMinBlocks)
                 ? 2
                 : (nb >= kWideMinBlocks ? 1 : 0));
  if (tier == 2) {
    DENSE_CALL(W);
  } else if (tier == 1) {
    DENSE_CALL(M_);
  } else {
    DENSE_CALL(N_);
  }
}
#undef DENSE_CALL

// 1x1 conv over an un-materialized concat of two tensors; UP1 applies a
// nearest-2x upsample to source 1 on the fly.
template <int ID, bool UP1, int NT, int BM, int BN, int BK, int WM, int WN>
void split_impl(const Ctx& c, const __half* in1, const __half* in2,
                __half* out, int Hout, int Wout, int split, int in1_bs,
                int in2_bs, int out_bs, int src1_hw, int src1_w, int in_ix,
                int in2_ix) {
  constexpr ConvSpec s = kSpec[ID];
  constexpr int NTILES = s.cin / BK;
  static_assert(s.cin % BK == 0, "BK must divide Cin");
  dim3 g(s.cout / BM, (kN[ID] + BN - 1) / BN, c.bn);
  conv_kernel<NT, BM, BN, BK, NTILES, pf_for(NTILES), WM, WN, 1, 1, true, false,
              true, UP1><<<g, NT, 0, c.st>>>(
      in1, c.w + kW.woff[ID], c.w + kW.boff[ID], out, nullptr, Hout, Wout, Hout,
      Wout, in1_bs, out_bs, 0, in2, in2_bs, split, src1_hw, src1_w, c.io, in_ix,
      in2_ix, -1);
}

#define SPLIT_CALL(F)                                                          \
  split_impl<ID, UP1, (F).nt, (F).bm, (F).bn, (F).bk, (F).wm, (F).wn>(          \
      c, in1, in2, out, Hout, Wout, split, in1_bs, in2_bs, out_bs, src1_hw,     \
      src1_w, in_ix, in2_ix)

template <int ID, bool UP1>
void dense_split(const Ctx& c, const __half* in1, const __half* in2,
                 __half* out, int Hout, int Wout, int split, int in1_bs,
                 int in2_bs, int out_bs, int src1_hw, int src1_w,
                 int in_ix = -1, int in2_ix = -1) {
  if (!take(c)) return;
  constexpr Cfg W = {256, 64, 64, 64, 2, 4}, M_ = {256, 64, 32, 64, 2, 4},
               N_ = {256, 32, 32, 64, 2, 4};
  const int nb = nblocks(ID, W, c.bn);
  const int tier =
      g_tier[ID] >= 0
          ? g_tier[ID]
          : (nb >= (mid_helps(ID) ? kFullWideMinBlocks : kWideMinBlocks)
                 ? 2
                 : (nb >= kWideMinBlocks ? 1 : 0));
  if (tier == 2) {
    SPLIT_CALL(W);
  } else if (tier == 1) {
    SPLIT_CALL(M_);
  } else {
    SPLIT_CALL(N_);
  }
}
#undef SPLIT_CALL

template <int ID, int K, int STRIDE, bool ACT, bool RESID>
void depthwise(const Ctx& c, const __half* in, __half* out,
               const __half* resid, int Hin, int Win, int Hout, int Wout,
               int in_bs, int out_bs, int resid_bs) {
  if (!take(c)) return;
  dim3 g(kSpec[ID].cout, c.bn);
  dw_kernel<K, STRIDE, ACT, RESID><<<g, 256, 0, c.st>>>(
      in, c.w + kW.woff[ID], c.w + kW.boff[ID], out, resid, Hin, Win, Hout,
      Wout, in_bs, out_bs, resid_bs);
}

// ===========================================================================
// The whole neck, issued onto c.st.  Split out of the entry points so that the
// *identical* launch sequence can be either executed directly or recorded once
// into a CUDA graph.  Everything the sequence touches is at a fixed address
// (the packed weights and the workspace are allocated once) except the 3
// harness inputs and the 3 returned outputs, which are reached indirectly
// through c.io -- so one recording stays valid for every later call.
// ===========================================================================
void issue(const Ctx& c) {
  // --- c2f_p4 (40x40): cat1 = up2(p5b) | p4b is read, never built ---------
  dense_split<A_CV1, true>(c, nullptr, nullptr, bufp(c, Y_A), 40, 40, 256,
                    256 * 400, 128 * 1600, bstride(Y_A), 400, 20,
                    IO_P5B, IO_P4B);
  dense<A_B1, 3, 1, true, false>(c, bufp(c, Y_A, 64), bufp(c, T_A), nullptr, 40,
                           40, 40, 40, bstride(Y_A), bstride(T_A), 0);
  dense<A_B2, 3, 1, true, false>(c, bufp(c, T_A), bufp(c, Y_A, 128), nullptr,
                           40, 40, 40, 40, bstride(T_A), bstride(Y_A), 0);
  // p4 lands straight in its slot in the cat3 buffer.
  dense<A_CV2, 1, 1, true, false>(c, bufp(c, Y_A), bufp(c, CAT3, 64), nullptr,
                           40, 40, 40, 40, bstride(Y_A), bstride(CAT3), 0);

  // --- c2f_p3 (80x80): cat2 = up2(p4) | p3b ------------------------------
  dense_split<B_CV1, true>(c, bufp(c, CAT3, 64), nullptr, bufp(c, Y_B), 80, 80,
                    128, bstride(CAT3), 64 * 6400, bstride(Y_B), 1600, 40,
                    -1, IO_P3B);
  dense<B_B1, 3, 1, true, false>(c, bufp(c, Y_B, 32), bufp(c, T_B), nullptr, 80,
                           80, 80, 80, bstride(Y_B), bstride(T_B), 0);
  dense<B_B2, 3, 1, true, false>(c, bufp(c, T_B), bufp(c, Y_B, 64), nullptr, 80,
                           80, 80, 80, bstride(T_B), bstride(Y_B), 0);
  dense<B_CV2, 1, 1, true, false>(c, bufp(c, Y_B), nullptr, nullptr, 80, 80, 80,
                           80, bstride(Y_B), 64 * 6400, 0, -1, IO_P3);

  // --- c2f_n4 (40x40) ----------------------------------------------------
  dense<DP3, 3, 2, true, false>(c, nullptr, bufp(c, CAT3), nullptr, 80, 80, 40,
                           40, 64 * 6400, bstride(CAT3), 0, IO_P3);
  dense<D_CV1, 1, 1, true, false>(c, bufp(c, CAT3), bufp(c, Y_D), nullptr, 40,
                           40, 40, 40, bstride(CAT3), bstride(Y_D), 0);
  dense<D_B1, 3, 1, true, false>(c, bufp(c, Y_D, 64), bufp(c, T_D), nullptr, 40,
                           40, 40, 40, bstride(Y_D), bstride(T_D), 0);
  dense<D_B2, 3, 1, true, false>(c, bufp(c, T_D), bufp(c, Y_D, 128), nullptr,
                           40, 40, 40, 40, bstride(T_D), bstride(Y_D), 0);
  dense<D_CV2, 1, 1, true, false>(c, bufp(c, Y_D), nullptr, nullptr, 40, 40, 40,
                           40, bstride(Y_D), 128 * 1600, 0, -1, IO_N4);

  // --- SCDown + c2fcib_n5 (20x20): cat4 = down_n4 out | p5b --------------
  dense<S_CV1, 1, 1, true, false>(c, nullptr, bufp(c, T_S), nullptr, 40, 40, 40,
                           40, 128 * 1600, bstride(T_S), 0, IO_N4);
  depthwise<S_CV2, 3, 2, false, false>(c, bufp(c, T_S), bufp(c, T_S2), nullptr,
                                40, 40, 20, 20, bstride(T_S), bstride(T_S2), 0);
  dense_split<E_CV1, false>(c, bufp(c, T_S2), nullptr, bufp(c, Y_E), 20, 20, 128,
                     bstride(T_S2), 256 * 400, bstride(Y_E), 0, 0, -1, IO_P5B);
  depthwise<I0, 3, 1, true, false>(c, bufp(c, Y_E, 128), bufp(c, T0), nullptr,
                               20, 20, 20, 20, bstride(Y_E), bstride(T0), 0);
  dense<I1, 1, 1, true, false>(c, bufp(c, T0), bufp(c, T1), nullptr, 20, 20, 20,
                           20, bstride(T0), bstride(T1), 0);
  depthwise<I2, 7, 1, true, false>(c, bufp(c, T1), bufp(c, T2), nullptr, 20, 20,
                               20, 20, bstride(T1), bstride(T2), 0);
  dense<I3, 1, 1, true, false>(c, bufp(c, T2), bufp(c, T3), nullptr, 20, 20, 20,
                           20, bstride(T2), bstride(T3), 0);
  depthwise<I4, 3, 1, true, true>(c, bufp(c, T3), bufp(c, Y_E, 256),
                              bufp(c, Y_E, 128), 20, 20, 20, 20, bstride(T3),
                              bstride(Y_E), bstride(Y_E));
  dense<E_CV2, 1, 1, true, false>(c, bufp(c, Y_E), nullptr, nullptr, 20, 20, 20,
                           20, bstride(Y_E), 256 * 400, 0, -1, IO_N5);
}

// Rotating slot in the pinned host ring `publish` copies from.  The CPU runs
// many iterations ahead of the GPU in the harness timing loop, so the slot a
// still-queued copy has to read must not have been overwritten yet; one lap of
// the ring is far longer than the driver's pending-launch depth.
int g_slot = 0;

Ctx make_ctx(const at::Tensor& wpack, const at::Tensor& ws,
             const at::Tensor& io_d, int bn, int budget) {
  Ctx c;
  c.w = (const __half*)wpack.data_ptr();
  c.ws = (__half*)ws.data_ptr();
  c.io = (const int64_t*)io_d.data_ptr();
  c.bn = bn;
  c.st = at::cuda::getCurrentCUDAStream();
  c.budget = budget;
  return c;
}

}  // namespace

// ---------------------------------------------------------------------------
std::vector<int64_t> weight_layout() {
  std::vector<int64_t> v;
  for (int i = 0; i < NCONV; ++i) {
    v.push_back(kW.woff[i]);
    v.push_back(wnumel(kSpec[i]));
    v.push_back(kW.boff[i]);
    v.push_back(kSpec[i].cout);
  }
  v.push_back(kW.total);
  return v;
}

int64_t ws_elems(int64_t bn) { return kB.per_batch * bn; }
int64_t io_slots() { return NIO; }
// --- autotune hooks -------------------------------------------------------
// Not used by `forward`; they exist so a dev script can force a tile shape and
// time two candidates *inside one process*.  That matters here because the SM
// clock cannot be locked on this machine and floats 1.6x between runs, so
// comparing separate bench runs is unreliable (see ITERATIONS.md).
int64_t nconv() { return NCONV; }
int64_t pf_depth() { return NECK_PF; }
void set_tier(int64_t t) {
  for (int i = 0; i < NCONV; ++i) g_tier[i] = (int)t;
}
void set_tiers(std::vector<int64_t> t) {
  TORCH_CHECK((int)t.size() == NCONV, "tier table must have NCONV entries");
  for (int i = 0; i < NCONV; ++i) g_tier[i] = (int)t[i];
}

// Allocate this call's 3 outputs and stage the 6 harness-facing addresses into
// the device array the recorded kernels dereference.  This is everything a
// graph replay cannot do for itself: 3 caching-allocator hits, 6 host stores
// and one 48-byte H2D copy, all stream-ordered ahead of the replay.
std::vector<at::Tensor> publish(at::Tensor p3b, at::Tensor p4b, at::Tensor p5b,
                                at::Tensor io_d, at::Tensor io_h) {
  const int bn = (int)p3b.size(0);
  auto opts = p3b.options();
  at::Tensor p3 = at::empty({bn, 64, 80, 80}, opts);
  at::Tensor n4 = at::empty({bn, 128, 40, 40}, opts);
  at::Tensor n5 = at::empty({bn, 256, 20, 20}, opts);

  const int64_t nslot = io_h.numel() / NIO;
  const int64_t slot = g_slot;
  g_slot = (int)((slot + 1) % nslot);
  int64_t* h = io_h.data_ptr<int64_t>() + slot * NIO;
  h[IO_P3B] = (int64_t)p3b.data_ptr();
  h[IO_P4B] = (int64_t)p4b.data_ptr();
  h[IO_P5B] = (int64_t)p5b.data_ptr();
  h[IO_P3] = (int64_t)p3.data_ptr();
  h[IO_N4] = (int64_t)n4.data_ptr();
  h[IO_N5] = (int64_t)n5.data_ptr();
  CUDA_CHECK(cudaMemcpyAsync(io_d.data_ptr(), h, NIO * sizeof(int64_t),
                             cudaMemcpyHostToDevice,
                             at::cuda::getCurrentCUDAStream()));
  return {p3, n4, n5};
}

// The body of a graph capture: the 22 launches and nothing else.  No
// allocation and no memcpy happen here, so the recording contains only kernel
// nodes and stays valid for any later `publish`.
void capture_body(at::Tensor wpack, at::Tensor ws, at::Tensor io_d, int64_t bn,
                  int64_t budget) {
  issue(make_ctx(wpack, ws, io_d, (int)bn, (int)budget));
}

// Direct (un-captured) path: used for the first call of a shape, for the
// capture warm-up, and whenever graph capture is unavailable.
std::vector<at::Tensor> neck_forward(at::Tensor p3b, at::Tensor p4b,
                                     at::Tensor p5b, at::Tensor wpack,
                                     at::Tensor ws, at::Tensor io_d,
                                     at::Tensor io_h, int64_t budget) {
  auto outs = publish(p3b, p4b, p5b, io_d, io_h);
  issue(make_ctx(wpack, ws, io_d, (int)p3b.size(0), (int)budget));
  return outs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("weight_layout", &weight_layout);
  m.def("ws_elems", &ws_elems);
  m.def("io_slots", &io_slots);
  m.def("nconv", &nconv);
  m.def("pf_depth", &pf_depth);
  m.def("set_tier", &set_tier);
  m.def("set_tiers", &set_tiers);
  m.def("forward", &neck_forward);
  m.def("publish", &publish);
  m.def("capture_body", &capture_body);
}
