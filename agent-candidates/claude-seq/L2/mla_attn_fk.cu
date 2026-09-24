// Fused MLA dense-prefill kernels for the fastkernels L2 `mla_attention_impl` op.
//
// The reference path is three library calls per forward:
//   kv = F.linear(kv_c_normed, kv_b_proj.weight)   ->  [N, H*(128+128)]
//   k  = cat(k_nope, k_pe.expand(H))               ->  [N, H, 192]  (two strided copies)
//   o  = flash_attn_varlen(q, k, v, causal)        ->  [N, H, 128]
//
// On the short sequences an MLA layer actually sees (the captured shapes are
// 1, 26..88 and 443 new tokens plus one 16384 prefill) that is almost all
// dispatch: a couple of microseconds of arithmetic behind ~30 us of host work,
// most of it inside FA4's CuTeDSL launcher.  Up to `_FK_MAX_TOKENS` tokens the
// whole operator is therefore two kernels behind one pybind call:
//
//   `proj_kernel`  kv_c_normed[N,512] x W[H*256,512]^T -> k_nope as [H,Np,128]
//                  and v as *transposed* [H,128,Np], so the attention kernel's
//                  V operand (which `mma` wants n-major) is a contiguous run of
//                  keys and needs no shared-memory transpose.
//   `attn_kernel`  causal attention that never materializes `k`: the RoPE half
//                  is read straight from `k_pe` -- one row per token, shared by
//                  every head instead of broadcast into a [N,H,192] buffer --
//                  into the same shared tile as the NoPE half.  Launched with
//                  programmatic stream serialization so its blocks are resident
//                  (and the `q` tile already in flight) before the projection
//                  drains.
//
// Two more shapes get their own treatment: a single new token skips attention
// entirely (`vonly_kernel` -- see there), and beyond `_FK_MAX_TOKENS`, where
// cuBLAS and FA4 both beat plain `mma.sync`, only the concatenation is replaced
// (`build_k_kernel`).
//
// Both mma kernels are m16n8k16 bf16 with fp32 accumulation, `ldmatrix` operand
// loads and `cp.async` staging; k_nope/v are rounded to bf16 before the
// attention mma so the numerics follow the reference, whose up-projection also
// lands in bf16.
//
// Tokens are padded to a multiple of 64 in the `k_nope`/`v` scratch and the
// padding is *zero* (the projection zero-fills those rows of its A tile), so
// the attention kernel needs no bounds checks on the key axis -- padded keys
// are masked out of the softmax, and a zero (rather than uninitialized) V row
// keeps NaNs out of the accumulator either way.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <algorithm>

namespace fkmla {

typedef __nv_bfloat16 bf16;

static constexpr int DNOPE = 128;   // qk_nope_head_dim
static constexpr int DPE = 64;      // qk_rope_head_dim
static constexpr int DQK = DNOPE + DPE;
static constexpr int DV = 128;      // v_head_dim
static constexpr int BM = 64;       // tokens per block (both kernels)
static constexpr int BN = 64;       // keys per attention step

#define MMA_16816(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1)                     \
  asm volatile(                                                               \
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "                  \
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"               \
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)                                \
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ uint32_t ld32(const bf16* p) {
  return *reinterpret_cast<const uint32_t*>(p);
}

// One `ldmatrix.x4` in place of four 32-bit shared loads.  Both operand shapes
// this file needs map onto it: an A fragment is a 16x16 row-major tile, and a B
// fragment pair is two 8(n)x16(k) tiles out of an n-major tile -- the per-lane
// address is the only thing that differs, so the mma inner loops go from
// 12 shared loads per k-step to 3.
__device__ __forceinline__ void ldm_x4(uint32_t* r, const bf16* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
// Lane address for a 16x16 A tile based at (row, col) with row stride `stride`.
__device__ __forceinline__ int ldm_a_off(int lane, int row, int col, int stride) {
  return (row + (lane & 15)) * stride + col + ((lane & 16) >> 1);
}
// Lane address for the B fragments of two adjacent n-tiles based at n-row
// `row` (= 8*n_tile), reduction offset `col`.
__device__ __forceinline__ int ldm_b_off(int lane, int row, int col, int stride) {
  return (row + (lane & 7) + ((lane & 16) >> 1)) * stride + col + (lane & 8);
}

// 16 B global->shared copy.  ``keep=false`` zero-fills the destination instead
// (cp.async's src-size form), which is how out-of-range token rows are handled
// without a branch around the whole tile load.
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool keep) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  const int bytes = keep ? 16 : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::"r"(s),
               "l"(src), "r"(bytes));
}
#define CP_COMMIT() asm volatile("cp.async.commit_group;\n" ::)
#define CP_WAIT(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))

// ---------------------------------------------------------------------------
// Up-projection.  C = kv_c_normed @ W^T, split per head into k_nope and v.
//
// grid = (Np/BM, 4*H): blockIdx.y picks the head and which 64-wide quarter of
// that head's 256 output columns this block owns -- quarters 0/1 are k_nope,
// 2/3 are v.  Quarter boundaries never straddle the nope/v split, so the
// epilogue is one branch rather than a per-element test.
// ---------------------------------------------------------------------------
template <int LORA, int BN_>
__global__ __launch_bounds__(128) void proj_kernel(
    const bf16* __restrict__ kvc,   // [N, LORA]
    const bf16* __restrict__ W,     // [H*256, LORA]
    bf16* __restrict__ kbuf,        // [H, Np, DNOPE]
    bf16* __restrict__ vT,          // [H, DV, Np]
    int N, int Np) {
  // The whole K extent is staged in one shot -- (BM+BN_)*LORA/8 `cp.async`s with
  // a single wait -- because the 4 MB weight read is latency-bound, not
  // bandwidth-bound, at these tile counts: one HBM round trip per K tile *is*
  // the kernel (9.7 us measured for a BK=64 two-stage pipeline against the same
  // arithmetic).  For the same reason BN_ is kept narrow: the block count, not
  // the arithmetic, is what decides how much of the weight read is in flight.
  constexpr int AS = LORA + 8;    // padded row stride: 32-bit frag loads hit 32 banks
  constexpr int CSK = BN_ + 8;    // k_nope staging: [token][dim]
  constexpr int CSV = BM + 8;     // v staging: [dim][token]
  constexpr int CPH = 256 / BN_;  // column tiles per head
  extern __shared__ __align__(16) char smem_raw[];
  bf16* As = reinterpret_cast<bf16*>(smem_raw);
  bf16* Bs = As + BM * AS;
  bf16* Cs = As;                  // reused after the mma phase

  const int tok0 = blockIdx.x * BM;
  const int h = blockIdx.y / CPH;
  const int col0 = (blockIdx.y % CPH) * BN_;
  const bool is_v = col0 >= DNOPE;
  const int dim = is_v ? col0 - DNOPE : col0;

  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int mrow = warp * 16;     // 4 warps x 16 rows = BM

#pragma unroll
  for (int i = 0; i < BM * (LORA / 8) / 128; ++i) {
    const int u = tid + i * 128;
    const int r = u / (LORA / 8), c = (u % (LORA / 8)) * 8;
    const int tk = tok0 + r;
    const bool ok = tk < N;
    cp_async16(&As[r * AS + c], kvc + (size_t)(ok ? tk : 0) * LORA + c, ok);
  }
#pragma unroll
  for (int i = 0; i < BN_ * (LORA / 8) / 128; ++i) {
    const int u = tid + i * 128;
    const int r = u / (LORA / 8), c = (u % (LORA / 8)) * 8;
    cp_async16(&Bs[r * AS + c],
               W + (size_t)(h * 256 + col0 + r) * LORA + c, true);
  }
  CP_COMMIT();

  float acc[BN_ / 8][4];
#pragma unroll
  for (int ni = 0; ni < BN_ / 8; ++ni)
#pragma unroll
    for (int x = 0; x < 4; ++x) acc[ni][x] = 0.f;

  CP_WAIT(0);
  __syncthreads();

#pragma unroll 4
  for (int ks = 0; ks < LORA / 16; ++ks) {
    const int kb = ks * 16;
    uint32_t a[4];
    ldm_x4(a, &As[ldm_a_off(lane, mrow, kb, AS)]);
#pragma unroll
    for (int ni = 0; ni < BN_ / 8; ni += 2) {
      uint32_t b[4];
      ldm_x4(b, &Bs[ldm_b_off(lane, ni * 8, kb, AS)]);
      MMA_16816(acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3], a[0], a[1], a[2],
                a[3], b[0], b[1]);
      MMA_16816(acc[ni + 1][0], acc[ni + 1][1], acc[ni + 1][2], acc[ni + 1][3],
                a[0], a[1], a[2], a[3], b[2], b[3]);
    }
  }

  __syncthreads();
  if (!is_v) {
#pragma unroll
    for (int ni = 0; ni < BN_ / 8; ++ni) {
      const int c = ni * 8 + 2 * t;
      Cs[(mrow + g) * CSK + c] = __float2bfloat16_rn(acc[ni][0]);
      Cs[(mrow + g) * CSK + c + 1] = __float2bfloat16_rn(acc[ni][1]);
      Cs[(mrow + g + 8) * CSK + c] = __float2bfloat16_rn(acc[ni][2]);
      Cs[(mrow + g + 8) * CSK + c + 1] = __float2bfloat16_rn(acc[ni][3]);
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < BM * (BN_ / 8) / 128; ++i) {
      const int u = tid + i * 128;
      const int r = u / (BN_ / 8), c = (u % (BN_ / 8)) * 8;
      *reinterpret_cast<uint4*>(kbuf + ((size_t)h * Np + tok0 + r) * DNOPE + dim + c) =
          *reinterpret_cast<const uint4*>(&Cs[r * CSK + c]);
    }
  } else {
    // v half: staged transposed so each store to vT[h][dim][token] is one
    // contiguous run of tokens.
#pragma unroll
    for (int ni = 0; ni < BN_ / 8; ++ni) {
      const int c = ni * 8 + 2 * t;
      Cs[c * CSV + mrow + g] = __float2bfloat16_rn(acc[ni][0]);
      Cs[(c + 1) * CSV + mrow + g] = __float2bfloat16_rn(acc[ni][1]);
      Cs[c * CSV + mrow + g + 8] = __float2bfloat16_rn(acc[ni][2]);
      Cs[(c + 1) * CSV + mrow + g + 8] = __float2bfloat16_rn(acc[ni][3]);
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < BN_ * (BM / 8) / 128; ++i) {
      const int u = tid + i * 128;
      const int d = u / (BM / 8), c = (u % (BM / 8)) * 8;
      *reinterpret_cast<uint4*>(vT + ((size_t)h * DV + dim + d) * Np + tok0 + c) =
          *reinterpret_cast<const uint4*>(&Cs[d * CSV + c]);
    }
  }
}

// ---------------------------------------------------------------------------
// Single new token (all sequences length 1).  Causal softmax over one key is
// exactly 1, so the attention output *is* v -- the whole operator collapses to
// the v half of the up-projection, o[h*DV+d] = sum_k kv_c[k] * W[h*256+DNOPE+d][k].
// One warp per output row keeps the 2 MB weight slice fully in flight.
// ---------------------------------------------------------------------------
template <int LORA>
__global__ __launch_bounds__(128) void vonly_kernel(
    const bf16* __restrict__ kvc,   // [1, LORA]
    const bf16* __restrict__ W,     // [H*256, LORA]
    bf16* __restrict__ out,         // [1, H*DV]
    int H) {
  const int row = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (row >= H * DV) return;
  const int lane = threadIdx.x & 31;
  const int h = row / DV, d = row - h * DV;
  const bf16* wp = W + (size_t)(h * 256 + DNOPE + d) * LORA + lane * 8;
  const bf16* ap = kvc + lane * 8;
  float sum = 0.f;
#pragma unroll
  for (int i = 0; i < LORA / 256; ++i) {
    const uint4 wv = *reinterpret_cast<const uint4*>(wp + i * 256);
    const uint4 av = *reinterpret_cast<const uint4*>(ap + i * 256);
    const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv);
    const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&av);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 wf = __bfloat1622float2(w2[j]);
      const float2 af = __bfloat1622float2(a2[j]);
      sum = fmaf(wf.x, af.x, sum);
      sum = fmaf(wf.y, af.y, sum);
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
  if (lane == 0) out[row] = __float2bfloat16_rn(sum);
}

// ---------------------------------------------------------------------------
// Causal MLA attention over the new tokens.  grid = (Np/BM, H), one warp per 16
// query rows; the key loop stops at blockIdx.x so the fully-masked upper
// triangle of blocks is never visited.
// ---------------------------------------------------------------------------
template <int VSPLIT>
__global__ __launch_bounds__(128) void attn_kernel(
    const bf16* __restrict__ q,      // [N, H, DQK]
    const bf16* __restrict__ kbuf,   // [H, Np, DNOPE]
    const bf16* __restrict__ vT,     // [H, DV, Np]
    const bf16* __restrict__ kpe,    // [N, DPE], row stride kpe_s
    bf16* __restrict__ out,          // [N, H*DV]
    int N, int Np, int H, long kpe_s, float scale_log2e) {
  constexpr int KS = DQK + 8;
  constexpr int VS = BN + 8;
  constexpr int DVS = DV / VSPLIT;   // value dims owned by this block
  constexpr int OS = DVS + 8;
  extern __shared__ __align__(16) char smem_raw[];
  bf16* Qs = reinterpret_cast<bf16*>(smem_raw);            // [BM][KS]
  bf16* Ks = Qs + BM * KS;                                 // [2][BM][KS]
  bf16* Vs = Ks + 2 * BM * KS;                             // [2][DVS][VS]

  // Splitting the value dimension across blocks recomputes Q@K^T per split but
  // multiplies the block count, which is what short sequences need: at N<=64
  // there is one key block and one query block, so VSPLIT=1 would leave only
  // `num_heads` blocks for 148 SMs.
  const int qb = blockIdx.x, h = blockIdx.y, vs = blockIdx.z;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int r0 = qb * BM + warp * 16;
  const int qrow[2] = {r0 + g, r0 + g + 8};
  const int nkb = qb + 1;

  // q tile: 12 threads' worth of 8 bf16 per row (DQK/8 = 24 chunks per row).
  {
    const int lr = tid >> 2, lc = (tid & 3) * 8;
#pragma unroll
    for (int i = 0; i < BM / 32; ++i)
#pragma unroll
      for (int j = 0; j < DQK / 32; ++j) {
        const int r = lr + i * 32, c = lc + j * 32;
        const int tk = qb * BM + r;
        const bool ok = tk < N;
        cp_async16(&Qs[r * KS + c],
                   q + ((size_t)(ok ? tk : 0) * H + h) * DQK + c, ok);
      }
  }

  // Programmatic dependent launch: this grid is scheduled while the projection
  // is still draining, so the `q` tile -- which the projection does not touch --
  // is already in flight by the time the dependency is honoured.  Everything
  // below the wait reads `kbuf`/`vT`.
  asm volatile("griddepcontrol.wait;" ::: "memory");

#define ATTN_LOAD(stage, key0)                                                \
  do {                                                                        \
    _Pragma("unroll")                                                         \
    for (int i = 0; i < BM * (DNOPE / 8) / 128; ++i) {                        \
      const int u = tid + i * 128;                                            \
      const int r = u >> 4, c = (u & 15) * 8;                                 \
      cp_async16(&Ks[(stage) * BM * KS + r * KS + c],                         \
                 kbuf + ((size_t)h * Np + (key0) + r) * DNOPE + c, true);     \
    }                                                                         \
    _Pragma("unroll")                                                         \
    for (int i = 0; i < BM * (DPE / 8) / 128; ++i) {                          \
      const int u = tid + i * 128;                                            \
      const int r = u >> 3, c = (u & 7) * 8;                                  \
      const int tk = (key0) + r;                                              \
      const bool ok = tk < N;                                                 \
      cp_async16(&Ks[(stage) * BM * KS + r * KS + DNOPE + c],                 \
                 kpe + (size_t)(ok ? tk : 0) * kpe_s + c, ok);                \
    }                                                                         \
    _Pragma("unroll")                                                         \
    for (int i = 0; i < DVS * (BN / 8) / 128; ++i) {                          \
      const int u = tid + i * 128;                                            \
      const int d = u >> 3, c = (u & 7) * 8;                                  \
      cp_async16(&Vs[(stage) * DVS * VS + d * VS + c],                        \
                 vT + ((size_t)h * DV + vs * DVS + d) * Np + (key0) + c,      \
                 true);                                                       \
    }                                                                         \
    CP_COMMIT();                                                              \
  } while (0)

  ATTN_LOAD(0, 0);
  if (nkb > 1) ATTN_LOAD(1, BN); else CP_COMMIT();

  float o[DVS / 8][4];
#pragma unroll
  for (int nt = 0; nt < DVS / 8; ++nt)
#pragma unroll
    for (int x = 0; x < 4; ++x) o[nt][x] = 0.f;
  float mi[2] = {-1e30f, -1e30f}, li[2] = {0.f, 0.f};

  // The q tile lands with the first key group; both are waited on together.
  uint32_t qf[DQK / 16][4];
  bool q_ready = false;

#pragma unroll 1
  for (int kb = 0; kb < nkb; ++kb) {
    const int key0 = kb * BN;
    const int cur = kb & 1;
    CP_WAIT(1);
    __syncthreads();
    if (!q_ready) {
      q_ready = true;
#pragma unroll
      for (int j = 0; j < DQK / 16; ++j)
        ldm_x4(qf[j], &Qs[ldm_a_off(lane, warp * 16, j * 16, KS)]);
    }

    const bf16* kst = Ks + cur * BM * KS;
    const bf16* vst = Vs + cur * DVS * VS;
    float s[BN / 8][4];
#pragma unroll
    for (int nt = 0; nt < BN / 8; ++nt)
#pragma unroll
      for (int x = 0; x < 4; ++x) s[nt][x] = 0.f;
#pragma unroll
    for (int j = 0; j < DQK / 16; ++j) {
#pragma unroll
      for (int nt = 0; nt < BN / 8; nt += 2) {
        uint32_t b[4];
        ldm_x4(b, &kst[ldm_b_off(lane, nt * 8, j * 16, KS)]);
        MMA_16816(s[nt][0], s[nt][1], s[nt][2], s[nt][3],
                  qf[j][0], qf[j][1], qf[j][2], qf[j][3], b[0], b[1]);
        MMA_16816(s[nt + 1][0], s[nt + 1][1], s[nt + 1][2], s[nt + 1][3],
                  qf[j][0], qf[j][1], qf[j][2], qf[j][3], b[2], b[3]);
      }
    }

    float rmax[2] = {mi[0], mi[1]};
#pragma unroll
    for (int nt = 0; nt < BN / 8; ++nt) {
      const int kk = key0 + nt * 8 + 2 * t;
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const int key = kk + c;
        const bool live = key < N;
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
          float v = s[nt][2 * rr + c] * scale_log2e;
          if (!(live && key <= qrow[rr])) v = -1e30f;
          s[nt][2 * rr + c] = v;
          rmax[rr] = fmaxf(rmax[rr], v);
        }
      }
    }
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
      rmax[rr] = fmaxf(rmax[rr], __shfl_xor_sync(0xffffffffu, rmax[rr], 1));
      rmax[rr] = fmaxf(rmax[rr], __shfl_xor_sync(0xffffffffu, rmax[rr], 2));
    }
    const float corr[2] = {exp2f(mi[0] - rmax[0]), exp2f(mi[1] - rmax[1])};
    float rsum[2] = {0.f, 0.f};
#pragma unroll
    for (int nt = 0; nt < BN / 8; ++nt)
#pragma unroll
      for (int c = 0; c < 2; ++c)
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
          const float p = exp2f(s[nt][2 * rr + c] - rmax[rr]);
          s[nt][2 * rr + c] = p;
          rsum[rr] += p;
        }
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
      rsum[rr] += __shfl_xor_sync(0xffffffffu, rsum[rr], 1);
      rsum[rr] += __shfl_xor_sync(0xffffffffu, rsum[rr], 2);
      li[rr] = li[rr] * corr[rr] + rsum[rr];
      mi[rr] = rmax[rr];
    }
#pragma unroll
    for (int nt = 0; nt < DVS / 8; ++nt) {
      o[nt][0] *= corr[0];
      o[nt][1] *= corr[0];
      o[nt][2] *= corr[1];
      o[nt][3] *= corr[1];
    }

#pragma unroll
    for (int ks = 0; ks < BN / 16; ++ks) {
      const uint32_t pa[4] = {pack2(s[2 * ks][0], s[2 * ks][1]),
                              pack2(s[2 * ks][2], s[2 * ks][3]),
                              pack2(s[2 * ks + 1][0], s[2 * ks + 1][1]),
                              pack2(s[2 * ks + 1][2], s[2 * ks + 1][3])};
#pragma unroll
      for (int nt = 0; nt < DVS / 8; nt += 2) {
        uint32_t b[4];
        ldm_x4(b, &vst[ldm_b_off(lane, nt * 8, ks * 16, VS)]);
        MMA_16816(o[nt][0], o[nt][1], o[nt][2], o[nt][3],
                  pa[0], pa[1], pa[2], pa[3], b[0], b[1]);
        MMA_16816(o[nt + 1][0], o[nt + 1][1], o[nt + 1][2], o[nt + 1][3],
                  pa[0], pa[1], pa[2], pa[3], b[2], b[3]);
      }
    }
    if (kb + 2 < nkb) {
      __syncthreads();
      ATTN_LOAD(cur, (kb + 2) * BN);
    } else {
      CP_COMMIT();
    }
  }
#undef ATTN_LOAD

  __syncthreads();
  bf16* Os = Qs;   // BM * OS * 2 B <= BM * KS * 2 B
  const float rl[2] = {1.f / li[0], 1.f / li[1]};
#pragma unroll
  for (int nt = 0; nt < DVS / 8; ++nt) {
    const int c = nt * 8 + 2 * t;
    Os[(warp * 16 + g) * OS + c] = __float2bfloat16_rn(o[nt][0] * rl[0]);
    Os[(warp * 16 + g) * OS + c + 1] = __float2bfloat16_rn(o[nt][1] * rl[0]);
    Os[(warp * 16 + g + 8) * OS + c] = __float2bfloat16_rn(o[nt][2] * rl[1]);
    Os[(warp * 16 + g + 8) * OS + c + 1] = __float2bfloat16_rn(o[nt][3] * rl[1]);
  }
  __syncthreads();
  const int per = DVS / 2;
  const int r = tid >> 1, c0 = (tid & 1) * per;
  const int tk = qb * BM + r;
  if (tk < N) {
    bf16* dst = out + (size_t)tk * (H * DV) + h * DV + vs * DVS + c0;
#pragma unroll
    for (int i = 0; i < per; i += 8)
      *reinterpret_cast<uint4*>(dst + i) =
          *reinterpret_cast<const uint4*>(&Os[r * OS + c0 + i]);
  }
}

template <int VSPLIT>
static constexpr int attn_smem() {
  return (3 * BM * (DQK + 8) + 2 * (DV / VSPLIT) * (BN + 8)) * (int)sizeof(bf16);
}
template <int BN_>
static constexpr int proj_smem() {
  return (BM + BN_) * (512 + 8) * (int)sizeof(bf16);
}

template <int VSPLIT>
static void launch_attn(cudaStream_t stream, int Np, int H, const bf16* qp,
                        const bf16* kbuf, const bf16* vT, const bf16* kp,
                        bf16* op, int N, long ks, float sl) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(Np / BM, H, VSPLIT);
  cfg.blockDim = dim3(128, 1, 1);
  cfg.dynamicSmemBytes = attn_smem<VSPLIT>();
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, attn_kernel<VSPLIT>, qp, kbuf, vT, kp,
                                    op, N, Np, H, ks, sl));
}

// ---------------------------------------------------------------------------
// k[N][H][DQK] = [ kv[N][h][0:DNOPE] | k_pe[N][0:DPE] ].
//
// Used on the long-sequence path, where the reference up-projection (cuBLAS) and
// FA4 both stay but the concatenation does not: as two strided slice copies it
// writes 128 and then 64 of every 192 lanes, so every store is a partial sector.
// One pass with the RoPE half folded in writes whole 384 B rows instead.
// ---------------------------------------------------------------------------
__global__ void build_k_kernel(const bf16* __restrict__ kv,
                               const bf16* __restrict__ kpe,
                               bf16* __restrict__ k, int N, int H, long kpe_s) {
  constexpr int CPT = DQK / 8;        // 16 B chunks per (token, head)
  const int chunk = blockIdx.x * blockDim.x + threadIdx.x;
  if (chunk >= H * CPT) return;
  // CPT is a compile-time constant, so this is a multiply-shift.  The token
  // index is grid.y precisely so no runtime (64-bit) division is needed.
  const int h = chunk / CPT;
  const int c = (chunk - h * CPT) * 8;
  const bool is_pe = c >= DNOPE;
  const bf16* src = is_pe ? kpe + (c - DNOPE) : kv + (size_t)h * 256 + c;
  const long sstride = is_pe ? kpe_s : (long)H * 256;
  bf16* dst = k + (size_t)h * DQK + c;
  const long dstride = (long)H * DQK;
  for (int n = blockIdx.y; n < N; n += gridDim.y)
    *reinterpret_cast<uint4*>(dst + n * dstride) =
        *reinterpret_cast<const uint4*>(src + n * sstride);
}

at::Tensor build_k(at::Tensor kv, at::Tensor kpe, int64_t num_heads) {
  const long N = kv.size(0);
  const int H = (int)num_heads;
  TORCH_CHECK(kv.scalar_type() == at::kBFloat16, "build_k: bf16 only");
  at::Tensor k = at::empty({N, (long)H, (long)DQK}, kv.options());
  if (N == 0) return k;
  const int threads = 128;
  const int gx = (H * (DQK / 8) + threads - 1) / threads;
  const unsigned gy = (unsigned)std::min<long>(N, 65535);
  const c10::cuda::OptionalCUDAGuard guard(at::device_of(kv));
  build_k_kernel<<<dim3((unsigned)gx, gy), threads, 0,
                   at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(kv.data_ptr()),
      reinterpret_cast<const bf16*>(kpe.data_ptr()),
      reinterpret_cast<bf16*>(k.data_ptr()), (int)N, H, (long)kpe.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return k;
}

at::Tensor mla_prefill(at::Tensor q, at::Tensor kvc, at::Tensor W, at::Tensor kpe,
                       double scale, int64_t num_heads) {
  const int N = (int)q.size(0);
  const int H = (int)num_heads;
  const int LORA = (int)kvc.size(1);
  TORCH_CHECK(LORA == 512, "mla_prefill: kv_lora_rank must be 512");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16, "mla_prefill: bf16 only");
  const int Np = ((N + BM - 1) / BM) * BM;

  auto opts = q.options();
  at::Tensor ws = at::empty({(long)H * Np * (DNOPE + DV)}, opts);
  bf16* kbuf = reinterpret_cast<bf16*>(ws.data_ptr());
  bf16* vT = kbuf + (size_t)H * Np * DNOPE;
  at::Tensor out = at::empty({N, H * DV}, opts);

  const c10::cuda::OptionalCUDAGuard guard(at::device_of(q));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  static bool smem_ready = false;
  if (!smem_ready) {
    // The staged tiles exceed the 48 KB static limit; opt in to the full
    // per-SM shared window once.
    cudaFuncSetAttribute(proj_kernel<512, 64>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, proj_smem<64>());
    cudaFuncSetAttribute(attn_kernel<1>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, attn_smem<1>());
    cudaFuncSetAttribute(attn_kernel<2>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, attn_smem<2>());
    cudaFuncSetAttribute(attn_kernel<4>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, attn_smem<4>());
    smem_ready = true;
  }

  const bf16* kvcp = reinterpret_cast<const bf16*>(kvc.data_ptr());
  const bf16* Wp = reinterpret_cast<const bf16*>(W.data_ptr());
  if (N == 1) {
    vonly_kernel<512><<<(H * DV + 3) / 4, 128, 0, stream>>>(
        kvcp, Wp, reinterpret_cast<bf16*>(out.data_ptr()), H);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
  }
  // Narrow column tiles while there is SM headroom; wide ones once the token
  // axis alone fills the machine (they re-read `kv_c_normed` far less).
  proj_kernel<512, 64><<<dim3(Np / BM, 4 * H), 128, proj_smem<64>(), stream>>>(
      kvcp, Wp, kbuf, vT, N, Np);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const bf16* qp = reinterpret_cast<const bf16*>(q.data_ptr());
  const bf16* kp = reinterpret_cast<const bf16*>(kpe.data_ptr());
  bf16* op = reinterpret_cast<bf16*>(out.data_ptr());
  const long ks = (long)kpe.stride(0);
  const float sl = (float)(scale * 1.4426950408889634);
  // Enough blocks to fill the SM array; past that the split only duplicates
  // Q@K^T work.
  const int vsplit = Np <= 2 * BM ? 4 : (Np <= 3 * BM ? 2 : 1);
  if (vsplit == 4)
    launch_attn<4>(stream, Np, H, qp, kbuf, vT, kp, op, N, ks, sl);
  else if (vsplit == 2)
    launch_attn<2>(stream, Np, H, qp, kbuf, vT, kp, op, N, ks, sl);
  else
    launch_attn<1>(stream, Np, H, qp, kbuf, vT, kp, op, N, ks, sl);
  return out;
}

}  // namespace fkmla

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mla_prefill", &fkmla::mla_prefill, "Fused MLA dense prefill");
  m.def("build_k", &fkmla::build_k, "Fused k_nope || k_pe concatenation");
}
