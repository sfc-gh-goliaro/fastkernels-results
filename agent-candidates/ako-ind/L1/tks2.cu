// Specialized fused top-k + softmax for the captured regime:
//   bf16 logits, num_experts == 128, top_k == 8, renormalize == true,
//   no softcapping, no correction bias, no `finished` mask.
//
// Round-2 rewrite of the *selection algorithm*, plus a launch change.  Three
// things differ from round 1; (3) turned out to be worth the most:
//
// (1) The global softmax is elided.  With renormalize=true the global
//     denominator cancels exactly:
//         w_i = (exp(x_i-m)/S) / sum_j (exp(x_j-m)/S)
//             =  exp(x_i-m)     / sum_j  exp(x_j-m)
//     so the 128-wide expf pass and the 128-wide sum butterfly are
//     unnecessary.  Selection happens on the *raw* logits (exp is monotonic)
//     and only the 8 winners are exponentiated.
//
// (2) The 8 serial argmax passes are gone.  Each bf16 logit is packed with its
//     expert index into one 32-bit key that is monotonic under *unsigned
//     integer* compare:
//
//         u    = (uint32)bits(bf16) << 16          // == the fp32 bit pattern
//         s    = (int32)u >> 31                    // 0 / 0xffffffff
//         key  = (u ^ ((s << 16) | 0x80000000)) | (127 - expert)
//
//     The high 16 bits are the standard IEEE total-order flip (sign-magnitude
//     -> biased unsigned), the low 7 bits hold the *complemented* expert index
//     so that a lower expert index compares greater, reproducing the
//     reference's tie-break for free.  Selection is then pure IMNMX with the
//     index carried along, and key[0] also yields the row max for the exp
//     shift, so there is no separate max reduction either.
//
//     Ordering keys descending is exactly the reference's output order (the
//     reference pops (value, -index) maxima one at a time).
//
//     Per row: each thread reduces its VPT keys to a register-resident sorted
//     top-8 with 19-comparator sorting networks plus bitonic 8+8 -> top-8
//     merges, then the TPR threads of the row combine either by
//       SCHEME 0: 8 x `__reduce_max_sync` (one REDUX.MAX instruction per
//                 winner, no shuffles at all) popping from each lane's sorted
//                 list via a register shift, or
//       SCHEME 1: log2(TPR) bitonic 8+8 -> top-8 merges, 8 shuffles each.
//
// (3) Programmatic Dependent Launch.  The grid is dispatched while the kernel
//     that produced `inp` is still draining, and waits on it with
//     `cudaGridDependencySynchronize()` immediately before the first load -- so
//     only address arithmetic runs ahead of the producer, and every load and
//     store still happens after it.  Worth ~2 us per call at every M, which is
//     more than the selection rewrite bought outside M=16384.  Graph-safe.
//
// The -0.0 canonicalisation (NZ=1) matters for bit-exactness: exp(-0) and
// exp(+0) are both 1, so the reference treats -0.0 and +0.0 as tied and breaks
// by index, while the raw key order would rank +0.0 above -0.0.  Mapping
// bits(-0.0) -> bits(+0.0) before the flip restores the reference's answer.
//
// Two degenerate input families are *not* bit-exact and cannot be: a row whose
// 8th-largest logit is more than ~104 below the max (float32 exp underflows
// several distinct logits to exactly 0, and the reference then ranks those zeros
// by index while we rank them by value -- weights identical, ids differ in
// zero-weight slots), and a row where every logit is within ~1e-7 of zero (exp
// collapses distinct logits onto one float).  Neither is reachable from bf16
// logits at any normal scale; see prof/verify_extreme.py.

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda_bf16.h>

#include <cstdint>

#define WSZ 32

template <int N>
struct alignas(N * 4 > 16 ? 16 : N * 4) U32Vec {
  uint32_t d[N];
};

// ---------------------------------------------------------------------------
// key packing / unpacking
// ---------------------------------------------------------------------------

// u: fp32 bit pattern of the logit (low 16 bits zero, i.e. a widened bf16).
// idxc: 127 - expert, in [0, 127].
__device__ __forceinline__ uint32_t pack_key(uint32_t u, uint32_t idxc) {
  const uint32_t s = (uint32_t)(((int32_t)u) >> 31);       // 0 or ~0
  const uint32_t m = (s << 16) | 0x80000000u;              // 0x80000000/0xffff0000
  return (u ^ m) | idxc;
}

__device__ __forceinline__ float key_logit(uint32_t k) {
  const uint32_t f = k & 0xffff0000u;
  const uint32_t m = (f & 0x80000000u) ? 0x80000000u : 0xffff0000u;
  return __int_as_float(f ^ m);
}

__device__ __forceinline__ int key_expert(uint32_t k) {
  return 127 - (int)(k & 0x7fu);
}

// ---------------------------------------------------------------------------
// register sorting networks (descending)
// ---------------------------------------------------------------------------

#define CEX(a, b)                    \
  {                                  \
    const uint32_t _h = max(a, b);   \
    const uint32_t _l = min(a, b);   \
    (a) = _h;                        \
    (b) = _l;                        \
  }

// 19-comparator, depth-7 sorting network for 8 elements (verified exhaustively
// with the 0/1 principle).
__device__ __forceinline__ void sort8(uint32_t* r) {
  CEX(r[0], r[1]); CEX(r[2], r[3]); CEX(r[4], r[5]); CEX(r[6], r[7]);
  CEX(r[0], r[2]); CEX(r[1], r[3]); CEX(r[4], r[6]); CEX(r[5], r[7]);
  CEX(r[1], r[2]); CEX(r[5], r[6]); CEX(r[0], r[4]); CEX(r[3], r[7]);
  CEX(r[1], r[5]); CEX(r[2], r[6]);
  CEX(r[1], r[4]); CEX(r[3], r[6]);
  CEX(r[2], r[4]); CEX(r[3], r[5]);
  CEX(r[3], r[4]);
}

// Sort a bitonic sequence of 8 into descending order (12 comparators).
__device__ __forceinline__ void bmerge8(uint32_t* r) {
  CEX(r[0], r[4]); CEX(r[1], r[5]); CEX(r[2], r[6]); CEX(r[3], r[7]);
  CEX(r[0], r[2]); CEX(r[1], r[3]); CEX(r[4], r[6]); CEX(r[5], r[7]);
  CEX(r[0], r[1]); CEX(r[2], r[3]); CEX(r[4], r[5]); CEX(r[6], r[7]);
}

// r, s both sorted descending -> r := sorted-descending top 8 of the union.
__device__ __forceinline__ void merge_top8(uint32_t* r, const uint32_t* s) {
#pragma unroll
  for (int i = 0; i < 8; ++i) r[i] = max(r[i], s[7 - i]);
  bmerge8(r);
}

// ---------------------------------------------------------------------------
// VPT  : logits per thread (TPR = 128 / VPT threads cooperate on one row)
// WPC  : warps per CTA
// SCH  : 0 = REDUX.MAX pop, 1 = bitonic cross-lane merge tree
// NZ   : 1 = canonicalise -0.0 to +0.0 before packing
// FIN  : 0 = every lane computes all 8 weights, lane 0 does two 16B stores
//        1 = lane l computes/stores only output slot l (TPR >= 8 only)
// STG  : attribution stages (wrong results on purpose, same work profile)
//        0 = full, 1 = load+pack+row-max only, 2 = no cross-lane combine,
//        3 = empty kernel (grid-launch cost only), 4 = stores only (no load)
// PD   : 1 = wait on the programmatic-dependent-launch predecessor just before
//        the first load, so the grid can be dispatched while the kernel that
//        produced `inp` is still draining (host must set the matching launch
//        attribute).
// ---------------------------------------------------------------------------
template <int VPT, int WPC, int SCH, int NZ, int FIN, int STG = 0, int PD = 0>
__launch_bounds__(WPC* WSZ) __global__ void tks2_k8(
    const __nv_bfloat16* __restrict__ inp,
    float* __restrict__ ow,
    int* __restrict__ oi,
    int num_rows) {
  if (STG == 3) return;
  constexpr int E = 128;
  constexpr int K = 8;
  constexpr int TPR = E / VPT;              // threads per row
  constexpr int RPW = WSZ / TPR;            // rows per warp
  constexpr int RPC = WPC * RPW;            // rows per CTA
  constexpr int EPL = (VPT >= 8) ? 8 : VPT; // bf16 per vector load
  constexpr int NLDG = VPT / EPL;           // vector loads per thread
  constexpr int W32 = EPL / 2;              // uint32 per vector load
  using Vec = U32Vec<W32>;

  const int tid = threadIdx.x;
  const int lane = (TPR > 1) ? (tid & (TPR - 1)) : 0;
  const int rgroup = (TPR > 1) ? (tid / TPR) : tid;

  const int row_raw = (int)blockIdx.x * RPC + rgroup;
  const bool valid = row_raw < num_rows;
  const int row = valid ? row_raw : (num_rows - 1);

  const int first = lane * EPL;  // first expert column read by this thread

  if (STG == 4) {
    // stores only: how much of the per-call floor is the output write?
    if (lane == 0 && valid) {
      float4 f4 = make_float4(1.f, 0.f, 0.f, 0.f);
      float4* wp = reinterpret_cast<float4*>(ow + (int64_t)row * K);
      wp[0] = f4;
      wp[1] = f4;
      int4 i4 = make_int4(0, 1, 2, 3);
      int4* ip = reinterpret_cast<int4*>(oi + (int64_t)row * K);
      ip[0] = i4;
      ip[1] = i4;
    }
    return;
  }

  // ---- load ---------------------------------------------------------------
  const uint32_t* p =
      reinterpret_cast<const uint32_t*>(inp + (int64_t)row * E) + (first >> 1);
#if __CUDA_ARCH__ >= 900
  if (PD) cudaGridDependencySynchronize();
#endif
  Vec raw[NLDG];
#pragma unroll
  for (int i = 0; i < NLDG; ++i) {
    raw[i] = *reinterpret_cast<const Vec*>(p + i * (TPR * W32));
  }

  // ---- pack keys + per-thread sorted top-8 --------------------------------
  // One vector load == one 8-element chunk (4 when VPT==4): pack, sort, and
  // fold into the running sorted top-8.  All loads are issued above so the
  // chunk loop never waits on memory it could have prefetched.
  const uint32_t idxbase = (uint32_t)(127 - first);
  uint32_t r[8];
#pragma unroll
  for (int i = 0; i < NLDG; ++i) {
    uint32_t s[8];
#pragma unroll
    for (int t = 0; t < W32; ++t) {
      const uint32_t w = raw[i].d[t];
      uint32_t ulo = w << 16;
      uint32_t uhi = w & 0xffff0000u;
      if (NZ) {
        ulo = (ulo == 0x80000000u) ? 0u : ulo;
        uhi = (uhi == 0x80000000u) ? 0u : uhi;
      }
      const int c = i * (TPR * EPL) + 2 * t;
      s[2 * t] = pack_key(ulo, idxbase - (uint32_t)c);
      s[2 * t + 1] = pack_key(uhi, idxbase - (uint32_t)(c + 1));
    }
#pragma unroll
    for (int t = EPL; t < 8; ++t) s[t] = 0u;  // only when VPT < 8
    if (STG == 1) {
      // load + pack + a plain row max, nothing else
#pragma unroll
      for (int t = 1; t < 8; ++t) s[0] = max(s[0], s[t]);
      r[0] = (i == 0) ? s[0] : max(r[0], s[0]);
      continue;
    }
    sort8(s);
    if (i == 0) {
#pragma unroll
      for (int t = 0; t < 8; ++t) r[t] = s[t];
    } else {
      merge_top8(r, s);
    }
  }

  if (STG == 1) {
    // one cross-lane max, then the same 4 stores, so nothing is dead code
    uint32_t m = r[0];
    if (TPR > 1) {
#pragma unroll
      for (int d = TPR / 2; d > 0; d >>= 1) {
        m = max(m, __shfl_xor_sync(0xffffffffu, m, d, TPR));
      }
    }
    if (lane == 0 && valid) {
      const float v = key_logit(m);
      float4 f4 = make_float4(v, v, v, v);
      float4* wp = reinterpret_cast<float4*>(ow + (int64_t)row * K);
      wp[0] = f4;
      wp[1] = f4;
      const int e0 = key_expert(m);
      int4 i4 = make_int4(e0, e0, e0, e0);
      int4* ip = reinterpret_cast<int4*>(oi + (int64_t)row * K);
      ip[0] = i4;
      ip[1] = i4;
    }
    return;
  }

  // ---- combine the TPR lanes of the row ----------------------------------
  uint32_t o[8];
  if (TPR == 1 || STG == 2) {
#pragma unroll
    for (int i = 0; i < K; ++i) o[i] = r[i];
  } else if (SCH == 0) {
    const unsigned rowmask =
        (TPR == 32) ? 0xffffffffu
                    : (((1u << TPR) - 1u) << ((tid & (WSZ - 1)) & ~(TPR - 1)));
#pragma unroll
    for (int k = 0; k < K; ++k) {
      const uint32_t me = r[0];
      const uint32_t win = __reduce_max_sync(rowmask, me);
      o[k] = win;
      const bool won = (me == win);
#pragma unroll
      for (int i = 0; i < 7; ++i) r[i] = won ? r[i + 1] : r[i];
      r[7] = won ? 0u : r[7];
    }
  } else {
#pragma unroll
    for (int d = 1; d < TPR; d <<= 1) {
      uint32_t s[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        s[i] = __shfl_xor_sync(0xffffffffu, r[7 - i], d, TPR);
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) r[i] = max(r[i], s[i]);
      bmerge8(r);
    }
#pragma unroll
    for (int i = 0; i < K; ++i) o[i] = r[i];
  }

  // ---- weights + store ----------------------------------------------------
  if (FIN == 0 || TPR < 8) {
    const float xm = key_logit(o[0]);
    float e[K];
    e[0] = 1.f;
    float s = 1.f;
#pragma unroll
    for (int i = 1; i < K; ++i) {
      e[i] = expf(key_logit(o[i]) - xm);
      s += e[i];
    }
    const float inv = 1.f / s;
    if (lane == 0 && valid) {
      float4 w0, w1;
      w0.x = e[0] * inv; w0.y = e[1] * inv;
      w0.z = e[2] * inv; w0.w = e[3] * inv;
      w1.x = e[4] * inv; w1.y = e[5] * inv;
      w1.z = e[6] * inv; w1.w = e[7] * inv;
      float4* wp = reinterpret_cast<float4*>(ow + (int64_t)row * K);
      wp[0] = w0;
      wp[1] = w1;
      int4 i0, i1;
      i0.x = key_expert(o[0]); i0.y = key_expert(o[1]);
      i0.z = key_expert(o[2]); i0.w = key_expert(o[3]);
      i1.x = key_expert(o[4]); i1.y = key_expert(o[5]);
      i1.z = key_expert(o[6]); i1.w = key_expert(o[7]);
      int4* ip = reinterpret_cast<int4*>(oi + (int64_t)row * K);
      ip[0] = i0;
      ip[1] = i1;
    }
  } else {
    // Lane l owns output slot l: one expf, one float32 + one int32 store.
    uint32_t mine = o[0];
#pragma unroll
    for (int i = 1; i < K; ++i) mine = (lane == i) ? o[i] : mine;
    const float xm = key_logit(o[0]);
    const float ei = (lane == 0) ? 1.f : expf(key_logit(mine) - xm);
    float s = ei;
#pragma unroll
    for (int d = 1; d < K; d <<= 1) s += __shfl_xor_sync(0xffffffffu, s, d, TPR);
    if (lane < K && valid) {
      ow[(int64_t)row * K + lane] = ei / s;
      oi[(int64_t)row * K + lane] = key_expert(mine);
    }
  }
}

// ---------------------------------------------------------------------------
// Host dispatch.
//
// Variant id digits (decimal): S F N C W V
//   V = VPT index    0:4  1:8  2:16  3:32  4:64  5:128
//   W = WPC index    0:1  1:2  2:4   3:8
//   C = SCH, N = NZ, F = FIN, S = STG, P = PD (7th digit)
// So e.g. 1122 == PD 0, STG 0, FIN 0, NZ 1, SCH 1, WPC 4, VPT 16.
// ---------------------------------------------------------------------------

#define RPC_OF(VPT, WPC) ((WPC) * (WSZ) / (128 / (VPT)))
#define NB_OF(VPT, WPC) ((M + RPC_OF(VPT, WPC) - 1) / RPC_OF(VPT, WPC))

#define LAUNCH2(VPT, WPC, SCH, NZ, FIN, STG)                         \
  tks2_k8<VPT, WPC, SCH, NZ, FIN, STG, 0>                            \
      <<<NB_OF(VPT, WPC), WPC * WSZ, 0, stream>>>(xp, wp, ip, M)

// Same launch, but the grid may be dispatched while the kernel that produced
// `inp` is still draining; the kernel waits at `cudaGridDependencySynchronize`
// immediately before its first load.
#define LAUNCH2_PDL(VPT, WPC, SCH, NZ, FIN, STG)                             \
  {                                                                          \
    cudaLaunchConfig_t cfg = {};                                             \
    cfg.gridDim = dim3(NB_OF(VPT, WPC), 1, 1);                               \
    cfg.blockDim = dim3(WPC * WSZ, 1, 1);                                    \
    cfg.dynamicSmemBytes = 0;                                                \
    cfg.stream = stream;                                                     \
    cudaLaunchAttribute at1[1];                                              \
    at1[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;           \
    at1[0].val.programmaticStreamSerializationAllowed = 1;                    \
    cfg.attrs = at1;                                                         \
    cfg.numAttrs = 1;                                                        \
    cudaLaunchKernelEx(&cfg, tks2_k8<VPT, WPC, SCH, NZ, FIN, STG, 1>,        \
                       xp, wp, ip, M);                                       \
  }

#define C2(VI, VPT, WI, WPC, SCH, NZ, FIN, STG)                              \
  case (VI) + 10 * (WI) + 100 * (SCH) + 1000 * (NZ) + 10000 * (FIN) +        \
      100000 * (STG):                                                        \
    LAUNCH2(VPT, WPC, SCH, NZ, FIN, STG);                                    \
    break;

#define C2P(VI, VPT, WI, WPC, SCH, NZ, FIN, STG)                             \
  case 1000000 + (VI) + 10 * (WI) + 100 * (SCH) + 1000 * (NZ) +              \
      10000 * (FIN) + 100000 * (STG):                                        \
    LAUNCH2_PDL(VPT, WPC, SCH, NZ, FIN, STG);                                \
    break;

// The full WPC row for one VPT, at fixed (SCH, NZ, FIN, STG).
#define C2_W(VI, VPT, SCH, NZ, FIN, STG)  \
  C2(VI, VPT, 0, 1, SCH, NZ, FIN, STG)    \
  C2(VI, VPT, 1, 2, SCH, NZ, FIN, STG)    \
  C2(VI, VPT, 2, 4, SCH, NZ, FIN, STG)    \
  C2(VI, VPT, 3, 8, SCH, NZ, FIN, STG)

#define C2P_W(VI, VPT, SCH, NZ, FIN, STG)  \
  C2P(VI, VPT, 0, 1, SCH, NZ, FIN, STG)    \
  C2P(VI, VPT, 1, 2, SCH, NZ, FIN, STG)    \
  C2P(VI, VPT, 2, 4, SCH, NZ, FIN, STG)    \
  C2P(VI, VPT, 3, 8, SCH, NZ, FIN, STG)

void tks2(
    torch::Tensor& topk_weights,
    torch::Tensor& topk_indices,
    torch::Tensor& logits,
    int64_t variant) {
  const int M = (int)logits.size(0);
  if (M <= 0) return;
  const __nv_bfloat16* xp =
      reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr());
  float* wp = reinterpret_cast<float*>(topk_weights.data_ptr());
  int* ip = reinterpret_cast<int*>(topk_indices.data_ptr());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (variant) {
    // Default shipped configuration: bitonic cross-lane merge, -0.0
    // canonicalisation on, PDL enabled.
    //
    // Geometry switches on M because the two regimes have opposite needs. Below
    // the crossover the row count cannot fill the machine, so what matters is
    // parallelism: VPT=16 (8 threads/row, 4 warps/CTA) spreads each row over 8
    // lanes and pays 3 cross-lane merge stages for it. Above it the machine is
    // saturated and the 24 warp shuffles of those 3 stages become the cost, so
    // VPT=64 (2 threads/row, 1 warp/CTA) trades 8x more per-thread sorting work
    // for a single 8-shuffle merge stage. Measured crossover (marginal us with
    // the harness's copy predecessor): VPT=16 wins or ties up to M=10000
    // (6.15/6.15), VPT=64 wins from M=12288 (6.15 vs 6.44) and at M=16384
    // (6.8-7.0 vs 8.2).
    case 0:
      if (M >= 12000) {
        LAUNCH2_PDL(64, 1, 1, 1, 0, 0);
      } else {
        LAUNCH2_PDL(16, 4, 1, 1, 0, 0);
      }
      break;
    // same, no PDL (A/B reference)
    case 1:
      if (M >= 12000) {
        LAUNCH2(64, 1, 1, 1, 0, 0);
      } else {
        LAUNCH2(16, 4, 1, 1, 0, 0);
      }
      break;

    // ------------------------------------------------------------------
    // Probe instantiations, kept so the next round can re-sweep geometry and
    // re-derive the stage table without editing this file. Deliberately *not*
    // exhaustive: the SCH=0 (REDUX.MAX), NZ=0, FIN=1 and tree-merge variants
    // were all measured this round and lost, so they stay reachable in the
    // template but have no case here (see ITERATIONS.md "Dead ends").
    // ------------------------------------------------------------------

    // geometry grid, PDL off / on
    C2_W(1, 8, 1, 1, 0, 0)
    C2_W(2, 16, 1, 1, 0, 0)
    C2_W(3, 32, 1, 1, 0, 0)
    C2_W(4, 64, 1, 1, 0, 0)
    C2_W(5, 128, 1, 1, 0, 0)
    C2P_W(1, 8, 1, 1, 0, 0)
    C2P_W(2, 16, 1, 1, 0, 0)
    C2P_W(3, 32, 1, 1, 0, 0)
    C2P_W(4, 64, 1, 1, 0, 0)
    C2P_W(5, 128, 1, 1, 0, 0)

    // stage attribution for the two shipped geometries, PDL off / on:
    // STG 1 = load+pack+row max, 2 = no cross-lane combine, 3 = empty,
    // 4 = stores only
    C2(2, 16, 2, 4, 1, 1, 0, 1)   C2P(2, 16, 2, 4, 1, 1, 0, 1)
    C2(2, 16, 2, 4, 1, 1, 0, 2)   C2P(2, 16, 2, 4, 1, 1, 0, 2)
    C2(2, 16, 2, 4, 1, 1, 0, 3)   C2P(2, 16, 2, 4, 1, 1, 0, 3)
    C2(2, 16, 2, 4, 1, 1, 0, 4)   C2P(2, 16, 2, 4, 1, 1, 0, 4)
    C2(4, 64, 0, 1, 1, 1, 0, 1)   C2P(4, 64, 0, 1, 1, 1, 0, 1)
    C2(4, 64, 0, 1, 1, 1, 0, 2)   C2P(4, 64, 0, 1, 1, 1, 0, 2)
    C2(4, 64, 0, 1, 1, 1, 0, 3)   C2P(4, 64, 0, 1, 1, 1, 0, 3)
    C2(4, 64, 0, 1, 1, 1, 0, 4)   C2P(4, 64, 0, 1, 1, 1, 0, 4)

    default:
      TORCH_CHECK(false, "tks2: unknown variant ", variant);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tks2", &tks2, "round-2 key-packed top-k softmax (bf16/E=128/k=8)");
}
