// Fused MSA half of AlphaFold3 Algorithm 8, for B200 (sm_100).
//
// Replaces, per block, OuterProductMean + its residual into z, MSA row attention
// with pair bias + its residual into m, and the SwiGLU transition + its residual --
// the stages the frozen L2 winners spend 223 us of device time on at the captured
// shape (m bf16[1,8,16,64], z bf16[1,16,16,128]).
//
// WHY THIS EXISTS, AND WHY IT IS WIDE RATHER THAN FUSED
//
// The anchor composition over the frozen winners measures 1203 us, of which 955 us
// is device kernel *duration* and only 198 us is dependent-launch gap
// (profile/p1-anchor/results.md). The winners' kernels partition one row of the pair
// tensor per CTA, so they run 16 CTAs on a 148-SM device -- 0.108 waves, ~11% of the
// machine lit, and 6-42 us per launch for work whose arithmetic is ~1 us at peak.
//
// So the trade here runs the opposite way from "fuse everything": a launch gap costs
// 2.5 us while these durations cost 6-42 us, and splitting a stage into more kernels
// with wider grids pays whenever it removes more than 2.5 us of duration per launch
// added. Every kernel below is therefore sized to the *output* extent rather than to
// a row of z:
//
//     msa_prep          S*N   = 128 CTAs   one MSA token per CTA
//     opm_z_bias        N*N   = 256 CTAs   one pair (i,j) per CTA
//     msa_attn_out      S*N   = 128 CTAs   one query token per CTA
//     msa_transition    S*N   = 128 CTAs   one MSA token per CTA
//
// DEPENDENCY AND ALIASING TABLE
//
// Only a kernel boundary orders cross-CTA traffic; __syncthreads() does not. Each
// row states what a CTA reads beyond its own slice, so the boundaries below are
// derived rather than assumed.
//
//   kernel          reads beyond own slice          writes            aliasing
//   msa_prep        nothing (own token of m only)   a,b,v,g (own)     scratch is fresh
//   opm_z_bias      a[:,i,:], b[:,j,:] (all s)      z1[i,j], bias[:,i,j]
//                                                                     z1 fresh: z0 is
//                                                                     the harness's
//                                                                     input, never
//                                                                     written
//   msa_attn_out    bias[:,i,:], v[s,:,:] (all j)   m1[s,i]           m1 fresh, same
//                                                                     reason
//   msa_transition  nothing (own token of m1)       m1[s,i] in place  safe: no other
//                                                                     CTA reads it
//
// Two consequences worth stating because the plan asserts otherwise. First, MSA row
// attention needs only its *own* row of the pair tensor for the pair bias -- z_proj
// is [i,j,h] and a query row i consumes j=0..N-1 of row i alone -- so the bias for
// pair (i,j) is computed by the very CTA that just produced z1[i,j], with no extra
// boundary. Second, because the v/g projections are hoisted into msa_prep, no CTA
// ever reads a residue row of m that another CTA writes, so m needs no double buffer;
// the only rule is not to write the harness's input tensors. That is four kernels per
// block, and the boundary count follows from the table rather than from a target.
//
// NUMERICS
//
// Every rounding point the baseline materializes is reproduced, because four residual
// blocks compound anything that is merely inside tolerance:
//
//   * LayerNorm: fp32 mean/biased-variance/normalize/affine from the bf16 parameter
//     values the harness produced, eps = 1e-5, one round to bf16 at the end.
//   * Every projection: bf16 inputs, fp32 accumulation, one round to bf16 -- matching
//     F.linear's cuBLAS path.
//   * Each elementwise product and residual add rounds where the baseline
//     materializes a tensor, including the two roundings in
//     silu(linear_a(x)) * linear_b(x) and in sigmoid(linear_g(x)).
//   * OuterProductMean applies linear_out (the operator's only biased Linear) before
//     the division, so the bias is divided by norm too; norm is the pairwise mask
//     count accumulated in fp32 from bf16 masks, rounded to bf16, with eps added and
//     rounded again in bf16 -- adding eps in fp32 would change norm.
//   * Softmax is max-subtracted, accumulated in fp32, and its weights round to bf16
//     before the value contraction, as F.softmax on a bf16 input does.
//
// WEIGHT LAYOUT
//
// Every projection is stored reduction-major (`WT[k][d]`, k the contracted axis) so
// that a thread owning output channel d reads WT[k*D + d] and consecutive threads
// touch consecutive addresses. In the baseline's [out, in] layout the same access is
// strided by the reduction length -- for OPM's [128, 1024] output projection that is
// a 2 KB stride per thread, which serializes into 128 separate sectors per step. The
// host builds this transposed pack lazily on the first forward, because the harness
// casts parameters to bf16 and copies shared weights in only after __init__; the
// offsets come from `layout()` below so the host cannot disagree with the kernels.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include <set>
#include <utility>
#include <vector>

namespace {

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float ld(const bf16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ bf16 rn(float v) { return __float2bfloat16(v); }

constexpr float kLnEps = 1e-5f;

// ---------------------------------------------------------------------------
// Packed weight layout. Sections in kernel-consumption order; every projection
// transposed to [reduction][output]. Shared with the host through layout().
// ---------------------------------------------------------------------------
enum Section : int {
  OPM_LN_W = 0,   // [CM]
  OPM_LN_B,       // [CM]
  OPM_W1T,        // [CM][HOPM]
  OPM_W2T,        // [CM][HOPM]
  OPM_WOUTT,      // [HOPM*HOPM][CZ]
  OPM_OUT_BIAS,   // [CZ]
  ATT_LNZ_W,      // [CZ]
  ATT_LNZ_B,      // [CZ]
  ATT_WZT,        // [CZ][NH]
  ATT_LNM_W,      // [CM]
  ATT_LNM_B,      // [CM]
  ATT_WVT,        // [CM][NH*HD]
  ATT_WGT,        // [CM][NH*HD]
  ATT_WOT,        // [NH*HD][CM]
  TR_LN_W,        // [CM]
  TR_LN_B,        // [CM]
  TR_WAT,         // [CM][TR_HID]
  TR_WBT,         // [CM][TR_HID]
  TR_WOUTT,       // [TR_HID][CM]
  NUM_SECTIONS,
};

struct Layout {
  int off[NUM_SECTIONS];
  int num[NUM_SECTIONS];
  int total;
};

Layout make_layout(int c_m, int c_z, int c_hidden_opm, int no_heads_msa,
                   int c_hidden_msa_att, int transition_n) {
  const int nh_hd = no_heads_msa * c_hidden_msa_att;
  const int tr_hid = transition_n * c_m;
  const int opm_flat = c_hidden_opm * c_hidden_opm;
  int sizes[NUM_SECTIONS];
  sizes[OPM_LN_W] = c_m;
  sizes[OPM_LN_B] = c_m;
  sizes[OPM_W1T] = c_m * c_hidden_opm;
  sizes[OPM_W2T] = c_m * c_hidden_opm;
  sizes[OPM_WOUTT] = opm_flat * c_z;
  sizes[OPM_OUT_BIAS] = c_z;
  sizes[ATT_LNZ_W] = c_z;
  sizes[ATT_LNZ_B] = c_z;
  sizes[ATT_WZT] = c_z * no_heads_msa;
  sizes[ATT_LNM_W] = c_m;
  sizes[ATT_LNM_B] = c_m;
  sizes[ATT_WVT] = c_m * nh_hd;
  sizes[ATT_WGT] = c_m * nh_hd;
  sizes[ATT_WOT] = nh_hd * c_m;
  sizes[TR_LN_W] = c_m;
  sizes[TR_LN_B] = c_m;
  sizes[TR_WAT] = c_m * tr_hid;
  sizes[TR_WBT] = c_m * tr_hid;
  sizes[TR_WOUTT] = tr_hid * c_m;

  Layout L{};
  int acc = 0;
  for (int s = 0; s < NUM_SECTIONS; ++s) {
    // 8-element alignment keeps every section's base 16-byte aligned in bf16.
    acc = (acc + 7) & ~7;
    L.off[s] = acc;
    L.num[s] = sizes[s];
    acc += sizes[s];
  }
  L.total = acc;
  return L;
}

// ---------------------------------------------------------------------------
// Block-wide reductions over a power-of-two thread count, via shared memory.
// ---------------------------------------------------------------------------
template <int THREADS>
__device__ __forceinline__ float block_sum(float v, float* scratch) {
  const int t = threadIdx.x;
  scratch[t] = v;
  __syncthreads();
  for (int s = THREADS >> 1; s > 0; s >>= 1) {
    if (t < s) scratch[t] += scratch[t + s];
    __syncthreads();
  }
  const float out = scratch[0];
  __syncthreads();
  return out;
}

// LayerNorm of `src` (length C, bf16) into `dst` (bf16), fp32 throughout with one
// final round. `promote_fp32=True` in the L1 baseline, biased variance, eps 1e-5.
template <int THREADS>
__device__ __forceinline__ void layer_norm(const bf16* __restrict__ src,
                                           const bf16* __restrict__ w,
                                           const bf16* __restrict__ b,
                                           bf16* __restrict__ dst, int C,
                                           float* scratch) {
  const int t = threadIdx.x;
  float partial = 0.f, partial_sq = 0.f;
  for (int k = t; k < C; k += THREADS) {
    const float x = ld(src + k);
    partial += x;
    partial_sq += x * x;
  }
  const float sum = block_sum<THREADS>(partial, scratch);
  const float sum_sq = block_sum<THREADS>(partial_sq, scratch);
  const float inv_c = 1.f / static_cast<float>(C);
  const float mean = sum * inv_c;
  const float var = sum_sq * inv_c - mean * mean;
  const float inv_std = rsqrtf(var + kLnEps);
  for (int k = t; k < C; k += THREADS) {
    const float xhat = (ld(src + k) - mean) * inv_std;
    dst[k] = rn(xhat * ld(w + k) + ld(b + k));
  }
  __syncthreads();
}

// ---------------------------------------------------------------------------
// Kernel 1: per-token projections read only from the block's input m.
//
// One CTA per (sequence, residue) token. Produces everything downstream needs
// that depends on m alone: OuterProductMean's masked a/b, and MSA row attention's
// value and gate projections. Hoisting v/g here (rather than recomputing them in
// the attention kernel, which needs all N residues' values) is what removes the
// cross-CTA read of m, and with it the need to double-buffer m at all.
// ---------------------------------------------------------------------------
template <int THREADS>
__global__ void msa_prep_kernel(
    const bf16* __restrict__ m,          // [S, N, CM]
    const bf16* __restrict__ msa_mask,   // [S, N]
    const bf16* __restrict__ wt,         // packed weights for this block
    const int* __restrict__ off,
    bf16* __restrict__ a_out,            // [S, N, HOPM]
    bf16* __restrict__ b_out,            // [S, N, HOPM]
    bf16* __restrict__ v_out,            // [S, N, NH*HD] or null
    bf16* __restrict__ g_out,            // [S, N, NH*HD] or null
    int S, int N, int CM, int HOPM, int NH_HD, bool with_msa) {
  extern __shared__ char smem_raw[];
  float* scratch = reinterpret_cast<float*>(smem_raw);
  bf16* xn = reinterpret_cast<bf16*>(scratch + THREADS);

  const int token = blockIdx.x;             // s * N + i
  const int t = threadIdx.x;
  const bf16* mrow = m + static_cast<long>(token) * CM;

  // --- OuterProductMean's a and b, from its own LayerNorm of m.
  layer_norm<THREADS>(mrow, wt + off[OPM_LN_W], wt + off[OPM_LN_B], xn, CM, scratch);
  const float mask_v = ld(msa_mask + token);
  const bf16* w1t = wt + off[OPM_W1T];
  const bf16* w2t = wt + off[OPM_W2T];
  for (int d = t; d < HOPM; d += THREADS) {
    float acc1 = 0.f, acc2 = 0.f;
    for (int k = 0; k < CM; ++k) {
      const float x = ld(xn + k);
      acc1 += x * ld(w1t + k * HOPM + d);
      acc2 += x * ld(w2t + k * HOPM + d);
    }
    // F.linear rounds to bf16, then `* mask` rounds again. Both are kept: the
    // product of a rounded projection with a rounded mask is not the rounded
    // product of the unrounded pair once the mask is not exactly 1.
    a_out[static_cast<long>(token) * HOPM + d] = rn(__bfloat162float(rn(acc1)) * mask_v);
    b_out[static_cast<long>(token) * HOPM + d] = rn(__bfloat162float(rn(acc2)) * mask_v);
  }

  if (!with_msa) return;

  // --- MSA row attention's value and gate projections, from its own LayerNorm.
  __syncthreads();
  layer_norm<THREADS>(mrow, wt + off[ATT_LNM_W], wt + off[ATT_LNM_B], xn, CM, scratch);
  const bf16* wvt = wt + off[ATT_WVT];
  const bf16* wgt = wt + off[ATT_WGT];
  for (int d = t; d < NH_HD; d += THREADS) {
    float accv = 0.f, accg = 0.f;
    for (int k = 0; k < CM; ++k) {
      const float x = ld(xn + k);
      accv += x * ld(wvt + k * NH_HD + d);
      accg += x * ld(wgt + k * NH_HD + d);
    }
    v_out[static_cast<long>(token) * NH_HD + d] = rn(accv);
    // sigmoid of the *rounded* projection, then rounded again.
    const float gpre = __bfloat162float(rn(accg));
    g_out[static_cast<long>(token) * NH_HD + d] = rn(1.f / (1.f + __expf(-gpre)));
  }
}

// ---------------------------------------------------------------------------
// Kernel 2: OuterProductMean's contraction and residual, plus the pair bias.
//
// One CTA per pair (i, j). It is the only stage with a cross-CTA read: the outer
// product needs a[s,i,:] and b[s,j,:] for every sequence s, which msa_prep wrote
// from a different partition -- hence the kernel boundary before this one.
//
// The pair bias rides along at no extra boundary, because MSA row attention's bias
// for pair (i,j) is a function of z1[i,j,:] alone: linear_z(layer_norm_z(z))[i,j,h].
// The CTA that just produced z1[i,j,:] is exactly the one that can normalize it.
// ---------------------------------------------------------------------------
template <int THREADS>
__global__ void opm_z_bias_kernel(
    const bf16* __restrict__ z0,         // [N, N, CZ]
    const bf16* __restrict__ a,          // [S, N, HOPM]
    const bf16* __restrict__ b,          // [S, N, HOPM]
    const bf16* __restrict__ msa_mask,   // [S, N]
    const bf16* __restrict__ pair_mask,  // [N, N]
    const bf16* __restrict__ wt,
    const int* __restrict__ off,
    bf16* __restrict__ z1,               // [N, N, CZ]
    bf16* __restrict__ bias,             // [NH, N, N] or null
    int S, int N, int CZ, int HOPM, int NH, float eps, float inf, bool with_msa) {
  extern __shared__ char smem_raw[];
  float* scratch = reinterpret_cast<float*>(smem_raw);
  bf16* outer = reinterpret_cast<bf16*>(scratch + THREADS);      // [HOPM*HOPM]
  bf16* zrow = outer + HOPM * HOPM;                              // [CZ]
  bf16* znorm = zrow + CZ;                                       // [CZ]

  const int pair = blockIdx.x;
  const int i = pair / N;
  const int j = pair - i * N;
  const int t = threadIdx.x;
  const int flat = HOPM * HOPM;

  // outer[c*HOPM + e] = sum_s a[s,i,c] * b[s,j,e], fp32 accumulate, one round.
  // c-major, matching the baseline's reshape of [..., c, e].
  for (int k = t; k < flat; k += THREADS) {
    const int c = k / HOPM;
    const int e = k - c * HOPM;
    float acc = 0.f;
    for (int s = 0; s < S; ++s) {
      acc += ld(a + (static_cast<long>(s) * N + i) * HOPM + c)
           * ld(b + (static_cast<long>(s) * N + j) * HOPM + e);
    }
    outer[k] = rn(acc);
  }

  // norm = einsum over the mask, accumulated in fp32 from bf16 and rounded, then
  // eps added and rounded *in bf16*. Adding eps in fp32 changes norm and is a
  // detectable per-sub-op difference even where it stays inside tolerance.
  float norm_partial = 0.f;
  for (int s = t; s < S; s += THREADS) {
    norm_partial += ld(msa_mask + static_cast<long>(s) * N + i)
                  * ld(msa_mask + static_cast<long>(s) * N + j);
  }
  __syncthreads();
  const float norm_sum = block_sum<THREADS>(norm_partial, scratch);
  const float norm = __bfloat162float(rn(__bfloat162float(rn(norm_sum)) + eps));
  __syncthreads();

  // z1[i,j,d] = z0[i,j,d] + (linear_out(outer)[d] / norm). linear_out carries the
  // operator's only Linear bias, and it is applied before the division.
  const bf16* woutt = wt + off[OPM_WOUTT];
  const bf16* obias = wt + off[OPM_OUT_BIAS];
  const long base = static_cast<long>(pair) * CZ;
  for (int d = t; d < CZ; d += THREADS) {
    float acc = 0.f;
    for (int k = 0; k < flat; ++k) {
      acc += __bfloat162float(outer[k]) * ld(woutt + static_cast<long>(k) * CZ + d);
    }
    const float projected = __bfloat162float(rn(acc + ld(obias + d)));
    // A true divide, not a multiply by a reciprocal: the baseline's `outer / norm`
    // rounds once, and `projected * (1/norm)` rounds twice in fp32 before rounding to
    // bf16. The two agree bitwise on 0/1 masks -- which is all the harness generates --
    // so this costs nothing observable and removes a latent difference.
    const float divided = __bfloat162float(rn(projected / norm));
    const bf16 out = rn(ld(z0 + base + d) + divided);
    z1[base + d] = out;
    if (with_msa) zrow[d] = out;
  }

  if (!with_msa) return;
  __syncthreads();

  // Pair bias for this (i,j): linear_z(layer_norm_z(z1[i,j,:]))[h], plus the mask
  // bias inf*(mask-1) with i the row dim and j the key dim, added in bf16.
  layer_norm<THREADS>(zrow, wt + off[ATT_LNZ_W], wt + off[ATT_LNZ_B], znorm, CZ,
                      scratch);
  const bf16* wzt = wt + off[ATT_WZT];
  // `self.inf * (mask - 1)` on a bf16 tensor with a Python-float scalar: `mask - 1`
  // materializes as bf16 first, then the scale rounds again. Both roundings are kept.
  // Exact for the 0/1 masks the harness generates, and correct for any other.
  const float mask_minus_one = __bfloat162float(rn(ld(pair_mask + pair) - 1.f));
  const float mask_bias = __bfloat162float(rn(inf * mask_minus_one));
  for (int h = t; h < NH; h += THREADS) {
    float acc = 0.f;
    for (int k = 0; k < CZ; ++k) acc += __bfloat162float(znorm[k]) * ld(wzt + k * NH + h);
    bias[(static_cast<long>(h) * N + i) * N + j] =
        rn(__bfloat162float(rn(acc)) + mask_bias);
  }
}

// ---------------------------------------------------------------------------
// Kernel 3: the weighted average, the gate, the output projection, the residual.
//
// One CTA per query token (s, i). Reads bias[:, i, :] and v[s, :, :] -- both across
// the whole residue axis, written by other CTAs, which is why this is its own
// kernel. Writes only its own token of m1.
// ---------------------------------------------------------------------------
template <int THREADS>
__global__ void msa_attn_out_kernel(
    const bf16* __restrict__ m,      // [S, N, CM]
    const bf16* __restrict__ bias,   // [NH, N, N]
    const bf16* __restrict__ v,      // [S, N, NH*HD]
    const bf16* __restrict__ g,      // [S, N, NH*HD]
    const bf16* __restrict__ wt,
    const int* __restrict__ off,
    bf16* __restrict__ m1,           // [S, N, CM]
    int S, int N, int CM, int NH, int HD) {
  extern __shared__ char smem_raw[];
  float* scratch = reinterpret_cast<float*>(smem_raw);
  bf16* weights = reinterpret_cast<bf16*>(scratch + THREADS);  // [NH][N]
  bf16* gated = weights + NH * N;                              // [NH*HD]

  const int token = blockIdx.x;
  const int s = token / N;
  const int i = token - s * N;
  const int t = threadIdx.x;
  const int nh_hd = NH * HD;

  // Softmax over the key axis, per head: max-subtracted, fp32 accumulation, and the
  // weights round to bf16 before the value contraction -- what F.softmax on a bf16
  // input produces.
  for (int h = t; h < NH; h += THREADS) {
    const bf16* row = bias + (static_cast<long>(h) * N + i) * N;
    float mx = -INFINITY;
    for (int k = 0; k < N; ++k) mx = fmaxf(mx, ld(row + k));
    float denom = 0.f;
    for (int k = 0; k < N; ++k) denom += __expf(ld(row + k) - mx);
    const float inv = 1.f / denom;
    for (int k = 0; k < N; ++k) weights[h * N + k] = rn(__expf(ld(row + k) - mx) * inv);
  }
  __syncthreads();

  // o[h,c] = sum_k weights[h,k] * v[s,k,h,c], then gated and flattened head-major,
  // matching the baseline's einsum("...hqk,...hkc->...qhc") followed by o * g.
  for (int d = t; d < nh_hd; d += THREADS) {
    const int h = d / HD;
    float acc = 0.f;
    for (int k = 0; k < N; ++k) {
      acc += __bfloat162float(weights[h * N + k])
           * ld(v + (static_cast<long>(s) * N + k) * nh_hd + d);
    }
    const float o = __bfloat162float(rn(acc));
    gated[d] = rn(o * ld(g + static_cast<long>(token) * nh_hd + d));
  }
  __syncthreads();

  const bf16* wot = wt + off[ATT_WOT];
  const long base = static_cast<long>(token) * CM;
  for (int d = t; d < CM; d += THREADS) {
    float acc = 0.f;
    for (int k = 0; k < nh_hd; ++k)
      acc += __bfloat162float(gated[k]) * ld(wot + k * CM + d);
    m1[base + d] = rn(ld(m + base + d) + __bfloat162float(rn(acc)));
  }
}

// ---------------------------------------------------------------------------
// Kernel 4: the SwiGLU transition and its residual, in place.
//
// One CTA per token, entirely row-local, so it may update m1 in place: no other CTA
// reads this token. The block passes mask=None, so the transition's trailing
// multiply is by ones and is dropped -- pair_transition's multiply, which does take a
// mask, lives inside the frozen pair-block winner and is untouched.
// ---------------------------------------------------------------------------
template <int THREADS>
__global__ void msa_transition_kernel(
    bf16* __restrict__ m1,           // [S, N, CM], updated in place
    const bf16* __restrict__ wt,
    const int* __restrict__ off,
    int CM, int TR_HID) {
  extern __shared__ char smem_raw[];
  float* scratch = reinterpret_cast<float*>(smem_raw);
  bf16* xn = reinterpret_cast<bf16*>(scratch + THREADS);  // [CM]
  bf16* hid = xn + CM;                                    // [TR_HID]

  const int token = blockIdx.x;
  const int t = threadIdx.x;
  bf16* row = m1 + static_cast<long>(token) * CM;

  layer_norm<THREADS>(row, wt + off[TR_LN_W], wt + off[TR_LN_B], xn, CM, scratch);

  const bf16* wat = wt + off[TR_WAT];
  const bf16* wbt = wt + off[TR_WBT];
  for (int d = t; d < TR_HID; d += THREADS) {
    float acc_a = 0.f, acc_b = 0.f;
    for (int k = 0; k < CM; ++k) {
      const float x = __bfloat162float(xn[k]);
      acc_a += x * ld(wat + k * TR_HID + d);
      acc_b += x * ld(wbt + k * TR_HID + d);
    }
    // silu(linear_a(x)) * linear_b(x): the projection rounds, silu rounds, and the
    // product rounds. Three rounding points, all of which the baseline has.
    const float a_r = __bfloat162float(rn(acc_a));
    const float silu = __bfloat162float(rn(a_r / (1.f + __expf(-a_r))));
    hid[d] = rn(silu * __bfloat162float(rn(acc_b)));
  }
  __syncthreads();

  const bf16* woutt = wt + off[TR_WOUTT];
  for (int d = t; d < CM; d += THREADS) {
    float acc = 0.f;
    for (int k = 0; k < TR_HID; ++k)
      acc += __bfloat162float(hid[k]) * ld(woutt + k * CM + d);
    row[d] = rn(ld(row + d) + __bfloat162float(rn(acc)));
  }
}

// ---------------------------------------------------------------------------
// Host entry points.
// ---------------------------------------------------------------------------
constexpr int kPrepThreads = 128;
constexpr int kPairThreads = 128;
constexpr int kAttnThreads = 64;
constexpr int kTransThreads = 256;

// One-time opt-in for the dynamic shared memory the pair and transition kernels
// need above the 48 KB default. Idempotent per kernel per device.
template <typename Fn>
void raise_smem_limit(Fn kernel, int bytes) {
  static thread_local std::set<std::pair<const void*, int>> done;
  const void* key = reinterpret_cast<const void*>(kernel);
  if (bytes <= 48 * 1024) return;
  if (done.count({key, bytes})) return;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  done.insert({key, bytes});
}

torch::Tensor layout_op(int64_t c_m, int64_t c_z, int64_t c_hidden_opm,
                        int64_t no_heads_msa, int64_t c_hidden_msa_att,
                        int64_t transition_n) {
  const Layout L = make_layout(c_m, c_z, c_hidden_opm, no_heads_msa,
                               c_hidden_msa_att, transition_n);
  auto out = torch::empty({NUM_SECTIONS + 1, 2}, torch::dtype(torch::kInt32));
  auto acc = out.accessor<int, 2>();
  for (int s = 0; s < NUM_SECTIONS; ++s) {
    acc[s][0] = L.off[s];
    acc[s][1] = L.num[s];
  }
  acc[NUM_SECTIONS][0] = L.total;
  acc[NUM_SECTIONS][1] = NUM_SECTIONS;
  return out;
}

// Runs the MSA half of one block. Returns (m_next, z_next); with `with_msa` false
// (the last block, whose MSA update is dead) `m` is returned unchanged.
std::vector<torch::Tensor> msa_half_op(
    const torch::Tensor& m, const torch::Tensor& z,
    const torch::Tensor& msa_mask, const torch::Tensor& pair_mask,
    const torch::Tensor& packed, const torch::Tensor& offsets,
    int64_t c_hidden_opm, int64_t no_heads_msa, int64_t c_hidden_msa_att,
    int64_t transition_n, double eps, double inf, bool with_msa) {
  const at::cuda::OptionalCUDAGuard guard(at::device_of(m));
  auto stream = at::cuda::getCurrentCUDAStream();

  const int S = m.size(1);
  const int N = m.size(2);
  const int CM = m.size(3);
  const int CZ = z.size(3);
  const int HOPM = static_cast<int>(c_hidden_opm);
  const int NH = static_cast<int>(no_heads_msa);
  const int HD = static_cast<int>(c_hidden_msa_att);
  const int NH_HD = NH * HD;
  const int TR_HID = static_cast<int>(transition_n) * CM;

  const auto opts = m.options();
  auto a = torch::empty({S, N, HOPM}, opts);
  auto b = torch::empty({S, N, HOPM}, opts);
  auto z1 = torch::empty_like(z);

  const bf16* wt = reinterpret_cast<const bf16*>(packed.data_ptr());
  const int* off = offsets.data_ptr<int>();
  const bf16* mp = reinterpret_cast<const bf16*>(m.data_ptr());
  const bf16* zp = reinterpret_cast<const bf16*>(z.data_ptr());
  const bf16* mmp = reinterpret_cast<const bf16*>(msa_mask.data_ptr());
  const bf16* pmp = reinterpret_cast<const bf16*>(pair_mask.data_ptr());

  torch::Tensor v, g, bias, m1;
  bf16 *vp = nullptr, *gp = nullptr, *biasp = nullptr;
  if (with_msa) {
    v = torch::empty({S, N, NH_HD}, opts);
    g = torch::empty({S, N, NH_HD}, opts);
    bias = torch::empty({NH, N, N}, opts);
    vp = reinterpret_cast<bf16*>(v.data_ptr());
    gp = reinterpret_cast<bf16*>(g.data_ptr());
    biasp = reinterpret_cast<bf16*>(bias.data_ptr());
  }

  // 1. Per-token projections from m.
  {
    const int smem = kPrepThreads * sizeof(float) + CM * sizeof(bf16);
    msa_prep_kernel<kPrepThreads><<<S * N, kPrepThreads, smem, stream>>>(
        mp, mmp, wt, off, reinterpret_cast<bf16*>(a.data_ptr()),
        reinterpret_cast<bf16*>(b.data_ptr()), vp, gp,
        S, N, CM, HOPM, NH_HD, with_msa);
  }

  // 2. OPM contraction, residual into z, and the pair bias.
  {
    const int smem = kPairThreads * sizeof(float)
                   + (HOPM * HOPM + 2 * CZ) * sizeof(bf16);
    raise_smem_limit(opm_z_bias_kernel<kPairThreads>, smem);
    opm_z_bias_kernel<kPairThreads><<<N * N, kPairThreads, smem, stream>>>(
        zp, reinterpret_cast<const bf16*>(a.data_ptr()),
        reinterpret_cast<const bf16*>(b.data_ptr()), mmp, pmp, wt, off,
        reinterpret_cast<bf16*>(z1.data_ptr()), biasp,
        S, N, CZ, HOPM, NH, static_cast<float>(eps), static_cast<float>(inf),
        with_msa);
  }

  if (!with_msa) return {m, z1};

  // 3. Weighted average, gate, output projection, residual into a fresh m1.
  m1 = torch::empty_like(m);
  {
    const int smem = kAttnThreads * sizeof(float) + (NH * N + NH_HD) * sizeof(bf16);
    raise_smem_limit(msa_attn_out_kernel<kAttnThreads>, smem);
    msa_attn_out_kernel<kAttnThreads><<<S * N, kAttnThreads, smem, stream>>>(
        mp, biasp, reinterpret_cast<const bf16*>(v.data_ptr()),
        reinterpret_cast<const bf16*>(g.data_ptr()), wt, off,
        reinterpret_cast<bf16*>(m1.data_ptr()), S, N, CM, NH, HD);
  }

  // 4. SwiGLU transition, in place on m1.
  {
    const int smem = kTransThreads * sizeof(float) + (CM + TR_HID) * sizeof(bf16);
    raise_smem_limit(msa_transition_kernel<kTransThreads>, smem);
    msa_transition_kernel<kTransThreads><<<S * N, kTransThreads, smem, stream>>>(
        reinterpret_cast<bf16*>(m1.data_ptr()), wt, off, CM, TR_HID);
  }

  return {m1, z1};
}

}  // namespace

TORCH_LIBRARY(fk_af3_msa_module, lib) {
  lib.def("layout(int c_m, int c_z, int c_hidden_opm, int no_heads_msa, "
          "int c_hidden_msa_att, int transition_n) -> Tensor");
  lib.def("msa_half(Tensor m, Tensor z, Tensor msa_mask, Tensor pair_mask, "
          "Tensor packed, Tensor offsets, int c_hidden_opm, int no_heads_msa, "
          "int c_hidden_msa_att, int transition_n, float eps, float inf, "
          "bool with_msa) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(fk_af3_msa_module, CompositeExplicitAutograd, lib) {
  lib.impl("layout", TORCH_FN(layout_op));
}

TORCH_LIBRARY_IMPL(fk_af3_msa_module, CUDA, lib) {
  lib.impl("msa_half", TORCH_FN(msa_half_op));
}
