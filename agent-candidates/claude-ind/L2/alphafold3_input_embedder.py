"""Input embedder for AlphaFold3.

Produces initial single (s) and pair (z) representations from token and
atom features.

The all-atom path (``batch`` carrying ``ref_pos``) is the one that shows up in
captures and it is pathologically launch-bound: the reference composition of
AtomAttentionEncoder + relpos issues ~1275 CUDA kernels for 368 atoms / 16
tokens, so ~1 GFLOP of arithmetic costs milliseconds of launch latency.

This candidate keeps the reference module tree (same submodule names, so weight
loading is unchanged) but, for the captured geometry, folds the whole forward
into 12 custom kernels over pre-allocated buffers and replays them from a CUDA
graph. Anything else -- different shapes, no ``batch``, a failed JIT build --
falls through to the reference implementation below.

Reference: openfold3/core/model/feature_embedders/input_embedders.py
           InputEmbedderAllAtom
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.tensor_ops import OneHot, Pad
from .alphafold3_atom_attention import AtomAttentionEncoder

# --------------------------------------------------------------------------- #
# Fused CUDA fast path
# --------------------------------------------------------------------------- #

# geometry the kernels are specialised for
_NA, _AP, _CA, _CP, _NBB, _NQ, _NK, _NH, _DH = 368, 384, 128, 16, 12, 32, 128, 4, 32
_TT, _CTOK, _CSI, _CZ, _NTR, _KFEAT = 16, 384, 449, 128, 256, 384
_CSIP, _NTAIL = 464, 640

_CPP_SRC = """
void setup(std::vector<torch::Tensor>, std::vector<torch::Tensor>);
void ingest(std::vector<torch::Tensor>);
void run(int64_t);
"""

_CUDA_SRC = r'''// Fused AlphaFold3 InputEmbedder (all-atom path) for one captured geometry:
//   N_atom=368, N_token=16, c_atom=128, c_atom_pair=16, 4 heads x 32, n_query=32,
//   n_key=128, 3 atom-transformer blocks, c_token=384, c_s_input=449, c_z=128.
//
// The problem is ~1 GFLOP over 384 atoms, so nothing here is compute-bound: there
// are only 24 row-tiles of work, so a kernel gets a handful of warps per SM and has
// almost nothing to hide a latency behind. Three things dominated the measurements
// and shape every kernel below:
//   * wmma B-fragments read straight from global memory cost ~10x what they should;
//     each weight block is staged into shared memory by one bulk cp.async instead.
//   * every shared tile feeding ldmatrix is padded (PADA/PADB). On a power-of-two
//     row stride all 16 rows of a fragment land in one bank -> 16-way conflict.
//   * two ldmatrix per mma makes shared-memory bandwidth, not the tensor cores, the
//     limit, so each warp holds several output tiles to amortise the A fragment.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <mma.h>

namespace wm = nvcuda::wmma;
typedef __nv_bfloat16 bf16;

#define NA 368          // real atoms
#define AP 384          // atoms padded to a multiple of NQ
#define CA 128          // c_atom
#define CP 16           // c_atom_pair
#define NBB 12          // atom blocks
#define NQ 32
#define NK 128
#define NH 4            // heads
#define DH 32           // per-head dim
#define TT 16           // tokens
#define CTOK 384
#define CSI 449
#define CSIP 464        // CSI padded to a multiple of 16
#define CZ 128
#define NTR 256         // swiglu hidden
#define NTAIL 640       // c_token + 2 * c_z, the fused tail GEMM width
#define NTSZ 64         // tail GEMM columns per CTA
#define KFEAT 384       // padded ref-feature width (380 real)
#define NELEM 119
#define NCHAR 256
#define MT (AP / 16)    // row tiles
#define ALD1 (CA + PADA)
#define ALD2 (NTR + PADA)
#define LN_EPS 1e-5f
#ifndef NTH
#define NTH 512          // threads per CTA for the tiled kernels
#endif
#ifndef UNROLLK
#define UNROLLK 4        // wmma K-loop unroll; too high spills the fragments
#endif
#define NWARP (NTH / 32)
#ifndef MR_COND
#define MR_COND 2        // row-tiles per CTA in the conditioning kernel
#endif
#ifndef NTH_ATT
#define NTH_ATT 1024     // attention has more independent warps to feed
#endif
#ifndef PADA
#define PADA 8           // same padding for the A-operand tiles
#endif
#ifndef PADB
#define PADB 8           // shared-memory row padding: ldmatrix on an unpadded
#endif               // power-of-two stride hits 16-way bank conflicts
#define AF3_STR(x) #x
#define AF3_XSTR(x) AF3_STR(x)
#define AF3_UNROLL(n) _Pragma(AF3_XSTR(unroll n))

using facc = wm::fragment<wm::accumulator, 16, 16, 16, float>;
using fmA = wm::fragment<wm::matrix_a, 16, 16, 16, bf16, wm::row_major>;
using fmB = wm::fragment<wm::matrix_b, 16, 16, 16, bf16, wm::row_major>;
using fmBc = wm::fragment<wm::matrix_b, 16, 16, 16, bf16, wm::col_major>;

// ------------------------------------------------------------- GEMM primitives
// Asynchronous 16B global->shared copy. Staging the weight blocks is the binding
// cost in these kernels (one SM must pull ~350KB per transformer block), and
// cp.async keeps far more requests in flight than a load/store pair per thread.
__device__ __forceinline__ void cpa16(void* dst, const void* src) {
  const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(d), "l"(src));
}
__device__ __forceinline__ void cpa_wait() {
  asm volatile("cp.async.commit_group;\n");
  asm volatile("cp.async.wait_group 0;\n");
}
// Bulk copy of a [K x NT] bf16 weight block (row stride LDB) into shared memory.
template <int NT, int K, int LDB>
__device__ __forceinline__ void stageB(bf16* __restrict__ Bs, const bf16* __restrict__ B) {
  const int tid = threadIdx.x, nt = blockDim.x;
  constexpr int NTP = NT + PADB;
  for (int i = tid; i < (K * NT) / 8; i += nt) {
    const int r = (i * 8) / NT, c = (i * 8) % NT;
    cpa16(Bs + (size_t)r * NTP + c, B + (size_t)r * LDB + c);
  }
  cpa_wait();
}

// C[16*MR x NT] = A[16*MR x K] * Bs[K x NT], all operands in shared memory.
// MR > 1 lets one CTA cover more atom rows, which divides both the CTA count and the
// (redundant, one copy per row-tile) weight staging traffic by MR.
template <int MR, int NT, int K, int LDA, int LDC, int NW>
__device__ __forceinline__ void mmaTile(const bf16* __restrict__ A,
                                        const bf16* __restrict__ Bs,
                                        float* __restrict__ C) {
  const int warp = threadIdx.x >> 5;
  constexpr int NTILE = NT / 16;
  constexpr int TOT = MR * NTILE;
  // Two ldmatrix per mma makes shared-memory bandwidth, not the tensor cores, the
  // limit here. Holding TPW output tiles per warp amortises one A fragment over TPW
  // mmas, so aim for TPW ~ 4 even when that leaves warps out of the mma (they have
  // already done their share of the staging).
  constexpr int NWM = (TOT / 4 >= 4) ? (TOT / 4 <= NW ? TOT / 4 : NW)
                                     : (TOT >= 4 ? 4 : TOT);
  constexpr int TPW = TOT / NWM;
  if (warp >= NWM) return;
  facc acc[TPW];
#pragma unroll
  for (int u = 0; u < TPW; ++u) wm::fill_fragment(acc[u], 0.0f);
  AF3_UNROLL(UNROLLK)
  for (int k = 0; k < K / 16; ++k) {
    fmA af[MR];
#pragma unroll
    for (int m = 0; m < MR; ++m)
      wm::load_matrix_sync(af[m], A + (size_t)m * 16 * LDA + k * 16, LDA);
#pragma unroll
    for (int u = 0; u < TPW; ++u) {
      const int tile = warp + u * NWM;
      const int mt = tile / NTILE, nn = tile % NTILE;
      fmB bfr;
      wm::load_matrix_sync(bfr, Bs + (size_t)(k * 16) * (NT + PADB) + nn * 16, NT + PADB);
      wm::mma_sync(acc[u], af[mt], bfr, acc[u]);
    }
  }
#pragma unroll
  for (int u = 0; u < TPW; ++u) {
    const int tile = warp + u * NWM;
    const int mt = tile / NTILE, nn = tile % NTILE;
    wm::store_matrix_sync(C + (size_t)mt * 16 * LDC + nn * 16, acc[u], LDC, wm::mem_row_major);
  }
}

#define GEMM(MR, NT, K, LDA, LDC, LDB, NW, A, B, C, Bs)  \
  do {                                                   \
    stageB<NT, K, LDB>(Bs, B);                           \
    __syncthreads();                                     \
    mmaTile<MR, NT, K, LDA, LDC, NW>(A, Bs, C);          \
    __syncthreads();                                     \
  } while (0)

// row-wise LayerNorm (no affine) over a [ROWS x CA] fp32 shared tile
template <int ROWS>
__device__ __forceinline__ void ln16(const float* __restrict__ src, float* __restrict__ dst) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwarps = blockDim.x >> 5;
  for (int r = warp; r < ROWS; r += nwarps) {
    const float* p = src + r * CA;
    float s = 0.f;
#pragma unroll
    for (int i = lane; i < CA; i += 32) s += p[i];
#pragma unroll
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    const float m = s * (1.0f / CA);
    float s2 = 0.f;
#pragma unroll
    for (int i = lane; i < CA; i += 32) { const float d = p[i] - m; s2 += d * d; }
#pragma unroll
    for (int o = 16; o; o >>= 1) s2 += __shfl_xor_sync(0xffffffff, s2, o);
    const float rs = rsqrtf(s2 * (1.0f / CA) + LN_EPS);
    for (int i = lane; i < CA; i += 32) dst[r * CA + i] = (p[i] - m) * rs;
  }
}

struct Params {
  const bf16 *w_cl, *w_condh, *w_condc, *w_lq, *w_tail, *w_l, *w_m;
  const bf16 *w_qg[3], *w_kv[3], *w_o[3], *w_ab[3], *w_out[3];
  const float *b_condh, *b_condc, *b_qg[3];
  const float *w_ro, *w_iq, *w_vm, *w_p[3], *w_lnz, *w_z[3];
  const float *P1, *P2, *P3, *wr132, *wtb;
  bf16 *X, *clb, *clhb, *ogb;
  float *cl, *clh, *ul, *vm, *bmask, *maskv;
  bf16 *condh, *condc, *zb;
  float *a, *ai, *cnt;
  bf16 *qq, *gg, *kk, *vv;
  float *ampad, *rppad, *uidpad, *tfs, *ri, *ti, *asym, *ent, *sym, *tb;
  float *zi, *zj;
  int *safe, *a2ti;
  bf16 *o_sinput, *o_s, *o_z;
};
__constant__ Params gp;
static Params hp;

// -------------------------------------------------------------------- ingest
// CTA r < AP   : one padded atom row of the ref-feature matrix X
// CTA AP + b   : key-index table and block masks for atom block b
// CTA AP + NBB : token-level scalars
__global__ void k_ingest(const bf16* __restrict__ ref_pos, const bf16* __restrict__ ref_charge,
                         const bf16* __restrict__ ref_mask, const bf16* __restrict__ ref_elem,
                         const bf16* __restrict__ ref_chars, const bf16* __restrict__ ref_uid,
                         const bf16* __restrict__ atom_mask, const int64_t* __restrict__ a2t,
                         const bf16* __restrict__ tokfeat, const bf16* __restrict__ res_idx,
                         const bf16* __restrict__ tok_idx, const bf16* __restrict__ asym_id,
                         const bf16* __restrict__ entity_id, const bf16* __restrict__ sym_id,
                         const bf16* __restrict__ tok_bonds) {
  const int cta = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
  if (cta < AP) {
    const int r = cta;
    bf16* xr = gp.X + (size_t)r * KFEAT;
    if (r >= NA) {
      for (int i = tid; i < KFEAT; i += nt) xr[i] = __float2bfloat16(0.f);
      if (tid == 0) {
        gp.ampad[r] = 0.f; gp.uidpad[r] = 0.f; gp.a2ti[r] = -1;
        gp.rppad[r * 3] = gp.rppad[r * 3 + 1] = gp.rppad[r * 3 + 2] = 0.f;
      }
      return;
    }
    for (int i = tid; i < KFEAT; i += nt) {
      float v;
      if (i < 3)              v = __bfloat162float(ref_pos[r * 3 + i]);
      else if (i == 3)        v = asinhf(__bfloat162float(ref_charge[r]));
      else if (i == 4)        v = __bfloat162float(ref_mask[r]);
      else if (i < 5 + NELEM) v = __bfloat162float(ref_elem[(size_t)r * NELEM + (i - 5)]);
      else if (i < 380)       v = __bfloat162float(ref_chars[(size_t)r * NCHAR + (i - 124)]);
      else                    v = 0.f;
      xr[i] = __float2bfloat16(v);
    }
    if (tid == 0) {
      gp.ampad[r] = __bfloat162float(atom_mask[r]);
      gp.uidpad[r] = __bfloat162float(ref_uid[r]);
      gp.a2ti[r] = (int)a2t[r];
      for (int i = 0; i < 3; ++i) gp.rppad[r * 3 + i] = __bfloat162float(ref_pos[r * 3 + i]);
    }
    return;
  }
  if (cta < AP + NBB) {
    const int b = cta - AP;
    // n_real must reproduce the baseline's bf16 reduction of the padded atom mask
    __shared__ float sred[33];
    float s = 0.f;
    for (int i = tid; i < NA; i += nt) s += __bfloat162float(atom_mask[i]);
#pragma unroll
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    if ((tid & 31) == 0) sred[tid >> 5] = s;
    __syncthreads();
    if (tid == 0) {
      float t = 0.f;
      for (int i = 0; i < (nt >> 5); ++i) t += sred[i];
      sred[32] = t;
    }
    __syncthreads();
    const float n_real = __bfloat162float(__float2bfloat16(sred[32]));
    const float nm1 = __bfloat162float(__float2bfloat16(n_real - 1.0f));
    const int center = NQ / 2 + b * NQ;
    const float under = fmaxf(0.f, (float)(-(center - NK / 2)));
    const float over = fmaxf(0.f, __bfloat162float(__float2bfloat16((float)(center + NK / 2 - 1))) - nm1);
    const float shift = under > 0.f ? under : -over;
    const float hi = fmaxf(nm1, 0.f);
    for (int m = tid; m < NK; m += nt) {
      const int iv = center - NK / 2 + m;
      const float fin = __bfloat162float(__float2bfloat16(
          __bfloat162float(__float2bfloat16((float)iv)) + shift));
      const bool bad = (fin < 0.f) || (fin >= n_real);
      const int si = (int)fminf(fmaxf(fin, 0.f), hi);
      gp.safe[b * NK + m] = si;
      gp.maskv[b * NK + m] = bad ? 0.f : 1.f;
      const float mk = bad ? 0.f : ((si < NA) ? __bfloat162float(atom_mask[si]) : 0.f);
      for (int l = 0; l < NQ; ++l) {
        const int ga = b * NQ + l;
        const float mq = (ga < NA) ? __bfloat162float(atom_mask[ga]) : 0.f;
        gp.bmask[(b * NQ + l) * NK + m] = mq * mk;
      }
    }
    return;
  }
  for (int i = tid; i < TT; i += nt) {
    gp.ri[i] = __bfloat162float(res_idx[i]);
    gp.ti[i] = __bfloat162float(tok_idx[i]);
    gp.asym[i] = __bfloat162float(asym_id[i]);
    gp.ent[i] = __bfloat162float(entity_id[i]);
    gp.sym[i] = __bfloat162float(sym_id[i]);
  }
  for (int i = tid; i < TT * TT; i += nt) gp.tb[i] = __bfloat162float(tok_bonds[i]);
  for (int i = tid; i < TT * 65; i += nt) {
    const int t = i / 65, c = i % 65;
    gp.tfs[i] = __bfloat162float(tokfeat[(size_t)t * CTOK + (c < 64 ? c : CTOK - 1)]);
  }
}

// ---------------------------------------- K1: cl, cl_hat, ul, vm (grid MT)
__global__ void k_cl() {
  extern __shared__ char smem[];
  float* Cs = (float*)smem;                        // [16][CA]
  float* Hs = Cs + 16 * CA;                        // [16][CA]
  bf16* Xs = (bf16*)(Hs + 16 * CA);                // [16][KFEAT + PADA]
  bf16* Bs = Xs + 16 * (KFEAT + PADA);             // [KFEAT][CA + PADB]
  const int r0 = blockIdx.x * 16, tid = threadIdx.x, nt = blockDim.x;
  if (blockIdx.x == 0) {
    for (int i = tid; i < TT * CTOK; i += nt) gp.ai[i] = 0.f;
    for (int i = tid; i < TT; i += nt) gp.cnt[i] = 0.f;
  }
  for (int i = tid; i < (16 * KFEAT) / 8; i += nt) {
    const int r = (i * 8) / KFEAT, c = (i * 8) % KFEAT;
    *(int4*)(Xs + r * (KFEAT + PADA) + c) = *(const int4*)(gp.X + (size_t)(r0 + r) * KFEAT + c);
  }
  GEMM(1, CA, KFEAT, KFEAT + PADA, CA, CA, NWARP, Xs, gp.w_cl, Cs, Bs);
  float* clg = gp.cl + (size_t)r0 * CA;
  bf16* clbg = gp.clb + (size_t)r0 * CA;
  float* ag = gp.a + (size_t)r0 * CA;
  for (int i = tid; i < 16 * CA; i += nt) {
    const float v = Cs[i];
    clg[i] = v; ag[i] = v; clbg[i] = __float2bfloat16(v);   // a starts as cl.clone()
  }
  ln16<16>(Cs, Hs);
  __syncthreads();
  float* clhg = gp.clh + (size_t)r0 * CA;
  bf16* clhbg = gp.clhb + (size_t)r0 * CA;
  for (int i = tid; i < 16 * CA; i += nt) { clhg[i] = Hs[i]; clhbg[i] = __float2bfloat16(Hs[i]); }
  // ul / vm = relu(cl) @ {Wl, Wm}: 16x16 each, staged weights, plain FMA
  bf16* Ws = Xs;                                   // reuse
  for (int i = tid; i < 2 * CA * CP; i += nt) Ws[i] = (i < CA * CP) ? gp.w_l[i] : gp.w_m[i - CA * CP];
  __syncthreads();
  for (int i = tid; i < 2 * 16 * CP; i += nt) {
    const int which = i / (16 * CP), j = i % (16 * CP), r = j / CP, c = j % CP;
    const bf16* W = Ws + which * CA * CP;
    float acc = 0.f;
#pragma unroll 8
    for (int k = 0; k < CA; ++k) acc += fmaxf(Cs[r * CA + k], 0.f) * __bfloat162float(W[k * CP + c]);
    (which ? gp.vm : gp.ul)[(size_t)(r0 + r) * CP + c] = acc;
  }
}

// ------------------ K2: the 24 AdaLN conditioning projections (grid MT x 24)
//   group g < 18 : block g/6, slot g%6 in {gq,lsq,gk,lsk,gt,lst}, input cl_hat
//   group g >= 18: block (g-18)/2, slot in {ada_out gate, transition gate}, input cl
__global__ void k_cond() {
  constexpr int MR = MR_COND, ROWS = MR * 16;
  extern __shared__ char smem[];
  float* Cs = (float*)smem;                        // [ROWS][CA]
  bf16* As = (bf16*)(Cs + ROWS * CA);              // [ROWS][ALD1]
  bf16* Bs = As + ROWS * ALD1;                     // [CA][CA + PADB]
  const int r0 = blockIdx.x * ROWS, g = blockIdx.y, tid = threadIdx.x, nt = blockDim.x;
  const bool is_cl = g >= 18;
  const bf16* src = (is_cl ? gp.clb : gp.clhb) + (size_t)r0 * CA;
  for (int i = tid; i < (ROWS * CA) / 8; i += nt) {
    const int r = (i * 8) >> 7, c = (i * 8) & (CA - 1);
    *(int4*)(As + r * ALD1 + c) = *(const int4*)(src + (size_t)r * CA + c);
  }
  if (is_cl) GEMM(MR, CA, CA, ALD1, CA, 6 * CA, NWARP, As, gp.w_condc + (g - 18) * CA, Cs, Bs);
  else       GEMM(MR, CA, CA, ALD1, CA, 18 * CA, NWARP, As, gp.w_condh + g * CA, Cs, Bs);
  const int slot = is_cl ? (g - 18) % 2 : g % 6;
  const bool sig = is_cl || slot == 0 || slot == 2 || slot == 4;
  const float* bias = is_cl ? (gp.b_condc + (g - 18) * CA) : (gp.b_condh + g * CA);
  bf16* dst = (is_cl ? gp.condc : gp.condh) + (size_t)r0 * (is_cl ? 6 * CA : 18 * CA) + (is_cl ? (g - 18) : g) * CA;
  const int ld = is_cl ? 6 * CA : 18 * CA;
  for (int i = tid; i < ROWS * CA; i += nt) {
    const int r = i / CA, c = i % CA;
    float v = Cs[i] + bias[c];
    if (sig) v = 1.0f / (1.0f + __expf(-v));
    dst[(size_t)r * ld + c] = __float2bfloat16(v);
  }
}

// ----------------- K3: atom pair representation -> per-head attention bias zb
// grid (NBB, NK/8); one thread per (query l, key m) pair.
__global__ void k_pair() {
  __shared__ float Wro[3 * CP], Wiq[CP], Wvm[CP], Wpa[CP * CP], Wpb[CP * CP], Wpc[CP * CP];
  __shared__ float Wlnz[CP], Wz[3 * CP * NH];
  const int tid = threadIdx.x, nt = blockDim.x;
  const int b = blockIdx.x, m = blockIdx.y * 8 + (tid & 7), l = tid >> 3;
  for (int i = tid; i < 3 * CP; i += nt) Wro[i] = gp.w_ro[i];
  for (int i = tid; i < CP; i += nt) { Wiq[i] = gp.w_iq[i]; Wvm[i] = gp.w_vm[i]; Wlnz[i] = gp.w_lnz[i]; }
  for (int i = tid; i < CP * CP; i += nt) { Wpa[i] = gp.w_p[0][i]; Wpb[i] = gp.w_p[1][i]; Wpc[i] = gp.w_p[2][i]; }
  for (int i = tid; i < 3 * CP * NH; i += nt) Wz[i] = gp.w_z[i / (CP * NH)][i % (CP * NH)];
  __syncthreads();

  const int la = b * NQ + l;
  const int ma = gp.safe[b * NK + m];
  const float vk = gp.maskv[b * NK + m];
  const float bm = gp.bmask[(b * NQ + l) * NK + m];
  float d0 = (gp.rppad[la * 3 + 0] - gp.rppad[ma * 3 + 0] * vk) * bm;
  float d1 = (gp.rppad[la * 3 + 1] - gp.rppad[ma * 3 + 1] * vk) * bm;
  float d2 = (gp.rppad[la * 3 + 2] - gp.rppad[ma * 3 + 2] * vk) * bm;
  const float vlm = (gp.uidpad[la] == gp.uidpad[ma] * vk ? 1.0f : 0.0f) * bm;
  const float inv = 1.0f / (1.0f + d0 * d0 + d1 * d1 + d2 * d2);
  const float* ulp = gp.ul + (size_t)la * CP;
  const float* vmp = gp.vm + (size_t)ma * CP;
  float p[CP], h[CP], o[CP];
#pragma unroll
  for (int c = 0; c < CP; ++c)
    p[c] = (d0 * Wro[c] + d1 * Wro[CP + c] + d2 * Wro[2 * CP + c]) * vlm
         + inv * Wiq[c] * vlm + vlm * Wvm[c] * vlm + (ulp[c] + vmp[c] * vk) * bm;
  // pair MLP: three ReLU + 16x16 layers, kept in registers (no pointer aliasing)
#pragma unroll
  for (int c = 0; c < CP; ++c) {
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < CP; ++k) acc += fmaxf(p[k], 0.f) * Wpa[k * CP + c];
    h[c] = acc;
  }
#pragma unroll
  for (int c = 0; c < CP; ++c) {
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < CP; ++k) acc += fmaxf(h[k], 0.f) * Wpb[k * CP + c];
    o[c] = acc;
  }
#pragma unroll
  for (int c = 0; c < CP; ++c) {
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < CP; ++k) acc += fmaxf(o[k], 0.f) * Wpc[k * CP + c];
    h[c] = acc;
  }
  float mean = 0.f, var = 0.f;
#pragma unroll
  for (int c = 0; c < CP; ++c) { p[c] = (p[c] + h[c]) * bm; mean += p[c]; }
  mean *= 1.0f / CP;
#pragma unroll
  for (int c = 0; c < CP; ++c) { const float t = p[c] - mean; var += t * t; }
  const float rs = rsqrtf(var * (1.0f / CP) + LN_EPS);
#pragma unroll
  for (int c = 0; c < CP; ++c) p[c] = (p[c] - mean) * rs * Wlnz[c];
  bf16* zb = gp.zb;
#pragma unroll
  for (int j = 0; j < 3; ++j)
#pragma unroll
    for (int hh = 0; hh < NH; ++hh) {
      float acc = 0.f;
#pragma unroll
      for (int c = 0; c < CP; ++c) acc += p[c] * Wz[j * CP * NH + c * NH + hh];
      zb[(((size_t)j * NBB + b) * NH + hh) * (NQ * NK) + l * NK + m] =
          __float2bfloat16(acc + 1e9f * (bm - 1.0f));
    }
}

// --------------------------------- K4: sequence-local attention, grid (NBB, NH)
__global__ void k_attn(int j) {
  extern __shared__ char smem[];
  constexpr int LDD = DH + PADA, LDK = NK + PADA;
  bf16* Qs = (bf16*)smem;                    // [NQ][LDD]
  bf16* Kt = Qs + NQ * LDD;                  // [NK][LDD]
  bf16* Ps = Kt + NK * LDD;                  // [NQ][LDK]
  bf16* Vs = Ps + NQ * LDK;                  // [NK][LDD]
  float* S = (float*)(Vs + NK * LDD);        // [NQ][NK]
  const int b = blockIdx.x, hh = blockIdx.y;
  const int tid = threadIdx.x, nt = blockDim.x, warp = tid >> 5, nw = nt >> 5;
  const float scale = rsqrtf((float)DH);
  const int* safe = gp.safe + b * NK;
  const float* maskv = gp.maskv + b * NK;
  const bf16* qq = gp.qq; const bf16* kk = gp.kk; const bf16* vv = gp.vv;
  for (int i = tid; i < NQ * DH; i += nt) {
    const int l = i / DH, d = i % DH;
    Qs[l * LDD + d] = __float2bfloat16(__bfloat162float(qq[(size_t)(b * NQ + l) * CA + hh * DH + d]) * scale);
  }
  for (int i = tid; i < NK * DH; i += nt) {
    const int m = i / DH, d = i % DH;
    const size_t src = (size_t)safe[m] * CA + hh * DH + d;
    const float vk = maskv[m];
    Kt[m * LDD + d] = vk != 0.f ? kk[src] : __float2bfloat16(0.f);
    Vs[m * LDD + d] = vk != 0.f ? vv[src] : __float2bfloat16(0.f);
  }
  __syncthreads();
  for (int t = warp; t < (NQ / 16) * (NK / 16); t += nw) {
    const int mt = t / (NK / 16), nn = t % (NK / 16);
    facc acc;
    wm::fill_fragment(acc, 0.0f);
#pragma unroll
    for (int k = 0; k < DH / 16; ++k) {
      fmA af; fmBc bfr;
      wm::load_matrix_sync(af, Qs + mt * 16 * LDD + k * 16, LDD);
      wm::load_matrix_sync(bfr, Kt + (nn * 16) * LDD + k * 16, LDD);
      wm::mma_sync(acc, af, bfr, acc);
    }
    wm::store_matrix_sync(S + mt * 16 * NK + nn * 16, acc, NK, wm::mem_row_major);
  }
  __syncthreads();
  const bf16* zbp = gp.zb + (((size_t)j * NBB + b) * NH + hh) * (NQ * NK);
  const int lane = tid & 31;
  for (int r = warp; r < NQ; r += nw) {
    float mx = -1e30f;
    for (int i = lane; i < NK; i += 32) {
      const float v = S[r * NK + i] + __bfloat162float(zbp[r * NK + i]);
      S[r * NK + i] = v;
      mx = fmaxf(mx, v);
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
    float sum = 0.f;
    for (int i = lane; i < NK; i += 32) {
      const float e = __expf(S[r * NK + i] - mx);
      S[r * NK + i] = e; sum += e;
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
    const float rs = 1.0f / sum;
    for (int i = lane; i < NK; i += 32) Ps[r * LDK + i] = __float2bfloat16(S[r * NK + i] * rs);
  }
  __syncthreads();
  for (int t = warp; t < (NQ / 16) * (DH / 16); t += nw) {
    const int mt = t / (DH / 16), nn = t % (DH / 16);
    facc acc;
    wm::fill_fragment(acc, 0.0f);
#pragma unroll
    for (int k = 0; k < NK / 16; ++k) {
      fmA af; fmB bfr;
      wm::load_matrix_sync(af, Ps + mt * 16 * LDK + k * 16, LDK);
      wm::load_matrix_sync(bfr, Vs + (k * 16) * LDD + nn * 16, LDD);
      wm::mma_sync(acc, af, bfr, acc);
    }
    wm::store_matrix_sync(S + mt * 16 * DH + nn * 16, acc, DH, wm::mem_row_major);
  }
  __syncthreads();
  bf16* ogb = gp.ogb; const bf16* gg = gp.gg;
  for (int i = tid; i < NQ * DH; i += nt) {
    const size_t dst = (size_t)(b * NQ + i / DH) * CA + hh * DH + i % DH;
    ogb[dst] = __float2bfloat16(S[i] / (1.0f + __expf(-__bfloat162float(gg[dst]))));
  }
}

// --- K5: attention out-projection + residual + conditioned transition + next QKV
__global__ void k_block(int j) {
  extern __shared__ char smem[];
  float* Af = (float*)smem;                        // [16][CA]
  float* Cb = Af + 16 * CA;                        // [16][2*NTR]
  bf16* Ab = (bf16*)(Cb + 16 * 2 * NTR);           // [16][ALD1] x2, or [16][ALD2]
  bf16* Bs = Ab + 2 * 16 * ALD1;                   // up to [CA][2*NTR + PADB]
  const int r0 = blockIdx.x * 16, tid = threadIdx.x, nt = blockDim.x;
  const bf16* cch = gp.condh + (size_t)r0 * 18 * CA;
  const bf16* ccc = gp.condc + (size_t)r0 * 6 * CA;
  float* ag = gp.a + (size_t)r0 * CA;

  for (int i = tid; i < (16 * CA) / 8; i += nt) {
    const int r = (i * 8) >> 7, c = (i * 8) & (CA - 1);
    *(int4*)(Ab + r * ALD1 + c) = *(const int4*)(gp.ogb + (size_t)(r0 + r) * CA + c);
  }
  GEMM(1, CA, CA, ALD1, CA, CA, NWARP, Ab, gp.w_o[j], Cb, Bs);
  for (int i = tid; i < 16 * CA; i += nt) {
    const int r = i / CA, c = i % CA;
    float v = ag[i];
    if (r0 + r < NA) v += Cb[i] * __bfloat162float(ccc[r * 6 * CA + (j * 2) * CA + c]);
    Af[i] = v;
  }
  __syncthreads();
  ln16<16>(Af, Cb);
  __syncthreads();
  for (int i = tid; i < 16 * CA; i += nt) {
    const int r = i / CA, c = i % CA;
    const bf16* g = cch + r * 18 * CA + (j * 6 + 4) * CA;
    Ab[r * ALD1 + c] = __float2bfloat16(__bfloat162float(g[c])
                                        * (Cb[i] + __bfloat162float(g[CA + c])));
  }
  __syncthreads();
  GEMM(1, 2 * NTR, CA, ALD1, 2 * NTR, 2 * NTR, NWARP, Ab, gp.w_ab[j], Cb, Bs);
  for (int i = tid; i < 16 * NTR; i += nt) {
    const int r = i / NTR, c = i % NTR;
    const float x = Cb[r * 2 * NTR + c];
    Ab[r * ALD2 + c] = __float2bfloat16((x / (1.0f + __expf(-x))) * Cb[r * 2 * NTR + NTR + c]);
  }
  __syncthreads();
  GEMM(1, CA, NTR, ALD2, CA, CA, NWARP, Ab, gp.w_out[j], Cb, Bs);
  for (int i = tid; i < 16 * CA; i += nt) {
    const int r = i / CA, c = i % CA;
    const float am = gp.ampad[r0 + r];
    float v = Af[i] + __bfloat162float(ccc[r * 6 * CA + (j * 2 + 1) * CA + c]) * Cb[i] * am;
    if (j == 2) v *= am;
    Af[i] = v;
    ag[i] = v;
  }
  __syncthreads();
  if (j == 2) return;
  const int jn = j + 1;
  ln16<16>(Af, Cb);
  __syncthreads();
  for (int i = tid; i < 16 * CA; i += nt) {
    const int r = i / CA, c = i % CA;
    const bf16* g = cch + r * 18 * CA + jn * 6 * CA;
    Ab[r * ALD1 + c] = __float2bfloat16(__bfloat162float(g[c])
                                        * (Cb[i] + __bfloat162float(g[CA + c])));
    Ab[16 * ALD1 + r * ALD1 + c] = __float2bfloat16(__bfloat162float(g[2 * CA + c])
                                        * (Cb[i] + __bfloat162float(g[3 * CA + c])));
  }
  __syncthreads();
  GEMM(1, 2 * CA, CA, ALD1, 2 * CA, 2 * CA, NWARP, Ab, gp.w_qg[jn], Cb, Bs);
  bf16* qg = gp.qq + (size_t)r0 * CA;
  bf16* gg = gp.gg + (size_t)r0 * CA;
  const float* bq = gp.b_qg[jn];
  for (int i = tid; i < 16 * CA; i += nt) {
    const int r = i / CA, c = i % CA;
    qg[i] = __float2bfloat16(Cb[r * 2 * CA + c] + bq[c]);
    gg[i] = __float2bfloat16(Cb[r * 2 * CA + CA + c]);
  }
  __syncthreads();
  GEMM(1, 2 * CA, CA, ALD1, 2 * CA, 2 * CA, NWARP, Ab + 16 * ALD1, gp.w_kv[jn], Cb, Bs);
  bf16* kg = gp.kk + (size_t)r0 * CA;
  bf16* vg = gp.vv + (size_t)r0 * CA;
  for (int i = tid; i < 16 * CA; i += nt) {
    const int r = i / CA, c = i % CA;
    kg[i] = __float2bfloat16(Cb[r * 2 * CA + c]);
    vg[i] = __float2bfloat16(Cb[r * 2 * CA + CA + c]);
  }
}

// ----------------------------- K0b: first block's AdaLN-ed query/key projections
__global__ void k_qkv0() {
  extern __shared__ char smem[];
  float* Ln = (float*)smem;                        // [16][CA]
  float* Cb = Ln + 16 * CA;                        // [16][2*CA]
  bf16* Ab = (bf16*)(Cb + 16 * 2 * CA);            // [16][ALD1] x2
  bf16* Bs = Ab + 2 * 16 * ALD1;                   // [CA][2*CA + PADB]
  const int r0 = blockIdx.x * 16, tid = threadIdx.x, nt = blockDim.x;
  const bf16* cch = gp.condh + (size_t)r0 * 18 * CA;
  const float* ag = gp.a + (size_t)r0 * CA;
  for (int i = tid; i < 16 * CA; i += nt) Ln[i] = ag[i];
  __syncthreads();
  ln16<16>(Ln, Ln);   // safe in place: each lane rewrites only the element it read
  __syncthreads();
  for (int i = tid; i < 16 * CA; i += nt) {
    const int r = i / CA, c = i % CA;
    const bf16* g = cch + r * 18 * CA;
    Ab[r * ALD1 + c] = __float2bfloat16(__bfloat162float(g[c])
                                        * (Ln[i] + __bfloat162float(g[CA + c])));
    Ab[16 * ALD1 + r * ALD1 + c] = __float2bfloat16(__bfloat162float(g[2 * CA + c])
                                        * (Ln[i] + __bfloat162float(g[3 * CA + c])));
  }
  __syncthreads();
  GEMM(1, 2 * CA, CA, ALD1, 2 * CA, 2 * CA, NWARP, Ab, gp.w_qg[0], Cb, Bs);
  bf16* qg = gp.qq + (size_t)r0 * CA;
  bf16* gg = gp.gg + (size_t)r0 * CA;
  const float* bq = gp.b_qg[0];
  for (int i = tid; i < 16 * CA; i += nt) {
    const int r = i / CA, c = i % CA;
    qg[i] = __float2bfloat16(Cb[r * 2 * CA + c] + bq[c]);
    gg[i] = __float2bfloat16(Cb[r * 2 * CA + CA + c]);
  }
  __syncthreads();
  GEMM(1, 2 * CA, CA, ALD1, 2 * CA, 2 * CA, NWARP, Ab + 16 * ALD1, gp.w_kv[0], Cb, Bs);
  bf16* kg = gp.kk + (size_t)r0 * CA;
  bf16* vg = gp.vv + (size_t)r0 * CA;
  for (int i = tid; i < 16 * CA; i += nt) {
    const int r = i / CA, c = i % CA;
    kg[i] = __float2bfloat16(Cb[r * 2 * CA + c]);
    vg[i] = __float2bfloat16(Cb[r * 2 * CA + CA + c]);
  }
}

// --------------------- K6: atom projection + mean aggregation to tokens (MT x 3)
__global__ void k_agg() {
  extern __shared__ char smem[];
  float* Cs = (float*)smem;                        // [16][CA]
  bf16* Ab = (bf16*)(Cs + 16 * CA);                // [16][ALD1]
  bf16* Bs = Ab + 16 * ALD1;                       // [CA][CA + PADB]
  const int r0 = blockIdx.x * 16, c0 = blockIdx.y * CA, tid = threadIdx.x, nt = blockDim.x;
  const float* ag = gp.a + (size_t)r0 * CA;
  for (int i = tid; i < 16 * CA; i += nt) Ab[(i >> 7) * ALD1 + (i & (CA - 1))] = __float2bfloat16(ag[i]);
  __syncthreads();
  GEMM(1, CA, CA, ALD1, CA, CTOK, NWARP, Ab, gp.w_lq + c0, Cs, Bs);
  for (int i = tid; i < 16 * CA; i += nt) {
    const int r = i / CA, c = i % CA;
    const int t = gp.a2ti[r0 + r];
    if (t < 0) continue;
    const float am = gp.ampad[r0 + r];
    if (c == 0 && blockIdx.y == 0) atomicAdd(&gp.cnt[t], am);
    const float v = fmaxf(Cs[i], 0.f) * am;
    if (v != 0.f) atomicAdd(&gp.ai[(size_t)t * CTOK + c0 + c], v);
  }
}

// --------- K7: build s_input, then the fused tail GEMM -> s | z_i | z_j
__global__ void k_sz() {
  extern __shared__ char smem[];
  float* Cs = (float*)smem;                        // [16][NTSZ]
  bf16* As = (bf16*)(Cs + 16 * NTSZ);              // [16][CSIP + PADA]
  bf16* Bs = As + 16 * (CSIP + PADA);              // [CSIP][NTSZ + PADB]
  const int c0 = blockIdx.x * NTSZ, tid = threadIdx.x, nt = blockDim.x;
  stageB<NTSZ, CSIP, NTAIL>(Bs, gp.w_tail + c0);
  const float* ai = gp.ai; const float* cnt = gp.cnt; const float* tfs = gp.tfs;
  for (int r = 0; r < TT; ++r) {
    const float inv = 1.0f / fmaxf(cnt[r], 1.0f);
    for (int c = tid; c < CSIP; c += nt) {
      float v = 0.f;
      if (c < CTOK) v = ai[(size_t)r * CTOK + c] * inv;
      else if (c < CSI) v = tfs[r * 65 + (c - CTOK)];
      const bf16 hv = __float2bfloat16(v);
      As[(size_t)r * (CSIP + PADA) + c] = hv;
      if (blockIdx.x == 0 && c < CSI) gp.o_sinput[(size_t)r * CSI + c] = hv;
    }
  }
  __syncthreads();
  mmaTile<1, NTSZ, CSIP, CSIP + PADA, NTSZ, NWARP>(As, Bs, Cs);
  __syncthreads();
  for (int i = tid; i < TT * NTSZ; i += nt) {
    const int r = i / NTSZ, c = i % NTSZ;
    const int col = c0 + c;
    if (col < CTOK) gp.o_s[(size_t)r * CTOK + col] = __float2bfloat16(Cs[i]);
    else if (col < CTOK + CZ) gp.zi[(size_t)r * CZ + (col - CTOK)] = Cs[i];
    else gp.zj[(size_t)r * CZ + (col - CTOK - CZ)] = Cs[i];
  }
}

// ---------------------------------------- K8: pair representation z (grid TT x TT)
__device__ __forceinline__ int nbin(float pi, float pj, bool cond, int clip) {
  const float o = __bfloat162float(__float2bfloat16(pi - pj));
  float c = __bfloat162float(__float2bfloat16(o + (float)clip));
  c = fminf(fmaxf(c, 0.f), (float)(2 * clip));
  const float f = cond ? c : (float)(2 * clip + 1);
  return (f <= 0.f) ? 0 : (int)ceilf(f);
}
__global__ void k_z(int maxrelchain) {
  const int i = blockIdx.x, j = blockIdx.y, tid = threadIdx.x, nt = blockDim.x;
  const bool sc = gp.asym[i] == gp.asym[j];
  const bool se = gp.ent[i] == gp.ent[j];
  const float ri = gp.ri[i], rj = gp.ri[j];
  const int n1 = nbin(ri, rj, sc, 32);
  const int n2 = nbin(gp.ti[i], gp.ti[j], sc && (ri == rj), 32);
  const int n3 = nbin(gp.sym[i], gp.sym[j], se, maxrelchain);
  const float tbv = gp.tb[i * TT + j], sef = se ? 1.0f : 0.0f;
  const float* zi = gp.zi + i * CZ; const float* zj = gp.zj + j * CZ;
  const float* p1 = gp.P1 + n1 * CZ; const float* p2 = gp.P2 + n2 * CZ;
  const float* p3 = gp.P3 + n3 * CZ;
  bf16* dst = gp.o_z + ((size_t)i * TT + j) * CZ;
  for (int c = tid; c < CZ; c += nt)
    dst[c] = __float2bfloat16(zi[c] + zj[c] + p1[c] + p2[c] + gp.wr132[c] * sef + p3[c]
                              + tbv * gp.wtb[c]);
}

// ============================================================== host launchers
static bf16* BP(const torch::Tensor& t) { return (bf16*)t.data_ptr(); }
static float* FP(const torch::Tensor& t) { return t.data_ptr<float>(); }
static int g_smem_max = 0;

void setup(std::vector<torch::Tensor> W, std::vector<torch::Tensor> Bf) {
  int i = 0;
  hp.w_cl = BP(W[i++]); hp.w_condh = BP(W[i++]); hp.w_condc = BP(W[i++]);
  hp.w_lq = BP(W[i++]); hp.w_tail = BP(W[i++]);
  hp.w_l = BP(W[i++]); hp.w_m = BP(W[i++]);
  for (int j = 0; j < 3; ++j) hp.w_qg[j] = BP(W[i++]);
  for (int j = 0; j < 3; ++j) hp.w_kv[j] = BP(W[i++]);
  for (int j = 0; j < 3; ++j) hp.w_o[j] = BP(W[i++]);
  for (int j = 0; j < 3; ++j) hp.w_ab[j] = BP(W[i++]);
  for (int j = 0; j < 3; ++j) hp.w_out[j] = BP(W[i++]);
  hp.b_condh = FP(W[i++]); hp.b_condc = FP(W[i++]);
  for (int j = 0; j < 3; ++j) hp.b_qg[j] = FP(W[i++]);
  hp.w_ro = FP(W[i++]); hp.w_iq = FP(W[i++]); hp.w_vm = FP(W[i++]);
  for (int j = 0; j < 3; ++j) hp.w_p[j] = FP(W[i++]);
  hp.w_lnz = FP(W[i++]);
  for (int j = 0; j < 3; ++j) hp.w_z[j] = FP(W[i++]);
  hp.P1 = FP(W[i++]); hp.P2 = FP(W[i++]); hp.P3 = FP(W[i++]);
  hp.wr132 = FP(W[i++]); hp.wtb = FP(W[i++]);

  int k = 0;
  hp.X = BP(Bf[k++]); hp.clb = BP(Bf[k++]); hp.clhb = BP(Bf[k++]); hp.ogb = BP(Bf[k++]);
  hp.cl = FP(Bf[k++]); hp.clh = FP(Bf[k++]); hp.ul = FP(Bf[k++]); hp.vm = FP(Bf[k++]);
  hp.condh = BP(Bf[k++]); hp.condc = BP(Bf[k++]); hp.bmask = FP(Bf[k++]);
  hp.maskv = FP(Bf[k++]); hp.zb = BP(Bf[k++]);
  hp.a = FP(Bf[k++]); hp.qq = BP(Bf[k++]); hp.gg = BP(Bf[k++]);
  hp.kk = BP(Bf[k++]); hp.vv = BP(Bf[k++]); hp.ai = FP(Bf[k++]); hp.cnt = FP(Bf[k++]);
  hp.ampad = FP(Bf[k++]); hp.rppad = FP(Bf[k++]); hp.uidpad = FP(Bf[k++]);
  hp.tfs = FP(Bf[k++]); hp.ri = FP(Bf[k++]); hp.ti = FP(Bf[k++]);
  hp.asym = FP(Bf[k++]); hp.ent = FP(Bf[k++]); hp.sym = FP(Bf[k++]); hp.tb = FP(Bf[k++]);
  hp.zi = FP(Bf[k++]); hp.zj = FP(Bf[k++]);
  hp.safe = Bf[k++].data_ptr<int>(); hp.a2ti = Bf[k++].data_ptr<int>();
  hp.o_sinput = BP(Bf[k++]); hp.o_s = BP(Bf[k++]); hp.o_z = BP(Bf[k++]);

  TORCH_CHECK(cudaMemcpyToSymbol(gp, &hp, sizeof(Params)) == cudaSuccess, "param upload failed");
  cudaDeviceGetAttribute(&g_smem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  const void* fns[] = {(const void*)k_cl, (const void*)k_cond, (const void*)k_attn,
                       (const void*)k_block, (const void*)k_qkv0, (const void*)k_agg,
                       (const void*)k_sz};
  for (const void* f : fns)
    cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, g_smem_max);
}

void ingest(std::vector<torch::Tensor> in) {
  auto s = c10::cuda::getCurrentCUDAStream();
  k_ingest<<<AP + NBB + 1, 128, 0, s>>>(
      BP(in[0]), BP(in[1]), BP(in[2]), BP(in[3]), BP(in[4]), BP(in[5]), BP(in[6]),
      in[7].data_ptr<int64_t>(), BP(in[8]), BP(in[9]), BP(in[10]), BP(in[11]),
      BP(in[12]), BP(in[13]), BP(in[14]));
}

#define SM_CL    ((32 * CA) * 4 + (16 * (KFEAT + PADA) + KFEAT * (CA + PADB)) * 2)
#define SM_COND  ((MR_COND * 16 * CA) * 4 + (MR_COND * 16 * ALD1 + CA * (CA + PADB)) * 2)
#define SM_ATT   ((NQ * (DH + PADA) + 2 * NK * (DH + PADA) + NQ * (NK + PADA)) * 2 + NQ * NK * 4)
#define SM_BLK   ((16 * CA + 16 * 2 * NTR) * 4 + (2 * 16 * ALD1 + CA * (2 * NTR + PADB)) * 2)
#define SM_QKV   ((16 * CA + 16 * 2 * CA) * 4 + (2 * 16 * ALD1 + CA * (2 * CA + PADB)) * 2)
#define SM_AGG   ((16 * CA) * 4 + (16 * ALD1 + CA * (CA + PADB)) * 2)
#define SM_SZ    ((16 * NTSZ) * 4 + (16 * (CSIP + PADA) + CSIP * (NTSZ + PADB)) * 2)

void run(int64_t maxrelchain) {
  auto s = c10::cuda::getCurrentCUDAStream();
  k_cl<<<MT, NTH, SM_CL, s>>>();
  k_cond<<<dim3(MT / MR_COND, 24), NTH, SM_COND, s>>>();
  k_pair<<<dim3(NBB, NK / 8), 256, 0, s>>>();
  k_qkv0<<<MT, NTH, SM_QKV, s>>>();
  for (int j = 0; j < 3; ++j) {
    k_attn<<<dim3(NBB, NH), NTH_ATT, SM_ATT, s>>>(j);
    k_block<<<MT, NTH, SM_BLK, s>>>(j);
  }
  k_agg<<<dim3(MT, CTOK / CA), NTH, SM_AGG, s>>>();
  k_sz<<<NTAIL / NTSZ, NTH, SM_SZ, s>>>();
  k_z<<<dim3(TT, TT), 128, 0, s>>>((int)maxrelchain);
}
'''

_EXT = None
_EXT_FAILED = False


def _load_ext():
    """JIT-build the fused extension once; ``None`` if it cannot be built."""
    global _EXT, _EXT_FAILED
    if _EXT is None and not _EXT_FAILED:
        try:
            arch = os.environ.get("FASTKERNELS_CUDA_ARCH_LIST")
            if not arch:
                cap = torch.cuda.get_device_capability()
                arch = f"{cap[0]}.{cap[1]}" + ("a" if cap[0] in (9, 10, 12) else "")
            os.environ["TORCH_CUDA_ARCH_LIST"] = arch
            from torch.utils.cpp_extension import load_inline
            _EXT = load_inline(
                "af3_input_embedder_fused",
                cpp_sources=_CPP_SRC, cuda_sources=_CUDA_SRC,
                functions=["setup", "ingest", "run"],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                verbose=bool(os.environ.get("FK_VERBOSE")),
            )
        except Exception:
            _EXT_FAILED = True
    return _EXT


def _pack_weights(mod):
    """Flatten the module tree into the GEMM-ready operands the kernels expect.

    Every matmul operand is laid out [K, N] row-major so a wmma tile can read it
    with a constant leading dimension, and the per-AdaLN ``layer_norm_s`` scale is
    folded into the conditioning weights so all 24 conditioning projections share
    a single activation (cl_hat or cl).
    """
    ae = mod.atom_attn_enc
    rf = ae.ref_atom_feature_embedder
    W = lambda m: m.weight.float()
    B = lambda m: m.bias.float()
    dev = mod.linear_s.weight.device
    bf = lambda t: t.to(torch.bfloat16).contiguous()
    f32 = lambda t: t.float().contiguous()

    feat = torch.cat([W(m).t() for m in (rf.linear_ref_pos, rf.linear_ref_charge,
                                        rf.linear_ref_mask, rf.linear_ref_element,
                                        rf.linear_ref_atom_chars)], 0)
    Wcl = torch.zeros(_KFEAT, _CA, device=dev)
    Wcl[:feat.shape[0]] = feat

    hat_cols, hat_bias, cl_cols, cl_bias = [], [], [], []
    for blk in ae.atom_transformer.blocks:
        apb, ct = blk.attention_pair_bias, blk.conditioned_transition
        for adaln in (apb.layer_norm_a_q, apb.layer_norm_a_k, ct.layer_norm):
            w = W(adaln.layer_norm_s)[:, None]
            hat_cols += [W(adaln.linear_g).t() * w, W(adaln.linear_s).t() * w]
            hat_bias += [B(adaln.linear_g), torch.zeros_like(B(adaln.linear_g))]
        cl_cols += [W(apb.linear_ada_out).t(), W(ct.linear_g).t()]
        cl_bias += [B(apb.linear_ada_out), B(ct.linear_g)]

    tail = torch.zeros(_CSIP, _NTAIL, device=dev)
    tail[:_CSI, :_CTOK] = W(mod.linear_s).t()
    tail[:_CSI, _CTOK:_CTOK + _CZ] = W(mod.linear_z_i).t()
    tail[:_CSI, _CTOK + _CZ:] = W(mod.linear_z_j).t()
    out = [bf(Wcl), bf(torch.cat(hat_cols, 1)), bf(torch.cat(cl_cols, 1)),
           bf(W(ae.linear_q[0]).t()), bf(tail),
           bf(W(ae.linear_l).t()), bf(W(ae.linear_m).t())]
    blocks = list(ae.atom_transformer.blocks)
    for pick in (
        lambda b: torch.cat([W(b.attention_pair_bias.mha.linear_q).t(),
                             W(b.attention_pair_bias.mha.linear_g).t()], 1),
        lambda b: torch.cat([W(b.attention_pair_bias.mha.linear_k).t(),
                             W(b.attention_pair_bias.mha.linear_v).t()], 1),
        lambda b: W(b.attention_pair_bias.mha.linear_o).t(),
        lambda b: torch.cat([W(b.conditioned_transition.swiglu.linear_a).t(),
                             W(b.conditioned_transition.swiglu.linear_b).t()], 1),
        lambda b: W(b.conditioned_transition.linear_out).t(),
    ):
        out += [bf(pick(b)) for b in blocks]
    out += [f32(torch.cat(hat_bias, 0)), f32(torch.cat(cl_bias, 0))]
    out += [f32(torch.cat([B(b.attention_pair_bias.mha.linear_q),
                           torch.zeros_like(B(b.attention_pair_bias.mha.linear_q))], 0))
            for b in blocks]
    out += [f32(W(rf.linear_ref_offset).t()),
            f32(W(rf.linear_inv_sq_dists).t().reshape(-1)),
            f32(W(rf.linear_valid_mask).t().reshape(-1))]
    out += [f32(W(ae.pair_mlp[i]).t()) for i in (1, 3, 5)]
    out += [f32(W(ae.atom_transformer.layer_norm_z))]
    out += [f32(W(b.attention_pair_bias.linear_z).t()) for b in blocks]

    Wr = W(mod.linear_relpos)
    nb = 2 * mod.relpos_k + 2
    zc = torch.zeros(_CZ, 1, device=dev)
    pre = lambda cols: f32(torch.cat([zc, cols.cumsum(-1)], -1).t())
    out += [pre(Wr[:, :nb]), pre(Wr[:, nb:2 * nb]), pre(Wr[:, 2 * nb + 1:]),
            f32(Wr[:, 2 * nb]), f32(W(mod.linear_token_bonds)[:, 0])]
    return out


class _Engine:
    """Static buffers + captured graph for one (module, geometry) pair."""

    def __init__(self, mod, ext):
        dev = mod.linear_s.weight.device
        self.weights = _pack_weights(mod)
        h = torch.bfloat16
        z = lambda *s, dt=torch.float32: torch.zeros(*s, device=dev, dtype=dt)
        self.buf = B = [
            z(_AP, _KFEAT, dt=h), z(_AP, _CA, dt=h), z(_AP, _CA, dt=h), z(_AP, _CA, dt=h),
            z(_AP, _CA), z(_AP, _CA), z(_AP, _CP), z(_AP, _CP),
            z(_AP, 18 * _CA, dt=h), z(_AP, 6 * _CA, dt=h),
            z(_NBB, _NQ, _NK), z(_NBB, _NK),
            z(3, _NBB, _NH, _NQ, _NK, dt=h),
            z(_AP, _CA), z(_AP, _CA, dt=h), z(_AP, _CA, dt=h),
            z(_AP, _CA, dt=h), z(_AP, _CA, dt=h),
            z(_TT, _CTOK), z(_TT),
            z(_AP), z(_AP, 3), z(_AP), z(_TT, 65),
            z(_TT), z(_TT), z(_TT), z(_TT), z(_TT), z(_TT, _TT),
            z(_TT, _CZ), z(_TT, _CZ),
            z(_NBB, _NK, dt=torch.int32), z(_AP, dt=torch.int32),
            z(1, _TT, _CSI, dt=h), z(1, _TT, _CTOK, dt=h), z(1, _TT, _TT, _CZ, dt=h),
        ]
        self.out = (B[34], B[35], B[36])
        self.mrc = int(mod.max_relative_chain)
        self.ext = ext
        ext.setup(self.weights, B)
        self.graph = None
        self.captured = False

    def _capture(self):
        self.captured = True
        try:
            cur = torch.cuda.current_stream()
            side = torch.cuda.Stream()
            side.wait_stream(cur)
            with torch.cuda.stream(side):
                for _ in range(3):
                    self.ext.run(self.mrc)
            cur.wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.ext.run(self.mrc)
            self.graph = graph
        except Exception:
            self.graph = None

    def __call__(self, token_features, batch):
        self.ext.ingest([
            batch["ref_pos"], batch["ref_charge"], batch["ref_mask"],
            batch["ref_element"], batch["ref_atom_name_chars"], batch["ref_space_uid"],
            batch["atom_mask"], batch["atom_to_token_index"], token_features,
            batch["residue_index"], batch["token_index"], batch["asym_id"],
            batch["entity_id"], batch["sym_id"], batch["token_bonds"],
        ])
        graph = self.graph
        if graph is None:
            if not self.captured:
                self._capture()
                graph = self.graph
            if graph is None:
                self.ext.run(self.mrc)
                return self.out
        graph.replay()
        return self.out


_REQUIRED = ("ref_pos", "ref_charge", "ref_mask", "ref_element", "ref_atom_name_chars",
             "ref_space_uid", "atom_mask", "atom_to_token_index", "residue_index",
             "token_index", "asym_id", "entity_id", "sym_id", "token_bonds",
             "token_mask")


def _fast_signature(mod, token_features, batch):
    """The geometry/dtype tuple the kernels require, or ``None`` if unsupported."""
    ae = getattr(mod, "atom_attn_enc", None)
    if ae is None or ae.noisy_position_embedder is not None:
        return None
    if (mod.c_s_input, mod.c_s, mod.c_z) != (_CSI, _CTOK, _CZ):
        return None
    if (ae.n_query, ae.n_key) != (_NQ, _NK):
        return None
    tr = ae.atom_transformer
    if len(tr.blocks) != 3:
        return None
    mha = tr.blocks[0].attention_pair_bias.mha
    if (mha.no_heads, mha.c_hidden) != (_NH, _DH):
        return None
    if tr.blocks[0].conditioned_transition.swiglu.linear_a.weight.shape[0] != _NTR:
        return None
    if any(k not in batch for k in _REQUIRED):
        return None
    if token_features.dtype != torch.bfloat16 or not token_features.is_cuda:
        return None
    if tuple(token_features.shape) != (1, _TT, _CTOK):
        return None
    if tuple(batch["atom_mask"].shape) != (1, _NA):
        return None
    if tuple(batch["ref_element"].shape) != (1, _NA, 119):
        return None
    if batch["ref_atom_name_chars"].shape[-2:] != torch.Size((1, 256)):
        return None
    if batch["atom_to_token_index"].dtype != torch.int64:
        return None
    for k in _REQUIRED:
        t = batch[k]
        if not t.is_contiguous() or (t.is_floating_point() and t.dtype != torch.bfloat16):
            return None
    if 2 * mod.relpos_k + 2 != 66 or mod.linear_relpos.weight.shape[-1] != 139:
        return None
    return (token_features.shape, batch["atom_mask"].shape)


# --------------------------------------------------------------------------- #
# Reference path (unchanged)
# --------------------------------------------------------------------------- #


def _binned_one_hot(
    x: torch.Tensor, boundaries: torch.Tensor,
) -> torch.Tensor:
    """One-hot encoding with bin boundaries (matches reference binned_one_hot)."""
    return (x[..., None] > boundaries).to(dtype=x.dtype)


def relpos_complex(
    batch: dict,
    max_relative_idx: int,
    max_relative_chain: int,
) -> torch.Tensor:
    """Build relative position features matching the reference implementation.

    Produces 139 features when max_relative_idx=32, max_relative_chain=2:
      66 (rel_pos) + 66 (rel_token) + 1 (same_entity) + 6 (rel_chain)

    Reference: openfold3/core/utils/relpos.py relpos_complex
    """
    res_idx = batch["residue_index"]
    asym_id = batch["asym_id"]
    entity_id = batch["entity_id"]
    same_chain = asym_id[..., None] == asym_id[..., None, :]
    same_res = res_idx[..., None] == res_idx[..., None, :]
    same_entity = entity_id[..., None] == entity_id[..., None, :]

    def _relpos(
        pos: torch.Tensor, condition: torch.BoolTensor, rel_clip_idx: int,
    ) -> torch.Tensor:
        offset = pos[..., None] - pos[..., None, :]
        clipped_offset = torch.clamp(offset + rel_clip_idx, min=0, max=2 * rel_clip_idx)
        final_offset = torch.where(
            condition,
            clipped_offset,
            (2 * rel_clip_idx + 1) * torch.ones_like(clipped_offset),
        )
        boundaries = torch.arange(
            start=0, end=2 * rel_clip_idx + 2, device=final_offset.device,
        ).to(dtype=final_offset.dtype)
        return _binned_one_hot(final_offset, boundaries)

    rel_pos = _relpos(pos=res_idx, condition=same_chain, rel_clip_idx=max_relative_idx)
    rel_token = _relpos(
        pos=batch["token_index"],
        condition=same_chain & same_res,
        rel_clip_idx=max_relative_idx,
    )
    rel_chain = _relpos(
        pos=batch["sym_id"],
        condition=same_entity,
        rel_clip_idx=max_relative_chain,
    )

    same_entity_feat = same_entity[..., None].to(dtype=rel_pos.dtype)

    return torch.cat([rel_pos, rel_token, same_entity_feat, rel_chain], dim=-1)


class InputEmbedder(nn.Module):
    """Produces initial single and pair representations from token features.

    Matches InputEmbedderAllAtom: runs AtomAttentionEncoder to get a
    token-level representation, concatenates with restype/profile/deletion_mean
    to form s_input (449 dims), then projects to s and z.

    Args:
        c_s_input: Input single representation dimension (449 for all-atom)
        c_s: Single representation dimension
        c_z: Pair representation dimension
        relpos_k: Maximum relative residue position
        max_relative_chain: Maximum relative chain index
        c_atom: Atom single representation dim
        c_atom_pair: Atom pair representation dim
        c_token: Token dim for atom attention encoder output
    """

    def __init__(
        self,
        c_s_input: int,
        c_s: int,
        c_z: int,
        relpos_k: int = 32,
        max_relative_chain: int = 2,
        c_atom: int = 128,
        c_atom_pair: int = 16,
        c_token: int | None = None,
    ):
        super().__init__()
        self.c_s_input = c_s_input
        self.c_s = c_s
        self.c_z = c_z
        self.relpos_k = relpos_k
        self.max_relative_chain = max_relative_chain
        self._one_hot = OneHot()
        self._pad = Pad()

        if c_token is None:
            c_token = c_s

        self.atom_attn_enc = AtomAttentionEncoder(
            c_atom=c_atom,
            c_atom_pair=c_atom_pair,
            c_token=c_token,
            add_noisy_pos=False,
        )

        self.linear_s = Linear(c_s_input, c_s, bias=False)
        self.linear_z_i = Linear(c_s_input, c_z, bias=False)
        self.linear_z_j = Linear(c_s_input, c_z, bias=False)

        num_rel_pos_bins = 2 * relpos_k + 2
        num_rel_token_bins = 2 * relpos_k + 2
        num_rel_chain_bins = 2 * max_relative_chain + 2
        num_same_entity_features = 1
        n_relpos_features = (
            num_rel_pos_bins + num_rel_token_bins
            + num_rel_chain_bins + num_same_entity_features
        )
        self.linear_relpos = Linear(n_relpos_features, c_z, bias=False)

        self.linear_token_bonds = Linear(1, c_z, bias=False)

        self._engine = None
        self._engine_sig = None
        self._engine_off = False

    def _fast(self, token_features, batch):
        """Return the fused engine for this call, or ``None`` to use the reference."""
        if self._engine_off:
            return None
        sig = _fast_signature(self, token_features, batch)
        if sig is None:
            return None
        if self._engine is not None and sig == self._engine_sig:
            return self._engine
        ext = _load_ext()
        if ext is None:
            self._engine_off = True
            return None
        try:
            self._engine = _Engine(self, ext)
        except Exception:
            self._engine, self._engine_off = None, True
            return None
        self._engine_sig = sig
        return self._engine

    def forward(
        self,
        token_features: torch.Tensor,
        residue_index: torch.Tensor,
        batch: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            token_features: [*, N_token, c_s_input] per-token features.
                If batch contains ref_pos (atom features), only restype/profile/deletion_mean
                are expected here and atom_attn_enc produces the remaining features.
                Otherwise, treated as pre-built s_input.
            residue_index:  [*, N_token] residue indices
            batch: Feature dict for relpos and atom attention.

        Returns:
            s_input: [*, N_token, c_s_input] input single representation
            s: [*, N_token, C_s] single representation
            z: [*, N_token, N_token, C_z] pair representation
        """
        if batch is not None and "ref_pos" in batch:
            engine, sig = self._engine, self._engine_sig
            if engine is not None and sig is not None \
                    and token_features.shape == sig[0] \
                    and batch["atom_mask"].shape == sig[1]:
                return engine(token_features, batch)
            engine = self._fast(token_features, batch)
            if engine is not None:
                return engine(token_features, batch)
            a, _, _, _ = self.atom_attn_enc(batch=batch)
            s_input = torch.cat(
                [
                    a,
                    batch.get("restype", token_features[..., :32]),
                    batch.get("profile", token_features[..., 32:64]),
                    batch.get("deletion_mean", token_features[..., -1:]).unsqueeze(-1)
                    if batch.get("deletion_mean") is not None and batch["deletion_mean"].dim() == token_features.dim() - 1
                    else batch.get("deletion_mean", token_features[..., -1:]),
                ],
                dim=-1,
            )
        else:
            s_input = token_features

        s = self.linear_s(s_input)

        z_i = self.linear_z_i(s_input)[..., :, None, :]
        z_j = self.linear_z_j(s_input)[..., None, :, :]
        z = z_i + z_j

        if batch is not None and "asym_id" in batch:
            relpos_feats = relpos_complex(
                batch=batch,
                max_relative_idx=self.relpos_k,
                max_relative_chain=self.max_relative_chain,
            ).to(dtype=z.dtype)
        else:
            d = residue_index[..., :, None] - residue_index[..., None, :]
            d = d.clamp(-self.relpos_k, self.relpos_k) + self.relpos_k
            n_bins = 2 * self.relpos_k + 2
            relpos_feats = self._one_hot(d.long(), n_bins).to(
                dtype=z.dtype,
            )
            n_relpos_in = self.linear_relpos.weight.shape[-1]
            if relpos_feats.shape[-1] < n_relpos_in:
                pad_size = n_relpos_in - relpos_feats.shape[-1]
                relpos_feats = self._pad(relpos_feats, (0, pad_size))

        z = z + self.linear_relpos(relpos_feats)

        if batch is not None and "token_bonds" in batch:
            token_bonds_emb = self.linear_token_bonds(
                batch["token_bonds"].unsqueeze(-1).to(dtype=s.dtype)
            )
            z = z + token_bonds_emb

        return s_input, s, z
