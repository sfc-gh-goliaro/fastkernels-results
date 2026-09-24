"""Outer product mean for AlphaFold3 (L2) -- one fused CUDA kernel.

Implements AF3 Algorithm 9 (openfold3 ``OuterProductMean``) in a single kernel
launch.  The baseline chains eleven ops -- LayerNorm, two projections, two mask
multiplies, the outer-product einsum, a reshape/copy, the output projection, the
normalizer einsum and a divide -- and at the captured shape (``m:[1, 8, 16, 64]``,
``c_hidden=32``, ``c_z=128``) not one of them is big enough to cover its own
launch.  The whole forward is 50 MFLOP, yet it runs 13 kernels and measures
120-160 us; ~58 us of that is GPU time and the rest is launch gaps.  So the
target is not FLOP/s, it is *one launch*: an empty kernel measures 1.5-2.2 us
here (which GPU it lands on matters more than anything in this file), and each
additional kernel in the stream costs ~4 us.  That budget is what shaped
everything below.

Why the kernel is not a transcription of the baseline's dataflow
---------------------------------------------------------------
The baseline materializes the full outer product before projecting it::

    G[b,d,c,e] = sum_s a[s,b,c] * b[s,d,e]          # 256 x 1024 = 262144 values
    out[b,d,z] = sum_{c,e} W[z, c*32+e] * G[b,d,c,e]

That second contraction is 33.5 MMAC and ``G`` is 512 KB -- too big to keep on
chip, so any block-level tiling of it re-reads either ``G`` or all of ``W``.
Pulling the output projection inside the sequence sum removes both problems::

    Y[z,s,d,c] = sum_e b[s,d,e] * W[z,c,e]          # per output channel z
    out[b,d,z] = sum_{s,c} a[s,b,c] * Y[z,s,d,c]

25.2 MMAC instead of 35.6, nothing bigger than 8 KB per channel, and -- the
reason this operator can be one kernel at all -- **the work for a channel ``z``
needs only ``a``, ``b`` and ``W[z]``, so blocks partition over ``c_z`` with no
cross-block dependency and no grid sync.**  Each block owns ``ZT`` channels,
reads its own slice of ``W`` (which is therefore read exactly once across the
grid), and recomputes the LayerNorm and the two projections for itself.  That
recomputation is the one deliberate inefficiency: 256 of a block's 448 mma are
redundant with its 63 peers.  It is still the right trade, because the
alternative is a second kernel at ~4 us, while those 256 mma cost ~1 us on an
otherwise idle SM -- and the equations above are written in the *mirrored* form
(``Y`` from ``b``, contracted against ``a``) purely so that ``W[z]``, indexed
``[c][e]``, is already in the layout an mma B operand wants.

``ZT`` trades the duplicated staging against the per-block mma count: every
block reads all of ``m`` (16 KB) plus both projections, so halving the block
count halves that traffic but adds 96 mma per extra channel.  ZT=2 (64 blocks)
measured best; ZT=1 and ZT=4 are both ~5% slower.

What the kernel spends its time on, measured per phase at ZT=2 (on a "slow"
GPU where the empty-kernel floor is 2.15 us of the 7.4 us total): staging
1.7 us, LayerNorm 0.3, normalizer 0.2, projections 1.0, ``Y`` 0.5, the pair
contraction 0.5, the divide and store 0.8.  Three findings drove the code:

* Every global read is a ``cp.async`` batch (phase 0).  Read synchronously, each
  one costs a *serialized* cold DRAM latency -- the store waits on its own load
  and nothing else issues meanwhile.  The mask alone cost 1.8 us that way.
* Fragments are loaded with explicit ``ldmatrix`` from shared memory rather than
  ``nvcuda::wmma``, which on this toolchain emits generic ``LD.E`` per fragment
  and cost 4 us in the projections alone.  See the mma helpers below.
* All extents are pinned as template parameters for the captured shape.  Left
  runtime, every tile-index decode is a signed integer division (~40
  instructions, no hardware divider) and there are a couple of hundred per
  thread; that alone was a 3x difference.

Anything the kernel does not cover -- non-bf16 input, extra batch dims, extents
that are not 16-multiples, an MSA block too large for shared memory, or a
grad-enabled call -- falls back to the baseline's own torch dataflow below.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear

# ---------------------------------------------------------------------------
# Fused kernel
# ---------------------------------------------------------------------------
_CUDA_SRC = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <torch/extension.h>

namespace {

typedef __nv_bfloat16 bf16;

constexpr int NWARP = FK_NWARP;
constexpr int NTHREAD = NWARP * 32;
constexpr int ZT = FK_ZT;          // pair channels owned by one block
// Lanes cooperating on one MSA row in the LayerNorm pass.  Four gives the whole
// MSA block to one pass of NTHREAD/LPR = 128 rows at the captured shape, so no
// thread walks a second row, and the group reduction is two shuffles instead of
// three -- measured 3.5% of the whole kernel against LPR = 8.
constexpr int LPR = 4;
constexpr int MAXNT = 8;           // n-tiles per accumulator set (c_hidden <= 64)
constexpr int MAXEPL = 16;         // channels a lane owns in the LayerNorm pass

struct Params {
  const bf16 *__restrict__ m;
  const bf16 *__restrict__ mask;   // may be null -> all ones
  const bf16 *__restrict__ gam;    // LayerNorm scale, may be null
  const bf16 *__restrict__ bet;    // LayerNorm offset, may be null
  const bf16 *__restrict__ w1;
  const bf16 *__restrict__ w2;
  const bf16 *__restrict__ wout;
  const bf16 *__restrict__ bout;   // may be null
  bf16 *__restrict__ out;
  int S, R, CM, H, CZ;
  float eps, ln_eps;
};

__device__ __forceinline__ float f32(bf16 v) { return __bfloat162float(v); }
__device__ __host__ __forceinline__ size_t up32(size_t n) { return (n + 31) & ~size_t(31); }

// ---------------------------------------------------------------------------
// mma.sync m16n8k16 (bf16 in, fp32 accumulate), fed by ldmatrix from shared
// memory.
//
// nvcuda::wmma is not used even though the tile shapes suit it: nvcc emits
// wmma.load with no state space, so ptxas cannot prove the address is shared and
// compiles each fragment load into four *generic* LD.E with the mma waiting on
// them -- the projection phase alone measured 4 us that way.  Addressing shared
// memory explicitly instead makes one ldmatrix cover a whole 16x16 tile.
//
// One ldmatrix.sync.m8n8.x4 loads four 8x8 blocks; lane l supplies the address
// of row (l%8) of block (l/8), i.e. base + (l&15)*ld + (l>>4)*8 for a 16x16
// region, and the blocks come back as the four registers of an m16n8k16 A
// fragment (rows 0-7/8-15 x k 0-7/8-15).  Applied to a B operand stored
// column-major ([N][K], so k is contiguous) the same four registers are instead
// the b0/b1 pairs of *two* adjacent n-tiles -- {r0,r2} and {r1,r3} -- so one
// instruction feeds two mma.  Every operand below is laid out to make that work:
// that is why a is kept sequence-major and b residue-major.
//
// Accumulator: lane l holds rows l/4 and l/4+8, columns (l%4)*2 and +1, so four
// results leave as two 8-byte stores.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void ldm_x4(uint32_t *r, const bf16 *base, int ld, int lane) {
  const uint32_t a =
      (uint32_t)__cvta_generic_to_shared(base + (lane & 15) * ld + (lane >> 4) * 8);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

__device__ __forceinline__ void mma(float *c, const uint32_t *a, uint32_t b0,
                                    uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Two accumulator values -> one 4-byte store of a k-adjacent bf16 pair.
__device__ __forceinline__ void sts32(bf16 *p, float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  *(uint32_t *)__builtin_assume_aligned(p, 4) = *(const uint32_t *)&v;
}

// ---------------------------------------------------------------------------
// Shared-memory plan.  Everything the kernel reads after phase 0 lives here.
// Row strides carry 8 bf16 of padding so that fragment loads, which walk down a
// column of 16 rows, spread across banks.
// ---------------------------------------------------------------------------
struct Plan {
  size_t ln, at, bs, y, w1, w2, wz, gam, bet, mask, bias, norm, scratch, total;
  int ldln, lda, ldk, ldw, nslot, ksplit;
};

__device__ __host__ inline Plan make_plan(int S, int R, int CM, int H) {
  const int NR = S * R, MT = R / 16, NP = R / 16;  // one slot per 16x16 pair tile
  Plan p;
  p.ldln = CM + 8;   // [NR][CM] MSA rows, and [H][CM] projection weights
  p.lda = H + 8;     // [NR][H] the second projection, row-major
  p.ldk = S * H + 8; // [R][S*H] sequence-major operands
  p.ldw = H + 8;     // [ZT][H][H] output-projection slice
  int ks = NWARP / (ZT * MT * NP);
  if (ks < 1) ks = 1;
  const int kmax = S * H / 16;
  if (ks > kmax) ks = kmax;
  p.ksplit = ks;
  p.nslot = ZT * MT * NP * ks;
  size_t o = 0;
  p.ln = o;      o = up32(o + (size_t)NR * p.ldln * 2);
  p.at = o;      o = up32(o + (size_t)R * p.ldk * 2);
  p.bs = o;      o = up32(o + (size_t)NR * p.lda * 2);
  p.y = o;       o = up32(o + (size_t)ZT * R * p.ldk * 2);
  p.w1 = o;      o = up32(o + (size_t)H * p.ldln * 2);
  p.w2 = o;      o = up32(o + (size_t)H * p.ldln * 2);
  p.wz = o;      o = up32(o + (size_t)ZT * H * p.ldw * 2);
  p.gam = o;     o = up32(o + (size_t)CM * 2);
  p.bet = o;     o = up32(o + (size_t)CM * 2);
  p.mask = o;    o = up32(o + (size_t)NR * 2);
  p.bias = o;    o = up32(o + (size_t)(ZT < 8 ? 8 : ZT) * 2);
  p.norm = o;    o = up32(o + (size_t)R * R * 4);
  p.scratch = o; o = up32(o + (size_t)p.nslot * 256 * 4);
  p.total = o;
  return p;
}

// Copy a contiguous [rows][cols] block into a shared buffer of row stride
// ``ld``, sixteen bytes at a time.  ``cols`` is a multiple of 8, so no chunk
// straddles a row and both ends stay 16-byte aligned.
//
// cp.async, not load-to-register-then-store: a plain copy loop makes every store
// wait on its own load, so the ~10 chunks a thread stages cost ten cold DRAM
// latencies back to back.
__device__ __forceinline__ void stage(bf16 *dst, const bf16 *src, int n8, int cols,
                                      int ld, int tid) {
  for (int i = tid; i < n8; i += NTHREAD) {
    const int f = i * 8;
    const int r = f / cols, c = f - r * cols;
    __pipeline_memcpy_async(dst + (size_t)r * ld + c, src + f, 16);
  }
}

// Shape template arguments: nonzero pins a dimension at compile time.  The
// captured shape gets its own instantiation, which matters more than it looks
// like it should -- with the extents runtime, every tile-index decode is a signed
// integer division (~40 instructions, no hardware divider) and there are a couple
// of hundred of them per thread.
template <int CMc, int Hc, int Rc, int Sc>
__global__ __launch_bounds__(NTHREAD) void opm_kernel(const Params p) {
  extern __shared__ __align__(32) char smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int gid = lane >> 2, tig = lane & 3;
  const int S = Sc ? Sc : p.S, R = Rc ? Rc : p.R;
  const int CM = CMc ? CMc : p.CM, H = Hc ? Hc : p.H;
  const int CZ = p.CZ, NR = S * R;
  const int bi = blockIdx.y;
  const int z0 = blockIdx.x * ZT;
  const int zn = min(ZT, CZ - z0);

  const Plan pl = make_plan(S, R, CM, H);
  // __restrict__: the regions are disjoint by construction, and saying so stops
  // the compiler from ordering each tile's fragment loads after the previous
  // tile's shared-memory stores.
  bf16 *__restrict__ ln = (bf16 *)(smem + pl.ln);
  bf16 *__restrict__ at = (bf16 *)(smem + pl.at);
  bf16 *__restrict__ bs = (bf16 *)(smem + pl.bs);
  bf16 *__restrict__ ys = (bf16 *)(smem + pl.y);
  bf16 *__restrict__ w1s = (bf16 *)(smem + pl.w1);
  bf16 *__restrict__ w2s = (bf16 *)(smem + pl.w2);
  bf16 *__restrict__ wzs = (bf16 *)(smem + pl.wz);
  bf16 *__restrict__ gam = (bf16 *)(smem + pl.gam);
  bf16 *__restrict__ bet = (bf16 *)(smem + pl.bet);
  bf16 *__restrict__ msk = (bf16 *)(smem + pl.mask);
  bf16 *__restrict__ bias = (bf16 *)(smem + pl.bias);
  float *__restrict__ norm = (float *)(smem + pl.norm);
  float *__restrict__ scratch = (float *)(smem + pl.scratch);
  const int ldln = pl.ldln, lda = pl.lda, ldk = pl.ldk, ldw = pl.ldw;
  const int ksplit = pl.ksplit;

  // --- Phase 0: start every global read, as one batch of async copies -------
  // Every byte the kernel reads goes through cp.async here, so all of it is in
  // flight at once and one DRAM latency covers the lot.  Reading any of it
  // synchronously instead costs a *serialized* latency: the store waits on the
  // load, and nothing else issues meanwhile.  Measured, one read at a time:
  // the MSA block 0.61 us, the LayerNorm scale/offset 0.51 us, the projection
  // weights 0.22 us, the output-projection slice 0.35 us -- additive, because
  // none of them overlapped the others.
  // The output-projection slice is a second group: nothing before phase C1
  // needs it, so its latency hides behind the LayerNorm and the projections.
  stage(ln, p.m + (size_t)bi * NR * CM, NR * CM / 8, CM, ldln, tid);
  stage(w1s, p.w1, H * CM / 8, CM, ldln, tid);
  stage(w2s, p.w2, H * CM / 8, CM, ldln, tid);
  if (p.gam)
    stage(gam, p.gam, CM / 8, CM, CM, tid);
  else
    for (int i = tid; i < CM; i += NTHREAD) gam[i] = __float2bfloat16(1.0f);
  if (p.bet)
    stage(bet, p.bet, CM / 8, CM, CM, tid);
  else
    for (int i = tid; i < CM; i += NTHREAD) bet[i] = __float2bfloat16(0.0f);
  if (p.mask)
    stage(msk, p.mask + (size_t)bi * NR, NR / 8, NR, NR, tid);
  else
    for (int i = tid; i < NR; i += NTHREAD) msk[i] = __float2bfloat16(1.0f);
  // zn == ZT: a ragged last tile would make the wide copy read past the end of
  // the bias tensor.
  if (p.bout && zn == ZT && (ZT * 2 == 4 || ZT * 2 == 8 || ZT * 2 == 16)) {
    if (tid == 0) __pipeline_memcpy_async(bias, p.bout + z0, ZT * 2);
  } else {
    for (int i = tid; i < zn; i += NTHREAD)
      bias[i] = p.bout ? p.bout[z0 + i] : __float2bfloat16(0.0f);
  }
  __pipeline_commit();
  stage(wzs, p.wout + (size_t)z0 * H * H, zn * H * H / 8, H, ldw, tid);
  __pipeline_commit();
  __pipeline_wait_prior(1);
  __syncthreads();

  // --- Phase A: LayerNorm of the staged MSA block, in place ----------------
  // LPR lanes per row -- four, so that one pass of NTHREAD/LPR = 128 rows covers
  // the whole MSA block at the captured shape and the group reduction is two
  // shuffles.  Each lane rewrites only its own slice, so nothing inside needs
  // synchronizing.  fp32 reduction, bf16 result: exactly what the baseline's
  // promote_fp32 path hands to cuBLAS, and the same rounding the mma sees.
  const int epl = CM / LPR, sub = lane & (LPR - 1);
  const float inv_cm = 1.0f / (float)CM;
  // A lane always owns the same channels, so its scale/offset slice is read once
  // into registers rather than per row.
  float gv[MAXEPL], bv[MAXEPL];
#pragma unroll
  for (int c = 0; c < MAXEPL; c += 8) {
    if (c >= epl) continue;
    const float4 g4 = *(const float4 *)(gam + sub * epl + c);
    const float4 b4 = *(const float4 *)(bet + sub * epl + c);
    const bf16 *gq = (const bf16 *)&g4;
    const bf16 *bq = (const bf16 *)&b4;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      gv[c + k] = f32(gq[k]);
      bv[c + k] = f32(bq[k]);
    }
  }
  for (int row = tid / LPR; row < NR; row += NTHREAD / LPR) {
    bf16 *q = ln + (size_t)row * ldln + sub * epl;
    float xv[MAXEPL];
    float s = 0.f, sq = 0.f;
#pragma unroll
    for (int c = 0; c < MAXEPL; c += 8) {
      if (c >= epl) continue;
      const float4 r4 = *(const float4 *)(q + c);
      const bf16 *v = (const bf16 *)&r4;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const float x = f32(v[k]);
        xv[c + k] = x;
        s += x;
        sq = fmaf(x, x, sq);
      }
    }
#pragma unroll
    for (int off = LPR >> 1; off; off >>= 1) {
      s += __shfl_xor_sync(0xffffffffu, s, off);
      sq += __shfl_xor_sync(0xffffffffu, sq, off);
    }
    const float mean = s * inv_cm;
    const float rstd = rsqrtf(fmaf(-mean, mean, sq * inv_cm) + p.ln_eps);
#pragma unroll
    for (int c = 0; c < MAXEPL; c += 8) {
      if (c >= epl) continue;
      bf16 o[8];
#pragma unroll
      for (int k = 0; k < 8; ++k)
        o[k] = __float2bfloat16(fmaf((xv[c + k] - mean) * rstd, gv[c + k], bv[c + k]));
      *(float4 *)(q + c) = *(const float4 *)o;
    }
  }

  __syncthreads();            // the normalized MSA block is visible to all warps

  // --- The pair normalizer: sum_s mask[s,b] * mask[s,d] + eps --------------
  for (int i = tid; i < R * R; i += NTHREAD) {
    const int b = i / R, d = i - b * R;
    float acc = 0.f;
    for (int s = 0; s < S; ++s) acc = fmaf(f32(msk[s * R + b]), f32(msk[s * R + d]), acc);
    norm[i] = acc + p.eps;
  }

  // --- Phase B: the two projections, one warp per (matrix, row-tile) -------
  // a lands sequence-major as at[r][(s,h)] (the A operand of phase C2) and b
  // lands row-major as bs[(s,r)][e] (the A operand of phase C1).
  const int SRT = NR / 16, HN = H / 8, KT = CM / 16;
  for (int j = warp; j < 2 * SRT; j += NWARP) {
    const int which = j / SRT, mt = j - which * SRT;
    const bf16 *__restrict__ W = which ? w2s : w1s;
    const bf16 *__restrict__ A = ln + (size_t)mt * 16 * ldln;
    float acc[MAXNT][4];
#pragma unroll
    for (int n = 0; n < MAXNT; ++n)
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[n][k] = 0.f;
#pragma unroll 4
    for (int kt = 0; kt < KT; ++kt) {
      uint32_t af[4];
      ldm_x4(af, A + kt * 16, ldln, lane);
      for (int n = 0; n < HN; n += 2) {
        uint32_t bf[4];
        ldm_x4(bf, W + (size_t)n * 8 * ldln + kt * 16, ldln, lane);
        mma(acc[n], af, bf[0], bf[2]);
        mma(acc[n + 1], af, bf[1], bf[3]);
      }
    }
    // A 16-row tile lies inside one sequence (R is a multiple of 16), so the
    // sequence index is per-tile rather than per-row.
    const int rowb = mt * 16, s0 = rowb / R, r0 = rowb - s0 * R;
    const float m0 = f32(msk[rowb + gid]), m1 = f32(msk[rowb + gid + 8]);
    for (int n = 0; n < HN; ++n) {
      const int col = n * 8 + tig * 2;
      if (which == 0) {
        sts32(at + (size_t)(r0 + gid) * ldk + s0 * H + col, acc[n][0] * m0,
              acc[n][1] * m0);
        sts32(at + (size_t)(r0 + gid + 8) * ldk + s0 * H + col, acc[n][2] * m1,
              acc[n][3] * m1);
      } else {
        sts32(bs + (size_t)(rowb + gid) * lda + col, acc[n][0] * m0, acc[n][1] * m0);
        sts32(bs + (size_t)(rowb + gid + 8) * lda + col, acc[n][2] * m1, acc[n][3] * m1);
      }
    }
  }
  __pipeline_wait_prior(0);   // the output-projection slice
  __syncthreads();

  // --- Phase C1: Y[z][(s,d)][c] = sum_e b[(s,d)][e] * W[z][c][e] ----------
  // Pulling the output projection inside the sequence sum is what makes the
  // whole operator fit in one kernel: the work for a channel z needs only a, b
  // and W[z], so blocks partition over c_z with no cross-block dependency, and
  // it is 25.2 MMAC instead of the 35.6 MMAC the materialized outer product
  // costs.  W[z] is read as the B operand with (k, n) = (e, c), which is its
  // natural [c][e] row-major layout.
  for (int j = warp; j < ZT * SRT; j += NWARP) {
    const int zl = j / SRT, mt = j - zl * SRT;
    if (zl >= zn) continue;
    const bf16 *__restrict__ A = bs + (size_t)mt * 16 * lda;
    const bf16 *__restrict__ W = wzs + (size_t)zl * H * ldw;
    float acc[MAXNT][4];
#pragma unroll
    for (int n = 0; n < MAXNT; ++n)
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[n][k] = 0.f;
#pragma unroll 2
    for (int kt = 0; kt < H / 16; ++kt) {
      uint32_t af[4];
      ldm_x4(af, A + kt * 16, lda, lane);
      for (int n = 0; n < HN; n += 2) {
        uint32_t bf[4];
        ldm_x4(bf, W + (size_t)n * 8 * ldw + kt * 16, ldw, lane);
        mma(acc[n], af, bf[0], bf[2]);
        mma(acc[n + 1], af, bf[1], bf[3]);
      }
    }
    const int rowb = mt * 16, s0 = rowb / R, r0 = rowb - s0 * R;
    bf16 *__restrict__ dst = ys + (size_t)zl * R * ldk + s0 * H;
    for (int n = 0; n < HN; ++n) {
      const int col = n * 8 + tig * 2;
      sts32(dst + (size_t)(r0 + gid) * ldk + col, acc[n][0], acc[n][1]);
      sts32(dst + (size_t)(r0 + gid + 8) * ldk + col, acc[n][2], acc[n][3]);
    }
  }
  __syncthreads();

  // --- Phase C2: out[b][d] = sum_{(s,c)} at[b][(s,c)] * Y[z][d][(s,c)] -----
  // One output tile per (channel, 16x8 residue block); with only two of those at
  // the captured shape the K = N_seq*c_hidden reduction is split ksplit ways
  // across warps and phase D sums the partials.
  const int MT = R / 16, NP = R / 16, KTOT = (S * H) / 16;
  for (int j = warp; j < ZT * MT * NP * ksplit; j += NWARP) {
    int r = j;
    const int ks = r % ksplit; r /= ksplit;
    const int np = r % NP; r /= NP;
    const int mt = r % MT;
    const int zl = r / MT;
    float acc0[4] = {0.f, 0.f, 0.f, 0.f};
    float acc1[4] = {0.f, 0.f, 0.f, 0.f};
    if (zl < zn) {
      const bf16 *__restrict__ A = at + (size_t)mt * 16 * ldk;
      const bf16 *__restrict__ Bm = ys + (size_t)zl * R * ldk + (size_t)np * 16 * ldk;
#pragma unroll 2
      for (int kt = ks; kt < KTOT; kt += ksplit) {
        uint32_t af[4], bf[4];
        ldm_x4(af, A + kt * 16, ldk, lane);
        ldm_x4(bf, Bm + kt * 16, ldk, lane);
        mma(acc0, af, bf[0], bf[2]);
        mma(acc1, af, bf[1], bf[3]);
      }
    }
    float *__restrict__ sc = scratch + (size_t)j * 256;
    *(float2 *)(sc + gid * 16 + tig * 2) = make_float2(acc0[0], acc0[1]);
    *(float2 *)(sc + (gid + 8) * 16 + tig * 2) = make_float2(acc0[2], acc0[3]);
    *(float2 *)(sc + gid * 16 + 8 + tig * 2) = make_float2(acc1[0], acc1[1]);
    *(float2 *)(sc + (gid + 8) * 16 + 8 + tig * 2) = make_float2(acc1[2], acc1[3]);
  }
  __syncthreads();

  // --- Phase D: sum the partials, divide by the normalizer, store ----------
  // One thread per residue pair writes all ZT of its channels as one
  // transaction: z is the fastest axis of the output while it is the axis the
  // grid partitions, so ZT contiguous channels are the most that can be merged.
  // Even at ZT=2 (a 4-byte store) the strided pattern costs only 0.25 us.
  const bool vecst = (zn == ZT) && (CZ % ZT == 0);
  for (int i = tid; i < R * R; i += NTHREAD) {
    const int b = i / R, d = i - b * R;
    const int slot0 = ((b >> 4) * NP + (d >> 4)) * ksplit;
    const int off = (b & 15) * 16 + (d & 15);
    const float inv = 1.0f / norm[i];
    bf16 *o = p.out + ((size_t)bi * R * R + i) * CZ + z0;
    bf16 v[ZT];
    for (int zl = 0; zl < zn; ++zl) {
      const float *__restrict__ sc =
          scratch + (size_t)(zl * MT * NP * ksplit + slot0) * 256 + off;
      float acc = 0.f;
      for (int ks = 0; ks < ksplit; ++ks) acc += sc[(size_t)ks * 256];
      v[zl] = __float2bfloat16((acc + f32(bias[zl])) * inv);
    }
    // A wide store needs the *row* stride to be a multiple of the vector width
    // too: c_z not divisible by ZT leaves odd element offsets (and a misaligned
    // store) even though z0 itself is aligned.
    if (vecst && (ZT % 8) == 0) {
#pragma unroll
      for (int c = 0; c < ZT; c += 8) *(float4 *)(o + c) = *(const float4 *)(v + c);
    } else if (vecst && (ZT % 4) == 0) {
#pragma unroll
      for (int c = 0; c < ZT; c += 4) *(float2 *)(o + c) = *(const float2 *)(v + c);
    } else if (vecst && (ZT % 2) == 0) {
#pragma unroll
      for (int c = 0; c < ZT; c += 2) *(float *)(o + c) = *(const float *)(v + c);
    } else {
      for (int zl = 0; zl < zn; ++zl) o[zl] = v[zl];
    }
  }
}

// The opt-in shared-memory limit is raised once per instantiation, and again if
// a later shape needs more than the first one did -- a stale smaller limit makes
// the launch fail with cudaErrorInvalidValue.
template <int CMc, int Hc, int Rc, int Sc>
void launch(const Params &p, dim3 grid, size_t smem, cudaStream_t stream) {
  static size_t granted = 0;
  if (smem > granted) {
    TORCH_CHECK(cudaFuncSetAttribute(opm_kernel<CMc, Hc, Rc, Sc>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)smem) == cudaSuccess,
                "opm: shared memory request rejected");
    granted = smem;
  }
  opm_kernel<CMc, Hc, Rc, Sc><<<grid, NTHREAD, smem, stream>>>(p);
}

// Device shared-memory-per-block opt-in limit, queried once.
inline int smem_limit() {
  static int lim = [] {
    int v = 0;
    if (cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0)
        != cudaSuccess)
      v = 48 * 1024;
    return v;
  }();
  return lim;
}

}  // namespace

at::Tensor outer_product_mean(const at::Tensor &m, const c10::optional<at::Tensor> &mask_,
                              const c10::optional<at::Tensor> &gam_,
                              const c10::optional<at::Tensor> &bet_, const at::Tensor &w1,
                              const at::Tensor &w2, const at::Tensor &wout,
                              const c10::optional<at::Tensor> &bout_, double eps,
                              double ln_eps) {
  // Everything here is on the measured critical path: at this size the GPU is
  // idle waiting for the launch, so host-side work counts one for one.  Hence
  // checks instead of .contiguous() calls, and no temporaries.
  TORCH_CHECK(m.is_cuda() && m.dim() == 4 && m.scalar_type() == at::kBFloat16
                  && m.is_contiguous(),
              "opm: expected a contiguous 4-D bf16 CUDA tensor");
  const int B = (int)m.size(0), S = (int)m.size(1), R = (int)m.size(2), CM = (int)m.size(3);
  const int H = (int)w1.size(0), CZ = (int)wout.size(0);
  const int NR = S * R;
  TORCH_CHECK(R % 16 == 0 && H % 16 == 0 && H <= 8 * MAXNT && CM % (8 * LPR) == 0
                  && CM <= MAXEPL * LPR,
              "opm: unsupported shape");
  TORCH_CHECK(w1.size(1) == CM && w2.size(1) == CM && w2.size(0) == H
                  && wout.size(1) == (int64_t)H * H,
              "opm: weight shape mismatch");

  Params p;
  p.m = (const bf16 *)m.const_data_ptr();
  p.w1 = (const bf16 *)w1.const_data_ptr();
  p.w2 = (const bf16 *)w2.const_data_ptr();
  p.wout = (const bf16 *)wout.const_data_ptr();
  p.mask = nullptr;
  p.gam = nullptr;
  p.bet = nullptr;
  p.bout = nullptr;
  if (mask_.has_value() && mask_->defined()) {
    TORCH_CHECK(mask_->is_contiguous() && mask_->numel() == (int64_t)B * NR
                    && mask_->scalar_type() == at::kBFloat16,
                "opm: bad mask");
    p.mask = (const bf16 *)mask_->const_data_ptr();
  }
  if (gam_.has_value() && gam_->defined()) {
    TORCH_CHECK(gam_->is_contiguous() && gam_->numel() == CM
                    && gam_->scalar_type() == at::kBFloat16,
                "opm: bad layer-norm scale");
    p.gam = (const bf16 *)gam_->const_data_ptr();
  }
  if (bet_.has_value() && bet_->defined()) {
    TORCH_CHECK(bet_->is_contiguous() && bet_->numel() == CM
                    && bet_->scalar_type() == at::kBFloat16,
                "opm: bad layer-norm offset");
    p.bet = (const bf16 *)bet_->const_data_ptr();
  }
  if (bout_.has_value() && bout_->defined()) p.bout = (const bf16 *)bout_->const_data_ptr();
  p.S = S; p.R = R; p.CM = CM; p.H = H; p.CZ = CZ;
  p.eps = (float)eps;
  p.ln_eps = (float)ln_eps;

  const c10::cuda::CUDAGuard guard(m.device());
  at::Tensor out = at::empty({B, R, R, CZ}, m.options());
  p.out = (bf16 *)out.data_ptr();

  const size_t smem = make_plan(S, R, CM, H).total;
  TORCH_CHECK(smem <= (size_t)smem_limit(), "opm: MSA block too large for shared memory");
  const dim3 grid((CZ + ZT - 1) / ZT, B);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // The captured shape gets a fully static instantiation; anything else runs the
  // same code with runtime extents.
  if (CM == 64 && H == 32 && R == 16 && S == 8)
    launch<64, 32, 16, 8>(p, grid, smem, stream);
  else
    launch<0, 0, 0, 0>(p, grid, smem, stream);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  mod.def("outer_product_mean", &outer_product_mean, "fused AF3 outer product mean");
}
"""

_ZT = int(os.environ.get("FK_OPM_ZT", "2"))
_NWARP = int(os.environ.get("FK_OPM_NWARP", "16"))
_OPM = None
_LOADED = False


def _pin_arch() -> None:
    """Build for the local arch only (the ambient list has six)."""
    override = os.environ.get("FASTKERNELS_CUDA_ARCH_LIST")
    if override is not None:
        if override.strip():
            os.environ["TORCH_CUDA_ARCH_LIST"] = override
        return
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:  # noqa: BLE001 - no device: leave the ambient list alone
        return
    suffix = "a" if major in (9, 10, 12) else ""
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}{suffix}"


def _load() -> None:
    global _OPM, _LOADED
    _LOADED = True
    try:
        from torch.utils.cpp_extension import load_inline

        _pin_arch()
        ext = load_inline(
            name=f"fk_l2_af3_opm_z{_ZT}w{_NWARP}",
            cpp_sources="",
            cuda_sources=_CUDA_SRC,
            extra_cuda_cflags=[
                "-O3",
                f"-DFK_ZT={_ZT}",
                f"-DFK_NWARP={_NWARP}",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            ],
            verbose=False,
        )
        _OPM = ext.outer_product_mean
    except Exception:  # noqa: BLE001 - no nvcc / no GPU: run the torch path
        _OPM = None


class OuterProductMean(nn.Module):
    """AF3 Algorithm 9: Outer product mean.

    Args:
        c_m: MSA embedding channel dimension
        c_z: Pair embedding channel dimension
        c_hidden: Hidden channel dimension
        eps: Epsilon for numerical stability
    """

    def __init__(self, c_m: int, c_z: int, c_hidden: int, eps: float = 1e-3):
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps

        self.layer_norm = LayerNorm(c_m)
        self.linear_1 = Linear(c_m, c_hidden, bias=False)
        self.linear_2 = Linear(c_m, c_hidden, bias=False)
        self.linear_out = Linear(c_hidden ** 2, c_z, bias=True)

        if not _LOADED:
            _load()
        # Parameter handles, resolved once: nn.Module attribute lookup walks
        # _parameters on every access, and eight of those cost more than the
        # kernel's own arguments.  load_state_dict / .to() mutate these objects
        # in place, so the handles stay valid.
        self._fast_args: tuple | None = None

    def _resolve(self) -> tuple:
        args = (
            self.layer_norm.weight,
            self.layer_norm.bias,
            self.linear_1.weight,
            self.linear_2.weight,
            self.linear_out.weight,
            self.linear_out.bias,
            float(self.eps),
            float(self.layer_norm.eps),
        )
        self._fast_args = args
        return args

    def forward(
        self,
        m: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            m:    [*, N_seq, N_res, C_m] MSA embedding
            mask: [*, N_seq, N_res] MSA mask

        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        # torch.is_grad_enabled(): the kernel returns a leaf tensor, so a
        # training-mode caller has to go down the autograd-capable torch path.
        # The bench times (and checks) under no_grad.
        if (_OPM is not None and m.dim() == 4 and m.dtype is torch.bfloat16
                and m.is_cuda and not torch.is_grad_enabled()):
            args = self._fast_args or self._resolve()
            try:
                return _OPM(m, mask, args[0], args[1], args[2], args[3], args[4],
                            args[5], args[6], args[7])
            except RuntimeError:
                pass  # shape the kernel does not cover: torch path below
        return self._torch_forward(m, mask)

    def _torch_forward(
        self, m: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        ln = self.layer_norm(m)

        mask = mask.unsqueeze(-1)
        a = self.linear_1(ln) * mask
        b = self.linear_2(ln) * mask

        del ln

        # [*, N_res, N_seq, C]
        a = a.transpose(-2, -3)
        b = b.transpose(-2, -3)

        # [*, N_res, N_res, C, C]
        outer = torch.einsum("...bac,...dae->...bdce", a, b)

        # [*, N_res, N_res, C * C]
        outer = outer.reshape(outer.shape[:-2] + (-1,))

        # [*, N_res, N_res, C_z]
        outer = self.linear_out(outer)

        # Normalization: count valid sequence pairs per residue pair
        norm = torch.einsum("...abc,...adc->...bdc", mask, mask)
        norm = norm + self.eps

        outer = outer / norm

        return outer
