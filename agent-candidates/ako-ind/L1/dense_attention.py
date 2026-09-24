"""Dense (non-paged) multi-head attention.

Unlike the paged attention ops (FlashAttnPrefill/Decode) which use KV cache
and varlen APIs, this op handles full dense attention with a standard
(batch, seq_len, num_heads, head_dim) layout. Supports both causal and
non-causal modes.

Backend selection is controlled via the ``backend`` parameter:

  ``"auto"`` (default) — picks the fastest available backend:
    Ampere / Hopper (cc 8.x–9.x):
      FA3 via ``fa3_fwd_interface`` > ``flash_attn_interface`` > FA2 via
      ``flash_attn`` > PyTorch SDPA.
    Blackwell+ (cc >= 10.0) or pre-Ampere:
      PyTorch SDPA (dispatches to cuDNN flash attention on supported GPUs).

  ``"sdpa"`` — always use ``F.scaled_dot_product_attention``.  Fully
    ``torch.compile``-friendly and produces numerically identical results
    to diffusers' ``AttnProcessor2_0``.

  ``"flash_attn"`` — always use the flash-attention fallback chain
    (FA3 > FA2); raises if none is installed.

  ``"cudnn"`` — pin the cuDNN flash attention backend via
    ``torch.nn.attention.sdpa_kernel``. Required to actually get cuDNN
    selection through ``torch.compile``: without the context, Inductor
    bakes ``mem_efficient`` (cutlass FMHA, ``sm80`` fallback on
    Blackwell) into the compiled graph and runtime
    ``enable_*_sdp`` toggles don't override it. cuDNN's
    ``sdpa_sm100_flash_*`` kernels are ~2.7× faster than cutlass FMHA
    at typical (1024×9216 with mask) attention shapes on B200.
    ``MATH`` is included as a last-resort fallback for masks cuDNN
    can't handle.

Short-sequence fast path
------------------------
Independently of ``backend``, an unmasked call with ``seq_len <= 256`` and a
power-of-two ``head_dim <= 128`` is served by a fused forward kernel instead of
SDPA.  This is not a micro-optimization of the dispatch path -- for a *batch* of
very short sequences PyTorch's SDPA picks a cuDNN kernel that is catastrophically
mis-tiled.  Measured on B200 (fp16, H=16, D=64):

  (B=144, S=5)  cudnn_generated_fort_native_sdpa_sm80_flash_fprop_wmma_f16
                _knob_6_128x...   ->  68.8 us of GPU time for 0.15 GFLOP
  (B=144, S=2)  same sm80 WMMA kernel                    ->  69.1 us

i.e. cuDNN falls off its sm100 flash path onto an Ampere WMMA kernel with
128x128 tiles and then runs 2304 independent 2x2 attention problems through
it.  The fused kernels do the whole thing -- QK^T, softmax and PV -- in one
launch, reading q/k/v through their captured strides so no permute/contiguous
copies are needed either.  Longer sequences (and anything the fast path cannot
express) fall through to the untouched SDPA/cuDNN/flash chain, which still
owns the large shapes where cuDNN's sm100 flash kernel runs at ~1.5 PFLOP/s.

The fast path has three tiers, tried in order:

1. ``attn_grouped`` / ``attn_tiled`` -- bespoke CUDA C, compiled once per shape
   with nvrtc and launched through ``cuLaunchKernel``.  ``attn_grouped`` serves
   ``seq_len <= 8`` with plain FFMA and no shared memory; ``attn_tiled`` serves
   longer sequences with ``mma.m16n8k16`` tensor-core instructions over a
   ``cp.async``-staged K/V tile.  Every failure mode here -- no cuda-python, no
   nvrtc, unsupported arch, a compile error, a layout that cannot be vectorized
   -- is caught and memoized, so this tier can only ever be faster or absent.
2. ``_short_attn_fwd`` / ``_grouped_attn_fwd`` -- the equivalent Triton kernels.
   These serve anything tier 1 declines.
3. The untouched cuDNN / SDPA / flash / flex chain, which owns everything else.

Used by diffusion models (FLUX, SDXL) and any architecture that needs
stateless multi-head attention without KV cache, including encoder-style
bidirectional attention.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# cuDNN's SDPA kernels are limited to head_dim <= 128 ("head_dim should be no
# more than 128" in sdp_utils.cpp); larger heads must use EFFICIENT/MATH.
_CUDNN_MAX_HEAD_DIM = 128


def _resolve_flash_attn_func():
    """Return the flash-attention callable for Ampere/Hopper.

    Same order as vllm-omni's CUDA FA resolver: FA3 (fa3-fwd) >
    FA3 (source-built flash_attn_interface) > FA2.
    """
    for mod in ("fa3_fwd_interface", "flash_attn_interface"):
        try:
            return __import__(mod, fromlist=["flash_attn_func"]).flash_attn_func
        except (ImportError, ModuleNotFoundError):
            pass
    from flash_attn import flash_attn_func
    return flash_attn_func


# ---------------------------------------------------------------------------
# Fused short-sequence attention (single K/V block, no online rescaling).
# ---------------------------------------------------------------------------
# The whole key/value sequence fits in one BLOCK_N tile, so the usual
# flash-attention loop collapses to: load Q tile, load all K, one QK^T, one
# softmax, load all V, one PV.  There is no running (m, l) to rescale, which
# removes the loop-carried dependency entirely -- what is left is two `tl.dot`s
# and a row reduction, and every program is independent.
#
# The grid is (ceil(S / BLOCK_M), B * H): one program per (query tile, batch,
# head).  For S in the low hundreds a single BLOCK_N tile would have to be
# padded to the next power of two (S=144 -> 256, 44% waste in both the loads and
# the MMA), so BLOCK_N is allowed to be smaller than the padded length, turning
# the kernel into a short online-softmax loop.  Sequences of <= 8 tokens go to
# `_grouped_attn_fwd` instead, which packs several (batch, head) problems into
# one tile.
#
# Strides for the batch / seq / head axes are runtime arguments, so the kernel
# consumes the captured non-contiguous views (a BHSD-contiguous tensor viewed as
# BSHD, or a slice of a packed QKV buffer) directly.  Only `stride(-1) == 1` is
# required, which the caller checks.
_LOG2_E = 1.4426950408889634


@triton.jit
def _short_attn_fwd(
    Q, K, V, Out,
    sqb, sqs, sqh,
    skb, sks, skh,
    svb, svs, svh,
    sob, sos, soh,
    H, S, qk_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, N_ITERS: tl.constexpr,
    D: tl.constexpr, CAUSAL: tl.constexpr,
    MASK_M: tl.constexpr, MASK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh - b * H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    q_ptr = Q + b * sqb + h * sqh + offs_m[:, None] * sqs + offs_d[None, :]
    if MASK_M:
        q = tl.load(q_ptr, mask=offs_m[:, None] < S, other=0.0)
    else:
        q = tl.load(q_ptr)

    kv_base_k = K + b * skb + h * skh + offs_d[None, :]
    kv_base_v = V + b * svb + h * svh + offs_d[None, :]

    if N_ITERS == 1:
        # Whole key/value sequence in one tile: no running (m, l), so no
        # loop-carried rescaling -- just QK^T, softmax, PV.
        k_ptr = kv_base_k + offs_n[:, None] * sks
        v_ptr = kv_base_v + offs_n[:, None] * svs
        if MASK_N:
            n_ok = offs_n[:, None] < S
            k = tl.load(k_ptr, mask=n_ok, other=0.0)
            v = tl.load(v_ptr, mask=n_ok, other=0.0)
        else:
            k = tl.load(k_ptr)
            v = tl.load(v_ptr)
        # fp32 accumulation; the scale is pre-multiplied by log2(e) so the
        # softmax uses exp2 (one MUFU instruction) instead of exp.
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * qk_scale
        if MASK_N:
            qk = tl.where(offs_n[None, :] < S, qk, -1.0e30)
        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, -1.0e30)
        m_i = tl.max(qk, 1)
        p = tl.exp2(qk - m_i[:, None])
        l_i = tl.sum(p, 1)
        acc = tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
    else:
        # A handful of key tiles: standard online-softmax accumulation. Used
        # when a single tile would have to be padded far past S (e.g. S=144
        # needs BLOCK_N=256, wasting 44% of both loads and MMA work).
        acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        m_i = tl.full((BLOCK_M,), -1.0e30, dtype=tl.float32)
        for it in tl.range(0, N_ITERS):
            offs_j = it * BLOCK_N + offs_n
            k_ptr = kv_base_k + offs_j[:, None] * sks
            v_ptr = kv_base_v + offs_j[:, None] * svs
            if MASK_N:
                n_ok = offs_j[:, None] < S
                k = tl.load(k_ptr, mask=n_ok, other=0.0)
                v = tl.load(v_ptr, mask=n_ok, other=0.0)
            else:
                k = tl.load(k_ptr)
                v = tl.load(v_ptr)
            qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * qk_scale
            if MASK_N:
                qk = tl.where(offs_j[None, :] < S, qk, -1.0e30)
            if CAUSAL:
                qk = tl.where(offs_m[:, None] >= offs_j[None, :], qk, -1.0e30)
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v,
                                                out_dtype=tl.float32)
            m_i = m_new

    acc = acc / l_i[:, None]
    o_ptr = Out + b * sob + h * soh + offs_m[:, None] * sos + offs_d[None, :]
    if MASK_M:
        tl.store(o_ptr, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < S)
    else:
        tl.store(o_ptr, acc.to(Out.dtype.element_ty))


@triton.jit
def _grouped_attn_fwd(
    Q, K, V, Out,
    sqb, sqs, sqh,
    skb, sks, skh,
    svb, svs, svh,
    sob, sos, soh,
    H, S, BH, qk_scale,
    BLOCK: tl.constexpr, SG: tl.constexpr, D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """One tile, several *independent* (batch, head) problems packed into it.

    ``tl.dot`` cannot go below a 16x16 tile, so a sequence of 2..8 tokens wastes
    most of the tile -- and, worse, needs one program per (batch, head), i.e.
    2304 programs for the captured (B=144, H=16) shapes.  Program count is not
    free: an *empty* Triton kernel measured through the benchmark's timing loop
    costs 5.12 us up to ~1152 programs and 7.17 us at 2304 (B200).

    So pack ``BLOCK // SG`` problems into the 16-row tile, each owning ``SG``
    rows, and keep the QK^T tile block-diagonal by masking the cross-problem
    blocks.  The MMA work is identical (the tile was mostly padding anyway) and
    the program count drops by that factor.
    """
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    grp = offs // SG          # which packed problem this row belongs to
    row = offs % SG           # token index within that problem
    bh = pid * (BLOCK // SG) + grp
    b = bh // H
    h = bh - b * H
    ok_row = (bh < BH) & (row < S)
    offs_d = tl.arange(0, D)

    q = tl.load(Q + (b * sqb + h * sqh + row * sqs)[:, None] + offs_d[None, :],
                mask=ok_row[:, None], other=0.0)
    k = tl.load(K + (b * skb + h * skh + row * sks)[:, None] + offs_d[None, :],
                mask=ok_row[:, None], other=0.0)
    v = tl.load(V + (b * svb + h * svh + row * svs)[:, None] + offs_d[None, :],
                mask=ok_row[:, None], other=0.0)

    qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * qk_scale
    keep = (grp[:, None] == grp[None, :]) & ok_row[None, :]
    if CAUSAL:
        keep = keep & (row[:, None] >= row[None, :])
    qk = tl.where(keep, qk, -1.0e30)

    m_i = tl.max(qk, 1)
    p = tl.exp2(qk - m_i[:, None])
    l_i = tl.sum(p, 1)
    acc = tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
    acc = acc / l_i[:, None]

    tl.store(Out + (b * sob + h * soh + row * sos)[:, None] + offs_d[None, :],
             acc.to(Out.dtype.element_ty), mask=ok_row[:, None])


# ---------------------------------------------------------------------------
# Tier 1: bespoke CUDA C kernels, compiled with nvrtc at first use.
# ---------------------------------------------------------------------------
# Why hand-written CUDA at all, when Triton already serves these shapes?  Not
# for launch overhead -- that was measured and is *not* addressable: an empty
# kernel launched by Triton, by nvrtc + ``cuLaunchKernel``, and by nvcc +
# ``<<<>>>`` all add exactly 2.20-2.21 us over a no-kernel floor of 2.91 us in
# the benchmark's own timing loop, at every grid size.  What *is* addressable is
# the kernel body: 4.05-4.10 us of the 9.17-9.22 us these shapes measure.
#
# For the shapes this fast path serves, that body is pure latency.  The captured
# tiny cases move 1.8-4.4 MB (a fraction of a microsecond of B200 HBM) and do
# under 0.2 us of arithmetic, so what the Triton kernel spends its ~4 us on is
# the *chain*: loads that depend on earlier loads, `tl.dot` tile setup for
# 16x16 matrices that are mostly padding, and cross-tile online-softmax
# rescaling.  Written by hand the S<=8 case becomes one flat basic block per
# warp: every Q/K/V load for a problem issued back-to-back as 128-bit
# `ld.global.nc.v4.b32` so the cold-HBM round trips overlap instead of chaining,
# everything else in registers, no shared memory, no barriers, and plain FFMA
# (at S<=8 the matrices are far too small for an MMA to pay for its operand
# staging).
#
# Everything here is best-effort.  `_cu_init` and `_cu_function` swallow every
# failure -- no cuda-python, no nvrtc, an unsupported arch, a compile error, a
# module that will not load -- and memoize the failure, so the fast path silently
# degrades to the Triton kernels below, which in turn degrade to the untouched
# SDPA/cuDNN chain.  A build problem must never become an exception.
_CU_PRELUDE = r"""
typedef unsigned short u16;
typedef unsigned int   u32;
typedef long long      i64;

__device__ __forceinline__ float bits2f(u32 x) {
    float f; asm("mov.b32 %0, %1;" : "=f"(f) : "r"(x)); return f;
}
__device__ __forceinline__ float h2f(u16 h) {
    float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f;
}
__device__ __forceinline__ u16 f2h(float f) {
    u16 h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h;
}
/* bf16 -> f32 is an exact left shift; f32 -> bf16 needs round-to-nearest-even. */
__device__ __forceinline__ float b2f(u16 h) { return bits2f(((u32)h) << 16); }
__device__ __forceinline__ u16 f2b(float f) {
    u16 h; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h;
}
__device__ __forceinline__ float ex2(float x) {
    float r; asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x)); return r;
}
__device__ __forceinline__ float rcp(float x) {
    float r; asm("rcp.approx.f32 %0, %1;" : "=f"(r) : "f"(x)); return r;
}
__device__ __forceinline__ float bfly(float v, int m) {
    float r;
    asm("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
        : "=f"(r) : "f"(v), "r"(m));
    return r;
}
#if FP16
#define TOF(x)   h2f(x)
#define FROMF(x) f2h(x)
#else
#define TOF(x)   b2f(x)
#define FROMF(x) f2b(x)
#endif

/* NW = 32-bit words each lane loads per row; DPL = 2*NW elements of the head
   dimension it therefore owns.  LG lanes cooperate on one row so that a
   warp-wide load is one contiguous 4*32*NW-byte span per problem, and PPW
   independent problems share the warp. */
#define DPL (2 * NW)
#define LG  (D / DPL)
#define PPW (32 / LG)

__device__ __forceinline__ u32 ldg1(const void* p) {
    u32 r; asm("ld.global.nc.b32 %0, [%1];" : "=r"(r) : "l"(p)); return r;
}
/* Generic -> shared window address, as a 32-bit shared-space offset. */
__device__ __forceinline__ u32 smem_addr(const void* p) {
    u32 a;
    asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }"
        : "=r"(a) : "l"(p));
    return a;
}
__device__ __forceinline__ u32 lds1(u32 a) {
    u32 r; asm("ld.shared.b32 %0, [%1];" : "=r"(r) : "r"(a)); return r;
}
/* 16-byte global -> shared copy that does not pass through a register, so an
   arbitrary number can be in flight at once. */
__device__ __forceinline__ void cpasync16(u32 dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                 :: "r"(dst), "l"(src));
}
__device__ __forceinline__ void cpasync_wait_all() {
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
}
__device__ __forceinline__ void barrier() { asm volatile("bar.sync 0;"); }
__device__ __forceinline__ void stg1(void* p, u32 v) {
    asm("st.global.b32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
/* Transpose an 8x8 b16 matrix held across the warp -- one instruction, no
   shared memory.  The fragment slot is the same one `mma` uses: lane l holds
   M[l>>2][2*(l&3) + {0,1}], low half first. */
__device__ __forceinline__ u32 trans8x8(u32 x) {
    u32 r;
    asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(r) : "r"(x));
    return r;
}

#if FP16
#define MMAT  "f16"
#define CVT2X "cvt.rn.f16x2.f32"
#else
#define MMAT  "bf16"
#define CVT2X "cvt.rn.bf16x2.f32"
#endif

/* Two f32 -> one packed pair.  `cvt.rn.*x2.f32 d, a, b` puts a in the HIGH
   half and b in the LOW half (verified against torch in dev/frag.py). */
__device__ __forceinline__ u32 cvt2(float hi, float lo) {
    u32 r; asm(CVT2X " %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r;
}

/* D += A x B^T for A(16x16), B(8x16), D(16x8 f32).  Fragment slots, with
   g = lane>>2 and e = lane&3 (PTX ISA "Matrix Fragments for mma.m16n8k16"):
     a[0]=A[g][2e..]  a[1]=A[g+8][2e..]  a[2]=A[g][2e+8..]  a[3]=A[g+8][2e+8..]
     b[0]=B[g][2e..]  b[1]=B[g][2e+8..]
     d[0]=D[g][2e]  d[1]=D[g][2e+1]  d[2]=D[g+8][2e]  d[3]=D[g+8][2e+1]      */
__device__ __forceinline__ void mma16816(float* d, const u32* a, const u32* b) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32." MMAT "." MMAT ".f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

#if NW == 1
#define LDV(p, w) asm("ld.global.nc.b32 %0, [%1];" : "=r"(w[0]) : "l"(p))
#define STV(p, w) asm("st.global.b32 [%0], %1;" :: "l"(p), "r"(w[0]) : "memory")
#elif NW == 2
#define LDV(p, w) asm("ld.global.nc.v2.b32 {%0,%1}, [%2];" \
                      : "=r"(w[0]), "=r"(w[1]) : "l"(p))
#define STV(p, w) asm("st.global.v2.b32 [%0], {%1,%2};" \
                      :: "l"(p), "r"(w[0]), "r"(w[1]) : "memory")
#else
#define LDV(p, w) asm("ld.global.nc.v4.b32 {%0,%1,%2,%3}, [%4];" \
                      : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3]) : "l"(p))
#define STV(p, w) asm("st.global.v4.b32 [%0], {%1,%2,%3,%4};" \
                      :: "l"(p), "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]) \
                       : "memory")
#endif
"""

# Regime A: S <= 8.  One warp holds PPW = 32*DPL/D whole attention problems.
#
# Lane assignment: lane = slot*LG + chunk, so LG lanes each own a 16-byte
# (DPL=8 element) slice of the head dimension for one (batch, head) problem, and
# the 32 lanes of a warp cover PPW independent problems.  Q/K/V for a problem are
# S rows of DPL elements per lane -- all 3*S loads are issued before the first
# use, which is the whole point: one HBM round trip for the problem instead of a
# dependent chain.  S is a compile-time constant so every loop unrolls and every
# array stays in registers (no local memory, no dynamic indexing).
#
# QK^T needs a sum across the LG lanes of a slot; that is log2(LG) butterfly
# shuffles (3 for D=64) on a value all LG lanes then need anyway.  P*V needs no
# communication at all: a lane already holds V for its own DPL columns.
_CU_GROUPED = r"""
extern "C" __global__ __launch_bounds__(WARPS * 32, MINB)
void attn_grouped(const u16* __restrict__ Q, const u16* __restrict__ K,
                  const u16* __restrict__ V, u16* __restrict__ O,
                  i64 sqb, i64 sqs, i64 sqh,
                  i64 skb, i64 sks, i64 skh,
                  i64 svb, i64 svs, i64 svh,
                  int NH, int BH, float scale)
{
    const int lane = (int)(threadIdx.x & 31u);
    const int wid  = (int)(threadIdx.x >> 5);
    const int slot = lane / LG;
    const int cl   = lane - slot * LG;

    /* Out-of-range slots must not exit: they still have to take part in the
       butterfly shuffles below.  Clamp to problem 0 and drop only the store. */
    const int bh0  = ((int)blockIdx.x * WARPS + wid) * PPW + slot;
    const bool live = bh0 < BH;
    const int bh   = live ? bh0 : 0;
    const int b    = bh / NH;
    const int h    = bh - b * NH;

    const u16* qp = Q + b * sqb + h * sqh + cl * DPL;
    const u16* kp = K + b * skb + h * skh + cl * DPL;
    const u16* vp = V + b * svb + h * svh + cl * DPL;

    u32 qw[S][NW], kw[S][NW], vw[S][NW];
    #pragma unroll
    for (int i = 0; i < S; ++i) LDV(qp + (i64)i * sqs, qw[i]);
    #pragma unroll
    for (int i = 0; i < S; ++i) LDV(kp + (i64)i * sks, kw[i]);
    #pragma unroll
    for (int i = 0; i < S; ++i) LDV(vp + (i64)i * svs, vw[i]);

    float kf[S][DPL], vf[S][DPL];
    #pragma unroll
    for (int j = 0; j < S; ++j) {
        #pragma unroll
        for (int t = 0; t < NW; ++t) {
            kf[j][2 * t]     = TOF((u16)(kw[j][t] & 0xffffu));
            kf[j][2 * t + 1] = TOF((u16)(kw[j][t] >> 16));
            vf[j][2 * t]     = TOF((u16)(vw[j][t] & 0xffffu));
            vf[j][2 * t + 1] = TOF((u16)(vw[j][t] >> 16));
        }
    }

    u16* op = O + ((i64)bh * S) * D + cl * DPL;
    #pragma unroll
    for (int i = 0; i < S; ++i) {
        float qf[DPL];
        #pragma unroll
        for (int t = 0; t < NW; ++t) {
            qf[2 * t]     = TOF((u16)(qw[i][t] & 0xffffu));
            qf[2 * t + 1] = TOF((u16)(qw[i][t] >> 16));
        }
        /* Scores for this query row.  With CAUSAL the j > i half is dropped at
           compile time (both loop bounds are constants after unrolling). */
        float p[S];
        float mx = -3.0e38f;
        #pragma unroll
        for (int j = 0; j < S; ++j) {
            if (CAUSAL && j > i) { p[j] = 0.0f; continue; }
            float a = 0.0f;
            #pragma unroll
            for (int t = 0; t < DPL; ++t) a += qf[t] * kf[j][t];
            #pragma unroll
            for (int m = LG >> 1; m > 0; m >>= 1) a += bfly(a, m);
            a *= scale;                 /* scale already folded with log2(e) */
            p[j] = a;
            mx = a > mx ? a : mx;
        }
        float l = 0.0f;
        #pragma unroll
        for (int j = 0; j < S; ++j) {
            if (CAUSAL && j > i) continue;
            p[j] = ex2(p[j] - mx);
            l += p[j];
        }
        float acc[DPL];
        #pragma unroll
        for (int t = 0; t < DPL; ++t) acc[t] = 0.0f;
        #pragma unroll
        for (int j = 0; j < S; ++j) {
            if (CAUSAL && j > i) continue;
            #pragma unroll
            for (int t = 0; t < DPL; ++t) acc[t] += p[j] * vf[j][t];
        }
        const float r = rcp(l);
        u32 ow[NW];
        #pragma unroll
        for (int t = 0; t < NW; ++t)
            ow[t] = (u32)FROMF(acc[2 * t] * r)
                  | ((u32)FROMF(acc[2 * t + 1] * r) << 16);
        if (live) { u16* o = op + (i64)i * D; STV(o, ow); }
    }
}
"""

# Regime B: 8 < S <= 256, D a multiple of 16.  Here the matrices are big enough
# that FFMA is hopeless -- the captured `[3,144,16,64]` case is 127 MFMA, which is
# 3.4 us of B200 FP32 FFMA on its own -- so this uses `mma.m16n8k16` tensor-core
# instructions and lands the arithmetic in ~0.4 us instead.
#
# A CTA owns one (batch, head); its WARPS warps each own MT query rows of it.
#
#  * K and V for the whole key sequence are copied global -> shared with
#    `cp.async` before anything else runs.  Those copies never pass through a
#    register, so every one of them is in flight simultaneously: the kernel pays
#    *one* HBM round trip and then never touches global memory again for K/V.
#    (Reading MMA fragments straight from global instead was measured at 13.2 us
#    against Triton's 9.2 -- 9 key tiles is 9 serialized round trips, and holding
#    more than one tile of fragments in registers does not fit.)
#  * Q and K feed `mma.m16n8k16.row.col` fragments with no rearrangement at all:
#    the A slot wants two consecutive head-dim elements of one row, and so does
#    the B slot, so both are plain 32-bit loads.  The shared row stride is padded
#    by 8 elements so that the 8 rows x 4 lanes of one fragment read hit 32
#    distinct banks.
#  * The QK^T accumulator comes out in *exactly* the slot pattern the PV
#    multiply's A operand wants (n-subtile 2t supplies a0/a1 and 2t+1 supplies
#    a2/a3), so the softmax result flows into the second MMA with no data
#    movement beyond an f32 -> f16x2 pack.
#  * PV's B operand is the one thing that needs a transpose (it wants V indexed
#    [d][j] while V is stored [j][d]).  `movmatrix.sync.aligned.m8n8.trans.b16`
#    does that across the warp in one instruction -- no separate transposed
#    staging buffer and no `ldmatrix.trans` addressing.
#  * The softmax row reductions are over the four lanes of a fragment group, i.e.
#    two butterfly shuffles each.
#
# Row/column masking is compile-time-eliminated when S is a multiple of both MT
# and BN (which the captured S=144 is, for MT=BN=16), and the causal case skips
# whole key tiles above the diagonal.
_CU_TILED = r"""
#define MB  (MT / 16)               /* 16-row m-tiles per warp            */
#define NB  (BN / 8)                /* 8-key n-subtiles per key tile      */
#define KB  (D / 16)                /* 16-wide k-steps of QK^T            */
#define DB  (D / 8)                 /* 8-column n-subtiles of PV          */
#define KJ  (BN / 16)               /* 16-key A-fragments per key tile    */
#define NQ  ((S + MT - 1) / MT)     /* query blocks per (batch, head)      */
#define CPB ((NQ + MQ - 1) / MQ)    /* CTAs per (batch, head)              */
#define NT  ((S + BN - 1) / BN)     /* key tiles                           */
#define WARPS MQ
#define SMASK ((S % MT != 0) || (S % BN != 0))
#define NEG (-3.0e38f)
#define SST (D + 8)                 /* shared row stride, in elements      */
#define NCH (S * (D / 8))           /* 16-byte chunks per staged tensor     */
#define NTH (WARPS * 32)

extern "C" __global__ __launch_bounds__(NTH, MINB)
void attn_tiled(const u16* __restrict__ Q, const u16* __restrict__ K,
                const u16* __restrict__ V, u16* __restrict__ O,
                i64 sqb, i64 sqs, i64 sqh,
                i64 skb, i64 sks, i64 skh,
                i64 svb, i64 svs, i64 svh,
                int NH, int BH, float scale)
{
    extern __shared__ __align__(16) u16 smem[];
    const int tid  = (int)threadIdx.x;
    const int lane = tid & 31;
    const int g = lane >> 2, e = lane & 3;
    /* Every warp of a CTA shares one (batch, head) so they can share the staged
       K/V.  The warp index splits into (query block, key chunk). */
    const int bh   = (int)blockIdx.x / CPB;
    const int cb   = (int)blockIdx.x - bh * CPB;
    const int mblk = cb * MQ + (tid >> 5);
    const int b = bh / NH, h = bh - b * NH;
    const int m0 = mblk * MT;

    const u16* qp = Q + b * sqb + h * sqh;
    const u16* kp = K + b * skb + h * skh;
    const u16* vp = V + b * svb + h * svh;

    /* Stage K and V.  All NCH*2/NTH copies are issued before the first wait. */
    const u32 skb32 = smem_addr(smem);
    const u32 svb32 = skb32 + (u32)(S * SST) * 2u;
    for (int c = tid; c < NCH; c += NTH) {
        const int j = c / (D / 8);
        const int d0 = (c - j * (D / 8)) * 8;
        const u32 off = (u32)(j * SST + d0) * 2u;
        cpasync16(skb32 + off, kp + (i64)j * sks + d0);
        cpasync16(svb32 + off, vp + (i64)j * svs + d0);
    }

    /* Q fragments, loaded once and held for the whole key loop.  Rows past S
       read row 0 instead; their outputs are never stored.  These are global
       loads, issued while the cp.async copies are still in flight. */
    u32 qf[MB][KB][4];
    #pragma unroll
    for (int mi = 0; mi < MB; ++mi) {
        int r0 = m0 + mi * 16 + g, r1 = r0 + 8;
        if (r0 >= S) r0 = 0;
        if (r1 >= S) r1 = 0;
        #pragma unroll
        for (int kk = 0; kk < KB; ++kk) {
            const int d0 = kk * 16 + 2 * e;
            qf[mi][kk][0] = ldg1(qp + (i64)r0 * sqs + d0);
            qf[mi][kk][1] = ldg1(qp + (i64)r1 * sqs + d0);
            qf[mi][kk][2] = ldg1(qp + (i64)r0 * sqs + d0 + 8);
            qf[mi][kk][3] = ldg1(qp + (i64)r1 * sqs + d0 + 8);
        }
    }

    float acc[MB][DB][4], mrow[MB][2], lrow[MB][2];
    #pragma unroll
    for (int mi = 0; mi < MB; ++mi) {
        #pragma unroll
        for (int c = 0; c < DB; ++c)
            #pragma unroll
            for (int t = 0; t < 4; ++t) acc[mi][c][t] = 0.0f;
        mrow[mi][0] = mrow[mi][1] = NEG;
        lrow[mi][0] = lrow[mi][1] = 0.0f;
    }

    cpasync_wait_all();
    barrier();

    /* Warps past the last query block only existed to help stage K/V. */
    if (mblk >= NQ) return;

    #pragma unroll TUNROLL
    for (int tile = 0; tile < NT; ++tile) {
        const int jb = tile * BN;
        if (CAUSAL && jb > m0 + MT - 1) break;

        /* K fragments (B operand of QK^T: two consecutive head-dim elements of
           one key row -- contiguous, so a plain 32-bit shared load). */
        u32 kf[NB][KB][2];
        #pragma unroll
        for (int n = 0; n < NB; ++n) {
            int j = jb + n * 8 + g;
            if (SMASK && j >= S) j = 0;
            const u32 row = skb32 + (u32)(j * SST) * 2u;
            #pragma unroll
            for (int kk = 0; kk < KB; ++kk) {
                const u32 d0 = (u32)(kk * 16 + 2 * e) * 2u;
                kf[n][kk][0] = lds1(row + d0);
                kf[n][kk][1] = lds1(row + d0 + 16u);
            }
        }
        /* V fragments (B operand of PV, which needs V indexed [d][j]).  Read in
           the natural [j][d] slot and transpose across the warp. */
        u32 vf[KJ][DB][2];
        #pragma unroll
        for (int jt = 0; jt < KJ; ++jt) {
            int ja = jb + jt * 16 + g, jc = ja + 8;
            if (SMASK) { if (ja >= S) ja = 0; if (jc >= S) jc = 0; }
            const u32 ra = svb32 + (u32)(ja * SST) * 2u;
            const u32 rc = svb32 + (u32)(jc * SST) * 2u;
            #pragma unroll
            for (int c = 0; c < DB; ++c) {
                const u32 d0 = (u32)(c * 8 + 2 * e) * 2u;
                vf[jt][c][0] = trans8x8(lds1(ra + d0));
                vf[jt][c][1] = trans8x8(lds1(rc + d0));
            }
        }

        #pragma unroll
        for (int mi = 0; mi < MB; ++mi) {
            /* QK^T for this (m-tile, key tile). */
            float s[NB][4];
            #pragma unroll
            for (int n = 0; n < NB; ++n) {
                #pragma unroll
                for (int t = 0; t < 4; ++t) s[n][t] = 0.0f;
                #pragma unroll
                for (int kk = 0; kk < KB; ++kk)
                    mma16816(s[n], qf[mi][kk], kf[n][kk]);
            }
            /* Scale, mask, and this tile's row maxima.  Slot t < 2 is row
               g, t >= 2 is row g+8; column is 8n + 2e + (t & 1). */
            const int rlo = m0 + mi * 16 + g, rhi = rlo + 8;
            float mx0 = NEG, mx1 = NEG;
            #pragma unroll
            for (int n = 0; n < NB; ++n) {
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const int r = (t < 2) ? rlo : rhi;
                    const int j = jb + n * 8 + 2 * e + (t & 1);
                    float x = s[n][t] * scale;
                    bool ok = true;
                    if (SMASK) ok = (j < S) && (r < S);
                    if (CAUSAL) ok = ok && (j <= r);
                    x = ok ? x : NEG;
                    s[n][t] = x;
                    if (t < 2) { if (x > mx0) mx0 = x; }
                    else       { if (x > mx1) mx1 = x; }
                }
            }
            /* The 8 columns of a row live in the 4 lanes of one fragment
               group, so a row reduction is two butterfly shuffles. */
            #pragma unroll
            for (int q = 1; q < 4; q <<= 1) {
                const float t0 = bfly(mx0, q), t1 = bfly(mx1, q);
                if (t0 > mx0) mx0 = t0;
                if (t1 > mx1) mx1 = t1;
            }
            const float mn0 = mx0 > mrow[mi][0] ? mx0 : mrow[mi][0];
            const float mn1 = mx1 > mrow[mi][1] ? mx1 : mrow[mi][1];
            const float a0 = ex2(mrow[mi][0] - mn0);
            const float a1 = ex2(mrow[mi][1] - mn1);
            float sum0 = 0.0f, sum1 = 0.0f;
            #pragma unroll
            for (int n = 0; n < NB; ++n) {
                s[n][0] = ex2(s[n][0] - mn0); sum0 += s[n][0];
                s[n][1] = ex2(s[n][1] - mn0); sum0 += s[n][1];
                s[n][2] = ex2(s[n][2] - mn1); sum1 += s[n][2];
                s[n][3] = ex2(s[n][3] - mn1); sum1 += s[n][3];
            }
            #pragma unroll
            for (int q = 1; q < 4; q <<= 1) {
                sum0 += bfly(sum0, q);
                sum1 += bfly(sum1, q);
            }
            lrow[mi][0] = lrow[mi][0] * a0 + sum0;
            lrow[mi][1] = lrow[mi][1] * a1 + sum1;
            mrow[mi][0] = mn0;
            mrow[mi][1] = mn1;
            #pragma unroll
            for (int c = 0; c < DB; ++c) {
                acc[mi][c][0] *= a0; acc[mi][c][1] *= a0;
                acc[mi][c][2] *= a1; acc[mi][c][3] *= a1;
            }
            /* PV.  The QK^T accumulator slots are already the A slots the PV
               mma wants: n-subtile 2*jt gives a0/a1, 2*jt+1 gives a2/a3. */
            #pragma unroll
            for (int jt = 0; jt < KJ; ++jt) {
                u32 af[4];
                af[0] = cvt2(s[2 * jt][1], s[2 * jt][0]);
                af[1] = cvt2(s[2 * jt][3], s[2 * jt][2]);
                af[2] = cvt2(s[2 * jt + 1][1], s[2 * jt + 1][0]);
                af[3] = cvt2(s[2 * jt + 1][3], s[2 * jt + 1][2]);
                #pragma unroll
                for (int c = 0; c < DB; ++c)
                    mma16816(acc[mi][c], af, vf[jt][c]);
            }
        }
    }
    }

    u16* op = O + ((i64)bh * S) * D;
    #pragma unroll
    for (int mi = 0; mi < MB; ++mi) {
        const float r0 = rcp(lrow[mi][0]), r1 = rcp(lrow[mi][1]);
        const int rlo = m0 + mi * 16 + g, rhi = rlo + 8;
        #pragma unroll
        for (int c = 0; c < DB; ++c) {
            const int d0 = c * 8 + 2 * e;
            const u32 w0 = cvt2(acc[mi][c][1] * r0, acc[mi][c][0] * r0);
            const u32 w1 = cvt2(acc[mi][c][3] * r1, acc[mi][c][2] * r1);
            if (!SMASK || rlo < S) stg1(op + (i64)rlo * D + d0, w0);
            if (!SMASK || rhi < S) stg1(op + (i64)rhi * D + d0, w1);
        }
    }
}
"""

# Argument block for `attn_grouped`, packed once per call with `struct.pack` and
# handed to `cuLaunchKernel` through the driver's CU_LAUNCH_PARAM_BUFFER_POINTER
# mechanism.  Offsets are naturally aligned (4 pointers, 9 int64, 2 int, float),
# which is what the kernel's parameter space expects.
_CU_GROUPED_FMT = "<4Q9q2if"

_CU_STATE: dict | None | bool = None


def _cu_init():
    """Bring up nvrtc + the driver API once.  ``False`` means unavailable."""
    global _CU_STATE
    if _CU_STATE is not None:
        return _CU_STATE
    _CU_STATE = False
    try:
        if not torch.cuda.is_available():
            return _CU_STATE
        from cuda.bindings import driver as cud, nvrtc
        # torch's lazy init is what creates/binds the primary context; without a
        # current context cuModuleLoadData fails with CUDA_ERROR_INVALID_CONTEXT.
        torch.cuda.init()
        torch.empty(1, device="cuda")
        err, ctx = cud.cuCtxGetCurrent()
        if err != cud.CUresult.CUDA_SUCCESS or int(ctx) == 0:
            return _CU_STATE
        major, minor = torch.cuda.get_device_capability()
        if major < 8:                    # bf16 cvt / shfl.sync need sm_80+
            return _CU_STATE
        _CU_STATE = {
            "cud": cud,
            "nvrtc": nvrtc,
            "arch": f"--gpu-architecture=sm_{major}{minor}".encode(),
            "fns": {},
        }
    except Exception:
        _CU_STATE = False
    return _CU_STATE


def _cu_function(key, source, name, defines):
    """Compile (once) and return a CUfunction, or ``None`` if anything failed."""
    st = _cu_init()
    if not st:
        return None
    cache = st["fns"]
    if key in cache:
        return cache[key]
    cache[key] = None                    # memoize failure by default
    try:
        cud, nvrtc = st["cud"], st["nvrtc"]
        src = ("".join(f"#define {k} {v}\n" for k, v in defines.items())
               + _CU_PRELUDE + source)
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"fk.cu", 0, [], [])
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            return None
        opts = [st["arch"], b"--std=c++17"]
        status = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)[0]
        if status != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            return None
        size = nvrtc.nvrtcGetCUBINSize(prog)[1]
        cubin = bytearray(size)
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, mod = cud.cuModuleLoadData(bytes(cubin))
        if err != cud.CUresult.CUDA_SUCCESS:
            return None
        err, fn = cud.cuModuleGetFunction(mod, name.encode())
        if err != cud.CUresult.CUDA_SUCCESS:
            return None
        # Keep the module alive for the process; CUfunction does not own it.
        st.setdefault("mods", []).append(mod)
        cache[key] = fn
    except Exception:
        cache[key] = None
    return cache[key]


class _CuLauncher:
    """Pre-built ``cuLaunchKernel`` argument block for one compiled kernel.

    The packed parameter buffer and the ``extra`` descriptor are allocated once;
    each call only rewrites the buffer's bytes, so a launch is one
    ``struct.pack_into`` plus the driver call.
    """

    __slots__ = ("fn", "fmt", "block", "shmem", "buf", "extra", "extra_addr",
                 "_csize", "_cud", "_pack", "_launch")

    def __init__(self, fn, fmt, block, cud, shmem=0):
        import ctypes
        import struct
        self.fn, self.fmt, self.block, self.shmem = fn, fmt, block, shmem
        self._cud = cud
        self._pack = struct.pack_into
        self._launch = cud.cuLaunchKernel
        size = struct.calcsize(fmt)
        self.buf = ctypes.create_string_buffer(size)
        self._csize = ctypes.c_size_t(size)          # must outlive `extra`
        # {CU_LAUNCH_PARAM_BUFFER_POINTER, buf, CU_LAUNCH_PARAM_BUFFER_SIZE,
        #  &size, CU_LAUNCH_PARAM_END}
        self.extra = (ctypes.c_void_p * 5)(
            1, ctypes.addressof(self.buf), 2, ctypes.addressof(self._csize), 0)
        self.extra_addr = ctypes.addressof(self.extra)

    def __call__(self, grid, stream, *args):
        self._pack(self.fmt, self.buf, 0, *args)
        return self._launch(self.fn, grid, 1, 1, self.block, 1, 1, self.shmem,
                            stream, 0, self.extra_addr)


# Largest key/value sequence the single-block fast path will take.  Beyond this
# the BLOCK_M x BLOCK_N score tile stops fitting in registers and a real
# flash-attention loop (i.e. SDPA's cuDNN kernel) is the right tool.
_FAST_MAX_SEQ = 256
_FAST_DTYPES = (torch.float16, torch.bfloat16)
_FAST_HEAD_DIMS = (16, 32, 64, 128)

# Tile shape as a function of the *padded* key length, i.e. the smallest
# power-of-two >= seq_len:  (BLOCK_M, BLOCK_N, num_warps, num_stages).  A
# BLOCK_N below the padded length turns the kernel into an online-softmax loop
# over ceil(seq/BLOCK_N) tiles, which is how the seq=144 shapes avoid paying for
# a 256-wide tile.  Tuned on B200; see the sweep table in ITERATIONS.md.
_TILES = {
    16: (16, 16, 1, 1),
    32: (16, 32, 1, 1),
    64: (16, 32, 1, 2),
    128: (16, 32, 1, 4),
    256: (16, 32, 1, 4),
}

# (S, D, causal, dtype) -> launch configuration.  Keyed on the things that change
# the compiled kernel; strides and batch/head counts stay runtime arguments so
# one entry serves every batch size.
_LAUNCH_CACHE: dict = {}

# Warps per CTA for the bespoke S<=8 CUDA kernel.  Threads per CTA are free but
# CTAs are not: an empty kernel measures 5.12 us at 576 CTAs x 128 threads and
# 7.0 us at 1152 CTAs x 32 threads, because a wider grid's launch ramp is itself
# part of the kernel's duration and can push it over a step boundary.  With D=64
# a warp holds 4 problems, so 4 warps/CTA covers the captured B*H = 2304 in 144.
_CU_WARPS = 4
# 32-bit words each lane loads per row.  NW=4 is a 128-bit load, which puts
# 32/(D/8) whole problems in one warp; NW=1 spreads one problem across all 32
# lanes instead.  Once MINB below stops ptxas from serializing the loads, the
# whole (NW x WARPS x MINB) space measures the same, so this is simply the widest
# load -- see the flat sweep in ITERATIONS.md.
_CU_NW = 4
# minBlocksPerMultiprocessor hint.  `-maxrregcount` is only a ceiling: what
# actually decides how many registers ptxas grants -- and therefore whether all
# 3*S loads stay in flight or get serialized into a dependent chain -- is the
# occupancy target it infers from __launch_bounds__.
_CU_MINB = 2

# Tuning constants for the 8 < S <= 256 tensor-core kernel.  MT is the query rows
# one warp owns (a multiple of 16, the MMA's m); raising it amortizes the K/V
# fragment loads over more rows at the cost of MT/16 x D/8 x 4 accumulator
# registers.  BN is the keys per online-softmax step (a multiple of 16).
_CUT_MT = 16
_CUT_BN = 48
# Query blocks per CTA.  This is the K/V-traffic dial: the CTA stages K and V for
# its whole (batch, head), so `ceil(ceil(S/MT)/MQ)` CTAs per (batch, head) each
# re-read them.  MQ=3 is the smallest value that still puts >= 148 CTAs on the
# captured shapes, i.e. that uses every SM.
_CUT_MQ = 3
_CUT_MINB = 1
_CUT_UNROLL = 3

_CU_CACHE: dict = {}
_MISSING = object()


def _cu_vectorizable(t, elems: int) -> bool:
    """Can every (b, s, h) row of ``t`` be reached by ``elems``-wide loads?"""
    s = t.stride()
    return (t.data_ptr() % (2 * elems) == 0 and s[0] % elems == 0
            and s[1] % elems == 0 and s[2] % elems == 0)


def _cu_plan(seq: int, head_dim: int, causal: bool, dtype):
    """Compile (once) the bespoke kernel for this shape, or return ``None``.

    Returns ``(launcher, align_elems, programs_per_bh, programs_per_cta)``, where
    the launch grid is ``ceil(B*H*programs_per_bh / programs_per_cta)``.

    ``None`` is the normal answer for anything the CUDA tier does not cover, and
    also for every kind of build failure -- the caller then uses Triton.
    """
    key = (seq, head_dim, causal, dtype)
    if key in _CU_CACHE:
        return _CU_CACHE[key]
    _CU_CACHE[key] = None
    st = _cu_init()
    if not st:
        return _CU_CACHE[key]
    fp16 = int(dtype is torch.float16)

    if seq <= 8:
        # Widest vector load that still leaves at least one lane group per warp.
        nw = min(_CU_NW, head_dim // 2)
        while nw > 1 and (head_dim // (2 * nw)) not in (1, 2, 4, 8, 16, 32):
            nw //= 2
        dpl = 2 * nw
        if head_dim % dpl or head_dim // dpl not in (1, 2, 4, 8, 16, 32):
            return _CU_CACHE[key]
        defines = {
            "S": seq, "D": head_dim, "CAUSAL": int(causal), "NW": nw,
            "FP16": fp16, "WARPS": _CU_WARPS, "MINB": _CU_MINB,
        }
        fn = _cu_function(("grouped",) + tuple(defines.items()),
                          _CU_GROUPED, "attn_grouped", defines)
        block, align, shmem = _CU_WARPS * 32, dpl, 0
        # problems per CTA = (32 / lanes-per-problem) * warps
        per_bh, per_cta = 1, (32 // (head_dim // dpl)) * _CU_WARPS
    else:
        mq = _CUT_MQ
        # A 16-key tile must not straddle the end of the sequence by more than
        # the mask can express, so prefer the largest step that divides S.
        bn = next((n for n in (_CUT_BN, 32, 16) if seq % n == 0), 16)
        if head_dim % 16 or _CUT_MT % 16 or mq > 32:
            return _CU_CACHE[key]
        # K and V for the whole key sequence, rows padded by 8 elements so that
        # one fragment read touches 32 distinct banks.
        shmem = 4 * seq * (head_dim + 8)
        if shmem > _cu_max_shmem(st):
            return _CU_CACHE[key]
        defines = {
            "S": seq, "D": head_dim, "CAUSAL": int(causal), "FP16": fp16,
            "MT": _CUT_MT, "BN": bn, "MQ": mq,
            "MINB": _CUT_MINB, "TUNROLL": _CUT_UNROLL,
        }
        fn = _cu_function(("tiled",) + tuple(defines.items()),
                          _CU_TILED, "attn_tiled", defines)
        block, align = mq * 32, 8
        nq = -(-seq // _CUT_MT)
        per_bh, per_cta = -(-nq // mq), 1

    if fn is None:
        return _CU_CACHE[key]
    try:
        cud = st["cud"]
        if shmem > 48 * 1024:
            # Anything above the 48 KB static limit has to be opted into.
            err = cud.cuFuncSetAttribute(
                fn,
                cud.CUfunction_attribute
                   .CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shmem)
            if err[0] != cud.CUresult.CUDA_SUCCESS:
                return _CU_CACHE[key]
        launcher = _CuLauncher(fn, _CU_GROUPED_FMT, block, cud, shmem)
    except Exception:
        return _CU_CACHE[key]
    _CU_CACHE[key] = (launcher, align, per_bh, per_cta)
    return _CU_CACHE[key]


def _cu_max_shmem(st) -> int:
    """Per-block shared memory the device will opt in to, cached on ``st``."""
    n = st.get("maxsh")
    if n is None:
        n = 0
        try:
            cud = st["cud"]
            dev = cud.cuCtxGetDevice()[1]
            n = cud.cuDeviceGetAttribute(
                cud.CUdevice_attribute
                   .CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                dev)[1]
        except Exception:
            n = 48 * 1024
        st["maxsh"] = n
    return n


def _short_attn_config(seq: int, head_dim: int, causal: bool):
    # Very short sequences: pack several (batch, head) problems per 16-row tile.
    if seq <= 8:
        sg = max(1, triton.next_power_of_2(seq))
        return (True, 16 // sg, 16, sg, head_dim, causal, 1, 1)
    block_m, block_n, num_warps, num_stages = _TILES[
        max(16, triton.next_power_of_2(seq))]
    n_iters = triton.cdiv(seq, block_n)
    return (
        False,
        triton.cdiv(seq, block_m),     # grid rows (batch*head is the 2nd axis)
        block_m,
        block_n,
        n_iters,
        head_dim,
        causal,
        seq % block_m != 0,            # MASK_M
        n_iters * block_n != seq,      # MASK_N
        num_warps,
        num_stages,
    )


def _short_attn(query, key, value, softmax_scale, causal, cache_key):
    batch, seq, heads, head_dim = query.shape
    # Allocated (B, H, S, D)-contiguous and returned as a (B, S, H, D) view --
    # the same layout the SDPA path returns. A program owns one (batch, head)
    # and a run of consecutive tokens, so this makes its output rows one
    # contiguous span instead of D-sized chunks strided by H*D.
    out = torch.empty((batch, heads, seq, head_dim),
                      dtype=query.dtype, device=query.device)
    sq = query.stride()
    sk = key.stride()
    sv = value.stride()
    scale = head_dim ** -0.5 if softmax_scale is None else softmax_scale
    bh = batch * heads

    # Tier 1: bespoke CUDA kernel.  Needs 16-byte-addressable rows; a layout that
    # is not vectorizable falls through to Triton, which loads element-wise.
    cu = _CU_CACHE.get(cache_key, _MISSING)
    if cu is _MISSING:
        cu = _cu_plan(seq, head_dim, bool(causal), query.dtype)
    if cu is not None and (_cu_vectorizable(query, cu[1])
                           and _cu_vectorizable(key, cu[1])
                           and _cu_vectorizable(value, cu[1])):
        launcher, _, per_bh, per_cta = cu
        launcher(
            -(-(bh * per_bh) // per_cta),
            torch.cuda.current_stream().cuda_stream,
            query.data_ptr(), key.data_ptr(), value.data_ptr(), out.data_ptr(),
            sq[0], sq[1], sq[2],
            sk[0], sk[1], sk[2],
            sv[0], sv[1], sv[2],
            heads, bh, scale * _LOG2_E,
        )
        return out.permute(0, 2, 1, 3)

    # Tier 2: Triton.
    cfg = _LAUNCH_CACHE.get(cache_key)
    if cfg is None:
        cfg = _short_attn_config(seq, head_dim, bool(causal))
        _LAUNCH_CACHE[cache_key] = cfg

    if cfg[0]:
        _, per_prog, block, sg, _, is_causal, num_warps, num_stages = cfg
        _grouped_attn_fwd[(-(-bh // per_prog),)](
            query, key, value, out,
            sq[0], sq[1], sq[2],
            sk[0], sk[1], sk[2],
            sv[0], sv[1], sv[2],
            heads * seq * head_dim, head_dim, seq * head_dim,
            heads, seq, bh, scale * _LOG2_E,
            BLOCK=block, SG=sg, D=head_dim, CAUSAL=is_causal,
            num_warps=num_warps, num_stages=num_stages,
        )
        return out.permute(0, 2, 1, 3)

    (_, grid_m, block_m, block_n, n_iters, _, is_causal,
     mask_m, mask_n, num_warps, num_stages) = cfg
    _short_attn_fwd[(grid_m, bh)](
        query, key, value, out,
        sq[0], sq[1], sq[2],
        sk[0], sk[1], sk[2],
        sv[0], sv[1], sv[2],
        heads * seq * head_dim, head_dim, seq * head_dim,
        heads, seq, scale * _LOG2_E,
        BLOCK_M=block_m, BLOCK_N=block_n, N_ITERS=n_iters, D=head_dim,
        CAUSAL=is_causal, MASK_M=mask_m, MASK_N=mask_n,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out.permute(0, 2, 1, 3)


class DenseAttention(nn.Module):
    """Dense multi-head attention.

    Input layout: (batch, seq_len, num_heads, head_dim).

    Args:
        backend: Which kernel to use.
            ``"auto"`` selects flash-attention on Ampere/Hopper when
            available, SDPA everywhere else.
            ``"sdpa"`` always uses ``F.scaled_dot_product_attention``
            (PyTorch's heuristic chooses among flash/cuDNN/mem_eff/math).
            ``"flash_attn"`` always uses the flash-attention package.
            ``"cudnn"`` pins the cuDNN flash backend via
            ``torch.nn.attention.sdpa_kernel`` (with MATH fallback for
            masks cuDNN can't handle). Required to get cuDNN flash
            through ``torch.compile`` on Blackwell.

    Regardless of ``backend``, unmasked calls with ``seq_len <= 256`` and a
    power-of-two ``head_dim <= 128`` are served by the fused Triton kernel
    (see the module docstring); everything else uses the selected backend.
    """

    def __init__(self, backend: Literal["auto", "sdpa", "flash_attn", "cudnn", "flex"] = "auto"):
        super().__init__()
        self.fa_func = None
        self.use_cudnn_kernel = False
        self.use_flex_kernel = False
        self._flex_fn = None

        if backend == "sdpa":
            return

        if backend == "cudnn":
            self.use_cudnn_kernel = True
            return

        if backend == "flex":
            from torch.nn.attention.flex_attention import flex_attention
            self.use_flex_kernel = True
            self._flex_fn = torch.compile(flex_attention, dynamic=False)
            return

        if backend == "flash_attn":
            self.fa_func = _resolve_flash_attn_func()
            return

        # backend == "auto": flash-attn on Ampere/Hopper (80<=cc<100); cuDNN flash
        # on Blackwell (cc>=100), where PyTorch's SDPA heuristic otherwise picks
        # FA2 (~3.6x slower than cuDNN for large joint-attention shapes on B200).
        # This mirrors vllm-omni's platform selector, which pins cuDNN/TRTLLM on
        # Blackwell. The cuDNN forward path already falls back to mem-efficient/MATH
        # for shapes/masks cuDNN rejects, so this is safe as a default.
        cc = (torch.cuda.get_device_capability()[0] * 10
              + torch.cuda.get_device_capability()[1])
        if 80 <= cc < 100:
            self.fa_func = _resolve_flash_attn_func()
        elif cc >= 100:
            self.use_cudnn_kernel = True

    def forward(
        self,
        query,
        key,
        value,
        softmax_scale=None,
        causal=False,
        attn_mask: torch.Tensor | None = None,
    ):
        # Fused short-sequence path (see module docstring). The predicate is a
        # handful of attribute reads; the resulting launch config is memoized on
        # (seq_len, head_dim, causal), so nothing here re-derives tile shapes.
        if attn_mask is None and not self.use_flex_kernel:
            shape = query.shape
            if (len(shape) == 4
                    and shape[1] <= _FAST_MAX_SEQ
                    and shape[3] in _FAST_HEAD_DIMS
                    and query.dtype in _FAST_DTYPES
                    and query.is_cuda
                    and key.shape == shape and value.shape == shape
                    and query.stride(3) == 1
                    and key.stride(3) == 1
                    and value.stride(3) == 1
                    # The fused kernel is forward-only; SDPA owns anything that
                    # needs a backward.
                    and not (query.requires_grad or key.requires_grad
                             or value.requires_grad)):
                return _short_attn(query, key, value, softmax_scale, causal,
                                   (shape[1], shape[3], bool(causal),
                                    query.dtype))

        if self.fa_func is not None and attn_mask is None and query.dtype != torch.float32:
            out = self.fa_func(
                query, key, value,
                softmax_scale=softmax_scale,
                causal=causal,
            )
            if isinstance(out, tuple):
                out = out[0]
            return out

        # SDPA handles both the masked case and the plain causal/non-causal case.
        # Custom masks force is_causal=False; FlashAttn does not support arbitrary masks.
        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)
        if self.use_flex_kernel:
            # FlexAttention generates a fused Triton fwd+bwd kernel autotuned
            # for the exact (B, H, S_q, S_kv, D) shape and the user-provided
            # mask. ``attn_mask`` here is repurposed to accept a
            # ``BlockMask`` (from ``create_block_mask``) instead of a dense
            # bool tensor. On B200 with chunked-suffix shapes
            # (Q=1024, KV=9216, D=64), the fused fwd+bwd is ~1.37x faster
            # than cuDNN flash with the equivalent dense mask
            # (microbenched). Same numerical agreement vs the fp32 MATH
            # reference (~1e-2 max-abs-diff in bf16, identical to cuDNN).
            q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
            out = self._flex_fn(
                q, k, v,
                block_mask=attn_mask,
                scale=softmax_scale,
            )
        elif self.use_cudnn_kernel:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            # An explicit mask plus is_causal=True is ambiguous, and the two code
            # paths here would resolve it differently: this branch would hand both
            # to SDPA (which applies the causal mask *on top of* attn_mask), while
            # the non-cuDNN branch below drops is_causal and treats attn_mask as
            # authoritative. SDPA itself accepts the combination on this backend
            # rather than rejecting it, so nothing would surface the disagreement
            # -- reject it here instead of silently masking twice.
            if attn_mask is not None and causal:
                raise ValueError(
                    "DenseAttention: pass either attn_mask or causal=True, not both "
                    "(an explicit mask must already encode causality). Got "
                    f"attn_mask={tuple(attn_mask.shape)} with causal=True."
                )
            # The sdpa_kernel context below FORCES cuDNN, and on Blackwell (sm100,
            # cuDNN 9.19) the cuDNN flash kernel accepts the permuted, non-contiguous
            # q/k/v views directly -- so we skip the q/k/v .contiguous() clones (they
            # were a real cost: 3 clones/block x54 blocks). Verified bit-identical and
            # faster; if cuDNN ever rejects a layout it raises -> MATH fallback below.
            if attn_mask is not None and not attn_mask.is_contiguous():
                attn_mask = attn_mask.contiguous()
            # Try strict cuDNN first. Adding MATH as a fallback in the
            # ``sdpa_kernel`` list causes PyTorch's selection heuristic to
            # pick MATH over cuDNN (~10× slower) for inputs both can
            # handle. If cuDNN rejects (e.g. head_dim=16, fp32, or some
            # mask shape it doesn't support), fall back through MATH.
            #
            # head_dim > 128 is rejected by cuDNN unconditionally ("head_dim
            # should be no more than 128"), so route it straight to the backends
            # that can serve it. The try/except below only recovers in eager --
            # under torch.compile the RuntimeError surfaces during fake-tensor
            # tracing and aborts the whole graph rather than taking the handler,
            # which is how a head_dim=256 model (Gemma-2B in Pi0) failed to
            # compile at all.
            if q.shape[-1] > _CUDNN_MAX_HEAD_DIM:
                # EFFICIENT_ATTENTION requires an additive bias in the query's
                # dtype ("invalid dtype for bias - should match query's dtype");
                # cuDNN tolerated an fp32 mask against bf16 q/k/v. A bool mask is
                # passed through -- coercing it would turn True/False into a
                # 1.0/0.0 additive bias.
                if attn_mask is not None and attn_mask.dtype not in (torch.bool, q.dtype):
                    attn_mask = attn_mask.to(dtype=q.dtype)
                with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                    out = F.scaled_dot_product_attention(
                        q, k, v,
                        attn_mask=attn_mask,
                        dropout_p=0.0,
                        is_causal=causal,
                        scale=softmax_scale,
                    )
            else:
                try:
                    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                        out = F.scaled_dot_product_attention(
                            q, k, v,
                            attn_mask=attn_mask,
                            dropout_p=0.0,
                            is_causal=causal,
                            scale=softmax_scale,
                        )
                except RuntimeError:
                    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                        out = F.scaled_dot_product_attention(
                            q, k, v,
                            attn_mask=attn_mask,
                            dropout_p=0.0,
                            is_causal=causal,
                            scale=softmax_scale,
                        )
        else:
            # SDPA accepts a boolean mask (True = attend) directly; only a float
            # (additive) mask needs dtype coercion. Coercing a bool mask to q.dtype
            # would turn True/False into a 1.0/0.0 additive bias (wrong semantics) --
            # e.g. the HunyuanVideo key-padding mask would then fail to mask padding
            # on non-cuDNN backends.
            if attn_mask is not None and attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.to(dtype=q.dtype)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False if attn_mask is not None else causal,
                scale=softmax_scale,
            )
        return out.permute(0, 2, 1, 3)
