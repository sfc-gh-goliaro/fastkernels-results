"""Kimi-Linear decoder layer: fused KDA core + whole-layer CUDA graph.

Two things dominate this layer at the shapes the model actually runs, and
neither is arithmetic.

1. Launch overhead. At 1-64 tokens the whole layer is ~200 us of device work
   behind ~2.3 ms of wall clock: ~60 kernels, a third of them Triton (the FLA
   chunk/gate/l2norm pipeline) whose Python launch path costs tens of
   microseconds each, plus one hard device sync per call from the boolean-mask
   select that zeroes non-initial recurrent states. Fixed by recording the
   whole layer into a CUDA graph per (num_tokens, has_residual) shape: replay
   issues the identical kernel sequence with identical arguments, so results
   are bit-for-bit the eager ones and only the host-side work disappears.

2. The KDA core. The reference prefill path is ~15 launches -- three varlen
   depthwise convs, two L2 norms, a fused gate+cumsum, and the six-kernel FLA
   chunk pipeline. Its gated ``k k^T`` cannot be a plain matmul (the decay is
   per-channel), so it is scalar work even when the sequence is long. Replaced
   by a Triton pre-kernel (conv+silu+L2 norm+gate+beta in one pass over the
   projections) plus a CUDA gated-delta-rule scan that keeps the recurrent
   state in registers for the whole sequence. Long sequences stay on the
   reference path -- not for speed but for numerics; see ``_FastKDA``.

A third, smaller win: several kernels in the layer are independent (the MoE's
shared expert, KDA's output-gate projection and conv-state roll), so they are
captured on side streams and run as parallel graph nodes.

Everything unexpected -- a non-KDA layer, a batch with decodes, an unseen head
dim, a capture or JIT-build failure, or a fused result that does not match the
reference on the first call -- falls back to the reference path.
"""

from __future__ import annotations

import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from ..L1.rms_norm import RMSNorm
from ..L2.kimi_delta_attention import KimiDeltaAttention
from ..L2.kimi_mla_attention import KimiMLAAttention
from ..L2.kimi_moe import KimiMoE
from ..L2.llama_mlp import LlamaMLP
from ....infra.context import get_context

# Longest sequence that takes the fused path. Bounded by numerics, not speed:
# see the note in ``_FastKDA``. Covers every short shape this model decodes at.
_SCAN_MAX_T = int(os.environ.get("FK_KDA_SCAN_MAX_T", "96"))
# ---------------------------------------------------------------------------
# Fused KDA core: two kernels in place of the reference's ~15.
#
#   * ``_kda_pre_kernel``: conv+silu for q/k/v, the L2 norms, the gate, and the
#     beta sigmoid, in one pass over the projections. One program per (token
#     block, head) -- a head owns exactly the 128 channels the L2 norm reduces
#     over, so the norm folds in instead of costing its own pass.
#   * ``kda_scan`` (CUDA): the gated delta rule as a register-resident scan.
#     Thread (v, k-group) holds S[v, kslice] in registers for the whole
#     sequence, so both ``S k`` and ``S q`` reduce inside the thread apart from
#     one KG-lane shuffle, and the state moves once per call rather than once
#     per chunk. Triton cannot express that layout -- its tiles put the
#     reduction axis across lanes, costing a cross-lane reduction per token,
#     which measured ~15x slower here.
# ---------------------------------------------------------------------------

_KDA_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>

#define HDIM 128

template <int BV, int KG, int TILE>
__global__ __launch_bounds__(BV * KG) void kda_scan_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const float* __restrict__ EXPG,
    const float* __restrict__ BETA,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ STATE,
    const int* __restrict__ SIDX,
    const bool* __restrict__ HASINIT,
    int T, int H, float scale) {
  constexpr int BK = HDIM / KG;
  constexpr int NT = BV * KG;
  const int nv = HDIM / BV;
  const int i_h = blockIdx.x / nv;
  const int i_v = blockIdx.x % nv;
  const int tid = threadIdx.x;
  const int vl = tid / KG;
  const int kg = tid % KG;
  const int P = H * HDIM;
  const int hbase = i_h * HDIM;
  const int v0 = i_v * BV;

  __shared__ float sq[TILE][HDIM];
  __shared__ float sk[TILE][HDIM];
  __shared__ float sg[TILE][HDIM];
  __shared__ float sv[TILE][BV];
  __shared__ float sb[TILE];

  const int sidx = SIDX[0];
  const bool hasinit = HASINIT[0];
  float* st = STATE + ((long)sidx * H + i_h) * (HDIM * HDIM)
              + (long)(v0 + vl) * HDIM + kg * BK;

  float s[BK];
  if (hasinit) {
#pragma unroll
    for (int i = 0; i < BK; ++i) s[i] = st[i];
  } else {
#pragma unroll
    for (int i = 0; i < BK; ++i) s[i] = 0.f;
  }

  // Lanes of one v row are adjacent, so the k-group reduction is a shuffle
  // inside a KG-wide segment of the warp.
  const unsigned rmask = ((1u << KG) - 1u) << (((tid & 31) / KG) * KG);

  for (int t0 = 0; t0 < T; t0 += TILE) {
    const int n = min(TILE, T - t0);
    __syncthreads();
    for (int idx = tid; idx < n * HDIM; idx += NT) {
      const int tt = idx / HDIM, dd = idx % HDIM;
      const long off = (long)(t0 + tt) * P + hbase + dd;
      sq[tt][dd] = __bfloat162float(Q[off]);
      sk[tt][dd] = __bfloat162float(K[off]);
      sg[tt][dd] = EXPG[off];
    }
    for (int idx = tid; idx < n * BV; idx += NT) {
      const int tt = idx / BV, dd = idx % BV;
      sv[tt][dd] = __bfloat162float(V[(long)(t0 + tt) * P + hbase + v0 + dd]);
    }
    if (tid < n) sb[tid] = BETA[(long)(t0 + tid) * H + i_h];
    __syncthreads();

    for (int tt = 0; tt < n; ++tt) {
      const float4* kk = reinterpret_cast<const float4*>(&sk[tt][kg * BK]);
      const float4* gg = reinterpret_cast<const float4*>(&sg[tt][kg * BK]);
      const float4* qq = reinterpret_cast<const float4*>(&sq[tt][kg * BK]);
      float t1 = 0.f;
#pragma unroll
      for (int i = 0; i < BK / 4; ++i) {
        float4 g4 = gg[i], k4 = kk[i];
        s[4 * i + 0] *= g4.x; t1 += s[4 * i + 0] * k4.x;
        s[4 * i + 1] *= g4.y; t1 += s[4 * i + 1] * k4.y;
        s[4 * i + 2] *= g4.z; t1 += s[4 * i + 2] * k4.z;
        s[4 * i + 3] *= g4.w; t1 += s[4 * i + 3] * k4.w;
      }
#pragma unroll
      for (int m = 1; m < KG; m <<= 1) t1 += __shfl_xor_sync(rmask, t1, m);
      const float u = (sv[tt][vl] - t1) * sb[tt];
      float acc = 0.f;
#pragma unroll
      for (int i = 0; i < BK / 4; ++i) {
        float4 k4 = kk[i], q4 = qq[i];
        s[4 * i + 0] += u * k4.x; acc += s[4 * i + 0] * q4.x;
        s[4 * i + 1] += u * k4.y; acc += s[4 * i + 1] * q4.y;
        s[4 * i + 2] += u * k4.z; acc += s[4 * i + 2] * q4.z;
        s[4 * i + 3] += u * k4.w; acc += s[4 * i + 3] * q4.w;
      }
#pragma unroll
      for (int m = 1; m < KG; m <<= 1) acc += __shfl_xor_sync(rmask, acc, m);
      if (kg == 0)
        O[(long)(t0 + tt) * P + hbase + v0 + vl] = __float2bfloat16(acc * scale);
    }
  }
#pragma unroll
  for (int i = 0; i < BK; ++i) st[i] = s[i];
}

#define FK_LAUNCH(BV, KG, TILE)                                               \
  kda_scan_kernel<BV, KG, TILE><<<H*(HDIM/BV), BV*KG, 0, stream>>>(           \
      (const __nv_bfloat16*)q.data_ptr(), (const __nv_bfloat16*)k.data_ptr(), \
      (const __nv_bfloat16*)v.data_ptr(), expg.data_ptr<float>(),             \
      beta.data_ptr<float>(), (__nv_bfloat16*)o.data_ptr(),                   \
      state.data_ptr<float>(), sidx.data_ptr<int>(),                          \
      (const bool*)hasinit.data_ptr(), T, H, scale)

void kda_scan(torch::Tensor q, torch::Tensor k, torch::Tensor v,
              torch::Tensor expg, torch::Tensor beta, torch::Tensor o,
              torch::Tensor state, torch::Tensor sidx, torch::Tensor hasinit,
              int64_t H, double scale_, int64_t bv, int64_t kg, int64_t tile) {
  const int T = q.size(0);
  const float scale = (float)scale_;
  auto stream = c10::cuda::getCurrentCUDAStream();
  // Only the shapes _scan_cfg can ask for: TILE x 3 x 128 floats of staging
  // has to stay under the 48 KB static shared limit.
  if (bv == 8 && kg == 16 && tile == 16) { FK_LAUNCH(8, 16, 16); }
  else if (bv == 16 && kg == 8 && tile == 16) { FK_LAUNCH(16, 8, 16); }
  else if (bv == 32 && kg == 4 && tile == 16) { FK_LAUNCH(32, 4, 16); }
  else { TORCH_CHECK(false, "unsupported kda_scan config"); }
}

TORCH_LIBRARY_FRAGMENT(fk_kda, m) {
  m.def("kda_scan(Tensor q, Tensor k, Tensor v, Tensor expg, Tensor beta, "
        "Tensor(f!) o, Tensor(g!) state, Tensor sidx, Tensor hasinit, "
        "int H, float scale, int bv, int kg, int tile) -> ()");
}
TORCH_LIBRARY_IMPL(fk_kda, CUDA, m) { m.impl("kda_scan", kda_scan); }
"""

_EXT_STATE: list = [None]


def _kda_scan_op():
    """JIT-build (once, cached on disk) and return the CUDA scan op.

    Returns None if the build is unavailable, in which case the caller uses the
    Triton scan below -- slower (Triton cannot put the reduction axis inside a
    thread) but still far ahead of the reference chunk pipeline at these lengths.
    """
    st = _EXT_STATE[0]
    if st is None:
        try:
            from torch.utils.cpp_extension import load_inline
            load_inline(
                name="fk_kda_scan_ext",
                cpp_sources="",
                cuda_sources=_KDA_CUDA_SRC,
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                is_python_module=False,
                verbose=False,
            )
            st = torch.ops.fk_kda.kda_scan
        except Exception:
            st = False
        _EXT_STATE[0] = st
    return st or None


@triton.jit
def _kda_scan_triton_kernel(
    Q, K, V, EXPG, BETA, O, STATE, SIDX, HASINIT, T, SCALE,
    P: tl.constexpr, H: tl.constexpr, D: tl.constexpr, BV: tl.constexpr,
):
    """Portable fallback for the CUDA scan (same recurrence, tiled state)."""
    pid = tl.program_id(0)
    NV: tl.constexpr = D // BV
    i_h = pid // NV
    i_v = pid % NV
    ok = tl.arange(0, D)
    ov = i_v * BV + tl.arange(0, BV)
    sidx = tl.load(SIDX).to(tl.int64)
    hinit = tl.load(HASINIT).to(tl.int1)
    p_st = STATE + sidx * (H * D * D) + i_h * (D * D) + ov[:, None] * D + ok[None, :]
    b_h = tl.where(hinit, tl.load(p_st), tl.zeros((BV, D), dtype=tl.float32))

    p_q = Q + i_h * D + ok
    p_k = K + i_h * D + ok
    p_v = V + i_h * D + ov
    p_g = EXPG + i_h * D + ok
    p_b = BETA + i_h
    p_o = O + i_h * D + ov
    for _ in range(0, T):
        b_q = tl.load(p_q).to(tl.float32) * SCALE
        b_k = tl.load(p_k).to(tl.float32)
        b_v = tl.load(p_v).to(tl.float32)
        b_h *= tl.load(p_g)[None, :]
        b_u = (b_v - tl.sum(b_h * b_k[None, :], 1)) * tl.load(p_b).to(tl.float32)
        b_h += b_u[:, None] * b_k[None, :]
        tl.store(p_o, tl.sum(b_h * b_q[None, :], 1).to(p_o.dtype.element_ty))
        p_q += P
        p_k += P
        p_v += P
        p_g += P
        p_b += H
        p_o += P
    tl.store(p_st, b_h)


def _kda_scan(q, k, v, expg, beta, core, rec, sidx, hasinit, H, D, scale, T):
    op = _kda_scan_op()
    if op is not None:
        bv, kg, tile = _scan_cfg(T)
        op(q, k, v, expg, beta, core, rec, sidx, hasinit, H, scale, bv, kg, tile)
        return
    _kda_scan_triton_kernel[(H * (D // 16),)](
        q, k, v, expg, beta, core, rec, sidx, hasinit, T, scale,
        P=H * D, H=H, D=D, BV=16, num_warps=4,
    )


@triton.jit
def _kda_pre_kernel(
    QKVB, RAWG, WQ, WK, WV, CSQ, CSK, CSV, ALOG, DTB,
    Q, K, V, EXPG, BETA, SIDX, HASINIT, T,
    S_QKVB: tl.constexpr, P: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    W: tl.constexpr, BT: tl.constexpr, EPS: tl.constexpr, THRESH: tl.constexpr,
):
    """Depthwise conv + silu on q/k/v, L2 norm on q/k, KDA gate, beta sigmoid."""
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    rows = i_t * BT + tl.arange(0, BT)
    cols = i_h * D + tl.arange(0, D)
    rmask = rows < T
    sidx = tl.load(SIDX).to(tl.int64)
    hinit = tl.load(HASINIT).to(tl.int1)
    cs_off = sidx * ((W - 1) * P) + cols[None, :]

    accq = tl.zeros((BT, D), dtype=tl.float32)
    acck = tl.zeros((BT, D), dtype=tl.float32)
    accv = tl.zeros((BT, D), dtype=tl.float32)
    for s in tl.static_range(W):
        ridx = rows - (W - 1) + s
        m_in = ((ridx >= 0) & (ridx < T))[:, None]
        base = QKVB + ridx[:, None] * S_QKVB + cols[None, :]
        xq = tl.load(base, mask=m_in, other=0.0).to(tl.float32)
        xk = tl.load(base + P, mask=m_in, other=0.0).to(tl.float32)
        xv = tl.load(base + 2 * P, mask=m_in, other=0.0).to(tl.float32)
        # Taps before the sequence come from the cached conv state (oldest first);
        # the two masks are disjoint so adding is a select.
        m_h = ((ridx < 0) & hinit)[:, None]
        hoff = cs_off + (ridx + (W - 1))[:, None] * P
        xq += tl.load(CSQ + hoff, mask=m_h, other=0.0).to(tl.float32)
        xk += tl.load(CSK + hoff, mask=m_h, other=0.0).to(tl.float32)
        xv += tl.load(CSV + hoff, mask=m_h, other=0.0).to(tl.float32)
        accq += xq * tl.load(WQ + cols * W + s)[None, :]
        acck += xk * tl.load(WK + cols * W + s)[None, :]
        accv += xv * tl.load(WV + cols * W + s)[None, :]

    # The reference writes the conv output to bf16 before the L2 norm reads it,
    # so round first or the norm sees more precision than it should.
    yq = (accq / (1.0 + tl.exp(-accq))).to(tl.bfloat16).to(tl.float32)
    yk = (acck / (1.0 + tl.exp(-acck))).to(tl.bfloat16).to(tl.float32)
    yv = (accv / (1.0 + tl.exp(-accv))).to(tl.bfloat16)
    rq = tl.rsqrt(tl.sum(yq * yq, axis=1) + EPS)
    rk = tl.rsqrt(tl.sum(yk * yk, axis=1) + EPS)
    ooff = rows[:, None] * P + cols[None, :]
    tl.store(Q + ooff, (yq * rq[:, None]).to(tl.bfloat16), mask=rmask[:, None])
    tl.store(K + ooff, (yk * rk[:, None]).to(tl.bfloat16), mask=rmask[:, None])
    tl.store(V + ooff, yv, mask=rmask[:, None])

    b_g = tl.load(RAWG + ooff, mask=rmask[:, None], other=0.0).to(tl.float32)
    b_g += tl.load(DTB + cols)[None, :]
    b_sp = tl.where(b_g > THRESH, b_g, tl.log(1.0 + tl.exp(b_g)))
    b_a = -tl.exp(tl.load(ALOG + i_h).to(tl.float32))
    tl.store(EXPG + ooff, tl.exp(b_a * b_sp), mask=rmask[:, None])

    b_b = tl.load(QKVB + rows * S_QKVB + (3 * P + i_h), mask=rmask,
                  other=0.0).to(tl.float32)
    tl.store(BETA + rows * H + i_h, 1.0 / (1.0 + tl.exp(-b_b)), mask=rmask)


@triton.jit
def _kda_conv_state_kernel(
    QKVB, CSQ, CSK, CSV, SIDX, HASINIT, T,
    S_QKVB: tl.constexpr, P: tl.constexpr, W: tl.constexpr, BLOCK: tl.constexpr,
):
    """Roll the last W-1 projection rows into the cached conv state."""
    c = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = c < P
    sidx = tl.load(SIDX).to(tl.int64)
    hinit = tl.load(HASINIT).to(tl.int1)
    base = sidx * ((W - 1) * P) + c
    for j in tl.static_range(W - 1):
        ridx = T - (W - 1) + j
        m_x = m & (ridx >= 0)
        # A sequence shorter than the window keeps the tail of the old state.
        m_o = m & (ridx < 0) & hinit
        src = base + (j + T) * P
        dst = base + j * P
        xq = tl.load(QKVB + ridx * S_QKVB + c, mask=m_x, other=0.0)
        xk = tl.load(QKVB + ridx * S_QKVB + P + c, mask=m_x, other=0.0)
        xv = tl.load(QKVB + ridx * S_QKVB + 2 * P + c, mask=m_x, other=0.0)
        oq = tl.load(CSQ + src, mask=m_o, other=0.0)
        okk = tl.load(CSK + src, mask=m_o, other=0.0)
        ov = tl.load(CSV + src, mask=m_o, other=0.0)
        tl.store(CSQ + dst, tl.where(ridx >= 0, xq, oq), mask=m)
        tl.store(CSK + dst, tl.where(ridx >= 0, xk, okk), mask=m)
        tl.store(CSV + dst, tl.where(ridx >= 0, xv, ov), mask=m)


def _pre_block(t: int) -> int:
    """Token tile for the pre-kernel. Small: it carries three fp32 [BT, 128]
    conv accumulators, so wider tiles spill."""
    if t <= 4:
        return 4
    return 8


def _scan_cfg(t: int) -> tuple[int, int, int]:
    """(BV, KG, TILE) for the scan. Narrow value blocks win at every length
    measured: they trade redundant q/k/gate reads for 16x the blocks, and this
    kernel is occupancy-bound, not bandwidth-bound."""
    del t
    return 8, 16, 16


class _FastKDA(KimiDeltaAttention):
    """KDA with the fused front end + register scan for short sequences.

    Subclasses the reference layer so the parameter tree, the state-manager prep
    (which finds the attention module by type) and the fallback are unchanged.

    Only short sequences take the fused path. Not because the scan gets slow --
    it is ~1.1 us/token at any length -- but because the reference *chunk*
    pipeline inverts a 64x64 unit-triangular matrix per chunk (``solve_tril``),
    and with enough chunks that inverse amplifies a 1-ulp difference in q/k into
    a visible one. The scan is the numerically stable formulation of the same
    recurrence, which is exactly why it cannot reproduce the reference bit for
    bit over many chunks. Below ``_SCAN_MAX_T`` there are at most a couple of
    chunks and the two agree to ~1e-3 relative; above it we hand the whole layer
    back to the reference.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # q/k/v/b and the two gate down-projections all read the same
        # hidden_states, so they become one GEMM instead of three launches.
        # Built lazily: weights are only loaded after __init__.
        self._in_weight: torch.Tensor | None = None
        self._verified = False
        self._fast_ok = (
            self.qkvb_proj is not None
            and self.head_dim == 128
            and self.conv_size == 4
            and self.local_num_heads * self.head_dim
            == self.q_conv1d.weight.shape[0]
        )

    def _fused_in_weight(self) -> torch.Tensor:
        w = self._in_weight
        if w is None:
            w = torch.cat(
                (self.qkvb_proj.weight, self.f_a_proj.weight, self.g_a_proj.weight),
                dim=0,
            ).contiguous()
            self._in_weight = w
        return w

    def _fast_meta(self, hidden_states: torch.Tensor):
        """Metadata + state for the fused path, or None to use the reference."""
        if (not self._fast_ok or hidden_states.dtype != torch.bfloat16
                or not hidden_states.is_cuda or hidden_states.dim() != 2):
            return None
        T = hidden_states.shape[0]
        if not 1 <= T <= _SCAN_MAX_T or not hidden_states.is_contiguous():
            return None
        ctx = get_context()
        state = getattr(ctx, "kda_state", None)
        meta = getattr(ctx, "kda_metadata", None)
        if state is None or meta is None:
            return None
        if (meta.num_prefills != 1 or meta.num_decodes != 0
                or meta.num_actual_tokens != T):
            return None
        idx = meta.state_indices
        if idx is None or idx.numel() != 1 or meta.has_initial_state is None:
            return None
        if state.q_conv_states[self.layer_idx] is None:
            return None
        return meta, state

    def _recurrent_buffers(self, state):
        li = self.layer_idx
        return (state.q_conv_states[li], state.k_conv_states[li],
                state.v_conv_states[li], state.recurrent_states[li])

    def _verify(self, hidden_states, meta, state) -> None:
        """One-time check of the fused core against the reference, on real data.

        The reference chunk pipeline's triangular inverse can amplify rounding
        for some weight draws; when it does, no reformulation of the recurrence
        will track it, so measure once and step aside if the two disagree.
        """
        self._verified = True
        bufs = [(t, t.clone()) for t in self._recurrent_buffers(state)
                if torch.is_tensor(t)]
        try:
            ref = KimiDeltaAttention.forward(self, hidden_states.clone())
            for live, saved in bufs:
                live.copy_(saved)
            got = self._forward_fused(hidden_states.clone(), meta, state)
            for live, saved in bufs:
                live.copy_(saved)
            rel = (torch.linalg.vector_norm((got - ref).float())
                   / torch.linalg.vector_norm(ref.float()).clamp_min(1e-30))
            if not (rel.item() <= 1e-2):
                self._fast_ok = False
        except Exception:
            self._fast_ok = False

    def _forward_fused(self, hidden_states, meta, state):
        T = hidden_states.shape[0]
        H, D, W = self.local_num_heads, self.head_dim, self.conv_size
        P = H * D
        dev = hidden_states.device
        csq, csk, csv, rec = self._recurrent_buffers(state)
        sidx, hasinit = meta.state_indices, meta.has_initial_state

        big = F.linear(hidden_states, self._fused_in_weight())
        # g_b only feeds the output gate, so it is off the scan's critical path.
        cur = torch.cuda.current_stream(dev)
        side = _side_stream(dev, 0)
        side.wait_stream(cur)
        with torch.cuda.stream(side):
            g2 = self.g_b_proj(big[:, 3 * P + H + D:])
        raw_g = self.f_b_proj(big[:, 3 * P + H:3 * P + H + D])

        q = torch.empty((T, P), device=dev, dtype=torch.bfloat16)
        k = torch.empty((T, P), device=dev, dtype=torch.bfloat16)
        v = torch.empty((T, P), device=dev, dtype=torch.bfloat16)
        expg = torch.empty((T, P), device=dev, dtype=torch.float32)
        beta = torch.empty((T, H), device=dev, dtype=torch.float32)
        core = torch.empty((T, P), device=dev, dtype=torch.bfloat16)
        bt = _pre_block(T)
        _kda_pre_kernel[(-(-T // bt), H)](
            big, raw_g,
            self.q_conv1d.weight.view(P, W), self.k_conv1d.weight.view(P, W),
            self.v_conv1d.weight.view(P, W),
            csq, csk, csv, self.A_log.view(-1), self.dt_bias.view(-1),
            q, k, v, expg, beta, sidx, hasinit, T,
            S_QKVB=big.shape[1], P=P, H=H, D=D, W=W, BT=bt,
            EPS=1e-6, THRESH=20.0, num_warps=4,
        )
        # The roll has to follow the pre-kernel (which reads the old state) but
        # nothing downstream needs it, so it rides the side stream.
        side.wait_stream(cur)
        with torch.cuda.stream(side):
            _kda_conv_state_kernel[(-(-P // 256),)](
                big, csq, csk, csv, sidx, hasinit, T,
                S_QKVB=big.shape[1], P=P, W=W, BLOCK=256, num_warps=4,
            )
        _kda_scan(q, k, v, expg, beta, core, rec, sidx, hasinit,
                  H, D, D ** -0.5, T)

        cur.wait_stream(side)
        g2.record_stream(cur)
        core = self.o_norm(core.view(1, T, H, D), g2.view(T, H, D))
        return self.o_proj(core.view(T, P))

    def forward(self, hidden_states, state_manager=None):
        fast = self._fast_meta(hidden_states)
        if fast is not None and not self._verified:
            # Verification needs a host sync, so it cannot happen mid-capture;
            # the reference path covers that (never reached in practice -- graph
            # capture is always preceded by eager warm-up).
            if torch.cuda.is_current_stream_capturing():
                fast = None
            else:
                self._verify(hidden_states, *fast)
                if not self._fast_ok:
                    fast = None
        if fast is None:
            return super().forward(hidden_states, state_manager=state_manager)
        del state_manager
        self._ensure_triton_allocator(hidden_states.device)
        return self._forward_fused(hidden_states, *fast)


# ---------------------------------------------------------------------------
# Intra-layer overlap
#
# Inside a graph the layer is one long serial stream of ~20 kernels, most of
# them a couple of microseconds of weight streaming that neither fills the
# machine nor saturates HBM -- they are latency, not throughput. Several are
# independent: the MoE's shared expert does not depend on the router, and KDA's
# ``g_b`` projection and conv-state roll do not feed the scan. Capturing those
# on side streams turns them into parallel graph nodes, which is free
# concurrency at replay: same kernels, same arguments, bit-identical results.
# ---------------------------------------------------------------------------

_SIDE_STREAMS: dict[tuple, "torch.cuda.Stream"] = {}


def _side_stream(device, slot: int = 0):
    key = (torch.cuda.current_device() if device is None else device.index, slot)
    s = _SIDE_STREAMS.get(key)
    if s is None:
        s = torch.cuda.Stream(device=device)
        _SIDE_STREAMS[key] = s
    return s


class _FastMoE(KimiMoE):
    """KimiMoE with the shared expert issued alongside the routed path.

    trtllm-gen's MoE deliberately leaves SMs idle ("140 SMs used for MoE, 8
    reserved for overlapping kernels"), and the router GEMV ahead of it is pure
    latency, so the shared expert costs almost nothing once it is off the
    critical path. Installed by re-binding ``__class__`` so the parameter tree
    and ``state_dict`` are unchanged.
    """

    def forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.use_trtllm or self.shared_experts is None:
            return super().forward_impl(hidden_states)
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)

        cur = torch.cuda.current_stream()
        side = _side_stream(hidden_states.device, 1)
        side.wait_stream(cur)
        with torch.cuda.stream(side):
            shared_output = self.shared_experts(hidden_states)

        router_logits = self.gate_linear(
            hidden_states, self.gate.weight, out_dtype=torch.float32,
        )
        out = self.trtllm_moe(
            hidden_states, self.w13, self.w2, router_logits,
            routing_bias=self.gate.e_score_correction_bias,
        )

        cur.wait_stream(side)
        shared_output.record_stream(cur)
        out = out + shared_output
        if self.tp_size > 1 and not self._use_custom_op:
            out = self.allreduce(out)
        return out.view(orig_shape)


def _upgrade_moe(module: nn.Module) -> None:
    """Re-bind every plain ``KimiMoE`` in *module* to the overlapped variant."""
    for sub in module.modules():
        if type(sub) is KimiMoE:
            sub.__class__ = _FastMoE


# ---------------------------------------------------------------------------
# Whole-layer CUDA graph
#
# The caller rebuilds the recurrent metadata on the global Context before every
# forward, so its tensor addresses change between calls while a graph bakes them
# in. We capture against a layer-owned copy and re-sync it whenever the Context
# hands us a new metadata object. The recurrent state buffers live on the state
# manager and are likewise read by address, so a graph is only replayed while
# the manager it was captured against is still live (identity + data_ptr).
# ---------------------------------------------------------------------------

_META_TENSORS = (
    "query_start_loc", "seq_lens", "state_indices", "has_initial_state",
    "query_start_loc_int32", "state_indices_long", "slot_mapping",
    "block_tables", "batch_ptr", "token_chunk_offset_ptr",
)

_MAX_GRAPHS = 8
_WARMUP_ITERS = 3


class _Replay:
    """A captured graph plus the buffers its replay reads and writes."""

    __slots__ = ("graph", "in_hs", "in_res", "out_hs", "out_res",
                 "meta", "meta_src", "state_obj", "state_ptrs")

    def __init__(self, graph, in_hs, in_res, out_hs, out_res, meta, meta_src,
                 state_obj, state_ptrs):
        self.graph = graph
        self.in_hs = in_hs
        self.in_res = in_res
        self.out_hs = out_hs
        self.out_res = out_res
        self.meta = meta
        self.meta_src = meta_src
        self.state_obj = state_obj
        self.state_ptrs = state_ptrs


def _state_ptrs(state, layer_idx):
    """Addresses of the per-layer recurrent buffers a graph closes over."""
    ptrs = []
    for name in ("q_conv_states", "k_conv_states", "v_conv_states",
                 "recurrent_states", "gdn_conv", "recurrent"):
        lst = getattr(state, name, None)
        t = lst[layer_idx] if lst is not None and layer_idx < len(lst) else None
        ptrs.append(t.data_ptr() if torch.is_tensor(t) else 0)
    return tuple(ptrs)


def _clone_metadata(meta):
    """A layer-owned copy of the recurrent metadata with stable addresses.

    The varlen conv kernel reads ``batch_ptr`` / ``token_chunk_offset_ptr``
    through ``nums_dict``, aliased to the top-level fields, so the clone has to
    preserve that aliasing.
    """
    owned = copy.copy(meta)
    for field in _META_TENSORS:
        t = getattr(owned, field, None)
        if torch.is_tensor(t):
            setattr(owned, field, t.clone())
    nums_dict = getattr(meta, "nums_dict", None)
    if isinstance(nums_dict, dict):
        aliases = (
            ("batch_ptr", getattr(meta, "batch_ptr", None), owned.batch_ptr),
            ("token_chunk_offset_ptr", getattr(meta, "token_chunk_offset_ptr", None),
             owned.token_chunk_offset_ptr),
        )
        owned.nums_dict = {}
        for block_m, entry in nums_dict.items():
            entry = dict(entry)
            for key, src_top, own_top in aliases:
                t = entry.get(key)
                if torch.is_tensor(t):
                    entry[key] = own_top if t is src_top else t.clone()
            owned.nums_dict[block_m] = entry
    return owned


def _sync_metadata(owned, src) -> bool:
    """Refresh ``owned``'s device tensors from ``src``; False if incompatible."""
    for field in _META_TENSORS:
        s = getattr(src, field, None)
        o = getattr(owned, field, None)
        if torch.is_tensor(s) != torch.is_tensor(o):
            return False
        if torch.is_tensor(s):
            if s.shape != o.shape or s.dtype != o.dtype:
                return False
            o.copy_(s)
    for field in ("num_actual_tokens", "num_prefills", "num_prefill_tokens",
                  "num_decodes", "num_decode_tokens", "max_query_len",
                  "max_seq_len"):
        if getattr(src, field, None) != getattr(owned, field, None):
            return False
    src_nums = getattr(src, "nums_dict", None)
    own_nums = getattr(owned, "nums_dict", None)
    if isinstance(src_nums, dict) and isinstance(own_nums, dict):
        if src_nums.keys() != own_nums.keys():
            return False
        for block_m, entry in src_nums.items():
            mine = own_nums[block_m]
            if entry.get("tot") != mine.get("tot"):
                return False
            for key in ("batch_ptr", "token_chunk_offset_ptr"):
                s, o = entry.get(key), mine.get(key)
                if torch.is_tensor(s) and torch.is_tensor(o):
                    if s.shape != o.shape:
                        return False
                    o.copy_(s)
    return True


class KimiLinearDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int, quant_config: dict | None = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_kda = config.is_kda_layer(layer_idx)

        if self.is_kda:
            self.self_attn = _FastKDA(
                config,
                layer_idx=layer_idx,
                quant_config=quant_config,
            )
        else:
            self.self_attn = KimiMLAAttention(
                config,
                quant_config=quant_config,
            )

        if config.is_moe_layer(layer_idx):
            self.block_sparse_moe = KimiMoE(config, quant_config=quant_config)
            self.mlp = self.block_sparse_moe
        else:
            self.mlp = LlamaMLP(config, quant_config=quant_config)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
        )

        _upgrade_moe(self)
        self._replays: dict[tuple, _Replay] = {}
        self._graph_ok = True

    # -- eager path ---------------------------------------------------------

    def _forward_eager(self, hidden_states, residual, state_manager=None):
        if residual is None:
            residual = hidden_states.clone()
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(hidden_states, state_manager=state_manager)
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual,
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    # -- graph capture ------------------------------------------------------

    def _capture(self, hidden_states, residual, state_manager, ctx):
        device = hidden_states.device
        src_meta = ctx.kda_metadata
        owned = _clone_metadata(src_meta)

        in_hs = torch.empty_like(hidden_states)
        in_res = None if residual is None else torch.empty_like(residual)

        def load():
            in_hs.copy_(hidden_states)
            if in_res is not None:
                in_res.copy_(residual)

        ctx.kda_metadata = owned
        try:
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for _ in range(_WARMUP_ITERS):
                    load()
                    self._forward_eager(in_hs, in_res, state_manager)
            torch.cuda.current_stream(device).wait_stream(stream)
            torch.cuda.synchronize(device)

            graph = torch.cuda.CUDAGraph()
            load()
            with torch.cuda.graph(graph):
                out_hs, out_res = self._forward_eager(
                    in_hs, in_res, state_manager,
                )
        finally:
            ctx.kda_metadata = src_meta

        return _Replay(
            graph, in_hs, in_res, out_hs, out_res, owned, src_meta,
            ctx.kda_state, _state_ptrs(ctx.kda_state, self.layer_idx),
        )

    def _lookup(self, hidden_states, residual, state_manager):
        """The replay for this shape, capturing it if needed (None => eager)."""
        ctx = get_context()
        state = getattr(ctx, "kda_state", None)
        meta = getattr(ctx, "kda_metadata", None)
        if state is None or meta is None or meta.num_prefills <= 0:
            return None

        key = (hidden_states.shape, hidden_states.dtype, hidden_states.device,
               residual is None)
        entry = self._replays.get(key)
        if entry is None:
            if len(self._replays) >= _MAX_GRAPHS:
                return None
            try:
                entry = self._capture(hidden_states, residual, state_manager, ctx)
            except Exception:
                self._graph_ok = False
                return None
            self._replays[key] = entry
            return entry

        if (entry.state_obj is not state
                or entry.state_ptrs != _state_ptrs(state, self.layer_idx)):
            return None
        if entry.meta_src is not meta:
            if not _sync_metadata(entry.meta, meta):
                return None
            entry.meta_src = meta
        return entry

    # -- dispatch -----------------------------------------------------------

    def forward(self, hidden_states, residual, state_manager=None):
        if (self._graph_ok and self.is_kda and hidden_states.is_cuda
                and not torch.compiler.is_compiling()
                and not torch.cuda.is_current_stream_capturing()):
            entry = self._lookup(hidden_states, residual, state_manager)
            if entry is not None:
                entry.in_hs.copy_(hidden_states)
                if entry.in_res is not None:
                    entry.in_res.copy_(residual)
                entry.graph.replay()
                return entry.out_hs, entry.out_res
        return self._forward_eager(hidden_states, residual, state_manager)
