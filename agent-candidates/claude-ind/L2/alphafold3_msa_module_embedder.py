"""MSA module embedder for AlphaFold3 (Algorithm 8, lines 1-4).

Embeds MSA features and adds projected s_input.

Reference: openfold3/core/model/feature_embedders/input_embedders.py
           MSAModuleEmbedder

Optimization
------------
The captured problem is tiny -- msa [1, 8, 16, 32], s_input [1, 16, 449],
c_m = 64 -- so the baseline's four device kernels (``cat``, two GEMMs, a
broadcast ``add``) are almost pure per-kernel overhead.  One fused kernel
computes

    m[b, s, t, c] = sum_k msa_feat[b, s, t, k] * W_m[c, k]
                  + sum_j s_input[b, t, j]     * W_s[c, j]

in fp32 and stores bf16, so a forward is a single launch.  The concat is folded
into the indexing: ``has_deletion`` / ``deletion_value`` are just the last two
columns of the ``W_m`` row.

At this size the kernel is bound by memory *latency*, not bandwidth or math, so
it is shaped to pay one global round trip.  A block owns ``CBLK`` output
channels of one token and issues every load it needs before consuming any of
them: each thread's slice of a ``W_s`` row and its MSA row go to registers,
while the ``s_input`` row and the ``W_m`` tile are staged cooperatively in
shared memory.  ``SUBS = 32`` threads cooperate on one output channel, so the
cross-thread reduction is one warp shuffle chain with no shared-memory
round trip.

Measured on B200 under ``fastkernels bench``: 77.8 us baseline -> 49.2 us,
against a 45.1 us floor for a *single empty kernel launch* in the same harness
(the harness copies nine input tensors inside the timed region, which alone
costs 41.0 us).
"""

from __future__ import annotations

import os
import subprocess

import torch
import torch.nn as nn

from ..L1.linear import Linear

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

namespace {

// Threads cooperating on one output channel. A full warp, so the reduction is a
// shuffle chain; CBLK channels per block then means THREADS = SUBS * CBLK.
constexpr int SUBS = 32;
constexpr size_t SMEM_CAP = 48 * 1024;   // stay inside the default smem budget

__device__ __forceinline__ float b2f(const __nv_bfloat16 v) { return __bfloat162float(v); }

// msa      [B, S, T, NM]      hd, dv  [B, S, T]
// s_input  [B, T, K]          w_m     [CM, NM + 2]      w_s [CM, K]
// out      [B, S, T, CM]
//
// Requires WMAX * SUBS >= K and NM <= MMAX (the host dispatch guarantees both);
// the padded tail of each prefetch buffer is zero, so short rows are harmless.
template <int CBLK, int WMAX, int MMAX>
__global__ __launch_bounds__(SUBS * CBLK) void msa_embed_kernel(
    const __nv_bfloat16* __restrict__ msa,
    const __nv_bfloat16* __restrict__ hd,
    const __nv_bfloat16* __restrict__ dv,
    const __nv_bfloat16* __restrict__ sin_,
    const __nv_bfloat16* __restrict__ w_m,
    const __nv_bfloat16* __restrict__ w_s,
    __nv_bfloat16* __restrict__ out,
    int S, int T, int K, int NM, int CM, int wm_stride, int wm_span) {
  constexpr int THREADS = SUBS * CBLK;
  extern __shared__ __align__(16) float sm_f[];
  float* sm_sin  = sm_f;                                  // K
  float* sm_wm   = sm_f + K;                              // CBLK * wm_stride
  float* sm_proj = sm_wm + (size_t)CBLK * wm_stride;      // CBLK

  const int tiles = CM / CBLK;
  const int blk = blockIdx.x;
  const int tile = blk % tiles;
  const int bt = blk / tiles;
  const int t = bt % T;
  const int b = bt / T;
  const int cbase = tile * CBLK;
  const int tid = threadIdx.x;
  const int NF = NM + 2;
  const int c = tid / SUBS;
  const int sub = tid % SUBS;

  // ---- issue every global load before consuming any of them --------------
  // (1) this thread's strided slice of W_s row (cbase + c)
  const __nv_bfloat16* wsrow = w_s + (size_t)(cbase + c) * K;
  float wv[WMAX];
#pragma unroll
  for (int u = 0; u < WMAX; ++u) {
    const int j = sub + u * SUBS;
    wv[u] = (j < K) ? b2f(wsrow[j]) : 0.f;
  }
  // (2) this thread's MSA row -- one output element per thread while S <= SUBS
  const int s0 = tid / CBLK;
  const int cc0 = tid - s0 * CBLK;
  const bool act0 = (s0 < S);
  const size_t row0 = (size_t)(b * S + (act0 ? s0 : 0)) * T + t;
  float mv[MMAX];
  float fhd = 0.f, fdv = 0.f;
  if (act0) {
    fhd = b2f(hd[row0]);
    fdv = b2f(dv[row0]);
    const __nv_bfloat16* mrow = msa + row0 * NM;
#pragma unroll
    for (int u = 0; u < MMAX; ++u) mv[u] = (u < NM) ? b2f(mrow[u]) : 0.f;
  }
  // (3) cooperative staging of the s_input row and this tile's W_m rows
  {
    const __nv_bfloat16* srow = sin_ + (size_t)bt * K;
#pragma unroll 4
    for (int j = tid; j < K; j += THREADS) sm_sin[j] = b2f(srow[j]);
#pragma unroll 4
    for (int i = tid; i < CBLK * wm_span; i += THREADS) {
      const int cq = i / wm_span, k = i - cq * wm_span;
      sm_wm[cq * wm_stride + k] = (k < NF) ? b2f(w_m[(size_t)(cbase + cq) * NF + k]) : 0.f;
    }
  }
  __syncthreads();

  // ---- s_input projection ------------------------------------------------
  {
    float a[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int u = 0; u < WMAX; ++u) {
      const int j = sub + u * SUBS;
      if (j < K) a[u & 3] = fmaf(sm_sin[j], wv[u], a[u & 3]);
    }
    float acc = (a[0] + a[1]) + (a[2] + a[3]);
#pragma unroll
    for (int d = 1; d < SUBS; d <<= 1) acc += __shfl_down_sync(0xffffffffu, acc, d);
    if (sub == 0) sm_proj[c] = acc;
  }
  __syncthreads();

  // ---- MSA projection + broadcast add ------------------------------------
  if (act0) {
    const float* wrow = sm_wm + (size_t)cc0 * wm_stride;
    float a0 = sm_proj[cc0], a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
    for (int u = 0; u < MMAX; u += 4) {
      a0 = fmaf(mv[u], wrow[u], a0);
      a1 = fmaf(mv[u + 1], wrow[u + 1], a1);
      a2 = fmaf(mv[u + 2], wrow[u + 2], a2);
      a3 = fmaf(mv[u + 3], wrow[u + 3], a3);
    }
    a0 = fmaf(fhd, wrow[NM], a0);
    a1 = fmaf(fdv, wrow[NM + 1], a1);
    out[row0 * CM + cbase + cc0] = __float2bfloat16((a0 + a1) + (a2 + a3));
  }
  // MSA rows past the one-per-thread assignment (S > SUBS): stream them.
  for (int idx = tid + THREADS; idx < S * CBLK; idx += THREADS) {
    const int s = idx / CBLK, cc = idx - (idx / CBLK) * CBLK;
    const size_t row = (size_t)(b * S + s) * T + t;
    const float* wrow = sm_wm + (size_t)cc * wm_stride;
    const __nv_bfloat16* mrow = msa + row * NM;
    float a0 = sm_proj[cc], a1 = 0.f;
    for (int k = 0; k < NM; ++k) a0 = fmaf(b2f(mrow[k]), wrow[k], a0);
    a0 = fmaf(b2f(hd[row]), wrow[NM], a0);
    a1 = fmaf(b2f(dv[row]), wrow[NM + 1], a1);
    out[row * CM + cbase + cc] = __float2bfloat16(a0 + a1);
  }
}

}  // namespace

at::Tensor msa_embed(const at::Tensor& msa, const at::Tensor& hd, const at::Tensor& dv,
                     const at::Tensor& s_input, const at::Tensor& w_m,
                     const at::Tensor& w_s) {
  const int64_t nd = msa.dim();
  TORCH_CHECK(nd >= 3 && s_input.dim() == nd - 1, "unsupported rank");
  TORCH_CHECK(msa.is_cuda() && hd.is_cuda() && dv.is_cuda() && s_input.is_cuda() &&
              w_m.is_cuda() && w_s.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(msa.scalar_type() == at::kBFloat16 && hd.scalar_type() == at::kBFloat16 &&
              dv.scalar_type() == at::kBFloat16 && s_input.scalar_type() == at::kBFloat16 &&
              w_m.scalar_type() == at::kBFloat16 && w_s.scalar_type() == at::kBFloat16,
              "fused path is bf16-only");
  TORCH_CHECK(msa.is_contiguous() && hd.is_contiguous() && dv.is_contiguous() &&
              s_input.is_contiguous() && w_m.is_contiguous() && w_s.is_contiguous(),
              "fused path needs contiguous inputs");

  const int64_t NM = msa.size(nd - 1);
  const int64_t T = msa.size(nd - 2);
  const int64_t S = msa.size(nd - 3);
  int64_t B = 1;
  for (int64_t i = 0; i < nd - 3; ++i) B *= msa.size(i);
  const int64_t CM = w_m.size(0);
  const int64_t K = w_s.size(1);
  TORCH_CHECK(w_m.size(1) == NM + 2 && w_s.size(0) == CM, "weight/feature mismatch");
  TORCH_CHECK(s_input.size(nd - 2) == K && s_input.size(nd - 3) == T &&
              s_input.numel() == B * T * K, "s_input shape mismatch");
  TORCH_CHECK(hd.numel() == B * S * T && dv.numel() == B * S * T, "mask shape mismatch");
  TORCH_CHECK(K <= 32 * SUBS, "K too wide for the prefetch");
  TORCH_CHECK(NM <= 64, "too many MSA feature channels");

  auto sizes = msa.sizes().vec();
  sizes[nd - 1] = CM;
  at::Tensor out = at::empty(sizes, msa.options());
  if (out.numel() == 0) return out;

  const int wm_span = (int)std::max<int64_t>(NM + 2, NM <= 32 ? 32 : 64);
  const int wm_stride = wm_span | 1;    // odd -> conflict-free smem rows
  const size_t shmem = ((size_t)K + (size_t)wm_stride + 1) * sizeof(float);  // CBLK=1 lower bound
  TORCH_CHECK(shmem <= SMEM_CAP, "s_input row too wide for shared memory");
  auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH(CB, WM, MM)                                                        \
  do {                                                                           \
    constexpr int TH = SUBS * (CB);                                              \
    const size_t sh = ((size_t)K + (size_t)(CB) * wm_stride + (CB)) * sizeof(float); \
    if (sh <= SMEM_CAP) {                                                        \
      msa_embed_kernel<CB, WM, MM><<<(int)(B * T * (CM / (CB))), TH, sh, stream>>>( \
          (const __nv_bfloat16*)msa.data_ptr(), (const __nv_bfloat16*)hd.data_ptr(), \
          (const __nv_bfloat16*)dv.data_ptr(),                                   \
          (const __nv_bfloat16*)s_input.data_ptr(),                              \
          (const __nv_bfloat16*)w_m.data_ptr(), (const __nv_bfloat16*)w_s.data_ptr(), \
          (__nv_bfloat16*)out.data_ptr(), (int)S, (int)T, (int)K, (int)NM, (int)CM, \
          wm_stride, wm_span);                                                   \
      return out;                                                                \
    }                                                                            \
  } while (0)

  // WMAX = 16 covers K <= 512 with 32 threads per channel; 32 covers K <= 1024.
  const bool wide_k = (K > (int64_t)16 * SUBS);
  const bool wide_m = (NM > 32);
#define DISPATCH(CB)                                 \
  do {                                               \
    if (wide_k && wide_m) LAUNCH(CB, 32, 64);        \
    else if (wide_k)      LAUNCH(CB, 32, 32);        \
    else if (wide_m)      LAUNCH(CB, 16, 64);        \
    else                  LAUNCH(CB, 16, 32);        \
  } while (0)

  // CBLK = 4 (128 threads, 4 output channels per block) measured fastest: this
  // kernel is latency-bound, so the extra blocks buy more than the extra W_s
  // traffic costs. Narrower tiles only serve channel counts 4 does not divide.
  if (CM % 4 == 0) DISPATCH(4);
  if (CM % 2 == 0) DISPATCH(2);
  DISPATCH(1);
#undef DISPATCH
#undef LAUNCH
  TORCH_CHECK(false, "no fused configuration fits");
}
"""

_CPP_SRC = r"""
at::Tensor msa_embed(const at::Tensor& msa, const at::Tensor& hd, const at::Tensor& dv,
                     const at::Tensor& s_input, const at::Tensor& w_m,
                     const at::Tensor& w_s);
"""


def _pin_build_arch() -> None:
    """Build for the local GPU only -- a seven-architecture fatbin just costs
    compile time here."""
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=10)
    except Exception:
        return
    caps = sorted({c.strip() for c in out.splitlines() if c.strip()})
    if caps:
        os.environ["TORCH_CUDA_ARCH_LIST"] = " ".join(caps)


def _load_ext():
    from torch.utils.cpp_extension import load_inline

    _pin_build_arch()
    return load_inline(
        name="fk_af3_msa_module_embedder",
        cpp_sources=_CPP_SRC,
        cuda_sources=_CUDA_SRC,
        functions=["msa_embed"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


try:
    _C = _load_ext()
except Exception:  # pragma: no cover - no CUDA toolchain: eager path below
    _C = None


class MSAModuleEmbedder(nn.Module):
    """AF3 Algorithm 8, lines 1-4: MSA feature embedding.

    Args:
        c_m_feats: MSA input features channel dimension (34 = 32 msa + has_deletion + deletion_value)
        c_m: MSA channel dimension
        c_s_input: Single (s_input) channel dimension
    """

    def __init__(
        self,
        c_m_feats: int = 34,
        c_m: int = 64,
        c_s_input: int = 449,
    ):
        super().__init__()
        self.linear_m = Linear(c_m_feats, c_m, bias=False)
        self.linear_s_input = Linear(c_s_input, c_m, bias=False)
        # Plain tuple in __dict__: skips nn.Module.__getattr__ on the hot path.
        # The Parameter objects survive .to() and load_state_dict(), so caching
        # them stays valid even as their storage is replaced.
        self._wts = (self.linear_m.weight, self.linear_s_input.weight)
        self._fused = _C is not None

    def forward(
        self,
        batch: dict,
        s_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch: needs msa [*, N_msa, N_token, 32],
                   has_deletion [*, N_msa, N_token],
                   deletion_value [*, N_msa, N_token],
                   msa_mask [*, N_msa, N_token]
            s_input: [*, N_token, c_s_input]

        Returns:
            m: [*, N_seq, N_token, c_m]
            msa_mask: [*, N_seq, N_token]
        """
        if self._fused:
            w_m, w_s = self._wts
            try:
                m = _C.msa_embed(batch["msa"], batch["has_deletion"],
                                 batch["deletion_value"], s_input, w_m, w_s)
                return m, batch["msa_mask"]
            except Exception:
                # Shapes/dtypes the fused path does not cover: eager from here on.
                self._fused = False

        msa_feat = torch.cat(
            [
                batch["msa"],
                batch["has_deletion"].unsqueeze(-1),
                batch["deletion_value"].unsqueeze(-1),
            ],
            dim=-1,
        )
        msa_mask = batch["msa_mask"]

        m = self.linear_m(msa_feat)
        m = m + self.linear_s_input(s_input).unsqueeze(-3)

        return m, msa_mask
