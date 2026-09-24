"""Adaptive Layer Norm modules for diffusion transformers (L2 composite).

AdaLayerNormZero: 6-output adaLN-Zero for dual-stream FLUX blocks.
AdaLayerNormZeroSingle: 3-output adaLN-Zero for single-stream FLUX blocks.

Optimized against diffusers' reference implementation, which runs the module as
six separate eager kernels::

    silu(emb) -> linear -> chunk -> layer_norm(x) -> *(1+scale) -> +shift

That costs six passes over ``x`` (layer_norm writes it, the broadcast multiply
reads and writes it, the add reads and writes it -- and the broadcast operands
push both elementwise ops onto a non-vectorized TensorIterator path at ~1.5
TB/s), plus a cuBLAS GEMV on a degenerate M=1 problem.  Here the module is two
hand-written CUDA kernels behind a single pybind call:

1. ``silu_gemv`` -- SiLU on ``emb`` staged through shared memory, then the
   ``[6D, D]`` (or ``[3D, D]``) projection as a proper bandwidth-bound GEMV: one
   warp per output row, ``KU`` 128-bit streaming loads issued per step, and the
   SiLU vector pre-unpacked into fp32 registers so the inner loop is 8 unpacks +
   8 FMAs per 16 B of weight.  The weight read is the dominant traffic of the
   whole op, so this kernel is what sets the floor; it runs a little ahead of
   cuBLAS on both shapes.
2. ``ln_affine`` -- LayerNorm statistics, normalization, ``*(1+scale)`` and
   ``+shift`` in a single pass.  ``x`` is read once into registers and written
   once; ``1+scale`` and ``shift`` are staged in shared memory so the rows a
   block owns share one load of them.

Numerics follow the reference op-for-op, including the intermediate rounding to
bfloat16 after SiLU, after the linear, after the norm and after the multiply, so
the fused path stays inside a bfloat16 ULP or two of the eager chain.

The conditioning chunks the caller gets back (``gate_msa``, ``shift_mlp``,
``scale_mlp``, ``gate_mlp``) are sliced out of the single projection tensor in
C++, and the per-call Python work is one attribute fetch plus the call, because
at these sizes kernel-launch and dispatch overhead is a visible fraction of the
measured latency.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.silu import SiLU

# ---------------------------------------------------------------------------
# CUDA extension
# ---------------------------------------------------------------------------

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

union V8 { uint4 u; bf16 h[8]; uint32_t w[4]; };

__device__ __forceinline__ float b2f(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16 f2b(float v) { return __float2bfloat16(v); }

// Two packed bf16 -> two floats with pure bit ops (cheaper than two cvt).
__device__ __forceinline__ void up2(uint32_t v, float& a, float& b) {
  a = __uint_as_float(v << 16);
  b = __uint_as_float(v & 0xffff0000u);
}

// ---------------------------------------------------------------------------
// Kernel 1: out[m] = bias[m] + dot(silu(emb), W[m, :])
//
// One warp owns one output row and walks it in steps of KU 128-bit loads, so a
// warp step is KU fully coalesced 512 B segments and a warp covers its row with
// K/256 loads total.  silu(emb) is staged once per block in shared memory and
// then unpacked into fp32 registers, leaving 8 unpacks + 8 FMAs per 16 B read.
// Requires K % (256 * KU) == 0.
// ---------------------------------------------------------------------------
template <int WARPS, int KU>
__global__ __launch_bounds__(WARPS * 32) void silu_gemv_k(
    const bf16* __restrict__ W, const bf16* __restrict__ emb,
    const bf16* __restrict__ bias, bf16* __restrict__ out, int M, int K) {
  extern __shared__ bf16 se[];
  const int tid = threadIdx.x;
  constexpr int NT = WARPS * 32;

  for (int i = tid * 8; i < K; i += NT * 8) {
    V8 v, o;
    v.u = *reinterpret_cast<const uint4*>(emb + i);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float f = b2f(v.h[j]);
      o.h[j] = f2b(f * __frcp_rn(1.f + __expf(-f)));
    }
    *reinterpret_cast<uint4*>(se + i) = o.u;
  }
  __syncthreads();

  const int warp = tid >> 5, lane = tid & 31;
  const int row = blockIdx.x * WARPS + warp;
  if (row >= M) return;
  const bf16* wp = W + (size_t)row * (size_t)K;

  float acc = 0.f;
  for (int k = lane * 8; k < K; k += 32 * 8 * KU) {
    V8 w[KU];
    float sv[KU][8];
#pragma unroll
    for (int u = 0; u < KU; ++u)
      w[u].u = *reinterpret_cast<const uint4*>(wp + k + u * 32 * 8);
#pragma unroll
    for (int u = 0; u < KU; ++u) {
      V8 s;
      s.u = *reinterpret_cast<const uint4*>(se + k + u * 32 * 8);
#pragma unroll
      for (int q = 0; q < 4; ++q) up2(s.w[q], sv[u][2 * q], sv[u][2 * q + 1]);
    }
#pragma unroll
    for (int u = 0; u < KU; ++u) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        float x0, x1;
        up2(w[u].w[q], x0, x1);
        acc = fmaf(x0, sv[u][2 * q], acc);
        acc = fmaf(x1, sv[u][2 * q + 1], acc);
      }
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (lane == 0) out[row] = f2b(acc + (bias != nullptr ? b2f(bias[row]) : 0.f));
}

// ---------------------------------------------------------------------------
// Kernel 2: out[n, c] = norm(x[n, :])[c] * (1 + scale[c]) + shift[c]
//
// TPR threads cooperate on one row (TPR * VPT * 8 == K) and a block owns RPB
// rows, so the block's single staging of (1+scale) / shift into shared memory is
// amortized over RPB rows and x is read exactly once, straight into registers.
// Rounding matches the eager chain: bf16 after the norm, the multiply, the add.
// ---------------------------------------------------------------------------
template <int TPR, int VPT, int RPB>
__global__ __launch_bounds__(TPR * RPB) void ln_affine_k(
    const bf16* __restrict__ x, const bf16* __restrict__ shift,
    const bf16* __restrict__ scale, bf16* __restrict__ out, int N, int K,
    float eps) {
  constexpr int NT = TPR * RPB, NW = TPR / 32;
  extern __shared__ bf16 sm[];  // [0, K) = 1 + scale ; [K, 2K) = shift
  __shared__ float red[2 * RPB * (NW > 1 ? NW : 1)];

  const int tid = threadIdx.x;
  for (int i = tid * 8; i < K; i += NT * 8) {
    V8 a, b;
    a.u = *reinterpret_cast<const uint4*>(scale + i);
    b.u = *reinterpret_cast<const uint4*>(shift + i);
#pragma unroll
    for (int j = 0; j < 8; ++j) a.h[j] = f2b(1.f + b2f(a.h[j]));
    *reinterpret_cast<uint4*>(sm + i) = a.u;
    *reinterpret_cast<uint4*>(sm + K + i) = b.u;
  }
  __syncthreads();

  const int grp = tid / TPR, sub = tid - grp * TPR, lane = tid & 31;
  const int row = blockIdx.x * RPB + grp;
  // A partial last block must not return here: the groups that do have a row
  // reach a __syncthreads() below, so an early exit would hang the block.
  const bool live = row < N;

  int cb[VPT];
  V8 v[VPT];
#pragma unroll
  for (int i = 0; i < VPT; ++i) cb[i] = (i * TPR + sub) * 8;
  float s = 0.f, q = 0.f;
  if (live) {
    const bf16* xp = x + (size_t)row * (size_t)K;
#pragma unroll
    for (int i = 0; i < VPT; ++i)
      v[i].u = *reinterpret_cast<const uint4*>(xp + cb[i]);
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float f = b2f(v[i].h[j]);
        s += f;
        q = fmaf(f, f, q);
      }
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    s += __shfl_xor_sync(0xffffffffu, s, o);
    q += __shfl_xor_sync(0xffffffffu, q, o);
  }
  if (NW > 1) {
    const int slot = 2 * (grp * NW + (sub >> 5));
    if (lane == 0) {
      red[slot] = s;
      red[slot + 1] = q;
    }
    __syncthreads();
    s = 0.f;
    q = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) {
      s += red[2 * (grp * NW + w)];
      q += red[2 * (grp * NW + w) + 1];
    }
  }

  if (!live) return;  // past every barrier now

  const float inv = 1.f / (float)K;
  const float mean = s * inv;
  const float var = q * inv - mean * mean;
  const float rstd = rsqrtf(var + eps);
  bf16* op = out + (size_t)row * (size_t)K;
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    V8 o, sc, sh;
    sc.u = *reinterpret_cast<const uint4*>(sm + cb[i]);
    sh.u = *reinterpret_cast<const uint4*>(sm + K + cb[i]);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float n = b2f(f2b((b2f(v[i].h[j]) - mean) * rstd));
      const float t = b2f(f2b(n * b2f(sc.h[j])));
      o.h[j] = f2b(t + b2f(sh.h[j]));
    }
    *reinterpret_cast<uint4*>(op + cb[i]) = o.u;
  }
}

// ---------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------
#define GEMV_CASE(KU_)                                                        \
  case KU_:                                                                   \
    silu_gemv_k<WARPS, KU_><<<grid, WARPS * 32, smem, stream>>>(              \
        Wp, ep, bp, op, M, K);                                                \
    return;

static void launch_silu_gemv(const bf16* Wp, const bf16* ep, const bf16* bp,
                             bf16* op, int M, int K, cudaStream_t stream) {
  constexpr int WARPS = 32;
  const dim3 grid((M + WARPS - 1) / WARPS);
  const size_t smem = (size_t)K * sizeof(bf16);
  int ku = 1;
  for (int c : {6, 4, 3, 2}) {
    if (K % (256 * c) == 0) {
      ku = c;
      break;
    }
  }
  switch (ku) {
    GEMV_CASE(6)
    GEMV_CASE(4)
    GEMV_CASE(3)
    GEMV_CASE(2)
    default:
      GEMV_CASE(1)
  }
}

#define LN_CASE(VPT_, RPB_)                                                   \
  case VPT_: {                                                               \
    constexpr int RPB = RPB_;                                                 \
    const dim3 grid((N + RPB - 1) / RPB);                                     \
    ln_affine_k<TPR, VPT_, RPB><<<grid, TPR * RPB, smem, stream>>>(           \
        xp, shiftp, scalep, outp, N, K, eps);                                 \
    return;                                                                   \
  }

static void launch_ln_affine(const bf16* xp, const bf16* shiftp,
                             const bf16* scalep, bf16* outp, int N, int K,
                             float eps, cudaStream_t stream) {
  constexpr int TPR = 64;
  const size_t smem = (size_t)2 * K * sizeof(bf16);
  switch (K / (TPR * 8)) {
    LN_CASE(1, 8)
    LN_CASE(2, 4)
    LN_CASE(3, 2)
    LN_CASE(4, 2)
    LN_CASE(5, 2)
    LN_CASE(6, 2)
    LN_CASE(7, 2)
    LN_CASE(8, 2)
    LN_CASE(9, 1)
    LN_CASE(10, 1)
    LN_CASE(11, 1)
    LN_CASE(12, 1)
    LN_CASE(13, 1)
    LN_CASE(14, 1)
    LN_CASE(15, 1)
    LN_CASE(16, 1)
    default:
      TORCH_CHECK(false, "ada_layer_norm: unsupported hidden size ", K);
  }
}

// proj = silu(emb) @ W^T + bias ; out = ln(x) * (1 + proj[D:2D]) + proj[0:D]
static std::pair<torch::Tensor, torch::Tensor> run_core(
    const torch::Tensor& x, const torch::Tensor& W,
    const c10::optional<torch::Tensor>& bias, const torch::Tensor& emb,
    double eps) {
  const int K = (int)W.size(1);
  const int M = (int)W.size(0);
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 &&
                  emb.scalar_type() == torch::kBFloat16 &&
                  W.scalar_type() == torch::kBFloat16,
              "ada_layer_norm: bfloat16 only");
  TORCH_CHECK(x.is_contiguous() && emb.is_contiguous() && W.is_contiguous(),
              "ada_layer_norm: contiguous inputs required");
  TORCH_CHECK(x.size(-1) == K && emb.numel() == K, "ada_layer_norm: bad shape");
  // With a single conditioning vector the reference broadcasts ``[..., N, D]``
  // against ``[1, 1, D]``, so the result keeps x's shape only from rank 3 up; a
  // rank-2 x would be promoted to ``[1, N, D]``.  Leave that to the eager path.
  TORCH_CHECK(x.dim() >= 3, "ada_layer_norm: expected x of rank >= 3");
  TORCH_CHECK(K % 512 == 0 && K <= 8192, "ada_layer_norm: bad hidden size");
  // 128-bit loads: a misaligned base would fault rather than raise.
  TORCH_CHECK((((uintptr_t)x.data_ptr() | (uintptr_t)emb.data_ptr() |
                (uintptr_t)W.data_ptr()) & 15u) == 0,
              "ada_layer_norm: inputs must be 16-byte aligned");
  const int64_t N64 = x.numel() / K;
  TORCH_CHECK(N64 > 0 && N64 <= 2147483647, "ada_layer_norm: bad row count");
  const bf16* bp = nullptr;
  if (bias.has_value() && bias->defined()) {
    TORCH_CHECK(bias->scalar_type() == torch::kBFloat16 &&
                    bias->is_contiguous() && bias->numel() == M,
                "ada_layer_norm: bad bias");
    bp = (const bf16*)bias->data_ptr();
  }

  const at::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  // The GEMV is the long pole, so get it in flight before doing anything else
  // host-side: at these sizes the device sits idle until the first launch lands.
  auto proj = torch::empty({1, M}, emb.options());
  const bf16* pp = (const bf16*)proj.data_ptr();
  launch_silu_gemv((const bf16*)W.data_ptr(), (const bf16*)emb.data_ptr(), bp,
                   (bf16*)proj.data_ptr(), M, K, stream);
  auto out = torch::empty_like(x);
  launch_ln_affine((const bf16*)x.data_ptr(), pp, pp + K, (bf16*)out.data_ptr(),
                   (int)N64, K, (float)eps, stream);
  return {out, proj};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
ada_zero(const torch::Tensor& x, const torch::Tensor& W,
         const c10::optional<torch::Tensor>& bias, const torch::Tensor& emb,
         double eps) {
  const int64_t D = W.size(1);
  TORCH_CHECK(W.size(0) == 6 * D, "ada_layer_norm: expected a 6D projection");
  auto r = run_core(x, W, bias, emb, eps);
  return {r.first, r.second.slice(1, 2 * D, 3 * D),
          r.second.slice(1, 3 * D, 4 * D), r.second.slice(1, 4 * D, 5 * D),
          r.second.slice(1, 5 * D, 6 * D)};
}

std::tuple<torch::Tensor, torch::Tensor> ada_single(
    const torch::Tensor& x, const torch::Tensor& W,
    const c10::optional<torch::Tensor>& bias, const torch::Tensor& emb,
    double eps) {
  const int64_t D = W.size(1);
  TORCH_CHECK(W.size(0) == 3 * D, "ada_layer_norm: expected a 3D projection");
  auto r = run_core(x, W, bias, emb, eps);
  return {r.first, r.second.slice(1, 2 * D, 3 * D)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ada_zero", &ada_zero, "fused adaLN-Zero (6 chunks)");
  m.def("ada_single", &ada_single, "fused adaLN-Zero single-stream (3 chunks)");
}
"""

_EXT = None
_EXT_TRIED = False


def _ext():
    """JIT-build (once) and return the fused adaLN extension, or ``None``."""
    global _EXT, _EXT_TRIED
    if _EXT_TRIED:
        return _EXT
    _EXT_TRIED = True
    try:
        from torch.utils.cpp_extension import load_inline

        if "TORCH_CUDA_ARCH_LIST" not in os.environ:
            major, minor = torch.cuda.get_device_capability()
            suffix = "a" if major in (9, 10, 12) else ""
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}{suffix}"
        _EXT = load_inline(
            name="fk_ada_layer_norm_fused",
            cpp_sources="",
            cuda_sources=_CUDA_SRC,
            extra_cuda_cflags=[
                "-O3",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
            ],
            extra_cflags=["-O3"],
            verbose=False,
        )
    except Exception:  # noqa: BLE001 - no nvcc / unsupported arch: stay on eager
        _EXT = None
    return _EXT


def _setup(mod: nn.Module, entry: str, chunks: int):
    """Bind the fused entry point for *mod*, or return ``False`` for eager.

    Called once, from the first forward: only then are the parameters on their
    final device and dtype.  The cached handles live in ``mod.__dict__`` so the
    hot path is a plain attribute fetch.
    """
    ext = _ext()
    lin, norm = mod.linear, mod.norm
    w, b = lin.weight, lin.bias
    d = w.shape[1]
    ok = (
        ext is not None
        and not norm.promote_fp32
        and not norm.elementwise_affine
        and getattr(mod, "emb", None) is None
        and w.is_cuda
        and w.dtype is torch.bfloat16
        and w.is_contiguous()
        and w.shape[0] == chunks * d
        and (b is None or (b.dtype is torch.bfloat16 and b.is_contiguous()))
        # 128-bit vectorized loads; one 8-element chunk per thread slot.
        and d % 512 == 0
        and d <= 8192
        and w.data_ptr() % 16 == 0
    )
    if not ok:
        return False
    # object.__setattr__: nn.Module.__setattr__ would re-register ``w`` as a
    # parameter named ``_w`` (duplicating it in state_dict), and a plain
    # __dict__ entry is also the fastest attribute fetch on the hot path.
    object.__setattr__(mod, "_w", w)
    object.__setattr__(mod, "_b", b)
    object.__setattr__(mod, "_eps", float(norm.eps))
    return getattr(ext, entry)


class AdaLayerNormZero(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: int | None = None,
                 norm_type="layer_norm", bias=True, promote_fp32: bool = True):
        super().__init__()
        self.emb = None

        self.silu = SiLU()
        self.linear = Linear(embedding_dim, 6 * embedding_dim, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6,
                                  promote_fp32=promote_fp32)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm'."
            )
        self._fused = None

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor | None = None,
        class_labels: torch.LongTensor | None = None,
        hidden_dtype: torch.dtype | None = None,
        emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fused = self._fused
        if fused is None:
            fused = self._fused = _setup(self, "ada_zero", 6)
        if fused is not False and emb is not None and x.dtype is torch.bfloat16:
            try:
                return fused(x, self._w, self._b, emb, self._eps)
            except (RuntimeError, TypeError):
                # The extension validates shapes/layout before it launches
                # anything, so a reject here is clean: drop to eager for good.
                self._fused = False

        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroSingle(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero) for single-stream blocks.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
    """

    def __init__(self, embedding_dim: int, norm_type="layer_norm", bias=True,
                 promote_fp32: bool = True):
        super().__init__()

        self.silu = SiLU()
        self.linear = Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6,
                                  promote_fp32=promote_fp32)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm'."
            )
        self._fused = None

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused = self._fused
        if fused is None:
            fused = self._fused = _setup(self, "ada_single", 3)
        if fused is not False and emb is not None and x.dtype is torch.bfloat16:
            try:
                return fused(x, self._w, self._b, emb, self._eps)
            except (RuntimeError, TypeError):
                # The extension validates shapes/layout before it launches
                # anything, so a reject here is clean: drop to eager for good.
                self._fused = False

        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa
