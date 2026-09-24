"""Rotary position embedding for diffusion models (interleaved / GPT-J style).

Fast path: a hand-written CUDA kernel for the interleaved layout on contiguous
bf16/fp16 ``(batch, seqlen, nheads, headdim)`` input with ``rotary_dim ==
headdim``.  This is a pure streaming (1 read + 1 write) elementwise op, so the
kernel is written to (a) touch every byte of ``x`` exactly once with 128-bit
vector loads/stores and (b) keep per-call host work near the launch floor,
which is what actually dominates at these sizes.

General fallback: the Triton kernel from ``flash_attn.ops.triton.rotary``
(Tri Dao, 2023), via ``vllm.vllm_flash_attn.ops.triton.rotary`` -- used for the
NeoX/half-split layout, varlen/cu_seqlens, partial ``rotary_dim``, tensor
``seqlen_offsets``, non-contiguous input and any other dtype.
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional, Union

import torch
import torch.nn as nn

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# CUDA fast path
# ---------------------------------------------------------------------------

_CPP_SRC = r"""
#include <ATen/ATen.h>
#include <optional>
std::optional<at::Tensor> fk_rope(const at::Tensor &x, const at::Tensor &cos,
                                  const at::Tensor &sin, bool interleaved);
"""

_CUDA_SRC = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <optional>

namespace {

template <class T> struct Pk;
template <> struct Pk<__nv_bfloat16> {
  using v2 = __nv_bfloat162;
  static __device__ __forceinline__ float f(__nv_bfloat16 a) { return __bfloat162float(a); }
  static __device__ __forceinline__ v2 mk(float a, float b) { return __floats2bfloat162_rn(a, b); }
};
template <> struct Pk<__half> {
  using v2 = __half2;
  static __device__ __forceinline__ float f(__half a) { return __half2float(a); }
  static __device__ __forceinline__ v2 mk(float a, float b) { return __floats2half2_rn(a, b); }
};

constexpr int THREADS = 128;

// One block per (batch, seq) row of `nheads * headdim` elements.  Thread t owns
// the VPT 16-byte vectors {t, t + THREADS, ...} of that row; because THREADS is
// a multiple of NVEC_HD (= headdim/8) every one of those vectors lands on the
// same head-dim offset, so the four (cos, sin) values the thread needs are
// fetched once and reused across all VPT heads it touches.  Each x element is
// read exactly once: the even/odd lanes of a pair arrive together inside one
// 128-bit load and are combined in registers.
template <class T, int VPT, int NVEC_HD>
__global__ __launch_bounds__(THREADS) void rope_il_kernel(
    T *__restrict__ out, const T *__restrict__ x, const T *__restrict__ cosp,
    const T *__restrict__ sinp) {
  using v2 = typename Pk<T>::v2;
  union V16 { uint4 u; v2 h2[4]; };
  union V8 { uint2 u; T h[4]; };
  // Both derivable from the template params, so they stay out of the launch
  // parameter buffer (measurably cheaper per call at these sizes).
  constexpr int nvec_row = VPT * THREADS;  // = nheads * headdim/8
  constexpr int half = NVEC_HD * 4;        // = headdim/2

  const int s = blockIdx.x;
  const long row = (long)blockIdx.y * (long)gridDim.x + (long)s;
  const int t = threadIdx.x;
  const int j0 = (t % NVEC_HD) * 4;

  V8 cc, ss;
  cc.u = *reinterpret_cast<const uint2 *>(cosp + (long)s * half + j0);
  ss.u = *reinterpret_cast<const uint2 *>(sinp + (long)s * half + j0);
  float cf[4], sf[4];
#pragma unroll
  for (int m = 0; m < 4; ++m) {
    cf[m] = Pk<T>::f(cc.h[m]);
    sf[m] = Pk<T>::f(ss.h[m]);
  }

  const long base = row * (long)nvec_row * 8 + (long)t * 8;
  V16 xin[VPT];
#pragma unroll
  for (int k = 0; k < VPT; ++k)
    xin[k].u = *reinterpret_cast<const uint4 *>(x + base + (long)k * THREADS * 8);

#pragma unroll
  for (int k = 0; k < VPT; ++k) {
    V16 xo;
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      const float a = Pk<T>::f(xin[k].h2[m].x);
      const float b = Pk<T>::f(xin[k].h2[m].y);
      xo.h2[m] = Pk<T>::mk(a * cf[m] - b * sf[m], a * sf[m] + b * cf[m]);
    }
    *reinterpret_cast<uint4 *>(out + base + (long)k * THREADS * 8) = xo.u;
  }
}

template <class T, int NVEC_HD>
bool dispatch_vpt(int vpt, dim3 grid, cudaStream_t st, T *o, const T *x, const T *c,
                  const T *s) {
#define FK_CASE(V)                                                                       \
  case V:                                                                                \
    rope_il_kernel<T, V, NVEC_HD><<<grid, THREADS, 0, st>>>(o, x, c, s);                  \
    return true;
  switch (vpt) {
    FK_CASE(1)
    FK_CASE(2)
    FK_CASE(3)
    FK_CASE(4)
    FK_CASE(6)
    FK_CASE(8)
    default:
      return false;
  }
#undef FK_CASE
}

template <class T>
bool dispatch_hd(int nvec_hd, int vpt, dim3 grid, cudaStream_t st, void *o, const void *x,
                 const void *c, const void *s) {
  T *op = reinterpret_cast<T *>(o);
  const T *xp = reinterpret_cast<const T *>(x);
  const T *cp = reinterpret_cast<const T *>(c);
  const T *sp = reinterpret_cast<const T *>(s);
  if (nvec_hd == 16) return dispatch_vpt<T, 16>(vpt, grid, st, op, xp, cp, sp);
  if (nvec_hd == 8) return dispatch_vpt<T, 8>(vpt, grid, st, op, xp, cp, sp);
  return false;
}

inline bool aligned16(const void *p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }

}  // namespace

// Returns std::nullopt when this shape/layout is not covered; the caller then
// takes the general Triton path.
std::optional<at::Tensor> fk_rope(const at::Tensor &x, const at::Tensor &cos,
                                  const at::Tensor &sin, bool interleaved) {
  if (!interleaved || x.dim() != 4 || !x.is_cuda() || !x.is_contiguous()) return std::nullopt;
  const auto dt = x.scalar_type();
  if (dt != at::kBFloat16 && dt != at::kHalf) return std::nullopt;
  if (cos.scalar_type() != dt || sin.scalar_type() != dt) return std::nullopt;
  if (cos.device() != x.device() || sin.device() != x.device()) return std::nullopt;

  at::Tensor c = cos, s = sin;
  if (c.dim() == 3) {  // (1, seqlen_ro, rotary_dim/2)
    if (c.size(0) < 1 || s.dim() != 3) return std::nullopt;
    c = c.select(0, 0);
    s = s.select(0, 0);
  }
  if (c.dim() != 2 || s.dim() != 2 || !c.is_contiguous() || !s.is_contiguous())
    return std::nullopt;
  if (c.sizes() != s.sizes()) return std::nullopt;

  const int64_t batch = x.size(0), seqlen = x.size(1), nheads = x.size(2), hd = x.size(3);
  const int64_t half = c.size(1);
  if (half * 2 != hd) return std::nullopt;      // partial rotary_dim -> fallback
  if (c.size(0) < seqlen) return std::nullopt;
  if (hd != 64 && hd != 128) return std::nullopt;
  const int64_t nvec_hd = hd / 8;
  const int64_t nvec_row = nheads * nvec_hd;
  if (nvec_row % THREADS != 0) return std::nullopt;
  const int64_t vpt = nvec_row / THREADS;
  if (batch < 1 || seqlen < 1 || batch > 65535) return std::nullopt;

  const c10::cuda::CUDAGuard guard(x.device());
  at::Tensor out = at::empty_like(x);
  if (!aligned16(x.data_ptr()) || !aligned16(out.data_ptr()) || !aligned16(c.data_ptr()) ||
      !aligned16(s.data_ptr()))
    return std::nullopt;

  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  dim3 grid((unsigned)seqlen, (unsigned)batch);
  bool ok;
  if (dt == at::kBFloat16)
    ok = dispatch_hd<__nv_bfloat16>((int)nvec_hd, (int)vpt, grid, st, out.data_ptr(),
                                    x.data_ptr(), c.data_ptr(), s.data_ptr());
  else
    ok = dispatch_hd<__half>((int)nvec_hd, (int)vpt, grid, st, out.data_ptr(), x.data_ptr(),
                              c.data_ptr(), s.data_ptr());
  if (!ok) return std::nullopt;
  return out;
}
"""


def _build_ext():
    if not torch.cuda.is_available():
        return None
    from torch.utils.cpp_extension import load_inline

    tag = hashlib.sha1((_CPP_SRC + _CUDA_SRC).encode()).hexdigest()[:12]
    prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    # Build only for the arch actually present (the ambient list has 6+ arches,
    # which makes the one-time build several times slower); +PTX so a newer
    # device can still JIT.
    major, minor = torch.cuda.get_device_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}+PTX"
    try:
        return load_inline(
            name=f"fk_diffusion_rope_{tag}",
            cpp_sources=[_CPP_SRC],
            cuda_sources=[_CUDA_SRC],
            functions=["fk_rope"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    finally:
        if prev is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev


try:
    _EXT = _build_ext()
except Exception:  # noqa: BLE001 -- any build/runtime problem falls back to Triton
    _EXT = None


def _fk_rope_unavailable(x, cos, sin, interleaved):
    """Stand-in when the extension could not be built: always take the fallback."""
    return None


_FK_ROPE = _EXT.fk_rope if _EXT is not None else _fk_rope_unavailable


# ---------------------------------------------------------------------------
# General Triton fallback
# (from flash_attn / vllm_flash_attn, Copyright (c) 2023 Tri Dao)
# ---------------------------------------------------------------------------

@triton.jit
def _rotary_kernel(
    OUT, X, COS, SIN, CU_SEQLENS, SEQLEN_OFFSETS,
    seqlen, rotary_dim, seqlen_ro,
    stride_out_batch, stride_out_seqlen, stride_out_nheads, stride_out_headdim,
    stride_x_batch, stride_x_seqlen, stride_x_nheads, stride_x_headdim,
    BLOCK_K: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)
    pid_batch = tl.program_id(axis=2)
    rotary_dim_half = rotary_dim // 2

    if not IS_VARLEN:
        X = X + pid_batch * stride_x_batch + pid_head * stride_x_nheads
        OUT = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads
    else:
        start_idx = tl.load(CU_SEQLENS + pid_batch)
        seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
        X = X + start_idx * stride_x_seqlen + pid_head * stride_x_nheads
        OUT = OUT + start_idx * stride_out_seqlen + pid_head * stride_out_nheads

    if pid_m * BLOCK_M >= seqlen:
        return
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if not IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + SEQLEN_OFFSETS
    else:
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)
    rk = tl.arange(0, BLOCK_K)
    rk_half = tl.arange(0, BLOCK_K // 2)

    if not INTERLEAVED:
        X = X + (rm[:, None] * stride_x_seqlen + rk_half[None, :] * stride_x_headdim)
        COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        cos = tl.load(
            COS, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=1.0
        ).to(tl.float32)
        sin = tl.load(
            SIN, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=0.0
        ).to(tl.float32)
        x0 = tl.load(
            X, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half), other=0.0
        ).to(tl.float32)
        x1 = tl.load(
            X + rotary_dim_half * stride_x_headdim,
            mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        OUT = OUT + (rm[:, None] * stride_out_seqlen + rk_half[None, :] * stride_out_headdim)
        tl.store(OUT, o0, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half))
        tl.store(
            OUT + rotary_dim_half * stride_out_headdim,
            o1,
            mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
        )
    else:
        rk_swap = rk + ((rk + 1) % 2) * 2 - 1  # 1, 0, 3, 2, 5, 4, ...
        rk_repeat = tl.arange(0, BLOCK_K) // 2
        X0 = X + (rm[:, None] * stride_x_seqlen + rk[None, :] * stride_x_headdim)
        X1 = X + (rm[:, None] * stride_x_seqlen + rk_swap[None, :] * stride_x_headdim)
        COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        cos = tl.load(
            COS,
            mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half),
            other=1.0,
        ).to(tl.float32)
        sin = tl.load(
            SIN,
            mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        x0 = tl.load(X0, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim), other=0.0).to(
            tl.float32
        )
        x1 = tl.load(
            X1, mask=(rm[:, None] < seqlen) & (rk_swap[None, :] < rotary_dim), other=0.0
        ).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        x0_cos = x0 * cos
        x1_sin = x1 * sin
        out = tl.where(rk[None, :] % 2 == 0, x0_cos - x1_sin, x0_cos + x1_sin)
        OUT = OUT + (rm[:, None] * stride_out_seqlen + rk[None, :] * stride_out_headdim)
        tl.store(OUT, out, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim))


def _apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved: bool = False,
    inplace: bool = False,
    conjugate: bool = False,
) -> torch.Tensor:
    """Launch the general Triton rotary-embedding kernel.

    Args:
        x: (batch, seqlen, nheads, headdim) or (total_seqlen, nheads, headdim)
            if ``cu_seqlens`` is provided.
        cos, sin: (seqlen_ro, rotary_dim / 2)
    """
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert max_seqlen is not None
        total_seqlen, nheads, headdim = x.shape
        batch = cu_seqlens.shape[0] - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim = cos.shape
    rotary_dim *= 2
    assert rotary_dim <= headdim
    assert headdim <= 256
    assert seqlen_ro >= seqlen

    cos, sin = cos.contiguous(), sin.contiguous()
    if isinstance(seqlen_offsets, torch.Tensor):
        seqlen_offsets = seqlen_offsets.contiguous()

    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])

    BLOCK_K = (
        32 if rotary_dim <= 32
        else (64 if rotary_dim <= 64
              else (128 if rotary_dim <= 128 else 256))
    )
    BLOCK_M = 4 if interleaved else (8 if rotary_dim <= 128 else 4)
    grid = lambda META: (triton.cdiv(seqlen, META["BLOCK_M"]), nheads, batch)  # noqa

    with torch.cuda.device(x.device.index):
        _rotary_kernel[grid](
            output, x, cos, sin, cu_seqlens, seqlen_offsets,
            seqlen, rotary_dim, seqlen_ro,
            output.stride(0) if not is_varlen else 0,
            output.stride(-3), output.stride(-2), output.stride(-1),
            x.stride(0) if not is_varlen else 0,
            x.stride(-3), x.stride(-2), x.stride(-1),
            BLOCK_K,
            isinstance(seqlen_offsets, torch.Tensor),
            is_varlen, interleaved, conjugate, BLOCK_M,
            num_warps=2 if rotary_dim <= 64 else 4,
        )
    return output


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class DiffusionRoPE(nn.Module):
    """Apply rotary embeddings given pre-computed (cos, sin) tensors.

    Parameters
    ----------
    is_neox_style : bool
        If True, use the GPT-NeoX (half-split) layout.
        If False (default for FLUX), use the interleaved (GPT-J) layout.
    """

    def __init__(self, is_neox_style: bool = False) -> None:
        super().__init__()
        self.interleaved = not is_neox_style

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        # All shape/layout eligibility checks happen inside the extension, so the
        # per-call Python work is one global load, one attribute load and one
        # call.  ``None`` means "not covered, take the general path".
        out = _FK_ROPE(x, cos, sin, self.interleaved)
        if out is not None:
            return out
        if cos.dim() == 3:
            cos = cos[0]
            sin = sin[0]
        return _apply_rotary(x, cos, sin, interleaved=self.interleaved)

    # The whole op is a few microseconds of GPU work, so nn.Module.__call__'s
    # hook dispatch (~1.4 us/call here) is a measurable share of the runtime.
    # This module has no hooks, no parameters and no buffers, so calling forward
    # directly is equivalent.
    __call__ = forward
