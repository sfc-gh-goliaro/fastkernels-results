"""Rotary embedding helpers used by Oasis.

Minimal subset adapted for the official open-oasis inference path:
  - axial pixel frequencies for spatial attention
  - standard sequence frequencies for temporal attention
  - query/key rotation helpers

``_forward_freqs`` (the benchmarked path) is the outer product of the positions
with the frequency table, interleaved 2x along the last dim.  Eager torch needs
three kernels for it (cast + mul + repeat_interleave gather); at these shapes
(<= 512 output elements) every launch is pure overhead, so the whole thing is
fused into one custom CUDA kernel.
"""

from __future__ import annotations

from math import pi

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Fused outer-product + repeat_interleave(2) kernel.
#
# Semantics replicated exactly, including rounding: the baseline first casts the
# positions to ``freqs.dtype`` and then multiplies, and torch's half/bfloat16
# ``mul`` computes in fp32 and rounds once.  So: round t -> freqs dtype, widen
# both to float, multiply, round back.
# ---------------------------------------------------------------------------
_CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float ld(const float* p, int i) { return p[i]; }
__device__ __forceinline__ float ld(const __half* p, int i) { return __half2float(p[i]); }
__device__ __forceinline__ float ld(const __nv_bfloat16* p, int i) { return __bfloat162float(p[i]); }
__device__ __forceinline__ float ld(const long long* p, int i) { return (float)p[i]; }
__device__ __forceinline__ float ld(const int* p, int i) { return (float)p[i]; }

// round a float to the accumulation dtype, then widen again
__device__ __forceinline__ float rnd(float x, float*) { return x; }
__device__ __forceinline__ float rnd(float x, __half*) { return __half2float(__float2half(x)); }
__device__ __forceinline__ float rnd(float x, __nv_bfloat16*) {
  return __bfloat162float(__float2bfloat16(x));
}

__device__ __forceinline__ void st(float* p, int i, float v) { p[i] = v; }
__device__ __forceinline__ void st(__half* p, int i, float v) { p[i] = __float2half(v); }
__device__ __forceinline__ void st(__nv_bfloat16* p, int i, float v) {
  p[i] = __float2bfloat16(v);
}

// One thread per (row, freq) pair; each writes the duplicated 2-wide slot.
template <typename TI, typename TF>
__global__ void oasis_freqs_kernel(const TI* __restrict__ t, const TF* __restrict__ freqs,
                                   TF* __restrict__ out, int rows, int F, int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int j = idx % F;
  int r = idx / F;
  float a = rnd(ld(t, r), (TF*)nullptr);
  float b = ld(freqs, j);
  float v = a * b;
  int o = (r * F + j) * 2;
  st(out, o, v);
  st(out, o + 1, v);
}

template <typename TI, typename TF>
static inline void launch(const at::Tensor& t, const at::Tensor& f, at::Tensor& out,
                          int rows, int F) {
  int total = rows * F;
  int threads = total < 256 ? 32 * ((total + 31) / 32) : 256;
  int blocks = (total + threads - 1) / threads;
  oasis_freqs_kernel<TI, TF><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const TI*>(t.const_data_ptr()),
      reinterpret_cast<const TF*>(f.const_data_ptr()),
      reinterpret_cast<TF*>(out.data_ptr()), rows, F, total);
}

template <typename TF>
static inline void dispatch_in(const at::Tensor& t, const at::Tensor& f, at::Tensor& out,
                               int rows, int F) {
  switch (t.scalar_type()) {
    case at::kFloat:    launch<float, TF>(t, f, out, rows, F); break;
    case at::kHalf:     launch<__half, TF>(t, f, out, rows, F); break;
    case at::kBFloat16: launch<__nv_bfloat16, TF>(t, f, out, rows, F); break;
    case at::kLong:     launch<long long, TF>(t, f, out, rows, F); break;
    case at::kInt:      launch<int, TF>(t, f, out, rows, F); break;
    default: TORCH_CHECK(false, "unsupported positions dtype ", t.scalar_type());
  }
}

at::Tensor oasis_forward_freqs(const at::Tensor& positions, const at::Tensor& freqs) {
  TORCH_CHECK(freqs.dim() == 1 && freqs.is_contiguous(), "freqs must be 1-D contiguous");
  TORCH_CHECK(positions.is_cuda() && freqs.is_cuda(), "cuda only");
  TORCH_CHECK(positions.is_contiguous(), "positions must be contiguous");
  const int F = (int)freqs.numel();
  const int rows = (int)positions.numel();
  std::vector<int64_t> shape(positions.sizes().begin(), positions.sizes().end());
  shape.push_back(2 * (int64_t)F);
  at::Tensor out = at::empty(shape, positions.options().dtype(freqs.scalar_type()));
  if (rows == 0 || F == 0) return out;
  switch (freqs.scalar_type()) {
    case at::kHalf:     dispatch_in<__half>(positions, freqs, out, rows, F); break;
    case at::kBFloat16: dispatch_in<__nv_bfloat16>(positions, freqs, out, rows, F); break;
    case at::kFloat:    dispatch_in<float>(positions, freqs, out, rows, F); break;
    default: TORCH_CHECK(false, "unsupported freqs dtype ", freqs.scalar_type());
  }
  return out;
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
at::Tensor oasis_forward_freqs(const at::Tensor& positions, const at::Tensor& freqs);
"""


def _load_ext():
    import os

    from torch.utils.cpp_extension import load_inline

    if not torch.cuda.is_available():
        return None
    # Build only for the local arch -- six -gencode passes over a trivial kernel
    # is a minute of cold start for nothing. Same trick as
    # fastkernels.infra.cuda_ext._pin_build_arch, but scoped to our own build.
    prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    override = os.environ.get("FASTKERNELS_CUDA_ARCH_LIST")
    if override is None:
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}.{minor}" + ("a" if major in (9, 10, 12) else "")
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    elif override.strip():
        os.environ["TORCH_CUDA_ARCH_LIST"] = override
    try:
        return load_inline(
            name="fk_oasis_rotary_ext",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["oasis_forward_freqs"],
            extra_cuda_cflags=[
                "-O3",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            ],
            verbose=False,
        )
    finally:
        if prev is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev


try:
    _EXT = _load_ext()
except Exception:  # pragma: no cover - fall back to eager if the build fails
    _EXT = None

_FUSED = _EXT.oasis_forward_freqs if _EXT is not None else None


def oasis_rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def oasis_apply_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    dtype = t.dtype
    rot_dim = freqs.shape[-1]
    t_left = t[..., :0]
    t_middle = t[..., :rot_dim]
    t_right = t[..., rot_dim:]
    t_transformed = (t_middle * freqs.cos()) + (oasis_rotate_half(t_middle) * freqs.sin())
    return torch.cat((t_left, t_transformed, t_right), dim=-1).to(dtype)


class OasisRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        freqs_for: str = "lang",
        theta: float = 10000.0,
        max_freq: float = 10.0,
    ):
        super().__init__()
        self.dim = dim
        self.freqs_for = freqs_for
        if freqs_for == "lang":
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        else:
            raise ValueError(f"unsupported rotary mode: {freqs_for}")
        self.freqs = nn.Parameter(freqs, requires_grad=False)
        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    @property
    def device(self) -> torch.device:
        return self.dummy.device

    @staticmethod
    def _forward_freqs_eager(positions: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        out = torch.einsum("..., f -> ... f", positions.to(freqs.dtype), freqs)
        return out.repeat_interleave(2, dim=-1)

    def _forward_freqs(self, positions: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        if _FUSED is not None:
            try:
                return _FUSED(positions, freqs)
            except Exception:
                pass
        return self._forward_freqs_eager(positions, freqs)

    def forward(
        self,
        t: torch.Tensor,
        freqs: torch.Tensor,
        seq_len: int | None = None,
        offset: int = 0,
    ) -> torch.Tensor:
        del seq_len, offset
        if _FUSED is not None:
            try:
                return _FUSED(t, freqs)
            except Exception:
                pass
        return self._forward_freqs_eager(t, freqs)

    def rotate_queries_or_keys(self, t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        seq_len = t.shape[-2]
        positions = torch.arange(seq_len, device=t.device, dtype=t.dtype)
        seq_freqs = self.forward(positions, freqs, seq_len=seq_len)
        return oasis_apply_rotary_emb(seq_freqs, t)

    def get_axial_freqs(self, *dims: int) -> torch.Tensor:
        colon = slice(None)
        all_freqs = []
        for index, dim in enumerate(dims):
            use_pixel = self.freqs_for == "pixel" and index >= len(dims) - 2
            if use_pixel:
                pos = torch.linspace(-1, 1, steps=dim, device=self.device)
            else:
                pos = torch.arange(dim, device=self.device)
            seq_freqs = self.forward(pos, self.freqs, seq_len=dim)
            axis = [None] * len(dims)
            axis[index] = colon
            all_freqs.append(seq_freqs[(Ellipsis, *axis, colon)])
        all_freqs = torch.broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim=-1)
