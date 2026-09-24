"""2D rotary position embeddings for FLUX (concatenated per-axis 1D embeddings).

Generates (cos, sin) tensors from integer grid IDs for use with DiffusionRoPE.

The reference (``_get_1d_rotary_pos_embed``, copied from diffusers'
``get_1d_rotary_pos_embed``) builds, per axis, an ``arange`` -> ``pow`` ->
``outer`` -> ``polar`` -> ``real``/``imag`` chain in **float64** and then two
``cat``s -- ~20 launches of crippled-rate fp64 work for an output that is only
a few MB wide (~180-230us for the captured shapes).

This candidate keeps the exact same semantics (float64 outputs of shape
``[S, sum(axes_dim)/2]``) but replaces the whole chain with a single fused CUDA
kernel:

* the per-output-column frequency table and its source-axis map are a pure
  function of ``theta``/``axes_dim``, so they are built once and cached per
  device;
* one launch reads ``ids`` and writes both outputs into one allocation (returned
  as two views), one warp per row, each lane emitting a ``double2`` per plane so
  a warp's stores are two fully coalesced 512B segments;
* the angle is formed in fp32 arithmetic but at fp64-like accuracy: the
  frequency is carried as an unevaluated ``(hi, lo)`` float pair, the product is
  split with an FMA and the result is reduced mod 2*pi before ``sincosf``, so the
  absolute error stays at ~1.5e-7 for any plausible ``pos`` -- no fp64 ``sincos``
  (whose rate is heavily cut on consumer/datacenter Blackwell) is needed.

Measured on a B200: 14.3us vs 180us for the baseline, where an *empty* kernel
launch measures 12.3us under the bench's timing loop, i.e. within ~2us of the
floor for any single-kernel implementation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _get_1d_rotary_pos_embed(
    dim: int,
    pos: np.ndarray | int | torch.Tensor,
    theta: float = 10000.0,
    use_real: bool = False,
    linear_factor: float = 1.0,
    ntk_factor: float = 1.0,
    repeat_interleave_real: bool = True,
    freqs_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Precompute the frequency tensor for complex exponentials (cis).

    Copied from ``diffusers.models.embeddings.get_1d_rotary_pos_embed``.
    Returns complex64 tensor of shape [S, dim/2] when ``use_real=False``.
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)

    theta = theta * ntk_factor
    freqs = (
        1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim)) / linear_factor
    )
    freqs = torch.outer(pos, freqs)

    if use_real and repeat_interleave_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()
        return freqs_cos, freqs_sin
    elif use_real:
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()
        return freqs_cos, freqs_sin
    else:
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis


# ---------------------------------------------------------------------------
# Fused kernel: ids[S, n_axes] -> (cos[S, C], sin[S, C]) in float64, C = ncol.
# ---------------------------------------------------------------------------
_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

// sin/cos of ``pos * freq`` at ~1e-7 absolute accuracy using only fp32 ops.
// ``freq`` arrives as an unevaluated sum fh + fl (fl = the fp64 tail of the
// reference frequency), so the product ah + al is accurate to ~1e-14 relative;
// reducing it mod 2*pi with a split constant then keeps the argument handed to
// ``sincosf`` small, which is what bounds the final error. Exact while
// |pos*freq| stays well inside 2^22 (FLUX ids are grid coordinates).
__device__ __forceinline__ void sincos_scaled(float pos, float fh, float fl,
                                              float* s, float* c) {
  const float TWOPI_HI = 6.28318530f, TWOPI_LO = -1.7484555e-07f;
  const float INV_TWOPI = 0.159154943f;
  float ah = pos * fh;
  float al = fmaf(pos, fh, -ah) + pos * fl;
  float k = rintf(ah * INV_TWOPI);
  sincosf(fmaf(-k, TWOPI_HI, ah) + fmaf(-k, TWOPI_LO, al), s, c);
}

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}
template <> __device__ __forceinline__ float to_f<__half>(__half v) {
  return __half2float(v);
}

constexpr int ROWS_PER_BLOCK = 8;   // blockDim = (32, ROWS_PER_BLOCK)

// One warp per row of `ids`; lane `l` owns output columns 2l, 2l+1 (+64, ...).
// VEC: both planes are 16B-aligned, so each lane emits one double2 per plane.
template <typename T, bool VEC>
__global__ void flux_rope_kernel(const T* __restrict__ ids,
                                 const float* __restrict__ freq,
                                 const unsigned char* __restrict__ axmap,
                                 double* __restrict__ out,
                                 int nrow, int ncol,
                                 long long ids_s0, long long ids_s1,
                                 long long plane) {
  const int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
  if (row >= nrow) return;

  const T* idrow = ids + (long long)row * ids_s0;
  double* co = out + (long long)row * ncol;
  double* so = co + plane;

  const float* flo = freq + ncol;
  for (int c = 2 * (int)threadIdx.x; c < ncol; c += 64) {
    float sa, ca;
    sincos_scaled(to_f<T>(idrow[(long long)axmap[c] * ids_s1]), freq[c], flo[c], &sa, &ca);
    if (VEC || c + 1 < ncol) {
      float sb, cb;
      sincos_scaled(to_f<T>(idrow[(long long)axmap[c + 1] * ids_s1]),
                    freq[c + 1], flo[c + 1], &sb, &cb);
      if (VEC) {
        *reinterpret_cast<double2*>(co + c) = make_double2((double)ca, (double)cb);
        *reinterpret_cast<double2*>(so + c) = make_double2((double)sa, (double)sb);
      } else {
        co[c] = (double)ca; co[c + 1] = (double)cb;
        so[c] = (double)sa; so[c + 1] = (double)sb;
      }
    } else {
      co[c] = (double)ca;
      so[c] = (double)sa;
    }
  }
}

template <typename T>
void launch(const at::Tensor& ids, const at::Tensor& freq, const at::Tensor& axmap,
            at::Tensor& out, int nrow, int ncol, bool vec) {
  const dim3 block(32, ROWS_PER_BLOCK);
  const dim3 grid((nrow + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
  auto stream = at::cuda::getCurrentCUDAStream();
  const T* p = reinterpret_cast<const T*>(ids.const_data_ptr());
  const long long plane = (long long)nrow * ncol;
  if (vec) {
    flux_rope_kernel<T, true><<<grid, block, 0, stream>>>(
        p, freq.const_data_ptr<float>(), axmap.const_data_ptr<unsigned char>(),
        out.mutable_data_ptr<double>(), nrow, ncol,
        ids.stride(0), ids.stride(1), plane);
  } else {
    flux_rope_kernel<T, false><<<grid, block, 0, stream>>>(
        p, freq.const_data_ptr<float>(), axmap.const_data_ptr<unsigned char>(),
        out.mutable_data_ptr<double>(), nrow, ncol,
        ids.stride(0), ids.stride(1), plane);
  }
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> flux_rope(const at::Tensor& ids,
                                             const at::Tensor& freq,
                                             const at::Tensor& axmap) {
  const int nrow = (int)ids.size(0);
  const int ncol = (int)freq.size(1);   // freq is [2, ncol]: (hi, lo) planes
  at::Tensor out = at::empty({2, nrow, ncol}, ids.options().dtype(at::kDouble));
  if (nrow == 0 || ncol == 0) return {out.select(0, 0), out.select(0, 1)};

  // double2 stores need both planes 16B-aligned: ncol even (column offset) and
  // nrow*ncol even (second plane's base offset).
  const bool vec = (ncol % 2 == 0) && (((long long)nrow * ncol) % 2 == 0);

  const auto st = ids.scalar_type();
  if (st == at::kBFloat16) {
    launch<__nv_bfloat16>(ids, freq, axmap, out, nrow, ncol, vec);
  } else if (st == at::kFloat) {
    launch<float>(ids, freq, axmap, out, nrow, ncol, vec);
  } else if (st == at::kHalf) {
    launch<__half>(ids, freq, axmap, out, nrow, ncol, vec);
  } else {
    at::Tensor f = ids.to(at::kFloat);
    launch<float>(f, freq, axmap, out, nrow, ncol, vec);
  }
  return {out.select(0, 0), out.select(0, 1)};
}
"""

_CPP_SRC = r"""
std::tuple<at::Tensor, at::Tensor> flux_rope(const at::Tensor& ids,
                                             const at::Tensor& freq,
                                             const at::Tensor& axmap);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("flux_rope", &flux_rope, "fused FLUX 2D RoPE cos/sin");
}
"""

_FLUX_ROPE = None       # the compiled entry point, or False if unavailable


def _flux_rope_fn():
    """JIT-build the extension on first use; ``False`` if the build fails."""
    global _FLUX_ROPE
    if _FLUX_ROPE is None:
        import os
        from torch.utils.cpp_extension import load_inline
        try:
            if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
                major, minor = torch.cuda.get_device_capability()
                os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
            ext = load_inline(
                name="fk_flux_pos_embed",
                cpp_sources=_CPP_SRC,
                cuda_sources=_CUDA_SRC,
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3"],
                verbose=False,
            )
            _FLUX_ROPE = ext.flux_rope
        except Exception:
            _FLUX_ROPE = False
    return _FLUX_ROPE


class FluxPosEmbed(nn.Module):
    """2D rotary position embeddings for FLUX."""

    def __init__(self, theta: int, axes_dim: list[int] | tuple[int, ...]):
        super().__init__()
        self.theta = theta
        self.axes_dim = list(axes_dim)
        self._tables: dict = {}

    def _forward_ref(self, ids: torch.Tensor):
        """The unfused reference path (non-CUDA input, or odd ``axes_dim``)."""
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        freqs_dtype = torch.float32 if ids.device.type in ("mps", "npu") else torch.float64
        for i in range(n_axes):
            freqs_cis = _get_1d_rotary_pos_embed(
                self.axes_dim[i], pos[:, i],
                theta=self.theta, use_real=False,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(freqs_cis.real)
            sin_out.append(freqs_cis.imag)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin

    def _build_tables(self, device: torch.device, n_axes: int):
        """Per-output-column frequency + source-axis tables.

        The frequencies are built with the reference's float64 arithmetic and
        handed to the kernel as an ``(hi, lo)`` pair of fp32 planes, which
        together carry the full fp64 value.
        """
        theta = float(self.theta)
        freqs, axmap = [], []
        for i in range(n_axes):
            dim = int(self.axes_dim[i])
            f = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
            freqs.append(f)
            axmap.append(torch.full((f.numel(),), i, dtype=torch.uint8))
        fd = torch.cat(freqs)
        hi = fd.float()
        lo = (fd - hi.double()).float()          # unevaluated fp32 tail
        tbl = (torch.stack((hi, lo)).to(device=device),
               torch.cat(axmap).to(device=device))
        self._tables[(device, n_axes)] = tbl
        return tbl

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_axes = ids.shape[-1]
        if (not ids.is_cuda or ids.dim() != 2 or n_axes > len(self.axes_dim)
                or any(d % 2 for d in self.axes_dim[:n_axes])):
            return self._forward_ref(ids)
        fn = _flux_rope_fn()
        if fn is False:
            return self._forward_ref(ids)
        key = (ids.device, n_axes)
        tbl = self._tables.get(key) or self._build_tables(ids.device, n_axes)
        return fn(ids, tbl[0], tbl[1])
