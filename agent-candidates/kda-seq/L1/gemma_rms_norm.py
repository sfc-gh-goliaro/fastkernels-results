"""GemmaRMSNorm backed by a single-pass, register-resident CUDA row kernel.

Semantics are vLLM's ``GemmaRMSNorm`` (see ``baseline.py``): the stored weight is
an offset from 1.0, and the cast back to the input dtype happens *after* the
weight multiply (huggingface/transformers#29402).  Per row of the last dimension::

    xf   = x.float()
    var  = sum(xf * xf) / H          # fp32 accumulation, then divide
    xn   = xf * rsqrt(var + eps)
    out  = xn * (1.0 + weight.float())
    store(out)                       # round-to-nearest-even into x.dtype

Why a hand-written kernel: ``torch.compile`` lowers this to Inductor's *looped*
(non-persistent) reduction template ``triton_red_fused_*``, whose own metadata
reports ``num_load: 3, num_store: 1``.  Its first loop accumulates ``sum(x*x)``
and its second loop re-loads exactly the same addresses of ``x`` to apply the
scale, so it issues three units of global load/store traffic where two suffice.
Measured on B200, that redundant unit is *requested* but not fetched from DRAM:
L1 global-load sectors are exactly 2.00x this kernel's for the same shape, while
``dram__bytes_read.sum`` is identical for both, because Triton asks for the first
pass with ``evict_last`` and the second with ``evict_first`` and a block's rows
stay resident in L1.  So the cost it removes is on-chip request traffic and issue
slots, not HBM bandwidth -- worth 1.38x on the kernel in isolation.  See
``profile/phase1_bytes/REPORT.md``.

Layout on the fast path: one warp owns one row.  A row is held as raw 16-byte
packets (``uint4``), converted to fp32 on the fly rather than kept as fp32, so a
2048-wide bf16 row costs 32 registers of payload instead of 64.  Lane ``l``
handles vectors ``l, l+32, l+64, ...`` so every step is a fully coalesced
512-byte warp transaction, and a block of four warps normalizes four rows.  The
reduction itself needs no shared memory and no barrier -- it is a shuffle
butterfly -- but the block does cooperatively stage the weight row into 4 KiB of
dynamic shared memory and ``__syncthreads()`` once before scaling, which is what
keeps a second dependent global round trip off the critical path.

Rows too wide to hold in registers take a second, generic kernel: one 1024-thread
block per row, scalar indexing so any hidden size representable by the kernel's
32-bit row index works (including odd ones), and two passes over the row
(accumulate, then re-read to scale).  It reads ``x`` twice by construction,
which is why it is confined to shapes the register-resident path cannot serve; it
exists so the deliverable degrades gracefully at large ``H`` instead of dropping
to eager PyTorch, and no captured shape reaches it.

Known limits of the kernel path (everything else routes to ``forward_native``,
which is an exact copy of the baseline math, so correctness never depends on the
kernel being applicable):

  - CUDA tensors only, dtype in {bfloat16, float16, float32}, and
    ``weight.dtype == x.dtype``.  A mixed fp32 weight cannot share ``x``'s
    128-bit vector indexing, so it is refused rather than special-cased.
  - ``x`` contiguous with a unit inner stride.  A row-strided input would make
    the kernel read the wrong memory for every row past the first -- the exact
    failure ``fastkernels/tasks/baseline/L1/rms_norm.py`` documents.
  - Either the register-resident geometry -- ``H`` a multiple of the 16-byte
    vector width (8 for bf16/fp16, 4 for fp32), no wider than ``32 * 8`` vectors
    per row (``H <= 2048`` for bf16/fp16, ``H <= 1024`` for fp32), and 16-byte
    aligned ``x`` / ``weight`` / ``out`` -- or, for rows *wider* than that budget,
    the generic block-per-row kernel, which has no divisibility or alignment
    requirement but still needs ``H`` to fit a 32-bit int.  A row that is narrow
    but neither packet-divisible nor aligned falls to PyTorch rather than being
    served by the wide kernel, since it is off every captured shape and not worth
    a third code path.  So does a row wider than ``2**31 - 1``, which no kernel
    here can index.
  - No autograd: ``requires_grad`` under an enabled grad mode takes the native
    path, which is differentiable.
  - ``residual is not None`` takes the native path.  The residual variant is not
    exercised by any captured shape and its fp16 promotion asymmetry is easier to
    keep exactly right in PyTorch.

One numerical caveat, stated because it is real rather than because it is
reachable here: Triton emits ``rsqrt.approx.ftz.f32`` for the reciprocal square
root, while this kernel is built with nvcc's default ``-ftz=false`` and so keeps
the subnormal-handling path around ``MUFU.RSQ``.  For any normal ``var + eps``
both execute the same instruction and agree; but if ``var + eps`` is *subnormal*
-- which needs ``eps == 0`` and a row of magnitude around ``2**-65`` -- the
*compiled* reference flushes to zero and returns infinity where this kernel
returns a finite value.  ``forward_native`` agrees with the kernel there, so the
divergence is specifically against the compiled path.  Every captured case uses
``eps = 1e-6``, which is normal, so it is unreachable through the benched
configuration; ``scratch/selftest.py`` pins the behaviour down explicitly.


The extension is built at *import* time, never lazily inside ``forward``: the
bench harness fails a candidate whose thread count grows across the timed region,
and a first-call JIT compile would do exactly that.  A failed build is not fatal
-- ``_EXT`` stays ``None`` and every input takes the native path.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Build environment.  Set up before the extension is loaded, and derived from
# this file's own location so that sibling agents building a same-named
# extension concurrently cannot pick up each other's artifacts.
# ---------------------------------------------------------------------------

_WORKSPACE = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    "TORCH_EXTENSIONS_DIR", str(_WORKSPACE / ".torch_extensions")
)


def _local_cuda_arch() -> str | None:
    """Compute-capability list for the local device, in ``TORCH_CUDA_ARCH_LIST``
    form.  Mirrors ``fastkernels/infra/cuda_ext.py::_local_cuda_arch``, including
    its mapping of the Hopper/Blackwell majors onto their architecture-specific
    ``a`` variants.  Prefers torch over shelling out to ``nvidia-smi`` so that
    the answer reflects the *visible* device rather than every GPU in the box.
    """
    caps: set[str] = set()
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                major, minor = torch.cuda.get_device_capability(i)
                caps.add(f"{major}.{minor}")
    except Exception:  # noqa: BLE001 - fall through to nvidia-smi
        caps = set()
    if not caps:
        try:
            import subprocess

            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            caps = {c.strip() for c in out.splitlines() if c.strip()}
        except Exception:  # noqa: BLE001 - no device info available
            return None
    mapped = []
    for cap in sorted(caps):
        major = cap.split(".")[0]
        mapped.append(f"{cap}a" if major in ("9", "10", "12") else cap)
    return " ".join(mapped) or None


def _pin_build_arch() -> str | None:
    """Choose the build architecture list, preferring what the caller already set.

    Importing a module must not silently reconfigure the process for every
    extension built afterwards, so an inherited ``TORCH_CUDA_ARCH_LIST`` is left
    exactly as found -- byte for byte, whitespace included. Only when the caller
    has expressed no preference at all does this derive one:

      1. an existing ``TORCH_CUDA_ARCH_LIST`` wins and is returned untouched;
      2. otherwise a non-empty ``FASTKERNELS_CUDA_ARCH_LIST`` is applied verbatim;
      3. otherwise the local device's compute capability is pinned, the way
         ``fastkernels/infra/cuda_ext.py::_local_cuda_arch`` does it.

    Note this deliberately differs from ``cuda_ext._pin_build_arch``, which lets
    ``FASTKERNELS_CUDA_ARCH_LIST`` override an ambient value. Honoring the ambient
    list can be markedly slower to compile -- a container exporting six
    architectures plus PTX multiplies the cold build -- but that is the caller's
    stated intent, and the build is announced and streamed so a slow one is
    visible rather than silent.
    """
    ambient = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if ambient is not None:
        return ambient
    override = os.environ.get("FASTKERNELS_CUDA_ARCH_LIST")
    if override is not None and override.strip():
        os.environ["TORCH_CUDA_ARCH_LIST"] = override
        return override
    arch = _local_cuda_arch()
    if arch:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    return os.environ.get("TORCH_CUDA_ARCH_LIST")


_ARCH = _pin_build_arch()

# ---------------------------------------------------------------------------
# Kernel sources.
# ---------------------------------------------------------------------------

_CPP_SOURCE = r"""
void gemma_rms_norm(at::Tensor& out, const at::Tensor& x,
                    const at::Tensor& weight, double eps);
"""

_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/all.h>

#include <limits>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;   // one row per warp -> four rows per block
constexpr int kMaxVecPerThread = 8; // widest row the register budget covers
constexpr int kGenericBlock = 1024; // one block per row on the wide-row path

// Scalar fp32 -> T store with explicit round-to-nearest-even, matching
// `Tensor.to(dtype)`.  Used by the wide-row path, which indexes elementwise.
template <typename T>
__device__ __forceinline__ T to_rne(float v);

template <>
__device__ __forceinline__ __nv_bfloat16 to_rne<__nv_bfloat16>(float v) {
  return __float2bfloat16_rn(v);
}

template <>
__device__ __forceinline__ __half to_rne<__half>(float v) {
  return __float2half_rn(v);
}

template <>
__device__ __forceinline__ float to_rne<float>(float v) {
  return v;
}

// A 16-byte packet of `T`.  The row is deliberately *kept* as packets rather
// than as unpacked floats: at eight packets per thread that is 32 registers of
// payload instead of 64, which is what keeps occupancy up on a purely
// bandwidth-bound pass.  Both operations below convert to fp32 two elements at a
// time and consume them immediately, so no fp32 view of a packet is ever
// materialized -- staging one through a `float[kElems]` array costs ~30 extra
// registers at VPT=8 and makes ptxas spill at VPT=1.
template <typename T>
struct Packet;

template <>
struct Packet<__nv_bfloat16> {
  static constexpr int kElems = 8;

  __device__ __forceinline__ static void to_float(const uint4& raw, float* f) {
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int i = 0; i < kElems / 2; ++i) {
      const float2 v = __bfloat1622float2(p[i]);
      f[2 * i] = v.x;
      f[2 * i + 1] = v.y;
    }
  }

  // Explicit round-to-nearest-even, matching `Tensor.to(torch.bfloat16)` and the
  // `cvt.rn.bf16.f32` the Triton baseline emits.  Truncating would be cheaper
  // and wrong.
  __device__ __forceinline__ static uint4 from_float(const float* f) {
    uint4 raw;
    __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&raw);
#pragma unroll
    for (int i = 0; i < kElems / 2; ++i) {
      p[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
    }
    return raw;
  }
};

template <>
struct Packet<__half> {
  static constexpr int kElems = 8;

  __device__ __forceinline__ static void to_float(const uint4& raw, float* f) {
    const __half2* p = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
    for (int i = 0; i < kElems / 2; ++i) {
      const float2 v = __half22float2(p[i]);
      f[2 * i] = v.x;
      f[2 * i + 1] = v.y;
    }
  }

  __device__ __forceinline__ static uint4 from_float(const float* f) {
    uint4 raw;
    __half2* p = reinterpret_cast<__half2*>(&raw);
#pragma unroll
    for (int i = 0; i < kElems / 2; ++i) {
      p[i] = __floats2half2_rn(f[2 * i], f[2 * i + 1]);
    }
    return raw;
  }
};

template <>
struct Packet<float> {
  static constexpr int kElems = 4;

  __device__ __forceinline__ static void to_float(const uint4& raw, float* f) {
    f[0] = __uint_as_float(raw.x);
    f[1] = __uint_as_float(raw.y);
    f[2] = __uint_as_float(raw.z);
    f[3] = __uint_as_float(raw.w);
  }

  __device__ __forceinline__ static uint4 from_float(const float* f) {
    return make_uint4(__float_as_uint(f[0]), __float_as_uint(f[1]),
                      __float_as_uint(f[2]), __float_as_uint(f[3]));
  }
};

// One warp per row.  VPT packets per lane cover the row, so a bf16 row of 2048
// elements is 256 packets = 8 per lane, and every lane holds the whole row's
// worth of its own stripe before the reduction starts.
template <typename T, int VPT>
__global__ __launch_bounds__(kWarpSize* kWarpsPerBlock) void gemma_rms_norm_row_kernel(
    T* __restrict__ out, const T* __restrict__ inp, const T* __restrict__ wgt,
    const float eps, const float hidden, const int nvec, const int64_t nrows) {
  constexpr int kElems = Packet<T>::kElems;

  const int lane = threadIdx.x;
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + threadIdx.y;
  // Masked rather than returned early: staging the weight below is a whole-block
  // cooperative load, so every thread has to reach its __syncthreads.  A whole
  // warp shares one row, so `active` is warp-uniform and the full shuffle mask
  // stays valid for every warp that survives past the barrier.
  const bool active = row < nrows;

  const int64_t offset = (active ? row : 0) * static_cast<int64_t>(nvec);
  const uint4* __restrict__ xv = reinterpret_cast<const uint4*>(inp) + offset;
  uint4* __restrict__ ov = reinterpret_cast<uint4*>(out) + offset;
  const uint4* __restrict__ wv = reinterpret_cast<const uint4*>(wgt);

  // The one and only read of the row.  VPT independent 16-byte loads per lane
  // are what give this kernel its memory-level parallelism.
  uint4 buf[VPT];
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    const int v = lane + i * kWarpSize;
    buf[i] = (active && v < nvec) ? xv[v] : make_uint4(0u, 0u, 0u, 0u);
  }

  // Stage the weight row in shared memory while those x loads are still in
  // flight.  Reading the weight down in the scaling loop instead would put a
  // second *dependent* global round trip on the critical path -- load x, reduce,
  // load weight, store -- which measures as a flat ~2 us penalty on the small row
  // counts, where this kernel is latency- rather than bandwidth-bound.  Holding
  // it in registers instead fixes that but pushes VPT=8 from 48 to 80 registers
  // and gives back most of the win on the largest shape.  Shared memory gets the
  // early issue without the register cost, and one copy per block means the
  // block's warps share the load instead of each repeating it.
  extern __shared__ uint4 smem_weight[];
  for (int v = threadIdx.y * kWarpSize + lane; v < nvec;
       v += kWarpSize * kWarpsPerBlock) {
    smem_weight[v] = wv[v];
  }
  __syncthreads();

  if (!active) return;

  float acc = 0.0f;
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    float f[kElems];
    Packet<T>::to_float(buf[i], f);
#pragma unroll
    for (int k = 0; k < kElems; ++k) acc += f[k] * f[k];
  }

  // Butterfly so every lane ends with the row sum: no extra shared memory, and
  // no need to broadcast back from a single lane.
#pragma unroll
  for (int shift = kWarpSize / 2; shift > 0; shift >>= 1) {
    acc += __shfl_xor_sync(0xffffffffu, acc, shift);
  }

  // Divide (not multiply by a reciprocal) then add eps then rsqrt, matching the
  // reference chain exactly.
  const float rs = rsqrtf(acc / hidden + eps);

#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    const int v = lane + i * kWarpSize;
    if (v >= nvec) continue;
    float f[kElems];
    float w[kElems];
    Packet<T>::to_float(buf[i], f);
    Packet<T>::to_float(smem_weight[v], w);
#pragma unroll
    for (int k = 0; k < kElems; ++k) {
      // Two separate multiplies in this order, matching the reference's
      // `(x * rsqrt) * (1 + w)`.  Pre-folding rs into (1 + w), or distributing
      // the outer multiply into an fma, would be a different operator.  Plain `*`
      // already compiles that way without `--use_fast_math`, but that is a
      // property of the current flags rather than a guarantee, so pin it: the
      // `_rn` intrinsics are round-to-nearest and are not contractable.
      f[k] = __fmul_rn(__fmul_rn(f[k], rs), 1.0f + w[k]);
    }
    ov[v] = Packet<T>::from_float(f);
  }
}

// Generality path for rows too wide to hold in registers.  One block per row,
// scalar indexing so any hidden size works including odd ones, and two passes
// over the row: accumulate, then re-read to scale.  It *does* read `x` twice --
// that is the price of not having the row resident -- so it is deliberately kept
// off the measured path, where the whole point is reading once.  Present so the
// deliverable degrades gracefully at large `H` instead of falling back to eager
// PyTorch; validated by the self-test only, never by a benched shape.
template <typename T>
__global__ __launch_bounds__(kGenericBlock) void gemma_rms_norm_wide_kernel(
    T* __restrict__ out, const T* __restrict__ inp, const T* __restrict__ wgt,
    const float eps, const float hidden, const int64_t nrows, const int hidden_i) {
  const int64_t row = blockIdx.x;
  if (row >= nrows) return;

  const T* __restrict__ xrow = inp + row * static_cast<int64_t>(hidden_i);
  T* __restrict__ orow = out + row * static_cast<int64_t>(hidden_i);

  float acc = 0.0f;
  for (int j = threadIdx.x; j < hidden_i; j += kGenericBlock) {
    const float v = static_cast<float>(xrow[j]);
    acc += v * v;
  }

  // Reduce within each warp, then across warps through shared memory.
#pragma unroll
  for (int shift = kWarpSize / 2; shift > 0; shift >>= 1) {
    acc += __shfl_xor_sync(0xffffffffu, acc, shift);
  }
  __shared__ float warp_sums[kGenericBlock / kWarpSize];
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  if (lane == 0) warp_sums[warp] = acc;
  __syncthreads();

  // One warp folds the per-warp sums, then the result is broadcast to the block.
  __shared__ float row_rsqrt;
  if (threadIdx.x == 0) {
    float total = 0.0f;
#pragma unroll
    for (int i = 0; i < kGenericBlock / kWarpSize; ++i) total += warp_sums[i];
    row_rsqrt = rsqrtf(total / hidden + eps);
  }
  __syncthreads();
  const float rs = row_rsqrt;

  for (int j = threadIdx.x; j < hidden_i; j += kGenericBlock) {
    const float v = static_cast<float>(xrow[j]);
    const float w = static_cast<float>(wgt[j]);
    orow[j] = to_rne<T>(__fmul_rn(__fmul_rn(v, rs), 1.0f + w));
  }
}

template <typename T>
void launch_wide_kernel(at::Tensor& out, const at::Tensor& x,
                        const at::Tensor& weight, double eps, int64_t nrows,
                        int hidden, cudaStream_t stream) {
  TORCH_CHECK(nrows <= 2147483647LL, "row count ", nrows, " exceeds grid limit");
  gemma_rms_norm_wide_kernel<T>
      <<<static_cast<unsigned int>(nrows), kGenericBlock, 0, stream>>>(
          reinterpret_cast<T*>(out.data_ptr()),
          reinterpret_cast<const T*>(x.data_ptr()),
          reinterpret_cast<const T*>(weight.data_ptr()),
          static_cast<float>(eps), static_cast<float>(hidden), nrows, hidden);
}

template <typename T>
void launch_row_kernel(at::Tensor& out, const at::Tensor& x,
                       const at::Tensor& weight, double eps, int64_t nrows,
                       int hidden, cudaStream_t stream) {
  constexpr int kElems = Packet<T>::kElems;
  TORCH_CHECK(hidden % kElems == 0, "hidden size ", hidden,
              " is not a multiple of the ", kElems, "-element vector width");
  const int nvec = hidden / kElems;
  TORCH_CHECK(nvec <= kWarpSize * kMaxVecPerThread, "hidden size ", hidden,
              " exceeds the register-resident row budget");

  const int64_t nblocks = (nrows + kWarpsPerBlock - 1) / kWarpsPerBlock;
  TORCH_CHECK(nblocks <= 2147483647LL, "row count ", nrows, " exceeds grid limit");
  const dim3 grid(static_cast<unsigned int>(nblocks));
  const dim3 block(kWarpSize, kWarpsPerBlock);

  // One 16-byte packet of weight per row vector; 4 KiB at the widest row the
  // register budget admits, so this never approaches the 48 KiB default limit.
  const size_t smem_bytes = static_cast<size_t>(nvec) * sizeof(uint4);
  TORCH_CHECK(smem_bytes <= 48u * 1024u, "weight staging needs ", smem_bytes,
              " bytes of shared memory, above the 48 KiB default limit");

  T* out_p = reinterpret_cast<T*>(out.data_ptr());
  const T* x_p = reinterpret_cast<const T*>(x.data_ptr());
  const T* w_p = reinterpret_cast<const T*>(weight.data_ptr());
  const float epsf = static_cast<float>(eps);
  const float hiddenf = static_cast<float>(hidden);

  const int vpt = (nvec + kWarpSize - 1) / kWarpSize;

#define FK_LAUNCH_VPT(VPT)                                                 \
  gemma_rms_norm_row_kernel<T, VPT><<<grid, block, smem_bytes, stream>>>(  \
      out_p, x_p, w_p, epsf, hiddenf, nvec, nrows)

  if (vpt <= 1) {
    FK_LAUNCH_VPT(1);
  } else if (vpt <= 2) {
    FK_LAUNCH_VPT(2);
  } else if (vpt <= 4) {
    FK_LAUNCH_VPT(4);
  } else {
    FK_LAUNCH_VPT(8);
  }
#undef FK_LAUNCH_VPT
}

inline bool is_16b_aligned(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

// The register-resident path needs the row to divide into 16-byte packets, to fit
// the register budget, and to be 16-byte aligned.  Anything wider goes to the
// generic path; anything else never reaches the extension (the Python guard sends
// it to the exact PyTorch implementation instead).
template <typename T>
bool row_kernel_applicable(const at::Tensor& out, const at::Tensor& x,
                           const at::Tensor& weight, int hidden) {
  constexpr int kElems = Packet<T>::kElems;
  if (hidden % kElems != 0) return false;
  if (hidden / kElems > kWarpSize * kMaxVecPerThread) return false;
  return is_16b_aligned(x.data_ptr()) && is_16b_aligned(weight.data_ptr()) &&
         is_16b_aligned(out.data_ptr());
}

template <typename T>
void launch_typed(at::Tensor& out, const at::Tensor& x, const at::Tensor& weight,
                  double eps, int64_t nrows, int hidden, cudaStream_t stream) {
  if (row_kernel_applicable<T>(out, x, weight, hidden)) {
    launch_row_kernel<T>(out, x, weight, eps, nrows, hidden, stream);
  } else {
    launch_wide_kernel<T>(out, x, weight, eps, nrows, hidden, stream);
  }
}

}  // namespace

void gemma_rms_norm(at::Tensor& out, const at::Tensor& x,
                    const at::Tensor& weight, double eps) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda() && out.is_cuda(),
              "x, weight and out must all be CUDA tensors");
  // The launch runs under out's device guard, so a tensor living on another
  // device would have its pointer dereferenced by the wrong context.  The Python
  // guard already refuses this, but the binding is reachable directly.
  TORCH_CHECK(x.device() == out.device() && weight.device() == out.device(),
              "x, weight and out must be on the same CUDA device (got ",
              x.device(), ", ", weight.device(), ", ", out.device(), ")");
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() && out.is_contiguous(),
              "x, weight and out must all be contiguous");
  TORCH_CHECK(out.scalar_type() == x.scalar_type(),
              "out dtype ", out.scalar_type(), " != x dtype ", x.scalar_type());
  TORCH_CHECK(weight.scalar_type() == x.scalar_type(),
              "weight dtype ", weight.scalar_type(), " != x dtype ",
              x.scalar_type());
  TORCH_CHECK(out.sizes() == x.sizes(), "out shape must match x");
  TORCH_CHECK(x.dim() >= 1, "x must have at least one dimension");
  TORCH_CHECK(weight.dim() == 1, "weight must be 1-D");

  // Both kernels index within a row with `int`, so the row width has to fit one.
  // Checked *before* the narrowing cast: a silent wrap would make `hidden`
  // negative, `nrows` zero, and this function return without launching, handing
  // back the uninitialized `torch::empty_like` buffer as if it were the answer.
  // The Python guard already refuses this, but the binding is reachable directly.
  TORCH_CHECK(x.size(-1) <= std::numeric_limits<int>::max(),
              "x.size(-1) ", x.size(-1), " exceeds the kernel index range (",
              std::numeric_limits<int>::max(), ")");
  const int hidden = static_cast<int>(x.size(-1));
  TORCH_CHECK(hidden == weight.numel(), "x.size(-1) ", hidden,
              " != weight.numel() ", weight.numel());
  const int64_t nrows = hidden > 0 ? x.numel() / hidden : 0;
  if (nrows == 0 || hidden == 0) return;

  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(out));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (x.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_typed<__nv_bfloat16>(out, x, weight, eps, nrows, hidden, stream);
      break;
    case at::ScalarType::Half:
      launch_typed<__half>(out, x, weight, eps, nrows, hidden, stream);
      break;
    case at::ScalarType::Float:
      launch_typed<float>(out, x, weight, eps, nrows, hidden, stream);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype ", x.scalar_type());
  }
  // Non-synchronizing: turns a bad launch configuration into a Python
  // exception here instead of a confusing failure at the next sync.
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

# Deliberately *not* `--use_fast_math`: it would let nvcc reassociate
# `(xf * rs) * (1 + wf)` into a single fused scale, which is a different
# operator.  Mirrors the relevant subset of
# ``fastkernels/infra/cuda_ext.py::_BASE_CUDA_CFLAGS``.
_CUDA_CFLAGS = [
    "-O3",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "--expt-relaxed-constexpr",
    # Line-table metadata only: it does not change codegen or register allocation
    # (verified -- bf16 VPT=8 stays at 95 registers with zero spills), but without
    # it Nsight Compute cannot map SASS back to source, so no per-line stall
    # attribution is possible for the *shipped* binary. That attribution is
    # required evidence, and it cannot be added retroactively to a report.
    "-lineinfo",
]
_CFLAGS = ["-O3"]

# The build directory is keyed by extension name, so fold the sources, flags,
# torch build and target architecture into it.  A stale `.so` from an older
# revision of this file can then never be mistaken for a current one, and two
# workspaces with different sources never contend for one directory.
_DIGEST = hashlib.sha256(
    "\0".join(
        [
            _CPP_SOURCE,
            _CUDA_SOURCE,
            *_CUDA_CFLAGS,
            *_CFLAGS,
            torch.__version__,
            _ARCH or "auto",
        ]
    ).encode()
).hexdigest()[:12]
_EXT_NAME = f"fk_cand_gemma_rmsnorm_{_DIGEST}"


def _build_extension():
    """Compile and load the extension, announcing a pending build first.

    The bench worker is killed after 600 s without output, so a cold compile
    must not be silent: print one flushed line up front and let ninja stream its
    per-file progress.  A warm cache skips both.
    """
    from torch.utils.cpp_extension import _get_build_directory, load_inline

    pending = True
    try:
        so = Path(_get_build_directory(_EXT_NAME, verbose=False)) / f"{_EXT_NAME}.so"
        pending = not so.exists()
    except Exception:  # noqa: BLE001 - probing the cache must never be fatal
        pending = True

    cuda_cflags = list(_CUDA_CFLAGS)
    if os.environ.get("FK_GEMMA_RMSNORM_BREAK_BUILD") == "1":
        # Test hook: make nvcc reject the translation unit so the failed-build
        # path is exercised for real rather than mocked.
        cuda_cflags.append("--this-flag-does-not-exist")
        pending = True

    if pending:
        print(
            f"[gemma_rms_norm] building CUDA extension {_EXT_NAME!r} for arch "
            f"{os.environ.get('TORCH_CUDA_ARCH_LIST', 'auto')!r} -- one-time JIT "
            f"compile, streaming ninja progress ...",
            flush=True,
        )

    return load_inline(
        name=_EXT_NAME,
        cpp_sources=_CPP_SOURCE,
        cuda_sources=_CUDA_SOURCE,
        functions=["gemma_rms_norm"],
        extra_cflags=_CFLAGS,
        extra_cuda_cflags=cuda_cflags,
        verbose=pending,
    )


# Built at import, never inside `forward`.
if os.environ.get("FK_GEMMA_RMSNORM_EXT") == "0":
    _EXT = None
    _EXT_ERROR = "disabled by FK_GEMMA_RMSNORM_EXT=0"
else:
    try:
        _EXT = _build_extension()
        _EXT_ERROR = None
    except Exception as exc:  # noqa: BLE001 - a failed build must not be fatal
        _EXT = None
        _EXT_ERROR = f"{type(exc).__name__}: {exc}"
        print(
            f"[gemma_rms_norm] CUDA extension unavailable ({_EXT_ERROR}); every "
            f"input will take the pure-PyTorch path",
            file=sys.stderr,
            flush=True,
        )

# Elements per 16-byte vector, and the widest row the kernel covers.
_VEC_ELEMS = {torch.bfloat16: 8, torch.float16: 8, torch.float32: 4}
_MAX_VECTORS_PER_ROW = 32 * 8
# Both kernels index within a row with a C++ `int`, so this is the widest row
# either can address. Anything past it takes the pure-PyTorch path.
_MAX_HIDDEN = 2**31 - 1


class GemmaRMSNorm(nn.Module):
    """RMSNorm with the weight stored as an offset from 1.0 (Gemma convention).

    Nothing derived from ``weight`` is cached anywhere.  The bench harness calls
    ``load_state_dict(..., strict=False)`` *after* construction, which would
    silently leave a precomputed ``(1 + weight)`` stale, so ``forward`` reads
    ``self.weight`` on every call.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    # -- Pure PyTorch path: an exact copy of the baseline math -----------------

    @staticmethod
    def _norm_no_residual(
        weight: torch.Tensor,
        variance_epsilon: float,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + variance_epsilon)
        x = x * (1.0 + weight.float())
        return x.to(orig_dtype)

    @staticmethod
    def _norm_with_residual(
        weight: torch.Tensor,
        variance_epsilon: float,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        # fp16 is the only dtype whose residual add is promoted; the returned
        # residual is the *rounded* sum while the normalization below consumes
        # the unrounded fp32 one.
        x = (
            x.float() + residual.float()
            if orig_dtype == torch.float16
            else x + residual
        )
        residual = x.to(orig_dtype) if x.dtype != orig_dtype else x

        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + variance_epsilon)
        x = x * (1.0 + weight.float())
        return x.to(orig_dtype), residual

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self._norm_no_residual(
                self.weight.data, self.variance_epsilon, x,
            )
        return self._norm_with_residual(
            self.weight.data, self.variance_epsilon, x, residual,
        )

    # -- CUDA kernel path -----------------------------------------------------

    def _kernel_mode(self, x: torch.Tensor, residual: torch.Tensor | None) -> str | None:
        """Which kernel this input is eligible for: ``"row"``, ``"wide"``, or None.

        ``"row"`` is the register-resident single-read path that every captured
        shape takes. ``"wide"`` is the generic block-per-row path, used only when
        the row exceeds the register budget. None means the exact pure-PyTorch
        implementation handles it.
        """
        if _EXT is None or residual is not None:
            return None
        if not x.is_cuda or x.dim() < 1 or x.numel() == 0:
            return None
        vec = _VEC_ELEMS.get(x.dtype)
        if vec is None:
            return None
        weight = self.weight.data
        if weight.dtype is not x.dtype or weight.device != x.device:
            return None
        hidden = x.size(-1)
        if hidden != weight.numel():
            return None
        # Both kernels index within a row with a 32-bit int.
        if hidden > _MAX_HIDDEN:
            return None
        if not x.is_contiguous() or not weight.is_contiguous():
            return None
        if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
            return None
        aligned = x.data_ptr() % 16 == 0 and weight.data_ptr() % 16 == 0
        budget = vec * _MAX_VECTORS_PER_ROW
        if hidden % vec == 0 and hidden <= budget and aligned:
            return "row"
        if hidden > budget:
            # Too wide to hold in registers; the generic kernel indexes
            # elementwise, so an odd hidden size and a misaligned pointer are both
            # fine here.
            return "wide"
        # Narrow but awkward -- not packet-divisible, or misaligned. Rare, off
        # every captured shape, and exactly what the PyTorch path is for.
        return None

    def _kernel_eligible(
        self, x: torch.Tensor, residual: torch.Tensor | None
    ) -> bool:
        """Whether any CUDA kernel handles this input."""
        return self._kernel_mode(x, residual) is not None

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        mode = self._kernel_mode(x, residual)
        if mode is not None:
            # Fresh storage: the result must never be written over x, whose
            # buffer the harness reuses across timed iterations.
            out = torch.empty_like(x, memory_format=torch.contiguous_format)
            # The register-resident path indexes `out` in 16-byte packets; the
            # generic path does not care.  Caching allocator blocks are far more
            # aligned than this, so the check is insurance, not a real branch.
            if mode == "wide" or out.data_ptr() % 16 == 0:
                _EXT.gemma_rms_norm(
                    out, x, self.weight.data, self.variance_epsilon,
                )
                return out
        return self.forward_native(x, residual)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if torch.compiler.is_compiling():
            return self.forward_native(x, residual)
        return self.forward_cuda(x, residual)
