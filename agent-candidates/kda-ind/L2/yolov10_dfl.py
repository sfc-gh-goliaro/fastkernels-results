"""YOLOv10 Distribution Focal Loss layer, fused into a single CUDA kernel.

The baseline composes three kernels: a softmax over the ``c1`` axis, a copy that
makes the transposed view contiguous, and a cuBLAS 1x1 convolution contracting
that axis against ``conv.weight``.  Elementwise the whole thing is just

    out[b, j, i] = sum_k w[k] * softmax_k( x[b, j*c1 + k, i] )

the expected value of a per-position softmax distribution over ``c1`` channels --
the standard DFL "integral" step.

``x`` is row-major in its last two axes, so element ``(n, j*c1+k, i)`` sits at
``n*stride0 + (j*c1 + k)*a + i``.  Grouping ``g = n*4 + j`` makes the operation
exactly: given ``G = 4*b`` blocks of ``(C, A)`` row-major values, produce a
contiguous ``(G, A)`` tensor by a softmax-weighted average over ``C``.  The
baseline's ``transpose`` needs no data movement, and the resulting ``(G, A)``
buffer *is* the ``(b, 4, a)`` output, bit-layout identical.

Note that ``x`` is *not* fully contiguous in the captured workload: the recorded
stride is ``[1209600, 8400, 1]`` for shape ``[4, 64, 8400]``, i.e. these 64 DFL
channels are a slice of a 144-channel head tensor, and consecutive batch elements
sit ``144*8400`` apart rather than ``64*8400``.  The kernel therefore takes the
batch stride as an argument; only row-major-ness within a batch element is
required, which is what the capture actually provides.

On the captured workload ``validate.py`` reports a maximum absolute error of exactly
0, i.e. the output is bit-identical there.  That is deliberate, not incidental.  But it
is **not** a general guarantee: the contraction is a serial FMA chain over ``c1``, while
cuBLAS uses a different reduction tree, and fp32 accumulation does not make different
summation orders agree bit-for-bit.  With ``[+65504]*8 + [-65504]*8`` weights and random
logits, about 3 % of outputs differ from the baseline by one fp16 ulp -- comfortably
inside the harness metric, but not identical.  What *is* matched exactly is the
sequence of intermediate roundings, and three of them are observable:

* The baseline's softmax emits a tensor in the input dtype, so every one of the
  ``c1`` probabilities is rounded to fp16/bf16 *before* the convolution contracts it.
  Skipping that rounding looks harmless -- fp32 throughout is nominally more accurate
  -- but it is not: where the weighted sum cancels, the baseline's rounding collapses
  near-equal probabilities to identical values so the cancellation is exact, while an
  fp32 probability keeps a tiny difference that a large weight then amplifies.  With
  ``conv.weight = [+65504]*8 + [-65504]*8`` (ordinary, legal fp16) and a logit offset
  of 0.01, contracting unrounded probabilities gave 327.5 against the baseline's
  319.75 -- ``matched_ratio`` 0.0.
* The exponential is the accurate ``expf`` rather than ``ex2.approx.f32``.  ATen's
  softmax epilogue is ``std::exp(input - max) / sum`` in fp32, and on this build the
  two agree; that agreement is empirical, not something the APIs promise across
  architectures.  The approximate form costs 2 ulp, which survives the same
  cancellation: 40 of 400 randomised trials over dtype-maximum opposing weights fell
  below ``matched_ratio`` 1.0.
* The division ``e / den`` uses a Newton-corrected reciprocal (see
  ``divide_probability``), which measured exact on every trial run here; a bare
  reciprocal multiply left 48 of those 400 trials short.

Only the *contraction* is accumulated in fp32, which is what cuBLAS does for the
baseline's fp16 1x1 convolution anyway -- except where the weights are large enough
that the output can reach the dtype's overflow threshold, in which case it accumulates
in fp64 so that whether the result is finite or infinite follows the exact sum rather
than the accumulation order.  Overflow is never saturated away: the rounded
probabilities do not sum to one, so with every weight at the dtype maximum the exact
result really does exceed the range and the baseline really does return an infinity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Elements handled per thread per row, and threads per block.
#
# One element per thread, chosen on measurement rather than by argument.  The wide
# alternative -- eight halves, a single 16-byte load -- looks better on paper and was
# the original default.  It is 18% slower here, and the cause is *not* isolated: moving
# from V=8 to V=1 changes registers (167 -> 32), the occupancy limit (3 -> 16
# blocks/SM), the grid (144 -> 1056 blocks at b=4), achieved occupancy (5.8% -> 35.9%)
# and the front-end footprint (10624 -> 2304 fill sectors) all at once, and at V=8 the
# grid held only 144 blocks for 148 SMs so the register cap was not even binding.
# Separating them needs a V=2 or V=4 instantiation, which is not compiled.  What is
# clear is that the gap appeared only once the exponential became the accurate `expf`;
# with `ex2.approx` every configuration was within 1%.  Measured through `validate.py`:
#
#     V = 1, BLOCK = 128:  b=4 0.0092 ms (3.23x)   b=1 0.0154 ms (2.07x)
#     V = 8, BLOCK = 128:  b=4 0.0113 ms (2.63x)   b=1 0.0174 ms (1.71x)
#
# BLOCK is flat across 64/128/256 at V = 1; 128 is the middle of the measured range.
# The 16-byte instantiation is kept and still reachable through the C++ entry point,
# because the (V, BLOCK) comparison in the profile report depends on it.
_VEC = 1
_BLOCK = 128

# Channel count the CUDA template is instantiated for; anything else falls back.
_FUSED_CHANNELS = 16
# blockIdx.y selects the group, so 4*b must fit the grid's y extent.
_MAX_GRID_Y = 65535


def _has_forward_tangent(t: torch.Tensor) -> bool:
    """Whether *t* carries a forward-mode AD tangent the kernel would discard."""
    try:
        return torch.autograd.forward_ad.unpack_dual(t).tangent is not None
    except Exception:  # noqa: BLE001 - no dual level active, or an old torch
        return False

_CPP_SOURCE = r"""
#include <torch/extension.h>

at::Tensor dfl_integral(const at::Tensor& x, const at::Tensor& w, int64_t vec,
                        int64_t block, int64_t batch_stride);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dfl_integral", &dfl_integral, "Fused DFL softmax integral");
}
"""

_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <limits>

namespace {
// DEVICE-CODE-BEGIN  (tools/gen_profile_harness.py slices between these markers so
// the profiling harness compiles the exact device code that ships)

constexpr int kChannels = 16;
// The DFL layer always splits its channels into four boxes; a power of two, so
// the group index decomposes with a shift and a mask rather than a division.
constexpr int kGroupsPerBatch = 4;
constexpr int kGroupShift = 2;

__device__ __forceinline__ float widen(const __half v) { return __half2float(v); }
__device__ __forceinline__ float widen(const __nv_bfloat16 v) {
  return __bfloat162float(v);
}
__device__ __forceinline__ void narrow(__half& dst, const float v) {
  dst = __float2half_rn(v);
}
__device__ __forceinline__ void narrow(__nv_bfloat16& dst, const float v) {
  dst = __float2bfloat16_rn(v);
}

// Straight from fp64, rounding once.  Going through fp32 first would round twice, and
// the second rounding can manufacture an infinity: the exact sum -65519.999996 lies
// just inside fp16's -65520 overflow threshold and converts to -65504, but rounded to
// fp32 it becomes exactly -65520.0, which then converts to -inf.  The baseline returns
// -65504 there, and so must this.
__device__ __forceinline__ void narrow(__half& dst, const double v) {
  dst = __double2half(v);
}
__device__ __forceinline__ void narrow(__nv_bfloat16& dst, const double v) {
  dst = __double2bfloat16(v);
}

// num/den for 0 < num <= den, matching a correctly-rounded fp32 division in three
// instructions rather than the dozen a full-precision divide expands to.
//
// The quotient is a softmax probability that then gets rounded to T, and where the
// weighted sum cancels, any error here is multiplied by a large weight instead of
// cancelling.  Measured over 400 randomised trials with dtype-maximum opposing
// weights: a plain reciprocal-multiply leaves 48 of 400 below matched_ratio 1.0
// (worst 0.997), and this form leaves none.  A true `/` is equally exact but cost
// b = 4 about 0.6x of speedup for no accuracy gain over this.
//
// The restricted domain is what makes three instructions sufficient: num is in [0, 1]
// and den is in [1, C], so there is no negative, infinite or divide-by-zero case to
// handle.  num *can* be zero or subnormal, when a logit is far enough below the
// maximum that the exponential underflows; the sequence handles both -- the products
// and the residual are then zero or subnormal too, and the result is the correctly
// rounded tiny quotient.  What is not claimed is correct rounding in general: it
// measured exact on every trial run here, which is evidence rather than proof.
__device__ __forceinline__ float divide_probability(const float num, const float den,
                                                   const float inv_den) {
  const float q = num * inv_den;
  return fmaf(fmaf(-q, den, num), inv_den, q);
}

// Largest finite value of T.  Used only to decide when the contraction needs more
// precision than fp32 -- never to saturate the result.  Saturating would be wrong:
// the rounded probabilities do not sum to one (measured 1.0002737 for one fp16 case),
// so with every weight at the dtype maximum the exact result genuinely exceeds the
// dtype's range and the baseline correctly returns an infinity.
template <typename T>
struct MaxFinite;
template <>
struct MaxFinite<__half> {
  static constexpr float value = 65504.0f;
};
template <>
struct MaxFinite<__nv_bfloat16> {
  static constexpr float value = 3.3895314e38f;
};

// V elements of T, aligned to the load width so a whole Vec moves in one
// instruction.  Capped at 16 bytes; V = 1 stays naturally aligned so the general
// instantiation can run on pointers the wide one would reject.
template <typename T, int V>
constexpr int vec_align() {
  return (static_cast<int>(sizeof(T)) * V > 16) ? 16 : static_cast<int>(sizeof(T)) * V;
}

template <typename T, int V>
struct alignas(vec_align<T, V>()) Vec {
  T v[V];
};

// Softmax-weighted average over the C axis of each of G row-major (C, A) blocks,
// writing a contiguous (G, A) result.  Blocks are four to a batch element, and
// batch elements are batch_stride apart -- which is not necessarily 4*C*A, since
// the captured input is a channel slice of a wider tensor.
//
// The grid is 2-D: blockIdx.y selects the group and blockIdx.x tiles the A axis,
// which avoids any device-side division by the non-power-of-two A.  Each thread
// owns V contiguous columns and walks all C rows of them, so a warp reads
// 32*V*sizeof(T) contiguous bytes per row and drives C independent coalesced
// streams -- C loads in flight per thread is the memory-level parallelism this
// latency-bound shape needs.
//
// Values are held in their packed 16-bit form across the three passes (max,
// denominator, contraction) and widened on use, keeping the per-thread data footprint
// at C*V halves rather than C*V floats.
template <typename T, int C, int V>
__global__ void dfl_integral_kernel(const T* __restrict__ x,
                                    const T* __restrict__ w,
                                    T* __restrict__ out,
                                    int a_size,
                                    long long batch_stride) {
  // The C weights are identical for every thread in the block; stage them once.
  __shared__ float ws[C];
  if (static_cast<int>(threadIdx.x) < C) {
    ws[threadIdx.x] = widen(w[threadIdx.x]);
  }
  __syncthreads();

  // Largest weight magnitude in the block, read from the staged values.  Every thread
  // derives the same number, so the branch it selects below is block-uniform and never
  // divergent.
  float w_absmax = 0.0f;
#pragma unroll
  for (int k = 0; k < C; ++k) {
    w_absmax = fmaxf(w_absmax, fabsf(ws[k]));
  }
  // The exact result is bounded by max|w| * sum(p), and sum(p) is 1 plus a few
  // roundings of T's epsilon -- so the output can only approach T's overflow threshold
  // when max|w| is itself within a small factor of it.  A quarter of the range is a
  // conservative cutoff; the benchmarked weights are two orders of magnitude
  // smaller still and always take the fp32 path.
  const bool high_weight = w_absmax > MaxFinite<T>::value * 0.25f;

  const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) * V;
  if (col >= a_size) {
    return;
  }

  const int group = static_cast<int>(blockIdx.y);
  const T* __restrict__ src = x
      + static_cast<long long>(group >> kGroupShift) * batch_stride
      + static_cast<long long>(group & (kGroupsPerBatch - 1)) * C * a_size
      + col;

  Vec<T, V> packed[C];
#pragma unroll
  for (int k = 0; k < C; ++k) {
    packed[k] = *reinterpret_cast<const Vec<T, V>*>(
        src + static_cast<long long>(k) * a_size);
  }

  float m[V];
#pragma unroll
  for (int i = 0; i < V; ++i) {
    m[i] = widen(packed[0].v[i]);
  }
#pragma unroll
  for (int k = 1; k < C; ++k) {
#pragma unroll
    for (int i = 0; i < V; ++i) {
      m[i] = fmaxf(m[i], widen(packed[k].v[i]));
    }
  }

  // Denominator first, on its own, because each probability has to be rounded to T
  // before it is contracted and that cannot happen until den is known.
  //
  // `expf`, the accurate exponential, not `ex2.approx.f32`: ATen's softmax epilogue
  // is `std::exp(input - max) / sum` in fp32, so using the same function makes this
  // kernel agree with the baseline to the last bit.  With `ex2.approx` (2 ulp) the
  // captured shapes still matched at max_abs = one fp16 ulp, but 40 of 400 randomised
  // cancelling-weight trials fell below matched_ratio 1.0.  With `expf` the captured
  // shapes report max_abs = 0 -- bit-identical output -- and none of the 400 deviate.
  // Cost, measured: b = 4 goes 3.22x -> 2.64x, b = 1 goes 1.95x -> 1.67x.
  float den[V];
#pragma unroll
  for (int i = 0; i < V; ++i) {
    den[i] = 0.0f;
  }
#pragma unroll
  for (int k = 0; k < C; ++k) {
#pragma unroll
    for (int i = 0; i < V; ++i) {
      den[i] += expf(widen(packed[k].v[i]) - m[i]);
    }
  }
  float inv_den[V];
#pragma unroll
  for (int i = 0; i < V; ++i) {
    inv_den[i] = __frcp_rn(den[i]);  // den >= 1, so this cannot overflow
  }
  // Contract the *rounded* probabilities.  The baseline's softmax emits a tensor in
  // the input dtype, so every probability it feeds the convolution has been rounded
  // to T; that rounding is what makes a cancelling weighted sum cancel exactly, and
  // skipping it changes the operator's result for large opposing weights.  So round
  // e/den to T, widen it back, and accumulate w*p in fp32 -- which is what cuBLAS
  // does for the baseline's fp16 1x1 convolution.  The exponential is recomputed
  // rather than carried over from the pass above: holding C*V floats would add 128
  // registers at V = 8, which is far more than the second `expf` costs in time.
  Vec<T, V> res;
  if (high_weight) {
    // Weights large enough that the output can land near T's overflow threshold, where
    // an fp32 sum's rounding error is comparable to the gap between T's largest finite
    // value and infinity.  Accumulating in fp64 makes the finite-versus-infinite
    // decision follow the exact sum rather than the accumulation order, so a
    // legitimate overflow stays an infinity and a spurious one does not become one.
    // The conversion is the ordinary IEEE one, straight from fp64 -- nothing is
    // saturated, and nothing is rounded twice.
    double num[V];
#pragma unroll
    for (int i = 0; i < V; ++i) {
      num[i] = 0.0;
    }
#pragma unroll
    for (int k = 0; k < C; ++k) {
      const double wk = static_cast<double>(ws[k]);
#pragma unroll
      for (int i = 0; i < V; ++i) {
        const float e = expf(widen(packed[k].v[i]) - m[i]);
        T p;
        narrow(p, divide_probability(e, den[i], inv_den[i]));
        num[i] = fma(wk, static_cast<double>(widen(p)), num[i]);
      }
    }
#pragma unroll
    for (int i = 0; i < V; ++i) {
      narrow(res.v[i], num[i]);  // fp64 -> T directly; see the narrow() overload
    }
  } else {
    float num[V];
#pragma unroll
    for (int i = 0; i < V; ++i) {
      num[i] = 0.0f;
    }
#pragma unroll
    for (int k = 0; k < C; ++k) {
      const float wk = ws[k];
#pragma unroll
      for (int i = 0; i < V; ++i) {
        const float e = expf(widen(packed[k].v[i]) - m[i]);
        T p;
        narrow(p, divide_probability(e, den[i], inv_den[i]));
        num[i] = fmaf(wk, widen(p), num[i]);
      }
    }
#pragma unroll
    for (int i = 0; i < V; ++i) {
      narrow(res.v[i], num[i]);
    }
  }
  *reinterpret_cast<Vec<T, V>*>(
      out + static_cast<long long>(group) * a_size + col) = res;
}
// DEVICE-CODE-END

template <typename T, int C, int V>
void launch(const at::Tensor& x, const at::Tensor& w, at::Tensor& out,
            int64_t groups, int64_t a_size, int64_t block, int64_t batch_stride) {
  const int64_t n_vec = (a_size + V - 1) / V;
  const dim3 grid(static_cast<unsigned>((n_vec + block - 1) / block),
                  static_cast<unsigned>(groups));
  const dim3 threads(static_cast<unsigned>(block));
  dfl_integral_kernel<T, C, V><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const T*>(x.data_ptr()),
      reinterpret_cast<const T*>(w.data_ptr()),
      reinterpret_cast<T*>(out.data_ptr()),
      static_cast<int>(a_size),
      static_cast<long long>(batch_stride));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void dispatch_vec(const at::Tensor& x, const at::Tensor& w, at::Tensor& out,
                  int64_t groups, int64_t a_size, int64_t block,
                  int64_t batch_stride, bool wide) {
  if (wide) {
    launch<T, kChannels, 8>(x, w, out, groups, a_size, block, batch_stride);
  } else {
    launch<T, kChannels, 1>(x, w, out, groups, a_size, block, batch_stride);
  }
}

}  // namespace

// x: (b, 4*kChannels, A), row-major in its last two axes, with batch elements
// batch_stride apart.  Returns a contiguous (b, 4, A) tensor holding
// sum_k w[k] * softmax_k(x[b, j*kChannels + k, i]).
at::Tensor dfl_integral(const at::Tensor& x, const at::Tensor& w, int64_t vec,
                        int64_t block, int64_t batch_stride) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(w.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(x.device() == w.device(), "x and weight must share a device");
  TORCH_CHECK(x.dim() == 3, "x must be 3-D, got ", x.dim());
  TORCH_CHECK(w.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(x.scalar_type() == w.scalar_type(),
              "x and weight must share a dtype, got ", x.scalar_type(), " and ",
              w.scalar_type());
  TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
              "unsupported dtype ", x.scalar_type());
  TORCH_CHECK(vec == 1 || vec == 8, "vec must be 1 or 8, got ", vec);
  TORCH_CHECK(w.dim() == 1 && w.size(0) == kChannels,
              "weight must be a flat tensor of ", kChannels, " values, got shape ",
              w.sizes());
  TORCH_CHECK(x.size(1) == kGroupsPerBatch * kChannels, "x.size(1) must be ",
              kGroupsPerBatch * kChannels, ", got ", x.size(1));
  // The upper bound is 256, not the hardware's 1024: the widest instantiation needs
  // 167 registers per thread, so 512 threads would ask for more than the SM's register
  // file and fail at launch.  Nothing measured wants a larger block.
  TORCH_CHECK(block >= kChannels && block % 32 == 0 && block <= 256,
              "block must be a multiple of 32 in [", kChannels, ", 256], got ",
              block);

  const int64_t a_size = x.size(2);
  const int64_t groups = x.size(0) * kGroupsPerBatch;
  TORCH_CHECK(a_size > 0 && groups > 0, "x must be non-empty, got sizes ", x.sizes());
  TORCH_CHECK(groups <= 65535, "too many groups for the grid: ", groups);
  TORCH_CHECK(a_size <= std::numeric_limits<int>::max(), "A too large: ", a_size);

  // Row-major in the last two axes is the whole layout requirement; the batch
  // stride is free, which is what lets a channel slice of a wider tensor through.
  TORCH_CHECK(x.stride(2) == 1, "x must have unit stride along A, got ",
              x.stride(2));
  TORCH_CHECK(x.stride(1) == a_size, "x rows must be A apart, got stride ",
              x.stride(1), " for A=", a_size);
  TORCH_CHECK(batch_stride == x.stride(0), "batch_stride ", batch_stride,
              " does not match x.stride(0) ", x.stride(0));
  TORCH_CHECK(x.size(0) == 1 || batch_stride >= x.size(1) * a_size,
              "batch elements overlap: stride ", batch_stride, " < ",
              x.size(1) * a_size);

  const c10::cuda::CUDAGuard guard(x.device());
  at::Tensor out = at::empty({x.size(0), kGroupsPerBatch, a_size}, x.options());

  // The wide instantiation moves one 16-byte vector per thread per row, so A and
  // batch_stride must both be whole numbers of vectors -- that makes every row
  // start 16-byte aligned given an aligned base, and keeps the last thread of a
  // row from reaching into the next one.  Divisibility is tested in elements
  // rather than bytes because a caller-supplied stride can be large enough that
  // multiplying it by the element size would overflow int64.
  const int64_t elt = x.element_size();
  const int64_t per_vec = 16 / elt;  // == 8 for both supported dtypes
  const bool wide =
      vec == per_vec && 16 % elt == 0 &&
      a_size % per_vec == 0 && batch_stride % per_vec == 0 &&
      reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0;

  if (x.scalar_type() == at::kHalf) {
    dispatch_vec<__half>(x, w, out, groups, a_size, block, batch_stride, wide);
  } else {
    dispatch_vec<__nv_bfloat16>(x, w, out, groups, a_size, block, batch_stride,
                                wide);
  }
  return out;
}
"""


def cuda_flags(major: int, minor: int) -> list[str]:
    """Build flags for one architecture.

    Naming the ``-gencode`` here rather than through ``TORCH_CUDA_ARCH_LIST`` keeps
    the build to the device actually present -- torch skips its own arch flags once
    it sees one of ours -- without mutating the environment of whatever process
    imported this module.
    """
    return [
        "-O3",
        "-lineinfo",
        f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}",
    ]


def _build_extension():
    """Compile the fused kernel, or return ``None`` if that is not possible.

    A failure here is never fatal: the module falls back to the torch path, which
    is also the oracle the correctness tests compare against.
    """
    if not torch.cuda.is_available():
        return None
    try:
        from torch.utils.cpp_extension import load_inline

        major, minor = torch.cuda.get_device_capability()
        return load_inline(
            name=f"fk_yolov10_dfl_sm{major}{minor}",
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            extra_cuda_cflags=cuda_flags(major, minor),
            verbose=False,
        )
    except Exception:  # noqa: BLE001 - any build failure degrades to torch
        return None


_EXT = _build_extension()


class YOLODFL(nn.Module):
    """Drop-in replacement for the baseline DFL layer.

    ``conv`` is a minimal local weight holder rather than an imported ``Conv2d``,
    so ``state_dict()`` produces exactly ``conv.weight`` of shape
    ``(1, c1, 1, 1)`` -- matching the baseline key for key -- while keeping this
    file importable on its own.
    """

    class _WeightHolder(nn.Module):
        def __init__(self, c1: int):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(1, c1, 1, 1))

    def __init__(self, c1: int = 16):
        super().__init__()
        self.conv = YOLODFL._WeightHolder(c1)
        self.conv.requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1
        self._softmax = nn.Softmax(dim=1)
        # Counts fused-kernel calls.  Tests read it to tell the two paths apart;
        # nothing in the forward path depends on it.
        self.fused_calls = 0

    def _reference(self, x: torch.Tensor) -> torch.Tensor:
        """The baseline's own computation, with the contraction accumulated in fp32.

        The softmax runs in ``x``'s dtype, exactly as the baseline's does: its output
        tensor is fp16/bf16, so each of the ``c1`` probabilities is rounded to that
        dtype *before* the convolution contracts it.  That rounding is observable
        whenever the weighted sum cancels -- it collapses near-equal probabilities to
        identical values, making the cancellation exact -- so computing the softmax in
        fp32 here would silently change the operator's semantics for large opposing
        weights.  Only the contraction is widened, which is what cuBLAS does for the
        baseline's fp16 1x1 convolution anyway.

        Keeping ``F.conv2d`` rather than an explicit weighted sum means any weight
        shape the baseline accepts is accepted identically, and any shape it rejects
        is rejected identically -- including its accumulation order, which is why the
        convolution is *not* widened to fp32 here: doing so shifted results by an ulp
        against the baseline.  ``reshape`` rather than ``view`` so a strided or
        transposed input is copied instead of rejected; for the contiguous inputs the
        fast path also accepts, the two are the same view.
        """
        b, _, a = x.shape
        p = self._softmax(x.reshape(b, 4, self.c1, a).transpose(2, 1))
        w = self.conv.weight.to(device=p.device, dtype=p.dtype)
        return F.conv2d(p, w).reshape(b, 4, a).to(x.dtype)

    def _can_fuse(self, x: torch.Tensor) -> bool:
        """Whether the fused kernel's contract holds for this call.

        Full contiguity is deliberately *not* required.  The captured input is a
        64-channel slice of a 144-channel head tensor, so its batch stride exceeds
        ``64*a`` and ``is_contiguous()`` is False at ``b > 1``; only row-major-ness
        within a batch element matters to the kernel.
        """
        if _EXT is None or self.c1 != _FUSED_CHANNELS:
            return False
        # Anything that is not a plain dense tensor -- a subclass, a functorch or
        # forward-AD wrapper, a fake or meta tensor -- goes through torch, which
        # knows how to propagate whatever it carries.
        if type(x) is not torch.Tensor or not x.is_cuda or x.dim() != 3:
            return False
        if x.dtype not in (torch.float16, torch.bfloat16):
            return False
        if x.size(0) == 0 or x.size(1) != 4 * self.c1 or x.size(2) == 0:
            return False
        if x.stride(2) != 1 or x.stride(1) != x.size(2):
            return False
        if x.size(0) > 1 and x.stride(0) < x.size(1) * x.size(2):
            return False
        # blockIdx.y indexes the groups, so the grid's y extent bounds the batch.
        if x.size(0) * 4 > _MAX_GRID_Y:
            return False
        w = self.conv.weight
        # The exact baseline weight shape, not merely the right element count: a
        # differently shaped weight is one the baseline's conv2d rejects, and the
        # reference path reproduces that rejection.
        if tuple(w.shape) != (1, self.c1, 1, 1):
            return False
        if w.dtype != x.dtype or w.device != x.device or not w.is_contiguous():
            return False
        if torch.is_grad_enabled() and (x.requires_grad or w.requires_grad):
            return False
        if _has_forward_tangent(x) or _has_forward_tangent(w):
            return False
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._can_fuse(x):
            # Read the weights off the parameter on every call, so an in-place
            # update through ``weight.data[:]`` is picked up here.  ``reshape`` on
            # the contiguous (1, c1, 1, 1) parameter is a view, not a copy.
            w = self.conv.weight.detach().reshape(-1)
            out = _EXT.dfl_integral(x, w, _VEC, _BLOCK, x.stride(0))
            self.fused_calls += 1
            return out
        return self._reference(x)
