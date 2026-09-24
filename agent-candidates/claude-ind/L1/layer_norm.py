"""LayerNorm backed by a single fused CUDA kernel.

Semantics match ``baseline.py`` exactly -- fp32 reduction, optional
scale/offset, output in the input dtype -- but the work takes one launch instead
of three.  The baseline's ``promote_fp32`` path materializes ``x.float()``, runs
``F.layer_norm`` on it and casts back; even with ``promote_fp32=False`` torch's
own kernel reads ``x`` twice (Welford pass, then normalize).  Here a block owns
a row and keeps it in registers across the reduction, so each element is read
once and written once, in one launch.

Two properties of this operator's captured shapes drove the design:

* Three of the benchmarked shapes are tiny ([256], 256x128, 16x384).  At that
  size an empty kernel launch and a full LayerNorm launch measure the same, so
  only the *number* of launches matters -- hence one fused kernel, and a forward
  that does no allocation or dtype juggling in Python.
* The two large shapes (16170x4608, 5940x4608) are bound by how many loads the
  kernel keeps in flight, and that is capped by registers rather than by DRAM
  (ncu: 17% DRAM throughput, occupancy limited to 4 blocks/SM by registers).  So
  the staged row is held in the input dtype rather than fp32 -- half the
  registers per element -- and the next row's loads are issued before the
  current row's reduction, so nothing waits on the barrier.

The CUDA source is inlined below so this file is the whole deliverable; it is
JIT-compiled once per machine and cached by ``torch.utils.cpp_extension``.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

_CUDA_SRC = r"""// Fused LayerNorm forward: one launch, one pass over x, fp32 accumulation.
//
// A block owns one row at a time and keeps it in registers between the
// reduction and the write-back, so every element is read once and written once.
// The baseline needs three launches on the promote_fp32 path (x.float(),
// F.layer_norm, cast back), and even in bf16 torch's own kernel reads x twice
// (Welford pass, then normalize).
//
// Two details that the captured shapes (16170x4608 and 5940x4608 bf16) made
// worth the trouble, both about keeping loads in flight:
//
//   * The staged row is held in the *input* dtype.  Staging it as fp32 doubles
//     the registers per element, and registers -- not DRAM -- are what cap the
//     number of resident blocks here, so the fp32 stage measured ~20% slower
//     despite doing fewer conversions.
//   * The next row's loads are issued before the current row's reduction, so
//     the block's memory pipeline is not idle across the barrier.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <type_traits>

namespace {

__device__ __forceinline__ float cvt_f(float v) { return v; }
__device__ __forceinline__ float cvt_f(__half v) { return __half2float(v); }
__device__ __forceinline__ float cvt_f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ void cvt_t(float &d, float s) { d = s; }
__device__ __forceinline__ void cvt_t(__half &d, float s) { d = __float2half_rn(s); }
__device__ __forceinline__ void cvt_t(__nv_bfloat16 &d, float s) { d = __float2bfloat16_rn(s); }

// The widest load/store a thread can issue.
template <typename T>
struct alignas(16) Vec {
  static constexpr int E = 16 / sizeof(T);
  T d[E];
};

__device__ __forceinline__ void warp_red2(float &a, float &b) {
#pragma unroll
  for (int off = 16; off; off >>= 1) {
    a += __shfl_xor_sync(0xffffffffu, a, off);
    b += __shfl_xor_sync(0xffffffffu, b, off);
  }
}

// Sum (a, b) across the block and leave the total in every thread.  One
// barrier: the partials are summed redundantly by all threads rather than by
// one warp followed by a broadcast, which needs a second barrier and measured
// slower.  Consecutive rows are handed different `sm` halves, so no
// anti-dependency barrier is needed either.
__device__ __forceinline__ void block_red2(float &a, float &b, float *sm) {
  const int nw = blockDim.x >> 5;
  warp_red2(a, b);
  if (nw == 1) return;
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  if (lane == 0) {
    sm[wid] = a;
    sm[32 + wid] = b;
  }
  __syncthreads();
  float sa = 0.f, sb = 0.f;
  for (int i = 0; i < nw; ++i) {
    sa += sm[i];
    sb += sm[32 + i];
  }
  a = sa;
  b = sb;
}

// Largest block a given VPT may be launched with: VPT * threads has to cover
// nvec <= 1024, rounded up to a warp.
constexpr int lb_for(int vpt) {
  return vpt == 1 ? 1024 : (vpt == 2 ? 512 : (vpt == 3 ? 352 : (vpt == 4 ? 256
       : (vpt == 6 ? 192 : 128))));
}

template <typename T, int VPT>
__global__ __launch_bounds__(lb_for(VPT)) void ln_row(
    const T *__restrict__ X, T *__restrict__ Y, const T *__restrict__ W,
    const T *__restrict__ B, int n, int nvec, float eps, long rows) {
  constexpr int E = Vec<T>::E;
  using V = Vec<T>;
  __shared__ float sm[128];
  const int tid = threadIdx.x, nt = blockDim.x;
  const float inv_n = 1.0f / (float)n;
  const V *WV = reinterpret_cast<const V *>(W);
  const V *BV = reinterpret_cast<const V *>(B);

  int idx[VPT];
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    const int j = tid + i * nt;
    idx[i] = j < nvec ? j : -1;
  }

  const long step = (long)gridDim.x;
  long row = (long)blockIdx.x;
  V cur[VPT], nxt[VPT];
  if (row < rows) {
    const V *xv = reinterpret_cast<const V *>(X + row * (long)n);
#pragma unroll
    for (int i = 0; i < VPT; ++i)
      if (idx[i] >= 0) cur[i] = xv[idx[i]];
  }
  int parity = 0;
  for (; row < rows; row += step) {
    const long nrow = row + step;
    if (nrow < rows) {  // issue the next row before stalling on this one
      const V *xn = reinterpret_cast<const V *>(X + nrow * (long)n);
#pragma unroll
      for (int i = 0; i < VPT; ++i)
        if (idx[i] >= 0) nxt[i] = xn[idx[i]];
    }
    float s = 0.f, q = 0.f;
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      if (idx[i] >= 0) {
#pragma unroll
        for (int k = 0; k < E; ++k) {
          const float f = cvt_f(cur[i].d[k]);
          s += f;
          q += f * f;
        }
      }
    }
    block_red2(s, q, sm + (parity ? 64 : 0));
    parity ^= 1;
    const float mean = s * inv_n;
    const float rstd = rsqrtf(fmaxf(q * inv_n - mean * mean, 0.f) + eps);
    V *yv = reinterpret_cast<V *>(Y + row * (long)n);
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      const int j = idx[i];
      if (j >= 0) {
        V wv, bv, o;
        if (W) wv = WV[j];
        if (B) bv = BV[j];
#pragma unroll
        for (int k = 0; k < E; ++k) {
          float f = (cvt_f(cur[i].d[k]) - mean) * rstd;
          if (W) f *= cvt_f(wv.d[k]);
          if (B) f += cvt_f(bv.d[k]);
          cvt_t(o.d[k], f);
        }
        yv[j] = o;
      }
    }
#pragma unroll
    for (int i = 0; i < VPT; ++i) cur[i] = nxt[i];
  }
}

// Gather E consecutive affine terms into fp32 registers.  WT may be wider than
// T (an fp32 gamma against a bf16 activation), in which case this issues
// several 16B loads instead of one.
template <typename WT, int E>
__device__ __forceinline__ void load_w(const WT *p, int j, float *out) {
  constexpr int PER = 16 / sizeof(WT);
  const WT *q = p + (long)j * E;
#pragma unroll
  for (int c = 0; c < E; c += PER) {
    Vec<WT> v = *reinterpret_cast<const Vec<WT> *>(q + c);
#pragma unroll
    for (int k = 0; k < PER; ++k) out[c + k] = cvt_f(v.d[k]);
  }
}

// Rows too long to stage in registers, or an affine dtype wider than x: two
// vectorized passes, the second one reading x back out of L2.
template <typename T, typename WT>
__global__ __launch_bounds__(256) void ln_vec2(
    const T *__restrict__ X, T *__restrict__ Y, const WT *__restrict__ W,
    const WT *__restrict__ B, int n, int nvec, float eps, long rows) {
  constexpr int E = Vec<T>::E;
  __shared__ float sm[128];
  const int tid = threadIdx.x, nt = blockDim.x;
  const float inv_n = 1.0f / (float)n;
  int parity = 0;
  for (long row = (long)blockIdx.x; row < rows; row += (long)gridDim.x) {
    const Vec<T> *xv = reinterpret_cast<const Vec<T> *>(X + row * (long)n);
    float s = 0.f, q = 0.f;
    for (int j = tid; j < nvec; j += nt) {
      Vec<T> v = xv[j];
#pragma unroll
      for (int k = 0; k < E; ++k) {
        const float f = cvt_f(v.d[k]);
        s += f;
        q += f * f;
      }
    }
    block_red2(s, q, sm + (parity ? 64 : 0));
    parity ^= 1;
    const float mean = s * inv_n;
    const float rstd = rsqrtf(fmaxf(q * inv_n - mean * mean, 0.f) + eps);
    Vec<T> *yv = reinterpret_cast<Vec<T> *>(Y + row * (long)n);
    for (int j = tid; j < nvec; j += nt) {
      Vec<T> v = xv[j], o;
      float wv[E], bv[E];
      if (W) load_w<WT, E>(W, j, wv);
      if (B) load_w<WT, E>(B, j, bv);
#pragma unroll
      for (int k = 0; k < E; ++k) {
        float f = (cvt_f(v.d[k]) - mean) * rstd;
        if (W) f *= wv[k];
        if (B) f += bv[k];
        cvt_t(o.d[k], f);
      }
      yv[j] = o;
    }
  }
}

// Last resort: any n, any alignment.  Two scalar passes.
template <typename T, typename WT>
__global__ __launch_bounds__(256) void ln_gen(
    const T *__restrict__ X, T *__restrict__ Y, const WT *__restrict__ W,
    const WT *__restrict__ B, int n, float eps, long rows) {
  __shared__ float sm[128];
  const int tid = threadIdx.x, nt = blockDim.x;
  const float inv_n = 1.0f / (float)n;
  int parity = 0;
  for (long row = (long)blockIdx.x; row < rows; row += (long)gridDim.x) {
    const T *xr = X + row * (long)n;
    T *yr = Y + row * (long)n;
    float s = 0.f, q = 0.f;
    for (int i = tid; i < n; i += nt) {
      const float f = cvt_f(xr[i]);
      s += f;
      q += f * f;
    }
    block_red2(s, q, sm + (parity ? 64 : 0));
    parity ^= 1;
    const float mean = s * inv_n;
    const float rstd = rsqrtf(fmaxf(q * inv_n - mean * mean, 0.f) + eps);
    for (int i = tid; i < n; i += nt) {
      float f = (cvt_f(xr[i]) - mean) * rstd;
      if (W) f *= cvt_f(W[i]);
      if (B) f += cvt_f(B[i]);
      cvt_t(yr[i], f);
    }
  }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------
// Block width to aim for.  192 threads (6 warps) won every sweep on the large
// shapes: enough warps to cover the reduction, few enough that several blocks
// stay resident per SM.
constexpr int kTargetThreads = 192;
// Resident-block budget for the persistent grid; past ~4 blocks/SM this made no
// measurable difference, and a block that owns several rows amortizes its setup.
constexpr int kBlocksPerSM = 8;

inline int round32(int v) { return ((v + 31) / 32) * 32; }

int sm_count() {
  static int n = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  return n;
}

template <typename T, typename WT>
void launch(const at::Tensor &x, at::Tensor &out, const WT *w, const WT *b, int n,
            long rows, float eps, bool vec_ok) {
  const T *xp = reinterpret_cast<const T *>(x.const_data_ptr());
  T *yp = reinterpret_cast<T *>(out.data_ptr());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int nvec = n / Vec<T>::E;
  const long gcap = (long)kBlocksPerSM * sm_count();
  const unsigned grid = (unsigned)(rows < gcap ? rows : gcap);

  if constexpr (std::is_same<WT, T>::value) {
    if (vec_ok && nvec <= 1024) {
      static const int kVpt[] = {1, 2, 3, 4, 6, 8};
      int vi = 0;
      while (vi < 5 &&
             round32((nvec + kVpt[vi] - 1) / kVpt[vi]) > kTargetThreads) ++vi;
      const int vpt = kVpt[vi];
      int threads = round32((nvec + vpt - 1) / vpt);
      if (threads < 32) threads = 32;
#define LAUNCH(V) \
  ln_row<T, V><<<grid, threads, 0, stream>>>(xp, yp, w, b, n, nvec, eps, rows)
      switch (vpt) {
        case 1: LAUNCH(1); break;
        case 2: LAUNCH(2); break;
        case 3: LAUNCH(3); break;
        case 4: LAUNCH(4); break;
        case 6: LAUNCH(6); break;
        default: LAUNCH(8); break;
      }
#undef LAUNCH
      return;
    }
  }
  if (vec_ok) {
    ln_vec2<T, WT><<<grid, 256, 0, stream>>>(xp, yp, w, b, n, nvec, eps, rows);
    return;
  }
  int threads = round32(n < 256 ? n : 256);
  if (threads < 32) threads = 32;
  if (threads > 256) threads = 256;
  ln_gen<T, WT><<<grid, threads, 0, stream>>>(xp, yp, w, b, n, eps, rows);
}

inline bool aligned16(const void *p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }

const at::Tensor *opt_tensor(const c10::optional<at::Tensor> &o) {
  if (o.has_value() && o->defined()) return &o.value();
  return nullptr;
}

}  // namespace

at::Tensor fk_layer_norm(const at::Tensor &x_in, const c10::optional<at::Tensor> &w_opt,
                         const c10::optional<at::Tensor> &b_opt, double eps) {
  TORCH_CHECK(x_in.is_cuda(), "layer_norm: x must be a CUDA tensor");
  TORCH_CHECK(x_in.dim() >= 1, "layer_norm: x must have >= 1 dim");
  const auto dt = x_in.scalar_type();
  TORCH_CHECK(dt == at::kBFloat16 || dt == at::kHalf || dt == at::kFloat,
              "layer_norm: unsupported dtype");

  at::Tensor x = x_in.is_contiguous() ? x_in : x_in.contiguous();
  const int n = (int)x.size(-1);
  at::Tensor out = at::empty(x_in.sizes(), x_in.options());
  if (n == 0 || x.numel() == 0) return out;
  const long rows = (long)(x.numel() / n);

  const at::Tensor *w = opt_tensor(w_opt);
  const at::Tensor *b = opt_tensor(b_opt);
  at::Tensor wc, bc;
  if (w) {
    TORCH_CHECK(w->is_cuda() && w->numel() == n, "layer_norm: bad weight");
    wc = w->is_contiguous() ? *w : w->contiguous();
  }
  if (b) {
    TORCH_CHECK(b->is_cuda() && b->numel() == n, "layer_norm: bad bias");
    bc = b->is_contiguous() ? *b : b->contiguous();
  }
  at::ScalarType wdt = w ? wc.scalar_type() : (b ? bc.scalar_type() : dt);
  if (w && b) TORCH_CHECK(wc.scalar_type() == bc.scalar_type(),
                          "layer_norm: weight/bias dtype mismatch");
  TORCH_CHECK(wdt == dt || wdt == at::kFloat, "layer_norm: unsupported affine dtype");

  const c10::cuda::CUDAGuard guard(x.device());
  const void *wp = w ? wc.const_data_ptr() : nullptr;
  const void *bp = b ? bc.const_data_ptr() : nullptr;
  const int E = dt == at::kFloat ? 4 : 8;
  const bool vec_ok = (n % E == 0) && aligned16(x.const_data_ptr()) &&
                      aligned16(out.data_ptr()) && (!wp || aligned16(wp)) &&
                      (!bp || aligned16(bp));

#define DISPATCH_W(T)                                                                  \
  if (wdt == at::kFloat)                                                               \
    launch<T, float>(x, out, (const float *)wp, (const float *)bp, n, rows, (float)eps, \
                     vec_ok);                                                          \
  else                                                                                 \
    launch<T, T>(x, out, (const T *)wp, (const T *)bp, n, rows, (float)eps, vec_ok);

  if (dt == at::kBFloat16) {
    DISPATCH_W(__nv_bfloat16)
  } else if (dt == at::kHalf) {
    DISPATCH_W(__half)
  } else {
    launch<float, float>(x, out, (const float *)wp, (const float *)bp, n, rows,
                         (float)eps, vec_ok);
  }
#undef DISPATCH_W
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("layer_norm", &fk_layer_norm, "fused layer norm", py::arg("x"), py::arg("weight"),
        py::arg("bias"), py::arg("eps"));
}
"""

_EXT = None   # kept only to own the compiled module
_LN = None    # the kernel entry point, or None if it could not be built
_LOADED = False


def _pin_arch() -> None:
    """Build for the local arch only.

    The ambient ``TORCH_CUDA_ARCH_LIST`` in this environment lists six
    architectures, which turns a ~1 min build into a ~6 min one; the fastkernels
    CUDA loader (``infra/cuda_ext.py``) overrides it the same way, and honours
    the same ``FASTKERNELS_CUDA_ARCH_LIST`` escape hatch.
    """
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
    """JIT-compile the fused kernel; leave ``_LN`` as None if that is not possible."""
    global _EXT, _LN, _LOADED
    _LOADED = True
    try:
        from torch.utils.cpp_extension import load_inline

        _pin_arch()
        _EXT = load_inline(
            name="fk_l1_layer_norm_fused",
            cpp_sources="",
            cuda_sources=_CUDA_SRC,
            extra_cuda_cflags=[
                "-O3",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            ],
            verbose=False,
        )
        _LN = _EXT.layer_norm
    except Exception:  # noqa: BLE001 - no nvcc / no GPU: run the torch path
        _EXT = None
        _LN = None


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        create_scale: bool = True,
        create_offset: bool = True,
        promote_fp32: bool = True,
    ):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.promote_fp32 = promote_fp32

        if elementwise_affine and create_scale:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter("weight", None)

        if elementwise_affine and create_offset:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("bias", None)

        if not _LOADED:
            _load()

        # fp32 views of weight/bias for the torch fallback (see _promoted).
        self._cast_done = False
        self._src_w: torch.Tensor | None = None
        self._src_b: torch.Tensor | None = None
        self._w32: torch.Tensor | None = None
        self._b32: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The kernel reduces in fp32 whatever the input dtype, which is what
        # both baseline paths do -- promote_fp32 explicitly, and torch's own
        # layer_norm via its fp32 accumulator -- so one path serves both.  eps
        # is applied to the fp32 variance exactly as F.layer_norm does.
        # torch.is_grad_enabled(): the kernel returns a leaf tensor, so a
        # training-mode caller has to go down the autograd-capable torch path.
        # The bench times under no_grad, and this costs ~100 ns against a ~5 us
        # launch.
        if _LN is not None and x.is_cuda and not torch.is_grad_enabled():
            d = x.dtype
            if d is torch.bfloat16 or d is torch.float16 or d is torch.float32:
                w, b = self.weight, self.bias
                if (x.shape[-1:] == self.normalized_shape
                        and (w is None or w.dtype is d)
                        and (b is None or b.dtype is d)):
                    return _LN(x, w, b, self.eps)
        if not self.promote_fp32:
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps,
            )
        return self._promoted(x)

    def _promoted(self, x: torch.Tensor) -> torch.Tensor:
        """Baseline fp32 path, kept for inputs the kernel does not claim."""
        orig_dtype = x.dtype
        if (not self._cast_done
                or self._src_w is not self.weight
                or self._src_b is not self.bias):
            w, b = self.weight, self.bias
            self._src_w, self._src_b = w, b
            self._w32 = (w.float()
                         if w is not None and w.dtype != torch.float32 else w)
            self._b32 = (b.float()
                         if b is not None and b.dtype != torch.float32 else b)
            self._cast_done = True
        return F.layer_norm(
            x.float(), self.normalized_shape, self._w32, self._b32, self.eps,
        ).to(orig_dtype)
