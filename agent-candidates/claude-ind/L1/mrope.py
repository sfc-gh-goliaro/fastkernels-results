"""Multi-dimensional Rotary Position Embedding (M-RoPE) for Qwen VL models.

Optimized candidate. The baseline, for every 2D (multimodal) call:

  1. casts the whole fp32 ``cos_sin_cache`` (max_pos*4 x head_dim -- 512 MB for
     the captured Qwen3-VL config) to the activation dtype,
  2. advanced-indexes it with ``positions`` to build a (3, seq, head_dim) tile,
  3. splits + ``.contiguous()`` that tile into cos/sin,
  4. runs a Triton kernel that re-reads the tile and rotates q/k.

Steps 1-3 cost far more than the rotation itself (the cast alone is ~0.75 GB of
traffic per call, independent of sequence length) and cost 5 kernel launches.

This version keeps the cache buffer (so the eager / compile fallbacks still
work) but the fast path never touches it: a single fused CUDA kernel recomputes
the angle per (token, rotary pair) as ``(float)pos * inv_freq[j]`` -- bit-wise
the same fp32 product the table was built from -- evaluates cos/sin once per
(token, j) and amortizes that over all q/k heads. One launch, and the only
memory traffic is q/k plus the 3 positions per token.

The kernel source is inlined below (_CUDA_SRC): thread mapping, head
batching, and the Cody-Waite argument reduction that keeps the trig off libm's
Payne-Hanek slow path.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import triton
import triton.language as tl

from . import rotary_emb as _rotary_emb_reg  # noqa: F401 — registers fastkernels_rope ops

# --------------------------------------------------------------------------
# Fused kernel. Kept inline (load_inline) so this module is self-contained;
# the build is cached in the torch extensions dir and happens on first use.
# --------------------------------------------------------------------------
_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <type_traits>

namespace {

// cos/sin of x = fl(pos * inv_freq[j]). sincosf() is correct here but our
// arguments reach ~2^18 rad, which lands in its Payne-Hanek slow path (~250
// cycles); with one lane in JPT diverging into it, every warp pays. Reduce mod
// 2*pi with a two-term Cody-Waite step instead -- error ~1.2e-7 rad measured
// against fp64 over the captured position range, i.e. far below a bf16 ulp --
// and finish on the MUFU units.
__device__ __forceinline__ void fast_sincos(float x, float* s, float* c) {
  const float INV_2PI = 0.15915494309189535f;
  const float C1 = 6.28318548202514648f;       // (float)(2*pi)
  const float C2 = -1.74845553146951720e-7f;   // 2*pi - C1
  const float k = rintf(x * INV_2PI);
  float r = __fmaf_rn(-k, C1, x);
  r = __fmaf_rn(-k, C2, r);
  __sincosf(r, s, c);
}

template <typename T, int N>
struct alignas(sizeof(T) * N <= 16 ? sizeof(T) * N : 16) Vec {
  T d[N];
};

template <typename T>
struct Cvt;

template <>
struct Cvt<__nv_bfloat16> {
  static constexpr bool fast_trig = true;
  static __device__ __forceinline__ float to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
  static __device__ __forceinline__ __nv_bfloat16 of_f(float x) { return __float2bfloat16_rn(x); }
  // The baseline rounds the table to the activation dtype before multiplying.
  static __device__ __forceinline__ float quant(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
  }
};

template <>
struct Cvt<__half> {
  static constexpr bool fast_trig = true;
  static __device__ __forceinline__ float to_f(__half x) { return __half2float(x); }
  static __device__ __forceinline__ __half of_f(float x) { return __float2half_rn(x); }
  static __device__ __forceinline__ float quant(float x) {
    return __half2float(__float2half_rn(x));
  }
};

template <>
struct Cvt<float> {
  static constexpr bool fast_trig = false;  // fp32 is compared at atol 1e-5
  static __device__ __forceinline__ float to_f(float x) { return x; }
  static __device__ __forceinline__ float of_f(float x) { return x; }
  static __device__ __forceinline__ float quant(float x) { return x; }
};

// Heads loaded per batch. The store of head h would otherwise order against
// the (disjoint, but opaque to the compiler) load of head h+1, costing a full
// memory round trip per head.
constexpr int kHeadBatch = 4;

// Thread mapping: one thread owns JPT consecutive rotary pairs (j0 .. j0+JPT-1,
// i.e. elements j and j + half of a head) of a single token and walks HPT heads
// (q's heads first, then k's), so its JPT cos/sin evaluations are reused across
// every head it touches. blockIdx.y splits the heads into HPT-sized groups --
// short sequences need that extra parallelism more than they need the sharing.
template <typename T, int JPT>
__global__ void mrope_kernel(
    T* __restrict__ q, T* __restrict__ k,
    const int64_t* __restrict__ pos,
    const float* __restrict__ inv_freq,
    int n_tokens, int n_qh, int nh, int half, int head_dim,
    int tpt_shift, int hpt,
    int64_t q_row, int64_t k_row,
    int64_t pos_s0, int64_t pos_s1,
    unsigned long long mh, unsigned long long mw) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int token = gid >> tpt_shift;
  if (token >= n_tokens) return;
  const int j0 = (gid - (token << tpt_shift)) * JPT;

  const int64_t* pp = pos + (int64_t)token * pos_s1;
  const int64_t p0 = pp[0];
  const int64_t p1 = pp[pos_s0];
  const int64_t p2 = pp[2 * pos_s0];

  float cs[JPT], sn[JPT];
#pragma unroll
  for (int i = 0; i < JPT; ++i) {
    const int j = j0 + i;
    const unsigned long long bit = 1ull << j;
    const int64_t p = (mh & bit) ? p1 : ((mw & bit) ? p2 : p0);
    const float ang = (float)p * inv_freq[j];
    float s, c;
    if (Cvt<T>::fast_trig) {
      fast_sincos(ang, &s, &c);
    } else {
      sincosf(ang, &s, &c);
    }
    cs[i] = Cvt<T>::quant(c);
    sn[i] = Cvt<T>::quant(s);
  }

  int h = blockIdx.y * hpt;
  int h_end = h + hpt;
  if (h_end > nh) h_end = nh;

  using V = Vec<T, JPT>;
  T* qbase = q + (int64_t)token * q_row + j0;
  T* kbase = k + (int64_t)token * k_row + j0;
  constexpr int U = kHeadBatch;
  for (; h + U <= h_end; h += U) {
    T* pa[U];
    V a[U], b[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int hh = h + u;
      pa[u] = (hh < n_qh) ? (qbase + (int64_t)hh * head_dim)
                          : (kbase + (int64_t)(hh - n_qh) * head_dim);
      a[u] = *reinterpret_cast<const V*>(pa[u]);
      b[u] = *reinterpret_cast<const V*>(pa[u] + half);
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
#pragma unroll
      for (int i = 0; i < JPT; ++i) {
        const float x1 = Cvt<T>::to_f(a[u].d[i]);
        const float x2 = Cvt<T>::to_f(b[u].d[i]);
        a[u].d[i] = Cvt<T>::of_f(x1 * cs[i] - x2 * sn[i]);
        b[u].d[i] = Cvt<T>::of_f(x2 * cs[i] + x1 * sn[i]);
      }
      *reinterpret_cast<V*>(pa[u]) = a[u];
      *reinterpret_cast<V*>(pa[u] + half) = b[u];
    }
  }
  for (; h < h_end; ++h) {
    T* pa = (h < n_qh) ? (qbase + (int64_t)h * head_dim)
                       : (kbase + (int64_t)(h - n_qh) * head_dim);
    V a = *reinterpret_cast<const V*>(pa);
    V b = *reinterpret_cast<const V*>(pa + half);
#pragma unroll
    for (int i = 0; i < JPT; ++i) {
      const float x1 = Cvt<T>::to_f(a.d[i]);
      const float x2 = Cvt<T>::to_f(b.d[i]);
      a.d[i] = Cvt<T>::of_f(x1 * cs[i] - x2 * sn[i]);
      b.d[i] = Cvt<T>::of_f(x2 * cs[i] + x1 * sn[i]);
    }
    *reinterpret_cast<V*>(pa) = a;
    *reinterpret_cast<V*>(pa + half) = b;
  }
}

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void launch(void* qp, void* kp, const int64_t* pos, const float* inv_freq,
            int n_tokens, int n_qh, int nh, int half, int head_dim,
            int64_t q_row, int64_t k_row, int64_t pos_s0, int64_t pos_s1,
            unsigned long long mh, unsigned long long mw,
            int jpt, int block, int hpt, cudaStream_t stream) {
  // Vector width must divide the rotary half and keep every access aligned.
  const size_t es = sizeof(T);
  size_t mask = (size_t)(uintptr_t)qp | (size_t)(uintptr_t)kp |
                (size_t)(q_row * es) | (size_t)(k_row * es) |
                (size_t)(head_dim * es) | (size_t)(half * es);
  while (jpt > 1 && ((half % jpt) || (mask & (jpt * es - 1)))) jpt >>= 1;

  const int tpt = half / jpt;
  int tpt_shift = 0;
  while ((1 << tpt_shift) < tpt) ++tpt_shift;
  // The caller only takes this path for a power-of-two rotary half, which
  // keeps token = gid >> tpt_shift exact.
  TORCH_CHECK((1 << tpt_shift) == tpt, "mrope_fwd: bad rotary half ", half);
  dim3 grid(ceil_div(n_tokens << tpt_shift, block), ceil_div(nh, hpt));
  T* q = reinterpret_cast<T*>(qp);
  T* k = reinterpret_cast<T*>(kp);
#define LAUNCH_JPT(J)                                                          \
  mrope_kernel<T, J><<<grid, block, 0, stream>>>(                              \
      q, k, pos, inv_freq, n_tokens, n_qh, nh, half, head_dim, tpt_shift, hpt, \
      q_row, k_row, pos_s0, pos_s1, mh, mw)
  switch (jpt) {
    case 8: LAUNCH_JPT(8); break;
    case 4: LAUNCH_JPT(4); break;
    case 2: LAUNCH_JPT(2); break;
    default: LAUNCH_JPT(1); break;
  }
#undef LAUNCH_JPT
}

}  // namespace

// In-place M-RoPE over (3, n_tokens) positions. ``query``/``key`` are
// (n_tokens, n_heads * head_dim) with a contiguous innermost dim; the rotary
// dim equals head_dim (neox style: pair j is (j, j + head_dim/2)).
void mrope_fwd(at::Tensor positions, at::Tensor query, at::Tensor key,
               at::Tensor inv_freq, int64_t head_dim, int64_t mask_h,
               int64_t mask_w, int64_t jpt, int64_t block, int64_t hpt) {
  const int n_tokens = (int)positions.size(1);
  if (n_tokens <= 0) return;
  const int half = (int)(head_dim / 2);
  const int n_qh = (int)(query.numel() / n_tokens / head_dim);
  const int n_kh = (int)(key.numel() / n_tokens / head_dim);
  const int nh = n_qh + n_kh;
  if (nh <= 0) return;

  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t* pos = positions.data_ptr<int64_t>();
  const float* ifr = inv_freq.data_ptr<float>();
  const auto mh = (unsigned long long)mask_h;
  const auto mw = (unsigned long long)mask_w;
  const int64_t q_row = query.stride(0);
  const int64_t k_row = key.stride(0);
  const int64_t ps0 = positions.stride(0), ps1 = positions.stride(1);

  switch (query.scalar_type()) {
    case at::kBFloat16:
      launch<__nv_bfloat16>(query.data_ptr(), key.data_ptr(), pos, ifr, n_tokens,
                            n_qh, nh, half, (int)head_dim, q_row, k_row, ps0,
                            ps1, mh, mw, (int)jpt, (int)block, (int)hpt, stream);
      break;
    case at::kHalf:
      launch<__half>(query.data_ptr(), key.data_ptr(), pos, ifr, n_tokens, n_qh,
                     nh, half, (int)head_dim, q_row, k_row, ps0, ps1, mh, mw,
                     (int)jpt, (int)block, (int)hpt, stream);
      break;
    case at::kFloat:
      launch<float>(query.data_ptr(), key.data_ptr(), pos, ifr, n_tokens, n_qh,
                    nh, half, (int)head_dim, q_row, k_row, ps0, ps1, mh, mw,
                    (int)jpt, (int)block, (int)hpt, stream);
      break;
    default:
      TORCH_CHECK(false, "mrope_fwd: unsupported dtype ", query.scalar_type());
  }
}
"""

_CPP_SRC = """
void mrope_fwd(at::Tensor positions, at::Tensor query, at::Tensor key,
               at::Tensor inv_freq, int64_t head_dim, int64_t mask_h,
               int64_t mask_w, int64_t jpt, int64_t block, int64_t hpt);
"""

_EXT = None
_EXT_TRIED = False


def _ext():
    """JIT-build (once, cached on disk) and return the fused extension.

    Returns None if the build is unavailable, in which case the module falls
    back to the baseline gather + Triton path.
    """
    global _EXT, _EXT_TRIED
    if _EXT_TRIED:
        return _EXT
    _EXT_TRIED = True
    old_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    try:
        from torch.utils.cpp_extension import load_inline
        major, minor = torch.cuda.get_device_capability()
        # Build for the local arch only; the ambient list spans six of them.
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        _EXT = load_inline(
            name="fk_cand_mrope_fused",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["mrope_fwd"],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
            verbose=False,
        )
    except Exception:  # pragma: no cover - fall back to the baseline path
        _EXT = None
    finally:
        if old_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old_arch
    return _EXT


def _as_i64(mask: int) -> int:
    """Reinterpret a 64-bit mask as a signed int64 (pybind takes int64_t)."""
    if mask >= (1 << 63):
        mask -= 1 << 64
    return mask


@triton.jit
def _mrope_kernel(
    q_ptr, k_ptr, cos_ptr, sin_ptr,
    num_tokens,
    n_qh: tl.constexpr, n_kh: tl.constexpr,
    hd: tl.constexpr, rd: tl.constexpr,
    pad_n_qh: tl.constexpr, pad_n_kh: tl.constexpr, pad_hd: tl.constexpr,
    mrope_section_t: tl.constexpr,
    mrope_section_h: tl.constexpr,
    mrope_section_w: tl.constexpr,
    is_interleaved: tl.constexpr,
):
    pid = tl.program_id(0)
    q_ptr = q_ptr + pid * (n_qh * hd)
    k_ptr = k_ptr + pid * (n_kh * hd)

    half_rd = rd // 2
    t_cos = cos_ptr + pid * half_rd
    h_cos = t_cos + num_tokens * half_rd
    w_cos = h_cos + num_tokens * half_rd
    t_sin = sin_ptr + pid * half_rd
    h_sin = t_sin + num_tokens * half_rd
    w_sin = h_sin + num_tokens * half_rd

    cos_offsets = tl.arange(0, pad_hd // 2)
    if is_interleaved:
        h_mask = ((cos_offsets % 3) == 1) & (cos_offsets <= 3 * mrope_section_h)
        w_mask = ((cos_offsets % 3) == 2) & (cos_offsets <= 3 * mrope_section_w)
        t_mask = ~(h_mask | w_mask)
    else:
        t_end = mrope_section_t
        h_end = t_end + mrope_section_h
        t_mask = cos_offsets < mrope_section_t
        h_mask = (t_end <= cos_offsets) & (cos_offsets < h_end)
        w_mask = (h_end <= cos_offsets) & (cos_offsets < half_rd)

    t_cos_row = tl.load(t_cos + cos_offsets, mask=t_mask, other=0)
    h_cos_row = tl.load(h_cos + cos_offsets, mask=h_mask, other=0)
    w_cos_row = tl.load(w_cos + cos_offsets, mask=w_mask, other=0)
    t_sin_row = tl.load(t_sin + cos_offsets, mask=t_mask, other=0)
    h_sin_row = tl.load(h_sin + cos_offsets, mask=h_mask, other=0)
    w_sin_row = tl.load(w_sin + cos_offsets, mask=w_mask, other=0)

    cos_row = t_cos_row + h_cos_row + w_cos_row
    sin_row = t_sin_row + h_sin_row + w_sin_row

    first_half_q_offsets = (
        tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    )
    first_half_k_offsets = (
        tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    )
    first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (
        tl.arange(0, pad_hd // 2)[None, :] < rd // 2
    )
    first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (
        tl.arange(0, pad_hd // 2)[None, :] < rd // 2
    )

    q_tile_1 = tl.load(q_ptr + first_half_q_offsets, mask=first_q_mask, other=0).to(sin_row.dtype)
    k_tile_1 = tl.load(k_ptr + first_half_k_offsets, mask=first_k_mask, other=0).to(sin_row.dtype)

    second_half_q_offsets = first_half_q_offsets + (rd // 2)
    second_half_k_offsets = first_half_k_offsets + (rd // 2)

    q_tile_2 = tl.load(q_ptr + second_half_q_offsets, mask=first_q_mask, other=0).to(sin_row.dtype)
    k_tile_2 = tl.load(k_ptr + second_half_k_offsets, mask=first_k_mask, other=0).to(sin_row.dtype)

    new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
    tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
    new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
    tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=first_q_mask)

    new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
    tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
    new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
    tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=first_k_mask)


class MRotaryEmbedding(nn.Module):
    """M-RoPE for Qwen2-VL / Qwen3-VL.

    positions can be either:
      - 1D (seq_len,) for text-only (all 3 dims identical -> standard RoPE)
      - 2D (3, seq_len) for multimodal (T/H/W positions differ)

    mrope_section: list of 3 ints [t, h, w] summing to rotary_dim // 2
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        mrope_section: list[int],
        mrope_interleaved: bool = False,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = head_dim
        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved
        assert sum(mrope_section) == head_dim // 2

        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        t = torch.arange(max_position_embeddings * 4, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).float()
        self.register_buffer("cos_sin_cache", cache, persistent=False)
        # Fast path state: the same inv_freq the table was built from, plus the
        # per-rotary-pair T/H/W selection encoded as two 64-bit masks.
        self.register_buffer("_inv_freq_f32", inv_freq.clone(), persistent=False)
        self._mask_h, self._mask_w = self._section_masks()
        self._fast_fwd = None        # ext.mrope_fwd once compiled
        self._fast_inv_freq = None   # inv_freq on the activation device
        self._fast_failed = False

    def _section_masks(self) -> tuple[int, int]:
        """Bit j of (mask_h, mask_w) selects positions[1] / positions[2] for
        rotary pair j; a clear bit in both means positions[0]. Mirrors the
        masks in ``_mrope_kernel`` exactly."""
        s = self.mrope_section
        half = self.head_dim // 2
        mh = mw = 0
        for j in range(half):
            if self.mrope_interleaved:
                if j % 3 == 1 and j <= 3 * s[1]:
                    mh |= 1 << j
                elif j % 3 == 2 and j <= 3 * s[2]:
                    mw |= 1 << j
            else:
                if s[0] <= j < s[0] + s[1]:
                    mh |= 1 << j
                elif s[0] + s[1] <= j:
                    mw |= 1 << j
        return _as_i64(mh), _as_i64(mw)

    # ---- fast path ------------------------------------------------------
    def _plan(self, n_tokens: int) -> tuple[int, int, int]:
        """(rotary pairs per thread, block size, heads per thread).

        Tuned on B200: large sequences want every head on one thread (each
        sincos amortized over all 17 of them), short ones want the heads spread
        over more blocks instead, since at ~1 K tokens a single head-group
        leaves most SMs idle and the redundant sincos is nearly free."""
        if n_tokens >= 2048:
            return 4, 64, 9
        if n_tokens >= 512:
            return 4, 128, 3
        return 4, 128, 1

    def _fused(self, positions, query, key) -> bool:
        """Run the fused kernel; False if this call is outside what it covers."""
        fwd = self._fast_fwd
        if fwd is None:
            fwd = self._fast_setup()
            if fwd is None:
                return False
        hd = self.head_dim
        if positions.dtype != torch.int64 or positions.shape[0] != 3:
            return False
        if query.dim() > 3 or key.dim() > 3:
            return False
        if query.stride(-1) != 1 or key.stride(-1) != 1:
            return False
        if query.dim() == 3 and query.stride(1) != hd:
            return False
        if key.dim() == 3 and key.stride(1) != hd:
            return False
        n = positions.shape[1]
        if n and ((query.numel() // n) % hd or (key.numel() // n) % hd):
            return False
        ifr = self._fast_inv_freq
        if ifr is None or ifr.device != query.device:
            ifr = self._inv_freq_f32.to(device=query.device, dtype=torch.float32)
            self._fast_inv_freq = ifr
        fwd(positions, query, key, ifr, hd, self._mask_h, self._mask_w,
            *self._plan(n))
        return True

    def _fast_setup(self):
        """JIT-compile the kernel on first use; None if unavailable."""
        if self._fast_failed:
            return None
        half = self.head_dim // 2
        ext = _ext()
        if (ext is None or self.rotary_dim != self.head_dim
                or self.head_dim % 8 or (half & (half - 1))):
            self._fast_failed = True
            return None
        self._fast_fwd = ext.mrope_fwd
        return self._fast_fwd

    def _apply_sgl_rope(self, positions_1d, query, key):
        """Apply standard RoPE for 1D positions (decode or text-only)."""
        cache = self.cos_sin_cache
        if cache.dtype != query.dtype:
            cache = cache.to(query.dtype)
        if torch.compiler.is_compiling():
            from .rotary_emb import RotaryEmbedding
            return RotaryEmbedding.forward_native(
                positions_1d,
                query.view(query.shape[0], -1),
                key.view(key.shape[0], -1),
                self.head_dim, cache,
            )
        torch.ops.fastkernels_rope.rotary_embedding(
            positions_1d,
            query.view(query.shape[0], -1),
            key.view(key.shape[0], -1),
            self.head_dim,
            cache,
            True,
        )
        return query, key

    def forward_native_2d(self, positions, query, key):
        """Pure PyTorch MRoPE for (3, seq_len) positions -- Inductor-friendly."""
        cache = self.cos_sin_cache
        if cache.dtype != query.dtype:
            cache = cache.to(query.dtype)

        num_tokens = query.shape[0]
        cos_sin = cache[positions]          # (3, seq_len, head_dim)
        cos, sin = cos_sin.chunk(2, dim=-1) # each (3, seq_len, head_dim/2)

        if self.mrope_interleaved:
            cos = self._apply_interleaved(cos)
            sin = self._apply_interleaved(sin)
        else:
            cos = torch.cat(
                [m[i] for i, m in enumerate(cos.split(self.mrope_section, dim=-1))],
                dim=-1,
            )
            sin = torch.cat(
                [m[i] for i, m in enumerate(sin.split(self.mrope_section, dim=-1))],
                dim=-1,
            )
        # cos, sin: (seq_len, head_dim/2)

        hd = self.head_dim
        half = hd // 2
        q_shape = query.shape
        k_shape = key.shape
        q = query.view(num_tokens, -1, hd)
        k = key.view(num_tokens, -1, hd)

        cos = cos.unsqueeze(1)  # (seq_len, 1, head_dim/2)
        sin = sin.unsqueeze(1)

        q1 = q[..., :half]
        q2 = q[..., half:]
        k1 = k[..., :half]
        k2 = k[..., half:]

        new_q = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        new_k = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)

        return new_q.view(q_shape), new_k.view(k_shape)

    def forward(self, positions, query, key):
        """Apply M-RoPE in-place.

        Args:
            positions: (seq_len,) or (3, seq_len) int64 tensor
            query: (seq_len, num_heads, head_dim)
            key: (seq_len, num_kv_heads, head_dim)
        """
        if positions.ndim == 1:
            return self._apply_sgl_rope(positions, query, key)

        if torch.compiler.is_compiling():
            return self.forward_native_2d(positions, query, key)

        if self._fused(positions, query, key):
            return query, key

        # ---- fallback: the baseline gather + Triton path ------------------
        cache = self.cos_sin_cache
        if cache.dtype != query.dtype:
            cache = cache.to(query.dtype)

        num_tokens = positions.shape[-1]
        cos_sin = cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)

        cos_3d = cos.contiguous()
        sin_3d = sin.contiguous()

        hd = self.head_dim
        q_was_2d = query.ndim == 2
        if q_was_2d:
            n_qh = query.shape[1] // hd
            n_kh = key.shape[1] // hd
        else:
            n_qh = query.shape[1]
            n_kh = key.shape[1]

        q_flat = query.reshape(num_tokens, -1).contiguous()
        k_flat = key.reshape(num_tokens, -1).contiguous()
        pad_hd = triton.next_power_of_2(hd)
        pad_n_qh = triton.next_power_of_2(n_qh)
        pad_n_kh = triton.next_power_of_2(n_kh)

        _mrope_kernel[(num_tokens,)](
            q_flat, k_flat, cos_3d, sin_3d,
            num_tokens, n_qh, n_kh, hd, hd,
            pad_n_qh, pad_n_kh, pad_hd,
            self.mrope_section[0], self.mrope_section[1], self.mrope_section[2],
            self.mrope_interleaved,
        )

        return q_flat.view_as(query), k_flat.view_as(key)

    def _apply_interleaved(self, x):
        """Reorganize from [TTT...HHH...WWW] to interleaved [THWTHW...]."""
        s = self.mrope_section
        result = x[0].clone()
        result[..., 1:s[1] * 3:3] = x[1, ..., 1:s[1] * 3:3]
        result[..., 2:s[2] * 3:3] = x[2, ..., 2:s[2] * 3:3]
        return result
