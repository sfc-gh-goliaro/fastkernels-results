"""Bilinear interpolation of learned 2D position embeddings (Qwen3-VL).

Owns a learned embedding weight of (num_grid_per_side^2, hidden_size).
forward() interpolates these onto arbitrary (h, w) grids using bilinear
weights, then reshuffles by spatial_merge_size for the vision encoder.

The reference implementation walks ``grid_thw_list`` in Python and, per image,
rebuilds the bilinear stencil with ~20 tiny elementwise kernels (two
``linspace``s, floors/clamps, three ``meshgrid``s, four weight terms, two
``stack``s), gathers ``4 * h * w`` embedding rows, multiplies, reduces, then
materializes two more copies (the ``permute`` reshuffle and the ``expand(t)``
frame repeat) before a final ``cat``. For a batch of eight 22x40 grids that is
~200 launches moving ~150 MB to produce 32 MB: ~1.7 ms, about 100x off the
bandwidth floor.

Two properties of the operator drive this rewrite:

1. The *entire* stencil -- which four embedding rows an output row mixes, and
   with what four weights -- depends only on ``(num_grid_per_side,
   spatial_merge_size, h, w)``, never on the weight values. So it is built once
   per distinct grid shape and memoized as a descriptor table: four source row
   ids plus the four coefficients, the latter pre-rounded to the output dtype
   exactly as the baseline's ``weights.to(dtype)`` does. A repeated shape then
   costs one dict hit.
2. Two grid entries with the same ``(h, w)`` -- every frame of a video, every
   image of one resolution -- therefore have *byte-identical* output blocks. So
   the descriptor is indexed by *distinct* row, carrying the list of output rows
   it lands on. That is the baseline's own ``expand(t)`` trick extended across
   the batch: for the captured 8x[2, 22, 40] video batch it collapses 14080
   gathers into 880, and what is left is bandwidth on the store side.

What runs on the GPU is a single kernel with no index math at all: a warp reads
one descriptor, streams the four source rows through 8-byte vector loads,
accumulates in fp32, rounds once, and stores the result to each of its
destinations. Because a destination is just a row offset, the ``permute``
reshuffle, the frame repeat and the ``cat`` are all absorbed into the store
address -- one kernel, one pass, no intermediates. Rows are split into column
chunks when there are too few of them to fill the device.

Anything the kernel cannot serve (CPU/non-contiguous weight, a dtype with no
path, a grid not divisible by ``spatial_merge_size``) falls back to the
reference loop in :meth:`_reference`.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from ..L1.embedding import Embedding

_CUDA_SRC = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>

// ---------------------------------------------------------------------------
// Per-dtype traits. The vector is always 8 bytes (one 64-bit load per thread,
// 256 B per warp instruction -- measurably better here than 16 B, which halves
// the loop trip count and with it the memory-pipeline overlap).
//
// bf16 -> fp32 is a pure 16-bit shift (no conversion unit), and fp32 -> bf16
// packs two lanes per ``cvt.rn.bf16x2.f32``, so the inner loop stays fma-bound.
// ---------------------------------------------------------------------------
struct TrBF16 {
  using scalar = __nv_bfloat16;
  static constexpr int NE = 4;                     // scalars per 8B vector
  __device__ static float toF(scalar s) { return __bfloat162float(s); }
  __device__ static scalar fromF(float f) { return __float2bfloat16(f); }
  __device__ static void unpack(const uint2& v, float* f) {
    f[0] = __int_as_float(v.x << 16);
    f[1] = __int_as_float(v.x & 0xffff0000u);
    f[2] = __int_as_float(v.y << 16);
    f[3] = __int_as_float(v.y & 0xffff0000u);
  }
  __device__ static uint2 pack(const float* f) {
    __nv_bfloat162 a = __floats2bfloat162_rn(f[0], f[1]);
    __nv_bfloat162 b = __floats2bfloat162_rn(f[2], f[3]);
    return make_uint2(*reinterpret_cast<const unsigned*>(&a),
                      *reinterpret_cast<const unsigned*>(&b));
  }
};

struct TrFP16 {
  using scalar = __half;
  static constexpr int NE = 4;
  __device__ static float toF(scalar s) { return __half2float(s); }
  __device__ static scalar fromF(float f) { return __float2half(f); }
  __device__ static void unpack(const uint2& v, float* f) {
    const __half2* h = reinterpret_cast<const __half2*>(&v);
    float2 a = __half22float2(h[0]), b = __half22float2(h[1]);
    f[0] = a.x; f[1] = a.y; f[2] = b.x; f[3] = b.y;
  }
  __device__ static uint2 pack(const float* f) {
    __half2 a = __floats2half2_rn(f[0], f[1]), b = __floats2half2_rn(f[2], f[3]);
    return make_uint2(*reinterpret_cast<const unsigned*>(&a),
                      *reinterpret_cast<const unsigned*>(&b));
  }
};

struct TrFP32 {
  using scalar = float;
  static constexpr int NE = 2;
  __device__ static float toF(scalar s) { return s; }
  __device__ static scalar fromF(float f) { return f; }
  __device__ static void unpack(const uint2& v, float* f) {
    f[0] = __int_as_float(v.x); f[1] = __int_as_float(v.y);
  }
  __device__ static uint2 pack(const float* f) {
    return make_uint2(__float_as_int(f[0]), __float_as_int(f[1]));
  }
};

// ---------------------------------------------------------------------------
// One warp owns (one distinct output row) x (one chunk of its D elements).
//
// ``idx``/``cf``/``meta`` are indexed by *distinct* row: two grid entries with
// the same (h, w) -- every frame of a video, every image of one resolution --
// produce byte-identical output blocks, so such a row is mixed once and then
// stored to each of its ``meta.y`` destinations. That is the baseline's own
// ``expand(t)`` trick extended across the batch: it turns a 4-row gather per
// output row into one gather plus N cheap stores.
//
// ``MULTI`` is false whenever every row has a single destination, which lets the
// store address stay loop-invariant (the common all-distinct-shapes batch).
// There is no index math in either path -- the descriptor is built once per
// shape on the host -- so the kernel stays short and cheap to start cold.
// ---------------------------------------------------------------------------
#define BX 32      // lanes per row
#define RPB 4      // rows per block

template <typename Tr, bool MULTI>
__global__ __launch_bounds__(BX* RPB) void interp_vec(
    const typename Tr::scalar* __restrict__ W, typename Tr::scalar* __restrict__ O,
    const int4* __restrict__ idx, const float4* __restrict__ cf,
    const int4* __restrict__ meta, const int* __restrict__ dst,
    int nv, int D, int nrows, int vpc) {
  constexpr int NE = Tr::NE;
  const int row = blockIdx.x * RPB + threadIdx.y;
  if (row >= nrows) return;
  const int cend = min(nv, (int)(blockIdx.y + 1) * vpc);
  const int4 ix = idx[row];
  const float4 c = cf[row];
  const int4 mt = meta[row];                 // (dst_off, ndst, row_in_block, dst0)
  const uint2* A = reinterpret_cast<const uint2*>(W + (long long)ix.x * D);
  const uint2* B = reinterpret_cast<const uint2*>(W + (long long)ix.y * D);
  const uint2* C = reinterpret_cast<const uint2*>(W + (long long)ix.z * D);
  const uint2* E = reinterpret_cast<const uint2*>(W + (long long)ix.w * D);
  uint2* O0 = reinterpret_cast<uint2*>(O + (long long)mt.w * D);
  for (int i = blockIdx.y * vpc + threadIdx.x; i < cend; i += BX) {
    uint2 va = A[i], vb = B[i], vc = C[i], ve = E[i];
    float fa[NE], fb[NE], fc[NE], fe[NE], fo[NE];
    Tr::unpack(va, fa); Tr::unpack(vb, fb);
    Tr::unpack(vc, fc); Tr::unpack(ve, fe);
#pragma unroll
    for (int k = 0; k < NE; ++k)
      fo[k] = c.x * fa[k] + c.y * fb[k] + c.z * fc[k] + c.w * fe[k];
    const uint2 v = Tr::pack(fo);
    O0[i] = v;
    if (MULTI) {
      for (int k = 1; k < mt.y; ++k)
        reinterpret_cast<uint2*>(O + (long long)(dst[mt.x + k] + mt.z) * D)[i] = v;
    }
  }
}

// Scalar fallback: any D / alignment the vector kernel cannot take.
template <typename Tr>
__global__ __launch_bounds__(BX* RPB) void interp_sca(
    const typename Tr::scalar* __restrict__ W, typename Tr::scalar* __restrict__ O,
    const int4* __restrict__ idx, const float4* __restrict__ cf,
    const int4* __restrict__ meta, const int* __restrict__ dst, int D, int nrows) {
  using S = typename Tr::scalar;
  const int row = blockIdx.x * RPB + threadIdx.y;
  if (row >= nrows) return;
  const int4 ix = idx[row];
  const float4 c = cf[row];
  const int4 mt = meta[row];
  const S* A = W + (long long)ix.x * D;
  const S* B = W + (long long)ix.y * D;
  const S* C = W + (long long)ix.z * D;
  const S* E = W + (long long)ix.w * D;
  for (int i = threadIdx.x; i < D; i += BX) {
    S v = Tr::fromF(c.x * Tr::toF(A[i]) + c.y * Tr::toF(B[i]) +
                    c.z * Tr::toF(C[i]) + c.w * Tr::toF(E[i]));
    O[(long long)mt.w * D + i] = v;
    for (int k = 1; k < mt.y; ++k) O[(long long)(dst[mt.x + k] + mt.z) * D + i] = v;
  }
}

// Enough warp-tasks to fill the device; below this each warp takes a whole row
// (fewer descriptor reads), above it rows are split into BX-vector chunks.
#define TARGET_WARPS 6144

template <typename Tr>
static void launch(const at::Tensor& w, at::Tensor& out, const int4* idx,
                   const float4* cf, const int4* meta, const int* dst,
                   int64_t D, int64_t nrows, bool multi, cudaStream_t s) {
  using S = typename Tr::scalar;
  const S* wp = reinterpret_cast<const S*>(w.const_data_ptr());
  S* op = reinterpret_cast<S*>(out.data_ptr());
  const dim3 block(BX, RPB);
  const uintptr_t a = (uintptr_t)wp | (uintptr_t)op |
                      (uintptr_t)(D * (int64_t)sizeof(S));
  if ((a & 7) == 0 && D % Tr::NE == 0) {
    const int nv = (int)(D / Tr::NE);
    int per_row = (int)((TARGET_WARPS + nrows - 1) / nrows);           // chunks/row
    per_row = std::max(1, std::min(per_row, (nv + BX - 1) / BX));
    int vpc = ((nv + per_row - 1) / per_row + BX - 1) / BX * BX;       // multiple of BX
    const int chunks = (nv + vpc - 1) / vpc;
    const dim3 grid((unsigned)((nrows + RPB - 1) / RPB), (unsigned)chunks);
    if (multi)
      interp_vec<Tr, true><<<grid, block, 0, s>>>(wp, op, idx, cf, meta, dst, nv,
                                                 (int)D, (int)nrows, vpc);
    else
      interp_vec<Tr, false><<<grid, block, 0, s>>>(wp, op, idx, cf, meta, dst, nv,
                                                   (int)D, (int)nrows, vpc);
  } else {
    const dim3 grid((unsigned)((nrows + RPB - 1) / RPB));
    interp_sca<Tr><<<grid, block, 0, s>>>(wp, op, idx, cf, meta, dst, (int)D,
                                          (int)nrows);
  }
}

// ``idx``/``cf`` are the memoized descriptor tables: one row each per output row.
at::Tensor run(const at::Tensor& w, const at::Tensor& idx, const at::Tensor& cf,
               const at::Tensor& meta, const at::Tensor& dst, int64_t nrows,
               int64_t total_rows, bool multi) {
  const int64_t D = w.size(1);
  at::Tensor out = at::empty({total_rows, D}, w.options());
  if (nrows == 0 || D == 0) return out;
  TORCH_CHECK(idx.numel() >= 4 * nrows && cf.numel() >= 4 * nrows &&
              meta.numel() >= 4 * nrows, "short descriptor");
  const int4* ip = reinterpret_cast<const int4*>(idx.const_data_ptr<int>());
  const float4* cp = reinterpret_cast<const float4*>(cf.const_data_ptr<float>());
  const int4* mp = reinterpret_cast<const int4*>(meta.const_data_ptr<int>());
  const int* dp = dst.const_data_ptr<int>();
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  if (w.scalar_type() == at::kBFloat16)   launch<TrBF16>(w, out, ip, cp, mp, dp, D, nrows, multi, s);
  else if (w.scalar_type() == at::kHalf)  launch<TrFP16>(w, out, ip, cp, mp, dp, D, nrows, multi, s);
  else if (w.scalar_type() == at::kFloat) launch<TrFP32>(w, out, ip, cp, mp, dp, D, nrows, multi, s);
  else TORCH_CHECK(false, "unsupported dtype");
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "fused bilinear position-embedding interpolation");
}
'''


def _build():
    """JIT-compile the fused kernel, pinned to the local GPU arch."""
    from torch.utils.cpp_extension import load_inline
    try:
        from fastkernels.infra.cuda_ext import _pin_build_arch
        _pin_build_arch()
    except Exception:
        try:
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        except Exception:
            pass
    return load_inline(
        name="fk_cand_vision_pos_embed_interp",
        cpp_sources="",
        cuda_sources=_CUDA_SRC,
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


try:
    _EXT = _build()
except Exception:  # no nvcc / unsupported toolchain -> reference path
    _EXT = None

_FAST_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_MAX_PLANS = 32          # bounded memo (descriptors are ~48 B / distinct row)
_MAX_SHAPES = 64
_MAX_DST = 1024          # destinations a single descriptor record scatters to


class VisionPosEmbedInterpolate(nn.Module):
    def __init__(self, num_position_embeddings: int, hidden_size: int,
                 spatial_merge_size: int):
        super().__init__()
        self._embed = Embedding(num_position_embeddings, hidden_size)
        self.num_grid_per_side = int(num_position_embeddings ** 0.5)
        self.spatial_merge_size = spatial_merge_size
        self.hidden_size = hidden_size
        self._plans: dict = {}    # grid signature -> launch plan
        self._shapes: dict = {}   # (h, w, dtype) -> that grid's (idx, cf)

    def _apply(self, *args, **kwargs):
        # A device / dtype move invalidates every memoized descriptor.
        self._plans.clear()
        self._shapes.clear()
        return super()._apply(*args, **kwargs)

    # -- descriptor construction (once per distinct grid shape) -------------
    def _shape_desc(self, h: int, w: int, dtype: torch.dtype, device):
        """(idx, cf) for one (h, w) grid, in final (merge-reshuffled) row order.

        Mirrors the baseline exactly: ``linspace`` stencil, ``floor``/``clamp``
        corners, the same four weight expressions in fp32, then ``.to(dtype)``
        -- so the coefficients are bit-identical to the ones it multiplies by.
        """
        key = (h, w, dtype)
        hit = self._shapes.get(key)
        if hit is not None:
            return hit
        g, m = self.num_grid_per_side, self.spatial_merge_size
        hi = torch.linspace(0, g - 1, h, dtype=torch.float32, device=device)
        wi = torch.linspace(0, g - 1, w, dtype=torch.float32, device=device)
        h_floor, w_floor = hi.long(), wi.long()
        h_ceil = torch.clamp(h_floor + 1, max=g - 1)
        w_ceil = torch.clamp(w_floor + 1, max=g - 1)
        dh, dw = hi - h_floor, wi - w_floor

        # Output row r = ((hb * (w/m) + wb) * m + hi) * m + wi  <=>  the
        # baseline's reshape(h/m, m, w/m, m, D).permute(0, 2, 1, 3, 4). Build the
        # (hb, wb, hi, wi) position grids and index the stencil with them.
        step = torch.arange(m, device=device)
        ho = (torch.arange(h // m, device=device).unsqueeze(1) * m + step).view(
            h // m, 1, m, 1).expand(h // m, w // m, m, m)
        wo = (torch.arange(w // m, device=device).unsqueeze(1) * m + step).view(
            1, w // m, 1, m).expand(h // m, w // m, m, m)

        h0, h1, w0, w1 = h_floor[ho], h_ceil[ho], w_floor[wo], w_ceil[wo]
        idx = torch.stack([h0 * g + w0, h0 * g + w1, h1 * g + w0, h1 * g + w1],
                          dim=-1).reshape(-1, 4).to(torch.int32).contiguous()
        dh_g, dw_g = dh[ho], dw[wo]
        w11 = dh_g * dw_g
        w10 = dh_g - w11
        w01 = dw_g - w11
        w00 = 1 - dh_g - w01
        cf = torch.stack([w00, w01, w10, w11], dim=-1).reshape(-1, 4).to(
            dtype).to(torch.float32).contiguous()
        if len(self._shapes) >= _MAX_SHAPES:
            self._shapes.pop(next(iter(self._shapes)))
        self._shapes[key] = (idx, cf)
        return idx, cf

    def _make_plan(self, grid_thw_list, dtype, device):
        """Descriptor table over *distinct* rows + where each one is stored.

        Entries sharing an (h, w) -- all frames of a video, all images of one
        resolution -- have byte-identical output blocks, so they collapse to one
        set of descriptor rows plus a list of destination row offsets.
        """
        groups: dict = {}
        rows = 0
        for t, h, w in grid_thw_list:
            hw = h * w
            base = groups.setdefault((h, w), [])
            base.extend(rows + f * hw for f in range(t))
            rows += t * hw

        idxs, cfs, metas, dsts, nrows, multi = [], [], [], [], 0, False
        for (h, w), base in groups.items():
            idx, cf = self._shape_desc(h, w, dtype, device)
            n = h * w
            ar = torch.arange(n, dtype=torch.int32, device=device)
            for k in range(0, len(base), _MAX_DST):   # keep one meta record small
                chunk = base[k:k + _MAX_DST]
                idxs.append(idx)
                cfs.append(cf)
                metas.append(torch.stack(
                    [torch.full_like(ar, len(dsts)), torch.full_like(ar, len(chunk)),
                     ar, ar + chunk[0]], dim=-1))
                multi |= len(chunk) > 1
                dsts.extend(chunk)
                nrows += n
        idx = idxs[0] if len(idxs) == 1 else torch.cat(idxs)
        cf = cfs[0] if len(cfs) == 1 else torch.cat(cfs)
        meta = metas[0] if len(metas) == 1 else torch.cat(metas)
        dst = torch.tensor(dsts, dtype=torch.int32).to(device)
        return (idx.contiguous(), cf.contiguous(), meta.contiguous(), dst, nrows,
                rows, multi)

    def _fast_ok(self, grid_thw_list, dtype) -> bool:
        w = self._embed.emb.weight
        if _EXT is None or not w.is_cuda or w.dim() != 2 or not w.is_contiguous():
            return False
        if w.dtype is not dtype or dtype not in _FAST_DTYPES:
            return False
        m = self.spatial_merge_size
        if m < 1 or not grid_thw_list:
            return False
        for g in grid_thw_list:
            if len(g) != 3:
                return False
            t, h, ww = g
            if t < 1 or h < 1 or ww < 1 or h % m or ww % m:
                return False
        return True

    def _reference(self, grid_thw_list, dtype, device):
        num_grid = self.num_grid_per_side
        m_size = self.spatial_merge_size
        hidden_dim = self.hidden_size

        outputs = []
        for t, h, w in grid_thw_list:
            h_idxs = torch.linspace(0, num_grid - 1, h, dtype=torch.float32, device=device)
            w_idxs = torch.linspace(0, num_grid - 1, w, dtype=torch.float32, device=device)

            h_floor = h_idxs.long()
            w_floor = w_idxs.long()
            h_ceil = torch.clamp(h_floor + 1, max=num_grid - 1)
            w_ceil = torch.clamp(w_floor + 1, max=num_grid - 1)

            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
            h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
            h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")

            w11 = dh_grid * dw_grid
            w10 = dh_grid - w11
            w01 = dw_grid - w11
            w00 = 1 - dh_grid - w01

            h_grid = torch.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
            w_grid = torch.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
            indices = (h_grid * num_grid + w_grid).reshape(4, -1)
            weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1).to(dtype=dtype)

            embeds = self._embed(indices) * weights
            combined = embeds.sum(dim=0)
            combined = combined.reshape(
                h // m_size, m_size, w // m_size, m_size, hidden_dim
            ).permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
            repeated = combined.expand(t, -1, -1).reshape(-1, hidden_dim)
            outputs.append(repeated)

        return torch.cat(outputs, dim=0)

    def forward(
        self,
        grid_thw_list: list[list[int]],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        w = self._embed.emb.weight
        # ``_prepare_module``-style recasts replace ``weight.data`` without going
        # through ``_apply``, so the weight dtype is part of the memo key.
        key = (tuple(map(tuple, grid_thw_list)), dtype, w.dtype)
        plan = self._plans.get(key)
        if plan is None:
            if not self._fast_ok(grid_thw_list, dtype):
                return self._reference(grid_thw_list, dtype, device)
            plan = self._make_plan(grid_thw_list, dtype, w.device)
            if len(self._plans) >= _MAX_PLANS:
                self._plans.pop(next(iter(self._plans)))
            self._plans[key] = plan
        return _EXT.run(w, *plan)
