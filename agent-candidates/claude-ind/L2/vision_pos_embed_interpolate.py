"""Bilinear interpolation of learned 2D position embeddings (Qwen3-VL).

Owns a learned embedding weight of (num_grid_per_side^2, hidden_size).
forward() interpolates these onto arbitrary (h, w) grids using bilinear
weights, then reshuffles by spatial_merge_size for the vision encoder.

Fused single-kernel implementation: the whole per-call pipeline (linspace ->
floor/ceil corner indices -> bilinear weights -> 4-way gather/accumulate ->
spatial_merge shuffle -> temporal repeat -> concat over images) collapses into
one CUDA kernel that writes the final concatenated tensor directly.  All
per-image metadata rides in the kernel parameter block, so there is no host
side tensor construction, no H2D copy and exactly one launch per call.

Images sharing the same (h, w) produce bit-identical rows, so they are grouped:
the gather/interpolate runs once per distinct (h, w) and the result is written
to every destination slice (including the ``t`` temporal repeats).
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from ..L1.embedding import Embedding

_CUDA_SRC = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <algorithm>
#include <vector>

#define MAXG 32
#define MAXD 192

// Per-call metadata: rides in the kernel parameter block (no H2D copy).
struct MetaBlk {
  int h[MAXG];
  int w[MAXG];
  int hm[MAXG];   // h / merge
  int wm[MAXG];   // w / merge
  int nd[MAXG];   // number of destination row-slices
  int dbeg[MAXG]; // offset into dest[]
  float sh[MAXG]; // linspace step along h
  float sw[MAXG]; // linspace step along w
  int dest[MAXD];
};

// Bit-exact replica of ATen's linspace(0, endv, steps) element ``i`` in fp32
// (RangeFactories.cu: halfway split, fma on the upper half).  ``step`` is
// endv/(steps-1) rounded in fp32 on the host, which matches div.rn.f32.
__device__ __forceinline__ float lin_idx(int i, int steps, float endv, float step) {
  if (steps <= 1) return 0.0f;
  return (i < (steps >> 1)) ? (step * (float)i)
                            : __fmaf_rn(-step, (float)(steps - i - 1), endv);
}

__device__ __forceinline__ float to_f(const __nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float to_f(const __half x) { return __half2float(x); }
__device__ __forceinline__ float to_f(const float x) { return x; }

template <typename T> __device__ __forceinline__ T from_f(float x);
template <> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float x) {
  return __float2bfloat16(x);
}
template <> __device__ __forceinline__ __half from_f<__half>(float x) { return __float2half(x); }
template <> __device__ __forceinline__ float from_f<float>(float x) { return x; }

template <typename T, int VPT>
struct alignas((VPT * (int)sizeof(T) >= 16) ? 16 : (int)sizeof(T)) Pack {
  T e[VPT];
};

template <typename T, int VPT>
__device__ __forceinline__ Pack<T, VPT> ld_pack(const T* __restrict__ p) {
  Pack<T, VPT> r;
  if (VPT * sizeof(T) == 16) {
    *reinterpret_cast<uint4*>(&r) = *reinterpret_cast<const uint4*>(p);
  } else {
#pragma unroll
    for (int e = 0; e < VPT; ++e) r.e[e] = p[e];
  }
  return r;
}

template <typename T, int VPT>
__device__ __forceinline__ void st_pack(T* __restrict__ p, const Pack<T, VPT>& r) {
  if (VPT * sizeof(T) == 16) {
    *reinterpret_cast<uint4*>(p) = *reinterpret_cast<const uint4*>(&r);
  } else {
#pragma unroll
    for (int e = 0; e < VPT; ++e) p[e] = r.e[e];
  }
}

// One block owns one output row (token) of one group, plus a chunk of that
// group's destination slices, and walks the hidden dim in VPT-wide vectors.
// Blocks are indexed so the whole (merge-shuffled) token decode is a couple of
// shifts: no integer division, and the linspace step arrives precomputed from
// the host, so the per-token scalar setup is a few dozen instructions.
//   blockIdx.x = merge-col jj * MSZ^2 + sub-token, blockIdx.z = merge-row ii,
//   blockIdx.y = group * nchunk + dest chunk.
// TPB = 1 -> one token per block; TPB = 0 -> the whole MSZ*MSZ merge tile.
template <typename T, int VPT, int MSZ, int TPB>
__global__ void interp_kernel(const T* __restrict__ tbl, T* __restrict__ out, const MetaBlk md,
                              int ng, int hidden, int nvec, int ndpb, int chunk_mask,
                              int chunk_log) {
  constexpr int NT = MSZ * MSZ;
  constexpr int LOGM = (MSZ == 4) ? 2 : ((MSZ == 2) ? 1 : 0);
  constexpr int NTK = (TPB == 1) ? 1 : NT;

  const int g = blockIdx.y >> chunk_log;
  const int ii = blockIdx.z;
  const int jj = (TPB == 1) ? (int)(blockIdx.x >> (2 * LOGM)) : (int)blockIdx.x;
  const int hm = md.hm[g];
  const int wm = md.wm[g];
  if (ii >= hm || jj >= wm) return;
  const int nd = md.nd[g];
  const int d0 = (blockIdx.y & chunk_mask) * ndpb;
  if (d0 >= nd) return;
  const int d1 = min(nd, d0 + ndpb);
  const int dbeg = md.dbeg[g] + d0;
  const int ndd = d1 - d0;

  const int h = md.h[g];
  const int w = md.w[g];
  const float endv = (float)(ng - 1);
  const int tk0 = (TPB == 1) ? (int)(blockIdx.x & (NT - 1)) : 0;
  const int qbase = (ii * wm + jj) * NT;

  int r00[NTK], r01[NTK], r10[NTK], r11[NTK], qq[NTK];
  float c00[NTK], c01[NTK], c10[NTK], c11[NTK];
#pragma unroll
  for (int k = 0; k < NTK; ++k) {
    const int tk = tk0 + k;
    const int a = tk >> LOGM;
    const int b = tk & (MSZ - 1);
    const float xh = lin_idx(ii * MSZ + a, h, endv, md.sh[g]);
    const float xw = lin_idx(jj * MSZ + b, w, endv, md.sw[g]);
    const int fh = (int)xh;
    const int fw = (int)xw;
    const int ch = min(fh + 1, ng - 1);
    const int cw = min(fw + 1, ng - 1);
    const float dh = xh - (float)fh;
    const float dw = xw - (float)fw;
    const float t11 = dh * dw;
    c11[k] = t11;
    c10[k] = dh - t11;
    c01[k] = dw - t11;
    c00[k] = 1.0f - dh - (dw - t11);
    r00[k] = fh * ng + fw;
    r01[k] = fh * ng + cw;
    r10[k] = ch * ng + fw;
    r11[k] = ch * ng + cw;
    qq[k] = qbase + tk;
  }

  const int tid = threadIdx.x;
  const int nthr = blockDim.x;
  for (int v = tid; v < nvec; v += nthr) {
    const int o = v * VPT;
#pragma unroll
    for (int k = 0; k < NTK; ++k) {
      Pack<T, VPT> a00 = ld_pack<T, VPT>(tbl + (long)r00[k] * hidden + o);
      Pack<T, VPT> a01 = ld_pack<T, VPT>(tbl + (long)r01[k] * hidden + o);
      Pack<T, VPT> a10 = ld_pack<T, VPT>(tbl + (long)r10[k] * hidden + o);
      Pack<T, VPT> a11 = ld_pack<T, VPT>(tbl + (long)r11[k] * hidden + o);
      Pack<T, VPT> res;
#pragma unroll
      for (int e = 0; e < VPT; ++e) {
        float acc = c00[k] * to_f(a00.e[e]);
        acc = __fmaf_rn(c01[k], to_f(a01.e[e]), acc);
        acc = __fmaf_rn(c10[k], to_f(a10.e[e]), acc);
        acc = __fmaf_rn(c11[k], to_f(a11.e[e]), acc);
        res.e[e] = from_f<T>(acc);
      }
      for (int d = 0; d < ndd; ++d) {
        st_pack<T, VPT>(out + (long)(md.dest[dbeg + d] + qq[k]) * hidden + o, res);
      }
    }
  }
}

template <typename T, int VPT, int MSZ, int TPB>
static void launch_tpb(const T* tbl, T* out, const MetaBlk& md, int ngroup, int ng, int hidden,
                       int maxhm, int maxwm, int maxnd, int ndpb, cudaStream_t stream) {
  const int nvec = hidden / VPT;
  int nthr = ((nvec + 31) / 32) * 32;
  if (nthr > 512) nthr = 512;
  if (nthr < 32) nthr = 32;
  int nchunk = (maxnd + ndpb - 1) / ndpb;
  int chunk_log = 0;
  while ((1 << chunk_log) < nchunk) ++chunk_log;
  nchunk = 1 << chunk_log;
  const int nx = (TPB == 1) ? maxwm * MSZ * MSZ : maxwm;
  dim3 grid(nx, ngroup * nchunk, maxhm);
  interp_kernel<T, VPT, MSZ, TPB><<<grid, nthr, 0, stream>>>(tbl, out, md, ng, hidden, nvec, ndpb,
                                                             nchunk - 1, chunk_log);
}

template <typename T, int VPT, int MSZ>
static void launch_msz(const T* tbl, T* out, const MetaBlk& md, int ngroup, int ng, int hidden,
                       int maxhm, int maxwm, int maxnd, int ndpb, int tpb, cudaStream_t stream) {
  if (tpb == 1) {
    launch_tpb<T, VPT, MSZ, 1>(tbl, out, md, ngroup, ng, hidden, maxhm, maxwm, maxnd, ndpb,
                               stream);
  } else {
    launch_tpb<T, VPT, MSZ, 0>(tbl, out, md, ngroup, ng, hidden, maxhm, maxwm, maxnd, ndpb,
                               stream);
  }
}

template <typename T, int VPT>
static void launch_vpt(const T* tbl, T* out, const MetaBlk& md, int ngroup, int ng, int msz,
                       int hidden, int maxhm, int maxwm, int maxnd, int ndpb, int tpb,
                       cudaStream_t stream) {
  if (msz == 2) {
    launch_msz<T, VPT, 2>(tbl, out, md, ngroup, ng, hidden, maxhm, maxwm, maxnd, ndpb, tpb,
                          stream);
  } else if (msz == 1) {
    launch_msz<T, VPT, 1>(tbl, out, md, ngroup, ng, hidden, maxhm, maxwm, maxnd, ndpb, tpb,
                          stream);
  } else {
    launch_msz<T, VPT, 4>(tbl, out, md, ngroup, ng, hidden, maxhm, maxwm, maxnd, ndpb, tpb,
                          stream);
  }
}

template <typename T>
static void launch_t(const T* tbl, T* out, const MetaBlk& md, int ngroup, int ng, int msz,
                     int hidden, int maxhm, int maxwm, int maxnd, int ndpb, int tpb,
                     cudaStream_t stream) {
  constexpr int VW = (int)(16 / sizeof(T));
  if (hidden % VW == 0) {
    launch_vpt<T, VW>(tbl, out, md, ngroup, ng, msz, hidden, maxhm, maxwm, maxnd, ndpb, tpb,
                      stream);
  } else {
    launch_vpt<T, 1>(tbl, out, md, ngroup, ng, msz, hidden, maxhm, maxwm, maxnd, ndpb, tpb,
                     stream);
  }
}

struct Grp {
  int h, w;
  int nd;
  int dest[MAXD];
};

at::Tensor pos_embed_interp(const at::Tensor& tbl, const std::vector<std::vector<int64_t>>& grids,
                            int64_t num_grid, int64_t msz, int64_t ndpb_in, int64_t tpb_in) {
  TORCH_CHECK(tbl.is_cuda() && tbl.dim() == 2 && tbl.is_contiguous());
  const int hidden = (int)tbl.size(1);
  const int ng = (int)num_grid;
  const int m = (int)msz;
  TORCH_CHECK(m == 1 || m == 2 || m == 4, "unsupported spatial_merge_size");

  const int nimg = (int)grids.size();
  std::vector<Grp> groups;
  groups.reserve(nimg);
  long total = 0;
  for (int i = 0; i < nimg; ++i) {
    TORCH_CHECK(grids[i].size() == 3, "grid_thw_list entries must be (t, h, w)");
    const long t = grids[i][0];
    const long h = grids[i][1];
    const long w = grids[i][2];
    TORCH_CHECK(t >= 0 && h >= 0 && w >= 0, "negative grid extent");
    TORCH_CHECK(h % m == 0 && w % m == 0, "grid not divisible by spatial_merge_size");
    const long hw = h * w;
    const long base = total;
    total += hw * t;
    TORCH_CHECK(total <= 0x7fffffffL, "too many output rows");
    // Every image with the same (h, w) interpolates to the same rows, and the t
    // temporal repeats are copies of each other: one group, many destinations.
    for (long s = 0; s < t;) {
      int gi = -1;
      for (int k = 0; k < (int)groups.size(); ++k) {
        if (groups[k].h == (int)h && groups[k].w == (int)w && groups[k].nd < MAXD) {
          gi = k;
          break;
        }
      }
      if (gi < 0) {
        groups.push_back(Grp{(int)h, (int)w, 0, {}});
        gi = (int)groups.size() - 1;
      }
      Grp& gp = groups[gi];
      while (s < t && gp.nd < MAXD) gp.dest[gp.nd++] = (int)(base + hw * s++);
    }
  }

  at::Tensor out = at::empty({total, (long)hidden}, tbl.options());
  if (total == 0) return out;

  const c10::cuda::CUDAGuard guard(tbl.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const float endv = (float)(ng - 1);

  size_t gi = 0;
  while (gi < groups.size()) {
    MetaBlk md;
    int ngroup = 0, ndest = 0, maxhm = 0, maxwm = 0, maxnd = 0;
    while (gi < groups.size() && ngroup < MAXG && ndest + groups[gi].nd <= MAXD) {
      const Grp& gp = groups[gi];
      md.h[ngroup] = gp.h;
      md.w[ngroup] = gp.w;
      md.hm[ngroup] = gp.h / m;
      md.wm[ngroup] = gp.w / m;
      md.sh[ngroup] = gp.h > 1 ? endv / (float)(gp.h - 1) : 0.0f;
      md.sw[ngroup] = gp.w > 1 ? endv / (float)(gp.w - 1) : 0.0f;
      md.nd[ngroup] = gp.nd;
      md.dbeg[ngroup] = ndest;
      for (int d = 0; d < gp.nd; ++d) md.dest[ndest + d] = gp.dest[d];
      ndest += gp.nd;
      maxhm = std::max(maxhm, md.hm[ngroup]);
      maxwm = std::max(maxwm, md.wm[ngroup]);
      maxnd = std::max(maxnd, gp.nd);
      ++ngroup;
      ++gi;
    }
    // One token per block is the default: the 4-way gather, the interpolation
    // and the stores to every destination slice are then each paid exactly once.
    // With no destination slices to share (no repeated (h, w) and t == 1) the
    // gather is the bottleneck instead, so fall back to a whole merge tile per
    // block -- its MSZ*MSZ tokens hit the same table rows and share them in L1 --
    // provided there are still plenty of tiles to fill the GPU.
    const long ntok = (long)maxhm * maxwm * m * m * ngroup;
    int tpb = (int)tpb_in;
    if (tpb < 0) tpb = (maxnd <= 2 && ntok >= 4096) ? 0 : 1;
    int ndpb = (int)ndpb_in;
    if (ndpb < 1) {
      // As many slices per block as possible, but keep >= ~512 blocks resident.
      const long nblk = tpb == 1 ? ntok : ntok / (m * m);
      long want = (512 + nblk - 1) / nblk;
      if (want < 1) want = 1;
      if (want > maxnd) want = maxnd;
      ndpb = (int)((maxnd + want - 1) / want);
    }
    if (ndpb < 1) ndpb = 1;
    if (ndpb > maxnd) ndpb = maxnd;
    AT_DISPATCH_SWITCH(
        tbl.scalar_type(), "pos_embed_interp",
        AT_DISPATCH_CASE(at::kBFloat16,
                         [&] {
                           launch_t<__nv_bfloat16>((const __nv_bfloat16*)tbl.data_ptr(),
                                                   (__nv_bfloat16*)out.data_ptr(), md, ngroup, ng,
                                                   m, hidden, maxhm, maxwm, maxnd, ndpb, tpb, stream);
                         })
            AT_DISPATCH_CASE(at::kHalf,
                             [&] {
                               launch_t<__half>((const __half*)tbl.data_ptr(),
                                                (__half*)out.data_ptr(), md, ngroup, ng, m, hidden,
                                                maxhm, maxwm, maxnd, ndpb, tpb, stream);
                             })
                AT_DISPATCH_CASE(at::kFloat, [&] {
                  launch_t<float>((const float*)tbl.data_ptr(), (float*)out.data_ptr(), md, ngroup,
                                  ng, m, hidden, maxhm, maxwm, maxnd, ndpb, tpb, stream);
                }));
  }
  return out;
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
at::Tensor pos_embed_interp(const at::Tensor& tbl,
                            const std::vector<std::vector<int64_t>>& grids,
                            int64_t num_grid, int64_t msz, int64_t ndpb, int64_t tpb);
"""


def _build_ext():
    from torch.utils.cpp_extension import load_inline

    # Build only for the GPU we are going to run on (a full-arch build takes
    # minutes).  Restored afterwards so later compiles are unaffected.
    prev_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if torch.cuda.is_available() and not prev_arch:
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    try:
        return _load(load_inline)
    finally:
        if prev_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev_arch


def _load(load_inline):
    return load_inline(
        name="fk_vision_pos_embed_interp",
        cpp_sources=[_CPP_SRC],
        cuda_sources=[_CUDA_SRC],
        functions=["pos_embed_interp"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        verbose=False,
    )


_EXT = None
try:
    _EXT = _build_ext()
except Exception:  # pragma: no cover - fall back to the reference path
    _EXT = None

_NDPB = int(os.environ.get("FK_VPE_NDPB", "0"))
_TPB = int(os.environ.get("FK_VPE_TPB", "-1"))


class VisionPosEmbedInterpolate(nn.Module):
    def __init__(self, num_position_embeddings: int, hidden_size: int,
                 spatial_merge_size: int):
        super().__init__()
        self._embed = Embedding(num_position_embeddings, hidden_size)
        self.num_grid_per_side = int(num_position_embeddings ** 0.5)
        self.spatial_merge_size = spatial_merge_size
        self.hidden_size = hidden_size

    def forward(
        self,
        grid_thw_list: list[list[int]],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        weight = self._embed.emb.weight
        if (_EXT is not None and weight.is_cuda and weight.is_contiguous()
                and device.type == "cuda"
                and (device.index is None or device.index == weight.device.index)):
            out_dtype = torch.promote_types(weight.dtype, dtype)
            if out_dtype == weight.dtype and out_dtype in (
                    torch.bfloat16, torch.float16, torch.float32):
                return _EXT.pos_embed_interp(
                    weight, grid_thw_list, self.num_grid_per_side,
                    self.spatial_merge_size, _NDPB, _TPB)
        return self._reference(grid_thw_list, dtype, device)

    def _reference(
        self,
        grid_thw_list: list[list[int]],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
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
