"""Vision encoder rotary position embeddings (fused CUDA).

Same semantics as the baseline: a cos/sin cache is precomputed from fixed
inv_freq (base=10000, no scaling) and ``forward()`` gathers it with 2D
(height, width) position ids built from ``grid_thw_list`` and shuffled by
``spatial_merge_size``.

The baseline spends all of its time on the host: a per-image numpy loop builds
the (N, 2) position ids, ships them over PCIe, then torch runs a dtype cast plus
two fancy-index gathers. For these shapes (N <= ~32k rows, 36 columns) that is
pure launch/CPU overhead -- the GPU work is a few microseconds.

Here the whole thing is one kernel launch. The grid metadata never leaves the
kernel launch packet (it is passed by value in a small struct, so there is no
H2D copy and no extra allocation), the position ids are decoded arithmetically
inside the kernel, and cos/sin are read straight out of the fp32 cache and
converted on the fly -- bit-identical to the baseline's ``cache.to(dtype)``
followed by a gather.

Kernel mapping: ``blockIdx.y`` = image, ``blockIdx.z`` = frame within the
image's temporal extent, ``blockIdx.x``/``threadIdx.y`` = flattened spatial
position, ``threadIdx.x`` = which pair of output columns. Consecutive threads
therefore cover consecutive output elements, so both the 4-byte stores and the
float2 cache loads are fully coalesced.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

_CUDA_SRC = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#define FK_MAX_IMGS 96

struct FKMeta {
    int off[FK_MAX_IMGS];        // first output row of image i
    int hw[FK_MAX_IMGS];         // h * w
    int wblk[FK_MAX_IMGS];       // w / spatial_merge_size
    int nt[FK_MAX_IMGS];         // temporal extent t
    float inv_wblk[FK_MAX_IMGS]; // 1 / wblk
};

// a / b for small non-negative a and b >= 1, via a float reciprocal plus one
// exact correction step (integer division on GPU costs ~20 instructions and
// this sits on the critical path of every thread).
__device__ __forceinline__ int fk_div(int a, int b, float inv_b) {
    int q = __float2int_rz(__int2float_rn(a) * inv_b);
    int r = a - q * b;
    if (r < 0) q -= 1;
    else if (r >= b) q += 1;
    return q;
}

template <typename T>
struct alignas(2 * sizeof(T)) FKPair { T x, y; };

template <typename T>
__global__ void fk_vision_rope_kernel(
        const float* __restrict__ cache,
        T* __restrict__ cos_out,
        T* __restrict__ sin_out,
        const FKMeta meta,
        int hd,          // cache row width == output row width (rotary_dim)
        int npair,       // hd / 2 : threads per output row
        int sms,
        float inv_sms) {
    const int img = blockIdx.y;
    if (blockIdx.z >= meta.nt[img]) return;
    const int hw = meta.hw[img];
    const int p = blockIdx.x * blockDim.y + threadIdx.y;   // spatial position
    if (p >= hw) return;

    // Undo the spatial_merge_size shuffle: the flattened order is
    // (h/sms, w/sms, sms, sms) -> (hb, wb, hi, wi).
    const int wblk = meta.wblk[img];
    const int t1 = fk_div(p, sms, inv_sms);
    const int wi = p - t1 * sms;
    const int t2 = fk_div(t1, sms, inv_sms);
    const int hi = t1 - t2 * sms;
    const int hb = fk_div(t2, wblk, meta.inv_wblk[img]);
    const int wb = t2 - hb * wblk;

    // Columns [0, hd/2) come from the h position, [hd/2, hd) from the w one.
    const int k = threadIdx.x;               // pair index in [0, npair)
    const int half_pairs = npair >> 1;
    const bool is_w = (k >= half_pairs);
    const int pos = is_w ? (wb * sms + wi) : (hb * sms + hi);
    const int kk = is_w ? (k - half_pairs) : k;

    const float2* src = reinterpret_cast<const float2*>(cache + (long long)pos * hd);
    const float2 c = src[kk];                 // cos half of the cache row
    const float2 s = src[half_pairs + kk];    // sin half

    const long long row = (long long)meta.off[img] + (long long)blockIdx.z * hw + p;
    const long long e = row * hd;
    reinterpret_cast<FKPair<T>*>(cos_out + e)[k] =
        FKPair<T>{static_cast<T>(c.x), static_cast<T>(c.y)};
    reinterpret_cast<FKPair<T>*>(sin_out + e)[k] =
        FKPair<T>{static_cast<T>(s.x), static_cast<T>(s.y)};
}

void fk_vision_rope_launch(
        const at::Tensor& cache, at::Tensor& cos_out, at::Tensor& sin_out,
        const FKMeta& meta, int n_img, int max_hw, int max_t, int sms, int hd) {
    const int npair = hd / 2;
    int rows_per_block = 256 / npair;
    if (rows_per_block < 1) rows_per_block = 1;
    const dim3 block(npair, rows_per_block);
    const dim3 grid((max_hw + rows_per_block - 1) / rows_per_block, n_img, max_t);
    auto stream = at::cuda::getCurrentCUDAStream();
    const float* cp = cache.data_ptr<float>();
    const float inv_sms = 1.0f / (float)sms;

    switch (cos_out.scalar_type()) {
    case at::kBFloat16:
        fk_vision_rope_kernel<at::BFloat16><<<grid, block, 0, stream>>>(
            cp, cos_out.data_ptr<at::BFloat16>(), sin_out.data_ptr<at::BFloat16>(),
            meta, hd, npair, sms, inv_sms);
        break;
    case at::kHalf:
        fk_vision_rope_kernel<at::Half><<<grid, block, 0, stream>>>(
            cp, cos_out.data_ptr<at::Half>(), sin_out.data_ptr<at::Half>(),
            meta, hd, npair, sms, inv_sms);
        break;
    default:
        fk_vision_rope_kernel<float><<<grid, block, 0, stream>>>(
            cp, cos_out.data_ptr<float>(), sin_out.data_ptr<float>(),
            meta, hd, npair, sms, inv_sms);
        break;
    }
}
"""

_CPP_SRC = r"""
#include <tuple>

#define FK_MAX_IMGS 96

struct FKMeta {
    int off[FK_MAX_IMGS];
    int hw[FK_MAX_IMGS];
    int wblk[FK_MAX_IMGS];
    int nt[FK_MAX_IMGS];
    float inv_wblk[FK_MAX_IMGS];
};

void fk_vision_rope_launch(
    const at::Tensor& cache, at::Tensor& cos_out, at::Tensor& sin_out,
    const FKMeta& meta, int n_img, int max_hw, int max_t, int sms, int hd);

namespace {

// Refcount guard for the PySequence_Fast results below.
struct FastSeq {
    PyObject* p{nullptr};
    ~FastSeq() { Py_XDECREF(p); }
};

inline long fk_as_long(PyObject* o) {
    if (PyLong_Check(o)) {
        long v = PyLong_AsLong(o);
        if (v == -1 && PyErr_Occurred()) { PyErr_Clear(); throw std::runtime_error("fk: bad int"); }
        return v;
    }
    Py_ssize_t v = PyNumber_AsSsize_t(o, nullptr);
    if (v == -1 && PyErr_Occurred()) { PyErr_Clear(); throw std::runtime_error("fk: bad int"); }
    return (long)v;
}

}  // namespace

// Returns (cos, sin), each (total_rows, rotary_dim) in the requested dtype.
// Throws (-> python falls back to the reference path) for anything the kernel
// does not cover.
std::tuple<at::Tensor, at::Tensor> fk_vision_rope(
        const at::Tensor& cache, pybind11::object grids, int64_t sms, int64_t dtype_code) {
    TORCH_CHECK(cache.is_cuda() && cache.is_contiguous() && cache.dim() == 2
                && cache.scalar_type() == at::kFloat, "fk: unusable cache");
    const int hd = (int)cache.size(1);
    const int max_pos = (int)cache.size(0);
    TORCH_CHECK(hd >= 4 && hd % 4 == 0 && hd / 2 <= 1024, "fk: unsupported rotary_dim");
    TORCH_CHECK(sms >= 1, "fk: bad spatial_merge_size");
    const int smsi = (int)sms;

    FastSeq outer;
    outer.p = PySequence_Fast(grids.ptr(), "fk: grid_thw_list must be a sequence");
    if (outer.p == nullptr) { PyErr_Clear(); throw std::runtime_error("fk: bad grid_thw_list"); }
    const Py_ssize_t n_img = PySequence_Fast_GET_SIZE(outer.p);
    TORCH_CHECK(n_img > 0 && n_img <= FK_MAX_IMGS, "fk: unsupported image count");
    PyObject** items = PySequence_Fast_ITEMS(outer.p);

    FKMeta meta;
    long long total = 0;
    int max_hw = 0, max_t = 0;
    for (Py_ssize_t i = 0; i < n_img; ++i) {
        FastSeq inner;
        inner.p = PySequence_Fast(items[i], "fk: grid entry must be a sequence");
        if (inner.p == nullptr) { PyErr_Clear(); throw std::runtime_error("fk: bad grid entry"); }
        TORCH_CHECK(PySequence_Fast_GET_SIZE(inner.p) == 3, "fk: grid entry must be (t, h, w)");
        PyObject** thw = PySequence_Fast_ITEMS(inner.p);
        const long t = fk_as_long(thw[0]);
        const long h = fk_as_long(thw[1]);
        const long w = fk_as_long(thw[2]);
        TORCH_CHECK(t >= 1 && h >= 1 && w >= 1, "fk: bad grid");
        TORCH_CHECK(h % smsi == 0 && w % smsi == 0, "fk: grid not divisible by merge size");
        TORCH_CHECK(h <= max_pos && w <= max_pos, "fk: grid exceeds cache");
        const int hw = (int)(h * w);
        meta.off[i] = (int)total;
        meta.hw[i] = hw;
        meta.wblk[i] = (int)(w / smsi);
        meta.nt[i] = (int)t;
        meta.inv_wblk[i] = 1.0f / (float)meta.wblk[i];
        total += (long long)t * hw;
        if (hw > max_hw) max_hw = hw;
        if ((int)t > max_t) max_t = (int)t;
    }
    TORCH_CHECK(total <= (long long)INT32_MAX, "fk: too many rows");

    const at::ScalarType st = dtype_code == 0 ? at::kBFloat16
                            : dtype_code == 1 ? at::kHalf : at::kFloat;
    // One allocation, two contiguous views: halves the allocator traffic on a
    // call whose cost is dominated by host-side work.
    at::Tensor buf = at::empty({2, total, hd},
                               at::TensorOptions().dtype(st).device(cache.device()));
    at::Tensor cos_out = buf.select(0, 0);
    at::Tensor sin_out = buf.select(0, 1);
    if (total > 0) {
        fk_vision_rope_launch(cache, cos_out, sin_out, meta, (int)n_img,
                              max_hw, max_t, smsi, hd);
    }
    return std::make_tuple(cos_out, sin_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vision_rope", &fk_vision_rope, "fused vision rotary cos/sin",
          pybind11::arg("cache"), pybind11::arg("grid_thw_list"),
          pybind11::arg("spatial_merge_size"), pybind11::arg("dtype_code"));
}
"""


def _load_ext():
    try:
        import os

        from torch.utils.cpp_extension import load_inline
        # Build for the local arch only: the default list spans six of them, which
        # is ~5x the cold-start JIT cost for no benefit here.
        if "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        return load_inline(
            name="fk_vision_rotary_emb",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            verbose=False,
        )
    except Exception:
        return None


_EXT = _load_ext()
_DTYPE_CODE = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, rotary_dim: int, max_grid_size: int = 8192):
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim
        ))
        t = torch.arange(max_grid_size, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
        self,
        grid_thw_list: list[list[int]],
        spatial_merge_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if _EXT is not None:
            code = _DTYPE_CODE.get(dtype)
            if code is not None:
                # __dict__ hit: skips nn.Module.__getattr__ on the hot path while
                # still seeing whatever device/dtype .to() left the buffer in.
                cache = self._buffers["cos_sin_cache"]
                try:
                    return _EXT.vision_rope(
                        cache, grid_thw_list, spatial_merge_size, code)
                except Exception:
                    pass
        return self._reference(grid_thw_list, spatial_merge_size, dtype, device)

    def _reference(
        self,
        grid_thw_list: list[list[int]],
        spatial_merge_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sms = spatial_merge_size
        pos_ids = []
        max_grid_size = 0
        for t, h, w in grid_thw_list:
            hpos = np.broadcast_to(np.arange(h).reshape(h, 1), (h, w))
            wpos = np.broadcast_to(np.arange(w).reshape(1, w), (h, w))
            hpos = hpos.reshape(h // sms, sms, w // sms, sms).transpose(0, 2, 1, 3).flatten()
            wpos = wpos.reshape(h // sms, sms, w // sms, sms).transpose(0, 2, 1, 3).flatten()
            hw = np.stack([hpos, wpos], axis=-1)
            pos_ids.append(np.tile(hw, (t, 1)) if t > 1 else hw)
            max_grid_size = max(max_grid_size, h, w)
        pos_ids = torch.from_numpy(np.concatenate(pos_ids, axis=0)).to(device)

        cache = self.cos_sin_cache[:max_grid_size].to(dtype=dtype)
        cos, sin = cache.chunk(2, dim=-1)
        return cos[pos_ids].flatten(1), sin[pos_ids].flatten(1)
