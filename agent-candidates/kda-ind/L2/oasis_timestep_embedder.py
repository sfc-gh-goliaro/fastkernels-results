"""Oasis timestep embedding, fused into two kernels behind one extension call.

The operator is a sinusoidal timestep embedding followed by a two-layer MLP. At the captured
sizes (``hidden_size=1024``, ``frequency_embedding_size=256``, ``t`` of length 2..6) it touches
about 5.25 MiB of weights and does about 16 MFLOP, both trivial for a B200 -- so it is
overhead-bound, and kernel count is the lever. The eager baseline issues thirteen kernels; this
issues two:

* the first computes the embedding into shared memory, then the first matrix-vector product,
  the bias and SiLU into ``h``;
* the second computes the second matrix-vector product and its bias into the output.

Both are launched from one host function on the current stream, so the first kernel's
completion orders the second's reads of ``h`` with no explicit barrier. A single fused kernel
with a grid-wide barrier would remove one launch gap and most of the first kernel's execution
time, and programmatic stream serialization would hide the first kernel behind the second's
4 MiB weight read; both are left for later work.

The arithmetic reproduces the baseline's tensor-float-32 operand quantization rather than
strict fp32. ``torch.backends.cuda.matmul.allow_tf32`` is on by default here, so the baseline's
two ``addmm`` calls round their operands to 10 explicit significand bits; an exact fp32 or fp64
recomputation matches only 83-85 % of elements against that and fails the harness's 99 % rule.
Rounding every product operand round-to-nearest-even and accumulating in fp32 brings it inside
the tolerance. What is *not* claimed is bitwise equality with the tensor core. A residual of
order 1e-5 remains, and it belongs to cuBLAS rather than to this kernel: measured against an
fp64 sum of the same rounded operands, the tf32 tensor core carries about fourteen times the
error a plain fp32 sum of those operands does, and with tf32 disabled on both sides the two
implementations agree to 8e-8. So the residual cannot be narrowed from the CUDA cores -- doing
that would mean issuing the same `mma`, which is left for later work. The flag is read the way
ATen reads it and the kernels are templated on it, so a strict-fp32 environment gets strict-fp32
kernels at no per-call cost.

Anything the kernels do not specialize -- another size, a batch outside the instantiated range,
a non-int64 or non-contiguous ``t``, gradients enabled, or a build that failed -- falls through
to the eager path, which is the baseline computation.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

from ..L1.linear import Linear
from ..L1.silu import SiLU

# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <vector>

torch::Tensor oasis_embed_mlp(const torch::Tensor& t, const torch::Tensor& w1,
                              const torch::Tensor& b1, const torch::Tensor& w2,
                              const torch::Tensor& b2);
std::vector<std::vector<int64_t>> instantiated_configs();
"""

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/Context.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>

// Sizes and the shape of the work decomposition. Overridable from the build so the tuning
// sweep can rebuild this same source at another point in the space instead of forking it.
#ifndef OASIS_FE
#define OASIS_FE 256
#endif
#ifndef OASIS_H
#define OASIS_H 1024
#endif
#ifndef OASIS_K1_WARPS
#define OASIS_K1_WARPS 8
#endif
#ifndef OASIS_K1_ROWS
#define OASIS_K1_ROWS 1
#endif
#ifndef OASIS_K2_WARPS
#define OASIS_K2_WARPS 8
#endif
#ifndef OASIS_K2_ROWS
#define OASIS_K2_ROWS 4
#endif
#ifndef OASIS_K2_KSPLIT
#define OASIS_K2_KSPLIT 4
#endif
// 1: stage h in shared memory; 0: read it through the read-only path and rely on L1.
#ifndef OASIS_K2_STAGE_H
#define OASIS_K2_STAGE_H 1
#endif

// The batch sizes for which a kernel exists. The eligibility gate on the Python side is
// checked against instantiated_configs() below, which is generated from this same list.
//
// A batch of one is deliberately absent. cuBLAS does not take the tensor-core path for a
// single row -- flipping allow_tf32 moves an addmm at M=1 by 5e-7, against 7e-4 at M>=2 -- so
// the baseline computes it in fp32 there, and kernels that reproduce the tf32 path match it
// only 84-87 %. A batch of one is not among the captured shapes; it takes the eager path.
#define OASIS_FOR_EACH_BATCH(F) F(2) F(3) F(4) F(5) F(6) F(7) F(8)

// fp32(ln 10000), written out in full. The 9.21034f that reads naturally is one ulp low; that
// is invisible over the captured t < 128 but grows to about 0.06 rad of phase error near
// t = 2^20, which would corrupt the large-t robustness check.
#define OASIS_NEG_LOG_MAX_PERIOD (-9.2103404998779296875f)

__device__ __forceinline__ float to_tf32(float x) {
  float y;
  asm("cvt.rn.tf32.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

template <bool TF32>
__device__ __forceinline__ float round_operand(float x) {
  return TF32 ? to_tf32(x) : x;
}

template <bool TF32>
__device__ __forceinline__ void round_operand(float4& v) {
  if (TF32) {
    v.x = to_tf32(v.x);
    v.y = to_tf32(v.y);
    v.z = to_tf32(v.z);
    v.w = to_tf32(v.w);
  }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}

// Embedding, first matrix-vector product, bias and SiLU -> h[B][H].
//
// The embedding is built once per block into shared memory. Every lane of every warp needs the
// whole [B][FE] tile, so recomputing it per lane would be 2*B sin/cos pairs per lane (24 at
// B=6, about a microsecond); cooperatively it is at most ceil(B*FE/2 / threads) pairs per
// thread. Both halves of the tile share one frequency range, so one shared row of FE floats
// serves the cosine and sine operands.
template <int B, int FE, int H, bool TF32>
__global__ void oasis_k1(const int64_t* __restrict__ t, const float* __restrict__ w1,
                         const float* __restrict__ b1, float* __restrict__ h) {
  constexpr int HALF = FE / 2;
  constexpr int THREADS = OASIS_K1_WARPS * 32;
  constexpr int ROWS = OASIS_K1_ROWS;
  constexpr int VEC = FE / 4;
  static_assert(FE % 8 == 0, "the embedding tile is addressed as float4 in both halves");
  static_assert(H % (OASIS_K1_WARPS * ROWS) == 0, "output rows must divide across the grid");

  __shared__ __align__(16) float sh_e[B * FE];
  __shared__ float sh_t[B];

  const int tid = threadIdx.x;
  // int64 -> fp32 on device, matching t.float(); reading t on the host would cost a
  // synchronization worth more than both kernels.
  if (tid < B) sh_t[tid] = static_cast<float>(t[tid]);
  __syncthreads();

  for (int idx = tid; idx < B * HALF; idx += THREADS) {
    const int b = idx / HALF;
    const int j = idx - b * HALF;
    const float freq =
        expf(OASIS_NEG_LOG_MAX_PERIOD * static_cast<float>(j) / static_cast<float>(HALF));
    float sn, cs;
    sincosf(sh_t[b] * freq, &sn, &cs);
    // Rounded on store, so the product loop below reads operands that are already tf32.
    sh_e[b * FE + j] = round_operand<TF32>(cs);
    sh_e[b * FE + HALF + j] = round_operand<TF32>(sn);
  }
  __syncthreads();

  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int row0 = (blockIdx.x * OASIS_K1_WARPS + warp) * ROWS;

  float acc[ROWS][B];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int b = 0; b < B; ++b) acc[r][b] = 0.f;

  // Each warp load is 32 lanes x 16 B of one contiguous weight row: fully coalesced.
  for (int c = lane; c < VEC; c += 32) {
    float4 wv[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      wv[r] = reinterpret_cast<const float4*>(w1 + static_cast<size_t>(row0 + r) * FE)[c];
      round_operand<TF32>(wv[r]);
    }
#pragma unroll
    for (int b = 0; b < B; ++b) {
      const float4 ev = reinterpret_cast<const float4*>(sh_e + b * FE)[c];
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        acc[r][b] += wv[r].x * ev.x + wv[r].y * ev.y + wv[r].z * ev.z + wv[r].w * ev.w;
      }
    }
  }

#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    const int row = row0 + r;
#pragma unroll
    for (int b = 0; b < B; ++b) {
      const float total = warp_sum(acc[r][b]);
      if (lane == 0) {
        const float v = total + b1[row];  // bias in fp32, as the cutlass epilogue does
        const float act = v / (1.f + expf(-v));  // ATen's SiLU, not x * sigmoid(x)
        // h is tf32-rounded here because the second product would round it on load anyway.
        h[b * H + row] = round_operand<TF32>(act);
      }
    }
  }
}

// Second matrix-vector product and bias -> out[B][H].
//
// A warp owns ROWS output rows over one of KSPLIT slices of k. At ROWS=1 every one of the
// H warps reads the whole [B][H] activation tile, which is about 25 MB of shared traffic at
// B=6 and dominates the 4 MiB of weight traffic; widening ROWS amortizes each activation read
// over ROWS weight rows. The KSPLIT warps of a row group are kept in one block because shared
// memory cannot reduce across blocks.
template <int B, int H, bool TF32>
__global__ void oasis_k2(const float* __restrict__ h, const float* __restrict__ w2,
                         const float* __restrict__ b2, float* __restrict__ out) {
  constexpr int THREADS = OASIS_K2_WARPS * 32;
  constexpr int ROWS = OASIS_K2_ROWS;
  constexpr int KSPLIT = OASIS_K2_KSPLIT;
  constexpr int GROUPS = OASIS_K2_WARPS / KSPLIT;
  constexpr int VEC = H / 4;
  constexpr int SLICE = VEC / KSPLIT;
  static_assert(H % 4 == 0, "k is addressed as float4, so it must be a multiple of four");
  static_assert(OASIS_K2_WARPS % KSPLIT == 0, "a row group's k-slices must share a block");
  static_assert(VEC % KSPLIT == 0, "k must divide evenly across the k-split");
  static_assert(H % (GROUPS * ROWS) == 0, "output rows must divide across the grid");

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int group = warp / KSPLIT;
  const int kslice = warp % KSPLIT;

#if OASIS_K2_STAGE_H
  __shared__ __align__(16) float sh_h[B * H];
  for (int i = tid * 4; i < B * H; i += THREADS * 4) {
    *reinterpret_cast<float4*>(sh_h + i) = *reinterpret_cast<const float4*>(h + i);
  }
  __syncthreads();
  const float* hbase = sh_h;
#else
  const float* hbase = h;
#endif

  const int row0 = (blockIdx.x * GROUPS + group) * ROWS;

  float acc[ROWS][B];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int b = 0; b < B; ++b) acc[r][b] = 0.f;

  const int cbeg = kslice * SLICE;
  for (int c = cbeg + lane; c < cbeg + SLICE; c += 32) {
    // h was already rounded on store by the first kernel, so it needs no rounding here.
    float4 hv[B];
#pragma unroll
    for (int b = 0; b < B; ++b) {
      const float4* row = reinterpret_cast<const float4*>(hbase + b * H);
#if OASIS_K2_STAGE_H
      hv[b] = row[c];
#else
      hv[b] = __ldg(row + c);
#endif
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      float4 wv = reinterpret_cast<const float4*>(w2 + static_cast<size_t>(row0 + r) * H)[c];
      round_operand<TF32>(wv);
#pragma unroll
      for (int b = 0; b < B; ++b) {
        acc[r][b] += wv.x * hv[b].x + wv.y * hv[b].y + wv.z * hv[b].z + wv.w * hv[b].w;
      }
    }
  }

#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int b = 0; b < B; ++b) acc[r][b] = warp_sum(acc[r][b]);

  if constexpr (KSPLIT == 1) {
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int b = 0; b < B; ++b) out[b * H + row0 + r] = acc[r][b] + b2[row0 + r];
    }
  } else {
    __shared__ float sh_partial[OASIS_K2_WARPS * ROWS * B];
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int b = 0; b < B; ++b) sh_partial[(warp * ROWS + r) * B + b] = acc[r][b];
    }
    __syncthreads();
    constexpr int NOUT = GROUPS * ROWS * B;
    for (int i = tid; i < NOUT; i += THREADS) {
      const int b = i % B;
      const int r = (i / B) % ROWS;
      const int g = i / (B * ROWS);
      float sum = 0.f;
#pragma unroll
      for (int k = 0; k < KSPLIT; ++k) {
        sum += sh_partial[((g * KSPLIT + k) * ROWS + r) * B + b];
      }
      const int row = (blockIdx.x * GROUPS + g) * ROWS + r;
      out[b * H + row] = sum + b2[row];
    }
  }
}

template <int B>
static void launch_pair(const int64_t* t, const float* w1, const float* b1, const float* w2,
                        const float* b2, float* h, float* out, bool tf32,
                        cudaStream_t stream) {
  constexpr int FE = OASIS_FE;
  constexpr int H = OASIS_H;
  constexpr int K1_THREADS = OASIS_K1_WARPS * 32;
  constexpr int K1_BLOCKS = H / (OASIS_K1_WARPS * OASIS_K1_ROWS);
  constexpr int K2_THREADS = OASIS_K2_WARPS * 32;
  constexpr int K2_GROUPS = OASIS_K2_WARPS / OASIS_K2_KSPLIT;
  constexpr int K2_BLOCKS = H / (K2_GROUPS * OASIS_K2_ROWS);
  if (tf32) {
    oasis_k1<B, FE, H, true><<<K1_BLOCKS, K1_THREADS, 0, stream>>>(t, w1, b1, h);
    oasis_k2<B, H, true><<<K2_BLOCKS, K2_THREADS, 0, stream>>>(h, w2, b2, out);
  } else {
    oasis_k1<B, FE, H, false><<<K1_BLOCKS, K1_THREADS, 0, stream>>>(t, w1, b1, h);
    oasis_k2<B, H, false><<<K2_BLOCKS, K2_THREADS, 0, stream>>>(h, w2, b2, out);
  }
}

static void check_weight(const torch::Tensor& p, const char* name, torch::IntArrayRef shape,
                        const torch::Device& device, bool vectorized) {
  TORCH_CHECK(p.device() == device, name, " is on ", p.device(), ", expected ", device);
  TORCH_CHECK(p.scalar_type() == torch::kFloat, name, " must be float32, got ",
              p.scalar_type());
  TORCH_CHECK(p.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(p.sizes() == shape, name, " has shape ", p.sizes(), ", expected ", shape);
  // Contiguity does not imply alignment: a contiguous view whose storage offset is an odd
  // number of floats satisfies every check above and would fault on the float4 weight loads.
  if (vectorized) {
    TORCH_CHECK(reinterpret_cast<uintptr_t>(p.data_ptr<float>()) % 16 == 0, name,
                " must be 16-byte aligned for vectorized loads");
  }
}

torch::Tensor oasis_embed_mlp(const torch::Tensor& t, const torch::Tensor& w1,
                              const torch::Tensor& b1, const torch::Tensor& w2,
                              const torch::Tensor& b2) {
  constexpr int64_t FE = OASIS_FE;
  constexpr int64_t H = OASIS_H;
  TORCH_CHECK(t.is_cuda(), "t must be a CUDA tensor");
  TORCH_CHECK(t.dim() == 1 && t.is_contiguous(), "t must be contiguous and 1-D");
  TORCH_CHECK(t.scalar_type() == torch::kLong, "t must be int64, got ", t.scalar_type());
  const int64_t batch = t.numel();

  const at::cuda::CUDAGuard guard(t.device());
  const auto device = t.device();
  check_weight(w1, "mlp.0.weight", {H, FE}, device, /*vectorized=*/true);
  check_weight(b1, "mlp.0.bias", {H}, device, /*vectorized=*/false);
  check_weight(w2, "mlp.2.weight", {H, H}, device, /*vectorized=*/true);
  check_weight(b2, "mlp.2.bias", {H}, device, /*vectorized=*/false);

  auto opts = w1.options();
  auto h = torch::empty({batch, H}, opts);
  auto out = torch::empty({batch, H}, opts);
  auto stream = c10::cuda::getCurrentCUDAStream();
  const bool tf32 = at::globalContext().allowTF32CuBLAS();

  const int64_t* tp = t.data_ptr<int64_t>();
  const float* w1p = w1.data_ptr<float>();
  const float* b1p = b1.data_ptr<float>();
  const float* w2p = w2.data_ptr<float>();
  const float* b2p = b2.data_ptr<float>();
  float* hp = h.data_ptr<float>();
  float* op = out.data_ptr<float>();

#define OASIS_DISPATCH(B)                                                  \
  case B:                                                                  \
    launch_pair<B>(tp, w1p, b1p, w2p, b2p, hp, op, tf32, stream);          \
    break;
  switch (batch) {
    OASIS_FOR_EACH_BATCH(OASIS_DISPATCH)
    default:
      TORCH_CHECK(false, "batch ", batch, " has no instantiated kernel");
  }
#undef OASIS_DISPATCH
  return out;
}

// The instantiated (batch, frequency_embedding_size, hidden_size) triples, generated from the
// same list the dispatch switch is generated from, so the Python gate can assert it admits no
// more than this.
std::vector<std::vector<int64_t>> instantiated_configs() {
  std::vector<std::vector<int64_t>> out;
#define OASIS_LIST(B) out.push_back({B, OASIS_FE, OASIS_H});
  OASIS_FOR_EACH_BATCH(OASIS_LIST)
#undef OASIS_LIST
  return out;
}
"""

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

#: The work decomposition the kernels are compiled for. ``scratch/sweep_configs.py`` rebuilds
#: the same source across the neighbourhood of this point and ranks it on the harness-shaped
#: timing rig; this is the winner of that 32-point sweep.
#:
#: Two results here are worth keeping in view. Reading the activation tile through the
#: read-only path beats staging it in shared memory at every other point in the space, because
#: Nsight Compute shows both kernels stalled on global-memory dependencies (74 % of the second
#: kernel's stall cycles) rather than on the shared pipe -- so a shared tile spends a barrier
#: and 25 KiB of occupancy budget duplicating what L1 already provides. And four warps per
#: block beats eight because it doubles the grid to 256 blocks: at eight warps the grid was 128
#: blocks against 148 SMs, under a third of one wave, so a fifth of the machine sat idle.
DEFAULT_CONFIG = {
    "FE": 256,
    "H": 1024,
    "K1_WARPS": 4,
    "K1_ROWS": 1,
    "K2_WARPS": 4,
    "K2_ROWS": 4,
    "K2_KSPLIT": 4,
    "K2_STAGE_H": 0,
}

_NVCC_FLAGS = ["-O3", "-lineinfo"]  # no fast math: it would swap in __expf / __sincosf


def _arch_list() -> str:
    """The single architecture to compile for, instead of the six in the ambient environment.

    ``load_inline`` only queries the device when ``TORCH_CUDA_ARCH_LIST`` is unset, so setting
    it is also what keeps a cold build to one architecture and about half a minute.
    """
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        # sm_90 and later gate some instructions behind the architecture-specific suffix.
        return f"{major}.{minor}a" if major >= 9 else f"{major}.{minor}"
    return "10.0a"


def build_extension(config: dict[str, int], *, verbose: bool = False):
    """Compile the fused kernels for one work decomposition.

    Also the entry point the tuning sweep uses, so a swept configuration is the same source as
    the delivered one rather than a copy of it. Each configuration gets its own extension name
    so the build cache neither collides across configurations nor recompiles within one.
    """
    from torch.utils.cpp_extension import load_inline

    flags = _NVCC_FLAGS + [f"-DOASIS_{k}={v}" for k, v in sorted(config.items())]
    tag = hashlib.sha1(
        (";".join(flags) + _CUDA_SOURCE + _CPP_SOURCE).encode()).hexdigest()[:12]
    name = f"oasis_tse_{tag}"

    workspace = Path(__file__).resolve().parents[2]
    build_dir = workspace / ".torch_extensions" / name
    build_dir.mkdir(parents=True, exist_ok=True)

    previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = _arch_list()
    try:
        return load_inline(
            name=name,
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["oasis_embed_mlp", "instantiated_configs"],
            extra_cuda_cflags=flags,
            build_directory=str(build_dir),
            verbose=verbose,
        )
    finally:
        if previous is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous


def _load_default_extension():
    """Compile at import, and say on stderr whether it worked.

    Import time rather than first call, so nothing compiles while the harness is timing (which
    would also trip its no-new-threads check). A failure falls back to the eager path so a
    broken build is never a hard error -- but a silent fallback would score as a
    baseline-speed pass, so the outcome goes to stderr, where the harness's per-operator worker
    log keeps it. Set ``OASIS_TSE_STRICT=1`` to make a failure raise instead, which is how the
    build is exercised during development.
    """
    try:
        ext = build_extension(DEFAULT_CONFIG)
    except Exception as exc:  # noqa: BLE001 - any build failure must degrade, not crash
        status = f"extension UNAVAILABLE, using eager fallback: {type(exc).__name__}: {exc}"
        print(f"[oasis_timestep_embedder] {status}", file=sys.stderr, flush=True)
        if os.environ.get("OASIS_TSE_STRICT"):
            raise
        return None, status, exc
    cfg = ",".join(f"{k}={v}" for k, v in sorted(DEFAULT_CONFIG.items()))
    status = f"fused extension loaded (arch={_arch_list()}, {cfg})"
    print(f"[oasis_timestep_embedder] {status}", file=sys.stderr, flush=True)
    return ext, status, None


_EXT, _STATUS, _BUILD_ERROR = _load_default_extension()

#: Every configuration the extension actually holds a kernel for. The gate is checked against
#: this rather than against a second hand-written list that could drift from it.
INSTANTIATED = frozenset(
    tuple(int(v) for v in c) for c in (_EXT.instantiated_configs() if _EXT else ()))
_INSTANTIATED_SHAPES = frozenset((fe, h) for _b, fe, h in INSTANTIATED)
#: Tested by membership rather than against an interval, so a gap in the instantiated batches
#: cannot let a batch through to a dispatch that has no kernel for it.
_ADMITTED_BATCHES = frozenset(b for b, _fe, _h in INSTANTIATED)


def status() -> str:
    """One line on whether the fused path is available, as printed at import."""
    return _STATUS


def build_error() -> BaseException | None:
    """The retained build exception, for diagnosing a fallback that should not have happened."""
    return _BUILD_ERROR


def admitted_configs() -> frozenset[tuple[int, int, int]]:
    """Every ``(batch, frequency_embedding_size, hidden_size)`` the gate can let through.

    Must be a subset of :data:`INSTANTIATED`: admitting a triple with no kernel behind it
    would mean launching one whose template parameters disagree with the runtime shapes. The
    gate tests the batch and the shape independently, so this is the cross-product of the two
    admitted sets -- which is how a ragged instantiation would show up as a subset violation.
    """
    return frozenset(
        (b, fe, h) for b in _ADMITTED_BATCHES for fe, h in _INSTANTIATED_SHAPES)


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class OasisTimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.ModuleList(
            [
                Linear(frequency_embedding_size, hidden_size, bias=True),
                SiLU(),
                Linear(hidden_size, hidden_size, bias=True),
            ]
        )
        self.frequency_embedding_size = frequency_embedding_size
        self._fused_shape_ok = (frequency_embedding_size, hidden_size) in _INSTANTIATED_SHAPES

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half,
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def _eager_forward(self, t: torch.Tensor) -> torch.Tensor:
        x = self.timestep_embedding(t, self.frequency_embedding_size)
        for layer in self.mlp:
            x = layer(x)
        return x

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Property reads and one extension call, no eager op: on this operator's timeline a
        # single eager op costs more than either kernel. The gate declines with gradients
        # enabled so the module stays differentiable outside the graded path, which always
        # runs under no_grad.
        if self._fused_shape_ok and _EXT is not None and not torch.is_grad_enabled():
            # Read the parameters here rather than from a cache built in __init__. Caching the
            # Parameter objects survives everything the harness does to a module, but not
            # load_state_dict(assign=True), which rebinds the registered Parameters and would
            # leave a cached tuple pointing at the previous ones. Six dict lookups are free
            # against a ~2 us kernel; a silently stale weight is not.
            first, second = self.mlp[0], self.mlp[2]
            weight1, bias1 = first.weight, first.bias
            weight2, bias2 = second.weight, second.bias
            if (
                type(t) is torch.Tensor
                and t.is_cuda
                and t.dtype is torch.int64
                and t.dim() == 1
                and t.is_contiguous()
                and t.numel() in _ADMITTED_BATCHES
                and t.device == weight1.device
                and weight1.dtype is torch.float32
                and bias1.dtype is torch.float32
                and weight2.dtype is torch.float32
                and bias2.dtype is torch.float32
                and weight1.is_contiguous()
                and bias1.is_contiguous()
                and weight2.is_contiguous()
                and bias2.is_contiguous()
                # Contiguity is not alignment: a contiguous weight view at an odd float
                # offset would fault on the kernels' float4 loads.
                and weight1.data_ptr() % 16 == 0
                and weight2.data_ptr() % 16 == 0
            ):
                return _EXT.oasis_embed_mlp(t, weight1, bias1, weight2, bias2)
        return self._eager_forward(t)
