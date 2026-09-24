"""Gated RMSNorm with element-wise multiplicative gate (L1).

Implements ``RMSNorm(x) * activation(z)`` in a single fused Triton kernel.
Hosts two vendored vLLM variants (kernel bodies unchanged):

- ``rmsnorm_fn`` / ``RMSNormGated`` (from FLA ``layernorm_guard``), used by
  Qwen3-Next's GDN output gating (``norm_before_gate=True``, ``swish``).
- ``rms_norm_gated`` / ``FusedRMSNormGated`` (from FLA ``layernorm_gated``),
  used by Kimi-Delta attention's output ``o_norm``.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable

import torch
import torch.nn as nn
import triton
import triton.language as tl


def cdiv(a: int, b: int) -> int:
    return -(a // -b)


def next_power_of_2(n: int) -> int:
    return 1 if n < 1 else 1 << (n - 1).bit_length()


def num_compute_units(device_id: int = 0) -> int:
    return torch.cuda.get_device_properties(device_id).multi_processor_count


def input_guard(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Ensure input tensors are contiguous and set the device from them."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_args = (
            i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args
        )
        contiguous_kwargs = {
            k: (v if not isinstance(v, torch.Tensor) else v.contiguous())
            for k, v in kwargs.items()
        }

        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        if tensor is not None:
            ctx = torch.accelerator.device_index(tensor.device.index)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["B"] is not None,
        "HAS_Z": lambda args: args["Z"] is not None,
    }
)
@triton.jit
def layer_norm_fwd_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Z,  # pointer to the other branch
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_z_row,
    M,  # number of rows in X
    N: tl.constexpr,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    # Map the program id to the starting row of X and Y it should compute.
    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    group = tl.program_id(1)

    # Create 2D tile: [ROWS_PER_BLOCK, BLOCK_N]
    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, BLOCK_N)

    # Compute offsets for 2D tile
    row_offsets = rows[:, None] * stride_x_row
    col_offsets = cols[None, :] + group * N

    # Base pointers
    X_base = X + row_offsets + col_offsets
    Y_base = Y + rows[:, None] * stride_y_row + col_offsets

    # Create mask for valid rows and columns
    row_mask = rows[:, None] < M
    col_mask = cols[None, :] < N
    mask = row_mask & col_mask

    # Load input data with 2D tile
    x = tl.load(X_base, mask=mask, other=0.0).to(tl.float32)

    if HAS_Z and not NORM_BEFORE_GATE:
        Z_base = Z + rows[:, None] * stride_z_row + col_offsets
        z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            x *= z * tl.sigmoid(z)
        elif ACTIVATION == "sigmoid":
            x *= tl.sigmoid(z)

    # Compute mean and variance per row (reduce along axis 1)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=1) / N  # Shape: [ROWS_PER_BLOCK]
        # Store mean for each row
        mean_offsets = group * M + rows
        mean_mask = rows < M
        tl.store(Mean + mean_offsets, mean, mask=mean_mask)
        # Broadcast mean back to 2D for subtraction
        xbar = tl.where(mask, x - mean[:, None], 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N  # Shape: [ROWS_PER_BLOCK]
    else:
        xbar = tl.where(mask, x, 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N  # Shape: [ROWS_PER_BLOCK]
        mean = 0.0  # Placeholder for RMS norm

    rstd = tl.rsqrt(var + eps)  # Shape: [ROWS_PER_BLOCK]

    # Store rstd for each row
    rstd_offsets = group * M + rows
    rstd_mask = rows < M
    tl.store(Rstd + rstd_offsets, rstd, mask=rstd_mask)

    # Load weights and biases (broadcast across rows)
    w_offsets = cols + group * N
    w_mask = cols < N
    w = tl.load(W + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

    if HAS_BIAS:
        b = tl.load(B + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

    # Normalize and apply linear transformation
    if not IS_RMS_NORM:
        x_hat = (x - mean[:, None]) * rstd[:, None]
    else:
        x_hat = x * rstd[:, None]

    y = x_hat * w[None, :] + b[None, :] if HAS_BIAS else x_hat * w[None, :]

    if HAS_Z and NORM_BEFORE_GATE:
        Z_base = Z + rows[:, None] * stride_z_row + col_offsets
        z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            y *= z * tl.sigmoid(z)
        elif ACTIVATION == "sigmoid":
            y *= tl.sigmoid(z)

    # Write output
    tl.store(Y_base, y, mask=mask)


def calc_rows_per_block(M: int, device: torch.device) -> int:
    sm_count = num_compute_units(device.index)
    rows_per_block = next_power_of_2(cdiv(M, 2 * sm_count))
    rows_per_block = min(rows_per_block, 4)
    return rows_per_block


def layer_norm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    z: torch.Tensor = None,
    out: torch.Tensor = None,
    group_size: int = None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = False,
    activation: str = "swish",
):
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    # allocate output
    if out is not None:
        assert out.shape == x.shape
    else:
        out = torch.empty_like(x)
    assert out.stride(-1) == 1
    mean = (
        torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
        if not is_rms_norm
        else None
    )
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    # Calculate rows per block based on SM count
    rows_per_block = calc_rows_per_block(M, x.device)
    # Update grid to use rows_per_block
    grid = (cdiv(M, rows_per_block), ngroups)
    layer_norm_fwd_kernel[grid](
        x,
        out,
        weight,
        bias,
        z,
        mean,
        rstd,
        x.stride(0),
        out.stride(0),
        z.stride(0) if z is not None else 0,
        M,
        group_size,
        eps,
        BLOCK_N=BLOCK_N,
        ROWS_PER_BLOCK=rows_per_block,
        HAS_BIAS=bias is not None,
        HAS_Z=z is not None,
        NORM_BEFORE_GATE=norm_before_gate,
        IS_RMS_NORM=is_rms_norm,
        num_warps=num_warps,
        ACTIVATION=activation,
    )
    return out, mean, rstd


def _layer_norm_fn_impl(
    x,
    weight,
    bias,
    z=None,
    eps=1e-6,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
    activation: str = "swish",
):
    """Triton layer/RMS norm with optional gating.

    If z is not None, computes norm(x) * silu(z) when norm_before_gate,
    else norm(x * silu(z)).

    This calls the triton kernel directly. The original code wrapped this
    in a torch.autograd.Function (LayerNormFn) to save tensors for a
    backward pass, but vLLM is inference-only so there is no backward pass.
    The autograd wrapper also prevented torch.compile/dynamo from tracing
    through the function due to its @staticmethod forward.
    """
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    if x.stride(-1) != 1:
        x = x.contiguous()
    if z is not None:
        assert z.shape == x_shape_og
        z = z.reshape(-1, z.shape[-1])
        if z.stride(-1) != 1:
            z = z.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    y, _, _ = layer_norm_fwd(
        x,
        weight,
        bias,
        eps,
        z=z,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
        activation=activation,
    )
    return y.reshape(x_shape_og)


@input_guard
def rmsnorm_fn(
    x,
    weight,
    bias,
    z=None,
    eps=1e-6,
    group_size=None,
    norm_before_gate=True,
    activation: str = "swish",
):
    return _layer_norm_fn_impl(
        x, weight, bias, z, eps, group_size, norm_before_gate, True, activation
    )


# ---------------------------------------------------------------------------
# Fused CUDA kernel for the gated RMSNorm above (see the comment at the top of
# _RMSG_CUDA_SRC for the kernel design).
#
# The Triton path stays as the fallback: it covers the shapes, dtypes and
# activations the CUDA kernel does not specialize for, and it is what runs if
# the extension cannot be built.
# ---------------------------------------------------------------------------

_RMSG_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// ---------------------------------------------------------------------------
// Fused gated RMSNorm
//     norm_before_gate:  y = (x * rstd(x)) * w * act(z)
//     otherwise:         y = (x * act(z)) * rstd(x * act(z)) * w
//
// A row of N columns is owned by a group of TPR = N/8 adjacent lanes of one
// warp: each thread holds exactly 8 columns, i.e. one 16-byte vector access per
// tensor, so a warp always issues fully coalesced 512-byte transactions and a
// whole row lives in registers.  The row reduction is a butterfly shuffle
// inside the lane group -- no shared memory, no block barrier, so rows never
// wait on each other and a block needs no tail handling beyond a row bound
// check.
//
// The grid is persistent and sized to exactly one resident wave (occupancy x
// SM count), with a grid-stride loop over the remaining rows.  On a B200 that
// is the clear optimum for the large-M case: fewer blocks starve the memory
// system, more blocks only add scheduling waves.  A small M launches fewer
// blocks than the cap and the stride loop degenerates to a single pass.
// ---------------------------------------------------------------------------

#define VEC 8

typedef __nv_bfloat16 bf16;

template <typename T> struct Cvt;

template <> struct Cvt<bf16> {
  __device__ __forceinline__ static void ld(const bf16* p, float* o) {
    uint4 v = *reinterpret_cast<const uint4*>(p);
    const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float2 f = __bfloat1622float2(b[i]);
      o[2 * i] = f.x;
      o[2 * i + 1] = f.y;
    }
  }
  __device__ __forceinline__ static void st(bf16* p, const float* o) {
    uint4 v;
    __nv_bfloat162* b = reinterpret_cast<__nv_bfloat162*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) b[i] = __floats2bfloat162_rn(o[2 * i], o[2 * i + 1]);
    *reinterpret_cast<uint4*>(p) = v;
  }
};

template <> struct Cvt<__half> {
  __device__ __forceinline__ static void ld(const __half* p, float* o) {
    uint4 v = *reinterpret_cast<const uint4*>(p);
    const __half2* b = reinterpret_cast<const __half2*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float2 f = __half22float2(b[i]);
      o[2 * i] = f.x;
      o[2 * i + 1] = f.y;
    }
  }
  __device__ __forceinline__ static void st(__half* p, const float* o) {
    uint4 v;
    __half2* b = reinterpret_cast<__half2*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) b[i] = __floats2half2_rn(o[2 * i], o[2 * i + 1]);
    *reinterpret_cast<uint4*>(p) = v;
  }
};

// swish through the hardware tanh unit: g*sigmoid(g) = h + h*tanh(h), h = g/2.
// One MUFU op instead of exp + reciprocal, which is worth real time in a kernel
// that evaluates the gate once per element; the error stays far inside one bf16
// ulp of the result.  sigmoid keeps the exp form (it is not on the hot path).
__device__ __forceinline__ float swish_act(float g) {
  const float h = 0.5f * g;
  float t;
  asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(h));
  return fmaf(h, t, h);
}

// ACT: 0 swish/silu, 1 sigmoid
template <int ACT>
__device__ __forceinline__ float gate(float g) {
  return (ACT == 0) ? swish_act(g) : __fdividef(1.0f, 1.0f + __expf(-g));
}

// MODE: 0 swish+norm_first, 1 swish+gate_first, 2 sigmoid+norm_first,
//       3 sigmoid+gate_first.
template <int TPR, int BLK, int MODE, typename ST>
__global__ __launch_bounds__(BLK) void rmsg_kernel(
    const ST* __restrict__ X, const ST* __restrict__ Z, const void* __restrict__ Wv,
    ST* __restrict__ Y, int M, float eps, float inv_n, int w_fp32) {
  constexpr int N = TPR * VEC;
  constexpr int GPB = BLK / TPR;          // rows a block covers per step
  constexpr int ACT = (MODE < 2) ? 0 : 1;
  constexpr bool NORM_FIRST = (MODE == 0 || MODE == 2);

  const int col = (threadIdx.x & (TPR - 1)) * VEC;

  float wf[VEC];
  if (w_fp32) {
    const float* wp = static_cast<const float*>(Wv) + col;
#pragma unroll
    for (int i = 0; i < VEC; ++i) wf[i] = wp[i];
  } else {
    Cvt<ST>::ld(static_cast<const ST*>(Wv) + col, wf);
  }

  const int step = gridDim.x * GPB;
  for (int row = blockIdx.x * GPB + (int)(threadIdx.x / TPR); row < M; row += step) {
    const long long off = (long long)row * N + col;
    float xf[VEC], zf[VEC];
    Cvt<ST>::ld(X + off, xf);
    Cvt<ST>::ld(Z + off, zf);

    // Everything that does not depend on rstd is folded in before the row
    // reduction, so the gate (and the weight, when it is applied after the
    // norm) is off the load -> reduce -> store critical path.  zf is reused as
    // the accumulator to keep the register count down.
    if (NORM_FIRST) {
#pragma unroll
      for (int i = 0; i < VEC; ++i) zf[i] = gate<ACT>(zf[i]) * wf[i];
    } else {
#pragma unroll
      for (int i = 0; i < VEC; ++i) xf[i] *= gate<ACT>(zf[i]);
    }
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < VEC; ++i) ss = fmaf(xf[i], xf[i], ss);
#pragma unroll
    for (int s = TPR / 2; s > 0; s >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, s);
    const float rstd = rsqrtf(ss * inv_n + eps);

    float yf[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i) yf[i] = xf[i] * rstd * (NORM_FIRST ? zf[i] : wf[i]);
    Cvt<ST>::st(Y + off, yf);
  }
}

#define RMSG_BLK 128

// Blocks that fit on the device at once, per kernel instantiation and device.
template <int TPR, int MODE, typename ST>
static int resident_blocks(int dev) {
  static int cache[16] = {0};
  const int slot = (dev >= 0 && dev < 16) ? dev : 0;
  int cap = cache[slot];
  if (cap == 0) {
    int per_sm = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, reinterpret_cast<const void*>(&rmsg_kernel<TPR, RMSG_BLK, MODE, ST>),
            RMSG_BLK, 0) != cudaSuccess || per_sm < 1) {
      per_sm = 1;
    }
    cap = per_sm * at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    cache[slot] = cap;
  }
  return cap;
}

#define LAUNCH(TPR, MODE, ST)                                                     \
  do {                                                                            \
    constexpr int GPB = RMSG_BLK / (TPR);                                          \
    int grid = (int)((M + GPB - 1) / GPB);                                          \
    const int cap = resident_blocks<TPR, MODE, ST>(dev);                           \
    if (grid > cap) grid = cap;                                                    \
    rmsg_kernel<TPR, RMSG_BLK, MODE, ST><<<grid, RMSG_BLK, 0, stream>>>(           \
        static_cast<const ST*>(xp), static_cast<const ST*>(zp), wp,                \
        static_cast<ST*>(op), M, eps, inv_n, w_fp32);                              \
  } while (0)

#define BY_MODE(TPR, ST)                          \
  switch (mode) {                                 \
    case 0: LAUNCH(TPR, 0, ST); break;            \
    case 1: LAUNCH(TPR, 1, ST); break;            \
    case 2: LAUNCH(TPR, 2, ST); break;            \
    default: LAUNCH(TPR, 3, ST); break;           \
  }

#define BY_DTYPE(TPR) \
  if (is_bf16) { BY_MODE(TPR, bf16); } else { BY_MODE(TPR, __half); }

at::Tensor rms_gated(const at::Tensor& x, const at::Tensor& z, const at::Tensor& w,
                     double eps_in, int64_t mode) {
  const auto st = x.scalar_type();
  const int N = (int)x.size(-1);
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(st == at::kBFloat16 || st == at::kHalf, "unsupported dtype");
  TORCH_CHECK(N == 64 || N == 128 || N == 256, "unsupported hidden size");
  TORCH_CHECK(w.numel() == N && w.is_contiguous(), "bad weight");
  TORCH_CHECK(w.scalar_type() == at::kFloat || w.scalar_type() == st, "bad weight dtype");
  TORCH_CHECK(z.scalar_type() == st && z.sizes() == x.sizes(), "bad gate tensor");
  TORCH_CHECK(x.is_contiguous() && z.is_contiguous(), "inputs must be contiguous");

  const at::cuda::OptionalCUDAGuard guard(at::device_of(x));
  at::Tensor out = at::empty_like(x);
  const long long rows = x.numel() / N;
  TORCH_CHECK(rows <= 2147483647LL, "too many rows");
  const int M = (int)rows;
  if (M == 0) return out;

  const void* xp = x.const_data_ptr();
  const void* zp = z.const_data_ptr();
  const void* wp = w.const_data_ptr();
  void* op = out.data_ptr();
  TORCH_CHECK(((reinterpret_cast<uintptr_t>(xp) | reinterpret_cast<uintptr_t>(zp) |
                reinterpret_cast<uintptr_t>(wp) | reinterpret_cast<uintptr_t>(op)) & 15u) == 0,
              "inputs must be 16B aligned");

  const int dev = (int)x.device().index();
  const int w_fp32 = (w.scalar_type() == at::kFloat) ? 1 : 0;
  const bool is_bf16 = (st == at::kBFloat16);
  const float eps = (float)eps_in;
  const float inv_n = 1.0f / (float)N;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (N == 128) {
    BY_DTYPE(16);
  } else if (N == 256) {
    BY_DTYPE(32);
  } else {
    BY_DTYPE(8);
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rms_gated", &rms_gated, "fused gated RMSNorm");
}
"""

# (activation, norm_before_gate) -> kernel MODE
_RMSG_MODES = {
    ("swish", True): 0, ("silu", True): 0,
    ("swish", False): 1, ("silu", False): 1,
    ("sigmoid", True): 2, ("sigmoid", False): 3,
}
_RMSG_DTYPES = (torch.bfloat16, torch.float16)
_RMSG_SIZES = (64, 128, 256)


def _load_rmsg_ext():
    """JIT-build the fused kernel; None if it cannot be built."""
    try:
        import os

        from torch.utils.cpp_extension import load_inline

        if not torch.cuda.is_available():
            return None
        if "TORCH_CUDA_ARCH_LIST" not in os.environ:
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = (
                f"{major}.{minor}a" if major in (9, 10, 12) else f"{major}.{minor}"
            )
        return load_inline(
            name="fk_rms_norm_gated_ext",
            cpp_sources="",
            cuda_sources=_RMSG_CUDA_SRC,
            functions=None,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    except Exception:  # noqa: BLE001 - any build failure just means Triton
        return None


_RMSG_EXT = _load_rmsg_ext()


__targets__ = ["RMSNormGated", "FusedRMSNormGated"]


class RMSNormGated(nn.Module):
    """Fused gated RMSNorm: ``out = activation(z) * RMSNorm(x, weight)``."""

    def __init__(self, hidden_size: int, eps: float = 1e-6,
                 norm_before_gate: bool = True, activation: str = "swish"):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.norm_before_gate = norm_before_gate
        self.activation = activation
        self.weight = nn.Parameter(torch.ones(hidden_size))
        mode = _RMSG_MODES.get((activation, bool(norm_before_gate)))
        self._rmsg_mode = -1 if mode is None else mode
        self._rmsg_fast = (_RMSG_EXT is not None and mode is not None
                           and hidden_size in _RMSG_SIZES)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if (self._rmsg_fast and x.dtype in _RMSG_DTYPES and z.dtype is x.dtype
                and x.shape[-1] == self.hidden_size and x.shape == z.shape
                and x.is_contiguous() and z.is_contiguous()):
            try:
                return _RMSG_EXT.rms_gated(x, z, self.weight, self.eps, self._rmsg_mode)
            except Exception:  # noqa: BLE001 - layout the kernel rejects
                pass
        return rmsnorm_fn(
            x, self.weight, bias=None,
            z=z, eps=self.eps,
            norm_before_gate=self.norm_before_gate,
            activation=self.activation,
        )


# --- FLA layernorm_gated variant (used by Kimi-Delta o_norm) ---

class CustomOp(nn.Module):
    def __init__(self, *, enforce_enable: bool = False, compile_native: bool = False):
        super().__init__()
        self._forward_method = (
            self.forward_hip if torch.version.hip is not None else self.forward_cuda
        )

    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs):
        raise NotImplementedError

    def forward_hip(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)

    @classmethod
    def register(cls, name: str, dynamic_arg_dims=None):
        def decorator(op_cls):
            op_cls.name = name
            op_cls._dynamic_arg_dims = dynamic_arg_dims
            return op_cls

        return decorator


@triton.heuristics(
    {
        "STORE_RESIDUAL_OUT": lambda args: args["residual_out"] is not None,
        "HAS_RESIDUAL": lambda args: args["residual"] is not None,
        "HAS_WEIGHT": lambda args: args["w"] is not None,
        "HAS_BIAS": lambda args: args["b"] is not None,
    }
)
@triton.jit
def layer_norm_gated_fwd_kernel(
    x,  # pointer to the input
    g,  # pointer to the gate
    y,  # pointer to the output
    w,  # pointer to the weights
    b,  # pointer to the biases
    residual,  # pointer to the residual
    residual_out,  # pointer to the residual
    mean,  # pointer to the mean
    rstd,  # pointer to the 1/std
    eps,  # epsilon to avoid division by zero
    T,  # number of rows in x
    D: tl.constexpr,  # number of columns in x
    BT: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_t = tl.program_id(0)

    o_d = tl.arange(0, BD)
    m_d = o_d < D

    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    if HAS_RESIDUAL:
        p_res = tl.make_block_ptr(
            residual, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0)
        )
        b_x += tl.load(p_res, boundary_check=(0, 1)).to(tl.float32)
    if STORE_RESIDUAL_OUT:
        p_res_out = tl.make_block_ptr(
            residual_out, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0)
        )
        tl.store(p_res_out, b_x.to(p_res_out.dtype.element_ty), boundary_check=(0, 1))
    if not IS_RMS_NORM:
        b_mean = tl.sum(b_x, axis=1) / D
        p_mean = tl.make_block_ptr(mean, (T,), (1,), (i_t * BT,), (BT,), (0,))
        tl.store(p_mean, b_mean.to(p_mean.dtype.element_ty), boundary_check=(0,))
        b_xbar = tl.where(m_d[None, :], b_x - b_mean[:, None], 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
    else:
        b_xbar = tl.where(m_d[None, :], b_x, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)

    p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (i_t * BT,), (BT,), (0,))
    tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), boundary_check=(0,))

    if HAS_WEIGHT:
        b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + o_d, mask=m_d).to(tl.float32)
    b_x_hat = (
        (b_x - b_mean[:, None]) * b_rstd[:, None]
        if not IS_RMS_NORM
        else b_x * b_rstd[:, None]
    )
    b_y = b_x_hat * b_w[None, :] if HAS_WEIGHT else b_x_hat
    if HAS_BIAS:
        b_y = b_y + b_b[None, :]

    # swish/sigmoid output gate
    p_g = tl.make_block_ptr(g, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    if ACTIVATION == "swish" or ACTIVATION == "silu":
        b_y = b_y * b_g * tl.sigmoid(b_g)
    elif ACTIVATION == "sigmoid":
        b_y = b_y * tl.sigmoid(b_g)

    # Write output
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "STORE_RESIDUAL_OUT": lambda args: args["residual_out"] is not None,
        "HAS_RESIDUAL": lambda args: args["residual"] is not None,
        "HAS_WEIGHT": lambda args: args["w"] is not None,
        "HAS_BIAS": lambda args: args["b"] is not None,
    }
)
@triton.jit
def layer_norm_gated_fwd_kernel1(
    x,  # pointer to the input
    g,  # pointer to the gate
    y,  # pointer to the output
    w,  # pointer to the weights
    b,  # pointer to the biases
    residual,  # pointer to the residual
    residual_out,  # pointer to the residual
    mean,  # pointer to the mean
    rstd,  # pointer to the 1/std
    eps,  # epsilon to avoid division by zero
    D: tl.constexpr,  # number of columns in x
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    g += i_t * D
    if HAS_RESIDUAL:
        residual += i_t * D
    if STORE_RESIDUAL_OUT:
        residual_out += i_t * D

    o_d = tl.arange(0, BD)
    m_d = o_d < D
    b_x = tl.load(x + o_d, mask=m_d, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        b_x += tl.load(residual + o_d, mask=m_d, other=0.0).to(tl.float32)
    if STORE_RESIDUAL_OUT:
        tl.store(residual_out + o_d, b_x, mask=m_d)
    if not IS_RMS_NORM:
        b_mean = tl.sum(b_x, axis=0) / D
        tl.store(mean + i_t, b_mean)
        b_xbar = tl.where(m_d, b_x - b_mean, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    else:
        b_xbar = tl.where(m_d, b_x, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)
    tl.store(rstd + i_t, b_rstd)

    if HAS_WEIGHT:
        b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + o_d, mask=m_d).to(tl.float32)
    b_x_hat = (b_x - b_mean) * b_rstd if not IS_RMS_NORM else b_x * b_rstd
    b_y = b_x_hat * b_w if HAS_WEIGHT else b_x_hat
    if HAS_BIAS:
        b_y = b_y + b_b

    # swish/sigmoid output gate
    b_g = tl.load(g + o_d, mask=m_d, other=0.0).to(tl.float32)
    if ACTIVATION == "swish" or ACTIVATION == "silu":
        b_y = b_y * b_g * tl.sigmoid(b_g)
    elif ACTIVATION == "sigmoid":
        b_y = b_y * tl.sigmoid(b_g)

    # Write output
    tl.store(y + o_d, b_y, mask=m_d)


def layer_norm_gated_fwd(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = "swish",
    eps: float = 1e-5,
    residual: torch.Tensor = None,
    out_dtype: torch.dtype = None,
    residual_dtype: torch.dtype = None,
    is_rms_norm: bool = False,
):
    if residual is not None:
        residual_dtype = residual.dtype
    T, D = x.shape
    if residual is not None:
        assert residual.shape == (T, D)
    if weight is not None:
        assert weight.shape == (D,)
    if bias is not None:
        assert bias.shape == (D,)
    # allocate output
    y = x if out_dtype is None else torch.empty_like(x, dtype=out_dtype)
    if residual is not None or (
        residual_dtype is not None and residual_dtype != x.dtype
    ):
        residual_out = torch.empty(T, D, device=x.device, dtype=residual_dtype)
    else:
        residual_out = None
    mean = (
        torch.empty((T,), dtype=torch.float, device=x.device)
        if not is_rms_norm
        else None
    )
    rstd = torch.empty((T,), dtype=torch.float, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps

    if D <= 512:
        BT = 32
        layer_norm_gated_fwd_kernel[(cdiv(T, BT),)](
            x=x,
            g=g,
            y=y,
            w=weight,
            b=bias,
            residual=residual,
            residual_out=residual_out,
            mean=mean,
            rstd=rstd,
            eps=eps,
            T=T,
            D=D,
            BD=BD,
            BT=BT,
            ACTIVATION=activation,
            IS_RMS_NORM=is_rms_norm,
            num_warps=4,
        )
    else:
        layer_norm_gated_fwd_kernel1[(T,)](
            x=x,
            g=g,
            y=y,
            w=weight,
            b=bias,
            residual=residual,
            residual_out=residual_out,
            mean=mean,
            rstd=rstd,
            eps=eps,
            D=D,
            BD=BD,
            ACTIVATION=activation,
            IS_RMS_NORM=is_rms_norm,
            num_warps=4,
        )
    # residual_out is None if residual is None and residual_dtype == input_dtype
    return y, mean, rstd, residual_out if residual_out is not None else x


def rms_norm_gated(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = "swish",
    residual: torch.Tensor | None = None,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    eps: float = 1e-6,
):
    x_shape_og = x.shape
    # reshape input data into 2D tensor
    x = x.contiguous().reshape(-1, x.shape[-1])
    g = g.contiguous().reshape(-1, g.shape[-1])
    if residual is not None:
        assert residual.shape == x_shape_og
        residual = residual.contiguous().reshape(-1, residual.shape[-1])
    residual_dtype = (
        residual.dtype
        if residual is not None
        else (torch.float if residual_in_fp32 else None)
    )
    y, _, _, residual_out = layer_norm_gated_fwd(
        x=x,
        g=g,
        weight=weight,
        bias=bias,
        activation=activation,
        eps=eps,
        residual=residual,
        residual_dtype=residual_dtype,
        is_rms_norm=True,
    )
    y = y.reshape(x_shape_og)
    return y if not prenorm else (y, residual_out.reshape(x_shape_og))


@CustomOp.register("fused_rms_norm_gated")
class FusedRMSNormGated(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        activation: str = "swish",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps
        self.activation = activation

        if self.activation not in ["swish", "silu", "sigmoid"]:
            raise ValueError(f"Unsupported activation: {self.activation}")

        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def forward_native(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        residual: torch.Tensor | None = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ) -> torch.Tensor:
        """Decomposed PyTorch ops for torch.compile/inductor fusion."""
        # TODO(https://github.com/vllm-project/vllm/issues/36175): implement
        # native residual/prenorm path and unify with RMSNormGated.
        # For now, fall back to the triton kernel.
        if residual is not None or prenorm:
            return self.forward_cuda(x, g, residual, prenorm, residual_in_fp32)
        x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_float * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            x_normed = x_normed * self.weight.float()
        g_float = g.float()
        if self.activation in ("swish", "silu"):
            out = x_normed * g_float * torch.sigmoid(g_float)
        else:  # sigmoid
            out = x_normed * torch.sigmoid(g_float)
        return out.to(x.dtype)

    def forward_cuda(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        residual: torch.Tensor | None = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ) -> torch.Tensor:
        return rms_norm_gated(
            x,
            g,
            self.weight,
            self.bias,
            self.activation,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )
