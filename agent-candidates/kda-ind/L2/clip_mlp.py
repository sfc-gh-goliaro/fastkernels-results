"""CLIP MLP and text embeddings (L2), optimized for B200 / sm_100.

Both classes keep the baseline's ``__init__(config)`` / ``forward`` signatures and its
``state_dict`` key names, and both fall back to a pure-torch path that reproduces the
baseline exactly whenever the fast path cannot serve an input.

What the fast paths do, and why:

``CLIPTextEmbeddings`` moves only 118 KB, so the baseline's cost is entirely the three
kernels it launches (two gathers and an add) plus their launch latency.  One kernel does
the whole thing.  ``position_ids`` is ``arange(max_position_embeddings)`` sliced to the
sequence length, so for a single batch row the position row index *is* the token index
and the second gather collapses into an index computation.  The add happens in fp32 and
is rounded back to the storage dtype, matching PyTorch's ``opmath_t = float`` for
reduced-precision elementwise add, which makes the result bit-exact.

``CLIPMLP`` is a pair of GEMMs streaming 18.9 MB of weights.  Measurements in
``profile/probes/`` (p7-p9) and ``profile/fc2_mma_vs_cublas_20260912/`` show three
things about this shape:

  * Host enqueue cost is invisible.  The benchmark enqueues a 253 MB L2 flush
    immediately before its start event and never synchronizes inside the timing loop, so
    the CPU runs 73-114 us ahead of the GPU.  Injecting up to 60 us of artificial host
    cost per call does not move the measured latency at all.  The measured window is GPU
    time, so there is nothing to win by reducing pybind calls or dispatch overhead --
    and, usefully, freshly allocating the intermediate every call is free.
  * Both GEMMs are latency-bound, not bandwidth-bound, and cuBLAS is very good at them.
    A custom tf32 ``mma.sync`` fc2 profiled at 7.8 % tensor-pipe utilization and 811k
    shared loads against cuBLAS's 378, losing 33.4 us to 19.1 us; a plain-FFMA version
    lost by 35x.  So both GEMMs stay on ``at::addmm``.
  * What *is* cheap to win is GPU-side work the baseline wastes: the NN weight layout
    (``addmm`` against a physically transposed weight is bitwise identical to
    ``F.linear`` against the original, verified over 100 seed/init pairs, and ~1.9 us
    faster per GEMM), and the activation, which the baseline spends three kernels and
    three round-trips of the 946 KB intermediate on.

At the captured shapes the result is bit-exact with the baseline -- ``max_abs_error ==
0.0`` on both classes, not merely inside tolerance.  Away from them the MLP is not
guaranteed bitwise: cuBLAS may select a different kernel for a different M, so e.g.
sequence length 7 shows ~5e-7 of benign fp32 ordering noise (well inside the fp32
tolerance, matched ratio still 1.0).  ``CLIPTextEmbeddings`` is bit-exact at every shape
and dtype it serves, because a gather plus an fp32 add rounded back to the storage dtype
is exactly what PyTorch itself computes.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextConfig

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// ---------------------------------------------------------------------------
// QuickGELU, fused.
//
// ATen computes x * sigmoid(1.702 * x) for float as: the scalar multiply in fp32,
// then 1/(1+exp(-y)) with opmath_t = float.  Reproducing that expression with the
// accurate libdevice expf and a correctly-rounded divide -- i.e. with no fast-math
// flags on this translation unit -- makes this bitwise identical to the baseline's
// three-kernel version, verified to 0 differing values and max_ulp 0 over 3M inputs
// spanning the activation's dynamic range (profile/probes/p7, section f).
// ---------------------------------------------------------------------------
__device__ __forceinline__ float quickgelu(float x) {
  return x * (1.0f / (1.0f + expf(-1.702f * x)));
}

// In-place: the buffer is this extension's own fresh allocation, never a caller's
// tensor, so overwriting it cannot surprise anyone and it saves a second buffer.
__global__ void quickgelu_inplace_kernel(float* __restrict__ x, int n4, int n) {
  int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
    float4 v = reinterpret_cast<float4*>(x)[i];
    v.x = quickgelu(v.x); v.y = quickgelu(v.y);
    v.z = quickgelu(v.z); v.w = quickgelu(v.w);
    reinterpret_cast<float4*>(x)[i] = v;
  }
  // Tail for a width not divisible by 4 (the captured shape is, other shapes need not be).
  for (int i = n4 * 4 + blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    x[i] = quickgelu(x[i]);
}

static void launch_quickgelu_inplace(at::Tensor& x, cudaStream_t stream) {
  int n = (int)x.numel();
  int n4 = n / 4;
  int threads = 256;
  int blocks = (n4 > 0 ? (n4 + threads - 1) / threads : 1);
  if (blocks > 2048) blocks = 2048;
  quickgelu_inplace_kernel<<<blocks, threads, 0, stream>>>(
      x.data_ptr<float>(), n4, n);
  AT_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// CLIPMLP: one entry point, three kernels (cuBLAS, activation, cuBLAS).
//
// w1n / w2n arrive in NN layout (physically transposed), which is bitwise identical
// to the TN form F.linear uses but selects a faster cuBLAS kernel.  The intermediate
// is allocated per call: host cost is hidden by the benchmark's L2 flush, so a cached
// workspace would buy nothing and would introduce a reentrancy hazard.
// ---------------------------------------------------------------------------
at::Tensor clip_mlp_forward(at::Tensor x2d, at::Tensor w1n, at::Tensor b1,
                            at::Tensor w2n, at::Tensor b2) {
  const at::cuda::CUDAGuard guard(x2d.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto h = at::empty({x2d.size(0), w1n.size(1)}, x2d.options());
  at::addmm_out(h, b1, x2d, w1n);
  launch_quickgelu_inplace(h, stream);
  auto out = at::empty({x2d.size(0), w2n.size(1)}, x2d.options());
  at::addmm_out(out, b2, h, w2n);
  return out;
}

// ---------------------------------------------------------------------------
// CLIPTextEmbeddings: one entry point, one kernel.
// ---------------------------------------------------------------------------

// torch's extension build defines __CUDA_NO_BFLOAT16_CONVERSIONS__ /
// __CUDA_NO_HALF_CONVERSIONS__, so the cast operators are unavailable.  These
// intrinsics are round-to-nearest-even, which is what c10's float -> bf16/half
// conversion does, so the fp32 add rounded back through them is bit-exact.
template <typename T> struct Conv;
template <> struct Conv<float> {
  static __device__ __forceinline__ float to_f(float v) { return v; }
  static __device__ __forceinline__ float from_f(float v) { return v; }
};
template <> struct Conv<__nv_bfloat16> {
  static __device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
  static __device__ __forceinline__ __nv_bfloat16 from_f(float v) { return __float2bfloat16(v); }
};
template <> struct Conv<__half> {
  static __device__ __forceinline__ float to_f(__half v) { return __half2float(v); }
  static __device__ __forceinline__ __half from_f(float v) { return __float2half(v); }
};

// Vector payload for one access.  The alignment must be exactly the payload size, not
// a hard-coded 16: `alignas(16)` on a one-element struct would make sizeof(Vec) 16 as
// well, so the scalar (VEC == 1) fallback would store 16 bytes where 2 or 4 were
// intended -- an out-of-bounds write past the end of the row.  Sized this way, VEC == 4
// or 8 gives a single 128-bit access and VEC == 1 degrades to a plain scalar one.
//
// Alignment holds for the wide widths because the launcher only selects them when
// hidden % VEC == 0: the row stride hidden*sizeof(T) is then a whole number of
// payloads, col is a multiple of VEC, and the allocator's base is 256-byte aligned.
template <typename T, int VEC>
struct alignas(sizeof(T) * VEC) Vec { T d[VEC]; };

template <typename T, int VEC>
__global__ void token_pos_embed_kernel(
    const int64_t* __restrict__ ids, const T* __restrict__ tok,
    const T* __restrict__ pos, T* __restrict__ out,
    int seq, int hidden, int64_t vocab) {
  using V = Vec<T, VEC>;
  int vec_per_row = hidden / VEC;
  int total = seq * vec_per_row;
  for (int lane = blockIdx.x * blockDim.x + threadIdx.x; lane < total;
       lane += gridDim.x * blockDim.x) {
    int row = lane / vec_per_row;
    int col = (lane - row * vec_per_row) * VEC;
    int64_t id = ids[row];
    // The operator must be correct for any id in [0, vocab) and must never read out
    // of bounds -- the benchmark's index generator happens to stay under 1024, which
    // is not something to rely on.
    if (id < 0 || id >= vocab) id = 0;
    V tv = *reinterpret_cast<const V*>(tok + id * (int64_t)hidden + col);
    V pv = *reinterpret_cast<const V*>(pos + (int64_t)row * hidden + col);
    V ov;
#pragma unroll
    for (int v = 0; v < VEC; ++v)
      ov.d[v] = Conv<T>::from_f(Conv<T>::to_f(tv.d[v]) + Conv<T>::to_f(pv.d[v]));
    *reinterpret_cast<V*>(out + (int64_t)row * hidden + col) = ov;
  }
}

template <typename T, int VEC>
static void launch_token_pos_embed(at::Tensor& out, const at::Tensor& ids,
                                   const at::Tensor& tok, const at::Tensor& pos,
                                   int seq, int hidden, int64_t vocab,
                                   cudaStream_t stream) {
  int total = seq * (hidden / VEC);
  int threads = 128;
  int blocks = (total + threads - 1) / threads;
  token_pos_embed_kernel<T, VEC><<<blocks, threads, 0, stream>>>(
      ids.data_ptr<int64_t>(), reinterpret_cast<const T*>(tok.data_ptr()),
      reinterpret_cast<const T*>(pos.data_ptr()),
      reinterpret_cast<T*>(out.data_ptr()), seq, hidden, vocab);
  AT_CUDA_CHECK(cudaGetLastError());
}

// A 128-bit access needs two things, and contiguity implies neither of them:
//   * the row stride must be a whole number of payloads  -> hidden % VEC == 0
//   * every base pointer must be 16-byte aligned
// The second is easy to lose: a perfectly contiguous tensor built as
// `base[1:].view(vocab, hidden)` carries a nonzero storage offset, so its data_ptr is
// not 16-byte aligned, and either embedding weight can legitimately be rebound to such
// a view.  When any pointer fails, drop to the scalar path rather than issuing a
// misaligned typed load.
static bool aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

at::Tensor token_pos_embed(at::Tensor ids, at::Tensor tok, at::Tensor pos) {
  const at::cuda::CUDAGuard guard(ids.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  int seq = (int)ids.size(-1);
  int hidden = (int)tok.size(1);
  int64_t vocab = tok.size(0);
  auto out = at::empty({1, seq, hidden}, tok.options());
  const bool ptrs_ok = aligned16(tok.data_ptr()) && aligned16(pos.data_ptr())
                       && aligned16(out.data_ptr());
  auto st = tok.scalar_type();
  if (st == at::kFloat) {
    if (hidden % 4 == 0 && ptrs_ok)
      launch_token_pos_embed<float, 4>(out, ids, tok, pos, seq, hidden, vocab, stream);
    else
      launch_token_pos_embed<float, 1>(out, ids, tok, pos, seq, hidden, vocab, stream);
  } else if (st == at::kBFloat16) {
    if (hidden % 8 == 0 && ptrs_ok)
      launch_token_pos_embed<__nv_bfloat16, 8>(out, ids, tok, pos, seq, hidden, vocab, stream);
    else
      launch_token_pos_embed<__nv_bfloat16, 1>(out, ids, tok, pos, seq, hidden, vocab, stream);
  } else if (st == at::kHalf) {
    if (hidden % 8 == 0 && ptrs_ok)
      launch_token_pos_embed<__half, 8>(out, ids, tok, pos, seq, hidden, vocab, stream);
    else
      launch_token_pos_embed<__half, 1>(out, ids, tok, pos, seq, hidden, vocab, stream);
  } else {
    TORCH_CHECK(false, "token_pos_embed: unsupported dtype ", st);
  }
  return out;
}
"""

_CPP_DECL = r"""
at::Tensor clip_mlp_forward(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor);
at::Tensor token_pos_embed(at::Tensor, at::Tensor, at::Tensor);
"""

# Name unique to this operator: the torch-extensions build directory is shared across
# workspaces, so a generic name could collide with another operator's cached .so.
_EXT_NAME = "fk_l2_clip_mlp_ext"

#: ``"ok"`` once the extension is loaded, ``"unavailable: <reason>"`` if the build
#: failed, ``"pending"`` before the first attempt.  A fast run and a fallback run must
#: never be indistinguishable from the outside.
EXTENSION_STATUS = "pending"

_ext = None
_ext_tried = False


def _pin_build_arch() -> None:
    """Target only the local architecture.

    The ambient ``TORCH_CUDA_ARCH_LIST`` in this environment names seven of them, which
    would multiply compile time for no benefit.  The ``a`` suffix follows the
    FastKernels convention in ``fastkernels/infra/cuda_ext.py`` for major >= 9.
    """
    override = os.environ.get("FASTKERNELS_CUDA_ARCH_LIST")
    if override is not None:
        if override.strip():
            os.environ["TORCH_CUDA_ARCH_LIST"] = override
        return
    if not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability()
    suffix = "a" if major >= 9 else ""
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}{suffix}"


def _load_extension():
    """Build (or reuse) the extension.  A failure degrades to the pure-torch path."""
    global _ext, _ext_tried, EXTENSION_STATUS
    if _ext_tried:
        return _ext
    _ext_tried = True

    if os.environ.get("FK_CLIP_MLP_DISABLE_EXT"):
        EXTENSION_STATUS = "unavailable: disabled by FK_CLIP_MLP_DISABLE_EXT"
        return None
    if not torch.cuda.is_available():
        EXTENSION_STATUS = "unavailable: no CUDA device"
        return None

    try:
        from torch.utils.cpp_extension import _get_build_directory, load_inline

        # Keep the build cache inside the workspace (already gitignored) so it is warm
        # across runs and isolated from other operators.
        os.environ.setdefault(
            "TORCH_EXTENSIONS_DIR",
            str(Path(__file__).resolve().parents[2] / ".torch_extensions"))
        _pin_build_arch()

        # A cold build must stream progress: the benchmark worker's stall watchdog is
        # evaluated against its log's mtime (600 s) inside a 1200 s per-worker budget,
        # so a silent multi-minute compile risks being killed as a hang.
        pending = True
        try:
            so = Path(_get_build_directory(_EXT_NAME, verbose=False)) / f"{_EXT_NAME}.so"
            pending = not so.exists()
        except Exception:
            pass
        if pending:
            print(f"[clip_mlp] building CUDA extension {_EXT_NAME!r} for arch "
                  f"{os.environ.get('TORCH_CUDA_ARCH_LIST', 'auto')!r} -- one-time JIT "
                  f"compile, streaming ninja progress ...", flush=True)

        # No fast-math: approximate expf or reciprocal division would cost the fused
        # activation its bit-exactness against ATen's sigmoid.
        _ext = load_inline(
            name=_EXT_NAME, cpp_sources=_CPP_DECL, cuda_sources=_CUDA_SRC,
            functions=["clip_mlp_forward", "token_pos_embed"],
            extra_cflags=["-O3"], extra_cuda_cflags=["-O3"], verbose=pending)
        EXTENSION_STATUS = "ok"
    except Exception as exc:  # noqa: BLE001 - any build failure must stay non-fatal
        _ext = None
        EXTENSION_STATUS = f"unavailable: {type(exc).__name__}: {exc}"
        warnings.warn(f"[clip_mlp] CUDA extension unavailable, using the pure-torch "
                      f"path: {EXTENSION_STATUS}", RuntimeWarning, stacklevel=2)
    return _ext


def _tensor_identity(t: torch.Tensor) -> tuple:
    """Cache key covering every way a parameter can change underneath a derived copy.

    ``data_ptr`` alone is not enough: a different tensor can land on the same address,
    and an in-place mutation keeps the address.  ``_version`` catches the mutation,
    the rest catch re-binding, resizing, restriding and device moves.
    """
    return (t.device, t.dtype, tuple(t.shape), tuple(t.stride()),
            t.storage_offset(), t.data_ptr(), t._version)


class Matmul(nn.Module):
    """Parameter-free, mirroring the baseline's ``Linear.matmul`` so key names match."""

    def forward(self, input, weight, bias=None):
        return F.linear(input, weight, bias)


class Linear(nn.Module):
    """Same parameters and submodule layout as ``L1.linear.Linear``.

    Only ``weight`` and ``bias`` appear in ``state_dict``, which is what the benchmark
    shares from the baseline.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.matmul = Matmul()

    def forward(self, input):
        return self.matmul(input, self.weight, self.bias)


class Embedding(nn.Module):
    """Same nesting as ``L1.embedding.Embedding``, so the key is ``<name>.emb.weight``."""

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 padding_idx: int | None = None):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, input_ids):
        return self.emb(input_ids)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class CLIPMLP(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.fc1 = Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.activation_fn = QuickGELU()
        # Derived weights are built on first forward, never in __init__: at
        # construction time the parameters are still uninitialized, and the benchmark
        # overwrites them afterwards via load_state_dict.
        self._nn_weights = None
        self._nn_key = None

    def _nn_layout_weights(self):
        """Physically transposed copies of both weights, cached until they change.

        ``addmm(bias, x, W.t().contiguous())`` is bitwise identical to
        ``F.linear(x, W, bias)`` -- verified over 20 input seeds x 5 weight
        initializations for both GEMMs and for the M-padded form
        (``profile/probes/p7``, section d) -- but it selects a faster cuBLAS kernel.
        """
        w1, w2 = self.fc1.weight, self.fc2.weight
        key = (_tensor_identity(w1), _tensor_identity(w2))
        if self._nn_key != key:
            self._nn_weights = (w1.t().contiguous(), w2.t().contiguous())
            self._nn_key = key
        return self._nn_weights

    def _torch_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        return self.fc2(hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ext = _load_extension()
        w1, w2 = self.fc1.weight, self.fc2.weight
        b1, b2 = self.fc1.bias, self.fc2.bias
        # Rank 3 only.  The flatten/unflatten would be exact for any rank >= 2, but the
        # acceptance criteria require unexpected ranks to take the pure-torch path, and
        # every call site that exists -- the captured case, L3 clip_encoder_layer, L4
        # clip_text_model -- passes rank 3, so nothing real is pushed onto the slow path.
        # tests/test_clip_mlp.py asserts the route for every case with a call-counting
        # spy rather than inferring it from output equality.
        if not (
            ext is not None
            and not torch.is_grad_enabled()
            and b1 is not None and b2 is not None
            and hidden_states.is_cuda and hidden_states.dtype is torch.float32
            and hidden_states.is_contiguous() and hidden_states.dim() == 3
            and hidden_states.size(-1) == w1.size(1)
            and w1.is_cuda and w2.is_cuda
            and w1.dtype is torch.float32 and w2.dtype is torch.float32
            and w1.is_contiguous() and w2.is_contiguous()
            and b1.dtype is torch.float32 and b2.dtype is torch.float32
            and b1.is_contiguous() and b2.is_contiguous()
        ):
            # Everything the kernels do not serve -- reduced precision, CPU tensors,
            # non-contiguous input, any rank other than 3, a missing bias, a hidden size
            # that disagrees with fc1, and anything that needs autograd -- goes to the
            # baseline's own formulation.
            return self._torch_forward(hidden_states)

        w1n, w2n = self._nn_layout_weights()
        shape = hidden_states.shape
        x2d = hidden_states.reshape(-1, shape[-1])
        out = ext.clip_mlp_forward(x2d, w1n, b1, w2n, b2)
        return out.view(*shape[:-1], w2n.size(1))


class CLIPTextEmbeddings(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.token_embedding = Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = Embedding(config.max_position_embeddings,
                                            config.hidden_size)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )
        self._canonical_pos_key = None
        self._canonical_pos_ok = False

    def _positions_are_canonical(self, position_ids: torch.Tensor) -> bool:
        """Confirm ``position_ids`` really is ``arange(...)`` before folding it away.

        The kernel replaces the position gather with a row-index computation, which is
        only valid for the canonical buffer.  The comparison is a real elementwise
        check, cached on the buffer's identity so it runs during the benchmark's
        correctness rounds rather than inside the timed window.  ``.to(device)``
        rebinds the buffer, so the key deliberately includes the device.
        """
        key = _tensor_identity(position_ids)
        if self._canonical_pos_key != key:
            expected = torch.arange(position_ids.size(-1),
                                    device=position_ids.device,
                                    dtype=position_ids.dtype)
            self._canonical_pos_ok = bool(
                position_ids.size(0) == 1
                and torch.equal(position_ids.view(-1), expected))
            self._canonical_pos_key = key
        return self._canonical_pos_ok

    def _torch_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_length = input_ids.shape[-1]
        position_ids = self.position_ids[:, :seq_length]
        return self.token_embedding(input_ids) + self.position_embedding(position_ids)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        ext = _load_extension()
        tok = self.token_embedding.emb.weight
        pos = self.position_embedding.emb.weight
        seq_length = input_ids.shape[-1]
        if not (
            ext is not None
            and not torch.is_grad_enabled()
            and input_ids.is_cuda and input_ids.dtype is torch.int64
            and input_ids.is_contiguous()
            # The position-index shortcut is only correct for one batch row; anything
            # wider is served by the torch path.
            and input_ids.dim() == 2 and input_ids.size(0) == 1
            and tok.is_cuda and pos.is_cuda
            and tok.dtype is pos.dtype
            and tok.dtype in (torch.float32, torch.bfloat16, torch.float16)
            and tok.is_contiguous() and pos.is_contiguous()
            and tok.dim() == 2 and pos.dim() == 2
            and tok.size(1) == pos.size(1)
            and seq_length <= pos.size(0)
            and self._positions_are_canonical(self.position_ids)
        ):
            return self._torch_forward(input_ids)

        return ext.token_pos_embed(input_ids, tok, pos)
