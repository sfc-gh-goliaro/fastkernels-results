"""AF3 MSA pair-weighted averaging (Algorithm 10) as one fused kernel.

``baseline.py`` composes this operator out of twenty CUDA kernels: two LayerNorms,
four projections, a mask-bias construction, a softmax, an einsum, a sigmoid and a
multiply.  On the captured shape that composition measures 232-341 us while a CUDA
trace of it shows only ~46 us of device time.  The gap is not arithmetic.

What the scored window is actually made of, measured through the harness's own
``_time_module`` (``profile/measure_launch_budget.py``, ``profile/measure_window_model.py``):

    window = 13.3 us  (three shifting-pool copies against a just-flushed L2)
           + sum over the kernels we put on the stream of (~2 us + that kernel's duration)

with two consequences that between them fix this design:

  * **Host-side Python is free.**  A ``forward`` doing 26.6 us of pure Python per call
    still measures 13.31 us -- byte-identical to an empty ``forward`` -- because the
    harness zeroes a 252 MB L2-flush buffer every iteration and never synchronizes
    inside the timing loop, so the CPU runs ~36 us ahead of the GPU and the two
    events bracket GPU time only.  Eighty parameter reads off the module (14.7 us of
    host time) likewise cost 0.00 us.  This is why the weights are read off the
    module on every call and nothing whatsoever is cached: the fully coherent option
    is also the free one.
  * **Kernel duration is *not* free.**  A tunable-duration kernel measured
    1.51/3.92/13.34/25.10/48.85/96.19/190.80 us of device time and moved the window by
    +4.06/+6.11/+14.30/+26.59/+51.17/+98.24/+192.45 us -- duration lands on the score
    roughly 1:1 on top of a fixed ~2 us per launch.  There is no pool of hidden GPU
    slack to spend arithmetic redundancy out of.

So the objective is the number of *kernels* on the stream, and after that their
duration.  One kernel it is: a second trivial raw kernel issued from inside the same
extension call measured +4.06 us (20.45 -> 24.51 us), and adding the workspace a real
split would need cost nothing further -- the launch is the whole price.  A split into
an s-independent pair-path kernel plus a per-row kernel would remove real GPU work,
but it has to buy back more than 4 us to pay for itself.  Recorded, not taken.

Two further measurements that did *not* change anything, kept because a null result is
worth as much as a win: reaching the same raw kernel through a bound ``torch.library``
overload instead of pybind measured -0.03 us (i.e. nothing), and passing eight extra
tensor arguments measured +0.00 us whether they came from a cached tuple or were read
off the module.

The kernel
----------
One block per ``(batch, sequence)`` row **and query-row tile** -- ``grid = (B*S,
ceil(N/tile))``, with the tile fixed at 4 by measurement -- 256 threads, dynamic shared
memory sized on the host.  Every stage strides its work across all 256 threads, so the
kernel is generic in ``B``, ``S`` and ``N``; the channel geometry is a template
parameter because the pair-path row has to stay in registers.  Stages, in order:

  staging       the eight weight tensors into shared memory, transposing the three
                projection matrices on the way in
  pair path     ``layer_norm_z`` fused with the head projection, one thread per
                ``(i, j)`` pair, the 128-channel row held in registers across both
                reduction passes and overwritten in place with the normalised value
  weights       softmax over ``j``, one thread per ``(head, i)`` row, serial and so
                needing no cross-lane communication at any ``N``
  msa path      ``layer_norm_m``, one warp per row
  values        ``V = m_hat . linear_v^T``
  average+gate  ``o = W . V``, then ``o * sigmoid(m_hat . linear_g^T)``
  output        ``o . linear_o^T`` straight to global memory

The three projection matrices are transposed during the staging copy rather than
pre-packed on the host.  The read pattern in the projection stages is "32 lanes,
consecutive output index, same reduction index"; against a row-major ``[DH][CM]`` that
is a stride of ``CM * 2 = 128`` bytes, and with 32 banks of 4 bytes every lane lands
in the same bank -- a 32-way conflict.  Storing reduction-major makes those 32 lanes
consecutive.  Transposing during staging (rather than into a packed host buffer) is
also what lets the live ``nn.Parameter`` tensors be passed straight through with no
derived state to go stale.

Numerics
--------
The composition rounds to bf16 in thirteen places, and the kernel reproduces every
one of them rather than resting on the harness's loose ``atol=rtol=1e-2``.  Writing
``[x]`` for round-to-nearest-even to bf16, with every unmarked accumulation in fp32,
and marking where each appears below:

     1  ``[mask - 1]``                          ``pair_bias``, ``s1``
     2  ``[inf * (1)]``                         ``pair_bias``, ``bias``
     3  ``[layer_norm_z(z)]``                   ``pair_path``, the in-place row rewrite
     4  ``[sum_c z_hat * wz]``                  ``pair_path``, ``logit_raw``
     5  ``[(4) + bias]``                        ``pair_path``, ``logit``
     6  ``[softmax_j (5)]``                     ``pair_weights``, the third pass
     7  ``[layer_norm_m(m)]``                   ``msa_path``, the store to ``Mn``
     8  ``[sum_c m_hat * wv]``                  ``value_proj``, the store to ``V``
     9  ``[sum_c m_hat * wg]``                  ``average_and_gate``, ``gate_raw``
    10  ``[sigmoid((9))]``                      ``average_and_gate``, ``gate``
    11  ``[sum_j W * V]``                       ``average_and_gate``, ``avg``
    12  ``[(11) * (10)]``                       ``average_and_gate``, the store to ``Og``
    13  ``[sum_d (12) * wo]``                   ``output_proj``, the store to global

Eager evaluates them in source order ``1 2 3 4 5 6 7 8 11 9 10 12 13``; the list above
is dependency order, and the two agree on every site.

Two of these are easy to get wrong and worth stating.  Site 2 does *not* round ``inf``
to bf16 first: a Python float meets a bf16 tensor as a wrapped scalar, which does not
promote the tensor, so the scalar stays fp32 and only the product is rounded -- hence
``inf`` arrives as a ``double`` and is used as ``float``.  Site 5 rounds the biased
logit to bf16 *before* the softmax reads it, so the logits are stored in bf16; holding
them in fp32 would be both wasteful and a skipped site.

What this buys is measured rather than assumed.  ``profile/derive_sites.py`` builds the
same thirteen sites out of explicit torch ops and reproduces ``baseline.py``
**bit-for-bit** (``max_abs = 0.0``) at three seeds and four mask patterns, and skipping
any single site moves the output by at most 6.1e-5 -- far inside ``atol=1e-2`` at this
benchmark's weight scale.  So the sites are not what makes the harness pass; they are
what makes the correctness argument independent of that scale, since the error at site
3 is relative to logit magnitude and a checkpoint with larger ``linear_z`` weights
turns a 2^-9 relative error into a ~1% softmax-weight error.  Bit-exact agreement is
*not* claimed anywhere and is not achievable: cuBLAS picks its own reduction tree and
ATen's LayerNorm reduces with online Welford where this kernel makes two register
passes.  Reproducing the rounding *sites* does not imply ULP equality.

Consequences of building the bias as ``inf * (mask - 1)`` in bf16 rather than as a
select: a caller passing ``inf=float('inf')`` gets NaN on *unmasked* entries
(``mask == 1`` so ``mask - 1 == 0``, and ``inf * 0`` is NaN) and ``-inf`` on masked
ones -- from this kernel and from the baseline alike.  The captured configuration uses
``inf=1e9``, which rounds to ``bf16(1e9) = 998244352``, and an all-masked row therefore
gives every logit the identical bf16 value (harness-scale logits are far below the
fp32 spacing of 64 at that exponent), so max-subtraction zeroes them and softmax
returns a uniform ``1/N``.  That is a fidelity property, and it is why the bias must
not be "optimised" into a branch.

What is not covered, and what happens instead
---------------------------------------------
A C++ predicate decides eligibility and returns an undefined tensor -- ``None`` through
pybind -- for everything it does not claim; ``forward`` then runs ``_reference``, which
reproduces the baseline *formula* through this module's own frozen submodules, so a
rejected input degrades to already-benchmarked kernels rather than to raw eager ATen.
Rejected: any dtype but bf16; an input with a leading batch rank other than one, since
the fused path takes exactly ``m[B, S, N, CM]`` and ``z[B, N, N, CZ]`` while the
signature's ``[*]`` permits any number of leading dimensions; a CUDA device whose
compute capability differs from the one the extension was compiled for; a
non-contiguous or misaligned input or weight; a CPU tensor; a tensor carrying a lazy negative or conjugate bit, a name, a forward-mode
tangent, or a non-strided layout; a tensor subclass, functorch wrapper, fake or
functional tensor, or an active ``__torch_function__`` mode; a call made under
``torch.jit.trace`` or CUDA autocast; a channel geometry other than the captured one;
a shape whose shared-memory requirement exceeds 48 KB; an empty tensor; and -- by
design -- any call made with gradient mode enabled.

The gradient gate is ``at::GradMode::is_enabled()`` *alone*, never ``requires_grad`` on
any tensor: this module's eight weights are ``nn.Parameter``s whose ``requires_grad``
is ``True``, so a ``requires_grad`` test would reject the fast path on every single
call.  Forward-mode duals are screened separately, because a dual tensor has
``requires_grad == False`` and reading its primal through a pointer would silently drop
the tangent.  The fused path is therefore explicitly inference-only; a training-time
caller transparently gets the composition, which is differentiable.

One case this argument list cannot see: ``torch.compile``.  Dynamo treats the pybind
entry point as opaque and graph-breaks rather than mis-tracing it, and during tracing
the FakeTensor arguments are rejected by the subclass screen, so the traced graph
contains the composition.  Correct, not fast, and not fixable from inside a raw
extension.  A tensor subclass that overrides only ``__torch_function__`` (rather than
``__torch_dispatch__``) is likewise invisible in C++: it carries no dispatch key and
ordinary storage, so the kernel reads the right bytes but the override does not run.

One narrow limitation is inherited rather than chosen, and is recorded because it is a
real difference from ``baseline.py``. The predicate declines a forward-mode dual and
routes it to the composition, which is the correct decision, and under ordinary
grad-enabled use the tangent then survives that composition intact -- which is how
forward-mode AD is actually used. It is dropped only in the combination of a dual
tensor *inside* ``torch.no_grad()``, where ``candidate/L1/layer_norm.py``'s own
``GradMode``-gated fused path engages and reads the primal through a pointer. Measured
in all four combinations: ``baseline.py`` preserves the tangent in both grad modes,
this module preserves it with grad enabled and drops it under ``no_grad``, identically
whether or not this file's own extension is loaded -- which is what locates the cause
in the frozen dependency rather than here. Unlike the lazy-bit case below it cannot be
repaired by normalising the input, because a tangent has to flow *through* each op
rather than be materialised before it, and the fix would mean not composing the frozen
winners this file is required to use.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.forward_ad as fwad

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.sigmoid import Sigmoid
from ..L1.softmax import Softmax

# Route codes reported by ``msa_plan``. Mirrors kRoute* in the CUDA source.
ROUTE_FALLBACK = 0
ROUTE_FUSED = 1

# Why the predicate declined, for tests that need to assert *which* screen fired
# rather than only that the fast path was not taken. Mirrors kReject* in the source.
REJECT_NONE = 0
REJECT_MODE = 1          # grad mode, tracing, autocast, __torch_function__ mode, TLS
REJECT_WRAPPER = 2       # subclass, dispatch key, missing or uninitialized storage
REJECT_LAZY = 3          # negative or conjugate bit, named tensor, forward-mode dual
REJECT_DEVICE = 4        # not CUDA, not strided, wrong dtype, mixed devices
REJECT_SHAPE = 5         # rank, extent disagreement, weight shape
REJECT_STRIDE = 6        # not standard-contiguous
REJECT_EMPTY = 7         # zero elements
REJECT_GEOMETRY = 8      # channel geometry with no instantiation
REJECT_SMEM = 9          # over the 48 KB dynamic shared-memory budget
REJECT_BOUNDS = 10       # extent or grid past what the launch can express
REJECT_ALIGN = 11        # a pointer, including the fresh output, not 16-byte aligned
REJECT_LAUNCH = 12       # occupancy query says this configuration cannot be resident

REJECT_NAMES = {
    REJECT_NONE: "none",
    REJECT_MODE: "dispatch_mode",
    REJECT_WRAPPER: "wrapper_or_storage",
    REJECT_LAZY: "lazy_bit_or_dual",
    REJECT_DEVICE: "device_dtype_layout",
    REJECT_SHAPE: "shape",
    REJECT_STRIDE: "stride",
    REJECT_EMPTY: "empty",
    REJECT_GEOMETRY: "channel_geometry",
    REJECT_SMEM: "shared_memory_budget",
    REJECT_BOUNDS: "integer_or_grid_bounds",
    REJECT_ALIGN: "alignment",
    REJECT_LAUNCH: "not_launchable",
}

PLAN_FIELDS = (
    "route", "reject", "batch", "seq", "n_res", "c_z", "c_m", "heads",
    "c_hidden", "d_head", "tile", "n_tiles", "threads", "smem_bytes",
)


def _permute_final_dims(tensor: torch.Tensor, inds: tuple[int, ...]) -> torch.Tensor:
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])

_CPP_SOURCE = r"""
#include <ATen/ATen.h>
#include <cstdint>
#include <optional>

at::Tensor msa_pair_weighted_avg(
    const at::Tensor& m, const at::Tensor& z, const std::optional<at::Tensor>& mask,
    const std::optional<at::Tensor>& gz, const std::optional<at::Tensor>& bz,
    const std::optional<at::Tensor>& gm, const std::optional<at::Tensor>& bm,
    const std::optional<at::Tensor>& wz, const std::optional<at::Tensor>& wv,
    const std::optional<at::Tensor>& wg, const std::optional<at::Tensor>& wo,
    double inf, int64_t tile);

at::Tensor msa_plan(
    const at::Tensor& m, const at::Tensor& z, const std::optional<at::Tensor>& mask,
    const std::optional<at::Tensor>& gz, const std::optional<at::Tensor>& bz,
    const std::optional<at::Tensor>& gm, const std::optional<at::Tensor>& bm,
    const std::optional<at::Tensor>& wz, const std::optional<at::Tensor>& wv,
    const std::optional<at::Tensor>& wg, const std::optional<at::Tensor>& wo,
    int64_t tile);

int64_t fused_calls();
int64_t declined_calls();
void reset_call_counters();
"""

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/PythonTorchFunctionTLS.h>
#include <ATen/autocast_mode.h>
#include <ATen/core/grad_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <cuda_bf16.h>

#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <limits>

namespace {

// How many calls each path served. Relaxed atomics, one increment per call, so a
// test can bracket a single forward and tell which route that specific call took.
// The build-status flag alone only proves the extension loaded, not that any
// particular call used it -- and a latency recorded while the reference was running
// is not a measurement of this kernel.
std::atomic<uint64_t> g_fused_calls{0};
std::atomic<uint64_t> g_declined_calls{0};

constexpr int64_t kRouteFallback = 0;
constexpr int64_t kRouteFused = 1;

constexpr int64_t kRejectNone = 0;
constexpr int64_t kRejectMode = 1;
constexpr int64_t kRejectWrapper = 2;
constexpr int64_t kRejectLazy = 3;
constexpr int64_t kRejectDevice = 4;
constexpr int64_t kRejectShape = 5;
constexpr int64_t kRejectStride = 6;
constexpr int64_t kRejectEmpty = 7;
constexpr int64_t kRejectGeometry = 8;
constexpr int64_t kRejectSmem = 9;
constexpr int64_t kRejectBounds = 10;
constexpr int64_t kRejectAlign = 11;
constexpr int64_t kRejectLaunch = 12;

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
// The default dynamic shared-memory limit. Zero of the 48 frozen files in
// candidate/L1 call cudaFuncSetAttribute to raise it; every one of them stays under
// it by construction and checks it. Staying inside the convention removes a class of
// launch failure, and a large-N fast path would need a tiled-j online softmax rather
// than a bigger allocation anyway: the O(H*N^2) logit term reaches ~786 KB at the
// real AF3 N=128, H=8.
constexpr int64_t kSmemBudget = 48 * 1024;
constexpr float kLayerNormEps = 1e-5f;

// ---------------------------------------------------------------------------
// Admission
// ---------------------------------------------------------------------------

// Tensors carrying any of these keys must not reach a raw-pointer path: some have no
// storage to point at (a functorch-batched tensor throws from data_ptr(), a nested
// tensor throws from strides()), some have a null pointer that would otherwise look
// 16-byte aligned (a ZeroTensor), and for the conjugate and negative keys the lazily
// applied transformation lives outside the bytes in storage.
const c10::DispatchKeySet kUnsupportedKeys({
    c10::DispatchKey::Conjugate,
    c10::DispatchKey::Negative,
    c10::DispatchKey::ZeroTensor,
    c10::DispatchKey::NestedTensor,
    c10::DispatchKey::BatchedNestedTensor,
    c10::DispatchKey::Python,
    c10::DispatchKey::PythonTLSSnapshot,
    c10::DispatchKey::Functionalize,
    c10::DispatchKey::FuncTorchBatched,
    c10::DispatchKey::FuncTorchGradWrapper,
    c10::DispatchKey::FuncTorchDynamicLayerFrontMode,
    c10::DispatchKey::FuncTorchDynamicLayerBackMode,
    c10::DispatchKey::Batched,
    c10::DispatchKey::VmapMode,
    c10::DispatchKey::FuncTorchVmapMode,
});

// Thread-local dispatch state is not visible in any tensor's key_set(), and the
// dispatcher computes its key from tensor keys *plus* the TLS included set *minus*
// the TLS excluded set. Rather than maintain a second incomplete list of keys to
// reject, allow only the two states an ordinary inference call produces: the default
// sets, or exactly the modification c10::InferenceMode makes. Anything else -- a mode
// stack, a partially entered guard, an exclusion we did not anticipate -- declines.
bool tls_state_is_ordinary() {
  const auto tls = c10::impl::tls_local_dispatch_key_set();
  if (tls.included_ == c10::default_included_set
      && tls.excluded_ == c10::default_excluded_set) {
    return true;
  }
  // Inference mode is the one non-default state that is safe, and this admits it
  // wholesale rather than matching its exact key sets: the fresh output allocated below
  // is then an inference tensor, which is what ATen would have produced anyway, and
  // reading an inference input through a pointer is no different from reading an
  // ordinary one. Note the consequence -- while inference mode is enabled this returns
  // true for *any* TLS state, so a mode entered inside inference mode is not screened
  // here. Everything with a dispatch key of its own is still caught by the per-tensor
  // key screen and by the torch_function / autocast / tracing tests above.
  return c10::InferenceMode::is_enabled();
}

// State that does not depend on any particular tensor. Checked before touching a
// tensor at all, so nothing here can throw from an accessor.
bool ambient_state_ok() {
  // Grad mode being *enabled* is the whole test, not whether some tensor currently
  // requires grad. This module holds eight nn.Parameters whose requires_grad is
  // True, so a requires_grad test would decline every call; and a caller who has not
  // entered no_grad may attach requires_grad later in the same graph, or wrap this in
  // checkpointing that replays the forward under different requires_grad state. The
  // kernel allocates with at::empty and records nothing, so a grad-enabled call must
  // reach the differentiable composition instead.
  if (at::GradMode::is_enabled()) return false;
  if (torch::jit::tracer::isTracing()) return false;
  // A traced or autocast call must see the ATen ops, not an opaque kernel. Autocast
  // would additionally have cast the inputs on the way in, which this never does.
  if (at::autocast::is_autocast_enabled(at::kCUDA)) return false;
  // isTensorSubclassLike catches an active __torch_dispatch__ mode but not a
  // __torch_function__ mode, which is a separate TLS stack.
  if (at::impl::torch_function_mode_enabled()) return false;
  return tls_state_is_ordinary();
}

// Shallow screen: everything that must hold before ordinary metadata accessors are
// safe to call. The subclass test comes before has_storage() -- and before anything
// else -- because a wrapped tensor can throw from the very accessors the later checks
// use, and because isTensorSubclassLike dereferences the impl and so must never see
// an undefined tensor.
bool shallow_ok(const at::Tensor& t) {
  if (!t.defined()) return false;
  if (at::isTensorSubclassLike(t)) return false;
  if (t.key_set().has_any(kUnsupportedKeys)) return false;
  if (!t.has_storage()) return false;
  // A nonempty tensor can still carry storage that was never allocated.
  if (!t.unsafeGetTensorImpl()->storage_initialized()) return false;
  return true;
}

// Per-tensor screen, run after shallow_ok for the same tensor. Returns the code of
// the screen that fired rather than a bool, so the route query can report *which*
// property disqualified an input -- a test that only knows "not fused" cannot tell a
// dtype rejection from a stride one.
// Per-tensor screen, run after `prescreen` for the same tensor. One check per line, in one
// order, and it is the order the checklist documents:
//
//     lazy bits / names / forward tangent -> CUDA device -> bf16 dtype -> strided layout -> rank
//
// then, back in the caller and in this order: cross-tensor device agreement, shapes, explicit
// standard strides, emptiness, and finally pointer validation.
//
// The order is not cosmetic and it is not only about safety. It fixes which answer a tensor
// that fails several checks reports, and that answer is the entire product of the route query.
// A negated bf16 view is a *lazy* rejection, not a dtype one, even though it is also not
// something the kernel could read.
//
// Two things this function deliberately does NOT do, both so the sequence stays honest:
//
//   * it does not test emptiness -- that has to follow the shape and stride checks, so an empty
//     tensor with a wrong rank reports the rank, and the caller runs it as its own pass;
//   * it does not validate the storage pointer -- an empty tensor may legitimately have a null
//     one, so rejecting on null here would report every empty input as a wrapper failure.
//     The caller validates the pointer only after emptiness has been decided, at which point a
//     null pointer really is a wrapper failure.
int64_t tensor_reject(const at::Tensor& t, int64_t rank) {
  // A lazily applied bit leaves the un-negated / unconjugated values in storage, which a raw
  // data_ptr() would read as if they were the real ones. (A tensor carrying the Negative or
  // Conjugate *dispatch key* never reaches here -- `prescreen` screens the key set first, which
  // is the correct precedence. These tests catch the bit without the key.)
  if (t.is_neg() || t.is_conj()) return kRejectLazy;
  // Names would be dropped by the fresh output. Suppressing them with a NoNamesGuard would turn
  // a name error into silent success, so decline instead.
  if (t.has_names()) return kRejectLazy;
  // A forward-mode dual has requires_grad == False, so the grad-mode gate does not catch it;
  // reading its primal through a pointer would drop the tangent with no error. This is the
  // predicate ATen's own isFwGradDefined uses.
  if (t._fw_grad(/*level=*/0).defined()) return kRejectLazy;

  if (!t.is_cuda()) return kRejectDevice;
  if (t.scalar_type() != at::kBFloat16) return kRejectDevice;  // allowlist, not negation
  if (t.layout() != at::kStrided) return kRejectDevice;

  if (t.dim() != rank) return kRejectShape;
  return kRejectNone;
}

// Emptiness, as its own pass so it runs after the shape and stride checks. An empty input
// reaches the composition, which raises whatever the baseline raises for it.
int64_t empty_reject(const at::Tensor& t) {
  return t.numel() == 0 ? kRejectEmpty : kRejectNone;
}

// Storage pointer, as its own pass so it runs after emptiness. By the time this runs the tensor
// is known to be non-empty, so a null pointer is a genuine wrapper failure rather than the
// ordinary consequence of having no elements -- and it must be caught explicitly, because
// `aligned16(nullptr)` is true and the alignment pass would wave it through.
int64_t pointer_reject(const at::Tensor& t) {
  return t.const_data_ptr() == nullptr ? kRejectWrapper : kRejectNone;
}

struct Weights {
  const at::Tensor* t[8];
};

// The pre-guard phase: ambient state, then the shallow wrapper screen for every present
// tensor, and nothing else. Split out from make_plan so that no entry point reads a
// tensor's device -- or installs a CUDAGuard on it -- before the screen that makes
// ordinary metadata accessors safe to call in the first place. That is the whole point of
// putting the wrapper screen first, and a caller that peeks at `m.is_cuda()` to decide
// whether to guard has already defeated it.
int64_t prescreen(const at::Tensor& m, const at::Tensor& z, const at::Tensor& mask,
                  const at::Tensor* const weights[8]) {
  if (!ambient_state_ok()) return kRejectMode;
  if (!shallow_ok(m) || !shallow_ok(z)) return kRejectWrapper;
  if (mask.defined() && !shallow_ok(mask)) return kRejectWrapper;
  for (int i = 0; i < 8; ++i) {
    if (weights[i] == nullptr || !shallow_ok(*weights[i])) return kRejectWrapper;
  }
  return kRejectNone;
}

// Contiguity by an explicit standard-stride comparison rather than is_contiguous()
// alone: the latter reports true for a tensor whose size-1 dimensions carry arbitrary
// strides, and the flat indexing below reads those strides as if they were standard.
bool standard_strides(const at::Tensor& t) {
  const auto sizes = t.sizes();
  const auto strides = t.strides();
  int64_t expect = 1;
  for (int64_t d = sizes.size() - 1; d >= 0; --d) {
    if (strides[d] != expect) return false;
    expect *= sizes[d];
  }
  return true;
}

bool aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

// Multiply with an explicit ceiling, so nothing is ever computed and *then* checked --
// by which point it has already wrapped. Returns false on overflow past the bound.
bool mul_within(int64_t a, int64_t b, int64_t bound, int64_t* out) {
  if (a < 0 || b < 0) return false;
  if (a != 0 && b > bound / a) return false;
  *out = a * b;
  return *out <= bound;
}

struct Plan {
  int64_t route = kRouteFallback;
  int64_t reject = kRejectNone;
  int64_t B = 0, S = 0, N = 0;
  int64_t CZ = 0, CM = 0, H = 0, CH = 0, DH = 0;
  int64_t tile = 0, ntiles = 0;
  int64_t threads = kThreads;
  int64_t smem = 0;
};

// Shared-memory layout, in bf16 elements, with every section start rounded up to a
// multiple of 8 so each one begins on a 16-byte boundary. Written without assuming
// DH == CM, which is an accident of the captured configuration: linear_o maps
// DH -> CM and the two are independent in general.
struct SmemLayout {
  int64_t gz, bz, gm, bm, wz, wv, wg, wo, logits, mnorm, values, gated, total;
};

int64_t round_up8(int64_t n) { return (n + 7) & ~static_cast<int64_t>(7); }

// Row padding for the three transposed projection matrices, in bf16 elements. The
// staging copy writes them transposed, so consecutive threads write down a column: at
// an unpadded stride of DH = 64 bf16 = 128 bytes every one of a warp's 32 lanes lands
// in the same 4-byte bank, and the store serializes 32 ways. Measured: weight staging
// alone cost 10.27 us of the kernel's 32.0 us before this padding. Two elements make
// the row stride 66 bf16 = 132 bytes = 33 four-byte words, so consecutive columns walk
// consecutive banks and the store is conflict-free -- while the *read* pattern in the
// projection stages (consecutive output index, same reduction index) stays contiguous
// and conflict-free either way, which is what the transpose was for.
constexpr int64_t kRowPad = 2;

SmemLayout smem_layout(int64_t CZ, int64_t CM, int64_t H, int64_t DH,
                       int64_t N, int64_t tile) {
  SmemLayout L{};
  int64_t at = 0;
  auto place = [&](int64_t n) { const int64_t off = at; at += round_up8(n); return off; };
  L.gz = place(CZ);
  L.bz = place(CZ);
  L.gm = place(CM);
  L.bm = place(CM);
  L.wz = place(H * CZ);
  L.wv = place(CM * (DH + kRowPad));
  L.wg = place(CM * (DH + kRowPad));
  L.wo = place(DH * (CM + kRowPad));
  // The logits are written by the pair path and overwritten in place by the softmax:
  // one thread owns a whole (head, i) row, so the row passes through registers and
  // the probabilities can land on the logits' storage. Both are bf16 and the same
  // extent, which halves this term -- the one that grows as O(H*N^2).
  L.logits = place(H * tile * N);
  L.mnorm = place(N * CM);
  L.values = place(N * DH);
  L.gated = place(tile * DH);
  L.total = at;
  return L;
}

// The launch shape a resolved plan implies. Both make_plan and the launcher derive
// their geometry from here, so a route can never be planned with a grid the launch
// cannot express.
bool launch_grid(const Plan& p, dim3* grid) {
  int64_t rows = 0;
  if (!mul_within(p.B, p.S, std::numeric_limits<int32_t>::max(), &rows)) return false;
  if (rows <= 0 || p.ntiles <= 0) return false;
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return false;
  const auto* props = at::cuda::getDeviceProperties(device);
  if (rows > props->maxGridSize[0] || p.ntiles > props->maxGridSize[1]) return false;
  *grid = dim3(static_cast<unsigned>(rows), static_cast<unsigned>(p.ntiles), 1);
  return true;
}

// ---------------------------------------------------------------------------
// The kernel
// ---------------------------------------------------------------------------

using bf16 = __nv_bfloat16;
using bf162 = __nv_bfloat162;

// Every conversion goes through float explicitly. Packed bf16 *loads* are free, but
// packed bf16 *arithmetic* accumulates at input precision, and every accumulation
// here has to happen in fp32 -- so the packed values are split into a float2 the
// moment they are used, never added as a pair.
__device__ __forceinline__ float2 to_f2(bf162 v) { return __bfloat1622float2(v); }
__device__ __forceinline__ bf16 to_bf(float v) { return __float2bfloat16_rn(v); }
__device__ __forceinline__ float from_bf(bf16 v) { return __bfloat162float(v); }

// Site 1 and site 2, in that order and with both roundings: [inf * [mask - 1]].
// Not a select and not a branch -- see the module docstring on inf=float('inf').
__device__ __forceinline__ bf16 pair_bias(float mask_value, float inf) {
  const bf16 s1 = to_bf(mask_value - 1.0f);           // site 1
  return to_bf(inf * from_bf(s1));                    // site 2
}

// ATen's sigmoid and softmax both widen to fp32, evaluate there, and narrow once.
// expf and the reciprocal spelling below are the ones libdevice gives ATen; nothing
// here is built with --use_fast_math, so __expf is not substituted.
__device__ __forceinline__ float sigmoid_f(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// One block per (batch, sequence) row; blockIdx.y selects the tile of query rows i
// this block owns. The channel geometry is a template parameter -- the pair-path row
// only stays in registers if its loop bounds are compile-time constants -- while B, S
// and N are runtime, every stage striding its work across all kThreads.
template <int CZ, int CM, int H, int CH>
__global__ __launch_bounds__(kThreads) void msa_pair_weighted_avg_kernel(
    const bf16* __restrict__ m, const bf16* __restrict__ z,
    const bf16* __restrict__ mask,
    const bf16* __restrict__ gz, const bf16* __restrict__ bz,
    const bf16* __restrict__ gm, const bf16* __restrict__ bm,
    const bf16* __restrict__ wz, const bf16* __restrict__ wv,
    const bf16* __restrict__ wg, const bf16* __restrict__ wo,
    bf16* __restrict__ out,
    int S, int N, int tile, float inf,
    int off_gz, int off_bz, int off_gm, int off_bm, int off_wz, int off_wv,
    int off_wg, int off_wo, int off_logits, int off_mnorm, int off_values,
    int off_gated) {
  constexpr int DH = H * CH;
  constexpr int kRowPad = 2;           // mirrors kRowPad on the host side
  constexpr int VEC = 8;               // bf16 per 16-byte access
  constexpr int ZV = CZ / VEC;         // 16-byte accesses per pair row
  static_assert(CZ % VEC == 0, "the pair row is read as whole 16-byte vectors");
  static_assert(CM % kWarpSize == 0, "one warp covers an msa row in equal slices");

  extern __shared__ bf16 smem[];
  bf16* s_gz = smem + off_gz;
  bf16* s_bz = smem + off_bz;
  bf16* s_gm = smem + off_gm;
  bf16* s_bm = smem + off_bm;
  bf16* s_wz = smem + off_wz;          // [H][CZ], row-major: the head dots broadcast
  // Reduction-major after transposing, with padded rows so the transposing store in
  // the staging loop is conflict-free (see kRowPad on the host side).
  constexpr int DHP = DH + kRowPad;
  constexpr int CMP = CM + kRowPad;
  bf16* s_wv = smem + off_wv;          // [CM][DHP]
  bf16* s_wg = smem + off_wg;          // [CM][DHP]
  bf16* s_wo = smem + off_wo;          // [DH][CMP]
  bf16* s_logits = smem + off_logits;  // [H][tile][N], reused in place as weights
  bf16* s_mnorm = smem + off_mnorm;    // [N][CM]
  bf16* s_values = smem + off_values;  // [N][DH]
  bf16* s_gated = smem + off_gated;    // [tile][DH]

  const int tid = threadIdx.x;
  const int row = blockIdx.x;          // the (batch, sequence) pair
  const int b = row / S;
  const int s = row - b * S;
  const int i0 = blockIdx.y * tile;
  const int rows_here = min(tile, N - i0);
  if (rows_here <= 0) return;

  // --- staging ------------------------------------------------------------
  // The three projection matrices are transposed on the way in: the global read stays
  // coalesced and the shared store takes the stride, which is the cheaper half of the
  // trade only because the destination rows are padded. Unpadded, this loop was the
  // single most expensive thing in the kernel -- 10.27 us of 32.0 us, all of it a
  // 32-way bank conflict on the store.
  for (int i = tid; i < CZ; i += kThreads) { s_gz[i] = gz[i]; s_bz[i] = bz[i]; }
  for (int i = tid; i < CM; i += kThreads) { s_gm[i] = gm[i]; s_bm[i] = bm[i]; }
  for (int i = tid; i < H * CZ; i += kThreads) s_wz[i] = wz[i];
  for (int i = tid; i < DH * CM; i += kThreads) {
    const int d = i / CM, c = i - d * CM;
    s_wv[c * DHP + d] = wv[i];
    s_wg[c * DHP + d] = wg[i];
  }
  for (int i = tid; i < CM * DH; i += kThreads) {
    const int e = i / DH, d = i - e * DH;
    s_wo[d * CMP + e] = wo[i];
  }
  __syncthreads();

  // --- pair path: layer_norm_z fused with the head projection -------------
  // One thread per (i, j) pair. The whole CZ-channel row is loaded once as ZV
  // 16-byte vectors and kept in registers across the mean pass and the variance
  // pass, then overwritten in place with the bf16-rounded normalised value -- which
  // is what makes site 3 free rather than a second trip through memory.
  {
    const int pairs = rows_here * N;
    for (int p = tid; p < pairs; p += kThreads) {
      const int li = p / N;
      const int j = p - li * N;
      const int i = i0 + li;

      uint4 packed[ZV];
      const uint4* src = reinterpret_cast<const uint4*>(
          z + ((static_cast<long long>(b) * N + i) * N + j) * CZ);
#pragma unroll
      for (int v = 0; v < ZV; ++v) packed[v] = src[v];
      bf162* pair = reinterpret_cast<bf162*>(packed);

      float sum = 0.0f;
#pragma unroll
      for (int k = 0; k < CZ / 2; ++k) {
        const float2 f = to_f2(pair[k]);
        sum += f.x + f.y;
      }
      const float mean = sum / static_cast<float>(CZ);
      float sq = 0.0f;
#pragma unroll
      for (int k = 0; k < CZ / 2; ++k) {
        const float2 f = to_f2(pair[k]);
        const float a = f.x - mean, c = f.y - mean;
        sq += a * a + c * c;
      }
      // Biased (population) variance, eps inside the reciprocal square root, and
      // rsqrtf rather than 1/sqrtf -- the spelling ATen's CUDA LayerNorm uses. This
      // kernel makes two register passes where ATen reduces with online Welford, so
      // the two agree to fp32 rounding but are not bit-identical; the bf16 store
      // below absorbs that, and no ULP claim is made anywhere.
      const float rstd = rsqrtf(sq / static_cast<float>(CZ) + kLayerNormEps);
#pragma unroll
      for (int k = 0; k < CZ / 2; ++k) {
        const float2 f = to_f2(pair[k]);
        const float2 gv = to_f2(reinterpret_cast<const bf162*>(s_gz)[k]);
        const float2 bv = to_f2(reinterpret_cast<const bf162*>(s_bz)[k]);
        pair[k] = __floats2bfloat162_rn((f.x - mean) * rstd * gv.x + bv.x,
                                        (f.y - mean) * rstd * gv.y + bv.y);
      }

      const float mask_value = mask == nullptr
          ? 1.0f
          : from_bf(mask[(static_cast<long long>(b) * N + i) * N + j]);
      const bf16 bias = pair_bias(mask_value, inf);

#pragma unroll
      for (int h = 0; h < H; ++h) {
        const bf162* wrow = reinterpret_cast<const bf162*>(s_wz + h * CZ);
        float acc = 0.0f;
#pragma unroll
        for (int k = 0; k < CZ / 2; ++k) {
          const float2 x = to_f2(pair[k]);
          const float2 w = to_f2(wrow[k]);      // warp-uniform: a broadcast, not 32 reads
          acc += x.x * w.x + x.y * w.y;
        }
        const bf16 logit_raw = to_bf(acc);                              // site 4
        const bf16 logit = to_bf(from_bf(logit_raw) + from_bf(bias));    // site 5
        s_logits[(h * tile + li) * N + j] = logit;
      }
    }
  }

  // --- msa path: layer_norm_m, one warp per row ---------------------------
  // Independent of the pair path, so it shares the same phase and one barrier. Every
  // row 0..N-1 is normalised, not just this block's tile: the value projection below
  // reduces over all of them.
  {
    constexpr int VPL = CM / kWarpSize;         // channels per lane
    const int warp = tid / kWarpSize;
    const int lane = tid % kWarpSize;
    constexpr int kWarps = kThreads / kWarpSize;
    for (int i = warp; i < N; i += kWarps) {
      const bf16* mrow = m + ((static_cast<long long>(b) * S + s) * N + i) * CM;
      float x[VPL];
      float sum = 0.0f;
#pragma unroll
      for (int t = 0; t < VPL; ++t) {
        x[t] = from_bf(mrow[t * kWarpSize + lane]);
        sum += x[t];
      }
#pragma unroll
      for (int off = kWarpSize / 2; off > 0; off >>= 1)
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
      const float mean = sum / static_cast<float>(CM);
      float sq = 0.0f;
#pragma unroll
      for (int t = 0; t < VPL; ++t) { const float d = x[t] - mean; sq += d * d; }
#pragma unroll
      for (int off = kWarpSize / 2; off > 0; off >>= 1)
        sq += __shfl_xor_sync(0xffffffffu, sq, off);
      const float rstd = rsqrtf(sq / static_cast<float>(CM) + kLayerNormEps);
#pragma unroll
      for (int t = 0; t < VPL; ++t) {
        const int c = t * kWarpSize + lane;
        s_mnorm[i * CM + c] =
            to_bf((x[t] - mean) * rstd * from_bf(s_gm[c]) + from_bf(s_bm[c]));  // site 7
      }
    }
  }
  __syncthreads();

  // --- pair weights: softmax over j --------------------------------------
  // One thread per (head, i) row, serially over j: no cross-lane communication and no
  // constraint that N divide the warp, which a shuffle butterfly would impose. The
  // exponentials are recomputed in the third pass rather than held in a register
  // array, because a runtime N would put that array in local memory; expf is
  // deterministic, so recomputing is numerically identical to keeping them.
  {
    const int nrows = H * rows_here;
    for (int r = tid; r < nrows; r += kThreads) {
      const int h = r / rows_here;
      const int li = r - h * rows_here;
      bf16* lg = s_logits + (h * tile + li) * N;
      float mx = -INFINITY;
      for (int j = 0; j < N; ++j) mx = fmaxf(mx, from_bf(lg[j]));
      float denom = 0.0f;
      for (int j = 0; j < N; ++j) denom += expf(from_bf(lg[j]) - mx);
      for (int j = 0; j < N; ++j)
        lg[j] = to_bf(expf(from_bf(lg[j]) - mx) / denom);                // site 6
    }
  }

  // --- values: V = m_hat . linear_v^T ------------------------------------
  {
    const int total = N * DH;
    for (int x = tid; x < total; x += kThreads) {
      const int j = x / DH;
      const int d = x - j * DH;
      const bf162* mn = reinterpret_cast<const bf162*>(s_mnorm + j * CM);
      float acc = 0.0f;
#pragma unroll
      for (int k = 0; k < CM / 2; ++k) {
        const float2 v = to_f2(mn[k]);
        acc += v.x * from_bf(s_wv[(2 * k) * DHP + d]);
        acc += v.y * from_bf(s_wv[(2 * k + 1) * DHP + d]);
      }
      s_values[x] = to_bf(acc);                                          // site 8
    }
  }
  __syncthreads();

  // --- weighted average and gate ------------------------------------------
  {
    const int total = rows_here * DH;
    for (int x = tid; x < total; x += kThreads) {
      const int li = x / DH;
      const int d = x - li * DH;
      const int h = d / CH;               // the flat head index: d == h * CH + c
      const bf16* w = s_logits + (h * tile + li) * N;
      float acc = 0.0f;
      for (int j = 0; j < N; ++j) acc += from_bf(w[j]) * from_bf(s_values[j * DH + d]);
      const bf16 avg = to_bf(acc);                                       // site 11

      const bf162* mn = reinterpret_cast<const bf162*>(s_mnorm + (i0 + li) * CM);
      float gacc = 0.0f;
#pragma unroll
      for (int k = 0; k < CM / 2; ++k) {
        const float2 v = to_f2(mn[k]);
        gacc += v.x * from_bf(s_wg[(2 * k) * DHP + d]);
        gacc += v.y * from_bf(s_wg[(2 * k + 1) * DHP + d]);
      }
      const bf16 gate_raw = to_bf(gacc);                                 // site 9
      const bf16 gate = to_bf(sigmoid_f(from_bf(gate_raw)));             // site 10

      s_gated[x] = to_bf(from_bf(avg) * from_bf(gate));                  // site 12
    }
  }
  __syncthreads();

  // --- output projection --------------------------------------------------
  {
    const int total = rows_here * CM;
    for (int x = tid; x < total; x += kThreads) {
      const int li = x / CM;
      const int e = x - li * CM;
      const bf16* og = s_gated + li * DH;
      float acc = 0.0f;
#pragma unroll
      for (int d = 0; d < DH; ++d) acc += from_bf(og[d]) * from_bf(s_wo[d * CMP + e]);
      out[((static_cast<long long>(b) * S + s) * N + i0 + li) * CM + e] =
          to_bf(acc);                                                    // site 13
    }
  }
}

// ---------------------------------------------------------------------------
// Instantiation, planning and launch
// ---------------------------------------------------------------------------

// The captured configuration is the only channel geometry instantiated. Templating is
// not a preference here: the pair-path row only stays in registers if CZ is a
// compile-time constant, so every geometry costs a separate instantiation and a
// separate cold-compile pass. Any other geometry declines to the composition, which is
// correct at every geometry and composed of frozen winners.
constexpr int kCZ = 128;
constexpr int kCM = 64;
constexpr int kH = 8;
constexpr int kCH = 8;

using KernelFn = void (*)(const bf16*, const bf16*, const bf16*, const bf16*,
                          const bf16*, const bf16*, const bf16*, const bf16*,
                          const bf16*, const bf16*, const bf16*, bf16*,
                          int, int, int, float, int, int, int, int, int, int, int,
                          int, int, int, int, int);

KernelFn kernel_for(int64_t CZ, int64_t CM, int64_t H, int64_t CH) {
  if (CZ == kCZ && CM == kCM && H == kH && CH == kCH) {
    return &msa_pair_weighted_avg_kernel<kCZ, kCM, kH, kCH>;
  }
  return nullptr;
}

// Whether this configuration can actually be resident, which covers the register
// budget as well as shared memory. Memoized on everything that can change the answer:
// the occupancy figure is a property of the device as much as of the kernel, so a
// module used on GPU 0 and then GPU 1 must not reuse GPU 0's verdict.
struct LaunchKey {
  int device;
  const void* fn;
  int threads;
  size_t smem;
  bool operator<(const LaunchKey& o) const {
    if (device != o.device) return device < o.device;
    if (fn != o.fn) return fn < o.fn;
    if (threads != o.threads) return threads < o.threads;
    return smem < o.smem;
  }
};

// Returns false only for a genuine capability rejection (a successful query reporting
// no resident blocks). A failed query is a real CUDA error and is raised, not quietly
// turned into a fallback: swallowing it would hide a broken context behind a
// performance regression.
bool launchable(const void* fn, int threads, size_t smem) {
  int device = 0;
  AT_CUDA_CHECK(cudaGetDevice(&device));
  static std::mutex mu;
  static std::map<LaunchKey, bool> cache;
  const LaunchKey key{device, fn, threads, smem};
  std::lock_guard<std::mutex> lock(mu);
  const auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  int per_sm = 0;
  AT_CUDA_CHECK(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, fn, threads, smem));
  const bool ok = per_sm > 0;
  cache.emplace(key, ok);
  return ok;
}


// The guarded phase. Every tensor has already passed `prescreen`, so ordinary metadata
// accessors are safe here and the caller has installed a CUDAGuard on the input's device --
// which matters because this function queries the device's grid limits and asks
// cudaOccupancyMaxActiveBlocksPerMultiprocessor whether the configuration can be resident,
// and both answer for whichever device is current.
Plan make_plan(const at::Tensor& m, const at::Tensor& z, const at::Tensor& mask,
               const Weights& w, int64_t tile_request) {
  Plan p{};
  auto decline = [&](int64_t why) { p.route = kRouteFallback; p.reject = why; return p; };

  // Re-run the pre-guard phase rather than trusting the caller to have done it: it is
  // cheap, it keeps this function correct in isolation, and a predicate whose safety
  // depends on call order is a predicate waiting to be called out of order.
  if (const int64_t why = prescreen(m, z, mask, w.t)) return decline(why);
  const bool has_mask = mask.defined();

  // Per-tensor screens, inputs and weights alike -- a Parameter can legally be a named,
  // non-contiguous, negated view, so none of these can be skipped for weights.
  if (const int64_t why = tensor_reject(m, 4)) return decline(why);
  if (const int64_t why = tensor_reject(z, 4)) return decline(why);
  if (has_mask) {
    if (const int64_t why = tensor_reject(mask, 3)) return decline(why);
  }
  const int64_t weight_rank[8] = {1, 1, 1, 1, 2, 2, 2, 2};
  for (int i = 0; i < 8; ++i) {
    if (const int64_t why = tensor_reject(*w.t[i], weight_rank[i])) return decline(why);
  }

  // Cross-tensor agreement, once every tensor is known to be an ordinary dense CUDA
  // bf16 tensor. Same device index, not merely both CUDA.
  const auto dev = m.device();
  if (z.device() != dev) return decline(kRejectDevice);
  if (has_mask && mask.device() != dev) return decline(kRejectDevice);
  for (int i = 0; i < 8; ++i) {
    if (w.t[i]->device() != dev) return decline(kRejectDevice);
  }

  p.B = m.size(0); p.S = m.size(1); p.N = m.size(2); p.CM = m.size(3);
  p.CZ = z.size(3);
  p.H = w.t[4]->size(0);
  if (p.H <= 0) return decline(kRejectShape);
  p.DH = w.t[5]->size(0);
  if (p.DH % p.H != 0) return decline(kRejectShape);
  p.CH = p.DH / p.H;

  // The baseline reads n_res from z, so z is what defines it; m and mask must agree or
  // the einsum would have failed.
  if (z.size(0) != p.B || z.size(1) != p.N || z.size(2) != p.N) return decline(kRejectShape);
  if (has_mask && (mask.size(0) != p.B || mask.size(1) != p.N || mask.size(2) != p.N))
    return decline(kRejectShape);
  if (w.t[0]->size(0) != p.CZ || w.t[1]->size(0) != p.CZ) return decline(kRejectShape);
  if (w.t[2]->size(0) != p.CM || w.t[3]->size(0) != p.CM) return decline(kRejectShape);
  if (w.t[4]->size(1) != p.CZ) return decline(kRejectShape);
  if (w.t[5]->size(1) != p.CM || w.t[6]->size(0) != p.DH || w.t[6]->size(1) != p.CM)
    return decline(kRejectShape);
  if (w.t[7]->size(0) != p.CM || w.t[7]->size(1) != p.DH) return decline(kRejectShape);

  if (!standard_strides(m) || !standard_strides(z)) return decline(kRejectStride);
  if (has_mask && !standard_strides(mask)) return decline(kRejectStride);
  for (int i = 0; i < 8; ++i) {
    if (!standard_strides(*w.t[i])) return decline(kRejectStride);
  }

  // Emptiness, after the shapes and strides so that an empty tensor with a wrong rank or a
  // wrong extent reports the shape rather than the emptiness.
  if (const int64_t why = empty_reject(m)) return decline(why);
  if (const int64_t why = empty_reject(z)) return decline(why);
  if (has_mask) {
    if (const int64_t why = empty_reject(mask)) return decline(why);
  }
  for (int i = 0; i < 8; ++i) {
    if (const int64_t why = empty_reject(*w.t[i])) return decline(why);
  }

  // Storage pointers, after emptiness: every tensor is now known to have elements, so a null
  // pointer is a wrapper failure rather than the ordinary consequence of being empty.
  if (const int64_t why = pointer_reject(m)) return decline(why);
  if (const int64_t why = pointer_reject(z)) return decline(why);
  if (has_mask) {
    if (const int64_t why = pointer_reject(mask)) return decline(why);
  }
  for (int i = 0; i < 8; ++i) {
    if (const int64_t why = pointer_reject(*w.t[i])) return decline(why);
  }

  const KernelFn fn = kernel_for(p.CZ, p.CM, p.H, p.CH);
  if (fn == nullptr) return decline(kRejectGeometry);

  // The extension is compiled for exactly one architecture -- the one the importing
  // process saw -- because six nvcc passes for five targets that will never run this
  // kernel is the difference between a comfortable cold build and the harness's 600 s
  // watchdog. On a machine with mixed GPUs a tensor on a different-capability device
  // would otherwise be admitted here and fail at launch, so the capability is part of
  // the predicate rather than an assumption.
  {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return decline(kRejectLaunch);
    const auto* props = at::cuda::getDeviceProperties(device);
    if (props == nullptr) return decline(kRejectLaunch);
    if (props->major != FK_MSA_ARCH_MAJOR || props->minor != FK_MSA_ARCH_MINOR) {
      return decline(kRejectDevice);
    }
  }

  // Every extent the kernel narrows to int, bounded before the narrowing rather than
  // after: an expanded view can carry a logical extent far past the grid limits on a
  // few kilobytes of storage.
  constexpr int64_t kI32 = std::numeric_limits<int32_t>::max();
  if (p.B > kI32 || p.S > kI32 || p.N > kI32) return decline(kRejectBounds);
  int64_t scratch = 0;
  if (!mul_within(p.B, p.S, kI32, &scratch)) return decline(kRejectBounds);
  if (!mul_within(p.N, p.CM, kI32, &scratch)) return decline(kRejectBounds);
  if (!mul_within(p.N, p.DH, kI32, &scratch)) return decline(kRejectBounds);
  if (!mul_within(p.H, p.N, kI32, &scratch)) return decline(kRejectBounds);

  // Query-row tiling: grid = (B*S, ceil(N/tile)), each block owning `tile` query rows.
  // A smaller tile raises the block count and cuts each block's share of the pair path,
  // at the cost of replaying the weight staging and the value projection in every tile.
  // Swept in one process on one GPU (profile/msa_fused_v6_pad_ab), at the captured shape:
  //
  //     tile   16      8      4      2      1
  //   blocks    8     16     32     64    128
  //   kernel 49.18  43.01  40.97  40.98  43.00  us
  //
  // so 4 it is, worth ~17% against serving the whole query axis in one block. (Those
  // absolute numbers are from a slower-clocked device than the one the bench ran on --
  // only their ratios are meaningful, and comparing across processes is exactly the
  // mistake that made this measurement take three attempts.)
  //
  // A consequence worth stating because it moves a documented boundary: with the tile
  // fixed at 4 the shared-memory requirement is O(H*tile*N) rather than O(H*N^2), i.e.
  // linear in N, so the fast path now covers every N up to 64 instead of stopping around
  // 24. That is the generality the design wanted; it is a side effect of a change made
  // for speed, not an independent decision.
  constexpr int64_t kDefaultTile = 4;
  p.tile = tile_request > 0 ? std::min<int64_t>(tile_request, p.N)
                           : std::min<int64_t>(kDefaultTile, p.N);
  if (p.tile <= 0) return decline(kRejectBounds);
  p.ntiles = (p.N + p.tile - 1) / p.tile;
  if (!mul_within(p.H, p.tile * p.N, kI32, &scratch)) return decline(kRejectBounds);

  const SmemLayout L = smem_layout(p.CZ, p.CM, p.H, p.DH, p.N, p.tile);
  p.smem = L.total * static_cast<int64_t>(sizeof(bf16));
  if (p.smem > kSmemBudget) return decline(kRejectSmem);
  p.threads = kThreads;

  dim3 grid;
  Plan probe = p;
  probe.route = kRouteFused;
  if (!launch_grid(probe, &grid)) return decline(kRejectBounds);

  // The input pointers, including the weights, must all permit the 16-byte accesses
  // the pair-path row load uses and the 4-byte accesses the packed shared reads use.
  const void* ptrs[11] = {
      m.const_data_ptr(), z.const_data_ptr(),
      has_mask ? mask.const_data_ptr() : m.const_data_ptr(),
      w.t[0]->const_data_ptr(), w.t[1]->const_data_ptr(), w.t[2]->const_data_ptr(),
      w.t[3]->const_data_ptr(), w.t[4]->const_data_ptr(), w.t[5]->const_data_ptr(),
      w.t[6]->const_data_ptr(), w.t[7]->const_data_ptr()};
  for (const void* ptr : ptrs) {
    if (!aligned16(ptr)) return decline(kRejectAlign);
  }

  if (!launchable(reinterpret_cast<const void*>(fn), kThreads,
                  static_cast<size_t>(p.smem))) {
    return decline(kRejectLaunch);
  }

  p.route = kRouteFused;
  p.reject = kRejectNone;
  return p;
}

at::Tensor run_fused(const Plan& p, const at::Tensor& m, const at::Tensor& z,
                     const at::Tensor& mask, const Weights& w, double inf) {
  const KernelFn fn = kernel_for(p.CZ, p.CM, p.H, p.CH);
  TORCH_CHECK(fn != nullptr, "msa: no kernel for this geometry; make_plan should "
                             "have declined");
  auto out = at::empty(m.sizes(), m.options());
  // The output comes from the caching allocator, whose alignment the kernel's
  // correctness must not rest on: re-check the actual allocation rather than assume it.
  if (!aligned16(out.mutable_data_ptr())) return at::Tensor();

  // The same derivation make_plan gated on, so plan and launch cannot disagree.
  dim3 grid;
  TORCH_CHECK(launch_grid(p, &grid),
              "msa: launch geometry outside CUDA's grid bounds; make_plan should have "
              "declined this shape");

  const SmemLayout L = smem_layout(p.CZ, p.CM, p.H, p.DH, p.N, p.tile);
  TORCH_CHECK(L.total * static_cast<int64_t>(sizeof(bf16)) == p.smem,
              "msa: shared-memory size disagrees with the plan");

  const auto* mask_ptr = mask.defined()
      ? reinterpret_cast<const bf16*>(mask.const_data_ptr()) : nullptr;
  auto* stream = at::cuda::getCurrentCUDAStream().stream();
  fn<<<grid, static_cast<unsigned>(p.threads), static_cast<size_t>(p.smem), stream>>>(
      reinterpret_cast<const bf16*>(m.const_data_ptr()),
      reinterpret_cast<const bf16*>(z.const_data_ptr()),
      mask_ptr,
      reinterpret_cast<const bf16*>(w.t[0]->const_data_ptr()),
      reinterpret_cast<const bf16*>(w.t[1]->const_data_ptr()),
      reinterpret_cast<const bf16*>(w.t[2]->const_data_ptr()),
      reinterpret_cast<const bf16*>(w.t[3]->const_data_ptr()),
      reinterpret_cast<const bf16*>(w.t[4]->const_data_ptr()),
      reinterpret_cast<const bf16*>(w.t[5]->const_data_ptr()),
      reinterpret_cast<const bf16*>(w.t[6]->const_data_ptr()),
      reinterpret_cast<const bf16*>(w.t[7]->const_data_ptr()),
      reinterpret_cast<bf16*>(out.mutable_data_ptr()),
      static_cast<int>(p.S), static_cast<int>(p.N), static_cast<int>(p.tile),
      static_cast<float>(inf),
      static_cast<int>(L.gz), static_cast<int>(L.bz), static_cast<int>(L.gm),
      static_cast<int>(L.bm), static_cast<int>(L.wz), static_cast<int>(L.wv),
      static_cast<int>(L.wg), static_cast<int>(L.wo), static_cast<int>(L.logits),
      static_cast<int>(L.mnorm), static_cast<int>(L.values), static_cast<int>(L.gated));
  AT_CUDA_CHECK(cudaGetLastError());
  return out;
}

at::Tensor opt_or_undef(const std::optional<at::Tensor>& t) {
  return t.has_value() ? *t : at::Tensor();
}

}  // namespace

// Returns an undefined tensor -- None through pybind -- for anything the predicate
// does not claim. That is a capability answer, not an error: the caller reproduces the
// baseline formula instead. Nothing here raises for an input the composition accepts.
at::Tensor msa_pair_weighted_avg(
    const at::Tensor& m, const at::Tensor& z, const std::optional<at::Tensor>& mask,
    const std::optional<at::Tensor>& gz, const std::optional<at::Tensor>& bz,
    const std::optional<at::Tensor>& gm, const std::optional<at::Tensor>& bm,
    const std::optional<at::Tensor>& wz, const std::optional<at::Tensor>& wv,
    const std::optional<at::Tensor>& wg, const std::optional<at::Tensor>& wo,
    double inf, int64_t tile) {
  const at::Tensor mk = opt_or_undef(mask);
  const at::Tensor tz[8] = {opt_or_undef(gz), opt_or_undef(bz), opt_or_undef(gm),
                            opt_or_undef(bm), opt_or_undef(wz), opt_or_undef(wv),
                            opt_or_undef(wg), opt_or_undef(wo)};
  Weights w{};
  for (int i = 0; i < 8; ++i) w.t[i] = &tz[i];

  // Two phases, in this order and for two different reasons.
  //
  // The pre-guard phase screens ambient dispatch state and then every present tensor for
  // wrapper-ness, and it must come first because a wrapped tensor can throw from the very
  // accessors everything after it uses -- `m.is_cuda()` included. Reading the device to
  // decide whether to guard, before that screen, defeats the screen.
  //
  // Only once `m` is known to be an ordinary tensor with initialised storage is its device
  // safe to read, and only then is the guard installed -- before `make_plan`, not merely
  // before the launch, because the planner queries the device's grid limits and asks
  // cudaOccupancyMaxActiveBlocksPerMultiprocessor whether the configuration can be
  // resident, and both answer for whichever device is current. On a non-current device
  // those were answers about the wrong GPU, recorded in the memo under the wrong key.
  if (prescreen(m, z, mk, w.t) != kRejectNone || !m.is_cuda()) {
    g_declined_calls.fetch_add(1, std::memory_order_relaxed);
    return at::Tensor();
  }
  const c10::cuda::CUDAGuard guard(m.device());
  const Plan p = make_plan(m, z, mk, w, tile);
  if (p.route != kRouteFused) {
    g_declined_calls.fetch_add(1, std::memory_order_relaxed);
    return at::Tensor();
  }
  at::Tensor out = run_fused(p, m, z, mk, w, inf);
  if (!out.defined()) {
    g_declined_calls.fetch_add(1, std::memory_order_relaxed);
    return out;
  }
  g_fused_calls.fetch_add(1, std::memory_order_relaxed);
  return out;
}

// Reports the routing decision without running it, so a test can assert that a given
// input really is served by the kernel it is credited to. Raises when the extension is
// unavailable rather than answering, so a build failure cannot be misread as a route.
// Separate from the hot entry point on purpose: that one carries no flag and no query.
at::Tensor msa_plan(
    const at::Tensor& m, const at::Tensor& z, const std::optional<at::Tensor>& mask,
    const std::optional<at::Tensor>& gz, const std::optional<at::Tensor>& bz,
    const std::optional<at::Tensor>& gm, const std::optional<at::Tensor>& bm,
    const std::optional<at::Tensor>& wz, const std::optional<at::Tensor>& wv,
    const std::optional<at::Tensor>& wg, const std::optional<at::Tensor>& wo,
    int64_t tile) {
  const at::Tensor mk = opt_or_undef(mask);
  const at::Tensor tz[8] = {opt_or_undef(gz), opt_or_undef(bz), opt_or_undef(gm),
                            opt_or_undef(bm), opt_or_undef(wz), opt_or_undef(wv),
                            opt_or_undef(wg), opt_or_undef(wo)};
  Weights w{};
  for (int i = 0; i < 8; ++i) w.t[i] = &tz[i];
  // Same two phases as the hot path, so the answer describes the route the forward would
  // actually take, on the device it would actually take it on. The device is read only
  // after the pre-guard phase, for the same throw-safety reason.
  std::optional<c10::cuda::CUDAGuard> guard;
  if (prescreen(m, z, mk, w.t) == kRejectNone && m.is_cuda()) guard.emplace(m.device());
  const Plan p = make_plan(m, z, mk, w, tile);
  auto t = at::zeros({14}, at::TensorOptions().dtype(at::kLong));
  auto a = t.accessor<int64_t, 1>();
  a[0] = p.route;  a[1] = p.reject; a[2] = p.B;     a[3] = p.S;
  a[4] = p.N;      a[5] = p.CZ;     a[6] = p.CM;    a[7] = p.H;
  a[8] = p.CH;     a[9] = p.DH;     a[10] = p.tile; a[11] = p.ntiles;
  a[12] = p.threads; a[13] = p.smem;
  return t;
}

int64_t fused_calls() {
  return static_cast<int64_t>(g_fused_calls.load(std::memory_order_relaxed));
}

int64_t declined_calls() {
  return static_cast<int64_t>(g_declined_calls.load(std::memory_order_relaxed));
}

void reset_call_counters() {
  g_fused_calls.store(0, std::memory_order_relaxed);
  g_declined_calls.store(0, std::memory_order_relaxed);
}
"""


def _target_arch() -> str:
    major, minor = torch.cuda.get_device_capability()
    return f"{major}.{minor}"


# The name carries a hash of the source, for two reasons. The harness imports
# ``baseline.py`` first, as the reference, so this file's extension name must differ
# from anything the baseline owns or the two would collide inside one process; and a
# source edit can never silently reuse a stale ``.so`` from the pinned directory.
_SOURCE_DIGEST = hashlib.sha1(
    (_CUDA_SOURCE + _CPP_SOURCE).encode("utf-8")).hexdigest()[:10]


def _load_extension():
    """Build (or reuse) the extension in a workspace-local, single-arch cache.

    Called at import, never from ``forward``. The harness counts threads around its
    timing loop, so a build deferred to the first call would put ninja's workers
    inside the measured region and read as injected threads.
    """
    from torch.utils.cpp_extension import load_inline

    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}.{minor}"
    name = f"fk_l2_msa_pwa_sm{arch.replace('.', '')}_{_SOURCE_DIGEST}"
    # Explicitly workspace-local: sibling operator agents share ``$HOME``, so the
    # default ``~/.cache/torch_extensions/<name>`` would collide with a concurrently
    # running sibling. ``load_inline`` does not create the directory itself.
    build_dir = Path(__file__).resolve().parents[2] / ".torch_extensions" / name
    build_dir.mkdir(parents=True, exist_ok=True)

    # The bench worker is killed after 600 s without output, so a cold compile must not
    # be silent: print one flushed line up front and let ninja stream its progress. A
    # warm cache skips both.
    cold = not (build_dir / f"{name}.so").exists()
    if cold:
        print(f"[alphafold3_msa_attention] building CUDA extension {name!r} for arch "
              f"{arch!r} -- one-time JIT compile, streaming ninja progress ...",
              flush=True)

    # Without narrowing, cpp_extension honours the ambient TORCH_CUDA_ARCH_LIST, which
    # in this environment names six architectures: six nvcc passes for five targets
    # that will never run this kernel. Derived from the live device rather than
    # hardcoded, and restored in a finally so no later build in this process is
    # affected. ``-lineinfo`` is line-table metadata only -- it does not change
    # register allocation -- and Nsight Compute needs it to attribute stalls to source.
    # No ``--use_fast_math``: it would substitute ``__expf`` for ``expf`` and change the
    # sigmoid and softmax away from the accurate implementations the reference uses.
    previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    try:
        return load_inline(
            name=name,
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["msa_pair_weighted_avg", "msa_plan", "fused_calls",
                       "declined_calls", "reset_call_counters"],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo",
                               "--expt-relaxed-constexpr",
                               f"-DFK_MSA_ARCH_MAJOR={major}",
                               f"-DFK_MSA_ARCH_MINOR={minor}"],
            build_directory=str(build_dir),
            verbose=cold,
        )
    finally:
        if previous is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous


if os.environ.get("FK_MSA_DISABLE_EXT"):
    _EXT, _EXT_ERROR = None, "disabled by FK_MSA_DISABLE_EXT"
elif not torch.cuda.is_available():
    _EXT, _EXT_ERROR = None, "no CUDA device"
else:
    try:
        _EXT, _EXT_ERROR = _load_extension(), None
    except Exception as exc:  # noqa: BLE001 - a build that cannot happen must degrade,
        # not take the module down with it: an import error costs every case at once,
        # and the composition below is correct without the extension.
        _EXT, _EXT_ERROR = None, f"{type(exc).__name__}: {exc}"
        print(f"[alphafold3_msa_attention] CUDA extension unavailable ({_EXT_ERROR}); "
              f"every call will take the reference composition",
              file=sys.stderr, flush=True)


def extension_loaded() -> bool:
    """Whether the fused kernel is available in this process."""
    return _EXT is not None


def extension_error() -> str | None:
    """Why the extension is unavailable, or None if it loaded."""
    return _EXT_ERROR


def fused_calls() -> int:
    """Monotonic count of calls the fused kernel served since the last reset."""
    if _EXT is None:
        raise RuntimeError(f"extension not loaded: {_EXT_ERROR}")
    return _EXT.fused_calls()


def declined_calls() -> int:
    """Monotonic count of calls the predicate declined since the last reset."""
    if _EXT is None:
        raise RuntimeError(f"extension not loaded: {_EXT_ERROR}")
    return _EXT.declined_calls()


def reset_call_counters() -> None:
    """Zero both counters, so a test can bracket a single forward."""
    if _EXT is None:
        raise RuntimeError(f"extension not loaded: {_EXT_ERROR}")
    _EXT.reset_call_counters()


def _has_forward_tangent(tensors) -> bool:
    """Whether any of `tensors` carries a forward-mode tangent.

    The predicate declines a dual and routes it here, but the frozen L1 LayerNorm and
    Softmax read primals through raw pointers, so composing them would drop the tangent
    silently. This is what selects the eager spelling below instead.
    """
    for t in tensors:
        if t is None:
            continue
        if fwad.unpack_dual(t).tangent is not None:
            return True
    return False


def _has_lazy_affine(norm: nn.Module) -> bool:
    """Whether either affine parameter carries a lazily applied negative/conjugate bit.

    Such a tensor keeps the untransformed values in storage, and the frozen LayerNorm's
    fused path reads that storage directly -- measured at max_abs 6.28 against the
    materialized tensor on a bare 64-channel LayerNorm.
    """
    for t in (norm.weight, norm.bias):
        if t is not None and (t.is_neg() or t.is_conj()):
            return True
    return False


def _eager_layer_norm(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """`norm(x)` spelled in ATen ops, reproducing the frozen LayerNorm's own formula.

    Mirrors its `promote_fp32` behaviour exactly: reduce and apply the affine in fp32 with
    the parameters upcast, round once on the way out. Used on the two paths where the
    frozen module cannot be trusted -- a lazily negated affine parameter (whose storage it
    would read untransformed) and a forward-mode dual (whose tangent it would drop).
    Materializing the affine bits is free for a tensor that does not carry them.
    """
    weight, bias = norm.weight, norm.bias
    if weight is not None:
        weight = weight.resolve_conj().resolve_neg()
    if bias is not None:
        bias = bias.resolve_conj().resolve_neg()
    if not norm.promote_fp32:
        return F.layer_norm(x, norm.normalized_shape, weight, bias, norm.eps)
    if weight is not None and weight.dtype != torch.float32:
        weight = weight.float()
    if bias is not None and bias.dtype != torch.float32:
        bias = bias.float()
    return F.layer_norm(
        x.float(), norm.normalized_shape, weight, bias, norm.eps).to(x.dtype)


def _reject(*_args, **_kwargs):
    """Stands in for the entry point when the extension is unavailable.

    Bound in place of it so the hot path keeps exactly one shape -- call, then test the
    result for None -- instead of growing a build-status branch.
    """
    return None


class MSARowAttentionWithPairBias(nn.Module):
    """AF3 MSA Pair-Weighted Averaging (Algorithm 10).

    Uses pair activations as weights (softmax over token dim) instead of
    key-query attention.  Parameter names match the checkpoint layout:
    linear_v, linear_g, linear_o (no nested mha).

    Args:
        c_m: MSA input channel dimension
        c_z: Pair embedding channel dimension
        c_hidden: Per-head hidden channel dimension
        no_heads: Number of attention heads
        inf: Large constant for masking
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        inf: float = 1e9,
    ):
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf

        self.layer_norm_m = LayerNorm(c_m)
        self.layer_norm_z = LayerNorm(c_z)
        self.linear_z = Linear(c_z, no_heads, bias=False)

        self.linear_v = Linear(c_m, c_hidden * no_heads, bias=False)
        self.linear_g = Linear(c_m, c_hidden * no_heads, bias=False)
        self.linear_o = Linear(c_hidden * no_heads, c_m, bias=False)

        self.sigmoid = Sigmoid()
        self.softmax = Softmax(dim=-1)

        # Bound once so the call site has no attribute chain into the extension module.
        self._fn = _EXT.msa_pair_weighted_avg if _EXT is not None else _reject
        # Query-row tile; 0 means "let C++ apply the measured default" (4 rows per
        # block). A launch parameter rather than a compile-time constant so it can be
        # swept without restructuring the kernel.
        # A malformed value must not take module construction down with it: this is a
        # private tuning knob, and 0 means "let C++ apply the measured default".
        try:
            self._tile = int(os.environ.get("FK_MSA_TILE", "0"))
        except ValueError:
            self._tile = 0

        # Nothing derived from any weight *value* is precomputed, and no weight tensor
        # is cached. The harness rebinds every ``p.data`` to a bf16 copy, then rewrites
        # the four uninitialized ``Linear`` weights and the two zero LayerNorm biases
        # in place with ``normal_``, then copies shared weights in with
        # ``load_state_dict`` -- all after ``__init__`` and before the first forward. At
        # ``__init__`` time those four weights are ``torch.empty`` garbage.
        #
        # Reading the eight parameters off the module on every call is also the *free*
        # option, not merely the safe one: eighty parameter reads per call (14.7 us of
        # host time) measured 0.00 us of difference in the scored window, and a cached
        # tuple of the eight Parameter objects measured +0.03 us, i.e. nothing. So the
        # option that stays coherent under every mutation -- including replacing a
        # submodule's parameter with a brand-new ``nn.Parameter``, which a cached tuple
        # would miss -- costs nothing to keep. Recorded because the alternative was
        # measured, not assumed.

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            m:    [*, N_seq, N_res, C_m] MSA embedding
            z:    [*, N_res, N_res, C_z] pair embedding
            mask: [*, N_res, N_res] pair mask

        Returns:
            [*, N_seq, N_res, C_m] updated MSA embedding
        """
        if z is None:
            return m
        # One call into the extension, which decides eligibility itself: nothing here
        # queries a dtype, a shape or a contiguity flag, and nothing calls a torch op.
        # ``None`` back means the predicate declined, so the composition runs instead.
        out = self._fn(m, z, mask,
                       self.layer_norm_z.weight, self.layer_norm_z.bias,
                       self.layer_norm_m.weight, self.layer_norm_m.bias,
                       self.linear_z.weight, self.linear_v.weight,
                       self.linear_g.weight, self.linear_o.weight,
                       self.inf, self._tile)
        if out is None:
            return self._reference(m, z, mask)
        return out

    def _reference(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """The baseline formula, for everything the predicate declines.

        Normally spelled through this module's own submodules: the LayerNorms, the
        projections, the sigmoid and the softmax are this workspace's frozen winners, so a
        declined call degrades to already-benchmarked kernels rather than to raw eager
        ATen, and reproduces the baseline *formula*, fp32 promotion included, rather than
        merely landing inside tolerance.

        Two kinds of input are spelled in plain ATen ops instead, because for them the
        frozen submodules are demonstrably wrong and cannot be edited:

        * **a lazily negated or conjugated tensor** -- ``candidate/L1/layer_norm.py``'s
          fused path reads storage through a raw pointer, so it sees the untransformed
          values (measured: max_abs 6.28 against the materialized tensor on a bare
          64-channel LayerNorm). For the *inputs* this is repaired by materializing on the
          way in; for the *affine parameters* it cannot be, because the submodule reads
          ``self.weight`` itself.
        * **a forward-mode dual** -- both ``candidate/L1/layer_norm.py`` (under
          ``no_grad``) and ``candidate/L1/softmax.py`` (in either grad mode) read primals
          through raw pointers and return a tensor with no tangent, so the composition
          silently drops it while ``baseline.py`` preserves it. Materializing cannot fix
          this: a tangent has to flow *through* each op. The eager spelling is the fix.

        Both exceptions are narrow and neither costs an ordinary call anything: the checks
        are a handful of predicates on a path that only runs when the kernel has already
        declined, and every ordinary declined input still composes the frozen winners.
        """
        m = m.resolve_conj().resolve_neg()
        z = z.resolve_conj().resolve_neg()
        if mask is None:
            mask = z.new_ones(z.shape[:-1])
        else:
            mask = mask.resolve_conj().resolve_neg()

        norm_m, norm_z = self.layer_norm_m, self.layer_norm_z
        wz, wv = self.linear_z.weight, self.linear_v.weight
        wg, wo = self.linear_g.weight, self.linear_o.weight

        # A tangent anywhere -- on an input or on any of the eight live parameters -- forces
        # the eager spelling. A lazily negated affine parameter forces the eager LayerNorm
        # for the same reason, and taking the whole eager path for it too keeps one branch
        # instead of two.
        eager = (
            _has_forward_tangent((m, z, mask, norm_z.weight, norm_z.bias,
                                  norm_m.weight, norm_m.bias, wz, wv, wg, wo))
            or _has_lazy_affine(norm_m)
            or _has_lazy_affine(norm_z)
        )

        # Pair bias: [*, 1, no_heads, N_res, N_res]
        mask_bias = (self.inf * (mask - 1))[..., None, None, :, :]
        z_norm = _eager_layer_norm(norm_z, z) if eager else norm_z(z)
        z_proj = F.linear(z_norm, wz) if eager else self.linear_z(z_norm)
        z_weights = _permute_final_dims(z_proj, (2, 0, 1)).unsqueeze(-4)
        z_weights = z_weights + mask_bias
        z_weights = F.softmax(z_weights, dim=-1) if eager else self.softmax(z_weights)

        m = _eager_layer_norm(norm_m, m) if eager else norm_m(m)

        v = F.linear(m, wv) if eager else self.linear_v(m)
        v = v.view(v.shape[:-1] + (self.no_heads, -1))
        v = v.transpose(-2, -3)  # [*, N_seq, H, N_res, C_hidden]

        o = torch.einsum("...hqk,...hkc->...qhc", z_weights, v)

        pre_gate = F.linear(m, wg) if eager else self.linear_g(m)
        g = torch.sigmoid(pre_gate) if eager else self.sigmoid(pre_gate)
        g = g.view(g.shape[:-1] + (self.no_heads, -1))

        o = o * g

        o = o.reshape(o.shape[:-2] + (-1,))
        return F.linear(o, wo) if eager else self.linear_o(o)

    def plan_for(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, int]:
        """The route ``forward`` would take for these inputs, without running it.

        ``route`` is ``ROUTE_FUSED`` or ``ROUTE_FALLBACK``; on a fallback ``reject``
        names the screen that fired (see ``REJECT_NAMES``). Raises if the extension is
        not loaded, so a caller cannot mistake a build failure for a routing answer.
        Deliberately not part of the hot entry point, which gains no flag and no query.
        """
        if _EXT is None:
            raise RuntimeError(f"extension not loaded: {_EXT_ERROR}")
        values = _EXT.msa_plan(
            m, z, mask,
            self.layer_norm_z.weight, self.layer_norm_z.bias,
            self.layer_norm_m.weight, self.layer_norm_m.bias,
            self.linear_z.weight, self.linear_v.weight,
            self.linear_g.weight, self.linear_o.weight,
            self._tile).tolist()
        return dict(zip(PLAN_FIELDS, values))
