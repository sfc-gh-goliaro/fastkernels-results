"""Vision patch embedding for Qwen VL models -- B200 / sm_100.

Same contract as the baseline: the Conv3d patch-embedding weight is reshaped into a linear
projection and applied with one GEMM. For the single captured `__init__` recipe
(`patch_size=16`, `temporal_patch_size=2`, `in_channels=3`, `embed_dim=1152`, `bias=True`)
`input_size` is 1536 and the operator is exactly

    C[M, 1152] = A[M, 1536] . B[1152, 1536]^T + bias[1152]

bf16 in, bf16 out, fp32 accumulate, both operands already K-contiguous -- the layout the
tensor cores want. There is no reduction and no elementwise tail, so there is nothing to fuse.
This is a race against cuBLASLt, and the way to win it turned out not to be a better kernel.

**What is fast here, and why.** `F.linear` hands the problem to cuBLASLt and takes whichever
algorithm the heuristic ranks first. The heuristic is a fast predictor, not a search, and on this
shape family its first pick is not its best pick. Enumerating the candidates it returns for each
scored shape, validating them numerically and timing them inside the scored window finds a
*different* cuBLASLt algorithm that is measurably faster on every one:

    M        default tile   chosen tile   measured ratio
    1760     456            333           1.089
    20680    23             184           1.037
    23760    23             201           1.032
    25168    184            201           1.014
    64680    23             184           1.068

Reproduced across fresh processes and two GPU leases, every chosen algorithm bit-exact against
`F.linear` (`matched = 1.0`, `max_abs = 0`), and every margin above that shape's own one-sided
null bound (1.002-1.011, measured from a delegating build where both sides run identical code).
The internal control is the reassuring part: re-running the heuristic's *own first choice* through
this same wrapper measures 0.995-1.001, i.e. it reproduces `F.linear`, so the win is the algorithm
change and not the wrapper.

**What was tried and lost.** A hand-written Triton persistent TMA GEMM -- host-built TMA
descriptors, group-major rastering, fp32 TMEM accumulators, warp-specialised mainloop, split
epilogue -- was swept over 480 configurations of `BM x BN x BK x warps x GROUP_M x subtile x grid`
and peaked at 0.86 of the reference. `ncu` attributes the shortfall to mainloop MMA issue density
(reference tensor pipe 80.6% of peak active cycles against 63.1%) while ruling memory out: the
Triton kernel reads 30% *fewer* DRAM bytes and hits L2 better. The cause of the density deficit is
not established here; the warp-specialised handoff and the reference's 2-CTA cooperative MMA are
both plausible and no ablation separates them. `docs/findings.md` carries the full accounting and
labels every claim as symptom, correlation, or established by ablation.

**Design consequences of the harness, all load-bearing.**

* Nothing is derived from `weight` or `bias` in `__init__`. The harness casts parameters to bf16
  *after* construction and only then copies the reference values in, so anything bound during
  `__init__` would address storage that has since been replaced. `_weight_2d` rebuilds its cached
  view whenever the parameter's `(data_ptr, dtype, shape, stride)` changes, and does not cache at
  all under grad mode -- a view built inside `no_grad` carries no `grad_fn`, so handing it back to
  a grad-enabled call would leave `weight.grad` empty where the baseline fills it in.
* `self.proj` keeps the baseline's structure. Weight sharing is
  `load_state_dict(..., strict=False)` inside a bare `except: pass`, so renaming it would make the
  load a silent no-op and leave this module on its own random weights -- which surfaces as a
  numerical failure, not a load error. Verified: a renamed variant scores `matched = 0.0129`.
* `forward` keeps `x.view(x.shape[0], self.input_size)` rather than `reshape`, so an input whose
  layout makes the view illegal raises the same error the baseline raises instead of silently
  costing a copy and a launch.
* The cuBLASLt plan is built on first use for a shape, never in `__init__` and never inside the
  timed window -- the harness runs correctness rounds and ten warmup iterations first. Only
  algorithms that need **zero** workspace are admitted, so `forward` allocates nothing but its
  output.
* Any failure -- the extension not building, the heuristic not offering the recorded algorithm,
  an unexpected dtype, layout, device or shape -- delegates to the frozen
  `candidate/L1/linear.Matmul`, which is the same call the baseline makes. A problem here can cost
  latency; it cannot produce a wrong answer.
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn

from ..L1.conv3d import Conv3d
from ..L1.linear import Matmul

# ---------------------------------------------------------------------------
# Admission table.
#
# Keyed on the exact problem `(M, N, K, has_bias)`. cuBLASLt's best algorithm is genuinely
# shape-specific here -- the same enumeration picks tile 333 at M=1760, 184 at M=20680 and 64680,
# and 201 at M=23760 and 25168 -- so an interval would be asserting a generalisation the
# measurement does not support. Each key is its own measured point, which makes every entry a
# closed region with both endpoints measured, degenerately but honestly. Extending coverage to the
# other 88 captured row counts means enumerating and timing each of them; the harness in
# `profile/01-sweep/cublaslt_algos.py` does exactly that and nothing else is needed.
#
# The value is the algorithm's *configuration*, not its position in the heuristic's list: index
# order is not a stable identifier across drivers or runs, so the plan builder re-enumerates and
# matches on configuration, and delegates when no returned algorithm matches.
#
#   (algo_id, tile_id, stages_id, split_k, cta_swizzling, custom_option)
# ---------------------------------------------------------------------------

_ADMITTED: dict[tuple, tuple] = {
    #  (M,     N,    K,    has_bias): (algo_id, tile, stages, split_k, swizzle, custom, ratio)
    (1760, 1152, 1536, True): (66, 333, 35, 1, 0, 1, 1.089),
    (20680, 1152, 1536, True): (66, 184, 35, 1, 0, 2, 1.037),
    (23760, 1152, 1536, True): (66, 201, 35, 1, 0, 2, 1.032),
    (25168, 1152, 1536, True): (66, 201, 35, 1, 0, 2, 1.014),
    (64680, 1152, 1536, True): (66, 184, 35, 1, 0, 2, 1.068),
}

# 128-bit vectorised loads want 16-byte-aligned bases. The harness's shifting pool hands out
# 256-byte-aligned slots, but that is a property of the pool, not a promise.
_ALIGN_BYTES = 16

_EXT = None
_STATUS = "uninitialised"

_CPP = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <map>
#include <memory>
#include <vector>

#define LT_CHECK(x) TORCH_CHECK((x) == CUBLAS_STATUS_SUCCESS, \
                                "cublasLt call failed at line ", __LINE__)

namespace {

// Everything below is per-device. cuBLASLt handles, matmul descriptors and matrix layouts are all
// bound to the context they were created under, so a plan built while device 0 was current cannot
// be used to launch on device 1 -- and `getCurrentCUDAStream()` with no argument would silently
// hand back device 0's stream. Every entry point therefore takes an explicit device index, installs
// a CUDAGuard first, and only then touches the library or the stream.
//
// Ownership is RAII: the destructors below run at interpreter teardown, so a long-lived process
// that rebuilds plans (repeated module construction, `.to()` between devices) does not accumulate
// library objects.

struct LayoutOwner {
  cublasLtMatrixLayout_t v = nullptr;
  ~LayoutOwner() { if (v) cublasLtMatrixLayoutDestroy(v); }
};

struct DescOwner {
  cublasLtMatmulDesc_t v = nullptr;
  ~DescOwner() { if (v) cublasLtMatmulDescDestroy(v); }
};

struct Plan {
  DescOwner op;
  LayoutOwner a, b, d;
  cublasLtMatmulAlgo_t algo{};
  int device = -1;
  int64_t M = 0, N = 0, K = 0;
};

struct HandleOwner {
  cublasLtHandle_t v = nullptr;
  ~HandleOwner() { if (v) cublasLtDestroy(v); }
};

std::map<int, std::unique_ptr<HandleOwner>> g_handles;
std::vector<std::unique_ptr<Plan>> g_plans;

cublasLtHandle_t handle_for(int device) {
  auto it = g_handles.find(device);
  if (it == g_handles.end()) {
    auto owner = std::make_unique<HandleOwner>();
    LT_CHECK(cublasLtCreate(&owner->v));
    it = g_handles.emplace(device, std::move(owner)).first;
  }
  return it->second->v;
}

int cfg(const cublasLtMatmulAlgo_t& algo, cublasLtMatmulAlgoConfigAttributes_t attr) {
  int v = -1;
  cublasLtMatmulAlgoConfigGetAttribute(&algo, attr, &v, sizeof(v), nullptr);
  return v;
}

// Build a plan on `device` and select the algorithm whose configuration matches the one recorded in
// the admission table. Returns the plan id, or -1 when the heuristic does not offer that algorithm
// (driver change, different hardware) or it would need a workspace.
int64_t make_plan(int64_t device, int64_t M, int64_t N, int64_t K, int64_t bias_addr,
                  int64_t want_algo, int64_t want_tile, int64_t want_stages,
                  int64_t want_splitk, int64_t want_swizzle, int64_t want_custom,
                  int64_t max_algos) {
  TORCH_CHECK(device >= 0, "make_plan needs a real device index");
  const at::cuda::CUDAGuard guard((c10::DeviceIndex)device);
  cublasLtHandle_t handle = handle_for((int)device);

  auto p = std::make_unique<Plan>();
  p->device = (int)device;
  p->M = M; p->N = N; p->K = K;
  void* bias_ptr = reinterpret_cast<void*>(bias_addr);

  // Column-major D[N,M] = op(weight)[N,K] * op(x)[K,M]: transa = T, transb = N. Row-major
  // weight[N,K] is column-major [K,N] with ld = K; row-major x[M,K] is [K,M] with ld = K;
  // row-major out[M,N] is [N,M] with ld = N. No operand is relaid out.
  LT_CHECK(cublasLtMatmulDescCreate(&p->op.v, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  LT_CHECK(cublasLtMatmulDescSetAttribute(p->op.v, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  LT_CHECK(cublasLtMatmulDescSetAttribute(p->op.v, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
  cublasLtEpilogue_t ep = CUBLASLT_EPILOGUE_BIAS;
  LT_CHECK(cublasLtMatmulDescSetAttribute(p->op.v, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof(ep)));
  LT_CHECK(cublasLtMatmulDescSetAttribute(p->op.v, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                          &bias_ptr, sizeof(bias_ptr)));
  LT_CHECK(cublasLtMatrixLayoutCreate(&p->a.v, CUDA_R_16BF, K, N, K));
  LT_CHECK(cublasLtMatrixLayoutCreate(&p->b.v, CUDA_R_16BF, K, M, K));
  LT_CHECK(cublasLtMatrixLayoutCreate(&p->d.v, CUDA_R_16BF, N, M, N));

  cublasLtMatmulPreference_t pref = nullptr;
  LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  size_t workspace = 0;   // only workspace-free algorithms are admitted, so forward allocates
                          // nothing beyond its output
  cublasStatus_t pst = cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace, sizeof(workspace));
  std::vector<cublasLtMatmulHeuristicResult_t> results((size_t)max_algos);
  int returned = 0;
  cublasStatus_t st = CUBLAS_STATUS_NOT_INITIALIZED;
  if (pst == CUBLAS_STATUS_SUCCESS) {
    st = cublasLtMatmulAlgoGetHeuristic(handle, p->op.v, p->a.v, p->b.v, p->d.v, p->d.v, pref,
                                        (int)max_algos, results.data(), &returned);
  }
  cublasLtMatmulPreferenceDestroy(pref);
  if (st != CUBLAS_STATUS_SUCCESS) returned = 0;

  bool ready = false;
  for (int i = 0; i < returned; ++i) {
    const auto& r = results[i];
    if (r.state != CUBLAS_STATUS_SUCCESS || r.workspaceSize != 0) continue;
    if (cfg(r.algo, CUBLASLT_ALGO_CONFIG_ID) != (int)want_algo) continue;
    if (cfg(r.algo, CUBLASLT_ALGO_CONFIG_TILE_ID) != (int)want_tile) continue;
    if (cfg(r.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID) != (int)want_stages) continue;
    if (cfg(r.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM) != (int)want_splitk) continue;
    if (cfg(r.algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING) != (int)want_swizzle) continue;
    if (cfg(r.algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION) != (int)want_custom) continue;
    p->algo = r.algo;
    ready = true;
    break;
  }
  if (!ready) return -1;   // owners free the descriptors as `p` goes out of scope
  g_plans.push_back(std::move(p));
  return (int64_t)g_plans.size() - 1;
}

void run(int64_t plan_id, torch::Tensor x, torch::Tensor weight, torch::Tensor out) {
  TORCH_CHECK(plan_id >= 0 && plan_id < (int64_t)g_plans.size(), "bad plan id");
  const Plan& p = *g_plans[(size_t)plan_id];

  // Re-validate at the boundary. The Python predicate already checked all of this, but a C++
  // launch with mismatched device, dtype or layout is undefined behaviour rather than a wrong
  // number, so it is worth the handful of integer comparisons.
  TORCH_CHECK(x.is_cuda() && weight.is_cuda() && out.is_cuda(), "operands must be on CUDA");
  TORCH_CHECK(x.get_device() == p.device && weight.get_device() == p.device
              && out.get_device() == p.device,
              "operands live on a different device than the plan was built for");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16
              && out.scalar_type() == at::kBFloat16, "operands must be bfloat16");
  TORCH_CHECK(x.dim() == 2 && weight.dim() == 2 && out.dim() == 2, "operands must be 2-D");
  TORCH_CHECK(x.size(0) == p.M && x.size(1) == p.K, "x does not match the plan");
  TORCH_CHECK(weight.size(0) == p.N && weight.size(1) == p.K, "weight does not match the plan");
  TORCH_CHECK(out.size(0) == p.M && out.size(1) == p.N, "out does not match the plan");
  TORCH_CHECK(x.stride(1) == 1 && x.stride(0) == p.K, "x must be row-major and unpadded");
  TORCH_CHECK(weight.stride(1) == 1 && weight.stride(0) == p.K,
              "weight must be row-major and unpadded");
  TORCH_CHECK(out.stride(1) == 1 && out.stride(0) == p.N, "out must be row-major and unpadded");

  // Guard first, then resolve the stream *for this device* -- the no-argument overload would
  // return the process's current device's stream instead.
  const at::cuda::CUDAGuard guard((c10::DeviceIndex)p.device);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream((c10::DeviceIndex)p.device);
  float alpha = 1.0f, beta = 0.0f;
  LT_CHECK(cublasLtMatmul(handle_for(p.device), p.op.v, &alpha,
                          weight.data_ptr(), p.a.v,
                          x.data_ptr(), p.b.v,
                          &beta,
                          out.data_ptr(), p.d.v,
                          out.data_ptr(), p.d.v,
                          &p.algo, nullptr, 0,
                          stream));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("make_plan", &make_plan);
  m.def("run", &run);
}
"""


def _init_extension() -> None:
    """Build the wrapper once, at import. Any failure degrades to delegation.

    Building here rather than on first use keeps compilation out of the timed window and out of
    the correctness rounds, and it happens while the bench worker is still producing output, clear
    of the stall watchdog. The result is cached by `load_inline` across processes, so only the
    first run in an environment pays for it.
    """
    global _EXT, _STATUS
    if not _ADMITTED:
        _STATUS = "disabled:no-admitted-shapes"
        return
    if not torch.cuda.is_available():
        _STATUS = "disabled:no-cuda-device"
        print(f"[candidate L2/vision_patch_embed] cuBLASLt selector unavailable, delegating "
              f"to torch: {_STATUS}", file=sys.stderr, flush=True)
        return
    try:
        from torch.utils.cpp_extension import load_inline
        _EXT = load_inline(name="vpe_cublaslt_select", cpp_sources=[_CPP],
                           extra_cflags=["-O2"], extra_ldflags=["-lcublasLt", "-lcublas"],
                           functions=None, with_cuda=True, verbose=False)
        _STATUS = "built"
    except Exception as exc:  # noqa: BLE001 - compiler, headers, or driver said no
        _EXT = None
        _STATUS = f"failed:{type(exc).__name__}: {exc}"
        print(f"[candidate L2/vision_patch_embed] cuBLASLt selector unavailable, delegating "
              f"to torch: {_STATUS}", file=sys.stderr, flush=True)


_init_extension()


def _plan(x2: torch.Tensor, weight: torch.Tensor, bias):
    """Return the recorded algorithm configuration for this call, or None to delegate.

    Pure and cheap: integer and attribute checks only, no CUDA calls and no device
    synchronisation, so it costs the same whether it admits or delegates. Every predicate guards
    something the fast path relies on.
    """
    if _EXT is None:
        return None
    # The fast path builds no graph, so grad mode has to delegate.
    if torch.is_grad_enabled():
        return None
    if x2.dtype is not torch.bfloat16 or weight.dtype is not torch.bfloat16:
        return None
    # A missing bias is not a cheaper case, it is a different epilogue: every table entry was
    # measured with `CUBLASLT_EPILOGUE_BIAS` and cannot describe a bias-free call.
    if bias is None or bias.dtype is not torch.bfloat16:
        return None
    if not x2.is_cuda or weight.device != x2.device or bias.device != x2.device:
        return None
    if weight.dim() != 2 or x2.dim() != 2:
        return None
    N, K = weight.shape
    if x2.shape[1] != K or K <= 0 or N <= 0:
        return None
    # The layouts describe unit-stride rows with no padding; a hidden `.contiguous()` would cost
    # a launch worth more than the algorithm change wins back.
    if not x2.is_contiguous() or weight.stride(-1) != 1 or weight.stride(0) != K:
        return None
    if bias.dim() != 1 or bias.shape[0] != N or not bias.is_contiguous():
        return None
    if (x2.data_ptr() % _ALIGN_BYTES or weight.data_ptr() % _ALIGN_BYTES
            or bias.data_ptr() % _ALIGN_BYTES):
        return None
    return _ADMITTED.get((x2.shape[0], N, K, True))


class VisionPatchEmbed(nn.Module):
    def __init__(self, patch_size: int, temporal_patch_size: int,
                 in_channels: int, embed_dim: int, bias: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_size = in_channels * temporal_patch_size * patch_size * patch_size
        kernel = (temporal_patch_size, patch_size, patch_size)
        self.proj = Conv3d(in_channels, embed_dim, kernel, bias=bias)
        self.linear = Matmul()
        # All populated on first use, never in __init__ -- see the module docstring.
        self._weight_view: torch.Tensor | None = None
        self._weight_key: tuple | None = None
        self._plans: dict[tuple, int] = {}

    def _weight_2d(self) -> torch.Tensor:
        """The [embed_dim, input_size] view of the Conv3d weight, cached across calls.

        The key covers every component a view depends on, so the harness's post-construction
        cast, an in-place `load_state_dict`, or a `.to()` that moves the parameter all invalidate
        it. Building the key is host-side arithmetic; it reads `data_ptr` but makes no CUDA call.

        Only inference views are cached. A view built while grad was disabled carries no
        `grad_fn`, so handing it back to a later grad-enabled call would quietly detach the
        parameter -- the output and the input gradient would still be right, and only
        `weight.grad` would go missing, which is exactly the kind of divergence a forward-only
        comparison cannot see.
        """
        w = self.proj.weight
        if torch.is_grad_enabled():
            return w.view(self.embed_dim, self.input_size)
        key = (w.data_ptr(), w.dtype, tuple(w.shape), w.stride())
        if key != self._weight_key:
            self._weight_view = w.view(self.embed_dim, self.input_size)
            self._weight_key = key
        return self._weight_view

    def _plan_id(self, x2: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                 config: tuple) -> int:
        """The cuBLASLt plan for this problem, built once and kept.

        Keyed on everything the descriptors bake in: the device (handles and descriptors belong to
        the context they were created under), the problem dimensions, and the bias pointer, which
        the operation descriptor stores directly. A negative id means the heuristic did not offer
        the recorded algorithm on this machine; it is cached too, so the miss costs one enumeration
        rather than one per call.
        """
        device = x2.get_device()
        key = (device, x2.shape[0], weight.shape[0], weight.shape[1], bias.data_ptr())
        plan_id = self._plans.get(key)
        if plan_id is None:
            algo, tile, stages, split_k, swizzle, custom = config[:6]
            try:
                plan_id = _EXT.make_plan(device, x2.shape[0], weight.shape[0], weight.shape[1],
                                         bias.data_ptr(), algo, tile, stages, split_k,
                                         swizzle, custom, 32)
            except Exception:  # noqa: BLE001 - a refusal is a delegation, never an error
                plan_id = -1
            self._plans[key] = plan_id
        return plan_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], self.input_size)
        weight = self._weight_2d()
        bias = self.proj.bias
        config = _plan(x, weight, bias)
        if config is None:
            return self.linear(x, weight, bias)
        plan_id = self._plan_id(x, weight, bias, config)
        if plan_id < 0:
            return self.linear(x, weight, bias)
        out = torch.empty((x.shape[0], self.embed_dim), device=x.device, dtype=x.dtype)
        _EXT.run(plan_id, x, weight, out)
        return out
