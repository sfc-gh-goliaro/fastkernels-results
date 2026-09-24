// Extension translation unit for the fused AlphaFold3 SwiGLU transition kernels.
//
// Holds only the ATen-facing half: allocation of the two intermediates, the
// warp-count choice, the launches, and the TORCH_LIBRARY registration. All device
// code lives in _af3_swiglu_kernels.cuh so the same kernels can be compiled by a
// standalone nvcc harness with -lineinfo for Nsight Compute.
//
// The includes are deliberately lean. This registers through TORCH_LIBRARY rather
// than pybind, so <torch/extension.h> -- which dominated the build in this
// workspace's probe -- is not needed.

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string_view>

#include "_af3_swiglu_kernels.cuh"

namespace {

using namespace fk_af3;  // NOLINT(build/namespaces) - one TU, one purpose

// ---------------------------------------------------------------------------
// Fast-path predicate, enforced here rather than in Python.
//
// The Python module carries the same predicate as readable, unit-tested helpers
// (`_activation_ok`, `_mask_ok`, `_param_ok`, `_flattens_row_for_row`) and those
// remain the specification. But running it in Python on every call is not free:
// measured at 7.5 us for SwiGLUTransition's five parameters and 13.8 us for
// ConditionedTransitionBlock's nine (`profile/p1-predicate/results.txt`), against
// a ~15 us window floor. Since the verdict cannot be cached -- a lazy negation view
// or a same-storage reshape keeps a parameter's identity *and* its address, so any
// key built from those serves a stale verdict and the kernel then reads storage
// that does not represent the tensor -- the choice is between paying 10 us of
// Python per call and doing the same comparisons where they cost nanoseconds.
//
// So the operator validates its own arguments and returns nullopt when it will not
// handle them; Python sees None, increments its fallback counter and runs the
// baseline composition. This is the shape `candidate/L1/layer_norm.py` already uses
// for the same reason. `profile/p1-contract/check_fastpath.py` asserts the two
// implementations agree case by case, so they cannot drift apart silently.
//
// Grad mode and autocast stay in Python: they are two cheap calls there and would
// need version-sensitive ATen APIs here.
// ---------------------------------------------------------------------------
constexpr int64_t kAlignBytes = 16;
constexpr int64_t kKMultiple = 16;

inline bool is_aligned(const at::Tensor& t) {
  return reinterpret_cast<uintptr_t>(t.const_data_ptr()) % kAlignBytes == 0;
}

// A lazy negation or conjugation view reads as -x / conj(x) to every ATen op while
// its storage holds x, so a kernel taking data_ptr() would silently return the
// wrong values. Contiguity does not imply resolved.
inline bool resolved(const at::Tensor& t) { return !t.is_neg() && !t.is_conj(); }

inline bool activation_ok(const at::Tensor& t, int64_t k) {
  return t.defined() && t.scalar_type() == at::kBFloat16 &&
         t.layout() == at::kStrided && t.is_cuda() && t.is_contiguous() &&
         resolved(t) && t.dim() >= 2 && t.size(-1) == k && is_aligned(t);
}

inline bool param_ok(const at::Tensor& t, at::IntArrayRef shape,
                     const at::Device& dev) {
  return t.defined() && t.scalar_type() == at::kBFloat16 &&
         t.layout() == at::kStrided && t.device() == dev &&
         t.sizes() == shape && t.is_contiguous() && resolved(t) && is_aligned(t);
}

// True iff a contiguous tensor of `shape` addresses `leading` row for row.
//
// Suffix equality alone is not enough: mask[3,4] against x[2,3,4,8] has the right
// suffix, but the baseline broadcast replicates it over the leading 2, so a flat
// row-indexed kernel would read past the end and mis-gate rows 12..23. The
// element-count clause forces every uncovered leading dimension to be 1, which is
// exactly when the broadcast is a no-op.
inline bool flattens_row_for_row(at::IntArrayRef shape, at::IntArrayRef leading) {
  const int64_t rank = shape.size();
  if (rank == 0 || rank > static_cast<int64_t>(leading.size())) {
    return false;
  }
  if (shape != leading.slice(leading.size() - rank, rank)) {
    return false;
  }
  int64_t rows = 1;
  for (const int64_t d : leading) {
    rows *= d;
  }
  int64_t elems = 1;
  for (const int64_t d : shape) {
    elems *= d;
  }
  return elems == rows;
}

// `mask` absent is admissible and cheaper than the baseline's own path: the
// baseline materializes new_ones(...) and multiplies, and multiplying by an exact
// 1.0 is the identity on the rounding chain for every finite value and infinity, so
// the fast path omits the factor. A present mask must share the activation's dtype,
// because `out * mask` promotes and an fp32 mask would change the output *dtype*.
inline bool mask_ok(const std::optional<at::Tensor>& mask,
                    at::IntArrayRef leading, const at::Device& dev) {
  if (!mask.has_value()) {
    return true;
  }
  const at::Tensor& m = *mask;
  return m.defined() && m.scalar_type() == at::kBFloat16 &&
         m.layout() == at::kStrided && m.is_cuda() && m.device() == dev &&
         m.is_contiguous() && resolved(m) &&
         flattens_row_for_row(m.sizes(), leading) && is_aligned(m);
}

inline bool rows_ok(int64_t rows) { return rows >= 1 && rows <= kMaxRows; }

inline bool k_ok(int64_t k) { return k > 0 && k % kKMultiple == 0; }

// ---------------------------------------------------------------------------
// Launch plumbing.
//
// One CTA owns one n8 output tile; its warps split the reduction extent. So the
// grid is ceil(M/BM) x ceil(N/8) whatever the warp count, and the warp count is a
// pure occupancy knob: it multiplies resident warps without touching the grid.
//
// The choice is measured, not derived -- see profile/p1-tiles/results.md for the
// sweep and profile/ctb-c768-adaln-down-v1/REPORT.md for why occupancy is the
// axis that matters on these shapes.
// ---------------------------------------------------------------------------
inline dim3 grid_for(int64_t rows, int n) {
  return dim3(static_cast<unsigned>((rows + kBM - 1) / kBM),
              static_cast<unsigned>((n + 7) / 8));
}

// Warp counts the kernels are instantiated for. 16 is the ceiling: row_stats
// divides blockDim across the BM=16 rows of the M tile, and one row's threads have
// to stay inside a warp for its butterfly reduction, so 16 warps x 32 lanes / 16
// rows is exactly 32 threads per row.
#define FK_AF3_DISPATCH_WARPS(WARPS, LAUNCH) \
  switch (WARPS) {                           \
    case 1: {                                \
      constexpr int kNW = 1;                 \
      LAUNCH;                                \
      break;                                 \
    }                                        \
    case 2: {                                \
      constexpr int kNW = 2;                 \
      LAUNCH;                                \
      break;                                 \
    }                                        \
    case 4: {                                \
      constexpr int kNW = 4;                 \
      LAUNCH;                                \
      break;                                 \
    }                                        \
    case 16: {                               \
      constexpr int kNW = 16;                \
      LAUNCH;                                \
      break;                                 \
    }                                        \
    default: {                               \
      constexpr int kNW = 8;                  \
      LAUNCH;                                \
      break;                                 \
    }                                        \
  }

// Which kernel is asking. The captured-shape table below has to be keyed on this
// as well as on the shape: CTB c_a=768's AdaLN and its gated down-projection have
// identical (rows, n, k_min) = (16, 768, 384) but measured optima of 16 and 8 warps
// respectively, so a shape-only table cannot express the measurement.
enum class Role { kAdaln, kUp, kDown };

// Per-(role, shape) minima from the isolated per-kernel sweep, at BM=16.
//
// **Not used by default, and that is the measured result rather than a shortcut.**
//
// `profile/p1-tiles/sweep_raw.txt` timed each kernel on its own -- 30 warmup + 300
// back-to-back launches -- for warp counts 1/2/4/8/16 and both BM values, and these
// are the literal minima of that sweep. Dispatching them end to end then made the
// scored window *worse* on four of the five shapes and tied on the fifth
// (`profile/p1-tiles/results.md`, five interleaved replicates per setting):
//
//   CTB c_a=768 M=16     rule 60.37 us   table 65.65 us
//   CTB c_a=128 M=368    rule 47.09 us   table 49.12 us
//   ST  c_in=128 M=256   rule 32.74 us   table 36.83 us
//   ST  c_in=64  M=128   rule 22.51 us   table 24.54 us
//   ST  c_in=384 M=16    rule 34.85 us   table 34.82 us
//
// Isolated minima do not compose here. Timing one kernel 300 times in a row keeps
// its weights hot in L2 and gives its grid the whole machine; in the scored chain
// each kernel runs once per iteration behind an L2 flush, with the tail of one
// overlapping the head of the next. So the sweep answers "which warp count is
// fastest for this kernel alone", which is not the question the dispatch has to
// answer.
//
// The table is retained, behind FK_AF3_SWIGLU_TUNE=table, so the comparison stays
// reproducible. The default is the general rule below.
struct WarpChoice {
  Role role;
  int64_t rows;
  int64_t n;
  int64_t k_min;
  int warps;
};

constexpr WarpChoice kMeasured[] = {
    // CTB c_a=768 c_s=384 n=2, M=16
    {Role::kAdaln, 16, 768, 384, 16},   // 12.56 us; w=8 was 13.21
    {Role::kUp, 16, 1536, 768, 4},      // 11.67 us; w=8 was 12.30
    {Role::kDown, 16, 768, 384, 8},     // 12.31 us; w=16 tied
    // CTB c_a=128 c_s=128 n=2, M=368
    {Role::kAdaln, 368, 128, 128, 2},   // 10.29 us; w=4 was 11.28
    {Role::kUp, 368, 256, 128, 1},      // 8.20 us; 1/2/4/8 all tied
    {Role::kDown, 368, 128, 128, 1},    // 10.25 us; every warp count tied
    // SwiGLUTransition c_in=128 n=4, M=256
    {Role::kUp, 256, 512, 128, 2},      // 10.91 us; w=16 was 22.54
    {Role::kDown, 256, 128, 512, 2},    // 8.20 us; 2/4/8/16 tied
    // SwiGLUTransition c_in=384 n=4, M=16
    {Role::kUp, 16, 1536, 384, 8},      // 10.81 us; w=16 was 12.30
    {Role::kDown, 16, 384, 1536, 16},   // 9.23 us; w=8 was 10.25
    // SwiGLUTransition c_in=64 n=4, M=128
    {Role::kUp, 128, 256, 64, 4},       // 6.68 us; w=1 was 8.21
    {Role::kDown, 128, 64, 256, 1},     // 6.15 us; every warp count tied
};

// The general rule, for every shape the table does not name.
//
// The grid is ceil(rows/BM) x ceil(n/8) CTAs and the split multiplies that. The
// target is about eight warps per SM across 148 SMs: enough that a warp stalling on
// a global load has something to switch to, which is exactly what the NCU run found
// missing (0.98 achieved warps per SM, 84-96% of scheduler slots with no eligible
// warp). Taking more warps than there are k16 tiles buys nothing and still pays the
// barrier -- measured as a 2x regression on one shape, so the cap is not cosmetic.
inline int warps_rule(int64_t rows, int64_t n, int ktiles) {
  const int64_t ctas = ((rows + kBM - 1) / kBM) * ((n + 7) / 8);
  int warps = 1;
  for (const int w : {2, 4, 8, 16}) {
    if (w <= ktiles && ctas * warps < 148 * 8) {
      warps = w;
    }
  }
  return warps;
}

// FK_AF3_SWIGLU_WARPS pins the split to a fixed value and FK_AF3_SWIGLU_TUNE=rule
// disables the measured table, which is how the sweep and the table-versus-rule
// comparison are taken through the official harness rather than a side binary.
// Both are read once per process.
inline int forced_warps() {
  static const int pinned = [] {
    const char* env = std::getenv("FK_AF3_SWIGLU_WARPS");
    if (env == nullptr) {
      return 0;
    }
    const int v = std::atoi(env);
    return (v == 1 || v == 2 || v == 4 || v == 8 || v == 16) ? v : 0;
  }();
  return pinned;
}

inline bool use_measured_table() {
  static const bool on = [] {
    const char* env = std::getenv("FK_AF3_SWIGLU_TUNE");
    return env != nullptr && std::string_view(env) == "table";
  }();
  return on;
}

// `n` is the kernel's output width; `k_min` the shortest reduction extent it runs
// (the gated down-projection has two, and a split above the shorter one's tile count
// would idle warps in that loop).
inline int warps_for(Role role, int64_t rows, int64_t n, int64_t k_min) {
  const int ktiles = static_cast<int>(k_min / 16);
  const int forced = forced_warps();
  if (forced != 0) {
    return forced <= ktiles ? forced : (ktiles >= 1 ? ktiles : 1);
  }
  if (use_measured_table()) {
    for (const WarpChoice& c : kMeasured) {
      if (c.role == role && c.rows == rows && c.n == n && c.k_min == k_min) {
        return c.warps <= ktiles ? c.warps : warps_rule(rows, n, ktiles);
      }
    }
  }
  return warps_rule(rows, n, ktiles);
}

#define FK_AF3_BF16(t) reinterpret_cast<const __nv_bfloat16*>((t).const_data_ptr())
#define FK_AF3_BF16_MUT(t) reinterpret_cast<__nv_bfloat16*>((t).mutable_data_ptr())

// Returns nullopt -- None in Python -- for anything outside the declared domain,
// which is the caller's signal to run the baseline composition.
std::optional<at::Tensor> swiglu_transition(
    const at::Tensor& x, const std::optional<at::Tensor>& mask,
    const at::Tensor& ln_w, const at::Tensor& ln_b, const at::Tensor& wa,
    const at::Tensor& wb, const at::Tensor& wout, double eps, int64_t variant) {
  // Rank is checked *before* anything reads size(-1) or slices sizes(). On a
  // rank-0 tensor `size(-1)` raises, and an exception thrown here escapes to the
  // caller instead of returning nullopt -- so Python never sees None, never counts
  // the fallback, and never runs the baseline. `activation_ok` enforces dim() >= 2
  // too, but by then it is too late to be the only guard.
  if (!x.defined() || x.dim() < 2 || !wa.defined() || wa.dim() != 2) {
    return std::nullopt;
  }
  {
    const int64_t k = x.size(-1);
    const int64_t hid = wa.size(0);
    const at::Device dev = x.device();
    if (!activation_ok(x, k) || !k_ok(k) || !k_ok(hid) ||
        !rows_ok(x.numel() / k) ||
        !mask_ok(mask, x.sizes().slice(0, x.dim() - 1), dev) ||
        !param_ok(ln_w, {k}, dev) || !param_ok(ln_b, {k}, dev) ||
        !param_ok(wa, {hid, k}, dev) || !param_ok(wb, {hid, k}, dev) ||
        !param_ok(wout, {k, hid}, dev)) {
      return std::nullopt;
    }
  }
  const c10::cuda::CUDAGuard guard(x.device());
  const auto stream = c10::cuda::getCurrentCUDAStream();
  const int c_in = static_cast<int>(x.size(-1));
  const int h = static_cast<int>(wa.size(0));
  const int64_t rows = x.numel() / c_in;

  at::Tensor hh = at::empty({rows, h}, x.options());
  at::Tensor out = at::empty(x.sizes(), x.options());
  const __nv_bfloat16* mask_p = mask.has_value() ? FK_AF3_BF16(*mask) : nullptr;
  const bool use_mma = (variant == 0);
  const int wu = warps_for(Role::kUp, rows, h, c_in);
  const int wd = warps_for(Role::kDown, rows, c_in, h);

  if (use_mma) {
    FK_AF3_DISPATCH_WARPS(
        wu, (up_swiglu_kernel<kNW, true, true>
             <<<grid_for(rows, h), kNW * kWarpSize, 0, stream>>>(
                 FK_AF3_BF16(x), FK_AF3_BF16(ln_w), FK_AF3_BF16(ln_b),
                 FK_AF3_BF16(wa), FK_AF3_BF16(wb), FK_AF3_BF16_MUT(hh), rows,
                 c_in, h, static_cast<float>(eps))))
    FK_AF3_DISPATCH_WARPS(
        wd, (down_kernel<kNW, true, false>
             <<<grid_for(rows, c_in), kNW * kWarpSize, 0, stream>>>(
                 FK_AF3_BF16(hh), FK_AF3_BF16(wout), nullptr, nullptr, nullptr,
                 mask_p, FK_AF3_BF16_MUT(out), rows, h, c_in, 0)))
  } else {
    up_swiglu_kernel<4, false, true>
        <<<grid_for(rows, h), 4 * kWarpSize, 0, stream>>>(
            FK_AF3_BF16(x), FK_AF3_BF16(ln_w), FK_AF3_BF16(ln_b), FK_AF3_BF16(wa),
            FK_AF3_BF16(wb), FK_AF3_BF16_MUT(hh), rows, c_in, h,
            static_cast<float>(eps));
    down_kernel<4, false, false>
        <<<grid_for(rows, c_in), 4 * kWarpSize, 0, stream>>>(
            FK_AF3_BF16(hh), FK_AF3_BF16(wout), nullptr, nullptr, nullptr, mask_p,
            FK_AF3_BF16_MUT(out), rows, h, c_in, 0);
  }
  return out;
}

std::optional<at::Tensor> conditioned_transition(
    const at::Tensor& a, const at::Tensor& s,
    const std::optional<at::Tensor>& mask, const at::Tensor& ln_s_w,
    const at::Tensor& ag_w, const at::Tensor& ag_b, const at::Tensor& as_w,
    const at::Tensor& wa, const at::Tensor& wb, const at::Tensor& g_w,
    const at::Tensor& g_b, const at::Tensor& wout, double eps_a, double eps_s,
    int64_t variant) {
  // Rank first, independently for `a` and `s`, for the reason
  // `swiglu_transition` documents: a rank-0 operand makes size(-1) throw out of the
  // operator, which bypasses the fallback entirely.
  if (!a.defined() || a.dim() < 2 || !s.defined() || s.dim() < 2 ||
      !wa.defined() || wa.dim() != 2) {
    return std::nullopt;
  }
  {
    const int64_t ca = a.size(-1);
    const int64_t cs = s.size(-1);
    const int64_t hid = wa.size(0);
    const at::Device dev = a.device();
    const at::IntArrayRef leading = a.sizes().slice(0, a.dim() - 1);
    // `s` gets the same row-for-row test as the mask: the baseline broadcasts
    // s-derived tensors against `a`, so an `s` whose leading shape merely has the
    // right suffix -- a[16,1,c_a] with s[1,16,c_s] -- would give the baseline a
    // [16,16,c_a] output rather than a.shape.
    if (!activation_ok(a, ca) || !activation_ok(s, cs) || s.device() != dev ||
        !k_ok(ca) || !k_ok(cs) || !k_ok(hid) || !rows_ok(a.numel() / ca) ||
        !flattens_row_for_row(s.sizes().slice(0, s.dim() - 1), leading) ||
        !mask_ok(mask, leading, dev) || !param_ok(ln_s_w, {cs}, dev) ||
        !param_ok(ag_w, {ca, cs}, dev) || !param_ok(ag_b, {ca}, dev) ||
        !param_ok(as_w, {ca, cs}, dev) || !param_ok(wa, {hid, ca}, dev) ||
        !param_ok(wb, {hid, ca}, dev) || !param_ok(g_w, {ca, cs}, dev) ||
        !param_ok(g_b, {ca}, dev) || !param_ok(wout, {ca, hid}, dev)) {
      return std::nullopt;
    }
  }
  const c10::cuda::CUDAGuard guard(a.device());
  const auto stream = c10::cuda::getCurrentCUDAStream();
  const int c_a = static_cast<int>(a.size(-1));
  const int c_s = static_cast<int>(s.size(-1));
  const int h = static_cast<int>(wa.size(0));
  const int64_t rows = a.numel() / c_a;

  at::Tensor a1 = at::empty({rows, c_a}, a.options());
  at::Tensor hh = at::empty({rows, h}, a.options());
  at::Tensor out = at::empty(a.sizes(), a.options());
  const __nv_bfloat16* mask_p = mask.has_value() ? FK_AF3_BF16(*mask) : nullptr;
  const bool use_mma = (variant == 0);
  // The AdaLN kernel reduces over c_s twice; the gated down-projection reduces
  // over h and over c_s, so the shorter of the two bounds its split.
  const int wa_n = warps_for(Role::kAdaln, rows, c_a, c_s);
  const int wu = warps_for(Role::kUp, rows, h, c_a);
  const int wd = warps_for(Role::kDown, rows, c_a, c_s < h ? c_s : h);

  if (use_mma) {
    FK_AF3_DISPATCH_WARPS(
        wa_n, (adaln_kernel<kNW, true>
             <<<grid_for(rows, c_a), kNW * kWarpSize, 0, stream>>>(
                 FK_AF3_BF16(a), FK_AF3_BF16(s), FK_AF3_BF16(ln_s_w),
                 FK_AF3_BF16(ag_w), FK_AF3_BF16(ag_b), FK_AF3_BF16(as_w),
                 FK_AF3_BF16_MUT(a1), rows, c_a, c_s, static_cast<float>(eps_a),
                 static_cast<float>(eps_s))))
    FK_AF3_DISPATCH_WARPS(
        wu, (up_swiglu_kernel<kNW, true, false>
             <<<grid_for(rows, h), kNW * kWarpSize, 0, stream>>>(
                 FK_AF3_BF16(a1), nullptr, nullptr, FK_AF3_BF16(wa),
                 FK_AF3_BF16(wb), FK_AF3_BF16_MUT(hh), rows, c_a, h, 0.0f)))
    FK_AF3_DISPATCH_WARPS(
        wd, (down_kernel<kNW, true, true>
             <<<grid_for(rows, c_a), kNW * kWarpSize, 0, stream>>>(
                 FK_AF3_BF16(hh), FK_AF3_BF16(wout), FK_AF3_BF16(s),
                 FK_AF3_BF16(g_w), FK_AF3_BF16(g_b), mask_p,
                 FK_AF3_BF16_MUT(out), rows, h, c_a, c_s)))
  } else {
    adaln_kernel<4, false><<<grid_for(rows, c_a), 4 * kWarpSize, 0, stream>>>(
        FK_AF3_BF16(a), FK_AF3_BF16(s), FK_AF3_BF16(ln_s_w), FK_AF3_BF16(ag_w),
        FK_AF3_BF16(ag_b), FK_AF3_BF16(as_w), FK_AF3_BF16_MUT(a1), rows, c_a, c_s,
        static_cast<float>(eps_a), static_cast<float>(eps_s));
    up_swiglu_kernel<4, false, false>
        <<<grid_for(rows, h), 4 * kWarpSize, 0, stream>>>(
            FK_AF3_BF16(a1), nullptr, nullptr, FK_AF3_BF16(wa), FK_AF3_BF16(wb),
            FK_AF3_BF16_MUT(hh), rows, c_a, h, 0.0f);
    down_kernel<4, false, true>
        <<<grid_for(rows, c_a), 4 * kWarpSize, 0, stream>>>(
            FK_AF3_BF16(hh), FK_AF3_BF16(wout), FK_AF3_BF16(s), FK_AF3_BF16(g_w),
            FK_AF3_BF16(g_b), mask_p, FK_AF3_BF16_MUT(out), rows, h, c_a, c_s);
  }
  return out;
}

}  // namespace

TORCH_LIBRARY(fk_af3_swiglu_transition_cand, m) {
  m.def(
      "swiglu_transition(Tensor x, Tensor? mask, Tensor ln_w, Tensor ln_b, "
      "Tensor wa, Tensor wb, Tensor wout, float eps, int variant) -> Tensor?",
      &swiglu_transition);
  m.def(
      "conditioned_transition(Tensor a, Tensor s, Tensor? mask, Tensor ln_s_w, "
      "Tensor ag_w, Tensor ag_b, Tensor as_w, Tensor wa, Tensor wb, "
      "Tensor g_w, Tensor g_b, Tensor wout, float eps_a, float eps_s, "
      "int variant) -> Tensor?",
      &conditioned_transition);
}
