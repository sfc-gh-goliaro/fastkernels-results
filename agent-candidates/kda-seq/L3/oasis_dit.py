"""Oasis diffusion transformer, driven from `forward` and fused where it is free.

The operator is 605 M fp32 parameters (2.42 GB) evaluated on 288-864 tokens, so it
is bound by neither arithmetic nor bandwidth.  `profile/launch_census.py` /
`launch_census.log` measures 1741 kernel launches per forward at `T=4` against
12.26 ms of GPU busy time in a ~25 ms window, and `profile/candidate_census.py` /
`candidate_census.log` puts the baseline's GPU busy at 10.41-14.33 ms across
`T = 2..6` with the L2 flushed before every call.  Of that, `profile/gemm_floor.py` /
`gemm_floor.log` prices the GEMMs alone at 3.39 ms and `profile/measure_baseline.py` /
`measure_baseline.log` the fp32 SDPA calls at 2.08 ms, the layer norms at 0.46 ms and
the GELUs at 0.20 ms -- 6.13 ms of arithmetic the operator actually needs.  The other
~1447 launches are bookkeeping it does not:

  * `get_axial_freqs(9, 16)` is rebuilt from `linspace`/`einsum`/`repeat_interleave`
    /`broadcast_tensors`/`cat` inside every one of the 32 attention calls, although
    it depends on nothing but a constant parameter;
  * `_modulate` and `_gate` each call `Tensor.repeat` with a repeat factor of 1 --
    a full copy that changes nothing -- 128 times per forward;
  * `silu(c)` is recomputed 33 times on the same `[1, T, 1024]` tensor;
  * the residual stream is a *permuted* view of an NCHW buffer, which is a large
    share of the 339 `aten::copy_` launches per forward that
    `profile/measure_baseline.py` / `measure_baseline.log` records -- how many of
    those are the norms' and the GEMMs' own inputs specifically is not separated
    out, so the count is quoted and the attribution is not.

So this file hoists the cross-block invariants (one rotary table per axis for 32
attentions, one `silu(c)` for 33 projections, one contiguous residual stream packed
once at the front) and replaces the remaining elementwise fan-out with four fp32
CUDA kernels.  The GEMMs, the fp32 SDPA calls, `native_layer_norm` and `gelu` stay
on torch, unchanged, because the correctness gate pins them.

Bit-exactness, not tolerance
----------------------------
`torch.get_float32_matmul_precision()` is `'high'` here, so every fp32 GEMM rounds
its operands to TF32.  That makes deviation *saturate*: `profile/tf32_control.log`
injects a relative perturbation after block 0 and measures that **854 elements out
of 589824, each moved by exactly one fp32 ulp**, drop the harness's match ratio to
0.808 against a 0.99 gate -- and that the ratio is then flat across four decades of
perturbation size, depending only on depth.  The same log runs the control: under
`float32_matmul_precision("highest")` the identical injections score 1.0000 and the
output floor falls from ~1e-3 to ~2e-6, so TF32 operand rounding is the mechanism
rather than a plausible story about one.  A single flipped TF32 operand rounding is
worth ~8192 fp32 ulps (`2^-11` against `2^-24`), which is what turns an
imperceptible input perturbation into a `rtol=1e-3`-scale output difference.

Structured errors behave the same way, measured rather than extrapolated:
`profile/isolate_frozen.log` scores the all-frozen composition at 0.796 matched for
1.08x, and two individual frozen winners whose own outputs deviate by 3.6e-7 and
6.3e-7 absolute *each* drive the whole stack to 0.79.  Every fusion here runs in
every block including block 0, so its error is front-loaded by construction and the
late-depth escape hatch in that table (0.9977 at block 14) is unavailable to it.

Bit-exactness is therefore the design target *and* the verification method: every
expression below reproduces the reference's rounding boundaries, and `tests/`
asserts `torch.equal` plus an integer-bitcast comparison rather than a tolerance.
Two consequences worth stating because they look like missed optimizations:

  * no FMA.  `x * (1 + scale) + shift` is three roundings, and contracting the last
    two into one costs 4.768e-07 (`profile/bitexact_probes.py` / `.log`, "modulate
    fused as addcmul", measured at `T = 2, 4, 6`), which the budget
    above makes fatal.  The kernels therefore spell every operation with
    `__fmul_rn`/`__fadd_rn` *and* compile with `-fmad=false`; either alone would be
    a claim about the compiler rather than a guarantee.
  * no fused layer norm, and this is the one place a launch-count lever was left on the
    table and then measured away. `profile/layer_norm_schedule.py` / `.log` implements
    ATen's `vectorized_layer_norm_kernel<float, float>` Welford schedule in CUDA --
    twelve combinations of block size, shuffle ladder, block combine and final scale --
    and none is bit-exact; the residuals are one fp32 ulp, the signature of a different
    reduction order. (Bounded honestly: the installed torch is a wheel, so this build's
    own `.cu` is not on the machine to read.) Its value also collapsed once the graph
    stage shipped: the 65 launches it would save now cost approximately no host time.
  * no custom attention and no custom GEMM.  The attention output feeds `to_out`'s
    TF32 GEMM in every block, so a different SDPA backend's error is front-loaded.
    Measured, not argued: `profile/kernel_vs_reference_probe.py` /
    `kernel_vs_reference_probe.log` forces each backend across the whole reference
    model and scores it with the harness's own comparison -- `math` lands at 0.7945
    matched and `max_abs` 9.99e-04, and `cudnn` and `flash` reject fp32 outright, so
    `mem_efficient` is the only admissible backend and is what the unforced heuristic
    already picks.  The 2.08 ms a faster one would target is real and unreachable
    while the gate stands.

Which frozen winners may be called
----------------------------------
The workspace default is to import the frozen lower-level winners and call them, and
most of this tree does: the 16 blocks are the *baseline's* own block objects (no
`candidate/L3/oasis_block.py` exists, so the candidate finder aliases the baseline
module), so they and their L1/L2 children are the reference itself.

Three of the winners this tree holds cannot be called.  What is *measured* here is
the deviation, at all five captured shapes, in `profile/front_end_probe.py` /
`front_end_probe.log`:

  * `L2.oasis_patch_embed.forward` deviates by 3.6e-07 - 7.2e-07;
  * `L2.oasis_timestep_embedder.forward` deviates by 6.3e-07 - 8.3e-07.

`profile/isolate_frozen.py` / `isolate_frozen.log` prices both: substituting either
one alone takes the whole stack to 0.7974 and 0.7937 matched, against a 0.99 gate.

Those two files' own docstrings explain *why* -- the patch embed records that its
fused route admits fp32 and reproduces a vendor TF32 plan by pre-rounding operands,
and the timestep embedder records a scalar `fmaf` chain plus a shuffle butterfly that
cannot reproduce cuBLAS's wider-than-fp32 internal sum, together with a mode sweep
whose bitwise-exact alternative costs that module 42%, selected by a constant that is
not environment-overridable in a file this workspace forbids editing.  Those are their
claims, read from `candidate/L2/oasis_patch_embed.py` and
`candidate/L2/oasis_timestep_embedder.py`, not measurements taken here; the deviation
and its price above are.

  * `L2.oasis_final_layer` is the third, and it is the one `isolate_frozen.log` says
    is *safe* -- 1.0000 matched, `max_abs` 0.00e+00. That measurement was taken with
    the residual stream in the layout the reference produces, a permuted view of an
    NCHW buffer. Handed the *contiguous* stream this file builds, the same module
    deviates by 1.2e-04 (`profile/frozen_final_layer_probe.py` / `.log`, which runs
    both layouts side by side). So the earlier result is a statement about a layout,
    not about the module, and it does not transfer to a restructured stream.

All three bodies are transcribed: `F.conv2d` with the module's own attributes, which
is literally the baseline `Conv2d.forward` body; the cached-frequency sinusoid
followed by `mlp[0]`/`F.silu`/`mlp[2]`; and `F.layer_norm`/`F.linear` for the final
layer.  Every spelling is measured bit-exact in the logs above -- and, for the final
layer, on *both* layouts, which is why the reference path transcribes it too rather
than resting on ATen propagating a strided layout through 64 residual adds.
`F.conv2d` is used rather than the frozen `L1.conv2d` (also bit-exact there) because
the frozen `Conv2d` routes through a batch-keyed admission table that accepts only
batches in `{1, 2}`, so its behaviour would change at `T >= 3`, and it lacks the
tensor-subclass and dispatch-stack guards its sibling has.

A fourth is not even constructed: the tree builds the **baseline**
`OasisRotaryEmbedding` rather than the frozen `..L1.oasis_rotary` winner. They are
bit-identical in fp32 and carry the same `state_dict`, but the frozen one's fused table
kernel is a raw pybind call below the dispatcher, so under an active autocast it stays
fp32 while the baseline's `einsum` casts to bf16 -- a 1.7-absolute table difference
feeding all 32 attention calls, which made the whole model disagree with the baseline
under autocast and violated AC-4.2's "falls back *and agrees*". See the import site and
`profile/autocast_probe.py` / `.log`.

Every other frozen winner in the tree is still imported and its parameters are still
used under the baseline's own names, so `state_dict` keys and shapes match exactly; what
changes is only which of them have their `forward` called.

What the reference path guarantees, and how
-------------------------------------------
Transcribing a `forward` skips that module's *hooks*, and AC-4.2 requires a registered
hook to fall back **and still agree**. The fast path is safe because the predicate
rejects any call where a hook exists anywhere in the tree; the reference path is the
thing such a call lands on, so it runs each bypassed module's forward pre-hooks and
forward hooks itself, per-module and global, through `_call_with_hooks`. Round 0's tests
used an identity hook and could not tell the difference.

Two inherited behaviours are kept rather than "fixed"
-----------------------------------------------------
  * `initialize_weights` tests `isinstance(module, Linear)` where `Linear` is the
    frozen class while the blocks' linears are the baseline class, so `_basic_init`
    no longer reaches them.  Harmless -- `load_state_dict` overwrites every one of
    those tensors before the first forward -- but it is a real difference from
    `baseline.py` and is left visible rather than papered over.
  * `_modulate` computes `scale`'s repeat factor from `shift.shape[0]` *after*
    reassigning `shift`.  At these shapes the factor is 1 either way, so it is
    harmless; the reference path reproduces it rather than correcting it.

Nothing is precomputed from a parameter in `__init__`: the harness casts dtype and
calls `load_state_dict` *after* construction, so anything derived there would be
stale.  The rotary tables, the sinusoid frequency vector and the structural guard
frequency vector are built lazily on first use and cached on the *instance* (the
structural and configuration snapshots are taken at the end of `__init__`, where a
lazy one would bake in a pre-first-call mutation) -- and never at module level, because
the bench builds five modules in one worker process with
`gc.collect()` and `empty_cache()` between cases, so the caching allocator can hand
a freed `freqs` address to a later instance and a module-level cache keyed on
`data_ptr` could then serve a stale table.
"""

from __future__ import annotations

import math
import os
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import forward_ad
from torch.nn.modules import module as _module_hooks

from ..L1.linear import Linear
# The *baseline* rotary class, not the frozen `..L1.oasis_rotary` winner, and this is
# measured rather than preferred. The two are bit-identical in fp32 (their axial tables
# compare equal, and `profile/isolate_frozen.log` scores the substitution at 1.0000 /
# max_abs 0.00e+00), and they carry the same `state_dict` (`['freqs']`, with `dummy`
# non-persistent), so nothing about AC-1 or AC-2 depends on the choice. Under an active
# autocast they diverge: the frozen winner builds its table with a raw pybind kernel
# that sits below the dispatcher and never sees autocast, so it stays fp32, while the
# baseline's `einsum` is on autocast's cast list and returns bf16 -- a 1.7-absolute
# difference in the table that then feeds all 32 attention calls. Because the 16 blocks
# in this tree are the baseline's own objects, they would call whichever instance is
# constructed here, so the frozen one made the whole model disagree with the baseline
# under autocast, which AC-4.2 forbids. `profile/autocast_probe.py` / `.log` measures
# both the divergence and its absence after this change.
from fastkernels.tasks.baseline.L1.oasis_rotary import OasisRotaryEmbedding
from ..L2.oasis_final_layer import OasisFinalLayer
from ..L2.oasis_patch_embed import OasisPatchEmbed
from ..L2.oasis_timestep_embedder import OasisTimestepEmbedder
from .oasis_block import SpatioTemporalDiTBlock

# Unique to this workspace: the name keys both the ninja build lock and the
# resulting .so, so it must not collide with any other extension in
# `.torch_extensions/` (`fk_cand_oasis_rotary_table`,
# `fk_cand_l2_oasis_spatial_axial`, `fk_cand_oasis_temporal_axial_fused`,
# `fk_oasis_tse_*`, `fk_gelu_*`, `fk_silu_sm100_v1`).
_EXTENSION_NAME = "fk_cand_l3_oasis_dit_fused"

_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>

namespace {

constexpr int kBlockSize = 256;
constexpr int64_t kMaxBlocks = 8192;

// Four floats, one 128-bit access. Every tensor these kernels touch is freshly
// allocated by the caller and has a hidden extent divisible by 4, so the wide
// access is always in bounds and always aligned; the C++ entry points verify both
// rather than assume them, and reject with TORCH_CHECK_VALUE (a Python
// ValueError), so a violated invariant surfaces instead of faulting. See the note
// above `CHECK_FUSED_F32` for why `_fast_forward` does not catch it.
struct alignas(16) F4 {
  float x, y, z, w;
};

inline int grid_for(int64_t work) {
  const int64_t wanted = (work + kBlockSize - 1) / kBlockSize;
  return static_cast<int>(wanted < kMaxBlocks ? wanted : kMaxBlocks);
}

// y = rn(rn(n * rn(1 + scale)) + shift), with shift/scale read as byte offsets into
// one projection output of shape [frames, groups * hidden] and broadcast over the
// tokens of a frame.
//
// The three roundings are the reference's: `1 + scale` is a whole tensor in the
// reference (`x * (1 + scale) + shift`), then the multiply, then the add. Spelling
// any two of them as one FMA is measured at 4.768e-07, which the error budget makes
// fatal -- hence the intrinsics here and -fmad=false on the command line.
__global__ void modulate_kernel(
    const float* __restrict__ n,
    const float* __restrict__ m,
    float* __restrict__ y,
    const int64_t vec_per_row,
    const int64_t total_vec,
    const int64_t tokens_per_frame,
    const int64_t m_row,
    const int64_t shift_off,
    const int64_t scale_off) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total_vec; i += stride) {
    const int64_t tok = i / vec_per_row;
    const int64_t d4 = i - tok * vec_per_row;
    const int64_t base = (tok / tokens_per_frame) * m_row + d4 * 4;
    const F4 nv = reinterpret_cast<const F4*>(n)[i];
    const F4 sc = *reinterpret_cast<const F4*>(m + base + scale_off);
    const F4 sh = *reinterpret_cast<const F4*>(m + base + shift_off);
    F4 out;
    out.x = __fadd_rn(__fmul_rn(nv.x, __fadd_rn(1.0f, sc.x)), sh.x);
    out.y = __fadd_rn(__fmul_rn(nv.y, __fadd_rn(1.0f, sc.y)), sh.y);
    out.z = __fadd_rn(__fmul_rn(nv.z, __fadd_rn(1.0f, sc.z)), sh.z);
    out.w = __fadd_rn(__fmul_rn(nv.w, __fadd_rn(1.0f, sc.w)), sh.w);
    reinterpret_cast<F4*>(y)[i] = out;
  }
}

// x = rn(x + rn(gate * o)), in place on the residual stream.
//
// The reference builds a new tensor (`x = x + _gate(...)`); updating in place is
// the same arithmetic on a buffer this file allocated and owns, and it saves an
// allocation per call without changing a value.
__global__ void gate_residual_kernel(
    float* __restrict__ x,
    const float* __restrict__ o,
    const float* __restrict__ m,
    const int64_t vec_per_row,
    const int64_t total_vec,
    const int64_t tokens_per_frame,
    const int64_t m_row,
    const int64_t gate_off) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total_vec; i += stride) {
    const int64_t tok = i / vec_per_row;
    const int64_t d4 = i - tok * vec_per_row;
    const F4 g = *reinterpret_cast<const F4*>(
        m + (tok / tokens_per_frame) * m_row + gate_off + d4 * 4);
    const F4 ov = reinterpret_cast<const F4*>(o)[i];
    F4* dst = reinterpret_cast<F4*>(x) + i;
    F4 xv = *dst;
    xv.x = __fadd_rn(xv.x, __fmul_rn(g.x, ov.x));
    xv.y = __fadd_rn(xv.y, __fmul_rn(g.y, ov.y));
    xv.z = __fadd_rn(xv.z, __fmul_rn(g.z, ov.z));
    xv.w = __fadd_rn(xv.w, __fmul_rn(g.w, ov.w));
    *dst = xv;
  }
}

// The rotation `oasis_apply_rotary_emb` actually performs. `get_axial_freqs` (and
// the temporal `_forward_freqs`) end in `repeat_interleave(2, -1)`, so each
// frequency occupies an *adjacent pair* of lanes, and `oasis_rotate_half` reshapes
// to (..., -1, 2) and stacks `(-x2, x1)`. Per pair p:
//
//   out[2p]   = rn( rn(t[2p]   * cos[2p])   + rn(-t[2p+1] * sin[2p])   )
//   out[2p+1] = rn( rn(t[2p+1] * cos[2p+1]) + rn( t[2p]   * sin[2p+1]) )
//
// Both lanes of a pair carry the same frequency -- `profile/rope_probe.log` reports
// `pair lanes identical: True` for both axes -- so reading each lane's own table
// entry, as below, is free of any assumption about that. The split-half convention
// that most rotary code uses is wrong here by up to 6.3 absolute (same log).
__device__ __forceinline__ F4 rotate_pairs(const F4 t, const F4 c, const F4 s) {
  F4 o;
  o.x = __fadd_rn(__fmul_rn(t.x, c.x), __fmul_rn(-t.y, s.x));
  o.y = __fadd_rn(__fmul_rn(t.y, c.y), __fmul_rn(t.x, s.y));
  o.z = __fadd_rn(__fmul_rn(t.z, c.z), __fmul_rn(-t.w, s.z));
  o.w = __fadd_rn(__fmul_rn(t.w, c.w), __fmul_rn(t.z, s.w));
  return o;
}

// Read [tokens, 3 * hidden] and write [2 or 3, frames, heads, n_spatial, head_dim],
// rotating q and k against a [n_spatial, head_dim] table and, when `pack_v`, copying
// v through as a third slice.
//
// The linear index enumerates (frame, head, spatial, head_dim/4) in exactly the
// output's nesting order, so the store offset is `i * 4` with no address algebra.
// `profile/sdpa_probe.py` / `.log` confirms the q and k slices are bit-identical to
// the tensors the reference's reshape/rotary/transpose/permute chain hands SDPA and
// carry the same strides. `v` is *not* packed on the fast path: the reference hands
// SDPA a strided view of `qkv`, so the caller rebuilds that view instead, which is
// what makes the strides equivalent rather than merely the values equal. `pack_v`
// remains because the passthrough is a contractual behaviour of this kernel and is
// tested as one.
__global__ void rope_pack_spatial_kernel(
    const float* __restrict__ qkv,
    const float* __restrict__ cos_t,
    const float* __restrict__ sin_t,
    float* __restrict__ out,
    const int64_t heads,
    const int64_t n_spatial,
    const int64_t vec_per_head,
    const int64_t head_dim,
    const int64_t hidden,
    const int64_t total_vec,
    const int64_t slice_stride,
    const bool pack_v) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total_vec; i += stride) {
    int64_t rest = i / vec_per_head;
    const int64_t d4 = (i - rest * vec_per_head) * 4;
    const int64_t s = rest % n_spatial;
    rest /= n_spatial;
    const int64_t head = rest % heads;
    const int64_t frame = rest / heads;

    const F4 c = *reinterpret_cast<const F4*>(cos_t + s * head_dim + d4);
    const F4 sn = *reinterpret_cast<const F4*>(sin_t + s * head_dim + d4);
    const float* src = qkv + (frame * n_spatial + s) * 3 * hidden + head * head_dim + d4;
    F4* dst = reinterpret_cast<F4*>(out) + i;

    dst[0] = rotate_pairs(*reinterpret_cast<const F4*>(src), c, sn);
    dst[slice_stride] = rotate_pairs(
        *reinterpret_cast<const F4*>(src + hidden), c, sn);
    if (pack_v) {
      dst[2 * slice_stride] = *reinterpret_cast<const F4*>(src + 2 * hidden);
    }
  }
}

// Read [tokens, 3 * hidden] and write [3, n_spatial, heads, frames, head_dim],
// rotating q and k against a [frames, head_dim] table and copying v through.
__global__ void rope_pack_temporal_kernel(
    const float* __restrict__ qkv,
    const float* __restrict__ cos_t,
    const float* __restrict__ sin_t,
    float* __restrict__ out,
    const int64_t heads,
    const int64_t frames,
    const int64_t n_spatial,
    const int64_t vec_per_head,
    const int64_t head_dim,
    const int64_t hidden,
    const int64_t total_vec,
    const int64_t slice_stride,
    const bool pack_v) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total_vec; i += stride) {
    int64_t rest = i / vec_per_head;
    const int64_t d4 = (i - rest * vec_per_head) * 4;
    const int64_t frame = rest % frames;
    rest /= frames;
    const int64_t head = rest % heads;
    const int64_t s = rest / heads;

    const F4 c = *reinterpret_cast<const F4*>(cos_t + frame * head_dim + d4);
    const F4 sn = *reinterpret_cast<const F4*>(sin_t + frame * head_dim + d4);
    const float* src = qkv + (frame * n_spatial + s) * 3 * hidden + head * head_dim + d4;
    F4* dst = reinterpret_cast<F4*>(out) + i;

    dst[0] = rotate_pairs(*reinterpret_cast<const F4*>(src), c, sn);
    dst[slice_stride] = rotate_pairs(
        *reinterpret_cast<const F4*>(src + hidden), c, sn);
    if (pack_v) {
      dst[2 * slice_stride] = *reinterpret_cast<const F4*>(src + 2 * hidden);
    }
  }
}

// Guards use TORCH_CHECK_VALUE, which surfaces as a Python ValueError rather than a
// RuntimeError, so a caller can tell "this input is not for me" from a real failure
// while an out-of-memory or a CUDA fault still propagates.
//
// `_fast_forward` does *not* catch it, and that is deliberate rather than an
// oversight: every operand it passes is a tensor it allocated itself on the line
// before -- `F.conv2d`, `F.layer_norm` and `F.linear` outputs, and the one packed
// residual stream -- so all of these conditions are internal invariants and a
// violation is a bug in this file, which should surface rather than be silently
// papered over by a slower path that computes the same thing. Choosing between the
// kernels and the torch spelling happens once, at the top of `_fast_forward`, on
// whether the extension built at all.
//
// `data_ptr()` already includes the tensor's storage offset, so an offset view is
// served correctly rather than rejected -- what matters is 16-byte alignment of the
// resulting address, which is checked.
#define CHECK_FUSED_F32(t, name)                                              \
  TORCH_CHECK_VALUE((t).is_cuda(), name " must be a CUDA tensor");             \
  TORCH_CHECK_VALUE((t).scalar_type() == at::ScalarType::Float,                \
                    name " must be float32");                                  \
  TORCH_CHECK_VALUE((t).is_contiguous(), name " must be contiguous");          \
  TORCH_CHECK_VALUE(!(t).is_neg() && !(t).is_conj(),                           \
                    name " must read as stored");                              \
  TORCH_CHECK_VALUE(                                                          \
      reinterpret_cast<uintptr_t>((t).data_ptr()) % 16 == 0,                    \
      name " must be 16-byte aligned")

}  // namespace

at::Tensor oasis_dit_modulate(
    const at::Tensor& n, const at::Tensor& m,
    const int64_t shift_off, const int64_t scale_off) {
  CHECK_FUSED_F32(n, "modulate: n");
  CHECK_FUSED_F32(m, "modulate: m");
  TORCH_CHECK_VALUE(n.dim() == 2 && m.dim() == 2, "modulate: needs 2-D operands");
  TORCH_CHECK_VALUE(n.device() == m.device(), "modulate: operands on one device");
  TORCH_CHECK_VALUE(!(at::GradMode::is_enabled() &&
                      (n.requires_grad() || m.requires_grad())),
                    "modulate: fused path is inference-only");
  const int64_t tokens = n.size(0);
  const int64_t hidden = n.size(1);
  const int64_t frames = m.size(0);
  const int64_t m_row = m.size(1);
  TORCH_CHECK_VALUE(hidden % 4 == 0 && m_row % 4 == 0,
                    "modulate: hidden and projection row must be divisible by 4");
  TORCH_CHECK_VALUE(frames > 0 && tokens % frames == 0,
                    "modulate: tokens must divide by frames");
  TORCH_CHECK_VALUE(shift_off >= 0 && scale_off >= 0
                        && shift_off % 4 == 0 && scale_off % 4 == 0
                        && shift_off + hidden <= m_row
                        && scale_off + hidden <= m_row,
                    "modulate: offsets out of range or unaligned");

  const c10::cuda::CUDAGuard guard(n.device());
  at::Tensor y = at::empty({tokens, hidden}, n.options());
  const int64_t total_vec = tokens * (hidden / 4);
  if (total_vec == 0) {
    return y;
  }
  modulate_kernel<<<grid_for(total_vec), kBlockSize, 0,
                    at::cuda::getCurrentCUDAStream()>>>(
      n.data_ptr<float>(), m.data_ptr<float>(), y.data_ptr<float>(),
      hidden / 4, total_vec, tokens / frames, m_row, shift_off, scale_off);
  AT_CUDA_CHECK(cudaGetLastError());
  return y;
}

void oasis_dit_gate_residual_(
    at::Tensor& x, const at::Tensor& o, const at::Tensor& m,
    const int64_t gate_off) {
  CHECK_FUSED_F32(x, "gate_residual: x");
  CHECK_FUSED_F32(o, "gate_residual: o");
  CHECK_FUSED_F32(m, "gate_residual: m");
  TORCH_CHECK_VALUE(x.dim() == 2 && o.dim() == 2 && m.dim() == 2,
                    "gate_residual: needs 2-D operands");
  TORCH_CHECK_VALUE(x.sizes() == o.sizes(), "gate_residual: x and o must agree");
  TORCH_CHECK_VALUE(x.device() == o.device() && x.device() == m.device(),
                    "gate_residual: operands on one device");
  TORCH_CHECK_VALUE(!(at::GradMode::is_enabled() &&
                      (x.requires_grad() || o.requires_grad() || m.requires_grad())),
                    "gate_residual: fused path is inference-only");
  // `x` is written through a `__restrict__` pointer while `o` and `m` are read
  // through others, so any *overlap* between the written range and either read
  // range is a race and a violated restrict contract -- not only the case where
  // the two base pointers coincide. Shifted, still-16-byte-aligned views of one
  // buffer would pass a base-pointer comparison and corrupt each other.
  // Compared as integers, not as pointers: relational operators on `char*` from
  // two unrelated allocations are unspecified in C++, and this is a public entry
  // point that a caller could reach with tensors from different blocks.
  const auto ranges_overlap = [](const at::Tensor& a, const at::Tensor& b) {
    const auto pa = reinterpret_cast<uintptr_t>(a.data_ptr());
    const auto pb = reinterpret_cast<uintptr_t>(b.data_ptr());
    const auto na = static_cast<uintptr_t>(a.numel()) * a.element_size();
    const auto nb = static_cast<uintptr_t>(b.numel()) * b.element_size();
    return pa < pb + nb && pb < pa + na;
  };
  TORCH_CHECK_VALUE(!ranges_overlap(x, o) && !ranges_overlap(x, m),
                    "gate_residual: the written tensor must not overlap either "
                    "read operand");
  const int64_t tokens = x.size(0);
  const int64_t hidden = x.size(1);
  const int64_t frames = m.size(0);
  const int64_t m_row = m.size(1);
  TORCH_CHECK_VALUE(hidden % 4 == 0 && m_row % 4 == 0,
                    "gate_residual: hidden and projection row must divide by 4");
  TORCH_CHECK_VALUE(frames > 0 && tokens % frames == 0,
                    "gate_residual: tokens must divide by frames");
  TORCH_CHECK_VALUE(gate_off >= 0 && gate_off % 4 == 0 && gate_off + hidden <= m_row,
                    "gate_residual: offset out of range or unaligned");

  const int64_t total_vec = tokens * (hidden / 4);
  if (total_vec == 0) {
    return;
  }
  const c10::cuda::CUDAGuard guard(x.device());
  gate_residual_kernel<<<grid_for(total_vec), kBlockSize, 0,
                         at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), o.data_ptr<float>(), m.data_ptr<float>(),
      hidden / 4, total_vec, tokens / frames, m_row, gate_off);
  AT_CUDA_CHECK(cudaGetLastError());
}

namespace {

void check_rope_operands(const at::Tensor& qkv, const at::Tensor& cos_t,
                         const at::Tensor& sin_t) {
  CHECK_FUSED_F32(qkv, "rope_pack: qkv");
  CHECK_FUSED_F32(cos_t, "rope_pack: cos");
  CHECK_FUSED_F32(sin_t, "rope_pack: sin");
  TORCH_CHECK_VALUE(qkv.dim() == 2 && cos_t.dim() == 2 && sin_t.dim() == 2,
                    "rope_pack: needs 2-D operands");
  TORCH_CHECK_VALUE(cos_t.sizes() == sin_t.sizes(),
                    "rope_pack: cos and sin must agree");
  TORCH_CHECK_VALUE(qkv.device() == cos_t.device() && qkv.device() == sin_t.device(),
                    "rope_pack: operands on one device");
  TORCH_CHECK_VALUE(!(at::GradMode::is_enabled() &&
                      (qkv.requires_grad() || cos_t.requires_grad()
                       || sin_t.requires_grad())),
                    "rope_pack: fused path is inference-only");
  TORCH_CHECK_VALUE(qkv.size(1) % 3 == 0, "rope_pack: qkv must be [tokens, 3*hidden]");
}

}  // namespace

at::Tensor oasis_dit_rope_pack_spatial(
    const at::Tensor& qkv, const at::Tensor& cos_t, const at::Tensor& sin_t,
    const bool pack_v) {
  check_rope_operands(qkv, cos_t, sin_t);
  const int64_t tokens = qkv.size(0);
  const int64_t hidden = qkv.size(1) / 3;
  const int64_t n_spatial = cos_t.size(0);
  const int64_t head_dim = cos_t.size(1);
  TORCH_CHECK_VALUE(head_dim % 4 == 0 && head_dim > 0,
                    "rope_pack_spatial: head_dim must be a positive multiple of 4");
  TORCH_CHECK_VALUE(hidden % head_dim == 0,
                    "rope_pack_spatial: hidden must divide by head_dim");
  TORCH_CHECK_VALUE(n_spatial > 0 && tokens % n_spatial == 0,
                    "rope_pack_spatial: tokens must divide by n_spatial");
  const int64_t heads = hidden / head_dim;
  const int64_t frames = tokens / n_spatial;

  const c10::cuda::CUDAGuard guard(qkv.device());
  // `pack_v == false` allocates two slices, not three: the caller then hands SDPA
  // the reference's own strided `v` view of `qkv` instead of a packed copy, which is
  // what makes the strides SDPA receives equivalent to the reference's. The three-
  // slice form is kept because the `v` passthrough is a contractual behaviour of this
  // kernel and is tested as one.
  at::Tensor out = at::empty({pack_v ? 3 : 2, frames, heads, n_spatial, head_dim},
                             qkv.options());
  const int64_t slice_stride = frames * heads * n_spatial * (head_dim / 4);
  if (slice_stride == 0) {
    return out;
  }
  rope_pack_spatial_kernel<<<grid_for(slice_stride), kBlockSize, 0,
                             at::cuda::getCurrentCUDAStream()>>>(
      qkv.data_ptr<float>(), cos_t.data_ptr<float>(), sin_t.data_ptr<float>(),
      out.data_ptr<float>(), heads, n_spatial, head_dim / 4, head_dim, hidden,
      slice_stride, slice_stride, pack_v);
  // Attribute a launch failure here rather than at some later synchronization in
  // unrelated code.
  AT_CUDA_CHECK(cudaGetLastError());
  return out;
}

at::Tensor oasis_dit_rope_pack_temporal(
    const at::Tensor& qkv, const at::Tensor& cos_t, const at::Tensor& sin_t,
    const bool pack_v) {
  check_rope_operands(qkv, cos_t, sin_t);
  const int64_t tokens = qkv.size(0);
  const int64_t hidden = qkv.size(1) / 3;
  const int64_t frames = cos_t.size(0);
  const int64_t head_dim = cos_t.size(1);
  TORCH_CHECK_VALUE(head_dim % 4 == 0 && head_dim > 0,
                    "rope_pack_temporal: head_dim must be a positive multiple of 4");
  TORCH_CHECK_VALUE(hidden % head_dim == 0,
                    "rope_pack_temporal: hidden must divide by head_dim");
  TORCH_CHECK_VALUE(frames > 0 && tokens % frames == 0,
                    "rope_pack_temporal: tokens must divide by frames");
  const int64_t heads = hidden / head_dim;
  const int64_t n_spatial = tokens / frames;

  const c10::cuda::CUDAGuard guard(qkv.device());
  at::Tensor out = at::empty({pack_v ? 3 : 2, n_spatial, heads, frames, head_dim},
                             qkv.options());
  const int64_t slice_stride = frames * heads * n_spatial * (head_dim / 4);
  if (slice_stride == 0) {
    return out;
  }
  rope_pack_temporal_kernel<<<grid_for(slice_stride), kBlockSize, 0,
                              at::cuda::getCurrentCUDAStream()>>>(
      qkv.data_ptr<float>(), cos_t.data_ptr<float>(), sin_t.data_ptr<float>(),
      out.data_ptr<float>(), heads, frames, n_spatial, head_dim / 4, head_dim,
      hidden, slice_stride, slice_stride, pack_v);
  AT_CUDA_CHECK(cudaGetLastError());
  return out;
}
"""

_CPP_SOURCE = r"""
#include <torch/extension.h>

at::Tensor oasis_dit_modulate(const at::Tensor& n, const at::Tensor& m,
                              int64_t shift_off, int64_t scale_off);
void oasis_dit_gate_residual_(at::Tensor& x, const at::Tensor& o,
                              const at::Tensor& m, int64_t gate_off);
at::Tensor oasis_dit_rope_pack_spatial(const at::Tensor& qkv, const at::Tensor& cos_t,
                                       const at::Tensor& sin_t, bool pack_v);
at::Tensor oasis_dit_rope_pack_temporal(const at::Tensor& qkv, const at::Tensor& cos_t,
                                        const at::Tensor& sin_t, bool pack_v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("modulate", &oasis_dit_modulate,
        "y = rn(rn(n * rn(1 + scale)) + shift), broadcasting a projection row");
  m.def("gate_residual_", &oasis_dit_gate_residual_,
        "x += rn(gate * o), in place on the residual stream");
  m.def("rope_pack_spatial", &oasis_dit_rope_pack_spatial,
        "Rotate q/k on adjacent pairs and pack [2 or 3, frames, heads, n_spatial, d]",
        py::arg("qkv"), py::arg("cos"), py::arg("sin"), py::arg("pack_v") = true);
  m.def("rope_pack_temporal", &oasis_dit_rope_pack_temporal,
        "Rotate q/k on adjacent pairs and pack [2 or 3, n_spatial, heads, frames, d]",
        py::arg("qkv"), py::arg("cos"), py::arg("sin"), py::arg("pack_v") = true);
}
"""


def _local_arch_list() -> str | None:
    """Local compute capability, in the form nvcc wants for this build.

    Compute capabilities 9.0 and up need the architecture-specific `a` variant.
    Returning None leaves `TORCH_CUDA_ARCH_LIST` alone, which is the right thing
    when the capability cannot be read.
    """
    if not torch.cuda.is_available():
        return None
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:  # pragma: no cover - driver dependent
        return None
    return f"{major}.{minor}a" if major >= 9 else f"{major}.{minor}"


def _build_fused_extension():
    """Compile the four kernels into the workspace-local build directory.

    `-fmad=false` is the load-bearing flag: it forbids the compiler from
    contracting any multiply-add anywhere in this translation unit. The
    intrinsics in the source already spell each rounding, so the flag is
    belt-and-braces -- but a contraction is priced at 4.768e-07 and the error
    budget makes that fatal, so one guarantee is not enough.

    The environment ships a multi-architecture arch list; compiling all of it
    would cost minutes of wall clock for kernels that only run on this GPU.
    """
    from torch.utils.cpp_extension import load_inline

    # Exists so a *genuine* build failure can be tested. `OASIS_DIT_FUSED=0` skips the
    # build entirely and so never reaches the `except` clause below; this raises from
    # inside it, which is the only way to check that the exception is recorded in
    # `FUSED_BUILD_ERROR` and that all five shapes stay bit-exact without the kernels.
    if os.environ.get("OASIS_DIT_FORCE_BUILD_ERROR", "0") == "1":
        raise RuntimeError("OASIS_DIT_FORCE_BUILD_ERROR=1: simulated nvcc failure")

    build_dir = Path(__file__).resolve().parents[2] / ".torch_extensions" / _EXTENSION_NAME
    build_dir.mkdir(parents=True, exist_ok=True)

    arch = _local_arch_list()
    previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if arch is not None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    try:
        return load_inline(
            name=_EXTENSION_NAME,
            cpp_sources=[_CPP_SOURCE],
            cuda_sources=[_CUDA_SOURCE],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-fmad=false", "--expt-relaxed-constexpr"],
            build_directory=str(build_dir),
            verbose=False,
        )
    finally:
        if arch is not None:
            if previous is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = previous


#: Which of the restructuring's stages to run, so each can be benchmarked by the
#: official harness rather than described. Read once, here, so no call pays for it and
#: no timed region can see it change.
#:
#:   ``reference``    the reference path only: the two front-end bodies and the final
#:                    layer transcribed, blocks called as the baseline calls them.
#:   ``torch``        the restructuring with the torch spelling of all four kernels.
#:                    Also what a failed build leaves behind.
#:   ``elementwise``  adds the ``modulate`` and ``gate_residual`` kernels; the rotary
#:                    pack stays on torch.
#:   ``full``         all four kernels, with 32 separate adaLN projections. The M3
#:                    configuration, kept for comparison.
#:
#: `OASIS_DIT_DISABLE_FAST_PATH=1` and `OASIS_DIT_FUSED=0` are kept as they were and
#: are equivalent to ``reference`` and ``torch`` respectively.
#:   ``walk``         ``full`` but resolving the per-axis modules by attribute walk on
#:                    every call instead of from the cached plan, so `validate.py` can
#:                    price the pre-resolution.
#:   ``concat_adaln`` ``full`` plus one concatenated [32*6*hidden, hidden] adaLN GEMM.
#:                    Bit-exact and ~0.37 ms faster than ``full`` at every captured
#:                    shape through `validate.py`.
#:   ``graph``        ``concat_adaln`` behind a per-`T` captured CUDA graph, returning a
#:                    fresh non-aliased output. **The default and the deliverable.**
#:                    Bit-exact and a further 1.12-1.19x through `validate.py`
#:                    (`profile/bench_stage_graph.json`). Each candidate ships only
#:                    because it stayed bit-exact *and* won on the official harness;
#:                    every stage above it is kept so the chain stays priceable.
_STAGES = ("reference", "torch", "elementwise", "full", "walk", "concat_adaln",
           "graph")
STAGE = os.environ.get("OASIS_DIT_STAGE", "graph")
if STAGE not in _STAGES:
    raise ValueError(f"OASIS_DIT_STAGE must be one of {_STAGES}, got {STAGE!r}")

#: Set `OASIS_DIT_DISABLE_FAST_PATH=1` to force every call down the reference path.
DISABLE_FAST_PATH = (os.environ.get("OASIS_DIT_DISABLE_FAST_PATH", "0") == "1"
                     or STAGE == "reference")

#: Why the extension is not available, if it is not. A build failure must cost
#: speed rather than correctness, and must leave its reason visible rather than be
#: swallowed: the restructured torch path still runs without the kernels, and the
#: reference path still runs without either.
FUSED_BUILD_ERROR: str | None = None

_FUSED = None
if DISABLE_FAST_PATH:
    FUSED_BUILD_ERROR = "not built: reference path forced"
elif os.environ.get("OASIS_DIT_FUSED", "1") == "0" or STAGE == "torch":
    FUSED_BUILD_ERROR = "not built: torch spelling selected"
else:
    try:
        _FUSED = _build_fused_extension()
    except Exception as exc:  # pragma: no cover - build environment dependent
        FUSED_BUILD_ERROR = f"{type(exc).__name__}: {exc}"

#: Whether the fused kernels are live. A benchmark taken with this False measured
#: the torch spelling of the restructuring, not the kernels.
FUSED_AVAILABLE = _FUSED is not None

#: Host-side count of calls served by the fast path, across every instance. The
#: predicate can decline silently, and a silently-declining predicate produces an
#: honest-looking 1.0x that is otherwise indistinguishable from a regression -- so
#: the tests assert on this rather than inferring it from a latency.
_FASTPATH_HITS = 0

# Tensor types whose logical value is the bytes in their storage. A `Parameter`
# belongs here -- it is a subclass, but it adds only autograd bookkeeping and no
# dispatch behaviour -- while any other subclass may implement `__torch_dispatch__`
# and compute its value from something else entirely.
_STORAGE_FAITHFUL_TYPES = (torch.Tensor, nn.Parameter)


def _call_with_hooks(module: nn.Module, body, *args, **kwargs):
    """Run `body` as `nn.Module._call_impl` would run `module.forward`.

    Every body this file transcribes replaces a module's `forward`, which means the
    module's own hooks -- and the global registries -- would never run. The fast path
    is safe there because the predicate rejects any call where a hook exists at all;
    the *reference* path is not, because it is the fallback that such a call lands on.
    AC-4.2 requires a registered hook to fall back **and still agree**, so the
    fallback has to reproduce the hook semantics, not just the arithmetic.

    Transcribed from the installed `Module._call_impl`, in its order, including:

      * the fast exit when no hook of any kind is registered;
      * global pre-hooks before per-module ones, and the same for post-hooks;
      * the `with_kwargs` variants of both, read only from the registries this torch
        actually has (there is no `_global_forward_pre_hooks_with_kwargs`);
      * PyTorch's own validation that a `with_kwargs` pre-hook returns `None` or a
        2-tuple, raising the same `RuntimeError` with the same message shape;
      * the non-kwargs pre-hook's single-value-to-tuple promotion;
      * `always_call=True` post-hooks re-run from an `except` clause when the body
        raises, each guarded so its own failure is warned about rather than replacing
        the original exception, and then the original re-raised.

    Backward hooks are deliberately not reproduced: they change what the reference
    builds for the backward pass rather than the values it returns, and both paths only
    run with grad disabled. `_slow_forward` under a tracing state is likewise out of
    scope -- the predicate rejects a non-empty dispatch or function stack, and the
    reference path here is not being traced.

    Because a legal `with_kwargs` pre-hook may *remap a positional argument to a
    keyword*, every body passed here must take the real forward's parameter names:
    `input` for `Linear`, `x` for `Conv2d`/`LayerNorm`/`SiLU`, `x, random_sample` for
    `OasisPatchEmbed`, `t` for `OasisTimestepEmbedder`, `x, c` for `OasisFinalLayer`.
    A hook returning `(), {"input": t}` succeeds on the baseline, and before those
    names matched it raised `TypeError` here.
    """
    pre_hooks = (*_module_hooks._global_forward_pre_hooks.items(),
                 *module._forward_pre_hooks.items())
    post_hooks = (*_module_hooks._global_forward_hooks.items(),
                  *module._forward_hooks.items())
    if not pre_hooks and not post_hooks:
        return body(*args, **kwargs)

    result = None
    called_always_called = set()
    global_post_with_kwargs = getattr(
        _module_hooks, "_global_forward_hooks_with_kwargs", {})
    global_always_called = getattr(
        _module_hooks, "_global_forward_hooks_always_called", set())

    def inner():
        nonlocal result, args, kwargs
        pre_with_kwargs = module._forward_pre_hooks_with_kwargs
        for hook_id, hook in pre_hooks:
            if hook_id in pre_with_kwargs:
                args_kwargs_result = hook(module, args, kwargs)
                if args_kwargs_result is not None:
                    if (isinstance(args_kwargs_result, tuple)
                            and len(args_kwargs_result) == 2):
                        args, kwargs = args_kwargs_result
                    else:
                        raise RuntimeError(
                            "forward pre-hook must return None or a tuple "
                            f"of (new_args, new_kwargs), but got "
                            f"{args_kwargs_result}.")
            else:
                args_result = hook(module, args)
                if args_result is not None:
                    if not isinstance(args_result, tuple):
                        args_result = (args_result,)
                    args = args_result

        result = body(*args, **kwargs)
        post_with_kwargs = module._forward_hooks_with_kwargs
        always_called = module._forward_hooks_always_called
        for hook_id, hook in post_hooks:
            if hook_id in always_called or hook_id in global_always_called:
                called_always_called.add(hook_id)
            if hook_id in post_with_kwargs or hook_id in global_post_with_kwargs:
                hook_result = hook(module, args, kwargs, result)
            else:
                hook_result = hook(module, args, result)
            if hook_result is not None:
                result = hook_result
        return result

    try:
        return inner()
    except Exception:
        for registry, always, with_kw in (
                (_module_hooks._global_forward_hooks, global_always_called,
                 global_post_with_kwargs),
                (module._forward_hooks, module._forward_hooks_always_called,
                 module._forward_hooks_with_kwargs)):
            for hook_id, hook in registry.items():
                if hook_id in always and hook_id not in called_always_called:
                    try:
                        if hook_id in with_kw:
                            hook_result = hook(module, args, kwargs, result)
                        else:
                            hook_result = hook(module, args, result)
                        if hook_result is not None:
                            result = hook_result
                    except Exception as exc:  # noqa: BLE001 - mirrors _call_impl
                        warnings.warn(
                            "module forward hook with ``always_call=True`` raised an "
                            "exception that was silenced as another error was raised "
                            f"in forward: {exc}", stacklevel=2)
                        continue
        raise


# --- transcribed forwards, named after the forwards they replace --------------
# A `with_kwargs=True` pre-hook may legally remap a positional argument to a keyword,
# so each of these takes its original's parameter names. They are built per call site
# rather than bound once because they close over the module whose parameters they read,
# and those are re-read live on every call.

def _linear_body(module: nn.Module):
    def forward(input):  # noqa: A002 - `Linear.forward`'s own parameter name
        return F.linear(input, module.weight, module.bias)
    return forward


def _silu_body(module: nn.Module):
    def forward(x):
        return F.silu(x)
    return forward


def _layer_norm_body(module: nn.Module):
    def forward(x):
        # `LayerNorm.forward` verbatim, including the `promote_fp32` round trip: it
        # reduces in fp32 and casts the *result* back to the input's dtype, so dropping
        # either step adds or removes a rounding. In fp32 -- the only dtype the harness
        # runs -- `.float()` and `.to(float32)` both return `self` and this is exactly
        # `F.layer_norm(x, ...)`. Under an active autocast they are not no-ops, and
        # omitting them was worth 1.6e-02 (`profile/autocast_probe.py` / `.log`).
        if not getattr(module, "promote_fp32", False):
            return F.layer_norm(x, module.normalized_shape, module.weight,
                                module.bias, module.eps)
        weight, bias = module.weight, module.bias
        if weight is not None and weight.dtype is not torch.float32:
            weight = weight.float()
        if bias is not None and bias.dtype is not torch.float32:
            bias = bias.float()
        return F.layer_norm(x.float(), module.normalized_shape, weight, bias,
                            module.eps).to(x.dtype)
    return forward


def _conv2d_body(module: nn.Module):
    def forward(x):
        return F.conv2d(x, module.weight, module.bias, stride=module.stride,
                        padding=module.padding, dilation=module.dilation,
                        groups=module.groups)
    return forward


def _reads_as_stored(t: torch.Tensor) -> bool:
    """Whether a tensor's logical values equal the bytes in its storage.

    `neg` and `conj` views carry a lazy flag that ATen applies when it reads them,
    so the values a PyTorch op sees are not the values in memory. A kernel that
    dereferences the pointer never sees that flag.
    """
    return not t.is_neg() and not t.is_conj()


# --- torch spellings of the four kernels -------------------------------------
# These are the fallback when the extension is unavailable, and they are also what
# the per-kernel tests compare the kernels against. Each reproduces the reference's
# rounding boundaries with separate ATen ops, so no contraction can appear.

def _modulate_torch(n: torch.Tensor, m: torch.Tensor, shift_off: int,
                    scale_off: int) -> torch.Tensor:
    tokens, hidden = n.shape
    frames = m.shape[0]
    shift = m[:, shift_off:shift_off + hidden].unsqueeze(1)
    scale = m[:, scale_off:scale_off + hidden].unsqueeze(1)
    y = n.view(frames, tokens // frames, hidden) * (1 + scale) + shift
    return y.view(tokens, hidden)


def _gate_residual_torch_(x: torch.Tensor, o: torch.Tensor, m: torch.Tensor,
                          gate_off: int) -> None:
    tokens, hidden = x.shape
    frames = m.shape[0]
    gate = m[:, gate_off:gate_off + hidden].unsqueeze(1)
    per_frame = tokens // frames
    x.view(frames, per_frame, hidden).add_(
        gate * o.view(frames, per_frame, hidden))


def _rotate_pairs_torch(t: torch.Tensor, cos_t: torch.Tensor,
                        sin_t: torch.Tensor) -> torch.Tensor:
    a, b = t[..., 0::2], t[..., 1::2]
    even = a * cos_t[..., 0::2] + (-b) * sin_t[..., 0::2]
    odd = b * cos_t[..., 1::2] + a * sin_t[..., 1::2]
    return torch.stack((even, odd), dim=-1).flatten(-2)


def _rope_pack_spatial_torch(qkv: torch.Tensor, cos_t: torch.Tensor,
                             sin_t: torch.Tensor, pack_v: bool = True) -> torch.Tensor:
    tokens = qkv.shape[0]
    hidden = qkv.shape[1] // 3
    n_spatial, head_dim = cos_t.shape
    heads = hidden // head_dim
    frames = tokens // n_spatial
    out = torch.empty(3 if pack_v else 2, frames, heads, n_spatial, head_dim,
                      dtype=qkv.dtype, device=qkv.device)
    for i in range(3 if pack_v else 2):
        part = (qkv[:, i * hidden:(i + 1) * hidden]
                .reshape(frames, n_spatial, heads, head_dim)
                .permute(0, 2, 1, 3))
        out[i].copy_(_rotate_pairs_torch(part, cos_t, sin_t) if i < 2 else part)
    return out


def _rope_pack_temporal_torch(qkv: torch.Tensor, cos_t: torch.Tensor,
                              sin_t: torch.Tensor, pack_v: bool = True) -> torch.Tensor:
    tokens = qkv.shape[0]
    hidden = qkv.shape[1] // 3
    frames, head_dim = cos_t.shape
    heads = hidden // head_dim
    n_spatial = tokens // frames
    out = torch.empty(3 if pack_v else 2, n_spatial, heads, frames, head_dim,
                      dtype=qkv.dtype, device=qkv.device)
    for i in range(3 if pack_v else 2):
        part = (qkv[:, i * hidden:(i + 1) * hidden]
                .reshape(frames, n_spatial, heads, head_dim)
                .permute(1, 2, 0, 3))
        out[i].copy_(_rotate_pairs_torch(part, cos_t, sin_t) if i < 2 else part)
    return out


class OasisDiT(nn.Module):
    def __init__(
        self,
        *,
        input_h: int = 18,
        input_w: int = 32,
        patch_size: int = 2,
        in_channels: int = 16,
        hidden_size: int = 1024,
        depth: int = 16,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        external_cond_dim: int = 25,
        max_frames: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.max_frames = max_frames

        self.x_embedder = OasisPatchEmbed(input_h, input_w, patch_size, in_channels, hidden_size, flatten=False)
        self.t_embedder = OasisTimestepEmbedder(hidden_size)
        head_dim = hidden_size // num_heads
        self.spatial_rotary_emb = OasisRotaryEmbedding(dim=head_dim // 2, freqs_for="pixel", max_freq=256)
        self.temporal_rotary_emb = OasisRotaryEmbedding(dim=head_dim, freqs_for="lang")
        self.external_cond = Linear(external_cond_dim, hidden_size, bias=True) if external_cond_dim > 0 else nn.Identity()
        self.blocks = nn.ModuleList(
            [
                SpatioTemporalDiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    is_causal=True,
                    spatial_rotary_emb=self.spatial_rotary_emb,
                    temporal_rotary_emb=self.temporal_rotary_emb,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = OasisFinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

        # Shape constants only. Nothing here reads a parameter: the harness casts
        # dtype and calls `load_state_dict` after construction, so a value derived
        # from a weight now would be stale by the first forward.
        self._input_h = input_h
        self._input_w = input_w
        self._grid_h = input_h // patch_size
        self._grid_w = input_w // patch_size
        self._n_spatial = self._grid_h * self._grid_w
        self._hidden = hidden_size
        self._heads = num_heads
        self._head_dim = head_dim
        self._norm_shape = (hidden_size,)
        self._external_cond_dim = external_cond_dim
        self._image_size = (input_h, input_w)
        self._eps = 1e-6

        # Lazily built, cached on the instance, never at module level: the bench
        # builds five instances in one worker process with `gc.collect()` and
        # `empty_cache()` between cases, so the caching allocator can hand a freed
        # `freqs` address to a later instance and a module-level cache keyed on
        # `data_ptr` could serve a stale table.
        self._sinusoid: tuple | None = None
        #: `(key, weight, bias)` for the concatenated adaLN projection, or None.
        self._adaln_concat: tuple | None = None
        #: `frames -> (graph, static_x, static_t, static_ec, static_out, key)`, for the
        #: `graph` stage only. Never populated otherwise.
        self._graphs: dict = {}
        #: Pre-resolved `(adaln, attn, mlp, temporal)` per axis, or None. Validated by
        #: `_tree_intact`, which already compares every module's children against
        #: construction, so a replaced child invalidates it without a second guard.
        self._axis_plan: tuple | None = None
        self._spatial_table: tuple | None = None
        self._temporal_cache: dict = {}

        #: Per-instance count of calls served by the fast path.
        self.fastpath_calls = 0

        # Structure and configuration only, and taken now rather than lazily --
        # see `_snapshot_tree`.
        self._guard_classes: tuple = ()
        self._guard_globals: tuple = ()
        self._guard: tuple = self._snapshot_tree()
        self._config_guard: tuple = self._snapshot_config()

    def initialize_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        weight = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(weight.view(weight.shape[0], -1))
        if self.x_embedder.proj.bias is not None:
            nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.s_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.s_adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.t_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.t_adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        if self.final_layer.linear.bias is not None:
            nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = x.shape[1]
        w = x.shape[2]
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    # ------------------------------------------------------------------
    # Cached tables
    # ------------------------------------------------------------------
    def _sinusoid_freqs(self, half: int, device: torch.device) -> torch.Tensor:
        """`timestep_embedding`'s frequency vector, cached.

        The vector depends only on `half`, `max_period` and the device, so the 33
        `arange`/`exp` pairs the reference issues per forward collapse to one build
        per instance. Measured bit-exact against the per-call expression at `T = 2, 4,
        6` in `profile/bitexact_probes.py` / `.log` ("timestep sinusoid from a cached
        frequency vector"), and at all five captured shapes as part of the whole
        transcribed body in `profile/front_end_probe.py` / `.log`
        ("cached-sinusoid + F.linear/F.silu transcription").
        """
        cached = self._sinusoid
        if cached is not None and cached[0] == half and cached[1] == device:
            return cached[2]
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=device)
            / half,
        )
        self._sinusoid = (half, device, freqs)
        return freqs

    @staticmethod
    def _freqs_key(param: torch.Tensor) -> tuple:
        """A cache key that changes whenever the frequency parameter changes.

        Both mutations the harness performs are covered: `p.data = p.data.to(dtype)`
        moves `data_ptr` and leaves `_version` at 0, and `load_state_dict`'s
        `copy_` bumps `_version` and leaves `data_ptr`. A write through
        `param.data` would move neither and is not supported -- unreachable here,
        where every mutation precedes the first forward, and declared rather than
        silently assumed.
        """
        return (param.data_ptr(), param._version, param.dtype, param.device,
                param.shape)

    @staticmethod
    def _rotary_key(rot: nn.Module) -> tuple:
        """The frequency parameter, plus the two attributes that decide how
        `get_axial_freqs` and `_forward_freqs` read it.

        `freqs_for` selects `linspace(-1, 1, dim)` positions against `arange(dim)`,
        so flipping it changes every value in the table while leaving the parameter
        untouched; `dim` sets the table's width. Neither is guarded by the parameter
        key alone. `_config_intact` also rejects a change to either, so this is the
        belt to that braces -- the key stays correct even if the predicate is ever
        relaxed.
        """
        return (OasisDiT._freqs_key(rot.freqs), rot.freqs_for, rot.dim,
                rot.dummy.device)

    def _spatial_tables(self) -> tuple:
        """`cos`/`sin` of `get_axial_freqs(grid_h, grid_w)` as `[n_spatial, head_dim]`.

        The reference rebuilds this table inside all 32 attention calls though it
        depends only on the constant `freqs` parameter; that is 65 `cos`, 65 `sin` and
        33 `arange` launches per forward (`profile/launch_census.py` /
        `launch_census.log`, top-launch-count table at `T=4`). The pairwise rotation
        the kernels apply to it is pinned in `profile/rope_probe.py` / `.log`, and the
        packed layout it feeds in `profile/sdpa_probe.py` / `.log`.
        """
        rot = self.spatial_rotary_emb
        key = self._rotary_key(rot) + (self._grid_h, self._grid_w)
        cached = self._spatial_table
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        freqs = rot.get_axial_freqs(self._grid_h, self._grid_w)
        flat = freqs.reshape(self._n_spatial, freqs.shape[-1])
        cos_t, sin_t = flat.cos().contiguous(), flat.sin().contiguous()
        self._spatial_table = (key, cos_t, sin_t)
        return cos_t, sin_t

    def _temporal_tables(self, frames: int, device: torch.device) -> tuple:
        """`cos`/`sin` of the temporal table as `[frames, head_dim]`.

        Built through the same entry point the reference uses, with the positions
        it uses: `rotate_queries_or_keys` draws `arange(seq_len, dtype=q.dtype)`,
        and `q` is fp32 here.
        """
        rot = self.temporal_rotary_emb
        key = self._rotary_key(rot) + (frames, device)
        cached = self._temporal_cache.get(frames)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        pos = torch.arange(frames, device=device, dtype=torch.float32)
        table = rot(pos, rot.freqs, seq_len=frames)
        cos_t, sin_t = table.cos().contiguous(), table.sin().contiguous()
        self._temporal_cache[frames] = (key, cos_t, sin_t)
        return cos_t, sin_t

    def _axis_modules(self) -> tuple:
        """The 32 `(adaln, attn, mlp, temporal)` tuples, resolved once.

        Only *module references* are cached, never parameter values: the loop still
        reads `adaln.weight`, `attn.to_qkv.weight` and so on live on every call, so a
        reassigned weight is followed rather than guarded against. The references
        themselves need no new guard, because `_tree_intact` already compares every
        module's `_modules` tuple against what `__init__` recorded, so a replaced child
        rejects the whole call before this is consulted.
        """
        plan = self._axis_plan
        if plan is None:
            entries = []
            for block in self.blocks:
                entries.append((block.s_adaLN_modulation[1], block.s_attn,
                                block.s_mlp, False))
                entries.append((block.t_adaLN_modulation[1], block.t_attn,
                                block.t_mlp, True))
            plan = self._axis_plan = tuple(entries)
        return plan

    def _adaln_concat_weights(self, plan: tuple) -> tuple:
        """One `[32 * 6 * hidden, hidden]` weight and its bias, cached and guarded.

        `profile/m4_experiments.log` measures the 32 separate projections at 0.472 ms
        against 0.172 ms for one concatenated GEMM, with every slice bit-identical. The
        catch is that this caches a *copy* of 32 weight tensors, which is exactly the
        live-parameter re-read the rest of this file preserves -- so the copy is keyed on
        every source parameter's `data_ptr`, `_version`, dtype, device and shape, and
        rebuilt whenever any of them moves. An in-place `copy_` bumps `_version`; a
        reassignment or a dtype cast moves `data_ptr`; `load_state_dict` does the first.
        """
        key = tuple(
            (p.data_ptr(), p._version, p.dtype, p.device, p.shape)
            for adaln, _, _, _ in plan for p in (adaln.weight, adaln.bias)
        )
        cached = self._adaln_concat
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        weight = torch.cat([adaln.weight for adaln, _, _, _ in plan], dim=0)
        bias = torch.cat([adaln.bias for adaln, _, _, _ in plan], dim=0)
        self._adaln_concat = (key, weight, bias)
        return weight, bias

    def _graph_key(self, frames: int) -> tuple:
        """Everything a captured graph bakes in and can therefore go stale on.

        A CUDA graph records the *addresses* its kernels read and write. That is the
        opposite of the property the rest of this file maintains -- parameters are
        re-read live on every call so a reassigned weight is followed -- so the graph is
        only replayable while every one of those addresses is still what it was at
        capture. The key covers every parameter, the two cached rotary tables, the
        cached sinusoid vector and the concatenated adaLN copy, on `data_ptr` and
        `_version` and dtype and shape. `_tree_intact` and `_config_intact` are checked
        separately by `_eligible` before this is consulted.
        """
        items = [frames]
        for param in self.parameters():
            items.append((param.data_ptr(), param._version, param.dtype, param.shape))
        for cached in (self._spatial_table, self._temporal_cache.get(frames),
                       self._sinusoid, self._adaln_concat):
            if cached is None:
                items.append(None)
                continue
            items.append(tuple(
                (t.data_ptr(), t._version) if isinstance(t, torch.Tensor) else t
                for t in cached))
        return tuple(items)

    def _graph_forward(self, x, t, external_cond, tokens):
        """`_fast_forward` behind a per-`T` captured graph, returning a fresh tensor.

        Three things make this a different contract from eager, and all three are
        handled here rather than assumed:

        * **inputs.** The harness's shifting pool hands a fresh `data_ptr` every call, so
          the graph cannot read the caller's tensors; they are copied into static storage.
        * **output.** Replay writes into the graph's own persistent buffer. Returning it
          would alias every call's result to the same memory, which is not what a module
          returns, so it is copied into a fresh tensor. Both copies are inside the timed
          region, which is what `profile/bench_stage_graph.json` prices.
        * **staleness.** See `_graph_key`. A mismatch discards the graph and recaptures.

        Any capture failure falls back to eager for that shape, permanently, rather than
        retrying every call.
        """
        entry = self._graphs.get(frames := x.shape[1])
        key = self._graph_key(frames)
        if entry is not None and entry[-1] != key:
            entry = None
            self._graphs.pop(frames, None)
        if entry is None:
            if self._graphs.get(("failed", frames)):
                return self._fast_forward(x, t, external_cond, tokens)
            static_x, static_t = x.clone(), t.clone()
            static_ec = external_cond.clone() if external_cond is not None else None
            # Warm the caches and any lazy allocator behaviour before capture, so the
            # graph records only the steady-state work.
            for _ in range(3):
                self._fast_forward(static_x, static_t, static_ec, tokens)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    static_out = self._fast_forward(static_x, static_t, static_ec,
                                                    tokens)
            except Exception:  # noqa: BLE001 - capture can reject the workload
                self._graphs[("failed", frames)] = True
                return self._fast_forward(x, t, external_cond, tokens)
            entry = (graph, static_x, static_t, static_ec, static_out,
                     self._graph_key(frames))
            self._graphs[frames] = entry

        graph, static_x, static_t, static_ec, static_out, _ = entry
        static_x.copy_(x)
        static_t.copy_(t)
        if static_ec is not None and external_cond is not None:
            static_ec.copy_(external_cond)
        graph.replay()
        return static_out.clone()

    # ------------------------------------------------------------------
    # Front-end bodies, shared by both paths
    # ------------------------------------------------------------------
    def _patch_embed(self, x: torch.Tensor) -> torch.Tensor:
        """`OasisPatchEmbed.forward` with `flatten=False`, through `F.conv2d`.

        This is the baseline `Conv2d.forward` body verbatim, followed by the
        baseline patch embed's own `permute`. The frozen `L2.oasis_patch_embed`
        cannot be called here: its fused route admits fp32 and emulates TF32
        rounding, and deviates by 3.6e-07 - 7.2e-07, which takes the whole stack to
        0.7974 matched (`profile/front_end_probe.py` / `.log`,
        `profile/isolate_frozen.py` / `.log`).

        No hooks are run here. This is the *fast path's* body; the reference path
        reaches the same arithmetic through `_patch_embed_hooked`, which does run
        them. The fast path does not need to, because the predicate rejects any call
        where a hook exists anywhere in the tree.
        """
        emb = self.x_embedder
        _, _, height, width = x.shape
        if (height, width) != emb.img_size:
            raise AssertionError(
                f"Input image size ({height}*{width}) doesn't match model {emb.img_size}.",
            )
        proj = emb.proj
        y = F.conv2d(x, proj.weight, proj.bias, stride=proj.stride,
                     padding=proj.padding, dilation=proj.dilation, groups=proj.groups)
        y = y.permute(0, 2, 3, 1)
        return emb.norm(y) if emb.norm is not None else y

    def _patch_embed_hooked(self, x: torch.Tensor) -> torch.Tensor:
        """`_patch_embed`, but with every bypassed module's hooks run in place."""
        emb = self.x_embedder

        def forward(x, random_sample=False):  # `OasisPatchEmbed.forward`'s own names
            _, _, height, width = x.shape
            if not random_sample and (height, width) != emb.img_size:
                raise AssertionError(
                    f"Input image size ({height}*{width}) doesn't match model "
                    f"{emb.img_size}.",
                )
            y = _call_with_hooks(emb.proj, _conv2d_body(emb.proj), x)
            y = y.permute(0, 2, 3, 1)
            return emb.norm(y) if emb.norm is not None else y

        return _call_with_hooks(emb, forward, x)

    def _timestep_sinusoid(self, t: torch.Tensor) -> torch.Tensor:
        """`OasisTimestepEmbedder.timestep_embedding`, with the vector cached."""
        dim = self.t_embedder.frequency_embedding_size
        half = dim // 2
        freqs = self._sinusoid_freqs(half, t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])],
                                  dim=-1)
        return embedding

    def _timestep_embed(self, t: torch.Tensor) -> torch.Tensor:
        """`OasisTimestepEmbedder.forward`, with the frequency vector cached.

        The frozen `L2.oasis_timestep_embedder` cannot be called here: its residual
        comes from a scalar `fmaf` chain plus a shuffle butterfly that cannot
        reproduce cuBLAS's wider-than-fp32 internal sum, so it deviates by
        6.3e-07 - 8.3e-07 and takes the whole stack to 0.7937 matched (same logs).

        Hook-free, like `_patch_embed`; see `_timestep_embed_hooked`.
        """
        emb = self.t_embedder
        h = F.linear(self._timestep_sinusoid(t), emb.mlp[0].weight, emb.mlp[0].bias)
        h = F.silu(h)
        return F.linear(h, emb.mlp[2].weight, emb.mlp[2].bias)

    def _timestep_embed_hooked(self, t: torch.Tensor) -> torch.Tensor:
        emb = self.t_embedder

        def forward(t):  # `OasisTimestepEmbedder.forward`'s own name
            h = self._timestep_sinusoid(t)
            for layer in emb.mlp:
                body = (_silu_body(layer) if layer._parameters.get("weight") is None
                        else _linear_body(layer))
                h = _call_with_hooks(layer, body, h)
            return h

        return _call_with_hooks(emb, forward, t)

    def _final_layer_body(self, x: torch.Tensor, c: torch.Tensor,
                          hooked: bool) -> torch.Tensor:
        """`OasisFinalLayer.forward`, with its module calls replaced by the
        functions they wrap.

        This is a third front-end-style substitution, and it is here for a measured
        reason rather than for symmetry (`profile/frozen_final_layer_probe.log`).
        `profile/isolate_frozen.log` scores this frozen winner at 1.0000 matched, so
        the plan calls it -- but that measurement was taken with the residual stream
        in the layout the *reference* produces, a permuted view of an NCHW buffer.
        Handed a *contiguous* stream the frozen module deviates by 1.2e-04, which is
        two orders of magnitude past anything the error budget tolerates.

        So the fast path cannot call it at all, and the reference path should not:
        calling it there would make this file's fallback correct only for as long as
        ATen keeps propagating a strided output layout through 64 residual adds --
        true today, and not a property this file controls. The transcription is
        bit-exact on both layouts.
        """
        fl = self.final_layer
        silu_mod, adaln = fl.adaLN_modulation[0], fl.adaLN_modulation[1]
        if hooked:
            modulation = _call_with_hooks(silu_mod, _silu_body(silu_mod), c)
            modulation = _call_with_hooks(adaln, _linear_body(adaln), modulation)
        else:
            modulation = F.linear(F.silu(c), adaln.weight, adaln.bias)
        shift, scale = modulation.chunk(2, dim=-1)
        while shift.dim() < x.dim():
            shift = shift.unsqueeze(-2)
            scale = scale.unsqueeze(-2)
        norm = fl.norm_final
        norm_body = _layer_norm_body(norm)
        normed = _call_with_hooks(norm, norm_body, x) if hooked else norm_body(x)
        y = normed * (1 + scale) + shift
        if hooked:
            return _call_with_hooks(fl.linear, _linear_body(fl.linear), y)
        return F.linear(y, fl.linear.weight, fl.linear.bias)

    def _final_layer_reference(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        def forward(x, c):  # `OasisFinalLayer.forward`'s own names
            return self._final_layer_body(x, c, hooked=True)

        return _call_with_hooks(self.final_layer, forward, x, c)

    def _external_cond_reference(self, external_cond: torch.Tensor) -> torch.Tensor:
        """`Linear.forward` is `F.linear`; spelling it out keeps this path
        independent of which class holds the parameters, for the same reason the
        three bodies above are transcribed. `nn.Identity` (the
        `external_cond_dim == 0` case) is called, having no expression to substitute.
        """
        ec = self.external_cond
        if type(ec) is not Linear:
            return ec(external_cond)
        return _call_with_hooks(ec, _linear_body(ec), external_cond)

    # ------------------------------------------------------------------
    # Eligibility
    # ------------------------------------------------------------------
    def _snapshot_tree(self) -> tuple:
        """Record the call path of every module the fast path bypasses.

        The fast path drives the whole stack itself, so it does not call
        `x_embedder`, `t_embedder`, any block, `final_layer`, or anything below
        them -- 615 modules for the captured configuration. A hook, a replaced
        child or a `torch.compile` wrapper anywhere in there changes what the
        reference computes while leaving this module's own identity intact, so the
        guard has to cover the whole tree rather than one child.

        Two costs were measured before choosing this shape, in
        `profile/guard_cost.py` / `guard_cost.log`: walking the live `self.modules()`
        generator every call costs 271 us, while iterating a snapshot built once costs
        140 us. The snapshot is the cheaper *and* the stronger of the two, because it
        can also compare each module's children against what was there before -- which
        is what catches a *replaced* child, the case a generator walk cannot see.
        Together with `_config_intact` the whole guard is 161 us, 2.1% of the 7.58 ms
        this path measures at `T=4`.

        Parameters are deliberately *not* snapshotted. They are re-read from the
        modules on every call (a measured 119 us, same log), so a reassigned weight is
        picked up rather than guarded against -- correct by construction, and it
        removes the need for a per-parameter identity check in the hot loop.

        Taken at the end of `__init__`, and this is load-bearing: a snapshot built
        lazily on the first admitted call would record whatever the tree looked like
        *then*, so a child replaced before the first forward would be baked in as
        expected and never rejected. Structure is not data -- `to(device)`, the
        harness's dtype cast and `load_state_dict` all keep every module object --
        so recording it here does not fall foul of the rule that nothing may be
        derived from a *parameter* in `__init__`.
        """
        mods = tuple(self.modules())
        snapshot = tuple(
            (mod, type(mod), tuple(mod._modules.values())) for mod in mods
        )
        # Calling a module is `__call__` -> `_call_impl` (or `_compiled_call_impl`)
        # -> hooks -> `forward`, and every link is replaceable while leaving the
        # class object identical. There are only a dozen distinct classes in the
        # tree, so checking the functions once per class is far cheaper than once
        # per module, and `__code__` is compared as well because reassigning it in
        # place leaves every function identity intact.
        classes = {}
        for _, ty, _ in snapshot:
            if ty not in classes:
                classes[ty] = (ty.forward, ty._call_impl, ty.__call__,
                               ty.forward.__code__)
        self._guard_classes = tuple(
            (ty, fns) for ty, fns in classes.items()
        )
        # The reference's arithmetic is spelled partly in module-level *functions*
        # that the guarded `forward` bytecode resolves out of its module globals.
        # Rebinding one of those names changes what the reference computes while
        # leaving every function identity above intact -- the globals dict keeps its
        # own identity, so nothing shorter than naming the entries would see it.
        # Only the helpers whose arithmetic this file reimplements are listed; the
        # fused kernels stand in for exactly these.
        # The module objects are resolved from the *classes actually in the tree*
        # rather than by import path, which is the whole point. An earlier version
        # imported `..L1.oasis_rotary`, i.e. the candidate package's frozen winner,
        # while the baseline spatial attention in this tree resolves
        # `baseline.L1.oasis_rotary.oasis_apply_rotary_emb` -- a different function
        # object. The guard therefore watched a helper nothing on the reference path
        # calls, and rebinding the real one changed the reference while the fast path
        # stayed admitted. Deriving the modules from `type(mod).__module__` cannot
        # drift that way whatever the candidate finder aliases.
        import sys

        helper_names = ("_modulate", "_gate", "oasis_apply_rotary_emb",
                        "oasis_rotate_half")
        seen: dict = {}
        for mod in self.modules():
            owner = sys.modules.get(type(mod).__module__)
            if owner is None or id(owner) in seen:
                continue
            seen[id(owner)] = owner
        watched = []
        for owner in seen.values():
            for name in helper_names:
                if name in owner.__dict__:
                    watched.append((owner.__dict__, name, owner.__dict__[name]))
        self._guard_globals = tuple(watched)
        return snapshot

    def _snapshot_config(self) -> tuple:
        """Scalar configuration and optional-parameter slots the fast path bakes in.

        `_snapshot_tree` covers the *call path* of every module; it says nothing
        about the plain attributes those modules read per call. The fast path hard-
        codes several of them -- `is_causal=True` for the temporal attention,
        `eps=1e-6` and `weight=bias=None` for the norms, `approximate="tanh"` for the
        GELUs, `bias=None` for `to_qkv`, the image size and the convolution's stride
        recipe -- and every one is a plain attribute a caller can reassign in place,
        which leaves the class, the children and the hooks all identical. Setting
        `blocks[0].t_attn.is_causal = False` would change the reference and not this
        path.

        Read per call as **one** generator pass over `(dict, key)` pairs plus one
        tuple compare, rather than per-module tuple builds: the scalars live in each
        module's `__dict__`, whose identity is stable for the module's lifetime, so
        the pairs can be resolved once here. Flattening this way took the clause from
        192.5 us to 14.5 us (`profile/guard_cost.py` / `guard_cost.log`), which is why
        352 scalar attributes and 162 parameter slots can be checked every call.

        Optional parameters are handled in a separate small loop with `is None`
        rather than folded into the tuple compare, deliberately: `tuple.__eq__`
        compares elements with `==`, and `Parameter.__eq__(None)` returning anything
        other than a plain bool would turn this guard into an exception.
        """
        hidden, shape = self._hidden, self._norm_shape
        attrs: list = []
        expected: list = []
        none_slots: list = []

        def add(mod, scalars: dict = (), optional_params: tuple = ()):
            d = mod.__dict__
            for key, value in (scalars.items() if scalars else ()):
                attrs.append((d, key))
                expected.append(value)
            params = mod._parameters
            for name in optional_params:
                none_slots.append((params, name))

        emb = self.x_embedder
        add(emb, {"img_size": self._image_size, "flatten": False, "norm": None,
                  "patch_size": (self.patch_size, self.patch_size),
                  "grid_size": (self._grid_h, self._grid_w)})
        add(emb.proj, {"stride": emb.proj.stride, "padding": emb.proj.padding,
                       "dilation": emb.proj.dilation, "groups": emb.proj.groups})
        add(self.t_embedder, {"frequency_embedding_size":
                              self.t_embedder.frequency_embedding_size})
        for rot, dim in ((self.spatial_rotary_emb, self._head_dim // 2),
                         (self.temporal_rotary_emb, self._head_dim)):
            add(rot, {"freqs_for": rot.freqs_for, "dim": dim})
        for block in self.blocks:
            for norm in (block.s_norm1, block.s_norm2, block.t_norm1, block.t_norm2):
                add(norm, {"eps": self._eps, "normalized_shape": shape},
                    ("weight", "bias"))
            for attn, causal in ((block.s_attn, False), (block.t_attn, True)):
                # The spatial attention passes `causal=False` as a literal, so only
                # the temporal one has an `is_causal` attribute to guard.
                add(attn, {"heads": self._heads}
                    | ({"is_causal": causal} if causal else {}))
                # `DenseAttention(backend="sdpa")` resolves to plain SDPA with
                # `attn_mask=None`; a flag flipped in place would send the reference
                # to cuDNN or FlexAttention instead.
                add(attn.attn, {"fa_func": None, "use_cudnn_kernel": False,
                                "use_flex_kernel": False})
                # `to_qkv` is built with `bias=False`, so this path calls
                # `F.linear(y, weight)` with no bias term at all. Both slots are
                # checked because `Linear.__init__`'s `self.bias = None` lands in
                # `__dict__` while assigning a `Parameter` later moves it into
                # `_parameters` and out of `__dict__` -- either alone would miss one
                # of the two ways a bias can appear. (`to_out.bias` needs no guard:
                # it is read live and passed through.)
                add(attn.to_qkv, {"bias": None}, ("bias",))
            for mlp in (block.s_mlp, block.t_mlp):
                add(mlp.act, {"approximate": "tanh"})
        add(self.final_layer.norm_final, {"eps": self._eps,
                                          "normalized_shape": shape},
            ("weight", "bias"))
        return tuple(attrs), tuple(expected), tuple(none_slots)

    def _config_intact(self) -> bool:
        attrs, expected, none_slots = self._config_guard
        if tuple(d.get(key) for d, key in attrs) != expected:
            return False
        for params, name in none_slots:
            if params.get(name) is not None:
                return False
        for globals_dict, name, obj in self._guard_globals:
            if globals_dict.get(name) is not obj:
                return False
        return True

    def _tree_intact(self) -> bool:
        for ty, (fwd, call_impl, dunder, code) in self._guard_classes:
            if (ty.forward is not fwd or ty._call_impl is not call_impl
                    or ty.__call__ is not dunder or ty.forward.__code__ is not code):
                return False
        for mod, ty, kids in self._guard:
            if (mod._forward_hooks or mod._forward_pre_hooks
                    or mod._compiled_call_impl is not None
                    or type(mod) is not ty
                    or "forward" in mod.__dict__
                    or "_call_impl" in mod.__dict__
                    or tuple(mod._modules.values()) != kids):
                return False
        return True

    def _eligible(self, x, t, external_cond) -> int | None:
        """Return the token count if this call may take the fast path, else None.

        Ordered cheapest-first, with the exact-type checks leading: a subclass is
        free to make `dtype` or `shape` raise, and the reference would still serve
        it, so reading either first would turn this guard into an exception the
        baseline does not throw.
        """
        if DISABLE_FAST_PATH:
            return None
        if type(x) not in _STORAGE_FAITHFUL_TYPES:
            return None
        if type(t) not in _STORAGE_FAITHFUL_TYPES:
            return None
        has_cond = torch.is_tensor(external_cond)
        if has_cond and type(external_cond) not in _STORAGE_FAITHFUL_TYPES:
            return None
        if external_cond is not None and not has_cond:
            # The reference silently ignores a non-tensor `external_cond`; so would
            # this path, but nothing has measured that, so it is not claimed.
            return None

        if x.dim() != 5:
            return None
        bsz, frames, channels, height, width = x.shape
        # Only `bsz == 1` is captured and measured. The claim is not that the
        # arithmetic would differ -- `c` and `x` share the batch dimension, so
        # `_modulate`'s repeat factor is 1 for any batch and the reference's reuse
        # of `shift.shape[0]` after reassigning `shift` is harmless -- but that
        # batch-dependent behaviour elsewhere (SDPA plan selection, the frozen
        # patch embed's batch-keyed admission table) is unmeasured.
        if bsz != 1 or not (1 <= frames <= self.max_frames):
            return None
        if (channels != self.in_channels or height != self._input_h
                or width != self._input_w):
            return None
        if x.dtype is not torch.float32 or not x.is_cuda:
            return None
        # Contiguity is a conservatism, not a necessity: this path reshapes `x`
        # exactly as the reference does, so a strided input would be copied
        # identically on both sides. Only contiguous inputs were measured.
        # `storage_offset` is deliberately *not* checked -- the harness's shifting
        # input pool hands the 60 timed calls a fresh, 256-byte-aligned, non-zero
        # offset every time, and rejecting on it would pass correctness while
        # silently timing the reference path at an honest-looking 1.0x.
        if not x.is_contiguous() or not _reads_as_stored(x):
            return None
        # `t.dtype` is checked even though the transcription casts with `.float()`
        # exactly as the reference does, so any dtype would agree: AC-4.2 names a
        # wrong `t` dtype as a rejection case, and only `int64` is captured.
        if (t.dtype is not torch.int64 or t.shape != (bsz, frames)
                or t.device != x.device):
            return None
        if has_cond:
            if (external_cond.dtype is not torch.float32
                    or external_cond.shape != (bsz, frames, self._external_cond_dim)
                    or external_cond.device != x.device
                    or not external_cond.is_contiguous()
                    or not _reads_as_stored(external_cond)):
                return None
            if type(self.external_cond) is not Linear:
                return None

        # The fast path reaches memory by pointer, below the dispatcher that
        # implements graph building, autocast, forward-mode tangents and any active
        # dispatch or function mode, so a call using any of them would silently
        # lose it. `no_grad` only turns off the reverse mode, hence the separate
        # forward-mode check -- and it is spelled `forward_ad._current_level >= 0`
        # rather than `torch._C._is_fwd_grad_enabled()`, which reads True in a
        # default process and would disable the fast path unconditionally.
        if torch.is_grad_enabled() or torch.is_autocast_enabled("cuda"):
            return None
        if (forward_ad._current_level >= 0
                or torch._C._len_torch_dispatch_stack()
                or torch._C._len_torch_function_stack()):
            return None
        # Global hook registries, not only per-module ones: a global forward hook
        # would run on the reference path and not here.
        if (_module_hooks._global_forward_hooks
                or _module_hooks._global_forward_pre_hooks):
            return None
        if not self._tree_intact() or not self._config_intact():
            return None
        return frames * self._n_spatial

    # ------------------------------------------------------------------
    # The two paths
    # ------------------------------------------------------------------
    def _reference_forward(self, x, t, external_cond):
        """`baseline.OasisDiT.forward`, with three bodies transcribed.

        The three -- patch embed, timestep embed, `final_layer` -- are the modules
        this tree holds as *frozen winners* while the baseline holds its own classes,
        and each is measured to deviate in fp32 (`profile/front_end_probe.log`,
        `profile/frozen_final_layer_probe.log`). Every one is replaced by the
        baseline's own expression, so this path reproduces the baseline's arithmetic
        rather than the tree's.

        The 16 blocks are *called*, and that is not an inconsistency: no
        `candidate/L3/oasis_block.py` exists, so `from .oasis_block import
        SpatioTemporalDiTBlock` resolves through the candidate finder to the baseline
        module, and these are literally the baseline's own block objects running the
        baseline's own L1/L2 children. They inherit the reference's quirks -- the
        repeat-factor-of-1 copies, `_modulate` reading `shift.shape[0]` after
        reassigning `shift` -- because they *are* the reference.
        """
        bsz, time, channels, height, width = x.shape
        x = x.reshape(bsz * time, channels, height, width)
        x = self._patch_embed_hooked(x)
        x = x.reshape(bsz, time, x.shape[1], x.shape[2], x.shape[3])
        t = t.reshape(bsz * time)
        c = self._timestep_embed_hooked(t).reshape(bsz, time, -1)
        if torch.is_tensor(external_cond):
            c = c + self._external_cond_reference(external_cond)
        for block in self.blocks:
            x = block(x, c)
        x = self._final_layer_reference(x, c)
        x = x.reshape(bsz * time, x.shape[2], x.shape[3], x.shape[4])
        x = self.unpatchify(x)
        return x.reshape(bsz, time, x.shape[1], x.shape[2], x.shape[3])

    def _fast_forward(self, x, t, external_cond, tokens):
        hidden = self._hidden
        frames = x.shape[1]
        n_spatial = self._n_spatial
        norm_shape = self._norm_shape
        eps = self._eps
        grid_h, grid_w = self._grid_h, self._grid_w
        heads, head_dim = self._heads, self._head_dim

        if _FUSED is None:
            modulate = _modulate_torch
            gate_residual_ = _gate_residual_torch_
            pack_spatial = _rope_pack_spatial_torch
            pack_temporal = _rope_pack_temporal_torch
        elif STAGE == "elementwise":
            modulate = _FUSED.modulate
            gate_residual_ = _FUSED.gate_residual_
            pack_spatial = _rope_pack_spatial_torch
            pack_temporal = _rope_pack_temporal_torch
        else:
            modulate = _FUSED.modulate
            gate_residual_ = _FUSED.gate_residual_
            pack_spatial = _FUSED.rope_pack_spatial
            pack_temporal = _FUSED.rope_pack_temporal

        # --- prologue -----------------------------------------------------
        proj = self.x_embedder.proj
        patched = F.conv2d(
            x.reshape(frames, x.shape[2], x.shape[3], x.shape[4]),
            proj.weight, proj.bias, stride=proj.stride, padding=proj.padding,
            dilation=proj.dilation, groups=proj.groups)
        # One packing copy, here, and the stream stays `[tokens, hidden]` contiguous
        # for the rest of the forward. The reference instead carries a permuted view
        # of this NCHW buffer, so its hidden axis -- the one every norm reduces and
        # every GEMM contracts along -- is strided by `n_spatial`, and each norm and
        # each `F.linear` materializes its own contiguous copy. Values are identical
        # either way (`profile/bitexact_probes.log`).
        stream = patched.permute(0, 2, 3, 1).reshape(tokens, hidden)

        c = self._timestep_embed(t.reshape(frames))
        if external_cond is not None:
            ec = self.external_cond
            c = c + F.linear(external_cond.reshape(frames, self._external_cond_dim),
                             ec.weight, ec.bias)
        # One `silu(c)`, shared by all 33 projections; the reference recomputes it
        # 33 times on the same tensor.
        sc = F.silu(c)

        cos_s, sin_s = self._spatial_tables()
        cos_t, sin_t = self._temporal_tables(frames, x.device)

        # `chunk(6, -1)` offsets into the single `[frames, 6 * hidden]` projection.
        shift_msa, scale_msa, gate_msa = 0, hidden, 2 * hidden
        shift_mlp, scale_mlp, gate_mlp = 3 * hidden, 4 * hidden, 5 * hidden

        # --- the 16-block stack, both axes ---------------------------------
        # The per-axis module references are pre-resolved once and cached
        # (`_axis_modules`). That needs no new guard: `_tree_intact` already compares
        # every module's `_modules` tuple against what `__init__` recorded, so a
        # replaced child rejects the call before the plan is consulted, and parameter
        # *values* are still read live off these modules on every call. The `walk` stage
        # keeps the live attribute walk so `validate.py` can price the difference rather
        # than a microbenchmark of shape reads.
        group = 6 * hidden
        if STAGE == "walk":
            axes = tuple(
                (block.t_adaLN_modulation[1] if temporal
                 else block.s_adaLN_modulation[1],
                 block.t_attn if temporal else block.s_attn,
                 block.t_mlp if temporal else block.s_mlp, temporal)
                for block in self.blocks for temporal in (False, True))
        else:
            axes = self._axis_modules()
        m_all = None
        if STAGE in ("concat_adaln", "graph"):
            # One GEMM for all 32 axes instead of 32. The kernels read
            # `shift`/`scale`/`gate` as offsets into a projection row, so a wider row
            # with a per-axis base offset needs no kernel change at all.
            big_w, big_b = self._adaln_concat_weights(axes)
            m_all = F.linear(sc, big_w, big_b)

        for axis_index, (adaln, attn, mlp, temporal) in enumerate(axes):
            if m_all is None:
                m = F.linear(sc, adaln.weight, adaln.bias)
                base = 0
            else:
                m = m_all
                base = axis_index * group
            normed = F.layer_norm(stream, norm_shape, None, None, eps)
            y = modulate(normed, m, base + shift_msa, base + scale_msa)
            qkv = F.linear(y, attn.to_qkv.weight)

            # `v` is the reference's own strided view of `qkv`, not a packed
            # copy. `q` and `k` reach SDPA contiguous on both sides because the
            # reference's rotary ends in a `cat` that allocates, but `v` never
            # passes through the rotary, so the reference hands SDPA a view with
            # the head axis strided by `head_dim` and the sequence axis by
            # `3 * hidden`. Packing it made the *values* bit-identical while the
            # strides differed, which AC-7's "equivalent q/k/v strides" does not
            # allow -- and it meant the fast path's bit-exactness rested on
            # `fmha_cutlassF_f32_aligned_64x64_rf_sm80` being layout-independent
            # rather than on handing it the same input. Rebuilding the view costs
            # nothing (no kernel, no allocation), lets `rope_pack` skip a third of
            # its stores, and makes the equivalence exact by construction.
            # `profile/sdpa_probe.py` / `.log` and
            # `profile/kernel_vs_reference_probe.py` / `.log` assert the strides.
            v_view = qkv.view(frames, n_spatial, 3, heads, head_dim)[:, :, 2]
            if temporal:
                packed = pack_temporal(qkv, cos_t, sin_t, False)
                a = F.scaled_dot_product_attention(
                    packed[0], packed[1], v_view.permute(1, 2, 0, 3),
                    attn_mask=None, dropout_p=0.0, is_causal=True, scale=None)
                # [n_spatial, heads, frames, d] -> [tokens, hidden] in the
                # reference's order. Handing `F.linear` the permuted view makes
                # ATen do exactly the one transposing copy the reference does.
                relayout = (a.reshape(grid_h, grid_w, heads, frames, head_dim)
                            .permute(3, 0, 1, 2, 4)
                            .reshape(tokens, hidden))
            else:
                packed = pack_spatial(qkv, cos_s, sin_s, False)
                a = F.scaled_dot_product_attention(
                    packed[0], packed[1], v_view.permute(0, 2, 1, 3),
                    attn_mask=None, dropout_p=0.0, is_causal=False, scale=None)
                relayout = a.permute(0, 2, 1, 3).reshape(tokens, hidden)

            o = F.linear(relayout, attn.to_out.weight, attn.to_out.bias)
            gate_residual_(stream, o, m, base + gate_msa)

            normed = F.layer_norm(stream, norm_shape, None, None, eps)
            y = modulate(normed, m, base + shift_mlp, base + scale_mlp)
            o = F.linear(
                F.gelu(F.linear(y, mlp.fc1.weight, mlp.fc1.bias),
                       approximate="tanh"),
                mlp.fc2.weight, mlp.fc2.bias)
            gate_residual_(stream, o, m, base + gate_mlp)

        # --- epilogue -----------------------------------------------------
        # `OasisFinalLayer.forward`, transcribed rather than called. Not a
        # launch-count choice: handed a *contiguous* stream the frozen winner
        # deviates by 1.2e-04 (`profile/frozen_final_layer_probe.log`), so calling it
        # here would be wrong by two orders of magnitude more than the budget allows.
        # Transcribing also keeps the stream 2-D to the end.
        fl = self.final_layer
        fl_adaln = fl.adaLN_modulation[1]
        mf = F.linear(sc, fl_adaln.weight, fl_adaln.bias)
        normed = F.layer_norm(stream, norm_shape, None, None, eps)
        y = modulate(normed, mf, 0, hidden)
        out = F.linear(y, fl.linear.weight, fl.linear.bias)
        # (the hook-running spelling of the same arithmetic is
        # `_final_layer_body(..., hooked=True)`, used by the reference path)

        # `unpatchify`: the einsum is a pure permutation, so the reshape after it is
        # the only copy either spelling makes.
        p = self.patch_size
        channels = self.out_channels
        out = (out.reshape(frames, grid_h, grid_w, p, p, channels)
               .permute(0, 5, 1, 3, 2, 4)
               .reshape(frames, channels, grid_h * p, grid_w * p))
        return out.reshape(1, frames, channels, grid_h * p, grid_w * p)

    def forward(self, x: torch.Tensor, t: torch.Tensor, external_cond: torch.Tensor | None = None) -> torch.Tensor:
        tokens = self._eligible(x, t, external_cond)
        if tokens is None:
            return self._reference_forward(x, t, external_cond)
        global _FASTPATH_HITS
        _FASTPATH_HITS += 1
        self.fastpath_calls += 1
        if STAGE == "graph":
            return self._graph_forward(x, t, external_cond, tokens)
        return self._fast_forward(x, t, external_cond, tokens)
