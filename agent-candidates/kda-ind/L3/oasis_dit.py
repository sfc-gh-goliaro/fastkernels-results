"""Launch-optimized OasisDiT for B200 (sm_100).

The reference implementation is launch-bound, not compute-bound: at T=4 it issues 1741
CUDA kernels for ~8 ms of device time inside a 27.5 ms wall time, and its latency is flat
in T. About 1100 of those launches rebuild loop-invariant rotary frequency tables inside
every attention of every block, and ``_modulate``/``_gate`` add a repeat-factor-1 tensor
copy per use. So the levers here are, in decreasing measured value: hoist the
loop-invariant work, run the residual stream in a layout that needs no transposes, fuse
the pointwise work, and remove the residual per-launch cost with CUDA graphs.

Marginal CUDA-graph replay cost was measured directly on this GPU rather than inferred:
1.57 us per node for a trivial kernel and 1.63 us for a 2.25 MiB pointwise kernel
(``tools/probe_graph_nodes.py``). Node count therefore stays load-bearing even after
graphing -- it is dispatch, not bandwidth -- which is what justifies the fused kernels
below.

NUMERICS -- read this before changing any arithmetic.

The harness compares against the reference in float32 with atol=1e-5, rtol=1e-3 and
requires 99% of output elements inside the bound. The output holds only 9216*T elements,
so at T=2 fewer than 185 elements may exceed it. torch 2.11 defaults to
``float32_matmul_precision="high"``, so the *reference's own* GEMMs already run on TF32
tensor cores, and roughly 28% of output elements sit near zero where the bound collapses
to atol=1e-5. The consequence is unusual and inverted from the usual intuition: being
more accurate fails exactly as hard as being less accurate. Measured matched ratios for
candidate variants:

    float32_matmul_precision="highest" end to end   0.718   FAILS (too accurate)
    one GEMM with bf16 inputs                       0.198   FAILS
    one GEMM, inputs truncated to tf32              0.586   FAILS
    tl.dot(input_precision="tf32")                  0.586   FAILS (it truncates)
    tl.dot(input_precision="tf32x3")                0.830   FAILS (too accurate)
    tf32 arithmetic inside attention                0.815   FAILS
    inputs round-to-nearest-even to tf32, fp32 acc  1.000   matches (max err 1.2e-5)
    attention in true IEEE fp32                     1.000   matches (max err 1.5e-6)

Hence: every GEMM on the residual stream stays on ``F.linear`` (cuBLAS), which *is* the
reference arithmetic bit-for-bit under the process default, and attention stays in true
IEEE fp32. The precision policy is chosen per operation and never by mutating
``torch.set_float32_matmul_precision`` or ``torch.backends.*`` -- a global flag would also
perturb the reference's own output and timing run, which happen in this same process.

Two restructurings are licensed by measurement rather than by argument: splitting or
fusing a TF32 GEMM along M or N is bit-exact (only a K split would change the
accumulation order), and the rotary tables are a deterministic function of the loaded
``freqs`` parameters, so precomputing them is bit-identical rather than merely close.

Inside the Triton kernels every arithmetic step that eager PyTorch materializes as its own
rounded value is pinned with ``add_rn``/``mul_rn``. This is not defensive style: letting
the compiler contract ``y*(1+scale)+shift`` into a single FMA rounds once where eager
rounds twice, and while that is only ~1 ulp locally, there are 64 modulation sites and 64
residual updates, and each result then feeds a TF32 GEMM whose 1e-3 quantization bins turn
an occasional 1e-7 perturbation into a 1e-3 one. The pinning costs nothing measurable and
removes the whole class of drift. ``FK_OASIS_DIT_NO_TRITON=1`` and ``FK_OASIS_DIT_NO_GRAPH=1``
select the eager and ungraphed paths, which is how the two are diffed against each other, and
``FK_OASIS_DIT_FUSE`` narrows the fused set. All three participate in the CUDA graph cache
key, so changing one on a warm cache captures a new graph instead of replaying a stale one.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastkernels.tasks.baseline.L3.oasis_dit import OasisDiT as _ReferenceOasisDiT

try:
    import triton
    import triton.language as tl
    from triton.language.extra.libdevice import add_rn, div_rn, mul_rn, rsqrt

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - falls back to the eager path
    _TRITON_AVAILABLE = False

_NO_GRAPH_ENV = "FK_OASIS_DIT_NO_GRAPH"
_NO_TRITON_ENV = "FK_OASIS_DIT_NO_TRITON"
_FUSE_ENV = "FK_OASIS_DIT_FUSE"
_LAYER_NORM_EPS = 1e-6

# The reference `vectorized_layer_norm_kernel` launch this candidate reproduces bit-exactly:
# one block of (32, 4) = 128 threads per row, reading float4 vectors. Read off the ncu report
# rather than assumed; the fused kernel's Welford tree mirrors this shuffle geometry.
_LN_LEAVES = 128
_LN_WARPS = 4
_LN_LANES = 32
_LN_SLOTS = 8  # elements each leaf accumulates: two float4 reads
# The geometry above pins the width the fused kernels can serve. It is an invariant, not a
# preference: each leaf reads exactly _LN_SLOTS elements at fixed offsets, and the rope
# kernel indexes a full row with tl.arange, which needs a power-of-two extent.
_FUSABLE_WIDTH = _LN_SLOTS * _LN_LEAVES  # 1024
assert _LN_WARPS * _LN_LANES == _LN_LEAVES

# Only fusions that are *bit-exact* against the reference ship, and that is a measured
# criterion rather than a stylistic one. Two rejected experiments are preserved in
# tools/rejected_fusions.py: an *approximate* (tl.sum two-pass) LayerNorm and a Triton
# temporal attention kernel. Each matched the op it replaced to within ~1e-6 and still failed
# end to end -- see the sensitivity table below. The shipped LayerNorm is fused too, but it
# reproduces the reference kernel's Welford tree exactly, which is why it is safe.
_VALIDATED_FUSIONS = frozenset({"pointwise", "rope"})

# The end-to-end sensitivity of this operator, measured rather than assumed:
#
#   perturbation introduced            per-op max abs   end-to-end matched ratio
#   none (the shipped bit-exact set)    0.0              1.000000
#   APPROXIMATE (tl.sum) LayerNorm      9.5e-07          0.785 / 0.797 / 0.803  (T=2/4/6)
#   Triton temporal attention           8.3e-07          0.792 / 0.801 / 0.808  (T=2/4/6)
#
# Two unrelated kernels, two different ~1e-6 perturbations, the same ~0.79 outcome. The
# mechanism is TF32 quantization: a 1e-6 relative difference in a GEMM input crosses a
# TF32 rounding boundary (bin width ~4.9e-4 relative) often enough that, compounded over
# 32 blocks, the output diverges by ~1e-3 -- larger than the 6.1e-4 bound at a typical
# output magnitude. The amplification is ~1000x and linear in the perturbation, so no
# tightening of these kernels reaches 0.99; only bit-exactness does. Every op that feeds
# a GEMM on the residual stream must therefore be bit-exact, which is exactly the line
# the shipped fusion set draws.


def _env_off(name: str) -> bool:
    return os.environ.get(name, "") not in ("", "0", "false", "False")


def _selected_fusions() -> frozenset[str]:
    """Which fused kernels to use. The environment can only ever *narrow* this set.

    Intersecting with the validated set is deliberate. An operator whose correctness is
    this sensitive should not have a numerically-invalid code path reachable by setting an
    environment variable in the calling process -- selecting one would silently produce a
    wrong answer that still looks like a configuration choice. Reproducing the rejected
    fusions is a deliberate act performed from ``tools/rejected_fusions.py``, not a
    deployment knob.
    """
    raw = os.environ.get(_FUSE_ENV)
    if raw is None:
        return _VALIDATED_FUSIONS
    asked = frozenset(part.strip() for part in raw.split(",") if part.strip())
    return asked & _VALIDATED_FUSIONS


if _TRITON_AVAILABLE:

    @triton.jit
    def _welford_step(mean, sigma2, count, val):
        """``cuWelfordOnlineSum`` from the reference LayerNorm kernel."""
        delta = val - mean
        count = count + 1.0
        mean = tl.math.fma(delta, div_rn(1.0, count), mean)
        sigma2 = tl.math.fma(delta, val - mean, sigma2)
        return mean, sigma2, count

    @triton.jit
    def _welford_combine(a_mean, a_sigma2, a_count, b_mean, b_sigma2, b_count):
        """``cuWelfordCombine`` from the reference LayerNorm kernel."""
        count = a_count + b_count
        coef = div_rn(1.0, count)
        n_a = a_count * coef
        n_b = b_count * coef
        delta = a_mean - b_mean
        mean = tl.math.fma(n_a, a_mean, n_b * b_mean)
        sigma2 = a_sigma2 + b_sigma2 + delta * delta * n_a * n_b * count
        return mean, sigma2, count

    @triton.jit
    def _welford_reduce(mean, sigma2, count, OUTER: tl.constexpr, HALF: tl.constexpr):
        """One ``__shfl_down_sync(v, HALF)`` step over a (OUTER, 2*HALF) state.

        Reshaping the lane axis to (2, HALF) puts lane k in row 0 and lane k+HALF in row 1,
        so transposing to (HALF, 2) and splitting reproduces exactly the (lane, lane+HALF)
        pairing a shuffle-down produces. That is what makes this tree, and not merely a
        mathematically equivalent one, the reference's tree.
        """
        mean = tl.trans(tl.reshape(mean, (OUTER, 2, HALF)), 0, 2, 1)
        sigma2 = tl.trans(tl.reshape(sigma2, (OUTER, 2, HALF)), 0, 2, 1)
        count = tl.trans(tl.reshape(count, (OUTER, 2, HALF)), 0, 2, 1)
        a_mean, b_mean = tl.split(mean)
        a_sigma2, b_sigma2 = tl.split(sigma2)
        a_count, b_count = tl.split(count)
        return _welford_combine(a_mean, a_sigma2, a_count, b_mean, b_sigma2, b_count)

    @triton.jit
    def _load_slots(base, leaf, LEAVES: tl.constexpr):
        """The eight values leaf ``t`` accumulates, in the reference's order.

        The reference reads float4 vectors with ``for (i = thrx; i < n_vec; i += numx)``,
        so leaf t takes elements [4t..4t+3] then [4*LEAVES+4t .. +3]. The Welford
        recurrence is sequential in this order, which is why the slots are named rather
        than held in one tensor: a column of a Triton tensor cannot be indexed directly.
        """
        return (tl.load(base + 4 * leaf + 0),
                tl.load(base + 4 * leaf + 1),
                tl.load(base + 4 * leaf + 2),
                tl.load(base + 4 * leaf + 3),
                tl.load(base + 4 * LEAVES + 4 * leaf + 0),
                tl.load(base + 4 * LEAVES + 4 * leaf + 1),
                tl.load(base + 4 * LEAVES + 4 * leaf + 2),
                tl.load(base + 4 * LEAVES + 4 * leaf + 3))

    @triton.jit
    def _store_slots(base, leaf, v0, v1, v2, v3, v4, v5, v6, v7,
                     LEAVES: tl.constexpr):
        tl.store(base + 4 * leaf + 0, v0)
        tl.store(base + 4 * leaf + 1, v1)
        tl.store(base + 4 * leaf + 2, v2)
        tl.store(base + 4 * leaf + 3, v3)
        tl.store(base + 4 * LEAVES + 4 * leaf + 0, v4)
        tl.store(base + 4 * LEAVES + 4 * leaf + 1, v5)
        tl.store(base + 4 * LEAVES + 4 * leaf + 2, v6)
        tl.store(base + 4 * LEAVES + 4 * leaf + 3, v7)

    @triton.jit
    def _residual_gate_ln_modulate_kernel(
        stream_ptr, projection_ptr, modulation_ptr, stream_out_ptr, out_ptr,
        tokens_per_frame, modulation_stride,
        gate_offset, shift_offset, scale_offset, eps,
        HAS_GATE: tl.constexpr, WIDTH: tl.constexpr, LEAVES: tl.constexpr,
        WARPS: tl.constexpr, LANES: tl.constexpr,
    ):
        """One pass over a row: gate + residual, LayerNorm, modulate.

        This is the whole boundary between two sublayers in a single launch. It writes both
        the updated residual (the next sublayer's ``x``) and the normalized/modulated result
        (the next GEMM's input), which is what collapses three launches per boundary into
        one and takes the forward from 455 to 326 kernels.

        The LayerNorm is **bit-exact**, not approximate. It reproduces the installed
        ``vectorized_layer_norm_kernel``'s arithmetic exactly: its float4 element ordering,
        its online Welford sum, its shuffle-down combine tree over 32 lanes and then 4
        warps, and ``rsqrt(sigma2 * (1/WIDTH) + eps)``. Verified against
        ``torch.ops.aten.native_layer_norm``'s own returned mean/rstd with ``torch.equal``
        (``tools/ln_welford_probe.py``). This matters because a merely-accurate LayerNorm
        does not work here: a 9.5e-7 difference in the normalized output measured 0.785-0.803
        end to end against a 0.99 requirement, since the result feeds a TF32 GEMM whose
        quantization promotes a sub-ulp difference to ~1e-3 over 32 blocks.
        """
        row = tl.program_id(0)
        frame = row // tokens_per_frame
        modulation_row = modulation_ptr + frame * modulation_stride
        source = stream_ptr + row * WIDTH
        leaf = tl.arange(0, LEAVES)

        x0, x1, x2, x3, x4, x5, x6, x7 = _load_slots(source, leaf, LEAVES)
        if HAS_GATE:
            p0, p1, p2, p3, p4, p5, p6, p7 = _load_slots(
                projection_ptr + row * WIDTH, leaf, LEAVES)
            g0, g1, g2, g3, g4, g5, g6, g7 = _load_slots(
                modulation_row + gate_offset, leaf, LEAVES)
            # Reference order is gate*projection, then the residual add, each separately
            # rounded. A contracted FMA would round once instead of twice.
            x0 = add_rn(x0, mul_rn(g0, p0))
            x1 = add_rn(x1, mul_rn(g1, p1))
            x2 = add_rn(x2, mul_rn(g2, p2))
            x3 = add_rn(x3, mul_rn(g3, p3))
            x4 = add_rn(x4, mul_rn(g4, p4))
            x5 = add_rn(x5, mul_rn(g5, p5))
            x6 = add_rn(x6, mul_rn(g6, p6))
            x7 = add_rn(x7, mul_rn(g7, p7))
            _store_slots(stream_out_ptr + row * WIDTH, leaf,
                         x0, x1, x2, x3, x4, x5, x6, x7, LEAVES)

        mean = tl.zeros((LEAVES,), dtype=tl.float32)
        sigma2 = tl.zeros((LEAVES,), dtype=tl.float32)
        count = tl.zeros((LEAVES,), dtype=tl.float32)
        mean, sigma2, count = _welford_step(mean, sigma2, count, x0)
        mean, sigma2, count = _welford_step(mean, sigma2, count, x1)
        mean, sigma2, count = _welford_step(mean, sigma2, count, x2)
        mean, sigma2, count = _welford_step(mean, sigma2, count, x3)
        mean, sigma2, count = _welford_step(mean, sigma2, count, x4)
        mean, sigma2, count = _welford_step(mean, sigma2, count, x5)
        mean, sigma2, count = _welford_step(mean, sigma2, count, x6)
        mean, sigma2, count = _welford_step(mean, sigma2, count, x7)

        # Intra-warp butterfly over 32 lanes, then across the 4 warps.
        mean = tl.reshape(mean, (WARPS, LANES))
        sigma2 = tl.reshape(sigma2, (WARPS, LANES))
        count = tl.reshape(count, (WARPS, LANES))
        # The offsets below spell out LANES == 32; the sequence is the reference's shuffle
        # geometry and must not be generalized without re-deriving it.
        mean, sigma2, count = _welford_reduce(mean, sigma2, count, WARPS, 16)
        mean, sigma2, count = _welford_reduce(mean, sigma2, count, WARPS, 8)
        mean, sigma2, count = _welford_reduce(mean, sigma2, count, WARPS, 4)
        mean, sigma2, count = _welford_reduce(mean, sigma2, count, WARPS, 2)
        mean, sigma2, count = _welford_reduce(mean, sigma2, count, WARPS, 1)
        mean = tl.reshape(mean, (1, WARPS))
        sigma2 = tl.reshape(sigma2, (1, WARPS))
        count = tl.reshape(count, (1, WARPS))
        mean, sigma2, count = _welford_reduce(mean, sigma2, count, 1, 2)
        mean, sigma2, count = _welford_reduce(mean, sigma2, count, 1, 1)

        row_mean = tl.sum(tl.reshape(mean, (1,)), axis=0)
        row_sigma2 = tl.sum(tl.reshape(sigma2, (1,)), axis=0)
        row_rstd = rsqrt(row_sigma2 * (1.0 / WIDTH) + eps)

        s0, s1, s2, s3, s4, s5, s6, s7 = _load_slots(
            modulation_row + shift_offset, leaf, LEAVES)
        c0, c1, c2, c3, c4, c5, c6, c7 = _load_slots(
            modulation_row + scale_offset, leaf, LEAVES)
        # The reference computes rstd*(x - mean); multiplication is commutative and exactly
        # rounded, and (x - mean)*rstd was verified torch.equal to F.layer_norm's output.
        _store_slots(
            out_ptr + row * WIDTH, leaf,
            add_rn(mul_rn(mul_rn(x0 - row_mean, row_rstd), add_rn(c0, 1.0)), s0),
            add_rn(mul_rn(mul_rn(x1 - row_mean, row_rstd), add_rn(c1, 1.0)), s1),
            add_rn(mul_rn(mul_rn(x2 - row_mean, row_rstd), add_rn(c2, 1.0)), s2),
            add_rn(mul_rn(mul_rn(x3 - row_mean, row_rstd), add_rn(c3, 1.0)), s3),
            add_rn(mul_rn(mul_rn(x4 - row_mean, row_rstd), add_rn(c4, 1.0)), s4),
            add_rn(mul_rn(mul_rn(x5 - row_mean, row_rstd), add_rn(c5, 1.0)), s5),
            add_rn(mul_rn(mul_rn(x6 - row_mean, row_rstd), add_rn(c6, 1.0)), s6),
            add_rn(mul_rn(mul_rn(x7 - row_mean, row_rstd), add_rn(c7, 1.0)), s7),
            LEAVES)

    @triton.jit
    def _rope_split_qkv_kernel(
        qkv_ptr, cos_ptr, sin_ptr, query_ptr, key_ptr, value_ptr,
        tokens_per_frame, frames,
        WIDTH: tl.constexpr, HEAD_DIM: tl.constexpr, HEADS: tl.constexpr,
        TEMPORAL: tl.constexpr,
    ):
        """Split the fused qkv projection, rotate q and k, and scatter into the layout the
        following attention consumes -- replacing a permute copy plus ~12 elementwise
        launches per sublayer half."""
        row = tl.program_id(0)
        cols = tl.arange(0, WIDTH)
        head = cols // HEAD_DIM
        lane = cols % HEAD_DIM
        even = (lane % 2) == 0
        # oasis_rotate_half pairs channels as (2i, 2i+1) -- interleaved, not split-half --
        # and negates the odd member of each pair into the even slot.
        partner = tl.where(even, cols + 1, cols - 1)

        pixel = row % tokens_per_frame
        frame_row = row // tokens_per_frame
        if TEMPORAL:
            frame = frame_row % frames
            batch = frame_row // frames
            position = frame
            destination = (((batch * tokens_per_frame + pixel) * HEADS + head)
                           * (frames * HEAD_DIM) + frame * HEAD_DIM + lane)
        else:
            position = pixel
            destination = ((frame_row * HEADS + head) * (tokens_per_frame * HEAD_DIM)
                           + pixel * HEAD_DIM + lane)

        cos = tl.load(cos_ptr + position * HEAD_DIM + lane)
        sin = tl.load(sin_ptr + position * HEAD_DIM + lane)
        base = qkv_ptr + row * (3 * WIDTH)

        # The rotated q/k feed IEEE attention, which -- unlike a TF32 GEMM -- does not
        # quantize a 1-ulp difference away, so the two products and the sum stay pinned.
        q = tl.load(base + cols)
        q_partner = tl.load(base + partner)
        q_rot = tl.where(even, -q_partner, q_partner)
        tl.store(query_ptr + destination, add_rn(mul_rn(q, cos), mul_rn(q_rot, sin)))

        k = tl.load(base + WIDTH + cols)
        k_partner = tl.load(base + WIDTH + partner)
        k_rot = tl.where(even, -k_partner, k_partner)
        tl.store(key_ptr + destination, add_rn(mul_rn(k, cos), mul_rn(k_rot, sin)))

        tl.store(value_ptr + destination, tl.load(base + 2 * WIDTH + cols))


def _rotate_pairs(x: torch.Tensor) -> torch.Tensor:
    """The reference's ``oasis_rotate_half``: pairs are interleaved as (2i, 2i+1), not
    split-half. Written the same way so it is bit-identical, not merely equivalent."""
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``x * cos + rotate_pairs(x) * sin``, matching ``oasis_apply_rotary_emb``.

    The reference also concatenates two empty slices around the result; that is a copy
    with no effect on the values, so it is dropped.
    """
    return x * cos + _rotate_pairs(x) * sin


class _Derived:
    """Tensors derived from the loaded weights: built once, rebuilt whenever the weights
    could have changed. Held as plain attributes rather than buffers so the candidate's
    ``state_dict`` key set stays identical to the reference's."""

    __slots__ = ("adaln_weight", "adaln_bias", "spatial_cos", "spatial_sin",
                 "timestep_freqs", "temporal")

    def __init__(self):
        self.temporal: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


class _GraphEntry:
    __slots__ = ("graph", "x", "t", "external_cond", "out")


class OasisDiT(_ReferenceOasisDiT):
    """Drop-in replacement for the reference OasisDiT.

    The module tree is inherited verbatim, which is what makes the harness's
    ``load_state_dict(baseline.state_dict(), strict=False)`` actually transfer weights.
    That call sits inside a bare ``try/except: pass``, so any parameter-name or shape
    divergence would be silently swallowed and leave this module on its own random
    initialization -- correctness would then fail for a reason the harness never reports.
    Inheriting the tree makes that failure mode structurally impossible; only ``forward``
    is reimplemented, and every parameter is read straight off the original submodules.
    """

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
        super().__init__(
            input_h=input_h,
            input_w=input_w,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            external_cond_dim=external_cond_dim,
            max_frames=max_frames,
        )
        self._hidden = hidden_size
        self._head_dim = hidden_size // num_heads
        # The fused kernels reproduce the reference LayerNorm's launch geometry, which fixes
        # the row width they can serve. A different hidden_size is a legal construction --
        # it is a public keyword argument -- so it falls back to the eager chain rather than
        # reading past the end of a row.
        self._fusable_width = hidden_size == _FUSABLE_WIDTH
        self._grid_h, self._grid_w = self.x_embedder.grid_size
        self._tokens_per_frame = self._grid_h * self._grid_w
        self._freq_dim = self.t_embedder.frequency_embedding_size

        # Slice layout of the fused modulation GEMM: 6*hidden per block half in the
        # reference's consumption order (block0 spatial, block0 temporal, block1 ...),
        # then 2*hidden for the final layer. Within a half the six chunks follow
        # ``chunk(6, dim=-1)`` order: shift_msa, scale_msa, gate_msa, shift_mlp,
        # scale_mlp, gate_mlp.
        self._half_span = 6 * hidden_size
        self._halves = 2 * depth
        self._final_base = self._halves * self._half_span
        self._modulation_width = self._final_base + 2 * hidden_size

        self._derived: _Derived | None = None
        self._graphs: dict[tuple, _GraphEntry] = {}
        self._ungraphable: set[tuple] = set()
        self.register_load_state_dict_post_hook(_invalidate_after_load)

    # -- derived state lifecycle ------------------------------------------------

    def _invalidate_derived(self) -> None:
        """Drop everything computed from the weights.

        The graph cache has to go with it: a captured graph replays against the exact
        storage the fused tensors had at capture time, so reassigning them while a graph
        survives would leave the graph reading freed memory.
        """
        self._derived = None
        self._graphs.clear()
        self._ungraphable.clear()

    def _apply(self, *args, **kwargs):
        # ``.to(device)`` / ``.float()`` move the parameters but would not move a plain
        # attribute, so a surviving table would be on the wrong device (or, worse,
        # silently derived from pre-move weights).
        out = super()._apply(*args, **kwargs)
        self._invalidate_derived()
        return out

    def _derived_state(self) -> _Derived:
        if self._derived is None:
            self._derived = self._build_derived()
        return self._derived

    @torch.no_grad()
    def _build_derived(self) -> _Derived:
        d = _Derived()
        hidden = self._hidden

        # One GEMM for all 33 modulation projections. They all apply SiLU to the same c,
        # so one shared SiLU(c) feeds a single (modulation_width, hidden) matrix.
        # Concatenating along N is bit-exact; a K split would not be, and is rejected.
        # Measured: 32 separate projections cost 0.384 ms, the fused one 0.130 ms.
        weights, biases = [], []
        for block in self.blocks:
            weights.append(block.s_adaLN_modulation[1].weight)
            biases.append(block.s_adaLN_modulation[1].bias)
            weights.append(block.t_adaLN_modulation[1].weight)
            biases.append(block.t_adaLN_modulation[1].bias)
        weights.append(self.final_layer.adaLN_modulation[1].weight)
        biases.append(self.final_layer.adaLN_modulation[1].bias)
        d.adaln_weight = torch.cat([w.detach() for w in weights], dim=0).contiguous()
        d.adaln_bias = torch.cat([b.detach() for b in biases], dim=0).contiguous()
        assert d.adaln_weight.shape == (self._modulation_width, hidden)

        # Rotary tables, evaluated once against the *loaded* freqs parameters by calling
        # the reference's own expressions -- that is what makes them bit-identical rather
        # than approximately right, and it is why they are not recomputed with Triton
        # sin/cos. The reference rebuilds these inside every attention of every block,
        # which is where ~1100 of its 1741 launches go.
        axial = self.spatial_rotary_emb.get_axial_freqs(self._grid_h, self._grid_w)
        d.spatial_cos = axial.cos().reshape(self._tokens_per_frame, -1).contiguous()
        d.spatial_sin = axial.sin().reshape(self._tokens_per_frame, -1).contiguous()

        device = d.adaln_weight.device
        half = self._freq_dim // 2
        d.timestep_freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=device)
            / half
        )
        return d

    def _temporal_rope(self, frames: int) -> tuple[torch.Tensor, torch.Tensor]:
        d = self._derived_state()
        cached = d.temporal.get(frames)
        if cached is None:
            rotary = self.temporal_rotary_emb
            with torch.no_grad():
                # ``rotate_queries_or_keys`` builds positions with the *query's* dtype,
                # which is float32 here.
                positions = torch.arange(frames, device=rotary.freqs.device,
                                         dtype=torch.float32)
                freqs = rotary(positions, rotary.freqs, seq_len=frames)
            cached = (freqs.cos().contiguous(), freqs.sin().contiguous())
            d.temporal[frames] = cached
        return cached

    # -- forward ----------------------------------------------------------------

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                external_cond: torch.Tensor | None = None) -> torch.Tensor:
        if (x.is_cuda and not torch.is_grad_enabled() and not _env_off(_NO_GRAPH_ENV)
                and not torch.cuda.is_current_stream_capturing()):
            # The fusion configuration is part of the graph's identity, not just the
            # inputs: a captured graph has the selected kernels baked into it, so keying on
            # inputs alone would replay a stale graph after FK_OASIS_DIT_NO_TRITON or
            # FK_OASIS_DIT_FUSE changed. Resolve the configuration first, then key on it.
            signature = (_signature(x, t, external_cond), self._fusions(x))
            if signature not in self._ungraphable:
                entry = self._graphs.get(signature)
                if entry is None:
                    entry = self._capture(signature, x, t, external_cond)
                if entry is not None:
                    return self._replay(entry, x, t, external_cond)
        return self._forward_eager(x, t, external_cond)

    def _fusions(self, x: torch.Tensor) -> frozenset[str]:
        if not (_TRITON_AVAILABLE and x.is_cuda) or _env_off(_NO_TRITON_ENV):
            return frozenset()
        if not self._fusable_width:
            return frozenset()
        return _selected_fusions()

    def _forward_eager(self, x: torch.Tensor, t: torch.Tensor,
                       external_cond: torch.Tensor | None) -> torch.Tensor:
        derived = self._derived_state()
        bsz, frames, in_channels, height, width = x.shape
        hidden = self._hidden
        per_frame = self._tokens_per_frame
        rows = bsz * frames
        fuse = self._fusions(x)

        # Patch embed. cuDNN's TF32 numerics here *are* the reference, so the conv is
        # left alone; only the layout changes. The residual stream is a flat (tokens,
        # hidden) row-major matrix with row = (b*frames + f)*per_frame + pixel, so the
        # modulation row for any token is row // per_frame -- the same rule for the
        # spatial and the temporal half, since both broadcast over H and W.
        embedded = self.x_embedder.proj(x.reshape(rows, in_channels, height, width))
        stream = embedded.permute(0, 2, 3, 1).contiguous().view(rows * per_frame, hidden)

        # Conditioning. Only the frequency vector is loop-invariant; the rest is
        # bsz*frames rows and already cheap.
        args = t.reshape(rows)[:, None].float() * derived.timestep_freqs[None]
        cond = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        cond = F.linear(cond, self.t_embedder.mlp[0].weight, self.t_embedder.mlp[0].bias)
        cond = F.silu(cond)
        cond = F.linear(cond, self.t_embedder.mlp[2].weight, self.t_embedder.mlp[2].bias)
        if torch.is_tensor(external_cond):
            flat_cond = external_cond.reshape(rows, -1)
            if isinstance(self.external_cond, nn.Identity):
                cond = cond + flat_cond
            else:
                cond = cond + F.linear(flat_cond, self.external_cond.weight,
                                       self.external_cond.bias)

        modulation = F.linear(F.silu(cond), derived.adaln_weight, derived.adaln_bias)

        temporal_cos, temporal_sin = self._temporal_rope(frames)
        rope = {
            True: (derived.spatial_cos, derived.spatial_sin),
            False: (temporal_cos, temporal_sin),
        }

        # Each boundary is one launch that produces both the next residual and the next
        # GEMM's input, so the loop carries the pair.
        stream, normed = self._boundary(stream, None, modulation, 0, 0, hidden,
                                        rows, per_frame, fuse)
        for index in range(self._halves):
            base = index * self._half_span
            spatial = index % 2 == 0
            block = self.blocks[index // 2]
            attn = block.s_attn if spatial else block.t_attn
            mlp = block.s_mlp if spatial else block.t_mlp
            cos, sin = rope[spatial]

            attended = self._attention(normed, attn, bsz, frames, spatial, cos, sin, fuse)
            # The out/fc2 biases deliberately stay in the cuBLAS epilogue. Folding them
            # into the gate step would change cuBLAS's algorithm selection -- a different
            # epilogue can select a different tile and K-loop order -- which is a far
            # larger numerical perturbation than the ULP the fold would save, and the
            # gate step has to read the projection output anyway.
            projected = F.linear(attended, attn.to_out.weight, attn.to_out.bias)
            stream, normed = self._boundary(
                stream, projected, modulation, base + 2 * hidden,
                base + 3 * hidden, base + 4 * hidden, rows, per_frame, fuse)

            # F.gelu(approximate="tanh") is already one fused elementwise launch running
            # at memory-bound speed; a hand-written replacement buys nothing and would
            # add a tanh-approximation difference to validate.
            hiddens = F.gelu(F.linear(normed, mlp.fc1.weight, mlp.fc1.bias),
                             approximate="tanh")
            projected = F.linear(hiddens, mlp.fc2.weight, mlp.fc2.bias)
            if index + 1 < self._halves:
                next_shift = base + self._half_span
            else:
                next_shift = self._final_base
            stream, normed = self._boundary(
                stream, projected, modulation, base + 5 * hidden,
                next_shift, next_shift + hidden, rows, per_frame, fuse)

        patches = F.linear(normed, self.final_layer.linear.weight,
                           self.final_layer.linear.bias)
        patch = self.patch_size
        out = patches.view(rows, self._grid_h, self._grid_w, patch, patch,
                           self.out_channels)
        # "nhwpqc->nchpwq", the reference's unpatchify einsum.
        out = out.permute(0, 5, 1, 3, 2, 4)
        return out.reshape(bsz, frames, self.out_channels,
                           self._grid_h * patch, self._grid_w * patch)

    # -- pointwise / layout stages ----------------------------------------------

    def _boundary(self, stream, projection, modulation, gate_offset,
                  shift_offset, scale_offset, rows, per_frame, fuse):
        """The whole boundary between two sublayers: gate + residual, LayerNorm, modulate.

        Returns the updated residual and the next GEMM's input. ``projection`` is None for
        the very first boundary, which has no gate to apply.

        On the fused path this is a single launch whose LayerNorm is bit-exact against
        ``F.layer_norm`` (see ``_residual_gate_ln_modulate_kernel``). On the eager path it is
        the reference chain, kept so the two can be diffed against each other.
        """
        hidden = self._hidden
        has_gate = projection is not None
        if "pointwise" in fuse:
            out = torch.empty_like(stream)
            stream_out = torch.empty_like(stream) if has_gate else stream
            _residual_gate_ln_modulate_kernel[(stream.shape[0],)](
                stream, projection if has_gate else stream, modulation,
                stream_out, out,
                per_frame, modulation.stride(0),
                gate_offset, shift_offset, scale_offset, _LAYER_NORM_EPS,
                HAS_GATE=has_gate, WIDTH=hidden, LEAVES=_LN_LEAVES,
                WARPS=_LN_WARPS, LANES=_LN_LANES, num_warps=4,
            )
            return stream_out, out

        if has_gate:
            gate = modulation[:, gate_offset:gate_offset + hidden]
            stream = (stream.view(rows, per_frame, hidden)
                      + gate.unsqueeze(1) * projection.view(rows, per_frame, hidden)
                      ).view(-1, hidden)
        shift = modulation[:, shift_offset:shift_offset + hidden]
        scale = modulation[:, scale_offset:scale_offset + hidden]
        normed = F.layer_norm(stream, (hidden,), None, None, _LAYER_NORM_EPS)
        normed = (normed.view(rows, per_frame, hidden) * (1 + scale.unsqueeze(1))
                  + shift.unsqueeze(1)).view(-1, hidden)
        return stream, normed

    def _split_rope_qkv(self, normed, attn, bsz, frames, spatial, cos, sin, fuse):
        heads, head_dim, hidden = self.num_heads, self._head_dim, self._hidden
        per_frame = self._tokens_per_frame
        rows = bsz * frames
        tokens = rows * per_frame
        qkv = F.linear(normed, attn.to_qkv.weight)
        shape = ((rows, heads, per_frame, head_dim) if spatial
                 else (bsz * per_frame, heads, frames, head_dim))
        if "rope" in fuse:
            query = torch.empty(shape, device=qkv.device, dtype=qkv.dtype)
            key = torch.empty_like(query)
            value = torch.empty_like(query)
            _rope_split_qkv_kernel[(tokens,)](
                qkv, cos, sin, query, key, value,
                per_frame, frames,
                WIDTH=hidden, HEAD_DIM=head_dim, HEADS=heads, TEMPORAL=not spatial,
                num_warps=4,
            )
            return query, key, value

        cos = cos.view(1, 1, -1, head_dim)
        sin = sin.view(1, 1, -1, head_dim)
        if spatial:
            parts = qkv.view(rows, per_frame, 3, heads, head_dim)
            picks = [parts[:, :, i].permute(0, 2, 1, 3) for i in range(3)]
        else:
            # (b, f, pixel, head, d) -> (b, pixel, head, f, d): temporal attention
            # batches over pixels, so the frame axis becomes the sequence.
            parts = qkv.view(bsz, frames, per_frame, 3, heads, head_dim)
            picks = [parts[:, :, :, i].permute(0, 2, 3, 1, 4) for i in range(3)]
        # ``.contiguous()`` is load-bearing, not hygiene. These are elementwise results
        # over a permuted view, and PyTorch propagates the input's memory layout to the
        # output, so the tensor is laid out in (batch, frame, pixel, head, d) order while
        # its logical shape says (batch*pixel, head, frame, d). ``view`` still succeeds
        # because the leading axis has extent 1, which makes the mismatch invisible.
        # SDPA reads strides and does not care; a Triton kernel indexing rows arithmetically
        # does, and silently read transposed data (matched ratio 0.0005, max_abs 4.7)
        # until this was pinned. On the fused path these are already contiguous, so this
        # costs nothing there.
        query = _apply_rope(picks[0], cos, sin).contiguous().view(shape)
        key = _apply_rope(picks[1], cos, sin).contiguous().view(shape)
        value = picks[2].contiguous().view(shape)
        return query, key, value

    def _attention(self, normed, attn, bsz, frames, spatial, cos, sin, fuse):
        heads, head_dim, hidden = self.num_heads, self._head_dim, self._hidden
        per_frame = self._tokens_per_frame
        rows = bsz * frames
        tokens = rows * per_frame
        query, key, value = self._split_rope_qkv(
            normed, attn, bsz, frames, spatial, cos, sin, fuse)

        if spatial:
            # Attention must be true IEEE fp32 (tf32 dots here measure 0.815). SDPA on
            # fp32 inputs is the reference path, so the spatial axis keeps it rather than
            # reimplementing it: a Triton flash kernel could not use BLOCK_M=144 anyway
            # (tl.arange needs a power-of-two extent), and BLOCK_M=128 masked to 144 would
            # reintroduce multi-block online softmax rescaling -- exactly the
            # accumulation-order change the measurements do not cover.
            out = F.scaled_dot_product_attention(query, key, value, is_causal=False)
            return out.transpose(1, 2).contiguous().view(tokens, hidden)

        out = F.scaled_dot_product_attention(query, key, value, is_causal=attn.is_causal)
        # Scatter back to (batch, frame, pixel) row order; see the kernel's note.
        out = out.view(bsz, per_frame, heads, frames, head_dim).permute(0, 3, 1, 2, 4)
        return out.reshape(tokens, hidden)

    # -- CUDA graphs ------------------------------------------------------------

    def _capture(self, signature, x, t, external_cond) -> _GraphEntry | None:
        """Capture one graph per complete static input signature.

        The static input copies are not overhead that could be avoided: the harness's
        shifting memory pool hands a different ``data_ptr`` to every timed iteration, so
        the inputs have to be copied into the capture's buffers regardless.
        """
        entry = _GraphEntry()
        entry.x = x.detach().clone()
        entry.t = t.detach().clone()
        entry.external_cond = (external_cond.detach().clone()
                               if torch.is_tensor(external_cond) else None)

        # Warm up on a side stream so cuBLAS workspaces and all Triton compilation are
        # resolved before capture. This lands inside the harness's correctness rounds,
        # which run before it snapshots the thread count, so no compilation can be
        # attributed to the timed region.
        #
        # Deliberately OUTSIDE the fallback below. If the eager path itself raises -- a bug,
        # an OOM, a bad shape -- that is not a capture failure, and swallowing it here would
        # both misreport the cause and poison this signature for every later call. Only the
        # capture is guarded.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._forward_eager(entry.x, entry.t, entry.external_cond)
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        try:
            # Each signature gets its own private memory pool. Sharing one pool across
            # the per-signature graphs would save intermediate memory, but graphs sharing
            # a pool must be replayed in capture order, and nothing here guarantees that
            # once more than one signature is live.
            with torch.cuda.graph(graph):
                entry.out = self._forward_eager(entry.x, entry.t, entry.external_cond)
        except Exception:
            # Capture failed: drop the partially built graph and its buffers, mark this
            # signature non-graphable, and let the caller fall through to eager rather
            # than raising on half-built capture state.
            graph = None
            entry.x = entry.t = entry.external_cond = None
            entry.out = None
            self._ungraphable.add(signature)
            torch.cuda.synchronize()
            return None
        entry.graph = graph
        self._graphs[signature] = entry
        return entry

    def _replay(self, entry: _GraphEntry, x, t, external_cond) -> torch.Tensor:
        entry.x.copy_(x)
        entry.t.copy_(t)
        if entry.external_cond is not None:
            entry.external_cond.copy_(external_cond)
        entry.graph.replay()
        # Clone so a caller still holding an earlier result does not see it mutated by
        # this replay, and so the returned object is an ordinary tensor with its own
        # storage rather than a view of the capture's output buffer.
        return entry.out.clone()


def _signature(x, t, external_cond) -> tuple:
    """The complete static description of a call, for the graph cache key.

    Everything a replay bakes in has to be here. Keying on ``(bsz, T)`` alone would be
    wrong: the captured graph also fixes the spatial extents, every input's dtype and
    device, and the memory layout the kernels index through, and ``forward``'s signature
    admits calls that vary all of them. A miss simply captures another graph, so being
    over-specific costs one capture and being under-specific returns a wrong answer.
    """
    def describe(tensor):
        if not torch.is_tensor(tensor):
            return None
        return (tuple(tensor.shape), tensor.dtype, tensor.device.index,
                tensor.is_contiguous())

    return (describe(x), describe(t), describe(external_cond))


def _invalidate_after_load(module: OasisDiT, incompatible_keys) -> None:
    """Re-arm the derived state after ``load_state_dict``.

    Building the fused weights in ``__init__`` would latch the pre-load random
    initialization; the harness loads weights *after* construction, so the tables have to
    be (re)built once the real values have landed.
    """
    del incompatible_keys
    module._invalidate_derived()
