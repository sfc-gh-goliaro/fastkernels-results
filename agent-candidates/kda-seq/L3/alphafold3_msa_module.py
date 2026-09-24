"""AlphaFold3 ``MSAModuleStack`` for B200 (sm_100), same contract as ``baseline.py``.

Algorithm 8 is four ``MSAModuleBlock`` iterations of OuterProductMean, MSA row
attention with pair bias, a SwiGLU transition, and a five-update PairBlock. On the
captured shape (``m bf16[1,8,16,64]``, ``z bf16[1,16,16,128]``) that is ~1.5 GFLOP
against 5.67 MiB of weights -- on the order of 1 us of real work -- while the eager
baseline measures 8.15 ms. Every microsecond of the gap is orchestration.

TWO LEVERS, AND WHICH ONE MATTERS HERE

This file's ``from ..L2.X import Y`` imports resolve to the workspace's *frozen L2
winners* (``fastkernels.tasks.candidate.L2.*``), while ``baseline.py``'s
identical-looking imports resolve to ``fastkernels.tasks.baseline.L2.*``. All four
winners take their fast path at this configuration, so rebuilding the baseline's
module tree here inherits fused kernels with no new CUDA code. That recomposition --
``_composed`` below -- measures **1.1909 ms, 6.84x**, and is both this file's fallback
and the number every further change has to beat.

Counting where its time goes is what set the direction (``profile/p1-anchor/``):

    score ~= max( host_enqueue , floor + SUM over kernels of (gap + duration) )

    anchor:   11 us floor + 79 launches x 2.5 us gap (198 us) + 955 us duration
              = 1163 us predicted, 1203 us measured, host enqueue only 664 us
    baseline: 588 launches, 3565 dispatches, host-bound at 9681 us of enqueue

So the anchor is **device-bound, and 79% of its score is kernel duration** rather than
launch gap. That contradicts the premise this operator's plan was written on, which
predicted the anchor at ``7 + 79*2.5 ~= 205 us`` and framed the deliverable as cutting
79 launches to 23 -- worth 140 us of a 1000 us gap. The cause is grid width: each
winner owns one row of the pair tensor per CTA, so 16 CTAs light 11% of a 148-SM
device, and ``trimul_proj_kernel`` runs 16.8 M MAC in 17.1 us -- 1.0 TFLOP/s against
~80 TFLOP/s of fp32 FFMA peak.

Which inverts the usual fusion rule. A launch gap costs 2.5 us while these durations
run 6-42 us, so splitting a stage into *more* kernels with wider grids pays whenever it
removes more than 2.5 us of duration per launch added.

WHAT SHIPS, AND WHAT DID NOT

The shipped ``forward`` takes the composition. ``msa_module_kernels.cu`` -- four kernels
per block replacing OuterProductMean with its residual into ``z``, MSA row attention with
its pair bias, the SwiGLU transition, and all three residual adds -- is **off by default
because it measured slower**: 1.6537 ms against 1.1909 ms, so 4.90x against 6.84x. It is
retained behind ``FK_L3_MSA_FUSED=1`` because it is *correct* (bitwise equal to the
composition per block, clean under racecheck/synccheck/initcheck, deterministic over 100
runs) and because NCU says exactly what is wrong with it, which is worth more than a blank
file. ``profile/p1-fused-msa-half/results.md`` has the numbers and the ranked retry.

The short version of that diagnosis, because it corrects this file's own earlier reasoning:
sizing the kernels at 128-256 CTAs was the right *direction* and an order of magnitude too
small. The capacity that matters is warp slots, not SMs -- 148 x 64 = 9472 -- so 256 CTAs of
128 threads is 10.8% of the machine, against the winners' 1.4%. NCU measured 3.1-10.2%
achieved occupancy against 75-100% theoretical, with ``long_scoreboard`` at 14.8-31.1 cycles
per issue-active: latency-bound on global loads, because the inner loop reads one bf16 of
weight per MAC where the tile should have been staged through shared memory once per CTA.
Grid width is necessary and nowhere near sufficient.

The pair stack is still the frozen winner and still owns ~75% of the device time (716 us of
955 us), which makes it worth three times a retry of the MSA half.

Two facts about the fused schedule remain established, both of which correct the plan and
both of which are derived in ``docs/dependency-table.md`` rather than assumed -- and both of
which were confirmed empirically by the kernels being bitwise-correct and race-free:

* MSA row attention needs only its **own row** of the pair tensor for the bias --
  ``linear_z(layer_norm_z(z))[i,j,h]``, and query row ``i`` consumes ``j`` of row ``i``
  alone. So the bias for pair ``(i,j)`` is computed by the very CTA that produced
  ``z1[i,j,:]``, at no extra boundary. What is cross-CTA is the *consumer*: the
  attention kernel needs bias entries from all 16 pair CTAs, so it is separate.
* ``m`` needs no double buffer, because hoisting the value and gate projections into
  the first kernel means no CTA reads a residue row of ``m`` that another CTA writes.
  ``m1`` is still allocated fresh, for the narrower reason that ``m0`` is a harness
  input tensor.

NUMERICS

Both paths reproduce the baseline's *rounding points*, not merely its end-to-end
tolerance -- fp32 inside every reduction, bf16 at every boundary the baseline
materializes, ``eps=1e-5`` in every LayerNorm, OuterProductMean's ``linear_out`` bias
applied before the division, and ``norm + eps`` accumulated and rounded in the mask's
dtype. ``tests/test_fused.py`` shows it **bitwise equal** to the composition block by
block under genuine 0/1 masks, which the official run never generates (it materializes
float masks as ``torch.ones``, so it cannot distinguish a correct mask from an ignored
one). ``tests/test_orientations.py`` pins the four contractions that unrolling
``_permute_final_dims`` plus ``einsum`` makes easy to get silently wrong.

CONTRACT AND OBSERVABILITY

The five baseline submodules stay real children, so ``state_dict`` keys match the
baseline verbatim (227 parameter tensors, 2,970,112 elements, no buffers) and nothing
is left at random init under ``load_state_dict(..., strict=False)``. The last block
builds no ``msa_att_row``/``msa_transition``: its MSA update is dead, and building them
anyway would contribute 13 keys the baseline never supplies. The weight pack is built
lazily on the first forward and held as a plain attribute, because the harness casts
parameters to bf16 and copies shared weights in only after ``__init__``, and because a
registered buffer would add keys the baseline does not have.

``BUILD_STATUS`` and ``MSAModuleStack.fast_path_reason`` are deliberately observable,
because a green ~1.0x run that silently fell back is the likeliest way this candidate
produces a false positive. Neither costs host time on the timed path: ``forward`` uses
``_eligible``, a string-free twin that ``tests/test_fused.py`` holds to the same verdict
over a matrix of inputs.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn

from ..L2.alphafold3_msa_attention import MSARowAttentionWithPairBias
from ..L2.alphafold3_outer_product_mean import OuterProductMean
from ..L2.alphafold3_pair_block import PairBlock
from ..L2.alphafold3_swiglu_transition import SwiGLUTransition


__targets__ = ["MSAModuleStack"]


# Unique to this file: torch keys both the build directory and the registered
# operator namespace on the extension name, so a name shared with another operator's
# extension makes the two sources invalidate each other's build on every import.
_LIBRARY_NAME = "fk_af3_msa_module"
_SOURCE = Path(__file__).with_name("msa_module_kernels.cu")


def _load_ops():
    """Build and register the fused MSA half at import, returning its overloads.

    Compiling here rather than inside ``forward`` keeps nvcc out of every timed
    region *and* out of the warmup iterations that ``_check_threads`` brackets: that
    guard samples ``threading.active_count()`` immediately around the timing call, so
    ninja's threads are only safe if they are already in the "before" sample.
    """
    from torch.utils.cpp_extension import load

    ns = getattr(torch.ops, _LIBRARY_NAME, None)
    if ns is None or not hasattr(ns, "msa_half"):
        # cpp_extension otherwise honours the ambient TORCH_CUDA_ARCH_LIST, which in
        # this environment names six architectures -- six nvcc passes for five targets
        # that will never run this kernel, and the cold build has to fit inside
        # with_gpu.py's 900 s BENCH_TIMEOUT alongside the four L2 winners' own builds.
        # Derived from the live device rather than hardcoded, and restored afterwards
        # so an importer's environment is not permanently mutated.
        previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        try:
            load(
                name=_LIBRARY_NAME,
                sources=[str(_SOURCE)],
                extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-lineinfo"],
                is_python_module=False,
                # The bench worker's stall watchdog reads the mtime of the log its
                # stdout is redirected to, so streaming ninja progress keeps
                # refreshing the 600 s clock. Verbose output is a safety feature.
                verbose=True,
            )
        finally:
            if previous is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = previous
        ns = getattr(torch.ops, _LIBRARY_NAME)
    # Bind the overloads, not the packets: a packet re-resolves the overload from the
    # argument types on every call, and this path is launch-latency bound.
    return ns.msa_half.default, ns.layout.default


try:
    _msa_half_op, _layout_op = _load_ops()
    BUILD_STATUS = "ok"
except Exception as exc:  # noqa: BLE001 - a build that cannot happen must degrade to
    # the frozen-winner composition, not take module import down and cost every case
    # at once. The reason is retained rather than swallowed.
    _msa_half_op = None
    _layout_op = None
    BUILD_STATUS = f"unavailable: {type(exc).__name__}: {exc}"


# The fused MSA half is **off by default because it measured slower**: 1.6537 ms against
# the composition's 1.1909 ms under `validate.py`, so 4.90x against 6.84x. It is correct
# -- bitwise equal to the composition per block, clean under racecheck/synccheck/
# initcheck, deterministic over 100 runs -- and it is retained rather than deleted so the
# measurement stays reproducible and a retry starts from correct kernels.
#
# NCU says why, and it is not the launch count (`profile/p1-fused-msa-half/results.md`):
# achieved occupancy 3.1-10.2% against 75-100% theoretical, because 128-256 CTAs is still
# only 2.7-10.8% of this device's 9472 warp slots; `long_scoreboard` 14.8-31.1 cycles per
# issue-active from one narrow global load per MAC where the weight tile should have been
# staged through shared memory; `barrier` 18.2 in the prep kernel from a 28-barrier
# shared-memory reduction that wants warp shuffles; and ~416 us of host time from 28
# allocations per forward and a boxed `Tensor[]` return.
#
# Read once at import: `forward` has no business touching the environment, and this
# decides a branch that must not cost host time on the timed path.
_FUSED_ENABLED = os.environ.get("FK_L3_MSA_FUSED", "0") == "1"


def _winner_build_status() -> dict[str, str]:
    """Each frozen L2 winner's extension status, by module name.

    The composition is only fast because every winner took its fast path; a winner
    whose extension failed to build degrades to its own eager fallback and shows up
    as a correct near-1.0x run with no other symptom. This makes that visible
    without running the timed path.

    Each winner reports differently -- ``alphafold3_pair_block`` exports
    ``BUILD_STATUS``, ``alphafold3_msa_attention`` exports
    ``extension_loaded()``/``extension_error()``, and the other two only bind their
    operator handle to a module global -- so this normalizes the four spellings
    into one. Read through ``importlib`` rather than at import time because a
    winner's build is lazy relative to nothing here, and this accessor must never
    be on a timed path.
    """
    import importlib

    status: dict[str, str] = {}

    pair = importlib.import_module("..L2.alphafold3_pair_block", __package__)
    status["alphafold3_pair_block"] = getattr(pair, "BUILD_STATUS", "not reported")

    att = importlib.import_module("..L2.alphafold3_msa_attention", __package__)
    if att.extension_loaded():
        status["alphafold3_msa_attention"] = "ok"
    else:
        status["alphafold3_msa_attention"] = f"unavailable: {att.extension_error()}"

    # These two bind their operator handle to a module global and have no reporting
    # accessor of their own; the handle being None *is* the build failure.
    opm = importlib.import_module("..L2.alphafold3_outer_product_mean", __package__)
    status["alphafold3_outer_product_mean"] = (
        "ok" if opm._fused_outer_product_mean is not None else "unavailable: build failed")

    swiglu = importlib.import_module("..L2.alphafold3_swiglu_transition", __package__)
    status["alphafold3_swiglu_transition"] = (
        "ok" if swiglu._OP_SWIGLU is not None else "unavailable: build failed")

    return status


# Section indices in the packed buffer, mirroring the ``Section`` enum in
# ``msa_module_kernels.cu``. The *offsets* are read from the extension's ``layout()``
# rather than recomputed here, so the host cannot disagree with the kernels about
# where a section lives; only the order is duplicated, and `_SECTION_SPECS` below is
# what keeps the two in step.
_SECTIONS = (
    "OPM_LN_W", "OPM_LN_B", "OPM_W1T", "OPM_W2T", "OPM_WOUTT", "OPM_OUT_BIAS",
    "ATT_LNZ_W", "ATT_LNZ_B", "ATT_WZT", "ATT_LNM_W", "ATT_LNM_B", "ATT_WVT",
    "ATT_WGT", "ATT_WOT", "TR_LN_W", "TR_LN_B", "TR_WAT", "TR_WBT", "TR_WOUTT",
)


class _PackedWeights:
    """One flat bf16 buffer per block, holding every projection reduction-major.

    Built lazily on the first forward, never in ``__init__``: the harness rebinds
    every ``p.data`` to a bf16 copy, rewrites the uninitialized weights with
    ``normal_``, and only then copies shared weights in with ``load_state_dict``.
    A pack built in ``__init__`` would hold pre-load garbage, and nothing would say
    so -- the run would simply be wrong.

    Held as a plain attribute rather than a registered buffer, because a buffer would
    add ``state_dict`` keys the baseline never supplies and break the empty
    ``unexpected_keys`` guarantee.

    The transposes are the point. A thread owning output channel ``d`` reads
    ``WT[k*D + d]``, so consecutive threads read consecutive addresses; in the
    baseline's ``[out, in]`` layout the same access strides by the reduction length,
    which for OPM's ``[128, 1024]`` output projection is 2 KB per thread.
    """

    __slots__ = ("buffer", "offsets", "total")

    def __init__(self, block, cfg, device):
        rows = _layout_op(*cfg).tolist()
        offsets = [r[0] for r in rows[:len(_SECTIONS)]]
        sizes = [r[1] for r in rows[:len(_SECTIONS)]]
        self.total = rows[len(_SECTIONS)][0]
        self.offsets = torch.tensor(offsets, dtype=torch.int32, device=device)

        opm = block.outer_product_mean
        att = getattr(block, "msa_att_row", None)
        trans = getattr(block, "msa_transition", None)

        # Section -> the source tensor, already in [reduction][output] orientation.
        # A LayerNorm affine is a vector and needs no transpose; every weight matrix
        # arrives as [out, in] and is transposed here once, on the host.
        values = {
            "OPM_LN_W": opm.layer_norm.weight,
            "OPM_LN_B": opm.layer_norm.bias,
            "OPM_W1T": opm.linear_1.weight.t(),
            "OPM_W2T": opm.linear_2.weight.t(),
            "OPM_WOUTT": opm.linear_out.weight.t(),
            "OPM_OUT_BIAS": opm.linear_out.bias,
        }
        if att is not None:
            values.update({
                "ATT_LNZ_W": att.layer_norm_z.weight,
                "ATT_LNZ_B": att.layer_norm_z.bias,
                "ATT_WZT": att.linear_z.weight.t(),
                "ATT_LNM_W": att.layer_norm_m.weight,
                "ATT_LNM_B": att.layer_norm_m.bias,
                "ATT_WVT": att.linear_v.weight.t(),
                "ATT_WGT": att.linear_g.weight.t(),
                "ATT_WOT": att.linear_o.weight.t(),
            })
        if trans is not None:
            values.update({
                "TR_LN_W": trans.layer_norm.weight,
                "TR_LN_B": trans.layer_norm.bias,
                "TR_WAT": trans.swiglu.linear_a.weight.t(),
                "TR_WBT": trans.swiglu.linear_b.weight.t(),
                "TR_WOUTT": trans.linear_out.weight.t(),
            })

        buf = torch.zeros(self.total, dtype=torch.bfloat16, device=device)
        for name, offset, size in zip(_SECTIONS, offsets, sizes):
            src = values.get(name)
            if src is None:
                continue  # the last block builds no attention or transition children
            flat = src.detach().to(device=device, dtype=torch.bfloat16).reshape(-1)
            if flat.numel() != size:
                raise RuntimeError(
                    f"packed section {name} expects {size} elements, weight has "
                    f"{flat.numel()}")
            buf[offset:offset + size] = flat
        self.buffer = buf


class MSAModuleBlock(nn.Module):
    """Single block of AF3 Algorithm 8.

    The baseline's block verbatim, over the frozen L2 winners. Kept as a real
    module rather than folded into the stack because ``state_dict`` keys are
    positional in the tree: the baseline's are
    ``blocks.{i}.{msa_att_row,msa_transition,outer_product_mean,pair_stack}.*``,
    and any other nesting silently loses weights under ``strict=False``.

    Args:
        c_m: MSA channel dimension
        c_z: Pair channel dimension
        c_hidden_msa_att: Hidden dim in MSA attention
        c_hidden_opm: Hidden dim in outer product mean
        c_hidden_mul: Hidden dim in triangle multiplication
        c_hidden_pair_att: Hidden dim in triangle attention
        no_heads_msa: Heads for MSA attention
        no_heads_pair: Heads for triangle attention
        transition_n: Transition layer scale
        msa_dropout: MSA dropout rate
        pair_dropout: Pair dropout rate
        opm_first: Whether OPM comes before MSA attention
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        transition_n: int,
        msa_dropout: float = 0.0,
        pair_dropout: float = 0.0,
        opm_first: bool = True,
        fuse_projection_weights: bool = False,
        inf: float = 1e9,
        eps: float = 1e-3,
        last_block: bool = False,
    ):
        super().__init__()
        self.opm_first = opm_first
        # The last block's MSA update is dead: nothing downstream reads ``m``
        # again, so the baseline never builds these two children there. Building
        # them anyway would add 13 state_dict keys the baseline never supplies,
        # which ``strict=False`` leaves at random init -- silently, and visible
        # only as an ``m`` mismatch.
        self.skip_msa_update = last_block and opm_first

        if not self.skip_msa_update:
            self.msa_att_row = MSARowAttentionWithPairBias(
                c_m=c_m, c_z=c_z,
                c_hidden=c_hidden_msa_att,
                no_heads=no_heads_msa,
                inf=inf,
            )

            self.msa_transition = SwiGLUTransition(c_in=c_m, n=transition_n)

        self.outer_product_mean = OuterProductMean(
            c_m=c_m, c_z=c_z, c_hidden=c_hidden_opm, eps=eps,
        )

        self.pair_stack = PairBlock(
            c_z=c_z,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_pair=no_heads_pair,
            transition_n=transition_n,
            pair_dropout=pair_dropout,
            inf=inf,
        )

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m:        [*, N_seq, N_res, C_m] MSA embedding
            z:        [*, N_res, N_res, C_z] pair embedding
            msa_mask: [*, N_seq, N_res] MSA mask
            pair_mask:[*, N_res, N_res] pair mask

        Returns:
            (m, z): updated MSA and pair embeddings
        """
        if self.opm_first:
            z = z + self.outer_product_mean(m, mask=msa_mask)

        if not self.skip_msa_update:
            m = m + self.msa_att_row(m, z=z, mask=pair_mask)
            m = m + self.msa_transition(m)

        if not self.opm_first:
            z = z + self.outer_product_mean(m, mask=msa_mask)

        z = self.pair_stack(z=z, pair_mask=pair_mask)

        return m, z


class MSAModuleStack(nn.Module):
    """AF3 Algorithm 8: MSA module stack.

    Args:
        c_m: MSA channel dimension
        c_z: Pair channel dimension
        c_hidden_msa_att: Hidden dim in MSA attention
        c_hidden_opm: Hidden dim in outer product mean
        c_hidden_mul: Hidden dim in triangle multiplication
        c_hidden_pair_att: Hidden dim in triangle attention
        no_heads_msa: Heads for MSA attention
        no_heads_pair: Heads for triangle attention
        no_blocks: Number of MSA module blocks
        transition_n: Transition scale
        opm_first: Whether OPM comes before MSA attention
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        no_blocks: int,
        transition_n: int,
        msa_dropout: float = 0.0,
        pair_dropout: float = 0.0,
        opm_first: bool = True,
        fuse_projection_weights: bool = False,
        blocks_per_ckpt: int | None = None,
        inf: float = 1e9,
        eps: float = 1e-3,
        **kwargs,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            MSAModuleBlock(
                c_m=c_m, c_z=c_z,
                c_hidden_msa_att=c_hidden_msa_att,
                c_hidden_opm=c_hidden_opm,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att,
                no_heads_msa=no_heads_msa,
                no_heads_pair=no_heads_pair,
                transition_n=transition_n,
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                opm_first=opm_first,
                inf=inf,
                eps=eps,
                last_block=(i == no_blocks - 1),
            )
            for i in range(no_blocks)
        ])

        # The geometry the fused MSA half is compiled for, as plain Python scalars so
        # the guard reads metadata only. Reading a tensor *value* in the guard would
        # need a host sync inside ``forward``.
        self._c_m = c_m
        self._c_z = c_z
        self._layout_cfg = (c_m, c_z, c_hidden_opm, no_heads_msa, c_hidden_msa_att,
                            transition_n)
        self._c_hidden_opm = c_hidden_opm
        self._no_heads_msa = no_heads_msa
        self._c_hidden_msa_att = c_hidden_msa_att
        self._transition_n = transition_n
        self._opm_first = bool(opm_first)
        self._eps = float(eps)
        self._inf = float(inf)
        # The kernels assume the per-head width times the head count fills c_m (the
        # gate multiplies the head-major output before linear_o contracts it back), and
        # that the outer product's flattened width is what linear_out consumes.
        self._geometry_ok = (
            opm_first
            and no_heads_msa * c_hidden_msa_att == c_m
            and c_hidden_opm * c_hidden_opm > 0
            and transition_n > 0
        )

        self._packed: list[_PackedWeights] | None = None
        # Any mutation of the tree invalidates the pack. A load hook rather than a
        # per-call staleness test, so the hot path stays free of the check.
        self.register_load_state_dict_post_hook(self._invalidate_hook)

    @staticmethod
    def _invalidate_hook(module: "MSAModuleStack", incompatible_keys) -> None:
        module._packed = None

    def _apply(self, *args, **kwargs):
        # ``.to(device)`` / ``.bfloat16()`` rebind parameter storage, so a pack
        # gathered from it is stale.
        self._packed = None
        return super()._apply(*args, **kwargs)

    def fast_path_reason(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> str | None:
        """``None`` if these inputs take the fused MSA half, else why they do not.

        Metadata only, and callable outside a timed region: this is how a green ~1.0x
        run that silently fell back is told apart from one that did not.

        Covers this file's own fused stages *and* delegates to the pair-block winner's
        predicate, because the pair stack still runs on every block and a fallback
        there costs as much as one here. The other three winners decide eligibility
        inside their C++ entry point and expose no predicate; their build status is
        checked instead.
        """
        if not _FUSED_ENABLED:
            return ("fused MSA half disabled by default: it measured 1.6537 ms against "
                    "the composition's 1.1909 ms; set FK_L3_MSA_FUSED=1 to enable "
                    "(see profile/p1-fused-msa-half/results.md)")
        if _msa_half_op is None:
            return f"extension unavailable ({BUILD_STATUS})"
        if torch.is_grad_enabled():
            return "grad mode is enabled"
        if not self._opm_first:
            return "opm_first=False places OuterProductMean after the MSA update"
        if not self._geometry_ok:
            return (f"geometry outside the kernels' assumptions (c_m={self._c_m}, "
                    f"no_heads_msa={self._no_heads_msa}, "
                    f"c_hidden_msa_att={self._c_hidden_msa_att}, "
                    f"transition_n={self._transition_n})")

        for name, t in (("m", m), ("z", z), ("msa_mask", msa_mask),
                        ("pair_mask", pair_mask)):
            if not isinstance(t, torch.Tensor):
                return f"{name} is not a Tensor"
            if t.dtype is not torch.bfloat16:
                return f"{name} dtype {t.dtype} is not bfloat16"
            if not t.is_cuda:
                return f"{name} is not on CUDA"
            if not t.is_contiguous():
                return f"{name} is not standard-contiguous"
            if t.requires_grad:
                return f"{name} requires grad"
            if t.is_conj() or t.is_neg():
                return f"{name} carries a conj or neg bit"
            if t.numel() == 0:
                return f"{name} is empty"
            if t.data_ptr() % 16:
                return f"{name} is not 16-byte aligned"

        if m.dim() != 4 or z.dim() != 4:
            return f"expected 4-D m and z, got {m.dim()}-D and {z.dim()}-D"
        b, n_seq, n_res, c_m = m.shape
        if b != 1:
            return f"batch {b} != 1; the kernels index a single batch element"
        if c_m != self._c_m:
            return f"m channel {c_m} != c_m {self._c_m}"
        if tuple(z.shape) != (b, n_res, n_res, self._c_z):
            return f"z shape {tuple(z.shape)} != {(b, n_res, n_res, self._c_z)}"
        if tuple(msa_mask.shape) != (b, n_seq, n_res):
            return f"msa_mask shape {tuple(msa_mask.shape)} != {(b, n_seq, n_res)}"
        if tuple(pair_mask.shape) != (b, n_res, n_res):
            return f"pair_mask shape {tuple(pair_mask.shape)} != {(b, n_res, n_res)}"

        param = self.blocks[0].outer_product_mean.linear_out.weight
        if param.dtype is not torch.bfloat16:
            return f"parameters are {param.dtype}, not bfloat16"
        if param.device != m.device:
            return f"parameters are on {param.device}, input on {m.device}"

        for i, block in enumerate(self.blocks):
            reason = block.pair_stack.fast_path_reason(z, pair_mask)
            if reason is not None:
                return f"blocks.{i}.pair_stack: {reason}"
        for name, status in _winner_build_status().items():
            if not status.startswith("ok"):
                return f"{name}: {status}"
        return None

    def _eligible(self, m, z, msa_mask, pair_mask) -> bool:
        """The gate actually on the timed path: metadata reads only, no strings.

        A separate, cheap twin of ``fast_path_reason``; ``tests/test_fused.py``
        asserts the two agree over a matrix of inputs, so the gate cannot drift from
        the explanation.
        """
        if not _FUSED_ENABLED or _msa_half_op is None or not self._geometry_ok:
            return False
        if torch.is_grad_enabled():
            return False
        for t in (m, z, msa_mask, pair_mask):
            if t.dtype is not torch.bfloat16 or not t.is_cuda:
                return False
            if not t.is_contiguous() or t.numel() == 0:
                return False
            if t.requires_grad or t.is_conj() or t.is_neg():
                return False
            if t.data_ptr() % 16:
                return False
        if m.dim() != 4 or z.dim() != 4:
            return False
        b, n_seq, n_res, c_m = m.shape
        if b != 1 or c_m != self._c_m:
            return False
        z_shape = z.shape
        if (z_shape[0] != b or z_shape[1] != n_res or z_shape[2] != n_res
                or z_shape[3] != self._c_z):
            return False
        mm, pm = msa_mask.shape, pair_mask.shape
        if mm[0] != b or mm[1] != n_seq or mm[2] != n_res:
            return False
        if pm[0] != b or pm[1] != n_res or pm[2] != n_res:
            return False
        param = self.blocks[0].outer_product_mean.linear_out.weight
        return param.dtype is torch.bfloat16 and param.device == m.device

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m:        [*, N_seq, N_res, C_m] MSA embedding
            z:        [*, N_res, N_res, C_z] pair embedding
            msa_mask: [*, N_seq, N_res] MSA mask
            pair_mask:[*, N_res, N_res] pair mask

        Returns:
            (m, z): updated MSA and pair embeddings
        """
        if not self._eligible(m, z, msa_mask, pair_mask):
            return self._composed(m, z, msa_mask, pair_mask)

        packed = self._packed
        if packed is None:
            packed = self._packed = [
                _PackedWeights(block, self._layout_cfg, m.device)
                for block in self.blocks
            ]

        # Two calls per block: the fused MSA half, then the frozen pair-block winner,
        # which still owns 75% of this candidate's device time and is the next target.
        for block, pack in zip(self.blocks, packed):
            m, z = _msa_half_op(
                m, z, msa_mask, pair_mask, pack.buffer, pack.offsets,
                self._c_hidden_opm, self._no_heads_msa, self._c_hidden_msa_att,
                self._transition_n, self._eps, self._inf,
                not block.skip_msa_update,
            )
            z = block.pair_stack(z=z, pair_mask=pair_mask)
        return m, z

    def _composed(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The baseline's own block sequence over the frozen L2 winners.

        Not a slow escape hatch: at the captured shape it measures 1.1909 ms / 6.84x,
        because each child is itself a fused single- or few-kernel operator. It is
        what every input the fused path declines gets, and the number the fused path
        had to beat to be kept (``benchmark.csv``).
        """
        for block in self.blocks:
            m, z = block(m=m, z=z, msa_mask=msa_mask, pair_mask=pair_mask)
        return m, z
