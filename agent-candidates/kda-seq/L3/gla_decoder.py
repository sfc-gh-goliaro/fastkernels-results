"""GLA / RetNet decoder layer.

Pre-norm residual, exactly the baseline's:
  attn_norm -> GatedLinearAttention -> residual
  mlp_norm  -> GLAMLP -> residual

Same ``__init__(config, layer_idx)`` / ``forward(hidden_states, attention_mask=None,
past_key_values=None, use_cache=False, **kwargs)`` contract, the same ``state_dict`` keys, the same
``named_modules()`` names and the same ``(hidden_states, attentions, past_key_values)`` return as
``baseline.py``. The five cases the harness actually scores are derived in
``docs/scored_cases.md`` -- ``docs/shapes.md`` is a top-eight-by-count view and not the population
the harness selects from -- and every number is in ``docs/results.md`` and ``benchmark.csv``.

Structurally this is the baseline. Most of the speedup is not in this file at all: the three
imports resolve to the frozen L1/L2 winners inside ``fastkernels.tasks.candidate``, and that alone
scores 2.6165x. Two changes are this layer's own, and both are gated by measurement.

**A fused residual-add + RMSNorm, for wide shapes only.** The second block reads ``x0`` and ``a1``,
writes ``x1 = x0 + a1``, then reads ``x1`` and writes ``h2 = mlp_norm(x1)``: five full-width passes.
One kernel computing both from one read of each operand is four. At the scored prefill shape a pass
is 195 661 x 2560 x 2 B = 1.002 GB. Measured against ``torch.add`` plus the frozen CUDA
``rms_norm``: **1.9309x at 195 661 rows** (1276.3 -> 661.0 us, +1.516 % of the whole 40.6 ms scored
case), and **a loss at every decode shape** -- 0.80x at 1 row, 0.84x at 64, 0.86x at 116, 0.85x at
256, because one program per row cannot fill 148 SMs at a few hundred rows. Hence
``_add_norm_ready``: the fused form serves only where it was measured to win, and the crossover was
bracketed by sweep -- **between 2048 and 4096 rows** -- rather than guessed.

**The final residual add in place, on the no-grad path only.** One fewer allocation and one fewer
full-width write. ``residual`` there is always this layer's own freshly allocated buffer, so the
write is unobservable to anyone else -- but with grad enabled it is what ``mlp_norm`` saved for its
backward, and writing into it after the MLP has run raises a version-counter error. The harness runs
everything under ``no_grad`` and would never catch that, which is why the restriction is explicit
and has its own backward test.

**What is deliberately not here: a whole-layer CUDA graph on the decode shapes.** It was built,
measured and **rejected**, and the reason belongs where the next reader will look. The plan's model
said decode was dispatch-bound with a ~20-25 us weight-traffic floor, and set ~1.5x on
``[256,1,2560]`` as the threshold below which its own dispatch-bound model "is wrong and the
lever's complexity is not justified". Measured through ``bench._time_module`` with fresh processes
per mode:
**1.044x** on that case, 1.0405-1.0586x geomean over the four decode cases. Two compounding
reasons, both since confirmed:

* The frozen ``candidate/L2/gla_attention.py`` sets ``_GRAPH_REPLAY`` from ``FK_GLA_GRAPH``
  defaulting to ``"1"``, so **its own decode graph is already on**. This layer never submitted ~10
  launches; it submits one inner graph replay plus about six. Most of the saving an L3 graph could
  take had been taken already.
* NCU (``profile/decode_graph_ncu/REPORT.md``) shows the region is 12 kernels of which **11 launch
  under one wave of 148 SMs**, the four cuBLAS GEMMs at 8.7-9.1 % occupancy and 14-29 % of DRAM
  peak. The region is **launch-geometry-bound** -- neither dispatch-bound nor bandwidth-bound -- so
  collapsing submission gaps recovers only the gaps.

The implementation and its lifecycle suite are in git history at commit ``67975c2``. What a later
phase should take from this file is that further *graph* work at decode has no headroom left, and
that raising occupancy is the lever that would matter.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.linear import Matmul
from ..L1.rms_norm import RMSNorm
from ..L2.gla_attention import GatedLinearAttention
from ..L2.gla_mlp import GLAMLP

_BF16 = torch.bfloat16

# ---------------------------------------------------------------------------
# Fused residual-add + RMSNorm, for the wide shapes where it was measured to win.
# ---------------------------------------------------------------------------
# ``FK_GLA_L3_ADD_NORM=0`` restores the frozen two-kernel pair, which is how
# ``tools/probe_candidate.py`` prices the difference.
_ADD_NORM = os.environ.get("FK_GLA_L3_ADD_NORM", "1") == "1"

# Rows below which the frozen pair is kept. 8192 is ~55 waves of a 148-SM device: an order of
# magnitude above the largest decode shape and an order of magnitude below the scored prefill one,
# so it sits on the flat part of the measured curve (1.91x at 8192 rows) rather than at the
# crossover, which ``tools/probe_add_norm.py`` brackets **between 2048 and 4096 rows**.
#
# That margin is not decoration. The sweep has been run twice and the two runs disagree below 4096
# rows -- 2048 rows measured 1.24x once and 0.89x the other time -- because at 20-30 us the wrapper's
# own Python is a measurable share, and only the later run drives the shipped code path. From 8192
# rows up the two agree to within 0.2 % (1.9099x / 1.9067x), which is why the gate is set there and
# not at the crossover the sweep happens to report.
_ADD_NORM_MIN_ROWS = int(os.environ.get("FK_GLA_L3_ADD_NORM_MIN_ROWS", "8192"))

# The hidden width this was measured at, and the *only* width admitted.
#
# This bound is load-bearing for correctness, not only for performance, and an earlier revision got
# it wrong: the predicate admitted any width whose norm weight matched while the launch below always
# passed a 4096-wide block, so at width 4097 every column past 4096 was left unwritten and the layer
# returned a wrong answer for an input it had claimed. The block is now derived from the width so the
# two cannot disagree, *and* admission is held to the measured width -- the same allow-list
# discipline the frozen modules use, where anything outside the measured set evaluates the baseline
# expression verbatim rather than a plausible-looking extrapolation.
_ADD_NORM_WIDTH = int(os.environ.get("FK_GLA_L3_ADD_NORM_WIDTH", "2560"))

# Ceiling on the grid in waves of the device's multiprocessor count. One program per row is
# 195 661 programs at prefill, whose scheduling alone costs more than the work: the frozen
# ``candidate/L1/rms_norm_kernels.cu`` records 826 -> 700 us from exactly this cap at ~10^6 rows,
# and the frozen L2 ``_epilogue_kernel`` records 908 -> 767 us.
_ADD_NORM_WAVES = 512
_ADD_NORM_WARPS = 4

_SM_COUNTS: dict[torch.device, int] = {}


def _sm_count(device: torch.device) -> int:
    """Multiprocessor count, read once per device.

    Keyed by device rather than held as one process-global integer, so a heterogeneous multi-device
    process does not size a grid for whichever device happened to be asked first.
    """
    count = _SM_COUNTS.get(device)
    if count is None:
        count = torch.cuda.get_device_properties(device).multi_processor_count
        _SM_COUNTS[device] = count
    return count


@triton.jit
def _add_rms_norm_kernel(X0, A1, X1, H2, W, eps, n_rows, row_stride,
                         N: tl.constexpr, BN: tl.constexpr):
    """``x1 = x0 + a1`` and ``h2 = rmsnorm(x1)`` in one grid-strided pass.

    Reproduces the two frozen kernels' arithmetic in their order: the add is performed in fp32 and
    **rounded once** to bfloat16, exactly as an ATen bf16 add does (its ``opmath_t`` is float); the
    sum of squares is an fp32 accumulation over those *rounded* values, not over the unrounded sum;
    the scale is ``rsqrt(acc / N + eps)``; and the store is ``(bf16)(x_f32 * inv * w_f32)`` with the
    multiplications in that order.

    ``BN`` must cover ``N``: the row is processed as one masked block, so a ``BN`` below ``N``
    silently drops the tail columns. The caller derives it from ``N`` for exactly that reason.
    """
    offsets = tl.arange(0, BN)
    mask = offsets < N
    weight = tl.load(W + offsets, mask=mask, other=0.0).to(tl.float32)
    for row in range(tl.program_id(0), n_rows, tl.num_programs(0)):
        base = row * row_stride + offsets
        x0 = tl.load(X0 + base, mask=mask, other=0.0).to(tl.float32)
        a1 = tl.load(A1 + base, mask=mask, other=0.0).to(tl.float32)
        # Rounded once, and the rounded values are what both the store and the reduction see.
        summed = (x0 + a1).to(tl.bfloat16)
        tl.store(X1 + base, summed, mask=mask)
        as_f32 = summed.to(tl.float32)
        acc = tl.sum(as_f32 * as_f32, axis=0)
        inv = tl.math.rsqrt(acc / N + eps)
        tl.store(H2 + base, (as_f32 * inv * weight).to(tl.bfloat16), mask=mask)


class GLADecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = GatedLinearAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            expand_k=config.expand_k,
            expand_v=config.expand_v,
            decay_mode=getattr(config, "decay_mode", "learned_low_rank"),
            gate_low_rank_dim=getattr(config, "gate_low_rank_dim", 16),
            gate_logit_normalizer=getattr(config, "gate_logit_normalizer", 16),
            use_rotary=getattr(config, "use_rotary", False),
            rotary_base=getattr(config, "rotary_base", 10000.0),
            rotary_max_position=getattr(config, "max_position_embeddings", 8192),
            norm_eps=config.norm_eps,
        )
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = GLAMLP(config.hidden_size, config.intermediate_size)
        self._match_baseline_module_surface()

    def _match_baseline_module_surface(self) -> None:
        """Make ``named_modules()`` exactly the baseline's, without changing any arithmetic.

        The harness shares weights with ``load_state_dict(..., strict=False)`` inside a bare
        ``except Exception: pass``, so a surface that drifts from the baseline's fails silently
        rather than loudly. Two drifts exist between the vendored modules and the frozen winners,
        and neither needs a frozen file edited:

        * The baseline's L1 ``Linear`` holds its functional operator as a ``self.matmul`` child; the
          frozen ``candidate/L1/linear.py`` dispatches without one, leaving the three MLP
          projections three names short. Attaching the frozen ``Matmul`` -- the same class, by
          relative import -- restores them. It holds no parameters and no buffers, and the frozen
          ``Linear.forward`` never calls it, so nothing but the name changes. The frozen
          ``GatedLinearAttention`` already does exactly this for its own projections.
        * The frozen ``GLAMLP`` owns a ``fused_act`` child the vendored one does not: the
          ``SiluAndMul`` whose kernel is that winner's entire speedup. It cannot be dropped, but it
          does not need to be *registered*. Moving it out of ``_modules`` into the instance
          ``__dict__`` keeps ``self.mlp.fused_act`` resolving to the same object -- ordinary
          attribute lookup precedes ``nn.Module.__getattr__`` -- while taking the name out of
          ``named_modules()``. The assertion below is what makes that safe rather than merely
          convenient: it holds no parameters, buffers or submodules, so nothing that ``_apply``,
          ``state_dict`` or ``train``/``eval`` would have done to it was doing anything.
        """
        for projection in (self.mlp.gate_proj, self.mlp.up_proj, self.mlp.down_proj):
            if not hasattr(projection, "matmul"):
                projection.matmul = Matmul()

        fused_act = self.mlp._modules.pop("fused_act", None)
        if fused_act is not None:
            if (list(fused_act.parameters()) or list(fused_act.buffers())
                    or list(fused_act.children())):
                # A future frozen revision gave it state. Put it back: leaving it unregistered
                # would hide that state from ``state_dict()`` and from the harness's own cast and
                # move, which is a far worse failure than one extra name in ``named_modules()``.
                self.mlp._modules["fused_act"] = fused_act
            else:
                # ``object.__setattr__`` because ``nn.Module.__setattr__`` would put a Module
                # straight back into ``_modules``.
                object.__setattr__(self.mlp, "fused_act", fused_act)

    # ------------------------------------------------------------------
    # The fused residual-add + normalization
    # ------------------------------------------------------------------

    def _add_norm_ready(self, x0: torch.Tensor, a1: torch.Tensor, rows: int) -> bool:
        """Whether the fused residual-add + normalization may serve this call.

        Attributes, shapes and dtypes only, decided before anything is allocated. Everything
        declined evaluates the baseline's two-kernel expression verbatim, so a decline is exactly as
        correct as the baseline.
        """
        if not _ADD_NORM or rows < _ADD_NORM_MIN_ROWS:
            return False
        # The kernel builds no autograd graph, and it produces ``x1`` and ``h2`` as separate tensors
        # that a backward pass would have to relate. The frozen ``RMSNorm`` has its own pure-PyTorch
        # path under compilation, and the baseline expression covers both cases.
        if torch.is_grad_enabled() or torch.compiler.is_compiling():
            return False
        norm = self.mlp_norm
        if not norm.elementwise_affine:
            return False
        weight = norm.weight
        if (x0.dtype is not _BF16 or a1.dtype is not _BF16 or weight.dtype is not _BF16
                or not x0.is_cuda):
            return False
        device = x0.device
        if a1.device != device or weight.device != device:
            return False
        # Flat addressing over the leading dimensions collapsed to rows: both operands must be
        # contiguous and agree on shape.
        if not x0.is_contiguous() or not a1.is_contiguous() or a1.shape != x0.shape:
            return False
        # The measured width, and the only one admitted -- see ``_ADD_NORM_WIDTH``.
        if x0.shape[-1] != _ADD_NORM_WIDTH:
            return False
        return (weight.dim() == 1 and weight.shape[0] == _ADD_NORM_WIDTH
                and weight.is_contiguous())

    def _fused_add_norm(self, x0: torch.Tensor, a1: torch.Tensor, rows: int):
        """``(x1, h2)`` out of place in both operands, one launch.

        Out of place is not a stylistic choice, and it is why the frozen
        ``RMSNorm(..., residual=...)`` entry point cannot serve this: ``fused_add_rms_norm`` writes
        **both** of its operands in place (``residual := residual + x``, ``x := rmsnorm(residual)``)
        and here ``x0`` is the **caller's** ``hidden_states``. Only ``a1`` -- the attention's own
        output -- is disposable. Cloning ``x0`` to get a second disposable operand would cost
        exactly the pass the fusion saves, so the kernel allocates its two outputs instead.
        """
        width = x0.shape[-1]
        flat_x0 = x0.reshape(rows, width)
        flat_a1 = a1.reshape(rows, width)
        x1 = torch.empty_like(flat_x0)
        h2 = torch.empty_like(flat_x0)
        grid = (min(rows, _sm_count(x0.device) * _ADD_NORM_WAVES),)
        _add_rms_norm_kernel[grid](
            flat_x0, flat_a1, x1, h2, self.mlp_norm.weight, self.mlp_norm.eps,
            rows, flat_x0.stride(0), N=width,
            # Derived from the width, never a fixed constant: a block narrower than the row would
            # silently drop the tail columns, which is the bug this replaced.
            BN=triton.next_power_of_2(width),
            num_warps=_ADD_NORM_WARPS,
        )
        return x1.view_as(x0), h2.view_as(x0)

    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, None, object | None]:
        residual = hidden_states
        h = self.attn_norm(
            hidden_states.reshape(-1, hidden_states.size(-1))
        ).reshape_as(hidden_states)
        h, attentions, past_key_values = self.attn(
            hidden_states=h,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        rows = 1
        for extent in hidden_states.shape[:-1]:
            rows *= extent
        if self._add_norm_ready(residual, h, rows):
            # One launch over four traffic units instead of two launches over five. ``residual`` is
            # still the caller's ``hidden_states`` here, so both of the kernel's outputs are fresh;
            # the buffer the epilogue then writes into is the ``x1`` it just allocated.
            residual, h = self._fused_add_norm(residual, h, rows)
            return self._residual_epilogue(residual, h, attentions, past_key_values)

        hidden_states = residual + h
        residual = hidden_states
        h = self.mlp_norm(
            hidden_states.reshape(-1, hidden_states.size(-1))
        ).reshape_as(hidden_states)
        return self._residual_epilogue(residual, h, attentions, past_key_values)

    def _residual_epilogue(self, residual, h, attentions, past_key_values):
        """``residual + mlp(h)``, in place where that is unobservable.

        ``residual`` is always this layer's own freshly allocated buffer -- both the residual add
        and the fused add-norm above allocate it -- never the caller's ``hidden_states``, so the
        write is observable to nobody else. Identical arithmetic, one fewer allocation and one fewer
        full-width write: at the scored prefill shape the out-of-place form reads both operands and
        writes a third 1.002 GB tensor.
        """
        projected = self.mlp(h)
        if torch.is_grad_enabled():
            # ``residual`` is what ``mlp_norm`` saved for its backward, so writing into it after the
            # MLP has run bumps a version counter autograd is watching and raises. The out-of-place
            # form is the baseline's own expression.
            return residual + projected, attentions, past_key_values
        return residual.add_(projected), attentions, past_key_values
