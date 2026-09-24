"""Decoder layer: attention + MLP with RMSNorm residual connections.

Unified across Llama, Qwen2, and Qwen3 architectures:
  - bias:    Qwen2 uses bias=True on QKV projection.
  - qk_norm: Qwen3 applies per-head RMSNorm to Q and K before RoPE.

This is the baseline's body over the frozen ``candidate/L1`` / ``candidate/L2``
winners.  There is no new kernel here and no restructured arithmetic: every hot
spot this layer owns lives one level down, the faster kernel for each is already
on the candidate path, and the whole job is *reaching* the ones that are safe to
reach from a module that stays interchangeable with the baseline -- same
``__init__`` and ``forward`` signatures, same submodule names, same
``state_dict()`` key set, same module tree, and the same two ``residual``
branches with the same caller-visible aliasing.

Three of the kernels arrive by import alone, and they are the win:

  * The rotary table copy disappears.  The benchmark reconstructs ``rotary_emb``
    from an ``$op_ref`` naming the *baseline* class, so the frozen attention
    builds its stand-in and reads the fp32 ``cos_sin_cache`` inside the kernel
    instead of re-materializing ``cache.to(query.dtype)`` per forward.  That copy
    is a whole launch and shape-independent -- 18.76 us at N=1, 19.19 us at
    N=16384, about 9% of the N=60 window (``profile/probe_regressions.log``).
  * The activation is 227.71 us against the vendored 567.16 us at N=16384, and
    the rope kernel 77.20 us against 200.40 us (same log).
  * The frozen attention's dispatch bypass is reached, and it is numerically
    free: reverting it recovers +0.000000 of matched ratio on every scored case
    (``profile/numerics_attribution.log``).

The frozen MLP's *fused* path is the one thing on the candidate path that this
layer reaches for and then declines -- see ``_MLP_DELEGATE_SHAPES``.  Its
activation kernel is still reached, through the delegate's ``mlp.act_fn``.

One frozen kernel is deliberately *not* reached, and that is the whole content of
``RMSNorm`` below.  See its docstring: it is a correctness requirement, measured,
not a performance preference.

One host-side routing decision is taken, in ``_MLP_DELEGATE_SHAPES``, and it is
guarded: a route that runs a submodule's body instead of calling the submodule owes
the caller a check that nothing was installed on it, or it silently drops a hook.
The first form of this route did exactly that on the one case it claimed.

Everything else the sweep measured was rejected, on two leases each, and the
numbers that rejected them are in ``profile/variant_sweep_leaseA.log`` and
``_leaseB.log``.
Both sweeps carry an *identity* null control -- the shipped layer timed against a
sibling instance of itself -- because without one there is no way to tell a real
3% from a 3% the construction produces for free.  The control reads 0.9918x to
1.0002x at N=1 and 0.9882x to 1.0018x elsewhere, and that band is the resolution
limit every rejection below is measured against:

  * bypassing ``nn.Module._call_impl`` for the two norms and calling the kernel
    directly, guarded by ``_would_only_call_forward``:
    1.0030x/1.0045x/0.9979x/0.9998x/0.9975x on one lease and
    0.9884x/0.9893x/1.0117x/1.0069x/1.0000x on the other -- inside the control's
    band on both, and not even agreeing on the sign.
  * binding the submodules to locals before use: 0.9971x/1.0056x/0.9916x/0.9996x
    /0.9995x and 0.9892x/1.0005x/1.0006x/1.0072x/0.9998x.  Each submodule is read
    exactly once in this body, so a local binding pays the same ``LOAD_ATTR`` and
    adds a ``STORE_FAST``; there was never a mechanism for it to win, and it does
    not.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.rms_norm import RMSNorm as _FrozenRMSNorm
from ..L2.attention import LlamaAttention, _would_only_call_forward
from ..L2.llama_mlp import LlamaMLP
from ...baseline.L1.rms_norm import RMSNorm as _VendoredRMSNorm


class RMSNorm(_FrozenRMSNorm):
    """The frozen norm module, pinned to the vendored kernel it falls back to.

    The frozen norm reads each row once and reduces the sum of squares through a
    warp shuffle and a shared-memory pass, where the vendored kernel streams the
    row twice and reduces its own way.  Both are correct and they agree: over row
    counts 1, 26, 60, 279, 1024 and 16384 at hidden=4096, on both entry points,
    not one element of either output falls outside the benchmark's
    ``atol=rtol=1e-2`` bound, the residual handed back is bit-identical, and the
    largest disagreement anywhere is 1.56e-2 absolute at 7.8e-3 relative -- a
    late-bit difference in the fp32 reduction order, nothing more
    (``profile/norm_divergence.log``).

    It does not survive the layer.  The normalized row feeds attention, and
    attention's softmax has no headroom for a late bit: on seed 201 the frozen
    norm's output differs from the vendored one on 0.02-0.03% of elements by at
    most 7.81e-3, and the attention output computed from it differs on **27.21%**
    of elements at N=60 and 24.31% at N=26, with a largest relative error of
    85.83 -- the attention entries that sit near zero are differences of large
    cancelling terms, so a one-ULP perturbation upstream lands there as a full-
    magnitude error.  The layer output ends up with 4.2-4.5% of its elements
    outside the bound, against the 1% the benchmark allows, and
    ``fastkernels bench`` returns INCORRECT_NUMERICAL on both cases at 0.9582 and
    0.9547 (``profile/validate_round0_first.log``).

    That failure is a coin flip, not a property of those two shapes.  It appears
    on the second of the harness's three seeds and not the first, which is why a
    single-seed probe read 1.0000 on the same composition
    (``profile/probe_frozen.log``); N=279 and N=1 pass here on the same seed only
    because none of their norm outputs happened to land on a rounding boundary.

    Reverting these two norms is also the *whole* fix: with them vendored, the
    worst matched ratio over the harness's three seeds is 1.000000 on all five
    scored cases, recovering +0.043384 at N=60 and +0.045335 at N=26 and
    +0.002193 at N=16384, while every other frozen substitution is worth at most
    +0.000038 (``profile/numerics_attribution.log``).

    ``matched=1.000000`` means every element is inside the benchmark's
    ``atol=rtol=1e-2`` bound -- it does **not** mean the outputs are equal to the
    baseline's.  ``bench_results/bench.json`` still reports max absolute errors of
    3.12e-2 to 6.25e-2 on four of the five cases, which is one to two bf16 ULPs at
    those magnitudes.  What the pinning buys is that no element is near the bound
    rather than 0.63% of the tolerance being spent before any fusion of ours adds
    to it; the remaining difference is the frozen activation and the fp32
    summation order, both well inside tolerance.

    Its cost is close to nothing and is negative on the largest case: the two
    vendored norms are 219.31 us against the frozen 262.35 us at N=16384, and
    5.38 + 5.48 us against 4.73 + 4.77 us at N=1 (``profile/probe_regressions.log``).

    Only the kernel is pinned.  ``forward`` is inherited from the frozen module
    unchanged, so the ``torch.compiler.is_compiling()`` dispatch to
    ``forward_native``, the ``elementwise_affine=False`` unit scale and the weight
    dtype/device coercion all still happen, and hooks, ``Module.compile()`` and
    autograd-shaped use are untouched.  Subclassing rather than swapping the
    submodule class keeps the class name, the parameter, the ``state_dict`` key
    and the module tree exactly as the baseline has them.
    """

    forward_cuda = staticmethod(_VendoredRMSNorm.forward_cuda)


# Row counts and weight geometries where an interleaved whole-layer measurement
# says the frozen MLP's own fused fast path loses to the composition it falls back
# to, keyed by ``(rows, hidden, intermediate)`` -- the same key
# ``LlamaMLP._fused_config`` decides on, so an entry here is exactly an override of
# one of its decisions and nothing else.
#
# ``candidate/L2/llama_mlp.py`` claims ``(1, 4096, 14336)`` because at L2 the fused
# kernel beat the composition on that geometry.  Inside this layer it does not.
# Timed against the same module with this table emptied, adjacent, five pairs per
# lease, one process at a time, **with the guard below in place**: the delegate wins
# by **+22.02 us (1.1311x)** on one lease and **+42.37 us (1.2549x)** on another, at
# matched ratio 1.000000, against a null control reading 1.0007x and 1.0006x on the
# same case (``profile/variant_sweep_guarded_lease1.log`` gpu 3,
# ``_lease2.log`` gpu 1).
#
# Those are the numbers that admit this entry, and they are deliberately not the
# earlier ``profile/variant_sweep_lease{A,B}.log`` pair (+62.11 us / 1.357x and
# +32.02 us / 1.193x).  Those were taken before the ``_would_only_call_forward``
# guard existed, so they admit a route that dropped MLP hooks -- a different route
# from this one.  A guard on the hot path costs something, and the re-measurement is
# what says the entry still earns its place with that cost paid.
#
# Nothing else belongs in the table: emptying it moves the other four cases by
# +1.84 to -0.22 us, inside that same control band, because on those shapes
# ``mlp(x)`` already runs this composition -- ``_FUSED_SHAPES`` claims no other
# scored geometry.  So the one entry is also the reason no Triton kernel is
# reachable on the scored path at all, which ``experiments/guard_check.py``
# asserts by kernel inventory rather than by reading the table.
#
# The delegate is ``LlamaMLP.forward``'s own miss body, so the numerics are the
# frozen module's own and a mistake in the shape predicate can only ever cost a
# delegation.
#
# It is not only the numerics that have to survive the substitution.  Running that
# body here instead of calling ``self.mlp(...)`` skips ``nn.Module._call_impl``,
# and with it every forward hook on the MLP, an instance-level ``forward`` in its
# ``__dict__``, and the ``_compiled_call_impl`` that ``Module.compile()`` installs.
# The baseline honours all three, so the route has to ask before it takes them
# away: ``_would_only_call_forward`` is imported from ``..L2.attention`` rather
# than restated so the two cannot drift, and it is evaluated per call because
# hooks and compilation both arrive after ``__init__`` -- a profiler installs a
# global hook, an engine compiles the model.  Without that guard a hook on the MLP
# is silently dropped on exactly the one scored case this route claims.
_MLP_DELEGATE_SHAPES = frozenset({(1, 4096, 14336)})


# The authored gate-up + SwiGLU-epilogue kernel is **not** here, and that is a
# measured outcome rather than an omission.  Phase 1 permits exactly one authored
# kernel and this was it: NCU said the gate-up GEMM at N=60 is DRAM-led at 68.37%
# of peak on 0.76 waves of 148 SMs and its own rule engine advised "more work per
# memory access (kernel fusion)", and ``candidate/L2/llama_mlp.py`` records this
# exact structure -- epilogue-fused gate-up followed by cuBLAS -- at 1.062x and
# 1.093x of its fallback at 26 and 64 rows.  So it was written and measured.
#
# It loses on every shape it could reach, at matched ratio 1.000000 throughout --
# the ``_swiglu`` epilogue imported from the frozen MLP reproduces both mandatory
# bf16 roundings, so the arithmetic was right and only the speed was wrong.
# Whole-layer, adjacent, kernel-on against kernel-off, three repeats per lease, one
# process at a time, each lease carrying an identity null control.
#
# N=60, all five configurations the plan's cap allows, on the three leases whose
# null control could resolve a 3% effect
# (``profile/gate_up_sweep_leaseA.log`` gpu 3, ``_leaseB.log`` gpu 0,
# ``_n26_lease1.log`` gpu 3, ``_n26_lease2.log`` gpu 1):
#
#   (rows, cols, k-step, warps, stages)   best of those leases   worst
#   (64, 128, 128, 8, 2)                  0.8217x -37.97us       0.8113x -40.88us
#   (64,  64, 128, 4, 4)                  0.9107x -17.14us       0.8990x -19.79us
#   (64,  64,  64, 4, 4)   <- fastest     0.9757x  -4.34us       0.9662x  -6.27us
#   (64,  32, 128, 4, 3)                  0.8942x -20.67us       0.8853x -22.83us
#   (64,  64,  64, 8, 6)                  0.9151x -16.18us       0.9104x -17.33us
#
# The fastest configuration carried unchanged to the other two small shapes -- not a
# sixth variant against the cap -- on two leases each:
#
#   N=26   0.9758x  -4.22us (gpu 3)   0.9751x  -4.37us (gpu 1)
#   N=279  0.6737x -120.93us (gpu 3)  0.7060x -110.75us (gpu 1)
#
# N=16384 was closed earlier on counters: its gate-up GEMM issues on ~99% of cycles
# at 12.64% DRAM, so the intermediate this fusion removes is overlapped rather than
# costly, and the gate would need the authored GEMM to be 18-45 us *faster* than
# nvjet (``profile/ncu-decoder-projections-n60-n16384/REPORT.md``).
#
# Why it loses, which is the part worth keeping.  At N=60 the best configuration is
# 2.4-3.4% short and clearing the gate from there needs about 10 us more -- the
# Triton GEMM beating nvjet's 42.30 us outright.  A 4x sweep of CTA count (112 to
# 448) and a deeper pipeline both made it *worse*, so the gap is structural rather
# than a tuning miss: nvjet runs 2-SM cooperative tcgen05 tiles fed by TMA, and
# masked pointer loads in Triton cannot use TMA at all -- which is exactly what a
# DRAM-led kernel needs.  At N=279 the loss is much larger for a second reason: the
# kernel tiles rows at 64, so 279 rows re-reads the 235 MB of gate-up weight five
# times, and that shape is not even bandwidth-limited to begin with (compute-led at
# 88.13% tensor pipe against 44.69% DRAM,
# ``profile/ncu-decoder-projections-n279/REPORT.md``).
#
# The kernel is kept, runnable, in ``experiments/gate_up_kernel.py`` beside the
# sweep that rejected it.  It is not kept here, because a dead kernel is not free:
# its import-time warming would spend the 900 s budget and its presence would be
# integrity surface on the scored path for a fast path nothing can reach.

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, rotary_emb: nn.Module | None = None,
                 bias: bool = False, qk_norm: bool = False,
                 quant_config: dict | None = None):
        super().__init__()
        self.self_attn = LlamaAttention(
            config.hidden_size, config.num_attention_heads,
            config.num_key_value_heads, config.head_dim,
            rotary_emb=rotary_emb,
            bias=bias, qk_norm=qk_norm,
            rms_norm_eps=config.rms_norm_eps,
            quant_config=quant_config,
        )
        self.mlp = LlamaMLP(config, quant_config=quant_config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual):
        # ``residual is None`` is not a rare branch here: it is the scored N=1
        # case.  It has to stay a *branch* rather than an allocation -- the tensor
        # the caller passed as ``hidden_states`` becomes the returned residual,
        # and the second norm's fused add fills it in place, so a
        # ``clone``/``empty_like`` would both cost a kernel and break the aliasing
        # the caller sees.
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        mlp = self.mlp
        # Shape lookup first, so a miss costs one integer divide, a tuple and one
        # set probe and nothing else.  The two guards behind it are only asked on a
        # hit: ``is_compiling`` because Inductor should see the frozen module's own
        # decision rather than a shape predicate specialized into its trace, and
        # ``_would_only_call_forward`` because taking this body means bypassing
        # ``nn.Module._call_impl`` and everything a caller may have installed on it.
        hidden = mlp.hidden_size
        if ((hidden_states.numel() // hidden, hidden, mlp.intermediate_size)
                in _MLP_DELEGATE_SHAPES
                and not torch.compiler.is_compiling()
                and _would_only_call_forward(mlp)):
            hidden_states = mlp.down_proj(mlp.act_fn(mlp.gate_up_proj(hidden_states)))
        else:
            hidden_states = mlp(hidden_states)
        return hidden_states, residual
