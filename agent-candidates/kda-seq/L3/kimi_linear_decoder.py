"""Kimi-Linear decoder layer composed over the frozen lower-level winners.

No new kernel lives here. The measured win at this level is entirely in *which*
existing implementation each stage reaches, so the ``forward`` body is the
baseline's, unchanged, and every stage is bound to a frozen file by import:

    ..L1.rms_norm             -> candidate/L1/rms_norm.py
    ..L2.kimi_delta_attention -> candidate/L2/kimi_delta_attention.py
    ..L2.kimi_mla_attention   -> candidate/L2/kimi_mla_attention.py
    ..L2.llama_mlp            -> candidate/L2/llama_mlp.py
    ..L2.trtllm_bf16_moe      -> candidate/L2/trtllm_bf16_moe.py
    ..L2.kimi_moe             -> baseline (no candidate file exists)

``kimi_moe`` is the one stage with no winner of its own, and it is where the
remaining work is. The candidate import finder resolves a missing candidate file
by handing back **the baseline module object itself**, so the aliased
``KimiMoE``'s own module-level imports stay inside the baseline package: the MoE
a plain composition would get is pinned to ``baseline.L2.trtllm_bf16_moe`` and
``baseline.L2.llama_mlp``, leaving both winners unused. :class:`KimiMoE` below
re-points those two submodules.

Per-stage measurement, baseline -> frozen, on the five scored cases
(``profile/headroom.json``; n = 1/16384/611/64/1):

    input norm        1.00x  1.10x  1.10x  1.00x  1.00x
    attention         2.11x  0.99x  2.19x  2.19x  2.22x
    post-attn norm    0.99x  1.10x  1.00x  1.00x  1.00x
    mlp / moe         1.12x  1.32x  1.08x  1.52x  1.57x

Composed end to end and measured on this file across repeated independent GPU
leases: 0 skipped scenarios on every lease, and on every lease that passed the
gate, 5/5 PASSED with a geomean above 1.0 and a median above 1.6x. Per-case
speedup is roughly 1.9 / 1.2 / 1.9 / 1.7 / 1.6, i.e. large on the four host-bound
cases and small on the device-bound n=16384 one, which is where the frozen KDA
changes nothing (see below). One lease failed two cases on the upstream defect
described below -- not a candidate fault, and recorded rather than dropped. The
authoritative, regenerated figures -- per-lease geomeans, per-case medians and
ranges, case 2 as a range, and the failed lease in full -- live in
``docs/performance.md``, produced by ``experiments/summarize_leases.py`` from the
retained ``profile/bench_lease*.json``. Deliberately not restated here as
literals: the first version of this docstring carried hand-copied numbers that
went stale as soon as more leases landed.

Two things about those measurements are worth keeping close to the code.

**Case 2 gains little, and the capture bounds why.** An A/B NCU capture of the two
dominant case-2 kernels, taken from the baseline layer and from this one
(``profile/kda_case2_ncu_candidate/REPORT.md``), finds identical launch geometry,
register pressure, occupancy and store-sector counts, and 0.00% tensor pipe
utilisation on both. What that establishes is bounded and worth stating exactly:
**those two kernels are unchanged**, so whatever case 2 gains comes from
elsewhere in the layer. It does not establish that the frozen KDA's benefit is
host-side, and that would be the wrong conclusion -- the frozen file also fuses
four input projections into one GEMM plus one batched GEMM and folds three convs,
two L2 norms and a sigmoid into a single Triton kernel, all of which is device
work. Its own measurement of the projection fusion is 42-47 us -> 12-19 us at
n <= 443 but 622 us -> 664 us at n=16384, i.e. a small regression at exactly this
shape. Per-case device time was never measured for this layer, so the split
between host and device gains is unquantified here and is left that way rather
than guessed.

**``A_log`` is drawn fresh on every rebuild, and it drives both the error
magnitude and an outright failure mode.** It is a ``torch.empty`` parameter that
``_sanitize_float_params`` only re-initialises when ``|amax|`` is non-finite,
below 1e-6 or above 1e4, so each rebuild keeps whatever garbage lands inside that
band. fp32 ``exp`` overflows above ~88.7, so a survivor in ``(88.7, 1e4]`` makes
``-exp(A_log)`` ``-inf`` and the layer returns NaN from ``o_norm`` outward.
Observed in 2 of 55 retained case builds, both inside the same lease; a further 40
natural rebuilds in one process produced none. That is an observed incidence, not
a rate -- the clustering is unexplained and two events do not establish one.

Two consequences, both measured rather than argued:

* Milder survivors make the recurrence sharply sensitive, which is what turns a
  1.56e-2 per-stage difference into a ~0.6 ``max_abs`` outlier on a handful of
  elements. The worst ``matched_ratio`` observed anywhere is ~0.9999 against a
  required 0.99. A surrogate with ``A_log`` pinned to ``normal_(0, 0.02)``
  (``profile/composed_estimate.json``) reports 1.0000 at ``max_abs`` 1.56e-2 for
  the same case, and most real leases agree with it.
* A poisoned draw hits **both sides identically**, because the harness shares
  weights from the baseline into the candidate:
  ``experiments/diag_nan_sides.py`` finds ``A_log`` identical on both sides in
  40/40 natural rebuilds, and reproducing the observed 2.96e3 value yields equal
  NaN counts on reference and candidate with zero candidate-only occurrences. So
  neither the outlier nor the NaN is a divergence in what this file computes, and
  neither is reachable from it -- the baseline is constructed first and the
  reference is computed from it.

The aliased ``..L2.kimi_moe`` module is the baseline module object, so rebinding
a name on it would change what the *reference* side builds and make the
correctness comparison compare the candidate against itself. Nothing here
rebinds anything on it; the swap is done by subclassing, in a constructor.
"""

from __future__ import annotations

import torch.nn as nn

from ..L1.rms_norm import RMSNorm
from ..L2.kimi_delta_attention import KimiDeltaAttention
from ..L2.kimi_mla_attention import KimiMLAAttention
from ..L2.kimi_moe import KimiMoE as _BaselineKimiMoE
from ..L2.llama_mlp import LlamaMLP
from ..L2.trtllm_bf16_moe import ROUTING_DEEPSEEK_V3, TrtLlmBf16MoE


class KimiMoE(_BaselineKimiMoE):
    """Baseline ``KimiMoE`` with its two pinned submodules re-pointed at the winners.

    Subclassed rather than reimplemented on purpose. The base ``__init__`` owns
    the router bias, the four weight loaders, the ``w13``/``w2`` shapes and the
    ``process_weights_after_loading`` BlockMajorK shuffle; restating any of that
    risks a state-dict key or layout divergence that the harness's
    ``load_state_dict(strict=False)`` would swallow without a word. The cost is
    one discarded shared-expert ``LlamaMLP`` per layer at construction (~14 MB,
    returned to the allocator cache).

    The class name has to stay ``KimiMoE``: ``auto_register_no_compile_layers``
    matches submodules by ``type(mod).__name__`` against a literal set, and
    ``enable_custom_ops`` only ever flips ``_use_custom_op`` on a module that
    matched and was registered. A rename silently drops the compile boundary.

    Both swaps happen here, in the constructor, never afterwards: weight sharing
    and the dtype cast both rebind ``param.data`` once every module is built, so
    a late replacement would be left running on its own random init.

    ``trtllm_moe`` is worth 3.777 -> 2.872 ms at n=16384, 0.904 -> 0.598 at
    n=64 and 0.879 -> 0.569 at n=1: the frozen runner holds its ``MoERunner``
    instead of rebuilding it and its ``AutoTuner`` on every call. The
    shared-expert swap is worth ~0.013 ms once that is in place and is taken
    because it is measured non-negative everywhere and because the frozen
    ``LlamaMLP``'s parameter names are identical to the baseline's, not as a
    claimed win.

    Nothing added here is derived from weight *values*. The harness shares
    weights into existing storage and then runs ``process_weights_after_loading``,
    which *replaces* ``w13``/``w2``, so a value-derived attribute set in a
    constructor would answer for the pre-sharing random init.
    """

    def __init__(self, config, quant_config: dict | None = None):
        super().__init__(config, quant_config=quant_config)

        # ``forward_impl`` branches on ``use_trtllm`` while the runner lives in
        # ``trtllm_moe``; both derive from the same support probe, so guard on
        # both and let a future divergence surface here rather than as a
        # silently unswapped runner.
        assert self.use_trtllm == (self.trtllm_moe is not None), (
            "use_trtllm disagrees with trtllm_moe presence; the frozen runner "
            "would be swapped into a path that never calls it"
        )
        if self.use_trtllm:
            self.trtllm_moe = TrtLlmBf16MoE(
                num_experts=self.num_experts,
                top_k=self.top_k,
                intermediate_size_per_partition=self.intermediate_per_tp,
                routing_method_type=ROUTING_DEEPSEEK_V3,
                num_expert_group=self.num_expert_group,
                topk_group=self.topk_group,
                routed_scaling_factor=self.routed_scaling_factor,
            )

        # None when ``num_shared_experts`` is 0. The intermediate size and
        # ``reduce_results=False`` reproduce the base constructor's arguments
        # exactly, so the parameter tree does not move.
        if self.shared_experts is not None:
            self.shared_experts = LlamaMLP(
                config,
                quant_config=quant_config,
                intermediate_size=(
                    config.moe_intermediate_size * self.num_shared_experts
                ),
                reduce_results=False,
            )


class KimiLinearDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int, quant_config: dict | None = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_kda = config.is_kda_layer(layer_idx)

        if self.is_kda:
            # The frozen KDA subclasses the baseline class, which is what keeps
            # the harness's ``isinstance(sub, baseline KimiDeltaAttention)``
            # lookup working. A same-named standalone class here would make
            # every case report ``no KDA/GDN submodule found``.
            self.self_attn = KimiDeltaAttention(
                config,
                layer_idx=layer_idx,
                quant_config=quant_config,
            )
        else:
            self.self_attn = KimiMLAAttention(
                config,
                quant_config=quant_config,
            )

        if config.is_moe_layer(layer_idx):
            self.block_sparse_moe = KimiMoE(config, quant_config=quant_config)
            # One object under both names, as in the baseline: ``state_dict()``
            # then carries both prefixes and the harness's name-matched weight
            # sharing stays exact.
            self.mlp = self.block_sparse_moe
        else:
            self.mlp = LlamaMLP(config, quant_config=quant_config)

        # ``eps`` is not optional here: the frozen RMSNorm defaults to 1e-6 and
        # this config wants 1e-5.
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
        )

    def forward(self, hidden_states, residual, state_manager=None):
        # Identical to the baseline body. It is the only version whose numerics
        # are known to match, and the stage breakdown says the body itself costs
        # nothing at any scored shape: the two norms together are under 40 us,
        # and the entry clone is 4.6 KB at n=1 and 2.8 MB at n=611 -- at or
        # below the harness's ~2.04 us timing quantisation either way.
        #
        # Aliasing ``residual = hidden_states`` instead of cloning would remove
        # the copy, but the returned residual would then alias the caller's
        # input buffer, which the bench's shifting pool rewrites between
        # iterations.
        if residual is None:
            residual = hidden_states.clone()
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(hidden_states, state_manager=state_manager)
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual,
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
