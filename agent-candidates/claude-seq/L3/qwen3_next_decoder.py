"""Qwen3-Next decoder layer: hybrid GDN/full attention + MoE.

Dispatches to GDN linear attention or full attention based on layer type.
All layers use MoE (every layer is sparse in Qwen3-Next).
Uses GemmaRMSNorm (weight + 1 convention).

Under tensor parallelism the two parallel regions per layer (attention output,
MoE output) hand back un-reduced partials and the following norm does
all-reduce + residual-add + RMSNorm in one FlashInfer kernel -- the same fusion
vLLM's ``fuse_allreduce_rms`` pass applies
(``AllReduceFusedAddGemmaRMSNormPattern``). The MoE's partial is consumed by the
*next* layer's ``input_layernorm``, or by ``Qwen3NextModel.norm`` for the last
layer, so the model owns that half of the contract.


Why this layer is not just "the fastest kernel at every step"
-------------------------------------------------------------
The MoE picks 10 of 512 experts per token by ``topk`` on the router logits, and
the gap between the 10th and 11th logit is routinely smaller than one bf16 ulp
of the router *input*. So the layer is a chaotic map: perturb the hidden state
feeding ``mlp`` by a single ulp and ~0.5% of the output rows change by a whole
expert's contribution, which is far outside ``rtol=1e-2``. Measured on this
device, against the reference layer:

    perturbation of the MoE input        output elements outside tolerance
    one bf16 ulp (a different but        0.6%   (16384 tokens)
      equally valid rounding)
    GDN output off by 7.7e-3 relative    1.3%   (16384 tokens)

The bench allows 1%. Both halves of the layer therefore have to reproduce the
reference's *arithmetic*, not merely approximate its result -- and no
independent implementation can: the reference's own chunked delta-rule kernel
sits 3.3e-3 from an fp64 evaluation of the same recurrence (2x the error of
simply storing that answer in bf16), so "more accurate than the reference" is
just as wrong as "less accurate" as far as the router is concerned.

What that leaves, and what this file does with it:

* ``linear_attn`` -- the fused single-launch GDN path only survives the router
  at one token, where the delta rule collapses to two dot products and an outer
  product with no chunk-to-chunk accumulation to round. That is also where it
  wins most (the reference spends ~1.7 ms of host time on a one-token layer for
  ~40 us of GPU work), so the dispatch is: one token -> fused kernels, longer
  prefill -> reference recurrence. At 60 and 301 tokens the fused path is 2.2x
  faster than the reference and lands at 98.9% / 98.4% of elements inside
  tolerance, just under the 99% the bench requires -- which is what makes this a
  dispatch rather than a preference.
* ``mlp`` -- the fused MoE reproduces the reference bit-for-bit given the same
  input (it rounds its router logits to bf16 exactly where the reference's
  ``F.linear`` does), so it is used unconditionally.
* the two norms -- ``fused_add_gemma_rms_norm`` below is one kernel for what the
  reference runs as a ``torch.compile``d region, and it is *calibrated* against
  that region at the first call of each shape: it only stays enabled while it
  matches bit-for-bit, and reverts to the reference otherwise. That keeps the
  one part of the layer we can own honest about the constraint above.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice

from ....infra.tp import _tp_size
from ..L2.flashinfer_allreduce_fusion import fused_allreduce_add_gemma_rmsnorm
from ..L2.qwen3_next_gdn_attention import Qwen3NextGDNAttention
from ..L2.qwen3_next_attention import Qwen3NextAttention
from ..L2.shared_expert_moe import SharedExpertMoE

# The reference norm and GDN recurrence. These two are the places where matching
# the reference's rounding is worth more than being faster than it (module
# docstring), so they are reached deliberately rather than by fallback.
#
# ``L1.gemma_rms_norm`` is *not* imported: its fast kernel covers only the
# residual-free call, and this layer's norms both carry a residual, where it runs
# the un-compiled PyTorch path -- both slower than the reference's compiled one
# and a rounding apart from it. ``FusedGemmaRMSNorm`` below replaces it on the
# shapes it can prove itself on.
from ...baseline.L1.gemma_rms_norm import GemmaRMSNorm as _RefNorm
from ...baseline.L2.qwen3_next_gdn_attention import (
    Qwen3NextGDNAttention as _RefGDN,
)


# ###########################################################################
# Fused (residual-add +) Gemma RMSNorm -- one launch for what the reference
# runs as a ``torch.compile``d region, and bit-identical to it.
#
# Matching matters more than accuracy here (module docstring), so this kernel is
# not "a correct RMSNorm": it is the reference region's arithmetic, op for op.
# Two details of that arithmetic are easy to get wrong and both cost ~0.3% of
# the layer's output when wrong:
#
# * the residual add is nominally a bf16 add, but Inductor computes it in fp32
#   and only rounds on the store -- so the *variance* is taken over the
#   unrounded fp32 sum while the normalized value is the rounded one. Squaring
#   the rounded sum instead perturbs the output by ~1e-5 relative, which is
#   enough to move ~0.3% of elements across a bf16 rounding boundary.
# * ``tl.sum`` over the 2048-wide fp32 tile is only order-stable for a given
#   tile shape and warp count; ``_NUM_WARPS = 8`` with the whole row in one
#   block is the reference region's own choice. Other orders differ in the last
#   fp32 bit of the variance, which again lands ~1 in 4e4 elements one bf16 ulp
#   away.
#
# Where the reference reloads the residual it just stored, this keeps it in
# registers -- ``bf16(t)`` in a register and the reload of the same bf16 store
# are the same bits -- and it does the whole row in one pass rather than the
# reference's reduce-then-normalize loop pair. Four array-sized passes over HBM
# instead of five, measured 2.2x faster at 16384 rows (55 us vs 123 us), and one
# kernel launch instead of a guarded compiled call at one row (8 us vs 25 us).
# ###########################################################################

_NUM_WARPS = 8


@triton.jit(do_not_specialize=["M"])
def _add_gemma_rms_kernel(X, R, W, OUT, RES, M, N: tl.constexpr,
                          EPS: tl.constexpr, XBLOCK: tl.constexpr,
                          HAS_RES: tl.constexpr):
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < M
    rindex = tl.arange(0, N)[None, :]
    off = rindex + N * xindex
    t = tl.load(X + off, xmask, other=0.0).to(tl.float32)
    if HAS_RES:
        t += tl.load(R + off, xmask, other=0.0).to(tl.float32)
        rounded = t.to(X.dtype.element_ty)
        tl.store(RES + off, rounded, xmask)
        v = rounded.to(tl.float32)
    else:
        v = t
    acc = tl.where(xmask, t * t, 0.0)
    ssq = tl.sum(acc, 1)[:, None]
    n_f = tl.full([1, 1], float(N), tl.float32)
    scale = libdevice.rsqrt(ssq / n_f + EPS)
    gain = tl.load(W + rindex).to(tl.float32) + 1.0
    tl.store(OUT + off, (v * scale) * gain, xmask)


def fused_add_gemma_rms_norm(x, residual, weight, eps, shape):
    """``(norm(x + residual), x + residual)``, or just ``norm(x)``."""
    nrows, ncols = shape
    out = torch.empty_like(x)
    res = torch.empty_like(x) if residual is not None else out
    # One row per program is the lowest-latency shape at decode sizes; wider
    # blocks only start to pay once there are enough rows to fill the device.
    xblock = 1 if nrows < 4096 else 2
    _add_gemma_rms_kernel[((nrows + xblock - 1) // xblock,)](
        x, residual, weight, out, res, nrows, N=ncols, EPS=eps, XBLOCK=xblock,
        HAS_RES=residual is not None, num_warps=_NUM_WARPS, num_stages=1,
    )
    return (out, res) if residual is not None else out


class FusedGemmaRMSNorm(_RefNorm):
    """``GemmaRMSNorm`` on one kernel, while it provably matches the reference.

    The kernel above is bit-identical to the reference region at every shape
    this layer sees, but "bit-identical" is a property of a Triton version, a
    warp count and a row width, not something a comment can guarantee. So the
    first call at each (shape, has-residual) pair runs the reference too and
    compares exactly; only a match arms the kernel for that shape, and a
    mismatch falls back permanently.

    Everything that decision needs is checked once, during calibration: the
    steady-state path is a single shape compare against the armed shape.
    The layer is host-bound at decode sizes -- a one-token forward is ~40 us of
    GPU work against 1.7 ms of reference host time -- so ten lines of eligibility
    checking per call is not free here, it is ~14 us, comparable to the kernel
    they guard.
    """

    _SUPPORTED = (torch.bfloat16,)

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__(hidden_size, eps)
        # Exact input shape the kernel is armed for, per variant; None = none.
        # Armed on the full shape rather than the row count so a 3-D activation
        # with a matching leading dim cannot slip onto the 2-D fast path.
        self._shape_res: tuple[int, ...] | None = None
        self._shape_plain: tuple[int, ...] | None = None
        self._checked: dict[tuple[tuple[int, ...], bool], bool] = {}
        self._eps = float(eps)
        self._ncols = int(hidden_size)

    def _eligible(self, x, residual) -> bool:
        return (
            x.dtype in self._SUPPORTED
            and x.dim() == 2
            and x.is_contiguous()
            and x.is_cuda
            and x.shape[1] == self._ncols
            and self._ncols & (self._ncols - 1) == 0
            and self.weight.dtype == x.dtype
            and self.weight.is_contiguous()
            and (residual is None
                 or (residual.shape == x.shape and residual.dtype == x.dtype
                     and residual.is_contiguous()))
        )

    def _arm(self, x, residual) -> bool:
        """Arm the kernel for ``x``'s shape iff it matches the reference.

        The verdict per (shape, has-residual) is computed once and remembered;
        the armed shape is a single tuple per variant, so a workload that
        alternates shapes re-arms from the cached verdict instead of re-checking.
        """
        shape = tuple(x.shape)
        key = (shape, residual is not None)
        ok = self._checked.get(key)
        if ok is None:
            ok = False
            if self._eligible(x, residual):
                try:
                    ref = super().forward(x, residual)
                    got = fused_add_gemma_rms_norm(
                        x, residual, self.weight, self._eps, shape,
                    )
                    pairs = (zip(got, ref) if residual is not None
                             else ((got, ref),))
                    ok = all(torch.equal(a, b) for a, b in pairs)
                except Exception:  # noqa: BLE001 - no fast path is always an option
                    ok = False
            self._checked[key] = ok
        if ok:
            if residual is not None:
                self._shape_res = shape
            else:
                self._shape_plain = shape
        return ok

    def forward(self, x, residual=None):
        shape = self._shape_plain if residual is None else self._shape_res
        if x.shape == shape:
            return fused_add_gemma_rms_norm(
                x, residual, self.weight, self._eps, shape,
            )
        if torch.compiler.is_compiling() or not self._arm(x, residual):
            return super().forward(x, residual)
        shape = self._shape_plain if residual is None else self._shape_res
        return fused_add_gemma_rms_norm(
            x, residual, self.weight, self._eps, shape,
        )


class _HybridGDN(Qwen3NextGDNAttention):
    """GDN linear attention: fused kernels at one token, reference beyond it.

    ``Qwen3NextGDNAttention`` (L2) is a rewrite of the whole block as Triton
    kernels and is correct *as an operator* -- its output sits 7.7e-3 from the
    reference's, comfortably inside bf16 tolerance. Inside this layer that
    output is the router's input two ops later, and 7.7e-3 there costs 1.3% of
    the layer's output elements (module docstring). A single token is the
    exception: there is one chunk of one position, so the chunked transform the
    error comes from degenerates and the fused kernels land on the same bf16
    values the reference does (measured: every element inside tolerance, on the
    layer output, at both benched one-token shapes).
    """

    def forward_impl(self, hidden_states: torch.Tensor, state_manager=None):
        if hidden_states.numel() == self.hidden_size:
            return Qwen3NextGDNAttention.forward_impl(
                self, hidden_states, state_manager,
            )
        return _RefGDN.forward_impl(self, hidden_states, state_manager)


def fused_ar_norm(norm, hidden_states, residual, fuse: bool):
    """``norm(all_reduce(hidden_states), residual)``, fused when ``fuse``."""
    if fuse:
        # Opaque under torch.compile: tracing into FlashInfer's fused
        # collective hits Python logging / datetime and aborts Dynamo.
        if torch.compiler.is_compiling():
            return torch.ops.fastkernels.fused_allreduce_add_gemma_rmsnorm(
                hidden_states, residual, norm.weight, float(norm.variance_epsilon),
            )
        return fused_allreduce_add_gemma_rmsnorm(hidden_states, residual, norm)
    return norm(hidden_states, residual)


class Qwen3NextDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        # Only worth deferring when there is a collective to defer.
        self.fuse_ar_norm = _tp_size() > 1

        if self.layer_type == "linear_attention":
            self.linear_attn = _HybridGDN(
                hidden_size=config.hidden_size,
                num_k_heads=config.linear_num_key_heads,
                num_v_heads=config.linear_num_value_heads,
                head_k_dim=config.linear_key_head_dim,
                head_v_dim=config.linear_value_head_dim,
                layer_idx=layer_idx,
                conv_kernel_size=config.linear_conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                reduce_output=not self.fuse_ar_norm,
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                layer_idx=layer_idx,
                rms_norm_eps=config.rms_norm_eps,
                reduce_output=not self.fuse_ar_norm,
            )
        else:
            raise ValueError(f"Invalid layer_type: {self.layer_type}")

        # MoE for all Qwen3-Next layers (every layer is sparse).
        self.mlp = SharedExpertMoE(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            moe_intermediate_size=config.moe_intermediate_size,
            routing="softmax",
            correction_bias=False,
            renormalize=config.norm_topk_prob,
            routed_scaling_factor=1.0,
            shared_expert_intermediate_size=config.shared_expert_intermediate_size,
            shared_expert_attr_name="shared_expert",
            shared_expert_gate=True,
            reduce_results=not self.fuse_ar_norm,
        )

        self.input_layernorm = FusedGemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = FusedGemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, hidden_states, residual, positions=None,
                rotary_emb=None, state_manager=None):
        if residual is None:
            # Layer 0: the input is the vocab-parallel embedding's output, which
            # is already reduced, and there is no residual stream yet.
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = fused_ar_norm(
                self.input_layernorm, hidden_states, residual, self.fuse_ar_norm,
            )

        # Attention
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states, state_manager=state_manager,
            )
        else:
            hidden_states = self.self_attn(
                hidden_states, rotary_emb=rotary_emb, positions=positions,
                state_manager=state_manager,
            )

        # Post-attention norm + MLP
        hidden_states, residual = fused_ar_norm(
            self.post_attention_layernorm, hidden_states, residual,
            self.fuse_ar_norm,
        )
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual
