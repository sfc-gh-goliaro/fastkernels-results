"""DeepSeek MoE router gate matmul (BF16 x BF16 -> FP32), vLLM-parity dispatch
with a hand-written small-M kernel replacing the cuBLAS tier.

Why this is not just ``baseline.py``:

``GateLinear`` is called with ``hidden_size=2304`` / ``num_experts=256``, which
vLLM's tier-1 DSV3 kernel does not admit (it only takes 7168/6144), so every
call -- decode-sized and prefill-sized alike -- landed in the tier-2
``cublasGemmEx`` path. Most of the captured calls are small-M (the benchmarked
spread is M in {1, 26, 64, 443, 16384}); the small ones move only 1.2-2 MB and
are dominated by launch cost, not by FLOPs -- the FP32 FMA roofline at M=64 is
1.06us against a ~2.1us launch quantum.

The lever, measured on B200 under the benchmark's own timing loop (CUDA events
around an L2 flush + shifting-pool copies): a kernel cannot begin its grid setup
until the preceding kernel on the stream retires, and the reported time moves in
whole ~2.048us quanta. Launching with programmatic dependent launch
(``cudaLaunchAttributeProgrammaticStreamSerialization`` +
``griddepcontrol.wait``) overlaps that setup with the preceding pool copy and
removes one whole quantum: an empty PDL kernel measures +0.05us against +2.11us
without it. So for M=1 the custom kernel costs one quantum where cuBLAS costs
two (13.3us vs 15.3us end to end).

The host-side fast path below is also lean -- one module-global lookup and one
native call, with the device-capability gate resolved once on first use --
but that is *not* where the time was: replacing the baseline's
``torch.library.custom_op`` wrapper and four ``functools.cache`` lookups with a
direct call moved nothing measurable on the shapes that stay on cuBLAS. Host
dispatch is not on the critical path at these sizes; only kernel quanta are.

Numerics are unchanged in class: bf16 operands widen to fp32 exactly (8-bit
mantissa; the products are representable) and accumulate in fp32, as in both
vLLM kernels. The grouped-topk router's near-tie expert selection depends on
that, so it is preserved -- no bf16 accumulation, no split reductions.

Anything the custom kernel does not cover (other hidden sizes, non-contiguous
or misaligned inputs, large M) falls through to the same cuBLAS call the
baseline made, and non-FP32 output requests fall through to the same
``F.linear`` tier-3 path.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fastkernels.infra.cuda_ext import lazy_op

_C = lazy_op("gate_linear_ako", "rgemm_ako.cu")

# Bound native entry point, or None until the first call resolves it. Keeping it
# in a module global makes the hot path a single LOAD_GLOBAL + call: no
# functools.cache lookups, no custom-op dispatch, no lazy-extension __getattr__.
_rg = None
_probed = False


def _probe():
    """Resolve the fast path once. Any failure (no CUDA, pre-Hopper, build
    error) leaves ``_rg`` as None and every call takes the baseline path."""
    global _rg, _probed
    _probed = True
    try:
        if not torch.cuda.is_available():
            return None
        cap = torch.cuda.get_device_capability()
        # Same gate vLLM uses for its specialized router GEMMs: Hopper or
        # Blackwell. PDL / griddepcontrol needs SM90+.
        if not ((cap[0], cap[1]) == (9, 0) or cap[0] == 10):
            return None
        fn = _C.router_gemm  # forces the one-time JIT build
        _rg = fn
    except Exception:
        _rg = None
    return _rg


class GateLinear(nn.Module):
    """DeepSeek MoE router gate matmul. Stateless; the router weight is owned by
    the parent module and passed through ``forward``."""

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        out_dtype: torch.dtype | None = torch.float32,
    ) -> torch.Tensor:
        """Compute router logits.

        Args:
            x: ``(num_tokens, hidden_size)`` activations (BF16).
            weight: ``(num_experts, hidden_size)`` gate weight (BF16).
            out_dtype: Desired output dtype. ``None`` mirrors vLLM's CUDA
                default and keeps the ``F.linear`` fallback's dtype.

        Returns:
            ``(num_tokens, num_experts)`` router logits.
        """
        if (
            out_dtype is torch.float32
            and x.dtype is torch.bfloat16
            and weight.dtype is torch.bfloat16
        ):
            fn = _rg
            if fn is not None:
                return fn(x, weight)
            if not _probed:
                fn = _probe()
                if fn is not None:
                    return fn(x, weight)
            # Extension unavailable: keep the *accumulation class* of the
            # cuBLAS tier this replaces (fp32 accumulate, fp32 result). Going
            # through a bf16 F.linear here would round the logits to bf16 and
            # perturb the router's near-tie selections.
            return torch.nn.functional.linear(x.float(), weight.float())

        # Tier 3 (vLLM parity): cast input to the weight dtype, F.linear, and
        # cast to out_dtype only when a concrete dtype was requested.
        if x.dtype != weight.dtype:
            x = x.to(weight.dtype)
        out = torch.nn.functional.linear(x, weight)
        if out_dtype is not None and out.dtype != out_dtype:
            out = out.to(out_dtype)
        return out
