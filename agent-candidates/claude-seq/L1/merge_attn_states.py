"""Online softmax merge for two attention partitions.

Combines two partial attention results (prefix and suffix) using their
log-sum-exps so the result is numerically equivalent to a single attention over
the full KV span.

Interface:

    merge(output, prefix_output, prefix_lse, suffix_output, suffix_lse,
          output_lse=None)

where ``output`` and ``output_lse`` are written in-place. ``output_lse`` may be
``None`` if the caller does not need the merged LSE (final reduction step).

The compute lives in ``merge_attn_states_fast.cu``; see the header there for the
kernel design. Anything the CUDA kernel cannot express (FP8 *inputs*, CPU
tensors, a head size that is not a multiple of the 16-byte pack) falls back to
the vendored Triton kernel, exactly as the reference implementation does.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fastkernels.infra.cuda_ext import lazy_op

_C = lazy_op("merge_attn_states_fast", "merge_attn_states_fast.cu")

_SUPPORTED_DTYPES = (torch.float32, torch.half, torch.bfloat16)


def _pack_size(dtype: torch.dtype) -> int:
    # The kernel loads/stores 128 bits (16 bytes) per memory issue, so the head
    # size must be a multiple of the pack size implied by the input dtype.
    return 4 if dtype == torch.float32 else 8


def _triton_fallback(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: torch.Tensor | None,
    prefill_tokens_with_context: int | None,
    output_scale: torch.Tensor | None,
) -> None:
    from fastkernels.tasks.baseline.L1.triton_merge_attn_states import (
        merge_attn_states as _triton_merge,
    )

    _triton_merge(
        output,
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        output_lse,
        prefill_tokens_with_context,
        output_scale,
    )


# Implements section 2.2 of https://www.arxiv.org/pdf/2501.01005
# can be used to combine partial attention results (in the split-KV case)
def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: torch.Tensor | None = None,
    prefill_tokens_with_context: int | None = None,
    output_scale: torch.Tensor | None = None,
) -> None:
    # Both the CUDA and Triton kernels derive the suffix head stride from
    # prefix_output, so suffix_output must share the same head stride.
    assert prefix_output.stride(1) == suffix_output.stride(1), (
        "merge_attn_states requires prefix_output and suffix_output to have "
        f"matching head strides, got {prefix_output.stride(1)} and "
        f"{suffix_output.stride(1)}"
    )

    # FP8 output requires output_scale to be set.
    if output.dtype not in _SUPPORTED_DTYPES:
        assert output_scale is not None, (
            f"output_scale is required when output is {output.dtype}"
        )

    # FP8 *inputs* are not supported by the CUDA kernel; fall back to Triton.
    if (
        prefix_output.is_cuda
        and prefix_output.dtype in _SUPPORTED_DTYPES
        and prefix_output.shape[2] % _pack_size(prefix_output.dtype) == 0
    ):
        _C.merge_attn_states(
            output,
            output_lse,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            prefill_tokens_with_context,
            output_scale,
        )
    else:
        _triton_fallback(
            output,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            output_lse,
            prefill_tokens_with_context,
            output_scale,
        )


class MergeAttnStates(nn.Module):
    """Online softmax merge of two attention partitions."""

    def __init__(self) -> None:
        super().__init__()
        # This runs once per attention layer per decode step on a ~40 us kernel,
        # so the Python spent per call is a real part of the latency. Bind the
        # extension entry point up front and keep ``forward`` down to a single
        # call: the kernel itself validates dtypes, shapes and layout.
        self._merge = _C.merge

    def forward(
        self,
        output: torch.Tensor,
        prefix_output: torch.Tensor,
        prefix_lse: torch.Tensor,
        suffix_output: torch.Tensor,
        suffix_lse: torch.Tensor,
        output_lse: torch.Tensor | None = None,
    ) -> None:
        try:
            self._merge(
                output,
                output_lse,
                prefix_output,
                prefix_lse,
                suffix_output,
                suffix_lse,
            )
        except RuntimeError:
            # Inputs the kernel rejects (FP8/odd head size, CPU tensors, ...)
            # take the general path, including the Triton fallback.
            merge_attn_states(
                output,
                prefix_output,
                prefix_lse,
                suffix_output,
                suffix_lse,
                output_lse,
            )
