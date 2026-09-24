"""Triton fused MoE grouped GEMM kernel with FP8 W8A8 block-scaled support.

Also supports DeepGEMM m_grouped_fp8_gemm_nt_contiguous for Hopper+ GPUs
when shapes are aligned, matching vLLM's TritonOrDeepGemmExperts behavior.

Triton kernel config selection uses the bundled ``MOE_TRITON_CONFIGS`` table
(same data as vLLM's per-device JSON dumps), falling back to hardcoded defaults
when no entry matches. ``FASTKERNELS_TUNED_CONFIG_FOLDER`` JSON overrides still win.
"""

from __future__ import annotations

import functools
import json
import os

import torch
import torch.nn as nn
import triton
import triton.language as tl

import deep_gemm as _dg


_BLOCK_ALIGNMENT = 128


def _is_deep_gemm_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] >= 9


@functools.cache
def _deep_gemm_alignment() -> int:
    return _dg.get_mk_alignment_for_contiguous_layout()


def _valid_deep_gemm_shape(M: int, N: int, K: int) -> bool:
    align = _deep_gemm_alignment()
    return align <= M and N % align == 0 and K % align == 0


def _valid_deep_gemm(hidden_states: torch.Tensor, w1: torch.Tensor,
                     w2: torch.Tensor) -> bool:
    if not _is_deep_gemm_supported():
        return False
    M = hidden_states.size(0)
    _, K, N = w2.size()
    if not _valid_deep_gemm_shape(M, N, K):
        return False
    if N <= 512:
        return False
    if w1.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        return False
    if not (hidden_states.is_contiguous() and w1.is_contiguous() and w2.is_contiguous()):
        return False
    return True


def m_grouped_fp8_gemm_nt_contiguous(a_and_scale, b_and_scale, output, expert_ids):
    """Wrapper for deep_gemm.m_grouped_fp8_gemm_nt_contiguous.

    Args:
        a_and_scale: tuple of (a_fp8, a_scale)
        b_and_scale: tuple of (b_fp8, b_scale)
        output: output buffer
        expert_ids: per-row expert assignment (int32, -1 = skip)
    """
    # Match the SF format to the arch: B200 (e8m0) needs the UE8M0 cast, Hopper
    # keeps float32. Hardcoding True asserts on B200 ("Unsupported architecture
    # or scaling factor types"). Mirrors ``fp8_linear._fp8_gemm_nt_impl``.
    from .fp8_linear import _is_deep_gemm_e8m0_used
    _dg.m_grouped_fp8_gemm_nt_contiguous(
        a_and_scale, b_and_scale, output, expert_ids,
        disable_ue8m0_cast=not _is_deep_gemm_e8m0_used(),
    )


@triton.jit
def _fused_moe_kernel(
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N, K, EM,
    num_valid_tokens,
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_asm, stride_ask,
    stride_bse, stride_bsk, stride_bsn,
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    NAIVE_BLOCK_ASSIGNMENT: tl.constexpr = False,
    EXPERT_BLOCK_M: tl.constexpr = 0,
    SKIP_EMPTY: tl.constexpr = False,
    EVEN_K: tl.constexpr = False,
    EVEN_N: tl.constexpr = False,
    SCALAR_B_SCALE: tl.constexpr = False,
    INT32_ADDR: tl.constexpr = False,
    NUM_K_STEPS: tl.constexpr = 0,
    K_UNROLL: tl.constexpr = 0,
):
    """``EXPERT_BLOCK_M`` is the row granularity ``sorted_token_ids`` was padded
    to (i.e. the ``BLOCK_SIZE_M`` handed to ``moe_align_block_size``), which is
    what indexes ``expert_ids``. Decoupling it from the *compute* tile
    ``BLOCK_SIZE_M`` lets a tile cover a sub-range of one expert's padded region:
    with ~30-60 real tokens per expert but a 128-row alignment, most of every
    expert block is padding. ``SKIP_EMPTY`` then drops the sub-tiles that are
    all padding before the K loop -- those rows are already masked out of the C
    store, so skipping them is numerically identical.

    The four remaining constexprs shrink the *inner loop*, which is where this
    kernel spends everything once the block mapping is right. Measured on a B200,
    the fp8 block-scale promote (``acc += dot * a_scale[:, None] *
    b_scale[None, :]``) compiles to 244 registers/thread against 96 for a plain
    chained ``tl.dot`` -- the fp32 dot result and both broadcast operands stay
    live alongside the accumulator -- and costs 18-45% of the kernel depending on
    the shape. Each flag removes one contributor:

    * ``SCALAR_B_SCALE`` -- ``offs_bn // group_n`` is uniform across the tile when
      ``BLOCK_SIZE_N`` divides ``group_n`` and the tile is group-aligned, so
      ``b_scale`` is a *scalar*: one ``BLOCK_SIZE_M``-length ``a_scale *
      b_scale`` and one broadcast multiply instead of two full-tile ones.
    * ``EVEN_K`` -- drops ``k_mask`` from both loads (and with it a
      ``BLOCK_SIZE_K`` compare + two mask ANDs per step).
    * ``EVEN_N`` -- drops the ``% N`` wrap on ``offs_bn`` and the N bound from the
      store mask.
    * ``INT32_ADDR`` -- keeps tile/advance offsets in 32-bit. The host only sets
      it when every offset a benched shape can produce is provably < 2**31; row
      offsets are clamped to the valid range first so the padded-row sentinel
      cannot overflow the C address computation.

    ``NUM_K_STEPS`` makes the K trip count a compile-time constant (it was
    ``tl.cdiv(K, BLOCK_SIZE_K)`` on a runtime ``K``), which is worth ~6% on its
    own; ``K_UNROLL`` feeds ``tl.range``'s ``loop_unroll_factor``. Both, plus
    ``disallow_acc_multi_buffer``, are pure scheduling hints -- see ``_K_UNROLL``
    for the measured table behind the chosen values.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    if INT32_ADDR:
        offs_m = tl.arange(0, BLOCK_SIZE_M)
    else:
        offs_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)

    if NAIVE_BLOCK_ASSIGNMENT:
        offs_token = tl.where(offs_m == 0, pid_m, num_valid_tokens)
    else:
        offs_token_id = pid_m * BLOCK_SIZE_M + offs_m
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    if INT32_ADDR:
        offs_token = offs_token.to(tl.int32)
    else:
        offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    if SKIP_EMPTY:
        # An all-padding sub-tile stores nothing; bail out before the K loop.
        if tl.min(offs_token) >= num_valid_tokens:
            return

    if INT32_ADDR:
        # The padded-row sentinel is `num_valid_tokens`, whose C address
        # (sentinel * stride_cm) can exceed 2**31 even when every *real* row is
        # comfortably in range. Those rows are masked out of every load and the
        # store, so folding them onto row 0 is value-identical and keeps the
        # 32-bit address math provably in range.
        offs_token = tl.where(token_mask, offs_token, 0)

    if EXPERT_BLOCK_M == BLOCK_SIZE_M:
        off_expert = tl.load(expert_ids_ptr + pid_m)
    else:
        off_expert = tl.load(
            expert_ids_ptr + pid_m // (EXPERT_BLOCK_M // BLOCK_SIZE_M)
        )
    if INT32_ADDR:
        off_expert = off_expert.to(tl.int32)
    else:
        off_expert = off_expert.to(tl.int64)

    if INT32_ADDR:
        offs_bn_lin = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    else:
        offs_bn_lin = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    if EVEN_N:
        offs_bn = offs_bn_lin
    else:
        offs_bn = offs_bn_lin % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (off_expert * stride_be + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    if use_fp8_w8a8:
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            if SCALAR_B_SCALE:
                # BLOCK_SIZE_N divides group_n and the tile is group-aligned, so
                # every column of the tile shares one b_scale element.
                b_scale_ptrs = (
                    b_scale_ptr + off_expert * stride_bse
                    + (pid_n * BLOCK_SIZE_N // group_n) * stride_bsn
                )
            else:
                offs_bsn = offs_bn // group_n
                b_scale_ptrs = (
                    b_scale_ptr + off_expert * stride_bse + offs_bsn * stride_bsn
                )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, NUM_K_STEPS,
                      disallow_acc_multi_buffer=True,
                      loop_unroll_factor=K_UNROLL):
        if EVEN_K:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b = tl.load(b_ptrs)
        else:
            k_mask = (k * BLOCK_SIZE_K + offs_k) < K
            a = tl.load(a_ptrs, mask=token_mask[:, None] & k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)

        if use_fp8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_SIZE_K
                offs_ks = k_start // group_k
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
                if SCALAR_B_SCALE:
                    accumulator += tl.dot(a, b) * (a_scale * b_scale)[:, None]
                else:
                    accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            else:
                accumulator = tl.dot(a, b, acc=accumulator)
        else:
            accumulator = tl.dot(a.to(compute_type), b.to(compute_type), accumulator)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if use_fp8_w8a8 and not (group_k > 0 and group_n > 0):
        a_scale = tl.load(a_scale_ptr)
        b_scale = tl.load(b_scale_ptr + off_expert)
        accumulator = accumulator * a_scale * b_scale

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    c_ptrs = c_ptr + offs_token[:, None] * stride_cm + offs_bn_lin[None, :] * stride_cn
    if EVEN_N:
        c_mask = token_mask[:, None]
    else:
        c_mask = token_mask[:, None] & (offs_bn_lin[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# Bundled Triton MoE launch configs (formerly moe_configs/*.json).
# Key: (E, N, device_name, dtype|None, block_shape|None)
#   -> {M: {BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, num_warps, num_stages}}
# Override at runtime via FASTKERNELS_TUNED_CONFIG_FOLDER (JSON).
MOE_TRITON_CONFIGS: dict[
    tuple[int, int, str, str | None, tuple[int, int] | None],
    dict[int, dict[str, int]],
] = {
    (16, 1024, 'NVIDIA_B200', None, None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 5},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        1536: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 32, 'num_warps': 8, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 32, 'num_warps': 8, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
    },
    (16, 1024, 'NVIDIA_B200', 'fp8_w8a8', None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 5},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        512: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1536: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        4096: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
    },
    (16, 4096, 'NVIDIA_B200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 5},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 5},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        96: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
        128: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
        256: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 8, 'num_stages': 4},
        512: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 8, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 8, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 8, 'num_stages': 4},
    },
    (16, 14336, 'NVIDIA_B200', None, None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        96: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 4},
        128: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 4},
        200: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 4},
        256: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 4},
        512: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 4},
    },
    (20, 2560, 'NVIDIA_B200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 5},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        48: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        64: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        128: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        256: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        512: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
    },
    (20, 2560, 'NVIDIA_GB200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 5},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        48: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        64: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        96: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        128: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        256: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        512: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
    },
    (32, 1408, 'NVIDIA_B200', None, None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        128: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        512: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        2048: {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        3072: {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        4096: {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
    },
    (32, 2048, 'NVIDIA_B200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 5},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        96: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        128: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        256: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 8, 'num_stages': 5},
        512: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 8, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 8, 'num_stages': 3},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 8, 'num_stages': 3},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
    },
    (40, 1536, 'NVIDIA_B200', 'fp8_w8a8', None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 5},
        1536: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 3},
        3072: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
    },
    (40, 2560, 'NVIDIA_B200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        512: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
    },
    (40, 2560, 'NVIDIA_GB200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        128: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        512: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
    },
    (64, 512, 'NVIDIA_B200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
    },
    (64, 1408, 'NVIDIA_B200', None, None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 5},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        256: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 5},
        512: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        2048: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        4096: {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
    },
    (128, 384, 'NVIDIA_B200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        512: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
    },
    (128, 384, 'NVIDIA_GB200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        512: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
    },
    (128, 512, 'NVIDIA_B200', None, None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 5},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 2},
        2048: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        3072: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        4096: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 3},
    },
    (128, 512, 'NVIDIA_B200', 'fp8_w8a8', None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 2},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 2},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 2},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 5},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        3072: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
    },
    (128, 512, 'NVIDIA_B200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
    },
    (128, 512, 'NVIDIA_GB200', 'fp8_w8a8', None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 2},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 2},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 2},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 5},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        3072: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
    },
    (128, 704, 'NVIDIA_B200', 'fp8_w8a8', None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 5},
        4096: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
    },
    (128, 768, 'NVIDIA_B200', None, None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 2},
        512: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        1536: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        4096: {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
    },
    (128, 768, 'NVIDIA_B200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
    },
    (128, 768, 'NVIDIA_GB200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 8, 'num_stages': 3},
    },
    (128, 1856, 'NVIDIA_B200', None, None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 5},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        320: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2},
        768: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 5},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        1536: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
    },
    (160, 384, 'NVIDIA_B200', 'fp8_w8a8', None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 5},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 5},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 5},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 5},
        1536: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 5},
        8192: {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        16384: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
    },
    (160, 640, 'NVIDIA_B200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
    },
    (160, 640, 'NVIDIA_GB200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
    },
    (256, 256, 'NVIDIA_B200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 8, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 8, 'num_stages': 4},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 5},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1536: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        2048: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        3072: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        4096: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
    },
    (256, 512, 'NVIDIA_B200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
    },
    (384, 128, 'NVIDIA_B200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 2},
        1024: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1536: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
    },
    (384, 128, 'NVIDIA_GB200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1536: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
    },
    (384, 256, 'NVIDIA_B200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3},
    },
    (384, 256, 'NVIDIA_GB200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1536: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
    },
    (512, 64, 'NVIDIA_B200', None, None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1536: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        2048: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        3072: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4096: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 2},
    },
    (512, 128, 'NVIDIA_B200', None, None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1536: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        4096: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 2},
    },
    (512, 128, 'NVIDIA_B200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 2},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1536: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
    },
    (512, 128, 'NVIDIA_GB200', 'fp8_w8a8', None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        2: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        1536: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        4096: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
    },
    (512, 256, 'NVIDIA_B200', None, None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 2},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 2},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1536: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 64, 'num_warps': 8, 'num_stages': 3},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 3},
        4096: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 2},
    },
    # vLLM fused_moe/configs/E=512,N=256,device_name=NVIDIA_H200.json
    # (Qwen3-Next TP=2: 512 experts, intermediate 256 per rank). Without this
    # the Hopper path fell through to the legacy heuristic.
    (512, 256, 'NVIDIA_H200', None, None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 2},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 2},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 2},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 2},
        1024: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 2},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4096: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
    },
    (512, 256, 'NVIDIA_B200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 2},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        1536: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
    },
    (512, 256, 'NVIDIA_GB200', 'fp8_w8a8', None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 5},
        4096: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
    },
    (512, 512, 'NVIDIA_B200', None, None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 32, 'num_warps': 8, 'num_stages': 3},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        4096: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 4},
    },
    (512, 512, 'NVIDIA_B200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
    },
    (512, 512, 'NVIDIA_GB200', 'fp8_w8a8', None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 5},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 5},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 32, 'num_warps': 8, 'num_stages': 5},
        4096: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
    },
    (512, 672, 'NVIDIA_B200', None, None): {
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 5},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 5},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 5},
        1024: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 2},
    },
    (512, 1344, 'NVIDIA_B200', None, None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 2},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2},
        768: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 5},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 5},
        1536: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
    },
}

def _get_config_file_name(E: int, N: int, dtype: str | None,
                          block_shape: list[int] | None = None) -> str:
    """Override JSON filename (``FASTKERNELS_TUNED_CONFIG_FOLDER`` / vLLM checkout)."""
    device_name = torch.cuda.get_device_name().replace(" ", "_")
    if "H200" in device_name.split("_"):
        device_name = "NVIDIA_H200"
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    block_shape_selector = (
        "" if not block_shape or not all(block_shape) else f",block_shape={block_shape}"
    ).replace(" ", "")
    return f"E={E},N={N},device_name={device_name}{dtype_selector}{block_shape_selector}.json"


def _device_name() -> str:
    device_name = torch.cuda.get_device_name().replace(" ", "_")
    if "H200" in device_name.split("_"):
        device_name = "NVIDIA_H200"
    return device_name


@functools.lru_cache
def _get_moe_configs(E: int, N: int, dtype: str | None,
                     block_n: int | None = None,
                     block_k: int | None = None) -> dict[int, dict] | None:
    block_shape = [block_n, block_k] if block_n and block_k else None
    json_file_name = _get_config_file_name(E, N, dtype, block_shape)

    # User / adjacent-vLLM JSON overrides still supported.
    config_file_paths: list[str] = []
    user_folder = os.environ.get("FASTKERNELS_TUNED_CONFIG_FOLDER")
    if user_folder is not None:
        config_file_paths.append(os.path.join(user_folder, json_file_name))

    vllm_configs_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "..",
        "vllm_repo", "vllm", "vllm", "model_executor", "layers",
        "fused_moe", "configs",
    )
    if os.path.isdir(vllm_configs_dir):
        config_file_paths.append(os.path.join(vllm_configs_dir, json_file_name))

    for config_file_path in config_file_paths:
        if os.path.exists(config_file_path):
            with open(config_file_path) as f:
                tuned_config = json.load(f)
                tuned_config.pop("triton_version", None)
                return {int(key): val for key, val in tuned_config.items() if str(key).isdigit()}

    bs_key: tuple[int, int] | None = None
    if block_shape is not None and all(block_shape):
        bs_key = (int(block_shape[0]), int(block_shape[1]))
    bundled = MOE_TRITON_CONFIGS.get((E, N, _device_name(), dtype, bs_key))
    if bundled is not None:
        return dict(bundled)
    return None


    return None


_DEFAULT_CONFIG_HEURISTIC = {
    "small": {
        "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 5,
    },
    "medium": {
        "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 64, "num_warps": 4, "num_stages": 3,
    },
    "large": {
        "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 4,
    },
}


def _get_vllm_default_config(M: int, E: int = 0, dtype: str | None = None) -> dict:
    """vLLM-style BF16/FP16 MoE defaults.

    Gemma4's BF16 top-8 experts are much closer to vLLM's generic MoE path
    than to the older fastkernels heuristic, especially in decode where tokens are
    spread thinly across 128 experts.
    """
    if M <= 32:
        block_m = 16
    elif M <= 96:
        block_m = 32
    elif M <= 1024:
        block_m = 64
    else:
        block_m = 128

    block_n = 64 if M <= 64 else 128
    block_k = 128 if dtype == "fp8_w8a8" or M <= 64 else 64
    tokens_per_expert = M // max(E, 1)
    group_m = 16 if tokens_per_expert > 128 else 1
    num_warps = 4 if M <= 1024 else 8
    num_stages = 4 if M <= 32 else 3

    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": group_m,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }


def _get_default_config(M: int, E: int = 0, N: int = 0,
                        block_shape: list[int] | None = None) -> dict:
    if block_shape is not None and all(block_shape):
        return {
            "BLOCK_SIZE_M": 16 if M <= 64 else 64,
            "BLOCK_SIZE_N": block_shape[0],
            "BLOCK_SIZE_K": block_shape[1],
            "GROUP_SIZE_M": 1 if M <= 16 else 32,
            "num_warps": 4,
            "num_stages": 3,
        }
    if M <= 4:
        return dict(_DEFAULT_CONFIG_HEURISTIC["small"])
    if M <= 64:
        return dict(_DEFAULT_CONFIG_HEURISTIC["medium"])
    return dict(_DEFAULT_CONFIG_HEURISTIC["large"])


def get_triton_config(M: int, w1_shape: tuple[int, ...], w2_shape: tuple[int, ...],
                      top_k: int, use_fp8: bool,
                      block_shape: list[int] | None = None,
                      default_style: str = "legacy") -> dict:
    """Select best Triton kernel config, preferring JSON tuning files."""
    E, _, N = w2_shape
    dtype = "fp8_w8a8" if use_fp8 else None
    block_n = block_shape[0] if block_shape else 0
    block_k = block_shape[1] if block_shape else 0

    configs = _get_moe_configs(E, N, dtype, block_n, block_k)
    if configs:
        config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
        return dict(config)

    if default_style == "vllm":
        return _get_vllm_default_config(M, E, dtype)
    if default_style != "legacy":
        raise ValueError(f"Unknown MoE config style: {default_style}")
    return _get_default_config(M, E, N, block_shape)


_CONFIG_OVERRIDE: dict | None = None

# Largest compute tile in M worth using for the block-scaled FP8 path: at 128
# the fp32 accumulator plus the in-loop scale multiply blows the register budget.
_MAX_BLOCK_M = 64


def _env_config_override() -> dict | None:
    """``FK_MOE_CFG=BM,BN,BK,GM,warps,stages`` -- tuning-script escape hatch."""
    raw = os.environ.get("FK_MOE_CFG")
    if not raw:
        return _CONFIG_OVERRIDE
    keys = ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K",
            "GROUP_SIZE_M", "num_warps", "num_stages")
    return dict(zip(keys, (int(x) for x in raw.split(","))))


_MIN_BLOCK_M = 16  # smallest M any Triton MMA shape accepts


def _fit_block_m(want: int, align_block_m: int) -> int:
    """Largest power of two <= ``want`` that divides ``align_block_m``.

    A tile must not straddle two expert blocks, so ``BLOCK_SIZE_M`` has to divide
    the alignment granularity; Triton additionally needs a power of two. Falls
    back to ``align_block_m`` itself when no such tile exists -- that is the
    caller's own choice, so it is always launchable.
    """
    block_m = min(want, align_block_m)
    while block_m >= _MIN_BLOCK_M and align_block_m % block_m:
        block_m //= 2
    return block_m if block_m >= _MIN_BLOCK_M else align_block_m


_INT32_MAX = 2 ** 31 - 1


def _tensor_span(t: torch.Tensor | None) -> int:
    """Largest element index the kernel can form for *t* (0 if absent).

    Every tile offset the kernel builds is a sum of ``coord * stride`` terms with
    ``coord`` inside the tensor's own extent, so the per-dimension span is an
    upper bound for all of them. Row offsets that come from ``sorted_token_ids``
    are clamped to the valid range inside the kernel (see ``INT32_ADDR``), so the
    padded-row sentinel does not widen this bound.
    """
    if t is None:
        return 0
    return sum(max(s - 1, 0) * abs(st) for s, st in zip(t.shape, t.stride())) + 1


def _fits_int32(A, B, C, topk_weights, sorted_token_ids, expert_ids,
                a_scale, b_scale, top_k: int, EM: int) -> bool:
    """True when 32-bit offsets provably cover every address this launch forms.

    int64 tile offsets double the register footprint of the address math and
    halve its vector width. All the benched shapes are far inside 32 bits (the
    largest single span is B at ~4.0e8 elements), but the ``forward`` contract
    admits arbitrary sizes, so the fast path is gated on an actual bound rather
    than on an assumption.
    """
    spans = [
        _tensor_span(A), _tensor_span(B), _tensor_span(C),
        _tensor_span(topk_weights), _tensor_span(sorted_token_ids),
        _tensor_span(expert_ids), _tensor_span(a_scale), _tensor_span(b_scale),
        EM, A.size(0) * top_k,
    ]
    return all(0 <= s <= _INT32_MAX for s in spans)


_SMEM_STAGE_BUDGET = 160 * 1024

# `loop_unroll_factor` for the K loop. Measured on a B200 across the four
# non-trivial benched shapes (kernel-only us, BM=64/BN=128/BK=128/warps=4):
#
#   hint                                c1     c3     c2     c4
#   runtime trip count (was)          1202  138.4   95.4   97.4
#   constexpr trip count              1127  133.8   81.1   95.4
#   + disallow_acc_multi_buffer       1084  130.2   85.2  104.7
#   + unroll 2 (both hints)           1061  132.3   79.0   93.3
#
# So a compile-time trip count is worth ~6% by itself, and unroll-2 plus
# `disallow_acc_multi_buffer` is the best single setting for every shape except
# c3, where it costs 2 us. Unrolling further is not available: unroll 4 exceeds
# the shared-memory budget and a full unroll of the 3-step shapes asks for 2048
# tensor-memory columns against a 512 limit (Triton 3.6 gives every `tl.dot`
# result its own permanent tmem slot).
_K_UNROLL = 2

# Unrolling by U makes the pipeliner allocate U tiles per stage, so the
# shared-memory budget has to be divided by it.
_TARGET_STAGES = 3


def _num_k_steps(K: int, block_k: int) -> int:
    return (K + block_k - 1) // max(block_k, 1)


def _plan_launch(align_block_m: int, num_valid_tokens: int, E: int,
                 N: int, K: int, group_n: int, group_k: int,
                 naive: bool, elem_size: int = 1) -> dict:
    """Pick the *compute* tile for one grouped-GEMM launch.

    Deliberately separate from :func:`_get_default_config`: that function defines
    the row granularity ``sorted_token_ids`` was padded to (the caller passes the
    very same config to ``moe_align_block_size``), so changing it would
    invalidate the ``expert_ids`` indexing. The compute tile is free as long as
    ``BLOCK_SIZE_M`` divides ``align_block_m``.

    The M tile is driven by the *occupancy* of an expert block rather than by the
    global token count. At 128 experts a 1000-token top-8 decode batch puts ~62
    tokens in each 128-row expert block, so a 128-row tile spends half its MMA on
    padding; a 64-row tile spends almost none. Falling all the way to 16 trades
    that back for weaker MMA shapes and more B re-reads, so we stop at the
    smallest power-of-two tile that still covers a typical expert.

    64 is a hard cap even when experts are full: a 128x128 fp32 accumulator plus
    the per-K-step block-scale multiply does not fit the register budget, and a
    128-row tile measured 2.9x *slower* than a 64-row one on the 131072-token
    prefill shape -- where padding is only 6% and the tile size is the whole
    story.
    """
    override = _env_config_override()
    if override is not None:
        cfg = dict(override)
        cfg["BLOCK_SIZE_M"] = _fit_block_m(cfg["BLOCK_SIZE_M"], align_block_m)
        return cfg

    if naive:
        # One block per (token, expert) pair with a single valid row in it, so the
        # narrowest tile is strictly best; expert_ids is indexed per block, hence
        # EXPERT_BLOCK_M == BLOCK_SIZE_M here.
        block_m = _MIN_BLOCK_M
    else:
        # 64 rows even when an expert holds far fewer. Sizing the tile to the
        # occupancy of an expert block (the round-1 rule, `_pow2_at_least(
        # num_valid_tokens // E)`) is the right instinct for MMA *utilisation* but
        # the wrong one on Blackwell: Triton puts an M >= 64 fp32 accumulator in
        # tcgen05 tensor memory and an M <= 32 one in registers, so dropping below
        # 64 trades a little padding for a completely different (and much slower)
        # accumulator path. Measured on the 343-token shape, where an expert block
        # holds only ~21 rows so BM=32 is the "efficient" choice: BM=64 79 us,
        # BM=32 106 us, BM=16 106 us. Also 2.2x on the 1000-token shape.
        block_m = _fit_block_m(_MAX_BLOCK_M, align_block_m)

    # BLOCK_SIZE_K is clamped to the scale granularity by the caller anyway;
    # matching group_k exactly gives one scale load per K step instead of two.
    block_k = min(128, group_k) if group_k > 0 else 128
    while block_k > _MIN_BLOCK_M and block_k > K:
        block_k //= 2

    block_n = 128
    # An fp8 tile is 24 KiB/stage and the K loop is unrolled by `_K_UNROLL`, so
    # the pipeliner holds that many tiles per stage; the same tile in bf16 is
    # twice as big. Derive the depth from the budget rather than hardcoding it so
    # the 2-byte path shortens the pipeline instead of hitting OutOfResources
    # (unroll 2 at 5 stages already measured `OutOfResources: shared memory` on
    # the fp8 path).
    unroll = min(_K_UNROLL, max(1, _num_k_steps(K, block_k)))
    per_stage = (block_m * block_k + block_k * block_n) * elem_size * unroll
    num_stages = max(2, min(_TARGET_STAGES, _SMEM_STAGE_BUDGET // max(per_stage, 1)))

    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        # Grouping 8 M-tiles per N-sweep is worth ~6% on the 131072-token prefill
        # and ~1-2% on the smaller shapes; 16/32 are indistinguishable from 8.
        "GROUP_SIZE_M": 8,
        # 4 warps. 8 is worth ~2 us on each of the two K=4096 shapes and costs
        # ~6 us on the K=384 prefill; end to end that is +0.05% of score, i.e.
        # run-to-run noise, so it is not worth keying `num_warps` off the K length.
        "num_warps": 4,
        "num_stages": num_stages,
    }


class MoeGroupedGemm(nn.Module):
    @staticmethod
    def get_config(M: int, N: int = 0, E: int = 0,
                   use_fp8: bool = False,
                   block_shape: list[int] | None = None) -> dict:
        """Select best kernel config based on batch size M and output dim N."""
        if E > 0:
            w2_shape = (E, 0, N // 2 if N > 0 else 0)
            w1_shape = (E, N, 0)
            return get_triton_config(M, w1_shape, w2_shape, 1, use_fp8, block_shape)
        return _get_default_config(M, E, N, block_shape)

    def forward(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        topk_weights: torch.Tensor | None,
        sorted_token_ids: torch.Tensor | None,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor,
        mul_routed_weight: bool,
        top_k: int,
        config: dict | None = None,
        a_scale: torch.Tensor | None = None,
        b_scale: torch.Tensor | None = None,
        use_fp8_w8a8: bool = False,
        block_shape: list[int] | None = None,
    ):
        if config is None:
            config = _get_default_config(A.size(0), N=B.size(1))

        # The incoming config's BLOCK_SIZE_M is the granularity the caller padded
        # sorted_token_ids / indexed expert_ids with; it is a property of the
        # routing metadata, not a tunable.
        align_block_m = int(config["BLOCK_SIZE_M"])
        naive = sorted_token_ids is None

        if use_fp8_w8a8 and block_shape is not None:
            group_n, group_k = block_shape[0], block_shape[1]
        else:
            group_n, group_k = 0, 0

        config = _plan_launch(align_block_m, A.size(0) * top_k, B.size(0),
                              B.size(1), B.size(2), group_n, group_k, naive,
                              A.element_size())

        if naive:
            expert_block_m = config["BLOCK_SIZE_M"]
            EM = expert_ids.numel() * expert_block_m
        else:
            expert_block_m = align_block_m
            EM = sorted_token_ids.size(0)
            if A.size(0) < align_block_m:
                EM = min(EM, A.size(0) * top_k * align_block_m)

        grid = (
            triton.cdiv(EM, config["BLOCK_SIZE_M"]) * triton.cdiv(B.size(1), config["BLOCK_SIZE_N"]),
        )

        if use_fp8_w8a8:
            compute_type = tl.bfloat16
        elif A.dtype == torch.bfloat16:
            compute_type = tl.bfloat16
        elif A.dtype == torch.float16:
            compute_type = tl.float16
        else:
            compute_type = tl.float32

        if group_k > 0 and group_n > 0:
            config["BLOCK_SIZE_K"] = min(config["BLOCK_SIZE_K"], min(group_n, group_k))

        launch_kwargs = {}
        if "num_warps" in config:
            launch_kwargs["num_warps"] = config["num_warps"]
        if "num_stages" in config:
            launch_kwargs["num_stages"] = config["num_stages"]

        sorted_ids_ptr = sorted_token_ids if sorted_token_ids is not None else A
        a_scale_ptr = a_scale if a_scale is not None else A
        b_scale_ptr = b_scale if b_scale is not None else B

        block_n = config["BLOCK_SIZE_N"]
        block_k = config["BLOCK_SIZE_K"]
        N_, K_ = B.size(1), B.size(2)
        even_k = K_ % block_k == 0
        even_n = N_ % block_n == 0
        # b_scale is uniform across an N tile only when the tile sits inside one
        # scale group; group-alignment follows from EVEN_N plus the divisibility.
        scalar_b_scale = bool(
            use_fp8_w8a8 and group_n > 0 and group_k > 0 and even_n
            and block_n <= group_n and group_n % block_n == 0
            and b_scale is not None and b_scale.ndim == 3
        )
        int32_addr = _fits_int32(A, B, C, topk_weights, sorted_token_ids,
                                 expert_ids, a_scale, b_scale, top_k, EM)
        num_k_steps = _num_k_steps(K_, block_k)
        k_unroll = min(_K_UNROLL, max(1, num_k_steps))

        _fused_moe_kernel[grid](
            A, B, C,
            a_scale_ptr, b_scale_ptr,
            topk_weights,
            sorted_ids_ptr,
            expert_ids,
            num_tokens_post_padded,
            B.size(1), B.size(2), EM,
            A.size(0) * top_k,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(2), B.stride(1),
            C.stride(0), C.stride(1),
            a_scale.stride(0) if a_scale is not None and a_scale.ndim >= 2 else 0,
            a_scale.stride(1) if a_scale is not None and a_scale.ndim >= 2 else 0,
            b_scale.stride(0) if b_scale is not None and b_scale.ndim >= 2 else 0,
            b_scale.stride(2) if b_scale is not None and b_scale.ndim == 3 else 0,
            b_scale.stride(1) if b_scale is not None and b_scale.ndim >= 2 else 0,
            group_n=group_n,
            group_k=group_k,
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
            GROUP_SIZE_M=config["GROUP_SIZE_M"],
            NAIVE_BLOCK_ASSIGNMENT=naive,
            EXPERT_BLOCK_M=expert_block_m,
            SKIP_EMPTY=config["BLOCK_SIZE_M"] < expert_block_m,
            EVEN_K=even_k,
            EVEN_N=even_n,
            SCALAR_B_SCALE=scalar_b_scale,
            INT32_ADDR=int32_addr,
            NUM_K_STEPS=num_k_steps,
            K_UNROLL=k_unroll,
            **launch_kwargs,
        )
