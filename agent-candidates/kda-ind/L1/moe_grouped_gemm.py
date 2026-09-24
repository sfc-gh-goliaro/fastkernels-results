"""Fused-MoE grouped GEMM specialized for DeepSeek-style blockwise FP8 on sm_100.

The operator is a drop-in replacement for the baseline vLLM-style kernel: for
each padded row slot ``r``, ``t = sorted_token_ids[r]`` names a token-expert pair
and ``e = expert_ids[r // align_m]`` the expert whose weights that slot's block
uses.  Slots with ``t >= num_valid_tokens`` are padding and write nothing.

Three kernels sit behind one host-side dispatch:

``_moe_blockwise_fp8_kernel``
    The path taken by every benchmarked shape except the decode one
    (``use_fp8_w8a8`` with ``block_shape == [128, 128]``).  It differs from the
    reference kernel in four ways that matter: the K tile equals the scale group, so
    each accumulator element is promoted once per 128 of K instead of twice; the N
    tile is a whole multiple of the scale group, so the weight scale is a handful of
    scalars rather than a tile-wide vector; the M tile may be smaller than the
    metadata's alignment and a tile of pure padding leaves before any MMA, which is
    where the sparsely-occupied expert blocks are recovered; and addressing is
    32-bit.

``_moe_thin_decode_kernel``
    The decode regime, where there are too few token-expert pairs to fill the
    device and the shape is bound by a memory-latency chain rather than by
    arithmetic.  Same work, spread over more programs: it allows an N tile *below*
    the weight scale group, which the kernel above forbids.

``_moe_reference_kernel``
    The baseline's semantics, unchanged, for every other mode -- non-fp8 compute
    types, per-tensor scales, other block shapes, the naive block assignment,
    ragged K/N, and index ranges that do not fit in 32 bits.  Correctness for
    those modes is a requirement; speed on them is not.

The two experiment-only knobs below (``_TILE_OVERRIDE``, ``_FORCE_REFERENCE``) are
read inside memoized tile choosers, so a steady-state call never consults them.
"""

from __future__ import annotations

import functools

import torch
import torch.nn as nn
import triton
import triton.language as tl

from fastkernels.tasks.baseline.L1.moe_grouped_gemm import (
    _get_default_config,
    get_triton_config,
)

# DeepSeek blockwise quantization: activations 1x128, weights 128x128.  The fast
# path is selected only for exactly this scheme, because both the single-promotion
# K tile and the scalar weight scale are derived from it.
_SCALE_GROUP = 128

# Accumulator registers per thread that a tile may ask for.  See _tile_is_admissible.
_MAX_ACCUMULATOR_REGISTERS = 128

# Element indices are computed in 32-bit on the fast path (base pointers stay
# 64-bit).  The margin covers the one extra tile-stride advance the K loop makes
# past the final load.
_INT32_MARGIN = 1 << 24
_INT32_LIMIT = (1 << 31) - 1 - _INT32_MARGIN


@triton.jit
def _moe_blockwise_fp8_kernel(
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    K, num_pid_m, num_pid_n,
    num_valid_tokens,
    stride_am,
    stride_be, stride_bn,
    stride_cm,
    stride_asm,
    stride_bse, stride_bsn,
    GROUP_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ALIGN_M: tl.constexpr,
    N_SCALE_GROUPS: tl.constexpr,
    EVEN_K: tl.constexpr,
    SKIP_EMPTY_TILES: tl.constexpr = True,
):
    """Blockwise-FP8 grouped GEMM, one scale promotion per 128 of K.

    The argument list is deliberately short.  The decode shape spends about 17 of
    its 24 host microseconds inside Triton's launch path, which scales with the
    number of arguments, so the innermost stride of every tensor is required to be
    1 (checked on the host, which falls back to the reference kernel otherwise)
    and the grid dimensions arrive pre-divided rather than as N and EM.
    """
    # The sub-tile has to lie wholly inside one alignment block, or part of it
    # would take its expert id from the neighbouring block.
    tl.static_assert(ALIGN_M % BLOCK_SIZE_M == 0)
    tl.static_assert(BLOCK_SIZE_M <= ALIGN_M)
    # One scale promotion per group requires the K tile to *be* the group.
    tl.static_assert(BLOCK_SIZE_K == GROUP_K)
    # The N tile must be a whole number of scale groups, or the weight scale is not
    # constant across the slice it is applied to.  Stated as an equality rather than
    # a modulo so it also pins N_SCALE_GROUPS, which the caller derives separately.
    tl.static_assert(BLOCK_SIZE_N == N_SCALE_GROUPS * GROUP_K)

    pid = tl.program_id(axis=0)
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    # This guard must precede every metadata load.  MoeAlign fills expert_ids only
    # over cdiv(num_tokens_post_padded, ALIGN_M) entries and leaves the rest of the
    # torch.empty buffer undefined; the B load below is not masked by token_mask,
    # so a garbage expert id is a wild address, not a wrong number.
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    tile_start = pid_m * BLOCK_SIZE_M
    if tile_start >= num_tokens_post_padded:
        return

    offs_token = tl.load(sorted_token_ids_ptr + tile_start + tl.arange(0, BLOCK_SIZE_M))
    token_mask = offs_token < num_valid_tokens
    # A tile holding nothing but padding stores nothing (the store mask is
    # token_mask & ...), so returning here changes no value.  With 128 experts and
    # a few thousand token-expert pairs most alignment blocks are mostly padding,
    # which is what makes this branch worth its cost.  SKIP_EMPTY_TILES exists so
    # that "changes no value" can be tested rather than asserted: tools/audit.py
    # compiles the same kernel with it off and compares bit patterns.
    if SKIP_EMPTY_TILES:
        if tl.max(token_mask.to(tl.int32)) == 0:
            return

    # Loading the expert id only after the padding-tile test keeps an all-padding
    # tile from paying for it, which measured better on both the decode and the
    # low-occupancy shape than issuing the two metadata loads together.
    off_expert = tl.load(expert_ids_ptr + tile_start // ALIGN_M)

    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_row = offs_token // top_k

    a_ptrs = a_ptr + (offs_row[:, None] * stride_am + offs_k[None, :])
    b_ptrs = b_ptr + (off_expert * stride_be + offs_k[:, None]
                      + offs_bn[None, :] * stride_bn)

    a_scale_ptrs = a_scale_ptr + offs_row * stride_asm
    # BLOCK_SIZE_N is a whole multiple of the weight scale group, so the scale
    # column index is constant across each group-wide slice of the tile: at
    # N_SCALE_GROUPS == 1 the weight scale is one scalar for the whole tile, and
    # the promotion costs a multiply and an add per accumulator element instead of
    # the reference kernel's two multiplies and an add.
    b_scale_ptrs = b_scale_ptr + off_expert * stride_bse + pid_n * N_SCALE_GROUPS * stride_bsn
    if N_SCALE_GROUPS > 1:
        b_scale_ptrs = b_scale_ptrs + (tl.arange(0, BLOCK_SIZE_N) // GROUP_K) * stride_bsn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if EVEN_K:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b = tl.load(b_ptrs)
        else:
            k_mask = (k * BLOCK_SIZE_K + offs_k) < K
            a = tl.load(a_ptrs, mask=token_mask[:, None] & k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        # Derived rather than assumed: with BLOCK_SIZE_K == GROUP_K this folds to
        # k, and the fast path is only selected when that holds.
        offs_ks = k * BLOCK_SIZE_K // GROUP_K
        a_scale = tl.load(a_scale_ptrs + offs_ks, mask=token_mask, other=0.0)
        b_scale = tl.load(b_scale_ptrs + offs_ks)
        if N_SCALE_GROUPS == 1:
            accumulator += tl.dot(a, b) * (a_scale * b_scale)[:, None]
        else:
            accumulator += tl.dot(a, b) * (a_scale[:, None] * b_scale[None, :])
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator = accumulator * moe_weight[:, None]

    c_ptrs = c_ptr + offs_token[:, None] * stride_cm + offs_bn[None, :]
    tl.store(c_ptrs, accumulator.to(compute_type), mask=token_mask[:, None])


@triton.jit
def _moe_thin_decode_kernel(
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    K, num_pid_m, num_pid_n,
    num_valid_tokens,
    stride_am,
    stride_be, stride_bn,
    stride_cm,
    stride_asm,
    stride_bse, stride_bsn,
    SCALE_GROUP: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    ALIGN_M: tl.constexpr,
):
    """Decode specialization: the same arithmetic, spread over more programs.

    The decode shape is not short of arithmetic -- a handful of valid rows against
    12 MB of weights -- it is short of programs.  It runs about half a wave, so its
    duration is a memory-latency chain with almost nothing to overlap it with.  The
    only axis this operator may be split along more finely is N: splitting K would
    need independent programs to accumulate partial sums into a bf16 ``C`` that
    arrives holding arbitrary prior contents, which takes atomics, a workspace, or a
    second reduction pass.

    So ``BLOCK_SIZE_N`` here is allowed to be *smaller* than the weight scale group,
    which the general fast path forbids.  That stays correct because a
    ``BLOCK_SIZE_N``-wide tile aligned to ``BLOCK_SIZE_N`` lies wholly inside one
    scale group whenever ``BLOCK_SIZE_N`` divides it, so the scale is still a single
    scalar -- taken at ``pid_n * BLOCK_SIZE_N // SCALE_GROUP`` rather than at
    ``pid_n``.
    """
    tl.static_assert(ALIGN_M % BLOCK_SIZE_M == 0)
    tl.static_assert(BLOCK_SIZE_M <= ALIGN_M)
    tl.static_assert(BLOCK_SIZE_K == SCALE_GROUP)
    # One scalar per tile needs the tile to sit inside a single scale group.
    tl.static_assert(SCALE_GROUP % BLOCK_SIZE_N == 0)

    pid = tl.program_id(axis=0)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Same ordering obligation as the fast path: the capacity guard precedes every
    # metadata load, because expert_ids is uninitialized past the padded length.
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    tile_start = pid_m * BLOCK_SIZE_M
    if tile_start >= num_tokens_post_padded:
        return

    offs_token = tl.load(sorted_token_ids_ptr + tile_start + tl.arange(0, BLOCK_SIZE_M))
    token_mask = offs_token < num_valid_tokens
    if tl.max(token_mask.to(tl.int32)) == 0:
        return

    off_expert = tl.load(expert_ids_ptr + tile_start // ALIGN_M)

    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_row = offs_token // top_k

    a_ptrs = a_ptr + (offs_row[:, None] * stride_am + offs_k[None, :])
    b_ptrs = b_ptr + (off_expert * stride_be + offs_k[:, None]
                      + offs_bn[None, :] * stride_bn)
    a_scale_ptrs = a_scale_ptr + offs_row * stride_asm
    b_scale_ptrs = (b_scale_ptr + off_expert * stride_bse
                    + (pid_n * BLOCK_SIZE_N // SCALE_GROUP) * stride_bsn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K // BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        b = tl.load(b_ptrs)
        a_scale = tl.load(a_scale_ptrs + k, mask=token_mask, other=0.0)
        b_scale = tl.load(b_scale_ptrs + k)
        accumulator += tl.dot(a, b) * (a_scale * b_scale)[:, None]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator = accumulator * moe_weight[:, None]

    c_ptrs = c_ptr + offs_token[:, None] * stride_cm + offs_bn[None, :]
    tl.store(c_ptrs, accumulator.to(compute_type), mask=token_mask[:, None])


@triton.jit
def _moe_reference_kernel(
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
):
    """The reference semantics, for every mode outside the fast path.

    Kept deliberately close to the baseline kernel: this is the correctness
    fallback for modes the benchmark never times, so being recognizably the same
    code is worth more here than being fast.  The M tile always equals the
    metadata alignment, which is why ``expert_ids`` is indexed by ``pid_m``
    directly and why the naive assignment -- where each block owns exactly one
    row and alignment semantics do not apply -- can share this kernel.
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

    offs_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)

    if NAIVE_BLOCK_ASSIGNMENT:
        offs_token = tl.where(offs_m == 0, pid_m, num_valid_tokens)
    else:
        offs_token_id = pid_m * BLOCK_SIZE_M + offs_m
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (off_expert * stride_be + offs_k[:, None] * stride_bk
                      + offs_bn[None, :] * stride_bn)

    if use_fp8_w8a8:
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = b_scale_ptr + off_expert * stride_bse + offs_bsn * stride_bsn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask = (k * BLOCK_SIZE_K + offs_k) < K
        a = tl.load(a_ptrs, mask=token_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)

        if use_fp8_w8a8:
            if group_k > 0 and group_n > 0:
                offs_ks = k * BLOCK_SIZE_K // group_k
                a_scale = tl.load(a_scale_ptrs + offs_ks * stride_ask,
                                  mask=token_mask, other=0.0)
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
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
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_token[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# Experiment-only knobs, both read inside the memoized ``_fast_tile`` so the timed
# path never consults them: a cache hit does not re-enter the function.  Anything
# that sets one must call ``_fast_tile.cache_clear()``.
#
# ``_TILE_OVERRIDE`` forces one tile geometry (tools/bench_kernel.py, tools/audit.py).
# ``_FORCE_REFERENCE`` sends every call to the reference kernel, which is how the
# ablation control is reproduced: the reference kernel is the control every measured
# delta in benchmark.csv is taken against, so it has to be reachable on demand and
# verifiable as a candidate in its own right (tools/verify.py --force-reference).
_TILE_OVERRIDE: tuple[int, int, int, int, int, int] | None = None
_FORCE_REFERENCE = False


def _tile_is_admissible(tile, N: int, K: int, align_m: int) -> bool:
    """Every precondition the blockwise-FP8 kernel relies on, in one place.

    These are not preferences.  The kernel promotes each accumulator element once
    per scale group, which is only the same thing as once per K tile when the two
    are equal; and it applies one weight-scale scalar per ``GROUP_K``-wide slice of
    the N tile, which is only correct when the tile is a whole number of such
    slices.  The kernel asserts both itself, but a violation should be refused here,
    before a launch, rather than surfacing as a compile error -- and the sweep and
    audit tools drive this path, so it has to hold for an overridden tile too.
    """
    block_m, block_n, block_k, _group_m, num_warps, _stages = tile
    if not (block_k == _SCALE_GROUP
            and block_n > 0 and block_n % _SCALE_GROUP == 0
            and 16 <= block_m <= align_m and align_m % block_m == 0
            and N % block_n == 0
            and K % _SCALE_GROUP == 0):
        return False
    # The fp32 accumulator is register-resident even on the tcgen05 path, because the
    # per-group scale promotion reads each MMA result back out of tensor memory.  Its
    # area per thread is block_m * block_n / (32 * num_warps) registers, and the file
    # holds 255.  This is not a tuning preference: at 128x256 over 4 warps -- 256
    # registers for the accumulator alone -- ptxas refuses the tile outright at
    # BLOCK_K = 128 (error C7600) and, at BLOCK_K = 64, compiles it and returns
    # *wrong numbers* (measured: 0.85% of elements out of tolerance, max abs error
    # 848, against a bit-exact result for the same tile at 8 warps).  So a tile whose
    # accumulator cannot plausibly fit is refused here rather than trusted to fail
    # loudly.  The bound is the largest area any measured-correct configuration uses.
    return block_m * block_n <= _MAX_ACCUMULATOR_REGISTERS * 32 * num_warps


def _clear_dispatch_caches() -> None:
    """Re-read the experiment-only knobs above.

    Both tile choosers are memoized, which is what keeps those knobs off the timed
    path; the price is that a tool changing one has to say so.
    """
    _fast_tile.cache_clear()
    _thin_decode_tile.cache_clear()


@functools.lru_cache(maxsize=512)
def _fast_tile(rows: int, top_k: int, N: int, K: int, num_experts: int,
               align_m: int) -> tuple[int, int, int, int, int, int] | None:
    """``(BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages)`` for the
    blockwise-FP8 path, or ``None`` when no admissible tile exists.

    A fixed table over host-visible sizes.  There is deliberately no autotuner: a
    shape first seen inside a timed region would otherwise compile there.

    Every value below comes from a measured sweep on B200, not from an estimate, and
    the profiles under ``profile/`` say why.  The load-bearing fact, established by
    reading SASS out of the reports rather than by reasoning about the source, is
    that Triton 3.6 lowers this ``tl.dot`` two different ways depending on the M
    tile: at ``BLOCK_M >= 64`` it emits ``UTCQMMA`` with a TMEM result and ``LDTM``
    to read it back, and at 16 or 32 it emits classic ``HMMA.16816.F32``.  So:

    * ``BLOCK_M = 64`` where the alignment allows it.  It is the smallest tile that
      still reaches the tensor-memory MMA path, so it gets that path *and* the least
      padding waste -- and the padding is what dominates here, since with 128 experts
      an alignment block often holds only ~21 valid rows out of 128.
    * ``BLOCK_M = 16`` below that.  Small alignment means a decode batch, whose kernel
      is a single sub-wave of a few hundred programs bound by memory latency rather
      than by MMA throughput; giving up the tensor-memory path there measured faster
      (14.1 us against 14.8 us at 32, reference 15.5 us).
    * ``BLOCK_N = 128``.  256 lost on every shape where the two differ, by 14% to 7x.
      Because the per-group scale promotion has to pull each MMA result out of TMEM,
      the *running* accumulator lives in registers, so doubling the tile's N extent
      doubles register-resident state: the reference kernel sits on the 255-register
      ceiling and issues 10-84 M local (spill) load/store instructions, while the
      64x128 geometry issues exactly zero.  At 4 warps a 128x256 tile does not even
      allocate -- ptxas reports C7600.
    * ``BLOCK_K = 128`` equals the scale group, so each accumulator element is
      promoted once per 128 of K rather than twice.
    * The grouped swizzle pays only once M is large enough for weight reuse across M
      tiles to matter: about 10% at 131072 token-expert pairs, nothing measurable at
      8000.  Those are the only two regimes the benchmarked shapes sample, so the
      threshold between them is a boundary, not a measured optimum.
    """
    if _FORCE_REFERENCE:
        return None
    if _TILE_OVERRIDE is not None:
        return _TILE_OVERRIDE if _tile_is_admissible(_TILE_OVERRIDE, N, K, align_m) else None

    # The single scale promotion needs the K tile to be exactly one scale group.
    if K % _SCALE_GROUP:
        return None
    # The weight scale is only a per-slice scalar while the N tile is a whole
    # number of scale groups, and dropping the store mask needs N to tile evenly.
    block_n = _SCALE_GROUP
    if N % block_n:
        return None

    # 64 where the alignment allows it; 16 below that.  The small-alignment case is
    # the decode batch, whose kernel is a single sub-wave bound by memory latency
    # rather than by MMA throughput, and there the narrower tile measured faster
    # (14.1 us against 14.8 us at BLOCK_M=32, baseline 15.5 us).
    block_m = 64 if align_m >= 64 else min(16, align_m)
    if block_m < 16 or align_m % block_m:
        return None

    group_m = 8 if rows * top_k >= 1 << 15 else 1
    tile = (block_m, block_n, _SCALE_GROUP, group_m, 4, 4)
    # The table is fixed, so this cannot fail -- but it is the same predicate the
    # override path goes through, so nothing reaches the kernel unchecked.
    return tile if _tile_is_admissible(tile, N, K, align_m) else None


# Decode regime: few enough token-expert pairs that the kernel cannot fill the
# device, so more N-parallelism is worth more than a wider tile.  A threshold on the
# pair count rather than on the exact benchmarked shape -- keying the dispatch to
# K = 384 and N = 4096 would be fitting the shipped code to the benchmark, and the
# regime is what the specialization is actually about.
_THIN_DECODE_MAX_PAIRS = 64

# Set by the measurement in tools/thin_decode_experiment.py.  See that file and
# docs/results.md: the specialization is kept only if it wins on its own regime.
_THIN_DECODE_ENABLED = True
_THIN_DECODE_BLOCK_N = 64
_THIN_DECODE_BLOCK_M = 16


@functools.lru_cache(maxsize=64)
def _thin_decode_tile(rows: int, top_k: int, N: int, K: int,
                      align_m: int) -> tuple[int, int, int, int, int] | None:
    """``(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)`` or ``None``."""
    if not _THIN_DECODE_ENABLED or _FORCE_REFERENCE or _TILE_OVERRIDE is not None:
        return None
    if rows * top_k > _THIN_DECODE_MAX_PAIRS or K % _SCALE_GROUP:
        return None
    block_m = min(_THIN_DECODE_BLOCK_M, align_m)
    block_n = _THIN_DECODE_BLOCK_N
    if block_m < 16 or align_m % block_m or N % block_n or _SCALE_GROUP % block_n:
        return None
    return block_m, block_n, _SCALE_GROUP, 4, 4


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
            # The same guess the reference makes, so the alignment this kernel
            # reads is the one the caller built the metadata with.
            config = _get_default_config(A.size(0), N=B.size(1))
        # config is read, never written: both L2 callers hand the *same* dict to
        # GEMM1 and GEMM2 after feeding BLOCK_SIZE_M to MoeAlign, so mutating it
        # would desynchronize the second launch from the routing metadata.
        align_m = config["BLOCK_SIZE_M"]

        if use_fp8_w8a8:
            compute_type = tl.bfloat16
        elif A.dtype == torch.bfloat16:
            compute_type = tl.bfloat16
        elif A.dtype == torch.float16:
            compute_type = tl.float16
        else:
            compute_type = tl.float32

        # One .shape unpack rather than a .size() call per dimension: the decode
        # shape is bound by host time, so the count of Python-level calls in here
        # is itself a performance parameter.
        num_experts, N_out, K = B.shape
        rows = A.size(0)
        num_valid_tokens = rows * top_k

        if (sorted_token_ids is not None
                and use_fp8_w8a8
                and block_shape is not None
                and block_shape[0] == _SCALE_GROUP
                and block_shape[1] == _SCALE_GROUP
                and a_scale is not None and a_scale.ndim == 2
                and b_scale is not None and b_scale.ndim == 3):
            thin = _thin_decode_tile(rows, top_k, N_out, K, align_m)
            tile = _fast_tile(rows, top_k, N_out, K, num_experts, align_m) if thin is None \
                else None
            if thin is not None or tile is not None:
                EM = sorted_token_ids.size(0)
                if rows < align_m:
                    EM = min(EM, num_valid_tokens * align_m)
                bs_experts, bs_groups, _ = b_scale.shape
                if self._launch_blockwise_fp8(
                        A, B, C, a_scale, b_scale, topk_weights, sorted_token_ids,
                        expert_ids, num_tokens_post_padded, N_out, K, EM,
                        num_valid_tokens, mul_routed_weight, top_k, compute_type,
                        align_m, tile, num_experts, C.size(0), bs_experts, bs_groups,
                        thin):
                    return None

        self._launch_reference(
            A, B, C, a_scale, b_scale, topk_weights, sorted_token_ids, expert_ids,
            num_tokens_post_padded, N_out, K, rows, num_valid_tokens,
            mul_routed_weight, top_k, compute_type, config, use_fp8_w8a8, block_shape)
        return None

    @staticmethod
    def _launch_blockwise_fp8(A, B, C, a_scale, b_scale, topk_weights,
                              sorted_token_ids, expert_ids, num_tokens_post_padded,
                              N, K, EM, num_valid_tokens, mul_routed_weight, top_k,
                              compute_type, align_m, tile, num_experts, c_rows,
                              bs_experts, bs_groups, thin=None) -> bool:
        """Launch the fast path, or return False so the caller falls back.

        Everything this checks is a *precondition of the kernel*, not a
        performance preference: unit innermost strides (the kernel hard-codes
        them to keep its argument list short) and 32-bit element indices.
        """
        if thin is not None:
            block_m, block_n, block_k, num_warps, num_stages = thin
            group_m = 1
        else:
            block_m, block_n, block_k, group_m, num_warps, num_stages = tile
        stride_am, stride_ak = A.stride()
        stride_be, stride_bn, stride_bk = B.stride()
        stride_cm, stride_cn = C.stride()
        stride_asm, stride_ask = a_scale.stride()
        stride_bse, stride_bsn, stride_bsk = b_scale.stride()
        if stride_ak != 1 or stride_bk != 1 or stride_cn != 1 or stride_ask != 1 \
                or stride_bsk != 1:
            return False

        # 32-bit element indices need every strided tensor's largest index -- plus
        # one tile stride of slack for the advance the K loop makes past its final
        # load -- to fit.  Anything larger falls through rather than wrapping.
        rows = A.size(0)
        widest = max((rows - 1) * stride_am,
                     (num_experts - 1) * stride_be + (N - 1) * stride_bn,
                     (c_rows - 1) * stride_cm + N - 1,
                     (rows - 1) * stride_asm,
                     (bs_experts - 1) * stride_bse + (bs_groups - 1) * stride_bsn)
        if widest + K > _INT32_LIMIT:
            return False

        grid_m = -(-EM // block_m)
        kernel = _moe_thin_decode_kernel if thin is not None else _moe_blockwise_fp8_kernel
        kernel[(grid_m * (N // block_n),)](
            A, B, C,
            a_scale, b_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            K, grid_m, N // block_n,
            num_valid_tokens,
            stride_am,
            stride_be, stride_bn,
            stride_cm,
            stride_asm,
            stride_bse, stride_bsn,
            _SCALE_GROUP,
            mul_routed_weight,
            top_k,
            compute_type,
            *((block_m, block_n, block_k, align_m) if thin is not None else
              (block_m, block_n, block_k, group_m, align_m,
               block_n // _SCALE_GROUP, K % block_k == 0)),
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return True

    @staticmethod
    def _launch_reference(A, B, C, a_scale, b_scale, topk_weights, sorted_token_ids,
                          expert_ids, num_tokens_post_padded, N, K, rows,
                          num_valid_tokens, mul_routed_weight, top_k, compute_type,
                          config, use_fp8_w8a8, block_shape) -> None:
        block_m = config["BLOCK_SIZE_M"]
        block_n = config["BLOCK_SIZE_N"]
        block_k = config["BLOCK_SIZE_K"]
        naive = sorted_token_ids is None
        if naive:
            EM = expert_ids.numel() * block_m
        else:
            EM = sorted_token_ids.size(0)
            if rows < block_m:
                EM = min(EM, num_valid_tokens * block_m)

        if use_fp8_w8a8 and block_shape is not None:
            group_n, group_k = block_shape[0], block_shape[1]
            block_k = min(block_k, min(group_n, group_k))
        else:
            group_n, group_k = 0, 0

        launch_kwargs = {}
        if "num_warps" in config:
            launch_kwargs["num_warps"] = config["num_warps"]
        if "num_stages" in config:
            launch_kwargs["num_stages"] = config["num_stages"]

        _moe_reference_kernel[(triton.cdiv(EM, block_m) * triton.cdiv(N, block_n),)](
            A, B, C,
            a_scale if a_scale is not None else A,
            b_scale if b_scale is not None else B,
            topk_weights,
            sorted_token_ids if sorted_token_ids is not None else A,
            expert_ids,
            num_tokens_post_padded,
            N, K, EM,
            num_valid_tokens,
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
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            GROUP_SIZE_M=config["GROUP_SIZE_M"],
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            NAIVE_BLOCK_ASSIGNMENT=naive,
            **launch_kwargs,
        )
