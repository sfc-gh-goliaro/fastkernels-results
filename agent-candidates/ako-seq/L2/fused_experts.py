"""Fused MoE experts: two grouped GEMMs with SiLU-mul in between.

Supports both BF16 and FP8 W8A8 block-scaled expert weights.

The FP8 W8A8 + SiLU path (every captured scenario) runs a purpose-built pair of
Triton kernels instead of the generic ``MoeGroupedGemm`` sequence:

* ``_gemm1_act_quant_kernel`` (and ``_gemm1_tma_kernel``, the same kernel with
  the weight tiles fetched through a TMA descriptor) computes *both* halves of
  the gate/up projection for one 128-wide slice of the intermediate dimension,
  applies SiLU-mul and the per-128-group FP8 quantization in its epilogue, and
  writes only the FP8 activation plus its scale.  The generic path materializes
  ``[M*top_k, 2N]`` bf16, reads it back for the activation, writes
  ``[M*top_k, N]`` bf16, and reads *that* back for the quantizer -- 4.6 KB of
  HBM traffic per routed token that never leaves registers here.
* ``_gemm2_kernel`` is the down projection, each program walking a run of N
  tiles (K is only 384, so one output tile per program is pure load latency)
  with the A tiles and their scales held across that loop.

Two things decide how fast either kernel can run, and both cost nothing in
arithmetic:

1. **Keep every ``tl.dot`` result live across the block-scale promote.**  The
   promote ``acc += dot(a, b) * (a_scale * b_scale)`` must happen once per
   128-element K group because the weight scales are arbitrary fp32 that vary
   with k.  Written as one statement, Triton gives the dot a single tensor-memory
   buffer, so the next MMA cannot issue until that buffer has been read back and
   scaled -- the promote sits on the critical path.  Naming the results first
   (``tg``/``tu`` here, ``t_0..t_2`` in GEMM2) forces one TMEM buffer each, and
   the MMAs pipeline behind the promotes: GEMM1 819 -> 717 us and GEMM2
   423 -> 409 us at M=16384, and a win at every shape.
2. **Operand bytes per flop.**  Both kernels are limited by the L2 read rate
   (~11 TB/s here), not by the MMA, so the tile shape is chosen to minimize
   ``(BLOCK_M + BLOCK_N_total) / (2 * BLOCK_M * BLOCK_N_total)``: GEMM1 shares
   one A tile between the gate and up halves, and GEMM2 uses a 128-row tile,
   which is why ``MoeAlign`` is asked for a 128-row alignment.  ``BLOCK_M=128``
   for GEMM1 would halve its traffic again but does not fit -- the fp32
   accumulators plus the tensor-memory read temporaries exceed 255 registers,
   and ptxas spills.

Both kernels are driven directly by ``MoeAlign``'s routing metadata, so the
compute tile is decoupled from the alignment granularity in the same way the L1
grouped GEMM does it, and both take part in programmatic dependent launch, so
each one's blocks are resident before its producer's tail has drained.
Everything else (routing metadata, activation quantization, top-k reduction)
reuses the L1 winners.

When DeepGEMM is available *and* the shapes suit it, the DeepGEMM contiguous
grouped-GEMM path is used exactly as before -- that branch consumes the
``*_scale_dg`` scales, so it has to be selected on the same condition as the
baseline.  (On sm100 it is also numerically unusable for the non-``_dg`` scales:
only the UE8M0 kernel exists there, so it would round every weight scale to a
power of two.)  Anything else falls back to the generic Triton path, which is
what the M=1 shape uses -- eight routed rows do not fill a single 128-row expert
block, let alone the GPU.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

from ..L1.fp8_linear import PerTokenGroupQuantFp8
from ..L1.moe_align import MoeAlign
from ..L1.moe_grouped_gemm import (
    MoeGroupedGemm,
    _valid_deep_gemm,
    get_triton_config,
    m_grouped_fp8_gemm_nt_contiguous,
)
from ..L1.moe_sum import MoeSum
from ..L1.gelu_and_mul import GeluAndMul
from ..L1.silu_and_mul import SiluAndMul
from ..L1.silu_mul_quant_fp8 import SiluMulQuantFp8

SPARSITY_FACTOR = 4
_FP8_GROUP_SIZE = 128
_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)


def _iv(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


# Tunables.  Env overrides exist only so the sweep script can drive them; the
# defaults below are what the shipped kernel uses.
_ALIGN_BIG = _iv("FK_L2_ALIGNB", 128)  # row granularity handed to MoeAlign
_ALIGN_ROWS = _iv("FK_L2_ALIGNR", 0)   # rows at which to use it; 0 = always
_ALIGN = _iv("FK_L2_ALIGN", 64)      # ... below that (unused at the default)
_BM1 = _iv("FK_L2_BM1", 64)          # GEMM1 compute tile rows
_BM2_BIG = _iv("FK_L2_BM2B", 128)    # GEMM2 compute tile rows
_BM2 = _iv("FK_L2_BM2", 64)          # ... at the smaller alignment
_BN2 = _iv("FK_L2_BN2", 64)          # GEMM2 output columns per program-iteration
_W2 = _iv("FK_L2_W2", 4)
_S2 = _iv("FK_L2_S2", 2)
_TARGET_BLOCKS = _iv("FK_L2_TB", 8)  # GEMM2 blocks per SM to aim the grid at
_WAVE_CUTOFF = _iv("FK_L2_WC", 4)    # GEMM1 pipeline-shape breakpoint, in waves
_TMA_WAVES = _iv("FK_L2_TW", 4)      # wave count at which TMA descriptors pay


_SMS: dict = {}

# Triton's device-side TMA descriptors need a global scratch allocation.  The
# allocator is called once per launch; keeping the largest buffer seen avoids a
# fresh allocation (and any allocator work under CUDA-graph capture) after the
# first call for a given grid size.
_TMA_SCRATCH: dict = {}
_TMA_READY = [False]


def _tma_alloc(size: int, alignment: int, stream):
    idx = torch.cuda.current_device()
    buf = _TMA_SCRATCH.get(idx)
    if buf is None or buf.numel() < size:
        # Generous: the request scales with the grid, and growing it later
        # would mean allocating inside a CUDA-graph capture.
        buf = torch.empty(max(size, 1 << 23), dtype=torch.int8, device=f"cuda:{idx}")
        _TMA_SCRATCH[idx] = buf
    return buf


def _tma_available() -> bool:
    if not _TMA_READY[0]:
        if not hasattr(tl, "make_tensor_descriptor"):
            return False
        triton.set_allocator(_tma_alloc)
        _TMA_READY[0] = True
    return True


def _num_sms(device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    n = _SMS.get(idx)
    if n is None:
        n = torch.cuda.get_device_properties(idx).multi_processor_count
        _SMS[idx] = n
    return n


def _compute_aligned_M(M: int, num_topk: int, local_num_experts: int,
                        alignment: int) -> int:
    """Compute aligned total rows for DeepGEMM."""
    M_sum = (M * num_topk) + local_num_experts * (alignment - 1)
    remainder = M_sum % alignment
    if remainder != 0:
        M_sum += alignment - remainder
    return M_sum


def _deepgemm_permute(
    hidden_states: torch.Tensor,
    a_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    local_num_experts: int,
    alignment: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Permute tokens by expert assignment for DeepGEMM contiguous layout."""
    M, K = hidden_states.size()
    top_k = topk_ids.size(1)
    device = hidden_states.device

    M_sum = _compute_aligned_M(M, top_k, local_num_experts, alignment)
    scale_cols = K // _FP8_GROUP_SIZE

    flat_ids = topk_ids.view(-1).to(torch.int64)
    num_tokens_total = flat_ids.size(0)

    expert_num_tokens = torch.zeros(local_num_experts, dtype=torch.int64, device=device)
    expert_num_tokens.scatter_add_(0, flat_ids,
                                   torch.ones(num_tokens_total, dtype=torch.int64, device=device))

    aligned_counts = ((expert_num_tokens + alignment - 1) // alignment) * alignment
    expert_offsets = torch.zeros(local_num_experts + 1, dtype=torch.int64, device=device)
    torch.cumsum(aligned_counts, dim=0, out=expert_offsets[1:])

    pos_idx = torch.arange(M_sum, device=device, dtype=torch.int64)
    expert_for_pos = torch.searchsorted(expert_offsets, pos_idx, right=True) - 1
    expert_for_pos = expert_for_pos.clamp_(0, local_num_experts - 1)
    local_pos = pos_idx - expert_offsets[expert_for_pos]
    valid = local_pos < expert_num_tokens[expert_for_pos]
    expert_ids = torch.where(valid, expert_for_pos.to(torch.int32),
                             torch.tensor(-1, dtype=torch.int32, device=device))

    sorted_order = torch.argsort(flat_ids, stable=True)

    sorted_experts = flat_ids[sorted_order]
    rank_in_sorted = torch.arange(num_tokens_total, device=device, dtype=torch.int64)
    expert_first = torch.full((local_num_experts,), num_tokens_total,
                              dtype=torch.int64, device=device)
    expert_first.scatter_reduce_(0, sorted_experts,
                                 rank_in_sorted, reduce="amin",
                                 include_self=False)
    within_expert_idx = torch.zeros(num_tokens_total, dtype=torch.int64, device=device)
    within_expert_idx[sorted_order] = rank_in_sorted - expert_first[sorted_experts]

    dest_positions = expert_offsets[flat_ids] + within_expert_idx

    a_perm = torch.zeros(M_sum, K, dtype=hidden_states.dtype, device=device)
    a_scale_perm = torch.zeros(M_sum, scale_cols, dtype=torch.float32, device=device)

    token_indices = torch.arange(M, device=device).unsqueeze(1).expand(M, top_k).reshape(-1)

    a_perm[dest_positions] = hidden_states[token_indices]
    a_scale_perm[dest_positions] = a_scale[token_indices]

    inv_perm = dest_positions.view(M, top_k).to(torch.int32)

    return a_perm, a_scale_perm, expert_ids, inv_perm


def _deepgemm_unpermute_and_reduce(
    mm2_out: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    inv_perm: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Unpermute DeepGEMM output and reduce across top-k experts."""
    M, K = output.size()
    top_k = topk_ids.size(1)

    flat_positions = inv_perm.to(torch.int64).view(-1)
    gathered = mm2_out[flat_positions].view(M, top_k, K)
    weights = topk_weights.unsqueeze(-1)
    output.copy_((gathered.to(output.dtype) * weights).sum(dim=1))


# ---------------------------------------------------------------------------
# Fused FP8 W8A8 path
# ---------------------------------------------------------------------------
@triton.jit
def _gemm1_act_quant_kernel(
    a_ptr, as_ptr, b_ptr, bs_ptr, o_ptr, os_ptr,
    sorted_ids_ptr, expert_ids_ptr, ntpp_ptr,
    num_valid_tokens,
    stride_am, stride_ak,
    stride_asm, stride_ask,
    stride_be, stride_bn, stride_bk,
    stride_bse, stride_bsn, stride_bsk,
    stride_om,
    stride_osm, stride_osg,
    NG: tl.constexpr,
    N_HALF: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_K: tl.constexpr,
    EXPERT_BLOCK_M: tl.constexpr,
    SKIP_EMPTY: tl.constexpr,
    K_UNROLL: tl.constexpr,
    INT64: tl.constexpr,
    FP8_MAX: tl.constexpr,
    PDL: tl.constexpr,
):
    """Gate/up projection + SiLU-mul + per-128-group FP8 quant, fused.

    One program owns ``BLOCK_M`` routed rows and the ``pid_n``-th 128-column
    group of the intermediate dimension.  ``BLOCK_N == 128`` is deliberate: it is
    both the weight-scale group width (so each expert scale is a scalar per K
    step) and the activation-quant group width (so the epilogue's row absmax is
    over exactly the group the reference quantizer uses).  The gate and up halves
    of ``w13`` share the A tile, so carrying two accumulators halves the A
    traffic *and* doubles the MMA per block relative to one 128-column tile.
    """
    pid = tl.program_id(0)
    # N-groups vary fastest so the three programs sharing an A tile are adjacent.
    pid_n = pid % NG
    pid_m = pid // NG

    if PDL:
        gdc_wait()
    if pid_m * BLOCK_M >= tl.load(ntpp_ptr):
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_token = tl.load(sorted_ids_ptr + pid_m * BLOCK_M + offs_m)
    token_mask = offs_token < num_valid_tokens
    if SKIP_EMPTY:
        if tl.min(offs_token) >= num_valid_tokens:
            return
    # Padded rows are masked out of every load and the store; folding them onto
    # row 0 keeps the address math inside the tensors (and, on the 32-bit path,
    # keeps the padded-row sentinel from overflowing a row offset).
    offs_token = tl.where(token_mask, offs_token, 0)
    if INT64:
        offs_token = offs_token.to(tl.int64)

    if EXPERT_BLOCK_M == BLOCK_M:
        off_e = tl.load(expert_ids_ptr + pid_m)
    else:
        off_e = tl.load(expert_ids_ptr + pid_m // (EXPERT_BLOCK_M // BLOCK_M))

    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_row = offs_token // TOP_K

    a_ptrs = a_ptr + offs_row[:, None] * stride_am + offs_k[None, :] * stride_ak
    bg_ptrs = (b_ptr + off_e * stride_be + offs_k[:, None] * stride_bk
               + offs_bn[None, :] * stride_bn)
    bu_ptrs = bg_ptrs + N_HALF * stride_bn

    as_ptrs = as_ptr + offs_row * stride_asm
    bsg_ptr = bs_ptr + off_e * stride_bse + pid_n * stride_bsn
    bsu_ptr = bsg_ptr + NG * stride_bsn

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, NUM_K, disallow_acc_multi_buffer=True,
                      loop_unroll_factor=K_UNROLL):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        bg = tl.load(bg_ptrs)
        bu = tl.load(bu_ptrs)
        a_s = tl.load(as_ptrs + k * stride_ask, mask=token_mask, other=0.0)
        sg = tl.load(bsg_ptr + k * stride_bsk)
        su = tl.load(bsu_ptr + k * stride_bsk)
        # Keeping *both* dot results live across the promote is what makes the
        # promote overlap the MMA.  Written as ``acc_g += dot(...) * s``, Triton
        # allocates one TMEM buffer for the dot result, so the second MMA cannot
        # issue until the first result has been read back out of tensor memory
        # and scaled -- the promote lands on the critical path.  With ``tg`` and
        # ``tu`` both live it must allocate two, and the up-half MMA runs while
        # the gate half is being promoted.  Measured GEMM1 at M=16384:
        # 819 -> 723 us, and a win at every shape.  (The flag that is supposed
        # to do this, ``disallow_acc_multi_buffer=False``, is broken: wrong
        # results and 3-7x slower.)
        tg = tl.dot(a, bg)
        tu = tl.dot(a, bu)
        acc_g += tg * (a_s * sg)[:, None]
        acc_u += tu * (a_s * su)[:, None]
        a_ptrs += BLOCK_K * stride_ak
        bg_ptrs += BLOCK_K * stride_bk
        bu_ptrs += BLOCK_K * stride_bk

    # Reference chain: the GEMM output is rounded to bf16, SiLU is evaluated in
    # fp32 and rounded back to bf16, the product is a bf16 multiply, and the
    # quantizer reads that bf16 value.  Reproduce it step for step.
    gate = acc_g.to(tl.bfloat16).to(tl.float32)
    up = acc_u.to(tl.bfloat16).to(tl.float32)
    silu = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    h = (silu * up).to(tl.bfloat16).to(tl.float32)

    absmax = tl.maximum(tl.max(tl.abs(h), axis=1), 1e-10)
    scale = tl.math.exp2(tl.math.ceil(tl.math.log2(absmax * (1.0 / FP8_MAX))))
    q = tl.clamp(h / scale[:, None], -FP8_MAX, FP8_MAX)

    tl.store(o_ptr + offs_token[:, None] * stride_om + offs_bn[None, :],
             q.to(o_ptr.dtype.element_ty), mask=token_mask[:, None])
    tl.store(os_ptr + offs_token * stride_osm + pid_n * stride_osg, scale,
             mask=token_mask)
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _gemm1_tma_kernel(
    a_ptr, as_ptr, b_ptr, bs_ptr, o_ptr, os_ptr,
    sorted_ids_ptr, expert_ids_ptr, ntpp_ptr,
    num_valid_tokens,
    b_rows, b_cols,
    stride_am, stride_ak,
    stride_asm, stride_ask,
    stride_bse, stride_bsn, stride_bsk,
    stride_om,
    stride_osm, stride_osg,
    NG: tl.constexpr,
    N_HALF: tl.constexpr,
    N2: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_K: tl.constexpr,
    EXPERT_BLOCK_M: tl.constexpr,
    SKIP_EMPTY: tl.constexpr,
    K_UNROLL: tl.constexpr,
    INT64: tl.constexpr,
    FP8_MAX: tl.constexpr,
    PDL: tl.constexpr,
):
    """``_gemm1_act_quant_kernel`` with the two weight tiles fetched by TMA.

    Identical arithmetic (still bit-exact against the reference); the only change
    is how ``w13`` reaches the MMA.  The pointer form materializes a
    ``[BLOCK_K, BLOCK_N]`` tensor of pointers per weight half and increments both
    every K step; the descriptor form needs one scalar coordinate pair, and the
    copy issues as a single bulk transfer.  Both forms sit at ptxas's 255-register
    ceiling, but the pointer one spills 2 registers there and this one spills
    none.  ``w13`` is viewed as ``[E * N2, K]`` so a plain 2-D descriptor
    covers every (expert, N-group) tile -- gate at row ``e*N2 + n0`` and up
    ``N_HALF`` rows below it.
    """
    pid = tl.program_id(0)
    pid_n = pid % NG
    pid_m = pid // NG

    if PDL:
        gdc_wait()
    if pid_m * BLOCK_M >= tl.load(ntpp_ptr):
        return

    bdesc = tl.make_tensor_descriptor(
        b_ptr, shape=[b_rows, b_cols], strides=[b_cols, 1],
        block_shape=[BLOCK_N, BLOCK_K],
    )

    offs_m = tl.arange(0, BLOCK_M)
    offs_token = tl.load(sorted_ids_ptr + pid_m * BLOCK_M + offs_m)
    token_mask = offs_token < num_valid_tokens
    if SKIP_EMPTY:
        if tl.min(offs_token) >= num_valid_tokens:
            return
    offs_token = tl.where(token_mask, offs_token, 0)
    if INT64:
        offs_token = offs_token.to(tl.int64)

    if EXPERT_BLOCK_M == BLOCK_M:
        off_e = tl.load(expert_ids_ptr + pid_m)
    else:
        off_e = tl.load(expert_ids_ptr + pid_m // (EXPERT_BLOCK_M // BLOCK_M))

    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_row = offs_token // TOP_K

    a_ptrs = a_ptr + offs_row[:, None] * stride_am + offs_k[None, :] * stride_ak
    row_g = off_e * N2 + pid_n * BLOCK_N
    row_u = row_g + N_HALF

    as_ptrs = as_ptr + offs_row * stride_asm
    bsg_ptr = bs_ptr + off_e * stride_bse + pid_n * stride_bsn
    bsu_ptr = bsg_ptr + NG * stride_bsn

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, NUM_K, disallow_acc_multi_buffer=True,
                      loop_unroll_factor=K_UNROLL):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        bg = bdesc.load([row_g, k * BLOCK_K])
        bu = bdesc.load([row_u, k * BLOCK_K])
        a_s = tl.load(as_ptrs + k * stride_ask, mask=token_mask, other=0.0)
        sg = tl.load(bsg_ptr + k * stride_bsk)
        su = tl.load(bsu_ptr + k * stride_bsk)
        # Keeping *both* dot results live across the promote is what makes the
        # promote overlap the MMA.  Written as ``acc_g += dot(...) * s``, Triton
        # allocates one TMEM buffer for the dot result, so the second MMA cannot
        # issue until the first result has been read back out of tensor memory
        # and scaled -- the promote lands on the critical path.  With ``tg`` and
        # ``tu`` both live it must allocate two, and the up-half MMA runs while
        # the gate half is being promoted.  Measured GEMM1 at M=16384:
        # 819 -> 723 us, and a win at every shape.  (The flag that is supposed
        # to do this, ``disallow_acc_multi_buffer=False``, is broken: wrong
        # results and 3-7x slower.)
        tg = tl.dot(a, tl.trans(bg))
        tu = tl.dot(a, tl.trans(bu))
        acc_g += tg * (a_s * sg)[:, None]
        acc_u += tu * (a_s * su)[:, None]
        a_ptrs += BLOCK_K * stride_ak

    gate = acc_g.to(tl.bfloat16).to(tl.float32)
    up = acc_u.to(tl.bfloat16).to(tl.float32)
    silu = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    h = (silu * up).to(tl.bfloat16).to(tl.float32)

    absmax = tl.maximum(tl.max(tl.abs(h), axis=1), 1e-10)
    scale = tl.math.exp2(tl.math.ceil(tl.math.log2(absmax * (1.0 / FP8_MAX))))
    q = tl.clamp(h / scale[:, None], -FP8_MAX, FP8_MAX)

    tl.store(o_ptr + offs_token[:, None] * stride_om + offs_bn[None, :],
             q.to(o_ptr.dtype.element_ty), mask=token_mask[:, None])
    tl.store(os_ptr + offs_token * stride_osm + pid_n * stride_osg, scale,
             mask=token_mask)
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _gemm2_kernel(
    a_ptr, as_ptr, b_ptr, bs_ptr, c_ptr, tw_ptr,
    sorted_ids_ptr, expert_ids_ptr, ntpp_ptr,
    num_valid_tokens,
    stride_am, stride_ak,
    stride_asm, stride_ask,
    stride_be, stride_bn, stride_bk,
    stride_bse, stride_bsn, stride_bsk,
    stride_cm,
    NUM_M_BLOCKS,
    N_PER_CHUNK: tl.constexpr,
    GROUP_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_K: tl.constexpr,
    EXPERT_BLOCK_M: tl.constexpr,
    SKIP_EMPTY: tl.constexpr,
    INT64: tl.constexpr,
    SCALAR_B_SCALE: tl.constexpr,
    HOIST_A: tl.constexpr,
    PDL: tl.constexpr,
):
    """Down projection with the routed weight folded into the epilogue.

    K is only 384 here (three scale groups), so a program that owns a single
    output tile has three MMA steps to hide a global-load round trip behind and
    loses badly to latency.  Instead each program owns ``N_PER_CHUNK``
    consecutive output tiles of one row block and walks them in a pipelined
    loop: the routing metadata and A tiles are loaded once, and tile *i*'s store
    overlaps tile *i+1*'s weight loads.  The host picks ``N_PER_CHUNK`` so the
    grid still fills the GPU -- one chunk covering all of N for the prefill
    shapes, down to one tile per chunk when there are only a handful of row
    blocks.

    Row blocks vary fastest across the grid so the programs sharing an N range
    (and therefore a ``w2`` slice) run together and hit in L2.
    """
    pid = tl.program_id(0)
    pid_m = pid % NUM_M_BLOCKS
    pid_c = pid // NUM_M_BLOCKS

    if PDL:
        gdc_wait()
    if pid_m * BLOCK_M >= tl.load(ntpp_ptr):
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_token = tl.load(sorted_ids_ptr + pid_m * BLOCK_M + offs_m)
    token_mask = offs_token < num_valid_tokens
    if SKIP_EMPTY:
        if tl.min(offs_token) >= num_valid_tokens:
            return
    offs_token = tl.where(token_mask, offs_token, 0)
    if INT64:
        offs_token = offs_token.to(tl.int64)

    if EXPERT_BLOCK_M == BLOCK_M:
        off_e = tl.load(expert_ids_ptr + pid_m)
    else:
        off_e = tl.load(expert_ids_ptr + pid_m // (EXPERT_BLOCK_M // BLOCK_M))

    offs_k = tl.arange(0, BLOCK_K)
    offs_bn = tl.arange(0, BLOCK_N)
    a_base = a_ptr + offs_token[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_base = b_ptr + off_e * stride_be + offs_k[:, None] * stride_bk
    as_base = as_ptr + offs_token * stride_asm
    bs_base = bs_ptr + off_e * stride_bse
    w = tl.load(tw_ptr + offs_token, mask=token_mask, other=0.0)
    c_base = c_ptr + offs_token[:, None] * stride_cm

    n0 = pid_c * N_PER_CHUNK * BLOCK_N
    if HOIST_A:
        # The A tile is invariant across the N loop, but the ``tl.store`` to
        # ``c_ptr`` inside that loop stops the compiler from proving it does not
        # alias ``a_ptr``, so it reloads all NUM_K tiles for every N tile: at
        # ``N_PER_CHUNK`` covering all of N that is 32-64 re-reads of 24 KB per
        # program -- more traffic than this expert's whole ``w2`` slice.  An
        # ``[BLOCK_M, BLOCK_K]`` FP8 tile is only 16 registers per thread, so
        # holding all three K groups (and their scales) across the loop is free.
        a_0 = tl.load(a_base, mask=token_mask[:, None], other=0.0)
        a_1 = tl.load(a_base + BLOCK_K * stride_ak,
                      mask=token_mask[:, None], other=0.0)
        a_2 = tl.load(a_base + 2 * BLOCK_K * stride_ak,
                      mask=token_mask[:, None], other=0.0)
        as_0 = tl.load(as_base, mask=token_mask, other=0.0)
        as_1 = tl.load(as_base + stride_ask, mask=token_mask, other=0.0)
        as_2 = tl.load(as_base + 2 * stride_ask, mask=token_mask, other=0.0)
        for nb in tl.range(0, N_PER_CHUNK, disallow_acc_multi_buffer=True):
            n_off = n0 + nb * BLOCK_N
            bn = (n_off + offs_bn)[None, :] * stride_bn
            bsp = bs_base + (n_off // GROUP_N) * stride_bsn
            # Same reason as GEMM1's ``tg``/``tu``: three live dot results make
            # Triton give each K group its own TMEM buffer, so the three MMAs
            # pipeline instead of each waiting for the previous promote to read
            # tensor memory.  426 -> 410 us at M=16384, 46 -> 43 at M=1000.
            t_0 = tl.dot(a_0, tl.load(b_base + bn))
            t_1 = tl.dot(a_1, tl.load(b_base + bn + BLOCK_K * stride_bk))
            t_2 = tl.dot(a_2, tl.load(b_base + bn + 2 * BLOCK_K * stride_bk))
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            acc += t_0 * (as_0 * tl.load(bsp))[:, None]
            acc += t_1 * (as_1 * tl.load(bsp + stride_bsk))[:, None]
            acc += t_2 * (as_2 * tl.load(bsp + 2 * stride_bsk))[:, None]
            acc = acc * w[:, None]
            tl.store(c_base + (n_off + offs_bn)[None, :],
                     acc.to(c_ptr.dtype.element_ty), mask=token_mask[:, None])
        if PDL:
            gdc_launch_dependents()
        return
    for nb in tl.range(0, N_PER_CHUNK, disallow_acc_multi_buffer=True):
        n_off = n0 + nb * BLOCK_N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in tl.static_range(NUM_K):
            a = tl.load(a_base + k * BLOCK_K * stride_ak,
                        mask=token_mask[:, None], other=0.0)
            b = tl.load(b_base + (n_off + offs_bn)[None, :] * stride_bn
                        + k * BLOCK_K * stride_bk)
            a_s = tl.load(as_base + k * stride_ask, mask=token_mask, other=0.0)
            if SCALAR_B_SCALE:
                # The N tile sits inside one weight-scale group, so b_scale is a
                # scalar: one BLOCK_M-wide product instead of a full-tile
                # broadcast pair (which is what spills the accumulator).
                b_s = tl.load(bs_base + (n_off // GROUP_N) * stride_bsn
                              + k * stride_bsk)
                acc += tl.dot(a, b) * (a_s * b_s)[:, None]
            else:
                b_s = tl.load(bs_base + ((n_off + offs_bn) // GROUP_N) * stride_bsn
                              + k * stride_bsk)
                acc += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        acc = acc * w[:, None]
        tl.store(c_base + (n_off + offs_bn)[None, :],
                 acc.to(c_ptr.dtype.element_ty), mask=token_mask[:, None])
    if PDL:
        gdc_launch_dependents()


class _SharedBuf:
    """Mutable container so all FusedExperts layers share one set of scratch
    buffers. Layers execute sequentially so reuse is safe."""
    __slots__ = ("cache13", "a_fp8_1", "a_scale_1", "a_fp8_2", "a_scale_2",
                 "dg_ws1", "dg_ws2")
    def __init__(self):
        self.cache13 = None
        self.a_fp8_1 = None
        self.a_scale_1 = None
        self.a_fp8_2 = None
        self.a_scale_2 = None
        self.dg_ws1 = None
        self.dg_ws2 = None

_SHARED_BUF = _SharedBuf()


class FusedExperts(nn.Module):
    """Fused MoE experts: two grouped GEMMs with SiLU-mul in between."""

    def __init__(self, activation: str = "silu", config_style: str = "legacy"):
        super().__init__()
        if activation not in ("silu", "gelu_tanh"):
            raise ValueError(f"Unsupported MoE activation: {activation}")
        if config_style not in ("legacy", "vllm"):
            raise ValueError(f"Unsupported MoE config style: {config_style}")
        self.activation = activation
        self.config_style = config_style
        self.moe_align = MoeAlign()
        self.moe_grouped_gemm = MoeGroupedGemm()
        self.act_fn = SiluAndMul() if activation == "silu" else GeluAndMul("tanh")
        self.moe_sum = MoeSum()
        self.per_token_group_quant_fp8 = PerTokenGroupQuantFp8()
        self.silu_mul_quant_fp8 = SiluMulQuantFp8()
        self._sb = _SHARED_BUF

    def _get_cache13(self, total_elems, device, dtype):
        sb = self._sb
        if sb.cache13 is None or sb.cache13.numel() < total_elems:
            sb.cache13 = torch.empty(total_elems, device=device, dtype=dtype)
        return sb.cache13[:total_elems]

    def _get_fp8_bufs(self, buf_id, M, K, device):
        """``(fp8[M, K], fp32[M, ceil(K/128)])`` carved out of a shared flat store.

        Flat rather than 2-D so the returned views are always contiguous: a
        row/column slice of an over-wide buffer is not, and the L1 quantizer
        rejects a non-contiguous destination -- which a later, *narrower* call
        would otherwise hand it.
        """
        sb = self._sb
        attr_a = f"a_fp8_{buf_id}"
        attr_s = f"a_scale_{buf_id}"
        num_groups = math.ceil(K / _FP8_GROUP_SIZE)
        store_a = getattr(sb, attr_a)
        if store_a is None or store_a.numel() < M * K or store_a.device != device:
            store_a = torch.empty(M * K, dtype=torch.float8_e4m3fn, device=device)
            setattr(sb, attr_a, store_a)
        store_s = getattr(sb, attr_s)
        if (store_s is None or store_s.numel() < M * num_groups
                or store_s.device != device):
            store_s = torch.empty(M * num_groups, dtype=torch.float32, device=device)
            setattr(sb, attr_s, store_s)
        return (store_a[:M * K].view(M, K),
                store_s[:M * num_groups].view(M, num_groups))

    def _get_dg_workspace(self, buf_id, shape, device, dtype):
        sb = self._sb
        attr = f"dg_ws{buf_id}"
        existing = getattr(sb, attr)
        elem_size = torch.tensor([], dtype=dtype).element_size()
        needed_bytes = elem_size
        for s in shape:
            needed_bytes *= s
        if existing is None or existing.numel() < needed_bytes:
            setattr(sb, attr, torch.empty(needed_bytes, device=device, dtype=torch.uint8))
        raw = getattr(sb, attr)
        needed_elems = needed_bytes // elem_size
        return raw[:needed_bytes].view(dtype)[:needed_elems].view(shape)

    def forward(
        self,
        hidden_states: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        w13_scale: torch.Tensor | None = None,
        w2_scale: torch.Tensor | None = None,
        w13_scale_dg: torch.Tensor | None = None,
        w2_scale_dg: torch.Tensor | None = None,
        use_fp8_w8a8: bool = False,
        block_shape: list[int] | None = None,
    ) -> torch.Tensor:
        M, K = hidden_states.size()
        E, N2, _ = w13.size()
        N = N2 // 2
        top_k = topk_ids.size(1)

        if (self.activation == "silu"
                and use_fp8_w8a8
                and _valid_deep_gemm(hidden_states, w13, w2)
                and not torch.cuda.is_current_stream_capturing()):
            dg_w13_scale = w13_scale_dg if w13_scale_dg is not None else w13_scale
            dg_w2_scale = w2_scale_dg if w2_scale_dg is not None else w2_scale
            return self._forward_deep_gemm(
                hidden_states, w13, w2, topk_weights, topk_ids,
                num_experts, dg_w13_scale, dg_w2_scale, block_shape,
                M, K, E, N, N2, top_k,
            )
        # The fused pair partitions work by (expert block x 128-column group); with
        # only a handful of routed rows there are not enough of those to fill the
        # GPU, and the generic path's one-row-per-block naive mapping wins.
        enough_blocks = (min(num_experts, M * top_k) * (N // _FP8_GROUP_SIZE)
                         >= _num_sms(hidden_states.device) // 2)
        if enough_blocks and self._fused_ok(
                hidden_states, w13, w2, w13_scale, w2_scale,
                use_fp8_w8a8, block_shape, K, N, N2):
            return self._forward_fused(
                hidden_states, w13, w2, topk_weights, topk_ids,
                num_experts, w13_scale, w2_scale,
                M, K, E, N, N2, top_k,
            )
        return self._forward_triton(
            hidden_states, w13, w2, topk_weights, topk_ids,
            num_experts, w13_scale, w2_scale,
            use_fp8_w8a8, block_shape,
            M, K, E, N, N2, top_k,
        )

    def _fused_ok(self, hidden_states, w13, w2, w13_scale, w2_scale,
                  use_fp8_w8a8, block_shape, K, N, N2) -> bool:
        """The fused pair covers block-scaled FP8 SiLU MoE with 128-aligned dims."""
        if self.activation != "silu" or not use_fp8_w8a8:
            return False
        if block_shape is None or list(block_shape) != [_FP8_GROUP_SIZE, _FP8_GROUP_SIZE]:
            return False
        if w13_scale is None or w2_scale is None:
            return False
        if w13.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
            return False
        # bf16 only: the epilogue reproduces the reference's exact
        # fp32 -> bf16 -> SiLU -> bf16 -> quantize chain, which is what
        # ``MoeGroupedGemm``'s bf16 ``compute_type`` gives on the FP8 path.
        if hidden_states.dtype != torch.bfloat16:
            return False
        if N2 != 2 * N or K % _FP8_GROUP_SIZE or N % _FP8_GROUP_SIZE:
            return False
        if w13_scale.ndim != 3 or w2_scale.ndim != 3:
            return False
        if w13.stride(2) != 1 or w2.stride(2) != 1:
            return False
        if not hidden_states.is_contiguous():
            return False
        return True

    def _forward_fused(
        self,
        hidden_states, w13, w2, topk_weights, topk_ids,
        num_experts, w13_scale, w2_scale,
        M, K, E, N, N2, top_k,
    ) -> torch.Tensor:
        device = hidden_states.device
        rows = M * top_k
        ng = N // _FP8_GROUP_SIZE

        a_fp8, a_scale = self._get_fp8_bufs(1, M, K, device)
        self.per_token_group_quant_fp8(hidden_states, a_fp8, a_scale)

        # A 128-row alignment lets GEMM2 use a 128-row compute tile, which halves
        # the w2 bytes it moves per flop -- the dominant term at every size here.
        # It doubles the worst-case padding per live expert, which is why it used
        # to lose at M=314; with the promote overlapped (see the GEMM1 mainloop)
        # and TMA on the deep-pipeline branch it now wins or ties everywhere,
        # GEMM1+GEMM2 summed: 55.2 vs 56.0 us at M=314, 77.2 vs 77.9 at 643,
        # 109.4 vs 114.5 at 1000, 1126 vs 1202 at 16384.
        align = _ALIGN_BIG if rows >= _ALIGN_ROWS else _ALIGN

        sorted_ids, expert_ids, ntpp = self.moe_align(topk_ids, align, num_experts)

        EM = sorted_ids.size(0)
        if M < align:
            EM = min(EM, rows * align)

        h_fp8, h_scale = self._get_fp8_bufs(2, rows, N, device)

        # 32-bit tile offsets halve the address-math register footprint; only take
        # them when every offset this launch can form provably fits.
        int64_addr = max(rows * K, rows * N, E * N2 * K) > 2 ** 31 - 1

        # Programmatic dependent launch across the chain: each GEMM waits at
        # entry (before it touches MoeAlign's or its predecessor's output) and
        # signals after its stores, so the next kernel's blocks are resident by
        # the time the producer's tail drains.  ``MoeSum`` (L1) already launches
        # with the attribute and waits, so GEMM2's signal is what it was missing.
        pdl = bool(_iv("FK_L2_PDL", 1))
        block_m1 = min(_BM1, align)
        num_blocks1 = triton.cdiv(EM, block_m1) * ng
        # How much latency the grid itself hides decides the pipeline shape.
        # Under ~4 waves there is no second block per SM to overlap with, so buy
        # ILP inside the block (8 warps, two K steps per iteration); from ~4 to
        # ~16 waves a 2-stage pipeline keeps two blocks resident per SM, which is
        # worth more than a deeper one; past that the deep pipeline wins.
        waves1 = num_blocks1 / _num_sms(device)
        if _iv("FK_L2_W1", 0):
            warps1, stages1, unroll1 = (_iv("FK_L2_W1", 4), _iv("FK_L2_S1", 2),
                                        _iv("FK_L2_U1", 1))
        elif waves1 < _WAVE_CUTOFF:
            warps1, stages1, unroll1 = 8, 2, 2
        else:
            # With both dot results live the deep pipeline pays from ~4 waves up,
            # where before it only won past ~16 (GEMM1 at M = 643 / 1000:
            # 46.6 / 83.5 us at 4 warps / 3 stages against 51.9 / 94.8 at
            # 2 stages).  Below ~4 waves there is still no second block per SM to
            # overlap with, so ILP inside the block wins there.
            warps1, stages1, unroll1 = 4, 3, 1
        grid1 = (num_blocks1,)
        # The descriptor form views w13 as [E * N2, K], so it needs the tensor
        # fully contiguous (``_fused_ok`` only guarantees a unit last stride).
        # TMA needs the grid deep enough to amortize the per-block descriptor
        # setup, and the breakpoint is exactly the deep-pipeline one: on the
        # 4-warp / 3-stage branch descriptors win at every shape (GEMM1 at
        # M = 643 / 1000 / 16384: 40.7 / 74.3 / 721.9 us with, against
        # 46.4 / 83.7 / 860.0 without), while on the shallow branch below ~4
        # waves they lose (M=314: 35.4 against 33.0).
        use_tma = (waves1 >= _TMA_WAVES and w13.is_contiguous()
                   and int(_iv("FK_L2_TMA", 1)) and _tma_available())
        if use_tma:
            _gemm1_tma_kernel[grid1](
                a_fp8, a_scale, w13, w13_scale, h_fp8, h_scale,
                sorted_ids, expert_ids, ntpp,
                rows,
                E * N2, K,
                a_fp8.stride(0), a_fp8.stride(1),
                a_scale.stride(0), a_scale.stride(1),
                w13_scale.stride(0), w13_scale.stride(1), w13_scale.stride(2),
                h_fp8.stride(0),
                h_scale.stride(0), h_scale.stride(1),
                NG=ng,
                N_HALF=N,
                N2=N2,
                TOP_K=top_k,
                BLOCK_M=block_m1,
                BLOCK_N=_FP8_GROUP_SIZE,
                BLOCK_K=_FP8_GROUP_SIZE,
                NUM_K=K // _FP8_GROUP_SIZE,
                EXPERT_BLOCK_M=align,
                SKIP_EMPTY=block_m1 < align,
                K_UNROLL=unroll1,
                INT64=int64_addr,
                FP8_MAX=_FP8_MAX,
                PDL=pdl,
                num_warps=warps1,
                num_stages=stages1,
                launch_pdl=pdl,
            )
        else:
            _gemm1_act_quant_kernel[grid1](
                a_fp8, a_scale, w13, w13_scale, h_fp8, h_scale,
                sorted_ids, expert_ids, ntpp,
                rows,
                a_fp8.stride(0), a_fp8.stride(1),
                a_scale.stride(0), a_scale.stride(1),
                w13.stride(0), w13.stride(1), w13.stride(2),
                w13_scale.stride(0), w13_scale.stride(1), w13_scale.stride(2),
                h_fp8.stride(0),
                h_scale.stride(0), h_scale.stride(1),
                NG=ng,
                N_HALF=N,
                TOP_K=top_k,
                BLOCK_M=block_m1,
                BLOCK_N=_FP8_GROUP_SIZE,
                BLOCK_K=_FP8_GROUP_SIZE,
                NUM_K=K // _FP8_GROUP_SIZE,
                EXPERT_BLOCK_M=align,
                SKIP_EMPTY=block_m1 < align,
                K_UNROLL=unroll1,
                INT64=int64_addr,
                FP8_MAX=_FP8_MAX,
                PDL=pdl,
                num_warps=warps1,
                num_stages=stages1,
                launch_pdl=pdl,
            )

        mm2 = self._get_cache13(rows * K, device, hidden_states.dtype).view(rows, K)
        block_m2 = min(_BM2_BIG if align >= _ALIGN_BIG else _BM2, align)
        # The wider row tile needs the deeper pipeline and twice the warps: at
        # BLOCK_M=128 the measured GEMM2 at M=16384 is 423 us with 8 warps /
        # 3 stages against 527 with 2 stages, while BLOCK_M=64 wants 4 / 2.
        warps2, stages2 = ((8, 3) if block_m2 >= _ALIGN_BIG else (_W2, _S2))
        num_m_blocks = triton.cdiv(EM, block_m2)
        # A narrow N tile means more, shorter MMA steps per program.  r1 widened
        # it to 128 when row blocks alone did not fill the GPU; with the 128-row
        # tile and three live dot results that is now a loss everywhere -- GEMM2
        # at M = 314 / 643 / 1000 is 22.8 / 33.9 / 37.4 us at BLOCK_N=64 against
        # 26.8 / 38.5 / 41.9 at 128 (which needs 226 registers, and every one of
        # the three K-group temporaries scales with it).
        block_n2 = min(_BN2, K)
        while block_n2 > 16 and K % block_n2:
            block_n2 //= 2
        num_n_tiles = K // block_n2
        # Enough programs to fill the GPU, no more: every extra chunk re-reads the
        # expert's whole w2 slice (and, with HOIST_A, its A tile).  Once row
        # blocks alone are several waves deep, aim higher anyway: splitting N
        # shrinks the ``w2`` window that the resident blocks share, and that
        # window is what has to stay in L2.
        blocks_per_sm = (_TARGET_BLOCKS * 2 if num_m_blocks >= 4 * _num_sms(device)
                         else _TARGET_BLOCKS)
        target = blocks_per_sm * _num_sms(device)
        n_chunks = max(1, min(num_n_tiles, -(-target // max(num_m_blocks, 1))))
        while num_n_tiles % n_chunks:
            n_chunks -= 1
        _gemm2_kernel[(num_m_blocks * n_chunks,)](
            h_fp8, h_scale, w2, w2_scale, mm2, topk_weights,
            sorted_ids, expert_ids, ntpp,
            rows,
            h_fp8.stride(0), h_fp8.stride(1),
            h_scale.stride(0), h_scale.stride(1),
            w2.stride(0), w2.stride(1), w2.stride(2),
            w2_scale.stride(0), w2_scale.stride(1), w2_scale.stride(2),
            mm2.stride(0),
            num_m_blocks,
            N_PER_CHUNK=num_n_tiles // n_chunks,
            GROUP_N=_FP8_GROUP_SIZE,
            BLOCK_M=block_m2,
            BLOCK_N=block_n2,
            BLOCK_K=_FP8_GROUP_SIZE,
            NUM_K=N // _FP8_GROUP_SIZE,
            EXPERT_BLOCK_M=align,
            SKIP_EMPTY=block_m2 < align,
            INT64=int64_addr,
            SCALAR_B_SCALE=block_n2 <= _FP8_GROUP_SIZE,
            HOIST_A=(N // _FP8_GROUP_SIZE == 3
                     and block_n2 <= _FP8_GROUP_SIZE),
            PDL=pdl,
            num_warps=warps2,
            num_stages=stages2,
            launch_pdl=pdl,
        )

        return self.moe_sum(mm2, top_k)

    def _forward_deep_gemm(
        self,
        hidden_states, w13, w2, topk_weights, topk_ids,
        num_experts, w13_scale, w2_scale, block_shape,
        M, K, E, N, N2, top_k,
    ) -> torch.Tensor:
        """DeepGEMM path: permute -> grouped GEMM1 -> fused act+quant -> grouped GEMM2 -> unpermute."""
        alignment = _FP8_GROUP_SIZE

        M_sum = _compute_aligned_M(M, top_k, num_experts, alignment)

        a_fp8, a_scale = self._get_fp8_bufs(1, M, K, hidden_states.device)
        self.per_token_group_quant_fp8(hidden_states, a_fp8, a_scale)

        a1_perm, a1_scale_perm, expert_ids, inv_perm = _deepgemm_permute(
            a_fp8, a_scale, topk_ids, num_experts, alignment,
        )

        mm1_out = self._get_dg_workspace(1, (M_sum, N2), hidden_states.device, hidden_states.dtype)
        m_grouped_fp8_gemm_nt_contiguous(
            (a1_perm, a1_scale_perm), (w13, w13_scale), mm1_out, expert_ids,
        )

        quant_out = self._get_dg_workspace(
            2, (M_sum, N), hidden_states.device, torch.float8_e4m3fn,
        )
        a2_fp8, a2_scale = self.silu_mul_quant_fp8(
            mm1_out, output=quant_out,
        )

        mm2_out = self._get_dg_workspace(1, (M_sum, K), hidden_states.device, hidden_states.dtype)
        m_grouped_fp8_gemm_nt_contiguous(
            (a2_fp8, a2_scale), (w2, w2_scale), mm2_out, expert_ids,
        )

        output = torch.empty(M, K, dtype=hidden_states.dtype, device=hidden_states.device)
        _deepgemm_unpermute_and_reduce(mm2_out, topk_ids, topk_weights, inv_perm, output)
        return output

    def _forward_triton(
        self,
        hidden_states, w13, w2, topk_weights, topk_ids,
        num_experts, w13_scale, w2_scale,
        use_fp8_w8a8, block_shape,
        M, K, E, N, N2, top_k,
    ) -> torch.Tensor:
        """Generic Triton path (grouped GEMM per stage)."""
        config = get_triton_config(
            M, w13.shape, w2.shape, top_k,
            use_fp8=use_fp8_w8a8, block_shape=block_shape,
            default_style=self.config_style,
        )

        use_naive = (M * top_k * SPARSITY_FACTOR <= num_experts)

        sorted_token_ids, expert_ids, num_tokens_post_padded = self.moe_align(
            topk_ids, config["BLOCK_SIZE_M"], num_experts, naive=use_naive,
        )

        cache13_size = M * top_k * max(N2, K)
        cache13_flat = self._get_cache13(cache13_size, hidden_states.device, hidden_states.dtype)
        intermediate1 = cache13_flat[:M * top_k * N2].view(M * top_k, N2)
        intermediate3 = cache13_flat[:M * top_k * K].view(M * top_k, K)

        if use_fp8_w8a8:
            a_fp8, a_scale = self._get_fp8_bufs(1, M, K, hidden_states.device)
            self.per_token_group_quant_fp8(hidden_states, a_fp8, a_scale)
            gemm1_input = a_fp8
            gemm1_a_scale = a_scale
        else:
            gemm1_input = hidden_states
            gemm1_a_scale = None

        self.moe_grouped_gemm(
            gemm1_input, w13, intermediate1,
            topk_weights, sorted_token_ids, expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=False, top_k=top_k, config=config,
            a_scale=gemm1_a_scale, b_scale=w13_scale,
            use_fp8_w8a8=use_fp8_w8a8, block_shape=block_shape,
        )

        intermediate2 = self.act_fn(intermediate1)

        if use_fp8_w8a8:
            a2_fp8, a2_scale = self._get_fp8_bufs(2, M * top_k, N, hidden_states.device)
            self.per_token_group_quant_fp8(intermediate2, a2_fp8, a2_scale)
            gemm2_input = a2_fp8
            gemm2_a_scale = a2_scale
        else:
            gemm2_input = intermediate2
            gemm2_a_scale = None

        self.moe_grouped_gemm(
            gemm2_input, w2, intermediate3,
            topk_weights, sorted_token_ids, expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=True, top_k=1, config=config,
            a_scale=gemm2_a_scale, b_scale=w2_scale,
            use_fp8_w8a8=use_fp8_w8a8, block_shape=block_shape,
        )

        return self.moe_sum(intermediate3, top_k)
