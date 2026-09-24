"""Fused MoE experts: two grouped GEMMs with SiLU-mul in between.

Fast path (FP8 W8A8, 128x128 block-scaled weights, SiLU), five Triton kernels:

  1. per-token-group FP8 quantization of the activations (UE8M0 scales)
  2. routing: expert-sorted row blocks -- no kernel at all for a handful of
     rows, ``MoeAlign`` for mid sizes, a parallel histogram/scan/scatter at
     large batch (``moe_align_block_size`` scans every id from one block)
  3. grouped GEMM1 over all 2N gate/up columns, TMA-gathering the A rows
     straight out of the unpermuted activation buffer (no token permute pass)
  4. SiLU-mul + FP8 re-quantization of the [rows, N] intermediate
  5. grouped GEMM2 (routed-weight scaled) + top-k reduction

Numerics match the unfused reference exactly: per-128-K block scales are
applied to an FP32 accumulator (the scalar weight-block scale is folded into
the per-row activation scale so the promotion is a single FFMA), the GEMM1
accumulator is rounded to bf16 before the activation, SiLU is rounded to bf16
before the multiply, and the re-quantization uses the same UE8M0 group scales.

Anything outside that configuration (bf16 weights, other block shapes,
non-SiLU activation) falls back to the original unfused implementation.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

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
# FP8 E4M3 range and the quantization epsilon, as Triton constexprs (the
# reference CUDA/Triton quantizers use exactly these).
_TL_FP8_MAX = tl.constexpr(448.0)
_TL_EPS = tl.constexpr(1e-10)


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

@triton.jit
def _route_hist_kernel(ids_ptr, hist_ptr, sorted_ptr, R, max_padded,
                       E: tl.constexpr, BLK: tl.constexpr, FILL: tl.constexpr):
    """Per-block expert histogram + sentinel-fill of the sorted-id buffer."""
    pid = tl.program_id(0)
    offs = pid * BLK + tl.arange(0, BLK)
    n_valid = max(min(BLK, R - pid * BLK), 0)
    e = tl.load(ids_ptr + offs, mask=offs < R, other=0)
    h = tl.histogram(e, E)
    h = h - tl.where(tl.arange(0, E) == 0, BLK - n_valid, 0)
    tl.store(hist_ptr + pid * E + tl.arange(0, E), h)
    foffs = pid * FILL + tl.arange(0, FILL)
    tl.store(sorted_ptr + foffs, R, mask=foffs < max_padded)


@triton.jit
def _route_scan_kernel(hist_ptr, cursor_ptr, eid_ptr, npp_ptr, nb,
                       E: tl.constexpr, NBP: tl.constexpr, BM: tl.constexpr,
                       MAXB: tl.constexpr):
    """One program: expert offsets, per-block write cursors, block->expert map."""
    offs_b = tl.arange(0, NBP)
    offs_e = tl.arange(0, E)
    h = tl.load(hist_ptr + offs_b[:, None] * E + offs_e[None, :],
                mask=(offs_b < nb)[:, None], other=0)
    cnt = tl.sum(h, axis=0)
    nblk = (cnt + BM - 1) // BM
    padded = nblk * BM
    poff = tl.cumsum(padded) - padded
    boff = tl.cumsum(nblk) - nblk
    tl.store(npp_ptr, tl.sum(padded))
    tl.store(cursor_ptr + offs_b[:, None] * E + offs_e[None, :],
             poff[None, :] + tl.cumsum(h, axis=0) - h,
             mask=(offs_b < nb)[:, None])
    for j0 in range(0, tl.cdiv(tl.sum(nblk), MAXB) * MAXB, MAXB):
        offs_j = j0 + tl.arange(0, MAXB)
        tl.store(eid_ptr + offs_j,
                 tl.sum(tl.where(boff[None, :] <= offs_j[:, None], 1, 0), axis=1) - 1)


@triton.jit
def _route_scatter_kernel(ids_ptr, cursor_ptr, sorted_ptr, R,
                          E: tl.constexpr, BLK: tl.constexpr):
    """Place each flat (token, slot) id in its expert's region of the sorted array."""
    pid = tl.program_id(0)
    offs = pid * BLK + tl.arange(0, BLK)
    mask = offs < R
    e = tl.load(ids_ptr + offs, mask=mask, other=0)
    pos = tl.atomic_add(cursor_ptr + pid * E + e, 1, mask=mask)
    tl.store(sorted_ptr + pos, offs, mask=mask)


@triton.jit
def _act_quant_kernel(x_ptr, q_ptr, s_ptr, n_groups, GPB: tl.constexpr):
    """Per-128-element-group FP8 quantization with UE8M0 (power-of-two) scales.

    ``x`` is contiguous [M, K] with K % 128 == 0, so group ``g`` covers the flat
    range [g*128, (g+1)*128) and the scale buffer is [M, K/128] row-major.
    """
    pid = tl.program_id(0)
    offs_g = pid * GPB + tl.arange(0, GPB)
    gmask = offs_g < n_groups
    idx = offs_g[:, None] * 128 + tl.arange(0, 128)[None, :]
    x = tl.load(x_ptr + idx, mask=gmask[:, None], other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), _TL_EPS)
    scale = tl.math.exp2(tl.math.ceil(tl.math.log2(amax / _TL_FP8_MAX)))
    q = tl.clamp(x / scale[:, None], -_TL_FP8_MAX, _TL_FP8_MAX)
    tl.store(q_ptr + idx, q.to(q_ptr.dtype.element_ty), mask=gmask[:, None])
    tl.store(s_ptr + offs_g, scale, mask=gmask)


@triton.jit
def _gemm1_kernel(
    a_desc, as_ptr, w_ptr, ws_ptr, c_ptr,
    sorted_ptr, eid_ptr, npp_ptr,
    num_valid,
    K: tl.constexpr, N2: tl.constexpr, TOPK: tl.constexpr,
    stride_we: tl.constexpr, stride_wse: tl.constexpr, stride_wsn: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BALIGN: tl.constexpr,
):
    """Grouped GEMM1: C[rows, 2N] = A[token(rows)] @ W13[expert]^T.

    The A rows are TMA-gathered from the unpermuted activation buffer, so no
    token-permutation pass is needed.  A column tile of at most 128 keeps the
    weight-block scale scalar, which makes the per-128-K accumulator promotion a
    single FFMA.  grid = (2N / BN, ceil(max_padded / BM)).
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    row0 = pid_m * BM
    if row0 >= tl.load(npp_ptr):
        return

    e = tl.load(eid_ptr + row0 // BALIGN)
    offs_m = tl.arange(0, BM)
    tok = tl.load(sorted_ptr + row0 + offs_m)
    mask_m = tok < num_valid
    row = tl.where(mask_m, tok // TOPK, 0)

    n0 = pid_n * BN
    offs_k = tl.arange(0, 128)
    as_ptrs = as_ptr + row * (K // 128)
    wsp = ws_ptr + e * stride_wse + (n0 // 128) * stride_wsn
    # The weight tile is loaded with plain pointer arithmetic: combining a TMA
    # gather (A) with a TMA tile load (B) in one loop miscompiles on Triton
    # 3.6 / sm100 (nondeterministic results), and the gather is what matters.
    bp = (w_ptr + e.to(tl.int64) * stride_we
          + (n0 + tl.arange(0, BN))[None, :] * K + offs_k[:, None])

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K // 128):
        a = a_desc.gather(row, k * 128)
        b = tl.load(bp)
        s = tl.load(as_ptrs + k, mask=mask_m, other=0.0) * tl.load(wsp + k)
        acc += tl.dot(a, b) * s[:, None]
        bp += 128

    rows = row0 + offs_m
    tl.store(c_ptr + rows[:, None] * N2 + (n0 + tl.arange(0, BN))[None, :],
             acc.to(c_ptr.dtype.element_ty), mask=mask_m[:, None])


@triton.jit
def _silu_mul_quant_kernel(
    c_ptr, q_ptr, qs_ptr, npp_ptr,
    N: tl.constexpr, BR: tl.constexpr, NG: tl.constexpr, NGP: tl.constexpr,
):
    """SiLU(gate)*up on [rows, 2N] bf16 -> FP8 [rows, N] + UE8M0 group scales.

    Rounds SiLU to bf16 before the multiply and the product to bf16 before
    quantizing, matching the reference activation + quantization kernels.
    """
    pid = tl.program_id(0)
    r0 = pid * BR
    if r0 >= tl.load(npp_ptr):
        return
    rows = r0 + tl.arange(0, BR)
    offs_g = tl.arange(0, NGP)
    cols = offs_g[:, None] * 128 + tl.arange(0, 128)[None, :]
    gmask = (offs_g < NG)[None, :, None]
    base = c_ptr + rows[:, None, None] * (2 * N) + cols[None, :, :]
    g = tl.load(base, mask=gmask, other=0.0).to(tl.float32)
    u = tl.load(base + N, mask=gmask, other=0.0).to(tl.float32)
    sil = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    y = (sil * u).to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(y), axis=2), _TL_EPS)
    scale = tl.math.exp2(tl.math.ceil(tl.math.log2(amax / _TL_FP8_MAX)))
    q = tl.clamp(y / scale[:, :, None], -_TL_FP8_MAX, _TL_FP8_MAX)
    tl.store(q_ptr + rows[:, None, None] * N + cols[None, :, :],
             q.to(q_ptr.dtype.element_ty), mask=gmask)
    tl.store(qs_ptr + rows[:, None] * NG + offs_g[None, :], scale,
             mask=(offs_g < NG)[None, :])


@triton.jit
def _gemm2_kernel(
    q_ptr, qs_ptr, w_desc, ws_ptr, tw_ptr, out_ptr,
    sorted_ptr, eid_ptr, npp_ptr,
    num_valid,
    N: tl.constexpr, KK: tl.constexpr,
    stride_wse: tl.constexpr, stride_wsn: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, NL: tl.constexpr,
    BALIGN: tl.constexpr,
):
    """Grouped GEMM2 with routed-weight scaling, writing per-(token, slot) rows.

    K is small (one 128-group per iteration, KK/128 total), so the A tiles are
    hoisted out of the column loop and each program walks NL column tiles.
    grid = (N / (BN*NL), ceil(max_padded / BM)).
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    row0 = pid_m * BM
    if row0 >= tl.load(npp_ptr):
        return

    e = tl.load(eid_ptr + row0 // BALIGN)
    offs_m = tl.arange(0, BM)
    tok = tl.load(sorted_ptr + row0 + offs_m)
    mask_m = tok < num_valid
    rows = row0 + offs_m
    w = tl.load(tw_ptr + tok, mask=mask_m, other=0.0)

    offs_k = tl.arange(0, 128)
    a_ptrs = q_ptr + rows[:, None] * KK + offs_k[None, :]
    a0 = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
    a1 = tl.load(a_ptrs + 128, mask=mask_m[:, None], other=0.0)
    a2 = tl.load(a_ptrs + 256, mask=mask_m[:, None], other=0.0)
    sp = qs_ptr + rows * (KK // 128)
    s0 = tl.load(sp, mask=mask_m, other=0.0)
    s1 = tl.load(sp + 1, mask=mask_m, other=0.0)
    s2 = tl.load(sp + 2, mask=mask_m, other=0.0)
    wsbase = ws_ptr + e * stride_wse
    nrow0 = e * N

    for nl in range(0, NL):
        n0 = (pid_n * NL + nl) * BN
        sbp = wsbase + (n0 // 128) * stride_wsn
        acc = tl.dot(a0, tl.trans(w_desc.load([nrow0 + n0, 0]))) * (s0 * tl.load(sbp))[:, None]
        acc += tl.dot(a1, tl.trans(w_desc.load([nrow0 + n0, 128]))) * (s1 * tl.load(sbp + 1))[:, None]
        acc += tl.dot(a2, tl.trans(w_desc.load([nrow0 + n0, 256]))) * (s2 * tl.load(sbp + 2))[:, None]
        acc = acc * w[:, None]
        tl.store(out_ptr + tok[:, None] * N + (n0 + tl.arange(0, BN))[None, :],
                 acc.to(out_ptr.dtype.element_ty), mask=mask_m[:, None])


@triton.jit
def _topk_reduce_kernel(p_ptr, o_ptr, N: tl.constexpr, TOPK: tl.constexpr,
                        BLK: tl.constexpr):
    """out[m] = sum_s partial[m, s] accumulated in FP32 (matches moe_sum)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLK + tl.arange(0, BLK)
    base = pid_m * (TOPK * N) + offs_n
    acc = tl.zeros((BLK,), dtype=tl.float32)
    for s in tl.static_range(TOPK):
        acc += tl.load(p_ptr + base + s * N).to(tl.float32)
    tl.store(o_ptr + pid_m * N + offs_n, acc.to(o_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
# Tiling heuristics (B200, tuned per total row count)
# ---------------------------------------------------------------------------

_TILES = (
    #   rows <=           gemm1                        gemm2                 reduce
    (64,        dict(bm1=16,  bn1=128, w1=4, s1=3, bm2=16,  nl2=1,  w2=8, s2=2, red=512)),
    # Below ~512 rows the per-expert block padding (E * (align - 1) rows) is what
    # costs, not the tile shape, so keep the alignment small.
    (512,       dict(bm1=16,  bn1=128, w1=4, s1=3, bm2=16,  nl2=2,  w2=8, s2=2, red=512)),
    (3072,      dict(bm1=128, bn1=128, w1=8, s1=4, bm2=128, nl2=8,  w2=8, s2=2, red=512)),
    (6144,      dict(bm1=128, bn1=128, w1=8, s1=3, bm2=128, nl2=16, w2=8, s2=2, red=512)),
    (16384,     dict(bm1=64,  bn1=128, w1=4, s1=4, bm2=64,  nl2=16, w2=8, s2=3, red=512)),
    (1 << 60,   dict(bm1=64,  bn1=128, w1=4, s1=3, bm2=128, nl2=16, w2=8, s2=2, red=4096)),
)


def _tile_config(rows: int) -> dict:
    for bound, cfg in _TILES:
        if rows <= bound:
            cfg = dict(cfg)
            break
    env = os.environ.get("FK_MOE_TILES")
    if env:
        for item in env.split(","):
            k, v = item.split("=")
            cfg[k.strip()] = int(v)
    return cfg


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
    M, K = output.size()
    top_k = topk_ids.size(1)

    flat_positions = inv_perm.to(torch.int64).view(-1)
    gathered = mm2_out[flat_positions].view(M, top_k, K)
    weights = topk_weights.unsqueeze(-1)
    output.copy_((gathered.to(output.dtype) * weights).sum(dim=1))


class _SharedBuf:
    """Mutable container so all FusedExperts layers share one set of scratch
    buffers. Layers execute sequentially so reuse is safe."""
    __slots__ = ("cache13", "a_fp8_1", "a_scale_1", "a_fp8_2", "a_scale_2",
                 "dg_ws1", "dg_ws2", "mm1", "mid_q", "mid_s", "partial",
                 "hist", "cursor", "sorted_ids", "eid", "npp",
                 "naive_ids", "naive_npp", "naive_key")

    def __init__(self):
        self.cache13 = None
        self.a_fp8_1 = None
        self.a_scale_1 = None
        self.a_fp8_2 = None
        self.a_scale_2 = None
        self.dg_ws1 = None
        self.dg_ws2 = None
        self.mm1 = None
        self.mid_q = None
        self.mid_s = None
        self.partial = None
        self.hist = None
        self.cursor = None
        self.sorted_ids = None
        self.eid = None
        self.npp = None
        self.naive_ids = None
        self.naive_npp = None
        self.naive_key = None


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

    # -- scratch helpers ----------------------------------------------------
    def _get_cache13(self, total_elems, device, dtype):
        sb = self._sb
        if sb.cache13 is None or sb.cache13.numel() < total_elems:
            sb.cache13 = torch.empty(total_elems, device=device, dtype=dtype)
        return sb.cache13[:total_elems]

    def _get_fp8_bufs(self, buf_id, M, K, device):
        sb = self._sb
        attr_a = f"a_fp8_{buf_id}"
        attr_s = f"a_scale_{buf_id}"
        num_groups = math.ceil(K / _FP8_GROUP_SIZE)
        existing_a = getattr(sb, attr_a)
        if existing_a is None or existing_a.size(0) < M or existing_a.size(1) < K:
            setattr(sb, attr_a, torch.empty(M, K, dtype=torch.float8_e4m3fn, device=device))
            setattr(sb, attr_s, torch.empty(M, num_groups, dtype=torch.float32, device=device))
        a = getattr(sb, attr_a)
        s = getattr(sb, attr_s)
        return a[:M, :K], s[:M, :num_groups]

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

    def _get_mm1(self, rows, N2, device, dtype):
        sb = self._sb
        if (sb.mm1 is None or sb.mm1.size(0) < rows or sb.mm1.size(1) != N2
                or sb.mm1.dtype != dtype):
            sb.mm1 = torch.empty(rows, N2, dtype=dtype, device=device)
        return sb.mm1[:rows]

    def _get_mid(self, rows, N, device):
        sb = self._sb
        if sb.mid_q is None or sb.mid_q.size(0) < rows or sb.mid_q.size(1) != N:
            sb.mid_q = torch.empty(rows, N, dtype=torch.float8_e4m3fn, device=device)
            sb.mid_s = torch.empty(rows, N // _FP8_GROUP_SIZE, dtype=torch.float32,
                                   device=device)
        return sb.mid_q[:rows], sb.mid_s[:rows]

    def _get_partial(self, rows, K, device, dtype):
        sb = self._sb
        if (sb.partial is None or sb.partial.size(0) < rows
                or sb.partial.size(1) != K or sb.partial.dtype != dtype):
            sb.partial = torch.empty(rows, K, dtype=dtype, device=device)
        return sb.partial[:rows]


    # -- routing ------------------------------------------------------------
    # A handful of rows: skip routing entirely.  One block per (token, slot),
    # with the expert taken straight from ``topk_ids`` -- no sort needed, and the
    # sorted-id / padded-length tensors are constants we can cache.
    _NAIVE_MAX_ROWS = 32

    def _naive_route(self, topk_ids, block_m, R, device):
        sb = self._sb
        key = (R, block_m)
        if sb.naive_key != key:
            ids = torch.full((R * block_m,), R, dtype=torch.int32, device="cpu")
            ids[torch.arange(R) * block_m] = torch.arange(R, dtype=torch.int32)
            sb.naive_ids = ids.to(device, non_blocking=True)
            sb.naive_npp = torch.full((1,), R * block_m, dtype=torch.int32,
                                      device=device)
            sb.naive_key = key
        return sb.naive_ids, topk_ids.view(-1), sb.naive_npp

    _ROUTE_MIN_ROWS = 16384

    def _route(self, topk_ids, block_m, E, device):
        """Expert-sorted block layout (drop-in for ``MoeAlign``).

        A single block of ``moe_align_block_size`` scans every routing id
        serially, which dominates at large batch; this does the histogram, the
        expert offsets and the scatter in three parallel kernels.
        """
        sb = self._sb
        R = topk_ids.numel()
        max_padded = R + E * (block_m - 1) if R >= E else R * block_m
        max_padded = (max_padded + 127) // 128 * 128
        max_blocks = max_padded // block_m + 1
        nb = min(128, max(1, triton.cdiv(R, 256)))
        blk = triton.next_power_of_2(triton.cdiv(R, nb))
        nb = triton.cdiv(R, blk)
        nbp = triton.next_power_of_2(nb)
        eid_cap = (max_blocks + 255) // 256 * 256

        if sb.hist is None or sb.hist.numel() < nbp * E:
            sb.hist = torch.empty(nbp * E, dtype=torch.int32, device=device)
            sb.cursor = torch.empty(nbp * E, dtype=torch.int32, device=device)
        if sb.sorted_ids is None or sb.sorted_ids.numel() < max_padded:
            sb.sorted_ids = torch.empty(max_padded, dtype=torch.int32, device=device)
        if sb.eid is None or sb.eid.numel() < eid_cap:
            sb.eid = torch.empty(eid_cap, dtype=torch.int32, device=device)
        if sb.npp is None:
            sb.npp = torch.empty(1, dtype=torch.int32, device=device)

        ids = topk_ids.view(-1)
        fill = triton.next_power_of_2(triton.cdiv(max_padded, nb))
        _route_hist_kernel[(nb,)](ids, sb.hist, sb.sorted_ids, R, max_padded,
                                  E=E, BLK=blk, FILL=fill, num_warps=4)
        _route_scan_kernel[(1,)](sb.hist, sb.cursor, sb.eid, sb.npp, nb,
                                 E=E, NBP=nbp, BM=block_m, MAXB=256, num_warps=4)
        _route_scatter_kernel[(nb,)](ids, sb.cursor, sb.sorted_ids, R,
                                     E=E, BLK=blk, num_warps=4)
        return sb.sorted_ids[:max_padded], sb.eid, sb.npp

    # -- dispatch -----------------------------------------------------------
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
                and block_shape is not None
                and block_shape[0] == _FP8_GROUP_SIZE
                and block_shape[1] == _FP8_GROUP_SIZE
                and w13_scale is not None and w2_scale is not None
                and hidden_states.dtype == torch.bfloat16
                and w13.dtype == torch.float8_e4m3fn
                and w2.dtype == torch.float8_e4m3fn
                and K % _FP8_GROUP_SIZE == 0 and N % _FP8_GROUP_SIZE == 0
                and w2.size(1) == K and w2.size(2) == N
                and hidden_states.is_contiguous()
                and w13.is_contiguous() and w2.is_contiguous()
                and w13_scale.is_contiguous() and w2_scale.is_contiguous()
                and topk_weights.is_contiguous() and topk_ids.is_contiguous()
                and M > 0 and num_experts == E
                # all kernels index with 32-bit offsets
                and max(M, M * top_k) * max(K, N2) < 2 ** 31
                and E * N2 * K < 2 ** 31):
            return self._forward_fused(
                hidden_states, w13, w2, topk_weights, topk_ids,
                num_experts, w13_scale, w2_scale, M, K, E, N, N2, top_k,
            )

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
        return self._forward_triton(
            hidden_states, w13, w2, topk_weights, topk_ids,
            num_experts, w13_scale, w2_scale,
            use_fp8_w8a8, block_shape,
            M, K, E, N, N2, top_k,
        )

    # -- fused fast path ----------------------------------------------------
    def _forward_fused(
        self,
        hidden_states, w13, w2, topk_weights, topk_ids,
        num_experts, w13_scale, w2_scale,
        M, K, E, N, N2, top_k,
    ) -> torch.Tensor:
        device = hidden_states.device
        rows = M * top_k
        cfg = _tile_config(rows)
        bm1, bm2 = cfg["bm1"], cfg["bm2"]
        balign = max(bm1, bm2)
        ng = N // _FP8_GROUP_SIZE

        a_fp8, a_scale = self._get_fp8_bufs(1, M, K, device)
        n_groups = M * (K // _FP8_GROUP_SIZE)
        # Wide-but-not-too-wide grid: one program per GPB groups, keeping at
        # least ~1 program per SM.  Fewer, larger programs cut CTA-launch cost
        # on the big shapes (203 MB at M=16384 goes 1.8 -> 3.8 TB/s).
        gpb = 32
        while gpb > 1 and (n_groups % gpb or n_groups // gpb < 148):
            gpb //= 2
        _act_quant_kernel[(triton.cdiv(n_groups, gpb),)](
            hidden_states, a_fp8, a_scale, n_groups, GPB=gpb, num_warps=2,
        )

        if rows <= self._NAIVE_MAX_ROWS:
            sorted_ids, expert_ids, num_post = self._naive_route(
                topk_ids, balign, rows, device)
        elif rows >= self._ROUTE_MIN_ROWS:
            sorted_ids, expert_ids, num_post = self._route(
                topk_ids, balign, num_experts, device)
        else:
            sorted_ids, expert_ids, num_post = self.moe_align(
                topk_ids, balign, num_experts, naive=False,
            )
        max_padded = sorted_ids.size(0)

        mm1 = self._get_mm1(max_padded, N2, device, hidden_states.dtype)
        a_desc = TensorDescriptor.from_tensor(a_fp8, [1, 128])
        bn1 = min(cfg["bn1"], _FP8_GROUP_SIZE)
        _gemm1_kernel[(N2 // bn1, triton.cdiv(max_padded, bm1))](
            a_desc, a_scale, w13, w13_scale, mm1,
            sorted_ids, expert_ids, num_post, rows,
            K=K, N2=N2, TOPK=top_k, stride_we=w13.stride(0),
            stride_wse=w13_scale.stride(0), stride_wsn=w13_scale.stride(1),
            BM=bm1, BN=bn1, BALIGN=balign,
            num_warps=cfg["w1"], num_stages=cfg["s1"],
        )

        mid_q, mid_s = self._get_mid(max_padded, N, device)
        br = 8 if max_padded % 8 == 0 else 1
        _silu_mul_quant_kernel[(triton.cdiv(max_padded, br),)](
            mm1, mid_q, mid_s, num_post, N=N, BR=br, NG=ng,
            NGP=triton.next_power_of_2(ng), num_warps=2,
        )

        partial = self._get_partial(rows, K, device, hidden_states.dtype)
        w2_desc = TensorDescriptor.from_tensor(w2.view(E * K, N), [128, 128])
        _gemm2_kernel[(K // (128 * cfg["nl2"]), triton.cdiv(max_padded, bm2))](
            mid_q, mid_s, w2_desc, w2_scale, topk_weights, partial,
            sorted_ids, expert_ids, num_post, rows,
            N=K, KK=N,
            stride_wse=w2_scale.stride(0), stride_wsn=w2_scale.stride(1),
            BM=bm2, BN=128, NL=cfg["nl2"], BALIGN=balign,
            num_warps=cfg["w2"], num_stages=cfg["s2"],
        )

        out = torch.empty(M, K, dtype=hidden_states.dtype, device=device)
        blk = cfg["red"] if K % cfg["red"] == 0 else 128
        _topk_reduce_kernel[(M, K // blk)](
            partial, out, N=K, TOPK=top_k, BLK=blk, num_warps=4,
        )
        return out

    def _forward_deep_gemm(
        self,
        hidden_states, w13, w2, topk_weights, topk_ids,
        num_experts, w13_scale, w2_scale, block_shape,
        M, K, E, N, N2, top_k,
    ) -> torch.Tensor:
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
        """Triton fallback path (original implementation with JSON autotuning)."""
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
