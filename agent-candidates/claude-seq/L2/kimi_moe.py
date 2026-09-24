"""Kimi-Linear MoE as one fused pipeline: router, grouped GEMMs, reduction.

The baseline calls ``flashinfer::trtllm_bf16_moe`` for the routed experts and
runs the shared expert beside it as a separate ``LlamaMLP`` (two cuBLAS GEMMs, a
SiLU-mul and an add), with the gate GEMM, the routing kernel and the weighted
reduction on top -- eleven kernels per layer.

Everything here is Triton, and the shared expert becomes *slot ``num_experts``*
of the same grouped problem (its ``gate_up_proj`` / ``down_proj`` already have an
expert's shapes, ``[2I, H]`` and ``[H, I]``), so one pipeline of six kernels does
the whole layer with ``top_k + 1`` rows per token:

  1. ``_logits_kernel``    x @ gate.weight^T -> fp32 logits (split over K when a
                           single tile per expert block would not fill the GPU;
                           also clears the per-slot row counter for step 2)
  2. ``_route_kernel``     sigmoid + router bias -> top-k, renormalize, apply
                           ``routed_scaling_factor``, and scatter every
                           (token, slot) pair into that slot's row list
  3. ``_blockmap_kernel``  flatten the row lists into one (slot, first row, rows)
                           triple per GEMM row block
  4. ``_gemm1_kernel``     grouped ``[rows, H] @ w13[slot]^T`` with SwiGLU fused
                           into the epilogue, so the 2I-wide intermediate never
                           reaches HBM
  5. ``_gemm2_kernel``     grouped ``[rows, I] @ w2[slot]^T``, routing weight in
                           the epilogue, scattered into ``[M, top_k + 1, H]``
  6. ``_reduce_kernel``    sum each token's ``top_k + 1`` contributions -> bf16

Three things carry most of the speedup:

* **No prefix-sum pass and no host sync.** Row lists use a fixed
  ``C = round_up(M, BM)`` stride per slot, so ``_route_kernel`` takes each pair's
  slot-local rank straight from the counter's ``atomicAdd`` return value, and the
  whole scatter is one masked expert-wide vector store.
* **Block-major-K expert weights** (``[slot, K/128, rows, 128]``, built once in
  ``process_weights_after_loading``): every weight tile a GEMM step needs is then
  one contiguous run instead of ``BN`` chunks strided by ``K``.
* **An L2-friendly CTA order** (``_tile_of_pid``): grouping ``GM`` row blocks
  across all N tiles keeps a block's A rows and a slot's weight slice resident
  while they are reused, which at 16k tokens removes ~6 GB of re-read traffic
  from GEMM1 alone.

Small token counts are additionally captured into a CUDA graph: at one to a few
dozen tokens the pipeline is ~40 us of GPU work, too little for the host to keep
six Triton launches in flight.

The captured shapes span 1 to 16384 tokens, i.e. from pure HBM streaming (every
touched expert's 14 MB is read once, so latency is set by how many loads are in
flight) to compute-bound (2.1 TFLOP of bf16 MMA), which is why the tile table
below is indexed by token count.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ....infra.tp import _tp_rank, _tp_size
from ..L1.allreduce import AllReduce
from .llama_mlp import LlamaMLP
from .parallel_linear import ReplicatedLinear


# Weight K-blocking factor: the GEMM k-step, and the inner dimension of the
# block-major-K layout ``process_weights_after_loading`` produces.
_BLK = 128

# Expert-count padding for the router / block-map vector ops.
_EPOW = 512


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
@triton.jit
def _logits_kernel(
    x_ptr, w_ptr, out_ptr, cnt_ptr,
    M,
    K: tl.constexpr, NE: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    KSPLIT: tl.constexpr, CNT_LEN: tl.constexpr,
):
    """logits[split, M, NE] (fp32) = x[M, K] @ w[NE, K]^T; CTA 0 clears counters.

    A few tokens do not fill the GPU with one CTA per (M, NE) tile, so the K axis
    is cut into ``K // KSPLIT`` slices and the partials are summed by the router
    (cheap: a couple of extra 256-float rows per token) instead of with atomics,
    which would need the output cleared first.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    if (pid_m == 0) & (pid_n == 0) & (pid_k == 0):
        tl.store(cnt_ptr + tl.arange(0, CNT_LEN), tl.zeros((CNT_LEN,), tl.int32))

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = pid_k * KSPLIT + tl.arange(0, BK)
    m_mask = offs_m < M
    a_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = w_ptr + offs_n[None, :] * K + offs_k[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(KSPLIT // BK):
        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK
    out = out_ptr + pid_k * (M * NE) + offs_m[:, None] * NE + offs_n[None, :]
    tl.store(out, acc, mask=m_mask[:, None])


@triton.jit
def _route_kernel(
    logits_ptr, bias_ptr, cnt_ptr, stok_ptr, sw_ptr, sdst_ptr,
    M,
    NE: tl.constexpr, EPOW: tl.constexpr, TOPK: tl.constexpr, NSLOT: tl.constexpr,
    C: tl.constexpr, SCALE: tl.constexpr, SHARED: tl.constexpr, SPLIT: tl.constexpr,
):
    """Top-k over sigmoid(logits) + bias, then scatter the (token, slot) pairs.

    Selection uses the biased score; the routing weight is the *unbiased* sigmoid
    renormalized over the selected set and scaled by ``routed_scaling_factor`` --
    vLLM's grouped-topk semantics for ``n_group == topk_group == 1``.

    One CTA per token, and the whole scatter is expert-vector-wide: a single
    masked ``atomic_add`` over the counters hands every selected expert its
    slot-local rank at once, so the kernel costs ``top_k`` register reductions
    plus one atomic instead of ``top_k`` dependent round trips through memory.
    """
    m = tl.program_id(0)
    offs = tl.arange(0, EPOW)
    valid = offs < NE
    lp = logits_ptr + m * NE + offs
    logit = tl.load(lp, mask=valid, other=0.0)
    for sp in tl.static_range(1, SPLIT):
        logit += tl.load(lp + sp * (M * NE), mask=valid, other=0.0)
    score = tl.sigmoid(logit)
    bias = tl.load(bias_ptr + offs, mask=valid, other=0.0).to(tl.float32)
    biased = tl.where(valid, score + bias, -float("inf"))

    sel = tl.zeros((EPOW,), dtype=tl.int32)
    for _ in range(TOPK):
        hit = offs == tl.argmax(biased, 0)
        sel = tl.where(hit, 1, sel)
        biased = tl.where(hit, -float("inf"), biased)

    chosen = sel > 0
    wsum = tl.sum(tl.where(chosen, score, 0.0), 0)
    weight = score * (SCALE / wsum)
    lane = tl.cumsum(sel, 0) - 1  # 0..TOPK-1, ordered by expert index
    rank = tl.atomic_add(cnt_ptr + offs, tl.full((EPOW,), 1, tl.int32), mask=chosen)
    pos = offs * C + rank
    zero = tl.zeros((EPOW,), dtype=tl.int32)
    tl.store(stok_ptr + pos, zero + m, mask=chosen)
    tl.store(sw_ptr + pos, weight, mask=chosen)
    tl.store(sdst_ptr + pos, m * (TOPK + 1) + lane, mask=chosen)

    if SHARED:
        spos = (NSLOT - 1) * C + m
        tl.store(stok_ptr + spos, m)
        tl.store(sw_ptr + spos, 1.0)
        tl.store(sdst_ptr + spos, m * (TOPK + 1) + TOPK)
        if m == 0:
            tl.store(cnt_ptr + (NSLOT - 1), M)


@triton.jit
def _blockmap_kernel(
    cnt_ptr, bslot_ptr, brow_ptr, bnrow_ptr,
    NSLOT: tl.constexpr, EPOW: tl.constexpr, C: tl.constexpr, BM: tl.constexpr,
    MAXB: tl.constexpr,
):
    """Flatten the routing into one (slot, first row, rows) triple per GEMM block.

    A single CTA: the GEMM kernels then start from three scalar loads instead of
    re-deriving the block -> slot map from a 512-wide prefix sum each, which at
    16k tokens is ~10k CTAs paying for the same scan.  Blocks past the real count
    keep ``rows = 0`` and exit immediately (the grid is a host-side bound).
    """
    offs = tl.arange(0, EPOW)
    for base in range(0, MAXB, EPOW):
        tl.store(bnrow_ptr + base + offs, tl.zeros((EPOW,), tl.int32),
                 mask=base + offs < MAXB)
    valid = offs < NSLOT
    cnt = tl.load(cnt_ptr + offs, mask=valid, other=0)
    nblk = (cnt + (BM - 1)) // BM
    start = tl.cumsum(nblk, 0) - nblk
    for j in range(tl.max(nblk, 0)):
        pos = start + j
        live = valid & (j < nblk) & (pos < MAXB)
        tl.store(bslot_ptr + pos, offs, mask=live)
        tl.store(brow_ptr + pos, offs * C + j * BM, mask=live)
        tl.store(bnrow_ptr + pos, tl.minimum(cnt - j * BM, BM), mask=live)


@triton.jit
def _tile_of_pid(pid, nblk, NT: tl.constexpr, GM: tl.constexpr):
    """Map a flat CTA id to (row block, N tile), grouped ``GM`` blocks at a time.

    With the naive ``(block, n_tile)`` grid every concurrently resident CTA sits
    at the same N tile but a different row block, so the A tiles are re-read once
    per N tile -- at 16k tokens that is an extra ~6 GB for GEMM1 alone.  Walking
    ``GM`` row blocks across all N tiles instead keeps both the A rows and the
    expert's weight slice L2-resident (the same reason vLLM's fused-MoE kernel
    carries ``GROUP_SIZE_M``).
    """
    per_group = GM * NT
    gid = pid // per_group
    first = gid * GM
    gsize = min(nblk - first, GM)
    r = pid - gid * per_group
    return first + (r % gsize), r // gsize


@triton.jit
def _gemm1_kernel(
    x_ptr, w13_ptr, inter_ptr, bslot_ptr, brow_ptr, bnrow_ptr, stok_ptr, sdst_ptr,
    nblk,
    K: tl.constexpr, I: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, BLK: tl.constexpr,
    GM: tl.constexpr, WS: tl.constexpr,
):
    """inter[dst, n] = silu(x @ w1[slot]^T) * (x @ w3[slot]^T), grouped by slot.

    ``w13`` is in block-major-K layout ``[slot, K/BK, 2I, BK]`` so each of the two
    weight tiles a step needs is one contiguous ``BN * BK`` run.
    """
    blk, nt = _tile_of_pid(tl.program_id(0), nblk, I // BN, GM)
    nrow = tl.load(bnrow_ptr + blk)
    if nrow == 0:
        return
    slot = tl.load(bslot_ptr + blk)
    row0 = tl.load(brow_ptr + blk)
    roff = tl.arange(0, BM)
    rmask = roff < nrow
    tok = tl.load(stok_ptr + row0 + roff, mask=rmask, other=0)
    dst = tl.load(sdst_ptr + row0 + roff, mask=rmask, other=0)

    offs_n = nt * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_base = x_ptr + tok[:, None] * K + offs_k[None, :]
    g_base = (w13_ptr + slot.to(tl.int64) * (2 * I * K)
              + offs_n[None, :] * BLK + offs_k[:, None])
    u_base = g_base + I * BLK

    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    acc_u = tl.zeros((BM, BN), dtype=tl.float32)
    SUB: tl.constexpr = BLK // BK
    for i in tl.range(0, K // BK, warp_specialize=WS):
        koff = (i // SUB) * BLK + (i % SUB) * BK
        woff = (i // SUB) * (2 * I * BLK) + (i % SUB) * BK
        a = tl.load(a_base + koff, mask=rmask[:, None], other=0.0)
        acc_g = tl.dot(a, tl.load(g_base + woff), acc_g)
        acc_u = tl.dot(a, tl.load(u_base + woff), acc_u)

    out = acc_g * tl.sigmoid(acc_g) * acc_u
    tl.store(inter_ptr + dst[:, None] * I + offs_n[None, :],
             out.to(tl.bfloat16), mask=rmask[:, None])


@triton.jit
def _gemm2_kernel(
    inter_ptr, w2_ptr, opair_ptr, bslot_ptr, brow_ptr, bnrow_ptr, sw_ptr, sdst_ptr,
    nblk,
    H: tl.constexpr, I: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, BLK: tl.constexpr,
    GM: tl.constexpr, WS: tl.constexpr,
):
    """opair[dst, h] = routing_weight * (inter[dst] @ w2[slot]^T).

    ``w2`` is block-major-K: ``[slot, I/BK, H, BK]``.
    """
    blk, nt = _tile_of_pid(tl.program_id(0), nblk, H // BN, GM)
    nrow = tl.load(bnrow_ptr + blk)
    if nrow == 0:
        return
    slot = tl.load(bslot_ptr + blk)
    row0 = tl.load(brow_ptr + blk)
    roff = tl.arange(0, BM)
    rmask = roff < nrow
    dst = tl.load(sdst_ptr + row0 + roff, mask=rmask, other=0)
    rw = tl.load(sw_ptr + row0 + roff, mask=rmask, other=0.0)

    offs_n = nt * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_base = inter_ptr + dst[:, None] * I + offs_k[None, :]
    b_base = (w2_ptr + slot.to(tl.int64) * (H * I)
              + offs_n[None, :] * BLK + offs_k[:, None])

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    SUB: tl.constexpr = BLK // BK
    for i in tl.range(0, I // BK, warp_specialize=WS):
        koff = (i // SUB) * BLK + (i % SUB) * BK
        woff = (i // SUB) * (H * BLK) + (i % SUB) * BK
        a = tl.load(a_base + koff, mask=rmask[:, None], other=0.0)
        acc = tl.dot(a, tl.load(b_base + woff), acc)

    acc = acc * rw[:, None]
    tl.store(opair_ptr + dst[:, None] * H + offs_n[None, :],
             acc.to(tl.bfloat16), mask=rmask[:, None])


@triton.jit
def _reduce_kernel(
    opair_ptr, out_ptr, H: tl.constexpr, NP: tl.constexpr, BLK: tl.constexpr,
):
    """out[t] = sum over the NP = top_k + 1 expert contributions of token t."""
    t = tl.program_id(0)
    offs = tl.program_id(1) * BLK + tl.arange(0, BLK)
    mask = offs < H
    base = opair_ptr + t * (NP * H) + offs
    acc = tl.zeros((BLK,), dtype=tl.float32)
    for j in range(NP):
        acc += tl.load(base + j * H, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + t * H + offs, acc.to(tl.bfloat16), mask=mask)


# ---------------------------------------------------------------------------
# Tile selection + scratch buffers
# ---------------------------------------------------------------------------
# Per token-count regime, measured on B200 (the routing decides how many CTAs
# there are, so every column moves with M):
#   bm            rows per GEMM block -- also how often a slot's weights are
#                 re-read, so it grows with the rows a slot actually gets
#   bn/bk/st/nw   GEMM tile, k-step, pipeline depth, warps (1 = GEMM1, 2 = GEMM2)
#   nwr           warps for the router's top-k reductions
#   gm            row blocks grouped per N sweep (see _tile_of_pid)
#   ws            automatic warp specialization on the k loop (slower here)
#   lb*/lsp       logits-GEMM tile and K-split
_CFGS = (
    #  M<=      bm  bn1 bk1 st1 nw1  bn2 bk2 st2 nw2  nwr  gm  ws   lbm lbn lbk lsp
    (16,        16,  64,128,  3,  4,  64,128,  3,  4,   4,  1,  0,   16,  32, 128, 6),
    (32,        16,  64,128,  2,  4, 128,128,  3,  4,   4,  8,  0,   16,  32, 128, 6),
    (128,       16,  64, 32,  3,  4, 128, 32,  3,  4,   8,  1,  0,   16,  32, 128, 3),
    (1024,      32,  64,128,  2,  4, 128, 64,  2,  4,   4,  2,  0,   64,  64, 128, 3),
    (8192,     128, 128, 64,  3,  8, 256, 64,  3,  8,   4,  8,  0,  128, 256,  64, 1),
    (1 << 30,  256, 128, 64,  3,  8, 256, 64,  3,  8,   4,  8,  0,  128, 256,  64, 1),
)


# Feasible ``num_stages`` per (kernel, tile): Triton's shared-memory accounting
# for a pipelined tile pair is not worth predicting from the tile bytes, so the
# first launch of a shape walks the pipeline depth down until it fits and the
# answer is cached (resolved before any graph capture, which cannot tolerate the
# retry).
_STAGES: dict = {}


def _launch(kernel, grid, key, stages, *args, **kwargs):
    """Launch *kernel*, backing off ``num_stages`` until it fits in shared memory."""
    st = _STAGES.get(key, stages)
    while True:
        try:
            kernel[grid](*args, num_stages=st, **kwargs)
        except triton.runtime.errors.OutOfResources:
            if st <= 1:
                raise
            st -= 1
            continue
        _STAGES[key] = st
        return


# Dev hook: dict of {M: cfg tuple} to override the table above.
_CFG_OVERRIDE: dict | None = None


def _cfg(M: int, nslot: int, topk: int):
    """Tile shapes per token count, plus the host-side block-count bound.

    The block count is a *bound*, not the real count: it depends on the routing,
    which only exists on the device, and the GEMM grids are sized from it (tail
    blocks see ``rows == 0`` and exit).
    """
    table = _CFGS if _CFG_OVERRIDE is None else (
        (M, *_CFG_OVERRIDE[M]),)
    for lim, *cfg in table:
        if M <= lim:
            bm = cfg[0]
            # sum(ceil(cnt/BM)) <= touched slots + total rows / BM
            blocks = min(nslot, topk * M + 1) + ((topk + 1) * M) // bm
            return (*cfg, blocks)
    raise AssertionError


class _Scratch:
    """One set of per-shape scratch buffers, shared by every KimiMoE layer."""

    __slots__ = ("logits", "cnt", "stok", "sw", "sdst", "inter", "opair", "out",
                 "bslot", "brow", "bnrow", "blocks")


# Keyed by shape: a CUDA-graph-captured shape's buffers must stay alive for the
# lifetime of the graph, so those keys are pinned against recycling.
_SCRATCH: dict = {}
_PINNED: set = set()
_SCRATCH_MAX = 8

# Token counts at or below this are captured into a CUDA graph: the pipeline is
# six kernels over ~40 us of GPU work, which the host cannot issue that fast.
_GRAPH_MAX = int(os.environ.get("FK_KIMI_MOE_GRAPH_MAX", "64"))


# ---------------------------------------------------------------------------
class KimiMoE(nn.Module):
    def __init__(self, config, quant_config: dict | None = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.num_shared_experts = config.num_shared_experts
        self.num_expert_group = config.num_expert_group
        self.topk_group = config.topk_group
        self.routed_scaling_factor = config.routed_scaling_factor
        self.tp_size = _tp_size()
        self.intermediate_per_tp = config.moe_intermediate_size // self.tp_size

        self.gate = ReplicatedLinear(
            self.hidden_size,
            self.num_experts,
            bias=False,
            quant_config=None,
        )
        # Model dtype, not FP32: vLLM's KimiMoE declares this as
        # ``nn.Parameter(torch.empty(num_experts))`` under the model-dtype
        # default, unlike DeepSeek-V3 whose router bias really is FP32.
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(self.num_experts),
        )
        self.gate.e_score_correction_bias.weight_loader = (
            lambda p, w: p.data.copy_(w.to(p.dtype))
        )

        self.w13 = nn.Parameter(
            torch.empty(
                config.num_experts,
                2 * self.intermediate_per_tp,
                config.hidden_size,
            ),
        )
        self.w13.weight_loader = self._w13_weight_loader
        self.w2 = nn.Parameter(
            torch.empty(
                config.num_experts,
                config.hidden_size,
                self.intermediate_per_tp,
            ),
        )
        self.w2.weight_loader = self._w2_weight_loader
        self.shared_experts = (
            LlamaMLP(
                config,
                quant_config=quant_config,
                intermediate_size=config.moe_intermediate_size * self.num_shared_experts,
                reduce_results=False,
            )
            if self.num_shared_experts
            else None
        )
        self.allreduce = AllReduce()

        self.scoring_func = config.moe_router_activation_func
        self.renormalize = config.moe_renormalize
        # The shared expert becomes slot ``num_experts`` of the fused problem
        # when it has exactly one expert's shapes (Kimi: one shared expert of
        # moe_intermediate_size).
        self.fuse_shared = (
            self.shared_experts is not None and self.num_shared_experts == 1
        )
        self.fused_ok = (
            self.scoring_func == "sigmoid"
            and self.renormalize
            and self.num_expert_group in (None, 1)
            and self.hidden_size % _BLK == 0
            and self.intermediate_per_tp % _BLK == 0
            and self.num_experts + 1 <= _EPOW
        )
        self.num_slots = self.num_experts + (1 if self.fuse_shared else 0)
        self._weights_ready = False
        self._graphs: dict | None = {}
        self._scratch_key = None
        self._w13f: torch.Tensor | None = None
        self._w2f: torch.Tensor | None = None

        # Custom-op dispatch for torch.compile (flipped by enable_custom_ops
        # once the model is wrapped with torch.compile). ``_layer_name`` is
        # populated by auto_register_no_compile_layers.
        self._use_custom_op = False
        self._layer_name = ""

    def _w13_weight_loader(self, param, loaded_weight, expert_id: int, is_w1: bool):
        n = self.intermediate_per_tp
        shard = loaded_weight.narrow(0, _tp_rank() * n, n)
        offset = 0 if is_w1 else n
        param.data[expert_id, offset:offset + n, :].copy_(shard)

    def _w2_weight_loader(self, param, loaded_weight, expert_id: int):
        n = self.intermediate_per_tp
        param.data[expert_id].copy_(loaded_weight.narrow(1, _tp_rank() * n, n))

    def process_weights_after_loading(self) -> None:
        """Build the fused, block-major-K expert weights.

        Slot ``num_experts`` is the shared expert, whose ``gate_up_proj`` /
        ``down_proj`` already have the per-expert shapes ([2I, H] with the gate
        half first, and [H, I]), so it joins the grouped problem as one more
        expert.  ``[slots, rows, K]`` then becomes ``[slots, K/128, rows, 128]``:
        a GEMM step's weight tile is one contiguous ``BN * BK`` run instead of
        ``BN`` runs strided by ``K``.
        """
        if self._weights_ready:
            return
        self._weights_ready = True
        if not self.fused_ok:  # the eager fallback reads the checkpoint layout
            return
        w13, w2 = self.w13.data, self.w2.data
        if self.fuse_shared:
            sw13 = self.shared_experts.gate_up_proj.weight.data
            sw2 = self.shared_experts.down_proj.weight.data
            w13 = torch.cat([w13, sw13.reshape(1, *w13.shape[1:])], dim=0)
            w2 = torch.cat([w2, sw2.reshape(1, *w2.shape[1:])], dim=0)
        nb, n2, k = w13.shape
        self._w13f = (w13.view(nb, n2, k // _BLK, _BLK)
                      .permute(0, 2, 1, 3).contiguous())
        nb, h, i = w2.shape
        self._w2f = (w2.view(nb, h, i // _BLK, _BLK)
                     .permute(0, 2, 1, 3).contiguous())
        # The checkpoint-layout copies are dead once the fused ones exist.
        self.w13 = nn.Parameter(w13.new_empty(0), requires_grad=False)
        self.w2 = nn.Parameter(w2.new_empty(0), requires_grad=False)

    # -- scratch ------------------------------------------------------------
    def _scratch(self, M: int, bm: int, split: int, blocks: int, device):
        nslot = self.num_slots
        topk = self.top_k
        C = ((M + bm - 1) // bm) * bm
        key = (M, C, nslot, topk, split, blocks, str(device))
        self._scratch_key = key
        sb = _SCRATCH.get(key)
        if sb is not None:
            return sb, C
        if len(_SCRATCH) >= _SCRATCH_MAX:
            for k in list(_SCRATCH):
                if k not in _PINNED:
                    del _SCRATCH[k]
                    break
        sb = _Scratch()
        i32 = dict(dtype=torch.int32, device=device)
        f32 = dict(dtype=torch.float32, device=device)
        bf16 = dict(dtype=torch.bfloat16, device=device)
        npair = (topk + 1) * M
        sb.logits = torch.empty(split, M, self.num_experts, **f32)
        sb.cnt = torch.zeros(_EPOW, **i32)
        sb.stok = torch.empty(nslot * C, **i32)
        sb.sw = torch.empty(nslot * C, **f32)
        sb.sdst = torch.empty(nslot * C, **i32)
        sb.inter = torch.empty(npair, self.intermediate_per_tp, **bf16)
        sb.opair = torch.empty(npair, self.hidden_size, **bf16)
        sb.out = torch.empty(M, self.hidden_size, **bf16)
        sb.bslot = torch.empty(blocks, **i32)
        sb.brow = torch.empty(blocks, **i32)
        sb.bnrow = torch.zeros(blocks, **i32)
        sb.blocks = blocks
        _SCRATCH[key] = sb
        return sb, C

    # -- forward ------------------------------------------------------------
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._use_custom_op:
            # The all-reduce stays *outside* the opaque op: inside it Inductor
            # cannot see the collective, so ``AllReduceFusedAddRMSNormPass`` has
            # nothing to match at the MoE end of the layer -- half of every
            # layer's collectives.
            out = torch.ops.fastkernels.moe_forward(hidden_states, self._layer_name)
            if self.tp_size > 1:
                out = self.allreduce(out)
            return out
        return self.forward_impl(hidden_states)

    def forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        x = hidden_states.view(-1, self.hidden_size)
        M = x.shape[0]
        if not self._weights_ready:
            self.process_weights_after_loading()

        if not self.fused_ok:
            out = self._moe_eager(x)
        elif M <= _GRAPH_MAX and self._graphs is not None:
            out = self._moe_graphed(x.contiguous(), M)
        else:
            out = self._moe(x.contiguous(), M)

        if self.shared_experts is not None and not self.fuse_shared:
            out = out + self.shared_experts(x)
        if self.tp_size > 1 and not self._use_custom_op:
            out = self.allreduce(out)
        return out.view(orig_shape)

    def _moe_graphed(self, x: torch.Tensor, M: int) -> torch.Tensor:
        """Replay the whole six-kernel pipeline from a CUDA graph.

        At one to a few dozen tokens the pipeline is ~40 us of GPU work and the
        host cannot issue six Triton launches that fast; the graph turns the call
        into one copy plus one replay.  Routing still happens on the GPU from the
        live input -- only the shape and the buffer addresses are captured.
        """
        entry = self._graphs.get(M)
        if entry is None:
            try:
                entry = self._capture(x, M)
            except Exception:
                self._graphs = None  # capture unavailable: stay in eager launches
                return self._moe(x, M)
            self._graphs[M] = entry
        graph, sx, out = entry
        sx.copy_(x)
        graph.replay()
        return out

    def _capture(self, x: torch.Tensor, M: int):
        sx = x.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._moe(sx, M)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = self._moe(sx, M)
        _PINNED.add(self._scratch_key)
        return graph, sx, out

    def _moe_eager(self, x: torch.Tensor) -> torch.Tensor:
        """Reference path for routing/shape combinations the kernels do not cover
        (non-sigmoid scoring, no renormalize, real expert groups, odd sizes)."""
        logits = (x.to(torch.float32) @ self.gate.weight.t().to(torch.float32))
        scores = torch.sigmoid(logits) if self.scoring_func == "sigmoid" else \
            torch.softmax(logits, dim=-1)
        biased = scores + self.gate.e_score_correction_bias.float()
        g = self.num_expert_group or 1
        if g > 1:
            gs = biased.view(x.shape[0], g, -1).topk(2, dim=-1)[0].sum(-1)
            keep = torch.zeros_like(gs)
            keep.scatter_(1, gs.topk(self.topk_group, dim=-1)[1], 1)
            biased = biased.masked_fill(
                ~keep[..., None].expand(-1, -1, self.num_experts // g)
                .reshape(x.shape[0], -1).bool(), float("-inf"))
        ids = biased.topk(self.top_k, dim=-1)[1]
        w = scores.gather(1, ids)
        if self.renormalize:
            w = w / w.sum(-1, keepdim=True)
        w = w * self.routed_scaling_factor
        out = torch.zeros_like(x, dtype=torch.float32)
        for e in range(self.num_experts):
            hit = ids == e
            if not bool(hit.any()):
                continue
            rows = hit.any(-1).nonzero(as_tuple=True)[0]
            wt = (w * hit).sum(-1)[rows]
            h = x[rows] @ self.w13[e].t()
            d = h.shape[-1] // 2
            y = (torch.nn.functional.silu(h[:, :d]) * h[:, d:]) @ self.w2[e].t()
            out.index_add_(0, rows, y.float() * wt[:, None])
        if self.fuse_shared:
            out = out + self.shared_experts(x).float()
        return out.to(x.dtype)

    def _moe(self, x: torch.Tensor, M: int) -> torch.Tensor:
        H = self.hidden_size
        I = self.intermediate_per_tp
        NE = self.num_experts
        nslot = self.num_slots
        topk = self.top_k
        (bm, bn1, bk1, st1, nw1, bn2, bk2, st2, nw2, nwr, gm, ws,
         lbm, lbn, lbk, lsplit, blocks) = _cfg(M, nslot, topk)
        sb, C = self._scratch(M, bm, lsplit, blocks, x.device)

        # 1) router logits (fp32, split-K partials) + row-counter clear
        _launch(
            _logits_kernel, (triton.cdiv(M, lbm), NE // lbn, lsplit),
            (0, lbm, lbn, lbk), 3,
            x, self.gate.weight, sb.logits, sb.cnt, M,
            K=H, NE=NE, BM=lbm, BN=lbn, BK=lbk, KSPLIT=H // lsplit, CNT_LEN=_EPOW,
            num_warps=4,
        )
        # 2) top-k + renormalize + scatter (token, slot) pairs
        _route_kernel[(M,)](
            sb.logits, self.gate.e_score_correction_bias, sb.cnt,
            sb.stok, sb.sw, sb.sdst, M,
            NE=NE, EPOW=triton.next_power_of_2(NE), TOPK=topk, NSLOT=nslot,
            C=C, SCALE=float(self.routed_scaling_factor),
            SHARED=self.fuse_shared, SPLIT=lsplit, num_warps=nwr,
        )
        # 3) block -> slot map for the two grouped GEMMs
        _blockmap_kernel[(1,)](
            sb.cnt, sb.bslot, sb.brow, sb.bnrow,
            NSLOT=nslot, EPOW=_EPOW, C=C, BM=bm, MAXB=blocks, num_warps=4,
        )
        # 4) grouped GEMM 1 + fused SwiGLU
        _launch(
            _gemm1_kernel, (blocks * (I // bn1),), (1, bm, bn1, bk1, nw1, gm, ws), st1,
            x, self._w13f, sb.inter, sb.bslot, sb.brow, sb.bnrow, sb.stok, sb.sdst,
            blocks,
            K=H, I=I, BM=bm, BN=bn1, BK=bk1, BLK=_BLK, GM=gm, WS=ws, num_warps=nw1,
        )
        # 5) grouped GEMM 2, routing weight in the epilogue
        _launch(
            _gemm2_kernel, (blocks * (H // bn2),), (2, bm, bn2, bk2, nw2, gm, ws), st2,
            sb.inter, self._w2f, sb.opair, sb.bslot, sb.brow, sb.bnrow, sb.sw, sb.sdst,
            blocks,
            H=H, I=I, BM=bm, BN=bn2, BK=bk2, BLK=_BLK, GM=gm, WS=ws, num_warps=nw2,
        )
        # 6) reduce the top_k + 1 contributions of each token
        _reduce_kernel[(M, triton.cdiv(H, 512))](
            sb.opair, sb.out, H=H, NP=topk + 1, BLK=512, num_warps=4,
        )
        return sb.out
