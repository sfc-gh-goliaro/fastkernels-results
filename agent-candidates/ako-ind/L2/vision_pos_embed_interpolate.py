"""Bilinear interpolation of learned 2D position embeddings (Qwen3-VL).

Owns a learned embedding weight of (num_grid_per_side^2, hidden_size).
forward() interpolates these onto arbitrary (h, w) grids using bilinear
weights, then reshuffles by spatial_merge_size for the vision encoder.

Fused implementation.  The whole ``grid_thw_list`` is produced by one Triton
launch writing directly into a single preallocated output buffer, so the
per-grid Python loop (~20 tiny kernels each) and the final ``torch.cat`` are
gone, and the 4 x N x hidden gathered/weighted intermediates never exist.

Two shape-keyed caches do the bookkeeping, both pure functions of the
*shape* arguments (never of the embedding data):

* an arena of per-``(h, w)`` gather tables -- the four source row indices and
  the four bilinear weights for every post-spatial-merge output position.
  They are built with the reference formulas themselves, so the fused result
  is bit-identical to the baseline.
* a per-``grid_thw_list`` plan.  Grids that share ``(h, w)`` produce identical
  rows, so each distinct ``(h, w)`` is interpolated once and the result is
  stored to every destination block it feeds (the extra video frames of a
  ``t > 1`` grid are just more destinations).  In the captured video batches
  that folds 16 destination blocks onto one gather.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.embedding import Embedding


@triton.jit
def _store_at(base_ptr, row_ptr, blk, HID: tl.constexpr, val, mask):
    tl.store(row_ptr + base_ptr * HID, val, mask=mask & blk)


@triton.jit
def _interp_kernel(
    out_ptr, emb_ptr, meta_ptr, dst_ptr, idx_ptr, wt_ptr, tstride,
    HID: tl.constexpr,      # hidden_size
    BLOCK_R: tl.constexpr,  # output rows per program
    BLOCK_D: tl.constexpr,  # hidden columns per store
    CHUNKS: tl.constexpr,   # BLOCK_D chunks walked per program
    DMAX: tl.constexpr,     # destination blocks per program (power of two)
    PCAST: tl.constexpr,    # 1 bf16 / 2 fp16: round each product like the ref
):
    pid_d = tl.program_id(0)
    pid_r = tl.program_id(1)
    task = tl.program_id(2)

    mb = meta_ptr + task * 4
    toff = tl.load(mb + 0)      # this (h, w) block's row offset in the arena
    doff = tl.load(mb + 1)      # first destination slot
    dcnt = tl.load(mb + 2)      # number of live destinations (<= DMAX)
    nrows = tl.load(mb + 3)     # h * w

    # Destination bases up front: DMAX independent scalar loads (no chain).
    b0 = tl.load(dst_ptr + doff + 0).to(tl.int64)
    b1 = b0; b2 = b0; b3 = b0; b4 = b0; b5 = b0; b6 = b0; b7 = b0
    b8 = b0; b9 = b0; b10 = b0; b11 = b0; b12 = b0; b13 = b0; b14 = b0; b15 = b0
    if DMAX > 1:
        b1 = tl.load(dst_ptr + doff + 1).to(tl.int64)
    if DMAX > 2:
        b2 = tl.load(dst_ptr + doff + 2).to(tl.int64)
        b3 = tl.load(dst_ptr + doff + 3).to(tl.int64)
    if DMAX > 4:
        b4 = tl.load(dst_ptr + doff + 4).to(tl.int64)
        b5 = tl.load(dst_ptr + doff + 5).to(tl.int64)
        b6 = tl.load(dst_ptr + doff + 6).to(tl.int64)
        b7 = tl.load(dst_ptr + doff + 7).to(tl.int64)
    if DMAX > 8:
        b8 = tl.load(dst_ptr + doff + 8).to(tl.int64)
        b9 = tl.load(dst_ptr + doff + 9).to(tl.int64)
        b10 = tl.load(dst_ptr + doff + 10).to(tl.int64)
        b11 = tl.load(dst_ptr + doff + 11).to(tl.int64)
        b12 = tl.load(dst_ptr + doff + 12).to(tl.int64)
        b13 = tl.load(dst_ptr + doff + 13).to(tl.int64)
        b14 = tl.load(dst_ptr + doff + 14).to(tl.int64)
        b15 = tl.load(dst_ptr + doff + 15).to(tl.int64)

    r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    rmask = r < nrows
    q = toff + r
    i0 = tl.load(idx_ptr + q, mask=rmask, other=0)
    i1 = tl.load(idx_ptr + tstride + q, mask=rmask, other=0)
    i2 = tl.load(idx_ptr + 2 * tstride + q, mask=rmask, other=0)
    i3 = tl.load(idx_ptr + 3 * tstride + q, mask=rmask, other=0)
    w0 = tl.load(wt_ptr + q, mask=rmask, other=0).to(tl.float32)[:, None]
    w1 = tl.load(wt_ptr + tstride + q, mask=rmask, other=0).to(tl.float32)[:, None]
    w2 = tl.load(wt_ptr + 2 * tstride + q, mask=rmask, other=0).to(tl.float32)[:, None]
    w3 = tl.load(wt_ptr + 3 * tstride + q, mask=rmask, other=0).to(tl.float32)[:, None]

    e0 = emb_ptr + i0[:, None] * HID
    e1 = emb_ptr + i1[:, None] * HID
    e2 = emb_ptr + i2[:, None] * HID
    e3 = emb_ptr + i3[:, None] * HID
    ri = r.to(tl.int64)[:, None] * HID

    dbase = pid_d * (CHUNKS * BLOCK_D)
    for c in tl.static_range(CHUNKS):
        d = dbase + c * BLOCK_D + tl.arange(0, BLOCK_D)
        m2 = rmask[:, None] & (d < HID)[None, :]
        dd = d[None, :]
        p0 = tl.load(e0 + dd, mask=m2, other=0.0).to(tl.float32) * w0
        p1 = tl.load(e1 + dd, mask=m2, other=0.0).to(tl.float32) * w1
        p2 = tl.load(e2 + dd, mask=m2, other=0.0).to(tl.float32) * w2
        p3 = tl.load(e3 + dd, mask=m2, other=0.0).to(tl.float32) * w3
        # The reference rounds emb * weight to the output dtype before the
        # fp32 accumulation of `.sum(0)`; mirror that so results match bitwise.
        if PCAST == 1:
            p0 = p0.to(tl.bfloat16).to(tl.float32)
            p1 = p1.to(tl.bfloat16).to(tl.float32)
            p2 = p2.to(tl.bfloat16).to(tl.float32)
            p3 = p3.to(tl.bfloat16).to(tl.float32)
        elif PCAST == 2:
            p0 = p0.to(tl.float16).to(tl.float32)
            p1 = p1.to(tl.float16).to(tl.float32)
            p2 = p2.to(tl.float16).to(tl.float32)
            p3 = p3.to(tl.float16).to(tl.float32)
        av = (((p0 + p1) + p2) + p3).to(out_ptr.dtype.element_ty)

        rp = out_ptr + ri + dd
        tl.store(rp + b0 * HID, av, mask=m2)
        if DMAX > 1:
            _store_at(b1, rp, 1 < dcnt, HID, av, m2)
        if DMAX > 2:
            _store_at(b2, rp, 2 < dcnt, HID, av, m2)
            _store_at(b3, rp, 3 < dcnt, HID, av, m2)
        if DMAX > 4:
            _store_at(b4, rp, 4 < dcnt, HID, av, m2)
            _store_at(b5, rp, 5 < dcnt, HID, av, m2)
            _store_at(b6, rp, 6 < dcnt, HID, av, m2)
            _store_at(b7, rp, 7 < dcnt, HID, av, m2)
        if DMAX > 8:
            _store_at(b8, rp, 8 < dcnt, HID, av, m2)
            _store_at(b9, rp, 9 < dcnt, HID, av, m2)
            _store_at(b10, rp, 10 < dcnt, HID, av, m2)
            _store_at(b11, rp, 11 < dcnt, HID, av, m2)
            _store_at(b12, rp, 12 < dcnt, HID, av, m2)
            _store_at(b13, rp, 13 < dcnt, HID, av, m2)
            _store_at(b14, rp, 14 < dcnt, HID, av, m2)
            _store_at(b15, rp, 15 < dcnt, HID, av, m2)


class _TableArena:
    """Append-only (4, T) gather-index / bilinear-weight planes, one row block
    per distinct (h, w).  Built from the reference formulas, so the fused
    kernel reproduces the baseline bit for bit."""

    PAD = 128   # keep T's 16-divisibility class constant across regrowth

    def __init__(self, num_grid: int, merge: int, dtype: torch.dtype,
                 device: torch.device):
        self.ng = num_grid
        self.m = merge
        self.dtype = dtype
        self.device = device
        self.offset: dict[tuple[int, int], int] = {}
        self._parts_i: list[torch.Tensor] = []
        self._parts_w: list[torch.Tensor] = []
        self._rows = 0
        self.idx = torch.zeros((4, self.PAD), dtype=torch.int32, device=device)
        self.wt = torch.zeros((4, self.PAD), dtype=dtype, device=device)
        self.stride = self.PAD

    def _block(self, h: int, w: int):
        ng, m, dev = self.ng, self.m, self.device
        h_idxs = torch.linspace(0, ng - 1, h, dtype=torch.float32, device=dev)
        w_idxs = torch.linspace(0, ng - 1, w, dtype=torch.float32, device=dev)
        h_floor = h_idxs.long()
        w_floor = w_idxs.long()
        h_ceil = torch.clamp(h_floor + 1, max=ng - 1)
        w_ceil = torch.clamp(w_floor + 1, max=ng - 1)
        dh = h_idxs - h_floor
        dw = w_idxs - w_floor
        dh_g, dw_g = torch.meshgrid(dh, dw, indexing="ij")
        hf_g, wf_g = torch.meshgrid(h_floor, w_floor, indexing="ij")
        hc_g, wc_g = torch.meshgrid(h_ceil, w_ceil, indexing="ij")
        w11 = dh_g * dw_g
        w10 = dh_g - w11
        w01 = dw_g - w11
        w00 = 1 - dh_g - w01
        h_grid = torch.stack([hf_g, hf_g, hc_g, hc_g])
        w_grid = torch.stack([wf_g, wc_g, wf_g, wc_g])
        idx = (h_grid * ng + w_grid).reshape(4, -1)
        wt = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1).to(self.dtype)
        # rows in post-spatial-merge order: reshape(h/m, m, w/m, m).permute(0,2,1,3)
        perm = torch.arange(h * w, device=dev).reshape(
            h // m, m, w // m, m).permute(0, 2, 1, 3).reshape(-1)
        return idx[:, perm].to(torch.int32), wt[:, perm]

    def offset_of(self, h: int, w: int) -> int:
        off = self.offset.get((h, w))
        if off is not None:
            return off
        idx, wt = self._block(h, w)
        off = self._rows
        self.offset[(h, w)] = off
        self._parts_i.append(idx)
        self._parts_w.append(wt)
        self._rows += h * w
        pad = -self._rows % self.PAD
        parts_i = list(self._parts_i)
        parts_w = list(self._parts_w)
        if pad:
            parts_i.append(torch.zeros((4, pad), dtype=torch.int32, device=self.device))
            parts_w.append(torch.zeros((4, pad), dtype=self.dtype, device=self.device))
        self.idx = torch.cat(parts_i, dim=1).contiguous()
        self.wt = torch.cat(parts_w, dim=1).contiguous()
        self.stride = self.idx.shape[1]
        return off


_DT_CODE = {torch.bfloat16: 1, torch.float16: 2}
_PLAN_CACHE_LIMIT = 4096


class VisionPosEmbedInterpolate(nn.Module):
    # kernel tiling (swept on B200 / Triton 3.6)
    BLOCK_R = 8
    BLOCK_D = 128
    CHUNKS = 1
    NUM_WARPS = 4
    NUM_STAGES = 1
    DMAX_CAP = 16

    def __init__(self, num_position_embeddings: int, hidden_size: int,
                 spatial_merge_size: int):
        super().__init__()
        self._embed = Embedding(num_position_embeddings, hidden_size)
        self.num_grid_per_side = int(num_position_embeddings ** 0.5)
        self.spatial_merge_size = spatial_merge_size
        self.hidden_size = hidden_size
        self._arenas: dict = {}
        self._plans: dict = {}

    def _weight(self) -> torch.Tensor:
        emb = self._embed
        w = getattr(emb, "weight", None)
        return emb.emb.weight if w is None else w

    def _arena(self, dtype, device) -> _TableArena:
        key = (device, dtype)
        arena = self._arenas.get(key)
        if arena is None:
            arena = _TableArena(self.num_grid_per_side, self.spatial_merge_size,
                                dtype, device)
            self._arenas[key] = arena
        return arena

    def _plan(self, grid_thw_list, dtype, device):
        key = (device, dtype, tuple(map(tuple, grid_thw_list)))
        plan = self._plans.get(key)
        if plan is not None:
            return plan
        arena = self._arena(dtype, device)
        # One entry per distinct (h, w); every (grid, frame) that shares it is
        # a destination block for the same interpolated rows.
        groups: dict[tuple[int, int], list[int]] = {}
        order: list[tuple[int, int]] = []
        total = 0
        for t, h, w in grid_thw_list:
            t, h, w = int(t), int(h), int(w)
            hw = h * w
            dsts = groups.get((h, w))
            if dsts is None:
                dsts = groups[(h, w)] = []
                order.append((h, w))
            for f in range(t):
                dsts.append(total + f * hw)
            total += t * hw
        widest = max(len(groups[k]) for k in order)
        dmax = 1
        while dmax < widest and dmax < self.DMAX_CAP:
            dmax *= 2
        meta: list[int] = []
        flat: list[int] = []
        max_rows = 0
        for h, w in order:
            toff = arena.offset_of(h, w)
            dsts = groups[(h, w)]
            hw = h * w
            for s in range(0, len(dsts), dmax):
                blk = dsts[s:s + dmax]
                meta.extend((toff, len(flat), len(blk), hw))
                flat.extend(blk)
                flat.extend([blk[0]] * (dmax - len(blk)))
            if hw > max_rows:
                max_rows = hw
        plan = (torch.tensor(meta, dtype=torch.int32, device=device),
                torch.tensor(flat, dtype=torch.int32, device=device),
                total, max_rows, len(meta) // 4, dmax, arena)
        if len(self._plans) >= _PLAN_CACHE_LIMIT:
            self._plans.clear()
        self._plans[key] = plan
        return plan

    def forward(
        self,
        grid_thw_list: list[list[int]],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        weight = self._weight()
        hidden_dim = self.hidden_size
        out_dtype = torch.promote_types(weight.dtype, dtype)
        if not grid_thw_list:
            return torch.empty((0, hidden_dim), dtype=out_dtype, device=device)

        meta, dsts, total, max_rows, n_tasks, dmax, arena = self._plan(
            grid_thw_list, dtype, device)
        out = torch.empty((total, hidden_dim), dtype=out_dtype, device=device)
        if total == 0:
            return out

        block_d = self.BLOCK_D
        chunks = self.CHUNKS
        _interp_kernel[(triton.cdiv(hidden_dim, block_d * chunks),
                        triton.cdiv(max_rows, self.BLOCK_R),
                        n_tasks)](
            out, weight, meta, dsts, arena.idx, arena.wt, arena.stride,
            HID=hidden_dim,
            BLOCK_R=self.BLOCK_R,
            BLOCK_D=block_d,
            CHUNKS=chunks,
            DMAX=dmax,
            PCAST=_DT_CODE.get(out_dtype, 0),
            num_warps=self.NUM_WARPS,
            num_stages=self.NUM_STAGES,
        )
        return out
