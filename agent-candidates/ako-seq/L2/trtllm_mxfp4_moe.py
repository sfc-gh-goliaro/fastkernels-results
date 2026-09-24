"""TRTLLM-gen MXFP4 fused MoE -- own fused kernel for the small/medium-token regime.

Baseline (``baseline.py``) forwards straight into
``flashinfer.trtllm_fp4_block_scale_moe``.  Three measurements decide the shape of
this kernel (details and method in ``ITERATIONS.md``):

* The flashinfer python wrapper costs **~470 us of host time per call** while the
  actual device work at M=1 is only **21.7 us** (CUDA-graph replay).
* ``fastkernels.bench._time_module`` copies every contiguous input tensor through
  a ``_ShiftingPool`` *inside* the timed region -- ~966 MB for this operator, a
  **330 us additive floor** on every case.  So the GPU is busy for most of the
  wrapper's host time, and only a small part of the 470 us is actually reclaimable.
* Measured against a zero-arithmetic kernel that just streams the weights an
  M-token call needs at ~5 TB/s, the per-shape ceilings are 1.415x (M=1),
  1.144x (26), 1.054x (60), 1.306x (398), 1.000x (16384).

Hence the dispatch: our kernels run only where they clear that bar today (M<=8,
measured crossover), and everything else keeps the flashinfer call so the
compute-bound 16384-token case and the near-roofline 26/60/398 cases cannot
regress.

The M<=8 path is three launches: one CTA for top-k + softmax + counting sort +
work-tile descriptors, then the two gemms as hand-written CUDA
(``mxfp4_cuda.cu``) -- Triton's ``dot_scaled`` spends more on the block-scale
operand than the entire weight stream costs, so the fp4 -> bf16 upcast is done by
hand at 2.1 ops/element and fed to ``mma.sync.m16n8k16``.  At these token counts
the launch is per-warp latency bound rather than bandwidth bound, so each 16-row
weight group is given to four warps that split its K range and meet in shared
memory.  Any problem building the extension falls back to the Triton gemms.

The weights arrive already in trtllm-gen's shuffled layout (the benchmark never
calls :func:`prepare_trtllm_mxfp4_weights`), so the kernels read them through
index math that inverts that layout:

* row shuffle ``shuffle_matrix_a`` with ``epilogue_tile_m=128`` -> block size 32:
  logical row ``n`` lives at physical row
  ``(n//32)*32 + ((n%32)%4)*8 + ((n%32)//4)``;
* block scales are swizzled 128x4:
  ``(((mi//128)*ceil(K/4) + kk//4)*512 + (mi%32)*16 + ((mi%128)//32)*4 + kk%4)``;
* scale bytes are UE8M0 (``2**(byte-127)``), which is exactly what
  ``tl.dot_scaled(..., "e2m1")`` consumes;
* logical row ``2j`` is the SwiGLU **up** row, ``2j+1`` is the **gate** row.

The precision model reproduces flashinfer to ``matched >= 0.9999``: routing
weights, the gemm1 output, the activation and the per-expert gemm2 output are all
rounded to bf16, everything else accumulates in fp32.  The bf16 roundings of the
routing weight and of the per-expert output are load-bearing -- the random
e4m3-derived scale bytes span 2**-127..2**72, so each dot product is dominated by
a single 32-element block and a 0.4% error there lands straight on the output.
"""

from __future__ import annotations

import hashlib
import os

import torch
import torch.nn as nn
import triton
import triton.language as tl

from flashinfer import trtllm_fp4_block_scale_moe


# ---------------------------------------------------------------------------
# Hand-written CUDA inner loops (``mxfp4_cuda.cu``).  Triton's ``dot_scaled``
# spends more time turning the block-scale operand into an mma layout than the
# whole weight stream costs (measured: +131 us on a 604 MB tile, see
# ITERATIONS.md), so the fp4 -> bf16 upcast is done by hand: a PRMT byte-LUT with
# the UE8M0 scale folded into the LUT, 2.1 ops/element, feeding
# mma.sync.m16n8k16.  Worth 50 us -> 28 us on the two M=1 gemms.
#
# The build is cached on disk under a content hash, so a bench run only pays a
# ninja no-op, never a compile inside a timed call.  Any failure here leaves
# ``_EXT`` None and the Triton path runs unchanged.
# ---------------------------------------------------------------------------
def _load_ext():
    if os.environ.get("FASTKERNELS_MXFP4_CUDA", "1") == "0":
        return None
    src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mxfp4_cuda.cu")
    try:
        with open(src_path) as fh:
            src = fh.read()
        from torch.utils.cpp_extension import load

        tag = hashlib.sha256(src.encode()).hexdigest()[:10]
        root = os.environ.get("FASTKERNELS_MXFP4_BUILD", "/tmp/ako_mxfp4_ext")
        bdir = os.path.join(root, tag)
        os.makedirs(bdir, exist_ok=True)
        cu = os.path.join(bdir, "mxfp4_cuda.cu")
        # Write only when the content differs: an unchanged mtime keeps ninja a no-op.
        if not os.path.exists(cu) or open(cu).read() != src:
            with open(cu, "w") as fh:
                fh.write(src)
        prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"
        try:
            return load(
                name="mxfp4_cuda_" + tag,
                sources=[cu],
                build_directory=bdir,
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_100,code=sm_100"],
                verbose=False,
            )
        finally:
            if prev is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = prev
    except Exception:  # noqa: BLE001  -- any build/toolchain problem: use Triton
        return None


_EXT = _load_ext()

# Row-tile granularities the CUDA kernels assume (see mxfp4_cuda.cu).
_CU_ROWS = 64
_CU_KMUL = 512


def _cuda_ok(m: int, hidden: int, inter: int, n2: int, ho: int, hu: int) -> bool:
    """Whether the CUDA gemms can serve this shape."""
    return (
        _EXT is not None
        and hidden % _CU_KMUL == 0
        and inter % _CU_KMUL == 0
        and n2 % _CU_ROWS == 0
        and ho % _CU_ROWS == 0
        and hu % _CU_ROWS == 0
    )


TRTLLM_MXFP4_ALIGN = 256
ROUTING_RENORMALIZE_NAIVE = 4

SWIGLU_ALPHA = 1.702
SWIGLU_BETA = 1.0
SWIGLU_LIMIT = 7.0

DEFAULT_TUNE_MAX_NUM_TOKENS = 1024

_MXFP4_SF_BLOCK = 32
_EPILOGUE_TILE_M = 128

# Tokens above this go back to flashinfer.  Measured crossover (see ITERATIONS.md):
# the fused path wins at M<=8 and loses above, because flashinfer's device time is
# already within 5-14% of the pure weight-streaming floor for M in [16, 400] while
# our bf16-MMA path pays a fp4->bf16 upcast the native fp4 MMA does not.
FUSED_MAX_TOKENS = int(os.environ.get("FASTKERNELS_MXFP4_FUSED_MAX", "8"))

_BM = 16  # tl.dot minimum M

# Above this many tokens the [TP, TP] rank matrix in _route_small stops fitting
# in registers and the routing falls back to torch.argsort + _build_tiles.
_ROUTE_MAX = 16

# (BN, BK, num_warps, num_stages) for gemm1 / gemm2, chosen by token count.
# Few active experts -> few work tiles, so shrink BN to get enough CTAs in
# flight to cover memory latency; many experts -> the wide tile wins.
# num_warps=8 is a 3-7x regression at BN>=128 here (the mma tile-picker moves).
_CFG_TINY = ((32, 256, 2, 3), (32, 256, 2, 2))
_CFG_WIDE = ((128, 128, 4, 3), (128, 128, 4, 3))


def _tile_config(m: int):
    return _CFG_TINY if m <= 8 else _CFG_WIDE


def trtllm_mxfp4_moe_supported() -> bool:
    if os.environ.get("FASTKERNELS_TRTLLM_MXFP4_MOE", "1") == "0":
        return False
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 10


def round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


# ---------------------------------------------------------------------------
# Routing bookkeeping.  Work is decomposed over (expert, m_start) tiles so an
# expert's weights are streamed once for all of its tokens; the tile table is
# built on device (a host readback of the histogram would stall the launch
# pipeline behind the harness's 330 us input copy).
# ---------------------------------------------------------------------------
@triton.jit
def _build_tiles(counts_ptr, desc_ptr, nvalid_ptr,
                 E: tl.constexpr, TMAX: tl.constexpr, BM: tl.constexpr):
    """One CTA: per-expert histogram -> work-tile descriptors.

    ``desc`` is packed 4-wide so a gemm CTA fetches (expert, m_start, count,
    token_offset) with four independent loads out of one 16-byte line instead of
    a three-deep dependent pointer chase.
    """
    e = tl.arange(0, E)
    c = tl.load(counts_ptr + e).to(tl.int32)
    eoff = tl.cumsum(c) - c

    ntile = (c + (BM - 1)) // BM
    cum = tl.cumsum(ntile)
    start = cum - ntile
    tl.store(nvalid_ptr, tl.sum(ntile))

    s = tl.arange(0, TMAX)
    hit = (start[None, :] <= s[:, None]) & (cum[None, :] > s[:, None])
    te = tl.sum(tl.where(hit, e[None, :], 0), axis=1)
    ts = tl.sum(tl.where(hit, start[None, :], 0), axis=1)
    tc = tl.sum(tl.where(hit, c[None, :], 0), axis=1)
    to = tl.sum(tl.where(hit, eoff[None, :], 0), axis=1)
    tl.store(desc_ptr + s * 4 + 0, te)
    tl.store(desc_ptr + s * 4 + 1, (s - ts) * BM)
    tl.store(desc_ptr + s * 4 + 2, tc)
    tl.store(desc_ptr + s * 4 + 3, to)


@triton.jit
def _route_small(logit_ptr, tw_ptr, order_ptr, desc_ptr, nvalid_ptr, M, LS,
                 E: tl.constexpr, KTOP: tl.constexpr, TM: tl.constexpr,
                 TP: tl.constexpr, TMAX: tl.constexpr, BM: tl.constexpr):
    """One CTA: top-k + softmax + counting sort + tile descriptors for few tokens.

    Replaces topk / softmax / .to(bf16) / argsort / zero_ / index_add_ /
    _build_tiles -- 8+ launches, and ``torch.topk`` alone is 12 us of device time
    on a [1, 128] logit row (multi-pass radix select) against 6 us for this whole
    kernel.  Only valid while the ``[TP, TP]`` rank matrix fits in registers,
    hence the ``_ROUTE_MAX`` gate.
    """
    mrow = tl.arange(0, TM)
    ecol = tl.arange(0, E)
    mv = mrow < M
    lg = tl.load(logit_ptr + mrow[:, None] * LS + ecol[None, :],
                 mask=mv[:, None], other=float("-inf")).to(tl.float32)

    # top-k by KTOP passes of argmax; ties resolve to the lower expert index, as
    # torch.topk does for this input.
    kcol = tl.arange(0, KTOP)
    ep2 = tl.zeros((TM, KTOP), dtype=tl.int32)
    v2 = tl.full((TM, KTOP), float("-inf"), dtype=tl.float32)
    for j in tl.static_range(KTOP):
        idx = tl.argmax(lg, axis=1).to(tl.int32)
        val = tl.max(lg, axis=1)
        hit = kcol[None, :] == j
        ep2 = tl.where(hit, idx[:, None], ep2)
        v2 = tl.where(hit, val[:, None], v2)
        lg = tl.where(ecol[None, :] == idx[:, None], float("-inf"), lg)

    ex = tl.exp(v2 - tl.max(v2, axis=1, keep_dims=True))
    w2 = ex / tl.sum(ex, axis=1, keep_dims=True)

    p = tl.arange(0, TP)
    pv = p < M * KTOP
    ep = tl.reshape(ep2, (TP,))
    tl.store(tw_ptr + p, tl.reshape(w2, (TP,)).to(tl.bfloat16), mask=pv)

    ev = tl.arange(0, E)
    hit = (ep[:, None] == ev[None, :]) & pv[:, None]
    c = tl.sum(hit.to(tl.int32), axis=0)
    eoff = tl.cumsum(c) - c

    same = (ep[:, None] == ep[None, :]) & (p[None, :] < p[:, None]) & pv[None, :]
    rank = tl.sum(same.to(tl.int32), axis=1)
    base = tl.sum(tl.where(hit, eoff[None, :], 0), axis=1)
    tl.store(order_ptr + base + rank, p, mask=pv)

    ntile = (c + (BM - 1)) // BM
    cum = tl.cumsum(ntile)
    start = cum - ntile
    tl.store(nvalid_ptr, tl.sum(ntile))
    s = tl.arange(0, TMAX)
    th = (start[None, :] <= s[:, None]) & (cum[None, :] > s[:, None])
    tl.store(desc_ptr + s * 4 + 0, tl.sum(tl.where(th, ev[None, :], 0), axis=1))
    tl.store(desc_ptr + s * 4 + 1,
             (s - tl.sum(tl.where(th, start[None, :], 0), axis=1)) * BM)
    tl.store(desc_ptr + s * 4 + 2, tl.sum(tl.where(th, c[None, :], 0), axis=1))
    tl.store(desc_ptr + s * 4 + 3, tl.sum(tl.where(th, eoff[None, :], 0), axis=1))


@triton.jit
def _row_perm(n):
    """Physical (shuffled) row that holds logical row ``n``."""
    t = n % 32
    return n - t + (t % 4) * 8 + (t // 4)


@triton.jit
def _scale_addr(mi, kk, NT: tl.constexpr):
    """Byte offset of the UE8M0 scale for physical row ``mi``, 32-block ``kk``."""
    return (((mi // 128) * NT + kk // 4) * 512
            + (mi % 32) * 16 + ((mi % 128) // 32) * 4 + kk % 4)


# ---------------------------------------------------------------------------
# gemm1 + SwiGLU-OAI.  One CTA owns (work tile, N tile); the N tile spans
# complete logical (up, gate) row pairs so the activation is finished in-register.
# ---------------------------------------------------------------------------
@triton.jit
def _gemm1_swiglu(x_ptr, w_ptr, s_ptr, b_ptr, act_ptr, order_ptr,
                  desc_ptr, nvalid_ptr,
                  H: tl.constexpr, N2: tl.constexpr, I: tl.constexpr,
                  NT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                  BK: tl.constexpr, ALPHA: tl.constexpr, BETA: tl.constexpr,
                  LIMIT: tl.constexpr):
    tile = tl.program_id(0)
    if tile >= tl.load(nvalid_ptr):
        return
    d = desc_ptr + tile * 4
    e = tl.load(d + 0)
    m0 = tl.load(d + 1)
    cnt = tl.load(d + 2)
    off = tl.load(d + 3)

    q = off + m0 + tl.arange(0, BM)
    mm = (m0 + tl.arange(0, BM)) < cnt
    pair = tl.load(order_ptr + q, mask=mm, other=0)
    tok = pair // 4

    n0 = tl.program_id(1) * BN
    t = tl.arange(0, BN)
    mi = _row_perm(n0 + t)

    wbase = w_ptr + e.to(tl.int64) * (N2 * (H // 2)) + mi.to(tl.int64) * (H // 2)
    sbase = s_ptr + e.to(tl.int64) * (N2 * (H // 32))
    kb = tl.arange(0, BK // 2)
    sj = tl.arange(0, BK // 32)

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, H, BK):
        xt = tl.load(x_ptr + tok[:, None].to(tl.int64) * H + (k0 + tl.arange(0, BK))[None, :],
                     mask=mm[:, None], other=0.0)
        wt = tl.load(wbase[None, :] + (k0 // 2 + kb)[:, None])
        st = tl.load(sbase + _scale_addr(mi[:, None], (k0 // 32 + sj)[None, :], NT))
        acc = tl.dot_scaled(xt, None, "bf16", wt, st, "e2m1", acc)

    y = acc + tl.load(b_ptr + e.to(tl.int64) * N2 + mi)[None, :]
    y = y.to(tl.bfloat16).to(tl.float32)
    up, gate = tl.split(tl.reshape(y, (BM, BN // 2, 2)))
    g = tl.minimum(gate, LIMIT)
    u = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
    act = (u + BETA) * (g * tl.sigmoid(g * ALPHA))

    j = (n0 // 2) + tl.arange(0, BN // 2)
    tl.store(act_ptr + q[:, None].to(tl.int64) * I + j[None, :],
             act.to(tl.bfloat16), mask=mm[:, None])


# ---------------------------------------------------------------------------
# gemm2 + weighted finalize.  Accumulates fp32 into out[token, :] with atomics;
# the top_k=4 contributions per token are independent tiles.
# ---------------------------------------------------------------------------
@triton.jit
def _gemm2_finalize(act_ptr, w_ptr, s_ptr, b_ptr, tw_ptr, out_ptr, order_ptr,
                    desc_ptr, nvalid_ptr,
                    I: tl.constexpr, HO: tl.constexpr, HU: tl.constexpr,
                    NT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                    BK: tl.constexpr):
    tile = tl.program_id(0)
    if tile >= tl.load(nvalid_ptr):
        return
    d = desc_ptr + tile * 4
    e = tl.load(d + 0)
    m0 = tl.load(d + 1)
    cnt = tl.load(d + 2)
    off = tl.load(d + 3)

    q = off + m0 + tl.arange(0, BM)
    mm = (m0 + tl.arange(0, BM)) < cnt
    pair = tl.load(order_ptr + q, mask=mm, other=0)
    tok = pair // 4

    n0 = tl.program_id(1) * BN
    t = tl.arange(0, BN)
    h = n0 + t
    mi = _row_perm(h)

    wbase = w_ptr + e.to(tl.int64) * (HO * (I // 2)) + mi.to(tl.int64) * (I // 2)
    sbase = s_ptr + e.to(tl.int64) * (HO * (I // 32))
    kb = tl.arange(0, BK // 2)
    sj = tl.arange(0, BK // 32)

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, I, BK):
        at = tl.load(act_ptr + q[:, None].to(tl.int64) * I + (k0 + tl.arange(0, BK))[None, :],
                     mask=mm[:, None], other=0.0)
        wt = tl.load(wbase[None, :] + (k0 // 2 + kb)[:, None])
        st = tl.load(sbase + _scale_addr(mi[:, None], (k0 // 32 + sj)[None, :], NT))
        acc = tl.dot_scaled(at, None, "bf16", wt, st, "e2m1", acc)

    oe = acc + tl.load(b_ptr + e.to(tl.int64) * HO + mi)[None, :]
    oe = oe.to(tl.bfloat16).to(tl.float32)
    w = tl.load(tw_ptr + pair, mask=mm, other=0.0).to(tl.float32)
    tl.atomic_add(out_ptr + tok[:, None].to(tl.int64) * HU + h[None, :],
                  oe * w[:, None], mask=mm[:, None] & (h < HU)[None, :])


class TrtLlmMxfp4MoE(nn.Module):
    """Router + experts.  Own fused kernels for small/medium M, flashinfer above."""

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        intermediate_size: int,
        hidden_size_unpadded: int,
        max_capture_size: int = DEFAULT_TUNE_MAX_NUM_TOKENS,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.intermediate_size = intermediate_size
        self.hidden_size_unpadded = hidden_size_unpadded
        self.max_capture_size = max(int(max_capture_size), 1)
        dev = torch.cuda.current_device()
        self.register_buffer(
            "gemm1_alpha",
            torch.full((num_experts,), SWIGLU_ALPHA, dtype=torch.float32, device=dev),
            persistent=False,
        )
        self.register_buffer(
            "gemm1_beta",
            torch.full((num_experts,), SWIGLU_BETA, dtype=torch.float32, device=dev),
            persistent=False,
        )
        self.register_buffer(
            "gemm1_clamp_limit",
            torch.full((num_experts,), SWIGLU_LIMIT, dtype=torch.float32, device=dev),
            persistent=False,
        )
        self._ws: dict = {}

    # -- workspace ---------------------------------------------------------
    def _workspace(self, m: int, device, hidden: int):
        key = (m, device, hidden)
        ws = self._ws.get(key)
        if ws is None:
            e = self.num_experts
            p = m * self.top_k
            ws = {
                "counts": torch.empty(e, dtype=torch.int32, device=device),
                "ones": torch.ones(p, dtype=torch.int32, device=device),
                "order": torch.empty(p, dtype=torch.int32, device=device),
                "tw": torch.empty(p, dtype=torch.bfloat16, device=device),
                "act": torch.empty(p, self.intermediate_size,
                                   dtype=torch.bfloat16, device=device),
                "out": torch.empty(m, self.hidden_size_unpadded,
                                   dtype=torch.float32, device=device),
                "nvalid": torch.empty(1, dtype=torch.int32, device=device),
            }
            tmax = min(e, p) + p // _BM + 1
            ws["tmax"] = tmax
            # _build_tiles / _route_small emit next_power_of_2(tmax) descriptors
            # (arange needs a power of two); size the buffer for the padded count.
            ws["desc"] = torch.empty(triton.next_power_of_2(tmax) * 4,
                                     dtype=torch.int32, device=device)
            self._ws[key] = ws
        return ws

    # -- fused path --------------------------------------------------------
    def _fused(self, hidden_states, router_logits, w13_weight, w13_weight_scale,
               w13_bias, w2_weight, w2_weight_scale, w2_bias):
        m, hidden = hidden_states.shape
        dev = hidden_states.device
        e = self.num_experts
        k = self.top_k
        inter = self.intermediate_size
        hu = self.hidden_size_unpadded
        ws = self._workspace(m, dev, hidden)

        out = ws["out"]
        bm = _BM
        tmax_p2 = triton.next_power_of_2(ws["tmax"])
        if m <= _ROUTE_MAX and router_logits.stride(1) == 1:
            tw, order = ws["tw"], ws["order"]
            _route_small[(1,)](
                router_logits, tw, order, ws["desc"], ws["nvalid"], m,
                router_logits.stride(0),
                E=e, KTOP=k, TM=triton.next_power_of_2(m),
                TP=triton.next_power_of_2(m) * k, TMAX=tmax_p2,
                BM=bm, num_warps=4,
            )
        else:
            tv, ti = torch.topk(router_logits.float(), k, dim=-1)
            tw = torch.softmax(tv, dim=-1).to(torch.bfloat16).reshape(-1)
            flat_e = ti.reshape(-1)
            order = torch.argsort(flat_e, stable=True).to(torch.int32)
            counts = ws["counts"].zero_()
            counts.index_add_(0, flat_e, ws["ones"])
            _build_tiles[(1,)](counts, ws["desc"], ws["nvalid"], E=e,
                               TMAX=tmax_p2, BM=bm, num_warps=4)

        n2 = 2 * inter
        ho = w2_weight.shape[1]
        if _cuda_ok(m, hidden, inter, n2, ho, hu):
            small = m <= 8
            # gemm1 also zeroes `out` for gemm2's atomics -- one launch less.
            _EXT.gemm1(hidden_states, w13_weight, w13_weight_scale.view(torch.uint8),
                       w13_bias, order, ws["desc"], ws["nvalid"], ws["act"], out,
                       ws["tmax"], k, small)
            _EXT.gemm2(ws["act"], w2_weight, w2_weight_scale.view(torch.uint8),
                       w2_bias, tw, order, ws["desc"], ws["nvalid"], out,
                       ws["tmax"], k, small)
            return out.to(torch.bfloat16)

        out.zero_()
        c1, c2 = _tile_config(m)
        bn1, bk1, nw1, ns1 = c1
        _gemm1_swiglu[(ws["tmax"], n2 // bn1)](
            hidden_states, w13_weight, w13_weight_scale.view(torch.uint8), w13_bias,
            ws["act"], order, ws["desc"], ws["nvalid"],
            H=hidden, N2=n2, I=inter, NT=(hidden // 32 + 3) // 4,
            BM=bm, BN=bn1, BK=bk1,
            ALPHA=SWIGLU_ALPHA, BETA=SWIGLU_BETA, LIMIT=SWIGLU_LIMIT,
            num_warps=nw1, num_stages=ns1,
        )

        bn2, bk2, nw2, ns2 = c2
        _gemm2_finalize[(ws["tmax"], (hu + bn2 - 1) // bn2)](
            ws["act"], w2_weight, w2_weight_scale.view(torch.uint8), w2_bias, tw,
            out, order, ws["desc"], ws["nvalid"],
            I=inter, HO=ho, HU=hu, NT=(inter // 32 + 3) // 4,
            BM=bm, BN=bn2, BK=bk2, num_warps=nw2, num_stages=ns2,
        )
        return out.to(torch.bfloat16)

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        w13_weight: torch.Tensor,
        w13_weight_scale: torch.Tensor,
        w13_bias: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_scale: torch.Tensor,
        w2_bias: torch.Tensor,
    ) -> torch.Tensor:
        assert hidden_states.dtype == torch.bfloat16
        if hidden_states.dim() == 2 and hidden_states.shape[0] <= FUSED_MAX_TOKENS:
            return self._fused(hidden_states, router_logits, w13_weight,
                               w13_weight_scale, w13_bias, w2_weight,
                               w2_weight_scale, w2_bias)
        output = torch.empty(
            *hidden_states.shape[:-1],
            self.hidden_size_unpadded,
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )
        trtllm_fp4_block_scale_moe(
            routing_logits=router_logits.to(torch.bfloat16),
            routing_bias=None,
            hidden_states=hidden_states,
            hidden_states_scale=None,
            gemm1_weights=w13_weight,
            gemm1_weights_scale=w13_weight_scale,
            gemm1_bias=w13_bias,
            gemm1_alpha=self.gemm1_alpha,
            gemm1_beta=self.gemm1_beta,
            gemm1_clamp_limit=self.gemm1_clamp_limit,
            gemm2_weights=w2_weight,
            gemm2_weights_scale=w2_weight_scale,
            gemm2_bias=w2_bias,
            output1_scale_scalar=None,
            output1_scale_gate_scalar=None,
            output2_scale_scalar=None,
            num_experts=self.num_experts,
            top_k=self.top_k,
            n_group=None,
            topk_group=None,
            intermediate_size=self.intermediate_size,
            local_expert_offset=0,
            local_num_experts=self.num_experts,
            routed_scaling_factor=None,
            routing_method_type=ROUTING_RENORMALIZE_NAIVE,
            do_finalize=True,
            tune_max_num_tokens=self.max_capture_size,
            output=output,
        )
        return output
