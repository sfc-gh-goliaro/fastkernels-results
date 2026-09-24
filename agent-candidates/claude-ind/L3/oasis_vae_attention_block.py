"""Oasis VAE attention block -- single-block fused Triton implementation.

The baseline runs the block as ~35 separate eager ops: two fp32-promoted
LayerNorms (cast up, normalize, cast down), a QKV projection, a rotary
embedding built from ``cos``/``sin`` recomputed on every call plus a
``rotate_half`` that materializes three temporaries, three ``contiguous``
clones to reach the attention layout, SDPA, and an MLP.  At these shapes
(576 tokens per image, dim 1024) the block is dominated by launch and
memory-traffic overhead rather than by the 95 GFLOP of actual math.

This implementation writes the whole block as seven Triton kernels:

  1. LayerNorm 1                      (fp32 reduction, fp16 out)
  2. QKV GEMM   + bias + rotary       (rotary folded into the GEMM epilogue)
  3. Flash attention                  (reads Q/K/V straight out of the
                                       packed QKV buffer, writes the
                                       ``(B, S, H*D)`` layout directly)
  4. proj GEMM  + bias + residual
  5. LayerNorm 2
  6. fc1 GEMM   + bias + exact GELU
  7. fc2 GEMM   + bias + residual

The GEMMs are persistent Blackwell kernels: device-side TMA descriptors for
both operands, ``tl.dot`` lowered to ``tcgen05`` MMA, and warp specialization
on the K loop with the accumulator single-buffered in TMEM.  The whole chain is
captured into a CUDA graph on the second call with a given input layout, so
replay costs one launch instead of seven.  Every elementwise stage (bias, rotary, GELU, both residual
adds) lives in a GEMM epilogue, so no tensor is read back purely to have a
pointwise op applied to it.  The rotary ``cos``/``sin`` tables are built once
and padded with 1/0 past ``rot_dim`` so the rotation applies uniformly across
the whole head without a masked slice.

Numerics follow the baseline step for step: reductions and the rotary run in
fp32, and every value is rounded to fp16 exactly where the eager graph would
have materialized an fp16 tensor.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.layer_norm import LayerNorm
from ..L2.oasis_mlp import OasisMLP
from ..L2.oasis_vae_attention import OasisVAEAttention

# ---------------------------------------------------------------------------
# Scratch for device-side TMA descriptors.  Triton asks for a few hundred bytes
# per launch; serving it from one cached buffer keeps ``torch.empty`` out of the
# launch path (it would cost more CPU time than the kernels themselves at
# B=1).  Launches on a stream are ordered, so reusing the buffer is safe.
# ---------------------------------------------------------------------------
_TMA_SCRATCH: list[torch.Tensor] = []


def _tma_alloc(size: int, alignment: int, stream):
    if _TMA_SCRATCH and _TMA_SCRATCH[0].numel() >= size:
        return _TMA_SCRATCH[0]
    buf = torch.empty(max(int(size), 1 << 15), dtype=torch.int8, device="cuda")
    _TMA_SCRATCH.clear()
    _TMA_SCRATCH.append(buf)
    return buf


triton.set_allocator(_tma_alloc)

# CUDA-graph replay removes the ~4 us/launch host cost of the seven kernels.
# At B=1 the whole block is ~100 us of GPU work, so that is a real fraction of
# it.  Off via FK_OASIS_NO_GRAPH=1 for A/B measurement.
_USE_GRAPH = os.environ.get("FK_OASIS_NO_GRAPH", "") != "1"

LOG2E = 1.4426950408889634


# ---------------------------------------------------------------------------
# LayerNorm: fp32 reduction over a strided input, fp16 contiguous output.
# ---------------------------------------------------------------------------
@triton.jit
def _ln_kernel(X, Wg, Bg, Y, M, sb, sm, sd,
               S: tl.constexpr, N: tl.constexpr, BM: tl.constexpr,
               EPS: tl.constexpr, UNIFORM: tl.constexpr, AFFINE: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    ok = rm < M
    rmc = tl.where(ok, rm, 0)
    if UNIFORM:
        base = rmc * sm
    else:
        base = (rmc // S) * sb + (rmc % S) * sm
    cols = tl.arange(0, N)
    x = tl.load(X + base[:, None] + cols[None, :] * sd,
                mask=ok[:, None], other=0.0).to(tl.float32)
    mu = tl.sum(x, 1) * (1.0 / N)
    xc = x - mu[:, None]
    var = tl.sum(xc * xc, 1) * (1.0 / N)
    y = xc * tl.rsqrt(var + EPS)[:, None]
    if AFFINE:
        y = y * tl.load(Wg + cols).to(tl.float32)[None, :]
        y = y + tl.load(Bg + cols).to(tl.float32)[None, :]
    tl.store(Y + rm[:, None] * N + cols[None, :], y.to(tl.float16),
             mask=ok[:, None])


# ---------------------------------------------------------------------------
# Persistent TMA GEMM with a selectable fused epilogue.
#
#   EPI 0 -- QKV: bias, then interleaved rotary on the Q and K halves.
#   EPI 1 -- bias + residual add read through arbitrary strides.
#   EPI 2 -- bias + exact GELU.
# ---------------------------------------------------------------------------
@triton.jit
def _gemm_kernel(A, Bw, C, Bias, RES, COS, SIN,
                 M, N, K, sb, sm, sd,
                 S: tl.constexpr, D: tl.constexpr, ROT_SPLIT: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr,
                 WS: tl.constexpr, EPI: tl.constexpr, UNIFORM: tl.constexpr,
                 EXACT: tl.constexpr, DAMB: tl.constexpr):
    a_desc = tl.make_tensor_descriptor(A, shape=[M, K], strides=[K, 1],
                                      block_shape=[BM, BK])
    b_desc = tl.make_tensor_descriptor(Bw, shape=[K, N], strides=[N, 1],
                                      block_shape=[BK, BN])
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(N, BN)
    nk = tl.cdiv(K, BK)
    width = GROUP_M * nn
    for tile in tl.range(tl.program_id(0), nm * nn, NUM_SMS, flatten=True):
        gid = tile // width
        fm = gid * GROUP_M
        gsz = min(nm - fm, GROUP_M)
        pm = fm + ((tile % width) % gsz)
        pn = (tile % width) // gsz
        om = pm * BM
        on = pn * BN
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in tl.range(nk, warp_specialize=WS,
                          disallow_acc_multi_buffer=DAMB):
            acc = tl.dot(a_desc.load([om, k * BK]),
                         b_desc.load([k * BK, on]), acc)
        rm = om + tl.arange(0, BM)
        rn = on + tl.arange(0, BN)
        ok = rm < M
        acc += tl.load(Bias + rn).to(tl.float32)[None, :]
        v = acc.to(tl.float16)
        if EPI == 0:
            if on < ROT_SPLIT:
                vf = v.to(tl.float32)
                even, odd = tl.split(tl.reshape(vf, [BM, BN // 2, 2]))
                rot = tl.reshape(tl.join(-odd, even), [BM, BN])
                s = tl.where(ok, rm, 0) % S
                idx = s[:, None] * D + ((on + tl.arange(0, BN)) % D)[None, :]
                v = (vf * tl.load(COS + idx)
                     + rot * tl.load(SIN + idx)).to(tl.float16)
        elif EPI == 1:
            rmc = tl.where(ok, rm, 0)
            if UNIFORM:
                base = rmc * sm
            else:
                base = (rmc // S) * sb + (rmc % S) * sm
            r = tl.load(RES + base[:, None] + rn[None, :] * sd,
                        mask=ok[:, None], other=0.0)
            v = (v.to(tl.float32) + r.to(tl.float32)).to(tl.float16)
        elif EPI == 2:
            # Exact (erf) GELU.  libdevice's ``erff`` is a branchy multi-branch
            # polynomial: inlined into the epilogue it spilled enough registers
            # to slow the whole mainloop down by 4x.  Abramowitz & Stegun 7.1.26
            # instead needs one reciprocal, one exp2 and five FMAs, and its
            # 1.5e-7 worst-case error is two orders of magnitude below the fp16
            # rounding the result is about to take anyway.
            vf = v.to(tl.float32)
            u = tl.abs(vf) * 0.7071067811865476
            t = 1.0 / (1.0 + 0.3275911 * u)
            poly = 0.254829592 + t * (-0.284496736 + t * (1.421413741
                   + t * (-1.453152027 + t * 1.061405429)))
            e = 1.0 - t * poly * tl.exp2(u * u * -1.4426950408889634)
            v = (0.5 * (vf + tl.abs(vf) * e)).to(tl.float16)
        if EXACT:
            tl.store(C + rm[:, None] * N + rn[None, :], v)
        else:
            tl.store(C + rm[:, None] * N + rn[None, :], v, mask=ok[:, None])


# ---------------------------------------------------------------------------
# Flash attention over the packed QKV buffer.
#
# QKV is laid out ``(B*S, 3*H*D)`` so Q/K/V for one head are plain column
# slices; the output is written straight into ``(B*S, H*D)``, which is the
# ``(B, S, H*D)`` view the projection consumes -- no permute, no clone.
# ---------------------------------------------------------------------------
@triton.jit
def _attn_kernel(QKV, O, S: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, QK_SCALE: tl.constexpr,
                 EXACT: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    row = 3 * H * D
    d = tl.arange(0, D)
    rs = pid_m * BM + tl.arange(0, BM)
    acc = tl.zeros((BM, D), dtype=tl.float32)
    mi = tl.full((BM,), -float("inf"), tl.float32)
    li = tl.zeros((BM,), tl.float32)
    if EXACT:
        qp = QKV + (b * S + rs)[:, None] * row + h * D + d[None, :]
        q = tl.load(qp)
        for n0 in tl.range(0, S, BN):
            kb = QKV + (b * S + n0 + tl.arange(0, BN))[:, None] * row + h * D + d[None, :]
            qk = tl.dot(q, tl.trans(tl.load(kb + H * D))) * QK_SCALE
            mnew = tl.maximum(mi, tl.max(qk, 1))
            alpha = tl.exp2(mi - mnew)
            p = tl.exp2(qk - mnew[:, None])
            li = li * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(tl.float16), tl.load(kb + 2 * H * D), acc)
            mi = mnew
        tl.store(O + (b * S + rs)[:, None] * (H * D) + h * D + d[None, :],
                 (acc / li[:, None]).to(tl.float16))
    else:
        sok = rs < S
        qp = QKV + (b * S + tl.where(sok, rs, 0))[:, None] * row + h * D + d[None, :]
        q = tl.load(qp, mask=sok[:, None], other=0.0)
        for n0 in tl.range(0, S, BN):
            rn = n0 + tl.arange(0, BN)
            nok = rn < S
            kb = QKV + (b * S + tl.where(nok, rn, 0))[:, None] * row + h * D + d[None, :]
            k = tl.load(kb + H * D, mask=nok[:, None], other=0.0)
            qk = tl.dot(q, tl.trans(k)) * QK_SCALE
            qk = tl.where(nok[None, :], qk, -float("inf"))
            mnew = tl.maximum(mi, tl.max(qk, 1))
            alpha = tl.exp2(mi - mnew)
            p = tl.exp2(qk - mnew[:, None])
            li = li * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            v = tl.load(kb + 2 * H * D, mask=nok[:, None], other=0.0)
            acc = tl.dot(p.to(tl.float16), v, acc)
            mi = mnew
        tl.store(O + (b * S + rs)[:, None] * (H * D) + h * D + d[None, :],
                 (acc / li[:, None]).to(tl.float16), mask=sok[:, None])


# ---------------------------------------------------------------------------
# Launch configs.  Picked by an offline sweep over the two captured token
# counts (3456 and 576); ``_CFG[large]`` selects between them.
# ---------------------------------------------------------------------------
#                   BM   BN   BK  warps  stages  WS  DAMB  CTAS
_CFG = {
    ("qkv", True): (128, 128, 128, 8, 3, True, True, 1),
    ("qkv", False): (64, 64, 128, 4, 5, True, False, 1),
    ("proj", True): (128, 256, 64, 8, 4, True, True, 1),
    ("proj", False): (64, 64, 128, 4, 4, False, False, 1),
    ("fc1", True): (128, 256, 64, 8, 4, True, True, 1),
    ("fc1", False): (64, 64, 128, 4, 5, True, False, 1),
    ("fc2", True): (128, 256, 64, 8, 4, True, True, 1),
    ("fc2", False): (64, 64, 128, 4, 5, True, False, 1),
}
_ATTN_CFG = {True: (64, 64, 4, 2), False: (64, 64, 4, 3)}
_LN_CFG = (8, 4)             # BM, warps
_LARGE_M = 2048


def _grid(sms: int, ntiles: int, nctas: int) -> int:
    """One persistent CTA per SM, rounded down to a whole number of clusters."""
    g = min(sms, ntiles)
    if nctas > 1:
        g = max(nctas, (g // nctas) * nctas)
    return g


class OasisVAEAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        frame_height: int,
        frame_width: int,
        *,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.attn = OasisVAEAttention(
            dim,
            num_heads,
            frame_height,
            frame_width,
            qkv_bias=qkv_bias,
        )
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.mlp = OasisMLP(dim, hidden_features=int(dim * mlp_ratio), approximate_tanh=False)

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.seq_len = frame_height * frame_width
        self.hidden = int(dim * mlp_ratio)
        self._plan: dict | None = None
        self._bufs: dict[int, tuple] = {}
        self._graphs: dict = {}
        self._seen: dict = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._supported(x):
            return self._reference(x)
        if self._plan is None or self._plan["src"] is not self.attn.qkv.weight:
            self._plan = self._build_plan()
            self._graphs.clear()
            self._seen.clear()
        if not _USE_GRAPH:
            return self._run(x)
        key = (x.shape, x.stride())
        got = self._graphs.get(key)
        if got is False:
            return self._run(x)
        if got is not None:
            xs, graph, out = got
            xs.copy_(x)
            graph.replay()
            return out
        # Two eager calls first: they JIT the kernels and settle the cached
        # intermediate buffers, so capture records launches only.
        n = self._seen.get(key, 0) + 1
        self._seen[key] = n
        if n < 2:
            return self._run(x)
        try:
            xs = torch.empty_strided(x.shape, x.stride(), dtype=x.dtype,
                                     device=x.device)
            xs.copy_(x)
            self._run(xs)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = self._run(xs)
        except Exception:
            self._graphs[key] = False
            return self._run(x)
        self._graphs[key] = (xs, graph, out)
        xs.copy_(x)
        graph.replay()
        return out

    # -- reference path (any shape/dtype/device the fused path does not cover)
    def _reference(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

    # -- one-time weight staging ------------------------------------------
    def _build_plan(self):
        attn, mlp = self.attn, self.mlp
        dev = attn.qkv.weight.device
        dim, hd = self.dim, self.head_dim

        def zeros(n):
            return torch.zeros(n, dtype=torch.float16, device=dev)

        def bias_of(lin, n):
            return lin.bias if lin.bias is not None else zeros(n)

        # ``cos``/``sin`` padded to the full head dim with 1/0 so the rotation
        # is an identity past ``rot_dim`` and needs no masked slice.
        freqs = attn.rotary_freqs.reshape(self.seq_len, -1).float()
        rd = freqs.shape[-1]
        cos = torch.ones(self.seq_len, hd, dtype=torch.float32, device=dev)
        sin = torch.zeros(self.seq_len, hd, dtype=torch.float32, device=dev)
        cos[:, :rd] = freqs.cos()
        sin[:, :rd] = freqs.sin()

        def norm_wb(norm):
            w = norm.weight if norm.weight is not None else None
            b = norm.bias if norm.bias is not None else None
            if w is None and b is None:
                return None, None, False
            if w is None:
                w = torch.ones(dim, dtype=b.dtype, device=dev)
            if b is None:
                b = torch.zeros(dim, dtype=w.dtype, device=dev)
            return w, b, True

        # Materialize the TMA scratch here, not lazily inside the first
        # launch: a graph capture must record launches only, never an
        # allocation whose pointer would then be baked into the graph.
        _tma_alloc(1 << 15, 128, None)

        n1w, n1b, aff1 = norm_wb(self.norm1)
        n2w, n2b, aff2 = norm_wb(self.norm2)
        return {
            "wqkv": attn.qkv.weight.t().contiguous(),
            "bqkv": bias_of(attn.qkv, 3 * dim),
            "wproj": attn.proj.weight.t().contiguous(),
            "bproj": bias_of(attn.proj, dim),
            "w1": mlp.fc1.weight.t().contiguous(),
            "b1": bias_of(mlp.fc1, self.hidden),
            "w2": mlp.fc2.weight.t().contiguous(),
            "b2": bias_of(mlp.fc2, dim),
            "cos": cos,
            "sin": sin,
            "n1w": n1w, "n1b": n1b, "aff1": aff1,
            "n2w": n2w, "n2b": n2b, "aff2": aff2,
            "sms": torch.cuda.get_device_properties(dev).multi_processor_count,
            "src": attn.qkv.weight,
            "eps1": self.norm1.eps,
            "eps2": self.norm2.eps,
        }

    def _supported(self, x: torch.Tensor) -> bool:
        return (
            x.is_cuda
            and x.dtype is torch.float16
            and x.dim() == 3
            and x.shape[1] == self.seq_len
            and x.shape[2] == self.dim
            and self.head_dim % 2 == 0
            and self.head_dim in (32, 64, 128)
            and self.dim % 256 == 0
            and self.hidden % 256 == 0
            and self.attn.qkv.weight.dtype is torch.float16
            and torch.cuda.get_device_capability(x.device)[0] >= 9
        )

    def _get_bufs(self, M: int, dev):
        got = self._bufs.get(M)
        if got is None:
            e = torch.empty
            got = (
                e((M, self.dim), dtype=torch.float16, device=dev),          # xn
                e((M, 3 * self.dim), dtype=torch.float16, device=dev),      # qkv
                e((M, self.dim), dtype=torch.float16, device=dev),          # ao
                e((M, self.dim), dtype=torch.float16, device=dev),          # x1
                e((M, self.dim), dtype=torch.float16, device=dev),          # xn2
                e((M, self.hidden), dtype=torch.float16, device=dev),       # hh
            )
            self._bufs[M] = got
        return got

    def _run(self, x: torch.Tensor) -> torch.Tensor:
        p = self._plan
        B, S, N = x.shape
        M = B * S
        H, Dh = self.num_heads, self.head_dim
        dev = x.device
        xn, qkv, ao, x1, xn2, hh = self._get_bufs(M, dev)
        out = torch.empty((B, S, N), dtype=torch.float16, device=dev)

        sb, sm, sd = x.stride()
        uni = (B == 1) or (sb == S * sm)
        large = M >= _LARGE_M
        sms = p["sms"]
        lbm, lnw = _LN_CFG

        # 1. LayerNorm 1
        _ln_kernel[(triton.cdiv(M, lbm),)](
            x, p["n1w"], p["n1b"], xn, M, sb, sm, sd,
            S=S, N=N, BM=lbm, EPS=p["eps1"], UNIFORM=uni, AFFINE=p["aff1"],
            num_warps=lnw)

        # 2. QKV projection + rotary
        bm, bn, bk, nw, ns, ws, damb, nct = _CFG[("qkv", large)]
        grid = _grid(sms, triton.cdiv(M, bm) * triton.cdiv(3 * N, bn), nct)
        _gemm_kernel[(grid,)](
            xn, p["wqkv"], qkv, p["bqkv"], qkv, p["cos"], p["sin"],
            M, 3 * N, N, 0, 0, 0,
            S=S, D=Dh, ROT_SPLIT=2 * N, BM=bm, BN=bn, BK=bk, GROUP_M=8,
            NUM_SMS=grid, WS=ws, EPI=0, UNIFORM=True, EXACT=(M % bm == 0),
            DAMB=damb, num_warps=nw, num_stages=ns, num_ctas=nct)

        # 3. attention
        abm, abn, anw, ans = _ATTN_CFG[large]
        _attn_kernel[(triton.cdiv(S, abm), B * H)](
            qkv, ao, S=S, H=H, D=Dh, BM=abm, BN=abn,
            QK_SCALE=LOG2E / (Dh ** 0.5),
            EXACT=(S % abm == 0 and S % abn == 0),
            num_warps=anw, num_stages=ans)

        # 4. out projection + residual
        bm, bn, bk, nw, ns, ws, damb, nct = _CFG[("proj", large)]
        grid = _grid(sms, triton.cdiv(M, bm) * triton.cdiv(N, bn), nct)
        _gemm_kernel[(grid,)](
            ao, p["wproj"], x1, p["bproj"], x, p["cos"], p["sin"],
            M, N, N, sb, sm, sd,
            S=S, D=Dh, ROT_SPLIT=0, BM=bm, BN=bn, BK=bk, GROUP_M=8,
            NUM_SMS=grid, WS=ws, EPI=1, UNIFORM=uni, EXACT=(M % bm == 0),
            DAMB=damb, num_warps=nw, num_stages=ns, num_ctas=nct)

        # 5. LayerNorm 2
        _ln_kernel[(triton.cdiv(M, lbm),)](
            x1, p["n2w"], p["n2b"], xn2, M, 0, N, 1,
            S=S, N=N, BM=lbm, EPS=p["eps2"], UNIFORM=True, AFFINE=p["aff2"],
            num_warps=lnw)

        # 6. fc1 + GELU
        Hd = self.hidden
        bm, bn, bk, nw, ns, ws, damb, nct = _CFG[("fc1", large)]
        grid = _grid(sms, triton.cdiv(M, bm) * triton.cdiv(Hd, bn), nct)
        _gemm_kernel[(grid,)](
            xn2, p["w1"], hh, p["b1"], hh, p["cos"], p["sin"],
            M, Hd, N, 0, 0, 0,
            S=S, D=Dh, ROT_SPLIT=0, BM=bm, BN=bn, BK=bk, GROUP_M=8,
            NUM_SMS=grid, WS=ws, EPI=2, UNIFORM=True, EXACT=(M % bm == 0),
            DAMB=damb, num_warps=nw, num_stages=ns, num_ctas=nct)

        # 7. fc2 + residual
        bm, bn, bk, nw, ns, ws, damb, nct = _CFG[("fc2", large)]
        grid = _grid(sms, triton.cdiv(M, bm) * triton.cdiv(N, bn), nct)
        _gemm_kernel[(grid,)](
            hh, p["w2"], out, p["b2"], x1, p["cos"], p["sin"],
            M, N, Hd, 0, N, 1,
            S=S, D=Dh, ROT_SPLIT=0, BM=bm, BN=bn, BK=bk, GROUP_M=8,
            NUM_SMS=grid, WS=ws, EPI=1, UNIFORM=True, EXACT=(M % bm == 0),
            DAMB=damb, num_warps=nw, num_stages=ns, num_ctas=nct)
        return out
