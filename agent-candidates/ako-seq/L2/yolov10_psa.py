"""YOLOv10 PSA (Partial Self-Attention) block -- fused Triton implementation.

At the captured shapes (fp16 [4,256,20,20] / [1,256,20,20]; c=128, n=400, 2
heads, key_dim=32, head_dim=64) every tensor is under 1 MB and the whole block
is ~1 GFLOP, so nothing here is FLOP-bound: the baseline dispatches ~25
kernels/memcpys (7x conv+BN+SiLU with nothing calling ``fuse()``, a split, two
batched matmuls, a softmax, a depthwise 3x3, two residual adds and a cat) and
spends ~460 us of *CPU dispatch* on ~100 us of GPU work. What a fused version
pays instead is ~5 us per launch, and that cost is per-*pass over memory*, not
per-flop -- so the shape of the problem is "how few passes over these tensors
can the block be written in", not "how fast can the GEMMs go".

The block is collapsed into five Triton launches:

  1. ``_gemm_kernel``     cv1 (256->256, bias+SiLU) -> both halves of the cv2
                          input buffer, so the trailing cat costs nothing.
  2. ``_gemm_kernel``     attention qkv projection (128->256).
  3. ``_attn_kernel``     flash-style attention: q^T k * scale, online softmax
                          and v @ attn^T, fp32 accumulation, [400,400] never
                          materialized.
  4. ``_tail_kernel``     depthwise-3x3 ``pe`` gathered from the qkv buffer, the
                          attention residual, ``proj``, the ``b`` residual, then
                          both ffn 1x1 convs with SiLU between them and the
                          second residual -- one pass over the b half instead of
                          two, with every intermediate staying in registers.
  5. ``_gemm_kernel``     cv2 (256->256, bias+SiLU) -> the NCHW output.

Three things make that possible:

* Every BatchNorm is folded into its conv weight/bias once, lazily on the first
  forward (exact in eval mode), so BN disappears at runtime.
* Activations stay in the baseline's own NCHW (channel-major) layout, so each
  1x1 conv is a GEMM ``out[cout, p] = W[cout, cin] @ in[cin, p]`` whose weight,
  input tile and result are all contiguous along the pixel axis: no permutes, no
  reshape kernels, and the folded conv weight is used as-is. The ``a``/``b``
  split and the cat become channel offsets into one buffer.
* Bias, SiLU and both residual adds ride in GEMM epilogues, and every kernel
  signals its dependents at the tail and waits at its first load (PDL), so each
  launch stages inside the previous kernel's tail.

Kernels are launched directly rather than replayed from a CUDA graph: the
benchmark hands a fresh input pointer every iteration, and its pre-iteration L2
flush leaves enough queued GPU work to hide the Python launch cost, so a graph
only adds a per-call input copy.

Why five and not fewer: on this shape a stage costs ~5us of GPU time almost
independently of how much arithmetic it does (the `tl.dot`s are free; one pass
over the ~800KB activation costs ~3.2us whatever you do, and a dependent launch
boundary ~1.2us), so the only lever is the number of passes over global memory.
Fusing pe+proj with the ffn wins because both own the same [CH, BP] tile and the
fused kernel can keep the CTA count of the two it replaces. Fusing cv1 with qkv
(0.88x) and folding the attention into the tail (0.95x) both lose: they need a
CTA to hold the whole channel dimension, which cuts the CTA count these
latency-bound stages depend on. See ITERATIONS.md for the measurements.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .yolov10_attention import YOLOAttention
from .yolov10_conv import YOLOConv

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

    _HAS_TRITON = True
except Exception:  # pragma: no cover - triton is present on the bench host
    _HAS_TRITON = False


# Programmatic dependent launch. Step 0's predecessor is not a kernel (it is
# whatever produced the input), so it neither waits nor triggers; steps 1..5 do
# both, which is worth ~1.25x here.
_PDL = True

# Per-stage launch configs, tuned on B200 against the benchmark's own timer.
# 1x1-conv GEMMs: (BC out-channel tile, BP pixel tile, num_warps, num_stages);
# attention: (BP query tile, BN key tile, ...); pe+proj: (BP, BK channel chunk,
# ...); ffn: (BP, BF hidden chunk, ...).
_CFG = {
    "cv1": (32, 64, 16, 2),
    "qkv": (32, 32, 8, 2),
    "cv2": (32, 32, 8, 2),
    "attn": (16, 512, 4, 2),
    # pe+proj+ffn: (BP, BK channel chunk, BF hidden chunk, warps, stages)
    "tail": (16, 64, 128, 8, 3),
}


if _HAS_TRITON:

    @triton.jit
    def _gemm_kernel(
        IN, W, BIAS, RES, OUT, N, SB_IN, SB_RES, SB_OUT,
        KIN: tl.constexpr, BC: tl.constexpr, BP: tl.constexpr,
        SILU: tl.constexpr, ADD_RES: tl.constexpr, PDL: tl.constexpr,
    ):
        """out[c, p] = act(W[c, :] @ in[:, p] + bias[c] [+ res[c, p]]).

        Channel-major, so every 1x1 conv in the block maps onto this kernel with
        no layout shuffling. The PDL wait goes before *every* load, weights
        included: Triton will happily schedule a tl.load above
        griddepcontrol.wait, and a hoisted load of a predecessor's output reads
        stale memory (observed as garbage in the b half of the cv2 input).
        """
        pid_p = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_b = tl.program_id(2)
        offs_p = pid_p * BP + tl.arange(0, BP)
        mask_p = offs_p < N
        offs_c = pid_c * BC + tl.arange(0, BC)
        offs_k = tl.arange(0, KIN)

        if PDL:
            gdc_wait()
        w = tl.load(W + offs_c[:, None] * KIN + offs_k[None, :])
        bias = tl.load(BIAS + offs_c)[:, None]
        a = tl.load(IN + pid_b * SB_IN + offs_k[:, None] * N + offs_p[None, :],
                    mask=mask_p[None, :], other=0.0)
        acc = tl.dot(w, a) + bias
        if ADD_RES:
            acc += tl.load(RES + pid_b * SB_RES + offs_c[:, None] * N + offs_p[None, :],
                           mask=mask_p[None, :], other=0.0).to(tl.float32)
        if SILU:
            acc = acc * tl.sigmoid(acc)
        tl.store(OUT + pid_b * SB_OUT + offs_c[:, None] * N + offs_p[None, :],
                 acc.to(OUT.dtype.element_ty), mask=mask_p[None, :])
        if PDL:
            gdc_launch_dependents()

    @triton.jit
    def _attn_kernel(
        QKV, ATTN, N, scale, SB_QKV, SB_ATTN,
        KD: tl.constexpr, HD: tl.constexpr, HW: tl.constexpr,
        BP: tl.constexpr, BN: tl.constexpr, PDL: tl.constexpr,
    ):
        """Flash-style attention read straight out of the packed qkv buffer.

        attn[i, j] = softmax_j(scale * sum_d q[d, i] k[d, j]);
        out[e, i]  = sum_j attn[i, j] v[e, j]. Online softmax in fp32, so the
        [N, N] score matrix never leaves registers.
        """
        pid_p = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)
        offs_i = pid_p * BP + tl.arange(0, BP)
        mask_i = offs_i < N
        offs_d = tl.arange(0, KD)
        offs_e = tl.arange(0, HD)
        base = QKV + pid_b * SB_QKV + pid_h * (HW * N)
        if PDL:
            gdc_wait()

        q = tl.load(base + offs_d[None, :] * N + offs_i[:, None],
                    mask=mask_i[:, None], other=0.0)
        acc = tl.zeros((BP, HD), dtype=tl.float32)
        m_i = tl.full((BP,), -1e30, dtype=tl.float32)
        l_i = tl.zeros((BP,), dtype=tl.float32)
        kp = base + (KD + offs_d)[:, None] * N
        vp = base + (2 * KD + offs_e)[None, :] * N
        for n0 in tl.range(0, N, BN):
            offs_j = n0 + tl.arange(0, BN)
            mask_j = offs_j < N
            k = tl.load(kp + offs_j[None, :], mask=mask_j[None, :], other=0.0)
            s = tl.dot(q, k) * scale
            s = tl.where(mask_j[None, :], s, -1e30)
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            v = tl.load(vp + offs_j[:, None], mask=mask_j[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        out = acc / l_i[:, None]
        tl.store(ATTN + pid_b * SB_ATTN + (pid_h * HD + offs_e)[None, :] * N + offs_i[:, None],
                 out.to(ATTN.dtype.element_ty), mask=mask_i[:, None])
        if PDL:
            gdc_launch_dependents()

    @triton.jit
    def _tail_kernel(
        QKV, ATTN, PEW, BPE, W, BIAS, W0, B0, W1, B1, IO, N, H, WD,
        SB_QKV, SB_ATTN, SB_IO,
        KD: tl.constexpr, HD: tl.constexpr, HW: tl.constexpr, CH: tl.constexpr,
        CF: tl.constexpr, BP: tl.constexpr, BK: tl.constexpr, BF: tl.constexpr,
        PDL: tl.constexpr,
    ):
        """b += proj(attn + pe(v)); then b += ffn(b) -- one launch, one store.

        The 3x3 neighbourhood is gathered from the qkv buffer, so v is never
        re-materialized as its own [B, C, H, W] tensor. Each tap is a *constant*
        pixel shift dy*WD+dx off one hoisted pointer tile, so the gathers stay
        vectorized and the halo costs only a mask.

        pe+proj and the ffn own the same [CH, BP] tile of the b half, so keeping
        them in one kernel keeps the intermediate in registers: split across two
        launches it was stored and loaded straight back, a full round trip over
        409KB plus a dependent launch boundary. The fp16 round trip through ``b``
        is kept explicitly so the result is bit-identical to the two-kernel form
        -- and to the baseline, which stores ``b + attn(b)`` in fp16 before the
        ffn reads it.
        """
        pid_p = tl.program_id(0)
        pid_b = tl.program_id(1)
        offs_p = pid_p * BP + tl.arange(0, BP)
        mask_p = offs_p < N
        offs_c = tl.arange(0, CH)
        y = offs_p // WD
        xx = offs_p % WD
        if PDL:
            gdc_wait()
        acc = tl.zeros((CH, BP), dtype=tl.float32)
        for k0 in tl.range(0, CH, BK):
            c = k0 + tl.arange(0, BK)
            vcol = (c // HD) * HW + 2 * KD + (c % HD)
            pre = tl.load(ATTN + pid_b * SB_ATTN + c[:, None] * N + offs_p[None, :],
                          mask=mask_p[None, :], other=0.0).to(tl.float32)
            pre += tl.load(BPE + c)[:, None]
            vp = QKV + pid_b * SB_QKV + vcol[:, None] * N + offs_p[None, :]
            for t in tl.static_range(9):
                dy = (t // 3) - 1
                dx = (t % 3) - 1
                m = mask_p & (y + dy >= 0) & (y + dy < H) & (xx + dx >= 0) & (xx + dx < WD)
                v = tl.load(vp + (dy * WD + dx), mask=m[None, :], other=0.0)
                pre += v.to(tl.float32) * tl.load(PEW + t * CH + c)[:, None]
            w = tl.load(W + offs_c[:, None] * CH + c[None, :])
            acc = tl.dot(w, pre.to(w.dtype), acc)

        p = IO + pid_b * SB_IO + offs_c[:, None] * N + offs_p[None, :]
        acc += tl.load(BIAS + offs_c)[:, None]
        acc += tl.load(p, mask=mask_p[None, :], other=0.0).to(tl.float32)
        b = acc.to(IO.dtype.element_ty)

        acc2 = tl.zeros((CH, BP), dtype=tl.float32)
        for f0 in tl.range(0, CF, BF):
            f = f0 + tl.arange(0, BF)
            w0 = tl.load(W0 + f[:, None] * CH + offs_c[None, :])
            h = tl.dot(w0, b) + tl.load(B0 + f)[:, None]
            h = h * tl.sigmoid(h)
            w1 = tl.load(W1 + offs_c[:, None] * CF + f[None, :])
            acc2 = tl.dot(w1, h.to(w1.dtype), acc2)
        acc2 += tl.load(B1 + offs_c)[:, None] + b.to(tl.float32)
        tl.store(p, acc2.to(IO.dtype.element_ty), mask=mask_p[None, :])
        if PDL:
            gdc_launch_dependents()

def _fold_bn(yc: YOLOConv):
    """Fold ``yc.bn`` into ``yc.conv``; returns (weight, bias) in fp32.

    Exact in eval mode: y = gamma * (conv(x) - mu) / sqrt(var + eps) + beta.
    Handles an already-fused YOLOConv (no ``bn``, conv carries a bias) too.
    """
    conv = yc.conv
    w = conv.weight.detach().float()
    bias = (conv.bias.detach().float() if conv.bias is not None
            else torch.zeros(w.shape[0], dtype=torch.float32, device=w.device))
    bn = getattr(yc, "bn", None)
    if bn is not None:
        scale = bn.weight.detach().float() / torch.sqrt(
            bn.running_var.detach().float() + bn.eps)
        w = w * scale.reshape(-1, *([1] * (w.dim() - 1)))
        bias = bn.bias.detach().float() + (bias - bn.running_mean.detach().float()) * scale
    return w, bias


class YOLOPSA(nn.Module):
    def __init__(self, c1: int, c2: int, e: float = 0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = YOLOConv(c1, 2 * self.c, 1, 1)
        self.cv2 = YOLOConv(2 * self.c, c1, 1, 1)
        self.attn = YOLOAttention(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
        self.ffn = nn.Sequential(
            YOLOConv(self.c, self.c * 2, 1, 1),
            YOLOConv(self.c * 2, self.c, 1, 1, act=False),
        )
        self._plan = None      # (shape, dtype, device, weight-version, launch, out)
        self._steps = None      # per-stage launch thunks, for profiling
        self._no_fast = False   # set if the fused path cannot serve this config

    # -- reference path -----------------------------------------------------
    def _eager(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))

    # -- fused path ---------------------------------------------------------
    def _weights(self, dtype: torch.dtype):
        """Folded weights, each already laid out as the GEMM's [Cout, Cin]."""
        att = self.attn
        w = {}
        for name, mod in (("cv1", self.cv1), ("cv2", self.cv2), ("qkv", att.qkv),
                          ("proj", att.proj), ("f0", self.ffn[0]), ("f1", self.ffn[1])):
            wt, bs = _fold_bn(mod)
            w["w_" + name] = wt.reshape(wt.shape[0], -1).contiguous().to(dtype)
            w["b_" + name] = bs.contiguous()
        wpe, bpe = _fold_bn(att.pe)
        w["w_pe"] = wpe.reshape(self.c, 9).t().contiguous()   # [9, CH]
        w["b_pe"] = bpe.contiguous()
        return w

    def _build(self, x: torch.Tensor):
        bsz, cin, h, wd = x.shape
        n = h * wd
        ch, c2 = self.c, 2 * self.c
        cout = self.cv2.conv.weight.shape[0]
        att = self.attn
        kd, hd, nh = att.key_dim, att.head_dim, att.num_heads
        hw = 2 * kd + hd
        w = self._weights(x.dtype)
        opts = dict(dtype=x.dtype, device=x.device)

        cat = torch.empty((bsz, c2, n), **opts)      # [a | b], also the cv2 input
        qkv = torch.empty((bsz, hw * nh, n), **opts)
        attn = torch.empty((bsz, ch, n), **opts)
        out = torch.empty((bsz, cout, h, wd), **opts)
        bhalf = cat[:, ch:]                          # updated in place by _tail_kernel

        def gemm(inp, tag, kin, nco, o, sb_in, sb_out, silu, res=None, sb_res=0,
                 step=0):
            BC, BP, nw, ns = _CFG[tag]
            # keep the [BC, kin] weight tile inside a double-bufferable slice of
            # SMEM, and make BC divide the output channel count
            BC = min(BC, nco, max(16, 16384 // kin))
            while nco % BC:
                BC //= 2
            grid = (triton.cdiv(n, BP), nco // BC, bsz)
            src = inp if inp is not None else x
            return lambda t: _gemm_kernel[grid](
                t if inp is None else src, w["w_" + tag], w["b_" + tag],
                res if res is not None else src, o, n, sb_in, sb_res, sb_out,
                KIN=kin, BC=BC, BP=BP, SILU=silu, ADD_RES=res is not None,
                PDL=_PDL and step > 0, num_warps=nw, num_stages=ns,
                launch_pdl=_PDL and step > 0)

        steps = [
            gemm(None, "cv1", cin, c2, cat, cin * n, c2 * n, True, step=0),
            gemm(bhalf, "qkv", ch, hw * nh, qkv, c2 * n, hw * nh * n, False, step=1),
        ]

        # NB: each step's launch params get their own names -- a lambda that
        # closed over shared names would see whatever the *last* step rebound
        # them to.
        bp_a, bn_a, nw_a, ns_a = _CFG["attn"]
        grid_a = (triton.cdiv(n, bp_a), nh, bsz)
        steps.append(lambda t: _attn_kernel[grid_a](
            qkv, attn, n, att.scale, hw * nh * n, ch * n,
            KD=kd, HD=hd, HW=hw, BP=bp_a, BN=bn_a, PDL=_PDL,
            num_warps=nw_a, num_stages=ns_a, launch_pdl=_PDL))

        bp_t, bk_t, bf_t, nw_t, ns_t = _CFG["tail"]
        bf_t = min(bf_t, 2 * ch, max(16, 16384 // ch))
        grid_t = (triton.cdiv(n, bp_t), bsz)
        steps.append(lambda t: _tail_kernel[grid_t](
            qkv, attn, w["w_pe"], w["b_pe"], w["w_proj"], w["b_proj"],
            w["w_f0"], w["b_f0"], w["w_f1"], w["b_f1"], bhalf,
            n, h, wd, hw * nh * n, ch * n, c2 * n,
            KD=kd, HD=hd, HW=hw, CH=ch, CF=2 * ch, BP=bp_t,
            BK=min(bk_t, ch), BF=bf_t, PDL=_PDL,
            num_warps=nw_t, num_stages=ns_t, launch_pdl=_PDL))

        steps.append(gemm(cat, "cv2", c2, cout, out, c2 * n, cout * n, True,
                          step=len(steps)))

        def launch(t: torch.Tensor):
            for step in steps:
                step(t)
            return out

        self._steps = steps    # dev/stages2.py times prefixes of this
        return launch, out

    def _wversion(self):
        # Re-fold if the parameters were replaced or written in place (e.g. a
        # state_dict loaded after the first forward).
        p = self.cv1.conv.weight
        return (p.data_ptr(), p._version)

    def _fast_ok(self, x: torch.Tensor) -> bool:
        """The fused path assumes eval mode, no autograd, fp16/bf16 CUDA input,
        and channel counts that are valid Triton tile bounds."""
        att = self.attn
        dims = (x.shape[1], 2 * self.c, self.c, att.key_dim, att.head_dim,
                (2 * att.key_dim + att.head_dim) * att.num_heads,
                self.cv2.conv.weight.shape[0])
        return (_HAS_TRITON and not self._no_fast and not self.training
                and not torch.is_grad_enabled()
                and x.is_cuda and x.dim() == 4
                and x.dtype in (torch.float16, torch.bfloat16)
                and x.shape[1] == self.cv1.conv.weight.shape[1]
                and all(d % 16 == 0 for d in dims)
                and all(d & (d - 1) == 0 for d in (self.c, att.key_dim, att.head_dim))
                and self.c == att.num_heads * att.head_dim)

    def _first(self, x: torch.Tensor) -> torch.Tensor:
        if not self._fast_ok(x):
            return self._eager(x)
        xc = x.contiguous()
        try:
            launch, out = self._build(xc)
            res = launch(xc)
        except Exception:
            # e.g. a channel count whose tiles do not fit in SMEM: stop trying
            self._plan = None
            self._no_fast = True
            return self._eager(x)
        self._plan = (xc.shape, xc.dtype, xc.device, self._wversion(), launch, out)
        return res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        plan = self._plan
        if (plan is not None and x.shape == plan[0] and x.dtype == plan[1]
                and x.device == plan[2] and x.is_contiguous()
                and not self.training and not torch.is_grad_enabled()
                and self._wversion() == plan[3]):
            return plan[4](x)
        return self._first(x)
