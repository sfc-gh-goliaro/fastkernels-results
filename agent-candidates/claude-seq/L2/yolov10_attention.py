"""YOLOv10 spatial attention block -- fused Triton implementation.

The captured shapes are tiny ([1|4, 128, 20, 20], 2 heads, key_dim 32,
head_dim 64, 400 spatial positions), so the eager block is almost pure
overhead: three Conv2d+BatchNorm pairs, two batched GEMMs, a 400x400 softmax
and a pile of elementwise kernels -- ~18 launches for ~0.4 GFLOP of work.

Here the block is two launches:

* ``_qkv_kernel``  -- the 1x1 qkv conv (BatchNorm folded into the weight) as a
  plain ``[256,128] @ [128,N]`` GEMM per image, scattered straight into the
  layout the attention kernel wants: ``q`` row-major ``[b,N,2*DK]``, ``k``
  channel-major ``[b,2,DK,N]``, ``v`` row-major ``[b,N,C]`` (which is also
  ``v.reshape(b, C, h, w)``, so the positional encoding can read it back).
* ``_attn_kernel`` -- everything else, per (image, query tile): flash-style
  ``softmax(q^T k * scale) v`` over both heads without ever materialising the
  400x400 score matrix, the depth-wise 3x3 positional encoding on ``v``, the
  residual add, and the 1x1 output projection.

Only ``qkv`` and the final ``[b, C, h, w]`` tensor reach HBM.  At these sizes
the launch path is a first-class cost, so both kernels are pre-bound once per
batch size (see :class:`_FastLaunch`) and the scratch buffer is reused.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.softmax import Softmax
from .yolov10_conv import YOLOConv

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # pragma: no cover - no triton -> eager fallback
    _HAVE_TRITON = False


if _HAVE_TRITON:

    @triton.jit
    def _qkv_kernel(X, W, BIAS, BUF, OFF_Q, OFF_K, OFF_V, N,
                    CIN: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
                    BP: tl.constexpr):
        """qkv = W @ x + bias, written out as q / k / v in attention layout.

        Output channel block ``pid_c`` of 32 maps to a (head, role) pair:
        ``0 -> q, 1 -> k, 2 -> v[0:32], 3 -> v[32:64]`` within each head.
        """
        BC: tl.constexpr = 32
        HD: tl.constexpr = 2 * DK + DV
        pid_p = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_b = tl.program_id(2)

        head = pid_c // 4
        role = pid_c % 4
        oc0 = head * HD + role * BC

        offs_p = pid_p * BP + tl.arange(0, BP)
        offs_c = tl.arange(0, BC)
        offs_ic = tl.arange(0, CIN)
        pm = offs_p < N

        w = tl.load(W + (oc0 + offs_c)[:, None] * CIN + offs_ic[None, :])
        x = tl.load(X + pid_b * (CIN * N) + offs_ic[:, None] * N + offs_p[None, :],
                    mask=pm[None, :], other=0.0)
        acc = tl.dot(w, x, out_dtype=tl.float32)
        acc += tl.load(BIAS + oc0 + offs_c)[:, None]

        if role == 1:
            # k: channel-major [DK, N] per head -- the B operand of q @ k^T.
            K = BUF + OFF_K
            tl.store(K + pid_b * (2 * DK * N) + head * (DK * N)
                     + offs_c[:, None] * N + offs_p[None, :],
                     acc.to(K.dtype.element_ty), mask=pm[None, :])
        else:
            accT = tl.trans(acc)
            if role == 0:
                # q: row-major [N, 2*DK]
                Q = BUF + OFF_Q
                tl.store(Q + pid_b * (N * 2 * DK) + offs_p[:, None] * (2 * DK)
                         + (head * DK + offs_c)[None, :],
                         accT.to(Q.dtype.element_ty), mask=pm[:, None])
            else:
                # v: row-major [N, C] -- the same buffer as v.reshape(b, C, h, w)
                V = BUF + OFF_V
                col = head * DV + (role - 2) * BC + offs_c
                tl.store(V + pid_b * (N * 2 * DV) + offs_p[:, None] * (2 * DV)
                         + col[None, :],
                         accT.to(V.dtype.element_ty), mask=pm[:, None])

    @triton.jit
    def _attn_kernel(BUF, OFF_Q, OFF_K, OFF_V, WPE, BPE, WPROJT, BPROJ, OUT,
                     N, H, W, scale,
                     DK: tl.constexpr, DV: tl.constexpr,
                     BQ: tl.constexpr, BK: tl.constexpr, BOC: tl.constexpr):
        """Two-head attention + depth-wise 3x3 pe + residual + 1x1 proj."""
        C: tl.constexpr = 2 * DV
        Q = BUF + OFF_Q
        K = BUF + OFF_K
        V = BUF + OFF_V
        pid_q = tl.program_id(0)
        pid_b = tl.program_id(1)

        offs_q = pid_q * BQ + tl.arange(0, BQ)
        qm = offs_q < N
        offs_dk = tl.arange(0, DK)
        offs_dv = tl.arange(0, DV)

        qb = Q + pid_b * (N * 2 * DK) + offs_q[:, None] * (2 * DK)
        q0 = tl.load(qb + offs_dk[None, :], mask=qm[:, None], other=0.0)
        q1 = tl.load(qb + (DK + offs_dk)[None, :], mask=qm[:, None], other=0.0)
        kb = K + pid_b * (2 * DK * N) + offs_dk[:, None] * N
        vb = V + pid_b * (N * C)

        neg_inf = float("-inf")
        m0 = tl.full([BQ], neg_inf, tl.float32)
        m1 = tl.full([BQ], neg_inf, tl.float32)
        l0 = tl.zeros([BQ], tl.float32)
        l1 = tl.zeros([BQ], tl.float32)
        acc0 = tl.zeros([BQ, DV], tl.float32)
        acc1 = tl.zeros([BQ, DV], tl.float32)

        for j0 in range(0, N, BK):
            offs_j = j0 + tl.arange(0, BK)
            jm = offs_j < N
            k0 = tl.load(kb + offs_j[None, :], mask=jm[None, :], other=0.0)
            k1 = tl.load(kb + (DK * N) + offs_j[None, :], mask=jm[None, :], other=0.0)
            s0 = tl.dot(q0, k0, out_dtype=tl.float32) * scale
            s1 = tl.dot(q1, k1, out_dtype=tl.float32) * scale
            s0 = tl.where(jm[None, :], s0, neg_inf)
            s1 = tl.where(jm[None, :], s1, neg_inf)

            m0n = tl.maximum(m0, tl.max(s0, 1))
            m1n = tl.maximum(m1, tl.max(s1, 1))
            p0 = tl.exp(s0 - m0n[:, None])
            p1 = tl.exp(s1 - m1n[:, None])
            a0 = tl.exp(m0 - m0n)
            a1 = tl.exp(m1 - m1n)
            l0 = l0 * a0 + tl.sum(p0, 1)
            l1 = l1 * a1 + tl.sum(p1, 1)

            vrow = vb + offs_j[:, None] * C
            v0 = tl.load(vrow + offs_dv[None, :], mask=jm[:, None], other=0.0)
            v1 = tl.load(vrow + (DV + offs_dv)[None, :], mask=jm[:, None], other=0.0)
            acc0 = acc0 * a0[:, None] + tl.dot(p0.to(v0.dtype), v0, out_dtype=tl.float32)
            acc1 = acc1 * a1[:, None] + tl.dot(p1.to(v1.dtype), v1, out_dtype=tl.float32)
            m0 = m0n
            m1 = m1n

        t0 = acc0 / l0[:, None] + tl.load(BPE + offs_dv)[None, :]
        t1 = acc1 / l1[:, None] + tl.load(BPE + DV + offs_dv)[None, :]

        # depth-wise 3x3 positional encoding on v (padding 1), read back from V
        y = offs_q // W
        xx = offs_q - y * W
        for ky in tl.static_range(3):
            for kx in tl.static_range(3):
                yy = y + (ky - 1)
                xs = xx + (kx - 1)
                ok = qm & (yy >= 0) & (yy < H) & (xs >= 0) & (xs < W)
                grow = vb + (yy * W + xs)[:, None] * C
                g0 = tl.load(grow + offs_dv[None, :], mask=ok[:, None], other=0.0)
                g1 = tl.load(grow + (DV + offs_dv)[None, :], mask=ok[:, None], other=0.0)
                k9 = ky * 3 + kx
                t0 += g0 * tl.load(WPE + offs_dv * 9 + k9)[None, :]
                t1 += g1 * tl.load(WPE + (DV + offs_dv) * 9 + k9)[None, :]

        h0 = t0.to(WPROJT.dtype.element_ty)
        h1 = t1.to(WPROJT.dtype.element_ty)
        # 1x1 projection, one output-channel chunk at a time: a full [C, C]
        # weight tile in registers is what pushes this kernel into spills.
        ob = OUT + pid_b * (C * N)
        for oc0 in range(0, C, BOC):
            offs_oc = oc0 + tl.arange(0, BOC)
            wp0 = tl.load(WPROJT + offs_dv[:, None] * C + offs_oc[None, :])
            wp1 = tl.load(WPROJT + (DV + offs_dv)[:, None] * C + offs_oc[None, :])
            o = tl.dot(h0, wp0, out_dtype=tl.float32)
            o = tl.dot(h1, wp1, acc=o, out_dtype=tl.float32)
            o += tl.load(BPROJ + offs_oc)[None, :]
            tl.store(ob + offs_oc[:, None] * N + offs_q[None, :],
                     tl.trans(o).to(OUT.dtype.element_ty), mask=qm[None, :])


class _FastLaunch:
    """A Triton kernel pre-bound to everything except its changing pointers.

    ``JITFunction.__getitem__`` re-derives the specialization key, re-binds the
    signature and rebuilds launch metadata on *every* call -- ~11us of Python
    for a kernel that runs in ~4us.  Only a couple of pointers actually change
    between calls here, so bind once and keep just the C launcher plus its
    argument list; constant tensors are collapsed to raw addresses so the
    launcher has nothing to unwrap.
    """

    __slots__ = ("_run", "_fn", "_pm", "_g0", "_g1", "_g2", "_vals", "_slots")

    def __init__(self, jit_fn, grid, args, kwargs, live_names):
        from triton.runtime import driver

        kernel = jit_fn[grid](*args, **kwargs)
        binder = jit_fn.device_caches[driver.active.get_current_device()][-1]
        bound, _spec, _opts = binder(*args, **kwargs)
        names = list(bound.keys())
        live = set(live_names)
        self._vals = [
            v if (n in live or not torch.is_tensor(v)) else v.data_ptr()
            for n, v in zip(names, bound.values())
        ]
        self._slots = [names.index(n) for n in live_names]
        self._run = kernel.run
        self._fn = kernel.function
        self._pm = kernel.packed_metadata
        g = tuple(grid) + (1, 1)
        self._g0, self._g1, self._g2 = g[0], g[1], g[2]

    def __call__(self, stream, *live):
        vals = self._vals
        for i, t in zip(self._slots, live):
            vals[i] = t
        self._run(self._g0, self._g1, self._g2, stream, self._fn, self._pm,
                  None, None, None, *vals)


def _fold_bn(conv: nn.Module, bn: nn.Module | None):
    """Conv(+BN) -> (weight, bias) in fp32."""
    w = conv.weight.detach().float()
    b = conv.bias.detach().float() if conv.bias is not None else torch.zeros(
        w.shape[0], device=w.device, dtype=torch.float32)
    if bn is None:
        return w, b
    g = bn.weight.detach().float() / torch.sqrt(bn.running_var.detach().float() + bn.eps)
    b = bn.bias.detach().float() + g * (b - bn.running_mean.detach().float())
    return w * g.reshape(-1, *([1] * (w.dim() - 1))), b


class YOLOAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = YOLOConv(dim, h, 1, act=False)
        self.proj = YOLOConv(dim, dim, 1, act=False)
        self.pe = YOLOConv(dim, dim, 3, 1, g=dim, act=False)
        self._softmax = Softmax(dim=-1)
        self._plan = None
        self._launchers = {}

    # -- setup ---------------------------------------------------------------
    def _build_plan(self, x: torch.Tensor):
        dim = self.head_dim * self.num_heads
        # _qkv_kernel scatters in 32-channel groups, one per (head, role), which
        # only lines up for 2 heads with key_dim 32 / head_dim 64.
        if (not _HAVE_TRITON or not x.is_cuda or x.dtype != torch.float16
                or self.num_heads != 2 or self.key_dim != 32
                or self.head_dim != 64 or x.shape[1] != dim):
            self._plan = False
            return False
        wq, bq = _fold_bn(self.qkv.conv, getattr(self.qkv, "bn", None))
        wp, bp = _fold_bn(self.proj.conv, getattr(self.proj, "bn", None))
        we, be = _fold_bn(self.pe.conv, getattr(self.pe, "bn", None))
        self._plan = (
            wq.reshape(wq.shape[0], -1).half().contiguous(),
            bq.contiguous(),
            we.reshape(dim, 9).half().contiguous(),
            be.contiguous(),
            wp.reshape(dim, dim).t().half().contiguous(),
            bp.contiguous(),
        )
        return True

    def _build_launchers(self, x: torch.Tensor):
        b, c, h, w = x.shape
        n = h * w
        dk = self.key_dim
        dv = self.head_dim
        wq, bq, we, be, wpt, bp = self._plan
        off_q = 0
        off_k = off_q + b * n * 2 * dk
        off_v = off_k + b * 2 * dk * n
        nbuf = off_v + b * n * c
        buf = torch.empty(nbuf, device=x.device, dtype=x.dtype)
        out = torch.empty((b, c, h, w), device=x.device, dtype=x.dtype)
        k1 = _FastLaunch(
            _qkv_kernel, (-(-n // 128), 4 * self.num_heads, b),
            (x, wq, bq, buf, off_q, off_k, off_v, n),
            dict(CIN=c, DK=dk, DV=dv, BP=128, num_warps=8, num_stages=1),
            ("X",),
        )
        k2 = _FastLaunch(
            _attn_kernel, (-(-n // 16), b),
            (buf, off_q, off_k, off_v, we, be, wpt, bp, out, n, h, w, self.scale),
            dict(DK=dk, DV=dv, BQ=16, BK=256, BOC=64, num_warps=8, num_stages=2),
            ("OUT",),
        )
        shape = (b, c, h, w)
        dtype = x.dtype
        device = x.device
        get_stream = torch._C._cuda_getCurrentRawStream
        dev_index = x.device.index

        def launch(xt, _k1=k1, _k2=k2, _gs=get_stream, _di=dev_index,
                   _scratch=buf, _shape=shape, _dtype=dtype, _device=device):
            stream = _gs(_di)
            outi = torch.empty(_shape, device=_device, dtype=_dtype)
            _k1(stream, xt)
            _k2(stream, outi)
            return outi

        self._launchers[b] = launch
        return launch

    def _reference(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        qkv = self.qkv(x)
        q, k, v = qkv.view(b, self.num_heads, self.key_dim * 2 + self.head_dim, n).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = self._softmax(attn)
        x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.pe(v.reshape(b, c, h, w))
        return self.proj(x)

    # -- forward -------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        plan = self._plan
        if plan is None:
            plan = self._build_plan(x) and self._plan
        if plan is False:
            return self._reference(x)
        if not x.is_contiguous():
            x = x.contiguous()
        elif x.data_ptr() & 15:
            # the pre-bound launch is compiled against 16B-aligned pointers
            x = x.clone()
        try:
            launch = self._launchers.get(x.shape[0])
            if launch is None:
                launch = self._build_launchers(x)
            return launch(x)
        except Exception:
            return self._reference(x)
