"""Triangle attention for AlphaFold3 (L2), fused into one cooperative kernel.

Implements AF3 Algorithms 14 (starting node) and 15 (ending node).
Self-attention over one dimension of the pair representation with a
learned triangle bias from the other dimension.

Reference: openfold3/core/model/layers/triangular_attention.py TriangleAttention

The captured shape is tiny (``x[1, 16, 16, 128]``, 4 heads of 32): ~40 MFLOP of
work spread over ~30 eager kernel launches. Everything -- layer norm, the
triangle-bias projection, q/k/v/g, softmax attention with both biases, the
sigmoid gate and the output projection -- is fused into a single launch, with the
weights pre-packed once.

What is then left to pay for is memory: feeding a weight tile to ``tl.dot`` runs
at roughly 32 GB/s per SM (the tile has to land in registers and then be staged
into shared memory), and that rate is per-SM, not global. So the 160 KB of
q/k/v/g/o weights, which every program needs in full, is the whole cost. The
fast path (``_tri_attn_split``) therefore runs one program per (batch, row, head)
and gives each only its own head's 40 KB, at the price of two grid-wide
rendezvous: one to share the triangle bias (a function of the *whole* pair
representation) and one to share the gated hidden vector before the output
projection. ``_tri_attn_fused`` is the one-program-per-row fallback for shapes
where that does not fit.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from .alphafold3_of3_attention import OF3Attention

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None


def _permute_final_dims(tensor: torch.Tensor, inds: tuple[int, ...]) -> torch.Tensor:
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


if triton is not None:

    @triton.jit(do_not_specialize=["gen"])
    def _tri_attn_fused(
        X, LNP, WP, WZ, WOT, MASK, OUT, ZT, FLAG,
        sxb, sxi, sxj, sob, soi, soj, smb, smi, smk, gen,
        eps, inf, scale,
        N: tl.constexpr, C: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
        HP: tl.constexpr, G: tl.constexpr, GP: tl.constexpr,
        HAS_MASK: tl.constexpr, CONTIG: tl.constexpr, COOP: tl.constexpr,
    ):
        """One program per (batch, i): rows ``x[b, i, :, :]`` of the pair rep.

        The triangle bias is a function of the *whole* pair representation, so
        every program needs all of it. Rather than have all G programs redo that
        work, each program projects the rows it already layer-normed for its own
        q/k/v, publishes them, and waits on a grid-wide flag barrier (``COOP``).
        """
        DT = X.dtype.element_ty
        HD: tl.constexpr = H * D
        NR: tl.constexpr = N * N
        RP: tl.constexpr = HP // H

        pid = tl.program_id(0)
        b = pid // N
        i = pid % N

        c = tl.arange(0, C)
        hd = tl.arange(0, HD)
        hh = tl.arange(0, H)
        jj = tl.arange(0, N)
        xb = X + b * sxb

        # ---- weights first: the staging is bandwidth bound, so it wants to be
        # in flight across the layer norm and (under COOP) the barrier wait.
        wq = tl.load(WP + c[:, None] * HD + hd[None, :])
        wk = tl.load(WP + C * HD + c[:, None] * HD + hd[None, :])
        wv = tl.load(WP + 2 * C * HD + c[:, None] * HD + hd[None, :])
        wg = tl.load(WP + 3 * C * HD + c[:, None] * HD + hd[None, :])
        wo = tl.load(WOT + hd[:, None] * C + c[None, :])

        # ---- this program's own rows, layer-normed
        xo = tl.load(xb + i * sxi + jj[:, None] * sxj + c[None, :]).to(tl.float32)
        muo = tl.sum(xo, 1) * (1.0 / C)
        xo = xo - muo[:, None]
        rso = 1.0 / tl.sqrt(tl.sum(xo * xo, 1) * (1.0 / C) + eps)
        xn = (xo * rso[:, None] * tl.load(LNP + c)[None, :]
              + tl.load(LNP + C + c)[None, :]).to(DT)

        # ---- triangle bias. WZ holds linear_z replicated RP times across the
        # padded width (a dot needs at least 16 columns), so summing the copies
        # both restores the projection and dodges a register-tensor slice.
        wzp = tl.load(WZ + c[:, None] * HP + tl.arange(0, HP)[None, :])
        if COOP:
            zr = tl.sum(tl.reshape(tl.dot(xn, wzp), (N, RP, H)), 1)
            tl.store(ZT + b * H * NR + hh[None, :] * NR + i * N + jj[:, None], zr)
            tl.debug_barrier()
            tl.atomic_xchg(FLAG + pid + tl.zeros((1,), tl.int32), gen,
                           sem="release", scope="gpu")

        if HAS_MASK:
            m = tl.load(MASK + b * smb + i * smi + jj * smk).to(tl.float32)
            mb = inf * (m - 1.0)
        else:
            mb = tl.zeros((N,), tl.float32)

        if COOP:
            g = tl.arange(0, GP)
            done = 0
            while done == 0:
                f = tl.load(FLAG + g, mask=g < G, other=gen, volatile=True)
                done = (tl.min(f) >= gen).to(tl.int32)
            zt = tl.load(ZT + b * H * NR + hh[:, None, None] * NR
                         + jj[None, :, None] * N + jj[None, None, :], volatile=True)
        else:
            rr = tl.arange(0, NR)
            if CONTIG:
                ra = rr[:, None] * C
            else:
                ra = (rr // N)[:, None] * sxi + (rr % N)[:, None] * sxj
            xa = tl.load(xb + ra + c[None, :]).to(tl.float32)
            mua = tl.sum(xa, 1) * (1.0 / C)
            xa = xa - mua[:, None]
            rsa = 1.0 / tl.sqrt(tl.sum(xa * xa, 1) * (1.0 / C) + eps)
            xna = (xa * rsa[:, None] * tl.load(LNP + c)[None, :]
                   + tl.load(LNP + C + c)[None, :]).to(DT)
            za = tl.sum(tl.reshape(tl.dot(xna, wzp), (NR, RP, H)), 1)
            zt = tl.trans(tl.reshape(za, (N, N, H)), 2, 0, 1)

        # ---- q/k/v/g, then one batched dot for the scores of every head
        q3 = tl.trans(tl.reshape((tl.dot(xn, wq) * scale).to(DT), (N, H, D)), 1, 0, 2)
        k3 = tl.trans(tl.reshape(tl.dot(xn, wk).to(DT), (N, H, D)), 1, 2, 0)
        v3 = tl.trans(tl.reshape(tl.dot(xn, wv).to(DT), (N, H, D)), 1, 0, 2)
        gt = tl.dot(xn, wg)

        s = tl.dot(q3, k3) + zt + mb[None, None, :]
        s = tl.exp(s - tl.max(s, 2)[:, :, None])
        p = (s / tl.sum(s, 2)[:, :, None]).to(DT)
        o = tl.reshape(tl.trans(tl.dot(p, v3), 1, 0, 2), (N, HD)) * tl.sigmoid(gt)
        acc = tl.dot(o.to(DT), wo)

        tl.store(OUT + b * sob + i * soi + jj[:, None] * soj + c[None, :],
                 acc.to(OUT.dtype.element_ty))


    @triton.jit(do_not_specialize=["gen"])
    def _tri_attn_split(
        X, LNP, WPH, WZ, WOT, MASK, OUT, ZT, OG, CNT,
        sxb, sxi, sxj, sob, soi, soj, smb, smi, smk, gen,
        eps, inf, scale,
        N: tl.constexpr, C: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
        HP: tl.constexpr, G: tl.constexpr, HAS_MASK: tl.constexpr,
    ):
        """One program per (batch, i, head).

        Feeding a weight tile to a dot runs at roughly 32 GB/s per SM (it has to
        land in registers and then be staged into shared memory), so the 160 KB
        of q/k/v/g/o weights is what this kernel costs. Splitting the heads over
        H programs cuts that to a quarter per SM; the price is two grid-wide
        rendezvous, one to share the triangle bias and one to share the gated
        hidden vector before the output projection (whose columns are split H
        ways, so every program ends with an equal share of the work).
        """
        DT = X.dtype.element_ty
        HD: tl.constexpr = H * D
        NR: tl.constexpr = N * N
        RP: tl.constexpr = HP // H
        CS: tl.constexpr = C // H
        D4: tl.constexpr = 4 * D

        pid = tl.program_id(0)
        h = pid % H
        r = pid // H
        i = r % N
        b = r // N

        c = tl.arange(0, C)
        jj = tl.arange(0, N)
        hh = tl.arange(0, H)
        d = tl.arange(0, D)
        hd = tl.arange(0, HD)
        cs = tl.arange(0, CS)

        # ---- this head's slice of the weights, issued first
        wph = tl.load(WPH + h * C * D4 + c[:, None] * D4 + tl.arange(0, D4)[None, :])
        wos = tl.load(WOT + hd[:, None] * C + (h * CS + cs)[None, :])

        # ---- own rows, layer-normed
        xo = tl.load(X + b * sxb + i * sxi + jj[:, None] * sxj
                     + c[None, :]).to(tl.float32)
        muo = tl.sum(xo, 1) * (1.0 / C)
        xo = xo - muo[:, None]
        rso = 1.0 / tl.sqrt(tl.sum(xo * xo, 1) * (1.0 / C) + eps)
        xn = (xo * rso[:, None] * tl.load(LNP + c)[None, :]
              + tl.load(LNP + C + c)[None, :]).to(DT)

        # ---- publish this row of the triangle bias
        wzp = tl.load(WZ + c[:, None] * HP + tl.arange(0, HP)[None, :])
        zr = tl.sum(tl.reshape(tl.dot(xn, wzp), (N, RP, H)), 1)
        tl.store(ZT + b * H * NR + hh[None, :] * NR + i * N + jj[:, None], zr,
                 mask=hh[None, :] == h)
        tl.debug_barrier()
        tl.atomic_add(CNT + tl.zeros((1,), tl.int32), 1, sem="release", scope="gpu")

        if HAS_MASK:
            m = tl.load(MASK + b * smb + i * smi + jj * smk).to(tl.float32)
            mb = inf * (m - 1.0)
        else:
            mb = tl.zeros((N,), tl.float32)

        # ---- q/k/v/g for this head (interleaved along the last axis)
        e0, e1 = tl.split(tl.reshape(tl.dot(xn, wph), (N, D, 2, 2)))
        q, v = tl.split(e0)
        k, g = tl.split(e1)

        done = 0
        while done == 0:
            done = (tl.max(tl.load(CNT + tl.zeros((1,), tl.int32), volatile=True))
                    >= gen * G).to(tl.int32)
        zt = tl.load(ZT + b * H * NR + h * NR + jj[:, None] * N + jj[None, :],
                     volatile=True)

        sc = tl.dot((q * scale).to(DT), tl.trans(k.to(DT))) + zt + mb[None, :]
        sc = tl.exp(sc - tl.max(sc, 1)[:, None])
        p = (sc / tl.sum(sc, 1)[:, None]).to(DT)
        og = (tl.dot(p, v.to(DT)) * tl.sigmoid(g)).to(DT)

        # ---- publish o*g, then take an equal slice of the output projection
        ogb = OG + (r * N + jj[:, None]) * HD
        tl.store(ogb + (h * D + d)[None, :], og)
        tl.debug_barrier()
        tl.atomic_add(CNT + 1 + r + tl.zeros((1,), tl.int32), 1, sem="release",
                      scope="gpu")
        done = 0
        while done == 0:
            done = (tl.max(tl.load(CNT + 1 + r + tl.zeros((1,), tl.int32),
                                   volatile=True)) >= gen * H).to(tl.int32)
        acc = tl.dot(tl.load(ogb + hd[None, :], volatile=True), wos)

        tl.store(OUT + b * sob + i * soi + jj[:, None] * soj
                 + (h * CS + cs)[None, :], acc.to(OUT.dtype.element_ty))


def _pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


class TriangleAttention(nn.Module):
    """AF3 Algorithms 14/15: Triangle attention."""

    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        no_heads: int,
        starting: bool = True,
        inf: float = 1e9,
    ):
        super().__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf

        self.layer_norm = LayerNorm(c_in)
        self.linear_z = Linear(c_in, no_heads, bias=False)

        self.mha = OF3Attention(
            c_q=c_in,
            c_k=c_in,
            c_v=c_in,
            c_hidden=c_hidden,
            no_heads=no_heads,
        )

        self._pack: tuple | None = None
        self._pack_key: tuple | None = None
        self._coop: tuple | None = None
        self._gen = 0
        # Shapes Triton could not compile the fused kernel for (it runs out of
        # tensor memory once N gets large); those fall back to the eager path.
        self._nofuse: set | None = None

    def _pack_key_now(self) -> tuple:
        a = self.mha
        ps = (self.layer_norm.weight, self.layer_norm.bias, self.linear_z.weight,
              a.linear_q.weight, a.linear_k.weight, a.linear_v.weight,
              a.linear_g.weight, a.linear_o.weight)
        # Identity *and* version: ``load_state_dict`` copies into the existing
        # parameters, so identity alone would keep serving a stale pack.
        return tuple((p, None if p is None else p._version) for p in ps)

    def _packed(self, dtype: torch.dtype, device: torch.device):
        key = self._pack_key_now()
        cached = self._pack
        if cached is not None and self._pack_key == key:
            return cached
        c, h = self.c_in, self.no_heads
        hp = max(16, triton.next_power_of_2(h))
        rp = hp // h
        lw, lb = self.layer_norm.weight, self.layer_norm.bias
        with torch.no_grad():
            f32 = torch.float32
            a = self.mha
            wp = torch.stack([a.linear_q.weight.t(), a.linear_k.weight.t(),
                              a.linear_v.weight.t(), a.linear_g.weight.t()])
            wp = wp.to(dtype).contiguous()
            wot = a.linear_o.weight.t().to(dtype).contiguous()
            wz = self.linear_z.weight.t().to(f32).div(rp).to(dtype)
            wz = wz.repeat(1, rp).contiguous()
            lnp = torch.empty(2 * c, dtype=f32, device=device)
            lnp[:c] = 1.0 if lw is None else lw.to(f32)
            lnp[c:] = 0.0 if lb is None else lb.to(f32)
            ws = torch.stack([a.linear_q.weight.t(), a.linear_k.weight.t(),
                              a.linear_v.weight.t(), a.linear_g.weight.t()])
            hdim = ws.shape[-1]
            wph = ws.reshape(4, c, h, hdim // h).permute(2, 1, 3, 0)
            wph = wph.reshape(h, c, -1).to(dtype).contiguous()
        self._pack = (lnp, wp, wz, wot, hp, wph)
        self._pack_key = key
        return self._pack

    def _fusable(self, x: torch.Tensor, mask: torch.Tensor | None) -> bool:
        if triton is None or not x.is_cuda or x.dtype not in (torch.bfloat16,
                                                             torch.float16):
            return False
        if x.dim() < 3 or x.stride(-1) != 1 or x.shape[-1] != self.c_in:
            return False
        n = x.shape[-2]
        h, d = self.no_heads, self.c_hidden
        if x.shape[-3] != n or n < 16 or n > 64:
            return False
        if not (_pow2(n) and _pow2(self.c_in) and _pow2(h) and _pow2(d)):
            return False
        if d < 16 or self.c_in < 16 or h * d != self.c_in:
            return False
        if mask is not None and (mask.dim() != x.dim() - 1
                                 or mask.shape[-1] != n or mask.shape[-2] != n):
            return False
        return True

    def _fused_forward(self, x: torch.Tensor, mask: torch.Tensor | None):
        n, c = x.shape[-2], x.shape[-1]
        lead = x.shape[:-3]
        b = 1
        for s in lead:
            b *= s
        sxi, sxj = x.stride(-3), x.stride(-2)
        sxb = x.stride(-4) if x.dim() > 3 else n * n * c
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        if self.starting:
            soi, soj = n * c, c
        else:
            sxi, sxj = sxj, sxi
            soi, soj = c, n * c
        lnp, wp, wz, wot, hp, wph = self._packed(x.dtype, x.device)
        if mask is None:
            mptr, smb, smi, smk = x, 0, 0, 0
        else:
            mptr = mask
            smi, smk = mask.stride(-2), mask.stride(-1)
            if not self.starting:
                smi, smk = smk, smi
            smb = mask.stride(-3) if mask.dim() > 2 else n * n
        h = self.no_heads
        gr = b * n
        if self._nofuse is not None and (gr, n) in self._nofuse:
            return None
        coop = self._coop
        if coop is None or coop[0] != (gr, b, h, n, x.device):
            nsm = torch.cuda.get_device_properties(x.device).multi_processor_count
            # One program per (batch, i, head) needs every program resident for
            # the two rendezvous, and an equal share of linear_o's columns.
            if gr * h <= nsm and c % h == 0 and c // h >= 16:
                zt = torch.empty(b * h * n * n, dtype=torch.float32, device=x.device)
                og = torch.empty(b * n * n * c, dtype=x.dtype, device=x.device)
                cnt = torch.zeros(1 + b * n, dtype=torch.int32, device=x.device)
                coop = ((gr, b, h, n, x.device), zt, og, cnt, "split")
            elif n * n > 1024:
                # the whole-grid kernel materializes [N*N, C] tiles, which do not
                # fit in tensor memory past here -- leave it to the eager path
                coop = ((gr, b, h, n, x.device), x, x, False, "none")
            elif gr <= nsm:
                zt = torch.empty(b * h * n * n, dtype=torch.float32, device=x.device)
                flag = torch.zeros(triton.next_power_of_2(gr), dtype=torch.int32,
                                   device=x.device)
                coop = ((gr, b, h, n, x.device), zt, flag, True, "fused")
            else:
                coop = ((gr, b, h, n, x.device), x, x, False, "fused")
            self._coop = coop
            # the rendezvous counters are compared against gen * G, so a freshly
            # zeroed scratch has to restart the generation numbering
            self._gen = 0
        if coop[4] == "none":
            return None
        self._gen += 1
        if self._gen * gr * h > 2 ** 29:
            coop[3 if coop[4] == "split" else 2].zero_()
            self._gen = 1
        if coop[4] == "split":
            try:
                _tri_attn_split[(gr * h,)](
                    x, lnp, wph, wz, wot, mptr, out, coop[1], coop[2], coop[3],
                    sxb, sxi, sxj, n * n * c, soi, soj, smb, smi, smk, self._gen,
                    self.layer_norm.eps, self.inf, 1.0 / math.sqrt(self.c_hidden),
                    N=n, C=c, H=h, D=self.c_hidden, HP=hp, G=gr * h,
                    HAS_MASK=mask is not None, num_warps=8, num_stages=1,
                )
            except Exception:
                if self._nofuse is None:
                    self._nofuse = set()
                self._nofuse.add((gr, n))
                return None
            return out
        try:
            _tri_attn_fused[(gr,)](
                x, lnp, wp, wz, wot, mptr, out, coop[1], coop[2],
                sxb, sxi, sxj, n * n * c, soi, soj, smb, smi, smk, self._gen,
                self.layer_norm.eps, self.inf, 1.0 / math.sqrt(self.c_hidden),
                N=n, C=c, H=h, D=self.c_hidden, HP=hp,
                G=gr, GP=triton.next_power_of_2(gr),
                HAS_MASK=mask is not None, CONTIG=(sxi == n * c and sxj == c),
                COOP=coop[3], num_warps=8, num_stages=1,
            )
        except Exception:
            if self._nofuse is None:
                self._nofuse = set()
            self._nofuse.add((gr, n))
            return None
        return out

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        if self._fusable(x, mask):
            fused = self._fused_forward(x, mask)
            if fused is not None:
                return fused

        if mask is None:
            mask = x.new_ones(x.shape[:-1])

        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)

        x = self.layer_norm(x)
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]
        triangle_bias = _permute_final_dims(self.linear_z(x), (2, 0, 1))
        triangle_bias = triangle_bias.unsqueeze(-4)
        biases = [mask_bias, triangle_bias]
        x = self.mha(q_x=x, kv_x=x, biases=biases)
        if not self.starting:
            x = x.transpose(-2, -3)
        return x


TriangleAttentionStartingNode = TriangleAttention


class TriangleAttentionEndingNode(TriangleAttention):
    """AF3 Algorithm 15."""

    def __init__(self, c_in: int, c_hidden: int, no_heads: int, inf: float = 1e9):
        super().__init__(c_in=c_in, c_hidden=c_hidden, no_heads=no_heads,
                         starting=False, inf=inf)
