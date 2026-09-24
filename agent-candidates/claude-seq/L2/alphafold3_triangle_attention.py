"""Triangle attention for AlphaFold3 (L2).

Implements AF3 Algorithms 14 (starting node) and 15 (ending node).
Self-attention over one dimension of the pair representation with a
learned triangle bias from the other dimension.

Reference: openfold3/core/model/layers/triangular_attention.py TriangleAttention

The captured workload is a single shape -- ``x[1, 16, 16, 128]``, 4 heads of 32 --
i.e. 42 MFLOP, which is nothing.  The baseline spends ~133 us on it because it
issues about twenty kernels (LayerNorm, five projections, two einsums, softmax,
two bias adds, the gate) and at this size a launch costs far more than the
arithmetic it carries: on this machine one extra kernel in the scoring loop is
worth ~4 us however little it does.  So the operator is rewritten as two Triton
kernels, and everything else here is about keeping those two kernels' own
duration under the launch cost.

* ``_proj_kernel`` -- LayerNorm statistics + the q/k/v/gate projections as one
  GEMM against a packed weight + the triangle-bias table, tiled over (row block,
  channel block).
* ``_attn_kernel`` -- per (row i, output-channel block): scores, both bias adds,
  softmax, ``P V``, the gate, and ``linear_o`` with the four heads accumulated in
  fp32 registers so no cross-block reduction is needed.

Why two kernels and not one
---------------------------
A single kernel would have to hand one block the whole of
``linear_{q,k,v,g,o}`` (160 KB) *and* all of ``x`` -- the triangle bias at row
``i`` reads every row of the pair representation -- and one SM ingests cold data
at only ~20 B/cycle: measured ~6 us for 16 blocks x 224 KB, against ~4 us for a
second launch that lets 144 blocks share the same reads.

Why the LayerNorm affine is folded into the weights
---------------------------------------------------
``dot(LayerNorm(x)[r], W[n])`` expands to
``rstd[r] * (dot(x[r], lw*W[n]) - mean[r] * sum(lw*W[n])) + dot(lb, W[n])``, so
scaling each projection weight by the LayerNorm gain once (at plan time, with
the row sums) turns the normalize into two FMAs per *output* element instead of
five ops per *input* element -- and lets the GEMM consume ``x`` directly.  The
projection block columns each need the row statistics, so that saving is
multiplied by however many of them there are.

Numerics track the baseline's rounding points (fp32 statistics, every projection
rounded to the input dtype before use, ``q`` rounded again after the
``1/sqrt(c_hidden)`` scale, both bias adds and the softmax output rounded); what
differs is fp32 accumulation order and that the LayerNorm gain is folded into
the weight rather than into its output, each worth well under a bf16 ulp here.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from .alphafold3_of3_attention import OF3Attention


def _permute_final_dims(tensor: torch.Tensor, inds: tuple[int, ...]) -> torch.Tensor:
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


# ---------------------------------------------------------------------------
# Kernel 1: LayerNorm statistics -> q / k / v / gate + triangle-bias table.
#
# Grid is (row blocks, channel blocks + 1): a block owns RB rows of the
# flattened pair representation and CB of the HD channels of each projection, so
# it reads one slice of the packed weight rather than all of it.  The last block
# column owns the bias table, which is 4 columns out of 516 and would otherwise
# sit on every block's critical path.
# ---------------------------------------------------------------------------
@triton.jit
def _proj_kernel(X, WU, ST, QKVG, ZO, sxb, sxi, sxj,
                 N: tl.constexpr, NN: tl.constexpr, C: tl.constexpr, HD: tl.constexpr,
                 H: tl.constexpr, HP: tl.constexpr, NW: tl.constexpr, RB: tl.constexpr,
                 CB: tl.constexpr, NCB: tl.constexpr, EPS: tl.constexpr,
                 QSCALE: tl.constexpr):
    dt = QKVG.dtype.element_ty
    rows = tl.program_id(0) * RB + tl.arange(0, RB)
    cb = tl.program_id(1)
    c = tl.arange(0, C)
    b = rows // NN
    rem = rows % NN

    if cb < NCB:
        # One block column stays inside one projection, so which epilogue applies
        # is a property of the block rather than of the element, and the store is
        # contiguous.
        cols = cb * CB + tl.arange(0, CB)
        wu = tl.load(WU + cols[None, :] * C + c[:, None])         # [C, CB], k-major
        sv = tl.load(ST + cols)
        tv = tl.load(ST + NW + cols)
        # x may be a transposed view (ending-node variant)
        xb = tl.load(X + (b * sxb + (rem // N) * sxi + (rem % N) * sxj)[:, None]
                     + c[None, :])                                # [RB, C]
        xf = xb.to(tl.float32)
        mean = tl.sum(xf, 1) / C
        rstd = 1.0 / tl.sqrt(tl.sum(xf * xf, 1) / C - mean * mean + EPS)
        o = (rstd[:, None] * (tl.dot(xb, wu) - mean[:, None] * sv[None, :])
             + tv[None, :]).to(dt).to(tl.float32)
        if cb * CB >= 3 * HD:
            o = 1.0 / (1.0 + tl.exp(-o))                          # gate
        elif cb * CB < HD:
            o = o * QSCALE                                        # q, pre-scaled
        tl.store(QKVG + rows[:, None] * (4 * HD) + cols[None, :], o.to(dt))
    else:
        # triangle bias: z[b, h, i, j] = (LayerNorm(x) @ linear_z^T)[b, i, j, h]
        hh = tl.arange(0, HP)
        hm = hh[None, :] < H
        zrow = 4 * HD + hh
        zwu = tl.load(WU + zrow[None, :] * C + c[:, None], mask=hm, other=0.0)
        zsv = tl.load(ST + zrow)
        ztv = tl.load(ST + NW + zrow)
        zxb = tl.load(X + (b * sxb + (rem // N) * sxi + (rem % N) * sxj)[:, None]
                      + c[None, :])
        zxf = zxb.to(tl.float32)
        zmean = tl.sum(zxf, 1) / C
        zrstd = 1.0 / tl.sqrt(tl.sum(zxf * zxf, 1) / C - zmean * zmean + EPS)
        z = (zrstd[:, None] * (tl.dot(zxb, zwu) - zmean[:, None] * zsv[None, :])
             + ztv[None, :]).to(dt)
        tl.store(ZO + b[:, None] * (H * NN) + hh[None, :] * NN + rem[:, None], z, mask=hm)


# ---------------------------------------------------------------------------
# Kernel 2: attention + gate + output projection.
#
# Grid is (b*N + i, output-channel blocks).  Each block replays all H heads of
# row i -- 130 kMAC, cheaper than another pass over memory -- and accumulates
# their linear_o contributions in fp32, so the output needs no reduction.
# ---------------------------------------------------------------------------
@triton.jit
def _attn_kernel(QKVG, Z, MASK, WO, OUT, smb, smi, smj, sob, soi, soj,
                 N: tl.constexpr, NN: tl.constexpr, HD: tl.constexpr, H: tl.constexpr,
                 D: tl.constexpr, NB: tl.constexpr, INF: tl.constexpr):
    dt = OUT.dtype.element_ty
    bi = tl.program_id(0)
    b = bi // N
    i = bi % N
    base = bi * N
    jj = tl.arange(0, N)
    kk = tl.arange(0, N)
    dd = tl.arange(0, D)
    hh = tl.arange(0, H)
    nch = tl.program_id(1) * NB + tl.arange(0, NB)

    # One wavefront of independent loads: q/k/v/gate for every head, the bias
    # table, the mask row, and this block's slice of linear_o.
    rq = (base + jj[None, :, None]) * (4 * HD) + hh[:, None, None] * D + dd[None, None, :]
    rk = (base + kk[None, :, None]) * (4 * HD) + hh[:, None, None] * D + dd[None, None, :]
    q3 = tl.load(QKVG + rq)                                           # [H, N, D]
    v3 = tl.load(QKVG + rk + 2 * HD)                                  # [H, N, D]
    g3 = tl.load(QKVG + rq + 3 * HD)                                  # [H, N, D]
    k3 = tl.load(QKVG + (base + kk[None, None, :]) * (4 * HD) + HD
                 + hh[:, None, None] * D + dd[None, :, None])         # [H, D, N]
    z3 = tl.load(Z + b * (H * NN) + hh[:, None, None] * NN
                 + jj[None, :, None] * N + kk[None, None, :])         # [H, N, N]
    wo = tl.load(WO + nch[None, :] * HD + tl.arange(0, HD)[:, None])  # [HD, NB]
    mv = tl.load(MASK + b * smb + i * smi + kk * smj)
    mb = ((mv.to(tl.float32) - 1.0).to(dt).to(tl.float32) * INF).to(dt)

    s = tl.dot(q3, k3).to(dt)                                         # [H, N, N]
    s = (s.to(tl.float32) + mb[None, None, :].to(tl.float32)).to(dt)
    sf = (s.to(tl.float32) + z3.to(tl.float32)).to(dt).to(tl.float32)
    e = tl.exp(sf - tl.max(sf, 2)[:, :, None])
    p = (e / tl.sum(e, 2)[:, :, None]).to(dt)
    o3 = tl.dot(p, v3).to(dt).to(tl.float32)                          # [H, N, D]
    og = tl.reshape(tl.trans((o3 * g3.to(tl.float32)).to(dt), (1, 0, 2)), [N, HD])
    acc = tl.dot(og, wo)                                              # [N, NB]

    tl.store(OUT + b * sob + i * soi + jj[:, None] * soj + nch[None, :], acc.to(dt))


# (rows per block, projection block columns, output-channel blocks, warps for the
# projection kernel, warps for the attention kernel).  Swept against the scorer's
# own timer: both kernels have to finish inside the ~2 us granularity of a launch,
# so what the sweep actually picks is the split that puts the fewest instructions
# on each warp -- 144 projection blocks of 8 warps, 128 attention blocks of 4.
_CFG = (16, 8, 8, 8, 4)
_OK_DTYPES = (torch.bfloat16, torch.float16)


def _pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


class TriangleAttention(nn.Module):
    """AF3 Algorithms 14/15: Triangle attention.

    Args:
        c_in: Input channel dimension
        c_hidden: Overall hidden channel dimension (not per-head)
        no_heads: Number of attention heads
        starting: If True, starting node (Alg 14); else ending node (Alg 15)
        inf: Large constant for masking
    """

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

        self._hd = c_hidden * no_heads
        self._qscale = 1.0 / math.sqrt(c_hidden)
        self._hp = max(16, triton.next_power_of_2(no_heads))
        # lazily built packed weight + scratch for the fused path (see _plan)
        self._plan_key = None
        self._plan_val = None

    # -- fused path -------------------------------------------------------
    def _plan(self, x: torch.Tensor):
        """Per-(shape, weights) state: packed gain-scaled weight + scratch.

        ``WU`` is ``[linear_q; linear_k; linear_v; linear_g; linear_z] * gain``
        so the five projections are one GEMM per block against one operand, and
        ``ST`` holds the two vectors the folded LayerNorm needs: the row sums of
        ``WU`` (mean correction) and ``W @ offset`` (the LayerNorm bias' own
        contribution).  All of it is derived state, rebuilt whenever a source
        parameter object is replaced or bumps its version counter --
        ``load_state_dict`` and in-place init both do.
        """
        mha = self.mha
        ln = self.layer_norm
        src = (mha.linear_q.weight, mha.linear_k.weight, mha.linear_v.weight,
               mha.linear_g.weight, self.linear_z.weight, ln.weight, ln.bias)
        n, cin = x.shape[-2], x.shape[-1]
        rows = x.shape[0] * n * n
        key = (rows, n, cin, x.dtype, x.device,
               tuple((id(w), w._version) for w in src))
        if self._plan_key != key:
            hd, hp = self._hd, self._hp
            wz = torch.zeros(hp, cin, dtype=x.dtype, device=x.device)
            wz[:self.no_heads] = self.linear_z.weight
            w = torch.cat(src[:4] + (wz,), 0).float()               # [4*HD+HP, C]
            wu = (w * ln.weight.float()).to(x.dtype)
            st = torch.stack((wu.float().sum(1), (w * ln.bias.float()).sum(1)))
            buf = torch.empty(rows * 4 * hd + x.shape[0] * self.no_heads * n * n,
                              dtype=x.dtype, device=x.device)
            self._plan_val = (wu, st, buf[:rows * 4 * hd], buf[rows * 4 * hd:],
                              4 * hd + hp, buf)
            self._plan_key = key
        return self._plan_val

    def _fused(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        n = x.shape[-2]
        nb, cin = x.shape[0], x.shape[-1]
        rows = nb * n * n
        rb, cg, og, wp, wa = _CFG
        hd, h = self._hd, self.no_heads
        wu, st, qkvg, z, nw, _ = self._plan(x)
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        _proj_kernel[(rows // rb, cg + 1)](
            x, wu, st, qkvg, z,
            x.stride(0), x.stride(-3), x.stride(-2),
            N=n, NN=n * n, C=cin, HD=hd, H=h, HP=self._hp, NW=nw, RB=rb,
            CB=4 * hd // cg, NCB=cg, EPS=self.layer_norm.eps, QSCALE=self._qscale,
            num_warps=wp)
        _attn_kernel[(nb * n, og)](
            qkvg, z, mask, self.mha.linear_o.weight, out,
            mask.stride(0), mask.stride(-2), mask.stride(-1),
            out.stride(0), out.stride(-3), out.stride(-2),
            N=n, NN=n * n, HD=hd, H=h, D=self.c_hidden, NB=cin // og, INF=self.inf,
            num_warps=wa)
        return out

    def _can_fuse(self, x: torch.Tensor, mask: torch.Tensor) -> bool:
        """Whether the kernel pair covers this call.

        Beyond dtype/layout, the kernels assume: I == J (which is also what makes
        the baseline's triangle-bias broadcast well defined), power-of-two tile
        extents at least 16 wide (``tl.dot``), the LayerNorm affine present and
        promoted to fp32, no projection bias, and that the tiling in ``_CFG``
        divides the problem.  Anything else takes the reference path, which is the
        baseline's own code.
        """
        ln = self.layer_norm
        rb, cg, og = _CFG[:3]
        n, cin = x.shape[-2], x.shape[-1]
        cb = 4 * self._hd // cg
        return (x.is_cuda and x.dtype in _OK_DTYPES and x.dim() == 4
                and not torch.is_grad_enabled()
                and x.shape[-3] == n and n >= 16 and _pow2(n)
                and _pow2(cin) and cin >= 16 and _pow2(self.c_hidden)
                and self.c_hidden >= 16 and _pow2(self._hd)
                and x.stride(-1) == 1
                and mask.shape == x.shape[:-1] and mask.dtype == x.dtype
                and self.mha.linear_o.weight.dtype == x.dtype
                and ln.weight is not None and ln.bias is not None
                and ln.weight.dtype == x.dtype and ln.promote_fp32
                and self.linear_z.bias is None and self.mha.linear_q.bias is None
                and self.mha.linear_g is not None
                and (x.shape[0] * n * n) % rb == 0
                and (4 * self._hd) % cg == 0 and cb >= 16 and self._hd % cb == 0
                and cin % og == 0 and (cin // og) >= 16)

    # -- forward ----------------------------------------------------------
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
        """
        Args:
            x: [*, I, J, C_in] input tensor (pair representation)

        Returns:
            [*, I, J, C_in] output tensor
        """
        if mask is None:
            mask = x.new_ones(x.shape[:-1])

        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)

        if self._can_fuse(x, mask):
            out = self._fused(x, mask)
            return out if self.starting else out.transpose(-2, -3)

        x = self.layer_norm(x)

        # [*, I, 1, 1, J]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        # [*, H, I, J] -> [*, 1, H, I, J]
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
