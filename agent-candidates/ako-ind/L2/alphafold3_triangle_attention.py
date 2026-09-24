"""Triangle attention for AlphaFold3 (L2) -- one fused Triton kernel.

Implements AF3 Algorithms 14 (starting node) and 15 (ending node).
Self-attention over one dimension of the pair representation with a
learned triangle bias from the other dimension.

Reference: openfold3/core/model/layers/triangular_attention.py TriangleAttention

The captured workload is a single tiny shape -- x: bf16[1, 16, 16, 128], 64 KB
per call -- replayed thousands of times, so the op is entirely launch/dispatch
bound rather than FLOP bound: the reference composition emits 23 kernels whose
GPU work sums to ~57 us but whose *CPU* dispatch costs ~285 us.  Everything here
exists to collapse that into one launch:

    LayerNorm -> linear_z triangle bias -> q/k/v/g projections -> scaled QK^T
    -> mask bias -> triangle bias -> softmax -> AV -> sigmoid gate -> linear_o

with the activation resident in registers throughout, and every permute /
transpose / view of the reference turned into index arithmetic:

  * the q/k/v ``view(..., no_heads, c_hidden) + transpose(-2, -3)`` becomes a
    register reshape+permute feeding one batched dot per stage,
  * ``_permute_final_dims`` on the triangle bias becomes the way the padded
    linear_z dot result is sliced,
  * the starting- vs ending-node ``transpose(-2, -3)`` of x / mask / output is a
    compile-time stride swap, not a copy,
  * ``inf * (mask - 1)`` is folded into the scores instead of materialized.

One program handles one query row i.  Row i attends only over its own NJ tokens,
but the triangle bias it needs spans the *whole* NI x NJ token grid (bias[q, k]
is shared by every row), so each program recomputes the whole-grid bias.  That
redundancy costs ~4 us but is still cheaper than splitting the bias into its own
launch: a second launch costs ~2 us of GPU dispatch *and* doubles the per-call
CPU dispatch, which is what actually falls off a cliff when the host is busy.

Shapes / dtypes the fused path does not cover fall back to the reference
composition, which is preserved verbatim in ``_forward_ref``.
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


@triton.jit
def _fused_triangle_attention(
    X, MASK, OUT, W,
    NI: tl.constexpr, NJ: tl.constexpr, NT: tl.constexpr, C: tl.constexpr,
    H: tl.constexpr, D: tl.constexpr, HAS_MASK: tl.constexpr, TRANS: tl.constexpr,
    SCALE: tl.constexpr, EPS: tl.constexpr, INF: tl.constexpr,
):
    """One program == one query row i of one batch element.

    All weights arrive in a single buffer at compile-time offsets and every
    stride is derived from the constexpr shape, so a launch carries 4 pointers
    instead of 8 pointers + 9 strides + 3 scalars.  That halves the per-call CPU
    dispatch: the harness hides dispatch behind a 75 us L2 flush, and staying
    well under that budget is what keeps the score stable on a busy host.
    """
    pid = tl.program_id(0)
    b = pid // NI
    i = pid % NI
    HD: tl.constexpr = H * D
    dt = X.dtype.element_ty

    # Alg 15 is Alg 14 on x.transpose(-2, -3) / mask.transpose(-1, -2): a stride
    # swap resolved at compile time, never a copy kernel.
    sx_i: tl.constexpr = C if TRANS else NJ * C
    sx_j: tl.constexpr = NI * C if TRANS else C
    sm_i: tl.constexpr = 1 if TRANS else NJ
    sm_j: tl.constexpr = NI if TRANS else 1
    sx_b: tl.constexpr = NI * NJ * C
    sm_b: tl.constexpr = NI * NJ

    # Packed weights: q|k|v|g projections, linear_o, linear_z, LN scale, LN bias.
    WP = W
    WO = W + 4 * HD * C
    WZ = WO + HD * C
    LNW = WZ + C * 4 * H
    LNB = LNW + C

    c = tl.arange(0, C)
    j = tl.arange(0, NJ)
    hd = tl.arange(0, HD)
    xb = X + b * sx_b
    lw = tl.load(LNW + c).to(tl.float32)
    lb = tl.load(LNB + c).to(tl.float32)

    # ---- triangle bias over the whole token grid ---------------------------
    # bias[h, q, k] = linear_z(LayerNorm(x))[q, k, h], shared by every query row,
    # so it must be complete before this row's softmax.
    t = tl.arange(0, NT)
    xg = tl.load(xb + (t // NJ)[:, None] * sx_i + (t % NJ)[:, None] * sx_j + c[None, :])
    xgf = xg.to(tl.float32)
    mg = tl.sum(xgf, 1) / C
    rg = 1.0 / tl.sqrt(tl.sum(xgf * xgf, 1) / C - mg * mg + EPS)
    xgn = (((xgf - mg[:, None]) * rg[:, None]) * lw[None, :] + lb[None, :]).to(dt)
    z = tl.dot(xgn, tl.load(WZ + c[:, None] * (4 * H) + tl.arange(0, 4 * H)[None, :]))
    # linear_z is padded out to the minimum MMA width with head h at column 4h,
    # so [NT, 4H] -> [q, k, H, 2, 2] and two splits of the trailing axis isolate
    # the H real columns without a reduction: cheaper than masking them out.
    ze, _ = tl.split(tl.reshape(z, (NI, NJ, H, 2, 2)))
    zh, _ = tl.split(ze)
    bias = tl.permute(zh, (2, 0, 1))                        # [H, NJ, NJ]

    # ---- this row: LayerNorm (fp32 stats, activation dtype out) -------------
    x = tl.load(xb + i * sx_i + j[:, None] * sx_j + c[None, :]).to(tl.float32)
    mu = tl.sum(x, 1) / C
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, 1) / C - mu * mu + EPS)
    xn = (((x - mu[:, None]) * rstd[:, None]) * lw[None, :] + lb[None, :]).to(dt)

    # ---- q/k/v from the packed projection, then batched attention -----------
    wpk = WP + c[:, None] * (4 * HD)
    q = tl.dot(xn, tl.load(wpk + hd[None, :])).to(dt)
    k = tl.dot(xn, tl.load(wpk + HD + hd[None, :])).to(dt)
    v = tl.dot(xn, tl.load(wpk + 2 * HD + hd[None, :])).to(dt)
    q3 = tl.permute(tl.reshape(q, (NJ, H, D)), (1, 0, 2))   # [H, NJ, D]
    k3 = tl.permute(tl.reshape(k, (NJ, H, D)), (1, 2, 0))   # [H, D, NJ]
    v3 = tl.permute(tl.reshape(v, (NJ, H, D)), (1, 0, 2))   # [H, NJ, D]
    s = tl.dot(q3, k3) * SCALE + bias
    if HAS_MASK:
        s += INF * (tl.load(MASK + b * sm_b + i * sm_i + j * sm_j).to(tl.float32)
                    - 1.0)[None, None, :]
    p = tl.exp(s - tl.max(s, 2)[:, :, None])
    p = (p / tl.sum(p, 2)[:, :, None]).to(dt)

    # ---- AV, sigmoid gate, linear_o ----------------------------------------
    o = tl.reshape(tl.permute(tl.dot(p, v3), (1, 0, 2)), (NJ, HD))
    g = 1.0 / (1.0 + tl.exp(-tl.dot(xn, tl.load(wpk + 3 * HD + hd[None, :]))))
    out = tl.dot((o * g).to(dt), tl.load(WO + hd[:, None] * C + c[None, :]))
    tl.store(OUT + b * sx_b + i * sx_i + j[:, None] * sx_j + c[None, :], out.to(dt))


_FUSED_DTYPES = (torch.bfloat16, torch.float16)
# num_warps=8 is the measured optimum: 4 spills on the [NT, C] grid tile, 16
# starves the M=16 projection dots (see ITERATIONS.md).  num_stages is inert --
# the kernel has no loop for the pipeliner to work on.
_LAUNCH = {"num_warps": 8, "num_stages": 1}


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

        # Packed-weight cache, built on the first fused call.  Plain attributes,
        # so it never enters state_dict and weight loading is unaffected.
        self._pw = None
        self._pw_key = None

    # -- packed weights ----------------------------------------------------
    def _param_key(self):
        """Identity of every parameter folded into the packed buffer.

        ``(data_ptr, _version)`` catches a replaced or moved parameter (``.to()``
        swaps ``.data`` in place) as well as an in-place write -- ``load_state_dict``
        copies under ``no_grad``, which bumps the version counter.
        """
        m = self.mha
        ps = (self.layer_norm.weight, self.layer_norm.bias, self.linear_z.weight,
              m.linear_q.weight, m.linear_k.weight, m.linear_v.weight,
              m.linear_g.weight, m.linear_o.weight)
        return tuple((-1, -1) if p is None else (p.data_ptr(), p._version) for p in ps)

    def _packed(self, dtype: torch.dtype, device: torch.device):
        """One flat buffer holding every weight in the layout the kernel wants.

        With thousands of identically-shaped calls this setup is free, and it
        turns 8 weight pointers into 1.
        """
        key = (self._param_key(), dtype, device)
        if self._pw is not None and self._pw_key == key:
            return self._pw
        m = self.mha
        h, c_in = self.no_heads, self.c_in
        with torch.no_grad():
            # [C, 4 * H * D]: column = p * H * D + h * D + dd, which is exactly
            # the reference's view(..., no_heads, c_hidden) split of q/k/v/g.
            wp = torch.cat((m.linear_q.weight, m.linear_k.weight,
                            m.linear_v.weight, m.linear_g.weight), 0)
            wp = wp.t().contiguous().to(device=device, dtype=dtype)
            wo = m.linear_o.weight.t().contiguous().to(device=device, dtype=dtype)

            lnw, lnb = self.layer_norm.weight, self.layer_norm.bias
            lnw = (torch.ones(c_in, device=device, dtype=dtype) if lnw is None
                   else lnw.to(device=device, dtype=dtype).contiguous())
            lnb = (torch.zeros(c_in, device=device, dtype=dtype) if lnb is None
                   else lnb.to(device=device, dtype=dtype).contiguous())

            # linear_z transposed and zero-padded to the minimum MMA width, head
            # h at column 4h so the kernel's split tree lands on it.
            wz = torch.zeros(c_in, 4 * h, device=device, dtype=dtype)
            wz[:, ::4] = self.linear_z.weight.t().to(dtype)

            packed = torch.empty(wp.numel() + wo.numel() + wz.numel() + 2 * c_in,
                                 device=device, dtype=dtype)
            off = 0
            for part in (wp, wo, wz, lnw, lnb):
                packed[off:off + part.numel()] = part.reshape(-1)
                off += part.numel()
        self._pw = packed
        self._pw_key = key
        return packed

    # -- reference composition (fallback) ----------------------------------
    def _forward_ref(self, x, mask):
        if mask is None:
            mask = x.new_ones(x.shape[:-1])

        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)

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

    # -- fused path --------------------------------------------------------
    def _fused(self, x, mask):
        ni, nj, c = x.shape[-3], x.shape[-2], x.shape[-1]
        out = torch.empty_like(x)
        _fused_triangle_attention[(x.numel() // (nj * c),)](
            x, mask, out, self._packed(x.dtype, x.device),
            NI=ni, NJ=nj, NT=ni * nj, C=c, H=self.no_heads, D=self.c_hidden,
            HAS_MASK=mask is not None, TRANS=not self.starting,
            SCALE=1.0 / math.sqrt(self.c_hidden), EPS=self.layer_norm.eps,
            INF=self.inf, **_LAUNCH,
        )
        return out

    def _fusable(self, x, mask) -> bool:
        """Whether the fused kernel covers this call.

        ``NI == NJ`` is not a real restriction: the reference broadcasts the
        triangle bias' I axis against the scores' Q axis, so a non-square token
        grid cannot run at all.  The rest keeps every tile a power of two and
        the whole grid inside one program's register budget.
        """
        if not (x.is_cuda and x.dtype in _FUSED_DTYPES and x.dim() in (3, 4)):
            return False
        ni, nj, c = x.shape[-3], x.shape[-2], x.shape[-1]
        if not (ni == nj and _pow2(ni) and ni * nj <= 256 and _pow2(c) and c == self.c_in):
            return False
        if not (_pow2(self.c_hidden) and self.c_hidden >= 16 and _pow2(self.no_heads)
                and self.no_heads >= 4):
            return False
        if not x.is_contiguous():
            return False
        if self.layer_norm.eps <= 0 or not self.layer_norm.promote_fp32:
            return False
        if mask is not None:
            if (mask.dtype != x.dtype or mask.shape != x.shape[:-1]
                    or not mask.is_contiguous()):
                return False
        return True

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
        if self._fusable(x, mask):
            return self._fused(x, mask)
        return self._forward_ref(x, mask)


TriangleAttentionStartingNode = TriangleAttention


class TriangleAttentionEndingNode(TriangleAttention):
    """AF3 Algorithm 15."""

    def __init__(self, c_in: int, c_hidden: int, no_heads: int, inf: float = 1e9):
        super().__init__(c_in=c_in, c_hidden=c_hidden, no_heads=no_heads,
                         starting=False, inf=inf)
