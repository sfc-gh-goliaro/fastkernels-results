"""MSA pair-weighted averaging for AlphaFold3 (Algorithm 10).

Weighted averaging over the MSA representation using pair activations,
NOT key-query self-attention.

Reference: openfold3/core/model/layers/msa.py MSAPairWeightedAveraging

The whole forward is one Triton launch.  At the captured shapes
(m bf16[1, 8, 16, 64], z bf16[1, 16, 16, 128]) this op is ~4 MFLOP spread over
~20 eager kernels, and the bench's timed window is dominated by *device-side
kernel occupancy*: an empty forward measures 13.4 us (the harness' three
shifting-pool input copies) and every additional kernel launch adds ~2 us.
Collapsing the launch count is therefore the only lever; tiling, occupancy and
tensor-core utilisation are noise at this size.

One program per (batch, seq, query row).  It recomputes the 16-row slice of the
pair-bias softmax that its own query row needs and keeps every intermediate in
registers, so nothing but the output touches global memory.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.sigmoid import Sigmoid
from ..L1.softmax import Softmax
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear


def _permute_final_dims(tensor: torch.Tensor, inds: tuple[int, ...]) -> torch.Tensor:
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


# ---------------------------------------------------------------------------
# Fused kernel.
#
# Index arithmetic replaces the whole permute/view/transpose/reshape chain:
#
#   * ``linear_z``'s weight is stored with head ``h`` in column ``h * PW`` of a
#     [C_z, H*PW] operand (PW chosen so H*PW >= 16, the tl.dot minimum; the pad
#     columns hold zeros).  The projection therefore comes out as [N, H, PW]
#     after a reshape and a sum over the length-PW axis recovers the [N, H]
#     logits *exactly* -- the pad lanes contribute a hard zero.  This is what
#     lets every operand tile stay at HD=64 instead of being fanned out to
#     [C_z, H*D]; the wide form measured 2 us slower.
#   * the (q, h, k) -> (h, q, k) permute is just "row k of the z slice for this
#     program's q", so the softmax reduces over axis 0.
#   * the per-(h, d) fan-out of the attention weights is a broadcast + reshape
#     (pure layout, no MMA), after which the weighted average over k collapses
#     to an elementwise product reduced over axis 0.
#
# All global loads are issued up front so their (cold-L2) latencies overlap --
# worth 2 us over the natural stage-by-stage ordering.
#
# Every place eager materialises a bf16 tensor is rounded here too.  That is not
# cosmetic: the mask bias is O(1e9), so in bf16 it entirely swamps the logits
# and the softmax degenerates to a one-hot (uniform across exact ties).
# Rounding at the same points is what keeps the selected k -- and hence the
# output -- identical to eager.
# ---------------------------------------------------------------------------
@triton.jit
def _msa_pwa_fwd(
    m_ptr, z_ptr, mask_ptr, out_ptr, w_ptr,
    INF: tl.constexpr, EPS: tl.constexpr,
    S: tl.constexpr, N: tl.constexpr, CZ: tl.constexpr, CM: tl.constexpr,
    H: tl.constexpr, D: tl.constexpr, HD: tl.constexpr, PW: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    q = pid % N
    s = (pid // N) % S
    b = pid // (N * S)

    k = tl.arange(0, N)
    cz = tl.arange(0, CZ)
    cm = tl.arange(0, CM)
    hd = tl.arange(0, HD)
    hp = tl.arange(0, H * PW)

    OV: tl.constexpr = CZ * H * PW          # linear_v   [CM, HD]
    OG: tl.constexpr = OV + CM * HD         # linear_g   [CM, HD]
    OO: tl.constexpr = OG + CM * HD         # linear_o   [CM, HD]
    OL: tl.constexpr = OO + CM * HD         # lnz_w | lnz_b | lnm_w | lnm_b
    mo = (b * S + s) * N * CM

    # -- every global load, issued back to back --
    zb = tl.load(z_ptr + (b * N + q) * N * CZ + k[:, None] * CZ + cz[None, :])
    mb = tl.load(m_ptr + mo + k[:, None] * CM + cm[None, :])
    wz = tl.load(w_ptr + cz[:, None] * (H * PW) + hp[None, :])
    wv = tl.load(w_ptr + OV + cm[:, None] * HD + hd[None, :])
    wg = tl.load(w_ptr + OG + cm[:, None] * HD + hd[None, :])
    wo = tl.load(w_ptr + OO + cm[:, None] * HD + hd[None, :])
    lzw = tl.load(w_ptr + OL + cz).to(tl.float32)
    lzb = tl.load(w_ptr + OL + CZ + cz).to(tl.float32)
    lmw = tl.load(w_ptr + OL + 2 * CZ + cm).to(tl.float32)
    lmb = tl.load(w_ptr + OL + 2 * CZ + CM + cm).to(tl.float32)
    if HAS_MASK:
        mk = tl.load(mask_ptr + (b * N + q) * N + k).to(tl.float32)

    # -- layer_norm_z / layer_norm_m (fp32 reduction, bf16 result, as eager) --
    zf = zb.to(tl.float32)
    mf = mb.to(tl.float32)
    zc = zf - (tl.sum(zf, 1) / CZ)[:, None]
    mc = mf - (tl.sum(mf, 1) / CM)[:, None]
    zr = 1.0 / tl.sqrt(tl.sum(zc * zc, 1) / CZ + EPS)
    mr = 1.0 / tl.sqrt(tl.sum(mc * mc, 1) / CM + EPS)
    zn = (zc * zr[:, None] * lzw[None, :] + lzb[None, :]).to(tl.bfloat16)
    mn = (mc * mr[:, None] * lmw[None, :] + lmb[None, :]).to(tl.bfloat16)

    # -- linear_z (already permuted to [k, h]) / linear_v / linear_g --
    lg = tl.sum(tl.reshape(tl.dot(zn, wz), (N, H, PW)), 2).to(tl.bfloat16).to(tl.float32)
    v = tl.dot(mn, wv).to(tl.bfloat16).to(tl.float32)
    gl = tl.dot(mn, wg).to(tl.bfloat16).to(tl.float32)
    gl = tl.sum(tl.where(k[:, None] == q, gl, 0.0), 0)
    g = (1.0 / (1.0 + tl.exp(-gl))).to(tl.bfloat16).to(tl.float32)

    # -- mask bias folded into the softmax pass, softmax over k --
    if HAS_MASK:
        bias = ((mk - 1.0).to(tl.bfloat16).to(tl.float32) * INF)
        lg = (lg + bias.to(tl.bfloat16).to(tl.float32)[:, None]).to(tl.bfloat16).to(tl.float32)
    e = tl.exp(lg - tl.max(lg, 0)[None, :])
    w8 = (e / tl.sum(e, 0)[None, :]).to(tl.bfloat16)

    # -- weighted average over k, gate, linear_o --
    w = tl.reshape(tl.broadcast_to(w8[:, :, None], (N, H, D)), (N, HD)).to(tl.float32)
    o = tl.sum(w * v, 0).to(tl.bfloat16).to(tl.float32)
    o = (o * g).to(tl.bfloat16).to(tl.float32)
    res = tl.sum(wo.to(tl.float32) * o[None, :], 1)
    tl.store(out_ptr + mo + q * CM + cm, res.to(tl.bfloat16))


class MSARowAttentionWithPairBias(nn.Module):
    """AF3 MSA Pair-Weighted Averaging (Algorithm 10).

    Uses pair activations as weights (softmax over token dim) instead of
    key-query attention.  Parameter names match the checkpoint layout:
    linear_v, linear_g, linear_o (no nested mha).

    Args:
        c_m: MSA input channel dimension
        c_z: Pair embedding channel dimension
        c_hidden: Per-head hidden channel dimension
        no_heads: Number of attention heads
        inf: Large constant for masking
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        inf: float = 1e9,
    ):
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf

        self.layer_norm_m = LayerNorm(c_m)
        self.layer_norm_z = LayerNorm(c_z)
        self.linear_z = Linear(c_z, no_heads, bias=False)

        self.linear_v = Linear(c_m, c_hidden * no_heads, bias=False)
        self.linear_g = Linear(c_m, c_hidden * no_heads, bias=False)
        self.linear_o = Linear(c_hidden * no_heads, c_m, bias=False)

        self.sigmoid = Sigmoid()
        self.softmax = Softmax(dim=-1)

        # Launch plan, built on the first forward that matches its shapes.  It
        # cannot be built in __init__: callers (the bench included) overwrite
        # every weight via load_state_dict afterwards.  The post-hook below and
        # the _apply override drop it whenever the weights change or move, so
        # forward never has to re-validate them -- it is a shape compare, one
        # allocation and the launch.
        self._plan = None
        self.register_load_state_dict_post_hook(
            lambda mod, incompatible_keys: mod._invalidate())

    # -- plan lifecycle -----------------------------------------------------
    def _invalidate(self) -> None:
        self._plan = None

    def _apply(self, *args, **kwargs):
        self._invalidate()
        return super()._apply(*args, **kwargs)

    def _build_plan(self, m: torch.Tensor, z: torch.Tensor, mask):
        """Launch plan for these shapes, or None to fall back to eager."""
        bf16 = torch.bfloat16
        c_m, c_z, d, h = self.c_m, self.c_z, self.c_hidden, self.no_heads
        hd = d * h
        if m.dim() != 4 or z.dim() != 4 or m.dtype is not bf16 or z.dtype is not bf16:
            return None
        if m.device.type != "cuda":
            return None
        if mask is not None and (mask.dim() != 3 or mask.dtype is not bf16):
            return None
        bz, nz, nz2, cz = z.shape
        bm, s, n, cm = m.shape
        if cz != c_z or cm != c_m or nz != n or nz2 != n or bz != bm:
            return None
        # tl.dot needs every tile edge >= 16 and a power of two; PW pads the
        # head axis of the linear_z operand up to that minimum.
        pw = 1
        while h * pw < 16:
            pw *= 2
        for x in (n, c_m, c_z, hd, d, h * pw):
            if x & (x - 1) or x < 1:
                return None
        if n < 16 or c_m < 16 or c_z < 16 or hd < 16:
            return None
        if mask is not None and tuple(mask.shape) != (bz, n, n):
            return None
        if not (m.is_contiguous() and z.is_contiguous()
                and (mask is None or mask.is_contiguous())):
            return None
        for p in (self.linear_z.weight, self.linear_v.weight,
                  self.linear_g.weight, self.linear_o.weight):
            if p.dtype is not bf16:
                return None
        if (self.linear_z.bias is not None or self.linear_v.bias is not None
                or self.linear_g.bias is not None or self.linear_o.bias is not None):
            return None
        for ln, width in ((self.layer_norm_z, c_z), (self.layer_norm_m, c_m)):
            if (not ln.promote_fp32 or ln.normalized_shape != (width,)
                    or ln.eps != self.layer_norm_z.eps):
                return None
            for p in (ln.weight, ln.bias):
                if p is not None and p.dtype is not bf16:
                    return None
        dev = m.device

        def _affine(ln, width):
            w, bi = ln.weight, ln.bias
            w = torch.ones(width, device=dev, dtype=bf16) if w is None else w.detach()
            bi = torch.zeros(width, device=dev, dtype=bf16) if bi is None else bi.detach()
            return w.reshape(-1), bi.reshape(-1)

        lzw, lzb = _affine(self.layer_norm_z, c_z)
        lmw, lmb = _affine(self.layer_norm_m, c_m)
        # linear_z: [H, C_z] -> [C_z, H*PW] with head h in column h*PW.
        wz = torch.zeros(h * pw, c_z, device=dev, dtype=bf16)
        wz[::pw] = self.linear_z.weight.detach()
        packed = torch.cat((
            wz.t().reshape(-1),
            self.linear_v.weight.detach().t().reshape(-1),
            self.linear_g.weight.detach().t().reshape(-1),
            self.linear_o.weight.detach().reshape(-1),
            lzw, lzb, lmw, lmb,
        )).contiguous()
        return (m.shape, z.shape, mask is not None, (bz * s * n,),
                packed, float(self.inf), float(self.layer_norm_z.eps),
                s, n, c_z, c_m, h, d, hd, pw, mask is not None)

    # -- eager reference ----------------------------------------------------
    def _forward_eager(self, m, z, mask):
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        mask_bias = (self.inf * (mask - 1))[..., None, None, :, :]
        z_norm = self.layer_norm_z(z)
        z_proj = self.linear_z(z_norm)
        z_weights = _permute_final_dims(z_proj, (2, 0, 1)).unsqueeze(-4)
        z_weights = z_weights + mask_bias
        z_weights = self.softmax(z_weights)

        m = self.layer_norm_m(m)

        v = self.linear_v(m)
        v = v.view(v.shape[:-1] + (self.no_heads, -1))
        v = v.transpose(-2, -3)

        o = torch.einsum("...hqk,...hkc->...qhc", z_weights, v)

        g = self.sigmoid(self.linear_g(m))
        g = g.view(g.shape[:-1] + (self.no_heads, -1))

        o = o * g
        o = o.reshape(o.shape[:-2] + (-1,))
        return self.linear_o(o)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            m:    [*, N_seq, N_res, C_m] MSA embedding
            z:    [*, N_res, N_res, C_z] pair embedding
            mask: [*, N_res, N_res] pair mask

        Returns:
            [*, N_seq, N_res, C_m] updated MSA embedding
        """
        if z is None:
            return m

        p = self._plan
        if p is None or m.shape != p[0] or z.shape != p[1] or (mask is not None) != p[2]:
            p = self._plan = self._build_plan(m, z, mask)
            if p is None:
                return self._forward_eager(m, z, mask)

        out = torch.empty_like(m)
        _msa_pwa_fwd[p[3]](m, z, mask if mask is not None else z, out,
                           *p[4:], num_warps=4)
        return out
