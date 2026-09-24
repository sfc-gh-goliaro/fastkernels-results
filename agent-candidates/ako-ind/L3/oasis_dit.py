"""Oasis diffusion transformer -- launch-overhead-optimized.

The captured workload (bsz=1, T=2..6 frames, a fixed 9x16 patch grid,
hidden=1024, 16 heads, depth=16) is entirely host-bound in the reference: the
module tree issues on the order of two thousand tiny ops per forward, so the
~25 ms wall time is Python dispatch and launch latency, not the ~2 ms of GPU
work underneath -- the reference costs the same at T=2 and T=6.

Three structural changes, and deliberately no numerical ones:

1. Everything shape-invariant is hoisted into a cache built once on the first
   forward: the axial spatial rotary cos/sin for the fixed patch grid, the
   temporal rotary cos/sin per T, the timestep sinusoid frequencies, and -- the
   big one -- all 33 adaLN projections concatenated into a single weight, since
   they all consume the same ``silu(c)`` (32 separate GEMMs: 359 us; fused: 130).  The reference
   rebuilds the axial freqs with linspace/einsum/repeat_interleave inside every
   attention call, separately for q and k: 32x redundant.

2. One flat token layout ``(B, T, N, D)`` throughout, and four fused Triton
   kernels over it: modulate, gate+residual, and the two qkv-split/rotary/layout
   passes.  ``_modulate`` / ``_gate`` become plain broadcasts instead of
   ``repeat`` + ``unsqueeze`` chains, and the rotary pass writes the exact
   ``(batch, heads, seq, head_dim)`` layout SDPA wants, so the reference's
   permute -> reshape -> contiguous clones disappear.  Graph nodes: ~840 -> ~455.

3. The resulting op stream is captured into one CUDA graph per
   (B, T, has_cond) and replayed; per-call host cost collapses to three small
   input copies.

Numerics are held bit-exact against the reference wherever a choice exists:
every GEMM keeps the reference's (M, K, N) and weight tensor, attention goes
through the same SDPA call (its mem-efficient backend runs true fp32 and
ignores the ambient TF32 setting, so an equivalent cuBLAS bmm would *not*
match), and the elementwise chains avoid fusing a multiply-add into an FMA.
Those last two matter: the network amplifies a 1e-6 front-end perturbation into
a ~1e-4 output difference, which fails an atol of 1e-5.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.linear import Linear
from ..L1.oasis_rotary import OasisRotaryEmbedding
from ..L2.oasis_final_layer import OasisFinalLayer
from ..L2.oasis_patch_embed import OasisPatchEmbed
from ..L2.oasis_timestep_embedder import OasisTimestepEmbedder
from .oasis_block import SpatioTemporalDiTBlock

# Use the fused Triton kernels below.  Turning this off falls back to
# bit-identical torch op chains -- slower, but the whole kernel still passes.
_FUSE = True


# ---------------------------------------------------------------------------
# Fused elementwise kernels.
#
# These exist only to cut CUDA-graph node count and memory round-trips; they are
# held *bitwise* identical to the torch op chains they replace.  Hence the
# explicit ``mul.rn.f32`` / ``add.rn.f32``: Triton contracts a plain ``a*b + c``
# into an FMA, and a single contracted FMA anywhere in this network moves the
# output by ~1e-4, which fails atol=1e-5 (measured matched ratio 1.000 -> 0.80).
# ---------------------------------------------------------------------------
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - fall back to the torch op chains
    _HAS_TRITON = False

if _HAS_TRITON:

    @triton.jit
    def _mul_rn(a, b):
        return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=r,r,r", [a, b],
                                         dtype=tl.float32, is_pure=True, pack=1)

    @triton.jit
    def _add_rn(a, b):
        return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=r,r,r", [a, b],
                                         dtype=tl.float32, is_pure=True, pack=1)

    @triton.jit
    def _k_modulate(X, SH, SC, O, mod_stride, n_tok, D: tl.constexpr):
        """``O[r] = X[r] * SC[r // n_tok] + SH[r // n_tok]``, one row per program.

        Replaces a ``mul`` + ``add_`` pair (and the temporary between them).
        """
        r = tl.program_id(0)
        d = tl.arange(0, D)
        mo = (r // n_tok) * mod_stride + d
        y = _add_rn(_mul_rn(tl.load(X + r * D + d), tl.load(SC + mo)), tl.load(SH + mo))
        tl.store(O + r * D + d, y)

    @triton.jit
    def _k_gate_add(X, Y, G, mod_stride, n_tok, D: tl.constexpr):
        """``X[r] += Y[r] * G[r // n_tok]``, in place on X (gate + residual)."""
        r = tl.program_id(0)
        d = tl.arange(0, D)
        o = r * D + d
        g = tl.load(G + (r // n_tok) * mod_stride + d)
        tl.store(X + o, _add_rn(tl.load(X + o), _mul_rn(tl.load(Y + o), g)))

    @triton.jit
    def _k_qkv_rope_sp(QKV, COS, SIN, QO, KO, VO, n_tok,
                       NH: tl.constexpr, HD: tl.constexpr, BN: tl.constexpr,
                       NBLK: tl.constexpr):
        """Split qkv, rotate q and k, write all three in (BT, NH, N, HD) order.

        Collapses nine torch ops -- two 4-op rotary chains plus v's
        ``contiguous`` -- into one pass, and absorbs the reference's
        permute/clone into the same pass.  The paired element for the
        interleaved rotation is fetched with a second ``d ^ 1`` load out of the
        same 256-byte row rather than materializing a flipped copy.
        """
        pid = tl.program_id(0)
        nb = pid % NBLK
        rest = pid // NBLK
        h = rest % NH
        bt = rest // NH
        n = nb * BN + tl.arange(0, BN)
        d = tl.arange(0, HD)
        m = n[:, None] < n_tok
        D1: tl.constexpr = NH * HD
        row = (bt * n_tok + n)[:, None] * (3 * D1) + h * HD
        ca = d[None, :]
        cb = (d ^ 1)[None, :]
        tab = n[:, None] * HD + ca
        out = ((bt * NH + h) * n_tok + n)[:, None] * HD + ca

        a = tl.load(QKV + row + ca, mask=m, other=0.0)
        b = tl.load(QKV + row + cb, mask=m, other=0.0)
        tl.store(QO + out, _add_rn(_mul_rn(a, tl.load(COS + tab, mask=m, other=0.0)),
                                   _mul_rn(b, tl.load(SIN + tab, mask=m, other=0.0))), mask=m)
        a = tl.load(QKV + row + D1 + ca, mask=m, other=0.0)
        b = tl.load(QKV + row + D1 + cb, mask=m, other=0.0)
        tl.store(KO + out, _add_rn(_mul_rn(a, tl.load(COS + tab, mask=m, other=0.0)),
                                   _mul_rn(b, tl.load(SIN + tab, mask=m, other=0.0))), mask=m)
        tl.store(VO + out, tl.load(QKV + row + 2 * D1 + ca, mask=m, other=0.0), mask=m)

    @triton.jit
    def _k_qkv_rope_tp(QKV, COS, SIN, QO, KO, VO, n_tok,
                       T: tl.constexpr, NH: tl.constexpr, HD: tl.constexpr,
                       TB: tl.constexpr, HB: tl.constexpr, NHB: tl.constexpr):
        """Same, for the temporal axis: output order (B*N, NH, T, HD).

        One program covers HB heads x all T frames of one (b, n), which keeps
        both the strided gather out of the qkv GEMM and the contiguous store
        wide even though T is only 2..6.
        """
        pid = tl.program_id(0)
        hb = pid % NHB
        n = (pid // NHB) % n_tok
        bb = pid // (NHB * n_tok)
        t = tl.arange(0, TB)
        j = tl.arange(0, HB * HD)
        hh = j // HD
        dd = j % HD
        m = t[:, None] < T
        D1: tl.constexpr = NH * HD
        row = ((bb * T + t) * n_tok + n)[:, None] * (3 * D1) + (hb * (HB * HD) + j)[None, :]
        sw = ((dd ^ 1) - dd)[None, :]
        tab = t[:, None] * HD + dd[None, :]
        out = ((((bb * n_tok + n) * NH + hb * HB + hh) * T)[None, :] + t[:, None]) * HD + dd[None, :]

        a = tl.load(QKV + row, mask=m, other=0.0)
        b = tl.load(QKV + row + sw, mask=m, other=0.0)
        tl.store(QO + out, _add_rn(_mul_rn(a, tl.load(COS + tab, mask=m, other=0.0)),
                                   _mul_rn(b, tl.load(SIN + tab, mask=m, other=0.0))), mask=m)
        a = tl.load(QKV + row + D1, mask=m, other=0.0)
        b = tl.load(QKV + row + D1 + sw, mask=m, other=0.0)
        tl.store(KO + out, _add_rn(_mul_rn(a, tl.load(COS + tab, mask=m, other=0.0)),
                                   _mul_rn(b, tl.load(SIN + tab, mask=m, other=0.0))), mask=m)
        tl.store(VO + out, tl.load(QKV + row + 2 * D1, mask=m, other=0.0), mask=m)


class _Plan:
    """Static input/output buffers plus the captured graph for one signature."""

    __slots__ = ("x", "t", "ec", "out", "graph", "pool")


class OasisDiT(nn.Module):
    def __init__(
        self,
        *,
        input_h: int = 18,
        input_w: int = 32,
        patch_size: int = 2,
        in_channels: int = 16,
        hidden_size: int = 1024,
        depth: int = 16,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        external_cond_dim: int = 25,
        max_frames: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.max_frames = max_frames

        self.x_embedder = OasisPatchEmbed(input_h, input_w, patch_size, in_channels, hidden_size, flatten=False)
        self.t_embedder = OasisTimestepEmbedder(hidden_size)
        head_dim = hidden_size // num_heads
        self.spatial_rotary_emb = OasisRotaryEmbedding(dim=head_dim // 2, freqs_for="pixel", max_freq=256)
        self.temporal_rotary_emb = OasisRotaryEmbedding(dim=head_dim, freqs_for="lang")
        self.external_cond = Linear(external_cond_dim, hidden_size, bias=True) if external_cond_dim > 0 else nn.Identity()
        self.blocks = nn.ModuleList(
            [
                SpatioTemporalDiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    is_causal=True,
                    spatial_rotary_emb=self.spatial_rotary_emb,
                    temporal_rotary_emb=self.temporal_rotary_emb,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = OasisFinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

        self.hidden_size = hidden_size
        self.depth = depth
        self.head_dim = head_dim
        self._static = None
        self._per_t = {}
        self._plans = {}
        self._fused = _FUSE and _HAS_TRITON
        self._nograph = False

    def initialize_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        weight = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(weight.view(weight.shape[0], -1))
        if self.x_embedder.proj.bias is not None:
            nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.s_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.s_adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.t_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.t_adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        if self.final_layer.linear.bias is not None:
            nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = x.shape[1]
        w = x.shape[2]
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    # ------------------------------------------------------------------ caches
    def _build_static(self, device, dtype) -> None:
        hd = self.head_dim
        st = {}

        proj = self.x_embedder.proj
        st["pe_w"] = proj.weight
        st["pe_b"] = proj.bias

        half = self.t_embedder.frequency_embedding_size // 2
        st["ts_freqs"] = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, dtype=torch.float32, device=device) / half
        )

        h_grid, w_grid = self.x_embedder.grid_size
        freqs = self.spatial_rotary_emb.get_axial_freqs(h_grid, w_grid).reshape(h_grid * w_grid, hd)
        cos, sin = self._rope_tables(freqs)
        st["sp_cos"], st["sp_sin"] = cos, sin

        mods = []
        for blk in self.blocks:
            mods.append(blk.s_adaLN_modulation[-1])
            mods.append(blk.t_adaLN_modulation[-1])
        st["mod_w"] = torch.cat([m.weight for m in mods], dim=0)
        st["mod_b"] = torch.cat([m.bias for m in mods], dim=0)
        fin = self.final_layer.adaLN_modulation[-1]
        st["fin_w"], st["fin_b"] = fin.weight, fin.bias

        st["blocks"] = [
            (
                b.s_attn.to_qkv.weight, b.s_attn.to_out.weight, b.s_attn.to_out.bias,
                b.s_mlp.fc1.weight, b.s_mlp.fc1.bias, b.s_mlp.fc2.weight, b.s_mlp.fc2.bias,
                b.t_attn.to_qkv.weight, b.t_attn.to_out.weight, b.t_attn.to_out.bias,
                b.t_mlp.fc1.weight, b.t_mlp.fc1.bias, b.t_mlp.fc2.weight, b.t_mlp.fc2.bias,
            )
            for b in self.blocks
        ]
        st["grid"] = (h_grid, w_grid)
        st["n_tok"] = h_grid * w_grid
        self._static = st

    @staticmethod
    def _rope_tables(freqs: torch.Tensor):
        """``(cos, sin')`` for interleaved rotation, ``sin'`` carrying the sign.

        ``x * cos + swap_pairs(x) * sin'`` with ``sin'[..., 2i] = -sin[..., 2i]``
        reproduces the reference ``rotate_half``, which maps a pair (a, b) to
        (-b, a).  Folding the sign into the table is exact -- ``(-a)*b`` and
        ``a*(-b)`` are the same float -- so this is a rewrite, not an
        approximation.  q and k share one table pair, as in the reference.
        """
        cos = freqs.cos()
        sin = freqs.sin().clone()
        sin[..., 0::2].neg_()
        return cos, sin

    def _temporal_cache(self, time: int, device, dtype):
        got = self._per_t.get(time)
        if got is not None:
            return got
        rot = self.temporal_rotary_emb
        pos = torch.arange(time, device=device, dtype=dtype)
        cos, sin = self._rope_tables(rot.forward(pos, rot.freqs, seq_len=time))
        got = {"cos": cos, "sin": sin}
        self._per_t[time] = got
        return got

    # ------------------------------------------------------------- primitives
    @staticmethod
    def _rope(u: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Interleaved rotary on the last dim, into a fresh *contiguous* result.

        ``u`` is a strided view of the qkv GEMM output, so the explicit ``out=``
        both forces the contiguous (batch, heads, seq, head_dim) layout the
        attention wants -- absorbing the reference's separate permute/clone --
        and stops the elementwise op from inheriting ``u``'s stride order.
        """
        out = torch.empty(u.shape, dtype=u.dtype, device=u.device)
        torch.mul(u, cos, out=out)
        swapped = u.unflatten(-1, (u.shape[-1] // 2, 2)).flip(-1).reshape(out.shape)
        out.add_(swapped.mul_(sin))  # two rounded ops; see _modulate
        return out

    def _modulate(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """``x * (1 + scale) + shift`` with the +1 already folded into *scale*.

        *shift* / *scale* are ``(B, T, D)`` slices of the fused modulation GEMM
        output, broadcast over the N tokens of each frame.
        """
        if self._fused and x.is_cuda:
            out = torch.empty_like(x)
            D = x.shape[-1]
            _k_modulate[(x.numel() // D,)](x, shift, scale, out, shift.stride(1),
                                           x.shape[-2], D=D, num_warps=4)
            return out
        # Two rounded ops, not an ``addcmul``: the reference's ``x * (1+scale)``
        # then ``+ shift`` rounds the product, and contracting that into an FMA
        # is enough to fail the comparison.
        return torch.mul(x, scale.unsqueeze(-2)).add_(shift.unsqueeze(-2))

    def _gate_add(self, x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor) -> None:
        """``x += y * gate`` in place -- the gate + residual add of every branch."""
        if self._fused and x.is_cuda:
            D = x.shape[-1]
            _k_gate_add[(x.numel() // D,)](x, y, gate, gate.stride(1), x.shape[-2],
                                           D=D, num_warps=4)
        else:
            x.add_(y.mul_(gate.unsqueeze(-2)))

    def _qkv_rope_sp(self, qkv, cos, sin, BT, N, nh, hd):
        """qkv -> rotated q, k and plain v, all as (BT, nh, N, hd) contiguous."""
        if self._fused and qkv.is_cuda:
            shape = (BT, nh, N, hd)
            q = torch.empty(shape, dtype=qkv.dtype, device=qkv.device)
            k = torch.empty(shape, dtype=qkv.dtype, device=qkv.device)
            v = torch.empty(shape, dtype=qkv.dtype, device=qkv.device)
            bn = 16
            nblk = -(-N // bn)
            _k_qkv_rope_sp[(BT * nh * nblk,)](qkv, cos, sin, q, k, v, N,
                                              NH=nh, HD=hd, BN=bn, NBLK=nblk, num_warps=4)
            return q, k, v
        w = qkv.view(BT, N, 3, nh, hd)
        return (self._rope(w[:, :, 0].transpose(1, 2), cos, sin),
                self._rope(w[:, :, 1].transpose(1, 2), cos, sin),
                w[:, :, 2].transpose(1, 2).contiguous())

    def _qkv_rope_tp(self, qkv, cos, sin, B, time, N, nh, hd):
        """qkv -> rotated q, k and plain v, all as (B*N, nh, T, hd) contiguous."""
        if self._fused and qkv.is_cuda:
            shape = (B * N, nh, time, hd)
            q = torch.empty(shape, dtype=qkv.dtype, device=qkv.device)
            k = torch.empty(shape, dtype=qkv.dtype, device=qkv.device)
            v = torch.empty(shape, dtype=qkv.dtype, device=qkv.device)
            hb = 4 if nh % 4 == 0 else 1
            tb = max(2, triton.next_power_of_2(time))
            _k_qkv_rope_tp[(B * N * (nh // hb),)](qkv, cos, sin, q, k, v, N,
                                                  T=time, NH=nh, HD=hd, TB=tb, HB=hb,
                                                  NHB=nh // hb, num_warps=4)
            return q, k, v
        w = qkv.view(B, time, N, 3, nh, hd)
        shape = (B * N, nh, time, hd)
        return (self._rope(w[:, :, :, 0].permute(0, 2, 3, 1, 4), cos, sin).view(shape),
                self._rope(w[:, :, :, 1].permute(0, 2, 3, 1, 4), cos, sin).view(shape),
                w[:, :, :, 2].permute(0, 2, 3, 1, 4).contiguous().view(shape))

    # ------------------------------------------------------------------- body
    def _run(self, xin, tin, ecin, B, time):
        st = self._static
        D = self.hidden_size
        nh = self.num_heads
        hd = self.head_dim
        N = st["n_tok"]
        h_grid, w_grid = st["grid"]
        p = self.patch_size
        C = self.in_channels
        BT = B * time
        S = BT * N

        # ---- patch embed -------------------------------------------------
        # Kept as the reference's conv2d.  The stride == kernel means this is
        # exactly a linear map over the flattened (C, p, p) patch, and the
        # equivalent GEMM is ~1.6x cheaper here -- but cuDNN and cuBLAS disagree
        # in the last ULP, and that alone fails the output comparison.
        emb = F.conv2d(xin.reshape(BT, C, h_grid * p, w_grid * p), st["pe_w"], st["pe_b"], stride=(p, p))
        x = emb.permute(0, 2, 3, 1).reshape(S, D)
        x4 = x.view(B, time, N, D)

        # ---- conditioning ------------------------------------------------
        args = tin.reshape(BT).to(x.dtype).unsqueeze(1) * st["ts_freqs"]
        temb = torch.cat((args.cos(), args.sin()), dim=-1)
        mlp = self.t_embedder.mlp
        hc = F.silu(torch.addmm(mlp[0].bias, temb, mlp[0].weight.t()))
        c = torch.addmm(mlp[2].bias, hc, mlp[2].weight.t())
        if ecin is not None:
            lin = self.external_cond
            c = c + torch.addmm(lin.bias, ecin.reshape(BT, -1), lin.weight.t())
        sc = F.silu(c)

        # All 33 adaLN projections in one GEMM; ``1 + scale`` for all at once.
        mod = torch.addmm(st["mod_b"], sc, st["mod_w"].t()).view(B, time, 2 * self.depth, 6, D)
        fmod = torch.addmm(st["fin_b"], sc, st["fin_w"].t()).view(B, time, 2, D)
        mod[:, :, :, 1::3].add_(1.0)
        fmod[:, :, 1].add_(1.0)

        tc = self._temporal_cache(time, x.device, x.dtype)

        for i, w in enumerate(st["blocks"]):
            (qkv_s, out_s_w, out_s_b, f1s_w, f1s_b, f2s_w, f2s_b,
             qkv_t, out_t_w, out_t_b, f1t_w, f1t_b, f2t_w, f2t_b) = w

            # ---- spatial attention -------------------------------------
            m = mod[:, :, 2 * i]
            hx = F.layer_norm(x, (D,), None, None, 1e-6).view(B, time, N, D)
            hx = self._modulate(hx, m[:, :, 0], m[:, :, 1])
            qkv = torch.mm(hx.view(S, D), qkv_s.t())
            q, k, v = self._qkv_rope_sp(qkv, st["sp_cos"], st["sp_sin"], BT, N, nh, hd)
            att = F.scaled_dot_product_attention(q, k, v).view(BT, nh, N, hd).transpose(1, 2).reshape(S, D)
            att = torch.addmm(out_s_b, att, out_s_w.t()).view(B, time, N, D)
            self._gate_add(x4, att, m[:, :, 2])

            # ---- spatial mlp -------------------------------------------
            hx = F.layer_norm(x, (D,), None, None, 1e-6).view(B, time, N, D)
            hx = self._modulate(hx, m[:, :, 3], m[:, :, 4])
            u = F.gelu(torch.addmm(f1s_b, hx.view(S, D), f1s_w.t()), approximate="tanh")
            u = torch.addmm(f2s_b, u, f2s_w.t()).view(B, time, N, D)
            self._gate_add(x4, u, m[:, :, 5])

            # ---- temporal attention ------------------------------------
            m = mod[:, :, 2 * i + 1]
            hx = F.layer_norm(x, (D,), None, None, 1e-6).view(B, time, N, D)
            hx = self._modulate(hx, m[:, :, 0], m[:, :, 1])
            qkv = torch.mm(hx.view(S, D), qkv_t.t())
            q, k, v = self._qkv_rope_tp(qkv, tc["cos"], tc["sin"], B, time, N, nh, hd)
            att = F.scaled_dot_product_attention(q, k, v, is_causal=True).view(B, N, nh, time, hd).permute(0, 3, 1, 2, 4).reshape(S, D)
            att = torch.addmm(out_t_b, att, out_t_w.t()).view(B, time, N, D)
            self._gate_add(x4, att, m[:, :, 2])

            # ---- temporal mlp ------------------------------------------
            hx = F.layer_norm(x, (D,), None, None, 1e-6).view(B, time, N, D)
            hx = self._modulate(hx, m[:, :, 3], m[:, :, 4])
            u = F.gelu(torch.addmm(f1t_b, hx.view(S, D), f1t_w.t()), approximate="tanh")
            u = torch.addmm(f2t_b, u, f2t_w.t()).view(B, time, N, D)
            self._gate_add(x4, u, m[:, :, 5])

        # ---- final layer + unpatchify ------------------------------------
        hx = F.layer_norm(x, (D,), None, None, 1e-6).view(B, time, N, D)
        hx = self._modulate(hx, fmod[:, :, 0], fmod[:, :, 1])
        fl = self.final_layer.linear
        y = torch.addmm(fl.bias, hx.view(S, D), fl.weight.t())
        y = y.view(BT, h_grid, w_grid, p, p, C).permute(0, 5, 1, 3, 2, 4)
        return y.reshape(B, time, C, h_grid * p, w_grid * p)

    # ------------------------------------------------------------ entry point
    def forward(self, x: torch.Tensor, t: torch.Tensor, external_cond: torch.Tensor | None = None) -> torch.Tensor:
        has_cond = torch.is_tensor(external_cond)
        ec = external_cond if has_cond else None
        if not x.is_cuda:
            if self._static is None:
                self._build_static(x.device, x.dtype)
            with torch.no_grad():
                return self._run(x, t, ec, x.shape[0], x.shape[1])

        key = (x.shape[0], x.shape[1], has_cond, x.dtype)
        plan = self._plans.get(key)
        if plan is None:
            plan = self._capture(x, t, ec, key)
            if plan is None:
                # Capture is the whole point, but a correct answer beats none:
                # if the driver refuses to capture, run the same op stream eagerly.
                with torch.no_grad():
                    return self._run(x, t, ec, x.shape[0], x.shape[1])
        plan.x.copy_(x, non_blocking=True)
        plan.t.copy_(t, non_blocking=True)
        if has_cond:
            plan.ec.copy_(ec, non_blocking=True)
        plan.graph.replay()
        return plan.out

    def _capture(self, x, t, ec, key):
        """Build the static buffers and capture ``_run`` for one signature.

        Returns ``None`` if capture is unavailable, so ``forward`` can fall back.
        """
        if self._nograph:
            return None
        try:
            return self._capture_inner(x, t, ec, key)
        except Exception:  # noqa: BLE001 - any capture failure -> eager fallback
            self._nograph = True
            torch.cuda.synchronize()
            return None

    def _capture_inner(self, x, t, ec, key) -> _Plan:
        B, time = x.shape[0], x.shape[1]
        with torch.no_grad():
            if self._static is None:
                self._build_static(x.device, x.dtype)
            self._temporal_cache(time, x.device, x.dtype)
            plan = _Plan()
            plan.x = x.detach().clone()
            plan.t = t.detach().clone()
            plan.ec = ec.detach().clone() if ec is not None else None

            # Warm up cuBLAS handles/workspaces and any lazy kernel loads on a
            # side stream -- capture cannot tolerate them happening inline.
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._run(plan.x, plan.t, plan.ec, B, time)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()

            plan.pool = torch.cuda.graph_pool_handle()
            plan.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(plan.graph, pool=plan.pool):
                plan.out = self._run(plan.x, plan.t, plan.ec, B, time)
            torch.cuda.synchronize()
        self._plans[key] = plan
        return plan
