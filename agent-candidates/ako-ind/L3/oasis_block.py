"""Oasis DiT blocks -- launch-collapsed rewrite.

The captured shapes are tiny (T=2..6 frames of 9x16 tokens, ~0.6-1.8 MB of
activations, tens of microseconds of real GEMM on B200) while the eager
reference issues several hundred kernels per forward, so wall time is set by
dispatch and elementwise round-trips rather than by matmul efficiency.

Three structural changes:

  * The four ``LayerNorm -> modulate`` / ``gate -> residual`` glue chains
    become one Triton kernel each. The conditioning row is indexed per frame
    inside the kernel, so the reference's ``.repeat()`` / ``unsqueeze()``
    broadcast materialization disappears entirely.
  * The spatial and temporal adaLN projections are concatenated at first use
    into a single 1024->12288 SiLU+GEMM, and both axial attentions are single
    Triton kernels that fold RoPE into the Q/K load and read the packed qkv
    tensor with strided offsets -- no permute/reshape copies anywhere.
  * Every shape-dependent constant (rotary cos/sin tables, packed weights) is
    hoisted out of ``forward`` and cached, and the whole residual chain is
    captured into one CUDA graph per input signature.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.oasis_rotary import OasisRotaryEmbedding
from ..L1.silu import SiLU
from ..L2.oasis_mlp import OasisMLP
from ..L2.oasis_spatial_axial_attention import OasisSpatialAxialAttention
from ..L2.oasis_temporal_axial_attention import OasisTemporalAxialAttention

_NEG = tl.constexpr(-1e30)
_USE_GRAPH = os.environ.get("OASIS_NO_GRAPH", "0") != "1"


# ---------------------------------------------------------------------------
# Prologue: bring x from its captured [t][c][h][w] layout into a contiguous
# [tokens, dim] buffer, and apply the adaLN SiLU to c. Two unrelated jobs in one
# launch -- both are latency-bound at these sizes (the SiLU alone is 12 KB, and
# any standalone kernel costs ~2.5 us of device time), so the tail CTAs of the
# transpose grid do the SiLU for free.
# ---------------------------------------------------------------------------
@triton.jit
def _prologue_kernel(X, Y, C, CS, T, HW, D, sx_t, sx_p, sx_d, sc_m, NT_P, NT_D,
                     BM: tl.constexpr, BN: tl.constexpr, BC: tl.constexpr):
    pid = tl.program_id(0)
    ntile = T * NT_P * NT_D
    if pid < ntile:
        per_t = NT_P * NT_D
        t = pid // per_t
        rem = pid % per_t
        op = (rem // NT_D) * BM + tl.arange(0, BM)
        od = (rem % NT_D) * BN + tl.arange(0, BN)
        mp = op < HW
        md = od < D
        # Read [dim, pos] (pos contiguous in the source), store [pos, dim].
        v = tl.load(X + t * sx_t + od[:, None] * sx_d + op[None, :] * sx_p,
                    mask=md[:, None] & mp[None, :], other=0.0)
        tl.store(Y + (t * HW + op[:, None]) * D + od[None, :], tl.trans(v),
                 mask=mp[:, None] & md[None, :])
    else:
        row = pid - ntile
        o = tl.arange(0, BC)
        m = o < D
        z = tl.load(C + row * sc_m + o, mask=m, other=0.0).to(tl.float32)
        tl.store(CS + row * D + o, (z * tl.sigmoid(z)).to(CS.dtype.element_ty), mask=m)


# ---------------------------------------------------------------------------
# Glue kernels. ``P`` is the packed [T, 12*D] conditioning; the frame index is
# derived from the token id so shift/scale/gate are read broadcast, never
# materialized.
# ---------------------------------------------------------------------------
@triton.jit
def _ln_mod_kernel(X, P, Y, HW, EPS, sp_m, off_shift, off_scale,
                   D: tl.constexpr, BLOCK: tl.constexpr):
    n = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    mask = d < D
    x = tl.load(X + n * D + d, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, 0) / D
    xc = tl.where(mask, x - mean, 0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(xc * xc, 0) / D + EPS)
    xn = (xc * rstd).to(Y.dtype.element_ty).to(tl.float32)
    prow = P + (n // HW) * sp_m + d
    shift = tl.load(prow + off_shift, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(prow + off_scale, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + n * D + d, (xn * (1.0 + scale) + shift).to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _gate_res_ln_kernel(XP, R, P, XO, Y, HW, EPS, sp_m,
                        off_gate, off_shift, off_scale,
                        D: tl.constexpr, BLOCK: tl.constexpr, WITH_LN: tl.constexpr):
    n = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    mask = d < D
    prow = P + (n // HW) * sp_m + d
    g = tl.load(prow + off_gate, mask=mask, other=0.0).to(tl.float32)
    xp = tl.load(XP + n * D + d, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R + n * D + d, mask=mask, other=0.0).to(tl.float32)
    xh = (xp + g * r).to(XO.dtype.element_ty)
    tl.store(XO + n * D + d, xh, mask=mask)
    if WITH_LN:
        x = xh.to(tl.float32)
        mean = tl.sum(x, 0) / D
        xc = tl.where(mask, x - mean, 0.0)
        rstd = 1.0 / tl.sqrt(tl.sum(xc * xc, 0) / D + EPS)
        xn = (xc * rstd).to(Y.dtype.element_ty).to(tl.float32)
        shift = tl.load(prow + off_shift, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(prow + off_scale, mask=mask, other=0.0).to(tl.float32)
        tl.store(Y + n * D + d, (xn * (1.0 + scale) + shift).to(Y.dtype.element_ty),
                 mask=mask)


# ---------------------------------------------------------------------------
# Axial attention. RoPE is applied on the Q/K registers right after the load,
# so there is no separate elementwise pass and no rotate-half materialization.
# ---------------------------------------------------------------------------
@triton.jit
def _rope(x, cos, sin, M: tl.constexpr, DH: tl.constexpr):
    x0, x1 = tl.split(tl.reshape(x, (M, DH // 2, 2)))
    rot = tl.reshape(tl.join(-x1, x0), (M, DH))
    return x.to(tl.float32) * cos.to(tl.float32) + rot.to(tl.float32) * sin.to(tl.float32)


@triton.jit
def _spatial_attn_kernel(QKV, O, COS, SIN, S, H, sq_n, KOFF, VOFF, SCALE,
                         DH: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    frame = tl.program_id(1) // H
    head = tl.program_id(1) % H
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    d = tl.arange(0, DH)
    qp = QKV + (frame * S + offs_m[:, None]) * sq_n + (head * DH + d[None, :])
    q = tl.load(qp, mask=mask_m[:, None], other=0.0)
    cq = tl.load(COS + offs_m[:, None] * DH + d[None, :], mask=mask_m[:, None], other=0.0)
    sq = tl.load(SIN + offs_m[:, None] * DH + d[None, :], mask=mask_m[:, None], other=0.0)
    q = _rope(q, cq, sq, BLOCK_M, DH).to(QKV.dtype.element_ty)

    acc = tl.zeros((BLOCK_M, DH), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), _NEG, tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    for n0 in range(0, S, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        base = QKV + (frame * S + offs_n[:, None]) * sq_n + (head * DH + d[None, :])
        k = tl.load(base + KOFF, mask=mask_n[:, None], other=0.0)
        ck = tl.load(COS + offs_n[:, None] * DH + d[None, :], mask=mask_n[:, None], other=0.0)
        sk = tl.load(SIN + offs_n[:, None] * DH + d[None, :], mask=mask_n[:, None], other=0.0)
        k = _rope(k, ck, sk, BLOCK_N, DH).to(QKV.dtype.element_ty)
        v = tl.load(base + VOFF, mask=mask_n[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * SCALE
        s = tl.where(mask_n[None, :], s, _NEG)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        pr = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(pr, 1)
        acc = acc * alpha[:, None] + tl.dot(pr.to(QKV.dtype.element_ty), v)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(O + (frame * S + offs_m[:, None]) * (H * DH) + (head * DH + d[None, :]),
             acc.to(O.dtype.element_ty), mask=mask_m[:, None])


@triton.jit
def _temporal_attn_kernel(QKV, O, COS, SIN, T, HW, H, sq_n, KOFF, VOFF, SCALE,
                          IS_CAUSAL: tl.constexpr, DH: tl.constexpr, BT: tl.constexpr):
    pos = tl.program_id(0)
    head = tl.program_id(1)
    row0 = tl.program_id(2) * T * HW + pos  # batch item b starts at b*T*HW
    tt = tl.arange(0, BT)
    mt = tt < T
    d = tl.arange(0, DH)
    base = QKV + (row0 + tt[:, None] * HW) * sq_n + (head * DH + d[None, :])
    q = tl.load(base, mask=mt[:, None], other=0.0)
    k = tl.load(base + KOFF, mask=mt[:, None], other=0.0)
    v = tl.load(base + VOFF, mask=mt[:, None], other=0.0)
    cs = tl.load(COS + tt[:, None] * DH + d[None, :], mask=mt[:, None], other=0.0)
    sn = tl.load(SIN + tt[:, None] * DH + d[None, :], mask=mt[:, None], other=0.0)
    q = _rope(q, cs, sn, BT, DH).to(QKV.dtype.element_ty)
    k = _rope(k, cs, sn, BT, DH).to(QKV.dtype.element_ty)

    ok = mt[None, :]
    if IS_CAUSAL:
        ok = ok & (tt[None, :] <= tt[:, None])
    s = tl.where(ok, tl.dot(q, tl.trans(k)) * SCALE, _NEG)
    e = tl.exp(s - tl.max(s, 1)[:, None])
    out = tl.dot((e / tl.sum(e, 1)[:, None]).to(QKV.dtype.element_ty), v)
    tl.store(O + (row0 + tt[:, None] * HW) * (H * DH) + (head * DH + d[None, :]),
             out.to(O.dtype.element_ty), mask=mt[:, None])


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------
# Column layout of the packed conditioning P[T, 12*D]: the six spatial chunks
# in reference order, then the six temporal ones.
_S_SHIFT_MSA, _S_SCALE_MSA, _S_GATE_MSA = 0, 1, 2
_S_SHIFT_MLP, _S_SCALE_MLP, _S_GATE_MLP = 3, 4, 5
_T_SHIFT_MSA, _T_SCALE_MSA, _T_GATE_MSA = 6, 7, 8
_T_SHIFT_MLP, _T_SCALE_MLP, _T_GATE_MLP = 9, 10, 11

# Tuned on B200 by device-time attribution (see ITERATIONS.md); event-bracketed
# timing of a single kernel here measures the ~12 us Triton launch path instead.
_PROLOG_BM, _PROLOG_BN, _PROLOG_W = 64, 64, 4
_GLUE_W = 4
# (BLOCK_N, num_warps, num_stages) per spatial BLOCK_M.
_SPAT_CFG = {16: (64, 4, 3), 32: (64, 4, 2), 64: (32, 2, 3)}


def _spat_block_m(n_pairs: int, seq: int) -> int:
    """Largest query tile that still leaves >~2 CTAs per SM.

    The spatial attention is latency-bound, not bandwidth-bound: at every
    captured shape the fastest tile is whichever one puts the CTA count in the
    250-500 range, and the four best configs at a given T sit within 5% of each
    other, so a CTA-count rule beats a per-T table fitted to that noise."""
    for bm in (16, 32, 64):
        if triton.cdiv(seq, bm) * n_pairs <= 400:
            return bm
    return 64


class SpatioTemporalDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        is_causal: bool = True,
        spatial_rotary_emb: OasisRotaryEmbedding,
        temporal_rotary_emb: OasisRotaryEmbedding,
    ):
        super().__init__()
        self.s_norm1 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.s_attn = OasisSpatialAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            rotary_emb=spatial_rotary_emb,
        )
        self.s_norm2 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.s_mlp = OasisMLP(
            hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            approximate_tanh=True,
        )
        self.s_adaLN_modulation = nn.Sequential(
            SiLU(),
            Linear(hidden_size, 6 * hidden_size, bias=True),
        )

        self.t_norm1 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.t_attn = OasisTemporalAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            rotary_emb=temporal_rotary_emb,
            is_causal=is_causal,
        )
        self.t_norm2 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.t_mlp = OasisMLP(
            hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            approximate_tanh=True,
        )
        self.t_adaLN_modulation = nn.Sequential(
            SiLU(),
            Linear(hidden_size, 6 * hidden_size, bias=True),
        )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dim_head = hidden_size // num_heads
        self.is_causal = is_causal
        # Lazily built (weights are loaded and cast after __init__).
        self._packed = None
        self._rope_s = {}
        self._rope_t = {}
        self._graphs = {}
        self._pool = None
        self._graph_ok = _USE_GRAPH
        self._sig = None

    def _param_sig(self):
        """Address + in-place-version of every parameter.

        Two cached things outlive a weight change and must be rebuilt when it
        happens: the packed adaLN copy and the rotary cos/sin tables. A captured
        graph additionally bakes in the *addresses* of everything it reads, so a
        parameter whose storage was swapped invalidates the graph too. Weights
        read straight out of ``self`` would be picked up by a replay for free,
        but distinguishing those is not worth the extra bookkeeping -- the check
        is ~4 us of host time that the benchmark hides behind its L2 flush, and
        it never fires once weights are loaded."""
        return tuple((p.data_ptr(), p._version) for p in self.parameters())

    def _check_params(self):
        sig = self._param_sig()
        if sig != self._sig:
            self._sig = sig
            self._graphs = {}
            # Dropping the graphs frees the shared mempool; reusing its handle
            # afterwards trips an allocator assert, so take a fresh one.
            self._pool = None
            self._packed = None
            self._rope_s = {}
            self._rope_t = {}

    # -- cached constants ---------------------------------------------------
    def _packed_weights(self):
        """Concatenate the two adaLN projections into one 1024 -> 12288 GEMM.

        Rebuilt if a parameter object was replaced (``load_state_dict`` copies
        in place, but ``.to()`` / reassignment would swap the tensor).
        """
        sw = self.s_adaLN_modulation[1].weight
        tw = self.t_adaLN_modulation[1].weight
        pk = self._packed
        ver = (sw._version, tw._version)
        if (pk is not None and pk["src"] is sw and pk["src_t"] is tw
                and pk["ver"] == ver):
            return pk
        sb = self.s_adaLN_modulation[1].bias
        tb = self.t_adaLN_modulation[1].bias
        with torch.no_grad():
            # [K, 12*D]: addmm against a K-major B is marginally faster than
            # F.linear's transposed form, and cuBLAS's GEMV kernel reaches
            # 3.2 TB/s on the 25 MB weight -- well past anything hand-written
            # here, so this projection stays on the library path.
            pk = {
                "src": sw,
                "src_t": tw,
                "ver": ver,
                "wt": torch.cat([sw.detach(), tw.detach()], 0).t().contiguous(),
                "b": torch.cat([sb.detach(), tb.detach()], 0).contiguous(),
            }
        self._packed = pk
        return pk

    def _rope_tables_spatial(self, height: int, width: int):
        """cos/sin for the axial pixel frequencies, computed once per (h, w).

        Built by calling the reference rotary module so the fp16 einsum /
        repeat_interleave rounding matches the baseline exactly.
        """
        rot = self.s_attn.rotary_emb
        key = (height, width, rot.freqs.data_ptr(), rot.freqs.dtype,
               rot.freqs._version)
        hit = self._rope_s.get(key)
        if hit is None:
            with torch.no_grad():
                freqs = rot.get_axial_freqs(height, width)
                hit = (freqs.cos().reshape(height * width, -1).contiguous(),
                       freqs.sin().reshape(height * width, -1).contiguous())
            self._rope_s = {key: hit}
        return hit

    def _rope_tables_temporal(self, time: int, dtype, device):
        rot = self.t_attn.rotary_emb
        key = (time, rot.freqs.data_ptr(), rot.freqs.dtype, dtype,
               rot.freqs._version)
        hit = self._rope_t.get(key)
        if hit is None:
            with torch.no_grad():
                pos = torch.arange(time, device=device, dtype=dtype)
                freqs = rot.forward(pos, rot.freqs, seq_len=time)
                hit = (freqs.cos().contiguous(), freqs.sin().contiguous())
            self._rope_t = {key: hit}
        return hit

    # -- fused forward ------------------------------------------------------
    def _attn_branch(self, y, attn, cos, sin, bsz, time, hw, spatial):
        """qkv -> (RoPE + attention) -> out-proj, returning the branch output."""
        dim = y.shape[1]
        heads, dh = self.num_heads, self.dim_head
        qkv = F.linear(y, attn.to_qkv.weight)
        o = torch.empty_like(y)
        scale = dh ** -0.5
        if spatial:
            bm = _spat_block_m(bsz * time * heads, hw)
            bn, nw, ns = _SPAT_CFG[bm]
            _spatial_attn_kernel[(triton.cdiv(hw, bm), bsz * time * heads)](
                qkv, o, cos, sin, hw, heads, 3 * dim, dim, 2 * dim, scale,
                DH=dh, BLOCK_M=bm, BLOCK_N=bn, num_warps=nw, num_stages=ns,
            )
        else:
            # tl.dot needs M >= 16; T <= 6 so one padded 16x16 score tile covers
            # the whole causal sequence and the two dots are effectively free.
            _temporal_attn_kernel[(hw, heads, bsz)](
                qkv, o, cos, sin, time, hw, heads, 3 * dim, dim, 2 * dim, scale,
                IS_CAUSAL=self.is_causal, DH=dh,
                BT=max(16, triton.next_power_of_2(time)), num_warps=1, num_stages=1,
            )
        return F.linear(o, attn.to_out.weight, attn.to_out.bias)

    def _prologue(self, x, c, xc, cs):
        """x (captured [t][c][h][w] layout) -> contiguous xc; SiLU(c) -> cs.

        Reads x and c through pointers passed at launch, so this one kernel also
        serves as the CUDA graph's input staging step -- it replaces the pair of
        ``static.copy_(...)`` calls a graph normally needs, which together cost
        ~15 us of host time and an extra full pass over x.
        """
        bsz, time, height, width, dim = x.shape
        hw = height * width
        st = x.stride()
        # Need (h, w) to collapse into one position axis, and (b, t) into one
        # frame axis, so a single (frame, pos, chan) index covers the tensor.
        if st[3] * width != st[2] or (bsz > 1 and st[0] != time * st[1]):
            return False
        nframe = bsz * time
        nt_p = triton.cdiv(hw, _PROLOG_BM)
        nt_d = triton.cdiv(dim, _PROLOG_BN)
        _prologue_kernel[(nframe * nt_p * nt_d + nframe,)](
            x, xc, c, cs, nframe, hw, dim, st[1], st[3], st[4], c.stride(1),
            nt_p, nt_d, BM=_PROLOG_BM, BN=_PROLOG_BN,
            BC=triton.next_power_of_2(dim), num_warps=_PROLOG_W,
        )
        return True

    def _body(self, xc, cs, bsz, time, height, width, dim):
        hw = height * width
        n_tok = xc.shape[0]
        heads = self.num_heads
        eps = self.s_norm1.eps
        pk = self._packed_weights()
        cos_s, sin_s = self._rope_tables_spatial(height, width)
        cos_t, sin_t = self._rope_tables_temporal(time, xc.dtype, xc.device)

        # One 1024 -> 12288 GEMM for both adaLN projections.
        P = torch.addmm(pk["b"], cs, pk["wt"])

        y = torch.empty_like(xc)
        _ln_mod_kernel[(n_tok,)](
            xc, P, y, hw, eps, P.stride(0),
            _S_SHIFT_MSA * dim, _S_SCALE_MSA * dim,
            D=dim, BLOCK=triton.next_power_of_2(dim), num_warps=_GLUE_W,
        )

        branches = (
            (self.s_attn, cos_s, sin_s, True, _S_GATE_MSA, _S_SHIFT_MLP, _S_SCALE_MLP),
            (self.s_mlp, None, None, None, _S_GATE_MLP, _T_SHIFT_MSA, _T_SCALE_MSA),
            (self.t_attn, cos_t, sin_t, False, _T_GATE_MSA, _T_SHIFT_MLP, _T_SCALE_MLP),
            (self.t_mlp, None, None, None, _T_GATE_MLP, 0, 0),
        )
        for i, (mod, cos, sin, spatial, g_off, sh_off, sc_off) in enumerate(branches):
            if spatial is None:
                r = F.linear(
                    F.gelu(F.linear(y, mod.fc1.weight, mod.fc1.bias), approximate="tanh"),
                    mod.fc2.weight, mod.fc2.bias)
            else:
                r = self._attn_branch(y, mod, cos, sin, bsz, time, hw, spatial)
            last = i == len(branches) - 1
            xo = torch.empty_like(xc)
            _gate_res_ln_kernel[(n_tok,)](
                xc, r, P, xo, y, hw, eps, P.stride(0),
                g_off * dim, sh_off * dim, sc_off * dim,
                D=dim, BLOCK=triton.next_power_of_2(dim),
                WITH_LN=not last, num_warps=_GLUE_W,
            )
            xc = xo
        return xc.view(bsz, time, height, width, dim)

    def _bufs(self, x, c):
        bsz, time, height, width, dim = x.shape
        n_tok = bsz * time * height * width
        return (torch.empty((n_tok, dim), dtype=x.dtype, device=x.device),
                torch.empty((bsz * time, dim), dtype=c.dtype, device=c.device))

    def _run(self, x, c):
        bsz, time, height, width, dim = x.shape
        xc, cs = self._bufs(x, c)
        if not self._prologue(x, c, xc, cs):
            xc.copy_(x.reshape(xc.shape[0], dim))
            cs.copy_(F.silu(c.reshape(-1, dim)))
        return self._body(xc, cs, bsz, time, height, width, dim)

    # -- graph wrapper ------------------------------------------------------
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        self._check_params()
        if not (self._graph_ok and x.is_cuda):
            return self._run(x, c)
        key = (tuple(x.shape), x.stride(), x.dtype,
               tuple(c.shape), c.stride(), c.dtype, x.device)
        st = self._graphs.get(key)
        if st is None:
            st = {"n": 0, "g": None}
            self._graphs[key] = st
        if st["g"] is not None:
            self._prologue(x, c, st["xc"], st["cs"])
            st["g"].replay()
            return st["out"]
        st["n"] += 1
        if st["n"] < 2 or st["n"] > 2 or torch.cuda.is_current_stream_capturing():
            return self._run(x, c)
        try:
            return self._capture(st, x, c)
        except Exception:
            # Give up on graphing this signature only; other shapes may still
            # capture fine. st["n"] is already past the retry point.
            torch.cuda.synchronize()
            return self._run(x, c)

    def _capture(self, st, x, c):
        bsz, time, height, width, dim = x.shape
        xc, cs = self._bufs(x, c)
        if not self._prologue(x, c, xc, cs):
            raise RuntimeError("layout not supported by the graph path")
        args = (xc, cs, bsz, time, height, width, dim)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._body(*args)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self._pool):
            out = self._body(*args)
        st.update(g=g, xc=xc, cs=cs, out=out)
        # Capture records without executing, so out holds garbage until a replay.
        g.replay()
        return out
