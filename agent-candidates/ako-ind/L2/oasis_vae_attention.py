"""Oasis VAE self-attention."""

from __future__ import annotations

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from ..L1.dense_attention import DenseAttention
from ..L1.linear import Linear
from ..L1.oasis_rotary import OasisRotaryEmbedding, oasis_apply_rotary_emb

try:
    from torch._C import _cuda_getCurrentRawStream as _raw_stream
except ImportError:  # pragma: no cover - torch always ships this on CUDA builds
    _raw_stream = None

# Flash-kernel tile, tuned on B200 for (seq=576, head_dim=64, non-causal, fp16).
# BM=64 is the point of the kernel: it is what puts bsz=1 on one full wave.
_BM, _BN, _NUM_WARPS, _NUM_STAGES = 64, 64, 4, 3

_ANNOUNCED: set = set()


def _announce(p) -> None:
    """Say once per (shape, decision) which paths a plan ended up on.

    Both fast paths degrade rather than fail, and r1 spent a whole iteration on
    a silent degrade that looked exactly like "the change bought nothing", so
    the decision is visible without having to instrument anything.
    """
    key = (tuple(p.out_shape), p.dtype, p.run.__name__, p.attn_mode)
    if key in _ANNOUNCED:
        return
    _ANNOUNCED.add(key)
    print(f"[OasisVAEAttention] {tuple(p.out_shape)} {p.dtype} "
          f"rope={p.run.__name__} attn={p.attn_mode}", file=sys.stderr, flush=True)


@triton.jit
def _rope_split(
    QKV,                    # [b*SEQ, 3*DIM]  packed qkv GEMM output
    OUT,                    # [3, b*SEQ, DIM] contiguous q | k | v
    COS, SIN,               # [SEQ, HDP] fp32, padded with cos=1 / sin=0
    SEQ: tl.constexpr, DIM: tl.constexpr, HD: tl.constexpr, HDP: tl.constexpr,
    PART: tl.constexpr,     # element stride between the q/k/v planes of OUT
    BS: tl.constexpr, FULL: tl.constexpr,
):
    """Split a packed qkv GEMM output into contiguous q/k/v, rotating q and k.

    q/k/v come out in exactly the [b, seq, heads, head_dim] layout the GEMM
    already emits -- the reference path's 5D permute round trip cancels itself
    out -- so the only real work here is the rotary, done in fp32 registers
    against precomputed cos/sin tables.  Padding those tables with cos=1/sin=0
    past ``rot_dim`` lets the untouched high lanes ride the same fused
    expression (``x * 1 + p * 0`` is exact in fp32) instead of needing a
    load/store pair of their own, and keeps every access a full-width
    contiguous row.
    """
    pid_h = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_b = tl.program_id(2)

    sq = pid_s * BS + tl.arange(0, BS)              # position inside the frame
    row = pid_b * SEQ + sq                          # flat row of the GEMM output
    d = tl.arange(0, HDP)
    src = QKV + row[:, None] * (3 * DIM) + pid_h * HD + d[None, :]
    dst = OUT + row[:, None] * DIM + pid_h * HD + d[None, :]
    tab = sq[:, None] * HDP + d[None, :]

    if FULL:                                        # SEQ % BS == 0 and HD == HDP
        c = tl.load(COS + tab)
        sn = tl.load(SIN + tab)
        for i in tl.static_range(2):                # q, then k
            a = tl.load(src + i * DIM)
            a0, a1 = tl.split(tl.reshape(a, (BS, HDP // 2, 2)))
            p = tl.reshape(tl.join(-a1, a0), (BS, HDP))
            tl.store(dst + i * PART,
                     (a.to(tl.float32) * c + p.to(tl.float32) * sn).to(OUT.dtype.element_ty))
        tl.store(dst + 2 * PART, tl.load(src + 2 * DIM))
    else:
        m = (sq < SEQ)[:, None] & (d < HD)[None, :]
        c = tl.load(COS + tab, mask=m, other=1.0)
        sn = tl.load(SIN + tab, mask=m, other=0.0)
        for i in tl.static_range(2):
            a = tl.load(src + i * DIM, mask=m, other=0.0)
            a0, a1 = tl.split(tl.reshape(a, (BS, HDP // 2, 2)))
            p = tl.reshape(tl.join(-a1, a0), (BS, HDP))
            tl.store(dst + i * PART,
                     (a.to(tl.float32) * c + p.to(tl.float32) * sn).to(OUT.dtype.element_ty),
                     mask=m)
        tl.store(dst + 2 * PART, tl.load(src + 2 * DIM, mask=m, other=0.0), mask=m)


@triton.jit
def _attn_fwd(
    QKV,                    # flat q | k | v, PART elements apart, each (b,seq,h,hd)
    O,                      # [b*SEQ, DIM] output, written head-minor
    SEQ: tl.constexpr, DIM: tl.constexpr, HD: tl.constexpr, HDP: tl.constexpr,
    PART: tl.constexpr,     # element stride between the q/k/v planes of QKV
    NH: tl.constexpr,       # heads, to split the (batch, head) program id
    QK_SCALE: tl.constexpr, # 1/sqrt(head_dim) * log2(e)
    BM: tl.constexpr, BN: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_D: tl.constexpr,
):
    """Non-causal flash attention, read straight out of the fused rope workspace.

    Written for the case cuDNN handles worst.  cuDNN's sm100 flash kernel tiles
    128 query rows, so at bsz=1 it launches only b*h*ceil(576/128) = 80 CTAs on
    148 SMs and spends ~11us on 1.36 GFLOP (~125 TFLOPS); a 64-row tile gives
    9*16 = 144 CTAs, one full wave, and the same work lands in ~9us.  At bsz=6
    both are ~4-6 waves deep and cuDNN's tiling wins by a wide margin, so the
    plan only binds this kernel where the occupancy arithmetic says cuDNN is
    starved.

    q/k/v are indexed with row stride DIM and head offset ``pid_h * HD``, i.e.
    the (b, seq, heads, head_dim) layout the rope kernel already writes -- no
    repack -- and the result is stored in that same order, so the proj GEMM
    consumes it as (b*seq, DIM) for free.  That is the free-view property r1
    obtained from cuDNN's layout propagation, here simply chosen.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // NH
    pid_h = pid_bh % NH

    d = tl.arange(0, HDP)
    dm = d < HD
    m_off = pid_m * BM + tl.arange(0, BM)
    mm = m_off < SEQ
    qbase = QKV + pid_b * (SEQ * DIM) + pid_h * HD
    qp = qbase + m_off[:, None] * DIM + d[None, :]
    if EVEN_M and EVEN_D:
        q = tl.load(qp)
    else:
        q = tl.load(qp, mask=mm[:, None] & dm[None, :], other=0.0)

    m_i = tl.full([BM], -float("inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, HDP], tl.float32)

    for start_n in tl.range(0, SEQ, BN):
        n_off = start_n + tl.arange(0, BN)
        kp = qbase + PART + n_off[:, None] * DIM + d[None, :]
        if EVEN_N and EVEN_D:
            k = tl.load(kp)
            v = tl.load(kp + PART)
        else:
            msk = (n_off < SEQ)[:, None] & dm[None, :]
            k = tl.load(kp, mask=msk, other=0.0)
            v = tl.load(kp + PART, mask=msk, other=0.0)
        # log2(e) is folded into QK_SCALE so the softmax runs on exp2.
        s = tl.dot(q, tl.trans(k)) * QK_SCALE
        if not EVEN_N:
            s = tl.where((n_off < SEQ)[None, :], s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(QKV.dtype.element_ty), v, acc)
        m_i = m_new

    acc = acc / l_i[:, None]
    op = O + (pid_b * SEQ + m_off)[:, None] * DIM + pid_h * HD + d[None, :]
    if EVEN_M and EVEN_D:
        tl.store(op, acc.to(O.dtype.element_ty))
    else:
        tl.store(op, acc.to(O.dtype.element_ty), mask=mm[:, None] & dm[None, :])


class _Plan:
    """Everything ``forward`` needs for one input shape, resolved once.

    ~90 of the captured calls run this op at only ~37 GFLOP, so eager dispatch
    (not math) sets the floor: every attribute chain, allocation and shape
    computation that does not depend on the input *values* is hoisted in here.
    """

    __slots__ = ("idx", "attn", "attn_mode", "dtype", "device", "fallback", "rows", "dim", "hd", "out_shape",
                 "qkv_b", "qkv_w", "proj_b", "proj_w", "cos", "sin",
                 "run", "run_args", "buf", "q", "k", "v", "flat_shape", "flat_stride")

    def __init__(self, dtype=None, device=None, fallback=False):
        self.dtype = dtype
        self.device = device
        self.fallback = fallback
        self.attn = None            # bound flash launcher, or None to use cuDNN
        self.attn_mode = "cudnn"


class OasisVAEAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        frame_height: int,
        frame_width: int,
        *,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = Linear(dim, dim, bias=True)
        self.rotary = OasisRotaryEmbedding(
            dim=(dim // num_heads) // 4,
            freqs_for="pixel",
            max_freq=frame_height * frame_width,
        )
        self.register_buffer(
            "rotary_freqs",
            self.rotary.get_axial_freqs(frame_height, frame_width),
            persistent=False,
        )
        self.attn = DenseAttention(backend="sdpa")

        self.dim = dim
        self.head_dim = hd = dim // num_heads
        self.seq_len = seq = frame_height * frame_width
        self.rot_dim = rot = int(self.rotary_freqs.shape[-1])
        self.hdp = triton.next_power_of_2(hd)
        self.block = 16
        # The fused path assumes the rotary pairs adjacent lanes of the low
        # rot_dim lanes, and that frame_height x frame_width indexes
        # rotary_freqs row-major (which is what the reference's 5D reshape does).
        self._fusable = (
            rot % 2 == 0 and 0 < rot <= hd and num_heads * hd == dim
            and self.rotary_freqs.numel() == seq * rot
        )
        self._plans: dict[torch.Size, _Plan] = {}

    # ------------------------------------------------------------------ #
    # plan construction (cold path)
    # ------------------------------------------------------------------ #
    def _build_tables(self, device):
        """cos/sin of the frozen rotary buffer, padded to the kernel's lane width.

        ``rotary_freqs`` is a constant non-persistent buffer, so this is a pure
        function of it; it is materialized on *device* rather than at __init__
        on CPU so the values match the reference path's ``freqs.cos()`` exactly.
        """
        f = self.rotary_freqs.reshape(self.seq_len, self.rot_dim).to(device)
        cos = torch.ones(self.seq_len, self.hdp, device=device, dtype=torch.float32)
        sin = torch.zeros(self.seq_len, self.hdp, device=device, dtype=torch.float32)
        cos[:, : self.rot_dim] = f.cos()
        sin[:, : self.rot_dim] = f.sin()
        return cos, sin

    def _bind_launcher(self, p, grid, args, probe_in, probe_out):
        """Return a callable that launches the fused kernel with minimal host work.

        ``JITFunction.run`` re-derives the specialization key, rebinds arguments
        and reads two knob properties on *every* call -- around 9us of host time
        against a 2.4us kernel at bsz=1.  Nothing but the four pointers varies
        here, so the compiled kernel's raw launcher is called directly, with the
        result validated against the ordinary path below.  If Triton's internals
        are not shaped the way this expects, the ordinary path is kept.
        """
        def slow(qkv, buf, cos, sin):
            _rope_split[grid](qkv, buf, cos, sin, SEQ=args[0], DIM=args[1], HD=args[2],
                              HDP=args[3], PART=args[4], BS=args[5], FULL=args[6],
                              num_warps=8, num_stages=1)

        compiled = _rope_split[grid](
            probe_in, probe_out, p.cos, p.sin,
            SEQ=args[0], DIM=args[1], HD=args[2], HDP=args[3], PART=args[4],
            BS=args[5], FULL=args[6], num_warps=8, num_stages=1,
        )
        ref = probe_out.clone()
        if _raw_stream is None:
            return slow, ()
        try:
            compiled._init_handles()
            run, fn, meta = compiled.run, compiled.function, compiled.packed_metadata
            g0, g1, g2 = grid
            idx = p.idx
            probe_out.zero_()
            run(g0, g1, g2, _raw_stream(idx), fn, meta, None, None, None,
                probe_in, probe_out, p.cos, p.sin, *args)
            if not torch.equal(probe_out, ref):
                return slow, ()
        except Exception:
            return slow, ()

        def fast(qkv, buf, cos, sin, _args=args, _run=run, _fn=fn, _meta=meta,
                 _g=(g0, g1, g2), _idx=idx):
            _run(_g[0], _g[1], _g[2], _raw_stream(_idx), _fn, _meta, None, None, None,
                 qkv, buf, cos, sin, *_args)

        return fast, ()

    def _make_plan(self, x: torch.Tensor) -> _Plan:
        bsz, seq, dim = x.shape
        h, hd, rows = self.num_heads, self.head_dim, x.shape[0] * self.seq_len
        p = _Plan(x.dtype, x.device)
        idx = x.device.index
        p.idx = torch.cuda.current_device() if idx is None else idx
        p.rows, p.dim, p.hd = rows, dim, hd
        p.out_shape = (bsz, seq, dim)
        p.flat_shape, p.flat_stride = (rows, dim), (dim, 1)
        p.qkv_b, p.proj_b = self.qkv.bias, self.proj.bias
        p.qkv_w, p.proj_w = self.qkv.weight.t(), self.proj.weight.t()
        p.cos, p.sin = self._build_tables(x.device)
        # (b, heads, seq, head_dim) views over [3, b, seq, heads, head_dim]
        # storage.  cuDNN's sm100 flash kernel takes these strided views as-is
        # *and* returns its output in the same layout, so the reshape feeding
        # proj is a free view instead of a 7MB clone.  The workspace is reused
        # across calls: nothing in it outlives the forward that fills it.
        p.buf = torch.empty_strided(
            (3, bsz, h, seq, hd), (rows * dim, seq * dim, hd, dim, 1),
            dtype=x.dtype, device=x.device,
        )
        p.q, p.k, p.v = p.buf.unbind(0)

        bs = self.block
        args = (seq, dim, hd, self.hdp, rows * dim, bs,
                (seq % bs == 0) and (hd == self.hdp))
        grid = (h, triton.cdiv(seq, bs), bsz)
        probe_in = torch.randn(rows, 3 * dim, dtype=x.dtype, device=x.device)
        p.run, p.run_args = self._bind_launcher(
            p, grid, args, probe_in, torch.empty_like(p.buf))
        self._bind_attention(p, bsz, seq, dim, h, hd, rows)
        _announce(p)
        return p

    def _bind_attention(self, p, bsz, seq, dim, h, hd, rows):
        """Bind the flash kernel iff cuDNN would be occupancy-starved here.

        cuDNN's query tile is 128 rows, so it runs ``b*h*ceil(seq/128)`` CTAs.
        Below a couple of waves the machine is mostly idle and the 64-row tile
        wins; above it cuDNN is the better kernel and is kept.  The test is the
        occupancy arithmetic, never the literal shape, so an unseen batch size
        lands on whichever side it belongs to.

        Whatever is chosen is checked against cuDNN's own answer before it is
        used, and any reason for not using it is recorded in ``p.attn_mode``:
        r1 lost an iteration to a broad ``except`` silently degrading a fast
        path, so nothing here fails quietly.
        """
        nsm = torch.cuda.get_device_properties(p.idx).multi_processor_count
        if hd != self.hdp:
            p.attn_mode = f"cudnn (head_dim {hd} is not a power of two)"
            return
        if bsz * h * -(-seq // 128) >= 2 * nsm:
            p.attn_mode = "cudnn (enough waves for cuDNN's 128-row tile)"
            return
        nm = -(-seq // _BM)
        kw = dict(SEQ=seq, DIM=dim, HD=hd, HDP=self.hdp, PART=rows * dim, NH=h,
                  QK_SCALE=hd ** -0.5 * 1.4426950408889634, BM=_BM, BN=_BN,
                  EVEN_M=(seq % _BM == 0), EVEN_N=(seq % _BN == 0), EVEN_D=True)
        grid = (nm, bsz * h)
        try:
            # p.buf still holds whatever the allocator handed over, so seed it
            # before comparing the two attentions on it.
            p.buf.normal_()
            out = torch.empty(p.flat_shape, dtype=p.dtype, device=p.device)
            _attn_fwd[grid](p.buf, out, **kw, num_warps=_NUM_WARPS,
                            num_stages=_NUM_STAGES)
            ref = F.scaled_dot_product_attention(p.q, p.k, p.v)
            ref = ref.transpose(1, 2).reshape(rows, dim)
            worst = (out.float() - ref.float()).abs().max().item()
            if not (worst < 4e-3):          # also rejects nan
                p.attn_mode = f"cudnn (flash kernel disagreed by {worst:.2e})"
                return
        except Exception as exc:  # noqa: BLE001 - fall back, but say why
            p.attn_mode = f"cudnn ({type(exc).__name__}: {exc})"
            return

        def flash(buf, o, _g=grid, _kw=kw):
            _attn_fwd[_g](buf, o, **_kw, num_warps=_NUM_WARPS, num_stages=_NUM_STAGES)

        p.attn = flash
        p.attn_mode = f"flash BM={_BM} BN={_BN} nw={_NUM_WARPS} ns={_NUM_STAGES}"

    def plan_modes(self):
        """Per-shape record of which rope launcher and which attention was used."""
        return {("fallback" if p.fallback else tuple(p.out_shape)):
                ("reference" if p.fallback else f"{p.run.__name__}/{p.attn_mode}")
                for p in self._plans.values()}

    def _slow(self, x: torch.Tensor) -> torch.Tensor:
        """Cold path: (re)build the plan for this input, or fall back."""
        key = x.shape
        old = self._plans.get(key)
        if old is not None and old.fallback:
            return self._reference(x)
        if (self._fusable and x.is_cuda and x.dim() == 3
                and x.dtype in (torch.float16, torch.bfloat16)
                and x.shape[1] == self.seq_len and x.shape[2] == self.dim
                and self.proj.bias is not None):
            self._plans[key] = self._make_plan(x)
            return self.forward(x)
        self._plans[key] = _Plan(fallback=True)
        return self._reference(x)

    def _reference(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(bsz, self.frame_height, self.frame_width, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        k = k.reshape(bsz, self.frame_height, self.frame_width, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        v = v.reshape(bsz, self.frame_height, self.frame_width, self.num_heads, -1).permute(0, 3, 1, 2, 4)

        q = oasis_apply_rotary_emb(self.rotary_freqs, q)
        k = oasis_apply_rotary_emb(self.rotary_freqs, k)

        seq_len = self.frame_height * self.frame_width
        q = q.reshape(bsz, self.num_heads, seq_len, -1).transpose(1, 2)
        k = k.reshape(bsz, self.num_heads, seq_len, -1).transpose(1, 2)
        v = v.reshape(bsz, self.num_heads, seq_len, -1).transpose(1, 2)
        out = self.attn(q, k, v)
        out = out.reshape(bsz, seq_len, -1)
        return self.proj(out)

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self._plans.get(x.shape)
        if p is None or p.dtype is not x.dtype or p.device != x.device:
            return self._slow(x)

        rows, dim = p.rows, p.dim
        flat = x.reshape(rows, dim)
        qkv = torch.mm(flat, p.qkv_w) if p.qkv_b is None else torch.addmm(p.qkv_b, flat, p.qkv_w)
        p.run(qkv, p.buf, p.cos, p.sin, *p.run_args)
        if p.attn is None:
            out = F.scaled_dot_product_attention(p.q, p.k, p.v)
            if out.stride(1) == p.hd and out.stride(2) == dim:
                flat_out = out.as_strided(p.flat_shape, p.flat_stride)
            else:
                flat_out = out.transpose(1, 2).reshape(rows, dim)
        else:
            flat_out = torch.empty(p.flat_shape, dtype=p.dtype, device=p.device)
            p.attn(p.buf, flat_out)
        return torch.addmm(p.proj_b, flat_out, p.proj_w).view(p.out_shape)
