"""Oasis spatial axial attention.

Measured facts about this benchmark, which decide what is worth optimizing
(all at ``[1,4,9,16,1024]`` on B200; see ITERATIONS.md for the runs):

``_time_module`` zeroes a ``2 * L2_cache_size`` = 253 MiB flush buffer before
recording the start event.  That zero is **72.8us of device time** and only
5.1us of host time, so it hands the host a 72.8us head start: any candidate
whose host enqueue is under ~72us has its whole forward already queued when the
start event fires, and the measured span is **pure device time**.  The baseline
measures ~390us because *its* host cost (394us) exceeds that head start; r1's
kernel, at 54us of host, is already device-bound.

So the levers are device kernel *count* and device kernel *duration*, and CUDA
graphs -- which change neither -- are a measured wash: a graph does remove
~1.3us of inter-kernel gap per kernel, but ``_ShiftingPool`` gives ``x`` a new
``data_ptr`` every iteration, so a forward must be ``static_in.copy_(x);
g.replay()``, and that copy plus the eager->graph submission boundary costs
~5.3us -- more than the gaps it buys back.  Six arrangements were benchmarked;
all lost to eager, including one over the final three-kernel forward.

What this round changes: r1's four device kernels (QKV GEMM, rotary as one
complex multiply, cuDNN SDPA, output GEMM) become **three**, by fusing the
rotary into a hand-written flash-attention kernel:

1. ``torch.mm`` -> the packed ``[bt, n, 3, heads, dim_head]`` QKV slab.
2. ``_rope_attn_kernel`` reads that slab in place, applies the interleaved
   adjacent-pair rotation to q and k *on load* against cached full-width
   ``[n, dim_head]`` cos/sin tables, runs the attention, and writes
   ``[bt, n, heads, dim_head]`` -- already the contiguous operand the output
   projection wants, so the tail transpose/reshape r1 needed is gone too.
3. ``torch.addmm`` -> the output projection.

What makes the fusion cheap is that the rotation costs almost nothing *if every
tile keeps a stride-1 axis*.  Because the axial freqs are
``repeat_interleave(2)``, adjacent feature pairs share an angle, so the rotation
is ``t * cf + flip_pairs(t) * sf`` with ``rotate_half``'s sign folded into the
sin table -- a whole-tile elementwise op plus a register-level flip of a size-2
axis.  Reading the even and odd halves as two stride-2 loads instead leaves the
tile with no stride-1 axis at all, nothing vectorizes, and the same kernel costs
12.2us instead of 7.0us.  For reference, the attention alone (rotation deleted)
profiles at 6.45us, and r1's separate ``mul`` + cuDNN SDPA cost 6.37 + 7.51 =
13.9us -- so this kernel is both cheaper than the pair it replaces and one fewer
launch.

Numerics: the tables are built at ``promote_types(freqs.dtype, x.dtype)``.  The
harness casts ``rotary_emb.freqs`` to fp16, and recomputing cos/sin in fp32 is
*more accurate* and **wrong** -- it is a different table and the outputs
diverge.  ``max_abs_error`` is 1.22e-04 against atol/rtol 1e-2, bit-identical to
r1's.

Anything the fast path cannot prove correct -- rot_dim != dim_head, freqs that
are not pair-interleaved, a dtype Triton should not own here (fp32, where
``tl.dot`` would silently use tf32), an unusually large spatial extent, or a
Triton launch that reports OutOfResources -- falls back to r1's complex-multiply
path, and that in turn falls back to the reference implementation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.dense_attention import DenseAttention
from ..L1.linear import Linear
from ..L1.oasis_rotary import OasisRotaryEmbedding, oasis_apply_rotary_emb

try:
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001 - no Triton: the complex-multiply path still applies
    triton = None

# Tile shape and warp count, hardcoded rather than autotuned.  Picked from a
# 54-config module-level sweep (ITERATIONS.md); the top six are within 0.8us of
# each other but two nearby configs spill catastrophically, so letting
# @triton.autotune re-select per run is a far bigger variance source than the
# choice is worth on a metric quantized to ~2us.
_BLOCK_M = 32
_BLOCK_N = 128
_NUM_WARPS = 4
_NUM_STAGES = 1
_MAX_N = 4096          # spatial extent the fast path will take responsibility for


if triton is not None:

    @triton.jit
    def _rope_attn_kernel(
        QKV, CF, SF, OUT, SCALE,
        N: tl.constexpr, H: tl.constexpr, HD: tl.constexpr, HD2: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_T: tl.constexpr,
    ):
        """Fused rotary + flash attention.  One program per (query block, b*head).

        Reads q, k and v straight out of the packed ``[bt, N, 3, H, HD]`` slab the
        QKV GEMM produced, rotates q and k on load, and writes ``[bt, N, H, HD]``
        -- already the contiguous operand the output projection consumes.

        Every tile keeps a stride-1 axis so the loads vectorize: q and v are
        ``[*, HD]`` with HD contiguous, k is ``[HD, BLOCK_N]`` with HD contiguous
        (the canonical flash-attention K orientation), and the cos/sin tables are
        full ``[N, HD]`` width so they can be addressed the same way as the tile
        they multiply.  The adjacent-pair swap the rotation needs is then a
        register-level ``flip`` of the size-2 axis, not a second strided load:
        with two stride-2 loads instead (no stride-1 axis at all, so no
        vectorization) the same kernel measures 12.2us against 7.3us here.

        The KV sweep is split into unmasked full ``BLOCK_N`` blocks plus one
        narrow ``BLOCK_T``-wide masked tail.  For the captured N=144 with
        BLOCK_N=128 that is 128 + 16, covering every column with no masked MMA
        work at all and no predication on the full block's loads; a single
        ``BLOCK_N=256`` pass instead wastes 112 of 256 columns and costs 7.3us
        against **6.7us** here.
        """
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H
        row = 3 * H * HD          # slab stride between positions
        orow = H * HD             # output stride between positions
        DT = QKV.dtype.element_ty

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HD)
        mm = offs_m < N

        base = QKV + b * (N * row) + h * HD
        qp = base + offs_m[:, None] * row + offs_d[None, :]
        q = tl.load(qp, mask=mm[:, None], other=0.0).to(tl.float32)
        qsw = tl.reshape(tl.flip(tl.reshape(q, [BLOCK_M, HD2, 2]), 2), [BLOCK_M, HD])
        cq = tl.load(CF + offs_m[:, None] * HD + offs_d[None, :],
                     mask=mm[:, None], other=0.0).to(tl.float32)
        sq = tl.load(SF + offs_m[:, None] * HD + offs_d[None, :],
                     mask=mm[:, None], other=0.0).to(tl.float32)
        qr = (q * cq + qsw * sq).to(DT)

        m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, HD], tl.float32)
        kbase = base + H * HD
        vbase = base + 2 * H * HD

        NFULL: tl.constexpr = (N // BLOCK_N) * BLOCK_N
        for start in tl.range(0, NFULL, BLOCK_N):
            offs_n = start + tl.arange(0, BLOCK_N)
            # v is issued before the score so its latency overlaps the dots
            # rather than serializing behind the softmax.
            v = tl.load(vbase + offs_n[:, None] * row + offs_d[None, :])
            # k is loaded already transposed to [HD, BLOCK_N], head_dim
            # contiguous.  The position index being the fast axis means lanes
            # stride by `row`, but every sector fetched is one the v load needs
            # anyway and it comes out of L2 -- measured 2x faster than a
            # coalesced [BLOCK_N, HD] load plus tl.trans, which materializes
            # through shared memory.
            k = tl.load(kbase + offs_n[None, :] * row + offs_d[:, None])
            ksw = tl.reshape(tl.flip(tl.reshape(k, [HD2, 2, BLOCK_N]), 1), [HD, BLOCK_N])
            ck = tl.load(CF + offs_n[None, :] * HD + offs_d[:, None])
            sk = tl.load(SF + offs_n[None, :] * HD + offs_d[:, None])
            kr = (k.to(tl.float32) * ck.to(tl.float32)
                  + ksw.to(tl.float32) * sk.to(tl.float32)).to(DT)
            s = tl.dot(qr, kr) * SCALE
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = tl.dot(p.to(DT), v, acc=acc * alpha[:, None])
            m_i = m_new
        if NFULL < N:
            offs_n = NFULL + tl.arange(0, BLOCK_T)
            mn = offs_n < N
            v = tl.load(vbase + offs_n[:, None] * row + offs_d[None, :],
                        mask=mn[:, None], other=0.0)
            k = tl.load(kbase + offs_n[None, :] * row + offs_d[:, None],
                        mask=mn[None, :], other=0.0)
            ksw = tl.reshape(tl.flip(tl.reshape(k, [HD2, 2, BLOCK_T]), 1), [HD, BLOCK_T])
            ck = tl.load(CF + offs_n[None, :] * HD + offs_d[:, None],
                         mask=mn[None, :], other=0.0)
            sk = tl.load(SF + offs_n[None, :] * HD + offs_d[:, None],
                         mask=mn[None, :], other=0.0)
            kr = (k.to(tl.float32) * ck.to(tl.float32)
                  + ksw.to(tl.float32) * sk.to(tl.float32)).to(DT)
            s = tl.dot(qr, kr) * SCALE
            s = tl.where(mn[None, :], s, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = tl.dot(p.to(DT), v, acc=acc * alpha[:, None])
            m_i = m_new
        acc = acc / l_i[:, None]
        tl.store(OUT + b * (N * orow) + offs_m[:, None] * orow + h * HD + offs_d[None, :],
                 acc.to(DT), mask=mm[:, None])


def _tables(freqs: torch.Tensor, dtype: torch.dtype, n: int, dim_head: int):
    """Flat ``[n, dim_head // 2]`` cos/sin for the interleaved-pair rotation, or
    None if *freqs* is not the full-head ``repeat_interleave(2)`` layout the fast
    path assumes.

    Computed at the dtype the reference path promotes to, so the cached tables
    hold exactly the values ``freqs.cos()`` / ``freqs.sin()`` produce there -- at
    fp16 the table is coarse enough (top frequency 402 rad, fp16 spacing 0.25
    there) that recomputing it in fp32 would be a different table.
    """
    if freqs.shape[-1] != dim_head or dim_head % 2:
        return None
    f = freqs.to(torch.promote_types(freqs.dtype, dtype)).reshape(n, dim_head)
    even, odd = f[:, 0::2], f[:, 1::2]
    cos, sin = even.cos(), even.sin()
    if not (torch.equal(cos, odd.cos()) and torch.equal(sin, odd.sin())):
        return None
    return cos.contiguous(), sin.contiguous()


class OasisSpatialAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: OasisRotaryEmbedding,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.to_qkv = Linear(dim, dim_head * heads * 3, bias=False)
        self.to_out = Linear(dim_head * heads, dim, bias=True)
        self.rotary_emb = rotary_emb
        self.attn = DenseAttention(backend="sdpa")
        self._plan = None

    # -- plan -------------------------------------------------------------
    def _plan_key(self, x: torch.Tensor):
        freqs = self.rotary_emb.freqs
        return (x.shape, x.dtype, x.device, freqs, freqs.data_ptr())

    def _build_triton(self, x: torch.Tensor, freqs: torch.Tensor):
        bsz, time, height, width, dim = x.shape
        heads, dim_head = self.heads, self.dim_head
        bt, n = bsz * time, height * width
        if (triton is None or x.dtype not in (torch.float16, torch.bfloat16)
                or not 0 < n <= _MAX_N or dim_head % 2 or x.device.type != "cuda"):
            return None
        tb = _tables(freqs, x.dtype, n, dim_head)
        if tb is None:
            return None
        cos, sin = tb
        # Full dim_head width, with rotate_half's sign folded into sin, so the
        # rotation is `t * cf + flip_pairs(t) * sf` on whole tiles.  Negating an
        # fp16 sin is exact, so these hold exactly the reference's table values.
        cf = cos.repeat_interleave(2, dim=-1).contiguous()
        sgn = torch.where(torch.arange(dim_head, device=x.device) % 2 == 0,
                          -1.0, 1.0).to(sin.dtype)
        sf = (sin.repeat_interleave(2, dim=-1) * sgn).contiguous()
        npow = triton.next_power_of_2(n)
        bm = max(16, min(_BLOCK_M, npow))
        bn = max(16, min(_BLOCK_N, npow))
        rem = n % bn
        bt_tile = max(16, triton.next_power_of_2(rem)) if rem else 16
        qkv = torch.empty(bt * n, 3 * heads * dim_head, device=x.device, dtype=x.dtype)
        att = torch.empty(bt * n, heads * dim_head, device=x.device, dtype=x.dtype)
        args = (cf, sf, att, 1.0 / math.sqrt(dim_head))
        kwargs = dict(N=n, H=heads, HD=dim_head, HD2=dim_head // 2,
                      BLOCK_M=bm, BLOCK_N=bn, BLOCK_T=bt_tile,
                      num_warps=_NUM_WARPS, num_stages=_NUM_STAGES)
        grid = (triton.cdiv(n, bm), bt * heads)
        # Compile and launch once here rather than discovering an
        # OutOfResources (tensor memory) or a codegen failure inside a timed
        # forward; the bench's warmup makes this free.
        _rope_attn_kernel[grid](qkv, *args, **kwargs)
        torch.cuda.synchronize()
        return (1, (bt * n, dim), qkv, grid, args, kwargs,
                (bsz, time, height, width, heads * dim_head))

    def _build_complex(self, x: torch.Tensor, freqs: torch.Tensor):
        """r1's single complex multiply over the packed slab, as the fallback."""
        bsz, time, height, width, dim = x.shape
        heads, dim_head = self.heads, self.dim_head
        bt, n = bsz * time, height * width
        tb = _tables(freqs, x.dtype, n, dim_head)
        if tb is None:
            return None
        cos, sin = tb
        rotor = torch.view_as_complex(torch.stack(tb, dim=-1).contiguous())
        rotor = rotor.reshape(1, n, 1, 1, dim_head // 2)
        qkv = torch.empty(bt * n, 3 * heads * dim_head, device=x.device, dtype=x.dtype)
        qkv5 = qkv.view(bt, n, 3, heads, dim_head)
        qk_in = torch.view_as_complex(
            qkv5[:, :, :2].view(bt, n, 2, heads, dim_head // 2, 2))
        rot = torch.empty(bt, n, 2, heads, dim_head // 2,
                          device=x.device, dtype=rotor.dtype)
        rot_real = torch.view_as_real(rot).view(bt, n, 2, heads, dim_head)
        if rot_real.dtype is not x.dtype:
            return None
        return (2, (bt * n, dim), qkv, qk_in, rotor, rot,
                rot_real[:, :, 0].transpose(1, 2),      # q [bt, heads, n, dim_head]
                rot_real[:, :, 1].transpose(1, 2),      # k
                qkv5[:, :, 2].transpose(1, 2),          # v, unrotated, in place
                (bt * n, heads * dim_head),
                (bsz, time, height, width, heads * dim_head))

    def _build_plan(self, x: torch.Tensor):
        key = self._plan_key(x)
        for build in (self._build_triton, self._build_complex):
            try:
                plan = build(x, self.rotary_emb.get_axial_freqs(
                    x.shape[2], x.shape[3]))
            except Exception:  # noqa: BLE001 - any unsupported layout drops a level
                plan = None
            if plan is not None:
                return key + plan
        return key + (0,)

    # -- forward ----------------------------------------------------------
    def _grad_active(self, x: torch.Tensor) -> bool:
        return bool(x.requires_grad or self.to_qkv.weight.requires_grad
                    or self.to_out.weight.requires_grad
                    or self.to_out.bias.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        plan = self._plan
        if plan is None or plan[:5] != self._plan_key(x):
            plan = self._plan = self._build_plan(x)
        mode = plan[5]
        # Both fast paths write the QKV GEMM into a preallocated slab with
        # ``out=``, which autograd rejects; the harness is entirely under
        # ``no_grad``, so this short-circuits on the first call and only the
        # grad-enabled case pays for the parameter checks.
        if torch.is_grad_enabled() and self._grad_active(x):
            return self._reference(x)
        if mode == 1:
            _, _, _, _, _, _, x_2d, qkv, grid, args, kwargs, out_shape = plan
            torch.mm(x.reshape(x_2d), self.to_qkv.weight.t(), out=qkv)
            _rope_attn_kernel[grid](qkv, *args, **kwargs)
            return torch.addmm(self.to_out.bias, args[2],
                               self.to_out.weight.t()).view(out_shape)
        if mode == 2:
            (_, _, _, _, _, _, x_2d, qkv, qk_in, rotor, rot,
             q, k, v, out_2d, out_shape) = plan
            torch.mm(x.reshape(x_2d), self.to_qkv.weight.t(), out=qkv)
            torch.mul(qk_in, rotor, out=rot)
            out = F.scaled_dot_product_attention(q, k, v)
            out = out.transpose(1, 2).reshape(out_2d)
            return torch.addmm(self.to_out.bias, out,
                               self.to_out.weight.t()).view(out_shape)
        return self._reference(x)

    # -- reference --------------------------------------------------------
    def _reference(self, x: torch.Tensor) -> torch.Tensor:
        bsz, time, height, width, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.reshape(bsz * time, height, width, self.heads, -1).permute(0, 3, 1, 2, 4)
        k = k.reshape(bsz * time, height, width, self.heads, -1).permute(0, 3, 1, 2, 4)
        v = v.reshape(bsz * time, height, width, self.heads, -1).permute(0, 3, 1, 2, 4)

        freqs = self.rotary_emb.get_axial_freqs(height, width)
        q = oasis_apply_rotary_emb(freqs, q)
        k = oasis_apply_rotary_emb(freqs, k)

        q = q.reshape(bsz * time, self.heads, height * width, -1).transpose(1, 2)
        k = k.reshape(bsz * time, self.heads, height * width, -1).transpose(1, 2)
        v = v.reshape(bsz * time, self.heads, height * width, -1).transpose(1, 2)
        out = self.attn(q, k, v, causal=False)
        out = out.reshape(bsz, time, height, width, self.heads, -1).reshape(
            bsz, time, height, width, -1)
        return self.to_out(out.to(q.dtype))
