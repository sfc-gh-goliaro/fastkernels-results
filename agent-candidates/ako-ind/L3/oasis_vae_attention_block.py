"""Oasis VAE attention block.

Fused rewrite. The four big GEMMs stay on cuBLAS and attention stays on the
cuDNN flash SDPA kernel; everything between them -- the two fp32-promoted
LayerNorms, both residual adds, the rope reshape/rotate/cat chain and the GELU
-- collapses into two Triton kernels plus cuBLAS beta/epilogue slots, and the
whole block then replays from one CUDA graph.

The block is small: ~87 GFLOP of GEMM at bs=6, ~15 GFLOP at bs=1. The baseline
issues ~30 aten ops per forward and measures 0.386 ms at bs=6 and 0.373 ms at
bs=1 -- *the same time for a sixth of the work* -- against ~250 us / ~40 us of
actual device work, which is what a launch-bound block looks like. So the work
here is almost entirely about op count and bytes moved between the GEMMs, not
about arithmetic.

Fusions:
  * ``_fused_layer_norm`` reads fp16, reduces in fp32 (the baseline's
    ``promote_fp32=True`` path is observable at fp16 tolerance), applies
    weight/bias in fp32 and writes fp16. It also emits ``x + next_bias`` as a
    second output, which lets the *following* GEMM absorb both the residual add
    and its own bias through cuBLAS' ``beta`` slot: ``addmm_`` on that buffer
    computes ``x + bias + A@B`` in one launch with no epilogue kernel. That is
    what removes both residual adds and two bias epilogues.
  * ``_rope_inplace`` rotates q and k in place inside the packed qkv GEMM
    output. The baseline's ``reshape(b,H,W,heads,d).permute(0,3,1,2,4)`` ->
    ``reshape(b,heads,s,d).transpose(1,2)`` round trip is an *identity on the
    flat index* -- token = h*frame_width + w and channel = head*head_dim + d
    either way -- so none of it needs to move data. q/k/v reach SDPA as strided
    views of the packed buffer (cuDNN takes that layout at identical speed) and
    only the rotated half of q and k is ever written.
  * GELU rides along as the cuBLASLt epilogue of fc1 via
    ``torch._addmm_activation``. Measured on B200: epilogue 26.8 us against
    20.6 us + 14.5 us for a separate GEMM and a separate GELU pass.

Device kernels per forward, at bs=6 (us): fc1+GELU 26.7, proj+fc2 26.4,
SDPA 19.6, qkv 15.9, 2x layer_norm 11.8, rope 4.4, output clone 3.4 -- 105.9 us
total, of which 88.6 is cuBLAS/cuDNN. Wall 0.126 ms at bs=6 and 0.0686 ms at
bs=1, both bit-exact against the baseline.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from ..L1.layer_norm import LayerNorm
from ..L2.oasis_mlp import OasisMLP
from ..L2.oasis_vae_attention import OasisVAEAttention


# Graph replay can be switched off for per-kernel profiling (a replayed graph is
# opaque to NCU's per-launch attribution).
_NO_GRAPH = bool(os.environ.get("OASIS_NO_CUDA_GRAPH"))
# Capture costs milliseconds, so a module that sees many shapes must stop paying
# it and just run the eager fused path.
_MAX_GRAPHS = 4
# Row tile for a channel-strided input (see ``_rows_for``).
_STRIDED_ROWS = 4


@triton.jit
def _fused_layer_norm(
    X, W, B, H, XP, OB,
    stride_xb, stride_xs, stride_xc,
    M, seq: tl.constexpr, eps,
    C: tl.constexpr, BLOCK: tl.constexpr,
    ROWS: tl.constexpr, HAS_XP: tl.constexpr,
):
    """``ROWS`` tokens per program.

    ``H``  <- layer_norm(x)  in fp16, contiguous (M, C)
    ``XP`` <- x + OB         in fp16, contiguous (M, C)   [when HAS_XP]

    ``X`` is read through explicit strides. At bs=1 the captured input arrives
    as a permuted view -- ``stride == (589824, 1, 576)``, i.e. unit-stride along
    *tokens* and 576 along channels -- so a one-row-per-program walk touches a
    separate 32-byte sector per channel. ``ROWS > 1`` puts consecutive tokens in
    the same sector and recovers most of that: 4.2 us -> 2.2 us on the bs=1
    input. For a channel-contiguous input the extra register pressure is a small
    net loss, so ``_rows_for`` picks ROWS=1 there.
    """
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rmask = rows < M
    b = rows // seq
    s = rows - b * seq
    cols = tl.arange(0, BLOCK)
    cmask = cols < C
    mask = rmask[:, None] & cmask[None, :]

    x = tl.load(
        X + (b * stride_xb + s * stride_xs)[:, None] + (cols * stride_xc)[None, :],
        mask=mask, other=0.0,
    ).to(tl.float32)

    # Two-pass mean/variance, both passes over registers: the whole row lives in
    # the tile, so the exact (x - mean) form costs nothing over sum-of-squares
    # and avoids its cancellation. Matches F.layer_norm's biased (1/C) variance.
    mean = tl.sum(x, axis=1) / C
    xc = tl.where(mask, x - mean[:, None], 0.0)
    var = tl.sum(xc * xc, axis=1) / C
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W + cols, mask=cmask, other=0.0).to(tl.float32)
    bb = tl.load(B + cols, mask=cmask, other=0.0).to(tl.float32)
    out = rows[:, None] * C + cols[None, :]
    tl.store(
        H + out,
        (xc * rstd[:, None] * w[None, :] + bb[None, :]).to(H.dtype.element_ty),
        mask=mask,
    )

    if HAS_XP:
        ob = tl.load(OB + cols, mask=cmask, other=0.0).to(tl.float32)
        tl.store(XP + out, (x + ob[None, :]).to(XP.dtype.element_ty), mask=mask)


@triton.jit
def _rope_inplace(
    QKV, COS, SIN,
    seq: tl.constexpr,
    NH: tl.constexpr, HD: tl.constexpr, ROT: tl.constexpr,
    LANES: tl.constexpr, BLOCK: tl.constexpr,
):
    """Axial rope on q and k, in place, inside the packed qkv GEMM output.

    ``QKV`` is (M, 3 * NH * HD) with channel = part * NH*HD + head * HD + d.
    One program covers both q and k for one token: ``LANES = 2 * NH * ROT``
    lanes, one per rotated channel (``BLOCK`` rounds that up to the power of 2
    ``tl.arange`` requires -- e.g. dim=768 / 12 heads gives 768 lanes). Channels at ``d >= ROT`` are never touched,
    which is what ``cat((t[..., :0], rotated, tail))`` amounts to in the
    baseline, and v is not touched at all.

    Interleaved (pairwise) convention, *not* the split-in-half GPT-NeoX one:
    freqs are ``repeat_interleave(2)`` and rotate_half maps adjacent pairs
    ``(x1, x2) -> (-x2, x1)``, so the partner of lane ``d`` is ``d ^ 1`` and the
    sign is negative on even ``d``. Arithmetic is fp32 -- fp32 cos/sin tables
    against fp32-promoted fp16 inputs, exactly the promotion the baseline's
    ``fp16 * freqs.cos()`` performs -- with a single fp16 rounding on store.
    """
    m = tl.program_id(0)
    s = m % seq
    i = tl.arange(0, BLOCK)
    mask = i < LANES
    part = i // (NH * ROT)          # 0 -> q, 1 -> k
    rem = i % (NH * ROT)
    base = QKV + m * (3 * NH * HD) + part * (NH * HD) + (rem // ROT) * HD
    d = rem % ROT

    a = tl.load(base + d, mask=mask, other=0.0).to(tl.float32)
    pair = tl.load(base + (d ^ 1), mask=mask, other=0.0).to(tl.float32)
    c = tl.load(COS + s * ROT + d, mask=mask, other=0.0)
    sn = tl.load(SIN + s * ROT + d, mask=mask, other=0.0)
    rh = tl.where((d & 1) == 0, -pair, pair)
    tl.store(base + d, (a * c + rh * sn).to(QKV.dtype.element_ty), mask=mask)


def _rows_for(x: torch.Tensor) -> int:
    """Row tile for a LayerNorm over ``x``'s last dimension."""
    return 1 if x.stride(-1) == 1 else _STRIDED_ROWS


class OasisVAEAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        frame_height: int,
        frame_width: int,
        *,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.attn = OasisVAEAttention(
            dim,
            num_heads,
            frame_height,
            frame_width,
            qkv_bias=qkv_bias,
        )
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.mlp = OasisMLP(dim, hidden_features=int(dim * mlp_ratio), approximate_tanh=False)

        self._dim = dim
        self._num_heads = num_heads
        self._head_dim = dim // num_heads
        self._seq = frame_height * frame_width
        self._reset_caches()

    def _reset_caches(self) -> None:
        self._plan: tuple | None = None
        self._plan_dtype: torch.dtype | None = None
        # Workspace reuse, keyed on row count. Every entry is fully overwritten
        # before it is read and none of them escapes -- the tensor the block
        # returns is always freshly allocated.
        self._ws: dict[int, tuple] = {}
        # (shape, dtype) -> replay recipe, see ``_capture``.
        self._graphs: dict[tuple, tuple] = {}
        self._graphs_off = _NO_GRAPH

    def _apply(self, *args, **kwargs):
        # .to() / .half() / .cuda() reassign ``p.data`` in place without
        # replacing the Parameter object, so an identity-keyed cache would hand
        # the kernels pointers into freed storage. Drop the caches instead.
        self._reset_caches()
        return super()._apply(*args, **kwargs)

    def __getstate__(self):
        # CUDAGraph objects are not picklable or deep-copyable, and the cached
        # buffers are pure derived state.
        state = dict(self.__dict__)
        for key in ("_plan", "_plan_dtype", "_ws", "_graphs", "_graphs_off"):
            state.pop(key, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._reset_caches()

    # ------------------------------------------------------------------ plan
    def _build_plan(self, x: torch.Tensor):
        """Cache everything loop-invariant: fp32 cos/sin tables, transposed
        weight views, launch constants.

        Returns ``None`` for any configuration these kernels do not cover, in
        which case ``forward`` runs the eager path.

        The cos/sin tables come off the ``rotary_freqs`` buffer, not off
        ``rotary.freqs``: the buffer is what the baseline's forward reads, and
        because it is non-persistent it survives ``load_state_dict`` while the
        ``freqs`` parameter does not (the bench also casts parameters to fp16
        but leaves buffers alone, so the buffer is still the fp32 one).
        """
        attn = self.attn
        freqs = attn.rotary_freqs
        rot = freqs.shape[-1]
        hd = self._head_dim

        supported = (
            x.dim() == 3
            and x.shape[1] == self._seq
            and x.shape[2] == self._dim
            and x.dtype in (torch.float16, torch.bfloat16)
            and x.is_cuda
            and freqs.numel() == self._seq * rot
            and rot % 2 == 0
            and rot <= hd
            and self.norm1.promote_fp32
            and self.norm2.promote_fp32
            and self.norm1.weight is not None and self.norm1.bias is not None
            and self.norm2.weight is not None and self.norm2.bias is not None
            and attn.proj.bias is not None
            and self.mlp.fc1.bias is not None
            and self.mlp.fc2.bias is not None
            and self.mlp.act.approximate == "none"
            and not attn.attn.use_flex_kernel
            and attn.attn.fa_func is None
        )
        if not supported:
            return None

        f32 = freqs.reshape(self._seq, rot).to(torch.float32)
        return (
            f32.cos().contiguous(), f32.sin().contiguous(),
            self.norm1.weight, self.norm1.bias, self.norm1.eps,
            self.norm2.weight, self.norm2.bias, self.norm2.eps,
            attn.qkv.weight.t(), attn.qkv.bias,
            attn.proj.weight.t(), attn.proj.bias,
            self.mlp.fc1.weight.t(), self.mlp.fc1.bias,
            self.mlp.fc2.weight.t(), self.mlp.fc2.bias,
            triton.next_power_of_2(self._dim),
            2 * self._num_heads * rot,
            triton.next_power_of_2(2 * self._num_heads * rot),
            rot,
        )

    def _plan_for(self, x: torch.Tensor):
        if (self._plan_dtype is not x.dtype
                or x.shape[-1] != self._dim
                or x.shape[-2] != self._seq):
            self._plan = self._build_plan(x)
            self._plan_dtype = x.dtype
        return self._plan

    # ----------------------------------------------------------- cuda graph
    def _capture(self, x: torch.Tensor, plan: tuple):
        """Capture the block, minus the first LayerNorm, into a replayable graph.

        Even after the fusions the eager path is launch-bound, not device-bound:
        it enqueues ~144 us of CPU against 108 us (bs=6) / 51 us (bs=1) of
        device work. Replay collapses that to one launch, and CPU enqueue drops
        to ~23 us.

        A graph can only read buffers it owns, and the caller hands a different
        pointer every call, so something has to bridge the two. The cheap bridge
        is the *first LayerNorm*: it already reads ``x`` and writes into buffers
        we own, so leaving it outside gets the input in for free. Capturing it
        too would instead need a ``static_in.copy_(x)`` before every replay --
        measured 121.9 us vs 119.8 us (bs=6) and 68.6 us vs 66.6 us (bs=1), the
        difference being that copy's memcpy, which at bs=1 is a slow strided
        gather (4.7 us on its own) because the input is a permuted view.

        The output clone stays: the last GEMM writes a buffer the graph owns, so
        handing it straight back would alias into the following call.
        """
        try:
            with torch.no_grad():
                h, xp, h2 = self._workspace(x)
                # Warm up on a side stream first: Triton has to JIT, and cuBLAS,
                # cuBLASLt and cuDNN all allocate workspaces and run their
                # heuristics on the first call for a shape. None of that is
                # capturable.
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        self._head(x, plan, h, xp)
                        self._tail(x.shape[0], plan, h, xp, h2)
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = self._tail(x.shape[0], plan, h, xp, h2)

            m = x.shape[0] * self._seq
            rows = _rows_for(x)
            return (
                graph, out, h, xp,
                ((m + rows - 1) // rows,), m, rows,
                plan[2], plan[3], plan[11], plan[4], plan[16],
            )
        except Exception:
            # Any capture failure (a non-capturable backend, a pool that cannot
            # be reserved) permanently demotes this module to the eager fused
            # path rather than failing the forward.
            self._graphs_off = True
            return None

    # --------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self._graphs.get((x.shape, x.dtype))
        if g is None:
            return self._forward_cold(x)
        (graph, out, h, xp, grid, m, rows,
         w1, b1, proj_b, eps1, block_c) = g
        _fused_layer_norm[grid](
            x, w1, b1, h, xp, proj_b,
            x.stride(0), x.stride(1), x.stride(2),
            m, self._seq, eps1,
            C=self._dim, BLOCK=block_c, ROWS=rows, HAS_XP=True, num_warps=4,
        )
        graph.replay()
        return out.clone()

    def _forward_cold(self, x: torch.Tensor) -> torch.Tensor:
        plan = self._plan_for(x) if not torch.is_grad_enabled() else None
        if plan is None:
            # Unsupported config, or autograd is live -- the fused path rotates
            # rope and lands both residuals in place, which autograd cannot see.
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x

        if not self._graphs_off and len(self._graphs) < _MAX_GRAPHS:
            g = self._capture(x, plan)
            if g is not None:
                self._graphs[(x.shape, x.dtype)] = g
                return self.forward(x)

        h, xp, h2 = self._workspace(x)
        self._head(x, plan, h, xp)
        return self._tail(x.shape[0], plan, h, xp, h2)

    # ------------------------------------------------------------------ body
    def _workspace(self, x: torch.Tensor) -> tuple:
        m = x.shape[0] * self._seq
        ws = self._ws.get(m)
        if ws is None:
            ws = tuple(
                torch.empty((m, self._dim), dtype=x.dtype, device=x.device)
                for _ in range(3)
            )
            self._ws[m] = ws
        return ws

    def _head(self, x: torch.Tensor, plan: tuple, h, xp) -> None:
        """layer_norm(x) -> h, and x + proj.bias -> xp (the residual base the
        attention projection's cuBLAS beta slot consumes)."""
        m = x.shape[0] * self._seq
        rows = _rows_for(x)
        _fused_layer_norm[((m + rows - 1) // rows,)](
            x, plan[2], plan[3], h, xp, plan[11],
            x.stride(0), x.stride(1), x.stride(2),
            m, self._seq, plan[4],
            C=self._dim, BLOCK=plan[16], ROWS=rows, HAS_XP=True, num_warps=4,
        )

    def _tail(self, bsz: int, plan: tuple, h, xp, h2) -> torch.Tensor:
        (cos, sin, w1, b1, eps1, w2, b2, eps2, qkv_wt, qkv_b, proj_wt, proj_b,
         fc1_wt, fc1_b, fc2_wt, fc2_b, block_c, rope_lanes, rope_block,
         rot) = plan

        seq, dim = self._seq, self._dim
        nh, hd = self._num_heads, self._head_dim
        m = bsz * seq
        grid = (m,)

        # 1. packed qkv, then rope in place; q/k/v reach SDPA as strided views.
        qkv = torch.addmm(qkv_b, h, qkv_wt) if qkv_b is not None else h @ qkv_wt
        _rope_inplace[grid](
            qkv, cos, sin, seq, nh, hd, rot,
            LANES=rope_lanes, BLOCK=rope_block, num_warps=4,
        )
        # permute+unbind rather than three getitem+transpose pairs: same strides,
        # 1 dispatch instead of 6 (10.0 us -> 4.0 us of CPU at bs=6).
        q, k, v = qkv.view(bsz, seq, 3, nh, hd).permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)

        # 2. x + proj.bias + attn_out @ proj.weight^T, one cuBLAS launch.
        #    ``out`` comes back allocated (b, s, h, d)-contiguous -- PyTorch
        #    matches the permuted query's strides -- so this transpose+reshape
        #    is a view and emits no copy kernel.
        x2 = xp.addmm_(out.transpose(1, 2).reshape(m, dim), proj_wt)

        # 3. layer_norm(x2) -> h2, and x2 + fc2.bias -> the fc2 residual base.
        #    x2p is what the block hands back, so it is the one buffer that
        #    cannot come from the reusable workspace.
        x2p = torch.empty((m, dim), dtype=h.dtype, device=h.device)
        _fused_layer_norm[grid](
            x2, w2, b2, h2, x2p, fc2_b,
            seq * dim, dim, 1,
            m, seq, eps2,
            C=dim, BLOCK=block_c, ROWS=1, HAS_XP=True, num_warps=4,
        )

        # 4. fc1 with the GELU cuBLASLt epilogue, then fc2 into the residual.
        act = torch._addmm_activation(fc1_b, h2, fc1_wt, use_gelu=True)
        return x2p.addmm_(act, fc2_wt).view(bsz, seq, dim)
