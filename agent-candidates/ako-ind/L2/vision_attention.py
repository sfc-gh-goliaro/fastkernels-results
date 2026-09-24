"""Encoder-only attention for Qwen vision transformer blocks.

Non-causal, no KV cache. Uses FlashAttnPrefill L1 op with cu_seqlens
for variable-length sequence support within the vision encoder.

Two things dominate a call at these shapes, and neither is the math:

* **Host time.**  A forward enqueues only four kernels (qkv GEMM, RoPE,
  FlashAttention, proj GEMM) but spends ~290 us of *CPU* time doing it -- 145 us
  of that inside ``flash_attn``'s ``apply_rotary`` and ~76 us inside FA4's
  ~900-line CuTeDSL Python interface.  The benchmark enqueues its iterations
  back-to-back with no in-loop sync, so that CPU time is not hidden: the GPU
  drains and then idles *inside* the measured event window.  Measured on B200
  (host = wall time around a forward with an empty queue, GPU = summed kernel
  time)::

      tokens   event window   GPU kernels   host enqueue
        1760        305 us         49 us        289 us
       20680        458 us        421 us        286 us
       23760        529 us        485 us        295 us
       24992        557 us        517 us        286 us
       64680       1886 us       1670 us        320 us

  So the smallest shape is 6x host-bound and even the mid shapes stall.  The fix
  is to CUDA-graph the expensive-to-enqueue middle of the forward (RoPE +
  attention) and replay it: one graph launch instead of two Python wrappers.
  Both GEMMs stay *outside* the graph on purpose -- see :class:`_Plan`.

* **RoPE.**  ``apply_rotary``'s generic kernel is ~2.6x off its 190 MB
  round-trip roofline here (67 us vs 27 us at 20680 tokens) because
  ``rotary_dim=72`` makes it index ``arange(0, 64)`` masked to ``< 36`` -- 44% of
  its lanes idle -- with ``BLOCK_M=8``/``BLOCK_H=2``, i.e. ~41k tiny blocks.
  :func:`_qk_rope_kernel` below replaces it.

Together those take the host to ~55 us per call and leave the four large shapes
exactly GPU-bound (event window = device time + the harness's own input-shifting
memcpys, no measurable launch bubble)::

      tokens   before   after   of which FA4 (frozen)
        1760    305 us   75 us     21 us
       20680    458 us  441 us    215 us
       23760    529 us  501 us    246 us
       24992    557 us  514 us    259 us
       64680   1886 us 1757 us   1078 us

Output is bit-identical to the baseline on every benched shape.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
from flash_attn.ops.triton.rotary import apply_rotary

from ....infra.tp import _tp_size, _tp_rank
from ..L1.flash_attn_prefill import FlashAttnPrefill
from .parallel_linear import QKVParallelLinear, RowParallelLinear


# ---------------------------------------------------------------------------
# RoPE over the q|k half of a fused qkv row.
# ---------------------------------------------------------------------------
@triton.jit
def _qk_rope_kernel(QKV, COS, SIN, n_tok, stride_tok,
                    HALF: tl.constexpr, HDIM: tl.constexpr,
                    BLOCK_T: tl.constexpr, BLOCK_P: tl.constexpr):
    """Rotate q and k in place inside the fused qkv buffer.

    ``q`` and ``k`` are adjacent within each token's row, so together they form
    one contiguous run of ``2 * num_heads * head_dim`` elements (4608 B per
    token here).  One program therefore takes ``BLOCK_T`` tokens across *all*
    2*num_heads q/k heads and shares each token's ``HALF`` cos/sin values over
    every one of them.

    The run is indexed by *rotation pair* ``p = head * HALF + dm`` rather than by
    ``(head, dim)``.  ``head_dim`` is 72, so a ``(head, dim)`` block needs
    ``arange(0, next_pow2(72)) < 36`` and idles 44% of its lanes -- which is what
    ``flash_attn``'s kernel does.  Pair indexing needs only ``BLOCK_P |
    num_heads * head_dim`` (128 | 1152), so every lane is active and no
    dimension carries a mask.  Each head's half still starts 16 B-aligned
    (``head * 72`` elements = ``head * 144`` B), so the loads stay vectorized.
    """
    pid_t = tl.program_id(0)
    pid_p = tl.program_id(1)
    t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    m = t < n_tok
    p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    h = p // HALF
    dm = p - h * HALF
    lo = QKV + t[:, None] * stride_tok + (h * HDIM + dm)[None, :]
    hi = lo + HALF
    x0 = tl.load(lo, mask=m[:, None]).to(tl.float32)
    x1 = tl.load(hi, mask=m[:, None]).to(tl.float32)
    coff = t[:, None] * HALF + dm[None, :]
    c = tl.load(COS + coff, mask=m[:, None]).to(tl.float32)
    s = tl.load(SIN + coff, mask=m[:, None]).to(tl.float32)
    # Same fp32-accumulate / bf16-store order as ``flash_attn``'s kernel, so the
    # result is bit-identical to the baseline's.
    tl.store(lo, x0 * c - x1 * s, mask=m[:, None])
    tl.store(hi, x0 * s + x1 * c, mask=m[:, None])


_ROPE_BLOCK_T = 16
_ROPE_NUM_WARPS = 4
_ROPE_NUM_STAGES = 2


@triton.jit
def _stage_kernel(COS_SRC, SIN_SRC, CU_SRC, COS_DST, SIN_DST, CU_DST,
                  n_elem, n_cu, BLOCK: tl.constexpr, BLOCK_CU: tl.constexpr):
    """Copy the graph's three small inputs into its static buffers in one launch.

    cos/sin/cu_seqlens arrive at a fresh address on every call, so a captured
    graph cannot read them in place.  Three ``copy_`` calls become three
    ``Memcpy DtoD`` ops, and a DtoD memcpy costs ~4 us of device time here almost
    regardless of size -- ~12 us for 3 MB of actual data.  One kernel that takes
    all three source pointers as launch arguments pays that cost once.
    """
    pid = tl.program_id(0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    m = o < n_elem
    tl.store(COS_DST + o, tl.load(COS_SRC + o, mask=m), mask=m)
    tl.store(SIN_DST + o, tl.load(SIN_SRC + o, mask=m), mask=m)
    if pid == 0:
        oc = tl.arange(0, BLOCK_CU)
        mc = oc < n_cu
        tl.store(CU_DST + oc, tl.load(CU_SRC + oc, mask=mc), mask=mc)


_STAGE_BLOCK = 4096


def _rope_block_p(pairs: int) -> int | None:
    """Largest power-of-two ``BLOCK_P`` that divides the pair count exactly."""
    for bp in (128, 64, 32, 16, 8):
        if pairs % bp == 0:
            return bp
    return None


class _Plan:
    """One shape's static buffers, captured graph, and pre-bound call sequence.

    The graph covers RoPE + attention and stops there.  That cut line is the
    whole design: with both GEMMs left outside, the qkv GEMM reads the caller's
    ``x`` wherever it lands and writes *into* the graph's static qkv buffer, and
    the proj GEMM allocates its own fresh output -- so the only things that have
    to be copied per call are cos/sin/cu_seqlens, 36 elements per token against
    qkv's 3456.  Pulling the GEMMs in as well was built and measured (static x
    in, result cloned out) and lost its A/B even on the one host-bound shape:
    54.5 us for this tier against 55.8 us for the all-in one at 1760 tokens.

    Everything the fast path can hoist is hoisted here: the transposed weight
    views, the 2-D view of the graph's attention output, the staging grid and the
    output shape.  The host budget at 1760 tokens was 61.7 us of which 18.5 us
    was Python around four tensor calls (module ``__call__``s, ``weight.t()``,
    ``.view()``, attribute chains); this class removes that.
    """

    __slots__ = ("qkv", "cu", "graph", "attn2d", "wq_t", "bq", "wp_t", "bp",
                 "dsts", "x_shape", "out_shape", "stage_grid", "n_elem",
                 "n_cu", "block_cu")

    def __init__(self, mod, qkv, cos_s, sin_s, cu_s, graph, attn_out,
                 x_shape, out_shape):
        self.qkv, self.cu, self.graph = qkv, cu_s, graph
        self.dsts = [cos_s, sin_s, cu_s]
        self.wq_t = mod.qkv.weight.t()
        self.bq = mod.qkv.bias
        self.wp_t = mod.proj.weight.t()
        self.bp = mod.proj.bias
        self.attn2d = attn_out.view(attn_out.shape[0], -1)
        self.x_shape = x_shape
        self.out_shape = out_shape
        self.n_elem = cos_s.numel()
        self.n_cu = cu_s.numel()
        self.block_cu = max(8, triton.next_power_of_2(self.n_cu))
        self.stage_grid = (triton.cdiv(self.n_elem, _STAGE_BLOCK),)

    def run(self, x, cu_seqlens, cos, sin):
        # The qkv GEMM stays *outside* the graph so it can read the caller's ``x``
        # at whatever address it arrives at -- pulling it in would mean copying x
        # into a static input buffer (149 MB at 64680 tokens).  It writes straight
        # into the graph's static qkv buffer, so nothing is copied on the way in.
        torch.addmm(self.bq, x.view(self.x_shape), self.wq_t, out=self.qkv)
        # cos/sin/cu_seqlens *are* graph inputs and do arrive at a fresh address
        # every call (the harness hands out a different ``data_ptr`` per
        # iteration), so they must be copied in.  Keying the graph on the pointer
        # and skipping this would time garbage while still passing the separate
        # eager correctness check.  They are small: 36 elements per token against
        # qkv's 3456.
        # One kernel, not ``torch._foreach_copy_``: that emits three
        # ``Memcpy DtoD`` ops and each costs ~4 us of device time here almost
        # regardless of size.  Measured both ways in one process -- the kernel
        # wins on all five benched shapes (e.g. 439 vs 454 us at 20680 tokens).
        _stage_kernel[self.stage_grid](
            cos, sin, cu_seqlens, *self.dsts, self.n_elem, self.n_cu,
            BLOCK=_STAGE_BLOCK, BLOCK_CU=self.block_cu, num_warps=4,
        )
        self.graph.replay()
        # proj also stays outside, so it allocates its own fresh output exactly
        # like the eager path -- no copy out of the graph's private buffer.
        return torch.addmm(self.bp, self.attn2d, self.wp_t).view(self.out_shape)


_MAX_PLANS = 6


class VisionAttention(nn.Module):
    """Multi-head attention for vision encoder (Qwen2-VL / Qwen2.5-VL / Qwen3-VL).

    All heads are attention heads (no GQA). Uses full (non-causal) attention.
    Supports TP: QKV is sharded, then gathered for RoPE, then re-sharded.
    """

    def __init__(self, embed_dim: int, num_heads: int, projection_size: int | None = None):
        super().__init__()
        if projection_size is None:
            projection_size = embed_dim
        tp = _tp_size()
        self.tp_size = tp
        self.tp_rank = _tp_rank()
        self.head_dim = projection_size // num_heads
        self.num_heads = num_heads // tp

        self.qkv = QKVParallelLinear(
            embed_dim, self.head_dim, num_heads, num_heads, bias=True,
        )
        self.proj = RowParallelLinear(projection_size, embed_dim, bias=True)
        self.attn = FlashAttnPrefill(self.num_heads, self.num_heads, self.head_dim)
        self._plans: dict[tuple, _Plan] = {}

    # -- RoPE ---------------------------------------------------------------
    def _rope_(self, qkv2d: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
        """In-place RoPE on q|k of a ``(tokens, 3 * num_heads * head_dim)`` buffer."""
        n_tok = qkv2d.shape[0]
        half = self.head_dim // 2
        pairs = self.num_heads * self.head_dim  # 2*num_heads heads x half pairs
        block_p = _rope_block_p(pairs)
        grid = (triton.cdiv(n_tok, _ROPE_BLOCK_T), pairs // block_p)
        _qk_rope_kernel[grid](
            qkv2d, cos, sin, n_tok, qkv2d.stride(0),
            HALF=half, HDIM=self.head_dim,
            BLOCK_T=_ROPE_BLOCK_T, BLOCK_P=block_p,
            num_warps=_ROPE_NUM_WARPS, num_stages=_ROPE_NUM_STAGES,
        )

    def _eligible(self, x, cu_seqlens, cos, sin, batch_size) -> bool:
        """Structural checks, run once per shape (not per call)."""
        return (
            batch_size == 1
            and cos is not None and sin is not None
            and x.is_cuda and x.is_contiguous()
            and cos.shape[0] == x.shape[0] and sin.shape == cos.shape
            and cos.is_contiguous() and sin.is_contiguous()
            and cu_seqlens.is_contiguous()
            and cos.shape[-1] * 2 == self.head_dim
            and cos.dtype == x.dtype and sin.dtype == x.dtype
            # ``_stage_kernel`` materialises ``next_pow2(n_cu)`` lanes in one
            # program, so keep the sequence count small; a huge cu_seqlens goes
            # down the eager path instead.
            and 2 <= cu_seqlens.numel() <= 1024
            and self.tp_size == 1
            and not self.qkv.use_fp8 and not self.proj.use_fp8
            and self.qkv.bias is not None and self.proj.bias is not None
            and _rope_block_p(self.num_heads * self.head_dim) is not None
        )

    # -- graphed fast path --------------------------------------------------
    def _attn_region(self, plan_qkv, cos, sin, cu, max_seqlen):
        """The half of the forward that is expensive to *enqueue*: RoPE + FA."""
        H, D = self.num_heads, self.head_dim
        qs = H * D
        T = plan_qkv.shape[0]
        self._rope_(plan_qkv, cos, sin)
        views = [plan_qkv[:, i * qs:(i + 1) * qs].view(T, H, D) for i in range(3)]
        return self.attn(
            *views, cu, cu, max_seqlen, max_seqlen,
            softmax_scale=D ** -0.5, causal=False, num_splits=1,
        )

    def _build_plan(self, x, cu_seqlens, cos, sin, max_seqlen) -> _Plan | None:
        seq_len, batch_size, in_features = x.shape
        T = seq_len
        qs = self.num_heads * self.head_dim
        dev, dt = x.device, x.dtype
        qkv = torch.empty(T, 3 * qs, device=dev, dtype=dt)
        cos_s = torch.empty(T, self.head_dim // 2, device=dev, dtype=dt)
        sin_s = torch.empty_like(cos_s)
        cu_s = torch.empty_like(cu_seqlens)
        cos_s.copy_(cos); sin_s.copy_(sin); cu_s.copy_(cu_seqlens)
        torch.addmm(self.qkv.bias, x.view(T, in_features),
                    self.qkv.weight.t(), out=qkv)

        # Warm up on a side stream: the Triton kernels and FA4's CuTeDSL kernel
        # must both be compiled, and any cuBLAS/CuTe workspace allocated, before
        # capture -- compiling or cudaMalloc-ing during capture is illegal.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._attn_region(qkv, cos_s, sin_s, cu_s, max_seqlen)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            attn_out = self._attn_region(qkv, cos_s, sin_s, cu_s, max_seqlen)
        if not isinstance(attn_out, torch.Tensor):
            return None
        plan = _Plan(self, qkv, cos_s, sin_s, cu_s, graph, attn_out,
                     (T, in_features), (seq_len, batch_size, -1))
        plan.run(x, cu_seqlens, cos, sin)  # compile the staging kernel now
        return plan

    def _plan_for(self, key, x, cu_seqlens, cos, sin, max_seqlen) -> _Plan | None:
        plans = self._plans
        if len(plans) >= _MAX_PLANS:
            # Bound the cache: each plan pins a qkv buffer (3*1152*2 B/token,
            # 447 MB at 64680 tokens) plus its graph's private pool.  ``None``
            # entries (shapes that will never qualify) are bounded by the same
            # rule so a many-shape caller cannot grow the dict without limit.
            plans.pop(next(iter(plans)))
        if not self._eligible(x, cu_seqlens, cos, sin, x.shape[1]):
            plans[key] = None
            return None
        try:
            plan = self._build_plan(x, cu_seqlens, cos, sin, max_seqlen)
        except Exception:
            # Any capture problem (compile-during-capture, an allocation the
            # pool cannot serve, a stream-binding quirk in a future CuTe build)
            # must degrade to the eager path, not fail the layer.
            plan = None
        plans[key] = plan
        return plan

    # -- forward ------------------------------------------------------------
    def forward(
        self, x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        # Fast path first: one dict lookup plus a handful of C-level guards, and
        # nothing else -- every structural check lives in ``_eligible``, which runs
        # once per shape.  The key pins every property the captured graph baked in
        # (token count, batch -- the fast path is batch 1 only -- sequence count,
        # the host-side max_seqlen, dtypes and device) plus the cos/sin shapes, so
        # a broadcastable-but-wrong cos can never slip through.  A missing key
        # means "not tried yet" (``False``); a stored ``None`` means "tried, this
        # shape will never qualify" and goes straight to the eager path.
        if (rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None
                and not torch.is_grad_enabled()
                and not torch.cuda.is_current_stream_capturing()):
            key = (x.shape, cu_seqlens.shape, rotary_pos_emb_cos.shape,
                   rotary_pos_emb_sin.shape, max_seqlen, x.dtype,
                   cu_seqlens.dtype, rotary_pos_emb_cos.dtype, x.device)
            plan = self._plans.get(key, False)
            # The staged copies and the RoPE kernel address cos/sin/cu_seqlens
            # flat, and the qkv GEMM needs a flattenable x, so contiguity is
            # checked per call, not per shape: two callers can share a shape key
            # and disagree on layout, and reading a strided cos as if it were
            # packed would be silently wrong rather than slow.
            if (plan is not None and x.is_contiguous()
                    and cu_seqlens.is_contiguous()
                    and rotary_pos_emb_cos.is_contiguous()
                    and rotary_pos_emb_sin.is_contiguous()):
                if plan is not False:
                    return plan.run(x, cu_seqlens, rotary_pos_emb_cos,
                                    rotary_pos_emb_sin)
                plan = self._plan_for(
                    key, x, cu_seqlens, rotary_pos_emb_cos, rotary_pos_emb_sin,
                    self._resolve_max_seqlen(x, cu_seqlens, max_seqlen))
                if plan is not None:
                    return plan.run(x, cu_seqlens, rotary_pos_emb_cos,
                                    rotary_pos_emb_sin)

        seq_len, batch_size, _ = x.shape
        return self._forward_eager(
            x, cu_seqlens, rotary_pos_emb_cos, rotary_pos_emb_sin,
            self._resolve_max_seqlen(x, cu_seqlens, max_seqlen),
            seq_len, batch_size)

    def _resolve_max_seqlen(self, x, cu_seqlens, max_seqlen):
        if max_seqlen is not None:
            return max_seqlen
        # No captured call omits ``max_seqlen``, but when one does, avoid the
        # ``(cu[1:] - cu[:-1]).max().item()`` device->host sync.  FA4 never
        # passes ``max_seqlen_q/k`` to the kernel: the varlen grid comes from
        # ``ceil_div(total_q, tile)`` and per-sequence bounds from ``cu_seqlens``
        # on the device.  Host side the value only reaches
        # ``num_splits_heuristic`` (dead, ``num_splits=1`` is pinned),
        # ``seqlen_k_loaded`` (local attention only), the 2-CTA path (needs
        # ``cu_seqlens_q is None``) and ``q_stage = 2 if max_seqlen_q > tile_m
        # (128) else 1`` -- and the true max exceeds 128 whenever the token total
        # does, so the token total selects the identical kernel and grid.  It is
        # also FA4's own default here.  Measured: every ``max_seqlen > 128``
        # gives the same FA4 time to within noise, and forcing ``q_stage = 1``
        # with ``max_seqlen <= 128`` is 22-40% *slower*, so 2 is the right stage
        # count and the exact value is irrelevant.  FA2/FA3 *do* size their grid
        # from ``max_seqlen_q``, so they keep the exact value.
        if getattr(self.attn, "fa_version", None) == 4:
            return x.shape[0] * x.shape[1]
        return (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

    def _forward_eager(self, x, cu_seqlens, cos, sin, max_seqlen,
                       seq_len, batch_size):
        """Ungraphed path: any shape, any batch size, any dtype."""
        num_heads, head_dim = self.num_heads, self.head_dim
        qs = num_heads * head_dim
        qkv = self.qkv(x)

        # Zero-copy: q, k and v are strided *views* into the fused qkv GEMM
        # output and RoPE runs in place through them, so this path materialises
        # no tensor of its own either.  Only the last dim has to be contiguous
        # for FlashAttention (FA4 builds its CuTe tensors with
        # ``mark_layout_dynamic(leading_dim=-1)``; every other stride is a
        # runtime argument), and ``v`` was always passed as exactly this kind of
        # stride-3456 view.  This replaces ``qk.permute(2,1,0,3,4).contiguous()``,
        # which materialised ~95 MB per call at encoder shapes.
        if cos is not None and sin is not None:
            if (batch_size == 1 and qkv.is_contiguous()
                    and cos.shape[-1] * 2 == head_dim
                    and cos.shape[0] >= seq_len and sin.shape == cos.shape
                    and cos.is_contiguous() and sin.is_contiguous()
                    and cos.dtype == qkv.dtype and sin.dtype == qkv.dtype
                    and _rope_block_p(num_heads * head_dim) is not None):
                self._rope_(qkv.view(seq_len, 3 * qs), cos, sin)
            else:
                qk = qkv[..., : 2 * qs].view(
                    seq_len, batch_size, 2, num_heads, head_dim)
                # batch > 1 cannot fold (q|k, batch) into one rotary batch axis
                # -- the two dims have unrelated strides -- so rotate q and k
                # with one launch each.  Same math, same cos/sin rows.
                for i in (0, 1):
                    apply_rotary(qk[:, :, i].transpose(0, 1), cos, sin,
                                 inplace=True)

        def _flash_view(start: int) -> torch.Tensor:
            # (batch, seq) order matches q/k/v to each other and to cu_seqlens.
            # A view when batch_size == 1; a copy only for batch > 1.
            return (qkv[..., start: start + qs]
                    .view(seq_len, batch_size, num_heads, head_dim)
                    .transpose(0, 1)
                    .reshape(-1, num_heads, head_dim))

        out = self.attn(
            _flash_view(0), _flash_view(qs), _flash_view(2 * qs),
            cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
            softmax_scale=head_dim ** -0.5,
            causal=False,
            # Disable split-KV.  With ``num_splits=0`` (auto) FA4's CuTeDSL
            # kernel picks ``num_splits > 1`` for the few m-blocks an encoder
            # produces at moderate seqlens, enabling the ``is_split_kv`` path in
            # ``flash_fwd_sm100.py`` whose ``n_block_first`` is ``None`` on one
            # branch and ``Int32`` on another -- a TYPE_UNSTABLE_JOIN CuTe
            # compile error on Blackwell.  Encoder self-attention is balanced
            # (q_len == k_len) so split-KV never helps here anyway.
            num_splits=1,
        )
        return self.proj(out.view(seq_len, batch_size, -1))
