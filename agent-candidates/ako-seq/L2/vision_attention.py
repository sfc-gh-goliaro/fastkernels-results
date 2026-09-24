"""Encoder-only attention for Qwen vision transformer blocks.

Non-causal, no KV cache. Uses FlashAttnPrefill L1 op with cu_seqlens
for variable-length sequence support within the vision encoder.

The forward is written so that *nothing* between the fused QKV GEMM and the
FlashAttention call moves a byte it does not have to:

* ``qkv`` is one contiguous ``[seq, batch, 3*nh*hd]`` tensor, so q, k and v are
  already strided views of it with ``hd`` contiguous.  FA4's CuTeDSL path marks
  the q/k/v layouts dynamic except for a unit innermost stride
  (``to_cute_tensor`` -> ``mark_layout_dynamic``) and only forces a copy when
  ``stride(-1) != 1``, so those views go straight in.  No repack.
* Rotary runs *in place* on the q|k views, so the prologue makes exactly one
  read+write pass over q and k (the theoretical minimum) instead of two.

The previous version paid, per call at the captured 20-26k-token shapes:
``qkv[..., :2*q_size].permute(2, 1, 0, 3, 4).contiguous()`` (95 MB read + 95 MB
written, and because a 5-D permuted view drops TensorIterator into
``unrolled_elementwise<direct_copy>`` it ran at only ~1.6 TB/s -- 120 us), then
a second full pass over the same bytes inside ``apply_rotary`` (70 us). That was
190 us of a 530 us call, wrapped around an attention call of 216 us.

With the prologue at its floor, the attention call is 55-60% of every large
call, and FA4's stock heuristics have no tune for this geometry (bf16, 16 heads,
head_dim 72 -> ``head_dim_padded`` 80, non-causal, 24-32 ragged segments).  It
is therefore driven directly rather than through
``FlashAttnPrefill`` -> ``flash_attn_varlen_func`` -> ``interface._flash_attn_fwd``:
see ``_fa4_varlen_fwd`` below for what that turns on and why, and
``VisionAttention._attn`` for the gate.  Measured on the attention call alone
(CUDA events, L2 flushed, medians of interleaved repeats): 1.17x at seq=1760,
1.56-1.62x at 20-25k, 1.47x at 64680, bit-identical output on every case
tested.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ....infra.tp import _tp_size, _tp_rank
from ..L1.flash_attn_prefill import FlashAttnPrefill
from .parallel_linear import QKVParallelLinear, RowParallelLinear

# Lanes per rotary program. 2048 (= all 32 q|k heads of a tp=1 encoder x a
# 64-lane head-half) measured fastest across every benchmarked shape; see the
# tile note on ``_rope_launch_cfg``.
_ROPE_LANES = 2048


@triton.jit
def _rope_qk_inplace_kernel(
    QKV, COS, SIN,
    seq, nro,
    STRIDE_S: tl.constexpr,
    STRIDE_B: tl.constexpr,
    STRIDE_COS: tl.constexpr,
    RD_HALF: tl.constexpr,
    HD: tl.constexpr,
    H2: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Half-split rotary, in place, over the q|k region of a fused qkv tensor.

    q and k occupy the first ``2*nh*hd`` elements of each ``[seq, batch, ...]``
    row, so the pair (part, head) collapses into one index ``j = part*nh + head``
    and the region is simply a ``[seq, batch, 2*nh, hd]`` strided tensor -- the
    same address arithmetic covers both q and k in one launch, with each token's
    cos/sin loaded once and shared by all ``2*nh`` heads.

    Every shape is a ``constexpr`` (``HD`` especially: leaving the head stride a
    runtime argument costs 68.7 us vs 41.9 us at seq=20680, because Triton can
    no longer prove the innermost run is contiguous and falls back to narrow
    loads).
    """
    pid_t = tl.program_id(0)
    pid_j = tl.program_id(1)
    pid_b = tl.program_id(2)
    rt = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    rj = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
    rd = tl.arange(0, BLOCK_D)
    m_d = rd < RD_HALF

    # Positions past the end of the cos/sin table rotate by identity, matching
    # ``apply_rotary``'s ``other=1.0`` / ``other=0.0``.
    off_cs = rt[:, None] * STRIDE_COS + rd[None, :]
    m_cs = (rt < nro)[:, None] & m_d[None, :]
    cos = tl.load(COS + off_cs, mask=m_cs, other=1.0).to(tl.float32)[:, None, :]
    sin = tl.load(SIN + off_cs, mask=m_cs, other=0.0).to(tl.float32)[:, None, :]

    p = (QKV + pid_b * STRIDE_B + rt[:, None, None] * STRIDE_S
         + rj[None, :, None] * HD + rd[None, None, :])
    m = (rt < seq)[:, None, None] & m_d[None, None, :]
    if H2 % BLOCK_J != 0:
        m = m & (rj < H2)[None, :, None]
    x0 = tl.load(p, mask=m, other=0.0).to(tl.float32)
    x1 = tl.load(p + RD_HALF, mask=m, other=0.0).to(tl.float32)
    tl.store(p, x0 * cos - x1 * sin, mask=m)
    tl.store(p + RD_HALF, x0 * sin + x1 * cos, mask=m)


def _rope_launch_cfg(h2: int, rd_half: int) -> tuple[int, int, int, int]:
    """(BLOCK_T, BLOCK_J, BLOCK_D, num_warps) for a ``2*nh`` x ``rd_half`` tile.

    One token per program with as many heads as fit in ``_ROPE_LANES``: rotary
    here is pure streaming, so the only thing that matters is issuing the widest
    possible contiguous runs, and a whole token's q|k row is contiguous. Swept
    BLOCK_T in 1..32 x BLOCK_J in 1..32 x num_warps in 1..8 x num_stages in 1..2
    over all five benchmarked shapes; (1, 32, 64, 4) won at every one of them.
    """
    block_d = triton.next_power_of_2(rd_half)
    block_j = max(1, min(triton.next_power_of_2(h2), _ROPE_LANES // block_d))
    num_warps = min(8, max(1, block_j * block_d // 512))
    return 1, block_j, block_d, num_warps



# ===========================================================================
# Private FA4 (SM100 CuTeDSL) varlen forward.
#
# The baseline routes attention through ``FlashAttnPrefill`` -> vLLM's
# ``flash_attn_varlen_func`` -> ``cute.interface._flash_attn_fwd``, whose
# heuristics have no entry for this geometry (bf16, 16 heads, head_dim 72 ->
# ``head_dim_padded`` 80, non-causal, 24-32 ragged segments).  Three switches
# the stock path leaves off are worth 1.3-1.6x on the attention call here:
#
# * ``use_clc_scheduler`` -- cluster-launch-control work distribution.  Off by
#   default (``FA_CLC=0``) *and* force-disabled for varlen MHA in
#   ``interface.py`` ("CLC regressed for varlen MHA").  With ~830-token
#   segments the static varlen scheduler leaves 11-15 ragged waves of
#   single-tile CTAs on the table; CLC turns those into dynamically pulled
#   work.  Worth 1.17x (seq=1760) to 1.53x (20-25k) on its own, and it is the
#   whole win at the long-segment (``max_seqlen`` 4096) shapes.
# * ``use_2cta_instrs`` -- the 2-CTA ``tcgen05`` MMA.  Gated on
#   ``cu_seqlens_q is None and head_dim_padded in [128, 192]``, so a varlen
#   hd-80 call never gets it.  Alone it is a wash (1.01-1.05x): the cluster
#   halves the per-CTA KV smem but the epilogue then serialises.  Combined
#   with TMA-O it is another 1.06x over CLC on the short-segment shapes.
# * TMA-O for varlen -- ``use_tma_O = (...) and not is_varlen_q`` moves the O
#   store onto the correction warps.  ``candidate/L1/flash_attn_varlen.py``
#   (frozen L1 winner) already contains the fix, including the mandatory
#   ragged-last-tile fallback to the predicated copy, so it is imported rather
#   than re-derived.  Nothing in it is head-dim specific.
#
# Everything here is private: a subclass of FA4's forward kernel and a compile
# cache of our own.  No flash_attn / vllm / cutlass module global, class
# attribute or shared cache is mutated, at import time or per call -- the
# harness times the unmodified baseline in the same process.  Every gate reads
# host-side metadata only (shape / dtype / stride / numel / max_seqlen), never
# ``cu_seqlens`` contents, so no device sync is introduced, and any failure
# falls back to the baseline ``FlashAttnPrefill`` call.
# ===========================================================================
_FA4_ERROR: Exception | None = None
_FA4_TMAO_ERROR: Exception | None = None

try:
    import cutlass.cute as _cute
    from vllm.vllm_flash_attn.cute.cute_dsl_utils import (
        to_cute_tensor as _to_cute_tensor,
        torch2cute_dtype_map as _torch2cute,
    )
    from vllm.vllm_flash_attn.cute.flash_fwd_sm100 import (
        FlashAttentionForwardSm100 as _FwdSm100,
    )
    from vllm.vllm_flash_attn.cute.utils import AuxData as _AuxData

    from ....infra.fa_utils import FA_VERSION as _FA_VERSION

    # Built once: ``AuxData`` is a NamedTuple and this call site is on the
    # critical path of a ~70 us host-time forward.
    _AUX = _AuxData(None, None)

    class _ClcFwd(_FwdSm100):
        """Stock varlen forward with CLC work distribution.

        ``use_clc_scheduler`` is a plain ``__init__`` argument, so there is
        nothing to override; the subclass exists so that neither kernel we
        compile is ever the wheel's own class object, and so the two paths are
        symmetric.
        """

    try:
        # The TMA-O epilogue (and its ragged-tile fallback) from the frozen L1
        # winner.  ``_REGS`` there is ``(184, 72)``, tuned for causal MLA
        # prefill at 192/128; at hd 80 non-causal with the 2-CTA MMA the
        # surface is a cliff, not a slope (softmax 192 is 1.29x vs 1.61x at
        # 184), so it is re-tuned here.  Overriding the class attribute is
        # enough: ``_VarlenTmaOFwd.__init__`` reads ``self._REGS``.
        from ..L1.flash_attn_varlen import _VarlenTmaOFwd as _L1TmaOFwd

        if _L1TmaOFwd is None:
            raise ImportError("L1 _VarlenTmaOFwd unavailable on this wheel")

        class _Clc2CtaTmaOFwd(_L1TmaOFwd):
            _REGS = (184, 64)
    except Exception as _exc:  # pragma: no cover - wheel / L1 drift
        _FA4_TMAO_ERROR = _exc
        _Clc2CtaTmaOFwd = None

    _FA4_CACHE: dict = {}

    def _fa4_varlen_fwd(q, k, v, cu_q, cu_k, max_seqlen, softmax_scale, two_cta):
        """``interface._flash_attn_fwd``'s sm100 path, narrowed to what the gate
        in ``forward`` admits (fp16/bf16, MHA, non-causal, varlen, no paged KV /
        block sparsity / split-KV / fp8 / softcap / local / sink / LSE) and given
        its own compile cache.

        Skipping ``interface.py`` also skips its ~100 lines of per-call
        argument validation: 33.7 us -> 8.5 us of host time per call, of which
        4.7 us is the CuTe launch itself and 2.1 us the ``out`` allocation.
        That is what the ``seq=1760`` shape is actually bound by (~70 us of host
        time against ~69 us of GPU work).
        """
        total_q, num_head, head_dim = q.shape
        # ``q_stage`` mirrors interface.py: two 128-row stages per CTA unless the
        # longest segment fits in one.  ``max_seqlen`` is a host-side upper bound
        # (the captured callers pass a loose one), which is safe -- it only ever
        # picks the 2-stage pipeline.
        q_stage = 2 if max_seqlen > 128 else 1
        key = (_torch2cute[q.dtype], head_dim, q_stage, two_cta)
        compiled = _FA4_CACHE.get(key)
        if compiled is False:
            # Compilation (or the first launch) already failed for this key.
            # Without the memo a broken configuration would re-enter
            # ``cute.compile`` on every call before falling back -- ~1 s of host
            # time per call, i.e. far worse than a clean fallback.  Checked
            # before allocating, so the degraded path costs nothing.
            raise RuntimeError("fa4 varlen path previously failed")
        out = torch.empty(total_q, num_head, head_dim, dtype=q.dtype,
                          device=q.device)
        if compiled is None:
            _FA4_CACHE[key] = False
            cls = _Clc2CtaTmaOFwd if two_cta else _ClcFwd
            fa = cls(
                head_dim, head_dim,
                qhead_per_kvhead=1,
                is_causal=False,
                is_local=False,
                is_split_kv=False,
                pack_gqa=False,
                m_block_size=128,
                n_block_size=128,
                q_stage=q_stage,
                is_persistent=False,
                score_mod=None,
                mask_mod=None,
                has_aux_tensors=False,
                paged_kv_non_tma=False,
                is_varlen_q=True,
                q_subtile_factor=1,
                use_2cta_instrs=two_cta,
                use_clc_scheduler=True,
                output_quant_key=None,
            )
            compiled = _cute.compile(
                fa,
                *[_to_cute_tensor(t) for t in (q, k, v, out)],
                _to_cute_tensor(None, assumed_align=4),   # LSE: not requested
                softmax_scale,
                _to_cute_tensor(cu_q, assumed_align=4, leading_dim=0),
                _to_cute_tensor(cu_k, assumed_align=4, leading_dim=0),
                None, None,   # seqused_q, seqused_k
                None,         # dynamic_causal
                None,         # page_table
                None, None,   # window_size_left / right
                None,         # learnable_sink
                None,         # descale_tensors
                None,         # block sparse tensors
                _AUX,
                None,         # output_scale
                _cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )
            _FA4_CACHE[key] = compiled
        if q.requires_grad or k.requires_grad or v.requires_grad:
            # ``interface.py`` detaches unconditionally; three ``detach()``
            # calls are ~1.2 us of host time, and under the ``no_grad`` the
            # captured callers use there is nothing to detach.
            q, k, v = q.detach(), k.detach(), v.detach()
        try:
            compiled(
                q, k, v, out, None, softmax_scale, cu_q, cu_k,
                None, None, None, None, None, None, None, None, None,
                _AUX, None,
            )
        except Exception:
            _FA4_CACHE[key] = False
            raise
        return out

    # Same reason the baseline hides ``flash_attn_varlen_func`` from Dynamo:
    # FA4's CuTeDSL launcher rebuilds a Python closure per call, which Dynamo
    # guards on, so a traced caller recompiles on every call.
    _fa4_varlen_fwd = torch._dynamo.disable(_fa4_varlen_fwd)

except Exception as _exc:  # pragma: no cover - non-Blackwell / older wheel
    _FA4_ERROR = _exc
    _FA4_TMAO_ERROR = _exc
    _FA_VERSION = None
    _Clc2CtaTmaOFwd = None
    _fa4_varlen_fwd = None


_SM_COUNT: int | None = None


def _sm_count() -> int:
    global _SM_COUNT
    if _SM_COUNT is None:
        try:
            _SM_COUNT = torch.cuda.get_device_properties(
                torch.cuda.current_device()).multi_processor_count
        except Exception:
            _SM_COUNT = 132
    return _SM_COUNT


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

        self.softmax_scale = self.head_dim ** -0.5
        self.qkv_row = 3 * self.num_heads * self.head_dim
        # Shape-independent half of the FA4 gate, so a steady-state call only
        # checks the per-call metadata. ``head_dim`` is capped at 128 because
        # that is the envelope this path was measured and range-checked in;
        # above it FA4's own smem/tmem sizing takes different branches.
        self._fa4 = (_fa4_varlen_fwd is not None and _FA_VERSION == 4
                     and self.head_dim % 8 == 0 and 8 <= self.head_dim <= 128)
        self._fa4_2cta = self._fa4 and _Clc2CtaTmaOFwd is not None
        # (rd_half, cos row stride, batch) -> everything the rotary launch needs,
        # so a steady-state call is one dict lookup and one launch. Every
        # remaining Python statement in ``forward`` is on the critical path: the
        # whole call is only ~0.13 ms of host time, and at the smallest captured
        # shape (seq=1760) that is more than the GPU work it queues.
        self._rope_cfg: dict[tuple[int, int, int], tuple] = {}

    def _rope_cfg_for(self, rd_half: int, stride_cos: int, batch_size: int) -> tuple:
        assert 2 * rd_half <= self.head_dim, "rotary_dim must be <= head_dim"
        h2 = 2 * self.num_heads
        bt, bj, bd, nw = _rope_launch_cfg(h2, rd_half)
        # [0:4] grid + strides, [4:-1] the kernel's constexpr tail *in signature
        # order* so the launch can splat it, [-1] num_warps. BLOCK_T appears in
        # both halves: once to size the grid, once as the constexpr.
        cfg = (bt, -(-h2 // bj), batch_size * self.qkv_row, self.qkv_row,
               stride_cos, rd_half, self.head_dim, h2, bt, bj, bd, nw)
        self._rope_cfg[(rd_half, stride_cos, batch_size)] = cfg
        return cfg

    def _rope_qk_(self, qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                  seq_len: int, batch_size: int) -> None:
        """In-place half-split rotary on the q|k halves of ``qkv``."""
        if cos.stride(-1) != 1:
            cos = cos.contiguous()
        if sin.stride(-1) != 1:
            sin = sin.contiguous()
        rd_half = cos.shape[-1]
        stride_cos = cos.stride(0)
        key = (rd_half, stride_cos, batch_size)
        cfg = self._rope_cfg.get(key)
        if cfg is None:
            cfg = self._rope_cfg_for(rd_half, stride_cos, batch_size)
        bt, grid_j, stride_s, stride_b = cfg[0], cfg[1], cfg[2], cfg[3]
        # Positional, and the tail of ``cfg`` is already in signature order:
        # a kwargs launch costs ~1.8 us more per call.
        _rope_qk_inplace_kernel[
            (seq_len if bt == 1 else -(-seq_len // bt), grid_j, batch_size)
        ](qkv, cos, sin, seq_len, cos.shape[0], stride_s, stride_b, *cfg[4:-1],
          num_warps=cfg[-1], num_stages=2)

    def forward(
        self, x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        seq_len, batch_size, _ = x.shape
        qkv = self.qkv(x)

        nh, hd = self.num_heads, self.head_dim
        if (seq_len and rotary_pos_emb_cos is not None
                and rotary_pos_emb_sin is not None):
            self._rope_qk_(qkv, rotary_pos_emb_cos, rotary_pos_emb_sin,
                           seq_len, batch_size)

        q_size = nh * hd
        if batch_size == 1:
            # The (batch, seq, nh, hd) -> (batch*seq, nh, hd) tensor FA wants is
            # reachable in one step here, so name the strides directly: the
            # ``view -> index -> transpose -> reshape`` chain below is 12.1 us of
            # host time per call, three ``as_strided`` calls are 2.4 us.
            shape = (seq_len, nh, hd)
            stride = (self.qkv_row, hd, 1)
            base = qkv.storage_offset()
            q = qkv.as_strided(shape, stride, base)
            k = qkv.as_strided(shape, stride, base + q_size)
            v = qkv.as_strided(shape, stride, base + 2 * q_size)
        else:
            # No single stride triple flattens (batch, seq) for a seq-major
            # source, so ``reshape`` copies here -- which is the point: it keeps
            # v in the same (batch, seq) order as q/k rather than silently
            # disagreeing with them.
            qkv5 = qkv.view(seq_len, batch_size, 3, nh, hd)
            q, k, v = (qkv5[:, :, i].transpose(0, 1).reshape(-1, nh, hd)
                       for i in range(3))

        if max_seqlen is None:
            # Every captured caller passes ``max_seqlen``, so this is off the
            # measured path -- but when it is taken, one ``.tolist()`` is a
            # single 4*(nseg+1)-byte D2H copy, where
            # ``(cu[1:] - cu[:-1]).max().item()`` launched a sub, a max-reduce
            # and *then* still synchronised. Same value, one transfer, no
            # kernels. (A host-side cache keyed on the buffer would be faster
            # still, but cu_seqlens contents are not derivable from anything
            # host-visible, so any such cache can silently under-report
            # max_seqlen and truncate attention.)
            cu = cu_seqlens.tolist()
            max_seqlen = max(b - a for a, b in zip(cu, cu[1:]))

        out = self._attn(q, k, v, cu_seqlens, max_seqlen)

        out = out.view(seq_len, batch_size, -1)
        return self.proj(out)

    def _attn(self, q, k, v, cu_seqlens, max_seqlen):
        """Attention, on the tuned private FA4 path when it applies.

        Every condition below is host-side metadata (``shape`` / ``dtype`` /
        ``stride`` / ``numel`` / ``max_seqlen``); ``cu_seqlens`` is never read,
        so no device sync is introduced.  Anything outside the envelope, and any
        failure inside, falls through to the baseline ``FlashAttnPrefill`` call.
        """
        total_q = q.shape[0]
        if (self._fa4 and total_q > 0
                and 0 < max_seqlen <= total_q
                and q.dtype in (torch.bfloat16, torch.float16)
                and k.dtype is q.dtype and v.dtype is q.dtype
                and q.shape == k.shape == v.shape
                and q.stride(2) == 1 and k.stride(2) == 1 and v.stride(2) == 1
                and cu_seqlens.dtype is torch.int32
                and cu_seqlens.dim() == 1 and cu_seqlens.numel() >= 2
                and cu_seqlens.stride(0) == 1):
            nseg = cu_seqlens.numel() - 1
            # The 2-CTA MMA is a wash on its own and only pays with the TMA-O
            # epilogue, and then only inside two bounds, both measured:
            #
            # * ``max_seqlen <= 1024``.  The cluster's win is halving per-CTA
            #   KV smem, which stops mattering once a segment spans many more
            #   n-blocks than fit L2: at 2022-token segments (the max_seqlen
            #   4096 shapes) 2-CTA+TMA-O is 1.32x where CLC alone is 1.37x.
            #   Crossover measured at ~1500-token segments.
            # * enough m-tiles to fill the machine more than once.  One 256-row
            #   cluster tile per (segment, head); below ~1.3 waves the cluster
            #   only adds launch constraints (0.92x at nseg=2, 1.00x at nseg=3,
            #   1.02-1.06x from nseg=6 up).
            two_cta = (self._fa4_2cta and 256 < max_seqlen <= 1024
                       and nseg * q.shape[1] * -(-max_seqlen // 256) > _sm_count())
            try:
                return _fa4_varlen_fwd(q, k, v, cu_seqlens, cu_seqlens,
                                       max_seqlen, self.softmax_scale, two_cta)
            except Exception:  # pragma: no cover - fall through to the baseline
                pass

        return self.attn(
            q, k, v,
            cu_seqlens, cu_seqlens,
            max_seqlen, max_seqlen,
            softmax_scale=self.softmax_scale,
            causal=False,
            # Disable split-KV. With ``num_splits=0`` (auto) FA4's CuTeDSL
            # kernel runs ``num_splits_heuristic`` and, for the few m-blocks a
            # TP-sharded encoder produces (num_heads // tp, e.g. 16 // 4 = 4)
            # at moderate seqlens, picks ``num_splits > 1``. That enables the
            # ``is_split_kv`` path in ``flash_fwd_sm100.py``, whose
            # ``n_block_first`` is typed ``None`` on one branch and ``Int32``
            # on another -- a TYPE_UNSTABLE_JOIN CuTe compile error on
            # Blackwell (SM100).  Encoder self-attention is balanced
            # (q_len == k_len) so split-KV never helps here; forcing 1 is
            # numerically identical and sidesteps the kernel bug.  The paged
            # LLM prefill path (block_table/seqused_k) keeps auto-splitting,
            # where short-q-over-long-KV chunks do benefit.
            num_splits=1,
        )
