"""Encoder-only attention for Qwen vision transformer blocks.

Non-causal, no KV cache. Uses FlashAttnPrefill L1 op with cu_seqlens
for variable-length sequence support within the vision encoder.

Same contract as ``baseline.py``.  Two host-side changes, no new GPU kernel:

* **The materializing q/k copy is gone.**  ``self.qkv(x)`` returns a contiguous
  ``[S, 1, 3*q_size]`` tensor whose memory already *is* ``[S, 3, heads, dim]``, so
  q, k and v are taken as strided views of it (row stride ``3*q_size``, unit last
  stride) and handed to FlashAttention directly.  The baseline instead permutes
  the q|k half into a fresh ``(2, B, S, H, D)`` buffer, which at a full encoder
  batch is a large copy that ``scratch/stage_timing.log`` measures at 0.120 ms of a
  0.538 ms call at 20680 tokens -- 22% of the operator, for a relabelling.
* **One rotary launch instead of one over a copy.**  The rotary is a per-(token,
  head) map whose cos/sin depend only on the token row, so flash-attn's Triton
  ``apply_rotary`` is applied in place to a ``(1, S, 2*heads, dim)`` view of the
  q|k span of ``qkv``.  Going from ``(2, S, H, D)`` with grid
  ``(H/BLOCK_H, S/BLOCK_M, 2)`` to ``(1, S, 2H, D)`` with grid
  ``(2H/BLOCK_H, S/BLOCK_M, 1)`` changes only the block decomposition: every
  stride is passed to the kernel explicitly, ``cos``/``sin`` are indexed by the
  global token row independently of the batch and head ids, and the arithmetic is
  per element in fp32.  So the same elements get the same values.  The kernel is
  deliberately left unchanged -- it is what makes this provably value-identical to
  the baseline up to the attention call, which ``scratch/parity.py`` checks as a
  bitwise comparison rather than a tolerance.

The attention kernel itself is not reimplemented here: ``..L1.flash_attn_prefill``
resolves under the candidate loader to the frozen tuned FA4 CuTe winner, whose own
fast path is compiled for exactly this domain.

The no-copy path is an **allowlist**.  Everything it does not positively recognise
runs the baseline body, so behaviour outside the captured window is unchanged.
Each clause answers a specific way the views could be wrong:

* ``batch_size == 1`` -- the baseline orders q/k/v **batch-major** (its ``permute``
  puts batch outside seq); the views are **seq-major**.  Those agree only at
  ``B == 1``, which is every captured call.  At ``B > 1`` they genuinely differ,
  and ``cos`` would additionally need re-indexing per batch element.
* ``qkv.is_contiguous()`` -- the ``[S, 3, H, D]`` reinterpretation assumes it.
  ``F.linear`` guarantees it today; a future ``qkv`` returning a strided result
  would otherwise be silently misread.
* ``qkv.shape[-1] == 3 * q_size`` -- ``QKVParallelLinear`` replicates KV heads when
  ``total_num_kv_heads % tp != 0``, which widens the output past ``3 * q_size`` and
  invalidates the ``[q|k|v]``-thirds assumption.  The baseline's own slicing is
  equally wrong there, so falling back preserves behaviour rather than fixing it.
* Alignment -- ``flash_fwd_sm100`` calls ``assume_tensor_aligned`` on Q/K/V/O,
  which *assumes* via ``cute.assume`` (it does not assert) that the base pointer
  and every non-last stride is a multiple of 128 bits.  A violating layout would be
  silently wrong rather than an error, so the byte conditions are checked here
  instead of trusted.  For the captured bf16 / ``head_dim == 72`` case they hold
  comfortably: head stride 144 B, row stride 6912 B, q/k/v offsets 0 / 2304 / 4608 B.
* Inference only -- the in-place rotary now writes into the ``F.linear`` output
  rather than into a private copy, so a gradient-enabled call would see a mutated
  autograd input where the baseline saw a fresh buffer.  Those go to the baseline,
  which is the stance the frozen L1 winner takes too.  ``x`` is inspected as well as
  ``qkv``: under ``no_grad`` a caller's ``x.requires_grad`` does not propagate to
  ``qkv``, so checking ``qkv`` alone would admit an input the allowlist excludes.

* Fully specified -- both rotary tensors present and ``max_seqlen`` already a value.
  Neither is a *numerical* hazard: the fallback skips the rotary when either tensor is
  absent and derives a ``None`` ``max_seqlen`` from ``cu_seqlens`` exactly as the
  baseline does, so both would have been answered correctly on the fast path too.  The
  allowlist is nonetheless kept to the fully-specified call, which is the one that was
  measured; an incompletely specified call is cheap to hand back.

**Rejecting a layout has to reject it all the way to the kernel.** Only ``q`` and
``k`` are materialized by the fallback; ``v`` stays a view of ``qkv`` there, exactly
as in ``baseline.py``.  So a ``qkv`` turned away by the alignment clause would still
hand a same-alignment ``v`` to whatever attention runs next -- and the frozen L1
winner's own predicate looks only at ``stride(-1)``, not at base pointers or
non-last strides, so it would happily take its hand-built FA4 launch on a layout
this module had just refused.  That is a *divergence from the baseline*, which
reaches vLLM's wrapper instead.  So the attention call is part of the branch: the
admitted path uses the winner's ordinary ``forward``, and a rejected path is routed
to its baseline body (see ``_attention``).  Note what this does and does not buy --
vLLM's own FA4 prologue makes the same 16-byte assumption, so a misaligned pointer
is no safer there.  The property being established is narrower: a rejected input
follows the baseline's attention path, so the candidate cannot be wrong anywhere the
baseline is right.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from flash_attn.ops.triton.rotary import apply_rotary

from ....infra.tp import _tp_size, _tp_rank
from ..L1.flash_attn_prefill import FlashAttnPrefill
from .parallel_linear import QKVParallelLinear, RowParallelLinear

# What ``assume_tensor_aligned`` assumes of every pointer and non-last stride it is
# handed: a multiple of 128 bits.
_ALIGN_BYTES = 16


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

    def forward(
        self, x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        seq_len, batch_size, _ = x.shape
        qkv = self.qkv(x)

        # Recomputed per call, not cached in ``__init__``: ``num_heads`` and
        # ``head_dim`` are public attributes a caller can reach into, and a cached
        # product would silently disagree with the baseline after such a write.
        q_size = self.num_heads * self.head_dim

        admitted = self._admits_strided_qkv(
            qkv, batch_size, q_size, x, rotary_pos_emb_cos, rotary_pos_emb_sin,
            max_seqlen,
        )
        if admitted:
            q, k, v = self._strided_qkv(
                qkv, seq_len, q_size, rotary_pos_emb_cos, rotary_pos_emb_sin,
            )
        else:
            q, k, v = self._copied_qkv(
                qkv, seq_len, batch_size, q_size,
                rotary_pos_emb_cos, rotary_pos_emb_sin,
            )

        if max_seqlen is None:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        out = self._attention(admitted, q, k, v, cu_seqlens, max_seqlen)
        out = out.view(seq_len, batch_size, -1)
        return self.proj(out)

    def _attention(self, admitted, q, k, v, cu_seqlens, max_seqlen):
        """The attention call, which side of the branch it is on included.

        An admitted layout gets the frozen L1 winner's ordinary ``forward``, which is
        the whole point of the candidate.  A rejected one is sent to the winner's
        baseline body instead, so that a layout this module refused cannot be picked
        up again by the winner's own predicate -- which checks ``stride(-1)`` but not
        pointer or non-last-stride alignment -- and taken down its hand-built FA4
        launch.  That keeps a rejected input on the same attention path
        ``baseline.py`` would have used.
        """
        kwargs = dict(
            softmax_scale=self.head_dim ** -0.5,
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
        args = (q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen)
        if not admitted:
            # Absent under ``--standalone``, where ``..L1`` aliases the L1 baseline
            # and its ``forward`` already *is* the baseline body.
            baseline_body = getattr(self.attn, "_baseline_forward", None)
            if baseline_body is not None:
                return baseline_body(*args, **kwargs)
        return self.attn(*args, **kwargs)

    # ------------------------------------------------------------------
    # No-copy path
    # ------------------------------------------------------------------
    def _admits_strided_qkv(self, qkv, batch_size, q_size, x, cos, sin,
                            max_seqlen) -> bool:
        """True only for a layout the strided views provably reinterpret correctly.

        See the module docstring for why each clause is load-bearing.
        """
        if batch_size != 1:
            return False
        if qkv.shape[-1] != 3 * q_size:
            return False
        if not qkv.is_contiguous():
            return False

        # Only the fully-specified call is recognised. Both rotary tensors must be
        # present and ``max_seqlen`` must already be a value. Neither is a numerical
        # hazard -- the fallback skips the rotary when either tensor is absent, and
        # derives a ``None`` ``max_seqlen`` from ``cu_seqlens`` exactly as the baseline
        # does -- so declining them costs correctness nothing, and it keeps the
        # allowlist to the call this path was measured on.
        if cos is None or sin is None:
            return False
        if max_seqlen is None:
            return False

        # The in-place rotary writes into an autograd input rather than a private
        # copy, so anything that could be recording is handed to the baseline.
        # ``x`` is checked as well as ``qkv``: under ``no_grad`` the projection does
        # not propagate ``requires_grad``, so ``qkv`` alone would report clean for a
        # caller whose input is a graph leaf.
        if torch.is_grad_enabled() or qkv.requires_grad:
            return False
        if x.requires_grad:
            return False
        if cos is not None and cos.requires_grad:
            return False
        if sin is not None and sin.requires_grad:
            return False

        # What the FA4 CuTe kernel assumes rather than checks.  ``q``/``k``/``v``
        # get row stride ``3 * q_size`` and head stride ``head_dim``, at element
        # offsets 0, ``q_size`` and ``2 * q_size`` into ``qkv``.  The head-stride
        # clause formally implies the other two, since ``q_size`` is a multiple of
        # ``head_dim``; each physical requirement is still spelled out, because which
        # of them a future layout breaks is not obvious from the other.
        itemsize = qkv.element_size()
        if (self.head_dim * itemsize) % _ALIGN_BYTES:
            return False
        if (q_size * itemsize) % _ALIGN_BYTES:
            return False
        if (3 * q_size * itemsize) % _ALIGN_BYTES:
            return False
        if qkv.data_ptr() % _ALIGN_BYTES:
            return False
        return True

    def _strided_qkv(self, qkv, seq_len, q_size, cos, sin):
        """q, k and v as views of ``qkv``, rotated in place, without a copy."""
        if cos is not None and sin is not None:
            # One contiguous ``2 * q_size``-wide span per token row, relabelled as
            # ``2 * num_heads`` heads so the rotary covers q and k in one launch.
            # ``view`` rather than ``reshape`` on purpose: a copy here would be
            # rotated and then thrown away, so this must fail loudly instead.
            qk = (qkv.view(seq_len, 3 * q_size)[:, : 2 * q_size]
                  .view(1, seq_len, 2 * self.num_heads, self.head_dim))
            apply_rotary(qk, cos, sin, inplace=True)

        heads = qkv.view(seq_len, 3, self.num_heads, self.head_dim)
        # Each of these is ``(seq_len, num_heads, head_dim)`` with stride
        # ``(3 * q_size, head_dim, 1)``.  The unit last stride is what vLLM's
        # ``maybe_contiguous`` and the frozen L1 winner's guard look at, so nothing
        # downstream rewrites them.  The baseline already hands FlashAttention a
        # stride-``3 * q_size`` ``v``; this only puts q and k in the same position.
        return heads[:, 0], heads[:, 1], heads[:, 2]

    # ------------------------------------------------------------------
    # Fallback: the baseline's own dataflow
    # ------------------------------------------------------------------
    def _copied_qkv(self, qkv, seq_len, batch_size, q_size, cos, sin):
        """q, k and v exactly as ``baseline.py`` derives them."""
        # ``qkv`` is [q | k | v] on the last dim, so q and k are already
        # adjacent: take them as one slice, make that slice contiguous once, and
        # let rotary see it as a single (2*batch, seq, heads, dim) tensor. v then
        # needs no copy at all -- reshape gives a contiguous view because
        # batch_size is 1 here.
        qk = qkv[..., : 2 * q_size].view(
            seq_len, batch_size, 2, self.num_heads, self.head_dim,
        )
        # -> (2, batch, seq, heads, dim), one copy
        qk = qk.permute(2, 1, 0, 3, 4).contiguous()

        if cos is not None and sin is not None:
            flat = qk.view(2 * batch_size, seq_len, self.num_heads,
                           self.head_dim)
            apply_rotary(flat, cos, sin, inplace=True)

        q = qk[0].reshape(-1, self.num_heads, self.head_dim)
        k = qk[1].reshape(-1, self.num_heads, self.head_dim)
        # Keep v in the same (batch, seq) order as q/k. batch_size is 1 on every
        # current caller, but ordering v seq-major would silently disagree with
        # q/k if that ever changed.
        v = (qkv[..., 2 * q_size:]
             .view(seq_len, batch_size, self.num_heads, self.head_dim)
             .transpose(0, 1)
             .reshape(-1, self.num_heads, self.head_dim))
        return q, k, v
