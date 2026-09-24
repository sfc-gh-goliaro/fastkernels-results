"""Variable-length Flash Attention (no KV cache lookup).

Shape-specialised dispatch in front of vLLM's bundled FlashAttention build.
Two captured shape families dominate, and they want opposite things:

**MLA prefill** -- bf16, 16 heads, ``head_dim_qk=192`` / ``head_dim_v=128``,
``cu_seqlens.numel()`` between 2 and 4.

*Single sequence* (``numel() == 2``).  ``cu_seqlens_q == [0, total_q]`` means the
packed layout already *is* a dense ``[1, total_q, H, D]`` tensor, and handing
FA4's CuTeDSL forward no ``cu_seqlens`` unlocks the 2-CTA ``tcgen05`` MMA and the
dense tile scheduler.  Measured 1.12-1.30x (causal 16k x 16k) and 1.06-1.11x
(non-causal 16k x 64k).  Gated on the 192/128 geometry: the same rewrite *loses*
for symmetric small head dims (0.82x at ``d=64``, seqlen 8192).

*Multi sequence, causal* (``numel() >= 3``, or any ragged batch).  Here the
rewrite is unavailable, and the real cost turns out to be FA4's **epilogue**:
``flash_fwd_sm100.py`` sets ``use_tma_O = ... and not is_varlen_q``, so a varlen
call stores O with a predicated register->gmem copy issued by the *correction*
warps instead of a TMA bulk store issued by a dedicated warp.  A/B-ing that one
flag on a single-sequence input, where varlen and dense do byte-identical work,
accounts for the whole gap (1.24x of the 1.25x dense advantage on causal
16k x 16k; the tile scheduler and 2-CTA explain none of it -- causal disables
2-CTA anyway).  ``_VarlenTmaOFwd`` below turns TMA-O back on for varlen and keeps
it correct: a TMA store is clipped by the descriptor's ``total_q``, not by the
current segment, so the ragged last tile of a segment would overwrite the head of
the next one; that one tile per segment falls back to the predicated copy.
Measured 1.18-1.31x on causal 192/128 across ``nseg`` 1..32 and segment lengths
512..8192.  Non-causal loses (0.84-0.95x) and symmetric head dims are neutral, so
the path is gated on ``causal and (192, 128)``.

*Multi sequence, non-causal.*  Here what the dense rewrite buys is the 2-CTA
``tcgen05`` MMA, and ``use_2cta_instrs`` is gated on ``cu_seqlens_q is None`` for
no reason that survives testing: ``SingleTileVarlenScheduler`` already carries
``cluster_shape_m``, and passing ``use_2cta_instrs=True`` alongside ``cu_seqlens``
is 1.12-1.17x with an error indistinguishable from the 1-CTA path (checked against
an fp32 reference over 27 ragged/empty-segment configs), and re-tuning the
register split it then inherits from the *dense* 2-CTA kernel is another
1.12-1.17x on top (see ``_Varlen2CtaFwd``).  The ``not causal`` half of FA4's gate
is real, though: with a 256-row cluster tile the causal K-block bound is computed
per CTA, so half the K blocks get skipped -- 2.1x faster and max|err| 0.5.  (With
that bound corrected, causal 2-CTA is only ~1-2% over TMA-O alone, so it is not
worth the mask rework; measured, see ITERATIONS.md.)

**Short-sequence encoder attention** -- fp16, 16 heads, ``d=64``, 64-token
sequences, 1-32 of them per call.  Nothing here is compute bound: FA4's CuTeDSL
launcher costs more Python time than the kernel costs GPU time.  A plain Triton
tile kernel (written below, one CTA per q-tile x head x sequence) measures
1.25x-2.7x on that family purely by launching sooner.

Anything outside all three envelopes falls through to the unmodified baseline
call, and every specialised path is separately gated, so a failure in one cannot
affect the others.

Used by MLA prefill and chunked-context paths where Q, K, V are dense
``[total_tokens, num_heads, head_dim]`` tensors (no paged cache lookup,
no ``block_table``).  Supports ``return_softmax_lse`` for MLA chunked
prefix merging.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.fa_utils import FA_VERSION, flash_attn_varlen_func

try:  # FA4 (CuTeDSL) forward, the kernel vLLM selects on sm100.
    from vllm.vllm_flash_attn.cute.interface import (
        _flash_attn_fwd as _fa4_fwd_raw,
    )
except Exception:  # pragma: no cover - non-Blackwell / older wheel
    _fa4_fwd_raw = None

# Same reason the baseline hides ``flash_attn_varlen_func`` from Dynamo: FA4's
# CuTeDSL launcher rebuilds a Python closure per call, which Dynamo guards on.
_fa4_fwd = None if _fa4_fwd_raw is None else torch._dynamo.disable(_fa4_fwd_raw)

# MLA prefill geometry: the only (d_qk, d_v) pair where either FA4 rewrite below
# was measured faster than FA4's own varlen path.
_MLA_HEAD_DIMS = (192, 128)

# Short-sequence envelope for the Triton path (measured: 1.25x at seqlen 64,
# break-even by seqlen 512).
_TRI_MAX_SEQ = 128


# ---------------------------------------------------------------------------
# Triton varlen forward.  ``d_qk`` is fed as two power-of-two chunks so
# ``tl.arange`` stays legal (192 = 128 + 64) and no MMA lanes are wasted.
# ---------------------------------------------------------------------------
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _tri_varlen_fwd(
        Q, K, V, O, L,
        CUQ, CUK,
        scale_log2,
        sq_m, sq_h, sk_n, sk_h, sv_n, sv_h, so_m, so_h, sl_h,
        D1: tl.constexpr, D2: tl.constexpr, D_V: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        CAUSAL: tl.constexpr, WRITE_LSE: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)

        q_beg = tl.load(CUQ + pid_b)
        seqlen_q = tl.load(CUQ + pid_b + 1) - q_beg
        k_beg = tl.load(CUK + pid_b)
        seqlen_k = tl.load(CUK + pid_b + 1) - k_beg

        m_start = pid_m * BLOCK_M
        if m_start >= seqlen_q:
            return

        offs_m = m_start + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d1 = tl.arange(0, D1)
        offs_dv = tl.arange(0, D_V)
        m_valid = offs_m < seqlen_q

        q_row = Q + (q_beg + offs_m)[:, None] * sq_m + pid_h * sq_h
        q1 = tl.load(q_row + offs_d1[None, :], mask=m_valid[:, None], other=0.0)
        if D2 > 0:
            offs_d2 = D1 + tl.arange(0, D2)
            q2 = tl.load(q_row + offs_d2[None, :], mask=m_valid[:, None], other=0.0)

        acc = tl.zeros([BLOCK_M, D_V], dtype=tl.float32)
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

        # bottom-right aligned causal mask (FlashAttention varlen convention)
        delta = seqlen_k - seqlen_q
        if CAUSAL:
            n_hi = tl.minimum(seqlen_k, m_start + BLOCK_M + delta)
        else:
            n_hi = seqlen_k

        k_base = K + k_beg * sk_n + pid_h * sk_h
        v_base = V + k_beg * sv_n + pid_h * sv_h

        for n_start in tl.range(0, n_hi, BLOCK_N):
            nn = n_start + offs_n
            n_valid = nn < seqlen_k
            kcol = k_base + nn[None, :] * sk_n
            k1 = tl.load(kcol + offs_d1[:, None], mask=n_valid[None, :], other=0.0)
            qk = tl.dot(q1, k1)
            if D2 > 0:
                k2 = tl.load(kcol + offs_d2[:, None], mask=n_valid[None, :], other=0.0)
                qk = tl.dot(q2, k2, qk)
            qk = qk * scale_log2
            if CAUSAL:
                keep = (offs_m[:, None] + delta >= nn[None, :]) & n_valid[None, :]
            else:
                keep = n_valid[None, :]
            qk = tl.where(keep, qk, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            # A row with nothing visible yet (causal with seqlen_k < seqlen_q)
            # has m_i == m_new == -inf, and -inf - -inf is NaN.  Those rows have
            # l_i == 0 and acc == 0, so any finite rescale works; use 0.
            alive = m_new > float("-inf")
            alpha = tl.where(alive, tl.exp2(m_i - m_new), 0.0)
            p = tl.where(alive[:, None], tl.exp2(qk - m_new[:, None]), 0.0)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            v = tl.load(v_base + nn[:, None] * sv_n + offs_dv[None, :],
                        mask=n_valid[:, None], other=0.0)
            acc = tl.dot(p.to(v.dtype), v, acc)
            m_i = m_new

        l_safe = tl.where(l_i == 0.0, 1.0, l_i)
        acc = acc / l_safe[:, None]
        o_ptr = O + (q_beg + offs_m)[:, None] * so_m + pid_h * so_h + offs_dv[None, :]
        tl.store(o_ptr, acc.to(O.dtype.element_ty), mask=m_valid[:, None])
        if WRITE_LSE:
            lse = tl.where(l_i == 0.0, float("-inf"),
                           m_i * 0.6931471805599453 + tl.log(l_i))
            tl.store(L + pid_h * sl_h + q_beg + offs_m, lse, mask=m_valid)

except Exception:  # pragma: no cover - Triton missing
    triton = None
    _tri_varlen_fwd = None


_LOG2E = 1.4426950408889634
# ``d -> (d1, d2)`` power-of-two split, or None when the kernel cannot take it.
# ``D2 > 0`` (e.g. 96 = 64 + 32) makes Triton reject the second ``tl.dot`` at
# BLOCK_M=16, so only exact powers of two are accepted here.  The one
# non-power-of-two head dim in this workload (192) goes down the FA4 path.
_SPLIT_CACHE: dict[int, tuple[int, int] | None] = {}


def _split_pow2(d: int):
    try:
        return _SPLIT_CACHE[d]
    except KeyError:
        pass
    d1 = 1 << (d.bit_length() - 1)
    res = (d1, 0) if d1 == d else None
    _SPLIT_CACHE[d] = res
    return res


def _triton_varlen_attn(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                        softmax_scale, causal, return_softmax_lse, split):
    total_q, H, _ = q.shape
    d_v = v.shape[2]
    d1, d2 = split
    out = torch.empty((total_q, H, d_v), dtype=q.dtype, device=q.device)
    if return_softmax_lse:
        lse = torch.empty((H, total_q), dtype=torch.float32, device=q.device)
        sl_h = lse.stride(0)
    else:
        lse = out
        sl_h = 0
    # (BLOCK_M, BLOCK_N, num_warps) = (16, 64, 2) was the fastest of 36 configs
    # by *GPU* time on the 64-token case (2.67 us vs 2.87-7.58 us).  The harness
    # metric for this family tracks GPU time, not launch cost: its 252 MB
    # L2-flush memset queues ahead of the timed window and absorbs the CPU gap.
    grid = ((max_seqlen_q + 15) // 16, H, cu_seqlens_q.numel() - 1)
    _tri_varlen_fwd[grid](
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k,
        softmax_scale * _LOG2E,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1),
        v.stride(0), v.stride(1), out.stride(0), out.stride(1), sl_h,
        D1=d1, D2=d2, D_V=d_v, BLOCK_M=16, BLOCK_N=64,
        CAUSAL=causal, WRITE_LSE=return_softmax_lse,
        num_warps=2, num_stages=2,
    )
    return (out, lse) if return_softmax_lse else out


# ===========================================================================
# Source-level FA4 sm100 varlen forward: two mechanisms that ``interface.py``
# switches off purely because ``cu_seqlens_q is not None``.
#
#   causal      -> TMA-O epilogue (``_VarlenTmaOFwd``), 1.18-1.31x
#   non-causal  -> 2-CTA tcgen05 MMA + re-tuned registers (``_Varlen2CtaFwd``)
#
# Everything below is private to this module: a subclass of FA4's forward kernel
# plus a launcher with its own compile cache.  No flash_attn / vllm / cutlass
# module global, class attribute or cache is mutated, at import time or per call
# -- the harness times the unmodified baseline in the same process.
# ===========================================================================
_VARLEN_FA4_ERROR: Exception | None = None
_VARLEN_TMAO_ERROR: Exception | None = None

try:
    import cutlass as _cutlass
    import cutlass.cute as _cute
    from cutlass import Int32 as _Int32, const_expr as _const_expr
    from quack import copy_utils as _copy_utils, layout_utils as _layout_utils

    from vllm.vllm_flash_attn.cute import utils as _fa_utils
    from vllm.vllm_flash_attn.cute.cute_dsl_utils import (
        to_cute_tensor as _to_cute_tensor,
        torch2cute_dtype_map as _torch2cute,
    )
    from vllm.vllm_flash_attn.cute.flash_fwd_sm100 import (
        FlashAttentionForwardSm100 as _FwdSm100,
    )
    from vllm.vllm_flash_attn.cute.utils import AuxData as _AuxData

    class _VarlenTmaOFwd(_FwdSm100):
        """FA4's sm100 forward with TMA-O re-enabled for varlen.

        ``flash_fwd_sm100.py`` line ~190 reads

            self.use_tma_O = (... ) and not is_varlen_q
            self.use_correction_warps_for_epi = not self.use_tma_O

        so a varlen call gives up the TMA bulk store for O *and* moves the store
        onto the correction warps, where it serialises against the next tile's
        rescale.  That single flag is worth 1.18-1.31x on causal 192/128.

        Two things have to be fixed to turn it on:

        * **Correctness.**  A TMA store is clipped by the descriptor's global
          extent (``total_q``), not by the current segment, so the ragged last
          m-tile of segment *b* writes ``128 - seqlen_q % 128`` rows into the head
          of segment *b+1*.  ``epilogue_s2g`` below therefore stores that one tile
          per segment with the ordinary predicated register->gmem copy.  Tiles
          wholly inside a segment -- all of them when segments are 128-aligned --
          take the byte-identical-to-dense two-stage TMA path.
        * **Registers.**  Keeping the predicated copy reachable from the epilogue
          warp needs registers it does not have under the shipped causal-192
          split (softmax 192 / correction 72 -> ``num_regs_other`` 56): with the
          default split the whole win disappears (1.00-1.03x) even though the
          extra path is almost never taken.  ``184 / 72`` (-> other 72) restores
          it exactly; a variant that routes the ragged tile through TMA as well
          (wrong, diagnostic) measures 1.19-1.27x at *any* split, which is how the
          cost was attributed to register pressure rather than to the branch.
        """

        _REGS = (184, 72)  # (num_regs_softmax, num_regs_correction)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not self.pack_gqa and not self.is_split_kv
            assert self.is_varlen_q and not self.use_tma_O
            # (a) re-enable TMA-O and undo __init__'s warp-id shuffle (lines
            #     291-293): with TMA-O the epilogue is warp 13 alone and the
            #     correction warps only rescale.
            self.use_tma_O = True
            self.use_correction_warps_for_epi = False
            self.epilogue_warp_ids = (13,)
            self.empty_warp_ids = tuple(w for w in self.empty_warp_ids if w != 13)
            self.clc_scheduler_warp_id = (
                self.empty_warp_ids[0] if self.use_clc_scheduler else None
            )
            # (b) rebalance registers so the epilogue warp can afford the
            #     predicated fallback copy.
            self.num_regs_softmax, self.num_regs_correction = self._REGS
            self.num_regs_other = (
                512 - self.num_regs_softmax * 2 - self.num_regs_correction
            )

    class _Varlen2CtaFwd(_FwdSm100):
        """Non-causal varlen forward with the 2-CTA MMA and a re-tuned register
        split.

        The register counts come from ``_TUNING_CONFIG[(use_2cta_instrs,
        is_causal, head_dim_padded, is_sm103)]``, so switching 2-CTA on makes a
        varlen call inherit ``(True, False, 192) -> softmax 184 / correction 80``
        -- a split tuned for the *dense* 2-CTA kernel, where the epilogue is a
        separate TMA warp.  A varlen call has ``use_correction_warps_for_epi``,
        i.e. the correction warps do the predicated O store on top of the rescale,
        so they want registers the dense tune does not give them.  Moving 8 from
        each softmax group to correction, ``176 / 96`` (``num_regs_other``
        unchanged at 64), is 1.17x on nseg-2 and 1.12x on nseg-3 cross-attention,
        bit-exact, min-max bands disjoint.  The surface is a cliff, not a slope:
        ``176 / 88`` (same softmax, 8 fewer correction) is 0.63x.
        """

        _REGS = (176, 96)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert self.use_2cta_instrs and self.use_correction_warps_for_epi
            self.num_regs_softmax, self.num_regs_correction = self._REGS
            self.num_regs_other = (
                512 - self.num_regs_softmax * 2 - self.num_regs_correction
            )

    def _build_patched_call():
        """A private vendored copy of ``FlashAttentionForwardSm100.__call__``.

        Two edits, both inside its ``if const_expr(self.use_tma_O):`` arm:

        1. ``make_tiled_tma_atom`` rebinds ``mO`` to a pure *coordinate* tensor,
           which carries no pointer, so the epilogue could no longer do a
           predicated register->gmem store.  Keep the real tensor and rebuild the
           coordinate tensor on-device (see ``epilogue_s2g``).
        2. Build ``gmem_tiled_copy_O`` -- the source sets it to ``None`` when
           TMA-O is on -- so the ragged-tile fallback has a copy atom.  This is
           the source's own ``else``-branch code, verbatim.

        Patched from ``inspect.getsource`` rather than pasted so it tracks the
        installed wheel; both anchors must match exactly once or this raises and
        the caller falls back to the baseline.  Only ``linecache`` (under a
        private key, so ``inspect.getsource`` keeps working for the CuTe DSL's own
        AST pass) and this module's namespace are touched.
        """
        import inspect
        import linecache
        import textwrap

        import vllm.vllm_flash_attn.cute.flash_fwd_sm100 as _src_mod

        src = textwrap.dedent(inspect.getsource(_FwdSm100.__call__))
        edits = [
            ("tma_atom_O, mO = cpasync.make_tiled_tma_atom(",
             "tma_atom_O, _mO_tma_unused = cpasync.make_tiled_tma_atom("),
            ("        gmem_tiled_copy_O = None\n",
             "        universal_copy_bits = 128\n"
             "        async_copy_elems = universal_copy_bits // self.o_dtype.width\n"
             "        atom_universal_copy = cute.make_copy_atom(\n"
             "            cute.nvgpu.CopyUniversalOp(),\n"
             "            self.o_dtype,\n"
             "            num_bits_per_copy=universal_copy_bits,\n"
             "        )\n"
             "        tO_shape_dim_1 = sO_layout.outer.shape[1][0] // async_copy_elems\n"
             "        tO_layout = cute.make_ordered_layout(\n"
             "            (self.num_epilogue_threads // tO_shape_dim_1, tO_shape_dim_1),\n"
             "            order=(1, 0),\n"
             "        )\n"
             "        assert self.m_block_size % tO_layout.shape[0] == 0\n"
             "        vO_layout = cute.make_layout((1, async_copy_elems))\n"
             "        gmem_tiled_copy_O = cute.make_tiled_copy_tv(\n"
             "            atom_universal_copy, tO_layout, vO_layout\n"
             "        )\n"),
        ]
        for old, _ in edits:
            if src.count(old) != 1:
                raise RuntimeError(f"anchor matched {src.count(old)}x: {old!r}")
        for old, new in edits:
            src = src.replace(old, new)

        fname = "<fa4-varlen-tmao-patched-call>"
        linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)
        ns = dict(_src_mod.__dict__)
        exec(compile(src, fname, "exec"), ns)
        return ns["__call__"]

    try:
        _VarlenTmaOFwd.__call__ = _build_patched_call()
    except Exception as _exc:  # pragma: no cover - wheel drift
        _VARLEN_TMAO_ERROR = _exc
        _VarlenTmaOFwd = None

    _FA4_CACHE: dict = {}

    def _varlen_fa4_fwd(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                        max_seqlen_k, softmax_scale, causal, return_lse):
        """Trimmed private copy of ``interface._flash_attn_fwd``'s sm100 path.

        Narrowed to what the gate in ``forward`` admits (fp16/bf16, MHA, varlen,
        no paged KV / block sparsity / split-KV / fp8 / softcap / local / sink)
        and given its own compile cache, so FA4's global ``compile_cache`` is
        never keyed with one of our modified kernels.
        """
        total_q, num_head, head_dim = q.shape
        head_dim_v = v.shape[2]
        tile_m, tile_n = 128, 128
        q_stage = 2 if max_seqlen_q > tile_m else 1
        # 2-CTA is correct only for non-causal: with a 256-row cluster tile the
        # causal n-block bound is computed per CTA, so half the K blocks get
        # skipped (measured: 2.1x faster and max|err| 0.5).  See ITERATIONS.md.
        two_cta = not causal
        kernel_cls = _Varlen2CtaFwd if two_cta else _VarlenTmaOFwd

        out = torch.empty(total_q, num_head, head_dim_v,
                          dtype=q.dtype, device=q.device)
        lse = (torch.empty(num_head, total_q, dtype=torch.float32, device=q.device)
               if return_lse else None)

        key = (_torch2cute[q.dtype], head_dim, head_dim_v, causal,
               lse is None, tile_m, tile_n, q_stage, two_cta)
        compiled = _FA4_CACHE.get(key)
        if compiled is False:
            # Compilation already failed once for this key.  Without this memo a
            # broken configuration would re-enter ``cute.compile`` on *every*
            # call before falling back -- ~0.7 s of host time per call, i.e. a
            # 1000x slowdown rather than a clean fallback.
            raise RuntimeError("fa4 varlen compile previously failed")
        if compiled is None:
            _FA4_CACHE[key] = False
            fa = kernel_cls(
                head_dim, head_dim_v,
                qhead_per_kvhead=1,
                is_causal=causal,
                is_local=False,
                is_split_kv=False,
                pack_gqa=False,
                m_block_size=tile_m,
                n_block_size=tile_n,
                q_stage=q_stage,
                is_persistent=False,
                score_mod=None,
                mask_mod=None,
                has_aux_tensors=False,
                paged_kv_non_tma=False,
                is_varlen_q=True,
                q_subtile_factor=1,
                use_2cta_instrs=two_cta,
                use_clc_scheduler=False,
                output_quant_key=None,
            )
            compiled = _cute.compile(
                fa,
                *[_to_cute_tensor(t) for t in (q, k, v, out)],
                _to_cute_tensor(lse, assumed_align=4),
                softmax_scale,
                _to_cute_tensor(cu_seqlens_q, assumed_align=4, leading_dim=0),
                _to_cute_tensor(cu_seqlens_k, assumed_align=4, leading_dim=0),
                None, None,   # seqused_q, seqused_k
                None,         # dynamic_causal
                None,         # page_table
                None, None,   # window_size_left / right
                None,         # learnable_sink
                None,         # descale_tensors
                None,         # block sparse tensors
                _AuxData(None, None),
                None,         # output_scale
                _cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )
            _FA4_CACHE[key] = compiled

        compiled(
            q.detach(), k.detach(), v.detach(), out.detach(), lse,
            softmax_scale,
            cu_seqlens_q, cu_seqlens_k,
            None, None, None, None, None, None, None, None, None,
            _AuxData(None, None), None,
        )
        return (out, lse) if return_lse else out

    _varlen_fa4_fwd = torch._dynamo.disable(_varlen_fa4_fwd)

    def _2cta_disabled():
        """Honour ``FA_DISABLE_2CTA`` / the CUDA-12 2-CTA codegen workaround the
        same way ``interface.py`` does, without importing its private state at
        module scope (read-only)."""
        try:
            return _fa_utils._get_disable_2cta_default(is_fwd=True)
        except Exception:
            return True

except Exception as _exc:  # pragma: no cover - non-Blackwell / older wheel
    _VARLEN_FA4_ERROR = _exc
    _varlen_fa4_fwd = None
    _VarlenTmaOFwd = None

    def _2cta_disabled():
        return True


# Deliberately *not* decorated with ``@cute.jit`` and deliberately defined at
# module level: the CuTe DSL preprocesses a jit function by re-parsing the source
# *file* it was defined in and re-executing that file's module-level imports.
# This module's first import is the baseline's ``from ....infra.fa_utils import
# ...``, which only re-executes when the module happens to be loaded inside its
# package -- outside it the DSL raises ``ImportError: No module named 'infra'``
# and the kernel silently degrades to the baseline on every call.  So the body is
# written here for readability and then recompiled under a private key by
# ``_rebind_private`` below, where the only source the DSL can see is the function
# itself, with no imports to re-execute.
def _epilogue_s2g_impl(
    self,
    mO,
    sO,
    gmem_tiled_copy_O,
    tma_atom_O,
    pipeline_o_epi,
    block_info,
    num_splits,
    SeqlenInfoCls,
    mma_tile_coord_v=0,
    blocksparse_tensors=None,
    tile_scheduler=None,
):
    assert _const_expr(self.use_tma_O and not self.is_split_kv)
    assert _const_expr(not self.pack_gqa and not self.use_block_sparsity)
    # None would mean the patched __call__ did not run.
    assert _const_expr(gmem_tiled_copy_O is not None)
    tidx = _cute.arch.thread_idx()[0] % self.num_epilogue_threads

    # ``mO`` is the *real* tensor here (see ``_build_patched_call``).  The
    # TMA store wants the coordinate tensor ``make_tiled_tma_atom`` would
    # have returned in its place: the identity tensor with the contiguous
    # mode promoted to the descriptor's leading dimension.  For varlen O,
    # ``(total_q, head_dim_v, nheads)`` after ``__call__``'s transpose,
    # that is mode 1.
    mO_tma = _layout_utils.select(
        _cute.make_identity_tensor((mO.shape[1], mO.shape[0], mO.shape[2])),
        mode=[1, 0, 2],
    )
    tiler_gO = (self.mma_tiler_pv[0] * self.q_stage, self.head_dim_v_padded)

    epi_consumer_phase = _Int32(0)
    work_tile = tile_scheduler.initial_work_tile_info()
    while work_tile.is_valid_tile:
        m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
        seqlen = SeqlenInfoCls(batch_idx)

        def _tile(mT):
            cur = seqlen.offset_batch_Q(mT, batch_idx, dim=3)[None, None, head_idx]
            g = _cute.local_tile(cur, tiler_gO, (m_block, 0))
            g = _layout_utils.select(
                _cute.flat_divide(g, (self.mma_tiler_pv[0],)), mode=[0, 2, 1]
            )
            g = _cute.flat_divide(
                g, (self.mma_tiler_pv[0] // self.cta_group_size,)
            )[None, mma_tile_coord_v, None, None]
            return cur, g

        mO_cur, gO = _tile(mO)
        _, gO_tma = _tile(mO_tma)
        store_O, _, _ = _copy_utils.tma_get_copy_fn(
            tma_atom_O, 0, _cute.make_layout(1), sO, gO_tma
        )
        # First 128-row tile this work tile covers, within the segment.
        tile_base = m_block * self.q_stage * self.cta_group_size + mma_tile_coord_v
        last_tile = tile_base + (self.q_stage - 1) * self.cta_group_size
        if (last_tile + 1) * self.m_block_size <= seqlen.seqlen_q:
            for stage in _cutlass.range(self.q_stage, unroll_full=True):
                pipeline_o_epi.consumer_wait_w_index_phase(stage, epi_consumer_phase)
                store_O(src_idx=stage, dst_idx=stage)
                _cute.arch.cp_async_bulk_commit_group()
            for stage in _cutlass.range_constexpr(self.q_stage):
                _cute.arch.cp_async_bulk_wait_group(
                    self.q_stage - 1 - stage, read=True
                )
                pipeline_o_epi.consumer_release_w_index(stage)
        else:
            for stage in _cutlass.range_constexpr(self.q_stage):
                pipeline_o_epi.consumer_wait_w_index_phase(stage, epi_consumer_phase)
                m_tile_idx = tile_base + stage * self.cta_group_size
                if (m_tile_idx + 1) * self.m_block_size <= seqlen.seqlen_q:
                    store_O(src_idx=stage, dst_idx=stage)
                    _cute.arch.cp_async_bulk_commit_group()
                    _cute.arch.cp_async_bulk_wait_group(0, read=True)
                else:
                    self._store_O_to_gmem(
                        sO[None, None, stage], gO[None, None, stage], mO_cur,
                        gmem_tiled_copy_O, tidx, seqlen.seqlen_q, m_tile_idx,
                    )
                pipeline_o_epi.consumer_release_w_index(stage)
        epi_consumer_phase ^= 1
        work_tile = tile_scheduler.advance_to_next_work()


def _rebind_private(fn, names):
    """Recompile *fn* under a private linecache key in a namespace where every
    name it needs already exists, so the CuTe DSL's AST pass has no module-level
    imports to re-execute.  See the note on ``_epilogue_s2g_impl``."""
    import inspect
    import linecache
    import textwrap

    # The ``@cute.jit`` decorator is added here rather than at the ``def`` above,
    # so that importing this module never depends on the CuTe DSL being present.
    src = "@_cute.jit\n" + textwrap.dedent(inspect.getsource(fn))
    fname = f"<fa4-varlen-tmao-{fn.__name__}>"
    linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)
    ns = dict(names)
    exec(compile(src, fname, "exec"), ns)
    return ns[fn.__name__]


if _VarlenTmaOFwd is not None:
    try:
        _VarlenTmaOFwd.epilogue_s2g = _rebind_private(
            _epilogue_s2g_impl,
            {"_cute": _cute, "_cutlass": _cutlass, "_Int32": _Int32,
             "_const_expr": _const_expr, "_copy_utils": _copy_utils,
             "_layout_utils": _layout_utils},
        )
    except Exception as _exc:  # pragma: no cover
        # Without the override the class would inherit FA4's TMA-only epilogue,
        # which is *wrong* for ragged segments -- disable the path entirely.
        _VARLEN_TMAO_ERROR = _exc
        _VarlenTmaOFwd = None


class FlashAttnVarlen(nn.Module):
    """Variable-length Flash Attention without paged KV cache lookup."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float,
        causal: bool = True,
        return_softmax_lse: bool = False,
    ):
        # Everything below dispatches on host-side metadata only (``numel``,
        # ``shape``, ``dtype``, ``stride``, ``max_seqlen_*``); cu_seqlens is
        # never read on the host, so no device sync is introduced.
        d_qk = q.shape[2]
        d_v = v.shape[2]

        if (d_qk, d_v) == _MLA_HEAD_DIMS and q.shape[0] > 0 and FA_VERSION == 4:
            nseg_q = cu_seqlens_q.numel() - 1
            nseg_k = cu_seqlens_k.numel() - 1
            # numel()==2 means one sequence, so the varlen contract
            # (cu_seqlens_q == [0, q.shape[0]], cu_seqlens_k == [0, k.shape[0]])
            # makes the packed tensors *already* the dense batch-1 layout.  That
            # identity is the one thing here not re-derivable from host-side
            # metadata; verifying it would need a device sync, so we rely on the
            # contract, as every varlen caller in vLLM does.
            if _fa4_fwd is not None and nseg_q == 1 and nseg_k == 1:
                out, lse, _, _ = _fa4_fwd(
                    q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                    softmax_scale=softmax_scale,
                    causal=causal,
                    return_lse=return_softmax_lse,
                )
                out = out.squeeze(0)
                return (out, lse.squeeze(0)) if return_softmax_lse else out

            # Multi-sequence: causal -> TMA-O epilogue (1.18-1.31x);
            # non-causal -> 2-CTA MMA (1.12-1.17x).  Each is a loss for the other
            # polarity, and 2-CTA is outright *wrong* for causal, so the two are
            # picked strictly by ``causal`` inside ``_varlen_fa4_fwd``.
            if (_varlen_fa4_fwd is not None
                    and nseg_q == nseg_k >= 2
                    and (_VarlenTmaOFwd is not None if causal
                         else (max_seqlen_q > 256 and not _2cta_disabled()))
                    and 0 < max_seqlen_q <= q.shape[0]
                    and 0 < max_seqlen_k <= k.shape[0]
                    and q.dtype in (torch.float16, torch.bfloat16)
                    and k.dtype == q.dtype and v.dtype == q.dtype
                    and q.stride(2) == 1 and k.stride(2) == 1 and v.stride(2) == 1
                    and k.shape[1] == v.shape[1] == q.shape[1]
                    and cu_seqlens_q.dtype == cu_seqlens_k.dtype == torch.int32
                    and cu_seqlens_q.stride(0) == cu_seqlens_k.stride(0) == 1):
                try:
                    return _varlen_fa4_fwd(
                        q, k, v, cu_seqlens_q, cu_seqlens_k,
                        max_seqlen_q, max_seqlen_k, softmax_scale,
                        causal, return_softmax_lse)
                except Exception:  # pragma: no cover - fall through
                    pass

        if (_tri_varlen_fwd is not None
                and d_qk == d_v <= 128
                and 0 < max_seqlen_q <= _TRI_MAX_SEQ
                and 0 < max_seqlen_k <= _TRI_MAX_SEQ
                and q.shape[0] > 0
                and q.dtype in (torch.float16, torch.bfloat16)
                and q.stride(2) == 1 and k.stride(2) == 1 and v.stride(2) == 1
                and cu_seqlens_q.numel() == cu_seqlens_k.numel()):
            split = _split_pow2(d_qk)
            if split is not None:
                return _triton_varlen_attn(
                    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                    softmax_scale, causal, return_softmax_lse, split)

        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            return_softmax_lse=return_softmax_lse,
            fa_version=FA_VERSION,
        )
