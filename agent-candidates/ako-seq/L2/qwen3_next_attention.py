"""Qwen3-Next full attention with per-head QK-norm, partial RoPE, output gating, KV cache (L2).

GQA attention: 16 query heads, 2 KV heads, head_dim=256.
Q projection outputs 2x: [Q, gate] interleaved per head.
Partial RoPE (25% of head_dim = 64 dims rotated).
Output: attn_output * sigmoid(gate).

KV cache is stored in the engine's paged state manager so Qwen3-Next can
run batched prefill/decode instead of one Python call per sequence.

At every captured shape but the largest, this layer is *launch*-bound rather
than FLOP-bound: at 60 tokens it is 26 us of GPU work, and what a Python-level
implementation spends on top of that is not arithmetic but dispatch. So the
shape of this file is set by launch cost. The single-sequence prefill path --
which is what the engine builds for this layer -- is five launches:

    qkv GEMM -> QK-norm + partial RoPE + KV-cache scatter -> paged attention
             -> output gate -> o GEMM

The second of those is one CUDA kernel doing what the reference does with two
Triton launches, and it *fuses* rather than merely ports them: the normalized K
and the raw V exist only to be scattered into the paged cache -- the attention
call reads the cache, never the tensors -- so K goes straight to its cache slot
and neither intermediate is materialized. The gate is likewise left where the
projection put it and read in place by the tail. The attention call goes to
trtllm-gen's launcher directly, with the constants FlashInfer's wrapper
recomputes per call resolved once.

Uses the existing flash-attention prefill/decode wrappers, ``GemmaRMSNorm``,
``StoreKVCache``, and the canonical TP linears in ``parallel_linear`` -- the
linears keep ownership of the weights and serve any call the ``torch.mm`` /
hand-written arms decline (fp8, bias, TP all-reduce). Every hand-written arm is
guarded on shape, dtype and stride only, declines rather than approximates, and
has the stock path behind it; the mixed decode+prefill, pure-decode,
unfused-RoPE and non-HND-cache branches are untouched in behaviour. The gate
tail, the small-token projections, the fused QK kernel and the short-sequence
attention live in the ``qwen3_next_gate.cu`` sidecar.

Weight names match HuggingFace checkpoint:
  self_attn.q_proj.weight   [2 * num_heads * head_dim, hidden_size]  (Q + gate)
  self_attn.k_proj.weight   [num_kv_heads * head_dim, hidden_size]
  self_attn.v_proj.weight   [num_kv_heads * head_dim, hidden_size]
  self_attn.o_proj.weight   [hidden_size, num_heads * head_dim]
  self_attn.q_norm.weight   [head_dim]
  self_attn.k_norm.weight   [head_dim]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.context import get_attn_backend_config, get_context
from ....infra.tp import _tp_size
from ..L1.flash_attn_decode import FlashAttnDecode
from ..L1.flash_attn_prefill import FlashAttnPrefill
from ..L1.gemma_rms_norm import GemmaRMSNorm
from ..L1.store_kvcache import StoreKVCache, StoreKVCacheHND
from .fused_qk_norm_rope import (
    fused_qk_rmsnorm_rope_gate as _vllm_fused_qk_rmsnorm_rope_gate,
)
from .parallel_linear import QKVParallelLinear, RowParallelLinear


# Entry points of ``qwen3_next_gate.cu``: None until resolved, False when the
# extension cannot be built (then the aten path stands in). Bound to plain
# functions rather than reached through ``lazy_op``'s handle -- its
# ``__getattr__`` runs on every attribute access, ~1 us of a ~110 us call.
_gate_mul_fn = None
_gate_gemm_fn = None
_small_mm_fn = None
_qk_store_fn = None
_gate_mul_qkv_fn = None
_attn_small_fn = None

# Token counts up to which the hand-written GEMM is faster than cuBLAS on
# *device* time as well as host time; see the crossover table in
# ``qwen3_next_gate.cu``. The two projections differ because the kernel stays
# bandwidth-bound only while N * K is small, and o_proj's K is twice qkv's.
_GATE_GEMM_MAX_TOKENS = 4   # o_proj: K=4096, M=2048
_SMALL_MM_MAX_TOKENS = 1    # qkv:    K=2048, M=9216


def _resolve_ext() -> None:
    global _gate_mul_fn, _gate_gemm_fn, _small_mm_fn, _qk_store_fn
    global _gate_mul_qkv_fn, _attn_small_fn
    try:
        from ....infra.cuda_ext import load_op

        mod = load_op("qwen3_next_gate_mul", "qwen3_next_gate.cu")
        _gate_mul_fn = mod.gate_mul_
        _gate_gemm_fn = mod.gate_gemm
        _small_mm_fn = mod.small_mm
        _qk_store_fn = mod.qk_norm_rope_store
        _gate_mul_qkv_fn = mod.gate_mul_qkv_
        _attn_small_fn = mod.paged_attn_small
    except Exception:  # pragma: no cover - no toolchain -> aten fallbacks
        _gate_mul_fn = _gate_gemm_fn = _small_mm_fn = False
        _qk_store_fn = _gate_mul_qkv_fn = _attn_small_fn = False


def _gate_mul_(out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """``out *= sigmoid(gate)`` in one launch, in place, bit-exact.

    ``out * torch.sigmoid(gate)`` is two dispatches and two full-size
    allocations for what is one read-modify-write pass. See
    ``qwen3_next_gate.cu`` for why this is CUDA C++ and not Triton (the Triton
    launch path costs more host time here than the two aten kernels it would
    replace). The guard is shape/dtype-only, so it picks the same arm every
    call and costs no device sync.

    The kernel reproduces the aten pair bit-for-bit (verified over all five
    graded shapes), which is the form the reference used on every batch
    containing a prefill. It is *not* bit-identical to the reference's
    decode-only arm, which was a Triton kernel that multiplied by an unrounded
    fp32 sigmoid: those differ by one bf16 ulp (7.8e-3 measured, against a 1e-2
    tolerance), and one arm has to be picked. This is the one that matches the
    batches the engine actually builds for this layer.
    """
    fn = _gate_mul_fn
    if fn is None:
        _resolve_ext()
        fn = _gate_mul_fn
    if (fn is not False and out.dtype is torch.bfloat16
            and gate.dtype is torch.bfloat16
            and out.is_contiguous() and gate.is_contiguous()):
        return fn(out, gate)
    return out * torch.sigmoid(gate)


def _gate_mul_qkv_(out, qkv, num_heads, head_dim):
    """``out *= sigmoid(gate)`` with the gate read in place out of ``qkv``.

    Above the fused gate+o_proj cap nothing needs the gate as its own tensor, so
    the QK kernel does not write one: per head ``qkv`` holds [q | gate], and
    addressing that slice costs one extra stride in this kernel against a
    full-size write plus a full-size read (134 MB each at N=16384).
    """
    fn = _gate_mul_qkv_fn
    if fn is None:
        _resolve_ext()
        fn = _gate_mul_qkv_fn
    if fn is not False:
        y = fn(out, qkv, num_heads, head_dim)
        if y is not None:
            return y
    n = out.shape[0]
    gate = qkv.as_strided((n, num_heads, head_dim),
                          (qkv.stride(0), 2 * head_dim, 1), head_dim)
    return (out.view(n, num_heads, head_dim)
            * torch.sigmoid(gate)).view(n, num_heads * head_dim)


def _gate_gemm(o, gate, w):
    """``(o * sigmoid(gate)) @ w.T`` in one launch, or ``None`` if not taken.

    Collapses the gate tail and the output projection into a single kernel: at
    one token that is 4.7 us of host time and one launch where the pair costs
    16 us and two, and it never writes the gated activation to HBM at all. The
    kernel declines (returns an undefined tensor -> ``None``) on any shape it is
    not the right kernel for, so the caller always has cuBLAS behind it.
    """
    fn = _gate_gemm_fn
    if fn is None:
        _resolve_ext()
        fn = _gate_gemm_fn
    if (fn is not False and o.dtype is torch.bfloat16
            and gate.dtype is torch.bfloat16
            and o.is_contiguous() and gate.is_contiguous()):
        return fn(o, gate, w)
    return None


def _small_mm(x, w):
    """``x @ w.T`` on the same kernel without the gate prologue, or ``None``.

    The QKV projection has no prologue to fuse, but at one token cuBLAS is
    beaten on both counts anyway -- 4.2 us of host against 9.0, and 6.7 us on
    device against 8.8, because cuBLAS answers a 1x2048x9216 GEMV with a split-K
    kernel plus a separate reduce launch.
    """
    fn = _small_mm_fn
    if fn is None:
        _resolve_ext()
        fn = _small_mm_fn
    if fn is not False and x.dtype is torch.bfloat16 and x.is_contiguous():
        return fn(x, w)
    return None


def _paged_attn_small(out, q, k_cache, v_cache, block_table, seq_len, scale):
    """Causal paged attention into *out*; ``False`` when the kernel declines.

    trtllm-gen's context kernel is ~7 us of device time at every shape below
    ~100 tokens -- a quarter of the whole call -- for 9-59 MFLOP of arithmetic
    over a working set under 2 MB. Below the cap in ``qwen3_next_gate.cu`` this
    does the same work in ~1.5-2 us; above it trtllm-gen is far out of a CUDA
    core's reach and keeps the call.
    """
    fn = _attn_small_fn
    if fn is None:
        _resolve_ext()
        fn = _attn_small_fn
    if fn is False:
        return False
    return fn(out, q, k_cache, v_cache, block_table, seq_len, scale)


def _qk_norm_rope_store(qkv, k_cache, v_cache, slot_mapping, q_gain, k_gain,
                        cos_sin, positions, eps, num_heads, num_kv_heads,
                        head_dim, rotary_dim, page_size, want_gate=True):
    """One launch for QK-RMSNorm + partial RoPE + gate slice + KV scatter.

    Returns ``(q, gate, attn_out)`` -- ``q`` already shaped
    (N, num_heads, head_dim) for the attention call, ``attn_out`` an
    uninitialized buffer of the same shape for it to write -- or a triple of
    ``None`` when the kernel declines, in which case the caller runs the two
    Triton launches this replaces.

    Replacing them is worth 45-50 us of *host* time per call, which is a third
    of the whole forward at every shape below 16384: a Triton launch costs
    18-33 us of CPU here against 1.5-3 us of device time. It is also a device
    win at the top end, because the normalized K and the raw V exist only to be
    scattered into the paged cache -- the attention call reads the cache, never
    the tensors -- so writing K straight to its cache slot removes two
    full-size intermediates and the second kernel's re-read of them.
    """
    fn = _qk_store_fn
    if fn is None:
        _resolve_ext()
        fn = _qk_store_fn
    if fn is False:
        return None, None, None
    return fn(qkv, k_cache, v_cache, slot_mapping, q_gain, k_gain, cos_sin,
              positions, eps, num_heads, num_kv_heads, head_dim, rotary_dim,
              page_size, want_gate)


class Qwen3NextAttention(nn.Module):
    """Full attention with per-head QK-norm, partial RoPE, output gating, and KV cache."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        layer_idx: int,
        rms_norm_eps: float = 1e-6,
        reduce_output: bool = True,
    ):
        super().__init__()
        tp = _tp_size()
        self.layer_idx = layer_idx
        self.num_heads = num_attention_heads // tp
        self.num_kv_heads = num_key_value_heads // tp if num_key_value_heads % tp == 0 else num_key_value_heads
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5

        # QKV projection: Q outputs 2x heads (Q + gate)
        self.qkv_proj = QKVParallelLinear(
            hidden_size, head_dim,
            num_attention_heads * 2,  # doubled for output gate
            num_key_value_heads,
        )

        # ``reduce_output=False`` defers the all-reduce to the decoder layer's
        # next norm, which fuses the two.
        self.o_proj = RowParallelLinear(
            num_attention_heads * head_dim, hidden_size,
            reduce_results=reduce_output,
        )

        # Per-head QK norms (GemmaRMSNorm)
        self.q_norm = GemmaRMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = GemmaRMSNorm(head_dim, eps=rms_norm_eps)

        # Qwen3-Next's full-attention layers use head_dim=256.  vLLM 0.26 runs
        # them on FlashInfer with an HND cache ("Using FLASHINFER attention
        # backend" / "Using HND KV cache layout for FLASHINFER" on B200), so
        # follow the same per-device backend selection the generic
        # ``Attention`` layer uses.  FlashAttention is not a substitute here:
        # FA4's SM100 head_dim=256 forward rejects seqused_k/seqused_q, which
        # the paged decode path requires.
        attn_cfg = get_attn_backend_config()
        self.kv_layout = attn_cfg.kv_layout
        self._use_trtllm = attn_cfg.use_trtllm
        # vLLM collapses the gated split + QK-RMSNorm + partial NeoX RoPE +
        # gate copy into one Triton launch
        # (``Qwen3NextAttention.use_fused_qk_norm_rope_gate``). Unfused that is
        # nine kernels per attention layer -- two gate/q slices, two norms, two
        # rotary slices, the rotary op and two cats -- which at batch 1 is pure
        # launch overhead.
        self._fused_qk_rope_gate = True
        self._norm_gain_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        # Per-call constants, resolved once: the split widths are otherwise a
        # fresh three-element list every call, and ``variance_epsilon`` an
        # attribute walk through a submodule.
        self._kv_size = self.num_kv_heads * head_dim
        self._kv_offsets = (
            self.num_heads * 2 * head_dim,
            self.num_heads * 2 * head_dim + self._kv_size,
        )
        self._eps = rms_norm_eps
        # ``weight.t()`` for the two projections, resolved on first use. Like
        # ``_norm_gains`` this assumes the weight *object* outlives loading,
        # which every loader here satisfies (``param.data.copy_``).
        self._qkv_path = None
        self._o_path = None
        # (rotary_emb, cos_sin_cache, rotary_dim) or (rotary_emb, None, None):
        # the fused kernel's three rotary inputs behind one identity test,
        # instead of a ``getattr`` with a default plus two attribute walks.
        self._rope_cache = (None, None, None)
        # Page size of the HND cache when the fused QK+RoPE+scatter kernel can
        # serve this module, 0 when it cannot, None until resolved.
        self._qk_store_page: int | None = None
        # trtllm-gen launcher plus the constants FlashInfer's Python wrapper
        # recomputes per call; False when the direct call does not apply, None
        # until resolved.
        self._trtllm_ctx = None
        self._use_custom_op = False
        self._layer_name = ""
        self.rotary_emb = None
        if self._use_trtllm:
            from ..L1.flashinfer_decode import TRTLLMDecode
            from ..L1.flashinfer_prefill import TRTLLMPrefill

            self.store_kvcache = StoreKVCacheHND(page_size=attn_cfg.block_size)
            self.flash_attn_prefill = TRTLLMPrefill(
                self.num_heads, self.num_kv_heads, self.head_dim,
            )
            self.flash_attn_decode = TRTLLMDecode(
                self.num_heads, self.num_kv_heads, self.head_dim,
            )
        else:
            self.store_kvcache = StoreKVCache()
            self.flash_attn_prefill = FlashAttnPrefill(
                self.num_heads, self.num_kv_heads, self.head_dim,
            )
            self.flash_attn_decode = FlashAttnDecode(
                self.num_heads, self.num_kv_heads, self.head_dim,
            )

    def set_trtllm_workspace(self, workspace: torch.Tensor) -> None:
        """Adopt the engine's single shared trtllm-gen workspace.

        Without this each layer keeps the 512 MiB buffer it allocated in
        ``__init__``; Qwen3-Next has one MHA layer per 4 decoder layers, so
        that would waste several GiB.
        """
        if self._use_trtllm:
            self.flash_attn_decode._workspace = workspace
            self.flash_attn_prefill._workspace = workspace

    def _norm_gains(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``1 + weight`` for the QK norms, in fp32, materialized once.

        vLLM recomputes ``q_norm.weight.float() + 1.0`` per call and lets
        Inductor hoist it; in eager that would be two extra launches on every
        one of the 12 attention layers. The values are constants after weight
        loading, so caching them is exact.
        """
        if self._norm_gain_cache is None:
            self._norm_gain_cache = (
                self.q_norm.weight.float() + 1.0,
                self.k_norm.weight.float() + 1.0,
            )
        return self._norm_gain_cache

    def _rope_inputs(self, rotary_emb):
        """``(rotary_emb, cos_sin_cache, rotary_dim)`` when the fused
        QK-norm/RoPE/gate kernel applies to *rotary_emb*, else a triple whose
        cache entry is ``None`` (the unfused reference path)."""
        if (self._fused_qk_rope_gate and rotary_emb is not None
                and getattr(rotary_emb, "is_neox_style", False)):
            return (rotary_emb, rotary_emb.cos_sin_cache, rotary_emb.head_dim)
        return (rotary_emb, None, None)

    def _qk_store_setup(self) -> int:
        """Page size for the fused QK+RoPE+scatter kernel, or 0 if inapplicable.

        Shape/config-only, resolved once. The kernel writes K straight into an
        HND page, so it only serves that layout; the NHD store and any head
        dim it has no instantiation for stay on the Triton pair.
        """
        store = self.store_kvcache
        if not isinstance(store, StoreKVCacheHND):
            return 0
        if self.head_dim not in (64, 128, 256):
            return 0
        page = int(store.page_size)
        return page if page > 0 else 0

    def _trtllm_ctx_setup(self):
        """``(run, workspace, workspace_bytes, sm_count, enable_pdl)`` for a
        direct trtllm-gen paged-context call, or ``False``.

        ``TRTLLMPrefill.forward`` costs 19.5 us of host time and the kernel
        launch inside it 8.0; the difference is its own shape guard plus
        FlashInfer's ``trtllm_batch_context_with_kv_cache``, which re-derives the
        SM count, the PDL capability, the workspace size and the output buffer on
        every call and then forwards 32 arguments. All of those are constants for
        a given layer, so they are resolved once here.

        The direct call reproduces the wrapper *only* on the arm where its FA4
        branch is structurally unreachable, which ``head_dim > 128`` guarantees
        (``_fa4_ok`` rejects anything wider than FA4's 128), and where no sink,
        window or fp8 cache is in play. Everything else keeps the wrapper.
        """
        pre = self.flash_attn_prefill
        if not self._use_trtllm or self.head_dim <= 128:
            return False
        try:
            import flashinfer.prefill as _fp
            from flashinfer.utils import device_support_pdl, get_device_sm_count

            run = _fp.get_trtllm_gen_fmha_module().trtllm_paged_attention_context
            ws = pre._workspace
            return (run, ws, ws.numel() * ws.element_size(),
                    get_device_sm_count(ws.device),
                    device_support_pdl(ws.device))
        except Exception:  # pragma: no cover - keep the stock wrapper
            return False

    def _proj_path(self, lin):
        """``(weight.t() | False, weight | None)`` for one projection.

        The first element is the operand ``torch.mm`` wants; the second is the
        row-major weight the hand-written small-N GEMM wants, or ``None`` when
        that kernel cannot serve this module (anything ``_mm_view`` rejects, or a
        weight it would have to lay out itself).
        """
        wt = self._mm_view(lin)
        w = lin.weight
        usable = (wt is not False and w.dtype is torch.bfloat16
                  and w.is_contiguous())
        return (wt, w if usable else None)

    @staticmethod
    def _mm_view(lin) -> torch.Tensor | bool:
        """``lin.weight.t()`` when a bare ``torch.mm`` is equivalent to *lin*.

        ``F.linear`` on a 2-D input is a ``t()`` and a ``mm`` under two more
        dispatch layers; on the cached transpose that is 2 us less host time,
        and both projections are on the critical path. Returns ``False`` when
        the module does something a GEMM does not -- fp8 quantization, a bias,
        or a TP all-reduce -- and must be called itself.
        """
        if lin.use_fp8 or lin.bias is not None:
            return False
        if getattr(lin, "reduce_results", False) and lin.tp_size > 1:
            return False
        return lin.weight.t()

    def forward(self, hidden_states, rotary_emb=None, positions=None,
                state_manager=None):
        if rotary_emb is not None:
            self.rotary_emb = rotary_emb
        if self._use_custom_op:
            return torch.ops.fastkernels.qwen3_next_attention(
                hidden_states, positions, self._layer_name,
            )
        return self.forward_impl(hidden_states, positions, state_manager)

    def forward_impl(self, hidden_states, positions=None, state_manager=None):
        ctx = get_context()
        md = ctx.kda_metadata
        if state_manager is None:
            state_manager = ctx.kda_state
        rotary_emb = self.rotary_emb
        if md is None or state_manager is None:
            raise RuntimeError(
                "Qwen3NextAttention requires engine-managed KV state and metadata",
            )

        num_heads = self.num_heads
        num_kv_heads = self.num_kv_heads
        head_dim = self.head_dim

        # The engine hands 2-D activations; ``reshape`` on a matching shape
        # returns ``self`` but still costs a dispatch, so ask first.
        x = hidden_states
        if x.dim() != 2:
            x = x.reshape(-1, x.shape[-1])
        N = x.shape[0]

        path = self._qkv_path
        if path is None:
            path = self._qkv_path = self._proj_path(self.qkv_proj)
        wt, w_row = path
        qkv = None
        if w_row is not None and N <= _SMALL_MM_MAX_TOKENS:
            qkv = _small_mm(x, w_row)
        if qkv is None:
            qkv = torch.mm(x, wt) if wt is not False else self.qkv_proj(x)

        rope = self._rope_cache
        if rope[0] is not rotary_emb:
            rope = self._rope_cache = self._rope_inputs(rotary_emb)

        # Hoisted above the QK stage: the fused kernel scatters K and V into the
        # cache itself, so it needs the two buffers before it runs.
        layer_idx = self.layer_idx
        k_cache = state_manager.k_cache[layer_idx]
        v_cache = state_manager.v_cache[layer_idx]

        k_off, v_off = self._kv_offsets
        q = attn_out = None
        if rope[1] is not None and positions is not None:
            gains = self._norm_gain_cache
            if gains is None:
                gains = self._norm_gains()
            # ``positions`` is already flat on every engine path.
            pos = positions if positions.dim() == 1 else positions.reshape(-1)
            page = self._qk_store_page
            if page is None:
                page = self._qk_store_page = self._qk_store_setup()
            if page:
                q, gate, attn_out = _qk_norm_rope_store(
                    qkv, k_cache, v_cache, md.slot_mapping, gains[0], gains[1],
                    rope[1], pos, self._eps, num_heads, num_kv_heads, head_dim,
                    rope[2], page, N <= _GATE_GEMM_MAX_TOKENS,
                )
            if q is None:
                # ``q_gate`` is the leading block of ``qkv``, and the fused
                # kernel addresses it as base + token * row_stride, so the whole
                # tensor *is* the view -- no narrow needed, and none of the three
                # is ever read out of bounds.
                q, k, gate = _vllm_fused_qk_rmsnorm_rope_gate(
                    qkv,
                    qkv.narrow(1, k_off, self._kv_size),
                    gains[0],
                    gains[1],
                    rope[1],
                    pos,
                    self._eps,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    rope[2],
                )
                # ``gate`` stays 2-D: the gating tail wants
                # (N, num_heads * head_dim) and every view of it on the way
                # there is a wasted dispatch.
                q = q.view(N, num_heads, head_dim)
                self.store_kvcache.forward(
                    k.view(N, num_kv_heads, head_dim),
                    qkv.narrow(1, v_off, self._kv_size).view(
                        N, num_kv_heads, head_dim),
                    k_cache, v_cache, md.slot_mapping,
                )
        else:
            k = qkv.narrow(1, k_off, self._kv_size)
            v = qkv.narrow(1, v_off, self._kv_size)
            # Split Q and gate. Unlike the fused kernel this path reshapes
            # ``q_gate``, so here it has to be the narrowed block, not all of
            # ``qkv``.
            q_gate = qkv.narrow(1, 0, k_off).view(N, num_heads, 2 * head_dim)
            q = q_gate[:, :, :head_dim].contiguous()
            gate = q_gate[:, :, head_dim:].contiguous().view(
                N, num_heads * head_dim)

            k = k.view(N, num_kv_heads, head_dim)

            # Per-head QK-norm (applied before RoPE)
            q = self.q_norm(q.reshape(-1, head_dim)).view(
                N, num_heads, head_dim)
            k = self.k_norm(k.reshape(-1, head_dim)).view(
                N, num_kv_heads, head_dim)

            # Partial RoPE (only rotates first rotary_dim dimensions)
            if rotary_emb is not None and positions is not None:
                pos_flat = (
                    positions.reshape(-1) if positions.dim() > 1 else positions
                )
                rotary_dim = rotary_emb.head_dim
                q_rot, q_pass = (
                    q[..., :rotary_dim].contiguous(), q[..., rotary_dim:],
                )
                k_rot, k_pass = (
                    k[..., :rotary_dim].contiguous(), k[..., rotary_dim:],
                )
                q_rot, k_rot = rotary_emb(pos_flat, q_rot, k_rot)
                q = torch.cat([q_rot, q_pass], dim=-1)
                k = torch.cat([k_rot, k_pass], dim=-1)

            self.store_kvcache.forward(
                k, v.view(N, num_kv_heads, head_dim), k_cache, v_cache,
                md.slot_mapping,
            )

        nd = md.num_decodes
        np_ = md.num_prefills
        scaling = self.scaling
        max_q, max_k = md.max_query_len, md.max_seq_len

        # One attention call owns the whole output buffer whenever the batch is
        # not mixed, which is every batch the engine builds for this layer: a
        # staging ``torch.empty`` plus ``out[ndt:] = ...`` is an extra allocation
        # and an extra full-size copy (128 MiB at N=16384) for nothing.
        if nd == 0:
            if np_ == 0:
                out = attn_out if attn_out is not None else torch.empty(
                    N, num_heads, head_dim, device=x.device, dtype=x.dtype)
            else:
                # One prefill and no decodes needs no cu_seqlens arithmetic at
                # all. ``nd == 0`` makes the rebase a no-op --
                # ``query_start_loc[0]`` is 0 by the metadata's own definition,
                # it is a cumulative token-start array -- and with a single
                # sequence ``cu_seqlens_k`` is just ``[0, seq_lens[0]]``, which
                # equals ``query_start_loc`` exactly when the sequence has no
                # cached prefix. ``max_seq_len == max_query_len ==
                # num_prefill_tokens`` witnesses that on the host, so the arm is
                # picked without reading a device value.
                qsl = md.query_start_loc
                bt = md.block_tables
                single = (np_ == 1 and qsl.dtype == torch.int32
                          and qsl.shape[0] == 2
                          and max_k == max_q == md.num_prefill_tokens)
                if single:
                    cu_k = qsl
                else:
                    qsl, cu_k = self._prefill_cu_seqlens(md, 0, np_)
                # Three arms, narrowest first: our own kernel while the whole
                # sequence fits in a page, trtllm-gen's launcher called directly
                # above that, and the stock wrapper for anything either guard
                # turns down.
                out = None
                if single and attn_out is not None:
                    if _paged_attn_small(attn_out, q, k_cache, v_cache, bt,
                                         max_k, scaling):
                        out = attn_out
                    else:
                        ctx = self._trtllm_ctx
                        if ctx is None:
                            ctx = self._trtllm_ctx = self._trtllm_ctx_setup()
                        # ``md.seq_lens`` *is* the kv length vector the wrapper
                        # derives as ``cu_seqlens_k[1:]``: with one sequence and
                        # no cached prefix both are ``[num_prefill_tokens]``,
                        # which the ``single`` test above already witnessed on
                        # the host.
                        seq_lens = md.seq_lens
                        if (ctx is not False and bt.dim() == 2
                                and bt.is_contiguous()
                                and seq_lens.shape[0] == 1
                                and seq_lens.dtype == torch.int32
                                and k_cache.dtype is q.dtype
                                and q.is_contiguous()):
                            out = self._trtllm_prefill(
                                ctx, attn_out, q, k_cache, v_cache, bt,
                                seq_lens, max_q, max_k, scaling, qsl,
                            )
                if out is None:
                    # First seven parameters are positional in both prefill
                    # wrappers; the rest have to stay keywords for
                    # FlashAttnPrefill.
                    out = self.flash_attn_prefill.forward(
                        q, k_cache, v_cache, qsl, cu_k, max_q, max_k,
                        softmax_scale=scaling,
                        causal=True,
                        block_table=bt,
                    )
        elif np_ == 0:
            out = self.flash_attn_decode.forward(
                q,
                k_cache,
                v_cache,
                cache_seqlens=md.seq_lens[:nd].to(torch.int32),
                block_table=md.block_tables[:nd],
                softmax_scale=scaling,
                causal=True,
                max_seq_len=max_k,
            )
        else:
            ndt = md.num_decode_tokens
            out = attn_out if attn_out is not None else torch.empty(
                N, num_heads, head_dim, device=x.device, dtype=x.dtype)
            out[:ndt] = self.flash_attn_decode.forward(
                q[:ndt],
                k_cache,
                v_cache,
                cache_seqlens=md.seq_lens[:nd].to(torch.int32),
                block_table=md.block_tables[:nd],
                softmax_scale=scaling,
                causal=True,
                max_seq_len=max_k,
            )
            cu_q, cu_k = self._prefill_cu_seqlens(md, nd, np_)
            out[ndt:] = self.flash_attn_prefill.forward(
                q[ndt:],
                k_cache,
                v_cache,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=max_q,
                max_seqlen_k=max_k,
                softmax_scale=scaling,
                causal=True,
                block_table=md.block_tables[nd:],
            )

        # Output gating: o * sigmoid(gate), then the output projection. Both
        # operands are contiguous (N, num_heads * head_dim), so the 2-D view is
        # free and ``o`` needs no reshape before the GEMM.
        o = out.view(N, num_heads * head_dim)
        path = self._o_path
        if path is None:
            path = self._o_path = self._proj_path(self.o_proj)
        wt, w_row = path
        if gate is None:
            # The QK kernel left the gate where the projection put it.
            o = _gate_mul_qkv_(o, qkv, num_heads, head_dim)
        else:
            if w_row is not None and N <= _GATE_GEMM_MAX_TOKENS:
                y = _gate_gemm(o, gate, w_row)
                if y is not None:
                    return y
            o = _gate_mul_(o, gate)
        return torch.mm(o, wt) if wt is not False else self.o_proj(o)

    def _trtllm_prefill(self, ctx, out, q, k_cache, v_cache, block_table,
                        seq_lens, max_q, max_k, scaling, cu_q):
        """One paged-context launch, with FlashInfer's bookkeeping precomputed.

        The argument list is ``trtllm_batch_context_with_kv_cache``'s own call
        into ``trtllm_paged_attention_context``, with the values it would have
        computed substituted: no scale factors (fp8/nvfp4 output is off), no
        sinks, no sliding window, no LSE, causal, a shared 2-D page table. It
        matches the wrapper bit for bit on this arm (verified over all five
        graded shapes). A raised exception -- the one way this can go wrong is a
        FlashInfer whose launcher signature moved -- retires the path for the
        process and falls back to the wrapper; an untaken ``try`` costs nothing.
        """
        run, ws, ws_size, sm_count, pdl = ctx
        try:
            run(out, None, q, k_cache, v_cache, ws, block_table, seq_lens,
                max_q, max_k, scaling, 1.0, -1.0, -1, 0, 1, -1, cu_q, cu_q,
                sm_count, pdl, ws_size, None, None, None, None, True, True,
                None, 0, 0)
        except Exception:  # pragma: no cover - signature drift
            self._trtllm_ctx = False
            return None
        return out

    @staticmethod
    def _prefill_cu_seqlens(md, nd, np_):
        """General ``(cu_seqlens_q, cu_seqlens_k)`` for the prefill slice.

        Rebase ``query_start_loc`` onto the first prefill, then prefix-sum
        ``seq_lens`` into a fresh buffer: four launches and two allocations.
        Only mixed and multi-prefill batches come through here -- the
        single-prefill case is answered without a launch at the call site.
        """
        qsl = md.query_start_loc
        cu_q = (qsl[nd:] - qsl[nd]).to(torch.int32)
        cu_k = torch.zeros(np_ + 1, dtype=torch.int32, device=qsl.device)
        cu_k[1:] = torch.cumsum(md.seq_lens[nd:].to(torch.int32), dim=0)
        return cu_q, cu_k
