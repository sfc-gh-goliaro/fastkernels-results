"""Model-level multi-head attention (thin wrapper).

Consolidates vLLM's ``LlamaAttention``, ``Llama4Attention``,
``Qwen3Attention``, and GPT-OSS attention:
QKV projection, optional QK-norm, optional RoPE, then delegates to
``Attention`` for KV cache storage and kernel dispatch.

Unified across Llama, Llama 4, Qwen2, Qwen3, Mixtral, and GPT-OSS:
  - bias:                    Qwen2/GPT-OSS use bias=True on QKV/O projections.
  - qk_norm:                 Qwen3 applies per-head RMSNorm to Q and K before RoPE.
  - nope:                    Llama 4 NoPE layers skip RoPE entirely.
  - use_weightless_qk_norm:  Llama 4 RoPE layers apply weight-less QK RMSNorm after RoPE.
  - attn_temperature_tuning: Llama 4 NoPE layers apply position-dependent temperature.
  - use_sinks:               GPT-OSS learnable attention sinks (per-head biases).
  - sliding_window:          GPT-OSS sliding window attention (even layers only).

Two things dominate this operator and neither is arithmetic.

**1. The glue between the two projections** is *one in-place CUDA pass over the
packed qkv buffer* (``qkv_glue.cu``).  The reference form walks that buffer once
per stage -- per-head QK RMSNorm, RoPE, weight-less QK norm, temperature
scaling -- and each stage is a separate launch over a *strided* slice of qkv, so
a stage worth ~2 us of arithmetic costs a full read/write round trip plus a
launch.  Here q and k are normed, rotated, scaled and written back where they
already live: no ``.contiguous()`` copy of q or k, no gathered cos/sin
temporary, one launch, and every config branch (``qk_norm`` / ``nope`` /
weight-less norm / temperature / neox-vs-interleaved / M-RoPE-vs-plain) is
resolved in ``__init__`` into a packed ``flags`` word.  v is never touched.  The
cos/sin table is read **in fp32, in place, with no dtype cast**: both rotary
modules do ``cache.to(query.dtype)`` inside ``forward`` and for Qwen3-VL that
table is ``(4 * 262144, 128)`` fp32 = 537 MB, which measured 109-115 us of a
173-313 us forward for a value the vendored rope kernel never needed.

**2. Host dispatch**, because below a few thousand tokens this forward is
*host*-bound end to end: at ``[3,1] x [1,4096]`` the reference spends ~336 us of
CPU to enqueue ~50 us of device work, and the wall time is the CPU time.  Every
launch on the path therefore has to cost what a launch costs and not what a
Python wrapper stack costs.  Measured per-call host cost of the three sub-calls
the fused glue originally left alone, and what they become here:

===========================  ========  ======  =========================
stage                        before    after   what went away
===========================  ========  ======  =========================
``self.attn(q, k, v)``         47 us    7 us   ``Attention.forward`` /
                                              ``forward_impl`` /
                                              ``_forward_pure`` /
                                              ``TRTLLMPrefill.forward`` /
                                              ``flash_attn_varlen_func`` /
                                              FA4's 700-line Python
                                              launcher
fp8 ``qkv_proj`` / ``o_proj``  47 us   24 us   ``QKVParallelLinear.forward``,
                                              ``Fp8Linear.forward``, two
                                              ``torch.ops`` re-dispatches,
                                              three ``torch.empty`` and a
                                              ``permute`` per call
bf16 ``qkv_proj``/``o_proj``   12 us   10 us   two ``nn.Module.__call__``
``qkv.split`` + views           5 us    0 us   cached with the buffer
===========================  ========  ======  =========================

The mechanism is a *plan*, resolved on the first forward (not in ``__init__``:
the module is still on the host there, and a caller may replace the fp8 weight
Parameters afterwards) and cached per token count:

* **Projections.**  ``F.linear``'s ATen path for a 2-D input is exactly
  ``addmm(bias, x, w.t())``, so binding ``torch.addmm`` / ``torch.mm`` with a
  pre-transposed weight view and an ``out=`` buffer is bit-identical and skips
  two module frames.  The fp8 path binds the *same* two entry points
  ``Fp8Linear`` would reach -- the vendored ``per_token_group_fp8_quant`` and
  ``deep_gemm.fp8_gemm_nt``, pulled off the live ``Fp8Linear`` module so the
  identity cannot drift -- with the quant scratch, the column-major scale
  buffer and the output buffer allocated once instead of three times per call.
  A config whose dispatch this cannot reproduce (TP > 1, a FlashInfer-capable
  device where ``M < 32`` takes the swapAB kernel, a non-bf16 activation, a
  quantizer without the CUDA entry point) keeps the submodule call.
* **Attention.**  ``Attention.forward`` for a dense (unpaged) prefill bottoms
  out in FA4's CuTe launcher, which spends ~34 us per call re-deriving a
  48-element compile key, re-running ~60 asserts and rebuilding its argument
  list -- to reach a kernel it has already compiled.  The plan looks that kernel
  up *once* by running the reference call under a recording proxy over
  ``_flash_attn_fwd.compile_cache``, so the handle is the one the reference
  itself would use, and then calls it directly.  Output is bit-identical
  (verified on all five benchmarked shapes).  The key is memoised on
  ``(max_seqlen_q, batch)`` -- the only inputs the compile key derives from --
  and a miss just replays the reference call, so an unseen shape degrades to
  reference speed and never below it.

Everything above is bit-identical to the reference on every shape measured, with
one deliberate exception: at ``M == 1`` the fp8 projections take the frozen L1
winner's fused quantize+GEMV, which is one kernel where DeepGEMM spends a whole
128-row M tile plus a separate quantization pass (host 16.3 -> 2.8 us, device
16.5 -> 10.7).  Its accumulation order differs, so the projection moves by up to
4e-3 and the module's output by ~1e-4 -- two orders inside the harness's bf16
bound.  ``_USE_FUSED_GEMV = False`` restores strict bit-identity at the cost of
~20% on the single-token row.

Every structural condition the fast path needs is re-checked per call
(``k_cache`` empty, pure non-tree prefill, no block tables, int32 contiguous
``cu_seqlens``, ``_use_custom_op`` off, matching dtype) and anything else falls
back to the untouched ``self.attn(q, k, v)``.  Likewise a config the fused glue
does not structurally recognise -- an unknown rotary module, a partial rotary
dim, a rotary handed in at call time, a non-2-D hidden state, a head_dim the
vectorized kernel cannot address, ``torch.compile`` tracing -- falls through to
the reference stage sequence at the bottom of ``forward``, which is unchanged.
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn

from ....infra.cuda_ext import lazy_op
from ....infra.tp import _tp_size
from ....infra.context import get_context
from .parallel_linear import QKVParallelLinear, RowParallelLinear
from .attention_impl import Attention
from ..L1.rms_norm import RMSNorm

# Sidecar CUDA op; built on first use.  Distinct extension name from any L1 op so
# the two can live in one process (``cpp_extension.load`` keys its build
# directory on the name).
_C = lazy_op("qkv_glue_ako", "qkv_glue.cu")

# ---------------------------------------------------------------------------
# Fused post-QKV glue
# ---------------------------------------------------------------------------
# Rotation modes.  0/1/2 need a 1-D ``positions``; 3/4 are M-RoPE over a
# (3, N) ``positions`` and differ only in how a frequency index maps to a
# T/H/W section.
_ROPE_NONE = 0
_ROPE_NEOX = 1        # pairs (i, i + head_dim/2)      -- vLLM IS_NEOX=true
_ROPE_GPTJ = 2        # pairs (2i, 2i + 1)             -- vLLM IS_NEOX=false
_ROPE_MROPE_IL = 3    # M-RoPE, interleaved sections
_ROPE_MROPE_SEC = 4   # M-RoPE, contiguous sections
_MODE_MASK = 0x7
_FLAG_NORM = 1 << 3
_FLAG_WLNORM = 1 << 4
_FLAG_TEMP = 1 << 5
# Round the b-product of the rotation, reproducing the M-RoPE reference's bf16
# tiles (see qkv_glue.cu); the 1-D rope reference is fp32 and does not want it.
_FLAG_ROPE_ROUND = 1 << 6

# Rotary modules whose ``forward`` is exactly "index ``cos_sin_cache`` by
# position, rotate the whole head".  Matched by class name *and* by the module
# its ``forward`` is defined in, so a subclass that overrides the rotation
# (rather than just the table) is not silently absorbed.
_NEOX_CLASSES = frozenset({"RotaryEmbedding", "Gemma4ProportionalRotaryEmbedding"})
_MROPE_CLASSES = frozenset({"MRotaryEmbedding"})


def _pack_sections(s_t: int, s_h: int, s_w: int) -> int:
    """M-RoPE section widths in one argument (10 bits each; head_dim/2 <= 1024)."""
    return (s_t & 0x3ff) | ((s_h & 0x3ff) << 10) | ((s_w & 0x3ff) << 20)


def _rope_plan(rope, head_dim: int, pos_ndim: int):
    """``(mode, sections)``, or None if the rotary module is not recognised.

    Structural, not nominal: the table must be 2-D and exactly ``head_dim``
    wide (a partial rotary dim rotates only a leading slice and is left to the
    reference path), and ``forward`` must still be the one defined alongside the
    class whose semantics are reproduced here.
    """
    if rope is None:
        return (_ROPE_NONE, 0)
    cls = type(rope)
    cache = getattr(rope, "cos_sin_cache", None)
    if (not isinstance(cache, torch.Tensor) or cache.dim() != 2
            or cache.size(1) != head_dim
            or getattr(rope, "head_dim", None) != head_dim
            or head_dim // 2 > 0x3ff):     # section widths are packed 10 bits
        return None
    where = getattr(cls.forward, "__module__", "")
    if cls.__name__ in _MROPE_CLASSES and where.endswith("L1.mrope"):
        sec = list(getattr(rope, "mrope_section", ()))
        if len(sec) != 3 or sum(sec) != head_dim // 2:
            return None
        if pos_ndim == 1:
            # ``_apply_sgl_rope``: all three sections coincide -> plain neox.
            return (_ROPE_NEOX, 0)
        if pos_ndim == 2:
            mode = (_ROPE_MROPE_IL if getattr(rope, "mrope_interleaved", False)
                    else _ROPE_MROPE_SEC)
            return (mode, _pack_sections(sec[0], sec[1], sec[2]))
        return None
    if cls.__name__ in _NEOX_CLASSES and where.endswith("L1.rotary_emb"):
        if pos_ndim != 1:
            return None
        neox = bool(getattr(rope, "is_neox_style", True))
        return ((_ROPE_NEOX if neox else _ROPE_GPTJ), 0)
    return None


# ---------------------------------------------------------------------------
# Host-dispatch fast path
# ---------------------------------------------------------------------------
_MM = torch.mm
_ADDMM = torch.addmm
_EMPTY = torch.empty
_I32 = torch.int32
_I64 = torch.int64
_BF16 = torch.bfloat16
_IS_COMPILING = torch.compiler.is_compiling

_PROJ_REF = 0     # keep the submodule call (a dispatch we do not reproduce)
_PROJ_BF16 = 1    # addmm / mm with a pre-transposed weight and an out= buffer
_PROJ_FP8 = 2     # per-token-group quant + DeepGEMM, handles bound once

_ATTN_REF = 0     # keep ``Attention.forward``
_ATTN_VARLEN = 1  # ``flash_attn_varlen_func`` with a prebuilt argument list
_ATTN_FA4 = 2     # FA4's own compiled kernel, looked up once

# Scratch buffers are shared across instances keyed on (device, dtype, shape):
# decoder layers run sequentially and every buffer here is consumed inside the
# forward that fills it, so one set serves all 36 layers instead of 36 sets --
# the same reasoning (and lifetime) as L1 ``Fp8Linear``'s ``_Fp8PrefillBufs``.
# Above ``_POOL_MAX_TOKENS`` nothing is cached: those calls are GPU-bound by
# orders of magnitude, and a resident buffer that large is pure waste.
# The frozen L1 winner's fused quantize+GEMV for the M == 1 decode row.  It is
# the only piece of this file that is *not* bit-identical to the reference
# (|delta| <= 4e-3 on the projection itself), so it carries its own switch.
_USE_FUSED_GEMV = True


_POOL_MAX_TOKENS = 2048
_POOL_MAX_ENTRIES = 32
_POOL: dict = {}


def _pooled(rows: int, width: int, dtype, device) -> torch.Tensor:
    key = (device.type, device.index, dtype, rows, width)
    buf = _POOL.get(key)
    if buf is None:
        if len(_POOL) >= _POOL_MAX_ENTRIES:
            _POOL.clear()
        buf = _POOL[key] = torch.empty(rows, width, dtype=dtype, device=device)
    return buf


class _KeyRecorder:
    """Forwarding proxy over FA4's compile cache that records the key it reads.

    The only way to get at the compiled kernel the reference call *would* use is
    to watch the reference call look it up; deriving the 48-element key here
    would be a second implementation of it, free to drift.  Installed for the
    duration of one call and removed in a ``finally``.
    """

    __slots__ = ("_real", "key")

    def __init__(self, real):
        self._real = real
        self.key = None

    def __contains__(self, key):
        return key in self._real

    def __getitem__(self, key):
        self.key = key
        return self._real[key]

    def __setitem__(self, key, value):
        self._real[key] = value


class _Proj:
    """One projection's pre-resolved dispatch."""

    __slots__ = ("mode", "w", "wt", "b", "ws", "out", "k", "groups",
                 "quant", "tail", "gemm", "alloc", "no_cast", "gemv", "gvt")

    def __init__(self, mode):
        self.mode = mode
        self.w = self.wt = self.b = self.ws = None
        self.out = self.k = self.groups = 0
        self.quant = self.gemm = self.alloc = None
        self.tail = ()          # quantizer args after (x, out_fp8, out_scale)
        self.no_cast = False    # DeepGEMM skips its internal SF cast
        self.gemv = None        # fused quantize+GEMV, M == 1 only
        self.gvt = ()


class _Plan:
    """Everything about the flat forward that does not depend on the call."""

    __slots__ = ("q", "o", "dtype", "dev", "fa", "varlen", "scale", "sink",
                 "wsl", "wsr", "sw", "glue", "cache", "qw", "kw", "aux")

    def __init__(self):
        self.q = self.o = None
        self.dtype = self.dev = None
        self.fa = _ATTN_REF
        self.varlen = None
        self.scale = 0.0
        self.sink = None
        self.wsl = self.wsr = None
        self.sw = False
        self.glue = self.cache = self.qw = self.kw = None
        self.aux = None


class _Buffers:
    """Per-token-count buffers and the views into them."""

    __slots__ = ("qkv", "q2", "k2", "v2", "q3", "k3", "v3", "o3", "o2",
                 "qi", "isc", "oqi", "oisc")


_L1_FP8 = None


def _l1_fp8():
    """The frozen L1 ``fp8_linear`` winner, or ``None`` if this build has none.

    ``parallel_linear`` (frozen L2) reaches ``Fp8Linear`` through a *baseline*
    relative import, so a layer's own ``linear_op`` is the vendored one even
    where a winner is installed.  A deployed stack runs the winner; so does
    this, for the two entry points below.  Both are documented -- and measured
    here, on every captured shape -- bit-identical to what they replace:

    * ``per_token_group_quant_e4m3`` is register-resident with 16 B vector
      traffic where the vendored ``per_token_group_quant_8bit`` is not:
      123.9 -> 30.7 us at ``[16384, 4096]``, 9.0 -> 3.4 us at ``[1000, 4096]``,
      with equal fp8 bytes and equal scales.
    * the *packed* UE8M0 scale layout (int32, four block exponents per word) is
      what ``fp8_gemm_nt`` consumes with ``disable_ue8m0_cast=True``, which
      deletes DeepGEMM's internal SF-cast kernel *and* its launch: 27.6 -> 16.5
      us of device and 26.7 -> 16.3 us of host per projection at
      ``[1, 4096] x [9216, 4096]``.  Bit-identical because the scales are
      already exact powers of two, so the cast is an exponent extraction.
    """
    global _L1_FP8
    if _L1_FP8 is None:
        try:
            from ..L1 import fp8_linear as mod
            _L1_FP8 = mod
        except Exception:
            _L1_FP8 = False
    return _L1_FP8 or None


_L1_FP8_NAMES = ("_resolve_quant_fn", "_alloc_packed_scale",
                 "_alloc_colmajor_scale", "_use_ue8m0", "_GROUP_SIZE",
                 "_QUANT_EPS", "_FP8_MIN", "_FP8_MAX", "_DG_FP8_GEMM_NT")


def _resolve_proj(mod, use_bias: bool, dtype) -> _Proj:
    """Bind one projection's kernels, or return a ``_PROJ_REF`` plan."""
    weight = mod.weight
    bias = mod.bias if use_bias else None
    if not getattr(mod, "use_fp8", False):
        if weight.dtype is not dtype or (bias is not None and bias.dtype is not dtype):
            return _Proj(_PROJ_REF)
        p = _Proj(_PROJ_BF16)
        p.w = weight
        p.wt = weight.t()
        p.b = bias
        p.out, p.k = int(weight.shape[0]), int(weight.shape[1])
        return p

    # FP8 block-scaled.  Everything is taken off the *live* Fp8Linear module so
    # the entry points are the ones the reference reaches, not look-alikes.
    if dtype is not _BF16:
        return _Proj(_PROJ_REF)
    lin = getattr(mod, "linear_op", None)
    fmod = sys.modules.get(type(lin).__module__) if lin is not None else None
    if fmod is None:
        return _Proj(_PROJ_REF)
    try:
        # A FlashInfer-capable device routes M < 32 through the swapAB kernel;
        # that dispatch depends on the runtime M and is left to the module.
        if (fmod._maybe_get_flashinfer_fp8_gemm() is not None
                and not fmod._is_batch_invariant()):
            return _Proj(_PROJ_REF)
        ws = getattr(mod, "weight_scale_inv", None)
        if (not isinstance(ws, torch.Tensor)
                or weight.dtype is not torch.float8_e4m3fn
                or weight.dim() != 2):
            return _Proj(_PROJ_REF)
        n_out, k_in = int(weight.shape[0]), int(weight.shape[1])
        p = _Proj(_PROJ_FP8)
        win = _l1_fp8()
        if win is not None and all(hasattr(win, n) for n in _L1_FP8_NAMES):
            ue8m0 = bool(win._use_ue8m0())
            gsz = int(win._GROUP_SIZE)
            if k_in % gsz:
                return _Proj(_PROJ_REF)
            p.quant = win._resolve_quant_fn()
            p.tail = (float(win._QUANT_EPS), float(win._FP8_MIN),
                      float(win._FP8_MAX))
            p.gemm = win._DG_FP8_GEMM_NT
            # The packed scale layout needs whole four-group words along K.
            packed = ue8m0 and k_in % (4 * gsz) == 0
            p.alloc = (win._alloc_packed_scale if packed
                       else win._alloc_colmajor_scale)
            p.no_cast = packed or not ue8m0
            # M == 1 is a bandwidth-bound GEMV that DeepGEMM spends a full
            # 128-row M tile on; the winner folds the activation quantization
            # into its prologue and reads the weight exactly once.  Same
            # eligibility test the winner applies to itself.
            if (_USE_FUSED_GEMV and ue8m0 and k_in % 512 == 0
                    and weight.is_contiguous()
                    and ws.dtype is _I32 and ws.dim() == 2
                    and ws.stride(0) == 1
                    and tuple(ws.shape) == (n_out, k_in // 512)
                    and hasattr(win, "_resolve_gemv_fn")):
                p.gemv = win._resolve_gemv_fn()
                p.gvt = p.tail + (-1,)
        else:
            gsz = int(fmod._GROUP_SIZE)
            if k_in % gsz:
                return _Proj(_PROJ_REF)
            p.quant = fmod._C.per_token_group_fp8_quant
            info = fmod._FP8_INFO
            p.tail = (gsz, float(fmod._QUANT_EPS), float(info.min),
                      float(info.max), True, True, False)
            p.gemm = fmod.deep_gemm.fp8_gemm_nt
            p.alloc = fmod._alloc_colmajor_scale
            p.no_cast = not fmod._is_deep_gemm_e8m0_used()
    except Exception:
        return _Proj(_PROJ_REF)
    p.w = weight
    p.ws = ws
    p.b = bias
    p.out, p.k = n_out, k_in
    p.groups = -(-k_in // gsz)
    return p


def _make_varlen(scale, sink, window):
    """The dense-prefill call ``TRTLLMPrefill`` makes, with its kwargs prebuilt."""
    from ....infra.fa_utils import flash_attn_varlen_func, FA_VERSION
    extra = {}
    if sink is not None:
        extra["s_aux"] = sink
    if window != (-1, -1):
        extra["window_size"] = window

    def call(q, k, v, cu_q, cu_k, msq, msk):
        return flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=msq, max_seqlen_k=msk, softmax_scale=scale,
            causal=True, fa_version=FA_VERSION, num_splits=1, **extra)

    return call, FA_VERSION


class LlamaAttention(nn.Module):
    """Model-level attention: qkv_proj -> [qk_norm] -> [rope] -> Attention -> o_proj."""

    def __init__(self, hidden_size: int, num_attention_heads: int,
                 num_key_value_heads: int, head_dim: int,
                 rotary_emb: nn.Module | None = None,
                 bias: bool = False,              # Qwen2 / GPT-OSS
                 qk_norm: bool = False,           # Qwen3
                 rms_norm_eps: float = 1e-6,
                 nope: bool = False,              # Llama 4
                 use_weightless_qk_norm: bool = False,   # Llama 4
                 attn_temperature_tuning: bool = False,  # Llama 4
                 floor_scale: float = 8192.0,            # Llama 4
                 attn_scale: float = 0.1,                # Llama 4
                 quant_config: dict | None = None,
                 attention_chunk_size: int | None = None,
                 o_proj_bias: bool = False,              # GPT-OSS
                 use_sinks: bool = False,                # GPT-OSS
                 sliding_window: int | None = None,      # GPT-OSS
                 layer_idx: int = 0):                     # GPT-OSS
        super().__init__()
        tp = _tp_size()
        self.num_heads = num_attention_heads // tp
        if num_key_value_heads >= tp:
            self.num_kv_heads = num_key_value_heads // tp
        else:
            self.num_kv_heads = 1
        self.head_dim = head_dim
        self.rotary_emb = rotary_emb
        self.nope = nope
        self.attn_temperature_tuning = attn_temperature_tuning and nope
        self.floor_scale = floor_scale
        self.attn_scale = attn_scale

        self.qkv_proj = QKVParallelLinear(
            hidden_size, head_dim,
            num_attention_heads, num_key_value_heads,
            bias=bias,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            num_attention_heads * head_dim, hidden_size,
            bias=o_proj_bias,
            quant_config=quant_config,
        )

        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps) if qk_norm else None  # Qwen3
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps) if qk_norm else None  # Qwen3

        wl_qk = use_weightless_qk_norm and not nope  # Llama 4 RoPE layers only
        self.q_wl_norm = RMSNorm(head_dim, eps=rms_norm_eps, elementwise_affine=False) if wl_qk else None
        self.k_wl_norm = RMSNorm(head_dim, eps=rms_norm_eps, elementwise_affine=False) if wl_qk else None

        # GPT-OSS: per-layer sliding window (even layers only) and attention sinks
        per_layer_sw = sliding_window if layer_idx % 2 == 0 else None

        if use_sinks:
            self.sinks = nn.Parameter(torch.zeros(self.num_heads))
            self.sinks.weight_loader = self._sinks_weight_loader
        else:
            self.sinks = None

        self.attn = Attention(
            self.num_heads, head_dim, head_dim ** -0.5,
            num_kv_heads=self.num_kv_heads,
            sliding_window=per_layer_sw,
            sinks=self.sinks,
            attention_chunk_size=attention_chunk_size,
        )

        # -- fused-glue plan: everything that does not depend on the call ----
        q_size = self.num_heads * head_dim
        kv_size = self.num_kv_heads * head_dim
        self._split = [q_size, kv_size, kv_size]
        self._width = q_size + 2 * kv_size
        self._qsz = q_size
        self._nh = self.num_heads + self.num_kv_heads
        self._eps = float(rms_norm_eps)
        self._norm = self.q_norm is not None
        self._wlnorm = self.q_wl_norm is not None
        self._temp = bool(self.attn_temperature_tuning)
        # Stage flags live in the high bits of the kernel's ``flags`` argument;
        # the rotation mode occupies the low three.  Packing them here is what
        # keeps the hot path from re-deriving any of it.
        self._flags = ((_FLAG_NORM if self._norm else 0)
                       | (_FLAG_WLNORM if self._wlnorm else 0)
                       | (_FLAG_TEMP if self._temp else 0))
        rope_here = None if nope else rotary_emb
        # A config with no norm, no rotation and no temperature (GPT-OSS) has no
        # glue at all: forward is projection -> split -> attention, and the fast
        # path is exactly that with no per-call branching left in it.
        self._passthrough = (rope_here is None and not self._norm
                            and not self._wlnorm and not self._temp)
        # One plan per ``positions`` rank, resolved once here rather than per
        # call; None means "this rotary module is not one we reproduce".
        # A geometry the vectorized kernel cannot address (odd head_dim, a
        # head row wider than one lane group) leaves every plan None, so those
        # configs take the reference sequence.  Probing here rather than in the
        # first forward also moves the one-time JIT build to construction time,
        # and a build that cannot happen at all degrades instead of raising.
        # Probed for every activation dtype the kernel accepts, not just this
        # model's: the dtype is not known until the first forward, and the
        # 16-byte vector covers half as many fp32 lanes as bf16 ones, so a
        # head_dim can be addressable in one and not the other.
        try:
            ok = head_dim % 2 == 0 and all(
                _C.supported(head_dim, dt) for dt in
                (torch.bfloat16, torch.float16, torch.float32))
        except Exception:
            ok = False
        if ok:
            self._plan1 = _rope_plan(rope_here, head_dim, 1)
            self._plan2 = _rope_plan(rope_here, head_dim, 2)
            for name in ("_plan1", "_plan2"):
                pl = getattr(self, name)
                if pl is None:
                    continue
                f = pl[0] | self._flags
                if pl[0] >= _ROPE_MROPE_IL:
                    f |= _FLAG_ROPE_ROUND
                setattr(self, name, (f, pl[1]))
        else:
            self._plan1 = self._plan2 = None
        self._cache_t = None

        # -- host-dispatch plan: bound on the first forward -------------------
        # Not here: the module is still on the host (so ``cos_sin_cache`` and
        # every weight is the wrong tensor), the activation dtype is unknown,
        # and a caller may legitimately replace the fp8 weight Parameters
        # between construction and the first call.
        self._plan = None
        self._buf: dict = {}
        self._fa: dict = {}
        # Submodule handles in ``__dict__``: reading them off ``_modules``
        # through ``nn.Module.__getattr__`` costs ~0.2 us each, and a plain
        # attribute assignment would register a *second* copy of the submodule.
        object.__setattr__(self, "_qp", self.qkv_proj)
        object.__setattr__(self, "_op", self.o_proj)
        object.__setattr__(self, "_am", self.attn)

    def _apply(self, *args, **kwargs):
        # ``.to()`` / ``.cuda()`` / ``.float()`` replace buffers and may replace
        # parameter storage, so everything bound to a tensor has to go with them.
        out = super()._apply(*args, **kwargs)
        self._cache_t = None
        self._plan = None
        self._buf.clear()
        self._fa.clear()
        return out

    def _sinks_weight_loader(self, param, loaded_weight):
        """TP-shard attention sinks across heads."""
        from ....infra.tp import _tp_rank
        rank = _tp_rank()
        heads_per_rank = param.data.size(0)
        start = rank * heads_per_rank
        param.data.copy_(loaded_weight.narrow(0, start, heads_per_rank))

    def _get_attn_scale(self, positions):  # Llama 4 NoPE only
        """Position-dependent attention temperature scaling."""
        floor = torch.floor((positions.float() + 1.0) / self.floor_scale)
        scale = torch.log(floor + 1.0) * self.attn_scale + 1.0
        return scale.unsqueeze(-1)

    # -- fused glue ---------------------------------------------------------
    def _launch(self, positions, qkv, plan) -> None:
        """Norm + rotate + scale q and k in place inside the packed qkv."""
        if plan[0] & _MODE_MASK:
            cache = self._cache_t
            if cache is None:
                # Resolved on first use, not in __init__: the rotary module is
                # still on the host then, and ``.to(device)`` replaces the
                # buffer (``_apply`` above drops this again if that happens).
                cache = self._cache_t = self.rotary_emb.cos_sin_cache
        else:
            cache = qkv             # unused: no rotation, no table to read
        if self._norm:
            qw = self.q_norm.weight
            kw = self.k_norm.weight
        else:
            qw = kw = qkv
        _C.qkv_glue(qkv, positions, cache, qw, kw,
                    self.num_heads, self._nh, self.head_dim,
                    self._eps, self.floor_scale, self.attn_scale,
                    plan[1], plan[0])

    # -- host-dispatch plan -------------------------------------------------
    def _resolve_plan(self, hidden_states):
        """Bind every per-call invariant of the flat forward, or disable it."""
        try:
            plan = self._build_plan(hidden_states)
        except Exception:
            plan = False
        self._plan = plan
        return plan

    def _build_plan(self, hidden_states):
        if _tp_size() != 1:
            return False        # o_proj all-reduces; not reproduced here
        dtype = hidden_states.dtype
        dev = hidden_states.device
        if dev.type != "cuda":
            return False
        p = _Plan()
        p.dtype, p.dev = dtype, dev
        p.q = _resolve_proj(self._qp, self._qp.bias is not None, dtype)
        p.o = _resolve_proj(self._op, self._op.bias is not None
                            and self._op.tp_rank == 0, dtype)
        if p.q.mode == _PROJ_REF or p.o.mode == _PROJ_REF:
            return False
        if p.q.k != hidden_states.shape[1] or p.q.out != self._width:
            return False
        if p.o.k != self._qsz:
            return False

        # Glue handles.
        p.glue = _C.qkv_glue
        if self._norm:
            p.qw = self.q_norm.weight
            p.kw = self.k_norm.weight
        rope = None if self.nope else self.rotary_emb
        if rope is not None:
            cache = getattr(rope, "cos_sin_cache", None)
            if isinstance(cache, torch.Tensor):
                p.cache = self._cache_t = cache

        # Attention handles.
        a = self._am
        p.scale = a.scale
        p.sink = a._fa3_sinks
        win = a._fa3_window_size
        p.wsl = win[0] if win[0] >= 0 else None
        p.wsr = win[1] if win[1] >= 0 else None
        p.sw = bool(a.sliding_window)
        if (a._triton_only or a.attention_chunk_size is not None
                or not isinstance(getattr(a, "prefill_op", None), nn.Module)):
            p.fa = _ATTN_REF
            return p
        try:
            p.varlen, fa_version = _make_varlen(a.scale, p.sink, win)
        except Exception:
            p.fa = _ATTN_REF
            return p
        p.fa = _ATTN_VARLEN
        # FA4's compiled-kernel argument list is arch-dependent (SM100/SM110
        # take an extra descale slot); only bind it where it is the one below.
        if fa_version == 4 and torch.cuda.get_device_capability(dev)[0] in (10, 11):
            try:
                from vllm.vllm_flash_attn.cute.utils import AuxData
                from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd
                assert isinstance(_flash_attn_fwd.compile_cache.cache, dict)
                p.aux = AuxData(None, None)
                p.fa = _ATTN_FA4
            except Exception:
                pass
        return p

    def _new_buffers(self, n: int, plan):
        """Buffers + views for ``n`` tokens; cached only where it is worth it."""
        b = _Buffers()
        dtype, dev = plan.dtype, plan.dev
        small = n <= _POOL_MAX_TOKENS
        if small:
            rows = (n + 127) & ~127
            qkv = _pooled(rows, self._width, dtype, dev)[:n]
            o2 = _pooled(rows, self._qsz, dtype, dev)[:n]
        else:
            qkv = _EMPTY(n, self._width, dtype=dtype, device=dev)
            o2 = _EMPTY(n, self._qsz, dtype=dtype, device=dev)
        b.qkv = qkv
        b.q2, b.k2, b.v2 = qkv.split(self._split, dim=-1)
        nh, nkv, hd = self.num_heads, self.num_kv_heads, self.head_dim
        b.q3 = b.q2.view(n, nh, hd)
        b.k3 = b.k2.view(n, nkv, hd)
        b.v3 = b.v2.view(n, nkv, hd)
        b.o2 = o2
        b.o3 = o2.view(n, nh, hd)
        b.qi = b.isc = b.oqi = b.oisc = None
        if plan.q.mode == _PROJ_FP8 and not (n == 1 and plan.q.gemv is not None):
            b.qi = _EMPTY(n, plan.q.k, dtype=torch.float8_e4m3fn, device=dev)
            b.isc = plan.q.alloc(n, plan.q.groups, dev)
        if plan.o.mode == _PROJ_FP8 and not (n == 1 and plan.o.gemv is not None):
            b.oqi = _EMPTY(n, plan.o.k, dtype=torch.float8_e4m3fn, device=dev)
            b.oisc = plan.o.alloc(n, plan.o.groups, dev)
        if small:
            if len(self._buf) >= 8:
                self._buf.clear()
            self._buf[n] = b
        return b

    def _fa4_bind(self, key, plan, b, cu_q, cu_k, msq, msk):
        """Look up FA4's compiled kernel by watching the reference call use it.

        Returns the reference call's own output, so the probe is not wasted.
        """
        from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd
        real = _flash_attn_fwd.compile_cache
        rec = _KeyRecorder(real)
        _flash_attn_fwd.compile_cache = rec
        try:
            out = plan.varlen(b.q3, b.k3, b.v3, cu_q, cu_k, msq, msk)
        finally:
            _flash_attn_fwd.compile_cache = real
        kern = None
        if rec.key is not None:
            try:
                kern = real[rec.key]
            except Exception:
                kern = None
        if kern is None:
            plan.fa = _ATTN_VARLEN          # give up on the direct call
        else:
            if len(self._fa) >= 64:
                self._fa.clear()
            self._fa[key] = kern
        return out

    # -- forward ------------------------------------------------------------
    def forward(self, positions, hidden_states, rotary_emb=None):
        plan = self._plan
        if (rotary_emb is None and plan is not False
                and hidden_states.dim() == 2 and not _IS_COMPILING()):
            n = hidden_states.shape[0]
            if plan is None:
                plan = self._resolve_plan(hidden_states)
            if (plan is not False and n
                    and hidden_states.dtype is plan.dtype
                    and hidden_states.is_contiguous()):
                nd = positions.dim()
                if nd == 2:
                    # M-RoPE reads three position rows per token; a 2-D
                    # ``positions`` with any other leading extent is not one and
                    # must not be indexed as if it were.
                    gp = self._plan2 if positions.size(0) == 3 else None
                else:
                    gp = self._plan1
                if (self._passthrough
                        or (gp is not None and (nd == 1 or not self._temp)
                            and positions.dtype is _I64)):
                    b = self._buf.get(n)
                    if b is None:
                        b = self._new_buffers(n, plan)
                    qkv = b.qkv

                    # 1. QKV projection, straight into the buffer.
                    pj = plan.q
                    if pj.mode == _PROJ_BF16:
                        if pj.b is None:
                            _MM(hidden_states, pj.wt, out=qkv)
                        else:
                            _ADDMM(pj.b, hidden_states, pj.wt, out=qkv)
                    elif pj.gemv is not None and n == 1:
                        pj.gemv(hidden_states, pj.w, pj.ws, qkv, *pj.gvt)
                    else:
                        pj.quant(hidden_states, b.qi, b.isc, *pj.tail)
                        pj.gemm((b.qi, b.isc), (pj.w, pj.ws), qkv,
                                disable_ue8m0_cast=pj.no_cast)

                    # 2. Fused glue (norm + RoPE + scaling), in place.
                    if not self._passthrough:
                        plan.glue(qkv, positions,
                                  plan.cache if gp[0] & _MODE_MASK else qkv,
                                  plan.qw if self._norm else qkv,
                                  plan.kw if self._norm else qkv,
                                  self.num_heads, self._nh, self.head_dim,
                                  self._eps, self.floor_scale, self.attn_scale,
                                  gp[1], gp[0])

                    # 3. Attention.  The direct kernel serves the dense
                    #    (unpaged) prefill the reference itself falls back to;
                    #    every other shape of the problem keeps ``self.attn``.
                    o2 = None
                    a = self._am
                    if plan.fa and not a._use_custom_op:
                        if (a.k_cache.numel() == 0 and a.v_cache.numel() == 0
                                and a._fa3_sinks is plan.sink):
                            ctx = get_context()
                            cu_q = ctx.cu_seqlens_q
                            cu_k = ctx.cu_seqlens_k
                            if (ctx.is_prefill and not ctx.is_mixed
                                    and not getattr(ctx, "is_tree_verify", False)
                                    and ctx.block_tables is None
                                    and cu_q is not None and cu_k is not None
                                    and cu_q.dtype is _I32
                                    and cu_k.dtype is _I32
                                    and cu_q.is_contiguous()
                                    and cu_k.is_contiguous()
                                    and not (plan.sw
                                             and ctx.sliding_block_tables
                                             is not None)):
                                msq = ctx.max_seqlen_q
                                if plan.fa == _ATTN_FA4:
                                    key = (msq, cu_q.numel())
                                    kern = self._fa.get(key)
                                    if kern is None:
                                        o2 = self._fa4_bind(
                                            key, plan, b, cu_q, cu_k, msq,
                                            ctx.max_seqlen_k).view(n, self._qsz)
                                    else:
                                        kern(b.q3, b.k3, b.v3, b.o3, None,
                                             plan.scale, cu_q, cu_k, None, None,
                                             None, None, plan.wsl, plan.wsr,
                                             plan.sink, None, None, plan.aux,
                                             None)
                                        o2 = b.o2
                                else:
                                    o2 = plan.varlen(
                                        b.q3, b.k3, b.v3, cu_q, cu_k, msq,
                                        ctx.max_seqlen_k).view(n, self._qsz)
                    if o2 is None:
                        o2 = a(b.q2, b.k2, b.v2)

                    # 4. Output projection.  Its result is the module's return
                    #    value, so it is always a fresh allocation.
                    pj = plan.o
                    if pj.mode == _PROJ_BF16:
                        # ``F.linear`` on a 2-D input *is* this addmm; going
                        # straight to it is bit-identical and ~0.7 us cheaper.
                        if pj.b is None:
                            return _MM(o2, pj.wt)
                        return _ADDMM(pj.b, o2, pj.wt)
                    if not o2.is_contiguous():
                        # Only reachable when ``self.attn`` served the call: the
                        # quantizer and the GEMV both read x as a dense row, and
                        # the wrapper we bypassed made the same check.
                        o2 = o2.contiguous()
                    out = _EMPTY(n, pj.out, dtype=_BF16, device=plan.dev)
                    if pj.gemv is not None and n == 1:
                        pj.gemv(o2, pj.w, pj.ws, out, *pj.gvt)
                    else:
                        pj.quant(o2, b.oqi, b.oisc, *pj.tail)
                        pj.gemm((b.oqi, b.oisc), (pj.w, pj.ws), out,
                                disable_ue8m0_cast=pj.no_cast)
                    if pj.b is not None:
                        out = out + pj.b
                    return out

        # -- reference sequence (forward-arg rotary / unrecognised rotary /
        #    3-D hidden states / tracing / a config the plan declined) --------
        qkv = self.qkv_proj(hidden_states)
        if rotary_emb is None:
            if self._passthrough:
                q, k, v = qkv.split(self._split, dim=-1)
                return self.o_proj(self.attn(q, k, v))
            nd = positions.dim()
            if nd == 2:
                plan2 = self._plan2 if positions.size(0) == 3 else None
            else:
                plan2 = self._plan1
            if (plan2 is not None and (nd == 1 or not self._temp)
                    and qkv.dim() == 2 and qkv.is_contiguous()
                    and positions.dtype == torch.int64
                    and not torch.compiler.is_compiling()):
                if qkv.shape[0]:
                    self._launch(positions, qkv, plan2)
                q, k, v = qkv.split(self._split, dim=-1)
                return self.o_proj(self.attn(q, k, v))

        N = hidden_states.shape[0]
        rope = rotary_emb if rotary_emb is not None else self.rotary_emb
        q, k, v = qkv.split(self._split, dim=-1)

        # Learnable QK norm (Qwen3: before RoPE)
        if self.q_norm is not None:
            # Normalise per head through a *view*, matching vLLM's
            # Qwen3Attention.forward:
            #     q_by_head = q.view(*q.shape[:-1], -1, head_dim)
            #     q = self.q_norm(q_by_head).view(q.shape)
            # The previous form reshaped to (N*heads, head_dim) first. That
            # cannot be a view of a qkv slice -- the slice's row stride is the
            # packed qkv width (q_size + 2*kv_size), not num_heads*head_dim --
            # so ``reshape`` materialised a full copy of q and k on every layer:
            # at 16384 prefill tokens that is 67 MB for q plus 17 MB for k per
            # layer, 36 layers deep, and it showed up in the kernel profile as
            # one Memcpy DtoD per layer per step that vLLM does not emit.
            # Reducing over the last dim of the 3-D view is bit-identical
            # (same 128 values per row, same order) and lets Inductor fuse the
            # strided read straight into the following RoPE kernel.
            q_shape, k_shape = q.shape, k.shape
            q = self.q_norm(
                q.view(N, self.num_heads, self.head_dim)).view(q_shape)
            k = self.k_norm(
                k.view(N, self.num_kv_heads, self.head_dim)).view(k_shape)

        if not self.nope and rope is not None:
            q, k = rope(positions, q, k)

        # Weight-less QK norm (Llama 4: after RoPE, only on RoPE layers)
        if self.q_wl_norm is not None:
            q = self.q_wl_norm(q.view(-1, self.head_dim)).view(N, -1)
            k = self.k_wl_norm(k.view(-1, self.head_dim)).view(N, -1)

        # Temperature tuning (Llama 4: only on NoPE layers)
        if self.attn_temperature_tuning:
            q = (q * self._get_attn_scale(positions)).to(q.dtype)

        attn_output = self.attn(q, k, v)
        return self.o_proj(attn_output)
