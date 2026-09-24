"""Decoder layer: attention + MLP with RMSNorm residual connections.

Unified across Llama, Qwen2, and Qwen3 architectures:
  - bias:    Qwen2 uses bias=True on QKV projection.
  - qk_norm: Qwen3 applies per-head RMSNorm to Q and K before RoPE.

Where the time goes
-------------------
Every captured shape is one Llama-3.1-8B layer: hidden 4096, 32 q / 8 kv heads
of 128, intermediate 14336.  Its four projections hold 436 MB of bf16 weight
(``qkv`` 50, ``o`` 34, ``gate_up`` 235, ``down`` 117) and all of it crosses HBM
once per call whatever the token count is.  Under the scorer's regime -- one L2
flush per iteration, whose write-back overlaps the start of the timed region --
a pure ``uint4`` read of 436 MB takes 95 us, and cuBLAS runs the four
projections back to back in 104-121 us (M = 1..60).  Four kernels of unavoidable
ramp-up account for the difference: **the GEMMs are already at the roofline, and
replacing them loses.**  (Measured: a packed-weight Triton kernel is 0-12 us
slower per projection, and at 279 tokens cuBLAS is compute bound instead, where
a ``tl.dot`` pipeline is ~2x off.)

So for the short sequences -- 1, 26, 60 and 279 tokens, 4 of the 5 scored shapes
and >90% of the captured calls -- everything to win is *around* the GEMMs.  The
baseline spends ~45 us of GPU time and ~60 us of launch gap there:

* ``RotaryEmbedding.forward_cuda`` casts its fp32 ``cos_sin_cache`` to bf16 on
  **every call** and throws the result away.  At the captured
  ``max_position_embeddings=131072`` that is a 67 MB read plus a 34 MB write --
  12.5 us per call, 6% of a decode step, for a value that never changes.  A
  private bf16 copy is built once in ``_build_plan``; the (shared) rotary
  module is left alone.
* ``silu_and_mul`` is a separate kernel over the ``[M, 2I]`` projection output.
  Folding it into a ``gate_up`` epilogue removes that kernel and the
  un-activated tensor's round trip; the packed-weight Triton GEMM underneath it
  matches cuBLAS on this shape, so the activation comes for free (measured 4 us
  at 60 tokens, 12 us at 26).  Above one M tile the packed weight would be
  re-streamed per tile, so past 64 tokens cuBLAS keeps the GEMM and only the
  activation is replaced, by a flat kernel whose occupancy does not depend on I.
* The reference RoPE kernel gathers cos/sin per token and costs 5.3 us for ~1 MB
  of traffic; one program per (token, 4 heads) over the fused QKV buffer is ~3.
* At one token the attention is *exactly* V broadcast over each GQA group (a
  single key, so the softmax weight is 1.0), which skips a 12 us kernel.
* The layer runs ~20 eager ops and at these sizes the **host** is the critical
  path: the baseline leaves ~60 us of GPU idle between kernels.  The fast path is
  8 GPU ops behind ~12 Python statements, with the scratch buffers, the
  transposed weight views, the ``cu_seqlens`` and the Triton launcher all
  resolved once per shape instead of per call.  That is the single biggest term.
  (Capturing the whole path into a CUDA graph was then measured *slower* -- the
  remaining host work already fits under the ~130 us of GPU work, so a graph only
  adds its three input copies.)

Long sequences (16384 tokens) are compute bound -- cuBLAS sustains ~1.3 PFLOP/s
on the projections -- so they keep the frozen L1/L2 op sequence unchanged.

Net, on the five scored shapes (1, 16384, 279, 60, 26 tokens): 1.59x, 1.06x,
1.14x, 1.48x, 1.49x -- geomean 1.33x.  Two caveats on those numbers.  They move
with the GPU's clock state, which on this box drifts between 1155 and 1965 MHz
independently of what this process does: an otherwise identical run of the same
code measured 1.28x / 1.06x / 1.11x / 1.17x / 1.05x.  And the direction of that
drift is not symmetric -- the baseline spends ~30% of its wall time with the GPU
idle between launches, which does not scale with the SM clock, while this path is
almost entirely GPU work, so a slow clock compresses the margin.  Comparisons
between two candidate variants are only meaningful inside a single process
(``dev/ab.py``), where they reproduce to ~0.1 us.

Numerics
--------
This layer is chaotic in its last bit: perturbing a *single* one of the 245k
elements of the first norm's output by one bf16 ulp moves 1.5% of the layer's
output past the scorer's tolerance (29 elements moves 3.5%).  So every kernel
upstream of the MLP has to be bit-exact with the reference, not merely accurate
-- a hand-written attention that is *more* accurate than FlashAttention fails,
and so does an otherwise correct RMSNorm whose variance reduction order differs.

Accordingly the fast path reproduces the reference exactly up to the MLP:

* the projections stay on cuBLAS (``torch.mm`` into a cached buffer is the same
  call ``F.linear`` makes), and attention stays on the same FlashAttention
  invocation the reference's ``TRTLLMPrefill`` dense fallback makes;
* ``fused_add_rmsnorm`` and ``rope`` in ``llama_decoder_fk.cu`` clone the
  reference kernels' arithmetic -- see that file for what has to match;
* only the MLP epilogue differs, and only in the last bf16 ulp: its error is
  terminal (nothing downstream re-rounds it), which is what makes it the one
  place a rewritten kernel is safe.  The fused epilogue still reproduces vLLM's
  *double* rounding, ``bf16(bf16(silu(bf16(gate))) * bf16(up))`` -- computing
  ``silu(g)*u`` in one fp32 expression is more accurate but disagrees in 27% of
  elements by one ulp, which ``down`` then amplifies past tolerance.

Anything this path does not reproduce -- FP8 or biased projections, QK-norm,
NoPE, sliding windows, attention sinks, a populated paged KV cache, a
non-NeoX/scaled rotary, tensor parallelism, more than ``_FAST_MAX_M`` tokens --
falls back to the frozen L1/L2 op sequence (see ``_build_plan``).
"""


from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ....infra.context import get_context
from ....infra.cuda_ext import load_op
from ....infra.fa_utils import FA_VERSION as _FA_VERSION
from ....infra.fa_utils import flash_attn_varlen_func as _flash_attn_varlen
from ..L1.rms_norm import RMSNorm
from ..L2.attention import LlamaAttention
from ..L2.llama_mlp import LlamaMLP

_C = load_op("llama_decoder_fk", "llama_decoder_fk.cu")

# Above this many tokens the fast path stops paying for itself: the fixed costs
# it removes (the rotary cache cast, the activation kernel, the host-side launch
# gap) are a rounding error next to a compute-bound projection, while its cached
# scratch would grow to gigabytes.  Longer sequences keep the frozen op sequence,
# which measures the same there (8528 us vs 8528 us at 16384 tokens).
_FAST_MAX_M = 320

# The reference attention kernel has to be *the* reference kernel: this layer is
# chaotic in its last bit (perturbing a single one of the 245k elements of the
# first norm's output by one bf16 ulp moves 1.5% of the layer's output past the
# scorer's tolerance), so a hand-written attention -- even a more accurate one --
# cannot be substituted.  Everything upstream of the MLP is therefore
# bit-exact with the reference; only the MLP epilogue, whose error is terminal,
# is allowed to differ.
# (Capturing the fast path into a CUDA graph was measured *slower*: with the
# scratch, the transposed weight views and the Triton launchers all resolved once
# per shape, the remaining host work already fits under the ~130 us of GPU work,
# so a graph only adds the three input copies -- 3-8 us per call.)


# ---------------------------------------------------------------------------
# gate_up + SiLU-and-mul, one kernel.
#
# ``P`` is the [2I, H] gate_up weight packed to [I/BN, H/BK, 2, BK, BN]: the
# gate tile, the up tile and the next k-step sit back to back, so a block's
# whole k-loop is one linear read of HBM instead of BK scattered runs of
# ``2*BK`` bytes.
# ---------------------------------------------------------------------------
@triton.jit
def _gate_up_act_kernel(X, P, A, M, I,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                        NK: tl.constexpr, K: tl.constexpr,
                        MASK_M: tl.constexpr, MASK_N: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rk = tl.arange(0, BK)
    rn = tl.arange(0, BN)
    xp = X + rm[:, None] * K + rk[None, :]
    wp = P + pid_n * (NK * 2 * BK * BN) + rk[:, None] * BN + rn[None, :]
    mm = rm[:, None] < M
    accg = tl.zeros((BM, BN), dtype=tl.float32)
    accu = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(NK):
        g = tl.load(wp, eviction_policy="evict_first")
        u = tl.load(wp + BK * BN, eviction_policy="evict_first")
        if MASK_M:
            a = tl.load(xp, mask=mm, other=0.0, eviction_policy="evict_last")
        else:
            a = tl.load(xp, eviction_policy="evict_last")
        accg = tl.dot(a, g, accg)
        accu = tl.dot(a, u, accu)
        wp += 2 * BK * BN
        xp += BK
    ety = A.dtype.element_ty
    # The reference stores gate_up in ``ety`` and rounds silu(gate) again before
    # the multiply; both roundings are reproduced.
    gate = accg.to(ety).to(tl.float32)
    up = accu.to(ety).to(tl.float32)
    act = (gate * tl.sigmoid(gate)).to(ety).to(tl.float32)
    o = (act * up).to(ety)
    cn = pid_n * BN + rn
    ap = A + rm[:, None] * I + cn[None, :]
    if MASK_M and MASK_N:
        tl.store(ap, o, mask=mm & (cn[None, :] < I), eviction_policy="evict_last")
    elif MASK_M:
        tl.store(ap, o, mask=mm, eviction_policy="evict_last")
    elif MASK_N:
        tl.store(ap, o, mask=cn[None, :] < I, eviction_policy="evict_last")
    else:
        tl.store(ap, o, eviction_policy="evict_last")


# ---------------------------------------------------------------------------
# Standalone SiLU-and-mul for the large-M plan: flat over the [M, I] output, so
# block size and bytes-in-flight are independent of I (the reference kernel
# launches one block of ``I/vec`` threads per token, so at few tokens it runs on
# a handful of SMs).  Same double rounding as the fused epilogue.
# ---------------------------------------------------------------------------
@triton.jit
def _act_kernel(Y, A, D, N, BLK: tl.constexpr, VPT: tl.constexpr,
                MASKED: tl.constexpr):
    pid = tl.program_id(0)
    ety = A.dtype.element_ty
    for v in tl.range(VPT):
        off = (pid * VPT + v) * BLK + tl.arange(0, BLK)
        row = off // D
        gp = Y + row * D + off              # row*2D + (off - row*D)
        if MASKED:
            m = off < N
            g = tl.load(gp, mask=m, other=0.0).to(tl.float32)
            u = tl.load(gp + D, mask=m, other=0.0).to(tl.float32)
            act = (g * tl.sigmoid(g)).to(ety).to(tl.float32)
            tl.store(A + off, (act * u).to(ety), mask=m)
        else:
            g = tl.load(gp).to(tl.float32)
            u = tl.load(gp + D).to(tl.float32)
            act = (g * tl.sigmoid(g)).to(ety).to(tl.float32)
            tl.store(A + off, (act * u).to(ety))


# (BLK, VPT) by output size: bytes in flight per block is the knob that matters.
_ACT_TIERS = ((1 << 24, 2048, 2), (1 << 20, 1024, 2), (0, 512, 1))


def _act_launch(n):
    for lo, blk, vpt in _ACT_TIERS:
        if n >= lo:
            break
    return blk, vpt, -(-n // (blk * vpt)), (n % (blk * vpt)) != 0


# ---------------------------------------------------------------------------
# Launch helpers
# ---------------------------------------------------------------------------
class _Launch:
    """One pre-compiled Triton kernel, fixed grid and constexprs.

    ``JITFunction.__getitem__`` re-derives the cache key and re-binds every
    argument on each call (~13 us of Python, measured).  At these sizes the CPU
    is the critical path, so resolving the ``CompiledKernel`` once and calling
    its launcher directly matters; falls back to normal dispatch if the private
    API moves.
    """

    __slots__ = ("_fn", "_grid", "_const", "_nw", "_ns", "_run")

    def __init__(self, fn, grid, runtime_args, const_args, num_warps, num_stages):
        self._fn = fn
        self._grid = grid
        self._const = tuple(const_args)
        self._nw, self._ns = num_warps, num_stages
        self._run = None
        try:
            ck = fn.warmup(*runtime_args, *self._const, grid=grid,
                           num_warps=num_warps, num_stages=num_stages)
            ck._init_handles()
            self._run = ck[grid]
        except Exception:
            self._run = None

    def __call__(self, *runtime_args):
        if self._run is not None:
            self._run(*runtime_args, *self._const)
        else:
            self._fn[self._grid](*runtime_args, *self._const,
                                 num_warps=self._nw, num_stages=self._ns)


def _pack_gate_up(w, BN, BK):
    """[2I, H] -> flat [I/BN, H/BK, 2, BK, BN]; zero padded if not divisible."""
    two_i, H = w.shape
    I = two_i // 2
    nb, nk = -(-I // BN), -(-H // BK)
    if I % BN or H % BK:
        src = torch.zeros(2, nb * BN, nk * BK, device=w.device, dtype=w.dtype)
        src[:, :I, :H] = w.reshape(2, I, H)
    else:
        src = w.reshape(2, I, H)
    return src.view(2, nb, BN, nk, BK).permute(1, 3, 0, 4, 2).contiguous().view(-1)


# (BN, BK, num_warps, num_stages) for the fused gate_up kernel, by hidden size.
# Swept on the scorer's own timer (L2 flush + shifting input pool) over the
# captured shapes; the kernel is at its streaming floor across most of that
# space, so these are simply the entries that ranked first.
_GU_CFG = {4096: (64, 64, 8, 4)}

# Above this many rows the fused kernel's packed weight would be streamed
# once per M tile, so the large-M plan (cuBLAS GEMM + flat activation) wins.
_GU_FUSED_MAX_M = 64

# Distinct token counts whose scratch is kept alive before the cache is
# dropped (a serving engine can see arbitrarily many).
_MAX_CACHED_SHAPES = 32
_GU_CFG_DEFAULT = (64, 64, 4, 4)

class _Plan:
    """Everything the fast path needs, resolved once per layer."""

    __slots__ = ("wt_qkv", "wt_o", "wt_d", "wt_gu", "w_gu", "gu_cfg",
                 "packed_gu", "cos_sin", "eps_in",
                 "eps_post", "w_in", "w_post", "nh", "kvh", "hd", "q_size",
                 "kv_size", "inter", "hidden", "scale", "rot_heads", "sig",
                 "gu_param", "attn",
                 "bufs", "gu_launch", "act_launch", "cu")

    def __init__(self):
        self.bufs = {}
        self.gu_launch = {}
        self.act_launch = {}
        self.cu = {}


_ROPE_1D = "RotaryEmbedding"


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, rotary_emb: nn.Module | None = None,
                 bias: bool = False, qk_norm: bool = False,
                 quant_config: dict | None = None):
        super().__init__()
        self.self_attn = LlamaAttention(
            config.hidden_size, config.num_attention_heads,
            config.num_key_value_heads, config.head_dim,
            rotary_emb=rotary_emb,
            bias=bias, qk_norm=qk_norm,
            rms_norm_eps=config.rms_norm_eps,
            quant_config=quant_config,
        )
        self.mlp = LlamaMLP(config, quant_config=quant_config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._plan: _Plan | None = None
        self._planned = False

    # -- one-time resolution -------------------------------------------------
    def _build_plan(self) -> _Plan | None:
        """Resolve weights / shapes / launchers, or ``None`` for the slow path.

        Deferred out of ``__init__`` because the harness replaces and casts the
        weight Parameters *after* construction, so nothing branched on here is
        final yet.
        """
        at = self.self_attn
        mlp = self.mlp
        if at.q_norm is not None or at.q_wl_norm is not None or at.nope:
            return None
        if at.attn_temperature_tuning or at.sinks is not None:
            return None
        qkv, o = at.qkv_proj, at.o_proj
        gu, dn = mlp.gate_up_proj, mlp.down_proj
        if qkv.use_fp8 or o.use_fp8 or gu.use_fp8 or dn.use_fp8:
            return None
        if any(lin.bias is not None for lin in (qkv, o, gu, dn)):
            return None
        if dn.tp_size > 1 or getattr(gu, "disable_tp", False):
            return None
        ws = [qkv.weight, o.weight, gu.weight, dn.weight,
              self.input_layernorm.weight, self.post_attention_layernorm.weight]
        if any(w.dtype is not torch.bfloat16 or not w.is_contiguous() for w in ws):
            return None
        attn = at.attn
        if (attn.attention_chunk_size is not None or attn.sliding_window
                or attn._fa3_sinks is not None or attn._use_custom_op
                or getattr(attn, "_sliding_group_id", None) is not None
                or attn.k_cache.numel()):
            return None
        rope = at.rotary_emb
        if rope is None or type(rope).__name__ != _ROPE_1D or not rope.is_neox_style:
            return None
        cache = getattr(rope, "cos_sin_cache", None)
        hd = at.head_dim
        if (not isinstance(cache, torch.Tensor) or cache.dim() != 2
                or cache.size(1) != hd or hd not in (64, 128)
                or getattr(rope, "head_dim", None) != hd):
            return None
        if self.input_layernorm.eps != self.post_attention_layernorm.eps:
            return None
        # ``fused_add_rmsnorm`` keeps the row in registers, up to 8 16-byte
        # chunks per thread over a >= 256-thread block.
        if int(qkv.weight.shape[1]) > 8 * 8 * 256:
            return None

        p = _Plan()
        p.w_in = self.input_layernorm.weight.detach()
        p.w_post = self.post_attention_layernorm.weight.detach()
        p.eps_in = float(self.input_layernorm.eps)
        p.eps_post = float(self.post_attention_layernorm.eps)
        if at.num_heads % at.num_kv_heads:
            return None
        p.nh, p.kvh, p.hd = at.num_heads, at.num_kv_heads, hd
        p.q_size = p.nh * hd
        p.kv_size = p.kvh * hd
        p.hidden = int(qkv.weight.shape[1])
        p.inter = int(gu.weight.shape[0]) // 2
        p.scale = attn.scale
        # The engine hands a layer its paged KV cache *after* construction, so
        # whether there is one to store into has to be re-checked per call
        # rather than resolved here.
        p.attn = attn
        p.rot_heads = p.nh + p.kvh
        # Transposed views resolved once: ``torch.mm(x, wt, out=buf)`` is the
        # same cuBLAS call ``F.linear`` makes, minus a per-call ``.t()``.
        p.wt_qkv = qkv.weight.detach().t()
        p.wt_o = o.weight.detach().t()
        p.wt_d = dn.weight.detach().t()
        p.wt_gu = gu.weight.detach().t()
        bn, bk, nw, ns = _GU_CFG.get(p.hidden, _GU_CFG_DEFAULT)
        while bk > 16 and p.hidden % bk:
            bk //= 2
        while bn > 16 and p.inter % bn:
            bn //= 2
        if p.hidden % bk:
            return None
        p.gu_cfg = (bn, bk, nw, ns)
        p.w_gu = gu.weight.detach()
        p.packed_gu = None
        # The reference casts the fp32 cos/sin table to the activation dtype on
        # every call and throws it away (67 MB read + 34 MB write, 12.5 us);
        # keep a private bf16 copy instead and leave the shared rotary alone.
        p.cos_sin = cache if cache.dtype is torch.bfloat16 else cache.to(torch.bfloat16)
        if not p.cos_sin.is_contiguous():
            p.cos_sin = p.cos_sin.contiguous()
        # Cheap staleness guard: a reloaded checkpoint either replaces
        # ``weight.data`` (new data_ptr) or copies into it (bumped version),
        # and the transposed views / packed copy above would go stale.
        p.gu_param = gu.weight
        p.sig = (gu.weight.data_ptr(), gu.weight._version)
        return p

    def _bufs(self, M: int, p: _Plan, device):
        """Cached scratch for one token count: the fused QKV buffer (plus the
        Q/K/V head views FlashAttention wants), the o_proj output, the activated
        MLP tensor, and the single-token attention's destination.  Allocating and
        re-slicing these per call is ~10 us of Python at these sizes; none of
        them escape."""
        b = p.bufs.get(M)
        if b is None:
            if len(p.bufs) >= _MAX_CACHED_SHAPES:      # keep the scratch bounded
                p.bufs.clear()
                p.gu_launch.clear()
                p.act_launch.clear()
                p.cu.clear()
            # FlashAttention's cu_seqlens must live in a buffer this layer owns:
            # the caller's changes identity every step, and a captured graph
            # would keep replaying the address it was captured with.  For a
            # single packed sequence of M tokens it can only be [0, M].
            p.cu[M] = torch.tensor([0, M], dtype=torch.int32, device=device)
            qkv = torch.empty((M, p.q_size + 2 * p.kv_size),
                              dtype=torch.bfloat16, device=device)
            q = qkv[:, :p.q_size].view(M, p.nh, p.hd)
            k = qkv[:, p.q_size:p.q_size + p.kv_size].view(M, p.kvh, p.hd)
            v = qkv[:, p.q_size + p.kv_size:].view(M, p.kvh, p.hd)
            attn_o = torch.empty((M, p.q_size), dtype=torch.bfloat16,
                                 device=device)
            b = p.bufs[M] = (
                qkv, q, k, v, attn_o,
                torch.empty((M, p.hidden), dtype=torch.bfloat16, device=device),
                torch.empty((M, p.inter), dtype=torch.bfloat16, device=device),
                (torch.empty((M, 2 * p.inter), dtype=torch.bfloat16,
                             device=device) if M > _GU_FUSED_MAX_M else None),
                # Single-token attention: one key, so the softmax weight is
                # exactly 1.0 and the output is V broadcast across each GQA
                # group -- bit-identical to FlashAttention, without the kernel.
                (attn_o.view(M, p.kvh, p.nh // p.kvh, p.hd),
                 v.view(M, p.kvh, 1, p.hd)) if M == 1 else None,
            )
        return b

    def _gu(self, M: int, p: _Plan):
        ln = p.gu_launch.get(M)
        if ln is None:
            bn, bk, nw, ns = p.gu_cfg
            if p.packed_gu is None:
                p.packed_gu = _pack_gate_up(p.w_gu, bn, bk)
            bm = 16 if M <= 16 else 32 if M <= 32 else 64 if M <= 64 else 128
            x, a = p.bufs[M][5], p.bufs[M][6]
            ln = p.gu_launch[M] = _Launch(
                _gate_up_act_kernel,
                (-(-p.inter // bn), -(-M // bm), 1),
                (x, p.packed_gu, a, M, p.inter),
                (bm, bn, bk, p.hidden // bk, p.hidden,
                 M % bm != 0, p.inter % bn != 0), nw, ns)
        return ln

    def _act(self, M: int, p: _Plan):
        ln = p.act_launch.get(M)
        if ln is None:
            n = M * p.inter
            blk, vpt, grid, masked = _act_launch(n)
            y, a = p.bufs[M][7], p.bufs[M][6]
            ln = p.act_launch[M] = _Launch(
                _act_kernel, (grid, 1, 1), (y, a, p.inter, n),
                (blk, vpt, masked), 4, 2)
        return ln

    # -- fast path -----------------------------------------------------------
    def _fast(self, positions, hidden_states, residual, p: _Plan):
        M = hidden_states.shape[0]
        qkv, q, k, v, attn_o, proj_o, act, gub, one = self._bufs(
            M, p, hidden_states.device)
        if residual is None:
            residual = hidden_states
            h = self.input_layernorm(hidden_states)
        else:
            h = hidden_states
            _C.fused_add_rmsnorm(h, residual, p.w_in, p.eps_in)
        torch.mm(h, p.wt_qkv, out=qkv)
        _C.rope(qkv, positions, p.cos_sin, p.rot_heads)
        if one is not None:
            one[0].copy_(one[1])
        else:
            cu = p.cu[M]
            attn_o = _flash_attn_varlen(
                q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                max_seqlen_q=M, max_seqlen_k=M,
                softmax_scale=p.scale, causal=True, fa_version=_FA_VERSION,
                num_splits=1).view(M, p.q_size)
        torch.mm(attn_o, p.wt_o, out=proj_o)
        _C.fused_add_rmsnorm(proj_o, residual, p.w_post, p.eps_post)
        if gub is None:
            self._gu(M, p)(proj_o, p.packed_gu, act, M, p.inter)
        else:
            # Above the fused kernel's M tile the packed weight would be
            # re-streamed once per tile; cuBLAS wins the GEMM outright there and
            # only the activation is replaced.
            torch.mm(proj_o, p.wt_gu, out=gub)
            self._act(M, p)(gub, act, p.inter, M * p.inter)
        return torch.mm(act, p.wt_d), residual

    def forward(self, positions, hidden_states, residual):
        p = self._plan
        if p is None:
            if not self._planned:
                self._planned = True
                self._plan = p = self._build_plan()
        elif p.sig != (p.gu_param.data_ptr(), p.gu_param._version):
            self._plan = p = self._build_plan()
        if (p is not None and hidden_states.shape[0] <= _FAST_MAX_M
                and hidden_states.dim() == 2
                and hidden_states.dtype is torch.bfloat16
                and hidden_states.is_contiguous()
                and (residual is None or (residual.is_contiguous()
                     and residual.dtype is torch.bfloat16))
                and positions is not None and positions.dim() == 1
                and positions.dtype is torch.int64
                and hidden_states.shape[1] == p.hidden):
            ctx = get_context()
            if (ctx.is_prefill and not ctx.is_mixed and ctx.block_tables is None
                    and not p.attn.k_cache.numel()
                    and not getattr(ctx, "is_tree_verify", False)
                    and ctx.cu_seqlens_q is not None
                    # one packed sequence of exactly this many tokens, so
                    # ``cu_seqlens`` can only be [0, M] (see ``_bufs``)
                    and int(ctx.cu_seqlens_q.numel()) == 2
                    and int(ctx.max_seqlen_q) == hidden_states.shape[0]
                    and int(ctx.max_seqlen_k) == hidden_states.shape[0]):
                return self._fast(positions, hidden_states, residual, p)

        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
