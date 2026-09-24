"""FLUX attention module (L2 composite) -- fused pre-attention glue path.

Same contract as ``baseline.py``; the difference is everything between the QKV
projections and the attention call.

Baseline pre-attention path, per call::

    qkv = to_qkv(hidden_states)              # [1, S_img, 3*H*D]
    q, k, v = qkv.split(...)                 # strided views (row stride 3*H*D)
    q, k, v = [t.unflatten(-1, (H, D)) ...]
    q = norm_q(q); k = norm_k(k)             # each: .contiguous() clone + rms_norm
    # dual stream only:
    eqkv = add_kv_proj(encoder_hidden_states)
    ... same split/unflatten/norm for the 512 text tokens ...
    q = cat([eq, q], 1); k = cat([ek, k], 1); v = cat([ev, v], 1)
    cos = cos.to(bf16); sin = sin.to(bf16)   # from the captured float64 tables
    q = rope(q, cos, sin); k = rope(k, cos, sin)

That is 13 launches (dual) / 7 launches (single) and, measured on B200 at
[1, 4608, 3072], ~0.32 ms single / ~0.47 ms dual -- 45-55% of the whole
operator, running at ~0.6-1.5 TB/s against an ~8 TB/s HBM.  Two structural
reasons: ``RMSNorm.forward_cuda`` calls ``x.contiguous()`` on the strided
q/k views (a full extra 28 MB clone each), and the three ``torch.cat``s move
~170 MB of read+write traffic for zero math.

This file collapses all of it into **one** Triton launch over a single packed
QKV buffer:

* The buffer is ``[S_total, 3*H*D]`` bf16 laid out ``[q | k | v]`` on the last
  axis and ``[text || image]`` on the token axis -- exactly the order the
  baseline's three ``cat``s produce.  Both projections write straight into their
  own row range of it (``addmm(..., out=buf[:512])`` / ``out=buf[512:]``), so
  the concatenation costs *nothing*: no cat kernels, no extra tensors.
* q and k occupy one contiguous ``2*H*D`` span per token, so a single kernel
  streams that span, does the RMSNorm and the interleaved (non-neox) RoPE
  in-register, and writes the result back in place.  v is never touched.
* cos/sin are read as float64 straight from the captured tables and rounded to
  bf16 inside the kernel, which also removes the two dtype-cast launches.
* q/k/v are then handed to attention as strided views of the packed buffer.  The
  cuDNN SDPA backend accepts a 3*H*D row stride and returns bit-identical output.
  Measured in situ on B200 the packed layout does cost the attention kernel ~7%
  (181 us vs 168 us at S=4608), but the three ``.contiguous()`` calls needed to
  avoid that cost 75 us -- so keeping everything in one buffer wins by ~62 us.

Numerics are held to the baseline's exact arithmetic: the norm computes
``(x*rsqrt(mean(x^2)+eps))*w`` in fp32 and **rounds to bf16** before the RoPE
reads it (the baseline round-trips through memory there), and the RoPE keeps
the ``x_even*cos - x_odd*sin`` / ``x_odd*cos + x_even*sin`` pairing in fp32.

Any shape/dtype the fused path does not cover (fp8 weights, head_dim whose half
is not a power of two, partial rotary_dim, num_heads != num_kv_heads, or a
dual-stream call with batch > 1) falls back to the baseline sequence.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ....infra.tp import _tp_size
from ..L1.rms_norm import RMSNorm as FP32RMSNorm
from ..L1.diffusion_rope import DiffusionRoPE
from ..L1.dense_attention import DenseAttention
from .parallel_linear import (
    QKVParallelLinear,
    RowParallelLinear,
)


def _tensor_model_parallel_all_gather(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Gather tensor across TP ranks along the given dimension."""
    import torch.distributed as dist
    tp = _tp_size()
    if tp <= 1:
        return tensor
    gather_list = [torch.empty_like(tensor) for _ in range(tp)]
    dist.all_gather(gather_list, tensor)
    return torch.cat(gather_list, dim=dim)


# ---------------------------------------------------------------------------
# Fused qk-RMSNorm + interleaved RoPE, in place on a packed [q|k|v] buffer.
# ---------------------------------------------------------------------------
@triton.jit
def _qk_norm_rope_inplace(
    BUF,                      # [B, S, 3*H*D] bf16/fp16, contiguous rows
    COS, SIN,                 # [S_ro, D//2] any float dtype, last dim contiguous
    WQ, WK, WAQ, WAK,         # [D] norm weights (image pair / text pair)
    s_txt,                    # number of leading text tokens (0 if single-stream)
    stride_buf_b, stride_buf_t, stride_cos,
    S, eps,
    H: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr, NH2: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_H: tl.constexpr,
    HAS_ROPE: tl.constexpr, HAS_ADDED: tl.constexpr, MASK_T: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    t0 = pid_t * BLOCK_T
    t = t0 + tl.arange(0, BLOCK_T)
    # h in [0, 2*H): h < H are the q heads, h >= H the k heads.  Because q and k
    # are adjacent on the last axis of the packed buffer, h*D walks one
    # contiguous 2*H*D span per token -- v (at 2*H*D) is never addressed.
    h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    d = tl.arange(0, D)

    off = (pid_b * stride_buf_b
           + t[:, None, None] * stride_buf_t
           + h[None, :, None] * D
           + d[None, None, :])
    # The head-dim axis must stay a plain contiguous `arange` here: addressing
    # the interleaved pairs directly (`dp*2 + p`) stops Triton vectorizing the
    # access and costs 2x (32.9 us vs 16.5 us at S=4608 on B200).  Load the
    # contiguous row, then do the even/odd split in registers below.
    if MASK_T:
        m = (t < S)[:, None, None]
        x = tl.load(BUF + off, mask=m, other=0.0).to(tl.float32)
    else:
        x = tl.load(BUF + off).to(tl.float32)

    var = tl.sum(x * x, axis=2) * (1.0 / D)
    inv = tl.rsqrt(var + eps)

    wq = tl.load(WQ + d).to(tl.float32)
    wk = tl.load(WK + d).to(tl.float32)
    if HAS_ADDED:
        # BLOCK_T divides s_txt (checked on the host), so a program is either
        # all-text or all-image and this select is uniform across the program.
        is_text = t0 < s_txt
        wq = tl.where(is_text, tl.load(WAQ + d).to(tl.float32), wq)
        wk = tl.where(is_text, tl.load(WAK + d).to(tl.float32), wk)
    w = tl.where((h >= H)[:, None], wk[None, :], wq[None, :])

    # Round to the storage dtype before RoPE: the baseline's norm writes bf16 to
    # memory and the RoPE kernel reads it back, so this rounding is observable.
    xn = (x * inv[:, :, None] * w[None, :, :]).to(BUF.dtype.element_ty).to(tl.float32)

    if HAS_ROPE:
        dp = tl.arange(0, HALF)
        coff = t[:, None] * stride_cos + dp[None, :]
        if MASK_T:
            cm = (t < S)[:, None]
            cos = tl.load(COS + coff, mask=cm, other=1.0)
            sin = tl.load(SIN + coff, mask=cm, other=0.0)
        else:
            cos = tl.load(COS + coff)
            sin = tl.load(SIN + coff)
        # The baseline casts the float64 tables to the activation dtype once
        # (``cos.to(query.dtype)``), then widens to fp32 inside the rotary
        # kernel; mirror that, including the intermediate rounding.
        cos = cos.to(BUF.dtype.element_ty).to(tl.float32)
        sin = sin.to(BUF.dtype.element_ty).to(tl.float32)
        xe, xo = tl.split(tl.reshape(xn, (BLOCK_T, BLOCK_H, HALF, 2)))
        oe = xe * cos[:, None, :] - xo * sin[:, None, :]
        oo = xo * cos[:, None, :] + xe * sin[:, None, :]
        out = tl.reshape(tl.join(oe, oo), (BLOCK_T, BLOCK_H, D))
    else:
        out = xn

    if MASK_T:
        tl.store(BUF + off, out.to(BUF.dtype.element_ty), mask=m)
    else:
        tl.store(BUF + off, out.to(BUF.dtype.element_ty))


# (BLOCK_T, BLOCK_H, num_warps, num_stages).  One 2048-element tile per program
# on a single warp: 64 elements/thread is what lets Triton issue 128-bit loads
# back to back, and raising num_warps at this tile size halves throughput on the
# large shapes (16.5 -> 20.6/30.9 us at w=2/w=4).  Equal-best on all four
# captured shapes, so there is only one JIT specialization to warm up.
_CFG = (1, 16, 1, 2)


# Once the glue is one launch, the remaining cost on the two short-sequence
# shapes is host-side: at S=1536 the whole op is 82 us of device work but 138 us
# of wall clock, so Python/dispatch is the bottleneck there.  Triton's
# ``JITFunction.run`` re-binds and re-specializes all 22 arguments on every call
# (13.8 us measured); a ``CompiledKernel`` runner bound to a fixed grid skips
# that and costs ~6 us less.  The cache key carries every value the
# specialization depends on -- shapes, strides, the scalar arg values and the
# low bits of each pointer -- so a cached runner can never be reused for a
# differently-specialized launch.  (The benchmark's shifting input pool steps by
# whole 256 B blocks, so alignment is in practice constant, but keying on it
# means we do not have to rely on that.)
_RUNNER_CACHE: dict = {}


def _jit_launcher(grid, warps, stages):
    def launch(*args):
        _qk_norm_rope_inplace[grid](*args, num_warps=warps, num_stages=stages)
    return launch


def _launch_qk_norm_rope(buf, cos, sin, wq, wk, waq, wak, s_txt, num_heads,
                         head_dim, eps):
    """``buf``: [B, S, 3*H*D] with contiguous last axis; edited in place."""
    B, S, width = buf.shape
    nh2 = 2 * num_heads
    block_t, block_h, warps, stages = _CFG
    while nh2 % block_h:
        block_h //= 2
    grid = (-(-S // block_t), nh2 // block_h, B)
    sb0, sb1 = buf.stride(0), buf.stride(1)
    has_rope = cos is not None
    if has_rope:
        sc = cos.stride(-2)
        align = (buf.data_ptr() | cos.data_ptr() | sin.data_ptr()) & 15
    else:
        sc = 0
        cos = sin = None
        align = buf.data_ptr() & 15
    has_added = waq is not None
    if not has_added:
        waq, wak = wq, wk
    args = (buf, cos, sin, wq, wk, waq, wak, s_txt, sb0, sb1, sc, S, eps,
            num_heads, head_dim, head_dim // 2, nh2, block_t, block_h,
            has_rope, has_added, (S % block_t) != 0)
    key = (buf.dtype, None if cos is None else cos.dtype, wq.dtype, align,
           grid, s_txt, sb0, sb1, sc, S, eps, num_heads, head_dim,
           block_t, block_h, warps, stages, has_rope, has_added)
    run = _RUNNER_CACHE.get(key)
    if run is not None:
        run(*args)
        return
    compiled = _qk_norm_rope_inplace[grid](*args, num_warps=warps,
                                           num_stages=stages)
    try:
        _RUNNER_CACHE[key] = compiled[grid]
    except Exception:  # noqa: BLE001 - unexpected Triton version; keep the slow path
        _RUNNER_CACHE[key] = _jit_launcher(grid, warps, stages)


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


# Tried and rejected: calling ``torch.ops.aten._scaled_dot_product_cudnn_attention``
# directly instead of going through ``DenseAttention`` (which pins the backend
# with a ``sdpa_kernel`` context manager costing ~18 us of host time per call).
# Bit-identical output, and ~18 us cheaper in an isolated host-cost measurement,
# but worth 0 / +0.02x / 0 / +0.03x under the benchmark: after the plan cache
# below, the short shapes sit at the floor set by their device work, so shaving
# host time no longer moves the number.  Not worth the private-op dependency.


def _drop_plan(module, _incompatible_keys=None) -> None:
    module._plan = None
    module._plan_dtype = None


class FluxAttention(nn.Module):
    """Multi-head attention for FLUX diffusion transformer.

    Supports two modes controlled by constructor args:
    - Dual-stream (``added_kv_proj_dim is not None``): separate QKV for image
      and text streams, concatenated before attention, split after.
    - Single-stream / pre-only (``pre_only=True``): standard self-attention,
      no output projection (caller handles it).
    """

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: int | None = None,
        added_proj_bias: bool | None = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int | None = None,
        context_pre_only: bool | None = None,
        pre_only: bool = False,
        quant_config: dict | None = None,
    ):
        super().__init__()
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.dropout = dropout
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim

        self.norm_q = FP32RMSNorm(dim_head, eps=eps)
        self.norm_k = FP32RMSNorm(dim_head, eps=eps)

        self.to_qkv = QKVParallelLinear(
            hidden_size=query_dim,
            head_size=self.head_dim,
            total_num_heads=self.heads,
            total_num_kv_heads=self.heads,
            bias=bias,
            quant_config=quant_config,
        )

        if not self.pre_only:
            self.to_out = nn.ModuleList([
                RowParallelLinear(self.inner_dim, self.out_dim, bias=out_bias,
                                  quant_config=quant_config),
                nn.Dropout(dropout),
            ])

        if added_kv_proj_dim is not None:
            self.norm_added_q = FP32RMSNorm(dim_head, eps=eps)
            self.norm_added_k = FP32RMSNorm(dim_head, eps=eps)

            self.add_kv_proj = QKVParallelLinear(
                hidden_size=added_kv_proj_dim,
                head_size=self.head_dim,
                total_num_heads=self.heads,
                total_num_kv_heads=self.heads,
                bias=added_proj_bias if added_proj_bias is not None else True,
                quant_config=quant_config,
            )

            self.to_add_out = RowParallelLinear(
                self.inner_dim, query_dim, bias=out_bias,
                quant_config=quant_config,
            )

        self.rope = DiffusionRoPE(is_neox_style=False)
        self.attn = DenseAttention()

        self._plan = None
        self._plan_dtype = None
        self.register_load_state_dict_post_hook(_drop_plan)

    # -- fused path plumbing -------------------------------------------------

    def _apply(self, *args, **kwargs):
        # .to() / .cuda() / .float() replace parameter storages; the plan caches
        # tensor handles derived from them, so drop it.
        self._plan = None
        self._plan_dtype = None
        return super()._apply(*args, **kwargs)

    def _norm_weight(self, mod: nn.Module, dtype, device) -> torch.Tensor:
        w = mod.weight if mod.elementwise_affine else mod._unit_weight
        if w.dtype != dtype or w.device != device:
            w = w.to(device=device, dtype=dtype)
        return w

    def _build_plan(self, dtype) -> None:
        """Resolve everything the fused path needs once per (module, dtype).

        Cached in ``self._plan`` and dropped by ``_apply`` / the load-state-dict
        hook, so the per-call cost is one ``is not`` check on the dtype instead
        of ~30 ``nn.Module.__getattr__`` lookups and four ``.to()`` probes.
        ``None`` means "this module/dtype combination is not supported"; the
        forward then runs the baseline sequence.
        """
        self._plan_dtype = dtype
        self._plan = None
        qkv = self.to_qkv
        hd = self.head_dim
        if hd % 2 or not _is_pow2(hd // 2):
            return
        if qkv.num_heads != qkv.num_kv_heads or qkv.use_fp8:
            return
        if dtype not in (torch.bfloat16, torch.float16) or qkv.weight.dtype != dtype:
            return
        dev = qkv.weight.device
        added = self.added_kv_proj_dim is not None
        if added:
            akv = self.add_kv_proj
            if akv.use_fp8 or akv.weight.dtype != dtype:
                return
            awt, ab = akv.weight.t(), akv.bias
            waq = self._norm_weight(self.norm_added_q, dtype, dev)
            wak = self._norm_weight(self.norm_added_k, dtype, dev)
        else:
            awt = ab = waq = wak = None
        self._plan = (
            qkv.num_heads, hd, 3 * qkv.num_heads * hd,
            qkv.weight.t(), qkv.bias, awt, ab,
            self._norm_weight(self.norm_q, dtype, dev),
            self._norm_weight(self.norm_k, dtype, dev), waq, wak,
            self.norm_q.eps, dev,
            self.attn.forward,
            None if self.pre_only else self.to_out[0].forward,
            # nn.Dropout with p=0 is the identity in train and eval alike, so
            # skipping the module call changes nothing.
            None if (self.pre_only or self.dropout == 0.0) else self.to_out[1],
            self.to_add_out.forward if added else None,
        )

    def _packed_qkv(self, plan, hidden_states, encoder_hidden_states, cos, sin):
        """Build the ``[B, S_total, 3*H*D]`` packed, normed, rotated QKV buffer."""
        (nh, hd, width, wt, bias, awt, ab, wq, wk, waq, wak, eps, dev,
         _, _, _, _) = plan
        B, s_img, qdim = hidden_states.shape

        if encoder_hidden_states is None:
            x = hidden_states.reshape(-1, qdim)
            buf = (torch.mm(x, wt) if bias is None
                   else torch.addmm(bias, x, wt)).view(B, s_img, width)
            s_txt = 0
            waq = wak = None
        else:
            s_txt = encoder_hidden_states.shape[1]
            buf = torch.empty(s_txt + s_img, width, device=dev,
                              dtype=hidden_states.dtype)
            # The two projections write straight into their own row range, which
            # is what makes the [text || image] concatenation free.
            xt = encoder_hidden_states.reshape(-1, encoder_hidden_states.shape[-1])
            dst = buf[:s_txt]
            if ab is None:
                torch.mm(xt, awt, out=dst)
            else:
                torch.addmm(ab, xt, awt, out=dst)
            xi = hidden_states.reshape(-1, qdim)
            dst = buf[s_txt:]
            if bias is None:
                torch.mm(xi, wt, out=dst)
            else:
                torch.addmm(bias, xi, wt, out=dst)
            buf = buf.view(1, s_txt + s_img, width)

        _launch_qk_norm_rope(buf, cos, sin, wq, wk, waq, wak, s_txt, nh, hd, eps)
        return torch.unbind(buf.view(B, buf.shape[1], 3, nh, hd), 2)

    # -- baseline (fallback) pre-attention path ------------------------------

    def _apply_rope(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            cos = cos.to(query.dtype)
            sin = sin.to(query.dtype)
            query = self.rope(query, cos, sin)
            key = self.rope(key, cos, sin)
        return query, key

    def _reference_qkv(self, hidden_states, encoder_hidden_states, image_rotary_emb):
        num_heads = self.to_qkv.num_heads
        num_kv_heads = self.to_qkv.num_kv_heads

        qkv = self.to_qkv(hidden_states)
        q_size = num_heads * self.head_dim
        kv_size = num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)

        query = query.unflatten(-1, (num_heads, -1))
        key = key.unflatten(-1, (num_kv_heads, -1))
        value = value.unflatten(-1, (num_kv_heads, -1))

        query = self.norm_q(query)
        key = self.norm_k(key)

        if self.added_kv_proj_dim is not None:
            add_num_heads = self.add_kv_proj.num_heads
            add_num_kv_heads = self.add_kv_proj.num_kv_heads

            encoder_qkv = self.add_kv_proj(encoder_hidden_states)
            add_q_size = add_num_heads * self.head_dim
            add_kv_size = add_num_kv_heads * self.head_dim
            encoder_query, encoder_key, encoder_value = encoder_qkv.split(
                [add_q_size, add_kv_size, add_kv_size], dim=-1
            )

            encoder_query = encoder_query.unflatten(-1, (add_num_heads, -1))
            encoder_key = encoder_key.unflatten(-1, (add_num_kv_heads, -1))
            encoder_value = encoder_value.unflatten(-1, (add_num_kv_heads, -1))

            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        return self._apply_rope(query, key, image_rotary_emb) + (value,)

    # -- forward -------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self._plan_dtype is not hidden_states.dtype:
            self._build_plan(hidden_states.dtype)
        plan = self._plan
        if plan is not None and torch.is_grad_enabled():
            # The fused path edits the projection output in place, which trips
            # autograd's version counter; hand grad-enabled calls to the unfused
            # sequence (which is what the baseline runs anyway).
            plan = None

        cos = sin = None
        if plan is not None:
            if hidden_states.dim() != 3:
                plan = None
            elif image_rotary_emb is not None:
                cos, sin = image_rotary_emb
                if cos.dim() == 3:
                    cos, sin = cos[0], sin[0]
                if (cos.dim() != 2 or sin.shape != cos.shape
                        or cos.shape[-1] * 2 != plan[1]):
                    plan = None                     # partial rotary_dim etc.
                else:
                    if cos.stride(-1) != 1:
                        cos = cos.contiguous()
                    if sin.stride(-1) != 1:
                        sin = sin.contiguous()
        if plan is not None and encoder_hidden_states is not None:
            # The packed buffer interleaves [text || image] on the token axis, so
            # the two GEMMs can only write disjoint row ranges when B == 1.
            if (plan[5] is None or hidden_states.shape[0] != 1
                    or encoder_hidden_states.dim() != 3
                    or encoder_hidden_states.dtype is not hidden_states.dtype
                    or encoder_hidden_states.shape[1] % _CFG[0]):
                plan = None

        if plan is not None:
            query, key, value = self._packed_qkv(
                plan, hidden_states, encoder_hidden_states, cos, sin)
            attn_fwd, to_out0, drop, add_out = plan[13:17]
        else:
            query, key, value = self._reference_qkv(
                hidden_states, encoder_hidden_states, image_rotary_emb)
            attn_fwd = self.attn.forward
            to_out0 = None if self.pre_only else self.to_out[0].forward
            drop = None if self.pre_only else self.to_out[1]
            add_out = (self.to_add_out.forward
                       if self.added_kv_proj_dim is not None else None)

        softmax_scale = 1.0 / (self.head_dim ** 0.5)
        hidden_states = attn_fwd(query, key, value, softmax_scale=softmax_scale,
                                 causal=False)
        hidden_states = hidden_states.flatten(2, 3)
        if hidden_states.dtype is not query.dtype:
            hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]],
                dim=1,
            )
            hidden_states = to_out0(hidden_states.contiguous())
            if drop is not None:
                hidden_states = drop(hidden_states)
            encoder_hidden_states = add_out(encoder_hidden_states.contiguous())
            return hidden_states, encoder_hidden_states
        else:
            if _tp_size() > 1:
                hidden_states = _tensor_model_parallel_all_gather(hidden_states, dim=-1)
            return hidden_states
