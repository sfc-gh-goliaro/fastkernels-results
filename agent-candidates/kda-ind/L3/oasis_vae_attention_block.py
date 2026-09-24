"""Fused Oasis VAE attention block.

The baseline block is not arithmetic-bound: it issues 37 kernels per forward and
only about 90 us of its 302 us of self-CUDA time at B=6 is GEMM plus attention.
The rest is fp32<->fp16 casts, the rotary's ``cat``/``stack``/``mul`` chain, a
post-attention ``clone`` and two standalone residual ``add``s.

This subclass keeps the baseline module tree -- so every ``state_dict`` key, and
therefore the harness's ``load_state_dict(..., strict=False)`` weight sharing, is
identical by construction -- and replaces only ``forward`` with an eight-launch
path:

    1  row LayerNorm, reading the input at arbitrary strides   (Triton)
    2  fused QKV GEMM with its bias as a free epilogue         (cuBLASLt)
    3  rotary + split/reshape into (B, S, H, D) q, k, v        (Triton)
    4  non-causal SDPA, backend left to PyTorch's heuristic    (cuDNN)
    5  output projection, bias as a free epilogue              (cuBLASLt)
    6  residual add + row LayerNorm, also emitting the folded
       operand the final GEMM consumes as its ``beta=1`` input (Triton)
    7  fc1 GEMM with a GELU epilogue                           (cuBLASLt)
    8  fc2 GEMM whose ``beta=1`` operand carries both the
       second residual and the fc2 bias                        (cuBLASLt)

Splitting the work this way follows the Blackwell guidance in KernelWiki
``wiki/languages/triton-blackwell.md``: Triton for the memory-bound glue, vendor
kernels for the compute-bound GEMM and attention. Kernel 3 is the same shape of
kernel as ``pr-sglang-21019``, a fused projection split/reshape built from
``tl.load``/``tl.store`` only.

Anything the fast-path guard rejects is routed to ``super().forward``, which is
the unmodified baseline computation.

Numerical standing of each stage, against the baseline:

  * the rotary and the q/k/v split are bit-exact -- the rotation runs in fp32
    from the fp32 ``rotary_freqs`` buffer with a single fp16 rounding, and the
    non-rotated head channels are copied verbatim;
  * the two LayerNorm kernels, the GELU epilogue and the fc2 bias/residual fold
    are equal only within the fp16 comparison tolerance, because none of them
    can reproduce ``F.layer_norm``'s reduction order or cuBLASLt's epilogue
    rounding exactly. Each has its own switch (see the class attributes) so it
    can be measured, and reverted, on its own.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from fastkernels.tasks.baseline.L1.oasis_rotary import oasis_apply_rotary_emb
from fastkernels.tasks.baseline.L3.oasis_vae_attention_block import (
    OasisVAEAttentionBlock as _BaselineBlock,
)

# Sequence tile and warp counts for the three kernels. These are the lowest-median
# settings from scratch/sweep_repeat.py, which measures every setting in each of
# fifteen passes, shuffles the order per pass so position cannot bias a setting,
# and scores by the median of the paired per-pass total across both captured
# shapes. Ranking by the minimum of a few passes was not reproducible -- it named a
# different winner on consecutive runs -- so the decision rule is the median, and
# the lowest median ships with no preference for the incumbent.
#
# Both axes are nearly flat under that method: the nine rotary settings span 0.86 %
# and the four normalization settings 1.57 %, against a per-setting median absolute
# deviation of 1-6 us. That flatness is the finding, and it agrees with the profile
# -- the rotary kernel runs at 0.7 waves per SM, so it is launch-bound rather than
# tiling-bound and its tile size is not what limits it.
#
# Plain constants rather than a ``triton.autotune`` decorator on purpose: the
# harness snapshots ``threading.active_count()`` immediately before it times the
# candidate and the warmup iterations run inside that window, so an autotuner
# evaluated at call time is a direct hazard there.
_ROPE_BLOCK_S = 64
_ROPE_WARPS = 8
_NORM_WARPS = 4


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 1 else 1


@triton.jit
def _mul_sub_rn(a, b, c, d):
    """``fp32(fp32(a*b) - fp32(c*d))`` with all three roundings preserved.

    Written as inline PTX because neither ``a * b - c * d`` in Triton nor plain
    ``mul.f32``/``sub.f32`` survives as three separate roundings: ptxas contracts
    the pair into a single FFMA, which drops the rounding of the first product.
    Measured on a leased B200 (scratch/probe_no_contract.py): the contracted form
    disagrees with the baseline's fp32 evaluation on 95 of 884736 elements, and
    matches an emulated-FFMA reference on all of them; the ``.rn`` modifiers,
    which forbid contraction, bring the disagreement to zero. That is what makes
    the fused rotary bit-exact rather than merely within tolerance.
    """
    return tl.inline_asm_elementwise(
        "{ .reg .f32 p, q; mul.rn.f32 p, $1, $2; mul.rn.f32 q, $3, $4; "
        "sub.rn.f32 $0, p, q; }",
        "=f,f,f,f,f", [a, b, c, d], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _mul_add_rn(a, b, c, d):
    """``fp32(fp32(a*b) + fp32(c*d))``; see :func:`_mul_sub_rn`."""
    return tl.inline_asm_elementwise(
        "{ .reg .f32 p, q; mul.rn.f32 p, $1, $2; mul.rn.f32 q, $3, $4; "
        "add.rn.f32 $0, p, q; }",
        "=f,f,f,f,f", [a, b, c, d], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _rotate_pairs(chans, cos, sin, BLOCK_S: tl.constexpr, HALF: tl.constexpr):
    """Rotate interleaved (even, odd) channel pairs of *chans* by cos/sin.

    Mirrors ``oasis_apply_rotary_emb`` exactly: promote the fp16 channels to
    fp32, form ``even*cos - odd*sin`` / ``odd*cos + even*sin`` there with each
    product rounded on its own, and round back to fp16 once at the end.
    ``cos``/``sin`` are already fp32 and hold one value per pair -- the
    ``repeat_interleave(2)`` in the baseline's frequency table makes the even and
    odd entries bit-identical, so half the table is lossless.
    """
    even, odd = tl.split(tl.reshape(chans, (BLOCK_S, HALF, 2)))
    even = even.to(tl.float32)
    odd = odd.to(tl.float32)
    rotated = tl.join(_mul_sub_rn(even, cos, odd, sin).to(chans.dtype),
                      _mul_add_rn(odd, cos, even, sin).to(chans.dtype))
    return tl.reshape(rotated, (BLOCK_S, HALF * 2))


@triton.jit
def _rope_split_qkv_kernel(
    qkv_ptr, cos_ptr, sin_ptr, q_ptr, k_ptr, v_ptr,
    SEQ: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROT: tl.constexpr,
    HALF_ROT: tl.constexpr,
    PASS: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Rotate q and k and scatter q, k, v into (B, SEQ, HEADS, HEAD_DIM) buffers.

    One program per (sequence tile, head, batch). Every access is a contiguous
    run inside one head, and the destination layout is the ``(B, S, H, D)`` one
    that makes the transpose feeding attention a pure view.
    """
    DIM: tl.constexpr = HEADS * HEAD_DIM

    head = tl.program_id(1)
    batch = tl.program_id(2)
    rows = tl.program_id(0) * BLOCK_S + tl.arange(0, BLOCK_S)
    keep = (rows < SEQ)[:, None]
    flat = batch * SEQ + rows

    src = qkv_ptr + flat[:, None] * (3 * DIM) + head * HEAD_DIM
    dst = (flat * HEADS + head)[:, None] * HEAD_DIM

    pair_c = tl.arange(0, HALF_ROT)[None, :]
    cos = tl.load(cos_ptr + rows[:, None] * HALF_ROT + pair_c, mask=keep, other=0.0)
    sin = tl.load(sin_ptr + rows[:, None] * HALF_ROT + pair_c, mask=keep, other=0.0)

    rot_c = tl.arange(0, ROT)[None, :]
    q_rot = tl.load(src + rot_c, mask=keep, other=0.0)
    k_rot = tl.load(src + DIM + rot_c, mask=keep, other=0.0)
    tl.store(q_ptr + dst + rot_c,
             _rotate_pairs(q_rot, cos, sin, BLOCK_S, HALF_ROT), mask=keep)
    tl.store(k_ptr + dst + rot_c,
             _rotate_pairs(k_rot, cos, sin, BLOCK_S, HALF_ROT), mask=keep)

    if PASS > 0:
        pass_c = ROT + tl.arange(0, PASS)[None, :]
        tl.store(q_ptr + dst + pass_c, tl.load(src + pass_c, mask=keep, other=0.0),
                 mask=keep)
        tl.store(k_ptr + dst + pass_c,
                 tl.load(src + DIM + pass_c, mask=keep, other=0.0), mask=keep)

    full_c = tl.arange(0, HEAD_DIM)[None, :]
    tl.store(v_ptr + dst + full_c,
             tl.load(src + 2 * DIM + full_c, mask=keep, other=0.0), mask=keep)


@triton.jit
def _row_norm_kernel(
    x_ptr, out_ptr, w_ptr, b_ptr,
    stride_batch, stride_seq, stride_dim,
    eps,
    SEQ: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """LayerNorm one row per program, reading the input at arbitrary strides.

    Taking explicit strides is what lets the captured column-major
    ``[1, 576, 1024]`` case (stride ``(589824, 1, 576)``) be normalized in place
    of a separate ``contiguous()`` pass. Statistics are accumulated in fp32 in
    two passes: the row is already in registers, so computing the mean first and
    avoiding the ``E[x^2] - E[x]^2`` cancellation costs nothing.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    keep = cols < DIM
    base = x_ptr + (row // SEQ) * stride_batch + (row % SEQ) * stride_seq
    x = tl.load(base + cols * stride_dim, mask=keep, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / DIM
    centered = tl.where(keep, x - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / DIM
    rstd = tl.rsqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=keep, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=keep, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * DIM + cols,
             (centered * rstd * w + b).to(out_ptr.dtype.element_ty), mask=keep)


@triton.jit
def _row_add_norm_kernel(
    x_ptr, add_ptr, norm_ptr, residual_ptr, w_ptr, b_ptr, fold_bias_ptr,
    stride_batch, stride_seq, stride_dim,
    eps,
    SEQ: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FOLD_BIAS: tl.constexpr,
):
    """Residual add plus LayerNorm, emitting the normalized row and the residual.

    The rounding order is load-bearing. The baseline evaluates ``x + attn(...)``
    in fp16 and only then lets ``LayerNorm`` promote to fp32, so the sum is
    rounded to fp16 *before* the reduction sees it. The statistics are taken over
    that rounded row and nothing else: ``fold_bias_ptr`` contributes only to the
    second output, never to the mean or the variance, because it is destined for
    the final GEMM's ``beta=1`` operand rather than for the normalization.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    keep = cols < DIM
    base = x_ptr + (row // SEQ) * stride_batch + (row % SEQ) * stride_seq
    flat = row * DIM + cols
    x = tl.load(base + cols * stride_dim, mask=keep, other=0.0).to(tl.float32)
    a = tl.load(add_ptr + flat, mask=keep, other=0.0).to(tl.float32)

    residual = (x + a).to(residual_ptr.dtype.element_ty)
    promoted = residual.to(tl.float32)
    if FOLD_BIAS:
        fold = tl.load(fold_bias_ptr + cols, mask=keep, other=0.0).to(tl.float32)
        tl.store(residual_ptr + flat,
                 (promoted + fold).to(residual_ptr.dtype.element_ty), mask=keep)
    else:
        tl.store(residual_ptr + flat, residual, mask=keep)

    mean = tl.sum(promoted, axis=0) / DIM
    centered = tl.where(keep, promoted - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / DIM
    rstd = tl.rsqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=keep, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=keep, other=0.0).to(tl.float32)
    tl.store(norm_ptr + flat,
             (centered * rstd * w + b).to(norm_ptr.dtype.element_ty), mask=keep)


class OasisVAEAttentionBlock(_BaselineBlock):
    """Baseline-equivalent Oasis VAE attention block with a fused forward.

    The switches below are independent on purpose. ``fuse_rope_split`` is only a
    bisection aid: that kernel is bit-exact, so turning it off cannot change the
    numbers, only the launch count. The other four each trade an exactly
    reproducible baseline operation for a tolerance-equivalent fused one, and
    each can be turned off to recover the operation it replaced.
    """

    fuse_rope_split = True
    fuse_input_norm = True
    fuse_residual_norm = True
    gelu_epilogue = True
    fold_output_bias = True

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
        # Spelled out rather than forwarded through ``*args, **kwargs``: the
        # published signature is part of the contract this class is a drop-in
        # for, and callers that introspect it -- or generate a schema from it --
        # see the declaration, not what the body happens to accept.
        super().__init__(dim, num_heads, frame_height, frame_width,
                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias)
        # Plain ints, not tensors, so they stay out of ``state_dict``.
        self._fastpath_calls = 0
        self._fallback_calls = 0
        self._layout_repairs = 0

        attn = self.attn
        self._heads = attn.num_heads
        self._seq = attn.frame_height * attn.frame_width
        self._dim = attn.qkv.weight.shape[1]
        self._head_dim = self._dim // self._heads
        self._rot_dim = attn.rotary_freqs.shape[-1]
        pass_dim = self._head_dim - self._rot_dim
        # Every ``tl.arange`` extent has to be a power of two and the rotated
        # channels have to pair up. A configuration that misses any of this is
        # not rejected loudly, it simply never takes the fused path.
        self._shape_supported = (
            self._heads * self._head_dim == self._dim
            and self._rot_dim >= 2
            and self._rot_dim <= self._head_dim
            and _next_pow2(self._head_dim) == self._head_dim
            and _next_pow2(self._rot_dim) == self._rot_dim
            and (pass_dim == 0 or _next_pow2(pass_dim) == pass_dim)
        )
        # Every tensor the fused path reads, recorded once as (submodule,
        # attribute) so the guard can check them all without rebuilding the access
        # chain on each call. The fused path consumes far more than the QKV weight,
        # and a parameter left on another device or at another dtype would reach a
        # Triton kernel or a GEMM rather than the fallback.
        self._fused_params = tuple(
            (owner, name)
            for owner, names in ((self.norm1, ("weight", "bias")),
                                 (attn.qkv, ("weight", "bias")),
                                 (attn.proj, ("weight", "bias")),
                                 (self.norm2, ("weight", "bias")),
                                 (self.mlp.fc1, ("weight", "bias")),
                                 (self.mlp.fc2, ("weight", "bias")))
            for name in names)
        # Derived state (transposed weight views, the cos/sin table) is built on
        # the first fused call, not here: the harness casts parameters to fp16
        # and moves the module to the GPU after construction, so anything
        # derived now would be an fp32 CPU view.
        self._derived = None
        self._derived_key = None
        self._derived_sources = None

    def _can_fuse(self, x: torch.Tensor) -> bool:
        """Whether *x* is one of the layouts the fused path is validated for."""
        if not self._shape_supported or not x.is_cuda or x.dim() != 3:
            return False
        if x.dtype is not torch.float16:
            return False
        if x.shape[1] != self._seq or x.shape[2] != self._dim or x.shape[0] < 1:
            return False
        # The frequency table has to still be fp32. ``_prepare_module`` casts
        # parameters only and leaves buffers alone, but ``module.half()`` casts
        # buffers too -- and an fp16 table would hand fp16 values to inline PTX
        # declared with f32 operands, which is exactly the fast-and-wrong outcome
        # the guard exists to prevent.
        freqs = self.attn.rotary_freqs
        if freqs.device != x.device or freqs.dtype is not torch.float32:
            return False
        # Every parameter the fused path consumes, not just the QKV weight. The
        # QKV bias is the one that may legitimately be absent.
        device, dtype = x.device, x.dtype
        for owner, name in self._fused_params:
            param = getattr(owner, name)
            if param is None:
                if name == "bias" and owner is self.attn.qkv:
                    continue
                return False
            if param.device != device or param.dtype is not dtype:
                return False
        # The Triton kernels write into buffers they allocate themselves, so they
        # carry no autograd history: taking the fused path while a graph is being
        # built would silently produce partial gradients rather than raise. Grad
        # is off for the whole benchmark, so this costs one ``is_grad_enabled``
        # call in the hot path and the parameter scan never runs there.
        return not (torch.is_grad_enabled()
                    and (x.requires_grad
                         or any(p.requires_grad for p in self.parameters())))

    def _fused_state(self) -> dict:
        """Transposed weight views and the fp32 cos/sin table, built lazily.

        Invalidated on four independent signals, because no one of them covers
        the others:

        * the source tensors are held and compared with ``is``, the same
          ``self._src_w is not self.weight`` idiom the baseline ``LayerNorm``
          uses for its own fp32 cast cache. Holding them rather than their
          ``id()`` is what makes the comparison sound -- a bare id could be
          reused by a different tensor once the original was freed;
        * ``data_ptr``, because rebinding ``param.data`` to fresh storage leaves
          the ``Parameter`` object, its device and its dtype unchanged;
        * device and dtype, because these are plain attributes rather than
          buffers, so nothing would move them for us;
        * the frequency table's ``_version``, because cos/sin are materialized
          copies of it, while the four weight entries are ``.t()`` views that
          pick up an in-place update on their own.
        """
        attn, mlp = self.attn, self.mlp
        sources = (attn.qkv.weight, attn.proj.weight, mlp.fc1.weight,
                   mlp.fc2.weight, attn.rotary_freqs)
        key = (tuple((t.data_ptr(), t.device, t.dtype) for t in sources),
               attn.rotary_freqs._version)
        if (self._derived is not None and self._derived_key == key
                and all(a is b for a, b in zip(self._derived_sources, sources))):
            return self._derived

        # ``rotary_freqs`` is the fp32 non-persistent buffer, not the fp16
        # ``rotary.freqs`` parameter the harness downcasts, and cos/sin are
        # evaluated on the device so they are bit-identical to the baseline's
        # per-call ``freqs.cos()``. Only the even entries are kept: the table is
        # built with ``repeat_interleave(2)``, so the odd ones are duplicates.
        freqs = attn.rotary_freqs
        pairs = (self._seq, self._rot_dim // 2)
        state = {
            "qkv_t": attn.qkv.weight.t(),
            "proj_t": attn.proj.weight.t(),
            "fc1_t": mlp.fc1.weight.t(),
            "fc2_t": mlp.fc2.weight.t(),
            "cos": freqs.cos()[..., 0::2].reshape(pairs).contiguous(),
            "sin": freqs.sin()[..., 0::2].reshape(pairs).contiguous(),
        }
        self._derived = state
        self._derived_key = key
        self._derived_sources = sources
        return state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._can_fuse(x):
            self._fallback_calls += 1
            return super().forward(x)
        self._fastpath_calls += 1

        state = self._fused_state()
        attn, mlp = self.attn, self.mlp
        batch, seq, dim = x.shape
        rows = batch * seq

        normed = self._norm_rows(x, rows)
        # ``qkv`` is the one Linear here whose bias is optional (the baseline's own
        # default is ``qkv_bias=False``); proj, fc1 and fc2 always carry one.
        qkv_bias = attn.qkv.bias
        qkv = (torch.mm(normed, state["qkv_t"]) if qkv_bias is None
               else torch.addmm(qkv_bias, normed, state["qkv_t"]))

        query, key, value = self._project_heads(qkv, state, batch, seq, dim)
        context = F.scaled_dot_product_attention(query, key, value).transpose(1, 2)
        # cuDNN returns its (B, H, S, D) result laid out as (B, S, H, D), which
        # makes the transpose above a view and the flatten below free -- that is
        # what deletes the baseline's post-attention clone. It is a property of
        # the current backend rather than a guarantee, so it is checked.
        #
        # The repair is counted rather than performed silently. ``reshape`` below
        # would copy on its own for a layout that cannot be viewed, so without a
        # counter this branch is indistinguishable from the code that follows it,
        # and a future backend that changed its output layout would show up as an
        # unexplained extra copy instead of as a number.
        if not context.is_contiguous():
            self._layout_repairs += 1
            context = context.contiguous()
        projected = torch.addmm(attn.proj.bias, context.reshape(rows, dim),
                                state["proj_t"])

        normed2, residual = self._add_norm_rows(x, projected, rows)
        if self.gelu_epilogue:
            hidden = torch._addmm_activation(mlp.fc1.bias, normed2, state["fc1_t"],
                                             use_gelu=True)
        else:
            hidden = F.gelu(torch.addmm(mlp.fc1.bias, normed2, state["fc1_t"]))

        if self.fold_output_bias:
            # ``residual`` is a buffer this call allocated and nothing else holds,
            # so the final GEMM can accumulate into it: measured bit-identical to
            # the out-of-place form and 2-3 us cheaper, because a matrix ``input``
            # otherwise costs an extra copy. ``out=`` is rejected when an operand
            # requires grad, which cannot happen here -- the guard has already
            # routed any grad-building call to the exact fallback.
            out = torch.addmm(residual, hidden, state["fc2_t"], out=residual)
        else:
            out = residual + torch.addmm(mlp.fc2.bias, hidden, state["fc2_t"])
        return out.view(batch, seq, dim)

    def _norm_rows(self, x: torch.Tensor, rows: int) -> torch.Tensor:
        """First LayerNorm, flattened to ``(rows, dim)`` for the QKV GEMM."""
        dim = x.shape[2]
        if not self.fuse_input_norm:
            return self.norm1(x).reshape(rows, dim)
        out = torch.empty((rows, dim), dtype=x.dtype, device=x.device)
        _row_norm_kernel[(rows,)](
            x, out, self.norm1.weight, self.norm1.bias,
            x.stride(0), x.stride(1), x.stride(2),
            self.norm1.eps,
            SEQ=self._seq, DIM=dim, BLOCK_D=_next_pow2(dim),
            num_warps=_NORM_WARPS,
        )
        return out

    def _project_heads(self, qkv, state, batch, seq, dim):
        """Rotate q/k and split qkv into the layout attention wants.

        Returns ``(B, H, S, D)`` views. With the fused kernel they are views of
        ``(B, S, H, D)``-contiguous buffers, which is what makes the attention
        output's own transpose free.
        """
        heads, head_dim = self._heads, self._head_dim
        if not self.fuse_rope_split:
            freqs = self.attn.rotary_freqs
            parts = []
            for part in qkv.view(batch, seq, 3 * dim).chunk(3, dim=-1):
                part = part.reshape(batch, self.attn.frame_height,
                                    self.attn.frame_width, heads, head_dim)
                part = part.permute(0, 3, 1, 2, 4)
                if len(parts) < 2:
                    part = oasis_apply_rotary_emb(freqs, part)
                parts.append(part.reshape(batch, heads, seq, head_dim))
            return tuple(parts)

        shape = (batch, seq, heads, head_dim)
        query = torch.empty(shape, dtype=qkv.dtype, device=qkv.device)
        key = torch.empty(shape, dtype=qkv.dtype, device=qkv.device)
        value = torch.empty(shape, dtype=qkv.dtype, device=qkv.device)
        _rope_split_qkv_kernel[(triton.cdiv(seq, _ROPE_BLOCK_S), heads, batch)](
            qkv, state["cos"], state["sin"], query, key, value,
            SEQ=seq, HEADS=heads, HEAD_DIM=head_dim,
            ROT=self._rot_dim, HALF_ROT=self._rot_dim // 2,
            PASS=head_dim - self._rot_dim, BLOCK_S=_ROPE_BLOCK_S,
            num_warps=_ROPE_WARPS,
        )
        return query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

    def _add_norm_rows(self, x, projected, rows):
        """Second residual add and LayerNorm.

        Returns the normalized rows and the operand the final GEMM accumulates
        into: the plain residual, or the residual already carrying the fc2 bias
        when that fold is enabled.
        """
        dim = x.shape[2]
        fold = self.fold_output_bias
        bias = self.mlp.fc2.bias
        if not self.fuse_residual_norm:
            residual = x + projected.view_as(x)
            normed = self.norm2(residual).reshape(rows, dim)
            residual = residual.reshape(rows, dim)
            return normed, (residual + bias if fold else residual)

        normed = torch.empty((rows, dim), dtype=x.dtype, device=x.device)
        residual = torch.empty((rows, dim), dtype=x.dtype, device=x.device)
        _row_add_norm_kernel[(rows,)](
            x, projected, normed, residual,
            self.norm2.weight, self.norm2.bias, bias,
            x.stride(0), x.stride(1), x.stride(2),
            self.norm2.eps,
            SEQ=self._seq, DIM=dim, BLOCK_D=_next_pow2(dim), FOLD_BIAS=fold,
            num_warps=_NORM_WARPS,
        )
        return normed, residual
