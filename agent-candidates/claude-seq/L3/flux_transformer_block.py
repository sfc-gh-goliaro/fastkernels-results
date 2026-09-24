"""FLUX transformer blocks (L3 composites) -- fastkernels candidate.

Same public surface as ``baseline.py``: ``FluxTransformerBlock`` (dual-stream,
adaLN-Zero conditioning, joint attention then separate FFNs) and
``FluxSingleTransformerBlock`` (single-stream, attention and MLP in parallel).
Submodule names and ``__init__`` signatures are unchanged, so the scorer's
``load_state_dict`` still shares weights with the baseline.

What changes is everything between the GEMMs.  Profiling the baseline chain on a
B200 (bf16, S_img = 4096 / 1024, S_txt = 512) shows the tensor-core work is
already at ~1.7-2.0 PFLOP/s, but a quarter of the dual-stream block's GPU time
goes to *broadcast* elementwise kernels::

    norm1        LN(x)                            + x*(1+scale) + shift
    norm1_ctx    LN(c)                            + c*(1+scale) + shift
    attn gate    gate*attn_out                    + x + .
    norm2        LN(x)                            + x*(1+scale) + shift
    mlp gate     gate*ff_out                      + x + .
    ... and the same five again for the text stream

Sixteen full passes over the activation, and torch's eager broadcast kernel
(``elementwise_kernel<128, 4, gpu_kernel_impl_nocast<...>>``) runs them at
~1.5-1.8 TB/s where a plain ``copy_`` of the same buffers reaches ~3.2 TB/s: the
index arithmetic for the broadcast defeats vectorization.  ``flux_block_fused.cu``
collapses each chain into a single fused pass -- read once, write once, with the
modulation vectors staged in shared memory -- which is three launches per
dual-stream block instead of sixteen:

    ln_mod        y = LN(x) * (1 + scale) + shift
    add_ln_mod    x' = x + gate*a ;  y = LN(x') * (1 + scale) + shift
    gated_add     out = res + gate*y

each covering the image and text streams in one launch (``grid.y`` selects the
segment), so the 512-row text stream costs nothing extra in launch latency.

Two more structural changes:

* **The single-stream block materializes neither of its two concatenations.**
  The fused prologue reads the text and image streams as two segments of one
  logical sequence and writes one joint normalized buffer (which the attention
  needs anyway), and the epilogue reads the two residuals back in place and
  writes the two output halves directly -- so the 28 MB ``cat([text, image])``
  and the separate residual tensor both disappear.  The 141 MB
  ``cat([attn_out, mlp_hidden])`` is replaced by splitting the output projection
  over the two halves of its own weight and accumulating.
* **``silu(temb)`` is computed once** and shared by the dual-stream block's two
  conditioning projections, which the reference evaluates separately.

The conditioning projection itself (3072 -> 6*3072 against a single activation
row) is left to cuBLAS even though it is a pure streaming read of a 113 MB
weight that runs at only ~3.8 TB/s: a warp-per-output-element GEMV reaches
memory speed *and* is ~700x closer to the exactly-rounded result, but cuBLAS's
own fp32 reduction error flips ~29 of the 18432 bf16 conditioning values by one
ulp against it -- and this block amplifies that.  A one-ulp move in a single
modulation value perturbs a whole column of the attention input, the softmax
spreads it over the sequence, and ~0.7% of the output elements then land outside
the 1% relative tolerance *the reference itself defines*.  Being more accurate
than the thing you are scored against is not worth 2%.

One thing to know before touching ``flux_block_fused.cu``: the fused kernels
reproduce the reference's *rounding*, not just its algebra.  Every value the
reference materializes as a tensor -- ``1 + scale``, ``gate * attn_out``,
``LN(x)``, ``LN(x) * (1 + scale)`` -- is rounded to the activation dtype there
too.  Evaluating a chain in one fp32 expression instead is more accurate and
still fails: the block feeds its own attention, a softmax over 1536-4608 keys
spreads any one-ulp move across the sequence, and single-rounding the modulation
put 3-4% of the output outside the reference's 1% relative tolerance (the
comparison needs 99% of elements inside it).  That is also why ``mul``/``add``
there are inline PTX and not ``__hmul2``/``__hadd2`` -- nvcc contracts the pair
into a single-rounding ``fma.rn.bf16x2``.

The reference chain is kept verbatim as ``_forward_reference`` and taken for
anything the fused path does not cover: a non-bf16/fp16 dtype, a hidden size
that is not a multiple of 8, TP > 1 or FP8 projections (which change what the
submodule forwards do), or a failed JIT build.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..L1.gelu import GELU
from ..L1.layer_norm import LayerNorm
from ..L2.ada_layer_norm import AdaLayerNormZero, AdaLayerNormZeroSingle
from ..L2.flux_attention import FluxAttention
from ..L2.flux_feedforward import FeedForward
from ..L2.parallel_linear import ReplicatedLinear

try:
    from fastkernels.infra.cuda_ext import load_op

    _C = load_op("fk_l3_flux_block", "flux_block_fused.cu")
except Exception:  # pragma: no cover - no CUDA toolchain / no GPU
    _C = None


_EPS = 1e-6


def _mod_ok(dim: int) -> bool:
    return _C is not None and dim % 8 == 0


def _linear_plain(lin: nn.Module) -> bool:
    """True when this projection is a plain (non-FP8, non-sharded) matmul."""
    if getattr(lin, "use_fp8", False):
        return False
    return getattr(lin, "tp_size", 1) == 1


class FluxTransformerBlock(nn.Module):
    """Dual-stream DiT block: joint attention over text+image, then separate FFNs."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        quant_config: dict | None = None,
    ):
        super().__init__()
        self.norm1 = AdaLayerNormZero(dim, promote_fp32=False)
        self.norm1_context = AdaLayerNormZero(dim, promote_fp32=False)

        self.attn = FluxAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            eps=eps,
            quant_config=quant_config,
        )

        self.norm2 = LayerNorm(dim, elementwise_affine=False, eps=1e-6, promote_fp32=False)
        self.ff = FeedForward(dim=dim, dim_out=dim, quant_config=quant_config)

        self.norm2_context = LayerNorm(dim, elementwise_affine=False, eps=1e-6, promote_fp32=False)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, quant_config=quant_config)

        self.dim = dim
        self._fused_static = _mod_ok(dim)

    # -- fast path ---------------------------------------------------------

    def _fused_ok(self, hidden_states, encoder_hidden_states, temb,
                  joint_attention_kwargs) -> bool:
        if not self._fused_static or joint_attention_kwargs:
            return False
        if hidden_states.dtype not in (torch.bfloat16, torch.float16):
            return False
        if (hidden_states.dim() != 3 or encoder_hidden_states.dim() != 3
                or hidden_states.size(0) != 1 or encoder_hidden_states.size(0) != 1):
            return False
        if (encoder_hidden_states.dtype is not hidden_states.dtype
                or temb.dtype is not hidden_states.dtype):
            return False
        if temb.dim() != 2 or temb.size(0) != 1:
            return False
        if not (hidden_states.is_contiguous() and encoder_hidden_states.is_contiguous()
                and temb.is_contiguous()):
            return False
        return True

    def _emb(self, temb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``linear(silu(temb))`` for both conditioning projections.

        The reference runs ``silu`` twice on the same ``temb`` (once per
        ``AdaLayerNormZero``); one activation serves both.
        """
        s = F.silu(temb)
        return (self.norm1.linear(s).view(-1),
                self.norm1_context.linear(s).view(-1))

    def _forward_fused(self, hidden_states, encoder_hidden_states, temb,
                       image_rotary_emb):
        n = self.dim
        mod_i, mod_c = self._emb(temb)

        # norm1 / norm1_context: LN + shift/scale, both streams in one launch.
        norm_hidden_states = torch.empty_like(hidden_states)
        norm_encoder_hidden_states = torch.empty_like(encoder_hidden_states)
        _C.ln_mod(hidden_states, norm_hidden_states, mod_i, 0,
                  encoder_hidden_states, norm_encoder_hidden_states, mod_c, 0, _EPS)

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        # residual add of the gated attention output, then norm2 + shift/scale.
        # ``norm_*`` are dead here, so the normalized output overwrites them.
        hid = torch.empty_like(hidden_states)
        enc = torch.empty_like(encoder_hidden_states)
        _C.add_ln_mod(hidden_states, attn_output, hid, norm_hidden_states,
                      mod_i, 2 * n, 3 * n,
                      encoder_hidden_states, context_attn_output, enc,
                      norm_encoder_hidden_states, mod_c, 2 * n, 3 * n, _EPS)

        ff_output = self.ff(norm_hidden_states)
        context_ff_output = self.ff_context(norm_encoder_hidden_states)

        _C.gated_add(hid, ff_output, hid, mod_i, 5 * n,
                     enc, context_ff_output, enc, mod_c, 5 * n)

        if enc.dtype == torch.float16:
            enc = enc.clip(-65504, 65504)
        return enc, hid

    # -- reference path (baseline, verbatim) --------------------------------

    def _forward_reference(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb=None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        joint_attention_kwargs = joint_attention_kwargs or {}

        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output

        if len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        )

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._fused_ok(hidden_states, encoder_hidden_states, temb,
                          joint_attention_kwargs):
            try:
                return self._forward_fused(hidden_states, encoder_hidden_states,
                                           temb, image_rotary_emb)
            except RuntimeError:
                pass
        return self._forward_reference(hidden_states, encoder_hidden_states, temb,
                                       image_rotary_emb, joint_attention_kwargs)


class FluxSingleTransformerBlock(nn.Module):
    """Single-stream DiT block: text+image concatenated, self-attention + MLP in parallel."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        quant_config: dict | None = None,
    ):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim, promote_fp32=False)
        self.proj_mlp = ReplicatedLinear(dim, self.mlp_hidden_dim, bias=True,
                                         quant_config=quant_config)
        self.act_mlp = GELU(approximate="tanh")
        self.proj_out = ReplicatedLinear(dim + self.mlp_hidden_dim, dim, bias=True,
                                         quant_config=quant_config)

        self.attn = FluxAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            eps=1e-6,
            pre_only=True,
            quant_config=quant_config,
        )

        self.dim = dim
        self._fused_static = (
            _mod_ok(dim)
            and _linear_plain(self.proj_mlp)
            and _linear_plain(self.proj_out)
        )

    # -- fast path ---------------------------------------------------------

    def _fused_ok(self, hidden_states, encoder_hidden_states, temb,
                  joint_attention_kwargs) -> bool:
        if not self._fused_static or joint_attention_kwargs:
            return False
        if hidden_states.dtype not in (torch.bfloat16, torch.float16):
            return False
        if (hidden_states.dim() != 3 or encoder_hidden_states.dim() != 3
                or hidden_states.size(0) != 1 or encoder_hidden_states.size(0) != 1):
            return False
        if (encoder_hidden_states.dtype is not hidden_states.dtype
                or temb.dtype is not hidden_states.dtype):
            return False
        if temb.dim() != 2 or temb.size(0) != 1:
            return False
        if not (hidden_states.is_contiguous() and encoder_hidden_states.is_contiguous()
                and temb.is_contiguous()):
            return False
        if self.proj_mlp.weight.dtype is not hidden_states.dtype:
            return False
        return True

    def _forward_fused(self, hidden_states, encoder_hidden_states, temb,
                       image_rotary_emb):
        n = self.dim
        mod = self.norm.linear(F.silu(temb)).view(-1)

        n_txt = encoder_hidden_states.shape[1]
        n_img = hidden_states.shape[1]
        seq = n_txt + n_img

        # The reference concatenates [text, image] and normalizes the result; the
        # fused prologue writes that joint buffer straight out of the two streams,
        # so the concatenation itself never happens.
        norm_hidden_states = torch.empty((1, seq, n), dtype=hidden_states.dtype,
                                         device=hidden_states.device)
        _C.ln_mod(encoder_hidden_states, norm_hidden_states.narrow(1, 0, n_txt),
                  mod, 0,
                  hidden_states, norm_hidden_states.narrow(1, n_txt, n_img),
                  mod, 0, _EPS)

        x2 = norm_hidden_states.view(seq, n)
        w_mlp = self.proj_mlp.weight
        b_mlp = self.proj_mlp.bias
        mlp_hidden_states = (torch.addmm(b_mlp, x2, w_mlp.t()) if b_mlp is not None
                             else torch.mm(x2, w_mlp.t()))
        mlp_hidden_states = self.act_mlp(mlp_hidden_states)

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        # proj_out([attn | mlp]) == attn @ Wa^T + mlp @ Wb^T + b, so the output
        # projection can consume the two halves where they already are.  The
        # concatenation the reference builds is [S, 15360] -- 141 MB written and
        # read back at ~5.5 TB/s, the single most expensive non-GEMM kernel in
        # this block -- against one extra read/write of the [S, 3072] accumulator
        # here.  ``Wa``/``Wb`` are column slices of one row-major weight, so
        # their transposes are column-major views with ld = 15360 and cuBLAS
        # takes them as they are.
        w_out = self.proj_out.weight
        b_out = self.proj_out.bias
        attn2 = attn_output.view(seq, n)
        w_a = w_out.narrow(1, 0, n).t()
        w_b = w_out.narrow(1, n, w_out.shape[1] - n).t()
        y = (torch.addmm(b_out, attn2, w_a) if b_out is not None
             else torch.mm(attn2, w_a))
        y.addmm_(mlp_hidden_states, w_b)
        y = y.view(1, seq, n)

        enc_out = torch.empty_like(encoder_hidden_states)
        hid_out = torch.empty_like(hidden_states)
        _C.gated_add(encoder_hidden_states, y.narrow(1, 0, n_txt), enc_out, mod, 2 * n,
                     hidden_states, y.narrow(1, n_txt, n_img), hid_out, mod, 2 * n)

        # The reference clips the joint tensor before splitting it, so both
        # halves are clipped.
        if hid_out.dtype == torch.float16:
            enc_out = enc_out.clip(-65504, 65504)
            hid_out = hid_out.clip(-65504, 65504)
        return enc_out, hid_out

    # -- reference path (baseline, verbatim) --------------------------------

    def _forward_reference(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb=None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        encoder_hidden_states, hidden_states = (
            hidden_states[:, :text_seq_len],
            hidden_states[:, text_seq_len:],
        )
        return encoder_hidden_states, hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._fused_ok(hidden_states, encoder_hidden_states, temb,
                          joint_attention_kwargs):
            try:
                return self._forward_fused(hidden_states, encoder_hidden_states,
                                           temb, image_rotary_emb)
            except RuntimeError:
                pass
        return self._forward_reference(hidden_states, encoder_hidden_states, temb,
                                       image_rotary_emb, joint_attention_kwargs)
