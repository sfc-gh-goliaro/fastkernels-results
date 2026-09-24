"""GPT-OSS MoE: hand-written fused MXFP4 expert kernel (SM100).

The small/medium-token path -- which is where gpt-oss actually spends its MoE
time (decode batches of 1..~400 tokens) -- is owned end to end by
``moe_mxfp4.cu``: router GEMV, top-k + softmax renormalize, expert assignment,
the interleaved gate/up projection with OAI SwiGLU and the down projection with
router-weight combine are four launches, with the packed MXFP4 weights
dequantized in-register straight into ``mma.sync`` fragments.  Each activated
expert's weight bytes are read from HBM exactly once per forward and never
materialized as bf16.

The router GEMV runs through the tensor core as well above M=64 (a bf16
``mma.sync`` over 16-token tiles, ``k_router_mma``); below that the router is
bound by how many CTAs are pulling its 737 KB of weights rather than by
arithmetic, so the 4-token-tile FMA version keeps the wider grid.

Only the large-M (prefill) branch still delegates to the trtllm-gen kernel; at
M >> intermediate/top_k the problem is MMA-throughput-bound rather than
bandwidth-bound and the local kernel has no advantage there (see ITERATIONS.md).

Weight layout (built once in ``process_weights_after_loading``):

* packed values ``[E, N/8, K/64, 32, 8]`` uint8 -- one 8-byte per-lane load
  covers the four ``m16n8k16`` B-fragments of a 64-wide k group, so the tensor
  core is fed with no shuffles, no ``ldmatrix`` and no SMEM staging of weights.
* block scales ``[E, N/8, K/64, 8, 2]`` uint8, stored as the *relative* fp16
  exponent ``15 + (e_blk - e_row)``; folding a power of two into the fp16 B
  operand is exact, so the inner loop never rescales the accumulator.  The
  per-row reference ``2^(e_row-127)`` is applied once in the epilogue.
* router weights ``[E/8, K/16, 32, 4]`` bf16 -- the ``m16n8k16`` B-fragment order
  for ``k_router_mma``, a pure permute of the k axis (same 737 KB).

The activation tile the expert GEMMs read from shared memory is stored in
mma-fragment order (16 bytes per (k-tile, lane), exactly the register quad
``mma.sync`` wants), so a k-tile's A-fragment is one ``LDS.128`` with no operand
shuffling: 6.45 instructions per HMMA instead of 7.05.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

from ....infra.tp import _tp_rank, _tp_size
from ..L1.allreduce import AllReduce
from ..L1.linear import Linear
from .mxfp4_moe import Mxfp4MoE
from .trtllm_mxfp4_moe import (
    TRTLLM_MXFP4_ALIGN,
    TrtLlmMxfp4MoE,
    prepare_trtllm_mxfp4_weights,
    trtllm_mxfp4_moe_supported,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXT = None
_EXT_FAILED = False

# Largest token count handled by the local kernel.  Past ~576 tokens the average
# expert holds more than CHUNK rows, so the weights get read once per token chunk
# and the trtllm-gen delegate (which is MMA- rather than bandwidth-bound there)
# wins; measured crossover is M=576 (1.01x), so cut over at 512 (1.36x).
MAX_LOCAL_M = 512

SWIGLU_ALPHA = 1.702
SWIGLU_LIMIT = 7.0


def _load_ext():
    """JIT-build ``moe_mxfp4.cu`` (cached in ~/.cache/torch_extensions)."""
    global _EXT, _EXT_FAILED
    if _EXT is not None or _EXT_FAILED:
        return _EXT
    try:
        import hashlib
        from torch.utils.cpp_extension import load
        flags = os.environ.get("FK_MOE_FLAGS", "")
        tag = ("_" + hashlib.md5(flags.encode()).hexdigest()[:8]) if flags else ""
        _EXT = load(
            name="fk_gptoss_moe_mxfp4_v1" + tag,
            sources=[os.path.join(_HERE, "moe_mxfp4.cu")],
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3", "--use_fast_math", "-lineinfo",
                "-gencode=arch=compute_100a,code=sm_100a",
            ] + flags.split(),
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001 - fall back to the delegate path
        print(f"[gpt_oss_moe] local MXFP4 kernel unavailable: {exc!r}", file=sys.stderr)
        _EXT_FAILED = True
    return _EXT


def _round_up(x: int, align: int) -> int:
    return (x + align - 1) // align * align


# Output rows per expert-GEMM CTA: WARPS * NTBIG * 8 for the wide tiling and
# WARPS * 8 for the narrow one, so the packed layout's row count must be a
# multiple of the former (moe_mxfp4.cu rejects the ctx otherwise and we fall
# back to the delegate).
ROWS_CTA = 320


def _pack_mxfp4(w: torch.Tensor, s: torch.Tensor):
    """Permute packed MXFP4 values + E8M0 scales into the kernel's layout.

    ``w`` is ``[E, N, K/2]`` uint8 (2 fp4 per byte, k-major), ``s`` is
    ``[E, N, K/32]`` uint8 (E8M0).  Returns ``(values, rel_scales, row_factor)``.

    The value permutation is exactly the byte order lane ``l`` of an
    ``m16n8k16`` warp needs for output row ``l>>2``: within a 64-wide k group the
    lane's four B-fragments live at source bytes ``8c + (l&3) + 4p``, which is
    the row-major decomposition ``[K/64][4 c][2 p][4 tig]`` -- hence a pure
    reshape/permute rather than a gather.
    """
    E, N, Kh = w.shape
    K = Kh * 2
    pad = (-N) % ROWS_CTA
    if pad:
        w = torch.cat([w, w.new_zeros(E, pad, Kh)], dim=1)
        s = torch.cat([s, s.new_zeros(E, pad, s.shape[2])], dim=1)
        N += pad
    NT1, KGN = N // 8, K // 64
    vals = (w.view(E, NT1, 8, KGN, 4, 2, 4)
             .permute(0, 1, 3, 2, 6, 4, 5)
             .reshape(E, NT1, KGN, 32, 8)
             .contiguous())
    e_row = s.view(E, N, -1).max(dim=2).values                      # [E, N]
    # pre-shifted by 2 so one PRMT builds the fp16x2 power of two in-kernel
    rel = (((s.int() - e_row[:, :, None].int() + 15).clamp_(0, 30)) << 2).to(torch.uint8)
    rel = (rel.view(E, NT1, 8, KGN, 2)
              .permute(0, 1, 3, 2, 4)
              .reshape(E, NT1, KGN, 16)
              .contiguous())
    fac = torch.ldexp(torch.ones_like(e_row, dtype=torch.float64),
                      e_row.long() - 127).to(torch.float32)
    return vals, rel, fac, N


class GptOssMoE(nn.Module):
    """MXFP4-native MoE with a local fused expert kernel for small/medium M."""

    MXFP4_BLOCK = 32

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        tp = _tp_size()
        self.tp_size = tp
        self.intermediate_per_tp = config.intermediate_size // tp

        self.router = Linear(config.hidden_size, config.num_local_experts, bias=True)

        E = config.num_local_experts
        BLK = self.MXFP4_BLOCK

        self.use_trtllm = trtllm_mxfp4_moe_supported()
        if self.use_trtllm:
            I_pad = _round_up(self.intermediate_per_tp, TRTLLM_MXFP4_ALIGN)
            H_pad = _round_up(self.hidden_size, TRTLLM_MXFP4_ALIGN)
        else:
            I_pad = _round_up(self.intermediate_per_tp, 64)
            H_pad = self.hidden_size
        H = H_pad

        self._I_pad = I_pad
        self._H_pad = H_pad

        self.w13_weight = nn.Parameter(
            torch.zeros(E, 2 * I_pad, H // 2, dtype=torch.uint8), requires_grad=False)
        self.w13_weight_scale = nn.Parameter(
            torch.zeros(E, 2 * I_pad, H // BLK, dtype=torch.uint8), requires_grad=False)
        self.w13_bias = nn.Parameter(
            torch.zeros(E, 2 * I_pad, dtype=torch.bfloat16), requires_grad=False)

        self.w2_weight = nn.Parameter(
            torch.zeros(E, H, I_pad // 2, dtype=torch.uint8), requires_grad=False)
        self.w2_weight_scale = nn.Parameter(
            torch.zeros(E, H, I_pad // BLK, dtype=torch.uint8), requires_grad=False)
        self.w2_bias = nn.Parameter(
            torch.zeros(E, H, dtype=torch.bfloat16), requires_grad=False)

        self.w13_weight.weight_loader = self._w13_weight_loader
        self.w13_weight_scale.weight_loader = self._w13_scale_loader
        self.w13_bias.weight_loader = self._w13_bias_loader
        self.w2_weight.weight_loader = self._w2_weight_loader
        self.w2_weight_scale.weight_loader = self._w2_scale_loader
        self.w2_bias.weight_loader = self._w2_bias_loader

        self.allreduce = AllReduce()
        self.mxfp4_moe = Mxfp4MoE()
        self.trtllm_moe = (
            TrtLlmMxfp4MoE(
                num_experts=E,
                top_k=self.top_k,
                intermediate_size=I_pad,
                hidden_size_unpadded=self.hidden_size,
            )
            if self.use_trtllm
            else None
        )

        self._quant_config = None
        self._processed = False
        self._ctx = 0
        self._local = None
        self._has_local = False
        self._ext = None

        self._use_custom_op = False
        self._layer_name = ""

    # ------------------------------------------------------------- loaders
    def _w13_weight_loader(self, param, loaded_weight):
        if loaded_weight.ndim == 4:
            E, N, nb, bs = loaded_weight.shape
            loaded_weight = loaded_weight.reshape(E, N, nb * bs)
        rank = _tp_rank()
        I = self.intermediate_per_tp
        start = 2 * rank * I
        k = loaded_weight.shape[-1]
        param.data[:, :2*I, :k].copy_(loaded_weight[:, start : start + 2*I, :])

    def _w13_scale_loader(self, param, loaded_weight):
        rank = _tp_rank()
        I = self.intermediate_per_tp
        start = 2 * rank * I
        k = loaded_weight.shape[-1]
        param.data[:, :2*I, :k].copy_(loaded_weight[:, start : start + 2*I, :])

    def _w13_bias_loader(self, param, loaded_weight):
        rank = _tp_rank()
        I = self.intermediate_per_tp
        start = 2 * rank * I
        param.data[:, :2*I].copy_(loaded_weight[:, start : start + 2*I])

    def _w2_weight_loader(self, param, loaded_weight):
        if loaded_weight.ndim == 4:
            E, H, nb, bs = loaded_weight.shape
            loaded_weight = loaded_weight.reshape(E, H, nb * bs)
        tp, rank = _tp_size(), _tp_rank()
        I_half = self.intermediate_per_tp // 2
        h = loaded_weight.shape[1]
        param.data[:, :h, :I_half].copy_(
            loaded_weight[:, :, rank * I_half : rank * I_half + I_half])

    def _w2_scale_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        I_blk = self.intermediate_per_tp // self.MXFP4_BLOCK
        h = loaded_weight.shape[1]
        param.data[:, :h, :I_blk].copy_(
            loaded_weight[:, :, rank * I_blk : rank * I_blk + I_blk])

    def _w2_bias_loader(self, param, loaded_weight):
        if _tp_rank() == 0:
            param.data[:, : loaded_weight.shape[1]].copy_(loaded_weight)
        else:
            param.data.zero_()

    # --------------------------------------------------------- weight prep
    def _prepare_local(self):
        """Build the local kernel's weight layout from the loaded params.

        Any failure here (no compiler, unsupported shape, out of memory) leaves
        ``_has_local`` False and the module falls back to the delegate path, so
        the op stays correct on machines this kernel was not written for.
        """
        ext = _load_ext()
        if ext is None:
            return False
        try:
            return self._prepare_local_inner(ext)
        except Exception as exc:  # noqa: BLE001
            print(f"[gpt_oss_moe] local weight prep failed: {exc!r}", file=sys.stderr)
            self._local = None
            return False

    def _prepare_local_inner(self, ext):
        H, I, E = self.hidden_size, self.intermediate_per_tp, self.num_experts
        # The k axis of both expert GEMMs is tiled 64 wide; anything else would
        # need a zero-padded k tail, which is not worth carrying for shapes this
        # task never sees (gpt-oss at tp=1 is 2880/2880).  Fall back otherwise.
        if H % 64 or I % 64 or self.top_k != 4:
            raise RuntimeError(f"unsupported dims H={H} I={I} top_k={self.top_k}")
        BLK = self.MXFP4_BLOCK
        w13 = self.w13_weight.data[:, : 2 * I, : H // 2].contiguous()
        s13 = self.w13_weight_scale.data[:, : 2 * I, : H // BLK].contiguous()
        v13, r13, f13, rows13 = _pack_mxfp4(w13, s13)
        del w13, s13
        w2 = self.w2_weight.data[:, :H, : I // 2].contiguous()
        s2 = self.w2_weight_scale.data[:, :H, : I // BLK].contiguous()
        v2, r2, f2, rows2 = _pack_mxfp4(w2, s2)
        del w2, s2
        b13 = torch.zeros(E, rows13, dtype=torch.float32, device=f13.device)
        b13[:, : 2 * I] = self.w13_bias.data[:, : 2 * I].float()
        b2 = torch.zeros(E, rows2, dtype=torch.float32, device=f2.device)
        b2[:, :H] = self.w2_bias.data[:, :H].float()
        rw = self.router.weight.data.to(torch.bfloat16).contiguous()
        rb = self.router.bias.data.to(torch.bfloat16).contiguous()
        # Router weights in mma B-fragment order: for expert group g, k-tile T and
        # lane l = 4*nl + tig, the four bf16 at [g][T][l] are
        # rw[8g+nl][16T + 8*hi + 2*tig + lo] with (hi, lo) the register index --
        # a pure permute of the k axis decomposition [T][hi][tig][lo].
        rwf = (rw.view(E // 8, 8, H // 16, 2, 4, 2)
                 .permute(0, 2, 1, 4, 3, 5)
                 .reshape(E // 8, H // 16, 32, 4)
                 .contiguous())
        self._local = (v13, r13, f13, b13, v2, r2, f2, b2, rw, rwf, rb)
        torch.cuda.empty_cache()
        self._ctx = ext.make_ctx(v13.view(-1), r13.view(-1), b13, f13,
                                 v2.view(-1), r2.view(-1), b2, f2, rw, rwf, rb,
                                 H, I, E, self.top_k, MAX_LOCAL_M,
                                 rows13, rows2, SWIGLU_ALPHA, SWIGLU_LIMIT)
        if self._ctx == 0:
            raise RuntimeError("kernel rejected these dimensions")
        self._ext = ext
        return True

    def process_weights_after_loading(self):
        if self._processed:
            return

        has_local = self._prepare_local()

        if self.use_trtllm:
            (
                w13_weight, w13_scale, w13_bias,
                w2_weight, w2_scale, w2_bias,
            ) = prepare_trtllm_mxfp4_weights(
                self.w13_weight.data,
                self.w13_weight_scale.data,
                self.w13_bias.data,
                self.w2_weight.data,
                self.w2_weight_scale.data,
                self.w2_bias.data,
            )
            del self.w13_weight, self.w2_weight
            del self.w13_weight_scale, self.w2_weight_scale
            del self.w13_bias, self.w2_bias
            self._w13_shuffled = w13_weight
            self._w13_scale = w13_scale
            self._w13_bias_f32 = w13_bias
            self._w2_shuffled = w2_weight
            self._w2_scale = w2_scale
            self._w2_bias_f32 = w2_bias
            torch.cuda.empty_cache()
            self._processed = True
            self._has_local = has_local
            return

        self.w13_bias.data = self.w13_bias.data.float()
        self.w2_bias.data = self.w2_bias.data.float()
        w13_weight, w13_precision = Mxfp4MoE.prepare_weight(
            self.w13_weight.data, self.w13_weight_scale.data)
        w2_weight, w2_precision = Mxfp4MoE.prepare_weight(
            self.w2_weight.data, self.w2_weight_scale.data)
        del self.w13_weight, self.w2_weight
        del self.w13_weight_scale, self.w2_weight_scale
        self._w13_swizzled = w13_weight
        self._w2_swizzled = w2_weight
        self._quant_config = Mxfp4MoE.make_quant_config(
            w1_precision=w13_precision, w2_precision=w2_precision,
            w1_bias=self.w13_bias.data, w2_bias=self.w2_bias.data)
        self._processed = True
        self._has_local = has_local

    # ------------------------------------------------------------- forward
    def forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self._processed:
            self.process_weights_after_loading()

        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)

        if self._has_local and hidden_states.shape[0] <= MAX_LOCAL_M:
            output = self._ext.forward(self._ctx, hidden_states)
            if self.tp_size > 1 and not self._use_custom_op:
                output = self.allreduce(output)
            return output.view(orig_shape)

        router_logits = self.router(hidden_states)
        if self.use_trtllm:
            if self._H_pad != self.hidden_size:
                hidden_states = torch.nn.functional.pad(
                    hidden_states, (0, self._H_pad - self.hidden_size))
            output = self.trtllm_moe(
                hidden_states, router_logits,
                self._w13_shuffled, self._w13_scale, self._w13_bias_f32,
                self._w2_shuffled, self._w2_scale, self._w2_bias_f32)
        else:
            output = self.mxfp4_moe(
                hidden_states=hidden_states,
                w1=self._w13_swizzled, w2=self._w2_swizzled,
                gating_output=router_logits, topk=self.top_k, renormalize=True,
                quant_config=self._quant_config, apply_router_weight_on_input=False)

        if self.tp_size > 1 and not self._use_custom_op:
            output = self.allreduce(output)
        return output.view(orig_shape)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._use_custom_op:
            out = torch.ops.fastkernels.moe_forward(hidden_states, self._layer_name)
            if self.tp_size > 1:
                out = self.allreduce(out)
            return out
        return self.forward_impl(hidden_states)
