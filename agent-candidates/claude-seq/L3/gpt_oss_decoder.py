"""GPT-OSS decoder layer: attention + MoE with RMSNorm residual connections.

Structurally the baseline: the shared ``LlamaAttention`` with ``use_sinks=True``
and a per-layer sliding window, two fused add+RMSNorms, and the MXFP4 MoE.

The one thing replaced is *which* expert weights the MoE reads.  The trtllm-gen
fused MoE streams all 128 expert matrices on every call -- ~1.9 GB, a flat
~450 us on this GPU, which is 60% of the whole layer -- whether the step carries
16384 tokens or one.  A decode step routes to 4..112 of them, so
:class:`_RoutedMoE` runs its own experts for narrow batches: routing, then two
grouped GEMMs that touch only the selected experts (``gpt_oss_moe_fk.cu``).
Wide batches keep the baseline path, where every expert is selected anyway and
the all-expert pass moves strictly less memory than a routed one would.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.cuda_ext import lazy_op
from ..L1.rms_norm import RMSNorm
from ..L2.attention import LlamaAttention
from ..L2.gpt_oss_moe import GptOssMoE
from ..L2.trtllm_mxfp4_moe import SWIGLU_ALPHA, SWIGLU_BETA, SWIGLU_LIMIT

_C = lazy_op("gpt_oss_moe_fk", "gpt_oss_moe_fk.cu")

# Token slots per expert tile; must match TMAX in the .cu.
_TMAX = 4
# fp4 values a lane dequantizes per step: must divide K and be a multiple of 8.
# 32 (a full 16-byte load per lane) wins once all four token slots are busy;
# below that the narrower step keeps more loads in flight.
_VPL_WIDE, _VPL_NARROW = 32, 16
# Weight rows per CTA in the expert GEMMs.  Narrow batches produce few tiles, so
# the grid (not the per-CTA work) is what limits memory-level parallelism there.
_ROWS_WIDE, _ROWS_NARROW = 32, 16
# Beyond this many tokens the routing covers nearly every expert and the
# baseline's single all-expert pass is the cheaper one.
_MAX_ROUTED_TOKENS = 12
# Upper bound the single-CTA router is compiled for (MMAX in the .cu).
_MAX_ROUTER_TOKENS = 512


class _RoutedMoE(GptOssMoE):
    """MXFP4 MoE that reads only the experts a step actually routes to.

    Parameter names, shapes and ``forward`` semantics are the base class's, so
    weight loading (and the harness' ``load_state_dict`` sharing) is unchanged;
    only the narrow-batch expert evaluation is different.
    """

    def __init__(self, config):
        super().__init__(config)
        self._routed = None
        # Scratch buffers per token count.  A decode step is ~20 us of GPU work
        # here, so six allocator round trips per call are not noise.
        self._scratch: dict[int, tuple] = {}

    def _routed_supported(self) -> bool:
        h, i = self.hidden_size, self.intermediate_per_tp
        return bool(
            self.use_trtllm and self.tp_size == 1 and self.top_k == 4
            and self.num_experts <= 256
            and h % _ROWS_WIDE == 0 and (2 * i) % _ROWS_WIDE == 0
            and h % _VPL_WIDE == 0 and i % _VPL_WIDE == 0
        )

    def process_weights_after_loading(self):
        if self._processed:
            return
        if self._routed_supported():
            # ``prepare_trtllm_mxfp4_weights`` shuffles out of place, so the
            # packed MXFP4 tensors survive; grab them before the base class
            # deletes the Parameters that own them.
            self._routed = (
                self.w13_weight.data, self.w13_weight_scale.data,
                self.w13_bias.data.float().contiguous(),
                self.w2_weight.data, self.w2_weight_scale.data,
                self.w2_bias.data.float().contiguous(),
            )
        super().process_weights_after_loading()

    def _make_scratch(self, m: int, dev) -> tuple:
        h, i = self.hidden_size, self.intermediate_per_tp
        i32 = torch.int32
        # Tiles are bounded by "one per pair" and by "one per active expert plus
        # one per full tile", so the grid is sized without a device sync.
        p = 4 * m
        ntile = min(p, min(self.num_experts, p) + (p + _TMAX - 1) // _TMAX)
        nt = 1 if m == 1 else (2 if m == 2 else _TMAX)
        return (
            ntile, nt, _VPL_WIDE if nt == _TMAX else _VPL_NARROW,
            _ROWS_NARROW if m <= 8 else _ROWS_WIDE,
            torch.empty(m, 4, device=dev, dtype=i32),
            torch.empty(m, 4, device=dev, dtype=torch.float32),
            torch.empty(ntile * _TMAX, device=dev, dtype=i32),
            torch.empty(ntile, device=dev, dtype=i32),
            torch.empty(ntile, device=dev, dtype=i32),
            torch.empty(1, device=dev, dtype=i32),
            torch.empty(p, i, device=dev, dtype=torch.float16),
            torch.empty(p, h, device=dev, dtype=torch.float32),
        )

    def _routed_experts(self, hs: torch.Tensor) -> torch.Tensor:
        w13, w13s, w13b, w2, w2s, w2b = self._routed
        m = hs.shape[0]
        sc = self._scratch.get(m)
        if sc is None:
            sc = self._scratch[m] = self._make_scratch(m, hs.device)
        ntile, nt, vpl, rows, idx, wgt, sid, tile_e, tile_nv, total, y1, y2 = sc
        # The output is *not* cached: callers keep it, so it must not alias the
        # next call's result.
        out = torch.empty_like(hs)
        logits = torch.nn.functional.linear(hs, self.router.weight,
                                            self.router.bias)
        _C.moe_route(logits, idx, wgt, sid, tile_e, tile_nv, total)
        _C.moe_gemm1(hs, w13, w13s, w13b, y1, sid, tile_e, tile_nv, total,
                     ntile, self.intermediate_per_tp, nt, vpl, rows,
                     SWIGLU_ALPHA, SWIGLU_BETA, SWIGLU_LIMIT)
        _C.moe_gemm2(y1, w2, w2s, w2b, wgt, y2, sid, tile_e, tile_nv, total,
                     ntile, nt, vpl, rows)
        _C.moe_combine(y2, out)
        return out

    def forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self._processed:
            self.process_weights_after_loading()
        if self._routed is not None:
            shape = hidden_states.shape
            hs = hidden_states.reshape(-1, self.hidden_size)
            m = hs.shape[0]
            if (m <= _MAX_ROUTED_TOKENS and m <= _MAX_ROUTER_TOKENS
                    and hs.dtype == torch.bfloat16 and hs.is_contiguous()):
                return self._routed_experts(hs).view(shape)
        return super().forward_impl(hidden_states)


class GptOssDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = LlamaAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            bias=True,
            o_proj_bias=True,
            use_sinks=True,
            sliding_window=config.sliding_window,
            layer_idx=layer_idx,
        )
        self.mlp = _RoutedMoE(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual, rotary_emb):
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, rotary_emb=rotary_emb)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
