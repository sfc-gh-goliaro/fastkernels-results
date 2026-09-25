from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import triton
import triton.language as tl
from einops import rearrange

from ..L1.rms_norm_gated import RMSNormGated
from ..L1.kda import (
    chunk_kda_with_fused_gate,
    fused_kda_gate,
    fused_recurrent_kda,
)
from ..L1.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from ...baseline.L2.kimi_delta_attention import (
    KimiDeltaAttention as _BaselineKimiDeltaAttention,
)


def set_triton_allocator(device: torch.device):
    """Verbatim from vLLM's ``triton_utils.allocation.set_triton_allocator``."""

    def alloc_fn(size: int, alignment: int, stream: int | None):
        return torch.empty(size, device=device, dtype=torch.int8)

    triton.set_allocator(alloc_fn)


@triton.jit
def _rms_norm_sigmoid_gate_kernel(
    x,
    gate,
    weight,
    out,
    n_rows,
    eps,
    ROWS: tl.constexpr,
    N: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, N)
    mask = rows[:, None] < n_rows
    offsets = rows[:, None] * N + cols[None, :]
    x_values = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x_values * x_values, axis=1) / N
    rstd = tl.rsqrt(variance + eps)
    w = tl.load(weight + cols).to(tl.float32)
    g = tl.load(gate + offsets, mask=mask, other=0.0).to(tl.float32)
    y = x_values * rstd[:, None] * w[None, :] * tl.sigmoid(g)
    tl.store(out + offsets, y, mask=mask)


class _FusedRMSNormSigmoidGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        n_rows = x.numel() // self.hidden_size
        rows = 1 if n_rows < 256 else 2 if n_rows < 1024 else 8
        _rms_norm_sigmoid_gate_kernel[(triton.cdiv(n_rows, rows),)](
            x,
            gate,
            self.weight,
            out,
            n_rows,
            self.eps,
            ROWS=rows,
            N=self.hidden_size,
            num_warps=1,
        )
        return out


@triton.jit
def _beta_sigmoid_kernel(
    raw_beta,
    beta,
    n_tokens,
    stride_token,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    heads = tl.arange(0, HEADS)
    offsets = rows[:, None] * stride_token + heads[None, :]
    values = tl.load(
        raw_beta + offsets,
        mask=rows[:, None] < n_tokens,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        beta + rows[:, None] * HEADS + heads[None, :],
        tl.sigmoid(values),
        mask=rows[:, None] < n_tokens,
    )


@triton.jit
def _grouped_gate_up_kernel(
    f_a,
    g_a,
    f_weight,
    g_weight,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_f_m,
    stride_g_m,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    group_width = 8 * num_pid_n
    group_id = pid // group_width
    first_pid_m = group_id * 8
    group_size_m = min(num_pid_m - first_pid_m, 8)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % group_width) // group_size_m

    a_ptr = f_a
    weight_ptr = f_weight
    stride_a_m = stride_f_m
    if group == 1:
        a_ptr = g_a
        weight_ptr = g_weight
        stride_a_m = stride_g_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k_start in tl.static_range(0, K, BLOCK_K):
        a = tl.load(
            a_ptr
            + offs_m[:, None] * stride_a_m
            + (k_start + offs_k)[None, :],
            mask=(offs_m[:, None] < M) & (k_start + offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            weight_ptr
            + offs_n[None, :] * K
            + (k_start + offs_k)[:, None],
            mask=(offs_n[None, :] < N) & (k_start + offs_k[:, None] < K),
            other=0.0,
        )
        acc = tl.dot(a, b, acc)
    tl.store(
        out + group * M * N + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _fused_norm_output_projection_kernel(
    x,
    gate,
    norm_weight,
    proj_weight,
    out,
    eps,
    M: tl.constexpr,
    N: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, HEAD_DIM)
    row_mask = rows < M
    col_mask = cols < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    norm_w = tl.load(norm_weight + dims).to(tl.float32)
    for head in tl.static_range(HEADS):
        features = head * HEAD_DIM + dims
        x_values = tl.load(
            x + rows[:, None] * (HEADS * HEAD_DIM) + features[None, :],
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        gate_values = tl.load(
            gate + rows[:, None] * (HEADS * HEAD_DIM) + features[None, :],
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        variance = tl.sum(x_values * x_values, axis=1) / HEAD_DIM
        a = (
            x_values
            * tl.rsqrt(variance[:, None] + eps)
            * norm_w[None, :]
            * tl.sigmoid(gate_values)
        ).to(tl.bfloat16)
        b = tl.load(
            proj_weight
            + cols[None, :] * (HEADS * HEAD_DIM)
            + features[:, None],
            mask=col_mask[None, :],
            other=0.0,
        )
        acc = tl.dot(a, b, acc)
    tl.store(
        out + rows[:, None] * N + cols[None, :],
        acc,
        mask=row_mask[:, None] & col_mask[None, :],
    )


@triton.jit
def _fused_short_kda_recurrent_kernel(
    q,
    k,
    v,
    raw_g,
    beta,
    A_log,
    dt_bias,
    state,
    state_indices,
    out,
    T,
    SCALE: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    BV: tl.constexpr,
):
    v_block = tl.program_id(0)
    head = tl.program_id(1)
    dims = tl.arange(0, D)
    v_dims = v_block * BV + tl.arange(0, BV)
    v_mask = v_dims < D
    state_values = tl.zeros((BV, D), tl.float32)
    a = -tl.exp(tl.load(A_log + head).to(tl.float32))
    bias = tl.load(dt_bias + head * D + dims).to(tl.float32)

    for token in range(0, T):
        feature_base = (token * HEADS + head) * D
        q_values = tl.load(q + feature_base + dims).to(tl.float32)
        k_values = tl.load(k + feature_base + dims).to(tl.float32)
        v_values = tl.load(
            v + feature_base + v_dims,
            mask=v_mask,
            other=0.0,
        ).to(tl.float32)
        gate_input = (
            tl.load(raw_g + feature_base + dims).to(tl.float32) + bias
        )
        softplus = tl.where(
            gate_input > 20.0,
            gate_input,
            tl.log(1.0 + tl.exp(gate_input)),
        )
        gate_values = a * softplus

        q_values *= tl.rsqrt(tl.sum(q_values * q_values) + 1e-6)
        k_values *= tl.rsqrt(tl.sum(k_values * k_values) + 1e-6)
        state_values *= tl.exp(gate_values)[None, :]
        residual = v_values - tl.sum(
            state_values * k_values[None, :],
            axis=1,
        )
        beta_value = tl.load(beta + token * HEADS + head).to(tl.float32)
        residual *= beta_value
        state_values += residual[:, None] * k_values[None, :]
        output = tl.sum(
            state_values * (q_values * SCALE)[None, :],
            axis=1,
        )
        tl.store(
            out + feature_base + v_dims,
            output,
            mask=v_mask,
        )

    state_idx = tl.load(state_indices).to(tl.int64)
    state_offset = (
        (state_idx * HEADS + head) * D * D
        + v_dims[:, None] * D
        + dims[None, :]
    )
    tl.store(state + state_offset, state_values, mask=v_mask[:, None])


@triton.jit
def _fused_qkv_causal_conv1d_prefill_kernel(
    q,
    k,
    v,
    q_weight,
    k_weight,
    v_weight,
    q_state,
    k_state,
    v_state,
    state_indices,
    out,
    n_tokens,
    stride_q_token,
    stride_k_token,
    stride_v_token,
    stride_state_seq,
    stride_state_token,
    stride_out_group,
    PROJ: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    token_block = tl.program_id(0)
    feat_block = tl.program_id(1)
    group = tl.program_id(2)

    x = q
    weight = q_weight
    state = q_state
    stride_x_token = stride_q_token
    if group == 1:
        x = k
        weight = k_weight
        state = k_state
        stride_x_token = stride_k_token
    elif group == 2:
        x = v
        weight = v_weight
        state = v_state
        stride_x_token = stride_v_token

    tokens = token_block * BLOCK_M + tl.arange(0, BLOCK_M)
    feats = feat_block * BLOCK_N + tl.arange(0, BLOCK_N)
    feat_mask = feats < PROJ
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for j in tl.static_range(4):
        source_tokens = tokens + j - 3
        values = tl.load(
            x + source_tokens[:, None] * stride_x_token + feats[None, :],
            mask=(
                (source_tokens[:, None] >= 0)
                & (source_tokens[:, None] < n_tokens)
                & feat_mask[None, :]
            ),
            other=0.0,
        )
        w = tl.load(weight + feats * 4 + j, mask=feat_mask, other=0.0)
        acc += values * w[None, :]

    acc = acc / (1.0 + tl.exp(-acc))
    tl.store(
        out
        + group * stride_out_group
        + tokens[:, None] * PROJ
        + feats[None, :],
        acc,
        mask=(tokens[:, None] < n_tokens) & feat_mask[None, :],
    )

    if token_block == 0:
        state_idx = tl.load(state_indices).to(tl.int64)
        state_tokens = tl.arange(0, 4)
        state_source_tokens = n_tokens - 3 + state_tokens
        state_values = tl.load(
            x + state_source_tokens[:, None] * stride_x_token + feats[None, :],
            mask=(
                (state_tokens[:, None] < 3)
                & (state_source_tokens[:, None] >= 0)
                & feat_mask[None, :]
            ),
            other=0.0,
        )
        tl.store(
            state
            + state_idx * stride_state_seq
            + state_tokens[:, None] * stride_state_token
            + feats[None, :],
            state_values,
            mask=(state_tokens[:, None] < 3) & feat_mask[None, :],
        )


from ....infra.context import get_context
from ....infra.tp import _tp_rank, _tp_size
from .parallel_linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)


class _Conv1DWeights(nn.Module):
    """Sharded depthwise-conv weight holder with HF-compatible parameter names."""

    def __init__(self, output_size: int, kernel_size: int):
        super().__init__()
        tp = _tp_size()
        assert output_size % tp == 0
        self.output_size_per_partition = output_size // tp
        self.weight = nn.Parameter(
            torch.empty(
                self.output_size_per_partition,
                1,
                kernel_size,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.weight.weight_loader = self._weight_loader
        self.bias = None

    def _weight_loader(self, param, loaded_weight):
        shard = param.data.size(0)
        rank = _tp_rank()
        param.data.copy_(
            loaded_weight.narrow(0, rank * shard, shard).to(torch.float32),
        )


@dataclass
class _KDAStateView:
    q_conv_state: torch.Tensor
    k_conv_state: torch.Tensor
    v_conv_state: torch.Tensor
    recurrent_state: torch.Tensor


class KimiDeltaAttention(_BaselineKimiDeltaAttention):
    """Kimi Linear's KDA layer.

    Uses vLLM/FLA kernels for the gate and the gated delta attention core,
    while reading runtime state + metadata from fastkernels's global Context.
    """

    def __init__(self, config, layer_idx: int, quant_config: dict | None = None):
        nn.Module.__init__(self)
        self.tp_size = _tp_size()
        self.hidden_size = config.hidden_size
        kda_config = config.linear_attn_config
        self.head_dim = kda_config["head_dim"]
        self.num_heads = kda_config["num_heads"]
        self.layer_idx = layer_idx
        self.conv_size = kda_config["short_conv_kernel_size"]
        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = self.num_heads // self.tp_size

        projection_size = self.head_dim * self.num_heads

        # q/k/v/b are four ColumnParallelLinear GEMMs over the same
        # hidden_states, so at decode width they are four launches of a GEMV-shaped
        # kernel. Profiled at tp=2 bs=1 they are the
        # ``nvjet_sm100_tst_16x64_64x16_4x1_v_bz_TNN`` at 99.2 calls/step x 7.0 us =
        # 694 us of a 4.06 ms step (16.8%) -- a 16-row tile is the compiler telling
        # us there is not enough work per launch. One [3*proj + num_heads] GEMM does
        # the same math in one launch with a tile that fits.
        #
        # Only the loader needs care: ColumnParallelLinear shards its output, and
        # rank r needs *its own shard of each sub-projection* laid out end to end --
        # not the r-th contiguous slice of the concatenation. ``_qkvb_weight_loader``
        # places each one at its local offset. Kept separate under quantization,
        # where the block scales would have to be concatenated too.
        self.qkvb_proj = ColumnParallelLinear(
            self.hidden_size,
            3 * projection_size + self.num_heads,
            bias=False,
            quant_config=None,
        ) if quant_config is None else None
        if self.qkvb_proj is not None:
            _ps_local = projection_size // self.tp_size
            _nh_local = self.num_heads // self.tp_size
            self._ps_local = _ps_local
            self._nh_local = _nh_local

            def _qkvb_weight_loader(param, loaded_weight, shard_id):
                # shard_id 0/1/2/3 = q/k/v/b (see packed_modules_mapping in
                # L4/kimi_linear.py). b_proj is num_heads wide, the others
                # projection_size.
                rank = _tp_rank()
                if shard_id == 3:
                    local, offset = _nh_local, 3 * _ps_local
                else:
                    local, offset = _ps_local, shard_id * _ps_local
                shard = loaded_weight.narrow(0, rank * local, local)
                param.data[offset:offset + local].copy_(shard)
                if hasattr(self, "input_proj"):
                    self.input_proj.weight.data[offset:offset + local].copy_(shard)

            self.qkvb_proj.weight.weight_loader = _qkvb_weight_loader

        self.q_proj = ColumnParallelLinear(
            self.hidden_size, projection_size, bias=False,
            quant_config=quant_config,
        ) if quant_config is not None else None
        self.k_proj = ColumnParallelLinear(
            self.hidden_size, projection_size, bias=False,
            quant_config=quant_config,
        ) if quant_config is not None else None
        self.v_proj = ColumnParallelLinear(
            self.hidden_size, projection_size, bias=False,
            quant_config=quant_config,
        ) if quant_config is not None else None

        self.f_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
        )
        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=quant_config,
        )
        self.dt_bias = nn.Parameter(
            torch.empty(projection_size // self.tp_size, dtype=torch.float32),
        )
        self.dt_bias.weight_loader = self._shard0_loader

        self.b_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads,
            bias=False,
            quant_config=quant_config,
        ) if quant_config is not None else None

        self.q_conv1d = _Conv1DWeights(projection_size, self.conv_size)
        self.k_conv1d = _Conv1DWeights(projection_size, self.conv_size)
        self.v_conv1d = _Conv1DWeights(projection_size, self.conv_size)

        self.A_log = nn.Parameter(
            torch.empty(1, 1, self.local_num_heads, 1, dtype=torch.float32),
        )
        self.A_log.weight_loader = self._a_log_loader

        self.g_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
        )
        self.g_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=quant_config,
        )
        self.input_proj = None
        if quant_config is None:
            qkvb_local = 3 * self._ps_local + self._nh_local
            packed_output_size = self.tp_size * (
                qkvb_local + 2 * self.head_dim
            )
            self.input_proj = ColumnParallelLinear(
                self.hidden_size,
                packed_output_size,
                bias=False,
                quant_config=None,
            )

            def _f_a_weight_loader(param, loaded_weight):
                param.data.copy_(loaded_weight)
                self.input_proj.weight.data[
                    qkvb_local:qkvb_local + self.head_dim
                ].copy_(loaded_weight)

            def _g_a_weight_loader(param, loaded_weight):
                param.data.copy_(loaded_weight)
                self.input_proj.weight.data[
                    qkvb_local + self.head_dim:
                    qkvb_local + 2 * self.head_dim
                ].copy_(loaded_weight)

            self.f_a_proj.weight.weight_loader = _f_a_weight_loader
            self.g_a_proj.weight.weight_loader = _g_a_weight_loader

        self.o_norm = _FusedRMSNormSigmoidGated(
            self.head_dim,
            eps=config.rms_norm_eps,
        )
        self.o_proj = RowParallelLinear(
            projection_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
        )
        self._triton_allocator_ready = False
        self._use_custom_op = False
        self._layer_name = ""

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        result = nn.Module.load_state_dict(
            self, state_dict, strict=strict, assign=assign,
        )
        if self.input_proj is not None:
            qkvb_local = 3 * self._ps_local + self._nh_local
            with torch.no_grad():
                packed = self.input_proj.weight
                packed[:qkvb_local].copy_(self.qkvb_proj.weight)
                packed[qkvb_local:qkvb_local + self.head_dim].copy_(
                    self.f_a_proj.weight,
                )
                packed[
                    qkvb_local + self.head_dim:qkvb_local + 2 * self.head_dim
                ].copy_(self.g_a_proj.weight)
        return result

    @staticmethod
    def _shard0_loader(param, loaded_weight):
        shard = param.data.size(0)
        rank = _tp_rank()
        param.data.copy_(loaded_weight.narrow(0, rank * shard, shard).to(param.dtype))

    @staticmethod
    def _a_log_loader(param, loaded_weight):
        rank = _tp_rank()
        tp = _tp_size()
        shard = param.data.shape[2]
        param.data.copy_(
            loaded_weight.narrow(2, rank * shard, shard).to(param.dtype),
        )

    def _get_state(self) -> tuple[_KDAStateView | None, object | None]:
        ctx = get_context()
        kda_state = getattr(ctx, "kda_state", None)
        kda_meta = getattr(ctx, "kda_metadata", None)
        if kda_state is None or kda_meta is None:
            return None, None
        return _KDAStateView(
            q_conv_state=kda_state.q_conv_states[self.layer_idx],
            k_conv_state=kda_state.k_conv_states[self.layer_idx],
            v_conv_state=kda_state.v_conv_states[self.layer_idx],
            recurrent_state=kda_state.recurrent_states[self.layer_idx],
        ), kda_meta

    def _run_conv_prefill(self, x, state, conv_weight, meta):
        return causal_conv1d_fn(
            x.transpose(0, 1),
            conv_weight,
            None,
            activation="silu",
            conv_states=state.transpose(-1, -2),
            has_initial_state=meta.has_initial_state,
            cache_indices=meta.non_spec_state_indices_tensor,
            query_start_loc=meta.non_spec_query_start_loc.to(torch.int32),
            metadata=meta,
        ).transpose(0, 1)

    def _run_conv_decode(self, x, state, conv_weight, meta):
        return causal_conv1d_update(
            x,
            state.transpose(-1, -2),
            conv_weight,
            None,
            activation="silu",
            conv_state_indices=meta.non_spec_state_indices_tensor[:meta.num_actual_tokens],
            validate_data=True,
        )

    def _run_conv_prefill_fused(
        self,
        q,
        k,
        v,
        state_view,
        q_weight,
        k_weight,
        v_weight,
        meta,
    ):
        n_tokens = q.size(0)
        projection_size = q.size(1)
        out = torch.empty(
            (3, n_tokens, projection_size),
            dtype=q.dtype,
            device=q.device,
        )
        _fused_qkv_causal_conv1d_prefill_kernel[
            (triton.cdiv(n_tokens, 8), triton.cdiv(projection_size, 256), 3)
        ](
            q,
            k,
            v,
            q_weight,
            k_weight,
            v_weight,
            state_view.q_conv_state,
            state_view.k_conv_state,
            state_view.v_conv_state,
            meta.non_spec_state_indices_tensor,
            out,
            n_tokens,
            q.stride(0),
            k.stride(0),
            v.stride(0),
            state_view.q_conv_state.stride(0),
            state_view.q_conv_state.stride(1),
            out.stride(0),
            PROJ=projection_size,
            BLOCK_M=8,
            BLOCK_N=256,
            num_stages=2,
        )
        return out[0], out[1], out[2]

    def _ensure_triton_allocator(self, device: torch.device) -> None:
        if torch.compiler.is_compiling():
            return
        if not self._triton_allocator_ready:
            set_triton_allocator(device)
            self._triton_allocator_ready = True

    def forward_impl(
        self,
        q_proj_states: torch.Tensor,
        k_proj_states: torch.Tensor,
        v_proj_states: torch.Tensor,
        raw_g: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        state_view, meta = self._get_state()
        if state_view is None or meta is None:
            core_attn_out.zero_()
            return core_attn_out

        num_actual_tokens = meta.num_actual_tokens
        q_proj_states = q_proj_states[:num_actual_tokens]
        k_proj_states = k_proj_states[:num_actual_tokens]
        v_proj_states = v_proj_states[:num_actual_tokens]
        raw_g = raw_g[:, :num_actual_tokens]
        beta = beta[:, :num_actual_tokens]

        q_conv_weights = self.q_conv1d.weight.view(
            self.q_conv1d.weight.size(0),
            self.q_conv1d.weight.size(2),
        )
        k_conv_weights = self.k_conv1d.weight.view(
            self.k_conv1d.weight.size(0),
            self.k_conv1d.weight.size(2),
        )
        v_conv_weights = self.v_conv1d.weight.view(
            self.v_conv1d.weight.size(0),
            self.v_conv1d.weight.size(2),
        )

        if (
            meta.num_prefills == 1
            and meta.num_decodes == 0
            and not meta.any_have_initial_state
            and self.conv_size == 4
        ):
            q, k, v = self._run_conv_prefill_fused(
                q_proj_states,
                k_proj_states,
                v_proj_states,
                state_view,
                q_conv_weights,
                k_conv_weights,
                v_conv_weights,
                meta,
            )
        elif meta.num_prefills > 0:
            q = self._run_conv_prefill(q_proj_states, state_view.q_conv_state, q_conv_weights, meta)
            k = self._run_conv_prefill(k_proj_states, state_view.k_conv_state, k_conv_weights, meta)
            v = self._run_conv_prefill(v_proj_states, state_view.v_conv_state, v_conv_weights, meta)
        else:
            q = self._run_conv_decode(q_proj_states, state_view.q_conv_state, q_conv_weights, meta)
            k = self._run_conv_decode(k_proj_states, state_view.k_conv_state, k_conv_weights, meta)
            v = self._run_conv_decode(v_proj_states, state_view.v_conv_state, v_conv_weights, meta)

        q, k, v = (
            rearrange(q, "n (h d) -> 1 n h d", d=self.head_dim),
            rearrange(k, "n (h d) -> 1 n h d", d=self.head_dim),
            rearrange(v, "n (h d) -> 1 n h d", d=self.head_dim),
        )

        num_prefill_tokens = meta.num_prefill_tokens
        num_decode_tokens = meta.num_decode_tokens

        if (
            num_prefill_tokens == num_actual_tokens
            and num_prefill_tokens <= 128
            and meta.num_prefills == 1
            and meta.num_decodes == 0
            and not meta.any_have_initial_state
        ):
            state_indices = meta.non_spec_state_indices_tensor[:1]
            recurrent_out = torch.empty_like(q)
            _fused_short_kda_recurrent_kernel[
                (
                    triton.cdiv(self.head_dim, 8),
                    self.local_num_heads,
                )
            ](
                q,
                k,
                v,
                raw_g,
                beta,
                self.A_log,
                self.dt_bias,
                state_view.recurrent_state,
                state_indices,
                recurrent_out,
                num_prefill_tokens,
                SCALE=self.head_dim ** -0.5,
                HEADS=self.local_num_heads,
                D=self.head_dim,
                BV=8,
                num_warps=1,
                num_stages=3,
            )
            return recurrent_out

        if num_prefill_tokens > 0:
            pf_state_indices = meta.non_spec_state_indices_tensor[:meta.num_prefills]
            pf_has_initial = meta.has_initial_state[:meta.num_prefills]
            single_fresh_prefill = (
                meta.num_prefills == 1
                and meta.num_decodes == 0
                and not meta.any_have_initial_state
            )
            pf_cu_seqlens = None if single_fresh_prefill else (
                meta.non_spec_query_start_loc
                if meta.num_decodes == 0
                else meta.non_spec_query_start_loc[: meta.num_prefills + 1]
            )
            # int32, not int64: vLLM's GDN metadata builds
            # ``non_spec_query_start_loc`` as int32 and the FLA chunk kernels
            # index with that width. Handing them int64 offsets silently
            # changes the result (verified in isolation: identical inputs,
            # amax 0.042 with int32 vs 0.0005 with int64).
            if pf_cu_seqlens is not None:
                pf_cu_seqlens = pf_cu_seqlens.to(torch.int32)
            if single_fresh_prefill:
                pf_initial_state = None
            elif not meta.any_have_initial_state:
                pf_initial_state = state_view.recurrent_state[pf_state_indices].contiguous()
                pf_initial_state.zero_()
            elif meta.all_have_initial_state:
                pf_initial_state = state_view.recurrent_state[pf_state_indices].contiguous()
            else:
                zero_idx = pf_state_indices[~pf_has_initial]
                if zero_idx.numel() > 0:
                    state_view.recurrent_state[zero_idx] = 0
                pf_initial_state = state_view.recurrent_state[pf_state_indices].contiguous()
            # vLLM's Kimi prefill (``kimi_gdn_linear_attn._forward``) hands the
            # *raw* gate projection to ``chunk_kda_with_fused_gate``, which
            # applies ``A_log``/``dt_bias`` and the softplus in fp32 registers
            # inside the chunk kernel. Materializing the gate first with
            # ``fused_kda_gate`` and calling ``chunk_kda`` gives a bit-identical
            # output (verified: cos 1.000000, max|d| 0) but costs an extra pass
            # -- the fused form is 1.06-1.27x faster over 512..8192 tokens.
            pf_out, pf_last_state = chunk_kda_with_fused_gate(
                q=q[:, :num_prefill_tokens].contiguous(),
                k=k[:, :num_prefill_tokens].contiguous(),
                v=v[:, :num_prefill_tokens].contiguous(),
                raw_g=raw_g[:, :num_prefill_tokens].contiguous(),
                beta=beta[:, :num_prefill_tokens].contiguous(),
                A_log=self.A_log,
                g_bias=self.dt_bias,
                initial_state=pf_initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=pf_cu_seqlens,
            )
            state_view.recurrent_state[pf_state_indices] = pf_last_state
            if num_decode_tokens == 0 and num_prefill_tokens == core_attn_out.size(1):
                core_attn_out = pf_out
            else:
                core_attn_out[:, :num_prefill_tokens] = pf_out

        if num_decode_tokens > 0:
            dec_start = num_prefill_tokens
            dec_state_indices = meta.non_spec_state_indices_tensor
            if meta.num_prefills > 0:
                dec_state_indices = dec_state_indices[meta.num_prefills:]
            dec_q = q[:, dec_start:].contiguous()
            dec_k = k[:, dec_start:].contiguous()
            dec_v = v[:, dec_start:].contiguous()
            dec_beta = beta[:, dec_start:].contiguous()
            dec_cu = (
                meta.non_spec_query_start_loc
                if meta.num_prefills == 0
                else meta.non_spec_query_start_loc[: meta.num_decodes + 1]
            ).to(torch.int32)
            # vLLM's decode gates first with ``fused_kda_gate`` and then calls
            # ``fused_recurrent_kda`` against the full recurrent state, indexed
            # by ``ssm_state_indices``. This replaces a hand-written kernel whose
            # premise -- that ``chunk_kda`` and ``fused_recurrent_kda`` disagree
            # on the state layout -- could not be reproduced, and which was
            # slower at the batch sizes that matter (1.27x at 256 sequences).
            dec_g = fused_kda_gate(
                rearrange(raw_g[:, dec_start:], "1 n h d -> n (h d)"),
                self.A_log,
                self.head_dim,
                g_bias=self.dt_bias,
            ).unsqueeze(0)
            # This kernel treats state index 0 as a null/skip slot, the same
            # convention as ``causal_conv1d``'s ``NULL_BLOCK_ID``: given slot 0
            # it returns NaN and writes no state (measured: slot 0 -> nan with
            # zero state delta; slots 1 and 3 -> clean, delta ~0.088). The state
            # allocator reserves slot 0 so no call site has to special-case it.
            dec_out, _ = fused_recurrent_kda(
                q=dec_q,
                k=dec_k,
                v=dec_v,
                g=dec_g,
                beta=dec_beta,
                initial_state=state_view.recurrent_state,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=dec_cu,
                ssm_state_indices=dec_state_indices,
            )
            core_attn_out[:, dec_start:] = dec_out
        return core_attn_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        state_manager=None,
    ) -> torch.Tensor:
        del state_manager
        num_tokens = hidden_states.size(0)
        self._ensure_triton_allocator(hidden_states.device)

        if self.input_proj is not None:
            _p = self._ps_local
            qkvb_local = 3 * _p + self._nh_local
            packed = self.input_proj(hidden_states)
            qkvb = packed[..., :qkvb_local]
            q_proj_states = qkvb[..., :_p]
            k_proj_states = qkvb[..., _p:2 * _p]
            v_proj_states = qkvb[..., 2 * _p:3 * _p]
            raw_beta = qkvb[..., 3 * _p:]
            f_a = packed[..., qkvb_local:qkvb_local + self.head_dim]
            g_a = packed[
                ..., qkvb_local + self.head_dim:qkvb_local + 2 * self.head_dim
            ]
        else:
            q_proj_states = self.q_proj(hidden_states)
            k_proj_states = self.k_proj(hidden_states)
            v_proj_states = self.v_proj(hidden_states)
            raw_beta = self.b_proj(hidden_states)
            f_a = self.f_a_proj(hidden_states)
            g_a = self.g_a_proj(hidden_states)

        beta = torch.empty(
            (1, num_tokens, self.local_num_heads),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        beta_rows = 1 if num_tokens < 8 else 8
        _beta_sigmoid_kernel[(triton.cdiv(num_tokens, beta_rows),)](
            raw_beta,
            beta,
            num_tokens,
            raw_beta.stride(0),
            ROWS=beta_rows,
            HEADS=self.local_num_heads,
            num_warps=1,
        )
        if num_tokens <= 512 and not self.f_b_proj.use_fp8:
            projection_size = self.local_num_heads * self.head_dim
            gate_proj_states = torch.empty(
                (2, num_tokens, projection_size),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            block_m = 16 if num_tokens <= 16 else 32 if num_tokens <= 64 else 64
            block_n = 128
            _grouped_gate_up_kernel[
                (
                    triton.cdiv(num_tokens, block_m)
                    * triton.cdiv(projection_size, block_n),
                    2,
                )
            ](
                f_a,
                g_a,
                self.f_b_proj.weight,
                self.g_b_proj.weight,
                gate_proj_states,
                M=num_tokens,
                N=projection_size,
                K=self.head_dim,
                stride_f_m=f_a.stride(0),
                stride_g_m=g_a.stride(0),
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=32,
                num_warps=4,
                num_stages=3,
            )
            raw_g_states = gate_proj_states[0]
            g_proj_states = gate_proj_states[1]
        else:
            raw_g_states = self.f_b_proj(f_a)
            g_proj_states = self.g_b_proj(g_a)

        # Raw gate projection, shaped [1, n, H, D] and left ungated: the prefill
        # chunk kernel applies A_log/dt_bias itself, and the decode path gates
        # with ``fused_kda_gate`` just before its call.
        raw_g = rearrange(
            raw_g_states,
            "n (h d) -> 1 n h d",
            d=self.head_dim,
        )

        g2 = rearrange(g_proj_states, "... (h d) -> ... h d", d=self.head_dim)

        core_attn_out = torch.empty(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        if self._use_custom_op:
            torch.ops.fastkernels.kda_attention(
                q_proj_states,
                k_proj_states,
                v_proj_states,
                raw_g,
                beta,
                core_attn_out,
                self._layer_name,
            )
        else:
            core_attn_out = self.forward_impl(
                q_proj_states=q_proj_states,
                k_proj_states=k_proj_states,
                v_proj_states=v_proj_states,
                raw_g=raw_g,
                beta=beta,
                core_attn_out=core_attn_out,
            )

        if num_tokens <= 128 and not self.o_proj.use_fp8:
            out = torch.empty(
                (num_tokens, self.hidden_size),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            block_m = 16 if num_tokens <= 16 else 32
            block_n = 128
            _fused_norm_output_projection_kernel[
                (
                    triton.cdiv(num_tokens, block_m)
                    * triton.cdiv(self.hidden_size, block_n),
                )
            ](
                core_attn_out,
                g2,
                self.o_norm.weight,
                self.o_proj.weight,
                out,
                self.o_norm.eps,
                M=num_tokens,
                N=self.hidden_size,
                HEADS=self.local_num_heads,
                HEAD_DIM=self.head_dim,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=8,
                num_stages=3,
            )
            if self.tp_size > 1:
                out = self.o_proj.allreduce(out)
            return out

        core_attn_out = self.o_norm(core_attn_out, g2)
        core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
        return self.o_proj(core_attn_out)
