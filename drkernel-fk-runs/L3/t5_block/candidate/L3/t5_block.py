import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


def _ceil_div(a, b):
    return (a + b - 1) // b


@triton.jit
def bmm_a_bf16_bf16_to_fp32(
    A_ptr, B_ptr, C_ptr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # C pointers
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < BK:
        k_off = k + offs_k
        # A: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + k_off[None, :] * stride_ak)
        # B: [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + (k_off[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < BM) & (k_off[None, :] < BK), other=0.0)
        b = tl.load(b_ptrs, mask=(k_off[:, None] < BK) & (offs_n[None, :] < BN), other=0.0)

        a = a.to(tl.float32)
        b = b.to(tl.float32)
        acc += tl.dot(a, b)
        k += BLOCK_K

    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < BM) & (offs_n[None, :] < BN))


@triton.jit
def bmm_a_fp32_bf16_to_bf16(
    A_ptr, B_ptr, C_ptr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < BK:
        k_off = k + offs_k
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + k_off[None, :] * stride_ak)
        b_ptrs = B_ptr + (k_off[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < BM) & (k_off[None, :] < BK), other=0.0)
        b = tl.load(b_ptrs, mask=(k_off[:, None] < BK) & (offs_n[None, :] < BN), other=0.0)

        b = b.to(tl.float32)
        acc += tl.dot(a, b)
        k += BLOCK_K

    c = acc.to(tl.bfloat16)
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < BM) & (offs_n[None, :] < BN))


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # PyTorch LN is fine here.
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states


class T5SelfAttention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.d_kv

        # Assume tp=1 for this model; keep compatibility
        self.n_heads_per_partition = self.n_heads

        # QKV and output linears (F.linear / Parameter paths)
        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.d_model,
            head_size=self.d_kv,
            total_num_heads=self.n_heads,
            total_num_kv_heads=self.n_heads,
            bias=False,
        )
        self.o = RowParallelLinear(self.inner_dim, self.d_model, bias=False)

        self.has_relative_attention_bias = has_relative_attention_bias
        if has_relative_attention_bias:
            self.relative_attention_bias = Embedding(
                config.relative_attention_num_buckets, config.num_heads,
            )

    def _triton_qk(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        # query: [B, H, Lq, D] bf16
        # key:   [B, H, Lk, D] bf16
        B, H, Lq, D = query.shape
        _, _, Lk, Dk = key.shape
        assert D == Dk, f"Q/K D mismatch: {D} vs {Dk}"
        # Flatten to [BH, L, D] contiguous
        query_ = query.reshape(B * H, Lq, D).contiguous()
        key_ = key.reshape(B * H, Lk, D).contiguous()

        BM = B * H
        BN = Lk
        BK = D

        # C: [BM, BN] fp32
        C = torch.empty((BM, BN), device=query.device, dtype=torch.float32)

        # Strides in elements for contiguous [M,K]
        stride_am = D
        stride_ak = 1
        # Strides for B^T as [K,N]
        stride_bk = D
        stride_bn = 1
        # Strides for C [M,N]
        stride_cm = BN
        stride_cn = 1

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64  # match D=64
        grid = (_ceil_div(BM, BLOCK_M), _ceil_div(BN, BLOCK_N))

        bmm_a_bf16_bf16_to_fp32[grid](
            query_.view(-1), key_.view(-1), C.view(-1),
            BM, BN, BK,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )
        return C.view(B, H, Lq, Lk)

    def _triton_av(self, attn: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        # attn:  [B, H, Lq, Lk] fp32
        # value: [B, H, Lv, D] bf16
        B, H, Lq, Lk = attn.shape
        _, _, Lv, D = value.shape
        assert Lk == Lv, f"Mismatch: Lk={Lk} Lv={Lv}"
        attn_ = attn.reshape(B * H, Lq, Lk).contiguous()
        value_ = value.reshape(B * H, Lv, D).contiguous()

        BM = B * H
        BN = Lk
        BK = D

        C = torch.empty((BM, BN), device=value.device, dtype=torch.bfloat16)

        stride_am = Lk
        stride_ak = 1
        stride_bk = D
        stride_bn = 1
        stride_cm = BN
        stride_cn = 1

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid = (_ceil_div(BM, BLOCK_M), _ceil_div(BN, BLOCK_N))

        bmm_a_fp32_bf16_to_bf16[grid](
            attn_.view(-1), value_.view(-1), C.view(-1),
            BM, BN, BK,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )
        return C.view(B, H, Lq, D)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1) QKV linear
        qkv = self.qkv_proj(hidden_states)
        q_size = self.n_heads * self.d_kv
        kv_size = self.n_heads * self.d_kv
        query_states, key_states, value_states = qkv.split([q_size, kv_size, kv_size], dim=-1)

        # Reshape to [B, H, L, D]
        B = hidden_states.shape[0]
        L = hidden_states.shape[1]
        D = self.d_kv
        H = self.n_heads

        query_states = query_states.view(B, H, L, D)
        key_states = key_states.view(B, H, L, D)
        value_states = value_states.view(B, H, L, D)

        # 2) QK scores via Triton
        scores = self._triton_qk(query_states, key_states)  # [B,H,L,L] fp32

        # 3) Add position_bias if provided
        if position_bias is not None:
            # position_bias may be [1,H,L,L] -> expand batch
            if position_bias.dim() == 4 and position_bias.shape[0] == 1:
                position_bias = position_bias.expand(B, -1, -1, -1)
            # Upcast to fp32 for stable addition
            pb = position_bias.to(scores.dtype)
            scores = scores + pb

        # 4) Softmax over last dim
        probs = torch.softmax(scores, dim=-1)  # fp32

        # 5) AV via Triton
        attn_output = self._triton_av(probs, value_states)  # [B,H,L,D] bf16

        # 6) Output linear
        attn_output = attn_output.view(B, L, self.inner_dim)
        attn_output = self.o(attn_output)

        return attn_output, position_bias


class T5LayerFF(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config)
        else:
            self.DenseReluDense = T5DenseActDense(config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = self.layer_norm(hidden_states)
        ff_output = self.DenseReluDense(normed)
        hidden_states = hidden_states + ff_output
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        return hidden_states


class ModelNew(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.layer = nn.ModuleList([
            T5LayerSelfAttention(config, has_relative_attention_bias),
            T5LayerFF(config),
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, position_bias = self.layer[0](
            hidden_states, mask=mask, position_bias=position_bias,
        )
        hidden_states = self.layer[1](hidden_states)
        return hidden_states, position_bias

T5Block = ModelNew
