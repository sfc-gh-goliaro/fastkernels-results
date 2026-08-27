import math
import torch
import torch.nn as nn

import triton
import triton.language as tl

# ---------------------------
# Triton kernels: fused QK+bias+softmax and AV matmul
# ---------------------------

@triton.jit
def _qkv_softmax_rows_kernel(
    Q, K, Bias, Probs,
    BATCH: tl.constexpr, H: tl.constexpr, S: tl.constexpr, K: tl.constexpr,
    stride_qb, stride_qm, stride_qn, stride_qk,
    stride_kb, stride_km, stride_kn, stride_kk,
    stride_biasb, stride_biasm, stride_biasn,
    stride_pb, stride_pm, stride_pn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # program ids
    pid_m = tl.program_id(0)      # block over columns m
    pid_row = tl.program_id(1)    # row id over (B*H)
    pid_n = tl.program_id(2)      # block over columns n

    b = pid_row // H
    h = pid_row % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pass 1: row-wise max over N
    row_max = tl.full((BLOCK_M,), -1.0e30, dtype=tl.float32)
    for n in range(0, S, BLOCK_N):
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(
                Q + b * stride_qb + h * stride_qm + offs_m[:, None] * 0 + (k + offs_k[None, :]) * stride_qk,
                mask=(offs_m[:, None] < S) & (k + offs_k[None, :] < K),
                other=0.0,
            ).to(tl.float32)  # [BM, BK]
            b_ = tl.load(
                K + b * stride_kb + h * stride_km + (k + offs_k[:, None]) * stride_kk + (n + offs_n[None, :]) * stride_kn,
                mask=((n + offs_n[None, :]) < S) & (k + offs_k[:, None] < K),
                other=0.0,
            ).to(tl.float32)  # [BK, BN]
            acc += tl.dot(a, b_)  # [BM, BN]
        # add bias
        bias = tl.load(
            Bias + b * stride_biasb + h * stride_biasm + offs_m[:, None] * 0 + (n + offs_n[None, :]) * stride_biasn,
            mask=((n + offs_n[None, :]) < S) & (offs_m[:, None] < S),
            other=0.0,
        ).to(tl.float32)
        x = acc + bias
        # update max (mask invalids to -1e30)
        x_masked = tl.where(((n + offs_n[None, :]) < S) & (offs_m[:, None] < S), x, -1.0e30)
        col_max = tl.max(x_masked, axis=1)
        row_max = tl.maximum(row_max, col_max)

    # Pass 2: row-wise sum of exp over N
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for n in range(0, S, BLOCK_N):
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(
                Q + b * stride_qb + h * stride_qm + offs_m[:, None] * 0 + (k + offs_k[None, :]) * stride_qk,
                mask=(offs_m[:, None] < S) & (k + offs_k[None, :] < K),
                other=0.0,
            ).to(tl.float32)
            b_ = tl.load(
                K + b * stride_kb + h * stride_km + (k + offs_k[:, None]) * stride_kk + (n + offs_n[None, :]) * stride_kn,
                mask=((n + offs_n[None, :]) < S) & (k + offs_k[:, None] < K),
                other=0.0,
            ).to(tl.float32)
            acc += tl.dot(a, b_)
        bias = tl.load(
            Bias + b * stride_biasb + h * stride_biasm + offs_m[:, None] * 0 + (n + offs_n[None, :]) * stride_biasn,
            mask=((n + offs_n[None, :]) < S) & (offs_m[:, None] < S),
            other=0.0,
        ).to(tl.float32)
        x = acc + bias
        e = tl.exp(x - row_max[:, None])
        e = tl.where(((n + offs_n[None, :]) < S) & (offs_m[:, None] < S), e, 0.0)
        row_sum += tl.sum(e, axis=1)

    # Pass 3: write normalized softmax
    for n in range(0, S, BLOCK_N):
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(
                Q + b * stride_qb + h * stride_qm + offs_m[:, None] * 0 + (k + offs_k[None, :]) * stride_qk,
                mask=(offs_m[:, None] < S) & (k + offs_k[None, :] < K),
                other=0.0,
            ).to(tl.float32)
            b_ = tl.load(
                K + b * stride_kb + h * stride_km + (k + offs_k[:, None]) * stride_kk + (n + offs_n[None, :]) * stride_kn,
                mask=((n + offs_n[None, :]) < S) & (k + offs_k[:, None] < K),
                other=0.0,
            ).to(tl.float32)
            acc += tl.dot(a, b_)
        bias = tl.load(
            Bias + b * stride_biasb + h * stride_biasm + offs_m[:, None] * 0 + (n + offs_n[None, :]) * stride_biasn,
            mask=((n + offs_n[None, :]) < S) & (offs_m[:, None] < S),
            other=0.0,
        ).to(tl.float32)
        x = acc + bias
        e = tl.exp(x - row_max[:, None]) / row_sum[:, None]
        # store
        p_ptrs = Probs + b * stride_pb + h * stride_pm + offs_m[:, None] * 0 + (n + offs_n[None, :]) * stride_pn
        mask = ((n + offs_n[None, :]) < S) & (offs_m[:, None] < S)
        tl.store(p_ptrs, e, mask=mask)


@triton.jit
def _matmul_kernel(
    A, B, C,
    BATCH: tl.constexpr, H: tl.constexpr, M: tl.constexpr, K: tl.constexpr,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bm, stride_bk,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_bh = tl.program_id(2)

    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + b * stride_ab + h * stride_am + offs_m[:, None] * 0 + offs_k[None, :] * stride_ak
    b_ptrs = B + b * stride_bb + h * stride_bm + offs_k[:, None] * stride_bk + offs_n[None, :] * 0

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptrs + k * 0,
            mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        b_ = tl.load(
            b_ptrs + k * stride_bk,
            mask=(offs_n[None, :] < K) & (k + offs_k[:, None] < K),
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(a, b_)

    c_ptrs = C + b * stride_cb + h * stride_cm + offs_m[:, None] * 0 + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < K),
    )


# ---------------------------
# Python helpers
# ---------------------------

def _ceil_div(a, b):
    return (a + b - 1) // b


def _launch_qkv_softmax(Q, K, bias, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64):
    """
    Fused kernel: compute softmax over Q@K^T + bias row-wise.
    Inputs:
      Q: [B,H,S,K]
      K: [B,H,S,K]
      bias: [B,H,S,S] (will be broadcast if shape [1,H,S,S])
    Output:
      Probs: [B,H,S,S] float32
    """
    device = Q.device
    assert device.type == "cuda"
    B, H, S, K = Q.shape
    # Ensure contiguous
    Q_c = Q.contiguous()
    K_c = K.contiguous()
    if bias.shape[0] == 1:
        bias_c = bias.expand(B, -1, -1, -1).contiguous()
    else:
        bias_c = bias.contiguous()
    probs = torch.empty((B, H, S, S), device=device, dtype=torch.float32)

    grid = (_ceil_div(S, BLOCK_M), B * H, _ceil_div(S, BLOCK_N))

    _qkv_softmax_rows_kernel[grid](
        Q_c, K_c, bias_c, probs,
        B, H, S, K,
        Q_c.stride(0), Q_c.stride(1), Q_c.stride(2), Q_c.stride(3),
        K_c.stride(0), K_c.stride(1), K_c.stride(2), K_c.stride(3),
        bias_c.stride(0), bias_c.stride(1), bias_c.stride(3),
        probs.stride(0), probs.stride(1), probs.stride(3),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8, num_stages=4,
    )
    return probs


def _launch_matmul(c, A, B, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64):
    """
    c = A @ B
    A: [B,H,M,K]
    B: [B,H,K,N]
    c: [B,H,M,N]
    """
    device = A.device
    assert device.type == "cuda"
    B_count, H, M, K = A.shape
    N = B.shape[-1]
    A_c = A.contiguous()
    B_c = B.contiguous()
    c_c = torch.empty((B_count, H, M, N), device=device, dtype=torch.float32)

    grid = (_ceil_div(M, BLOCK_M), _ceil_div(N, BLOCK_N), B_count * H)

    _matmul_kernel[grid](
        A_c, B_c, c_c,
        B_count, H, M, K,
        A_c.stride(0), A_c.stride(1), A_c.stride(3),
        B_c.stride(0), B_c.stride(1), B_c.stride(3),
        c_c.stride(0), c_c.stride(1), c_c.stride(3),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8, num_stages=4,
    )
    return c_c


# ---------------------------
# Triton-optimized ModelNew
# ---------------------------

class ModelNew(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.d_kv
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance

        tp_size = _tp_size()
        assert self.n_heads % tp_size == 0, "n_heads must be divisible by tensor parallel size"
        self.n_heads_per_partition = self.n_heads // tp_size

        # Keep the same linear layers but math will be Triton
        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.d_model,
            head_size=self.d_kv,
            total_num_heads=self.n_heads,
            total_num_kv_heads=self.n_heads,
            bias=False,
        )

        self.o = RowParallelLinear(self.inner_dim, self.d_model, bias=False)

        if has_relative_attention_bias:
            self.relative_attention_bias = Embedding(
                self.relative_attention_num_buckets, self.n_heads,
            )

    @staticmethod
    def _relative_position_bucket(
        relative_position: torch.Tensor,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> torch.Tensor:
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position),
            )
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )
        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large,
        )
        return relative_buckets

    def compute_bias(self, query_length: int, key_length: int, device: torch.device) -> torch.Tensor:
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position, bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)
        # values shape: [S,S,n_heads]; we need [1, H_per, S, S]
        # Partition heads
        head_start = _tp_rank() * self.n_heads_per_partition
        head_end = head_start + self.n_heads_per_partition
        values = values[:, :, head_start:head_end]  # [S, S, H_per]
        values = values.permute(2, 0, 1).unsqueeze(0).contiguous()  # [1, H_per, S, S]
        return values

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Expect bfloat16 on CUDA
        assert hidden_states.dtype == torch.bfloat16, "Expected bfloat16 inputs"
        device = hidden_states.device
        assert device.type == "cuda", "Triton kernels require CUDA tensors"

        # 1) QKV projection (uses CUDA ops inside; output is bfloat16)
        qkv = self.qkv_proj(hidden_states)  # [B, S, 3*d_kv]
        q_size = self.n_heads_per_partition * self.d_kv
        kv_size = self.n_heads_per_partition * self.d_kv
        query_states, key_states, value_states = qkv.split(
            [q_size, kv_size, kv_size], dim=-1,
        )

        # Reshape to [B, H_per, S, d_kv]
        B, S = hidden_states.shape[0], hidden_states.shape[1]
        H_per = self.n_heads_per_partition

        query_states = query_states.view(B, H_per, self.d_kv, S).contiguous()  # [B, H_per, K, S]
        key_states = key_states.view(B, H_per, self.d_kv, S).contiguous()      # [B, H_per, K, S]
        value_states = value_states.view(B, H_per, self.d_kv, S).contiguous()  # [B, H_per, K, S]

        # Convert to [B, H_per, S, K] for kernel convenience
        Q = query_states.permute(0, 1, 3, 2).contiguous()  # [B, H, S, K]
        K = key_states.permute(0, 1, 3, 2).contiguous()    # [B, H, S, K]
        V = value_states.permute(0, 1, 3, 2).contiguous()  # [B, H, S, K]

        # 2) position_bias
        if position_bias is None:
            if self.has_relative_attention_bias:
                position_bias = self.compute_bias(S, S, device=device)
            else:
                position_bias = torch.zeros(
                    (1, H_per, S, S), device=device, dtype=torch.bfloat16,
                )
            if mask is not None:
                position_bias = position_bias + mask

        # 3) Fused QK+bias+softmax -> probs
        probs = _launch_qkv_softmax(Q, K, position_bias, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64)  # [B,H,S,S] float32

        # 4) AV = probs @ V  -> [B,H,S,K]
        out = _launch_matmul(torch.empty((B, H_per, S, self.d_kv), device=device, dtype=torch.float32),
                             probs, V, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64)  # [B,H,S,K] float32
        out = out.to(torch.bfloat16)

        # 5) Output linear
        attn_2d = out.permute(0, 2, 1, 3).contiguous().view(B, S, -1)  # [B,S,H*K]
        out_final = self.o(attn_2d)  # includes all-reduce

        return out_final, position_bias

T5SelfAttention = ModelNew
