import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _qkv_softmax_fused_kernel(
    Q, K, V, O,
    B, H, Q, Kdim, C,
    stride_q_b, stride_q_h, stride_q_q, stride_q_c,
    stride_k_b, stride_k_h, stride_k_k, stride_k_c,
    stride_v_b, stride_v_h, stride_v_k, stride_v_c,
    stride_o_b, stride_o_h, stride_o_q, stride_o_c,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # program ids
    pid_bh = tl.program_id(0)  # over B*H
    pid_m = tl.program_id(1)   # over Q tiles

    b = pid_bh // H
    h = pid_bh % H

    m_start = pid_m * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)
    mask_m = offs_m < Q

    # Pass 1: row-wise max over K from x = Q @ K^T
    row_max = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    k0 = 0
    while k0 < Kdim:
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < Kdim

        # Q block: [BM, BK]
        q_ptrs = Q + b * stride_q_b + h * stride_q_h + (offs_m[:, None] * stride_q_q) + (offs_k[None, :] * stride_q_c)
        q = tl.load(q_ptrs, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0).to(tl.float32)

        # K block: [BK] (column c=0)
        k_ptrs = K + b * stride_k_b + h * stride_k_h + offs_k * stride_k_k  # + 0 * stride_k_c
        k = tl.load(k_ptrs, mask=mask_k, other=0.0).to(tl.float32)

        # x = Q @ K^T => [BM]
        x = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for cc in range(0, BLOCK_K):
            x += q[:, cc] * k[cc]
        # invalidate out-of-bounds k with -inf so they don't affect max
        x = tl.where(mask_k[None, :], x, -float('inf'))

        row_max = tl.maximum(row_max, x)
        k0 += BLOCK_K

    # Pass 2: row-wise sum of exp(x - max)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    k0 = 0
    while k0 < Kdim:
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < Kdim

        q_ptrs = Q + b * stride_q_b + h * stride_q_h + (offs_m[:, None] * stride_q_q) + (offs_k[None, :] * stride_q_c)
        q = tl.load(q_ptrs, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0).to(tl.float32)

        k_ptrs = K + b * stride_k_b + h * stride_k_h + offs_k * stride_k_k
        k = tl.load(k_ptrs, mask=mask_k, other=0.0).to(tl.float32)

        x = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for cc in range(0, BLOCK_K):
            x += q[:, cc] * k[cc]
        x = tl.where(mask_k[None, :], x, -float('inf'))

        e = tl.exp(x - row_max)  # invalid -> 0
        e = tl.where(mask_m, e, 0.0)
        row_sum += e

        k0 += BLOCK_K

    # Pass 3: compute p = exp / sum and accumulate O = p @ V -> [BM, BN]
    offs_n = tl.arange(0, BLOCK_N)
    o_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k0 = 0
    while k0 < Kdim:
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < Kdim

        q_ptrs = Q + b * stride_q_b + h * stride_q_h + (offs_m[:, None] * stride_q_q) + (offs_k[None, :] * stride_q_c)
        q = tl.load(q_ptrs, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0).to(tl.float32)

        k_ptrs = K + b * stride_k_b + h * stride_k_h + offs_k * stride_k_k
        k = tl.load(k_ptrs, mask=mask_k, other=0.0).to(tl.float32)

        x = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for cc in range(0, BLOCK_K):
            x += q[:, cc] * k[cc]
        x = tl.where(mask_k[None, :], x, -float('inf'))

        e = tl.exp(x - row_max)
        p = e / row_sum  # [BM]

        n0 = 0
        while n0 < C:
            offs_n_curr = n0 + offs_n
            mask_n = offs_n_curr < C

            # V block: [BK, BN]
            v_ptrs = V + b * stride_v_b + h * stride_v_h + (offs_k[:, None] * stride_v_k) + (offs_n_curr[None, :] * stride_v_c)
            v = tl.load(v_ptrs, mask=(mask_k[:, None] & mask_n[None, :]), other=0.0).to(tl.float32)

            # y = p[:, None] @ v[None, :] => [BM, BN]
            y = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for kk in range(0, BLOCK_K):
                if mask_k[kk]:
                    vk = v[kk, :]  # [BN]
                    y += p[:, None] * vk[None, :]

            o_acc += y
            n0 += BLOCK_N

        k0 += BLOCK_K

    # Store O tile
    o_ptrs = O + b * stride_o_b + h * stride_o_h + (offs_m[:, None] * stride_o_q) + (offs_n[None, :] * stride_o_c)
    mask_out = (mask_m[:, None] & (offs_n[None, :] < C))
    tl.store(o_ptrs, o_acc, mask=mask_out)


def _triton_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    q: [B, H, Q, C] (contiguous), dtype bf16/fp16
    k: [B, H, K, C] (contiguous)
    v: [B, H, K, C] (contiguous)
    returns: o: [B, H, Q, C] (same dtype as input)
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Triton kernel requires CUDA tensors"
    assert q.dtype in (torch.bfloat16, torch.float16) and k.dtype == q.dtype and v.dtype == q.dtype, "Use bf16/fp16"

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    B, H, Q, C = q.shape
    assert k.shape[0] == B and k.shape[1] == H and v.shape[0] == B and v.shape[1] == H, "Batch/Head mismatch"
    Kdim = k.shape[2]
    assert v.shape[2] == Kdim, "Key dim must equal Value dim"
    assert q.shape[-1] == C and k.shape[-1] == C and v.shape[-1] == C, "Last dim mismatch"

    # Output in float32 for stability; cast later
    o = torch.empty((B, H, Q, C), device=q.device, dtype=torch.float32)

    # strides (elements)
    stride_q_b, stride_q_h, stride_q_q, stride_q_c = q.stride()
    stride_k_b, stride_k_h, stride_k_k, stride_k_c = k.stride()
    stride_v_b, stride_v_h, stride_v_k, stride_v_c = v.stride()
    stride_o_b, stride_o_h, stride_o_q, stride_o_c = o.stride()

    # block heuristics
    BLOCK_N = 128 if C >= 256 else 64
    BLOCK_M = 128 if Q >= 128 else 64
    BLOCK_K = 64 if Kdim >= 64 else 32

    grid = (B * H, triton.cdiv(Q, BLOCK_M))
    num_warps = 4 if BLOCK_N <= 64 else 8
    num_stages = 2

    _qkv_softmax_fused_kernel[grid](
        q, k, v, o,
        B, H, Q, Kdim, C,
        stride_q_b, stride_q_h, stride_q_q, stride_q_c,
        stride_k_b, stride_k_h, stride_k_k, stride_k_c,
        stride_v_b, stride_v_h, stride_v_k, stride_v_c,
        stride_o_b, stride_o_h, stride_o_q, stride_o_c,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )

    # cast to input dtype
    return o.to(q.dtype)


class ModelNew(nn.Module):
    """Triton-optimized multi-head attention with same API as the original Model.
    Entry point 'ModelNew' as requested.
    """
    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
        q_bias: bool = False,
    ):
        super().__init__()
        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        # Keep standard Linear layers (cuBLAS) for Q/K/V/O projections
        self.linear_q = nn.Linear(c_q, c_hidden * no_heads, bias=q_bias)
        self.linear_k = nn.Linear(c_k, c_hidden * no_heads, bias=False)
        self.linear_v = nn.Linear(c_v, c_hidden * no_heads, bias=False)
        self.linear_o = nn.Linear(c_hidden * no_heads, c_q, bias=False)

        self.linear_g = nn.Linear(c_q, c_hidden * no_heads, bias=False) if gating else None

    def _prep_qkv(
        self, q_x: torch.Tensor, kv_x: torch.Tensor, apply_scale: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Use cuBLAS GEMMs
        q = F.linear(q_x, self.linear_q.weight, self.linear_q.bias)
        k = F.linear(kv_x, self.linear_k.weight, self.linear_k.bias)
        v = F.linear(kv_x, self.linear_v.weight, self.linear_v.bias)

        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        q = q.transpose(-2, -3).contiguous()
        k = k.transpose(-2, -3).contiguous()
        v = v.transpose(-2, -3).contiguous()

        if apply_scale:
            q = q / math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
        if self.linear_g is not None:
            g = torch.sigmoid(F.linear(q_x, self.linear_g.weight, self.linear_g.bias))
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        o = o.reshape(o.shape[:-2] + (-1,))
        return F.linear(o, self.linear_o.weight, self.linear_o.bias)

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: list[torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Same signature as original Model.forward.
        Uses Triton kernel by default when available (CUDA + Triton).
        """
        if biases is None:
            biases = []

        q, k, v = self._prep_qkv(q_x, kv_x)

        if TRITON_AVAILABLE and q.is_cuda:
            o = _triton_attention(q, k, v)
        else:
            # PyTorch fallback: einsum + softmax
            scores = torch.einsum("...qc,...kc->...qk", q, k)
            for b in biases:
                scores = scores + b
            probs = F.softmax(scores, dim=-1)
            o = torch.einsum("...qk,...kc->...qc", probs.to(dtype=v.dtype), v)

        o = o.transpose(-2, -3)
        return self._wrap_up(o, q_x)

OF3Attention = ModelNew
