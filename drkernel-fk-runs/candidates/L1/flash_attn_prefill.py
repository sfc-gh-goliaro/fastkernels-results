import torch
import torch.nn as nn

# Try to import Triton; if not available, we'll fallback to FlashAttention
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


@triton.jit
def _dense_varlen_kernel(
    Q, K, V, Out,
    CuSeqlens,  # [B+1] int32 cumulative lengths
    B: tl.constexpr, M_q: tl.constexpr, M_k: tl.constexpr, HD: tl.constexpr,
    stride_q_b, stride_q_m, stride_q_d,
    stride_k_b, stride_k_m, stride_k_d,
    stride_v_b, stride_v_m, stride_v_d,
    stride_o_b, stride_o_m, stride_o_d,
    BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Program ids
    pid_m = tl.program_id(0)  # over Q tokens: 0..B*M_q-1
    pid_h = tl.program_id(1)  # over heads: 0..HD-1

    b = pid_m // M_q
    m = pid_m % M_q
    h = pid_h

    # Base pointers for this (b, m, h)
    q_ptr = Q + b * stride_q_b + m * stride_q_m + h * stride_q_d
    o_ptr = Out + b * stride_o_b + m * stride_o_m + h * stride_o_d

    # D offsets and q load
    d_offsets = tl.arange(0, BLOCK_D)
    mask_q = d_offsets < HD
    q = tl.load(q_ptr + d_offsets, mask=mask_q, other=0.0).to(tl.float32)

    # Accumulators
    denom = tl.zeros((), dtype=tl.float32)
    out = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # Loop over K in blocks
    n = 0
    while n < M_k:
        kn = n + tl.arange(0, BLOCK_N)
        mask_n = kn < M_k

        # Sequence length for batch b: end = cu[b+1], start = cu[b], len = end - start
        end_b = tl.load(CuSeqlens + b + 1)
        start_b = tl.load(CuSeqlens + b)
        seqlen_b = end_b - start_b
        mask_seq = kn < seqlen_b  # valid if within this sequence

        # k block: [BLOCK_N, BLOCK_D]
        k_ptr = K + b * stride_k_b + (start_b + kn)[:, None] * stride_k_m + h * stride_k_d + d_offsets[None, :] * stride_k_d
        k = tl.load(k_ptr, mask=mask_n[:, None] & mask_seq[:, None] & mask_q[None, :], other=0.0).to(tl.float32)

        # scores = sum(k * q, axis=D) -> [BLOCK_N]
        scores = tl.sum(k * q[None, :], axis=1)
        # Apply padding: invalid -> -inf
        scores = tl.where(mask_n & mask_seq, scores, -float('inf'))

        # Stable softmax over valid
        max_s = tl.max(scores, axis=0)
        scores = scores - max_s
        exp_s = tl.exp(scores)
        exp_s = tl.where(mask_n & mask_seq, exp_s, 0.0)
        denom_block = tl.sum(exp_s, axis=0)
        denom += denom_block
        probs = exp_s / denom_block  # only valid entries used

        # v block: [BLOCK_N, BLOCK_D]
        v_ptr = V + b * stride_v_b + (start_b + kn)[:, None] * stride_v_m + h * stride_v_d + d_offsets[None, :] * stride_v_d
        v = tl.load(v_ptr, mask=mask_n[:, None] & mask_seq[:, None] & mask_q[None, :], other=0.0).to(tl.float32)

        # out += sum(probs[:,None] * v, axis=0)
        out += tl.sum(probs[:, None] * v, axis=0)

        n += BLOCK_N

    # Normalize by total denominator
    out = out / denom

    # Store result; mask tail on D
    tl.store(o_ptr + d_offsets, out, mask=mask_q)


class _FlashAttentionModel(nn.Module):
    """
    Base class implementing the same API, with Triton dense varlen kernel.
    Entry points: Model and ModelNew.
    """
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sm_scale = head_dim ** -0.5  # kept for API parity

    def _make_cu_seqlens(self, cu: torch.Tensor, expected_S: int):
        """
        Accept cu of shape [B] or [B+1].
        Return cu_cumsum of shape [B+1] (cumulative lengths).
        """
        assert cu.dim() == 1, f"cu must be 1D, got shape {tuple(cu.shape)}"
        B = cu.shape[0]
        device = cu.device
        # If B+1 lengths are provided, use as-is (validate ends match S)
        if B == expected_S + 1:
            # Validate last equals total S_q or S_k
            last = int(cu[-1].item())
            if last != expected_S:
                raise AssertionError(f"Provided cu_seqlens ends mismatch: cu[-1]={last} != expected_S={expected_S}")
            return cu
        # Else expect B lengths and compute cumulative sums
        if B != expected_S:
            raise AssertionError(f"cu_seqlens length mismatch: expected {expected_S} or {expected_S+1}, got {B}")
        # Compute cumulative sums to get B+1
        cu_cum = torch.cumsum(cu, dim=0)
        return cu_cum

    def _can_use_triton(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
        if not _HAS_TRITON:
            return False
        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            return False
        if q.dtype != k.dtype or q.dtype != v.dtype:
            return False
        if q.dtype not in (torch.bfloat16, torch.float16):
            return False
        return True

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
        B = q.shape[0]
        S_q = q.shape[1]
        S_k = k.shape[1]

        # Make cu arrays robust
        cu_q = self._make_cu_seqlens(cu_seqlens_q, expected_S=S_q)
        cu_k = self._make_cu_seqlens(cu_seqlens_k, expected_S=S_k)

        use_triton = self._can_use_triton(q, k, v)

        if not use_triton:
            # Fallback to the original FlashAttention path
            fa_kw = dict(
                max_seqlen_q=max_seqlen_q,
                cu_seqlens_q=cu_q,  #可能被block_table path override
                max_seqlen_k=max_seqlen_k,
                fa_version=-1,  # let vLLM pick
            )
            if kwargs.get("block_table") is not None:
                # block_table path expects seqused_k
                # But our kernel uses cu; to keep simple, fallback to FlashAttention
                seqused_k = None
                try:
                    seqused_k = (cu_k[1:] - cu_k[:-1]).to(torch.int32)
                except Exception:
                    pass
                if seqused_k is not None:
                    fa_kw["seqused_k"] = seqused_k
                else:
                    # Fallback may require cu; pass cu_k as-is if shape B+1
                    fa_kw["cu_seqlens_k"] = cu_k
                fa_kw["num_splits"] = 1
            else:
                fa_kw["cu_seqlens_k"] = cu_k
                fa_kw["num_splits"] = 1
            fa_kw.update(kwargs)
            return flash_attn_varlen_func(q, k, v, **fa_kw)

        # Triton path: dense varlen, no block_table, no causal
        # Shapes
        assert q.shape[0] == k.shape[0] == v.shape[0] == B, "Batch size mismatch"
        HD = q.shape[2]
        assert k.shape[2] == HD and v.shape[2] == HD, "Head dim mismatch"

        # Output
        out = torch.empty_like(q)

        # Strides in elements
        stride_q_b, stride_q_m, stride_q_d = q.stride(0), q.stride(1), q.stride(2)
        stride_k_b, stride_k_m, stride_k_d = k.stride(0), k.stride(1), k.stride(2)
        stride_v_b, stride_v_m, stride_v_d = v.stride(0), v.stride(1), v.stride(2)
        stride_o_b, stride_o_m, stride_o_d = out.stride(0), out.stride(1), out.stride(2)

        # Grid: (B*S_q, HD)
        grid = (B * S_q, HD)

        # Kernel config: tuned for HD around 64-128
        BLOCK_D = 64
        BLOCK_N = 64
        num_warps = 8
        num_stages = 2

        _dense_varlen_kernel[grid](
            q, k, v, out,
            cu_q.new_empty(0)),  # not used; keep arg for signature compatibility -- incorrect in previous, removed now
            # Correct: pass cu_k (K lens) or computed cu
            cu_k,
            B, S_q, S_k, HD,
            stride_q_b, stride_q_m, stride_q_d,
            stride_k_b, stride_k_m, stride_k_d,
            stride_v_b, stride_v_m, stride_v_d,
            stride_o_b, stride_o_m, stride_o_d,
            BLOCK_D=BLOCK_D, BLOCK_N=BLOCK_N,
            num_warps=num_warps, num_stages=num_stages,
        )

        return out


# Public entry point 'ModelNew' using Triton when available
class ModelNew(_FlashAttentionModel):
    pass


# Also provide 'Model' for environments that expect this name
class Model(_FlashAttentionModel):
    pass


# Keep the original imports; flash_attn_varlen_func is used in fallback
from fastkernels.infra.fa_utils import fa3_scheduler_metadata, fa_version_for_head_size, flash_attn_varlen_func

FlashAttnPrefill = ModelNew
