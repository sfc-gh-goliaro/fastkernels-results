import math
import torch
import triton
import triton.language as tl


@triton.jit
def dense_mha_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_q_bh: tl.constexpr, stride_q_s: tl.constexpr, stride_q_d: tl.constexpr,
    stride_k_bh: tl.constexpr, stride_k_s: tl.constexpr, stride_k_d: tl.constexpr,
    stride_v_bh: tl.constexpr, stride_v_s: tl.constexpr, stride_v_d: tl.constexpr,
    stride_o_bh: tl.constexpr, stride_o_s: tl.constexpr, stride_o_d: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_M: tl.constexpr,
):
    # Program ids: grid = (B*H, S)
    pid_bh = tl.program_id(0)
    pid_row = tl.program_id(1)

    # Decode batch and head
    b = pid_bh // H
    h = pid_bh % H

    # Base offsets
    base_q_bh = b * stride_q_bh
    base_k_bh = b * stride_k_bh
    base_v_bh = b * stride_v_bh
    base_o_bh = b * stride_o_bh

    # Row offsets
    row_q = base_q_bh + pid_row * stride_q_s
    row_o = base_o_bh + pid_row * stride_o_s

    # 1) scores = Q[row] dot K^T[m] for all m
    scores = tl.zeros((S,), dtype=tl.float32)

    m0 = 0
    while m0 < S:
        offs_m = m0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S

        scores_block = tl.zeros((BLOCK_M,), dtype=tl.float32)

        d0 = 0
        while d0 < D:
            offs_d = d0 + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D

            # Q[row, d]
            q = tl.load(
                q_ptr + row_q + offs_d * stride_q_d,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)  # (BLOCK_D,)

            # K[m, d] -> (BLOCK_M, BLOCK_D)
            k = tl.load(
                k_ptr + base_k_bh + offs_m[:, None] * stride_k_s + offs_d[None, :] * stride_k_d,
                mask=mask_m[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)

            prod = k * q[None, :]
            scores_block += tl.sum(prod, axis=1)

            d0 += BLOCK_D

        scores = tl.where(mask_m, scores_block, scores)
        m0 += BLOCK_M

    # 2) Softmax (stable)
    row_max = tl.max(scores, axis=0)
    scores = scores - row_max
    numer = tl.exp(scores) * scale
    denom = tl.sum(numer, axis=0)
    softmax = numer / denom  # (S,)

    # 3) O[row, d] = sum_m softmax[m] * V[m, d]
    o_row = tl.zeros((D,), dtype=tl.float32)

    m0 = 0
    while m0 < S:
        offs_m = m0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S

        d0 = 0
        while d0 < D:
            offs_d = d0 + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D

            v = tl.load(
                v_ptr + base_v_bh + offs_m[:, None] * stride_v_s + offs_d[None, :] * stride_v_d,
                mask=mask_m[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)  # (BLOCK_M, BLOCK_D)

            sm = softmax[offs_m][:, None]  # (BLOCK_M, 1)
            contrib = v * sm                # (BLOCK_M, BLOCK_D)
            o_row += tl.sum(contrib, axis=0)

            d0 += BLOCK_D

        m0 += BLOCK_M

    # 4) Store output row (float32); cast in Python
    out_ptr = o_ptr + base_o_bh + pid_row * stride_o_s
    d0 = 0
    while d0 < D:
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        tl.store(out_ptr + offs_d * stride_o_d, o_row[offs_d], mask=mask_d)
        d0 += BLOCK_D


class Model(torch.nn.Module):
    def __init__(self, backend: Literal["auto", "sdpa", "flash_attn", "cudnn", "flex"] = "auto"):
        super().__init__()
        # We use our own Triton kernel; keep signature for compatibility.
        self.backend = backend

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        softmax_scale: float | None = None,
        causal: bool = False,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Fast path constraints
        if causal or attn_mask is not None:
            raise NotImplementedError("Model Triton kernel supports only causal=False and attn_mask=None.")
        if query.dtype != key.dtype or key.dtype != value.dtype:
            raise ValueError("query, key, value must have the same dtype.")
        if query.device != key.device or key.device != value.device:
            raise ValueError("query, key, value must be on the same device.")
        if query.dim() != 4:
            raise ValueError(f"Expected query.ndim == 4, got {query.dim()}.")

        B, S_q, H, D = query.shape
        Bk, S_k, Hk, Dk = key.shape
        Bv, S_v, Hv, Dv = value.shape
        if not (B == Bk == Bv and H == Hk == Hv and D == Dk == Dv and S_k == S_v):
            raise ValueError("query, key, value must have matching shapes except possibly S.")

        # Permute to (B, H, S, D)
        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)

        # Strides (elements)
        s_q_bh, s_q_h, s_q_s, s_q_d = q.stride(0), q.stride(1), q.stride(2), q.stride(3)
        s_k_bh, s_k_h, s_k_s, s_k_d = k.stride(0), k.stride(1), k.stride(2), k.stride(3)
        s_v_bh, s_v_h, s_v_s, s_v_d = v.stride(0), v.stride(1), v.stride(2), v.stride(3)

        # Output buffer (float32 for kernel, cast later)
        o = torch.empty_like(q, dtype=torch.float32)
        s_o_bh, s_o_h, s_o_s, s_o_d = o.stride(0), o.stride(1), o.stride(2), o.stride(3)

        # Grid
        grid = (B * H, S_q)

        # Scale
        if softmax_scale is None:
            scale = 1.0 / math.sqrt(D)
        else:
            scale = float(softmax_scale)

        # Block sizes
        BLOCK_D = 64 if D <= 64 else 128
        BLOCK_M = 64 if S_q <= 64 else 128

        dense_mha_fwd_kernel[grid](
            q, k, v, o,
            B, H, S_q, D,
            s_q_bh, s_q_s, s_q_d,
            s_k_bh, s_k_s, s_k_d,
            s_v_bh, s_v_s, s_v_d,
            s_o_bh, s_o_s, s_o_d,
            scale,
            BLOCK_D=BLOCK_D, BLOCK_M=BLOCK_M,
            num_warps=4,
        )

        # Cast and permute back
        out = o.to(query.dtype).permute(0, 2, 1, 3)
        return out


# Provide ModelNew as requested, identical behavior
class ModelNew(Model):
    pass

DenseAttention = ModelNew
