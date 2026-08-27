import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _fused_moe_topk_two_gemm_swiglu_kernel(
    # pointers
    x_ptr,                   # [B, H] bf16
    w13_ptr,                 # [E, (2*I)*H] bf16, contiguous
    w2_ptr,                  # [E, H*I] bf16, contiguous
    router_ptr,              # [B, E] float (any), we'll cast
    bias_ptr,                # [E] or nullptr
    y_ptr,                   # [B, H] fp32 (output)

    # sizes (constexpr)
    B: tl.constexpr,
    H: tl.constexpr,
    E: tl.constexpr,
    I: tl.constexpr,         # intermediate
    K1: tl.constexpr,        # = 2*I
    K2: tl.constexpr,        # = I

    # strides in elements
    x_stride_b: tl.constexpr,
    x_stride_h: tl.constexpr,
    w13_stride_e: tl.constexpr,
    w13_stride_k: tl.constexpr,   # should be 1
    w2_stride_e: tl.constexpr,
    w2_stride_k: tl.constexpr,    # should be 1
    router_stride_b: tl.constexpr,
    router_stride_e: tl.constexpr,
    bias_stride_e: tl.constexpr,  # if used
    y_stride_b: tl.constexpr,
    y_stride_h: tl.constexpr,

    # tiling
    BLOCK_M: tl.constexpr,   # tile in output H
    BLOCK_K1: tl.constexpr,  # tile in H for GEMM1
    BLOCK_K2: tl.constexpr,  # tile in I for GEMM2

    # routing
    top_k: tl.constexpr,          # 1 or 2
    routing_method_type: tl.constexpr,  # 1=renormalize, 2=deepseek_v3, others=standard softmax

    # program id
    pid_b: tl.constexpr,
):
    b = pid_b

    # -----------------------------
    # 1) Compute routing: scores, top-2
    # -----------------------------
    # Load and cast router to fp32
    scores = tl.zeros((E,), dtype=tl.float32)
    for e in range(0, E):
        s = tl.load(router_ptr + b * router_stride_b + e * router_stride_e)
        scores[e] = s.to(tl.float32)
    # Add bias if provided
    has_bias = bias_ptr != 0
    if has_bias:
        for e in range(0, E):
            scores[e] += tl.load(bias_ptr + e * bias_stride_e).to(tl.float32)

    # Compute lse stably: m = max; sum exp(s-m)
    m = -float("inf")
    for e in range(0, E):
        m = tl.maximum(m, scores[e])
    sumexp = 0.0
    for e in range(0, E):
        sumexp += tl.exp(scores[e] - m)
    lse = m + tl.log(sumexp)

    # Top-1 and Top-2 (track values and indices)
    top1_val = -float("inf")
    top1_idx = 0
    top2_val = -float("inf")
    top2_idx = 0
    for e in range(0, E):
        if scores[e] > top1_val:
            top2_val = top1_val
            top2_idx = top1_idx
            top1_val = scores[e]
            top1_idx = e
        elif scores[e] > top2_val:
            top2_val = scores[e]
            top2_idx = e

    # Weights
    if routing_method_type == 1 or routing_method_type == 4:  # renormalize or naive
        # softmax / lse
        w1 = tl.exp(top1_val - lse)
        if top_k == 1:
            w2 = 0.0
            e2 = 0
        else:
            w2 = tl.exp(top2_val - lse)
            e2 = top2_idx
    elif routing_method_type == 2:  # deepseek_v3
        numer = tl.exp(top1_val - lse)
        denom = numer + tl.exp(top2_val - lse)
        w1 = numer / denom
        w2 = 1.0 - w1
        e2 = top2_idx
    else:  # standard softmax
        w1 = tl.exp(top1_val) / tl.exp(lse)  # incorrect; fallback to softmax over selected
        # Fallback: use renormalize logic
        w1 = tl.exp(top1_val - lse)
        if top_k == 1:
            w2 = 0.0
            e2 = 0
        else:
            w2 = tl.exp(top2_val - lse)
            e2 = top2_idx

    # -----------------------------
    # 2) Accumulator for final output
    # -----------------------------
    y_accum = tl.zeros((H,), dtype=tl.float32)

    # Helper to compute per-expert contribution
    def contribute(e_idx, weight):
       非局部 y_accum
        # First GEMM: u = w13[e] @ x  -> shape (K1,)
        u = tl.zeros((K1,), dtype=tl.float32)
        base_w13 = e_idx * (K1 * H)
        for kk in range(0, K1, BLOCK_K1):
            k = kk + tl.arange(0, BLOCK_K1)
            k_mask = k < K1
            u_sub = tl.zeros((BLOCK_K1,), dtype=tl.float32)
            for kh in range(0, H, BLOCK_K1):
                h = kh + tl.arange(0, BLOCK_K1)
                h_mask = h < H
                x_vec = tl.load(
                    x_ptr + b * x_stride_b + h * x_stride_h,
                    mask=h_mask,
                    other=0.0,
                ).to(tl.float32)  # [BK1]
                ptr = w13_ptr + base_w13 + k[:, None] * H + h[None, :]
                mask = k_mask[:, None] & h_mask[None, :]
                w = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)  # [BK1, BH]
                prod = w * x_vec[None, :]
                u_sub += tl.sum(prod, axis=1)
            u[kk: kk + BLOCK_K1] = u_sub

        # SwiGLU: v = gate * x * sqrt(2)
        gate = tl.sigmoid(u[:I])
        x2 = u[I:]
        sqrt2 = 1.4142135623730951
        v = gate * x2 * sqrt2  # [I]

        # Second GEMM: out = w2[e] @ v -> [H]
        out_tile = tl.zeros((H,), dtype=tl.float32)
        base_w2 = e_idx * (H * K2)
        for n in range(0, H, BLOCK_M):
            n_idx = n + tl.arange(0, BLOCK_M)
            n_mask = n_idx < H
            out_sub = tl.zeros((BLOCK_M,), dtype=tl.float32)
            for k2 in range(0, K2, BLOCK_K2):
                k2_idx = k2 + tl.arange(0, BLOCK_K2)
                k2_mask = k2_idx < K2
                ptr_w2 = w2_ptr + base_w2 + n_idx[:, None] * K2 + k2_idx[None, :]
                mask_w2 = n_mask[:, None] & k2_mask[None, :]
                w_rows = tl.load(ptr_w2, mask=mask_w2, other=0.0).to(tl.float32)  # [BM, BK2]
                v_sub = tl.load(v + k2_idx, mask=k2_mask, other=0.0).to(tl.float32)  # [BK2]
                prod = w_rows * v_sub[None, :]
                out_sub += tl.sum(prod, axis=1)
            out_tile[n:n+BLOCK_M] = out_sub
        # Scale and accumulate
        y_accum += weight * out_tile

    # Contribution from top-1
    contribute(top1_idx, w1)
    # Contribution from top-2 if any
    if top_k == 2:
        contribute(e2, w2)

    # Store result
    tl.store(y_ptr + b * y_stride_b + tl.arange(0, H) * y_stride_h, y_accum)


class ModelNew(nn.Module):
    """Fused Triton kernel: routing (top-k) + two GEMMs + SwiGLU + weighted sum.

    Keeps the same constructor and forward signature as the original Model.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        intermediate_size_per_partition: int,
        routing_method_type: int = 1,
        local_expert_offset: int = 0,
        local_num_experts: int | None = None,
        num_expert_group: int | None = None,
        topk_group: int | None = None,
        routed_scaling_factor: float | None = None,
        tune_max_num_tokens: int = DEFAULT_TUNE_MAX_NUM_TOKENS,
    ):
        super().__init__()
        # Store config
        self.num_experts = num_experts
        self.top_k = top_k
        self.intermediate_size_per_partition = intermediate_size_per_partition
        self.routing_method_type = routing_method_type
        self.local_expert_offset = local_expert_offset
        self.local_num_experts = num_experts if local_num_experts is None else local_num_experts
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.tune_max_num_tokens = tune_max_num_tokens

        # Tiling defaults
        self._BLOCK_M = 128
        self._BLOCK_K1 = 64
        self._BLOCK_K2 = 64

    def forward(
        self,
        hidden_states: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        routing_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Fused forward:
          - Compute gating scores (with bias if provided).
          - Select top-k experts.
          - For each selected expert: compute u = w13 @ x, v = SwiGLU(u), out = w2 @ v.
          - Weighted sum and store y.

        Returns: tensor of shape [B, H] (float32).
        """
        if not hidden_states.is_cuda or not w13.is_cuda or not w2.is_cuda:
            raise RuntimeError("Triton kernel requires CUDA tensors.")

        device = hidden_states.device

        # Ensure layouts and dtypes
        x = hidden_states
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        x = x.contiguous()

        # w13: [E, 2*I, H] -> [E, (2*I)*H]
        E = w13.shape[0]
        twoI = w13.shape[1]
        H = w13.shape[2]
        I = self.intermediate_size_per_partition
        if twoI != 2 * I:
            raise ValueError(f"Expected w13 second dim 2*I={2*I}, got {twoI}.")
        w13_flat = w13.view(E, twoI * H).contiguous()

        # w2:  [E, H, I] -> [E, H*I]
        if w2.shape[0] != E or w2.shape[1] != H or w2.shape[2] != I:
            raise ValueError(f"w2 shape {w2.shape} incompatible with H={H}, I={I}.")
        w2_flat = w2.view(E, H * I).contiguous()

        B = x.shape[0]
        if x.shape[1] != H:
            raise ValueError(f"hidden_states second dim {x.shape[1]} != H={H}.")

        # Output
        y = torch.empty((B, H), device=device, dtype=torch.float32)

        # Router and bias
        router = router_logits
        if router.dtype != torch.float32:
            router = router.to(torch.float32)
        router = router.contiguous()
        bias = routing_bias
        if bias is not None:
            bias = bias.to(torch.float32).contiguous()
        else:
            bias = torch.empty(1, device=device, dtype=torch.float32)  # dummy

        # Strides
        x_stride_b = x.stride(0)
        x_stride_h = x.stride(1)
        w13_stride_e = w13_flat.stride(0)
        w13_stride_k = w13_flat.stride(1)  # 1
        w2_stride_e = w2_flat.stride(0)
        w2_stride_k = w2_flat.stride(1)    # 1
        router_stride_b = router.stride(0)
        router_stride_e = router.stride(1)
        bias_stride_e = bias.stride(0) if bias.dtype != torch.float32 else bias.stride(0)
        y_stride_b = y.stride(0)
        y_stride_h = y.stride(1)

        # Grid: one program per batch row
        grid = (B,)

        _fused_moe_topk_two_gemm_swiglu_kernel[grid](
            x, w13_flat, w2_flat, router, bias if bias is not None else torch.tensor([], device=device), y,
            B, H, E, I, 2 * I, I,
            x_stride_b, x_stride_h,
            w13_stride_e, w13_stride_k,
            w2_stride_e, w2_stride_k,
            router_stride_b, router_stride_e,
            bias_stride_e,
            y_stride_b, y_stride_h,
            self._BLOCK_M, self._BLOCK_K1, self._BLOCK_K2,
            self.top_k, self.routing_method_type,
            pid_b=tl.program_id(0),
            num_warps=4, num_stages=2,
        )

        return y

TrtLlmBf16MoE = ModelNew
