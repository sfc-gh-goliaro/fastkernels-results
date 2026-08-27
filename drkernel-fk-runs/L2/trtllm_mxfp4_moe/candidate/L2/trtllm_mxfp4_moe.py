import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _fused_moe_expert_sigmoid_clamp_kernel(
    X,                       # [M, H] bf16
    W1_u8, W1_sf_u8, B1,     # W1[K,K1], S1[K,K1//16], b1[K1]
    W2_u8, W2_sf_u8, B2,     # W2[H,K2], S2[H,K2//16], b2[K2]
    Out,                     # [M, H] bf16
    M, H, K, K1, K2,
    stride_x_m, stride_x_n,
    # W1 strides
    stride_w1_k, stride_w1_n,
    stride_w1sf_k, stride_w1sf_n,
    stride_b1,
    # W2 strides
    stride_w2_k, stride_w2_n,
    stride_w2sf_k, stride_w2sf_n,
    stride_b2,
    # Out strides
    stride_out_m, stride_out_n,
    alpha, beta, limit,      # float32 scalars
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid: (M, ceil_div(H, BLOCK_N))
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m = pid_m
    n0 = pid_n * BLOCK_N
    offs_n = n0 + tl.arange(0, BLOCK_N)
    n_mask = offs_n < H

    # 1) y1 = X[M,K] @ W1[K,K1]
    y1 = tl.zeros((BLOCK_M, K1), dtype=tl.float32)
    x_base = X + m * stride_x_m
    for k in range(0, K, BLOCK_K):
        k_off = k + tl.arange(0, BLOCK_K)
        k_mask = k_off < K
        # load x[k_off] -> shape [BK]
        x = tl.load(x_base + k_off * stride_x_n, mask=k_mask, other=0.0).to(tl.float32)  # [BK]
        x = x[:, None]  # [BK, 1]

        # W1 tile: [BK, BN1]
        offs_n1 = tl.arange(0, BLOCK_N)
        n1 = n0 + offs_n1
        n1_mask = n1 < K1
        w1_ptrs = W1_u8 + (k_off[:, None] * stride_w1_k + n1[None, :] * stride_w1_n)
        w1sf_ptrs = W1_sf_u8 + (k_off[:, None] * stride_w1sf_k + (n1[None, :] // 16) * stride_w1sf_n)
        w_mask = (k_mask[:, None]) & (n1_mask[None, :])

        w1_u8 = tl.load(w1_ptrs, mask=w_mask, other=0).to(tl.uint8)
        w1sf_u8 = tl.load(w1sf_ptrs, mask=w_mask, other=0).to(tl.uint8)

        # dequant: (w*16 + sf) * 2^-15
        w1_val = (w1_u8.to(tl.int32) * 16 + w1sf_u8.to(tl.int32)) * (1.0 / 32768.0)
        w1_val = w1_val.to(tl.float32)  # [BK, BN1]

        prod = x[:, :, None] * w1_val[None, :, :]  # [BK,1,BN1]
        partial = tl.sum(prod, axis=0)  # [1,BN1]
        y1 += partial

    # 2) y2 = X[M,H] @ W2[H,K2] -> but K2=K1 (intermediate)
    y2 = tl.zeros((BLOCK_M, K2), dtype=tl.float32)
    x_base = X + m * stride_x_m
    for k in range(0, H, BLOCK_K):
        k_off = k + tl.arange(0, BLOCK_K)
        k_mask = k_off < H
        x = tl.load(x_base + k_off * stride_x_n, mask=k_mask, other=0.0).to(tl.float32)  # [BK]
        x = x[:, None]  # [BK,1]

        offs_n2 = tl.arange(0, BLOCK_N)
        n2 = n0 + offs_n2
        n2_mask = n2 < K2
        w2_ptrs = W2_u8 + (k_off[:, None] * stride_w2_k + n2[None, :] * stride_w2_n)
        w2sf_ptrs = W2_sf_u8 + (k_off[:, None] * stride_w2sf_k + (n2[None, :] // 16) * stride_w2sf_n)
        w_mask = (k_mask[:, None]) & (n2_mask[None, :])

        w2_u8 = tl.load(w2_ptrs, mask=w_mask, other=0).to(tl.uint8)
        w2sf_u8 = tl.load(w2sf_ptrs, mask=w_mask, other=0).to(tl.uint8)

        w2_val = (w2_u8.to(tl.int32) * 16 + w2sf_u8.to(tl.int32)) * (1.0 / 32768.0)
        w2_val = w2_val.to(tl.float32)  # [BK, BN2]

        prod = x[:, :, None] * w2_val[None, :, :]
        partial = tl.sum(prod, axis=0)
        y2 += partial

    # 3) Epilogue: SwiGLU
    t = alpha * y1 + beta
    gate = 1.0 / (1.0 + tl.exp(-t))
    up = tl.minimum(y2, limit)
    # Interleave to width H: H = 2*K1
    I = K1
    out = tl.zeros((BLOCK_M, H), dtype=tl.float32)
    for j in range(0, I):
        out[:, 2 * j] = gate[:, j]
        out[:, 2 * j + 1] = up[:, j]

    # Store as bf16
    out_bf16 = out.to(tl.bfloat16)
    out_ptrs = Out + (m * stride_out_m + offs_n * stride_out_n)
    tl.store(out_ptrs, out_bf16[0, :], mask=n_mask)


class ModelNew(nn.Module):
    """Triton-optimized fused Mx(FP4) MoE with same signature as Model.

    Entry point: ModelNew
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        intermediate_size: int,
        hidden_size_unpadded: int,
        max_capture_size: int = 1024,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.intermediate_size = intermediate_size
        self.hidden_size_unpadded = hidden_size_unpadded
        self.max_capture_size = max(int(max_capture_size), 1)

        # Constants matching original
        self.swiglu_alpha = 1.702
        self.swiglu_beta = 1.0
        self.swiglu_limit = 7.0

    def _round_up_to_256(self, x: int) -> int:
        return ((x + 255) // 256) * 256

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        w13_weight: torch.Tensor,
        w13_weight_scale: torch.Tensor,
        w13_bias: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_scale: torch.Tensor,
        w2_bias: torch.Tensor,
    ) -> torch.Tensor:
        """
        hidden_states:  [B, H_unpadded] bf16
        router_logits:  [B, E] bf16
        w13_weight:     [E, 2I, H//2] uint8
        w13_weight_scale: [E, 2I, H//32] float8
        w13_bias:       [E, 2I] float32
        w2_weight:      [E, H, I//2] uint8
        w2_weight_scale: [E, H, I//32] float8
        w2_bias:        [E, H] float32

        Returns: [B, H_unpadded] bf16
        """
        assert hidden_states.dtype == torch.bfloat16, "Expect bf16 hidden"
        assert router_logits.dtype == torch.bfloat16, "Expect bf16 router logits"

        device = hidden_states.device
        require_cuda = torch.cuda.is_available()
        if not require_cuda:
            raise RuntimeError("CUDA is required for Triton kernels.")

        B = hidden_states.shape[0]
        H_un = self.hidden_size_unpadded
        H_pad = self._round_up_to_256(H_un)
        assert hidden_states.shape[1] == H_pad, f"Hidden last dim must be {H_pad}, got {hidden_states.shape[1]}"

        E = self.num_experts
        I = self.intermediate_size
        assert 2 * I == H_pad, f"SwiGLU requires hidden=2*intermediate; got {H_pad} vs {2*I}"
        assert w13_weight.shape[1] == 2 * I, f"w13 weight dim1 must be 2*I={2*I}"
        assert w2_weight.shape[1] == H_pad, f"w2 weight dim1 must be H={H_pad}"

        # Make sure tensors are contiguous
        X = hidden_states.contiguous()
        R = router_logits.contiguous()

        # Shapes
        M = B
        H = H_pad
        K = H // 2  # input reduction dim
        K1 = I
        K2 = I

        # Output buffer
        out = torch.zeros((M, H_un), dtype=torch.bfloat16, device=device)

        # Block sizes
        BLOCK_M = 1      # one row per program
        BLOCK_N = 128
        BLOCK_K = 64

        grid = (M, triton.cdiv(H, BLOCK_N))

        if self.top_k == 1:
            # Compute softmax probs and argmax (stay on GPU, no .item())
            probs = torch.softmax(R, dim=-1)  # [M,E]
            _, top1_idx = torch.topk(probs, k=1, largest=True)  # [M,1]
            e tensor = top1_idx.squeeze(-1)  # [M]

            # Loop over rows m
            for m in range(M):
                e = int(e tensor[m].detach().cpu().item())  # Python int OK
                # Slice weights for expert e
                We = w13_weight[e]              # [2I, K]
                Se = w13_weight_scale[e]        # [2I, K//16]
                be = w13_bias[e]                # [2I]
                W2e = w2_weight[e]              # [H, I]
                S2e = w2_weight_scale[e]        # [H, I//16]
                b2e = w2_bias[e]                # [H]

                # Make contiguous
                We = We.contiguous()
                Se = Se.contiguous()
                be = be.contiguous()
                W2e = W2e.contiguous()
                S2e = S2e.contiguous()
                b2e = b2e.contiguous()

                # Prepare W1, S1, b1
                # We is [2I, K]; need W1[K, K1]. Take transpose and slice.
                W1_t = We.transpose(0, 1).contiguous()  # [K, 2I]
                W1 = W1_t[:, :K1].contiguous()          # [K, K1]
                # Scales Se is [2I, K//16]; need S1[K, K1//16] -> take first K rows, first K1//16 cols
                S1 = Se[:K, : (K1 // 16)].contiguous()
                b1 = be[:K1].contiguous()

                # W2, S2, b2 (K2=I)
                W2 = W2e.contiguous()
                S2 = S2e[:, : (K2 // 16)].contiguous()
                b2 = b2e.contiguous()

                # Allocate output buffer [M,H] bf16
                Out = torch.empty((M, H), dtype=torch.bfloat16, device=device)

                # Strides
                stride_x_m, stride_x_n = X.stride(0), X.stride(1)
                # W1
                stride_w1_k, stride_w1_n = W1.stride(0), W1.stride(1)
                stride_w1sf_k, stride_w1sf_n = S1.stride(0), S1.stride(1)
                stride_b1 = b1.stride(0)
                # W2
                stride_w2_k, stride_w2_n = W2.stride(0), W2.stride(1)
                stride_w2sf_k, stride_w2sf_n = S2.stride(0), S2.stride(1)
                stride_b2 = b2.stride(0)
                # Out
                stride_out_m, stride_out_n = Out.stride(0), Out.stride(1)

                _fused_moe_expert_sigmoid_clamp_kernel[grid](
                    X, W1, S1, b1,
                    W2, S2, b2,
                    Out,
                    M, H, K, K1, K2,
                    stride_x_m, stride_x_n,
                    stride_w1_k, stride_w1_n,
                    stride_w1sf_k, stride_w1sf_n,
                    stride_b1,
                    stride_w2_k, stride_w2_n,
                    stride_w2sf_k, stride_w2sf_n,
                    stride_b2,
                    stride_out_m, stride_out_n,
                    float(self.swiglu_alpha), float(self.swiglu_beta), float(self.swiglu_limit),
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                )

                # Weight by prob and accumulate into out
                prob = probs[m, e]  # 0-d Tensor
                tmp = Out[m, :H_un] * prob  # elementwise
                out[m, :] = tmp

            return out

        # top_k > 1: torch.topk on GPU, loop over k
        probs = torch.softmax(R, dim=-1)  # [M,E]
        vals, idxs = torch.topk(probs, k=min(self.top_k, E), largest=True, sorted=False)
        # idxs shape [M,K]; vals [M,K]
        out_acc = torch.zeros((M, H_un), dtype=torch.bfloat16, device=device)

        for k in range(self.top_k):
            # expert indices tensor [M]
            e tensor = idxs[:, k]
            # probabilities tensor [M]
            val tensor = vals[:, k]
            # Loop over rows
            for m in range(M):
                e = int(e tensor[m].detach().cpu().item())
                prob = val tensor[m]  # 0-d Tensor

                # Reuse the same weights as above (per e)
                We = w13_weight[e]; Se = w13_weight_scale[e]; be = w13_bias[e]
                W2e = w2_weight[e]; S2e = w2_weight_scale[e]; b2e = w2_bias[e]

                We = We.contiguous(); Se = Se.contiguous(); be = be.contiguous()
                W2e = W2e.contiguous(); S2e = S2e.contiguous(); b2e = b2e.contiguous()

                W1_t = We.transpose(0, 1).contiguous(); W1 = W1_t[:, :K1].contiguous()
                S1 = Se[:K, : (K1 // 16)].contiguous(); b1 = be[:K1].contiguous()
                W2 = W2e.contiguous(); S2 = S2e[:, : (K2 // 16)].contiguous(); b2 = b2e.contiguous()

                Out = torch.empty((M, H), dtype=torch.bfloat16, device=device)

                stride_x_m, stride_x_n = X.stride(0), X.stride(1)
                stride_w1_k, stride_w1_n = W1.stride(0), W1.stride(1)
                stride_w1sf_k, stride_w1sf_n = S1.stride(0), S1.stride(1)
                stride_b1 = b1.stride(0)
                stride_w2_k, stride_w2_n = W2.stride(0), W2.stride(1)
                stride_w2sf_k, stride_w2sf_n = S2.stride(0), S2.stride(1)
                stride_b2 = b2.stride(0)
                stride_out_m, stride_out_n = Out.stride(0), Out.stride(1)

                _fused_moe_expert_sigmoid_clamp_kernel[grid](
                    X, W1, S1, b1,
                    W2, S2, b2,
                    Out,
                    M, H, K, K1, K2,
                    stride_x_m, stride_x_n,
                    stride_w1_k, stride_w1_n,
                    stride_w1sf_k, stride_w1sf_n,
                    stride_b1,
                    stride_w2_k, stride_w2_n,
                    stride_w2sf_k, stride_w2sf_n,
                    stride_b2,
                    stride_out_m, stride_out_n,
                    float(self.swiglu_alpha), float(self.swiglu_beta), float(self.swiglu_limit),
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                )

                tmp = Out[m, :H_un] * prob
                out_acc[m, :] = out_acc[m, :] + tmp

        return out_acc

TrtLlmMxfp4MoE = ModelNew
