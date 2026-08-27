import torch
import torch.nn as nn

# Try to import Triton
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


if _HAS_TRITON:
    @triton.jit
    def _fused_softmax_attn_matmul_kernel(
        Q, K, V, Mask, Out,
        # Shapes
        B: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
        Hheads: tl.constexpr,        # num_heads
        # Strides (elements)
        stride_qb, stride_qs, stride_qh, stride_qd,
        stride_kb, stride_ks, stride_kh, stride_kd,
        stride_vb, stride_vs, stride_vh, stride_vd,
        stride_mb, stride_mh, stride_ms, stride_mi,
        stride_ob, stride_os, stride_oh, stride_od,
        # Constants
        scale: tl.float32,
        # Tiling
        BLOCK_D: tl.constexpr,
    ):
        # One program per (b, hhead)
        pid = tl.program_id(0)
        b = pid // Hheads
        hhead = pid % Hheads

        # Base pointers for this (b, hhead)
        q_base = Q + b * stride_qb + hhead * stride_qh
        k_base = K + b * stride_kb + hhead * stride_kh
        v_base = V + b * stride_vb + hhead * stride_vh
        o_base = Out + b * stride_ob + hhead * stride_oh

        # Vector over D
        offs = tl.arange(0, BLOCK_D)

        # Loop over query row t
        t = 0
        while t < S:
            # Pass 1: compute Z (denom) and p array
            m = -float("inf")
            Z = 0.0
            p = tl.zeros([S], dtype=tl.float32)

            for i in range(0, S):
                # Load q_t (D)
                q = tl.zeros([D], dtype=tl.float32)
                for dd in range(0, D, BLOCK_D):
                    d = dd + offs
                    mask_d = d < D
                    q[d] = tl.load(q_base + t * stride_qs + d * stride_qd, mask=mask_d, other=0.0)
                # Load k_i (D)
                k = tl.zeros([D], dtype=tl.float32)
                for dd in range(0, D, BLOCK_D):
                    d = dd + offs
                    mask_d = d < D
                    k[d] = tl.load(k_base + i * stride_ks + d * stride_kd, mask=mask_d, other=0.0)
                # Dot
                dot = tl.sum(q * k, axis=0)
                l = dot * scale
                # Mask
                msk = tl.load(Mask + b * stride_mb + 0 * stride_mh + t * stride_ms + i * stride_mi)
                l = l + msk
                # Update m, Z, p
                m_new = tl.maximum(m, l)
                Z = Z * tl.exp(m - m_new) + tl.exp(l - m_new)
                p_i = tl.exp(l - m_new)
                p[i] = p_i
                m = m_new

            # s_t
            s_t = 0.0
            if t < S:
                s_t = p[t] / Z

            # Second pass: y = sum_i p_i/Z * V[i]
            y = tl.zeros([D], dtype=tl.float32)
            for i in range(0, S):
                v = tl.zeros([D], dtype=tl.float32)
                for dd in range(0, D, BLOCK_D):
                    d = dd + offs
                    mask_d = d < D
                    v[d] = tl.load(v_base + i * stride_vs + d * stride_vd, mask=mask_d, other=0.0)
                s_i = p[i] / Z
                y = y + s_i * v

            # Store y[t, :] as Out[b, t, hhead, 0: D]
            for dd in range(0, D, BLOCK_D):
                d = dd + offs
                mask_d = d < D
                tl.store(o_base + t * stride_os + 0 * stride_oh + d * stride_od, y[d], mask=mask_d)

            t += 1


class ModelNew(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        # Projections (cuBLAS-backed)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = hidden_states.device
        if not _HAS_TRITON or device.type != "cuda":
            # Fallback to PyTorch reference
            queries = self.q_proj(hidden_states)
            keys = self.k_proj(hidden_states)
            values = self.v_proj(hidden_states)

            B, S = hidden_states.shape[0], hidden_states.shape[1]
            # View to [B, S, H, D]
            queries = queries.view(B, S, self.num_heads, self.head_dim)
            keys = keys.view(B, S, self.num_heads, self.head_dim)
            values = values.view(B, S, self.num_heads, self.head_dim)

            attn_weights = torch.bmm(
                queries.view(B * self.num_heads, S, self.head_dim),
                keys.view(B * self.num_heads, self.head_dim, S),
            ).view(B, self.num_heads, S, S)
            attn_weights = attn_weights * self.scale
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_probs = torch.softmax(attn_weights, dim=-1)

            attn_output = torch.bmm(
                attn_probs.view(B * self.num_heads, S, S),
                values.view(B * self.num_heads, S, self.head_dim),
            ).view(B, self.num_heads, S, self.head_dim)
            attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
            return self.out_proj(attn_output)

        # Triton path: fuse softmax(QK^T+mask) @ V -> [B,S,H,D]; then out_proj
        B, S, H = hidden_states.shape
        D = self.head_dim
        assert H == self.num_heads * D, "embed_dim must equal num_heads * head_dim"

        # 1) Projections
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        # 2) View to [B, S, H, D] and make contiguous for simpler strides
        queries = queries.view(B, S, self.num_heads, D).contiguous()
        keys = keys.view(B, S, self.num_heads, D).contiguous()
        values = values.view(B, S, self.num_heads, D).contiguous()

        # 3) Dtypes: float32
        queries = queries.float()
        keys = keys.float()
        values = values.float()

        # 4) Attention mask
        if attention_mask is not None:
            assert attention_mask.shape[-2:] == (S, S), f"Expected mask last dims {S,S}, got {attention_mask.shape[-2:]}"
            attention_mask = attention_mask.to(device=device, dtype=torch.float32)
        else:
            attention_mask = torch.zeros((1, 1, S, S), device=device, dtype=torch.float32)

        # 5) Allocate intermediate OutInt [B, S, H, D] = float output of (softmax@V)
        OutInt = torch.empty((B, S, self.num_heads, D), device=device, dtype=torch.float32)

        # Strides (elements)
        stride_qb, stride_qs, stride_qh, stride_qd = queries.stride(0), queries.stride(1), queries.stride(2), queries.stride(3)
        stride_kb, stride_ks, stride_kh, stride_kd = keys.stride(0), keys.stride(1), keys.stride(2), keys.stride(3)
        stride_vb, stride_vs, stride_vh, stride_vd = values.stride(0), values.stride(1), values.stride(2), values.stride(3)

        MB, MH, MS, MI = attention_mask.shape
        stride_mb, stride_mh, stride_ms, stride_mi = attention_mask.stride(0), attention_mask.stride(1), attention_mask.stride(2), attention_mask.stride(3)

        stride_ob, stride_os, stride_oh, stride_od = OutInt.stride(0), OutInt.stride(1), OutInt.stride(2), OutInt.stride(3)

        # Grid: one program per (b, h)
        grid = (B * self.num_heads,)

        # Launch fused kernel
        BLOCK_D = 64  # match head_dim
        _fused_softmax_attn_matmul_kernel[grid](
            queries, keys, values, attention_mask, OutInt,
            B, S, D, self.num_heads,
            stride_qb, stride_qs, stride_qh, stride_qd,
            stride_kb, stride_ks, stride_kh, stride_kd,
            stride_vb, stride_vs, stride_vh, stride_vd,
            stride_mb, stride_mh, stride_ms, stride_mi,
            stride_ob, stride_os, stride_oh, stride_od,
            self.scale,
            BLOCK_D=BLOCK_D,
            num_warps=4,
            num_stages=2,
        )

        # 6) Final out_proj: [B,S,H,D] -> [B,S,H]
        # View as [B*S*H, D] @ [D,H] -> [B*S*H, H]
        OutInt = OutInt.view(B * S * self.num_heads, D)
        WOut = self.out_proj.weight  # [H, H] == [D*Hheads, H] if we think of H= D*Hheads ? No: H is output dim == embed Dim == D*Hheads
        # Wait: out_proj is [embed, embed] == [H, H]. We need [B,S,H,D] @ W^T -> [B,S,H].
        # So treat OutInt as [A, D] where A=B*S*Hheads, W^T as [D, H].
        # But WOut is [H, H]; we need [D, H]: take WOut^T reshaped.
        # Easiest: use torch.mm
        Out2D = OutInt @ WOut  # [B*S*Hheads, H]
        Out = Out2D.view(B, S, self.num_heads, H).sum(dim=2)  # sum over heads? No: out shape is [B,S,H]; each head contributes独立
        # Correction: out_proj should be applied per token and per head, then concatenated over head dim isn't correct.
        # Better: reshape back to [B,S,Hheads] and add bias.
        Out = Out2D.view(B, S, self.num_heads, H)  # [B,S,Hheads,H]
        # That's not right. Simplify: use torch.bmm per head is overkill here.
        # Instead, do a small GEMM with torch: OutInt [B,S,H,D] -> [B,S,H*D] @ W [H,H] -> [B,S,H].
        OutInt2 = OutInt.view(B, S, self.num_heads * D)
        Out = OutInt2 @ WOut + self.out_proj.bias

        return Out

CLIPAttention = ModelNew
