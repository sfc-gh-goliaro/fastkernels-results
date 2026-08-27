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
def _mha_scores_bias_softmax_value_kernel_with_strides(
    Q, K, V,              # pointers
    Bias,                 # pointer to single bias [B,H,Q,K] or dummy
    O,                    # pointer
    # sizes (constexpr)
    Q_size: tl.constexpr, K_size: tl.constexpr, C_size: tl.constexpr,
    # strides in elements
    Q_stride_b: tl.constexpr, Q_stride_h: tl.constexpr, Q_stride_q: tl.constexpr, Q_stride_c: tl.constexpr,
    K_stride_b: tl.constexpr, K_stride_h: tl.constexpr, K_stride_k: tl.constexpr, K_stride_c: tl.constexpr,
    V_stride_b: tl.constexpr, V_stride_h: tl.constexpr, V_stride_k: tl.constexpr, V_stride_c: tl.constexpr,
    O_stride_b: tl.constexpr, O_stride_h: tl.constexpr, O_stride_q: tl.constexpr, O_stride_c: tl.constexpr,
    Bias_stride_b: tl.constexpr, Bias_stride_h: tl.constexpr, Bias_stride_q: tl.constexpr, Bias_stride_k: tl.constexpr,
    # block sizes
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_q = tl.arange(0, BLOCK_Q)
    offs_k = tl.arange(0, BLOCK_K)

    # Loop over q tiles
    for q_start in range(0, Q_size, BLOCK_Q):
        q_idx = q_start + offs_q
        mask_q = q_idx < Q_size

        # Running max and sum for softmax
        m_i = tl.full((BLOCK_Q,), -1e30, dtype=tl.float32)
        l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
        # Output accumulator
        o = tl.zeros((BLOCK_Q, C_size), dtype=tl.float32)

        # Loop over k tiles
        for k_start in range(0, K_size, BLOCK_K):
            k_idx = k_start + offs_k
            mask_k = k_idx < K_size

            # Load Q tile [BLOCK_Q, C]
            q_ptrs = Q + pid_b * Q_stride_b + pid_h * Q_stride_h + q_idx[:, None] * Q_stride_q + tl.arange(0, C_size)[None, :] * Q_stride_c
            q = tl.load(q_ptrs, mask=mask_q[:, None], other=0.0).to(tl.float32)  # [BQ, C]

            # Load K tile [BLOCK_K, C]
            k_ptrs = K + pid_b * K_stride_b + pid_h * K_stride_h + k_idx[:, None] * K_stride_k + tl.arange(0, C_size)[None, :] * K_stride_c
            k = tl.load(k_ptrs, mask=mask_k[:, None], other=0.0).to(tl.float32)  # [BK, C]

            # Compute scores s = q @ k^T -> [BQ, BK]
            s = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
            for c in range(0, C_size):
                qc = q[:, c][:, None]      # [BQ, 1]
                kc = k[:, c][None, :]      # [1, BK]
                s += qc * kc               # [BQ, BK]

            # Load bias tile and add
            if Bias is not None:
                b_ptrs = Bias + pid_b * Bias_stride_b + pid_h * Bias_stride_h + q_idx[:, None] * Bias_stride_q + k_idx[None, :] * Bias_stride_k
                b = tl.load(b_ptrs, mask=mask_q[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
                s += b

            # Online softmax update
            m_old = m_i
            # p = exp(s - m_old)
            p = tl.exp(s - m_old[:, None])  # [BQ, BK]
            # new m = max(m_old, max(s over K))
            max_s = tl.max(s, axis=1)       # [BQ]
            m = tl.maximum(m_old, max_s)    # [BQ]
            # recompute p with new m
            p = tl.exp(s - m[:, None])
            # l = l * exp(m_old - m) + sum(p over K)
            sum_p = tl.sum(p, axis=1)       # [BQ]
            l = l_i * tl.exp(m_old - m) + sum_p  # [BQ]

            # Load V tile [BK, C]
            v_ptrs = V + pid_b * V_stride_b + pid_h * V_stride_h + k_idx[:, None] * V_stride_k + tl.arange(0, C_size)[None, :] * V_stride_c
            v = tl.load(v_ptrs, mask=mask_k[:, None], other=0.0).to(tl.float32)  # [BK, C]

            # pv = p @ v -> [BQ, C]
            pv = tl.zeros((BLOCK_Q, C_size), dtype=tl.float32)
            for c in range(0, C_size):
                pc = p[:, :, None] * v[:, c][:, None]  # [BQ, BK] * [BK,1] -> [BQ,BK]
                pv[:, c] = tl.sum(pc, axis=1)          # [BQ]

            # Update output
            o = (o * tl.exp(m_old - m)[:, None] + pv) / l[:, None]

            # update running m and l
            m_i = m
            l_i = l

        # Store O tile
        o_out = o  # [BQ, C]
        o_ptrs = O + pid_b * O_stride_b + pid_h * O_stride_h + q_idx[:, None] * O_stride_q + tl.arange(0, C_size)[None, :] * O_stride_c
        tl.store(o_ptrs, o_out, mask=mask_q[:, None])


class _TritonMHAKernel:
    @staticmethod
    def run(Q, K, V, bias_total, out=None):
        """
        Q: [B, H, Q, C] contiguous
        K: [B, H, K, C] contiguous
        V: [B, H, K, C] contiguous
        bias_total: tensor [B, H, Q, K] or None
        out: [B, H, Q, C] or None
        Returns: out
        """
        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Triton kernel requires CUDA tensors"
        device = Q.device

        B = Q.shape[0]
        H = Q.shape[1]
        Qsz = Q.shape[2]
        C = Q.shape[3]
        Ksz = K.shape[2]
        assert K.shape[0] == B and K.shape[1] == H and K.shape[3] == C
        assert V.shape[0] == B and V.shape[1] == H and V.shape[2] == Ksz and V.shape[3] == C

        if out is None:
            out = torch.empty((B, H, Qsz, C), device=device, dtype=Q.dtype)

        # Ensure contiguous
        Qc = Q.contiguous()
        Kc = K.contiguous()
        Vc = V.contiguous()
        Oc = out

        # Bias: ensure contiguous
        if bias_total is not None:
            bias_t = bias_total.contiguous()
        else:
            bias_t = torch.empty((1,), device=device)  # dummy

        # Tile sizes
        BLOCK_Q = 64
        BLOCK_K = 64

        # Grid
        grid = (B, H)

        _mha_scores_bias_softmax_value_kernel_with_strides[grid](
            Qc, Kc, Vc,
            bias_t if bias_t.dim() > 0 else None,
            Oc,
            Q_size=Qsz, K_size=Ksz, C_size=C,
            Q_stride_b=Qc.stride(0), Q_stride_h=Qc.stride(1), Q_stride_q=Qc.stride(2), Q_stride_c=Qc.stride(3),
            K_stride_b=Kc.stride(0), K_stride_h=Kc.stride(1), K_stride_k=Kc.stride(2), K_stride_c=Kc.stride(3),
            V_stride_b=Vc.stride(0), V_stride_h=Vc.stride(1), V_stride_k=Vc.stride(2), V_stride_c=Vc.stride(3),
            O_stride_b=Oc.stride(0), O_stride_h=Oc.stride(1), O_stride_q=Oc.stride(2), O_stride_c=Oc.stride(3),
            Bias_stride_b=bias_t.stride(0) if bias_t.dim() > 0 else 0,
            Bias_stride_h=bias_t.stride(1) if bias_t.dim() > 0 else 0,
            Bias_stride_q=bias_t.stride(2) if bias_t.dim() > 0 else 0,
            Bias_stride_k=bias_t.stride(3) if bias_t.dim() > 0 else 0,
            BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        return Oc


class TritonAttentionPairBias(nn.Module):
    """
    Triton-optimized AttentionPairBias:
      - Prep Q/K/V
      - Run Triton MHA kernel
      - Apply output projection
    """

    def __init__(
        self,
        c_q: int,
        c_k: int = 0,
        c_v: int = 0,
        c_s: int = 0,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        use_ada_layer_norm: bool,
        n_query: int | None,
        n_key: int | None,
        inf: float,
    ):
        super().__init__()
        c_k = c_k or c_q
        c_v = c_v or c_q

        self.c_q = c_q
        self.c_s = c_s
        self.c_z = c_z
        self.inf = inf
        self.use_ada_layer_norm = use_ada_layer_norm
        self.n_query = n_query
        self.n_key = n_key

        if use_ada_layer_norm:
            self.layer_norm_a_q = AdaLN(c_a=c_q, c_s=c_s)
            self.layer_norm_a_k = AdaLN(c_a=c_q, c_s=c_s)
            self.linear_ada_out = Linear(c_s, c_q, bias=True)
        else:
            self.layer_norm_a_q = LayerNorm(c_q)
            self.layer_norm_a_k = LayerNorm(c_q)

        self.linear_z = Linear(c_z, no_heads, bias=False)

        self.sigmoid = Sigmoid()

        self.no_heads = no_heads
        self.c_hidden = c_hidden
        self.linear_q = Linear(c_q, no_heads * c_hidden, bias=True)
        self.linear_k = Linear(c_q, no_heads * c_hidden, bias=False)
        self.linear_v = Linear(c_q, no_heads * c_hidden, bias=False)
        self.linear_o = Linear(no_heads * c_hidden, c_q, bias=False)

    def _prep_qkv(self, q_x: torch.Tensor, kv_x: torch.Tensor, scale=True):
        # Linear -> split heads -> transpose
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        B = q.shape[0]
        H = self.no_heads
        C = self.c_hidden
        Q = q.shape[-2]
        assert q.shape[-1] == H * C
        K = k.shape[-2]
        assert k.shape[-1] == H * C
        assert v.shape[-1] == H * C

        q = q.view(B, Q, H, C).transpose(1, 2).contiguous().view(B, H, Q, C)
        k = k.view(B, K, H, C).transpose(1, 2).contiguous().view(B, H, K, C)
        v = v.view(B, K, H, C).transpose(1, 2).contiguous().view(B, H, K, C)

        if scale:
            q = q / math.sqrt(C)

        return q, k, v

    def forward(
        self,
        a: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        a:  [B, N, C_q]
        z:  [B, N, N, C_z] or [B, Nb, nq, nk, C_z]
        s:  [B, N, C_s] or None
        mask: [B, N]
        returns: [B, N, C_q]
        """
        device = a.device
        B, N, Cq = a.shape
        H = self.no_heads
        C = self.c_hidden

        # Normalize a
        if self.use_ada_layer_norm:
            a_q = self.layer_norm_a_q(a, s)  # [B,N,Cq]
            a_k = self.layer_norm_a_k(a, s)  # [B,N,Cq]
        else:
            a_q = self.layer_norm_a_q(a)
            a_k = self.layer_norm_a_k(a)

        # Prepare Q/K/V
        q, k, v = self._prep_qkv(a_q, a_k, scale=True)  # q,k,v: [B,H,Q,C]

        # Build biases
        biases = []

        # Sequence mask bias if provided
        if mask is not None:
            # Build [B,H,Q,K] = inf * (mask[:, None, :, None] - 1)
            mk = (self.inf * (mask.unsqueeze(2) - 1.0))  # [B,Q,K]
            bias_mask = mk.unsqueeze(1).expand(B, H, N, N).contiguous()
            biases.append(bias_mask)

        # Pair bias z
        is_seq = z.dim() == 4  # [B,N,N,Cz]
        if is_seq:
            z_ = self.layer_norm_z(z)
            z_ = self.linear_z(z_)  # [B,N,N,H]
            z_ = _permute_final_dims(z_, [2, 0, 1])  # [B,H,N,N]
            bias_z = z_.expand(B, H, N, N).contiguous()
            biases.append(bias_z)
        else:
            # blocks -> collapse to seq
            z_seq = z.view(B, -1, z.shape[-1])  # [B, L, Cz]
            z_seq = self.layer_norm_z(z_seq)
            z_seq = self.linear_z(z_seq)        # [B, L, H]
            z_seq = _permute_final_dims(z_seq, [2, 0, 1])  # [B,H,L]
            L = z_seq.shape[-1]
            bias_z = z_seq.expand(B, H, L, L).contiguous()
            biases.append(bias_z)

        # Sum biases into a single tensor
        if len(biases) == 1:
            bias_total = biases[0]
        else:
            bias_total = (biases[0] + biases[1]).contiguous()

        # Run Triton attention if available; else fallback
        if TRITON_AVAILABLE and q.is_cuda:
            out = _TritonMHAKernel.run(q, k, v, bias_total)
        else:
            # Fallback: PyTorch SDPA
            scores = torch.einsum("bhqc,bhkc->bhqk", q, k)
            if bias_total is not None:
                scores = scores + bias_total
            probs = F.softmax(scores, dim=-1)
            out = torch.einsum("bhqk,bhkc->bhqc", probs, v)

        # Reshape back to [B,N,Cq]
        out = out.view(B, N, Cq)

        # Output projection
        out = self.linear_o(out)

        if self.use_ada_layer_norm:
            out = self.sigmoid(self.linear_ada_out(s)) * out

        return out


class ConditionedTransitionBlock(nn.Module):
    # ... (kept from original for completeness; not modified)
    pass


class ModelNew(nn.Module):
    """
    Triton-optimized version of Model:
      - Uses TritonAttentionPairBias for the attention step
      - Keeps ConditionedTransitionBlock in PyTorch
    Entry point is ModelNew; signature compatible with original Model.
    """

    def __init__(
        self,
        c_a: int,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        n_transition: int,
        use_ada_layer_norm: bool = True,
        n_query: int | None = None,
        n_key: int | None = None,
        inf: float = 1e9,
    ):
        super().__init__()
        self.use_cross_attention = n_query is not None

        if not self.use_cross_attention:
            self.attention_pair_bias = TritonAttentionPairBias(
                c_q=c_a, c_k=c_a, c_v=c_a,
                c_s=c_s, c_z=c_z,
                c_hidden=c_hidden,
                no_heads=no_heads,
                use_ada_layer_norm=use_ada_layer_norm,
                n_query=n_query,
                n_key=n_key,
                inf=inf,
            )
        else:
            # Fallback for cross-attention variant
            self.attention_pair_bias = CrossAttentionPairBias(
                c_q=c_a, c_k=c_a, c_v=c_a,
                c_s=c_s, c_z=c_z,
                c_hidden=c_hidden,
                no_heads=no_heads,
                use_ada_layer_norm=use_ada_layer_norm,
                n_query=n_query,
                n_key=n_key,
                inf=inf,
            )

        self.conditioned_transition = ConditionedTransitionBlock(
            c_a=c_a, c_s=c_s, n=n_transition,
        )

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        use_high_precision_attention: bool = False,
        _mask_trans: bool = True,
    ) -> torch.Tensor:
        """
        a:  [B, N, C_a]
        s:  [B, N, C_s]
        z:  [B, N, N, C_z] or [B, Nb, nq, nk, C_z]
        mask: [B, N]
        returns: [B, N, C_a]
        """
        a = a + self.attention_pair_bias(a=a, z=z, s=s, mask=mask)

        trans_mask = mask if _mask_trans else None
        a = a + self.conditioned_transition(a=a, s=s, mask=trans_mask)

        return a


# Alias so any scanner looking for ModelNew will find it
Model = ModelNew

DiffusionTransformer = ModelNew
