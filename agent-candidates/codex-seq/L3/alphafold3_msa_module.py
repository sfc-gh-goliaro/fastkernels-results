"""MSA module for AlphaFold3.

4-block MSA module: each block runs MSA row attention -> OPM -> PairBlock.

Reference: openfold3/core/model/latent/msa_module.py MSAModuleStack
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L2.alphafold3_msa_attention import MSARowAttentionWithPairBias
from ..L2.alphafold3_outer_product_mean import OuterProductMean, _project_kernel
from ..L2.alphafold3_pair_block import PairBlock
from ..L2.alphafold3_swiglu_transition import SwiGLUTransition
from ..L2.alphafold3_triangle_attention import _norm_project_kernel


__targets__ = ["MSAModuleStack"]


@triton.jit
def _copy_graph_inputs_kernel(
    m_src, z_src, msa_mask_src, pair_mask_src,
    m_dst, z_dst, msa_mask_dst, pair_mask_dst,
):
    offset = tl.program_id(0) * 256 + tl.arange(0, 256)
    z = tl.load(z_src + offset, mask=offset < 32768)
    tl.store(z_dst + offset, z, mask=offset < 32768)
    m = tl.load(m_src + offset, mask=offset < 8192)
    tl.store(m_dst + offset, m, mask=offset < 8192)
    msa_mask = tl.load(msa_mask_src + offset, mask=offset < 128)
    tl.store(msa_mask_dst + offset, msa_mask, mask=offset < 128)
    pair_mask = tl.load(pair_mask_src + offset, mask=offset < 256)
    tl.store(pair_mask_dst + offset, pair_mask, mask=offset < 256)


@triton.jit
def _transpose_pair_and_mask_kernel(z, mask, z_t, mask_t):
    row = tl.program_id(0) * 4 + tl.arange(0, 4)
    i = row // 16
    j = row % 16
    transposed_row = j * 16 + i
    channels = tl.arange(0, 128)
    values = tl.load(z + row[:, None] * 128 + channels[None, :])
    tl.store(
        z_t + transposed_row[:, None] * 128 + channels[None, :], values,
    )
    value_mask = tl.load(mask + row)
    tl.store(mask_t + transposed_row, value_mask)


@triton.jit
def _small_transition_residual_kernel(
    x, norm_weight, norm_bias,
    weight_a, weight_b, weight_out,
    mask, out,
    M: tl.constexpr, N: tl.constexpr, H: tl.constexpr,
    EPS: tl.constexpr, HAS_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_h = tl.arange(0, BLOCK_H)
    valid_rows = offs_m < M

    x_tile = tl.load(
        x + offs_m[:, None] * N + offs_k[None, :],
        mask=valid_rows[:, None] & (offs_k[None, :] < N),
        other=0.0,
    ).to(tl.float32)
    mean = tl.sum(x_tile, axis=1) / N
    centered = x_tile - mean[:, None]
    variance = tl.sum(centered * centered, axis=1) / N
    x_norm = centered * tl.rsqrt(variance[:, None] + EPS)
    scale = tl.load(norm_weight + offs_k, mask=offs_k < N, other=0.0)
    bias = tl.load(norm_bias + offs_k, mask=offs_k < N, other=0.0)
    x_norm = (x_norm * scale[None, :] + bias[None, :]).to(tl.bfloat16)

    result = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for h in range(0, H, BLOCK_H):
        h_idx = h + offs_h
        up_mask = (h_idx[:, None] < H) & (offs_k[None, :] < N)
        wa = tl.load(
            weight_a + h_idx[:, None] * N + offs_k[None, :],
            mask=up_mask,
            other=0.0,
        )
        wb = tl.load(
            weight_b + h_idx[:, None] * N + offs_k[None, :],
            mask=up_mask,
            other=0.0,
        )
        a = tl.dot(x_norm, tl.trans(wa)).to(tl.bfloat16).to(tl.float32)
        b = tl.dot(x_norm, tl.trans(wb)).to(tl.bfloat16).to(tl.float32)
        hidden = (
            (a * tl.sigmoid(a)).to(tl.bfloat16).to(tl.float32) * b
        ).to(tl.bfloat16)
        wo = tl.load(
            weight_out + offs_n[:, None] * H + h_idx[None, :],
            mask=(offs_n[:, None] < N) & (h_idx[None, :] < H),
            other=0.0,
        )
        result += tl.dot(hidden, tl.trans(wo))

    update = result.to(tl.bfloat16).to(tl.float32)
    if HAS_MASK:
        row_mask = tl.load(mask + offs_m, mask=valid_rows, other=0.0)
        update *= row_mask[:, None]
    residual = tl.load(
        x + offs_m[:, None] * N + offs_n[None, :],
        mask=valid_rows[:, None] & (offs_n[None, :] < N),
        other=0.0,
    )
    tl.store(
        out + offs_m[:, None] * N + offs_n[None, :],
        residual + update,
        mask=valid_rows[:, None] & (offs_n[None, :] < N),
    )


@triton.jit
def _attention_residual_kernel(
    projections, mask, output_weight, residual, out,
    TRANSPOSE: tl.constexpr,
    INF: tl.constexpr,
    N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, C: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    outer = tl.program_id(0)
    out_cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    q_idx = tl.arange(0, N)
    k_idx = tl.arange(0, N)
    d_idx = tl.arange(0, D)
    slab = N * N * C
    row_base = outer * N
    output = tl.zeros((N, BLOCK_N), tl.float32)

    mask_bias = (
        (tl.load(mask + outer * N + k_idx).to(tl.float32) - 1.0) * INF
    ).to(tl.bfloat16)
    for head in range(0, H):
        q = tl.load(
            projections
            + (row_base + q_idx[:, None]) * C
            + head * D + d_idx[None, :]
        )
        k = tl.load(
            projections + slab
            + (row_base + k_idx[:, None]) * C
            + head * D + d_idx[None, :]
        )
        v = tl.load(
            projections + 2 * slab
            + (row_base + k_idx[:, None]) * C
            + head * D + d_idx[None, :]
        )
        scores = tl.dot(q, tl.trans(k)).to(tl.bfloat16)
        scores = (scores + mask_bias[None, :]).to(tl.bfloat16)
        bias_offsets = (
            4 * slab
            + (q_idx[:, None] * N + k_idx[None, :]) * C
            + head
        )
        scores = (scores + tl.load(projections + bias_offsets)).to(tl.bfloat16)
        scores_f32 = scores.to(tl.float32)
        scores_f32 -= tl.max(scores_f32, axis=1)[:, None]
        numerator = tl.exp(scores_f32)
        probabilities = (
            numerator / tl.sum(numerator, axis=1)[:, None]
        ).to(tl.bfloat16)
        attended = tl.dot(probabilities, v).to(tl.bfloat16)
        gate = tl.load(
            projections + 3 * slab
            + (row_base + q_idx[:, None]) * C
            + head * D + d_idx[None, :]
        )
        gate = tl.sigmoid(gate.to(tl.float32)).to(tl.bfloat16)
        attended = (attended * gate).to(tl.bfloat16)
        weight = tl.load(
            output_weight
            + out_cols[None, :] * C
            + head * D + d_idx[:, None],
            mask=out_cols[None, :] < C,
            other=0.0,
        )
        output += tl.dot(attended, weight)

    logical_rows = row_base + q_idx
    if TRANSPOSE:
        dst_rows = q_idx * N + outer
    else:
        dst_rows = logical_rows
    offsets = dst_rows[:, None] * C + out_cols[None, :]
    update = output.to(tl.bfloat16)
    old = tl.load(
        residual + offsets,
        mask=out_cols[None, :] < C,
        other=0.0,
    )
    tl.store(
        out + offsets,
        old + update,
        mask=out_cols[None, :] < C,
    )


@triton.jit
def _outer_product_residual_kernel(
    projected, mask, weight, bias, residual, out,
    EPS: tl.constexpr,
    C_HIDDEN: tl.constexpr, C_Z: tl.constexpr, BLOCK_Z: tl.constexpr,
):
    res_i = tl.program_id(0)
    z_block = tl.program_id(1)
    res_j = tl.arange(0, 16)
    seq = tl.arange(0, 8)
    z = z_block * BLOCK_Z + tl.arange(0, BLOCK_Z)
    hidden_e = tl.arange(0, C_HIDDEN)

    acc = tl.zeros((16, BLOCK_Z), tl.float32)
    for hidden_c in range(C_HIDDEN):
        av = tl.load(
            projected
            + (res_i * 8 + seq) * (2 * C_HIDDEN)
            + hidden_c
        )
        b_offsets = (
            (res_j[:, None, None] * 8 + seq[None, :, None])
            * (2 * C_HIDDEN)
            + C_HIDDEN + hidden_e[None, None, :]
        )
        bv = tl.load(projected + b_offsets)
        outer = tl.sum(av[None, :, None] * bv, axis=1).to(tl.bfloat16)
        inner = hidden_c * C_HIDDEN + hidden_e
        weights = tl.load(
            weight
            + z[None, :] * (C_HIDDEN * C_HIDDEN)
            + inner[:, None]
        )
        acc += tl.dot(outer, weights)

    mask_i = tl.load(mask + seq * 16 + res_i)
    mask_j = tl.load(mask + seq[None, :] * 16 + res_j[:, None])
    norm = (
        tl.sum(mask_i[None, :] * mask_j, axis=1) + EPS
    ).to(tl.bfloat16)
    projected_bias = tl.load(bias + z)
    update = ((acc + projected_bias[None, :]) / norm[:, None]).to(
        tl.bfloat16
    )
    offsets = (res_i * 16 + res_j[:, None]) * C_Z + z[None, :]
    old = tl.load(residual + offsets)
    tl.store(out + offsets, old + update)


class MSAModuleBlock(nn.Module):
    """Single block of AF3 Algorithm 8.

    Args:
        c_m: MSA channel dimension
        c_z: Pair channel dimension
        c_hidden_msa_att: Hidden dim in MSA attention
        c_hidden_opm: Hidden dim in outer product mean
        c_hidden_mul: Hidden dim in triangle multiplication
        c_hidden_pair_att: Hidden dim in triangle attention
        no_heads_msa: Heads for MSA attention
        no_heads_pair: Heads for triangle attention
        transition_n: Transition layer scale
        msa_dropout: MSA dropout rate
        pair_dropout: Pair dropout rate
        opm_first: Whether OPM comes before MSA attention
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        transition_n: int,
        msa_dropout: float = 0.0,
        pair_dropout: float = 0.0,
        opm_first: bool = True,
        fuse_projection_weights: bool = False,
        inf: float = 1e9,
        eps: float = 1e-3,
        last_block: bool = False,
    ):
        super().__init__()
        self.opm_first = opm_first
        self.skip_msa_update = last_block and opm_first

        if not self.skip_msa_update:
            self.msa_att_row = MSARowAttentionWithPairBias(
                c_m=c_m, c_z=c_z,
                c_hidden=c_hidden_msa_att,
                no_heads=no_heads_msa,
                inf=inf,
            )

            self.msa_transition = SwiGLUTransition(c_in=c_m, n=transition_n)

        self.outer_product_mean = OuterProductMean(
            c_m=c_m, c_z=c_z, c_hidden=c_hidden_opm, eps=eps,
        )

        self.pair_stack = PairBlock(
            c_z=c_z,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_pair=no_heads_pair,
            transition_n=transition_n,
            pair_dropout=pair_dropout,
            inf=inf,
        )

    @staticmethod
    def _ending_attention(
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        module: nn.Module,
    ) -> torch.Tensor:
        z_t = torch.empty_like(z)
        mask_t = torch.empty_like(pair_mask)
        _transpose_pair_and_mask_kernel[(64,)](
            z, pair_mask, z_t, mask_t, num_warps=4,
        )
        projections = torch.empty(
            (5, 256, 128), device=z.device, dtype=z.dtype,
        )
        _norm_project_kernel[(16, 5, 4)](
            z_t,
            module.layer_norm.weight,
            module.layer_norm.bias,
            module.mha.linear_q.weight,
            module.mha.linear_k.weight,
            module.mha.linear_v.weight,
            module.mha.linear_g.weight,
            module.linear_z.weight,
            projections,
            M=256,
            C=128,
            BM=16,
            BN=32,
            num_warps=4,
        )
        out = torch.empty_like(z)
        _attention_residual_kernel[(16, 4)](
            projections,
            mask_t,
            module.mha.linear_o.weight,
            z,
            out,
            TRANSPOSE=True,
            INF=module.inf,
            N=16,
            H=4,
            D=32,
            C=128,
            BLOCK_N=32,
            num_warps=4,
        )
        return out

    @staticmethod
    def _starting_attention(
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        module: nn.Module,
    ) -> torch.Tensor:
        projections = torch.empty(
            (5, 256, 128), device=z.device, dtype=z.dtype,
        )
        _norm_project_kernel[(16, 5, 4)](
            z,
            module.layer_norm.weight,
            module.layer_norm.bias,
            module.mha.linear_q.weight,
            module.mha.linear_k.weight,
            module.mha.linear_v.weight,
            module.mha.linear_g.weight,
            module.linear_z.weight,
            projections,
            M=256,
            C=128,
            BM=16,
            BN=32,
            num_warps=4,
        )
        out = torch.empty_like(z)
        _attention_residual_kernel[(16, 4)](
            projections,
            pair_mask,
            module.mha.linear_o.weight,
            z,
            out,
            TRANSPOSE=False,
            INF=module.inf,
            N=16,
            H=4,
            D=32,
            C=128,
            BLOCK_N=32,
            num_warps=4,
        )
        return out

    @staticmethod
    def _transition_update(
        x: torch.Tensor,
        module: nn.Module,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n = x.shape[-1]
        m = x.numel() // n
        h = module.swiglu.linear_a.weight.shape[0]
        out = torch.empty_like(x)
        block_m = 16
        block_n = n
        block_h = n
        mask_ptr = mask if mask is not None else x
        _small_transition_residual_kernel[(triton.cdiv(m, block_m), 1)](
            x,
            module.layer_norm.weight,
            module.layer_norm.bias,
            module.swiglu.linear_a.weight,
            module.swiglu.linear_b.weight,
            module.linear_out.weight,
            mask_ptr,
            out,
            M=m,
            N=n,
            H=h,
            EPS=module.layer_norm.eps,
            HAS_MASK=mask is not None,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_H=block_h,
            BLOCK_K=n,
            num_warps=8,
            num_stages=3,
        )
        return out

    @staticmethod
    def _outer_product_update(
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        module: nn.Module,
    ) -> torch.Tensor:
        projected = torch.empty((16, 8, 64), device=m.device, dtype=m.dtype)
        _project_kernel[(8,)](
            m,
            msa_mask,
            module.layer_norm.weight,
            module.layer_norm.bias,
            module.linear_1.weight,
            module.linear_2.weight,
            projected,
            N_RES=16,
            C_M=64,
            C_HIDDEN=32,
            LN_EPS=module.layer_norm.eps,
            BLOCK_ROWS=16,
            num_warps=8,
        )
        out = torch.empty_like(z)
        _outer_product_residual_kernel[(16, 4)](
            projected,
            msa_mask,
            module.linear_out.weight,
            module.linear_out.bias,
            z,
            out,
            EPS=module.eps,
            C_HIDDEN=32,
            C_Z=128,
            BLOCK_Z=32,
            num_warps=2,
            num_stages=5,
        )
        return out

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m:        [*, N_seq, N_res, C_m] MSA embedding
            z:        [*, N_res, N_res, C_z] pair embedding
            msa_mask: [*, N_seq, N_res] MSA mask
            pair_mask:[*, N_res, N_res] pair mask

        Returns:
            (m, z): updated MSA and pair embeddings
        """
        fast_shape = (
            m.is_cuda
            and m.dtype == torch.bfloat16
            and m.shape == (1, 8, 16, 64)
            and z.shape == (1, 16, 16, 128)
            and msa_mask.shape == (1, 8, 16)
            and pair_mask.shape == (1, 16, 16)
        )
        if self.opm_first:
            if fast_shape:
                z = self._outer_product_update(
                    m, z, msa_mask, self.outer_product_mean,
                )
            else:
                z = z + self.outer_product_mean(m, mask=msa_mask)

        if not self.skip_msa_update:
            m = m + self.msa_att_row(m, z=z, mask=pair_mask)
            if (
                m.is_cuda
                and m.dtype == torch.bfloat16
                and m.shape == (1, 8, 16, 64)
            ):
                m = self._transition_update(m, self.msa_transition)
            else:
                m = m + self.msa_transition(m)

        if not self.opm_first:
            if fast_shape:
                z = self._outer_product_update(
                    m, z, msa_mask, self.outer_product_mean,
                )
            else:
                z = z + self.outer_product_mean(m, mask=msa_mask)

        if fast_shape:
            # PairBlock's fused attention approximation compounds across four
            # blocks. Keep its fast triangle contractions, then use the more
            # accurate component attention and transition kernels.
            pair_stack = self.pair_stack
            z = pair_stack._triangle_update(
                z, pair_mask, pair_stack.tri_mul_out, False,
            )
            z = pair_stack._triangle_update(
                z, pair_mask, pair_stack.tri_mul_in, True,
            )
            z = self._starting_attention(
                z, pair_mask, pair_stack.tri_att_start,
            )
            z = self._ending_attention(
                z, pair_mask, pair_stack.tri_att_end,
            )
            z = self._transition_update(
                z, pair_stack.pair_transition, pair_mask,
            )
        else:
            z = self.pair_stack(z=z, pair_mask=pair_mask)

        return m, z


class MSAModuleStack(nn.Module):
    """AF3 Algorithm 8: MSA module stack.

    Args:
        c_m: MSA channel dimension
        c_z: Pair channel dimension
        c_hidden_msa_att: Hidden dim in MSA attention
        c_hidden_opm: Hidden dim in outer product mean
        c_hidden_mul: Hidden dim in triangle multiplication
        c_hidden_pair_att: Hidden dim in triangle attention
        no_heads_msa: Heads for MSA attention
        no_heads_pair: Heads for triangle attention
        no_blocks: Number of MSA module blocks
        transition_n: Transition scale
        opm_first: Whether OPM comes before MSA attention
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        no_blocks: int,
        transition_n: int,
        msa_dropout: float = 0.0,
        pair_dropout: float = 0.0,
        opm_first: bool = True,
        fuse_projection_weights: bool = False,
        blocks_per_ckpt: int | None = None,
        inf: float = 1e9,
        eps: float = 1e-3,
        **kwargs,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            MSAModuleBlock(
                c_m=c_m, c_z=c_z,
                c_hidden_msa_att=c_hidden_msa_att,
                c_hidden_opm=c_hidden_opm,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att,
                no_heads_msa=no_heads_msa,
                no_heads_pair=no_heads_pair,
                transition_n=transition_n,
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                opm_first=opm_first,
                inf=inf,
                eps=eps,
                last_block=(i == no_blocks - 1),
            )
            for i in range(no_blocks)
        ])
        self._cuda_graph = None
        self._graph_inputs = None
        self._graph_outputs = None

    def _forward_eager(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            m, z = block(m=m, z=z, msa_mask=msa_mask, pair_mask=pair_mask)
        return m, z

    def _capture_cuda_graph(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> None:
        static_inputs = tuple(
            x.clone() for x in (m, z, msa_mask, pair_mask)
        )

        # Finish compilation and release warmup temporaries before the graph
        # allocator assigns its fixed storage.
        self._forward_eager(*static_inputs)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_outputs = self._forward_eager(*static_inputs)

        self._cuda_graph = graph
        self._graph_inputs = static_inputs
        self._graph_outputs = static_outputs

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m:        [*, N_seq, N_res, C_m] MSA embedding
            z:        [*, N_res, N_res, C_z] pair embedding
            msa_mask: [*, N_seq, N_res] MSA mask
            pair_mask:[*, N_res, N_res] pair mask

        Returns:
            (m, z): updated MSA and pair embeddings
        """
        graph_shape = (
            m.is_cuda
            and m.dtype == torch.bfloat16
            and m.shape == (1, 8, 16, 64)
            and z.shape == (1, 16, 16, 128)
            and msa_mask.shape == (1, 8, 16)
            and pair_mask.shape == (1, 16, 16)
        )
        if not graph_shape:
            return self._forward_eager(m, z, msa_mask, pair_mask)

        if self._cuda_graph is None:
            self._capture_cuda_graph(m, z, msa_mask, pair_mask)
        else:
            static_m, static_z, static_msa_mask, static_pair_mask = (
                self._graph_inputs
            )
            _copy_graph_inputs_kernel[(128,)](
                m, z, msa_mask, pair_mask,
                static_m, static_z, static_msa_mask, static_pair_mask,
                num_warps=4,
            )
        self._cuda_graph.replay()
        return self._graph_outputs
