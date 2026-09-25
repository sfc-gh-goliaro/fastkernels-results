"""Diffusion transformer for AlphaFold3.

24-block transformer used inside the diffusion module. Each block:
AttentionPairBias + ConditionedTransitionBlock (AdaLN-Zero).

Reference: openfold3/core/model/layers/diffusion_transformer.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._inductor.config as inductor_config

from ..L2.alphafold3_attention_pair_bias import AttentionPairBias, CrossAttentionPairBias
from ..L2.alphafold3_swiglu_transition import ConditionedTransitionBlock


__targets__ = ["DiffusionTransformer"]

# Keep compiled matrix products on generated Triton kernels.
inductor_config.max_autotune_gemm_backends = "TRITON"


class DiffusionTransformerBlock(nn.Module):
    """AF3 Algorithm 23: Diffusion transformer block.

    Args:
        c_a: Token activation channel dimension
        c_s: Single activation channel dimension
        c_z: Pair activation channel dimension
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        n_transition: Transition layer scale
        use_ada_layer_norm: Whether to use AdaLN-Zero
        inf: Large masking constant
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
            self.attention_pair_bias = AttentionPairBias(
                c_q=c_a, c_k=c_a, c_v=c_a,
                c_s=c_s, c_z=c_z,
                c_hidden=c_hidden,
                no_heads=no_heads,
                use_ada_layer_norm=use_ada_layer_norm,
                gating=True,
                inf=inf,
            )
        else:
            self.attention_pair_bias = CrossAttentionPairBias(
                c_q=c_a, c_k=c_a, c_v=c_a,
                c_s=c_s, c_z=c_z,
                c_hidden=c_hidden,
                no_heads=no_heads,
                use_ada_layer_norm=use_ada_layer_norm,
                n_query=n_query,
                n_key=n_key,
                gating=True,
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
        Args:
            a:    [*, N, C_token] token-level embedding
            s:    [*, N, C_s] single embedding
            z:    [*, N, N, C_z] pair embedding
            mask: [*, N] mask

        Returns:
            [*, N, C_token] updated token embedding
        """
        a = a + self.attention_pair_bias(a=a, z=z, s=s, mask=mask)

        trans_mask = mask if _mask_trans else None
        a = a + self.conditioned_transition(a=a, s=s, mask=trans_mask)

        return a


class DiffusionTransformer(nn.Module):
    """AF3 Algorithm 23: Diffusion transformer stack.

    Args:
        c_a: Token activation channel dimension
        c_s: Single activation channel dimension
        c_z: Pair activation channel dimension
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        no_blocks: Number of transformer blocks
        n_transition: Transition layer scale
        use_ada_layer_norm: Whether to use AdaLN-Zero
        inf: Large masking constant
    """

    def __init__(
        self,
        c_a: int,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_blocks: int,
        n_transition: int,
        use_ada_layer_norm: bool = True,
        n_query: int | None = None,
        n_key: int | None = None,
        inf: float = 1e9,
        blocks_per_ckpt: int | None = None,
        **kwargs,
    ):
        super().__init__()
        from ..L1.layer_norm import LayerNorm

        self.use_cross_attention = n_query is not None
        if self.use_cross_attention:
            self.layer_norm_z = LayerNorm(c_z, create_offset=False)

        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(
                c_a=c_a, c_s=c_s, c_z=c_z,
                c_hidden=c_hidden, no_heads=no_heads,
                n_transition=n_transition,
                use_ada_layer_norm=use_ada_layer_norm,
                n_query=n_query,
                n_key=n_key,
                inf=inf,
            )
            for _ in range(no_blocks)
        ])

        # The block count and channel sizes are fixed for each captured module.
        # Compiling across the stack lets Inductor keep the many AdaLN, gating,
        # masking, and residual pointwise operations out of eager dispatch.
        if self.use_cross_attention:
            self._compiled_forward = torch.compile(
                self._forward_impl,
                fullgraph=True,
                dynamic=False,
                mode="max-autotune",
            )
        else:
            self._compiled_block_count = min(2, no_blocks)
            self._compiled_forward = torch.compile(
                self._self_prefix,
                fullgraph=True,
                dynamic=False,
                mode="max-autotune",
            )

    @staticmethod
    def _layer_norm(x: torch.Tensor, module: nn.Module) -> torch.Tensor:
        dtype = x.dtype
        weight = module.weight
        bias = module.bias
        if weight is not None:
            weight = weight.float()
        if bias is not None:
            bias = bias.float()
        return F.layer_norm(
            x.float(), module.normalized_shape, weight, bias, module.eps,
        ).to(dtype)

    @classmethod
    def _adaln(
        cls, a: torch.Tensor, s: torch.Tensor, module: nn.Module,
    ) -> torch.Tensor:
        s_norm = cls._layer_norm(s, module.layer_norm_s)
        gate = torch.sigmoid(F.linear(
            s_norm, module.linear_g.weight, module.linear_g.bias,
        ))
        a_norm = cls._layer_norm(a, module.layer_norm_a)
        shift = F.linear(s_norm, module.linear_s.weight)
        return gate * (a_norm + shift)

    @staticmethod
    def _attention(
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: list[torch.Tensor],
        module: nn.Module,
    ) -> torch.Tensor:
        heads = module.no_heads
        hidden = module.c_hidden
        q = F.linear(q_x, module.linear_q.weight, module.linear_q.bias)
        k = F.linear(kv_x, module.linear_k.weight)
        v = F.linear(kv_x, module.linear_v.weight)
        q = q.view(*q.shape[:-1], heads, hidden).transpose(-2, -3)
        k = k.view(*k.shape[:-1], heads, hidden).transpose(-2, -3)
        v = v.view(*v.shape[:-1], heads, hidden).transpose(-2, -3)
        q = q / math.sqrt(hidden)

        scores = torch.einsum("...qc,...kc->...qk", q, k)
        for bias in biases:
            scores = scores + bias
        probs = F.softmax(scores, dim=-1)
        out = torch.einsum("...qk,...kc->...qc", probs.to(v.dtype), v)
        out = out.transpose(-2, -3)

        gate = torch.sigmoid(F.linear(q_x, module.linear_g.weight))
        gate = gate.view(*gate.shape[:-1], heads, hidden)
        out = (out * gate).reshape(*out.shape[:-2], heads * hidden)
        return F.linear(out, module.linear_o.weight)

    @classmethod
    def _transition(
        cls,
        a: torch.Tensor,
        s: torch.Tensor,
        mask: torch.Tensor,
        module: nn.Module,
    ) -> torch.Tensor:
        x = cls._adaln(a, s, module.layer_norm)
        left = F.linear(x, module.swiglu.linear_a.weight)
        right = F.linear(x, module.swiglu.linear_b.weight)
        hidden = F.silu(left) * right
        out = F.linear(hidden, module.linear_out.weight)
        gate = torch.sigmoid(F.linear(
            s, module.linear_g.weight, module.linear_g.bias,
        ))
        return gate * out * mask.unsqueeze(-1)

    @staticmethod
    def _block_indices(mask: torch.Tensor, n_query: int, n_key: int):
        batch_dims = mask.shape[:-1]
        n_atom = mask.shape[-1]
        num_blocks = math.ceil(n_atom / n_query)
        centers = (
            n_query // 2
            + torch.arange(num_blocks, device=mask.device) * n_query
        )
        centers = centers.reshape(*(1,) * len(batch_dims), num_blocks)
        centers = centers.expand(*batch_dims, num_blocks)
        n_real = mask.sum(dim=-1, keepdim=True).expand(
            *batch_dims, num_blocks,
        )
        initial = (
            centers.unsqueeze(-1)
            + torch.arange(-n_key // 2, n_key // 2, device=mask.device)
        ).int()
        underflow = F.relu(-initial[..., 0])
        overflow = F.relu(initial[..., -1] - (n_real - 1))
        shift = torch.where(underflow > 0, underflow, -overflow)
        final = initial + shift.unsqueeze(-1)
        n_real = n_real.unsqueeze(-1)
        invalid = (final < 0) | (final >= n_real)
        indices = torch.clamp(
            final, torch.zeros_like(n_real), (n_real - 1).clamp(min=0),
        )
        return indices.long(), invalid

    @classmethod
    def _to_blocks(
        cls,
        x: torch.Tensor,
        mask: torch.Tensor,
        n_query: int,
        n_key: int,
    ):
        batch_dims = x.shape[:-2]
        n_atom, channels = x.shape[-2:]
        num_blocks = math.ceil(n_atom / n_query)
        pad = (-n_atom) % n_query
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
            mask = F.pad(mask, (0, pad))
        query = x.reshape(*batch_dims, num_blocks, n_query, channels)
        mask = mask.expand(*batch_dims, -1)
        indices, invalid = cls._block_indices(mask, n_query, n_key)
        flat_batch = math.prod(batch_dims) if batch_dims else 1
        flat_indices = indices.reshape(flat_batch, num_blocks * n_key)
        key = torch.gather(
            x.reshape(flat_batch, n_atom + pad, channels),
            1,
            flat_indices.unsqueeze(-1).expand(-1, -1, channels),
        )
        key = key.masked_fill(
            invalid.reshape(flat_batch, -1, 1), 0.0,
        ).reshape(*batch_dims, num_blocks, n_key, channels)
        key_mask = torch.gather(
            mask.reshape(flat_batch, -1), 1, flat_indices,
        ).reshape(*batch_dims, num_blocks, n_key)
        key_mask = key_mask * (~invalid).to(mask.dtype)
        block_mask = (
            mask.reshape(*batch_dims, num_blocks, n_query).unsqueeze(-1)
            * key_mask.unsqueeze(-2)
        )
        return query, key, block_mask

    @classmethod
    def _self_block(
        cls,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        block: nn.Module,
    ) -> torch.Tensor:
        attention = block.attention_pair_bias
        z_norm = cls._layer_norm(z, attention.layer_norm_z)
        z_bias = F.linear(z_norm, attention.linear_z.weight)
        z_bias = z_bias.movedim(-1, -3)
        mask_bias = (attention.inf * (mask - 1))[..., None, None, :]
        norm_a = cls._adaln(a, s, attention.layer_norm_a)
        update = cls._attention(
            norm_a, norm_a, [mask_bias, z_bias], attention.mha,
        )
        update = (
            torch.sigmoid(F.linear(
                s,
                attention.linear_ada_out.weight,
                attention.linear_ada_out.bias,
            ))
            * update
        )
        a = a + update
        return a + cls._transition(
            a, s, mask, block.conditioned_transition,
        )

    @classmethod
    def _cross_block(
        cls,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        block: nn.Module,
    ) -> torch.Tensor:
        attention = block.attention_pair_bias
        a_q, a_k, block_mask = cls._to_blocks(
            a, mask, attention.n_query, attention.n_key,
        )
        s_q, s_k, _ = cls._to_blocks(
            s, mask, attention.n_query, attention.n_key,
        )
        z_bias = F.linear(z, attention.linear_z.weight).movedim(-1, -3)
        mask_bias = (attention.inf * (block_mask - 1))[..., None, :, :]
        q = cls._adaln(a_q, s_q, attention.layer_norm_a_q)
        k = cls._adaln(a_k, s_k, attention.layer_norm_a_k)
        update = cls._attention(q, k, [mask_bias, z_bias], attention.mha)
        update = update.reshape(*a.shape[:-2], -1, a.shape[-1])
        update = update[..., :a.shape[-2], :]
        update = (
            torch.sigmoid(F.linear(
                s,
                attention.linear_ada_out.weight,
                attention.linear_ada_out.bias,
            ))
            * update
        )
        a = a + update
        return a + cls._transition(
            a, s, mask, block.conditioned_transition,
        )

    def _forward_impl(
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
        if mask is None:
            mask = a.new_ones(a.shape[:-1])
        if self.use_cross_attention:
            z = self._layer_norm(z, self.layer_norm_z)

        for block in self.blocks:
            if self.use_cross_attention:
                a = self._cross_block(a, s, z, mask, block)
            else:
                a = self._self_block(a, s, z, mask, block)

        return a

    def _self_prefix(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        for index in range(self._compiled_block_count):
            a = self._self_block(a, s, z, mask, self.blocks[index])
        return a

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
        if not self.use_cross_attention:
            if mask is None:
                mask = a.new_ones(a.shape[:-1])
            a = self._compiled_forward(a, s, z, mask)
            for index in range(self._compiled_block_count, len(self.blocks)):
                a = self._self_block(a, s, z, mask, self.blocks[index])
            return a
        return self._compiled_forward(
            a,
            s,
            z,
            mask,
            use_deepspeed_evo_attention,
            use_cueq_triangle_kernels,
            use_lma,
            use_high_precision_attention,
            _mask_trans,
        )
