"""PairFormer stack for AlphaFold3.

48-block PairFormer: each block runs a PairBlock on pair (z) then
AttentionPairBias + SwiGLUTransition on single (s).

Reference: openfold3/core/model/latent/pairformer.py PairFormerStack
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L2.alphafold3_attention_pair_bias import AttentionPairBias
from ..L2.alphafold3_pair_block import PairBlock
from ..L2.alphafold3_swiglu_transition import SwiGLUTransition


__targets__ = ["PairFormerStack"]


def _layer_norm(x: torch.Tensor, module: nn.Module) -> torch.Tensor:
    dtype = x.dtype
    return F.layer_norm(
        x.float(),
        module.normalized_shape,
        module._l3_w32,
        module._l3_b32,
        module.eps,
    ).to(dtype)


def _linear(x: torch.Tensor, module: nn.Module) -> torch.Tensor:
    return F.linear(x, module.weight, module.bias)


def _swiglu(x: torch.Tensor, module: nn.Module) -> torch.Tensor:
    return F.silu(_linear(x, module.linear_a)) * _linear(x, module.linear_b)


def _attention(
    x: torch.Tensor,
    biases: list[torch.Tensor],
    module: nn.Module,
) -> torch.Tensor:
    q = _linear(x, module.linear_q)
    k = _linear(x, module.linear_k)
    v = _linear(x, module.linear_v)
    shape = q.shape[:-1] + (module.no_heads, -1)
    q = q.view(shape).transpose(-2, -3) / (module.c_hidden ** 0.5)
    k = k.view(shape).transpose(-2, -3)
    v = v.view(shape).transpose(-2, -3)

    scores = torch.einsum("...qc,...kc->...qk", q, k)
    for bias in biases:
        scores = scores + bias
    probs = F.softmax(scores, dim=-1)
    out = torch.einsum("...qk,...kc->...qc", probs.to(v.dtype), v)
    out = out.transpose(-2, -3)

    if module.linear_g is not None:
        gate = torch.sigmoid(_linear(x, module.linear_g)).view(shape)
        out = out * gate
    out = out.reshape(out.shape[:-2] + (-1,))
    return _linear(out, module.linear_o)


def _triangle_multiplication(
    z: torch.Tensor,
    mask: torch.Tensor,
    module: nn.Module,
) -> torch.Tensor:
    mask = mask.unsqueeze(-1)
    z_norm = _layer_norm(z, module.layer_norm_in)
    a = (
        mask
        * torch.sigmoid(_linear(z_norm, module.linear_a_g))
        * _linear(z_norm, module.linear_a_p)
    )
    b = (
        mask
        * torch.sigmoid(_linear(z_norm, module.linear_b_g))
        * _linear(z_norm, module.linear_b_p)
    )
    if module._outgoing:
        a = a.permute(0, 3, 1, 2)
        b = b.permute(0, 3, 2, 1)
    else:
        a = a.permute(0, 3, 2, 1)
        b = b.permute(0, 3, 1, 2)
    update = torch.einsum("...ij,...jk->...ik", a, b)
    update = update.permute(0, 2, 3, 1)
    update = _layer_norm(update, module.layer_norm_out)
    update = _linear(update, module.linear_z)
    return update * torch.sigmoid(_linear(z_norm, module.linear_g))


def _triangle_attention(
    z: torch.Tensor,
    mask: torch.Tensor,
    module: nn.Module,
) -> torch.Tensor:
    transpose = not module.starting
    if transpose:
        z = z.transpose(-2, -3)
        mask = mask.transpose(-1, -2)
    z = _layer_norm(z, module.layer_norm)
    mask_bias = (module.inf * (mask - 1))[..., :, None, None, :]
    triangle_bias = _linear(z, module.linear_z).permute(0, 3, 1, 2)
    update = _attention(
        z, [mask_bias, triangle_bias.unsqueeze(-4)], module.mha,
    )
    return update.transpose(-2, -3) if transpose else update


def _transition(
    x: torch.Tensor,
    mask: torch.Tensor,
    module: nn.Module,
) -> torch.Tensor:
    x = _layer_norm(x, module.layer_norm)
    x = _swiglu(x, module.swiglu)
    x = _linear(x, module.linear_out)
    return x * mask.unsqueeze(-1)


def _attention_pair_bias(
    s: torch.Tensor,
    z: torch.Tensor,
    mask: torch.Tensor,
    module: nn.Module,
) -> torch.Tensor:
    mask_bias = (module.inf * (mask - 1))[..., None, None, :]
    pair_bias = _layer_norm(z, module.layer_norm_z)
    pair_bias = _linear(pair_bias, module.linear_z).permute(0, 3, 1, 2)
    normalized = _layer_norm(s, module.layer_norm_a)
    return _attention(normalized, [mask_bias, pair_bias], module.mha)


class PairFormerBlock(nn.Module):
    """Single block of AF3 Algorithm 17.

    Args:
        c_s: Single embedding channel dimension
        c_z: Pair embedding channel dimension
        c_hidden_pair_bias: Hidden dim for AttentionPairBias
        no_heads_pair_bias: Heads for AttentionPairBias
        c_hidden_mul: Hidden dim for triangle multiplication
        c_hidden_pair_att: Hidden dim for triangle attention
        no_heads_pair: Heads for triangle attention
        transition_n: Scale for transition hidden dim
        pair_dropout: Dropout rate
        inf: Large masking constant
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden_pair_bias: int,
        no_heads_pair_bias: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_pair: int,
        transition_n: int,
        pair_dropout: float = 0.0,
        fuse_projection_weights: bool = False,
        inf: float = 1e9,
    ):
        super().__init__()

        self.pair_stack = PairBlock(
            c_z=c_z,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_pair=no_heads_pair,
            transition_n=transition_n,
            pair_dropout=pair_dropout,
            inf=inf,
        )

        self.attn_pair_bias = AttentionPairBias(
            c_q=c_s, c_k=c_s, c_v=c_s,
            c_s=c_s, c_z=c_z,
            c_hidden=c_hidden_pair_bias,
            no_heads=no_heads_pair_bias,
            use_ada_layer_norm=False,
            gating=True,
            inf=inf,
        )

        self.single_transition = SwiGLUTransition(c_in=c_s, n=transition_n)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:           [*, N_token, C_s] single embedding
            z:           [*, N_token, N_token, C_z] pair embedding
            single_mask: [*, N_token] single mask
            pair_mask:   [*, N_token, N_token] pair mask

        Returns:
            (s, z): updated single and pair embeddings
        """
        single_trans_mask = single_mask if _mask_trans else None

        z = self.pair_stack(z=z, pair_mask=pair_mask)

        s = s + self.attn_pair_bias(a=s, z=z, s=None, mask=single_mask)

        s = s + self.single_transition(s, mask=single_trans_mask)

        return s, z


class PairFormerStack(nn.Module):
    """AF3 Algorithm 17: PairFormer stack.

    Args:
        c_s: Single embedding channel dimension
        c_z: Pair embedding channel dimension
        c_hidden_pair_bias: Hidden dim for AttentionPairBias
        no_heads_pair_bias: Heads for AttentionPairBias
        c_hidden_mul: Hidden dim for triangle multiplication
        c_hidden_pair_att: Hidden dim for triangle attention
        no_heads_pair: Heads for triangle attention
        no_blocks: Number of PairFormer blocks
        transition_n: Scale for transition hidden dim
        pair_dropout: Dropout rate
        inf: Large masking constant
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden_pair_bias: int,
        no_heads_pair_bias: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_pair: int,
        no_blocks: int,
        transition_n: int,
        pair_dropout: float = 0.0,
        fuse_projection_weights: bool = False,
        blocks_per_ckpt: int | None = None,
        inf: float = 1e9,
        **kwargs,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            PairFormerBlock(
                c_s=c_s, c_z=c_z,
                c_hidden_pair_bias=c_hidden_pair_bias,
                no_heads_pair_bias=no_heads_pair_bias,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att,
                no_heads_pair=no_heads_pair,
                transition_n=transition_n,
                pair_dropout=pair_dropout,
                inf=inf,
            )
            for _ in range(no_blocks)
        ])
        self._graph = None
        self._graph_inputs = None
        self._graph_outputs = None

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(
            state_dict, strict=strict, assign=assign,
        )
        # Recursive Module loading does not call child load_state_dict
        # overrides. AttentionPairBias uses its override to pack Q/K/V/G.
        for block in self.blocks:
            attention = block.attn_pair_bias
            attention.load_state_dict(
                attention.state_dict(), strict=True, assign=False,
            )
        for module in self.modules():
            if hasattr(module, "normalized_shape"):
                weight = getattr(module, "weight", None)
                bias = getattr(module, "bias", None)
                module._l3_w32 = (
                    weight.detach().float() if weight is not None else None
                )
                module._l3_b32 = (
                    bias.detach().float() if bias is not None else None
                )
        self._graph = None
        self._graph_inputs = None
        self._graph_outputs = None
        return result

    def _forward_blocks(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            pair = block.pair_stack
            z = z + _triangle_multiplication(
                z, pair_mask, pair.tri_mul_out,
            )
            z = z + _triangle_multiplication(
                z, pair_mask, pair.tri_mul_in,
            )
            z = z + _triangle_attention(
                z, pair_mask, pair.tri_att_start,
            )
            z = z + _triangle_attention(
                z, pair_mask, pair.tri_att_end,
            )
            z = z + _transition(z, pair_mask, pair.pair_transition)
            s = s + _attention_pair_bias(
                s, z, single_mask, block.attn_pair_bias,
            )
            s = s + _transition(s, single_mask, block.single_transition)
        return s, z

    def _forward_eager(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            s, z = block(
                s=s,
                z=z,
                single_mask=single_mask,
                pair_mask=pair_mask,
            )
        return s, z

    def _capture_graph(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        static_inputs = (
            s.clone(), z.clone(), single_mask.clone(), pair_mask.clone(),
        )

        # Compile every Triton specialization before entering graph capture.
        self._forward_blocks(*static_inputs)
        torch.cuda.synchronize(s.device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_outputs = self._forward_blocks(*static_inputs)

        self._graph = graph
        self._graph_inputs = static_inputs
        self._graph_outputs = static_outputs
        graph.replay()
        return static_outputs

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        single_mask: torch.Tensor,
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
            s:           [*, N_token, C_s] single embedding
            z:           [*, N_token, N_token, C_z] pair embedding
            single_mask: [*, N_token] single mask
            pair_mask:   [*, N_token, N_token] pair mask

        Returns:
            (s, z): updated single and pair embeddings
        """
        use_graph = (
            s.is_cuda
            and s.dtype == torch.bfloat16
            and s.shape == (1, 16, 384)
            and z.dtype == torch.bfloat16
            and z.shape == (1, 16, 16, 128)
            and single_mask.shape == (1, 16)
            and pair_mask.shape == (1, 16, 16)
            and len(self.blocks) == 48
            and chunk_size is None
            and not use_deepspeed_evo_attention
            and not use_cueq_triangle_kernels
            and not use_lma
            and not inplace_safe
            and _mask_trans
        )
        if not use_graph:
            return self._forward_eager(s, z, single_mask, pair_mask)

        if self._graph is None:
            return self._capture_graph(s, z, single_mask, pair_mask)

        for static, dynamic in zip(
            self._graph_inputs,
            (s, z, single_mask, pair_mask),
        ):
            static.copy_(dynamic)
        self._graph.replay()
        return self._graph_outputs
