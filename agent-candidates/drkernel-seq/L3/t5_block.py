import torch
import torch.nn as nn
from transformers import T5Config

# Keep imports to frozen implementations
from ..L1.t5_layer_norm import T5LayerNorm  # noqa: F401
from ..L2.t5_attention import T5SelfAttention
from ..L2.t5_dense import T5DenseActDense, T5DenseGatedActDense

import triton
import triton.language as tl


@triton.jit
def elementwise_add_contig_kernel(
    A, B, C,
    NUMEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Assumes A, B, C are contiguous (stride==1 when viewed as 1D).
    Computes C = A + B over NUMEL elements.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < NUMEL
    a = tl.load(A + offs, mask=mask)
    b = tl.load(B + offs, mask=mask)
    c = a + b
    tl.store(C + offs, c, mask=mask)


def triton_elementwise_add_contig(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute c = a + b using a simple Triton kernel over contiguous memory.
    a, b must be same shape, same dtype, same device.
    """
    assert a.shape == b.shape, f"Shapes must match: {a.shape} vs {b.shape}"
    assert a.dtype == b.dtype, f"Dtypes must match: {a.dtype} vs {b.dtype}"
    assert a.device == b.device, f"Devices must match: {a.device} vs {b.device}"

    if not a.is_cuda:
        return a + b

    # Enforce contiguous to avoid illegal memory access
    a_c = a.contiguous()
    b_c = b.contiguous()
    c = torch.empty_like(a_c)

    numel = a_c.numel()
    # Heuristic block size
    block = 2048 if numel >= 1 << 20 else 1024
    grid = (triton.cdiv(numel, block),)

    elementwise_add_contig_kernel[grid](
        a_c, b_c, c,
        NUMEL=numel,
        BLOCK=block,
        num_warps=8 if block == 2048 else 4,
        num_stages=2,
    )
    return c.view_as(a)


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.SelfAttention = T5SelfAttention(config, has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed = self.layer_norm(hidden_states)
        attn_output, position_bias = self.SelfAttention(
            normed, mask=mask, position_bias=position_bias,
        )
        # Use Triton for residual add; enforce contiguous to prevent OOB
        hidden_states = triton_elementwise_add_contig(hidden_states, attn_output)
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        return hidden_states, position_bias


class T5LayerFF(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config)
        else:
            self.DenseReluDense = T5DenseActDense(config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = self.layer_norm(hidden_states)
        ff_output = self.DenseReluDense(normed)
        # Use Triton for residual add; enforce contiguous to prevent OOB
        hidden_states = triton_elementwise_add_contig(hidden_states, ff_output)
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        return hidden_states


class ModelNew(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.layer = nn.ModuleList([
            T5LayerSelfAttention(config, has_relative_attention_bias),
            T5LayerFF(config),
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, position_bias = self.layer[0](
            hidden_states, mask=mask, position_bias=position_bias,
        )
        hidden_states = self.layer[1](hidden_states)
        return hidden_states, position_bias

T5Block = ModelNew
