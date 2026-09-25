"""Oasis VAE self-attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastkernels.infra.cuda_ext import lazy_op

from ..L1.dense_attention import DenseAttention
from ..L1.linear import Linear
from ..L1.oasis_rotary import OasisRotaryEmbedding


_C = lazy_op(
    "oasis_vae_attention_candidate_v1", "oasis_vae_attention.cu"
)


class OasisVAEAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        frame_height: int,
        frame_width: int,
        *,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = Linear(dim, dim, bias=True)
        self.rotary = OasisRotaryEmbedding(
            dim=(dim // num_heads) // 4,
            freqs_for="pixel",
            max_freq=frame_height * frame_width,
        )
        y = torch.linspace(-1, 1, steps=frame_height)
        x = torch.linspace(-1, 1, steps=frame_width)
        y_freqs = (y[:, None] * self.rotary.freqs).repeat_interleave(2, dim=-1)
        x_freqs = (x[:, None] * self.rotary.freqs).repeat_interleave(2, dim=-1)
        y_freqs = y_freqs[:, None, :].expand(-1, frame_width, -1)
        x_freqs = x_freqs[None, :, :].expand(frame_height, -1, -1)
        self.register_buffer(
            "rotary_freqs",
            torch.cat((y_freqs, x_freqs), dim=-1),
            persistent=False,
        )
        self.register_buffer(
            "rotary_cos", self.rotary_freqs.cos(), persistent=False
        )
        self.register_buffer(
            "rotary_sin", self.rotary_freqs.sin(), persistent=False
        )
        self.attn = DenseAttention(backend="sdpa")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        seq_len = self.frame_height * self.frame_width
        qkv_flat = self.qkv(x)
        head_dim = qkv_flat.shape[-1] // (3 * self.num_heads)
        qkv = qkv_flat.reshape(bsz, seq_len, 3, self.num_heads, head_dim)
        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]
        _C.rotary_inplace(
            qkv_flat, self.rotary_cos, self.rotary_sin, self.num_heads
        )
        out = F.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3),
            k.permute(0, 2, 1, 3),
            v.permute(0, 2, 1, 3),
            dropout_p=0.0,
        ).permute(0, 2, 1, 3)
        out = out.reshape(bsz, seq_len, -1)
        return self.proj(out)
