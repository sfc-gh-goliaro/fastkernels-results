"""CLIP MLP and text embeddings (L2).

CLIPMLP: Linear -> QuickGELU -> Linear (no TP, frozen encoder).
CLIPTextEmbeddings: token + position embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice
from transformers import CLIPTextConfig

from ..L1.embedding import Embedding
from ..L1.linear import Linear


@triton.jit
def _linear_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    ACTIVATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        x = tl.load(
            x_ptr + offs_m[:, None] * K + (k + offs_k)[None, :],
            mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K),
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_n[None, :] * K + (k + offs_k)[:, None],
            mask=(offs_n[None, :] < N) & (k + offs_k[:, None] < K),
            other=0.0,
        )
        x_hi = x.to(tl.float16)
        w_hi = w.to(tl.float16)
        acc += tl.dot(x_hi, w_hi)

    acc += tl.load(bias_ptr + offs_n[None, :], mask=offs_n[None, :] < N)
    if ACTIVATE:
        acc /= 1.0 + libdevice.exp(-1.702 * acc)
    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _text_embedding_kernel(
    input_ids_ptr,
    token_ptr,
    position_ptr,
    out_ptr,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    token_id = tl.load(input_ids_ptr + row)
    mask = cols < HIDDEN
    token = tl.load(token_ptr + token_id * HIDDEN + cols, mask=mask)
    position = tl.load(position_ptr + row * HIDDEN + cols, mask=mask)
    tl.store(out_ptr + row * HIDDEN + cols, token + position, mask=mask)


class CLIPMLP(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.fc1 = Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = Linear(config.intermediate_size, config.hidden_size, bias=True)
        self._fc1_weight_fp16 = None
        self._fc2_weight_fp16 = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._fc1_weight_fp16 is None:
            self._fc1_weight_fp16 = self.fc1.weight.to(torch.float16)
            self._fc2_weight_fp16 = self.fc2.weight.to(torch.float16)
        m = hidden_states.numel() // hidden_states.shape[-1]
        intermediate = torch.empty(
            (m, self.fc1.weight.shape[0]),
            device=hidden_states.device,
            dtype=torch.float16,
        )
        _linear_kernel[(triton.cdiv(m, 32), triton.cdiv(intermediate.shape[1], 64))](
            hidden_states,
            self._fc1_weight_fp16,
            self.fc1.bias,
            intermediate,
            M=m,
            N=intermediate.shape[1],
            K=hidden_states.shape[-1],
            ACTIVATE=True,
            BLOCK_M=32,
            BLOCK_N=64,
            BLOCK_K=64,
            num_warps=4,
            num_stages=4,
        )
        output = torch.empty(
            (m, self.fc2.weight.shape[0]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        _linear_kernel[(triton.cdiv(m, 32), triton.cdiv(output.shape[1], 32))](
            intermediate,
            self._fc2_weight_fp16,
            self.fc2.bias,
            output,
            M=m,
            N=output.shape[1],
            K=intermediate.shape[-1],
            ACTIVATE=False,
            BLOCK_M=32,
            BLOCK_N=32,
            BLOCK_K=128,
            num_warps=4,
            num_stages=3,
        )
        return output.reshape(*hidden_states.shape[:-1], output.shape[-1])


class CLIPTextEmbeddings(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.token_embedding = Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = Embedding(config.max_position_embeddings, config.hidden_size)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        output = torch.empty(
            (*input_ids.shape, self.token_embedding.emb.weight.shape[1]),
            device=input_ids.device,
            dtype=self.token_embedding.emb.weight.dtype,
        )
        _text_embedding_kernel[
            (input_ids.numel(), triton.cdiv(output.shape[-1], 512))
        ](
            input_ids,
            self.token_embedding.emb.weight,
            self.position_embedding.emb.weight,
            output,
            HIDDEN=output.shape[-1],
            BLOCK=512,
            num_warps=4,
        )
        return output
