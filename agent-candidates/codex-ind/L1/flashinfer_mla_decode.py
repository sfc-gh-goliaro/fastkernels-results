from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _mla_decode_kernel(
    q,
    kv,
    block_table,
    seq_lens,
    out,
    q_stride,
    bt_stride,
    scale: tl.constexpr,
    bmm2_scale: tl.constexpr,
):
    batch = tl.program_id(0)

    heads = tl.arange(0, 16)
    d_latent = tl.arange(0, 512)
    d_rope = tl.arange(0, 64)
    tokens = tl.arange(0, 64)

    q_base = q + batch * q_stride + heads[:, None] * 576
    q_latent = tl.load(q_base + d_latent[None, :])
    q_rope = tl.load(q_base + 512 + d_rope[None, :])

    seq_len = tl.load(seq_lens + batch)
    page = 0
    running_max = tl.full((16,), -float("inf"), tl.float32)
    running_sum = tl.zeros((16,), tl.float32)
    accumulator = tl.zeros((16, 512), tl.float32)

    while page * 64 < seq_len:
        page_id = tl.load(block_table + batch * bt_stride + page).to(tl.int64)
        token_offsets = page * 64 + tokens
        valid = token_offsets < seq_len
        kv_base = kv + page_id * (64 * 576) + tokens[:, None] * 576

        k_latent = tl.load(
            kv_base + d_latent[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        k_rope = tl.load(
            kv_base + 512 + d_rope[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.dot(q_latent, tl.trans(k_latent))
        scores += tl.dot(q_rope, tl.trans(k_rope))
        scores = scores * scale
        scores = tl.where(valid[None, :], scores, -float("inf"))

        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, tile_max)
        alpha = tl.exp2((running_max - new_max) * 1.4426950408889634)
        probabilities = tl.exp2(
            (scores - new_max[:, None]) * 1.4426950408889634
        )

        accumulator *= alpha[:, None]
        accumulator += tl.dot(probabilities.to(tl.bfloat16), k_latent)
        running_sum = running_sum * alpha + tl.sum(probabilities, axis=1)
        running_max = new_max
        page += 1

    result = accumulator / running_sum[:, None]
    result *= bmm2_scale
    out_base = out + batch * (16 * 512) + heads[:, None] * 512
    tl.store(out_base + d_latent[None, :], result)


@triton.jit
def _mla_decode_split_kernel(
    q,
    kv,
    block_table,
    seq_lens,
    partial_out,
    partial_lse,
    q_stride,
    bt_stride,
    scale: tl.constexpr,
    SPLITS: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
):
    batch = tl.program_id(0)
    part = tl.program_id(1)

    heads = tl.arange(0, 16)
    d_latent = tl.arange(0, 512)
    d_rope = tl.arange(0, 64)
    tokens = tl.arange(0, 64)

    q_base = q + batch * q_stride + heads[:, None] * 576
    q_latent = tl.load(q_base + d_latent[None, :])
    q_rope = tl.load(q_base + 512 + d_rope[None, :])

    seq_len = tl.load(seq_lens + batch)
    num_pages = (seq_len + 63) // 64
    start_page = (num_pages * part) // SPLITS
    end_page = (num_pages * (part + 1)) // SPLITS
    running_max = tl.full((16,), -float("inf"), tl.float32)
    running_sum = tl.zeros((16,), tl.float32)
    accumulator = tl.zeros((16, 512), tl.float32)

    for page in tl.range(start_page, end_page, num_stages=PIPE_STAGES):
        page_id = tl.load(block_table + batch * bt_stride + page).to(tl.int64)
        token_offsets = page * 64 + tokens
        valid = token_offsets < seq_len
        kv_base = kv + page_id * (64 * 576) + tokens[:, None] * 576

        k_latent = tl.load(
            kv_base + d_latent[None, :], mask=valid[:, None], other=0.0
        )
        k_rope = tl.load(
            kv_base + 512 + d_rope[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.dot(q_latent, tl.trans(k_latent))
        scores += tl.dot(q_rope, tl.trans(k_rope))
        scores = tl.where(
            valid[None, :], scores * scale, -float("inf")
        )

        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, tile_max)
        alpha = tl.exp2((running_max - new_max) * 1.4426950408889634)
        probabilities = tl.exp2(
            (scores - new_max[:, None]) * 1.4426950408889634
        )
        accumulator *= alpha[:, None]
        accumulator += tl.dot(probabilities.to(tl.bfloat16), k_latent)
        running_sum = running_sum * alpha + tl.sum(probabilities, axis=1)
        running_max = new_max
    nonempty = running_sum > 0.0
    result = tl.where(
        nonempty[:, None],
        accumulator / running_sum[:, None],
        0.0,
    )
    base = (batch * SPLITS + part) * (16 * 512)
    tl.store(partial_out + base + heads[:, None] * 512 + d_latent[None, :], result)
    tl.store(
        partial_lse + (batch * SPLITS + part) * 16 + heads,
        tl.where(nonempty, running_max + tl.log(running_sum), -float("inf")),
    )


@triton.jit
def _mla_decode_reduce_kernel(
    partial_out,
    partial_lse,
    out,
    bmm2_scale: tl.constexpr,
    SPLITS: tl.constexpr,
):
    batch = tl.program_id(0)
    heads = tl.arange(0, 16)
    dims = tl.arange(0, 512)

    max_lse = tl.full((16,), -float("inf"), tl.float32)
    for part in tl.static_range(SPLITS):
        lse = tl.load(partial_lse + (batch * SPLITS + part) * 16 + heads)
        max_lse = tl.maximum(max_lse, lse)

    denominator = tl.zeros((16,), tl.float32)
    accumulator = tl.zeros((16, 512), tl.float32)
    for part in tl.static_range(SPLITS):
        lse = tl.load(partial_lse + (batch * SPLITS + part) * 16 + heads)
        weight = tl.exp2((lse - max_lse) * 1.4426950408889634)
        values = tl.load(
            partial_out
            + (batch * SPLITS + part) * (16 * 512)
            + heads[:, None] * 512
            + dims[None, :]
        )
        accumulator += weight[:, None] * values
        denominator += weight

    result = accumulator / denominator[:, None] * bmm2_scale
    tl.store(
        out + batch * (16 * 512) + heads[:, None] * 512 + dims[None, :],
        result,
    )


def flashinfer_mla_decode_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


class FlashInferMLADecode(nn.Module):
    def __init__(
        self,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        kv_lora_rank: int,
        workspace: torch.Tensor | None = None,
    ):
        super().__init__()
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self._workspace = workspace
        self._partial_out = None
        self._partial_lse = None

    @property
    def available(self) -> bool:
        return flashinfer_mla_decode_supported()

    def ensure_workspaces(self, device: torch.device) -> None:
        return None

    def forward(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        softmax_scale: float,
        max_seq_len: int,
        bmm2_scale: float = 1.0,
    ):
        assert q.shape[1:] == (1, 16, 576)
        assert kv_cache.shape[1:] == (64, 576)

        out = torch.empty(
            (q.shape[0], 1, 16, 512), device=q.device, dtype=q.dtype
        )
        batch = q.shape[0]
        if batch < 16 or max_seq_len <= 512:
            _mla_decode_kernel[(batch,)](
                q,
                kv_cache,
                block_table,
                cache_seqlens,
                out,
                q.stride(0),
                block_table.stride(0),
                scale=softmax_scale,
                bmm2_scale=bmm2_scale,
                num_warps=4,
                num_stages=1,
            )
        else:
            if batch >= 512:
                splits = 2
            elif max_seq_len > 32768:
                splits = 32
            else:
                splits = 8
            partial_shape = (batch, splits, 16, 512)
            if (
                self._partial_out is None
                or self._partial_out.shape != partial_shape
                or self._partial_out.device != q.device
            ):
                self._partial_out = torch.empty(
                    partial_shape, device=q.device, dtype=q.dtype
                )
                self._partial_lse = torch.empty(
                    (batch, splits, 16), device=q.device, dtype=torch.float32
                )
            _mla_decode_split_kernel[(batch, splits)](
                q,
                kv_cache,
                block_table,
                cache_seqlens,
                self._partial_out,
                self._partial_lse,
                q.stride(0),
                block_table.stride(0),
                scale=softmax_scale,
                SPLITS=splits,
                PIPE_STAGES=2,
                num_warps=4,
                num_stages=1,
            )
            _mla_decode_reduce_kernel[(batch,)](
                self._partial_out,
                self._partial_lse,
                out,
                bmm2_scale=bmm2_scale,
                SPLITS=splits,
                num_warps=4,
                num_stages=1,
            )
        return out, None
