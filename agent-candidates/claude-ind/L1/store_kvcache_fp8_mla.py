"""MLA KV cache store and gather -- optimized BF16 ("auto") path.

The benched workload uses ``kv_cache_dtype="auto"``: a BF16 paged cache of
shape ``[num_blocks, block_size, 576]`` where both the store and the gather
degenerate to a pure 1152-byte-per-token copy (512 ``kv_c_normed`` elements
followed by 64 ``k_pe`` elements).  The vendored vLLM kernels leave most of
HBM unused on that path -- ``concat_and_cache_mla_kernel`` moves *two bytes*
per thread and launches one CTA per token, and
``gather_and_maybe_dequant_cache`` launches one 64-thread CTA per token -- so
both are dominated by CTA scheduling rather than bandwidth.

``mla_kv_fast.cu`` replaces them with 128-bit-per-access kernels in which one
warp owns a whole token (making every ``slot_mapping`` / ``block_table`` /
``cu_seq_lens`` lookup warp-uniform) and a grid sized to the GPU instead of to
the token count.

The quantized layouts (``fp8_ds_mla``, ``fp8_e4m3``) are not exercised by this
workload; they keep the documented semantics via a straightforward PyTorch
implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fastkernels.infra.cuda_ext import lazy_op

# Distinct extension name: the baseline module (same process, under
# ``fastkernels.tasks.baseline``) owns the "store_kvcache_fp8_mla" build.
_C = lazy_op("mla_kv_fast_cand", "mla_kv_fast.cu")

_KV_C_DIM = 512
_K_PE_DIM = 64
_FP8_BYTES_PER_TOKEN = 656
_BF16_ELEMS_PER_TOKEN = _KV_C_DIM + _K_PE_DIM  # 576

_FP8_MAX = 448.0
_FP8_SCALE_DIVISOR = 448.0


def _fp8_quant(x: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
    return (x.float() / scale).clamp_(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)


class StoreKVCacheFP8MLA(nn.Module):
    """Store ``kv_c_normed`` and ``k_pe`` into the MLA paged cache.

    Args:
        kv_c_normed: ``[N, 512]`` BF16 -- compressed KV after layernorm.
        k_pe: ``[N, 1, 64]`` or ``[N, 64]`` BF16 -- RoPE key component.
        kv_cache: ``[num_blocks, block_size, 576|656]`` (BF16 / fp8 / uint8).
        slot_mapping: ``[N]`` int64 -- linear slot index per token (``-1`` skips).
    """

    def __init__(self, kv_cache_dtype: str = "auto"):
        super().__init__()
        assert kv_cache_dtype in ("auto", "fp8_ds_mla", "fp8_e4m3"), (
            f"StoreKVCacheFP8MLA: unsupported kv_cache_dtype={kv_cache_dtype!r}"
        )
        self.kv_cache_dtype = kv_cache_dtype
        self.register_buffer(
            "_k_scale", torch.ones(1, dtype=torch.float32), persistent=False,
        )

    def forward(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        k_pe_2d = k_pe.reshape(k_pe.shape[0], -1)
        if self.kv_cache_dtype == "auto":
            _C.store_mla(kv_c_normed, k_pe_2d, kv_cache, slot_mapping)
            return
        self._store_quantized(kv_c_normed, k_pe_2d, kv_cache, slot_mapping)

    def _store_quantized(
        self,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Reference PyTorch path for the two quantized cache layouts."""
        keep = slot_mapping >= 0
        if not bool(keep.all()):
            kv_c, k_pe, slot_mapping = kv_c[keep], k_pe[keep], slot_mapping[keep]
        if slot_mapping.numel() == 0:
            return
        if self.kv_cache_dtype == "fp8_e4m3":
            flat = kv_cache.view(-1, kv_cache.shape[-1])
            row = torch.cat((kv_c, k_pe), dim=-1)
            flat[slot_mapping] = _fp8_quant(row, self._k_scale)
            return
        # fp8_ds_mla: [0:512] fp8 kv_c, [512:528] four fp32 group scales,
        # [528:656] k_pe as bf16.
        n = kv_c.shape[0]
        groups = kv_c.float().view(n, 4, 128)
        scales = (groups.abs().amax(dim=-1) / _FP8_SCALE_DIVISOR).clamp_min(
            torch.finfo(torch.float32).tiny)
        quant = _fp8_quant(groups, scales.unsqueeze(-1)).view(n, _KV_C_DIM)
        row = torch.empty(n, _FP8_BYTES_PER_TOKEN, dtype=torch.uint8,
                          device=kv_c.device)
        row[:, :_KV_C_DIM] = quant.view(torch.uint8)
        row[:, _KV_C_DIM:_KV_C_DIM + 16] = scales.view(torch.uint8)
        row[:, _KV_C_DIM + 16:] = k_pe.contiguous().view(torch.uint8)
        kv_cache.view(-1, _FP8_BYTES_PER_TOKEN)[slot_mapping] = row


class GatherKVCacheFP8MLA(nn.Module):
    """Gather and upconvert KV from an FP8 MLA paged cache to BF16.

    Kept for call-site compatibility with the baseline module (this workload
    never constructs it); delegates to the vendored kernel via the baseline.
    """

    def forward(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        workspace_starts: torch.Tensor,
        num_seqs: int,
        workspace: torch.Tensor,
    ) -> None:
        from fastkernels.tasks.baseline.L1.store_kvcache_fp8_mla import (
            GatherKVCacheFP8MLA as _Base,
        )
        _Base.forward(self, kv_cache, block_table, seq_lens, workspace_starts,
                      num_seqs, workspace)


class GatherAndDequantKVCacheMLA(nn.Module):
    """Gather MLA KV cache rows into a contiguous BF16 workspace.

    Args:
        kv_cache: ``[num_blocks, block_size, 576]`` BF16 (``"auto"``) or
            ``[num_blocks, block_size, 656]`` uint8 (``fp8_ds_mla``).
        workspace: ``[total_tokens, 576]`` BF16 output buffer.
        block_table: ``[num_seqs, max_blocks]`` int32.
        cu_seq_lens: ``[num_seqs+1]`` int32 cumulative sequence lengths.
        token_to_seq: ``[total_tokens]`` int32 token -> sequence map.
        total_tokens: scalar int.
        workspace_starts: ``[num_seqs]`` int32 starting workspace row per
            sequence (chunked-context gathers).
    """

    def __init__(self, kv_cache_dtype: str = "fp8_ds_mla"):
        super().__init__()
        assert kv_cache_dtype in ("auto", "fp8_ds_mla", "fp8_e4m3"), (
            f"GatherAndDequantKVCacheMLA: unsupported "
            f"kv_cache_dtype={kv_cache_dtype!r}"
        )
        self.kv_cache_dtype = kv_cache_dtype
        self.register_buffer(
            "_k_scale", torch.ones(1, dtype=torch.float32), persistent=False,
        )

    def forward(
        self,
        kv_cache: torch.Tensor,
        workspace: torch.Tensor,
        block_table: torch.Tensor,
        cu_seq_lens: torch.Tensor,
        token_to_seq: torch.Tensor,
        total_tokens: int,
        workspace_starts: torch.Tensor,
    ) -> None:
        if self.kv_cache_dtype == "auto":
            _C.gather_mla(kv_cache, workspace, block_table, cu_seq_lens,
                          token_to_seq, total_tokens, workspace_starts)
            return
        self._gather_quantized(kv_cache, workspace, block_table, cu_seq_lens,
                               token_to_seq, total_tokens, workspace_starts)

    def _gather_quantized(
        self,
        kv_cache: torch.Tensor,
        workspace: torch.Tensor,
        block_table: torch.Tensor,
        cu_seq_lens: torch.Tensor,
        token_to_seq: torch.Tensor,
        total_tokens: int,
        workspace_starts: torch.Tensor,
    ) -> None:
        """Reference PyTorch path: dequantize with the per-tensor ``k_scale``."""
        block_size = kv_cache.shape[1]
        tok = torch.arange(total_tokens, device=kv_cache.device,
                           dtype=torch.int32)
        seq = token_to_seq[:total_tokens].long()
        off = (tok - cu_seq_lens[seq] + workspace_starts[seq]).long()
        blk = block_table[seq, off // block_size].long()
        rows = kv_cache.view(-1, kv_cache.shape[-1])[blk * block_size
                                                     + off % block_size]
        workspace[:total_tokens] = (rows.float() * self._k_scale).to(
            workspace.dtype)
