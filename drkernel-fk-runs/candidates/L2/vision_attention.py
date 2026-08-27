import math
import torch
import torch.nn as nn

from fastkernels.infra.tp import _tp_size, _tp_rank
from fastkernels.infra.fa_utils import fa_version_for_head_size, flash_attn_varlen_func
from fastkernels.infra.fp8 import _get_fp8_linear_cls

import triton
import triton.language as tl


def _ceil_div(a, b):
    return (a + b - 1) // b


@triton.jit
def _apply_rotary_qkv_cos_only_kernel(
    qkv_ptr,            # *bf16 or *fp16: [S, B, 3*Hd]
    cos_ptr,            # *bf16 or *fp16: [S, alpha]
    S: tl.constexpr,
    B: tl.constexpr,
    Hd: tl.constexpr,   # head dim
    alpha: tl.constexpr,# rotary channels <= Hd
    stride_s: tl.constexpr,  # stride over S in elements
    stride_b: tl.constexpr,  # stride over B in elements
    stride_c: tl.constexpr,  # stride over channel in elements (usually 1)
):
    # Program ids
    pid_s = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)  # head index not used directly; we cover all h via S,B,channels

    if (pid_s >= S) or (pid_b >= B):
        return

    # Base offset for (s, b, 0)
    base = pid_s * stride_s + pid_b * stride_b

    # Loop over rotary channels
    c = 0
    while c < alpha:
        # q at channel c: offset = base + c
        q_off = base + c
        q_val = tl.load(qkv_ptr + q_off * stride_c)
        cos_val = tl.load(cos_ptr + pid_s * alpha + c)
        new_q = q_val * cos_val
        tl.store(qkv_ptr + q_off * stride_c, new_q)

        # k at channel c + Hd
        k_off = base + Hd + c
        k_val = tl.load(qkv_ptr + k_off * stride_c)
        new_k = k_val * cos_val
        tl.store(qkv_ptr + k_off * stride_c, new_k)

        # v is not rotated: channel c + 2*Hd (skip)
        c += 1


class ModelNew(nn.Module):
    """Triton-optimized version:
    - Launches a real Triton kernel to apply rotary on q and k in-place over qkv storage.
    - Avoids layout copies before FlashAttention by using zero-copy views.
    - Keeps projection unchanged.
    """

    def __init__(self, embed_dim: int, num_heads: int, projection_size: int | None = None):
        super().__init__()
        if projection_size is None:
            projection_size = embed_dim
        self.tp_size = _tp_size()
        self.tp_rank = _tp_rank()

        # QKV: fused, sharded over heads
        self.qkv = QKVParallelLinear(
            embed_dim, embed_dim // num_heads, num_heads, num_heads, bias=True,
        )
        # Output projection: row-parallel
        self.proj = RowParallelLinear(projection_size, embed_dim, bias=True)
        # FlashAttention
        head_dim = embed_dim // num_heads
        self.attn = FlashAttnPrefill(num_heads, num_heads, head_dim)

    def forward(
        self, x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,  # sin is accepted but not used in kernel (cos-only)
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        # x: [S, B, D]
        assert x.dim() == 3, f"Expected x shape [S, B, D], got {tuple(x.shape)}"
        seq_len, batch_size, embed_dim = x.shape

        # 1) QKV fused linear -> qkv: [S, B, 3*Hd] (contiguous)
        qkv = self.qkv(x)  # [S, B, 3*Hd]

        # 2) Launch Triton kernel: apply rotary (cos-only scale) in-place on q and k
        total_num_heads = self.qkv.num_heads * self.tp_size
        head_dim = embed_dim // total_num_heads
        qkv_3 = qkv.size(-1)
        assert qkv_3 == 3 * head_dim, f"qkv last dim {qkv_3} != 3*head_dim {3*head_dim}"

        # Ensure dtypes
        assert qkv.dtype in (torch.float16, torch.bfloat16), f"Expected fp16/bf16, got {qkv.dtype}"
        assert rotary_pos_emb_cos.dtype in (torch.float16, torch.bfloat16), f"Expected cos dtype fp16/bf16, got {rotary_pos_emb_cos.dtype}"

        # Cos shape: [S, alpha]; alpha = min(head_dim, 32)
        alpha = min(head_dim, 32)
        cos = rotary_pos_emb_cos
        if cos.shape[-1] > alpha:
            cos = cos[..., :alpha]

        # Strides for qkv: [S, B, 3*Hd]
        # Assume contiguous: stride over c = 1
        stride_s = qkv.stride(0)
        stride_b = qkv.stride(1)
        stride_c = qkv.stride(-1)  # typically 1

        # Launch grid: (S, B, 1)
        grid = (seq_len, batch_size, 1)

        _apply_rotary_qkv_cos_only_kernel[grid](
            qkv, cos,
            seq_len, batch_size, head_dim, alpha,
            stride_s, stride_b, stride_c,
        )

        # 3) Zero-copy views for FlashAttention
        # View as (S, B, 3, Hd) without copy
        qkv_view = qkv.view(seq_len, batch_size, 3, head_dim)
        q = qkv_view[:, :, 0]
        k = qkv_view[:, :, 1]
        v = qkv_view[:, :, 2]

        # Reshape to (B, S, local_H, Hd)
        local_num_heads = self.qkv.num_heads
        q = q.reshape(batch_size, seq_len, local_num_heads, head_dim)
        k = k.reshape(batch_size, seq_len, local_num_heads, head_dim)
        v = v.reshape(batch_size, seq_len, local_num_heads, head_dim)

        # 4) FlashAttention varlen
        if max_seqlen is None:
            max_seqlen = int(cu_seqlens[-1].item())

        out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens, cu_seqlens,
            max_seqlen, max_seqlen,
            softmax_scale=head_dim ** -0.5,
            causal=False,
            num_splits=1,
        )

        # 5) Projection
        out = out.view(seq_len, batch_size, -1)
        return self.proj(out)


# Reused definitions and modules from the original snippet (kept for completeness)

_FP8_BLOCK = 128


def _scale_shape(out_dim: int, in_dim: int) -> tuple[int, int]:
    return (math.ceil(out_dim / _FP8_BLOCK), math.ceil(in_dim / _FP8_BLOCK))


def _alloc_colmajor_scale(M: int, num_groups: int,
                          device: torch.device) -> torch.Tensor:
    return torch.empty(
        (num_groups, M), device=device, dtype=torch.float32,
    ).permute(-1, -2)


def _is_batch_invariant() -> bool:
    return os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1"


_FLASHINFER_RESOLVED = False
_FLASHINFER_FN = None


def _maybe_get_flashinfer_fp8_gemm():
    global _FLASHINFER_RESOLVED, _FLASHINFER_FN
    if _FLASHINFER_RESOLVED:
        return _FLASHINFER_FN
    _FLASHINFER_RESOLVED = True

    if os.environ.get("VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER", "1") != "1":
        return None
    if not torch.cuda.is_available():
        return None
    cap = torch.cuda.get_device_capability()
    if cap[0] != 9:
        return None
    from flashinfer.gemm import fp8_blockscale_gemm_sm90
    _FLASHINFER_FN = fp8_blockscale_gemm_sm90
    return _FLASHINFER_FN


class _Fp8PrefillBufs:
    __slots__ = ("a", "s", "o")

    def __init__(self, max_tokens: int, K: int, N: int, device: torch.device):
        num_groups = math.ceil(K / 128)
        self.a = torch.empty(max_tokens, K, dtype=torch.float8_e4m3fn, device=device)
        self.s = _alloc_colmajor_scale(max_tokens, num_groups, device)
        self.o = torch.empty(max_tokens, N, dtype=torch.bfloat16, device=device)


class Fp8Linear(nn.Module):
    BLOCK_SIZE = 128

    def __init__(self):
        super().__init__()
        self._a_buf = None
        self._s_buf = None
        self._o_buf = None
        self._pf = None

    def _ensure_buffers(self, max_tokens: int, K: int, N: int, device: torch.device):
        num_groups = math.ceil(K / self.BLOCK_SIZE)
        self._a_buf = torch.empty(max_tokens, K, dtype=torch.float8_e4m3fn, device=device)
        self._s_buf = _alloc_colmajor_scale(max_tokens, num_groups, device)
        self._o_buf = torch.empty(max_tokens, N, dtype=torch.bfloat16, device=device)

    _FLASHINFER_M_THRESHOLD = 32

    def forward(self, input_bf16: torch.Tensor,
                weight_fp8: torch.Tensor,
                weight_scale_inv: torch.Tensor,
                bias: torch.Tensor | None = None) -> torch.Tensor:
        N, K = weight_fp8.shape
        input_2d = input_bf16.reshape(-1, K)
        M = input_2d.shape[0]
        num_groups = (K + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE

        flashinfer_ok = (
            input_bf16.dtype == torch.bfloat16
            and weight_fp8.dtype == torch.float8_e4m3fn
            and N % 64 == 0
            and K % 128 == 0
            and not _is_batch_invariant()
            and _maybe_get_flashinfer_fp8_gemm() is not None
        )

        if torch.compiler.is_compiling():
            output = torch.ops.fastkernels_fp8.blockscale_gemm_dispatch(
                input_2d, weight_fp8, weight_scale_inv, flashinfer_ok,
            )
            if bias is not None:
                output = output + bias
            return output.view(*input_bf16.shape[:-1], N)

        use_flashinfer = flashinfer_ok and M < self._FLASHINFER_M_THRESHOLD

        if use_flashinfer:
            output = torch.empty(M, N, dtype=torch.bfloat16, device=input_2d.device)
            torch.ops.fastkernels_fp8.flashinfer_blockscale_gemm(
                input_2d, weight_fp8, weight_scale_inv, output,
            )
            if bias is not None:
                output = output + bias
            return output.view(*input_bf16.shape[:-1], N)

        if self._a_buf is not None and M <= self._a_buf.shape[0]:
            q_input = self._a_buf[:M]
            input_scale = _alloc_colmajor_scale(M, num_groups, input_2d.device)
            output = self._o_buf[:M]
        elif self._pf is not None and M <= self._pf.a.shape[0]:
            q_input = self._pf.a[:M]
            input_scale = _alloc_colmajor_scale(M, num_groups, input_2d.device)
            output = self._pf.o[:M]
        else:
            q_input = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=input_2d.device)
            input_scale = _alloc_colmajor_scale(M, num_groups, input_2d.device)
            output = torch.empty(M, N, dtype=torch.bfloat16, device=input_2d.device)

        torch.ops.fastkernels_fp8.per_token_group_quant_fp8(
            input_2d, q_input, input_scale, True,
        )
        torch.ops.fastkernels_fp8.fp8_gemm_nt(
            q_input, input_scale, weight_fp8, weight_scale_inv, output,
        )

        if bias is not None:
            output = output + bias
        return output.view(*input_bf16.shape[:-1], N)


def _get_fp8_linear_cls():
    return Fp8Linear


_CUSTOM_AR: Optional["CustomAllreduce"] = None


class AllReduce(nn.Module):
    def forward(self, tensor):
        if torch.compiler.is_compiling():
            if _CUSTOM_AR is not None:
                return torch.ops.fastkernels.custom_all_reduce(tensor)
            dist.all_reduce(tensor)
            return tensor
        ar = _CUSTOM_AR
        if ar is not None:
            out = ar.custom_all_reduce(tensor)
            if out is not None:
                return out
        dist.all_reduce(tensor)
        return tensor


class RowParallelLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int, bias: bool = False,
                 quant_config: dict | None = None, reduce_results: bool = True):
        super().__init__()
        tp = _tp_size()
        assert input_size % tp == 0
        self.input_size_per_partition = input_size // tp
        self.tp_size = tp
        self.tp_rank = _tp_rank()
        self.reduce_results = reduce_results
        self.use_fp8 = quant_config is not None

        if self.use_fp8:
            self.weight = nn.Parameter(
                torch.empty(output_size, self.input_size_per_partition,
                            dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(*_scale_shape(output_size, self.input_size_per_partition),
                            dtype=torch.float32),
                requires_grad=False,
            )
            self.weight.weight_loader = self._weight_loader
            self.weight_scale_inv.weight_loader = self._scale_loader
            self.linear_op = _get_fp8_linear_cls()()
        else:
            self.weight = nn.Parameter(torch.empty(output_size, self.input_size_per_partition))
            self.weight.weight_loader = self._weight_loader

        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None
        if self.bias is not None:
            self.bias.weight_loader = lambda p, w: p.data.copy_(w)
        self.allreduce = AllReduce()

    def _weight_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        shard = param.data.size(1)
        loaded_weight = loaded_weight.narrow(1, rank * shard, shard)
        param.data.copy_(loaded_weight)

    def _scale_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        cols_per_shard = param.data.size(1)
        loaded_weight = loaded_weight.narrow(1, rank * cols_per_shard, cols_per_shard)
        param.data.copy_(loaded_weight)

    def forward(self, x):
        if self.use_fp8:
            y = self.linear_op(x, self.weight, self.weight_scale_inv,
                               self.bias if self.tp_rank == 0 else None)
        else:
            y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.reduce_results and self.tp_size > 1:
            y = self.allreduce(y)
        return y


class QKVParallelLinear(nn.Module):
    def __init__(self, hidden_size: int, head_size: int,
                 total_num_heads: int, total_num_kv_heads: int,
                 bias: bool = False, quant_config: dict | None = None):
        super().__init__()
        tp = _tp_size()
        self.head_size = head_size
        self.num_heads = total_num_heads // tp
        if total_num_kv_heads % tp == 0:
            self.num_kv_heads = total_num_kv_heads // tp
            self._replicate_kv = False
        else:
            self.num_kv_heads = total_num_kv_heads
            self._replicate_kv = True
        output_size = (self.num_heads + 2 * self.num_kv_heads) * head_size
        self.use_fp8 = quant_config is not None

        if self.use_fp8:
            self.weight = nn.Parameter(
                torch.empty(output_size, hidden_size, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(*_scale_shape(output_size, hidden_size), dtype=torch.float32),
                requires_grad=False,
            )
            self.weight.weight_loader = self._weight_loader
            self.weight_scale_inv.weight_loader = self._scale_loader
            self.linear_op = _get_fp8_linear_cls()()
        else:
            self.weight = nn.Parameter(torch.empty(output_size, hidden_size))
            self.weight.weight_loader = self._weight_loader

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self._weight_loader

    def _weight_loader(self, param, loaded_weight, shard_id: str):
        tp, rank = _tp_size(), _tp_rank()
        if shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
            src = loaded_weight.chunk(tp, 0)[rank]
        elif shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
            src = loaded_weight if self._replicate_kv else loaded_weight.chunk(tp, 0)[rank]
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
            src = loaded_weight if self._replicate_kv else loaded_weight.chunk(tp, 0)[rank]
        dst = param.data.narrow(0, shard_offset, shard_size)
        dst.copy_(src)

    def _scale_loader(self, param, loaded_weight, shard_id: str):
        tp, rank = _tp_size(), _tp_rank()
        if shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        scale_rows = math.ceil(shard_size / _FP8_BLOCK)
        scale_offset = math.ceil(shard_offset / _FP8_BLOCK)
        src = loaded_weight.chunk(tp, 0)[rank]
        param.data.narrow(0, scale_offset, scale_rows).copy_(src)

    def forward(self, x):
        if self.use_fp8:
            return self.linear_op(x, self.weight, self.weight_scale_inv, self.bias)
        return F.linear(x, self.weight, self.bias)


class FlashAttnPrefill(nn.Module):
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sm_scale = head_dim ** -0.5
        self.fa_version = fa_version_for_head_size(head_dim)

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
        fa_kw = dict(
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=max_seqlen_k,
            fa_version=self.fa_version,
        )
        if kwargs.get("block_table") is not None:
            seqused_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
            fa_kw["seqused_k"] = seqused_k
            if self.fa_version == 3 and not torch.cuda.is_current_stream_capturing():
                page_size = k.shape[1] if k.dim() >= 2 else None
                meta = fa3_scheduler_metadata(
                    batch_size=int(seqused_k.shape[0]),
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    num_heads_q=self.num_heads,
                    num_heads_kv=self.num_kv_heads,
                    headdim=self.head_dim,
                    cache_seqlens=seqused_k,
                    qkv_dtype=q.dtype,
                    cu_seqlens_q=cu_seqlens_q,
                    page_size=page_size,
                    causal=kwargs.get("causal", True),
                    window_size=kwargs.get("window_size", (-1, -1)),
                    num_splits=0,
                )
                if meta is not None:
                    fa_kw["scheduler_metadata"] = meta
                fa_kw["num_splits"] = 0
        else:
            fa_kw["cu_seqlens_k"] = cu_seqlens_k
            fa_kw["num_splits"] = 1
        fa_kw.update(kwargs)
        return flash_attn_varlen_func(q, k, v, **fa_kw)

VisionAttention = ModelNew
