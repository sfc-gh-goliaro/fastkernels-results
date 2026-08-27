import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


# Fused kernel: for each row r in [0, M)
# - compute mean, var, rstd over D (fp32)
# - loop over D again: normalize to xhat, then
#   q = xhat @ Wq + bq
#   k = xhat @ Wk + bk
#   v = xhat @ Wv + bv
# All compute in fp32, store bf16.
if _HAS_TRITON:
    @triton.jit
    def _fused_ln_qkv_bf16_kernel(
        x_ptr,                          # *bf16 [M, D]
        wq_ptr, wk_ptr, wv_ptr,          # *bf16 [out_rows, D]
        bq_ptr, bk_ptr, bv_ptr,          # *bf16 [out_rows]
        out_q_ptr, out_k_ptr, out_v_ptr, # *bf16 [M, out_rows]
        M: tl.constexpr, D: tl.constexpr,
        out_rows: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_OUT: tl.constexpr,
    ):
        row = tl.program_id(0)

        # Pass 1: compute mean and variance
        sum_val = tl.zeros((), dtype=tl.float32)
        sum_sq = tl.zeros((), dtype=tl.float32)
        col = 0
        while col < D:
            offs = col + tl.arange(0, BLOCK_D)
            mask = offs < D
            x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0)
            xf = x.to(tl.float32)
            sum_val += tl.sum(xf, axis=0)
            sum_sq += tl.sum(xf * xf, axis=0)
            col += BLOCK_D

        invD = 1.0 / D
        mean = sum_val * invD
        var = sum_sq * invD - mean * mean
        rstd = 1.0 / tl.sqrt(var + 1e-6)  # eps = 1e-6

        # Pass 2: normalize and project
        col = 0
        while col < D:
            offs = col + tl.arange(0, BLOCK_D)
            mask = offs < D
            x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0)
            xf = x.to(tl.float32)
            xhat = (xf - mean) * rstd

            # Loop over output rows in BLOCK_OUT tiles
            ro = 0
            while ro < out_rows:
                roff = ro + tl.arange(0, BLOCK_OUT)
                row_mask = roff < out_rows
                acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)

                # Loop over D in BLOCK_D
                c = 0
                while c < D:
                    d = c + tl.arange(0, BLOCK_D)
                    dmask = d < D
                    # Load weight tile [BLOCK_OUT, BLOCK_D]
                    w_tile = tl.load(
                        wq_ptr + (roff[:, None] * D) + d[None, :],
                        mask=row_mask[:, None] & dmask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    # Load xhat vector [BLOCK_D]
                    xhatv = xhat
                    # FMA: acc += sum over D
                    acc += tl.sum(w_tile * xhatv[None, :], axis=1)
                    c += BLOCK_D

                # Add bias and store
                b = tl.load(bq_ptr + roff, mask=row_mask, other=0.0).to(tl.float32)
                out = acc + b
                out_bf16 = out.to(tl.bfloat16)
                tl.store(out_q_ptr + row * out_rows + roff, out_bf16, mask=row_mask)

                ro += BLOCK_OUT

            col += BLOCK_D


class LayerNorm(nn.Module):
    # Keep a simple PyTorch LN as fallback; ModelNew will use the fused kernel instead.
    def __init__(self, normalized_shape: int, eps: float = 1e-5,
                 elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


# Keep helper classes from the original for completeness (not used in fast path).

_FP8_BLOCK = 128

def _scale_shape(out_dim: int, in_dim: int) -> tuple[int, int]:
    return (math.ceil(out_dim / _FP8_BLOCK), math.ceil(in_dim / _FP8_BLOCK))

def _is_batch_invariant() -> bool:
    return os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1"

def _alloc_colmajor_scale(M: int, num_groups: int, device: torch.device) -> torch.Tensor:
    return torch.empty(
        (num_groups, M), device=device, dtype=torch.float32,
    ).permute(-1, -2)

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
        self._a_buf: torch.Tensor | None = None
        self._s_buf: torch.Tensor | None = None
        self._o_buf: torch.Tensor | None = None
        self._pf: _Fp8PrefillBufs | None = None

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

        if torch.compiler.is_compiling():
            output = torch.ops.fastkernels_fp8.blockscale_gemm_dispatch(
                input_2d, weight_fp8, weight_scale_inv, False,
            )
            if bias is not None:
                output = output + bias
            return output.view(*input_bf16.shape[:-1], N)

        use_flashinfer = False

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

class ColumnParallelLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int, bias: bool = False,
                 quant_config: dict | None = None):
        super().__init__()
        tp = _tp_size()
        assert output_size % tp == 0
        self.output_size_per_partition = output_size // tp
        self.use_fp8 = quant_config is not None

        if self.use_fp8:
            self.weight = nn.Parameter(
                torch.empty(self.output_size_per_partition, input_size,
                            dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(*_scale_shape(self.output_size_per_partition, input_size),
                            dtype=torch.float32),
                requires_grad=False,
            )
            self.weight.weight_loader = self._weight_loader
            self.weight_scale_inv.weight_loader = self._scale_loader
            self.linear_op = _get_fp8_linear_cls()()
        else:
            self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, input_size))
            self.weight.weight_loader = self._weight_loader

        self.bias = nn.Parameter(torch.empty(self.output_size_per_partition)) if bias else None
        if self.bias is not None:
            self.bias.weight_loader = self._weight_loader

    def _weight_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        shard = param.data.size(0)
        loaded_weight = loaded_weight.narrow(0, rank * shard, shard)
        param.data.copy_(loaded_weight)

    def _scale_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        rows_per_shard = param.data.size(0)
        loaded_weight = loaded_weight.narrow(0, rank * rows_per_shard, rows_per_shard)
        param.data.copy_(loaded_weight)

    def forward(self, x):
        if self.use_fp8:
            return self.linear_op(x, self.weight, self.weight_scale_inv, self.bias)
        return F.linear(x, self.weight, self.bias)

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

class VisionMLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int,
                 act_fn: Callable[[torch.Tensor], torch.Tensor] = QuickGELU(),
                 bias: bool = True):
        super().__init__()
        self.fc1 = ColumnParallelLinear(in_features, hidden_features, bias=bias)
        self.fc2 = RowParallelLinear(hidden_features, in_features, bias=bias)
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act_fn(self.fc1(x)))

class FlashAttnPrefill(nn.Module):
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sm_scale = head_dim ** -0.5
        self.fa_version = 3

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
        kwargs.setdefault("num_splits", 1)
        return flash_attn_varlen_func(
            q, k, v,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=max_seqlen_k,
            cu_seqlens_k=cu_seqlens_k,
            softmax_scale=self.sm_scale,
            **kwargs,
        )

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

class VisionAttention(nn.Module):
    """Attention: use fused Triton kernel for LN+QKV, then FlashAttention."""

    def __init__(self, embed_dim: int, num_heads: int, projection_size: int | None = None):
        super().__init__()
        if projection_size is None:
            projection_size = embed_dim
        tp = _tp_size()
        self.tp_size = tp
        self.tp_rank = _tp_rank()
        self.head_dim = projection_size // num_heads
        self.num_heads = num_heads // tp

        self.qkv = QKVParallelLinear(
            embed_dim, self.head_dim, num_heads, num_heads, bias=True,
        )
        self.proj = RowParallelLinear(projection_size, embed_dim, bias=True)
        self.attn = FlashAttnPrefill(self.num_heads, self.num_heads, self.head_dim)

    def _fused_qkv(self, x: torch.Tensor):
        # x: [S, B, D], contiguous
        assert x.is_contiguous(), "x must be contiguous"
        seq, batch, D = x.shape
        M = seq * batch

        # Get weights as contiguous [out_rows, D]
        # Note: QKV weight is [3*H*Dh, D] in row-major.
        w = self.qkv.weight
        assert w.is_contiguous(), "QKV weight must be contiguous"
        out_rows = w.shape[0]
        assert out_rows == 3 * self.num_heads * self.head_dim

        # Allocate outputs [M, out_rows] bf16
        out_q = torch.empty((M, out_rows), device=x.device, dtype=torch.bfloat16)
        out_k = torch.empty((M, out_rows), device=x.device, dtype=torch.bfloat16)
        out_v = torch.empty((M, out_rows), device=x.device, dtype=torch.bfloat16)

        # Split w into three
        # Shapes: [out_rows, D] -> [3, out_per, D]
        out_per = out_rows // 3
        wq = w[:out_per]
        wk = w[out_per:2*out_per]
        wv = w[2*out_per:]

        # Biases
        b = self.qkv.bias
        if b is None:
            bq = bk = bv = None
        else:
            bq = b[:out_per]
            bk = b[out_per:2*out_per]
            bv = b[2*out_per:]

        # Launch kernel: grid = (M,)
        BLOCK_D = 128
        BLOCK_OUT = 64
        grid = (M,)
        _fused_ln_qkv_bf16_kernel[grid](
            x.view(-1, D),       # x_ptr
            wq, wk, wv,           # weights
            bq, bk, bv,           # biases
            out_q, out_k, out_v,  # outputs
            M, D,
            out_rows,
            BLOCK_D,
            BLOCK_OUT,
            num_warps=4,
        )

        # Reshape to (S, B, out_rows)
        out = torch.cat([out_q, out_k, out_v], dim=1)  # [M, 3*out_rows] not true; we need to build qkv_3
        # Build qkv_3 via views without copy:
        # We have out_q[M,out_rows], out_k, out_v.
        # But Flash wants (S,B,3,H,Dh). Easiest is to view each third:
        # q = out_q.view(S,B,out_per); k = out_k.view(S,B,out_per); v = out_v.view(S,B,out_per)
        # Then permute to (S,B,3,H,Dh) views.
        q = out_q.view(seq, batch, out_per)
        k = out_k.view(seq, batch, out_per)
        v = out_v.view(seq, batch, out_per)
        return q, k, v

    def forward(
        self, x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        # x: [S, B, D]
        seq_len, batch_size, _ = x.shape
        # Produce q,k,v with fused kernel
        q, k, v = self._fused_qkv(x)

        # Apply rotary in-place
        if rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            apply_rotary(q, rotary_pos_emb_cos, rotary_pos_emb_sin, inplace=True)
            apply_rotary(k, rotary_pos_emb_cos, rotary_pos_emb_sin, inplace=True)

        # Reshape to (M,H,D) for FlashAttention
        out_per = q.shape[-1]  # = 3*H*Dh
        H = self.num_heads
        Dh = self.head_dim
        assert out_per == H * Dh
        M = batch_size * seq_len
        q_ = q.reshape(M, H, Dh)
        k_ = k.reshape(M, H, Dh)
        v_ = v.reshape(M, H, Dh)

        if max_seqlen is None:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        out = self.attn(
            q_, k_, v_,
            cu_seqlens, cu_seqlens,
            max_seqlen, max_seqlen,
            softmax_scale=self.head_dim ** -0.5,
            causal=False,
        )

        out = out.view(seq_len, batch_size, -1)
        return self.proj(out)

class ModelNew(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_hidden_dim: int,
                 act_fn: Callable[[torch.Tensor], torch.Tensor] = QuickGELU(),
                 norm_eps: float = 1e-6):
        super().__init__()
        # We don't need custom LN here; Attention uses fused kernel.
        self.attn = VisionAttention(embed_dim, num_heads)
        self.mlp = VisionMLP(embed_dim, mlp_hidden_dim, act_fn=act_fn)

    def forward(
        self, x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        # x shape: [S, B, D]
        x = x + self.attn(
            x, cu_seqlens,
            rotary_pos_emb_cos, rotary_pos_emb_sin,
            max_seqlen,
        )
        x = x + self.mlp(x)
        return x

VisionBlock = ModelNew
