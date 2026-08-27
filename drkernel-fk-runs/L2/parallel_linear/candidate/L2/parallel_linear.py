import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _matmul_bias_kernel(
    X, W, BIAS, Y,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    HAS_BIAS,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for the first K-block
    x_ptrs = X + (offs_m[:, None] * stride_xm) + (offs_k[None, :] * stride_xk)       # [BM, BK]
    # W is [N, K]; we index as W^T[k, n] = W[n, k]
    w_ptrs = W + (offs_n[None, :] * stride_wn) + (offs_k[:, None] * stride_wk)       # [BK, BN]

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in blocks
    for k in range(0, K, BLOCK_K):
        k_mask = (k + offs_k) < K
        a = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < M) & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            w_ptrs,
            mask=k_mask[:, None] & (offs_n[None, :] < N),
            other=0.0,
        )
        # Cast to fp32 for dot
        a = a.to(tl.float32)
        b = b.to(tl.float32)
        acc += tl.dot(a, b)  # [BM, BK] @ [BK, BN] -> [BM, BN]

        # Advance pointers
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Add bias if present
    if HAS_BIAS:
        bias = tl.load(BIAS + offs_n, mask=(offs_n < N), other=0.0)
        acc = acc + bias[None, :]

    # Store result
    y_ptrs = Y + (offs_m[:, None] * stride_ym) + (offs_n[None, :] * stride_yn)
    tl.store(
        y_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _triton_linear(x: torch.Tensor,
                   w: torch.Tensor,
                   bias: torch.Tensor | None) -> torch.Tensor:
    """
    Compute y = x @ w.T + bias using a Triton kernel.
    Shapes:
      x: [M, K]
      w: [N, K]
      y: [M, N]
    Dtypes:
      x, w: float16 | bfloat16
      accumulate: float32
      output: same dtype as x
    """
    assert x.is_cuda and w.is_cuda, "Triton kernel requires CUDA tensors"
    assert x.dim() == 2 and w.dim() == 2, "Expect 2D tensors"
    M, Kx = x.shape
    Nw, Kw = w.shape
    assert Kx == Kw, f"Incompatible shapes: x.shape={x.shape}, w.shape={w.shape}"
    K = Kx
    N = Nw

    # Output (float32 for accumulation; cast later)
    y = torch.empty((M, N), device=x.device, dtype=torch.float32)

    # Strides (elements)
    stride_xm, stride_xk = x.stride(0), x.stride(1)
    stride_wn, stride_wk = w.stride(0), w.stride(1)
    stride_ym, stride_yn = y.stride(0), y.stride(1)

    # Tile sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    has_bias = 1 if (bias is not None) else 0

    _matmul_bias_kernel[grid](
        x, w, bias if bias is not None else x,  # dummy pointer if no bias (won't be read)
        M, N, K,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_ym, stride_yn,
        has_bias,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )

    # Cast to input dtype to mimic F.linear behavior
    if y.dtype != x.dtype:
        y = y.to(x.dtype)
    return y


# --- Modules: Triton-optimized forward, API-compatible constructors ---

class Model(nn.Module):
    def __init__(self, input_size: int, output_size: int, bias: bool = False,
                 quant_config: dict | None = None, *args, **kwargs):
        super().__init__()
        # Keep FP8 attribute but we'll use non-FP8 path
        self.use_fp8 = quant_config is not None
        if self.use_fp8:
            self.weight = nn.Parameter(
                torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(math.ceil(output_size / 128), math.ceil(input_size / 128),
                            dtype=torch.float32),
                requires_grad=False,
            )
        else:
            self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        b = self.bias
        # CPU fallback
        if not x.is_cuda or not w.is_cuda:
            return torch.nn.functional.linear(x, w, b)
        # Use Triton
        return _triton_linear(x, w, b)


class MergedColumnParallelLinear(nn.Module):
    def __init__(self, input_size: int, output_sizes: list[int], bias: bool = False,
                 quant_config: dict | None = None, disable_tp: bool = False,
                 *args, **kwargs):
        super().__init__()
        self.output_sizes = output_sizes
        total = sum(output_sizes)
        self.use_fp8 = quant_config is not None
        if self.use_fp8:
            self.weight = nn.Parameter(
                torch.empty(total, input_size, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(math.ceil(total / 128), math.ceil(input_size / 128),
                            dtype=torch.float32),
                requires_grad=False,
            )
        else:
            self.weight = nn.Parameter(torch.empty(total, input_size))
        self.bias = nn.Parameter(torch.empty(total)) if bias else None
        self.disable_tp = disable_tp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        b = self.bias
        if not x.is_cuda or not w.is_cuda:
            return torch.nn.functional.linear(x, w, b)
        return _triton_linear(x, w, b)


class QKVParallelLinear(nn.Module):
    def __init__(self, hidden_size: int, head_size: int,
                 total_num_heads: int, total_num_kv_heads: int,
                 bias: bool = False, quant_config: dict | None = None,
                 *args, **kwargs):
        super().__init__()
        self.head_size = head_size
        self.num_heads = total_num_heads
        self.num_kv_heads = total_num_kv_heads
        output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size
        self.use_fp8 = quant_config is not None
        if self.use_fp8:
            self.weight = nn.Parameter(
                torch.empty(output_size, hidden_size, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(math.ceil(output_size / 128), math.ceil(hidden_size / 128),
                            dtype=torch.float32),
                requires_grad=False,
            )
        else:
            self.weight = nn.Parameter(torch.empty(output_size, hidden_size))
        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        b = self.bias
        if not x.is_cuda or not w.is_cuda:
            return torch.nn.functional.linear(x, w, b)
        return _triton_linear(x, w, b)


class ReplicatedLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int, bias: bool = True,
                 quant_config: dict | None = None, *args, **kwargs):
        super().__init__()
        self.use_fp8 = quant_config is not None
        if self.use_fp8:
            self.weight = nn.Parameter(
                torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(math.ceil(output_size / 128), math.ceil(input_size / 128),
                            dtype=torch.float32),
                requires_grad=False,
            )
        else:
            self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        b = self.bias
        if not x.is_cuda or not w.is_cuda:
            return torch.nn.functional.linear(x, w, b)
        return _triton_linear(x, w, b)


class RowParallelLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int, bias: bool = False,
                 quant_config: dict | None = None, reduce_results: bool = True,
                 *args, **kwargs):
        super().__init__()
        self.tp_size = 1  # assume single device; real TP would shard
        self.reduce_results = reduce_results
        self.use_fp8 = quant_config is not None
        if self.use_fp8:
            self.weight = nn.Parameter(
                torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(math.ceil(output_size / 128), math.ceil(input_size / 128),
                            dtype=torch.float32),
                requires_grad=False,
            )
        else:
            self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None
        # Dummy all-reduce to mirror API (not used in this single-device version)
        self.allreduce = nn.Module()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        b = self.bias
        if not x.is_cuda or not w.is_cuda:
            y = torch.nn.functional.linear(x, w, b)
            return y
        y = _triton_linear(x, w, b)
        if self.reduce_results:
            # no-op in this single-device implementation
            pass
        return y


# Entry point: ModelNew
class ModelNew(Model):
    pass

ColumnParallelLinear = ModelNew
MergedColumnParallelLinear = ModelNew
QKVParallelLinear = ModelNew
ReplicatedLinear = ModelNew
RowParallelLinear = ModelNew
