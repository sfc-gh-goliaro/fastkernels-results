import torch
import torch.nn as nn
import triton
import triton.language as tl


def _torch_dtype_to_triton(dtype: torch.dtype):
    if dtype == torch.float32:
        return tl.float32
    if dtype == torch.bfloat16:
        return tl.bfloat16
    return tl.float32


@triton.jit
def _gemm_2d_kernel(
    A, B, Bias, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    # program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # pointers for the first K-block
    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K dimension
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        a = a.to(tl.float32)
        b = b.to(tl.float32)
        acc += tl.dot(a, b)
        # advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # add bias if present: shape [N], broadcast over M
    if HAS_BIAS:
        bias = tl.load(Bias + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]

    # store result
    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    out = acc.to(OUT_DTYPE)
    tl.store(c_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _gemm_3d_kernel(
    A, B, Bias, C,
    Batches, M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    # 3D launch: (b, m, n)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # base pointers for this batch
    a_batch = A + pid_b * stride_ab
    b_batch = B + pid_b * stride_bb
    c_batch = C + pid_b * stride_cb

    # K-loop tile pointers
    a_ptrs = a_batch + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_batch + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        a = a.to(tl.float32)
        b = b.to(tl.float32)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(Bias + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]

    c_ptrs = c_batch + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    out = acc.to(OUT_DTYPE)
    tl.store(c_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class ModelNew(nn.Module):
    """
    Triton-optimized replacement for the original Model.
    Signature: __init__(in_features, out_features, bias=True)
    forward(input) -> tensor, equivalent to F.linear(input, weight, bias) for 2D/3D cases.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()  # MUST come first
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.bias_enabled = bool(bias)
        # Create parameters after super().__init__()
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.empty(self.out_features)) if self.bias_enabled else None

        # Default tiling and launch parameters
        self.num_warps = 4
        self.num_stages = 2

    def _choose_blocks(self, M: int, N: int, K: int):
        # Simple heuristic for block sizes
        BM = 128 if M >= 128 else 64
        BN = 128 if N >= 128 else 64
        BK = 64 if K >= 2048 else 32
        return BM, BN, BK

    def _2d_gemm_triton(self, x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
        """
        Compute x @ w.T + bias using Triton, where:
          x: [M, K]
          w: [N, K] (note: not transposed; we index as W[n, k] but treat as B[k, n])
          bias: [N] or None
        Output: [M, N]
        """
        assert x.is_cuda and w.is_cuda, "Triton kernels require CUDA tensors"
        assert x.dtype in (torch.float32, torch.bfloat16), "Supported dtypes: float32, bfloat16"
        assert w.dtype == x.dtype, "x and w must have same dtype"

        M, K = x.shape
        N = w.shape[0]
        assert w.shape[1] == K, f"weight shape mismatch: got w.shape={w.shape}, expected (*, {K})"

        # allocate output
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        # strides (element strides)
        stride_am = x.stride(0)
        stride_ak = x.stride(1)
        # treat w as B with shape (K, N), strides:
        # w[n, k] -> b[k, n]; so stride_bk = w.stride(1), stride_bn = w.stride(0)
        stride_bk = w.stride(1)
        stride_bn = w.stride(0)
        stride_cm = out.stride(0)
        stride_cn = out.stride(1)

        has_bias = (bias is not None)
        bias_ptr = bias if has_bias else out  # dummy, not used if False

        BM, BN, BK = self._choose_blocks(M, N, K)
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))

        _gemm_2d_kernel[grid](
            x, w, bias_ptr, out,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            HAS_BIAS=has_bias,
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
            OUT_DTYPE=_torch_dtype_to_triton(out.dtype),
            num_warps=self.num_warps, num_stages=self.num_stages,
        )
        return out

    def _3d_gemm_triton(self, a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
        """
        Compute a @ b where:
          a: [B, M, K]
          b: [B, K, N]  (here b is weight with shape [N, K]; we index as (k, n)->b[n, k])
        Using Triton; output [B, M, N].
        """
        assert a.is_cuda and b.is_cuda, "Triton kernels require CUDA tensors"
        assert a.dtype in (torch.float32, torch.bfloat16), "Supported dtypes: float32, bfloat16"
        assert b.dtype == a.dtype, "a and b must have same dtype"

        B = a.shape[0]
        M, K = a.shape[1], a.shape[2]
        assert b.shape[0] == B, f"batch mismatch: a.shape[0]={B}, b.shape[0]={b.shape[0]}"
        assert b.shape[1] == K, f"inner dim mismatch: a*K={K}, b*K={b.shape[1]}"
        N = b.shape[2]

        out = torch.empty((B, M, N), device=a.device, dtype=a.dtype)

        # strides
        stride_ab, stride_am, stride_ak = a.stride(0), a.stride(1), a.stride(2)
        # b as (K, N): stride_bk = b.stride(1), stride_bn = b.stride(0)
        stride_bb, stride_bk, stride_bn = b.stride(0), b.stride(1), b.stride(0)
        stride_cb, stride_cm, stride_cn = out.stride(0), out.stride(1), out.stride(2)

        has_bias = (bias is not None)
        bias_ptr = bias if has_bias else out  # dummy

        BM, BN, BK = self._choose_blocks(M, N, K)
        grid = (B, triton.cdiv(M, BM), triton.cdiv(N, BN))

        _gemm_3d_kernel[grid](
            a, b, bias_ptr, out,
            B, M, N, K,
            stride_ab, stride_am, stride_ak,
            stride_bb, stride_bk, stride_bn,
            stride_cb, stride_cm, stride_cn,
            HAS_BIAS=has_bias,
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
            OUT_DTYPE=_torch_dtype_to_triton(out.dtype),
            num_warps=self.num_warps, num_stages=self.num_stages,
        )
        return out

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Entry point: same signature as original Model.forward(input).
        - If input is CUDA and 2D, compute x @ W^T + bias via Triton 2D kernel.
        - If input is CUDA and 3D, treat it as batched and use Triton 3D kernel.
        - Otherwise, fallback to torch operations.
        """
        # Handle 2D input: shape [M, K]
        if input.ndim == 2:
            if input.is_cuda and self.weight.is_cuda:
                return self._2d_gemm_triton(input, self.weight, self.bias)
            else:
                return torch.nn.functional.linear(input, self.weight, self.bias)

        # Handle 3D input: shape [B, M, K] with weight [N, K]
        if input.ndim == 3:
            if input.is_cuda and self.weight.is_cuda:
                return self._3d_gemm_triton(input, self.weight, self.bias)
            else:
                B, M, K = input.shape
                N = self.weight.shape[0]
                # torch path: use bmm after transpose expand
                wT = self.weight.t().unsqueeze(0).expand(B, -1, -1)  # [B, K, N]
                out = torch.bmm(input, wT)  # [B, M, N]
                if self.bias is not None:
                    out = out + self.bias  # broadcast
                return out

        # Fallback
        return torch.nn.functional.linear(input, self.weight, self.bias)

BMM = ModelNew
Linear = ModelNew
Matmul = ModelNew
