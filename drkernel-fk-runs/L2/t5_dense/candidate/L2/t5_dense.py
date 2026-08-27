import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


# -----------------------------
# Triton kernel: row-parallel linear + bias
# Computes: C[M, N] = A[M, K] @ B[K, N] + bias
# A: [M, K], contiguous last dim; B: [K, N], contiguous last dim
# -----------------------------
@triton.jit
def _matmul_rowparallel_bias_kernel(
    A, B, Bias, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(Bias + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# -----------------------------
# Triton kernel: column-parallel linear + bias
# Computes: C[M, Nout] = A[M, N] @ B[N, Nout] + bias
# A: [M, N]; B: [N, Nout]
# -----------------------------
@triton.jit
def _matmul_colparallel_bias_kernel(
    A, B, Bias, C,
    M, N, K,  # A:[M,N], B:[N,K], C:[M,K]
    stride_am, stride_an,
    stride_bk, stride_bn,
    stride_cm, stride_ck,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)

    a_ptrs = A + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_an)     # [BM, BN]
    b_ptrs = B + (offs_n[:, None] * stride_bk + offs_k[None, :] * stride_bn)     # [BN, BK]

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for n in range(0, N, BLOCK_N):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (n + offs_n[None, :] < N), other=0.0)
        b = tl.load(b_ptrs, mask=((n + offs_n)[:, None] < N) & (offs_k[None, :] < K), other=0.0)
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))
        a_ptrs += BLOCK_N * stride_an
        b_ptrs += BLOCK_N * stride_bk

    bias = tl.load(Bias + offs_k, mask=offs_k < K, other=0.0)  # [BK]
    acc = acc + bias[None, :]

    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_k[None, :] * stride_ck)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K))


def _triton_rowparallel_linear(a: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    a: [M, K], w: [N, K] (we will use w.T as B[K, N])
    returns c: [M, N] = a @ w.T + bias
    """
    assert a.is_cuda and w.is_cuda, "Triton kernels require CUDA tensors"
    M, K = a.shape
    N = w.shape[0]
    B = w.transpose(0, 1).contiguous()
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    stride_am = a.stride(0)
    stride_ak = a.stride(1)
    stride_bk = B.stride(0)
    stride_bn = B.stride(1)
    stride_cm = c.stride(0)
    stride_cn = c.stride(1)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_rowparallel_bias_kernel[grid](
        a, B, bias,
        c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )
    return c


def _triton_colparallel_linear(a: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    a: [M, N], w: [Nout, N] (we will use w.T as B[N, Nout])
    returns c: [M, Nout] = a @ w.T + bias
    """
    assert a.is_cuda and w.is_cuda, "Triton kernels require CUDA tensors"
    M, N = a.shape
    Nout = w.shape[0]
    B = w.transpose(0, 1).contiguous()  # [N, Nout]
    c = torch.empty((M, Nout), device=a.device, dtype=a.dtype)
    stride_am = a.stride(0)
    stride_an = a.stride(1)
    stride_bk = B.stride(0)  # N dim
    stride_bn = B.stride(1)  # Nout dim
    stride_cm = c.stride(0)
    stride_ck = c.stride(1)
    BLOCK_M = 128
    BLOCK_K = 128
    BLOCK_N = 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(Nout, BLOCK_K))
    _matmul_colparallel_bias_kernel[grid](
        a, B, bias,
        c,
        M, N, Nout,
        stride_am, stride_an,
        stride_bk, stride_bn,
        stride_cm, stride_ck,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=3,
    )
    return c


# -----------------------------
# Triton-optimized T5DenseGatedActDense
# -----------------------------
class _T5DenseGatedActDenseTriton(nn.Module):
    def __init__(self, d_model: int, d_ff: int, act_fn: str = "gelu", use_triton: bool = True):
        super().__init__()
        self.use_triton = use_triton and torch.cuda.is_available()
        # Weights as [out, in]
        self.wi = nn.Parameter(torch.empty(2 * d_ff, d_model))
        self.wo = nn.Parameter(torch.empty(d_model, d_ff))
        self.bias_i = nn.Parameter(torch.empty(2 * d_ff))
        self.bias_o = nn.Parameter(torch.empty(d_model))
        # Simple init
        bound = 1.0 / (d_model ** 0.5)
        with torch.no_grad():
            self.wi.uniform_(-bound, bound)
            self.wo.uniform_(-bound, bound)
            self.bias_i.uniform_(-bound, bound)
            self.bias_o.uniform_(-bound, bound)
        self.act_fn = act_fn  # kept for API symmetry

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape expected [M, K] = [*, d_model]
        if not self.use_triton:
            # Torch fallback
            gate_up = torch.nn.functional.linear(x, self.wi, self.bias_i)
            gate, up = gate_up.chunk(2, dim=-1)
            hidden = torch.nn.functional.gelu(gate) * up
            out = torch.nn.functional.linear(hidden, self.wo, self.bias_o)
            return out

        # 1) First GEMM + bias: gate_up = x @ wi^T + bias_i  -> [M, 2N]
        a1 = x
        wi_t = self.wi.transpose(0, 1).contiguous()  # [K, 2N]
        gate_up = _triton_rowparallel_linear(a1, self.wi, self.bias_i)  # [M, 2N]
        gate, up = gate_up.chunk(2, dim=-1)  # [M, N], [M, N]
        # 2) GELU on gate; elementwise in torch for parity
        gate = torch.nn.functional.gelu(gate)
        hidden = gate * up
        # 3) Second GEMM + bias: out = hidden @ wo^T + bias_o  -> [M, d_model]
        wo_t = self.wo.transpose(0, 1).contiguous()  # [N, Nout]
        out = _triton_colparallel_linear(hidden, self.wo, self.bias_o)  # [M, Nout = d_model]
        return out


# -----------------------------
# Entry points: Model and ModelNew
# Both accept a T5Config and use the Triton-optimized path.
# -----------------------------
class Model(_T5DenseGatedActDenseTriton):
    def __init__(self, config: T5Config, use_triton: bool = True):
        super().__init__(config.d_model, config.d_ff, act_fn=config.dense_act_fn, use_triton=use_triton)


class ModelNew(Model):
    pass

T5DenseGatedActDense = ModelNew
