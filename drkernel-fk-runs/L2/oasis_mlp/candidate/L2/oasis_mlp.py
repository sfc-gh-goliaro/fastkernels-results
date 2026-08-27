import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import Triton
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


# Fused kernel: Y = GELU_tanh( X @ W^T + b )
# X: [M, K], W: [N, K] (we index as W^T using strides), Bias: [N], Y: [M, N]
@triton.jit
def _matmul_bias_gelu_tanh_kernel(
    X, W, BIAS, Y,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,   # W strides: out_dim (n), in_dim (k)
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ids for tiling over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Set up pointers for the first k-block
    x_ptrs = X + (offs_m[:, None] * stride_xm) + (offs_k[None, :] * stride_xk)
    # W is [N, K]; we want B = W^T [K, N]; element (k, n) = W[n, k]
    w_ptrs = W + (offs_n[None, :] * stride_wn) + (offs_k[:, None] * stride_wk)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in blocks
    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        # Bounds masks
        k_mask_x = offs_k[None, :] < k_remaining
        k_mask_w = offs_k[:, None] < k_remaining
        m_mask = offs_m[:, None] < M
        n_mask = offs_n[None, :] < N

        # Load X tile [BM, BK] and W tile [BK, BN]; cast to fp32
        x = tl.load(x_ptrs, mask=m_mask & k_mask_x, other=0.0).to(tl.float32)
        w = tl.load(w_ptrs, mask=n_mask & k_mask_w, other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(x, w)

        # Advance pointers to next k-block
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Add bias: bias is [N]
    bias = tl.load(BIAS + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)  # [BN]
    acc = acc + bias[None, :]  # broadcast over M

    # GELU(tanh approximation): 0.5*x*(1 + tanh(√(2/π)*(x + 0.044715 x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = acc * acc * acc
    t = c * (acc + 0.044715 * x3)
    # tanh(t) = 2 / (1 + exp(-2t)) - 1
    e = tl.exp(-2.0 * t)
    tanh_t = 2.0 / (1.0 + e) - 1.0
    gelu = 0.5 * acc * (1.0 + tanh_t)

    # Store result (cast to fp16)
    y = gelu.to(tl.float16)
    y_ptrs = Y + (offs_m[:, None] * stride_ym) + (offs_n[None, :] * stride_yn)
    store_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, y, mask=store_mask)


def _triton_linear_gelu(x: torch.Tensor,
                        w: torch.Tensor,
                        bias: torch.Tensor | None,
                        use_tanh_approx: bool) -> torch.Tensor:
    """
    Fused Linear + GELU using Triton (tanh approximation path).
    x: [M, K], w: [N, K], bias: [N] or None
    returns y: [M, N]
    """
    assert use_tanh_approx, "This Triton kernel only implements tanh-approx GELU"
    assert x.is_cuda and w.is_cuda, "Triton kernel requires CUDA tensors"
    assert x.dtype == torch.float16 and w.dtype == torch.float16, "This kernel expects float16 inputs/weights"
    M, K = x.shape
    N = w.shape[0]
    assert w.shape[1] == K, f"Weight shape mismatch: w.shape={w.shape}, K={K}"

    # Make tensors contiguous for simpler, faster strides
    x_ = x.contiguous()
    w_ = w.contiguous()

    # Strides
    stride_xm, stride_xk = x_.stride(0), x_.stride(1)
    stride_wn, stride_wk = w_.stride(0), w_.stride(1)

    # Allocate output
    y = torch.empty((M, N), device=x.device, dtype=torch.float16)
    stride_ym, stride_yn = y.stride(0), y.stride(1)

    # Bias: if None, use zeros
    if bias is None:
        bias_t = torch.zeros((N,), device=x.device, dtype=torch.float16)
    else:
        bias_t = bias

    # Block sizes: good starting point for K=1024, N up to 1024–2048
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _matmul_bias_gelu_tanh_kernel[grid](
        x_, w_, bias_t, y,
        M, N, K,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_ym, stride_yn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    return y


class ModelNew(nn.Module):
    """
    Triton-optimized version:
    - Fuses first linear + bias + GELU into a single Triton kernel on CUDA (tanh approximation).
    - Falls back to PyTorch on CPU or if Triton is unavailable, and uses exact GELU to match 'none'.
    Same __init__ and forward signature as the original Model.
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        *,
        approximate_tanh: bool = False,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = Linear(in_features, hidden_features, bias=True)
        self.act = GELU(approximate="tanh" if approximate_tanh else "none")
        self.fc2 = Linear(hidden_features, out_features, bias=True)
        self.approximate_tanh = approximate_tanh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use Triton on CUDA with tanh approx; fallback otherwise with exact GELU
        if _HAS_TRITON and x.is_cuda and self.approximate_tanh:
            # Flatten to 2D: [M, K]
            K = x.shape[-1]
            M = x.numel() // K
            x_2d = x.reshape(M, K).contiguous()

            w1 = self.fc1.weight  # [H1, K]
            b1 = self.fc1.bias    # [H1]

            # Fused first layer (tanh-approx GELU inside kernel)
            y1_2d = _triton_linear_gelu(x_2d, w1, b1, use_tanh_approx=True)  # [M, H1]

            # Second layer: standard F.linear
            out = F.linear(y1_2d, self.fc2.weight, self.fc2.bias)
            return out.reshape(*x.shape[:-1], out.shape[-1])
        else:
            # Fallback: pure PyTorch, honor approximate flag
            return self.fc2(self.act(self.fc1(x)))


# Original component classes
class Matmul(nn.Module):
    def forward(self, input, weight, bias=None):
        return F.linear(input, weight, bias)

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.matmul = Matmul()

    def forward(self, input):
        return self.matmul(input, self.weight, self.bias)

class GELU(nn.Module):
    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate=self.approximate)

OasisMLP = ModelNew
