import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

# Fused layer norm (no affine) + affine: out = ((x - mean)/sqrt(var+eps)) * weight
# Computes in fp32, stores in x.dtype. Assumes x is 1D contiguous of length N.
@triton.jit
def _lnorm_affine_kernel(x_ptr, weight_ptr, out_ptr,
                         N, D,
                         eps,
                         BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.float32)

    # mean
    mean = tl.sum(x, axis=0) / D
    x_centered = x - mean

    # var (population)
    var = tl.sum(x_centered * x_centered, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)

    # normalize
    y = x_centered * rstd  # fp32

    # affine: y * weight
    w = tl.load(weight_ptr + offs, mask=mask, other=1).to(tl.float32)
    out = y * w

    tl.store(out_ptr + offs, out, mask=mask)


# Fused sigmoid + multiply: out = sigmoid(x) * y
# Computes in fp32, stores in x.dtype. Assumes 1D contiguous.
@triton.jit
def _sigmoid_mul_kernel(x_ptr, y_ptr, out_ptr,
                        N,
                        BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.float32)
    y = tl.load(y_ptr + offs, mask=mask, other=0).to(tl.float32)

    sig = 1.0 / (1.0 + tl.exp(-x))
    out = sig * y

    tl.store(out_ptr + offs, out, mask=mask)


# Original SwiGLU model: out = SiLU(x @ W_a^T) * (x @ W_b^T)
class Model(nn.Module):
    """SwiGLU activation: SiLU(Wa x) * Wb x.

    Args:
        c_in: Number of input channels
        c_out: Number of output channels
    """

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear_a = nn.Linear(c_in, c_out, bias=False)
        self.linear_b = nn.Linear(c_in, c_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CPU fallback: pure PyTorch
        if not x.is_cuda:
            ya = self.linear_a(x)
            yb = self.linear_b(x)
            return self.silu(ya) * yb

        # 1) GEMMs on cuBLAS
        ya = torch.nn.functional.linear(x, self.linear_a.weight, bias=None)
        yb = torch.nn.functional.linear(x, self.linear_b.weight, bias=None)

        # 2) Fused Triton kernel: out = silu(ya) * yb
        ya32 = ya.float().contiguous()
        yb32 = yb.float().contiguous()
        N = ya32.numel()
        out32 = torch.empty_like(ya32)

        BLOCK = 4096
        grid = (triton.cdiv(N, BLOCK),)

        _sigmoid_mul_kernel[grid](
            ya32.view(-1), yb32.view(-1), out32.view(-1),
            N,
            BLOCK_SIZE=BLOCK,
            num_warps=4,
        )

        # silu = x * sigmoid(x); we computed sigmoid(ya) * yb.
        # To get silu(ya) * yb, multiply out by ya: out = (sigmoid(ya)*yb) * ya = silu(ya)*yb
        # Wait, no: we want silu(ya)*yb = (ya*sigmoid(ya))*yb. What we computed is sigmoid(ya)*yb.
        # That's incorrect. Fix: compute silu first: silu = ya * sigmoid(ya), then multiply by yb.
        # Redefine kernel or do it here in PyTorch for safety:
        # Better: redefine kernel to compute silu.

@triton.jit
def _silu_mul_kernel(a_ptr, b_ptr, out_ptr,
                     N,
                     BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    a = tl.load(a_ptr + offs, mask=mask, other=0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0).to(tl.float32)

    sig = 1.0 / (1.0 + tl.exp(-a))
    silu = a * sig
    out = silu * b

    tl.store(out_ptr + offs, out, mask=mask)

# Replace previous call
        out32 = torch.empty_like(ya32)
        _silu_mul_kernel[grid](
            ya32.view(-1), yb32.view(-1), out32.view(-1),
            N,
            BLOCK_SIZE=BLOCK,
            num_warps=4,
        )
        return out32.to(ya.dtype)

# AdaLN model: out = g * (norm_a(a) + linear_s(norm_s(s)))
class Model(nn.Module):
    """
    AdaLN:
      out = g * (norm_a(a) + linear_s(norm_s(s)))
    where
      norm_a: LayerNorm over a's last dim, no affine
      norm_s: LayerNorm over s's last dim, elementwise_affine (weight only)
      linear_g: Linear(c_s, c_a, bias=True) -> g = sigmoid(linear_g(norm_s(s)))
      linear_s: Linear(c_s, c_a, bias=False) -> t = linear_s(norm_s(s))
    We keep GEMMs on cuBLAS and fuse elementwise sequences with Triton.
    """

    def __init__(self, c_a: int, c_s: int):
        super().__init__()
        self.c_a = c_a
        self.c_s = c_s

        # Match the original modules
        self.layer_norm_a = LayerNorm(c_a, create_scale=False, create_offset=False)
        self.layer_norm_s = LayerNorm(c_s, create_offset=False)
        self.linear_g = Linear(c_s, c_a, bias=True)
        self.linear_s = Linear(c_s, c_a, bias=False)

    def _fused_layernorm_affine(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        """
        Fused: out = ((x - mean)/sqrt(var+eps)) * weight
        - x: arbitrary shape, contiguous
        - weight: 1D, length = x.shape[-1]
        Returns tensor same shape/dtype as x.
        """
        assert x.is_cuda, "Triton kernel requires CUDA tensor"
        x_c = x.contiguous()
        weight_c = weight.contiguous()
        # Flatten to 1D
        N = x_c.numel()
        D = x_c.shape[-1]
        x1 = x_c.view(-1)
        out1 = torch.empty_like(x1)

        BLOCK = 4096
        grid = (triton.cdiv(N, BLOCK),)

        _lnorm_affine_kernel[grid](
            x1, weight_c.float(), out1,
            N, D,
            eps,
            BLOCK_SIZE=BLOCK,
            num_warps=4,
        )
        return out1.view_as(x_c)

    def _fused_sigmoid_mul(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Fused: out = sigmoid(x) * y
        Shapes must match, CUDA tensors.
        """
        assert x.is_cuda and y.is_cuda, "Triton kernel requires CUDA tensors"
        x_c = x.contiguous()
        y_c = y.contiguous()
        N = x_c.numel()
        out = torch.empty_like(x_c.view(-1))

        BLOCK = 4096
        grid = (triton.cdiv(N, BLOCK),)

        _sigmoid_mul_kernel[grid](
            x_c.view(-1), y_c.view(-1), out.view(-1),
            N,
            BLOCK_SIZE=BLOCK,
            num_warps=4,
        )
        return out.view_as(x_c)

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Compute:
          s_norm = layer_norm_s(s)
          t = linear_s(s_norm)         # [*, c_a]
          g_pre = linear_g(s_norm)     # [*, c_a]
          g = sigmoid(g_pre + bias_g)
          a_norm = layer_norm_a(a)     # [*, c_a]
          out = g * (a_norm + t)
        """
        # CPU fallback: pure PyTorch
        if not a.is_cuda or not s.is_cuda:
            s_norm = self.layer_norm_s(s)
            t = self.linear_s(s_norm)
            g_pre = self.linear_g(s_norm)
            g = torch.sigmoid(g_pre)
            a_norm = self.layer_norm_a(a)
            return g * (a_norm + t)

        # 1) s_norm = layer_norm_s(s) with weight affine
        s_contig = s.contiguous()
        Ds = s_contig.shape[-1]
        weight_s = self.layer_norm_s.weight  # 1D
        eps_s = self.layer_norm_s.eps
        s_norm = self._fused_layernorm_affine(s_contig, weight_s, eps_s)

        # 2) t = linear_s(s_norm)
        t = torch.nn.functional.linear(s_norm, self.linear_s.weight, bias=None)

        # 3) g_pre = linear_g(s_norm)  -> then add bias and sigmoid
        g_pre = torch.nn.functional.linear(s_norm, self.linear_g.weight, bias=None)
        bias_g = self.linear_g.bias
        g = torch.sigmoid(g_pre + bias_g)

        # 4) a_norm = layer_norm_a(a)  (no affine): use PyTorch for numerical parity
        a_contig = a.contiguous()
        Da = a_contig.shape[-1]
        a_norm = torch.nn.functional.layer_norm(a_contig, (Da,), eps=self.layer_norm_a.eps)

        # 5) out = g * (a_norm + t)
        return g * (a_norm + t)

AdaLN = ModelNew
SwiGLU = ModelNew
