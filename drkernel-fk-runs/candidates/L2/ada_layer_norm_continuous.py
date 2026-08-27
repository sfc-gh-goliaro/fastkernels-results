import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _layernorm_two_affine_kernel(
    X_ptr,                 # *x, shape [B*L, D]
    Y_ptr,                 # *y, shape [B*L, D]
    S0_ptr, B0_ptr,        # *s0, *b0, shape [D]
    S1_ptr, B1_ptr,        # *s1, *b1, shape [D]
    D: tl.constexpr,       # feature dim
    eps,                   # epsilon
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_start = row * D

    # Accumulate sum and sum of squares in fp32
    sum_x = tl.zeros((), dtype=tl.float32)
    sum_x2 = tl.zeros((), dtype=tl.float32)

    for col in range(0, D, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        x = tl.load(X_ptr + row_start + offs, mask=mask, other=0.0)
        xf = x.to(tl.float32)
        sum_x += tl.sum(xf, axis=0)
        sum_x2 += tl.sum(xf * xf, axis=0)

    mean = sum_x / D
    var = sum_x2 / D - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize + two elementwise affines, store
    for col in range(0, D, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < D

        x = tl.load(X_ptr + row_start + offs, mask=mask, other=0.0).to(tl.float32)
        norm = (x - mean) * rstd

        # load per-feature params (fp32)
        s0 = tl.load(S0_ptr + offs, mask=mask, other=0.0)
        b0 = tl.load(B0_ptr + offs, mask=mask, other=0.0)
        s1 = tl.load(S1_ptr + offs, mask=mask, other=0.0)
        b1 = tl.load(B1_ptr + offs, mask=mask, other=0.0)

        y = norm * (1.0 + s0) + b0
        y = y * (1.0 + s1) + b1

        tl.store(Y_ptr + row_start + offs, y.to(x.dtype), mask=mask)


def _triton_layernorm_two_affine_per_batch(
    x: torch.Tensor,
    s0: torch.Tensor, b0: torch.Tensor,
    s1: torch.Tensor, b1: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    x: [B, L, D], CUDA tensor
    s0,b0,s1,b1: list of B tensors, each shape [D], on device
    Returns y with same shape/dtype as x.
    """
    assert x.is_cuda, "Triton kernel requires CUDA tensor"
    assert x.ndim == 3, f"Expected x.ndim=3, got {x.ndim}"
    B, L, D = x.shape

    x = x.contiguous()
    y = torch.empty_like(x)

    # Launch grid over all rows (b, l)
    grid = (B * L,)

    block_size = 1024
    num_warps = 4

    for b in range(B):
        _layernorm_two_affine_kernel[grid](
            x.view(-1, D),            # X
            y.view(-1, D),            # Y
            s0[b], b0[b],             # S0,B0
            s1[b], b1[b],             # S1,B1
            D,
            eps,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=2,
        )
    return y


class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(x)


class Matmul(nn.Module):
    def forward(self, input, weight, bias=None):
        return torch.nn.functional.linear(input, weight, bias)


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.matmul = Matmul()
    def forward(self, input):
        return self.matmul(input, self.weight, self.bias)


class ModelNew(nn.Module):
    r"""
    Triton-optimized version of Adaptive LayerNormContinuous with signature-compatible constructor.

    Args (must match original Model):
        embedding_dim (`int`): D.
        conditioning_embedding_dim (`int`): C.
        elementwise_affine (`bool`): ignored (no LN affine).
        eps (`float`): epsilon for LN.
        bias (`bool`): ignored.
        norm_type (`str`): must be "layer_norm".
        promote_fp32 (`bool`): ignored.
    """
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
        promote_fp32=True,
    ):
        super().__init__()
        self.silu = SiLU()
        self.linear = Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        if norm_type != "layer_norm":
            raise ValueError(f"unknown norm_type {norm_type}")
        self.eps = eps
        self.D = embedding_dim

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        # 1) modulation
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))  # [B, 2D]
        B, _ = emb.shape
        D = self.D
        assert emb.shape[-1] == 2 * D, f"Expected 2D={2*D}, got {emb.shape[-1]}"

        # Split into two halves along last dim -> [B, D]
        e0 = emb[:, :D]          # [B, D]
        e1 = emb[:, D:]          # [B, D]

        # Compute s0,b0,s1,b1 from e0,e1
        # e0, e1 are [B, D]; stack along last dim then view into (B, 2, D)
        e = torch.stack([e0, e1], dim=1)  # [B, 2, D]
        # s =sigmoid(e); b=e  ->但原文是 scale = e, shift = e
        #原文：split emb into scale, shift; it's raw linear output, not activated again.
        # so scale = e0, shift = e1
        # But wait: Model does silu(c) -> linear -> emb; then split emb into scale,shift.
        # So scale,shift are raw linear outputs (can be negative); not silu.
        # Consistency: we must match original: scale,shift are exactly the two halves of emb.
        # Apply the two transforms:
        # y = LN(x) * (1 + scale0) + shift0
        # y = y  * (1 + scale1) + shift1
        #所以 s0=e0, b0=e1的第一半误解。正确：s0是e0, b0是e1。
        #但是为了 clarity, 我们直接使用 e0,e1 分别作为 scale0,shift0 和 scale1,shift1 的组成部分
        #但是根据原文，split emb into scale, shift; then y = LN * (1+s0)+b0; y = y * (1+s1)+b1
        #所以 s0 = e0, b0 = e1 第一段？不， e0 is scale0, e1 is shift0? No: it splits emb into scale,shift -> so scale = first half, shift = second half.
        #所以 scale0 = e0; shift0 = e1; then scale1,shift1 do not exist. Wait: read again.
        # Original:
        # emb = Linear( SiLU(c) )
        # scale, shift = torch.chunk(emb, 2, dim=1)  -> each [B, D]
        # x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        # So: scale = e0; shift = e1; that's it. There is no second pair.
        # My previous over-reading added a non-existent second pair. Fix:
        # Only one pair (scale,shift). So only one affine after LN.

        # Extract scale (s0) and shift (b0)
        # e0 is scale; e1 is shift
        s0 = e[:, 0, :]  # [B, D]
        b0 = e[:, 1, :]  # [B, D]

        if not x.is_cuda:
            # CPU fallback: use PyTorch LayerNorm + one elementwise op
            y = torch.nn.functional.layer_norm(x, (self.D,), weight=None, bias=None, eps=self.eps)
            y = y * (1.0 + s0) + b0
            return y

        # 2) Triton: fused layer norm + one elementwise affine
        B, L, D = x.shape
        x_c = x.contiguous()
        y = torch.empty_like(x_c)

        # Move mods to device and float32
        device = x.device
        s0_ = s0.to(device=device, dtype=torch.float32).contiguous()
        b0_ = b0.to(device=device, dtype=torch.float32).contiguous()

        grid = (B * L,)

        block_size = 1024
        num_warps = 4

        # Simplify kernel: only one affine (s0,b0). Pass s1=b1=zeros to satisfy signature.
        s1_ = torch.zeros_like(s0_)
        b1_ = torch.zeros_like(b0_)

        _layernorm_two_affine_kernel[grid](
            x_c.view(-1, D),                 # X
            y.view(-1, D),                   # Y
            s0_, b0_,                        # S0,B0
            s1_, b1_,                        # S1,B1 (zeros)
            D,
            self.eps,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=2,
        )
        return y

AdaLayerNormContinuous = ModelNew
