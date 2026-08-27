import math
import torch
import torch.nn as nn

# Keep definitions for completeness.
class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        groups: int = 1,
        dilation: int | tuple[int, int] = 1,
        bias: bool = True,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

class BatchNorm2d(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.track_running_stats and self.num_batches_tracked is not None:
            self.num_batches_tracked.add_(1)
        return torch.nn.functional.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            self.momentum,
            self.eps,
        )

class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(x)

class Softmax(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softmax(x, dim=self.dim)

class YOLOConv(nn.Module):
    default_act = SiLU()

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p=None,
        g: int = 1,
        d: int = 1,
        act=True,
    ):
        super().__init__()
        self.conv = Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self._is_fused = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_fused:
            return self.act(self.conv(x))
        return self.act(self.bn(self.conv(x)))

    @torch.no_grad()
    def fuse(self):
        if self._is_fused:
            return self
        # Simplified fuse assuming eval-mode running stats
        w = self.conv.weight  # [oc, ic, 1, 1]
        oc, ic, kh, kw = w.shape
        w2d = w.view(oc, ic * kh * kw)
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        weight = self.bn.weight
        bias = self.bn.bias
        eps = self.bn.eps
        scale = weight / torch.sqrt(running_var + eps)      # [oc]
        shift = bias - running_mean * scale                  # [oc]
        new_w = (w2d * scale.view(-1, 1)).view(oc, ic, kh, kw)
        new_b = shift
        conv = self.conv
        conv.weight.data.copy_(new_w)
        conv.bias = nn.Parameter(new_b)
        delattr(self, "bn")
        self._is_fused = True
        self.act = nn.Identity()
        return self

def autopad(k: int | tuple[int, int], p=None, d: int = 1):
    if isinstance(k, tuple):
        if d > 1:
            k = tuple(d * (x - 1) + 1 for x in k)
        if p is None:
            return tuple(x // 2 for x in k)
        return p
    if d > 1:
        k = d * (k - 1) + 1
    return k // 2 if p is None else p

# Triton kernel: fused (q@k^T)->softmax->(v@attn^T)
import triton
import triton.language as tl

def _ceil_div(a, b):
    return (a + b - 1) // b

@triton.jit
def _matmul_softmax_matmul_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    B: tl.constexpr, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
    stride_q_b, stride_q_m, stride_q_k, stride_q_n,
    stride_k_b, stride_k_m, stride_k_k, stride_k_n,
    stride_v_b, stride_v_k, stride_v_n,
    stride_out_b, stride_out_k, stride_out_n,
    scale: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # program ids
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)  # which (b, m) row

    # iterate over N in blocks
    n0 = 0
    while n0 < N:
        offs_n = n0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        # 1) Compute S block: s_blk = sum_k q[k] * k[:, n] over K
        s_blk = tl.zeros((BLOCK_N,), dtype=tl.float32)
        k0 = 0
        while k0 < K:
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            # q[m, kk, n] -> shape [BK, BN]
            q = tl.load(
                q_ptr + pid_b * stride_q_b + pid_m * stride_q_m
                + (offs_k[:, None] * stride_q_k) + (offs_n[None, :] * stride_q_n),
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
            ).to(tl.float32)  # [BK, BN]

            # k[m, kk, n] -> shape [BK, BN]
            k_ = tl.load(
                k_ptr + pid_b * stride_k_b + pid_m * stride_k_m
                + (offs_k[:, None] * stride_k_k) + (offs_n[None, :] * stride_k_n),
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
            ).to(tl.float32)  # [BK, BN]

            # s contribution: sum over K-axis
            prod = q * k_  # [BK, BN]
            s_blk += tl.sum(prod, axis=0)  # [BN]

            k0 += BLOCK_K

        # apply scale
        s_blk = s_blk * scale

        # softmax over BN: numer, denom
        max_val = tl.max(s_blk, axis=0)
        s_blk = s_blk - max_val
        numer = tl.exp(s_blk)
        denom = tl.sum(numer, axis=0)
        attn_blk = numer / denom  # [BN], float32

        # 2) Compute O block: o_blk = sum_k v[k] * attn[k]
        o_blk = tl.zeros((BLOCK_N,), dtype=tl.float32)
        k0 = 0
        while k0 < K:
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            v_ = tl.load(
                v_ptr + pid_b * stride_v_b + (offs_k[:, None] * stride_v_k) + (offs_n[None, :] * stride_v_n),
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
            ).to(tl.float32)  # [BK, BN]

            a_ = tl.load(
                attn_blk + offs_n,  # [BN] stored linear
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)     # [BN]

            # Broadcast a_ over rows: [1, BN]
            contrib = v_ * a_[None, :]  # [BK, BN]
            o_blk += tl.sum(contrib, axis=0)  # [BN]

            k0 += BLOCK_K

        # store O block as float16
        tl.store(
            out_ptr + pid_b * stride_out_b + (offs_n * stride_out_n),
            o_blk.to(tl.float16),
            mask=mask_n,
        )

        n0 += BLOCK_N

# Python wrapper
def triton_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     scale: float,
                     block_n: int = 128, block_k: int = 32) -> torch.Tensor:
    """
    q: [B, M, K, N], k: [B, M, K, N], v: [B, K, N] (float16), CUDA
    Returns O: [B, K, N] = v @ softmax(q @ k^T)^T as float16
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    B, M, K, N = q.shape[0], q.shape[1], q.shape[2], q.shape[3]
    # contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    out = torch.empty((B, K, N), device=q.device, dtype=torch.float16)

    # strides in elements
    sq_b, sq_m, sq_k, sq_n = q.stride()
    sk_b, sk_m, sk_k, sk_n = k.stride()
    sv_b, sv_k, sv_n = v.stride()
    so_b, so_k, so_n = out.stride()

    grid = (B, M)

    _matmul_softmax_matmul_kernel[grid](
        q, k, v, out,
        B, M, K, N,
        sq_b, sq_m, sq_k, sq_n,
        sk_b, sk_m, sk_k, sk_n,
        sv_b, sv_k, sv_n,
        so_b, so_k, so_n,
        scale,
        BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=4, num_stages=2,
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = YOLOConv(dim, h, 1, act=False)  # 1x1 conv
        self.proj = YOLOConv(dim, dim, 1, act=False)  # 1x1 conv
        self.pe = YOLOConv(dim, dim, 3, 1, g=dim, act=False)  # 3x3 depthwise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "ModelNew requires CUDA tensors."
        # qkv -> [B, h, H, W]
        qkv = self.qkv(x)
        B, Ch, H, W = qkv.shape
        n = H * W
        M = self.num_heads * self.key_dim
        K = self.num_heads * self.head_dim

        # Reshape and views
        qkv = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, n)
        q = qkv[:, :, :self.key_dim, :]     # [B, H, Kq, N]
        k = qkv[:, :, self.key_dim:self.key_dim * 2, :]  # [B, H, Kq, N]
        v = qkv[:, :, self.key_dim * 2:, :]  # [B, H, Kv, N]

        # Permute to [B, M, K, N]
        q = q.reshape(B, self.num_heads, self.key_dim, n).permute(0, 1, 3, 2).contiguous()  # [B, H, N, Kq]
        q = q.reshape(B, M, n, self.key_dim).permute(0, 1, 3, 2).contiguous()               # [B, M, Kq, N]
        k = k.reshape(B, self.num_heads, self.key_dim, n).permute(0, 1, 3, 2).contiguous()
        k = k.reshape(B, M, n, self.key_dim).permute(0, 1, 3, 2).contiguous()
        v = v.reshape(B, self.num_heads, self.head_dim, n).permute(0, 1, 3, 2).contiguous() # [B, H, N, Kv]
        v = v.reshape(B, self.num_heads * self.head_dim, n).contiguous()                    # [B, Kv, N]

        # Launch Triton kernel to get O = v @ softmax(q @ k^T)^T  -> [B, K, N]
        O = triton_attention(q, k, v, scale=self.scale, block_n=128, block_k=32)

        # Reshape back to [B, C, H, W]
        # O is [B, K, N] with K=C
        x_out = O.view(B, self.num_heads, self.head_dim, H, W).permute(0, 1, 3, 4, 2).contiguous()
        x_out = x_out.view(B, Ch, H, W)

        # pe on v-reshape(x): interpret as pe on x_out
        pe = self.pe(x_out)
        x_out = x_out + pe

        # final proj
        out = self.proj(x_out)
        return out

YOLOAttention = ModelNew
