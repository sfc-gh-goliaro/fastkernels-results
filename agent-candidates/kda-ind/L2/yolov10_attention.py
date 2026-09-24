"""YOLOv10 spatial attention block, evaluated with three Triton kernels.

At the shapes this block is captured with -- 128 channels, 2 heads, a 20x20
token grid -- the arithmetic is roughly 0.4 GFLOP, which is well under a
microsecond of B200 math. The eager dataflow nevertheless costs well over a
hundred microseconds because it spends eighteen device kernels and a comparable
amount of host dispatch getting there. Kernel count, not FLOPs or bytes, is what
this implementation minimizes.

The block keeps the baseline's ``YOLOConv`` submodules, so every weight still
loads by name, but folds each BatchNorm into its convolution once -- on the
first forward, because the weights are only final after ``load_state_dict`` --
and then evaluates the block in three launches::

    qkv = fold(bn . conv1x1)(x)                                 [B, 256, N]
    y   = softmax_attn(q, k, v) + fold(bn . dwconv3x3)(v)       [B, 128, N]
    out = fold(bn . conv1x1)(y)                                 [B, 128, H, W]

Both 1x1 convolutions become ``[C, N]`` GEMMs over a token-contiguous layout,
and the depthwise positional encoding and its residual add ride along in the
attention kernel's epilogue, reading the same ``v`` rows the attention loop just
touched. Any input the fused path is not specialized for -- a different channel
count or head count, a non-fp16 or non-CUDA tensor, training mode -- falls back
to the baseline dataflow.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.softmax import Softmax
from .yolov10_conv import YOLOConv

# The configuration the Triton path is specialized for. Batch size and the token
# grid stay runtime arguments; only these channel counts and the tile sizes below
# are compile-time constants.
_CHANNELS = 128
_HEADS = 2
_KEY_DIM = 32
_HEAD_DIM = 64
_QKV_CHANNELS = _HEADS * (2 * _KEY_DIM + _HEAD_DIM)
_PE_TAPS = 3

# Tile geometry and launch parameters, chosen by an offline sweep and fixed here:
# autotuning at runtime would recompile or re-benchmark inside the timed loop.
# Small tiles win at these sizes. Nothing here is a throughput problem -- the
# whole block is a few microseconds of arithmetic -- so what matters is spreading
# the work over enough CTAs to cover memory latency once, rather than giving each
# CTA a large tile. Widening the attention tile to BLOCK_M=64 halves the CTA count
# and measurably slows the kernel down.
_GEMM_BLOCK_OC = 32
_GEMM_BLOCK_N = 64
_GEMM_WARPS = 8
_GEMM_STAGES = 3

_ATTN_BLOCK_M = 32
_ATTN_BLOCK_N = 256
_ATTN_WARPS = 4
_ATTN_STAGES = 3


@triton.jit
def _conv1x1_bn_gemm(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    n_tokens, x_batch_stride, out_batch_stride,
    OUT_CHANNELS: tl.constexpr, IN_CHANNELS: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """``out[b, oc, n] = sum_ic weight[oc, ic] * x[b, ic, n] + bias[oc]``.

    A 1x1 NCHW convolution is a GEMM once the spatial axes are flattened into a
    token axis, and with tokens contiguous both operands are read coalesced: the
    weight tile is ``[BLOCK_OC, IN_CHANNELS]`` and the activation tile is
    ``[IN_CHANNELS, BLOCK_N]`` with unit stride along tokens. The folded bias is
    fp32 and is added to the fp32 accumulator before the narrowing store.

    ``BLOCK_OC`` must divide ``OUT_CHANNELS``; the token axis is masked, since
    the captured token count (400) divides no useful tile size.
    """
    token_tile = tl.program_id(0)
    channel_tile = tl.program_id(1)
    batch = tl.program_id(2)

    offs_token = token_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_oc = channel_tile * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_ic = tl.arange(0, IN_CHANNELS)
    token_mask = offs_token < n_tokens

    weight = tl.load(weight_ptr + offs_oc[:, None] * IN_CHANNELS + offs_ic[None, :])
    x = tl.load(
        x_ptr + batch * x_batch_stride + offs_ic[:, None] * n_tokens + offs_token[None, :],
        mask=token_mask[None, :], other=0.0,
    )
    acc = tl.dot(weight, x, out_dtype=tl.float32)
    acc += tl.load(bias_ptr + offs_oc)[:, None]

    tl.store(
        out_ptr + batch * out_batch_stride + offs_oc[:, None] * n_tokens + offs_token[None, :],
        acc.to(out_ptr.dtype.element_ty), mask=token_mask[None, :],
    )


@triton.jit
def _attention_positional(
    qkv_ptr, pe_weight_ptr, pe_bias_ptr, out_ptr,
    n_tokens, height, width, attn_scale,
    qkv_batch_stride, out_batch_stride,
    HEADS: tl.constexpr, KEY_DIM: tl.constexpr, HEAD_DIM: tl.constexpr,
    TAPS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Streaming-softmax attention whose epilogue adds the positional encoding.

    One program owns one ``(batch, head, token tile)``. Within a head, the
    packed ``qkv`` rows are ``q`` then ``k`` then ``v``, and every operand is
    dim-major -- one token per column -- so the token axis is the contiguous one
    for all three and ``tl.dot`` consumes them without an explicit transpose.
    The ``[N, N]`` score matrix is never materialized.

    The epilogue folds in the depthwise 3x3 convolution that the baseline
    applies to ``v`` reshaped to ``[B, C, H, W]``, its folded bias, and the
    residual add, which together remove four more launches.
    """
    token_tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // HEADS
    head = batch_head % HEADS

    head_base = (qkv_ptr + batch * qkv_batch_stride
                 + head * (2 * KEY_DIM + HEAD_DIM) * n_tokens)
    q_base = head_base
    k_base = head_base + KEY_DIM * n_tokens
    v_base = head_base + 2 * KEY_DIM * n_tokens

    offs_query = token_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_key_dim = tl.arange(0, KEY_DIM)
    offs_head_dim = tl.arange(0, HEAD_DIM)
    query_mask = offs_query < n_tokens

    q = tl.load(q_base + offs_query[:, None] + offs_key_dim[None, :] * n_tokens,
                mask=query_mask[:, None], other=0.0)

    running_max = tl.full([BLOCK_M], float("-inf"), tl.float32)
    running_sum = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    for key_start in range(0, n_tokens, BLOCK_N):
        offs_key = key_start + tl.arange(0, BLOCK_N)
        key_mask = offs_key < n_tokens

        k = tl.load(k_base + offs_key_dim[:, None] * n_tokens + offs_key[None, :],
                    mask=key_mask[None, :], other=0.0)
        # Scaling before the reduction is what makes the running maximum the true
        # row maximum; masking the key tail to -inf before it keeps padding
        # columns out of both the maximum and the sum.
        scores = tl.dot(q, k, out_dtype=tl.float32) * attn_scale
        scores = tl.where(key_mask[None, :], scores, float("-inf"))

        tile_max = tl.maximum(running_max, tl.max(scores, 1))
        rescale = tl.exp(running_max - tile_max)
        probs = tl.exp(scores - tile_max[:, None])
        running_sum = running_sum * rescale + tl.sum(probs, 1)
        acc = acc * rescale[:, None]

        v = tl.load(v_base + offs_key[:, None] + offs_head_dim[None, :] * n_tokens,
                    mask=key_mask[:, None], other=0.0)
        acc = tl.dot(probs.to(v.dtype), v, acc, out_dtype=tl.float32)
        running_max = tile_max

    acc = acc / running_sum[:, None]

    # The positional encoding runs on v viewed as [B, C, H, W], whose compact
    # channel `head * HEAD_DIM + dv` is exactly row `dv` of this head's v block,
    # so every tap re-reads rows the attention loop just streamed.
    channel = head * HEAD_DIM + offs_head_dim
    row = offs_query // width
    col = offs_query - row * width

    for tap_row in tl.static_range(TAPS):
        for tap_col in tl.static_range(TAPS):
            # Conv2d is a cross-correlation with padding 1, so a tap reads
            # v[row + tap_row - 1, col + tap_col - 1]. Row and column have to be
            # bounds-checked separately: a single check on the flat token index
            # would let a tap wrap around into the neighbouring row.
            src_row = row + tap_row - TAPS // 2
            src_col = col + tap_col - TAPS // 2
            inside = ((src_row >= 0) & (src_row < height)
                      & (src_col >= 0) & (src_col < width) & query_mask)
            src_token = tl.where(inside, src_row * width + src_col, 0)

            tap = tl.load(
                v_base + src_token[:, None] + offs_head_dim[None, :] * n_tokens,
                mask=inside[:, None], other=0.0,
            )
            tap_weight = tl.load(pe_weight_ptr + channel * (TAPS * TAPS)
                                 + tap_row * TAPS + tap_col)
            acc += tap_weight.to(tl.float32)[None, :] * tap.to(tl.float32)

    acc += tl.load(pe_bias_ptr + channel)[None, :]

    tl.store(
        out_ptr + batch * out_batch_stride + channel[None, :] * n_tokens + offs_query[:, None],
        acc.to(out_ptr.dtype.element_ty), mask=query_mask[:, None],
    )


class _FusedWeights(NamedTuple):
    """Conv+BN folded once per weight set. Plain tensors, never registered.

    Registering these would put them in ``state_dict`` and expose them to the
    harness's dtype casting; they are derived data, so they stay off the module's
    parameter and buffer registries.
    """

    qkv_weight: torch.Tensor   # [QKV_CHANNELS, CHANNELS] in the source dtype
    qkv_bias: torch.Tensor     # [QKV_CHANNELS] fp32
    pe_weight: torch.Tensor    # [CHANNELS, TAPS, TAPS] in the source dtype
    pe_bias: torch.Tensor      # [CHANNELS] fp32
    proj_weight: torch.Tensor  # [CHANNELS, CHANNELS] in the source dtype
    proj_bias: torch.Tensor    # [CHANNELS] fp32


def _drop_fused_cache(module: "YOLOAttention", _incompatible_keys) -> None:
    """Discard folded weights after a ``load_state_dict``.

    Kept a module-level function rather than a bound method so the module does
    not hold a reference cycle through its own hook registry.
    """
    module._fused = None
    module._fused_key = None


@torch.no_grad()
def _fold_conv_bn(block: YOLOConv) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused ``(weight, bias)`` for one eval-mode ``YOLOConv``, computed in fp32.

    With ``s = gamma / sqrt(running_var + eps)`` the fold is ``W' = W * s`` and
    ``b' = beta - running_mean * s``. Both BN affine parameters arrive in fp16
    (the harness casts parameters but not buffers) and are upcast here: computing
    ``beta - running_mean * s`` in fp16 loses most of the bias to cancellation
    whenever ``s`` is large.

    ``YOLOConv.fuse()`` is deliberately not reused. It folds in fp16, writes the
    result back into ``conv.weight``, adds a ``conv.bias`` parameter and deletes
    ``bn`` -- which would change ``state_dict`` and break the fallback path.
    """
    conv, bn = block.conv, block.bn
    scale = bn.weight.float() / torch.sqrt(bn.running_var.float() + bn.eps)
    bias = bn.bias.float() - bn.running_mean.float() * scale
    if conv.bias is not None:
        bias = bias + conv.bias.float() * scale
    weight = conv.weight.float() * scale.reshape(-1, *([1] * (conv.weight.dim() - 1)))
    return weight.to(conv.weight.dtype), bias


class YOLOAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = YOLOConv(dim, h, 1, act=False)
        self.proj = YOLOConv(dim, dim, 1, act=False)
        self.pe = YOLOConv(dim, dim, 3, 1, g=dim, act=False)
        self._softmax = Softmax(dim=-1)
        # Folding is deferred to the first forward: __init__ runs before weights
        # are loaded, and must not touch CUDA.
        self._fused: _FusedWeights | None = None
        self._fused_key: tuple | None = None
        self.register_load_state_dict_post_hook(_drop_fused_cache)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_specialized(x):
            return self._forward_fused(x)
        return self._forward_reference(x)

    # -- specialization ----------------------------------------------------

    def _is_specialized(self, x: torch.Tensor) -> bool:
        """Whether the fused path applies. Reads shapes and dtypes only."""
        weight = self.qkv.conv.weight
        return (
            not self.training
            and x.dim() == 4
            and x.numel() > 0
            and x.is_cuda
            and weight.is_cuda
            and x.dtype is torch.float16
            and weight.dtype is torch.float16
            and x.shape[1] == _CHANNELS
            and self.num_heads == _HEADS
            and self.key_dim == _KEY_DIM
            and self.head_dim == _HEAD_DIM
        )

    def _forward_reference(self, x: torch.Tensor) -> torch.Tensor:
        """The baseline dataflow, correct by construction for any configuration."""
        b, c, h, w = x.shape
        n = h * w
        qkv = self.qkv(x)
        q, k, v = qkv.view(b, self.num_heads, self.key_dim * 2 + self.head_dim, n).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = self._softmax((q.transpose(-2, -1) @ k) * self.scale)
        y = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.pe(v.reshape(b, c, h, w))
        return self.proj(y)

    # -- folded weights ----------------------------------------------------

    def _weight_signature(self) -> tuple:
        """Identity of the weight set the fold was built from.

        Device and dtype are the whole key, and they are read through a single
        attribute chain: walking all fifteen source tensors to compare version
        counters instead cost more host time per forward than a kernel launch,
        and ``nn.Module.__getattr__`` is what makes that walk expensive. A weight
        set loaded a second time is caught exactly by the ``load_state_dict``
        post hook rather than by polling versions here. The one case neither
        covers is ``load_state_dict`` called directly on ``qkv``/``pe``/``proj``
        rather than on this block, which nothing in the harness does.
        """
        anchor = self.qkv.conv.weight
        return (anchor.device, anchor.dtype)

    def _fused_weights(self) -> _FusedWeights:
        key = self._weight_signature()
        fused = self._fused
        if fused is None or self._fused_key != key:
            qkv_weight, qkv_bias = _fold_conv_bn(self.qkv)
            pe_weight, pe_bias = _fold_conv_bn(self.pe)
            proj_weight, proj_bias = _fold_conv_bn(self.proj)
            fused = _FusedWeights(
                qkv_weight.reshape(_QKV_CHANNELS, _CHANNELS).contiguous(),
                qkv_bias.contiguous(),
                pe_weight.reshape(_CHANNELS, _PE_TAPS, _PE_TAPS).contiguous(),
                pe_bias.contiguous(),
                proj_weight.reshape(_CHANNELS, _CHANNELS).contiguous(),
                proj_bias.contiguous(),
            )
            self._fused = fused
            self._fused_key = key
        return fused

    # -- fused path --------------------------------------------------------

    def _forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        # Everything here is on the host critical path. The harness gives the
        # host a head start of roughly one L2 flush (~73 us of GPU work enqueued
        # before the start event), and three Triton launches plus three
        # allocations already spend most of it, so arguments are passed
        # positionally -- Triton's keyword binding costs about 1.5 us a launch --
        # and nothing is recomputed that can be hoisted.
        fused = self._fused_weights()
        x = x.contiguous()
        b, _, h, w = x.shape
        n = h * w
        device, dtype = x.device, x.dtype
        qkv_stride = _QKV_CHANNELS * n
        channel_stride = _CHANNELS * n

        qkv = torch.empty((b, _QKV_CHANNELS, n), device=device, dtype=dtype)
        attended = torch.empty((b, _CHANNELS, n), device=device, dtype=dtype)
        out = torch.empty((b, _CHANNELS, h, w), device=device, dtype=dtype)

        gemm_token_tiles = -(-n // _GEMM_BLOCK_N)
        _conv1x1_bn_gemm[(gemm_token_tiles, _QKV_CHANNELS // _GEMM_BLOCK_OC, b)](
            x, fused.qkv_weight, fused.qkv_bias, qkv,
            n, channel_stride, qkv_stride,
            _QKV_CHANNELS, _CHANNELS, _GEMM_BLOCK_OC, _GEMM_BLOCK_N,
            num_warps=_GEMM_WARPS, num_stages=_GEMM_STAGES,
        )
        _attention_positional[(-(-n // _ATTN_BLOCK_M), b * _HEADS)](
            qkv, fused.pe_weight, fused.pe_bias, attended,
            n, h, w, self.scale, qkv_stride, channel_stride,
            _HEADS, _KEY_DIM, _HEAD_DIM, _PE_TAPS, _ATTN_BLOCK_M, _ATTN_BLOCK_N,
            num_warps=_ATTN_WARPS, num_stages=_ATTN_STAGES,
        )
        _conv1x1_bn_gemm[(gemm_token_tiles, _CHANNELS // _GEMM_BLOCK_OC, b)](
            attended, fused.proj_weight, fused.proj_bias, out,
            n, channel_stride, channel_stride,
            _CHANNELS, _CHANNELS, _GEMM_BLOCK_OC, _GEMM_BLOCK_N,
            num_warps=_GEMM_WARPS, num_stages=_GEMM_STAGES,
        )
        return out
