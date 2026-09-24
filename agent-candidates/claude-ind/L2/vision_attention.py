"""Encoder-only attention for Qwen vision transformer blocks.

Non-causal, no KV cache. Uses FlashAttnPrefill L1 op with cu_seqlens
for variable-length sequence support within the vision encoder.

Optimization notes (vs. the baseline)
-------------------------------------
Measured split of the baseline on B200 for the hot 23760-token shape
(embed_dim 1152, 16 heads, head_dim 72): QKV GEMM 19%, q/k contiguous-copy 21%,
Triton rotary 11%, FA4 attention 42%, out-projection 8%.  Three changes, in
order of what they buy:

1. **No q/k copy.**  The baseline materializes a contiguous ``(2, seq, heads,
   dim)`` q|k tensor purely so attention gets contiguous inputs -- a 219 MB
   read+write gather that torch lowers to a generic ``elementwise_kernel`` and
   that runs at ~1.6 TB/s, a third of HBM speed.  FA4 never needed it: its
   ``maybe_contiguous`` only requires ``stride(-1) == 1``, so k and v are handed
   over as strided views straight into the fused QKV buffer.

2. **One fused RoPE kernel** (``rope_qk``) replaces flash-attn's Triton rotary.
   It reads each token's q|k halves once, rotates in fp32, rotates k in place,
   and writes q out pre-padded (below).  ~3.3 TB/s versus the Triton kernel's
   ~3.0 TB/s on a third of the total traffic, and it drops a second full pass
   over 219 MB plus ~190 us/call of Python (``apply_rotary`` re-validates and
   re-contiguous-es its arguments on every call).

3. **head_dim 96 for the QK product.**  FA4 on SM100 rounds head_dim up to a
   multiple of 16, and its 80-wide layout is a slow path: for this shape it
   costs 434 us at (72, 72) but only 346 us at (96, 72) -- 25% of the whole
   operator.  ``rope_qk`` therefore writes q into a 96-wide per-head buffer with
   the tail zeroed, and k is passed as an *overlapping* 96-wide view of the
   packed buffer: q's zero tail already kills every q.k term past 72, so k's
   tail may hold whatever the neighbouring head left there.  head_dim_v stays
   72, so attention still writes a packed [tokens, heads, 72] output and
   ``proj`` is untouched.

Numerics are bit-identical to the baseline on every captured shape (the rotary
runs in fp32 and rounds once, exactly like the Triton kernel, and the padded
lanes contribute exact zeros).

Per-call Python is 280 us in the baseline and 50 us here, which is what the
1760-token shape is actually bound by (its GPU work is ~46 us): the wrapper
frames down to FA4's kernel are gone (see ``_fa4_kernel``), ``apply_rotary``'s
re-validation with them, the two linears skip ``nn.Module.__call__``, and the
GEMM scratch is allocated once per shape instead of per call.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn.ops.triton.rotary import apply_rotary

from ....infra.tp import _tp_size, _tp_rank
from ..L1.flash_attn_prefill import FlashAttnPrefill
from .parallel_linear import QKVParallelLinear, RowParallelLinear


# ---------------------------------------------------------------------------
# Fused RoPE over the packed [tokens, (q|k|v) * heads * dim] GEMM output.
# ---------------------------------------------------------------------------
_ROPE_CUDA = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

struct __align__(8) bf16x4 { __nv_bfloat162 a, b; };
struct __align__(16) bf16x8 { __nv_bfloat162 a, b, c, d; };

// One block handles one token: its 2 * NH * (HALF/4) lanes each own a 4-element
// chunk of one rotary half of one (q|k) head, so every lane issues exactly two
// 8-byte loads and two 8-byte stores (the 36-element halves rule out 16-byte
// accesses -- 36 is not a multiple of 8).  q goes to a DPAD-wide per-head
// buffer with the [HD, DPAD) tail zeroed; k is rotated in place.
template <int NH, int HD, int HALF, int DPAD>
__global__ void rope_qk_kernel(
    __nv_bfloat16* __restrict__ qkv,
    __nv_bfloat16* __restrict__ qpad,
    const __nv_bfloat16* __restrict__ cos_ptr,
    const __nv_bfloat16* __restrict__ sin_ptr,
    const long row_stride) {
  constexpr int CH = HALF / 4;
  constexpr int NPAD8 = NH * (DPAD - HD) / 8;   // 16B zero stores per token
  const int tid = threadIdx.x;
  const int which = tid / (NH * CH);            // 0 = q, 1 = k
  const int r2 = tid - which * (NH * CH);
  const int head = r2 / CH;
  const int c4 = (r2 - head * CH) * 4;
  const long row = blockIdx.x;

  const __nv_bfloat16* xp = qkv + row * row_stride
      + (long)which * (NH * HD) + (long)head * HD + c4;
  __nv_bfloat16* op = which
      ? (__nv_bfloat16*)xp
      : qpad + row * (NH * DPAD) + (long)head * DPAD + c4;

  const long cs = row * HALF + c4;
  const bf16x4 cv = *reinterpret_cast<const bf16x4*>(cos_ptr + cs);
  const bf16x4 sv = *reinterpret_cast<const bf16x4*>(sin_ptr + cs);
  const bf16x4 x0 = *reinterpret_cast<const bf16x4*>(xp);
  const bf16x4 x1 = *reinterpret_cast<const bf16x4*>(xp + HALF);

  const float2 c0 = __bfloat1622float2(cv.a), c1 = __bfloat1622float2(cv.b);
  const float2 s0 = __bfloat1622float2(sv.a), s1 = __bfloat1622float2(sv.b);
  const float2 a0 = __bfloat1622float2(x0.a), a1 = __bfloat1622float2(x0.b);
  const float2 b0 = __bfloat1622float2(x1.a), b1 = __bfloat1622float2(x1.b);

  bf16x4 o0, o1;
  o0.a = __floats2bfloat162_rn(a0.x * c0.x - b0.x * s0.x,
                               a0.y * c0.y - b0.y * s0.y);
  o0.b = __floats2bfloat162_rn(a1.x * c1.x - b1.x * s1.x,
                               a1.y * c1.y - b1.y * s1.y);
  o1.a = __floats2bfloat162_rn(a0.x * s0.x + b0.x * c0.x,
                               a0.y * s0.y + b0.y * c0.y);
  o1.b = __floats2bfloat162_rn(a1.x * s1.x + b1.x * c1.x,
                               a1.y * s1.y + b1.y * c1.y);
  *reinterpret_cast<bf16x4*>(op) = o0;
  *reinterpret_cast<bf16x4*>(op + HALF) = o1;

  if (tid < NPAD8) {
    constexpr int PZ = (DPAD - HD) / 8;
    const int zh = tid / PZ;
    const int zc = (tid - zh * PZ) * 8;
    bf16x8 z;
    z.a = z.b = z.c = z.d = __floats2bfloat162_rn(0.f, 0.f);
    *reinterpret_cast<bf16x8*>(
        qpad + row * (NH * DPAD) + (long)zh * DPAD + HD + zc) = z;
  }
}

void rope_qk(at::Tensor qkv, at::Tensor qpad, at::Tensor cos, at::Tensor sin,
             int64_t num_heads, int64_t head_dim) {
  TORCH_CHECK(qkv.dim() == 2 && qkv.stride(1) == 1);
  constexpr int NH = 16, HD = 72, HALF = 36, DPAD = 96;
  TORCH_CHECK(num_heads == NH && head_dim == HD);
  const int n_rows = (int)qkv.size(0);
  if (n_rows == 0) return;
  constexpr int NTHREADS = 2 * NH * (HALF / 4);
  rope_qk_kernel<NH, HD, HALF, DPAD><<<n_rows, NTHREADS, 0,
                                       c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<__nv_bfloat16*>(qkv.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(qpad.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(cos.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(sin.data_ptr()),
      (long)qkv.stride(0));
}
"""

_DPAD = 96          # per-head width handed to FA4 for the QK product
_NUM_HEADS = 16
_HEAD_DIM = 72
_EXT = None
_EXT_FAILED = False


def _ext():
    """JIT-build (once) the fused RoPE extension; None if it cannot be built."""
    global _EXT, _EXT_FAILED
    if _EXT is None and not _EXT_FAILED:
        try:
            from torch.utils.cpp_extension import load_inline
            from ....infra import cuda_ext as _cuda_ext
            try:
                _cuda_ext._pin_build_arch()
            except Exception:
                pass
            _EXT = load_inline(
                name="fk_vision_attention_rope",
                cpp_sources=("#include <torch/extension.h>\n"
                             "void rope_qk(at::Tensor qkv, at::Tensor qpad,"
                             " at::Tensor cos, at::Tensor sin,"
                             " int64_t num_heads, int64_t head_dim);\n"),
                cuda_sources=_ROPE_CUDA,
                functions=["rope_qk"],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                verbose=False,
            )
        except Exception:
            _EXT_FAILED = True
    return _EXT


# ---------------------------------------------------------------------------
# Direct entry into FA4's compiled kernel.
#
# ``_flash_attn_fwd`` spends ~30 us of Python per call re-deriving a config that
# is constant for this operator (tile size, q_stage, scheduler flags, ~35-field
# compile key) before it invokes the kernel its JIT cache already holds; calling
# that cached kernel costs 5.6 us. Which matters: the 1760-token shape is
# Python-bound, not GPU-bound.
#
# The kernel is captured by diffing FA4's JIT cache across one ordinary call,
# then *verified bit-for-bit* against that call's own output before it is ever
# used. Capture or verification failing just leaves the normal path in place.
# ---------------------------------------------------------------------------
_FA_MISS = object()
_FA_KERNELS: dict = {}


def _fa4_kernel(fa4, key, q, k, v, cu_seqlens, max_seqlen, scale):
    """(out, kernel_or_None) -- run the full path once and capture its kernel."""
    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd
    cache = getattr(_flash_attn_fwd, "compile_cache", None)
    inner = getattr(cache, "cache", None)
    before = set(inner) if isinstance(inner, dict) else None
    out = fa4(q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
              max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
              softmax_scale=scale, causal=False, num_splits=1)[0]
    _FA_KERNELS[key] = None
    if before is None or torch.cuda.get_device_capability()[0] != 10:
        return out, None
    new = set(inner) - before
    if len(new) != 1:
        return out, None
    try:
        from vllm.vllm_flash_attn.cute.utils import AuxData
        kern = inner[new.pop()]
        probe = torch.empty_like(out)
        kern(q, k, v, probe, None, scale, cu_seqlens, cu_seqlens, None, None,
             None, None, None, None, None, None, None, AuxData(None, None), None)
        if not torch.equal(probe, out):
            return out, None
    except Exception:
        return out, None
    _FA_KERNELS[key] = (kern, AuxData(None, None))
    return out, _FA_KERNELS[key]


class VisionAttention(nn.Module):
    """Multi-head attention for vision encoder (Qwen2-VL / Qwen2.5-VL / Qwen3-VL).

    All heads are attention heads (no GQA). Uses full (non-causal) attention.
    Supports TP: QKV is sharded, then gathered for RoPE, then re-sharded.
    """

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
        self.softmax_scale = self.head_dim ** -0.5
        # The fused kernel is specialized for this encoder's geometry; anything
        # else (TP shards, other models) takes the baseline path below.
        self._fused_ok = (self.num_heads == _NUM_HEADS
                          and self.head_dim == _HEAD_DIM)
        self._plain_qkv = not self.qkv.use_fp8
        self._plain_proj = (self.tp_size == 1 and not self.proj.use_fp8)
        self._wq_t = None
        self._wp_t = None
        self._buf_qkv = None
        self._buf_q = None
        self._buf_n = 0
        self._fa4 = None
        if self.attn.fa_version == 4:
            try:
                from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd
                self._fa4 = _flash_attn_fwd
            except Exception:
                self._fa4 = None

    # -- baseline path: q/k contiguous copy + flash-attn's Triton rotary -----
    def _copy_qkv(self, qkv, seq_len, batch_size, cos, sin):
        q_size = self.num_heads * self.head_dim
        qk = qkv[..., : 2 * q_size].view(
            seq_len, batch_size, 2, self.num_heads, self.head_dim,
        )
        qk = qk.permute(2, 1, 0, 3, 4).contiguous()
        if cos is not None and sin is not None:
            flat = qk.view(2 * batch_size, seq_len, self.num_heads, self.head_dim)
            apply_rotary(flat, cos, sin, inplace=True)
        q = qk[0].reshape(-1, self.num_heads, self.head_dim)
        k = qk[1].reshape(-1, self.num_heads, self.head_dim)
        v = (qkv[..., 2 * q_size:]
             .view(seq_len, batch_size, self.num_heads, self.head_dim)
             .transpose(0, 1)
             .reshape(-1, self.num_heads, self.head_dim))
        return q, k, v

    def forward(
        self, x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        seq_len, batch_size, _ = x.shape
        H, D = self.num_heads, self.head_dim
        fused = (
            self._fused_ok
            and seq_len > 0
            and batch_size == 1
            and self._plain_qkv
            and x.dtype == torch.bfloat16
            and x.is_cuda
            and x.is_contiguous()
            and rotary_pos_emb_cos is not None
            and rotary_pos_emb_sin is not None
            and rotary_pos_emb_cos.dtype == torch.bfloat16
            and rotary_pos_emb_sin.dtype == torch.bfloat16
            and rotary_pos_emb_cos.shape[0] == seq_len
            and rotary_pos_emb_cos.shape[1] * 2 == D
            and rotary_pos_emb_sin.shape[0] == seq_len
            and rotary_pos_emb_sin.shape[1] * 2 == D
            and rotary_pos_emb_cos.is_contiguous()
            and rotary_pos_emb_sin.is_contiguous()
            and _ext() is not None
        )

        if not fused:
            return self._slow_forward(x, cu_seqlens, rotary_pos_emb_cos,
                                      rotary_pos_emb_sin, max_seqlen)

        # Scratch (QKV GEMM output, padded q) is reused across calls: at 20k+
        # tokens these are 160 MB / 70 MB allocations, and the ~6 us of allocator
        # work per call is real money on the small shape, which is Python-bound.
        if seq_len > self._buf_n:   # rare: first call, or a longer sequence
            self._buf_qkv = torch.empty((seq_len, 3 * H * D), dtype=x.dtype,
                                        device=x.device)
            self._buf_q = torch.empty((seq_len, H, _DPAD), dtype=x.dtype,
                                      device=x.device)
            self._buf_n = seq_len
        if (self._wq_t is None
                or self._wq_t.data_ptr() != self.qkv.weight.data_ptr()
                or self._wp_t.data_ptr() != self.proj.weight.data_ptr()):
            # Transposed views, not copies -- but re-taken if a parameter's
            # storage was swapped out (``p.data = ...``) after the last call.
            self._wq_t = self.qkv.weight.t()
            self._wp_t = self.proj.weight.t()
        flat = (self._buf_qkv if seq_len == self._buf_n
                else self._buf_qkv[:seq_len])
        q = (self._buf_q if seq_len == self._buf_n else self._buf_q[:seq_len])
        torch.addmm(self.qkv.bias, x.view(seq_len, -1), self._wq_t, out=flat)

        _ext().rope_qk(flat, q, rotary_pos_emb_cos, rotary_pos_emb_sin, H, D)
        rs = flat.stride(0)
        off = flat.storage_offset()
        # k: overlapping _DPAD-wide view of the packed buffer (only q's padding
        # has to be zero for q.k to stay exact). v: plain packed view.
        k = flat.as_strided((seq_len, H, _DPAD), (rs, D, 1), off + H * D)
        v = flat.as_strided((seq_len, H, D), (rs, D, 1), off + 2 * H * D)

        if max_seqlen is None:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        if self._fa4 is not None:
            # q_stage (and so the compiled kernel) depends on max_seqlen > tile_m.
            key = (_DPAD, D, max_seqlen > 128)
            entry = _FA_KERNELS.get(key, _FA_MISS)
            if entry is _FA_MISS:
                out, entry = _fa4_kernel(self._fa4, key, q, k, v, cu_seqlens,
                                         max_seqlen, self.softmax_scale)
            elif entry is None:
                out = self._fa4(
                    q, k, v,
                    cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                    softmax_scale=self.softmax_scale, causal=False,
                    num_splits=1,
                )[0]
            else:
                out = torch.empty((seq_len, H, D), dtype=q.dtype,
                                  device=q.device)
                entry[0](q, k, v, out, None, self.softmax_scale,
                         cu_seqlens, cu_seqlens, None, None, None, None, None,
                         None, None, None, None, entry[1], None)
        else:
            out = self.attn(q, k, v, cu_seqlens, cu_seqlens,
                            max_seqlen, max_seqlen,
                            softmax_scale=self.softmax_scale, causal=False,
                            num_splits=1)

        if self._plain_proj:
            y = torch.addmm(self.proj.bias, out.view(seq_len, H * D), self._wp_t)
            return y.view(seq_len, batch_size, -1)
        return self.proj(out.view(seq_len, batch_size, -1))

    # -- baseline path (TP shards, fp8, other geometries, no RoPE) -----------
    def _slow_forward(self, x, cu_seqlens, cos, sin, max_seqlen):
        seq_len, batch_size, _ = x.shape
        if self._plain_qkv:
            qkv = F.linear(x, self.qkv.weight, self.qkv.bias)
        else:
            qkv = self.qkv(x)
        q, k, v = self._copy_qkv(qkv, seq_len, batch_size, cos, sin)
        if max_seqlen is None:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        out = self.attn(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                        softmax_scale=self.softmax_scale, causal=False,
                        num_splits=1)
        return self.proj(out.view(seq_len, batch_size, -1))
