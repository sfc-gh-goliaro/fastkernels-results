"""CLIP self-attention (L2) -- fused Triton implementation.

Captured workload: batch 1, seq 77, hidden 768, 12 heads, fp32 -- tiny, so the
eager baseline's ~10 kernel launches plus cuBLAS' fp32 GEMM path cost ~102us.
Under the benchmark's per-iteration L2 flush even an *empty* kernel launch costs
~4us of wall clock, so the op is collapsed into three launches, each of which
stays within ~1 launch-quantum of work:

  1. ``_gemm_bias_kernel``  X[S,768] @ Wqkv^T + b  -> QKV[S,2304] (fp16)
  2. ``_attn_kernel``       per-head SDPA, whole 77x77 score block in registers
  3. ``_gemm_bias_kernel``  O[S,768] @ Wo^T  + b   -> out[1,S,768] (fp32)

Numerics.  With torch >= 2.9 the default fp32 matmul precision is ``high``, so
every cuBLAS GEMM the baseline runs rounds its operands to TF32 (11 significant
bits) and accumulates in fp32 -- the baseline is *not* exact fp32, and a more
accurate candidate fails the comparison (a bf16x3 / exact-fp32 version matches
only ~83% of elements).  fp16 has the same 11-bit significand as TF32, so over
this op's dynamic range ``fp16(v) == tf32_rne(v)``; feeding fp16 operands to the
tensor cores with fp32 accumulation reproduces the baseline's arithmetic, with
only summation order left over (~1e-6 relative).  That also makes it correct to
*store* the QKV and attention results as fp16: it is exactly the rounding the
baseline's next GEMM would have applied, and it halves their traffic.  Triton's
``input_precision="tf32"`` truncates rather than rounds to TF32 and lands an
order of magnitude further from cuBLAS, so it is deliberately not used.

Weights are packed to fp16 lazily on first use (after the harness has shared the
baseline's weights via ``load_state_dict``) and re-packed if they are mutated.
Anything outside the fast path -- other dtypes, seq outside [32, 128], batch > 1,
an unexpected mask layout, no Triton -- falls back to the eager formulation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import CLIPTextConfig

from ..L1.linear import BMM, Linear
from ..L1.softmax import Softmax

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # pragma: no cover - eager fallback
    _HAVE_TRITON = False


if _HAVE_TRITON:

    @triton.jit
    def _gemm_bias_kernel(
        X, W, B, OUT, S, K, sx_m, sw_k, so_m,
        FP16_OUT: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        """OUT[S, N] = X[S, K] @ W[K, N] + B[N]; W fp16, fp32 accumulation."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        x_ptrs = X + offs_m[:, None] * sx_m + offs_k[None, :]
        w_ptrs = offs_k[:, None] * sw_k + offs_n[None, :]
        mrow = offs_m[:, None] < S

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            a = tl.load(x_ptrs, mask=mrow, other=0.0)
            b = tl.load(W + w_ptrs)
            acc = tl.dot(a.to(tl.float16), b, acc)
            x_ptrs += BK
            w_ptrs += BK * sw_k

        acc += tl.load(B + offs_n)[None, :]
        if FP16_OUT:
            acc = acc.to(tl.float16)
        tl.store(OUT + offs_m[:, None] * so_m + offs_n[None, :], acc, mask=mrow)

    @triton.jit
    def _attn_kernel(
        QKV, MASK, O, S, scale, s_qkv, s_mask, s_o,
        HAS_MASK: tl.constexpr, D: tl.constexpr, HD: tl.constexpr,
        BM: tl.constexpr, BS: tl.constexpr,
    ):
        """O[:, h] = softmax(Q_h K_h^T * scale + mask) V_h for one head."""
        h = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_s = tl.arange(0, BS)
        offs_d = h * HD + tl.arange(0, HD)

        mrow = offs_m[:, None] < S
        kval = offs_s[None, :] < S
        krow = offs_s[:, None] < S

        q = tl.load(QKV + offs_m[:, None] * s_qkv + offs_d[None, :], mask=mrow, other=0.0)
        k = tl.load(QKV + offs_s[:, None] * s_qkv + (D + offs_d)[None, :], mask=krow, other=0.0)
        v = tl.load(QKV + offs_s[:, None] * s_qkv + (2 * D + offs_d)[None, :], mask=krow, other=0.0)

        s = tl.dot(q, tl.trans(k)) * scale
        if HAS_MASK:
            s += tl.load(MASK + offs_m[:, None] * s_mask + offs_s[None, :],
                         mask=mrow & kval, other=0.0)
        s = tl.where(kval, s, float("-inf"))
        p = tl.exp(s - tl.max(s, 1)[:, None])
        p = p / tl.sum(p, 1)[:, None]

        o = tl.dot(p.to(tl.float16), v)
        tl.store(O + offs_m[:, None] * s_o + offs_d[None, :],
                 o.to(tl.float16), mask=mrow)


# Tile shapes / launch params (swept on the target GPU; all three kernels sit at
# the measured launch-quantum floor for their shape).
#
# Shortest sequence handled by the fast path.  For small M, cuBLAS stops using
# the tensor-core path (at M == 1 ``F.linear`` is a GEMV and runs in exact fp32),
# so the TF32-matching argument above no longer holds and those cases go to the
# eager formulation instead.  The captured shape (77) is comfortably inside.
_MIN_SEQ = 32
_QKV_BM, _QKV_BN, _QKV_BK, _QKV_W, _QKV_S = 16, 64, 64, 4, 4
_ATT_BM, _ATT_BS, _ATT_W = 16, 128, 4
_OUT_BM, _OUT_BN, _OUT_BK, _OUT_W, _OUT_S = 16, 64, 128, 4, 4


class CLIPAttention(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = Linear(self.embed_dim, self.embed_dim, bias=True)

        self.bmm = BMM()
        self.softmax = Softmax(dim=-1)

        self._packed = None
        self._pack_tag = None
        self._scratch = None

    # -- fp16 weight packing, refreshed if the parameters change --------------
    def _weight_tag(self):
        ps = (self.q_proj, self.k_proj, self.v_proj, self.out_proj)
        return tuple((p.weight._version, p.bias._version, p.weight.data_ptr()) for p in ps)

    def _pack(self, tag):
        w = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], 0)
        wqkv = w.t().contiguous().to(torch.float16)
        wo = self.out_proj.weight.t().contiguous().to(torch.float16)
        bqkv = torch.cat(
            [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]).contiguous()
        self._packed = (wqkv, bqkv, wo, self.out_proj.bias.contiguous())
        self._pack_tag = tag
        return self._packed

    def _buffers_for(self, seq: int, device):
        sc = self._scratch
        if sc is None or sc[0] != seq or sc[1].device != device:
            qkv = torch.empty((seq, 3 * self.embed_dim), device=device, dtype=torch.float16)
            o = torch.empty((seq, self.embed_dim), device=device, dtype=torch.float16)
            self._scratch = sc = (seq, qkv, o)
        return sc[1], sc[2]

    def _eager(self, hidden_states, attention_mask):
        batch_size, seq_length, _ = hidden_states.shape
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)
        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        attn_weights = self.bmm(queries, keys.transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = self.softmax(attn_weights.float()).to(queries.dtype)
        attn_output = self.bmm(attn_weights, values)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.embed_dim)
        return self.out_proj(attn_output)

    def _fast_ok(self, hidden_states, attention_mask) -> bool:
        D = self.embed_dim
        if (
            hidden_states.dtype is not torch.float32
            or not hidden_states.is_cuda
            or hidden_states.dim() != 3
            or hidden_states.shape[0] != 1
            or hidden_states.shape[2] != D
            or not (_MIN_SEQ <= hidden_states.shape[1] <= _ATT_BS)
            or self.head_dim not in (16, 32, 64, 128)
            or D % _QKV_BN or D % _OUT_BN or D % _QKV_BK or D % _OUT_BK
            or not hidden_states.is_contiguous()
        ):
            return False
        if attention_mask is not None:
            S = hidden_states.shape[1]
            if (
                attention_mask.dtype is not torch.float32
                or tuple(attention_mask.shape) != (1, 1, S, S)
                or not attention_mask.is_contiguous()
            ):
                return False
        return True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not _HAVE_TRITON or not self._fast_ok(hidden_states, attention_mask):
            return self._eager(hidden_states, attention_mask)

        D = self.embed_dim
        S = hidden_states.shape[1]
        tag = self._weight_tag()
        packed = self._packed if tag == self._pack_tag else self._pack(tag)
        wqkv, bqkv, wo, bo = packed
        qkv, obuf = self._buffers_for(S, hidden_states.device)
        out = torch.empty_like(hidden_states)
        has_mask = attention_mask is not None

        _gemm_bias_kernel[(triton.cdiv(S, _QKV_BM), 3 * D // _QKV_BN)](
            hidden_states, wqkv, bqkv, qkv, S, D, D, 3 * D, 3 * D,
            FP16_OUT=True, BM=_QKV_BM, BN=_QKV_BN, BK=_QKV_BK,
            num_warps=_QKV_W, num_stages=_QKV_S)
        _attn_kernel[(self.num_heads, triton.cdiv(S, _ATT_BM))](
            qkv, attention_mask if has_mask else qkv, obuf,
            S, self.scale, 3 * D, S, D,
            HAS_MASK=has_mask, D=D, HD=self.head_dim,
            BM=_ATT_BM, BS=_ATT_BS, num_warps=_ATT_W)
        _gemm_bias_kernel[(triton.cdiv(S, _OUT_BM), D // _OUT_BN)](
            obuf, wo, bo, out, S, D, D, D, D,
            FP16_OUT=False, BM=_OUT_BM, BN=_OUT_BN, BK=_OUT_BK,
            num_warps=_OUT_W, num_stages=_OUT_S)
        return out
