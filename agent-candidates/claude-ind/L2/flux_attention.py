"""FLUX attention module (L2 composite) -- fused candidate.

Same math as ``baseline.py``; what changes is the glue between the projections
and the attention kernel.

The baseline spends over half its time *outside* the QKV/out projections and the
attention kernel, in six memory-bound passes over the 85 MB fused-QKV
activation::

    qkv.split -> unflatten -> norm_q (a .contiguous() clone, then vLLM's
    rms_norm) -> norm_k (ditto) -> two `_rotary_kernel` launches ->
    (dual-stream) three `cat`s to prepend the text stream

``_fused_qk_norm_rope`` replaces all of it with a single in-place pass.  Each
program owns BS tokens of one head: it reads that head's q and k slice straight
out of the fused QKV row, RMS-normalizes over the head dim, applies the
interleaved (GPT-J) rotary embedding, and writes the result back where it came
from.  v is never touched, and attention reads q/k/v as strided views of the
same buffer -- so the whole step costs one read and one write of q and k
(112 MB) instead of the baseline's ~450 MB, with no extra allocation.

For dual-stream blocks the two projections are written into *one* joint
``[s_txt + s_img, 3*H*D]`` buffer (text rows first, image rows after) with
``addmm(out=)``, which costs exactly what ``F.linear`` costs.  The
concatenation of the two streams is then just where each GEMM wrote, the three
``cat``s disappear, and one kernel launch normalizes both streams -- rows below
``n_txt`` simply pick up the ``norm_added_{q,k}`` weights.

Because q/k/v stay in (batch, seq, head, dim) order, cuDNN's SDPA returns its
output in that same physical order, so the trailing ``permute(0,2,1,3)`` +
``flatten(2,3)`` (and the dual-stream split) are pure views -- the baseline's
reshape copy disappears too.

Numerics follow the baseline op for op, including the intermediate rounding to
the activation dtype between the norm and the rotary (vLLM's ``rms_norm``
stores bf16; ``_rotary_kernel`` reloads and upcasts) and the
``cos/sin -> bf16`` cast before the rotary, so outputs land within bf16 ULP of
the baseline.  Anything the fused path does not cover -- fp8 projections, fp32
activations, an odd head_dim, batch > 1, a partial rotary dim -- falls through
to ``_forward_ref``, which is the baseline forward verbatim.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import triton
import triton.language as tl

from ....infra.tp import _tp_size
from ..L1.rms_norm import RMSNorm as FP32RMSNorm
from ..L1.diffusion_rope import DiffusionRoPE
from ..L1.dense_attention import DenseAttention
from .parallel_linear import (
    QKVParallelLinear,
    RowParallelLinear,
)


def _tensor_model_parallel_all_gather(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Gather tensor across TP ranks along the given dimension."""
    import torch.distributed as dist
    tp = _tp_size()
    if tp <= 1:
        return tensor
    gather_list = [torch.empty_like(tensor) for _ in range(tp)]
    dist.all_gather(gather_list, tensor)
    return torch.cat(gather_list, dim=dim)


# ---------------------------------------------------------------------------
# Fused qk-norm + rotary
# ---------------------------------------------------------------------------

@triton.jit
def _fused_qk_norm_rope(
    QKV, WQ, WK, WQ_T, WK_T, COS, SIN,
    n_tok, n_txt, sq,
    eps: tl.constexpr, H: tl.constexpr, D: tl.constexpr, BS: tl.constexpr,
    ROPE: tl.constexpr, HAS_W: tl.constexpr, JOINT: tl.constexpr,
):
    """RMSNorm + interleaved RoPE over q and k of a fused QKV activation, in place.

    grid = (cdiv(n_tok, BS), H).  Program (i, h) owns tokens [i*BS, i*BS+BS) of
    head ``h``: two [BS, D] tiles (q at column h*D, k at column (H+h)*D), read
    and written at the same addresses.  A token's row index in ``QKV`` is its
    position in the joint sequence, which is also its row in the rotary table.

    With ``JOINT``, rows below ``n_txt`` are the text stream and take the
    ``*_T`` norm weights -- that is all that distinguishes the two streams once
    both projections have been written into one buffer.
    """
    pid_s = tl.program_id(0)
    h = tl.program_id(1)
    s = pid_s * BS + tl.arange(0, BS)
    m = s[:, None] < n_tok
    d = tl.arange(0, D)

    if ROPE:
        cs = s[:, None] * (D // 2) + tl.arange(0, D // 2)[None, :]
        # FLUX hands the rotary table over in float64.  Rounding it to the
        # activation dtype here -- rather than in a separate pass over the
        # table -- is what the baseline's ``cos.to(query.dtype)`` does, and it
        # is cheaper: the table is a few MB, so every re-read hits L2.
        rdt = QKV.dtype.element_ty
        cos = tl.load(COS + cs, mask=m, other=1.0).to(rdt).to(tl.float32)
        sin = tl.load(SIN + cs, mask=m, other=0.0).to(rdt).to(tl.float32)
    if HAS_W:
        wq = tl.load(WQ + d)[None, :].to(tl.float32)
        wk = tl.load(WK + d)[None, :].to(tl.float32)
        if JOINT:
            txt = s[:, None] < n_txt
            wq = tl.where(txt, tl.load(WQ_T + d)[None, :].to(tl.float32), wq)
            wk = tl.where(txt, tl.load(WK_T + d)[None, :].to(tl.float32), wk)

    base = QKV + s[:, None] * sq + d[None, :]
    for t in tl.static_range(2):
        p = base + (h + t * H) * D
        x = tl.load(p, mask=m, other=0.0)
        xf = x.to(tl.float32)
        var = tl.sum(xf * xf, 1) * (1.0 / D)
        xn = xf * tl.rsqrt(var + eps)[:, None]
        if HAS_W:
            xn = xn * (wq if t == 0 else wk)
        # vLLM's rms_norm stores the activation dtype and the rotary kernel
        # reloads it: keep that intermediate rounding.
        xn = xn.to(x.dtype).to(tl.float32)
        if ROPE:
            x0, x1 = tl.split(tl.reshape(xn, (BS, D // 2, 2)))
            xn = tl.reshape(tl.join(x0 * cos - x1 * sin, x0 * sin + x1 * cos), (BS, D))
        tl.store(p, xn.to(x.dtype), mask=m)


_BS = 16
_NUM_WARPS = 8


class FluxAttention(nn.Module):
    """Multi-head attention for FLUX diffusion transformer.

    Supports two modes controlled by constructor args:
    - Dual-stream (``added_kv_proj_dim is not None``): separate QKV for image
      and text streams, concatenated before attention, split after.
    - Single-stream / pre-only (``pre_only=True``): standard self-attention,
      no output projection (caller handles it).
    """

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: int | None = None,
        added_proj_bias: bool | None = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int | None = None,
        context_pre_only: bool | None = None,
        pre_only: bool = False,
        quant_config: dict | None = None,
    ):
        super().__init__()
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.dropout = dropout
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.eps = eps

        self.norm_q = FP32RMSNorm(dim_head, eps=eps)
        self.norm_k = FP32RMSNorm(dim_head, eps=eps)

        self.to_qkv = QKVParallelLinear(
            hidden_size=query_dim,
            head_size=self.head_dim,
            total_num_heads=self.heads,
            total_num_kv_heads=self.heads,
            bias=bias,
            quant_config=quant_config,
        )

        if not self.pre_only:
            self.to_out = nn.ModuleList([
                RowParallelLinear(self.inner_dim, self.out_dim, bias=out_bias,
                                  quant_config=quant_config),
                nn.Dropout(dropout),
            ])

        if added_kv_proj_dim is not None:
            self.norm_added_q = FP32RMSNorm(dim_head, eps=eps)
            self.norm_added_k = FP32RMSNorm(dim_head, eps=eps)

            self.add_kv_proj = QKVParallelLinear(
                hidden_size=added_kv_proj_dim,
                head_size=self.head_dim,
                total_num_heads=self.heads,
                total_num_kv_heads=self.heads,
                bias=added_proj_bias if added_proj_bias is not None else True,
                quant_config=quant_config,
            )

            self.to_add_out = RowParallelLinear(
                self.inner_dim, query_dim, bias=out_bias,
                quant_config=quant_config,
            )

        self.rope = DiffusionRoPE(is_neox_style=False)
        self.attn = DenseAttention()

    # -- fused path ----------------------------------------------------------

    @staticmethod
    def _norm_weight(norm, dtype):
        w = getattr(norm, "weight", None)
        if w is None:
            return None
        return w if w.dtype == dtype else w.to(dtype)

    def _fast_eligible(self, hidden_states, encoder_hidden_states, image_rotary_emb):
        H = self.to_qkv.num_heads
        D = self.head_dim
        if H != self.to_qkv.num_kv_heads or D & (D - 1) or D < 2:
            return False
        if hidden_states.dim() != 3 or hidden_states.shape[0] != 1:
            return False
        if hidden_states.dtype not in (torch.bfloat16, torch.float16):
            return False
        if not hidden_states.is_cuda:
            return False
        if (getattr(self.norm_q, "weight", None) is None
                or getattr(self.norm_k, "weight", None) is None):
            return False
        if self.added_kv_proj_dim is not None:
            if encoder_hidden_states is None or encoder_hidden_states.dim() != 3:
                return False
            if encoder_hidden_states.dtype != hidden_states.dtype:
                return False
            if self.add_kv_proj.num_heads != H or self.add_kv_proj.num_kv_heads != H:
                return False
            # The joint path writes both projections into one buffer with
            # ``addmm(out=)``; the fp8 linear has no such entry point.
            if self.to_qkv.use_fp8 or self.add_kv_proj.use_fp8:
                return False
            if (getattr(self.norm_added_q, "weight", None) is None
                    or getattr(self.norm_added_k, "weight", None) is None):
                return False
        elif encoder_hidden_states is not None:
            return False
        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            if cos.shape != sin.shape or cos.dim() not in (2, 3):
                return False
            # Partial rotary (rotary_dim < head_dim) would need the untouched
            # tail copied through; the FLUX tables always cover the full head.
            if cos.shape[-1] * 2 != D:
                return False
            n_tot = hidden_states.shape[1] + (
                0 if encoder_hidden_states is None else encoder_hidden_states.shape[1])
            if cos.shape[-2] < n_tot:
                return False
        return True

    @staticmethod
    def _proj_into(x, lin, out):
        """``F.linear(x, lin.weight, lin.bias)``, written straight into ``out``.

        cuBLAS applies the bias through the same epilogue either way, so this
        costs exactly what ``F.linear`` costs -- but it lets both streams'
        projections land in one joint buffer, which is what makes the
        concatenation free.
        """
        if lin.bias is None:
            return torch.mm(x, lin.weight.t(), out=out)
        return torch.addmm(lin.bias, x, lin.weight.t(), out=out)

    def _forward_fused(self, hidden_states, encoder_hidden_states, image_rotary_emb):
        H = self.to_qkv.num_heads
        D = self.head_dim
        HD = H * D
        dtype = hidden_states.dtype
        s_img = hidden_states.shape[1]
        s_txt = 0 if encoder_hidden_states is None else encoder_hidden_states.shape[1]
        n_tot = s_img + s_txt
        joint = s_txt > 0

        if joint:
            # Text rows first, image rows after: the baseline's
            # ``cat([encoder, image], dim=1)``, done by the two GEMMs.
            qkv = torch.empty(n_tot, 3 * HD, device=hidden_states.device, dtype=dtype)
            self._proj_into(encoder_hidden_states.reshape(-1, encoder_hidden_states.shape[-1]),
                            self.add_kv_proj, qkv[:s_txt])
            self._proj_into(hidden_states.reshape(-1, hidden_states.shape[-1]),
                            self.to_qkv, qkv[s_txt:])
            wq_t = self._norm_weight(self.norm_added_q, dtype)
            wk_t = self._norm_weight(self.norm_added_k, dtype)
        else:
            qkv = self.to_qkv(hidden_states).view(n_tot, 3 * HD)
            wq_t = wk_t = None

        if image_rotary_emb is None:
            cos = sin = None
        else:
            cos, sin = image_rotary_emb
            if cos.dim() == 3:
                cos, sin = cos[0], sin[0]
            cos, sin = cos.contiguous(), sin.contiguous()

        wq = self._norm_weight(self.norm_q, dtype)
        wk = self._norm_weight(self.norm_k, dtype)
        _fused_qk_norm_rope[(triton.cdiv(n_tot, _BS), H)](
            qkv, wq, wk, wq_t, wk_t, cos, sin,
            n_tot, s_txt, qkv.stride(0),
            self.eps, H, D, _BS, cos is not None, wq is not None, joint,
            num_warps=_NUM_WARPS, num_stages=2,
        )

        qkv5 = qkv.view(1, n_tot, 3, H, D)
        hidden_states = self.attn(qkv5[:, :, 0], qkv5[:, :, 1], qkv5[:, :, 2],
                                  softmax_scale=1.0 / (D ** 0.5), causal=False)
        hidden_states = hidden_states.flatten(2, 3)

        if not joint:
            if _tp_size() > 1:
                hidden_states = _tensor_model_parallel_all_gather(hidden_states, dim=-1)
            return hidden_states
        encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
            [s_txt, s_img], dim=1)
        hidden_states = self.to_out[0](hidden_states.contiguous())
        hidden_states = self.to_out[1](hidden_states)
        encoder_hidden_states = self.to_add_out(encoder_hidden_states.contiguous())
        return hidden_states, encoder_hidden_states

    # -- reference path (baseline, verbatim) ---------------------------------

    def _apply_rope(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            cos = cos.to(query.dtype)
            sin = sin.to(query.dtype)
            query = self.rope(query, cos, sin)
            key = self.rope(key, cos, sin)
        return query, key

    def _forward_ref(self, hidden_states, encoder_hidden_states, image_rotary_emb):
        num_heads = self.to_qkv.num_heads
        num_kv_heads = self.to_qkv.num_kv_heads

        qkv = self.to_qkv(hidden_states)
        q_size = num_heads * self.head_dim
        kv_size = num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)

        query = query.unflatten(-1, (num_heads, -1))
        key = key.unflatten(-1, (num_kv_heads, -1))
        value = value.unflatten(-1, (num_kv_heads, -1))

        query = self.norm_q(query)
        key = self.norm_k(key)

        if self.added_kv_proj_dim is not None:
            add_num_heads = self.add_kv_proj.num_heads
            add_num_kv_heads = self.add_kv_proj.num_kv_heads

            encoder_qkv = self.add_kv_proj(encoder_hidden_states)
            add_q_size = add_num_heads * self.head_dim
            add_kv_size = add_num_kv_heads * self.head_dim
            encoder_query, encoder_key, encoder_value = encoder_qkv.split(
                [add_q_size, add_kv_size, add_kv_size], dim=-1
            )

            encoder_query = encoder_query.unflatten(-1, (add_num_heads, -1))
            encoder_key = encoder_key.unflatten(-1, (add_num_kv_heads, -1))
            encoder_value = encoder_value.unflatten(-1, (add_num_kv_heads, -1))

            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        query, key = self._apply_rope(query, key, image_rotary_emb)

        softmax_scale = 1.0 / (self.head_dim ** 0.5)
        hidden_states = self.attn(query, key, value, softmax_scale=softmax_scale, causal=False)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]],
                dim=1,
            )
            hidden_states = self.to_out[0](hidden_states.contiguous())
            hidden_states = self.to_out[1](hidden_states)
            encoder_hidden_states = self.to_add_out(encoder_hidden_states.contiguous())
            return hidden_states, encoder_hidden_states
        else:
            if _tp_size() > 1:
                hidden_states = _tensor_model_parallel_all_gather(hidden_states, dim=-1)
            return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self._fast_eligible(hidden_states, encoder_hidden_states, image_rotary_emb):
            return self._forward_fused(hidden_states, encoder_hidden_states,
                                       image_rotary_emb)
        return self._forward_ref(hidden_states, encoder_hidden_states, image_rotary_emb)
