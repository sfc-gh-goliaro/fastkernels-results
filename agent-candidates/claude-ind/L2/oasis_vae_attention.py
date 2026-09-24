"""Oasis VAE self-attention -- fused Triton implementation.

The baseline runs 26 kernels per call: two cuBLAS GEMMs (QKV + output
projection), a cuDNN flash attention, and 23 elementwise/copy kernels for the
axial rotary embedding -- which recomputes ``freqs.cos()``/``freqs.sin()`` every
call and materializes the permuted q/k views.  That rotary chain alone is ~70%
of the GPU time.  Everything here collapses into three kernels:

1. ``_qkv_rope`` -- QKV projection (TMA-fed, warp-specialized Blackwell GEMM)
   whose epilogue applies the rotary embedding to Q/K and scatters the result
   straight into the ``[3, B, H, S, D]`` layout attention wants.
2. ``_attn``     -- flash attention, writing ``[B, S, H*D]`` so the projection
   can read it as a plain row-major matrix.
3. ``_proj``     -- output projection GEMM + bias.

The rotary cos/sin tables are position-only constants, so they are materialized
once in ``__init__`` instead of per call.  They are stored one entry per rotary
*pair* (the axial freqs are ``repeat_interleave(2)``'d, so both lanes of a pair
share a cos/sin) and padded out to ``head_dim/2`` with cos=1/sin=0 for the tail
the baseline passes through untouched -- so the epilogue needs no masking and
reads half as much table as a naive per-lane layout would.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.dense_attention import DenseAttention
from ..L1.linear import Linear
from ..L1.oasis_rotary import OasisRotaryEmbedding, oasis_apply_rotary_emb

try:
    import triton
    import triton.language as tl
    from triton.tools.tensor_descriptor import TensorDescriptor
    _HAVE_TRITON = True
except Exception:  # pragma: no cover
    _HAVE_TRITON = False


if _HAVE_TRITON:

    @triton.jit
    def _tile_id(tile, num_pid_m, num_pid_n, GROUP_M: tl.constexpr):
        npg = GROUP_M * num_pid_n
        first_m = (tile // npg) * GROUP_M
        gsz = min(num_pid_m - first_m, GROUP_M)
        return (first_m + ((tile % npg) % gsz), (tile % npg) // gsz)

    @triton.jit
    def _qkv_rope(dx, dw, Bias, COS, SIN, OUT,
                  M, S: tl.constexpr, D: tl.constexpr, HD: tl.constexpr, BHSD,
                  NUM_SMS: tl.constexpr, HAS_BIAS: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  GROUP_M: tl.constexpr, WS: tl.constexpr):
        start = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n: tl.constexpr = (3 * HD) // BLOCK_N
        for tile in tl.range(start, num_pid_m * num_pid_n, NUM_SMS, flatten=True):
            pid_m, pid_n = _tile_id(tile, num_pid_m, num_pid_n, GROUP_M)
            om = pid_m * BLOCK_M
            on = pid_n * BLOCK_N

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in tl.range(HD // BLOCK_K, warp_specialize=WS):
                acc = tl.dot(dx.load([om, k * BLOCK_K]), dw.load([on, k * BLOCK_K]).T, acc)

            offs_m = om + tl.arange(0, BLOCK_M)
            offs_n = on + tl.arange(0, BLOCK_N)
            mmask = offs_m < M
            if HAS_BIAS:
                acc += tl.load(Bias + offs_n)[None, :].to(tl.float32)
            # the baseline rounds the GEMM result to fp16 before the rotary
            acc = acc.to(tl.float16).to(tl.float32)

            qkv_id = on // HD
            if qkv_id < 2:
                # freqs are repeat_interleave(2)'d, so both lanes of a rotary
                # pair share one cos/sin -- index the tables by pair, which
                # halves the table traffic versus a full [BLOCK_M, BLOCK_N] load.
                pair = ((on // 2) + tl.arange(0, BLOCK_N // 2)) % (D // 2)
                rows = (offs_m % S)[:, None] * (D // 2) + pair[None, :]
                c = tl.load(COS + rows, mask=mmask[:, None], other=1.0)
                s = tl.load(SIN + rows, mask=mmask[:, None], other=0.0)
                even, odd = tl.split(tl.reshape(acc, (BLOCK_M, BLOCK_N // 2, 2)))
                acc = tl.reshape(tl.join(even * c - odd * s, odd * c + even * s),
                                 (BLOCK_M, BLOCK_N))

            ptrs = (OUT + qkv_id * BHSD
                    + (offs_m // S)[:, None] * (HD * S) + ((offs_n % HD) // D)[None, :] * (S * D)
                    + (offs_m % S)[:, None] * D + (offs_n % D)[None, :])
            tl.store(ptrs, acc.to(tl.float16), mask=mmask[:, None])

    @triton.jit
    def _attn(Q, K, V, O, QSCALE,
              S: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, WS: tl.constexpr):
        pid_m = tl.program_id(0)
        bh = tl.program_id(1)
        base = bh * (S * D)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        offs_n = tl.arange(0, BLOCK_N)
        qm = offs_m < S
        # softmax_scale * log2(e) is folded into q so the loop is a bare exp2
        q = tl.load(Q + base + offs_m[:, None] * D + offs_d[None, :],
                    mask=qm[:, None], other=0.0)
        q = (q.to(tl.float32) * QSCALE).to(tl.float16)

        m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
        k_ptrs = K + base + offs_n[:, None] * D + offs_d[None, :]
        v_ptrs = V + base + offs_n[:, None] * D + offs_d[None, :]
        for _ in tl.range(0, S, BLOCK_N, warp_specialize=WS):
            qk = tl.dot(q, tl.load(k_ptrs).T)
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(tl.float16), tl.load(v_ptrs), acc)
            m_i = m_new
            k_ptrs += BLOCK_N * D
            v_ptrs += BLOCK_N * D

        acc = acc / l_i[:, None]
        o_ptrs = (O + (bh // H) * (S * H * D) + offs_m[:, None] * (H * D)
                  + (bh % H) * D + offs_d[None, :])
        tl.store(o_ptrs, acc.to(tl.float16), mask=qm[:, None])

    @triton.jit
    def _proj(da, dw, Bias, C, M, N: tl.constexpr, K: tl.constexpr,
              NUM_SMS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
              BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr, WS: tl.constexpr):
        start = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n: tl.constexpr = N // BLOCK_N
        for tile in tl.range(start, num_pid_m * num_pid_n, NUM_SMS, flatten=True):
            pid_m, pid_n = _tile_id(tile, num_pid_m, num_pid_n, GROUP_M)
            om = pid_m * BLOCK_M
            on = pid_n * BLOCK_N
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in tl.range(K // BLOCK_K, warp_specialize=WS):
                acc = tl.dot(da.load([om, k * BLOCK_K]), dw.load([on, k * BLOCK_K]).T, acc)
            offs_m = om + tl.arange(0, BLOCK_M)
            offs_n = on + tl.arange(0, BLOCK_N)
            acc += tl.load(Bias + offs_n)[None, :].to(tl.float32)
            tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16),
                     mask=(offs_m < M)[:, None])


_LOG2E = 1.4426950408889634

# Tuned on B200 for the two captured shapes (M = B*576 with B in {1, 6}); the
# "big"/"small" split is on rows, since M = 576 leaves parts of the GPU idle at
# the tile sizes that win for M = 3456.
# GEMM: (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, warp_specialize, warps, stages)
_QKV = (128, 128, 128, 1, True, 8, 3)
_PROJ_BIG = (128, 256, 64, 8, True, 8, 4)
_PROJ_SMALL = (128, 128, 128, 1, True, 8, 3)
# attention: (BLOCK_M, BLOCK_N, warp_specialize, warps, stages)
_ATTN_BIG = (64, 64, False, 4, 2)
_ATTN_SMALL = (64, 64, False, 4, 4)
_BIG_M = 1024


class OasisVAEAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        frame_height: int,
        frame_width: int,
        *,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = Linear(dim, dim, bias=True)
        self.rotary = OasisRotaryEmbedding(
            dim=(dim // num_heads) // 4,
            freqs_for="pixel",
            max_freq=frame_height * frame_width,
        )
        freqs = self.rotary.get_axial_freqs(frame_height, frame_width)
        self.register_buffer("rotary_freqs", freqs, persistent=False)

        seq_len = frame_height * frame_width
        head_dim = dim // num_heads
        rot_dim = freqs.shape[-1]
        # One cos/sin entry per rotary *pair*, widened to head_dim/2: the tail
        # the baseline passes through untouched gets cos=1, sin=0, so the
        # epilogue needs no masking.
        cos = torch.ones(seq_len, head_dim // 2, dtype=torch.float32)
        sin = torch.zeros(seq_len, head_dim // 2, dtype=torch.float32)
        flat = freqs.reshape(seq_len, rot_dim).to(torch.float32)[:, ::2]
        cos[:, : rot_dim // 2] = flat.cos()
        sin[:, : rot_dim // 2] = flat.sin()
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.attn = DenseAttention(backend="sdpa")
        # The kernels below are specialized for head_dim 64 (the attention tile
        # width), a 64-aligned sequence length (the attention K/V loop is
        # unmasked) and a 256-aligned model dim (the GEMM N-blocks).  The
        # pair-indexed rotary tables additionally assume the freqs really are
        # repeat_interleave(2)'d.  Anything else falls back to ``_slow_forward``.
        paired = rot_dim % 2 == 0 and torch.equal(
            freqs[..., 0::2].reshape(-1), freqs[..., 1::2].reshape(-1))
        self._fast_ok = (
            _HAVE_TRITON
            and head_dim == 64
            and dim == num_heads * head_dim
            and dim % 256 == 0
            and rot_dim <= head_dim
            and paired
            and seq_len % 64 == 0
        )
        self._wdesc_cache: dict = {}

    def _wdesc(self, key, w, block):
        ident = (w.data_ptr(), block)
        hit = self._wdesc_cache.get(key)
        if hit is not None and hit[0] == ident:
            return hit[1]
        desc = TensorDescriptor.from_tensor(w, list(block))
        self._wdesc_cache[key] = (ident, desc)
        return desc

    def _slow_forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(bsz, self.frame_height, self.frame_width, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        k = k.reshape(bsz, self.frame_height, self.frame_width, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        v = v.reshape(bsz, self.frame_height, self.frame_width, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        q = oasis_apply_rotary_emb(self.rotary_freqs, q)
        k = oasis_apply_rotary_emb(self.rotary_freqs, k)
        seq_len = self.frame_height * self.frame_width
        q = q.reshape(bsz, self.num_heads, seq_len, -1).transpose(1, 2)
        k = k.reshape(bsz, self.num_heads, seq_len, -1).transpose(1, 2)
        v = v.reshape(bsz, self.num_heads, seq_len, -1).transpose(1, 2)
        out = self.attn(q, k, v)
        out = out.reshape(bsz, seq_len, -1)
        return self.proj(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S = self.frame_height * self.frame_width
        if not (self._fast_ok and x.is_cuda and x.dtype == torch.float16
                and x.dim() == 3 and x.shape[1] == S and x.is_contiguous()):
            return self._slow_forward(x)

        bsz, _, dim = x.shape
        H = self.num_heads
        D = dim // H
        M = bsz * S
        big = M >= _BIG_M
        nsms = _num_sms(x.device)
        xf = x.reshape(M, dim)

        bm, bn, bk, gm, ws, nw, ns = _QKV
        nsm = min(nsms, triton.cdiv(M, bm) * ((3 * dim) // bn))
        bias = self.qkv.bias
        qkv = torch.empty((3, bsz, H, S, D), device=x.device, dtype=x.dtype)
        _qkv_rope[(nsm,)](
            TensorDescriptor.from_tensor(xf, [bm, bk]),
            self._wdesc("qkv", self.qkv.weight, (bn, bk)),
            bias if bias is not None else xf,
            self.rope_cos, self.rope_sin, qkv,
            M, S, D, dim, bsz * H * S * D,
            NUM_SMS=nsm, HAS_BIAS=bias is not None,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=gm, WS=ws,
            num_warps=nw, num_stages=ns,
        )

        abm, abn, aws, anw, ans = _ATTN_BIG if big else _ATTN_SMALL
        ctx = torch.empty((M, dim), device=x.device, dtype=x.dtype)
        _attn[(triton.cdiv(S, abm), bsz * H)](
            qkv[0], qkv[1], qkv[2], ctx, _LOG2E / (D ** 0.5),
            S, H, D, BLOCK_M=abm, BLOCK_N=abn, WS=aws,
            num_warps=anw, num_stages=ans,
        )

        bm, bn, bk, gm, ws, nw, ns = _PROJ_BIG if big else _PROJ_SMALL
        nsm = min(nsms, triton.cdiv(M, bm) * (dim // bn))
        out = torch.empty((M, dim), device=x.device, dtype=x.dtype)
        _proj[(nsm,)](
            TensorDescriptor.from_tensor(ctx, [bm, bk]),
            self._wdesc("proj", self.proj.weight, (bn, bk)),
            self.proj.bias, out, M, dim, dim,
            NUM_SMS=nsm, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=gm, WS=ws,
            num_warps=nw, num_stages=ns,
        )
        return out.view(bsz, S, dim)


_SMS: dict = {}


def _num_sms(device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    n = _SMS.get(idx)
    if n is None:
        n = torch.cuda.get_device_properties(idx).multi_processor_count
        _SMS[idx] = n
    return n
