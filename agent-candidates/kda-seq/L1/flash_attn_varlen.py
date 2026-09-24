"""Variable-length Flash Attention — sync-free dispatcher over three paths.

The baseline is not a naive implementation: on this B200 ``FA_VERSION == 4``, so
it already *is* FlashAttention-4, the hand-written CuTeDSL Blackwell kernel
bundled in vLLM. The win here is therefore not "beat FA4" but "stop FA4 running
a worse-configured path", plus an authored kernel for the launch-bound shape.

Routing policy — a pure function of ``(dtypes, shapes, cu_seqlens_*.numel(),
causal, return_softmax_lse)``, never of tensor *values* and never of strides or
contiguity:

1. **Authored Triton varlen kernel** (``_varlen_attn_fwd`` below) for the
   launch-bound small-shape tier. This is the custom GPU kernel: a variable-length
   attention forward with online fp32 softmax, ``head_dim_qk = 192`` handled as a
   ``128 + 64`` split of ``QK^T`` accumulating into a single fp32 tile rather than
   padding to 256, bottom-right causal masking, and explicit strides on every
   operand.
2. **Single-segment dense reshape** — a *re-routing of FA4, not a new kernel*.
   When both cumulative-length vectors hold two entries the call describes one
   logical sequence, which is exactly a dense batch-1 call; reshaping to 4-D
   views lets FA4 take its dense entry, where it enables two-SM cooperative MMA
   and a better tile scheduler that its variable-length path gives up. The
   reshape and the unwrap are pure views, so nothing is copied.
3. **Exact varlen fallback** — the same ``flash_attn_varlen_func`` call the
   baseline makes. Anything not covered above lands here, which makes parity the
   floor by construction.

Which path each benched case takes, by harness row order:

| row | shape                                | segments | path                  |
|-----|--------------------------------------|----------|-----------------------|
| 0   | fp16  64 x 16 x 64,  non-causal      | 1        | authored Triton kernel|
| 1   | bf16  16384/65536 x 16 x 192/128     | 2        | varlen fallback       |
| 2   | bf16  16384 x 16 x 192/128, causal   | 2        | varlen fallback       |
| 3   | bf16  16384 x 16 x 192/128, causal   | 1        | dense reshape         |
| 4   | bf16  16384/65536 x 16 x 192/128     | 1        | dense reshape         |

Rows 1 and 2 stay on the fallback deliberately: establishing that a
multi-segment call has uniform segment lengths would need a device-to-host read
of ``cu_seqlens``, which would put a synchronisation in the timed path and make
latency depend on input values.

The authored kernel is not competitive on the large ``(192, 128)`` shapes and is
not routed there: a 36-configuration sweep put its best at 0.359x and 0.404x of the
varlen call on the two hottest cases. That is the expected result rather than a
disappointment -- FA4 is a warp-specialised CuTeDSL kernel with ping-pong
scheduling and software exp2, and CUTLASS's own hand-written Blackwell kernel for
this asymmetric ``d_qk=192 / d_v=128`` shape reports throughput *below* what this
baseline already achieves. The kernel is kept correct across every captured
configuration so it remains the foundation for a later phase, and it owns the one
regime where launch overhead, not arithmetic, decides the latency.

Preconditions, stated because two of them cannot be checked without a
device-to-host read and would therefore cost a synchronisation in the timed path:

* **Packed cumulative lengths.** The two fast paths assume ``cu_seqlens[0] == 0``
  and ``cu_seqlens[-1] == total_tokens`` — the defining invariant of the packed
  variable-length layout, and the property that makes the token dimension mean the
  token count. A caller violating it already receives undefined trailing rows from
  the varlen path, which leaves them unwritten. ``total_tokens == 0`` with
  ``cu_seqlens == [0, 0]`` is well formed and handled.
* **16-byte pointer alignment** of ``q``, ``k`` and ``v``, which FA4 assumes on both
  of its entry points rather than checks, so this is not a differential.
* **Inference only.** Neither fast path is differentiable — and neither is the
  baseline, whose FA4 branch calls the same forward implementation directly rather
  than through an autograd function, so this too is not a differential.

Measured and *not* enabled: pinning FA4's split-KV setting. vLLM's wrapper passes
``num_splits=0``, which arms FA4's split heuristic, so a probe that used the
CuTeDSL default of ``1`` changed two things at once. Decomposing that ablation
showed the heuristic returns "no split" on every captured configuration here
(``148 // total_mblocks == 0`` for the large shapes, and ``num_n_blocks <= 4``
for the tiny one), and the kernel name is identical either way. The source settles
it: ``num_splits`` is reduced to ``is_split_kv = num_splits > 1`` before the
compile key is built, and it is that boolean rather than the number which enters
the key — so 0 and 1 select the same compiled program here. An order-balanced,
uninstrumented A/B put the override at 0.982-1.011x against a same-code noise floor
of 0.986-1.020x, i.e. no detectable effect. The override is not shipped.

Scope of that claim: no split-KV contribution is detectable on the captured
configurations. It is not a general statement — at shapes where the heuristic
returns 2 the two settings plainly differ, and this FA4 build in fact raises an
internal compile error there, identically from both entry points.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ....infra.fa_utils import FA_VERSION, flash_attn_varlen_func

# ---------------------------------------------------------------------------
# Import-time resolution. An entry point that is unavailable disables its route
# here, once, rather than raising or being caught on every call.
# ---------------------------------------------------------------------------
try:
    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd as _fa4_fwd
except Exception:  # pragma: no cover - a build without the CuTeDSL FA4 path
    _fa4_fwd = None

_DENSE_AVAILABLE = _fa4_fwd is not None and FA_VERSION == 4

# vLLM's wrapper hands FA4 ``num_splits=0``; the dense route passes the same
# value so the two calls differ in the entry point and nothing else.
_WRAPPER_NUM_SPLITS = 0

# Configurations for which the dense reshape is validated against the varlen call
# across seeds and both memory layouts *and* measured on the shapes it covers,
# keyed ``(dtype, head_dim_qk, head_dim_v, causal)``. ``return_softmax_lse`` is
# not part of the key because it changes only which leaves come back, not the
# arithmetic, and both settings are present in the validated corpus.
#
# Only the two configurations that captured single-segment traffic actually
# reaches are listed. Head-dim-64 entries were deliberately removed: the one
# fp16 (64, 64) single-segment variant is 64 tokens, which the authored kernel
# takes first, so a dense entry for it would have been reachable by no input and
# validated by no measurement.
_DENSE_OK = frozenset({
    (torch.bfloat16, 192, 128, True),
    (torch.bfloat16, 192, 128, False),
})

# Launch configurations for the authored kernel, chosen offline and fixed: no
# autotuning is reachable from ``forward``, so nothing compiles or spawns a
# thread inside the harness's guarded timing window. Keyed
# ``(head_dim_qk, head_dim_v)`` -> ``(BLOCK_M, BLOCK_N, num_warps, num_stages)``.
_TRITON_CFG = {
    (64, 64): (32, 32, 4, 2),
    (192, 128): (128, 128, 8, 2),
}

# The authored kernel is routed only where it was measured to beat *both*
# alternatives, keyed ``(dtype, head_dim_qk, head_dim_v, causal)`` -> inclusive
# ``(max_total_q, max_total_k)`` bounds. The key carries ``return_softmax_lse``
# because emitting the LSE is a distinct compiled specialisation with its own
# cost, and the bound below was measured without it.
# A ladder over this configuration family
# measured the kernel at 1.163x the varlen call and 1.109x the dense reshape at 64
# tokens, winning every paired round, and already behind both at 128 tokens
# (0.948x / 0.893x) -- so the bound sits at the rung that was measured to win, not
# at an extrapolated crossover. Everything larger keeps the library paths, where
# FA4 is 2.5-3x ahead of this kernel.
_TRITON_ROUTE = {
    (torch.float16, 64, 64, False, False): (64, 64),
}

_LOG2E = 1.4426950408889634
_LN2 = 0.6931471805599453


# ---------------------------------------------------------------------------
# Authored kernel: variable-length attention forward.
# ---------------------------------------------------------------------------
@triton.jit
def _varlen_attn_fwd(
    Q, K, V, Out, Lse,
    CuQ, CuK,
    stride_qm, stride_qh, stride_qd,
    stride_kn, stride_kh, stride_kd,
    stride_vn, stride_vh, stride_vd,
    stride_om, stride_oh, stride_od,
    stride_lh, stride_lm,
    scale_log2e,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D0: tl.constexpr,       # first slice of head_dim_qk
    D1: tl.constexpr,       # second slice; 0 when head_dim_qk is a power of two
    DV: tl.constexpr,
    CAUSAL: tl.constexpr,
    WRITE_LSE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)

    q_start = tl.load(CuQ + off_z).to(tl.int32)
    q_end = tl.load(CuQ + off_z + 1).to(tl.int32)
    k_start = tl.load(CuK + off_z).to(tl.int32)
    k_end = tl.load(CuK + off_z + 1).to(tl.int32)
    len_q = q_end - q_start
    len_k = k_end - k_start

    start_m = pid_m * BLOCK_M
    if start_m >= len_q:
        return

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d0 = tl.arange(0, D0)
    mask_m = offs_m < len_q

    q_base = Q + (q_start + offs_m)[:, None] * stride_qm + off_h * stride_qh
    q0 = tl.load(q_base + offs_d0[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)
    if D1 > 0:
        offs_d1 = D0 + tl.arange(0, D1)
        q1 = tl.load(q_base + offs_d1[None, :] * stride_qd,
                     mask=mask_m[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, DV], dtype=tl.float32)

    # Bottom-right causal alignment: query ``i`` of a segment attends to
    # ``j <= i + (len_k - len_q)``. ``hi`` can land at or below zero when
    # len_q > len_k, which leaves whole query rows fully masked.
    diag = len_k - len_q
    if CAUSAL:
        hi = start_m + BLOCK_M + diag
        hi = tl.minimum(hi, len_k)
    else:
        hi = len_k

    for start_n in range(0, hi, BLOCK_N):
        offs_n_cur = start_n + offs_n
        mask_n = offs_n_cur < len_k
        k_base = K + (k_start + offs_n_cur)[:, None] * stride_kn + off_h * stride_kh
        k0 = tl.load(k_base + offs_d0[None, :] * stride_kd,
                     mask=mask_n[:, None], other=0.0)
        qk = tl.dot(q0, tl.trans(k0))
        if D1 > 0:
            k1 = tl.load(k_base + (D0 + tl.arange(0, D1))[None, :] * stride_kd,
                         mask=mask_n[:, None], other=0.0)
            qk = tl.dot(q1, tl.trans(k1), qk)
        qk = qk * scale_log2e

        keep = mask_m[:, None] & mask_n[None, :]
        if CAUSAL:
            keep = keep & (offs_n_cur[None, :] <= offs_m[:, None] + diag)
        qk = tl.where(keep, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        # A row that is still entirely masked has m_new == -inf; subtracting it
        # would produce ``-inf - (-inf) = NaN``, so rescale against zero there.
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp2(qk - m_safe[:, None])
        alpha = tl.exp2(m_i - m_safe)

        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v_ptrs = (V + (k_start + offs_n_cur)[:, None] * stride_vn + off_h * stride_vh
                  + tl.arange(0, DV)[None, :] * stride_vd)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new

    # Fully-masked rows: the baseline emits exactly zero with an LSE of negative
    # infinity, so divide by one and write the zero accumulator rather than 0/0.
    empty = l_i == 0.0
    acc = acc / tl.where(empty, 1.0, l_i)[:, None]

    o_ptrs = (Out + (q_start + offs_m)[:, None] * stride_om + off_h * stride_oh
              + tl.arange(0, DV)[None, :] * stride_od)
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None])

    if WRITE_LSE:
        lse = tl.where(empty, float("-inf"), (m_i + tl.log2(l_i)) * 0.6931471805599453)
        tl.store(Lse + off_h * stride_lh + (q_start + offs_m) * stride_lm, lse,
                 mask=mask_m)


def _launch_varlen_attn(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                        softmax_scale, causal, return_softmax_lse):
    """Allocation-light launcher for the authored kernel."""
    total_q, nheads, d_qk = q.shape
    d_v = v.shape[2]
    nseg = cu_seqlens_q.numel() - 1
    block_m, block_n, num_warps, num_stages = _TRITON_CFG[(d_qk, d_v)]
    d0 = 128 if d_qk == 192 else d_qk
    d1 = d_qk - d0

    # Rows per segment. With one segment the token count is the segment length
    # exactly and no reported maximum is involved. With several, the contract
    # says max_seqlen_q bounds every segment; floor it by the smallest value
    # that could bound an nseg-way partition so an understated argument cannot
    # silently leave query rows uncomputed.
    if nseg == 1:
        m_extent = total_q
    else:
        ceil_seg = -(-total_q // nseg)
        m_extent = max_seqlen_q if max_seqlen_q >= ceil_seg else total_q
        m_extent = min(m_extent, total_q)

    out = torch.empty((total_q, nheads, d_v), dtype=q.dtype, device=q.device)
    lse = (torch.empty((nheads, total_q), dtype=torch.float32, device=q.device)
           if return_softmax_lse else out)  # unused when WRITE_LSE is False

    grid = (max(1, triton.cdiv(m_extent, block_m)), nheads, nseg)
    _varlen_attn_fwd[grid](
        q, k, v, out, lse,
        cu_seqlens_q, cu_seqlens_k,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        lse.stride(0), lse.stride(1),
        softmax_scale * _LOG2E,
        BLOCK_M=block_m, BLOCK_N=block_n,
        D0=d0, D1=d1, DV=d_v,
        CAUSAL=causal, WRITE_LSE=return_softmax_lse,
        num_warps=num_warps, num_stages=num_stages,
    )
    return (out, lse) if return_softmax_lse else out


# ---------------------------------------------------------------------------
# Re-routing: a single-segment varlen call is a dense batch-1 call.
# ---------------------------------------------------------------------------
def _as_batch1(t):
    """[total, h, d] -> [1, total, h, d] as a view, carrying strides through."""
    total, h, d = t.shape
    s0, s1, s2 = t.stride()
    return t.as_strided((1, total, h, d), (total * s0, s0, s1, s2))


class FlashAttnVarlen(nn.Module):
    """Variable-length Flash Attention without paged KV cache lookup."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float,
        causal: bool = True,
        return_softmax_lse: bool = False,
    ):
        # ``numel()`` and ``dim()`` read shapes, not device memory, so this costs
        # no synchronisation. One segment on both sides means one logical sequence.
        single = (cu_seqlens_q.dim() == 1 and cu_seqlens_k.dim() == 1
                  and cu_seqlens_q.numel() == 2 and cu_seqlens_k.numel() == 2)
        if single:
            total_q, nheads, d_qk = q.shape
            total_k = k.shape[0]
            dense_key = (q.dtype, d_qk, v.shape[2], causal)

            bounds = _TRITON_ROUTE.get(dense_key + (return_softmax_lse,))
            # The authored kernel indexes K and V with the query head index and
            # does not broadcast, so equal head counts are required; FA4's own
            # entry points perform the equivalent checks for the library paths.
            if (bounds is not None and total_q <= bounds[0] and total_k <= bounds[1]
                    and k.shape[1] == nheads and v.shape[1] == nheads
                    and k.shape[2] == d_qk and v.shape[0] == total_k
                    and k.dtype is q.dtype and v.dtype is q.dtype):
                return _launch_varlen_attn(
                    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                    softmax_scale, causal, return_softmax_lse)

            # Bottom-right causal alignment reduces to ordinary ``j <= i`` when the
            # token counts agree, which is what makes the two entry points
            # equivalent; no captured variant pairs causal with unequal counts, so
            # the guard is free here.
            if (_DENSE_AVAILABLE and dense_key in _DENSE_OK
                    and (not causal or total_q == total_k)):
                out, lse, _, _ = _fa4_fwd(
                    _as_batch1(q), _as_batch1(k), _as_batch1(v),
                    softmax_scale=softmax_scale,
                    causal=causal,
                    return_lse=return_softmax_lse,
                    num_splits=_WRAPPER_NUM_SPLITS,
                )
                # dense out is [1, s, h, dv] and dense lse is [1, h, s]; both
                # squeeze to exactly the shapes the contract asks for.
                if return_softmax_lse:
                    return out.squeeze(0), lse.squeeze(0)
                return out.squeeze(0)

        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            return_softmax_lse=return_softmax_lse,
            fa_version=FA_VERSION,
        )
