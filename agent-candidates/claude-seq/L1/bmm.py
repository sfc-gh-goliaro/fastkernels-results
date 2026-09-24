"""Batched matrix multiply ``(B, N, M) @ (B, M, P) -> (B, N, P)``.

A Triton tiled GEMM (bf16/fp16 in, fp32 accumulate) replaces ``torch.bmm``.

The captured shapes are tiny -- B=16 with inner dims 128/512 -- so each call
moves only a few MB and is entirely launch/latency bound rather than
compute bound.  Two things matter at that size:

* **Programmatic dependent launch.**  The grid is submitted with
  ``cudaLaunchAttributeProgrammaticStreamSerialization`` (Triton's
  ``launch_pdl``) so the CTAs are scheduled while the kernel that produced
  ``a``/``b`` is still draining; ``griddepcontrol.wait`` (``gdc_wait``)
  re-establishes the ordering before the first global load.  Without this a
  dependent launch costs ~4 us of otherwise unhidden front-end latency.
* **Per-shape tile selection.**  The grid has to cover the SM array without
  re-reading the shared operand more than necessary, which for these aspect
  ratios means very different tiles for ``K=128,P=512`` than for
  ``K=512,P=128``.

Anything this kernel cannot handle (non-contiguous, non-3D, other dtypes,
awkward divisibility) falls back to ``torch.bmm``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import gdc_wait

    _HAVE_TRITON = True
except Exception:  # pragma: no cover - Triton unavailable
    _HAVE_TRITON = False


if _HAVE_TRITON:

    @triton.jit
    def _bmm_kernel(A, Bp, C, M, sab, sam, sbb, sbk, scb, scm,
                    K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                    BK: tl.constexpr, NTM: tl.constexpr, NTN: tl.constexpr,
                    EVEN: tl.constexpr):
        pid = tl.program_id(0)
        bi = pid // (NTM * NTN)
        r = pid % (NTM * NTN)
        tm = r // NTN
        tn = r % NTN
        offm = tm * BM + tl.arange(0, BM)
        offn = tn * BN + tl.arange(0, BN)
        offk = tl.arange(0, BK)
        ap = A + bi * sab + offm[:, None] * sam + offk[None, :]
        bp = Bp + bi * sbb + offk[:, None] * sbk + offn[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        mask_m = offm < M
        gdc_wait()
        for _ in tl.range(0, K // BK):
            if EVEN:
                a = tl.load(ap)
            else:
                a = tl.load(ap, mask=mask_m[:, None], other=0.0)
            b = tl.load(bp)
            acc = tl.dot(a, b, acc)
            ap += BK
            bp += BK * sbk
        cp = C + bi * scb + offm[:, None] * scm + offn[None, :]
        out = acc.to(C.dtype.element_ty)
        if EVEN:
            tl.store(cp, out)
        else:
            tl.store(cp, out, mask=mask_m[:, None])


# (BM, BN, BK, num_warps, num_stages), tuned on a B200 for the captured shapes.
_TUNED = {
    # (K, P, M)
    (128, 512, 1): (16, 64, 128, 8, 3),
    (128, 512, 64): (32, 128, 32, 4, 5),
    (512, 128, 1): (16, 32, 256, 4, 4),
    (512, 128, 64): (32, 32, 128, 4, 4),
    (512, 128, 188): (64, 64, 64, 8, 4),
    (512, 128, 997): (128, 128, 64, 8, 4),
}

# Same two operand shapes, other row counts: tile choice only depends on which
# M bucket we are in.  (M_max, cfg), first match wins.
_BUCKETS = {
    (128, 512): ((2, (16, 64, 128, 8, 3)), (48, (32, 64, 64, 8, 3)),
                 (96, (32, 128, 32, 4, 5))),
    (512, 128): ((2, (16, 32, 256, 4, 4)), (48, (16, 32, 256, 4, 3)),
                 (96, (32, 32, 128, 4, 4)), (320, (64, 64, 64, 8, 4)),
                 (1 << 30, (128, 128, 64, 8, 4))),
}

_HEUR: dict = {}
_SMS: int | None = None
# Set to False the first time a launch rejects ``launch_pdl`` (older Triton).
_PDL = True


def _num_sms(dev) -> int:
    global _SMS
    if _SMS is None:
        _SMS = torch.cuda.get_device_properties(dev).multi_processor_count
    return _SMS


def _heuristic(M: int, K: int, P: int, sms: int):
    """Largest tile whose grid still roughly covers the machine twice over."""
    best = None
    for BM in (16, 32, 64, 128):
        # a tile more than ~2x taller than M only wastes tensor-core rows
        if BM > 16 and BM > 2 * max(16, ((M + 15) // 16) * 16):
            continue
        for BN in (16, 32, 64, 128, 256):
            if P % BN:
                continue
            for BK in (128, 64, 32):
                if K % BK:
                    continue
                ntiles = 16 * ((M + BM - 1) // BM) * (P // BN)
                score = (abs(ntiles - 2 * sms) / sms, -(BM * BN), -BK)
                if best is None or score < best[0]:
                    nw = 8 if BM * BN >= 8192 else 4
                    best = (score, (BM, BN, BK, nw, 4 if BK <= 64 else 3))
    return None if best is None else best[1]


def _pick(M: int, K: int, P: int, dev):
    cfg = _TUNED.get((K, P, M))
    if cfg is not None:
        return cfg
    for m_max, cfg in _BUCKETS.get((K, P), ()):
        if M <= m_max:
            return cfg
    key = (M, K, P)
    cfg = _HEUR.get(key)
    if cfg is None:
        cfg = _heuristic(M, K, P, _num_sms(dev))
        _HEUR[key] = cfg
    return cfg


class BatchMatMul(nn.Module):
    """Batched matrix multiply ``(B, N, M) @ (B, M, P) -> (B, N, P)``."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if (_HAVE_TRITON and a.is_cuda and a.dim() == 3 and b.dim() == 3
                and a.dtype == b.dtype
                and a.dtype in (torch.bfloat16, torch.float16)
                and a.shape[0] == b.shape[0] and a.shape[2] == b.shape[1]
                and a.is_contiguous() and b.is_contiguous()):
            B, M, K = a.shape
            P = b.shape[2]
            if M > 0 and K >= 32 and P >= 32:
                cfg = _pick(M, K, P, a.device)
                if cfg is not None:
                    return _launch(a, b, B, M, K, P, cfg)
        return torch.bmm(a, b)


def _launch(a, b, B, M, K, P, cfg):
    global _PDL
    BM, BN, BK, nw, ns = cfg
    c = torch.empty((B, M, P), device=a.device, dtype=a.dtype)
    ntm = (M + BM - 1) // BM
    ntn = P // BN
    args = (a, b, c, M, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1))
    kw = dict(K=K, BM=BM, BN=BN, BK=BK, NTM=ntm, NTN=ntn,
              EVEN=(M % BM == 0), num_warps=nw, num_stages=ns)
    grid = (B * ntm * ntn,)
    if _PDL:
        try:
            _bmm_kernel[grid](*args, launch_pdl=True, **kw)
            return c
        except TypeError:  # Triton without programmatic-dependent-launch support
            _PDL = False
    _bmm_kernel[grid](*args, **kw)
    return c
