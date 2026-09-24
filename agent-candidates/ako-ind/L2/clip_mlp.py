"""CLIP MLP and text embeddings (L2) -- fused Triton kernels.

CLIPMLP: Linear -> QuickGELU -> Linear (no TP, frozen encoder).
CLIPTextEmbeddings: token + position embeddings.

Both modules keep the baseline's ``__init__``/``forward`` contracts and their
submodule names, so the benchmark's ``load_state_dict`` weight sharing works.
The repacked weights are built lazily on first use and invalidated whenever the
parameters are replaced (``_apply`` / ``_load_from_state_dict``); anything the
fast path does not cover falls back to the reference op chain.

CLIPMLP
-------
At the captured shape (fp32 [1, 77, 768] -> 3072 -> 768) the op chain is launch-
and weight-bandwidth bound: ~19 MB of fp32 W1+W2 per call against only
~0.7 GFLOP of math, spread over three kernel launches (the QuickGELU alone is
three eager ops, i.e. pure dispatch).  The fused path

* partitions the 3072-wide intermediate into ``_S`` column slabs.  One CTA per
  (slab, row block) reads its own W1 slab, computes
  ``h = quickgelu(x @ W1_slab^T + b1)`` in registers, and immediately
  multiplies ``h`` by the matching K-slab of W2 -- the 77x3072 intermediate is
  never written to memory;
* reduces the ``_S`` partial 77x768 products in a second short kernel that also
  folds in ``b2`` (split-K via plain stores: fp32 global atomics measure ~5x
  worse than a second kernel here, launch included).  That second launch is
  chained with **Programmatic Dependent Launch**, which is what takes the module
  from 23.6us to 19.5us -- see below;
* repacks W1/W2 once into the slab-contiguous layout the CTAs stream
  (``w1p[s,k,j] = W1[s*SN+j,k]``, ``w2p[s,j,n] = W2[n,s*SN+j]``) so each CTA
  reads one contiguous run, and stores them as **fp16**.

Why fp16.  ``torch.backends.cuda.matmul.allow_tf32`` is True by default, so the
reference ``F.linear`` runs TF32.  fp16 and TF32 carry the same 10 explicit
mantissa bits, so fp16 operands with fp32 accumulation land on exactly the same
rounded operands as the reference (measured: an fp16 GEMM reproduces the
reference's TF32 fc1 at a matched ratio of 1.00000) -- while halving the bytes
streamed, 9.4 MB instead of 19 MB.  Higher-precision variants (exact fp32,
tf32x3, bf16/fp16 2-term splits) are *less* accurate relative to this reference
and fail its tolerance, because the reference's own TF32 error exceeds
``rtol*|y|`` wherever |y| is small.

Weights are pre-scaled by a power of two so their max magnitude is ~1024, which
keeps small weights off fp16's subnormal floor; being a power of two the scale
only shifts exponents and cannot perturb mantissa rounding.  ``x`` and ``h`` are
converted unscaled, which keeps the full fp16 normal range available above them
(overflow is destructive, subnormal underflow of a negligible term is not).

The row-block count is deliberately 5 (``_BM = 16`` at M=77, 120 CTAs): read
bandwidth on this device needs ~120 CTAs, and buying them by re-reading the
weight slab per row block is cheaper than the alternative of more slabs, which
would multiply both the ``x`` re-reads and the partial buffers.

What this op is actually bound by
---------------------------------
Not weight bandwidth.  ncu on the slab kernel reports ``dram__bytes = 9.71 MB``
at **5.8% of peak DRAM throughput** -- L2 absorbs the entire per-row-block weight
re-read (67 MB of L2 traffic behind 9.7 MB of DRAM traffic), so the redundant
reads never reach memory.  What it is bound by is issue latency:
``Active Warps Per Scheduler = 1.01`` out of 16, ``Eligible = 0.18``, one
instruction issued every 5.7 cycles.  At M=77 the grid is 120 CTAs of 4 warps on
148 SMs, so there is nothing to hide a stall behind.  Two consequences, both
measured: cutting traffic does not pay (halving the partial traffic via S=12 is
*slower*, 23.6 -> 27.7us), and the split-K reduce costs the same 4.1us whether it
reads 10.8 MB or 5.4 MB.

Which makes the second launch, not the second kernel, the thing worth removing:
an *empty* dependent kernel on this stream measures exactly +2.048us, and the
reduce measures +4.1us.  Hence PDL below.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import triton
import triton.language as tl
from transformers import CLIPTextConfig

from ..L1.embedding import Embedding
from ..L1.linear import Linear
from ..L1.quickgelu import QuickGELU

# Programmatic Dependent Launch.  The split-K reduce is a second, dependent
# launch, and at this problem size that launch costs a measured 2.048us of pure
# gap (an *empty* dependent kernel on this stream measures exactly that) on top
# of the reduce's own ~2us.  PDL lets the reduce's CTAs become resident and run
# their prologue while the slab kernel's tail drains; ``gdc_wait`` then supplies
# the ordering before the first partial is read.  Measured 23.6us -> 19.5us:
# both the launch gap and the reduce's own work disappear.
#
# Guarded twice over: if the intrinsics are missing ``_HAVE_PDL`` is False and
# the stubs below make the kernels compile to exactly r1's code, and if the
# ``launch_pdl`` launch kwarg is rejected the one-off probe in ``_pdl_ok`` turns
# the whole thing off.  Either way the result is the previous, correct kernel.
try:
    from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait
    _HAVE_PDL = True
except Exception:                                        # pragma: no cover
    _HAVE_PDL = False

    @triton.jit
    def gdc_launch_dependents():
        pass

    @triton.jit
    def gdc_wait():
        pass


_PDL = None                      # None = not probed yet


def _pdl_ok():
    """True once we have confirmed Triton accepts ``launch_pdl`` on this build."""
    global _PDL
    if _PDL is None:
        _PDL = False
        if _HAVE_PDL and os.environ.get("CLIP_MLP_NO_PDL") is None:
            try:
                q = torch.empty(1, device="cuda", dtype=torch.float32)
                _pdl_probe[(1,)](q, launch_pdl=True)
                torch.cuda.synchronize()
                _PDL = True
            except Exception:
                _PDL = False
    return _PDL


# Slab/tile configuration, tuned for the captured 77x768x3072 shape.
_S = 24            # intermediate-dim slabs; 3072 / _S must be a power of two
_BM = 16           # rows per CTA -> 5 row blocks at M=77, 120 CTAs
_BK = 128          # fc1 contraction tile
_BN = 256          # fc2 output tile
_WARPS = 4
_STAGES = 4
_RED_BH = 128      # reduce: columns per CTA
_RED_WARPS = 2
_RED_STAGES = 1    # pipelining this short streaming loop is a net loss


@triton.jit
def _pdl_probe(p):
    """Smallest kernel that exercises the launch_pdl path (never stores)."""
    if tl.program_id(0) < 0:
        tl.store(p, 0.0)
    gdc_launch_dependents()


@triton.jit
def _slab_kernel(x_ptr, w1_ptr, b1_ptr, w2_ptr, part_ptr, M,
                 K: tl.constexpr, P: tl.constexpr, SN: tl.constexpr,
                 BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
                 IW1: tl.constexpr, IW2: tl.constexpr, KST: tl.constexpr,
                 PDL: tl.constexpr):
    """One CTA per (intermediate slab, row block): fc1 -> QuickGELU -> fc2 partial."""
    s = tl.program_id(0)
    rm = tl.program_id(1) * BM + tl.arange(0, BM)
    mm = rm < M
    rsn = tl.arange(0, SN)

    acc = tl.zeros([BM, SN], dtype=tl.float32)
    w1b = w1_ptr + s * (K * SN)
    for k0 in tl.range(0, K, BK, num_stages=KST):
        rk = k0 + tl.arange(0, BK)
        xt = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=mm[:, None], other=0.0)
        wt = tl.load(w1b + rk[:, None] * SN + rsn[None, :])
        acc = tl.dot(xt.to(tl.float16), wt, acc)

    h = acc * IW1 + tl.load(b1_ptr + s * SN + rsn)[None, :]
    h = (h * tl.sigmoid(1.702 * h)).to(tl.float16)

    w2b = w2_ptr + s * (SN * P)
    pb = part_ptr + s * (M * P)
    for n0 in tl.range(0, P, BN):
        rn = n0 + tl.arange(0, BN)
        o = tl.dot(h, tl.load(w2b + rsn[:, None] * P + rn[None, :])) * IW2
        tl.store(pb + rm[:, None] * P + rn[None, :], o, mask=mm[:, None])
    if PDL:
        # Last statement: every partial store is issued, so the dependent
        # reduce may be released.
        gdc_launch_dependents()


@triton.jit
def _reduce_kernel(part_ptr, b2_ptr, out_ptr, M, S: tl.constexpr, P: tl.constexpr,
                   BH: tl.constexpr, RST: tl.constexpr, PDL: tl.constexpr):
    """Sum the S partial products and fold in b2."""
    m = tl.program_id(0)
    rn = tl.program_id(1) * BH + tl.arange(0, BH)
    off = m * P + rn
    acc = tl.load(b2_ptr + rn)          # b2 is not producer-written: load it first
    if PDL:
        gdc_wait()                      # ordering for the partials read below
    for s in tl.range(0, S, num_stages=RST):
        acc += tl.load(part_ptr + s * (M * P) + off)
    tl.store(out_ptr + off, acc)


@triton.jit
def _embed_kernel(ids_ptr, pos_id_ptr, tok_ptr, pos_ptr, out_ptr, L,
                  H: tl.constexpr, BH: tl.constexpr):
    """out[b, l, :] = tok_w[ids[b, l], :] + pos_w[position_ids[l], :], one pass, one store."""
    row = tl.program_id(0)
    rh = tl.program_id(1) * BH + tl.arange(0, BH)
    mask = rh < H
    tok = tl.load(ids_ptr + row)
    pos = tl.load(pos_id_ptr + (row % L))
    v = (tl.load(tok_ptr + tok * H + rh, mask=mask, other=0.0).to(tl.float32)
         + tl.load(pos_ptr + pos * H + rh, mask=mask, other=0.0).to(tl.float32))
    tl.store(out_ptr + row * H + rh, v.to(out_ptr.dtype.element_ty), mask=mask)


def _pow2_scale(amax: float, target: float) -> float:
    """Largest power of two ``s`` with ``amax * s <= target``."""
    if not (amax > 0.0) or not math.isfinite(amax):
        return 1.0
    return float(2.0 ** math.floor(math.log2(target / amax)))


class CLIPMLP(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.fc1 = Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.activation_fn = QuickGELU()
        self._packed = None
        self._part = None

    # --- repack cache -----------------------------------------------------
    def _invalidate(self):
        self._packed = None
        self._part = None

    def _apply(self, *args, **kwargs):
        self._invalidate()
        return super()._apply(*args, **kwargs)

    def _load_from_state_dict(self, *args, **kwargs):
        self._invalidate()
        return super()._load_from_state_dict(*args, **kwargs)

    def _pack(self):
        """Repack W1/W2 into the slab-contiguous fp16 layout the CTAs stream."""
        w1, b1 = self.fc1.weight, self.fc1.bias
        w2, b2 = self.fc2.weight, self.fc2.bias
        n, k = w1.shape
        p = w2.shape[0]
        sn = n // _S if _S else 0
        if (b1 is None or b2 is None or w2.shape[1] != n or p != k
                or w1.dtype != torch.float32 or w2.dtype != torch.float32
                or not w1.is_cuda or n % _S or sn & (sn - 1) or sn < 16
                or p % _RED_BH or p % _BN or k % _BK):
            self._packed = False        # unsupported geometry: reference path
            return
        with torch.no_grad():
            a1 = _pow2_scale(w1.abs().max().item(), 1024.0)
            a2 = _pow2_scale(w2.abs().max().item(), 1024.0)
            # w1p[s, k, j] = W1[s*sn + j, k] * a1  -> contiguous (k, sn) per slab
            w1p = (w1 * a1).view(_S, sn, k).transpose(1, 2).contiguous().to(torch.float16)
            # w2p[s, j, n] = W2[n, s*sn + j] * a2  -> contiguous (sn, p) per slab
            w2p = (w2 * a2).t().contiguous().view(_S, sn, p).contiguous().to(torch.float16)
        self._packed = (w1p, b1.contiguous(), w2p, b2.contiguous(), k, p, sn,
                        1.0 / a1, 1.0 / a2)

    def _reference(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        return self.fc2(hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._packed is None:
            self._pack()
        packed = self._packed
        if packed is False or hidden_states.dtype != torch.float32 \
                or not hidden_states.is_cuda or not hidden_states.is_contiguous() \
                or hidden_states.shape[-1] != packed[4]:
            return self._reference(hidden_states)

        w1p, b1, w2p, b2, k, p, sn, iw1, iw2 = packed
        m = hidden_states.numel() // k
        # At M == 1 the reference dispatches to a GEMV that does *not* use TF32,
        # so a TF32-precision kernel would fall outside its tolerance there.
        if m < 2:
            return self._reference(hidden_states)

        out = torch.empty(hidden_states.shape[:-1] + (p,), device=hidden_states.device,
                          dtype=torch.float32)
        part = self._part
        if part is None or part.shape[1] != m:
            part = torch.empty(_S, m, p, device=hidden_states.device, dtype=torch.float32)
            self._part = part
        pdl = _pdl_ok()
        kw = {"launch_pdl": True} if pdl else {}
        _slab_kernel[(_S, (m + _BM - 1) // _BM)](
            hidden_states, w1p, b1, w2p, part, m, k, p, sn, _BM, _BK, _BN,
            iw1, iw2, _STAGES, pdl, num_warps=_WARPS, num_stages=_STAGES, **kw)
        _reduce_kernel[(m, p // _RED_BH)](
            part, b2, out, m, _S, p, _RED_BH, _RED_STAGES, pdl,
            num_warps=_RED_WARPS, num_stages=1, **kw)
        return out


class CLIPTextEmbeddings(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.token_embedding = Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = Embedding(config.max_position_embeddings, config.hidden_size)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_length = input_ids.shape[-1]
        position_ids = self.position_ids[:, :seq_length]
        te, pe = self.token_embedding.emb, self.position_embedding.emb
        tok, pos = te.weight, pe.weight
        if (not input_ids.is_cuda or not input_ids.is_contiguous()
                or not position_ids.is_contiguous() or input_ids.dim() < 1
                or input_ids.numel() == 0 or seq_length == 0
                or te.padding_idx is not None or pe.padding_idx is not None
                or tok.dtype not in (torch.float32, torch.float16, torch.bfloat16)
                or not tok.is_contiguous() or not pos.is_contiguous()):
            return self.token_embedding(input_ids) + self.position_embedding(position_ids)

        h = tok.shape[1]
        rows = input_ids.numel()
        out = torch.empty(input_ids.shape + (h,), device=input_ids.device, dtype=tok.dtype)
        bh = 256 if h % 256 == 0 else triton.next_power_of_2(h)
        _embed_kernel[(rows, (h + bh - 1) // bh)](
            input_ids, position_ids, tok, pos, out, seq_length, h, bh,
            num_warps=4, num_stages=1)
        return out
