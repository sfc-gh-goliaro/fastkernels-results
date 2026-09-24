"""Feed-forward blocks for encoder models.

Both blocks are ``GEMM -> pointwise tail``.  The GEMMs are the captured shapes

    EncoderIntermediate   [M, 1024] x [1024, 4096]
    EncoderOutput         [M, 4096] x [4096, 1024]        M in {64, 512, 2048}

and on this GPU (B200) they are the one part of the op that is *already* at the
wall: cuBLAS' ``nvjet_sm100`` kernels run the M=2048 ``EncoderIntermediate``
GEMM at 1.19 PFLOP/s, ~53% of the 2.25 PFLOP/s the machine does on fp16 dense.
A Triton ``tl.dot`` kernel does reach the 5th-gen tensor cores here (the PTX
carries ``tcgen05.mma``), but measured across ~700 tile/stage/warp/cluster
configurations -- pointer loads, device-side TMA, host-side TMA, warp
specialisation, ``num_ctas`` 2/4 for multicast -- the best of them lands at
0.6-0.8x of cuBLAS on every captured shape.  What Triton has no way to express
is the 2-SM cooperative ``tcgen05`` tile that nvjet uses, and with these N/K the
single-CTA alternatives are L2-bandwidth bound long before the MMA saturates.
So the GEMMs stay on the reference path and the work goes into the tails.

What is left to win is *passes and launches*, and both matter more than they
look.  Under the scorer's timing loop (L2 flushed before every iteration, one
CUDA-event pair per call) a kernel boundary costs ~4 us of measured time on this
machine -- an empty Triton kernel measures 7.2 us, two measure 10.4, four 18.5.
On the small captured shapes that dwarfs the arithmetic.

EncoderOutput
-------------
The baseline tail is three launches: ``dense``, ``+ input_tensor``, LayerNorm --
and the add is a whole extra round trip (read 2 x MxN, write MxN) whose result
LayerNorm immediately reads back.  :func:`_add_ln` folds the residual add, the
fp32 reduction and the affine into one kernel that touches each row once, in
place over the GEMM's output buffer, so the tail is one launch and one pass:

                          launches      MxN traffic (M=2048)
    baseline              3             25.2 MB
    frozen L1 LayerNorm   3             25.2 MB
    this file             1             12.6 MB

Measured against the baseline (fp16, captured shapes, L2-flushed):
1.38x at M=64, 1.46x at M=512, 1.51x at M=2048 -- versus 1.09x / 1.16x / 1.21x
for the frozen L1 LayerNorm on the same GEMM.

EncoderIntermediate
-------------------
Left on the frozen L1 path (``Linear`` -> cuBLAS, then the L1 CUDA GELU), which
is already the floor.  GELU cannot be folded into a cuBLAS epilogue from here,
so the tail is one unavoidable extra pass, and that pass is *faster than a plain
copy of the same bytes*: at M=2048 the GELU launch adds 9.4 us over the GEMM
while ``Tensor.copy_`` of the same 16.8 MB tensor costs 11.6 us.  A fused
GEMM+bias+GELU kernel (one launch instead of two, 33.6 MB less traffic) was
written and tuned anyway; it loses to cuBLAS+GELU by 0.5-4 us on all three
shapes because the GEMM it replaces is 1.3-2x slower, so it is not used.

Numerics
--------
``_add_ln`` reproduces the baseline's rounding exactly: the residual sum is
rounded to the input dtype before the reduction (the baseline's ``+`` produces
an fp16 tensor, which ``F.layer_norm`` then reads), mean/variance and the affine
are fp32, and the result is rounded once on the store.  Max deviation from the
baseline over the captured shapes is 1.0e-3 (M=64/512) and 2.0e-3 (M=2048),
5-10x inside the scorer's fp16 tolerance.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.gelu import GELU
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover -- no Triton: the reference tail is used
    triton = None


if triton is not None:

    @triton.jit
    def _add_ln_kernel(Y, R, G, BE, O, M, N, eps,
                       RB: tl.constexpr, BN: tl.constexpr, EVEN: tl.constexpr,
                       HAS_G: tl.constexpr, HAS_BE: tl.constexpr):
        """out[r] = layer_norm(Y[r] + R[r]) for RB rows per program.

        A program owns whole rows, so the row never leaves registers between the
        reduction and the write-back and ``O`` may alias ``Y`` (every load of the
        tile is issued before any store to it).
        """
        rows = tl.program_id(0) * RB + tl.arange(0, RB)
        cols = tl.arange(0, BN)
        off = rows[:, None].to(tl.int64) * N + cols[None, :]
        if EVEN:
            v = tl.load(Y + off).to(tl.float32) + tl.load(R + off).to(tl.float32)
        else:
            keep = rows[:, None] < M
            v = (tl.load(Y + off, mask=keep, other=0.0).to(tl.float32)
                 + tl.load(R + off, mask=keep, other=0.0).to(tl.float32))
        # The baseline adds the residual in the input dtype and hands the
        # rounded tensor to F.layer_norm; round here too so the reduction sees
        # the same values.
        v = v.to(Y.dtype.element_ty).to(tl.float32)
        mean = tl.sum(v, 1) / N
        d = v - mean[:, None]
        o = d * tl.rsqrt(tl.sum(d * d, 1) / N + eps)[:, None]
        if HAS_G:
            o = o * tl.load(G + cols).to(tl.float32)[None, :]
        if HAS_BE:
            o = o + tl.load(BE + cols).to(tl.float32)[None, :]
        o = o.to(O.dtype.element_ty)
        if EVEN:
            tl.store(O + off, o)
        else:
            tl.store(O + off, o, mask=rows[:, None] < M)

    def _add_ln(y, r, g, be, eps):
        """In-place ``layer_norm(y + r)`` over the last axis of a 2-D fp16/bf16 y."""
        M, N = y.shape
        # ~2 K elements per program: enough for 32 B per thread at 4 warps, and
        # still one program per row-pair so a 64-row case keeps 32 SMs busy.
        # Swept RB in 1..16 x num_warps in 2..8: RB=1 wins below a few hundred
        # rows, RB=2048/N above, and the rest are within 2%.
        rb = 1 if M < 256 else max(1, 2048 // N)
        _add_ln_kernel[(triton.cdiv(M, rb),)](
            y, r, g if g is not None else y, be if be is not None else y, y,
            M, N, eps, RB=rb, BN=N, EVEN=(M % rb == 0),
            HAS_G=g is not None, HAS_BE=be is not None,
            num_warps=4, num_stages=1)
        return y


_FAST_DTYPES = (torch.float16, torch.bfloat16)


class EncoderIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.intermediate_act_fn = GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.intermediate_act_fn(self.dense(hidden_states))


class EncoderOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.intermediate_size, config.hidden_size, bias=True)
        # promote_fp32=False: vLLM's bert.py / roberta.py use a plain
        # nn.LayerNorm here (see encoder_embeddings for the full rationale).
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps,
                                   promote_fp32=False)

    def _fast(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> bool:
        if triton is None or not hidden_states.is_cuda:
            return False
        if hidden_states.dtype not in _FAST_DTYPES:
            return False
        if input_tensor.dtype is not hidden_states.dtype:
            return False
        if self.LayerNorm.promote_fp32 or torch.is_grad_enabled():
            return False
        if self.dense.bias is None or not input_tensor.is_contiguous():
            return False
        if input_tensor.numel() == 0:
            return False
        n = self.LayerNorm.normalized_shape[0]
        # BN == N must be a Triton block size, and the whole row is held in
        # registers, so cap it where that stops being free.
        if input_tensor.shape[-1] != n or n & (n - 1) or not 16 <= n <= 8192:
            return False
        for p in (self.LayerNorm.weight, self.LayerNorm.bias):
            if p is not None and p.dtype is not hidden_states.dtype:
                return False
        return True

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        if self._fast(hidden_states, input_tensor):
            y = torch.addmm(self.dense.bias,
                            hidden_states.reshape(-1, hidden_states.shape[-1]),
                            self.dense.weight.t())
            _add_ln(y, input_tensor.reshape(-1, y.shape[1]),
                    self.LayerNorm.weight, self.LayerNorm.bias, self.LayerNorm.eps)
            return y.view(input_tensor.shape)
        return self.LayerNorm(self.dense(hidden_states) + input_tensor)
