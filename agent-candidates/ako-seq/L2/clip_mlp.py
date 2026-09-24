"""CLIP MLP and text embeddings (L2).

Cost model for this benchmark, measured on B200 (the timing loop flushes L2 with
a 265 MB ``zero_()`` before every timed iteration and copies each input through a
shifting pool *inside* the timed region, so both items start cold and pay for
every launch):

* one extra back-to-back kernel costs ~2.0 us of GPU time however small it is;
* one extra **aten** op costs ~4.05 us, because CPU dispatch is only partly
  hidden at these sizes;
* 18.9 MB of fp32 weights stream at ~3 TB/s for a burst this short (~6 us).

**CLIPTextEmbeddings: 3 launches -> 1, 2.11x.** ``position_ids`` is
``arange(max_position_embeddings)`` sliced to ``seq_len``, so the position rows
are the *contiguous leading slab* ``position_embedding.weight[:seq_len]``, not a
gather. One kernel does ``out[b, s, :] = tok_w[ids[b, s], :] + pos_w[s, :]``, and
Programmatic Dependent Launch hides its launch behind the harness's input copy --
it measures at the no-op-module floor plus exactly one launch, which is the floor
for any implementation that runs a kernel at all.

**CLIPMLP: left composed.** Fusing fc1 + QuickGELU + fc2 into one kernel is the
obvious move -- it would drop 5 launches to 2 and keep the 946 KB fp32
intermediate out of HBM -- but on this shape (M = 77, 768 -> 3072 -> 768) it
loses to cuBLAS. Measured GPU time for the whole module: eager 35.9 us, composed
L1 path 26.6 us (``candidate/L1/quickgelu.cu`` already collapses eager's three
activation kernels into one), best fused Triton kernel 29.7 us, best fused
TileLang kernel 27.7 us -- and both fused kernels additionally need a companion
kernel to pre-round the activations and seed the output for their atomics. The
composed path stays. ITERATIONS.md records the full sweep, the three separate
reasons the fused versions lose, and what a next attempt would have to change.

Numerics note that any future MLP kernel here must respect:
``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE`` is set in this container, so the fp32
reference resolves to a **TF32** cuBLAS kernel on both GEMM shapes. Against the
benchmark's fp32 tolerance (atol 1e-5 / rtol 1e-3 on 99% of elements) an
exact-fp32 candidate matches only 0.83 and a truncating-tf32 candidate 0.58 --
both FAIL. Rounding both operands to tf32 with round-to-nearest-even *before* the
dot matches 1.0000 (bit-exactly, for TileLang's truncating ``T.gemm``).

Both fast paths are gated on the captured dtype / shape / contiguity and fall
back to the composed L1 ops otherwise, so a miss costs correctness nothing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
from transformers import CLIPTextConfig

from ..L1.embedding import Embedding
from ..L1.linear import Linear
from ..L1.quickgelu import QuickGELU

try:  # Triton 3.6+; the intrinsic is a no-op without ``launch_pdl=True``.
    from triton.language.extra.cuda import gdc_wait
    _HAS_PDL = True
except ImportError:  # pragma: no cover - older Triton
    _HAS_PDL = False


# ---------------------------------------------------------------------------
# CLIPTextEmbeddings kernel
# ---------------------------------------------------------------------------
@triton.jit
def _embed_add(TW, PW, IDS, OUT, S, D: tl.constexpr, BD: tl.constexpr,
               ONE_ROWBLOCK: tl.constexpr, PDL: tl.constexpr):
    """out[r, :] = tok_w[ids[r], :] + pos_w[r % S, :] for r in [0, nrows).

    One row per ``gridDim.y``, one ``BD``-wide column block per ``gridDim.x``.
    The add is done in fp32 and rounded back on store, which is what
    ``TensorIterator``'s ``opmath_type`` does for the eager ``+`` -- for bf16/fp16
    inputs fp32 holds the sum exactly, so the result is bit-identical.
    """
    r = tl.program_id(1)
    rd = tl.program_id(0) * BD + tl.arange(0, BD)
    md = rd < D
    if PDL:
        # Before the first load of IDS, which the harness's input-shifting pool
        # writes in the immediately preceding kernel.
        gdc_wait()
    idx = tl.load(IDS + r).to(tl.int64)
    # position_ids is arange(), so row r of the flattened (B, S) batch takes
    # position row r % S -- and the whole modulo folds away when B == 1.
    s = r if ONE_ROWBLOCK else r % S
    t = tl.load(TW + idx * D + rd, mask=md, other=0.0).to(tl.float32)
    p = tl.load(PW + s.to(tl.int64) * D + rd, mask=md, other=0.0).to(tl.float32)
    tl.store(OUT + r.to(tl.int64) * D + rd, (t + p).to(OUT.dtype.element_ty), mask=md)


# BD / num_warps were swept through the harness's own timing loop over
# BD in {128,256,512,1024} x warps in {1,2,4,8}: every config except warps=8
# lands on the same launch-hidden plateau, so this picks the one that divides
# D = 768 exactly (3 blocks, no masked lanes).
_EMB_CFG = {"BD": 256, "warps": 2}


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------
class CLIPMLP(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.fc1 = Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.activation_fn = QuickGELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPTextEmbeddings(nn.Module):
    """Token + position embedding in one launch.

    With PDL this measures at the harness's no-op-module floor plus exactly one
    launch (~2.0 us): the launch itself hides behind the input-shifting pool's
    copy kernel, so there is nothing left to collapse. Eager pays three launches
    (~4.05 us each through this harness, being partly CPU-dispatch bound) to move
    236 KB.
    """

    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.token_embedding = Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = Embedding(config.max_position_embeddings, config.hidden_size)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )
        # Identity of the buffer we know to be arange(): the kernel bakes in
        # ``position_ids[0, s] == s``, so a caller that swapped the buffer for
        # something else must take the composed path.  Stashed in __dict__ so it
        # never becomes a second state_dict key.  ``.to()`` / ``.cuda()`` rebuild
        # the buffer *object*, so the alias is refreshed from ``_apply`` rather
        # than trusted for the module's lifetime.
        self.__dict__["_pids0"] = self.position_ids

    def _apply(self, *args, **kwargs):
        out = super()._apply(*args, **kwargs)
        self.__dict__["_pids0"] = self.position_ids
        return out

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        tw = self.token_embedding.emb.weight
        pw = self.position_embedding.emb.weight
        if (input_ids.dim() == 2 and input_ids.is_cuda
                and input_ids.dtype in (torch.int64, torch.int32)
                and input_ids.is_contiguous()
                and self.position_ids is self.__dict__["_pids0"]
                and tw.dtype is pw.dtype and tw.is_contiguous()
                and pw.is_contiguous() and tw.device == input_ids.device
                and pw.device == input_ids.device
                and tw.shape[1] == pw.shape[1]
                and not torch.is_grad_enabled()
                and input_ids.shape[-1] <= pw.shape[0]
                and input_ids.numel() <= 65535):  # one row per gridDim.y
            B, S = input_ids.shape
            D = tw.shape[1]
            BD = min(triton.next_power_of_2(D), _EMB_CFG["BD"])
            out = torch.empty((B, S, D), dtype=tw.dtype, device=tw.device)
            _embed_add[(triton.cdiv(D, BD), B * S)](
                tw, pw, input_ids, out, S, D, BD, B == 1, _HAS_PDL,
                num_warps=_EMB_CFG["warps"], launch_pdl=_HAS_PDL)
            return out
        seq_length = input_ids.shape[-1]
        position_ids = self.position_ids[:, :seq_length]
        return self.token_embedding(input_ids) + self.position_embedding(position_ids)
