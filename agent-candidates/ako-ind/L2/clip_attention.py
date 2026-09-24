"""CLIP self-attention (L2) -- collapsed to a 3-launch chain.

The captured workload is one tiny fp32 shape (B=1, S=77, D=768, 12 heads x 64),
~380 MFLOP.  Measured on B200, the whole GPU chain of the baseline takes ~38us
while its *host* side takes ~195us: a cuBLAS call costs ~13us of CPU and an
elementwise op ~5us, so wall time is set by dispatch count, not math.  The
harness times each call after an L2 flush (a ~70us memset), which hides the
first ~70us of host work -- so the goal is to get total host work under that
runway, at which point the measured time collapses to the GPU time.

This kernel therefore issues exactly three launches:
  1. one packed [2304, 768] QKV GEMM (was three skinny [768, 768] ones), built
     lazily because weights arrive via ``load_state_dict`` *after* ``__init__``,
     with ``scale`` folded into the Q rows -- 0.125 is a power of two, so the
     fold is bit-exact and the ``* self.scale`` launch disappears;
  2. one fused attention kernel (``clip_attn.cu``) covering QK^T, the additive
     mask, the row softmax and P@V, writing straight into the head-concatenated
     [B, S, 768] layout ``out_proj`` consumes, so the score/prob temporaries,
     the ``.transpose(1, 2).contiguous()`` copy and the no-op ``.float()`` /
     ``.to()`` casts are all gone;
  3. one out_proj GEMM.

Q/K/V are never materialized as separate tensors: the fused kernel reads them
out of the packed buffer with strided (float4) loads.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextConfig

from ..L1.linear import BMM, Linear
from ..L1.softmax import Softmax

# --- fused attention extension ------------------------------------------------
# Compiled once per source revision (the content hash is part of the extension
# name, so an edited .cu can never be served from a stale build cache) and
# reused by every benchmark subprocess.
_SRC = Path(__file__).resolve().parent / "clip_attn.cu"
_FUSED = None
if _SRC.is_file():
    try:
        from torch.utils.cpp_extension import load as _load

        os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"  # B200; skip the 6-arch fat build
        _tag = hashlib.sha1(_SRC.read_bytes()).hexdigest()[:8]
        _FUSED = _load(
            name=f"clip_attn_{_tag}",
            sources=[str(_SRC)],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            extra_cflags=["-O3"],
            verbose=False,
        ).clip_fused_attn
    except Exception:  # pragma: no cover - fall back to the torch path
        _FUSED = None

_MAX_FUSED_SEQ = 320  # shared memory holds K and V for the whole sequence
# For S <= 2 torch's attention matmuls degenerate to non-tensor-core paths, so
# the baseline is exact fp32 there and the TF32-matching core would not match;
# such shapes are never benchmarked, so they take the reference path.
_MIN_FUSED_SEQ = 4


def _invalidate_pack(module, incompatible_keys):  # load_state_dict post hook
    module._packed = None


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
        self.register_load_state_dict_post_hook(_invalidate_pack)

    def _pack(self):
        """Concatenate q/k/v weights+biases once, with ``scale`` folded into Q.

        Also caches everything the hot path reads, so a forward touches one
        attribute instead of a dozen.
        """
        with torch.no_grad():
            w = torch.cat((self.q_proj.weight * self.scale,
                           self.k_proj.weight,
                           self.v_proj.weight), 0).contiguous()
            b = torch.cat((self.q_proj.bias * self.scale,
                           self.k_proj.bias,
                           self.v_proj.bias), 0).contiguous()
        # The fused kernel is fp32/CUDA/head_dim-64 only; anything else (a
        # bf16-cast module, CPU, another head size) takes the reference path
        # rather than raising out of the extension.
        fused = (_FUSED is not None and self.head_dim == 64 and w.is_cuda
                 and w.dtype == torch.float32)
        # Handed to the fused kernel as an L2 prefetch target.  cuBLAS runs the
        # out_proj GEMM on 12 of 148 SMs, so it is latency-bound on its cold
        # weight fetch; the fused attention kernel runs immediately before it and
        # is itself latency-bound with spare memory slots, so pulling those 2.36MB
        # into L2 there is close to free.  The same prologue is what PDL below
        # overlaps with the QKV GEMM's tail.
        ow = self.out_proj.weight
        pref = ow if (fused and ow.is_contiguous() and ow.is_cuda) else None
        # Match whatever precision the *baseline*'s cuBLAS matmuls use: this
        # environment force-enables TF32, and an exact-fp32 core would differ
        # from a TF32 baseline by ~6.6e-4 relative -- outside the harness's
        # (1e-5, 1e-3) fp32 tolerance.
        tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
        self._packed = (w, b, ow, self.out_proj.bias,
                        self.num_heads, fused, tf32, pref)
        return self._packed

    def _attn_reference(self, qkv, attention_mask, batch_size, seq_length):
        """Generic path (head_dim != 64, very long sequences, exotic masks)."""
        nh, hd = self.num_heads, self.head_dim
        qkv = qkv.view(batch_size, seq_length, 3, nh, hd).permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        attn_weights = torch.matmul(queries, keys.transpose(-1, -2))
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = torch.softmax(attn_weights, -1)
        attn_output = torch.matmul(attn_weights, values).transpose(1, 2)
        return attn_output.reshape(batch_size, seq_length, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        packed = self._packed
        if packed is None:
            packed = self._pack()

        qkv = F.linear(hidden_states, packed[0], packed[1])
        seq = qkv.size(1)
        if (packed[5] and seq <= _MAX_FUSED_SEQ and seq >= _MIN_FUSED_SEQ
                and (attention_mask is None or attention_mask.dim() == 4)):
            # pref_mode=1 (L2 hint), pdl=True: the fused kernel is launched with
            # programmatic stream serialization so its grid setup and out_proj
            # weight prefetch overlap the QKV GEMM's tail, and it waits on
            # cudaGridDependencySynchronize() just before its first qkv read.
            # Measured together: -4.2us of span vs neither.
            attn = _FUSED(qkv, attention_mask, packed[4], packed[6], packed[7],
                          1, True, 1)
        else:
            shape = hidden_states.shape
            attn = self._attn_reference(qkv, attention_mask, shape[0], shape[1])
        return F.linear(attn, packed[2], packed[3])
