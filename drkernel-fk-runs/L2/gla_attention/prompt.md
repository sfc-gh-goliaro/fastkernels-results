You write custom Triton kernels to replace the pytorch operators in the given architecture to get speedups.

    You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom Triton kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.


        Here's an example to show you the syntax of inline embedding custom Triton kernels in torch: The example given architecture is:

        ```
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, a, b):
                return a + b

        def get_inputs():
            # randomly generate input tensors based on the model architecture
            a = torch.randn(1, 128).cuda()
            b = torch.randn(1, 128).cuda()
            return [a, b]

        def get_init_inputs():
            # randomly generate tensors required for initialization based on the model architecture
            return []
        ```

        The example new arch with custom Triton kernels looks like this:
        ```
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import triton
        import triton.language as tl

        @triton.jit
        def add_kernel(
            x_ptr,  # Pointer to first input
            y_ptr,  # Pointer to second input
            out_ptr,  # Pointer to output
            n_elements,  # Total number of elements in input/output
            BLOCK_SIZE: tl.constexpr,
        ):
            # Each program handles a contiguous block of data of size BLOCK_SIZE
            block_start = tl.program_id(0) * BLOCK_SIZE
            # Create a range of offsets [0..BLOCK_SIZE-1]
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            # Mask to ensure we don't go out of bounds
            mask = offsets < n_elements
            # Load input values
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
            # Perform the elementwise addition
            out = x + y
            # Store the result
            tl.store(out_ptr + offsets, out, mask=mask)

        def triton_add(x: torch.Tensor, y: torch.Tensor):
            """
            This function wraps the Triton kernel call. It:
              1. Ensures the inputs are contiguous on GPU.
              2. Calculates the grid (blocks) needed.
              3. Launches the Triton kernel.
            """
            assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA."
            x = x.contiguous()
            y = y.contiguous()

            # Prepare output tensor
            out = torch.empty_like(x)

            # Number of elements in the tensor
            n_elements = x.numel()
            BLOCK_SIZE = 128  # Tunable parameter for block size

            # Determine the number of blocks needed
            grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

            # Launch the Triton kernel
            add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
            return out

        class ModelNew(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, a, b):
                # Instead of "return a + b", call our Triton-based addition
                return triton_add(a, b)
        ```
        
    You are given the following architecture:
    ```
from fastkernels.infra.cuda_ext import lazy_op
from fla.ops.gla import chunk_gla
from fla.ops.gla import fused_recurrent_gla
from fla.ops.retention import chunk_retention
from fla.ops.retention import fused_recurrent_retention
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)

def _compute_scaled_inv_freq(
    inv_freq: torch.Tensor,
    scaling_factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_position_embeddings: int,
) -> torch.Tensor:
    low_wl = original_max_position_embeddings / low_freq_factor
    high_wl = original_max_position_embeddings / high_freq_factor
    wl = 2 * math.pi / inv_freq
    if low_freq_factor != high_freq_factor:
        smooth = (original_max_position_embeddings / wl - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
    else:
        smooth = torch.zeros_like(inv_freq)
    return torch.where(
        wl < high_wl,
        inv_freq,
        torch.where(
            wl > low_wl,
            inv_freq / scaling_factor,
            (1 - smooth) * inv_freq / scaling_factor + smooth * inv_freq,
        ),
    )

class RotaryEmbedding(nn.Module):
    """RoPE with optional Llama 3.1-style frequency scaling.

    When rope_scaling_factor == 1.0 (default), behaves as standard RoPE.
    When rope_scaling_factor != 1.0, applies the Llama 3.1 piecewise
    frequency scaling controlled by low/high freq factors.
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        rope_scaling_factor: float = 1.0,
        rope_low_freq_factor: float = 1.0,
        rope_high_freq_factor: float = 1.0,
        rope_original_max_position_embeddings: int | None = None,
        is_neox_style: bool = True,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.is_neox_style = is_neox_style
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))

        if rope_scaling_factor != 1.0 and rope_original_max_position_embeddings is not None:
            inv_freq = _compute_scaled_inv_freq(
                inv_freq,
                rope_scaling_factor,
                rope_low_freq_factor,
                rope_high_freq_factor,
                rope_original_max_position_embeddings,
            )

        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).float()
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @staticmethod
    def forward_native(positions, query, key, head_dim, cos_sin_cache):
        """Pure PyTorch NeOX-style RoPE matching the CUDA kernel.

        The cache stores [cos, sin] each with embed_dim = head_dim/2 entries.
        Rotation pairs elements (i, i + embed_dim) across the full head,
        exactly matching the CUDA kernel's IS_NEOX=true path:
          out[i]            = x[i]*cos[i] - x[i+embed_dim]*sin[i]
          out[i+embed_dim]  = x[i+embed_dim]*cos[i] + x[i]*sin[i]
        """
        cos_sin = cos_sin_cache[positions]
        embed_dim = cos_sin.shape[-1] // 2
        cos = cos_sin[..., :embed_dim]
        sin = cos_sin[..., embed_dim:]

        q_shape = query.shape
        k_shape = key.shape
        q = query.view(q_shape[0], -1, head_dim)
        k = key.view(k_shape[0], -1, head_dim)

        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        q1, q2 = q[..., :embed_dim], q[..., embed_dim:]
        k1, k2 = k[..., :embed_dim], k[..., embed_dim:]

        query = torch.cat([q1 * cos - q2 * sin,
                           q2 * cos + q1 * sin], dim=-1).view(q_shape)
        key = torch.cat([k1 * cos - k2 * sin,
                         k2 * cos + k1 * sin], dim=-1).view(k_shape)
        return query, key

    @staticmethod
    def forward_native_interleaved(positions, query, key, head_dim, cos_sin_cache):
        """Pure PyTorch GPT-J/interleaved RoPE matching CUDA IS_NEOX=false."""
        cos_sin = cos_sin_cache[positions]
        embed_dim = cos_sin.shape[-1] // 2
        cos = cos_sin[..., :embed_dim].unsqueeze(1)
        sin = cos_sin[..., embed_dim:].unsqueeze(1)

        q_shape = query.shape
        k_shape = key.shape
        q = query.view(q_shape[0], -1, head_dim)
        k = key.view(k_shape[0], -1, head_dim)

        q_even, q_odd = q[..., 0::2], q[..., 1::2]
        k_even, k_odd = k[..., 0::2], k[..., 1::2]

        q_rot = torch.stack(
            (q_even * cos - q_odd * sin,
             q_odd * cos + q_even * sin),
            dim=-1,
        ).flatten(-2)
        k_rot = torch.stack(
            (k_even * cos - k_odd * sin,
             k_odd * cos + k_even * sin),
            dim=-1,
        ).flatten(-2)
        return q_rot.view(q_shape), k_rot.view(k_shape)

    def forward_cuda(self, positions, query, key):
        """CUDA kernel path for eager mode."""
        cache = self.cos_sin_cache
        if cache.dtype != query.dtype:
            cache = cache.to(query.dtype)
        torch.ops.fastkernels_rope.rotary_embedding(
            positions, query, key, self.head_dim, cache, self.is_neox_style,
        )
        return query, key

    def forward(self, positions, query, key):
        if torch.compiler.is_compiling():
            cache = self.cos_sin_cache
            if cache.dtype != query.dtype:
                cache = cache.to(query.dtype)
            if self.is_neox_style:
                return self.forward_native(
                    positions, query, key, self.head_dim, cache,
                )
            return self.forward_native_interleaved(
                positions, query, key, self.head_dim, cache,
            )
        return self.forward_cuda(positions, query, key)

_C = lazy_op("rms_norm", "rms_norm.cu")

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6,
                 elementwise_affine: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            # Match vLLM's has_weight=False path: use the same CUDA RMSNorm
            # kernel with a non-persistent unit scale instead of falling back
            # to torch.nn.functional.rms_norm in eager/CUDA-graph decode.
            self.register_buffer(
                "_unit_weight",
                torch.ones(hidden_size),
                persistent=False,
            )

    # -- Pure PyTorch path (used under torch.compile so Inductor can fuse) --

    @staticmethod
    def forward_native(
        x: torch.Tensor,
        weight: torch.Tensor | None,
        eps: float,
        hidden_size: int,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Pure PyTorch RMSNorm matching vLLM's forward_static."""
        orig_dtype = x.dtype
        x = x.float()
        if residual is not None:
            x = x + residual.float()
            residual = x.to(orig_dtype)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        x = x.to(orig_dtype)
        if weight is not None:
            x = x * weight
        if residual is None:
            return x
        return x, residual

    # -- CUDA kernel path (used in eager mode / CUDA graph replay) --

    @staticmethod
    def forward_cuda(
        x: torch.Tensor,
        weight: torch.Tensor | None,
        eps: float,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if weight is not None:
            # The CUDA rms_norm / fused_add_rms_norm kernels assume the row
            # dimension is contiguous (row stride == hidden size). A strided
            # input — e.g. the K slice of a fused QKV output when num_kv_heads
            # collapses to a single head under tensor parallelism, so the
            # per-head reshape yields a non-contiguous view — makes the kernel
            # read the wrong memory for every row past the first, silently
            # corrupting the output. Force contiguity here (a no-op when the
            # tensor is already contiguous) so every caller is safe.
            x = x.contiguous()
            if residual is not None:
                residual = residual.contiguous()
            if residual is None:
                out = torch.empty_like(x)
                _C.rms_norm(out, x, weight, eps)
                return out
            _C.fused_add_rms_norm(x, residual, weight, eps)
            return x, residual
        if residual is None:
            return F.rms_norm(x, (x.size(-1),), eps=eps)
        x = x + residual
        residual = x
        return F.rms_norm(x, (x.size(-1),), eps=eps), residual

    def forward(self, x, residual=None):
        if torch.compiler.is_compiling():
            return self.forward_native(
                x, self.weight if self.elementwise_affine else None,
                self.eps, self.hidden_size, residual,
            )
        weight = self.weight if self.elementwise_affine else self._unit_weight
        if weight.dtype != x.dtype or weight.device != x.device:
            weight = weight.to(device=x.device, dtype=x.dtype)
        return self.forward_cuda(
            x, weight, self.eps, residual,
        )

class LogSigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.logsigmoid(x)

class Matmul(nn.Module):
    """Pure functional linear: takes input, weight, and optional bias as forward args."""

    def forward(self, input, weight, bias=None):
        return F.linear(input, weight, bias)

class Linear(nn.Module):
    """Parametric linear: stores weight and bias internally."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.matmul = Matmul()

    def forward(self, input):
        return self.matmul(input, self.weight, self.bias)

def naive_recurrent_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Naive loop-based GLA recurrence.

    Args:
        q:  [B, H, T, K]
        k:  [B, H, T, K]
        v:  [B, H, T, V]
        gk: [B, H, T, K]  log-space forget gate (broadcast across T for RetNet)
        scale: query scaling factor (default: K**-0.5)
        initial_state: [B, H, K, V] initial recurrent state
        output_final_state: whether to return final state

    Returns:
        o: [B, H, T, V]  output
        final_state: [B, H, K, V] or None
    """
    B, H, T, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5

    h = q.new_zeros(B, H, K, V, dtype=torch.float32)
    o = torch.zeros_like(v)

    if initial_state is not None:
        h = h + initial_state.float()

    for i in range(T):
        q_i = q[:, :, i] * scale
        k_i = k[:, :, i]
        v_i = v[:, :, i]
        gk_i = gk[:, :, i].float().exp()

        h = h * gk_i[..., None] + k_i.float()[..., None] * v_i.float()[..., None, :]
        o[:, :, i] = (q_i.float()[..., None] * h).sum(-2).to(v.dtype)

    final_state = h if output_final_state else None
    return o, final_state

class NaiveRecurrentGLA(nn.Module):
    """nn.Module wrapper around `naive_recurrent_gla` for L1 compliance."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gk: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return naive_recurrent_gla(
            q, k, v, gk,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )

class FusedRecurrentRetention(nn.Module):
    """Triton fused-recurrent retention kernel."""

    def forward(
        self,
        q: torch.Tensor,  # [B, T, H, K]
        k: torch.Tensor,  # [B, T, H, K]
        v: torch.Tensor,  # [B, T, H, V]
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,  # [B, H, K, V]
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return fused_recurrent_retention(
            q=q, k=k, v=v,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )

class FusedRecurrentGLA(nn.Module):
    """Triton fused-recurrent GLA kernel."""

    def forward(
        self,
        q: torch.Tensor,  # [B, T, H, K]
        k: torch.Tensor,  # [B, T, H, K]
        v: torch.Tensor,  # [B, T, H, V]
        gk: torch.Tensor | None = None,  # [B, T, H, K]
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,  # [B, H, K, V]
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return fused_recurrent_gla(
            q=q, k=k, v=v, gk=gk,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )

class ChunkRetention(nn.Module):
    """Triton chunk RetNet kernel."""

    def forward(
        self,
        q: torch.Tensor,  # [B, T, H, K]
        k: torch.Tensor,  # [B, T, H, K]
        v: torch.Tensor,  # [B, T, H, V]
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,  # [B, H, K, V]
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return chunk_retention(
            q=q, k=k, v=v,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )

class ChunkGLA(nn.Module):
    """Triton chunk GLA kernel."""

    def forward(
        self,
        q: torch.Tensor,  # [B, T, H, K]
        k: torch.Tensor,  # [B, T, H, K]
        v: torch.Tensor,  # [B, T, H, V]
        g: torch.Tensor,  # [B, T, H, K]  log-space forget gate
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,  # [B, H, K, V]
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return chunk_gla(
            q=q, k=k, v=v, g=g,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )

"""Gated linear attention (covers both GLA and RetNet).

The forward signature matches FLA's ``GatedLinearAttention.forward``
exactly so fastkernels kernels are drop-in for FLA users:

    forward(hidden_states, attention_mask=None,
            past_key_values=None, use_cache=False, **kwargs)
        -> (output, attentions, past_key_values)

Per the "Condense Variants" rule, this single class subsumes FLA's
``GatedLinearAttention`` (GLA, learned data-dependent gate) and
``MultiScaleRetention`` (RetNet, fixed-per-head decay + rotary).
The two architectures differ only in:

  * ``decay_mode``:
      - ``"learned_low_rank"`` (GLA): per-token, per-head, per-channel gk
        from a low-rank projection: ``gk = logsigmoid(W2(W1(x))) / norm``.
      - ``"fixed_per_head"`` (RetNet): data-independent gk[..., t, :] =
        log(gamma_h) for ``gamma_h = 1 - 2^(-5-h)``, broadcast across T.
  * ``use_rotary``: RetNet applies rotary to q/k; GLA does not.

Both feed into the SAME L1 recurrence kernel ``naive_recurrent_gla``
(RetNet is the constant-gk special case), and both finish with a per-head
RMSNorm + swish output gate. This consolidation keeps the L2 surface
small while preserving FLA's two distinct config knobs.

``nn.Sequential`` and ``nn.ModuleList`` are used here as pure-Python
*containers* over L1 ops (mirroring how every L4 model uses
``nn.ModuleList`` to hold L3 layers); the L2 "no torch.nn" rule applies
to *kernel* modules (Linear, LayerNorm, GroupNorm, activations) which we
unconditionally route through L1.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


# Threshold (matches FLA's own dispatch in fla.layers.rwkv7) — below this
# the chunk kernel's launch overhead exceeds its parallel speedup, so the
# fused-recurrent path is faster for short sequences (typical decode T=1).
_CHUNK_THRESHOLD = 64


class Model(nn.Module):
    """Unified L2 attention for GLA and RetNet.

    Args:
        hidden_size: Model hidden size.
        num_heads: Number of attention heads.
        expand_k: Key expansion ratio (GLA: 0.5, RetNet: 1.0).
        expand_v: Value expansion ratio (GLA: 1.0, RetNet: 2.0).
        decay_mode: Which forget-gate mechanism to use.
        gate_low_rank_dim: Low-rank dim for the GLA gate (ignored for
            ``fixed_per_head``).
        gate_logit_normalizer: Normalizer applied after logsigmoid in the
            GLA gate (ignored for ``fixed_per_head``).
        use_rotary: Whether to apply rotary to q/k (RetNet uses this).
        rotary_base: Rotary base (theta).
        rotary_max_position: Max sequence length the rotary cache covers.
        norm_eps: RMSNorm epsilon for the per-head output norm.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        decay_mode: Literal["learned_low_rank", "fixed_per_head"] = "learned_low_rank",
        gate_low_rank_dim: int = 16,
        gate_logit_normalizer: int = 16,
        use_rotary: bool = False,
        rotary_base: float = 10000.0,
        rotary_max_position: int = 8192,
        norm_eps: float = 1e-6,
        use_fast_kernels: bool = True,
    ):
        super().__init__()
        assert decay_mode in ("learned_low_rank", "fixed_per_head"), (
            f"unknown decay_mode: {decay_mode!r}"
        )
        self.num_heads = num_heads
        self.decay_mode = decay_mode
        self.use_rotary = use_rotary
        self.gate_logit_normalizer = gate_logit_normalizer

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        self.q_proj = Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = Linear(hidden_size, self.value_dim, bias=False)
        self.g_proj = Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = Linear(self.value_dim, hidden_size, bias=False)

        if decay_mode == "learned_low_rank":
            # FLA stores this as ``gk_proj = nn.Sequential(Linear, Linear)``
            # so the checkpoint paths are ``gk_proj.0.weight`` and
            # ``gk_proj.1.{weight,bias}``. nn.Sequential is used here purely
            # as a container; both children are L1 Linear ops.
            self.gk_proj = nn.Sequential(
                Linear(hidden_size, gate_low_rank_dim, bias=False),
                Linear(gate_low_rank_dim, self.key_dim, bias=True),
            )
            self.log_sigmoid = LogSigmoid()
        else:
            # RetNet: fixed per-head decay gamma_h = 1 - 2^(-5-h).
            # Stored as a non-persistent buffer so it auto-moves with the
            # module and is not written to checkpoints.
            h_idx = torch.arange(num_heads, dtype=torch.float32)
            gamma = 1.0 - torch.pow(torch.tensor(2.0, dtype=torch.float32), -5.0 - h_idx)
            log_gamma = torch.log(gamma)
            self.register_buffer("log_gamma", log_gamma, persistent=False)

        if use_rotary:
            self.rotary_emb = RotaryEmbedding(
                head_dim=self.head_k_dim,
                max_position_embeddings=rotary_max_position,
                rope_theta=rotary_base,
            )

        # Fast paths (Triton, FLA-vendored) + naive fallback (pure PyTorch).
        # The fast/slow choice is decided per-forward based on T and
        # ``use_fast_kernels``: chunk for prefill (T >= 64), fused-recurrent
        # for decode (T < 64). The naive path stays available for CPU
        # fallback / numerical reference.
        self.use_fast_kernels = use_fast_kernels
        self.naive_recurrence = NaiveRecurrentGLA()
        if use_fast_kernels:
            if decay_mode == "learned_low_rank":
                self.fused_recurrence = FusedRecurrentGLA()
                self.chunk = ChunkGLA()
            else:
                self.fused_recurrence = FusedRecurrentRetention()
                self.chunk = ChunkRetention()

        self.g_norm_swish_gate = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.gate_act = SiLU()

    def _compute_gk(
        self, hidden_states: torch.Tensor, B: int, T: int,
    ) -> torch.Tensor:
        """Returns gk shaped [B, num_heads, T, head_k_dim] in log-space.

        Used by the naive recurrence path. The fast path uses
        :meth:`_compute_gk_bthk` to skip an unnecessary transpose.
        """
        if self.decay_mode == "learned_low_rank":
            gk = self.gk_proj(hidden_states)
            gk = self.log_sigmoid(gk) / self.gate_logit_normalizer
            return gk.view(B, T, self.num_heads, self.head_k_dim).transpose(1, 2)
        return self.log_gamma.to(hidden_states.dtype).view(
            1, self.num_heads, 1, 1
        ).expand(B, self.num_heads, T, self.head_k_dim)

    def _compute_gk_bthk(
        self, hidden_states: torch.Tensor, B: int, T: int,
    ) -> torch.Tensor:
        """Returns gk shaped [B, T, num_heads, head_k_dim] in log-space."""
        gk = self.gk_proj(hidden_states)
        gk = self.log_sigmoid(gk) / self.gate_logit_normalizer
        return gk.view(B, T, self.num_heads, self.head_k_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, None, object | None]:
        B, T, _ = hidden_states.shape
        cu_seqlens = kwargs.get("cu_seqlens")
        max_seqlen = None
        if cu_seqlens is not None:
            if B != 1:
                raise ValueError("cu_seqlens prefill expects packed hidden_states with batch size 1")
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            max_seqlen = int(lengths.max().item()) if lengths.numel() else 0

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        g = self.g_proj(hidden_states)

        if self.use_rotary:
            # Build per-token absolute positions. For uncached single-shot
            # forward we use 0..T-1 per row. For cached prefill / decode the
            # engine passes ``past_key_values.seq_offsets`` (int or [B]
            # int64) giving the global position of token 0 in this call,
            # per row. Without that offset, RoPE would re-encode every
            # decode step at position 0 — totally breaking RetNet.
            #
            # NOTE: must materialize a contiguous int64 buffer with B*T real
            # elements. ``arange(T).expand(B, T).reshape(-1)`` returns a
            # stride-0 view (only T elements of storage); the CUDA RoPE
            # kernel does flat ``positions[token_idx]`` indexing which would
            # read out-of-bounds for token_idx >= T → illegal access.
            offsets = None
            if past_key_values is not None:
                offsets = getattr(past_key_values, "seq_offsets", None)
            if cu_seqlens is not None:
                # Packed varlen [1, total_T]: positions restart at each
                # sequence boundary. token t's position = its per-sequence local
                # index + that sequence's global start offset (seq_offsets, or
                # 0). This must be a flat [total_T] vector -- the dense branch
                # below builds [B*T], which is wrong for a packed batch and
                # feeds the RoPE kernel a positions length != query rows.
                lengths = cu_seqlens[1:] - cu_seqlens[:-1]
                seg_start = torch.repeat_interleave(cu_seqlens[:-1], lengths)
                positions = torch.arange(T, device=q.device, dtype=torch.int64) - seg_start
                if isinstance(offsets, int):
                    positions = positions + offsets
                elif offsets is not None:
                    positions = positions + torch.repeat_interleave(
                        offsets.to(device=q.device, dtype=torch.int64), lengths)
                positions = positions.contiguous()
            else:
                local = torch.arange(T, device=q.device, dtype=torch.int64)
                if offsets is None:
                    positions = local.repeat(B)
                elif isinstance(offsets, int):
                    positions = (local + offsets).repeat(B)
                else:
                    # [B] int64 tensor of per-row prefix lengths
                    positions = (offsets.to(device=q.device, dtype=torch.int64)
                                 .unsqueeze(1) + local.unsqueeze(0)).reshape(-1)
                    positions = positions.contiguous()
            q_flat = q.reshape(B * T, self.num_heads * self.head_k_dim).contiguous()
            k_flat = k.reshape(B * T, self.num_heads * self.head_k_dim).contiguous()
            q_flat, k_flat = self.rotary_emb(positions, q_flat, k_flat)
            q = q_flat.view(B, T, self.num_heads, self.head_k_dim)
            k = k_flat.view(B, T, self.num_heads, self.head_k_dim)
        else:
            q = q.view(B, T, self.num_heads, self.head_k_dim)
            k = k.view(B, T, self.num_heads, self.head_k_dim)

        v = v.view(B, T, self.num_heads, self.head_v_dim)

        initial_state = None
        if past_key_values is not None and getattr(past_key_values, "states", None):
            initial_state = past_key_values.states.get(id(self))

        # Dispatch:
        #   T >= 64 + fast kernels -> chunk (prefill / training)
        #   T  < 64 + fast kernels -> fused_recurrent (decode)
        #   no fast kernels         -> naive PyTorch (CPU / debug / reference)
        if self.use_fast_kernels and q.is_cuda:
            dispatch_len = max_seqlen if max_seqlen is not None else T
            if self.decay_mode == "learned_low_rank":
                # gk in [B, T, H, K] log-space, NOT transposed
                gk_btHK = self._compute_gk_bthk(hidden_states, B, T)
                if dispatch_len >= _CHUNK_THRESHOLD:
                    o, final_state = self.chunk(
                        q=q, k=k, v=v, g=gk_btHK,
                        initial_state=initial_state,
                        output_final_state=use_cache,
                        cu_seqlens=cu_seqlens,
                    )
                else:
                    o, final_state = self.fused_recurrence(
                        q=q, k=k, v=v, gk=gk_btHK,
                        initial_state=initial_state,
                        output_final_state=use_cache,
                        cu_seqlens=cu_seqlens,
                    )
            else:  # RetNet — kernel bakes in the per-head decay
                if dispatch_len >= _CHUNK_THRESHOLD:
                    o, final_state = self.chunk(
                        q=q, k=k, v=v,
                        initial_state=initial_state,
                        output_final_state=use_cache,
                        cu_seqlens=cu_seqlens,
                    )
                else:
                    o, final_state = self.fused_recurrence(
                        q=q, k=k, v=v,
                        initial_state=initial_state,
                        output_final_state=use_cache,
                        cu_seqlens=cu_seqlens,
                    )
            # Fast-path output is already [B, T, H, V] — no transpose needed.
        else:
            # Naive path expects [B, H, T, D]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            gk = self._compute_gk(hidden_states, B, T)
            o, final_state = self.naive_recurrence(
                q, k, v, gk,
                initial_state=initial_state,
                output_final_state=use_cache,
            )
            o = o.transpose(1, 2)  # [B, H, T, V] -> [B, T, H, V]

        if use_cache and past_key_values is not None:
            if not hasattr(past_key_values, "states"):
                past_key_values.states = {}
            past_key_values.states[id(self)] = final_state

        o = self.g_norm_swish_gate(o.reshape(-1, self.head_v_dim))
        o = o.view(B, T, self.value_dim)
        o = o * self.gate_act(g)

        return self.o_proj(o), None, past_key_values

    ```

ModelNew must use the same `__init__` arguments and `forward` signature as `Model`.

The kernel will be evaluated on these captured input shapes:

### GatedLinearAttention

| count | args |
|------:|------|
| 46816 | `hidden_states:bfloat16[256, 1, 2560] attention_mask:None` |
| 7264 | `hidden_states:bfloat16[1, 1, 2560] attention_mask:None` |
| 3168 | `hidden_states:bfloat16[4, 1, 2560] attention_mask:None` |
| 2656 | `hidden_states:bfloat16[64, 1, 2560] attention_mask:None` |
| 2592 | `hidden_states:bfloat16[60, 1, 2560] attention_mask:None` |
| 2592 | `hidden_states:bfloat16[8, 1, 2560] attention_mask:None` |
| 2496 | `hidden_states:bfloat16[1, 4096, 2560] attention_mask:None` |
| 1472 | `hidden_states:bfloat16[31, 1, 2560] attention_mask:None` |

    
Optimize the architecture named Model with custom Triton operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Let's think step by step.
