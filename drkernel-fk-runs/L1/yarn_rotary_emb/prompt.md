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
import math
import torch
import torch.nn as nn

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

"""Rotary position embeddings with YaRN / YARN scaling.

Two variants:
  - ``YaRNRotaryEmbedding``: NeoX-style YaRN RoPE used by GPT-OSS. Applies
    magnitude correction via ``mscale``.
  - ``YarnRotaryEmbedding``: DeepSeek-style YARN RoPE (interleaved, NON-NeoX)
    used by DeepSeek V3.  Supports separate ``mscale`` / ``mscale_all_dim``
    knobs and exposes ``softmax_mscale`` as an attention scaling factor.

Both classes share the same L1 CUDA rotary kernel via
``torch.ops.fastkernels_rope.rotary_embedding``; the only differences are how
the ``cos_sin_cache`` is computed and whether NeoX layout is used.

References:
  - Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models"
  - vLLM: ``vllm/model_executor/layers/rotary_embedding/yarn_scaling_rope.py``
  - vLLM: ``vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py``
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# Detect FlashInfer rotary op once at import time.  vLLM's
# ``torch.ops.vllm.flashinfer_rotary_embedding`` is a thin wrapper around
# ``flashinfer.rope.apply_rope_with_cos_sin_cache_inplace`` (see
# ``vllm/model_executor/layers/rotary_embedding/common.py``), so we call the
# FlashInfer package directly instead of importing vllm to register the op.
try:
    from flashinfer.rope import (
        apply_rope_with_cos_sin_cache_inplace as _flashinfer_apply_rope,
    )
    _USE_FLASHINFER_ROPE = True
except Exception:
    _flashinfer_apply_rope = None
    _USE_FLASHINFER_ROPE = False


# GLM-5.2's plain "default" rope is applied via the vendored vLLM rotary
# kernel (base RotaryEmbedding.forward_cuda), exposed as the
# ``torch.ops.fastkernels_rope.rotary_embedding`` custom op registered by
# ``rotary_emb``. Importing it here ensures the op is defined.


def _yarn_find_correction_dim(
    num_rotations: float, dim: int, base: float, max_position_embeddings: int,
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _yarn_find_correction_range(
    low_rot: float, high_rot: float, dim: int, base: float,
    max_position_embeddings: int, truncate: bool = True,
) -> tuple[float | int, float | int]:
    low = _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    high = _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(
    low: float, high: float, dim: int, dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    if low == high:
        high += 0.001
    linear_func = (torch.arange(dim, dtype=dtype) - low) / (high - low)
    return torch.clamp(linear_func, 0, 1)


def _yarn_get_mscale(scale: float) -> float:
    """GPT-OSS style mscale (no explicit mscale parameter)."""
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


def yarn_get_mscale(scale: float, mscale: float) -> float:
    """DeepSeek-style mscale with explicit parameter (matches vLLM)."""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class Model(nn.Module):
    """YaRN RoPE with precomputed cos/sin cache.

    Uses the same L1 CUDA kernel as RotaryEmbedding for the rotation step
    (NeoX layout).  Used by GPT-OSS.
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        scaling_factor: float,
        original_max_position_embeddings: int,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        truncate: bool = True,
    ):
        super().__init__()
        self.head_dim = head_dim
        rotary_dim = head_dim

        pos_freqs = rope_theta ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = _yarn_find_correction_range(
            beta_fast, beta_slow, rotary_dim, rope_theta,
            original_max_position_embeddings, truncate,
        )
        inv_freq_mask = (
            1 - _yarn_linear_ramp_mask(low, high, rotary_dim // 2, dtype=torch.float)
        )
        inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_mask)
            + inv_freq_extrapolation * inv_freq_mask
        )

        mscale = _yarn_get_mscale(scaling_factor)

        max_t = int(max_position_embeddings * scaling_factor)
        t = torch.arange(max_t, dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * mscale
        sin = freqs.sin() * mscale
        cache = torch.cat((cos, sin), dim=-1).float()
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(self, positions, query, key):
        cache = self.cos_sin_cache
        if cache.dtype != query.dtype:
            cache = cache.to(query.dtype)
        if torch.compiler.is_compiling():
            return RotaryEmbedding.forward_native(
                positions, query, key, self.head_dim, cache,
            )
        torch.ops.fastkernels_rope.rotary_embedding(
            positions, query, key, self.head_dim, cache, True,
        )
        return query, key


class YarnRotaryEmbedding(nn.Module):
    """DeepSeek-style YARN (Yet Another RoPE extensioN) RoPE.

    Uses NON-NeoX (interleaved) layout, matching vLLM's
    ``DeepseekScalingRotaryEmbedding``.  The cos/sin cache is scaled by
    ``softmax_mscale`` which folds the attention magnitude correction into
    the rotary cache (so attention scores do not need to multiply by
    ``softmax_mscale`` separately).
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        scaling_factor: float,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
        is_neox_style: bool = False,
        is_plain: bool = False,
        cache_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.is_neox_style = is_neox_style
        # ``is_plain`` marks a degenerate (scaling_factor==1.0) instance that is
        # really standard RoPE — e.g. GLM-5.2's ``rope_type: "default"``. vLLM
        # maps a "default" rope to the base ``RotaryEmbedding``, which does NOT
        # use the FlashInfer kernel and casts the cos/sin cache to the model
        # dtype (bf16). DeepSeek-V3.2 YARN (scaling_factor>1) keeps FlashInfer +
        # fp32 cache. Threading this flag lets both match vLLM exactly.
        self.is_plain = is_plain
        rotary_dim = head_dim
        base = rope_theta

        softmax_mscale = (
            yarn_get_mscale(scaling_factor, mscale)
            / yarn_get_mscale(scaling_factor, mscale_all_dim)
            * attn_factor
        )
        self.softmax_mscale = softmax_mscale

        pos_freqs = base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)
        low, high = _yarn_find_correction_range(
            beta_fast, beta_slow, rotary_dim, base, max_position_embeddings,
        )
        inv_freq_mask = (
            1 - _yarn_linear_ramp_mask(low, high, rotary_dim // 2, dtype=torch.float)
        ) * extrapolation_factor
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

        t = torch.arange(max_position_embeddings * scaling_factor, dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * softmax_mscale
        sin = freqs.sin() * softmax_mscale
        cache = torch.cat((cos, sin), dim=-1).float()
        # Plain "default" rope (GLM-5.2): vLLM's base ``RotaryEmbedding`` stores
        # the cos/sin cache in the model compute dtype (bf16) once at init, so
        # its forward never re-casts. Match that — computing in fp32 then
        # casting to bf16 here is bit-identical to casting per-forward, and
        # skips a full-cache dtype conversion on every rope call. YARN
        # (is_plain=False) keeps the fp32 cache for the FlashInfer path.
        if self.is_plain and cache_dtype is not None:
            cache = cache.to(cache_dtype)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(self, positions, query, key):
        # vLLM's ``DeepseekScalingRotaryEmbedding.forward_cuda`` prefers the
        # FlashInfer fused kernel when available (see
        # ``vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py:181-198``).
        # FlashInfer keeps ``cos_sin_cache`` in float32; only the fastkernels
        # CUDA kernel needs the cache cast to query.dtype.
        if _USE_FLASHINFER_ROPE and not self.is_plain \
                and query.dtype in (torch.float16, torch.bfloat16) \
                and self.head_dim in (64, 128, 256, 512):
            # Mirrors vLLM's ``flashinfer_rotary_embedding`` custom op, which
            # just forwards to this FlashInfer entry point in-place.
            _flashinfer_apply_rope(
                positions=positions,
                query=query,
                key=key,
                head_size=self.head_dim,
                cos_sin_cache=self.cos_sin_cache,
                is_neox=self.is_neox_style,
            )
            return query, key
        cache = self.cos_sin_cache
        if cache.dtype != query.dtype:
            cache = cache.to(query.dtype)
        # GLM-5.2 plain "default" rope and the scaled path both go through the
        # vendored vLLM rotary kernel. Call it via the registered
        # ``fastkernels_rope`` custom op (whose CUDA impl is exactly
        # ``_C.rotary_embedding``) rather than the raw pybind function, so
        # ``torch.compile`` / cudagraph capture can trace it. Numerically
        # identical to calling ``_C.rotary_embedding`` directly.
        torch.ops.fastkernels_rope.rotary_embedding(
            positions, query, key, self.head_dim, cache, self.is_neox_style,
        )
        return query, key

    ```

ModelNew must use the same `__init__` arguments and `forward` signature as `Model`.

The kernel will be evaluated on these captured input shapes:

### YaRNRotaryEmbedding

| count | args |
|------:|------|
| 6876 | `positions:int64[60] query:bfloat16[60, 2048] key:bfloat16[60, 256]` |
| 4680 | `positions:int64[1] query:bfloat16[1, 2048] key:bfloat16[1, 256]` |
| 3096 | `positions:int64[16384] query:bfloat16[16384, 2048] key:bfloat16[16384, 256]` |
| 1152 | `positions:int64[31] query:bfloat16[31, 2048] key:bfloat16[31, 256]` |
| 1116 | `positions:int64[26] query:bfloat16[26, 2048] key:bfloat16[26, 256]` |
| 1080 | `positions:int64[29] query:bfloat16[29, 2048] key:bfloat16[29, 256]` |
| 684 | `positions:int64[30] query:bfloat16[30, 2048] key:bfloat16[30, 256]` |
| 576 | `positions:int64[67] query:bfloat16[67, 2048] key:bfloat16[67, 256]` |

    
Optimize the architecture named Model with custom Triton operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Let's think step by step.
