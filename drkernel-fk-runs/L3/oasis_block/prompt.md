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
from math import pi
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

def _resolve_flash_attn_func():
    """Return the flash-attention callable for Ampere/Hopper.

    Same order as vllm-omni's CUDA FA resolver: FA3 (fa3-fwd) >
    FA3 (source-built flash_attn_interface) > FA2.
    """
    for mod in ("fa3_fwd_interface", "flash_attn_interface"):
        try:
            return __import__(mod, fromlist=["flash_attn_func"]).flash_attn_func
        except (ImportError, ModuleNotFoundError):
            pass
    from flash_attn import flash_attn_func
    return flash_attn_func

_CUDNN_MAX_HEAD_DIM = 128

class DenseAttention(nn.Module):
    """Dense multi-head attention.

    Input layout: (batch, seq_len, num_heads, head_dim).

    Args:
        backend: Which kernel to use.
            ``"auto"`` selects flash-attention on Ampere/Hopper when
            available, SDPA everywhere else.
            ``"sdpa"`` always uses ``F.scaled_dot_product_attention``
            (PyTorch's heuristic chooses among flash/cuDNN/mem_eff/math).
            ``"flash_attn"`` always uses the flash-attention package.
            ``"cudnn"`` pins the cuDNN flash backend via
            ``torch.nn.attention.sdpa_kernel`` (with MATH fallback for
            masks cuDNN can't handle). Required to get cuDNN flash
            through ``torch.compile`` on Blackwell.
    """

    def __init__(self, backend: Literal["auto", "sdpa", "flash_attn", "cudnn", "flex"] = "auto"):
        super().__init__()
        self.fa_func = None
        self.use_cudnn_kernel = False
        self.use_flex_kernel = False
        self._flex_fn = None

        if backend == "sdpa":
            return

        if backend == "cudnn":
            self.use_cudnn_kernel = True
            return

        if backend == "flex":
            from torch.nn.attention.flex_attention import flex_attention
            self.use_flex_kernel = True
            self._flex_fn = torch.compile(flex_attention, dynamic=False)
            return

        if backend == "flash_attn":
            self.fa_func = _resolve_flash_attn_func()
            return

        # backend == "auto": flash-attn on Ampere/Hopper (80<=cc<100); cuDNN flash
        # on Blackwell (cc>=100), where PyTorch's SDPA heuristic otherwise picks
        # FA2 (~3.6x slower than cuDNN for large joint-attention shapes on B200).
        # This mirrors vllm-omni's platform selector, which pins cuDNN/TRTLLM on
        # Blackwell. The cuDNN forward path already falls back to mem-efficient/MATH
        # for shapes/masks cuDNN rejects, so this is safe as a default.
        cc = (torch.cuda.get_device_capability()[0] * 10
              + torch.cuda.get_device_capability()[1])
        if 80 <= cc < 100:
            self.fa_func = _resolve_flash_attn_func()
        elif cc >= 100:
            self.use_cudnn_kernel = True

    def forward(
        self,
        query,
        key,
        value,
        softmax_scale=None,
        causal=False,
        attn_mask: torch.Tensor | None = None,
    ):
        if self.fa_func is not None and attn_mask is None and query.dtype != torch.float32:
            out = self.fa_func(
                query, key, value,
                softmax_scale=softmax_scale,
                causal=causal,
            )
            if isinstance(out, tuple):
                out = out[0]
            return out

        # SDPA handles both the masked case and the plain causal/non-causal case.
        # Custom masks force is_causal=False; FlashAttn does not support arbitrary masks.
        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)
        if self.use_flex_kernel:
            # FlexAttention generates a fused Triton fwd+bwd kernel autotuned
            # for the exact (B, H, S_q, S_kv, D) shape and the user-provided
            # mask. ``attn_mask`` here is repurposed to accept a
            # ``BlockMask`` (from ``create_block_mask``) instead of a dense
            # bool tensor. On B200 with chunked-suffix shapes
            # (Q=1024, KV=9216, D=64), the fused fwd+bwd is ~1.37x faster
            # than cuDNN flash with the equivalent dense mask
            # (microbenched). Same numerical agreement vs the fp32 MATH
            # reference (~1e-2 max-abs-diff in bf16, identical to cuDNN).
            q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
            out = self._flex_fn(
                q, k, v,
                block_mask=attn_mask,
                scale=softmax_scale,
            )
        elif self.use_cudnn_kernel:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            # An explicit mask plus is_causal=True is ambiguous, and the two code
            # paths here would resolve it differently: this branch would hand both
            # to SDPA (which applies the causal mask *on top of* attn_mask), while
            # the non-cuDNN branch below drops is_causal and treats attn_mask as
            # authoritative. SDPA itself accepts the combination on this backend
            # rather than rejecting it, so nothing would surface the disagreement
            # -- reject it here instead of silently masking twice.
            if attn_mask is not None and causal:
                raise ValueError(
                    "DenseAttention: pass either attn_mask or causal=True, not both "
                    "(an explicit mask must already encode causality). Got "
                    f"attn_mask={tuple(attn_mask.shape)} with causal=True."
                )
            # The sdpa_kernel context below FORCES cuDNN, and on Blackwell (sm100,
            # cuDNN 9.19) the cuDNN flash kernel accepts the permuted, non-contiguous
            # q/k/v views directly -- so we skip the q/k/v .contiguous() clones (they
            # were a real cost: 3 clones/block x54 blocks). Verified bit-identical and
            # faster; if cuDNN ever rejects a layout it raises -> MATH fallback below.
            if attn_mask is not None and not attn_mask.is_contiguous():
                attn_mask = attn_mask.contiguous()
            # Try strict cuDNN first. Adding MATH as a fallback in the
            # ``sdpa_kernel`` list causes PyTorch's selection heuristic to
            # pick MATH over cuDNN (~10× slower) for inputs both can
            # handle. If cuDNN rejects (e.g. head_dim=16, fp32, or some
            # mask shape it doesn't support), fall back through MATH.
            #
            # head_dim > 128 is rejected by cuDNN unconditionally ("head_dim
            # should be no more than 128"), so route it straight to the backends
            # that can serve it. The try/except below only recovers in eager --
            # under torch.compile the RuntimeError surfaces during fake-tensor
            # tracing and aborts the whole graph rather than taking the handler,
            # which is how a head_dim=256 model (Gemma-2B in Pi0) failed to
            # compile at all.
            if q.shape[-1] > _CUDNN_MAX_HEAD_DIM:
                # EFFICIENT_ATTENTION requires an additive bias in the query's
                # dtype ("invalid dtype for bias - should match query's dtype");
                # cuDNN tolerated an fp32 mask against bf16 q/k/v. A bool mask is
                # passed through -- coercing it would turn True/False into a
                # 1.0/0.0 additive bias.
                if attn_mask is not None and attn_mask.dtype not in (torch.bool, q.dtype):
                    attn_mask = attn_mask.to(dtype=q.dtype)
                with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                    out = F.scaled_dot_product_attention(
                        q, k, v,
                        attn_mask=attn_mask,
                        dropout_p=0.0,
                        is_causal=causal,
                        scale=softmax_scale,
                    )
            else:
                try:
                    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                        out = F.scaled_dot_product_attention(
                            q, k, v,
                            attn_mask=attn_mask,
                            dropout_p=0.0,
                            is_causal=causal,
                            scale=softmax_scale,
                        )
                except RuntimeError:
                    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                        out = F.scaled_dot_product_attention(
                            q, k, v,
                            attn_mask=attn_mask,
                            dropout_p=0.0,
                            is_causal=causal,
                            scale=softmax_scale,
                        )
        else:
            # SDPA accepts a boolean mask (True = attend) directly; only a float
            # (additive) mask needs dtype coercion. Coercing a bool mask to q.dtype
            # would turn True/False into a 1.0/0.0 additive bias (wrong semantics) --
            # e.g. the HunyuanVideo key-padding mask would then fail to mask padding
            # on non-cuDNN backends.
            if attn_mask is not None and attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.to(dtype=q.dtype)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False if attn_mask is not None else causal,
                scale=softmax_scale,
            )
        return out.permute(0, 2, 1, 3)

def oasis_rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)

def oasis_apply_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    dtype = t.dtype
    rot_dim = freqs.shape[-1]
    t_left = t[..., :0]
    t_middle = t[..., :rot_dim]
    t_right = t[..., rot_dim:]
    t_transformed = (t_middle * freqs.cos()) + (oasis_rotate_half(t_middle) * freqs.sin())
    return torch.cat((t_left, t_transformed, t_right), dim=-1).to(dtype)

class OasisRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        freqs_for: str = "lang",
        theta: float = 10000.0,
        max_freq: float = 10.0,
    ):
        super().__init__()
        self.dim = dim
        self.freqs_for = freqs_for
        if freqs_for == "lang":
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        else:
            raise ValueError(f"unsupported rotary mode: {freqs_for}")
        self.freqs = nn.Parameter(freqs, requires_grad=False)
        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    @property
    def device(self) -> torch.device:
        return self.dummy.device

    def _forward_freqs(self, positions: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        freqs = torch.einsum("..., f -> ... f", positions.to(freqs.dtype), freqs)
        return freqs.repeat_interleave(2, dim=-1)

    def forward(
        self,
        t: torch.Tensor,
        freqs: torch.Tensor,
        seq_len: int | None = None,
        offset: int = 0,
    ) -> torch.Tensor:
        del seq_len, offset
        return self._forward_freqs(t, freqs)

    def rotate_queries_or_keys(self, t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        seq_len = t.shape[-2]
        positions = torch.arange(seq_len, device=t.device, dtype=t.dtype)
        seq_freqs = self.forward(positions, freqs, seq_len=seq_len)
        return oasis_apply_rotary_emb(seq_freqs, t)

    def get_axial_freqs(self, *dims: int) -> torch.Tensor:
        colon = slice(None)
        all_freqs = []
        for index, dim in enumerate(dims):
            use_pixel = self.freqs_for == "pixel" and index >= len(dims) - 2
            if use_pixel:
                pos = torch.linspace(-1, 1, steps=dim, device=self.device)
            else:
                pos = torch.arange(dim, device=self.device)
            seq_freqs = self.forward(pos, self.freqs, seq_len=dim)
            axis = [None] * len(dims)
            axis[index] = colon
            all_freqs.append(seq_freqs[(Ellipsis, *axis, colon)])
        all_freqs = torch.broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim=-1)

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

class OasisTemporalAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: OasisRotaryEmbedding,
        *,
        is_causal: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.to_qkv = Linear(dim, dim_head * heads * 3, bias=False)
        self.to_out = Linear(dim_head * heads, dim, bias=True)
        self.rotary_emb = rotary_emb
        self.is_causal = is_causal
        self.attn = DenseAttention(backend="sdpa")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, time, height, width, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.reshape(bsz, time, height, width, self.heads, -1).permute(0, 2, 3, 4, 1, 5)
        k = k.reshape(bsz, time, height, width, self.heads, -1).permute(0, 2, 3, 4, 1, 5)
        v = v.reshape(bsz, time, height, width, self.heads, -1).permute(0, 2, 3, 4, 1, 5)

        q = q.reshape(bsz * height * width, self.heads, time, -1)
        k = k.reshape(bsz * height * width, self.heads, time, -1)
        v = v.reshape(bsz * height * width, self.heads, time, -1)

        q = self.rotary_emb.rotate_queries_or_keys(q, self.rotary_emb.freqs)
        k = self.rotary_emb.rotate_queries_or_keys(k, self.rotary_emb.freqs)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = self.attn(q, k, v, causal=self.is_causal)
        out = out.reshape(bsz, height, width, time, self.heads, -1)
        out = out.permute(0, 3, 1, 2, 4, 5).reshape(bsz, time, height, width, -1)
        return self.to_out(out.to(q.dtype))

class OasisSpatialAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: OasisRotaryEmbedding,
    ):
        super().__init__()
        self.heads = heads
        self.to_qkv = Linear(dim, dim_head * heads * 3, bias=False)
        self.to_out = Linear(dim_head * heads, dim, bias=True)
        self.rotary_emb = rotary_emb
        self.attn = DenseAttention(backend="sdpa")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, time, height, width, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.reshape(bsz * time, height, width, self.heads, -1).permute(0, 3, 1, 2, 4)
        k = k.reshape(bsz * time, height, width, self.heads, -1).permute(0, 3, 1, 2, 4)
        v = v.reshape(bsz * time, height, width, self.heads, -1).permute(0, 3, 1, 2, 4)

        freqs = self.rotary_emb.get_axial_freqs(height, width)
        q = oasis_apply_rotary_emb(freqs, q)
        k = oasis_apply_rotary_emb(freqs, k)

        q = q.reshape(bsz * time, self.heads, height * width, -1).transpose(1, 2)
        k = k.reshape(bsz * time, self.heads, height * width, -1).transpose(1, 2)
        v = v.reshape(bsz * time, self.heads, height * width, -1).transpose(1, 2)
        out = self.attn(q, k, v, causal=False)
        out = out.reshape(bsz, time, height, width, self.heads, -1).reshape(bsz, time, height, width, -1)
        return self.to_out(out.to(q.dtype))

class GELU(nn.Module):
    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate=self.approximate)

class OasisMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        *,
        approximate_tanh: bool = False,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = Linear(in_features, hidden_features, bias=True)
        self.act = GELU(approximate="tanh" if approximate_tanh else "none")
        self.fc2 = Linear(hidden_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))

class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)

class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        create_scale: bool = True,
        create_offset: bool = True,
        promote_fp32: bool = True,
    ):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.promote_fp32 = promote_fp32

        if elementwise_affine and create_scale:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter("weight", None)

        if elementwise_affine and create_offset:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("bias", None)

        # fp32 views of weight/bias, filled on the first forward (see there).
        # A separate flag rather than a None check on _w32, so a module with no
        # affine params does not retry the cast every call.
        self._cast_done = False
        self._src_w: torch.Tensor | None = None
        self._src_b: torch.Tensor | None = None
        self._w32: torch.Tensor | None = None
        self._b32: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.promote_fp32:
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps,
            )

        # Promote to fp32 for the reduction to match vLLM's
        # ``vllm/model_executor/layers/layernorm.py:LayerNorm`` which keeps
        # ``weight`` / ``bias`` in fp32 and runs the reduction in fp32.
        # Matters for the DeepSeek-V3.2 indexer ``k_norm`` — running the
        # reduction in bf16 biases the variance enough to shift the
        # FP8-quantized indexer K cache, which in turn changes the top-2048
        # selection in every sparse layer.
        orig_dtype = x.dtype
        # Cast weight/bias to fp32 ONCE, not per call. These are parameters, so
        # the cast is loop-invariant, but re-running it cost ~50 kernel launches
        # per decode step across the 21 indexer compute layers -- the same defect
        # as the indexer rope re-casting its cos/sin cache every call. Weight
        # loading completes before the first forward, so a lazy cache is safe.
        # Re-derive if the parameter object was replaced or moved. ``_w32`` is a
        # plain attribute, not a buffer, so ``module.to(device)`` would not move
        # it -- 72 modules across the tree use this op, and a stale cache there
        # would be a device mismatch (or worse, silently old weights). The guard
        # is two identity compares, ~100 ns against the ~2 us kernel launch it
        # saves.
        if (not self._cast_done
                or self._src_w is not self.weight
                or self._src_b is not self.bias):
            w, b = self.weight, self.bias
            self._src_w, self._src_b = w, b
            self._w32 = (w.float()
                         if w is not None and w.dtype != torch.float32 else w)
            self._b32 = (b.float()
                         if b is not None and b.dtype != torch.float32 else b)
            self._cast_done = True
        weight, bias = self._w32, self._b32
        return F.layer_norm(
            x.float(), self.normalized_shape, weight, bias, self.eps,
        ).to(orig_dtype)

"""Oasis DiT blocks."""

from __future__ import annotations

import torch
import torch.nn as nn



def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    fixed_dims = [1] * len(shift.shape[1:])
    shift = shift.repeat(x.shape[0] // shift.shape[0], *fixed_dims)
    scale = scale.repeat(x.shape[0] // shift.shape[0], *fixed_dims)
    while shift.dim() < x.dim():
        shift = shift.unsqueeze(-2)
        scale = scale.unsqueeze(-2)
    return x * (1 + scale) + shift


def _gate(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    fixed_dims = [1] * len(g.shape[1:])
    g = g.repeat(x.shape[0] // g.shape[0], *fixed_dims)
    while g.dim() < x.dim():
        g = g.unsqueeze(-2)
    return g * x


class Model(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        is_causal: bool = True,
        spatial_rotary_emb: OasisRotaryEmbedding,
        temporal_rotary_emb: OasisRotaryEmbedding,
    ):
        super().__init__()
        self.s_norm1 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.s_attn = OasisSpatialAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            rotary_emb=spatial_rotary_emb,
        )
        self.s_norm2 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.s_mlp = OasisMLP(
            hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            approximate_tanh=True,
        )
        self.s_adaLN_modulation = nn.Sequential(
            SiLU(),
            Linear(hidden_size, 6 * hidden_size, bias=True),
        )

        self.t_norm1 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.t_attn = OasisTemporalAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            rotary_emb=temporal_rotary_emb,
            is_causal=is_causal,
        )
        self.t_norm2 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.t_mlp = OasisMLP(
            hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            approximate_tanh=True,
        )
        self.t_adaLN_modulation = nn.Sequential(
            SiLU(),
            Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        s_shift_msa, s_scale_msa, s_gate_msa, s_shift_mlp, s_scale_mlp, s_gate_mlp = (
            self.s_adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + _gate(self.s_attn(_modulate(self.s_norm1(x), s_shift_msa, s_scale_msa)), s_gate_msa)
        x = x + _gate(self.s_mlp(_modulate(self.s_norm2(x), s_shift_mlp, s_scale_mlp)), s_gate_mlp)

        t_shift_msa, t_scale_msa, t_gate_msa, t_shift_mlp, t_scale_mlp, t_gate_mlp = (
            self.t_adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + _gate(self.t_attn(_modulate(self.t_norm1(x), t_shift_msa, t_scale_msa)), t_gate_msa)
        x = x + _gate(self.t_mlp(_modulate(self.t_norm2(x), t_shift_mlp, t_scale_mlp)), t_gate_mlp)
        return x

    ```

ModelNew must use the same `__init__` arguments and `forward` signature as `Model`.

The kernel will be evaluated on these captured input shapes:

### SpatioTemporalDiTBlock

| count | args |
|------:|------|
| 320 | `x:float16[1, 2, 9, 16, 1024] c:float16[1, 2, 1024]` |
| 320 | `x:float16[1, 3, 9, 16, 1024] c:float16[1, 3, 1024]` |
| 320 | `x:float16[1, 4, 9, 16, 1024] c:float16[1, 4, 1024]` |
| 320 | `x:float16[1, 5, 9, 16, 1024] c:float16[1, 5, 1024]` |
| 320 | `x:float16[1, 6, 9, 16, 1024] c:float16[1, 6, 1024]` |

    
Optimize the architecture named Model with custom Triton operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Let's think step by step.
