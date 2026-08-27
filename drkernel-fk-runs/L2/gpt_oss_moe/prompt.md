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
from dataclasses import dataclass
from flashinfer import trtllm_fp4_block_scale_moe
from typing import Any
from typing import Optional
import functools
import importlib.util
import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

def trtllm_mxfp4_moe_supported() -> bool:
    """True when the trtllm-gen MXFP4 MoE kernel can run on this device.

    Same gate as vLLM's ``TrtLlmMxfp4ExpertsBase._supports_current_device``:
    CUDA, SM100 family, FlashInfer present.

    ``FASTKERNELS_TRTLLM_MXFP4_MOE=0`` forces the Triton path instead. This
    kernel segfaults inside ``flashinfer::FP4BlockScaleLauncher::run`` on the
    first autotune profile for gpt-oss-120b at tp=1 with a 16384-token chunk
    (fine at tp=2, and fine at tp=1 with short prompts), so a switch is needed
    to isolate it and to keep that configuration runnable.
    """
    if os.environ.get("FASTKERNELS_TRTLLM_MXFP4_MOE", "1") == "0":
        return False
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 10

_MXFP4_SF_BLOCK = 32

def _swap_every_two_rows(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Swap adjacent pairs along ``axis`` (trtllm-gen's SwiGLU half order)."""
    shape = x.shape
    if axis < 0:
        axis = len(shape) + axis
    new_shape = list(shape)
    new_shape[axis] = shape[axis] // 2
    new_shape.insert(axis + 1, 2)
    x = x.reshape(*new_shape)
    x = x.flip(axis + 1)
    return x.reshape(*shape)

_EPILOGUE_TILE_M = 128

def prepare_trtllm_mxfp4_weights(
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w13_bias: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w2_bias: torch.Tensor,
    permute_cache: dict | None = None,
) -> tuple[torch.Tensor, ...]:
    """Shuffle loaded MXFP4 expert weights into the trtllm-gen layout.

    Expects the *padded* shapes the kernel is configured for:
      ``w13_weight``       ``[E, 2*I, H // 2]``   uint8
      ``w13_weight_scale`` ``[E, 2*I, H // 32]``  uint8 (E8M0)
      ``w13_bias``         ``[E, 2*I]``
      ``w2_weight``        ``[E, H, I // 2]``     uint8
      ``w2_weight_scale``  ``[E, H, I // 32]``    uint8 (E8M0)
      ``w2_bias``          ``[E, H]``

    Returns the same six tensors in kernel layout, with the scales viewed as
    ``float8_e4m3fn`` and the biases in float32.
    """
    from flashinfer.fp4_quantization import nvfp4_block_scale_interleave
    from flashinfer.fused_moe.core import get_w2_permute_indices_with_cache

    if permute_cache is None:
        permute_cache = {}

    num_experts = w13_weight.shape[0]
    intermediate_size = w13_weight.shape[1] // 2
    hidden_size = w13_weight.shape[2] * 2

    w13_bias = w13_bias.to(torch.float32)
    w2_bias = w2_bias.to(torch.float32)

    # trtllm-gen's SwiGLU takes the two halves in the opposite order.
    w13_weight_scale = _swap_every_two_rows(w13_weight_scale, -2)
    w13_weight = _swap_every_two_rows(w13_weight, -2)
    w13_bias = _swap_every_two_rows(w13_bias, -1)

    g1_w, g1_s, g1_b = [], [], []
    g2_w, g2_s, g2_b = [], [], []
    for i in range(num_experts):
        idx = get_w2_permute_indices_with_cache(
            permute_cache, w13_weight[i].view(torch.uint8), _EPILOGUE_TILE_M,
        )
        g1_w.append(
            w13_weight[i].view(torch.uint8)[idx.to(w13_weight.device)].contiguous()
        )
        sf_idx = get_w2_permute_indices_with_cache(
            permute_cache, w13_weight_scale[i].view(torch.uint8),
            _EPILOGUE_TILE_M, num_elts_per_sf=16,
        )
        g1_s.append(
            nvfp4_block_scale_interleave(
                w13_weight_scale[i]
                .view(torch.uint8)[sf_idx.to(w13_weight_scale.device)]
                .contiguous()
            )
        )
        b_idx = get_w2_permute_indices_with_cache(
            permute_cache, w13_bias[i].clone().reshape(-1, 1), _EPILOGUE_TILE_M,
        )
        g1_b.append(
            w13_bias[i].clone().reshape(-1, 1)[b_idx.to(w13_bias.device)].contiguous()
        )

        idx = get_w2_permute_indices_with_cache(
            permute_cache, w2_weight[i].view(torch.uint8), _EPILOGUE_TILE_M,
        )
        g2_w.append(
            w2_weight[i].view(torch.uint8)[idx.to(w2_weight.device)].contiguous()
        )
        sf_idx = get_w2_permute_indices_with_cache(
            permute_cache, w2_weight_scale[i].view(torch.uint8),
            _EPILOGUE_TILE_M, num_elts_per_sf=16,
        )
        g2_s.append(
            nvfp4_block_scale_interleave(
                w2_weight_scale[i]
                .view(torch.uint8)[sf_idx.to(w2_weight_scale.device)]
                .contiguous()
            )
        )
        b_idx = get_w2_permute_indices_with_cache(
            permute_cache, w2_bias[i].clone().reshape(-1, 1), _EPILOGUE_TILE_M,
        )
        g2_b.append(
            w2_bias[i].clone().reshape(-1, 1)[b_idx.to(w2_bias.device)].contiguous()
        )

    w13_weight = torch.stack(g1_w)
    w13_weight_scale = (
        torch.stack(g1_s)
        .reshape(num_experts, 2 * intermediate_size, hidden_size // _MXFP4_SF_BLOCK)
        .view(torch.float8_e4m3fn)
    )
    w2_weight = torch.stack(g2_w)
    w2_weight_scale = (
        torch.stack(g2_s)
        .reshape(num_experts, hidden_size, intermediate_size // _MXFP4_SF_BLOCK)
        .view(torch.float8_e4m3fn)
    )
    w13_bias = torch.stack(g1_b).reshape(num_experts, -1)
    w2_bias = torch.stack(g2_b).reshape(num_experts, -1)
    return (
        w13_weight, w13_weight_scale, w13_bias,
        w2_weight, w2_weight_scale, w2_bias,
    )

DEFAULT_TUNE_MAX_NUM_TOKENS = 1024

ROUTING_RENORMALIZE_NAIVE = 4

SWIGLU_LIMIT = 7.0

SWIGLU_ALPHA = 1.702

SWIGLU_BETA = 1.0

class TrtLlmMxfp4MoE(nn.Module):
    """Router + experts in one trtllm-gen launch.

    ``hidden_states`` arrives at the *padded* hidden width; the returned tensor
    is ``hidden_size_unpadded`` wide, matching vLLM's ``has_unpadded_output``.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        intermediate_size: int,
        hidden_size_unpadded: int,
        max_capture_size: int = DEFAULT_TUNE_MAX_NUM_TOKENS,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.intermediate_size = intermediate_size
        self.hidden_size_unpadded = hidden_size_unpadded
        self.max_capture_size = max(int(max_capture_size), 1)
        dev = torch.cuda.current_device()
        # Per-expert scalars, exactly as TrtLlmMxfp4ExpertsBase builds them.
        self.register_buffer(
            "gemm1_alpha",
            torch.full((num_experts,), SWIGLU_ALPHA, dtype=torch.float32, device=dev),
            persistent=False,
        )
        self.register_buffer(
            "gemm1_beta",
            torch.full((num_experts,), SWIGLU_BETA, dtype=torch.float32, device=dev),
            persistent=False,
        )
        self.register_buffer(
            "gemm1_clamp_limit",
            torch.full((num_experts,), SWIGLU_LIMIT, dtype=torch.float32, device=dev),
            persistent=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        w13_weight: torch.Tensor,
        w13_weight_scale: torch.Tensor,
        w13_bias: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_scale: torch.Tensor,
        w2_bias: torch.Tensor,
    ) -> torch.Tensor:
        assert hidden_states.dtype == torch.bfloat16
        output = torch.empty(
            *hidden_states.shape[:-1],
            self.hidden_size_unpadded,
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )
        trtllm_fp4_block_scale_moe(
            routing_logits=router_logits.to(torch.bfloat16),
            routing_bias=None,
            hidden_states=hidden_states,
            hidden_states_scale=None,
            gemm1_weights=w13_weight,
            gemm1_weights_scale=w13_weight_scale,
            gemm1_bias=w13_bias,
            gemm1_alpha=self.gemm1_alpha,
            gemm1_beta=self.gemm1_beta,
            gemm1_clamp_limit=self.gemm1_clamp_limit,
            gemm2_weights=w2_weight,
            gemm2_weights_scale=w2_weight_scale,
            gemm2_bias=w2_bias,
            output1_scale_scalar=None,
            output1_scale_gate_scalar=None,
            output2_scale_scalar=None,
            num_experts=self.num_experts,
            top_k=self.top_k,
            n_group=None,
            topk_group=None,
            intermediate_size=self.intermediate_size,
            local_expert_offset=0,
            local_num_experts=self.num_experts,
            routed_scaling_factor=None,
            routing_method_type=ROUTING_RENORMALIZE_NAIVE,
            do_finalize=True,
            tune_max_num_tokens=self.max_capture_size,
            output=output,
        )
        return output

TRTLLM_MXFP4_ALIGN = 256

def _ensure_triton_kernels_on_path() -> None:
    """Make the OpenAI ``triton_kernels`` package importable.

    ``triton_kernels`` is an external dependency; prefer an already-installed
    copy. Only if that is missing do we extend ``sys.path`` with an explicit
    override (``FASTKERNELS_TRITON_KERNELS_PATH``) or, as a last resort, vLLM's
    bundled ``third_party`` dir -- without requiring the vLLM package.
    """
    if importlib.util.find_spec("triton_kernels") is not None:
        return

    candidates: list[str] = []
    override = os.environ.get("FASTKERNELS_TRITON_KERNELS_PATH")
    if override:
        candidates.append(override)

    # Last resort: vLLM bundles triton_kernels under third_party. Locate it
    # only if vLLM happens to be installed; never call into vLLM.
    vllm_spec = importlib.util.find_spec("vllm")
    if vllm_spec is not None and vllm_spec.origin is not None:
        candidates.append(
            os.path.join(os.path.dirname(vllm_spec.origin), "third_party")
        )

    for path in candidates:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)

class Mxfp4MoEQuantConfig:
    """Minimal quant config carrying the per-MoE precision/bias tensors.

    Attribute names match the subset of ``FusedMoEQuantConfig`` consumed
    by ``triton_kernel_fused_experts`` (``w{1,2}_precision`` and
    ``w{1,2}_bias``), so the call sites stay essentially unchanged.
    """

    w1_precision: Any  # triton_kernels.matmul_ogs.PrecisionConfig
    w2_precision: Any  # triton_kernels.matmul_ogs.PrecisionConfig
    w1_bias: torch.Tensor | None = None
    w2_bias: torch.Tensor | None = None

def _resize_cache(x: torch.Tensor, v: tuple[int, ...]) -> torch.Tensor:
    """Shrink ``x`` and reshape it to ``v``. Used for intermediate caches."""
    n = 1
    for d in v:
        n *= d
    assert n <= x.numel(), f"{v} ({n}) <= {x.shape} ({x.numel()})"
    return x.flatten()[:n].view(*v)

_MATMUL_OGS_ROW_TILE = 128

def _tile_rows(rows: int) -> int:
    """Round a ragged-operand row count up to ``matmul_ogs``'s tile."""
    tile = _MATMUL_OGS_ROW_TILE
    return ((rows + tile - 1) // tile) * tile

def _fused_experts(
    output_tensor: torch.Tensor,
    hidden_states: torch.Tensor,
    w1,
    w2,
    routing_data,
    gather_indx,
    scatter_indx,
    topk: int,
    quant_config: Mxfp4MoEQuantConfig,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    """Run the two fused MXFP4 matmuls with OAI SwiGLU in between."""
    _ensure_triton_kernels_on_path()
    import triton_kernels.swiglu
    from triton_kernels.matmul_ogs import FnSpecs, FusedActivation, matmul_ogs

    assert hidden_states.dtype == torch.bfloat16
    assert quant_config.w1_bias is None or quant_config.w1_bias.dtype == torch.float32
    assert quant_config.w2_bias is None or quant_config.w2_bias.dtype == torch.float32
    assert hidden_states.ndim == 2
    assert hidden_states.shape[-1] == w1.shape[-2]
    assert w2.shape[-1] == w1.shape[1]

    batch_dim = 1
    M, K = hidden_states.shape[-2:]
    _, _, N = w1.shape

    intermediate_cache = torch.empty(
        (batch_dim, _tile_rows(M * topk), N // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache = _resize_cache(intermediate_cache, (batch_dim, M * topk, N // 2))
    output_tensor = _resize_cache(output_tensor, (batch_dim, M, K))

    # ``reduction_n`` is an argument of ``FusedActivation`` (positional, after
    # the activation args), not of ``FnSpecs``, in the bundled triton_kernels.
    act = FusedActivation(
        FnSpecs("swiglu", triton_kernels.swiglu.swiglu_fn, ("alpha", "limit")),
        (swiglu_alpha, swiglu_limit),
        2,
    )
    gammas = routing_data.gate_scal if routing_data else None

    matmul_ogs(
        hidden_states,
        w1,
        quant_config.w1_bias,
        routing_data,
        gather_indx=gather_indx,
        precision_config=quant_config.w1_precision,
        gammas=gammas if apply_router_weight_on_input else None,
        fused_activation=act,
        y=intermediate_cache,
    )
    matmul_ogs(
        intermediate_cache.view(M * topk, N // 2),
        w2,
        quant_config.w2_bias,
        routing_data,
        scatter_indx=scatter_indx,
        precision_config=quant_config.w2_precision,
        gammas=None if apply_router_weight_on_input else gammas,
        y=output_tensor,
    )
    return output_tensor.view(M, K)

def _swizzle_mxfp4(quant_tensor: torch.Tensor, scale: torch.Tensor, num_warps: int):
    """Swizzle MXFP4 weight + E8M0 scales into the layout matmul_ogs wants.

    Returns ``(packed_tensor, in_flex_data, scale_tensor)`` where the two
    tensor returns are ``triton_kernels.tensor.Tensor`` wrappers, ready
    to be plugged into a ``PrecisionConfig``.

    Copied from ``vllm.model_executor.layers.quantization.utils.mxfp4_utils._swizzle_mxfp4``,
    minus the ROCm/Hopper-old-torch fallbacks that FastKernels does not exercise.
    """
    _ensure_triton_kernels_on_path()
    import triton_kernels.matmul_ogs_details.opt_flags as opt_flags
    from triton_kernels.numerics import InFlexData
    from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
    from triton_kernels.tensor_details import layout

    cap = torch.cuda.get_device_capability()

    value_layout_opts: dict[str, Any] = {}
    scale_layout_opts: dict[str, Any] = {}
    value_layout, value_layout_opts = layout.make_default_matmul_mxfp4_w_layout(
        mx_axis=1
    )
    scale_layout, scale_layout_opts = layout.make_default_matmul_mxfp4_w_scale_layout(
        mx_axis=1, num_warps=num_warps
    )

    if cap[0] == 9:
        opt_flags.update_opt_flags_constraints({"split_k": 1})
    elif cap[0] == 10:
        opt_flags.update_opt_flags_constraints(
            {"is_persistent": True, "epilogue_subtile": 1}
        )

    # transpose so the quantization axis is on dim 1
    quant_tensor = quant_tensor.transpose(-2, -1)
    scale = scale.transpose(-2, -1)
    quant_tensor = convert_layout(
        wrap_torch_tensor(quant_tensor, dtype=FP4),
        value_layout,
        **value_layout_opts,
    )
    scale = convert_layout(
        wrap_torch_tensor(scale), scale_layout, **scale_layout_opts
    )
    return quant_tensor, InFlexData(), scale

def _routing_from_logits(logits: torch.Tensor, n_expts_act: int, sm_first: bool):
    """Compute ``(RoutingData, GatherIndx, ScatterIndx)`` from gating logits.

    Delegates to ``triton_kernels.routing.routing``, which fuses softmax, top-k,
    bitmatrix packing and routing-metadata construction into a single launch.
    This is the same entry point vLLM's GPT-OSS MoE uses; earlier revisions of
    this file reimplemented it against a ``SparseMatrix`` /
    ``make_ragged_tensor_metadata`` API that has since been removed from
    ``triton_kernels``.
    """
    _ensure_triton_kernels_on_path()
    from triton_kernels.routing import routing

    return routing(logits, n_expts_act, sm_first=sm_first)

class Mxfp4MoE(nn.Module):
    """MXFP4-quantized fused MoE primitive (routing + matmul_ogs experts).

    The module is stateless -- expert weights, biases, and the
    :class:`Mxfp4MoEQuantConfig` are passed to ``forward`` so a single
    instance can serve any number of MoE layers. Weight preparation is
    exposed as static helpers so the L2 caller does not need to import
    ``triton_kernels`` directly.
    """

    @staticmethod
    def prepare_weight(
        quant_tensor: torch.Tensor,
        scale: torch.Tensor,
        num_warps: int = 8,
    ):
        """Swizzle an MXFP4 expert weight and build its ``PrecisionConfig``.

        Returns ``(swizzled_weight, precision_config)`` ready to feed
        into :meth:`make_quant_config` and :meth:`forward`.
        """
        _ensure_triton_kernels_on_path()
        from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig

        weight, flex, scale_tensor = _swizzle_mxfp4(quant_tensor, scale, num_warps)
        precision = PrecisionConfig(
            weight_scale=scale_tensor, flex_ctx=FlexCtx(rhs_data=flex)
        )
        return weight, precision

    @staticmethod
    def make_quant_config(
        w1_precision: Any,
        w2_precision: Any,
        w1_bias: torch.Tensor | None = None,
        w2_bias: torch.Tensor | None = None,
    ) -> Mxfp4MoEQuantConfig:
        """Construct an MXFP4 W4A16 quant config from per-expert precisions/biases."""
        return Mxfp4MoEQuantConfig(
            w1_precision=w1_precision,
            w2_precision=w2_precision,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        w1,
        w2,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        quant_config: Mxfp4MoEQuantConfig,
        apply_router_weight_on_input: bool = False,
    ) -> torch.Tensor:
        """End-to-end MXFP4 MoE forward (routing + fused experts).

        ``w1``/``w2`` must already be swizzled (see :meth:`prepare_weight`)
        and ``quant_config`` must carry the matching precision configs and
        expert biases. ``hidden_states`` must be bfloat16 and 2D.
        """
        routing_data, gather_idx, scatter_idx = _routing_from_logits(
            gating_output, topk, sm_first=not renormalize
        )
        # Over-allocate the output rows for the same reason as the intermediate
        # cache above; ``_fused_experts`` narrows it back to ``M`` rows.
        output = torch.empty(
            (_tile_rows(hidden_states.shape[0]), hidden_states.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        return _fused_experts(
            output,
            hidden_states,
            w1,
            w2,
            routing_data,
            gather_idx,
            scatter_idx,
            topk=topk,
            quant_config=quant_config,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

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

_CUSTOM_AR: Optional["CustomAllreduce"] = None

class AllReduce(nn.Module):
    def forward(self, tensor):
        if torch.compiler.is_compiling():
            # Route through the custom op rather than falling straight to NCCL:
            # the op is opaque to inductor, so the custom IPC all-reduce still
            # runs inside a compiled graph. At decode message sizes (one token
            # wide) NCCL is far slower, which showed up as decode getting *worse*
            # from tp=1 to tp=2 while vLLM's improved.
            if _CUSTOM_AR is not None:
                return torch.ops.fastkernels.custom_all_reduce(tensor)
            dist.all_reduce(tensor)
            return tensor
        ar = _CUSTOM_AR
        if ar is not None:
            out = ar.custom_all_reduce(tensor)
            if out is not None:
                return out
        dist.all_reduce(tensor)
        return tensor

"""GPT-OSS MoE: MXFP4-native fused MoE composed from FastKernels L1 ops.

128 experts (top-4, softmax routing), router bias, expert gate/up/down biases,
OAI SwiGLU activation fused inside the expert kernel.

Expert weights are kept in packed MXFP4 uint8 format (2× FP4 per byte) with
E8M0 block scales. No dequantization is performed.

Two expert kernels exist, and which one vLLM picks depends on the device
(``Mxfp4MoEMethod`` -> ``select_deepseek_v4_mxfp4_moe_backend``):

* **SM100** -> ``FLASHINFER_TRTLLM_MXFP4_BF16`` /
  ``TrtLlmMxfp4ExpertsMonolithic``, i.e. ``flashinfer.trtllm_fp4_block_scale_moe``
  (:mod:`..L1.trtllm_mxfp4_moe`). This needs hidden/intermediate rounded up to
  256 and a shuffled weight layout.
* **otherwise** -> the OAI Triton ``matmul_ogs`` kernel
  (:mod:`..L1.mxfp4_moe`).

Both are kept so the module matches vLLM on whichever device it runs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fastkernels.infra.tp import _tp_rank, _tp_size


def _round_up(x: int, align: int) -> int:
    return (x + align - 1) // align * align


class Model(nn.Module):
    """MXFP4-native MoE composed from FastKernels L1 ops.

    Weights stay in packed uint8 MXFP4 format. Routing, swizzling and the
    fused matmul_ogs forward are all delegated to ``L1.mxfp4_moe``.
    """

    MXFP4_BLOCK = 32

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        tp = _tp_size()
        self.tp_size = tp
        self.intermediate_per_tp = config.intermediate_size // tp

        self.router = Linear(config.hidden_size, config.num_local_experts, bias=True)

        E = config.num_local_experts
        BLK = self.MXFP4_BLOCK

        # Which expert kernel, and therefore which alignment. vLLM's
        # ``mxfp4_round_up_hidden_size_and_intermediate_size`` uses 256 for the
        # TRTLLM backends and 64 for the Triton one, and it pads *hidden* too
        # (2880 -> 3072 for gpt-oss), not just the intermediate.
        self.use_trtllm = trtllm_mxfp4_moe_supported()
        if self.use_trtllm:
            I_pad = _round_up(self.intermediate_per_tp, TRTLLM_MXFP4_ALIGN)
            H_pad = _round_up(self.hidden_size, TRTLLM_MXFP4_ALIGN)
        else:
            I_pad = _round_up(self.intermediate_per_tp, 64)
            H_pad = self.hidden_size
        H = H_pad

        self._I_pad = I_pad
        self._H_pad = H_pad

        # Expert weights in packed MXFP4 uint8 (2× FP4 per byte)
        self.w13_weight = nn.Parameter(
            torch.zeros(E, 2 * I_pad, H // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        self.w13_weight_scale = nn.Parameter(
            torch.zeros(E, 2 * I_pad, H // BLK, dtype=torch.uint8),
            requires_grad=False,
        )
        self.w13_bias = nn.Parameter(
            torch.zeros(E, 2 * I_pad, dtype=torch.bfloat16),
            requires_grad=False,
        )

        self.w2_weight = nn.Parameter(
            torch.zeros(E, H, I_pad // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        self.w2_weight_scale = nn.Parameter(
            torch.zeros(E, H, I_pad // BLK, dtype=torch.uint8),
            requires_grad=False,
        )
        self.w2_bias = nn.Parameter(
            torch.zeros(E, H, dtype=torch.bfloat16),
            requires_grad=False,
        )

        # Set up weight loaders for checkpoint loading
        self.w13_weight.weight_loader = self._w13_weight_loader
        self.w13_weight_scale.weight_loader = self._w13_scale_loader
        self.w13_bias.weight_loader = self._w13_bias_loader
        self.w2_weight.weight_loader = self._w2_weight_loader
        self.w2_weight_scale.weight_loader = self._w2_scale_loader
        self.w2_bias.weight_loader = self._w2_bias_loader

        self.allreduce = AllReduce()
        self.mxfp4_moe = Mxfp4MoE()
        self.trtllm_moe = (
            TrtLlmMxfp4MoE(
                num_experts=E,
                top_k=self.top_k,
                intermediate_size=I_pad,
                hidden_size_unpadded=self.hidden_size,
            )
            if self.use_trtllm
            else None
        )

        # Populated after process_weights_after_loading
        self._quant_config = None
        self._processed = False

        # Custom-op dispatch for torch.compile (set by engine after model init)
        self._use_custom_op = False
        self._layer_name = ""

    def _w13_weight_loader(self, param, loaded_weight):
        """Load w13 MXFP4 packed weight with TP sharding.

        Checkpoint shape: [E, 2*I_full, num_blocks, 16] (4D blocks) or
                          [E, 2*I_full, H//2] (pre-flattened).
        Gate/up rows are interleaved (gate_0, up_0, gate_1, up_1, ...);
        we keep them interleaved, matching the expert kernel's expectation.

        The destination may be padded in *both* dims (the TRTLLM backend rounds
        hidden and intermediate up to 256), so the copy is bounded by the
        checkpoint's own extents and the padding stays zero.
        """
        if loaded_weight.ndim == 4:
            E, N, nb, bs = loaded_weight.shape
            loaded_weight = loaded_weight.reshape(E, N, nb * bs)
        rank = _tp_rank()
        I = self.intermediate_per_tp
        start = 2 * rank * I
        k = loaded_weight.shape[-1]
        param.data[:, :2*I, :k].copy_(loaded_weight[:, start : start + 2*I, :])

    def _w13_scale_loader(self, param, loaded_weight):
        """Load w13 scales with TP shard, keeping interleaved layout."""
        rank = _tp_rank()
        I = self.intermediate_per_tp
        start = 2 * rank * I
        k = loaded_weight.shape[-1]
        param.data[:, :2*I, :k].copy_(loaded_weight[:, start : start + 2*I, :])

    def _w13_bias_loader(self, param, loaded_weight):
        """Load w13 bias [E, 2*I] with TP shard, keeping interleaved layout."""
        rank = _tp_rank()
        I = self.intermediate_per_tp
        start = 2 * rank * I
        param.data[:, :2*I].copy_(loaded_weight[:, start : start + 2*I])

    def _w2_weight_loader(self, param, loaded_weight):
        """Load w2 MXFP4 packed weight with TP shard.

        Checkpoint shape: [E, H, num_blocks, 16] (4D blocks) or
                          [E, H, I//2] (pre-flattened).
        """
        if loaded_weight.ndim == 4:
            E, H, nb, bs = loaded_weight.shape
            loaded_weight = loaded_weight.reshape(E, H, nb * bs)
        tp, rank = _tp_size(), _tp_rank()
        I_half = self.intermediate_per_tp // 2
        h = loaded_weight.shape[1]
        param.data[:, :h, :I_half].copy_(
            loaded_weight[:, :, rank * I_half : rank * I_half + I_half]
        )

    def _w2_scale_loader(self, param, loaded_weight):
        """Load w2 scales with TP shard."""
        tp, rank = _tp_size(), _tp_rank()
        I_blk = self.intermediate_per_tp // self.MXFP4_BLOCK
        h = loaded_weight.shape[1]
        param.data[:, :h, :I_blk].copy_(
            loaded_weight[:, :, rank * I_blk : rank * I_blk + I_blk]
        )

    def _w2_bias_loader(self, param, loaded_weight):
        """Load w2 bias [E, H]. Only rank 0 loads; others zero (reduced by allreduce)."""
        if _tp_rank() == 0:
            param.data[:, : loaded_weight.shape[1]].copy_(loaded_weight)
        else:
            param.data.zero_()

    def process_weights_after_loading(self):
        """Convert MXFP4 weights into the selected expert kernel's layout.

        Must be called after all weights are loaded and moved to GPU.
        """
        if self._processed:
            return

        if self.use_trtllm:
            # trtllm-gen wants float32 biases, a gate/up row swap, and the
            # shuffled/interleaved weight+scale layout for its transposed MMA
            # epilogue (vLLM's ``convert_gpt_oss_weight_to_mxfp4_moe_kernel_format``).
            (
                w13_weight, w13_scale, w13_bias,
                w2_weight, w2_scale, w2_bias,
            ) = prepare_trtllm_mxfp4_weights(
                self.w13_weight.data,
                self.w13_weight_scale.data,
                self.w13_bias.data,
                self.w2_weight.data,
                self.w2_weight_scale.data,
                self.w2_bias.data,
            )
            del self.w13_weight, self.w2_weight
            del self.w13_weight_scale, self.w2_weight_scale
            del self.w13_bias, self.w2_bias
            self._w13_shuffled = w13_weight
            self._w13_scale = w13_scale
            self._w13_bias_f32 = w13_bias
            self._w2_shuffled = w2_weight
            self._w2_scale = w2_scale
            self._w2_bias_f32 = w2_bias
            torch.cuda.empty_cache()
            self._processed = True
            return

        # Biases must be float32 for the Triton kernel
        self.w13_bias.data = self.w13_bias.data.float()
        self.w2_bias.data = self.w2_bias.data.float()

        w13_weight, w13_precision = Mxfp4MoE.prepare_weight(
            self.w13_weight.data, self.w13_weight_scale.data
        )
        w2_weight, w2_precision = Mxfp4MoE.prepare_weight(
            self.w2_weight.data, self.w2_weight_scale.data
        )

        # prepare_weight returns triton_kernels.Tensor objects, not
        # torch.Tensor; store as plain attributes (the original nn.Parameters
        # are no longer used)
        del self.w13_weight, self.w2_weight
        del self.w13_weight_scale, self.w2_weight_scale
        self._w13_swizzled = w13_weight
        self._w2_swizzled = w2_weight

        self._quant_config = Mxfp4MoE.make_quant_config(
            w1_precision=w13_precision,
            w2_precision=w2_precision,
            w1_bias=self.w13_bias.data,
            w2_bias=self.w2_bias.data,
        )
        self._processed = True

    def forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self._processed:
            self.process_weights_after_loading()

        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)

        router_logits = self.router(hidden_states)

        if self.use_trtllm:
            # Zero-pad the activation into the kernel's hidden width; the
            # kernel writes an unpadded output (vLLM: forward padding in
            # ``MoERunner`` + ``has_unpadded_output``).
            if self._H_pad != self.hidden_size:
                hidden_states = torch.nn.functional.pad(
                    hidden_states, (0, self._H_pad - self.hidden_size),
                )
            output = self.trtllm_moe(
                hidden_states,
                router_logits,
                self._w13_shuffled,
                self._w13_scale,
                self._w13_bias_f32,
                self._w2_shuffled,
                self._w2_scale,
                self._w2_bias_f32,
            )
        else:
            output = self.mxfp4_moe(
                hidden_states=hidden_states,
                w1=self._w13_swizzled,
                w2=self._w2_swizzled,
                gating_output=router_logits,
                topk=self.top_k,
                renormalize=True,
                quant_config=self._quant_config,
                apply_router_weight_on_input=False,
            )

        if self.tp_size > 1 and not self._use_custom_op:
            output = self.allreduce(output)

        return output.view(orig_shape)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._use_custom_op:
            # The all-reduce stays *outside* the opaque op on purpose. Inside it
            # Inductor cannot see the collective, so the AR+RMSNorm fusion has
            # nothing to match at the MoE end of the layer -- half of every
            # layer's collectives. vLLM keeps its MoE reduction in traced Python
            # for the same reason (``moe_runner._maybe_reduce_final_output``).
            out = torch.ops.fastkernels.moe_forward(hidden_states, self._layer_name)
            if self.tp_size > 1:
                out = self.allreduce(out)
            return out
        return self.forward_impl(hidden_states)

    ```

ModelNew must use the same `__init__` arguments and `forward` signature as `Model`.

The kernel will be evaluated on these captured input shapes:

### GptOssMoE

| count | args |
|------:|------|
| 6876 | `hidden_states:bfloat16[60, 2880]` |
| 4680 | `hidden_states:bfloat16[1, 2880]` |
| 3096 | `hidden_states:bfloat16[16384, 2880]` |
| 1152 | `hidden_states:bfloat16[31, 2880]` |
| 1116 | `hidden_states:bfloat16[26, 2880]` |
| 1080 | `hidden_states:bfloat16[29, 2880]` |
| 684 | `hidden_states:bfloat16[30, 2880]` |
| 576 | `hidden_states:bfloat16[67, 2880]` |

    
Optimize the architecture named Model with custom Triton operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Let's think step by step.
