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
from fastkernels.infra.context import get_context
from fastkernels.infra.cuda_ext import lazy_op
from fastkernels.infra.fa_utils import FA_VERSION, flash_attn_varlen_func
from fastkernels.infra.tp import _tp_size, _tp_rank
from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla
from flashinfer.prefill import trtllm_ragged_attention_deepseek
from typing import Optional
from vllm.third_party.flashmla.flash_mla_interface import flash_mla_sparse_fwd
from vllm.third_party.flashmla.flash_mla_interface import flash_mla_with_kvcache, get_mla_metadata
from vllm.v1.attention.ops.flashmla import flash_mla_with_kvcache_fp8, get_mla_metadata_dense_fp8
import math
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

_FP8_BLOCK = 128

def _scale_shape(out_dim: int, in_dim: int) -> tuple[int, int]:
    return (math.ceil(out_dim / _FP8_BLOCK), math.ceil(in_dim / _FP8_BLOCK))

def _is_batch_invariant() -> bool:
    """vLLM's dynamic FP8 blockscale dispatch forces the DeepGEMM path (skips the
    FlashInfer swapAB kernel) for ALL M under batch-invariant determinism mode
    (``VLLM_BATCH_INVARIANT=1``) — see the early-out in
    ``scaled_mm/flashinfer.py`` and ``grouped_topk._is_batch_invariant``. Mirror
    it so fastkernels matches vLLM in that mode."""
    return os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1"

_FLASHINFER_RESOLVED = False

_FLASHINFER_FN: object | None = None

def _maybe_get_flashinfer_fp8_gemm():
    """Return ``flashinfer.gemm.fp8_blockscale_gemm_sm90`` if importable +
    enabled by env, otherwise ``None``.

    Mirrors vLLM's enablement gate in
    ``vllm/utils/flashinfer.py:is_flashinfer_fp8_blockscale_gemm_supported``:
    ``VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER`` (default 1) AND
    ``has_flashinfer_fp8_blockscale_gemm()`` (Hopper + flashinfer wheel
    exposes ``fp8_blockscale_gemm_sm90``).
    """
    global _FLASHINFER_RESOLVED, _FLASHINFER_FN
    if _FLASHINFER_RESOLVED:
        return _FLASHINFER_FN
    _FLASHINFER_RESOLVED = True

    if os.environ.get("VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER", "1") != "1":
        return None
    if not torch.cuda.is_available():
        return None
    cap = torch.cuda.get_device_capability()
    if cap[0] != 9:  # Hopper only — same gate as vLLM.
        return None
    from flashinfer.gemm import fp8_blockscale_gemm_sm90
    _FLASHINFER_FN = fp8_blockscale_gemm_sm90
    return _FLASHINFER_FN

def _alloc_colmajor_scale(M: int, num_groups: int,
                          device: torch.device) -> torch.Tensor:
    """Allocate a per-token-group scale tensor with column-major strides.

    Layout matches vLLM's DeepGEMM dense path
    (``vllm/model_executor/layers/quantization/utils/fp8_utils.py:914-918``):
    physical storage is ``(num_groups, M)`` row-major and we expose it as
    ``(M, num_groups)`` via ``.permute(-1, -2)`` so DeepGEMM's
    ``fp8_gemm_nt`` sees ``stride(0)==1`` SF — i.e. SF columns are
    contiguous, which the kernel expects for TMA loads.
    """
    return torch.empty(
        (num_groups, M), device=device, dtype=torch.float32,
    ).permute(-1, -2)

class _Fp8PrefillBufs:
    """Shared prefill buffers for FP8 activation quantization.

    Since decoder layers execute sequentially, a single set of buffers
    (sized for max_num_batched_tokens) can be reused across all Fp8Linear
    instances, eliminating per-layer dynamic allocation during prefill.
    One instance per unique (K, N) weight shape.

    The scale buffer ``s`` is column-major (matches vLLM's DeepGEMM path).
    """
    __slots__ = ("a", "s", "o")

    def __init__(self, max_tokens: int, K: int, N: int, device: torch.device):
        num_groups = math.ceil(K / 128)
        self.a = torch.empty(max_tokens, K, dtype=torch.float8_e4m3fn, device=device)
        self.s = _alloc_colmajor_scale(max_tokens, num_groups, device)
        self.o = torch.empty(max_tokens, N, dtype=torch.bfloat16, device=device)

class Fp8Linear(nn.Module):
    """Block-scaled FP8 linear using deep_gemm.fp8_gemm_nt.

    Weights are stored in float8_e4m3fn with pre-processed UE8M0 block scales
    (transformed via deep_gemm.transform_sf_into_required_layout at load time).
    Activations are dynamically quantized to FP8 per-token-group (group=128)
    using in-place ops for CUDA graph compatibility.
    """

    BLOCK_SIZE = 128

    def __init__(self):
        super().__init__()
        self._a_buf: torch.Tensor | None = None
        self._s_buf: torch.Tensor | None = None
        self._o_buf: torch.Tensor | None = None
        self._pf: _Fp8PrefillBufs | None = None

    def _ensure_buffers(self, max_tokens: int, K: int, N: int, device: torch.device):
        """Pre-allocate activation FP8 buffers for CUDA graph capture.

        The scale buffer is **column-major** to match vLLM's DeepGEMM
        dense FP8 path (see ``_alloc_colmajor_scale``).
        """
        num_groups = math.ceil(K / self.BLOCK_SIZE)
        self._a_buf = torch.empty(max_tokens, K, dtype=torch.float8_e4m3fn, device=device)
        self._s_buf = _alloc_colmajor_scale(max_tokens, num_groups, device)
        self._o_buf = torch.empty(max_tokens, N, dtype=torch.bfloat16, device=device)

    # Threshold matching vLLM's ``W8A8BlockFp8LinearOp.apply`` /
    # ``_flashinfer_fp8_blockscale_gemm_impl``: below this M the swapAB
    # kernel inside FlashInfer's ``fp8_blockscale_gemm_sm90`` is used; above,
    # external-quant + DeepGEMM ``fp8_gemm_nt``.  The threshold is hard-coded
    # to 32 in vLLM (``fp8_utils.py:308``).
    _FLASHINFER_M_THRESHOLD = 32

    def forward(self, input_bf16: torch.Tensor,
                weight_fp8: torch.Tensor,
                weight_scale_inv: torch.Tensor,
                bias: torch.Tensor | None = None) -> torch.Tensor:
        """FP8 block-scaled GEMM with vLLM-parity dispatch.

        * ``M < 32`` and FlashInfer available → FlashInfer swapAB kernel
          (BF16 in, internal quant, FP8 GEMM, BF16 out).  Same path as
          ``vllm/.../fp8_utils.py:_run_flashinfer``.
        * Otherwise → external ``per_token_group_quant_fp8`` (column-major
          UE8M0 scales) + ``deep_gemm.fp8_gemm_nt`` (with
          ``disable_ue8m0_cast`` set per the resolved oracle).  Same path
          as ``vllm/.../fp8_utils.py:_run_deepgemm``.

        FP8 ops are routed through ``torch.ops.fastkernels_fp8.*`` so they stay
        opaque to ``torch.compile`` (mirrors vLLM's
        ``torch.ops.vllm.fp8_gemm_nt_op`` / ``flashinfer_fp8_blockscale_gemm``).
        """
        N, K = weight_fp8.shape
        input_2d = input_bf16.reshape(-1, K)
        M = input_2d.shape[0]
        num_groups = (K + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE

        # M-independent FlashInfer eligibility (mirrors vLLM
        # ``should_use_flashinfer_for_blockscale_fp8_gemm``: N % 64 == 0,
        # K % 128 == 0, plus the SM90+FlashInfer availability gate). Batch-
        # invariant mode forces DeepGEMM for all M (vLLM's dynamic-dispatch
        # early-out), so exclude it here.
        flashinfer_ok = (
            input_bf16.dtype == torch.bfloat16
            and weight_fp8.dtype == torch.float8_e4m3fn
            and N % 64 == 0
            and K % 128 == 0
            and not _is_batch_invariant()
            and _maybe_get_flashinfer_fp8_gemm() is not None
        )

        # Under torch.compile / CUDA-graph capture, the M<32 (FlashInfer) vs
        # M>=32 (DeepGEMM) choice must be made at RUNTIME, not frozen at trace
        # time. A Python ``if M < 32`` (or ``torch.compiler.is_compiling()``)
        # gate would bake the branch into the graph and drop FlashInfer's
        # low-batch path. Route through the opaque ``blockscale_gemm_dispatch``
        # custom op (like vLLM's ``dynamic_flashinfer_deepgemm_blockscale_gemm``)
        # which branches on the runtime M internally. The eager path below keeps
        # its buffer-reuse fast path.
        if torch.compiler.is_compiling():
            output = torch.ops.fastkernels_fp8.blockscale_gemm_dispatch(
                input_2d, weight_fp8, weight_scale_inv, flashinfer_ok,
            )
            if bias is not None:
                output = output + bias
            return output.view(*input_bf16.shape[:-1], N)

        use_flashinfer = flashinfer_ok and M < self._FLASHINFER_M_THRESHOLD

        if use_flashinfer:
            output = torch.empty(
                M, N, dtype=torch.bfloat16, device=input_2d.device,
            )
            torch.ops.fastkernels_fp8.flashinfer_blockscale_gemm(
                input_2d, weight_fp8, weight_scale_inv, output,
            )
            if bias is not None:
                output = output + bias
            return output.view(*input_bf16.shape[:-1], N)

        # Eager only (the compile path returned above), so buffer reuse is always
        # safe here — no ``torch.compiler.is_compiling()`` guard needed.
        if self._a_buf is not None and M <= self._a_buf.shape[0]:
            q_input = self._a_buf[:M]
            input_scale = _alloc_colmajor_scale(M, num_groups, input_2d.device)
            output = self._o_buf[:M]
        elif self._pf is not None and M <= self._pf.a.shape[0]:
            q_input = self._pf.a[:M]
            input_scale = _alloc_colmajor_scale(M, num_groups, input_2d.device)
            output = self._pf.o[:M]
        else:
            q_input = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=input_2d.device)
            input_scale = _alloc_colmajor_scale(M, num_groups, input_2d.device)
            output = torch.empty(M, N, dtype=torch.bfloat16, device=input_2d.device)

        # ``column_major_scales=True`` matches the SF layout DeepGEMM expects
        # (``vllm/.../fp8_utils.py:289-294`` uses the same setting). The
        # ``input_scale`` buffer was allocated with column-major strides above.
        torch.ops.fastkernels_fp8.per_token_group_quant_fp8(
            input_2d, q_input, input_scale, True,
        )
        torch.ops.fastkernels_fp8.fp8_gemm_nt(
            q_input, input_scale, weight_fp8, weight_scale_inv, output,
        )

        if bias is not None:
            output = output + bias

        return output.view(*input_bf16.shape[:-1], N)

def _get_fp8_linear_cls():
    return Fp8Linear

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

class RowParallelLinear(nn.Module):
    """Splits input dim across TP ranks, all-reduces output."""

    def __init__(self, input_size: int, output_size: int, bias: bool = False,
                 quant_config: dict | None = None, reduce_results: bool = True):
        super().__init__()
        tp = _tp_size()
        assert input_size % tp == 0
        self.input_size_per_partition = input_size // tp
        self.tp_size = tp
        self.tp_rank = _tp_rank()
        self.reduce_results = reduce_results
        self.use_fp8 = quant_config is not None

        if self.use_fp8:
            self.weight = nn.Parameter(
                torch.empty(output_size, self.input_size_per_partition,
                            dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(*_scale_shape(output_size, self.input_size_per_partition),
                            dtype=torch.float32),
                requires_grad=False,
            )
            self.weight.weight_loader = self._weight_loader
            self.weight_scale_inv.weight_loader = self._scale_loader
            self.linear_op = _get_fp8_linear_cls()()
        else:
            self.weight = nn.Parameter(torch.empty(output_size, self.input_size_per_partition))
            self.weight.weight_loader = self._weight_loader

        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None
        if self.bias is not None:
            self.bias.weight_loader = lambda p, w: p.data.copy_(w)
        self.allreduce = AllReduce()

    def _weight_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        shard = param.data.size(1)
        loaded_weight = loaded_weight.narrow(1, rank * shard, shard)
        param.data.copy_(loaded_weight)

    def _scale_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        cols_per_shard = param.data.size(1)
        loaded_weight = loaded_weight.narrow(1, rank * cols_per_shard, cols_per_shard)
        param.data.copy_(loaded_weight)

    def forward(self, x):
        if self.use_fp8:
            y = self.linear_op(x, self.weight, self.weight_scale_inv,
                               self.bias if self.tp_rank == 0 else None)
        else:
            y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.reduce_results and self.tp_size > 1:
            y = self.allreduce(y)
        return y

class ReplicatedLinear(nn.Module):
    """Full weight replicated on every TP rank (no sharding, no all-reduce)."""

    def __init__(self, input_size: int, output_size: int, bias: bool = True,
                 quant_config: dict | None = None):
        super().__init__()
        self.use_fp8 = quant_config is not None

        if self.use_fp8:
            self.weight = nn.Parameter(
                torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(*_scale_shape(output_size, input_size),
                            dtype=torch.float32),
                requires_grad=False,
            )
            self.weight.weight_loader = lambda p, w: p.data.copy_(w)
            self.weight_scale_inv.weight_loader = lambda p, w: p.data.copy_(w)
            self.linear_op = _get_fp8_linear_cls()()
        else:
            self.weight = nn.Parameter(torch.empty(output_size, input_size))
            self.weight.weight_loader = lambda p, w: p.data.copy_(w)

        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None
        if self.bias is not None:
            self.bias.weight_loader = lambda p, w: p.data.copy_(w)

    def forward(self, x):
        if self.use_fp8:
            return self.linear_op(x, self.weight, self.weight_scale_inv, self.bias)
        return F.linear(x, self.weight, self.bias)

class ColumnParallelLinear(nn.Module):
    """Splits output dim across TP ranks."""

    def __init__(self, input_size: int, output_size: int, bias: bool = False,
                 quant_config: dict | None = None):
        super().__init__()
        tp = _tp_size()
        assert output_size % tp == 0
        self.output_size_per_partition = output_size // tp
        self.use_fp8 = quant_config is not None

        if self.use_fp8:
            self.weight = nn.Parameter(
                torch.empty(self.output_size_per_partition, input_size,
                            dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(*_scale_shape(self.output_size_per_partition, input_size),
                            dtype=torch.float32),
                requires_grad=False,
            )
            self.weight.weight_loader = self._weight_loader
            self.weight_scale_inv.weight_loader = self._scale_loader
            self.linear_op = _get_fp8_linear_cls()()
        else:
            self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, input_size))
            self.weight.weight_loader = self._weight_loader

        self.bias = nn.Parameter(torch.empty(self.output_size_per_partition)) if bias else None
        if self.bias is not None:
            self.bias.weight_loader = self._weight_loader

    def _weight_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        shard = param.data.size(0)
        loaded_weight = loaded_weight.narrow(0, rank * shard, shard)
        param.data.copy_(loaded_weight)

    def _scale_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        rows_per_shard = param.data.size(0)
        loaded_weight = loaded_weight.narrow(0, rank * rows_per_shard, rows_per_shard)
        param.data.copy_(loaded_weight)

    def forward(self, x):
        if self.use_fp8:
            return self.linear_op(x, self.weight, self.weight_scale_inv, self.bias)
        return F.linear(x, self.weight, self.bias)

float8_info = torch.finfo(torch.float8_e4m3fn)

def merge_attn_states_kernel(
    output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    output_lse,  # [NUM_HEADS, NUM_TOKENS]
    prefix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    prefix_lse,  # [NUM_HEADS, NUM_TOKENS]
    suffix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    suffix_lse,  # [NUM_HEADS, NUM_TOKENS]
    prefix_head_stride,
    output_head_stride,
    output_scale,  # scale tensor or None
    HEAD_SIZE: tl.constexpr,
    PADDED_HEAD_SIZE: tl.constexpr,
    OUTPUT_LSE: tl.constexpr,
    prefill_tokens_with_context: tl.constexpr,
    USE_FP8: tl.constexpr,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    token_idx = tl.program_id(0)
    num_tokens = tl.num_programs(0)
    head_idx = tl.program_id(1)
    num_heads = tl.num_programs(1)

    prefix_mask = token_idx < prefill_tokens_with_context

    head_arange = tl.arange(0, PADDED_HEAD_SIZE)
    head_mask = head_arange < HEAD_SIZE

    # For tokens without context (token_idx >= prefill_tokens_with_context),
    # directly copy from suffix_output
    if not prefix_mask:
        s_lse = tl.load(suffix_lse + head_idx * num_tokens + token_idx)
        if OUTPUT_LSE:
            tl.store(output_lse + head_idx * num_tokens + token_idx, s_lse)

        s_out = tl.load(
            suffix_output
            + token_idx * num_heads * prefix_head_stride
            + head_idx * prefix_head_stride
            + head_arange,
            mask=head_mask,
        )

        if USE_FP8:
            s_out = s_out * (1.0 / tl.load(output_scale))
            s_out = tl.clamp(s_out, FP8_MIN, FP8_MAX)
            s_out = s_out.to(output.dtype.element_ty)

        tl.store(
            output
            + token_idx * num_heads * output_head_stride
            + head_idx * output_head_stride
            + head_arange,
            s_out,
            mask=head_mask,
        )
        return

    # For tokens with context (token_idx < prefill_tokens_with_context),
    # perform normal merge operation
    p_lse = tl.load(prefix_lse + head_idx * num_tokens + token_idx)
    s_lse = tl.load(suffix_lse + head_idx * num_tokens + token_idx)

    # FA2 and FA3 have different behavior for when the sum-exp is 0, this namely
    # arises with 0 len seqlens. FA3 returns -inf here while FA2 returns inf.
    # If we see an inf assume FA2 and convert inf to -inf for consistency
    # and correctness. Inf generally doesn't make sense in this context outside
    # of undefined-behavior/FA2-case, so I think this a safe assumption.
    p_lse = float("-inf") if p_lse == float("inf") else p_lse
    s_lse = float("-inf") if s_lse == float("inf") else s_lse

    max_lse = tl.maximum(p_lse, s_lse)
    p_lse = p_lse - max_lse
    s_lse = s_lse - max_lse
    # Will reuse precomputed Exp values for scale factor computation.
    p_se = tl.exp(p_lse)
    s_se = tl.exp(s_lse)
    out_se = p_se + s_se

    if OUTPUT_LSE:
        out_lse = tl.log(out_se) + max_lse
        # Both sides empty (max_lse == -inf) => undefined merge; keep -inf so
        # downstream merges continue to treat the token as empty.
        out_lse = tl.where(max_lse == float("-inf"), float("-inf"), out_lse)
        tl.store(output_lse + head_idx * num_tokens + token_idx, out_lse)

    p_out = tl.load(
        prefix_output
        + token_idx * num_heads * prefix_head_stride
        + head_idx * prefix_head_stride
        + head_arange,
        mask=head_mask,
    )
    s_out = tl.load(
        suffix_output
        + token_idx * num_heads * prefix_head_stride
        + head_idx * prefix_head_stride
        + head_arange,
        mask=head_mask,
    )

    # NOTE(woosuk): Be careful with the numerical stability.
    # We should compute the scale first, and then multiply it with the output.
    # Do not multiply the output with tl.exp(p_lse) or tl.exp(s_lse) directly.
    p_scale = p_se / out_se
    s_scale = s_se / out_se
    out = p_out * p_scale + s_out * s_scale
    # If both sides are empty (max_lse == -inf) the scales are 0/0 = NaN; emit
    # zeros rather than NaN. Callers with empty chunks (see mask_empty_context)
    # zero those inputs, so this only guards the fully-undefined corner.
    out = tl.where(max_lse == float("-inf"), 0.0, out)

    if USE_FP8:
        out = out * (1.0 / tl.load(output_scale))
        out = tl.clamp(out, FP8_MIN, FP8_MAX)
        out = out.to(output.dtype.element_ty)

    tl.store(
        output
        + token_idx * num_heads * output_head_stride
        + head_idx * output_head_stride
        + head_arange,
        out,
        mask=head_mask,
    )

def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: torch.Tensor | None = None,
    prefill_tokens_with_context: int | None = None,
    output_scale: torch.Tensor | None = None,
) -> None:
    num_tokens = output.shape[0]
    num_query_heads = output.shape[1]
    head_size = output.shape[2]
    padded_head_size = triton.next_power_of_2(head_size)
    # We assume the output stride on num_head is not always as same as the
    # `suffix_output` and `prefix_output`, as them might be padded by the
    # attention backend.
    prefix_head_stride = prefix_output.stride(1)
    output_head_stride = output.stride(1)

    # If prefill_tokens_with_context is None, all tokens should use prefix context
    if prefill_tokens_with_context is None:
        prefill_tokens_with_context = num_tokens

    # TODO(woosuk): Use CUDA kernel instead of Triton to minimize CPU overhead.
    merge_attn_states_kernel[(num_tokens, num_query_heads)](
        output,
        output_lse,
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        prefix_head_stride,
        output_head_stride,
        output_scale,
        head_size,
        padded_head_size,
        output_lse is not None,
        prefill_tokens_with_context,
        output_scale is not None,
    )

_C = lazy_op("merge_attn_states", "merge_attn_states.cu")

def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: torch.Tensor | None = None,
    prefill_tokens_with_context: int | None = None,
    output_scale: torch.Tensor | None = None,
) -> None:
    # Both the CUDA and Triton kernels derive the suffix head stride from
    # prefix_output, so suffix_output must share the same head stride.
    assert prefix_output.stride(1) == suffix_output.stride(1), (
        "merge_attn_states requires prefix_output and suffix_output to have "
        f"matching head strides, got {prefix_output.stride(1)} and "
        f"{suffix_output.stride(1)}"
    )

    # NOTE(DefTruth): Currently, custom merge_attn_states CUDA kernel
    # does not support FP8 dtype for inputs, fallback to use Triton kernel.
    # However, when output_scale is provided, the inputs are still BF16/FP16
    # and the output is FP8 — both CUDA and Triton support this.
    # FP8 output requires output_scale to be set.
    if output.dtype not in (torch.float32, torch.half, torch.bfloat16):
        assert output_scale is not None, (
            f"output_scale is required when output is {output.dtype}"
        )

    def supported_dtypes(prefix: torch.Tensor) -> bool:
        return prefix.dtype in [torch.float32, torch.half, torch.bfloat16]

    # NOTE(DefTruth): Currently, custom merge_attn_states CUDA
    # kernel load/store 128b(16 bytes) per memory issue within
    # thread. Namely, the headsize(headdim) must be multiple of
    # pack_size based on input dtype (float32 -> 4, half/bfloat16 -> 8).
    def supported_headdim(prefix: torch.Tensor) -> bool:
        headdim = prefix.shape[2]  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
        if prefix.dtype == torch.float32:
            return headdim % 4 == 0
        return headdim % 8 == 0

    if (
        prefix_output.is_cuda
        and supported_dtypes(prefix_output)
        and supported_headdim(prefix_output)
    ):
        _C.merge_attn_states(
            output,
            output_lse,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            prefill_tokens_with_context,
            output_scale,
        )
    else:

        _triton_merge(
            output,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            output_lse,
            prefill_tokens_with_context,
            output_scale,
        )

class MergeAttnStates(nn.Module):
    """Online softmax merge of two attention partitions."""

    def forward(
        self,
        output: torch.Tensor,
        prefix_output: torch.Tensor,
        prefix_lse: torch.Tensor,
        suffix_output: torch.Tensor,
        suffix_lse: torch.Tensor,
        output_lse: torch.Tensor | None = None,
    ) -> None:
        merge_attn_states(
            output=output,
            prefix_output=prefix_output,
            prefix_lse=prefix_lse,
            suffix_output=suffix_output,
            suffix_lse=suffix_lse,
            output_lse=output_lse,
        )

_MLA_HEAD_DIM_V = 512

def flashinfer_mla_decode_supported() -> bool:
    """True when the trtllm-gen MLA decode kernel can run on this device."""
    if not torch.cuda.is_available():
        return False
    # trtllm-gen MLA is a Blackwell (SM100) path.
    return torch.cuda.get_device_capability()[0] >= 10

class FlashMLADecode(nn.Module):
    """Wraps ``flash_mla_with_kvcache`` for paged MLA decode.

    Used for:
    - BF16 dense decode (``is_fp8_kvcache=False``)
    - Sparse FP8 decode with ``indices`` (DSA path)

    Dense FP8 decode goes through :class:`FlashMLADecodeFP8` instead, which
    matches vLLM's ``flash_mla_with_kvcache_fp8`` entry point with
    ``descale_q`` / ``descale_k`` and ``num_splits``.
    """

    def forward(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        head_dim_v: int,
        tile_scheduler_metadata: torch.Tensor,
        softmax_scale: float,
        causal: bool = True,
        is_fp8_kvcache: bool = False,
        indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return flash_mla_with_kvcache(
            q,
            kv_cache,
            block_table,
            cache_seqlens,
            head_dim_v=head_dim_v,
            tile_scheduler_metadata=tile_scheduler_metadata,
            softmax_scale=softmax_scale,
            causal=causal,
            is_fp8_kvcache=is_fp8_kvcache,
            indices=indices,
        )

def _compute_fp8_decode_padded_heads(num_heads: int) -> int:
    return 64 if num_heads <= 64 else 128

MIN_HEADS_FOR_BF16_PREFILL = 32

class FlashMLASparsePrefill(nn.Module):
    """Wraps flash_mla.flash_mla_sparse_fwd for sparse BF16 prefill.

    Used when prefill has sparse indices (DSA). The workspace must
    already contain BF16 KV data gathered from the FP8 cache.
    """

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        softmax_scale: float,
        d_v: int = 512,
    ) -> torch.Tensor:
        return flash_mla_sparse_fwd(q, kv, indices, softmax_scale, d_v=d_v)

class BatchMatMul(nn.Module):
    """Batched matrix multiply ``(B, N, M) @ (B, M, P) -> (B, N, P)``."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.bmm(a, b)

class FlashMLAGetMetadataDenseFP8(nn.Module):
    """Wraps ``get_mla_metadata_dense_fp8`` for the FP8 dense decode kernel.

    Returns ``(tile_scheduler_metadata, num_splits)`` which both must be fed
    into :class:`FlashMLADecodeFP8`. Matches the vLLM call site in
    ``vllm/v1/attention/backends/mla/flashmla.py:171-178``.
    """

    available: bool = get_mla_metadata_dense_fp8 is not None

    def forward(
        self,
        cache_seqlens: torch.Tensor,
        num_q_tokens_per_head_k: int,
        num_heads_k: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert get_mla_metadata_dense_fp8 is not None, (
            "get_mla_metadata_dense_fp8 not available — "
            "vLLM build must include _flashmla_extension_C"
        )
        return get_mla_metadata_dense_fp8(
            cache_seqlens, num_q_tokens_per_head_k, num_heads_k,
        )

_C = lazy_op("store_kvcache_fp8_mla", "store_kvcache_fp8_mla.cu")

class GatherAndDequantKVCacheMLA(nn.Module):
    """Gather MLA KV cache into a BF16 workspace using the
    ``gather_and_maybe_dequant_cache`` kernel (vLLM's chunked-context helper).

    Required arguments match the kernel's signature:
        ``kv_cache``: ``[num_blocks, block_size, 576]`` BF16 (``"auto"``) or
                      ``[num_blocks, block_size, 656]`` uint8 (``fp8_ds_mla``).
        ``workspace``: ``[total_tokens, 576]`` BF16 output buffer.
        ``block_table``: ``[num_seqs, max_blocks]`` int32.
        ``cu_seq_lens``: ``[num_seqs+1]`` int32 cumulative sequence lengths.
        ``token_to_seq``: ``[total_tokens]`` int32 mapping.
        ``total_tokens``: scalar int.
        ``workspace_starts``: ``[num_seqs]`` int32 — starting workspace row
                             per sequence (for chunked context gathers).

    ``kv_cache_dtype`` selects the source layout and must match the cache the
    owning ``MLAAttention`` allocated; vLLM likewise forwards its own
    ``self.kv_cache_dtype`` here, and passing ``"fp8_ds_mla"`` for a BF16
    cache reinterprets the bytes and silently corrupts the gathered context.
    """

    def __init__(self, kv_cache_dtype: str = "fp8_ds_mla"):
        super().__init__()
        assert kv_cache_dtype in ("auto", "fp8_ds_mla", "fp8_e4m3"), (
            f"GatherAndDequantKVCacheMLA: unsupported "
            f"kv_cache_dtype={kv_cache_dtype!r}"
        )
        self.kv_cache_dtype = kv_cache_dtype
        # ONE, not zero: on the ``fp8_e4m3`` path the kernel MULTIPLIES the
        # gathered fp8 values by this scale to dequantize (vLLM passes
        # ``layer._k_scale``, default 1.0). Zero would blank the gathered
        # context. Ignored for ``auto`` and for the block-scaled ``fp8_ds_mla``.
        self.register_buffer(
            "_k_scale", torch.ones(1, dtype=torch.float32), persistent=False,
        )

    def forward(
        self,
        kv_cache: torch.Tensor,
        workspace: torch.Tensor,
        block_table: torch.Tensor,
        cu_seq_lens: torch.Tensor,
        token_to_seq: torch.Tensor,
        total_tokens: int,
        workspace_starts: torch.Tensor,
    ) -> None:
        _C.gather_and_maybe_dequant_cache(
            kv_cache, workspace,
            block_table, cu_seq_lens, token_to_seq,
            total_tokens,
            self.kv_cache_dtype,
            self._k_scale,
            workspace_starts,
        )

def _compute_prefill_padding() -> int:
    """Mirror vLLM's BF16 sparse prefill padding selection.

    See ``vllm/v1/attention/backends/mla/flashmla_sparse.py:565-568``:
    Hopper (SM90) requires 64-element head padding while Blackwell (SM100)
    requires 128. Older arches default to 64.
    """
    try:
        major, _ = torch.cuda.get_device_capability()
    except Exception:
        return 64
    return 128 if major == 10 else 64

_WORKSPACE_BYTES = 512 * 1024 * 1024

_sparse_workspace: torch.Tensor | None = None

def _sparse_ws(device: torch.device) -> torch.Tensor:
    """Shared int8 workspace for the sparse MLA decode kernel.

    int8 (not uint8) because FlashInfer's CuteDSL MLA-decode tactic requires a
    signed workspace while the trtllm-gen path views it as uint8 — the same
    reasoning as vLLM's ``_get_workspace_buffer``.
    """
    global _sparse_workspace
    if _sparse_workspace is None or _sparse_workspace.device != device:
        _sparse_workspace = torch.zeros(
            _WORKSPACE_BYTES, dtype=torch.int8, device=device,
        )
    return _sparse_workspace

_ragged_workspace: torch.Tensor | None = None

def _ragged_ws(device: torch.device) -> torch.Tensor:
    """Shared uint8 workspace for the ragged prefill kernel.

    vLLM allocates this one as uint8 through its workspace manager
    (``TrtllmRaggedPrefillBackend.__init__``), and keeps it separate from the
    decode workspace, so the two kernels never alias.
    """
    global _ragged_workspace
    if _ragged_workspace is None or _ragged_workspace.device != device:
        _ragged_workspace = torch.zeros(
            _WORKSPACE_BYTES, dtype=torch.uint8, device=device,
        )
    return _ragged_workspace

def ensure_workspaces(device: torch.device) -> None:
    """Materialize both shared workspaces on ``device``.

    MUST be called before CUDA graph capture. A workspace allocated lazily on
    first use is created *inside* the capturing region, so it comes from that
    one graph's private memory pool -- and every later graph (one per captured
    batch size) then records kernels that atomically reduce into another graph's
    pool. Replay fails with ``cudaErrorInvalidAddressSpace`` ("operation not
    permitted on the memory's address space", i.e. an atomic on memory the
    kernel may not atomically touch). vLLM sidesteps this by taking its
    workspaces from a module-level buffer / its workspace manager at model init.
    """
    _sparse_ws(device)
    _ragged_ws(device)

class FlashInferMLASparseDecode(nn.Module):
    """Sparse top-k MQA over an fp8 paged MLA cache (trtllm-gen).

    Mirrors ``FlashInferMLASparseImpl.forward_mqa``
    (vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py:388-470): the
    per-token top-k *global slot* indices are handed to the kernel as a
    ``block_tables`` of page size 1, so ``seq_lens`` is the per-token count of
    valid slots and ``max_seq_len`` is ``index_topk``.

    Every query token carries its own top-k row, so the ``q_len_per_request``
    dim is a bare ``unsqueeze(1)`` — vLLM does the same and notes the
    multi-token grouping is a perf-only layout deferred until MTP is validated.
    """

    def __init__(self, qk_nope_head_dim: int, qk_rope_head_dim: int,
                 kv_lora_rank: int):
        super().__init__()
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank

    def ensure_workspaces(self, device: torch.device) -> None:
        """See :func:`ensure_workspaces` -- pre-capture workspace allocation."""
        ensure_workspaces(device)

    def forward(
        self,
        q: torch.Tensor,              # [N, H, 576] fp8_e4m3 (ql_nope || q_pe)
        kv_cache: torch.Tensor,       # [num_blocks, block_size, 576] fp8_e4m3
        topk_slots: torch.Tensor,     # [N, topk] int32 global slots (-1 = pad)
        seq_lens: torch.Tensor,       # [N] int32 valid slots per token
        topk_tokens: int,
        bmm1_scale: float,
        bmm2_scale: float,
    ) -> torch.Tensor:
        """Returns ``[N, H, kv_lora_rank]`` bf16 (pre ``v_up_proj``)."""
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

        out = trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv_cache.unsqueeze(1),
            workspace_buffer=_sparse_ws(q.device),
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=topk_slots.unsqueeze(1),
            seq_lens=seq_lens,
            max_seq_len=topk_tokens,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            sparse_mla_top_k=topk_tokens,
            return_lse=False,
        )
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.view(-1, out.shape[-2], out.shape[-1])

def _default_kv_cache_dtype() -> str:
    """Resolve the MLA KV cache dtype from ``FASTKERNELS_KV_CACHE_DTYPE``.

    Defaults to ``"auto"`` (BF16), matching stock vLLM on DeepSeek-V3.2.
    ``fp8``/``fp8_ds_mla`` select DeepSeek's 656-byte block-scaled layout;
    ``fp8_e4m3`` selects the plain per-tensor 576-byte layout that vLLM's
    FLASHINFER_MLA_SPARSE backend uses. The two are NOT interchangeable, so
    ``fp8`` is left aliased to ``fp8_ds_mla`` for backwards compatibility and
    ``fp8_e4m3`` must be spelled out.
    """
    v = os.environ.get("FASTKERNELS_KV_CACHE_DTYPE", "auto").strip().lower()
    if v in ("", "auto", "bf16", "bfloat16"):
        return "auto"
    if v in ("fp8", "fp8_ds_mla"):
        return "fp8_ds_mla"
    if v in ("fp8_e4m3", "fp8_e4m3fn"):
        return "fp8_e4m3"
    raise ValueError(f"Unsupported FASTKERNELS_KV_CACHE_DTYPE={v!r}")

def _convert_req_index_to_global_index_kernel(
    req_id_ptr,
    block_table_ptr,
    token_indices_ptr,
    out_ptr,
    valid_count_ptr,
    prefill_request_id_ptr,
    workspace_starts_ptr,
    max_num_blocks_per_req: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_PREFILL: tl.constexpr,
    COUNT_VALID: tl.constexpr,
    bt_stride0,
    bt_stride1,
    ti_stride0,
    ti_stride1,
    out_stride0,
    out_stride1,
):
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    req = tl.load(req_id_ptr + token_id)
    ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
    tok = tl.load(ti_ptr)

    is_invalid_tok = tok < 0
    is_prefill = False
    if HAS_PREFILL:
        prefill_req_id = tl.load(prefill_request_id_ptr + token_id)
        is_prefill = prefill_req_id >= 0

    block_id = tok // BLOCK_SIZE
    inblock_off = tok % BLOCK_SIZE

    valid_block = (block_id < max_num_blocks_per_req) & (block_id >= 0)
    bt_ptr = block_table_ptr + req * bt_stride0 + block_id * bt_stride1
    is_invalid_tok |= ~valid_block
    base = tl.load(bt_ptr, mask=valid_block & ~is_prefill, other=0)
    out_val = base * BLOCK_SIZE + inblock_off

    if HAS_PREFILL:
        workspace_start = tl.load(
            workspace_starts_ptr + prefill_req_id, mask=is_prefill, other=0
        )
        prefill_out = workspace_start + tok
        out_val = tl.where(is_prefill, prefill_out, out_val)
    out_val = tl.where(is_invalid_tok, -1, out_val)

    out_ptr_ij = out_ptr + token_id * out_stride0 + indice_id * out_stride1
    tl.store(out_ptr_ij, out_val)

    # Per-row count of valid (non ``-1``) slots, accumulated across column tiles
    # with one atomic per tile. Mirrors vLLM's ``COUNT_VALID`` branch
    # (v1/attention/backends/mla/sparse_utils.py:114-117); the FlashInfer sparse
    # MLA decode kernel takes these counts as its ``seq_lens``, so they must be
    # computed in the same pass rather than from a separate reduction over
    # ``out`` (which would be a second kernel on the attention hot path).
    if COUNT_VALID:
        tile_valid_count = tl.sum((~is_invalid_tok).to(tl.int32))
        tl.atomic_add(valid_count_ptr + token_id, tile_valid_count)

class ConvertIndicesToGlobal(nn.Module):
    """Map per-request token indices to global linear cache slots.

    Supports both simple decode-only mode (block_table lookup) and
    mixed prefill+decode mode with workspace offset mapping.
    """

    def forward(
        self,
        indices: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        req_ids: torch.Tensor | None = None,
        prefill_request_ids: torch.Tensor | None = None,
        prefill_workspace_starts: torch.Tensor | None = None,
        return_valid_counts: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Convert local indices to global slot indices.

        Args:
            indices: ``[num_tokens, topk]`` int32.
            block_table: ``[num_reqs, max_blocks]`` int32.
            block_size: tokens per block.
            req_ids: ``[num_tokens]`` int32 — which request each token belongs to.
                If None, assumes identity mapping (token i -> request i).
            prefill_request_ids: ``[num_tokens]`` int32 — -1 for decode,
                >=0 for prefill (index into prefill_workspace_starts).
            prefill_workspace_starts: ``[num_prefills]`` int32 — workspace
                start offset per prefill request.
            return_valid_counts: also return ``[num_tokens]`` int32 counts of
                valid (non ``-1``) slots per row, computed in the same kernel
                pass. Needed by the FlashInfer sparse MLA decode kernel.

        Returns:
            ``global_indices``: ``[num_tokens, topk]`` int32, or
            ``(global_indices, valid_counts)`` when ``return_valid_counts``.
        """
        num_tokens, topk = indices.shape
        has_prefill = prefill_request_ids is not None and prefill_workspace_starts is not None

        if req_ids is None:
            req_ids = torch.arange(num_tokens, dtype=torch.int32, device=indices.device)

        BLOCK_N = min(128, topk)
        assert topk % BLOCK_N == 0

        max_num_blocks_per_req = block_table.shape[1]
        tiles_per_row = topk // BLOCK_N

        # Materialize contiguous copies BEFORE reading strides. The kernel
        # indexes with the strides passed here, so they must belong to the
        # exact tensors the kernel receives. ``block_table`` in eager decode is
        # a non-contiguous slice (``_eager_block_tables[:n, :bt_cols]`` whose
        # row stride is the full buffer width, not ``bt_cols``); taking strides
        # from the slice while passing a fresh ``.contiguous()`` copy made the
        # kernel read request rows at the wrong offset -> garbage global slots
        # for every request but the first (req 0 starts at 0 either way, so
        # single-sequence / offset-0 batches masked the bug).
        req_ids = req_ids.contiguous()
        block_table = block_table.contiguous()
        indices = indices.contiguous()

        out = torch.empty_like(indices)
        # Must be zero-initialized: the kernel accumulates into it atomically.
        valid_counts = (
            torch.zeros(num_tokens, dtype=torch.int32, device=indices.device)
            if return_valid_counts else None
        )

        bt_stride0, bt_stride1 = block_table.stride()
        ti_stride0, ti_stride1 = indices.stride()
        out_stride0, out_stride1 = out.stride()

        grid = (num_tokens, tiles_per_row)
        _convert_req_index_to_global_index_kernel[grid](
            req_ids,
            block_table,
            indices,
            out,
            valid_counts,
            prefill_request_ids if has_prefill else prefill_request_ids,
            prefill_workspace_starts if has_prefill else prefill_workspace_starts,
            max_num_blocks_per_req,
            block_size,
            BLOCK_N,
            has_prefill,
            return_valid_counts,
            bt_stride0, bt_stride1,
            ti_stride0, ti_stride1,
            out_stride0, out_stride1,
        )
        if return_valid_counts:
            return out, valid_counts
        return out

def mask_empty_context_kernel(
    lse,
    is_empty,
    query_start_loc,
    context_start_loc,
    lse_head_stride,
    lse_token_stride,
    num_reqs,
    NUM_HEADS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
):
    query_block_idx = tl.program_id(0)

    lanes = tl.arange(0, 32)
    chunk_start = 0
    req_idx = 0
    req_idx_found = False
    while (chunk_start < num_reqs) & (not req_idx_found):
        req_offsets = chunk_start + lanes
        req_mask = req_offsets < num_reqs
        query_starts = tl.load(query_start_loc + req_offsets, mask=req_mask)
        # Assume the worst-case number of blocks for each request.
        req_block_starts = query_starts // BLOCK_SIZE + req_offsets
        matched_idx = tl.sum(
            (req_mask & (req_block_starts <= query_block_idx)).to(tl.int32)
        )
        # matched_idx == 32 means the match is past this warp chunk.
        req_idx = chunk_start + matched_idx - 1
        req_idx_found = matched_idx < 32
        chunk_start += 32

    query_start = tl.load(query_start_loc + req_idx)
    query_end = tl.load(query_start_loc + req_idx + 1)
    query_len = query_end - query_start
    req_first_block = query_start // BLOCK_SIZE + req_idx
    block_in_req = query_block_idx - req_first_block
    token_offset = block_in_req * BLOCK_SIZE
    if token_offset >= query_len:
        return

    context_start = tl.load(context_start_loc + req_idx)
    context_end = tl.load(context_start_loc + req_idx + 1)
    if context_start != context_end:
        return

    token_offsets = token_offset + tl.arange(0, BLOCK_SIZE)
    token_indices = query_start + token_offsets
    token_lse_offsets = token_indices * lse_token_stride
    valid_tokens = token_offsets < query_len
    tl.store(is_empty + token_indices, True, mask=valid_tokens)
    head_offsets = tl.arange(0, BLOCK_HEADS)
    for head_start in range(0, NUM_HEADS, BLOCK_HEADS):
        head_indices = head_start + head_offsets
        lse_ptrs = (
            lse + head_indices[:, None] * lse_head_stride + token_lse_offsets[None, :]
        )
        valid_heads = head_indices < NUM_HEADS
        tl.store(
            lse_ptrs,
            float("-inf"),
            mask=valid_heads[:, None] & valid_tokens[None, :],
        )

def mask_empty_context(
    lse: torch.Tensor,
    output: torch.Tensor,
    query_start_loc: torch.Tensor,
    context_start_loc: torch.Tensor,
) -> None:
    """Neutralize context chunks that cover no keys before merging.

    A prefill query whose context chunk is empty attended to no keys, so its
    partial attention is undefined: the backend leaves the output rows as
    uninitialized scratch (which may hold NaN/Inf) even when it reports an LSE
    of -inf. Sanitize both here so ``merge_attn_states`` can stay generic:
    force the LSE to -inf (zero softmax weight) and zero the undefined output
    rows (so a zero weight cannot combine with NaN/Inf). Emptiness is derived
    from the context offsets, not from the -inf LSE, so no merge kernel has to
    reason about undefined partials.

    Args:
        lse: Chunk log-sum-exp, shape [num_heads, num_tokens].
        output: Chunk attention output, shape [num_tokens, num_heads, ...].
        query_start_loc: Prefill query cumulative offsets, shape [num_reqs + 1].
        context_start_loc: Chunk context cumulative offsets,
            shape [num_reqs + 1]; an empty chunk has a zero-length span.
    """
    num_heads, num_tokens = lse.shape
    num_reqs = query_start_loc.shape[0] - 1
    block_size = 128
    # Reserve the worst-case number of request-local blocks.
    num_query_blocks = num_tokens // block_size + num_reqs
    is_empty = torch.zeros(num_tokens, dtype=torch.bool, device=lse.device)
    mask_empty_context_kernel[(num_query_blocks,)](
        lse,
        is_empty,
        query_start_loc,
        context_start_loc,
        lse.stride(0),
        lse.stride(1),
        num_reqs,
        NUM_HEADS=num_heads,
        BLOCK_SIZE=block_size,
        BLOCK_HEADS=8,
        num_warps=8,
    )
    output.masked_fill_(is_empty[:, None, None], 0.0)

class StoreKVCacheFP8MLA(nn.Module):
    """Store ``kv_c_normed`` and ``k_pe`` into MLA paged cache.

    Wraps vendored ``_C.concat_and_cache_mla``. Dispatches on
    ``kv_cache_dtype``:

    * ``"auto"``: expects a BF16 cache of shape
      ``[num_blocks, block_size, 576]``; the kernel writes the
      concatenation of ``kv_c_normed`` (512) and ``k_pe`` (64) directly.
    * ``"fp8_ds_mla"``: expects a uint8 cache of shape
      ``[num_blocks, block_size, 656]``; the kernel fuses per-block
      UE8M0 FP8 quantization of ``kv_c_normed`` with BF16 ``k_pe`` storage.
    * ``"fp8_e4m3"``: expects a ``float8_e4m3fn`` cache of shape
      ``[num_blocks, block_size, 576]``; the kernel divides both halves by
      ``k_scale`` and casts to fp8. This is the layout vLLM's
      FLASHINFER_MLA_SPARSE backend uses -- plain per-tensor fp8, no block
      scales -- and it is NOT interchangeable with ``fp8_ds_mla``.

    Args:
        kv_c_normed: ``[N, 512]`` BF16 — compressed KV after layernorm.
        k_pe: ``[N, 1, 64]`` or ``[N, 64]`` BF16 — RoPE key component.
        kv_cache: ``[num_blocks, block_size, 576|656]`` (BF16 / fp8 / uint8).
        slot_mapping: ``[N]`` int64 — linear slot index per token (``-1`` skips).
    """

    def __init__(self, kv_cache_dtype: str = "auto"):
        super().__init__()
        assert kv_cache_dtype in ("auto", "fp8_ds_mla", "fp8_e4m3"), (
            f"StoreKVCacheFP8MLA: unsupported kv_cache_dtype={kv_cache_dtype!r}"
        )
        self.kv_cache_dtype = kv_cache_dtype
        # ``k_scale`` is the per-tensor dequant scale the kernel divides by on
        # the ``fp8_e4m3`` path (it is ignored for ``auto`` and for the
        # block-scaled ``fp8_ds_mla`` layout). vLLM initialises ``layer._k_scale``
        # to 1.0 and only overwrites it from a checkpoint's calibration scales,
        # which nvidia/GLM-5.2-NVFP4 does not ship -- so ONE, not zero. A zero
        # here silently turned every stored KV element into inf/nan.
        self.register_buffer(
            "_k_scale", torch.ones(1, dtype=torch.float32), persistent=False,
        )

    def forward(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        k_pe_2d = k_pe.reshape(k_pe.shape[0], -1)
        _C.concat_and_cache_mla(
            kv_c_normed, k_pe_2d, kv_cache, slot_mapping,
            self.kv_cache_dtype, self._k_scale,
        )

class TrtllmRaggedPrefill(nn.Module):
    """Dense varlen MHA prefill for MLA (trtllm-gen ragged, DeepSeek dims).

    Mirrors ``TrtllmRaggedPrefillBackend``: ``bmm1_scale=self.scale``,
    ``bmm2_scale=1.0``, ``o_sf_scale=1.0``, ``window_left=-1``,
    ``enable_pdl=False``, and the LSE transposed from ``(q_len, num_heads)`` to
    ``(num_heads, q_len)`` for the merge step. This is the backend vLLM's
    ``get_mla_prefill_backend`` selects on sm100 for GLM-5.2's
    ``(qk_nope=192, rope=64, v=256)`` dims — NOT FlashAttention, whose different
    accumulation order would show up in every prefill token.
    """

    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def ensure_workspaces(self, device: torch.device) -> None:
        """See :func:`ensure_workspaces` -- pre-capture workspace allocation."""
        ensure_workspaces(device)

    def forward(
        self,
        q: torch.Tensor,               # [T, H, 256] bf16
        k: torch.Tensor,               # [T, H, 256] bf16
        v: torch.Tensor,               # [T, H, 256] bf16
        seq_lens: torch.Tensor,        # [B] int32 per-request KV length
        cu_seq_lens_q: torch.Tensor,   # [B+1] int32
        cu_seq_lens_kv: torch.Tensor,  # [B+1] int32
        max_q_len: int,
        max_kv_len: int,
        is_causal: bool,
        return_lse: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from flashinfer.prefill import trtllm_ragged_attention_deepseek

        out = torch.empty(
            q.shape[0], q.shape[1], v.shape[2],
            device=q.device, dtype=q.dtype,
        )
        ret = trtllm_ragged_attention_deepseek(
            query=q,
            key=k,
            value=v,
            workspace_buffer=_ragged_ws(q.device),
            seq_lens=seq_lens,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            o_sf_scale=1.0,
            batch_size=seq_lens.shape[0],
            window_left=-1,
            cum_seq_lens_q=cu_seq_lens_q,
            cum_seq_lens_kv=cu_seq_lens_kv,
            enable_pdl=False,
            is_causal=is_causal,
            return_lse=return_lse,
            out=out,
        )
        if isinstance(ret, tuple):
            # (q_len, num_heads) -> (num_heads, q_len), matching vLLM so the
            # LSE feeds ``merge_attn_states`` in the layout it expects.
            return ret[0], ret[1].transpose(0, 1).contiguous()
        return ret

class FlashMLAGetMetadata(nn.Module):
    def forward(
        self,
        cache_seqlens: torch.Tensor,
        num_q_tokens_per_head_k: int,
        num_heads_k: int = 1,
        topk: int | None = None,
        num_heads_q: int | None = None,
        is_fp8_kvcache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kwargs: dict = {}
        if topk is not None:
            kwargs["topk"] = topk
        if num_heads_q is not None:
            kwargs["num_heads_q"] = num_heads_q
            kwargs["num_heads_k"] = num_heads_k
        if is_fp8_kvcache:
            kwargs["is_fp8_kvcache"] = is_fp8_kvcache
        if kwargs:
            return get_mla_metadata(
                cache_seqlens=cache_seqlens,
                num_q_tokens_per_head_k=num_q_tokens_per_head_k,
                **kwargs,
            )
        return get_mla_metadata(cache_seqlens, num_q_tokens_per_head_k, num_heads_k)

def flashinfer_mla_sparse_available() -> bool:
    """True iff vLLM's FLASHINFER_MLA_SPARSE backend would be usable here.

    Matches ``FlashInferMLASparseTRTLLMBackend.supports_compute_capability``
    (sm100 exactly) plus the presence of both flashinfer entry points. The
    ``index_topk`` / ``qk_nope_head_dim`` parts of ``supports_combination`` are
    model properties and are checked by the caller.
    """
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 10

_UNSET = object()

class GatherKVCacheFP8MLA(nn.Module):
    """Gather and upconvert KV from FP8 MLA paged cache to BF16.

    Wraps vendored ``_C.cp_gather_and_upconvert_fp8_kv_cache``
    which gathers FP8-quantized kv_c_normed and BF16 k_pe from paged cache,
    dequantizes the FP8 portion, and writes the result as a contiguous
    BF16 workspace tensor.

    Returns:
        ``workspace``: ``[total_tokens, 576]`` BF16 — dequantized kv_c_normed
        (512 dims) concatenated with k_pe (64 dims).
    """

    def forward(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        workspace_starts: torch.Tensor,
        num_seqs: int,
        workspace: torch.Tensor,
    ) -> None:
        # ``seq_lens`` is unused by the kernel; lengths are implied by
        # ``workspace_starts`` + ``workspace.size(0)``. Kept in the Module API
        # for call-site compatibility.
        del seq_lens
        _C.cp_gather_and_upconvert_fp8_kv_cache(
            kv_cache, workspace, block_table, workspace_starts, num_seqs, None,
        )

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

_FP8_INFO = torch.finfo(torch.float8_e4m3fn)

def _cat_quant_fp8_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    scale_ptr,
    a_stride_n: tl.int64,
    a_stride_h: tl.int64,
    b_stride_n: tl.int64,
    b_stride_h: tl.int64,
    fp8_min,
    fp8_max,
    H: tl.constexpr,
    DA: tl.constexpr,
    DB: tl.constexpr,
    BLOCK_A: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """One program per (token, head) row of the output.

    ``a`` is the absorbed q_nope, which reaches here as a ``bmm(...).transpose``
    view, and ``b`` is a trailing slice of the projected query -- neither is
    contiguous across (token, head), so both take explicit strides. The last
    dimension is contiguous in both cases, so each half is a single vector load.
    """
    row = tl.program_id(0).to(tl.int64)
    n = row // H
    h = row % H

    # Reciprocal once per row, not per element.
    inv_scale = 1.0 / tl.load(scale_ptr)
    out_row = out_ptr + row * (DA + DB)

    offs_a = tl.arange(0, BLOCK_A)
    mask_a = offs_a < DA
    a = tl.load(a_ptr + n * a_stride_n + h * a_stride_h + offs_a,
                mask=mask_a, other=0.0).to(tl.float32)
    a = tl.minimum(tl.maximum(a * inv_scale, fp8_min), fp8_max)
    tl.store(out_row + offs_a, a.to(out_ptr.dtype.element_ty), mask=mask_a)

    offs_b = tl.arange(0, BLOCK_B)
    mask_b = offs_b < DB
    b = tl.load(b_ptr + n * b_stride_n + h * b_stride_h + offs_b,
                mask=mask_b, other=0.0).to(tl.float32)
    b = tl.minimum(tl.maximum(b * inv_scale, fp8_min), fp8_max)
    tl.store(out_row + DA + offs_b, b.to(out_ptr.dtype.element_ty), mask=mask_b)

class CatQuantFP8(nn.Module):
    """Concat two ``[N, H, D*]`` halves and per-tensor FP8-quantize in one pass."""

    def forward(
        self,
        a: torch.Tensor,       # [N, H, DA] -- absorbed q_nope
        b: torch.Tensor,       # [N, H, DB] -- q_pe
        scale: torch.Tensor,   # [1] fp32, static per-tensor
    ) -> torch.Tensor:
        assert a.shape[:2] == b.shape[:2], (a.shape, b.shape)
        assert a.stride(-1) == 1 and b.stride(-1) == 1, "last dim must be dense"
        n_tok, n_head, d_a = a.shape
        d_b = b.shape[2]

        out = torch.empty((n_tok, n_head, d_a + d_b),
                          dtype=torch.float8_e4m3fn, device=a.device)
        rows = n_tok * n_head
        if rows == 0:
            return out
        _cat_quant_fp8_kernel[(rows,)](
            a, b, out, scale,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            _FP8_INFO.min, _FP8_INFO.max,
            H=n_head, DA=d_a, DB=d_b,
            BLOCK_A=triton.next_power_of_2(d_a),
            BLOCK_B=triton.next_power_of_2(d_b),
            num_warps=4,
        )
        return out

class FlashMLADecodeFP8(nn.Module):
    """Dense FP8 decode wrapper matching vLLM's ``flash_mla_with_kvcache_fp8``.

    Requires ``descale_q`` / ``descale_k`` (per-layer Q/K dequantization
    scales) and a ``num_splits`` tensor produced by
    :class:`FlashMLAGetMetadataDenseFP8`.
    """

    def forward(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        head_dim_v: int,
        tile_scheduler_metadata: torch.Tensor,
        num_splits: torch.Tensor,
        softmax_scale: float,
        causal: bool = True,
        descale_q: torch.Tensor | None = None,
        descale_k: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return flash_mla_with_kvcache_fp8(
            q=q,
            k_cache=k_cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            head_dim_v=head_dim_v,
            tile_scheduler_metadata=tile_scheduler_metadata,
            num_splits=num_splits,
            softmax_scale=softmax_scale,
            causal=causal,
            descale_q=descale_q,
            descale_k=descale_k,
        )

class FlashInferMLADecode(nn.Module):
    """trtllm-gen MLA decode over a paged latent KV cache.

    Parameters mirror the FlashMLA decode op this stands in for, so the
    call site does not need to branch beyond choosing the module.
    """

    # vLLM keeps one workspace per (return_lse) variant; a single shared
    # buffer is enough here since we never request the LSE.
    _WORKSPACE_BYTES = 128 * 1024 * 1024

    def __init__(
        self,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        kv_lora_rank: int,
        workspace: torch.Tensor | None = None,
    ):
        super().__init__()
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self._workspace = workspace

    @property
    def available(self) -> bool:
        return flashinfer_mla_decode_supported()

    def ensure_workspaces(self, device: torch.device) -> None:
        """Materialize the trtllm-gen MLA decode workspace before graph capture.

        See ``TopKPerRow.ensure_workspaces``: a workspace first allocated inside
        a capture region belongs to that graph's private pool, and later graphs
        replaying against it fault.
        """
        self._get_workspace(device)

    def _get_workspace(self, device: torch.device) -> torch.Tensor:
        if self._workspace is None or self._workspace.device != device:
            self._workspace = torch.zeros(
                self._WORKSPACE_BYTES, dtype=torch.uint8, device=device,
            )
        return self._workspace

    def forward(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        softmax_scale: float,
        max_seq_len: int,
        bmm2_scale: float = 1.0,
    ):
        """
        Parameters
        ----------
        q : ``[num_decodes, q_len, num_heads, qk_head_dim]``
            Same layout the FlashMLA decode op receives (``q.unsqueeze(1)``).
        kv_cache : ``[num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]``
            Paged latent cache. vLLM passes ``kv_c_and_k_pe_cache.unsqueeze(1)``,
            i.e. a singleton head dim, which we add here if absent.
        block_table, cache_seqlens
            Per-request page table and context lengths.

        Returns ``(out, None)`` to match the FlashMLA op's ``(out, lse)``.
        """
        if kv_cache.dim() == 3:
            kv_cache = kv_cache.unsqueeze(1)

        # trtllm-gen walks the page table in 128-token strides, so it rejects a
        # width that is not a multiple of ceil(128 / page_size). Callers size
        # the table from max_model_len, so pad here as a backstop; the extra
        # columns are never read (the walk is bounded by seq_lens).
        page_size = kv_cache.shape[2]
        gran = max(1, -(-128 // page_size))
        if block_table.shape[-1] % gran:
            pad = gran - block_table.shape[-1] % gran
            block_table = torch.nn.functional.pad(block_table, (0, pad))

        # trtllm-gen reads the page table and seq lens as dense row-major
        # tensors; a sliced view would make every row > 0 read at the wrong
        # stride. vLLM asserts strict contiguity for the same reason.
        out = trtllm_batch_decode_with_kv_cache_mla(
            query=q.contiguous(),
            kv_cache=kv_cache,
            workspace_buffer=self._get_workspace(q.device),
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_table.contiguous(),
            seq_lens=cache_seqlens.contiguous(),
            max_seq_len=int(max_seq_len),
            bmm1_scale=softmax_scale,
            bmm2_scale=bmm2_scale,
            return_lse=False,
        )
        return out, None

_MLA_WORKSPACE_HEAD_SIZE = 576

class MLAAttention(nn.Module):
    """MLA attention with FP8 paged KV cache.

    Unlike standard Attention which has separate k_cache and v_cache,
    MLA uses a single unified cache since kv_c_normed + k_pe are stored together.

    Attributes:
        k_cache, v_cache: both point to the same tensor for engine discovery
        _num_kv_heads: always 1 (MLA = multi-query on the latent)
        _head_dim: kv_lora_rank + qk_rope_head_dim (for cache slot size)
    """

    def __init__(self, num_heads: int, scale: float,
                 qk_nope_head_dim: int, qk_rope_head_dim: int,
                 v_head_dim: int, kv_lora_rank: int,
                 is_sparse: bool = False,
                 kv_cache_dtype: str | None = None,
                 topk_tokens: int | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = scale
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.is_sparse = is_sparse
        # ``index_topk``. Needed by the FLASHINFER_MLA_SPARSE path both as the
        # kernel's ``sparse_mla_top_k`` / ``max_seq_len`` and as the threshold
        # that decides whether prefill can take the dense MHA route.
        self.topk_tokens = topk_tokens

        if kv_cache_dtype is None:
            kv_cache_dtype = _default_kv_cache_dtype()
        assert kv_cache_dtype in ("auto", "fp8_ds_mla", "fp8_e4m3"), (
            f"MLAAttention: unsupported kv_cache_dtype={kv_cache_dtype!r}"
        )
        self.kv_cache_dtype = kv_cache_dtype
        self.use_fp8_kv_cache = kv_cache_dtype == "fp8_ds_mla"
        # vLLM's FLASHINFER_MLA_SPARSE backend: plain per-tensor fp8 cache with
        # the trtllm-gen sparse MQA kernel. Only reachable for a sparse (DSA)
        # model — vLLM's ``supports_combination`` requires ``index_topk``.
        self.use_flashinfer_sparse = kv_cache_dtype == "fp8_e4m3"
        if self.use_flashinfer_sparse:
            if not is_sparse or topk_tokens is None:
                raise ValueError(
                    "kv_cache_dtype='fp8_e4m3' selects vLLM's "
                    "FLASHINFER_MLA_SPARSE backend, which only supports sparse "
                    "(DSA) models with an index_topk"
                )
            if not flashinfer_mla_sparse_available():
                raise RuntimeError(
                    "kv_cache_dtype='fp8_e4m3' needs a Blackwell (sm100) GPU "
                    "with flashinfer's trtllm_batch_decode_with_kv_cache_mla "
                    "and trtllm_ragged_attention_deepseek"
                )
            if qk_nope_head_dim not in (128, 192):
                raise ValueError(
                    "FlashInfer MLA Sparse kernel requires qk_nope_head_dim in "
                    f"[128, 192], but got {qk_nope_head_dim}"
                )

        self._num_kv_heads = 1
        # ``_head_dim`` is used by external callers (e.g. the engine) to
        # size the cache. For BF16 it's the packed
        # ``kv_lora_rank + qk_rope_head_dim`` (576). For FP8 the cache is
        # uint8 with 656 bytes/token, which we continue to advertise here.
        # ``fp8_e4m3`` keeps the 576 slot width but stores one byte per element,
        # matching ``FlashInferMLASparseTRTLLMBackend.get_kv_cache_shape``.
        self._head_dim = (
            kv_lora_rank + qk_rope_head_dim if not self.use_fp8_kv_cache else 656
        )

        self.k_cache = self.v_cache = torch.tensor([])

        self.fp8_decode_padded_heads = _compute_fp8_decode_padded_heads(num_heads)
        # BF16 sparse prefill kernel head-pad: 64 on Hopper, 128 on Blackwell
        # (matches vLLM's ``FlashMLASparseImpl.prefill_padding``).
        self.prefill_padding = _compute_prefill_padding()

        # W_UV: absorbed V projection from kv_b_proj, computed after weight loading.
        # Shape: [num_heads, kv_lora_rank, v_head_dim]
        self.W_UV: torch.Tensor | None = None
        # W_UK_T: absorbed K projection transposed, for decode query absorption.
        # Shape: [num_heads, qk_nope_head_dim, kv_lora_rank]
        self.W_UK_T: torch.Tensor | None = None

        self.store_kvcache = StoreKVCacheFP8MLA(kv_cache_dtype=kv_cache_dtype)
        self.gather_kvcache = GatherKVCacheFP8MLA()
        self.gather_dequant_kvcache = GatherAndDequantKVCacheMLA(
            kv_cache_dtype=kv_cache_dtype,
        )
        self.decode_op = FlashMLADecode()
        # Dense FP8 decode entry-point (matches vLLM's
        # ``flash_mla_with_kvcache_fp8`` path used in
        # ``vllm/v1/attention/backends/mla/flashmla.py``).
        self.decode_op_fp8 = FlashMLADecodeFP8()
        # Blackwell: FlashMLA's dense decode is SM90a-only, so use the
        # trtllm-gen MLA decode kernel vLLM selects there (FLASHINFER_MLA).
        # ``None`` on other devices keeps the FlashMLA path untouched.
        self.decode_op_flashinfer = (
            FlashInferMLADecode(
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                kv_lora_rank=kv_lora_rank,
            )
            if flashinfer_mla_decode_supported()
            else None
        )
        self.sparse_prefill_op = FlashMLASparsePrefill()
        # FLASHINFER_MLA_SPARSE ops. Built only for that cache dtype so the other
        # paths do not pay the flashinfer import or the workspace allocation.
        if self.use_flashinfer_sparse:
            self.fi_sparse_decode = FlashInferMLASparseDecode(
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                kv_lora_rank=kv_lora_rank,
            )
            self.ragged_prefill = TrtllmRaggedPrefill(scale=scale)
            # Fused concat+quant for the decode query: byte-identical to
            # ``cat`` + ``QuantFp8MLAQuery`` but one kernel instead of two. See
            # cat_quant_fp8.py for why vLLM gets this fusion from Inductor and
            # fastkernels has to write it by hand.
            self.cat_quant = CatQuantFP8()
        self._fi_bmm_scale_cache: tuple[float, float] | None = None
        self.get_metadata = FlashMLAGetMetadata()
        self.get_metadata_dense_fp8 = FlashMLAGetMetadataDenseFP8()
        self.varlen_attn = FlashAttnVarlen()
        self.merge_states = MergeAttnStates()
        self.bmm = BatchMatMul()
        self.convert_indices = ConvertIndicesToGlobal()

        # Per-layer dequant scales for FP8 dense MLA decode. vLLM populates
        # these via ``maybe_calc_kv_scales`` (currently 1.0 unless calibrated).
        # We mirror the ``layer._q_scale`` / ``layer._k_scale`` buffers from
        # ``vllm/model_executor/layers/attention/attention.py:95-100``.
        self.register_buffer(
            "_q_scale", torch.ones(1, dtype=torch.float32), persistent=False,
        )
        self.register_buffer(
            "_k_scale", torch.ones(1, dtype=torch.float32), persistent=False,
        )

        # Custom-op dispatch scaffolding (matches Attention L2 module):
        # ``_use_custom_op`` is flipped to True by ``enable_custom_ops`` once
        # the model is wrapped with ``torch.compile``. ``_layer_name`` is
        # populated by ``auto_register_no_compile_layers``.
        self._use_custom_op = False
        self._layer_name = ""
        # Reference to the enclosing ``kv_b_proj`` module, set by the parent
        # ``DeepSeekMLAAttention``. Stored via ``object.__setattr__`` at the
        # parent site to avoid double-registration as an ``nn.Module``
        # submodule (which would shadow parent weights). ``None`` until
        # wired up.
        self._kv_b_proj: nn.Module | None = None

    def forward(self, q: torch.Tensor, kv_c_normed: torch.Tensor,
                k_pe: torch.Tensor, kv_b_proj: nn.Module | None = None,
                topk_indices: torch.Tensor | None = None,
                output_shape: tuple | None = None) -> torch.Tensor:
        # Keep the historical positional ``kv_b_proj`` argument for direct
        # (eager / unit-test) callers but prefer the stored reference so the
        # torch.compile custom-op path only has tensor-typed arguments.
        if kv_b_proj is not None and self._kv_b_proj is None:
            object.__setattr__(self, "_kv_b_proj", kv_b_proj)

        # ``output_shape`` is intentionally ignored on the dispatch path: the
        # output is always reshaped to ``(N, num_heads * v_head_dim)`` where
        # ``N = q.shape[0]``. Computing it from ``q`` keeps the batch dim
        # symbolic under torch.compile (passing a precomputed ``int[]`` here
        # would force Dynamo to specialize ``q.shape[0]`` to a constant).
        if self._use_custom_op:
            return torch.ops.fastkernels.unified_mla_attention(
                q, kv_c_normed, k_pe, topk_indices, self._layer_name,
            )
        return self.forward_impl(q, kv_c_normed, k_pe, topk_indices)

    def forward_impl(self, q: torch.Tensor, kv_c_normed: torch.Tensor,
                     k_pe: torch.Tensor,
                     topk_indices: torch.Tensor | None = None) -> torch.Tensor:
        ctx = get_context()
        N = q.shape[0]

        kv_cache = self.k_cache
        kv_b_proj = self._kv_b_proj
        assert kv_b_proj is not None, "MLAAttention._kv_b_proj is not wired"

        if kv_cache.numel() and ctx.slot_mapping is not None:
            self.store_kvcache(kv_c_normed, k_pe, kv_cache, ctx.slot_mapping)

        if self.is_sparse and topk_indices is not None and kv_cache.ndim >= 2:
            o = self._forward_sparse(q, kv_c_normed, k_pe, kv_b_proj, kv_cache, ctx, topk_indices)
        elif ctx.is_mixed:
            o = self._forward_mixed(q, kv_c_normed, k_pe, kv_b_proj, kv_cache, ctx)
        else:
            o = self._forward_pure(q, kv_c_normed, k_pe, kv_b_proj, kv_cache, ctx)

        return o.view(N, self.num_heads * self.v_head_dim)

    def _forward_pure(self, q, kv_c_normed, k_pe, kv_b_proj, kv_cache, ctx):
        if ctx.is_prefill:
            return self._forward_mha(q, kv_c_normed, k_pe, kv_b_proj, kv_cache, ctx)
        return self._forward_dense_decode(q, kv_cache, ctx)

    def _run_prefill_new_tokens(self, q, k, v, cu_seqlens_q, cu_seqlens_k,
                                max_seqlen_q, max_seqlen_k,
                                return_softmax_lse=False):
        """Run causal attention on new prefill tokens.

        Uses the trtllm-gen ragged kernel when the layer is on the
        FLASHINFER_MLA_SPARSE path -- that is the MLA prefill backend vLLM's
        ``get_mla_prefill_backend`` selects on sm100 for these head dims
        ("Using TRTLLM_RAGGED MLA prefill backend"), and FlashAttention's
        different accumulation order would show up in every prefill token.
        Every other path keeps the L1 ``FlashAttnVarlen`` op, which is what vLLM
        selects for them.
        """
        if self.use_flashinfer_sparse:
            seq_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int32)
            ret = self.ragged_prefill(
                q, k, v,
                seq_lens=seq_lens,
                cu_seq_lens_q=cu_seqlens_q,
                cu_seq_lens_kv=cu_seqlens_q,
                max_q_len=max_seqlen_q,
                max_kv_len=max_seqlen_q,
                is_causal=True,
                return_lse=return_softmax_lse,
            )
            if isinstance(ret, tuple):
                return ret[0], ret[1]
            if return_softmax_lse:
                return ret, None
            return ret
        attn_out = self.varlen_attn(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            return_softmax_lse=return_softmax_lse,
        )
        if isinstance(attn_out, tuple):
            return attn_out[0], attn_out[1]
        if return_softmax_lse:
            return attn_out, None
        return attn_out

    def _run_prefill_context_chunk(self, q, k, v, cu_seqlens_q, cu_seqlens_k,
                                   max_seqlen_q, max_seqlen_k,
                                   chunk_seq_lens=None):
        """Run non-causal attention on a context chunk (always returns LSE).

        ``chunk_seq_lens`` is the per-request KV length of this chunk; the
        trtllm-gen ragged kernel takes it explicitly (vLLM passes
        ``chunked_context.seq_lens[chunk_idx]``), whereas FlashAttention derives
        it from ``cu_seqlens_k``. Falls back to the difference of
        ``cu_seqlens_k`` when the caller does not supply it.
        """
        if self.use_flashinfer_sparse:
            if chunk_seq_lens is None:
                chunk_seq_lens = (cu_seqlens_k[1:] - cu_seqlens_k[:-1])
            ret = self.ragged_prefill(
                q, k, v,
                seq_lens=chunk_seq_lens.to(
                    device=q.device, dtype=torch.int32, non_blocking=True,
                ),
                cu_seq_lens_q=cu_seqlens_q,
                cu_seq_lens_kv=cu_seqlens_k,
                max_q_len=max_seqlen_q,
                max_kv_len=max_seqlen_k,
                is_causal=False,
                return_lse=True,
            )
            return ret[0], ret[1]
        attn_out = self.varlen_attn(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=False,
            return_softmax_lse=True,
        )
        if isinstance(attn_out, tuple):
            return attn_out[0], attn_out[1]
        return attn_out, None

    def _concat_k_nope_k_pe(self, k_nope, k_pe):
        """Concatenate k_nope and expanded k_pe along the head_dim."""
        k = torch.empty(
            (*k_nope.shape[:-1], k_nope.shape[-1] + k_pe.shape[-1]),
            dtype=k_nope.dtype, device=k_nope.device,
        )
        k[..., :k_nope.shape[-1]] = k_nope
        k[..., k_nope.shape[-1]:] = k_pe
        return k

    def _compute_prefill_context(self, q, kv_cache, kv_b_proj, ctx):
        """Gather cached context, up-project, run non-causal attn, merge chunks.

        Matches vllm's MLACommonImpl._compute_prefill_context:
        for each context chunk, gather from FP8 cache into BF16 workspace,
        split into kv_c_normed and k_pe, project kv_c_normed through kv_b_proj
        to get k_nope and v, run non-causal attention, merge with
        merge_attn_states.
        """
        chunked_ctx = ctx.chunked_context
        assert chunked_ctx is not None

        output = None
        output_lse = None
        iters = len(chunked_ctx.seq_tot)
        workspace = chunked_ctx.workspace

        if ctx.is_mixed:
            query_start_loc = ctx.prefill_cu_seqlens_q
            max_query_len = ctx.prefill_max_seqlen_q
        else:
            query_start_loc = ctx.cu_seqlens_q
            max_query_len = ctx.max_seqlen_q

        for i in range(iters):
            toks = chunked_ctx.seq_tot[i]

            block_table = (
                ctx.prefill_block_tables if ctx.is_mixed else ctx.block_tables
            )

            self.gather_dequant_kvcache(
                kv_cache, workspace, block_table,
                chunked_ctx.cu_seq_lens[i],
                chunked_ctx.token_to_seq[i],
                chunked_ctx.chunk_total_token[i],
                chunked_ctx.starts[i],
            )

            kv_c_normed = workspace[:toks, :self.kv_lora_rank]
            k_pe = workspace[:toks, self.kv_lora_rank:].unsqueeze(1)

            kv_nope = kv_b_proj(kv_c_normed)
            kv_nope = kv_nope.view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            k = self._concat_k_nope_k_pe(k_nope, k_pe)

            attn_output, attn_softmax_lse = self._run_prefill_context_chunk(
                q=q, k=k, v=v,
                cu_seqlens_q=query_start_loc,
                cu_seqlens_k=chunked_ctx.cu_seq_lens[i],
                max_seqlen_q=max_query_len,
                max_seqlen_k=chunked_ctx.max_seq_lens[i],
                # ``seq_lens`` is a CPU ``[num_chunks, num_prefills]`` tensor;
                # row ``i`` is this chunk's per-request KV length, which is what
                # vLLM hands the ragged kernel as
                # ``chunked_context.seq_lens[chunk_idx]``.
                chunk_seq_lens=chunked_ctx.seq_lens[i],
            )

            # A request with no context in this chunk attended to zero keys,
            # so the backend left its output rows as scratch (possibly
            # NaN/Inf) even though the LSE is -inf.  Neutralize before the
            # merge, exactly as vLLM does.
            if (
                i < len(chunked_ctx.has_empty_context)
                and chunked_ctx.has_empty_context[i]
            ):
                _mask_empty_context(
                    attn_softmax_lse,
                    attn_output,
                    query_start_loc,
                    chunked_ctx.cu_seq_lens[i],
                )

            if output is None:
                output = attn_output
                output_lse = attn_softmax_lse
            else:
                output_tmp = torch.empty_like(output)
                output_lse_tmp = torch.empty_like(output_lse)
                self.merge_states(
                    output=output_tmp,
                    prefix_output=output,
                    prefix_lse=output_lse,
                    suffix_output=attn_output,
                    suffix_lse=attn_softmax_lse,
                    output_lse=output_lse_tmp,
                )
                output = output_tmp
                output_lse = output_lse_tmp

        return output, output_lse

    def _forward_mha(self, q, kv_c_normed, k_pe, kv_b_proj, kv_cache, ctx):
        """Dense prefill with chunked context support (matches vllm forward_mha)."""
        N = q.shape[0]
        has_context = ctx.chunked_context is not None

        kv = kv_b_proj(kv_c_normed)
        kv = kv.view(N, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = self._concat_k_nope_k_pe(k_nope, k_pe)

        if ctx.is_mixed:
            cu_q = ctx.prefill_cu_seqlens_q
            cu_k = ctx.prefill_cu_seqlens_k
            max_sq = ctx.prefill_max_seqlen_q
            max_sk = ctx.prefill_max_seqlen_k
        else:
            cu_q = ctx.cu_seqlens_q
            cu_k = ctx.cu_seqlens_k
            max_sq = ctx.max_seqlen_q
            max_sk = ctx.max_seqlen_k

        output_prefill = self._run_prefill_new_tokens(
            q, k, v,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_q,
            max_seqlen_q=max_sq, max_seqlen_k=max_sq,
            return_softmax_lse=has_context,
        )

        if has_context:
            suffix_output, suffix_lse = output_prefill
            context_output, context_lse = self._compute_prefill_context(
                q, kv_cache, kv_b_proj, ctx)

            output = torch.empty(N, self.num_heads, self.v_head_dim,
                                 dtype=q.dtype, device=q.device)
            self.merge_states(
                output=output,
                prefix_output=context_output,
                prefix_lse=context_lse,
                suffix_output=suffix_output[..., :self.v_head_dim],
                suffix_lse=suffix_lse,
            )
            return output.reshape(N, self.num_heads * self.v_head_dim)
        else:
            o = output_prefill
            if isinstance(o, tuple):
                o = o[0]
            return o.reshape(N, self.num_heads * self.v_head_dim)

    def _v_up_proj(self, attn_out: torch.Tensor) -> torch.Tensor:
        """Project FlashMLA output from kv_lora_rank to v_head_dim per head.

        Matches vllm's MLAAttention._v_up_proj: (B, N, L) -> (N, B, L) x
        (N, L, V) -> (N, B, V) -> (B, N*V).
        """
        if self.W_UV is None:
            return attn_out[..., :self.v_head_dim]
        N = attn_out.shape[0]
        o = attn_out.view(N, self.num_heads, self.kv_lora_rank)
        o = o.transpose(0, 1)  # (N, B, L)
        out = self.bmm(o, self.W_UV)  # (N, B, V)
        return out.transpose(0, 1).reshape(N, self.num_heads * self.v_head_dim)

    def _forward_dense_decode(self, q, kv_cache, ctx):
        cache_seqlens = ctx.context_lens
        block_table = ctx.block_tables
        if not self.use_fp8_kv_cache:
            q = self._absorb_q_to_latent(q)
            q = q.unsqueeze(1)
            # FlashMLA's dense decode kernel is SM90a-only ("Dense decode MLA
            # is only supported on SM90a architecture"), so on Blackwell vLLM
            # runs FLASHINFER_MLA / trtllm-gen instead. Follow the same choice.
            if self.decode_op_flashinfer is not None:
                o, _ = self.decode_op_flashinfer(
                    q,
                    kv_cache,
                    block_table,
                    cache_seqlens,
                    softmax_scale=self.scale,
                    max_seq_len=ctx.max_context_len,
                )
                o = o.reshape(-1, o.shape[-2], o.shape[-1])
                return self._v_up_proj(o)
            # Prefer the engine's persistent FlashMLASchedMeta (filled
            # outside CUDA-graph capture). Fall back to a fresh empty
            # object for eager mixed/prefill.
            tile_sched_meta = getattr(ctx, "mla_sched_meta", None)
            if tile_sched_meta is None:
                tile_sched_meta, _ = self.get_metadata(
                    cache_seqlens, self.num_heads, num_heads_k=1,
                )
            o, _ = self.decode_op(
                q,
                kv_cache.unsqueeze(-2),
                block_table,
                cache_seqlens,
                head_dim_v=_MLA_HEAD_DIM_V,
                tile_scheduler_metadata=tile_sched_meta,
                softmax_scale=self.scale,
                causal=True,
            )
            o = o.reshape(-1, o.shape[-2], o.shape[-1])
            return self._v_up_proj(o)

        # Mirrors vLLM's dense FP8 MLA decode path
        # (vllm/v1/attention/backends/mla/flashmla.py:289-302):
        # specialized ``flash_mla_with_kvcache_fp8`` with per-layer
        # ``descale_q`` / ``descale_k`` and ``causal=True``.
        tile_sched_meta, num_splits = self.get_metadata_dense_fp8(
            cache_seqlens, self.num_heads, num_heads_k=1,
        )
        o, _ = self.decode_op_fp8(
            q.unsqueeze(1), kv_cache.view(torch.uint8).unsqueeze(-2),
            block_table, cache_seqlens,
            head_dim_v=_MLA_HEAD_DIM_V,
            tile_scheduler_metadata=tile_sched_meta,
            num_splits=num_splits,
            softmax_scale=self.scale,
            causal=True,
            descale_q=self._q_scale.reshape(1),
            descale_k=self._k_scale.reshape(1),
        )
        o = o.reshape(-1, o.shape[-2], o.shape[-1])
        return self._v_up_proj(o)

    def _forward_sparse(self, q, kv_c_normed, k_pe, kv_b_proj, kv_cache, ctx, topk_indices):
        """Sparse attention dispatcher.

        Mirrors ``vllm/v1/attention/backends/mla/flashmla_sparse.py``'s
        ``FlashMLASparseImpl.forward_mqa`` selection:

        * ``kv_cache_dtype="auto"`` (BF16): **single kernel path** for
          both prefill and decode — ``flash_mla_sparse_fwd`` reads the
          BF16 paged cache directly. Matches vLLM's ``_forward_bf16_kv``.
        * ``kv_cache_dtype="fp8_ds_mla"``:

          * mixed-batch FP8 path (``num_heads < MIN_HEADS_FOR_BF16_PREFILL``):
            one ``flash_mla_with_kvcache`` call for all tokens.
          * separate prefill / decode FP8 path (large head count): BF16
            workspace prefill + FP8 decode kernel.
          * pure FP8 decode path.
        """
        N = q.shape[0]

        if self.use_flashinfer_sparse:
            return self._forward_sparse_flashinfer(
                q, kv_c_normed, k_pe, kv_b_proj, kv_cache, ctx, topk_indices,
            )

        if not self.use_fp8_kv_cache:
            return self._forward_sparse_bf16(q, kv_cache, ctx, topk_indices)

        use_mixed_batch = self.num_heads < MIN_HEADS_FOR_BF16_PREFILL

        if ctx.is_prefill:
            num_pf, num_dc = N, 0
            is_mixed_or_prefill = True
        elif ctx.is_mixed:
            num_pf = ctx.num_prefill_tokens
            num_dc = ctx.num_decode_tokens
            is_mixed_or_prefill = True
        else:
            return self._forward_sparse_decode(q, kv_cache, ctx, topk_indices)

        if is_mixed_or_prefill and use_mixed_batch:
            return self._forward_sparse_mixed_batch(
                q, kv_cache, ctx, topk_indices,
                num_prefill_tokens=num_pf,
                num_decode_tokens=num_dc,
            )
        return self._forward_sparse_separate(
            q, kv_c_normed, k_pe, kv_b_proj, kv_cache, ctx, topk_indices,
            num_prefill_tokens=num_pf, num_decode_tokens=num_dc)

    def _build_unified_sparse_batch_meta(self, q, ctx):
        """Unified block table + per-token request ids for the sparse paths.

        The sparse kernels index a single flat cache, so the per-request top-k
        indices must be translated with one block table covering the whole
        packed batch. fastkernels packs decode rows first and prefill rows
        after, so the table is ``cat([decode_block_tables, prefill_block_tables])``
        (right-padded to a common width) and request ids are numbered in that
        same order.

        Memoized on the context: this is batch metadata, identical for every
        layer, and vLLM likewise builds it once per step in its metadata builder.
        Recomputing it per layer costs two ``torch.cat``s of the block tables on
        every one of the model's 78 attention layers.

        Returns ``None`` when the batch has no block table at all (nothing to
        attend to), letting the caller return zeros.
        """
        cached = getattr(ctx, "_fk_sparse_batch_meta", _UNSET)
        if cached is not _UNSET:
            return cached
        meta = self._compute_unified_sparse_batch_meta(q, ctx)
        ctx._fk_sparse_batch_meta = meta
        return meta

    def _compute_unified_sparse_batch_meta(self, q, ctx):
        N = q.shape[0]
        if ctx.is_mixed:
            num_dc = ctx.num_decode_tokens
            num_pf = ctx.num_prefill_tokens
            num_decode_seqs = (
                ctx.decode_block_tables.shape[0]
                if ctx.decode_block_tables is not None and num_dc > 0 else 0
            )
            num_prefill_seqs = (
                ctx.prefill_block_tables.shape[0]
                if ctx.prefill_block_tables is not None and num_pf > 0 else 0
            )
            if num_decode_seqs > 0 and num_prefill_seqs > 0:
                d_bt = ctx.decode_block_tables
                p_bt = ctx.prefill_block_tables
                max_b = max(d_bt.shape[1], p_bt.shape[1])
                if d_bt.shape[1] < max_b:
                    pad = torch.full((d_bt.shape[0], max_b - d_bt.shape[1]),
                                     -1, dtype=d_bt.dtype, device=d_bt.device)
                    d_bt = torch.cat([d_bt, pad], dim=1)
                if p_bt.shape[1] < max_b:
                    pad = torch.full((p_bt.shape[0], max_b - p_bt.shape[1]),
                                     -1, dtype=p_bt.dtype, device=p_bt.device)
                    p_bt = torch.cat([p_bt, pad], dim=1)
                unified_block_table = torch.cat([d_bt, p_bt], dim=0)
            elif num_prefill_seqs > 0:
                unified_block_table = ctx.prefill_block_tables
            elif num_decode_seqs > 0:
                unified_block_table = ctx.decode_block_tables
            else:
                return None
        else:
            if ctx.block_tables is None:
                return None
            unified_block_table = ctx.block_tables
            num_dc = 0 if ctx.is_prefill else N
            num_pf = N if ctx.is_prefill else 0
            num_decode_seqs = (
                0 if ctx.is_prefill else unified_block_table.shape[0]
            )
            num_prefill_seqs = (
                unified_block_table.shape[0] if ctx.is_prefill else 0
            )

        req_ids = ctx.req_id_per_token
        if req_ids is None:
            req_ids = torch.zeros(N, dtype=torch.int32, device=q.device)
            if ctx.is_mixed:
                for i in range(num_decode_seqs):
                    req_ids[i] = i
                pf_cu_q = ctx.prefill_cu_seqlens_q
                if pf_cu_q is not None:
                    for r in range(num_prefill_seqs):
                        qs = int(pf_cu_q[r].item()) + num_dc
                        qe = int(pf_cu_q[r + 1].item()) + num_dc
                        req_ids[qs:qe] = num_decode_seqs + r
            else:
                cu_q = ctx.cu_seqlens_q
                if cu_q is not None:
                    nseqs = (
                        num_prefill_seqs if num_prefill_seqs > 0
                        else num_decode_seqs
                    )
                    for r in range(nseqs):
                        qs = int(cu_q[r].item())
                        qe = int(cu_q[r + 1].item())
                        req_ids[qs:qe] = r
        return unified_block_table, req_ids, num_dc, num_pf

    def _forward_sparse_bf16(self, q, kv_cache, ctx, topk_indices):
        """BF16 KV cache sparse path, identical to vLLM's ``_forward_bf16_kv``.

        All tokens (prefill, decode, mixed) go through a single
        ``flash_mla_sparse_fwd`` call over the BF16 paged cache:

        * convert per-request ``topk_indices`` into global slot indices
          (``convert_indices`` with the batch's unified block table);
        * absorb ``q`` through ``W_UK_T`` and concat with ``q_pe`` to get
          a 576-D head ("MQA 576/512 approach");
        * pad the head count to ``self.prefill_padding`` (64 on Hopper /
          128 on Blackwell) as required by the BF16 sparse kernel;
        * call ``flash_mla_sparse_fwd(q, kv_cache.view(-1, 1, 576),
          topk_indices.view(N, 1, topk), sm_scale)``;
        * slice output heads back to ``num_heads`` and up-project via
          ``W_UV``.
        """
        N = q.shape[0]
        block_size = int(kv_cache.shape[1])
        num_heads = self.num_heads
        pad_h = self.prefill_padding

        meta = self._build_unified_sparse_batch_meta(q, ctx)
        if meta is None:
            return torch.zeros(
                N, num_heads * self.v_head_dim, dtype=q.dtype, device=q.device,
            )
        unified_block_table, req_ids, _num_dc, _num_pf = meta

        topk_global = self.convert_indices(
            topk_indices, unified_block_table, block_size, req_ids=req_ids,
        )

        # Absorb q into the 576-D latent space.
        q_latent = self._absorb_q_to_latent(q)  # [N, H, 576]

        # Pad heads to multiple of ``prefill_padding`` (BF16 sparse kernel
        # requirement, see ``vllm/v1/attention/backends/mla/flashmla_sparse
        # .py:_bf16_flash_mla_kernel``).
        actual_heads = q_latent.shape[1]
        if actual_heads % pad_h != 0:
            assert pad_h % actual_heads == 0
            q_pad = q_latent.new_empty(N, pad_h, q_latent.shape[2])
            q_pad[:, :actual_heads, :] = q_latent
            q_latent = q_pad

        # View the BF16 paged cache as (num_blocks * block_size, 1, 576)
        # so ``flash_mla_sparse_fwd`` can gather by global slot index.
        kv_flat = kv_cache.view(-1, 1, kv_cache.shape[-1])
        topk_3d = topk_global.view(N, 1, -1)

        out = self.sparse_prefill_op(q_latent, kv_flat, topk_3d, self.scale)
        if isinstance(out, (tuple, list)):
            out = out[0]
        # Trim padded heads back to num_heads.
        out = out[:, :num_heads, :]
        return self._v_up_proj(out)

    def _forward_sparse_flashinfer(self, q, kv_c_normed, k_pe, kv_b_proj,
                                   kv_cache, ctx, topk_indices):
        """vLLM's FLASHINFER_MLA_SPARSE path over a plain fp8 MLA cache.

        Mirrors the split in ``MLAAttention.forward_impl``
        (vllm/model_executor/layers/attention/mla_attention.py:753-903):

        * The packed batch is decode-first, so ``q[:num_mqa]`` are the sparse
          MQA tokens and ``q[num_mqa:]`` the dense MHA (prefill) ones.
        * Prefill takes the **dense** MHA route only when every prefill sequence
          is at most ``index_topk`` long -- at that length the indexer's top-k
          selects the whole context, so dense and sparse compute the same
          attention and vLLM uses the cheaper dense kernel. Longer prefills fall
          back to sparse for ALL tokens (``num_mqa_tokens = q.size(0)``).
        * The MQA half absorbs q into the 576-D latent, quantizes it to fp8
          (the trtllm-gen kernel needs q and cache to share a dtype), and calls
          ``trtllm_batch_decode_with_kv_cache_mla`` with the per-token top-k
          global slots as a page-size-1 block table.

        ``bmm1_scale``/``bmm2_scale`` fold the q/k dequant scales in exactly as
        ``FlashInferMLASparseImpl.forward_mqa`` does.
        """
        N = q.shape[0]
        num_heads = self.num_heads
        block_size = int(kv_cache.shape[1])

        meta = self._build_unified_sparse_batch_meta(q, ctx)
        if meta is None:
            return torch.zeros(
                N, num_heads * self.v_head_dim, dtype=q.dtype, device=q.device,
            )
        unified_block_table, req_ids, num_dc, _num_pf = meta

        # --- Decide the prefill route (vLLM's ``use_mha`` gate) --------------
        num_mqa = num_dc
        num_mha = N - num_mqa
        if num_mha > 0:
            prefill_max_seq_len = (
                ctx.prefill_max_seqlen_k if ctx.is_mixed else ctx.max_seqlen_k
            )
            if prefill_max_seq_len > self.topk_tokens:
                num_mqa, num_mha = N, 0

        # Decode-only steps (and any batch that went fully sparse) are the hot
        # path: return the up-projection directly instead of staging it through
        # a full-size buffer.
        if num_mha == 0:
            return self._sparse_mqa(
                q, topk_indices, kv_cache, unified_block_table, req_ids,
                block_size, num_mqa,
            )

        out = torch.empty(
            N, num_heads * self.v_head_dim, dtype=q.dtype, device=q.device,
        )

        if num_mha > 0:
            # Dense prefill over the new tokens (plus chunked context when the
            # scheduler split the prompt). ``_forward_mha`` reads the prefill
            # slice of the context, so hand it the prefill tokens only.
            mha_out = self._forward_mha(
                q[num_mqa:], kv_c_normed[num_mqa:], k_pe[num_mqa:],
                kv_b_proj, kv_cache, ctx,
            )
            out[num_mqa:] = mha_out

        if num_mqa > 0:
            out[:num_mqa] = self._sparse_mqa(
                q, topk_indices, kv_cache, unified_block_table, req_ids,
                block_size, num_mqa,
            )

        return out

    def _sparse_mqa(self, q, topk_indices, kv_cache, block_table, req_ids,
                    block_size, num_mqa):
        """The sparse top-k MQA half of ``forward_mqa`` for ``q[:num_mqa]``.

        Translates each token's top-k request-local positions into global cache
        slots (and the count of valid ones, which the kernel takes as its
        ``seq_lens``), absorbs q into the 576-D latent, quantizes it to fp8, and
        runs the trtllm-gen sparse kernel. Returns ``[num_mqa, H*v_head_dim]``.
        """
        topk_global, valid_counts = self.convert_indices(
            topk_indices[:num_mqa], block_table, block_size,
            req_ids=req_ids[:num_mqa], return_valid_counts=True,
        )
        # One kernel for concat+quantize: the bf16 [n, H, 576] concatenation is
        # never materialised.
        q_fp8 = self.cat_quant(*self._absorb_q_parts(q[:num_mqa]),
                               self._q_scale)
        o = self.fi_sparse_decode(
            q_fp8, kv_cache, topk_global, valid_counts,
            self.topk_tokens, *self.finalize_kv_scales(),
        )
        return self._v_up_proj(o)

    def finalize_kv_scales(self) -> tuple[float, float]:
        """Resolve the sparse-MLA bmm scales to Python floats, once.

        ``(bmm1_scale, bmm2_scale)`` mirrors ``FlashInferMLASparseImpl.forward_mqa``:
        ``bmm1 = sm_scale * q_scale * k_scale`` and ``bmm2 = k_scale`` for a
        quantized cache. vLLM memoizes these too (``if self.bmm1_scale is None``)
        because reading ``_q_scale`` / ``_k_scale`` as Python floats is a
        device-to-host copy, and paying it per layer per decode step would
        serialize every step against the copy engine.

        This MUST run outside CUDA graph capture. The engine calls it from KV
        cache allocation -- after weight loading, before capture -- because the
        pre-capture warmup forward runs with an empty cache and so never reaches
        the sparse decode path: the lazy read would otherwise first happen
        *during* capture, where a D2H copy is recorded into the graph and replay
        fails with ``cudaErrorInvalidAddressSpace`` ("operation not supported on
        global/shared address space").
        """
        cached = self._fi_bmm_scale_cache
        if cached is None:
            k_scale = float(self._k_scale)
            cached = (self.scale * float(self._q_scale) * k_scale, k_scale)
            self._fi_bmm_scale_cache = cached
        return cached

    def _absorb_q_parts(
        self, q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The two halves of the absorbed latent query, unconcatenated.

        ``(q_absorbed [.., H, L], q_pe [.., H, rope])``. Kept separate so the
        fp8 decode path can hand both to one fused concat+quant kernel instead
        of materialising the bf16 concatenation first.
        """
        q_nope = q[..., :self.qk_nope_head_dim]
        q_pe = q[..., self.qk_nope_head_dim:]
        # (H, N, P) @ (H, P, L) -> (H, N, L) -> (N, H, L)
        q_absorbed = self.bmm(
            q_nope.transpose(0, 1), self.W_UK_T,
        ).transpose(0, 1)
        return q_absorbed, q_pe

    def _absorb_q_to_latent(self, q: torch.Tensor) -> torch.Tensor:
        """Absorb q_nope through W_UK_T into the latent space and concat q_pe.

        Output shape: ``[..., H, kv_lora_rank + qk_rope_head_dim]`` (576 for
        DeepSeek-V3.2). Matches vLLM's MLA decode/sparse query absorption.
        """
        return torch.cat(self._absorb_q_parts(q), dim=-1)

    def _forward_sparse_mixed_batch(self, q, kv_cache, ctx, topk_indices,
                                    num_prefill_tokens, num_decode_tokens):
        """Mixed-batch FP8 sparse path (vLLM's ``_forward_fp8_kv_mixed_batch``).

        All tokens are treated as one logical batch of length ``T = N``,
        ``B = 1``, ``H = padded_heads``. This avoids the BF16 prefill kernel's
        head padding overhead and exactly matches what vLLM uses when
        ``num_heads < MIN_HEADS_FOR_BF16_PREFILL`` (e.g. TP=8, 16 heads).

        Mirrors ``vllm/v1/attention/backends/mla/flashmla_sparse.py:
        _forward_fp8_kv_mixed_batch``.
        """
        N = q.shape[0]
        block_size = int(kv_cache.shape[1])

        # Mixed-batch always sources K from the paged FP8 cache (prefill tokens
        # have already been written via ``store_kvcache`` before attention).
        # The triton kernel only needs the per-token req_id + the full
        # block_table — workspace_starts are unused (HAS_PREFILL=False branch).
        req_ids = ctx.req_id_per_token
        if req_ids is None:
            req_ids = torch.arange(N, dtype=torch.int32, device=q.device)
        block_table = ctx.block_tables

        topk_global = self.convert_indices(
            topk_indices, block_table, block_size, req_ids=req_ids,
        )

        # Absorb q into the 576-D latent space.
        q_latent = self._absorb_q_to_latent(q)  # [N, H, 576]

        # Pad heads to 64 or 128 (FP8 sparse decode kernel requirement).
        # Reshape to (B=1, T=N, H, D) and pad along the head dim.
        q_4d = q_latent.unsqueeze(0)  # (1, N, H, 576)
        q_4d, actual_heads = self._pad_q_for_fp8(q_4d)
        padded_heads = q_4d.shape[-2]
        topk_3d = topk_global.unsqueeze(0)  # (1, N, topk)

        # Single-batch metadata (matches vLLM's ``_build_fp8_mixed_decode_prefill``).
        topk = topk_indices.shape[-1]
        topk_tensor = torch.tensor(
            [topk], dtype=torch.int32, device=q.device,
        )
        dummy_bt = torch.empty(
            (1, 1), dtype=torch.int32, device=q.device,
        )
        # Single "sequence" containing all N tokens with padded_heads queries each.
        tile_sched_meta, _ = self.get_metadata(
            topk_tensor, N * padded_heads,
            topk=topk, num_heads_q=padded_heads,
            num_heads_k=1, is_fp8_kvcache=True,
        )

        o, _ = self.decode_op(
            q_4d, kv_cache.view(torch.uint8).unsqueeze(-2),
            dummy_bt, topk_tensor,
            head_dim_v=_MLA_HEAD_DIM_V,
            tile_scheduler_metadata=tile_sched_meta,
            softmax_scale=self.scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=topk_3d,
        )

        # (1, N, padded_heads, 512) -> (N, num_heads, 512)
        o = o.view(N, padded_heads, o.shape[-1])
        if actual_heads < padded_heads:
            o = o[:, :actual_heads, :]
        return self._v_up_proj(o)

    def _pad_q_for_fp8(self, q: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Pad num_heads to 64 or 128 as required by the FP8 sparse decode kernel."""
        actual_heads = q.shape[-2]
        padded_heads = self.fp8_decode_padded_heads
        if actual_heads >= padded_heads:
            return q, actual_heads
        pad_shape = list(q.shape)
        pad_shape[-2] = padded_heads
        q_padded = q.new_zeros(pad_shape)
        q_padded[..., :actual_heads, :] = q
        return q_padded, actual_heads

    def _forward_sparse_decode(self, q, kv_cache, ctx, topk_indices):
        """Sparse FP8 decode: absorb q into latent space, then FlashMLA sparse.

        The sparse decode kernel requires head_size_k == 576 (kv_lora_rank +
        qk_rope_head_dim). We absorb q_nope via W_UK_T: (N,H,P)@(H,P,L) →
        (N,H,L), then concatenate with q_pe to get (N,H,576).
        Matches vllm's MLACommonImpl decode query absorption.
        """
        N = q.shape[0]
        block_size = int(kv_cache.shape[1])
        num_decodes = ctx.block_tables.shape[0]

        req_ids = ctx.req_id_per_token
        if req_ids is None:
            req_ids = torch.arange(N, dtype=torch.int32, device=q.device)

        topk_indices = self.convert_indices(
            topk_indices, ctx.block_tables, block_size, req_ids=req_ids)

        # Absorb q_nope into latent space via W_UK_T
        q_nope = q[..., :self.qk_nope_head_dim]   # [N, H, P]
        q_pe = q[..., self.qk_nope_head_dim:]      # [N, H, rope]

        # (H, N, P) @ (H, P, L) -> (H, N, L) -> (N, H, L)
        q_absorbed = self.bmm(
            q_nope.transpose(0, 1), self.W_UK_T,
        ).transpose(0, 1)

        # Concat absorbed nope + rope -> [N, H, L+rope=576]
        q_latent = torch.cat([q_absorbed, q_pe], dim=-1)

        decode_query_len = N // num_decodes if num_decodes > 0 else N
        q_4d = q_latent.view(num_decodes, decode_query_len, self.num_heads, q_latent.shape[-1])
        topk_4d = topk_indices.view(num_decodes, decode_query_len, -1)

        q_4d, actual_heads = self._pad_q_for_fp8(q_4d)
        padded_heads = q_4d.shape[-2]

        topk = topk_indices.shape[-1]
        topk_tensor = torch.full(
            (num_decodes,), topk, dtype=torch.int32, device=q.device)
        dummy_bt = torch.empty(
            (num_decodes, 1), dtype=torch.int32, device=q.device)

        tile_sched_meta, _ = self.get_metadata(
            topk_tensor, decode_query_len * padded_heads,
            topk=topk, num_heads_q=padded_heads,
            num_heads_k=1, is_fp8_kvcache=True)

        o, _ = self.decode_op(
            q_4d, kv_cache.view(torch.uint8).unsqueeze(-2),
            dummy_bt, topk_tensor,
            head_dim_v=_MLA_HEAD_DIM_V,
            tile_scheduler_metadata=tile_sched_meta,
            softmax_scale=self.scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=topk_4d,
        )

        o = o.view(-1, padded_heads, o.shape[-1])
        if actual_heads < padded_heads:
            o = o[:, :actual_heads, :]
        return self._v_up_proj(o)

    def _forward_sparse_separate(self, q, kv_c_normed, k_pe, kv_b_proj,
                                 kv_cache, ctx, topk_indices,
                                 num_prefill_tokens, num_decode_tokens):
        """Separate prefill (BF16 workspace) and decode (FP8 kernel).

        Mirrors vLLM's ``_forward_fp8_kv_separate_prefill_decode``: ALL
        tokens flow through sparse attention (workspace-gather for prefill,
        direct FP8 paged decode for decode tokens). We must NOT fall back
        to the dense ``_forward_mha`` path here — that bypasses the DSA
        top-k indices entirely and produces dense full attention output.
        """
        N = q.shape[0]
        block_size = int(kv_cache.shape[1])

        # ``ctx.block_tables`` is only populated by the pure-prefill /
        # pure-decode set_context paths.  ``prepare_mixed_batch`` populates
        # ``prefill_block_tables`` and ``decode_block_tables`` separately
        # (vLLM keeps a single unified block_table covering all sequences).
        # Pick the right table(s) so we have something to feed to
        # ``convert_indices`` and to derive ``num_seqs_total``.
        if ctx.is_mixed:
            num_decode_seqs = (
                ctx.decode_block_tables.shape[0]
                if ctx.decode_block_tables is not None and num_decode_tokens > 0
                else 0
            )
            num_prefill_seqs = (
                ctx.prefill_block_tables.shape[0]
                if ctx.prefill_block_tables is not None
                and num_prefill_tokens > 0
                else 0
            )
            num_seqs_total = num_decode_seqs + num_prefill_seqs

            # Build a unified block_table for ``convert_indices``: rows
            # ``[0:num_decode_seqs]`` are decode requests, rows
            # ``[num_decode_seqs:]`` are prefill requests. For pure
            # prefill (no decode) this is just ``prefill_block_tables``.
            if num_decode_seqs > 0 and num_prefill_seqs > 0:
                # Pad to common max_blocks before concatenating.
                d_bt = ctx.decode_block_tables
                p_bt = ctx.prefill_block_tables
                max_b = max(d_bt.shape[1], p_bt.shape[1])
                if d_bt.shape[1] < max_b:
                    pad = torch.full(
                        (d_bt.shape[0], max_b - d_bt.shape[1]),
                        -1, dtype=d_bt.dtype, device=d_bt.device,
                    )
                    d_bt = torch.cat([d_bt, pad], dim=1)
                if p_bt.shape[1] < max_b:
                    pad = torch.full(
                        (p_bt.shape[0], max_b - p_bt.shape[1]),
                        -1, dtype=p_bt.dtype, device=p_bt.device,
                    )
                    p_bt = torch.cat([p_bt, pad], dim=1)
                unified_block_table = torch.cat([d_bt, p_bt], dim=0)
            elif num_prefill_seqs > 0:
                unified_block_table = ctx.prefill_block_tables
            elif num_decode_seqs > 0:
                unified_block_table = ctx.decode_block_tables
            else:
                # Nothing to do.
                return torch.zeros(
                    N, self.num_heads * self.v_head_dim,
                    dtype=q.dtype, device=q.device,
                )
        else:
            if ctx.block_tables is None:
                # Truly nothing to attend to — return zeros (matches an
                # empty MLA call).
                return torch.zeros(
                    N, self.num_heads * self.v_head_dim,
                    dtype=q.dtype, device=q.device,
                )
            unified_block_table = ctx.block_tables
            num_seqs_total = unified_block_table.shape[0]
            num_decode_seqs = getattr(
                ctx, 'num_decode_seqs',
                num_seqs_total if num_decode_tokens > 0 else 0,
            )
            num_prefill_seqs = num_seqs_total - num_decode_seqs

        req_ids = ctx.req_id_per_token
        if req_ids is None:
            # Per-token request id. For mixed batch:
            #   decode tokens ``[0:num_decode_tokens]`` -> request 0..num_dc_seqs-1
            #   prefill tokens ``[num_decode_tokens:]`` derived from prefill_cu_q
            # For pure prefill (single sequence, our diagnostic case) all
            # tokens belong to request 0; the previous implementation used
            # ``arange`` which over-indexed the block_table for any
            # single-sequence prefill > 1 token. Build req_ids correctly
            # from the cumulative seqlens metadata.
            req_ids = torch.zeros(N, dtype=torch.int32, device=q.device)
            if ctx.is_mixed:
                # Decode rows come first.
                for i in range(num_decode_seqs):
                    req_ids[i] = i
                pf_cu_q = ctx.prefill_cu_seqlens_q
                if pf_cu_q is not None:
                    for r in range(num_prefill_seqs):
                        qs = int(pf_cu_q[r].item()) + num_decode_tokens
                        qe = int(pf_cu_q[r + 1].item()) + num_decode_tokens
                        # Decode block_table sits at rows [0:num_decode_seqs];
                        # prefill rows are appended after, hence + offset.
                        req_ids[qs:qe] = num_decode_seqs + r
            else:
                cu_q = ctx.cu_seqlens_q
                if cu_q is not None:
                    for r in range(num_prefill_seqs if num_prefill_seqs > 0 else num_decode_seqs):
                        qs = int(cu_q[r].item())
                        qe = int(cu_q[r + 1].item())
                        req_ids[qs:qe] = r

        prefill_request_ids = None
        prefill_workspace_starts = None
        has_prefill = num_prefill_tokens > 0

        if has_prefill:
            if ctx.is_mixed:
                pf_bt = (
                    ctx.prefill_block_tables
                    if ctx.prefill_block_tables is not None
                    else unified_block_table[num_decode_seqs:]
                )
                pf_cu = ctx.prefill_cu_seqlens_k
                pf_seq_lens = pf_cu[1:] - pf_cu[:-1]
            else:
                pf_bt = unified_block_table
                pf_cu = ctx.cu_seqlens_k
                pf_seq_lens = pf_cu[1:] - pf_cu[:-1]

            prefill_request_ids = torch.full((N,), -1, dtype=torch.int32, device=q.device)
            prefill_workspace_starts = torch.zeros(num_prefill_seqs, dtype=torch.int32, device=q.device)

            if num_prefill_seqs > 1:
                prefill_workspace_starts[1:] = torch.cumsum(pf_seq_lens[:-1], dim=0).int()

            if ctx.is_mixed:
                pf_cu_q = ctx.prefill_cu_seqlens_q
                for req_idx in range(num_prefill_seqs):
                    qs = int(pf_cu_q[req_idx].item()) + num_decode_tokens
                    qe = int(pf_cu_q[req_idx + 1].item()) + num_decode_tokens
                    prefill_request_ids[qs:qe] = req_idx
            else:
                cu_q = ctx.cu_seqlens_q
                for req_idx in range(num_prefill_seqs):
                    qs = int(cu_q[req_idx].item())
                    qe = int(cu_q[req_idx + 1].item())
                    prefill_request_ids[qs:qe] = req_idx

        topk_global = self.convert_indices(
            topk_indices, unified_block_table, block_size,
            req_ids=req_ids,
            prefill_request_ids=prefill_request_ids,
            prefill_workspace_starts=prefill_workspace_starts,
        )

        # Absorb q into the 576-D latent space (mirrors what vLLM's
        # MLAAttention.forward_impl does *before* calling
        # ``forward_mqa``).  Both the BF16 sparse-prefill kernel and the
        # FP8 sparse-decode kernel expect ``q`` with head-dim
        # ``kv_lora_rank + qk_rope_head_dim`` (576 for V3.2), not the
        # un-absorbed 192-D layout produced by ``q_b_proj``.
        q = self._absorb_q_to_latent(q)  # [N, H, kv_lora_rank+rope]

        out = torch.empty(N, self.num_heads, self.kv_lora_rank,
                          dtype=q.dtype, device=q.device)

        if num_decode_tokens > 0:
            nd = num_decode_tokens
            q_dc = q[:nd]
            topk_dc = topk_global[:nd]
            num_decodes = num_decode_seqs

            q_dc_4d = q_dc.view(num_decodes, -1, self.num_heads, q.shape[-1])
            topk_dc_4d = topk_dc.view(num_decodes, -1, topk_dc.shape[-1])
            q_dc_4d, actual_heads = self._pad_q_for_fp8(q_dc_4d)
            padded_heads = q_dc_4d.shape[-2]
            decode_query_len = q_dc_4d.shape[1]

            topk = topk_dc.shape[-1]
            topk_tensor = torch.full(
                (num_decodes,), topk, dtype=torch.int32, device=q.device)
            dummy_bt = torch.empty(
                (num_decodes, 1), dtype=torch.int32, device=q.device)

            tile_sched_meta, _ = self.get_metadata(
                topk_tensor, decode_query_len * padded_heads,
                topk=topk, num_heads_q=padded_heads,
                num_heads_k=1, is_fp8_kvcache=True)

            o_dc, _ = self.decode_op(
                q_dc_4d, kv_cache.view(torch.uint8).unsqueeze(-2),
                dummy_bt, topk_tensor,
                head_dim_v=_MLA_HEAD_DIM_V,
                tile_scheduler_metadata=tile_sched_meta,
                softmax_scale=self.scale,
                # Sparse attention: the FlashMLA kernel asserts causal==False
                # whenever ``indices`` is set (the top-k already encodes the
                # causal span). ``decode_op`` (FlashMLADecode) defaults
                # causal=True, so pass it explicitly here — matching the
                # mixed-batch and sparse-decode call sites above and vLLM's
                # FlashMLASparseImpl (which never passes causal).
                causal=False,
                is_fp8_kvcache=True,
                indices=topk_dc_4d,
            )
            o_dc = o_dc.view(-1, padded_heads, o_dc.shape[-1])
            if actual_heads < padded_heads:
                o_dc = o_dc[:, :actual_heads, :]
            out[:nd] = o_dc

        if num_prefill_tokens > 0:
            np_ = num_prefill_tokens
            q_pf = q[num_decode_tokens:] if ctx.is_mixed else q
            topk_pf = topk_global[num_decode_tokens:] if ctx.is_mixed else topk_global

            total_seq_len = int(pf_seq_lens.sum().item())
            workspace = torch.empty(total_seq_len, _MLA_WORKSPACE_HEAD_SIZE,
                                    dtype=torch.bfloat16, device=q.device)
            self.gather_kvcache(
                kv_cache, pf_bt, pf_seq_lens,
                prefill_workspace_starts, num_prefill_seqs, workspace,
            )

            workspace_kv = workspace.view(-1, 1, _MLA_WORKSPACE_HEAD_SIZE)

            prefill_padding = self.prefill_padding
            actual_h = q_pf.shape[1]
            q_pf_3d = q_pf
            if actual_h % prefill_padding != 0:
                pad_h = prefill_padding
                q_padded = q_pf_3d.new_empty(q_pf_3d.shape[0], pad_h, q_pf_3d.shape[2])
                q_padded[:, :actual_h, :] = q_pf_3d
                q_pf_3d = q_padded

            topk_pf_3d = topk_pf.view(np_, 1, -1)
            pf_out = self.sparse_prefill_op(
                q_pf_3d, workspace_kv, topk_pf_3d, self.scale)

            if isinstance(pf_out, (tuple, list)):
                pf_out = pf_out[0]
            pf_out = pf_out[:, :actual_h, :]

            if ctx.is_mixed:
                out[num_decode_tokens:] = pf_out
            else:
                out[:] = pf_out

        return self._v_up_proj(out.view(N, self.num_heads, self.kv_lora_rank))

    def _forward_mixed(self, q, kv_c_normed, k_pe, kv_b_proj, kv_cache, ctx):
        """Mixed batch for dense (non-sparse) attention."""
        np_ = ctx.num_prefill_tokens
        nd = ctx.num_decode_tokens
        out = torch.empty(np_ + nd, self.num_heads * self.v_head_dim,
                          dtype=q.dtype, device=q.device)

        if np_ > 0:
            q_pf = q[:np_]
            kv_c_pf = kv_c_normed[:np_]
            k_pe_pf = k_pe[:np_]

            has_context = ctx.chunked_context is not None
            kv = kv_b_proj(kv_c_pf)
            kv = kv.view(np_, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = self._concat_k_nope_k_pe(k_nope, k_pe_pf)

            output_prefill = self._run_prefill_new_tokens(
                q_pf, k, v,
                cu_seqlens_q=ctx.prefill_cu_seqlens_q,
                cu_seqlens_k=ctx.prefill_cu_seqlens_q,
                max_seqlen_q=ctx.prefill_max_seqlen_q,
                max_seqlen_k=ctx.prefill_max_seqlen_q,
                return_softmax_lse=has_context,
            )

            if has_context:
                suffix_output, suffix_lse = output_prefill
                context_output, context_lse = self._compute_prefill_context(
                    q_pf, kv_cache, kv_b_proj, ctx)

                pf_result = torch.empty(np_, self.num_heads, self.v_head_dim,
                                        dtype=q.dtype, device=q.device)
                self.merge_states(
                    output=pf_result,
                    prefix_output=context_output,
                    prefix_lse=context_lse,
                    suffix_output=suffix_output[..., :self.v_head_dim],
                    suffix_lse=suffix_lse,
                )
                out[:np_] = pf_result.reshape(np_, self.num_heads * self.v_head_dim)
            else:
                pf_out = output_prefill
                if isinstance(pf_out, tuple):
                    pf_out = pf_out[0]
                out[:np_] = pf_out.reshape(np_, self.num_heads * self.v_head_dim)

        if nd > 0:
            q_dc = q[np_:]
            cache_seqlens = ctx.decode_context_lens
            block_table = ctx.decode_block_tables

            if not self.use_fp8_kv_cache:
                q_dc = self._absorb_q_to_latent(q_dc)
                q_dc = q_dc.unsqueeze(1)
                tile_sched_meta, _ = self.get_metadata(
                    cache_seqlens, self.num_heads, num_heads_k=1,
                )
                o, _ = self.decode_op(
                    q_dc,
                    kv_cache.unsqueeze(-2),
                    block_table,
                    cache_seqlens,
                    head_dim_v=_MLA_HEAD_DIM_V,
                    tile_scheduler_metadata=tile_sched_meta,
                    softmax_scale=self.scale,
                    causal=True,
                )
                o = o.reshape(-1, o.shape[-2], o.shape[-1])
                out[np_:] = self._v_up_proj(o)
            else:
                tile_sched_meta, num_splits = self.get_metadata_dense_fp8(
                    cache_seqlens, self.num_heads, num_heads_k=1,
                )
                o, _ = self.decode_op_fp8(
                    q_dc.unsqueeze(1), kv_cache.view(torch.uint8).unsqueeze(-2),
                    block_table, cache_seqlens,
                    head_dim_v=_MLA_HEAD_DIM_V,
                    tile_scheduler_metadata=tile_sched_meta,
                    num_splits=num_splits,
                    softmax_scale=self.scale,
                    causal=True,
                    descale_q=self._q_scale.reshape(1),
                    descale_k=self._k_scale.reshape(1),
                )
                o = o.reshape(-1, o.shape[-2], o.shape[-1])
                out[np_:] = self._v_up_proj(o)

        return out

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

from __future__ import annotations

import torch
import torch.nn as nn

from fastkernels.infra.tp import _tp_size


class Model(nn.Module):
    """Kimi MLA path matching vLLM's latent-attention formulation."""

    def __init__(self, config, quant_config: dict | None = None):
        super().__init__()
        tp = _tp_size()
        self.hidden_size = config.hidden_size
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.num_heads = config.num_attention_heads
        self.num_local_heads = self.num_heads // tp
        self.scaling = self.qk_head_dim ** -0.5

        assert self.q_lora_rank is None
        assert getattr(config, "mla_use_nope", True)

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
        )
        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
        )
        self.kv_a_layernorm = RMSNorm(
            self.kv_lora_rank,
            eps=config.rms_norm_eps,
        )
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
        )

        self.attn = MLAAttention(
            num_heads=self.num_local_heads,
            scale=self.scaling,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            is_sparse=False,
        )
        object.__setattr__(self.attn, "_kv_b_proj", self.kv_b_proj)

    def compute_absorbed_weights(self):
        """Compute absorbed MLA decode weights from ``kv_b_proj``."""
        weight = self.kv_b_proj.weight.data
        if hasattr(self.kv_b_proj, "use_fp8") and self.kv_b_proj.use_fp8:
            scale = self.kv_b_proj.weight_scale_inv.data
            weight = self._dequant_fp8_block(weight, scale)
        else:
            weight = weight.to(torch.bfloat16)

        weight = weight.T
        latent = self.kv_lora_rank
        heads = self.num_local_heads
        nope = self.qk_nope_head_dim
        value = self.v_head_dim
        weight = weight.view(latent, heads, nope + value)
        w_uk = weight[:, :, :nope]
        w_uv = weight[:, :, nope:]
        self.attn.W_UV = w_uv.permute(1, 0, 2).contiguous()
        self.attn.W_UK_T = w_uk.permute(1, 2, 0).contiguous()

    @staticmethod
    def _dequant_fp8_block(
        w_fp8: torch.Tensor,
        scale_inv: torch.Tensor,
        block_size: int = 128,
    ) -> torch.Tensor:
        import math

        n, k = w_fp8.shape
        sn = math.ceil(n / block_size)
        sk = math.ceil(k / block_size)
        scale = scale_inv[:sn, :sk]
        scale_expanded = scale.repeat_interleave(block_size, dim=0)[:n]
        scale_expanded = scale_expanded.repeat_interleave(block_size, dim=1)[:, :k]
        return (w_fp8.float() * scale_expanded).to(torch.bfloat16)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        state_manager=None,
    ) -> torch.Tensor:
        del positions, state_manager
        num_tokens = hidden_states.shape[0]

        q = self.q_proj(hidden_states)
        q = q.view(num_tokens, self.num_local_heads, self.qk_head_dim)

        kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_c, k_pe = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c = self.kv_a_layernorm(kv_c)
        k_pe = k_pe.unsqueeze(1)

        attn_output = self.attn(
            q,
            kv_c,
            k_pe,
            output_shape=(num_tokens, self.num_local_heads * self.v_head_dim),
        )
        return self.o_proj(attn_output)

    ```

ModelNew must use the same `__init__` arguments and `forward` signature as `Model`.

The kernel will be evaluated on these captured input shapes:

### KimiMLAAttention

| count | args |
|------:|------|
| 1834 | `hidden_states:bfloat16[64, 2304] positions:None` |
| 889 | `hidden_states:bfloat16[1, 2304] positions:None` |
| 784 | `hidden_states:bfloat16[16384, 2304] positions:None` |
| 287 | `hidden_states:bfloat16[26, 2304] positions:None` |
| 210 | `hidden_states:bfloat16[31, 2304] positions:None` |
| 119 | `hidden_states:bfloat16[30, 2304] positions:None` |
| 119 | `hidden_states:bfloat16[88, 2304] positions:None` |
| 105 | `hidden_states:bfloat16[29, 2304] positions:None` |

    
Optimize the architecture named Model with custom Triton operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Let's think step by step.
