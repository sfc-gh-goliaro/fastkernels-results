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
from typing import Optional
import deep_gemm as _dg
import functools
import json
import math
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

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

def _is_batch_invariant() -> bool:
    """vLLM's dynamic FP8 blockscale dispatch forces the DeepGEMM path (skips the
    FlashInfer swapAB kernel) for ALL M under batch-invariant determinism mode
    (``VLLM_BATCH_INVARIANT=1``) — see the early-out in
    ``scaled_mm/flashinfer.py`` and ``grouped_topk._is_batch_invariant``. Mirror
    it so fastkernels matches vLLM in that mode."""
    return os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1"

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

_FP8_BLOCK = 128

def _scale_shape(out_dim: int, in_dim: int) -> tuple[int, int]:
    return (math.ceil(out_dim / _FP8_BLOCK), math.ceil(in_dim / _FP8_BLOCK))

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

def _deep_gemm_alignment() -> int:
    return _dg.get_mk_alignment_for_contiguous_layout()

def _valid_deep_gemm_shape(M: int, N: int, K: int) -> bool:
    align = _deep_gemm_alignment()
    return align <= M and N % align == 0 and K % align == 0

def _is_deep_gemm_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] >= 9

def _valid_deep_gemm(hidden_states: torch.Tensor, w1: torch.Tensor,
                     w2: torch.Tensor) -> bool:
    if not _is_deep_gemm_supported():
        return False
    M = hidden_states.size(0)
    _, K, N = w2.size()
    if not _valid_deep_gemm_shape(M, N, K):
        return False
    if N <= 512:
        return False
    if w1.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        return False
    if not (hidden_states.is_contiguous() and w1.is_contiguous() and w2.is_contiguous()):
        return False
    return True

_C = lazy_op("moe_align", "moe_align.cu")

class MoeAlign(nn.Module):
    """MoE token-to-expert alignment using sgl_kernel.

    Pre-allocates output buffers for reuse and CUDA graph compatibility.
    """

    def __init__(self):
        super().__init__()
        self._sorted_token_ids = None
        self._expert_ids = None
        self._num_tokens_post_padded = None
        self._cumsum_buffer = None

    def _ensure_buffers(self, max_padded, max_blocks, num_experts, device):
        if (self._sorted_token_ids is None
                or self._sorted_token_ids.size(0) < max_padded):
            self._sorted_token_ids = torch.empty(
                max_padded, dtype=torch.int32, device=device,
            )
        if (self._expert_ids is None
                or self._expert_ids.size(0) < max_blocks):
            self._expert_ids = torch.empty(
                max_blocks, dtype=torch.int32, device=device,
            )
        if (self._num_tokens_post_padded is None
                or self._num_tokens_post_padded.device != device):
            self._num_tokens_post_padded = torch.zeros(
                1, dtype=torch.int32, device=device,
            )
        if (self._cumsum_buffer is None
                or self._cumsum_buffer.size(0) < num_experts + 1):
            self._cumsum_buffer = torch.zeros(
                num_experts + 1, dtype=torch.int32, device=device,
            )

    def _naive_forward(
        self,
        topk_ids: torch.Tensor,
        block_size: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """Fast path: skip full alignment when tokens * top_k is very small."""
        numel = topk_ids.numel()
        max_num_tokens_padded = numel * block_size
        expert_ids = topk_ids.view(-1).to(torch.int32)
        # Allocate fresh each call (matching vllm_fused_experts) rather than
        # reusing a persistent buffer.  A persisted scalar gets created as an
        # inference tensor during the inference_mode forward, and a later
        # in-place ``fill_`` from Inductor's autotuning/benchmark pass (which
        # runs under ``no_grad``, not ``inference_mode``) would raise
        # "Inplace update to inference tensor outside InferenceMode".
        num_tokens_post_padded = torch.full(
            (1,), max_num_tokens_padded, dtype=torch.int32,
            device=topk_ids.device,
        )
        return None, expert_ids, num_tokens_post_padded

    def forward(
        self,
        topk_ids: torch.Tensor,
        block_size: int,
        num_experts: int,
        naive: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        if naive:
            return self._naive_forward(topk_ids, block_size)

        numel = topk_ids.numel()
        if numel < num_experts:
            max_padded = numel * block_size
        else:
            max_padded = numel + num_experts * (block_size - 1)
        max_blocks = triton.cdiv(max_padded, block_size)

        self._ensure_buffers(max_padded, max_blocks, num_experts, topk_ids.device)

        sorted_token_ids = self._sorted_token_ids[:max_padded]
        expert_ids = self._expert_ids[:max_blocks]

        _C.moe_align_block_size(
            topk_ids.view(-1).contiguous(),
            num_experts, block_size,
            sorted_token_ids, expert_ids,
            self._num_tokens_post_padded,
            self._cumsum_buffer[:num_experts + 1],
            True,
        )

        return sorted_token_ids, expert_ids, self._num_tokens_post_padded

def _compute_aligned_M(M: int, num_topk: int, local_num_experts: int,
                        alignment: int) -> int:
    """Compute aligned total rows for DeepGEMM."""
    M_sum = (M * num_topk) + local_num_experts * (alignment - 1)
    remainder = M_sum % alignment
    if remainder != 0:
        M_sum += alignment - remainder
    return M_sum

_FP8_GROUP_SIZE = 128

def _deepgemm_permute(
    hidden_states: torch.Tensor,
    a_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    local_num_experts: int,
    alignment: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Permute tokens by expert assignment for DeepGEMM contiguous layout.

    Uses vectorized PyTorch ops (scatter_add, argsort) to avoid Python loops.

    Returns:
        (a_perm, a_scale_perm, expert_ids, inv_perm)
    """
    M, K = hidden_states.size()
    top_k = topk_ids.size(1)
    device = hidden_states.device

    M_sum = _compute_aligned_M(M, top_k, local_num_experts, alignment)
    scale_cols = K // _FP8_GROUP_SIZE

    flat_ids = topk_ids.view(-1).to(torch.int64)
    num_tokens_total = flat_ids.size(0)

    expert_num_tokens = torch.zeros(local_num_experts, dtype=torch.int64, device=device)
    expert_num_tokens.scatter_add_(0, flat_ids,
                                   torch.ones(num_tokens_total, dtype=torch.int64, device=device))

    aligned_counts = ((expert_num_tokens + alignment - 1) // alignment) * alignment
    expert_offsets = torch.zeros(local_num_experts + 1, dtype=torch.int64, device=device)
    torch.cumsum(aligned_counts, dim=0, out=expert_offsets[1:])

    # Build expert_ids without host-device sync (.item()) so this is safe
    # inside CUDA graph capture.  For each position in [0, M_sum), determine
    # which expert's aligned block it falls into via searchsorted, then check
    # whether it's within the actual (non-padding) token count.
    pos_idx = torch.arange(M_sum, device=device, dtype=torch.int64)
    # searchsorted(offsets, pos, right=True) - 1 gives the expert whose block
    # contains `pos`.  expert_offsets has E+1 entries (0-based cumsum).
    expert_for_pos = torch.searchsorted(expert_offsets, pos_idx, right=True) - 1
    expert_for_pos = expert_for_pos.clamp_(0, local_num_experts - 1)
    local_pos = pos_idx - expert_offsets[expert_for_pos]
    valid = local_pos < expert_num_tokens[expert_for_pos]
    # Use torch.where (element-wise, fixed output size) instead of boolean
    # indexing which produces data-dependent shapes and breaks CUDA graphs.
    expert_ids = torch.where(valid, expert_for_pos.to(torch.int32),
                             torch.tensor(-1, dtype=torch.int32, device=device))

    sorted_order = torch.argsort(flat_ids, stable=True)

    # Compute within-expert indices using only GPU ops.
    sorted_experts = flat_ids[sorted_order]
    rank_in_sorted = torch.arange(num_tokens_total, device=device, dtype=torch.int64)
    # For each expert, find the first position in sorted order.
    expert_first = torch.full((local_num_experts,), num_tokens_total,
                              dtype=torch.int64, device=device)
    expert_first.scatter_reduce_(0, sorted_experts,
                                 rank_in_sorted, reduce="amin",
                                 include_self=False)
    within_expert_idx = torch.zeros(num_tokens_total, dtype=torch.int64, device=device)
    within_expert_idx[sorted_order] = rank_in_sorted - expert_first[sorted_experts]

    dest_positions = expert_offsets[flat_ids] + within_expert_idx

    a_perm = torch.zeros(M_sum, K, dtype=hidden_states.dtype, device=device)
    a_scale_perm = torch.zeros(M_sum, scale_cols, dtype=torch.float32, device=device)

    token_indices = torch.arange(M, device=device).unsqueeze(1).expand(M, top_k).reshape(-1)

    a_perm[dest_positions] = hidden_states[token_indices]
    a_scale_perm[dest_positions] = a_scale[token_indices]

    inv_perm = dest_positions.view(M, top_k).to(torch.int32)

    return a_perm, a_scale_perm, expert_ids, inv_perm

class PerTokenGroupQuantFp8(nn.Module):
    """In-place per-token-group FP8 quantization (single Triton/CUDA kernel).

    Public ``nn.Module`` wrapper around the registered
    ``torch.ops.fastkernels_fp8.per_token_group_quant_fp8`` custom op.  L2
    callers that need activation quantization outside of ``Fp8Linear``
    (e.g. ``DeepSeekMoE``, ``SparseAttnIndexer``) should use this module
    instead of importing the private ``_per_token_group_quant_fp8``.
    """

    def forward(self, x: torch.Tensor, out_fp8: torch.Tensor,
                out_scale: torch.Tensor) -> None:
        torch.ops.fastkernels_fp8.per_token_group_quant_fp8(
            x.contiguous() if not x.is_contiguous() else x,
            out_fp8, out_scale,
        )

def _deepgemm_unpermute_and_reduce(
    mm2_out: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    inv_perm: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Unpermute DeepGEMM output and reduce across top-k experts.

    Uses vectorized gather + weighted sum to avoid Python loops.
    """
    M, K = output.size()
    top_k = topk_ids.size(1)

    flat_positions = inv_perm.to(torch.int64).view(-1)
    gathered = mm2_out[flat_positions].view(M, top_k, K)
    weights = topk_weights.unsqueeze(-1)
    output.copy_((gathered.to(output.dtype) * weights).sum(dim=1))

class _SharedBuf:
    """Mutable container so all FusedExperts layers share one set of scratch
    buffers. Layers execute sequentially so reuse is safe."""
    __slots__ = ("cache13", "a_fp8_1", "a_scale_1", "a_fp8_2", "a_scale_2",
                 "dg_ws1", "dg_ws2")
    def __init__(self):
        self.cache13 = None
        self.a_fp8_1 = None
        self.a_scale_1 = None
        self.a_fp8_2 = None
        self.a_scale_2 = None
        self.dg_ws1 = None
        self.dg_ws2 = None

_SHARED_BUF = _SharedBuf()

_DEFAULT_CONFIG_HEURISTIC = {
    "small": {
        "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 5,
    },
    "medium": {
        "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 64, "num_warps": 4, "num_stages": 3,
    },
    "large": {
        "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 4,
    },
}

def _get_default_config(M: int, E: int = 0, N: int = 0,
                        block_shape: list[int] | None = None) -> dict:
    if block_shape is not None and all(block_shape):
        return {
            "BLOCK_SIZE_M": 16 if M <= 64 else 64,
            "BLOCK_SIZE_N": block_shape[0],
            "BLOCK_SIZE_K": block_shape[1],
            "GROUP_SIZE_M": 1 if M <= 16 else 32,
            "num_warps": 4,
            "num_stages": 3,
        }
    if M <= 4:
        return dict(_DEFAULT_CONFIG_HEURISTIC["small"])
    if M <= 64:
        return dict(_DEFAULT_CONFIG_HEURISTIC["medium"])
    return dict(_DEFAULT_CONFIG_HEURISTIC["large"])

def _get_config_file_name(E: int, N: int, dtype: str | None,
                          block_shape: list[int] | None = None) -> str:
    """Override JSON filename (``FASTKERNELS_TUNED_CONFIG_FOLDER`` / vLLM checkout)."""
    device_name = torch.cuda.get_device_name().replace(" ", "_")
    if "H200" in device_name.split("_"):
        device_name = "NVIDIA_H200"
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    block_shape_selector = (
        "" if not block_shape or not all(block_shape) else f",block_shape={block_shape}"
    ).replace(" ", "")
    return f"E={E},N={N},device_name={device_name}{dtype_selector}{block_shape_selector}.json"

def _device_name() -> str:
    device_name = torch.cuda.get_device_name().replace(" ", "_")
    if "H200" in device_name.split("_"):
        device_name = "NVIDIA_H200"
    return device_name

MOE_TRITON_CONFIGS: dict[
    tuple[int, int, str, str | None, tuple[int, int] | None],
    dict[int, dict[str, int]],
] = {
    (16, 1024, 'NVIDIA_B200', None, None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 5},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        1536: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 32, 'num_warps': 8, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 32, 'num_warps': 8, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
    },
    (16, 1024, 'NVIDIA_B200', 'fp8_w8a8', None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 5},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        512: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1536: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        4096: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
    },
    (16, 4096, 'NVIDIA_B200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 5},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 5},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        96: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
        128: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
        256: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 8, 'num_stages': 4},
        512: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 8, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 8, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 8, 'num_stages': 4},
    },
    (16, 14336, 'NVIDIA_B200', None, None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        96: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 4},
        128: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 4},
        200: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 4},
        256: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 4},
        512: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 4},
    },
    (20, 2560, 'NVIDIA_B200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 5},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        48: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        64: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        128: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        256: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        512: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
    },
    (20, 2560, 'NVIDIA_GB200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 5},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        48: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        64: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        96: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        128: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        256: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        512: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
    },
    (32, 1408, 'NVIDIA_B200', None, None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        128: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        512: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        2048: {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        3072: {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        4096: {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
    },
    (32, 2048, 'NVIDIA_B200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 5},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        96: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        128: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        256: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 8, 'num_stages': 5},
        512: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 8, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 8, 'num_stages': 3},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 8, 'num_stages': 3},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
    },
    (40, 1536, 'NVIDIA_B200', 'fp8_w8a8', None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 5},
        1536: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 3},
        3072: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
    },
    (40, 2560, 'NVIDIA_B200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        512: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
    },
    (40, 2560, 'NVIDIA_GB200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        128: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        512: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
    },
    (64, 512, 'NVIDIA_B200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
    },
    (64, 1408, 'NVIDIA_B200', None, None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 5},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        256: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 5},
        512: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        2048: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        4096: {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
    },
    (128, 384, 'NVIDIA_B200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        512: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
    },
    (128, 384, 'NVIDIA_GB200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        512: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
    },
    (128, 512, 'NVIDIA_B200', None, None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 5},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 2},
        2048: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        3072: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        4096: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 3},
    },
    (128, 512, 'NVIDIA_B200', 'fp8_w8a8', None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 2},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 2},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 2},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 5},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        3072: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
    },
    (128, 512, 'NVIDIA_B200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
    },
    (128, 512, 'NVIDIA_GB200', 'fp8_w8a8', None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 2},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 2},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 2},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 5},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        3072: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
    },
    (128, 704, 'NVIDIA_B200', 'fp8_w8a8', None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 5},
        4096: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
    },
    (128, 768, 'NVIDIA_B200', None, None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 2},
        512: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        1536: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        4096: {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
    },
    (128, 768, 'NVIDIA_B200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
    },
    (128, 768, 'NVIDIA_GB200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 8, 'num_stages': 3},
    },
    (128, 1856, 'NVIDIA_B200', None, None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 5},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        320: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2},
        768: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 5},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        1536: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
    },
    (160, 384, 'NVIDIA_B200', 'fp8_w8a8', None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 5},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 5},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 5},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 5},
        1536: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 5},
        8192: {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        16384: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
    },
    (160, 640, 'NVIDIA_B200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
    },
    (160, 640, 'NVIDIA_GB200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
    },
    (256, 256, 'NVIDIA_B200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 8, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 8, 'num_stages': 4},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 5},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1536: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        2048: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        3072: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        4096: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
    },
    (256, 512, 'NVIDIA_B200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
    },
    (384, 128, 'NVIDIA_B200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 2},
        1024: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1536: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
    },
    (384, 128, 'NVIDIA_GB200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1536: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
    },
    (384, 256, 'NVIDIA_B200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3},
    },
    (384, 256, 'NVIDIA_GB200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1536: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
    },
    (512, 64, 'NVIDIA_B200', None, None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1536: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        2048: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        3072: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4096: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 2},
    },
    (512, 128, 'NVIDIA_B200', None, None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1536: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        4096: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 2},
    },
    (512, 128, 'NVIDIA_B200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 2},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1536: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
    },
    (512, 128, 'NVIDIA_GB200', 'fp8_w8a8', None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        2: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        1536: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        4096: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
    },
    (512, 256, 'NVIDIA_B200', None, None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 2},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 2},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1536: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 64, 'num_warps': 8, 'num_stages': 3},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 3},
        4096: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 2},
    },
    # vLLM fused_moe/configs/E=512,N=256,device_name=NVIDIA_H200.json
    # (Qwen3-Next TP=2: 512 experts, intermediate 256 per rank). Without this
    # the Hopper path fell through to the legacy heuristic.
    (512, 256, 'NVIDIA_H200', None, None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 2},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 2},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 2},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 2},
        1024: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 2},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4096: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
    },
    (512, 256, 'NVIDIA_B200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 2},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        1024: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        1536: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
    },
    (512, 256, 'NVIDIA_GB200', 'fp8_w8a8', None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 5},
        4096: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
    },
    (512, 512, 'NVIDIA_B200', None, None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 32, 'num_warps': 8, 'num_stages': 3},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3},
        4096: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 4},
    },
    (512, 512, 'NVIDIA_B200', 'fp8_w8a8', (128, 128)): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 3},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        4096: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
    },
    (512, 512, 'NVIDIA_GB200', 'fp8_w8a8', None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3},
        1024: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 4},
        1536: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 5},
        2048: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 8, 'num_stages': 5},
        3072: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 32, 'num_warps': 8, 'num_stages': 5},
        4096: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 4},
    },
    (512, 672, 'NVIDIA_B200', None, None): {
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 5},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 5},
        512: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 5},
        1024: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 2},
    },
    (512, 1344, 'NVIDIA_B200', None, None): {
        1: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'num_warps': 4, 'num_stages': 2},
        2: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        4: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        8: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
        16: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        24: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        32: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        48: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        64: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 4},
        96: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 4},
        128: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        256: {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 64, 'num_warps': 4, 'num_stages': 3},
        512: {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 2},
        768: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 5},
        1024: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 5},
        1536: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 4},
    },
}

def _get_moe_configs(E: int, N: int, dtype: str | None,
                     block_n: int | None = None,
                     block_k: int | None = None) -> dict[int, dict] | None:
    block_shape = [block_n, block_k] if block_n and block_k else None
    json_file_name = _get_config_file_name(E, N, dtype, block_shape)

    # User / adjacent-vLLM JSON overrides still supported.
    config_file_paths: list[str] = []
    user_folder = os.environ.get("FASTKERNELS_TUNED_CONFIG_FOLDER")
    if user_folder is not None:
        config_file_paths.append(os.path.join(user_folder, json_file_name))

    vllm_configs_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "..",
        "vllm_repo", "vllm", "vllm", "model_executor", "layers",
        "fused_moe", "configs",
    )
    if os.path.isdir(vllm_configs_dir):
        config_file_paths.append(os.path.join(vllm_configs_dir, json_file_name))

    for config_file_path in config_file_paths:
        if os.path.exists(config_file_path):
            with open(config_file_path) as f:
                tuned_config = json.load(f)
                tuned_config.pop("triton_version", None)
                return {int(key): val for key, val in tuned_config.items() if str(key).isdigit()}

    bs_key: tuple[int, int] | None = None
    if block_shape is not None and all(block_shape):
        bs_key = (int(block_shape[0]), int(block_shape[1]))
    bundled = MOE_TRITON_CONFIGS.get((E, N, _device_name(), dtype, bs_key))
    if bundled is not None:
        return dict(bundled)
    return None


    return None

def _get_vllm_default_config(M: int, E: int = 0, dtype: str | None = None) -> dict:
    """vLLM-style BF16/FP16 MoE defaults.

    Gemma4's BF16 top-8 experts are much closer to vLLM's generic MoE path
    than to the older fastkernels heuristic, especially in decode where tokens are
    spread thinly across 128 experts.
    """
    if M <= 32:
        block_m = 16
    elif M <= 96:
        block_m = 32
    elif M <= 1024:
        block_m = 64
    else:
        block_m = 128

    block_n = 64 if M <= 64 else 128
    block_k = 128 if dtype == "fp8_w8a8" or M <= 64 else 64
    tokens_per_expert = M // max(E, 1)
    group_m = 16 if tokens_per_expert > 128 else 1
    num_warps = 4 if M <= 1024 else 8
    num_stages = 4 if M <= 32 else 3

    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": group_m,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }

def get_triton_config(M: int, w1_shape: tuple[int, ...], w2_shape: tuple[int, ...],
                      top_k: int, use_fp8: bool,
                      block_shape: list[int] | None = None,
                      default_style: str = "legacy") -> dict:
    """Select best Triton kernel config, preferring JSON tuning files."""
    E, _, N = w2_shape
    dtype = "fp8_w8a8" if use_fp8 else None
    block_n = block_shape[0] if block_shape else 0
    block_k = block_shape[1] if block_shape else 0

    configs = _get_moe_configs(E, N, dtype, block_n, block_k)
    if configs:
        config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
        return dict(config)

    if default_style == "vllm":
        return _get_vllm_default_config(M, E, dtype)
    if default_style != "legacy":
        raise ValueError(f"Unknown MoE config style: {default_style}")
    return _get_default_config(M, E, N, block_shape)

def _fused_moe_kernel(
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N, K, EM,
    num_valid_tokens,
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_asm, stride_ask,
    stride_bse, stride_bsk, stride_bsn,
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    NAIVE_BLOCK_ASSIGNMENT: tl.constexpr = False,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)

    if NAIVE_BLOCK_ASSIGNMENT:
        offs_token = tl.where(offs_m == 0, pid_m, num_valid_tokens)
    else:
        offs_token_id = pid_m * BLOCK_SIZE_M + offs_m
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (off_expert * stride_be + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    if use_fp8_w8a8:
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = (
                b_scale_ptr + off_expert * stride_bse + offs_bsn * stride_bsn
            )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask = (k * BLOCK_SIZE_K + offs_k) < K
        a = tl.load(a_ptrs, mask=token_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)

        if use_fp8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_SIZE_K
                offs_ks = k_start // group_k
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
                accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            else:
                accumulator = tl.dot(a, b, acc=accumulator)
        else:
            accumulator = tl.dot(a.to(compute_type), b.to(compute_type), accumulator)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if use_fp8_w8a8 and not (group_k > 0 and group_n > 0):
        a_scale = tl.load(a_scale_ptr)
        b_scale = tl.load(b_scale_ptr + off_expert)
        accumulator = accumulator * a_scale * b_scale

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_token[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

class MoeGroupedGemm(nn.Module):
    @staticmethod
    def get_config(M: int, N: int = 0, E: int = 0,
                   use_fp8: bool = False,
                   block_shape: list[int] | None = None) -> dict:
        """Select best kernel config based on batch size M and output dim N."""
        if E > 0:
            w2_shape = (E, 0, N // 2 if N > 0 else 0)
            w1_shape = (E, N, 0)
            return get_triton_config(M, w1_shape, w2_shape, 1, use_fp8, block_shape)
        return _get_default_config(M, E, N, block_shape)

    def forward(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        topk_weights: torch.Tensor | None,
        sorted_token_ids: torch.Tensor | None,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor,
        mul_routed_weight: bool,
        top_k: int,
        config: dict | None = None,
        a_scale: torch.Tensor | None = None,
        b_scale: torch.Tensor | None = None,
        use_fp8_w8a8: bool = False,
        block_shape: list[int] | None = None,
    ):
        if config is None:
            config = _get_default_config(A.size(0), N=B.size(1))
        else:
            config = config.copy()

        naive = sorted_token_ids is None
        if naive:
            EM = expert_ids.numel() * config["BLOCK_SIZE_M"]
        else:
            EM = sorted_token_ids.size(0)
            if A.size(0) < config["BLOCK_SIZE_M"]:
                EM = min(EM, A.size(0) * top_k * config["BLOCK_SIZE_M"])

        grid = (
            triton.cdiv(EM, config["BLOCK_SIZE_M"]) * triton.cdiv(B.size(1), config["BLOCK_SIZE_N"]),
        )

        if use_fp8_w8a8:
            compute_type = tl.bfloat16
        elif A.dtype == torch.bfloat16:
            compute_type = tl.bfloat16
        elif A.dtype == torch.float16:
            compute_type = tl.float16
        else:
            compute_type = tl.float32

        if use_fp8_w8a8 and block_shape is not None:
            group_n, group_k = block_shape[0], block_shape[1]
            config["BLOCK_SIZE_K"] = min(config["BLOCK_SIZE_K"], min(group_n, group_k))
        else:
            group_n, group_k = 0, 0

        launch_kwargs = {}
        if "num_warps" in config:
            launch_kwargs["num_warps"] = config["num_warps"]
        if "num_stages" in config:
            launch_kwargs["num_stages"] = config["num_stages"]

        sorted_ids_ptr = sorted_token_ids if sorted_token_ids is not None else A
        a_scale_ptr = a_scale if a_scale is not None else A
        b_scale_ptr = b_scale if b_scale is not None else B

        _fused_moe_kernel[grid](
            A, B, C,
            a_scale_ptr, b_scale_ptr,
            topk_weights,
            sorted_ids_ptr,
            expert_ids,
            num_tokens_post_padded,
            B.size(1), B.size(2), EM,
            A.size(0) * top_k,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(2), B.stride(1),
            C.stride(0), C.stride(1),
            a_scale.stride(0) if a_scale is not None and a_scale.ndim >= 2 else 0,
            a_scale.stride(1) if a_scale is not None and a_scale.ndim >= 2 else 0,
            b_scale.stride(0) if b_scale is not None and b_scale.ndim >= 2 else 0,
            b_scale.stride(2) if b_scale is not None and b_scale.ndim == 3 else 0,
            b_scale.stride(1) if b_scale is not None and b_scale.ndim >= 2 else 0,
            group_n=group_n,
            group_k=group_k,
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
            GROUP_SIZE_M=config["GROUP_SIZE_M"],
            NAIVE_BLOCK_ASSIGNMENT=naive,
            **launch_kwargs,
        )

_C = lazy_op("silu_and_mul", "silu_and_mul.cu")

class SiluAndMul(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward_native(x: torch.Tensor) -> torch.Tensor:
        """Pure PyTorch implementation — visible to Inductor for fusion."""
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    @staticmethod
    def forward_cuda(x: torch.Tensor) -> torch.Tensor:
        d = x.size(-1) // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        _C.silu_and_mul(out, x)
        return out

    def forward(self, x):
        if torch.compiler.is_compiling():
            return self.forward_native(x)
        return self.forward_cuda(x)

_FP8_INFO = torch.finfo(torch.float8_e4m3fn)

_GROUP_SIZE = 128

def _silu_mul_per_token_group_quant_fp8_colmajor(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    M,
    N,
    y_s_col_stride: tl.int64,
    eps,
    fp8_min,
    fp8_max,
    use_ue8m0: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    N_2 = N // 2

    m_offset = pid_m * BLOCK_M
    n_offset = pid_n * BLOCK_N
    if m_offset >= M:
        return

    offs_n = tl.arange(0, BLOCK_N).to(tl.int64)
    offs_m = tl.arange(0, BLOCK_M).to(tl.int64)

    base_y_ptr = y_ptr + m_offset * N + n_offset
    act_in_ptrs = base_y_ptr + offs_m[:, None] * N + offs_n[None, :]

    act_in = tl.load(act_in_ptrs)
    mul_in = tl.load(act_in_ptrs + N_2)

    act_in = act_in.to(tl.float32)
    one_f32 = tl.cast(1, tl.float32)
    silu_out = (act_in / (one_f32 + tl.exp(-act_in))).to(y_ptr.dtype.element_ty)
    y = (silu_out * mul_in).to(tl.float32)

    _absmax = tl.maximum(tl.max(tl.abs(y), axis=1), eps)
    # Multiply-by-reciprocal (not division) to match vLLM's
    # ``_silu_mul_per_token_group_quant_fp8_colmajor`` (fp8_utils.py:408):
    # GPU fast-division for a constexpr divisor introduces a 1-ULP error that
    # flips FP8 quantization at representable-value boundaries.
    scale_raw = _absmax * (1.0 / fp8_max)
    y_s = tl.math.exp2(tl.ceil(tl.log2(scale_raw))) if use_ue8m0 else scale_raw
    y_s = tl.reshape(y_s, (BLOCK_M, 1))
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    base_y_q_ptr = y_q_ptr + m_offset * N_2 + n_offset
    y_q_ptrs = base_y_q_ptr + offs_m[:, None] * N_2 + offs_n[None, :]
    tl.store(y_q_ptrs, y_q)

    group_id = n_offset // GROUP_SIZE
    base_y_s_ptr = y_s_ptr + group_id * y_s_col_stride + m_offset
    y_s_ptrs = base_y_s_ptr + offs_m
    y_s = tl.reshape(y_s, (BLOCK_M,))
    tl.store(y_s_ptrs, y_s)

class SiluMulQuantFp8(nn.Module):
    """Fused SiLU-mul + per-token-group FP8 quantization (colmajor scales).

    Stateless wrapper around the Triton kernel
    :func:`_silu_mul_per_token_group_quant_fp8_colmajor`.  Mirrors vLLM's
    ``silu_mul_per_token_group_quant_fp8_colmajor`` exactly.
    """

    def forward(
        self,
        input: torch.Tensor,
        output: torch.Tensor | None = None,
        use_ue8m0: bool = True,
        eps: float = 1e-10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused SiLU-mul + per-token-group FP8 quantization.

        Args:
            input: [M, N] where N = 2 * intermediate_size (gate/up concatenated)
            output: Optional pre-allocated [M, N//2] FP8 output buffer
            use_ue8m0: Use power-of-two (UE8M0) scales for DeepGEMM
            eps: Minimum absmax to avoid division by zero

        Returns:
            (output_fp8, output_scales) where output_fp8 is [M, N//2] in
            float8_e4m3fn and output_scales is [M, (N//2)//128] in float32
            (column-major layout)
        """
        assert input.ndim == 2
        M, N = input.size()
        N_2 = N // 2

        assert M % _GROUP_SIZE == 0, f"M={M} must be divisible by {_GROUP_SIZE}"
        assert N_2 % _GROUP_SIZE == 0, f"N//2={N_2} must be divisible by {_GROUP_SIZE}"

        if output is None:
            output = torch.empty(
                (M, N_2), dtype=torch.float8_e4m3fn, device=input.device,
            )

        output_scales = torch.empty(
            (N_2 // _GROUP_SIZE, M), dtype=torch.float32, device=input.device,
        ).transpose(0, 1)

        BLOCK_M = 8
        BLOCK_N = _GROUP_SIZE
        assert M % BLOCK_M == 0
        assert N_2 % BLOCK_N == 0

        fp8_min = _FP8_INFO.min
        fp8_max = _FP8_INFO.max

        grid = (M // BLOCK_M, N_2 // BLOCK_N)

        _silu_mul_per_token_group_quant_fp8_colmajor[grid](
            input, output, output_scales,
            M, N,
            output_scales.stride(-1),
            eps,
            fp8_min, fp8_max,
            use_ue8m0,
            _GROUP_SIZE, BLOCK_M, BLOCK_N,
        )

        return output, output_scales

_C = lazy_op("gelu_and_mul", "gelu_and_mul.cu")

class GeluAndMul(nn.Module):
    """Apply GELU to the gate half and multiply by the up half."""

    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate
        if approximate not in ("none", "tanh"):
            raise ValueError(f"Unsupported GELU approximation: {approximate}")
        self.op = (
            _C.gelu_tanh_and_mul
            if approximate == "tanh"
            else _C.gelu_and_mul
        )

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.gelu(x[..., :d], approximate=self.approximate) * x[..., d:]

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
        self.op(out, x)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.compiler.is_compiling():
            return self.forward_native(x)
        return self.forward_cuda(x)

SPARSITY_FACTOR = 4

def m_grouped_fp8_gemm_nt_contiguous(a_and_scale, b_and_scale, output, expert_ids):
    """Wrapper for deep_gemm.m_grouped_fp8_gemm_nt_contiguous.

    Args:
        a_and_scale: tuple of (a_fp8, a_scale)
        b_and_scale: tuple of (b_fp8, b_scale)
        output: output buffer
        expert_ids: per-row expert assignment (int32, -1 = skip)
    """
    # Match the SF format to the arch: B200 (e8m0) needs the UE8M0 cast, Hopper
    # keeps float32. Hardcoding True asserts on B200 ("Unsupported architecture
    # or scaling factor types"). Mirrors ``fp8_linear._fp8_gemm_nt_impl``.
    _dg.m_grouped_fp8_gemm_nt_contiguous(
        a_and_scale, b_and_scale, output, expert_ids,
        disable_ue8m0_cast=not _is_deep_gemm_e8m0_used(),
    )

_C = lazy_op("moe_sum", "moe_sum.cu")

class MoeSum(nn.Module):
    """Fused top-k reduction for MoE outputs using sgl_kernel."""

    def __init__(self):
        super().__init__()
        self._output = None

    def forward(
        self,
        input: torch.Tensor,
        topk: int,
    ) -> torch.Tensor:
        """Sum over the topk dimension.

        Args:
            input: [M * topk, D] tensor
            topk: number of experts per token

        Returns:
            output: [M, D] tensor
        """
        total = input.size(0)
        M = total // topk
        D = input.size(1)

        if self._output is None or self._output.size(0) < M or self._output.size(1) < D:
            self._output = torch.empty(M, D, device=input.device, dtype=input.dtype)
        output = self._output[:M, :D]

        _C.moe_sum(input.view(M, topk, D), output)

        return output

class FusedExperts(nn.Module):
    """Fused MoE experts: two grouped GEMMs with SiLU-mul in between.

    On Hopper+ GPUs with DeepGEMM available and valid shapes:
      permute -> DeepGEMM GEMM1 -> fused SiLU+mul+FP8 quant -> DeepGEMM GEMM2 -> unpermute
    Otherwise (Triton fallback):
      MoeAlign -> Triton grouped GEMM1 -> SiLU+mul -> FP8 quant -> Triton grouped GEMM2 -> MoeSum
    """

    def __init__(self, activation: str = "silu", config_style: str = "legacy"):
        super().__init__()
        if activation not in ("silu", "gelu_tanh"):
            raise ValueError(f"Unsupported MoE activation: {activation}")
        if config_style not in ("legacy", "vllm"):
            raise ValueError(f"Unsupported MoE config style: {config_style}")
        self.activation = activation
        self.config_style = config_style
        self.moe_align = MoeAlign()
        self.moe_grouped_gemm = MoeGroupedGemm()
        self.act_fn = SiluAndMul() if activation == "silu" else GeluAndMul("tanh")
        self.moe_sum = MoeSum()
        self.per_token_group_quant_fp8 = PerTokenGroupQuantFp8()
        self.silu_mul_quant_fp8 = SiluMulQuantFp8()
        self._sb = _SHARED_BUF

    def _get_cache13(self, total_elems, device, dtype):
        sb = self._sb
        if sb.cache13 is None or sb.cache13.numel() < total_elems:
            sb.cache13 = torch.empty(total_elems, device=device, dtype=dtype)
        return sb.cache13[:total_elems]

    def _get_fp8_bufs(self, buf_id, M, K, device):
        sb = self._sb
        attr_a = f"a_fp8_{buf_id}"
        attr_s = f"a_scale_{buf_id}"
        num_groups = math.ceil(K / _FP8_GROUP_SIZE)
        existing_a = getattr(sb, attr_a)
        if existing_a is None or existing_a.size(0) < M or existing_a.size(1) < K:
            setattr(sb, attr_a, torch.empty(M, K, dtype=torch.float8_e4m3fn, device=device))
            setattr(sb, attr_s, torch.empty(M, num_groups, dtype=torch.float32, device=device))
        a = getattr(sb, attr_a)
        s = getattr(sb, attr_s)
        return a[:M, :K], s[:M, :num_groups]

    def _get_dg_workspace(self, buf_id, shape, device, dtype):
        sb = self._sb
        attr = f"dg_ws{buf_id}"
        existing = getattr(sb, attr)
        elem_size = torch.tensor([], dtype=dtype).element_size()
        needed_bytes = elem_size
        for s in shape:
            needed_bytes *= s
        if existing is None or existing.numel() < needed_bytes:
            setattr(sb, attr, torch.empty(needed_bytes, device=device, dtype=torch.uint8))
        raw = getattr(sb, attr)
        needed_elems = needed_bytes // elem_size
        return raw[:needed_bytes].view(dtype)[:needed_elems].view(shape)

    def forward(
        self,
        hidden_states: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        w13_scale: torch.Tensor | None = None,
        w2_scale: torch.Tensor | None = None,
        w13_scale_dg: torch.Tensor | None = None,
        w2_scale_dg: torch.Tensor | None = None,
        use_fp8_w8a8: bool = False,
        block_shape: list[int] | None = None,
    ) -> torch.Tensor:
        M, K = hidden_states.size()
        E, N2, _ = w13.size()
        N = N2 // 2
        top_k = topk_ids.size(1)

        if (self.activation == "silu"
                and use_fp8_w8a8
                and _valid_deep_gemm(hidden_states, w13, w2)
                and not torch.cuda.is_current_stream_capturing()):
            dg_w13_scale = w13_scale_dg if w13_scale_dg is not None else w13_scale
            dg_w2_scale = w2_scale_dg if w2_scale_dg is not None else w2_scale
            return self._forward_deep_gemm(
                hidden_states, w13, w2, topk_weights, topk_ids,
                num_experts, dg_w13_scale, dg_w2_scale, block_shape,
                M, K, E, N, N2, top_k,
            )
        else:
            return self._forward_triton(
                hidden_states, w13, w2, topk_weights, topk_ids,
                num_experts, w13_scale, w2_scale,
                use_fp8_w8a8, block_shape,
                M, K, E, N, N2, top_k,
            )

    def _forward_deep_gemm(
        self,
        hidden_states, w13, w2, topk_weights, topk_ids,
        num_experts, w13_scale, w2_scale, block_shape,
        M, K, E, N, N2, top_k,
    ) -> torch.Tensor:
        """DeepGEMM path: permute -> grouped GEMM1 -> fused act+quant -> grouped GEMM2 -> unpermute."""
        alignment = _FP8_GROUP_SIZE

        M_sum = _compute_aligned_M(M, top_k, num_experts, alignment)

        a_fp8, a_scale = self._get_fp8_bufs(1, M, K, hidden_states.device)
        self.per_token_group_quant_fp8(hidden_states, a_fp8, a_scale)

        a1_perm, a1_scale_perm, expert_ids, inv_perm = _deepgemm_permute(
            a_fp8, a_scale, topk_ids, num_experts, alignment,
        )

        mm1_out = self._get_dg_workspace(1, (M_sum, N2), hidden_states.device, hidden_states.dtype)
        m_grouped_fp8_gemm_nt_contiguous(
            (a1_perm, a1_scale_perm), (w13, w13_scale), mm1_out, expert_ids,
        )

        quant_out = self._get_dg_workspace(
            2, (M_sum, N), hidden_states.device, torch.float8_e4m3fn,
        )
        a2_fp8, a2_scale = self.silu_mul_quant_fp8(
            mm1_out, output=quant_out,
        )

        mm2_out = self._get_dg_workspace(1, (M_sum, K), hidden_states.device, hidden_states.dtype)
        m_grouped_fp8_gemm_nt_contiguous(
            (a2_fp8, a2_scale), (w2, w2_scale), mm2_out, expert_ids,
        )

        output = torch.empty(M, K, dtype=hidden_states.dtype, device=hidden_states.device)
        _deepgemm_unpermute_and_reduce(mm2_out, topk_ids, topk_weights, inv_perm, output)
        return output

    def _forward_triton(
        self,
        hidden_states, w13, w2, topk_weights, topk_ids,
        num_experts, w13_scale, w2_scale,
        use_fp8_w8a8, block_shape,
        M, K, E, N, N2, top_k,
    ) -> torch.Tensor:
        """Triton fallback path (original implementation with JSON autotuning)."""
        config = get_triton_config(
            M, w13.shape, w2.shape, top_k,
            use_fp8=use_fp8_w8a8, block_shape=block_shape,
            default_style=self.config_style,
        )

        use_naive = (M * top_k * SPARSITY_FACTOR <= num_experts)

        sorted_token_ids, expert_ids, num_tokens_post_padded = self.moe_align(
            topk_ids, config["BLOCK_SIZE_M"], num_experts, naive=use_naive,
        )

        cache13_size = M * top_k * max(N2, K)
        cache13_flat = self._get_cache13(cache13_size, hidden_states.device, hidden_states.dtype)
        intermediate1 = cache13_flat[:M * top_k * N2].view(M * top_k, N2)
        intermediate3 = cache13_flat[:M * top_k * K].view(M * top_k, K)

        if use_fp8_w8a8:
            a_fp8, a_scale = self._get_fp8_bufs(1, M, K, hidden_states.device)
            self.per_token_group_quant_fp8(hidden_states, a_fp8, a_scale)
            gemm1_input = a_fp8
            gemm1_a_scale = a_scale
        else:
            gemm1_input = hidden_states
            gemm1_a_scale = None

        self.moe_grouped_gemm(
            gemm1_input, w13, intermediate1,
            topk_weights, sorted_token_ids, expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=False, top_k=top_k, config=config,
            a_scale=gemm1_a_scale, b_scale=w13_scale,
            use_fp8_w8a8=use_fp8_w8a8, block_shape=block_shape,
        )

        intermediate2 = self.act_fn(intermediate1)

        if use_fp8_w8a8:
            a2_fp8, a2_scale = self._get_fp8_bufs(2, M * top_k, N, hidden_states.device)
            self.per_token_group_quant_fp8(intermediate2, a2_fp8, a2_scale)
            gemm2_input = a2_fp8
            gemm2_a_scale = a2_scale
        else:
            gemm2_input = intermediate2
            gemm2_a_scale = None

        self.moe_grouped_gemm(
            gemm2_input, w2, intermediate3,
            topk_weights, sorted_token_ids, expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=True, top_k=1, config=config,
            a_scale=gemm2_a_scale, b_scale=w2_scale,
            use_fp8_w8a8=use_fp8_w8a8, block_shape=block_shape,
        )

        return self.moe_sum(intermediate3, top_k)

_C = lazy_op("topk_softmax", "topk_softmax.cu")

class TopKSoftmax(nn.Module):
    """Fused top-k selection with softmax normalization.

    Pre-allocates topk_weights and topk_ids buffers for CUDA graph
    compatibility.
    """

    def __init__(self):
        super().__init__()
        self._topk_weights = None
        self._topk_ids = None

    def _ensure_buffers(self, M, top_k, device):
        if self._topk_weights is None or self._topk_weights.size(0) < M:
            self._topk_weights = torch.empty(
                M, top_k, device=device, dtype=torch.float32,
            )
            self._topk_ids = torch.empty(
                M, top_k, device=device, dtype=torch.int32,
            )

    def forward(
        self,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select top-k experts with softmax weights.

        Args:
            router_logits: [M, num_experts] router scores
            top_k: number of experts per token
            renormalize: renormalize weights to sum to 1

        Returns:
            topk_weights: [M, top_k] float32
            topk_ids: [M, top_k] int32
        """
        M = router_logits.size(0)
        self._ensure_buffers(M, top_k, router_logits.device)
        topk_weights = self._topk_weights[:M]
        topk_ids = self._topk_ids[:M]
        _C.topk_softmax(topk_weights, topk_ids, router_logits,
                        renormalize, 0.0, None)
        return topk_weights, topk_ids

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

"""Qwen3 Mixture-of-Experts block with fused Triton grouped GEMM.

Supports FP8 W8A8 block-scaled expert weights. When quant_config is set,
expert weights are stored as float8_e4m3fn with per-block scale factors,
and the fused experts kernel runs FP8 dot products with FP32 accumulation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from fastkernels.infra.tp import _tp_rank, _tp_size

_FP8_BLOCK = 128


class Model(nn.Module):
    """Qwen3 Mixture-of-Experts with fused Triton grouped GEMM.

    Weights (FP8 mode):
      gate:     [num_experts, hidden_size] (bfloat16, replicated)
      w13:      [E, 2*moe_intermediate_per_tp, hidden_size] (float8_e4m3fn)
      w13_scale:[E, scale_rows_13, scale_cols_13] (float32)
      w2:       [E, hidden_size, moe_intermediate_per_tp] (float8_e4m3fn)
      w2_scale: [E, scale_rows_2, scale_cols_2] (float32)

    Weights (BF16 mode):
      gate:  [num_experts, hidden_size]
      w13:   [E, 2*moe_intermediate_per_tp, hidden_size]
      w2:    [E, hidden_size, moe_intermediate_per_tp]
    """

    def __init__(self, config, quant_config: dict | None = None):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        tp = _tp_size()
        self.tp_size = tp
        self.intermediate_per_tp = config.moe_intermediate_size // tp
        self.renormalize = getattr(config, "norm_topk_prob", True)
        self.use_fp8 = quant_config is not None

        self.gate = ReplicatedLinear(
            config.hidden_size, config.num_experts, bias=False,
        )

        w13_rows = 2 * self.intermediate_per_tp
        w2_cols = self.intermediate_per_tp

        if self.use_fp8:
            block_size = quant_config.get("weight_block_size", [128, 128])
            self.block_shape = block_size
            block_n, block_k = block_size[0], block_size[1]

            self.w13 = nn.Parameter(torch.empty(
                config.num_experts, w13_rows, config.hidden_size,
                dtype=torch.float8_e4m3fn,
            ), requires_grad=False)
            self.w13_scale = nn.Parameter(torch.ones(
                config.num_experts,
                math.ceil(w13_rows / block_n),
                math.ceil(config.hidden_size / block_k),
                dtype=torch.float32,
            ), requires_grad=False)

            self.w2 = nn.Parameter(torch.empty(
                config.num_experts, config.hidden_size, w2_cols,
                dtype=torch.float8_e4m3fn,
            ), requires_grad=False)
            self.w2_scale = nn.Parameter(torch.ones(
                config.num_experts,
                math.ceil(config.hidden_size / block_n),
                math.ceil(w2_cols / block_k),
                dtype=torch.float32,
            ), requires_grad=False)

            self.w13.weight_loader = self._w13_weight_loader_fp8
            self.w13_scale.weight_loader = self._w13_scale_loader
            self.w2.weight_loader = self._w2_weight_loader_fp8
            self.w2_scale.weight_loader = self._w2_scale_loader
        else:
            self.block_shape = None
            self.w13 = nn.Parameter(torch.empty(
                config.num_experts, w13_rows, config.hidden_size,
            ))
            self.w13.weight_loader = self._w13_weight_loader

            self.w2 = nn.Parameter(torch.empty(
                config.num_experts, config.hidden_size, w2_cols,
            ))
            self.w2.weight_loader = self._w2_weight_loader

            self.w13_scale = None
            self.w2_scale = None

        self.topk_softmax = TopKSoftmax()
        self.fused_experts = FusedExperts()
        self.allreduce = AllReduce()

        # Custom-op dispatch for torch.compile (set by engine after model init)
        self._use_custom_op = False
        self._layer_name = ""

    # --- BF16 weight loaders ---

    def _w13_weight_loader(self, param, loaded_weight, expert_id: int, is_w1: bool):
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp
        shard = loaded_weight.narrow(0, rank * N, N)
        offset = 0 if is_w1 else N
        param.data[expert_id, offset:offset + N, :].copy_(shard)

    def _w2_weight_loader(self, param, loaded_weight, expert_id: int):
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp
        param.data[expert_id].copy_(loaded_weight.narrow(1, rank * N, N))

    # --- FP8 weight loaders ---

    def _w13_weight_loader_fp8(self, param, loaded_weight, expert_id: int, is_w1: bool):
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp
        shard = loaded_weight.narrow(0, rank * N, N)
        offset = 0 if is_w1 else N
        param.data[expert_id, offset:offset + N, :].copy_(shard)

    def _w13_scale_loader(self, param, loaded_weight, expert_id: int, is_w1: bool):
        tp, rank = _tp_size(), _tp_rank()
        block_n = self.block_shape[0]
        N = self.intermediate_per_tp
        scale_rows_per_shard = math.ceil(N / block_n)
        full_scale_rows = loaded_weight.shape[0]
        rows_per_tp = full_scale_rows // tp
        src = loaded_weight.narrow(0, rank * rows_per_tp, rows_per_tp)
        offset = 0 if is_w1 else scale_rows_per_shard
        param.data[expert_id, offset:offset + rows_per_tp, :].copy_(src)

    def _w2_weight_loader_fp8(self, param, loaded_weight, expert_id: int):
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp
        param.data[expert_id].copy_(loaded_weight.narrow(1, rank * N, N))

    def _w2_scale_loader(self, param, loaded_weight, expert_id: int):
        tp, rank = _tp_size(), _tp_rank()
        block_k = self.block_shape[1]
        N = self.intermediate_per_tp
        scale_cols_per_shard = math.ceil(N / block_k)
        full_scale_cols = loaded_weight.shape[1]
        cols_per_tp = full_scale_cols // tp
        src = loaded_weight.narrow(1, rank * cols_per_tp, cols_per_tp)
        param.data[expert_id].copy_(src)

    def forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Core MoE logic, callable from both eager and custom-op paths."""
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)

        router_logits = self.gate(hidden_states)
        topk_weights, topk_ids = self.topk_softmax(
            router_logits, self.top_k, renormalize=self.renormalize,
        )

        w13_scale_dg = getattr(self, 'w13_scale_dg', None)
        w2_scale_dg = getattr(self, 'w2_scale_dg', None)

        out = self.fused_experts(
            hidden_states, self.w13, self.w2,
            topk_weights, topk_ids, self.num_experts,
            w13_scale=self.w13_scale,
            w2_scale=self.w2_scale,
            w13_scale_dg=w13_scale_dg,
            w2_scale_dg=w2_scale_dg,
            use_fp8_w8a8=self.use_fp8,
            block_shape=self.block_shape,
        )

        if self.tp_size > 1 and not self._use_custom_op:
            out = self.allreduce(out)

        return out.view(orig_shape)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._use_custom_op:
            # The all-reduce stays *outside* the opaque op: inside it Inductor
            # cannot see the collective, so ``AllReduceFusedAddRMSNormPass`` has
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

### Qwen3MoE

| count | args |
|------:|------|
| 86010 | `hidden_states:bfloat16[1000, 4096]` |
| 23876 | `hidden_states:bfloat16[1, 4096]` |
| 4982 | `hidden_states:bfloat16[16384, 4096]` |
| 564 | `hidden_states:bfloat16[473, 4096]` |
| 564 | `hidden_states:bfloat16[314, 4096]` |
| 470 | `hidden_states:bfloat16[804, 4096]` |
| 470 | `hidden_states:bfloat16[429, 4096]` |
| 470 | `hidden_states:bfloat16[318, 4096]` |

    
Optimize the architecture named Model with custom Triton operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Let's think step by step.
