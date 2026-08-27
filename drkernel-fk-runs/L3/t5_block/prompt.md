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
from fastkernels.infra.tp import _tp_size, _tp_rank
from transformers import T5Config
from typing import Optional
import math
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

_FP8_BLOCK = 128

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

def _is_batch_invariant() -> bool:
    """vLLM's dynamic FP8 blockscale dispatch forces the DeepGEMM path (skips the
    FlashInfer swapAB kernel) for ALL M under batch-invariant determinism mode
    (``VLLM_BATCH_INVARIANT=1``) — see the early-out in
    ``scaled_mm/flashinfer.py`` and ``grouped_topk._is_batch_invariant``. Mirror
    it so fastkernels matches vLLM in that mode."""
    return os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1"

_FLASHINFER_FN: object | None = None

_FLASHINFER_RESOLVED = False

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

def _scale_shape(out_dim: int, in_dim: int) -> tuple[int, int]:
    return (math.ceil(out_dim / _FP8_BLOCK), math.ceil(in_dim / _FP8_BLOCK))

class MergedColumnParallelLinear(nn.Module):
    """gate_proj + up_proj merged into one linear, sharded across TP."""

    def __init__(self, input_size: int, output_sizes: list[int], bias: bool = False,
                 quant_config: dict | None = None, disable_tp: bool = False):
        super().__init__()
        tp = _tp_size()
        self.disable_tp = disable_tp
        self.output_sizes = output_sizes
        total = sum(output_sizes)
        if not disable_tp:
            assert all(s % tp == 0 for s in output_sizes)
        self.use_fp8 = quant_config is not None

        effective_tp = 1 if disable_tp else tp
        if self.use_fp8:
            self.weight = nn.Parameter(
                torch.empty(total // effective_tp, input_size, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(*_scale_shape(total // effective_tp, input_size), dtype=torch.float32),
                requires_grad=False,
            )
            self.weight.weight_loader = self._weight_loader
            self.weight_scale_inv.weight_loader = self._scale_loader
            self.linear_op = _get_fp8_linear_cls()()
        else:
            self.weight = nn.Parameter(torch.empty(total // effective_tp, input_size))
            self.weight.weight_loader = self._weight_loader

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(total // tp))
            self.bias.weight_loader = self._weight_loader

    def _weight_loader(self, param, loaded_weight, shard_id: int | None = None):
        tp, rank = _tp_size(), _tp_rank()
        if shard_id is None:
            # Fused weight: ``loaded_weight`` is the full ``[sum(output_sizes), in]``
            # tensor.  Recurse per-shard so each output block is sharded across
            # TP ranks independently (mirrors vLLM's ``MergedColumnParallelLinear``
            # weight loader when called without an explicit shard id).
            offset = 0
            for sid, sz in enumerate(self.output_sizes):
                self._weight_loader(
                    param, loaded_weight.narrow(0, offset, sz), sid,
                )
                offset += sz
            return
        effective_tp = 1 if self.disable_tp else tp
        shard_offset = sum(self.output_sizes[:shard_id]) // effective_tp
        shard_size = self.output_sizes[shard_id] // effective_tp
        dst = param.data.narrow(0, shard_offset, shard_size)
        if self.disable_tp:
            dst.copy_(loaded_weight)
        else:
            src = loaded_weight.chunk(tp, 0)[rank]
            dst.copy_(src)

    def _scale_loader(self, param, loaded_weight, shard_id: int):
        tp, rank = _tp_size(), _tp_rank()
        effective_tp = 1 if self.disable_tp else tp
        shard_size_out = self.output_sizes[shard_id] // effective_tp
        scale_rows = math.ceil(shard_size_out / _FP8_BLOCK)
        shard_offset_out = sum(self.output_sizes[:shard_id]) // effective_tp
        scale_offset = math.ceil(shard_offset_out / _FP8_BLOCK)
        if self.disable_tp:
            param.data.narrow(0, scale_offset, scale_rows).copy_(loaded_weight)
        else:
            src = loaded_weight.chunk(tp, 0)[rank]
            param.data.narrow(0, scale_offset, scale_rows).copy_(src)

    def forward(self, x):
        if self.use_fp8:
            return self.linear_op(x, self.weight, self.weight_scale_inv, self.bias)
        return F.linear(x, self.weight, self.bias)

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

class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)

class NewGELUActivation(nn.Module):
    """GELU approximation matching HuggingFace's NewGELUActivation exactly."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class GELU(nn.Module):
    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate=self.approximate)

def _get_act_fn(name: str) -> nn.Module:
    act_fns = {
        "relu": nn.ReLU(),
        "gelu": GELU(),
        "gelu_new": NewGELUActivation(),
        "silu": SiLU(),
    }
    if name in act_fns:
        return act_fns[name]
    raise ValueError(f"Unknown activation function: {name}")

class T5DenseGatedActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = MergedColumnParallelLinear(
            config.d_model, [config.d_ff, config.d_ff], bias=False,
        )
        self.wo = RowParallelLinear(config.d_ff, config.d_model, bias=False)
        self.act = _get_act_fn(config.dense_act_fn)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = self.wi(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden_states = self.act(gate) * up
        hidden_states = self.wo(hidden_states)
        return hidden_states

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

class T5DenseActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = ColumnParallelLinear(config.d_model, config.d_ff, bias=False)
        self.wo = RowParallelLinear(config.d_ff, config.d_model, bias=False)
        self.act = _get_act_fn(config.dense_act_fn)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states

class Softmax(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=self.dim)

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 padding_idx: int | None = None):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim,
                                padding_idx=padding_idx)

    def forward(self, input_ids):
        return self.emb(input_ids)

class QKVParallelLinear(nn.Module):
    """Q, K, V projections merged and sharded across TP."""

    def __init__(self, hidden_size: int, head_size: int,
                 total_num_heads: int, total_num_kv_heads: int,
                 bias: bool = False, quant_config: dict | None = None):
        super().__init__()
        tp = _tp_size()
        self.head_size = head_size
        self.num_heads = total_num_heads // tp
        # Replicate KV heads when not evenly divisible by TP
        if total_num_kv_heads % tp == 0:
            self.num_kv_heads = total_num_kv_heads // tp
            self._replicate_kv = False
        else:
            self.num_kv_heads = total_num_kv_heads
            self._replicate_kv = True
        output_size = (self.num_heads + 2 * self.num_kv_heads) * head_size
        self.use_fp8 = quant_config is not None

        if self.use_fp8:
            self.weight = nn.Parameter(
                torch.empty(output_size, hidden_size, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(*_scale_shape(output_size, hidden_size), dtype=torch.float32),
                requires_grad=False,
            )
            self.weight.weight_loader = self._weight_loader
            self.weight_scale_inv.weight_loader = self._scale_loader
            self.linear_op = _get_fp8_linear_cls()()
        else:
            self.weight = nn.Parameter(torch.empty(output_size, hidden_size))
            self.weight.weight_loader = self._weight_loader

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self._weight_loader

    def _weight_loader(self, param, loaded_weight, shard_id: str):
        tp, rank = _tp_size(), _tp_rank()
        if shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
            src = loaded_weight.chunk(tp, 0)[rank]
        elif shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
            src = loaded_weight if self._replicate_kv else loaded_weight.chunk(tp, 0)[rank]
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
            src = loaded_weight if self._replicate_kv else loaded_weight.chunk(tp, 0)[rank]
        dst = param.data.narrow(0, shard_offset, shard_size)
        dst.copy_(src)

    def _scale_loader(self, param, loaded_weight, shard_id: str):
        tp, rank = _tp_size(), _tp_rank()
        if shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        scale_rows = math.ceil(shard_size / _FP8_BLOCK)
        scale_offset = math.ceil(shard_offset / _FP8_BLOCK)
        src = loaded_weight.chunk(tp, 0)[rank]
        param.data.narrow(0, scale_offset, scale_rows).copy_(src)

    def forward(self, x):
        if self.use_fp8:
            return self.linear_op(x, self.weight, self.weight_scale_inv, self.bias)
        return F.linear(x, self.weight, self.bias)

class BMM(nn.Module):
    """Batch matrix multiply: torch.matmul(a, b)."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)

class T5SelfAttention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.d_kv
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance

        tp_size = _tp_size()
        assert self.n_heads % tp_size == 0
        self.n_heads_per_partition = self.n_heads // tp_size

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.d_model,
            head_size=self.d_kv,
            total_num_heads=self.n_heads,
            total_num_kv_heads=self.n_heads,
            bias=False,
        )

        self.o = RowParallelLinear(self.inner_dim, self.d_model, bias=False)

        self.bmm = BMM()
        self.softmax = Softmax(dim=-1)

        if has_relative_attention_bias:
            self.relative_attention_bias = Embedding(
                self.relative_attention_num_buckets, self.n_heads,
            )

    @staticmethod
    def _relative_position_bucket(
        relative_position: torch.Tensor,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> torch.Tensor:
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position),
            )
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )
        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large,
        )
        return relative_buckets

    def compute_bias(self, query_length: int, key_length: int, device: torch.device) -> torch.Tensor:
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position, bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)
        tp_rank = _tp_rank()
        head_start = tp_rank * self.n_heads_per_partition
        head_end = head_start + self.n_heads_per_partition
        values = values[:, :, head_start:head_end]
        values = values.permute(2, 0, 1).unsqueeze(0)
        return values

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length = hidden_states.shape[:2]

        qkv = self.qkv_proj(hidden_states)
        q_size = self.n_heads_per_partition * self.d_kv
        kv_size = self.n_heads_per_partition * self.d_kv
        query_states, key_states, value_states = qkv.split(
            [q_size, kv_size, kv_size], dim=-1,
        )

        query_states = query_states.view(
            batch_size, seq_length, self.n_heads_per_partition, self.d_kv,
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_length, self.n_heads_per_partition, self.d_kv,
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_length, self.n_heads_per_partition, self.d_kv,
        ).transpose(1, 2)

        scores = self.bmm(query_states, key_states.transpose(3, 2))

        if position_bias is None:
            if self.has_relative_attention_bias:
                position_bias = self.compute_bias(
                    seq_length, seq_length, device=scores.device,
                )
            else:
                position_bias = torch.zeros(
                    (1, self.n_heads_per_partition, seq_length, seq_length),
                    device=scores.device, dtype=scores.dtype,
                )
            if mask is not None:
                position_bias = position_bias + mask

        scores += position_bias
        attn_weights = self.softmax(scores.float()).type_as(scores)
        attn_output = self.bmm(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, -1)
        attn_output = self.o(attn_output)

        return attn_output, position_bias

class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states

"""T5 encoder block: self-attention + FFN with pre-norm residuals (L3).

T5LayerSelfAttention: T5LayerNorm -> T5SelfAttention -> residual add.
T5LayerFF: T5LayerNorm -> T5Dense{Gated}ActDense -> residual add.
T5Block: T5LayerSelfAttention + T5LayerFF.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import T5Config



__targets__ = ["T5Block"]


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.SelfAttention = T5SelfAttention(config, has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed = self.layer_norm(hidden_states)
        attn_output, position_bias = self.SelfAttention(
            normed, mask=mask, position_bias=position_bias,
        )
        hidden_states = hidden_states + attn_output
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        return hidden_states, position_bias


class T5LayerFF(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config)
        else:
            self.DenseReluDense = T5DenseActDense(config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = self.layer_norm(hidden_states)
        ff_output = self.DenseReluDense(normed)
        hidden_states = hidden_states + ff_output
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        return hidden_states


class Model(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.layer = nn.ModuleList([
            T5LayerSelfAttention(config, has_relative_attention_bias),
            T5LayerFF(config),
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, position_bias = self.layer[0](
            hidden_states, mask=mask, position_bias=position_bias,
        )
        hidden_states = self.layer[1](hidden_states)
        return hidden_states, position_bias

    ```

ModelNew must use the same `__init__` arguments and `forward` signature as `Model`.

The kernel will be evaluated on these captured input shapes:

### T5Block

| count | args |
|------:|------|
| 92 | `hidden_states:bfloat16[1, 512, 4096] mask:None position_bias:bfloat16[1, 64, 512, 512]` |
| 4 | `hidden_states:bfloat16[1, 512, 4096] mask:None position_bias:None` |

    
Optimize the architecture named Model with custom Triton operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Let's think step by step.
