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

"""AllReduce L1 operator with custom IPC all-reduce and NCCL fallback.

Includes the CustomAllreduce class (ported from vLLM, simplified) which
uses JIT-compiled CUDA kernels for intra-node P2P cross-device reduction.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ProcessGroup


# ---------------------------------------------------------------------------
# Global custom allreduce communicator (set by engine, used by TP layers)
# ---------------------------------------------------------------------------
_CUSTOM_AR: Optional["CustomAllreduce"] = None


def set_custom_ar(ar):
    global _CUSTOM_AR
    _CUSTOM_AR = ar


def get_custom_ar():
    return _CUSTOM_AR


# ---------------------------------------------------------------------------
# AllReduce L1 operator
# ---------------------------------------------------------------------------
class Model(nn.Module):
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


# ---------------------------------------------------------------------------
# Custom all-reduce via CUDA IPC
# ---------------------------------------------------------------------------
def _load_ops():
    from fastkernels.infra.cuda_ext import load_op
    return load_op("allreduce", "allreduce.cu")


_ops = None


def _get_ops():
    global _ops
    if _ops is None:
        _ops = _load_ops()
    return _ops


def is_weak_contiguous(inp: torch.Tensor) -> bool:
    return inp.is_contiguous() or (
        inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )


class CustomAllreduce:
    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]

    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        max_size: int = 8192 * 1024,
    ) -> None:
        self._IS_CAPTURING = False
        self.disabled = True

        ops = _get_ops()
        self.ops = ops

        self.group = group
        assert dist.get_backend(group) != dist.Backend.NCCL, (
            "CustomAllreduce should be attached to a non-NCCL group."
        )

        rank = dist.get_rank(group=self.group)
        self.rank = rank
        world_size = dist.get_world_size(group=self.group)
        if world_size == 1:
            return
        if world_size not in self._SUPPORTED_WORLD_SIZES:
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self.device = device

        self.disabled = False
        self.meta_ptrs = self._create_shared_buffer(
            ops.meta_size() + max_size, group=group
        )
        self.buffer_ptrs = self._create_shared_buffer(max_size, group=group)
        self.rank_data = torch.empty(
            8 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        self.max_size = max_size
        self.world_size = world_size
        self.fully_connected = True
        self._ptr = ops.init_custom_ar(
            self.meta_ptrs, self.rank_data, rank, self.fully_connected
        )
        ops.register_buffer(self._ptr, self.buffer_ptrs)

    @contextmanager
    def capture(self):
        """Track buffer addresses during CUDA graph capture, then register them."""
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            if not self.disabled:
                self._register_graph_buffers()

    def _register_graph_buffers(self):
        ops = self.ops
        handle, offset = ops.get_graph_buffer_ipc_meta(self._ptr)
        if self.rank == 0:
            print(f"  Registering {len(offset)} custom AR graph buffer addresses")
        all_data: list[list[list[int] | None]] = [
            [None, None] for _ in range(self.world_size)
        ]
        all_data[self.rank] = [handle, offset]
        ranks = sorted(dist.get_process_group_ranks(group=self.group))
        for i, r in enumerate(ranks):
            dist.broadcast_object_list(
                all_data[i], src=r, group=self.group, device="cpu"
            )
        handles = [d[0] for d in all_data]
        offsets = [d[1] for d in all_data]
        ops.register_graph_buffers(self._ptr, handles, offsets)

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        if self.world_size == 2 or self.fully_connected:
            return inp_size <= self.max_size
        return False

    def all_reduce(
        self, inp: torch.Tensor, *, out: Optional[torch.Tensor] = None,
        registered: bool = False
    ) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(inp)
        if registered:
            self.ops.all_reduce(self._ptr, inp, out, 0, 0)
        else:
            self.ops.all_reduce(
                self._ptr, inp, out,
                self.buffer_ptrs[self.rank], self.max_size
            )
        return out

    def custom_all_reduce(self, input: torch.Tensor) -> Optional[torch.Tensor]:
        """Main API: returns reduced tensor or None if custom AR can't handle it."""
        if self.disabled:
            return None
        if self._IS_CAPTURING:
            if not is_weak_contiguous(input):
                return None
            if torch.cuda.is_current_stream_capturing():
                return self.all_reduce(input, registered=True)
            else:
                return torch.empty_like(input)
        if not self.should_custom_ar(input):
            return None
        return self.all_reduce(input, registered=False)

    def close(self):
        if not self.disabled and hasattr(self, '_ptr') and self._ptr:
            self.ops.dispose(self._ptr)
            self._ptr = 0
            self._free_shared_buffer(self.meta_ptrs, rank=self.rank)
            self._free_shared_buffer(self.buffer_ptrs, rank=self.rank)

    def __del__(self):
        self.close()

    def _create_shared_buffer(
        self, size_in_bytes: int, group: ProcessGroup
    ) -> list[int]:
        ops = self.ops
        pointer, handle = ops.allocate_shared_buffer_and_handle(size_in_bytes)

        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=group)

        pointers: list[int] = []
        for i, h in enumerate(handles):
            if i == rank:
                pointers.append(pointer)
            else:
                pointers.append(ops.open_mem_handle(h))
        return pointers

    def _free_shared_buffer(
        self, pointers: list[int], rank: int
    ) -> None:
        self.ops.free_shared_buffer(pointers[rank])

    ```

ModelNew must use the same `__init__` arguments and `forward` signature as `Model`.

The kernel will be evaluated on these captured input shapes:

### AllReduce

| count | args |
|------:|------|
| 172935 | `tensor:bfloat16[1000, 4096]` |
| 48006 | `tensor:bfloat16[1, 4096]` |
| 14410 | `tensor:bfloat16[64, 2304]` |
| 13943 | `tensor:bfloat16[60, 2880]` |
| 10017 | `tensor:bfloat16[16384, 4096]` |
| 9490 | `tensor:bfloat16[1, 2880]` |
| 6985 | `tensor:bfloat16[1, 2304]` |
| 6278 | `tensor:bfloat16[16384, 2880]` |

    
Optimize the architecture named Model with custom Triton operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Let's think step by step.
