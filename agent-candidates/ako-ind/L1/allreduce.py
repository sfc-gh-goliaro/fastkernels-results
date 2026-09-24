"""AllReduce L1 operator with custom IPC all-reduce and NCCL fallback.

Includes the CustomAllreduce class (ported from vLLM, simplified) which
uses JIT-compiled CUDA kernels for intra-node P2P cross-device reduction.

Differences from the baseline port, all in service of per-call overhead on
B200 (see ``fastar.cu`` for the device-side argument):

* the eager path no longer issues a separate ``cudaMemcpyAsync`` to stage the
  input into the IPC buffer -- the copy is fused into the reduce kernel, so a
  non-captured all-reduce is one launch instead of two;
* launch geometry and the one-shot/two-stage crossover are tuned for this
  fabric instead of inheriting vLLM's A100/H100 constants;
* the communicator self-initializes on first use when the serving engine has
  not installed one, so a bare ``AllReduce`` module still takes the IPC path
  instead of falling through to NCCL at decode message sizes.
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
# Self-initialization
#
# The serving engine builds a CustomAllreduce and installs it via
# ``set_custom_ar``. Anything else that instantiates this operator (a bare
# module in a plain torch.distributed job) would otherwise fall through to
# NCCL for every call, which at one-token message sizes costs ~20 us against
# ~5 us for the IPC path. So the first forward that finds no communicator
# builds one itself. This is collective (it opens a gloo side-group and
# exchanges IPC handles), which is safe because every rank runs the same
# operator code in lockstep -- the same property the all-reduce itself needs.
# ---------------------------------------------------------------------------
_AUTO_INIT_DONE = False


def _auto_init_custom_ar():
    global _AUTO_INIT_DONE
    if _AUTO_INIT_DONE or _CUSTOM_AR is not None:
        return _CUSTOM_AR
    _AUTO_INIT_DONE = True
    if os.environ.get("FASTKERNELS_DISABLE_CUSTOM_AR", "0") == "1":
        return None
    try:
        if not dist.is_available() or not dist.is_initialized():
            return None
        world_size = dist.get_world_size()
        if world_size not in CustomAllreduce._SUPPORTED_WORLD_SIZES:
            return None
        if not torch.cuda.is_available():
            return None
        # CustomAllreduce only uses the group for object collectives during
        # setup, and refuses an NCCL group, so give it a CPU side-group.
        group = dist.new_group(backend="gloo")
        ar = CustomAllreduce(group, torch.cuda.current_device())
        if ar.disabled:
            return None
        set_custom_ar(ar)
    except Exception:
        set_custom_ar(None)
    return _CUSTOM_AR


# ---------------------------------------------------------------------------
# AllReduce L1 operator
# ---------------------------------------------------------------------------
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
        if ar is None:
            ar = _auto_init_custom_ar()
        if ar is not None:
            # Any refusal here is a pure function of shape/dtype/world size, so
            # every rank takes the same branch and the fallback stays collective.
            try:
                out = ar.custom_all_reduce(tensor)
            except Exception:
                out = None
            if out is not None:
                return out
        dist.all_reduce(tensor)
        return tensor


# ---------------------------------------------------------------------------
# Custom all-reduce via CUDA IPC
# ---------------------------------------------------------------------------
def _load_ops():
    from fastkernels.infra.cuda_ext import load_op
    return load_op("fastar", "fastar.cu")


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


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return int(v)


# Launch geometry / algorithm crossover. Values are the ones the sweeps in
# ITERATIONS.md settled on for 4x B200 (NVLink, 148 SMs); the env overrides
# exist so the sweep harness can drive them without a rebuild.
_TUNE_DEFAULTS = dict(
    # Crossover, measured on 4x B200 by forcing both algorithms across
    # 512 KiB..7.81 MB (full curve in ITERATIONS.md). one-shot wins by 12.2 us at
    # 512 KiB, 6.1 at 2 MiB, 2.0 at 3 MiB, then loses by 2.0 at 4 MiB and 16.4 at
    # 7.81 MB -- so the crossover is between 3 and 4 MiB, 7x above vLLM's
    # hardcoded 512 KiB, and this value straddles it. It also puts the scored
    # bf16[698,2048] (2.79 MB) on one-shot, which with `push` below runs 35.9 us
    # against two-stage's 40.0.
    oneshot_max_bytes=4 * 1024 * 1024,
    # The registered (CUDA-graph) path is not exercised by the bench, so keep its
    # crossover where the argument, not a measurement, puts it: one-shot moves 3x
    # the message in remote bytes, so hand over sooner.
    oneshot_max_bytes_reg=1024 * 1024,
    # Latency path: at one packed unit per thread the grid is one block whatever
    # the limit, and 512 threads covers 8 KiB in a single block.
    oneshot_threads=512,
    oneshot_blocks=148,
    # Bandwidth path: this is the single biggest lever in the round. vLLM's
    # 36 blocks x 512 threads scores 95.3 us on bf16[1000,4096]; 128 x 1024
    # scores 52.2 us. B200 has 148 SMs, and the curve is monotone in
    # blocks*threads up to ~131k threads, then flat (36->64->96->128 blocks at
    # 1024 threads: 72.7 -> 58.2 -> 56.4 -> 52.2 us).
    twostage_threads=1024,
    twostage_blocks=128,
    force_algo=0,
    double_buffer=1,
    oneshot_region_bytes=4 * 1024 * 1024,
    oneshot_regmode=1,
    # 0 (one ld.acquire.sys per poll) measured ~2 us better than 1 (volatile
    # poll + a single fence.acq_rel.sys) on every shape; see ITERATIONS.md.
    barrier_mode=0,
    # Push (write-based) one-shot: each rank writes its input into every
    # peer's slot and then reduces ngpus slots that are all local, instead of
    # reading every peer's staged copy. Measured 35.9 vs 38.0 us on
    # bf16[698,2048] and identical at the latency floor; at 7.81 MB the
    # crossover keeps that shape on two-stage, so this only ever applies
    # below oneshot_max_bytes where one-shot already won.
    push=1,
)

_TUNE_ENV = {
    "oneshot_max_bytes": "FASTAR_OS_MAX",
    "oneshot_max_bytes_reg": "FASTAR_OS_MAX_REG",
    "oneshot_threads": "FASTAR_OS_THREADS",
    "oneshot_blocks": "FASTAR_OS_BLOCKS",
    "twostage_threads": "FASTAR_TS_THREADS",
    "twostage_blocks": "FASTAR_TS_BLOCKS",
    "force_algo": "FASTAR_FORCE_ALGO",
    "double_buffer": "FASTAR_DB",
    "oneshot_region_bytes": "FASTAR_REGION",
    "oneshot_regmode": "FASTAR_REGMODE",
    "barrier_mode": "FASTAR_BMODE",
    "push": "FASTAR_PUSH",
}


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
        self.tune = dict(_TUNE_DEFAULTS)
        for key, env in _TUNE_ENV.items():
            self.tune[key] = _env_int(env, self.tune[key])

        # The staging allocation is max_size (the size a call must fit in to
        # qualify) plus two dedicated one-shot slots. Keeping those slots out of
        # the offset-0 region is what lets the one-shot kernel drop its closing
        # barrier without ever racing a two-stage call's staging writes.
        region = self.tune["oneshot_region_bytes"]
        # Layout of the staging allocation, low to high:
        #   [0, max_size)                          every path that keeps a
        #                                          closing barrier (two-stage,
        #                                          and one-shot when it cannot
        #                                          double-buffer)
        #   [max_size, +2*world_size*push_slot)    push one-shot slots, one per
        #                                          (rank, parity)
        #   top 2*region                           pull one-shot slots, one per
        #                                          parity
        # The kernel derives the pull slots from the TOP of the allocation
        # (capacity - 2*region), so sizing the buffer to hold both leaves the
        # three regions disjoint. Disjointness is what lets the one-shot paths
        # drop their closing barrier without ever racing a two-stage call's
        # staging writes.
        #
        # A push slot only has to hold a one-shot-sized message, i.e. anything
        # below the crossover; above it the call takes two-stage and the
        # offset-0 region instead.
        push_slot = _env_int(
            "FASTAR_PUSH_SLOT",
            min(max_size, self.tune["oneshot_max_bytes"]),
        )
        push_base = max_size
        buffer_size = push_base + 2 * world_size * push_slot + 2 * region
        self.buffer_size = buffer_size
        self.push_base = push_base
        self.push_slot = push_slot

        self.meta_ptrs = self._create_shared_buffer(
            ops.meta_size() + max_size, group=group
        )
        self.buffer_ptrs = self._create_shared_buffer(buffer_size, group=group)
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
        ops.set_layout(self._ptr, max_size, self.push_base, self.push_slot)
        self._push_tune()

    def _push_tune(self):
        t = self.tune
        self.ops.set_tune(
            self._ptr,
            t["oneshot_max_bytes"],
            t["oneshot_max_bytes_reg"],
            t["oneshot_threads"],
            t["oneshot_blocks"],
            t["twostage_threads"],
            t["twostage_blocks"],
            t["force_algo"],
            t["double_buffer"],
            t["oneshot_region_bytes"],
            t["oneshot_regmode"],
            t["barrier_mode"],
            t["push"],
        )

    def retune(self, **kwargs):
        """Override tuning knobs in place (used by the sweep harness)."""
        self.tune.update(kwargs)
        self._push_tune()

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
                self.buffer_ptrs[self.rank], self.buffer_size
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
