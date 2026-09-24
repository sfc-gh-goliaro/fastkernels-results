"""Fused top-k + softmax routing for Mixture-of-Experts.

Same contract as the reference implementation: ``forward(router_logits, top_k,
renormalize=True)`` returns ``(topk_weights[M, top_k] float32, topk_ids[M, top_k]
int32)``.  Output buffers grow monotonically and are reused, so the steady-state
call allocates nothing and issues exactly one kernel launch.

**Output lifetime.**  Like the reference implementation, the fused path returns
*views into persistent buffers*, not fresh tensors.  A later call at the same or a
smaller ``M`` overwrites what an earlier call returned, and a returned view is only
valid until the next call on the same module.  Callers that need to keep a result
must clone it.  This is what makes the steady-state call allocation-free and CUDA
graph capturable, and it is deliberately the same contract the reference has; it
also means a single module instance must not be driven from two streams
concurrently.  The fallback path, by contrast, returns freshly allocated tensors.

bf16 inputs with ``num_experts`` in {32, 64, 128, 256} and ``top_k`` in {1, 2, 4, 8}
under renormalization -- which covers the (128, 8) case the captured workload uses
-- go to a single fused CUDA kernel in ``topk_softmax_fused.cu`` that selects on the
raw logits via a monotonic packed integer key and then takes only ``top_k``
exponentials instead of ``num_experts``, because the softmax denominator cancels
under renormalization.

Everything else routes to a pure-PyTorch path that reproduces the reference
implementation's semantics exactly, including both of its tie orders: the butterfly
kernel's lowest-index-wins rule for a power-of-two expert count up to 256, and the
block-reduce kernel's reduction-tree order for anything else.

Known divergences of the fused path from the reference kernel.  None was observed
on the benchmark's input distribution, and the measured margin to each condition is
four to five orders of magnitude -- but ``torch.randn`` has unbounded support, so
that is a bound on the probability, not an impossibility proof.  See
``docs/divergence.md`` for the margins and the derivation:

* **Signed zero.**  The packed key orders ``+0.0`` strictly above ``-0.0``, where
  IEEE comparison calls them equal.  A row holding both near the top-k cut picks
  the ``+0.0`` slot; the reference kernel picks the lower expert index.  This is
  the only pair in all 65,536 bf16 patterns where the two orders disagree, and
  the affected weights differ by at most ~3e-8.
* **Probability collapse.**  The reference kernel argmaxes over fp32
  probabilities, so two distinct bf16 logits that exponentiate to the same fp32
  probability tie there but not here.  That needs a row span beyond ~87.34 nats
  or an eighth-largest logit below about ``2**-16``.  Every slot that diverges
  this way carries weight exactly ``0.0``, so the weights stay bit-identical.
* **NaN ordering.**  A NaN logit compares false against everything in the
  reference kernel but has a defined key position here.  Rows containing ``+inf``
  make the reference kernel emit NaN weights, so neither implementation is
  usable on them.
"""

from __future__ import annotations

import hashlib
import os

import torch
import torch.nn as nn

from fastkernels.infra.cuda_ext import lazy_op

# The extension name is the sole key for the ninja build directory
# (``~/.cache/torch_extensions/<pyver_cuver>/<name>``) and for the resulting
# ``<name>.so``, and that cache is shared across every operator workspace and
# across runs.  Reusing the reference implementation's name would make two
# different sources contend for one build directory in a process that imports
# both modules, so the name is both operator-specific and content-addressed: any
# edit to the sidecar moves the build directory, and a stale or concurrently
# written ``.so`` can never be picked up.
_SOURCE = "topk_softmax_fused.cu"
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), _SOURCE), "rb") as _f:
    _SOURCE_DIGEST = hashlib.sha256(_f.read()).hexdigest()[:12]

_C = lazy_op(f"topk_softmax_fused_{_SOURCE_DIGEST}", _SOURCE)

# The configurations the fused kernel is compiled for.  The captured workload only
# ever uses (128, 8); the rest exist so the module is honest about its contract
# without falling back to PyTorch for every nearby shape.
_FUSED_NUM_EXPERTS = (32, 64, 128, 256)
_FUSED_TOP_K = (1, 2, 4, 8)
_FUSED_ALIGNMENT = 16  # bytes; every compiled lane slice is a multiple of this
# The kernel forms element offsets as `row * num_experts + ...` in int32, so the
# bound is on the element *count*, not the row count.  A fixed row cutoff would be
# wrong: at 256 experts it admits twice as many rows as int32 can address, which is
# an out-of-bounds read rather than a wrong answer.
_FUSED_MAX_ELEMENTS = (1 << 31) - 1

# What the reference implementation accepts; anything else must be rejected rather
# than silently upcast.
_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _reference_topk_softmax(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference semantics in pure PyTorch, for inputs the fused kernel declines.

    Mirrors the reference kernel step for step: upcast to fp32, subtract the row
    max, exponentiate, scale by the reciprocal of the row sum, then take the
    ``top_k`` largest probabilities with ties broken toward the lower expert
    index, and optionally renormalize over just those.

    A bare ``torch.topk`` will not do here -- it gives no guarantee about which of
    two equal values it returns first, so it does not reproduce the reference
    kernel's tie order.  A stable descending sort does: among equal probabilities
    it keeps the original column order, which is lowest-index-first.
    """
    if router_logits.dim() != 2:
        raise ValueError(
            f"router_logits must be 2D [num_tokens, num_experts], got "
            f"{tuple(router_logits.shape)}"
        )
    if router_logits.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"router_logits must be float32, float16 or bfloat16, got "
            f"{router_logits.dtype}"
        )
    num_experts = router_logits.size(1)
    # Both bounds matter.  The upper one mirrors the reference implementation's
    # own check; the lower one is needed because a negative top_k would otherwise
    # slip through Python slicing as `order[:, :-1]` and silently return
    # num_experts - 1 columns instead of failing.
    if not 0 <= top_k <= num_experts:
        raise ValueError(
            f"top_k ({top_k}) must satisfy 0 <= top_k <= num_experts ({num_experts})"
        )

    logits = router_logits.float()
    shifted = logits - logits.amax(dim=-1, keepdim=True)
    exponentials = torch.exp(shifted)
    probabilities = exponentials * exponentials.sum(dim=-1, keepdim=True).reciprocal()

    if _uses_block_reduce_selection(num_experts, top_k):
        weights, selected = _cub_topk_selection(probabilities, top_k)
    else:
        order = torch.sort(probabilities, dim=-1, descending=True, stable=True).indices
        selected = order[:, :top_k].contiguous()
        weights = torch.gather(probabilities, 1, selected)

    if renormalize:
        weights = weights * weights.sum(dim=-1, keepdim=True).reciprocal()
    return weights.contiguous(), selected.to(torch.int32)


def _uses_block_reduce_selection(num_experts: int, top_k: int) -> bool:
    """Whether the reference would select with the block-reduce kernel.

    The reference dispatches on the expert count: a power of two up to 256 goes to
    the butterfly kernel whose tie rule is lowest-index-wins, and everything else
    goes to `moeSoftmax` followed by a block-reduce selection.  Of those, only the
    `top_k >= 2` variant has a tie order that depends on the reduction tree -- the
    `top_k == 1` variant's reducer breaks ties by lower key, which is
    tie-consistent and so already reproduced by the stable sort.
    """
    is_power_of_two = num_experts != 0 and (num_experts & (num_experts - 1)) == 0
    return (not is_power_of_two or num_experts > 256) and top_k >= 2



# ---------------------------------------------------------------------------
# Exact reproduction of the reference extension's non-power-of-two selection.
#
# The reference has two selection implementations with *different* tie
# semantics.  For a power-of-two expert count up to 256 it runs a butterfly
# argmax carrying an explicit `other_expert < expert` rule, so ties go to the
# lower expert index and a stable descending sort reproduces it.  For any other
# expert count it runs a block-wide reduction whose reducer compares with a
# strict `>`, which makes it non-commutative on ties: the *second* argument wins.
# The winner of a tie is therefore decided by the shape of the reduction tree.
#
# That tree is fully determined, so it can be reproduced rather than guessed at:
#
#   * 256 threads, 8 warps of 32.  Thread t scans experts t, t+256, ... keeping a
#     (max, secondMax) pair, updating with strict `>` so the earliest expert wins
#     a tie within one thread.
#   * Each warp reduces by shuffle-down with offsets 1, 2, 4, 8, 16, combining
#     `op(own, lane + offset)` -- so within a warp a tie goes to the *higher*
#     lane.
#   * Thread 0 then folds the eight warp aggregates sequentially,
#     `op(accumulator, warp[w])` for w = 1..7 -- so across warps a tie goes to
#     the *later* warp.
#
# Together those explain the reference's answer for an all-equal row of 100
# experts, `[99, 95, 98, 94, 97, 93, 96, 92]`: expert 99 wins outright as the
# highest lane of the last populated warp, and the runner-up is 95 rather than 98
# because warps 0-2 aggregate to 95 and the reducer's `>` rejects the equal
# candidate from warp 3.
#
# `top_k == 1` takes a different reference kernel whose reducer breaks ties by
# lower key, which is tie-consistent and therefore tree-independent; the stable
# sort already reproduces it.
# ---------------------------------------------------------------------------
_CUB_THREADS_PER_BLOCK = 256
_CUB_WARP_SIZE = 32
_CUB_WARPS = _CUB_THREADS_PER_BLOCK // _CUB_WARP_SIZE
# The reference initializes every thread's pair to this, and blanks a consumed
# winner to the same value.  Probabilities are non-negative, so it is below any
# real candidate.
_CUB_EMPTY_VALUE = -1.0


def _cub_pair_argmax(first, second):
    """The reference's pair reducer, batched over rows.

    Mirrors it operation for operation, including the fact that it re-identifies
    which side supplied the maximum by comparing *keys* rather than by carrying a
    flag -- that detail matters whenever two candidates share a key, which
    happens for every thread that never saw a valid expert.
    """
    max_value_1, max_key_1, second_value_1, second_key_1 = first
    max_value_2, max_key_2, second_value_2, second_key_2 = second

    # Strict `>`: on a tie the *second* candidate supplies the maximum.
    first_wins = max_value_1 > max_value_2
    global_max_value = torch.where(first_wins, max_value_1, max_value_2)
    global_max_key = torch.where(first_wins, max_key_1, max_key_2)

    supplied_by_first = global_max_key == max_key_1

    keep_first_second = second_value_1 > max_value_2
    if_first = (torch.where(keep_first_second, second_value_1, max_value_2),
                torch.where(keep_first_second, second_key_1, max_key_2))
    keep_second_second = second_value_2 > max_value_1
    if_second = (torch.where(keep_second_second, second_value_2, max_value_1),
                 torch.where(keep_second_second, second_key_2, max_key_1))

    global_second_value = torch.where(supplied_by_first, if_first[0], if_second[0])
    global_second_key = torch.where(supplied_by_first, if_first[1], if_second[1])
    return (global_max_value, global_max_key, global_second_value, global_second_key)


def _cub_block_reduce_pair(probabilities):
    """One block-wide (max, secondMax) reduction in the reference's exact order."""
    rows, num_experts = probabilities.shape
    device = probabilities.device
    threads = _CUB_THREADS_PER_BLOCK

    max_value = probabilities.new_full((rows, threads), _CUB_EMPTY_VALUE)
    max_key = torch.zeros(rows, threads, dtype=torch.int64, device=device)
    second_value = probabilities.new_full((rows, threads), _CUB_EMPTY_VALUE)
    second_key = torch.zeros(rows, threads, dtype=torch.int64, device=device)

    lane = torch.arange(threads, device=device)
    for chunk in range((num_experts + threads - 1) // threads):
        expert = chunk * threads + lane
        valid = expert < num_experts
        gathered = probabilities[:, expert.clamp(max=num_experts - 1)]
        candidate = torch.where(valid, gathered,
                                probabilities.new_full((), _CUB_EMPTY_VALUE))
        expert_row = expert.expand(rows, threads)

        beats_max = valid & (candidate > max_value)
        beats_second = valid & (~beats_max) & (candidate > second_value)
        second_value = torch.where(beats_max, max_value,
                                   torch.where(beats_second, candidate, second_value))
        second_key = torch.where(beats_max, max_key,
                                 torch.where(beats_second, expert_row, second_key))
        max_value = torch.where(beats_max, candidate, max_value)
        max_key = torch.where(beats_max, expert_row, max_key)

    # Per-warp shuffle-down reduction: op(own, lane + offset).
    state = tuple(t.view(rows, _CUB_WARPS, _CUB_WARP_SIZE)
                  for t in (max_value, max_key, second_value, second_key))
    offset = 1
    while offset < _CUB_WARP_SIZE:
        keep = _CUB_WARP_SIZE - offset
        combined = _cub_pair_argmax(tuple(t[:, :, :keep] for t in state),
                                    tuple(t[:, :, offset:] for t in state))
        state = tuple(torch.cat([combined[i], state[i][:, :, keep:]], dim=-1)
                      for i in range(4))
        offset *= 2

    # Thread 0 folds the warp aggregates in order.
    aggregate = tuple(t[:, 0, 0] for t in state)
    for warp in range(1, _CUB_WARPS):
        aggregate = _cub_pair_argmax(aggregate, tuple(t[:, warp, 0] for t in state))
    return aggregate


def _cub_topk_selection(probabilities, top_k):
    """Reproduce the reference's two-winners-per-round selection over `probabilities`.

    `probabilities` is consumed: consumed winners are blanked in place exactly as
    the reference blanks its workspace.
    """
    rows = probabilities.size(0)
    ids = torch.zeros(rows, top_k, dtype=torch.int64, device=probabilities.device)
    weights = probabilities.new_zeros(rows, top_k)
    rounds = (top_k + 1) // 2
    for round_index in range(rounds):
        max_value, max_key, second_value, second_key = _cub_block_reduce_pair(probabilities)
        for half in range(2):
            slot = round_index * 2 + half
            if slot >= top_k:
                break
            value = max_value if half == 0 else second_value
            key = max_key if half == 0 else second_key
            ids[:, slot] = key
            weights[:, slot] = value
            probabilities.scatter_(1, key.unsqueeze(1),
                                   probabilities.new_full((rows, 1), _CUB_EMPTY_VALUE))
    return weights, ids

class TopKSoftmax(nn.Module):
    """Fused top-k selection with softmax normalization.

    Pre-allocates topk_weights and topk_ids buffers for CUDA graph
    compatibility.
    """

    def __init__(self):
        super().__init__()
        self._topk_weights = None
        self._topk_ids = None
        self._buffer_top_k = 0
        # Steady-state calls repeat a handful of row counts, so the `[:M]` slices are
        # cached rather than re-derived.  Keyed on (M, top_k) and dropped wholesale
        # whenever the backing storage moves, since every cached view would then
        # point into a freed allocation.
        self._view_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def _ensure_buffers(self, M, top_k, device):
        buffer = self._topk_weights
        if (buffer is None or buffer.size(0) < M or self._buffer_top_k != top_k
                or buffer.device != device):
            self._topk_weights = torch.empty(
                M, top_k, device=device, dtype=torch.float32,
            )
            self._topk_ids = torch.empty(
                M, top_k, device=device, dtype=torch.int32,
            )
            self._buffer_top_k = top_k
            self._view_cache.clear()

    def _output_views(self, M, top_k, device):
        """The `[:M]` views of the persistent buffers, cached per (M, top_k)."""
        self._ensure_buffers(M, top_k, device)
        key = (M, top_k)
        views = self._view_cache.get(key)
        if views is None:
            views = (self._topk_weights[:M], self._topk_ids[:M])
            self._view_cache[key] = views
        return views

    def _fused_path_applies(self, router_logits, top_k, renormalize) -> bool:
        """Whether the specialized kernel is exactly equivalent for this call.

        Contiguity is not enough for the vector load: a contiguous view can start
        at an odd storage offset (``torch.empty(M * 128 + 1)[1:].view(M, 128)``),
        so the alignment of the actual pointer is what gets tested.  The row-count
        bound is here rather than only in the extension so that an oversized input
        degrades to the reference path instead of raising.
        """
        return (
            bool(renormalize)
            and top_k in _FUSED_TOP_K
            and router_logits.dtype is torch.bfloat16
            and router_logits.dim() == 2
            and router_logits.size(1) in _FUSED_NUM_EXPERTS
            and router_logits.size(0) * router_logits.size(1) <= _FUSED_MAX_ELEMENTS
            and router_logits.is_cuda
            and router_logits.is_contiguous()
            and router_logits.data_ptr() % _FUSED_ALIGNMENT == 0
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
        if not self._fused_path_applies(router_logits, top_k, renormalize):
            return _reference_topk_softmax(router_logits, top_k, renormalize)

        M = router_logits.size(0)
        topk_weights, topk_ids = self._output_views(M, top_k, router_logits.device)
        if M == 0:
            return topk_weights, topk_ids
        _C.topk_softmax_fused(topk_weights, topk_ids, router_logits, renormalize)
        return topk_weights, topk_ids
