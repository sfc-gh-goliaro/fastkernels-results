"""Fused recurrent GLA — single-launch decode-step kernel.

Same ``__init__``/``forward`` contract as the baseline, which forwards straight to
``fla.ops.gla.fused_recurrent_gla``. The difference is a specialization for the decode step: for a
single timestep with an incoming recurrent state, the recurrence collapses to

    h = exp(gk) * h_in + k (x) v
    o = scale * sum_k q[k] * h[k, :]

so one program per (sequence-head, value tile) reads each state tile once, writes the updated tile
once, and reduces the output out of registers. There is no time loop, no partial-output buffer, no
separate dtype cast, and no ``torch.autograd.Function`` / ``input_guard`` / ``autocast_custom_fwd`` /
``triton.autotune`` / ``triton.heuristics`` work per call. That matters twice over: the large-state
shapes are bound by state traffic and the small-state shapes are bound by per-call dispatch, so both
the kernel and the Python around it are on the critical path.

Everything else -- every multi-timestep call, a cold start, a variable-length batch, any other dtype
-- is delegated to the reference, so the general contract stays exact. A single-launch kernel for
``T > 1`` was built and measured; it is faster on the one benchmarked sequence shape but is kept out
of this module deliberately, because this operator is specified as a decode-step specialization with
multi-timestep calls delegated. It lives in ``tools/sequence_kernel_experiment.py`` with its timing
record.

Delegation is keyword-only on purpose: the reference's fifth positional parameter is ``gv``, a value
gate, so a positional ``scale`` would bind to it -- which in practice fails to compile, since a float
then reaches the reference kernel where a pointer is expected.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import triton
import triton.language as tl
from fla.ops.gla import fused_recurrent_gla
from triton.runtime import driver

# `fla.ops.utils.op.exp` switches from `tl.exp` to `libdevice.fast_expf` when this is set, and the
# specialized kernels are only gate-for-gate identical to the reference against plain `tl.exp`.
# Rather than quietly drift outside the fp32 state tolerance, stand down entirely.
_GATE_MATCHES_REFERENCE = os.environ.get("FLA_USE_FAST_OPS", "0") != "1"

# Tile candidates, widest first. A tile has to divide its axis, and both axes are required to be
# multiples of 16, so 16 is always valid and the selections below can never come up empty.
_K_TILES = (256, 128, 64, 32, 16, 8, 4)
_V_TILES = (512, 256, 128, 64, 32, 16)

# Programs to aim for, so a small batch still spreads across the device.
_TARGET_PROGRAMS = 592

# A [BK, BV] fp32 tile costs BK*BV/(32*num_warps) registers per thread. Cap the tile at 64
# registers per thread so it alone cannot push a kernel into spilling.
_TILE_REGS_PER_THREAD = 64

# Device SM count, read once and cached: tile choices are expressed in waves of programs.
_SM_COUNT: int | None = None

# Shape -> tile choice, and shape -> ready-to-call launcher. Both are pure functions of the shape,
# so they are computed once per distinct shape and never re-derived per call.
_CONFIG_CACHE: dict[tuple, dict] = {}
_LAUNCHER_CACHE: dict[tuple, tuple] = {}

# Back both output leaves with a single storage allocation, so a call that returns the updated state
# still makes exactly one allocation. Carving typed views out of one buffer costs about 2 us more
# Python per call than two `empty_like` calls, but that cost hides behind the pipelined launches and
# measures as no difference in the benchmark, so the literal one-allocation form is what ships. Set
# FK_GLA_FUSED_ALLOC=0 to compare against two separate allocations; `docs/tuning.md` has the numbers.
_FUSE_OUTPUT_ALLOCATION = os.environ.get("FK_GLA_FUSED_ALLOC", "1") == "1"


@triton.jit(do_not_specialize=["n_seq_heads"])
def _gla_decode_step_kernel(
    q,
    k,
    v,
    gk,
    h_in,
    o,
    h_out,
    scale,
    n_seq_heads,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    STATE_EVICTION: tl.constexpr,
    STATE_CACHE_MODIFIER: tl.constexpr,
    VALUE_TILE_FAST: tl.constexpr,
):
    """One program per (sequence-head, value tile) of a single decode step.

    ``q``, ``k``, ``gk`` are ``[N, 1, H, K]``, ``v`` and ``o`` are ``[N, 1, H, V]`` and the state is
    ``[N, H, K, V]``. With a single timestep, element ``(n, 0, h, d)`` sits at ``(n*H + h)*D + d``,
    which is the same flattened sequence-head index the state uses, so one index serves every
    tensor.

    With ``VALUE_TILE_FAST`` the value tile is the fast-varying grid dimension, so consecutive
    programs stream one contiguous state slice back to back; the other ordering spreads a wave
    across sequence-heads instead, and exists so the choice can be measured rather than assumed.
    """
    pid = tl.program_id(0).to(tl.int64)
    if VALUE_TILE_FAST:
        i_nh, i_v = pid // NV, pid % NV
    else:
        i_nh, i_v = pid % n_seq_heads, pid // n_seq_heads

    off_v = i_v * BV + tl.arange(0, BV)
    b_v = tl.load(v + i_nh * V + off_v).to(tl.float32)  # [BV], loop-invariant

    p_state = h_in + i_nh * K * V + off_v[None, :]
    p_out_state = h_out + i_nh * K * V + off_v[None, :]
    acc = tl.zeros([BV], dtype=tl.float32)

    for i_k in range(0, K, BK):
        off_k = i_k + tl.arange(0, BK)
        # The reference scales the query before the reduction; keep that order so the output
        # rounds the same way.
        b_q = tl.load(q + i_nh * K + off_k).to(tl.float32) * scale  # [BK]
        b_k = tl.load(k + i_nh * K + off_k).to(tl.float32)  # [BK]
        b_gk = tl.load(gk + i_nh * K + off_k).to(tl.float32)  # [BK]

        # Decay the incoming state, then apply the rank-1 update, then reduce -- the reference's
        # order, so the returned state matches it elementwise.
        b_h = tl.load(
            p_state + off_k[:, None] * V,
            cache_modifier=STATE_CACHE_MODIFIER,
            eviction_policy=STATE_EVICTION,
        )  # [BK, BV] fp32
        b_h = b_h * tl.exp(b_gk)[:, None]
        b_h += b_k[:, None] * b_v[None, :]

        if STORE_FINAL_STATE:
            tl.store(p_out_state + off_k[:, None] * V, b_h, eviction_policy=STATE_EVICTION)

        acc += tl.sum(b_h * b_q[:, None], axis=0)

    tl.store(o + i_nh * V + off_v, acc.to(o.dtype.element_ty))


def _multiprocessor_count() -> int:
    global _SM_COUNT
    if _SM_COUNT is None:
        _SM_COUNT = torch.cuda.get_device_properties(
            torch.cuda.current_device()).multi_processor_count
    return _SM_COUNT


def _widest_value_tile(n_seq_heads: int, V: int, v_tiles: list[int]) -> int:
    """Widest value tile that still spreads enough programs over the device, narrowest as floor."""
    bv = v_tiles[-1]
    for candidate in v_tiles:
        bv = candidate
        if n_seq_heads * (V // candidate) >= _TARGET_PROGRAMS:
            break
    return bv


def _decode_config(n_seq_heads: int, K: int, V: int) -> dict:
    """Tile and warp choice for a decode step, from shape alone.

    Static by design: the tile never depends on a timing measurement taken at call time, so a given
    shape always compiles to exactly one kernel variant and a steady stream of same-shape calls
    neither retunes nor recompiles.

    How many programs a shape can offer is what decides the tile, because the kernel is a pure state
    stream with no reuse. Three regimes, all measured over the captured shapes; the measurements are
    in ``docs/tuning.md``:

    * Fewer sequence-heads than SMs. The value axis has to be split hard just to reach one program
      per SM, and with barely one wave in flight the shortest dependency chain wins, so the whole
      key axis is covered in a single pass.
    * Between one and roughly eight waves' worth. A narrow value tile supplies the programs; the key
      axis is tiled so several tiles are in flight per program.
    * Many waves' worth. Parallelism is already abundant, so widening the value tile pays off
      instead: each program walks a longer contiguous run of the state, and wave quantization stops
      mattering. Widening past ``V/2`` measured slower everywhere, because at ``BV == V`` the grid
      collapses to one program per sequence-head and the partial last wave costs more than the extra
      contiguity gains.

    Pipeline depth, the L2 eviction policy and the cache modifier all measured within noise of one
    another on every benchmarked shape, so they stay at their plain defaults.
    """
    v_tiles = [b for b in _V_TILES if V % b == 0]
    k_tiles = [b for b in _K_TILES if K % b == 0]
    num_warps = 4
    tile_budget = _TILE_REGS_PER_THREAD * 32 * num_warps

    narrow_v = min(b for b in v_tiles if b >= 32) if any(b >= 32 for b in v_tiles) else v_tiles[-1]
    wide_v = max(b for b in v_tiles if b <= V // 2) if V // 2 >= v_tiles[-1] else v_tiles[-1]

    waves = n_seq_heads / _multiprocessor_count()
    if waves >= 8:
        bv, want_bk = wide_v, tile_budget // wide_v
    elif waves >= 1:
        bv, want_bk = narrow_v, 64
    else:
        bv, want_bk = narrow_v, K

    want_bk = min(want_bk, K, max(1, tile_budget // bv))
    bk = max(b for b in k_tiles if b <= want_bk or b == k_tiles[-1])
    return {
        "BK": bk,
        "BV": bv,
        "num_warps": num_warps,
        "num_stages": 2,
        "eviction": "",
        "cache_modifier": "",
        "value_tile_fast": True,
    }


class FusedRecurrentGLA(nn.Module):
    """GLA fused-recurrent operator with a single-launch decode-step path."""

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
        dims = _decode_dims(q, k, v, gk, scale, initial_state, cu_seqlens)
        if dims is not None:
            return _decode_step(q, k, v, gk, scale, initial_state, output_final_state, dims)
        return fused_recurrent_gla(
            q=q, k=k, v=v, gk=gk,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )


def _decode_dims(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor | None,
    scale: float | None,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> tuple[int, int, int, int, int, int] | None:
    """``(B, T, H, K, V, device_index)`` if the decode kernel may serve this call, else ``None``.

    Deliberately strict: everything it rejects goes to the reference, so a merely unusual call costs
    a delegation rather than risking a wrong answer. The admissible set is a single timestep with an
    incoming state, in ``bfloat16``, contiguous, on the current CUDA device, with ``K`` and ``V``
    multiples of 16 and a plain scalar ``scale``. Notably the kernel assumes the ``[N, H, K, V]``
    state layout the signature implies and does not accept the value-first layout some call sites
    use; fp16 and fp32 activations are delegated even though the arithmetic would work, because the
    routing table this operator is specified against says so; and the fast path declines whenever a
    gradient could be expected of it.
    """
    if cu_seqlens is not None or gk is None or initial_state is None:
        return None
    if not _GATE_MATCHES_REFERENCE:
        return None
    shape = q.shape
    # A single timestep with an incoming state is the whole admissible set; every multi-timestep
    # call is delegated, which is also what keeps this kernel's flattened addressing valid.
    if len(shape) != 4 or shape[1] != 1:
        return None
    B, T, H, K = shape
    if v.dim() != 4:
        return None
    V = v.shape[3]
    # Every extent has to be non-empty: an empty key or value axis gives the reference a zero-width
    # tile and it raises rather than returning anything, so there is no result to agree with.
    if B <= 0 or T <= 0 or H <= 0 or K <= 0 or V <= 0 or K % 16 or V % 16:
        return None
    # The kernels multiply the query by a plain fp32 scalar. Anything else -- a 0-d tensor, a numpy
    # scalar -- would change the kernel's argument type, so it goes to the reference.
    if scale is not None and not isinstance(scale, (int, float)):
        return None
    if (q.dtype is not torch.bfloat16 or k.dtype is not torch.bfloat16
            or v.dtype is not torch.bfloat16 or gk.dtype is not torch.bfloat16):
        return None
    if k.shape != shape or gk.shape != shape or v.shape != (B, T, H, V):
        return None
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and gk.is_contiguous()):
        return None
    if initial_state.dtype is not torch.float32:
        return None
    if initial_state.shape != (B, H, K, V) or not initial_state.is_contiguous():
        return None
    tensors = [q, k, v, gk, initial_state]
    # One CUDA device for everything, and it has to be the active one. `get_device()` returns -1 for
    # a CPU tensor, so this subsumes the CUDA check. The reference enters the first tensor's device
    # context for itself; a specialized launch would instead go to whatever device is current, so a
    # mismatch is delegated rather than mis-launched.
    device = q.get_device()
    if device < 0 or any(t.get_device() != device for t in tensors):
        return None
    if device != torch.cuda.current_device():
        return None
    # The specialized paths are inference-only; let the reference handle anything that could need a
    # backward pass rather than silently dropping the graph.
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        return None
    return B, T, H, K, V, device


def _is_decode_step(q, k, v, gk, initial_state, cu_seqlens, scale=None) -> bool:
    """Whether the decode kernel may serve this call. Exposed for the equivalence tests."""
    return _decode_dims(q, k, v, gk, scale, initial_state, cu_seqlens) is not None


def _allocate_outputs(
    q: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor | None,
    dims: tuple[int, int, int, int, int, int],
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The output leaf and, when requested, the updated-state leaf.

    With ``_FUSE_OUTPUT_ALLOCATION`` both leaves are strided views into one buffer, so the call
    makes a single allocation. The fp32 state comes first, which keeps the bf16 output's offset a
    multiple of 4 bytes and -- because ``V`` is a multiple of 16 -- 16-byte aligned as well.
    """
    B, T, H, K, V = dims[:5]
    if not output_final_state:
        return torch.empty_like(v), None
    if not _FUSE_OUTPUT_ALLOCATION:
        return torch.empty_like(v), torch.empty_like(initial_state)
    state_n = B * H * K * V
    out_n = B * T * H * V
    buf = torch.empty(state_n + (out_n + 1) // 2, dtype=torch.float32, device=q.device)
    h_out = torch.as_strided(buf, (B, H, K, V), (H * K * V, K * V, V, 1))
    o = torch.as_strided(buf.view(torch.bfloat16), (B, T, H, V),
                         (T * H * V, H * V, V, 1), 2 * state_n)
    return o, h_out


def _decode_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    scale: float | None,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    dims: tuple[int, int, int, int, int, int],
    config: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the decode kernel. ``config`` overrides the static tile choice, for offline sweeps."""
    B, T, H, K, V, device = dims
    n_seq_heads = B * H
    # The reference derives the default from the key width, not the query width.
    if scale is None:
        scale = K ** -0.5
    else:
        # Always hand the kernel a Python float. An integer scale would compile to a different
        # kernel signature -- and Triton specializes the integer 1 as a compile-time constant -- so
        # passing one through unchanged would make the launcher cache serve a kernel built for a
        # different scale.
        scale = float(scale)

    o, h_out = _allocate_outputs(q, v, initial_state, dims, output_final_state)
    return _launch_decode(q, k, v, gk, scale, initial_state, o, h_out, dims, config)


def _aligned16(*tensors: torch.Tensor) -> bool:
    bits = 0
    for t in tensors:
        bits |= t.data_ptr()
    return not (bits & 15)


def _launch_decode(q, k, v, gk, scale, initial_state, o, h_out, dims, config):
    B, T, H, K, V, device = dims
    n_seq_heads = B * H
    output_final_state = h_out is not None
    # When the updated state is not wanted, STORE_FINAL_STATE compiles the store away and the
    # destination pointer is never dereferenced; the incoming state stands in for it only so the
    # launch has a valid argument.
    state_out = h_out if h_out is not None else initial_state

    key = None
    if config is None:
        # Compiled kernels are specialized on 16-byte pointer alignment and are loaded per device,
        # so both belong in the key; anything unaligned falls through to the ordinary JIT path,
        # which will specialize correctly for itself.
        key = ("decode", device, n_seq_heads, K, V, output_final_state)
        cached = _LAUNCHER_CACHE.get(key)
        if cached is not None and _aligned16(q, k, v, gk, initial_state, o, state_out):
            launch, cfg, nv = cached
            launch(q, k, v, gk, initial_state, o, state_out, scale, n_seq_heads,
                   K, V, cfg["BK"], cfg["BV"], nv, output_final_state,
                   cfg["eviction"], cfg["cache_modifier"], cfg["value_tile_fast"],
                   stream=driver.active.get_current_stream(
                       driver.active.get_current_device()))
            return o, h_out
        cfg = _CONFIG_CACHE.get(("decode", device, n_seq_heads, K, V))
        if cfg is None:
            cfg = _CONFIG_CACHE[("decode", device, n_seq_heads, K, V)] = _decode_config(
                n_seq_heads, K, V)
    else:
        cfg = config

    nv = V // cfg["BV"]
    compiled = _gla_decode_step_kernel[(n_seq_heads * nv,)](
        q, k, v, gk, initial_state, o, state_out, scale, n_seq_heads,
        K=K, V=V, BK=cfg["BK"], BV=cfg["BV"], NV=nv,
        STORE_FINAL_STATE=output_final_state,
        STATE_EVICTION=cfg["eviction"],
        STATE_CACHE_MODIFIER=cfg["cache_modifier"],
        VALUE_TILE_FAST=cfg["value_tile_fast"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    if (key is not None and compiled is not None
            and _aligned16(q, k, v, gk, initial_state, o, state_out)):
        # Remember the launcher this shape compiled to. Later calls at the same shape skip argument
        # binding, specialization-key computation and cache lookup, which measure as roughly half
        # the per-call dispatch cost -- and dispatch is what the small-state shapes are bound by.
        _LAUNCHER_CACHE[key] = (compiled[(n_seq_heads * nv, 1, 1)], cfg, nv)
    return o, h_out
