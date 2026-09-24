"""KDA layer specialised for the single-sequence prefill regime.

The baseline forward is host-bound by 5-9x on every benched shape below 16k
tokens (measured: n=1 1171 us of wall against 125 us of GPU work; n=64
1297/166). Almost all of that is Python and launch machinery, so this subclass
attacks launch count and host work rather than arithmetic. Per forward at n=64 it
takes kernel launches from 42 to 21, of which Triton launches go from 13 to 9:

* the device-to-host sync in the recurrent-state preparation is gone -- when the
  metadata says no sequence carries state, the gathered initial state is
  provably all-zero and ``initial_state=None`` is bit-identical (see
  ``_forward_single_prefill``),
* one Triton kernel replaces the three depthwise causal convolutions, their silu,
  the q/k l2 norm, the beta sigmoid and the conv-state tail write
  (``_fused_input_stage_kernel``),
* ``core_attn_out`` is no longer allocated-and-zeroed just to be overwritten,
* ``einops.rearrange`` is replaced by ``view``/``reshape`` (3.8 us -> 0.8 us of
  host per site),
Everything outside that regime -- decode, mixed batches, multi-sequence prefill,
carried state, quantized construction, graph capture, and any head width, conv
width or activation dtype outside the verified set -- runs the baseline's own code
through ``super()``. The gate is evaluated on host-side metadata only, so testing
it never costs the sync it removes. A second, narrower gate covers what only the
live tensors can answer (packed weight present, conv-cache layout, offset range);
what it rejects keeps the sync-free prefill and falls back to the three
convolution launches.

Two further changes were prototyped and dropped: merging ``f_a_proj`` and
``g_a_proj`` into one GEMM, and reaching ``o_norm``'s kernel without its Python
wrapper. Measured as completed CUDA-event latency rather than enqueue cost, the
wrapper bypass is worth nothing at any shape and the GEMM merge is worth ~0.2% at
n=16384 and nothing below it -- not enough to carry a derived weight buffer.
``profile/increments_ab.txt`` has the numbers.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fastkernels.tasks.baseline.L1.kda import chunk_kda_with_fused_gate
from fastkernels.tasks.baseline.L2.kimi_delta_attention import (
    KimiDeltaAttention as _BaselineKimiDeltaAttention,
)

__all__ = ["KimiDeltaAttention"]

_PACKED_CONV_WEIGHT = "_packed_conv_weight"

# The regime this layer's fast path is verified over, and nothing wider. The
# kernel is dtype- and width-generic, but "it should work" is not evidence, and
# admitting an untested width turned out to be actively wrong: at conv_size=5 the
# fused stage disagrees with the baseline, because
# ``_causal_conv1d_fwd_kernel``'s compute loop only has branches for
# ``KERNEL_WIDTH`` 2, 3 and 4 -- at width 5 it accumulates ``col0 * w_col0`` five
# times, never reads the current token, and never rotates its registers (measured:
# max|d| 5.66 against a plain depthwise reference, while widths 2-4 are exact).
# The fused kernel walks ``static_range(CONV_SIZE)`` correctly, so it differs -- by
# being right, which does not help when the contract is equivalence.
#
# So: the widths and head dims are exactly what ``tools/check_fused_input_stage.py``
# covers. bf16 is the benched dtype; fp16 is admitted because the fallback matrix
# has a real end-to-end positive case for it. fp32 is refused and runs the
# baseline.
_SUPPORTED_ACT_DTYPES = (torch.bfloat16, torch.float16)
_SUPPORTED_HEAD_DIMS = (128,)
_SUPPORTED_CONV_SIZES = (4,)


@triton.jit
def _fused_input_stage_kernel(
    qkvb,               # [n, COLS] activation dtype: q | k | v | beta by column
    packed_w,           # [3, CONV_SIZE, PROJ] indexed [which, tap, channel]
    out,                # [3, n, HEADS, HEAD_DIM] activation dtype
    beta_out,           # [1, n, HEADS] fp32
    q_state,            # [slots, STATE_LEN, PROJ] contiguous, per projection
    k_state,
    v_state,
    state_indices,      # [>=1] int32; element 0 is this sequence's cache slot
    n_tokens,
    PROJ: tl.constexpr,
    HEADS: tl.constexpr,
    HEADS_PAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CONV_SIZE: tl.constexpr,
    STATE_LEN: tl.constexpr,
    STATE_PAD: tl.constexpr,
    COLS: tl.constexpr,
    EPS: tl.constexpr,
    BM: tl.constexpr,
):
    """One depthwise causal conv + silu + optional l2 norm, per (token block, head,
    projection), plus the beta sigmoid and the conv-state tail.

    Replaces three ``causal_conv1d_fn`` launches, two ``l2norm_fwd`` launches and
    the beta sigmoid with one launch. The layer is host-bound by 5-9x below 16k
    tokens, so the launch count is the point; the arithmetic is unchanged.
    """
    i_t = tl.program_id(0)
    i_h = tl.program_id(1)
    i_w = tl.program_id(2)                       # 0 = q, 1 = k, 2 = v

    lane = tl.arange(0, HEAD_DIM)
    # 32-bit offsets throughout. Profiling showed 66% of issue slots going to the
    # integer/logic pipeline against only 10% of FP32 peak, i.e. the kernel was
    # spending its time on address arithmetic rather than on the convolution;
    # widening these to int64 doubles that work. The caller guarantees every
    # offset here fits in int32 (see ``_fused_input_stage_ok``).
    rows = i_t * BM + tl.arange(0, BM)
    row_ok = rows < n_tokens
    col = i_w * PROJ + i_h * HEAD_DIM + lane
    w_base = packed_w + i_w * CONV_SIZE * PROJ + i_h * HEAD_DIM + lane

    # ``beta`` is produced outside the convolution in the baseline, so it is
    # written whatever the cache slot says -- see the null-slot note below.
    if i_w == 0:
        if i_h == 0:
            bcol = tl.arange(0, HEADS_PAD)
            bmask = row_ok[:, None] & (bcol < HEADS)[None, :]
            b_raw = tl.load(qkvb + rows[:, None] * COLS + 3 * PROJ + bcol[None, :],
                            mask=bmask, other=0.0)
            # fp32 sigmoid, matching ``raw_beta.float().sigmoid()``; doing it in
            # bf16 is visibly different.
            tl.store(beta_out + rows[:, None] * HEADS + bcol[None, :],
                     tl.sigmoid(b_raw.to(tl.float32)), mask=bmask)

    # Slot 0 is the reserved null cache line. ``causal_conv1d`` returns from the
    # whole program on it, writing neither output nor state, so this kernel does
    # the same: the output stays as allocated (``torch.empty``, exactly as the
    # baseline's conv allocates it) and the cache is left alone. The slot lives on
    # the device, so the host-side gate cannot decide this.
    slot = tl.load(state_indices)
    if slot != 0:
        acc = tl.zeros((BM, HEAD_DIM), dtype=tl.float32)
        # Taps accumulated in order, one row-window load per tap. The four loads
        # cover only BM + CONV_SIZE - 1 distinct rows and repeat the same
        # addresses within the program, so DRAM traffic stays at ~(BM+3)/BM of
        # ideal; ``.ca`` asks for the reuse to be served by L1, which is what the
        # baseline's own halo loads request.
        for j in tl.static_range(CONV_SIZE):
            t = rows - (CONV_SIZE - 1) + j
            x = tl.load(qkvb + t[:, None] * COLS + col[None, :],
                        mask=row_ok[:, None] & (t >= 0)[:, None], other=0.0,
                        cache_modifier=".ca")
            w = tl.load(w_base + j * PROJ)
            # Promotion is per-operand and deliberately not pre-normalised: with
            # bf16 weights the product rounds to bf16 before this fp32 add, with
            # fp32 weights it does not. Packing in the source dtype reproduces
            # whichever regime the caller is in.
            acc += x * w[None, :]

        # Division form, not ``acc * sigmoid(acc)``: the baseline writes
        # ``acc / (1 + exp(-acc))`` and the two differ in fp32.
        acc = acc / (1.0 + tl.exp(-acc))
        if i_w != 2:
            # q and k only. The baseline stores the conv result as bf16 and then
            # runs l2norm over *that*, so the accumulator has to be rounded
            # through the activation dtype before the reduction or the norm is
            # computed from more precision than the reference had.
            acc = acc.to(out.dtype.element_ty).to(tl.float32)
            acc = acc * tl.rsqrt(tl.sum(acc * acc, axis=1) + EPS)[:, None]

        o_off = (i_w * n_tokens + rows[:, None]) * PROJ + i_h * HEAD_DIM + lane[None, :]
        tl.store(out + o_off, acc.to(out.dtype.element_ty), mask=row_ok[:, None])

        # Conv-state tail, folded in at no extra launch. The cache is allocated
        # [slot, tap, channel] contiguous and the layer hands the conv a
        # transpose(-1, -2) view of it, so the write is channel-contiguous with
        # the tap outermost. It stores the *raw pre-activation* rows -- never the
        # activated accumulator -- left-padded with exact zeros when the sequence
        # is shorter than the window, which is what both of the baseline's state
        # branches collapse to.
        if i_t == tl.cdiv(n_tokens, BM) - 1:
            taps = tl.arange(0, STATE_PAD)
            ts = n_tokens - STATE_LEN + taps
            keep = (taps < STATE_LEN)[:, None]
            tail = tl.load(qkvb + ts[:, None] * COLS + col[None, :],
                           mask=keep & (ts >= 0)[:, None], other=0.0)
            s_off = (slot * STATE_LEN + taps[:, None]) * PROJ + i_h * HEAD_DIM + lane[None, :]
            if i_w == 0:
                tl.store(q_state + s_off, tail, mask=keep)
            elif i_w == 1:
                tl.store(k_state + s_off, tail, mask=keep)
            else:
                tl.store(v_state + s_off, tail, mask=keep)


# One tile for every shape, chosen by measurement (``tools/probe_fused_tiles.py``,
# 1000 iterations per point, recorded in ``profile/fused_tiles.txt``).
#
# Specialising the block height by sequence length looks obviously right and is
# not: below roughly a thousand tokens this kernel costs ~35 us whatever the tile,
# because that is the host cost of one Triton launch and the GPU work is under a
# microsecond (the DRAM floor at n=64 is 0.5 us). BM=1 and BM=64 measure within
# noise of each other at n=1. So the padded arithmetic a large tile does on a
# 1-token sequence is hidden behind submission latency, and a per-shape table
# would only add compiled variants. Note the narrower claim: that cost is hidden,
# not absent, and could resurface under many concurrent streams or graph replay.
#
# At n=16384, where the GPU work finally dominates, the ordering is
# BM=32/8 warps 235.7 us < BM=64/8 240.9 < BM=16/4 266.4 < BM=8/4 305.4, and
# BM=32 with 8 warps compiles to 46 registers and no spills, against 76 for
# BM=64/8 and 167 for BM=64/4. (Note the kernel is issue-bound there, not
# bandwidth-bound: DRAM throughput is 30% while issue slots are 87% busy.)
#
# Between 443 and 16384 tokens the crossover from launch-bound to GPU-bound has
# not been measured; no benched shape lands there.
_FUSED_BLOCK_TOKENS = 32
_FUSED_NUM_WARPS = 8


def fused_input_stage(
    qkvb: torch.Tensor,
    packed_w: torch.Tensor,
    q_state: torch.Tensor,
    k_state: torch.Tensor,
    v_state: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    conv_size: int,
    block_tokens: int | None = None,
    num_warps: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """q, k, v as [1, n, H, D] and beta as [1, n, H] fp32, conv states updated.

    q, k and v share one allocation so the kernel can select its output slice
    with ``program_id``, which is a runtime value and cannot index a tuple of
    tensors. ``torch.empty`` rather than ``zeros``: for the reserved null cache
    slot the baseline's conv also leaves its output buffer untouched.
    """
    proj = num_heads * head_dim
    state_len = conv_size - 1
    tile = block_tokens or _FUSED_BLOCK_TOKENS
    warps = num_warps or _FUSED_NUM_WARPS

    out = torch.empty((3, num_tokens, num_heads, head_dim),
                      dtype=qkvb.dtype, device=qkvb.device)
    beta = torch.empty((1, num_tokens, num_heads),
                       dtype=torch.float32, device=qkvb.device)
    _fused_input_stage_kernel[(triton.cdiv(num_tokens, tile), num_heads, 3)](
        qkvb, packed_w, out, beta, q_state, k_state, v_state, state_indices,
        num_tokens,
        PROJ=proj,
        HEADS=num_heads,
        HEADS_PAD=triton.next_power_of_2(num_heads),
        HEAD_DIM=head_dim,
        CONV_SIZE=conv_size,
        STATE_LEN=state_len,
        STATE_PAD=triton.next_power_of_2(state_len),
        COLS=qkvb.shape[-1],
        EPS=1e-6,
        BM=tile,
        num_warps=warps,
    )
    qkv = out.unsqueeze(1)
    return qkv[0], qkv[1], qkv[2], beta


class KimiDeltaAttention(_BaselineKimiDeltaAttention):
    """Baseline KDA with a fast path for one prefill sequence and no carried state.

    Subclassing is load-bearing, not stylistic: the benchmark harness locates the
    KDA layer with ``isinstance`` against the baseline class, and a candidate that
    merely reimplements the interface is reported as *skipped* -- which the
    harness still counts as an overall pass, so it would look green while
    measuring nothing.
    """

    def __init__(self, config, layer_idx: int, quant_config: dict | None = None):
        super().__init__(config, layer_idx, quant_config)
        # Declared here but deliberately left empty: the packing is derived from
        # the conv weights, which at construction time are still ``torch.empty``
        # garbage. Registering the name up front (rather than attaching the
        # tensor later) keeps it a buffer rather than a plain attribute, and
        # ``persistent=False`` keeps it out of ``state_dict()`` so the harness's
        # baseline -> candidate weight share stays clean. A ``Parameter`` would be
        # worse than useless: the harness overwrites any float parameter whose
        # amax falls outside [1e-6, 1e4] with ``normal_(0, 0.02)``.
        self.register_buffer(_PACKED_CONV_WEIGHT, None, persistent=False)
        self._packed_conv_signature: tuple | None = None

    # -- derived conv weights -------------------------------------------------

    def _conv_weight_sources(self):
        return (self.q_conv1d.weight, self.k_conv1d.weight, self.v_conv1d.weight)

    @staticmethod
    def _conv_weight_signature(weights) -> tuple:
        """Host-only fingerprint that changes whenever the packing goes stale.

        ``_version`` catches in-place writes (``load_state_dict``'s ``copy_``, the
        harness's parameter sanitizer), while dtype/device/shape catch the whole
        tensor being replaced -- which is what ``_prepare_module`` does when it
        downcasts these fp32 parameters to the case dtype.
        """
        return tuple(
            (w.dtype, w.device, tuple(w.shape), w._version) for w in weights
        )

    def _packed_conv_weight_or_none(self) -> torch.Tensor | None:
        """The packed conv weight, rebuilt if the sources moved under it.

        Returns ``None`` when the three projections do not agree on dtype, device
        or shape, which is the caller's signal to use the unfused conv path.
        """
        weights = self._conv_weight_sources()
        signature = self._conv_weight_signature(weights)
        packed = getattr(self, _PACKED_CONV_WEIGHT)
        if packed is not None and signature == self._packed_conv_signature:
            return packed
        first = weights[0]
        if any(w.dtype != first.dtype or w.device != first.device
               or w.shape != first.shape for w in weights[1:]):
            return None
        if first.dim() != 3 or first.shape[1] != 1:
            return None
        channels, _, taps = first.shape
        # [which, tap, channel]: each tap is one contiguous channel-major row, so
        # a program covering 128 channels of one head loads 256 B per tap. Kept in
        # the source dtype -- these are declared fp32 and only bf16 because the
        # harness downcasts them, and Triton's per-operand promotion differs
        # between the two (bf16 x bf16 rounds the product before the fp32
        # accumulate, bf16 x fp32 does not), so pre-casting would break one
        # regime.
        packed = torch.stack([
            w.detach().reshape(channels, taps).t().contiguous() for w in weights
        ])
        setattr(self, _PACKED_CONV_WEIGHT, packed)
        self._packed_conv_signature = signature
        return packed

    def process_weights_after_loading(self) -> None:
        """The only hook the harness runs after the real weights land."""
        hook = getattr(super(), "process_weights_after_loading", None)
        if callable(hook):
            hook()
        self._packed_conv_weight_or_none()

    def _single_prefill_ok(self, hidden_states, state_view, meta) -> bool:
        """Host-only gate for the fast path.

        Reads scalars, dtypes, tensor shapes and ``is None`` -- never the *value*
        of a device tensor. Inspecting ``meta.has_initial_state`` elementwise here
        would reintroduce exactly the stream sync this path exists to remove,
        which is why the decision is taken from ``any_have_initial_state``
        instead. That field defaults to ``True`` in ``KimiLinearMetadata``, so a
        producer that does not fill it in falls back -- the safe direction.
        """
        if state_view is None or meta is None:
            return False
        # ``_ps_local`` and the packed projection only exist unquantized, and the
        # custom-op path has its own dispatch surface.
        if self.qkvb_proj is None or self._use_custom_op:
            return False
        if hidden_states.dim() != 2:
            return False
        if hidden_states.dtype not in _SUPPORTED_ACT_DTYPES:
            return False
        # Config-level shape support, checked here rather than deeper down so that
        # anything unsupported runs the baseline's whole forward rather than a
        # partly-specialised path.
        if self.head_dim not in _SUPPORTED_HEAD_DIMS:
            return False
        if self.conv_size not in _SUPPORTED_CONV_SIZES:
            return False
        num_tokens = hidden_states.size(0)
        if num_tokens <= 0:
            return False
        if meta.num_prefills != 1:
            return False
        if meta.num_decodes != 0 or meta.num_decode_tokens != 0:
            return False
        if meta.num_prefill_tokens != num_tokens:
            return False
        if meta.num_actual_tokens != num_tokens:
            return False
        if meta.any_have_initial_state:
            return False
        state_indices = meta.non_spec_state_indices_tensor
        cu_seqlens = meta.non_spec_query_start_loc
        if state_indices is None or cu_seqlens is None:
            return False
        if state_indices.numel() < 1:
            return False
        # ``num_prefills == 1`` is a host scalar a producer can set inconsistently
        # with the offsets, so the offsets are checked rather than trusted: the
        # fused stage convolves the whole token range as one sequence (exact zeros
        # before position 0) while ``chunk_kda_with_fused_gate`` splits on whatever
        # ``cu_seqlens`` says, so the two stages would be describing different
        # batches. Measured, that does not produce a wrong answer -- the
        # recurrent-state write-back raises on the shape mismatch, exactly as the
        # baseline does, since one state index cannot receive a two-row final
        # state. This turns an identical crash into a clean delegation. Comparing
        # the shape is host-side; comparing the values would not be.
        if cu_seqlens.dim() != 1 or cu_seqlens.numel() != 2:
            return False
        # Under capture the baseline zeroes the gathered state unconditionally,
        # ignoring ``has_initial_state``; rather than argue that the two agree,
        # hand capture back to the baseline.
        if torch.cuda.is_current_stream_capturing():
            return False
        return True

    def _fused_input_stage_ok(self, qkvb, state_view, packed) -> bool:
        """Host-only gate for the fused stage, inside the already-gated fast path.

        ``_single_prefill_ok`` already settled the regime (dtype, head width,
        conv width, one sequence). What is left here is what only the live tensors
        can answer: whether the packed weight exists, whether the conv cache has
        the layout this kernel writes, and whether the flattened offsets still fit
        in int32. Anything it rejects keeps the sync-free prefill and falls back to
        the three ``causal_conv1d_fn`` launches.
        """
        if packed is None:
            return False
        if not qkvb.is_contiguous():
            return False
        # The kernel indexes with int32 to keep address arithmetic off the
        # critical path, so every offset it forms must fit. The largest are the
        # activation read (n * cols) and the packed output write (3 * n * proj).
        num_tokens, cols = qkvb.shape
        proj = self.local_num_heads * self.head_dim
        if max(num_tokens * cols, 3 * num_tokens * proj) >= 2 ** 31:
            return False
        states = (state_view.q_conv_state, state_view.k_conv_state,
                  state_view.v_conv_state)
        want = (self.conv_size - 1, proj)
        for st in states:
            # The conv cache is allocated [slot, tap, channel] contiguous; the
            # kernel writes channel-contiguous with the tap outermost and would
            # scatter into the wrong places under any other layout.
            if st.dim() != 3 or tuple(st.shape[1:]) != want:
                return False
            if not st.is_contiguous():
                return False
            # The baseline casts the conv input to the cache dtype before
            # convolving, so a differing cache dtype changes the arithmetic.
            if st.dtype != qkvb.dtype:
                return False
        return True

    def _prefill_cu_seqlens(self, meta) -> torch.Tensor:
        """int32 sequence offsets, as the kernels document they want.

        ``causal_conv1d_fn`` and the FLA chunk kernels both specify
        ``query_start_loc: (batch + 1) int32``. The baseline's comment claims int64
        offsets silently change the result (amax 0.042 against 0.0005); measured on
        this library version it does not reproduce -- ``tools/check_host_cleanup.py``
        finds the two bit-identical across one to three sequences on both consumers
        of the field. int32 is passed because it is the documented contract and
        because ``query_start_loc_int32`` already carries it precomputed, so the
        cast happens once per step rather than once per layer -- not because a
        divergence was observed here.
        """
        cu = meta.query_start_loc_int32
        if cu is None or cu.dtype is not torch.int32:
            cu = meta.non_spec_query_start_loc.to(torch.int32)
        return cu

    def _forward_single_prefill(self, hidden_states, state_view, meta):
        num_tokens = hidden_states.size(0)
        heads, head_dim = self.local_num_heads, self.head_dim
        proj = self._ps_local

        qkvb = self.qkvb_proj(hidden_states)
        pf_state_indices = meta.non_spec_state_indices_tensor[:1]

        packed = self._packed_conv_weight_or_none()
        fused = self._fused_input_stage_ok(qkvb, state_view, packed)
        if fused:
            # One launch for the three convs, the silu, the q/k l2 norm, the beta
            # sigmoid and the conv-state tail.
            q, k, v, beta = fused_input_stage(
                qkvb, packed,
                state_view.q_conv_state, state_view.k_conv_state,
                state_view.v_conv_state, pf_state_indices,
                num_tokens=num_tokens, num_heads=heads, head_dim=head_dim,
                conv_size=self.conv_size,
            )
        else:
            q_proj_states = qkvb[..., :proj]
            k_proj_states = qkvb[..., proj:2 * proj]
            v_proj_states = qkvb[..., 2 * proj:3 * proj]
            # sigmoid in fp32, matching the baseline's ``.float().sigmoid()``; the
            # ``.float()`` already materialises a contiguous copy.
            beta = qkvb[..., 3 * proj:].float().sigmoid().unsqueeze(0)
            q_conv_weights = self.q_conv1d.weight.view(proj, self.conv_size)
            k_conv_weights = self.k_conv1d.weight.view(proj, self.conv_size)
            v_conv_weights = self.v_conv1d.weight.view(proj, self.conv_size)
            q = self._run_conv_prefill(
                q_proj_states, state_view.q_conv_state, q_conv_weights, meta)
            k = self._run_conv_prefill(
                k_proj_states, state_view.k_conv_state, k_conv_weights, meta)
            v = self._run_conv_prefill(
                v_proj_states, state_view.v_conv_state, v_conv_weights, meta)
            # The conv writes a channel-last buffer and hands back a [n, proj]
            # transpose of it, which is contiguous, so these are free reshapes.
            q = q.reshape(1, num_tokens, heads, head_dim)
            k = k.reshape(1, num_tokens, heads, head_dim)
            v = v.reshape(1, num_tokens, heads, head_dim)

        # Raw gate projection, left ungated: the chunk kernel applies
        # A_log/dt_bias and the softplus itself in fp32 registers.
        raw_g = self.f_b_proj(self.f_a_proj(hidden_states)).view(
            1, num_tokens, heads, head_dim)
        g2 = self.g_b_proj(self.g_a_proj(hidden_states)).view(
            num_tokens, heads, head_dim)
        # ``initial_state=None`` rather than a gathered zero state. With
        # ``any_have_initial_state`` false every slot the chunk kernel would read
        # is zero, and ``chunk_gated_delta_rule_fwd_h`` maps ``h0=None`` to
        # ``USE_INITIAL_STATE=False``, whose kernel starts from ``tl.zeros`` and
        # only ever ``+=``s h0 -- so the two are bit-identical. Building the
        # zero state instead costs a boolean mask, a ``nonzero()`` with its
        # device-to-host copy, an ``index_put`` and a gather: 97 us of host and a
        # full queue drain, every call.
        core_attn_out, last_state = chunk_kda_with_fused_gate(
            q=q,
            k=k,
            v=v,
            raw_g=raw_g,
            beta=beta,
            A_log=self.A_log,
            g_bias=self.dt_bias,
            initial_state=None,
            output_final_state=True,
            # The flag is a misnomer: it selects a *pre-pass* of two separate
            # ``l2norm_fwd`` launches, so leaving it on after the fused stage has
            # already normalised q and k would both double-normalise and pay two
            # launches for it.
            use_qk_l2norm_in_kernel=not fused,
            cu_seqlens=self._prefill_cu_seqlens(meta),
        )
        # Part of the operator contract even though the benchmark never reads it
        # back: the next step's prefill continuation needs it.
        state_view.recurrent_state[pf_state_indices] = last_state

        # No zero-filled staging buffer and no slice-copy: with no decode tokens
        # the chunk output *is* the whole [1, n, H, D] result. Note it aliases the
        # ``v`` buffer -- ``chunk_gla_fwd_o_gk`` writes its output there -- so
        # ``v`` must not be read after this point.
        out = self.o_norm(core_attn_out, g2)
        return self.o_proj(out.view(num_tokens, heads * head_dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        state_manager=None,
    ) -> torch.Tensor:
        state_view, meta = self._get_state()
        if not self._single_prefill_ok(hidden_states, state_view, meta):
            return super().forward(hidden_states, state_manager)
        self._ensure_triton_allocator(hidden_states.device)
        return self._forward_single_prefill(hidden_states, state_view, meta)
