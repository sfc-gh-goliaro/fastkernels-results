"""Qwen3-Next Gated Delta Net linear attention, restructured for eager latency.

Same ``__init__``/``forward`` contract as the baseline; the difference is where the
work happens.

The baseline issues thirteen kernels per prefill call and spends roughly 32 us of
Python per launch, so at 60 tokens a call costs ~467 us of which the GPU is busy
for ~69. At the other end, 16384 tokens is bandwidth-bound, and three consecutive
kernels -- the projection deinterleave, the causal conv, and the post-conv prep --
each make a full pass over the same 8192-wide activation for ~429 us of a ~2026 us
call. The two GEMMs already run at ~98% of achievable bf16 throughput and the
chunked GDN scan is parallelism-limited by its own algorithm, so those are the
floor.

Both ends are served by the same restructuring:

* **The input projection's weight rows are permuted once**, at weight-preparation
  time, into the order the consumers want. A linear layer's output columns permute
  with its weight rows, so the per-token gather the baseline performs on every call
  becomes a one-off gather on the weight, and the four consumers (``mixed_qkv``,
  ``z``, ``b``, ``a``) become zero-copy strided column slices of the single
  projection output. Every kernel that reads them therefore takes an explicit row
  stride -- and the gated output norm in particular must take the *grouped* route
  through ``layer_norm_fwd_kernel``, because ``z.reshape(-1, 128)`` on a strided
  view is not expressible as a view and would silently materialize the copy the
  removed deinterleave was supposed to save.
* **The remaining stages are launched directly** rather than through their Python
  wrappers, into capacity-managed preallocated buffers. The wrappers' cost is
  argument validation, ``.contiguous()`` sweeps, uncached device-property queries
  and output allocation -- all of it per call, none of it needed here.

Delegation, not duplication: everything outside the fast path -- an unprepared
module, a non-Blackwell device, a missing FlashInfer kernel -- routes back to the
inherited implementation, which is the reference this module is checked against.

Because the projection rows no longer sit in checkpoint order, ``state_dict()``
after preparation would export a permuted ``in_proj_qkvz.weight``, and a
subsequent ``load_state_dict()`` of checkpoint-order rows would write them into
permuted positions. The baseline already documents that reloading weights after
preparation is unsupported; here that is enforced rather than documented, by a
load hook that raises. Preparation itself is idempotent and atomic: it rebinds
nothing until every buffer it needs exists, so a failure part-way leaves the
module unprepared and still numerically identical to the baseline.

Single-stream, single-threaded: the per-instance scratch makes this module
non-reentrant, matching the persistent-scratch pattern the ``mamba2_mixer`` and
``moe_sum`` baselines already use here. The baseline holds no per-call scratch and
so is trivially reentrant; this is a deliberate narrowing of that contract.
"""

from __future__ import annotations

import math

import torch
import triton

from fastkernels.infra.context import get_context
from fastkernels.infra.cuda_ext import load_op
from fastkernels.infra.mamba_state import compute_causal_conv1d_metadata
from fastkernels.tasks.baseline.L1.causal_conv1d import (
    _causal_conv1d_fwd_kernel,
    causal_conv1d_update as _vllm_causal_conv1d_update,
)
from fastkernels.tasks.baseline.L1.gated_delta_rule import (
    _fused_post_conv_kernel,
    fused_sigmoid_gating_delta_rule_update as _vllm_fused_sigmoid_gating_update,
)
from fastkernels.tasks.baseline.L1.rms_norm_gated import layer_norm_fwd_kernel
from fastkernels.tasks.baseline.L2.qwen3_next_gdn_attention import (
    Qwen3NextGDNAttention as _BaselineGDN,
    _split_conv_qkv,
)

__targets__ = ["Qwen3NextGDNAttention"]

# vLLM's varlen conv kernel constants. ``NULL_BLOCK_ID`` marks a padded cache
# slot the kernel must skip; ``PAD_SLOT_ID`` marks a padded sequence.
_CONV_BLOCK_M = 8
_CONV_BLOCK_N = 256
_PAD_SLOT_ID = -1
_NULL_BLOCK_ID = 0

_L2NORM_EPS = 1e-6
_SOFTPLUS_THRESHOLD = 20.0


class Qwen3NextGDNAttention(_BaselineGDN):
    """Gated Delta Net linear attention for Qwen3-Next."""

    def __init__(
        self,
        hidden_size: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        layer_idx: int,
        conv_kernel_size: int = 4,
        rms_norm_eps: float = 1e-6,
        reduce_output: bool = True,
    ):
        super().__init__(
            hidden_size, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
            layer_idx, conv_kernel_size, rms_norm_eps, reduce_output,
        )
        # Set once preparation has installed the packed row order. Until then the
        # module is byte-for-byte the baseline and the forward delegates.
        self._packed_projection = False
        self._in_proj_wt: torch.Tensor | None = None
        self._out_proj_wt: torch.Tensor | None = None
        self._norm_weight_packed: torch.Tensor | None = None
        self._scratch: dict[str, torch.Tensor] = {}
        self._sm_count = 0
        # Signature of the last metadata whose *contents* were validated, so the
        # one host read those checks need happens per distinct step rather than
        # per call. Includes each tensor's version counter, so an in-place edit
        # invalidates it.
        self._metadata_signature: tuple | None = None
        # Whether the validated plan contains a zero-length sequence, which the
        # recurrence leaves no output state for.
        self._empty_sequence_present = False
        self._gdn_scale = 1.0 / math.sqrt(self.head_k_dim)
        self._sm100_gdn = None
        self._post_recurrence = None
        self._pre_recurrence = None
        # When set to a tensor, the fused pre-recurrence kernel also writes the
        # post-conv activations there so they can be compared against the stage
        # they replace. Left None in production, where they never reach memory.
        self._conv_tap: torch.Tensor | None = None
        self._register_load_state_dict_pre_hook(self._reject_reload_after_packing)

    # -- weight preparation ------------------------------------------------

    def _projection_row_order(self, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Row indices that put both projections in packed-consumer order.

        ``in_proj_qkvz`` emits one group per K head -- ``[q(K) k(K) v(VP*V)
        z(VP*V)]`` -- and ``in_proj_ba`` the same way as ``[b(VP) a(VP)]``. The
        consumers want ``[q_all | k_all | v_all | z_all]`` and ``[b_all | a_all]``.
        Both permutations are block-diagonal within their own half of the merged
        buffer, so the stacked result is exactly ``[q k v z b a]`` and neither half
        has to leave the one buffer the baseline already builds -- a duplicate
        would cost 25 MiB per layer.
        """
        h, k = self.local_k_heads, self.head_k_dim
        v, vp = self.head_v_dim, self.v_per_k
        group = 2 * k + 2 * vp * v
        heads = torch.arange(h, device=device, dtype=torch.int64) * group
        dk = torch.arange(k, device=device, dtype=torch.int64)
        dv = torch.arange(vp * v, device=device, dtype=torch.int64)
        rows_qkvz = torch.cat([
            (heads[:, None] + dk[None, :]).reshape(-1),
            (heads[:, None] + k + dk[None, :]).reshape(-1),
            (heads[:, None] + 2 * k + dv[None, :]).reshape(-1),
            (heads[:, None] + 2 * k + vp * v + dv[None, :]).reshape(-1),
        ])
        pair = torch.arange(h, device=device, dtype=torch.int64) * (2 * vp)
        dp = torch.arange(vp, device=device, dtype=torch.int64)
        rows_ba = torch.cat([
            (pair[:, None] + dp[None, :]).reshape(-1),
            (pair[:, None] + vp + dp[None, :]).reshape(-1),
        ])
        return rows_qkvz, rows_ba

    def process_weights_after_loading(self) -> None:
        """Build the merged projection directly in packed row order.

        The baseline stacks ``in_proj_ba``'s rows under ``in_proj_qkvz``'s so both
        run as one GEMM; this additionally reorders the rows so the GEMM's output
        needs no deinterleave at all. Building the buffer in final order rather
        than permuting an existing one avoids an overlapping gather, and doing it
        after loading rather than in ``__init__`` matters for the reason the
        baseline gives: the loader moves the whole module with
        ``model.to(device, dtype)``, which reassigns every parameter's storage and
        would leave an earlier view dangling.

        Two phases, and the split is load-bearing. Everything fallible -- the
        allocation, the two gathers, the replicated norm weight, the device query,
        the dispatch binding and the extension build -- happens first, into locals.
        Only once all of it has succeeded does the second phase rebind the two
        parameter storages and publish the fields, ending with the prepared flag.
        A raise anywhere in the first phase therefore leaves the projection in
        checkpoint order with its original storage, which is what makes the
        inherited fallback correct rather than silently wrong: that path applies
        the baseline deinterleave, which is only right for unpermuted rows. It
        also leaves the module retryable, because the idempotence guard reads the
        prepared flag rather than a field the first phase could have set.
        """
        if self._packed_projection:
            return
        qkvz = self.in_proj_qkvz.weight
        ba = self.in_proj_ba.weight
        n_qkvz = qkvz.shape[0]

        # --- phase one: everything that can fail, into locals ----------------
        rows_qkvz, rows_ba = self._projection_row_order(qkvz.device)
        merged = torch.empty(
            n_qkvz + ba.shape[0], qkvz.shape[1],
            dtype=qkvz.dtype, device=qkvz.device,
        )
        torch.index_select(qkvz.data, 0, rows_qkvz, out=merged[:n_qkvz])
        torch.index_select(ba.data, 0, rows_ba, out=merged[n_qkvz:])
        # The grouped gated norm reads one weight slice per value head, so the
        # per-head [head_v_dim] norm weight is replicated once here instead of
        # being broadcast on every call.
        norm_packed = (self.norm.weight.detach().float()
                       .repeat(self.local_v_heads).contiguous())
        # Transposes are cached rather than rebuilt per call, and detached so the
        # GEMMs can take an output parameter. The loader has already moved the
        # module by the time this runs, so these views cannot dangle.
        in_proj_wt = merged.t()
        out_proj_wt = self.out_proj.weight.detach().t()
        sm_count = torch.cuda.get_device_properties(
            qkvz.device.index).multi_processor_count
        sm100_gdn = self._resolve_gdn_dispatch(qkvz.device)
        ext = self._load_extension()
        post_recurrence = getattr(ext, "gdn_post_recurrence", None)
        pre_recurrence = getattr(ext, "gdn_pre_recurrence", None)

        # --- phase two: commit; nothing below this line can raise ------------
        self._qkvz_dim = n_qkvz
        self.in_proj_qkvz.weight.data = merged[:n_qkvz]
        self.in_proj_ba.weight.data = merged[n_qkvz:]
        self._in_proj_w = merged
        self._in_proj_wt = in_proj_wt
        self._out_proj_wt = out_proj_wt
        self._norm_weight_packed = norm_packed
        self._sm_count = sm_count
        self._sm100_gdn = sm100_gdn
        self._post_recurrence = post_recurrence
        self._pre_recurrence = pre_recurrence
        self._packed_projection = True

    @staticmethod
    def _load_extension():
        """Build the fused kernels now, not on first forward.

        Compiling here means it happens during weight preparation, before the
        harness samples the thread count or starts timing. A build failure is not
        fatal: the Triton stages this module otherwise launches directly are still
        correct, so it keeps those and says so.
        """
        try:
            return load_op("fastkernels_gdn_fused",
                           "qwen3_next_gdn_attention_kernels.cu")
        except Exception as exc:  # noqa: BLE001 - no toolchain, no ninja, no nvcc
            import warnings
            warnings.warn(
                f"fused GDN kernels unavailable ({exc}); falling back to the "
                "Triton conv, post-conv prep and grouped gated norm",
                RuntimeWarning, stacklevel=2)
            return None

    def _resolve_gdn_dispatch(self, device):
        """Bind the Blackwell GDN entry point, or return None to keep the wrappers.

        An absent kernel or a non-Blackwell device is not an error here: the
        inherited routing (public wrapper, then the vLLM chunk kernel) is still
        correct, just slower. Resolving once at preparation time is also what keeps
        the capability query out of the per-call path.
        """
        if not (self._use_flashinfer_prefill and device.type == "cuda"
                and torch.cuda.get_device_capability(device)[0] == 10):
            return None
        cuda_major = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else 0
        if cuda_major < 13 or self.head_k_dim != 128:
            return None
        try:
            from flashinfer.gdn_kernels import chunk_gated_delta_rule_sm100
        except ImportError:
            return None
        if chunk_gated_delta_rule_sm100 is None:
            return None
        return chunk_gated_delta_rule_sm100

    def _reject_reload_after_packing(self, state_dict, prefix, local_metadata,
                                     strict, missing_keys, unexpected_keys,
                                     error_msgs) -> None:
        """Refuse a checkpoint reload once the packed row order is installed.

        ``state_dict()`` after preparation exports ``in_proj_qkvz.weight`` in packed
        rather than checkpoint row order, so writing checkpoint-order rows back
        would place them in permuted positions and corrupt the layer silently.
        Loaders are expected to run before ``process_weights_after_loading``, which
        is what the benchmark harness does.
        """
        if self._packed_projection:
            raise RuntimeError(
                "in_proj weights are in packed row order; load them before "
                "process_weights_after_loading()"
            )

    # -- scratch -----------------------------------------------------------

    def _scratch_buffer(self, key: str, lead: int, trail: tuple[int, ...],
                        dtype: torch.dtype, device) -> torch.Tensor:
        """Grow-only scratch, validated on device, dtype and every trailing dim.

        Keyed by name, not by shape: a dictionary that grows an entry per distinct
        token count is unbounded over a real serving run. The leading dimension is
        a capacity, so retained memory is one high-water allocation per buffer;
        every trailing dimension is fixed and validated, because the state-shaped
        buffers are sized by batch count as well as token count and a cache that
        ignored that would hand back an undersized buffer at a larger batch.

        Each allocation comes straight from the caching allocator rather than
        being carved out of one flat workspace, so every buffer is at least
        256-byte aligned -- more than the 16 bytes the FlashInfer kernel assumes,
        with none of the offset arithmetic a carved workspace would need.
        """
        buf = self._scratch.get(key)
        if (buf is None or buf.dtype is not dtype or buf.device != device
                or tuple(buf.shape[1:]) != trail or buf.shape[0] < lead):
            buf = torch.empty((lead, *trail), dtype=dtype, device=device)
            self._scratch[key] = buf
        return buf[:lead]

    # -- stage launches ----------------------------------------------------

    def _causal_conv(self, x_view: torch.Tensor, out: torch.Tensor,
                     conv_state: torch.Tensor, md) -> None:
        """Causal conv + SiLU over the strided projection view.

        The vLLM wrapper allocates its output with ``empty_like`` on the
        transposed input. For a strided projection view that is not dense, so
        ``preserve_format`` falls back to a channel-major buffer whose innermost
        stride is the token count -- which the post-conv prep, which indexes
        channels contiguously, cannot read. Launching the kernel directly is what
        lets the input stay strided *and* the output stay token-major.
        """
        dim = out.shape[1]
        width = self.conv_kernel_size
        weight = self.conv1d.weight
        nums_dict = md.nums_dict
        if nums_dict is None:
            nums_dict, _, _ = compute_causal_conv1d_metadata(md.query_start_loc_int32)
        nums = nums_dict[_CONV_BLOCK_M]
        cache_indices = md.non_spec_state_indices_tensor
        _causal_conv1d_fwd_kernel[
            (nums["tot"], triton.cdiv(dim, _CONV_BLOCK_N))
        ](
            x_view, weight, None, conv_state, cache_indices,
            md.has_initial_state, md.query_start_loc_int32,
            nums["batch_ptr"], nums["token_chunk_offset_ptr"],
            None, None, None, None,
            out,
            dim, conv_state.shape[0],
            # Both activations are read as (channel, token): innermost stride 1
            # along channels, row stride along tokens.
            1, x_view.stride(0),
            weight.stride(0), weight.stride(1),
            conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
            cache_indices.stride(0),
            1, out.stride(0),
            1,
            _PAD_SLOT_ID, _NULL_BLOCK_ID,
            HAS_BIAS=False, KERNEL_WIDTH=width, SILU_ACTIVATION=True,
            IS_APC_ENABLED=False, HAS_NULL_BLOCK=True,
            NP2_STATELEN=triton.next_power_of_2(width - 1),
            BLOCK_M=_CONV_BLOCK_M, BLOCK_N=_CONV_BLOCK_N, num_stages=2,
        )

    def _post_conv_prep(self, conv_out, a_view, b_view, q, k, v, g, beta,
                        fold_exp: bool) -> None:
        """Split, L2-normalize q/k, and form the gates, reading a/b strided.

        ``fold_exp`` folds the forget gate's exponential in here, which removes the
        separate elementwise launch the baseline pays for it. That exponential is
        then Triton's fast lowering rather than ``torch.exp``; the measured
        deviation is ~1.2e-7 absolute / ~2.1e-7 relative in fp32, recorded in
        ``profile/p1_stage_derisk/``. Every other output is bitwise identical. The
        vLLM chunk kernel wants the gate in the log domain, so it is only folded
        for the FlashInfer route that consumes ``exp(g)``.
        """
        n = conv_out.shape[0]
        block_t = 16
        _fused_post_conv_kernel[
            (triton.cdiv(n, block_t), self.local_k_heads + self.local_v_heads)
        ](
            mixed_qkv_ptr=conv_out, a_ptr=a_view, b_ptr=b_view,
            A_log_ptr=self.A_log, dt_bias_ptr=self.dt_bias,
            q_ptr=q, k_ptr=k, v_ptr=v, g_ptr=g, beta_ptr=beta,
            stride_x_tok=conv_out.stride(0), stride_a_tok=a_view.stride(0),
            stride_b_tok=b_view.stride(0), stride_q_tok=q.stride(0),
            stride_k_tok=k.stride(0), stride_v_tok=v.stride(0),
            L=n, H=self.local_k_heads, HV=self.local_v_heads,
            K=self.head_k_dim, V=self.head_v_dim,
            APPLY_L2NORM=True, L2NORM_EPS=_L2NORM_EPS, OUTPUT_G_EXP=fold_exp,
            SOFTPLUS_THRESHOLD=_SOFTPLUS_THRESHOLD,
            BLOCK_T=block_t,
            BK=triton.next_power_of_2(self.head_k_dim),
            BV=triton.next_power_of_2(self.head_v_dim),
            num_warps=4, num_stages=2,
        )

    def _gated_norm(self, o_flat, z_view, out, device) -> None:
        """RMSNorm(o) * silu(z) with ``z`` read in place, one launch, no copies.

        ``RMSNormGated`` flattens token and head into one axis, which for a strided
        gate is not a view -- and its input guard would call ``.contiguous()``
        anyway. The underlying kernel already supports a grouped layout with an
        explicit gate row stride, so with one weight slice per value head it
        computes exactly the baseline's arithmetic, including the same
        ``y *= z * sigmoid(z)`` association, straight off the projection buffer.
        """
        n = o_flat.shape[0]
        group = self.head_v_dim
        ngroups = self.local_v_heads
        rstd = self._scratch_buffer("rstd", ngroups * n, (), torch.float32, device)
        rows = min(triton.next_power_of_2(triton.cdiv(n, 2 * self._sm_count)), 4)
        block_n = min(65536 // o_flat.element_size(),
                      triton.next_power_of_2(group))
        layer_norm_fwd_kernel[(triton.cdiv(n, rows), ngroups)](
            o_flat, out, self._norm_weight_packed, None, z_view, None, rstd,
            o_flat.stride(0), out.stride(0), z_view.stride(0),
            n, group, self.norm.eps,
            BLOCK_N=block_n, ROWS_PER_BLOCK=rows,
            HAS_BIAS=False, HAS_Z=True, NORM_BEFORE_GATE=True, IS_RMS_NORM=True,
            num_warps=min(max(block_n // 256, 1), 8), ACTIVATION="swish",
        )

    # -- forward -----------------------------------------------------------

    @staticmethod
    def _pre_tile_width(max_seqlen: int) -> int:
        """Tokens per tile in the fused pre-recurrence kernel.

        Each tile reloads the three conv history taps preceding it, so a wider tile
        does less redundant reading -- but the kernel is latency-bound on global
        loads (Nsight shows `long_scoreboard` dominant at every shape), so what it
        actually wants is blocks in flight, and narrow tiles give more of them.
        Measured per launch, in microseconds, over the sweep in
        `profile/p1_tile_sweep/`:

            tokens   tile 8   tile 16   tile 32   tile 64
                60     19.7      18.4      28.2      47.9
               445     14.5      16.8      29.1      53.7
              1024     20.9      23.0      31.1      53.7
              2048     37.1      37.3      41.3      57.8
              4096     70.1      72.2      81.5      93.9
              8192    133.5     131.1     140.4     158.9
             16384     258.5     252.6     256.6     274.4

        Eight wins from 445 through 4096 tokens, where parallelism is still scarce
        enough that block count dominates; sixteen wins at 8192 and above, where
        there are blocks to spare and the redundant history reads start to cost. The
        crossover therefore sits between 4096 and 8192 and the threshold is set at
        the measured boundary rather than interpolated. At 60 tokens the two are
        within 1.3 us of each other and the shape is host-bound anyway, so it
        follows the small-shape branch.

        Two knobs were swept alongside this and rejected on the measurements, both
        left as inert `#ifdef`s in the kernel source so the sweep reproduces:

        * A `__launch_bounds__` register cap, to lift the 62.5% occupancy ceiling
          Nsight attributes to 48 registers per thread. No change below 4096 tokens
          and 8-12% worse at 16384.
        * Cache/load policies on the conv taps -- `__ldg`, `evict_last` and
          `evict_first`. Within half a percent of a plain load from 445 tokens up,
          and `evict_first` is consistently worse (up to 14% at 1024). The taps are
          re-read by the following tile, so telling the cache to drop them is the
          wrong hint; telling it to keep them is what the hardware already does.
        """
        return 16 if max_seqlen >= 8192 else 8

    def _can_fuse_pre(self, md, conv_state, dtype) -> bool:
        """Whether the fused pre-recurrence kernel accepts this call.

        This is the *capability* half of the predicate -- 128-wide heads, conv width
        four, bf16 activations, a channel-innermost conv state, an available
        extension, and a known host-side maximum query length to size the tile
        grid. Anything outside it is not an error: it routes to the two Triton
        kernels, which this module launches directly and which the parity harness
        covers on the same cases.

        Chunk-plan validity is not checked here. ``_validate_metadata`` owns that
        and has already run, because malformed metadata is a contract violation
        rather than a capability gap and has nowhere safe to fall back to.
        """
        return (
            self._pre_recurrence is not None
            and self._sm100_gdn is not None      # the gate is folded for this route
            and dtype is torch.bfloat16
            and conv_state.dtype is torch.bfloat16
            and conv_state.stride(1) == 1
            and self.conv_kernel_size == 4
            and self.head_k_dim == self.head_v_dim == 128
            # A sequence shorter than the conv history is fine: it is a single
            # tile, and the tile-zero path assembles its state from the previous
            # one. The "tiles are at least three tokens wide" invariant the later
            # tiles rely on is a property of the tile size, checked in the kernel.
            and md.max_query_len >= 1
        )

    def forward_impl(self, hidden_states: torch.Tensor, state_manager=None):
        md = get_context().kda_metadata
        if state_manager is None:
            state_manager = get_context().kda_state
        if md is None or state_manager is None:
            raise RuntimeError(
                "Qwen3NextGDNAttention requires engine-managed recurrent state "
                "and metadata",
            )
        if not self._packed_projection:
            # Preparation never ran, or failed part-way. The projection is still
            # in checkpoint order, so the inherited path is exactly right.
            return super().forward_impl(hidden_states, state_manager)
        self._ensure_triton_allocator(hidden_states.device)
        if md.num_prefills > 0:
            return self._packed_prefill(hidden_states, state_manager, md)
        return self._packed_decode(hidden_states, state_manager, md)

    def _project(self, hidden_states, device):
        """One GEMM into scratch, plus the four zero-copy consumer views."""
        x = hidden_states.reshape(-1, self.hidden_size)
        n = x.shape[0]
        hv = self.local_v_heads
        conv_dim = 2 * self.local_k_heads * self.head_k_dim + hv * self.head_v_dim
        total = self._in_proj_w.shape[0]
        proj = self._scratch_buffer("proj", n, (total,), x.dtype, device)
        torch.mm(x, self._in_proj_wt, out=proj)
        qkvz_dim = self._qkvz_dim
        return (
            n,
            proj[:, :conv_dim],
            proj[:, conv_dim:qkvz_dim],
            proj[:, qkvz_dim:qkvz_dim + hv],
            proj[:, qkvz_dim + hv:],
        )

    def _packed_prefill(self, hidden_states, state_manager, md):
        device = hidden_states.device
        dt = hidden_states.dtype
        h, hv = self.local_k_heads, self.local_v_heads
        k_dim, v_dim = self.head_k_dim, self.head_v_dim
        n, mixed_view, z_view, b_view, a_view = self._project(
            hidden_states, device)
        conv_dim = mixed_view.shape[1]

        cu_seqlens = md.query_start_loc_int32
        if cu_seqlens is None:
            cu_seqlens = md.non_spec_query_start_loc.to(torch.int32)
        state_idx = md.non_spec_state_indices_tensor
        # Ahead of *any* kernel: both custom kernels index the chunk plan directly,
        # so validating it only before the recurrence would leave the fused
        # pre-recurrence kernel reading it unchecked.
        self._validate_metadata(cu_seqlens, state_idx, n, md.max_query_len)

        conv_state = state_manager.gdn_conv[self.layer_idx]
        q = self._scratch_buffer("q", n, (h, k_dim), dt, device)
        k = self._scratch_buffer("k", n, (h, k_dim), dt, device)
        v = self._scratch_buffer("v", n, (hv, v_dim), dt, device)
        gate = self._scratch_buffer("gate", n, (hv,), torch.float32, device)
        beta = self._scratch_buffer("beta", n, (hv,), torch.float32, device)
        if self._can_fuse_pre(md, conv_state, dt):
            # One pass over the projection: conv, SiLU, split, L2 norm and both
            # gates, with the 8192-wide conv result never reaching HBM.
            self._pre_recurrence(
                mixed_view, self.conv1d.weight, conv_state, cu_seqlens,
                md.has_initial_state, state_idx,
                a_view, b_view, self.A_log, self.dt_bias,
                q, k, v, gate, beta, self._conv_tap,
                md.max_query_len, self._pre_tile_width(md.max_query_len))
        else:
            conv_out = self._scratch_buffer("conv_out", n, (conv_dim,), dt, device)
            self._causal_conv(mixed_view, conv_out, conv_state, md)
            self._post_conv_prep(conv_out, a_view, b_view, q, k, v, gate, beta,
                                 fold_exp=(self._sm100_gdn is not None))

        o, final_state = self._recurrence(
            q, k, v, gate, beta, state_manager, md, cu_seqlens, device)

        recurrent_full = state_manager.recurrent[self.layer_idx]
        state_idx = md.state_indices_long
        if state_idx is None:
            state_idx = md.non_spec_state_indices_tensor.long()

        y = self._scratch_buffer("y", n, (hv * v_dim,), dt, device)
        fused = (self._post_recurrence is not None
                 and recurrent_full.dtype is torch.bfloat16
                 and dt is torch.bfloat16
                 and final_state.dtype is torch.float32
                 and v_dim == self.head_k_dim == 128)
        if fused:
            # One kernel: the gated norm off the strided gate, plus the narrowing
            # and scatter of the final state into the cache.
            self._post_recurrence(o, z_view, self._norm_weight_packed, y,
                                  final_state, recurrent_full, state_idx,
                                  self.norm.eps)
        else:
            narrowed = self._scratch_buffer(
                "state_narrow", final_state.shape[0],
                tuple(final_state.shape[1:]), recurrent_full.dtype, device)
            narrowed.copy_(final_state)
            recurrent_full.index_copy_(0, state_idx, narrowed)
            self._gated_norm(o.view(n, hv * v_dim), z_view, y, device)
        out = torch.empty(n, self._out_proj_wt.shape[1], dtype=dt, device=device)
        torch.mm(y, self._out_proj_wt, out=out)
        return out

    def _initial_state(self, recurrent_full, md, device, batch, hv):
        """The initial recurrent state, gathering only when a sequence carries one.

        The gather, the mask and the fp32 promotion are three kernels and ~19 us of
        Python per call, and in the common prefill case they all produce zeros. A
        persistent zeroed fp32 buffer produces the same argument with none of that.
        It is a read-only argument to the kernel, so it stays zero; and because the
        FlashInfer compile cache is keyed on *whether* an initial state is present
        rather than on its contents, this keeps the baseline's single cache entry
        where passing ``None`` would add a second CuTe-DSL compilation per process.

        Which slots carry state is decided host-side when the step's chunk plan is
        built, so the mask never has to come back from the device; the device mask
        remains the fallback for metadata that does not carry the host summary.
        """
        shape = (hv, self.head_k_dim, self.head_v_dim)
        if md.has_initial_state is not None and not md.any_have_initial_state:
            buf = self._scratch.get("zero_state")
            if (buf is None or buf.device != device
                    or tuple(buf.shape[1:]) != shape or buf.shape[0] < batch):
                buf = torch.zeros((batch, *shape), dtype=torch.float32, device=device)
                self._scratch["zero_state"] = buf
            return buf[:batch]
        state_idx = md.state_indices_long
        if state_idx is None:
            state_idx = md.non_spec_state_indices_tensor.long()
        gathered = self._scratch_buffer(
            "init_state", batch, shape, torch.float32, device)
        gathered.copy_(recurrent_full.index_select(0, state_idx))
        if md.has_initial_state is not None and not md.all_have_initial_state:
            keep = md.has_initial_state.view(-1, *([1] * (gathered.dim() - 1)))
            gathered.masked_fill_(~keep, 0)
        return gathered

    def _validate_metadata(self, cu_seqlens, state_idx, tokens: int,
                           max_query_len: int) -> None:
        """Establish what the chunk plan must satisfy before any kernel reads it.

        Both custom kernels index `cu_seqlens` directly, so this has to run before
        either of them, not just before the recurrence. Malformed metadata raises
        rather than falling back: once the projection is in packed row order the
        inherited implementation is not a safe destination, because it would apply
        the baseline deinterleave to permuted rows.

        The first-zero / monotone / last-is-T checks need the values on the host,
        which is a synchronization. They run once per distinct metadata object --
        keyed on pointer, version counter, element count, token count and device --
        because the chunk planner builds one of these per step, not per layer.
        """
        sig = (cu_seqlens.data_ptr(), cu_seqlens._version, cu_seqlens.numel(),
               state_idx.data_ptr(), state_idx._version, state_idx.numel(),
               tokens, max_query_len, cu_seqlens.device)
        if sig == self._metadata_signature:
            return
        for name, t, dtype in (("cu_seqlens", cu_seqlens, torch.int32),
                               ("state_indices", state_idx, torch.int32)):
            if t.dtype is not dtype:
                raise ValueError(f"{name} must be {dtype}, got {t.dtype}")
            if not t.is_cuda or t.device != cu_seqlens.device:
                raise ValueError(f"{name} must be CUDA and share one device")
            if t.dim() != 1 or not t.is_contiguous():
                raise ValueError(f"{name} must be 1-D and contiguous")
        if cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must hold at least [0, T]")
        if state_idx.numel() != cu_seqlens.numel() - 1:
            raise ValueError(
                f"one state index per sequence: {state_idx.numel()} indices for "
                f"{cu_seqlens.numel() - 1} sequences")
        host = cu_seqlens.to("cpu", non_blocking=False).tolist()
        if host[0] != 0:
            raise ValueError(f"cu_seqlens must start at 0, got {host[0]}")
        if any(b < a for a, b in zip(host, host[1:])):
            raise ValueError(f"cu_seqlens must be non-decreasing, got {host}")
        if host[-1] != tokens:
            raise ValueError(
                f"cu_seqlens must end at the token count {tokens}, got {host[-1]}")
        # The fused pre-recurrence grid is sized from the host-side maximum query
        # length, so a value smaller than the real maximum would not raise anywhere
        # -- it would simply launch too few tiles and leave the tail of the longest
        # sequence unwritten. Since the lengths are already on the host here, check
        # it rather than trust it.
        longest = max((b - a for a, b in zip(host, host[1:])), default=0)
        if max_query_len < longest:
            raise ValueError(
                f"max_query_len {max_query_len} is below the longest sequence "
                f"{longest}; the tile grid would leave tokens unwritten")
        self._empty_sequence_present = any(
            b == a for a, b in zip(host, host[1:]))
        self._metadata_signature = sig

    def _check_dispatch_preconditions(self, q, k, v, o, gate, beta, cu_seqlens,
                                      init_state, out_state, scale) -> None:
        """Everything the public wrapper would have checked, checked here.

        Bypassing the wrapper transfers its validation to this module, and a
        violated precondition inside the SM100 adapter is a memory fault rather
        than an exception -- the adapter's 16-byte alignment is an ``assumed_align``
        annotation, not a check. These run on the parity harness and are cheap
        enough to leave on: they are attribute reads, not device work.
        """
        dk = self.head_k_dim
        assert dk == 128, f"Blackwell GDN prefill requires head_size=128, got {dk}"
        n = q.shape[0]
        device = q.device
        # q and k carry one head per key head; v and the output carry one per value
        # head, which is the grouped-value shape this operator always has. Getting
        # these exact rather than merely consistent is the check that would have
        # caught a caller passing the wrong head count for q.
        heads = {"q": self.local_k_heads, "k": self.local_k_heads,
                 "v": self.local_v_heads, "output": self.local_v_heads}
        for name, t in (("q", q), ("k", k), ("v", v), ("output", o)):
            assert t.dtype is q.dtype and t.dtype in (
                torch.bfloat16, torch.float16), f"{name} must be fp16/bf16"
            assert t.is_contiguous(), f"{name} must be contiguous"
            assert tuple(t.shape) == (n, heads[name], dk), (
                f"{name} must be [{n}, {heads[name]}, {dk}], got {tuple(t.shape)}")
            assert t.device == device, f"{name} must share one device"
            assert t.data_ptr() % 16 == 0, f"{name} must be 16-byte aligned"
        for name, t in (("gate", gate), ("beta", beta)):
            assert t.dtype is torch.float32, f"{name} must be float32"
            assert t.is_contiguous(), f"{name} must be contiguous"
            assert tuple(t.shape) == (n, self.local_v_heads), f"{name} shape"
            assert t.device == device, f"{name} must share one device"
            assert t.data_ptr() % 16 == 0, f"{name} must be 16-byte aligned"
        assert cu_seqlens.is_cuda and cu_seqlens.dtype is torch.int32, "cu_seqlens"
        assert cu_seqlens.device == device, "cu_seqlens must share one device"
        batch = cu_seqlens.numel() - 1
        for name, t in (("initial_state", init_state), ("output_state", out_state)):
            if t is None:
                continue
            assert tuple(t.shape) == (batch, self.local_v_heads, dk, dk), (
                f"{name} must be [{batch}, {self.local_v_heads}, {dk}, {dk}], "
                f"got {tuple(t.shape)}")
            assert t.dtype is torch.float32, f"{name} must be float32 here"
            assert t.device == device, f"{name} must share one device"
            assert t.is_contiguous() and t.data_ptr() % 16 == 0, f"{name} layout"
        # The wrapper substitutes 1/sqrt(head_dim) for a zero or absent scale, and
        # that exact value is what the kernel was validated against; note that
        # `head_dim ** -0.5` is a different float.
        assert scale == 1.0 / math.sqrt(dk), (
            f"scale must be exactly 1/sqrt({dk}), got {scale!r}")

    def _recurrence(self, q, k, v, gate, beta, state_manager, md,
                    cu_seqlens, device):
        """Chunked GDN scan, dispatched straight into the Blackwell kernel.

        The public wrapper costs 49 us of Python per call against 11 us for the
        entry point it eventually calls, almost all of it argument validation plus
        uncached capability and SM-count queries. Dropping it means adopting its
        preconditions, which is what ``_check_dispatch_preconditions`` does.

        ``gate`` already carries ``exp(g)``, which is the form this kernel wants;
        the vLLM fallback gets the log-domain gate because it exponentiates
        internally.
        """
        from fastkernels.tasks.baseline.L2.qwen3_next_gdn_attention import (
            _flashinfer_gdn_prefill, _vllm_chunk_gated_delta_rule,
        )
        recurrent_full = state_manager.recurrent[self.layer_idx]
        n, hv = q.shape[0], self.local_v_heads
        batch = cu_seqlens.numel() - 1
        if not (self._sm100_gdn is not None):
            # Same descent the inherited implementation makes: the public
            # FlashInfer wrapper while its prefill path is available, and the vLLM
            # chunk kernel only below that. Both take the gate in the log domain,
            # which is why it is folded only for the direct route.
            init_state = self._initial_state(
                recurrent_full, md, device, batch, hv).to(recurrent_full.dtype)
            if self._use_flashinfer_prefill:
                o, final_state = _flashinfer_gdn_prefill(
                    q=q.unsqueeze(0), k=k.unsqueeze(0), v=v.unsqueeze(0),
                    g=gate.unsqueeze(0), beta=beta.unsqueeze(0),
                    initial_state=init_state, output_final_state=True,
                    cu_seqlens=cu_seqlens,
                )
            else:
                o, final_state = _vllm_chunk_gated_delta_rule(
                    q=q.unsqueeze(0), k=k.unsqueeze(0), v=v.unsqueeze(0),
                    g=gate.unsqueeze(0), beta=beta.unsqueeze(0),
                    initial_state=init_state, output_final_state=True,
                    cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=False,
                )
            return o.squeeze(0), final_state
        init_state = self._initial_state(recurrent_full, md, device, batch, hv)
        o = self._scratch_buffer("o", n, (hv, self.head_v_dim), q.dtype, device)
        out_state = self._scratch_buffer(
            "out_state", batch, (hv, self.head_k_dim, self.head_v_dim),
            torch.float32, device)
        if self._empty_sequence_present:
            # A zero-length sequence gets no output-state write from the kernel, so
            # whatever the buffer held is what the scatter carries into the cache --
            # undefined in the baseline too, whose output state is a fresh
            # torch.empty. Seeding from the initial state makes an empty sequence's
            # final state equal its initial state, which is the answer the recurrence
            # gives when nothing happens. Keyed on whether the plan actually contains
            # an empty sequence, which `_validate_metadata` already knows from the
            # host copy it reads: a batch size of one does not make this safe, since
            # `cu_seqlens = [0, 0]` is a valid single empty sequence.
            out_state.copy_(init_state)
        self._check_dispatch_preconditions(
            q, k, v, o, gate, beta, cu_seqlens, init_state, out_state,
            self._gdn_scale)
        self._sm100_gdn(q, k, v, gate, beta, o, cu_seqlens, init_state,
                        out_state, self._gdn_scale)
        return o, out_state

    def _packed_decode(self, hidden_states, state_manager, md):
        """Decode: the baseline's kernels, fed from the packed projection.

        Never exercised by the benchmark -- the harness publishes
        ``num_prefills=1`` for every case -- but it is part of the operator's
        contract, and feeding the packed projection straight into the inherited
        deinterleave would produce silently wrong values.
        """
        device = hidden_states.device
        dt = hidden_states.dtype
        hv, v_dim = self.local_v_heads, self.head_v_dim
        n, mixed_view, z_view, b_view, a_view = self._project(
            hidden_states, device)
        cu_seqlens = md.query_start_loc_int32
        if cu_seqlens is None:
            cu_seqlens = md.non_spec_query_start_loc.to(torch.int32)
        self._validate_metadata(
            cu_seqlens[: md.num_decodes + 1],
            md.non_spec_state_indices_tensor[: md.num_decodes], n,
            md.max_query_len)
        mixed_qkv = _vllm_causal_conv1d_update(
            mixed_view.contiguous(),
            state_manager.gdn_conv[self.layer_idx], self.conv1d.weight, None,
            activation="silu",
            conv_state_indices=md.non_spec_state_indices_tensor[: md.num_decodes],
            null_block_id=-1, validate_data=True,
        )
        q, k, v = _split_conv_qkv(
            mixed_qkv, self.local_k_heads, self.head_k_dim, hv, v_dim)
        o, _ = _vllm_fused_sigmoid_gating_update(
            A_log=self.A_log, a=a_view.contiguous(), b=b_view.contiguous(),
            dt_bias=self.dt_bias, q=q, k=k, v=v,
            initial_state=state_manager.recurrent[self.layer_idx],
            inplace_final_state=True,
            cu_seqlens=cu_seqlens[: md.num_decodes + 1],
            ssm_state_indices=md.non_spec_state_indices_tensor,
            use_qk_l2norm_in_kernel=True,
        )
        y = self._scratch_buffer("y", n, (hv * v_dim,), dt, device)
        self._gated_norm(o.reshape(n, hv * v_dim), z_view, y, device)
        out = torch.empty(n, self._out_proj_wt.shape[1], dtype=dt, device=device)
        torch.mm(y, self._out_proj_wt, out=out)
        return out
