"""KimiDeltaAttention -- launch/CPU-overhead-optimized.

Measured starting point (B200, tp=1, bench's single-sequence prefill Context,
per-stage CPU wall clock at ``hidden_states=[1, 2304]``): 1069 us of Python /
launch work against 215 us of GPU work.  Four of the five benched shapes
(1, 26, 64, 443 tokens) sit at ~1.10 ms, i.e. they are *entirely* CPU bound --
the GPU idles waiting for launches.  So the lever is Python statements and
kernel launches, not FLOPs.

Where the 1069 us went, and what this file does about it:

  440 us  chunk_kda_with_fused_gate  -- 9 triton launches. Untouched (FLA core).
  147 us  3x causal_conv1d_fn        -- 3 triton launches. Untouched; only the
                                        per-call Python around it is cached.
  112 us  gate ladders + rearranges  -> 21 us  (the two down-projections fold
                                        into the qkvb GEMM; one bmm for the up)
  128 us  has_initial_state handling -> 0 us   (host-side summary; see below)
   57 us  o_norm                     -> 53 us  (call rms_norm_gated directly)
   44 us  4x einops rearrange        -> 4 us   (plain .view; all are views)
   24 us  zeros + copy of the output -> 0 us   (alias the kernel's own buffer)
   20 us  conv weight/state reshapes -> 2 us   (cached)
   22 us  state store                -> 16 us  (index_copy_ beats index_put_)

The single largest item is the data-dependent-shape host sync: the parent's
``zero_idx = pf_state_indices[~pf_has_initial]`` runs ``nonzero()``, which
stalls the CPU on the device (72 us) before an ``index_put_`` (40 us) and a
gather (16 us).  ``KimiLinearMetadata`` already carries the answer host-side
(``all_have_initial_state`` / ``any_have_initial_state``, filled by
``engine.py``), exactly as the Qwen3-Next GDN baseline consumes it -- and when
nothing carries initial state, the FLA chunk kernel's ``USE_INITIAL_STATE``
heuristic takes ``None`` and reads zeros from registers, so the gather
disappears too.  Metadata without the summary falls back to a mask-fill on the
gathered copy: shapes stay static (CUDA-graph safe) and no ``nonzero`` runs.
"""

from __future__ import annotations

import torch

from ...baseline.L2.kimi_delta_attention import KimiDeltaAttention as _BaselineKDA
from ..L1.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from ..L1.kda import (
    chunk_kda_with_fused_gate,
    fused_kda_gate,
    fused_recurrent_kda,
)
from ..L1.rms_norm_gated import rms_norm_gated
from ....infra.context import get_context


class KimiDeltaAttention(_BaselineKDA):
    """Subclasses the baseline so the parameter set -- and therefore the
    ``state_dict`` keys the harness shares baseline -> candidate -- is
    identical, and so ``bench._locate_recurrent_attn``'s ``isinstance`` check
    finds this layer (it imports the baseline class by name; a standalone class
    makes every shape SKIP)."""

    def __init__(self, config, layer_idx: int, quant_config: dict | None = None):
        super().__init__(config, layer_idx, quant_config)
        # Everything below is derived from parameter *storage*, which
        # ``nn.Module._apply`` replaces on ``.to(device)`` -- so it is built on
        # first use (after loading), never in __init__.
        self._fused_ready = False
        self._in_wt = None       # [hidden, 3*proj + nh + 2*head_dim] GEMM rhs
        self._in_split = 0       # column where the gate down-projection starts
        self._gate_b_wt = None   # [2, head_dim, proj_local] f_b | g_b, bmm rhs
        self._o_wt = None        # o_proj weight transpose, when mm is equivalent
        self._conv_w = None      # 3x [dim, width]
        self._norm_w = None
        self._norm_b = None
        self._cs_src: tuple = ()
        self._cs_t: tuple = ()

    # ------------------------------------------------------------------
    # Weight fusion
    # ------------------------------------------------------------------
    def process_weights_after_loading(self) -> None:
        """Hook the loader (and ``bench._prep_kimi_recurrent``) calls after the
        checkpoint is in place. Idempotent; ``forward`` also builds lazily
        because the Kimi loader only calls this hook for a fixed class list."""
        self._build_fused()

    def _build_fused(self) -> None:
        """Merge the projections and cache the per-call reshapes.

        ``f_a_proj`` and ``g_a_proj`` are ``hidden -> head_dim`` over the *same*
        ``hidden_states`` -- structurally identical to q/k/v/b -- so their rows
        stack straight into the qkvb GEMM the layer already runs: five GEMM
        launches become one, 2304x12320 -> 2304x12576 (+2% FLOPs, no extra
        launch, and the 256-column gate GEMM that cuBLAS was serving with a
        split-K kernel plus its reduction disappears entirely).

        ``f_b_proj``/``g_b_proj`` are also both ``head_dim -> projection_size``,
        but over *different* activations, so they are not a wider GEMM -- they
        are a batch of two, which ``torch.bmm`` does in one launch for the same
        FLOPs (bit-identical to the split form, max|d| = 0 measured). A
        block-diagonal single GEMM would work too but doubles the FLOPs and
        loses at 16384 tokens (87.3 vs 81.1 us).

        Every merged parameter is rebound as a contiguous *view* into the joint
        buffer -- the trick ``Qwen3NextGDNAttention.process_weights_after_loading``
        uses for its own input projections -- so nothing is duplicated at steady
        state, the ``state_dict`` keys are unchanged, and a later
        ``load_state_dict`` writes straight through. It has to run after loading,
        not in ``__init__``: ``nn.Module._apply`` replaces every parameter object
        on ``.to(device)``, which would leave an earlier view dangling.
        """
        if self.qkvb_proj is not None:
            # quant_config was None (that is what selects the fused qkvb), so
            # none of these ladders are block-scaled.
            qkvb_w = self.qkvb_proj.weight
            fa_w, ga_w = self.f_a_proj.weight, self.g_a_proj.weight
            fb_w, gb_w = self.f_b_proj.weight, self.g_b_proj.weight
            n0, hd = qkvb_w.shape[0], fa_w.shape[0]
            m = torch.empty(n0 + 2 * hd, qkvb_w.shape[1],
                            dtype=qkvb_w.dtype, device=qkvb_w.device)
            m[:n0].copy_(qkvb_w.data)
            m[n0:n0 + hd].copy_(fa_w.data)
            m[n0 + hd:].copy_(ga_w.data)
            self.qkvb_proj.weight.data = m[:n0]
            self.f_a_proj.weight.data = m[n0:n0 + hd]
            self.g_a_proj.weight.data = m[n0 + hd:]
            self._in_wt = m.t()
            self._in_split = n0

            b = torch.empty(2, *fb_w.shape, dtype=fb_w.dtype, device=fb_w.device)
            b[0].copy_(fb_w.data)
            b[1].copy_(gb_w.data)
            self.f_b_proj.weight.data = b[0]
            self.g_b_proj.weight.data = b[1]
            self._gate_b_wt = b.transpose(1, 2)
        else:
            self._in_wt = None
            self._gate_b_wt = None

        # ``F.linear`` -> ``torch.mm`` against a cached transpose is 2-3 us of
        # Python cheaper per call and bit-identical; only valid where the
        # wrapper would not have added a bias or an all-reduce.
        op = self.o_proj
        self._o_wt = (
            op.weight.t()
            if not op.use_fp8 and op.bias is None
            and not (op.reduce_results and op.tp_size > 1)
            else None
        )

        # ``weight`` is [dim, 1, width]; the conv kernel wants [dim, width].
        self._conv_w = tuple(
            c.weight.view(c.weight.shape[0], c.weight.shape[2])
            for c in (self.q_conv1d, self.k_conv1d, self.v_conv1d)
        )
        # ``.data``, not the Parameter: ``nn.Module.__setattr__`` registers any
        # Parameter assigned to an attribute, which would add a ``_norm_w`` key
        # to ``state_dict`` and -- because ``named_parameters`` deduplicates --
        # hide ``o_norm.weight`` from every loader that walks it.
        nw = self.o_norm.weight
        nb = self.o_norm.bias
        self._norm_w = None if nw is None else nw.data
        self._norm_b = None if nb is None else nb.data
        self._fused_ready = True

    def _apply(self, *args, **kwargs):
        """``.to()`` / ``.cuda()`` / ``.float()`` go through here, and
        ``nn.Module._apply`` replaces every parameter object -- which would
        leave the merged buffer and every cached view aliasing freed storage,
        silently. Drop the cache; the next forward rebuilds it from the moved
        parameters. Zero cost at steady state (this runs once, at load)."""
        out = super()._apply(*args, **kwargs)
        if "_fused_ready" in self.__dict__:
            self._fused_ready = False
            self._in_wt = None
            self._gate_b_wt = None
            self._o_wt = None
            self._conv_w = None
            self._norm_w = None
            self._norm_b = None
            self._cs_src = ()
            self._cs_t = ()
        return out

    def _conv_states_t(self, q_s, k_s, v_s) -> tuple:
        """``state.transpose(-1, -2)`` for the three conv caches, memoized on
        tensor identity (the state manager hands back the same objects every
        step, so this is three ``is`` compares instead of three transposes)."""
        src = self._cs_src
        if src and src[0] is q_s and src[1] is k_s and src[2] is v_s:
            return self._cs_t
        t = (q_s.transpose(-1, -2), k_s.transpose(-1, -2), v_s.transpose(-1, -2))
        self._cs_src = (q_s, k_s, v_s)
        self._cs_t = t
        return t

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------
    def _initial_state(self, recurrent_state, idx_long, has_initial, meta):
        """The prefill initial state, without a device->host round trip.

        ``None`` means "all zeros" to the chunk kernel (its ``USE_INITIAL_STATE``
        heuristic), which is bit-identical to handing it a zeroed buffer and
        saves the gather.  The parent zeroed the *persistent* slots first; that
        was always redundant, since every gathered slot is overwritten with
        ``pf_last_state`` at the end of the call.
        """
        if has_initial is None or meta.all_have_initial_state:
            return recurrent_state.index_select(0, idx_long)
        if not meta.any_have_initial_state:
            return None
        init = recurrent_state.index_select(0, idx_long)
        init.masked_fill_(
            ~has_initial.view(-1, *([1] * (init.dim() - 1))), 0,
        )
        return init

    def _core(self, q_proj_states, k_proj_states, v_proj_states, raw_g, beta,
              out, num_tokens):
        """Run the recurrence. Returns the [1, num_tokens, H, D] output --
        ``out`` when one was supplied, otherwise the kernel's own buffer (the
        parent always allocated ``torch.zeros`` and then copied a slice over
        every element of it; both branches fully overwrite what they own)."""
        ctx = get_context()
        state = getattr(ctx, "kda_state", None)
        meta = getattr(ctx, "kda_metadata", None)
        H, D = self.local_num_heads, self.head_dim
        if state is None or meta is None:
            if out is None:
                out = torch.zeros((1, num_tokens, H, D),
                                  dtype=q_proj_states.dtype,
                                  device=q_proj_states.device)
            else:
                out.zero_()
            return out

        li = self.layer_idx
        q_state = state.q_conv_states[li]
        k_state = state.k_conv_states[li]
        v_state = state.v_conv_states[li]
        recurrent_state = state.recurrent_states[li]

        nat = meta.num_actual_tokens
        npt = meta.num_prefill_tokens
        ndt = meta.num_decode_tokens
        if nat != num_tokens:
            q_proj_states = q_proj_states[:nat]
            k_proj_states = k_proj_states[:nat]
            v_proj_states = v_proj_states[:nat]
            raw_g = raw_g[:, :nat]
            beta = beta[:, :nat]

        qw, kw, vw = self._conv_w
        qs_t, ks_t, vs_t = self._conv_states_t(q_state, k_state, v_state)
        cu32 = meta.query_start_loc_int32
        if cu32 is None:
            cu32 = meta.non_spec_query_start_loc.to(torch.int32)
        state_idx = meta.non_spec_state_indices_tensor

        if meta.num_prefills > 0:
            hi = meta.has_initial_state
            q = causal_conv1d_fn(
                q_proj_states.transpose(0, 1), qw, None, qs_t, cu32,
                state_idx, hi, "silu", metadata=meta,
            ).transpose(0, 1)
            k = causal_conv1d_fn(
                k_proj_states.transpose(0, 1), kw, None, ks_t, cu32,
                state_idx, hi, "silu", metadata=meta,
            ).transpose(0, 1)
            v = causal_conv1d_fn(
                v_proj_states.transpose(0, 1), vw, None, vs_t, cu32,
                state_idx, hi, "silu", metadata=meta,
            ).transpose(0, 1)
        else:
            dec_idx = state_idx[:nat]
            q = causal_conv1d_update(q_proj_states, qs_t, qw, None, "silu",
                                     conv_state_indices=dec_idx,
                                     validate_data=True)
            k = causal_conv1d_update(k_proj_states, ks_t, kw, None, "silu",
                                     conv_state_indices=dec_idx,
                                     validate_data=True)
            v = causal_conv1d_update(v_proj_states, vs_t, vw, None, "silu",
                                     conv_state_indices=dec_idx,
                                     validate_data=True)

        # ``causal_conv1d_fn`` builds its output with ``empty_like`` on the
        # transposed view, so it comes back [dim, n] in the *token*-major
        # layout: the transpose above is contiguous and the split into heads is
        # a pure view (checked: strides (4096, 1), reshape shares storage).
        q = q.reshape(1, nat, H, D)
        k = k.reshape(1, nat, H, D)
        v = v.reshape(1, nat, H, D)

        pf_out = dec_out = None
        if npt > 0:
            n_pf = meta.num_prefills
            pf_idx = state_idx[:n_pf]
            pf_idx_long = meta.state_indices_long
            pf_idx_long = (pf_idx.long() if pf_idx_long is None
                           else pf_idx_long[:n_pf])
            pf_cu = (cu32 if meta.num_decodes == 0 else cu32[:n_pf + 1])
            init = self._initial_state(
                recurrent_state, pf_idx_long,
                None if meta.has_initial_state is None
                else meta.has_initial_state[:n_pf],
                meta,
            )
            if npt == nat:
                pq, pk, pv, pg, pb = q, k, v, raw_g, beta
            else:
                pq, pk, pv = q[:, :npt], k[:, :npt], v[:, :npt]
                pg, pb = raw_g[:, :npt], beta[:, :npt]
            pf_out, pf_last_state = chunk_kda_with_fused_gate(
                q=pq,
                k=pk,
                v=pv,
                raw_g=pg,
                beta=pb,
                A_log=self.A_log,
                g_bias=self.dt_bias,
                initial_state=init,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=pf_cu,
            )
            recurrent_state.index_copy_(0, pf_idx_long, pf_last_state)

        if ndt > 0:
            dec_idx = state_idx[meta.num_prefills:] if npt > 0 else state_idx
            dec_cu = (cu32 if meta.num_prefills == 0
                      else cu32[:meta.num_decodes + 1])
            dec_g = fused_kda_gate(
                raw_g[0, npt:].reshape(ndt, H * D),
                self.A_log, D, g_bias=self.dt_bias,
            ).unsqueeze(0)
            dec_out, _ = fused_recurrent_kda(
                q=q[:, npt:].contiguous(),
                k=k[:, npt:].contiguous(),
                v=v[:, npt:].contiguous(),
                g=dec_g,
                beta=beta[:, npt:].contiguous(),
                initial_state=recurrent_state,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=dec_cu,
                ssm_state_indices=dec_idx,
            )

        # Alias the kernel's own output when it already *is* the whole answer:
        # ``chunk_gla_fwd_o_gk`` writes through the value buffer, so ``pf_out``
        # is a contiguous [1, T, H, D] tensor nobody else needs.
        if out is None and nat == num_tokens:
            if ndt == 0:
                return pf_out
            if npt == 0:
                return dec_out
        if out is None:
            out = torch.empty((1, num_tokens, H, D), dtype=q.dtype,
                              device=q.device)
            if nat != num_tokens:
                out[:, nat:].zero_()
        if pf_out is not None:
            out[:, :npt] = pf_out
        if dec_out is not None:
            out[:, npt:nat] = dec_out
        return out

    def forward_impl(
        self,
        q_proj_states: torch.Tensor,
        k_proj_states: torch.Tensor,
        v_proj_states: torch.Tensor,
        raw_g: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        """Out-parameter form kept for ``fastkernels::kda_attention``, the
        splitting op ``infra.compilation`` calls under piecewise compilation.
        Reachable without going through ``forward`` (the op looks the layer up
        by name), so it builds the cache too."""
        if not self._fused_ready:
            self._build_fused()
        self._core(q_proj_states, k_proj_states, v_proj_states, raw_g, beta,
                   core_attn_out, core_attn_out.shape[1])

    # ------------------------------------------------------------------
    def forward(self, hidden_states: torch.Tensor, state_manager=None):
        del state_manager
        if not self._fused_ready:
            self._build_fused()
        if not self._triton_allocator_ready:
            self._ensure_triton_allocator(hidden_states.device)

        num_tokens = hidden_states.shape[0]
        H, D = self.local_num_heads, self.head_dim

        if self._in_wt is not None:
            p = self._ps_local
            proj = torch.mm(hidden_states, self._in_wt)
            q_proj_states = proj[:, :p]
            k_proj_states = proj[:, p:2 * p]
            v_proj_states = proj[:, 2 * p:3 * p]
            raw_beta = proj[:, 3 * p:self._in_split]
            # [n, 2*head_dim] -> [2, n, head_dim]: splitting the last axis is a
            # pure view even on the strided slice, and cuBLAS takes the
            # resulting batch stride directly (no copy).
            fg = torch.bmm(
                proj[:, self._in_split:].view(num_tokens, 2, D).transpose(0, 1),
                self._gate_b_wt,
            )
            raw_g = fg[0].view(1, num_tokens, H, D)
            g2 = fg[1].view(num_tokens * H, D)
        else:
            if self.qkvb_proj is not None:
                p = self._ps_local
                qkvb = self.qkvb_proj(hidden_states)
                q_proj_states = qkvb[:, :p]
                k_proj_states = qkvb[:, p:2 * p]
                v_proj_states = qkvb[:, 2 * p:3 * p]
                raw_beta = qkvb[:, 3 * p:]
            else:
                q_proj_states = self.q_proj(hidden_states)
                k_proj_states = self.k_proj(hidden_states)
                v_proj_states = self.v_proj(hidden_states)
                raw_beta = self.b_proj(hidden_states)
            raw_g = self.f_b_proj(self.f_a_proj(hidden_states)).view(
                1, num_tokens, H, D)
            g2 = self.g_b_proj(self.g_a_proj(hidden_states)).view(
                num_tokens * H, D)

        # ``.float()`` materializes the strided beta slice contiguous, so it
        # never reaches a kernel strided.
        beta = raw_beta.float().sigmoid().unsqueeze(0)

        if self._use_custom_op:
            core = torch.empty((1, num_tokens, H, D), dtype=hidden_states.dtype,
                               device=hidden_states.device)
            torch.ops.fastkernels.kda_attention(
                q_proj_states, k_proj_states, v_proj_states, raw_g, beta, core,
                self._layer_name,
            )
        else:
            core = self._core(q_proj_states, k_proj_states, v_proj_states,
                              raw_g, beta, None, num_tokens)

        # ``rms_norm_gated`` flattens to [-1, last] itself; handing it the
        # already-flat pair skips two no-op reshapes and the CustomOp dispatch.
        core = rms_norm_gated(
            core.reshape(num_tokens * H, D), g2, self._norm_w, self._norm_b,
            self.o_norm.activation, None, False, False, self.o_norm.eps,
        )
        core = core.view(num_tokens, H * D)
        if self._o_wt is not None:
            return torch.mm(core, self._o_wt)
        return self.o_proj(core)
