"""Qwen3-Next decoder layer: hybrid GDN/full attention + MoE.

Dispatches to GDN linear attention or full attention based on layer type.
All layers use MoE (every layer is sparse in Qwen3-Next).
Uses GemmaRMSNorm (weight + 1 convention).

Optimized against the baseline's eager op-by-op composition. The submodule tree
(and therefore the ``state_dict``) is identical to the baseline; the decoder
layer's ``forward`` drives its own kernels over those parameters instead of
calling the L1/L2 wrappers one at a time:

* ``input_layernorm`` / ``post_attention_layernorm`` -- one Triton kernel that
  does the residual add, keeps the bf16 sum as the new residual and emits the
  Gemma-convention ``(1 + w)`` scaled norm. The baseline routes these through
  ``torch.compile``, which splits the row reduction over too few programs.
* The GDN input projection's *deinterleave* is gone rather than reordered.
  ``in_proj_qkvz`` emits one ``[q k v z]`` group per K head while the conv weight
  and conv-state cache use the packed ``[q_all | k_all | v_all]`` order; carrying
  both index maps into the kernels below costs nothing, removes the baseline's
  ``_unpack_qkvz_ba`` pass over the whole ``[T, 12352]`` projection output, and
  leaves the weights themselves untouched.
* Causal conv1d, the q/k L2 norm, the v copy and the ``g``/``beta`` gating are
  one kernel (``_gdn_conv_prep``) reading the projection output once, rather
  than ``causal_conv1d_fn`` -> ``fused_post_conv_prep`` -> ``exp`` staging the
  whole ``[T, 8192]`` conv output through HBM in between.
* The output gate (``RMSNorm(o) * silu(z)``) reads ``z`` in place out of the
  projection output -- a strided view, no ``contiguous()`` copy -- and writes
  the ``[T, value_dim]`` matrix ``out_proj`` consumes.
* The recurrent state's cast-and-scatter writeback rides along in that same
  launch, and the chunk kernel's zero initial state is a cached buffer instead of
  a gather that is then zeroed.

What is deliberately *not* touched: the input projection, the chunk recurrence,
the output projection, the MoE, and the router projection. The MoE's top-k is a
discrete choice over 512 near-tied scores, so a one-ULP change anywhere upstream
of the router flips an expert for ~1 token in 50 and the layer stops matching.
Every kernel above is therefore bit-exact against the op it replaces: the norms
reproduce Inductor's asymmetric fusion (variance from the unrounded fp32 sum,
numerator from the bf16 residual it stored), the conv keeps vLLM's
activation-dtype products, and the output gate keeps the reference's tile shape
and warp count so its 128-wide reduction associates the same way.
``dev/exactstages.py`` checks each one. What is left is a single bf16 ULP on the
layer output past one token: the output projection's operand now comes from a
different allocation, and cuBLAS picks its kernel partly by pointer alignment, so
that GEMM's own reduction order shifts. It costs 4e-5 of the match ratio.

The routed experts keep running trtllm-gen's kernel, whose expert-major GEMM
reads each expert slab once however many tokens chose it -- a per-token gather
over 10 of 512 slabs measured 2x slower even at a single token. The shared
expert stays on cuBLAS too: a Triton launch costs more host time than the
split-K pair it would replace.

Under tensor parallelism the two parallel regions per layer (attention output,
MoE output) hand back un-reduced partials and the following norm does
all-reduce + residual-add + RMSNorm in one FlashInfer kernel -- the same fusion
vLLM's ``fuse_allreduce_rms`` pass applies
(``AllReduceFusedAddGemmaRMSNormPattern``). The MoE's partial is consumed by the
*next* layer's ``input_layernorm``, or by ``Qwen3NextModel.norm`` for the last
layer, so the model owns that half of the contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from ....infra.context import get_context
from ....infra.tp import _tp_size
from ..L2.flashinfer_allreduce_fusion import fused_allreduce_add_gemma_rmsnorm
from ..L1.gemma_rms_norm import GemmaRMSNorm
from ..L2.qwen3_next_gdn_attention import Qwen3NextGDNAttention
from ..L2.qwen3_next_attention import Qwen3NextAttention
from ..L2.shared_expert_moe import SharedExpertMoE


# ###########################################################################
# Gemma RMSNorm (+ residual add)
# ###########################################################################
@triton.jit
def _add_gemma_rmsnorm_kernel(
    X, R, W, Y, RO,
    stride_x, stride_r, stride_y, stride_ro,
    N: tl.constexpr, eps,
    HAS_RES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``y = (1 + w) * rms(x + r)``, ``ro = x + r`` (the bf16 sum, as vLLM keeps it).

    One program per row: the hidden size is 2048, so a row is a single 4 KiB
    vector load and the reduction stays inside one block.
    """
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_RES:
        r = tl.load(R + row * stride_r + cols, mask=mask, other=0.0).to(tl.float32)
        # Inductor fuses this pair asymmetrically and the MoE router downstream
        # is sensitive to the last bit, so keep its exact arithmetic: the
        # variance comes from the *unrounded* fp32 sum, the numerator from the
        # bf16 value that was stored as the new residual.
        f = x + r
        s = f.to(RO.dtype.element_ty)
        tl.store(RO + row * stride_ro + cols, s, mask=mask)
        num = s.to(tl.float32)
    else:
        f = x
        num = x
    var = tl.sum(f * f, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = num * rstd * (1.0 + w)
    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


def _gemma_rmsnorm(x: torch.Tensor, residual: torch.Tensor | None,
                   weight: torch.Tensor, eps: float):
    """``(norm, new_residual)``; ``new_residual`` is ``x`` itself when no residual."""
    n_rows, n = x.shape
    y = torch.empty_like(x)
    ro = torch.empty_like(x) if residual is not None else x
    if n_rows == 0:
        return y, ro
    block = triton.next_power_of_2(n)
    _add_gemma_rmsnorm_kernel[(n_rows,)](
        x, residual if residual is not None else x, weight, y, ro,
        x.stride(0), residual.stride(0) if residual is not None else 0,
        y.stride(0), ro.stride(0),
        N=n, eps=eps, HAS_RES=residual is not None, BLOCK=block,
        num_warps=8 if block >= 2048 else 4,
    )
    return y, ro


# ###########################################################################
# GDN: causal conv1d + silu + q/k L2 norm + v copy + gating, in one pass
# ###########################################################################
@triton.jit
def _gdn_conv_prep_kernel(
    X,                      # [T, D_PROJ] projection output, K-head interleaved
    CW,                     # [conv_dim, W] conv weight
    A_LOG, DT_BIAS,         # [HV]
    CACHE_IDX,              # [1] int32 -- conv-state slot for this sequence
    Q, K, V,                # [T, H, DK] / [T, H, DK] / [T, HV, DV]
    G, BETA,                # [T, HV] float32
    CSTATE,                 # conv-state cache
    stride_x, stride_cw,
    stride_cs_slot, stride_cs_dim, stride_cs_tok,
    T,
    GROUP: tl.constexpr,    # projection columns per K head: 2*DK + 2*VP*DV
    BA_OFF: tl.constexpr,   # first column of the b/a block in X
    KDIM: tl.constexpr,     # 2 * H * DK  (start of the V segment in conv order)
    NQ: tl.constexpr,       # number of q programs == H
    NK: tl.constexpr,       # NQ + H
    HV: tl.constexpr,       # number of V heads
    VP: tl.constexpr,       # V heads per K head
    D: tl.constexpr,        # channels per program (head dim)
    WIDTH: tl.constexpr,
    L2NORM_EPS: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """One program = one head (128 channels) x BLOCK_T tokens.

    ``program_id(1)`` picks the head: ``[0, H)`` -> Q, ``[H, 2H)`` -> K (both
    L2-normalized), beyond that -> V plus that V head's ``g``/``beta``. All three
    read their own contiguous 128-channel slice of the projection output, so the
    whole ``[T, conv_dim]`` conv output never has to exist in memory.

    Two index maps meet here. ``in_proj_qkvz`` emits one ``[q k v z]`` group per
    K head, which is where a head's columns live in ``X``; the conv weight and
    the conv-state cache are in the packed ``[q_all | k_all | v_all]`` order.
    Reading each off its own layout is what removes the baseline's separate
    deinterleave pass -- and it leaves the projection weight untouched, so the
    GEMM stays bit-for-bit the reference's.
    """
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = t < T
    d = tl.arange(0, D)

    if pid_h < NQ:                            # Q head pid_h
        cols = pid_h * GROUP + d
        cmix = pid_h * D + d
    elif pid_h < NK:                          # K head pid_h - NQ
        hh = pid_h - NQ
        cols = hh * GROUP + D + d
        cmix = NQ * D + hh * D + d
    else:                                     # V head pid_h - NK
        hv = pid_h - NK
        cols = (hv // VP) * GROUP + 2 * D + (hv % VP) * D + d
        cmix = KDIM + hv * D + d

    # 4-tap causal conv: four shifted loads of the same [BLOCK_T, D] tile. The
    # shifted rows overlap, so after the first load they come out of L1.
    acc = tl.zeros((BLOCK_T, D), dtype=tl.float32)
    for j in tl.static_range(WIDTH):
        tj = t - (WIDTH - 1 - j)
        m = tmask[:, None] & (tj >= 0)[:, None]
        xj = tl.load(X + tj[:, None].to(tl.int64) * stride_x + cols[None, :],
                     mask=m, other=0.0)
        wj = tl.load(CW + cmix * stride_cw + j)
        # vLLM's conv kernel accumulates activation-dtype products into an fp32
        # running sum; keep the same rounding so the recurrence sees the same
        # q/k/v the reference does.
        acc += (xj * wj[None, :]).to(tl.float32)
    acc = acc / (1.0 + tl.exp(-acc))                   # silu
    y = acc.to(Q.dtype.element_ty).to(tl.float32)      # the baseline stores bf16

    if pid_h < NK:
        inv = 1.0 / tl.sqrt(tl.sum(y * y, axis=1) + L2NORM_EPS)
        yn = (y * inv[:, None]).to(Q.dtype.element_ty)
        if pid_h < NQ:
            tl.store(Q + t[:, None].to(tl.int64) * (NQ * D)
                     + (pid_h * D + d)[None, :], yn, mask=tmask[:, None])
        else:
            tl.store(K + t[:, None].to(tl.int64) * (NQ * D)
                     + ((pid_h - NQ) * D + d)[None, :], yn, mask=tmask[:, None])
    else:
        hv = pid_h - NK
        tl.store(V + t[:, None].to(tl.int64) * (HV * D) + (hv * D + d)[None, :],
                 y.to(V.dtype.element_ty), mask=tmask[:, None])
        # g = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b)
        # ``in_proj_ba`` is grouped per K head as ``[b(VP) a(VP)]``.
        ba = BA_OFF + (hv // VP) * 2 * VP + (hv % VP)
        bv = tl.load(X + t.to(tl.int64) * stride_x + ba, mask=tmask,
                     other=0.0).to(tl.float32)
        av = tl.load(X + t.to(tl.int64) * stride_x + (ba + VP), mask=tmask,
                     other=0.0).to(tl.float32)
        alog = tl.load(A_LOG + hv).to(tl.float32)
        dtb = tl.load(DT_BIAS + hv).to(tl.float32)
        u = av + dtb
        sp = tl.where(u > 0, u + tl.log(1.0 + tl.exp(-u)), tl.log(1.0 + tl.exp(u)))
        sp = tl.where(u <= SOFTPLUS_THRESHOLD, sp, u)
        tl.store(G + t * HV + hv, tl.exp(-tl.exp(alog) * sp), mask=tmask)
        tl.store(BETA + t * HV + hv, tl.sigmoid(bv), mask=tmask)

    # The tail programs also refresh this sequence's conv state: the last
    # ``WIDTH - 1`` *inputs*, which is what the next step's conv rolls in.
    if pid_t == tl.num_programs(0) - 1:
        slot = tl.load(CACHE_IDX).to(tl.int64)
        if slot >= 0:
            for j in tl.static_range(WIDTH - 1):
                ts = T - (WIDTH - 1) + j
                xv = tl.load(X + ts * stride_x + cols,
                             mask=(d * 0 + ts) >= 0, other=0.0)
                tl.store(
                    CSTATE + slot * stride_cs_slot + cmix * stride_cs_dim
                    + j * stride_cs_tok,
                    xv,
                )


def _gdn_conv_prep(proj, conv_weight, a_log, dt_bias, cache_idx, conv_state,
                   n_tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
                   ba_off):
    dev, dt = proj.device, proj.dtype
    q = torch.empty(n_tokens, num_k_heads, head_k_dim, dtype=dt, device=dev)
    k = torch.empty(n_tokens, num_k_heads, head_k_dim, dtype=dt, device=dev)
    v = torch.empty(n_tokens, num_v_heads, head_v_dim, dtype=dt, device=dev)
    g = torch.empty(n_tokens, num_v_heads, dtype=torch.float32, device=dev)
    beta = torch.empty(n_tokens, num_v_heads, dtype=torch.float32, device=dev)
    if n_tokens == 0:
        return q, k, v, g, beta
    width = conv_weight.shape[1]
    # Tuned on B200: the four shifted loads of the conv window want a small
    # token tile (the overlap then stays in L1) and few warps per 128-channel row.
    block_t = 16 if n_tokens > 32 else 8
    _gdn_conv_prep_kernel[(triton.cdiv(n_tokens, block_t),
                           2 * num_k_heads + num_v_heads)](
        proj, conv_weight, a_log, dt_bias, cache_idx,
        q, k, v, g, beta, conv_state,
        proj.stride(0), conv_weight.stride(0),
        conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
        n_tokens,
        GROUP=2 * head_k_dim + 2 * (num_v_heads // num_k_heads) * head_v_dim,
        BA_OFF=ba_off,
        KDIM=2 * num_k_heads * head_k_dim,
        NQ=num_k_heads, NK=2 * num_k_heads, HV=num_v_heads,
        VP=num_v_heads // num_k_heads,
        D=head_k_dim, WIDTH=width,
        L2NORM_EPS=1e-6, SOFTPLUS_THRESHOLD=20.0,
        BLOCK_T=block_t, num_warps=2,
    )
    return q, k, v, g, beta


# ###########################################################################
# GDN epilogue: RMSNorm(o) * silu(z) (z read in place from the projection),
# plus the recurrent-state writeback riding along in the same launch
# ###########################################################################
@triton.jit
def _rmsnorm_gated_kernel(
    O, Z, W, Y, SRC, DST, IDX,
    stride_o, stride_z, stride_y, stride_dst_slot,
    n_rows, n_gate_blocks, state_numel, eps,
    HV: tl.constexpr, D: tl.constexpr,
    GROUP: tl.constexpr, VP: tl.constexpr,
    ROWS: tl.constexpr, BLOCK_S: tl.constexpr,
):
    """``y[t, h] = rms(o[t, h]) * w * silu(z[t, h])`` over head vectors.

    A "row" is one ``(token, v-head)`` pair. ``Z`` is a strided window into the
    input projection's output, so the gate never needs materializing. Programs
    past ``n_gate_blocks`` instead cast the chunk kernel's final recurrent state
    into its cache slot -- unrelated work, but it is two more launches otherwise
    and nothing between them has to be ordered.
    """
    pid = tl.program_id(0)
    if pid >= n_gate_blocks:
        slot = tl.load(IDX).to(tl.int64)
        off = (pid - n_gate_blocks) * BLOCK_S + tl.arange(0, BLOCK_S)
        m = off < state_numel
        v = tl.load(SRC + off, mask=m, other=0.0)
        tl.store(DST + slot * stride_dst_slot + off,
                 v.to(DST.dtype.element_ty), mask=m)
        return
    r = pid * ROWS + tl.arange(0, ROWS)
    rmask = r < n_rows
    t = r // HV
    h = r % HV
    d = tl.arange(0, D)
    o = tl.load(O + r[:, None].to(tl.int64) * stride_o + d[None, :],
                mask=rmask[:, None], other=0.0).to(tl.float32)
    # ``z`` is the tail of each K head's group, after q, k and its VP v heads.
    zcol = (h // VP) * GROUP + (2 + VP + (h % VP)) * D
    z = tl.load(Z + t[:, None].to(tl.int64) * stride_z + zcol[:, None]
                + d[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    w = tl.load(W + d).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(o * o, axis=1) / D + eps)
    y = o * rstd[:, None] * w[None, :]
    y = y * (z * tl.sigmoid(z))
    tl.store(Y + t[:, None].to(tl.int64) * stride_y + (h * D)[:, None] + d[None, :],
             y.to(Y.dtype.element_ty), mask=rmask[:, None])


def _rmsnorm_gated_store(o, z_src, weight, eps, n_tokens, num_v_heads,
                         head_v_dim, group, v_per_k, final_state,
                         recurrent_full, state_idx):
    out = torch.empty(n_tokens, num_v_heads * head_v_dim,
                      dtype=o.dtype, device=o.device)
    n_rows = n_tokens * num_v_heads
    # Same tile shape and warp count as the op this replaces: ``tl.sum`` over the
    # head vector reduces in whatever order the layout gives it, and the MoE
    # router two ops downstream resolves a different last bit into a different
    # expert. Costs nothing -- the reference config is already the right one for
    # a 128-wide row.
    sms = torch.cuda.get_device_properties(o.device).multi_processor_count
    rows = min(triton.next_power_of_2(triton.cdiv(n_rows, 2 * sms)), 4)
    block_s = 1024
    n_gate = triton.cdiv(n_rows, rows)
    n_state = triton.cdiv(final_state.numel(), block_s)
    _rmsnorm_gated_kernel[(n_gate + n_state,)](
        o, z_src, weight, out, final_state, recurrent_full, state_idx,
        o.stride(0), z_src.stride(0), out.stride(0), recurrent_full.stride(0),
        n_rows, n_gate, final_state.numel(), eps,
        HV=num_v_heads, D=head_v_dim, GROUP=group, VP=v_per_k,
        ROWS=rows, BLOCK_S=block_s,
        num_warps=1,
    )
    return out


_GDN_CHUNK = None


def _gdn_chunk_fn():
    """FlashInfer's GDN chunk kernel, resolved once (the import is not free)."""
    global _GDN_CHUNK
    if _GDN_CHUNK is None:
        from flashinfer.gdn_prefill import chunk_gated_delta_rule
        _GDN_CHUNK = chunk_gated_delta_rule
    return _GDN_CHUNK


def fused_ar_norm(norm: GemmaRMSNorm, hidden_states, residual, fuse: bool):
    """``norm(all_reduce(hidden_states), residual)``, fused when ``fuse``."""
    if fuse:
        # Opaque under torch.compile: tracing into FlashInfer's fused
        # collective hits Python logging / datetime and aborts Dynamo.
        if torch.compiler.is_compiling():
            return torch.ops.fastkernels.fused_allreduce_add_gemma_rmsnorm(
                hidden_states, residual, norm.weight, float(norm.variance_epsilon),
            )
        return fused_allreduce_add_gemma_rmsnorm(hidden_states, residual, norm)
    return norm(hidden_states, residual)


class Qwen3NextDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        # Only worth deferring when there is a collective to defer.
        self.fuse_ar_norm = _tp_size() > 1

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3NextGDNAttention(
                hidden_size=config.hidden_size,
                num_k_heads=config.linear_num_key_heads,
                num_v_heads=config.linear_num_value_heads,
                head_k_dim=config.linear_key_head_dim,
                head_v_dim=config.linear_value_head_dim,
                layer_idx=layer_idx,
                conv_kernel_size=config.linear_conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                reduce_output=not self.fuse_ar_norm,
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                layer_idx=layer_idx,
                rms_norm_eps=config.rms_norm_eps,
                reduce_output=not self.fuse_ar_norm,
            )
        else:
            raise ValueError(f"Invalid layer_type: {self.layer_type}")

        # MoE for all Qwen3-Next layers (every layer is sparse).
        self.mlp = SharedExpertMoE(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            moe_intermediate_size=config.moe_intermediate_size,
            routing="softmax",
            correction_bias=False,
            renormalize=config.norm_topk_prob,
            routed_scaling_factor=1.0,
            shared_expert_intermediate_size=config.shared_expert_intermediate_size,
            shared_expert_attr_name="shared_expert",
            shared_expert_gate=True,
            reduce_results=not self.fuse_ar_norm,
        )

        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # The fast path owns the whole GDN block; it needs the un-sharded
        # single-GPU shapes the kernels above assume.
        self._fast = (
            self.layer_type == "linear_attention"
            and not self.fuse_ar_norm
            and _tp_size() == 1
        )
        self._ready = False
        self._zero_state: torch.Tensor | None = None

    # -- weight preparation ------------------------------------------------
    def process_weights_after_loading(self) -> None:
        """Run the submodules' own preparation, then arm the fused GDN path.

        Nothing is rewritten here: the fused kernels read ``in_proj``'s output in
        the interleaved layout the checkpoint already produces, so the weights
        stay exactly as the reference leaves them -- no second copy, and the
        reference forward keeps working. All this needs is the GDN block's own
        hook having run, which is what merges ``in_proj_qkvz`` and ``in_proj_ba``
        into the single GEMM both paths use.
        """
        for sub in self.modules():
            if sub is self:
                continue
            fn = getattr(sub, "process_weights_after_loading", None)
            if callable(fn):
                fn()
        if not self._fast:
            return
        la = self.linear_attn
        w = la._in_proj_w
        self._ready = bool(
            w is not None
            and w.shape[0] == la._qkvz_dim + 2 * la.num_v_heads
            and la._qkvz_dim == 2 * la.key_dim + 2 * la.value_dim
            # the kernels use one head dim for q, k and v alike
            and la.head_k_dim == la.head_v_dim
        )

    # -- GDN fast path -----------------------------------------------------
    def _gdn_forward(self, hidden_states, state_manager, md):
        la = self.linear_attn
        la._ensure_triton_allocator(hidden_states.device)
        x = hidden_states.reshape(-1, la.hidden_size)
        n = x.shape[0]

        proj = F.linear(x, la._in_proj_w)
        cu = md.query_start_loc_int32
        if cu is None:
            cu = md.non_spec_query_start_loc.to(torch.int32)
        q, k, v, g_exp, beta = _gdn_conv_prep(
            proj, la.conv1d.weight, la.A_log, la.dt_bias,
            md.non_spec_state_indices_tensor,
            state_manager.gdn_conv[la.layer_idx],
            n, la.local_k_heads, la.local_v_heads, la.head_k_dim, la.head_v_dim,
            la._qkvz_dim,
        )

        recurrent_full = state_manager.recurrent[la.layer_idx]
        if self._zero_state is None:
            self._zero_state = torch.zeros(
                1, la.local_v_heads, la.head_k_dim, la.head_v_dim,
                dtype=torch.float32, device=x.device,
            )
        init_state = self._zero_state
        o, final_state = _gdn_chunk_fn()(
            q=q, k=k, v=v, g=g_exp, beta=beta,
            initial_state=init_state, output_final_state=True, cu_seqlens=cu,
        )
        gated = _rmsnorm_gated_store(
            o.reshape(-1, la.head_v_dim), proj, la.norm.weight, la.norm.eps,
            n, la.local_v_heads, la.head_v_dim,
            2 * la.head_k_dim + 2 * la.v_per_k * la.head_v_dim, la.v_per_k,
            final_state, recurrent_full, md.non_spec_state_indices_tensor,
        )
        return F.linear(gated, la.out_proj.weight)

    def forward(self, hidden_states, residual, positions=None,
                rotary_emb=None, state_manager=None):
        fast = self._fast and self._ready
        md = None
        if fast:
            ctx = get_context()
            md = ctx.kda_metadata
            if state_manager is None:
                state_manager = ctx.kda_state
            fast = (
                md is not None and state_manager is not None
                and hidden_states.dim() == 2
                # One prefill sequence covering the whole batch: the conv kernel
                # treats [0, T) as a single causal run, and the recurrence gets a
                # single state slot.
                and md.num_prefills == 1 and md.num_decode_tokens == 0
                and md.num_prefill_tokens == hidden_states.shape[0]
                and md.state_indices is not None
                and md.state_indices.numel() == 1
                # ... starting from zero state, which is what lets the chunk
                # kernel take a shared zero buffer instead of a masked gather.
                and md.has_initial_state is not None
                and not md.all_have_initial_state
                and not md.any_have_initial_state
            )
        if not fast:
            return self._forward_ref(hidden_states, residual, positions,
                                     rotary_emb, state_manager)

        w_in = self.input_layernorm.weight
        eps = self.input_layernorm.variance_epsilon
        if residual is None:
            # Layer 0: the input is the vocab-parallel embedding's output, which
            # is already reduced, and there is no residual stream yet.
            hidden_states, residual = _gemma_rmsnorm(hidden_states, None, w_in, eps)
        else:
            hidden_states, residual = _gemma_rmsnorm(
                hidden_states, residual, w_in, eps)

        hidden_states = self._gdn_forward(hidden_states, state_manager, md)

        hidden_states, residual = _gemma_rmsnorm(
            hidden_states, residual, self.post_attention_layernorm.weight,
            self.post_attention_layernorm.variance_epsilon,
        )
        return self.mlp(hidden_states), residual

    # -- reference path ----------------------------------------------------
    def _forward_ref(self, hidden_states, residual, positions,
                     rotary_emb, state_manager):
        """The baseline composition, verbatim.

        Used for full-attention layers, decode / multi-sequence / warm-state GDN
        batches, tensor parallelism, and any config the fused kernels do not
        cover. Nothing above mutates a weight, so this stays exactly as correct
        as the baseline it is copied from.
        """
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = fused_ar_norm(
                self.input_layernorm, hidden_states, residual, self.fuse_ar_norm,
            )

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states, state_manager=state_manager,
            )
        else:
            hidden_states = self.self_attn(
                hidden_states, rotary_emb=rotary_emb, positions=positions,
                state_manager=state_manager,
            )

        hidden_states, residual = fused_ar_norm(
            self.post_attention_layernorm, hidden_states, residual,
            self.fuse_ar_norm,
        )
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual
