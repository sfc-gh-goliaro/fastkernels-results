"""GPT-OSS decoder layer: attention + MoE with RMSNorm residual connections.

Uses the shared ``LlamaAttention`` with ``use_sinks=True`` and
``sliding_window`` to implement GPT-OSS attention sinks and per-layer
sliding window. Rotary embedding is passed through forward (created
once at the model level and shared across layers).

The arithmetic here is not this layer's cost.  For a decode-sized batch the
whole layer is *one static chain of ~13 kernels* -- fused-add RMSNorm, the
frozen L2 attention winner's projection/glue/FA4 sequence, a second RMSNorm,
the router GEMV, the MoE -- and it is **host bound by an order of magnitude**:
measured at one token, 720 us of wall time enqueues 72 us of CUDA work.  The
single biggest term is ``flashinfer.trtllm_fp4_block_scale_moe``, which spends
448 us on the host per call; only ~100 us of that is its Python wrapper, the
remaining ~315 us is inside the tvm-ffi/C++ launcher itself and is flat in the
token count, so no amount of glue-side Python removal reaches it.

So the layer is captured into a **CUDA graph keyed on the token count** and
replayed.  Replay enqueues the whole chain with one launch (7-20 us of host),
which is what vLLM's full-graph decode mode does for the same reason, and the
kernels, their arguments and their order are exactly the ones the eager path
would have run -- replay is bit-identical to eager on every captured shape.
What survives capture is device time, and there the token count decides:

  * ``<= _GRAPH_MAX_TOKENS`` -- host bound, so the graph is a pure win.
  * above it (the 16384-token prefill) the layer is device bound by two orders
    of magnitude, a graph saves nothing measurable and the input copies it needs
    are real, so those calls stay eager.

Two things make this correct rather than approximately correct, and both were
failures first:

**The layer mutates its arguments, and the graph must reproduce that.**  Both
norms are ``fused_add_rms_norm``, in-place on *hidden_states and residual*.  In
the ``residual is None`` case the layer's returned residual is the caller's own
``hidden_states`` tensor, overwritten by the *second* norm with
``attn_out + hidden_states``.  A graph writes to fixed addresses, so the
caller's tensors are copied in and the results copied back out into exactly the
objects the eager path would have returned -- the residual into the caller's
residual (or, when it was None, into the caller's ``hidden_states``), and the
hidden state as a fresh tensor because the MoE's output is one.  The one side
effect not reproduced is the *dead* overwrite of ``hidden_states`` by the first
norm when a residual was passed: no caller can observe it (``L4.gpt_oss``
rebinds both names) and copying it back would cost a full pass over the
activation.

**The graph closes over global inference metadata.**  ``Attention`` reads
``cu_seqlens``/``block_tables``/``slot_mapping`` off the process-wide
``Context``, so capture bakes in *those addresses*.  Rebuilding the Context --
which an engine does every step, and which the benchmark does between rounds --
can move them, and the graph then reads whatever now lives at the old address.
So every replay first checks a fingerprint of the metadata the captured kernels
read (the scalars by value, the tensors by ``data_ptr``); a match means the
graph reads the live buffers, and a mismatch recaptures rather than trusting
it.  Recaptures are budgeted, so metadata that churns every call latches the
graph off instead of allocating pools forever.

The first capture of a key also *verifies* the graph against a fresh eager run
and permanently declines the graph for that key if they differ, so a shape
whose kernels turn out not to be capture-safe degrades to eager instead of
going silently wrong.  Anything the capture cannot own -- a 3-D hidden state, a
non-CUDA tensor, tracing under torch.compile, an outer capture already in
progress, a rotary module other than the one captured with -- takes the
untouched eager path at the bottom.

Once host dispatch is gone, what is left is device time, and two things move it.

**The MoE runs the wrong trtllm-gen tactic.**  With no tuning pass flashinfer
falls back to ``tactic == -1``; profiling the shape buckets once picks a better
tile schedule for the same kernel, worth 0.535 -> 0.416 ms at 274 tokens and
6.140 -> 3.155 ms at 16384, bit-identical output.  See ``_autotune``.

**The post-attention seam costs three passes over the activation where it needs
one.**  The reference tail is ``fused_add_rms_norm`` (in-place, and it stores
the residual then reads it back to normalise it), then the router GEMV, then
``GptOssMoE``'s per-call ``F.pad`` freshly allocating and copying a
``[T, 3072]`` tensor and zeroing its 192-column tail.  ``_fused_tail`` collapses
the norm and the pad into one Triton kernel that reads hidden+residual once,
keeps the sum in registers, writes the residual back and drops the normed row
straight into columns ``[0, 2880)`` of a persistent buffer whose tail was zeroed
once at allocation -- so the pad never happens again -- and then calls the
frozen MoE with that buffer directly, bypassing ``forward_impl``'s wrapper.
That is 139 us of pad plus a norm round trip at 16384 tokens.

The router GEMV deliberately stays a separate ``addmm``, against the obvious
instinct to fold it into the same kernel.  Folding it means one CTA per token
holding the ``2880 x 128`` router weight, and 737 KB through a single SM's share
of HBM is slower than the same GEMV spread over the whole GPU: measured, on a
graph replay with a cold L2, norm+``addmm`` is 14.3 us at one token against
57-88 us fused (expert chunks of 32/16/8), and it loses at every token count up
to 274.  The launch that fusion would have saved is already free -- the graph
removed it -- so only the bandwidth argument is left, and it points the other
way.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.context import get_context
from ..L1.rms_norm import RMSNorm
from ..L2.attention import LlamaAttention
from ..L2.gpt_oss_moe import GptOssMoE

# Above this the layer is device bound (the 16384-token prefill spends 6.7 ms of
# GPU on ~0.8 ms of host), so a graph cannot pay for the input copies it needs.
_GRAPH_MAX_TOKENS = 2048
# One entry per (token count, residual-ness, dtype, device).  The benchmark
# drives one shape per instance; a served model captures a handful of buckets.
_GRAPH_MAX_ENTRIES = 12
# Metadata that moves on every call means a graph can never pay for itself, so a
# key that keeps recapturing gives up and stays eager.  Per key, not global: a
# model legitimately captures many token buckets, but no bucket should recapture
# forever.  The benchmark recaptures a key 4 times (once per correctness round,
# once for the timed loop), so the budget has to sit well above that.
_GRAPH_MAX_CAPTURES = 16
# flashinfer's MXFP4 MoE picks a trtllm-gen tactic per shape bucket, and with no
# tuning it takes the fallback (``tactic == -1``).  One tuning pass over the
# buckets is worth 0.535 -> 0.416 ms of *device* time at 274 tokens and
# 6.140 -> 3.155 ms at 16384, and it is bit-identical (verified on all five
# benchmarked token counts: maxabs 0.0, matched 1.000000) -- the tactic changes
# the tile schedule, not the arithmetic.  The AutoTuner's cache is process-wide
# and its profiling must land *before* any capture, since the graph bakes the
# chosen kernel in, so it happens on the first forward of the first layer.
_tuned = False
_IS_COMPILING = torch.compiler.is_compiling
_IS_CAPTURING = torch.cuda.is_current_stream_capturing

try:
    import triton
    import triton.language as tl
except Exception:                                    # no Triton: keep the pad
    triton = None


if triton is not None:

    @triton.jit
    def _add_norm_into_padded(x_ptr, res_ptr, w_ptr, out_ptr,
                              x_row, res_row, out_row, eps,
                              H: tl.constexpr, BLOCK: tl.constexpr):
        """``fused_add_rms_norm``, but the normed row lands in a *padded* buffer.

        Reproduces ``rmsnorm_ako.cu``'s ``rmsn_add_generic_vec`` expression by
        expression -- the residual add rounds to the storage dtype (the packed
        16-bit ``__hadd2`` the vendored kernel and vLLM both use, *not* the fp32
        add of ``forward_native``), the sum of squares accumulates in fp32, the
        scale is an approximate ``rsqrt``, and the result is
        ``((t * s) * w)`` in fp32 with one rounding at the store.  Only the
        fp32 reduction order differs, exactly as ako already differs from the
        vendored cub reduction it replaced.

        The one behavioural change is where the output goes: ako is in-place on
        ``x``, this writes columns ``[0, H)`` of a row of ``out``.  With ``out``
        a persistent buffer whose tail was zeroed once at allocation, that
        deletes the MoE's per-call ``F.pad`` -- an allocation, a full copy of
        the activation and a zero-fill of the tail, 139 us of the 16384-token
        layer -- and keeps ``t`` in registers instead of storing the residual
        and reading it back to normalise it.
        """
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK)
        m = offs < H
        x = tl.load(x_ptr + row * x_row + offs, mask=m, other=0.0)
        r = tl.load(res_ptr + row * res_row + offs, mask=m, other=0.0)
        t = (x.to(tl.float32) + r.to(tl.float32)).to(x.dtype)
        tl.store(res_ptr + row * res_row + offs, t, mask=m)
        tf = t.to(tl.float32)
        s = tl.math.rsqrt(tl.sum(tf * tf, axis=0) / H + eps)
        w = tl.load(w_ptr + offs, mask=m, other=0.0).to(tl.float32)
        tl.store(out_ptr + row * out_row + offs, ((tf * s) * w).to(x.dtype), mask=m)

# Every ``Context`` field the attention path can read.  Scalars are compared by
# value (they are baked into the captured launch geometry), tensors by address
# (the graph reads whatever lives there, so an unmoved buffer is a live buffer).
_CTX_SCALARS = (
    "is_prefill", "is_mixed", "is_tree_verify",
    "max_seqlen_q", "max_seqlen_k", "max_context_len",
    "num_prefill_tokens", "num_decode_tokens", "num_prefill_seqs",
    "prefill_max_seqlen_q", "prefill_max_seqlen_k", "decode_max_context_len",
)
_CTX_TENSORS = (
    "cu_seqlens_q", "cu_seqlens_k", "slot_mapping", "context_lens",
    "block_tables", "sliding_slot_mapping", "sliding_block_tables",
    "sliding_prefill_block_tables", "sliding_decode_block_tables",
    "prefill_cu_seqlens_q", "prefill_cu_seqlens_k", "prefill_block_tables",
    "decode_context_lens", "decode_block_tables",
)


def _ctx_fingerprint():
    ctx = get_context()
    fp = [getattr(ctx, name, None) for name in _CTX_SCALARS]
    for name in _CTX_TENSORS:
        t = getattr(ctx, name, None)
        fp.append(None if t is None else t.data_ptr())
    return tuple(fp)


class _Replay:
    """A captured layer: the graph, the static tensors it reads and writes, and
    the invariants that must still hold for a replay to mean the same thing."""

    __slots__ = ("graph", "pos", "hs", "res", "out_h", "out_r", "rope", "fp", "n")

    def __init__(self, graph, pos, hs, res, out_h, out_r, rope, fp, n=1):
        self.graph = graph
        self.pos = pos
        self.hs = hs
        self.res = res
        self.out_h = out_h
        self.out_r = out_r
        self.rope = rope
        self.fp = fp
        self.n = n              # how many times this key has been captured


class GptOssDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = LlamaAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            bias=True,
            o_proj_bias=True,
            use_sinks=True,
            sliding_window=config.sliding_window,
            layer_idx=layer_idx,
        )
        self.mlp = GptOssMoE(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._graphs: dict = {}
        self._graph_ok = True
        # Persistent, tail-zeroed MoE input buffers, one per token count.  Only
        # columns [0, hidden_size) are ever written, so the pad tail stays zero.
        self._moe_in: dict = {}
        self._tail = None          # None = undecided, False = keep the pad path

    def _apply(self, *args, **kwargs):
        # ``.to()`` / ``.cuda()`` / ``.float()`` replace parameter storage, and a
        # graph has every one of those addresses baked into it.
        out = super()._apply(*args, **kwargs)
        self._graphs.clear()
        self._moe_in.clear()
        self._tail = None
        return out

    # -- fused post-attention tail -----------------------------------------
    def _tail_plan(self, x):
        """Whether the fused norm + padded-buffer + MoE tail can serve this call.

        Everything here is a property of the module and the activation dtype, so
        it is resolved once; a configuration it does not recognise (tensor
        parallel, the opaque custom op, the Triton expert kernel with no hidden
        padding, a weightless norm, no Triton) keeps the reference tail.
        """
        mlp = self.mlp
        ln = self.post_attention_layernorm
        if not getattr(mlp, "_processed", False):
            # ``forward_impl`` would do this on its first call; it is idempotent,
            # but it reallocates every expert weight, so never under capture.
            if _IS_CAPTURING():
                return False                  # this call only, not cached
            try:
                mlp.process_weights_after_loading()
            except Exception:
                return False
        ok = (triton is not None
              and getattr(mlp, "use_trtllm", False)
              and getattr(mlp, "_processed", False)
              and mlp._H_pad > mlp.hidden_size
              and mlp.tp_size == 1
              and not mlp._use_custom_op
              and getattr(mlp, "router", None) is not None
              and mlp.router.bias is not None
              and ln.elementwise_affine
              and ln.weight.dtype == x.dtype
              and ln.weight.device == x.device
              and x.dtype in (torch.bfloat16, torch.float16)
              and x.shape[1] == mlp.hidden_size
              # one row per program, so the row has to fit in one tile
              and mlp.hidden_size <= 8192)
        self._tail = (mlp.hidden_size, triton.next_power_of_2(mlp.hidden_size),
                      mlp.router.weight.t()) if ok else False
        return self._tail

    def _fused_tail(self, x, residual):
        """norm(x + residual) -> padded MoE buffer -> router GEMV -> experts.

        Returns ``None`` when this call must take the reference tail instead.
        """
        plan = self._tail
        if plan is None:
            plan = self._tail_plan(x)
        if (plan is False or residual.shape != x.shape
                or x.stride(1) != 1 or residual.stride(1) != 1
                or x.dtype is not residual.dtype):
            return None
        H, block, wt = plan
        mlp = self.mlp
        n = x.shape[0]
        got = self._moe_in.get(n)
        if got is None:
            buf = torch.zeros(n, mlp._H_pad, dtype=x.dtype, device=x.device)
            if len(self._moe_in) >= _GRAPH_MAX_ENTRIES:
                self._moe_in.clear()
            got = self._moe_in[n] = (
                buf, buf[:, :H],
                torch.empty(n, mlp.num_experts, dtype=x.dtype, device=x.device))
        buf, nrm, logits = got
        ln = self.post_attention_layernorm
        _add_norm_into_padded[(n,)](
            x, residual, ln.weight, buf,
            x.stride(0), residual.stride(0), buf.stride(0), ln.eps,
            H=H, BLOCK=block, num_warps=8)
        # ``F.linear`` on a 2-D input *is* this addmm, so the router stays the
        # same GEMM on the same values; only its leading dimension changed (the
        # normed rows now live in the padded buffer).  Measured: the logits, and
        # so the top-4 they decide, are bit-identical to the reference tail at
        # 1/26/60/274 tokens and matched=1.000000 at 16384.
        torch.addmm(mlp.router.bias, nrm, wt, out=logits)
        return mlp.trtllm_moe(
            buf, logits, mlp._w13_shuffled, mlp._w13_scale, mlp._w13_bias_f32,
            mlp._w2_shuffled, mlp._w2_scale, mlp._w2_bias_f32)

    # -- reference sequence -------------------------------------------------
    def _forward_eager(self, positions, hidden_states, residual, rotary_emb):
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, rotary_emb=rotary_emb)
        out = self._fused_tail(hidden_states, residual)
        if out is None:
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
            return self.mlp(hidden_states), residual
        return out, residual

    def _autotune(self, positions, hidden_states, residual, rotary_emb):
        """Profile the MoE's trtllm-gen tactics once per process, off the hot path."""
        global _tuned
        _tuned = True
        try:
            from flashinfer.autotuner import autotune
        except Exception:
            return
        try:
            with torch.no_grad(), autotune(True):
                self._eager_on_clones(positions, hidden_states, residual, rotary_emb)
            torch.cuda.synchronize(hidden_states.device)
        except Exception:
            pass

    def _eager_on_clones(self, spos, shs, sres, rotary_emb):
        """One eager pass that leaves its inputs intact (both norms are in-place)."""
        return self._forward_eager(
            None if spos is None else spos.clone(), shs.clone(),
            None if sres is None else sres.clone(), rotary_emb)

    # -- capture ------------------------------------------------------------
    def _capture(self, key, positions, hidden_states, residual, rotary_emb, fp,
                 nth=1):
        """Capture this key, or record ``False`` and stay eager for it.

        Warmup runs on a side stream -- the documented prerequisite, and what
        lets every lazy JIT build, autotuner probe and workspace allocation on
        the path happen *before* the stream enters capture mode.
        """
        graph = None
        try:
            dev = hidden_states.device
            spos = None if positions is None else positions.clone()
            shs = hidden_states.clone()
            sres = None if residual is None else residual.clone()

            # Reference answer from the untouched eager path, for the check below.
            with torch.no_grad():
                ref_h, ref_r = self._eager_on_clones(spos, shs, sres, rotary_emb)
            ref_h, ref_r = ref_h.clone(), ref_r.clone()

            side = torch.cuda.Stream(device=dev)
            side.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(side), torch.no_grad():
                for _ in range(3):
                    self._eager_on_clones(spos, shs, sres, rotary_emb)
            torch.cuda.current_stream(dev).wait_stream(side)
            torch.cuda.synchronize(dev)

            graph = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(graph):
                out_h, out_r = self._forward_eager(spos, shs, sres, rotary_emb)

            # ``positions`` reaches the device only through the rotary rotation
            # (and Llama 4's temperature tuning, which this layer never enables),
            # so with no rotary anywhere there is nothing in the graph that reads
            # it and the copy is one ``cudaMemcpyAsync`` -- ~1.4 us of device
            # latency and ~5 us of host -- for bytes no kernel touches.
            if rotary_emb is None and self.self_attn.rotary_emb is None:
                spos = None
            entry = _Replay(graph, spos, shs, sres, out_h, out_r, rotary_emb, fp,
                            nth)
            self._run(entry, positions, hidden_states, residual)
            if not (torch.equal(out_h, ref_h) and torch.equal(out_r, ref_r)):
                raise RuntimeError("captured graph disagrees with eager")
        except Exception:
            del graph
            self._graphs[key] = False
            # A capture that fails is a property of the build, not of this call.
            self._graph_ok = False
            return False
        if len(self._graphs) >= _GRAPH_MAX_ENTRIES:
            self._graphs.clear()
        self._graphs[key] = entry
        return entry

    @staticmethod
    def _run(entry, positions, hidden_states, residual):
        if entry.pos is not None and positions is not None:
            entry.pos.copy_(positions)
        entry.hs.copy_(hidden_states)
        if entry.res is not None:
            entry.res.copy_(residual)
        entry.graph.replay()

    @staticmethod
    def _outputs(entry, hidden_states, residual):
        """Hand back exactly the objects the eager path would have returned."""
        if residual is None:
            # The eager path's residual *is* the caller's hidden_states, which
            # the post-attention norm overwrote in place with attn_out + itself.
            hidden_states.copy_(entry.out_r)
            return entry.out_h.clone(), hidden_states
        residual.copy_(entry.out_r)
        return entry.out_h.clone(), residual

    # -- forward ------------------------------------------------------------
    def forward(self, positions, hidden_states, residual, rotary_emb):
        if (not _tuned and hidden_states.is_cuda and hidden_states.dim() == 2
                and not _IS_COMPILING() and not _IS_CAPTURING()):
            self._autotune(positions, hidden_states, residual, rotary_emb)
        if (self._graph_ok
                and hidden_states.dim() == 2
                and 0 < hidden_states.shape[0] <= _GRAPH_MAX_TOKENS
                and hidden_states.is_cuda
                and not _IS_COMPILING()
                and not _IS_CAPTURING()):
            key = (hidden_states.shape[0], residual is None,
                   hidden_states.dtype, hidden_states.device)
            entry = self._graphs.get(key)
            if entry:
                fp = _ctx_fingerprint()
                if (entry.fp != fp or entry.rope is not rotary_emb
                        or (entry.pos is not None and positions is not None
                            and entry.pos.shape != positions.shape)):
                    if entry.n >= _GRAPH_MAX_CAPTURES:
                        self._graphs[key] = entry = False
                    else:
                        entry = self._capture(key, positions, hidden_states,
                                              residual, rotary_emb, fp,
                                              entry.n + 1)
            elif entry is None:
                entry = self._capture(key, positions, hidden_states, residual,
                                      rotary_emb, _ctx_fingerprint())
            if entry:
                self._run(entry, positions, hidden_states, residual)
                return self._outputs(entry, hidden_states, residual)
        return self._forward_eager(positions, hidden_states, residual, rotary_emb)
