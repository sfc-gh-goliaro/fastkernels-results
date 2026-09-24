"""Gated multi-head attention with an additive bias list (L2), latency-tuned.

Same dataflow as ``baseline.py``::

    q = Wq · q_x (+bq)   k = Wk · kv_x   v = Wv · kv_x   g = Wg · q_x
    scores = (q / sqrt(c_hidden)) · kᵀ + Σ biases
    out    = Wo · flatten(softmax(scores) · v * sigmoid(g))

The captured workload is tiny -- at most ~94 MFLOP per call, and
``prod(batch) · no_heads`` never exceeds 64 programs -- so essentially all of the
baseline's latency is launch and dispatch overhead rather than arithmetic. The
optimization is therefore about shortening the op graph, not about arithmetic
throughput:

* ``q`` and ``g`` both read ``q_x``, and ``k`` and ``v`` both read ``kv_x``, so
  the five projections collapse into three GEMMs against row-concatenated
  weights. The concatenations live in a lazily built, non-persistent cache, so
  ``state_dict()`` is unchanged and the cache follows in-place weight edits.
* The projections are driven by ``torch.mm`` / ``torch.addmm`` on the cached
  weights instead of the ``Linear`` submodules, which removes the module-call
  hops (each ``Linear`` delegates to an inner ``Matmul``, so every projection
  costs two).
* One Triton kernel does QKᵀ, the scale, the bias adds, the softmax, PV, the
  sigmoid gate and the head-to-feature relayout in a single launch. It reads the
  raw GEMM outputs by stride and writes its result straight into ``linear_o``'s
  input layout, so no host-side view, transpose, slice or reshape is needed and
  the relayout copy the baseline pays for disappears.
* Everything derived from shapes -- block sizes, warp count, and each bias's
  four-dimensional stride mapping onto ``[*, H, Q, K]`` -- is computed once per
  shape signature and memoised, so a steady-state call is three ``mm``s, one
  kernel launch and one ``view``.

Three paths sit behind the baseline's contract:

``_reference_forward``
    A faithful transcription of the baseline, used as the fallback for anything
    the fast paths do not cover and as the oracle for the parity tests.
``_fused_forward`` with ``_single_pass_core``
    The three fused GEMMs plus the one Triton kernel above. Requires the whole
    score tile to fit in registers, which the captured shapes do comfortably
    (``Q <= 32``, ``K <= 128``), so there is no key loop and no online softmax
    rescaling.
``_fused_forward`` with ``_sdpa_core``
    The three fused GEMMs plus ``F.scaled_dot_product_attention``, for a ``K``
    too large for the single-pass tile. Real OpenFold3 pair attention has
    ``K = n_res``, so this is the branch that keeps the module a usable
    implementation of the operator rather than a fit to the captured shapes.

On top of the single-pass path, ``_replay`` captures the whole fused forward into a
CUDA graph once per shape signature and replays it afterwards, trading four
per-call dispatches for one. Each replay copies the *current* inputs into the
captured static buffers and clones the static output, so nothing is memoised
across calls and no returned tensor can be overwritten by a later one. Capture is
declined -- silently, falling back to eager -- for a broadcast input that has no
non-overlapping static twin, while a stream is already capturing, or while autograd
is recording. The cache is cleared whenever the packed weights are rebuilt, because
a captured graph holds their addresses. Because the replay path routes every call
through one set of static buffers, it is not safe to call a single instance
concurrently from two threads; the operator is used single-threaded.

Two documented divergences from the baseline:

* The fast paths keep fp32 through the score scale, the bias add, the softmax and
  the gate, where the baseline rounds to bf16 after the scale divide and again
  before the softmax. The fast paths are therefore slightly *more* accurate, and
  the gap sits well inside the bf16 tolerance the bench compares at.
  Probabilities are still rounded to the value dtype before the PV product,
  matching the baseline's ``scores.to(dtype=value.dtype)``.
* A row whose every key is masked to ``-inf`` has no defined value: ``softmax``
  of an all-``-inf`` row is ``NaN``. The reference path and the single-pass
  Triton core both propagate that ``NaN``, matching the baseline. The
  scaled-dot-product-attention core does not -- cuDNN applies a safe softmax and
  returns finite values for such a row. Rows that retain at least one unmasked
  key agree on every path.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.linear import Linear

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - Triton ships with the target torch build
    triton = None
    tl = None

# Only these reach the fused cores. fp32 is routed to the reference path: it is
# compared at atol=1e-5, and Triton's default fp32 dot uses TF32, which would not
# hold that.
_FAST_DTYPES = (torch.bfloat16, torch.float16)

# Widest score tile the single-pass core will keep in registers. Beyond this the
# fused path switches to the scaled-dot-product-attention core.
_MAX_BLOCK_K = 256
_MAX_BLOCK_C = 128

# Bias delivery: pass the tensor to the kernel as-is at the recorded strides, or
# expand it to the full score shape and make it contiguous first.
_BIAS_AS_IS = 0
_BIAS_MATERIALIZE = 1

# Whether the fused path may replay a captured CUDA graph instead of issuing its
# four launches eagerly. Module-level so a measurement harness can compare the two
# configurations; nothing in the operator's own logic ever changes it.
_ENABLE_CUDA_GRAPHS = True

# Marker stored in the graph cache for a signature that was tried and cannot be
# captured, so the admission work is not repeated on every call.
_NOT_GRAPHABLE = object()


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    biases: list[torch.Tensor],
) -> torch.Tensor:
    """Core SDPA: scores = softmax(Q K^T + biases) V.

    Args:
        query: [*, H, Q, C_hidden]
        key:   [*, H, K, C_hidden]
        value: [*, H, V, C_hidden]
        biases: list of tensors broadcastable to [*, H, Q, K]

    Returns:
        [*, H, Q, C_hidden]
    """
    scores = torch.einsum("...qc,...kc->...qk", query, key)

    for b in biases:
        scores = scores + b

    scores = F.softmax(scores, dim=-1)

    return torch.einsum("...qk,...kc->...qc", scores.to(dtype=value.dtype), value)


if triton is not None:

    @triton.jit
    def _attention_gate_kernel(
        qg_ptr, kv_ptr, og_ptr, bias0_ptr, bias1_ptr,
        qg_batch, qg_row, qg_col,
        kv_batch, kv_row, kv_col,
        og_row, og_col,
        b0_batch, b0_head, b0_q, b0_k,
        b1_batch, b1_head, b1_q, b1_k,
        q_len, k_len, head_dim, hidden,
        scale,
        BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr,
        NUM_BIAS: tl.constexpr, HAS_GATE: tl.constexpr,
    ):
        """One program per (flattened batch, head): attention, gate and relayout.

        ``qg`` is the packed query/gate projection ``[N, Q, 2·H·C]`` and ``kv`` the
        packed key/value projection ``[N, K, 2·H·C]``; this program reads column
        block ``h·C`` for query and key and ``H·C + h·C`` for gate and value, so
        the packing is undone by pointer arithmetic rather than by host-side
        slicing. The result is written to ``og[N·Q, H·C]`` at column ``h·C``,
        which is exactly the layout ``linear_o`` consumes -- the transpose the
        baseline materializes as a copy is expressed here as a store pattern.

        Biases arrive as up to two pointers with an explicit
        ``(batch, head, query, key)`` stride each. A broadcast dimension arrives
        as stride 0, and no stride is assumed to be unit: the captured biases have
        key-axis strides of 4 and 16.
        """
        pid_n = tl.program_id(0)
        pid_h = tl.program_id(1)

        offs_q = tl.arange(0, BLOCK_Q)
        offs_k = tl.arange(0, BLOCK_K)
        offs_c = tl.arange(0, BLOCK_C)
        mask_q = offs_q < q_len
        mask_k = offs_k < k_len
        mask_c = offs_c < head_dim

        col_qk = pid_h * head_dim + offs_c
        col_gv = hidden + col_qk

        q = tl.load(
            qg_ptr + pid_n * qg_batch + offs_q[:, None] * qg_row + col_qk[None, :] * qg_col,
            mask=mask_q[:, None] & mask_c[None, :], other=0.0,
        )
        k = tl.load(
            kv_ptr + pid_n * kv_batch + offs_k[:, None] * kv_row + col_qk[None, :] * kv_col,
            mask=mask_k[:, None] & mask_c[None, :], other=0.0,
        )

        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale

        if NUM_BIAS > 0:
            scores += tl.load(
                bias0_ptr + pid_n * b0_batch + pid_h * b0_head
                + offs_q[:, None] * b0_q + offs_k[None, :] * b0_k,
                mask=mask_q[:, None] & mask_k[None, :], other=0.0,
            ).to(tl.float32)
        if NUM_BIAS > 1:
            scores += tl.load(
                bias1_ptr + pid_n * b1_batch + pid_h * b1_head
                + offs_q[:, None] * b1_q + offs_k[None, :] * b1_k,
                mask=mask_q[:, None] & mask_k[None, :], other=0.0,
            ).to(tl.float32)

        # Keys past k_len must not enter the softmax. A row that is entirely -inf
        # (every real key masked by a bias) yields NaN here, matching the
        # baseline's softmax.
        scores = tl.where(mask_k[None, :], scores, float("-inf"))
        probs = tl.exp(scores - tl.max(scores, 1)[:, None])
        probs = probs / tl.sum(probs, 1)[:, None]

        v = tl.load(
            kv_ptr + pid_n * kv_batch + offs_k[:, None] * kv_row + col_gv[None, :] * kv_col,
            mask=mask_k[:, None] & mask_c[None, :], other=0.0,
        )
        # Rounding the probabilities to the value dtype reproduces the baseline's
        # scores.to(dtype=value.dtype) before its second einsum.
        acc = tl.dot(probs.to(v.dtype), v, out_dtype=tl.float32)

        if HAS_GATE:
            g = tl.load(
                qg_ptr + pid_n * qg_batch + offs_q[:, None] * qg_row
                + col_gv[None, :] * qg_col,
                mask=mask_q[:, None] & mask_c[None, :], other=0.0,
            ).to(tl.float32)
            acc = acc / (1.0 + tl.exp(-g))

        tl.store(
            og_ptr + (pid_n * q_len + offs_q)[:, None] * og_row + col_qk[None, :] * og_col,
            acc.to(og_ptr.dtype.element_ty),
            mask=mask_q[:, None] & mask_c[None, :],
        )


def _tile(extent: int) -> int:
    """Smallest power-of-two tile covering *extent*; ``tl.dot`` needs at least 16."""
    return max(16, triton.next_power_of_2(extent))


def _warps_for(block_q: int, block_k: int, block_c: int) -> int:
    """Warp count for one score tile, from a table measured offline.

    Deliberately a table and not ``triton.autotune``: a tuning sweep would run
    inside the benchmark's timed region and perturb the measurement it is meant
    to improve.

    Both thresholds are measured, and only measured values appear here. Sweeping
    1/2/4/8 warps five times per specialization over every captured shape
    (``tests/warp_sweep.py``, raw runs in
    ``profile/of3_attention_v2_graph_replay/analysis/warp_sweep.json``) shows the
    warp count barely matters for the small tiles -- the pooled medians for
    ``16x16x32`` and ``16x16x64`` span 45.06 to 45.15 us across all four choices,
    which is inside the run-to-run spread -- and matters only for the widest tile,
    where ``32x128x32`` measures 73.72 / 66.52 / 66.94 / 65.55 us for 1 / 2 / 4 / 8
    warps. So: the fewest warps where the choice is free, and the most where it
    pays.
    """
    if block_q * block_k * block_c <= 16384:
        return 1
    return 8


def _tensor_signature(t: torch.Tensor) -> tuple:
    """Everything about *t* that a captured graph is specialized on.

    A graph bakes in addresses, extents and strides, and the kernel's launch
    parameters are derived from these, so two calls may share a captured graph only
    if all four agree. Values are deliberately absent -- those are copied in per
    call.
    """
    return (t.shape, t.stride(), t.dtype, t.device)


def _is_dense_layout(t: torch.Tensor) -> bool:
    """Whether *t*'s shape and strides describe a non-overlapping, dense region.

    A static replacement for *t* is only meaningful when the layout can be
    reproduced by ``empty_strided`` and written by ``copy_`` without two logical
    indices aliasing one address. Permuted views qualify -- the captured
    ``[1,1,4,16,16]`` bias at stride ``[1024,4,1,64,4]`` is one -- but a
    broadcast view does not, because a stride-0 axis maps many indices onto the
    same element.
    """
    extents = sorted((s, d) for s, d in zip(t.stride(), t.shape) if d != 1)
    expected = 1
    for stride, size in extents:
        if stride != expected:
            return False
        expected *= size
    return True


def _static_like(t: torch.Tensor) -> torch.Tensor:
    """An uninitialized tensor with *t*'s exact shape, stride, dtype and device.

    The stride must be preserved, not just the shape: the launch plan is keyed on
    the input strides and the kernel indexes the biases with them.
    """
    if t.is_contiguous():
        return torch.empty_like(t)
    return torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device)


def _collapse_batch_stride(dims: tuple, strides: tuple) -> int | None:
    """One stride reproducing ``sum(idx_i · strides_i)`` from the flat batch index.

    The kernel indexes the batch with a single ``pid_n · stride``, so a bias whose
    batch dimensions do not reduce to one stride cannot be read in place. Returns
    ``None`` in that case, and the caller materializes the bias instead.
    """
    found = None
    inner = 1
    for i in range(len(dims) - 1, -1, -1):
        if dims[i] != 1:
            if strides[i] % inner:
                return None
            step = strides[i] // inner
            if found is None:
                found = step
            elif step != found:
                return None
        inner *= dims[i]
    return 0 if found is None else found


def _bias_strides(bias: torch.Tensor, score_shape: tuple) -> tuple | None:
    """``(batch, head, query, key)`` strides for *bias* against the score shape.

    The baseline adds each bias to the scores with ordinary right-aligned
    broadcasting, so a bias's trailing three dimensions land on ``(H, Q, K)`` and
    everything before them on the batch prefix. A dimension of size 1 broadcasts
    and therefore contributes stride 0. Returns ``None`` when the mapping cannot
    be expressed as four strides.
    """
    rank = len(score_shape)
    if bias.dim() > rank:
        return None
    pad = rank - bias.dim()
    sizes = (1,) * pad + tuple(bias.shape)
    strides = (0,) * pad + tuple(bias.stride())
    aligned = []
    for size, stride, extent in zip(sizes, strides, score_shape):
        if size == 1:
            aligned.append(0)
        elif size == extent:
            aligned.append(stride)
        else:
            return None
    n_batch_dims = rank - 3
    batch_stride = _collapse_batch_stride(score_shape[:n_batch_dims],
                                          tuple(aligned[:n_batch_dims]))
    if batch_stride is None:
        return None
    return (batch_stride, aligned[-3], aligned[-2], aligned[-1])


class OF3Attention(nn.Module):
    """Standard multi-head attention with gating and bias list support.

    Reference: openfold3/core/model/primitives/attention.py Attention

    Args:
        c_q: Input dimension of query data
        c_k: Input dimension of key data
        c_v: Input dimension of value data
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        gating: Whether to gate output using query data
    """

    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
        q_bias: bool = False,
    ):
        super().__init__()
        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        self.linear_q = Linear(c_q, c_hidden * no_heads, bias=q_bias)
        self.linear_k = Linear(c_k, c_hidden * no_heads, bias=False)
        self.linear_v = Linear(c_v, c_hidden * no_heads, bias=False)
        self.linear_o = Linear(c_hidden * no_heads, c_q, bias=False)

        self.linear_g = None
        if gating:
            self.linear_g = Linear(c_q, c_hidden * no_heads, bias=False)

        # Derived state is held in plain attributes -- never Parameters and never
        # buffers -- so state_dict() carries exactly the baseline's keys. The
        # projection cache is filled on the first forward, because the bench
        # casts, re-randomizes and then load_state_dict()s the parameters after
        # construction; building it here would freeze the wrong weights.
        self._projection_cache = None
        self._plan_cache: dict = {}
        self._graph_cache: dict = {}
        self._hidden = c_hidden * no_heads
        self._scale = 1.0 / math.sqrt(c_hidden)
        # k and v are both applied to kv_x, so the baseline could not run with a
        # mismatched pair either; checked once so the fused kv weight can be
        # trusted per call.
        self._kv_shared = c_k == c_v

    # -- weight-derived cache -------------------------------------------------

    def _projection_sources(self) -> tuple[torch.Tensor, ...]:
        """Every parameter the fused weights are derived from."""
        linear_q = self.linear_q
        sources = (linear_q.weight, self.linear_k.weight, self.linear_v.weight,
                   self.linear_o.weight)
        if self.linear_g is not None:
            sources += (self.linear_g.weight,)
        if linear_q.bias is not None:
            sources += (linear_q.bias,)
        return sources

    @staticmethod
    def _identity(sources: tuple[torch.Tensor, ...]) -> tuple:
        """A key that changes whenever a source weight's contents could have.

        ``data_ptr`` witnesses every mutation that reallocates storage -- the
        bench's ``p.data = p.data.to(dtype)`` cast, a ``.data`` reassignment,
        ``load_state_dict(assign=True)``, ``module.to()``. ``_version`` witnesses
        the mutations that keep the storage: ``normal_``, ``copy_``, and the
        in-place copy a plain ``load_state_dict`` performs (measured: it bumps the
        parameter's ``_version`` and leaves ``data_ptr`` alone). ``dtype``,
        ``shape``, ``stride`` and ``device`` witness a metadata rebind that reuses
        the storage: ``p.data = p.data.t()`` on a *square* weight preserves
        ``data_ptr``, ``_version``, ``dtype`` and ``shape`` and changes only the
        stride -- and ``c_hidden * no_heads == c_q`` holds in every captured
        variant, so that weight really can be square here. Reading all six costs
        ~1.5 us per call, which is the price of the key being sound rather than
        merely sufficient for one caller.

        One limitation remains, and it cannot be closed cheaply:
        ``p.data.copy_(...)`` mutates the values without bumping ``p._version``,
        because ``.data`` hands out a fresh version counter. Such a write is
        invisible to any metadata-only key, and the cache would keep serving the
        previous weights. Detecting it would mean reading weight values on every
        call, which costs more than the fusion saves. Mutate parameters as
        ``p.copy_(...)`` under ``torch.no_grad()`` -- the supported idiom, and one
        this key does see. The bench performs no write through ``.data`` after the
        cache is built, so this cannot fire under ``fastkernels bench``.
        """
        return tuple(
            (t.data_ptr(), t._version, t.dtype, t.shape, t.stride(), t.device)
            for t in sources
        )

    def _fused_weights(self):
        """``(w_qg, w_kv, b_qg, w_o)`` ready for ``mm``/``addmm``, rebuilt on demand.

        ``w_qg`` maps ``c_q -> 2·H·C`` (query then gate; just ``H·C`` when gating
        is off), ``w_kv`` maps ``c_k -> 2·H·C`` (key then value), and ``w_o`` maps
        ``H·C -> c_q``. All are transposed views of row-concatenated weights,
        which is the operand layout ``F.linear`` hands cuBLAS anyway.
        """
        sources = self._projection_sources()
        identity = self._identity(sources)
        cached = self._projection_cache
        if cached is not None and cached[0] == identity:
            return cached[1]

        w_q = self.linear_q.weight
        if self.linear_g is None:
            w_qg = w_q.t()
        else:
            w_qg = torch.cat((w_q, self.linear_g.weight), 0).t()
        w_kv = torch.cat((self.linear_k.weight, self.linear_v.weight), 0).t()

        b_q = self.linear_q.bias
        if b_q is None:
            b_qg = None
        elif self.linear_g is None:
            b_qg = b_q
        else:
            # linear_g never carries a bias, so a zero tail is exact.
            b_qg = torch.cat((b_q, b_q.new_zeros(self._hidden)), 0)

        weights = (w_qg, w_kv, b_qg, self.linear_o.weight.t())
        self._projection_cache = (identity, weights)
        # Any captured graph holds the *addresses* of the previous packed weights,
        # so it is stale the moment they are rebuilt.
        self._graph_cache.clear()
        return weights

    # -- reference path -------------------------------------------------------

    def _prep_qkv(
        self, q_x: torch.Tensor, kv_x: torch.Tensor, apply_scale: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        if apply_scale:
            q = q / math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
        if self.linear_g is not None:
            g = torch.sigmoid(self.linear_g(q_x))
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        o = o.reshape(o.shape[:-2] + (-1,))
        return self.linear_o(o)

    def _reference_forward(
        self, q_x: torch.Tensor, kv_x: torch.Tensor, biases: list[torch.Tensor],
    ) -> torch.Tensor:
        """The baseline dataflow, unoptimized: fallback branch and parity oracle."""
        q, k, v = self._prep_qkv(q_x, kv_x)
        o = _attention(q, k, v, biases)
        o = o.transpose(-2, -3)
        return self._wrap_up(o, q_x)

    # -- fast-path admission --------------------------------------------------

    def _can_fuse(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: list[torch.Tensor],
        q_shape: torch.Size,
        kv_shape: torch.Size,
    ) -> bool:
        """Whether the fused-projection path applies to this call.

        Everything rejected here falls through to ``_reference_forward``, which
        implements the general operator.
        """
        if q_x.dtype not in _FAST_DTYPES or kv_x.dtype is not q_x.dtype:
            return False
        if not self._kv_shared:
            return False
        rank = len(q_shape)
        if rank < 2 or len(kv_shape) != rank:
            return False
        if q_shape[-1] != self.c_q or kv_shape[-1] != self.c_k:
            return False
        if q_shape[:-2] != kv_shape[:-2]:
            return False
        if q_shape[-2] == 0 or kv_shape[-2] == 0:
            return False
        if not (q_x.is_cuda and kv_x.is_cuda):
            return False
        score_rank = rank + 1
        for b in biases:
            if type(b) is not torch.Tensor or b.device != q_x.device:
                return False
            # The baseline adds the bias with ordinary type promotion, so a bias
            # wider than the activations keeps its own precision all the way
            # through the softmax. Both fused cores would narrow it instead --
            # the single-pass kernel accumulates scores in fp32, and the cuDNN
            # core casts its mask to the activation dtype -- which loses the
            # resolution the baseline retains and shows up as a parity failure
            # once the bias magnitude is large. Only an exact dtype match can be
            # served faithfully; anything else takes the reference path. Every
            # captured bias is already the activation dtype, so no benched shape
            # is affected.
            if b.dtype is not q_x.dtype or b.dim() > score_rank:
                return False
        if torch.is_grad_enabled() and self._needs_autograd(q_x, kv_x, biases):
            return False
        return True

    def _needs_autograd(
        self, q_x: torch.Tensor, kv_x: torch.Tensor, biases: list[torch.Tensor],
    ) -> bool:
        """Whether this call would have to build a backward graph.

        The fused cores are inference-only, so anything differentiable takes the
        reference path, which is ordinary autograd-visible torch.
        """
        if q_x.requires_grad or kv_x.requires_grad:
            return True
        for b in biases:
            if b.requires_grad:
                return True
        for p in self.parameters():
            if p.requires_grad:
                return True
        return False

    # -- per-shape launch plan ------------------------------------------------

    def _plan(
        self, q_shape: torch.Size, kv_shape: torch.Size, biases: list[torch.Tensor],
    ) -> tuple:
        """Memoised launch plan for one shape signature.

        Block sizes, warp count and every bias's four strides depend only on
        shapes and layouts, never on values, so they are computed once per
        signature. A steady-state call does one dict lookup instead.
        """
        key = (q_shape, kv_shape, tuple((b.shape, b.stride()) for b in biases))
        plan = self._plan_cache.get(key)
        if plan is None:
            plan = self._build_plan(q_shape, kv_shape, biases)
            self._plan_cache[key] = plan
        return plan

    def _build_plan(
        self, q_shape: torch.Size, kv_shape: torch.Size, biases: list[torch.Tensor],
    ) -> tuple:
        batch = tuple(q_shape[:-2])
        q_len = q_shape[-2]
        k_len = kv_shape[-2]
        n_batch = math.prod(batch)
        score_shape = batch + (self.no_heads, q_len, k_len)

        block_k = _tile(k_len) if triton is not None else 0
        block_c = _tile(self.c_hidden) if triton is not None else 0
        single_pass = (triton is not None
                       and block_k <= _MAX_BLOCK_K and block_c <= _MAX_BLOCK_C)
        if not single_pass:
            return (n_batch, q_len, k_len, score_shape, False, None, None)

        block_q = _tile(q_len)
        launch = (block_q, block_k, block_c, _warps_for(block_q, block_k, block_c))

        # More than two biases are folded into one before the kernel sees them;
        # the fold produces the full score shape, whose strides are canonical.
        canonical = (self.no_heads * q_len * k_len, q_len * k_len, k_len, 1)
        if len(biases) > 2:
            bias_plan = ((_BIAS_MATERIALIZE, canonical),)
        else:
            bias_plan = tuple(
                (_BIAS_AS_IS, strides) if strides is not None
                else (_BIAS_MATERIALIZE, canonical)
                for strides in (_bias_strides(b, score_shape) for b in biases)
            )
        return (n_batch, q_len, k_len, score_shape, True, launch, bias_plan)

    # -- fused path -----------------------------------------------------------

    def _fused_forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: list[torch.Tensor],
        q_shape: torch.Size,
        kv_shape: torch.Size,
    ) -> torch.Tensor:
        # Ordered deliberately: resolving the weights may invalidate the graph
        # cache, so it has to happen before a captured graph is consulted.
        weights = self._fused_weights()
        plan = self._plan(q_shape, kv_shape, biases)

        replayed = self._replay(q_x, kv_x, biases, q_shape, kv_shape, weights, plan)
        if replayed is not None:
            return replayed
        return self._fused_eager(q_x, kv_x, biases, q_shape, kv_shape, weights, plan)

    def _fused_eager(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: list[torch.Tensor],
        q_shape: torch.Size,
        kv_shape: torch.Size,
        weights: tuple,
        plan: tuple,
    ) -> torch.Tensor:
        """Three GEMMs plus one attention core, issued launch by launch."""
        w_qg, w_kv, b_qg, w_o = weights
        n_batch, q_len, k_len, score_shape, single_pass, launch, bias_plan = plan

        q_2d = q_x.reshape(n_batch * q_len, self.c_q)
        kv_2d = kv_x.reshape(n_batch * k_len, self.c_k)
        qg = torch.mm(q_2d, w_qg) if b_qg is None else torch.addmm(b_qg, q_2d, w_qg)
        kv = torch.mm(kv_2d, w_kv)

        if single_pass:
            og = self._single_pass_core(qg, kv, biases, n_batch, q_len, k_len,
                                        score_shape, launch, bias_plan)
        else:
            og = self._sdpa_core(qg, kv, biases, n_batch, q_len, k_len, score_shape)
        return torch.mm(og, w_o).view(q_shape[:-1] + (self.c_q,))

    # -- captured-graph replay ------------------------------------------------

    def _replay(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: list[torch.Tensor],
        q_shape: torch.Size,
        kv_shape: torch.Size,
        weights: tuple,
        plan: tuple,
    ) -> torch.Tensor | None:
        """Replay a captured graph for this signature, or ``None`` to go eager.

        At four launches the residual per-call cost is dispatch, not GPU work, so
        the whole fused forward is captured once per shape signature and replayed
        afterwards. Every replay copies the *current* inputs into the captured
        static buffers first and clones the static output on the way out, so
        nothing is reused across calls and no returned tensor can be overwritten
        by a later one.
        """
        if not (_ENABLE_CUDA_GRAPHS and plan[4]):
            return None

        # Transient states are checked before the cache is touched, and never
        # recorded. Capture is impossible while autograd is recording or while an
        # outer capture owns the stream, but both conditions end -- writing
        # `_NOT_GRAPHABLE` for them would leave a signature eager forever after one
        # such call. The marker is reserved for permanent, layout-based rejection.
        if torch.is_grad_enabled() or torch.cuda.is_current_stream_capturing():
            return None

        key = (_tensor_signature(q_x), _tensor_signature(kv_x),
               tuple(_tensor_signature(b) for b in biases),
               plan, self.gating, id(self._projection_cache[0]),
               # A graph captured under autocast bakes in that autocast's dtype
               # choices, so it must not be replayed outside it, or vice versa.
               torch.is_autocast_enabled(), torch.get_autocast_dtype("cuda"))
        entry = self._graph_cache.get(key)
        if entry is _NOT_GRAPHABLE:
            return None
        if entry is None:
            # A layout with no non-overlapping static twin is a property of the
            # signature, so that rejection is permanent and worth remembering.
            # It is decided here rather than inside `_capture` so that everything
            # `_capture` can fail on stays retryable.
            if not all(_is_dense_layout(t) for t in (q_x, kv_x, *biases)):
                self._graph_cache[key] = _NOT_GRAPHABLE
                return None
            entry = self._capture(q_x, kv_x, biases, q_shape, kv_shape, weights, plan)
            if entry is None:
                # Allocation, warmup or synchronization failed -- transient (an OOM
                # under memory pressure, say). Stay eager for this call and retry
                # on the next one; caching it would make one bad moment permanent.
                return None
            self._graph_cache[key] = entry

        graph, static_inputs, static_out, identity, dense, strided = entry
        # The graph embeds the packed weights' addresses. `_fused_weights` clears
        # this cache when it rebuilds them, so a surviving entry must still be
        # holding the identity it was captured against.
        if identity is not self._projection_cache[0]:
            self._graph_cache.clear()
            return None

        sources = [q_x, kv_x, *biases]
        if dense:
            torch._foreach_copy_([static_inputs[i] for i in dense],
                                 [sources[i] for i in dense])
        for i in strided:
            static_inputs[i].copy_(sources[i])
        graph.replay()
        # Never hand back the static buffer: the next replay overwrites it.
        return static_out.clone()

    def _capture(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: list[torch.Tensor],
        q_shape: torch.Size,
        kv_shape: torch.Size,
        weights: tuple,
        plan: tuple,
    ) -> tuple | None:
        """Capture the fused forward for this signature, or ``None`` if it cannot be.

        Returns ``None`` for a *retryable* failure -- an allocation, warmup or
        synchronization error such as an OOM under momentary memory pressure -- so
        the caller stays eager for this one call and tries again later. Permanent
        rejections are decided by the caller, which is what makes it safe for it to
        remember those and not these.
        """
        # `_replay` has already screened the transient conditions (recording
        # autograd, an outer capture holding the stream) and the permanent one (a
        # layout with no non-overlapping static twin), so every `None` returned
        # from here is a retryable failure.
        sources = [q_x, kv_x, *biases]

        try:
            static_inputs = [_static_like(t) for t in sources]
            for dst, src in zip(static_inputs, sources):
                dst.copy_(src)
            static_biases = static_inputs[2:]

            # Warm up on a side stream so Triton is compiled, cuBLAS handles and
            # workspaces exist, and the allocator is settled before capture. A
            # first-time initialization inside the capture would either fail or be
            # baked into the graph.
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._fused_eager(static_inputs[0], static_inputs[1],
                                      static_biases, q_shape, kv_shape, weights, plan)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
        except (RuntimeError, torch.cuda.CudaError):
            # Nothing has been captured at this point, so declining and running
            # eagerly is still sound.
            return None

        # Capture on the stream that was just warmed, because a cuBLAS workspace is
        # keyed by handle *and* stream: capturing on a different stream could
        # allocate a fresh one inside the capture.
        #
        # Exceptions from here are deliberately not caught. A capture that begins
        # and then fails can leave the stream context unrestored, so quietly
        # continuing eagerly would be unsound -- better to surface it.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            static_out = self._fused_eager(
                static_inputs[0], static_inputs[1], static_biases,
                q_shape, kv_shape, weights, plan)

        # Split the per-call copy-in once: contiguous inputs go through a single
        # `_foreach_copy_`, and each captured non-contiguous bias layout takes its
        # own `copy_`.
        dense = tuple(i for i, t in enumerate(sources) if t.is_contiguous())
        strided = tuple(i for i in range(len(sources)) if i not in dense)
        identity = self._projection_cache[0]
        return (graph, static_inputs, static_out, identity, dense, strided)

    def _single_pass_core(
        self,
        qg: torch.Tensor,
        kv: torch.Tensor,
        biases: list[torch.Tensor],
        n_batch: int,
        q_len: int,
        k_len: int,
        score_shape: tuple,
        launch: tuple,
        bias_plan: tuple,
    ) -> torch.Tensor:
        """The whole attention core in one kernel, returning ``og`` as ``[N·Q, H·C]``."""
        hidden = self._hidden
        og = torch.empty((n_batch * q_len, hidden), dtype=qg.dtype, device=qg.device)

        if len(bias_plan) < len(biases):
            folded = biases[0]
            for extra in biases[1:]:
                folded = folded + extra
            biases = [folded]

        # An unused bias slot still needs a valid pointer; the kernel's NUM_BIAS
        # removes the load, so qg stands in and is never read through it.
        prepared = [(qg, (0, 0, 0, 0)), (qg, (0, 0, 0, 0))]
        for slot, (bias, (mode, strides)) in enumerate(zip(biases, bias_plan)):
            if mode == _BIAS_MATERIALIZE:
                bias = bias.expand(score_shape).contiguous()
            prepared[slot] = (bias, strides)
        (bias0, s0), (bias1, s1) = prepared

        block_q, block_k, block_c, num_warps = launch
        _attention_gate_kernel[(n_batch, self.no_heads)](
            qg, kv, og, bias0, bias1,
            qg.stride(0) * q_len, qg.stride(0), qg.stride(1),
            kv.stride(0) * k_len, kv.stride(0), kv.stride(1),
            og.stride(0), og.stride(1),
            s0[0], s0[1], s0[2], s0[3],
            s1[0], s1[1], s1[2], s1[3],
            q_len, k_len, self.c_hidden, hidden,
            self._scale,
            BLOCK_Q=block_q, BLOCK_K=block_k, BLOCK_C=block_c,
            NUM_BIAS=len(biases), HAS_GATE=self.linear_g is not None,
            num_warps=num_warps,
        )
        return og

    def _sdpa_core(
        self,
        qg: torch.Tensor,
        kv: torch.Tensor,
        biases: list[torch.Tensor],
        n_batch: int,
        q_len: int,
        k_len: int,
        score_shape: tuple,
    ) -> torch.Tensor:
        """``F.scaled_dot_product_attention`` for a score tile too wide for registers.

        q, k and v are made contiguous: cuDNN's attention rejects the strided
        views that fall out of the packed projections and silently drops SDPA to
        its unfused math backend, which costs fifteen kernels instead of one.
        """
        heads = self.no_heads
        head_dim = self.c_hidden
        hidden = self._hidden

        q = qg[:, :hidden].view(n_batch, q_len, heads, head_dim).transpose(1, 2)
        k = kv[:, :hidden].view(n_batch, k_len, heads, head_dim).transpose(1, 2)
        v = kv[:, hidden:].view(n_batch, k_len, heads, head_dim).transpose(1, 2)

        mask = None
        if biases:
            summed = biases[0]
            for extra in biases[1:]:
                summed = summed + extra
            mask = summed.expand(score_shape).reshape(n_batch, heads, q_len, k_len)
            if mask.dtype is not qg.dtype:
                mask = mask.to(qg.dtype)

        o = F.scaled_dot_product_attention(
            q.contiguous(), k.contiguous(), v.contiguous(),
            attn_mask=mask, scale=self._scale)
        og = o.transpose(1, 2).reshape(n_batch * q_len, hidden)

        if self.linear_g is not None:
            og = og * torch.sigmoid(qg[:, hidden:])
        return og

    # -- contract -------------------------------------------------------------

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: list[torch.Tensor] | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        lma_q_chunk_size: int = 1024,
        lma_kv_chunk_size: int = 4096,
        use_high_precision: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            q_x:  [*, Q, C_q] query data
            kv_x: [*, K, C_k] key data
            biases: List of biases that broadcast to [*, H, Q, K]

        Returns:
            [*, Q, C_q] attention update
        """
        if biases is None:
            biases = []

        q_shape = q_x.shape
        kv_shape = kv_x.shape
        if self._can_fuse(q_x, kv_x, biases, q_shape, kv_shape):
            return self._fused_forward(q_x, kv_x, biases, q_shape, kv_shape)
        return self._reference_forward(q_x, kv_x, biases)
