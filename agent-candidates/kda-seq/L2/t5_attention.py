"""T5 encoder self-attention with the whole non-GEMM middle fused into one kernel.

The two projections are cuBLAS-shaped and left alone; everything between them is the
operator. Stage timing on a leased B200 at 1500 MHz (``profile/baseline_v0/stage_timing.py``)
puts 76 % of the forward there, and almost none of it is arithmetic:

    scores += position_bias   (strided bf16)   133.0 us   30.0 %
    scores.float()            (33.5 -> 67 MB)   74.6 us   16.8 %
    fp32 softmax              (67 MB in/out)    44.1 us    9.9 %
    attn_weights.type_as      (67 -> 33.5 MB)   26.6 us    6.0 %
    QK^T + PV                 (4.3 GFLOP)       41.0 us    9.2 %
    attn_output.transpose(1, 2).contiguous()    17.5 us    3.9 %

The 33.5 MB score tensor is materialized, promoted to a 67 MB fp32 copy, reduced, demoted, and
read again -- roughly 400 MB of DRAM traffic to do 4.3 GFLOP. One kernel that never lets a score
leave registers deletes all of it, and writing ``[B, S, H*D]`` from the epilogue deletes the
transpose-copy too.

It walks the key axis **twice**: pass 1 accumulates each query row's max and sum in fp32 with online
rescaling and rounds nothing, pass 2 recomputes the twice-rounded scores and rounds the *normalized*
probability for the ``PV`` MMA. Two passes rather than one because the reference rounds the
normalized weight, and tiled rather than full-width because that is faster, not slower: a 64x64 tile
compiles to 90 registers and 17 KB of shared memory with no spilling, against 255 registers, 82 KB
and 208 spill slots for a whole fp32 score row, and the resource relief buys more than the
recomputed ``QK^T`` and second bias read cost -- 50.2 us against 64.4 us over 150 measured
configurations (``docs/measurements.md`` section 9). A one-pass online form is faster still at
31.8 us but rounds the *unnormalized* exponentials, whose effective coefficients differ from the
reference's normalized weights on 24 % of elements, so it is recorded and not shipped.

Two constraints shape the kernel, and both are measured rather than assumed.

**The bf16 score roundings are observable.** T5 applies no ``1/sqrt(d)`` scaling, so with the
harness's weights the scores reach |s| ~ 75; bf16 spacing there is 0.5, and ``exp(0.25) =
1.28``. The softmax over 512 keys is consequently near one-hot, and the reference rounds the
scores to bf16 twice before it -- once as the output dtype of the ``QK^T`` bmm, once by the
in-place ``scores += position_bias``. Rounding therefore decides *which key wins*, and the
reference is the rounded version. Measured against it (``profile/baseline_v0/design_probes.py``,
gate = 99 % of elements within ``1e-2 + 1e-2*|ref|``): ``F.scaled_dot_product_attention`` with
the bias as ``attn_mask`` matches 0.419, a fully-fp32 path 0.424, and the same math with both
roundings reproduced matches 1.00000 at ``max_abs = 0.0``. No library attention kernel adds a
bias in bf16, so none can be used here; accuracy in the wrong place is a correctness failure.
The two ``.to(bfloat16)`` calls below are the whole cost of being faithful.

**The captured bias layout is the most expensive input.** ``position_bias`` arrives as
``[1, 64, 512, 512]`` with stride ``[64, 1, 32768, 64]`` -- ``compute_bias``'s
``permute(2, 0, 1)`` view of an ``[i, j, h]`` buffer, so the *head* index is stride-1 and
consecutive keys are 128 B apart. For a single head no orientation of an ``(i, j)`` tile
coalesces: a 2-byte load pulls a 32-byte sector shared by 16 heads, turning 33.5 MB of useful
bias into ~537 MB of L1<-L2 sector traffic (the whole tensor fits in the 126 MB L2, so DRAM is
not the limit). The bias strides are runtime arguments so one compiled kernel serves that layout
and a contiguous one, and ``_transpose_bias_to_head_major`` is the measured alternative that
pays 67 MB of coalesced pack traffic to make the re-read contiguous. ``docs/measurements.md``
records which one won and by how much.

Case B (``position_bias=None``, ``has_relative_attention_bias=True``) additionally pays
222-336 us of ``arange``/``abs``/``log``/``where``/gather/permute launches to produce a 33.5 MB
result whose value depends only on ``(bucket(j - i), head)``. ``_relative_position_bucket`` is a
pure elementwise function of ``j - i`` in ``[-(S-1), S-1]``, so a ``2S-1`` entry table built
*with this module's own bucketing function* makes the buckets identical by construction rather
than by reimplementation. ``_generate_relative_position_bias`` gathers the whole bias from that
table in its own kernel and the attention kernel then reads it as an ordinary supplied bias.
Generating it *inside* the attention kernel saves the re-read but costs more than it saves --
104.7 us against 85.0 us -- because the fused form has to carry an int32 bucket tile the shape of
its score tile, which forces the attention math onto a smaller query tile. The returned tensor is
contiguous where the baseline's is a permuted view; the bench compares shape
and values only, and contiguous is the layout every consumer wants, but it is observable through
``.stride()`` and is therefore an intentional relaxation rather than an oversight.

Two differences from the baseline are observable and intentional. The case-B bias is returned
contiguous rather than as a permuted view (above). And the fused path reads `q`/`k`/`v` and the
embedding by pointer instead of calling `self.bmm`, `self.softmax` and `self.relative_attention_bias`,
so a forward hook registered on one of those submodules, or an override of `compute_bias`, does not
run when the fused path is taken -- the reference body still honours all of them. The bench cannot
see either difference; a caller that installs hooks can.

Nothing here caches anything derived from an input tensor. The bucket table is keyed on shape
and config only. The harness passes the identical non-contiguous bias object on all 60 timing
iterations, so a cache keyed on ``data_ptr`` or tensor identity would make the bias cost vanish
from the measurement without making the operator faster, and would go stale under an in-place
mutation of the caller's bias.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from transformers import T5Config

from ....infra.tp import _tp_size, _tp_rank
from ..L1.embedding import Embedding
from .parallel_linear import QKVParallelLinear, RowParallelLinear

# The reference body has to be the *baseline's* computation, so it imports the baseline's own L1
# implementations by absolute-relative path rather than through ``..L1.*``. Those resolve through
# ``fastkernels.list``'s candidate finder, which rewrites only names under the
# ``fastkernels.tasks.candidate.`` prefix -- so ``..L1.softmax`` from here gives the frozen L1
# *winner* (a custom CUDA extension) while ``baseline.py``'s identical-looking import gives
# ``F.softmax``. Reading the import statement is not enough; the resolved module is what matters.
# ``BMM`` is ``torch.matmul`` in both packages, but it is taken from the same place for consistency.
from ...baseline.L1.linear import BMM
from ...baseline.L1.softmax import Softmax

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    # Triton kernels may only read globals declared as constexpr.
    _NEG_INF = tl.constexpr(float("-inf"))
    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - a Triton-less install must still be correct
    _TRITON_AVAILABLE = False


# Frozen tile geometry, from the offline sweep in ``profile/candidate_v1/sweep_algorithms.py``
# (150 configurations over algorithm x query tile x key tile x warps x stages, full table with
# losers in ``docs/measurements.md``). Literals rather than ``triton.autotune`` on purpose: a
# runtime sweep would make the reported latency depend on cache warmth and would eat into the
# harness's 1200 s per-operator budget.
#
# The winner is the two-pass form at a 64x64 tile with 4 warps: 50.2 us against 64.4 us for the
# best full-width configuration, at 96 registers and 16 KB of shared memory instead of 255
# registers and 82 KB. A one-pass online form reached 31.8 us but rounds the *unnormalized*
# exponentials for the PV MMA, which is a different probability tensor than the reference's; see
# ``docs/measurements.md`` for its numbers and why it is not shipped.
_QUERY_TILE = 64
_KEY_TILE = 64
_NUM_WARPS = 4
_NUM_STAGES = 1

# Pack tile for the head-major transpose: 64 keys x 64 heads is 8 KB of bf16, so the read is one
# contiguous 4096-element run of the ``[i, j, h]`` buffer. The sweep is flat from 64 to 256 rows at
# 17.4 us, so the smallest tile that reaches the plateau ships.
_PACK_ROW_TILE = 64
_PACK_NUM_WARPS = 4

# Standalone relative-position bias generator (case B). Generating the bias in its own kernel and
# letting the attention kernel read it as an ordinary supplied bias beats generating it inside the
# attention kernel -- 85.0 us against 104.7 us -- because the fused form has to carry an int32
# bucket tile the shape of its score tile, which forces a smaller query tile on the attention math.
_BIAS_GEN_ROW_TILE = 16
_BIAS_GEN_NUM_WARPS = 8

# Whether a strided bias is packed head-major before the attention kernel reads it, or read at
# its captured stride directly. The plan expected these to be close and defaulted to the direct
# read; measured end to end in one process they are not close -- 138.3 us packing against
# 248.9 us reading the captured stride, against a 262.0 us baseline. See ``docs/measurements.md``.
_PACK_STRIDED_BIAS = True

# The captured domain, and nothing wider. The reference body below is the baseline's, so refusing a
# shape costs only speed on inputs nobody benches -- whereas an unmeasured fast path is a
# correctness risk with no reward. Widening this is a measurement away, not a rewrite: the kernels
# are shape-generic apart from needing a power-of-two key length.
_FUSED_SEQ_LEN = 512
_FUSED_HEAD_DIM = 64
_FUSED_HEADS = 64

# Set if a kernel ever fails to compile or launch. ``_TRITON_AVAILABLE`` only says the import
# succeeded; a compile failure or an unsupported architecture surfaces at the first launch, and that
# must degrade to the reference body rather than take the operator down. Latched, so the failure is
# paid once instead of on every call.
_fused_launch_failed = False

# Auxiliary stream for overlapping the bias work with ``qkv_proj``. The two are independent -- the
# bias depends only on ``position_bias`` or on the embedding weight, ``qkv_proj`` only on
# ``hidden_states`` -- and they are shaped to overlap well: the pack is bandwidth-bound at 24.6 % of
# peak DRAM read while ``qkv_proj`` runs at 1025 TFLOP/s. Measured worth 5.0 us on each case, ~4 %
# (``docs/measurements.md`` section 15). Created lazily and cached per device; ordering is by stream
# waits only, with no device synchronize and nothing cached across calls.
_aux_streams: dict[torch.device, "torch.cuda.Stream"] = {}


def _auxiliary_stream(device: torch.device) -> "torch.cuda.Stream":
    stream = _aux_streams.get(device)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _aux_streams[device] = stream
    return stream

# Instrumentation, so an over-strict allowlist cannot quietly earn baseline-speed timing while
# reporting PASSED. Read with ``execution_counts()``; a Python increment beside a ~100 us forward
# is not measurable.
_fused_calls = 0
_reference_calls = 0


def execution_counts() -> dict[str, int]:
    """``{"fused": n, "reference": m}`` -- how many forwards took each path."""
    return {"fused": _fused_calls, "reference": _reference_calls}


def reset_execution_counts() -> None:
    global _fused_calls, _reference_calls
    _fused_calls = 0
    _reference_calls = 0


if _TRITON_AVAILABLE:

    @triton.jit
    def _twice_rounded_scores(
        qkv_ptr, bias_ptr, q, rows, keys, batch, head,
        stride_qkv_batch, stride_qkv_row,
        stride_bias_head, stride_bias_query, stride_bias_key,
        N_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
    ):
        """One score tile, carrying the baseline's two bf16 roundings.

        Rounding #1: the reference's ``QK^T`` is a bf16 bmm, so its fp32 accumulator is rounded to
        bf16 on the way out. Rounding #2: ``scores += position_bias`` is an in-place bf16 add, i.e.
        ``bf16(f32(s) + f32(bias))``. Both are load-bearing -- see the module docstring. ``k`` is
        loaded key-major (head dim innermost, stride 1) and transposed for the MMA rather than
        loaded transposed, because a transposed load would stride by ``3 * d_model`` on its
        fastest axis.
        """
        dims = tl.arange(0, HEAD_DIM)
        k = tl.load(
            qkv_ptr + batch * stride_qkv_batch + head * HEAD_DIM + N_HEADS * HEAD_DIM
            + keys[:, None] * stride_qkv_row + dims[None, :]
        )
        score_bias = tl.load(
            bias_ptr + head * stride_bias_head
            + rows[:, None] * stride_bias_query + keys[None, :] * stride_bias_key
        )
        scores = tl.dot(q, tl.trans(k)).to(tl.bfloat16)
        return (scores.to(tl.float32) + score_bias.to(tl.float32)).to(tl.bfloat16).to(tl.float32)

    @triton.jit
    def _fused_attention_given_bias(
        qkv_ptr,
        out_ptr,
        bias_ptr,
        stride_qkv_batch,
        stride_qkv_row,
        stride_out_batch,
        stride_out_row,
        stride_bias_head,
        stride_bias_query,
        stride_bias_key,
        N_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        KEY_LEN: tl.constexpr,
        QUERY_TILE: tl.constexpr,
        KEY_TILE: tl.constexpr,
    ):
        """Fused attention over a caller-supplied bias at arbitrary strides, in two passes.

        Pass 1 walks the key axis accumulating the row max and sum in fp32 with online rescaling
        and rounds nothing. Pass 2 walks it again, recomputes the twice-rounded scores, and rounds
        the **normalized** probability -- which is what ``softmax(scores.float()).type_as(scores)``
        produces, so the tensor entering the ``PV`` MMA is the reference's. A one-pass online form
        would instead round the unnormalized exponentials and feed a different tensor to that MMA.
        The price is a recomputed ``QK^T`` and a second bias read; measured, that price is negative
        -- 50.2 us against 64.4 us for holding a whole fp32 score row, because a 64x64 tile needs
        96 registers and 16 KB of shared memory where the full row needs 255 and 82 KB.

        The three bias strides are runtime arguments precisely so the captured permuted layout and
        a contiguous one share this kernel; baking in the captured stride would make the
        contiguous case wrong.
        """
        rows = tl.program_id(0) * QUERY_TILE + tl.arange(0, QUERY_TILE)
        head = tl.program_id(1)
        batch = tl.program_id(2)
        dims = tl.arange(0, HEAD_DIM)
        q = tl.load(
            qkv_ptr + batch * stride_qkv_batch + head * HEAD_DIM
            + rows[:, None] * stride_qkv_row + dims[None, :]
        )

        row_max = tl.full((QUERY_TILE,), _NEG_INF, tl.float32)
        row_sum = tl.zeros((QUERY_TILE,), tl.float32)
        for start in tl.range(0, KEY_LEN, KEY_TILE):
            keys = start + tl.arange(0, KEY_TILE)
            scores = _twice_rounded_scores(
                qkv_ptr, bias_ptr, q, rows, keys, batch, head,
                stride_qkv_batch, stride_qkv_row,
                stride_bias_head, stride_bias_query, stride_bias_key, N_HEADS, HEAD_DIM,
            )
            new_max = tl.maximum(row_max, tl.max(scores, 1))
            # A key block that is entirely -inf leaves ``new_max`` at -inf, and ``-inf - -inf`` is
            # NaN -- which would poison the running sum for a row whose *later* blocks are finite,
            # where a whole-row softmax recovers. Substituting 0 for a still-empty max keeps
            # ``exp(-inf - 0) = 0``, so such a block contributes nothing and the rescale is exact
            # once real values arrive. Unreachable on the captured bias, which is finite, but the
            # allowlist admits any finite-or-not bf16 bias.
            safe_max = tl.where(new_max == _NEG_INF, 0.0, new_max)
            row_sum = row_sum * libdevice.exp(row_max - safe_max) + tl.sum(
                libdevice.exp(scores - safe_max[:, None]), 1)
            row_max = new_max

        # ``libdevice.exp``, not ``tl.exp``. Neither ``tl.exp`` nor ``tl.exp2`` is the accurate
        # exponential on this backend. Measured against the reference's own
        # ``softmax(scores.float())``, the normalized bf16 probabilities differ on 1.6e-04 of
        # elements with ``tl.exp`` and 9.9e-04 with ``tl.exp2``, against 1.3e-05 with libdevice's
        # ``__nv_expf`` -- the same function ATen's softmax calls, and the floor two Torch
        # implementations of this softmax reach against each other (``tools/probe_ac23.py``).
        # Anything less accurate misses the acceptance criterion on the probability tensor.
        #
        # One reciprocal per query row rather than one division per element; measured identical in
        # latency and mismatch, so the cheaper form ships.
        # Same substitution as pass 1. A row that is -inf everywhere still yields NaN here, via
        # ``row_sum = 0`` -- which is what a reference softmax over an all--inf row produces too.
        row_max = tl.where(row_max == _NEG_INF, 0.0, row_max)
        scale = 1.0 / row_sum
        acc = tl.zeros((QUERY_TILE, HEAD_DIM), tl.float32)
        for start in tl.range(0, KEY_LEN, KEY_TILE):
            keys = start + tl.arange(0, KEY_TILE)
            scores = _twice_rounded_scores(
                qkv_ptr, bias_ptr, q, rows, keys, batch, head,
                stride_qkv_batch, stride_qkv_row,
                stride_bias_head, stride_bias_query, stride_bias_key, N_HEADS, HEAD_DIM,
            )
            probs = (libdevice.exp(scores - row_max[:, None]) * scale[:, None]).to(tl.bfloat16)
            v = tl.load(
                qkv_ptr + batch * stride_qkv_batch + head * HEAD_DIM + 2 * N_HEADS * HEAD_DIM
                + keys[:, None] * stride_qkv_row + dims[None, :]
            )
            acc += tl.dot(probs, v)

        # Writes ``[B, S, H*D]`` directly, which is what the baseline's
        # ``transpose(1, 2).contiguous()`` was for.
        tl.store(
            out_ptr + batch * stride_out_batch + rows[:, None] * stride_out_row
            + head * HEAD_DIM + dims[None, :],
            acc.to(tl.bfloat16),
        )

    @triton.jit
    def _generate_relative_position_bias(
        bias_ptr,
        bucket_ptr,
        emb_ptr,
        stride_emb_bucket,
        N_HEADS: tl.constexpr,
        KEY_LEN: tl.constexpr,
        ROW_TILE: tl.constexpr,
    ):
        """Contiguous ``[1, H, S, S]`` relative-position bias from the cached bucket table.

        ``bucket_ptr`` holds ``_relative_position_bucket(j - i)`` for every ``j - i``, built by this
        module's own bucketing function, so the gather is exact rather than approximately right --
        ``tools/verify_numerics.py`` checks the result ``torch.equal`` against ``compute_bias``.
        Both lookup tables are a few KB and stay resident in L1.

        Emitted contiguous: what the value comparison wants, what the attention kernel likes, and
        the layout layers 1..23 of a fused encoder would want handed to them.
        """
        rows = tl.program_id(0) * ROW_TILE + tl.arange(0, ROW_TILE)
        head = tl.program_id(1)
        keys = tl.arange(0, KEY_LEN)
        # ``KEY_LEN - 1`` is the table's zero offset, which is ``query_length - 1``; the two agree
        # because this is self-attention and the allowlist admits only ``S x S``.
        bucket = tl.load(bucket_ptr + (keys[None, :] - rows[:, None] + (KEY_LEN - 1)))
        tl.store(
            bias_ptr + head * KEY_LEN * KEY_LEN + rows[:, None] * KEY_LEN + keys[None, :],
            tl.load(emb_ptr + bucket * stride_emb_bucket + head),
        )

    @triton.jit
    def _transpose_bias_to_head_major(
        src_ptr,
        dst_ptr,
        stride_src_key,
        stride_src_head,
        n_rows,
        N_HEADS: tl.constexpr,
        ROW_TILE: tl.constexpr,
    ):
        """``[i, j, h] -> [h, i, j]``, reading and writing coalesced.

        The captured bias is a contiguous ``[S*S, H]`` matrix in disguise, so the pack is a plain
        2-D transpose: the read runs along the stride-1 head axis and each head's write lands in
        one contiguous run of ``ROW_TILE`` keys. Whether paying 67 MB here beats the ~537 MB of
        sector traffic the attention kernel pulls when it reads the captured stride directly is a
        measurement -- see ``docs/measurements.md``.
        """
        rows = tl.program_id(0) * ROW_TILE + tl.arange(0, ROW_TILE)
        heads = tl.arange(0, N_HEADS)
        in_range = rows < n_rows
        tile = tl.load(
            src_ptr + rows[:, None] * stride_src_key + heads[None, :] * stride_src_head,
            mask=in_range[:, None],
            other=0.0,
        )
        tl.store(
            dst_ptr + heads[None, :] * n_rows + rows[:, None],
            tile,
            mask=in_range[:, None],
        )


def _reads_as_stored(tensor: torch.Tensor) -> bool:
    """Whether a raw load of ``tensor``'s storage sees the values PyTorch would report.

    ``torch._neg_view`` and ``torch._conj`` return tensors that are contiguous and share the
    original ``data_ptr`` while carrying the negation or conjugation as a *flag* that only the
    PyTorch dispatcher honours. A Triton kernel loading the storage directly would read the
    unnegated values and be silently wrong, so such a tensor goes to the reference body.
    """
    return not tensor.is_neg() and not tensor.is_conj()


class T5SelfAttention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.d_kv
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance

        tp_size = _tp_size()
        assert self.n_heads % tp_size == 0
        self.n_heads_per_partition = self.n_heads // tp_size

        # Submodule set and parameter names are load-bearing, not stylistic: the harness shares
        # weights with ``load_state_dict(baseline.state_dict(), strict=False)`` inside a bare
        # ``try/except: pass``, so one renamed or reshaped parameter is swallowed silently and
        # the candidate runs on its own random weights with correctness collapsing for a reason
        # that looks numerical. ``qkv_proj.weight``, ``o.weight`` and
        # ``relative_attention_bias.emb.weight`` must stay exactly those names.
        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.d_model,
            head_size=self.d_kv,
            total_num_heads=self.n_heads,
            total_num_kv_heads=self.n_heads,
            bias=False,
        )

        self.o = RowParallelLinear(self.inner_dim, self.d_model, bias=False)

        # Constructed eagerly and from the baseline's own L1 package, so the reference body is the
        # baseline's computation rather than something numerically equivalent to it. Eagerly because
        # the frozen L1 ``Embedding`` below builds a CUDA extension at import, and a lazily
        # constructed submodule would risk putting a build inside the first call that needs it --
        # which in the harness can be a timed one.
        self.bmm = BMM()
        self.softmax = Softmax(dim=-1)

        if has_relative_attention_bias:
            self.relative_attention_bias = Embedding(
                self.relative_attention_num_buckets, self.n_heads,
            )

        # Shape- and config-derived only: ``{(q_len, k_len, num_buckets, max_distance, device):
        # table}``. The bucket count and max distance belong in the key because a
        # differently-configured module with the same lengths has a different table. Nothing
        # derived from an input tensor's identity, address or version is ever cached.
        self._bucket_tables: dict[tuple, torch.Tensor] = {}

    @staticmethod
    def _relative_position_bucket(
        relative_position: torch.Tensor,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> torch.Tensor:
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position),
            )
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )
        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large,
        )
        return relative_buckets

    def compute_bias(self, query_length: int, key_length: int, device: torch.device) -> torch.Tensor:
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position, bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)
        tp_rank = _tp_rank()
        head_start = tp_rank * self.n_heads_per_partition
        head_end = head_start + self.n_heads_per_partition
        values = values[:, :, head_start:head_end]
        values = values.permute(2, 0, 1).unsqueeze(0)
        return values

    def _bucket_table(
        self, query_length: int, key_length: int, device: torch.device,
    ) -> torch.Tensor:
        """``table[d + query_length - 1] = bucket(d)`` for every ``d = j - i``.

        ``compute_bias`` evaluates ``_relative_position_bucket`` on a ``[q, k]`` grid of
        ``j - i``, but the function is pure and elementwise, so evaluating it on the
        ``query_length + key_length - 1`` distinct differences yields the same buckets -- and
        calling this module's own bucketing function means the ``torch.log`` /
        ``.to(torch.long)`` truncation sequence is identical by construction.
        ``tools/verify_numerics.py`` checks the resulting bias with ``torch.equal`` against
        ``compute_bias`` rather than trusting that argument.
        """
        key = (query_length, key_length, self.relative_attention_num_buckets,
               self.relative_attention_max_distance, device)
        table = self._bucket_tables.get(key)
        if table is None:
            offsets = torch.arange(
                -(query_length - 1), key_length, dtype=torch.long, device=device,
            )
            table = self._relative_position_bucket(
                offsets, bidirectional=True,
                num_buckets=self.relative_attention_num_buckets,
                max_distance=self.relative_attention_max_distance,
            ).to(torch.int32)
            self._bucket_tables[key] = table
        return table

    def _fused_path_supported(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None,
        position_bias: torch.Tensor | None,
        batch_size: int,
        seq_length: int,
    ) -> bool:
        """Eligibility from the inputs alone, so the bias work can be launched before ``qkv_proj``.

        Everything here is decidable without the projection's output. ``_qkv_layout_supported``
        re-checks what the kernel assumes about ``qkv`` itself once it exists.
        """
        if not _TRITON_AVAILABLE or _fused_launch_failed:
            return False
        if torch.is_grad_enabled():
            return False
        if hidden_states.dtype is not torch.bfloat16 or not hidden_states.is_cuda:
            return False
        # ``batch_size`` and ``seq_length`` come from ``hidden_states.shape[:2]``, so a 4-D input
        # would leave a trailing dimension the kernel's index arithmetic knows nothing about --
        # it would silently attend over the wrong elements where the baseline's ``view`` raises.
        if hidden_states.dim() != 3 or hidden_states.shape[2] != self.d_model:
            return False
        if batch_size != 1 or seq_length != _FUSED_SEQ_LEN:
            return False
        if self.d_kv != _FUSED_HEAD_DIM or self.n_heads_per_partition != _FUSED_HEADS:
            return False
        if _tp_size() != 1:
            return False
        # The baseline folds ``mask`` into the bias only on the ``position_bias is None`` branch
        # and silently ignores it otherwise. Rather than reproduce half of that quirk in a
        # kernel, any non-None mask goes to the reference body, which *is* that quirk.
        if mask is not None:
            return False
        if position_bias is None:
            if not self.has_relative_attention_bias:
                return False
            weight = self.relative_attention_bias.emb.weight
            return (weight.dtype is torch.bfloat16 and weight.is_cuda
                    and weight.device == hidden_states.device and weight.stride(1) == 1
                    and _reads_as_stored(weight)
                    and tuple(weight.shape)
                    == (self.relative_attention_num_buckets, self.n_heads))
        return (position_bias.dtype is torch.bfloat16
                and position_bias.is_cuda
                and position_bias.device == hidden_states.device
                and _reads_as_stored(position_bias)
                and tuple(position_bias.shape)
                == (1, self.n_heads_per_partition, seq_length, seq_length))

    def _qkv_layout_supported(
        self, qkv: torch.Tensor, batch_size: int, seq_length: int,
    ) -> bool:
        """What the kernel assumes about the projection's output, checked once it exists."""
        return (qkv.dtype is torch.bfloat16 and qkv.is_cuda and qkv.is_contiguous()
                and qkv.shape
                == (batch_size, seq_length, 3 * self.n_heads_per_partition * self.d_kv))

    def _prepare_bias(
        self, position_bias: torch.Tensor | None, seq_length: int,
        device: torch.device, dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(bias the kernel reads, bias to return)``.

        Case B generates the bias in its own kernel; the attention kernel then reads it as an
        ordinary supplied bias. Generating it *inside* the attention kernel saves the re-read but
        costs more than it saves -- 104.7 us against 85.0 us -- because the fused form has to carry
        an int32 bucket tile the shape of its score tile, which forces a smaller query tile on the
        attention math.

        Case A returns the caller's object, never the pack's scratch buffer: the bench compares the
        returned bias leaf too, and in case A it *is* the input tensor.
        """
        heads = self.n_heads_per_partition
        if position_bias is None:
            # The fast path requires tp size 1, so the per-partition head slice ``compute_bias``
            # takes is the identity and the embedding rows are used whole.
            emb = self.relative_attention_bias.emb.weight
            generated = torch.empty(
                (1, heads, seq_length, seq_length), device=device, dtype=dtype)
            _generate_relative_position_bias[(seq_length // _BIAS_GEN_ROW_TILE, heads)](
                generated, self._bucket_table(seq_length, seq_length, device), emb, emb.stride(0),
                N_HEADS=heads, KEY_LEN=seq_length, ROW_TILE=_BIAS_GEN_ROW_TILE,
                num_warps=_BIAS_GEN_NUM_WARPS, num_stages=_NUM_STAGES,
            )
            return generated, generated
        if _PACK_STRIDED_BIAS and position_bias.stride(3) != 1:
            return self._pack_bias(position_bias, seq_length), position_bias
        return position_bias, position_bias

    def _fused_attention(
        self, qkv: torch.Tensor, bias: torch.Tensor, batch_size: int, seq_length: int,
    ) -> torch.Tensor:
        heads = self.n_heads_per_partition
        attn_output = torch.empty(
            (batch_size, seq_length, heads * self.d_kv),
            device=qkv.device, dtype=qkv.dtype,
        )
        _fused_attention_given_bias[(seq_length // _QUERY_TILE, heads, batch_size)](
            qkv, attn_output, bias,
            qkv.stride(0), qkv.stride(1),
            attn_output.stride(0), attn_output.stride(1),
            bias.stride(1), bias.stride(2), bias.stride(3),
            N_HEADS=heads, HEAD_DIM=self.d_kv,
            KEY_LEN=seq_length, QUERY_TILE=_QUERY_TILE, KEY_TILE=_KEY_TILE,
            num_warps=_NUM_WARPS, num_stages=_NUM_STAGES,
        )
        return attn_output

    def _pack_bias(self, bias: torch.Tensor, seq_length: int) -> torch.Tensor:
        """Head-major copy of ``bias``, or ``bias`` itself when the pack does not apply.

        The pack only makes sense while the ``(query, key)`` plane is a contiguous ``[S*S, H]``
        matrix, i.e. while ``stride_query == S * stride_key``; that is the captured layout.
        Anything else is read at its own strides.
        """
        if bias.stride(2) != seq_length * bias.stride(3):
            return bias
        heads = bias.shape[1]
        n_rows = seq_length * seq_length
        packed = torch.empty(
            (1, heads, seq_length, seq_length), device=bias.device, dtype=bias.dtype,
        )
        _transpose_bias_to_head_major[(triton.cdiv(n_rows, _PACK_ROW_TILE),)](
            bias, packed, bias.stride(3), bias.stride(1), n_rows,
            N_HEADS=heads, ROW_TILE=_PACK_ROW_TILE,
            num_warps=_PACK_NUM_WARPS, num_stages=_NUM_STAGES,
        )
        return packed

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global _fused_calls, _reference_calls
        batch_size, seq_length = hidden_states.shape[:2]

        if self._fused_path_supported(
            hidden_states, mask, position_bias, batch_size, seq_length,
        ):
            try:
                # The bias work and ``qkv_proj`` are independent, so the bias goes on the auxiliary
                # stream while the projection runs on the caller's. Ordering is by stream waits
                # only: the auxiliary stream waits on the caller's so it cannot start ahead of the
                # caller's prior work, and the caller waits on it before the attention kernel reads
                # the bias. ``record_stream`` keeps the caching allocator from reissuing memory the
                # auxiliary stream is still using.
                caller = torch.cuda.current_stream(hidden_states.device)
                auxiliary = _auxiliary_stream(hidden_states.device)
                auxiliary.wait_stream(caller)
                with torch.cuda.stream(auxiliary):
                    read_bias, returned_bias = self._prepare_bias(
                        position_bias, seq_length, hidden_states.device, hidden_states.dtype,
                    )
                    read_bias.record_stream(auxiliary)
                    if position_bias is not None:
                        position_bias.record_stream(auxiliary)
                    else:
                        self.relative_attention_bias.emb.weight.record_stream(auxiliary)
                qkv = self.qkv_proj(hidden_states)
                caller.wait_stream(auxiliary)
                if self._qkv_layout_supported(qkv, batch_size, seq_length):
                    attn_output = self._fused_attention(
                        qkv, read_bias, batch_size, seq_length,
                    )
                    _fused_calls += 1
                    return self.o(attn_output), returned_bias
            except Exception:  # noqa: BLE001 - a compile or launch failure must not be fatal
                global _fused_launch_failed
                _fused_launch_failed = True
                qkv = self.qkv_proj(hidden_states)
        else:
            qkv = self.qkv_proj(hidden_states)

        _reference_calls += 1
        q_size = self.n_heads_per_partition * self.d_kv
        kv_size = self.n_heads_per_partition * self.d_kv
        query_states, key_states, value_states = qkv.split(
            [q_size, kv_size, kv_size], dim=-1,
        )

        query_states = query_states.view(
            batch_size, seq_length, self.n_heads_per_partition, self.d_kv,
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_length, self.n_heads_per_partition, self.d_kv,
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_length, self.n_heads_per_partition, self.d_kv,
        ).transpose(1, 2)

        scores = self.bmm(query_states, key_states.transpose(3, 2))

        if position_bias is None:
            if self.has_relative_attention_bias:
                position_bias = self.compute_bias(
                    seq_length, seq_length, device=scores.device,
                )
            else:
                position_bias = torch.zeros(
                    (1, self.n_heads_per_partition, seq_length, seq_length),
                    device=scores.device, dtype=scores.dtype,
                )
            if mask is not None:
                position_bias = position_bias + mask

        scores += position_bias
        attn_weights = self.softmax(scores.float()).type_as(scores)
        attn_output = self.bmm(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, -1)
        attn_output = self.o(attn_output)

        return attn_output, position_bias
