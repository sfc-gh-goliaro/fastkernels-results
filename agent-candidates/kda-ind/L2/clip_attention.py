"""CLIP self-attention (L2).

Standard multi-head self-attention with separate Q/K/V projections and manual
SDPA, computing the same thing as ``baseline.py`` but with far fewer device ops.

Two properties of this operator drive the implementation.

*The reference is TF32, not fp32.* ``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`` is set in
this container, so every ``F.linear`` and ``matmul`` in the reference rounds both
operands to a 10-bit mantissa before multiplying, and the reference output carries
that error. A *more accurate* implementation therefore disagrees with the reference
by more than the comparison tolerance: exact fp32 and fp64 both land at ~0.837 of
elements in tolerance, and ``scaled_dot_product_attention`` at ~0.894. So the
projections stay on cuBLAS TF32, and anything hand-written has to reproduce
cuBLAS's fp32 -> tf32 rounding rather than improve on it.

*It is latency-bound, not FLOP-bound.* The whole operator is ~381 MFLOP over
9.44 MB of weights; at the captured shape the reference spends ~39 us of kernel
time inside a ~99 us measurement, so most of the wall time is inter-kernel gap.
Fewer device ops is worth more than faster ones, which is why Q/K/V share one
packed projection and the attention body is a single kernel that reads its inputs
strided out of that packed buffer instead of materializing reshaped copies.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from transformers import CLIPTextConfig

# Tile shape of the attention kernel, fixed as constexpr values: triton.autotune at
# runtime would compile inside the timed window and spawn threads, both of which the
# benchmark rejects. Chosen offline over BLOCK_M in {16, 32, 64, 128} x num_warps in
# {1, 2, 4, 8} by profile/probe_tile_selection.py, which re-checks on every run that this
# pair is still the winner. The good tiles differ by only a couple of microseconds of wall
# time, which is less than this box's between-session drift, so the selection is made on
# native kernel duration instead; absolute microseconds move run to run but the ordering
# does not. This pair is the fastest on that signal, and gives cdiv(S, 16) * H = 60 CTAs
# rather than 36, filling more of a 148-SM machine. Starved configs (num_warps=1, or
# BLOCK_M >= 64 with too few warps) are worse by an order of magnitude, reproducibly.
_BLOCK_M = 16
_NUM_WARPS = 8
# Capability bounds. These are hard: relax one and the kernel fails to compile or
# computes the wrong thing, on any device. `_MIN_BLOCK` is tl.dot's minimum tile
# extent, and it does double duty -- forcing head_dim to a power of two of at least 16
# also keeps head_dim 4-aligned, which matters because the score product reduces over
# K = head_dim and is subject to exactly the same cuBLAS TF32 dispatch rule described
# for `_MIN_SEQ_FAST` below. Anyone relaxing the power-of-two requirement to admit a
# head_dim of 48 or 80 by padding walks straight into that hazard.
_MIN_BLOCK = 16
# Largest head_dim the kernel will tile in one BLOCK_D, keeping registers bounded.
_MAX_HEAD_DIM = 128
# Largest padded sequence tile. next_pow2(S) has to fit one BLOCK_N for the whole score
# row to be resident, which is what makes the single-pass softmax legal.
_MAX_BLOCK_N = 256

# Numerical-agreement bound, a different kind of constant from the three above. Relaxing
# this yields a *correct* kernel that is more accurate than the graded reference, on a
# boundary belonging to one cuBLAS version on one architecture -- so re-measure it after
# a toolkit bump rather than assuming it holds. Below it, cuBLAS does not reliably put
# the reference's PV product on TF32 tensor cores: that product reduces over K = S, and
# for S < 20 it only takes the TF32 path when S is a multiple of four, falling back to
# exact fp32 otherwise. Reproducing TF32 faithfully then makes us the *less* accurate
# side and we disagree by more than the tolerance -- measured at 0.80-0.86 of elements
# in tolerance for every S < 20 that is not 4-aligned, against 0.9995 or better from
# S = 20 up. Short sequences take the fallback, which is bit-exact because it issues the
# very same cuBLAS calls. It fails safe: a wrong threshold costs latency, not accuracy.
_MIN_SEQ_FAST = 20


@triton.jit
def _round_to_tf32(x):
    """Round an fp32 tile to the tf32 grid, in registers, keeping fp32 storage.

    This looks like a bug until you know that the reference is TF32: cuBLAS rounds
    both operands from fp32 to tf32 (10 explicit mantissa bits) before multiplying,
    and the reference output carries that error, so matching it means reproducing
    the rounding rather than avoiding it. Triton's own tl.dot(input_precision="tf32")
    truncates the low 13 bits instead of rounding to nearest even, which is measured
    at 0.66 of elements in tolerance against a 0.99 threshold -- so we round here
    and tl.dot's truncation then has nothing left to remove.

    Adding 0xFFF + lsb before masking carries into bit 13 exactly when the discarded
    remainder is above half, or is exactly half with an odd bit 13. fp32 is
    sign-magnitude, so incrementing the pattern increments the magnitude and
    round-half-to-even on the magnitude is round-half-to-even on the value; the
    arithmetic shift sign-extends but "& 1" still reads bit 13.

    This is exact for every finite binary32 value: negatives, signed zeros,
    subnormals, and overflow of the largest finite float to Inf, which is what RNE
    should produce there rather than saturating. It is *not* correct for NaN with a
    mantissa payload -- 0x7F800001 comes back as +Inf, and 0x7FFFFFFF wraps to -0 --
    but non-finite inputs are out of contract here and the benchmark rejects
    non-finite output anyway. One portability caveat: PTX leaves tensor-core
    subnormal handling unspecified, so the subnormal agreement is measured on this
    device rather than guaranteed. The mask is spelled -8192 and not 0xFFFFE000
    because the hex literal infers as uint32 and would not combine with an int32 tile.
    """
    i = x.to(tl.int32, bitcast=True)
    i = (i + 0xFFF + ((i >> 13) & 1)) & -8192
    return i.to(tl.float32, bitcast=True)


@triton.jit
def _clip_attention_kernel(
    QKV, MASK, OUT,
    stride_qkv_row, stride_out_row,
    stride_mask_b, stride_mask_h, stride_mask_m, stride_mask_n,
    seq_length, embed_dim, num_heads, scale,
    HAS_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Scores, scale, mask, softmax and the PV product for one tile of one head.

    q, k and v are read strided straight out of the packed [B*S, 3E] projection
    output, which is what removes the reference's three view/transpose steps, and
    the result is stored at OUT[b*S + m, h*D + d], which removes its
    transpose(1, 2).contiguous() copy. Because next_pow2(S) fits in one BLOCK_N
    tile the whole score row is resident, so there is no online-softmax rescaling
    and the reference's op order can be followed literally.

    BLOCK_D is the head dimension itself rather than a padded tile of it, so the d
    lanes need no predicate: the caller passes BLOCK_D=head_dim, and head_dim is a
    power of two that CLIPTextConfig guarantees divides embed_dim. embed_dim is still
    needed separately as the q -> k -> v column pitch in the packed buffer.
    """
    tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // num_heads
    head = batch_head % num_heads

    offs_m = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    m_valid = offs_m < seq_length
    n_valid = offs_n < seq_length
    row = batch * seq_length
    col = head * BLOCK_D + offs_d

    # Every load is predicated with other=0.0, so lanes past S read exact zeros
    # instead of whatever the neighbouring allocation happens to hold.
    queries = tl.load(QKV + (row + offs_m)[:, None] * stride_qkv_row + col[None, :],
                      mask=m_valid[:, None], other=0.0)
    keys_t = tl.load(QKV + (row + offs_n)[None, :] * stride_qkv_row
                     + (embed_dim + col)[:, None],
                     mask=n_valid[None, :], other=0.0)

    scores = tl.dot(_round_to_tf32(queries), _round_to_tf32(keys_t),
                    input_precision="tf32")
    scores = scores * scale
    if HAS_MASK:
        scores = scores + tl.load(
            MASK + batch * stride_mask_b + head * stride_mask_h
            + offs_m[:, None] * stride_mask_m + offs_n[None, :] * stride_mask_n,
            mask=m_valid[:, None] & n_valid[None, :], other=0.0)
    # Zero-filled key columns past S would otherwise score 0, and exp(0) = 1 would
    # pollute the denominator; -inf makes them contribute exactly nothing.
    scores = tl.where(n_valid[None, :], scores, float("-inf"))

    # Same reduction order as the reference's softmax_warp_forward: row max, then
    # exp(x - max), then divide by the sum.
    probs = tl.exp(scores - tl.max(scores, 1)[:, None])
    probs = probs / tl.sum(probs, 1)[:, None]

    values = tl.load(QKV + (row + offs_n)[:, None] * stride_qkv_row
                     + (2 * embed_dim + col)[None, :],
                     mask=n_valid[:, None], other=0.0)
    out = tl.dot(_round_to_tf32(probs), _round_to_tf32(values),
                 input_precision="tf32")
    tl.store(OUT + (row + offs_m)[:, None] * stride_out_row + col[None, :],
             out, mask=m_valid[:, None])


def _next_pow2(n: int) -> int:
    """Smallest power of two at least ``n``.

    ``triton.next_power_of_2`` computes the same thing, but it is a ConstexprFunction
    wrapper that measures ~0.65 us per call against ~0.03 us here, and this runs on
    every forward. Same reason ``triton.cdiv`` is spelled as a ceiling division below.
    """
    return 1 << (n - 1).bit_length()


class _Projection(nn.Module):
    """A weight/bias container laid out exactly like the reference's L1 ``Linear``.

    The names and shapes matter more than the arithmetic: the benchmark shares
    weights by calling ``load_state_dict(reference.state_dict(), strict=False)``
    inside a bare ``try/except``, and ``strict=False`` does not complain about a key
    mismatch in the first place. A submodule renamed here would leave this module on
    its own random weights; the correctness comparison would then fail, but it would
    fail looking like bad arithmetic rather than like unshared weights, which is why
    ``checks/check_contract.py`` asserts the key set directly.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class CLIPAttention(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = _Projection(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = _Projection(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = _Projection(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = _Projection(self.embed_dim, self.embed_dim, bias=True)

        # The packed Q/K/V projection lives in plain attributes rather than a Parameter
        # or a buffer. Keeping state_dict at the reference's eight keys is the reason to
        # avoid a *persistent* buffer, but `persistent=False` would satisfy that too;
        # the reason to avoid a buffer entirely is that `._apply()` drags a registered
        # buffer through every `.to()`/`.half()` into a copy that is discarded anyway,
        # since moving the module changes the parameters' data_ptr and forces a rebuild.
        # Concatenating per call is the alternative, and it would cost a device op: at
        # this size a 2.25 MB copy is worth about a quarter of the whole budget.
        self._packed_key: tuple | None = None
        self._packed_weight: torch.Tensor | None = None
        self._packed_weight_t: torch.Tensor | None = None
        self._packed_bias: torch.Tensor | None = None
        # Read by checks/check_contract.py, which uses it to prove the pack is built
        # once across warmup plus every timed iteration -- the only externally
        # observable difference between a weight prepack and a cached result.
        self._packed_builds = 0

    # ------------------------------------------------------------------ #
    # packed Q/K/V projection
    # ------------------------------------------------------------------ #
    @staticmethod
    def _source_identity(tensor: torch.Tensor) -> tuple:
        """Everything about a source tensor that the pack depends on, except its values.

        Enumerated as a helper applied to every source rather than spelled out inline per
        source, because an inline key is exactly how this went wrong twice: it listed
        ``stride(0)`` for the weights and no layout at all for the biases, so an
        ``as_strided`` rebind to strides ``(768, 0)`` kept the pointer, the version, the
        shape, the dtype, the device *and* ``stride(0)`` identical while changing the values
        the pack would read. Centralizing the schema is what stops weights and biases
        drifting apart again; it does not stop a whole source being left out of the call
        site below, so that list is worth reading against ``__init__``.

        ``is_neg`` and ``is_conj`` are here because they are lazy: ``w.detach()._neg_view()``
        leaves every other field identical while ``torch.cat`` reads negated values.
        """
        return (tensor.data_ptr(), tensor._version, tensor.shape, tensor.stride(),
                tensor.storage_offset(), tensor.dtype, tensor.device,
                tensor.is_neg(), tensor.is_conj())

    def _packed_qkv(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The packed weight as an ``[E, 3E]`` op=T view, and the ``[3E]`` bias.

        Rebuilt whenever any source's recorded fingerprint changes -- data pointer, version
        counter, shape, strides, storage offset, dtype, device, or the lazy negative and
        conjugate bits. For ordinary dense strided tensors that detects the layout and
        address changes those fields represent, plus any mutation that increments that
        source's version counter: moving the module to a device replaces each parameter's
        storage, ``load_state_dict`` copies in place and bumps ``_version``, and rebinding a
        parameter to a same-storage view of itself changes its strides or offset. A
        ``load_state_dict`` post-hook would be the more obvious mechanism, but the benchmark
        swallows exceptions from that call, so a hook that ever raised would leave the
        module quietly running on unshared weights.

        Unversioned storage changes and other unrecorded tensor state are outside that
        guarantee, and no metadata key can bring them inside it. Specifically: mutation
        through ``.data`` (``p.data.mul_(2)`` changes values without bumping ``p._version``,
        where ``p.mul_(2)`` does); writes through another tensor aliasing the same
        ``UntypedStorage``, which bump only that alias's counter; a freed storage whose
        address the caching allocator hands back to a new same-metadata tensor, since the
        cached key holds an integer rather than a storage reference; and tensor subclasses
        whose ``__torch_dispatch__`` changes what ``cat`` reads without touching any of
        these fields. None occurs in this benchmark, which mutates weights only through
        ``load_state_dict``.

        Costs ~2.7 us of host time per call, and whether that reaches the measured latency
        depends on which GPU the run lands on. The benchmark memsets a 253 MB L2 flush
        buffer before its start event; when that parks the device long enough for the host to
        queue the whole forward ahead of it, the host work is free, and when it does not, the
        host is the bottleneck and its cost passes straight through. Both were measured:
        bypassing all fast-path host work moved the measurement by +0.02 us on one leased gpu
        and +12.66 us on another. Completing this key -- rather than the earlier version that
        keyed only ``stride(0)`` on the weights -- accounts for +3.11 us of that, below the
        noise floor of the run that measured it. So this is a real if small cost on
        host-bound GPUs, accepted because the narrow key was unsound. See
        ``profile/probe_candidate_perf.log`` section [4], which prices both.
        """
        key = (self._source_identity(self.q_proj.weight),
               self._source_identity(self.q_proj.bias),
               self._source_identity(self.k_proj.weight),
               self._source_identity(self.k_proj.bias),
               self._source_identity(self.v_proj.weight),
               self._source_identity(self.v_proj.bias))
        if key != self._packed_key:
            with torch.no_grad():
                self._packed_weight = torch.cat(
                    [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], 0)
                # addmm wants op=T, and the transpose is a view derived from the pack,
                # so the key that invalidates the pack covers it too.
                self._packed_weight_t = self._packed_weight.t()
                self._packed_bias = torch.cat(
                    [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], 0)
            self._packed_key = key
            self._packed_builds += 1
        return self._packed_weight_t, self._packed_bias

    # ------------------------------------------------------------------ #
    # fast-path applicability
    # ------------------------------------------------------------------ #
    def _fast_path_mask(self, hidden_states: torch.Tensor,
                        attention_mask: torch.Tensor | None):
        """Return ``(True, mask_or_None)`` when the fast path applies.

        The mask is returned already expanded to ``[B, H, S, S]`` so the broadcast
        dimensions carry zero strides; anything read by explicit strides has to be
        handed a view in that form or a size-1 dimension's packed stride would walk
        off the end of the tensor.
        """
        if hidden_states.dtype != torch.float32 or not hidden_states.is_cuda:
            return False, None
        if hidden_states.dim() != 3 or hidden_states.shape[-1] != self.embed_dim:
            return False, None
        # Conservative rather than required: the kernel never reads hidden_states (it
        # reads the packed projection output), and `reshape` below would silently
        # materialize a contiguous copy. But that copy is a fourth device op, and the
        # whole point of this path is that it is exactly three, so route it away.
        if not hidden_states.is_contiguous():
            return False, None
        head_dim = self.head_dim
        if head_dim & (head_dim - 1) or not _MIN_BLOCK <= head_dim <= _MAX_HEAD_DIM:
            return False, None
        batch, seq_length, _ = hidden_states.shape
        if seq_length < _MIN_SEQ_FAST or _next_pow2(seq_length) > _MAX_BLOCK_N:
            return False, None
        if attention_mask is None:
            return True, None
        if (attention_mask.dtype != hidden_states.dtype
                or attention_mask.device != hidden_states.device):
            return False, None
        try:
            expanded = attention_mask.expand(batch, self.num_heads,
                                             seq_length, seq_length)
        except RuntimeError:
            return False, None
        return True, expanded

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        applies, mask = self._fast_path_mask(hidden_states, attention_mask)
        if not applies:
            return self._reference_forward(hidden_states, attention_mask)
        return self._fused_forward(hidden_states, mask)

    def _fused_forward(self, hidden_states: torch.Tensor,
                       attention_mask: torch.Tensor | None) -> torch.Tensor:
        """Three device ops: one projection for Q/K/V, attention, one projection out.

        Fusing the three projections into a single ``addmm`` is bit-exact against
        three separate ``F.linear`` calls, because cuBLAS's accumulation over K does
        not depend on how the output columns are tiled. Both projections stay on
        cuBLAS so they are bit-exact by construction; only the attention body in the
        middle is hand-written, and it carries a measured ~1.1e-05 residual against
        a comparison bound of ~4.2e-05.
        """
        batch, seq_length, embed_dim = hidden_states.shape
        heads, head_dim = self.num_heads, self.head_dim
        packed_weight_t, packed_bias = self._packed_qkv()

        qkv = torch.addmm(packed_bias, hidden_states.reshape(batch * seq_length, embed_dim),
                          packed_weight_t)
        attn_output = torch.empty((batch * seq_length, embed_dim),
                                  dtype=qkv.dtype, device=qkv.device)

        if attention_mask is None:
            # HAS_MASK=False compiles the load away, so MASK is never dereferenced;
            # it still needs to be some valid pointer, and qkv is one already at hand.
            mask_arg, mask_strides = qkv, (0, 0, 0, 0)
        else:
            mask_arg, mask_strides = attention_mask, attention_mask.stride()

        block_n = _next_pow2(seq_length)
        grid = (-(-seq_length // _BLOCK_M), batch * heads)
        _clip_attention_kernel[grid](
            qkv, mask_arg, attn_output,
            qkv.stride(0), attn_output.stride(0), *mask_strides,
            seq_length, embed_dim, heads, self.scale,
            HAS_MASK=attention_mask is not None,
            BLOCK_M=_BLOCK_M, BLOCK_N=block_n, BLOCK_D=head_dim,
            num_warps=_NUM_WARPS,
        )

        out = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        return out.view(batch, seq_length, embed_dim)

    def _reference_forward(self, hidden_states: torch.Tensor,
                           attention_mask: torch.Tensor | None) -> torch.Tensor:
        """The reference computation, op for op, for inputs the fast path rejects.

        Correctness never depends on the fast path being applicable: a non-fp32
        dtype, a head dimension that is not a small power of two, a sequence longer
        than the kernel's padded range, a non-contiguous input or a mask that does
        not broadcast to ``[B, H, S, S]`` all land here.
        """
        batch_size, seq_length, _ = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights.float(), dim=-1).to(queries.dtype)

        attn_output = torch.matmul(attn_weights, values)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.embed_dim)
        return self.out_proj(attn_output)
