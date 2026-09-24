"""Qwen3 Mixture-of-Experts with the expert dispatch fused into three kernels.

The reference module delegates expert dispatch to ``L2/fused_experts.py``, whose
DeepGEMM path expresses the permute / unpermute glue as roughly 35 eager PyTorch
operations. Measured on B200 (``profile/baseline_profile.txt``) that glue, not the
arithmetic, is the cost: at ``M = 16384`` the two DeepGEMM calls are 2.30 ms of an
8.99 ms forward and the permute plus unpermute are 5.7 ms of it; at ``M = 1`` only
65.1 us of the 184.2 us forward is kernel time at all, the rest being launch and
Python overhead.

Every one of the 128 experts is active even at ``M = 314`` (2512 assignments), so
any implementation has to stream ``w13 + w2 = 2.42 GB`` of expert weights -- a
~302 us floor at 8 TB/s that the measured GEMMs (465-534 us) already sit within
1.5-1.8x of. So the win here is overhead removal, not a better GEMM. DeepGEMM
stays, ``F.linear`` stays for the router, the vendored activation quantizers stay
(or are reproduced byte-for-byte), and the glue around them becomes:

* ``_derive_dg_metadata_kernel`` -- one pass over the aligner's output producing
  DeepGEMM's per-row ``m_indices`` and the token-pair -> row map ``dest``,
  replacing an argsort / searchsorted / scatter_reduce / arange chain of about 24
  launches and 340 us.
* ``_permute_quant_scatter_kernel`` -- reads each token's bfloat16 row once,
  quantizes it to FP8 with UE8M0 group scales, and writes the bytes and scales to
  all ``top_k`` destination rows. Replaces the activation quantizer, a 603 MB
  ``torch.zeros``, an ``index_put`` scatter that ran at 214 GB/s, and five scale
  index kernels -- about 2.8 ms at ``M = 16384``.
* ``_unpermute_weighted_sum_kernel`` -- gathers each token's ``top_k`` rows,
  weights them and reduces in float32. Replaces three gathers, a broadcast
  multiply and a ``sum(dim=1)`` -- about 3.08 ms at ``M = 16384``.

Numerics drive most of the design. The harness compares bfloat16 at
``atol = rtol = 1e-2`` against a reference that is itself an FP8 pipeline, and an
e4m3 mantissa is 3 bits, so one quantization step is 12.5% of an element. A
perturbation the size of a bfloat16 ULP (0.39%) landing *upstream* of the second
quantizer flips ~3% of the FP8 bytes by a full step, which in a 1536-term dot
product is ~2% relative error -- over the bound, on most elements. Hence:

* the router GEMM stays ``F.linear`` and the top-k stays the reference's, so
  routing is bit-identical (one flipped expert moves a token's whole row by ~1/8);
* the path threshold stays the reference's ``M >= 128``, because the two paths use
  *different* second quantizers (fused ``SiluMulQuantFp8`` with
  ``absmax * (1/448)`` versus ``SiluAndMul`` + ``PerTokenGroupQuantFp8`` with
  ``absmax / 448``) and those are not the same function;
* ``SiluAndMul`` is imported from the reference's absolute module path rather than
  taken from ``candidate/L1``, whose bfloat16 default evaluates silu through
  ``tanh.approx.f32`` at ~2**-11 relative error -- about ten bfloat16 ULPs,
  landing exactly upstream of the second quantizer;
* the fused activation quantizer reproduces the vendored *CUDA* kernel's
  arithmetic bit-for-bit, which ``tools/check_quant.py`` asserts by
  ``torch.equal`` on both the FP8 bytes and the FP32 scale bit patterns.

The class subclasses the reference module rather than restating it. That is the
lower-risk half of the choice the plan left open: the parameter schema, all six
weight loaders and the bit-identical ``TopKSoftmax`` come along by construction
instead of by transcription, and ``isinstance(m, Qwen3MoE)`` -- which
``infra/weight_loader.py`` uses to find the modules that get ``w13_scale_dg`` --
keeps matching outside the bench, where the bench's own duck-typed
``_init_fp8_module_weights`` would not have needed it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

# Absolute reference imports. The candidate finder intercepts only
# ``fastkernels.tasks.candidate.*``, so these always resolve to the reference and
# are never redirected to a candidate file. Used where a component has to be
# bit-identical, or where the candidate must agree with the reference's own
# decision rather than with a reimplementation of it.
from fastkernels.tasks.baseline.L1.moe_grouped_gemm import (
    _deep_gemm_alignment,
    _is_deep_gemm_supported,
    get_triton_config,
)
# expf-based silu, not the ``tanh.approx.f32`` candidate: this feeds the second
# quantizer, and see the module docstring for why that matters.
from fastkernels.tasks.baseline.L1.silu_and_mul import SiluAndMul
from fastkernels.tasks.baseline.L1.silu_mul_quant_fp8 import SiluMulQuantFp8
from fastkernels.tasks.baseline.L2.qwen3_moe import Qwen3MoE as _ReferenceQwen3MoE

# Frozen lower-level winners, by relative import: ``..L1.x`` resolves to the
# frozen file when one exists and falls back to the reference when it does not.
# ``fp8_grouped_gemm_contiguous`` has no candidate file, so it is the reference's
# -- taken for its cached alignment / ``disable_ue8m0_cast`` and its flat
# signature, both of which keep pybind lookups out of the timed path.
from ..L1.fp8_grouped_gemm_contiguous import Fp8GroupedGemmContiguous
from ..L1.fp8_linear import PerTokenGroupQuantFp8
from ..L1.moe_align import MoeAlign
from ..L1.moe_grouped_gemm import MoeGroupedGemm
from ..L1.moe_sum import MoeSum

_FP8_GROUP = 128
_FP8_MAX = 448.0
_FP8_MIN = -448.0
# The vendored CUDA quantizer's epsilon, both as the absmax floor and as the
# floor on the quotient (``fp8_linear.cu``: ``local_absmax`` starts at ``eps``,
# and ``y_s = exp2f(ceilf(log2f(fmaxf(fabsf(y_s), 1e-10f))))``).
_QUANT_EPS = 1e-10

# The reference's ``FusedExperts`` rule for skipping the aligner's full sort.
_SPARSITY_FACTOR = 4

# Fuse SiluAndMul with the second quantizer on the Triton path. Both halves are
# byte-exact against the vendored kernels and verified separately
# (tools/probe_act_quant.py, tools/check_quant.py), and the pair is verified against
# the reference's two-launch chain (tools/check_act_quant.py). It exists to meet
# AC-9's launch clause at M = 1, where the reference and a two-launch candidate both
# sit at ten launches. Set to False to fall back to the two vendored launches.
_FUSE_ACT_QUANT = True

# Keep the aligner's full sort on the small-M path even where the reference's
# sparsity rule would skip it. The shortcut returns ``sorted_token_ids = None``,
# and the frozen grouped GEMM's thin-decode kernel requires that array -- so
# skipping the sort costs the fast kernel and falls back to the reference one.
# Measured in tools/measure_decode.py; that measurement is what set this.
_ALIGNED_DECODE = True

# Every flat element index the DeepGEMM path forms fits in signed 32-bit at the
# benchmarked shapes -- the largest is ``M_sum * K = 147328 * 4096 = 603,455,488``
# against a 2,147,483,647 limit. The kernels below still form their row bases in
# 64-bit (one scalar multiply per destination row, and no measurable cost at these
# sizes), so a shape past that bound stays correct rather than wrapping; the host
# assertion exists to say where the 32-bit reasoning would have run out.
_INT32_MAX = 2147483647

# Permute kernel geometry: how many 128-element scale groups one program owns, and
# how many warps run it. Measured on B200 at M = 16384 (688 MB moved) in
# tools/tune_kernels.py, best first:
#
#   GROUPS=32 warps=8  139.3 us  4.94 TB/s     GROUPS=8  warps=2  158.8 us  4.33
#   GROUPS=16 warps=8  144.4 us  4.76 TB/s     GROUPS=4  warps=1  205.9 us  3.34
#
# Wider wins because the kernel is LSU-bound rather than DRAM-bound: NCU puts
# l1tex__throughput at 76% of peak against 51% of DRAM peak, so what helps is
# fewer, wider requests per byte moved -- a whole 4096-wide row per program at
# GROUPS=32, scattered as eight 4 KiB runs. The table is indexed by how many groups
# divide K/128 so a hidden size that is not a multiple of 4096 still gets the widest
# geometry it can use rather than an assertion.
_PERMUTE_GEOMETRY = ((32, 8), (16, 8), (8, 4), (4, 1))

# Unpermute kernel geometry. Measured in the same sweep (1208 MB moved):
# BLOCK_K=1024 num_warps=2 at 224.3 us / 5.39 TB/s, against 235.6 us at 4 warps and
# 224.3 at BLOCK_K=512 num_warps=1. This one is much closer to the DRAM roofline
# (NCU: 64% of DRAM peak, L2 at 70%), so the geometry barely matters.
_UNPERMUTE_BLOCK_K = 1024
_UNPERMUTE_WARPS = 2


def _permute_geometry(K: int) -> tuple[int, int]:
    """``(groups_per_program, num_warps)`` for this hidden size."""
    n_groups = K // _FP8_GROUP
    for groups, warps in _PERMUTE_GEOMETRY:
        if n_groups % groups == 0:
            return groups, warps
    return 1, 1

_DERIVE_BLOCK = 1024


# ---------------------------------------------------------------------------
# Routing metadata
# ---------------------------------------------------------------------------
@triton.jit
def _derive_dg_metadata_kernel(
    sorted_ids_ptr,   # int32 [max_padded]; every padding slot holds `numel`
    expert_ids_ptr,   # int32 [max_blocks]; written only over the used blocks
    m_indices_ptr,    # int32 [M_sum]  out: per-row expert, -1 = skip
    dest_ptr,         # int32 [numel]  out: token-pair index -> padded row
    numel,            # M * top_k, and the aligner's padding sentinel
    max_padded,
    M_sum,
    LOG2_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Turn the aligner's ``(sorted_token_ids, expert_ids)`` into DeepGEMM's
    contiguous-layout metadata.

    Three facts about the aligner set the load order here. Its
    ``sorted_token_ids`` is defined only over ``[0, max_padded)``, and
    ``max_padded = numel + E*(block-1)`` is up to ``block-1`` rows *short* of the
    ``M_sum = max_blocks*block`` the host hands DeepGEMM -- so the tail is masked
    off and defaults to the sentinel, which marks it padding. Its ``expert_ids``
    is written only over the blocks it actually used, leaving the rest stale, so
    that array is read only after a slot has been shown to hold a real token --
    which puts the slot inside the used range by construction. And its padding
    sentinel is exactly ``numel``, so ``tok < numel`` is the liveness test.
    """
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_sum = r < M_sum
    in_src = r < max_padded
    tok = tl.load(sorted_ids_ptr + r, mask=in_src, other=numel)
    live = tok < numel
    eid = tl.load(expert_ids_ptr + (r >> LOG2_BLOCK), mask=live, other=-1)
    tl.store(m_indices_ptr + r, tl.where(live, eid, -1), mask=in_sum)
    # Injective: the aligner writes each token-pair index into exactly one slot.
    tl.store(dest_ptr + tok, r, mask=live)


# ---------------------------------------------------------------------------
# Activation quantization
# ---------------------------------------------------------------------------
@triton.jit
def _ue8m0_group_quant(x_f32, FP8_MAX: tl.constexpr, EPS: tl.constexpr):
    """Per-group UE8M0 FP8 quantization, byte-identical to the vendored CUDA
    ``per_token_group_quant_8bit_kernel`` (``baseline/L1/fp8_linear.cu``).

    ``x_f32`` is ``[GROUPS, group_size]``; returns ``(q_f32, y_s)`` with ``q_f32``
    already clamped and ready for a round-to-nearest-even cast to e4m3, and
    ``y_s`` the float32 scale, one per group.

    The reference computes, in this order:

        local_absmax = fmaxf over the group, seeded at eps
        y_s = local_absmax / max_8bit
        y_s = exp2f(ceilf(log2f(fmaxf(fabsf(y_s), 1e-10f))))
        q   = fminf(fmaxf(float(x) / y_s, min_8bit), max_8bit)

    Three of those four steps need care to reproduce exactly.

    *The division.* ``tl.fdiv(..., ieee_rounding=True)`` is used so this is the
    IEEE correctly-rounded quotient, matching the reference's ``/`` exactly rather
    than approximately -- Triton lowers a plain ``/`` by a constexpr divisor to a
    reciprocal multiply that is 1 ULP off.

    It is worth being precise about how much that 1 ULP actually costs here,
    because the intuition is wrong. It costs *nothing*: for this ``fp8_max`` the
    two forms give the same UE8M0 exponent for every float32 ``absmax``.
    ``448 = 1.75 * 2**8``, so ``fl(1/448)`` sits 0.75 * 2**-24 above the exact
    reciprocal, while the float32-representable quotients near a binade boundary
    are spaced 1.143 * 2**-24 apart with one of them landing exactly *on* the
    boundary -- a nudge that small, from a grid that coarse, anchored on the
    boundary, cannot cross it. ``tools/check_quant_domain.py`` derives that and
    checks it exhaustively over every reachable ``absmax``. The correctly-rounded
    form is still used, because matching the reference exactly costs nothing and
    the argument above depends on a specific constant.

    Where the vendored CUDA kernel and the vendored *Triton* fallback in
    ``fp8_linear.py`` genuinely differ is the **epsilon placement**: the CUDA one
    floors the quotient (``fmaxf(fabsf(y_s), 1e-10f)`` after dividing), the Triton
    one floors ``absmax`` before multiplying. That separates them on 13026 of the
    32641 reachable ``absmax`` values -- every small one -- and it is what
    ``tools/check_quant.py``'s mutation gate rests on.

    *The UE8M0 rounding.* ``exp2f(ceilf(log2f(v)))`` is reproduced by exponent
    arithmetic rather than by ``libdevice`` ``log2``/``exp2``: a 1 ULP error in a
    software ``log2`` moves ``ceil`` across every binade boundary. For a positive
    normal ``v = 2**(e-127) * (1+m)``, incrementing the biased exponent when the
    mantissa is non-zero *is* ``2**ceil(log2(v))``, and the same ``.cu`` file
    documents this form as bit-exact against the ``exp2f(ceilf(log2f(.)))``
    reference. ``v`` here is always a positive normal: it is floored at 1e-10 and
    bounded above by ``bf16_max / 448``, so the biased exponent stays in [94, 247]
    and neither the increment nor the reciprocal below can leave the normal range.

    *The scaling.* ``y_s`` is a power of two, so ``1/y_s`` is exact (biased
    exponent ``254 - exp_byte``, inside [7, 160]) and ``x * (1/y_s)`` is the
    identical IEEE result to ``x / y_s`` -- both operations are required to return
    the correctly rounded value of the same real number, and that holds even where
    the product is subnormal, since gradual underflow rounds the same real value
    the same way. The multiply is used because ``div.rn.f32`` on all ``M*K``
    elements would cost tens of microseconds at ``M = 16384`` for no numerical
    difference. ``check_quant.py`` covers it, including a group holding both a
    huge and a tiny value so the subnormal-product case is exercised rather than
    assumed.

    That equivalence is scoped to **finite** inputs, and this is the one place the
    two forms can be told apart: a bfloat16 ``+-Inf`` in the group makes
    ``absmax = Inf``, hence ``y_s = Inf``, and the exponent-bit construction then
    yields a reciprocal of the wrong sign rather than the ``0`` that dividing by
    ``Inf`` would give. The harness rejects any output containing ``Inf`` on either
    side, and the activations are ``randn``, so no such input reaches here -- but
    the claim is "byte-identical on finite inputs", not "on all bit patterns".
    """
    absmax = tl.maximum(tl.max(tl.abs(x_f32), axis=1), EPS)
    y_s = tl.fdiv(absmax, FP8_MAX, ieee_rounding=True)
    y_s = tl.maximum(y_s, EPS)
    # Signed int32 throughout: y_s is positive so its sign bit is clear, which
    # keeps the arithmetic shift equal to a logical one and keeps both
    # ``exp_byte << 23`` (<= 248 << 23) and ``(254 - exp_byte) << 23``
    # (<= 160 << 23) inside int32.
    bits = y_s.to(tl.int32, bitcast=True)
    exp_byte = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    y_s = (exp_byte << 23).to(tl.float32, bitcast=True)
    inv = ((254 - exp_byte) << 23).to(tl.float32, bitcast=True)
    q = tl.minimum(tl.maximum(x_f32 * inv[:, None], -FP8_MAX), FP8_MAX)
    return q, y_s


@triton.jit
def _permute_quant_scatter_kernel(
    x_ptr,            # bfloat16 [M, K]
    dest_ptr,         # int32    [M * TOP_K]
    a_perm_ptr,       # fp8      [M_sum, K]
    a_scale_ptr,      # float32  [M_sum, K // 128]
    K,
    NSCALE,           # K // 128
    FP8_MAX: tl.constexpr,
    EPS: tl.constexpr,
    TOP_K: tl.constexpr,
    GROUPS: tl.constexpr,
    GSIZE: tl.constexpr,
):
    """Read one token's slice of bfloat16 once; write FP8 bytes and FP32 scales to
    all ``top_k`` destination rows.

    Fusing the quantizer in saves reading and writing the intermediate ``[M, K]``
    FP8 tensor entirely (67 MB each way at ``M = 16384``) plus a launch, and the
    scatter is what removes the reference's ``index_put``, which moved 537 MB of
    single bytes at 214 GB/s.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    g0 = pid_k * GROUPS
    col0 = g0 * GSIZE
    offs = tl.arange(0, GROUPS)[:, None] * GSIZE + tl.arange(0, GSIZE)[None, :]

    x = tl.load(x_ptr + pid_m.to(tl.int64) * K + col0 + offs,
                eviction_policy="evict_first")
    q, y_s = _ue8m0_group_quant(x.to(tl.float32), FP8_MAX, EPS)
    qb = q.to(a_perm_ptr.dtype.element_ty)

    gidx = g0 + tl.arange(0, GROUPS)
    pair0 = dest_ptr + pid_m * TOP_K
    for k in tl.static_range(TOP_K):
        r = tl.load(pair0 + k).to(tl.int64)
        tl.store(a_perm_ptr + r * K + col0 + offs, qb)
        tl.store(a_scale_ptr + r * NSCALE + gidx, y_s)



@triton.jit
def _silu_mul_quant_kernel(
    x_ptr,            # bfloat16 [rows, 2N]  (gate | up)
    q_ptr,            # fp8      [rows, N]   out
    s_ptr,            # float32  [rows, N/128] out
    N2,               # 2N
    N,
    NSCALE,           # N // 128
    FP8_MAX: tl.constexpr,
    EPS: tl.constexpr,
    GROUPS: tl.constexpr,
    GSIZE: tl.constexpr,
):
    """``SiluAndMul`` followed by ``PerTokenGroupQuantFp8``, in one launch.

    This is the Triton path's second quantizer, and it is the only launch on that
    path that could be removed without touching ``F.linear`` or changing either
    quantizer's arithmetic -- which is what AC-9's launch clause needs at ``M = 1``,
    where the reference and a two-launch candidate both sit at ten.

    Fusing here puts an activation *upstream* of a quantizer, the one position the
    whole design treats as dangerous, so both halves are byte-exact rather than
    approximately right, and both are verified independently:

    * the activation reproduces ``baseline/L1/silu_and_mul.cu``'s
      ``packed_compute`` with ``act_first``, ``alpha = 1``, ``beta = 0`` -- that is
      ``bfloat16(silu_f32(gate))``, widened back to float32, times
      ``float32(up)``, rounded to bfloat16. The intermediate really is rounded to
      bfloat16 in the middle; skipping that would change the result. The rounding is
      also what makes the choice of ``exp`` immaterial: a bfloat16 significand is 8
      bits against ``expf``'s ~1 float32 ULP, and
      ``tools/probe_act_quant.py`` finds ``tl.exp`` bit-identical to the vendored
      kernel over 87 million elements. That is a measurement over the reachable
      distribution, not a proof for every bit pattern.
    * the quantizer is ``_ue8m0_group_quant``, byte-identical to the vendored CUDA
      kernel by ``tools/check_quant.py``.

    ``tools/check_act_quant.py`` then checks the *fused* pair against the two-launch
    chain the reference runs, which is the claim that actually matters.
    """
    row = tl.program_id(0)
    blk = tl.program_id(1)
    g0 = blk * GROUPS
    col0 = g0 * GSIZE
    offs = tl.arange(0, GROUPS)[:, None] * GSIZE + tl.arange(0, GSIZE)[None, :]

    base = x_ptr + row.to(tl.int64) * N2 + col0 + offs
    gate = tl.load(base).to(tl.float32)
    up = tl.load(base + N).to(tl.float32)
    # bfloat16 in the middle, exactly as the vendored kernel does.
    act = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16)
    y = (act.to(tl.float32) * up).to(tl.bfloat16)

    q, y_s = _ue8m0_group_quant(y.to(tl.float32), FP8_MAX, EPS)
    tl.store(q_ptr + row.to(tl.int64) * N + col0 + offs,
             q.to(q_ptr.dtype.element_ty))
    tl.store(s_ptr + row.to(tl.int64) * NSCALE + g0 + tl.arange(0, GROUPS), y_s)


# ---------------------------------------------------------------------------
# Unpermute
# ---------------------------------------------------------------------------
@triton.jit
def _unpermute_weighted_sum_kernel(
    mm2_ptr,          # bfloat16 [M_sum, K]
    dest_ptr,         # int32    [M * TOP_K]
    w_ptr,            # float32  [M, TOP_K]
    out_ptr,          # bfloat16 [M, K]
    K,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """``out[m] = sum_k topk_weights[m, k] * mm2_out[dest[m, k]]``.

    Same arithmetic as the reference expression
    ``(gathered.to(bf16) * weights).sum(dim=1)``: a bfloat16 value times a float32
    weight (``TopKSoftmax`` returns float32), accumulated in float32, stored
    bfloat16.

    The accumulate is written as a Triton multiply followed by an *opaque*
    ``add.rn.f32``, which is not decoration. The reference materialises the product
    tensor before summing it, so each ``w * v`` gets its own float32 rounding; a
    plain ``acc += w * v`` here contracts into an FMA, which rounds once and
    disagrees on 0.0019% of elements. LLVM cannot fold a multiply into inline asm,
    so this form keeps the two roundings the reference has. Measured in
    ``tools/probe_reduction.py``: it lowers the disagreeing fraction from 0.00746%
    to 0.00558% at ``M = 1000``, at no reliable cost. Dropping back to
    ``acc += w * v`` is a one-line change if the inline asm ever becomes a problem.

    What is left is the reduction *order*, and it cannot be matched.
    ``tools/probe_reduction.py`` probes torch's ``sum(dim=1)`` over eight terms with
    adversarial magnitudes and finds no fixed tree that reproduces it: the closest
    of five natural trees (four accumulators at stride four) agrees on 74.9% of
    elements, sequential ascending on 50.1%. The order is reproducible for a given
    shape but is chosen by a launch-configuration heuristic over the whole tensor,
    so there is nothing for a kernel to encode. The residual is characterised in
    ``tools/check_unpermute.py``.

    This is also where padding safety is decided: the gather reads only the rows
    named by ``dest``, and those are exactly the live rows. Output rows are
    independent dot products, so whatever a skipped row of ``mm2_out`` holds --
    including NaN -- cannot reach a row that is read. The claim is that
    independence, not the weaker "no NaN reaches a GEMM"; ``tools/probe_poison.py``
    tests it by poisoning the padding rows and comparing the output bitwise.

    ``EVEN_K`` is ``K % BLOCK_K == 0``, which holds for the captured
    ``hidden_size = 4096`` and lets the fast path compile without a mask. It must
    not be assumed: the path predicate this kernel sits behind is the reference's
    ``_valid_deep_gemm``, which accepts *any* ``K`` divisible by DeepGEMM's
    alignment (128 on this build) -- so ``K = 1152`` or ``4224`` reaches here, and
    an unmasked ``ceil(K / BLOCK_K)`` grid would read past ``mm2_out`` and write
    past ``out``, corrupting the following row. ``tools/check_unpermute.py`` guards
    both buffers with canaries at those sizes and fails against the unmasked form.
    """
    pid_m = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    pair0 = dest_ptr + pid_m * TOP_K
    wrow = w_ptr + pid_m * TOP_K
    if EVEN_K:
        for k in tl.static_range(TOP_K):
            r = tl.load(pair0 + k).to(tl.int64)
            w = tl.load(wrow + k)
            v = tl.load(mm2_ptr + r * K + cols, eviction_policy="evict_first")
            acc = tl.inline_asm_elementwise(
                "add.rn.f32 $0, $1, $2;", "=f,f,f",
                [acc, w * v.to(tl.float32)],
                dtype=tl.float32, is_pure=True, pack=1)
        tl.store(out_ptr + pid_m.to(tl.int64) * K + cols,
                 acc.to(out_ptr.dtype.element_ty))
    else:
        tail = cols < K
        for k in tl.static_range(TOP_K):
            r = tl.load(pair0 + k).to(tl.int64)
            w = tl.load(wrow + k)
            v = tl.load(mm2_ptr + r * K + cols, mask=tail, other=0.0,
                        eviction_policy="evict_first")
            acc = tl.inline_asm_elementwise(
                "add.rn.f32 $0, $1, $2;", "=f,f,f",
                [acc, w * v.to(tl.float32)],
                dtype=tl.float32, is_pure=True, pack=1)
        tl.store(out_ptr + pid_m.to(tl.int64) * K + cols,
                 acc.to(out_ptr.dtype.element_ty), mask=tail)


# ---------------------------------------------------------------------------
# Scratch
# ---------------------------------------------------------------------------
class _Scratch:
    """One set of expert-dispatch scratch buffers, shared by every layer.

    A real 94-layer model would otherwise hold 94 copies of ~2 GB. Layers run
    sequentially and every buffer is dead by the time ``forward`` returns, so one
    set is enough; the *output* is not shared, because the next layer can be
    reading a buffer while this one rewrites it.

    Buffers grow monotonically and are zero-filled only when (re)allocated, never
    per call -- that is what removes the reference's 151.5 us
    ``FillFunctor<Float8_e4m3fn>`` over 603 MB on every forward. ``generation``
    counts reallocations so callers can drop views that now alias a freed block.
    """

    __slots__ = ("a_perm", "a_scale", "ws", "quant_out", "m_indices", "dest",
                 "generation")

    def __init__(self) -> None:
        self.a_perm = None
        self.a_scale = None
        self.ws = None
        self.quant_out = None
        self.m_indices = None
        self.dest = None
        self.generation = 0

    def _get(self, name, numel, dtype, device, zero):
        buf = getattr(self, name)
        if buf is None or buf.numel() < numel or buf.device != device:
            new = (torch.zeros if zero else torch.empty)(
                numel, dtype=dtype, device=device)
            setattr(self, name, new)
            self.generation += 1
            return new[:numel]
        return buf[:numel]

    def plan(self, M_sum, K, N2, numel, device, dtype):
        """Views for one ``(M_sum, K, N2)``, plus the generation they belong to."""
        n_scale = K // _FP8_GROUP
        N = N2 // 2
        gen0 = self.generation
        # Zero-filled: only the live rows of these two are written each call, so a
        # padding row holds either a zero from allocation or a live value from an
        # earlier, larger call. Never uninitialized bytes -- an FP8 0x7f/0xff pair
        # is a NaN, and while the unpermute would not read it, handing DeepGEMM a
        # NaN on the very first call is not worth the saved memset.
        a_perm = self._get("a_perm", M_sum * K, torch.float8_e4m3fn, device, True)
        a_scale = self._get("a_scale", M_sum * n_scale, torch.float32, device, True)
        # ``ws`` backs mm1_out and then mm2_out. Aliasing them is exactly what the
        # reference does: by the time GEMM2 writes, mm1_out has already been
        # consumed by the second quantizer into ``quant_out``, so it is dead.
        ws = self._get("ws", M_sum * max(N2, K), dtype, device, True)
        quant_out = self._get("quant_out", M_sum * N, torch.float8_e4m3fn,
                              device, False)
        # Written over their whole extent on every call: ``m_indices`` for every
        # row in [0, M_sum) and ``dest`` for every token-pair index. No zero-fill.
        m_indices = self._get("m_indices", M_sum, torch.int32, device, False)
        dest = self._get("dest", numel, torch.int32, device, False)
        if self.generation != gen0:
            # An earlier view in this same call may alias a block freed by a later
            # growth, so take them all again now that nothing can grow.
            return self.plan(M_sum, K, N2, numel, device, dtype)
        return {
            "a_perm": a_perm.view(M_sum, K),
            "a_scale": a_scale.view(M_sum, n_scale),
            "mm1_out": ws[:M_sum * N2].view(M_sum, N2),
            "mm2_out": ws[:M_sum * K].view(M_sum, K),
            "quant_out": quant_out.view(M_sum, N),
            "m_indices": m_indices,
            "dest": dest,
            "generation": self.generation,
        }


_SCRATCH = _Scratch()


class _TritonScratch:
    """Scratch for the Triton path, shared for the same reason as ``_Scratch``."""

    __slots__ = ("cache13", "a_fp8_1", "a_scale_1", "a_fp8_2", "a_scale_2",
                 "generation")

    def __init__(self) -> None:
        self.cache13 = None
        self.a_fp8_1 = None
        self.a_scale_1 = None
        self.a_fp8_2 = None
        self.a_scale_2 = None
        self.generation = 0

    def cache(self, numel, device, dtype):
        buf = self.cache13
        if buf is None or buf.numel() < numel or buf.device != device:
            buf = torch.empty(numel, device=device, dtype=dtype)
            self.cache13 = buf
            self.generation += 1
        return buf

    def fp8(self, slot, M, K, device):
        a_name, s_name = f"a_fp8_{slot}", f"a_scale_{slot}"
        groups = -(-K // _FP8_GROUP)
        a = getattr(self, a_name)
        if a is None or a.size(0) < M or a.size(1) < K or a.device != device:
            setattr(self, a_name, torch.empty(M, K, dtype=torch.float8_e4m3fn,
                                              device=device))
            setattr(self, s_name, torch.empty(M, groups, dtype=torch.float32,
                                              device=device))
            a = getattr(self, a_name)
            self.generation += 1
        return a[:M, :K], getattr(self, s_name)[:M, :groups]


_TRITON_SCRATCH = _TritonScratch()


class Qwen3MoE(_ReferenceQwen3MoE):
    """Qwen3 MoE block with the expert dispatch absorbed and fused.

    Same ``__init__(config, quant_config)`` / ``forward(hidden_states)`` contract
    as the reference, and the same parameter schema, weight loaders,
    ``block_shape``, ``use_fp8`` and ``_use_custom_op`` / ``_layer_name`` surface,
    all inherited. Only ``forward_impl`` is replaced.
    """

    def __init__(self, config, quant_config: dict | None = None):
        super().__init__(config, quant_config)

        self.moe_align = MoeAlign()
        self.moe_grouped_gemm = MoeGroupedGemm()
        self.act_fn = SiluAndMul()
        self.moe_sum = MoeSum()
        self.per_token_group_quant_fp8 = PerTokenGroupQuantFp8()
        self.silu_mul_quant_fp8 = SiluMulQuantFp8()

        self._scratch = _SCRATCH
        self._triton_scratch = _TRITON_SCRATCH
        # Keyed on shape metadata only, never on a tensor address: the harness'
        # _ShiftingPool hands the input a different data_ptr every iteration, so a
        # pointer-keyed cache would miss on every call.
        self._dg_plans: dict = {}
        self._triton_plans: dict = {}
        self._triton_configs: dict = {}
        self._sfb_cache: dict = {}

        # Everything in ``_valid_deep_gemm`` that cannot change between calls,
        # folded in here so the per-call test is a comparison and an attribute
        # read. What is left dynamic is ``alignment <= M`` and the input's
        # contiguity, and they are ordered so M = 1 exits on the first test and
        # never pays the capture query.
        self._dg_align = 0
        self._dg_static_ok = False
        self._dg_e8m0 = False
        self._dg_gemm = None
        if self.use_fp8:
            try:
                align = _deep_gemm_alignment()
                N = self.intermediate_per_tp
                self._dg_static_ok = bool(
                    _is_deep_gemm_supported()
                    and N > 512
                    and N % align == 0
                    and self.hidden_size % align == 0
                    and self.w13.dtype == torch.float8_e4m3fn
                    and self.w2.dtype == torch.float8_e4m3fn
                    and self.w13.is_contiguous()
                    and self.w2.is_contiguous()
                )
                self._dg_align = align
                if self._dg_static_ok:
                    self._dg_gemm = Fp8GroupedGemmContiguous()
                    # The same oracle the GEMM wrapper resolves once, reused so the
                    # hoisted weight-scale transform lands in exactly the layout
                    # the GEMM would have produced for itself.
                    self._dg_e8m0 = not self._dg_gemm._disable_ue8m0_cast
            except Exception:  # noqa: BLE001 - no DeepGEMM: the Triton path covers it
                self._dg_static_ok = False
                self._dg_gemm = None

    # -- weight scale layout ----------------------------------------------
    def _sfb(self, scale, weight):
        """The expert weight scale in the layout DeepGEMM will use, transformed
        once instead of on every call.

        ``_postprocess_moe_fp8_weights`` builds ``w13_scale_dg`` / ``w2_scale_dg``
        with ``disable_ue8m0_cast=True``, so they are still float32 in the
        checkpoint block layout. On a part where E8M0 scaling factors are in use --
        which is every Blackwell build with ``VLLM_USE_DEEP_GEMM_E8M0`` set, so all
        of them by default -- DeepGEMM therefore re-transforms them inside *every*
        GEMM call: measured at 102.3 us of GEMM1's 495.6 us at M = 314, and the same
        ~89 us of ``_scatter_gather_elementwise_kernel`` plus part of the
        ``transpose_and_pack_fp32_into_ue8m0`` time appears at every shape, because
        the cost depends only on the weight and not on M.

        These are weights. They do not change between calls, so the transform is a
        host-side derivation that belongs outside the timed path -- the same
        category as folding the static ``_valid_deep_gemm`` predicates into
        ``__init__``. ``transform_sf_into_required_layout`` is idempotent on its own
        output, so handing DeepGEMM the transformed tensor is not merely equivalent
        but *bitwise* identical; ``tools/check_sfb_hoist.py`` asserts that, and the
        activation scale (SFA) is deliberately left alone, since emitting that in
        DeepGEMM's packed layout would mean changing what the permute kernel writes.

        Invalidation. The cache key is the scale tensor's identity, storage address
        and version counter, and ``_load_from_state_dict`` clears it outright. That
        covers every way these tensors are actually produced or replaced:
        ``_postprocess_moe_fp8_weights`` installs a *fresh* ``Parameter`` (identity
        changes), ``sp.data = sp.data.float()`` rebinds the storage (address
        changes), an in-place op on the parameter itself bumps the version, and a
        checkpoint load goes through ``_load_from_state_dict``. The one pattern it
        cannot see is a direct ``param.data.copy_(...)`` from user code outside a
        state-dict load: ``Tensor.data`` hands out a view with its *own* version
        counter, so such a write is invisible by construction. Call
        ``self._sfb_cache.clear()`` if you do that.

        Keying a cache on a weight's address is safe in a way that keying on an
        activation would not be: the harness' ``_ShiftingPool`` moves the *input*
        every iteration, but never the parameters.
        """
        if scale is None:
            return None
        cached = self._sfb_cache.get(id(scale))
        if (cached is not None and cached[0] is scale
                and cached[1] == scale._version
                and cached[2] == scale.data_ptr()):
            return cached[3]
        try:
            import deep_gemm
            out = deep_gemm.transform_sf_into_required_layout(
                sf=scale.data if hasattr(scale, "data") else scale,
                mn=weight.size(1), k=weight.size(2),
                recipe=(1, self.block_shape[0], self.block_shape[1]),
                num_groups=weight.size(0), is_sfa=False,
                disable_ue8m0_cast=not self._dg_e8m0,
            )
        except Exception:  # noqa: BLE001 - no helper on this build: pass it through
            out = scale
        self._sfb_cache[id(scale)] = (scale, scale._version, scale.data_ptr(), out)
        return out

    def _load_from_state_dict(self, *args, **kwargs):
        # A checkpoint load rewrites the block scales through
        # ``param.data.copy_()``, which no version counter can see; drop the
        # transformed copies so the next forward rebuilds them.
        self._sfb_cache.clear()
        return super()._load_from_state_dict(*args, **kwargs)

    # -- DeepGEMM path ----------------------------------------------------
    def _dg_plan(self, M, K, N2, device, dtype):
        """Host-side arithmetic and buffer views for one ``M``, computed once."""
        key = (M, K, N2, dtype, device)
        plan = self._dg_plans.get(key)
        if plan is not None and plan["generation"] == self._scratch.generation:
            return plan

        top_k = self.top_k
        E = self.num_experts
        numel = M * top_k
        # ``M_sum == max_blocks * block`` -- which is what makes the aligner's
        # ``max_blocks`` and DeepGEMM's ``_compute_aligned_M`` agree -- only holds
        # while the aligner takes its ``numel >= num_experts`` branch. On this
        # path M >= alignment = 128, so numel = top_k*M >= 128 >= E is not
        # guaranteed by M alone; assert it rather than assume it.
        assert numel >= E, (
            f"DeepGEMM path needs topk_ids.numel() ({numel}) >= num_experts "
            f"({E}): below that the aligner switches to "
            f"max_padded = numel * block_size and M_sum != max_blocks * 128")
        block = _FP8_GROUP
        max_padded = numel + E * (block - 1)
        max_blocks = -(-max_padded // block)
        M_sum = max_blocks * block
        assert M_sum * K <= _INT32_MAX, (
            f"M_sum * K ({M_sum * K}) exceeds the signed 32-bit limit "
            f"({_INT32_MAX}) this path's index reasoning is stated against")

        bufs = self._scratch.plan(M_sum, K, N2, numel, device, dtype)
        # The permute kernel's column coverage is exactly
        # ``n_kblk * groups * GSIZE``, and _permute_geometry only returns a
        # ``groups`` that divides ``K / GSIZE`` (falling back to 1, which always
        # does), so it needs no tail mask -- but that rests on K being a multiple
        # of the group size, which is also what the path predicate requires.
        assert K % _FP8_GROUP == 0, (
            f"hidden_size ({K}) must be a multiple of {_FP8_GROUP} on the DeepGEMM "
            f"path; the reference's _valid_deep_gemm requires it too")
        groups, warps = _permute_geometry(K)
        plan = {
            "M_sum": M_sum,
            "numel": numel,
            "max_padded": max_padded,
            "derive_grid": (triton.cdiv(M_sum, _DERIVE_BLOCK),),
            "permute_grid": (M, (K // _FP8_GROUP) // groups),
            "permute_groups": groups,
            "permute_warps": warps,
            "unpermute_grid": (M, triton.cdiv(K, _UNPERMUTE_BLOCK_K)),
            "unpermute_even_k": K % _UNPERMUTE_BLOCK_K == 0,
            "generation": bufs["generation"],
        }
        plan.update(bufs)
        self._dg_plans[key] = plan
        return plan

    def _forward_deep_gemm(self, x, topk_weights, topk_ids, M, K, N2,
                           w13_scale, w2_scale):
        p = self._dg_plan(M, K, N2, x.device, x.dtype)
        M_sum = p["M_sum"]
        dest = p["dest"]
        m_indices = p["m_indices"]

        sorted_ids, expert_ids, _ = self.moe_align(
            topk_ids, _FP8_GROUP, self.num_experts, naive=False,
        )
        _derive_dg_metadata_kernel[p["derive_grid"]](
            sorted_ids, expert_ids, m_indices, dest,
            p["numel"], p["max_padded"], M_sum,
            LOG2_BLOCK=7, BLOCK=_DERIVE_BLOCK, num_warps=4,
        )

        a_perm = p["a_perm"]
        a_scale = p["a_scale"]
        _permute_quant_scatter_kernel[p["permute_grid"]](
            x, dest, a_perm, a_scale, K, K // _FP8_GROUP,
            FP8_MAX=_FP8_MAX, EPS=_QUANT_EPS, TOP_K=self.top_k,
            GROUPS=p["permute_groups"], GSIZE=_FP8_GROUP,
            num_warps=p["permute_warps"],
        )

        mm1_out = p["mm1_out"]
        self._dg_gemm(a_perm, a_scale, self.w13, self._sfb(w13_scale, self.w13),
                      mm1_out, m_indices)

        # The one steady-state allocation on this path: SiluMulQuantFp8 always
        # allocates its column-major (N/128, M_sum) float32 scale tensor and takes
        # no preallocated buffer for it. Removing it means owning the fused
        # act+quant, which is deferred.
        a2_fp8, a2_scale = self.silu_mul_quant_fp8(mm1_out, output=p["quant_out"])

        mm2_out = p["mm2_out"]
        self._dg_gemm(a2_fp8, a2_scale, self.w2, self._sfb(w2_scale, self.w2),
                      mm2_out, m_indices)

        out = torch.empty(M, K, dtype=x.dtype, device=x.device)
        _unpermute_weighted_sum_kernel[p["unpermute_grid"]](
            mm2_out, dest, topk_weights, out, K,
            TOP_K=self.top_k, BLOCK_K=_UNPERMUTE_BLOCK_K,
            EVEN_K=p["unpermute_even_k"],
            num_warps=_UNPERMUTE_WARPS,
        )
        return out

    # -- Triton path ------------------------------------------------------
    def _triton_plan(self, M, K, N2, device, dtype):
        """Everything host-side about a Triton-path call at this ``M``, once.

        The decode shape is host-bound -- 62 us of kernel time inside a 118 us
        forward -- so the number of Python-level operations between launches is
        itself a performance parameter. Resolved here and cached: the config
        (``get_triton_config`` otherwise rebuilds a dict from a JSON table on every
        call), the aligner's shortcut decision, and every scratch view.

        The config dict is read and never written -- both GEMMs get the same one
        and the aligner's block size comes from it -- so handing out the cached
        object is safe.
        """
        key = (M, K, N2, dtype, device)
        plan = self._triton_plans.get(key)
        if plan is not None and plan["generation"] == self._triton_scratch.generation:
            return plan

        top_k = self.top_k
        N = N2 // 2
        rows = M * top_k
        cfg = self._triton_configs.get((M, N2, K))
        if cfg is None:
            cfg = get_triton_config(
                M, (self.num_experts, N2, K),
                (self.num_experts, K, self.intermediate_per_tp),
                top_k, use_fp8=self.use_fp8, block_shape=self.block_shape,
                default_style="legacy",
            )
            self._triton_configs[(M, N2, K)] = cfg

        sb = self._triton_scratch
        gen0 = sb.generation
        flat = sb.cache(rows * max(N2, K), device, dtype)
        a_fp8 = a_scale = a2_fp8 = a2_scale = None
        if self.use_fp8:
            a_fp8, a_scale = sb.fp8(1, M, K, device)
            a2_fp8, a2_scale = sb.fp8(2, rows, N, device)
        if sb.generation != gen0:
            # A view taken before a later growth may alias a freed block.
            return self._triton_plan(M, K, N2, device, dtype)

        groups, warps = _permute_geometry(N)
        plan = {
            "cfg": cfg,
            "N": N,
            # The fusion needs FP8 (bf16 mode has no second quantizer) and a group
            # count that divides N / 128 -- _permute_geometry guarantees the latter,
            # falling back to 1.
            "fuse_act_quant": bool(_FUSE_ACT_QUANT and self.use_fp8
                                   and N % _FP8_GROUP == 0),
            "act_quant_grid": (rows, (N // _FP8_GROUP) // groups),
            "act_quant_groups": groups,
            "act_quant_warps": warps,
            "block_m": cfg["BLOCK_SIZE_M"],
            # The reference's rule is ``M * top_k * 4 <= num_experts``, which is
            # True at M = 1. Taking that shortcut leaves ``sorted_token_ids`` at
            # None, and the frozen grouped GEMM's thin-decode path requires it --
            # so the shortcut forfeits the fast kernel and lands on the reference
            # one instead. _ALIGNED_DECODE keeps the sort; see the measurement in
            # tools/measure_decode.py, which is what decides it.
            "use_naive": (not _ALIGNED_DECODE
                          and M * top_k * _SPARSITY_FACTOR <= self.num_experts),
            "rows": rows,
            "inter1": flat[:rows * N2].view(rows, N2),
            "inter3": flat[:rows * K].view(rows, K),
            "a_fp8": a_fp8, "a_scale": a_scale,
            "a2_fp8": a2_fp8, "a2_scale": a2_scale,
            "generation": sb.generation,
        }
        self._triton_plans[key] = plan
        return plan

    def _forward_triton(self, x, topk_weights, topk_ids, M, K, N2,
                        w13_scale, w2_scale):
        p = self._triton_plan(M, K, N2, x.device, x.dtype)
        cfg = p["cfg"]
        inter1, inter3 = p["inter1"], p["inter3"]

        sorted_ids, expert_ids, num_post_pad = self.moe_align(
            topk_ids, p["block_m"], self.num_experts, naive=p["use_naive"],
        )

        if self.use_fp8:
            gemm1_in, gemm1_scale = p["a_fp8"], p["a_scale"]
            self.per_token_group_quant_fp8(x, gemm1_in, gemm1_scale)
        else:
            gemm1_in, gemm1_scale = x, None

        self.moe_grouped_gemm(
            gemm1_in, self.w13, inter1, topk_weights, sorted_ids, expert_ids,
            num_post_pad, mul_routed_weight=False, top_k=self.top_k, config=cfg,
            a_scale=gemm1_scale, b_scale=w13_scale,
            use_fp8_w8a8=self.use_fp8, block_shape=self.block_shape,
        )

        if p["fuse_act_quant"]:
            # One launch instead of SiluAndMul followed by PerTokenGroupQuantFp8.
            gemm2_in, gemm2_scale = p["a2_fp8"], p["a2_scale"]
            _silu_mul_quant_kernel[p["act_quant_grid"]](
                inter1, gemm2_in, gemm2_scale, 2 * p["N"], p["N"],
                p["N"] // _FP8_GROUP,
                FP8_MAX=_FP8_MAX, EPS=_QUANT_EPS,
                GROUPS=p["act_quant_groups"], GSIZE=_FP8_GROUP,
                num_warps=p["act_quant_warps"],
            )
        elif self.use_fp8:
            inter2 = self.act_fn(inter1)
            gemm2_in, gemm2_scale = p["a2_fp8"], p["a2_scale"]
            self.per_token_group_quant_fp8(inter2, gemm2_in, gemm2_scale)
        else:
            gemm2_in, gemm2_scale = self.act_fn(inter1), None

        self.moe_grouped_gemm(
            gemm2_in, self.w2, inter3, topk_weights, sorted_ids, expert_ids,
            num_post_pad, mul_routed_weight=True, top_k=1, config=cfg,
            a_scale=gemm2_scale, b_scale=w2_scale,
            use_fp8_w8a8=self.use_fp8, block_shape=self.block_shape,
        )

        return self.moe_sum(inter3, self.top_k)

    # -- entry point ------------------------------------------------------
    def _use_deep_gemm(self, x, M) -> bool:
        """The reference's path predicate, with its static half precomputed.

        Ordered so that M = 1 answers on the first comparison: the capture query
        is a CUDA API call and this is a 184 us forward. Equivalence with
        ``_valid_deep_gemm`` over the whole M range is asserted in
        ``tools/check_paths.py``.
        """
        return (M >= self._dg_align
                and self._dg_static_ok
                and x.is_contiguous()
                and not torch.cuda.is_current_stream_capturing())

    def forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        x = hidden_states.view(-1, self.hidden_size)
        M, K = x.shape

        router_logits = self.gate(x)
        topk_weights, topk_ids = self.topk_softmax(
            router_logits, self.top_k, renormalize=self.renormalize,
        )

        N2 = self.w13.size(1)
        if self._use_deep_gemm(x, M):
            w13_scale_dg = getattr(self, "w13_scale_dg", None)
            w2_scale_dg = getattr(self, "w2_scale_dg", None)
            out = self._forward_deep_gemm(
                x, topk_weights, topk_ids, M, K, N2,
                w13_scale_dg if w13_scale_dg is not None else self.w13_scale,
                w2_scale_dg if w2_scale_dg is not None else self.w2_scale,
            )
        else:
            out = self._forward_triton(
                x, topk_weights, topk_ids, M, K, N2,
                self.w13_scale, self.w2_scale,
            )

        if self.tp_size > 1 and not self._use_custom_op:
            out = self.allreduce(out)

        return out.view(orig_shape)
