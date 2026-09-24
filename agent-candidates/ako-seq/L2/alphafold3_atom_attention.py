"""Sequence-local atom attention for AlphaFold3.

AtomAttentionEncoder (Algorithm 5) and AtomAttentionDecoder (Algorithm 6).

Reference: openfold3/core/model/layers/sequence_local_atom_attention.py

At the captured shapes (N_atom=368, n_query=32, n_key=128 -> 12 blocks,
c_atom=128, c_atom_pair=16, N_token=16) every tensor in the encoder's
pre-transformer pipeline is tiny: the reference path spends 2.2 ms of the
encoder's 7.9 ms there, and a stage-by-stage measurement shows CPU time equal
to GPU time to within 5% -- it is a launch/dispatch storm, not math or
bandwidth.  So the encoder here is rebuilt around *launch count*:

* **Three kernels before the transformer, five for it, two after** (from
  ~1600 reference ops).  ``_k_proj`` does both trunk projections, ``_k_cl``
  builds the reference-feature GEMM and the pair-projection of ``cl``,
  ``_k_plm`` builds the whole ``[12, 32, 128, 16]`` pair tensor including the
  3-layer MLP; ``_k_sad``/``_k_zb`` and ``_k_xa``/``_k_xb`` are the transformer
  stack (below); ``_k_post`` and ``_k_ai`` finish the token aggregation.
* **No concatenated activations.**  The five bias-free ``RefAtomFeatureEmbedder``
  projections become one fp32 accumulation over three ``tl.dot`` segments
  (pos/charge/mask padded to 16, element padded to 128, name-chars 256) whose
  *weights* are pre-transposed and concatenated once and cached; the activation
  side is assembled in registers, so the 380-wide input is never materialized.
  ``linear_r(rl)`` is a fourth segment added into the same accumulator.
* **Block geometry computed, not gathered.**  ``_get_block_key_indices`` runs
  ~5 times per reference forward over the same mask.  Every fused kernel
  instead re-derives the window from ``atom_mask`` inline (a 512-lane sum plus
  a dozen scalar ops).  That includes reproducing the reference's *bf16
  index arithmetic*: ``initial + total_shift`` promotes an int32 index tensor
  to bf16, so key indices above 256 round to even and blocks 6..11 gather
  duplicated atoms.  ``_geom`` emulates it exactly -- verified against
  ``_get_block_key_indices`` for N in {16..4096} and for partially-zero masks --
  because getting it "right" instead of identical gathers different atoms.
* **relu + linear_l/linear_m hoisted to the unblocked ``cl``**, so the pair
  term is one ``[368, 128] x [128, 32]`` GEMM in ``_k_cl`` instead of the
  reference's 1536-row key-block duplication.  Sound because both terms are
  multiplied by the block mask, which is zero exactly where the blockified
  copy would have been zero-padded or invalid.
* **Kernels launched through their own C entry point.**  ``kernel[grid](...)``
  re-binds and re-specializes every argument per call: measured 8.8 us here
  against 3.6 us for the compiled kernel's raw launcher.  Every argument
  except the pointers is constexpr, so the binder's work is loop-invariant;
  ``_Launch`` compiles once at plan time and keeps the pieces (same technique
  as ``L1/layer_norm.py``).  Five launches * 5 us of binder overhead saved is
  larger than all the arithmetic in the stage.
* **One arena allocation per forward.**  Nine intermediates as ``as_strided``
  views (0.99 us) off one ``torch.empty`` (1.6 us) rather than nine allocations
  (14 us).

The decoder gets the same treatment for its own prologue and epilogue
(``_k_dec_in`` / ``_k_dec_out``).

**The atom transformer stack runs here too.**  ``self.atom_transformer`` is
still constructed exactly as before -- same class, same kwargs, same 79
parameters, still called on the reference path and whenever ``_xf_probe``
declines -- but on the fast path its arithmetic is five more kernels instead of
~1500 dispatches.  It was 5.45 ms of the encoder's 5.61 ms and 5.45 of the
decoder's 5.41, and CPU-bound in exactly the same way the pre-transformer path
was.  Three structural facts make it cheap:

* Atoms partition into query blocks with no overlap and every key comes from
  that block's own window of the block's *input* ``a``, so one program finishes
  32 atoms of one transformer block (``_k_xa`` / ``_k_xb``).
* Everything derived from ``s`` is row-wise and ``a``-independent: nine AdaLN
  instances plus two output gates per block become ``8 * no_blocks`` per-atom
  tables in one kernel (``_k_sad``), which the block kernels only gather rows
  from -- the 1536-row key-block duplication never happens.
* ``layer_norm_z`` belongs to the stack, not to a block, and each block's
  ``linear_z`` is c_z -> no_heads, so all blocks' pair biases are one
  ``[c_z, no_blocks*no_heads]`` matmul (``_k_zb``).

What sets the runtime of each of those kernels is how many SMs it can fill, not
its FLOPs: the whole stack is ~500 MMAC.  So ``_k_sad`` is one dot per program
(288 programs, not 6 -- 68 us -> 6 us), the attention runs per (block, head)
(48 programs, not 12) and everything after it is row-wise (23 programs).

Numerics are the real risk, not speed: three sequential blocks compound bf16
error into ``ql`` (a compared output) and then into ``ai`` and the decoder
result.  So the fused path rounds to bf16 at exactly the points torch does
rather than keeping fp32 (see ``_rb``), and reproduces the reference's
primitives rather than approximating them -- ``F.layer_norm``'s two-pass moments
(``_lnf``), ``torch.sigmoid`` (``_sig``), ``F.silu`` (``_silu``), ``std::exp``
in the softmax (``_expf``) and a single K=no_heads*c_hidden ``linear_o``.  Those
are the *baseline* L1 ops, not the candidate L1 winners this file imports for
its own fallback: a baseline module's relative import resolves inside the
baseline tree.  Matching them makes the difference between failing at large
activation scales and not -- see ``dev/t_scale.py`` and ITERATIONS.md.

Numerics: the fused path accumulates each projection in fp32 and rounds to
bf16 at the same points the reference does (after every ``Linear``, after
every ``+``), so it is never *less* accurate than the reference; the residual
difference is the reference's own intermediate bf16 rounding.  The transformer
(``L3.DiffusionTransformer``, shared with the decoder) is untouched.

The one place the fused path is deliberately *not* bit-comparable is ``ai``: the
reference's ``scatter_add_`` accumulates ~23 bf16 terms per token with atomics,
so it disagrees with itself by ~1% relative and, at large activation
magnitudes, scores 0.983 against its own rerun -- below the benchmark's 0.99
bar.  Accumulating in fp32 is the closest a deterministic kernel can sit to that
distribution (0.986 in the same measurement), so that is what ``_k_ai`` does.

Anything the fast path does not cover -- a non-bf16 dtype, a batch that does
not collapse to one row, a missing ``atom_to_token_index``, grad enabled, a
misaligned input pointer, or a Triton that refuses launcher memoization --
falls back to the reference implementation below, which is kept intact.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._C import _cuda_getCurrentRawStream as _raw_stream
from torch._C import _cuda_getDevice as _cur_device

try:                                    # bit-exact fp32 exp (see _expf)
    from triton.language.extra import libdevice as _libdev
except Exception:                       # noqa: BLE001 - binding moved; _expf falls back
    _libdev = None

from ..L1.relu import ReLU
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.tensor_ops import Pad


def _get_block_key_indices(
    atom_mask: torch.Tensor, n_query: int, n_key: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized computation of key-block gather indices.

    Returns:
        safe_indices: [*, N_blocks, n_key] clamped indices
        invalid_mask: [*, N_blocks, n_key] True where index is out of range
    """
    batch_dims = atom_mask.shape[:-1]
    n_atom = atom_mask.shape[-1]
    num_blocks = math.ceil(n_atom / n_query)
    device = atom_mask.device
    offset = n_query // 2

    subset_centers = offset + torch.arange(num_blocks, device=device) * n_query
    subset_centers = subset_centers.reshape(*(1,) * len(batch_dims), num_blocks)
    subset_centers = subset_centers.expand(*batch_dims, num_blocks)

    n_real = atom_mask.sum(dim=-1, keepdim=True).expand(*batch_dims, num_blocks)

    initial = (
        subset_centers.unsqueeze(-1)
        + torch.arange(-n_key // 2, n_key // 2, device=device)
    ).int()

    underflow = torch.relu(-initial[..., 0])
    overflow = torch.relu(initial[..., -1] - (n_real - 1))
    total_shift = torch.where(underflow > 0, underflow, -overflow)
    final = initial + total_shift.unsqueeze(-1)

    n_real_exp = n_real.unsqueeze(-1)
    invalid = (final < 0) | (final >= n_real_exp)
    safe = torch.clamp(final, torch.zeros_like(n_real_exp), (n_real_exp - 1).clamp(min=0))

    return safe.long(), invalid


def _convert_single_rep_to_blocks(
    ql: torch.Tensor,
    n_query: int,
    n_key: int,
    atom_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Convert flat atom representation to windowed block format (vectorized).

    Args:
        ql: [*, N_atom, C] atom features
        n_query: block height
        n_key: block width
        atom_mask: [*, N_atom] mask

    Returns:
        ql_query: [*, N_blocks, n_query, C]
        ql_key:   [*, N_blocks, n_key, C]
        mask_blocks: [*, N_blocks, n_query, n_key] or None
    """
    batch_dims = ql.shape[:-2]
    n_atom, c = ql.shape[-2], ql.shape[-1]
    num_blocks = math.ceil(n_atom / n_query)
    pad_q = (-n_atom) % n_query

    if pad_q > 0:
        ql = Pad()(ql, (0, 0, 0, pad_q))
        if atom_mask is not None:
            atom_mask = Pad()(atom_mask, (0, pad_q))

    ql_query = ql.reshape(*batch_dims, num_blocks, n_query, c)

    if atom_mask is None:
        atom_mask = ql.new_ones(*batch_dims, n_atom + pad_q)

    atom_mask = atom_mask.expand(*batch_dims, -1)
    key_indices, invalid_mask = _get_block_key_indices(atom_mask, n_query, n_key)

    flat_batch = int(math.prod(batch_dims)) if batch_dims else 1
    ql_flat = ql.reshape(flat_batch, n_atom + pad_q, c)
    idx_flat = key_indices.reshape(flat_batch, num_blocks * n_key)
    idx_expanded = idx_flat.unsqueeze(-1).expand(-1, -1, c)

    ql_key_flat = torch.gather(ql_flat, 1, idx_expanded)
    mask_flat = invalid_mask.reshape(flat_batch, num_blocks * n_key).unsqueeze(-1).expand(-1, -1, c)
    ql_key_flat.masked_fill_(mask_flat, 0.0)
    ql_key = ql_key_flat.reshape(*batch_dims, num_blocks, n_key, c)

    mask_q = atom_mask.reshape(*batch_dims, num_blocks, n_query)
    mask_k_valid = (~invalid_mask).to(atom_mask.dtype)
    atom_mask_at_keys = torch.gather(
        atom_mask.reshape(flat_batch, -1), 1,
        idx_flat,
    ).reshape(*batch_dims, num_blocks, n_key)
    mask_k_valid = mask_k_valid * atom_mask_at_keys
    mask_blocks = mask_q.unsqueeze(-1) * mask_k_valid.unsqueeze(-2)

    return ql_query, ql_key, mask_blocks


_apply_block_indices = _convert_single_rep_to_blocks


def _convert_pair_rep_to_blocks(
    batch: dict,
    zij_trunk: torch.Tensor,
    n_query: int,
    n_key: int,
) -> torch.Tensor:
    """Convert pair representation to block format for atom attention (vectorized).

    Args:
        batch: needs atom_mask, atom_to_token_index
        zij_trunk: [*, N_token, N_token, C_z]
        n_query: block height
        n_key: block width

    Returns:
        [*, N_blocks, n_query, n_key, C_z]
    """
    atom_mask = batch["atom_mask"]
    n_atoms = atom_mask.shape[-1]
    batch_dims = zij_trunk.shape[:-3]
    c_z = zij_trunk.shape[-1]

    if "atom_to_token_index" in batch:
        atom_to_token = batch["atom_to_token_index"]
        if atom_to_token.dim() > 1:
            atom_to_token = atom_to_token[0]
    else:
        n_token = zij_trunk.shape[-2]
        atom_to_token = torch.arange(n_token, device=zij_trunk.device)
        if n_atoms > n_token:
            atom_to_token = atom_to_token.repeat_interleave(
                (n_atoms + n_token - 1) // n_token
            )[:n_atoms]

    num_blocks = math.ceil(n_atoms / n_query)
    pad_q = (-n_atoms) % n_query

    atk_padded = Pad()(atom_to_token, (0, pad_q))
    q_indices = atk_padded.reshape(num_blocks, n_query)

    atom_mask_exp = atom_mask.expand(*batch_dims, -1)
    key_indices, invalid_mask = _get_block_key_indices(atom_mask_exp, n_query, n_key)

    flat_batch = int(math.prod(batch_dims)) if batch_dims else 1

    atk_flat = atom_to_token.expand(flat_batch, -1)
    key_idx_flat = key_indices.reshape(flat_batch, num_blocks * n_key)
    k_token_flat = torch.gather(atk_flat, 1, key_idx_flat.clamp(min=0, max=n_atoms - 1))
    k_indices = k_token_flat.reshape(flat_batch, num_blocks, n_key)

    zij_flat = zij_trunk.reshape(flat_batch, *zij_trunk.shape[-3:])
    batch_idx = torch.arange(flat_batch, device=zij_trunk.device).view(-1, 1, 1, 1)
    q_idx = q_indices.long().unsqueeze(0).expand(flat_batch, -1, -1)

    plm = zij_flat[batch_idx, q_idx.unsqueeze(-1), k_indices.unsqueeze(-2)]

    inv_expanded = invalid_mask.reshape(flat_batch, num_blocks, n_key)
    plm.masked_fill_(inv_expanded[:, :, None, :, None].expand_as(plm), 0.0)

    pair_mask = _get_pair_atom_block_mask(
        atom_mask=atom_mask_exp, num_blocks=num_blocks,
        n_query=n_query, n_key=n_key, pad_q=pad_q,
        key_indices=key_indices, invalid_mask=invalid_mask,
    )
    plm = plm * pair_mask.reshape(flat_batch, num_blocks, n_query, n_key, 1)
    plm = plm.reshape(*batch_dims, num_blocks, n_query, n_key, c_z)

    return plm


def _get_pair_atom_block_mask(
    atom_mask: torch.Tensor,
    num_blocks: int,
    n_query: int,
    n_key: int,
    pad_q: int,
    key_indices: torch.Tensor,
    invalid_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute pair atom block mask."""
    batch_dims = atom_mask.shape[:-1]
    flat_batch = int(math.prod(batch_dims)) if batch_dims else 1
    mask_flat = atom_mask.reshape(flat_batch, -1)

    mask_padded = Pad()(mask_flat, (0, pad_q))
    mask_q = mask_padded.reshape(flat_batch, num_blocks, n_query)

    idx_flat = key_indices.reshape(flat_batch, num_blocks * n_key)
    mask_k_vals = torch.gather(mask_flat, 1, idx_flat.clamp(min=0, max=mask_flat.shape[-1] - 1))
    mask_k = mask_k_vals.reshape(flat_batch, num_blocks, n_key)
    inv_flat = invalid_mask.reshape(flat_batch, num_blocks, n_key)
    mask_k = mask_k * (~inv_flat).to(mask_k.dtype)

    pair_mask = mask_q.unsqueeze(-1) * mask_k.unsqueeze(-2)
    return pair_mask.reshape(*batch_dims, num_blocks, n_query, n_key)


def _broadcast_token_feat_to_atoms(
    token_mask: torch.Tensor,
    num_atoms_per_token: torch.Tensor | None,
    token_feat: torch.Tensor,
    atom_to_token_index: torch.Tensor | None = None,
    n_atoms: int | None = None,
) -> torch.Tensor:
    """Broadcast token-level features to atom-level.

    Args:
        token_mask: [*, N_token]
        num_atoms_per_token: [*, N_token] or None
        token_feat: [*, N_token, C]
        atom_to_token_index: [*, N_atom] optional direct mapping
        n_atoms: total number of atoms if atom_to_token_index not provided

    Returns:
        [*, N_atom, C]
    """
    if atom_to_token_index is not None:
        idx = atom_to_token_index.long()
        while idx.dim() < token_feat.dim() - 1:
            idx = idx.unsqueeze(1)
        idx = idx.expand(*token_feat.shape[:-2], idx.shape[-1])
        return torch.gather(
            token_feat, -2,
            idx.unsqueeze(-1).expand(*idx.shape, token_feat.shape[-1]),
        )

    if num_atoms_per_token is not None:
        return torch.repeat_interleave(
            token_feat, num_atoms_per_token.long(), dim=-2,
        )

    return token_feat


def _aggregate_atom_feat_to_tokens(
    token_mask: torch.Tensor,
    atom_to_token_index: torch.Tensor,
    atom_mask: torch.Tensor,
    atom_feat: torch.Tensor,
    mode: str = "mean",
) -> torch.Tensor:
    """Aggregate atom-level features to token-level.

    Args:
        token_mask: [*, N_token]
        atom_to_token_index: [N_atom]
        atom_mask: [*, N_atom]
        atom_feat: [*, N_atom, C]
        mode: "mean" or "sum"

    Returns:
        [*, N_token, C]
    """
    n_token = token_mask.shape[-1]
    c = atom_feat.shape[-1]
    batch_shape = atom_feat.shape[:-2]

    atom_mask_expanded = atom_mask.expand(*batch_shape, -1)

    result = atom_feat.new_zeros(*batch_shape, n_token, c)
    masked_feat = atom_feat * atom_mask_expanded[..., None]

    idx = atom_to_token_index.long().expand(*batch_shape, -1)
    result.scatter_add_(-2, idx.unsqueeze(-1).expand_as(masked_feat), masked_feat)

    if mode == "mean":
        counts = torch.zeros(*batch_shape, n_token, dtype=result.dtype, device=result.device)
        counts.scatter_add_(-1, idx, atom_mask_expanded.to(dtype=result.dtype))
        counts = counts.clamp(min=1.0)
        result = result / counts.unsqueeze(-1)

    return result


__targets__ = ["AtomAttentionEncoder", "AtomAttentionDecoder"]


# ###########################################################################
# Fused encoder path
# ###########################################################################

@triton.jit
def _asinh(x):
    """``torch.arcsinh`` in fp32.  Written out rather than taken from Triton's
    ``extra`` libdevice bindings, which have moved between releases; this is
    bit-identical to ``torch.asinh`` once rounded to bf16 (which is what the
    caller does next), checked over 4094 values spanning 1e-8 .. 1e20, both
    signs, and +-inf.

    Three regimes: a Taylor term below 0.03 (where ``log`` of a value near 1
    loses the answer's leading digits), ``log(2a)`` above 1e15 (where ``a*a``
    overflows fp32), and the closed form between.
    """
    a = tl.abs(x)
    y = tl.where(a < 0.03, a * (1.0 - a * a * 0.16666666666666666),
                 tl.where(a > 1e15, tl.log(a) + 0.6931471805599453,
                          tl.log(a + tl.sqrt(a * a + 1.0))))
    return tl.where(x < 0.0, -y, y)


if _libdev is not None:
    @triton.jit
    def _expf(x):
        """``exp`` in fp32, bit-identical to torch's.

        ``tl.exp`` is the ``ex2.approx`` form: up to 3.6e-6 relative error,
        which differs from ``std::exp`` (what ``F.softmax`` uses for its fp32
        accumulator) on 40-85% of inputs.  Softmax probabilities are rounded to
        bf16 before the PV matmul, so most of that vanishes -- but not all of
        it, and at large activation scales the softmax is peaked enough that a
        one-ulp probability is a percent of the block's output.  ``libdevice``'s
        is exact against torch over 1e6 values at every scale tested.
        """
        return _libdev.exp(x)
else:
    @triton.jit
    def _expf(x):
        return tl.exp(x)


@triton.jit
def _rb(x):
    """Round a value to bf16 and keep computing in fp32.

    Marks one of the reference's rounding points.  Every torch op on bf16
    tensors computes in fp32 and rounds its *result* back to bf16, so a chain of
    N adds rounds N times; accumulating the same chain in fp32 and rounding once
    is more accurate but *different*, and where the chain ends in cancellation
    (``plm + pair_mlp(plm)``, ``relu``) the difference is one ulp of the largest
    intermediate rather than of the result -- which the benchmark's
    ``atol + rtol*|ref|`` bound only tolerates while activations stay small.
    Matching the rounding points instead makes the fused path scale-invariant.
    """
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _n_real(AMSK, N: tl.constexpr, BN: tl.constexpr):
    """``atom_mask.sum(-1)`` as the reference computes it: fp32 accumulation,
    rounded to the mask's bf16 dtype.  Padding lanes are zero in the reference's
    padded mask, so summing the unpadded mask is the same number."""
    i = tl.arange(0, BN)
    s = tl.sum(tl.load(AMSK + i, mask=i < N, other=0.0).to(tl.float32))
    return s.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _geom(nreal, b, NQ: tl.constexpr, NK: tl.constexpr):
    """Key window of block ``b``: ``(safe_index[NK], invalid[NK])``.

    Reproduces ``_get_block_key_indices`` bit-for-bit, including its accidental
    bf16 index arithmetic: ``initial`` is int32 and ``total_shift`` is bf16, so
    ``initial + total_shift`` and the clamp bounds are all evaluated in bf16.
    Above 256 that quantizes indices to even numbers, which is why blocks 6..11
    gather duplicated atoms at the captured N_atom=368.
    """
    nm1 = (nreal - 1.0).to(tl.bfloat16).to(tl.float32)
    init0 = NQ // 2 + b * NQ - NK // 2
    under = tl.maximum(-init0, 0).to(tl.float32)
    over = ((init0 + NK - 1).to(tl.bfloat16).to(tl.float32) - nm1)
    over = tl.maximum(over.to(tl.bfloat16).to(tl.float32), 0.0)
    shift = tl.where(under > 0.0, under, -over).to(tl.bfloat16).to(tl.float32)
    fin = (init0 + tl.arange(0, NK)).to(tl.bfloat16).to(tl.float32)
    fin = (fin + shift).to(tl.bfloat16).to(tl.float32)
    invalid = (fin < 0.0) | (fin >= nreal)
    safe = tl.minimum(tl.maximum(fin, 0.0), tl.maximum(nm1, 0.0)).to(tl.int32)
    return safe, invalid


@triton.jit
def _ln_dot(X, G, W, r, rm, K: tl.constexpr, NOUT: tl.constexpr,
            BR: tl.constexpr, BK: tl.constexpr, NCHUNK: tl.constexpr,
            BO: tl.constexpr, EPS: tl.constexpr):
    """``Linear(LayerNorm(X))`` for rows ``r``: fp32 row stats, bf16 store of the
    normalized row (what ``LayerNorm(promote_fp32=True)`` produces), then one
    fp32-accumulating dot against a pre-transposed ``[K, NOUT]`` weight.

    The row is read three times -- mean, variance, then normalize-and-dot --
    because a ``K``-wide register tile cannot be sliced per dot chunk.  All
    three reads hit L1; the row is at most 384 wide, and two passes over it are
    what reproduces ``F.layer_norm`` bit for bit (see ``_lnf``).
    """
    s = tl.zeros((BR,), tl.float32)
    for ci in tl.static_range(0, NCHUNK):
        kk = ci * BK + tl.arange(0, BK)
        km = kk < K
        x = tl.load(X + r[:, None] * K + kk[None, :],
                    mask=rm[:, None] & km[None, :], other=0.0).to(tl.float32)
        s += tl.sum(tl.where(km[None, :], x, 0.0), 1)
    mean = s / K
    s2 = tl.zeros((BR,), tl.float32)
    for ci in tl.static_range(0, NCHUNK):
        kk = ci * BK + tl.arange(0, BK)
        km = kk < K
        x = tl.load(X + r[:, None] * K + kk[None, :],
                    mask=rm[:, None] & km[None, :], other=0.0).to(tl.float32)
        d = tl.where(km[None, :], x - mean[:, None], 0.0)
        s2 += tl.sum(d * d, 1)
    rstd = tl.rsqrt(s2 / K + EPS)

    o = tl.arange(0, BO)
    om = o < NOUT
    acc = tl.zeros((BR, BO), tl.float32)
    for ci in tl.static_range(0, NCHUNK):
        kk = ci * BK + tl.arange(0, BK)
        km = kk < K
        x = tl.load(X + r[:, None] * K + kk[None, :],
                    mask=rm[:, None] & km[None, :], other=0.0).to(tl.float32)
        g = tl.load(G + kk, mask=km, other=0.0).to(tl.float32)
        y = ((x - mean[:, None]) * rstd[:, None] * g[None, :]).to(tl.bfloat16)
        w = tl.load(W + kk[:, None] * NOUT + o[None, :],
                    mask=km[:, None] & om[None, :], other=0.0)
        acc = tl.dot(y, w, acc)
    return acc, o, om


@triton.jit
def _k_proj(SI, ZIJ, SIP, ZP, WS, WZ, GS, GZ,
            T: tl.constexpr, CS: tl.constexpr, CZ: tl.constexpr,
            C: tl.constexpr, P: tl.constexpr,
            BT: tl.constexpr, BZ: tl.constexpr,
            BKS: tl.constexpr, NKS: tl.constexpr,
            BKZ: tl.constexpr, NKZ: tl.constexpr,
            BC: tl.constexpr, BP: tl.constexpr,
            EPS_S: tl.constexpr, EPS_Z: tl.constexpr):
    """Both trunk projections in one launch.

    Program 0 does ``linear_s(layer_norm_s(si_trunk))`` -> ``[T, c_atom]``;
    programs 1.. do ``linear_z(layer_norm_z(zij_trunk))`` -> ``[T*T, c_atom_pair]``
    in row strips of ``BZ``.  They are independent, so folding them into one
    grid costs nothing and saves a launch.
    """
    pid = tl.program_id(0)
    if pid == 0:
        r = tl.arange(0, BT)
        rm = r < T
        acc, o, om = _ln_dot(SI, GS, WS, r, rm, CS, C, BT, BKS, NKS, BC, EPS_S)
        tl.store(SIP + r[:, None] * C + o[None, :], acc.to(tl.bfloat16),
                 mask=rm[:, None] & om[None, :])
    else:
        # distinct names: Triton unifies same-named values across both arms of
        # an `if`, and the two projections have different tile shapes.
        rz = (pid - 1) * BZ + tl.arange(0, BZ)
        rmz = rz < T * T
        accz, oz, omz = _ln_dot(ZIJ, GZ, WZ, rz, rmz, CZ, P, BZ, BKZ, NKZ, BP, EPS_Z)
        tl.store(ZP + rz[:, None] * P + oz[None, :], accz.to(tl.bfloat16),
                 mask=rmz[:, None] & omz[None, :])


@triton.jit
def _k_cl(RPOS, RCHG, RMSK, RELEM, RCHAR, RL, SIP, A2T, CL, QL, CLP,
          WA, WB, WC, WR, WLM,
          N: tl.constexpr, C: tl.constexpr, P: tl.constexpr, T: tl.constexpr,
          KE: tl.constexpr, KN: tl.constexpr,
          BM: tl.constexpr, BC: tl.constexpr, BP2: tl.constexpr,
          BK: tl.constexpr, NKE: tl.constexpr, NKN: tl.constexpr,
          HAS_RL: tl.constexpr, HAS_SI: tl.constexpr):
    """``cl``, ``ql`` and the pair projection of ``cl``, for one strip of atoms.

    The five reference projections are three ``tl.dot`` segments sharing one
    fp32 accumulator, with the activation side assembled in registers so the
    380-wide concatenation never reaches memory: pos(3) / asinh(charge)(1) /
    mask(1) padded to a 16-deep segment, element(119) padded to 128, and
    name-chars(256).  ``linear_r(rl)`` is a fourth segment kept separate only
    because ``cl`` is returned without it.

    A strip owns *all* ``c_atom`` columns of its rows, so ``relu(cl) @
    [linear_l | linear_m]`` runs in the same launch off registers rather than
    re-reading a blockified 1536-row copy.
    """
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    rm = rows < N
    n = tl.arange(0, BC)
    nm = n < C
    k16 = tl.arange(0, 16)

    # --- ref_pos ------------------------------------------------------------
    # Five separate ``Linear``s, so five separate roundings: the projections
    # cannot share one fp32 accumulator (see ``_rb``).  ref_charge/ref_mask are
    # 1-wide, so their "GEMM" is an outer product.
    pos = tl.load(RPOS + rows[:, None] * 3 + k16[None, :],
                  mask=rm[:, None] & (k16 < 3)[None, :], other=0.0)
    wa = tl.load(WA + k16[:, None] * C + n[None, :],
                 mask=(k16 < 3)[:, None] & nm[None, :], other=0.0)
    acc = _rb(tl.dot(pos, wa))

    # --- arcsinh(ref_charge), ref_mask --------------------------------------
    chg = tl.load(RCHG + rows, mask=rm, other=0.0).to(tl.float32)
    chg = _rb(_asinh(chg))
    wchg = tl.load(WA + 3 * C + n, mask=nm, other=0.0).to(tl.float32)
    acc = _rb(acc + _rb(chg[:, None] * wchg[None, :]))
    msk = tl.load(RMSK + rows, mask=rm, other=0.0).to(tl.float32)
    wmsk = tl.load(WA + 4 * C + n, mask=nm, other=0.0).to(tl.float32)
    acc = _rb(acc + _rb(msk[:, None] * wmsk[None, :]))

    # --- ref_element --------------------------------------------------------
    e = tl.zeros((BM, BC), tl.float32)
    for ci in tl.static_range(0, NKE):
        kk = ci * BK + tl.arange(0, BK)
        km = kk < KE
        x = tl.load(RELEM + rows[:, None] * KE + kk[None, :],
                    mask=rm[:, None] & km[None, :], other=0.0)
        w = tl.load(WB + kk[:, None] * C + n[None, :],
                    mask=km[:, None] & nm[None, :], other=0.0)
        e = tl.dot(x, w, e)
    acc = _rb(acc + _rb(e))

    # --- ref_atom_name_chars ------------------------------------------------
    f = tl.zeros((BM, BC), tl.float32)
    for ci in tl.static_range(0, NKN):
        kk = ci * BK + tl.arange(0, BK)
        km = kk < KN
        x = tl.load(RCHAR + rows[:, None] * KN + kk[None, :],
                    mask=rm[:, None] & km[None, :], other=0.0)
        w = tl.load(WC + kk[:, None] * C + n[None, :],
                    mask=km[:, None] & nm[None, :], other=0.0)
        f = tl.dot(x, w, f)
    acc = _rb(acc + _rb(f))

    if HAS_SI:
        t = tl.load(A2T + rows, mask=rm, other=0).to(tl.int32)
        t = tl.minimum(tl.maximum(t, 0), T - 1)
        acc = _rb(acc + tl.load(SIP + t[:, None] * C + n[None, :],
                                mask=rm[:, None] & nm[None, :],
                                other=0.0).to(tl.float32))
    clb = acc.to(tl.bfloat16)
    tl.store(CL + rows[:, None] * C + n[None, :], clb, mask=rm[:, None] & nm[None, :])

    if HAS_RL:
        rt = tl.load(RL + rows[:, None] * 3 + k16[None, :],
                     mask=rm[:, None] & (k16 < 3)[None, :], other=0.0)
        wr = tl.load(WR + k16[:, None] * C + n[None, :],
                     mask=(k16 < 3)[:, None] & nm[None, :], other=0.0)
        qlb = (acc + _rb(tl.dot(rt, wr))).to(tl.bfloat16)
    else:
        qlb = clb
    tl.store(QL + rows[:, None] * C + n[None, :], qlb, mask=rm[:, None] & nm[None, :])

    # --- relu + linear_l / linear_m on the unblocked cl ---------------------
    j = tl.arange(0, BP2)
    jm = j < 2 * P
    rcl = tl.maximum(clb.to(tl.float32), 0.0).to(tl.bfloat16)
    wlm = tl.load(WLM + n[:, None] * (2 * P) + j[None, :],
                  mask=nm[:, None] & jm[None, :], other=0.0)
    clp = tl.dot(rcl, wlm)
    tl.store(CLP + rows[:, None] * (2 * P) + j[None, :], clp.to(tl.bfloat16),
             mask=rm[:, None] & jm[None, :])


@triton.jit
def _k_plm(RPOS, RUID, AMSK, A2T, CLP, ZP, PLM, WRO, WINV, WVM, M1, M2, M3,
           N: tl.constexpr, NQ: tl.constexpr, NK: tl.constexpr,
           P: tl.constexpr, T: tl.constexpr,
           BN: tl.constexpr, BP: tl.constexpr, HAS_Z: tl.constexpr):
    """One ``[n_key, c_atom_pair]`` pair tile -- everything the reference builds
    with a dozen elementwise ops, two gathers and four ``Linear`` calls.

    One program per ``(block, query)``.  ``dlm``/``vlm``/``inv_sq_dists`` never
    leave registers (the reference materializes three ``[12, 32, 128, 3]``
    tensors), the trunk-pair gather is an address computation, and the 3-layer
    16->16 ``pair_mlp`` plus its residual and trailing mask multiply are three
    ``tl.dot``s with the weights resident.

    Everywhere the block mask is zero the reference's result is exactly zero
    (``plm_ref`` carries a ``vlm`` factor, the trunk term a ``pair_mask``
    factor, ``cl_lm`` a ``block_mask`` factor, and ``mlp(0) = 0``), so a single
    ``bm`` multiplier reproduces both the pad rows and the invalid keys.
    """
    pid = tl.program_id(0)
    b = pid // NQ
    q = pid % NQ
    qi = b * NQ + q
    qok = qi < N

    nreal = _n_real(AMSK, N, BN)
    safe, invalid = _geom(nreal, b, NQ, NK)
    kok = (safe < N) & ~invalid

    mq = tl.load(AMSK + qi, mask=qok, other=0.0).to(tl.float32)
    mk = tl.load(AMSK + safe, mask=kok, other=0.0).to(tl.float32)
    bm = mq * mk

    p0 = tl.load(RPOS + qi * 3 + 0, mask=qok, other=0.0).to(tl.float32)
    p1 = tl.load(RPOS + qi * 3 + 1, mask=qok, other=0.0).to(tl.float32)
    p2 = tl.load(RPOS + qi * 3 + 2, mask=qok, other=0.0).to(tl.float32)
    q0 = tl.load(RPOS + safe * 3 + 0, mask=kok, other=0.0).to(tl.float32)
    q1 = tl.load(RPOS + safe * 3 + 1, mask=kok, other=0.0).to(tl.float32)
    q2 = tl.load(RPOS + safe * 3 + 2, mask=kok, other=0.0).to(tl.float32)
    # ``dlm = (d_l - d_m) * atom_mask`` is bf16 throughout, and so is
    # ``inv_sq_dists``: the squares, their sum, ``1 + s`` and the reciprocal are
    # each a separate bf16 op in the reference.
    d0 = _rb(_rb(p0 - q0) * bm)
    d1 = _rb(_rb(p1 - q1) * bm)
    d2 = _rb(_rb(p2 - q2) * bm)

    uq = tl.load(RUID + qi, mask=qok, other=0.0).to(tl.float32)
    uk = tl.load(RUID + safe, mask=kok, other=0.0).to(tl.float32)
    vlm = _rb(tl.where(uq == uk, 1.0, 0.0) * bm)
    inv = _rb(1.0 / _rb(1.0 + _rb(_rb(d0 * d0) + _rb(d1 * d1) + _rb(d2 * d2))))

    j = tl.arange(0, BP)
    jm = j < P
    w0 = tl.load(WRO + 0 * P + j, mask=jm, other=0.0).to(tl.float32)
    w1 = tl.load(WRO + 1 * P + j, mask=jm, other=0.0).to(tl.float32)
    w2 = tl.load(WRO + 2 * P + j, mask=jm, other=0.0).to(tl.float32)
    wi = tl.load(WINV + j, mask=jm, other=0.0).to(tl.float32)
    wv = tl.load(WVM + j, mask=jm, other=0.0).to(tl.float32)
    p = _rb(_rb(d0[:, None] * w0[None, :] + d1[:, None] * w1[None, :]
                + d2[:, None] * w2[None, :]) * vlm[:, None])
    p = _rb(p + _rb(_rb(inv[:, None] * wi[None, :]) * vlm[:, None]))
    p = _rb(p + _rb(_rb(vlm[:, None] * wv[None, :]) * vlm[:, None]))

    if HAS_Z:
        qt = tl.load(A2T + qi, mask=qok, other=0).to(tl.int32)
        qt = tl.minimum(tl.maximum(qt, 0), T - 1)
        kc = tl.minimum(tl.maximum(safe, 0), N - 1)
        kt = tl.load(A2T + kc).to(tl.int32)
        kt = tl.minimum(tl.maximum(kt, 0), T - 1)
        z = tl.load(ZP + (qt * T + kt)[:, None] * P + j[None, :],
                    mask=jm[None, :], other=0.0).to(tl.float32)
        p = _rb(p + _rb(z * tl.where(invalid, 0.0, bm)[:, None]))

    qc = tl.minimum(tl.maximum(qi, 0), N - 1)
    kc2 = tl.minimum(tl.maximum(safe, 0), N - 1)
    cq = tl.load(CLP + qc * (2 * P) + j, mask=jm, other=0.0).to(tl.float32)
    ck = tl.load(CLP + kc2[:, None] * (2 * P) + (P + j)[None, :],
                 mask=jm[None, :], other=0.0).to(tl.float32)
    p = _rb(p + _rb(_rb(cq[None, :] + ck) * bm[:, None]))

    pb = p.to(tl.bfloat16)
    jj = tl.arange(0, BP)
    wm = jm[:, None] & jm[None, :]
    a1 = tl.load(M1 + j[:, None] * P + jj[None, :], mask=wm, other=0.0)
    a2 = tl.load(M2 + j[:, None] * P + jj[None, :], mask=wm, other=0.0)
    a3 = tl.load(M3 + j[:, None] * P + jj[None, :], mask=wm, other=0.0)
    h = tl.maximum(pb.to(tl.float32), 0.0).to(tl.bfloat16)
    h = tl.maximum(tl.dot(h, a1), 0.0).to(tl.bfloat16)
    h = tl.maximum(tl.dot(h, a2), 0.0).to(tl.bfloat16)
    out = (_rb(p + _rb(tl.dot(h, a3))) * bm[:, None]).to(tl.bfloat16)
    tl.store(PLM + (qi * NK + tl.arange(0, NK))[:, None] * P + j[None, :], out,
             mask=jm[None, :])



# ###########################################################################
# Fused atom transformer (the L3 DiffusionTransformer's cross-attention stack)
# ###########################################################################
#
# ``self.atom_transformer`` stays exactly as constructed -- same class, same
# kwargs, same 79 parameters -- and is still called on the reference path and
# whenever ``_xf_plan`` declines.  What the kernels below do is read its
# parameters and run the same arithmetic in three launches instead of ~1500.
#
# The structure that makes this cheap: atoms partition into query blocks of
# ``n_query`` with **no overlap**, and every key of block ``b`` comes from
# ``b``'s own window of the block's *input* ``a``.  So attention, the output
# projection, both AdaLN-Zero gates, the whole transition and both residual
# adds for the 32 atoms of query block ``b`` depend on nothing outside ``b``:
# one program per query block finishes those atoms for that transformer block.
# Each program re-projects the k/v of its own 128-atom window (4x redundant,
# ~4 MMAC) rather than sharing them, because a launch costs more here than the
# arithmetic does.
#
# Two whole-tensor quantities are hoisted out of the per-block grid because
# they are row-wise and ``a``-independent:
#
# * everything derived from ``s``: three AdaLN instances per transformer block
#   (query, key, transition) plus ``linear_ada_out`` and the transition's
#   ``linear_g``.  ``layer_norm_s`` is weight-only, so one normalization of
#   ``s`` feeds all nine AdaLNs; ``_k_sad`` emits the eight per-atom [N, c]
#   tables each transformer block needs (gate + additive shift for each AdaLN,
#   and the two output gates) so the block kernels only gather rows.
# * the pair bias: ``layer_norm_z`` runs once for the whole stack and each
#   block's ``linear_z`` is c_z -> no_heads, so all blocks' biases are one
#   [c_z, no_blocks*no_heads] matmul on the normalized pair tensor (``_k_zb``,
#   or folded into ``_k_plm`` on the encoder side where it is already resident).
#
# Numerics: every GEMM accumulates in fp32 and rounds to bf16 at exactly the
# points torch does (after each ``Linear``, after each residual ``+``, after
# each elementwise product), softmax probabilities are rounded to bf16 before
# the PV matmul (``scores.to(value.dtype)``), and layer norms use L1's fused
# formula in fp32.  ``dev/t_xf.py`` checks the stack sublayer by sublayer
# against the real module.


@triton.jit
def _lnf(x, n, C: tl.constexpr, EPS: tl.constexpr):
    """Affine-free ``LayerNorm`` of a register tile of rows, in fp32.

    Two passes (mean, then ``sum((x-mean)^2)``) with ``rsqrt``: that is what
    reproduces ``F.layer_norm(x.float(), ...)``, which is what the *baseline*
    ``L1.LayerNorm`` calls and therefore what every layer norm in the reference
    transformer is.  The candidate ``L1.LayerNorm``'s shifted one-pass form --
    which r1's ``_ln_dot`` copied -- agrees only to ~2e-5 of elements here;
    measured against ``F.layer_norm`` over C in {16, 128} and three scales, two
    passes match exactly at these row counts where shifted does not
    (``dev/t_lnalgo.py``).  Columns at or past ``C`` come back zero.
    """
    m = n[None, :] < C
    mu = tl.sum(tl.where(m, x, 0.0), 1) / C
    d = tl.where(m, x - mu[:, None], 0.0)
    return d * tl.rsqrt(tl.sum(d * d, 1) / C + EPS)[:, None]


@triton.jit
def _sig(x):
    """``torch.sigmoid`` in fp32.

    Every gate in the stack is this one.  The transformer submodules are built
    from ``fastkernels.tasks.baseline.L1`` -- a *baseline* module's relative
    import resolves inside the baseline tree, so the candidate L1 winners (whose
    ``Sigmoid`` is a bf16 ``0.5+0.5*tanh(x/2)`` and whose ``LayerNorm`` is a
    shifted one-pass) are not what the reference runs, even though this file
    imports them for its own fallback path."""
    return 1.0 / (1.0 + _expf(-x))


@triton.jit
def _silu(x):
    """``F.silu`` on bf16: ``x*sigmoid(x)`` in fp32, rounded once."""
    return (x * _sig(x)).to(tl.bfloat16)


@triton.jit
def _adaln(x, GP, SP, rows, n, msk, C: tl.constexpr, EPS: tl.constexpr):
    """``AdaLN(a, s) = g * (layer_norm(a) + linear_s(layer_norm_s(s)))`` for the
    given rows, with the two ``s``-side terms read from the precomputed
    per-atom tables (``GP`` = gate, ``SP`` = additive shift)."""
    y = _lnf(x, n, C, EPS).to(tl.bfloat16)
    sa = tl.load(SP + rows[:, None] * C + n[None, :], mask=msk, other=0.0)
    g = tl.load(GP + rows[:, None] * C + n[None, :], mask=msk, other=0.0)
    t = (y.to(tl.float32) + sa.to(tl.float32)).to(tl.bfloat16)
    return (g.to(tl.float32) * t.to(tl.float32)).to(tl.bfloat16)


@triton.jit
def _k_sad(S, ST, GAMT, WT, BS,
           N: tl.constexpr, C: tl.constexpr, NG: tl.constexpr,
           NSTRIP: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr,
           EPS: tl.constexpr):
    """Every ``s``-dependent quantity of the whole stack: one program per
    (atom strip, output table).

    Per transformer block the tables are the query AdaLN's gate and additive
    shift, the key AdaLN's, the transition AdaLN's, and the two output gates
    (``linear_ada_out`` and the transition's ``linear_g``) -- ``8 * no_blocks``
    tables of ``[N, c_a]``, which is the ``ST`` layout the block kernel gathers
    rows from.

    ``layer_norm_s`` is weight-only, so the normalized row is computed once per
    program and each table only re-scales it by its own instance's weight
    (rounded to bf16 first, exactly where the reference rounds) before its own
    ``Linear``.  The two output gates take *raw* ``s`` instead, which is the only
    thing that differs between tables -- so the gamma, the weight and the bias
    are packed per table and the kernel is one dot with no branches.

    One dot per program rather than one program per strip: the whole thing is
    ~150 MMAC, so what decides its runtime is how many SMs it can fill.  At the
    captured shape that is 12 strips x 24 tables = 288 programs instead of 6,
    which measured 68 us -> 6 us.
    """
    pid = tl.program_id(0)
    gg = pid // NSTRIP
    rows = (pid % NSTRIP) * BM + tl.arange(0, BM)
    rm = rows < N
    n = tl.arange(0, BC)
    nm = n < C
    msk = rm[:, None] & nm[None, :]

    x = tl.load(S + rows[:, None] * C + n[None, :], mask=msk, other=0.0).to(tl.float32)
    g = tl.load(GAMT + gg * C + n, mask=nm, other=0.0).to(tl.float32)
    # Tables 0..5 of each block consume layer_norm_s(s); tables 6 and 7 consume
    # raw s.  Both are cheap, so select rather than branch.
    y = tl.where((gg % 8) < 6, (_lnf(x, n, C, EPS) * g[None, :]).to(tl.bfloat16),
                 x.to(tl.bfloat16))
    w = tl.load(WT + gg * (C * C) + n[:, None] * C + n[None, :],
                mask=nm[:, None] & nm[None, :], other=0.0)
    b = tl.load(BS + gg * C + n, mask=nm, other=0.0).to(tl.float32)
    acc = _rb(tl.dot(y, w) + b[None, :])
    # The gates are sigmoid'd; the additive shifts (odd tables below 6) are not.
    out = tl.where(((gg % 8) >= 6) | ((gg % 2) == 0), _sig(acc), acc)
    tl.store(ST + gg * (N * C) + rows[:, None] * C + n[None, :],
             out.to(tl.bfloat16), mask=msk)


@triton.jit
def _zbias(x, GZ, WZ, j, jm, o, P: tl.constexpr, CZP: tl.constexpr,
           EPS: tl.constexpr):
    """All blocks' attention pair biases for one ``[n_key, c_z]`` pair tile.

    ``layer_norm_z`` belongs to the stack, not to a block, so it runs once; the
    per-block ``linear_z`` weights (c_z -> no_heads each) are concatenated into
    one ``[c_z, no_blocks*no_heads]`` matrix.  Returned transposed, so the
    caller's store is contiguous along the key axis (which is how the block
    kernel reads it back)."""
    u = _lnf(x, j, P, EPS)
    g = tl.load(GZ + j, mask=jm, other=0.0).to(tl.float32)
    y = (u * g[None, :]).to(tl.bfloat16)
    w = tl.load(WZ + j[:, None] * CZP + o[None, :], mask=jm[:, None], other=0.0)
    return tl.trans(tl.dot(y, w).to(tl.bfloat16))


@triton.jit
def _k_zb(PLM, ZB, GZ, WZ,
          NK: tl.constexpr, P: tl.constexpr, CZP: tl.constexpr,
          BP: tl.constexpr, EPS: tl.constexpr):
    """``_zbias`` over the pair tensor, one program per query row.  Used by the
    decoder, whose ``plm`` arrives as an input; the encoder folds the same
    computation into ``_k_plm``, which already has the tile in registers."""
    qi = tl.program_id(0)
    kk = tl.arange(0, NK)
    j = tl.arange(0, BP)
    o = tl.arange(0, CZP)
    jm = j < P
    x = tl.load(PLM + (qi * NK + kk)[:, None] * P + j[None, :],
                mask=jm[None, :], other=0.0).to(tl.float32)
    zb = _zbias(x, GZ, WZ, j, jm, o, P, CZP, EPS)
    tl.store(ZB + qi * (CZP * NK) + o[:, None] * NK + kk[None, :], zb)


@triton.jit
def _k_xa(A, ST, ZB, AMSK, OG, W,
          N: tl.constexpr, C: tl.constexpr, CH: tl.constexpr,
          NQ: tl.constexpr, NK: tl.constexpr, H: tl.constexpr,
          D: tl.constexpr, NT: tl.constexpr, CZP: tl.constexpr,
          BC: tl.constexpr, BD: tl.constexpr, BN: tl.constexpr,
          EPS: tl.constexpr,
          RSQ: tl.constexpr, INF: tl.constexpr):
    """``CrossAttentionPairBias`` for one (query block, head), up to the gated
    attention output -- the part that needs the block's key window.

    One program per (block, head) rather than per block: at the captured shape
    that is 48 programs instead of 12, and the kernel is latency-bound, not
    FLOP-bound (12 programs x 10 MMAC reached ~5% of the 12 SMs they occupy).
    The head's key/value projections over its own 128-atom window are the bulk
    of the arithmetic and are redundant across heads *and* across overlapping
    windows, which is still the right trade -- a launch costs more here than
    ~2 MMAC does.

    Writes its ``c_hidden`` slice of the ``[N, no_heads*c_hidden]`` gated output
    so that ``linear_o`` can be one K=no_heads*c_hidden dot in ``_k_xb``, which
    is the reduction the reference's GEMM does.
    """
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    n = tl.arange(0, BC)
    nm = n < C
    dd = tl.arange(0, BD)
    dm = dd < D
    kk = tl.arange(0, NK)
    NC: tl.constexpr = N * C
    OCH: tl.constexpr = C * CH
    WBQ: tl.constexpr = 5 * OCH + 2 * C * NT + NT * C

    qi = b * NQ + tl.arange(0, NQ)
    qok = qi < N
    nreal = _n_real(AMSK, N, BN)
    safe, invalid = _geom(nreal, b, NQ, NK)
    kok = (safe < N) & ~invalid
    mq = tl.load(AMSK + qi, mask=qok, other=0.0).to(tl.float32)
    mk = tl.load(AMSK + safe, mask=kok, other=0.0).to(tl.float32)
    bm = (mq[:, None] * mk[None, :]).to(tl.bfloat16).to(tl.float32)
    mbias = (INF * (bm - 1.0)).to(tl.bfloat16).to(tl.float32)

    qmsk = qok[:, None] & nm[None, :]
    kmsk = kok[:, None] & nm[None, :]
    xq = tl.load(A + qi[:, None] * C + n[None, :], mask=qmsk, other=0.0).to(tl.float32)
    aq = _adaln(xq, ST, ST + NC, qi, n, qmsk, C, EPS)
    xk = tl.load(A + safe[:, None] * C + n[None, :], mask=kmsk, other=0.0).to(tl.float32)
    ak = _adaln(xk, ST + 2 * NC, ST + 3 * NC, safe, n, kmsk, C, EPS)

    hd = h * D + dd
    wnd = nm[:, None] & dm[None, :]
    q = (tl.dot(aq, tl.load(W + n[:, None] * CH + hd[None, :], mask=wnd, other=0.0))
         + tl.load(W + WBQ + hd, mask=dm, other=0.0).to(tl.float32)[None, :])
    q = (q.to(tl.bfloat16).to(tl.float32) / RSQ).to(tl.bfloat16)
    kt = tl.dot(ak, tl.load(W + OCH + n[:, None] * CH + hd[None, :],
                            mask=wnd, other=0.0)).to(tl.bfloat16)
    sc = (tl.dot(q, tl.trans(kt)).to(tl.bfloat16).to(tl.float32) + mbias).to(tl.bfloat16)
    zb = tl.load(ZB + qi[:, None] * (CZP * NK) + h * NK + kk[None, :],
                 mask=qok[:, None], other=0.0)
    sc = (sc.to(tl.float32) + zb.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    e = _expf(sc - tl.max(sc, 1)[:, None])
    pr = (e / tl.sum(e, 1)[:, None]).to(tl.bfloat16)
    vt = tl.dot(ak, tl.load(W + 2 * OCH + n[:, None] * CH + hd[None, :],
                            mask=wnd, other=0.0)).to(tl.bfloat16)
    oh = tl.dot(pr, vt).to(tl.bfloat16).to(tl.float32)
    gh = tl.dot(aq, tl.load(W + 3 * OCH + n[:, None] * CH + hd[None, :],
                            mask=wnd, other=0.0)).to(tl.bfloat16)
    gh = _sig(gh.to(tl.float32)).to(tl.bfloat16)
    tl.store(OG + qi[:, None] * CH + hd[None, :],
             (oh * gh.to(tl.float32)).to(tl.bfloat16),
             mask=qok[:, None] & dm[None, :])


@triton.jit
def _k_xb(A, OG, ST, AMSK, OUT, W,
          N: tl.constexpr, C: tl.constexpr, CH: tl.constexpr,
          NT: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr,
          BCH: tl.constexpr, BNT: tl.constexpr, EPS: tl.constexpr):
    """Everything after the attention: ``linear_o``, the ``linear_ada_out``
    gate, the first residual add, then the whole AdaLN -> SwiGLU -> gated
    transition and the second residual add.

    All of it is row-wise once the gated attention output exists, so the grid is
    atom strips rather than query blocks -- twice the programs of the block grid,
    and it no longer has to wait on the key window.
    """
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    rm = rows < N
    n = tl.arange(0, BC)
    nm = n < C
    ch = tl.arange(0, BCH)
    tt = tl.arange(0, BNT)
    tm = tt < NT
    msk = rm[:, None] & nm[None, :]
    NC: tl.constexpr = N * C
    OCH: tl.constexpr = C * CH
    OA: tl.constexpr = 5 * OCH
    OB: tl.constexpr = OA + C * NT
    OT: tl.constexpr = OB + C * NT

    og = tl.load(OG + rows[:, None] * CH + ch[None, :],
                 mask=rm[:, None] & (ch < CH)[None, :], other=0.0)
    acc = tl.dot(og, tl.load(W + 4 * OCH + ch[:, None] * C + n[None, :],
                             mask=(ch < CH)[:, None] & nm[None, :], other=0.0))
    xq = tl.load(A + rows[:, None] * C + n[None, :], mask=msk, other=0.0).to(tl.float32)
    ga = tl.load(ST + 6 * NC + rows[:, None] * C + n[None, :], mask=msk, other=0.0)
    upd = (acc.to(tl.bfloat16).to(tl.float32) * ga.to(tl.float32)).to(tl.bfloat16)
    a1 = (xq + upd.to(tl.float32)).to(tl.bfloat16)

    x2 = _adaln(a1.to(tl.float32), ST + 4 * NC, ST + 5 * NC, rows, n, msk, C, EPS)
    wnt = nm[:, None] & tm[None, :]
    h1 = tl.dot(x2, tl.load(W + OA + n[:, None] * NT + tt[None, :],
                            mask=wnt, other=0.0)).to(tl.bfloat16).to(tl.float32)
    h2 = tl.dot(x2, tl.load(W + OB + n[:, None] * NT + tt[None, :],
                            mask=wnt, other=0.0)).to(tl.bfloat16)
    bb = (_silu(h1).to(tl.float32) * h2.to(tl.float32)).to(tl.bfloat16)
    o2 = tl.dot(bb, tl.load(W + OT + tt[:, None] * C + n[None, :],
                            mask=tm[:, None] & nm[None, :], other=0.0)).to(tl.bfloat16)
    gt = tl.load(ST + 7 * NC + rows[:, None] * C + n[None, :], mask=msk, other=0.0)
    mq = tl.load(AMSK + rows, mask=rm, other=0.0).to(tl.float32)
    u2 = (gt.to(tl.float32) * o2.to(tl.float32)).to(tl.bfloat16)
    u2 = (u2.to(tl.float32) * mq[:, None]).to(tl.bfloat16)
    tl.store(OUT + rows[:, None] * C + n[None, :],
             (a1.to(tl.float32) + u2.to(tl.float32)).to(tl.bfloat16), mask=msk)


@triton.jit
def _k_dec_in(AI, QL, A2T, OUT, WQI,
              N: tl.constexpr, C: tl.constexpr, CTK: tl.constexpr,
              T: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr,
              BK: tl.constexpr, NK: tl.constexpr):
    """``ql + broadcast(linear_q_in(ai))`` in one launch.

    Instead of projecting all ``N_token`` rows and then gathering them to atoms,
    a strip of atoms gathers *its own* ``ai`` rows (one address computation) and
    projects them, so the broadcast disappears into the GEMM's A operand.  The
    duplicated arithmetic is 12 x [32, 768] x [768, 128] -- 38 MMAC, free at
    these sizes -- and it removes a Linear, a gather and an add.
    """
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    rm = rows < N
    n = tl.arange(0, BC)
    nm = n < C
    t = tl.load(A2T + rows, mask=rm, other=0).to(tl.int32)
    t = tl.minimum(tl.maximum(t, 0), T - 1)
    acc = tl.zeros((BM, BC), tl.float32)
    for ci in tl.static_range(0, NK):
        kk = ci * BK + tl.arange(0, BK)
        km = kk < CTK
        x = tl.load(AI + t[:, None] * CTK + kk[None, :],
                    mask=rm[:, None] & km[None, :], other=0.0)
        w = tl.load(WQI + kk[:, None] * C + n[None, :],
                    mask=km[:, None] & nm[None, :], other=0.0)
        acc = tl.dot(x, w, acc)
    q = tl.load(QL + rows[:, None] * C + n[None, :],
                mask=rm[:, None] & nm[None, :], other=0.0).to(tl.float32)
    tl.store(OUT + rows[:, None] * C + n[None, :], (q + _rb(acc)).to(tl.bfloat16),
             mask=rm[:, None] & nm[None, :])


@triton.jit
def _k_dec_out(QL, OUT, WQO, GLN,
               N: tl.constexpr, C: tl.constexpr, CO: tl.constexpr,
               BM: tl.constexpr, BK: tl.constexpr, NK: tl.constexpr,
               BO: tl.constexpr, EPS: tl.constexpr):
    """``linear_q_out(layer_norm(ql))`` in one launch (two ops, ~14 us of
    dispatch, become one 3.6 us launch)."""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    rm = rows < N
    acc, o, om = _ln_dot(QL, GLN, WQO, rows, rm, C, CO, BM, BK, NK, BO, EPS)
    tl.store(OUT + rows[:, None] * CO + o[None, :], acc.to(tl.bfloat16),
             mask=rm[:, None] & om[None, :])


@triton.jit
def _k_post(QL, AMSK, QLM, AP, WQ,
            N: tl.constexpr, C: tl.constexpr, CT: tl.constexpr,
            BM: tl.constexpr, BC: tl.constexpr, BT: tl.constexpr,
            NCT: tl.constexpr):
    """``ql * atom_mask`` and ``relu(linear_q(.))`` for one strip of atoms.

    The strip owns every ``c_token`` column, so the mask multiply is fused into
    the GEMM's A operand and written out once for the caller instead of costing
    its own elementwise launch.
    """
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    rm = rows < N
    n = tl.arange(0, BC)
    nm = n < C
    x = tl.load(QL + rows[:, None] * C + n[None, :],
                mask=rm[:, None] & nm[None, :], other=0.0).to(tl.float32)
    am = tl.load(AMSK + rows, mask=rm, other=0.0).to(tl.float32)
    qm = (x * am[:, None]).to(tl.bfloat16)
    tl.store(QLM + rows[:, None] * C + n[None, :], qm, mask=rm[:, None] & nm[None, :])
    for ci in tl.static_range(0, NCT):
        cols = ci * BT + tl.arange(0, BT)
        cm = cols < CT
        w = tl.load(WQ + n[:, None] * CT + cols[None, :],
                    mask=nm[:, None] & cm[None, :], other=0.0)
        a = tl.maximum(tl.dot(qm, w), 0.0).to(tl.bfloat16)
        tl.store(AP + rows[:, None] * CT + cols[None, :], a,
                 mask=rm[:, None] & cm[None, :])


@triton.jit
def _k_ai(AP, AMSK, A2T, AI,
          N: tl.constexpr, T: tl.constexpr, CT: tl.constexpr,
          BT: tl.constexpr, BI: tl.constexpr, NIC: tl.constexpr,
          NCB: tl.constexpr):
    """Masked mean of ``atom_proj`` over each token's atoms.

    One segment reduction instead of the reference's ``scatter_add_`` pair: a
    program owns ``(token, column strip)`` and walks the atom axis testing
    ``atom_to_token_index``, so it never needs the index sorted and never
    atomically accumulates in bf16.
    """
    pid = tl.program_id(0)
    t = pid // NCB
    cols = (pid % NCB) * BT + tl.arange(0, BT)
    cm = cols < CT
    acc = tl.zeros((BT,), tl.float32)
    cnt = 0.0
    for ci in tl.static_range(0, NIC):
        ii = ci * BI + tl.arange(0, BI)
        im = ii < N
        idx = tl.load(A2T + ii, mask=im, other=-1).to(tl.int32)
        sel = (idx == t) & im
        wgt = tl.where(sel, tl.load(AMSK + ii, mask=im, other=0.0).to(tl.float32), 0.0)
        cnt += tl.sum(wgt)
        a = tl.load(AP + ii[:, None] * CT + cols[None, :],
                    mask=sel[:, None] & cm[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(a * wgt[:, None], 0)
    ai = acc / tl.maximum(cnt, 1.0)
    tl.store(AI + t * CT + cols, ai.to(tl.bfloat16), mask=cm)


def _align(n: int) -> int:
    """Round an element count up so every arena slice starts 16-byte aligned
    (the alignment Triton's pointer specialization assumes)."""
    return (n + 63) & ~63


class _Launch:
    """A Triton kernel compiled once, then launched through its own C entry point.

    ``kernel[grid](...)`` re-binds and re-specializes every argument, hashes the
    result and rebuilds the launch metadata on each call -- 8.8 us measured here
    against 3.6 us for the raw launcher, which is more than the kernels
    themselves cost at these sizes.  Every argument except the pointers is
    constexpr, so all of that work is loop-invariant: compile once (with the
    real first-call tensors, so the compiling call is also a working call), then
    keep the compiled function, the invariant launcher prefix, the trailing
    weight pointers and the constexpr tail.  Same technique as
    ``L1/layer_norm.py``; if a future Triton reshapes ``CudaLauncher``, ``raw``
    stays None and the caller uses the reference path.
    """

    __slots__ = ("gx", "tail", "cargs", "raw", "pre")

    def __init__(self, fn, gx, args, ndyn, cargs, warps=4, stages=1):
        self.raw = None
        kern = fn[(gx,)](*args, *cargs, num_warps=warps, num_stages=stages)
        launcher = getattr(kern, "run", None)
        raw = getattr(launcher, "launch", None)
        if (raw is None
                or getattr(launcher, "global_scratch_size", None) != 0
                or getattr(launcher, "profile_scratch_size", None) != 0):
            return
        self.gx = gx
        self.cargs = cargs
        self.tail = tuple(a.data_ptr() if torch.is_tensor(a) else a
                          for a in args[ndyn:])
        self.pre = (kern.function, launcher.launch_cooperative_grid,
                    launcher.launch_pdl, None, None, kern.packed_metadata,
                    None, None, None)
        self.raw = raw

    def __call__(self, dev, *dyn):
        self.raw(self.gx, 1, 1, _raw_stream(dev), *self.pre,
                 *dyn, *self.tail, *self.cargs)


class _Plan:
    """Everything about one (module, input-shape) pair that does not change per
    call: the constexpr geometry, the arena layout, the cached transposed
    weights and the five memoized launchers."""

    __slots__ = ("noisy", "n", "c", "p", "t", "nq", "nk", "nb", "ct", "dev",
                 "ke", "kn", "cs", "cz", "off", "size", "shp", "w", "xf",
                 "k_proj", "k_cl", "k_plm", "k_post", "k_ai",
                 "k_din", "k_dout")

    def __init__(self):
        self.k_proj = None
        self.xf = None


def _t2(w, dt=torch.bfloat16):
    """``w[out, in]`` -> a contiguous ``[in, out]`` copy, so a kernel column
    walks contiguous memory."""
    return w.detach().t().contiguous().to(dt)



class _Xf:
    """The atom transformer's packed parameters and its three launchers.

    ``self.atom_transformer`` is left exactly as constructed -- same class, same
    kwargs, same parameter tree, so ``state_dict`` keys and the reference
    fallback are untouched.  This object only *reads* those parameters (once per
    parameter version; ``_drop_plan`` rebuilds it after a state-dict load) and
    packs them the way the kernels want them.
    """

    __slots__ = ("nblk", "h", "d", "ch", "nt", "p", "czp", "n", "c", "nq", "nk",
                 "nb", "nbq", "st_n", "zb_n", "og_n", "w", "k_sad", "k_zb",
                 "k_xa", "k_xb")


def _lin_ok(lin, shape, bias):
    """A ``Linear`` with the expected weight shape and bias presence."""
    return (lin is not None and getattr(lin, "weight", None) is not None
            and tuple(lin.weight.shape) == shape
            and (getattr(lin, "bias", None) is not None) is bias)


def _bias_of(lin, m, dev, bf=torch.bfloat16):
    b = getattr(lin, "bias", None)
    if b is None:
        return torch.zeros(m, device=dev, dtype=bf)
    return b.detach().contiguous().to(bf)


def _xf_probe(xf, c, p, nq, nk):
    """Structural check of the atom transformer.  Returns
    ``(nblk, h, d, ch, nt, eps_a, eps_s, eps_z, inf)`` or None.

    Deliberately strict: anything the fused stack does not reproduce exactly --
    a different AdaLN wiring, a missing gate, a non-uniform eps, an ``inf`` that
    is not the masking constant -- has to reach the real module instead.
    """
    if not getattr(xf, "use_cross_attention", False):
        return None
    lnz = getattr(xf, "layer_norm_z", None)
    if (lnz is None or getattr(lnz, "weight", None) is None
            or getattr(lnz, "bias", None) is not None
            or tuple(lnz.normalized_shape) != (p,)):
        return None
    blocks = getattr(xf, "blocks", None)
    if not blocks or len(blocks) < 1:
        return None
    ref = None
    for blk in blocks:
        pb = getattr(blk, "attention_pair_bias", None)
        ct = getattr(blk, "conditioned_transition", None)
        if pb is None or ct is None or not getattr(pb, "use_ada_layer_norm", False):
            return None
        if getattr(pb, "n_query", None) != nq or getattr(pb, "n_key", None) != nk:
            return None
        mha = getattr(pb, "mha", None)
        if mha is None or mha.linear_g is None or not mha.gating:
            return None
        h, d = int(mha.no_heads), int(mha.c_hidden)
        ch = h * d
        if not (_lin_ok(mha.linear_q, (ch, c), True)
                and _lin_ok(mha.linear_k, (ch, c), False)
                and _lin_ok(mha.linear_v, (ch, c), False)
                and _lin_ok(mha.linear_o, (c, ch), False)
                and _lin_ok(mha.linear_g, (ch, c), False)
                and _lin_ok(pb.linear_z, (h, p), False)
                and _lin_ok(pb.linear_ada_out, (c, c), True)
                and _lin_ok(ct.linear_g, (c, c), True)):
            return None
        sw = getattr(ct, "swiglu", None)
        if sw is None or getattr(sw, "linear_a", None) is None:
            return None
        nt = int(sw.linear_a.weight.shape[0])
        if not (_lin_ok(sw.linear_a, (nt, c), False)
                and _lin_ok(sw.linear_b, (nt, c), False)
                and _lin_ok(ct.linear_out, (c, nt), False)):
            return None
        eps_a = eps_s = None
        for ad in (pb.layer_norm_a_q, pb.layer_norm_a_k, ct.layer_norm):
            la, ls = getattr(ad, "layer_norm_a", None), getattr(ad, "layer_norm_s", None)
            if (la is None or ls is None
                    or getattr(la, "weight", None) is not None
                    or getattr(la, "bias", None) is not None
                    or getattr(ls, "weight", None) is None
                    or getattr(ls, "bias", None) is not None
                    or tuple(la.normalized_shape) != (c,)
                    or tuple(ls.normalized_shape) != (c,)
                    or not _lin_ok(ad.linear_g, (c, c), True)
                    or not _lin_ok(ad.linear_s, (c, c), False)):
                return None
            if eps_a is None:
                eps_a, eps_s = float(la.eps), float(ls.eps)
            elif (float(la.eps), float(ls.eps)) != (eps_a, eps_s):
                return None
        sig = (h, d, ch, nt, eps_a, eps_s, float(getattr(pb, "inf", 0.0)))
        if ref is None:
            ref = sig
        elif sig != ref:
            return None
    h, d, ch, nt, eps_a, eps_s, inf = ref
    if not (h >= 1 and d >= 16 and nt >= 16 and inf > 0.0):
        return None
    return len(blocks), h, d, ch, nt, eps_a, eps_s, float(lnz.eps), inf


def _xf_sizes(probe, n, c, nq, nk):
    """``(st_elems, zb_elems, og_elems)`` the arena has to hold, from the probe
    alone -- the caller needs them before it can lay the arena out."""
    if probe is None:
        return 0, 0, 0
    nblk, h, d = probe[0], probe[1], probe[2]
    nb = -(-n // nq)
    czp = max(16, triton.next_power_of_2(nblk * h))
    return nblk * 8 * n * c, nb * nq * czp * nk, n * h * d


def _xf_build(xf, probe, n, c, p, nq, nk, arena, st_off, zb_off, og_off,
              a_off, amsk):
    """Pack the parameters and compile the launchers.  ``arena``/offsets only
    provide real tensors for the compiling first call."""
    if probe is None:
        return None
    nblk, h, d, ch, nt, eps_a, eps_s, eps_z, inf = probe
    bf = torch.bfloat16
    dev = amsk.device
    nb = -(-n // nq)
    nbq = nb * nq
    czp = max(16, triton.next_power_of_2(nblk * h))
    if czp > 256 or n * c <= 0:
        return None

    x = _Xf()
    x.nblk, x.h, x.d, x.ch, x.nt, x.p, x.czp = nblk, h, d, ch, nt, p, czp
    x.n, x.c, x.nq, x.nk, x.nb, x.nbq = n, c, nq, nk, nb, nbq
    x.st_n = nblk * 8 * n * c
    x.zb_n = nbq * czp * nk
    x.og_n = n * ch

    ones = torch.ones(c, device=dev, dtype=bf)
    zero = torch.zeros(c, device=dev, dtype=bf)
    gamt, wt, bs = [], [], []
    wz = torch.zeros(p, czp, device=dev, dtype=bf)
    packs = []
    for i, blk in enumerate(xf.blocks):
        pb, ct = blk.attention_pair_bias, blk.conditioned_transition
        # Tables 0..5: (gate, shift) for the query, key and transition AdaLNs.
        for ad in (pb.layer_norm_a_q, pb.layer_norm_a_k, ct.layer_norm):
            gw = ad.layer_norm_s.weight.detach().contiguous().to(bf)
            gamt += [gw, gw]
            wt += [_t2(ad.linear_g.weight), _t2(ad.linear_s.weight)]
            bs += [_bias_of(ad.linear_g, c, dev), zero]
        # Tables 6, 7: the two output gates, over raw s (gamma unused).
        gamt += [ones, ones]
        wt += [_t2(pb.linear_ada_out.weight), _t2(ct.linear_g.weight)]
        bs += [_bias_of(pb.linear_ada_out, c, dev), _bias_of(ct.linear_g, c, dev)]
        wz[:, i * h:(i + 1) * h].copy_(_t2(pb.linear_z.weight))
        mha = pb.mha
        packs.append(torch.cat([t.reshape(-1) for t in (
            _t2(mha.linear_q.weight), _t2(mha.linear_k.weight),
            _t2(mha.linear_v.weight), _t2(mha.linear_g.weight),
            _t2(mha.linear_o.weight), _t2(ct.swiglu.linear_a.weight),
            _t2(ct.swiglu.linear_b.weight), _t2(ct.linear_out.weight),
            _bias_of(mha.linear_q, ch, dev))]).contiguous())
    gz = xf.layer_norm_z.weight.detach().contiguous().to(bf)
    x.w = (torch.cat(gamt).contiguous(),
           torch.cat([t.reshape(-1) for t in wt]).contiguous(),
           torch.cat(bs).contiguous(), gz, wz.contiguous(), tuple(packs))

    bc = triton.next_power_of_2(c)
    bp = max(16, triton.next_power_of_2(p))
    s0 = arena.as_strided((n, c), (c, 1), a_off)
    st = arena.as_strided((x.st_n,), (1,), st_off)
    zb = arena.as_strided((x.zb_n,), (1,), zb_off)
    og = arena.as_strided((n, ch), (ch, 1), og_off)
    try:
        _xf_compile(x, arena, s0, st, zb, og, amsk, n, c, p, nq, nk, nb, nbq,
                    nblk, h, d, ch, nt, czp, bc, bp, eps_a, eps_s, eps_z,
                    inf, packs)
    except Exception:  # noqa: BLE001
        # A wide geometry can exceed the register / tensor-memory budget of the
        # block kernel (``no_heads*c_hidden`` and ``n_transition*c_a`` are both
        # register tiles).  That is a "call the real module" answer, not an
        # error: nothing has been mutated yet.
        return None
    if (x.k_sad.raw is None or x.k_zb.raw is None
            or any(k.raw is None for k in x.k_xa + x.k_xb)):
        return None
    return x


def _xf_compile(x, arena, s0, st, zb, og, amsk, n, c, p, nq, nk, nb, nbq,
                nblk, h, d, ch, nt, czp, bc, bp, eps_a, eps_s, eps_z, inf, packs):
    """Compile and memoize the three launchers (see ``_Launch``)."""
    gz, wz = x.w[3], x.w[4]
    nstrip = -(-n // 32)
    x.k_sad = _Launch(
        _k_sad, nstrip * 8 * nblk, (s0, st) + x.w[:3], 2,
        (n, c, 8 * nblk, nstrip, 32, bc, eps_s), warps=4)
    x.k_zb = _Launch(
        _k_zb, nbq, (arena, zb, gz, wz), 2,
        (nk, p, czp, bp, eps_z))
    bch = max(16, triton.next_power_of_2(ch))
    bnt = max(16, triton.next_power_of_2(nt))
    x.k_xa = tuple(
        _Launch(_k_xa, nb * h, (s0, st, zb, amsk, og, packs[i]), 5,
                (n, c, ch, nq, nk, h, d, nt, czp, bc,
                 max(16, triton.next_power_of_2(d)),
                 triton.next_power_of_2(n), eps_a, float(math.sqrt(d)), inf),
                warps=8)
        for i in range(nblk))
    x.k_xb = tuple(
        _Launch(_k_xb, -(-n // 32), (s0, og, st, amsk, s0, packs[i]), 5,
                (n, c, ch, nt, 32, bc, bch, bnt, eps_a), warps=8)
        for i in range(nblk))

class RefAtomFeatureEmbedder(nn.Module):
    """Embeds reference atom features (Algorithm 5, lines 1-6).

    Args:
        c_atom_ref_element: Reference element one-hot dim (119)
        c_atom_ref_name_chars: Reference atom name chars dim (256 = 4*64)
        c_atom: Atom single conditioning dim
        c_atom_pair: Atom pair conditioning dim
    """

    def __init__(
        self,
        c_atom_ref_element: int = 119,
        c_atom_ref_name_chars: int = 256,
        c_atom: int = 128,
        c_atom_pair: int = 16,
    ):
        super().__init__()
        self.linear_ref_pos = Linear(3, c_atom, bias=False)
        self.linear_ref_charge = Linear(1, c_atom, bias=False)
        self.linear_ref_mask = Linear(1, c_atom, bias=False)
        self.linear_ref_element = Linear(c_atom_ref_element, c_atom, bias=False)
        self.linear_ref_atom_chars = Linear(c_atom_ref_name_chars, c_atom, bias=False)
        self.linear_ref_offset = Linear(3, c_atom_pair, bias=False)
        self.linear_inv_sq_dists = Linear(1, c_atom_pair, bias=False)
        self.linear_valid_mask = Linear(1, c_atom_pair, bias=False)

    def forward(
        self,
        batch: dict,
        n_query: int,
        n_key: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = batch["ref_pos"].dtype

        cl = self.linear_ref_pos(batch["ref_pos"])
        cl = cl + self.linear_ref_charge(
            torch.arcsinh(batch["ref_charge"].unsqueeze(-1))
        )
        cl = cl + self.linear_ref_mask(batch["ref_mask"].unsqueeze(-1).to(dtype=dtype))
        cl = cl + self.linear_ref_element(batch["ref_element"].to(dtype=dtype))
        cl = cl + self.linear_ref_atom_chars(
            batch["ref_atom_name_chars"].flatten(start_dim=-2).to(dtype=dtype)
        )

        d_l, d_m, atom_mask = _convert_single_rep_to_blocks(
            ql=batch["ref_pos"],
            n_query=n_query, n_key=n_key,
            atom_mask=batch["atom_mask"],
        )
        v_l, v_m, _ = _convert_single_rep_to_blocks(
            ql=batch["ref_space_uid"].unsqueeze(-1),
            n_query=n_query, n_key=n_key,
            atom_mask=batch["atom_mask"],
        )

        dlm = (d_l.unsqueeze(-2) - d_m.unsqueeze(-3)) * atom_mask.unsqueeze(-1)
        vlm = (v_l.unsqueeze(-2) == v_m.unsqueeze(-3)).to(
            dtype=dlm.dtype
        ) * atom_mask.unsqueeze(-1)

        plm = self.linear_ref_offset(dlm) * vlm

        inv_sq_dists = 1.0 / (1 + torch.sum(dlm ** 2, dim=-1, keepdim=True))
        plm = plm + self.linear_inv_sq_dists(inv_sq_dists) * vlm
        plm = plm + self.linear_valid_mask(vlm) * vlm

        return cl, plm


class NoisyPositionEmbedder(nn.Module):
    """Embeds noisy positions and trunk embeddings (Algorithm 5, lines 8-12).

    Args:
        c_s: Single representation channel dimension
        c_z: Pair representation channel dimension
        c_atom: Atom single conditioning channel dimension
        c_atom_pair: Atom pair conditioning channel dimension
    """

    def __init__(self, c_s: int, c_z: int, c_atom: int, c_atom_pair: int):
        super().__init__()
        self.layer_norm_s = LayerNorm(c_s, create_offset=False)
        self.linear_s = Linear(c_s, c_atom, bias=False)
        self.layer_norm_z = LayerNorm(c_z, create_offset=False)
        self.linear_z = Linear(c_z, c_atom_pair, bias=False)
        self.linear_r = Linear(3, c_atom, bias=False)

    def forward(
        self,
        batch: dict,
        cl: torch.Tensor,
        plm: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
        rl: torch.Tensor,
        n_query: int,
        n_key: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        si_trunk_proj = self.linear_s(self.layer_norm_s(si_trunk))
        si_trunk_proj = _broadcast_token_feat_to_atoms(
            token_mask=batch["token_mask"],
            num_atoms_per_token=batch.get("num_atoms_per_token"),
            token_feat=si_trunk_proj,
            atom_to_token_index=batch.get("atom_to_token_index"),
        )
        cl = cl + si_trunk_proj

        zij_trunk_proj = self.linear_z(self.layer_norm_z(zij_trunk))
        zij_trunk_block = _convert_pair_rep_to_blocks(
            batch=batch, zij_trunk=zij_trunk_proj,
            n_query=n_query, n_key=n_key,
        )
        plm = plm + zij_trunk_block

        ql = cl + self.linear_r(rl)

        return cl, plm, ql


class AtomAttentionEncoder(nn.Module):
    """AF3 Algorithm 5: Atom attention encoder.

    Args:
        c_atom: Atom single representation channel dimension
        c_atom_pair: Atom pair representation channel dimension
        c_token: Token single representation output channel dimension
        c_atom_ref_element: Reference element one-hot dim
        c_atom_ref_name_chars: Reference atom name chars dim
        add_noisy_pos: Whether to embed noisy positions and trunk reps
        c_s: Single representation dim (optional, needed if add_noisy_pos)
        c_z: Pair representation dim (optional, needed if add_noisy_pos)
        c_hidden: Per-head hidden dim for atom transformer
        no_heads: Number of attention heads
        no_blocks: Number of transformer blocks
        n_transition: Transition blocks per transformer block
        n_query: Block height for sequence-local attention
        n_key: Block width for sequence-local attention
        use_ada_layer_norm: Whether to use AdaLN
    """

    def __init__(
        self,
        c_atom: int = 128,
        c_atom_pair: int = 16,
        c_token: int = 384,
        c_atom_ref_element: int = 119,
        c_atom_ref_name_chars: int = 256,
        add_noisy_pos: bool = False,
        c_s: int | None = None,
        c_z: int | None = None,
        c_hidden: int = 32,
        no_heads: int = 4,
        no_blocks: int = 3,
        n_transition: int = 2,
        n_query: int = 32,
        n_key: int = 128,
        use_ada_layer_norm: bool = True,
        transformer_cls=None,
    ):
        super().__init__()
        if transformer_cls is None:
            from ..L3.alphafold3_diffusion_transformer import DiffusionTransformer
            transformer_cls = DiffusionTransformer

        self.n_query = n_query
        self.n_key = n_key

        self.ref_atom_feature_embedder = RefAtomFeatureEmbedder(
            c_atom_ref_element=c_atom_ref_element,
            c_atom_ref_name_chars=c_atom_ref_name_chars,
            c_atom=c_atom,
            c_atom_pair=c_atom_pair,
        )

        self.noisy_position_embedder: NoisyPositionEmbedder | None = None
        if add_noisy_pos:
            assert c_s is not None and c_z is not None
            self.noisy_position_embedder = NoisyPositionEmbedder(
                c_s=c_s, c_z=c_z, c_atom=c_atom, c_atom_pair=c_atom_pair,
            )

        self.relu = ReLU()
        self.linear_l = Linear(c_atom, c_atom_pair, bias=False)
        self.linear_m = Linear(c_atom, c_atom_pair, bias=False)

        self.pair_mlp = nn.Sequential(
            ReLU(),
            Linear(c_atom_pair, c_atom_pair, bias=False),
            ReLU(),
            Linear(c_atom_pair, c_atom_pair, bias=False),
            ReLU(),
            Linear(c_atom_pair, c_atom_pair, bias=False),
        )

        self.atom_transformer = transformer_cls(
            c_a=c_atom, c_s=c_atom, c_z=c_atom_pair,
            c_hidden=c_hidden, no_heads=no_heads,
            no_blocks=no_blocks, n_transition=n_transition,
            use_ada_layer_norm=use_ada_layer_norm,
            n_query=n_query, n_key=n_key,
        )

        self.linear_q = nn.Sequential(
            Linear(c_atom, c_token, bias=False),
            ReLU(),
        )

        # Fused-path state: the plan is derived on the first supported call and
        # invalidated by a weight reload (the packed weights are copies).
        self._plan = None
        self._no_fast = False
        self.register_load_state_dict_post_hook(_drop_plan)

    # ------------------------------------------------------------------
    # fused path
    # ------------------------------------------------------------------
    def _build_plan(self, batch, rl, si_trunk, zij_trunk, noisy):
        """Validate that the fused path applies, then cache its geometry,
        transposed weights and memoized launchers.  Returns None (and disables
        further attempts) for anything the kernels do not cover."""
        bf = torch.bfloat16
        try:
            am = batch["atom_mask"]
            a2t = batch["atom_to_token_index"]
            tm = batch["token_mask"]
            rpos = batch["ref_pos"]
            rchg = batch["ref_charge"]
            rmsk = batch["ref_mask"]
            relem = batch["ref_element"]
            rchar = batch["ref_atom_name_chars"]
            ruid = batch["ref_space_uid"]
        except (KeyError, TypeError):
            return None
        if not (torch.is_tensor(am) and am.is_cuda and am.dtype is bf):
            return None

        e = self.ref_atom_feature_embedder
        n = int(am.shape[-1])
        t = int(tm.shape[-1])
        c = int(e.linear_ref_pos.weight.shape[0])
        pp = int(e.linear_ref_offset.weight.shape[0])
        ke = int(e.linear_ref_element.weight.shape[1])
        kn = int(e.linear_ref_atom_chars.weight.shape[1])
        ct = int(self.linear_q[0].weight.shape[0])
        nq, nk = int(self.n_query), int(self.n_key)
        # BN = next_pow2(n) is a register tile in _k_plm and NIC = n/128 is an
        # unrolled loop in _k_ai; both stop being sensible well before this.
        if not (0 < n <= 8192 and t > 0 and nq > 0 and nk >= 16 and nk % 2 == 0
                and c >= 16 and 2 * pp >= 16 and pp >= 1
                and e.linear_ref_pos.weight.shape[1] == 3):
            return None
        for tt, nel in ((am, n), (rchg, n), (rmsk, n), (ruid, n), (rpos, 3 * n),
                        (relem, ke * n), (rchar, kn * n)):
            if not (torch.is_tensor(tt) and tt.is_cuda and tt.dtype is bf
                    and tt.is_contiguous() and tt.numel() == nel):
                return None
        if not (torch.is_tensor(a2t) and a2t.dtype is torch.int64
                and a2t.is_contiguous() and a2t.numel() == n):
            return None
        lins = [e.linear_ref_pos, e.linear_ref_charge, e.linear_ref_mask,
                e.linear_ref_element, e.linear_ref_atom_chars,
                e.linear_ref_offset, e.linear_inv_sq_dists, e.linear_valid_mask,
                self.linear_l, self.linear_m, self.linear_q[0],
                self.pair_mlp[1], self.pair_mlp[3], self.pair_mlp[5]]
        cs = cz = 0
        npe = self.noisy_position_embedder
        if noisy:
            if not (torch.is_tensor(si_trunk) and torch.is_tensor(zij_trunk)
                    and torch.is_tensor(rl)):
                return None
            cs = int(si_trunk.shape[-1])
            cz = int(zij_trunk.shape[-1])
            for tt, nel in ((si_trunk, t * cs), (zij_trunk, t * t * cz), (rl, 3 * n)):
                if not (tt.dtype is bf and tt.is_contiguous() and tt.numel() == nel):
                    return None
            if not (npe.layer_norm_s.bias is None and npe.layer_norm_z.bias is None
                    and npe.linear_s.weight.shape == (c, cs)
                    and npe.linear_z.weight.shape == (pp, cz)
                    and npe.linear_r.weight.shape == (c, 3)):
                return None
            lins += [npe.linear_s, npe.linear_z, npe.linear_r]
        for lin in lins:
            if getattr(lin, "bias", None) is not None:
                return None

        dev = am.get_device()
        nb = -(-n // nq)
        p = _Plan()
        p.noisy, p.n, p.c, p.p, p.t = noisy, n, c, pp, t
        p.nq, p.nk, p.nb, p.ct, p.dev = nq, nk, nb, ct, dev
        p.ke, p.kn, p.cs, p.cz = ke, kn, cs, cz

        # --- arena layout -------------------------------------------------
        o_cl = 0
        o_ql = o_cl + _align(n * c)
        o_plm = o_ql + _align(n * c)
        o_qlm = o_plm + _align(nb * nq * nk * pp)
        o_ai = o_qlm + _align(n * c)
        o_sip = o_ai + _align(t * ct)
        o_zp = o_sip + _align(t * c)
        o_clp = o_zp + _align(t * t * pp)
        o_ap = o_clp + _align(n * 2 * pp)
        probe = _xf_probe(self.atom_transformer, c, pp, nq, nk)
        st_n, zb_n, og_n = _xf_sizes(probe, n, c, nq, nk)
        o_st = o_ap + _align(n * ct)
        o_zb = o_st + _align(st_n)
        o_og = o_zb + _align(zb_n)
        o_x0 = o_og + _align(og_n)
        o_x1 = o_x0 + _align(n * c if probe else 0)
        p.off = (o_cl, o_ql, o_plm, o_qlm, o_ai, o_sip, o_zp, o_clp, o_ap,
                 o_st, o_zb, o_x0, o_x1, o_og)
        p.size = o_x1 + _align(n * c if probe else 0)

        cl_shape = tuple(rpos.shape[:-1]) + (c,)
        ql_shape = (tuple(torch.broadcast_shapes(cl_shape, tuple(rl.shape[:-1]) + (c,)))
                    if noisy else cl_shape)
        plm_shape = tuple(rpos.shape[:-2]) + (nb, nq, nk, pp)
        ai_shape = ql_shape[:-2] + (t, ct)
        p.shp = tuple((s, _cstride(s)) for s in
                      (cl_shape, ql_shape, plm_shape, ql_shape, ai_shape))

        # --- packed weights (transposed once; kernels read columns) --------
        wa = torch.zeros(5, c, device=am.device, dtype=bf)
        wa[0:3].copy_(_t2(e.linear_ref_pos.weight))
        wa[3].copy_(e.linear_ref_charge.weight[:, 0])
        wa[4].copy_(e.linear_ref_mask.weight[:, 0])
        w = [wa,
             _t2(e.linear_ref_element.weight),
             _t2(e.linear_ref_atom_chars.weight),
             _t2(npe.linear_r.weight) if noisy else wa,
             torch.cat([_t2(self.linear_l.weight), _t2(self.linear_m.weight)],
                       1).contiguous(),
             _t2(e.linear_ref_offset.weight),
             e.linear_inv_sq_dists.weight[:, 0].detach().contiguous().to(bf),
             e.linear_valid_mask.weight[:, 0].detach().contiguous().to(bf),
             _t2(self.pair_mlp[1].weight),
             _t2(self.pair_mlp[3].weight),
             _t2(self.pair_mlp[5].weight),
             _t2(self.linear_q[0].weight)]
        if noisy:
            ones = torch.ones(1, device=am.device, dtype=bf)
            gs = npe.layer_norm_s.weight
            gz = npe.layer_norm_z.weight
            w += [_t2(npe.linear_s.weight), _t2(npe.linear_z.weight),
                  (gs.detach().contiguous().to(bf) if gs is not None
                   else ones.expand(cs).contiguous()),
                  (gz.detach().contiguous().to(bf) if gz is not None
                   else ones.expand(cz).contiguous())]
        p.w = tuple(w)
        WA, WB, WC, WR, WLM, WRO, WINV, WVM, M1, M2, M3, WQ = w[:12]

        # --- compile + memoize the launchers on real first-call tensors ----
        arena = torch.zeros(p.size, device=am.device, dtype=bf)
        cl = arena.as_strided(*p.shp[0], o_cl)
        ql = arena.as_strided(*p.shp[1], o_ql)
        plm = arena.as_strided(*p.shp[2], o_plm)
        sip = arena.as_strided((t, c), (c, 1), o_sip)
        zp = arena.as_strided((t * t, pp), (pp, 1), o_zp)
        clp = arena.as_strided((n, 2 * pp), (2 * pp, 1), o_clp)
        bp = max(16, triton.next_power_of_2(pp))
        bc = triton.next_power_of_2(c)
        if noisy:
            WS, WZ, GS, GZ = w[12:16]
            bt = max(16, triton.next_power_of_2(t))
            bz = 32
            p.k_proj = _Launch(
                _k_proj, 1 + -(-(t * t) // bz),
                (si_trunk, zij_trunk, sip, zp, WS, WZ, GS, GZ), 4,
                (t, cs, cz, c, pp, bt, bz, 128, -(-cs // 128), 128,
                 -(-cz // 128), bc, bp,
                 float(npe.layer_norm_s.eps), float(npe.layer_norm_z.eps)))
        p.k_cl = _Launch(
            _k_cl, -(-n // 32),
            (rpos, rchg, rmsk, relem, rchar, rl if noisy else rpos, sip, a2t,
             cl, ql, clp, WA, WB, WC, WR, WLM), 11,
            (n, c, pp, t, ke, kn, 32, bc, max(16, triton.next_power_of_2(2 * pp)),
             128, -(-ke // 128), -(-kn // 128), noisy, noisy))
        p.k_plm = _Launch(
            _k_plm, nb * nq,
            (rpos, ruid, am, a2t, clp, zp, plm, WRO, WINV, WVM, M1, M2, M3), 7,
            (n, nq, nk, pp, t, triton.next_power_of_2(n), bp, noisy))
        qlm = arena.as_strided(*p.shp[3], o_qlm)
        ai = arena.as_strided(*p.shp[4], o_ai)
        ap = arena.as_strided((n, ct), (ct, 1), o_ap)
        p.k_post = _Launch(
            _k_post, -(-n // 32), (ql, am, qlm, ap, WQ), 4,
            (n, c, ct, 32, bc, 128, -(-ct // 128)))
        ncb = -(-ct // 128)
        p.k_ai = _Launch(
            _k_ai, t * ncb, (ap, am, a2t, ai), 4,
            (n, t, ct, 128, 128, -(-n // 128), ncb))
        if any(k is not None and k.raw is None
               for k in (p.k_proj, p.k_cl, p.k_plm, p.k_post, p.k_ai)):
            return None
        p.xf = _xf_build(self.atom_transformer, probe, n, c, pp, nq, nk,
                         arena, o_st, o_zb, o_og, o_cl, am)
        return p

    def _forward_ref(self, batch, rl, si_trunk, zij_trunk):
        atom_mask = batch["atom_mask"]

        cl, plm = self.ref_atom_feature_embedder(
            batch=batch, n_query=self.n_query, n_key=self.n_key,
        )

        if rl is not None and self.noisy_position_embedder is not None:
            cl, plm, ql = self.noisy_position_embedder(
                batch=batch, cl=cl, plm=plm,
                si_trunk=si_trunk, zij_trunk=zij_trunk, rl=rl,
                n_query=self.n_query, n_key=self.n_key,
            )
        else:
            ql = cl.clone()

        cl_l, cl_m, block_mask = _convert_single_rep_to_blocks(
            ql=cl, n_query=self.n_query, n_key=self.n_key, atom_mask=atom_mask,
        )

        cl_lm = (
            self.linear_l(self.relu(cl_l.unsqueeze(-2)))
            + self.linear_m(self.relu(cl_m.unsqueeze(-3)))
        )
        if block_mask is not None:
            cl_lm = cl_lm * block_mask.unsqueeze(-1)

        plm = plm + cl_lm
        plm = plm + self.pair_mlp(plm)
        if block_mask is not None:
            plm = plm * block_mask.unsqueeze(-1)

        ql = self.atom_transformer(a=ql, s=cl, z=plm, mask=atom_mask)
        ql = ql * atom_mask.unsqueeze(-1)
        atom_proj = self.linear_q(ql)

        if "atom_to_token_index" in batch:
            ai = _aggregate_atom_feat_to_tokens(
                token_mask=batch["token_mask"],
                atom_to_token_index=batch["atom_to_token_index"],
                atom_mask=atom_mask,
                atom_feat=atom_proj,
                mode="mean",
            )
        else:
            ai = atom_proj

        return ai, ql, cl, plm

    def forward(
        self,
        batch: dict,
        rl: torch.Tensor | None = None,
        si_trunk: torch.Tensor | None = None,
        zij_trunk: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            ai: [*, N_token, c_token] token representation
            ql: [*, N_atom, c_atom] atom single representation
            cl: [*, N_atom, c_atom] atom single conditioning
            plm: [*, N_blocks, n_query, n_key, c_atom_pair] atom pair rep
        """
        if not self._no_fast and not torch.is_grad_enabled():
            noisy = rl is not None and self.noisy_position_embedder is not None
            p = self._plan
            am = batch.get("atom_mask")
            if p is None or p.noisy is not noisy or (
                    am is not None and p.n != am.shape[-1]):
                try:
                    p = self._build_plan(batch, rl, si_trunk, zij_trunk, noisy)
                except Exception:  # noqa: BLE001 - unsupported shape: use the reference
                    p = None
                self._plan = p
                if p is None:
                    self._no_fast = True
            if p is not None:
                out = self._fused(p, batch, rl, si_trunk, zij_trunk)
                if out is not None:
                    return out
        return self._forward_ref(batch, rl, si_trunk, zij_trunk)

    def _fused(self, p, batch, rl, si_trunk, zij_trunk):
        bf = torch.bfloat16
        am = batch["atom_mask"]
        a2t = batch["atom_to_token_index"]
        rpos = batch["ref_pos"]
        rchg = batch["ref_charge"]
        rmsk = batch["ref_mask"]
        relem = batch["ref_element"]
        rchar = batch["ref_atom_name_chars"]
        ruid = batch["ref_space_uid"]
        n, c, t, pp, ct = p.n, p.c, p.t, p.p, p.ct
        if batch["token_mask"].shape[-1] != t:
            return None
        if not (am.dtype is bf and rchg.dtype is bf and rmsk.dtype is bf
                and ruid.dtype is bf and rpos.dtype is bf and relem.dtype is bf
                and rchar.dtype is bf and a2t.dtype is torch.int64
                and am.numel() == n and rchg.numel() == n and rmsk.numel() == n
                and ruid.numel() == n and a2t.numel() == n
                and rpos.numel() == 3 * n and relem.numel() == p.ke * n
                and rchar.numel() == p.kn * n
                and am.is_contiguous() and rchg.is_contiguous()
                and rmsk.is_contiguous() and ruid.is_contiguous()
                and rpos.is_contiguous() and relem.is_contiguous()
                and rchar.is_contiguous() and a2t.is_contiguous()):
            return None
        dev = p.dev
        if am.get_device() != dev or _cur_device() != dev:
            return None
        q_am, q_a2t = am.data_ptr(), a2t.data_ptr()
        q_pos, q_chg, q_msk = rpos.data_ptr(), rchg.data_ptr(), rmsk.data_ptr()
        q_elem, q_char, q_uid = relem.data_ptr(), rchar.data_ptr(), ruid.data_ptr()
        bad = (q_am | q_a2t | q_pos | q_chg | q_msk | q_elem | q_char | q_uid)
        if p.noisy:
            q_rl, q_si, q_zij = rl.data_ptr(), si_trunk.data_ptr(), zij_trunk.data_ptr()
            bad |= q_rl | q_si | q_zij
            if not (rl.is_contiguous() and si_trunk.is_contiguous()
                    and zij_trunk.is_contiguous() and rl.numel() == 3 * n
                    and si_trunk.numel() == t * p.cs
                    and zij_trunk.numel() == t * t * p.cz
                    and rl.dtype is bf and si_trunk.dtype is bf
                    and zij_trunk.dtype is bf):
                return None
        else:
            q_rl = q_pos
        if bad & 15:
            return None

        arena = torch.empty(p.size, device=am.device, dtype=bf)
        base = arena.data_ptr()
        off = p.off
        shp = p.shp
        cl = arena.as_strided(*shp[0], off[0])
        ql = arena.as_strided(*shp[1], off[1])
        plm = arena.as_strided(*shp[2], off[2])
        q_cl = base + off[0] * 2
        q_ql = base + off[1] * 2
        q_plm = base + off[2] * 2
        q_sip = base + off[5] * 2
        q_zp = base + off[6] * 2
        q_clp = base + off[7] * 2

        if p.noisy:
            p.k_proj(dev, q_si, q_zij, q_sip, q_zp)
        p.k_cl(dev, q_pos, q_chg, q_msk, q_elem, q_char, q_rl, q_sip, q_a2t,
               q_cl, q_ql, q_clp)
        p.k_plm(dev, q_pos, q_uid, q_am, q_a2t, q_clp, q_zp, q_plm)

        if p.xf is not None:
            q_xf = _xf_run(p.xf, dev, q_ql, q_cl, q_plm, q_am, base, off[9],
                           off[10], off[13], off[11], off[12])
        else:
            xf = self.atom_transformer(a=ql, s=cl, z=plm, mask=am)
            if (xf.shape != shp[1][0] or xf.dtype is not bf
                    or not xf.is_contiguous() or xf.data_ptr() & 15):
                xf = xf * am.unsqueeze(-1)
                return (_aggregate_atom_feat_to_tokens(
                    token_mask=batch["token_mask"], atom_to_token_index=a2t,
                    atom_mask=am, atom_feat=self.linear_q(xf), mode="mean"),
                    xf, cl, plm)
            q_xf = xf.data_ptr()
        qlm = arena.as_strided(*shp[3], off[3])
        ai = arena.as_strided(*shp[4], off[4])
        p.k_post(dev, q_xf, q_am, base + off[3] * 2, base + off[8] * 2)
        p.k_ai(dev, base + off[8] * 2, q_am, q_a2t, base + off[4] * 2)
        return ai, qlm, cl, plm



def _xf_run(x, dev, q_a, q_s, q_z, q_am, base, o_st, o_zb, o_og, o_x0, o_x1):
    """Run the fused stack; returns the device pointer of its output rows.

    Buffers ping-pong: block ``i`` reads what block ``i-1`` wrote and writes
    the other slot, which nothing still reads (a program only ever writes its
    own query block, but reads key rows from anywhere in the *input*).
    """
    x.k_sad(dev, q_s, base + o_st * 2)
    x.k_zb(dev, q_z, base + o_zb * 2)
    stride = x.n * x.c * 16          # bytes per transformer block of ST (8 slabs)
    zstride = x.h * x.nk * 2         # bytes per transformer block of ZB
    bufs = (base + o_x0 * 2, base + o_x1 * 2)
    q_og = base + o_og * 2
    src = q_a
    for i in range(x.nblk):
        dst = bufs[i & 1]
        st_i = base + o_st * 2 + i * stride
        x.k_xa[i](dev, src, st_i, base + o_zb * 2 + i * zstride, q_am, q_og)
        x.k_xb[i](dev, src, q_og, st_i, q_am, dst)
        src = dst
    return src


def _drop_plan(module, incompatible_keys):
    """Invalidate the cached weight pack after a state-dict load: the pack is a
    transposed *copy*, so an in-place weight update must rebuild it."""
    module._plan = None


def _cstride(shape):
    st, acc = [], 1
    for s in reversed(shape):
        st.append(acc)
        acc *= s
    return tuple(reversed(st))


class AtomAttentionDecoder(nn.Module):
    """AF3 Algorithm 6: Atom attention decoder.

    Args:
        c_atom: Atom single representation channel dimension
        c_atom_pair: Atom pair representation channel dimension
        c_token: Token diffusion channel dimension
        c_hidden: Per-head hidden dim
        no_heads: Number of attention heads
        no_blocks: Number of transformer blocks
        n_transition: Transition blocks per transformer block
        n_query: Block height
        n_key: Block width
        use_ada_layer_norm: Whether to use AdaLN
    """

    def __init__(
        self,
        c_atom: int = 128,
        c_atom_pair: int = 16,
        c_token: int = 768,
        c_hidden: int = 32,
        no_heads: int = 4,
        no_blocks: int = 3,
        n_transition: int = 2,
        n_query: int = 32,
        n_key: int = 128,
        use_ada_layer_norm: bool = True,
        transformer_cls=None,
    ):
        super().__init__()
        if transformer_cls is None:
            from ..L3.alphafold3_diffusion_transformer import DiffusionTransformer
            transformer_cls = DiffusionTransformer

        self.linear_q_in = Linear(c_token, c_atom, bias=False)

        self.atom_transformer = transformer_cls(
            c_a=c_atom, c_s=c_atom, c_z=c_atom_pair,
            c_hidden=c_hidden, no_heads=no_heads,
            no_blocks=no_blocks, n_transition=n_transition,
            use_ada_layer_norm=use_ada_layer_norm,
            n_query=n_query, n_key=n_key,
        )

        self.layer_norm = LayerNorm(c_atom, create_offset=False)
        self.linear_q_out = Linear(c_atom, 3, bias=False)

        self._plan = None
        self._no_fast = False
        self.register_load_state_dict_post_hook(_drop_plan)

    def _build_plan(self, batch, ai, ql, cl, plm):
        """Validate and cache the two-launch decoder path.  Only 91 us of the
        decoder's 5545 us sits outside the (shared, untouched) transformer, so
        this is a small win by construction -- but it is the same op storm as
        the encoder's and costs two launches to remove."""
        bf = torch.bfloat16
        a2t = batch.get("atom_to_token_index")
        am = batch.get("atom_mask")
        tm = batch.get("token_mask")
        if not (torch.is_tensor(a2t) and torch.is_tensor(am) and torch.is_tensor(tm)
                and torch.is_tensor(ai) and torch.is_tensor(ql)):
            return None
        c = int(self.linear_q_in.weight.shape[0])
        ctk = int(self.linear_q_in.weight.shape[1])
        co = int(self.linear_q_out.weight.shape[0])
        n = int(am.shape[-1])
        t = int(tm.shape[-1])
        if not (0 < n <= 1 << 20 and t > 0 and c >= 16 and co >= 1
                and ai.dtype is bf and ql.dtype is bf and a2t.dtype is torch.int64
                and ai.is_cuda and ai.is_contiguous() and ql.is_contiguous()
                and a2t.is_contiguous() and ai.numel() == t * ctk
                and ql.numel() == n * c and a2t.numel() == n
                and self.linear_q_in.bias is None
                and self.linear_q_out.bias is None
                and self.layer_norm.bias is None
                and self.layer_norm.weight is not None):
            return None

        # Geometry of the (untouched) transformer, so its stack can be fused;
        # anything unexpected leaves p.xf None and the real module is called.
        xfm = self.atom_transformer
        try:
            pb0 = xfm.blocks[0].attention_pair_bias
            nq, nk = int(pb0.n_query), int(pb0.n_key)
        except (AttributeError, IndexError, TypeError):
            nq, nk = 0, 0
        pp = int(plm.shape[-1]) if plm.ndim >= 4 else 0
        nbk = -(-n // nq) if nq > 0 else 0
        probe = None
        if (nq > 0 and nk >= 16 and nk % 2 == 0 and pp >= 1
                and n <= 8192 and am.dtype is bf and am.is_contiguous()
                and am.numel() == n
                and cl.dtype is bf and cl.is_cuda and cl.is_contiguous()
                and cl.numel() == n * c and not (cl.data_ptr() & 15)
                and plm.dtype is bf and plm.is_contiguous()
                and plm.numel() == nbk * nq * nk * pp
                and tuple(plm.shape[-3:]) == (nq, nk, pp)
                and not (plm.data_ptr() & 15)):
            probe = _xf_probe(xfm, c, pp, nq, nk)

        dev = am.get_device()
        p = _Plan()
        p.noisy, p.n, p.c, p.t, p.ct, p.dev = False, n, c, t, ctk, dev
        p.p = co
        p.nq, p.nk, p.nb = nq, nk, nbk
        p.cz = pp
        o_q2 = 0
        o_out = _align(n * c)
        st_n, zb_n, og_n = _xf_sizes(probe, n, c, nq, nk)
        o_st = o_out + _align(n * co)
        o_zb = o_st + _align(st_n)
        o_og = o_zb + _align(zb_n)
        o_x0 = o_og + _align(og_n)
        o_x1 = o_x0 + _align(n * c if probe else 0)
        p.off = (o_q2, o_out, o_st, o_zb, o_x0, o_x1, o_og)
        p.size = o_x1 + _align(n * c if probe else 0)
        q2_shape = tuple(ql.shape[:-1]) + (c,)
        out_shape = tuple(ql.shape[:-1]) + (co,)
        p.shp = ((q2_shape, _cstride(q2_shape)), (out_shape, _cstride(out_shape)))
        wqi = _t2(self.linear_q_in.weight)
        wqo = _t2(self.linear_q_out.weight)
        gln = self.layer_norm.weight.detach().contiguous().to(bf)
        p.w = (wqi, wqo, gln)

        arena = torch.zeros(p.size, device=am.device, dtype=bf)
        q2 = arena.as_strided(*p.shp[0], o_q2)
        out = arena.as_strided(*p.shp[1], o_out)
        bc = triton.next_power_of_2(c)
        p.k_din = _Launch(
            _k_dec_in, -(-n // 32), (ai, ql, a2t, q2, wqi), 4,
            (n, c, ctk, t, 32, bc, 128, -(-ctk // 128)))
        p.k_dout = _Launch(
            _k_dec_out, -(-n // 32), (q2, out, wqo, gln), 2,
            (n, c, co, 32, 128, -(-c // 128),
             max(16, triton.next_power_of_2(co)), float(self.layer_norm.eps)))
        if p.k_din.raw is None or p.k_dout.raw is None:
            return None
        p.xf = _xf_build(xfm, probe, n, c, pp, nq, nk, arena, o_st, o_zb,
                         o_og, o_q2, am)
        return p

    def forward(
        self,
        batch: dict,
        ai: torch.Tensor,
        ql: torch.Tensor,
        cl: torch.Tensor,
        plm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            rl_update: [*, N_atom, 3] atom position updates
        """
        if not self._no_fast and not torch.is_grad_enabled():
            p = self._plan
            am = batch.get("atom_mask")
            if p is None or (am is not None and p.n != am.shape[-1]):
                try:
                    p = self._build_plan(batch, ai, ql, cl, plm)
                except Exception:  # noqa: BLE001 - unsupported shape: use the reference
                    p = None
                self._plan = p
                if p is None:
                    self._no_fast = True
            if p is not None:
                out = self._fused(p, batch, ai, ql, cl, plm)
                if out is not None:
                    return out

        ai_broadcast = _broadcast_token_feat_to_atoms(
            token_mask=batch["token_mask"],
            num_atoms_per_token=batch.get("num_atoms_per_token"),
            token_feat=self.linear_q_in(ai),
            atom_to_token_index=batch.get("atom_to_token_index"),
        )
        ql = ql + ai_broadcast

        ql = self.atom_transformer(
            a=ql, s=cl, z=plm, mask=batch["atom_mask"],
        )

        rl_update = self.linear_q_out(self.layer_norm(ql))

        return rl_update

    def _fused(self, p, batch, ai, ql, cl, plm):
        bf = torch.bfloat16
        am = batch["atom_mask"]
        a2t = batch["atom_to_token_index"]
        n, c, t = p.n, p.c, p.t
        if not (ai.dtype is bf and ql.dtype is bf and a2t.dtype is torch.int64
                and ai.numel() == t * p.ct and ql.numel() == n * c
                and a2t.numel() == n and batch["token_mask"].shape[-1] == t
                and ai.is_contiguous() and ql.is_contiguous()
                and a2t.is_contiguous() and tuple(ql.shape[:-1]) + (c,) == p.shp[0][0]):
            return None
        dev = p.dev
        if am.get_device() != dev or _cur_device() != dev:
            return None
        q_ai, q_ql, q_a2t = ai.data_ptr(), ql.data_ptr(), a2t.data_ptr()
        if (q_ai | q_ql | q_a2t) & 15:
            return None

        q_am = am.data_ptr()
        if p.xf is not None:
            if not (cl.dtype is bf and cl.is_contiguous()
                    and cl.numel() == n * c and plm.dtype is bf
                    and plm.is_contiguous()
                    and plm.numel() == p.nb * p.nq * p.nk * p.cz
                    and am.dtype is bf and am.is_contiguous()
                    and am.numel() == n):
                return None
            q_cl, q_plm = cl.data_ptr(), plm.data_ptr()
            if (q_cl | q_plm | q_am) & 15:
                return None

        arena = torch.empty(p.size, device=am.device, dtype=bf)
        base = arena.data_ptr()
        q2 = arena.as_strided(*p.shp[0], p.off[0])
        p.k_din(dev, q_ai, q_ql, q_a2t, base + p.off[0] * 2)

        if p.xf is not None:
            q_xf = _xf_run(p.xf, dev, base + p.off[0] * 2, q_cl, q_plm, q_am,
                           base, p.off[2], p.off[3], p.off[6], p.off[4],
                           p.off[5])
        else:
            xf = self.atom_transformer(a=q2, s=cl, z=plm, mask=am)
            if (xf.shape != p.shp[0][0] or xf.dtype is not bf
                    or not xf.is_contiguous() or xf.data_ptr() & 15):
                return self.linear_q_out(self.layer_norm(xf))
            q_xf = xf.data_ptr()
        out = arena.as_strided(*p.shp[1], p.off[1])
        p.k_dout(dev, q_xf, base + p.off[1] * 2)
        return out
