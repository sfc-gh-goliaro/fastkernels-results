"""Sequence-local atom attention for AlphaFold3.

AtomAttentionEncoder (Algorithm 5) and AtomAttentionDecoder (Algorithm 6).

Reference: openfold3/core/model/layers/sequence_local_atom_attention.py

Optimized: the captured workload (368 atoms, 128 channels, 3 transformer
blocks) is entirely launch-bound -- the reference path issues ~400 tiny CUDA
kernels per forward for ~1 GFLOP of math.  This file keeps the reference module
tree (so weights load unchanged) but replaces the forward with a handful of
fused Triton kernels: one for the windowing metadata, one for the atom single
conditioning + all AdaLN conditioning projections, one for the whole pair
stack, and one per transformer block (attention + transition + the next
block's per-atom projections).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..L1.relu import ReLU
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.tensor_ops import Pad

try:  # pragma: no cover - triton is present on the bench host
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False


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




# ===========================================================================
# Fused Triton implementation
# ===========================================================================
# Design notes.  The captured problem is tiny (368 atoms x 128 channels, 12
# sequence-local windows) but deep: the reference issues ~400 kernels for ~1
# GFLOP.  Two things dominate on a B200:
#   * launch cost -- so the whole forward is 8-10 kernels;
#   * per-``tl.dot`` fixed cost -- with only a handful of CTAs resident there is
#     nothing to hide the MMA setup latency behind, so a straight-line sequence
#     of small dots costs ~1us *each*.  Every kernel below therefore minimises
#     the number of sequential dots (batching the 4 attention heads into one
#     3-D dot, interleaving the two SwiGLU projections into one wide dot,
#     concatenating the 5 reference-feature projections into one K=512 dot) and
#     spreads work over as many programs as the data allows.
def _np2(n: int) -> int:
    return 1 << max(0, int(n) - 1).bit_length()


if _HAS_TRITON:

    @triton.jit
    def _pro_kernel(
        mask_ptr, a2t_ptr,
        kidx_ptr, kval_ptr, ktok_ptr, mq_ptr, qtok_ptr, cinv_ptr, trng_ptr,
        tin_ptr, twt_ptr, tout_ptr,
        z_ptr, wz_ptr, zo_ptr,
        A, T, NR,
        NB: tl.constexpr, Q: tl.constexpr, K: tl.constexpr,
        ABLK: tl.constexpr, QBLK: tl.constexpr, TBLK: tl.constexpr,
        C: tl.constexpr, CIN: tl.constexpr, CINP: tl.constexpr,
        CZI: tl.constexpr, CZ: tl.constexpr, RBLK: tl.constexpr, TKC: tl.constexpr,
        NEED_CNT: tl.constexpr, TMODE: tl.constexpr, MODE_Z: tl.constexpr,
        EPS: tl.constexpr,
    ):
        """Prologue: sequence-local window metadata, per-token atom counts and
        the token-level trunk projections -- everything that only depends on the
        masks and the token inputs, in one launch."""
        pid = tl.program_id(0)
        if pid < NB:
            # Window key indices.  ``_get_block_key_indices`` mixes int32
            # offsets with a bf16 ``n_real``, so every index is rounded to bf16:
            # above 256 odd indices collapse onto even neighbours and 367 is
            # pushed to 368, which then reads back as *invalid*.  The reference
            # output depends on that, so reproduce the cast sequence exactly.
            b = pid
            offa = tl.arange(0, ABLK)
            m = tl.load(mask_ptr + offa, mask=offa < A, other=0.0).to(tl.float32)
            nreal = tl.sum(m, axis=0).to(tl.bfloat16).to(tl.float32)
            j = tl.arange(0, K)
            first = Q // 2 + b * Q - (K // 2)
            init_f = (first + j).to(tl.float32)
            init_bf = init_f.to(tl.bfloat16).to(tl.float32)
            under = tl.maximum(-tl.min(init_f), 0.0).to(tl.bfloat16).to(tl.float32)
            nm1 = (nreal - 1.0).to(tl.bfloat16).to(tl.float32)
            over = tl.maximum(tl.max(init_bf) - nm1, 0.0).to(tl.bfloat16).to(tl.float32)
            shift = tl.where(under > 0.0, under, -over)
            final = (init_bf + shift).to(tl.bfloat16).to(tl.float32)
            invalid = (final < 0.0) | (final >= nreal)
            safe = tl.minimum(tl.maximum(final, 0.0), tl.maximum(nm1, 0.0)).to(tl.int32)
            mk = tl.load(mask_ptr + safe, mask=safe < A, other=0.0).to(tl.float32)
            idxc = tl.minimum(safe, A - 1)
            tl.store(kidx_ptr + b * K + j, idxc)
            tl.store(kval_ptr + b * K + j, tl.where(invalid, 0.0, mk))
            tl.store(ktok_ptr + b * K + j, tl.load(a2t_ptr + idxc).to(tl.int32))
        elif pid == NB:
            off = tl.arange(0, QBLK)
            om = off < NB * Q
            tl.store(mq_ptr + off,
                     tl.load(mask_ptr + off, mask=om & (off < A), other=0.0).to(tl.float32),
                     mask=om)
            tl.store(qtok_ptr + off,
                     tl.load(a2t_ptr + off, mask=om & (off < A), other=0).to(tl.int32),
                     mask=om)
        elif pid == NB + 1:
            if NEED_CNT:
                off1 = tl.arange(0, ABLK)
                m1 = tl.load(mask_ptr + off1, mask=off1 < A, other=0.0).to(tl.float32)
                tk1 = tl.load(a2t_ptr + off1, mask=off1 < A, other=-1).to(tl.int32)
                tt = tl.arange(0, TBLK)
                eqb = tk1[None, :] == tt[:, None]
                eq = eqb.to(tl.float32)
                cnt = tl.maximum(tl.sum(eq * m1[None, :], axis=1), 1.0)
                tl.store(cinv_ptr + tt, 1.0 / cnt, mask=tt < T)
                # First / last atom index per token, so the aggregation kernel
                # only sweeps the row blocks that can contribute.
                tl.store(trng_ptr + tt,
                         tl.min(tl.where(eqb, off1[None, :], ABLK), axis=1), mask=tt < T)
                tl.store(trng_ptr + TBLK + tt,
                         tl.max(tl.where(eqb, off1[None, :], -1), axis=1), mask=tt < T)
        elif pid == NB + 2:
            if TMODE > 0:
                r = tl.arange(0, TBLK)
                rmt = r < T
                c2 = tl.arange(0, C)
                mut = tl.zeros([TBLK], tl.float32)
                rst = tl.zeros([TBLK], tl.float32) + 1.0
                if TMODE == 1:
                    cp = tl.arange(0, CINP)
                    cpm = cp < CIN
                    xf = tl.load(tin_ptr + r[:, None] * CIN + cp[None, :],
                                 mask=rmt[:, None] & cpm[None, :], other=0.0).to(tl.float32)
                    mut = tl.sum(xf, axis=1) / CIN
                    xcf = tl.where(cpm[None, :], xf - mut[:, None], 0.0)
                    rst = tl.rsqrt(tl.sum(xcf * xcf, axis=1) / CIN + EPS)
                acct = tl.zeros([TBLK, C], tl.float32)
                for k0 in range(0, CIN, TKC):
                    kk = k0 + tl.arange(0, TKC)
                    xx = tl.load(tin_ptr + r[:, None] * CIN + kk[None, :],
                                 mask=rmt[:, None], other=0.0).to(tl.float32)
                    if TMODE == 1:
                        xx = (xx - mut[:, None]) * rst[:, None]
                    acct += tl.dot(xx.to(tl.bfloat16),
                                   tl.load(twt_ptr + kk[:, None] * C + c2[None, :]))
                tl.store(tout_ptr + r[:, None] * C + c2[None, :],
                         acct.to(tl.bfloat16), mask=rmt[:, None])
        else:
            if MODE_Z:
                rows2 = (pid - NB - 3) * RBLK + tl.arange(0, RBLK)
                rm2 = rows2 < NR
                ci = tl.arange(0, CZI)
                x2 = tl.load(z_ptr + rows2[:, None] * CZI + ci[None, :],
                             mask=rm2[:, None], other=0.0).to(tl.float32)
                mu2 = tl.sum(x2, axis=1) / CZI
                xc2 = x2 - mu2[:, None]
                var2 = tl.sum(xc2 * xc2, axis=1) / CZI
                xh2 = (xc2 * tl.rsqrt(var2 + EPS)[:, None]).to(tl.bfloat16)
                cz = tl.arange(0, CZ)
                o2 = tl.dot(xh2, tl.load(wz_ptr + ci[:, None] * CZ + cz[None, :]))
                tl.store(zo_ptr + rows2[:, None] * CZ + cz[None, :],
                         o2.to(tl.bfloat16), mask=rm2[:, None])

    @triton.jit
    def _cl_kernel(
        pos_ptr, chg_ptr, rmsk_ptr, elem_ptr, chars_ptr, wc_ptr, wvec_ptr,
        siproj_ptr, a2t_ptr, rl_ptr, wlm_ptr,
        cl_ptr, ql_ptr, uv_ptr,
        A,
        C: tl.constexpr, CE: tl.constexpr, CN: tl.constexpr, OFC: tl.constexpr,
        KEND: tl.constexpr, KC: tl.constexpr, UV: tl.constexpr, M: tl.constexpr,
        HAS_SI: tl.constexpr, HAS_RL: tl.constexpr,
    ):
        """Atom single conditioning ``cl``, noisy-position ``ql`` and the two
        halves of the pair projection.  ref_element / ref_name_chars share one
        padded operand so the wide part is one dot per K chunk; the five
        one-channel features (ref_pos xyz, charge, mask) and the three noisy
        position channels are rank-1 updates on the accumulator instead of
        their own dots -- each avoided dot is ~3us of MMA latency here."""
        pid = tl.program_id(0)
        rows = pid * M + tl.arange(0, M)
        rm = rows < A
        cc = tl.arange(0, C)

        acc = tl.zeros([M, C], tl.float32)
        for k0 in tl.static_range(0, KEND, KC):
            cp = k0 + tl.arange(0, KC)
            x = tl.load(elem_ptr + rows[:, None] * CE + cp[None, :],
                        mask=rm[:, None] & (cp[None, :] < CE), other=0.0)
            x += tl.load(chars_ptr + rows[:, None] * CN + (cp[None, :] - OFC),
                         mask=rm[:, None] & (cp[None, :] >= OFC) & (cp[None, :] < OFC + CN),
                         other=0.0)
            acc += tl.dot(x, tl.load(wc_ptr + cp[:, None] * C + cc[None, :]))
        for c in tl.static_range(3):
            pv = tl.load(pos_ptr + rows * 3 + c, mask=rm, other=0.0).to(tl.float32)
            acc += pv[:, None] * tl.load(wvec_ptr + c * C + cc)[None, :]
        ch = tl.load(chg_ptr + rows, mask=rm, other=0.0).to(tl.float32)
        ch = tl.log(ch + tl.sqrt(ch * ch + 1.0)).to(tl.bfloat16).to(tl.float32)
        acc += ch[:, None] * tl.load(wvec_ptr + 3 * C + cc)[None, :]
        rk = tl.load(rmsk_ptr + rows, mask=rm, other=0.0).to(tl.float32)
        acc += rk[:, None] * tl.load(wvec_ptr + 4 * C + cc)[None, :]

        if HAS_SI:
            tok = tl.load(a2t_ptr + rows, mask=rm, other=0).to(tl.int32)
            acc += tl.load(siproj_ptr + tok[:, None] * C + cc[None, :],
                           mask=rm[:, None], other=0.0).to(tl.float32)
        clv = acc.to(tl.bfloat16)
        tl.store(cl_ptr + rows[:, None] * C + cc[None, :], clv, mask=rm[:, None])
        if HAS_RL:
            q = acc
            for c in tl.static_range(3):
                rv = tl.load(rl_ptr + rows * 3 + c, mask=rm, other=0.0).to(tl.float32)
                q += rv[:, None] * tl.load(wvec_ptr + (5 + c) * C + cc)[None, :]
            tl.store(ql_ptr + rows[:, None] * C + cc[None, :], q.to(tl.bfloat16), mask=rm[:, None])
        else:
            tl.store(ql_ptr + rows[:, None] * C + cc[None, :], clv, mask=rm[:, None])

        rcl = tl.maximum(clv.to(tl.float32), 0.0).to(tl.bfloat16)
        nuv = tl.arange(0, UV)
        tl.store(uv_ptr + rows[:, None] * UV + nuv[None, :],
                 tl.dot(rcl, tl.load(wlm_ptr + cc[:, None] * UV + nuv[None, :])).to(tl.bfloat16),
                 mask=rm[:, None])

    @triton.jit
    def _pair_kernel(
        pos_ptr, uid_ptr, kidx_ptr, kval_ptr, ktok_ptr, mq_ptr, qtok_ptr,
        wfeat_ptr, w1_ptr, w2_ptr, w3_ptr,
        uv_ptr, zij_ptr, wzf_ptr, plm_ptr, zb_ptr,
        A, NTOK,
        Q: tl.constexpr, K: tl.constexpr, CZ: tl.constexpr, UV: tl.constexpr,
        NZH: tl.constexpr, HAS_Z: tl.constexpr, EPS: tl.constexpr,
    ):
        """Whole atom-pair stack for one (window, query) row of keys: reference
        offsets -> trunk pair -> cl_lm -> 3-layer MLP -> layer_norm -> the
        per-transformer-block attention bias."""
        pid = tl.program_id(0)
        b = pid // Q
        qi = pid
        j = tl.arange(0, K)
        cz = tl.arange(0, CZ)
        c16 = tl.arange(0, 16)

        kidx = tl.load(kidx_ptr + b * K + j)
        bm = tl.load(mq_ptr + qi) * tl.load(kval_ptr + b * K + j)

        dq = tl.load(pos_ptr + qi * 3 + c16, mask=(c16 < 3) & (qi < A), other=0.0).to(tl.float32)
        dk = tl.load(pos_ptr + kidx[:, None] * 3 + c16[None, :],
                     mask=(c16[None, :] < 3), other=0.0).to(tl.float32)
        dd = ((dq[None, :] - dk) * bm[:, None]).to(tl.bfloat16).to(tl.float32)
        uq = tl.load(uid_ptr + qi, mask=qi < A, other=0.0).to(tl.float32)
        uk = tl.load(uid_ptr + kidx).to(tl.float32)
        vlm = tl.where(uq == uk, 1.0, 0.0) * bm
        invsq = 1.0 / (1.0 + tl.sum(dd * dd, axis=1))
        feat = (dd + tl.where(c16[None, :] == 3, invsq[:, None], 0.0)
                + tl.where(c16[None, :] == 4, vlm[:, None], 0.0)).to(tl.bfloat16)
        p = tl.dot(feat, tl.load(wfeat_ptr + c16[:, None] * CZ + cz[None, :])) * vlm[:, None]

        if HAS_Z:
            tq = tl.load(qtok_ptr + qi)
            tk = tl.load(ktok_ptr + b * K + j)
            p += tl.load(zij_ptr + (tq * NTOK + tk)[:, None] * CZ + cz[None, :]).to(tl.float32) * bm[:, None]

        u = tl.load(uv_ptr + qi * UV + cz, mask=qi < A, other=0.0).to(tl.float32)
        v = tl.load(uv_ptr + kidx[:, None] * UV + (CZ + cz)[None, :]).to(tl.float32)
        p += (u[None, :] + v) * bm[:, None]

        h = tl.dot(tl.maximum(p, 0.0).to(tl.bfloat16),
                   tl.load(w1_ptr + cz[:, None] * CZ + cz[None, :]))
        h = tl.dot(tl.maximum(h, 0.0).to(tl.bfloat16),
                   tl.load(w2_ptr + cz[:, None] * CZ + cz[None, :]))
        h = tl.dot(tl.maximum(h, 0.0).to(tl.bfloat16),
                   tl.load(w3_ptr + cz[:, None] * CZ + cz[None, :]))
        pf = (p + h) * bm[:, None]
        tl.store(plm_ptr + (qi * K + j)[:, None] * CZ + cz[None, :], pf.to(tl.bfloat16))

        mu = tl.sum(pf, axis=1) / CZ
        pc = pf - mu[:, None]
        var = tl.sum(pc * pc, axis=1) / CZ
        xh = (pc * tl.rsqrt(var + EPS)[:, None]).to(tl.bfloat16)
        zb = tl.trans(tl.dot(xh, tl.load(wzf_ptr + cz[:, None] * 16 + c16[None, :]))).to(tl.bfloat16)
        tl.store(zb_ptr + (qi * NZH + c16[:, None]) * K + j[None, :], zb,
                 mask=c16[:, None] < NZH)

    @triton.jit
    def _zbias_kernel(
        plm_ptr, wzf_ptr, zb_ptr,
        K: tl.constexpr, CZ: tl.constexpr, NZH: tl.constexpr, EPS: tl.constexpr,
    ):
        """layer_norm(z) + per-block linear_z bias for a supplied pair tensor."""
        qi = tl.program_id(0)
        j = tl.arange(0, K)
        cz = tl.arange(0, CZ)
        c16 = tl.arange(0, 16)
        x = tl.load(plm_ptr + (qi * K + j)[:, None] * CZ + cz[None, :]).to(tl.float32)
        mu = tl.sum(x, axis=1) / CZ
        xc = x - mu[:, None]
        var = tl.sum(xc * xc, axis=1) / CZ
        xh = (xc * tl.rsqrt(var + EPS)[:, None]).to(tl.bfloat16)
        zb = tl.trans(tl.dot(xh, tl.load(wzf_ptr + cz[:, None] * 16 + c16[None, :]))).to(tl.bfloat16)
        tl.store(zb_ptr + (qi * NZH + c16[:, None]) * K + j[None, :], zb,
                 mask=c16[:, None] < NZH)

    @triton.jit
    def _prepw_kernel(
        s_ptr, wp_ptr, bp_ptr, sprep_ptr, A,
        C: tl.constexpr, M: tl.constexpr, EPS: tl.constexpr,
    ):
        """One AdaLN conditioning projection tile.  Grid is (atom tiles) x
        (3 blocks * 8 projections) so every program owns a distinct weight."""
        pid = tl.program_id(0)
        n = tl.program_id(1)
        rows = pid * M + tl.arange(0, M)
        rm = rows < A
        cc = tl.arange(0, C)
        x = tl.load(s_ptr + rows[:, None] * C + cc[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        kind = n % 8
        raw = kind >= 6
        mu = tl.sum(x, axis=1) / C
        xc = x - mu[:, None]
        var = tl.sum(xc * xc, axis=1) / C
        xin = tl.where(raw, x, xc * tl.rsqrt(var + EPS)[:, None]).to(tl.bfloat16)
        o = (tl.dot(xin, tl.load(wp_ptr + (n * C + cc[:, None]) * C + cc[None, :]))
             + tl.load(bp_ptr + n * C + cc)[None, :])
        o = tl.where(raw | (kind % 2 == 0), 1.0 / (1.0 + tl.exp(-o)), o)
        tl.store(sprep_ptr + (n * A + rows[:, None]) * C + cc[None, :],
                 o.to(tl.bfloat16), mask=rm[:, None])

    @triton.jit
    def _qkv_kernel(
        a_ptr, aout_ptr, aip_ptr, a2t_ptr, sprep_ptr,
        wqkv_ptr, bqkv_ptr, qkv_ptr, A,
        C: tl.constexpr, HD: tl.constexpr, M: tl.constexpr,
        ADD_AI: tl.constexpr, EPS: tl.constexpr,
    ):
        """Per-atom q/gate (side 0) and k/v (side 1) projections for block 0."""
        pid = tl.program_id(0)
        side = tl.program_id(1)
        rows = pid * M + tl.arange(0, M)
        rm = rows < A
        cc = tl.arange(0, C)
        a = tl.load(a_ptr + rows[:, None] * C + cc[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        if ADD_AI:
            tok = tl.load(a2t_ptr + rows, mask=rm, other=0).to(tl.int32)
            ab = (a + tl.load(aip_ptr + tok[:, None] * C + cc[None, :],
                              mask=rm[:, None], other=0.0).to(tl.float32)).to(tl.bfloat16)
            if side == 0:
                tl.store(aout_ptr + rows[:, None] * C + cc[None, :], ab, mask=rm[:, None])
            a = ab.to(tl.float32)
        mu = tl.sum(a, axis=1) / C
        ac = a - mu[:, None]
        var = tl.sum(ac * ac, axis=1) / C
        an = (ac * tl.rsqrt(var + EPS)[:, None]).to(tl.bfloat16).to(tl.float32)
        k0 = side * 2
        g = tl.load(sprep_ptr + (k0 * A + rows[:, None]) * C + cc[None, :],
                    mask=rm[:, None], other=0.0).to(tl.float32)
        bb = tl.load(sprep_ptr + ((k0 + 1) * A + rows[:, None]) * C + cc[None, :],
                     mask=rm[:, None], other=0.0).to(tl.float32)
        xin = (g * (an + bb)).to(tl.bfloat16)
        n2 = tl.arange(0, 2 * HD)
        o = (tl.dot(xin, tl.load(wqkv_ptr + (side * C + cc[:, None]) * (2 * HD) + n2[None, :]))
             + tl.load(bqkv_ptr + side * (2 * HD) + n2)[None, :])
        o = tl.where((side == 0) & (n2[None, :] >= HD), 1.0 / (1.0 + tl.exp(-o)), o)
        tl.store(qkv_ptr + ((side * A + rows[:, None]) * (2 * HD)) + n2[None, :],
                 o.to(tl.bfloat16), mask=rm[:, None])

    @triton.jit
    def _attn_kernel(
        qkv_ptr, kidx_ptr, kval_ptr, mq_ptr, zb_ptr, og_ptr, A,
        TI: tl.constexpr, Q: tl.constexpr, K: tl.constexpr, H: tl.constexpr,
        D: tl.constexpr, HD: tl.constexpr, NZH: tl.constexpr, QS: tl.constexpr,
    ):
        """Gated pair-biased softmax attention for one (window, query-tile,
        head).  Kept out of the block kernel because the [QS, K] fp32 score
        tile is what pushes that kernel into register spilling."""
        pid = tl.program_id(0)
        nsub = Q // QS
        h = pid % H
        sub = (pid // H) % nsub
        b = pid // (H * nsub)
        rows = b * Q + sub * QS + tl.arange(0, QS)
        rm = rows < A
        j = tl.arange(0, K)
        hc = h * D + tl.arange(0, D)
        kidx = tl.load(kidx_ptr + b * K + j)
        q = tl.load(qkv_ptr + rows[:, None] * (2 * HD) + hc[None, :], mask=rm[:, None], other=0.0)
        k = tl.load(qkv_ptr + A * 2 * HD + kidx[:, None] * (2 * HD) + hc[None, :])
        sc = tl.dot(q, tl.trans(k))
        sc += 1e9 * (tl.load(mq_ptr + rows)[:, None]
                     * tl.load(kval_ptr + b * K + j)[None, :] - 1.0)
        sc += tl.load(zb_ptr + (rows[:, None] * NZH + (TI * H + h)) * K + j[None, :],
                      mask=rm[:, None], other=0.0).to(tl.float32)
        e = tl.exp(sc - tl.max(sc, axis=1)[:, None])
        pw = (e / tl.sum(e, axis=1)[:, None]).to(tl.bfloat16)
        v = tl.load(qkv_ptr + A * 2 * HD + kidx[:, None] * (2 * HD) + HD + hc[None, :])
        g = tl.load(qkv_ptr + rows[:, None] * (2 * HD) + HD + hc[None, :],
                    mask=rm[:, None], other=0.0).to(tl.float32)
        tl.store(og_ptr + rows[:, None] * HD + hc[None, :],
                 (tl.dot(pw, v) * g).to(tl.bfloat16), mask=rm[:, None])

    @triton.jit
    def _blk_kernel(
        a_ptr, qkv_ptr, og_ptr, sprep_ptr,
        mask_ptr, wo_ptr, wab_ptr, wout_ptr, wqkv2_ptr, bqkv2_ptr,
        tw_ptr, to1_ptr, to2_ptr, A,
        TI: tl.constexpr, C: tl.constexpr, Q: tl.constexpr,
        HD: tl.constexpr, CH: tl.constexpr,
        NEXT: tl.constexpr, TAIL: tl.constexpr,
        CTOK: tl.constexpr, NCT: tl.constexpr, QS: tl.constexpr,
        NCH: tl.constexpr, EPS: tl.constexpr,
    ):
        """One diffusion-transformer block over one sequence-local window:
        gated pair-biased attention (all heads in three batched dots) + the
        AdaLN-Zero SwiGLU transition, then optionally the next block's per-atom
        projections and/or the output head."""
        pid = tl.program_id(0)
        b = pid // (Q // QS)
        rows = b * Q + (pid % (Q // QS)) * QS + tl.arange(0, QS)
        rm = rows < A
        cc = tl.arange(0, C)
        nh = tl.arange(0, HD)
        og = tl.load(og_ptr + rows[:, None] * HD + nh[None, :], mask=rm[:, None], other=0.0)
        oacc = tl.dot(og, tl.load(wo_ptr + nh[:, None] * C + cc[None, :]))

        sp = sprep_ptr + (TI * 8) * A * C
        gada = tl.load(sp + (6 * A + rows[:, None]) * C + cc[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        a0 = tl.load(a_ptr + rows[:, None] * C + cc[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        anew = (a0 + (oacc * gada).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)

        mu = tl.sum(anew, axis=1) / C
        ac = anew - mu[:, None]
        var = tl.sum(ac * ac, axis=1) / C
        an = (ac * tl.rsqrt(var + EPS)[:, None]).to(tl.bfloat16).to(tl.float32)
        gt = tl.load(sp + (4 * A + rows[:, None]) * C + cc[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        bt = tl.load(sp + (5 * A + rows[:, None]) * C + cc[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        gout = tl.load(sp + (7 * A + rows[:, None]) * C + cc[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        x = (gt * (an + bt)).to(tl.bfloat16)
        y = tl.zeros([QS, C], tl.float32)
        for c0 in tl.static_range(0, CH, NCH):
            n2c = 2 * c0 + tl.arange(0, 2 * NCH)
            hab = tl.dot(x, tl.load(wab_ptr + cc[:, None] * (2 * CH) + n2c[None, :]))
            ha, hb = tl.split(tl.reshape(hab, [QS, NCH, 2]))
            ha = ha.to(tl.bfloat16).to(tl.float32)
            hp = ((ha / (1.0 + tl.exp(-ha))).to(tl.bfloat16).to(tl.float32)
                  * hb.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
            chh = c0 + tl.arange(0, NCH)
            y += tl.dot(hp, tl.load(wout_ptr + chh[:, None] * C + cc[None, :]))
        mval = tl.load(mask_ptr + rows, mask=rm, other=0.0).to(tl.float32)
        afin = (anew + (y.to(tl.bfloat16).to(tl.float32) * gout * mval[:, None]).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        tl.store(a_ptr + rows[:, None] * C + cc[None, :], afin, mask=rm[:, None])

        if NEXT:
            af = afin.to(tl.float32)
            mu2 = tl.sum(af, axis=1) / C
            ac2 = af - mu2[:, None]
            var2 = tl.sum(ac2 * ac2, axis=1) / C
            an2 = (ac2 * tl.rsqrt(var2 + EPS)[:, None]).to(tl.bfloat16).to(tl.float32)
            sp2 = sprep_ptr + ((TI + 1) * 8) * A * C
            gq = tl.load(sp2 + (0 * A + rows[:, None]) * C + cc[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
            bq = tl.load(sp2 + (1 * A + rows[:, None]) * C + cc[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
            gk = tl.load(sp2 + (2 * A + rows[:, None]) * C + cc[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
            bk = tl.load(sp2 + (3 * A + rows[:, None]) * C + cc[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
            n2 = tl.arange(0, 2 * HD)
            oq = (tl.dot((gq * (an2 + bq)).to(tl.bfloat16),
                         tl.load(wqkv2_ptr + cc[:, None] * (2 * HD) + n2[None, :]))
                  + tl.load(bqkv2_ptr + n2)[None, :])
            oq = tl.where(n2[None, :] >= HD, 1.0 / (1.0 + tl.exp(-oq)), oq)
            tl.store(qkv_ptr + rows[:, None] * (2 * HD) + n2[None, :], oq.to(tl.bfloat16), mask=rm[:, None])
            ok = tl.dot((gk * (an2 + bk)).to(tl.bfloat16),
                        tl.load(wqkv2_ptr + (C + cc[:, None]) * (2 * HD) + n2[None, :]))
            tl.store(qkv_ptr + (A + rows[:, None]) * (2 * HD) + n2[None, :],
                     ok.to(tl.bfloat16), mask=rm[:, None])

        if TAIL == 1:
            qlv = (afin.to(tl.float32) * mval[:, None]).to(tl.bfloat16)
            tl.store(to1_ptr + rows[:, None] * C + cc[None, :], qlv, mask=rm[:, None])
            for n0 in tl.static_range(0, CTOK, NCT):
                nn = n0 + tl.arange(0, NCT)
                tl.store(to2_ptr + rows[:, None] * CTOK + nn[None, :],
                         tl.maximum(tl.dot(qlv, tl.load(tw_ptr + cc[:, None] * CTOK + nn[None, :])),
                                    0.0).to(tl.bfloat16), mask=rm[:, None])
        elif TAIL == 2:
            af3 = afin.to(tl.float32)
            mu3 = tl.sum(af3, axis=1) / C
            ac3 = af3 - mu3[:, None]
            var3 = tl.sum(ac3 * ac3, axis=1) / C
            an3 = (ac3 * tl.rsqrt(var3 + EPS)[:, None]).to(tl.bfloat16)
            n16 = tl.arange(0, 16)
            r = tl.dot(an3, tl.load(tw_ptr + cc[:, None] * 16 + n16[None, :]))
            tl.store(to1_ptr + rows[:, None] * 3 + n16[None, :], r.to(tl.bfloat16),
                     mask=rm[:, None] & (n16[None, :] < 3))

    @triton.jit
    def _agg_kernel(
        proj_ptr, a2t_ptr, mask_ptr, cinv_ptr, trng_ptr, out_ptr,
        A, TBLK: tl.constexpr, CTOK: tl.constexpr, NC: tl.constexpr,
        RB: tl.constexpr,
    ):
        """Masked mean of the atom projection over each token's atoms, swept
        only over the row blocks between that token's first and last atom."""
        t = tl.program_id(0)
        cols = tl.program_id(1) * NC + tl.arange(0, NC)
        acc = tl.zeros([NC], tl.float32)
        b0 = tl.load(trng_ptr + t) // RB
        b1 = tl.load(trng_ptr + TBLK + t) // RB
        for blk in range(b0, b1 + 1):
            rows = blk * RB + tl.arange(0, RB)
            rmm = rows < A
            tk = tl.load(a2t_ptr + rows, mask=rmm, other=-1).to(tl.int32)
            mv = tl.load(mask_ptr + rows, mask=rmm, other=0.0).to(tl.float32)
            sel = tl.where(tk == t, mv, 0.0)
            pr = tl.load(proj_ptr + rows[:, None] * CTOK + cols[None, :],
                         mask=rmm[:, None], other=0.0).to(tl.float32)
            acc += tl.sum(pr * sel[:, None], axis=0)
        tl.store(out_ptr + t * CTOK + cols, (acc * tl.load(cinv_ptr + t)).to(tl.bfloat16))


_EPS = 1e-5
# Tile / launch shapes, tuned on B200 for the captured 368-atom window.  Small
# row tiles (16 atoms) beat large ones here: the kernels are latency- rather
# than bandwidth-bound, so more resident programs wins even though it re-reads
# the weights.
_PREP_M = 32          # atoms per program in the AdaLN-projection grid
_CL_M = 16            # atoms per program in the cl / qkv kernels
_BLK_QS = 16          # queries per program in the attention / block kernels
_AGG_NC = 128         # output columns per token-aggregation program
_ZRB = 64             # trunk pair rows per prologue program
_NCH = 256            # SwiGLU hidden columns per chunk
_CL_KC = 256          # K chunk for the reference-feature projection
_W_BLK, _W_CL, _W_PAIR, _W_PREP = 8, 8, 4, 4
_W_QKV, _W_PRO, _W_AGG, _W_ATT = 4, 4, 4, 4


def _fold(w: torch.Tensor, lnw: torch.Tensor | None = None) -> torch.Tensor:
    """``[out, in]`` parameter -> ``[in, out]`` bf16 operand, optionally with a
    weight-only layer_norm scale folded in (the norm scales the input, so it
    commutes into the columns of the weight)."""
    x = w.detach().float()
    if lnw is not None:
        x = x * lnw.detach().float()[None, :]
    return x.t().contiguous().to(torch.bfloat16)


def _pow2(n: int) -> bool:
    return n >= 16 and (n & (n - 1)) == 0


def _pack_transformer(tr, C, CZ, H, D, CH, NBLK):
    """Fold every per-block weight of the atom transformer into the layouts the
    fused kernels consume.  Runs once per module -- the weights are static."""
    dev = tr.layer_norm_z.weight.device
    bf = torch.bfloat16
    HD = H * D
    p = {
        "wp": torch.zeros(NBLK * 8, C, C, dtype=bf, device=dev),
        "bp": torch.zeros(NBLK * 8, C, dtype=torch.float32, device=dev),
        "wzf": torch.zeros(CZ, 16, dtype=bf, device=dev),
    }
    for name in ("wqkv", "bqkv", "wo", "wab", "wout"):
        p[name] = []
    lnz = tr.layer_norm_z.weight
    scale = 1.0 / math.sqrt(D)
    for t, blk in enumerate(tr.blocks):
        ap = blk.attention_pair_bias
        ct = blk.conditioned_transition
        base = t * 8
        for gi, ada in enumerate((ap.layer_norm_a_q, ap.layer_norm_a_k, ct.layer_norm)):
            lw = ada.layer_norm_s.weight
            p["wp"][base + 2 * gi] = _fold(ada.linear_g.weight, lw)
            p["bp"][base + 2 * gi] = ada.linear_g.bias.detach().float()
            p["wp"][base + 2 * gi + 1] = _fold(ada.linear_s.weight, lw)
        p["wp"][base + 6] = _fold(ap.linear_ada_out.weight)
        p["bp"][base + 6] = ap.linear_ada_out.bias.detach().float()
        p["wp"][base + 7] = _fold(ct.linear_g.weight)
        p["bp"][base + 7] = ct.linear_g.bias.detach().float()
        mha = ap.mha
        wqkv = torch.empty(2, C, 2 * HD, dtype=bf, device=dev)
        wqkv[0] = torch.cat([mha.linear_q.weight.detach().float() * scale,
                             mha.linear_g.weight.detach().float()], 0).t().to(bf)
        wqkv[1] = torch.cat([mha.linear_k.weight.detach().float(),
                             mha.linear_v.weight.detach().float()], 0).t().to(bf)
        bqkv = torch.zeros(2, 2 * HD, dtype=torch.float32, device=dev)
        bqkv[0, :HD] = mha.linear_q.bias.detach().float() * scale
        p["wqkv"].append(wqkv)
        p["bqkv"].append(bqkv)
        p["wo"].append(_fold(mha.linear_o.weight))
        # SwiGLU: interleave the two projections so one dot produces both and a
        # single tl.split separates them.
        p["wab"].append(torch.stack(
            [_fold(ct.swiglu.linear_a.weight), _fold(ct.swiglu.linear_b.weight)],
            dim=2).reshape(C, 2 * CH).contiguous())
        p["wout"].append(_fold(ct.linear_out.weight))
        p["wzf"][:, t * H:(t + 1) * H] = (
            ap.linear_z.weight.detach().float() * lnz.detach().float()[None, :]
        ).t().to(bf)
    return p


def _tr_ok(tr, NBLK):
    return (getattr(tr, "use_cross_attention", False)
            and hasattr(tr, "layer_norm_z") and hasattr(tr, "blocks")
            and len(tr.blocks) == NBLK
            and all(getattr(b.attention_pair_bias, "use_ada_layer_norm", False)
                    and hasattr(b.attention_pair_bias, "layer_norm_a_q")
                    and hasattr(b, "conditioned_transition") for b in tr.blocks))


def _geom(self, A, T, dev):
    """Window geometry, folded weights and scratch buffers."""
    Q, K = self.n_query, self.n_key
    tr = self.atom_transformer
    if not _tr_ok(tr, len(tr.blocks)):
        return None
    ap0 = tr.blocks[0].attention_pair_bias
    C, CZ = ap0.c_q, ap0.c_z
    H, D = ap0.mha.no_heads, ap0.mha.c_hidden
    CH = tr.blocks[0].conditioned_transition.linear_out.weight.shape[1]
    NBLK = len(tr.blocks)
    NB = -(-A // Q)
    if not (_pow2(Q) and _pow2(K) and _pow2(C) and _pow2(CZ) and _pow2(D)
            and _pow2(CH) and _pow2(H * D) and C % 128 == 0
            and A <= 4096 and NB * Q <= 4096 and T <= 1024 and NBLK * H <= 16):
        return None
    bf, f32 = torch.bfloat16, torch.float32
    g = {
        "A": A, "T": T, "Q": Q, "K": K, "NB": NB, "C": C, "CZ": CZ, "H": H,
        "D": D, "HD": H * D, "CH": CH, "NBLK": NBLK, "NZH": NBLK * H,
        "ABLK": _np2(A), "QBLK": _np2(NB * Q), "TBLK": _np2(T),
        "kidx": torch.empty(NB * K, dtype=torch.int32, device=dev),
        "kval": torch.empty(NB * K, dtype=f32, device=dev),
        "ktok": torch.empty(NB * K, dtype=torch.int32, device=dev),
        "mq": torch.empty(NB * Q, dtype=f32, device=dev),
        "qtok": torch.empty(NB * Q, dtype=torch.int32, device=dev),
        "cinv": torch.empty(_np2(T), dtype=f32, device=dev),
        "trng": torch.empty(2 * _np2(T), dtype=torch.int32, device=dev),
        "sprep": torch.empty(NBLK * 8 * A * C, dtype=bf, device=dev),
        "qkv": torch.empty(2 * A * 2 * H * D, dtype=bf, device=dev),
        "og": torch.empty(A * H * D, dtype=bf, device=dev),
        "zb": torch.empty(NB * Q * NBLK * H * K, dtype=bf, device=dev),
        "nprep": -(-A // _PREP_M), "ncl": -(-A // _CL_M),
        "p": _pack_transformer(tr, C, CZ, H, D, CH, NBLK),
    }
    return g


def _run_transformer(g, a_buf, mask, tail_w, tail_o1, tail_o2, tail_mode,
                     ctok, nct):
    """Three fused transformer-block launches (each also builds the next
    block's per-atom projections, so there is no separate qkv launch)."""
    p = g["p"]
    A, NB, C, Q, K = g["A"], g["NB"], g["C"], g["Q"], g["K"]
    H, D, HD, CH, NBLK = g["H"], g["D"], g["HD"], g["CH"], g["NBLK"]
    zb, sprep, qkv = g["zb"], g["sprep"], g["qkv"]
    kidx, kval, mq = g["kidx"], g["kval"], g["mq"]
    last = NBLK - 1
    og = g["og"]
    nsub = Q // _BLK_QS
    for t in range(NBLK):
        nxt = t < last
        _attn_kernel[(NB * nsub * H,)](
            qkv, kidx, kval, mq, zb, og, A,
            TI=t, Q=Q, K=K, H=H, D=D, HD=HD, NZH=g["NZH"], QS=_BLK_QS,
            num_warps=_W_ATT)
        _blk_kernel[(NB * nsub,)](
            a_buf, qkv, og, sprep, mask,
            p["wo"][t], p["wab"][t], p["wout"][t],
            p["wqkv"][t + 1 if nxt else t], p["bqkv"][t + 1 if nxt else t],
            tail_w, tail_o1, tail_o2,
            A, TI=t, C=C, Q=Q, HD=HD, CH=CH,
            NEXT=nxt, TAIL=0 if nxt else tail_mode, CTOK=ctok, NCT=nct,
            QS=_BLK_QS, NCH=min(CH, _NCH), EPS=_EPS, num_warps=_W_BLK)




def _graph_step(st):
    """Run the fused pipeline, replaying a captured CUDA graph when possible.

    The pipeline is 10-11 tiny kernels whose *launch* cost (~18us each, mostly
    argument marshalling) dwarfs their ~5us of device time, so the forward is
    CPU-bound.  Inputs are copied into fixed staging buffers -- a handful of
    ``copy_`` calls -- and everything downstream is captured once and replayed,
    which turns the launch cost into a single graph launch.  Capture happens on
    the second call so every Triton kernel is already compiled; any failure
    falls back to plain launches for good.
    """
    gph = st["graph"]
    if gph is not None:
        gph.replay()
        return
    launch = st["launch"]
    launch()
    if st["gcount"] < 0:
        return
    st["gcount"] += 1
    if st["gcount"] < 2:
        return
    try:
        torch.cuda.synchronize()
        gph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gph):
            launch()
        gph.replay()
        st["graph"] = gph
    except Exception:
        st["gcount"] = -1
        st["graph"] = None
        launch()


__targets__ = ["AtomAttentionEncoder", "AtomAttentionDecoder"]


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
        self._fk_native = transformer_cls is None
        if transformer_cls is None:
            from ..L3.alphafold3_diffusion_transformer import DiffusionTransformer
            transformer_cls = DiffusionTransformer

        self.n_query = n_query
        self.n_key = n_key
        self.c_token = c_token

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

        self._fk_ok = _HAS_TRITON and self._fk_native and use_ada_layer_norm
        self._fk_cache: dict = {}

    # -- fused path ---------------------------------------------------------
    def _fk_build(self, batch, rl, si_trunk, zij_trunk):
        am = batch.get("atom_mask")
        tm = batch.get("token_mask")
        a2t = batch.get("atom_to_token_index")
        need = ("ref_pos", "ref_charge", "ref_mask", "ref_element",
                "ref_atom_name_chars", "ref_space_uid")
        if not (torch.is_tensor(am) and torch.is_tensor(tm) and torch.is_tensor(a2t)
                and am.is_cuda and am.dtype == torch.bfloat16
                and all(k in batch for k in need)):
            return None
        A, T, dev = am.shape[-1], tm.shape[-1], am.device
        if am.numel() != A or a2t.numel() != A:
            return None
        g = _geom(self, A, T, dev)
        if g is None:
            return None
        C, CZ, NB, Q, K = g["C"], g["CZ"], g["NB"], g["Q"], g["K"]
        emb = self.ref_atom_feature_embedder
        pos = batch["ref_pos"]
        CE = emb.linear_ref_element.weight.shape[1]
        CN = emb.linear_ref_atom_chars.weight.shape[1]
        CTOK = self.linear_q[0].weight.shape[0]
        if (pos.numel() != A * 3 or batch["ref_element"].shape[-1] != CE
                or batch["ref_atom_name_chars"].numel() != A * CN
                or CTOK % _AGG_NC):
            return None
        nz = self.noisy_position_embedder
        has_rl = rl is not None and nz is not None
        if has_rl:
            CS = nz.linear_s.weight.shape[1]
            CZI = nz.linear_z.weight.shape[1]
            if (CS % 128 or not _pow2(CZI) or rl.numel() != A * 3
                    or not torch.is_tensor(si_trunk) or not torch.is_tensor(zij_trunk)
                    or si_trunk.numel() != T * CS or zij_trunk.numel() != T * T * CZI):
                return None
        else:
            CS, CZI = 128, 128
        bf = torch.bfloat16
        OFC = -(-CE // 128) * 128
        KC = min(_CL_KC, 256)
        KEND = -(-(OFC + CN) // KC) * KC
        wc = torch.zeros(KEND, C, dtype=bf, device=dev)
        wc[0:CE] = _fold(emb.linear_ref_element.weight)
        wc[OFC:OFC + CN] = _fold(emb.linear_ref_atom_chars.weight)
        wvec = torch.zeros(8, C, dtype=torch.float32, device=dev)
        wvec[0:3] = emb.linear_ref_pos.weight.detach().float().t()
        wvec[3] = emb.linear_ref_charge.weight.detach().float().reshape(-1)
        wvec[4] = emb.linear_ref_mask.weight.detach().float().reshape(-1)
        wfeat = torch.zeros(16, CZ, dtype=bf, device=dev)
        wfeat[0:3] = _fold(emb.linear_ref_offset.weight)
        wfeat[3:4] = _fold(emb.linear_inv_sq_dists.weight)
        wfeat[4:5] = _fold(emb.linear_valid_mask.weight)
        st = {
            "g": g, "CE": CE, "CN": CN, "OFC": OFC, "KEND": KEND, "KC": KC,
            "CTOK": CTOK, "wvec": wvec,
            "NCT": 256 if CTOK % 256 == 0 else 128,
            "has_rl": has_rl, "CS": CS, "CSP": _np2(CS), "CZI": CZI,
            "TKC": 256 if CS % 256 == 0 else 128,
            "nz": -(-(T * T) // _ZRB),
            "wc": wc, "wfeat": wfeat,
            "w1": _fold(self.pair_mlp[1].weight),
            "w2": _fold(self.pair_mlp[3].weight),
            "w3": _fold(self.pair_mlp[5].weight),
            "wlm": torch.cat([self.linear_l.weight.detach().float(),
                              self.linear_m.weight.detach().float()], 0).t().contiguous().to(bf),
            "wqt": _fold(self.linear_q[0].weight),
            "cl": torch.empty(A * C, dtype=bf, device=dev),
            "ql": torch.empty(A * C, dtype=bf, device=dev),
            "qlo": torch.empty(A * C, dtype=bf, device=dev),
            "uv": torch.empty(A * 2 * CZ, dtype=bf, device=dev),
            "plm": torch.empty(NB * Q * K * CZ, dtype=bf, device=dev),
            "proj": torch.empty(A * CTOK, dtype=bf, device=dev),
            "ai": torch.empty(T * CTOK, dtype=bf, device=dev),
            "graph": None, "gcount": 0,
        }
        # Staging buffers: the bench hands us a different address every call, so
        # copy into fixed storage and keep the kernel-side views constant.
        def stage(src, shape):
            buf = torch.empty(src.shape, dtype=src.dtype, device=dev)
            return buf, buf.reshape(shape)
        st["s_am"], st["v_am"] = stage(am, -1)
        st["s_a2t"], st["v_a2t"] = stage(a2t, -1)
        st["s_pos"], st["v_pos"] = stage(pos, (-1, 3))
        st["s_chg"], st["v_chg"] = stage(batch["ref_charge"], -1)
        st["s_rmk"], st["v_rmk"] = stage(batch["ref_mask"], -1)
        st["s_elem"], st["v_elem"] = stage(batch["ref_element"], (-1, CE))
        st["s_chars"], st["v_chars"] = stage(batch["ref_atom_name_chars"], (-1, CN))
        st["s_uid"], st["v_uid"] = stage(batch["ref_space_uid"], -1)
        st["skeys"] = ("atom_mask", "ref_pos", "ref_charge", "ref_mask",
                       "ref_element", "ref_atom_name_chars", "ref_space_uid")
        st["dsts"] = [st["s_am"], st["s_pos"], st["s_chg"], st["s_rmk"],
                      st["s_elem"], st["s_chars"], st["s_uid"]]
        if has_rl:
            st["s_rl"], st["v_rl"] = stage(rl, (-1, 3))
            st["s_si"], st["v_si"] = stage(si_trunk, (-1, CS))
            st["s_zij"], st["v_zij"] = stage(zij_trunk, (-1, CZI))
            st["dsts"] += [st["s_rl"], st["s_si"], st["s_zij"]]
            wvec[5:8] = nz.linear_r.weight.detach().float().t()
            st["wsi"] = _fold(nz.linear_s.weight, nz.layer_norm_s.weight)
            st["wzij"] = _fold(nz.linear_z.weight, nz.layer_norm_z.weight)
            st["siproj"] = torch.empty(T * C, dtype=bf, device=dev)
            st["zijproj"] = torch.empty(T * T * CZ, dtype=bf, device=dev)
        else:
            st["s_rl"], st["v_rl"] = st["s_pos"], st["v_pos"]
            st["s_si"], st["v_si"] = st["s_pos"], st["v_pos"]
            st["s_zij"], st["v_zij"] = st["s_pos"], st["v_pos"]
            st["wsi"] = wc
            st["wzij"] = wfeat
            st["siproj"] = st["cl"]
            st["zijproj"] = st["uv"]
        cl_shape = tuple(pos.shape[:-1]) + (C,)
        ql_shape = (cl_shape if not has_rl else
                    tuple(torch.broadcast_shapes(torch.Size(cl_shape), rl.shape[:-1] + (C,))))
        st["out"] = (st["ai"].view(tuple(ql_shape[:-2]) + (T, CTOK)),
                     st["qlo"].view(ql_shape),
                     st["cl"].view(cl_shape),
                     st["plm"].view(tuple(pos.shape[:-2]) + (NB, Q, K, CZ)))
        st["launch"] = lambda: self._fk_launch(st)
        return st

    def _fk_launch(self, st):
        g = st["g"]
        A, T, NB, Q = g["A"], g["T"], g["NB"], g["Q"]
        C, CZ, HD, NBLK = g["C"], g["CZ"], g["HD"], g["NBLK"]
        p = g["p"]
        mask, a2t, pos = st["v_am"], st["v_a2t"], st["v_pos"]
        hz = st["has_rl"]
        _pro_kernel[(NB + 3 + (st["nz"] if hz else 0),)](
            mask, a2t, g["kidx"], g["kval"], g["ktok"], g["mq"], g["qtok"], g["cinv"],
            g["trng"], st["v_si"], st["wsi"], st["siproj"],
            st["v_zij"], st["wzij"], st["zijproj"],
            A, T, T * T,
            NB=NB, Q=Q, K=g["K"], ABLK=g["ABLK"], QBLK=g["QBLK"], TBLK=g["TBLK"],
            C=C, CIN=st["CS"], CINP=st["CSP"], CZI=st["CZI"], CZ=CZ, RBLK=_ZRB,
            TKC=st["TKC"], NEED_CNT=True, TMODE=1 if hz else 0, MODE_Z=hz, EPS=_EPS, num_warps=_W_PRO)
        _cl_kernel[(g["ncl"],)](
            pos, st["v_chg"], st["v_rmk"], st["v_elem"], st["v_chars"],
            st["wc"], st["wvec"], st["siproj"], a2t,
            st["v_rl"], st["wlm"], st["cl"], st["ql"], st["uv"],
            A, C=C, CE=st["CE"], CN=st["CN"], OFC=st["OFC"], KEND=st["KEND"],
            KC=st["KC"], UV=2 * CZ, M=_CL_M,
            HAS_SI=hz, HAS_RL=hz, num_warps=_W_CL)
        _pair_kernel[(NB * Q,)](
            pos, st["v_uid"], g["kidx"], g["kval"], g["ktok"], g["mq"], g["qtok"],
            st["wfeat"], st["w1"], st["w2"], st["w3"],
            st["uv"], st["zijproj"], p["wzf"], st["plm"], g["zb"],
            A, T, Q=Q, K=g["K"], CZ=CZ, UV=2 * CZ, NZH=g["NZH"], HAS_Z=hz,
            EPS=_EPS, num_warps=_W_PAIR)
        _prepw_kernel[(g["nprep"], NBLK * 8)](
            st["cl"], p["wp"], p["bp"], g["sprep"], A,
            C=C, M=_PREP_M, EPS=_EPS, num_warps=_W_PREP)
        _qkv_kernel[(g["ncl"], 2)](
            st["ql"], st["ql"], st["cl"], a2t, g["sprep"],
            p["wqkv"][0], p["bqkv"][0], g["qkv"], A,
            C=C, HD=HD, M=_CL_M, ADD_AI=False, EPS=_EPS, num_warps=_W_QKV)
        _run_transformer(g, st["ql"], mask, st["wqt"], st["qlo"], st["proj"],
                         1, st["CTOK"], st["NCT"])
        _agg_kernel[(T, st["CTOK"] // _AGG_NC)](
            st["proj"], a2t, mask, g["cinv"], g["trng"], st["ai"],
            A, TBLK=g["TBLK"], CTOK=st["CTOK"], NC=_AGG_NC, RB=64,
            num_warps=_W_AGG)

    def _fk_run(self, st, batch, rl, si_trunk, zij_trunk):
        srcs = [batch[k] for k in st["skeys"]]
        if st["has_rl"]:
            srcs += [rl, si_trunk, zij_trunk]
        torch._foreach_copy_(st["dsts"], srcs)
        st["s_a2t"].copy_(batch["atom_to_token_index"])
        _graph_step(st)
        return st["out"]

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
        if self._fk_ok:
            am = batch.get("atom_mask")
            sig = (am.shape[-1] if torch.is_tensor(am) else None, rl is not None)
            st = self._fk_cache.get(sig)
            if st is None:
                st = self._fk_build(batch, rl, si_trunk, zij_trunk)
                if st is None:
                    self._fk_ok = False
                else:
                    self._fk_cache[sig] = st
            if st is not None:
                return self._fk_run(st, batch, rl, si_trunk, zij_trunk)
        return self._ref_forward(batch, rl, si_trunk, zij_trunk)

    def _ref_forward(
        self,
        batch: dict,
        rl: torch.Tensor | None = None,
        si_trunk: torch.Tensor | None = None,
        zij_trunk: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        ql = self.atom_transformer(
            a=ql, s=cl, z=plm, mask=atom_mask,
        )

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
        self._fk_native = transformer_cls is None
        if transformer_cls is None:
            from ..L3.alphafold3_diffusion_transformer import DiffusionTransformer
            transformer_cls = DiffusionTransformer

        self.n_query = n_query
        self.n_key = n_key

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

        self._fk_ok = _HAS_TRITON and self._fk_native and use_ada_layer_norm
        self._fk_cache: dict = {}

    def _fk_build(self, batch, ai, ql, cl, plm):
        am = batch.get("atom_mask")
        a2t = batch.get("atom_to_token_index")
        bf = torch.bfloat16
        if not (torch.is_tensor(am) and torch.is_tensor(a2t) and am.is_cuda
                and am.dtype == bf and ql.dtype == bf and cl.dtype == bf
                and plm.dtype == bf and ai.dtype == bf):
            return None
        A, T = am.shape[-1], ai.shape[-2]
        if am.numel() != A or a2t.numel() != A:
            return None
        g = _geom(self, A, T, am.device)
        if g is None:
            return None
        C, CZ, NB, Q, K = g["C"], g["CZ"], g["NB"], g["Q"], g["K"]
        CT = self.linear_q_in.weight.shape[1]
        if (ql.numel() != A * C or cl.numel() != A * C
                or plm.numel() != NB * Q * K * CZ or ai.numel() != T * CT
                or CT % 128):
            return None
        dev = am.device
        wq = torch.zeros(C, 16, dtype=bf, device=dev)
        wq[:, :3] = _fold(self.linear_q_out.weight, self.layer_norm.weight)
        st = {
            "g": g, "CT": CT, "CTP": _np2(CT),
            "TKC": 256 if CT % 256 == 0 else 128,
            "wqin": _fold(self.linear_q_in.weight), "wqout": wq,
            "a": torch.empty(A * C, dtype=bf, device=dev),
            "aip": torch.empty(T * C, dtype=bf, device=dev),
            "rl": torch.empty(A * 3, dtype=bf, device=dev),
            "graph": None, "gcount": 0,
        }

        def stage(src, shape):
            buf = torch.empty(src.shape, dtype=src.dtype, device=dev)
            return buf, buf.reshape(shape)
        st["s_am"], st["v_am"] = stage(am, -1)
        st["s_a2t"], st["v_a2t"] = stage(a2t, -1)
        st["s_ai"], st["v_ai"] = stage(ai, (-1, CT))
        st["s_ql"], st["v_ql"] = stage(ql, -1)
        st["s_cl"], st["v_cl"] = stage(cl, -1)
        st["s_plm"], st["v_plm"] = stage(plm, -1)
        st["dsts"] = [st["s_am"], st["s_ai"], st["s_ql"], st["s_cl"], st["s_plm"]]
        st["out"] = st["rl"].view(tuple(ql.shape[:-1]) + (3,))
        st["launch"] = lambda: self._fk_launch(st)
        return st

    def _fk_launch(self, st):
        g = st["g"]
        A, T, NB, Q = g["A"], g["T"], g["NB"], g["Q"]
        C, CZ, HD, NBLK = g["C"], g["CZ"], g["HD"], g["NBLK"]
        p = g["p"]
        mask, a2t = st["v_am"], st["v_a2t"]
        _pro_kernel[(NB + 3,)](
            mask, a2t, g["kidx"], g["kval"], g["ktok"], g["mq"], g["qtok"], g["cinv"],
            g["trng"], st["v_ai"], st["wqin"], st["aip"], st["v_ai"], st["wqin"], st["aip"],
            A, T, 0,
            NB=NB, Q=Q, K=g["K"], ABLK=g["ABLK"], QBLK=g["QBLK"], TBLK=g["TBLK"],
            C=C, CIN=st["CT"], CINP=st["CTP"], CZI=128, CZ=CZ, RBLK=_ZRB,
            TKC=st["TKC"], NEED_CNT=False, TMODE=2, MODE_Z=False, EPS=_EPS, num_warps=_W_PRO)
        _zbias_kernel[(NB * Q,)](
            st["v_plm"], p["wzf"], g["zb"],
            K=g["K"], CZ=CZ, NZH=g["NZH"], EPS=_EPS, num_warps=_W_PAIR)
        _prepw_kernel[(g["nprep"], NBLK * 8)](
            st["v_cl"], p["wp"], p["bp"], g["sprep"], A,
            C=C, M=_PREP_M, EPS=_EPS, num_warps=_W_PREP)
        _qkv_kernel[(g["ncl"], 2)](
            st["v_ql"], st["a"], st["aip"], a2t, g["sprep"],
            p["wqkv"][0], p["bqkv"][0], g["qkv"], A,
            C=C, HD=HD, M=_CL_M, ADD_AI=True, EPS=_EPS, num_warps=_W_QKV)
        _run_transformer(g, st["a"], mask, st["wqout"], st["rl"], st["rl"], 2, 16, 16)

    def _fk_run(self, st, batch, ai, ql, cl, plm):
        torch._foreach_copy_(st["dsts"], [batch["atom_mask"], ai, ql, cl, plm])
        st["s_a2t"].copy_(batch["atom_to_token_index"])
        _graph_step(st)
        return st["out"]

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
        if self._fk_ok:
            am = batch.get("atom_mask")
            sig = (am.shape[-1] if torch.is_tensor(am) else None, tuple(ai.shape))
            st = self._fk_cache.get(sig)
            if st is None:
                st = self._fk_build(batch, ai, ql, cl, plm)
                if st is None:
                    self._fk_ok = False
                else:
                    self._fk_cache[sig] = st
            if st is not None:
                return self._fk_run(st, batch, ai, ql, cl, plm)
        return self._ref_forward(batch, ai, ql, cl, plm)

    def _ref_forward(
        self,
        batch: dict,
        ai: torch.Tensor,
        ql: torch.Tensor,
        cl: torch.Tensor,
        plm: torch.Tensor,
    ) -> torch.Tensor:
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
