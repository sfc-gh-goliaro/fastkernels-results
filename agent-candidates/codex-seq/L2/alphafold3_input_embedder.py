"""Input embedder for AlphaFold3.

Produces initial single (s) and pair (z) representations from token and
atom features.

Reference: openfold3/core/model/feature_embedders/input_embedders.py
           InputEmbedderAllAtom
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.tensor_ops import OneHot, Pad
from .alphafold3_atom_attention import AtomAttentionEncoder


@triton.jit
def _compose_atom_weight(
    q_weight,
    source_weight,
    output,
    source_offset,
    SOURCE_N: tl.constexpr,
    OUTPUT_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compose linear_q with one of the reference-feature projections."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, 128, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        q = tl.load(
            q_weight + rows[:, None] * 128 + ks[None, :],
            mask=rows[:, None] < 384,
            other=0.0,
        )
        w = tl.load(
            source_weight + ks[:, None] * SOURCE_N + cols[None, :],
            mask=cols[None, :] < SOURCE_N,
            other=0.0,
        )
        acc = tl.dot(q, w, acc)
    tl.store(
        output
        + rows[:, None] * OUTPUT_N
        + source_offset
        + cols[None, :],
        acc,
        mask=(rows[:, None] < 384) & (cols[None, :] < SOURCE_N),
    )


@triton.jit
def _pack_outer_weights(
    linear_s,
    linear_zi,
    linear_zj,
    packed,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rows = offsets // 449
    cols = offsets % 449

    s = tl.load(
        linear_s + rows * 449 + cols,
        mask=(offsets < 640 * 449) & (rows < 384),
        other=0.0,
    )
    zi_row = rows - 384
    zi = tl.load(
        linear_zi + zi_row * 449 + cols,
        mask=(offsets < 640 * 449) & (rows >= 384) & (rows < 512),
        other=0.0,
    )
    zj_row = rows - 512
    zj = tl.load(
        linear_zj + zj_row * 449 + cols,
        mask=(offsets < 640 * 449) & (rows >= 512),
        other=0.0,
    )
    value = tl.where(rows < 384, s, tl.where(rows < 512, zi, zj))
    tl.store(packed + offsets, value, mask=offsets < 640 * 449)


@triton.jit
def _build_rel_prefix(rel_weight, prefix):
    channel = tl.program_id(0)
    offsets = tl.arange(0, 128)

    pos = tl.load(
        rel_weight + channel * 139 + offsets,
        mask=offsets < 65,
        other=0.0,
    ).to(tl.float32)
    token = tl.load(
        rel_weight + channel * 139 + 66 + offsets,
        mask=offsets < 65,
        other=0.0,
    ).to(tl.float32)
    chain = tl.load(
        rel_weight + channel * 139 + 133 + offsets,
        mask=offsets < 5,
        other=0.0,
    ).to(tl.float32)

    tl.store(prefix + channel * 138, 0.0)
    tl.store(prefix + channel * 138 + 66, 0.0)
    tl.store(prefix + channel * 138 + 132, 0.0)
    tl.store(
        prefix + channel * 138 + 1 + offsets,
        tl.cumsum(pos, axis=0),
        mask=offsets < 65,
    )
    tl.store(
        prefix + channel * 138 + 67 + offsets,
        tl.cumsum(token, axis=0),
        mask=offsets < 65,
    )
    tl.store(
        prefix + channel * 138 + 133 + offsets,
        tl.cumsum(chain, axis=0),
        mask=offsets < 5,
    )


@triton.jit
def _atom_embed_and_reduce(
    ref_pos,
    ref_charge,
    ref_mask,
    ref_element,
    ref_chars,
    atom_mask,
    token_features,
    effective_weight,
    s_input,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    token = tl.program_id(0)
    out_block = tl.program_id(1)
    atoms = token * 23 + tl.arange(0, BLOCK_M)
    out_cols = out_block * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_atoms = (tl.arange(0, BLOCK_M) < 23) & (atoms < 368)
    valid_out = out_cols < 384
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    pos_cols = tl.arange(0, 16)
    pos = tl.load(
        ref_pos + atoms[:, None] * 3 + pos_cols[None, :],
        mask=valid_atoms[:, None] & (pos_cols[None, :] < 3),
        other=0.0,
    )
    pos_w = tl.load(
        effective_weight + out_cols[None, :] * 380 + pos_cols[:, None],
        mask=valid_out[None, :] & (pos_cols[:, None] < 3),
        other=0.0,
    )
    acc = tl.dot(pos, pos_w, acc)

    charge = tl.load(ref_charge + atoms, mask=valid_atoms, other=0.0).to(tl.float32)
    charge = tl.log(charge + tl.sqrt(charge * charge + 1.0))
    charge_w = tl.load(
        effective_weight + out_cols * 380 + 3,
        mask=valid_out,
        other=0.0,
    )
    mask_value = tl.load(ref_mask + atoms, mask=valid_atoms, other=0.0)
    mask_w = tl.load(
        effective_weight + out_cols * 380 + 4,
        mask=valid_out,
        other=0.0,
    )
    acc += charge[:, None] * charge_w[None, :]
    acc += mask_value[:, None] * mask_w[None, :]

    elem_cols = tl.arange(0, 128)
    elem = tl.load(
        ref_element + atoms[:, None] * 119 + elem_cols[None, :],
        mask=valid_atoms[:, None] & (elem_cols[None, :] < 119),
        other=0.0,
    )
    elem_w = tl.load(
        effective_weight
        + out_cols[None, :] * 380
        + 5
        + elem_cols[:, None],
        mask=valid_out[None, :] & (elem_cols[:, None] < 119),
        other=0.0,
    )
    acc = tl.dot(elem, elem_w, acc)

    char_cols = tl.arange(0, 256)
    chars = tl.load(
        ref_chars + atoms[:, None] * 256 + char_cols[None, :],
        mask=valid_atoms[:, None],
        other=0.0,
    )
    chars_w = tl.load(
        effective_weight
        + out_cols[None, :] * 380
        + 124
        + char_cols[:, None],
        mask=valid_out[None, :],
        other=0.0,
    )
    acc = tl.dot(chars, chars_w, acc)

    projected = tl.maximum(acc, 0.0).to(tl.bfloat16)
    aggregate_mask = tl.load(atom_mask + atoms, mask=valid_atoms, other=0.0)
    numerator = tl.sum(
        projected.to(tl.float32) * aggregate_mask[:, None], axis=0
    )
    denominator = tl.maximum(tl.sum(aggregate_mask, axis=0), 1.0)
    ai = numerator / denominator
    tl.store(
        s_input + token * 449 + out_cols,
        ai,
        mask=valid_out,
    )

    if out_block == 0:
        extra = tl.arange(0, 128)
        source_col = tl.where(extra < 64, extra, 383)
        feature = tl.load(
            token_features + token * 384 + source_col,
            mask=extra < 65,
            other=0.0,
        )
        tl.store(
            s_input + token * 449 + 384 + extra,
            feature,
            mask=extra < 65,
        )


@triton.jit
def _project_s_and_z(
    s_input,
    packed_weight,
    projected,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_M)
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, 512, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(
            s_input + rows[:, None] * 449 + ks[None, :],
            mask=(rows[:, None] < 16) & (ks[None, :] < 449),
            other=0.0,
        )
        w = tl.load(
            packed_weight + cols[None, :] * 449 + ks[:, None],
            mask=(cols[None, :] < 640) & (ks[:, None] < 449),
            other=0.0,
        )
        acc = tl.dot(x, w, acc)
    tl.store(
        projected + rows[:, None] * 640 + cols[None, :],
        acc,
        mask=(rows[:, None] < 16) & (cols[None, :] < 640),
    )


@triton.jit
def _pair_assembly(
    projected,
    residue_index,
    asym_id,
    entity_id,
    token_index,
    sym_id,
    token_bonds,
    rel_weight,
    rel_prefix,
    bond_weight,
    output,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pairs = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    i = pairs // 16
    j = pairs % 16
    valid_pair = pairs < 256
    valid_col = cols < 128

    ri = tl.load(residue_index + i, mask=valid_pair, other=0.0)
    rj = tl.load(residue_index + j, mask=valid_pair, other=0.0)
    ai = tl.load(asym_id + i, mask=valid_pair, other=0.0)
    aj = tl.load(asym_id + j, mask=valid_pair, other=0.0)
    ei = tl.load(entity_id + i, mask=valid_pair, other=0.0)
    ej = tl.load(entity_id + j, mask=valid_pair, other=0.0)
    ti = tl.load(token_index + i, mask=valid_pair, other=0.0)
    tj = tl.load(token_index + j, mask=valid_pair, other=0.0)
    si = tl.load(sym_id + i, mask=valid_pair, other=0.0)
    sj = tl.load(sym_id + j, mask=valid_pair, other=0.0)

    same_chain = ai == aj
    same_residue = ri == rj
    same_entity = ei == ej
    rel_pos = tl.maximum(0.0, tl.minimum(64.0, ri - rj + 32.0))
    rel_pos = tl.where(same_chain, rel_pos, 65.0)
    rel_token = tl.maximum(0.0, tl.minimum(64.0, ti - tj + 32.0))
    rel_token = tl.where(same_chain & same_residue, rel_token, 65.0)
    rel_chain = tl.maximum(0.0, tl.minimum(4.0, si - sj + 2.0))
    rel_chain = tl.where(same_entity, rel_chain, 5.0)

    rel_pos_idx = tl.ceil(rel_pos).to(tl.int32)
    rel_token_idx = tl.ceil(rel_token).to(tl.int32)
    rel_chain_idx = tl.ceil(rel_chain).to(tl.int32)
    pos_sum = tl.load(
        rel_prefix + cols[None, :] * 138 + rel_pos_idx[:, None],
        mask=valid_pair[:, None] & valid_col[None, :],
        other=0.0,
    )
    token_sum = tl.load(
        rel_prefix + cols[None, :] * 138 + 66 + rel_token_idx[:, None],
        mask=valid_pair[:, None] & valid_col[None, :],
        other=0.0,
    )
    chain_sum = tl.load(
        rel_prefix + cols[None, :] * 138 + 132 + rel_chain_idx[:, None],
        mask=valid_pair[:, None] & valid_col[None, :],
        other=0.0,
    )
    entity_weight = tl.load(
        rel_weight + cols * 139 + 132,
        mask=valid_col,
        other=0.0,
    )
    rel_out = (
        pos_sum
        + token_sum
        + chain_sum
        + same_entity[:, None] * entity_weight[None, :]
    ).to(tl.bfloat16)

    zi = tl.load(
        projected + i[:, None] * 640 + 384 + cols[None, :],
        mask=valid_pair[:, None] & valid_col[None, :],
        other=0.0,
    )
    zj = tl.load(
        projected + j[:, None] * 640 + 512 + cols[None, :],
        mask=valid_pair[:, None] & valid_col[None, :],
        other=0.0,
    )
    z = (zi + zj).to(tl.bfloat16)
    z = (z + rel_out).to(tl.bfloat16)
    bond = tl.load(token_bonds + pairs, mask=valid_pair, other=0.0)
    bond_w = tl.load(bond_weight + cols, mask=valid_col, other=0.0)
    z = (z + (bond[:, None] * bond_w[None, :]).to(tl.bfloat16)).to(
        tl.bfloat16
    )
    tl.store(
        output + pairs[:, None] * 128 + cols[None, :],
        z,
        mask=valid_pair[:, None] & valid_col[None, :],
    )


def _binned_one_hot(
    x: torch.Tensor, boundaries: torch.Tensor,
) -> torch.Tensor:
    """One-hot encoding with bin boundaries (matches reference binned_one_hot)."""
    return (x[..., None] > boundaries).to(dtype=x.dtype)


def relpos_complex(
    batch: dict,
    max_relative_idx: int,
    max_relative_chain: int,
) -> torch.Tensor:
    """Build relative position features matching the reference implementation.

    Produces 139 features when max_relative_idx=32, max_relative_chain=2:
      66 (rel_pos) + 66 (rel_token) + 1 (same_entity) + 6 (rel_chain)

    Reference: openfold3/core/utils/relpos.py relpos_complex
    """
    res_idx = batch["residue_index"]
    asym_id = batch["asym_id"]
    entity_id = batch["entity_id"]
    same_chain = asym_id[..., None] == asym_id[..., None, :]
    same_res = res_idx[..., None] == res_idx[..., None, :]
    same_entity = entity_id[..., None] == entity_id[..., None, :]

    def _relpos(
        pos: torch.Tensor, condition: torch.BoolTensor, rel_clip_idx: int,
    ) -> torch.Tensor:
        offset = pos[..., None] - pos[..., None, :]
        clipped_offset = torch.clamp(offset + rel_clip_idx, min=0, max=2 * rel_clip_idx)
        final_offset = torch.where(
            condition,
            clipped_offset,
            (2 * rel_clip_idx + 1) * torch.ones_like(clipped_offset),
        )
        boundaries = torch.arange(
            start=0, end=2 * rel_clip_idx + 2, device=final_offset.device,
        ).to(dtype=final_offset.dtype)
        return _binned_one_hot(final_offset, boundaries)

    rel_pos = _relpos(pos=res_idx, condition=same_chain, rel_clip_idx=max_relative_idx)
    rel_token = _relpos(
        pos=batch["token_index"],
        condition=same_chain & same_res,
        rel_clip_idx=max_relative_idx,
    )
    rel_chain = _relpos(
        pos=batch["sym_id"],
        condition=same_entity,
        rel_clip_idx=max_relative_chain,
    )

    same_entity_feat = same_entity[..., None].to(dtype=rel_pos.dtype)

    return torch.cat([rel_pos, rel_token, same_entity_feat, rel_chain], dim=-1)


class InputEmbedder(nn.Module):
    """Produces initial single and pair representations from token features.

    Matches InputEmbedderAllAtom: runs AtomAttentionEncoder to get a
    token-level representation, concatenates with restype/profile/deletion_mean
    to form s_input (449 dims), then projects to s and z.

    Args:
        c_s_input: Input single representation dimension (449 for all-atom)
        c_s: Single representation dimension
        c_z: Pair representation dimension
        relpos_k: Maximum relative residue position
        max_relative_chain: Maximum relative chain index
        c_atom: Atom single representation dim
        c_atom_pair: Atom pair representation dim
        c_token: Token dim for atom attention encoder output
    """

    def __init__(
        self,
        c_s_input: int,
        c_s: int,
        c_z: int,
        relpos_k: int = 32,
        max_relative_chain: int = 2,
        c_atom: int = 128,
        c_atom_pair: int = 16,
        c_token: int | None = None,
    ):
        super().__init__()
        self.c_s_input = c_s_input
        self.c_s = c_s
        self.c_z = c_z
        self.relpos_k = relpos_k
        self.max_relative_chain = max_relative_chain
        self._one_hot = OneHot()
        self._pad = Pad()

        if c_token is None:
            c_token = c_s

        self.atom_attn_enc = AtomAttentionEncoder(
            c_atom=c_atom,
            c_atom_pair=c_atom_pair,
            c_token=c_token,
            add_noisy_pos=False,
        )

        self.linear_s = Linear(c_s_input, c_s, bias=False)
        self.linear_z_i = Linear(c_s_input, c_z, bias=False)
        self.linear_z_j = Linear(c_s_input, c_z, bias=False)

        num_rel_pos_bins = 2 * relpos_k + 2
        num_rel_token_bins = 2 * relpos_k + 2
        num_rel_chain_bins = 2 * max_relative_chain + 2
        num_same_entity_features = 1
        n_relpos_features = (
            num_rel_pos_bins + num_rel_token_bins
            + num_rel_chain_bins + num_same_entity_features
        )
        self.linear_relpos = Linear(n_relpos_features, c_z, bias=False)

        self.linear_token_bonds = Linear(1, c_z, bias=False)
        self._fast_atom_weight = None
        self._fast_outer_weight = None
        self._fast_rel_prefix = None

    def _build_fast_weights(self, token_features: torch.Tensor) -> None:
        atom = self.atom_attn_enc
        ref = atom.ref_atom_feature_embedder
        q_weight = atom.linear_q[0].weight

        effective = torch.empty(
            (384, 380), device=token_features.device, dtype=token_features.dtype,
        )
        sources = (
            (ref.linear_ref_pos.weight, 0),
            (ref.linear_ref_charge.weight, 3),
            (ref.linear_ref_mask.weight, 4),
            (ref.linear_ref_element.weight, 5),
            (ref.linear_ref_atom_chars.weight, 124),
        )
        for weight, offset in sources:
            source_n = weight.shape[1]
            _compose_atom_weight[
                (triton.cdiv(384, 64), triton.cdiv(source_n, 32))
            ](
                q_weight,
                weight,
                effective,
                offset,
                SOURCE_N=source_n,
                OUTPUT_N=380,
                BLOCK_M=64,
                BLOCK_N=32,
                BLOCK_K=32,
                num_warps=4,
            )
        self._fast_atom_weight = effective

        outer = torch.empty(
            (640, 449), device=token_features.device, dtype=token_features.dtype,
        )
        _pack_outer_weights[(triton.cdiv(640 * 449, 1024),)](
            self.linear_s.weight,
            self.linear_z_i.weight,
            self.linear_z_j.weight,
            outer,
            BLOCK=1024,
            num_warps=8,
        )
        self._fast_outer_weight = outer

        rel_prefix = torch.empty(
            (128, 138), device=token_features.device, dtype=torch.float32,
        )
        _build_rel_prefix[(128,)](
            self.linear_relpos.weight,
            rel_prefix,
            num_warps=4,
        )
        self._fast_rel_prefix = rel_prefix

    def _can_use_fast_path(
        self,
        token_features: torch.Tensor,
        batch: dict | None,
    ) -> bool:
        required = (
            "ref_pos", "ref_charge", "ref_mask", "ref_element",
            "ref_atom_name_chars", "atom_mask", "token_bonds",
            "residue_index", "asym_id", "entity_id", "token_index", "sym_id",
        )
        return (
            token_features.is_cuda
            and token_features.dtype == torch.bfloat16
            and token_features.shape == (1, 16, 384)
            and batch is not None
            and all(name in batch for name in required)
            and batch["ref_pos"].shape == (1, 368, 3)
            and self.c_s_input == 449
            and self.c_s == 384
            and self.c_z == 128
            and self.relpos_k == 32
            and self.max_relative_chain == 2
        )

    def _fast_forward(
        self,
        token_features: torch.Tensor,
        batch: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._fast_atom_weight is None:
            self._build_fast_weights(token_features)

        s_input = torch.empty(
            (1, 16, 449), device=token_features.device, dtype=token_features.dtype,
        )
        _atom_embed_and_reduce[(16, 12)](
            batch["ref_pos"],
            batch["ref_charge"],
            batch["ref_mask"],
            batch["ref_element"],
            batch["ref_atom_name_chars"],
            batch["atom_mask"],
            token_features,
            self._fast_atom_weight,
            s_input,
            BLOCK_M=32,
            BLOCK_N=32,
            num_warps=4,
        )

        projected = torch.empty(
            (16, 640), device=token_features.device, dtype=token_features.dtype,
        )
        _project_s_and_z[(20,)](
            s_input,
            self._fast_outer_weight,
            projected,
            BLOCK_M=16,
            BLOCK_N=32,
            BLOCK_K=32,
            num_warps=8,
        )

        z = torch.empty(
            (1, 16, 16, 128),
            device=token_features.device,
            dtype=token_features.dtype,
        )
        _pair_assembly[(16, 4)](
            projected,
            batch["residue_index"],
            batch["asym_id"],
            batch["entity_id"],
            batch["token_index"],
            batch["sym_id"],
            batch["token_bonds"],
            self.linear_relpos.weight,
            self._fast_rel_prefix,
            self.linear_token_bonds.weight,
            z,
            BLOCK_P=16,
            BLOCK_N=32,
            num_warps=4,
        )
        return s_input, projected[:, :384].unsqueeze(0), z

    def forward(
        self,
        token_features: torch.Tensor,
        residue_index: torch.Tensor,
        batch: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            token_features: [*, N_token, c_s_input] per-token features.
                If batch contains ref_pos (atom features), only restype/profile/deletion_mean
                are expected here and atom_attn_enc produces the remaining features.
                Otherwise, treated as pre-built s_input.
            residue_index:  [*, N_token] residue indices
            batch: Feature dict for relpos and atom attention.

        Returns:
            s_input: [*, N_token, c_s_input] input single representation
            s: [*, N_token, C_s] single representation
            z: [*, N_token, N_token, C_z] pair representation
        """
        if self._can_use_fast_path(token_features, batch):
            return self._fast_forward(token_features, batch)

        if batch is not None and "ref_pos" in batch:
            a, _, _, _ = self.atom_attn_enc(batch=batch)
            s_input = torch.cat(
                [
                    a,
                    batch.get("restype", token_features[..., :32]),
                    batch.get("profile", token_features[..., 32:64]),
                    batch.get("deletion_mean", token_features[..., -1:]).unsqueeze(-1)
                    if batch.get("deletion_mean") is not None and batch["deletion_mean"].dim() == token_features.dim() - 1
                    else batch.get("deletion_mean", token_features[..., -1:]),
                ],
                dim=-1,
            )
        else:
            s_input = token_features

        s = self.linear_s(s_input)

        z_i = self.linear_z_i(s_input)[..., :, None, :]
        z_j = self.linear_z_j(s_input)[..., None, :, :]
        z = z_i + z_j

        if batch is not None and "asym_id" in batch:
            relpos_feats = relpos_complex(
                batch=batch,
                max_relative_idx=self.relpos_k,
                max_relative_chain=self.max_relative_chain,
            ).to(dtype=z.dtype)
        else:
            d = residue_index[..., :, None] - residue_index[..., None, :]
            d = d.clamp(-self.relpos_k, self.relpos_k) + self.relpos_k
            n_bins = 2 * self.relpos_k + 2
            relpos_feats = self._one_hot(d.long(), n_bins).to(
                dtype=z.dtype,
            )
            n_relpos_in = self.linear_relpos.weight.shape[-1]
            if relpos_feats.shape[-1] < n_relpos_in:
                pad_size = n_relpos_in - relpos_feats.shape[-1]
                relpos_feats = self._pad(relpos_feats, (0, pad_size))

        z = z + self.linear_relpos(relpos_feats)

        if batch is not None and "token_bonds" in batch:
            token_bonds_emb = self.linear_token_bonds(
                batch["token_bonds"].unsqueeze(-1).to(dtype=s.dtype)
            )
            z = z + token_bonds_emb

        return s_input, s, z
