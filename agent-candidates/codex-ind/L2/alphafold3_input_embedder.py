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
from triton.language.extra import libdevice

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.tensor_ops import OneHot, Pad
from .alphafold3_atom_attention import AtomAttentionEncoder


@triton.jit
def _embed_atoms_kernel(
    ref_pos,
    ref_charge,
    ref_mask,
    ref_element,
    ref_chars,
    w_pos,
    w_charge,
    w_mask,
    w_element,
    w_chars,
    cl,
    N_ATOM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    atoms = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    channels = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_m = atoms < N_ATOM
    valid_n = channels < 128

    k16 = tl.arange(0, 16)
    pos = tl.load(
        ref_pos + atoms[:, None] * 3 + k16[None, :],
        mask=valid_m[:, None] & (k16[None, :] < 3),
        other=0.0,
    )
    wp = tl.load(
        w_pos + channels[None, :] * 3 + k16[:, None],
        mask=valid_n[None, :] & (k16[:, None] < 3),
        other=0.0,
    )
    acc = tl.dot(pos, wp).to(tl.bfloat16)

    charge = libdevice.asinh(
        tl.load(ref_charge + atoms, mask=valid_m, other=0.0).to(tl.float32)
    ).to(tl.bfloat16)
    wc = tl.load(w_charge + channels, mask=valid_n, other=0.0)
    acc = (acc + charge[:, None] * wc[None, :]).to(tl.bfloat16)

    mask_value = tl.load(ref_mask + atoms, mask=valid_m, other=0.0)
    wm = tl.load(w_mask + channels, mask=valid_n, other=0.0)
    acc = (acc + mask_value[:, None] * wm[None, :]).to(tl.bfloat16)

    k128 = tl.arange(0, 128)
    element = tl.load(
        ref_element + atoms[:, None] * 119 + k128[None, :],
        mask=valid_m[:, None] & (k128[None, :] < 119),
        other=0.0,
    )
    we = tl.load(
        w_element + channels[None, :] * 119 + k128[:, None],
        mask=valid_n[None, :] & (k128[:, None] < 119),
        other=0.0,
    )
    acc = (acc + tl.dot(element, we)).to(tl.bfloat16)

    chars0 = tl.load(
        ref_chars + atoms[:, None] * 256 + k128[None, :],
        mask=valid_m[:, None],
    )
    wchars0 = tl.load(
        w_chars + channels[None, :] * 256 + k128[:, None],
        mask=valid_n[None, :],
        other=0.0,
    )
    chars1 = tl.load(
        ref_chars + atoms[:, None] * 256 + 128 + k128[None, :],
        mask=valid_m[:, None],
    )
    wchars1 = tl.load(
        w_chars + channels[None, :] * 256 + 128 + k128[:, None],
        mask=valid_n[None, :],
        other=0.0,
    )
    chars_out = tl.dot(chars0, wchars0) + tl.dot(chars1, wchars1)
    acc = (acc + chars_out).to(tl.bfloat16)
    tl.store(
        cl + atoms[:, None] * 128 + channels[None, :],
        acc,
        mask=valid_m[:, None] & valid_n[None, :],
    )


@triton.jit
def _project_atoms_to_tokens_kernel(
    cl,
    atom_mask,
    token_features,
    w_q,
    s_input,
    BLOCK_N: tl.constexpr,
):
    token = tl.program_id(0)
    pid_n = tl.program_id(1)
    atoms = token * 23 + tl.arange(0, 32)
    atom_valid = tl.arange(0, 32) < 23
    channels = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    channel_valid = channels < 384

    acc = tl.zeros((32, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, 128, 32):
        k = k0 + tl.arange(0, 32)
        x = tl.load(
            cl + atoms[:, None] * 128 + k[None, :],
            mask=atom_valid[:, None],
            other=0.0,
        )
        w = tl.load(
            w_q + channels[None, :] * 128 + k[:, None],
            mask=channel_valid[None, :],
            other=0.0,
        )
        acc += tl.dot(x, w)

    mask = tl.load(atom_mask + atoms, mask=atom_valid, other=0.0)
    projected = tl.maximum(acc, 0.0).to(tl.bfloat16)
    projected = (projected * mask[:, None]).to(tl.bfloat16)
    total = tl.sum(projected.to(tl.float32), axis=0)
    count = tl.maximum(tl.sum(mask.to(tl.float32), axis=0), 1.0)
    tl.store(
        s_input + token * 449 + channels,
        total / count,
        mask=channel_valid,
    )

    if pid_n == 0:
        tail = tl.arange(0, 128)
        source_channel = tl.where(tail < 64, tail, 383)
        values = tl.load(
            token_features + token * 384 + source_channel,
            mask=tail < 65,
            other=0.0,
        )
        tl.store(
            s_input + token * 449 + 384 + tail,
            values,
            mask=tail < 65,
        )


@triton.jit
def _project_single_kernel(
    s_input,
    weight,
    output,
    BLOCK_N: tl.constexpr,
):
    channels = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    tokens = tl.arange(0, 16)
    acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, 512, 32):
        k = k0 + tl.arange(0, 32)
        x = tl.load(
            s_input + tokens[:, None] * 449 + k[None, :],
            mask=k[None, :] < 449,
            other=0.0,
        )
        w = tl.load(
            weight + channels[None, :] * 449 + k[:, None],
            mask=(channels[None, :] < 384) & (k[:, None] < 449),
            other=0.0,
        )
        acc += tl.dot(x, w)
    tl.store(
        output + tokens[:, None] * 384 + channels[None, :],
        acc,
        mask=channels[None, :] < 384,
    )


@triton.jit
def _project_pair_kernel(
    s_input,
    residue_index,
    token_index,
    asym_id,
    entity_id,
    sym_id,
    weight_i,
    weight_j,
    weight_rel,
    output,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pairs = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    channels = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_pair = pairs < 256
    valid_channel = channels < 128
    token_i = pairs // 16
    token_j = pairs % 16

    acc_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_j = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, 512, 32):
        k = k0 + tl.arange(0, 32)
        xi = tl.load(
            s_input + token_i[:, None] * 449 + k[None, :],
            mask=valid_pair[:, None] & (k[None, :] < 449),
            other=0.0,
        )
        xj = tl.load(
            s_input + token_j[:, None] * 449 + k[None, :],
            mask=valid_pair[:, None] & (k[None, :] < 449),
            other=0.0,
        )
        wi = tl.load(
            weight_i + channels[None, :] * 449 + k[:, None],
            mask=valid_channel[None, :] & (k[:, None] < 449),
            other=0.0,
        )
        wj = tl.load(
            weight_j + channels[None, :] * 449 + k[:, None],
            mask=valid_channel[None, :] & (k[:, None] < 449),
            other=0.0,
        )
        acc_i += tl.dot(xi, wi)
        acc_j += tl.dot(xj, wj)

    ri = tl.load(residue_index + token_i, mask=valid_pair, other=0.0)
    rj = tl.load(residue_index + token_j, mask=valid_pair, other=0.0)
    ai = tl.load(asym_id + token_i, mask=valid_pair, other=0.0)
    aj = tl.load(asym_id + token_j, mask=valid_pair, other=0.0)
    ei = tl.load(entity_id + token_i, mask=valid_pair, other=0.0)
    ej = tl.load(entity_id + token_j, mask=valid_pair, other=0.0)
    same_chain = ai == aj
    same_res = ri == rj
    same_entity = ei == ej

    rel_pos = (ri - rj).to(tl.bfloat16)
    rel_pos = (rel_pos + 32.0).to(tl.bfloat16)
    rel_pos = tl.minimum(tl.maximum(rel_pos, 0.0), 64.0)
    rel_pos = tl.where(same_chain, rel_pos, 65.0)

    ti = tl.load(token_index + token_i, mask=valid_pair, other=0.0)
    tj = tl.load(token_index + token_j, mask=valid_pair, other=0.0)
    rel_token = (ti - tj).to(tl.bfloat16)
    rel_token = (rel_token + 32.0).to(tl.bfloat16)
    rel_token = tl.minimum(tl.maximum(rel_token, 0.0), 64.0)
    rel_token = tl.where(same_chain & same_res, rel_token, 65.0)

    syi = tl.load(sym_id + token_i, mask=valid_pair, other=0.0)
    syj = tl.load(sym_id + token_j, mask=valid_pair, other=0.0)
    rel_chain = (syi - syj).to(tl.bfloat16)
    rel_chain = (rel_chain + 2.0).to(tl.bfloat16)
    rel_chain = tl.minimum(tl.maximum(rel_chain, 0.0), 4.0)
    rel_chain = tl.where(same_entity, rel_chain, 5.0)

    acc_rel = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, 160, 32):
        k = k0 + tl.arange(0, 32)
        is_pos = k[None, :] < 66
        is_token = (k[None, :] >= 66) & (k[None, :] < 132)
        is_entity = k[None, :] == 132
        is_chain = (k[None, :] >= 133) & (k[None, :] < 139)
        feature = tl.where(is_pos, rel_pos[:, None] > k[None, :], 0.0)
        feature += tl.where(
            is_token, rel_token[:, None] > (k[None, :] - 66), 0.0
        )
        feature += tl.where(is_entity, same_entity[:, None], 0.0)
        feature += tl.where(
            is_chain, rel_chain[:, None] > (k[None, :] - 133), 0.0
        )
        wr = tl.load(
            weight_rel + channels[None, :] * 139 + k[:, None],
            mask=valid_channel[None, :] & (k[:, None] < 139),
            other=0.0,
        )
        acc_rel += tl.dot(feature.to(tl.bfloat16), wr)

    pair_projection = (
        acc_i.to(tl.bfloat16) + acc_j.to(tl.bfloat16)
    ).to(tl.bfloat16)
    result = (pair_projection + acc_rel.to(tl.bfloat16)).to(tl.bfloat16)
    tl.store(
        output + pairs[:, None] * 128 + channels[None, :],
        result,
        mask=valid_pair[:, None] & valid_channel[None, :],
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
    def _fast_atom_embedding(self, batch: dict) -> torch.Tensor:
        ref = self.atom_attn_enc.ref_atom_feature_embedder
        cl = torch.empty(
            (batch["ref_pos"].shape[0], 368, 128),
            device=batch["ref_pos"].device,
            dtype=batch["ref_pos"].dtype,
        )
        _embed_atoms_kernel[(triton.cdiv(368, 32), triton.cdiv(128, 32))](
            batch["ref_pos"],
            batch["ref_charge"],
            batch["ref_mask"],
            batch["ref_element"],
            batch["ref_atom_name_chars"],
            ref.linear_ref_pos.weight,
            ref.linear_ref_charge.weight,
            ref.linear_ref_mask.weight,
            ref.linear_ref_element.weight,
            ref.linear_ref_atom_chars.weight,
            cl,
            N_ATOM=368,
            BLOCK_M=32,
            BLOCK_N=32,
            num_warps=4,
        )
        return cl

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
        if batch is not None and "ref_pos" in batch:
            cl = self._fast_atom_embedding(batch)
            s_input = torch.empty(
                (*token_features.shape[:-1], self.c_s_input),
                device=token_features.device,
                dtype=token_features.dtype,
            )
            _project_atoms_to_tokens_kernel[(16, triton.cdiv(384, 64))](
                cl,
                batch["atom_mask"],
                token_features,
                self.atom_attn_enc.linear_q[0].weight,
                s_input,
                BLOCK_N=64,
                num_warps=4,
            )
            s = torch.empty(
                (*token_features.shape[:-1], self.c_s),
                device=token_features.device,
                dtype=token_features.dtype,
            )
            _project_single_kernel[(triton.cdiv(384, 64),)](
                s_input,
                self.linear_s.weight,
                s,
                BLOCK_N=64,
                num_warps=4,
            )
            z = torch.empty(
                (*token_features.shape[:-2], 16, 16, self.c_z),
                device=token_features.device,
                dtype=token_features.dtype,
            )
            _project_pair_kernel[(triton.cdiv(256, 32), triton.cdiv(128, 32))](
                s_input,
                batch["residue_index"],
                batch["token_index"],
                batch["asym_id"],
                batch["entity_id"],
                batch["sym_id"],
                self.linear_z_i.weight,
                self.linear_z_j.weight,
                self.linear_relpos.weight,
                z,
                BLOCK_M=32,
                BLOCK_N=32,
                num_warps=8,
            )
            return s_input, s, z
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
