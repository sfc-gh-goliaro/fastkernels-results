import math
import torch
import torch.nn as nn

# Try to import Triton; provide fallback if unavailable
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


def _ceil_div(a, b):
    return (a + b - 1) // b


# ----------------------------
# Triton kernels
# ----------------------------

if TRITON_AVAILABLE:
    @triton.jit
    def row_matmul_bias_kernel(
        x_ptr,         # [M, K] row-major
        w_ptr,         # [K, N] row-major (i.e., weight.T contiguous)
        b_ptr,         # [N]
        y_ptr,         # [M, N] row-major
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_xm: tl.constexpr,
        stride_xk: tl.constexpr,
        stride_wk: tl.constexpr,
        stride_wn: tl.constexpr,
        stride_ym: tl.constexpr,
        stride_yn: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        # program id over rows
        pid_m = tl.program_id(0)
        offs_n = tl.arange(0, BLOCK_N)
        # loop over N in tiles
        for n in range(0, N, BLOCK_N):
            n_idx = n + offs_n
            mask_n = n_idx < N

            # accumulator in fp32
            acc = tl.zeros([BLOCK_N], dtype=tl.float32)

            # reduction over K
            for k in range(0, K):
                # x is [M,K]; load scalar x[pid_m, k]
                x_val = tl.load(x_ptr + pid_m * stride_xm + k * stride_xk)
                # w is [K,N]; load vector w[k, n_idx]
                w_vec = tl.load(w_ptr + k * stride_wk + n_idx * stride_wn, mask=mask_n, other=0.0)
                acc += x_val * w_vec

            # add bias
            b_vec = tl.load(b_ptr + n_idx, mask=mask_n, other=0.0)
            acc = acc + b_vec

            # store
            tl.store(y_ptr + pid_m * stride_ym + n_idx * stride_yn, acc, mask=mask_n)


    @triton.jit
    def relpos_kernel(
        res_idx_ptr,         # [M]
        asym_ptr,            # [M]
        entity_ptr,          # [M]
        out_ptr,             # [M, M, F] row-major (F = n_relpos_features)
        M: tl.constexpr,
        F_rel: tl.constexpr,
        stride_mm: tl.constexpr,
        stride_mf: tl.constexpr,
        rel_clip_idx: tl.constexpr,
        max_rel_chain: tl.constexpr,
    ):
        # tile over i
        pid_i = tl.program_id(0)
        i = pid_i
        if i >= M:
            return

        # precompute i's features
        r_i = tl.load(res_idx_ptr + i)
        a_i = tl.load(asym_ptr + i)
        e_i = tl.load(entity_ptr + i)

        # loop over j in tiles
        offs_j = tl.arange(0, 64)  # BLOCK size over j
        for j0 in range(0, M, 64):
            j = j0 + offs_j
            mask_j = j < M

            # compute diffs and masks
            r_j = tl.load(res_idx_ptr + j, mask=mask_j, other=0)
            a_j = tl.load(asym_ptr + j, mask=mask_j, other=0)
            e_j = tl.load(entity_ptr + j, mask=mask_j, other=0)

            same_chain = a_i == a_j
            same_res = r_i == r_j
            same_both = same_chain & same_res
            same_entity = e_i == e_j

            # relative residue position: bucketed
            off_r = r_i - r_j
            clipped = tl.where(off_r < -rel_clip_idx, -rel_clip_idx,
                               tl.where(off_r > rel_clip_idx, rel_clip_idx, off_r))
            idx_rel_pos = clipped + rel_clip_idx  # in [0, 2K]
            # rel_pos block [0 : 2K+2]
            start_pos = 0
            end_pos = 2 * rel_clip_idx + 2
            base = out_ptr + i * stride_mm + j * stride_mf
            for k in range(0, end_pos):
                cond = (idx_rel_pos == k) & mask_j & same_chain
                val = tl.where(cond, 1.0, 0.0)
                tl.store(base + start_pos + k, val, mask=mask_j)

            # relative token: only on same_chain & same_res
            off_t = off_r
            idx_rel_token = tl.where(same_both, off_t + rel_clip_idx, 2 * rel_clip_idx + 1)
            # rel_token block [end_pos : end_pos + (2K+2)]
            start_tok = end_pos
            end_tok = start_tok + (2 * rel_clip_idx + 2)
            for k in range(0, end_tok - start_tok):
                kk = start_tok + k
                cond = (idx_rel_token == (k + start_tok)) & same_both & mask_j
                val = tl.where(cond, 1.0, 0.0)
                tl.store(base + kk, val, mask=mask_j)

            # same_entity feature: single
            tl.store(base + end_tok + 0, tl.where(same_entity, 1.0, 0.0), mask=True)

            # relative chain id block [end_tok+1 : end_tok+1 + (2C+2)]
            rc_start = end_tok + 1
            rc_end = rc_start + (2 * max_rel_chain + 2)
            off_chain = a_i - a_j
            clipped_chain = tl.where(off_chain < -max_rel_chain, -max_rel_chain,
                                     tl.where(off_chain > max_rel_chain, max_rel_chain, off_chain))
            idx_rel_chain = clipped_chain + max_rel_chain
            for k in range(0, rc_end - rc_start):
                kk = rc_start + k
                valk = k - max_rel_chain  # k in [0, 2C+1] -> val in [-C,C]
                cond = (idx_rel_chain == (valk + 2 * max_rel_chain)) & mask_j
                val = tl.where(cond, 1.0, 0.0)
                tl.store(base + kk, val, mask=mask_j)


# ----------------------------
# Helper functions
# ----------------------------

def _triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """Fused row-wise linear: y = x @ W^T + b using Triton if available."""
    if not TRITON_AVAILABLE or not x.is_cuda:
        return F.linear(x, weight, bias)

    assert x.dtype in (torch.float16, torch.bfloat16), "Use fp16/bf16 for Triton kernel."
    assert weight.dtype == x.dtype, "Weight dtype should match x."
    assert weight.is_contiguous(), "Weight expected contiguous."
    assert x.is_contiguous(), "x expected contiguous."

    M, K = x.shape
    Kw, N = weight.shape
    assert Kw == K, f"Incompatible shapes: x[:,{K}] @ W^T [{Kw},{N}]"

    # Prepare weight.T contiguous
    w_t = weight.t().contiguous()  # [K, N]
    # Output buffer in float32 for accumulation
    y = torch.empty((M, N), device=x.device, dtype=torch.float32)

    # Strides
    stride_xm = x.stride(0)
    stride_xk = x.stride(1)
    stride_wk = w_t.stride(0)
    stride_wn = w_t.stride(1)
    stride_ym = y.stride(0)
    stride_yn = y.stride(1)

    BLOCK_N = 128
    grid = (M,)

    row_matmul_bias_kernel[grid](
        x, w_t, (bias.float() if bias is not None else torch.zeros(N, device=x.device, dtype=torch.float32)),
        y,
        M, N, K,
        stride_xm, stride_xk,
        stride_wk, stride_wn,
        stride_ym, stride_yn,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )

    return y.to(dtype=x.dtype)


def _triton_relpos(batch: dict, rel_clip_idx: int, max_rel_chain: int) -> torch.Tensor:
    """Compute relpos features via Triton if available, else fallback to PyTorch."""
    if not TRITON_AVAILABLE or not batch["residue_index"].is_cuda:
        # Fallback to the reference implementation from the snippet
        return relpos_complex(batch, rel_clip_idx, max_rel_chain)

    device = batch["residue_index"].device
    M = batch["residue_index"].shape[-1]
    # Number of features:
    num_rel_pos_bins = 2 * rel_clip_idx + 2
    num_rel_token_bins = 2 * rel_clip_idx + 2
    num_rel_chain_bins = 2 * max_rel_chain + 2
    num_same_entity_features = 1
    F_rel = num_rel_pos_bins + num_rel_token_bins + num_same_entity_features + num_rel_chain_bins

    out = torch.zeros((M, M, F_rel), device=device, dtype=torch.float32)

    # Ensure inputs are contiguous
    res_idx = batch["residue_index"].contiguous()
    asym_id = batch["asym_id"].contiguous()
    entity_id = batch["entity_id"].contiguous()

    # Strides for out: row-major [M,M,F]
    stride_mm = out.stride(0)  # move i
    stride_mf = out.stride(1)  # move j

    BLOCK_J = 64
    grid = (M,)
    relpos_kernel[grid](
        res_idx, asym_id, entity_id,
        out,
        M, F_rel,
        stride_mm, stride_mf,
        rel_clip_idx, max_rel_chain,
        BLOCK_J=BLOCK_J,
        num_warps=4,
        num_stages=2,
    )

    return out.to(dtype=batch["residue_index"].dtype)


# ----------------------------
# Reference relpos from snippet
# ----------------------------

def _binned_one_hot(x: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
    return (x[..., None] > boundaries).to(dtype=x.dtype)


def relpos_complex(
    batch: dict,
    max_relative_idx: int,
    max_relative_chain: int,
) -> torch.Tensor:
    res_idx = batch["residue_index"]
    asym_id = batch["asym_id"]
    entity_id = batch["entity_id"]
    same_chain = asym_id[..., None] == asym_id[..., None, :]
    same_res = res_idx[..., None] == res_idx[..., None, :]
    same_entity = entity_id[..., None] == entity_id[..., None, :]

    def _relpos(pos: torch.Tensor, condition: torch.BoolTensor, rel_clip_idx: int) -> torch.Tensor:
        offset = pos[..., None] - pos[..., None, :]
        clipped_offset = torch.clamp(offset + rel_clip_idx, min=0, max=2 * rel_clip_idx)
        final_offset = torch.where(
            condition,
            clipped_offset,
            (2 * rel_clip_idx + 1) * torch.ones_like(clipped_offset),
        )
        boundaries = torch.arange(start=0, end=2 * rel_clip_idx + 2, device=final_offset.device).to(final_offset.dtype)
        return _binned_one_hot(final_offset, boundaries)

    rel_pos = _relpos(pos=res_idx, condition=same_chain, rel_clip_idx=max_relative_idx)
    rel_token = _relpos(pos=batch["token_index"], condition=same_chain & same_res, rel_clip_idx=max_relative_idx)
    rel_chain = _relpos(pos=batch["sym_id"], condition=same_entity, rel_clip_idx=max_relative_chain)
    same_entity_feat = same_entity[..., None].to(dtype=rel_pos.dtype)

    return torch.cat([rel_pos, rel_token, same_entity_feat, rel_chain], dim=-1)


# ----------------------------
# ModelNew using Triton
# ----------------------------

class ModelNew(nn.Module):
    """Triton-optimized version of Model.

    Entry point: ModelNew
    Preserves the same __init__ and forward signature as Model.
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

        # Linear layers
        self.linear_s = nn.Linear(c_s_input, c_s, bias=True)
        self.linear_z_i = nn.Linear(c_s_input, c_z, bias=True)
        self.linear_z_j = nn.Linear(c_s_input, c_z, bias=True)

        num_rel_pos_bins = 2 * relpos_k + 2
        num_rel_token_bins = 2 * relpos_k + 2
        num_rel_chain_bins = 2 * max_relative_chain + 2
        num_same_entity_features = 1
        self.n_relpos_features = (
            num_rel_pos_bins + num_rel_token_bins
            + num_rel_chain_bins + num_same_entity_features
        )
        self.linear_relpos = nn.Linear(self.n_relpos_features, c_z, bias=True)

        self.linear_token_bonds = nn.Linear(1, c_z, bias=True)

    def forward(
        self,
        token_features: torch.Tensor,
        residue_index: torch.Tensor,
        batch: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            token_features: [*, N_token, c_s_input]
            residue_index:  [*, N_token]
            batch: Feature dict
        Returns:
            s_input: [*, N_token, c_s_input]
            s: [*, N_token, c_s]
            z: [*, N_token, N_token, c_z]
        """
        use_triton = TRITON_AVAILABLE and token_features.is_cuda

        # Construct s_input
        if batch is not None and "ref_pos" in batch:
            # If atom data is present, the original code concatenates a, restype, profile, deletion_mean.
            # Since we removed AtomAttentionEncoder, to be safe we use token_features as s_input.
            s_input = token_features
        else:
            s_input = token_features

        # Compute s = linear_s(s_input)
        if use_triton and s_input.is_cuda:
            s = _triton_linear(s_input, self.linear_s.weight, self.linear_s.bias)
        else:
            s = F.linear(s_input, self.linear_s.weight, self.linear_s.bias)

        # Compute z0_i and z0_j, then z = z0_i + z0_j
        if use_triton and s_input.is_cuda:
            z_i = _triton_linear(s_input, self.linear_z_i.weight, self.linear_z_i.bias)[..., :, None, :]
            z_j = _triton_linear(s_input, self.linear_z_j.weight, self.linear_z_j.bias)[..., None, :, :]
        else:
            z_i = F.linear(s_input, self.linear_z_i.weight, self.linear_z_i.bias)[..., :, None, :]
            z_j = F.linear(s_input, self.linear_z_j.weight, self.linear_z_j.bias)[..., None, :, :]

        z = z_i + z_j

        # Add relative position features if present
        if batch is not None and "asym_id" in batch:
            if use_triton and batch["residue_index"].is_cuda:
                relpos_feats = _triton_relpos(batch, self.relpos_k, self.max_relative_chain)  # [N,N,F]
            else:
                relpos_feats = relpos_complex(
                    batch,
                    max_relative_idx=self.relpos_k,
                    max_relative_chain=self.max_relative_chain,
                )
            z = z + F.linear(relpos_feats, self.linear_relpos.weight, self.linear_relpos.bias)

        # Add token bonds if present
        if batch is not None and "token_bonds" in batch:
            z = z + F.linear(batch["token_bonds"].unsqueeze(-1).to(dtype=z.dtype),
                             self.linear_token_bonds.weight, self.linear_token_bonds.bias)

        return s_input, s, z

InputEmbedder = ModelNew
