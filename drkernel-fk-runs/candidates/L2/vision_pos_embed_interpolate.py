import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _embed_bilinear_weighted_sum_kernel(
    emb_ptr,        # *T, shape [G^2, H]
    idx_ptr,        # *int32, shape [4, P]
    wgt_ptr,        # *float, shape [4, P]
    out_ptr,        # *T, shape [P, H]
    H: tl.constexpr,
    P: tl.constexpr,   # number of output pixels = h*w
    BLOCK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # program id over pixels
    pid = tl.program_id(0)
    p = pid * BLOCK + tl.arange(0, BLOCK)
    mask_p = p < P

    # loop over H dimension in blocks
    for h_start in range(0, H, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        mask_h = h_offsets < H

        # accumulate in float32 for stability
        out = tl.zeros((BLOCK, BLOCK_H), dtype=tl.float32)

        # sum over 4 neighbors
        for row in range(4):
            # load indices [BLOCK]
            idx = tl.load(idx_ptr + row * P + p, mask=mask_p, other=0).to(tl.int32)
            # load weights [BLOCK]
            w = tl.load(wgt_ptr + row * P + p, mask=mask_p, other=0.0).to(tl.float32)

            # compute flat addresses: idx * H + h_offsets
            ptrs = emb_ptr + idx[:, None] * H + h_offsets[None, :]
            vals = tl.load(ptrs, mask=(mask_p[:, None] & mask_h[None, :]), other=0.0)
            vals_f32 = vals.to(tl.float32)
            out += vals_f32 * w[:, None]

        # store result; out is float32. emb is typically float32, so this matches.
        store_ptrs = out_ptr + p[:, None] * H + h_offsets[None, :]
        tl.store(store_ptrs, out, mask=(mask_p[:, None] & mask_h[None, :]))


class ModelNew(nn.Module):
    def __init__(self, num_position_embeddings: int, hidden_size: int,
                 spatial_merge_size: int):
        super().__init__()
        self._embed = nn.Embedding(num_position_embeddings, hidden_size)
        self.num_grid_per_side = int(num_position_embeddings ** 0.5)
        self.spatial_merge_size = spatial_merge_size
        self.hidden_size = hidden_size

    def forward(
        self,
        grid_thw_list: list[list[int]],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        num_grid = self.num_grid_per_side
        m_size = self.spatial_merge_size
        hidden_dim = self.hidden_size

        outputs = []
        for t, h, w in grid_thw_list:
            G = num_grid

            # Original PyTorch path to compute indices and weights EXACTLY as in the spec
            h_idxs = torch.linspace(0, G - 1, h, dtype=torch.float32, device=device)
            w_idxs = torch.linspace(0, G - 1, w, dtype=torch.float32, device=device)

            h_floor = h_idxs.long()
            w_floor = w_idxs.long()
            h_ceil = torch.clamp(h_floor + 1, max=G - 1)
            w_ceil = torch.clamp(w_floor + 1, max=G - 1)

            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
            h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
            h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")

            w11 = dh_grid * dw_grid
            w10 = dh_grid - w11
            w01 = dw_grid - w11
            w00 = 1 - dh_grid - w01

            h_grid = torch.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
            w_grid = torch.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
            indices = (h_grid * num_grid + w_grid).reshape(4, -1)  # [4, P], long
            # weights from original: shape [4, P, 1]
            weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1).to(dtype=dtype)

            P = indices.shape[-1]

            # Ensure indices are int32 for Triton
            if indices.dtype != torch.int32:
                indices = indices.to(torch.int32)

            # Embedding parameter
            emb = self._embed.weight  # [G^2, H]
            H = emb.shape[1]

            use_triton = TRITON_AVAILABLE and emb.is_cuda

            if not use_triton:
                # Pure PyTorch weighted sum
                embeds = self._embed(indices) * weights
                combined = embeds.sum(dim=0)
                combined = combined.reshape(
                    h // m_size, m_size, w // m_size, m_size, hidden_dim
                ).permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
                repeated = combined.expand(t, -1, -1).reshape(-1, hidden_dim)
                outputs.append(repeated)
                continue

            # Flatten weights to [4, P]
            wgt = weights.reshape(4, P).to(torch.float32)

            # Allocate output [P, H] in embedding dtype
            out = torch.empty((P, H), device=device, dtype=emb.dtype)

            # Launch Triton kernel
            BLOCK = 128
            BLOCK_H = 64
            grid = (triton.cdiv(P, BLOCK),)

            _embed_bilinear_weighted_sum_kernel[grid](
                emb, indices, wgt, out,
                H=H,
                P=P,
                BLOCK=BLOCK,
                BLOCK_H=BLOCK_H,
            )

            # Reshape to [h, w, H]
            out = out.view(h, w, H)

            # Post-processing identical to original
            combined = out.reshape(
                h // m_size, m_size, w // m_size, m_size, hidden_dim
            ).permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
            repeated = combined.expand(t, -1, -1).reshape(-1, hidden_dim)
            outputs.append(repeated)

        return torch.cat(outputs, dim=0)

VisionPosEmbedInterpolate = ModelNew
