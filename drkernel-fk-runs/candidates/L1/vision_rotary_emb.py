import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


def _ceil_div(a, b):
    return (a + b - 1) // b


@triton.jit
def _gather_rope_features_kernel(
    out_ptr,            # *T, shape [N, DD] where DD = rotary_dim
    pos_ptr,            # *int32, shape [N, 2]; column 0 is t
    cache_ptr,          # *T, shape [M, D] where D = 2 * DD (cos+sin concatenated)
    N: tl.constexpr,
    DD: tl.constexpr,   # rotary_dim (features)
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,  # tile over features
):
    # 2D launch: pid0 over N, pid1 over DD
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    mask_n = offs_n < N
    mask_d = offs_d < DD

    # Load t for these rows (int32)
    t = tl.load(pos_ptr + offs_n * 2 + 0, mask=mask_n, other=0)

    # Base pointer into cache row: t * D
    base = t * (2 * DD)

    # Build 2D indices into cache: [BN, BD]
    idx = base[:, None] + offs_d[None, :]  # -> positions in cache

    # Mask
    mask = (mask_n[:, None]) & (mask_d[None, :])

    # Load values
    vals = tl.load(cache_ptr + idx, mask=mask, other=0.0)

    # Store to out: out is [N, DD]
    out_idx = (offs_n[:, None] * DD) + offs_d[None, :]
    tl.store(out_ptr + out_idx, vals, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, rotary_dim: int, max_grid_size: int = 8192):
        super().__init__()
        D_feat = rotary_dim
        assert D_feat > 0, "rotary_dim must be positive"
        half = D_feat // 2  # not used directly; for clarity

        # Build inv_freq: i = 0..half-1 -> 1 / (10000^(2i / D_feat))
        # We will generate frequencies and then cos/sin; but to match original we can build freqs matrix
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, D_feat, 2, dtype=torch.float32) * (2.0 / D_feat)))
        # Build cache: for t in [0, max_grid_size), compute [cos(t*inv_freq), sin(t*inv_freq)] -> shape [T, 2*half] -> concat to D_feat
        t = torch.arange(max_grid_size, dtype=torch.float32)
        # Compute frequencies matrix [T, half]
        # Note: inv_freq length = half
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # [T, half]

        # Allocate cache [T, 2*half] then concat to D_feat
        cache = torch.empty((max_grid_size, 2 * (D_feat // 2)), dtype=torch.float32)
        # Fill: d = 0..D_feat-1
        for i in range(D_feat // 2):
            cf = torch.cos(freqs[:, i])
            sf = torch.sin(freqs[:, i])
            cache[:, 2 * i] = cf
            cache[:, 2 * i + 1] = sf

        # Alternatively, simpler: stack cos+sin -> [T, 2*half], which equals D_feat
        # But to be robust, keep as [T, 2*half].
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _build_pos_ids_torch(self, grid_thw_list, device):
        """
        Reproduce the original indexing/order using vectorized PyTorch.
        Returns Tensor [N, 2] int64 on 'device'.
        """
        parts = []
        for t, h, w in grid_thw_list:
            dev = device
            x = torch.arange(w, device=dev)
            y = torch.arange(h, device=dev)
            # 2D coords
            xx = x.repeat(h, 1)           # [H, W]
            yy = y.repeat_interleave(w).reshape(h, w)  # [H, W]
            # Flatten to [P, 2] where P = H*W
            pos2d = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
            # If t > 1, tile
            if t > 1:
                pos2d = pos2d.repeat(t, 1)  # [t*P, 2]
            parts.append(pos2d)
        pos_ids = torch.cat(parts, dim=0)  # [N, 2], long
        return pos_ids

    def forward(
        self,
        grid_thw_list: list[list[int]],
        spatial_merge_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1) Build pos_ids [N, 2]int64 via torch (correct order, fast)
        pos_ids = self._build_pos_ids_torch(grid_thw_list, device=device).to(device=device, dtype=torch.int32)

        D_feat = pos_ids.shape[0] =? Not here; use buffer shape inference instead:
        # Get feature dim from buffer: buffer is [M, 2*half] with half = D_feat//2 => D_concat = D_feat
        # But to be robust, inspect:
        cache = self.cos_sin_cache
        D_concat = cache.shape[1]
        D_feat = D_concat // 2  #因为是 cos+sin concatenate, so features = D_concat / 2
        # Actually, original code concatenates cos and sin to make length = 2 * rotary_dim.
        # But it returns only rotary_dim features (split). So D_feat = rotary_dim.
        # However, __init__ takes rotary_dim and builds cache of shape [M, 2*rotary_dim].
        # So D_feat = rotary_dim.
        # We'll use the constructor arg: store it.
        # Amend __init__ to store D_feat.

        # Revised __init__ below stores self.rotary_dim; using it here.

        # The following line assumes self has rotary_dim:
        D_feat = self.rotary_dim

        # 2) Cache to device/dtype
        cache = self.cos_sin_cache.to(device=device, dtype=dtype)
        M, D_concat = cache.shape
        assert M >= int(pos_ids[:, 0].max().item()) + 1, "max_grid_size too small for some t"

        # 3) Allocate outputs [N, D_feat]
        N = pos_ids.shape[0]
        out = torch.empty((N, D_feat), dtype=dtype, device=device)

        # 4) Launch 2D kernel over (N, D_feat)
        BLOCK_N = 128
        BLOCK_D = 128
        grid = (_ceil_div(N, BLOCK_N), _ceil_div(D_feat, BLOCK_D))
        _gather_rope_features_kernel[grid](
            out,                 # [N, D_feat]
            pos_ids,             # [N,2]
            cache,               # [M, 2*D_feat]
            N=N,
            DD=D_feat,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
        )

        # out now contains cache[pos, :D_feat] i.e., cos features concatenated
        # To get sin features, we can run the kernel again for d = D_feat:cache[:, D_feat:2*D_feat]
        # But to avoid extra pass, we can slice cache directly on host:
        # However, that would be another gather; instead, just run once more with offset DD.

        # Easier: modify kernel to accept offset; or run second kernel with base = t * D_concat + DD.
        # I'll add an offset argument.

        # Second pass for sin
        out_sin = torch.empty((N, D_feat), dtype=dtype, device=device)
        BLOCK_N2 = 128
        BLOCK_D2 = 128
        grid2 = (_ceil_div(N, BLOCK_N2), _ceil_div(D_feat, BLOCK_D2))
        _gather_rope_features_kernel[grid2](
            out_sin,                 # [N, D_feat]
            pos_ids,                 # [N,2]
            cache,                   # [M, 2*D_feat]
            N=N,
            DD=D_feat,
            OFFSET=D_feat,           # start at D_feat to get sin
            BLOCK_N=BLOCK_N2,
            BLOCK_D=BLOCK_D2,
        )

        return out, out_sin

# Amend kernel to support OFFSET
@triton.jit
def _gather_rope_features_kernel(
    out_ptr,            # *T, shape [N, DD]
    pos_ptr,            # *int32, shape [N, 2]; column 0 is t
    cache_ptr,          # *T, shape [M, D] where D = 2 * DD
    N: tl.constexpr,
    DD: tl.constexpr,   # rotary_dim (features)
    OFFSET: tl.constexpr = 0,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    mask_n = offs_n < N
    mask_d = offs_d < DD

    t = tl.load(pos_ptr + offs_n * 2 + 0, mask=mask_n, other=0)

    # Total D = 2*DD
    D_total = 2 * DD
    base = t * D_total + OFFSET  # OFFSET = 0 for cos, = DD for sin

    idx = base[:, None] + offs_d[None, :]
    mask = (mask_n[:, None]) & (mask_d[None, :])

    vals = tl.load(cache_ptr + idx, mask=mask, other=0.0)

    out_idx = (offs_n[:, None] * DD) + offs_d[None, :]
    tl.store(out_ptr + out_idx, vals, mask=mask)


# Revised __init__ to store rotary_dim
class ModelNew(nn.Module):
    def __init__(self, rotary_dim: int, max_grid_size: int = 8192):
        super().__init__()
        self.rotary_dim = int(rotary_dim)
        D_feat = self.rotary_dim
        assert D_feat > 0, "rotary_dim must be positive"
        half = D_feat // 2

        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32) * (2.0 / D_feat)))
        t = torch.arange(max_grid_size, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # [T, half]

        cache = torch.empty((max_grid_size, 2 * half), dtype=torch.float32)
        for i in range(half):
            cf = torch.cos(freqs[:, i])
            sf = torch.sin(freqs[:, i])
            cache[:, 2 * i] = cf
            cache[:, 2 * i + 1] = sf

        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _build_pos_ids_torch(self, grid_thw_list, device):
        parts = []
        for t, h, w in grid_thw_list:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            xx = x.repeat(h, 1)
            yy = y.repeat_interleave(w).reshape(h, w)
            pos2d = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
            if t > 1:
                pos2d = pos2d.repeat(t, 1)
            parts.append(pos2d)
        return torch.cat(parts, dim=0)

    def forward(
        self,
        grid_thw_list: list[list[int]],
        spatial_merge_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos_ids = self._build_pos_ids_torch(grid_thw_list, device=device).to(device=device, dtype=torch.int32)

        cache = self.cos_sin_cache.to(device=device, dtype=dtype)
        M, D_concat = cache.shape
        assert M >= int(pos_ids[:, 0].max().item()) + 1, "max_grid_size too small for some t"

        N = pos_ids.shape[0]
        D_feat = self.rotary_dim

        out_cos = torch.empty((N, D_feat), dtype=dtype, device=device)
        out_sin = torch.empty((N, D_feat), dtype=dtype, device=device)

        BLOCK_N = 128
        BLOCK_D = 128
        grid = (_ceil_div(N, BLOCK_N), _ceil_div(D_feat, BLOCK_D))
        # cos part: OFFSET = 0
        _gather_rope_features_kernel[grid](
            out_cos,
            pos_ids,
            cache,
            N=N,
            DD=D_feat,
            OFFSET=0,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
        )
        # sin part: OFFSET = D_feat
        _gather_rope_features_kernel[grid](
            out_sin,
            pos_ids,
            cache,
            N=N,
            DD=D_feat,
            OFFSET=D_feat,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
        )

        return out_cos, out_sin

VisionRotaryEmbedding = ModelNew
