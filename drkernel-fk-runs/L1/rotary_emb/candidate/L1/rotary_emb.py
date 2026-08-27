import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


# ---------------------------
# Triton kernels
# ---------------------------

if _TRITON_AVAILABLE:
    @triton.jit
    def _rotary_neox_kernel(
        x_ptr,               # *dtype, shape [M, D]
        cos_ptr, sin_ptr,    # *float32, shape [P, D] (P = num_positions)
        positions_ptr,       # *int32 or *int64, shape [M]
        M: tl.constexpr,     # number of rows (B*S)
        D: tl.constexpr,     # head_dim
        E: tl.constexpr,     # embed = D//2
        P: tl.constexpr,     # num_positions
        BLOCK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)  # row id
        pid_j = tl.program_id(1)  # block id along D

        j = pid_j * BLOCK + tl.arange(0, BLOCK)
        mask = j < D

        row_offset = pid_m * D

        x1 = tl.load(x_ptr + row_offset + j, mask=mask, other=0)
        x2 = tl.load(x_ptr + row_offset + (j + E), mask=mask, other=0)

        pos = tl.load(positions_ptr + pid_m).to(tl.int32)

        cos = tl.load(cos_ptr + pos * D + j, mask=mask, other=0.0)
        sin = tl.load(sin_ptr + pos * D + j, mask=mask, other=0.0)

        cos = cos.to(x1.dtype)
        sin = sin.to(x1.dtype)

        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin

        tl.store(x_ptr + row_offset + j, y1, mask=mask)
        tl.store(x_ptr + row_offset + (j + E), y2, mask=mask)


    @triton.jit
    def _rotary_interleaved_kernel(
        x_ptr,               # *dtype, shape [M, D]
        cos_ptr, sin_ptr,    # *float32, shape [P, D//2]
        positions_ptr,       # *int32 or *int64, shape [M]
        M: tl.constexpr,
        D: tl.constexpr,
        P: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_j = tl.program_id(1)

        j = pid_j * BLOCK + tl.arange(0, BLOCK)
        mask = j < D

        row_offset = pid_m * D

        partner = j ^ 1

        xj = tl.load(x_ptr + row_offset + j, mask=mask, other=0)
        xpart = tl.load(x_ptr + row_offset + partner, mask=mask, other=0)

        pos = tl.load(positions_ptr + pid_m).to(tl.int32)

        k = j // 2
        mask_k = k < (D // 2)

        cos = tl.load(cos_ptr + pos * (D // 2) + k, mask=mask_k, other=0.0)
        sin = tl.load(sin_ptr + pos * (D // 2) + k, mask=mask_k, other=0.0)

        cos = cos.to(xj.dtype)
        sin = sin.to(xj.dtype)

        out_j = xj * cos - xpart * sin
        out_p = xpart * cos + xj * sin

        tl.store(x_ptr + row_offset + j, out_j, mask=mask)
        tl.store(x_ptr + row_offset + partner, out_p, mask=mask)


# ---------------------------
# Python wrapper
# ---------------------------

def _launch_rotary_triton(x: torch.Tensor,
                          cos_sin_cache: torch.Tensor,
                          positions: torch.Tensor,
                          is_neox: bool):
    """
    x: [B, S, D] contiguous
    cos_sin_cache: [P, 2*D] float32
    positions: [B, S] int64 or int32
    is_neox: bool
    Mutates x in-place.
    """
    assert x.is_cuda, "Triton kernel requires CUDA tensor"
    assert x.is_contiguous(), "x must be contiguous"
    assert cos_sin_cache.is_cuda, "cache must be CUDA"
    assert cos_sin_cache.dtype == torch.float32, "cache must be float32"
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32), "unsupported dtype"

    B, S, D = x.shape
    M = B * S
    P = cos_sin_cache.shape[0]
    # cache is [P, 2*D]
    assert cos_sin_cache.shape[1] == 2 * D, f"cache second dim must be 2*D, got {cos_sin_cache.shape[1]} vs 2*{D}"

    x_flat = x.view(M, D)
    pos = positions.contiguous().view(-1)
    pos32 = pos if pos.dtype == torch.int32 else pos.to(torch.int32)

    def _choose_block(d):
        if d >= 256:
            return 256
        elif d >= 128:
            return 128
        else:
            return 64
    BLOCK = _choose_block(D)
    grid = (M, triton.cdiv(D, BLOCK))

    cos = cos_sin_cache[:, :D]
    sin = cos_sin_cache[:, D:]

    if is_neox:
        _rotary_neox_kernel[grid](
            x_flat,
            cos, sin,
            pos32,
            M, D, (D // 2), P,
            BLOCK=BLOCK,
            num_warps=4,
        )
    else:
        _rotary_interleaved_kernel[grid](
            x_flat,
            cos, sin,
            pos32,
            M, D, P,
            BLOCK=BLOCK,
            num_warps=4,
        )


# ---------------------------
# ModelNew: Triton-optimized entry point (shape-flexible)
# ---------------------------

class ModelNew(nn.Module):
    """Rotary position embeddings (RoPE), with optional Llama 3.1-style frequency scaling.

    Notes:
    - Does NOT assert a fixed head_dim in __init__; instead adapts to the runtime
      shapes of query and key in forward.
    - CUDA path: high-performance Triton kernel (in-place).
    - CPU path: clean PyTorch fallback.
    Same forward signature as the original Model: forward(positions, query, key).
    """

    def __init__(
        self,
        head_dim: int | None = None,  # unused; kept for API compatibility
        max_position_embeddings: int,
        rope_theta: float,
        rope_scaling_factor: float = 1.0,
        rope_low_freq_factor: float = 1.0,
        rope_high_freq_factor: float = 1.0,
        rope_original_max_position_embeddings: int | None = None,
        is_neox_style: bool = True,
    ):
        super().__init__()
        # Store parameters but do not hardcode dims
        self.max_position_embeddings = int(max_position_embeddings)
        self.rope_theta = float(rope_theta)
        self.rope_scaling_factor = float(rope_scaling_factor)
        self.rope_low_freq_factor = float(rope_low_freq_factor)
        self.rope_high_freq_factor = float(rope_high_freq_factor)
        self.rope_original_max_position_embeddings = None if rope_original_max_position_embeddings is None else int(rope_original_max_position_embeddings)
        self.is_neox_style = bool(is_neox_style)

    def _build_cache_for_dim(self, D: int) -> torch.Tensor:
        """
        Build cos/sin cache for a given head dimension D.
        Returns shape [P, 2*D] float32 on CUDA (or CPU if tensors are not CUDA).
        """
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        P = self.max_position_embeddings

        # Build inv_freq using D
        half = D // 2
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, half, dtype=torch.float, device=device) * (2.0 / D)))

        # Optional Llama 3.1 scaling
        if (self.rope_scaling_factor != 1.0) and (self.rope_original_max_position_embeddings is not None):
            low_freq_factor = float(self.rope_low_freq_factor)
            high_freq_factor = float(self.rope_high_freq_factor)
            original_max_position_embeddings = int(self.rope_original_max_position_embeddings)

            low_wl = original_max_position_embeddings / low_freq_factor
            high_wl = original_max_position_embeddings / high_freq_factor
            wl = 2 * math.pi / inv_freq
            smooth = torch.zeros_like(inv_freq)
            if low_freq_factor != high_freq_factor:
                smooth = (original_max_position_embeddings / wl - low_freq_factor) / (high_freq_factor - low_freq_factor)

            inv_freq = torch.where(
                wl < high_wl,
                inv_freq,
                torch.where(
                    wl > low_wl,
                    inv_freq / self.rope_scaling_factor,
                    (1 - smooth) * inv_freq / self.rope_scaling_factor + smooth * inv_freq,
                ),
            )

        # Build cache: [P, 2*D] = cat(cos, sin)
        t = torch.arange(P, dtype=torch.float, device=device)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)  # [P, half]
        cos = torch.cos(freqs).float()
        sin = torch.sin(freqs).float()
        cache = torch.cat((cos, sin), dim=-1).contiguous()  # [P, D]
        return cache

    def forward(self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor):
        """
        positions: shape [B, S] (ints)
        query: shape [B, S, Dq]
        key: shape [B, S, Dk] or None
        Returns (query, key).
        """
        # Extract actual dims
        Bq, Sq, Dq = query.shape
        if key is not None:
            Bk, Sk, Dk = key.shape
            if (Bk != Bq) or (Sk != Sq):
                raise ValueError("query and key must have matching batch and seq dims")
            if Dk != Dq:
                # Use separate caches if dims differ
                pass
        else:
            Dk = None

        # Determine device and dtype
        use_cuda = _TRITON_AVAILABLE and query.is_cuda
        query_dev = query.device
        key_dev = key.device if key is not None else query_dev

        # Build or reuse cache for query dim
        if use_cuda:
            # Ensure cache is on CUDA and matches Dq
            cache_q = self.get_buffer(f"cos_sin_cache_q_{Dq}", None)
            if (cache_q is None) or (cache_q.shape[1] != 2 * Dq) or (cache_q.device != query_dev):
                cache_q = self._build_cache_for_dim(Dq).to(query_dev)
                self.register_buffer(f"cos_sin_cache_q_{Dq}", cache_q, persistent=False)
            cos_sin_cache = cache_q
        else:
            # CPU path: build CPU cache
            cos_sin_cache = self._build_cache_for_dim(Dq)

        # Launch for query
        if use_cuda:
            query = query.contiguous()
            _launch_rotary_triton(query, cos_sin_cache, positions, is_neox=self.is_neox_style)
        else:
            # CPU fallback
            D = Dq
            embed = D // 2
            q = query
            pos = positions
            if pos.dim() == 1:
                pos = pos.unsqueeze(1)  # [B,1]
            elif pos.dim() == 2:
                pass
            else:
                raise ValueError("positions must be 1D [S] or 2D [B,S]")
            pos = pos.to(torch.long)

            if self.is_neox_style:
                cos = cos_sin_cache[pos][:, :, :embed]  # [B,S,embed]
                sin = cos_sin_cache[pos][:, :, embed:]  # [B,S,embed]
                q1, q2 = q[..., :embed], q[..., embed:]
                q = torch.cat([q1 * cos - q2 * sin,
                               q2 * cos + q1 * sin], dim=-1)
            else:
                embed_half = D // 2
                cos = cos_sin_cache[pos][:, :, :embed_half]  # [B,S,embed_half]
                sin = cos_sin_cache[pos][:, :, embed_half:]  # [B,S,embed_half]
                q_even, q_odd = q[..., 0::2], q[..., 1::2]
                q_even_rot = q_even * cos - q_odd * sin
                q_odd_rot  = q_odd  * cos + q_even * sin
                q = torch.empty_like(q)
                q[..., 0::2] = q_even_rot
                q[..., 1::2] = q_odd_rot

        # Handle key if present
        if key is not None:
            if use_cuda:
                # Ensure cache is on CUDA and matches Dk
                cache_k = self.get_buffer(f"cos_sin_cache_k_{Dk if Dk is not None else Dq}", None)
                if (cache_k is None) or (cache_k.shape[1] != 2 * Dk) or (cache_k.device != key_dev):
                    cache_k = self._build_cache_for_dim(Dk if Dk is not None else Dq).to(key_dev)
                    self.register_buffer(f"cos_sin_cache_k_{Dk if Dk is not None else Dq}", cache_k, persistent=False)
                cos_sin_cache_k = cache_k
                key_ = key.contiguous()
                _launch_rotary_triton(key_, cos_sin_cache_k, positions, is_neox=self.is_neox_style)
            else:
                # CPU fallback for key
                Dk = key.shape[-1]
                cos_sin_cache_k = self._build_cache_for_dim(Dk)
                D = Dk
                embed = D // 2
                k = key
                pos = positions
                if pos.dim() == 1:
                    pos = pos.unsqueeze(1)  # [B,1]
                elif pos.dim() == 2:
                    pass
                else:
                    raise ValueError("positions must be 1D [S] or 2D [B,S]")
                pos = pos.to(torch.long)

                if self.is_neox_style:
                    cos = cos_sin_cache_k[pos][:, :, :embed]  # [B,S,embed]
                    sin = cos_sin_cache_k[pos][:, :, embed:]  # [B,S,embed]
                    k1, k2 = k[..., :embed], k[..., embed:]
                    k = torch.cat([k1 * cos - k2 * sin,
                                   k2 * cos + k1 * sin], dim=-1)
                else:
                    embed_half = D // 2
                    cos = cos_sin_cache_k[pos][:, :, :embed_half]  # [B,S,embed_half]
                    sin = cos_sin_cache_k[pos][:, :, embed_half:]  # [B,S,embed_half]
                    k_even, k_odd = k[..., 0::2], k[..., 1::2]
                    k_even_rot = k_even * cos - k_odd * sin
                    k_odd_rot  = k_odd  * cos + k_even * sin
                    k = torch.empty_like(k)
                    k[..., 0::2] = k_even_rot
                    k[..., 1::2] = k_odd_rot

        return query, (key if key is not None else None)

RotaryEmbedding = ModelNew
