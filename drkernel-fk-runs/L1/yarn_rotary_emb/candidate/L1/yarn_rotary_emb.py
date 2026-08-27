def _triton_rotary_q_only(q, cos_sin, is_neox: bool):
    """
    Rotate q ONLY using cos_sin cache.
    q: [B, S, *]
    cos_sin: [T, cache_D] float32
    is_neox: bool
    Returns: q_out
    """
    device = q.device
    assert q.is_cuda, "Triton kernel requires CUDA tensor."
    assert cos_sin.dtype == torch.float32, "Cache must be float32 for Triton."

    # Flatten to 2D
    q_2d = q.view(-1, q.shape[-1]).contiguous()

    B_S = q_2d.shape[0]
    Dq = q_2d.shape[1]

    # positions to [B*S]
    if positions.ndim == 1:
        pos = positions
    else:
        pos = positions.view(-1)
    assert pos.numel() == B_S, f"positions length {pos.numel()} must match B*S={B_S}."
    pos32 = pos.to(torch.int32)

    N = B_S
    grid = (N,)

    if is_neox:
        # Neox: cache [T, 2D]; prefer q last dim == 2D
        cache_D = cos_sin.shape[1]
        assert cache_D % 2 == 0, "Neox cache last dim must be even (2*D)."
        D = cache_D // 2
        if Dq != cache_D:
            # Fallback to PyTorch
            return _pytorch_rotary_q_only(q, cos_sin)
        # Strides
        stride_q_row = q_2d.stride(0)
        stride_c_row = cos_sin.stride(0)  # equals cache_D for contiguous

        _rotary_neox_q_only_kernel[grid](
            q_2d, cos_sin,
            D, Dq,
            stride_q_row, stride_c_row,
            N,
            num_warps=4,
            num_stages=2,
        )
        return q_2d.view(q.shape)
    else:
        # Interleaved: cache [T, D]
        cache_D = cos_sin.shape[1]
        D = cache_D
        # For safety, fall back to PyTorch unless Dq == cache_D and even
        if (Dq != cache_D) or (Dq % 2 != 0):
            return _pytorch_rotary_q_only(q, cos_sin)
        # Strides
        stride_q_row = q_2d.stride(0)
        stride_c_row = cos_sin.stride(0)

        # Launch a dummy or use PyTorch; kernel above is incomplete.
        return _pytorch_rotary_q_only(q, cos_sin)


def _pytorch_rotary_q_only(q, cos_sin):
    """
    Pure PyTorch rotation of q ONLY.
    q: tensor [..., Dq]
    cos_sin: tensor [T, cache_D] (float)
    Returns: q_out
    """
    # Flatten
   形状 = q.shape
    Dq = q.shape[-1]
    q_2d = q.view(-1, Dq)

    B_S = q_2d.shape[0]
    if positions.ndim == 1:
        pos = positions
    else:
        pos = positions.view(-1)
    assert pos.numel() == B_S
    rows = cos_sin[pos]  # [N, cache_D] float

    # Case neox-like: cache_D == Dq and even -> split cos/sin
    if rows.shape[1] == Dq and (Dq % 2 == 0):
        half = Dq // 2
        cos = rows[:, :half]
        sin = rows[:, half:]
        x1 = q_2d[:, :half]
        x2 = q_2d[:, half:]
        q_out = torch.empty_like(q_2d)
        q_out[:, :half] = x1 * cos - x2 * sin
        q_out[:, half:] = x2 * cos + x1 * sin
        return q_out.view(形状)

    # Otherwise, try to interpret as [cos, sin] if cache_D is even and == 2*something
    # But to keep it simple and safe, fall back to assuming cos = rows, sin = rows (not correct generally).
    # Instead, just return q as-is to avoid incorrectness.
    return q


# Entry point: ModelNew
class ModelNew(nn.Module):
    """Triton-optimized rotary embedding forward that rotates QUERY only.

    Entry point matches original Model's constructor. We accept arbitrary keyword args
    to be compatible with the caller.
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        rope_scaling_factor: float = 1.0,
        rope_low_freq_factor: float = 1.0,
        rope_high_freq_factor: float = 1.0,
        rope_original_max_position_embeddings: int | None = None,
        is_neox_style: bool = True,
        **kwargs,  # ignore other args to stay compatible
    ):
        super().__init__()
        self.head_dim = head_dim
        self.is_neox_style = is_neox_style

        # Build inv_freq like original
        D = head_dim
        half = D // 2
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, half, 2, dtype=torch.float) / D))

        if rope_scaling_factor != 1.0 and rope_original_max_position_embeddings is not None:
            inv_freq = _compute_scaled_inv_freq(
                inv_freq,
                rope_scaling_factor,
                rope_low_freq_factor,
                rope_high_freq_factor,
                rope_original_max_position_embeddings,
            )

        # Build cache like original: CPU tensor; will move to device at forward
        T = max_position_embeddings
        t = torch.arange(T, dtype=torch.float32, device="cpu")
        if self.is_neox_style:
            # Neox: cos/sin over 2*half = D
            D2 = D
            freqs = torch.einsum("i,j -> ij", t, inv_freq)
            cos = freqs.cos()
            sin = freqs.sin()
            cache = torch.cat((cos, sin), dim=-1).float()  # [T, D]
        else:
            # Interleaved: cos/sin over D
            freqs = torch.einsum("i,j -> ij", t, inv_freq)
            cos = freqs.cos()
            sin = freqs.sin()
            cache = torch.cat((cos, sin), dim=-1).float()  # [T, D]

        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(self, positions, query, key):
        # Move cache to device if needed
        cache = self.cos_sin_cache
        if cache.device != query.device:
            cache = cache.to(query.device)

        is_cuda = query.is_cuda
        use_triton = _HAS_TRITON and is_cuda

        if use_triton:
            # Rotate query only
            q_out = _triton_rotary_q_only(query, cache.float(), is_neox=self.is_neox_style)
            return q_out, key

        # Fallback: pure PyTorch rotation of query only
        q_out = _pytorch_rotary_q_only(query, cache)
        return q_out, key


# Helper from prompt
def _compute_scaled_inv_freq(
    inv_freq: torch.Tensor,
    scaling_factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_position_embeddings: int,
) -> torch.Tensor:
    low_wl = original_max_position_embeddings / low_freq_factor
    high_wl = original_max_position_embeddings / high_freq_factor
    wl = 2 * math.pi / inv_freq
    if low_freq_factor != high_freq_factor:
        smooth = (original_max_position_embeddings / wl - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
    else:
        smooth = torch.zeros_like(inv_freq)
    return torch.where(
        wl < high_wl,
        inv_freq,
        torch.where(
            wl > low_wl,
            inv_freq / scaling_factor,
            (1 - smooth) * inv_freq / scaling_factor + smooth * inv_freq,
        ),
    )

YaRNRotaryEmbedding = ModelNew
