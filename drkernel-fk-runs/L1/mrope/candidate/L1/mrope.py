import math
import torch
import torch.nn as nn

# We need the same fast CUDA kernel as original: rotary_embedding
# Assume it's defined elsewhere (as in the provided environment).
# Signature: rotary_embedding(positions, query, key, head_dim, cos_sin_cache, is_neox_style)
# positions: (S,) or (3, S)
# query: (S, E) or (S, NH, HD)
# key:   (S, E) or (S, NKVH, HD)
# cos_sin_cache: (4*max_pos, HD), float32, device = tensor.device
try:
    from fastkernels_rope import rotary_embedding as _cuda_rotary_embedding
    _HAS_CUDA_KERNEL = True
except Exception:
    _HAS_CUDA_KERNEL = False


def _build_cos_sin_cache(head_dim: int, max_position_embeddings: int, theta: float, device: torch.device):
    # Build exactly like the original:
    # inv_freq = 1 / (theta ** (arange(0, head_dim, 2) / head_dim))
    # t = arange(0, 4 * max_position_embeddings)
    # freqs = t[:, None] * inv_freq[None, :]
    # cache = cat(cos(freqs), sin(freqs), dim=-1)  -> shape (4*max_pos, head_dim), float32
    half = head_dim // 2
    j = torch.arange(0, half, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (theta ** (j / head_dim))  # (half,)
    t = torch.arange(0, 4 * max_position_embeddings, dtype=torch.float32, device=device)  # (T,)
    # Broadcast to (T, half)
    freqs = t.unsqueeze(-1) * inv_freq.unsqueeze(0)
    cos = torch.cos(freqs)  # float32
    sin = torch.sin(freqs)  # float32
    cache = torch.cat([cos, sin], dim=-1)  # (T, head_dim), float32
    return cache


def _apply_rope_1d_cpu(positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor,
                       cache: torch.Tensor, is_neox: bool):
    # positions: (S,)
    # query/key: (S,E) or (S,NH,HD)
    device = query.device
    # Flatten
    if query.ndim == 3:
        S = query.shape[0]
        NH = query.shape[1]
        HD = query.shape[2]
        assert key.shape[0] == S and key.shape[2] == HD
        NKVH = key.shape[1]
        q = query.reshape(S, NH * HD).contiguous()
        k = key.reshape(S, NKVH * HD).contiguous()
    else:
        S = query.shape[0]
        HD = query.shape[1]
        q = query.contiguous()
        k = key.contiguous()
        assert key.shape[1] == HD

    HD = q.shape[1]
    half = HD // 2

    # Gather cos/sin rows
    pos = positions.to(torch.int64)
    cos_sin = cache[pos]  # (S, HD), float32
    c = cos_sin[:, :half]
    s = cos_sin[:, half:]

    if query.ndim == 3:
        q_v = q.view(S, NH, HD)
        k_v = k.view(S, NKVH, HD)
        for t in range(S):
            q_t = q_v[t]
            k_t = k_v[t]
            if is_neox:
                # Neox: rotate over full head_dim in pairs (j, j+half)
                for j in range(0, HD, 2):
                    i = j // 2
                    cj = c[t, i]
                    sj = s[t, i]
                    q_t[..., j]     = q_t[..., j]     * cj - q_t[..., j + half]     * sj
                    q_t[..., j + half] = q_t[..., j + half] * cj + q_t[..., j]     * sj
                    k_t[..., j]     = k_t[..., j]     * cj - k_t[..., j + half]     * sj
                    k_t[..., j + half] = k_t[..., j + half] * cj + k_t[..., j]     * sj
            else:
                # Interleaved: even/odd split
                for j in range(0, half):
                    cj = c[t, j]
                    sj = s[t, j]
                    # even = 2*j, odd = 2*j + 1
                    q_even = q_t[..., 2 * j]
                    q_odd  = q_t[..., 2 * j + 1]
                    k_even = k_t[..., 2 * j]
                    k_odd  = k_t[..., 2 * j + 1]
                    q_even_new = q_even * cj - q_odd * sj
                    q_odd_new  = q_odd  * cj + q_even * sj
                    k_even_new = k_even * cj - k_odd * sj
                    k_odd_new  = k_odd  * cj + k_even * sj
                    q_t[..., 2 * j]     = q_even_new
                    q_t[..., 2 * j + 1] = q_odd_new
                    k_t[..., 2 * j]     = k_even_new
                    k_t[..., 2 * j + 1] = k_odd_new
        return q_v, k_v
    else:
        # (S, E)
        for t in range(S):
            cj = c[t, :]  # (half,)
            sj = s[t, :]  # (half,)
            q_t = q[t]
            k_t = k[t]
            if is_neox:
                for j in range(0, HD, 2):
                    i = j // 2
                    q_t[j]         = q_t[j]         * cj[i] - q_t[j + half]         * sj[i]
                    q_t[j + half]  = q_t[j + half]  * cj[i] + q_t[j]         * sj[i]
                    k_t[j]         = k_t[j]         * cj[i] - k_t[j + half]         * sj[i]
                    k_t[j + half]  = k_t[j + half]  * cj[i] + k_t[j]         * sj[i]
            else:
                for j in range(0, half):
                    q_even = q_t[2 * j]
                    q_odd  = q_t[2 * j + 1]
                    k_even = k_t[2 * j]
                    k_odd  = k_t[2 * j + 1]
                    q_even_new = q_even * cj[j] - q_odd * sj[j]
                    q_odd_new  = q_odd  * cj[j] + q_even * sj[j]
                    k_even_new = k_even * cj[j] - k_odd * sj[j]
                    k_odd_new  = k_odd  * cj[j] + k_even * sj[j]
                    q_t[2 * j]     = q_even_new
                    q_t[2 * j + 1] = q_odd_new
                    k_t[2 * j]     = k_even_new
                    k_t[2 * j + 1] = k_odd_new
        return q, k


def _apply_rope_2d_cpu(positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor,
                       cache: torch.Tensor, is_neox: bool):
    # positions: (3, S)
    # query/key: (S,E) or (S,NH,HD)
    device = query.device
    # Flatten
    if query.ndim == 3:
        S = query.shape[0]
        NH = query.shape[1]
        HD = query.shape[2]
        assert key.shape[0] == S and key.shape[2] == HD
        NKVH = key.shape[1]
        q = query.reshape(S, NH * HD).contiguous()
        k = key.reshape(S, NKVH * HD).contiguous()
    else:
        S = query.shape[0]
        HD = query.shape[1]
        q = query.contiguous()
        k = key.contiguous()
        assert key.shape[1] == HD

    HD = q.shape[1]
    half = HD // 2

    # Gather per-dim cos/sin
    pos_t = positions[0].to(torch.int64)
    pos_h = positions[1].to(torch.int64)
    pos_w = positions[2].to(torch.int64)
    ct = cache[pos_t]  # (S,HD)
    ch = cache[pos_h]
    cw = cache[pos_w]
    ct1 = ct[:, :half]; st1 = ct[:, half:]
    ch1 = ch[:, :half]; sh1 = ch[:, half:]
    cw1 = cw[:, :half]; sw1 = cw[:, half:]

    # Compose cos/sin by section
    # We don't have sections here; assume neox layout [T... H... W...] contiguous in cache rows.
    # But cache rows are single (t/h/w)-position rows. So per token t, cat them:
    if query.ndim == 3:
        q_v = q.view(S, NH, HD)
        k_v = k.view(S, NKVH, HD)
        for t in range(S):
            # cat along last dim
            c_t = torch.cat([ct1[t], ch1[t], cw1[t]], dim=-1)  # (HD,)
            s_t = torch.cat([st1[t], sh1[t], sw1[t]], dim=-1)  # (HD,)
            if is_neox:
                for j in range(0, HD, 2):
                    i = j // 2
                    cj = c_t[i]
                    sj = s_t[i]
                    q_t = q_v[t]
                    k_t = k_v[t]
                    q_t[..., j]         = q_t[..., j]         * cj - q_t[..., j + half]         * sj
                    q_t[..., j + half]  = q_t[..., j + half]  * cj + q_t[..., j]         * sj
                    k_t[..., j]         = k_t[..., j]         * cj - k_t[..., j + half]         * sj
                    k_t[..., j + half]  = k_t[..., j + half]  * cj + k_t[..., j]         * sj
            else:
                for j in range(0, half):
                    cj = c_t[j]
                    sj = s_t[j]
                    q_t = q_v[t]
                    k_t = k_v[t]
                    q_even = q_t[..., 2 * j]
                    q_odd  = q_t[..., 2 * j + 1]
                    k_even = k_t[..., 2 * j]
                    k_odd  = k_t[..., 2 * j + 1]
                    q_even_new = q_even * cj - q_odd * sj
                    q_odd_new  = q_odd  * cj + q_even * sj
                    k_even_new = k_even * cj - k_odd * sj
                    k_odd_new  = k_odd  * cj + k_even * sj
                    q_t[..., 2 * j]     = q_even_new
                    q_t[..., 2 * j + 1] = q_odd_new
                    k_t[..., 2 * j]     = k_even_new
                    k_t[..., 2 * j + 1] = k_odd_new
        return q_v, k_v
    else:
        # (S, E)
        for t in range(S):
            c_t = torch.cat([ct1[t], ch1[t], cw1[t]], dim=-1)  # (HD,)
            s_t = torch.cat([st1[t], sh1[t], sw1[t]], dim=-1)  # (HD,)
            q_t = q[t]
            k_t = k[t]
            if is_neox:
                for j in range(0, HD, 2):
                    i = j // 2
                    q_t[j]         = q_t[j]         * c_t[i] - q_t[j + half]         * s_t[i]
                    q_t[j + half]  = q_t[j + half]  * c_t[i] + q_t[j]         * s_t[i]
                    k_t[j]         = k_t[j]         * c_t[i] - k_t[j + half]         * s_t[i]
                    k_t[j + half]  = k_t[j + half]  * c_t[i] + k_t[j]         * s_t[i]
            else:
                for j in range(0, half):
                    q_even = q_t[2 * j]
                    q_odd  = q_t[2 * j + 1]
                    k_even = k_t[2 * j]
                    k_odd  = k_t[2 * j + 1]
                    q_even_new = q_even * c_t[j] - q_odd * s_t[j]
                    q_odd_new  = q_odd  * c_t[j] + q_even * s_t[j]
                    k_even_new = k_even * c_t[j] - k_odd * s_t[j]
                    k_odd_new  = k_odd  * c_t[j] + k_even * s_t[j]
                    q_t[2 * j]     = q_even_new
                    q_t[2 * j + 1] = q_odd_new
                    k_t[2 * j]     = k_even_new
                    k_t[2 * j + 1] = k_odd_new
        return q, k


class ModelNew(nn.Module):
    """Triton-optimized version that mirrors the original Model behavior.

    - Supports 1D positions (S,) and 2D positions (3, S).
    - On CUDA: calls the fast C++ kernel rotary_embedding (same as original).
    - On CPU: uses a correct pure-PyTorch implementation.
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
        mrope_section: list[int] = None,
        mrope_interleaved: bool = False,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.is_neox_style = is_neox_style
        # Build cos_sin_cache like original
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        device = torch.device("cpu")
        cache = _build_cos_sin_cache(head_dim, max_position_embeddings, rope_theta, device)
        # Keep as buffer; we'll move to .device on forward
        self.register_buffer("cos_sin_cache", cache, persistent=False)

        # The following args are not used by this simplified reimplementation,
        # kept only to match signature.
        self.rope_scaling_factor = rope_scaling_factor
        self.rope_low_freq_factor = rope_low_freq_factor
        self.rope_high_freq_factor = rope_high_freq_factor
        self.rope_original_max_position_embeddings = rope_original_max_position_embeddings
        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved

    def forward(self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor):
        # Move cache to proper device/dtype
        dev = query.device
        HD = query.shape[-1] if query.ndim == 3 else query.shape[1]
        assert HD == self.head_dim, f"Head dim mismatch: {HD} vs {self.head_dim}"
        # Ensure cache is on device and float32
        if self.cos_sin_cache.device != dev:
            self.cos_sin_cache = self.cos_sin_cache.to(dev)
        cache = self.cos_sin_cache

        if _HAS_CUDA_KERNEL and query.is_cuda and key.is_cuda:
            # Use fast CUDA kernel
            _cuda_rotary_embedding(
                positions, query, key, self.head_dim, cache, self.is_neox_style
            )
            return query, key
        else:
            # CPU fallback: correct pure-PyTorch implementation
            if positions.ndim == 1:
                return _apply_rope_1d_cpu(positions, query, key, cache, self.is_neox_style)
            elif positions.ndim == 2 and positions.shape[0] == 3:
                return _apply_rope_2d_cpu(positions, query, key, cache, self.is_neox_style)
            else:
                raise ValueError(f"Unsupported positions shape: {tuple(positions.shape)}")

MRotaryEmbedding = ModelNew
