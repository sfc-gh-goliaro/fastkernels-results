import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


@triton.jit
def _rope_2d_concat_kernel(
    out_cos_ptr, out_sin_ptr,
    ids_ptr,          # [B] positions as float32
    inv_ptr,          # [K_total] precomputed inv_freq (float32)
    base_ptr,         # [A] int32, base col for each axis in concatenated output
    axes_ptr,         # [A] int32, d for each axis
    A: tl.constexpr,  # number of axes
    BLOCK_K: tl.constexpr
):
    pid = tl.program_id(axis=0)  # row/position index
    pos = tl.load(ids_ptr + pid).to(tl.float32)

    # loop over axes
    for i in range(0, A):
        d_i = tl.load(axes_ptr + i).to(tl.int32)
        K_i = d_i // 2
        base_i = tl.load(base_ptr + i).to(tl.int32)

        # block loop over k in this axis
        for kk in range(0, K_i, BLOCK_K):
            offs = kk + tl.arange(0, BLOCK_K)
            mask = offs < K_i

            # index into global inv_freq array (length K_total) is contiguous per-axis
            idx = offs  # local offset within this axis block; globals start at cumulative sum
            # But inv_ptr is a flat array; we must compute global index.
            # We don't have cumulative starts here; so pass a starts array or compute via base+idx.
            # Rewrite: pass 'starts' array: start of each axis in inv_ptr.
        # Note: the above comment indicates a design gap. Fix by passing 'starts'.


# Redesign kernel to pass starts array so we can compute global inv indices
@triton.jit
def _rope_2d_concat_kernel_v2(
    out_cos_ptr, out_sin_ptr,
    ids_ptr,          # [B] float32
    inv_ptr,          # [K_total] float32
    base_ptr,         # [A] int32, base col in output
    axes_ptr,         # [A] int32, d for each axis
    starts_ptr,       # [A] int32, start index in inv_ptr for each axis
    A: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    pid = tl.program_id(axis=0)  # row
    pos = tl.load(ids_ptr + pid).to(tl.float32)

    for i in range(0, A):
        d_i = tl.load(axes_ptr + i).to(tl.int32)
        K_i = d_i // 2
        base_i = tl.load(base_ptr + i).to(tl.int32)
        start_i = tl.load(starts_ptr + i).to(tl.int32)

        for kk in range(0, K_i, BLOCK_K):
            offs = kk + tl.arange(0, BLOCK_K)
            mask = offs < K_i

            global_idx = start_i + offs
            inv = tl.load(inv_ptr + global_idx, mask=mask, other=0.0).to(tl.float32)

            theta = pos * inv
            c = tl.cos(theta)
            s = tl.sin(theta)

            tl.store(out_cos_ptr + base_i + offs, c, mask=mask)
            tl.store(out_sin_ptr + base_i + offs, s, mask=mask)


class _BaseRoPE(nn.Module):
    """Base class implementing both Model and ModelNew entry points.

    __init__(theta, axes_dim)
    forward(ids) -> (cos Tensor, sin Tensor)
    """
    def __init__(self, theta: int, axes_dim: list[int] | tuple[int, ...]):
        super().__init__()
        self.theta = float(theta)
        self.axes_dim = list(axes_dim)
        if any(d % 2 != 0 for d in self.axes_dim):
            raise ValueError("Each axis dimension must be even for RoPE.")

    def _triton_forward(self, ids: torch.Tensor):
        # Flatten to [B, A]
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        shape = ids.shape
        B = 1
        for d in shape[:-1]:
            B *= d
        A = shape[-1]
        ids_2d = ids.reshape(B, A).contiguous()

        device = ids.device
        sum_d = sum(self.axes_dim)
        out_shape = shape[:-1] + (sum_d,)
        cos = torch.empty(out_shape, device=device, dtype=torch.float32)
        sin = torch.empty(out_shape, device=device, dtype=torch.float32)
        cos_2d = cos.reshape(B, sum_d).contiguous()
        sin_2d = sin.reshape(B, sum_d).contiguous()

        # Precompute inv_freq: K_total = sum(d//2)
        K_total = sum(d // 2 for d in self.axes_dim)
        inv_freq = torch.empty(K_total, device=device, dtype=torch.float32)
        starts = torch.empty(A, device=device, dtype=torch.int32)
        cum_start = 0
        for j, d in enumerate(self.axes_dim):
            K = d // 2
            if K > 0:
                arange = torch.arange(0, K, device=device, dtype=torch.float32)
                # inv_freq[k] = theta**(-(1 + 2k/d))
                exponents = 1.0 + arange * (2.0 / d)
                vals = torch.exp(-exponents * math.log(self.theta))
                inv_freq[cum_start:cum_start + K].copy_(vals)
                starts[j] = cum_start
                cum_start += K
            else:
                starts[j] = cum_start  # unused but safe

        # Bases for output columns
        bases = [0]
        for d in self.axes_dim:
            bases.append(bases[-1] + d)
        bases = bases[:-1]
        base_t = torch.tensor(bases, device=device, dtype=torch.int32)
        axes_t = torch.tensor(self.axes_dim, device=device, dtype=torch.int32)

        ids_f = ids_2d.float().contiguous()

        BLOCK_K = 128
        grid = (B,)
        _rope_2d_concat_kernel_v2[grid](
            cos_2d, sin_2d,
            ids_f,
            inv_freq,
            base_t,
            axes_t,
            starts,
            A,
            BLOCK_K=BLOCK_K,
            num_warps=4,
        )

        return cos, sin

    def forward(self, ids: torch.Tensor):
        # If Triton+CUDA, use kernel
        if _HAS_TRITON and ids.is_cuda:
            return self._triton_forward(ids)
        # Otherwise, fallback to PyTorch helper
        pos = ids.float()
        cos_list = []
        sin_list = []
        for i in range(ids.shape[-1]):
            part = pos.select(-1, i)
            freqs_cis = _get_1d_rotary_pos_embed(
                self.axes_dim[i], part,
                theta=self.theta,
                use_real=False,
            )
            cos_list.append(freqs_cis.real)
            sin_list.append(freqs_cis.imag)
        cos = torch.cat(cos_list, dim=-1).to(torch.float32)
        sin = torch.cat(sin_list, dim=-1).to(torch.float32)
        return cos, sin


# Entry points: both Model and ModelNew
class Model(_BaseRoPE):
    pass


class ModelNew(_BaseRoPE):
    pass


# Original helper for CPU/fallback
def _get_1d_rotary_pos_embed(
    dim: int,
    pos: np.ndarray | int | torch.Tensor,
    theta: float = 10000.0,
    use_real: bool = False,
    linear_factor: float = 1.0,
    ntk_factor: float = 1.0,
    repeat_interleave_real: bool = True,
    freqs_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)

    theta = theta * ntk_factor
    K = dim // 2
    arange = torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device)
    inv_freq = 1.0 / (theta ** (arange / dim)) / linear_factor  # shape [K]
    freqs = torch.outer(pos, inv_freq)  # [P, K]

    if use_real and repeat_interleave_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()
        return freqs_cos, freqs_sin
    elif use_real:
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()
        return freqs_cos, freqs_sin
    else:
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

FluxPosEmbed = ModelNew
