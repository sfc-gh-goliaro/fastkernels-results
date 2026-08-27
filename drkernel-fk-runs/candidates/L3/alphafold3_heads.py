import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# -------------------------
# Triton kernels
# -------------------------

if TRITON_AVAILABLE:
    @triton.jit
    def _gemv_bias_kernel(
        x_ptr,           # *f16/f32 [M, K]
        w_ptr,           # *f32 [K, N]
        b_ptr,           # *f32 [N]
        y_ptr,           # *f32 [M, N]
        M: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        stride_x_m: tl.constexpr,
        stride_x_k: tl.constexpr,
        stride_w_k: tl.constexpr,
        stride_w_n: tl.constexpr,
        stride_y_m: tl.constexpr,
        stride_y_n: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        if pid >= M:
            return

        offs_n = tl.arange(0, BLOCK_N)
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)

        k = 0
        while k < K:
            k_offs = k + tl.arange(0, BLOCK_K)
            mask_k = k_offs < K

            x = tl.load(
                x_ptr + pid * stride_x_m + k_offs * stride_x_k,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)

            w = tl.load(
                w_ptr + k_offs[:, None] * stride_w_k + offs_n[None, :] * stride_w_n,
                mask=(mask_k[:, None] & (offs_n[None, :] < N)),
                other=0.0,
            ).to(tl.float32)

            prod = x[:, None] * w
            acc += tl.sum(prod, axis=0)

            k += BLOCK_K

        b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += b

        tl.store(y_ptr + pid * stride_y_m + offs_n * stride_y_n, acc, mask=offs_n < N)


    @triton.jit
    def _add_symmetric_kernel(
        z_ptr,           # *f32/f16 [B, N, N, C]
        stride_b: tl.constexpr,
        stride_i: tl.constexpr,
        stride_j: tl.constexpr,
        stride_c: tl.constexpr,
        N: tl.constexpr,
        C: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        # program ids: (b, i, block_c)
        pid_b = tl.program_id(axis=0)
        pid_i = tl.program_id(axis=1)
        pid_bc = tl.program_id(axis=2)

        c0 = pid_bc * BLOCK_C
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        # iterate all j >= i
        j = pid_i
        while j < N:
            ptr_ij = z_ptr + pid_b * stride_b + pid_i * stride_i + j * stride_j + offs_c * stride_c
            ptr_ji = z_ptr + pid_b * stride_b + j * stride_i + pid_i * stride_j + offs_c * stride_c

            val_ij = tl.load(ptr_ij, mask=mask_c, other=0.0)
            val_ji = tl.load(ptr_ji, mask=mask_c, other=0.0)

            new_ij = val_ij + val_ji
            new_ji = val_ji + val_ij  # same as new_ij

            tl.store(ptr_ij, new_ij, mask=mask_c)
            tl.store(ptr_ji, new_ji, mask=mask_c)

            j += 1


    @triton.jit
    def _ln_gemv_bias_kernel(
        x_ptr,           # *f16/f32 [M, K]
        w_ptr,           # *f32 [K, N]
        b_ptr,           # *f32 [N]
        y_ptr,           # *f32 [M, N]
        M: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        stride_x_m: tl.constexpr,
        stride_x_k: tl.constexpr,
        stride_w_k: tl.constexpr,
        stride_w_n: tl.constexpr,
        stride_y_m: tl.constexpr,
        stride_y_n: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        if pid >= M:
            return

        # Pass 1: mean/var
        sum_x = 0.0
        sum_x2 = 0.0
        k = 0
        while k < K:
            k_offs = k + tl.arange(0, BLOCK_K)
            mask_k = k_offs < K
            x = tl.load(
                x_ptr + pid * stride_x_m + k_offs * stride_x_k,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            sum_x += tl.sum(x, axis=0)
            sum_x2 += tl.sum(x * x, axis=0)
            k += BLOCK_K
        Kf = tl.full((), K, dtype=tl.float32)
        mean = sum_x / Kf
        var = sum_x2 / Kf - mean * mean
        rstd = 1.0 / tl.sqrt(var + eps)

        # Pass 2: normalize + GEMV + bias
        offs_n = tl.arange(0, BLOCK_N)
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)

        k = 0
        while k < K:
            k_offs = k + tl.arange(0, BLOCK_K)
            mask_k = k_offs < K

            x = tl.load(
                x_ptr + pid * stride_x_m + k_offs * stride_x_k,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            xhat = (x - mean) * rstd

            w = tl.load(
                w_ptr + k_offs[:, None] * stride_w_k + offs_n[None, :] * stride_w_n,
                mask=(mask_k[:, None] & (offs_n[None, :] < N)),
                other=0.0,
            ).to(tl.float32)

            prod = xhat[:, None] * w
            acc += tl.sum(prod, axis=0)

            k += BLOCK_K

        b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += b

        tl.store(y_ptr + pid * stride_y_m + offs_n * stride_y_n, acc, mask=offs_n < N)


# -------------------------
# Functional LayerNorm (no params)
# -------------------------

class LayerNorm(nn.Module):
    """LayerNorm without learnable affine; matches F.layer_norm usage."""
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = int(normalized_shape)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., C]
        assert x.shape[-1] == self.normalized_shape, f"Expected last dim {self.normalized_shape}, got {x.shape[-1]}"
        # Use torch implementation for simplicity and numerical parity
        # shape: (..., C) -> (x.numel() // C, C)
        C = self.normalized_shape
        x_ = x.reshape(-1, C).contiguous()
        mean = x_.mean(dim=-1, keepdim=True)
        var = x_.var(dim=-1, unbiased=False, keepdim=True)
        rstd = torch.rsqrt(var + self.eps)
        y = (x_ - mean) * rstd
        return y.view_as(x)


# -------------------------
# Python wrappers
# -------------------------

def _gemv_bias_triton(x, w, b, block_k=128, block_n=64, num_warps=4, num_stages=2):
    # x: [M,K]; w: [K,N]; b: [N]
    assert x.is_cuda and w.is_cuda and b.is_cuda, "Triton requires CUDA tensors"
    M, K = x.shape
    N = w.shape[1]
    x_ = x
    if x_.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        x_ = x_.float()
    w_ = w.float()
    b_ = b.float()

    y = torch.empty((M, N), device=x.device, dtype=torch.float32)

    grid = (triton.cdiv(M, 1),)
    _gemv_bias_kernel[grid](
        x_, w_, b_, y,
        M, K, N,
        x_.stride(0), x_.stride(1),
        w_.stride(0), w_.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_K=block_k, BLOCK_N=block_n,
        num_warps=num_warps, num_stages=num_stages,
    )
    return y


def _add_symmetric_triton(z, block_c=64, num_warps=4, num_stages=2):
    # z: [B,N,N,C]
    assert z.is_cuda, "Triton requires CUDA tensor"
    B, N, _, C = z.shape
    z32 = z.float()
    grid = (B, N, triton.cdiv(C, block_c))
    _add_symmetric_kernel[grid](
        z32,
        z32.stride(0), z32.stride(1), z32.stride(2), z32.stride(3),
        N, C,
        BLOCK_C=block_c,
        num_warps=num_warps, num_stages=num_stages,
    )
    return z32.to(z.dtype)


def _ln_gemv_bias_triton(x, w, b, eps=1e-5, block_k=128, block_n=64, num_warps=4, num_stages=2):
    # x: [M,K]; w: [K,N]; b: [N]
    assert x.is_cuda and w.is_cuda and b.is_cuda, "Triton requires CUDA tensors"
    M, K = x.shape
    N = w.shape[1]
    x_ = x
    if x_.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        x_ = x_.float()
    w_ = w.float()
    b_ = b.float()

    y = torch.empty((M, N), device=x.device, dtype=torch.float32)

    grid = (triton.cdiv(M, 1),)
    _ln_gemv_bias_kernel[grid](
        x_, w_, b_, y,
        M, K, N,
        x_.stride(0), x_.stride(1),
        w_.stride(0), w_.stride(1),
        y.stride(0), y.stride(1),
        eps,
        BLOCK_K=block_k, BLOCK_N=block_n,
        num_warps=num_warps, num_stages=num_stages,
    )
    return y


# -------------------------
# Heads using Triton
# -------------------------

class DistogramHead(nn.Module):
    def __init__(self, c_z: int, no_bins: int = 64):
        super().__init__()
        self.linear = nn.Linear(c_z, no_bins, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B,N,N,Cz]
        if (not TRITON_AVAILABLE) or (not z.is_cuda):
            logits = self.linear(z.view(-1, z.shape[-1])).view(*z.shape[:-1], self.linear.out_features)
            # full symmetric copy
            return logits + logits.transpose(-2, -3).flip(-1)

        B, N, _, Cz = z.shape
        W = self.linear.weight  # [B_d, Cz]
        M = B * N * N
        y = _gemv_bias_triton(z.reshape(M, Cz), W.t(), torch.zeros(Cz, device=z.device))
        y = y.view(B, N, N, -1)
        y = _add_symmetric_triton(y)
        return y


class PAEHead(nn.Module):
    def __init__(self, c_z: int, no_bins: int = 64):
        super().__init__()
        self.layer_norm = LayerNorm(c_z, eps=1e-5)
        self.linear = nn.Linear(c_z, no_bins, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B,N,N,Cz]
        if (not TRITON_AVAILABLE) or (not z.is_cuda):
            z_hat = self.layer_norm(z)
            logits = self.linear(z_hat.view(-1, z_hat.shape[-1])).view(*z_hat.shape[:-1], self.linear.out_features)
            return logits

        z_ = z
        W = self.linear.weight  # [B_d, Cz]
        B, N, _, Cz = z_.shape
        M = B * N * N
        y = _ln_gemv_bias_triton(z_.reshape(M, Cz), W.t(), torch.zeros(Cz, device=z_.device), eps=self.layer_norm.eps)
        return y.view(B, N, N, -1)


class PDEHead(nn.Module):
    def __init__(self, c_z: int, no_bins: int = 64):
        super().__init__()
        self.layer_norm = LayerNorm(c_z, eps=1e-5)
        self.linear = nn.Linear(c_z, no_bins, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if (not TRITON_AVAILABLE) or (not z.is_cuda):
            z_hat = self.layer_norm(z)
            logits = self.linear(z_hat.view(-1, z_hat.shape[-1])).view(*z_hat.shape[:-1], self.linear.out_features)
            logits = logits + logits.transpose(-2, -3).flip(-1)
            return logits

        z_ = z
        W = self.linear.weight  # [B_d, Cz]
        B, N, _, Cz = z_.shape
        M = B * N * N
        y = _ln_gemv_bias_triton(z_.reshape(M, Cz), W.t(), torch.zeros(Cz, device=z_.device), eps=self.layer_norm.eps)
        y = y.view(B, N, N, -1)
        y = _add_symmetric_triton(y)
        return y


class PLDDTHead(nn.Module):
    def __init__(self, c_s: int, no_bins: int = 50, max_atoms_per_token: int = 23):
        super().__init__()
        self.no_bins = no_bins
        self.max_atoms_per_token = max_atoms_per_token
        self.layer_norm = LayerNorm(c_s, eps=1e-5)
        self.linear = nn.Linear(c_s, max_atoms_per_token * no_bins, bias=False)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # s: [B,N,Cs]
        if (not TRITON_AVAILABLE) or (not s.is_cuda):
            s_hat = self.layer_norm(s)
            out = self.linear(s_hat.view(-1, s_hat.shape[-1])).view(*s_hat.shape[:-1], self.max_atoms_per_token, self.no_bins)
            return out

        s_ = s
        W = self.linear.weight.t()  # [Cs, out]
        B, N, Cs = s_.shape
        M = B * N
        y = _ln_gemv_bias_triton(s_.reshape(M, Cs), W, torch.zeros(Cs, device=s_.device), eps=self.layer_norm.eps)
        return y.view(B, N, self.max_atoms_per_token, self.no_bins)


class ExperimentallyResolvedHead(nn.Module):
    def __init__(self, c_s: int, no_bins: int = 2, max_atoms_per_token: int = 23):
        super().__init__()
        self.no_bins = no_bins
        self.max_atoms_per_token = max_atoms_per_token
        self.layer_norm = LayerNorm(c_s, eps=1e-5)
        self.linear = nn.Linear(c_s, max_atoms_per_token * no_bins, bias=False)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        if (not TRITON_AVAILABLE) or (not s.is_cuda):
            s_hat = self.layer_norm(s)
            out = self.linear(s_hat.view(-1, s_hat.shape[-1])).view(*s_hat.shape[:-1], self.max_atoms_per_token, self.no_bins)
            return out

        s_ = s
        W = self.linear.weight.t()  # [Cs, out]
        B, N, Cs = s_.shape
        M = B * N
        y = _ln_gemv_bias_triton(s_.reshape(M, Cs), W, torch.zeros(Cs, device=s_.device), eps=self.layer_norm.eps)
        return y.view(B, N, self.max_atoms_per_token, self.no_bins)


# -------------------------
# ModelNew entry point
# -------------------------

class ModelNew(nn.Module):
    """Triton-optimized auxiliary heads mirroring the original Model.

    forward(s, z) -> dict of logits:
      - distogram_logits: [B, N, N, 64]
      - pae_logits:       [B, N, N, 64]
      - pde_logits:       [B, N, N, 64]
      - plddt_logits:     [B, N, 23, 50]
      - experimentally_resolved_logits: [B, N, 23, 2]
    """

    def __init__(self, c_s_input: int, c_z: int, c_s: int, max_atoms_per_token: int = 23):
        super().__init__()
        # Store signature-compatible args (even if unused) to match original.
        self.c_s_input = c_s_input
        self.c_z = c_z
        self.c_s = c_s
        self.max_atoms_per_token = max_atoms_per_token

        # Instantiate heads
        self.distogram = DistogramHead(c_z, no_bins=64)
        self.pae = PAEHead(c_z, no_bins=64)
        self.pde = PDEHead(c_z, no_bins=64)
        self.plddt = PLDDTHead(c_s, no_bins=50, max_atoms_per_token=max_atoms_per_token)
        self.exp_resolved = ExperimentallyResolvedHead(c_s, no_bins=2, max_atoms_per_token=max_atoms_per_token)

    def forward(self, s: torch.Tensor, z: torch.Tensor) -> dict[str, torch.Tensor]:
        # Fallback to torch if not CUDA or Triton not available
        if (not TRITON_AVAILABLE) or (not s.is_cuda) or (not z.is_cuda):
            disto = self.distogram(z)
            pae = self.pae(z)
            pde = self.pde(z)
            plddt = self.plddt(s)
            exp = self.exp_resolved(s)
            return {
                "distogram_logits": disto,
                "pae_logits": pae,
                "pde_logits": pde,
                "plddt_logits": plddt,
                "experimentally_resolved_logits": exp,
            }

        # Triton path
        disto = self.distogram(z)
        pae = self.pae(z)
        pde = self.pde(z)
        plddt = self.plddt(s)
        exp = self.exp_resolved(s)

        return {
            "distogram_logits": disto,
            "pae_logits": pae,
            "pde_logits": pde,
            "plddt_logits": plddt,
            "experimentally_resolved_logits": exp,
        }

AuxiliaryHeads = ModelNew
