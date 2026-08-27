import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _fused_epilogue_dot_kernel(
    t_ptr,         # [R, H] = silu(gate) * up
    Wd_ptr,        # [N, H] = down_proj.weight (row-major: Wd[n, h])
    y_ptr,         # [R, N] output

    R: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,

    stride_tr: tl.constexpr, stride_th: tl.constexpr,   # strides for t
    stride_Wdn: tl.constexpr, stride_Wdh: tl.constexpr, # strides for Wd
    stride_yr: tl.constexpr, stride_yn: tl.constexpr,   # strides for y

    BLOCK_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # Program over row blocks
    pid_r = tl.program_id(0)
    r0 = pid_r * BLOCK_R
    r = r0 + tl.arange(0, BLOCK_R)
    mask_r = r < R

    # n vector
    n = tl.arange(0, BLOCK_N)
    mask_n = n < N

    # Accumulator for y block [BR, BN]
    acc = tl.zeros((BLOCK_R, BLOCK_N), dtype=tl.bfloat16)

    # Loop over H in blocks
    for h0 in range(0, H, BLOCK_H):
        # For each inner height index, load vectors and accumulate outer product
        for ih in range(0, BLOCK_H):
            hh = h0 + ih
            #防护：如果hh超出H，跳过
            # Triton for-loop bounds are compile-time; mask is enough.
            mask_h = hh < H

            # Build pointers for t[:, hh]: shape [BR]
            ptr_t = t_ptr + r * stride_tr + hh * stride_th
            # Load t vector with row mask
            t_vec = tl.load(ptr_t, mask=mask_r, other=0).to(tl.bfloat16)  # [BR]

            # Build pointers for Wd[n, hh]: shape [BN]
            ptr_Wd = Wd_ptr + n * stride_Wdn + hh * stride_Wdh
            Wd_vec = tl.load(ptr_Wd, mask=mask_n, other=0).to(tl.bfloat16)  # [BN]

            # Form outer product matrix tile [BR, BN] and accumulate
            # Use broadcast to create the 2D tile without 2D column indexing.
            prod = t_vec[:, None] * Wd_vec[None, :]
            acc += prod

    # Store result y[r, n] = acc
    ptr_y = y_ptr + r[:, None] * stride_yr + n[None, :] * stride_yn
    store_mask = mask_r[:, None] & mask_n[None, :]
    tl.store(ptr_y, acc, mask=store_mask)


class ModelNew(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        # Keep parameter structure identical to original
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        # Keep SiLU for potential fallback
        self.act = nn.SiLU()

        # Tunable tiling params
        self.BLOCK_N = 128
        self.BLOCK_R = 64
        self.BLOCK_H = 32
        self.num_warps = 4
        self.num_stages = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, S, N] in the eval; generalize to arbitrary S
        assert x.dim() == 3, f"Expected 3D tensor [B,S,N]; got shape {tuple(x.shape)}"
        B, S, N = x.shape

        if (not x.is_cuda) or (not TRITON_AVAILABLE):
            # Fallback to original PyTorch expression
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            return self.down_proj(self.act(gate) * up)

        # Flatten to [R, N], R = B*S (no copy)
        R = B * S
        x2d = x.view(R, N).contiguous()

        # Compute gate and up with cuBLAS mm (fast): [R,N] @ [N,H] -> [R,H]
        Wg = self.gate_proj.weight.contiguous()   # [H, N]
        Wu = self.up_proj.weight.contiguous()     # [H, N]
        gate = x2d.mm(Wg.t())                    # [R, H]
        up   = x2d.mm(Wu.t())                    # [R, H]

        # Elementwise silu on gate; then multiply by up -> t = silu(gate) * up
        # silu(g) = g * sigmoid(g) = g / (1 + exp(-g))
        t = gate * torch.sigmoid(gate)
        t = t * up

        # Prepare down weight Wd: [N, H]
        Wd = self.down_proj.weight.contiguous()   # [N, H]
        H = Wd.shape[1]
        assert t.shape[1] == H, f"Mismatch: t_h={t.shape[1]} != Wd_h={H}"

        # Allocate output y: [R, N]
        y = torch.empty((R, N), device=x.device, dtype=x.dtype)

        # Strides
        stride_tr, stride_th = t.stride(0), t.stride(1)
        stride_Wdn, stride_Wdh = Wd.stride(0), Wd.stride(1)
        stride_yr, stride_yn = y.stride(0), y.stride(1)

        # Launch grid over R in blocks
        grid = (triton.cdiv(R, self.BLOCK_R),)

        _fused_epilogue_dot_kernel[grid](
            t, Wd, y,
            R, N, H,
            stride_tr, stride_th,
            stride_Wdn, stride_Wdh,
            stride_yr, stride_yn,
            BLOCK_R=self.BLOCK_R,
            BLOCK_N=self.BLOCK_N,
            BLOCK_H=self.BLOCK_H,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # Reshape back to [B, S, N]
        return y.view(B, S, N)

GLAMLP = ModelNew
