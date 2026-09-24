import torch
import torch.nn as nn

# Try importing Triton
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


@triton.jit
def _silu_mul_kernel_3d(
    gate_ptr, up_ptr, out_ptr,
    B: tl.constexpr, M: tl.constexpr, I: tl.constexpr,
    stride_b: tl.constexpr, stride_m: tl.constexpr, stride_i: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    # Program ids for (b, m, tile along i)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_tile = tl.program_id(2)

    # Offsets along I for this program
    offs_i = pid_tile * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = offs_i < I

    # Base pointer offset for (b, m, :)
    base = pid_b * stride_b + pid_m * stride_m + offs_i * stride_i

    # Load gate and up; upcast to float32
    g = tl.load(gate_ptr + base, mask=mask, other=0.0)
    u = tl.load(up_ptr + base, mask=mask, other=0.0)

    g32 = g.to(tl.float32)
    u32 = u.to(tl.float32)

    # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-g32))
    silu = g32 * sig

    y32 = silu * u32
    y = y32.to(g.dtype)

    tl.store(out_ptr + base, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        # Keep parameter structure identical to original Model
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute: down_proj( silu(gate_proj(x)) * up_proj(x) )
        Using:
          - cuBLAS for the two GEMMs
          - Triton for the fused elementwise silu * up
        Falls back to pure PyTorch if Triton/CUDA is unavailable.
        """
        if not _HAS_TRITON or x.device.type != "cuda":
            # Fallback: pure PyTorch composition
            gate = torch.nn.functional.linear(x, self.gate_proj.weight, bias=None)
            up   = torch.nn.functional.linear(x, self.up_proj.weight,   bias=None)
            silu = gate * torch.sigmoid(gate)
            y    = silu * up
            out  = torch.nn.functional.linear(y, self.down_proj.weight, bias=None)
            return out

        # 1) GEMMs via cuBLAS
        gate = torch.nn.functional.linear(x, self.gate_proj.weight, bias=None)  # [B, M, I]
        up   = torch.nn.functional.linear(x, self.up_proj.weight,   bias=None)  # [B, M, I]

        B, M, I = gate.shape

        # 2) Triton-fused elementwise: y = silu(gate) * up
        y = torch.empty_like(gate)

        # Strides in elements
        stride_b, stride_m, stride_i = gate.stride(0), gate.stride(1), gate.stride(2)

        # Launch config: tile along I
        BLOCK_I = 256
        grid = (B, M, triton.cdiv(I, BLOCK_I))

        _silu_mul_kernel_3d[grid](
            gate, up, y,
            B, M, I,
            stride_b, stride_m, stride_i,
            BLOCK_I=BLOCK_I,
            num_warps=4,
            num_stages=2,
        )

        # 3) Final down projection GEMM (cuBLAS)
        out = torch.nn.functional.linear(y, self.down_proj.weight, bias=None)  # [B, M, N]
        return out

GLAMLP = ModelNew
