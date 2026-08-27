import math
import os
import torch

# Try Triton
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False

# FP8 info and constants
_FP8_INFO = torch.finfo(torch.float8_e4m3fn)
_GROUP_SIZE = 128
_QUANT_EPS = 1e-10

if _HAS_TRITON:
    @triton.jit
    def _fp8_group_quant_kernel(
        x_ptr, out_ptr, scale_ptr,
        stride_x_row, stride_out_row, stride_s_row, stride_s_group,
        num_cols,
        fp8_max: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        USE_UE8M0: tl.constexpr = True,
    ):
        pid = tl.program_id(0)
        groups_per_row = num_cols // GROUP_SIZE
        row = pid // groups_per_row
        group = pid % groups_per_row

        x_base = x_ptr + row * stride_x_row + group * GROUP_SIZE
        cols = tl.arange(0, GROUP_SIZE)
        x = tl.load(x_base + cols).to(tl.float32)

        absmax = tl.max(tl.abs(x))
        absmax = tl.maximum(absmax, 1e-10)
        scale_raw = absmax * (1.0 / fp8_max)
        if USE_UE8M0:
            # round to power-of-two exponent: 2^ceil(log2(scale_raw))
            scale = tl.math.exp2(tl.math.ceil(tl.math.log2(scale_raw)))
        else:
            scale = scale_raw

        x_scaled = x / scale
        x_clamped = tl.clamp(x_scaled, -fp8_max, fp8_max)
        x_fp8 = x_clamped.to(out_ptr.dtype.element_ty)

        out_base = out_ptr + row * stride_out_row + group * GROUP_SIZE
        tl.store(out_base + cols, x_fp8)

        scale_base = scale_ptr + row * stride_s_row + group * stride_s_group
        tl.store(scale_base, scale)

    def _per_token_group_quant_fp8_triton(
        x: torch.Tensor,
        out_fp8: torch.Tensor,
        out_scale: torch.Tensor,
        use_ue8m0: bool = True,
        column_major_scales: bool = False,
    ) -> None:
        # Preconditions
        if not x.is_cuda or not out_fp8.is_cuda or not out_scale.is_cuda:
            raise RuntimeError("Triton kernel requires CUDA tensors")
        if x.dtype != torch.bfloat16:
            raise TypeError(f"Expected BF16 input, got {x.dtype}")
        if out_fp8.dtype != torch.float8_e4m3fn:
            raise TypeError(f"Expected FP8 output, got {out_fp8.dtype}")
        if out_scale.dtype != torch.float32:
            raise TypeError(f"Expected float32 scales, got {out_scale.dtype}")

        M, K = x.shape
        groups_per_row = (K + _GROUP_SIZE - 1) // _GROUP_SIZE
        grid = (M * groups_per_row,)

        _fp8_group_quant_kernel[grid](
            x, out_fp8, out_scale,
            x.stride(0), out_fp8.stride(0),
            out_scale.stride(0), out_scale.stride(1),
            K,
            fp8_max=_FP8_INFO.max,
            GROUP_SIZE=_GROUP_SIZE,
            USE_UE8M0=use_ue8m0,
        )
else:
    def _per_token_group_quant_fp8_triton(*args, **kwargs):
        raise RuntimeError("Triton not available")

class Model(torch.nn.Module):
    """
    Entry point expected by the harness: forward(x, out_fp8, out_scale).
    Performs in-place per-token-group FP8 quantization of x into out_fp8,
    writing UE8M0 scales (power-of-two) into out_scale.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, out_fp8: torch.Tensor, out_scale: torch.Tensor) -> None:
        """
        x: (M, K), bfloat16, CUDA
        out_fp8: (M, K), float8_e4m3fn, CUDA
        out_scale: (M, G), float32, CUDA (G = ceil(K/128))
        """
        # Validate
        if not x.is_cuda or not out_fp8.is_cuda or not out_scale.is_cuda:
            raise RuntimeError("CUDA tensors required for Triton kernel")
        if x.dtype != torch.bfloat16:
            raise TypeError(f"Expected BF16 input, got {x.dtype}")
        if out_fp8.dtype != torch.float8_e4m3fn:
            raise TypeError(f"Expected FP8 output, got {out_fp8.dtype}")
        if out_scale.dtype != torch.float32:
            raise TypeError(f"Expected float32 scales, got {out_scale.dtype}")

        M, K = x.shape
        # Ensure out buffers are correct shape
        if out_fp8.shape != (M, K):
            raise ValueError(f"out_fp8 shape {tuple(out_fp8.shape)} != {(M, K)}")
        if out_scale.shape[0] != M:
            raise ValueError(f"out_scale leading dim {out_scale.shape[0]} != {M}")

        groups_per_row = (K + _GROUP_SIZE - 1) // _GROUP_SIZE
        expected_last_dim = groups_per_row
        if out_scale.shape[1] != expected_last_dim:
            # Allocate a view or resize is not safe in-place; instead reallocate
            # but to avoid overhead, enforce shape match.
            raise ValueError(
                f"out_scale.shape[1] = {out_scale.shape[1]} != expected {expected_last_dim}"
            )

        use_ue8m0 = bool(int(os.environ.get("VLLM_USE_DEEP_GEMM_E8M0", "1")))
        _per_token_group_quant_fp8_triton(
            x, out_fp8, out_scale,
            use_ue8m0=use_ue8m0,
            column_major_scales=False,  # not used here
        )

Fp8Linear = ModelNew
PerTokenGroupQuantFp8 = ModelNew
