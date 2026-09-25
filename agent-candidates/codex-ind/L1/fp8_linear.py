"""Blackwell-tuned block-scaled FP8 linear and activation quantization."""

import torch
import torch.nn as nn
import triton
import triton.language as tl


def _triton_allocator(size: int, alignment: int, stream):
    return torch.empty(size, dtype=torch.int8, device="cuda")


# Tensor descriptors need a small device-side workspace.
triton.set_allocator(_triton_allocator)


@triton.jit
def _group_quant_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    stride_s_row,
    stride_s_group,
    num_groups,
    groups_per_row: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
):
    group = tl.program_id(0) * BLOCK_GROUPS + tl.arange(0, BLOCK_GROUPS)
    cols = tl.arange(0, 128)
    valid = group[:, None] < num_groups
    offsets = group[:, None] * 128 + cols[None, :]

    x = tl.load(x_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-10)
    scale = tl.math.exp2(
        tl.math.ceil(tl.math.log2(absmax * (1.0 / 448.0)))
    )
    q = tl.clamp(x / scale[:, None], -448.0, 448.0)
    tl.store(out_ptr + offsets, q, mask=valid)

    row = group // groups_per_row
    col = group % groups_per_row
    tl.store(
        scale_ptr + row * stride_s_row + col * stride_s_group,
        scale,
        mask=group < num_groups,
    )


class PerTokenGroupQuantFp8(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        out_fp8: torch.Tensor,
        out_scale: torch.Tensor,
    ) -> None:
        x = x.contiguous() if not x.is_contiguous() else x
        groups_per_row = x.shape[1] // 128
        num_groups = x.shape[0] * groups_per_row

        if num_groups <= 64:
            block_groups, num_warps = 8, 4
        elif groups_per_row == 3 and num_groups < 24000:
            block_groups, num_warps = 16, 8
        else:
            block_groups, num_warps = 32, 8

        _group_quant_kernel[(triton.cdiv(num_groups, block_groups),)](
            x,
            out_fp8,
            out_scale,
            out_scale.stride(0),
            out_scale.stride(1),
            num_groups,
            groups_per_row=groups_per_row,
            BLOCK_GROUPS=block_groups,
            num_warps=num_warps,
        )


@triton.jit
def _linear_quant_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    num_groups,
    BLOCK_GROUPS: tl.constexpr,
):
    group = tl.program_id(0) * BLOCK_GROUPS + tl.arange(0, BLOCK_GROUPS)
    cols = tl.arange(0, 128)
    valid = group[:, None] < num_groups
    offsets = group[:, None] * 128 + cols[None, :]

    x = tl.load(input_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-10)
    scale = tl.math.exp2(
        tl.math.ceil(tl.math.log2(absmax * (1.0 / 448.0)))
    )
    q = tl.clamp(x / scale[:, None], -448.0, 448.0)
    tl.store(output_ptr + offsets, q, mask=valid)

    exponent = (scale.to(tl.int32, bitcast=True) >> 23).to(tl.uint8)
    scale_offsets = group[:, None] * 4 + tl.arange(0, 4)[None, :]
    tl.store(
        scale_ptr + scale_offsets,
        exponent[:, None],
        mask=group[:, None] < num_groups,
    )


@triton.jit
def _linear_kernel(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_is_m,
    stride_is_k,
    stride_ws_n,
    stride_ws_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_sk = tl.arange(0, BLOCK_K // 32)

    input_desc = tl.make_tensor_descriptor(
        input_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    weight_desc = tl.make_tensor_descriptor(
        weight_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_N, BLOCK_K],
    )
    output_desc = tl.make_tensor_descriptor(
        output_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k_start in tl.range(0, K, BLOCK_K):
        xq = input_desc.load([pid_m * BLOCK_M, k_start])
        x_scales = tl.load(
            input_scale_ptr
            + offs_m[:, None] * stride_is_m
            + (k_start // 32 + offs_sk[None, :]) * stride_is_k,
            mask=offs_m[:, None] < M,
            other=0,
        )
        weight = weight_desc.load([pid_n * BLOCK_N, k_start]).T

        weight_group = k_start // 128 + offs_sk // 4
        packed_scale = tl.load(
            weight_scale_ptr
            + offs_n[:, None] * stride_ws_n
            + (weight_group[None, :] // 4) * stride_ws_k,
            mask=offs_n[:, None] < N,
            other=0,
        )
        weight_scales = (
            (packed_scale >> ((weight_group[None, :] % 4) * 8)) & 0xff
        ).to(tl.uint8)

        acc = tl.dot_scaled(
            xq,
            x_scales,
            "e4m3",
            weight,
            weight_scales,
            "e4m3",
            acc=acc,
            fast_math=True,
        )

    output_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], acc)


class Fp8Linear(nn.Module):
    BLOCK_SIZE = 128

    def __init__(self):
        super().__init__()
        self._a_buf: torch.Tensor | None = None
        self._s_buf: torch.Tensor | None = None
        self._o_buf: torch.Tensor | None = None
        self._pf = None

    def _ensure_buffers(
        self,
        max_tokens: int,
        K: int,
        N: int,
        device: torch.device,
    ) -> None:
        self._a_buf = torch.empty(
            max_tokens, K, dtype=torch.float8_e4m3fn, device=device
        )
        self._s_buf = torch.empty(
            max_tokens, K // 32, dtype=torch.uint8, device=device
        )
        self._o_buf = torch.empty(
            max_tokens, N, dtype=torch.bfloat16, device=device
        )

    def forward(
        self,
        input_bf16: torch.Tensor,
        weight_fp8: torch.Tensor,
        weight_scale_inv: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        N, K = weight_fp8.shape
        input_2d = input_bf16.reshape(-1, K)
        M = input_2d.shape[0]

        q_input = torch.empty(
            M, K, dtype=torch.float8_e4m3fn, device=input_2d.device
        )
        input_scale = torch.empty(
            M, K // 32, dtype=torch.uint8, device=input_2d.device
        )
        output = torch.empty(
            M, N, dtype=torch.bfloat16, device=input_2d.device
        )

        num_groups = M * (K // 128)
        block_groups = 8 if num_groups <= 64 else 32
        quant_warps = 4 if num_groups <= 64 else 8
        _linear_quant_kernel[(triton.cdiv(num_groups, block_groups),)](
            input_2d,
            q_input,
            input_scale,
            num_groups,
            BLOCK_GROUPS=block_groups,
            num_warps=quant_warps,
        )

        block_n = 256 if M >= 4096 or (K == 2048 and M >= 128) else 128
        _linear_kernel[
            (triton.cdiv(M, 128), triton.cdiv(N, block_n))
        ](
            q_input,
            input_scale,
            weight_fp8,
            weight_scale_inv,
            output,
            M=M,
            N=N,
            K=K,
            stride_is_m=input_scale.stride(0),
            stride_is_k=input_scale.stride(1),
            stride_ws_n=weight_scale_inv.stride(0),
            stride_ws_k=weight_scale_inv.stride(1),
            BLOCK_M=128,
            BLOCK_N=block_n,
            BLOCK_K=256,
            num_warps=4,
            num_stages=3,
        )

        if bias is not None:
            output = output + bias
        return output.view(*input_bf16.shape[:-1], N)
