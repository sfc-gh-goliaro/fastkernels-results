import torch
import triton
import triton.language as tl

@triton.jit
def _merge_attn_states_kernel(
    output,               # *ptr* [T, H, D]
    output_lse,           # *ptr* [H, T] or dummy
    prefix_output,        # *ptr* [T, H, D]
    prefix_lse,           # *ptr* [H, T] (fp32)
    suffix_output,        # *ptr* [T, H, D]
    suffix_lse,           # *ptr* [H, T] (fp32)
    output_scale,         # *ptr* scalar tensor (fp32) or dummy
    HEAD_SIZE: tl.constexpr,
    USE_FP8: tl.constexpr,
    OUTPUT_LSE: tl.constexpr,
):
    # Program ids
    t = tl.program_id(0)  # token id
    h = tl.program_id(1)  # head id

    # Vector of indices along D
    offs = tl.arange(0, HEAD_SIZE)

    # Strides for contiguous [T, H, D]
    stride_t = HEAD_SIZE * tl.num_programs(1)  # == D * H
    stride_h = HEAD_SIZE                       # == D

    # Load lse scalars (fp32)
    p_lse = tl.load(prefix_lse + h * tl.num_programs(0) + t).to(tl.float32)
    s_lse = tl.load(suffix_lse + h * tl.num_programs(0) + t).to(tl.float32)

    # Compute merge scalars
    max_lse = tl.maximum(p_lse, s_lse)
    p_rel = p_lse - max_lse
    s_rel = s_lse - max_lse
    p_se = tl.exp(p_rel)
    s_se = tl.exp(s_rel)
    out_se = p_se + s_se
    p_scale = p_se / out_se
    s_scale = s_se / out_se

    # Load outputs (vector); upcast to fp32
    p_vec = tl.load(prefix_output + t * stride_t + h * stride_h + offs)
    s_vec = tl.load(suffix_output + t * stride_t + h * stride_h + offs)
    p_vec = p_vec.to(tl.float32)
    s_vec = s_vec.to(tl.float32)

    # Compute merged
    merged = p_vec * p_scale + s_vec * s_scale

    # Apply fp8 scaling+clamping if requested
    if USE_FP8:
        scale = tl.load(output_scale)  # scalar fp32
        merged = merged * (1.0 / scale)
        # fp8 E4M3 approximate range
        FP8_MIN = -1.887
        FP8_MAX = 1.887
        merged = tl.maximum(merged, FP8_MIN)
        merged = tl.minimum(merged, FP8_MAX)
        # Cast to fp8 on store (Triton will handle)

    # Store merged
    tl.store(output + t * stride_t + h * stride_h + offs, merged)

    # Store out_lse if requested
    if OUTPUT_LSE:
        out_lse_val = tl.log(out_se) + max_lse
        tl.store(output_lse + h * tl.num_programs(0) + t, out_lse_val)


class ModelNew(torch.nn.Module):
    """
    Triton-optimized online softmax merge of two attention partitions.

    Signature matches the original Model:
        forward(output, prefix_output, prefix_lse, suffix_output, suffix_lse, output_lse=None)
    In-place writes to `output` and `output_lse` if provided.
    """

    def __init__(self, use_fp8: bool = False, block_size: int | None = None, num_warps: int = 4, num_stages: int = 3):
        super().__init__()
        self.use_fp8 = use_fp8
        self.block_size = block_size
        self.num_warps = num_warps
        self.num_stages = num_stages

    def forward(
        self,
        output: torch.Tensor,
        prefix_output: torch.Tensor,
        prefix_lse: torch.Tensor,
        suffix_output: torch.Tensor,
        suffix_lse: torch.Tensor,
        output_lse: torch.Tensor | None = None,
        prefill_tokens_with_context: int | None = None,
    ) -> None:
        # CPU fallback
        if not output.is_cuda:
            # Simple torch implementation
            T, H, D = output.shape
            if prefill_tokens_with_context is None:
                prefill_tokens_with_context = T
            for t in range(T):
                for h in range(H):
                    p_lse = float(prefix_lse[h, t].item())
                    s_lse = float(suffix_lse[h, t].item())
                    p_vec = prefix_output[t, h, :].to(torch.float32)
                    s_vec = suffix_output[t, h, :].to(torch.float32)
                    if t >= prefill_tokens_with_context:
                        out = s_vec
                    else:
                        max_lse = max(p_lse, s_lse)
                        p_se = math.exp(p_lse - max_lse)
                        s_se = math.exp(s_lse - max_lse)
                        out_se = p_se + s_se
                        p_scale = p_se / out_se
                        s_scale = s_se / out_se
                        out = p_vec * p_scale + s_vec * s_scale
                        if max_lse == float("-inf"):
                            out.zero_()
                    output[t, h, :] = out.to(output.dtype)
                    if output_lse is not None:
                        output_lse[h, t] = float(math.log(out_se) + max_lse)
            return

        # Validate shapes
        assert prefix_output.shape == suffix_output.shape == output.shape, (
            f"Shape mismatch: output {output.shape}, prefix {prefix_output.shape}, "
            f"suffix {suffix_output.shape}"
        )
        T, H, D = output.shape
        assert prefix_lse.shape == (H, T), f"Expected prefix_lse shape (H,T), got {prefix_lse.shape}"
        assert suffix_lse.shape == (H, T), f"Expected suffix_lse shape (H,T), got {suffix_lse.shape}"
        if output_lse is not None:
            assert output_lse.shape == (H, T), f"Expected output_lse shape (H,T), got {output_lse.shape}"

        # Stride check between prefix and suffix head strides (dim=1)
        assert prefix_output.stride(1) == suffix_output.stride(1), (
            f"Head strides must match: prefix {prefix_output.stride(1)}, suffix {suffix_output.stride(1)}"
        )

        # Ensure contiguous
        out = output.contiguous()
        p_out = prefix_output.contiguous()
        s_out = suffix_output.contiguous()
        p_lse = prefix_lse.contiguous()
        s_lse = suffix_lse.contiguous()
        output_lse_t = output_lse.contiguous() if output_lse is not None else None

        # Normalize prefill
        if prefill_tokens_with_context is None:
            prefill_tokens_with_context = T

        # Dtype checks
        valid_in_dtypes = (torch.float32, torch.float16, torch.bfloat16)
        assert out.dtype in valid_in_dtypes, f"Unsupported output dtype {out.dtype}"
        assert p_out.dtype == s_out.dtype == out.dtype, (
            f"All output tensors must have same dtype; got {p_out.dtype}, {s_out.dtype}, {out.dtype}"
        )
        assert p_lse.dtype == s_lse.dtype == torch.float32, (
            f"lse tensors must be float32; got {p_lse.dtype}, {s_lse.dtype}"
        )
        if output_lse_t is not None:
            assert output_lse_t.dtype == torch.float32, f"output_lse must be float32; got {output_lse_t.dtype}"

        # fp8 scale
        use_fp8 = self.use_fp8 and (out.dtype == torch.float8_e4m3fn)
        if use_fp8:
            # Expect user to pass a scale; if not, use 1.0
            if output_scale is None:
                output_scale = torch.tensor(1.0, dtype=torch.float32, device=out.device)
            else:
                assert isinstance(output_scale, torch.Tensor), "output_scale must be a tensor or float"
                output_scale = output_scale.to(dtype=torch.float32, device=out.device)
                if output_scale.numel() != 1:
                    raise ValueError("output_scale must be a scalar tensor")
            output_scale = output_scale.contiguous()
        else:
            # dummy tensor (won't be used when USE_FP8=False)
            output_scale = torch.tensor(1.0, dtype=torch.float32, device=out.device)

        # Vectorization parameters
        head_size = D
        block = self.block_size
        if block is None:
            block = head_size  # 128
        num_warps = self.num_warps
        num_stages = self.num_stages

        # Launch grid: (T, H)
        grid = (T, H)

        _merge_attn_states_kernel[grid](
            out,
            output_lse_t if output_lse_t is not None else out,  # dummy if None
            p_out,
            p_lse,            # fp32
            s_out,
            s_lse,            # fp32
            output_scale,     # scalar tensor
            HEAD_SIZE=head_size,
            USE_FP8=use_fp8,
            OUTPUT_LSE=(output_lse_t is not None),
            num_warps=num_warps,
            num_stages=num_stages,
        )

MergeAttnStates = ModelNew
