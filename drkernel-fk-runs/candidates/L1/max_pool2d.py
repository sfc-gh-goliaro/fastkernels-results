import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def maxpool2d_tiled_kernel(
    x_ptr,                     # *const T
    y_ptr,                     # *T
    # sizes (runtime)
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    # params (runtime ints OK)
    SH: tl.constexpr,
    SW: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    # tiling
    BLOCK_OH: tl.constexpr,
    BLOCK_OW: tl.constexpr,
    # dtype code
    DTYPE: tl.constexpr,       # 0=f16, 1=f32
    # kernel sizes as constexpr for full unrolling
    KH: tl.constexpr,
    KW: tl.constexpr,
):
    # Program ids
    pid_nc = tl.program_id(0)      # over N*C
    pid_oh_blk = tl.program_id(1)  # over blocks of OH
    pid_ow_blk = tl.program_id(2)  # over blocks of OW

    # Decode n, c
    n = pid_nc // C
    c = pid_nc % C

    # Tile origins
    oh_start = pid_oh_blk * BLOCK_OH
    ow_start = pid_ow_blk * BLOCK_OW

    # Create 2D offsets for the tile
    oh_ids = oh_start + tl.arange(0, BLOCK_OH)     # [BOH]
    ow_ids = ow_start + tl.arange(0, BLOCK_OW)     # [BOW]

    # Masks for boundaries
    mask_oh = oh_ids < OH
    mask_ow = ow_ids < OW

    # Expand to 2D
    oh = oh_ids[:, None]  # [BOH, 1]
    ow = ow_ids[None, :]  # [1, BOW]
    mask = (oh < OH) & (ow < OW)  # [BOH, BOW]

    # Initialize max in f32: shape [BOH, BOW]
    maxv = tl.full((BLOCK_OH, BLOCK_OW), -float('inf'), dtype=tl.float32)

    # Precompute start indices
    ih_start = oh * SH - PH        # [BOH, 1]
    iw_start = ow * SW - PW        # [1, BOW]

    # Class base: ((n*C + c)*H)*W
    class_hw_base = ((n * C + c) * H) * W

    # Output pointer offset base: ((n*C + c)*OH + oh)*OW + ow  =>  (oh*OW + ow) + base_nc
    y_base_nc = ((n * C + c) * OH) * OW
    y_offsets = (oh * OW) + ow  # [BOH, BOW]

    # Fully unrolled kernel loops
    for kh in range(0, KH):  # unrolled
        ih = ih_start + kh         # [BOH, 1]
        valid_h = (ih >= 0) & (ih < H)  # [BOH, 1]
        row_base = class_hw_base + (ih * W)  # [BOH, 1]
        for kw in range(0, KW):  # unrolled
            iw = iw_start + kw     # [1, BOW]
            valid_w = (iw >= 0) & (iw < W)  # [1, BOW]

            # Combine masks: output mask & input bounds
            m = mask & valid_h & valid_w     # [BOH, BOW]

            # Compute pointers: x[row_base + iw]
            ptrs = x_ptr + row_base + iw  # [BOH, BOW]

            # Load, cast to f32
            vals = tl.load(ptrs, mask=m, other=-float('inf'))
            vals_f32 = vals.to(tl.float32)

            # Max update
            maxv = tl.maximum(maxv, vals_f32)

    # Store results to y: y[y_base_nc + y_offsets]
    y_ptrs = y_ptr + y_base_nc + y_offsets  # [BOH, BOW]

    # Cast to output dtype
    out = maxv
    if DTYPE == 0:
        out = out.to(tl.float16)

    # Store with mask
    tl.store(y_ptrs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
        ceil_mode: bool = False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback conditions
        use_triton = (
            TRITON_AVAILABLE and
            x.is_cuda and
            not self.ceil_mode and
            x.dtype in (torch.float16, torch.float32) and
            x.dim() == 4
        )
        if not use_triton:
            return F.max_pool2d(
                x,
                self.kernel_size,
                self.stride,
                self.padding,
                ceil_mode=self.ceil_mode,
            )

        # Ensure contiguous
        if not x.is_contiguous():
            x = x.contiguous()

        N, C, H, W = x.shape

        # Normalize parameters
        def to_2tuple(v):
            return (v, v) if isinstance(v, int) else v

        kh, kw = to_2tuple(self.kernel_size)
        sh, sw = to_2tuple(self.stride)
        ph, pw = to_2tuple(self.padding)

        # Output sizes (floor)
        def out_dim(L, K, S, P):
            return (L + 2 * P - K) // S + 1

        OH = out_dim(H, kh, sh, ph)
        OW = out_dim(W, kw, sw, pw)

        # Allocate output
        y = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)

        # Dtype code
        DTYPE = 0 if x.dtype == torch.float16 else 1

        # Heuristic block sizes adapted to problem
        # Small OH -> small BLOCK_OH to reduce masking; OW ~20 -> BLOCK_OW 32 is good.
        BLOCK_OH = 2 if OH <= 4 else 8
        BLOCK_OW = 32 if OW <= 32 else 64

        grid = (
            N * C,
            triton.cdiv(OH, BLOCK_OH),
            triton.cdiv(OW, BLOCK_OW),
        )

        # Launch heuristics
        num_warps = 4
        num_stages = 2

        maxpool2d_tiled_kernel[grid](
            x, y,
            N, C, H, W,
            OH, OW,
            sh, sw,
            ph, pw,
            BLOCK_OH,
            BLOCK_OW,
            DTYPE,
            kh, kw,  # constexpr for full unrolling
            num_warps=num_warps,
            num_stages=num_stages,
        )

        return y

MaxPool2d = ModelNew
