import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _concat_1d_kernel(
    out_ptr,                # *T, output flattened
    inp_ptrs,               # array of *T pointers to inputs flattened (length K)
    offsets,                # int32[K]: start index in 'out' for each input
    sizes,                  # int32[K]: number of elements in each input
    K: tl.constexpr,        # number of inputs
    out_numel,              # int32: total number of elements in output
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    idx = start + tl.arange(0, BLOCK)
    inbounds = idx < out_numel

    # Loop over inputs k = 0..K-1
    for k in range(K):
        off = tl.load(offsets + k)     # start offset for input k
        sz = tl.load(sizes + k)        # number of elements for input k
        end = off + sz

        # Mask for this k's region
        mask_k = inbounds & (idx >= off) & (idx < end)
        local = idx - off

        # Load from inp[k] at local, store to out at idx
        ptr_k = inp_ptrs[k]
        val = tl.load(ptr_k + local, mask=mask_k, other=0)
        tl.store(out_ptr + idx, val, mask=mask_k)


class ModelNew(nn.Module):
    def __init__(self, dimension: int = 1):
        super().__init__()
        self.d = dimension

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        # Input checks
        if not isinstance(xs, (list, tuple)):
            raise TypeError(f"Expected a list/tuple of tensors, got {type(xs)}")
        if len(xs) == 0:
            raise ValueError("xs must be non-empty")
        if len(xs) == 1:
            return xs[0].contiguous()

        # Basics
        ndim = xs[0].ndim
        device = xs[0].device
        dtype = xs[0].dtype
        for t in xs:
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor, got {type(t)}")
            if t.ndim != ndim:
                raise ValueError(f"All tensors must have the same ndim, got {t.ndim} vs {ndim}")
            if t.device != device:
                raise ValueError(f"All tensors must be on the same device, got devices {set(t.device for t in xs)}")
            if t.dtype != dtype:
                raise ValueError(f"All tensors must have the same dtype, got dtypes {set(t.dtype for t in xs)}")

        # Normalize dim
        d = self.d
        if d < 0:
            d = d + ndim
        if not (0 <= d < ndim):
            raise IndexError(f"Dimension out of range (given {self.d}, normalized {d}, ndim={ndim})")

        # Autograd and device fallbacks
        if any(t.requires_grad for t in xs):
            return torch.cat(xs, dim=self.d)
        if device.type != "cuda":
            return torch.cat(xs, dim=self.d)

        # Contiguity: require contiguous for fast path
        for t in xs:
            if not t.is_contiguous():
                return torch.cat(xs, dim=self.d)

        K = len(xs)

        # Compute M_list = sizes along dim d, and row_length = product of other dims
        M_list = [t.shape[d] for t in xs]
        M_total = sum(M_list)

        # Base (expected) shape: same as first, but we only need to enforce equality of other dims
        base_shape = list(xs[0].shape)

        # Validate shapes: must match except possibly at dim d
        for k, t in enumerate(xs):
            if t.shape != base_shape:
                # Only dim d is allowed to differ
                if t.shape[d] != base_shape[d]:
                    raise ValueError(f"Shapes must match except at dim {d}: got {t.shape} vs {base_shape}")

        row_length = 1
        for i, s in enumerate(base_shape):
            if i != d:
                row_length *= s

        # Output shape
        out_shape = list(base_shape)
        out_shape[d] = M_total
        out = torch.empty(out_shape, dtype=dtype, device=device)

        # Total elements
        out_numel = out.numel()

        # Compute contiguous offsets and sizes for each input
        # offset_k = (sum_{i<k} M_i) * row_length
        cumsum_M = [0]
        for m in M_list:
            cumsum_M.append(cumsum_M[-1] + m)
        offsets = [cumsum_M[k] * row_length for k in range(K)]
        sizes = [M_list[k] * row_length for k in range(K)]

        # Allocate device tensors for offsets/sizes (int32 is sufficient)
        offsets_t = torch.tensor(offsets, dtype=torch.int32, device=device)
        sizes_t = torch.tensor(sizes, dtype=torch.int32, device=device)

        # Flatten inputs for 1D copying
        inp_ptrs = [t.view(-1) for t in xs]

        # Kernel launch config
        BLOCK = 1024
        grid = (triton.cdiv(out_numel, BLOCK),)

        _concat_1d_kernel[grid](
            out.view(-1),
            inp_ptrs,
            offsets_t,
            sizes_t,
            K=K,
            out_numel=out_numel,
            BLOCK=BLOCK,
            num_warps=4,
        )

        return out

YOLOConcat = ModelNew
