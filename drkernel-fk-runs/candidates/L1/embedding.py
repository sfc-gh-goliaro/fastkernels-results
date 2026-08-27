import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


# -----------------------------
# Triton forward kernel: block of rows x block of cols
# -----------------------------
if _HAS_TRITON:
    @triton.jit
    def emb_gather_kernel_2d(
        w_ptr,            # *ptr to weight [E, D]
        ids_ptr,          # *ptr to indices [N] (int64)
        out_ptr,          # *ptr to output [N, D]
        N, D,             # ints: number of indices, embedding dim
        padding_idx,      # int or -1
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        # Program ids: block of rows and column tile
        pid_n = tl.program_id(0)
        pid_d = tl.program_id(1)

        rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)        # [BLOCK_N]
        cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)        # [BLOCK_D]

        mask_rows = rows < N
        mask_cols = cols < D

        # Load indices for these rows (int64)
        idx64 = tl.load(ids_ptr + rows, mask=mask_rows, other=0)  # [BLOCK_N]

        # Pad check per row
        is_pad = idx64 == padding_idx  # [BLOCK_N]

        # Compute 2D pointers for weight and output
        # Shape broadcasting to [BLOCK_N, BLOCK_D]
        base_w = idx64[:, None] * D + cols[None, :]      # [BN, BD]
        base_o = rows[:, None] * D + cols[None, :]       # [BN, BD]

        # Combine masks
        mask = (mask_rows[:, None]) & (mask_cols[None, :])

        # Load values; if pad, use zeros
        vals = tl.load(w_ptr + base_w, mask=mask & (~is_pad[:, None]), other=0)
        vals = tl.where(is_pad[:, None], tl.zeros([BLOCK_N, BLOCK_D], dtype=vals.dtype), vals)

        # Store
        tl.store(out_ptr + base_o, vals, mask=mask)


# -----------------------------
# Autograd Function: Triton forward + torch backward
# -----------------------------
class TritonEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, indices: torch.Tensor, padding_idx: int | None):
        # Fallback conditions
        if (not _HAS_TRITON) or (not weight.is_cuda) or (not indices.is_cuda):
            out = torch.nn.functional.embedding(indices, weight, padding_idx=padding_idx)
            ctx.save_for_backward(indices)
            ctx.weight_shape = weight.shape
            ctx.padding_idx = padding_idx
            ctx.use_triton = False
            return out

        # Ensure contiguous
        w = weight.contiguous()
        ids = indices.contiguous()

        # Dtype support check
        if w.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            out = torch.nn.functional.embedding(indices, weight, padding_idx=padding_idx)
            ctx.save_for_backward(indices)
            ctx.weight_shape = weight.shape
            ctx.padding_idx = padding_idx
            ctx.use_triton = False
            return out

        # Flatten indices to 1D
        orig_shape = ids.shape
        N = ids.numel()
        ids_1d = ids.view(-1)

        E, D = w.shape
        # Prepare output
        out = torch.empty((N, D), device=w.device, dtype=w.dtype)

        # padding handling
        if padding_idx is None:
            pad = -1
        else:
            pad = int(padding_idx)

        # Ensure indices are int64 for kernel
        if ids_1d.dtype != torch.long:
            ids_1d = ids_1d.long()

        # Tuning: block sizes and num_warps
        # Heuristic: use larger BLOCK_D for larger D; BLOCK_N to batch rows
        if D >= 512:
            BLOCK_D = 256
            num_warps = 8
        elif D >= 128:
            BLOCK_D = 128
            num_warps = 4
        else:
            BLOCK_D = 64
            num_warps = 2

        # Batch rows: 8 or 16 usually good
        BLOCK_N = 16 if N >= 256 else 8

        grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(D, BLOCK_D))

        emb_gather_kernel_2d[grid](
            w, ids_1d, out,
            N, D,
            pad,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
            num_warps=num_warps,
            num_stages=2,
        )

        # Save for backward
        ctx.save_for_backward(ids_1d)
        ctx.weight_shape = (E, D)
        ctx.padding_idx = pad
        ctx.use_triton = True
        ctx.orig_shape = orig_shape
        ctx.D = D
        return out.view(*orig_shape, D)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        ids_1d, = ctx.saved_tensors
        weight_shape = ctx.weight_shape
        pad = ctx.padding_idx
        D = ctx.D

        # If fallback was used, do torch-only backward
        if not getattr(ctx, 'use_triton', False):
            go = grad_output
            N = ids_1d.numel()
            go_2d = go.reshape(N, D)
            grad_w = torch.zeros(weight_shape, device=go.device, dtype=go.dtype)
            idx = ids_1d if ids_1d.dtype == torch.long else ids_1d.long()
            grad_w.index_add_(0, idx, go_2d)
            if pad is not None and 0 <= pad < weight_shape[0]:
                grad_w[pad].zero_()
            return grad_w, None, None

        # Triton forward was used; do backward with torch.index_add_ (fast and correct)
        go = grad_output
        N = ids_1d.numel()
        go_2d = go.reshape(N, D)
        grad_w = torch.zeros(weight_shape, device=go.device, dtype=go.dtype)
        idx = ids_1d if ids_1d.dtype == torch.long else ids_1d.long()
        grad_w.index_add_(0, idx, go_2d)
        if pad is not None and 0 <= pad < weight_shape[0]:
            grad_w[pad].zero_()
        return grad_w, None, None


# -----------------------------
# Triton-optimized Module entry point: ModelNew
# -----------------------------
class ModelNew(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None):
        super().__init__()
        # Use nn.Embedding to inherit PyTorch's reset_parameters and initialization
        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, input_ids: torch.Tensor):
        w = self.emb.weight
        # Ensure input on same device
        if input_ids.device != w.device:
            input_ids = input_ids.to(w.device)
        # Use Triton path if possible; else fallback inside Function
        return TritonEmbeddingFunction.apply(w, input_ids, self.emb.padding_idx)

Embedding = ModelNew
