import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


@triton.jit
def _topk_softmax_kernel(
    logits_ptr,                # *const T
    out_weights_ptr,           # *float32  (staging for K maxima per row)
    out_ids_ptr,               # *int32
    M,                         # int: rows
    E: tl.constexpr,           # int: cols (specialized)
    stride_lm,                 # int: row stride (elements)
    stride_le,                 # int: col stride (elements)
    K: tl.constexpr,           # int: top-k (specialized)
    renorm: tl.constexpr,      # bool: renormalize
):
    pid = tl.program_id(0)
    row_ptr = logits_ptr + pid * stride_lm

    # column indices
    offs = tl.arange(0, E)

    # load row, upcast to float32
    v = tl.load(row_ptr + offs * stride_le)
    v = v.to(tl.float32)

    # iterative top-K selection
    for t in range(K):
        # max value
        cur_max = tl.max(v, axis=0)
        # mask of max locations
        is_max = v == cur_max
        # first index among maxima: min(where(is_max))
        idx = offs
        cand = tl.where(is_max, idx, 0x7fffffff)  # large int for non-max
        arg = tl.min(cand, axis=0).to(tl.int32)

        # store id and value
        tl.store(out_ids_ptr + pid * K + t, arg)
        tl.store(out_weights_ptr + pid * K + t, cur_max)

        # exclude chosen for next iteration
        v = tl.where(offs == arg, -float("inf"), v)

    # softmax over the K stored maxima (read-back from out_weights_ptr)
    # Three small passes: max, sum of exp, write normalized
    m = -float("inf")
    # pass 1: max
    for tt in range(K):
        val = tl.load(out_weights_ptr + pid * K + tt)
        m = tl.maximum(m, val)
    # pass 2: sum of exp(val - m)
    Z = 0.0
    for tt in range(K):
        val = tl.load(out_weights_ptr + pid * K + tt)
        Z += tl.exp(val - m)
    # pass 3: write normalized
    for tt in range(K):
        val = tl.load(out_weights_ptr + pid * K + tt)
        out = tl.exp(val - m) / Z if renorm else tl.exp(val - m)
        tl.store(out_weights_ptr + pid * K + tt, out)


class ModelNew(nn.Module):
    """Fused top-k selection + softmax normalization (Triton-optimized).

    forward(router_logits, top_k, renormalize=True) -> (weights [M,K] float32, ids [M,K] int32)
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(router_logits):
            raise TypeError("router_logits must be a torch.Tensor")
        if router_logits.dim() != 2:
            raise ValueError(f"Expected 2D tensor [M, E], got shape {tuple(router_logits.shape)}")
        M, E = router_logits.shape

        if top_k < 1 or top_k > E:
            raise ValueError(f"top_k must be in (0, E]; got top_k={top_k}, E={E}")

        device = router_logits.device
        if device.type != "cuda" or not _HAS_TRITON:
            # Fallback: torch.topk + softmax
            vals, inds = torch.topk(router_logits, top_k, dim=-1)
            probs = torch.softmax(vals.float(), dim=-1)
            return probs, inds.int()

        # Allocate outputs (we will reuse out_weights as staging for maxima)
        topk_weights = torch.empty((M, top_k), device=device, dtype=torch.float32)
        topk_ids = torch.empty((M, top_k), device=device, dtype=torch.int32)

        # Strides in elements
        stride_lm = router_logits.stride(0)
        stride_le = router_logits.stride(1)

        # Launch: one program per row
        grid = (M,)
        _topk_softmax_kernel[grid](
            router_logits,
            topk_weights,        # used as staging for K maxima, then final weights
            topk_ids,
            M, E,
            stride_lm, stride_le,
            top_k,
            renormalize,
            num_warps=1,
            num_stages=1,
        )

        return topk_weights, topk_ids


# Optional quick test
if __name__ == "__main__":
    torch.manual_seed(0)
    M, E = 8, 16
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        x = torch.randn(M, E, device="cuda", dtype=dtype)
        model = ModelNew().cuda()
        w, i = model(x, top_k=4, renormalize=True)
        ref_vals, ref_inds = torch.topk(x, 4, dim=-1)
        ref = torch.softmax(ref_vals.float(), dim=-1)
        print(dtype, "weights max abs diff:", (ref - w).abs().max().item(), "ids equal:", torch.equal(ref_inds.int(), i))

TopKSoftmax = ModelNew
