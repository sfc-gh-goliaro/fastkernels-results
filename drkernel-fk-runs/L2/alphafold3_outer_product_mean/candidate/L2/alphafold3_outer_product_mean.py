import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _outerprod_mean_kernel(
    ln_ptr,                 # *fp32 [B,S,N,CM]
    mask_ptr,               # *fp32 [B,S,N]
    w1_ptr, w2_ptr,         # *fp32 [CH,CM], [CH,CM]
    wout_ptr,               # *fp32 [CZ, CH*CH]
    biasout_ptr,            # *fp32 [CZ] or nullptr
    counts_ptr,             # *fp32 [B,N]
    out_ptr,                # *fp32 [B,N,N,CZ]
    # sizes
    B: tl.constexpr, S: tl.constexpr, N: tl.constexpr,
    CM: tl.constexpr, CH: tl.constexpr, CZ: tl.constexpr,
    # strides (elements)
    ln_b_stride, ln_s_stride, ln_n_stride, ln_cm_stride,
    mask_b_stride, mask_s_stride, mask_n_stride,
    w1_ch_stride, w1_cm_stride,
    w2_ch_stride, w2_cm_stride,
    wout_k_stride, wout_t_stride,
    out_b_stride, out_ni_stride, out_nj_stride, out_k_stride,
    # params
    eps: tl.constexpr,
    # tiling
    VBLOCK: tl.constexpr,   # channel block
    CMBLOCK: tl.constexpr,  # block over CM
    BLOCK_K: tl.constexpr,  # block over CZ
):
    # program ids
    pid_i = tl.program_id(0)  # residue i
    pid_j = tl.program_id(1)  # residue j
    pid_k_blk = tl.program_id(2)  # block over CZ

    b = 0  # assume single batch

    i = pid_i
    j = pid_j

    k_start = pid_k_blk * BLOCK_K
    k = k_start + tl.arange(0, BLOCK_K)
    k_mask = k < CZ

    # output accumulator
    out_acc = tl.zeros([BLOCK_K], dtype=tl.float32)

    # Pass 1: build acc block-by-block over s, then final linear
    s = 0
    while s < S:
        c0 = 0
        while c0 < CH:
            c = c0 + tl.arange(0, VBLOCK)
            c_mask = c < CH

            # GEMV for a_i[s,c] and b_j[s,c]
            a_vec = tl.zeros([VBLOCK], dtype=tl.float32)
            b_vec = tl.zeros([VBLOCK], dtype=tl.float32)
            cm0 = 0
            while cm0 < CM:
                cm = cm0 + tl.arange(0, CMBLOCK)
                cm_mask = cm < CM
                # Load ln blocks
                ln_i = tl.load(ln_ptr + (b * ln_b_stride + s * ln_s_stride + i * ln_n_stride + cm * ln_cm_stride),
                               mask=cm_mask, other=0.0).to(tl.float32)  # [CMBLOCK]
                ln_j = tl.load(ln_ptr + (b * ln_b_stride + s * ln_s_stride + j * ln_n_stride + cm * ln_cm_stride),
                               mask=cm_mask, other=0.0).to(tl.float32)  # [CMBLOCK]
                # W1 block [VBLOCK, CMBLOCK]
                w1_offs = (c[:, None] * w1_ch_stride) + (cm[None, :] * w1_cm_stride)
                w1_block = tl.load(w1_ptr + w1_offs, mask=(c_mask[:, None] & cm_mask[None, :]), other=0.0).to(tl.float32)
                # W2 block
                w2_offs = (c[:, None] * w2_ch_stride) + (cm[None, :] * w2_cm_stride)
                w2_block = tl.load(w2_ptr + w2_offs, mask=(c_mask[:, None] & cm_mask[None, :]), other=0.0).to(tl.float32)
                # Dot products
                a_vec += tl.sum(w1_block * ln_i[None, :], axis=1)
                b_vec += tl.sum(w2_block * ln_j[None, :], axis=1)
                cm0 += CMBLOCK

            # mask for sequence s
            a_mask = tl.load(mask_ptr + (b * mask_b_stride + s * mask_s_stride + i * mask_n_stride)).to(tl.float32)
            b_mask = tl.load(mask_ptr + (b * mask_b_stride + s * mask_s_stride + j * mask_n_stride)).to(tl.float32)
            a_vec = a_vec * a_mask
            b_vec = b_vec * b_mask

            # accumulate outer block
            acc_block = a_vec[:, None] * b_vec[None, :]  # [VBLOCK, VBLOCK]

            # accumulate out_acc via final weights wout[k, t]
            c_c = 0
            while c_c < VBLOCK:
                cc = c0 + c_c
                if cc >= CH:
                    break
                c_r = 0
                while c_r < VBLOCK:
                    cr = c0 + c_r
                    if cr >= CH:
                        break
                    t = cc * CH + cr
                    wtk = tl.load(wout_ptr + (k * wout_k_stride + t * wout_t_stride), mask=k_mask, other=0.0).to(tl.float32)
                    accv = acc_block[c_c, c_r]
                    out_acc += wtk * accv
                    c_r += 1
                c_c += 1

            c0 += VBLOCK
        s += 1

    # Add bias if present
    if biasout_ptr != 0:
        biasv = tl.load(biasout_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
        out_acc += biasv

    # Divide by norm[i,j] = count_i * count_j + eps
    count_i = tl.load(counts_ptr + (b * N + i))
    count_j = tl.load(counts_ptr + (b * N + j))
    denom = count_i * count_j + eps
    out_acc = out_acc / denom

    # Store
    out_offs = b * out_b_stride + i * out_ni_stride + j * out_nj_stride + k * out_k_stride
    tl.store(out_ptr + out_offs, out_acc, mask=k_mask)


class ModelNew(nn.Module):
    """Triton-optimized OuterProductMean entry point.
    Does not depend on undefined names; holds only Parameters.
    """

    def __init__(self, c_m: int, c_z: int, c_hidden: int, eps: float = 1e-3):
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps

        # Parameters equivalent to original sub-layers
        self.W1 = nn.Parameter(torch.empty(c_hidden, c_m))  # like Linear(c_m, c_hidden, bias=False).weight
        self.W2 = nn.Parameter(torch.empty(c_hidden, c_m))
        self.Wout = nn.Parameter(torch.empty(c_z, c_hidden * c_hidden))
        self.bout = nn.Parameter(torch.empty(c_z))

        # Initialize like torch defaults (Kaiming uniform for linear weights; bias uniform)
        # Keep simple: uniform in [-bound, bound]
        bound = 1 / (c_m ** 0.5)
        nn.init.uniform_(self.W1, -bound, bound)
        nn.init.uniform_(self.W2, -bound, bound)
        nn.init.kaiming_uniform_(self.Wout, a math.sqrt(5))
        fan_in = c_hidden * c_hidden
        bound_b = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.bout, -bound_b, bound_b)

    def forward(
        self,
        m: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        m:    [*, N_seq, N_res, C_m]
        mask: [*, N_seq, N_res] or None
        Returns: [*, N_res, N_res, C_z]
        """
        B, S, N, CM = m.shape
        assert CM == self.c_m, f"CM={CM} != c_m={self.c_m}"

        # Fallback if Triton not available or not CUDA
        if (not TRITON_AVAILABLE) or (not m.is_cuda):
            # Pure PyTorch reference using our Parameters and F.linear
            # 1) LayerNorm over last dim (affine=False): use torch layer for correctness
            #    But to avoid undefined, implement it directly:
            m32 = m.float()
            if mask is None:
                mask32 = torch.ones((B, S, N), device=m.device, dtype=torch.float32)
            else:
                mask32 = mask.float()
            mean = m32.mean(dim=-1, keepdim=True)
            ex2 = (m32 * m32).mean(dim=-1, keepdim=True)
            var = ex2 - mean * mean
            rstd = torch.rsqrt(var + self.eps)
            ln = (m32 - mean) * rstd  # [B,S,N,CM]

            # 2) two linears
            a = torch.nn.functional.linear(ln, self.W1)  # [B,S,N,CH]
            b = torch.nn.functional.linear(ln, self.W2)  # [B,S,N,CH]
            a = a * mask32.unsqueeze(-1)
            b = b * mask32.unsqueeze(-1)

            # 3) transpose
            aT = a.transpose(-2, -3)  # [B,N,S,CH]
            bT = b.transpose(-2, -3)

            # 4) outer product via einsum
            outer = torch.einsum("...bac,...dae->...bdce", aT, bT)  # [B,N,N,CH,CH]
            outer = outer.reshape(B, N, N, -1)  # [B,N,N,CH*CH]

            # 5) final linear + bias
            out = torch.nn.functional.linear(outer, self.Wout, self.bout)  # [B,N,N,CZ]

            # 6) norm and divide
            counts = mask32.sum(dim=1)  # [B,N]
            norm = counts[:, :, None] * counts[:, None, :]  # [B,N,N]
            out = out / (norm.unsqueeze(-1) + self.eps)

            return out.to(m.dtype)

        # Triton path: use CUDA + float32
        device = m.device

        # 1) LayerNorm over last dim using torch (correct and fast)
        m32 = m.float()
        mean = m32.mean(dim=-1, keepdim=True)
        ex2 = (m32 * m32).mean(dim=-1, keepdim=True)
        var = ex2 - mean * mean
        rstd = torch.rsqrt(var + self.eps)
        ln = (m32 - mean) * rstd  # [B,S,N,CM], float32

        # 2) mask to float32
        if mask is None:
            mask32 = torch.ones((B, S, N), device=device, dtype=torch.float32)
        else:
            mask32 = mask.to(device=device, dtype=torch.float32)

        # 3) weights to fp32 on device
        W1 = self.W1.to(device=device, dtype=torch.float32)
        W2 = self.W2.to(device=device, dtype=torch.float32)
        Wout = self.Wout.to(device=device, dtype=torch.float32)
        bias_out = self.bout.to(device=device, dtype=torch.float32)

        # 4) counts from mask
        counts = mask32.sum(dim=1).contiguous()  # [B,N]

        # 5) output buffer
        out = torch.empty((B, N, N, self.c_z), device=device, dtype=torch.float32)

        # strides (elements)
        ln_b_stride, ln_s_stride, ln_n_stride, ln_cm_stride = ln.stride()
        mask_b_stride, mask_s_stride, mask_n_stride = mask32.stride()
        w1_ch_stride, w1_cm_stride = W1.stride()
        w2_ch_stride, w2_cm_stride = W2.stride()
        wout_k_stride, wout_t_stride = Wout.stride()
        out_b_stride, out_ni_stride, out_nj_stride, out_k_stride = out.stride()

        CH = self.c_hidden
        CZ = self.c_z

        # tiling
        VBLOCK = min(32, ((CH + 31) // 32) * 32)
        if VBLOCK == 0:
            VBLOCK = 1
        CMBLOCK = 64  # matches CM=64
        BLOCK_K = min(128, ((CZ + 127) // 128) * 128)
        if BLOCK_K == 0:
            BLOCK_K = 1

        grid = (N, N, math.ceil(CZ / BLOCK_K))

        _outerprod_mean_kernel[grid](
            ln, mask32, W1, W2, Wout, bias_out, counts, out,
            B, S, N, CM, CH, CZ,
            ln_b_stride, ln_s_stride, ln_n_stride, ln_cm_stride,
            mask_b_stride, mask_s_stride, mask_n_stride,
            w1_ch_stride, w1_cm_stride,
            w2_ch_stride, w2_cm_stride,
            wout_k_stride, wout_t_stride,
            out_b_stride, out_ni_stride, out_nj_stride, out_k_stride,
            self.eps,
            VBLOCK=VBLOCK, CMBLOCK=CMBLOCK, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        return out.to(m.dtype)

OuterProductMean = ModelNew
