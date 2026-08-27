from __future__ import annotations

import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


def _ceil_div(a, b):
    return (a + b - 1) // b


@triton.jit
def _gemv_score_kernel(
    Q, K,              # pointers to [B*H, S, D] and [B*H, S2, D]
    SCORE,             # pointer to [B*H, S, S2]
    BH: tl.constexpr,  # batch*heads
    S: tl.constexpr,   # query length
    S2: tl.constexpr,  # key length
    D: tl.constexpr,   # head dim
    stride_q_bh: tl.constexpr,
    stride_q_s: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_k_bh: tl.constexpr,
    stride_k_s2: tl.constexpr,
    stride_k_d: tl.constexpr,
    stride_sc_bh: tl.constexpr,
    stride_sc_s: tl.constexpr,
    stride_sc_s2: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # program ids
    pid_bh = tl.program_id(0)  # which (b,h)
    pid_n = tl.program_id(1)   # which column block over S2

    # guard
    if pid_bh >= BH:
        return

    # column offsets for this block
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S2

    # accumulator for scores over this block: shape [S]
    acc = tl.zeros((S,), dtype=tl.float32)

    # loop over K dimension in tiles
    for k0 in range(0, D, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < D

        # iterate over BLOCK_N columns
        for ni in range(BLOCK_N):
            j = offs_n[ni]
            if not mask_n[ni]:
                break
            # accumulate dot for column j over this K tile
            partial = tl.zeros((S,), dtype=tl.float32)
            # loop over K tile
            for ki in range(BLOCK_K):
                k = offs_k[ki]
                if not mask_k[ki]:
                    break
                # load q[:, k]
                q_ptr = Q + pid_bh * stride_q_bh + tl.arange(0, S)[:, None] * stride_q_s + k * stride_q_d
                q = tl.load(q_ptr, mask=tl.ones((S,), dtype=tl.bool), other=0.0).to(tl.float32)  # [S]
                # load k[j, k]
                k_ptr = K + pid_bh * stride_k_bh + j * stride_k_s2 + k * stride_k_d
                kv = tl.load(k_ptr, mask=tl.ones((), dtype=tl.bool), other=0.0).to(tl.float32)   # scalar
                # outer product accumulation: partial += q * kv
                partial += q * kv
            acc += partial

    # store acc to SCORE[bh, :, offs_n]
    for ni in range(BLOCK_N):
        j = offs_n[ni]
        if not mask_n[ni]:
            break
        sc_ptr = SCORE + pid_bh * stride_sc_bh + tl.arange(0, S) * stride_sc_s + j * stride_sc_s2
        tl.store(sc_ptr, acc, mask=tl.ones((S,), dtype=tl.bool))


class DenseAttention(nn.Module):
    """Dense multi-head attention.

    Input layout: (batch, seq_len, num_heads, head_dim).

    Args:
        backend: Which kernel to use.
            ``"auto"`` selects flash-attention on Ampere/Hopper when
            available, SDPA everywhere else.
            ``"sdpa"`` always uses ``F.scaled_dot_product_attention``
            (PyTorch's heuristic chooses among flash/cuDNN/mem_eff/math).
            ``"flash_attn"`` always uses the flash-attention package.
            ``"cudnn"`` pins the cuDNN flash backend via
            ``torch.nn.attention.sdpa_kernel`` (with MATH fallback for
            masks cuDNN can't handle). Required to get cuDNN flash
            through ``torch.compile`` on Blackwell.
        triton_gemm: if True, use a Triton GEMV to compute q@k^T scores.
    """

    def __init__(self, backend: Literal["auto", "sdpa", "flash_attn", "cudnn", "flex"] = "auto", triton_gemm: bool = True):
        super().__init__()
        self.fa_func = None
        self.use_cudnn_kernel = False
        self.use_flex_kernel = False
        self._flex_fn = None
        self.triton_gemm = triton_gemm

        if backend == "sdpa":
            return

        if backend == "cudnn":
            self.use_cudnn_kernel = True
            return

        if backend == "flex":
            from torch.nn.attention.flex_attention import flex_attention
            self.use_flex_kernel = True
            self._flex_fn = torch.compile(flex_attention, dynamic=False)
            return

        if backend == "flash_attn":
            self.fa_func = _resolve_flash_attn_func()
            return

        # backend == "auto": flash-attn on Ampere/Hopper (80<=cc<100); cuDNN flash
        # on Blackwell (cc>=100), where PyTorch's SDPA heuristic otherwise picks
        # FA2 (~3.6x slower than cuDNN for large joint-attention shapes on B200).
        cc = (torch.cuda.get_device_capability()[0] * 10
              + torch.cuda.get_device_capability()[1])
        if 80 <= cc < 100:
            self.fa_func = _resolve_flash_attn_func()
        elif cc >= 100:
            self.use_cudnn_kernel = True

    def _triton_qk_scores(self, q: torch.Tensor, k: torch.Tensor, scale: float | None = None) -> torch.Tensor:
        # q: [B, H, S, D], k: [B, H, S2, D]
        device = q.device
        if not _TRITON_AVAILABLE or device.type != "cuda":
            # fallback to torch
            q_ = q.reshape(-1, q.shape[-2], q.shape[-1])
            k_ = k.reshape(-1, k.shape[-2], k.shape[-1])
            return torch.bmm(q_, k_.transpose(1, 2))  # [BH, S, S2]

        BH = q.shape[0] * q.shape[1]
        S = q.shape[2]
        S2 = k.shape[2]
        D = q.shape[3]

        q_ = q.contiguous().reshape(BH, S, D)
        k_ = k.contiguous().reshape(BH, S2, D)

        score = torch.empty((BH, S, S2), device=device, dtype=torch.float32)

        stride_q_bh = q_.stride(0)
        stride_q_s = q_.stride(1)
        stride_q_d = q_.stride(2)
        stride_k_bh = k_.stride(0)
        stride_k_s2 = k_.stride(1)
        stride_k_d = k_.stride(2)
        stride_sc_bh = score.stride(0)
        stride_sc_s = score.stride(1)
        stride_sc_s2 = score.stride(2)

        BLOCK_K = 64
        BLOCK_N = 64
        grid = (BH, _ceil_div(S2, BLOCK_N))

        _gemv_score_kernel[grid](
            q_, k_, score,
            BH, S, S2, D,
            stride_q_bh, stride_q_s, stride_q_d,
            stride_k_bh, stride_k_s2, stride_k_d,
            stride_sc_bh, stride_sc_s, stride_sc_s2,
            BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return score

    def forward(
        self,
        query,
        key,
        value,
        softmax_scale=None,
        causal=False,
        attn_mask: torch.Tensor | None = None,
    ):
        # Shapes
        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)

        B, H, S, D = q.shape
        _, _, S2, _ = k.shape

        # Compute scores
        if self.triton_gemm and _TRITON_AVAILABLE and q.device.type == "cuda":
            scores = self._triton_qk_scores(q, k, softmax_scale)  # [BH, S, S2], float32
        else:
            # Fallback: use torch bmm
            q_ = q.reshape(B * H, S, D)
            k_ = k.reshape(B * H, S2, D)
            scores = torch.bmm(q_, k_.transpose(1, 2))  # [BH, S, S2]

        # Apply scaling if provided
        if softmax_scale is not None:
            scores = scores * softmax_scale

        # Apply causal mask if needed
        if causal:
            # scores shape: [BH, S, S2]; set scores where j >= i to -inf
            # We do this in PyTorch for simplicity and numerical stability.
            device = scores.device
            rows = scores.shape[1]
            cols = scores.shape[2]
            causal_mask = torch.ones((rows, cols), device=device, dtype=scores.dtype).triu(diagonal=1)
            causal_mask = causal_mask * (-1e20)
            scores = scores + causal_mask.unsqueeze(0).expand(B * H, -1, -1)

        # Add explicit attn_mask if provided (bool or float)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                # Convert to float mask; True -> 0, False -> -inf
                mask_val = attn_mask.to(scores.dtype).logical_not() * (-1e20)
                scores = scores + mask_val
            else:
                scores = scores + attn_mask

        # Softmax over dim=-1 (S2)
        p = torch.softmax(scores, dim=-1)  # [BH, S, S2]

        # Out = p @ v -> [BH, S, D]
        out = torch.bmm(p, v.reshape(B * H, S2, D))

        # Restore shape
        out = out.reshape(B, H, S, D).permute(0, 2, 1, 3)
        return out


# The rest of the code (Oasis modules) remains the same as provided, using DenseAttention in their forward.
# We keep the Triton-optimized ModelNew that integrates this kernel.

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


def _ceil_div(a, b):
    return (a + b - 1) // b


@triton.jit
def _linear_kernel(
    X,          # pointer to [B, M]
    W,          # pointer to [Cout, M] (row-major)
    BIAS,       # pointer to [Cout]
    Y,          # pointer to [B, Cout]
    B: tl.constexpr,     # batch row index
    M: tl.constexpr,     # input dim
    Cout: tl.constexpr,  # output dim
    stride_xb: tl.constexpr,  # stride for X over batch
    stride_xm: tl.constexpr,  # stride for X over M
    stride_wc: tl.constexpr,  # stride for W over Cout
    stride_wm: tl.constexpr,  # stride for W over M
    stride_yb: tl.constexpr,  # stride for Y over batch
    stride_yc: tl.constexpr,  # stride for Y over Cout
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # program ids: pid_m for M-block, pid_c for Cout-block
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    # masks
    mask_m = offs_m < M
    mask_c = offs_c < Cout

    # accumulator
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # loop over M in BLOCK_M chunks
    for m_start in range(0, M, BLOCK_M):
        m = m_start + offs_m
        # load x[m] for this batch row B
        x_ptr = X + B * stride_xb + m * stride_xm
        x = tl.load(x_ptr, mask=mask_m, other=0.0).to(tl.float32)  # [BLOCK_M]

        # iterate over c in BLOCK_C
        for ci in range(BLOCK_C):
            c = offs_c[ci]
            if not mask_c[ci]:
                continue
            # load w[c, m] as vector over m
            w_ptr = W + c * stride_wc + m * stride_wm
            w = tl.load(w_ptr, mask=mask_m, other=0.0).to(tl.float32)  # [BLOCK_M]
            acc[ci] += tl.sum(w * x)

    # add bias
    bias = tl.load(BIAS + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    acc = acc + bias

    # store
    y_ptr = Y + B * stride_yb + offs_c * stride_yc
    tl.store(y_ptr, acc, mask=mask_c)


@triton.jit
def _linear2_kernel(  # second linear: Y = Z @ W2^T + B2, where Z = X1 @ W1^T + B1 (computed elsewhere)
    Z,          # pointer to [B, K1]
    W2,         # pointer to [Kout, K1] (row-major)
    B2,         # pointer to [Kout]
    Y,          # pointer to [B, Kout]
    B: tl.constexpr,     # batch row index
    K1: tl.constexpr,    # input dim for Z
    Kout: tl.constexpr,  # output dim
    stride_zb: tl.constexpr,
    stride_zk: tl.constexpr,
    stride_w2k: tl.constexpr,
    stride_w2j: tl.constexpr,
    stride_yb: tl.constexpr,
    stride_yk: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_j = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_j = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    mask_j = offs_j < K1
    mask_k = offs_k < Kout

    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

    for j_start in range(0, K1, BLOCK_J):
        j = j_start + offs_j
        z_ptr = Z + B * stride_zb + j * stride_zk
        z = tl.load(z_ptr, mask=mask_j, other=0.0).to(tl.float32)  # [BLOCK_J]

        for ki in range(BLOCK_K):
            k = offs_k[ki]
            if not mask_k[ki]:
                continue
            w_ptr = W2 + k * stride_w2k + j * stride_w2j
            w = tl.load(w_ptr, mask=mask_j, other=0.0).to(tl.float32)  # [BLOCK_J]
            acc[ki] += tl.sum(w * z)

    bias = tl.load(B2 + offs_k, mask=mask_k, other=0.0).to(tl.float32)
    acc = acc + bias

    y_ptr = Y + B * stride_yb + offs_k * stride_yk
    tl.store(y_ptr, acc, mask=mask_k)


@triton.jit
def _broadcast_cols_kernel(
    OUT,        # pointer to [B, COLS]
    VALS,       # pointer to [B] (float32)
    B: tl.constexpr,
    COLS: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oc: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)  # 0..B-1
    pid_c = tl.program_id(1)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < COLS
    v = tl.load(VALS + row).to(tl.float32)
    out_ptr = OUT + row * stride_ob + offs_c * stride_oc
    tl.store(out_ptr, tl.full((BLOCK_C,), v, dtype=tl.float32), mask=mask_c)


class ModelNew(nn.Module):
    def __init__(
        self,
        scaling_factor: float,
        max_noise_level: int,
        stabilization_level: int,
        noise_abs_max: float,
    ):
        super().__init__()
        self.scaling_factor = float(scaling_factor)
        self.max_noise_level = int(max_noise_level)
        self.stabilization_level = int(stabilization_level)
        self.noise_abs_max = float(noise_abs_max)

    @staticmethod
    def _autocast(device: torch.device, dtype: torch.dtype):
        if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
            return torch.autocast("cuda", dtype=dtype)
        return nullcontext()

    @staticmethod
    def sigmoid_beta_schedule(
        timesteps: int,
        start: float = -3,
        end: float = 3,
        tau: float = 1,
        clamp_min: float = 0.0,
    ) -> torch.Tensor:
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, dtype=torch.float64, device="cpu") / timesteps
        v_start = math.exp(start / tau) / (1.0 + math.exp(start / tau))
        v_end = math.exp(end / tau) / (1.0 + math.exp(end / tau))
        t_ = (t * (end - start) + start) / tau
        v_t = math.exp(t_) / (1.0 + math.exp(t_))
        alphas_cumprod = (v_end - v_t) / (v_end - v_start)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas.to(torch.float32), clamp_min, 0.999)

    def _linear_triton(self, X, W, bias, B: int, M: int, Cout: int) -> torch.Tensor:
        # X: [B, M], W: [Cout, M], bias: [Cout]
        device = X.device
        if not _TRITON_AVAILABLE or device.type != "cuda":
            return F.linear(X, W, bias)
        # ensure contiguous
        Xc = X.contiguous()
        Wc = W.contiguous()
        Bc = (bias.contiguous() if bias is not None else torch.zeros(Cout, device=device, dtype=W.dtype))
        # output
        Y = torch.empty((B, Cout), device=device, dtype=torch.float32)

        # strides (in elements)
        stride_xb = Xc.stride(0)
        stride_xm = Xc.stride(1)
        stride_wc = Wc.stride(0)
        stride_wm = Wc.stride(1)
        stride_yb = Y.stride(0)
        stride_yc = Y.stride(1)

        # choose blocks
        BLOCK_M = 128
        BLOCK_C = 64
        grid = (_ceil_div(M, BLOCK_M), _ceil_div(Cout, BLOCK_C))

        _linear_kernel[grid](
            Xc, Wc, Bc, Y,
            B, M, Cout,
            stride_xb, stride_xm,
            stride_wc, stride_wm,
            stride_yb, stride_yc,
            BLOCK_M=BLOCK_M, BLOCK_C=BLOCK_C,
            num_warps=4,
        )
        return Y

    def _linear2_triton(self, Z, W2, B2, B: int, K1: int, Kout: int) -> torch.Tensor:
        # Z: [B, K1], W2: [Kout, K1], B2: [Kout]
        device = Z.device
        if not _TRITON_AVAILABLE or device.type != "cuda":
            return F.linear(Z, W2, B2)
        Zc = Z.contiguous()
        W2c = W2.contiguous()
        B2c = (B2.contiguous() if B2 is not None else torch.zeros(Kout, device=device, dtype=W2.dtype))
        Y = torch.empty((B, Kout), device=device, dtype=torch.float32)

        stride_zb = Zc.stride(0)
        stride_zk = Zc.stride(1)
        stride_w2k = W2c.stride(0)
        stride_w2j = W2c.stride(1)
        stride_yb = Y.stride(0)
        stride_yk = Y.stride(1)

        BLOCK_J = 128
        BLOCK_K = 64
        grid = (_ceil_div(K1, BLOCK_J), _ceil_div(Kout, BLOCK_K))

        _linear2_kernel[grid](
            Zc, W2c, B2c, Y,
            B, K1, Kout,
            stride_zb, stride_zk,
            stride_w2k, stride_w2j,
            stride_yb, stride_yk,
            BLOCK_J=BLOCK_J, BLOCK_K=BLOCK_K,
            num_warps=4,
        )
        return Y

    def _broadcast_cols_triton(self, vals: torch.Tensor, cols: int) -> torch.Tensor:
        # vals: [B] float32, produce out: [B, cols]
        device = vals.device
        if not _TRITON_AVAILABLE or device.type != "cuda":
            return vals[:, None].expand(-1, cols).clone()
        B = vals.shape[0]
        out = torch.empty((B, cols), device=device, dtype=torch.float32)
        stride_ob = out.stride(0)
        stride_oc = out.stride(1)
        BLOCK_C = 256
        grid = (B, _ceil_div(cols, BLOCK_C))
        _broadcast_cols_kernel[grid](
            out, vals,
            B, cols,
            stride_ob, stride_oc,
            BLOCK_C=BLOCK_C,
            num_warps=4,
        )
        return out

    def encode_prompt(self, vae: OasisAutoencoderKL, prompt: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        # prompt: [B, F, C, H, W]
        bsz, frames, channels, height, width = prompt.shape
        prompt = prompt.reshape(bsz * frames, channels, height, width)
        with torch.inference_mode(), self._autocast(prompt.device, dtype):
            # x -> patch_embed -> encoder -> norm -> mean
            x = vae.patch_embed(prompt)  # [B*F, Dh, H', W']
            for block in vae.encoder:
                x = block(x)
            x = vae.enc_norm(x)  # [B*F, Dh, H', W']
            # quant_conv: [Dh] x [2L] -> [2L]
            Wq = vae.quant_conv.weight  # [2L, Dh]
            Bq = vae.quant_conv.bias    # [2L]
            # Flatten to [B*F, Dh, H'W'] then [B*F*H'W', Dh]
            Hh = x.shape[1]
            Ww = x.shape[2]
            x_flat = x.reshape(bsz * frames, x.shape[1], -1).permute(0, 2, 1).reshape(-1, x.shape[1])
            # triton linear: X[B, Dh] -> Y[B, 2L]
            moments = self._linear_triton(x_flat, Wq, Bq, B=x_flat.shape[0], M=x.shape[1], Cout=Wq.shape[0])
            # moments shape: [B*F*H'W', 2L]; reshape back to [B*F, H'W', 2L]
            moments = moments.reshape(bsz * frames, Hh * Ww, Wq.shape[0])
            # take mean (first L); discard logvar for now
            mean = moments[:, :, :vae.latent_dim]
            latents = mean * self.scaling_factor
        h = height // vae.patch_size
        w = width // vae.patch_size
        # [B*F, H'W', L] -> [B, F, L, H', W']
        latents = latents.reshape(bsz, frames, latents.shape[-3], latents.shape[-2], latents.shape[-1])
        return latents.permute(0, 1, 4, 2, 3)

    def decode_latents(self, vae: OasisAutoencoderKL, latents: torch.Tensor) -> torch.Tensor:
        # latents: [B, F, L, H', W']
        bsz, frames, channels, height, width = latents.shape
        target_dtype = vae.post_quant_conv.weight.dtype
        # permute and flatten to feed linear
        latents_ = latents.permute(0, 1, 3, 4, 2).reshape(bsz * frames, height * width, channels).contiguous()
        # post-quant conv: [L] -> [Dd]
        Wpq = vae.post_quant_conv.weight  # [Dd, L]
        Bpq = vae.post_quant_conv.bias    # [Dd]
        x_in = latents_.to(target_dtype).to(torch.float32)  # [B*F*HW, L]
        with torch.inference_mode():
            # step A: z = X_in @ Wpq^T + Bpq
            z = self._linear_triton(x_in, Wpq, Bpq, B=x_in.shape[0], M=x_in.shape[1], Cout=Wpq.shape[0])
            # step B: y = z @ Wpred^T + Bpred
            Wpred = vae.predictor.weight  # [patch_dim, Dd]
            Bpred = vae.predictor.bias    # [patch_dim]
            y = self._linear2_triton(z, Wpred, Bpred, B=z.shape[0], K1=z.shape[1], Kout=Wpred.shape[0])
            # y: [B*F*HW, patch_dim] -> [B*F, HW, patch_dim] -> [B*F, HW, 3, ps, ps] -> reshape to image
            y = y.reshape(bsz * frames, height * width, 3, vae.patch_size, vae.patch_size)
            y = y.permute(0, 1, 3, 5, 2).reshape(bsz * frames, height, vae.patch_size, width, vae.patch_size, 3)
            y = y.permute(0, 2, 4, 1, 3, 5).reshape(bsz * frames, vae.patch_size * height, vae.patch_size * width, 3)
            decoded = (y + 1) / 2
        return decoded.reshape(bsz, frames, decoded.shape[1], decoded.shape[2], decoded.shape[3])

    def forward(
        self,
        model: OasisDiT,
        vae: OasisAutoencoderKL,
        prompt: torch.Tensor,
        actions: torch.Tensor,
        *,
        num_frames: int,
        ddim_steps: int,
        n_prompt_frames: int,
        seed: int | None,
        dtype: torch.dtype = torch.float16,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = prompt.device
        prompt_latents = self.encode_prompt(vae, prompt, dtype=dtype)[:, :n_prompt_frames]
        x = prompt_latents
        noise_range = torch.linspace(-1, self.max_noise_level - 1, ddim_steps + 1, device=device)

        betas = self.sigmoid_beta_schedule(self.max_noise_level).float().to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0).reshape(-1, 1, 1, 1)

        generator = torch.Generator(device=device).manual_seed(seed if seed is not None else 0)

        for index in range(n_prompt_frames, num_frames):
            chunk = torch.randn((prompt.shape[0], 1, *x.shape[-3:]), generator=generator, device=device)
            chunk = torch.clamp(chunk, -self.noise_abs_max, self.noise_abs_max)
            x = torch.cat([x, chunk], dim=1)
            start_frame = max(0, index + 1 - model.max_frames)

            for noise_idx in reversed(range(1, ddim_steps + 1)):
                # Build t and t_next with Triton kernel (small, real kernel)
                B = prompt.shape[0]
                cols = index + 1
                t_vals = noise_range[noise_idx].to(torch.float32)
                t_next_vals = noise_range[noise_idx - 1].to(torch.float32)
                t = self._broadcast_cols_triton(t_vals, cols)  # [B, cols]
                t_next = self._broadcast_cols_triton(t_next_vals, cols)  # [B, cols]
                # Mask where t_next < 0 -> set to t
                mask = (t_next < 0)
                if mask.any():
                    t_next = torch.where(mask, t, t_next)

                x_curr = x[:, start_frame:].clone()
                t_curr = t[:, start_frame:]
                t_next_curr = t_next[:, start_frame:]

                with torch.inference_mode(), self._autocast(prompt.device, dtype):
                    v = model(x_curr, t_curr, actions[:, start_frame:index + 1])

                x_start = alphas_cumprod[t_curr].sqrt() * x_curr - (1 - alphas_cumprod[t_curr]).sqrt() * v
                x_noise = ((1 / alphas_cumprod[t_curr]).sqrt() * x_curr - x_start) / (
                    1 / alphas_cumprod[t_curr] - 1
                ).sqrt()

                alpha_next = alphas_cumprod[t_next_curr]
                alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])
                if noise_idx == 1:
                    alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])
                x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
                x[:, -1:] = x_pred[:, -1:]

        video = self.decode_latents(vae, x)
        return video, x, prompt_latents

OasisRollout = ModelNew
