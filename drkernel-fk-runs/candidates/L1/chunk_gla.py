import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def chunk_gla_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr,    # [B,T,H,D], [B,T,H,D], [B,T,H,V], [B,T,H,D]
    y_ptr,                         # [B,T,H,V]
    B: tl.constexpr, T: tl.constexpr, H: tl.constexpr,
    D: tl.constexpr, V: tl.constexpr,
    CHUNK: tl.constexpr,
    # strides in elements
    stride_b_q: tl.constexpr, stride_t_q: tl.constexpr, stride_h_q: tl.constexpr, stride_d_q: tl.constexpr,
    stride_b_k: tl.constexpr, stride_t_k: tl.constexpr, stride_h_k: tl.constexpr, stride_d_k: tl.constexpr,
    stride_b_v: tl.constexpr, stride_t_v: tl.constexpr, stride_h_v: tl.constexpr, stride_v_v: tl.constexpr,
    stride_b_g: tl.constexpr, stride_t_g: tl.constexpr, stride_h_g: tl.constexpr, stride_d_g: tl.constexpr,
    stride_b_y: tl.constexpr, stride_t_y: tl.constexpr, stride_h_y: tl.constexpr, stride_v_y: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    # running state s [D,V] fp32
    s = tl.zeros((D, V), dtype=tl.float32)

    t = tl.arange(0, CHUNK)

    t0 = 0
    while t0 < T:
        tm = t0 + t  # [CHUNK]
        valid = tm < T  # [CHUNK]

        # Load blocks: shapes [CHUNK, D], [CHUNK, D], [CHUNK, V], [CHUNK]
        q_ = tl.load(
            q_ptr + b * stride_b_q + h * stride_h_q + (t0 + t)[:, None] * stride_t_q + tl.arange(0, D)[None, :] * stride_d_q,
            mask=valid[:, None],
            other=0
        ).to(tl.float32)
        k_ = tl.load(
            k_ptr + b * stride_b_k + h * stride_h_k + (t0 + t)[:, None] * stride_t_k + tl.arange(0, D)[None, :] * stride_d_k,
            mask=valid[:, None],
            other=0
        ).to(tl.float32)
        v_ = tl.load(
            v_ptr + b * stride_b_v + h * stride_h_v + (t0 + t)[:, None] * stride_t_v + tl.arange(0, V)[None, :] * stride_v_v,
            mask=valid[:, None],
            other=0
        ).to(tl.float32)
        g_ = tl.load(
            g_ptr + b * stride_b_g + h * stride_h_g + (t0 + t) * stride_t_g + tl.arange(0, D) * stride_d_g,
            mask=valid,
            other=0
        ).to(tl.float32)

        # Intra-chunk loop
        for m in range(CHUNK):
            tm_m = t0 + m
            valid_m = tm_m < T

            gm = g_[m]
            egm = tl.exp(gm)
            thgm = tl.tanh(gm)

            # kv = k[m] @ v[m] -> [D,V]
            km = k_[m, :]  # [D]
            vm = v_[m, :]  # [V]
            kv = tl.dot(km[:, None], vm[None, :])  # [D,V]

            # s update
            s = egm * s + thgm * kv

            # y update if valid
            if valid_m:
                qm = q_[m, :]  # [D]
                ym = tl.dot(qm[None, :], s)[0, :]  # [V]
                y_off = b * stride_b_y + tm_m * stride_t_y + h * stride_h_y + tl.arange(0, V) * stride_v_y
                tl.store(y_ptr + y_off, ym.to(tl.bfloat16))

        t0 += CHUNK

    # end of kernel


class ModelNew(nn.Module):
    def __init__(self, chunk_size: int = 64, num_warps: int = 4, num_stages: int = 2):
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.num_warps = int(num_warps)
        self.num_stages = int(num_stages)

    def forward(
        self,
        q: torch.Tensor,  # [B, T, H, D]
        k: torch.Tensor,  # [B, T, H, D]
        v: torch.Tensor,  # [B, T, H, V]
        g: torch.Tensor,  # [B, T, H, D]
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,  # None or (H,D,V) or (1,H,D,V)
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Triton implementation of chunked GLA.
        Supports B>=1, arbitrary T from inputs. Computes in fp32, stores in bf16.
        """
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton is not available. Please install triton to use ModelNew.")
        if not q.is_cuda:
            raise RuntimeError("Inputs must be CUDA tensors for Triton kernel.")

        # Validate dims
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4 and g.dim() == 4, "Expected 4D tensors [B,T,H,D/V]"
        B, T, H, D = q.shape
        _, T_k, H_k, D_k = k.shape
        _, T_v, H_v, V = v.shape
        _, T_g, H_g, D_g = g.shape
        assert T == T_k == T_v == T_g, f"Time dims must match: {T},{T_k},{T_v},{T_g}"
        assert H == H_k == H_v == H_g, f"H dims must match: {H},{H_k},{H_v},{H_g}"
        assert D == D_k == D_g, f"D dims must match: {D},{D_k},{D_g}"

        # Output
        out = torch.empty((B, T, H, V), device=q.device, dtype=torch.bfloat16)

        # Strides (elements)
        stride_b_q, stride_t_q, stride_h_q, stride_d_q = q.stride()
        stride_b_k, stride_t_k, stride_h_k, stride_d_k = k.stride()
        stride_b_v, stride_t_v, stride_h_v, stride_v_v = v.stride()
        stride_b_g, stride_t_g, stride_h_g, stride_d_g = g.stride()
        stride_b_y, stride_t_y, stride_h_y, stride_v_y = out.stride()

        # Grid
        grid = (B * H,)

        # Prime initial state if provided
        s = None
        if initial_state is not None:
            assert initial_state.dim() in (3, 4), f"initial_state must be 3D or 4D, got shape {initial_state.shape}"
            assert initial_state.shape[-3:] == (H, D, V), f"Last dims must match (H,D,V): got {initial_state.shape}"
            if initial_state.dim() == 3:
                # (H,D,V)
                s = initial_state.to(dtype=torch.float32).reshape(D, V)
            else:
                # (Cs,H,D,V)
                Cs, Hs, Ds, Vs = initial_state.shape
                assert (Hs, Ds, Vs) == (H, D, V), f"Shape mismatch: {initial_state.shape} vs (B,T,H,D)"
                # Use the first provided state; Cs may be 1 or C
                s0 = initial_state[:, 0].reshape(D, V).to(dtype=torch.float32)
                s = s0

        if s is None:
            s = torch.zeros((D, V), dtype=torch.float32, device=q.device)

        # Launch kernel
        chunk_gla_kernel[grid](
            q, k, v, g,
            out,
            B, T, H, D, V,
            CHUNK=self.chunk_size,
            # strides
            stride_b_q, stride_t_q, stride_h_q, stride_d_q,
            stride_b_k, stride_t_k, stride_h_k, stride_d_k,
            stride_b_v, stride_t_v, stride_h_v, stride_v_v,
            stride_b_g, stride_t_g, stride_h_g, stride_d_g,
            stride_b_y, stride_t_y, stride_h_y, stride_v_y,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # final state: recompute with python loop (fp32) to match numerics
        final_state = None
        if output_final_state:
            s_run = s.clone()
            for t in range(T):
                kt = k[:, t].reshape(D)
                vt = v[:, t].reshape(V)
                gm = g[:, t].reshape(D)[0]
                egm = torch.exp(gm)
                thgm = torch.tanh(gm)
                kv = torch.matmul(kt[:, None], vt[None, :])  # [D,V]
                s_run = egm * s_run + thgm * kv
            final_state = s_run  # [D,V] float32

        return out, final_state

ChunkGLA = ModelNew
