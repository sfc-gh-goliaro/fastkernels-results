"""FP8 linear (block-scaled FP8 matrix multiply), Blackwell fast path.

Three kernels, all written here rather than dispatched to a vendor library:

* Per-token-group quantization (``fp8_fast.cu``) -- used by
  ``PerTokenGroupQuantFp8`` and by ``Fp8Linear``'s activation quant.  A
  128-element group is handled by 16 lanes, 8 bf16 each: one 16 B vector load
  in, one 8 B vector store out, the group absmax reduced with four
  ``__shfl_xor_sync`` steps, nothing through shared memory and no block
  barrier.  The vLLM kernel the baseline calls stages every group through smem
  behind a ``__syncthreads`` and runs at ~1.0 TB/s; this one reaches ~4.3 TB/s,
  85% of a plain device copy of the same bytes.

* The GEMM -- a Triton kernel using ``tl.dot_scaled``, i.e. Blackwell's
  ``tcgen05.mma.kind::mxf8f6f4.block_scale``.  A 128-element UE8M0 group is
  exactly four MXFP8 32-element groups sharing one exponent, so replicating
  each exponent 4x lets the tensor core apply the block scales itself and the
  fp32 accumulator never has to be rescaled in CUDA cores (doing that promotion
  explicitly measured ~3x slower).  Output is bit-identical to the reference
  block-scaled GEMM on the captured shapes.

* A fused quantize+GEMV (``fp8_fast.cu``) for M <= 2, where the MMA path throws
  away 128x of the tensor core on a problem that is purely weight-bandwidth
  bound.  One warp per weight row, one launch for the whole linear.

``weight_scale_inv`` is DeepGEMM's packed UE8M0 weight scale: ``int32[N,
K/512]`` with ``stride == (1, N)``, where byte ``b`` of word ``(n, j)`` is the
exponent (bias 127) of K-block ``4*j + b`` for output row ``n``.  It is decoded
inline in the consuming kernels -- no pre-pass -- and since one 128x128 weight
block shares an exponent, a GEMM N tile of <= 128 rows needs only a single
scalar word from it per K-block.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from fastkernels.infra.cuda_ext import lazy_op

_C = lazy_op("fp8_fast", "fp8_fast.cu")

_GROUP_SIZE = 128
# The fused GEMV kernel is weight-bandwidth bound and does M FMA passes per
# weight byte, so it only wins at very small M; above this the MMA path does.
_SMALL_M_MAX = 2


# ---------------------------------------------------------------------------
# Triton needs a scratch allocator to build device-side TMA descriptors.  One
# cached buffer per size is enough: descriptors are consumed by the kernel that
# was launched on the same stream right after they are written.
# ---------------------------------------------------------------------------
_scratch: dict[int, torch.Tensor] = {}


def _tma_alloc(size: int, alignment: int, stream) -> torch.Tensor:
    buf = _scratch.get(size)
    if buf is None:
        buf = torch.empty(size, dtype=torch.int8, device="cuda")
        _scratch[size] = buf
    return buf


triton.set_allocator(_tma_alloc)


# ---------------------------------------------------------------------------
# Fallback quantization (odd shapes / layouts the CUDA kernel does not take).
# ---------------------------------------------------------------------------
@triton.jit
def _quant_fallback_kernel(x_ptr, out_ptr, s_ptr, stride_x, stride_o,
                           stride_s0, stride_s1, groups_per_row,
                           GROUP: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // groups_per_row
    grp = pid % groups_per_row
    cols = tl.arange(0, GROUP)
    x = tl.load(x_ptr + row * stride_x + grp * GROUP + cols).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)
    bits = (absmax * (1.0 / 448.0)).to(tl.int32, bitcast=True)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    e = tl.maximum(tl.minimum(e, 254), 1)
    scale = (e << 23).to(tl.float32, bitcast=True)
    q = tl.clamp(x / scale, -448.0, 448.0)
    tl.store(out_ptr + row * stride_o + grp * GROUP + cols,
             q.to(out_ptr.dtype.element_ty))
    tl.store(s_ptr + row * stride_s0 + grp * stride_s1, scale)


def _quant_fallback(x: torch.Tensor, out_fp8: torch.Tensor,
                    out_scale: torch.Tensor) -> None:
    x2 = x.reshape(-1, x.shape[-1])
    o2 = out_fp8.reshape(-1, out_fp8.shape[-1])
    s2 = out_scale.reshape(-1, out_scale.shape[-1])
    M, K = x2.shape
    gpr = K // _GROUP_SIZE
    _quant_fallback_kernel[(M * gpr,)](
        x2, o2, s2, x2.stride(0), o2.stride(0), s2.stride(0), s2.stride(1),
        gpr, _GROUP_SIZE,
    )


def _quant_f32(x: torch.Tensor, out_fp8: torch.Tensor,
               out_scale: torch.Tensor) -> None:
    """Per-token-group FP8 quant with float32 power-of-two (UE8M0) scales."""
    if not x.is_contiguous():
        x = x.contiguous()
    fast = (
        x.shape[-1] % _GROUP_SIZE == 0
        and x.dtype in (torch.bfloat16, torch.float16)
        and out_fp8.is_contiguous()
        and out_fp8.dtype == torch.float8_e4m3fn
        and out_scale.is_contiguous()
        and out_scale.dtype == torch.float32
        and out_scale.numel() * _GROUP_SIZE == x.numel()
    )
    if fast:
        _C.quant_f32(x, out_fp8, out_scale)
    else:
        _quant_fallback(x, out_fp8, out_scale)


class PerTokenGroupQuantFp8(nn.Module):
    """In-place per-token-group (128) FP8 quantization with UE8M0 scales."""

    def forward(self, x: torch.Tensor, out_fp8: torch.Tensor,
                out_scale: torch.Tensor) -> None:
        _quant_f32(x, out_fp8, out_scale)


# ---------------------------------------------------------------------------
# Block-scaled FP8 GEMM: C[M,N] = A[M,K] @ B[N,K]^T with 1x128 activation and
# 128x128 weight UE8M0 scales, executed on tcgen05 block-scaled MMA.
# ---------------------------------------------------------------------------
@triton.jit
def _fp8_bs_gemm(A_desc, ASC, B_desc, WSI, s_wn, s_wk, C_desc, M, N, K,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 GM: tl.constexpr, NSTAGE: tl.constexpr, WS: tl.constexpr):
    pid = tl.program_id(0)
    nn = tl.cdiv(N, BN)
    nm = tl.cdiv(M, BM)
    ngm = GM * nn
    gid = pid // ngm
    fm = gid * GM
    gm = min(nm - fm, GM)
    pm = fm + (pid % ngm) % gm
    pn = (pid % ngm) // gm
    om = pm * BM
    on = pn * BN
    offm = om + tl.arange(0, BM)
    # MXFP8 scale factors cover 32 elements each, so a BK-wide tile needs BK/32
    # bytes per row and every group of 4 of them shares one UE8M0 exponent.
    NKB: tl.constexpr = BK // 128
    NSF: tl.constexpr = BK // 32
    sfk = tl.arange(0, NSF) // 4
    # Loop-invariant unit (2^0) scale for the B operand; see the fold below.
    b_unit = tl.full((BN, NSF), 127, tl.uint8)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for it in tl.range(0, K // BK, num_stages=NSTAGE, warp_specialize=WS):
        a = A_desc.load([om, it * BK])
        b = B_desc.load([on, it * BK])
        sf = tl.zeros((BM, NSF), dtype=tl.int32)
        for u in tl.static_range(NKB):
            kb = it * NKB + u
            # Activation exponent: K-major int32 (byte replicated 4x), so this
            # load is contiguous across the M tile.
            ea = tl.load(ASC + kb * M + offm, mask=offm < M, other=0) & 0xFF
            # Weight exponent: constant over the whole N tile because BN divides
            # the 128x128 weight block, so it is one scalar load from the packed
            # UE8M0 words.
            eb = (tl.load(WSI + on * s_wn + (kb // 4) * s_wk) >> ((kb % 4) * 8)) & 0xFF
            # Both scales are powers of two, so their product is itself one E8M0
            # code: 2^(ea-127) * 2^(eb-127) = 2^((ea+eb-127)-127).  Folding them
            # leaves the B scale loop-invariant, so its SMEM->TMEM scale copy
            # hoists out of the loop -- that copy is what limits this pipeline.
            # An out-of-range sum means the true product already over/underflowed
            # fp32, so the clamp only touches results that were garbage anyway.
            comb = tl.minimum(tl.maximum(ea + eb - 127, 0), 254)
            if NKB == 1:
                sf = tl.broadcast_to(comb[:, None], (BM, NSF))
            else:
                sf = tl.where(sfk[None, :] == u, comb[:, None], sf)
        acc = tl.dot_scaled(a, sf.to(tl.uint8), "e4m3", tl.trans(b), b_unit,
                            "e4m3", acc)
    C_desc.store([om, on], acc.to(tl.bfloat16))


# Tile/pipeline choice, measured on B200 over the captured shapes.  BN must
# divide 128 so the folded weight exponent stays constant over the N tile.
def _pick_cfg(M: int, N: int, K: int):
    """(BM, BN, BK, GROUP_M, num_stages, warp_specialize, num_warps)."""
    if M <= 512:
        # Few tiles, so the K loop is exposed to load latency rather than
        # bandwidth: a wide BK (fewer, fatter pipeline steps) wins by ~2.3x over
        # BK=128 here.
        cfg = (128, 64, 512, 8, 3, False, 4)
    elif M <= 2048:
        # Deeper pipelines only pay off once the K loop is long enough to fill
        # them; at K<=2048 (8 steps of BK=256) fewer stages measured faster.
        cfg = (128, 128, 256, 8, 4 if K > 2048 else 2, False, 4)
    else:
        # Enough tiles to saturate L2 bandwidth, which is then the limit, so the
        # smallest tile that keeps the MMA fed (BK=128, more stages) wins.
        cfg = (128, 128, 128, 8, 4, False, 4)
    BM, BN, BK, GM, ns, ws, nw = cfg
    while BK > 128 and K % BK:     # K must be a whole number of BK steps
        BK //= 2
    return BM, BN, BK, GM, ns, ws, nw


def _fp8_gemm_nt(aq: torch.Tensor, asc: torch.Tensor, w: torch.Tensor,
                 wsi: torch.Tensor, M: int, N: int, K: int) -> torch.Tensor:
    BM, BN, BK, GM, ns, ws, nw = _pick_cfg(M, N, K)
    out = torch.empty(M, N, dtype=torch.bfloat16, device=aq.device)
    ad = TensorDescriptor(aq, [M, K], [K, 1], [BM, BK])
    bd = TensorDescriptor(w, [N, K], [K, 1], [BN, BK])
    cd = TensorDescriptor(out, [M, N], [N, 1], [BM, BN])
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _fp8_bs_gemm[grid](ad, asc, bd, wsi, wsi.stride(0), wsi.stride(1), cd,
                       M, N, K, BM, BN, BK, GM, ns, ws, num_warps=nw)
    return out


class Fp8Linear(nn.Module):
    """Block-scaled FP8 linear: BF16 in, FP8 weight + UE8M0 scales, BF16 out."""

    BLOCK_SIZE = 128

    def __init__(self):
        super().__init__()
        self._a_buf: torch.Tensor | None = None
        self._s_buf: torch.Tensor | None = None
        self._o_buf: torch.Tensor | None = None
        self._pf = None

    def _ensure_buffers(self, max_tokens: int, K: int, N: int,
                        device: torch.device):
        """No-op, kept for API parity with the baseline (L2 callers pre-size
        buffers for CUDA-graph capture).  Intermediates here are allocated per
        call through the caching allocator, which is capture-safe."""

    def forward(self, input_bf16: torch.Tensor,
                weight_fp8: torch.Tensor,
                weight_scale_inv: torch.Tensor,
                bias: torch.Tensor | None = None) -> torch.Tensor:
        N, K = weight_fp8.shape
        input_2d = input_bf16.reshape(-1, K)
        if not input_2d.is_contiguous():
            input_2d = input_2d.contiguous()
        M = input_2d.shape[0]
        dev = input_2d.device

        # The fused path stages the whole activation in shared memory, so it is
        # gated on that fitting inside a 192 KB dynamic-smem budget (sm_100
        # allows 227 KB per SM).
        if (M <= _SMALL_M_MAX and K % 512 == 0
                and weight_scale_inv.stride(0) == 1
                and M * K + M * (K // 128) * 4 <= 192 * 1024):
            # Fused quantize+GEMV: one launch, weight-bandwidth bound.
            output = _C.small_m_linear(input_2d, weight_fp8, weight_scale_inv)
            if bias is not None:
                output = output + bias
            return output.view(*input_bf16.shape[:-1], N)

        aq = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=dev)
        asc = torch.empty(K // self.BLOCK_SIZE, M, dtype=torch.int32, device=dev)
        _C.quant_e8m0_t(input_2d, aq, asc)
        output = _fp8_gemm_nt(aq, asc, weight_fp8, weight_scale_inv, M, N, K)

        if bias is not None:
            output = output + bias
        return output.view(*input_bf16.shape[:-1], N)


# ---------------------------------------------------------------------------
# Weight post-processing (load time only; identical to the baseline).
# ---------------------------------------------------------------------------
def postprocess_fp8_weights(weight_fp8: torch.Tensor,
                            scale_inv: torch.Tensor):
    from fastkernels.tasks.baseline.L1.fp8_linear import (
        postprocess_fp8_weights as _impl,
    )
    return _impl(weight_fp8, scale_inv)


def postprocess_fp8_weights_batched(weight_fp8: torch.Tensor,
                                    scale_inv: torch.Tensor) -> None:
    from fastkernels.tasks.baseline.L1.fp8_linear import (
        postprocess_fp8_weights_batched as _impl,
    )
    return _impl(weight_fp8, scale_inv)
