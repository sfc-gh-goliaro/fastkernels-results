"""Vision patch embedding for Qwen VL models.

Flattens 3D video/image patches via a Conv3d weight reshaped into a linear
projection:  out[M, embed_dim] = x[M, input_size] @ W^T + b.

Unified across Qwen2-VL and Qwen3-VL:
  - bias: Qwen2-VL uses bias=False, Qwen3-VL uses bias=True.

The projection is computed by a hand-written 2-SM (cluster) tcgen05 bf16 GEMM
written in TileLang (``_build_gemm``).  ``F.linear`` is used only for geometries
the kernel cannot tile, which the captured configuration never hits.

Design
------
* A cluster of 2 CTAs cooperates on one output tile of (2*BM) rows x BN cols.
  Each CTA owns BM rows of the accumulator in its own TMEM and holds only *half*
  of the BN columns of B; the tcgen05 ``cta_group::2`` MMA sources the other half
  from the peer CTA's SMEM, so B is fetched from L2 once per CTA pair instead of
  once per CTA, and A rows are never duplicated.
* Warp-specialized, persistent over output tiles:
      warp 0           -> TMA producer (both CTAs, cta_group::2 loads)
      warp 1 (leader)  -> issues the 2-CTA tcgen05 MMA
      warps 4..        -> epilogue: tmem -> reg -> +bias -> bf16 -> smem -> TMA
* Two TMEM accumulators selected by tile parity, so the epilogue of tile i
  overlaps the MMA of tile i+1 (+10% at large M over a single accumulator).
* The epilogue stages through SMEM in N-subtiles (SBN) so the TMA staging buffer
  stays small and a deeper load pipeline fits in SMEM.
* One mbarrier arrival per CTA rather than per producer thread (``lite_arrive``),
  which removes 30 barrier atomics per k-step (+4% at M=25168).

TileLang 0.1.9 constraints that shape the code:
* A TMEM column offset must be a compile-time constant, and layout inference
  additionally rejects a non-zero column min -- so the accumulator is read whole
  (offset 0) into registers and subtiling happens on the register->SMEM leg,
  and accumulator selection is a branch on tile parity, not an indexed slice.
* Nested ``def``s inside a ``T.prim_func`` do not work and ``for i in range(n)``
  is rewritten to ``T.serial``, so meta-level repetition goes through ``T.macro``
  calls guarded by Python-level ``if``s (which the tracer folds eagerly).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.conv3d import Conv3d
from ..L1.linear import Matmul


def _identity(*args, **kwargs):
    """No-op stand-in for the TileLang decorators when TileLang is missing."""
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return _identity


class _NoTileLang:
    def __getattr__(self, name):
        return _identity


try:
    import tilelang
    import tilelang.language as T

    _HAVE_TILELANG = True
except Exception:  # pragma: no cover - the module falls back to F.linear
    tilelang = T = _NoTileLang()
    _HAVE_TILELANG = False


def cdiv(a, b):
    return (a + b - 1) // b


@T.macro
def _mma_all(A_sh, B_sh, acc, loaded, consumed, w, k_blocks, stages, BN):
    """Issue the whole K loop of one output tile into `acc` (leader warp)."""
    for k in T.serial(k_blocks):
        ph = w * k_blocks + k
        T.mbarrier_wait_parity(loaded[ph % stages], (ph // stages) & 1)
        T.tcgen05_gemm(
            A_sh[ph % stages, :, :],
            B_sh[ph % stages, :, :],
            acc,
            transpose_B=True,
            mbar=consumed[ph % stages],
            clear_accum=(k == 0),
            use_2cta=True,
        )


@T.macro
def _epi_store(C_frag, C_sh, Bias, Out, mrow, ncol, BM, BN, SBN, sub, NB, dtype, accum):
    """Stage one N-subtile of the (already register-resident) tile and TMA it out.

    `sub` is a Python literal, so the C_frag column offset folds to a constant.
    """
    for ii, jj in T.Parallel(BM, SBN):
        C_sh[ii, jj] = T.cast(
            C_frag[ii, sub * SBN + jj]
            + T.cast(Bias[T.min(ncol * BN + sub * SBN + jj, NB - 1)], accum), dtype)
    T.copy(C_sh, Out[mrow * BM, ncol * BN + sub * SBN])


@T.macro
def _epi_all(acc, C_frag, C_sh, C_cast, Bias, Out, mrow, ncol, BM, BN, SBN,
             n_sub, use_tma, NB, dtype, accum):
    """Whole-tile epilogue: one TMEM read (offset 0 -- the only offset 0.1.9's
    layout inference accepts), then bias + cast + store."""
    T.copy(acc[:, 0:BN], C_frag)
    if use_tma:
        _epi_store(C_frag, C_sh, Bias, Out, mrow, ncol, BM, BN, SBN, 0, NB, dtype, accum)
        if n_sub > 1:
            _epi_store(C_frag, C_sh, Bias, Out, mrow, ncol, BM, BN, SBN, 1, NB, dtype, accum)
        if n_sub > 2:
            _epi_store(C_frag, C_sh, Bias, Out, mrow, ncol, BM, BN, SBN, 2, NB, dtype, accum)
            _epi_store(C_frag, C_sh, Bias, Out, mrow, ncol, BM, BN, SBN, 3, NB, dtype, accum)
        if n_sub > 4:
            _epi_store(C_frag, C_sh, Bias, Out, mrow, ncol, BM, BN, SBN, 4, NB, dtype, accum)
            _epi_store(C_frag, C_sh, Bias, Out, mrow, ncol, BM, BN, SBN, 5, NB, dtype, accum)
            _epi_store(C_frag, C_sh, Bias, Out, mrow, ncol, BM, BN, SBN, 6, NB, dtype, accum)
            _epi_store(C_frag, C_sh, Bias, Out, mrow, ncol, BM, BN, SBN, 7, NB, dtype, accum)
    else:
        for ii, jj in T.Parallel(BM, BN):
            C_cast[ii, jj] = T.cast(
                C_frag[ii, jj]
                + T.cast(Bias[T.min(ncol * BN + jj, NB - 1)], accum), dtype)
        T.copy(C_cast, Out[mrow * BM, ncol * BN])


@tilelang.jit(pass_configs={"tl.disable_warp_specialized": True})
def _build_gemm(M, N, K, BM, BN, BK, stages, SBN, num_clusters, n_acc=2,
                use_tma=1, lite_arrive=1, dtype="bfloat16", accum="float32",
                threads=256):
    n_blocks = cdiv(N, BN)
    m_clusters = cdiv(M, 2 * BM)
    total_tiles = n_blocks * m_clusters
    k_blocks = cdiv(K, BK)
    num_clusters = min(num_clusters, total_tiles)
    waves = cdiv(total_tiles, num_clusters)
    n_sub = BN // SBN
    n_epi = threads - 128
    assert BN % SBN == 0 and BN % 2 == 0 and K % BK == 0
    assert n_sub in (1, 2, 4, 8)
    assert n_acc in (1, 2)

    @T.prim_func
    def main(
        X: T.Tensor((M, K), dtype),
        Wt: T.Tensor((N, K), dtype),
        Bias: T.Tensor((N,), dtype),
        Out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(2 * num_clusters, threads=threads, cluster_dims=(2, 1, 1)) as bid:
            A_sh = T.alloc_shared((stages, BM, BK), dtype)
            B_sh = T.alloc_shared((stages, BN // 2, BK), dtype)
            acc0 = T.alloc_tmem([BM, BN], accum)
            acc1 = T.alloc_tmem([BM, BN], accum)
            C_frag = T.alloc_fragment((BM, BN), accum)
            C_sh = T.alloc_shared((BM, SBN if use_tma else 8), dtype)
            C_cast = T.alloc_fragment((BM, BN if not use_tma else 8), dtype)

            loaded = T.alloc_cluster_barrier([(2 if lite_arrive else 32 * 2)] * stages)
            consumed = T.alloc_cluster_barrier([1] * stages)
            tmem_full = T.alloc_cluster_barrier([1] * n_acc)
            tmem_empty = T.alloc_cluster_barrier([n_epi * 2] * n_acc)

            tx = T.get_thread_binding()
            cta = T.block_rank_in_cluster()
            T.assume(cta < 2)
            cid = bid // 2

            if tx < 32:  # ---------------- producer: TMA (cta_group::2)
                for w in T.serial(waves):
                    if cid + w * num_clusters < total_tiles:
                        mrow = ((cid + w * num_clusters) // n_blocks) * 2 + cta
                        ncol = (cid + w * num_clusters) % n_blocks
                        for k in T.serial(k_blocks):
                            ph = w * k_blocks + k
                            T.mbarrier_wait_parity(consumed[ph % stages],
                                                   ((ph // stages) & 1) ^ 1)
                            T.tma_copy(
                                X[mrow * BM:(mrow + 1) * BM, k * BK:(k + 1) * BK],
                                A_sh[ph % stages, :, :],
                                barrier=loaded[ph % stages],
                            )
                            T.tma_copy(
                                Wt[(ncol * 2 + cta) * (BN // 2):
                                   (ncol * 2 + cta + 1) * (BN // 2),
                                   k * BK:(k + 1) * BK],
                                B_sh[ph % stages, :, :],
                                barrier=loaded[ph % stages],
                            )
                            if lite_arrive:
                                # one arrival per CTA instead of one per producer
                                # thread: 30 fewer barrier atomics per k-step
                                if tx == 0:
                                    T.mbarrier_arrive(loaded[ph % stages], 0)
                            else:
                                T.mbarrier_arrive(loaded[ph % stages], 0)

            elif tx < 64 and cta == 0:  # ------- leader: issue the 2-CTA MMA
                for w in T.serial(waves):
                    if cid + w * num_clusters < total_tiles:
                        if n_acc == 1:
                            T.mbarrier_wait_parity(tmem_empty[0], (w & 1) ^ 1)
                            _mma_all(A_sh, B_sh, acc0, loaded, consumed, w,
                                     k_blocks, stages, BN)
                            T.tcgen05_mma_arrive(tmem_full[0], arrive_2cta=True)
                        else:
                            T.mbarrier_wait_parity(tmem_empty[w % 2],
                                                   ((w // 2) & 1) ^ 1)
                            if (w % 2) == 0:
                                _mma_all(A_sh, B_sh, acc0, loaded, consumed, w,
                                         k_blocks, stages, BN)
                                T.tcgen05_mma_arrive(tmem_full[0], arrive_2cta=True)
                            else:
                                _mma_all(A_sh, B_sh, acc1, loaded, consumed, w,
                                         k_blocks, stages, BN)
                                T.tcgen05_mma_arrive(tmem_full[1], arrive_2cta=True)

            elif tx >= 128:  # ---------------------------------- epilogue
                for w in T.serial(waves):
                    if cid + w * num_clusters < total_tiles:
                        mrow = ((cid + w * num_clusters) // n_blocks) * 2 + cta
                        ncol = (cid + w * num_clusters) % n_blocks
                        if n_acc == 1:
                            T.mbarrier_wait_parity(tmem_full[0], w & 1)
                            _epi_all(acc0, C_frag, C_sh, C_cast, Bias, Out, mrow,
                                     ncol, BM, BN, SBN, n_sub, use_tma, N, dtype, accum)
                            T.mbarrier_arrive(tmem_empty[0], 0)
                        else:
                            T.mbarrier_wait_parity(tmem_full[w % 2], (w // 2) & 1)
                            if (w % 2) == 0:
                                _epi_all(acc0, C_frag, C_sh, C_cast, Bias, Out, mrow,
                                         ncol, BM, BN, SBN, n_sub, use_tma, N, dtype, accum)
                            else:
                                _epi_all(acc1, C_frag, C_sh, C_cast, Bias, Out, mrow,
                                         ncol, BM, BN, SBN, n_sub, use_tma, N, dtype, accum)
                            T.mbarrier_arrive(tmem_empty[w % 2], 0)

    return main


# ---------------------------------------------------------------------------
# Static M -> config dispatch.  Tuned offline against a replica of
# ``bench._time_module``; no autotuning happens at bench time.
#   cfg = (BM, BN, BK, stages, SBN, n_acc, threads)
# BN=192 tiles N=1152 exactly in 6 and cuts A traffic by a third versus BN=128,
# which wins at every M large enough to fill the machine with 6 N-tiles.  At
# small M the 9 N-tiles of BN=128 are what keep enough clusters busy.
# ---------------------------------------------------------------------------
_CFG_SMALL_M = (128, 128, 64, 8, 32, 2, 256)
_CFG_LARGE_M = (128, 192, 64, 7, 96, 2, 256)
_SMALL_M_MAX = 8192

_KERNELS = {}
_ZERO_BIAS = {}


def _pick_cfg(M):
    return _CFG_SMALL_M if M <= _SMALL_M_MAX else _CFG_LARGE_M


def _get_kernel(M, N, K, device):
    key = (M, N, K)
    kern = _KERNELS.get(key)
    if kern is None:
        BM, BN, BK, stages, SBN, n_acc, threads = _pick_cfg(M)
        n_sms = torch.cuda.get_device_properties(device).multi_processor_count
        kern = _build_gemm(M, N, K, BM, BN, BK, stages, SBN, n_sms // 2,
                           n_acc, 1, 1, threads=threads)
        _KERNELS[key] = kern
    return kern


def _supported(x, weight, M, N, K):
    if not _HAVE_TILELANG:
        return False
    if x.dtype is not torch.bfloat16 or weight.dtype is not torch.bfloat16:
        return False
    if not (x.is_cuda and x.is_contiguous() and weight.is_contiguous()):
        return False
    if torch.cuda.get_device_capability(x.device)[0] != 10:
        return False
    BM, BN, BK, _, _, _, _ = _pick_cfg(M)
    return M > 0 and N % 64 == 0 and N >= BN and K % BK == 0


def _linear(x, weight, bias, fallback):
    M, K = x.shape
    N = weight.shape[0]
    if not _supported(x, weight, M, N, K):
        return fallback(x, weight, bias)
    if bias is None:
        zb = _ZERO_BIAS.get((N, x.device))
        if zb is None:
            zb = torch.zeros(N, dtype=torch.bfloat16, device=x.device)
            _ZERO_BIAS[(N, x.device)] = zb
        bias = zb
    elif bias.dtype is not torch.bfloat16 or not bias.is_contiguous():
        return fallback(x, weight, bias)
    try:
        kern = _get_kernel(M, N, K, x.device)
    except Exception:
        return fallback(x, weight, bias)
    out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    kern(x, weight, bias, out)
    return out


class VisionPatchEmbed(nn.Module):
    def __init__(self, patch_size: int, temporal_patch_size: int,
                 in_channels: int, embed_dim: int, bias: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_size = in_channels * temporal_patch_size * patch_size * patch_size
        kernel = (temporal_patch_size, patch_size, patch_size)
        self.proj = Conv3d(in_channels, embed_dim, kernel, bias=bias)
        self.linear = Matmul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], self.input_size)
        return _linear(
            x,
            self.proj.weight.view(self.embed_dim, self.input_size),
            self.proj.bias,
            self.linear,
        )
