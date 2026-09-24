"""Optimized YOLOv10 PSA (Partial Self-Attention) block.

Drop-in replacement for ``fastkernels.tasks.baseline.L2.yolov10_psa.YOLOPSA``.

The operator is launch-bound, not compute-bound: ~1.1 GFLOP fp16 (~0.5 us at B200
peak) over ~1.3 MB, against a benched baseline of ~410-480 us (measured with the
harness's own timing recipe -- see ``docs/floor.md``). The baseline spends that on
~30 Python-dispatched ops and ~26 kernels, so the levers are fewer launches and
fewer, cheaper kernels; the arithmetic never matters.

Three paths, selected per call:

1. **Reference** (``_reference_forward``) -- literally the baseline's forward over
   the baseline's own submodules. Handles every shape, dtype, device and training
   mode. Anything the strict predicate rejects lands here, so correctness never
   depends on the optimized paths being general.
2. **PyTorch fast path** (``_fast_forward_torch``) -- BatchNorm folded into the
   weights, a row-major NHWC ``[M, Ch]`` layout that turns every 1x1 convolution
   into a ``torch.addmm``, SDPA on seq-major q/k/v, library depthwise ``pe``.
3. **Extension** -- one pybind call into a CUDA C++ extension that launches all
   six fused kernels from C++. Used when it built; otherwise path 2 takes over.

**Nothing derived from the weights is cached.** An earlier revision kept a packed
blob (relaid-out weights plus a pre-folded BatchNorm affine) validated by a
fingerprint over device, dtype, training mode and ``_version`` counters. Review
reproduced two wrong answers against that design -- a changed ``cv2.bn.eps``
reusing a stale pack (max-abs 0.147), and ``.data`` mutation and child-parameter
replacement both going unobserved -- and the general problem is that a cache needs
a validity proof covering *every* input, which a ``_version`` scan cannot give:
``.data`` deliberately bypasses version counting, and detecting a replaced child
parameter needs a recursive module walk that measured ~125 us per forward, more
than the entire GPU computation.

So the extension consumes the live module state directly instead. It reads the
native ``[Cout, Cin]`` conv weight as a column-major GEMM-B operand, folds
BatchNorm from live ``gamma/beta/running_mean/running_var/eps`` in the kernel
epilogue, maps logical ``[q|k|v]`` columns onto the native head-interleaved ``qkv``
rows inside ``k_qkv``, and indexes the native ``[C,1,3,3]`` ``pe`` taps directly.
Every tensor is resolved from the module on the call that uses it, so staleness is
not merely detected -- it cannot exist. The per-call fingerprint scan is gone with
it, and the PyTorch path builds its operands per call for the same reason.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastkernels.tasks.baseline.L2.yolov10_attention import YOLOAttention
from fastkernels.tasks.baseline.L2.yolov10_conv import YOLOConv

__all__ = ["YOLOPSA"]

# The one configuration the optimized paths are compiled and tiled for. The
# predicate pins every one of these rather than testing divisibility: H*W == 400
# also admits 10x40, which has a different 3x3 `pe` neighbourhood, and
# `self.c % 64 == 0` also admits YOLOPSA(512, 512) with four heads.
_PINNED = dict(c1=256, c2=256, c=128, num_heads=2, head_dim=64, key_dim=32, H=20, W=20)

# Exactly what each of the seven Conv+BatchNorm owners must look like for the
# kernels to be correct, in the order the extension expects them:
# cv1, attn.qkv, attn.pe, attn.proj, ffn[0], ffn[1], cv2.
#
# Checked against the *current* modules on every call, so a replaced owner or a
# reshaped weight routes to the reference path instead of being read with the wrong
# stride. The kernels index these as flat [Cout][K] (and [C][9] for the depthwise
# taps), so the shape is what makes that indexing valid.
_EXPECT = (
    ((256, 256, 1, 1), 256),   # cv1      K=256 -> 256
    ((256, 128, 1, 1), 256),   # attn.qkv K=128 -> 256 (logical [q|k|v])
    ((128, 1, 3, 3), 128),     # attn.pe  depthwise 3x3
    ((128, 128, 1, 1), 128),   # attn.proj
    ((256, 128, 1, 1), 256),   # ffn[0]
    ((128, 256, 1, 1), 128),   # ffn[1]
    ((256, 256, 1, 1), 256),   # cv2      K=256 -> 256
)

# The only batch sizes that are ever benched (docs/shapes.md: two captures at N=4
# deduplicate to one case, plus one at N=1). The tiling is per-sample -- the grid
# is (ceil(H*W/BM), ., N) so no M-tile straddles a sample boundary -- and would
# therefore be correct for any N; pinning the set anyway keeps every benched call
# on a path that has actually been checked. Widening it is a one-line change.
_PINNED_N = (1, 4)


def _fold_bn(conv: nn.Module, bn: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel fp32 ``(scale, bias)`` equivalent to this conv's BatchNorm.

    ``y = x * s + b`` with ``s = g / sqrt(var + eps)`` and
    ``b = beta - g * mean / sqrt(var + eps)``, which is
    ``yolov10_conv._fuse_conv_bn``'s fold expressed as a channel vector instead of
    a scaled weight matrix.

    Computed in fp32 from the *live* buffers. Two things this must not assume: the
    harness leaves BN buffers fp32 while casting parameters to fp16, and
    ``_sanitize_float_params`` happens to leave ``bn.weight == 1``,
    ``running_mean == 0``, ``running_var == 1``. Neither is relied on -- the real
    ``g, beta, mean, var, eps`` are always read, so the module is correct for
    arbitrary loaded state. The CUDA kernels compute this same expression in
    their epilogues from the same live tensors.
    """
    var = bn.running_var.detach().float()
    mean = bn.running_mean.detach().float()
    gamma = bn.weight.detach().float()
    beta = bn.bias.detach().float()
    inv = torch.rsqrt(var + bn.eps)
    return gamma * inv, beta - gamma * mean * inv


class YOLOPSA(nn.Module):
    # Set FK_PSA_NO_FALLBACK=1 to make the reference path raise instead of
    # silently degrading -- turns "correct but 1x" into a hard failure.
    _debug_no_fallback = bool(int(os.environ.get("FK_PSA_NO_FALLBACK", "0")))
    # Set FK_PSA_FORCE_TORCH=1 to keep the PyTorch fast path even when the
    # extension is available (used by the harnesses to diff the two).
    _force_torch = bool(int(os.environ.get("FK_PSA_FORCE_TORCH", "0")))

    def __init__(self, c1: int, c2: int, e: float = 0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        # Baseline submodules, constructed verbatim: identical state_dict keys,
        # and the reference path is then the literal baseline computation.
        self.cv1 = YOLOConv(c1, 2 * self.c, 1, 1)
        self.cv2 = YOLOConv(2 * self.c, c1, 1, 1)
        self.attn = YOLOAttention(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
        self.ffn = nn.Sequential(
            YOLOConv(self.c, self.c * 2, 1, 1),
            YOLOConv(self.c * 2, self.c, 1, 1, act=False),
        )

        # Nothing about the submodules is snapshotted here -- not the owner list,
        # not their BatchNorms, not whether the configuration is supported. Two
        # earlier revisions cached progressively less and were still wrong: caching
        # the folded weights went stale on a weight change, and caching the module
        # objects that hold those weights went stale on `mod.cv1 = <new owner>` or
        # `mod.cv1.bn = <new BN>` (measured max-abs 5.2 and 3.1 respectively, with
        # the fast path still being taken). Anything remembered about a mutable
        # object graph needs a validity proof; resolving it per call needs none.
        # `_current_owners()` is the single place that walks it.

        # Routing evidence for the tests.
        self.n_fast_ext = 0
        self.n_fast_torch = 0
        self.n_fallback = 0

    # ----------------------------------------------------------- live operands

    def _current_owners(self):
        """The seven Conv+BatchNorm owners as they are *right now*.

        Resolved from `self` on every call. This is the only place the submodule
        graph is walked, and it is deliberately not cached: `mod.cv1 = other`,
        `mod.attn.qkv = other` and `mod.cv1.bn = other` are all ordinary PyTorch and
        all silently invalidated a stored list.
        """
        a = self.attn
        return (self.cv1, a.qkv, a.pe, a.proj, self.ffn[0], self.ffn[1], self.cv2)

    def _live_ext_operands(self):
        """The live tensors the kernels read, in the order `psa_run` expects.

        Seven groups of five -- ``(conv.weight, bn.weight, bn.bias,
        bn.running_mean, bn.running_var)`` for ``cv1, qkv, pe, proj, ffn.0, ffn.1,
        cv2`` -- plus the seven live ``bn.eps`` values and ``attn.scale``. Resolved
        fresh here on every call, which is what makes ``.data`` mutation, parameter
        or buffer replacement, a changed ``eps`` and a changed ``scale`` all
        visible without any cache-validity machinery. The C++ side asserts the
        count and each dtype, so a mis-ordered list fails loudly rather than
        silently computing the wrong thing.
        """
        t: list[torch.Tensor] = []
        eps: list[float] = []
        for o in self._current_owners():
            bn = o.bn
            t += [o.conv.weight, bn.weight, bn.bias, bn.running_mean, bn.running_var]
            eps.append(bn.eps)
        return t, eps, self.attn.scale

    def _fast_operands(self, x: torch.Tensor):
        """Live operands for the kernels, or ``None`` if this call is not supported.

        Validation and collection in one pass over the *current* owners, so the
        module graph is walked once per forward and every clause is checked against
        the modules that will actually be read. Returning ``None`` rather than
        raising is what lets an unsupported configuration route to the reference
        path.
        """
        if self.training or torch.is_grad_enabled():
            return None
        if not (x.is_cuda and x.dtype == torch.float16 and not x.requires_grad
                and x.dim() == 4 and x.is_contiguous()):
            return None
        if (x.shape[0] not in _PINNED_N or x.shape[1] != _PINNED["c1"]
                or x.shape[2] != _PINNED["H"] or x.shape[3] != _PINNED["W"]):
            return None
        a = self.attn
        if (self.c != _PINNED["c"] or a.num_heads != _PINNED["num_heads"]
                or a.head_dim != _PINNED["head_dim"] or a.key_dim != _PINNED["key_dim"]):
            return None
        # The kernels are compiled for sm_100 only; without this a Hopper or Ada
        # device would enter the fast path and fail at launch instead of quietly
        # taking the reference path.
        dev = x.device
        if torch.cuda.get_device_capability(dev) != (10, 0):
            return None

        owners = self._current_owners()
        t: list[torch.Tensor] = []
        eps: list[float] = []
        for o, (wshape, chan) in zip(owners, _EXPECT):
            conv = getattr(o, "conv", None)
            bn = getattr(o, "bn", None)
            if conv is None or bn is None:
                return None
            # A BatchNorm in training mode uses batch statistics, which a folded
            # eval-mode affine cannot express at all. Checked on the BN module
            # itself: its parent's `.training` stays False when only the child is
            # switched.
            if bn.training:
                return None
            w = conv.weight
            if (w.dtype != torch.float16 or w.device != dev
                    or not w.is_contiguous() or tuple(w.shape) != wshape):
                return None
            # gamma/beta are fp16 parameters and the running stats are fp32
            # buffers, which is how the bench harness leaves them; anything else
            # routes away rather than being reinterpreted.
            g, b_, mu, var = bn.weight, bn.bias, bn.running_mean, bn.running_var
            for tensor, dt in ((g, torch.float16), (b_, torch.float16),
                               (mu, torch.float32), (var, torch.float32)):
                if (tensor is None or tensor.dtype != dt or tensor.device != dev
                        or not tensor.is_contiguous() or tensor.numel() != chan):
                    return None
            t += [w, g, b_, mu, var]
            eps.append(bn.eps)
        return t, eps, a.scale

    @torch.no_grad()
    def _fold_live(self) -> dict:
        """Operands for the PyTorch fast path, built fresh on every call.

        Cached derived state is what produced the stale-weight defects, so this
        path pays the rebuild rather than reintroducing a cache it cannot validate.
        It is the fallback for configurations the extension does not serve, and it
        is not the timed path when the extension is available.
        """
        a = self.attn
        c, nh, kd, hd = self.c, a.num_heads, a.key_dim, a.head_dim
        f: dict = {}

        def pack(conv, s, b):
            # [K, N] = [Cin, Cout] fp16 with the BN scale folded in -- exactly what
            # the baseline's own YOLOConv.fuse() does, so one addmm per convolution.
            w = conv.weight.detach().float().flatten(1)
            return (w * s[:, None]).t().contiguous().half(), b.half()

        names = ("cv1", "qkv", "pe", "proj", "ffn0", "ffn1", "cv2")
        owners = self._current_owners()
        folds = {n: _fold_bn(o.conv, o.bn) for n, o in zip(names, owners)}

        for n in ("cv1", "proj", "ffn0", "ffn1", "cv2"):
            o = owners[names.index(n)]
            f[n] = pack(o.conv, *folds[n])

        # qkv: permute output rows to [q | k | v] so v becomes a contiguous
        # 128-channel block already in the basis pe/proj/residuals/cv2 live in,
        # and fold the attention scale into q's affine terms (the baseline computes
        # (q^T k) * scale; scaling q, bias included, is the same product).
        per = 2 * kd + hd
        perm = torch.tensor(
            [h * per + i for h in range(nh) for i in range(kd)]
            + [h * per + kd + i for h in range(nh) for i in range(kd)]
            + [h * per + 2 * kd + i for h in range(nh) for i in range(hd)],
            device=a.qkv.conv.weight.device, dtype=torch.long)
        s_q, b_q = folds["qkv"]
        s_q, b_q = s_q[perm].clone(), b_q[perm].clone()
        nq = nh * kd
        s_q[:nq] *= a.scale
        b_q[:nq] *= a.scale

        class _Permuted:
            def __init__(self, w):
                self.weight = w

        f["qkv"] = pack(_Permuted(a.qkv.conv.weight.detach()[perm]), s_q, b_q)

        # Depthwise 3x3: keep NCHW [C,1,3,3] for F.conv2d, scale folded in, BN bias
        # passed as the conv bias.
        s_pe, b_pe = folds["pe"]
        pe_w = a.pe.conv.weight.detach().float()
        f["pe"] = ((pe_w * s_pe[:, None, None, None]).half(), b_pe.half())
        return f

    # ------------------------------------------------------------- predicate

    def _fast_path_ok(self, x: torch.Tensor) -> bool:
        """Does this call match the exact configuration the fast paths are built for?

        Deliberately over-specified. Every clause is an equality against a pinned
        constant rather than a divisibility or product test, because the loose forms
        admit configurations the kernels are wrong for: ``H*W == 400`` also matches
        10x40 (different 3x3 neighbourhood) and ``self.c % 64 == 0`` also matches
        ``YOLOPSA(512, 512)`` (four heads).
        """
        return self._fast_operands(x) is not None

    # --------------------------------------------------------------- forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # One pass over the current module graph: validates and collects together,
        # so the fast path never walks it twice and never validates modules other
        # than the ones it is about to read.
        ops = self._fast_operands(x)
        if ops is None:
            self.n_fallback += 1
            if self._debug_no_fallback:
                raise RuntimeError(
                    f"fast path rejected x[{tuple(x.shape)} {x.dtype} "
                    f"{x.device} contig={x.is_contiguous()}] "
                    f"training={self.training} grad={torch.is_grad_enabled()}"
                )
            return self._reference_forward(x)

        ext = None if self._force_torch else _extension()
        if ext is not None:
            self.n_fast_ext += 1
            t, eps, scale = ops
            return ext.psa_forward(x, t, eps, scale)
        self.n_fast_torch += 1
        return self._fast_forward_torch(x, self._fold_live())

    def _reference_forward(self, x: torch.Tensor) -> torch.Tensor:
        """The baseline computation, verbatim, over the baseline's own submodules."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))

    def _fast_forward_torch(self, x: torch.Tensor, f: dict) -> torch.Tensor:
        """BN-folded NHWC ``torch.mm`` chain: the fallback when the extension is absent.

        Internal layout is row-major ``[M, Ch]`` with ``M = N*H*W`` (i.e. NHWC),
        which makes every 1x1 convolution a plain ``[M,K] @ [K,N]`` GEMM with the K
        axis contiguous, and makes ``v`` seq-major for attention -- the layout
        attention wants, and the reason the baseline's strided ``q/k/v`` views are
        so slow.

        ``cat[M, 256]`` doubles as the concatenation buffer: ``cv1`` writes its
        natural 256-wide output straight in, so ``a`` lands in columns ``0:c`` and
        ``b`` in ``c:2c``; every later stage updates ``c:2c`` in place; ``cv2``
        reads the whole thing. The baseline's ``torch.cat`` disappears.
        """
        n, _, h, w = x.shape
        c, s = self.c, h * w
        m = n * s
        a = self.attn
        nh, kd, hd = a.num_heads, a.key_dim, a.head_dim

        # NCHW -> NHWC [M, 2c].  cv1: GEMM + folded BN + SiLU.
        xn = x.permute(0, 2, 3, 1).reshape(m, -1)
        cat = F.silu(torch.addmm(f["cv1"][1], xn, f["cv1"][0]))
        cat_b = cat[:, c:]                       # a view; every update is in place

        # qkv, rows already permuted to [q | k | v] and q pre-scaled.
        qkv = torch.addmm(f["qkv"][1], cat_b, f["qkv"][0])
        q3 = qkv.view(n, s, -1)
        # [N, S, nh, d] -> [N, nh, S, d]: head_dim stays contiguous.
        q = q3[:, :, : nh * kd].reshape(n, s, nh, kd).transpose(1, 2)
        k = q3[:, :, nh * kd: 2 * nh * kd].reshape(n, s, nh, kd).transpose(1, 2)
        v = q3[:, :, 2 * nh * kd:].reshape(n, s, nh, hd).transpose(1, 2)

        # The scale is already folded into q, hence scale=1.0. SDPA replaces
        # bmm + softmax + bmm and, because it consumes the strided views directly,
        # also three hidden `.contiguous()` copies. It keeps the probabilities in
        # fp32 where the baseline rounds them to fp16, making this path slightly
        # *more* accurate than the baseline rather than less; the CUDA `k_attn`
        # reproduces the baseline's fp16 boundary exactly, and `check_stages.py`
        # compares every stage against the baseline sub-computation itself.
        ctx = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        ctx = ctx.transpose(1, 2).reshape(m, c)

        # Depthwise 3x3 `pe` on v. `v.reshape(N, c, H, W)` in the baseline
        # materializes head0's 64 v-channels then head1's -- exactly the order our
        # permuted v block is already in, so nothing needs reordering downstream.
        v_img = q3[:, :, 2 * nh * kd:].reshape(n, h, w, c).permute(0, 3, 1, 2)
        pe = F.conv2d(v_img, f["pe"][0], f["pe"][1], padding=1, groups=c)
        att = ctx + pe.permute(0, 2, 3, 1).reshape(m, c)

        # proj + residual 1, in place into cat[:, c:].
        cat_b.add_(torch.addmm(f["proj"][1], att, f["proj"][0]))
        # ffn (c -> 2c -> c) + residual 2, in place.
        hid = F.silu(torch.addmm(f["ffn0"][1], cat_b, f["ffn0"][0]))
        cat_b.add_(torch.addmm(f["ffn1"][1], hid, f["ffn1"][0]))

        # cv2 over the full [M, 2c], then NHWC -> contiguous NCHW.
        out = F.silu(torch.addmm(f["cv2"][1], cat, f["cv2"][0]))
        return out.view(n, h, w, -1).permute(0, 3, 1, 2).contiguous()


# --------------------------------------------------------------- extension

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <mutex>
#include <type_traits>
#include <vector>

using namespace nvcuda;

// ---------------------------------------------------------------------------
// Fixed configuration. The Python predicate pins every one of these before the
// extension is ever called, so nothing here is a runtime assumption.
// ---------------------------------------------------------------------------
namespace cfg {
constexpr int H = 20, W = 20, S = H * W;   // 400 positions per sample
constexpr int C2 = 256;                    // 2c, the cat/qkv width
constexpr int C  = 128;                    // c
constexpr int NH = 2, KD = 32, HD = 64;    // heads, key dim, head dim
constexpr int PER = 2 * KD + HD;           // 128: native qkv channels per head
constexpr int NQ = NH * KD;                // 64: logical q columns
constexpr int BM = 32;                     // GEMM M-tile  (400 = 12*32 + 16)
constexpr int BMA = 32;                    // attention query tile
constexpr int NT = 256;                    // threads per CTA: 8 warps, 2x4
constexpr int APAD = 8;                    // keeps every ldm a multiple of 8, as
                                           // wmma requires, while breaking the
                                           // shared-memory conflict stride
constexpr int LDV = HD + APAD;             // 72: staged v. At 64 a row would be
                                           // exactly 32 words and all 16 rows of
                                           // a fragment load would hit one bank.

// Shared-memory region sizes in bytes; all multiples of 32, so carving them out
// of one dynamic allocation in order keeps every wmma fragment offset 32-byte
// aligned. Static __shared__ is capped at 48 KB and k_attn needs far more, so
// every kernel uses dynamic shared memory with a cudaFuncSetAttribute opt-in.
constexpr int SZ_A256  = BM * (C2 + APAD) * 2;     // 16896  A tile, K=256
constexpr int SZ_A128  = BM * (C + APAD) * 2;      //  8704  A tile, K=128
constexpr int SZ_AT256 = C2 * (BM + APAD) * 2;     // 20480  k-major A tile, K=256
constexpr int CPAD     = 4;                        // fp32 ldm must stay a multiple of 4
constexpr int LDC64    = 64 + CPAD;                // 68
constexpr int LDC128   = C + CPAD;                 // 132
constexpr int SZ_C64   = BM * LDC64 * 4;           //  8704  fp32 acc, BN=64
constexpr int SZ_C128  = BM * LDC128 * 4;          // 16896  fp32 acc, BN=128
constexpr int SZ_AFF   = 256 * 2 * 4;              //  2048  per-column BN affine
constexpr int LDS      = S + CPAD;                 // 404: see CPAD above. At 400 the
                                                   // row stride is exactly 400 words
                                                   // (400 % 32 == 16), so a 16-row
                                                   // fragment store hits 2 banks.
constexpr int SZ_SS    = BMA * LDS * 4;            // 51712  fp32 scores
constexpr int SZ_PS    = BMA * S * 2;              // 25600  fp16 probabilities
constexpr int SZ_QS    = BMA * (KD + APAD) * 2;    //  2560  staged q
constexpr int SZ_VS    = S * LDV * 2;              // 57600  staged v
constexpr int SZ_TS    = 9 * HD * 2;               //  1152  staged pe taps

// k_ffn uses a 16-row tile, so its regions are sized independently.
constexpr int SZ_F_A = 16 * (C + APAD) * 2;        //  4352
constexpr int SZ_F_H = 16 * (C2 + APAD) * 2;       //  8448
constexpr int SZ_F_C = 16 * LDC128 * 4;            //  8448

constexpr int SM_CV1  = SZ_AT256 + SZ_C64 + SZ_AFF;                 // 30720
constexpr int SM_QKV  = SZ_A128 + SZ_C64 + SZ_AFF;                  // 18944
constexpr int SM_ATTN = SZ_SS + SZ_PS + SZ_QS + SZ_VS + SZ_TS + SZ_AFF;  // 140160
constexpr int SM_PROJ = SZ_A128 + SZ_C64 + SZ_AFF;                  // 18944
constexpr int SM_FFN  = SZ_F_A + SZ_F_H + SZ_F_C + 2 * SZ_AFF;      // 25088
constexpr int SM_CV2  = SZ_AT256 + SZ_C64 + SZ_AFF;                 // 30720
constexpr int SM_MAX  = SM_ATTN;
}  // namespace cfg

extern __shared__ __align__(32) char g_smem[];

// ---------------------------------------------------------------------------
// Logical -> native qkv channel.
//
// The kernels work in a logical [q(64) | k(64) | v(128)] column order, because
// that makes v a contiguous 128-channel block already in the basis that pe, proj,
// both residuals and cv2 live in. The module's actual qkv weight is
// head-interleaved: head h owns native rows [h*128, (h+1)*128) as q(32) k(32)
// v(64). This maps one to the other *inside the kernel*, so no permuted copy of
// the weight has to exist -- which is what lets the extension read the live
// parameter on every call instead of a derived one whose validity it must prove.
//
// Every boundary here (32, 64, 96, 128, 192) is a multiple of 16, so a 16-column
// wmma tile never straddles one and a whole tile maps to 16 consecutive native
// rows starting at a multiple of 16.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int qkv_native(int j) {
  if (j < cfg::NQ)                                        // q
    return (j / cfg::KD) * cfg::PER + (j % cfg::KD);
  if (j < 2 * cfg::NQ) {                                  // k
    const int t = j - cfg::NQ;
    return (t / cfg::KD) * cfg::PER + cfg::KD + (t % cfg::KD);
  }
  const int t = j - 2 * cfg::NQ;                          // v
  return (t / cfg::HD) * cfg::PER + 2 * cfg::KD + (t % cfg::HD);
}

// ---------------------------------------------------------------------------
// Fold BatchNorm from live parameters into a per-output-column affine.
//
//   s = gamma / sqrt(var + eps)
//   b = beta - gamma * mean / sqrt(var + eps)
//
// Computed in fp32 from the module's live gamma/beta/mean/var/eps, once per CTA
// per output column (not per element), into shared memory. `gamma` and `beta` are
// fp16 parameters while `mean` and `var` are fp32 buffers -- exactly how the bench
// harness leaves them, and read at their real precision either way.
// ---------------------------------------------------------------------------
template <int BN_, bool QKV_MAP>
__device__ __forceinline__ void fill_affine(
    float* __restrict__ sa, float* __restrict__ ba, int n0,
    const __half* __restrict__ gamma, const __half* __restrict__ beta,
    const float* __restrict__ mean, const float* __restrict__ var, float eps,
    float qscale) {
  for (int j = threadIdx.x; j < BN_; j += cfg::NT) {
    const int lc = n0 + j;
    const int ch = QKV_MAP ? qkv_native(lc) : lc;
    const float g = __half2float(gamma[ch]);
    const float inv = rsqrtf(var[ch] + eps);
    float s = g * inv;
    float b = __half2float(beta[ch]) - g * mean[ch] * inv;
    // The attention scale rides on q rather than on the scores: the baseline
    // computes (q^T k) * scale, and scaling q -- including its BN bias -- is the
    // same product with one fewer elementwise pass over 32x400 scores per CTA.
    if (QKV_MAP && lc < cfg::NQ) { s *= qscale; b *= qscale; }
    sa[j] = s;
    ba[j] = b;
  }
}

// ---------------------------------------------------------------------------
// The shared GEMM core: As (fp16, smem) times the module's *native* weight
// Wn[Cout][K] (fp16, global), accumulated in fp32 into Cs[BM][BN] (fp32, smem).
//
// Wn is consumed as a col_major matrix_b with ldm = K, which reads
// B[k][n] = Wn[n][k] straight out of the live [Cout, Cin] parameter. That is the
// whole reason no relayout is needed: element (r, c) of the tile at (k0, nbase)
// is Wn[nbase+c][k0+r] = ptr[c*K + r] with ptr = Wn + nbase*K + k0.
//
// Every instantiation resolves to a 16x16 warp tile = one wmma 16x16x16 fragment
// = 8 fp32 accumulator registers per warp, which is why one routine covers every
// kernel's shape.
// ---------------------------------------------------------------------------
template <int BM_, int BN_, int K_, int WARPS_M, int WARPS_N,
          bool A_COL_MAJOR = false, bool QKV_MAP = false>
__device__ __forceinline__ void mma_tile(
    const __half* __restrict__ As, int lda,
    const __half* __restrict__ Wn, int n0,
    float* __restrict__ Cs, int ldc) {
  constexpr int WT_M = BM_ / WARPS_M;
  constexpr int WT_N = BN_ / WARPS_N;
  constexpr int FM = WT_M / 16;
  constexpr int FN = WT_N / 16;
  const int warp = threadIdx.x >> 5;
  const int wm = warp / WARPS_N, wn = warp % WARPS_N;
  const int row0 = wm * WT_M, col0 = wn * WT_N;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FM][FN];
#pragma unroll
  for (int i = 0; i < FM; ++i)
#pragma unroll
    for (int j = 0; j < FN; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

  using ALayout = typename std::conditional<A_COL_MAJOR, wmma::col_major,
                                            wmma::row_major>::type;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, ALayout> af[FM];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bf[FN];

  // Native row base of each 16-column tile. Hoisted: it depends only on the warp.
  int nbase[FN];
#pragma unroll
  for (int j = 0; j < FN; ++j) {
    const int lc = n0 + col0 + j * 16;
    nbase[j] = QKV_MAP ? qkv_native(lc) : lc;
  }

#pragma unroll
  for (int k = 0; k < K_; k += 16) {
#pragma unroll
    for (int i = 0; i < FM; ++i) {
      // col_major A: As is [K][BM + pad], element (m, k) at k*lda + m.
      // row_major A: As is [BM][K + pad], element (m, k) at m*lda + k.
      const __half* ap = A_COL_MAJOR ? (As + k * lda + row0 + i * 16)
                                     : (As + (row0 + i * 16) * lda + k);
      wmma::load_matrix_sync(af[i], ap, lda);
    }
#pragma unroll
    for (int j = 0; j < FN; ++j)
      wmma::load_matrix_sync(bf[j], Wn + (size_t)nbase[j] * K_ + k, K_);
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
      for (int j = 0; j < FN; ++j)
        wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
  }

  // Through a shared-memory scratch tile rather than straight to global: wmma's
  // fragment element -> (row, col) mapping is implementation-defined, so an
  // epilogue that indexes frag.x[] directly is not portable. Going through
  // scratch also lets k_cv2 turn its NHWC->NCHW scatter into a coalesced write.
#pragma unroll
  for (int i = 0; i < FM; ++i)
#pragma unroll
    for (int j = 0; j < FN; ++j)
      wmma::store_matrix_sync(Cs + (row0 + i * 16) * ldc + col0 + j * 16,
                              acc[i][j], ldc, wmma::mem_row_major);
}

__device__ __forceinline__ float silu(float v) {
  return v / (1.0f + __expf(-v));
}

// Stage a [rows][K] block of a row-major source into As[BM][lda], zero-filling
// the masked tail so the mma never reads garbage. Vectorized 8 halves at a time:
// K, the source stride and lda are all multiples of 8 and every base offset is
// too, so each uint4 is 16-byte aligned.
template <int BM_, int K_>
__device__ __forceinline__ void stage_a_rowmajor(
    __half* As, int lda, const __half* __restrict__ src, int src_ld, int rows) {
  constexpr int V = 8;
  for (int e = threadIdx.x; e < BM_ * (K_ / V); e += cfg::NT) {
    const int m = e / (K_ / V), kv = e - m * (K_ / V);
    const uint4 v = (m < rows)
        ? *reinterpret_cast<const uint4*>(src + (size_t)m * src_ld + kv * V)
        : make_uint4(0u, 0u, 0u, 0u);
    *reinterpret_cast<uint4*>(As + m * lda + kv * V) = v;
  }
}

// ===========================================================================
// 1. k_cv1 : x (NCHW) -> cat[M,256].  GEMM K=256 N=256, live BN, SiLU, with the
//            NCHW->NHWC gather fused into the A-tile staging.
// ===========================================================================
__global__ __launch_bounds__(cfg::NT) void k_cv1(
    const __half* __restrict__ x, const __half* __restrict__ w,
    const __half* __restrict__ g, const __half* __restrict__ bt,
    const float* __restrict__ mu, const float* __restrict__ var, float eps,
    __half* __restrict__ cat) {
  constexpr int BN = 64, K = cfg::C2, LDAT = cfg::BM + cfg::APAD;
  __half* As = reinterpret_cast<__half*>(g_smem);            // [K][BM + APAD]
  float* Cs = reinterpret_cast<float*>(g_smem + cfg::SZ_AT256);
  float* sa = reinterpret_cast<float*>(g_smem + cfg::SZ_AT256 + cfg::SZ_C64);
  float* ba = sa + BN;

  const int n = blockIdx.z, m0 = blockIdx.x * cfg::BM, n0 = blockIdx.y * BN;
  const int rows = min(cfg::BM, cfg::S - m0);

  fill_affine<BN, false>(sa, ba, n0, g, bt, mu, var, eps, 1.0f);

  // NCHW -> NHWC, staged k-major so both sides of the copy are contiguous: in
  // NCHW the 8 spatial positions of a uint4 are adjacent while 8 k values are 400
  // elements apart, so the read must run along m -- and storing As as [K][BM]
  // means the write runs along m too, moving the transpose into the fragment
  // layout where it is free. `rows` is 32 or 16 and m0 is a multiple of 32, so
  // each group of 8 lies wholly inside or wholly outside the sample.
  constexpr int V = 8;
  for (int e = threadIdx.x; e < K * (cfg::BM / V); e += cfg::NT) {
    const int k = e / (cfg::BM / V), mv = e - k * (cfg::BM / V);
    const int m = mv * V;
    const uint4 v = (m + V <= rows)
        ? *reinterpret_cast<const uint4*>(
              x + (size_t)(n * cfg::C2 + k) * cfg::S + m0 + m)
        : make_uint4(0u, 0u, 0u, 0u);
    *reinterpret_cast<uint4*>(As + k * LDAT + m) = v;
  }
  __syncthreads();

  mma_tile<cfg::BM, BN, K, 2, 4, /*A_COL_MAJOR=*/true>(As, LDAT, w, n0, Cs, cfg::LDC64);
  __syncthreads();

  for (int e = threadIdx.x; e < cfg::BM * BN; e += cfg::NT) {
    const int m = e / BN, j = e - m * BN;
    if (m >= rows) continue;
    cat[(size_t)(n * cfg::S + m0 + m) * cfg::C2 + n0 + j] =
        __float2half_rn(silu(Cs[m * cfg::LDC64 + j] * sa[j] + ba[j]));
  }
}

// ===========================================================================
// 2. k_qkv : cat[:,128:256] -> qkv[M,256].  GEMM K=128 N=256, live BN, no act.
//            Output columns are logical [q | k | v]; the weight rows they read
//            are mapped to the module's native head-interleaved layout inside
//            mma_tile and fill_affine, so no permuted weight copy exists.
// ===========================================================================
__global__ __launch_bounds__(cfg::NT) void k_qkv(
    const __half* __restrict__ cat, const __half* __restrict__ w,
    const __half* __restrict__ g, const __half* __restrict__ bt,
    const float* __restrict__ mu, const float* __restrict__ var, float eps,
    float scale, __half* __restrict__ qkv) {
  constexpr int BN = 64, K = cfg::C, LDA = K + cfg::APAD;
  __half* As = reinterpret_cast<__half*>(g_smem);
  float* Cs = reinterpret_cast<float*>(g_smem + cfg::SZ_A128);
  float* sa = reinterpret_cast<float*>(g_smem + cfg::SZ_A128 + cfg::SZ_C64);
  float* ba = sa + BN;

  const int n = blockIdx.z, m0 = blockIdx.x * cfg::BM, n0 = blockIdx.y * BN;
  const int rows = min(cfg::BM, cfg::S - m0);
  const size_t base = (size_t)(n * cfg::S + m0) * cfg::C2;

  fill_affine<BN, true>(sa, ba, n0, g, bt, mu, var, eps, scale);
  stage_a_rowmajor<cfg::BM, K>(As, LDA, cat + base + cfg::C, cfg::C2, rows);
  __syncthreads();

  mma_tile<cfg::BM, BN, K, 2, 4, false, /*QKV_MAP=*/true>(As, LDA, w, n0, Cs, cfg::LDC64);
  __syncthreads();

  for (int e = threadIdx.x; e < cfg::BM * BN; e += cfg::NT) {
    const int m = e / BN, j = e - m * BN;
    if (m >= rows) continue;
    qkv[(size_t)(n * cfg::S + m0 + m) * cfg::C2 + n0 + j] =
        __float2half_rn(Cs[m * cfg::LDC64 + j] * sa[j] + ba[j]);
  }
}

// ===========================================================================
// 3. k_attn : qkv -> att[M,128].  Attention over S=400 plus the fused depthwise
//             3x3 pe, both for one (sample, head).
//
// One CTA owns BMA=32 query rows of one (sample, head). Because it needs all 400
// v rows for P.V anyway, the 3x3 halo is a subset of data it has already touched:
// fusing pe costs no extra global traffic, needs no extra kernel, and creates no
// cross-head coupling, since head h's attention output occupies exactly pe output
// channels h*64 .. h*64+63.
// ===========================================================================
__global__ __launch_bounds__(cfg::NT) void k_attn(
    const __half* __restrict__ qkv, const __half* __restrict__ w_pe,
    const __half* __restrict__ g, const __half* __restrict__ bt,
    const float* __restrict__ mu, const float* __restrict__ var, float eps,
    __half* __restrict__ att) {
  constexpr int LDP = cfg::S;          // 400, a multiple of 8 as wmma requires
  constexpr int LDQ = cfg::KD + cfg::APAD;
  float* Ss = reinterpret_cast<float*>(g_smem);                      // fp32 scores
  __half* Ps = reinterpret_cast<__half*>(g_smem + cfg::SZ_SS);       // fp16 probs
  __half* Qs = reinterpret_cast<__half*>(g_smem + cfg::SZ_SS + cfg::SZ_PS);
  __half* Vs = reinterpret_cast<__half*>(g_smem + cfg::SZ_SS + cfg::SZ_PS
                                         + cfg::SZ_QS);              // [S][LDV]
  __half* Ts = reinterpret_cast<__half*>(g_smem + cfg::SZ_SS + cfg::SZ_PS
                                         + cfg::SZ_QS + cfg::SZ_VS); // [9][HD]
  float* sa = reinterpret_cast<float*>(g_smem + cfg::SZ_SS + cfg::SZ_PS
                                       + cfg::SZ_QS + cfg::SZ_VS + cfg::SZ_TS);
  float* ba = sa + cfg::HD;
  // The P.V accumulator aliases the score tile: scores are dead once the
  // probabilities have been written, and the barrier between makes it safe.
  // 32*64*4 = 8192 B fits inside the 51200 B score tile.
  float* Cs = Ss;

  const int n = blockIdx.z / cfg::NH, h = blockIdx.z % cfg::NH;
  const int m0 = blockIdx.x * cfg::BMA;
  const int rows = min(cfg::BMA, cfg::S - m0);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const size_t sbase = (size_t)(n * cfg::S) * cfg::C2;

  const __half* qbase = qkv + sbase + (size_t)m0 * cfg::C2 + h * cfg::KD;
  const __half* kbase = qkv + sbase + cfg::NQ + h * cfg::KD;
  const __half* vbase = qkv + sbase + 2 * cfg::NQ + h * cfg::HD;

  // pe's BN affine over this head's 64 output channels, from live parameters.
  fill_affine<cfg::HD, false>(sa, ba, h * cfg::HD, g, bt, mu, var, eps, 1.0f);

  // pe taps, native [C][9], staged tap-major so the inner loop reads shared
  // memory coalesced across channels.
  for (int e = threadIdx.x; e < 9 * cfg::HD; e += cfg::NT) {
    const int t = e / cfg::HD, cl = e - t * cfg::HD;
    Ts[t * cfg::HD + cl] = w_pe[(size_t)(h * cfg::HD + cl) * 9 + t];
  }

  // Stage q so the second 16-row fragment of a partial tile reads zeros rather
  // than the next sample's rows (or past the end of the buffer).
  for (int e = threadIdx.x; e < cfg::BMA * cfg::KD; e += cfg::NT) {
    const int i = e / cfg::KD, d = e - i * cfg::KD;
    Qs[i * LDQ + d] = (i < rows) ? qbase[(size_t)i * cfg::C2 + d]
                                 : __float2half_rn(0.0f);
  }
  __syncthreads();

  // --- scores[32][400] = q[32,32] . k[400,32]^T ---------------------------
  // k is consumed as a col_major matrix_b with ldm = C2, which reads
  // k[j0+j][d] straight out of the row-major qkv without any transpose pass.
  {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2];
    // 400 = 25 * 16 exactly; the 8 warps stride the 25 n-tiles so a warp holds
    // only two fragments at a time rather than 2x4.
    for (int jt = warp; jt < cfg::S / 16; jt += 8) {
      wmma::fill_fragment(acc[0], 0.0f);
      wmma::fill_fragment(acc[1], 0.0f);
      for (int d = 0; d < cfg::KD; d += 16) {
        wmma::load_matrix_sync(af[0], Qs + 0 * LDQ + d, LDQ);
        wmma::load_matrix_sync(af[1], Qs + 16 * LDQ + d, LDQ);
        wmma::load_matrix_sync(bf, kbase + (size_t)(jt * 16) * cfg::C2 + d,
                               cfg::C2);
        wmma::mma_sync(acc[0], af[0], bf, acc[0]);
        wmma::mma_sync(acc[1], af[1], bf, acc[1]);
      }
      wmma::store_matrix_sync(Ss + 0 * cfg::LDS + jt * 16, acc[0], cfg::LDS,
                              wmma::mem_row_major);
      wmma::store_matrix_sync(Ss + 16 * cfg::LDS + jt * 16, acc[1], cfg::LDS,
                              wmma::mem_row_major);
    }
  }
  __syncthreads();

  // Stage all 400 v rows for this head. Issued before the softmax so the global
  // loads are in flight while the softmax works the score tile and one barrier
  // covers both. The P.V loop and the fused pe halo then both read shared memory.
  {
    constexpr int V = 8;
    for (int e = threadIdx.x; e < cfg::S * (cfg::HD / V); e += cfg::NT) {
      const int r = e / (cfg::HD / V), jv = e - r * (cfg::HD / V);
      *reinterpret_cast<uint4*>(Vs + r * cfg::LDV + jv * V) =
          *reinterpret_cast<const uint4*>(vbase + (size_t)r * cfg::C2 + jv * V);
    }
  }

  // --- softmax over the 400-wide row, then round P to fp16 ----------------
  // The baseline's score path is half(q@k) -> *scale in fp16 -> softmax (fp32
  // internally) -> fp16 P, so the score is rounded to fp16 and back before the
  // softmax: softmaxing a more precise input is not the same as being closer to
  // the reference. Rounding is applied inline in each pass rather than as a
  // separate sweep over the 51.2 KB tile -- two extra roundings per element are
  // far cheaper than one extra full pass through shared memory.
  for (int i = warp; i < rows; i += 8) {
    const float* row = Ss + i * cfg::LDS;
    float mx = -INFINITY;
    for (int j = lane; j < cfg::S; j += 32)
      mx = fmaxf(mx, __half2float(__float2half_rn(row[j])));
#pragma unroll
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
    float sum = 0.0f;
    for (int j = lane; j < cfg::S; j += 32)
      sum += __expf(__half2float(__float2half_rn(row[j])) - mx);
#pragma unroll
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
    const float inv = 1.0f / sum;
    for (int j = lane; j < cfg::S; j += 32)
      Ps[i * LDP + j] = __float2half_rn(
          __expf(__half2float(__float2half_rn(row[j])) - mx) * inv);
  }
  // Zero the masked tail rows so their fragments contribute nothing.
  for (int e = threadIdx.x; e < (cfg::BMA - rows) * LDP; e += cfg::NT)
    Ps[rows * LDP + e] = __float2half_rn(0.0f);
  __syncthreads();

  // --- ctx[32][64] = P[32,400] . V[400,64] --------------------------------
  {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    // 8 warps over a 32x64 output: warp/4 picks the 16-row block, warp%4 the
    // 16-column block, so each warp carries a single accumulator fragment.
    const int mf = warp >> 2, cb = warp & 3;
    wmma::fill_fragment(acc, 0.0f);
    for (int k = 0; k < cfg::S; k += 16) {
      wmma::load_matrix_sync(af, Ps + mf * 16 * LDP + k, LDP);
      wmma::load_matrix_sync(bf, Vs + k * cfg::LDV + cb * 16, cfg::LDV);
      wmma::mma_sync(acc, af, bf, acc);
    }
    // Ps must be fully consumed by every warp before the accumulator overwrites
    // the aliased score tile.
    __syncthreads();
    wmma::store_matrix_sync(Cs + mf * 16 * cfg::LDC64 + cb * 16, acc, cfg::LDC64,
                            wmma::mem_row_major);
  }
  __syncthreads();

  // --- fused depthwise 3x3 pe on v, + BN, + ctx, one fp16 round -----------
  // Exact 20x20 coordinates with zero padding, never H*W-generic arithmetic:
  // 10x40 has the same H*W but a different neighbourhood. Indices stay inside
  // the sample, so sample boundaries are hard by construction.
  for (int e = threadIdx.x; e < cfg::BMA * cfg::HD; e += cfg::NT) {
    const int i = e / cfg::HD, jj = e - i * cfg::HD;
    if (i >= rows) continue;
    const int p = m0 + i, y = p / cfg::W, xx = p - y * cfg::W;
    float pe = 0.0f;
#pragma unroll
    for (int ky = 0; ky < 3; ++ky) {
      const int yy = y + ky - 1;
      if (yy < 0 || yy >= cfg::H) continue;
#pragma unroll
      for (int kx = 0; kx < 3; ++kx) {
        const int xn = xx + kx - 1;
        if (xn < 0 || xn >= cfg::W) continue;
        pe = fmaf(__half2float(Ts[(ky * 3 + kx) * cfg::HD + jj]),
                  __half2float(Vs[(yy * cfg::W + xn) * cfg::LDV + jj]), pe);
      }
    }
    // The baseline adds two fp16 tensors here: the fp16 attention output and the
    // fp16 output of BN(depthwise conv). Round each to fp16 separately before
    // summing -- one fp32 rounding would be *more* accurate than the reference
    // and would show up as stage disagreement.
    const float ctx_h = __half2float(__float2half_rn(Cs[i * cfg::LDC64 + jj]));
    const float pe_h = __half2float(__float2half_rn(pe * sa[jj] + ba[jj]));
    att[(size_t)(n * cfg::S + p) * cfg::C + h * cfg::HD + jj] =
        __float2half_rn(ctx_h + pe_h);
  }
}

// ===========================================================================
// 4. k_proj : att, cat[:,128:] -> cat[:,128:].  GEMM K=128 N=128 + residual 1.
//
// Safe in place under any tiling, including the output-column split below:
// output (m, j) reads att[m, :] and cat[m, 128+j] and writes cat[m, 128+j] --
// the residual touches the same element it produces, so no CTA can observe
// another's write.
// ===========================================================================
__global__ __launch_bounds__(cfg::NT) void k_proj(
    const __half* __restrict__ att, const __half* __restrict__ w,
    const __half* __restrict__ g, const __half* __restrict__ bt,
    const float* __restrict__ mu, const float* __restrict__ var, float eps,
    __half* __restrict__ cat) {
  constexpr int BN = 64, K = cfg::C, LDA = K + cfg::APAD;
  __half* As = reinterpret_cast<__half*>(g_smem);
  float* Cs = reinterpret_cast<float*>(g_smem + cfg::SZ_A128);
  float* sa = reinterpret_cast<float*>(g_smem + cfg::SZ_A128 + cfg::SZ_C64);
  float* ba = sa + BN;

  const int n = blockIdx.z, m0 = blockIdx.x * cfg::BM, n0 = blockIdx.y * BN;
  const int rows = min(cfg::BM, cfg::S - m0);
  const size_t rbase = (size_t)(n * cfg::S + m0);

  fill_affine<BN, false>(sa, ba, n0, g, bt, mu, var, eps, 1.0f);
  stage_a_rowmajor<cfg::BM, K>(As, LDA, att + rbase * cfg::C, cfg::C, rows);
  __syncthreads();

  mma_tile<cfg::BM, BN, K, 2, 4>(As, LDA, w, n0, Cs, cfg::LDC64);
  __syncthreads();

  for (int e = threadIdx.x; e < cfg::BM * BN; e += cfg::NT) {
    const int m = e / BN, j = e - m * BN;
    if (m >= rows) continue;
    __half* dst = cat + (rbase + m) * cfg::C2 + cfg::C + n0 + j;
    *dst = __float2half_rn(__half2float(*dst)
                           + Cs[m * cfg::LDC64 + j] * sa[j] + ba[j]);
  }
}

// ===========================================================================
// 5. k_ffn : cat[:,128:] -> cat[:,128:].  128 -> 256 (+SiLU) -> 128 + residual 2,
//            with the hidden activation in shared memory, not a global buffer.
//
// Genuinely hazardous, unlike k_proj: this reads all 128 input columns of a row
// and writes all 128 output columns of the same row. It is safe only because
// BN = 128 is the full output width, so one CTA owns every output column for its
// rows. The order below is load-bearing: stage cat[:,128:] -> barrier -> use the
// staged copy as BOTH the GEMM input and the residual -> only then store.
// ===========================================================================
__global__ __launch_bounds__(cfg::NT) void k_ffn(
    const __half* __restrict__ w0, const __half* __restrict__ g0,
    const __half* __restrict__ bt0, const float* __restrict__ mu0,
    const float* __restrict__ var0, float eps0,
    const __half* __restrict__ w1, const __half* __restrict__ g1,
    const __half* __restrict__ bt1, const float* __restrict__ mu1,
    const float* __restrict__ var1, float eps1,
    __half* __restrict__ cat) {
  constexpr int BMF = 16;      // 400 = 25*16 exactly, so no masked tail here
  constexpr int BN = cfg::C, K0 = cfg::C, K1 = cfg::C2;
  constexpr int LDA = K0 + cfg::APAD, LDH = K1 + cfg::APAD;
  __half* As = reinterpret_cast<__half*>(g_smem);
  __half* Hs = reinterpret_cast<__half*>(g_smem + cfg::SZ_F_A);
  float* Cs = reinterpret_cast<float*>(g_smem + cfg::SZ_F_A + cfg::SZ_F_H);
  float* sa = reinterpret_cast<float*>(g_smem + cfg::SZ_F_A + cfg::SZ_F_H
                                       + cfg::SZ_F_C);
  float* ba = sa + cfg::C2;                 // ffn.0 affine, 256 wide
  float* sb = ba + cfg::C2;
  float* bb = sb + cfg::C;                  // ffn.1 affine, 128 wide

  const int n = blockIdx.z, m0 = blockIdx.x * BMF;
  const int rows = min(BMF, cfg::S - m0);
  const size_t rbase = (size_t)(n * cfg::S + m0);

  fill_affine<cfg::C2, false>(sa, ba, 0, g0, bt0, mu0, var0, eps0, 1.0f);
  fill_affine<cfg::C, false>(sb, bb, 0, g1, bt1, mu1, var1, eps1, 1.0f);
  stage_a_rowmajor<BMF, K0>(As, LDA, cat + rbase * cfg::C2 + cfg::C,
                            cfg::C2, rows);
  __syncthreads();

  // ffn.0 : 128 -> 256, live BN, SiLU, into shared memory. Two BN=128 halves
  // reuse the one GEMM routine instead of needing a 256-wide warp arrangement.
#pragma unroll
  for (int half_ = 0; half_ < 2; ++half_) {
    mma_tile<BMF, BN, K0, 1, 8>(As, LDA, w0, half_ * BN, Cs, cfg::LDC128);
    __syncthreads();
    for (int e = threadIdx.x; e < BMF * BN; e += cfg::NT) {
      const int m = e / BN, j = e - m * BN;
      const int col = half_ * BN + j;
      Hs[m * LDH + col] = (m < rows)
          ? __float2half_rn(silu(Cs[m * cfg::LDC128 + j] * sa[col] + ba[col]))
          : __float2half_rn(0.0f);
    }
    __syncthreads();
  }

  // ffn.1 : 256 -> 128, live BN, + residual 2 read from the staged copy.
  mma_tile<BMF, BN, K1, 1, 8>(Hs, LDH, w1, 0, Cs, cfg::LDC128);
  __syncthreads();

  for (int e = threadIdx.x; e < BMF * BN; e += cfg::NT) {
    const int m = e / BN, j = e - m * BN;
    if (m >= rows) continue;
    cat[(rbase + m) * cfg::C2 + cfg::C + j] = __float2half_rn(
        __half2float(As[m * LDA + j]) + Cs[m * cfg::LDC128 + j] * sb[j] + bb[j]);
  }
}

// ===========================================================================
// 6. k_cv2 : cat[M,256] -> out (NCHW).  GEMM K=256 N=256, live BN, SiLU, with
//            the NHWC->NCHW scatter fused into the epilogue.
// ===========================================================================
__global__ __launch_bounds__(cfg::NT) void k_cv2(
    const __half* __restrict__ cat, const __half* __restrict__ w,
    const __half* __restrict__ g, const __half* __restrict__ bt,
    const float* __restrict__ mu, const float* __restrict__ var, float eps,
    __half* __restrict__ out) {
  constexpr int BN = 64, K = cfg::C2, LDA = K + cfg::APAD;
  __half* As = reinterpret_cast<__half*>(g_smem);
  float* Cs = reinterpret_cast<float*>(g_smem + cfg::SZ_A256);
  float* sa = reinterpret_cast<float*>(g_smem + cfg::SZ_A256 + cfg::SZ_C64);
  float* ba = sa + BN;

  const int n = blockIdx.z, m0 = blockIdx.x * cfg::BM, n0 = blockIdx.y * BN;
  const int rows = min(cfg::BM, cfg::S - m0);
  const size_t rbase = (size_t)(n * cfg::S + m0);

  fill_affine<BN, false>(sa, ba, n0, g, bt, mu, var, eps, 1.0f);
  stage_a_rowmajor<cfg::BM, K>(As, LDA, cat + rbase * cfg::C2, cfg::C2, rows);
  __syncthreads();

  mma_tile<cfg::BM, BN, K, 2, 4>(As, LDA, w, n0, Cs, cfg::LDC64);
  __syncthreads();

  // NHWC -> NCHW. Iterating with m innermost makes the global write coalesced
  // (32 contiguous halves); the natural (m, j) order would stride by S*2 = 800 B.
  for (int e = threadIdx.x; e < cfg::BM * BN; e += cfg::NT) {
    const int j = e / cfg::BM, m = e - j * cfg::BM;
    if (m >= rows) continue;
    out[(size_t)(n * cfg::C2 + n0 + j) * cfg::S + m0 + m] =
        __float2half_rn(silu(Cs[m * cfg::LDC64 + j] * sa[j] + ba[j]));
  }
}

// ===========================================================================
// Entry point: one pybind call, all six kernels launched from C++.
// ===========================================================================
// cudaGetLastError() catches launch-configuration failures but not asynchronous
// illegal accesses, which surface later and get attributed to whatever runs next.
// FK_PSA_DEBUG_SYNC=1 synchronizes after every launch so a fault is pinned to the
// kernel that caused it. Off by default: it would serialize the whole chain.
static bool debug_sync() {
  static const bool on = getenv("FK_PSA_DEBUG_SYNC") != nullptr;
  return on;
}

#define LAUNCH_CHECK(name)                                                     \
  do {                                                                         \
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,                             \
                "psa_forward: launch failed: ", name);                         \
    if (debug_sync()) {                                                        \
      cudaError_t e_ = cudaDeviceSynchronize();                                \
      TORCH_CHECK(e_ == cudaSuccess, "psa_forward: ", name, " faulted: ",      \
                  cudaGetErrorString(e_));                                     \
    }                                                                          \
  } while (0)

// Shared-memory opt-in is PER DEVICE. cudaFuncSetAttribute applies to the current
// device's context and the opt-in ceiling is device-specific, so a single
// process-global guard would let the first device's initialization suppress a
// second device's -- and the predicate admits any sm_100 device, including after
// a .to(other_device) move.
static void opt_in_smem(int device) {
  constexpr int kMaxDevices = 64;
  static std::once_flag flags[kMaxDevices];
  static std::mutex fallback_mu;
  auto init = [device] {
    int optin = 0;
    TORCH_CHECK(cudaDeviceGetAttribute(&optin,
                    cudaDevAttrMaxSharedMemoryPerBlockOptin, device)
                    == cudaSuccess,
                "could not query the shared-memory opt-in ceiling for device ",
                device);
    // Checked against the LARGEST request (k_attn), not an arbitrary kernel.
    TORCH_CHECK(optin >= cfg::SM_MAX, "device ", device, " allows only ", optin,
                " bytes of opt-in shared memory per block; the largest kernel "
                "needs ", cfg::SM_MAX);
#define OPT_IN(k, sz)                                                        \
  TORCH_CHECK(cudaFuncSetAttribute(                                          \
                  k, cudaFuncAttributeMaxDynamicSharedMemorySize, (sz))      \
                  == cudaSuccess,                                            \
              "cudaFuncSetAttribute failed for " #k " at ", (sz), " bytes")
    OPT_IN(k_cv1, cfg::SM_CV1);
    OPT_IN(k_qkv, cfg::SM_QKV);
    OPT_IN(k_attn, cfg::SM_ATTN);
    OPT_IN(k_proj, cfg::SM_PROJ);
    OPT_IN(k_ffn, cfg::SM_FFN);
    OPT_IN(k_cv2, cfg::SM_CV2);
#undef OPT_IN
  };
  if (device >= 0 && device < kMaxDevices) {
    std::call_once(flags[device], init);
  } else {
    std::lock_guard<std::mutex> lk(fallback_mu);
    init();
  }
}

// Live operand order, mirrored by _EXPECT in the Python module. Seven groups
// of five: (conv.weight, bn.weight, bn.bias, bn.running_mean, bn.running_var).
enum : int { G_CV1 = 0, G_QKV = 1, G_PE = 2, G_PROJ = 3, G_FFN0 = 4, G_FFN1 = 5,
             G_CV2 = 6, N_GROUPS = 7, G_STRIDE = 5 };

static at::Tensor psa_run(at::Tensor x, const std::vector<at::Tensor>& t,
                          const std::vector<double>& eps, double scale,
                          std::vector<at::Tensor>* snaps) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf, "x must be cuda fp16");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous NCHW");
  TORCH_CHECK(x.dim() == 4 && x.size(1) == cfg::C2 && x.size(2) == cfg::H
                  && x.size(3) == cfg::W, "x must be [N,256,20,20]");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0,
              "x must be 16-byte aligned");
  TORCH_CHECK(t.size() == (size_t)(N_GROUPS * G_STRIDE),
              "expected ", N_GROUPS * G_STRIDE, " live tensors, got ", t.size());
  TORCH_CHECK(eps.size() == (size_t)N_GROUPS, "expected ", N_GROUPS, " eps values");
  // Exact expected extent of every operand. The kernels index these as flat
  // [Cout][K] (and [C][9] for the depthwise taps), so dtype and contiguity alone do
  // not make that indexing valid -- a same-dtype contiguous tensor of the wrong
  // length would be read out of bounds. This is the backstop; the Python predicate
  // checks the same shapes first, so an ordinary forward with an incompatible owner
  // routes to the reference path instead of raising.
  constexpr int64_t kOutC[N_GROUPS] = {cfg::C2, cfg::C2, cfg::C, cfg::C,
                                       cfg::C2, cfg::C, cfg::C2};
  constexpr int64_t kWNumel[N_GROUPS] = {
      (int64_t)cfg::C2 * cfg::C2,     // cv1  256x256
      (int64_t)cfg::C2 * cfg::C,      // qkv  256x128
      (int64_t)cfg::C * 9,            // pe   128x1x3x3
      (int64_t)cfg::C * cfg::C,       // proj 128x128
      (int64_t)cfg::C2 * cfg::C,      // ffn0 256x128
      (int64_t)cfg::C * cfg::C2,      // ffn1 128x256
      (int64_t)cfg::C2 * cfg::C2};    // cv2  256x256
  for (size_t i = 0; i < t.size(); ++i) {
    const int grp = (int)(i / G_STRIDE), slot = (int)(i % G_STRIDE);
    TORCH_CHECK(t[i].is_cuda() && t[i].is_contiguous(),
                "live tensor ", i, " must be cuda and contiguous");
    TORCH_CHECK(t[i].device() == x.device(), "live tensor ", i,
                " is on a different device than x");
    const bool want_f32 = slot >= 3;             // running_mean, running_var
    TORCH_CHECK(t[i].scalar_type() == (want_f32 ? at::kFloat : at::kHalf),
                "live tensor ", i, " has dtype ", t[i].scalar_type(),
                "; expected ", want_f32 ? "float32" : "float16");
    if (slot == 0) {
      TORCH_CHECK(t[i].numel() == kWNumel[grp] && t[i].size(0) == kOutC[grp],
                  "weight ", i, " (group ", grp, ") has shape ", t[i].sizes(),
                  ", numel ", t[i].numel(), "; expected numel ", kWNumel[grp],
                  " with size(0) == ", kOutC[grp]);
    } else {
      TORCH_CHECK(t[i].numel() == kOutC[grp],
                  "BatchNorm operand ", i, " (group ", grp, ") has ", t[i].numel(),
                  " elements; expected ", kOutC[grp]);
    }
  }

  const c10::cuda::OptionalCUDAGuard guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  opt_in_smem(x.device().index());

  const int n = x.size(0);
  const int m = n * cfg::S;
  auto out = at::empty_like(x);

  // ONE flat workspace per call, sliced internally. Measured free, and it removes
  // the entire re-entrancy / cross-stream bug class that a Python-side cached
  // scratch buffer would introduce.
  const int64_t n_cat = (int64_t)m * cfg::C2;
  const int64_t n_qkv = (int64_t)m * cfg::C2;
  const int64_t n_att = (int64_t)m * cfg::C;
  auto ws = at::empty({n_cat + n_qkv + n_att}, x.options());
  auto* p = reinterpret_cast<__half*>(ws.data_ptr());
  __half* cat = p;
  __half* qkv = p + n_cat;
  __half* att = p + n_cat + n_qkv;

  const auto* xp = reinterpret_cast<const __half*>(x.data_ptr());
  auto* op = reinterpret_cast<__half*>(out.data_ptr());

#define HP(i) reinterpret_cast<const __half*>(t[(i)].data_ptr())
#define FP(i) t[(i)].data_ptr<float>()
#define GRP(g) HP((g) * G_STRIDE), HP((g) * G_STRIDE + 1), HP((g) * G_STRIDE + 2), \
               FP((g) * G_STRIDE + 3), FP((g) * G_STRIDE + 4), (float)eps[(g)]

  auto snap = [&](const __half* src, int64_t cols) {
    if (!snaps) return;
    auto s = at::empty({m, cols}, x.options());
    C10_CUDA_CHECK(cudaMemcpyAsync(s.data_ptr(), src,
                                   (size_t)m * cols * sizeof(__half),
                                   cudaMemcpyDeviceToDevice, stream));
    snaps->push_back(s);
  };

  constexpr int MT = (cfg::S + cfg::BM - 1) / cfg::BM;      // 13, last tile masked
  constexpr int MTA = (cfg::S + cfg::BMA - 1) / cfg::BMA;   // 13

  // grid.y is the output-column axis: at BN=64 over 256 output columns a 2-D grid
  // would leave three quarters of every output row unwritten.
  k_cv1<<<dim3(MT, cfg::C2 / 64, n), cfg::NT, cfg::SM_CV1, stream>>>(
      xp, GRP(G_CV1), cat);
  LAUNCH_CHECK("k_cv1");
  snap(cat, cfg::C2);
  k_qkv<<<dim3(MT, cfg::C2 / 64, n), cfg::NT, cfg::SM_QKV, stream>>>(
      cat, GRP(G_QKV), (float)scale, qkv);
  LAUNCH_CHECK("k_qkv");
  snap(qkv, cfg::C2);
  k_attn<<<dim3(MTA, 1, n * cfg::NH), cfg::NT, cfg::SM_ATTN, stream>>>(
      qkv, GRP(G_PE), att);
  LAUNCH_CHECK("k_attn");
  snap(att, cfg::C);
  k_proj<<<dim3(MT, cfg::C / 64, n), cfg::NT, cfg::SM_PROJ, stream>>>(
      att, GRP(G_PROJ), cat);
  LAUNCH_CHECK("k_proj");
  snap(cat, cfg::C2);
  k_ffn<<<dim3(cfg::S / 16, 1, n), cfg::NT, cfg::SM_FFN, stream>>>(
      GRP(G_FFN0), GRP(G_FFN1), cat);
  LAUNCH_CHECK("k_ffn");
  snap(cat, cfg::C2);
  k_cv2<<<dim3(MT, cfg::C2 / 64, n), cfg::NT, cfg::SM_CV2, stream>>>(
      cat, GRP(G_CV2), op);
  LAUNCH_CHECK("k_cv2");
#undef HP
#undef FP
#undef GRP
  return out;
}

at::Tensor psa_forward(at::Tensor x, std::vector<at::Tensor> t,
                       std::vector<double> eps, double scale) {
  return psa_run(x, t, eps, scale, nullptr);
}

// Debug-only: the five stage intermediates plus the output. Not on the timed path.
std::vector<at::Tensor> psa_forward_stages(at::Tensor x, std::vector<at::Tensor> t,
                                           std::vector<double> eps, double scale) {
  std::vector<at::Tensor> snaps;
  auto out = psa_run(x, t, eps, scale, &snaps);
  snaps.push_back(out);
  return snaps;
}
"""

CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>

// Declarations only: load_inline compiles the pybind stub as its own translation
// unit, which cannot see definitions living in the .cu file.
at::Tensor psa_forward(at::Tensor x, std::vector<at::Tensor> t,
                       std::vector<double> eps, double scale);
std::vector<at::Tensor> psa_forward_stages(at::Tensor x, std::vector<at::Tensor> t,
                                           std::vector<double> eps, double scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("psa_forward", &psa_forward,
        "fused YOLOv10 PSA forward (6 kernels, live module state)");
  m.def("psa_forward_stages", &psa_forward_stages,
        "debug: the five stage intermediates plus the output");
}
"""


_EXT: object | None = None
_EXT_TRIED = False
_EXT_ERROR: str | None = None
_EXT_LOCK = threading.Lock()


def _extension():
    """The CUDA extension, or None if it could not be built.

    Built lazily on the first fast-path call rather than at import, and never
    allowed to propagate a failure: if the build breaks, the class must stay
    importable and forwards must degrade to the PyTorch fast path rather than the
    harness seeing no candidate at all. The reason is kept in
    ``_EXT_ERROR`` so a silent degradation is still diagnosable.

    The build directory defaults to a git-ignored ``.torch_extensions/`` inside the
    workspace so repeat runs do not recompile; ``TORCH_EXTENSIONS_DIR`` overrides
    it. ``FK_PSA_NO_EXT`` disables the build (used to test degradation), and
    ``FK_PSA_BREAK_BUILD`` injects a compile error (used to test that a *real*
    build failure degrades rather than raising).
    """
    global _EXT, _EXT_TRIED, _EXT_ERROR
    if _EXT_TRIED:
        return _EXT
    with _EXT_LOCK:
        if _EXT_TRIED:
            return _EXT
        _EXT_TRIED = True
        if os.environ.get("FK_PSA_NO_EXT"):
            _EXT_ERROR = "disabled by FK_PSA_NO_EXT"
            return None
        try:
            os.environ.setdefault(
                "TORCH_EXTENSIONS_DIR",
                str(Path(__file__).resolve().parents[2] / ".torch_extensions"),
            )
            from torch.utils.cpp_extension import load_inline
            cuda = CUDA_SRC
            name = "fk_yolov10_psa"
            if os.environ.get("FK_PSA_BREAK_BUILD"):
                cuda += "\n#error injected build failure (FK_PSA_BREAK_BUILD)\n"
                name += "_broken"
            if os.environ.get("FK_PSA_LINEINFO"):
                # A distinct name, or the cached non-lineinfo .so would be reused
                # and the profiler's source view would be empty.
                name += "_li"
            _EXT = load_inline(
                name=name,
                cpp_sources=CPP_SRC,
                cuda_sources=cuda,
                extra_cuda_cflags=[
                    "-O3", "-std=c++17",
                    "-gencode=arch=compute_100,code=sm_100",
                ] + (["-lineinfo"] if os.environ.get("FK_PSA_LINEINFO") else []),
                verbose=bool(os.environ.get("FK_PSA_VERBOSE_BUILD")),
            )
        except Exception as exc:                      # noqa: BLE001
            _EXT, _EXT_ERROR = None, f"{type(exc).__name__}: {exc}"
        return _EXT
