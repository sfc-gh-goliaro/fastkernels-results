"""YOLOv10 CIB (Compact Inverted Block).

The captured configuration (c1 == c2 == 128, e == 1.0, lk=True, 20x20 fp16) is a
strictly sequential chain of five tiny conv+BN+SiLU stages. In eval mode every
BN folds into its conv, and the RepVGGDW 7x7/3x3 pair folds into a single 7x7,
so the whole block becomes five biased convolutions and a residual add. Those
are far too small to keep a GPU busy one layer at a time -- the baseline is
entirely kernel-launch bound -- so the fast path runs all five in ONE kernel and
uses thread-block clusters as the (cheap) cross-block barrier.

Anything else falls back to the reference module.
"""

from __future__ import annotations

import os
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F

from .yolov10_conv import YOLOConv
from .yolov10_repvggdw import YOLORepVGGDW

_CSZ = 16          # blocks per cluster (must divide 128)
# Clusters per image. Each cluster owns a row band and recomputes the rows its
# depthwise layers need from outside it, so small bands trade extra work for
# more blocks; these are the measured optima on B200.
_CPI = {1: 5, 2: 3, 3: 2, 4: 2}
_SCLUSTER = 344064  # halves of global scratch per cluster
_WBUF_N = 80384     # halves
_BBUF_N = 896       # floats

def _invalidate_hook(module, incompatible_keys):
    module._fk_w = None


_DBG = None
if os.environ.get("FK_CIB_DEBUG"):
    import atexit
    _DBG = {}

    def _dump():
        import statistics
        with open(os.path.join(os.path.dirname(__file__), "..", "..", "fk_cib_debug.txt"), "a") as fh:
            fh.write(f"calls={_DBG}\n")
            try:
                import torch as _t
                fh.write(f"cc={_t.cuda.get_device_capability()} name={_t.cuda.get_device_name()}\n")
                for n in list(_DBG):
                    x = _t.randn(n, 128, 20, 20, device="cuda", dtype=_t.float16)
                    mod = _DBG[n][3] if len(_DBG[n]) > 3 else None
            except Exception as e:
                fh.write(f"err {e!r}\n")
    atexit.register(_dump)

_ext_lock = threading.Lock()
_ext = None
_ext_bad = False


def _cuda_source() -> str:
    return _CUDA_SRC


def _get_ext():
    global _ext, _ext_bad
    if _ext is not None or _ext_bad:
        return _ext
    with _ext_lock:
        if _ext is not None or _ext_bad:
            return _ext
        try:
            from torch.utils.cpp_extension import load_inline
            prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
            cc = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{cc[0]}.{cc[1]}a"
            try:
                _ext = load_inline(
                    name="fk_yolov10_cib_fused",
                    cpp_sources=_CPP_SRC,
                    cuda_sources=_CUDA_SRC,
                    functions=["cib_run"],
                    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
                    verbose=False,
                )
            finally:
                if prev is None:
                    os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
                else:
                    os.environ["TORCH_CUDA_ARCH_LIST"] = prev
        except Exception as exc:  # pragma: no cover - fall back to eager reference
            if os.environ.get("FK_CIB_DEBUG"):
                import traceback
                traceback.print_exc()
            _ext = None
            _ext_bad = True
    return _ext


def _fuse(conv: nn.Module, bn: nn.Module):
    """conv(+no bias) followed by eval-mode BN -> (weight, bias) in float32."""
    w = conv.weight.detach().float()
    rv = bn.running_var.detach().float()
    rm = bn.running_mean.detach().float()
    gamma = bn.weight.detach().float()
    beta = bn.bias.detach().float()
    scale = gamma / torch.sqrt(rv + bn.eps)
    fw = w * scale.view(-1, *([1] * (w.dim() - 1)))
    fb = beta - rm * scale
    if conv.bias is not None:
        fb = fb + conv.bias.detach().float() * scale
    return fw, fb




class YOLOCIB(nn.Module):
    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, lk: bool = False):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Sequential(
            YOLOConv(c1, c1, 3, g=c1),
            YOLOConv(c1, 2 * c_, 1),
            YOLOConv(2 * c_, 2 * c_, 3, g=2 * c_) if not lk else YOLORepVGGDW(2 * c_),
            YOLOConv(2 * c_, c2, 1),
            YOLOConv(c2, c2, 3, g=c2),
        )
        self.add = shortcut and c1 == c2
        self._fk_ok = (c1 == 128 and c2 == 128 and int(c2 * e) == 128 and lk
                       and bool(self.add))
        self._fk_w = None
        self._fk_b = None
        self._fk_src = None     # tensors the fused weights were derived from
        self._fk_ver = None     # their in-place version counters
        self._fk_scratch = None
        if self._fk_ok:
            self.register_load_state_dict_post_hook(_invalidate_hook)

    # Any structural change (.to(), .half(), .cuda(), load_state_dict) rebuilds
    # the folded weights; in-place edits are caught by the version counters.
    def _apply(self, *args, **kwargs):
        self._fk_w = None
        return super()._apply(*args, **kwargs)

    def train(self, mode: bool = True):
        self._fk_w = None
        return super().train(mode)

    # ---------------- fast path -------------------------------------------
    def _fk_prepare(self, device):
        seq = self.cv1
        w1, b1 = _fuse(seq[0].conv, seq[0].bn)
        w2, b2 = _fuse(seq[1].conv, seq[1].bn)
        w3a, b3a = _fuse(seq[2].conv.conv, seq[2].conv.bn)
        w3b, b3b = _fuse(seq[2].conv1.conv, seq[2].conv1.bn)
        w3 = w3a + F.pad(w3b, [2, 2, 2, 2])
        b3 = b3a + b3b
        w4, b4 = _fuse(seq[3].conv, seq[3].bn)
        w5, b5 = _fuse(seq[4].conv, seq[4].bn)
        # layouts: dw -> [taps][C], pw -> [cout][cin]
        parts = [
            w1.reshape(128, 9).t().contiguous().reshape(-1),
            w2.reshape(256, 128).reshape(-1),
            w3.reshape(256, 49).t().contiguous().reshape(-1),
            w4.reshape(128, 256).reshape(-1),
            w5.reshape(128, 9).t().contiguous().reshape(-1),
        ]
        wbuf = torch.cat(parts).to(device=device, dtype=torch.float16)
        bbuf = torch.cat([b1, b2, b3, b4, b5]).to(device=device, dtype=torch.float32)
        assert wbuf.numel() == _WBUF_N and bbuf.numel() == _BBUF_N
        self._fk_w = wbuf
        self._fk_b = bbuf
        src = []
        for sub in (seq[0], seq[1], seq[2].conv, seq[2].conv1, seq[3], seq[4]):
            src.append(sub.conv.weight)
            src.extend((sub.bn.weight, sub.bn.bias, sub.bn.running_mean, sub.bn.running_var))
        self._fk_src = src
        self._fk_ver = [t._version for t in src]

    def _fk_forward(self, x: torch.Tensor):
        ext = _get_ext()
        if ext is None:
            return None
        n = x.shape[0]
        cpi = _CPI.get(n, 1)
        if self._fk_w is None or self._fk_w.device != x.device or any(
                t._version != v for t, v in zip(self._fk_src, self._fk_ver)):
            self._fk_prepare(x.device)
        nclus = n * cpi
        scr = self._fk_scratch
        if scr is None or scr.numel() < nclus * _SCLUSTER or scr.device != x.device:
            scr = torch.zeros(nclus * _SCLUSTER, dtype=torch.float16, device=x.device)
            self._fk_scratch = scr
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        ext.cib_run(x, out, self._fk_w, self._fk_b, scr, cpi)
        if _DBG is not None:
            _DBG.setdefault(n, [0, cpi, nclus * 16])
            _DBG[n][0] += 1
        return out

    # ---------------- reference -------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (self._fk_ok and not self.training and x.is_cuda
                and x.dtype == torch.float16 and x.dim() == 4
                and x.shape[1] == 128 and x.shape[2] == 20 and x.shape[3] == 20
                and x.stride(3) == 1 and x.stride(2) == 20 and x.stride(1) == 400
                and (x.shape[0] == 1 or x.stride(0) >= 51200)):
            out = self._fk_forward(x)
            if out is not None:
                return out
        y = self.cv1(x)
        return x + y if self.add else y


_CPP_SRC = r"""
#include <torch/extension.h>
void cib_run(torch::Tensor x, torch::Tensor out, torch::Tensor wb,
             torch::Tensor bb, torch::Tensor scr, int64_t cpi);
"""

_CUDA_SRC = r"""
// Fused YOLOv10 CIB (c1=c2=128, e=1.0, lk=True) forward, 20x20 fp16, one kernel.
//
// BN folded into each conv; RepVGGDW's 7x7 + 3x3 folded into one 7x7. Chain:
//   t1 = silu(dw3x3(x)  + b1)  128ch      t2 = silu(pw(t1) + b2)  128->256
//   t3 = silu(dw7x7(t2) + b3)  256ch      t4 = silu(pw(t3) + b4)  256->128
//   out = x + silu(dw3x3(t4) + b5)
//
// The block is tiny (400 px/image) so latency, not throughput, dominates. Hence:
// one launch, thread-block clusters as the cross-block barrier (~10x cheaper
// than a grid-wide barrier), every operand staged into shared memory in bulk,
// and zero-padded shared tiles so the depthwise inner loops are branch-free.
// Each cluster owns a horizontal band of one image and recomputes the (cheap)
// rows its depthwise layers need from outside the band, so clusters never talk.
#include <cuda_fp16.h>
#include <mma.h>
#include <cooperative_groups.h>
#include <cuda_pipeline.h>

namespace cg = cooperative_groups;
using namespace nvcuda;

#define IH 20
#define IW 20
#define IP 400
#define CC1 128
#define CC2 256
#define CSZ 16
#ifndef CPASYNC
#define CPASYNC 1
#endif
#ifndef NTHR
#define NTHR 256
#endif
#ifndef PH3_2ROW
#define PH3_2ROW 1
#endif
#ifndef PW_STORE_H2
#define PW_STORE_H2 1
#endif
#ifndef PH4_KC
#define PH4_KC 256
#endif
#define NWARP (NTHR / 32)
#define PPAD 448        // pixel slack in the scratch slab (pw tiles overrun npix)

#define OFF_W1 0        // [9][128]
#define OFF_W2 1152     // [256][128]
#define OFF_W3 33920    // [49][256]
#define OFF_W4 46464    // [128][256]
#define OFF_W5 79232    // [9][128]
#define WBUF_N 80384

#define BO1 0
#define BO2 128
#define BO3 384
#define BO4 640
#define BO5 768
#define BBUF_N 896

#define S_T1 0
#define S_T2 (PPAD * CC1)
#define S_T3 (S_T2 + PPAD * CC2)
#define S_T4 (S_T3 + PPAD * CC2)
#define S_CLUSTER (S_T4 + PPAD * CC1)

#define SMEM_B ((PH4_KC == 256) ? 50688 : 39680)

__device__ __forceinline__ float silu_f(float v) {
    float t;
    asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(v * 0.5f));
    return 0.5f * v * (1.0f + t);
}

__device__ __forceinline__ __half2 silu_h2(__half2 v) {
    __half2 h = __hmul2(v, __float2half2_rn(0.5f));
    unsigned hi = *reinterpret_cast<unsigned *>(&h), to;
    asm("tanh.approx.f16x2 %0, %1;" : "=r"(to) : "r"(hi));
    __half2 t = *reinterpret_cast<__half2 *>(&to);
    return __hmul2(h, __hadd2(t, __float2half2_rn(1.0f)));
}

// ---------------------------------------------------------------------------
// Pointwise layer: dst[p][cout] = silu(sum_k W[cout][k] * src[p][k] + bias).
// Blocks of the cluster take (cout group, pixel group) items; both operands are
// staged in shared memory so the mma fragments come from ldmatrix, not global.
// ---------------------------------------------------------------------------
template <int KDIM, int COUTALL, int CB, int PB, int KC>
__device__ __forceinline__ void pw_layer(const __half *__restrict__ Wg,
                                         const __half *__restrict__ src,
                                         __half *__restrict__ dst,
                                         const float *__restrict__ bias,
                                         int npix, int rank, int warp, int lane,
                                         unsigned char *smem) {
    const int KP = KC + 8;
    __half *sW = (__half *)smem;
    __half *sB = sW + CB * KP;
    float *sA = (float *)smem;   // aliased: sW/sB are dead once the K loop ends
    const int NCT = CB / 16, NPT = PB / 16;
    const int NTILE = NCT * NPT;
    const int TPW = (NTILE + NWARP - 1) / NWARP;
    const int NCG = COUTALL / CB;
    const int tid = threadIdx.x;
    const int NPG = (npix + PB - 1) / PB;

    for (int item = rank; item < NCG * NPG; item += CSZ) {
        const int cg = item % NCG, pg = item / NCG;
        const int c0 = cg * CB, p0 = pg * PB;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[TPW];
#pragma unroll
        for (int u = 0; u < TPW; ++u) wmma::fill_fragment(acc[u], 0.0f);

        for (int kc = 0; kc < KDIM; kc += KC) {
            __syncthreads();
#if CPASYNC
#pragma unroll
            for (int i = tid; i < CB * (KC / 8); i += NTHR) {
                const int r = i / (KC / 8), j = i - r * (KC / 8);
                __pipeline_memcpy_async(sW + r * KP + j * 8,
                                        Wg + (size_t)(c0 + r) * KDIM + kc + j * 8, 16);
            }
#pragma unroll
            for (int i = tid; i < PB * (KC / 8); i += NTHR) {
                const int r = i / (KC / 8), j = i - r * (KC / 8);
                __pipeline_memcpy_async(sB + r * KP + j * 8,
                                        src + (size_t)(p0 + r) * KDIM + kc + j * 8, 16);
            }
            __pipeline_commit();
            __pipeline_wait_prior(0);
#else
#pragma unroll
            for (int i = tid; i < CB * (KC / 8); i += NTHR) {
                const int r = i / (KC / 8), j = i - r * (KC / 8);
                *(uint4 *)(sW + r * KP + j * 8) =
                    *(const uint4 *)(Wg + (size_t)(c0 + r) * KDIM + kc + j * 8);
            }
#pragma unroll
            for (int i = tid; i < PB * (KC / 8); i += NTHR) {
                const int r = i / (KC / 8), j = i - r * (KC / 8);
                *(uint4 *)(sB + r * KP + j * 8) =
                    *(const uint4 *)(src + (size_t)(p0 + r) * KDIM + kc + j * 8);
            }
#endif
            __syncthreads();
#pragma unroll
            for (int u = 0; u < TPW; ++u) {
                const int tl = warp + u * NWARP;
                if (tl >= NTILE) break;
                const int ct = tl % NCT, pt = tl / NCT;
#pragma unroll
                for (int k = 0; k < KC; k += 16) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                    wmma::load_matrix_sync(fa, sW + (ct * 16) * KP + k, KP);
                    wmma::load_matrix_sync(fb, sB + (pt * 16) * KP + k, KP);
                    wmma::mma_sync(acc[u], fa, fb, acc[u]);
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (int u = 0; u < TPW; ++u) {
            const int tl = warp + u * NWARP;
            if (tl >= NTILE) break;
            const int ct = tl % NCT, pt = tl / NCT;
            float *sa = sA + warp * 256;
            wmma::store_matrix_sync(sa, acc[u], 16, wmma::mem_col_major);
            __syncwarp();
#if PW_STORE_H2
            // two adjacent cout per lane: 8B shared read, one f16x2 tanh, 4B store
            const int mh = (lane & 7) << 1;
            const int n0 = lane >> 3;
            const float b0 = bias[c0 + ct * 16 + mh];
            const float b1 = bias[c0 + ct * 16 + mh + 1];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int nn = n0 + (r << 2);
                const float2 v = *(const float2 *)(sa + mh + nn * 16);
                const __half2 h = __floats2half2_rn(v.x + b0, v.y + b1);
                *(__half2 *)(dst + (size_t)(p0 + pt * 16 + nn) * COUTALL + c0 + ct * 16 + mh) =
                    silu_h2(h);
            }
#else
            const int m = lane & 15;
            const float bv = bias[c0 + ct * 16 + m];
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int idx = lane + (r << 5);
                const int nn = idx >> 4;
                dst[(size_t)(p0 + pt * 16 + nn) * COUTALL + c0 + ct * 16 + m] =
                    __float2half(silu_f(sa[idx] + bv));
            }
#endif
            __syncwarp();
        }
    }
}

extern "C" __global__ void __cluster_dims__(CSZ, 1, 1) __launch_bounds__(NTHR)
cib_kernel(const __half *__restrict__ X, __half *__restrict__ OUT,
           const __half *__restrict__ WB, const float *__restrict__ BB,
           __half *__restrict__ SCR, int CPI, long long XBS) {
    extern __shared__ __align__(128) unsigned char smem[];
    cg::cluster_group cl = cg::this_cluster();

    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int rank = blockIdx.x % CSZ;
    const int clid = blockIdx.x / CSZ;
    const int img = clid / CPI, cid = clid % CPI;

    const int R0 = cid * IH / CPI, R1 = (cid + 1) * IH / CPI;
    const int A0 = R0 > 1 ? R0 - 1 : 0, A1 = R1 + 1 < IH ? R1 + 1 : IH;
    const int B0 = A0 > 3 ? A0 - 3 : 0, B1 = A1 + 3 < IH ? A1 + 3 : IH;
    const int NPB = (B1 - B0) * IW, NPA = (A1 - A0) * IW;

    const __half *xn = X + (long long)img * XBS;
    __half *on = OUT + (size_t)img * CC1 * IP;
    __half *t1 = SCR + (size_t)clid * S_CLUSTER + S_T1;
    __half *t2 = SCR + (size_t)clid * S_CLUSTER + S_T2;
    __half *t3 = SCR + (size_t)clid * S_CLUSTER + S_T3;
    __half *t4 = SCR + (size_t)clid * S_CLUSTER + S_T4;

    // ---- PH1: dw3x3 on x -> t1 (channel split, zero-padded shared tile) ----
    {
        const int CPB = CC1 / CSZ;      // 8 channels
        const int NPR1 = CPB >> 1;      // 4 half2 lanes
        const int SR = (B1 - B0) + 2;   // shared rows, image row B0-1+rr
        const int CP = 22;              // shared cols, image col cc-1
        const int c0 = rank * CPB;
        unsigned *st = (unsigned *)smem;
        const int ntot = NPR1 * SR * CP;
        for (int i = tid; i < ntot; i += NTHR) st[i] = 0u;
        __syncthreads();
        for (int i = tid; i < (B1 - B0 + 2) * IW; i += NTHR) {
            const int rr = i / IW, xx = i - rr * IW;
            const int y = B0 - 1 + rr;
            if (y < 0 || y >= IH) continue;
            const __half *xp = xn + (size_t)c0 * IP + y * IW + xx;
            __half va[NPR1 * 2];
#pragma unroll
            for (int j = 0; j < NPR1 * 2; ++j) va[j] = xp[(size_t)j * IP];
#pragma unroll
            for (int j = 0; j < NPR1; ++j) {
                const __half2 ab = __halves2half2(va[2 * j], va[2 * j + 1]);
                st[(j * SR + rr) * CP + xx + 1] = *(const unsigned *)&ab;
            }
        }
        __syncthreads();
        const int j = tid & (NPR1 - 1);
        const int pg = tid / NPR1;
        unsigned wr[9];
#pragma unroll
        for (int t = 0; t < 9; ++t)
            wr[t] = *(const unsigned *)(WB + OFF_W1 + t * CC1 + c0 + 2 * j);
        const __half2 bias =
            __floats2half2_rn(BB[BO1 + c0 + 2 * j], BB[BO1 + c0 + 2 * j + 1]);
        const unsigned *sp = st + j * SR * CP;
        for (int i = pg; i < NPB; i += NTHR / NPR1) {
            const int rr = i / IW, xx = i - rr * IW;
            const unsigned *base = sp + rr * CP + xx;  // == row(rr+1-1), col(xx+1-1)
            __half2 acc = bias;
#pragma unroll
            for (int dy = 0; dy < 3; ++dy) {
                const unsigned *row = base + dy * CP;
#pragma unroll
                for (int dx = 0; dx < 3; ++dx) {
                    const unsigned v = row[dx];
                    acc = __hfma2(*(const __half2 *)&wr[dy * 3 + dx],
                                  *(const __half2 *)&v, acc);
                }
            }
            *(__half2 *)(t1 + (size_t)i * CC1 + c0 + 2 * j) = silu_h2(acc);
        }
    }
    __syncthreads();
    cl.sync();

    // ---- PH2: pw 128 -> 256 -------------------------------------------------
    pw_layer<CC1, CC2, 64, 80, 128>(WB + OFF_W2, t1, t2, BB + BO2, NPB, rank,
                                    warp, lane, smem);
    __syncthreads();
    cl.sync();

    // ---- PH3: dw7x7 on t2 -> t3 (channel split) ----------------------------
    {
        const int CPB = CC2 / CSZ;      // 16 channels
        const int NP2 = CPB >> 1;       // 8 half2 lanes
        const int SR = (B1 - B0) + 6;   // image row B0-3+rr
        const int CP = 26;              // image col cc-3
        const int c0 = rank * CPB;
        unsigned *st = (unsigned *)smem;
        const int ntot = NP2 * SR * CP;
        for (int i = tid; i < ntot; i += NTHR) st[i] = 0u;
        __syncthreads();
        for (int i = tid; i < NPB; i += NTHR) {
            const int rr = i / IW, xx = i - rr * IW;
            const unsigned *src = (const unsigned *)(t2 + (size_t)i * CC2 + c0);
#pragma unroll
            for (int j = 0; j < NP2; ++j)
#if CPASYNC
                __pipeline_memcpy_async(&st[(j * SR + rr + 3) * CP + xx + 3], &src[j], 4);
#else
                st[(j * SR + rr + 3) * CP + xx + 3] = src[j];
#endif
        }
#if CPASYNC
        __pipeline_commit();
        __pipeline_wait_prior(0);
#endif
        __syncthreads();
        const int j = tid & (NP2 - 1);
        const int tk = tid / NP2;       // 0..31
        unsigned wr[49];
#pragma unroll
        for (int t = 0; t < 49; ++t)
            wr[t] = *(const unsigned *)(WB + OFF_W3 + t * CC2 + c0 + 2 * j);
        const __half2 bias =
            __floats2half2_rn(BB[BO3 + c0 + 2 * j], BB[BO3 + c0 + 2 * j + 1]);
        const unsigned *sp = st + j * SR * CP;
#if PH3_2ROW
        const int nrp = (A1 - A0 + 1) >> 1;      // row pairs
        const int ntask = nrp * 5;
        for (int task = tk; task < ntask; task += NTHR / NP2) {
            const int rp = task / 5, gx = task - rp * 5;
            const int ry = rp << 1;
            const int y = A0 + ry, x0 = gx * 4;
            const bool two = (ry + 1) < (A1 - A0);
            const unsigned *base = sp + (y - B0) * CP + x0;
            __half2 a0 = bias, a1 = bias, a2 = bias, a3 = bias;
            __half2 c0_ = bias, c1_ = bias, c2_ = bias, c3_ = bias;
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const unsigned *row = base + r * CP;
                unsigned v[10];
#pragma unroll
                for (int t = 0; t < 10; ++t) v[t] = row[t];
                if (r < 7) {
#pragma unroll
                    for (int dx = 0; dx < 7; ++dx) {
                        const __half2 w = *(const __half2 *)&wr[r * 7 + dx];
                        a0 = __hfma2(w, *(const __half2 *)&v[dx], a0);
                        a1 = __hfma2(w, *(const __half2 *)&v[dx + 1], a1);
                        a2 = __hfma2(w, *(const __half2 *)&v[dx + 2], a2);
                        a3 = __hfma2(w, *(const __half2 *)&v[dx + 3], a3);
                    }
                }
                if (two && r > 0) {
#pragma unroll
                    for (int dx = 0; dx < 7; ++dx) {
                        const __half2 w = *(const __half2 *)&wr[(r - 1) * 7 + dx];
                        c0_ = __hfma2(w, *(const __half2 *)&v[dx], c0_);
                        c1_ = __hfma2(w, *(const __half2 *)&v[dx + 1], c1_);
                        c2_ = __hfma2(w, *(const __half2 *)&v[dx + 2], c2_);
                        c3_ = __hfma2(w, *(const __half2 *)&v[dx + 3], c3_);
                    }
                }
            }
            __half2 *dst = (__half2 *)(t3 + (size_t)(ry * IW + x0) * CC2 + c0 + 2 * j);
            dst[0] = silu_h2(a0);
            dst[CC2 / 2] = silu_h2(a1);
            dst[CC2] = silu_h2(a2);
            dst[3 * CC2 / 2] = silu_h2(a3);
            if (two) {
                __half2 *d2 = dst + (IW * CC2) / 2;
                d2[0] = silu_h2(c0_);
                d2[CC2 / 2] = silu_h2(c1_);
                d2[CC2] = silu_h2(c2_);
                d2[3 * CC2 / 2] = silu_h2(c3_);
            }
        }
#else
        const int ntask = (A1 - A0) * 5;
        for (int task = tk; task < ntask; task += NTHR / NP2) {
            const int ry = task / 5, gx = task - ry * 5;
            const int y = A0 + ry, x0 = gx * 4;
            const unsigned *base = sp + (y - B0) * CP + x0;
            __half2 a0 = bias, a1 = bias, a2 = bias, a3 = bias;
#pragma unroll
            for (int dy = 0; dy < 7; ++dy) {
                const unsigned *row = base + dy * CP;
                unsigned v[10];
#pragma unroll
                for (int t = 0; t < 10; ++t) v[t] = row[t];
#pragma unroll
                for (int dx = 0; dx < 7; ++dx) {
                    const __half2 w = *(const __half2 *)&wr[dy * 7 + dx];
                    a0 = __hfma2(w, *(const __half2 *)&v[dx], a0);
                    a1 = __hfma2(w, *(const __half2 *)&v[dx + 1], a1);
                    a2 = __hfma2(w, *(const __half2 *)&v[dx + 2], a2);
                    a3 = __hfma2(w, *(const __half2 *)&v[dx + 3], a3);
                }
            }
            __half2 *dst = (__half2 *)(t3 + (size_t)(ry * IW + x0) * CC2 + c0 + 2 * j);
            dst[0] = silu_h2(a0);
            dst[CC2 / 2] = silu_h2(a1);
            dst[CC2] = silu_h2(a2);
            dst[3 * CC2 / 2] = silu_h2(a3);
        }
#endif
    }
    __syncthreads();
    cl.sync();

    // ---- PH4: pw 256 -> 128 ------------------------------------------------
    pw_layer<CC2, CC1, 32, 64, PH4_KC>(WB + OFF_W4, t3, t4, BB + BO4, NPA, rank,
                                    warp, lane, smem);
    __syncthreads();
    cl.sync();

    // ---- PH5: dw3x3 on t4 + residual -> out --------------------------------
    {
        const int CPB = CC1 / CSZ;      // 8 channels
        const int NP2 = CPB >> 1;       // 4 half2 lanes
        const int SR = (A1 - A0) + 2;   // image row A0-1+rr
        const int CP = 22;
        const int c0 = rank * CPB;
        unsigned *st = (unsigned *)smem;
        const int ntot = NP2 * SR * CP;
        for (int i = tid; i < ntot; i += NTHR) st[i] = 0u;
        __syncthreads();
        for (int i = tid; i < NPA; i += NTHR) {
            const int rr = i / IW, xx = i - rr * IW;
#if CPASYNC
            const unsigned *src = (const unsigned *)(t4 + (size_t)i * CC1 + c0);
#pragma unroll
            for (int j = 0; j < NP2; ++j)
                __pipeline_memcpy_async(&st[(j * SR + rr + 1) * CP + xx + 1], &src[j], 4);
#else
            const uint4 v = *(const uint4 *)(t4 + (size_t)i * CC1 + c0);
            const unsigned *vv = (const unsigned *)&v;
#pragma unroll
            for (int j = 0; j < NP2; ++j)
                st[(j * SR + rr + 1) * CP + xx + 1] = vv[j];
#endif
        }
#if CPASYNC
        __pipeline_commit();
        __pipeline_wait_prior(0);
#endif
        __syncthreads();
        const int j = tid / (NTHR / NP2);
        const int tk = tid % (NTHR / NP2);
        const int c = c0 + 2 * j;
        unsigned wr[9];
#pragma unroll
        for (int t = 0; t < 9; ++t)
            wr[t] = *(const unsigned *)(WB + OFF_W5 + t * CC1 + c);
        const __half2 bias = __floats2half2_rn(BB[BO5 + c], BB[BO5 + c + 1]);
        const unsigned *sp = st + j * SR * CP;
        const int ntask = (R1 - R0) * 5;
        for (int task = tk; task < ntask; task += NTHR / NP2) {
            const int ry = task / 5, gx = task - ry * 5;
            const int y = R0 + ry, x0 = gx * 4;
            const unsigned *base = sp + (y - A0) * CP + x0;
            __half2 a0 = bias, a1 = bias, a2 = bias, a3 = bias;
#pragma unroll
            for (int dy = 0; dy < 3; ++dy) {
                const unsigned *row = base + dy * CP;
                unsigned v[6];
#pragma unroll
                for (int t = 0; t < 6; ++t) v[t] = row[t];
#pragma unroll
                for (int dx = 0; dx < 3; ++dx) {
                    const __half2 w = *(const __half2 *)&wr[dy * 3 + dx];
                    a0 = __hfma2(w, *(const __half2 *)&v[dx], a0);
                    a1 = __hfma2(w, *(const __half2 *)&v[dx + 1], a1);
                    a2 = __hfma2(w, *(const __half2 *)&v[dx + 2], a2);
                    a3 = __hfma2(w, *(const __half2 *)&v[dx + 3], a3);
                }
            }
            const __half2 r0 = silu_h2(a0), r1 = silu_h2(a1), r2 = silu_h2(a2),
                          r3 = silu_h2(a3);
            const int pix = y * IW + x0;
            const size_t o0 = (size_t)c * IP + pix, o1 = (size_t)(c + 1) * IP + pix;
            const __half *xa = xn + o0, *xb = xn + o1;
            __half oa[4], ob[4];
            oa[0] = __hadd(xa[0], __low2half(r0));  ob[0] = __hadd(xb[0], __high2half(r0));
            oa[1] = __hadd(xa[1], __low2half(r1));  ob[1] = __hadd(xb[1], __high2half(r1));
            oa[2] = __hadd(xa[2], __low2half(r2));  ob[2] = __hadd(xb[2], __high2half(r2));
            oa[3] = __hadd(xa[3], __low2half(r3));  ob[3] = __hadd(xb[3], __high2half(r3));
            *(uint2 *)(on + o0) = *(const uint2 *)oa;
            *(uint2 *)(on + o1) = *(const uint2 *)ob;
        }
    }
}

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

void cib_run(torch::Tensor x, torch::Tensor out, torch::Tensor wb,
             torch::Tensor bb, torch::Tensor scr, int64_t cpi) {
    static bool once = false;
    if (!once) {
        cudaFuncSetAttribute((void *)cib_kernel,
                             cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
        cudaFuncSetAttribute((void *)cib_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_B);
        once = true;
    }
    const int n = (int)x.size(0);
    const int grid = n * (int)cpi * CSZ;
    cib_kernel<<<grid, NTHR, SMEM_B, c10::cuda::getCurrentCUDAStream()>>>(
        (const __half *)x.data_ptr(), (__half *)out.data_ptr(),
        (const __half *)wb.data_ptr(), bb.data_ptr<float>(),
        (__half *)scr.data_ptr(), (int)cpi, (long long)x.stride(0));
}
int64_t scluster(){ return S_CLUSTER; }

"""
