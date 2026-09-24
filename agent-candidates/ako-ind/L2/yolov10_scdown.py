"""YOLOv10 SCDown (spatial channel downsampling) block.

The eager chain is ``cv1 conv1x1 -> BN -> SiLU -> cv2 dwconv(kxk, stride s) -> BN``:
five kernel launches for tensors of at most a few MB, which on B200 is entirely
launch/dispatch bound (~81 us of CPU dispatch for ~1 us of arithmetic).

This implementation

* constant-folds both BatchNorms into their convolution weight/bias the first
  time ``forward`` runs (the benchmark shares weights *after* ``__init__``, so
  the fold cannot happen in the constructor), pre-packs cv1's 1x1 filter as a
  ``[c2, c1]`` GEMM operand and cv2's depthwise filter as a tap-major
  ``[k*k, c2]`` table, and caches them as plain tensors, so ``forward`` does no
  BatchNorm math at all;
* runs the whole block as **one hand-written CUDA kernel** for the common
  ``k=3, s=2, fp16`` case (see ``_CUDA_SRC``): each CTA streams the post-SiLU
  rows it needs through shared memory, so the ``c2 x H x W`` intermediate is
  never materialized and the 1x1 mixing GEMM runs on tensor cores
  (``mma.m16n8k16``, fp32 accumulate) with SiLU evaluated exactly once per
  element.  The kernel is compiled with NVRTC on first use and launched
  through ``cuLaunchKernel`` with a pre-built argument pack;
* falls back to a two-Triton-kernel path (mixing GEMM + depthwise reduction,
  chained with a programmatic dependent launch) for every shape or dtype the
  CUDA kernel does not cover, and to folded eager convolutions below that.

Measured on B200 (see ITERATIONS.md): the fused kernel removes 4.1 us on the
two N=4 shapes and 2.1 us on ``(1, 64, 80, 80)`` relative to the two-pass
Triton path, and ties it on ``(1, 128, 40, 40)`` -- 15.4 / 13.3 / 15.5 / 13.3 us
against an 11.3 us floor, which is what *any* single kernel that reads ``x`` and
writes ``out`` measures under this benchmark's L2 flush.
"""

from __future__ import annotations

import ctypes

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from .yolov10_conv import YOLOConv

# ---------------------------------------------------------------------------
# Low-overhead launch path.
#
# At these sizes the whole block is launch-bound, and Triton's generic
# ``JITFunction.run`` costs ~13 us of CPU per launch (argument binding,
# specialization, cache key).  All of that is shape-invariant here, so the first
# call keeps the ``CompiledKernel`` and a pre-built argument list and every
# later call goes straight to the compiled launcher (~4 us).  This is the same
# trick ``torch.compile`` uses for its cached Triton kernels.  If anything about
# the internal API does not look as expected we simply keep using the generic
# path.
# ---------------------------------------------------------------------------
try:
    from triton.runtime import driver as _driver

    _get_device = _driver.active.get_current_device
    _get_stream = _driver.active.get_current_stream
except Exception:  # pragma: no cover - unexpected Triton layout
    _get_device = _get_stream = None

# Programmatic dependent launch: the depthwise pass is launched with
# CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION so it can start while
# the mixing pass drains, and `gdc_wait()` inside it re-establishes the
# dependency before the first load of y.  The attribute goes on the *consumer*
# only -- putting it on the producer would let the producer start before
# whatever precedes it in the stream, which buys nothing here.
try:
    from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

    _HAVE_PDL = True
except Exception:  # pragma: no cover - older Triton
    _HAVE_PDL = False


def _prep_launch(jit_fn, grid, tensors, cmeta, opts):
    """Compile *jit_fn* and return ``(run, args, ptr_slots)`` for direct launch.

    ``args`` matches the compiled launcher's positional signature
    ``(gx, gy, gz, stream, function, packed_metadata, launch_metadata,
    enter_hook, exit_hook, *kernel_args)``; ``ptr_slots`` are the indices in it
    that hold the tensor pointers that change from call to call.
    """
    kern = jit_fn[grid](*tensors, **cmeta, **opts)
    if _get_device is None or kern is None:
        return None
    try:
        names = list(jit_fn.arg_names)
        vals, ti = [], 0
        for nm in names:
            if nm in cmeta:
                vals.append(cmeta[nm])
            else:
                vals.append(tensors[ti].data_ptr())
                ti += 1
        if ti != len(tensors):
            return None
        g = list(grid) + [1, 1, 1]
        args = [g[0], g[1], g[2], 0, kern.function, kern.packed_metadata,
                None, None, None] + vals
        run = kern.run
        # Smoke the direct path once; if the launcher rejects it, give up.
        args[3] = _get_stream(_get_device())
        run(*args)
    except Exception:  # pragma: no cover - unexpected Triton layout
        return None
    return (run, args, [9 + i for i in range(len(tensors))])



# ---------------------------------------------------------------------------
# Minimal NVRTC / libcuda access through ctypes.  Both libraries ship with any
# CUDA-capable PyTorch install; if either is missing or refuses the source we
# simply never build a CUDA plan and the Triton path is used instead.
# ---------------------------------------------------------------------------
try:
    _LIBCUDA = ctypes.CDLL("libcuda.so.1")
    _CU_LAUNCH = _LIBCUDA.cuLaunchKernel
    _CU_LAUNCH.restype = ctypes.c_int
    _CU_LAUNCH.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + \
                          [ctypes.c_void_p] * 3
except Exception:  # pragma: no cover - no driver
    _LIBCUDA = _CU_LAUNCH = None

_LIBNVRTC = None
for _nm in ("libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so.11"):
    try:
        _LIBNVRTC = ctypes.CDLL(_nm)
        break
    except OSError:
        continue


def _nvrtc_cubin(src: str, arch: str) -> bytes:
    """Compile *src* straight to a cubin for *arch* (no PTX JIT at load)."""
    prog = ctypes.c_void_p()
    if _LIBNVRTC.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), b"scdown.cu",
                                    0, None, None):
        raise RuntimeError("nvrtcCreateProgram failed")
    opts = [f"--gpu-architecture={arch}".encode(), b"-default-device",
            b"--use_fast_math", b"-std=c++17"]
    rc = _LIBNVRTC.nvrtcCompileProgram(prog, len(opts),
                                       (ctypes.c_char_p * len(opts))(*opts))
    if rc:
        n = ctypes.c_size_t()
        _LIBNVRTC.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
        buf = ctypes.create_string_buffer(max(n.value, 1))
        _LIBNVRTC.nvrtcGetProgramLog(prog, buf)
        raise RuntimeError("nvrtc: " + buf.value.decode(errors="replace"))
    n = ctypes.c_size_t()
    if _LIBNVRTC.nvrtcGetCUBINSize(prog, ctypes.byref(n)):
        raise RuntimeError("nvrtcGetCUBINSize failed")
    cub = ctypes.create_string_buffer(n.value)
    if _LIBNVRTC.nvrtcGetCUBIN(prog, cub):
        raise RuntimeError("nvrtcGetCUBIN failed")
    _LIBNVRTC.nvrtcDestroyProgram(ctypes.byref(prog))
    return cub.raw


def _cu_load(cubin: bytes, smem: int):
    """Load *cubin*, opt the kernel into *smem* bytes of dynamic shared memory."""
    mod = ctypes.c_void_p()
    if _LIBCUDA.cuModuleLoadData(ctypes.byref(mod), cubin):
        raise RuntimeError("cuModuleLoadData failed")
    f = ctypes.c_void_p()
    if _LIBCUDA.cuModuleGetFunction(ctypes.byref(f), mod, b"scdown"):
        raise RuntimeError("cuModuleGetFunction failed")
    if smem > 48 * 1024:
        # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        if _LIBCUDA.cuFuncSetAttribute(f, 8, ctypes.c_int(smem)):
            raise RuntimeError("cuFuncSetAttribute(max dynamic smem) failed")
    # Keep the module alive for the process lifetime.
    _CU_MODS.setdefault("_keepalive", []).append(mod)
    return f, smem


# ---------------------------------------------------------------------------
# Single-launch fused CUDA kernel (k=3, s=2, fp16).
#
# Compiled with NVRTC the first time a covered shape is seen (~0.3 s, no build
# directory, no ninja) and launched straight through ``cuLaunchKernel`` with a
# pre-built argument pack, so a call costs one FFI hop and no allocation beyond
# the output tensor.  Everything that can be a compile-time constant is one, so
# the kernel carries no divisions and no runtime bounds arithmetic.
# ---------------------------------------------------------------------------
_CUDA_SRC = r"""
typedef unsigned short u16;
typedef unsigned int   u32;

#define NW  (NWM*NWN)
#define NTHREADS (NW*32)
#define MT  (TC/16)
#define MTW (MT/NWM)
#define KT  (C1/16)
#define NT  (W/8)
#define NTW ((NT+NWN-1)/NWN)
#define RY  (2*TOH+1)
#define NACC (TOH>1?2:1)
#define NTC (C2/TC)
#define NTH (HO/TOH)
#define TPC (NTHREADS/TC)
#define RUN (WO/TPC)
#define NXV (C1*(W/8))

#define SYNC() __syncthreads()

__device__ __forceinline__ u32 h2mul(u32 a, u32 b){u32 d; asm("mul.f16x2 %0,%1,%2;":"=r"(d):"r"(a),"r"(b)); return d;}
__device__ __forceinline__ u32 h2add(u32 a, u32 b){u32 d; asm("add.f16x2 %0,%1,%2;":"=r"(d):"r"(a),"r"(b)); return d;}
__device__ __forceinline__ u32 h2tanh(u32 a){u32 d; asm("tanh.approx.f16x2 %0,%1;":"=r"(d):"r"(a)); return d;}
/* cvt2h(a,b): a -> high half, b -> low half */
__device__ __forceinline__ u32 cvt2h(float a, float b){u32 d; asm("cvt.rn.f16x2.f32 %0,%1,%2;":"=r"(d):"f"(a),"f"(b)); return d;}
__device__ __forceinline__ float siluf(float z){float h=0.5f*z,t; asm("tanh.approx.f32 %0,%1;":"=f"(t):"f"(h)); return h*(1.f+t);}
__device__ __forceinline__ float h2lo(u32 v){float f; asm("cvt.f32.f16 %0, %1;":"=f"(f):"h"((u16)(v & 0xffffu))); return f;}
__device__ __forceinline__ float h2hi(u32 v){float f; asm("cvt.f32.f16 %0, %1;":"=f"(f):"h"((u16)(v >> 16)));   return f;}

extern "C" __global__ __launch_bounds__(NTHREADS) void scdown(
    const u16* __restrict__ Xg, const u16* __restrict__ W1g,
    const float* __restrict__ B1g, const u16* __restrict__ W2g,
    const float* __restrict__ B2g, u16* __restrict__ Og)
{
  extern __shared__ u16 smem[];
  u16* As = smem;                       /* [TC][C1P]      w1 block             */
  u16* Xb = As + TC*C1P;                /* [2][C1][WPX]   x row, double buffer */
  u16* Ys = Xb + 2*C1*WPX;              /* [TC][WPY]      post-SiLU row        */

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int gr = lane >> 2, tg = lane & 3;
  const int wm = warp % NWM, wn = warp / NWM;

  int t_ = blockIdx.x;
  const int ih = t_ % NTH;  t_ /= NTH;
  const int ic = t_ % NTC;  const int nimg = t_ / NTC;
  const int co0 = ic*TC, oh0 = ih*TOH;
  const long xbase = (long)nimg*C1*H*W;

  /* --- issue every cold global load of the prologue back to back --- */
#define ISSUE_X(buf, hhv)                                                      \
  { u16* dst_ = Xb + (buf)*(C1*WPX);                                          \
    const u16* src_ = Xg + xbase + (long)(hhv)*W;                             \
    _Pragma("unroll")                                                         \
    for (int i = tid; i < NXV; i += NTHREADS) {                               \
      const int r_ = i/(W/8), c_ = i - r_*(W/8);                              \
      const unsigned sa_ = (unsigned)__cvta_generic_to_shared(&dst_[r_*WPX + c_*8]); \
      asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"                \
                   :: "r"(sa_), "l"(&src_[(long)r_*H*W + c_*8]));             \
    }                                                                         \
    asm volatile("cp.async.commit_group;"); }

  ISSUE_X(0, oh0 > 0 ? 2*oh0 - 1 : 0)
  { const u16* wg = W1g + (long)co0*C1;
    #pragma unroll
    for (int i = tid; i < TC*(C1/8); i += NTHREADS) {
      const int r = i/(C1/8), c = i - r*(C1/8);
      const unsigned sa = (unsigned)__cvta_generic_to_shared(&As[r*C1P + c*8]);
      asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                   :: "r"(sa), "l"(&wg[r*C1 + c*8]));
    }
    asm volatile("cp.async.commit_group;"); }

  /* per-thread depthwise ownership: one channel, RUN consecutive output cols */
  const int dch  = tid / TPC;
  const int dow0 = (tid - dch*TPC) * RUN;
  float w2r[3][3];
  float b2r;
  float b1r[MTW][2];
  { u16 wv[9];
    #pragma unroll
    for (int q = 0; q < 9; ++q) wv[q] = W2g[q*C2 + co0 + dch];
    b2r = B2g[co0 + dch];
    #pragma unroll
    for (int m = 0; m < MTW; ++m) {
      b1r[m][0] = B1g[co0 + (wm*MTW + m)*16 + gr];
      b1r[m][1] = B1g[co0 + (wm*MTW + m)*16 + gr + 8];
    }
    #pragma unroll
    for (int q = 0; q < 9; ++q) { float f;
      asm("cvt.f32.f16 %0, %1;":"=f"(f):"h"(wv[q])); w2r[q/3][q%3] = f; }
  }

  /* zero the 2-half left pad of every Ys row (the ww = -1 depthwise tap) */
  for (int r = tid; r < TC; r += NTHREADS) *(u32*)(&Ys[r*WPY]) = 0u;

  asm volatile("cp.async.wait_group 0;");
  SYNC();

  /* --- A fragments: same for every row, so load them once --- */
  u32 af[MTW][KT][4];
  #pragma unroll
  for (int m = 0; m < MTW; ++m)
    #pragma unroll
    for (int k = 0; k < KT; ++k) {
      const u16* p = &As[((wm*MTW + m)*16 + gr)*C1P + k*16 + tg*2];
      af[m][k][0] = *(const u32*)(p);
      af[m][k][1] = *(const u32*)(p + 8*C1P);
      af[m][k][2] = *(const u32*)(p + 8);
      af[m][k][3] = *(const u32*)(p + 8*C1P + 8);
    }

  float acc[NACC][RUN];

  #pragma unroll
  for (int rr = 0; rr < RY; ++rr) {
    const int hh = 2*oh0 - 1 + rr;
    const int cur = rr & 1;

    if ((rr & 1) == 0 && (rr>>1) < TOH) {
      const int sl = (rr>>1) % NACC;
      #pragma unroll
      for (int r = 0; r < RUN; ++r) acc[sl][r] = b2r;
    }

    if (rr) asm volatile("cp.async.wait_group 0;");
    SYNC();                       /* Xb[cur] ready; Ys(rr-1) fully consumed */
    if (rr + 1 < RY) ISSUE_X(cur ^ 1, hh + 1 > 0 ? hh + 1 : 0)

    if (hh >= 0) {
      const u16* Xs = Xb + cur*(C1*WPX);
      float c[MTW][NTW][4];
      #pragma unroll
      for (int m=0;m<MTW;++m)
        #pragma unroll
        for (int t=0;t<NTW;++t)
          #pragma unroll
          for (int q=0;q<4;++q) c[m][t][q] = 0.f;

      #pragma unroll
      for (int t = 0; t < NTW; ++t) {
        const int nt = wn + t*NWN;
        if (nt < NT) {
          #pragma unroll
          for (int k = 0; k < KT; ++k) {
            u32 b0, b1;
            { const int rw = (lane & 7) + ((lane & 8) ? 8 : 0);
              const unsigned ad = (unsigned)__cvta_generic_to_shared(&Xs[(k*16 + rw)*WPX + nt*8]);
              asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                           : "=r"(b0), "=r"(b1) : "r"(ad)); }
            #pragma unroll
            for (int m = 0; m < MTW; ++m)
              asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                           "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                           : "+f"(c[m][t][0]), "+f"(c[m][t][1]), "+f"(c[m][t][2]), "+f"(c[m][t][3])
                           : "r"(af[m][k][0]), "r"(af[m][k][1]), "r"(af[m][k][2]), "r"(af[m][k][3]),
                             "r"(b0), "r"(b1));
          }
        }
      }

      #pragma unroll
      for (int t = 0; t < NTW; ++t) {
        const int nt = wn + t*NWN;
        if (nt < NT) {
          #pragma unroll
          for (int m = 0; m < MTW; ++m) {
            const int col = nt*8 + tg*2;
            const int row = (wm*MTW + m)*16 + gr;
#if SILU16
            const u32 z0 = cvt2h(c[m][t][1] + b1r[m][0], c[m][t][0] + b1r[m][0]);
            const u32 z1 = cvt2h(c[m][t][3] + b1r[m][1], c[m][t][2] + b1r[m][1]);
            const u32 h0 = h2mul(z0, 0x38003800u), h1 = h2mul(z1, 0x38003800u);
            const u32 y0 = h2mul(h0, h2add(h2tanh(h0), 0x3c003c00u));
            const u32 y1 = h2mul(h1, h2add(h2tanh(h1), 0x3c003c00u));
#else
            const u32 y0 = cvt2h(siluf(c[m][t][1] + b1r[m][0]), siluf(c[m][t][0] + b1r[m][0]));
            const u32 y1 = cvt2h(siluf(c[m][t][3] + b1r[m][1]), siluf(c[m][t][2] + b1r[m][1]));
#endif
            *(u32*)(&Ys[row*WPY + 2 + col])       = y0;
            *(u32*)(&Ys[(row + 8)*WPY + 2 + col]) = y1;
          }
        }
      }
    }
    SYNC();                       /* Ys(rr) complete */

    if (hh >= 0) {
      u32 yv[RUN+1];
      { const u32* yp = (const u32*)(&Ys[dch*WPY + 2*dow0]);
        #pragma unroll
        for (int r = 0; r <= RUN; ++r) yv[r] = yp[r]; }
      if (rr & 1) {
        const int sl = ((rr-1)>>1) % NACC;
        #pragma unroll
        for (int r = 0; r < RUN; ++r)
          acc[sl][r] += w2r[1][0]*h2hi(yv[r]) + w2r[1][1]*h2lo(yv[r+1])
                      + w2r[1][2]*h2hi(yv[r+1]);
      } else {
        if ((rr>>1) < TOH) {
          const int sl = (rr>>1) % NACC;
          #pragma unroll
          for (int r = 0; r < RUN; ++r)
            acc[sl][r] += w2r[0][0]*h2hi(yv[r]) + w2r[0][1]*h2lo(yv[r+1])
                        + w2r[0][2]*h2hi(yv[r+1]);
        }
        if ((rr>>1) >= 1) {
          const int sl = ((rr>>1)-1) % NACC;
          #pragma unroll
          for (int r = 0; r < RUN; ++r)
            acc[sl][r] += w2r[2][0]*h2hi(yv[r]) + w2r[2][1]*h2lo(yv[r+1])
                        + w2r[2][2]*h2hi(yv[r+1]);
        }
      }
    }

    if ((rr & 1) == 0 && (rr>>1) >= 1) {
      const int q = (rr>>1) - 1, sl = q % NACC;
      u16* op = Og + ((((long)nimg*C2 + co0 + dch)*HO) + oh0 + q)*WO + dow0;
      u32 qv[RUN/2];
      #pragma unroll
      for (int r = 0; r < RUN/2; ++r) qv[r] = cvt2h(acc[sl][2*r+1], acc[sl][2*r]);
      if constexpr (SW == 8) {
        #pragma unroll
        for (int c8 = 0; c8 < RUN/8; ++c8) {
          uint4 v; v.x=qv[4*c8]; v.y=qv[4*c8+1]; v.z=qv[4*c8+2]; v.w=qv[4*c8+3];
          *(uint4*)(op + 8*c8) = v;
        }
      } else if constexpr (SW == 4) {
        #pragma unroll
        for (int c4 = 0; c4 < RUN/4; ++c4) {
          uint2 v; v.x=qv[2*c4]; v.y=qv[2*c4+1];
          *(uint2*)(op + 4*c4) = v;
        }
      } else {
        #pragma unroll
        for (int c2 = 0; c2 < RUN/2; ++c2) *(u32*)(op + 2*c2) = qv[c2];
      }
    }
  }
}
"""

# Tuned per captured shape, key ``(n, c1, c2, h)`` -> ``(TC, TOH, NWM, NWN)``:
#   TC  channels per CTA          TOH output rows per CTA
#   NWM warp split over channels  NWN warp split over the width
# Chosen by a full sweep of the valid space at every captured shape; see
# ITERATIONS.md for the table.
_CU_CFG = {
    (4, 64, 128, 80): (32, 2, 1, 5),
    (1, 64, 128, 80): (64, 1, 2, 4),
    (4, 128, 256, 40): (64, 1, 4, 1),
    (1, 128, 256, 40): (64, 1, 4, 1),
}
_NSM = 148


def _xs_pad(w: int) -> int:
    """``Xs`` row stride: 16B aligned, and ``(stride/2) % 8 == 4`` so the eight
    rows one ``ldmatrix`` touches land in eight distinct four-bank groups."""
    for p in range(0, 64, 2):
        s = w + p
        if s % 8 == 0 and (s // 2) % 8 == 4:
            return s
    raise ValueError(w)


def _ys_pad(w: int) -> int:
    """``Ys`` row stride: room for the two-half left pad that carries the
    ``ww == -1`` depthwise tap, even, same bank spread as ``_xs_pad``."""
    for p in range(4, 64, 2):
        s = w + p
        if (s // 2) % 8 == 4:
            return s
    raise ValueError(w)


def _cu_smem(c1: int, tc: int, w: int) -> int:
    return (tc * (c1 + 8) + 2 * c1 * _xs_pad(w) + tc * _ys_pad(w)) * 2


def _cu_valid(c1, c2, h, w, ho, wo, tc, toh, nwm, nwn) -> bool:
    """Every divisibility the kernel's compile-time indexing assumes."""
    nthreads = nwm * nwn * 32
    if c1 % 16 or tc % 16 or c2 % tc or w % 8 or h % 2 or w % 2:
        return False
    if ho != h // 2 or wo != w // 2 or ho % toh:
        return False
    if (tc // 16) % nwm or nthreads % tc or not 32 <= nthreads <= 1024:
        return False
    tpc = nthreads // tc
    if wo % tpc:
        return False
    run = wo // tpc
    return run >= 2 and run % 2 == 0


def _cu_pick(n, c1, c2, h, w, ho, wo):
    """Config for a shape that is not in the measured table.

    The sweep's ranking was remarkably shape-independent: a 32-channel block
    over two output rows with all warps splitting the width (and each warp
    keeping both m-tiles, so one ``ldmatrix`` feeds two ``mma``) was at or
    within noise of the best config at every captured shape, and the runners-up
    were always its immediate neighbours.  So try that family first and widen
    outwards, taking the first candidate that both validates and puts at least
    one CTA on every SM.
    """
    prefer = [(32, 2, 1), (32, 1, 1), (64, 1, 4), (64, 2, 4), (16, 2, 1),
              (16, 1, 1), (128, 1, 4), (64, 1, 2), (32, 2, 2), (128, 2, 4)]
    fallback = None
    for tc, toh, nwm in prefer:
        if c2 % tc or ho % toh or _cu_smem(c1, tc, w) > 160 * 1024:
            continue
        for nwn in (5, 4, 10, 2, 8, 1):
            if not _cu_valid(c1, c2, h, w, ho, wo, tc, toh, nwm, nwn):
                continue
            ctas = n * (c2 // tc) * (ho // toh)
            if ctas >= _NSM:
                return (tc, toh, nwm, nwn)
            if fallback is None or ctas > fallback[0]:
                fallback = (ctas, (tc, toh, nwm, nwn))
    return fallback[1] if fallback else None


_CU_MODS: dict = {}


def _cu_build(key):
    """Compile (or reuse) the specialization for *key* and return its handle."""
    hit = _CU_MODS.get(key)
    if hit is not None:
        return hit
    c1, c2, h, w, ho, wo, tc, toh, nwm, nwn = key
    nthreads = nwm * nwn * 32
    tpc = nthreads // tc
    run = wo // tpc
    sw = 8 if (run % 8 == 0 and wo % 8 == 0) else (4 if (run % 4 == 0 and wo % 4 == 0) else 2)
    defs = dict(C1=c1, C2=c2, H=h, W=w, HO=ho, WO=wo, TC=tc, TOH=toh,
                NWM=nwm, NWN=nwn, WPX=_xs_pad(w), WPY=_ys_pad(w), C1P=c1 + 8,
                SILU16=0, SW=sw)
    src = "".join(f"#define {k} {v}\n" for k, v in defs.items()) + _CUDA_SRC
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}" + ("a" if major >= 9 else "")
    cubin = _nvrtc_cubin(src, arch)
    fn, smem = _cu_load(cubin, _cu_smem(c1, tc, w))
    _CU_MODS[key] = (fn, smem, nthreads)
    return _CU_MODS[key]


class _CudaLaunch:
    """Pre-built ``cuLaunchKernel`` argument pack; only x/out pointers move."""

    __slots__ = ("_fn", "_f", "_g", "_b", "_sm", "_p", "_arr", "xp", "op")

    def __init__(self, func, grid, block, smem, ptrs):
        self._fn = _CU_LAUNCH
        self._f = func
        self._g, self._b, self._sm = int(grid), int(block), int(smem)
        self.xp = ctypes.c_void_p(0)
        self.op = ctypes.c_void_p(0)
        self._p = [self.xp] + [ctypes.c_void_p(p) for p in ptrs] + [self.op]
        self._arr = (ctypes.c_void_p * len(self._p))(
            *[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in self._p])

    def __call__(self, stream):
        rc = self._fn(self._f, self._g, 1, 1, self._b, 1, 1, self._sm, stream,
                      self._arr, None)
        if rc:
            raise RuntimeError(f"cuLaunchKernel failed with {rc}")


@triton.jit
def _silu(z):
    """z * sigmoid(z) == (z/2) * (1 + tanh(z/2)); one MUFU op via tanh.approx."""
    h = z * 0.5
    t = tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;", "=f,f", [h], dtype=tl.float32, is_pure=True, pack=1
    )
    return h * (1.0 + t)


# ---------------------------------------------------------------------------
# Pass 1: y = SiLU(W1 @ x + b1), a [c2, c1] x [c1, H*W] GEMM per image.
# Both operands and the result are contiguous along the spatial axis, so every
# access is fully coalesced.
# ---------------------------------------------------------------------------
@triton.jit
def _mix_silu_kernel(
    X, W1, B1, Y,
    C1: tl.constexpr, C2: tl.constexpr, HW: tl.constexpr,
    BC: tl.constexpr, BP: tl.constexpr, NPT: tl.constexpr,
    C_EXACT: tl.constexpr, P_EXACT: tl.constexpr, PDL: tl.constexpr,
):
    pid = tl.program_id(0)
    n = tl.program_id(1)
    pt = pid % NPT
    ct = pid // NPT
    offs_c = ct * BC + tl.arange(0, BC)
    offs_p = pt * BP + tl.arange(0, BP)
    offs_k = tl.arange(0, C1)

    if C_EXACT:
        w1t = tl.load(W1 + offs_c[:, None] * C1 + offs_k[None, :])
        b1 = tl.load(B1 + offs_c)
    else:
        cm = offs_c < C2
        w1t = tl.load(W1 + offs_c[:, None] * C1 + offs_k[None, :], mask=cm[:, None],
                      other=0.0)
        b1 = tl.load(B1 + offs_c, mask=cm, other=0.0)

    xp = X + n * (C1 * HW) + offs_k[:, None] * HW + offs_p[None, :]
    if P_EXACT:
        xt = tl.load(xp)
    else:
        xt = tl.load(xp, mask=(offs_p < HW)[None, :], other=0.0)
    y = _silu(tl.dot(w1t, xt) + b1[:, None]).to(Y.dtype.element_ty)

    yp = Y + n * (C2 * HW) + offs_c[:, None] * HW + offs_p[None, :]
    if C_EXACT and P_EXACT:
        tl.store(yp, y)
    else:
        tl.store(yp, y, mask=(offs_c < C2)[:, None] & (offs_p < HW)[None, :])
    if PDL:
        gdc_launch_dependents()


# ---------------------------------------------------------------------------
# Pass 2: depthwise kxk stride-s reduction of y (+ folded cv2 bias).
# ---------------------------------------------------------------------------
@triton.jit
def _dw_kernel(
    Y, W2, B2, OUT,
    C2: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    HO: tl.constexpr, WO: tl.constexpr,
    KS: tl.constexpr, ST: tl.constexpr, PD: tl.constexpr,
    BC: tl.constexpr, BOH: tl.constexpr, BOW: tl.constexpr,
    NHT: tl.constexpr, NWT: tl.constexpr,
    C_EXACT: tl.constexpr, S_EXACT: tl.constexpr, PDL: tl.constexpr,
):
    pid = tl.program_id(0)
    n = tl.program_id(1)
    wt = pid % NWT
    rest = pid // NWT
    ht = rest % NHT
    ct = rest // NHT

    offs_c = ct * BC + tl.arange(0, BC)
    m = tl.arange(0, BOH * BOW)
    oh = ht * BOH + m // BOW
    ow = wt * BOW + m % BOW
    cm = offs_c < C2
    sm = (oh < HO) & (ow < WO)

    yn = Y + n * (C2 * H * W) + offs_c[:, None] * (H * W)
    wp = W2 + offs_c
    optr = OUT + n * (C2 * HO * WO) + offs_c[:, None] * (HO * WO) + (oh * WO + ow)[None, :]
    h0 = oh * ST - PD
    w0 = ow * ST - PD

    # All address arithmetic is done; only now re-establish the dependency on
    # the mixing pass (the overlap window is everything above this point).
    if PDL:
        gdc_wait()

    if C_EXACT:
        acc = tl.load(B2 + offs_c)[:, None] + tl.zeros([BC, BOH * BOW], dtype=tl.float32)
    else:
        acc = (tl.load(B2 + offs_c, mask=cm, other=0.0)[:, None]
               + tl.zeros([BC, BOH * BOW], dtype=tl.float32))
    for i in tl.static_range(KS):
        hh = h0 + i
        hok = (hh >= 0) & (hh < H)
        for j in tl.static_range(KS):
            ww = w0 + j
            ok = hok & (ww >= 0) & (ww < W)
            if not S_EXACT:
                ok = ok & sm
            msk = ok[None, :] if C_EXACT else (cm[:, None] & ok[None, :])
            yt = tl.load(yn + (hh * W + ww)[None, :], mask=msk, other=0.0)
            if C_EXACT:
                w2v = tl.load(wp + (i * KS + j) * C2)
            else:
                w2v = tl.load(wp + (i * KS + j) * C2, mask=cm, other=0.0)
            acc += w2v[:, None].to(tl.float32) * yt.to(tl.float32)

    val = acc.to(OUT.dtype.element_ty)
    if C_EXACT and S_EXACT:
        tl.store(optr, val)
    else:
        tl.store(optr, val, mask=cm[:, None] & sm[None, :])


# ---------------------------------------------------------------------------
# Tile tables, measured on B200 (see ITERATIONS.md).
#   key: (n, c1, c2, h)
#   mix: (BC, BP, num_warps, num_stages)
#   dw : (BC, BOH, BOW, num_warps, num_stages)
# Tuned jointly (mix x dw), not per kernel: on (4,128,256,40) the joint optimum
# is one launch-ladder step better than combining the two isolated optima.
# ---------------------------------------------------------------------------
_MIX_CFG = {
    (4, 64, 128, 80): (32, 128, 4, 1),
    (1, 64, 128, 80): (32, 64, 4, 1),
    (4, 128, 256, 40): (32, 64, 8, 3),
    (1, 128, 256, 40): (32, 64, 8, 2),
}
_DW_CFG = {
    (4, 64, 128, 80): (4, 4, 8, 1, 2),
    (1, 64, 128, 80): (4, 4, 8, 1, 2),
    (4, 128, 256, 40): (8, 4, 4, 1, 1),
    (1, 128, 256, 40): (8, 4, 4, 1, 1),
}
_NSM = 148


def _pow2_le(v: int) -> int:
    p = 1
    while p * 2 <= v:
        p *= 2
    return p


def _mix_fallback(n, c1, c2, hw):
    """Generic tiles: enough CTAs to cover the SMs, >= 16-wide dot."""
    best = None
    for bc in (16, 32, 64, 128):
        if bc > c2:
            continue
        for bp in (64, 128, 256):
            ctas = n * -(-c2 // bc) * -(-hw // bp)
            score = (min(ctas, 2 * _NSM), -abs(ctas - 2 * _NSM))
            if best is None or score > best[0]:
                best = (score, (bc, bp, 4, 2))
    return best[1]


def _dw_fallback(n, c2, ho, wo):
    """Generic depthwise tiles: >= 16 lanes, roughly two CTAs per SM."""
    boh = min(8, _pow2_le(ho))
    bow = min(8, _pow2_le(wo))
    while boh * bow < 16:
        if bow < 16:
            bow *= 2
        else:
            boh *= 2
    bc = 1
    while bc < 16 and n * -(-c2 // (2 * bc)) * -(-ho // boh) * -(-wo // bow) >= _NSM:
        bc *= 2
    return (min(bc, c2), boh, bow, 4, 1)


class YOLOSCDown(nn.Module):
    def __init__(self, c1: int, c2: int, k: int, s: int):
        super().__init__()
        self.cv1 = YOLOConv(c1, c2, 1, 1)
        self.cv2 = YOLOConv(c2, c2, k=k, s=s, g=c2, act=False)
        self._c1 = c1
        self._c2 = c2
        self._k = k
        self._s = s
        self._pd = k // 2
        self._folded = False
        self._plans: dict = {}

    # -- inference-time constant folding -----------------------------------
    @torch.no_grad()
    def _fold(self) -> None:
        def bn_fold(conv, bn, nout):
            w = conv.weight.float()
            b = (conv.bias.float() if conv.bias is not None
                 else torch.zeros(nout, device=w.device, dtype=torch.float32))
            if bn is not None:
                # (x - mean)/sqrt(var+eps) * gamma + beta  ==  x*scale + shift
                gamma = bn.weight.float() if bn.weight is not None else torch.ones_like(b)
                beta = bn.bias.float() if bn.bias is not None else torch.zeros_like(b)
                scale = gamma / torch.sqrt(bn.running_var.float() + bn.eps)
                w = w * scale[:, None, None, None]
                b = b * scale + (beta - bn.running_mean.float() * scale)
            return w, b

        cv1, cv2 = self.cv1, self.cv2
        w1, b1 = bn_fold(cv1.conv, getattr(cv1, "bn", None), self._c2)
        w2, b2 = bn_fold(cv2.conv, getattr(cv2, "bn", None), self._c2)
        dt = cv1.conv.weight.dtype
        k = self._k
        self._w1p = w1.reshape(self._c2, self._c1).contiguous().to(dt)
        self._b1p = b1.contiguous()
        # tap-major: one tap's per-channel filter is a single coalesced read.
        self._w2p = w2.reshape(self._c2, k * k).t().contiguous().to(dt)
        self._b2p = b2.contiguous()
        # BN-folded weights for the generic eager fallback.
        self._w1c = w1.reshape(self._c2, self._c1, 1, 1).contiguous().to(dt)
        self._w2c = w2.reshape(self._c2, 1, k, k).contiguous().to(dt)
        self._b1c = b1.to(dt).contiguous()
        self._b2c = b2.to(dt).contiguous()
        self._folded = True

    def _cuda_plan(self, x, n, c1, h, w, ho, wo, oshape):
        """Build the single-launch CUDA plan, or return None if not applicable."""
        c2, k, s, pd = self._c2, self._k, self._s, self._pd
        if _LIBCUDA is None or _LIBNVRTC is None:
            return None
        if (k, s, pd) != (3, 2, 1) or x.dtype is not torch.float16:
            return None
        if not x.is_cuda or c1 != self._c1 or c1 % 16 or c2 % 16:
            return None
        if h % 2 or w % 2 or w % 8 or ho != h // 2 or wo != w // 2:
            return None
        cfg = _CU_CFG.get((n, c1, c2, h)) or _cu_pick(n, c1, c2, h, w, ho, wo)
        if cfg is None or not _cu_valid(c1, c2, h, w, ho, wo, *cfg):
            return None
        tc, toh = cfg[0], cfg[1]
        try:
            fn, smem, nthreads = _cu_build((c1, c2, h, w, ho, wo) + tuple(cfg))
            launch = _CudaLaunch(
                fn, n * (c2 // tc) * (ho // toh), nthreads, smem,
                (self._w1p.data_ptr(), self._b1p.data_ptr(),
                 self._w2p.data_ptr(), self._b2p.data_ptr()))
            # Smoke it once on scratch buffers; a driver refusal must not leak
            # into the first real call.
            xd = torch.empty(tuple(x.shape), device=x.device, dtype=x.dtype)
            od = torch.empty(oshape, device=x.device, dtype=x.dtype)
            launch.xp.value = xd.data_ptr()
            launch.op.value = od.data_ptr()
            launch(_get_stream(_get_device()))
            torch.cuda.synchronize()
        except Exception:  # pragma: no cover - NVRTC/driver refusal
            return None
        return ("cu", oshape, launch)

    def _build_plan(self, key, x: torch.Tensor):
        n, c1, h, w = x.shape
        k, s, pd, c2 = self._k, self._s, self._pd, self._c2
        ho = (h + 2 * pd - k) // s + 1
        wo = (w + 2 * pd - k) // s + 1
        hw = h * w
        oshape = (n, c2, ho, wo)
        if ho > 0 and wo > 0 and x.is_contiguous():
            plan = self._cuda_plan(x, n, c1, h, w, ho, wo, oshape)
            if plan is not None:
                self._plans[key] = plan
                return plan
        usable = (
            x.is_cuda and x.is_contiguous() and c1 == self._c1 and ho > 0 and wo > 0
            and x.dtype in (torch.float16, torch.bfloat16)
            and c1 in (16, 32, 64, 128, 256, 512)
        )
        if not usable:
            plan = (None, oshape)
            self._plans[key] = plan
            return plan

        ck = (n, c1, c2, h)
        bc1, bp, nw1, ns1 = _MIX_CFG.get(ck) or _mix_fallback(n, c1, c2, hw)
        npt = -(-hw // bp)
        mix_meta = dict(C1=c1, C2=c2, HW=hw, BC=bc1, BP=bp, NPT=npt,
                        C_EXACT=(c2 % bc1 == 0), P_EXACT=(hw % bp == 0), PDL=_HAVE_PDL)
        bc2, boh, bow, nw2, ns2 = _DW_CFG.get(ck) or _dw_fallback(n, c2, ho, wo)
        nht = -(-ho // boh)
        nwt = -(-wo // bow)
        dw_meta = dict(C2=c2, H=h, W=w, HO=ho, WO=wo, KS=k, ST=s, PD=pd,
                       BC=bc2, BOH=boh, BOW=bow, NHT=nht, NWT=nwt,
                       C_EXACT=(c2 % bc2 == 0),
                       S_EXACT=(ho % boh == 0 and wo % bow == 0), PDL=_HAVE_PDL)

        y = torch.empty((n, c2, h, w), device=x.device, dtype=x.dtype)
        out = torch.empty(oshape, device=x.device, dtype=x.dtype)
        # Compile against a freshly allocated input so the kernel is always the
        # 16B-aligned specialization; ``forward`` only feeds it aligned pointers.
        xd = torch.empty(tuple(x.shape), device=x.device, dtype=x.dtype)
        mix_grid = (npt * -(-c2 // bc1), n)
        dw_grid = (nht * nwt * -(-c2 // bc2), n)
        fast_mix = _prep_launch(_mix_silu_kernel, mix_grid,
                                (xd, self._w1p, self._b1p, y), mix_meta,
                                dict(num_warps=nw1, num_stages=ns1))
        fast_dw = _prep_launch(_dw_kernel, dw_grid,
                               (y, self._w2p, self._b2p, out), dw_meta,
                               dict(num_warps=nw2, num_stages=ns2,
                                    launch_pdl=_HAVE_PDL))
        if fast_mix is None or fast_dw is None:
            plan = (False, oshape, y,
                    (mix_grid, dict(mix_meta, num_warps=nw1, num_stages=ns1)),
                    (dw_grid, dict(dw_meta, num_warps=nw2, num_stages=ns2,
                                   launch_pdl=_HAVE_PDL)))
        else:
            # arg slots that must be refreshed per call: mix's X, dw's OUT.
            plan = (True, oshape, y, fast_mix, fast_dw,
                    fast_mix[2][0], fast_dw[2][3])
        self._plans[key] = plan
        return plan

    def _eager(self, x: torch.Tensor) -> torch.Tensor:
        y = F.silu(F.conv2d(x, self._w1c, self._b1c))
        return F.conv2d(y, self._w2c, self._b2c, self._s, self._pd, 1, self._c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._folded:
            self._fold()
        plan = self._plans.get((x.shape, x.dtype))
        if plan is None:
            plan = self._build_plan((x.shape, x.dtype), x)
        mode = plan[0]
        if mode is None:
            return self._eager(x)
        # The plan is keyed on (shape, dtype) only, so re-check the two
        # properties the specialized kernels rely on but the key does not carry.
        xp = x.data_ptr()
        if xp % 16 or not x.is_contiguous():
            return self._eager(x)
        out = torch.empty(plan[1], device=x.device, dtype=x.dtype)
        if mode == "cu":
            launch = plan[2]
            launch.xp.value = xp
            launch.op.value = out.data_ptr()
            launch(_get_stream(_get_device()))
        elif mode:
            _, _, _, (run1, a1, _), (run2, a2, _), x_slot, o_slot = plan
            st = _get_stream(_get_device())
            a1[3] = st
            a1[x_slot] = xp
            run1(*a1)
            a2[3] = st
            a2[o_slot] = out.data_ptr()
            run2(*a2)
        else:
            y = plan[2]
            _mix_silu_kernel[plan[3][0]](x, self._w1p, self._b1p, y, **plan[3][1])
            _dw_kernel[plan[4][0]](y, self._w2p, self._b2p, out, **plan[4][1])
        return out
