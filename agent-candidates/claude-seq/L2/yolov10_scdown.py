"""YOLOv10 SCDown (spatial channel downsampling) block.

The captured workloads are tiny -- 0.4-1.6 MB of activations and ~0.4 GFLOP --
so on a B200 the baseline's five-launch chain (``conv1x1 -> batch_norm -> silu
-> conv3x3_depthwise -> batch_norm``) is dominated almost entirely by
*per-launch* cost.  Measured on this machine, an empty kernel launched from an
extension costs ~7 us of CUDA-event time and every additional launch on the
stream adds ~4-5 us, while the block's whole memory traffic (~2.5 MB) is worth
well under a microsecond.  The baseline measures 53-78 us; the arithmetic in it
is noise.

So the block is rewritten as two launches instead of five:

* ``stage1_tile`` -- the 1x1 convolution as a WMMA tensor-core GEMM.  For a 1x1
  stride-1 convolution one image of a contiguous NCHW tensor *is* a row-major
  ``[c1, H*W]`` matrix and the weight *is* a row-major ``[c2, c1]`` matrix, so
  the convolution is literally a matrix product -- no im2col, no gather, no
  layout change -- and BatchNorm + SiLU ride along in the epilogue.  Because
  ``K`` (= ``c1``) is only a couple of hundred channels, a block stages its
  *entire* ``K`` extent of both operands in one round of independent
  ``cp.async`` copies, so it pays one memory latency rather than one per K step.
* ``stage2_item`` -- the depthwise 3x3 stride-2 convolution with the second
  BatchNorm folded in.  Each thread owns four horizontally adjacent outputs
  whose 3x3 windows overlap, so a stencil row is ``2*4+1`` values fetched as
  five aligned ``__half2`` pairs.

Two launches and not one: collapsing them further was tried both ways and both
were slower.  Keeping a whole output tile's ``z`` in shared memory inside a
single kernel needs a halo plus a wide channel slab, which makes blocks big
enough that the shared-memory footprint starves the machine of them; running
the two stages as one cooperative launch with a ``grid.sync()`` costs more in
barrier plus lost occupancy than the launch it saves.

Both BatchNorms are collapsed at first use into per-channel ``(scale, shift)``
pairs, with the scales folded into the (fp16) convolution weights -- so
eval-mode BatchNorm costs one fp32 add inside an epilogue instead of a launch
of its own.  Anything the kernels do not cover (non-fp16, odd extents, channel
counts that do not fill a tile, a build failure) falls back to the baseline
module chain.
"""

from __future__ import annotations

import hashlib
import os

import torch
import torch.nn as nn

from .yolov10_conv import YOLOConv

# Tiling constants the eligibility test has to mirror.
_BM = 64    # stage-1 output channels per block
_BN = 64    # stage-1 spatial positions per block
_DWV = 4    # stage-2 outputs per thread (along q)

_CUDA_SRC = r"""
// Fused YOLOv10 SCDown for Blackwell (sm_100).
//
//   z   = silu(bn1(conv1x1(x)))            [N, C2, H, W]
//   out = bn2(conv3x3_dw_s2_p1(z))         [N, C2, H/2, W/2]
//
// The BatchNorm scales are pre-folded into w1s / w2s, so `b1` / `b2` are the
// only per-channel terms the kernels see.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define BM 64           // stage-1 output channels per block
#define BN 64           // stage-1 spatial positions per block
#define DWT 128         // stage-2 threads per block
#define DWV 4           // stage-2 outputs per thread (along q)

__device__ __forceinline__ float silu_f(float v) {
  return v * (1.0f / (1.0f + __expf(-v)));
}

// A 16 B global->shared copy that never lands in a register: one instruction
// instead of LDG + STS, with no scoreboard dependency between the two.
__device__ __forceinline__ void cp_async16(void *dst, const void *src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
               "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src));
}

// ---------------------------------------------------------------------------
// Stage 1: z[n, c, s] = silu(sum_ci w1s[c, ci] * x[n, ci, s] + b1[c])
//
// One tile is BM channels x BN positions of one image.  Warp `w` owns m tile
// `w % NM` and the n tiles `w / NM + j * NW/NM`.
// ---------------------------------------------------------------------------
template <int C1V, int NT>
__device__ void stage1_tile(const __half *__restrict__ x,
                            const __half *__restrict__ w1s,
                            const float *__restrict__ b1,
                            __half *__restrict__ z,
                            int C1, int C2, int S, int sTile, int cTile, int n) {
  constexpr int NW = NT / 32;          // warps
  constexpr int NM = BM / 16;          // m tiles
  constexpr int NN = BN / 16;          // n tiles
  constexpr int FPW = NM * NN / NW;    // accumulator fragments per warp
  constexpr int NSTEP = NW / NM;
  constexpr int LDB = BN + 8;          // +8 halves / +4 floats keeps the WMMA
  constexpr int LDC = BN + 4;          // tile rows off one bank group, and both
                                       // stay legal `ldm` (multiples of 16 B).
  const int lda = C1 + 8;

  extern __shared__ __half smem[];
  __half *As = smem;              // [BM][lda]  weight tile
  __half *Bs = smem + BM * lda;   // [C1][LDB]  activation tile

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int mTile = warp % NM;
  const int nGrp = warp / NM;
  const int s0 = sTile * BN;
  const int m0 = cTile * BM;

  const __half *w1m = w1s + (size_t)m0 * C1;
  const __half *xn = x + (size_t)n * C1 * S + s0;

  const int vpr = C1 >> 3;
  for (int i = tid; i < BM * vpr; i += NT) {
    const int r = i / vpr;
    const int c = (i - r * vpr) << 3;
    cp_async16(&As[r * lda + c], &w1m[(size_t)r * C1 + c]);
  }
  for (int i = tid; i < C1 * (BN / 8); i += NT) {
    const int r = i >> 3;
    const int c = (i & 7) << 3;
    cp_async16(&Bs[r * LDB + c], &xn[(size_t)r * S + c]);
  }
  asm volatile("cp.async.commit_group;\n" ::);
  asm volatile("cp.async.wait_group 0;\n" ::);
  __syncthreads();

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FPW];
#pragma unroll
  for (int j = 0; j < FPW; ++j) wmma::fill_fragment(acc[j], 0.0f);
#pragma unroll 8
  for (int kk = 0; kk < C1; kk += 16) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::load_matrix_sync(a, &As[(mTile * 16) * lda + kk], lda);
#pragma unroll
    for (int j = 0; j < FPW; ++j) {
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
      wmma::load_matrix_sync(b, &Bs[kk * LDB + (nGrp + j * NSTEP) * 16], LDB);
      wmma::mma_sync(acc[j], a, b, acc[j]);
    }
  }

  // The accumulator tile reuses the (now dead) operand staging area.
  float *Cs = (float *)smem;
  __syncthreads();
#pragma unroll
  for (int j = 0; j < FPW; ++j)
    wmma::store_matrix_sync(&Cs[(mTile * 16) * LDC + (nGrp + j * NSTEP) * 16],
                            acc[j], LDC, wmma::mem_row_major);
  __syncthreads();

  __half *zb = z + ((size_t)n * C2 + m0) * S + s0;
#pragma unroll
  for (int i = tid; i < BM * BN / 8; i += NT) {
    const int r = i >> 3;
    const int c = (i & 7) << 3;
    const float sh = b1[m0 + r];
    __half h[8];
#pragma unroll
    for (int j = 0; j < 8; ++j)
      h[j] = __float2half(silu_f(Cs[r * LDC + c + j] + sh));
    *(uint4 *)&zb[(size_t)r * S + c] = *(const uint4 *)h;
  }
}

// ---------------------------------------------------------------------------
// Stage 2: out[n, c, p, q] = sum_rs w2s[c, r, s] * z[n, c, 2p-1+r, 2q-1+s] + b2[c]
//
// One call is one thread's DWV horizontally adjacent outputs, at plane `bc`
// (= n*C2 + c, passed alongside `c` so no thread has to divide by C2), output
// row `p`, output columns [DWV*qg, DWV*qg + DWV).  With stride 2 and pad 1 on
// an even extent the only out-of-range taps are the row above p == 0 and the
// column left of q == 0, so everything else is unguarded.
// ---------------------------------------------------------------------------
__device__ void stage2_item(const __half *__restrict__ z,
                            const __half *__restrict__ w2s,
                            const float *__restrict__ b2,
                            __half *__restrict__ out,
                            int H, int W, int P, int Q,
                            int bc, int c, int p, int qg) {
  float wv[9];
#pragma unroll
  for (int j = 0; j < 9; ++j) wv[j] = __half2float(__ldg(&w2s[c * 9 + j]));
  float acc[DWV];
  const float bias = __ldg(&b2[c]);
#pragma unroll
  for (int j = 0; j < DWV; ++j) acc[j] = bias;

  const __half *zrow = z + (size_t)bc * H * W + (size_t)(2 * p - 1) * W
                       + (DWV * 2 * qg - 2);
#pragma unroll
  for (int r = 0; r < 3; ++r, zrow += W) {
    float v[DWV * 2 + 2];
    if (r == 0 && p == 0) {
#pragma unroll
      for (int k = 0; k < DWV * 2 + 2; ++k) v[k] = 0.0f;
    } else {
      if (qg == 0) {
        v[0] = 0.0f;
        v[1] = 0.0f;  // the pad column left of q == 0
      } else {
        const __half2 t = *(const __half2 *)zrow;
        v[0] = __low2float(t);
        v[1] = __high2float(t);
      }
#pragma unroll
      for (int k = 2; k < DWV * 2 + 2; k += 2) {
        const __half2 t = *(const __half2 *)(zrow + k);
        v[k] = __low2float(t);
        v[k + 1] = __high2float(t);
      }
    }
#pragma unroll
    for (int j = 0; j < DWV; ++j)
#pragma unroll
      for (int s = 0; s < 3; ++s)
        acc[j] = fmaf(wv[r * 3 + s], v[1 + 2 * j + s], acc[j]);
  }

  __half o[DWV];
#pragma unroll
  for (int j = 0; j < DWV; ++j) o[j] = __float2half(acc[j]);
  *(uint2 *)&out[(size_t)bc * P * Q + p * Q + DWV * qg] = *(const uint2 *)o;
}

// ---------------------------------------------------------------------------
// Entry kernels.
// ---------------------------------------------------------------------------
template <int C1V, int NT>
__global__ void k_conv1x1_bn_silu(const __half *__restrict__ x,
                                  const __half *__restrict__ w1s,
                                  const float *__restrict__ b1,
                                  __half *__restrict__ z,
                                  int C1r, int C2, int S) {
  stage1_tile<C1V, NT>(x, w1s, b1, z, C1V ? C1V : C1r, C2, S,
                       blockIdx.x, blockIdx.y, blockIdx.z);
}

// grid = (ceil(P*QG / DWT), C2, N): the channel and image indices come straight
// from the grid, so no thread ever divides by C2.
__global__ void k_dw3x3s2_bn(const __half *__restrict__ z,
                             const __half *__restrict__ w2s,
                             const float *__restrict__ b2,
                             __half *__restrict__ out,
                             int H, int W, int P, int Q, int QG) {
  const int idx = blockIdx.x * DWT + threadIdx.x;
  if (idx >= P * QG) return;
  const int p = idx / QG;
  const int c = blockIdx.y;
  stage2_item(z, w2s, b2, out, H, W, P, Q,
              blockIdx.z * gridDim.y + c, c, p, idx - p * QG);
}

// ---------------------------------------------------------------------------
// Host entry point.
// ---------------------------------------------------------------------------
struct Args {
  const __half *x, *w1, *w2;
  const float *b1, *b2;
  __half *z, *out;
  int C1, C2, S, H, W, P, Q, QG, nS, nM, N;
};

template <int C1V, int NT>
static void dispatch(const Args &a, cudaStream_t stream) {
  // The stage-1 epilogue reuses the operand staging area for its fp32 tile.
  const size_t smem = std::max(
      (size_t)(BM * (a.C1 + 8) + a.C1 * (BN + 8)) * sizeof(__half),
      (size_t)(BM * (BN + 4)) * sizeof(float));
  dim3 g1(a.nS, a.nM, a.N);
  k_conv1x1_bn_silu<C1V, NT><<<g1, NT, smem, stream>>>(
      a.x, a.w1, a.b1, a.z, a.C1, a.C2, a.S);
  dim3 g2((a.P * a.QG + DWT - 1) / DWT, a.C2, a.N);
  k_dw3x3s2_bn<<<g2, DWT, 0, stream>>>(
      a.z, a.w2, a.b2, a.out, a.H, a.W, a.P, a.Q, a.QG);
}

template <int NT>
static void dispatch_c1(const Args &a, cudaStream_t stream) {
  switch (a.C1) {
    case 64: dispatch<64, NT>(a, stream); break;
    case 128: dispatch<128, NT>(a, stream); break;
    default: dispatch<0, NT>(a, stream);
  }
}

torch::Tensor scdown_forward(torch::Tensor x, torch::Tensor w1s, torch::Tensor b1,
                             torch::Tensor w2s, torch::Tensor b2) {
  Args a;
  a.N = (int)x.size(0);
  a.C1 = (int)x.size(1);
  a.H = (int)x.size(2);
  a.W = (int)x.size(3);
  a.C2 = (int)w1s.size(0);
  a.S = a.H * a.W;
  a.P = a.H / 2;
  a.Q = a.W / 2;
  a.QG = a.Q / DWV;
  a.nS = a.S / BN;
  a.nM = a.C2 / BM;

  auto z = torch::empty({a.N, a.C2, a.H, a.W}, x.options());
  auto out = torch::empty({a.N, a.C2, a.P, a.Q}, x.options());
  a.x = (const __half *)x.data_ptr();
  a.w1 = (const __half *)w1s.data_ptr();
  a.w2 = (const __half *)w2s.data_ptr();
  a.b1 = b1.data_ptr<float>();
  a.b2 = b2.data_ptr<float>();
  a.z = (__half *)z.data_ptr();
  a.out = (__half *)out.data_ptr();

  auto stream = at::cuda::getCurrentCUDAStream();
  // Small grids want more warps per block to cover the staging latency; once
  // the grid alone fills the machine the extra warps only add contention.
  if (a.nS * a.nM * a.N <= 400) dispatch_c1<512>(a, stream);
  else                          dispatch_c1<256>(a, stream);
  return out;
}
"""

_CPP_SRC = r"""
torch::Tensor scdown_forward(torch::Tensor x, torch::Tensor w1s, torch::Tensor b1,
                             torch::Tensor w2s, torch::Tensor b2);
"""

_EXT = None
_EXT_FAILED = False


def _arch_list() -> str:
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return ""
    suffix = "a" if major >= 9 else ""
    return f"{major}.{minor}{suffix}"


def _ext():
    """The compiled extension, or ``None`` if it cannot be built here."""
    global _EXT, _EXT_FAILED
    if _EXT is None and not _EXT_FAILED:
        try:
            from torch.utils.cpp_extension import load_inline

            arch = _arch_list()
            if arch:
                os.environ["TORCH_CUDA_ARCH_LIST"] = arch
            tag = hashlib.sha1(_CUDA_SRC.encode()).hexdigest()[:10]
            _EXT = load_inline(
                name=f"fk_yolo_scdown_{tag}",
                cpp_sources=_CPP_SRC,
                cuda_sources=_CUDA_SRC,
                functions=["scdown_forward"],
                extra_cflags=["-O3"],
                extra_cuda_cflags=[
                    "-O3",
                    "--use_fast_math",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "--expt-relaxed-constexpr",
                ],
                verbose=False,
            )
        except Exception:
            _EXT_FAILED = True
    return _EXT


def _bn_affine(bn: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Eval-mode BatchNorm collapsed to fp32 ``(scale, shift)`` vectors."""
    scale = bn.weight.float() / torch.sqrt(bn.running_var.float() + bn.eps)
    shift = bn.bias.float() - bn.running_mean.float() * scale
    return scale, shift


class YOLOSCDown(nn.Module):
    def __init__(self, c1: int, c2: int, k: int, s: int):
        super().__init__()
        self.cv1 = YOLOConv(c1, c2, 1, 1)
        self.cv2 = YOLOConv(c2, c2, k=k, s=s, g=c2, act=False)
        self._k = k
        self._s = s
        self._packed = None
        self._shape = None

    # Weights arrive after __init__ (load_state_dict, in-place re-init), so the
    # fused pack is built on first use and dropped when state is reloaded.
    def _load_from_state_dict(self, *args, **kwargs):
        self._packed = None
        self._shape = None
        return super()._load_from_state_dict(*args, **kwargs)

    def _eligible(self, x: torch.Tensor) -> bool:
        if self._k != 3 or self._s != 2:
            return False
        if x.dtype is not torch.float16 or not x.is_cuda or x.dim() != 4:
            return False
        if not x.is_contiguous():
            return False
        c1, h, w = x.shape[1], x.shape[2], x.shape[3]
        c2 = self.cv1.conv.weight.shape[0]
        if self.cv1.conv.weight.shape[1] != c1 or self.cv2.conv.groups != c2:
            return False
        if self.cv1._is_fused or self.cv2._is_fused:
            return False
        if h % 2 or w % 2 or (h * w) % _BN or (w // 2) % _DWV:
            return False
        return c1 % 16 == 0 and c2 % _BM == 0

    def _pack(self) -> tuple:
        c2, c1 = self.cv1.conv.weight.shape[:2]
        s1, b1 = _bn_affine(self.cv1.bn)
        s2, b2 = _bn_affine(self.cv2.bn)
        # Folding the BatchNorm scales into the fp16 weights costs ~2^-11
        # relative on each weight and saves a load plus a multiply per output.
        w1s = (self.cv1.conv.weight.reshape(c2, c1).float() * s1[:, None]).half()
        w2s = (self.cv2.conv.weight.reshape(c2, 9).float() * s2[:, None]).half()
        packed = (w1s.contiguous(), b1.contiguous(), w2s.contiguous(), b2.contiguous())
        self._packed = packed
        return packed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self._packed
        if p is not None and x.shape == self._shape and x.is_contiguous():
            return _EXT.scdown_forward(x, p[0], p[1], p[2], p[3])
        if _ext() is None or not self._eligible(x):
            return self.cv2(self.cv1(x))
        self._shape = x.shape
        p = self._pack()
        return _EXT.scdown_forward(x, p[0], p[1], p[2], p[3])
