"""YOLOv10 C2f and C2fCIB blocks.

Fused implementation.  Each Conv-BN-SiLU of the block becomes one Triton
kernel -- an implicit GEMM for the dense convs, a direct stencil for the
depthwise ones -- with the batch-norm affine, the activation and the optional
residual folded into the epilogue.  Every kernel writes straight into the slice
of the concat buffer its output belongs to, so the ``chunk``/``cat``/residual
plumbing of the eager block costs nothing, and ``YOLORepVGGDW``'s two branches
are pre-merged into a single 7x7 depthwise weight.

All intermediates are kept channels-last (``[N, H*W, C]``).  That keeps the
innermost axis of every load contiguous *and* 16-byte aligned even for the
shifted 3x3/7x7 taps; in NCHW a tap at dx = +-1 shifts the contiguous axis by a
single fp16 element, which costs Triton its vectorized accesses.  Only the
block's own input and output stay NCHW, which for the 1x1 convs at either end
is just a different address expression rather than an extra pass.

These blocks are tiny -- single-digit microseconds of math behind ~14 eager op
dispatches, entirely launch-bound -- so the whole chain is captured once per
input shape into a CUDA graph and replayed.  The input and output pointers
change from call to call, so the graph's first and last kernel nodes have that
one pointer argument patched before each replay: ``cuFuncGetParamInfo`` gives
the node's exact parameter layout, which lets its argument list be snapshotted
once and re-submitted with a single pointer swapped.  Node updates are
documented to affect only subsequent launches, so this is safe while earlier
replays are still in flight.  If a node cannot be identified, the graph is
re-captured with two hand-written copy kernels bracketing the chain instead,
and if that fails too the chain is simply launched kernel by kernel.
"""
from __future__ import annotations

import os
import threading

import torch
import torch.nn as nn

from .yolov10_bottleneck import YOLOBottleneck
from .yolov10_cib import YOLOCIB
from .yolov10_conv import YOLOConv
from .yolov10_repvggdw import YOLORepVGGDW

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - no Triton -> eager fallback
    _HAS_TRITON = False


# ---------------------------------------------------------------------------
# CUDA-graph helper extension: boundary copies + patched replay.
# ---------------------------------------------------------------------------
_CU_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstring>
#include <vector>

// --- boundary copy kernels (fallback capture mode) -------------------------
// Two distinct entry points so each node can be found by function pointer.
__global__ void fk_copy_in(const uint4* __restrict__ src, uint4* __restrict__ dst,
                           long long n16, long long tail_src, long long tail_dst,
                           int ntail) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n16) dst[i] = src[i];
  if (i == 0 && ntail) {
    const __half* s = (const __half*)tail_src;
    __half* d = (__half*)tail_dst;
    for (int k = 0; k < ntail; ++k) d[k] = s[k];
  }
}

__global__ void fk_copy_out(const uint4* __restrict__ src, uint4* __restrict__ dst,
                            long long n16, long long tail_src, long long tail_dst,
                            int ntail) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n16) dst[i] = src[i];
  if (i == 0 && ntail) {
    const __half* s = (const __half*)tail_src;
    __half* d = (__half*)tail_dst;
    for (int k = 0; k < ntail; ++k) d[k] = s[k];
  }
}

void fk_launch_copy(int64_t which, int64_t src, int64_t dst, int64_t nelem) {
  long long n16 = nelem / 8;  // 8 fp16 per uint4
  int ntail = (int)(nelem - n16 * 8);
  long long ts = src + n16 * 16, td = dst + n16 * 16;
  int threads = 256;
  long long blocks = (n16 + threads - 1) / threads;
  if (blocks < 1) blocks = 1;
  auto stream = c10::cuda::getCurrentCUDAStream();
  if (which == 0)
    fk_copy_in<<<(unsigned)blocks, threads, 0, stream>>>(
        (const uint4*)src, (uint4*)dst, n16, ts, td, ntail);
  else
    fk_copy_out<<<(unsigned)blocks, threads, 0, stream>>>(
        (const uint4*)src, (uint4*)dst, n16, ts, td, ntail);
}

struct FkArgs {
  const void* src;
  void* dst;
  long long n16;
  long long tail_src;
  long long tail_dst;
  int ntail;
};

// --- one patchable pointer argument of a captured kernel node --------------
//
// ``cuFuncGetParamInfo`` gives the exact parameter layout of the (Triton-)
// compiled function, so the node's whole argument list can be snapshotted into
// our own storage once and re-submitted with a single pointer replaced.  The
// snapshot is required: ``cuGraphExecKernelNodeSetParams`` invalidates the
// storage the node handed back, so it can only be read before the first patch.
struct PatchTarget {
  CUgraphNode node = nullptr;
  CUDA_KERNEL_NODE_PARAMS np{};
  std::vector<char> blob;
  std::vector<void*> ptrs;
  size_t argoff = 0;
  bool ok = false;
};

static bool build_target(cudaGraph_t g, int64_t func, int64_t argidx, PatchTarget* t) {
  size_t n = 0;
  if (cudaGraphGetNodes(g, nullptr, &n) != cudaSuccess || n == 0) return false;
  std::vector<cudaGraphNode_t> all(n);
  if (cudaGraphGetNodes(g, all.data(), &n) != cudaSuccess) return false;
  CUgraphNode hit = nullptr;
  CUDA_KERNEL_NODE_PARAMS found{};
  int cnt = 0;
  for (size_t i = 0; i < n; ++i) {
    cudaGraphNodeType ty;
    if (cudaGraphNodeGetType(all[i], &ty) != cudaSuccess) { cudaGetLastError(); continue; }
    if (ty != cudaGraphNodeTypeKernel) continue;
    CUDA_KERNEL_NODE_PARAMS q;
    std::memset(&q, 0, sizeof(q));
    if (cuGraphKernelNodeGetParams((CUgraphNode)all[i], &q) != CUDA_SUCCESS) continue;
    if ((int64_t)q.func == func) { hit = (CUgraphNode)all[i]; found = q; ++cnt; }
  }
  if (hit == nullptr || cnt != 1 || found.func == nullptr || found.kernelParams == nullptr)
    return false;
  std::vector<size_t> off, sz;
  size_t total = 0;
  for (size_t i = 0;; ++i) {
    size_t o, s;
    if (cuFuncGetParamInfo(found.func, i, &o, &s) != CUDA_SUCCESS) break;
    off.push_back(o);
    sz.push_back(s);
    if (o + s > total) total = o + s;
  }
  if ((int64_t)off.size() <= argidx || sz[(size_t)argidx] != sizeof(void*) || total == 0)
    return false;
  t->node = hit;
  t->np = found;
  t->blob.assign(total, 0);
  for (size_t i = 0; i < off.size(); ++i)
    std::memcpy(t->blob.data() + off[i], found.kernelParams[i], sz[i]);
  t->ptrs.resize(off.size());
  for (size_t i = 0; i < off.size(); ++i) t->ptrs[i] = t->blob.data() + off[i];
  t->argoff = off[(size_t)argidx];
  t->ok = true;
  return true;
}

static void patch(CUgraphExec exec, PatchTarget* t, int64_t value) {
  *(int64_t*)(t->blob.data() + t->argoff) = value;
  CUDA_KERNEL_NODE_PARAMS p = t->np;
  p.kernelParams = t->ptrs.data();
  p.extra = nullptr;
  CUresult r = cuGraphExecKernelNodeSetParams(exec, t->node, &p);
  TORCH_CHECK(r == CUDA_SUCCESS, "fk: kernel node update failed (", (int)r, ")");
}

struct FkPlan {
  int mode = 0;  // 0 = boundary copies, 1 = patched Triton nodes
  cudaGraphExec_t exec = nullptr;
  cudaGraphNode_t nin = nullptr, nout = nullptr;
  cudaKernelNodeParams pin{}, pout{};
  FkArgs ain{}, aout{};
  PatchTarget tin, tout;
  std::vector<int64_t> oshape;
};

static std::vector<FkPlan*> g_plans;

int64_t fk_register_patched(int64_t graph, int64_t exec, std::vector<int64_t> oshape,
                            int64_t func_in, int64_t arg_in,
                            int64_t func_out, int64_t arg_out) {
  FkPlan* p = new FkPlan();
  if (!build_target((cudaGraph_t)graph, func_in, arg_in, &p->tin) ||
      !build_target((cudaGraph_t)graph, func_out, arg_out, &p->tout)) {
    delete p;
    return -1;
  }
  p->mode = 1;
  p->exec = (cudaGraphExec_t)exec;
  p->oshape = oshape;
  g_plans.push_back(p);
  return (int64_t)(g_plans.size() - 1);
}

int64_t fk_register_copy(int64_t graph, int64_t exec, std::vector<int64_t> oshape,
                         int64_t in_dst, int64_t in_n, int64_t out_src, int64_t out_n) {
  cudaGraph_t g = (cudaGraph_t)graph;
  size_t n = 0;
  if (cudaGraphGetNodes(g, nullptr, &n) != cudaSuccess || n == 0) return -1;
  std::vector<cudaGraphNode_t> nodes(n);
  if (cudaGraphGetNodes(g, nodes.data(), &n) != cudaSuccess) return -1;
  FkPlan* p = new FkPlan();
  int cin = 0, cout = 0;
  for (size_t i = 0; i < n; ++i) {
    cudaGraphNodeType ty;
    if (cudaGraphNodeGetType(nodes[i], &ty) != cudaSuccess) { cudaGetLastError(); continue; }
    if (ty != cudaGraphNodeTypeKernel) continue;
    cudaKernelNodeParams kp;
    std::memset(&kp, 0, sizeof(kp));
    if (cudaGraphKernelNodeGetParams(nodes[i], &kp) != cudaSuccess) { cudaGetLastError(); continue; }
    if (kp.func == (void*)fk_copy_in) { p->nin = nodes[i]; p->pin = kp; ++cin; }
    else if (kp.func == (void*)fk_copy_out) { p->nout = nodes[i]; p->pout = kp; ++cout; }
  }
  cudaGetLastError();
  if (cin != 1 || cout != 1) { delete p; return -1; }
  p->mode = 0;
  p->exec = (cudaGraphExec_t)exec;
  p->oshape = oshape;
  p->pin.kernelParams = nullptr;
  p->pin.extra = nullptr;
  p->pout.kernelParams = nullptr;
  p->pout.extra = nullptr;
  p->ain.dst = (void*)in_dst;
  p->ain.n16 = in_n / 8;
  p->ain.ntail = (int)(in_n - p->ain.n16 * 8);
  p->aout.src = (const void*)out_src;
  p->aout.n16 = out_n / 8;
  p->aout.ntail = (int)(out_n - p->aout.n16 * 8);
  p->aout.tail_src = out_src + p->aout.n16 * 16;
  g_plans.push_back(p);
  return (int64_t)(g_plans.size() - 1);
}

torch::Tensor fk_run(int64_t h, torch::Tensor x) {
  FkPlan* p = g_plans[(size_t)h];
  torch::Tensor y = at::empty(p->oshape, x.options());
  if (p->mode == 1) {
    patch((CUgraphExec)p->exec, &p->tin, (int64_t)x.data_ptr());
    patch((CUgraphExec)p->exec, &p->tout, (int64_t)y.data_ptr());
  } else {
    p->ain.src = x.data_ptr();
    p->ain.tail_src = (long long)x.data_ptr() + p->ain.n16 * 16;
    p->aout.dst = y.data_ptr();
    p->aout.tail_dst = (long long)y.data_ptr() + p->aout.n16 * 16;
    void* a1[6] = {(void*)&p->ain.src, (void*)&p->ain.dst, (void*)&p->ain.n16,
                   (void*)&p->ain.tail_src, (void*)&p->ain.tail_dst, (void*)&p->ain.ntail};
    void* a2[6] = {(void*)&p->aout.src, (void*)&p->aout.dst, (void*)&p->aout.n16,
                   (void*)&p->aout.tail_src, (void*)&p->aout.tail_dst, (void*)&p->aout.ntail};
    p->pin.func = (void*)fk_copy_in;
    p->pin.kernelParams = a1;
    p->pout.func = (void*)fk_copy_out;
    p->pout.kernelParams = a2;
    cudaError_t e = cudaGraphExecKernelNodeSetParams(p->exec, p->nin, &p->pin);
    TORCH_CHECK(e == cudaSuccess, "fk: in-node update failed (", (int)e, ")");
    e = cudaGraphExecKernelNodeSetParams(p->exec, p->nout, &p->pout);
    TORCH_CHECK(e == cudaSuccess, "fk: out-node update failed (", (int)e, ")");
  }
  cudaError_t e = cudaGraphLaunch(p->exec, c10::cuda::getCurrentCUDAStream());
  TORCH_CHECK(e == cudaSuccess, "fk: graph launch failed (", (int)e, ")");
  return y;
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
void fk_launch_copy(int64_t which, int64_t src, int64_t dst, int64_t nelem);
int64_t fk_register_patched(int64_t graph, int64_t exec, std::vector<int64_t> oshape,
                            int64_t func_in, int64_t arg_in,
                            int64_t func_out, int64_t arg_out);
int64_t fk_register_copy(int64_t graph, int64_t exec, std::vector<int64_t> oshape,
                         int64_t in_dst, int64_t in_n, int64_t out_src, int64_t out_n);
torch::Tensor fk_run(int64_t h, torch::Tensor x);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch_copy", &fk_launch_copy);
  m.def("register_patched", &fk_register_patched);
  m.def("register_copy", &fk_register_copy);
  m.def("run", &fk_run);
}
"""

_EXT = None
_EXT_TRIED = False
_LOCK = threading.Lock()


def _ext():
    global _EXT, _EXT_TRIED
    if _EXT_TRIED:
        return _EXT
    with _LOCK:
        if not _EXT_TRIED:
            try:
                from torch.utils.cpp_extension import load_inline

                _EXT = load_inline(
                    name="fk_yolo_c2f_g3",
                    cpp_sources=[_CPP_SRC],
                    cuda_sources=[_CU_SRC],
                    extra_cuda_cflags=["-O3"],
                    extra_ldflags=["-lcuda"],
                    verbose=False,
                )
            except Exception:
                _EXT = None
            _EXT_TRIED = True
    return _EXT


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------
if _HAS_TRITON:

    @triton.jit
    def _dense_conv(
        xp, wp, sp, bp, yp, rp,
        H: tl.constexpr, W: tl.constexpr,
        CIN: tl.constexpr, COUT: tl.constexpr,
        XCT: tl.constexpr, YCT: tl.constexpr, RCT: tl.constexpr,
        XCO: tl.constexpr, YCO: tl.constexpr, RCO: tl.constexpr,
        KS: tl.constexpr, ACT: tl.constexpr, HAS_RES: tl.constexpr,
        IN_NCHW: tl.constexpr, OUT_NCHW: tl.constexpr,
        BP: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr,
        UNROLL: tl.constexpr,
    ):
        """out[.., YCO+co] = act(scale * conv(x) + bias) (+ residual).

        Implicit GEMM: M = pixels (the block's tile of ``H*W``), N = output
        channels, K = CIN * KS * KS.  ``wp`` is pre-permuted to
        ``[KS*KS, CIN, COUT]`` so each tap's B-tile is output-channel
        contiguous, and ``x`` is channels-last so the innermost axis of the
        A-tile stays contiguous and 16-byte aligned even for a shifted tap.
        The block's own input/output are the exception: ``IN_NCHW`` /
        ``OUT_NCHW`` swap which axis of the tile is contiguous, which for a 1x1
        kernel is only a different address expression, not an extra pass.
        """
        HW: tl.constexpr = H * W
        NK: tl.constexpr = (CIN + BK - 1) // BK
        NIT: tl.constexpr = KS * KS * NK
        pid_p = tl.program_id(0)
        pid_c = tl.program_id(1)
        nb = tl.program_id(2)

        offp = pid_p * BP + tl.arange(0, BP)
        pm = offp < HW
        offc = pid_c * BC + tl.arange(0, BC)
        cm = offc < COUT
        ph = offp // W
        pw = offp - ph * W

        xbase = xp + nb * (XCT * HW)
        acc = tl.zeros((BP, BC), dtype=tl.float32)
        if UNROLL:
            for it in tl.static_range(NIT):
                t = it // NK
                kh = t // KS
                dh = kh - (KS // 2)
                dw = t - kh * KS - (KS // 2)
                if KS == 1:
                    vm = pm
                else:
                    vm = pm & (ph + dh >= 0) & (ph + dh < H) & (pw + dw >= 0) & (pw + dw < W)
                xo = offp + (dh * W + dw)
                ks = (it % NK) * BK + tl.arange(0, BK)
                km = ks < CIN
                if IN_NCHW:
                    xa = xbase + (XCO + ks)[None, :] * HW + xo[:, None]
                else:
                    xa = xbase + xo[:, None] * XCT + XCO + ks[None, :]
                b = tl.load(xa, mask=vm[:, None] & km[None, :], other=0.0)
                a = tl.load(wp + t * (CIN * COUT) + ks[:, None] * COUT + offc[None, :],
                            mask=km[:, None] & cm[None, :], other=0.0)
                acc = tl.dot(b, a, acc)
        else:
            for it in tl.range(0, NIT):
                t = it // NK
                kh = t // KS
                dh = kh - (KS // 2)
                dw = t - kh * KS - (KS // 2)
                if KS == 1:
                    vm = pm
                else:
                    vm = pm & (ph + dh >= 0) & (ph + dh < H) & (pw + dw >= 0) & (pw + dw < W)
                xo = offp + (dh * W + dw)
                ks = (it % NK) * BK + tl.arange(0, BK)
                km = ks < CIN
                if IN_NCHW:
                    xa = xbase + (XCO + ks)[None, :] * HW + xo[:, None]
                else:
                    xa = xbase + xo[:, None] * XCT + XCO + ks[None, :]
                b = tl.load(xa, mask=vm[:, None] & km[None, :], other=0.0)
                a = tl.load(wp + t * (CIN * COUT) + ks[:, None] * COUT + offc[None, :],
                            mask=km[:, None] & cm[None, :], other=0.0)
                acc = tl.dot(b, a, acc)

        s = tl.load(sp + offc, mask=cm, other=0.0)
        bb = tl.load(bp + offc, mask=cm, other=0.0)
        o = acc * s[None, :] + bb[None, :]
        if ACT:
            o = o * tl.sigmoid(o)
        if HAS_RES:
            r = tl.load(rp + nb * (RCT * HW) + offp[:, None] * RCT + RCO + offc[None, :],
                        mask=pm[:, None] & cm[None, :], other=0.0)
            o = o + r.to(tl.float32)
        oh = o.to(tl.float16)
        if OUT_NCHW:
            tl.store(yp + nb * (YCT * HW) + (YCO + offc)[None, :] * HW + offp[:, None],
                     oh, mask=pm[:, None] & cm[None, :])
        else:
            tl.store(yp + nb * (YCT * HW) + offp[:, None] * YCT + YCO + offc[None, :],
                     oh, mask=pm[:, None] & cm[None, :])

    @triton.jit
    def _dw_conv(
        xp, wp, bp, yp, rp,
        H: tl.constexpr, W: tl.constexpr, C: tl.constexpr,
        XCT: tl.constexpr, YCT: tl.constexpr, RCT: tl.constexpr,
        XCO: tl.constexpr, YCO: tl.constexpr, RCO: tl.constexpr,
        KS: tl.constexpr, ACT: tl.constexpr, HAS_RES: tl.constexpr,
        BP: tl.constexpr, BC: tl.constexpr,
    ):
        """Channels-last depthwise KSxKS stencil; the BN affine is already
        folded into ``wp`` (``[KS*KS, C]`` fp32) and ``bp`` (``[C]`` fp32)."""
        HW: tl.constexpr = H * W
        pid_p = tl.program_id(0)
        pid_c = tl.program_id(1)
        nb = tl.program_id(2)

        offp = pid_p * BP + tl.arange(0, BP)
        pm = offp < HW
        offc = pid_c * BC + tl.arange(0, BC)
        cm = offc < C
        ph = offp // W
        pw = offp - ph * W

        xbase = xp + nb * (XCT * HW) + XCO
        acc = tl.zeros((BP, BC), dtype=tl.float32)
        for t in tl.static_range(KS * KS):
            dh = t // KS - (KS // 2)
            dw = t % KS - (KS // 2)
            vm = pm & (ph + dh >= 0) & (ph + dh < H) & (pw + dw >= 0) & (pw + dw < W)
            wv = tl.load(wp + t * C + offc, mask=cm, other=0.0)
            v = tl.load(xbase + (offp + (dh * W + dw))[:, None] * XCT + offc[None, :],
                        mask=vm[:, None] & cm[None, :], other=0.0)
            acc += wv[None, :] * v.to(tl.float32)

        bb = tl.load(bp + offc, mask=cm, other=0.0)
        o = acc + bb[None, :]
        if ACT:
            o = o * tl.sigmoid(o)
        if HAS_RES:
            r = tl.load(rp + nb * (RCT * HW) + offp[:, None] * RCT + RCO + offc[None, :],
                        mask=pm[:, None] & cm[None, :], other=0.0)
            o = o + r.to(tl.float32)
        tl.store(yp + nb * (YCT * HW) + offp[:, None] * YCT + YCO + offc[None, :],
                 o.to(tl.float16), mask=pm[:, None] & cm[None, :])



# ---------------------------------------------------------------------------
# Weight preparation
# ---------------------------------------------------------------------------
class _Unsupported(Exception):
    pass


def _conv_affine(yc: YOLOConv):
    """(weight, scale_f32, bias_f32, has_act) for a YOLOConv, BN folded."""
    conv = yc.conv
    if tuple(conv.stride) != (1, 1) or tuple(conv.dilation) != (1, 1):
        raise _Unsupported("stride/dilation")
    k = int(conv.weight.shape[2])
    if conv.weight.shape[2] != conv.weight.shape[3] or tuple(conv.padding) != (k // 2, k // 2):
        raise _Unsupported("padding")
    if getattr(yc, "_is_fused", False) or not hasattr(yc, "bn"):
        s = torch.ones(conv.weight.shape[0], device=conv.weight.device, dtype=torch.float32)
        b = conv.bias.float() if conv.bias is not None else torch.zeros_like(s)
    else:
        bn = yc.bn
        s = bn.weight.float() * torch.rsqrt(bn.running_var.float() + bn.eps)
        b = bn.bias.float() - bn.running_mean.float() * s
        if conv.bias is not None:
            b = b + conv.bias.float() * s
    name = type(yc.act).__name__
    if name == "Identity":
        act = False
    elif name == "SiLU":
        act = True
    else:
        raise _Unsupported("activation")
    return conv.weight.data, s.contiguous(), b.contiguous(), act


def _dw_weights(yc: YOLOConv):
    """Fold BN into a depthwise YOLOConv -> ([KS*KS, C] fp32, [C] fp32, ...)."""
    conv = yc.conv
    w, s, b, act = _conv_affine(yc)
    c = w.shape[0]
    if conv.groups != c or w.shape[1] != 1:
        raise _Unsupported("not depthwise")
    ks = int(w.shape[2])
    wf = (w.float().reshape(c, ks * ks) * s[:, None]).t().contiguous()
    return wf, b.contiguous(), ks, act, c


def _repvgg_weights(mod: YOLORepVGGDW):
    """Merge the 7x7 and 3x3 depthwise branches (BN folded) into one 7x7."""
    if getattr(mod, "_is_fused", False) or not hasattr(mod, "conv1"):
        wf, bf, ks, _act, c = _dw_weights(mod.conv)
        return wf, bf, ks, True, c
    w7, s7, b7, a7 = _conv_affine(mod.conv)
    w3, s3, b3, a3 = _conv_affine(mod.conv1)
    if a7 or a3 or type(mod.act).__name__ != "SiLU":
        raise _Unsupported("repvgg activations")
    c = w7.shape[0]
    k7 = int(w7.shape[2])
    k3 = int(w3.shape[2])
    if w7.shape[1] != 1 or w3.shape[1] != 1 or k3 > k7:
        raise _Unsupported("repvgg shapes")
    m = w7.float().reshape(c, k7, k7) * s7[:, None, None]
    o = (k7 - k3) // 2
    m[:, o:o + k3, o:o + k3] += w3.float().reshape(c, k3, k3) * s3[:, None, None]
    wf = m.reshape(c, k7 * k7).t().contiguous()
    return wf, (b7 + b3).contiguous(), k7, True, c


# ---------------------------------------------------------------------------
# Tile selection.  ``_DENSE_CFG`` / ``_DW_CFG`` hold measured-best configs for
# the shapes this workload actually runs; anything else falls back to the
# heuristic, which aims for >= one block per SM with tiles that divide the
# channel extents evenly.
# ---------------------------------------------------------------------------
# (KS, H*W, CIN, COUT, N) -> (BP, BC, BK, num_warps, num_stages, unroll)
_DENSE_CFG = {
    (1, 400, 128, 256, 1): (32, 32, 128, 4, 3, 0),
    (1, 400, 128, 256, 4): (16, 64, 128, 4, 3, 0),
    (1, 400, 256, 128, 1): (16, 32, 64, 4, 3, 0),
    (1, 400, 256, 128, 4): (32, 64, 64, 4, 3, 0),
    (1, 400, 256, 256, 1): (16, 64, 64, 4, 4, 0),
    (1, 400, 384, 256, 1): (32, 32, 128, 4, 3, 0),
    (1, 400, 384, 256, 4): (32, 128, 64, 4, 3, 0),
    (1, 1600, 128, 128, 1): (32, 64, 128, 8, 4, 0),
    (1, 1600, 192, 128, 1): (32, 64, 64, 4, 3, 0),
    (1, 1600, 256, 128, 1): (32, 64, 64, 4, 4, 0),
    (1, 6400, 96, 64, 4): (32, 64, 32, 4, 3, 0),
    (1, 6400, 192, 64, 4): (32, 64, 64, 4, 2, 0),
    (1, 25600, 32, 32, 1): (32, 32, 32, 4, 4, 0),
    (1, 25600, 48, 32, 1): (32, 32, 32, 4, 3, 0),
    (3, 400, 128, 128, 1): (16, 32, 128, 4, 3, 0),
    (3, 1600, 64, 64, 1): (32, 32, 64, 4, 4, 0),
    (3, 6400, 32, 32, 4): (32, 32, 32, 4, 3, 0),
    (3, 25600, 16, 16, 1): (32, 16, 16, 4, 3, 0),
}
# (KS, H*W, C, N) -> (BP, BC, num_warps, num_stages)
_DW_CFG = {
    (3, 400, 128, 1): (16, 16, 4, 2),
    (3, 400, 128, 4): (16, 32, 4, 1),
    (7, 400, 256, 1): (16, 16, 4, 2),
    (7, 400, 256, 4): (16, 32, 4, 1),
}


def _div(v, cands):
    for t in cands:
        if v % t == 0:
            return t
    return cands[-1]


def _pick_dense(sig):
    hit = _DENSE_CFG.get(sig)
    if hit is not None:
        return hit
    ks, hw, cin, cout, nb = sig
    bc = _div(cout, (64, 32, 16))
    bk = _div(cin, (128, 64, 32, 16))
    ncb = -(-cout // bc)
    bp = 16
    for cand in (128, 64, 32, 16):
        if -(-hw // cand) * ncb * nb >= 148:
            bp = cand
            break
    nw = 8 if bp * bc >= 8192 else 4
    return bp, bc, bk, nw, 3, 0


def _pick_dw(sig):
    hit = _DW_CFG.get(sig)
    if hit is not None:
        return hit
    ks, hw, c, nb = sig
    bc = _div(c, (64, 32, 16))
    bp = 32
    for cand in (128, 64, 32):
        if -(-hw // cand) * -(-c // bc) * nb >= 148:
            bp = cand
            break
    return bp, bc, 4, 2


# ---------------------------------------------------------------------------
# Plan building
# ---------------------------------------------------------------------------
class _Step:
    """One kernel launch whose tile config can be swapped (for tuning)."""

    __slots__ = ("dense", "tensors", "kw", "sig", "hw", "nout", "nb", "cfg")

    def __init__(self, dense, tensors, kw, sig, hw, nout, nb, cfg):
        self.dense = dense
        self.tensors = tensors
        self.kw = kw
        self.sig = sig
        self.hw = hw
        self.nout = nout
        self.nb = nb
        self.cfg = cfg

    def __call__(self, cfg=None):
        c = cfg or self.cfg
        if self.dense:
            bp, bc, bk, nw, ns, un = c
            grid = (-(-self.hw // bp), -(-self.nout // bc), self.nb)
            return _dense_conv[grid](*self.tensors, BP=bp, BC=bc, BK=bk, UNROLL=un,
                                     num_warps=nw, num_stages=ns, **self.kw)
        bp, bc, nw, ns = c
        grid = (-(-self.hw // bp), -(-self.nout // bc), self.nb)
        return _dw_conv[grid](*self.tensors, BP=bp, BC=bc,
                              num_warps=nw, num_stages=ns, **self.kw)


class _Prog:
    """Collects the kernel launches of one block plus the buffers they use."""

    def __init__(self, device, n, h, w):
        self.device = device
        self.n, self.h, self.w = n, h, w
        self.keep = []
        self.steps = []
        self.xs = None  # staging copies of the block input / output (NCHW),
        self.ys = None  # only used by the non-patched capture + eager paths

    def buf(self, channels):
        t = torch.empty((self.n, self.h * self.w, channels),
                        device=self.device, dtype=torch.float16)
        self.keep.append(t)
        return t

    def dense(self, yc, xt, xct, xco, yt, yct, yco, rt=None, rct=0, rco=0,
              in_nchw=False, out_nchw=False):
        conv = yc.conv
        if conv.groups != 1:
            raise _Unsupported("grouped dense conv")
        w, s, b, act = _conv_affine(yc)
        cout, cin, ks, _ = (int(v) for v in w.shape)
        if in_nchw and ks != 1:
            raise _Unsupported("nchw input needs 1x1")
        wt = w.permute(2, 3, 1, 0).reshape(ks * ks, cin, cout).contiguous()
        hw = self.h * self.w
        has_res = rt is not None
        sig = (ks, hw, cin, cout, self.n)
        self.keep += [wt, s, b]
        kw = dict(H=self.h, W=self.w, CIN=cin, COUT=cout,
                  XCT=xct, YCT=yct, RCT=rct, XCO=xco, YCO=yco, RCO=rco,
                  KS=ks, ACT=act, HAS_RES=has_res,
                  IN_NCHW=in_nchw, OUT_NCHW=out_nchw)
        self.steps.append(_Step(True, (xt, wt, s, b, yt, rt if has_res else yt),
                                kw, sig, hw, cout, self.n, _pick_dense(sig)))

    def depthwise(self, wf, bf, c, ks, act, xt, xct, xco, yt, yct, yco,
                  rt=None, rct=0, rco=0):
        hw = self.h * self.w
        has_res = rt is not None
        sig = (ks, hw, c, self.n)
        self.keep += [wf, bf]
        kw = dict(H=self.h, W=self.w, C=c,
                  XCT=xct, YCT=yct, RCT=rct, XCO=xco, YCO=yco, RCO=rco,
                  KS=ks, ACT=act, HAS_RES=has_res)
        self.steps.append(_Step(False, (xt, wf, bf, yt, rt if has_res else yt),
                                kw, sig, hw, c, self.n, _pick_dw(sig)))


def _add_bottleneck(pg, m: YOLOBottleneck, buf, nc, in_off, out_off):
    if m.cv1.conv.groups != 1 or m.cv2.conv.groups != 1:
        raise _Unsupported("grouped bottleneck")
    mid = int(m.cv1.conv.weight.shape[0])
    t = pg.buf(mid)
    pg.dense(m.cv1, xt=buf, xct=nc, xco=in_off, yt=t, yct=mid, yco=0)
    if m.add:
        pg.dense(m.cv2, xt=t, xct=mid, xco=0, yt=buf, yct=nc, yco=out_off,
                 rt=buf, rct=nc, rco=in_off)
    else:
        pg.dense(m.cv2, xt=t, xct=mid, xco=0, yt=buf, yct=nc, yco=out_off)


def _add_cib(pg, m: YOLOCIB, buf, nc, in_off, out_off):
    seq = m.cv1
    if not isinstance(seq, nn.Sequential) or len(seq) != 5:
        raise _Unsupported("cib layout")
    s0, s1, s2, s3, s4 = seq

    w0, b0, k0, a0, c0 = _dw_weights(s0)
    t0 = pg.buf(c0)
    pg.depthwise(w0, b0, c0, k0, a0, xt=buf, xct=nc, xco=in_off,
                 yt=t0, yct=c0, yco=0)

    mid = int(s1.conv.weight.shape[0])
    t1 = pg.buf(mid)
    pg.dense(s1, xt=t0, xct=c0, xco=0, yt=t1, yct=mid, yco=0)

    if isinstance(s2, YOLORepVGGDW):
        w2, b2, k2, a2, cm_ = _repvgg_weights(s2)
    else:
        w2, b2, k2, a2, cm_ = _dw_weights(s2)
    if cm_ != mid:
        raise _Unsupported("cib middle width")
    t2 = pg.buf(mid)
    pg.depthwise(w2, b2, mid, k2, a2, xt=t1, xct=mid, xco=0,
                 yt=t2, yct=mid, yco=0)

    last = int(s3.conv.weight.shape[0])
    t3 = pg.buf(last)
    pg.dense(s3, xt=t2, xct=mid, xco=0, yt=t3, yct=last, yco=0)

    w4, b4, k4, a4, c4 = _dw_weights(s4)
    if c4 != last:
        raise _Unsupported("cib tail width")
    if m.add:
        pg.depthwise(w4, b4, c4, k4, a4, xt=t3, xct=last, xco=0,
                     yt=buf, yct=nc, yco=out_off, rt=buf, rct=nc, rco=in_off)
    else:
        pg.depthwise(w4, b4, c4, k4, a4, xt=t3, xct=last, xco=0,
                     yt=buf, yct=nc, yco=out_off)


def _build_prog(mod, x):
    n, cin, h, w = x.shape
    c = mod.c
    nblk = len(mod.m)
    if mod.cv1.conv.weight.shape[1] != cin or mod.cv1.conv.weight.shape[0] != 2 * c:
        raise _Unsupported("cv1 shape")
    nc = (2 + nblk) * c
    if mod.cv2.conv.weight.shape[1] != nc:
        raise _Unsupported("cv2 shape")
    if c % 8 or nc % 8:
        raise _Unsupported("channel alignment")
    cout = int(mod.cv2.conv.weight.shape[0])

    pg = _Prog(x.device, n, h, w)
    xs = torch.empty((n, cin, h, w), device=x.device, dtype=torch.float16)
    ys = torch.empty((n, cout, h, w), device=x.device, dtype=torch.float16)
    pg.keep += [xs, ys]
    pg.xs, pg.ys = xs, ys
    buf = pg.buf(nc)
    pg.dense(mod.cv1, xt=xs, xct=cin, xco=0, yt=buf, yct=nc, yco=0, in_nchw=True)
    for i, m in enumerate(mod.m):
        if isinstance(m, YOLOCIB):
            _add_cib(pg, m, buf, nc, (1 + i) * c, (2 + i) * c)
        elif isinstance(m, YOLOBottleneck):
            _add_bottleneck(pg, m, buf, nc, (1 + i) * c, (2 + i) * c)
        else:
            raise _Unsupported("unknown inner block")
    pg.dense(mod.cv2, xt=buf, xct=nc, xco=0, yt=ys, yct=cout, yco=0, out_nchw=True)
    return pg, (n, cout, h, w)


class _Plan:
    """A captured graph for one (module, input shape) pair."""

    # Argument positions of ``xp`` / ``yp`` in ``_dense_conv``'s signature.
    _ARG_X = 0
    _ARG_Y = 4

    def __init__(self, pg, oshape):
        self.pg = pg
        self.oshape = oshape
        self.handle = -1
        self.patched = False
        self.graph = None

    def _run_eager(self, x, y=None):
        """Launch the chain step by step, staging through the static buffers."""
        ext = _ext()
        if y is None:
            y = torch.empty(self.oshape, device=x.device, dtype=x.dtype)
        if ext is not None:
            ext.launch_copy(0, x.data_ptr(), self.pg.xs.data_ptr(), x.numel())
        else:
            self.pg.xs.copy_(x)
        for st in self.pg.steps:
            st()
        if ext is not None:
            ext.launch_copy(1, self.pg.ys.data_ptr(), y.data_ptr(), y.numel())
        else:
            y.copy_(self.pg.ys)
        return y

    def warmup(self, x):
        for _ in range(2):
            self._run_eager(x)
        torch.cuda.synchronize()

    def _capture_patched(self, x):
        """Graph of just the conv chain; the first/last node's input/output
        pointer is patched per call, so no boundary copies are needed."""
        ext = _ext()
        g = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(g):
            kfirst = self.pg.steps[0]()
            for st in self.pg.steps[1:-1]:
                st()
            klast = self.pg.steps[-1]()
        g.instantiate()
        if kfirst is None or klast is None or kfirst.function == klast.function:
            return
        h = ext.register_patched(g.raw_cuda_graph(), g.raw_cuda_graph_exec(),
                                 list(self.oshape), kfirst.function, self._ARG_X,
                                 klast.function, self._ARG_Y)
        if h >= 0:
            self.graph = g
            self.handle = int(h)
            self.patched = True

    def _capture_copy(self, x):
        ext = _ext()
        y = torch.empty(self.oshape, device=x.device, dtype=x.dtype)
        g = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(g):
            ext.launch_copy(0, x.data_ptr(), self.pg.xs.data_ptr(), x.numel())
            for st in self.pg.steps:
                st()
            ext.launch_copy(1, self.pg.ys.data_ptr(), y.data_ptr(), y.numel())
        g.instantiate()
        h = ext.register_copy(g.raw_cuda_graph(), g.raw_cuda_graph_exec(),
                              list(self.oshape), self.pg.xs.data_ptr(), x.numel(),
                              self.pg.ys.data_ptr(), y.numel())
        if h >= 0:
            self.graph = g
            self.handle = int(h)

    def capture(self, x):
        if _ext() is None or len(self.pg.steps) < 2:
            return
        for attempt in (self._capture_patched, self._capture_copy):
            try:
                attempt(x)
            except Exception:
                if os.environ.get("FK_DEBUG"):
                    raise
                self.handle = -1
            if self.handle >= 0 and self._verify(x):
                return
            self.handle = -1
            self.patched = False
            self.graph = None

    def _verify(self, x):
        """One replay against the step-by-step path (NaN-tolerant: uninitialized
        benchmark weights can legitimately produce NaNs)."""
        ref = self._run_eager(x)
        got = self.run(x)
        torch.cuda.synchronize()
        same = torch.equal(ref, got)
        if not same:
            same = bool(((ref == got) | (ref.isnan() & got.isnan())).all())
        return same

    def run(self, x):
        if self.handle >= 0:
            # A patched pointer inherits Triton's 16-byte alignment assumption.
            if self.patched and x.data_ptr() % 16:
                x = x.clone()
            return _ext().run(self.handle, x)
        return self._run_eager(x)


def _fused_forward(mod, x):
    if mod._fk_off or not _HAS_TRITON:
        return None
    if not (x.is_cuda and x.dtype == torch.float16 and x.dim() == 4
            and x.is_contiguous()):
        return None
    cache = mod._fk_plans
    if cache is None:
        cache = mod._fk_plans = {}
    key = (tuple(x.shape), x.device.index)
    plan = cache.get(key)
    if plan is None:
        try:
            pg, oshape = _build_prog(mod, x)
            plan = _Plan(pg, oshape)
            plan.warmup(x)
            plan.capture(x)
        except Exception:
            if os.environ.get("FK_DEBUG"):
                raise
            mod._fk_off = True
            return None
        cache[key] = plan
    return plan.run(x)


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------
class YOLOC2f(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = YOLOConv(c1, 2 * self.c, 1, 1)
        self.cv2 = YOLOConv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(
            YOLOBottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)
        )
        self._fk_plans = None
        self._fk_off = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = _fused_forward(self, x)
        if out is not None:
            return out
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class YOLOC2fCIB(YOLOC2f):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, lk: bool = False, g: int = 1, e: float = 0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(YOLOCIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))
