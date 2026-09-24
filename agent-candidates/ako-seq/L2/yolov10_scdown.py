"""YOLOv10 SCDown (spatial channel downsampling) block.

The whole block -- ``SiLU(BN(conv1x1(x)))`` followed by a 3x3 stride-2 depthwise
``BN(conv)`` -- is computed by ONE hand-written CUDA kernel (see
``scdown_fused.cu``), so the ``c2 x H x W`` intermediate is never materialized
and the block costs a single launch instead of five.  Both BatchNorms are folded
into their conv weight/bias lazily on the first forward (the harness shares
weights via ``load_state_dict`` *after* ``__init__``).

The fused kernel is gated narrowly (k=3, s=2, p=1, fp16, contiguous, even H/W,
W%8==0, c1%16==0); everything else -- other dtypes, other kernel shapes, CPU
tensors, odd geometry -- runs on the folded-eager fallback, which is itself
faster than the baseline because the two BN launches are gone.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .yolov10_conv import YOLOConv

# ---------------------------------------------------------------------------
# BN folding
# ---------------------------------------------------------------------------
def _fold_conv_bn(conv: nn.Module, bn: nn.Module | None):
    """(weight, bias) in fp32 with *bn* folded into *conv*.

    w' = w * gamma/sqrt(var+eps);  b' = (b - mean)*gamma/sqrt(var+eps) + beta.
    Handles the already-fused case (bn is None) and affine=False BN.
    """
    w = conv.weight.detach().float()
    b = conv.bias.detach().float() if conv.bias is not None else None
    if bn is None:
        if b is None:
            b = torch.zeros(w.shape[0], device=w.device, dtype=w.dtype)
        return w, b

    var, mean = bn.running_var, bn.running_mean
    if var is None or mean is None:  # track_running_stats=False -> not foldable
        return None, None
    var = var.detach().float()
    mean = mean.detach().float()
    gamma = bn.weight.detach().float() if getattr(bn, "weight", None) is not None else None
    beta = bn.bias.detach().float() if getattr(bn, "bias", None) is not None else None
    scale = torch.rsqrt(var + float(bn.eps))
    if gamma is not None:
        scale = scale * gamma
    if b is None:
        b = torch.zeros_like(mean)
    b = (b - mean) * scale
    if beta is not None:
        b = b + beta
    w = w * scale.reshape(-1, *([1] * (w.dim() - 1)))
    return w, b


def _act_kind(act: nn.Module | None) -> str:
    if act is None or isinstance(act, nn.Identity):
        return "identity"
    if isinstance(act, nn.SiLU) or type(act).__name__ == "SiLU":
        return "silu"
    return "other"


def _sig(cv: YOLOConv):
    """Cheap identity/version signature of everything the fold reads."""
    conv = cv.conv
    out = [id(conv.weight), conv.weight._version, conv.weight.dtype]
    bias = conv.bias
    out += [id(bias), bias._version if bias is not None else 0]
    bn = getattr(cv, "bn", None)
    if bn is not None:
        out.append(id(bn))
        for t in (bn.weight, bn.bias, bn.running_mean, bn.running_var):
            out += [id(t), t._version if t is not None else 0]
    else:
        out.append(0)
    return tuple(out)


# ---------------------------------------------------------------------------
# NVRTC: compile the fused kernel, once per (shape, config), cached process-wide
# ---------------------------------------------------------------------------
_SRC = Path(__file__).resolve().parent / "scdown_fused.cu"
_KERNELS: dict[tuple, object] = {}
_LIBS: list = []


def _libs():
    """(libnvrtc, libcuda) with cuLaunchKernel argtypes pinned."""
    if _LIBS:
        return _LIBS[0]
    major = str(torch.version.cuda or "12").split(".")[0]
    nvrtc = None
    for name in (f"libnvrtc.so.{major}", "libnvrtc.so"):
        try:
            nvrtc = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if nvrtc is None:
        raise OSError("libnvrtc not found")
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuLaunchKernel.restype = ctypes.c_int
    cuda.cuLaunchKernel.argtypes = [
        ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    _LIBS.append((nvrtc, cuda))
    return _LIBS[0]


def _nvrtc_cubin(src: str, arch: str) -> bytes:
    nvrtc, _ = _libs()
    prog = ctypes.c_void_p()
    rc = nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), b"scdown.cu",
                                  0, None, None)
    if rc != 0:
        raise RuntimeError(f"nvrtcCreateProgram failed ({rc})")
    opts = [f"--gpu-architecture={arch}".encode(), b"-default-device",
            b"--std=c++17"]
    arr = (ctypes.c_char_p * len(opts))(*opts)
    rc = nvrtc.nvrtcCompileProgram(prog, len(opts), arr)
    if rc != 0:
        n = ctypes.c_size_t()
        nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
        log = ctypes.create_string_buffer(n.value)
        nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(f"NVRTC failed:\n{log.value.decode(errors='replace')}")
    n = ctypes.c_size_t()
    nvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(n))
    buf = ctypes.create_string_buffer(n.value)
    nvrtc.nvrtcGetCUBIN(prog, buf)
    nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
    return buf.raw


def _ystride(w: int) -> int:
    """Ys row stride (halves): >= W + 8, multiple of 8, (stride/8) odd so the
    8 rows an ldmatrix / half2 store touches land in 8 distinct bank groups."""
    s = ((w + 8 + 7) // 8) * 8
    if (s // 8) % 2 == 0:
        s += 8
    return s


def _xstride(w: int) -> int:
    """Xs / Ws row stride (halves): multiple of 8 with (stride/8) odd, so the 8
    rows an ldmatrix touches land in 8 distinct 4-bank groups."""
    s = w
    if (s // 8) % 2 == 0:
        s += 8
    return s


_ENV_CFG = os.environ.get("SCDOWN_CFG", "")


_XDEPTH = 3          # x row buffers; must match XDEPTH in scdown_fused.cu
_SMEM_CAP = 200 * 1024


def _smem_bytes(c1: int, w: int, tc: int) -> int:
    return (_XDEPTH * c1 * _xstride(w) * 2 + 2 * tc * _ystride(w) * 2
            + tc * _xstride(c1) * 2 + tc * 9 * 4)


def _cfg_valid(c1: int, c2: int, h: int, w: int, cfg) -> bool:
    tc, toh, nwm, nwn = cfg
    oh, ow = h // 2, w // 2
    nthread = 32 * nwm * nwn
    return (tc <= c2 and c2 % tc == 0 and tc % (16 * nwm) == 0
            and 0 < toh <= oh and oh % toh == 0
            and 0 < nthread <= 1024 and nthread % tc == 0
            and ow >= nthread // tc and ow % (nthread // tc) == 0
            and _smem_bytes(c1, w, tc) <= _SMEM_CAP)


def _pick_cfg(c1: int, c2: int, h: int, w: int, n: int):
    """(TC, TOH, NWM, NWN) for a shape, or None if nothing is realizable.

    Measured on B200 (148 SMs): the kernel is latency-bound, not
    bandwidth-bound, so the choice is about how much of the machine the grid
    covers.  TOH=2 amortizes the 3-row halo over two output rows and is the
    winner whenever the grid still supplies ~2 CTAs/SM; below that the extra
    CTAs from TOH=1 matter more than the halo.  NWN ~ min(W/8, 5) puts one or
    two n-tiles on each warp; TC=32 keeps the resident A fragments small.
    """
    if _ENV_CFG:
        cfg = tuple(int(v) for v in _ENV_CFG.split(","))
        return cfg if _cfg_valid(c1, c2, h, w, cfg) else None
    oh = h // 2
    ntt = max(1, w // 8)
    tohs = (2, 1) if (oh // 2) * max(1, c2 // 32) * n >= 296 else (1, 2)
    for toh in tohs:
        for tc in (32, 16, 64, c2):
            for nwm in (1, 2, 4):
                for nwn in (min(ntt, 5), 5, 4, 2, min(ntt, 10), 8, 1):
                    cfg = (tc, toh, nwm, nwn)
                    if _cfg_valid(c1, c2, h, w, cfg):
                        return cfg
    return None


def _compile(c1: int, c2: int, h: int, w: int, cfg) -> object | None:
    """Compile + load the fused kernel; None if the config is not realizable."""
    key = (c1, c2, h, w, cfg)
    if key in _KERNELS:
        return _KERNELS[key]
    tc, toh, nwm, nwn = cfg
    oh, ow = h // 2, w // 2
    nthread = 32 * nwm * nwn
    xs, ys, ws = _xstride(w), _ystride(w), _xstride(c1)
    smem = _smem_bytes(c1, w, tc)
    ok = c1 % 16 == 0 and w % 8 == 0 and _cfg_valid(c1, c2, h, w, cfg)
    if not ok:
        _KERNELS[key] = None
        return None
    defs = dict(C1=c1, C2=c2, H=h, W=w, OH=oh, OW=ow, TC=tc, TOH=toh,
                NWM=nwm, NWN=nwn, XSTRIDE=xs, YSTRIDE=ys, WSTRIDE=ws)
    header = "".join(f"#define {k} {v}\n" for k, v in defs.items())
    src = header + _SRC.read_text()
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch = f"sm_{p.major}{p.minor}a" if p.major >= 9 else f"sm_{p.major}{p.minor}"
    try:
        _, cuda = _libs()
        cubin = _nvrtc_cubin(src, arch)
        mod = ctypes.c_void_p()
        if cuda.cuModuleLoadData(ctypes.byref(mod), cubin) != 0:
            raise RuntimeError("cuModuleLoadData failed")
        fn = ctypes.c_void_p()
        if cuda.cuModuleGetFunction(ctypes.byref(fn), mod, b"scdown_fused") != 0:
            raise RuntimeError("cuModuleGetFunction failed")
    except Exception:
        _KERNELS[key] = None
        return None
    if smem > 48 * 1024:   # opt in to the >48KB dynamic shared window
        if cuda.cuFuncSetAttribute(fn, 8, smem) != 0:  # MAX_DYNAMIC_SHARED_SIZE
            _KERNELS[key] = None
            return None
    ker = _Kernel(fn, mod, nthread, oh // toh, c2 // tc, smem)
    _KERNELS[key] = ker
    return ker


class _Kernel:
    """A compiled kernel, shared process-wide by every module with this shape.

    Deliberately holds no launch arguments: the argument cells live on the
    per-module ``_CudaState`` instead, because two YOLOSCDown instances with the
    same shape share this object and would otherwise overwrite each other's
    weight pointers.
    """

    __slots__ = ("fn", "mod", "nthread", "gx", "gy", "smem", "launch")

    def __init__(self, fn, mod, nthread, gx, gy, smem):
        self.fn = fn
        self.mod = mod
        self.nthread = nthread
        self.gx = gx
        self.gy = gy
        self.smem = smem
        self.launch = _libs()[1].cuLaunchKernel


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------
class _Plan:
    __slots__ = ("sig", "w1", "b1", "w2", "b2", "dtype", "device", "ok",
                 "cuda", "c1", "c2", "k", "s")

    def __init__(self, sig, w1, b1, w2, b2, dtype, device):
        self.sig = sig
        self.dtype = dtype
        self.device = device
        self.ok = w1 is not None and w2 is not None
        self.cuda = None
        if self.ok:
            self.w1 = w1.to(device=device, dtype=dtype).contiguous()
            self.b1 = b1.to(device=device, dtype=dtype).contiguous()
            self.w2 = w2.to(device=device, dtype=dtype).contiguous()
            self.b2 = b2.to(device=device, dtype=dtype).contiguous()
        else:
            self.w1 = self.b1 = self.w2 = self.b2 = None


class _CudaState:
    """Everything the fused path needs, resolved once per (module, shape)."""
    __slots__ = ("ker", "w1h", "b1f", "w2f", "b2f", "oh", "ow", "c2", "cells", "argv")


class YOLOSCDown(nn.Module):
    def __init__(self, c1: int, c2: int, k: int, s: int):
        super().__init__()
        self.cv1 = YOLOConv(c1, c2, 1, 1)
        self.cv2 = YOLOConv(c2, c2, k=k, s=s, g=c2, act=False)
        self._c1 = c1
        self._c2 = c2
        self._k = k
        self._s = s
        self._plan: _Plan | None = None
        self._cu: dict[tuple, object] = {}

    # -- plan management ---------------------------------------------------
    def _build_plan(self, dtype, device, sig) -> _Plan:
        w1, b1 = _fold_conv_bn(self.cv1.conv, getattr(self.cv1, "bn", None))
        w2, b2 = _fold_conv_bn(self.cv2.conv, getattr(self.cv2, "bn", None))
        plan = _Plan(sig, w1, b1, w2, b2, dtype, device)
        self._plan = plan
        self._cu.clear()
        return plan

    def _get_plan(self, x: torch.Tensor) -> _Plan:
        sig = (_sig(self.cv1), _sig(self.cv2))
        plan = self._plan
        if (plan is not None and plan.sig == sig and plan.dtype == x.dtype
                and plan.device == x.device):
            return plan
        return self._build_plan(x.dtype, x.device, sig)

    # -- fused CUDA path ---------------------------------------------------
    def _cuda_state(self, plan: _Plan, x: torch.Tensor):
        """Resolve (and cache) the fused-kernel state for this input shape."""
        n, c1, h, w = x.shape
        key = (c1, h, w, n)
        st = self._cu.get(key)
        if st is not None:
            return st
        self._cu[key] = False  # negative cache
        conv2 = self.cv2.conv
        c2 = plan.w1.shape[0]
        if not (x.dtype == torch.float16 and x.is_cuda
                and self._k == 3 and self._s == 2
                and tuple(conv2.stride) == (2, 2)
                and tuple(conv2.padding) == (1, 1)
                and tuple(conv2.dilation) == (1, 1)
                and conv2.groups == c2
                and plan.w2.shape[-2:] == (3, 3)
                and c1 == plan.w1.shape[1]
                and h % 2 == 0 and w % 2 == 0 and w % 8 == 0 and c1 % 16 == 0
                and _act_kind(self.cv1.act) == "silu"
                and _act_kind(self.cv2.act) == "identity"):
            return False
        cfg = _pick_cfg(c1, c2, h, w, n)
        if cfg is None:
            return False
        ker = _compile(c1, c2, h, w, cfg)
        if ker is None:
            return False
        st = _CudaState()
        st.ker = ker
        st.w1h = plan.w1.reshape(c2, c1).contiguous()
        st.b1f = plan.b1.float().contiguous()
        st.w2f = plan.w2.reshape(c2, 9).float().contiguous()
        st.b2f = plan.b2.float().contiguous()
        st.oh, st.ow, st.c2 = h // 2, w // 2, c2
        st.cells = [ctypes.c_void_p(0) for _ in range(6)]
        st.argv = (ctypes.c_void_p * 6)(
            *[ctypes.cast(ctypes.byref(c), ctypes.c_void_p) for c in st.cells])
        st.cells[1].value = st.w1h.data_ptr()
        st.cells[2].value = st.b1f.data_ptr()
        st.cells[3].value = st.w2f.data_ptr()
        st.cells[4].value = st.b2f.data_ptr()
        self._cu[key] = st
        return st

    # -- forward -----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        plan = self._get_plan(x)
        if not plan.ok:  # unfoldable BN -> exact baseline semantics
            return self.cv2(self.cv1(x))

        if x.dtype == torch.float16 and x.is_cuda and x.is_contiguous():
            st = self._cuda_state(plan, x)
            if st is not False and x.data_ptr() % 16 == 0:
                ker = st.ker
                n = x.shape[0]
                out = torch.empty((n, st.c2, st.oh, st.ow), dtype=torch.float16,
                                  device=x.device)
                st.cells[0].value = x.data_ptr()
                st.cells[5].value = out.data_ptr()
                ker.launch(ker.fn, ker.gx, ker.gy, n,
                           ker.nthread, 1, 1, ker.smem,
                           torch.cuda.current_stream().cuda_stream, st.argv, None)
                return out

        conv2 = self.cv2.conv
        y = F.conv2d(x, plan.w1, plan.b1)
        kind = _act_kind(self.cv1.act)
        if kind == "silu":
            y = F.silu(y, inplace=True)
        elif kind == "other":
            y = self.cv1.act(y)
        out = F.conv2d(y, plan.w2, plan.b2, stride=conv2.stride,
                       padding=conv2.padding, dilation=conv2.dilation,
                       groups=conv2.groups)
        kind2 = _act_kind(self.cv2.act)
        if kind2 == "silu":
            out = F.silu(out, inplace=True)
        elif kind2 == "other":
            out = self.cv2.act(out)
        return out
