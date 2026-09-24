"""FLUX feed-forward network (L2 composite).

Two-layer MLP: ColumnParallelLinear + GELU(tanh) -> RowParallelLinear.

Optimization
------------
Round 1 established that the only recoverable cost in this operator is the
standalone activation pass over the [M, 12288] intermediate: all three captured
shapes are compute-bound, and a Triton tanh-GELU already runs at 95-100% of the
pure-copy bandwidth floor, so the pass can only be removed by *fusing* it into
the epilogue of GEMM1.  Round 1 could not do that because its Triton GEMM only
reached 1354 TF/s against cuBLAS's 1705 TF/s -- ``tl.dot`` cannot emit the 2-SM
(``cta_group::2``) tcgen05 UMMA that cuBLAS uses.

Both GEMMs are now written in CUDA C++ on top of CUTLASS 4.x's Sm100 collective
builders (``solution/ako_ffn.cuh``).  A 256-wide MMA tile with an even cluster-M
selects the 2-SM UMMA atom, confirmed in the generated PTX:
``tcgen05.mma.cta_group::2.kind::f16`` plus the matching
``cp.async.bulk.tensor.3d.cta_group::2`` TMA load.

* **GEMM1** (K=3072, N=12288) reaches cuBLAS parity or better (1766 TF/s vs
  1756 TF/s at M=4096; 1.04-1.06x *over* cuBLAS at M=512/1024), which is what
  finally makes the fused bias + tanh-GELU epilogue free -- measured at +0.1us
  on a 198us kernel at M=4096, versus ~38us for a separate activation pass.
* **GEMM2** (K=12288, N=3072) only has 3072 output columns, so a data-parallel
  decomposition leaves a large partial wave (2.6 waves with a 128x256 CTA tile).
  A Stream-K decomposition recovers ~21us at M=4096.  At M<768 cuBLAS is
  genuinely faster here (46.1us vs 49.2us at M=512), so GEMM2 keeps cuBLAS
  there -- see ``_pick_g2``.

The extension is built once on first use (~35s cold, then cached by content
hash).  If the build is unavailable for any reason the module falls back to
round 1's path (cuBLAS GEMMs plus an in-place Triton tanh-GELU), which in turn
falls back to the unmodified baseline module tree; correctness never depends on
the fast path.
"""

from __future__ import annotations

import hashlib
import os
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl

from ..L1.gelu import GELU
from .parallel_linear import ColumnParallelLinear, RowParallelLinear


__targets__ = ["FeedForward"]


# ---------------------------------------------------------------------------
# Round 1's streaming tanh-GELU: kept as the fallback activation and used for
# any shape the fused GEMM does not cover.
# ---------------------------------------------------------------------------
_BLOCK = 2048
_WARPS = 4


@triton.jit
def _gelu_tanh_inplace(x_ptr, n, BLOCK: tl.constexpr, EXACT: tl.constexpr):
    """x <- 0.5*x*(1+tanh(c*(x+0.044715 x^3))), fp32 math, stored in place.

    Uses the identity 0.5*(1+tanh(u)) == sigmoid(2u), so this costs one exp
    rather than tanh's two.
    """
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if EXACT:
        x = tl.load(x_ptr + off, eviction_policy="evict_first").to(tl.float32)
        u = 1.5957691216057308 * (x + 0.044715 * x * x * x)
        tl.store(x_ptr + off, (x * tl.sigmoid(u)).to(x_ptr.dtype.element_ty))
    else:
        m = off < n
        x = tl.load(x_ptr + off, mask=m, eviction_policy="evict_first").to(tl.float32)
        u = 1.5957691216057308 * (x + 0.044715 * x * x * x)
        tl.store(x_ptr + off, (x * tl.sigmoid(u)).to(x_ptr.dtype.element_ty), mask=m)


def _gelu_tanh_(x: torch.Tensor) -> torch.Tensor:
    """In-place tanh-approximate GELU on a contiguous CUDA tensor."""
    n = x.numel()
    if n == 0:
        return x
    _gelu_tanh_inplace[(triton.cdiv(n, _BLOCK),)](
        x, n, BLOCK=_BLOCK, EXACT=(n % _BLOCK == 0), num_warps=_WARPS,
    )
    return x


# ---------------------------------------------------------------------------
# CUTLASS Sm100 fused GEMM1 (bias + tanh-GELU epilogue).
# ---------------------------------------------------------------------------
_SRC = ("ako_bind.cpp", "ako_g1a.cu", "ako_g1b.cu", "ako_g2a.cu", "ako_g2b.cu")
_HDR = ("ako_ffn.cuh", "ako_knobs.h")

# cfg ids must match the table in ako_bind.cpp.  Each entry is
# (cfg_id, swizzle, raster, splits, decomp); see _pick_g1 / _pick_g2.
_G1_192 = (0, 0, 0, 1, 0)   # MMA 256x192x64 c2x1, CLC
_G1_256 = (1, 0, 0, 1, 0)   # MMA 256x256x64 c2x2, CLC
_G2_192 = (2, 0, 0, 1, 0)   # MMA 256x192x64 c2x1, CLC
_G2_256_SK = (3, 2, 1, 1, 2)  # MMA 256x256x64 c2x1, Stream-K decomposition

_ext = None
_ext_tried = False


def _cutlass_includes() -> list[str] | None:
    """Locate the CUTLASS 4.x C++ headers (core + tools/util) on this machine.

    ``cutlass/util/packed_stride.hpp`` lives under ``tools/util/include``, which
    is a separate root from the core ``include``; both are required.
    """
    import importlib.util

    roots: list[pathlib.Path] = []
    spec = importlib.util.find_spec("cutlass_library")
    if spec is not None and spec.submodule_search_locations:
        roots.append(pathlib.Path(list(spec.submodule_search_locations)[0]) / "source")
    for env in ("CUTLASS_PATH", "CUTLASS_DIR"):
        v = os.environ.get(env)
        if v:
            roots.append(pathlib.Path(v))
    for root in roots:
        core, util = root / "include", root / "tools" / "util" / "include"
        if ((core / "cutlass" / "cutlass.h").is_file()
                and (util / "cutlass" / "util" / "packed_stride.hpp").is_file()):
            return [str(core), str(util)]
    return None


def _load_ext():
    """Compile + load the CUTLASS extension.  Returns None if unavailable."""
    global _ext, _ext_tried
    if _ext_tried:
        return _ext
    _ext_tried = True
    if os.environ.get("AKO_NO_CUTLASS"):
        return None
    try:
        from torch.utils.cpp_extension import load

        here = pathlib.Path(__file__).resolve().parent
        srcs = [here / s for s in _SRC]
        if not all(p.is_file() for p in srcs):
            raise FileNotFoundError(f"missing CUDA sources next to {here}")
        incs = _cutlass_includes()
        if incs is None:
            raise FileNotFoundError("CUTLASS 4.x headers not found")
        h = hashlib.sha1(
            b"".join((here / f).read_bytes() for f in (*_SRC, *_HDR))
        ).hexdigest()[:12]
        # sm_100a (not sm_100) -- CUTLASS gates the tcgen05 MMA atoms on
        # __CUDA_ARCH_FEAT_SM100_ALL, which only the 'a' target defines.
        os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0a"
        _ext = load(
            name=f"ako_flux_ffn_{h}",
            sources=[str(p) for p in srcs],
            extra_include_paths=incs,
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=[
                "-O3", "-std=c++17",
                *(f"-I{i}" for i in incs),
                "--expt-relaxed-constexpr", "--expt-extended-lambda",
                "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                "--use_fast_math",
            ],
            verbose=bool(os.environ.get("AKO_VERBOSE_BUILD")),
        )
    except Exception as exc:  # noqa: BLE001 - never let a build issue fail the op
        _ext = None
        # Loud on purpose: a silent fallback here looks identical to a working
        # fast path (this operator runs with ~zero weights, so the harness prints
        # ERR 0.00e+00 either way).
        import sys
        print(f"[ako] CUTLASS fused GEMM1 unavailable, using fallback: {exc!r}",
              file=sys.stderr, flush=True)
        if os.environ.get("AKO_REQUIRE_CUTLASS"):
            raise
    return _ext


_EMPTY_WS: dict[torch.device, torch.Tensor] = {}
_WS: dict[tuple, torch.Tensor] = {}

# Liveness counters.  This operator runs with near-zero weights, so a dead fast
# path still reports ERR 0.00e+00; these let a probe script assert which path ran.
STATS = {"fused": 0, "fused_g2": 0, "fallback": 0}


def _workspace(ext, plan: tuple, M: int, N: int, K: int, device) -> torch.Tensor:
    key = (plan, M, N, K, device)
    ws = _WS.get(key)
    if ws is None:
        nb = ext.ws_bytes(plan[0], M, N, K, plan[1], plan[2], plan[3], plan[4])
        if nb:
            ws = torch.empty(nb, dtype=torch.uint8, device=device)
        else:
            ws = _EMPTY_WS.get(device)
            if ws is None:
                ws = torch.empty(0, dtype=torch.uint8, device=device)
                _EMPTY_WS[device] = ws
        _WS[key] = ws
    return ws


def _pick_g1(M: int) -> tuple:
    """GEMM1 (K=3072, N=12288), fused bias + tanh-GELU.

    Interleaved paired measurements on B200 (148 SMs), fused kernel vs the two
    candidate tiles:
      M=512  -> 256x192 c2x1  42.0us  (256x256 c2x2: 48.2us)
      M=1024 -> 256x256 c2x2  62.4us  (256x192 c2x1: 62.5us -- near tie)
      M=4096 -> 256x192 c2x1 197.7us  (256x256 c2x2: 218.0us)
    """
    return _G1_256 if 512 < M < 2048 else _G1_192


def _pick_g2(M: int) -> tuple | None:
    """GEMM2 (K=12288, N=3072), bias only.  None means "keep cuBLAS".

    cuBLAS is genuinely better here at small M, so the fast path is only taken
    where it measured faster:
      M=512  -> cuBLAS 46.1us; best own kernel 49.2us (swept tiles down to
                64x128, 1-SM schedules, and Split-K 2..8 -- none catch it)
      M=1024 -> 256x192 c2x1 CLC     58.4us vs cuBLAS 62.4us
      M=4096 -> 256x256 c2x1 StreamK 215.0us vs cuBLAS 236.5us
    """
    if M < 768:
        return None
    return _G2_192 if M < 2048 else _G2_256_SK


class ColumnParallelApproxGELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, *, approximate: str, bias: bool = True,
                 quant_config: dict | None = None):
        super().__init__()
        self.proj = ColumnParallelLinear(dim_in, dim_out, bias=bias, quant_config=quant_config)
        self.gelu = GELU(approximate=approximate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.gelu(x)


class FeedForward(nn.Module):
    """FLUX FFN: GELU(tanh) linear -> linear with TP sharding."""

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        inner_dim: int | None = None,
        bias: bool = True,
        quant_config: dict | None = None,
    ) -> None:
        super().__init__()
        inner_dim = inner_dim or int(dim * mult)
        dim_out = dim_out or dim

        layers: list[nn.Module] = [
            ColumnParallelApproxGELU(dim, inner_dim, approximate="tanh", bias=bias,
                                      quant_config=quant_config),
            nn.Identity(),
            RowParallelLinear(inner_dim, dim_out, bias=bias, quant_config=quant_config),
        ]
        self.net = nn.ModuleList(layers)

    def _can_fuse(self) -> bool:
        act, out = self.net[0], self.net[2]
        # The activation runs in place, which autograd forbids on a tensor that
        # needs grad -- restrict the fast path to inference.
        return (not torch.is_grad_enabled()
                and not act.proj.use_fp8 and not out.use_fp8
                and act.gelu.approximate == "tanh"
                and out.tp_size == 1
                and act.proj.weight.is_cuda
                and act.proj.weight.dtype in (torch.bfloat16, torch.float16,
                                              torch.float32))

    @staticmethod
    def _gemm_ok(a: torch.Tensor, w: torch.Tensor, b) -> bool:
        """Can the CUTLASS TN bf16 GEMM take (a [M,K], w [N,K], bias [N])?"""
        if b is None or a.dtype is not torch.bfloat16:
            return False
        if w.dtype is not torch.bfloat16 or b.dtype is not torch.bfloat16:
            return False
        if not (a.is_contiguous() and w.is_contiguous() and b.is_contiguous()):
            return False
        N, K = w.shape
        # TMA descriptors need 16-byte alignment on the contiguous extent.
        return a.shape[-1] == K and b.shape == (N,) and K % 8 == 0 and N % 8 == 0

    def _cutlass_gemm(self, ext, plan, a: torch.Tensor, w: torch.Tensor,
                      b: torch.Tensor) -> torch.Tensor:
        M, K = a.shape
        N = w.shape[0]
        d = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)
        ws = _workspace(ext, plan, M, N, K, a.device)
        ext.gemm(plan[0], a, w, b, d, ws, plan[1], plan[2], plan[3], plan[4])
        return d

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._can_fuse():
            act, out = self.net[0], self.net[2]
            w1, b1, w2, b2 = act.proj.weight, act.proj.bias, out.weight, out.bias
            ext = _load_ext()
            lead = hidden_states.shape[:-1]

            if ext is not None and hidden_states.is_contiguous():
                a = hidden_states.reshape(-1, hidden_states.shape[-1])
                M = a.shape[0]
                if M and self._gemm_ok(a, w1, b1):
                    # One 2-SM tcgen05 GEMM does GEMM1 + bias + tanh-GELU, so the
                    # separate pass over the [M, 12288] intermediate disappears.
                    h = self._cutlass_gemm(ext, _pick_g1(M), a, w1, b1)
                    plan2 = _pick_g2(M)
                    if plan2 is not None and self._gemm_ok(h, w2, b2):
                        y = self._cutlass_gemm(ext, plan2, h, w2, b2)
                        STATS["fused_g2"] += 1
                    else:
                        y = F.linear(h, w2, b2)
                    STATS["fused"] += 1
                    return y.view(*lead, y.shape[-1])

            # Round 1 path: cuBLAS GEMM1 + own in-place streaming tanh-GELU.
            # ``h`` is a fresh temporary owned by this call, so the activation
            # runs in place: no second [M, 12288] buffer, and the stores land on
            # lines the loads just pulled in.
            h = F.linear(hidden_states, w1, b1)
            if h.is_contiguous():
                _gelu_tanh_(h)
                STATS["fallback"] += 1
                return F.linear(h, w2, b2)

        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states
