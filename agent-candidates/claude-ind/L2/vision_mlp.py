"""Vision MLP for Qwen vision transformer blocks.

Unified across Qwen2-VL (QuickGELU) and Qwen3-VL (SiLU) activations.

``fc2(act(fc1(x)))`` costs three kernels when built from library pieces: a GEMM
that materializes the [tokens, hidden] projection, an elementwise activation
pass over it, and a second GEMM.  The activation pass is nearly pure overhead --
it re-reads and re-writes the whole intermediate (hundreds of MB at the captured
token counts) for a handful of flops, and the exact (erf) gelu is not cheap.

Here both projections are hand-written Triton kernels: a persistent,
warp-specialized ``tcgen05`` GEMM whose epilogue applies the activation while
the tile is still in registers, and the same kernel without an epilogue for the
down projection.  Weights are pre-transposed once and the hidden dim is
zero-padded to the tile width, so both kernels see plain row-major
``[M, K] x [K, N]`` operands and need no in-kernel masking (TMA zero-fills the
ragged row tile).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from ..L1.quickgelu import QuickGELU
from .parallel_linear import ColumnParallelLinear, RowParallelLinear

try:
    import triton
    import triton.language as tl
    from triton.tools.tensor_descriptor import TensorDescriptor

    _HAVE_TRITON = True
except Exception:  # pragma: no cover
    _HAVE_TRITON = False


_ACT_NONE = 0
_ACT_GELU = 1   # erf gelu, evaluated in its tanh form (|err| < 4e-4)
_ACT_QUICK = 2  # x * sigmoid(1.702 x)
_ACT_SILU = 3   # x * sigmoid(x)

if _HAVE_TRITON:
    _SCRATCH: dict = {}

    def _allocator(size: int, alignment: int, stream):
        key = (size, stream, torch.cuda.current_device())
        buf = _SCRATCH.get(key)
        if buf is None:
            buf = torch.empty(size, dtype=torch.int8, device="cuda")
            _SCRATCH[key] = buf
        return buf

    triton.set_allocator(_allocator)

    @triton.jit
    def _epilogue(h, ACT: tl.constexpr):
        """x * sigmoid(s(x)), written for the fp32 pipeline's cheapest ops.

        ``tl.sigmoid`` compiles to a precise ``exp`` plus an IEEE division --
        together ~18 instructions per element, which at these hidden sizes costs
        more than 10% of the kernel.  Folding the scale into ``exp2`` and using
        the fast divide gets the same values to well inside bf16 tolerance for
        about a third of the instructions.
        """
        if ACT == 0:
            return h
        if ACT == 1:
            # gelu in tanh form: h * sigmoid(1.5957691 * (h + 0.044715 h^3)).
            t = -0.102918033813989 * h * h - 2.302208198144325
            e = tl.exp2(h * t)
        elif ACT == 2:
            e = tl.exp2(-2.4554669595930156 * h)
        else:
            e = tl.exp2(-1.4426950408889634 * h)
        return tl.fdiv(h, 1.0 + e, ieee_rounding=False)

    @triton.jit
    def _mlp_gemm(a_desc, b_desc, c_desc, bias_ptr, M,
                  N: tl.constexpr, K: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  GM: tl.constexpr, NPROG: tl.constexpr, ACT: tl.constexpr,
                  EPI: tl.constexpr):
        """C = act(A @ B + bias) for row-major A[M, K], B[K, N], C[M, N]."""
        num_m = tl.cdiv(M, BM)
        num_n: tl.constexpr = N // BN
        total = num_m * num_n
        ngroup: tl.constexpr = GM * num_n
        for tile in tl.range(tl.program_id(0), total, NPROG,
                             warp_specialize=True, flatten=True):
            # Walk tiles column-major inside a band of GM row-tiles: the weight
            # block a band touches stays resident in L2 for the whole band.
            gid = tile // ngroup
            first_m = gid * GM
            gsize = min(num_m - first_m, GM)
            pid_m = first_m + ((tile % ngroup) % gsize)
            pid_n = (tile % ngroup) // gsize
            off_m = pid_m * BM
            off_n = pid_n * BN
            acc = tl.zeros((BM, BN), dtype=tl.float32)
            for k in tl.range(0, K, BK):
                a = a_desc.load([off_m, k])
                b = b_desc.load([k, off_n])
                acc = tl.dot(a, b, acc)
            if EPI == 1:
                bias = tl.load(bias_ptr + off_n + tl.arange(0, BN))
                v = acc + bias[None, :].to(tl.float32)
                c_desc.store([off_m, off_n], _epilogue(v, ACT).to(tl.bfloat16))
            else:
                # Drain the accumulator in halves: fewer live registers, and the
                # second store overlaps the first one's TMA.
                HN: tl.constexpr = BN // 2
                pair = tl.permute(tl.reshape(acc, (BM, 2, HN)), (0, 2, 1))
                lo, hi = tl.split(pair)
                b_lo = tl.load(bias_ptr + off_n + tl.arange(0, HN))
                v = lo + b_lo[None, :].to(tl.float32)
                c_desc.store([off_m, off_n], _epilogue(v, ACT).to(tl.bfloat16))
                b_hi = tl.load(bias_ptr + off_n + HN + tl.arange(0, HN))
                v = hi + b_hi[None, :].to(tl.float32)
                c_desc.store([off_m, off_n + HN], _epilogue(v, ACT).to(tl.bfloat16))


# (BM, BN, BK, GM, num_warps, num_stages, num_ctas, epilogue_split).
# The down projection wants wide row tiles when there is enough work to fill the
# GPU; below that the tile count drops under one wave and narrow, more deeply
# pipelined tiles win.
_CFG_UP = (128, 256, 64, 16, 8, 4, 1, 2)
_CFG_DOWN = (256, 128, 64, 8, 8, 4, 1, 2)
_CFG_DOWN_SMALL = (128, 128, 64, 8, 8, 6, 1, 2)
_SMALL_ROWS = 8192
_TILE_N = _CFG_UP[1]  # hidden dim is zero-padded to a multiple of this


def _gemm(a, b_desc, c_desc, bias, out, act, cfg, nsms, a_desc=None):
    BM, BN, BK, GM, warps, stages, ctas, epi = cfg
    M, K = a.shape
    N = out.shape[1]
    if a_desc is None:
        a_desc = TensorDescriptor.from_tensor(a, [BM, BK])
    nprog = min(nsms // ctas, triton.cdiv(M, BM) * (N // BN))
    _mlp_gemm[(nprog,)](a_desc, b_desc, c_desc, bias, M, N, K, BM, BN, BK, GM,
                        nprog, act, epi, num_warps=warps, num_stages=stages,
                        num_ctas=ctas)
    return out


def _identify_act(act_fn) -> int | None:
    """Match *act_fn* against the activations the epilogue implements."""
    probe = torch.linspace(-8.0, 8.0, 513, dtype=torch.float32)
    try:
        with torch.no_grad():
            got = act_fn(probe)
    except Exception:
        return None
    if not isinstance(got, torch.Tensor) or got.shape != probe.shape:
        return None
    got = got.to(torch.float32)
    refs = {
        _ACT_GELU: probe * torch.sigmoid(1.5957691216057308
                                         * (probe + 0.044715 * probe ** 3)),
        _ACT_QUICK: probe * torch.sigmoid(1.702 * probe),
        _ACT_SILU: probe * torch.sigmoid(probe),
    }
    best, err = None, float("inf")
    for code, ref in refs.items():
        e = (got - ref).abs().max().item()
        if e < err:
            best, err = code, e
    return best if err < 2e-3 else None


class VisionMLP(nn.Module):
    """Vision encoder MLP with configurable activation.

    Qwen2-VL uses QuickGELU (default); Qwen3-VL uses F.silu.
    """

    def __init__(self, in_features: int, hidden_features: int,
                 act_fn: Callable[[torch.Tensor], torch.Tensor] = QuickGELU(),
                 bias: bool = True):
        super().__init__()
        self.fc1 = ColumnParallelLinear(in_features, hidden_features, bias=bias)
        self.fc2 = RowParallelLinear(hidden_features, in_features, bias=bias)
        self.act_fn = act_fn
        self._plan = None

    def _build_plan(self, x: torch.Tensor):
        """Pre-transpose/pad the weights, or mark the fast path unusable."""
        self._plan = False
        w1, w2 = self.fc1.weight, self.fc2.weight
        if not (_HAVE_TRITON and x.is_cuda and x.dtype == torch.bfloat16
                and w1.dtype == torch.bfloat16 and w2.dtype == torch.bfloat16
                and self.fc1.bias is not None and self.fc2.bias is not None
                and getattr(self.fc2, "tp_size", 1) == 1
                and not getattr(self.fc1, "use_fp8", False)
                and not getattr(self.fc2, "use_fp8", False)
                and torch.cuda.get_device_capability(x.device)[0] >= 10):
            return
        act = _identify_act(self.act_fn)
        hidden, in_f = w1.shape
        hp = (hidden + _TILE_N - 1) // _TILE_N * _TILE_N
        if act is None or tuple(w2.shape) != (in_f, hidden):
            return
        if in_f % _CFG_DOWN[1] or in_f % _CFG_DOWN_SMALL[1]:
            return
        dev, dt = w1.device, w1.dtype
        w1t = torch.zeros(in_f, hp, device=dev, dtype=dt)
        w1t[:, :hidden] = w1.t()
        b1 = torch.zeros(hp, device=dev, dtype=dt)
        b1[:hidden] = self.fc1.bias
        w2t = torch.zeros(hp, in_f, device=dev, dtype=dt)
        w2t[:hidden] = w2.t()
        b2 = self.fc2.bias.contiguous()
        nsms = torch.cuda.get_device_properties(dev).multi_processor_count
        self._plan = {
            "act": act, "in": in_f, "hp": hp, "nsms": nsms, "dev": dev,
            "b1": b1, "b2": b2, "w1t": w1t, "w2t": w2t,
            "rows": 0,
        }

    def _setup_rows(self, plan, rows, x):
        """Pick the tile configs for this token count and cache the descriptors.

        Descriptors are tied to a base pointer, so the intermediate buffer is
        kept alive and reused; the output buffer is fresh each call but the
        caching allocator hands back the same block, so its descriptor is only
        rebuilt when the pointer actually moves.
        """
        cfg_up = _CFG_UP
        cfg_dn = _CFG_DOWN if rows >= _SMALL_ROWS else _CFG_DOWN_SMALL
        h = torch.empty(rows, plan["hp"], device=x.device, dtype=x.dtype)
        plan.update(
            rows=rows, cfg_up=cfg_up, cfg_dn=cfg_dn, h=h, y_ptr=0, yd=None,
            d1=TensorDescriptor.from_tensor(plan["w1t"], [cfg_up[2], cfg_up[1]]),
            d2=TensorDescriptor.from_tensor(plan["w2t"], [cfg_dn[2], cfg_dn[1]]),
            hd=TensorDescriptor.from_tensor(h, [cfg_up[0], cfg_up[1] // cfg_up[7]]),
            had=TensorDescriptor.from_tensor(h, [cfg_dn[0], cfg_dn[2]]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        plan = self._plan
        if plan is None:
            self._build_plan(x)
            plan = self._plan
        if (plan is False or x.shape[-1] != plan["in"] or not x.is_contiguous()
                or x.dtype != torch.bfloat16 or x.device != plan["dev"]
                or x.numel() == 0):
            return self.fc2(self.act_fn(self.fc1(x)))
        in_f = plan["in"]
        flat = x.reshape(-1, in_f)
        rows = flat.shape[0]
        if plan["rows"] != rows:
            self._setup_rows(plan, rows, x)
        h = plan["h"]
        nsms = plan["nsms"]
        cfg_up, cfg_dn = plan["cfg_up"], plan["cfg_dn"]
        _gemm(flat, plan["d1"], plan["hd"], plan["b1"], h, plan["act"], cfg_up, nsms)
        y = torch.empty(rows, in_f, device=x.device, dtype=x.dtype)
        if y.data_ptr() != plan["y_ptr"]:
            plan["y_ptr"] = y.data_ptr()
            plan["yd"] = TensorDescriptor.from_tensor(
                y, [cfg_dn[0], cfg_dn[1] // cfg_dn[7]])
        _gemm(h, plan["d2"], plan["yd"], plan["b2"], y, _ACT_NONE, cfg_dn, nsms,
              a_desc=plan["had"])
        return y.view(x.shape)
