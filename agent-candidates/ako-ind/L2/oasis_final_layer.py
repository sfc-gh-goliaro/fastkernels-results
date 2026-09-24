"""Oasis final DiT projection layer, fused into two Triton launches.

The layer is
    m    = Linear_m(silu(c))                  # [*, 2H] -> shift, scale
    x    = LayerNorm(x) * (1 + scale) + shift
    out  = Linear(x)                          # H -> P*P*out_channels

At the captured sizes (H=1024, out=64, at most 6*144=864 token rows) this is
launch- and latency-bound, not FLOP bound: the whole op touches ~6MB (4MB of
which is the modulation weight) while the eager baseline spends ~67us on ~9
separate kernels plus ModuleList/chunk/unsqueeze Python work.  Everything
collapses into two kernels:

``_pre_kernel``    everything that depends only on the inputs, in one launch:
                   a split slice of the modulation GEMV *and* partial
                   sum/sum-of-squares for the LayerNorm.  Splitting the GEMV
                   over the contraction as well as over the 2H output rows
                   takes it from 128 to ~600 programs, and the 4MB weight from
                   0.7 to 1.9 TB/s.  Partials are left in fp32 for the consumer
                   to add up -- cheaper than a third launch or an atomics
                   buffer that would need zeroing.

``_fused_kernel``  norm + modulate + project, one program per (batch, 16 token)
                   tile.  Mean/rstd come from the prologue, so x is read
                   exactly once, normalized and modulated in registers, and fed
                   straight into an MMA against the projection weight.  Nothing
                   between the norm and the projection touches HBM.

Two launches is the optimum, not a compromise: in this harness each additional
launch costs ~2.05us of measured time, and splitting the projection's
contraction across CTAs (which does cut it from 7.7 to 5.6us) loses overall once
the fp32 partial buffer's reduce launch is paid for.

The captured ``x`` is a permuted view -- shape [1, B, 9, 16, 1024] with strides
[.., H*T, 16, 1, T] -- i.e. channel-major ``[B, C, T]`` in memory.  The kernel
takes the three strides directly instead of forcing a contiguous copy, so the
1024-wide reduction axis is the strided one and the 144 token columns are the
contiguous (coalesced) one.

Both kernels reproduce the eager rounding chain exactly (see ``_fused_kernel``)
-- being *more* accurate than the baseline is a correctness failure here, since
the harness compares against it elementwise at fp16 tolerances.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
@triton.jit
def _pre_kernel(CIN, WM, PART, X, S12,
                nb, spk, sxb, sxc, sxt, sps, soff, nr, T,
                C: tl.constexpr, J: tl.constexpr, BJ: tl.constexpr,
                BC: tl.constexpr, BB: tl.constexpr, NJ: tl.constexpr,
                NMOD: tl.constexpr, BTS: tl.constexpr, BCS: tl.constexpr,
                NTS: tl.constexpr):
    """Everything that depends only on the inputs, in one launch.

    Programs below NMOD do a slice of the modulation GEMV::

        PART[k, b, j] = sum_{c in chunk k} silu(CIN[b, c]) * WM[j, c]

    ``BB`` pads the batch extent up to the MMA minimum of 16; B is at most 6
    here, so the padded lanes are free next to the weight traffic.  The bias
    and the fp16 rounding are left to the consumer, after the partials are
    summed, so the rounding lands exactly where ``F.linear`` puts it.

    The remaining programs reduce a slice of a token row of x into partial
    sum / sum-of-squares, so the projection kernel gets mean and rstd for free
    and reads x exactly once.  Both halves are pure input reads, so they share
    a launch and their combined ~470 programs stream the 5.7MB of weights and
    activations with far more memory parallelism than the 54-program
    projection kernel could on its own.
    """
    pid = tl.program_id(0)
    if pid < NMOD:
        j = (pid % NJ) * BJ + tl.arange(0, BJ)
        co = (pid // NJ) * BC + tl.arange(0, BC)
        b = tl.arange(0, BB)
        bm = b < nb
        w = tl.load(WM + j[:, None] * C + co[None, :])
        v = tl.load(CIN + b[:, None] * C + co[None, :], mask=bm[:, None],
                    other=0.0).to(tl.float32)
        a = (v * tl.sigmoid(v)).to(w.dtype)
        acc = tl.dot(w, tl.trans(a))
        tl.store(PART + (pid // NJ) * spk + b[:, None] * J + j[None, :],
                 tl.trans(acc), mask=bm[:, None])
    else:
        # Distinct names from the branch above: Triton unifies same-named
        # values across the two blocks and these tiles differ in shape.
        q = pid - NMOD
        ct = q // nr
        r = q % nr
        st = (r % NTS) * BTS + tl.arange(0, BTS)
        stm = st < T
        sco = ct * BCS + tl.arange(0, BCS)
        xv = tl.load(X + (r // NTS) * sxb + st[None, :] * sxt + sco[:, None] * sxc,
                     mask=stm[None, :], other=0.0).to(tl.float32)
        p = S12 + ct * sps + (r // NTS) * T + st
        tl.store(p, tl.sum(xv, 0), mask=stm)
        tl.store(p + soff, tl.sum(xv * xv, 0), mask=stm)


@triton.jit
def _fused_kernel(X, PART, BM, S12, W, BIAS, OUT,
                  sxb, sxc, sxt, spk, sps, soff, T, O,
                  C: tl.constexpr, J: tl.constexpr, BT: tl.constexpr,
                  BC: tl.constexpr, BO: tl.constexpr, NT: tl.constexpr,
                  NS: tl.constexpr, NSS: tl.constexpr,
                  EPS: tl.constexpr, INVC: tl.constexpr):
    """OUT[b, t, :] = W @ (norm(X[b, :, t]) * (1 + scale) + shift) + BIAS.

    A program must own whole 1024-wide rows -- the fp16 rounding of the
    normalized value has to happen before the projection, exactly as eager does
    it -- which caps the grid at ceil(T/BT) per batch, 54 programs at the
    captured sizes.  That is few enough that the kernel costs the same at B=2 as
    at B=6, but splitting either the output columns or the contraction to buy
    grid measured worse (see ITERATIONS.md).
    """
    pid = tl.program_id(0)
    b = pid // NT
    t = (pid % NT) * BT + tl.arange(0, BT)
    tm = t < T
    xp = X + b * sxb + t[None, :] * sxt          # [1, BT]

    # ---- mean / rstd from the prologue's partial sums ----
    sp = S12 + b * T + t[None, :] + tl.arange(0, NSS)[:, None] * sps
    s1 = tl.sum(tl.load(sp, mask=tm[None, :], other=0.0), 0)
    s2 = tl.sum(tl.load(sp + soff, mask=tm[None, :], other=0.0), 0)
    mu = s1 * INVC
    rstd = 1.0 / tl.sqrt(s2 * INVC - mu * mu + EPS)

    # ---- single pass over x: modulate in registers, project through the MMA --
    o = tl.arange(0, BO)
    om = o < O
    pb = PART + b * J
    k = tl.arange(0, NS)
    acc = tl.zeros((BO, BT), dtype=tl.float32)
    for c0 in range(0, C, BC):
        co = c0 + tl.arange(0, BC)
        v = tl.load(xp + co[:, None] * sxc, mask=tm[None, :], other=0.0).to(tl.float32)
        wt = tl.load(W + o[:, None] * C + co[None, :], mask=om[:, None], other=0.0)
        kp = pb + k[:, None] * spk
        shf = tl.sum(tl.load(kp + co[None, :]), 0)
        scf = tl.sum(tl.load(kp + C + co[None, :]), 0)
        sh = (shf + tl.load(BM + co).to(tl.float32)).to(wt.dtype)
        sc = (scf + tl.load(BM + C + co).to(tl.float32)).to(wt.dtype)
        # The rounding points below mirror the eager chain exactly:
        # fp16(modulation) -> fp16(1 + scale) -> fp16(layernorm)
        # -> fp16(* (1+scale)) -> fp16(+ shift) -> fp16 mma.
        # This needs ``enable_fp_fusion=False`` at the launch site: with ptxas
        # fmad on, the mul/add contract into one fp16 fma and skip a rounding
        # the eager path performs, which flips ~23% of the fp16 results by a
        # ULP and drops the harness match ratio to ~0.984 on large weights.
        sc = (sc.to(tl.float32) + 1.0).to(wt.dtype)
        n1 = ((v - mu[None, :]) * rstd[None, :]).to(wt.dtype)
        acc = tl.dot(wt, n1 * sc[:, None] + sh[:, None], acc)
    acc += tl.load(BIAS + o, mask=om, other=0.0).to(tl.float32)[:, None]

    tl.store(OUT + b * (T * O) + t[:, None] * O + o[None, :],
             tl.trans(acc).to(OUT.dtype.element_ty), mask=tm[:, None] & om[None, :])


# ---------------------------------------------------------------------------
# Parameter containers -- names chosen so ``state_dict`` matches the baseline
# (``linear.{weight,bias}``, ``adaLN_modulation.1.{weight,bias}``).
# ---------------------------------------------------------------------------
class _Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

    def forward(self, x):  # kept for API parity; the fused path bypasses it
        return F.linear(x, self.weight, self.bias)


class _SiLU(nn.Module):
    def forward(self, x):
        return F.silu(x)


def _flat_stride(sizes, strides):
    """Base stride of a run of dims that is C-contiguous among itself, else None."""
    if not sizes:
        return 0
    base = strides[-1]
    acc = base
    for s, st in zip(reversed(sizes), reversed(strides)):
        if s != 1 and st != acc:
            return None
        acc *= s
    return base


def _raw_stream(index):
    """Current CUDA stream handle, via the cheapest API available.

    ``torch.cuda.current_stream().cuda_stream`` builds a Stream object; the
    private accessor is a single C call.  Read per launch rather than cached so
    the kernels land on whatever stream the caller (or a graph capture) is on.
    """
    return torch.cuda.current_stream(index).cuda_stream


_get_raw = getattr(torch._C, "_cuda_getCurrentRawStream", None)
if _get_raw is not None:
    _raw_stream = _get_raw


# Tile shapes picked by sweeping both kernels on B200 (see ITERATIONS.md).
_BJ, _BC_M, _W_M = 32, 128, 4        # modulation: 2H-row tile, contraction chunk
_BTS, _BC_S = 16, 256                # layer-norm statistics tiles
_BT, _BC_F, _W_F, _ST_F = 16, 512, 4, 2   # projection


class OasisFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_features = patch_size * patch_size * out_channels
        self.eps = 1e-6
        self.linear = _Linear(hidden_size, self.out_features, bias=True)
        self.adaLN_modulation = nn.ModuleList(
            [_SiLU(), _Linear(hidden_size, 2 * hidden_size, bias=True)]
        )
        self._cfg: dict = {}
        # Plain (unregistered) handles to the parameter objects: attribute
        # lookup through nn.Module.__getattr__ is not free at these latencies,
        # and holding the Parameter (not its .data) stays valid across
        # ``load_state_dict`` and dtype casts.
        object.__setattr__(self, "_wref", None)

    # -- eager fallback, for anything the fused path does not cover ----------
    def _fallback(self, x, c):
        m = self.adaLN_modulation[1](F.silu(c))
        shift, scale = m.chunk(2, dim=-1)
        while shift.dim() < x.dim():
            shift = shift.unsqueeze(-2)
            scale = scale.unsqueeze(-2)
        n = F.layer_norm(x.float(), (self.hidden_size,), None, None, self.eps).to(x.dtype)
        return self.linear(n * (1 + scale) + shift)

    def _build(self, x, c, key):
        """Validate the layout once per (shape, stride) and cache the launch plan."""
        H = self.hidden_size
        cfg = False
        w, bias, wm, bm = self._wref
        ok = (
            x.is_cuda
            and x.dtype in (torch.float16, torch.bfloat16)
            and c.dtype == x.dtype
            and c.is_contiguous()
            and x.dim() > c.dim()
            and x.shape[-1] == H
            and c.shape[-1] == H
            and c.shape[:-1] == x.shape[: c.dim() - 1]
            and bias is not None
            and w.is_contiguous()
            and wm.is_contiguous()
            and H % _BC_M == 0
            and H % _BC_F == 0
            and H % _BC_S == 0
            and (2 * H) % _BJ == 0
            and (H // _BC_S) == triton.next_power_of_2(H // _BC_S)
        )
        if ok:
            nb = c.dim() - 1
            sizes, strides = list(x.shape), list(x.stride())
            sxb = _flat_stride(sizes[:nb], strides[:nb])
            sxt = _flat_stride(sizes[nb:-1], strides[nb:-1])
            if sxb is not None and sxt is not None:
                B = 1
                for s in sizes[:nb]:
                    B *= s
                T = 1
                for s in sizes[nb:-1]:
                    T *= s
                O = self.out_features
                J, NS, NJ = 2 * H, H // _BC_M, (2 * H) // _BJ
                NT = triton.cdiv(T, _BT)
                NTS, NSS = triton.cdiv(T, _BTS), H // _BC_S
                BB = max(16, triton.next_power_of_2(B))
                nmod, nr = NJ * NS, B * NTS
                sxc = strides[-1]
                part = torch.empty((NS, B, J), device=x.device, dtype=torch.float32)
                s12 = torch.empty((2, NSS, B, T), device=x.device, dtype=torch.float32)
                # Positional argument lists in kernel-declaration order
                # (constexprs included) so they can be handed straight to the
                # cached CompiledKernel.run -- see ``forward``.
                # Pointer arguments are passed as raw ints: Triton's C launcher
                # otherwise does a ``data_ptr`` attribute lookup + call per
                # tensor argument, which is a real cost next to a ~1us kernel.
                # The referenced tensors are all kept alive (weights in _wref,
                # scratch in this cfg, x/c/out live across the launch).
                cfg = dict(
                    g1=nmod + NSS * nr, g2=B * NT,
                    oshape=tuple(sizes[:-1]) + (O,),
                    plan=None, part=part, s12=s12,
                    tp1=(wm, part, s12), tp2=(part, bm, s12, w, bias),
                    a1=[0, wm.data_ptr(), part.data_ptr(), 0, s12.data_ptr(),
                        B, B * J, sxb, sxc, sxt, B * T, NSS * B * T, nr, T,
                        H, J, _BJ, _BC_M, BB, NJ, nmod, _BTS, _BC_S, NTS],
                    a2=[0, part.data_ptr(), bm.data_ptr(), s12.data_ptr(),
                        w.data_ptr(), bias.data_ptr(), 0,
                        sxb, sxc, sxt, B * J, B * T, NSS * B * T, T, O,
                        H, J, _BT, _BC_F, triton.next_power_of_2(O), NT, NS,
                        NSS, self.eps, 1.0 / H],
                )
        self._cfg[key] = cfg
        return cfg

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if self._wref is None:
            object.__setattr__(self, "_wref",
                               (self.linear.weight, self.linear.bias,
                                self.adaLN_modulation[1].weight,
                                self.adaLN_modulation[1].bias))
        key = (x.shape, x.stride(), c.shape)
        cfg = self._cfg.get(key)
        if cfg is None:
            cfg = self._build(x, c, key)
        if cfg is False:
            return self._fallback(x, c)

        a1, a2 = cfg["a1"], cfg["a2"]
        xp = x.data_ptr()
        a1[0] = c.data_ptr()
        a1[3] = xp
        a2[0] = xp
        out = x.new_empty(cfg["oshape"])
        a2[6] = out.data_ptr()
        plan = cfg["plan"]
        if plan is not None:
            try:
                stream = _raw_stream(x.device.index)
                r1, f1, m1, r2, f2, m2 = plan
                r1(cfg["g1"], 1, 1, stream, f1, m1, None, None, None, *a1)
                r2(cfg["g2"], 1, 1, stream, f2, m2, None, None, None, *a2)
                return out
            except Exception:  # noqa: BLE001 - Triton internals moved; slow path
                cfg["plan"] = None
        # First call (or a launcher mismatch): go through the normal dispatch,
        # which compiles, and keep the CompiledKernel it returns.  Later calls
        # skip Triton's argument binding and cache-key hashing, ~10us of host
        # time per launch -- several times this op's GPU cost.  Compilation
        # needs real tensors so the binder infers pointer types.
        b1 = list(a1)
        b1[0], b1[3] = c, x
        b1[1], b1[2], b1[4] = cfg["tp1"]
        b2 = list(a2)
        b2[0], b2[6] = x, out
        b2[1], b2[2], b2[3], b2[4], b2[5] = cfg["tp2"]
        k1 = _pre_kernel[(cfg["g1"],)](*b1, num_warps=_W_M)
        k2 = _fused_kernel[(cfg["g2"],)](
            *b2, num_warps=_W_F, num_stages=_ST_F, enable_fp_fusion=False)
        try:
            cfg["plan"] = (k1.run, k1.function, k1.packed_metadata,
                           k2.run, k2.function, k2.packed_metadata)
        except AttributeError:
            cfg["plan"] = None
        return out
