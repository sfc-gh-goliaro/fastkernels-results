"""MSA pair-weighted averaging for AlphaFold3 (Algorithm 10).

Weighted averaging over the MSA representation using pair activations,
NOT key-query self-attention.

Reference: openfold3/core/model/layers/msa.py MSAPairWeightedAveraging

At the captured shape -- ``m[1, 8, 16, 64]``, ``z[1, 16, 16, 128]``,
``c_hidden=8``, ``no_heads=8`` -- this operator is ~80 KB of live tensors and
~0.26 MFLOP, and the reference path spends it on **20 kernel launches** (three
of them just the fp32 cast sandwich inside each LayerNorm). Measured on B200:
250 us of Python dispatch against 50 us of device time. It is a launch-count
problem, not a FLOP or a bandwidth problem, so the whole algorithm collapses
into **one** Triton kernel -- one program per output row of
``out[b, s, q, :]`` -- and the reference composition of the frozen L1 winners
stays as the correctness oracle and the fallback for anything off the fast path.

Why one program per *output row* rather than per sequence or per query row: with
``B * N_seq * N_res = 128`` programs on 148 SMs everything is one wave, so
per-program work *is* the kernel's latency and cross-program redundancy costs no
wall time at all. Coarser grids concentrate work instead of removing it.
Measured on B200 (paired against the reference in the benchmark's own timing
loop, since the clocks here cannot be locked):

===========================================  ========  =========  ========
variant                                          grid  device us  speedup
===========================================  ========  =========  ========
two kernels: w to HBM, then the output path    16 + 8       5.29     11.6x
one kernel, one program per query row              16       7.65     10.1x
one kernel, one program per output row            128       3.60     14.3x
===========================================  ========  =========  ========

The middle row is the instructive one: it buys the same single launch but folds
``linear_v``/``linear_g`` for all ``N_seq * N_res`` rows into 16 programs, i.e.
3.5x the MMA work on one SM, and loses more than the second launch cost.

Three things beyond the fusion itself matter, in decreasing order:

* **PDL.** The benchmark's timing loop copies the inputs into a shifting memory
  pool *inside* the timed region, so a producer is always draining ahead of us.
  Measured there: an extra Triton kernel costs 4.1 us without PDL and ~0 with
  it. ``gdc_wait()`` sits after the address arithmetic and before the first
  load -- it has to cover that pool copy, not merely a preceding kernel of ours,
  and (see ``_fused``) not merely the loads that read the copied tensors.
* **Weight layout, resolved once.** ``linear_z``'s weight is transposed *and*
  each head's column replicated ``c_hidden`` times, which lands the pair
  projection directly on the ``(head, channel)`` axis the value tensor uses; the
  softmax over ``k`` then runs on exactly the layout the weighted average wants,
  so the average is one multiply reduced over ``k`` -- no einsum, no per-head
  MMA, no block-diagonal operand, no transpose. ``linear_v`` and ``linear_g``
  share one pre-transposed buffer, and the LayerNorm affines are cast to fp32
  once instead of on every call.
* **Host cost.** The launch plan is memoized on ``(dtype, shape, stride)`` and
  after the first call the kernel is invoked through its own C launcher rather
  than Triton's per-call binder: 22 us of Python per call down to 12 us. At this
  size the L2 flush the harness performs before each timed iteration hides most
  of the host side, so this is worth little here -- but it is what keeps the
  operator device-bound rather than dispatch-bound if the host gets busier.

LayerNorm and softmax accumulate in fp32 while loading and storing the input
dtype, matching the L1 winners. The intermediates the reference materializes in
bf16 (``z_norm``, ``z_weights``, ``v``, ``g``, ``o``) are rounded to bf16 at the
same points, so the fused result tracks the reference's rounding and not merely
its algebra: max abs error 6.1e-5 against a 1e-2 tolerance, all elements matched.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._C import _cuda_getCurrentRawStream as _raw_stream
from torch._C import _cuda_getDevice as _cur_device
from triton.language.extra.cuda.gdc import gdc_launch_dependents, gdc_wait

from ..L1.sigmoid import Sigmoid
from ..L1.softmax import Softmax
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear

_LOG2E = tl.constexpr(1.4426950408889634)

# Largest register tile one program may hold. Every tile in the kernel is
# (rows x channels) of one pair row or one MSA row, so this single bound admits
# the fast path; anything wider goes to the reference, which is correct at any
# shape.
_MAX_TILE = 8192
_FAST_DTYPES = (torch.bfloat16, torch.float16)


def _permute_final_dims(tensor: torch.Tensor, inds: tuple[int, ...]) -> torch.Tensor:
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


@triton.jit
def _fused(Z, MASK, M, WZD, WVGT, WOT, LNZW, LNZB, LNMW, LNMB, OUT,
           S: tl.constexpr, R: tl.constexpr, CZ: tl.constexpr, CM: tl.constexpr,
           BR: tl.constexpr, BCZ: tl.constexpr, BCM: tl.constexpr,
           BHD: tl.constexpr, EXACT_R: tl.constexpr, EXACT_CZ: tl.constexpr,
           EXACT_CM: tl.constexpr, INF, EPSZ, EPSM, HAS_MASK: tl.constexpr,
           HAS_ZW: tl.constexpr, HAS_ZB: tl.constexpr, HAS_MW: tl.constexpr,
           HAS_MB: tl.constexpr, PDL: tl.constexpr):
    """All of Algorithm 10, one program per output row.

    ``pid`` walks ``(batch, sequence, query)`` in the output's own row order, so
    a program owns exactly ``out[b, s, q, :]`` and the store is one contiguous
    vector. It needs the pair row ``z[b, q, :, :]`` (recomputed once per
    sequence) and the MSA row ``m[b, s, :, :]`` (recomputed once per query row);
    both redundancies are concurrent, hence free.

    ``linear_o`` contracts the ``(head, channel)`` axis of a single row, so as a
    ``tl.dot`` it wastes fifteen sixteenths of the ``M >= 16`` tile. It is still
    the cheaper form: replacing it with a broadcast-multiply reduced over ``hd``
    measured 3.82 us against 4.25 us on B200, and doing the same to the gate
    projection cost 1.1 us -- at these sizes the tensor cores are so far from
    saturated that fp32 FMA reductions lose to a mostly-idle MMA.
    """
    pid = tl.program_id(0)                 # (b * N_seq + s) * N_res + q
    q = pid % R
    zrow = (pid // (S * R)) * R + q        # b * N_res + q
    k = tl.arange(0, BR)
    c = tl.arange(0, BCZ)
    cm = tl.arange(0, BCM)
    hd = tl.arange(0, BHD)
    zoff = zrow.to(tl.int64) * (R * CZ) + k[:, None] * CZ + c[None, :]
    moff = (pid // R).to(tl.int64) * (R * CM) + k[:, None] * CM + cm[None, :]
    wvg = cm[:, None] * BHD + hd[None, :]
    if PDL:
        # ``gdc_wait`` goes here, before *every* load, not merely before the ones
        # that read m/z/mask. It is tempting to hoist the 40 KB of weight loads
        # above it -- their latency would then overlap the producer's tail, which
        # is the usual advice -- but a pre-wait load can observe pre-producer
        # state from *anything* earlier on the stream, and on the call that
        # (re)builds the cache the weight buffers are themselves being written a
        # few launches back. Measured neutral at this size anyway: after the
        # first program the weights are in L2.
        gdc_wait()
    wzd_t = tl.load(WZD + c[:, None] * BHD + hd[None, :])
    wv_t = tl.load(WVGT + wvg)
    wg_t = tl.load(WVGT + BCM * BHD + wvg)
    wot_t = tl.load(WOT + hd[:, None] * BCM + cm[None, :])
    if HAS_ZW:
        lzw_t = tl.load(LNZW + c)
    if HAS_ZB:
        lzb_t = tl.load(LNZB + c)
    if HAS_MW:
        lmw_t = tl.load(LNMW + cm)
    if HAS_MB:
        lmb_t = tl.load(LNMB + cm)

    # -- pair bias: LayerNorm(z) -> linear_z -> mask bias -> softmax over k ----
    zmask = None
    if not EXACT_R:
        zmask = (k < R)[:, None]
    if not EXACT_CZ:
        czk = c < CZ
        zmask = czk[None, :] if zmask is None else zmask & czk[None, :]
    if zmask is None:
        x = tl.load(Z + zoff).to(tl.float32)
    else:
        x = tl.load(Z + zoff, mask=zmask, other=0.0).to(tl.float32)
    iz = 1.0 / CZ
    mu = tl.sum(x, 1)[:, None] * iz
    if EXACT_CZ:
        dz = x - mu
    else:
        # Padding lanes must contribute 0 to both sums, so they are zeroed after
        # the mean rather than loaded as 0.
        dz = tl.where(czk[None, :], x - mu, 0.0)
    yz = dz * (1.0 / tl.sqrt(tl.sum(dz * dz, 1)[:, None] * iz + EPSZ))
    if HAS_ZW:                              # cached fp32, zero-padded to BCZ
        yz = yz * lzw_t
    if HAS_ZB:
        yz = yz + lzb_t
    # The reference materializes ``z_norm`` in the input dtype before linear_z;
    # round here so the projection sees the same operand. WZD is zero-padded, so
    # neither operand needs a mask.
    logit = tl.dot(yz.to(Z.dtype.element_ty), wzd_t, out_dtype=tl.float32)
    if HAS_MASK:
        mv = tl.load(MASK + zrow.to(tl.int64) * R + k, mask=k < R,
                     other=1.0).to(tl.float32)
        logit += (INF * (mv - 1.0))[:, None]
    if not EXACT_R:
        logit = tl.where((k < R)[:, None], logit, float("-inf"))
    e = tl.exp2((logit - tl.max(logit, 0)[None, :]) * _LOG2E)
    # z_weights is bf16 in the reference before the einsum; round to match.
    wv = (e * (1.0 / tl.sum(e, 0)[None, :])).to(Z.dtype.element_ty)

    # -- MSA path: LayerNorm(m) -> linear_v / linear_g -> sigmoid -------------
    mmask = None
    if not EXACT_R:
        mmask = (k < R)[:, None]
    if not EXACT_CM:
        cmk = cm < CM
        mmask = cmk[None, :] if mmask is None else mmask & cmk[None, :]
    if mmask is None:
        xm = tl.load(M + moff).to(tl.float32)
    else:
        xm = tl.load(M + moff, mask=mmask, other=0.0).to(tl.float32)
    im = 1.0 / CM
    mum = tl.sum(xm, 1)[:, None] * im
    if EXACT_CM:
        dm = xm - mum
    else:
        dm = tl.where((cm < CM)[None, :], xm - mum, 0.0)
    ym = dm * (1.0 / tl.sqrt(tl.sum(dm * dm, 1)[:, None] * im + EPSM))
    if HAS_MW:
        ym = ym * lmw_t
    if HAS_MB:
        ym = ym + lmb_t
    ml = ym.to(M.dtype.element_ty)
    v = tl.dot(ml, wv_t, out_dtype=tl.float32).to(M.dtype.element_ty)
    g = tl.dot(ml, wg_t, out_dtype=tl.float32)
    g = 1.0 / (1.0 + tl.exp2((-g.to(M.dtype.element_ty).to(tl.float32))
                             * _LOG2E))

    # -- weighted average at this query row, gate, linear_o ------------------
    o = tl.sum(v.to(tl.float32) * wv.to(tl.float32), 0)
    # The gate is wanted only at this program's query row, but linear_g was
    # computed for every row and ``q`` is a runtime value, so the tile cannot be
    # indexed; a masked reduction selects the row.
    gsel = tl.sum(tl.where((k == q)[:, None], g, 0.0), 0)
    og = (o.to(M.dtype.element_ty).to(tl.float32) * gsel).to(M.dtype.element_ty)
    out = tl.dot(tl.broadcast_to(og[None, :], (BR, BHD)), wot_t,
                 out_dtype=tl.float32)
    omask = (k == 0)[:, None]
    if not EXACT_CM:
        omask = omask & (cm < CM)[None, :]
    tl.store(OUT + pid.to(tl.int64) * CM + cm[None, :]
             + tl.zeros((BR, 1), tl.int32), out.to(OUT.dtype.element_ty),
             mask=omask)
    if PDL:
        gdc_launch_dependents()


def _pdl_ok(device) -> bool:
    """PDL needs Hopper or newer; resolved once per plan, not per call."""
    try:
        return torch.cuda.get_device_capability(device)[0] >= 9
    except Exception:  # noqa: BLE001
        return False


class MSARowAttentionWithPairBias(nn.Module):
    """AF3 MSA Pair-Weighted Averaging (Algorithm 10).

    Uses pair activations as weights (softmax over token dim) instead of
    key-query attention.  Parameter names match the checkpoint layout:
    linear_v, linear_g, linear_o (no nested mha).

    Args:
        c_m: MSA input channel dimension
        c_z: Pair embedding channel dimension
        c_hidden: Per-head hidden channel dimension
        no_heads: Number of attention heads
        inf: Large constant for masking
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        inf: float = 1e9,
    ):
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf

        self.layer_norm_m = LayerNorm(c_m)
        self.layer_norm_z = LayerNorm(c_z)
        self.linear_z = Linear(c_z, no_heads, bias=False)

        self.linear_v = Linear(c_m, c_hidden * no_heads, bias=False)
        self.linear_g = Linear(c_m, c_hidden * no_heads, bias=False)
        self.linear_o = Linear(c_hidden * no_heads, c_m, bias=False)

        self.sigmoid = Sigmoid()
        self.softmax = Softmax(dim=-1)

        # Launch plans, memoized on the inputs' (dtype, shape, stride).
        self._plans: dict = {}
        # Cached weight buffers (pre-transposed, pre-concatenated, zero-padded,
        # fp32 affine) plus the signature of the Parameters they came from.
        self._wc: tuple | None = None
        self._wsig: tuple | None = None
        self._wkeep: tuple = ()
        # ``_parameters`` of each submodule holding a weight we fold, so the
        # per-call staleness guard is a dict lookup rather than a trip through
        # ``nn.Module.__getattr__`` -- which is a Python function and costs more
        # than the guard it feeds.
        self._pd = tuple(mod._parameters for mod in (
            self.layer_norm_m, self.layer_norm_z, self.linear_z,
            self.linear_v, self.linear_g, self.linear_o))

    # ------------------------------------------------------------------
    # weight preparation
    # ------------------------------------------------------------------
    def _params(self):
        """The eight Parameters the weight cache folds, in a fixed order."""
        pm, pz, lz, lv, lg, lo = self._pd
        return (pm["weight"], pm["bias"], pz["weight"], pz["bias"],
                lz["weight"], lv["weight"], lg["weight"], lo["weight"])

    def _sig(self, params):
        """Staleness signature of the folded Parameters -- ints only.

        Identity (via ``id``) catches a replaced ``Parameter`` or a ``.data``
        re-point; ``_version`` catches an in-place update, which is what
        ``load_state_dict`` does -- it copies into the existing storage and
        leaves ``data_ptr`` untouched, so a pointer guard would miss a
        checkpoint loaded after the first forward.

        Deliberately ints and not the tensors themselves: comparing two tuples
        that hold tensors falls back to ``Tensor.__eq__`` for any pair that is
        not identical, which returns a tensor and raises on ``bool()``. The
        objects are kept alive in ``_wkeep`` so an ``id`` can never be recycled
        under us. Eight C-level lookups, ~0.7 us, against the ten kernel
        launches the rebuild it protects would cost.
        """
        sig = []
        for p in params:
            sig.append(0 if p is None else id(p))
            sig.append(-1 if p is None else p._version)
        return tuple(sig)

    def _weights(self, dtype, device):
        """Build, and cache, the buffers the fused kernel reads.

        Derived *from* the public Parameters rather than replacing them, so
        checkpoint loading is unaffected: every ``Linear`` keeps its own
        ``weight`` and these are transposed, zero-padded copies.
        """
        wc = self._wc
        params = self._params()
        sig = self._sig(params)
        if wc is not None and self._wsig == sig and wc[0] == (dtype, device):
            return wc
        H, D = self.no_heads, self.c_hidden
        CZ, CM, HD = self.c_z, self.c_m, H * D
        BCZ = triton.next_power_of_2(CZ)
        BCM = triton.next_power_of_2(CM)
        BHD = max(16, triton.next_power_of_2(HD))
        opt = dict(dtype=dtype, device=device)

        # linear_z transposed, with each head's column replicated c_hidden
        # times: the projection then lands straight on the (head, channel) axis
        # the value tensor uses, so the softmax output needs no re-layout.
        wzd = torch.zeros((BCZ, BHD), **opt)
        wzd[:CZ, :HD] = self.linear_z.weight.detach().t().to(
            dtype).repeat_interleave(D, dim=1)
        # linear_v and linear_g: same input, same shape, no bias -- one
        # allocation, so the two dots share a cache line stream.
        wvgt = torch.zeros((2, BCM, BHD), **opt)
        wvgt[0, :CM, :HD] = self.linear_v.weight.detach().t().to(dtype)
        wvgt[1, :CM, :HD] = self.linear_g.weight.detach().t().to(dtype)
        wot = torch.zeros((BHD, BCM), **opt)
        wot[:HD, :CM] = self.linear_o.weight.detach().t().to(dtype)

        def f32(p, n):
            """fp32, zero-padded to the kernel's column tile.

            The cast is loop-invariant -- the reference redoes it every call --
            and the padding is what lets the kernel load the affine unmasked:
            padded lanes carry a zero deviation, so scale is irrelevant and a
            zero offset keeps them zero.
            """
            if p is None:
                return None
            buf = torch.zeros((n,), dtype=torch.float32, device=device)
            buf[:p.shape[0]] = p.detach().float()
            return buf

        wc = ((dtype, device), wzd, wvgt, wot,
              f32(self.layer_norm_z.weight, BCZ),
              f32(self.layer_norm_z.bias, BCZ),
              f32(self.layer_norm_m.weight, BCM),
              f32(self.layer_norm_m.bias, BCM))
        self._wc = wc
        self._wkeep = params      # pins the ids ``_wsig`` was built from
        self._wsig = sig
        return wc

    # ------------------------------------------------------------------
    # fused path
    # ------------------------------------------------------------------
    def _build_plan(self, m, z, mask):
        """Resolve the launch once for this (dtype, shape, stride) key.

        Returns ``run(m, z, mask) -> out``, or ``None`` when the shape is
        outside the fast path.
        """
        H = self.no_heads
        CZ, CM, HD = self.c_z, self.c_m, H * self.c_hidden
        if (m.dtype not in _FAST_DTYPES or m.dtype is not z.dtype
                or not m.is_cuda or m.ndim < 3 or z.ndim != m.ndim
                or not m.is_contiguous() or not z.is_contiguous()
                or m.shape[:-3] != z.shape[:-3]
                or m.shape[-1] != CM or z.shape[-1] != CZ
                or z.shape[-2] != z.shape[-3] or m.shape[-2] != z.shape[-2]
                or self.linear_z.weight.shape != (H, CZ)
                or self.linear_v.weight.shape != (HD, CM)
                or self.linear_g.weight.shape != (HD, CM)
                or self.linear_o.weight.shape != (CM, HD)):
            return None
        R, S = z.shape[-2], m.shape[-3]
        B = 1
        for s in m.shape[:-3]:
            B *= s
        if R == 0 or S == 0 or B == 0:
            return None
        BR = max(16, triton.next_power_of_2(R))
        BCZ = triton.next_power_of_2(CZ)
        BCM = triton.next_power_of_2(CM)
        BHD = max(16, triton.next_power_of_2(HD))
        if max(BR * BCZ, BR * BCM, BR * BHD, BHD * BCM) > _MAX_TILE:
            return None
        if mask is not None and (mask.dtype is not m.dtype
                                 or not mask.is_contiguous()
                                 or tuple(mask.shape) != tuple(z.shape[:-1])):
            return None
        dtype, device = m.dtype, m.device
        pdl = _pdl_ok(device)
        oshape = m.shape
        grid = B * S * R
        # One wave of tiny programs: deep per-thread ILP is pointless here, but
        # spreading the widest tile over more warps shortens the reduction trees
        # that the whole (latency-bound) program waits on.
        warps = min(8, max(1, (BR * max(BCZ, BCM, BHD)) // 256))
        # Trailing arguments of _fused, in signature order, everything after
        # OUT. All constexpr and invariant for this plan -- but the generated C
        # launcher still takes them positionally.
        tail = (S, R, CZ, CM, BR, BCZ, BCM, BHD, BR == R, BCZ == CZ, BCM == CM,
                float(self.inf), self.layer_norm_z.eps, self.layer_norm_m.eps,
                mask is not None,
                self.layer_norm_z.weight is not None,
                self.layer_norm_z.bias is not None,
                self.layer_norm_m.weight is not None,
                self.layer_norm_m.bias is not None, pdl)
        devidx = m.get_device()
        # Direct-launch state: [weight-cache identity, launch fn, invariant
        # prefix, weight pointers].
        st: list = [None, None, None, None]

        def run(m, z, mask):
            wc = self._weights(dtype, device)
            out = torch.empty(oshape, dtype=dtype, device=device)
            zp, mp, op = z.data_ptr(), m.data_ptr(), out.data_ptr()
            if st[0] is wc and not ((zp | mp | op) & 15):
                # Triton's ``kernel[grid](...)`` re-binds and re-specializes
                # every argument and rebuilds the launch metadata on each call:
                # ~10 us of Python, where the whole device side is 3.6 us.
                # Everything it derives is invariant for this plan except
                # pointer alignment -- hence the ``& 15`` guard -- so after the
                # first call we hold the compiled kernel's own C entry point and
                # pass the arguments ``CudaLauncher.__call__`` would insert.
                st[1](grid, 1, 1, _raw_stream(devidx), *st[2], zp,
                      None if mask is None else mask.data_ptr(), mp, *st[3],
                      op, *tail)
                return out
            kern = _fused[(grid,)](z, mask, m, wc[1], wc[2], wc[3], wc[4],
                                   wc[5], wc[6], wc[7], out, *tail,
                                   num_warps=warps, num_stages=1,
                                   launch_pdl=pdl)
            # ``CudaLauncher`` is Triton-internal, so every piece is fetched
            # defensively: if a future Triton reshapes it, or the kernel turns
            # out to need scratch (ours never does -- no in-kernel allocation,
            # no profiling), we never memoize and every call keeps going through
            # the supported path. Slower, still right.
            launcher = None if kern is None else kern.run
            raw = getattr(launcher, "launch", None)
            if (raw is not None and _cur_device() == devidx
                    and getattr(launcher, "global_scratch_size", None) == 0
                    and getattr(launcher, "profile_scratch_size", None) == 0):
                st[1] = raw
                st[2] = (kern.function, launcher.launch_cooperative_grid,
                         launcher.launch_pdl, None, None,
                         kern.packed_metadata, None, None, None)
                st[3] = tuple(None if t is None else t.data_ptr()
                              for t in wc[1:])
                st[0] = wc
            return out

        return run

    # ------------------------------------------------------------------
    # the unfused reference path -- the frozen L1 winners, composed
    # ------------------------------------------------------------------
    def _reference(self, m, z, mask):
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        # Pair bias: [*, 1, no_heads, N_res, N_res]
        mask_bias = (self.inf * (mask - 1))[..., None, None, :, :]
        z_norm = self.layer_norm_z(z)
        z_proj = self.linear_z(z_norm)
        z_weights = _permute_final_dims(z_proj, (2, 0, 1)).unsqueeze(-4)
        z_weights = z_weights + mask_bias
        z_weights = self.softmax(z_weights)

        m = self.layer_norm_m(m)

        # Value projection
        v = self.linear_v(m)
        v = v.view(v.shape[:-1] + (self.no_heads, -1))
        v = v.transpose(-2, -3)  # [*, N_seq, H, N_res, C_hidden]

        # Weighted average: [*, N_seq, H, N_res, C_hidden]
        o = torch.einsum("...hqk,...hkc->...qhc", z_weights, v)

        # Gating
        g = self.sigmoid(self.linear_g(m))
        g = g.view(g.shape[:-1] + (self.no_heads, -1))

        o = o * g

        # Flatten heads and project
        o = o.reshape(o.shape[:-2] + (-1,))
        o = self.linear_o(o)

        return o

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            m:    [*, N_seq, N_res, C_m] MSA embedding
            z:    [*, N_res, N_res, C_z] pair embedding
            mask: [*, N_res, N_res] pair mask

        Returns:
            [*, N_seq, N_res, C_m] updated MSA embedding
        """
        if z is None:
            return m
        # The fused kernel writes no autograd graph, so a grad-enabled call goes
        # to the reference; the benchmark and any inference caller run under
        # ``no_grad``.
        if not torch.is_grad_enabled():
            key = (m.dtype, m.shape, m.stride(), z.shape, z.stride(),
                   None if mask is None else (mask.shape, mask.stride()))
            plans = self._plans
            if key in plans:
                run = plans[key]
            else:
                try:
                    run = self._build_plan(m, z, mask)
                except Exception:  # noqa: BLE001 - a plan must never fail a call
                    run = None
                plans[key] = run
            if run is not None:
                return run(m, z, mask)
        return self._reference(m, z, mask)
