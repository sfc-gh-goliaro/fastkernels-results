"""MSA module embedder for AlphaFold3 (Algorithm 8, lines 1-4).

Embeds MSA features and adds projected s_input.

Reference: openfold3/core/model/feature_embedders/input_embedders.py
           MSAModuleEmbedder

Every captured call is the same tiny shape -- msa [1, 8, 16, 32],
has_deletion / deletion_value / msa_mask [1, 8, 16], s_input [1, 16, 449],
c_m_feats=34, c_m=64, c_s_input=449 -- about 0.7 MFLOP and a few kB of traffic.
The reference spends one ``cat``, two GEMMs and a broadcast add on that, i.e.
4-5 kernel launches plus the Python dispatch above them, and not one of those
launches is anywhere near memory- or math-bound.  So this is a launch-overhead
problem, not a math problem, and it is attacked on three fronts:

* **one launch.**  The whole forward is a single Triton kernel that reads
  ``msa`` / ``has_deletion`` / ``deletion_value`` / ``s_input`` from their own
  pointers and writes ``m`` [1, 8, 16, 64] directly.  ``msa_feat`` is never
  materialized; ``msa_mask`` is returned as a zero-copy passthrough (the
  reference never touches it either).
* **one cheap host call per launch.**  ``JITFunction.run`` re-binds and
  re-specializes the whole argument list and rebuilds launch metadata on every
  call (~10 us at this argument count, measured); the compiled launcher is
  called directly instead (~4.5 us, of which ~3.4 us is the bare
  ``cuLaunchKernel`` floor), and ``forward`` does nothing but four dict
  lookups, a few identity compares and the output allocation.
* **a wide grid.**  One CTA per MSA row leaves the ``linear_s_input``
  projection (K=449, one 57 kB weight stream per CTA) latency-bound.  The
  weight is instead packed per *output-channel* slice, so the grid is
  (c_m/DN) x (N_msa/ROWS_PER_CTA) CTAs and each CTA reads one contiguous
  slice.
* **one cold round trip.**  The benchmark flushes L2 before every iteration, so
  the packed weight is always fetched from HBM, and that fetch is what the score
  can still see: it lands in ~2 us quanta, one per *serialized* wave of cold
  misses (byte count barely matters -- 512 kB spread over 64 CTAs measured
  +0.05 us).  Two things put every miss in the same wave.  The K blocks are
  walked in a *staggered* order, ``(i + program_id(1)) % NBLK``, so concurrent
  row-CTAs never demand the same weight line at the same instant and pile up on
  one pending miss.  And ``linear_m``'s operands are loaded *before* the
  ``linear_s_input`` K loop rather than after it, so both projections' misses are
  in flight together instead of forming two dependent waves.  Either change alone
  is worth 0.0-0.1 us; together they are worth a full quantum.  See
  ITERATIONS.md.

Numerics: the reference rounds each projection to the activation dtype
*separately* and only then adds them, so the two accumulators are rounded
before the add rather than summed in one fp32 accumulator.  Reproducing that is
what keeps the fused kernel within a ULP of the reference instead of drifting.

The staggered K order is the one place where this kernel is not bit-identical to
the reference: a CTA sums its K blocks starting from its own block, so on some
weight draws the fp32 accumulation regroups and the s_input term lands one bf16
ULP away -- and, since the row-CTAs use different starting blocks, two MSA rows
can differ by that ULP where the reference has them equal.  Measured max_abs is
2e-3 against a bf16 eps of 7.8e-3, with matched_ratio 1.0.  Restoring an exact
canonical order (standalone dots reassembled with tl.where and summed in block
order) was measured and gives the ULP back but also gives the whole quantum back
(9.09 vs 7.25 us cold), so it is not worth it: the chained accumulator is what
makes the staggered fetch issue the way it does.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.linear import Linear


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@triton.jit
def _msa_embed(MSA, HD, DV, SIN, W, OUT,
               NS: tl.constexpr, NT: tl.constexpr, D: tl.constexpr,
               DN: tl.constexpr, KM: tl.constexpr, KS: tl.constexpr,
               BK: tl.constexpr, KROWS: tl.constexpr, NPG: tl.constexpr,
               NBLK: tl.constexpr):
    """m[b, s, t, j*DN : (j+1)*DN] for NPG consecutive MSA rows.

    ``P1`` -- the ``linear_m`` projection -- is computed without ever
    materializing ``torch.cat([msa, has_deletion, deletion_value])``: columns
    0..KM-1 come straight from ``msa`` and go through ``tl.dot``, while the two
    trailing scalar features enter as two rank-1 updates against rows KM and
    KM+1 of the packed weight.  Same arithmetic, no ``cat``, and because
    KM = c_m_feats - 2 is a power of two the K range needs no masking.  Its
    operands are loaded up front, ahead of P2's K loop, so that its cold misses
    join P2's wave instead of forming a second one.

    ``P2`` -- the ``linear_s_input`` projection -- has no MSA-row axis (the
    reference broadcasts it over N_msa), so each CTA computes it once for its
    own channel slice and reuses it across its NPG rows.  Its K loop is walked
    from a per-CTA starting block, so the row-CTAs -- which all need the same
    weight slice -- are never queued behind one another's pending miss.  Only
    the ``s_input`` side needs a bound check; ``W`` is zero-padded past KS, so
    whatever lands in the masked-out columns would be multiplied by zero anyway.

    ``W`` is one contiguous [c_m/DN, KROWS, DN] buffer built once per parameter
    version: per channel slice, ``linear_m.weight.T`` stacked on
    ``linear_s_input.weight.T``, zero-padded in K to a multiple of BK so the
    inner loop needs no weight masking.  Slicing on the output channel is what
    keeps each CTA's weight read a single contiguous run.
    """
    j = tl.program_id(0)
    g = tl.program_id(1)
    ot = OUT.dtype.element_ty
    rn = tl.arange(0, DN)
    rt = tl.arange(0, NT)
    rk = tl.arange(0, KM)
    Wj = W + j * KROWS * DN
    r0 = g * NPG                       # first MSA row, flattened over batch

    # ---- P1 operands, hoisted so their misses ride along with P2's ---------
    wm = tl.load(Wj + rk[:, None] * DN + rn[None, :])
    w0 = tl.load(Wj + KM * DN + rn).to(tl.float32)
    w1 = tl.load(Wj + (KM + 1) * DN + rn).to(tl.float32)
    row0 = r0 * NT + rt
    a0 = tl.load(MSA + row0[:, None] * KM + rk[None, :])
    h0 = tl.load(HD + row0).to(tl.float32)
    d0 = tl.load(DV + row0).to(tl.float32)

    # ---- P2: linear_s_input for this channel slice, staggered start block ---
    rk2 = tl.arange(0, BK)
    sp = SIN + (r0 // NS) * (NT * KS) + rt[:, None] * KS + rk2[None, :]
    wp = Wj + (KM + 2) * DN + rk2[:, None] * DN + rn[None, :]
    sh = g % NBLK
    acc2 = tl.zeros((NT, DN), dtype=tl.float32)
    for i in tl.static_range(NBLK):
        blk = (i + sh) % NBLK
        acc2 = tl.dot(tl.load(sp + blk * BK,
                              mask=(blk * BK + rk2 < KS)[None, :], other=0.0),
                      tl.load(wp + blk * BK * DN), acc2)
    p2 = acc2.to(ot).to(tl.float32)

    # ---- P1: linear_m, one MSA row at a time -------------------------------
    for si in tl.static_range(NPG):
        row = (r0 + si) * NT + rt
        if si == 0:
            a, h, dd = a0, h0, d0
        else:
            a = tl.load(MSA + row[:, None] * KM + rk[None, :])
            h = tl.load(HD + row).to(tl.float32)
            dd = tl.load(DV + row).to(tl.float32)
        acc1 = tl.dot(a, wm)
        acc1 += h[:, None] * w0[None, :] + dd[:, None] * w1[None, :]
        tl.store(OUT + row[:, None] * D + j * DN + rn[None, :],
                 (acc1.to(ot).to(tl.float32) + p2).to(ot))


# ---------------------------------------------------------------------------
# Launch plan: built once per (shape, dtype, parameter version), then reduced
# to a closure over the compiled launcher.
# ---------------------------------------------------------------------------
# (DN, BK, ROWS_PER_CTA, num_warps), swept on B200 over DN in {16, 32, 64} x BK
# in {64..512} x rows-per-CTA in {1, 2, 4, 8} x warps in {1..16} x num_stages in
# {1..4} x staggered/in-order K x hoisted/in-place P1 operands, timed in a cold-L2
# single-launch bracket and in the benchmark's own timing loop (a warm CUDA-graph
# replay ties 25 of 72 configs and measures the wrong thing).  DN=16 is what makes
# the staggered order pay: at DN=32 the same kernel is a full quantum slower.
_CFG = (16, 256, 1, 2)


class _Plan:
    __slots__ = ("msa_shape", "s_shape", "dtype", "out_shape", "W", "grid",
                 "meta", "pm_m", "pm_s", "wm", "ws", "wmv", "wsv", "launch")

    def __init__(self, mod, msa, s_input):
        wm = mod.linear_m.weight
        ws = mod.linear_s_input.weight
        b, ns, nt, km = msa.shape
        d = mod.c_m
        ks = s_input.shape[-1]
        dn, bk, npg, warps = _CFG
        dn = dn if d % dn == 0 else 16          # d is a multiple of 16 (checked)
        while ns % npg:
            npg -= 1
        nblk = -(-ks // bk)
        krows = (km + 2) + nblk * bk
        W = torch.zeros((d // dn, krows, dn), dtype=msa.dtype, device=msa.device)
        wmt = wm.t()
        wst = ws.t()
        for j in range(d // dn):
            sl = slice(j * dn, (j + 1) * dn)
            W[j, :km + 2] = wmt[:, sl]
            W[j, km + 2:km + 2 + ks] = wst[:, sl]
        self.W = W
        self.msa_shape = tuple(msa.shape)
        self.s_shape = tuple(s_input.shape)
        self.dtype = msa.dtype
        self.out_shape = (b, ns, nt, d)
        # Cheap staleness detection: the ``_parameters`` dicts survive
        # ``Module._apply`` (``.to()`` / ``.cuda()``) and direct reassignment,
        # so an identity compare there catches a swapped Parameter, and
        # ``_version`` catches an in-place ``load_state_dict`` copy -- all
        # without going through ``nn.Module.__getattr__`` on the hot path.
        self.pm_m = mod.linear_m._parameters
        self.pm_s = mod.linear_s_input._parameters
        self.wm, self.ws = wm, ws
        self.wmv, self.wsv = wm._version, ws._version
        self.grid = (d // dn, (b * ns) // npg)
        self.meta = dict(NS=ns, NT=nt, D=d, DN=dn, KM=km, KS=ks, BK=bk,
                         KROWS=krows, NPG=npg, NBLK=nblk, num_warps=warps,
                         num_stages=1)
        self.launch = self._first

    def _first(self, msa, hd, dv, s_input, out):
        """Compile through the ordinary path, then swap in a direct launcher."""
        m = self.meta
        grid = self.grid
        W = self.W
        k = _msa_embed[grid](msa, hd, dv, s_input, W, out, **m)
        tail = (m["NS"], m["NT"], m["D"], m["DN"], m["KM"], m["KS"], m["BK"],
                m["KROWS"], m["NPG"], m["NBLK"])

        def jit_launch(msa, hd, dv, s_input, out):
            _msa_embed[grid](msa, hd, dv, s_input, W, out, **m)

        ldr = k.run
        if ldr.global_scratch_size or ldr.profile_scratch_size:
            self.launch = jit_launch          # scratch alloc needs the wrapper
            return
        c_launch = ldr.launch
        coop, pdl = ldr.launch_cooperative_grid, ldr.launch_pdl
        fn, pmeta = k.function, k.packed_metadata
        gx, gy = grid
        raw_stream = torch._C._cuda_getCurrentRawStream
        dev = W.device.index
        wp = W.data_ptr()

        def launch(msa, hd, dv, s_input, out):
            # Handing the launcher raw integer addresses instead of tensors
            # skips six ``getattr(t, "data_ptr")`` + call round trips inside the
            # C launcher (5.9 -> 4.0 us measured).  Triton compiled this kernel
            # assuming its pointer arguments are 16-byte aligned, and the direct
            # call bypasses the per-call divisibility check that would otherwise
            # re-specialize, so verify that here and defer to the ordinary path
            # if a caller ever hands us an unaligned view.
            a = msa.data_ptr()
            b = hd.data_ptr()
            c = dv.data_ptr()
            d = s_input.data_ptr()
            e = out.data_ptr()
            if (a | b | c | d | e) & 15:
                jit_launch(msa, hd, dv, s_input, out)
                return
            c_launch(gx, gy, 1, raw_stream(dev), fn, coop, pdl, None, None,
                     pmeta, None, None, None, a, b, c, d, wp, e, *tail)

        # The direct call must reproduce the ordinary launch bit-for-bit before
        # we start using it.
        probe = torch.empty_like(out)
        launch(msa, hd, dv, s_input, probe)
        self.launch = launch if torch.equal(probe, out) else jit_launch


class MSAModuleEmbedder(nn.Module):
    """AF3 Algorithm 8, lines 1-4: MSA feature embedding.

    Args:
        c_m_feats: MSA input features channel dimension (34 = 32 msa + has_deletion + deletion_value)
        c_m: MSA channel dimension
        c_s_input: Single (s_input) channel dimension
    """

    def __init__(
        self,
        c_m_feats: int = 34,
        c_m: int = 64,
        c_s_input: int = 449,
    ):
        super().__init__()
        self.linear_m = Linear(c_m_feats, c_m, bias=False)
        self.linear_s_input = Linear(c_s_input, c_m, bias=False)
        self.c_m = c_m
        self._plan = None

    def _ref(self, batch, s_input):
        """The reference composition, for anything the kernel declines."""
        msa_feat = torch.cat(
            [
                batch["msa"],
                batch["has_deletion"].unsqueeze(-1),
                batch["deletion_value"].unsqueeze(-1),
            ],
            dim=-1,
        )
        m = self.linear_m(msa_feat)
        m = m + self.linear_s_input(s_input).unsqueeze(-3)
        return m, batch["msa_mask"]

    def _build(self, batch, s_input):
        msa = batch["msa"]
        hd = batch["has_deletion"]
        dv = batch["deletion_value"]
        wm = self.linear_m.weight
        ws = self.linear_s_input.weight
        d = self.c_m
        km = msa.shape[-1] if msa.ndim == 4 else 0
        if (msa.ndim != 4 or msa.device.type != "cuda"
                or msa.dtype not in (torch.float16, torch.bfloat16)
                or km < 16 or triton.next_power_of_2(km) != km
                or d % 16 or d < 16
                or triton.next_power_of_2(msa.shape[-2]) != msa.shape[-2]
                or msa.shape[-2] < 16
                or hd.shape != msa.shape[:-1] or dv.shape != msa.shape[:-1]
                or s_input.ndim != 3 or s_input.shape[0] != msa.shape[0]
                or s_input.shape[1] != msa.shape[-2]
                or wm.shape != (d, km + 2) or ws.shape != (d, s_input.shape[-1])
                or wm.dtype is not msa.dtype or ws.dtype is not msa.dtype
                or not (msa.is_contiguous() and hd.is_contiguous()
                        and dv.is_contiguous() and s_input.is_contiguous())):
            return None
        return _Plan(self, msa, s_input)

    def forward(
        self,
        batch: dict,
        s_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch: needs msa [*, N_msa, N_token, 32],
                   has_deletion [*, N_msa, N_token],
                   deletion_value [*, N_msa, N_token],
                   msa_mask [*, N_seq, N_token]
            s_input: [*, N_token, c_s_input]

        Returns:
            m: [*, N_seq, N_token, c_m]
            msa_mask: [*, N_seq, N_token]
        """
        p = self._plan
        msa = batch["msa"]
        hd = batch["has_deletion"]
        dv = batch["deletion_value"]
        if (p is None
                or msa.shape != p.msa_shape or s_input.shape != p.s_shape
                or msa.dtype is not p.dtype
                or p.pm_m["weight"] is not p.wm or p.wm._version != p.wmv
                or p.pm_s["weight"] is not p.ws or p.ws._version != p.wsv
                or not (msa.is_contiguous() and hd.is_contiguous()
                        and dv.is_contiguous() and s_input.is_contiguous())):
            p = self._build(batch, s_input)
            self._plan = p
            if p is None:
                return self._ref(batch, s_input)
        out = msa.new_empty(p.out_shape)
        p.launch(msa, hd, dv, s_input, out)
        return out, batch["msa_mask"]
