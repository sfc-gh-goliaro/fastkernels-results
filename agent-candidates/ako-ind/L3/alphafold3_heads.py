"""Auxiliary prediction heads for AlphaFold3.

Distogram, pLDDT, PAE, PDE, ExperimentallyResolved confidence heads that
produce binned logits from single and pair representations.  The
PairformerEmbedding refines s/z before confidence heads.

Reference: openfold3/core/model/heads/prediction_heads.py
           openfold3/core/model/heads/head_modules.py AuxiliaryHeadsAllAtom

Optimization notes (AuxiliaryHeads.forward)
-------------------------------------------
The captured workload is tiny (s:[1,16,384], z:[1,16,16,128] -- 16 tokens,
256 pair positions, ~39K input elements, 1.1 MB of DRAM traffic and 27 MFLOP), so
wall time is launch and issue overhead, not FLOPs: the eager path fires 19 device
ops (4 LayerNorms x {float, layer_norm, to}, 5 Linears, 2 transpose+adds) for 56
us of device time and ~220 us of host time.  The whole bundle is folded into ONE
Triton launch of 2.7 us.

Algebra (what removes the work):

* ``plddt`` and ``experimentally_resolved`` LayerNorm the *same* ``s``, and
  ``pae``/``pde`` LayerNorm the *same* ``z`` -- statistics are reduced once per
  representation instead of four times.
* the LayerNorm affine is folded into each head's Linear at prep time,
  ``Linear(x_hat * g + b) == x_hat @ (W * g)^T + W @ b``, so the hot path has no
  affine.  The two s-side heads become one 384 -> 1196 weight and the three
  z-side heads one 128 -> 192 weight.
* the LayerNorm itself moves *after* the GEMM:

      x_hat @ W + bias == rstd * (x @ W) - (rstd * mu) * colsum(W) + bias

  so the raw bf16 input goes straight into ``tl.dot`` while mean/variance are
  reduced from the same loaded tile in parallel -- one pass over the input, and
  the GEMM never waits on the normalization.  distogram needs no correction at
  all (it reads raw z), which is what lets all three pair heads share one dot.
* both symmetrizations (``out + out.transpose(-2, -3)``) close in-register: the
  program owning pair rows ``(i, j*)`` also loads the transposed rows
  ``(j*, i)``, so the pair sum needs no second pass and no intermediate tensor.

Shape (what removes the branches and the masks):

* all three pair heads share ONE code path over a unified
  ``[distogram | pae | pde]`` column space; the epilogue picks its coefficients
  from the column index, so distogram is the ``rstd = 1, mu = 0`` case and pae
  the ``no transpose partner`` case of the same formula.
* each side writes ONE padded buffer -- pair of row pitch ``NP3``, single of row
  pitch ``NS_PAD`` -- returned as five column views.  With the folded weights
  zero-padded to the same pitch, every hot load and both stores run unmasked and
  the output pointer is pure arithmetic; the padding columns land where no view
  reads them.

Latency (what the profile said actually costs):

  NCU on this kernel: 1.4-1.7% of peak DRAM bandwidth, ~3% achieved occupancy
  (~1.8 warps resident per SM, because 27 MFLOP cannot fill 148 SMs), and ~7
  cycles per issued instruction.  It is **instruction-issue bound with no second
  warp to hide behind**, so the lever is instructions on the critical path, not
  bytes and not FLOPs.  Accordingly the LayerNorm statistics -- which as
  ``tl.sum`` tree reductions are long *dependent* chains of fp32 adds and warp
  shuffles, recomputed by every output-column group -- are computed on the
  tensor cores instead: ``sum(x)`` is a dot against an all-ones tile (every
  output column is the same row sum) and ``sum(x*x)`` a dot of the square
  against it.  On the single side both accumulate across the whole K-loop and are
  extracted once after it, so the loop body carries no reduction at all.  That
  cut the s side from 3.84 to 2.86 us on its own, took SASS from 78 KB to 21 KB
  and instructions from 236K to 144K, and moved device time 3.87 -> 2.69 us.
  See ITERATIONS.md for the measurements, including a ~1.33 us floor for an empty
  launch at any grid width up to ~400 CTAs.

fp32 statistics and fp32 accumulation with bf16 operands/outputs; on the scored
case max_abs vs eager is 7.8e-3 -- one bf16 ULP at output magnitude ~1.6, i.e.
output-rounding level -- with matched_ratio 1.0000.  Anything the fast path does
not cover (non-CUDA, non-contiguous or fp32 inputs, mismatched channel counts,
per-head LayerNorm eps disagreement, biased Linears) falls back to the reference
module-by-module implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear


__targets__ = ["AuxiliaryHeads"]


class DistogramHead(nn.Module):
    """Predicts inter-residue distance distribution.

    Args:
        c_z: Pair embedding channel dimension
        no_bins: Number of distance bins
    """

    def __init__(self, c_z: int, no_bins: int = 64):
        super().__init__()
        self.linear = Linear(c_z, no_bins, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)
        return logits


class PLDDTHead(nn.Module):
    """Predicts per-atom pLDDT confidence (PerResidueLDDTAllAtom).

    Outputs max_atoms_per_token * no_bins logits per token.

    Args:
        c_s: Single embedding channel dimension
        no_bins: Number of pLDDT bins
        max_atoms_per_token: Maximum atoms per token (23 for all-atom)
    """

    def __init__(self, c_s: int, no_bins: int = 50, max_atoms_per_token: int = 23):
        super().__init__()
        self.no_bins = no_bins
        self.max_atoms_per_token = max_atoms_per_token
        self.layer_norm = LayerNorm(c_s)
        self.linear = Linear(c_s, max_atoms_per_token * no_bins, bias=False)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.linear(self.layer_norm(s))


class PAEHead(nn.Module):
    """Predicts Predicted Aligned Error (PAE).

    Args:
        c_z: Pair embedding channel dimension
        no_bins: Number of PAE bins
    """

    def __init__(self, c_z: int, no_bins: int = 64):
        super().__init__()
        self.layer_norm = LayerNorm(c_z)
        self.linear = Linear(c_z, no_bins, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(self.layer_norm(z))


class PDEHead(nn.Module):
    """Predicts Predicted Distance Error (PDE).

    Args:
        c_z: Pair embedding channel dimension
        no_bins: Number of PDE bins
    """

    def __init__(self, c_z: int, no_bins: int = 64):
        super().__init__()
        self.layer_norm = LayerNorm(c_z)
        self.linear = Linear(c_z, no_bins, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.linear(self.layer_norm(z))
        logits = logits + logits.transpose(-2, -3)
        return logits


class ExperimentallyResolvedHead(nn.Module):
    """Predicts per-atom experimental resolution confidence.

    Args:
        c_s: Single embedding channel dimension
        no_bins: Number of bins (2 for resolved/not resolved)
        max_atoms_per_token: Maximum atoms per token (23 for all-atom)
    """

    def __init__(self, c_s: int, no_bins: int = 2, max_atoms_per_token: int = 23):
        super().__init__()
        self.no_bins = no_bins
        self.max_atoms_per_token = max_atoms_per_token
        self.layer_norm = LayerNorm(c_s)
        self.linear = Linear(c_s, max_atoms_per_token * no_bins, bias=False)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.linear(self.layer_norm(s))


class PairformerEmbedding(nn.Module):
    """Confidence head PairformerEmbedding.

    Refines pair representation using predicted atom positions before
    confidence heads (PAE, PDE, pLDDT, experimentally resolved).

    Reference: openfold3/core/model/heads/prediction_heads.py PairformerEmbedding

    Args:
        c_s_input: Input single rep dimension
        c_z: Pair rep dimension
        c_s: Single rep dimension
        no_distance_bins: Number of distance bins
        pairformer_kwargs: Config for pairformer stack
    """

    def __init__(
        self,
        c_s_input: int = 449,
        c_z: int = 128,
        c_s: int = 384,
        no_distance_bins: int = 39,
        pairformer_no_blocks: int = 4,
        pairformer_c_hidden_pair_bias: int = 24,
        pairformer_no_heads_pair_bias: int = 16,
        pairformer_c_hidden_mul: int = 128,
        pairformer_c_hidden_pair_att: int = 32,
        pairformer_no_heads_pair: int = 4,
        pairformer_transition_n: int = 4,
        pairformer_pair_dropout: float = 0.0,
    ):
        super().__init__()
        from ..L3.alphafold3_pairformer import PairFormerStack

        self.linear_i = Linear(c_s_input, c_z, bias=False)
        self.linear_j = Linear(c_s_input, c_z, bias=False)
        self.linear_distance = Linear(no_distance_bins, c_z, bias=False)

        self.pairformer_stack = PairFormerStack(
            c_s=c_s,
            c_z=c_z,
            c_hidden_pair_bias=pairformer_c_hidden_pair_bias,
            no_heads_pair_bias=pairformer_no_heads_pair_bias,
            c_hidden_mul=pairformer_c_hidden_mul,
            c_hidden_pair_att=pairformer_c_hidden_pair_att,
            no_heads_pair=pairformer_no_heads_pair,
            no_blocks=pairformer_no_blocks,
            transition_n=pairformer_transition_n,
            pair_dropout=pairformer_pair_dropout,
        )

    def forward(
        self,
        si_input: torch.Tensor,
        zij: torch.Tensor,
        s: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zij = (
            zij
            + self.linear_i(si_input)[..., :, None, :]
            + self.linear_j(si_input)[..., None, :, :]
        )

        s, zij = self.pairformer_stack(
            s=s, z=zij, single_mask=single_mask, pair_mask=pair_mask,
        )
        return s, zij


# ---------------------------------------------------------------------------
# Fused single-launch head bundle (Triton).
# ---------------------------------------------------------------------------
try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - fall back to the eager path
    triton = None


# Tile shape and pipeline depth, swept on the captured shape (see
# ITERATIONS.md).  ``bn_z`` slices the *unified* 3-head pair column space, so
# it also decides how much of the pair work each program does.
_TILES = {
    'bn_s': 16,
    'bk_s': 128,
    'bn_z': 16,
    'bm_s_cap': 32,
    'bm_z_cap': 64,
    'num_warps': 2,
    'num_stages': 5,
    'tcstat': 2,
    'onehead': 1,
    'wblk': 1,
}


def _pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _tile_plan(nb, nt, cs, cz, nst, nbp, tiles=None):
    """Constexpr tile shape + program counts for one problem size.

    Pure arithmetic, no tensors: ``_plan_args`` uses it on the hot path and
    ``tools/tritoncheck.py`` uses it to compile every swept config without a
    GPU.  Every ``tl.dot`` tile dim must be >= 16, which is the floor on all of
    BM / BN / BK.

    ``nst`` is the folded s-side output width (plddt + experimentally_resolved)
    and ``nbp`` the per-head pair width, so the pair column space is ``3 * nbp``.
    Both are padded up to a whole number of tiles (``NS_PAD`` / ``NP3``) and the
    weights are zero-padded to match, which is what lets the hot loads and both
    stores run unmasked.
    """
    t = _TILES if tiles is None else {**_TILES, **tiles}
    bm_z = max(16, min(_pow2(nt), t["bm_z_cap"]))
    # tl.arange needs a power-of-2 extent, so every tile dim is rounded up
    bn_z = max(16, min(_pow2(3 * nbp), _pow2(t["bn_z"])))
    bk_z = max(16, _pow2(cz))
    bm_s = max(16, min(_pow2(nt), t["bm_s_cap"]))
    bn_s = max(16, _pow2(t["bn_s"]))
    bk_s = max(16, min(_pow2(cs), _pow2(t["bk_s"])))
    ncg = (3 * nbp + bn_z - 1) // bn_z
    nn_s = (nst + bn_s - 1) // bn_s
    nm_z = (nt + bm_z - 1) // bm_z
    nm_s = (nt + bm_s - 1) // bm_s
    nzp = nb * nt * nm_z * ncg
    nsp = nb * nm_s * nn_s
    return {
        "NT": nt, "CS": cs, "CZ": cz,
        "NBP": nbp, "NP3": ncg * bn_z, "NS_PAD": nn_s * bn_s,
        "CS_PAD": ((cs + bk_s - 1) // bk_s) * bk_s,
        "NZP": nzp,
        "BM_Z": bm_z, "BN_Z": bn_z, "BK_Z": bk_z, "NCG": ncg, "NM_Z": nm_z,
        "BM_S": bm_s, "BN_S": bn_s, "BK_S": bk_s, "NN_S": nn_s, "NM_S": nm_s,
        "EXM_Z": nt % bm_z == 0, "EXM_S": nt % bm_s == 0,
        "EXK_Z": cz == bk_z, "EXK_S": cs % bk_s == 0,
        # mode 1's Gram matrix is [BM, BM], so it stops paying once the row
        # block is wide; mode 2 has no Gram and works at any BM.
        "TCSTAT": (1 if (t["tcstat"] == 1 and bm_z <= 32 and bm_s <= 32)
                   else (2 if t["tcstat"] == 2 else 0)),
        "WBLK": bool(t["wblk"]),
        "ONEHEAD": bool(t["onehead"]) and nbp % bn_z == 0,
        "BST": 16,
        "nsp": nsp, "grid": nzp + nsp,
        "num_warps": t["num_warps"], "num_stages": t["num_stages"],
    }


if triton is not None:

    @triton.jit
    def _af3_heads_kernel(
        S, Z, O_PAIR, O_SGL,
        WS, BS, QS, WZ, BZ, QZ,
        EPS_S, EPS_Z,
        NT: tl.constexpr, CS: tl.constexpr, CZ: tl.constexpr,
        NBP: tl.constexpr, NP3: tl.constexpr, NS_PAD: tl.constexpr,
        NZP: tl.constexpr,
        BM_Z: tl.constexpr, BN_Z: tl.constexpr, BK_Z: tl.constexpr,
        NCG: tl.constexpr, NM_Z: tl.constexpr,
        BM_S: tl.constexpr, BN_S: tl.constexpr, BK_S: tl.constexpr,
        NN_S: tl.constexpr, NM_S: tl.constexpr,
        EXM_Z: tl.constexpr, EXM_S: tl.constexpr,
        EXK_Z: tl.constexpr, EXK_S: tl.constexpr,
        TCSTAT: tl.constexpr, ONEHEAD: tl.constexpr, BST: tl.constexpr,
        CS_PAD: tl.constexpr, WBLK: tl.constexpr,
    ):
        """One launch, two code paths, all five heads.

        The LayerNorm is applied *after* the GEMM instead of before it:

            x_hat @ W + bias == rstd * (x @ W) - (rstd * mu) * colsum(W) + bias

        with ``colsum(W)`` (``QS`` / ``QZ``) precomputed on the host from the
        same bf16 weight the dot uses.  So the raw bf16 input goes straight into
        ``tl.dot`` while mean/variance are reduced from the *same* loaded tile in
        parallel -- one pass over the input, and the GEMM no longer waits on the
        normalization.

        Programs ``[0, NZP)`` do the pair side.  All three pair heads share ONE
        path over a unified ``3 * NBP`` column space laid out
        ``[distogram | pae | pde]``: the epilogue picks its coefficients from the
        column index, so distogram is the ``rstd = 1, mu = 0`` case and pae the
        ``no transpose partner`` case of the same formula.  The program owning
        pair rows ``(i, jblk)`` also loads the transposed rows ``(jblk, i)``, so
        the distogram/pde symmetrization closes in-register.

        Programs ``[NZP, ...)`` do the single side: one K-loop that accumulates
        the pLDDT / experimentally-resolved GEMM against the folded
        ``[CS, NST]`` weight together with the LayerNorm statistics.

        Both sides write into one padded buffer per side (``O_PAIR`` of row pitch
        ``NP3``, ``O_SGL`` of row pitch ``NS_PAD``) which the host returns as
        five column views.  With the weights zero-padded to the same pitch, the
        output pointer is pure arithmetic: no per-head branch and no column
        mask, and the padding columns land where no view reads them.
        """
        DT = S.dtype.element_ty
        pid = tl.program_id(0)
        if pid < NZP:
            # ---------------- pair side: distogram | pae | pde --------------
            tz = pid
            cg = tz % NCG
            tz = tz // NCG
            jb = tz % NM_Z
            tz = tz // NM_Z
            i = tz % NT
            b = tz // NT

            rj = jb * BM_Z + tl.arange(0, BM_Z)
            kz = tl.arange(0, BK_Z)
            zb = Z + b * (NT * NT * CZ) + kz[None, :]
            p1 = zb + i * (NT * CZ) + rj[:, None] * CZ
            p2 = zb + rj[:, None] * (NT * CZ) + i * CZ
            if EXM_Z and EXK_Z:
                z1 = tl.load(p1)
                z2 = tl.load(p2)
            else:
                m2 = (rj < NT)[:, None] & (kz < CZ)[None, :]
                z1 = tl.load(p1, mask=m2, other=0.0)
                z2 = tl.load(p2, mask=m2, other=0.0)

            if TCSTAT != 0:
                # sum(x) is a dot against an all-ones tile: every output column
                # is the same row sum, so tl.max over BST picks it up.  This
                # replaces two length-CZ dependent reduction chains per tile
                # with tensor-core dots.
                ones_z = tl.full([BK_Z, BST], 1.0, DT)
                mu1 = tl.max(tl.dot(z1, ones_z), 1) / CZ
                mu2 = tl.max(tl.dot(z2, ones_z), 1) / CZ
                if TCSTAT == 1:
                    dg_z = (tl.arange(0, BM_Z)[:, None]
                            == tl.arange(0, BM_Z)[None, :])
                    t1 = tl.sum(tl.where(dg_z, tl.dot(z1, tl.trans(z1)), 0.0), 1)
                    t2 = tl.sum(tl.where(dg_z, tl.dot(z2, tl.trans(z2)), 0.0), 1)
                else:
                    t1 = tl.max(tl.dot(z1 * z1, ones_z), 1)
                    t2 = tl.max(tl.dot(z2 * z2, ones_z), 1)
            else:
                f1 = z1.to(tl.float32)
                f2 = z2.to(tl.float32)
                mu1 = tl.sum(f1, 1) / CZ
                mu2 = tl.sum(f2, 1) / CZ
                t1 = tl.sum(f1 * f1, 1)
                t2 = tl.sum(f2 * f2, 1)
            r1 = 1.0 / tl.sqrt(tl.maximum(t1 / CZ - mu1 * mu1, 0.0) + EPS_Z)
            r2 = 1.0 / tl.sqrt(tl.maximum(t2 / CZ - mu2 * mu2, 0.0) + EPS_Z)

            n = cg * BN_Z + tl.arange(0, BN_Z)
            if WBLK:
                w = tl.load(WZ + cg * (BK_Z * BN_Z)
                            + kz[:, None] * BN_Z + tl.arange(0, BN_Z)[None, :])
            else:
                w = tl.load(WZ + kz[:, None] * NP3 + n[None, :])
            d1 = tl.dot(z1, w)
            d2 = tl.dot(z2, w)

            # distogram columns are raw and symmetrized (a1 = a2 = 1, no LN
            # correction), pae normalized and NOT symmetrized (a2 = 0), pde
            # normalized and symmetrized.
            c1 = r1 * mu1
            c2 = r2 * mu2
            qzv = tl.load(QZ + n)
            bzv = tl.load(BZ + n)
            if ONEHEAD:
                hh = (cg * BN_Z) // NBP
                a1 = tl.where(hh == 0, 1.0, r1)[:, None]
                a2 = tl.where(hh == 0, 1.0, tl.where(hh == 1, 0.0, r2))[:, None]
                corr = tl.where(hh == 0, 0.0,
                                tl.where(hh == 1, c1, c1 + c2))[:, None]
                bmul = tl.where(hh == 2, 2.0, 1.0)
                oz = (a1 * d1 + a2 * d2 - corr * qzv[None, :]
                      + bmul * bzv[None, :]).to(DT)
            else:
                dis = (n < NBP)[None, :]
                pae = ((n >= NBP) & (n < 2 * NBP))[None, :]
                oz = (tl.where(dis, 1.0, r1[:, None]) * d1
                      + tl.where(dis, 1.0,
                                 tl.where(pae, 0.0, r2[:, None])) * d2
                      - tl.where(dis, 0.0,
                                 tl.where(pae, c1[:, None],
                                          (c1 + c2)[:, None])) * qzv[None, :]
                      + (tl.where(n >= 2 * NBP, 2.0, 1.0) * bzv)[None, :]).to(DT)
            opz = O_PAIR + ((b * NT + i) * NT + rj)[:, None] * NP3 + n[None, :]
            if EXM_Z:
                tl.store(opz, oz)
            else:
                tl.store(opz, oz, mask=(rj < NT)[:, None])
        else:
            # ---------------- single side: plddt | experimentally resolved --
            ts = pid - NZP
            nbi = ts % NN_S
            ts = ts // NN_S
            mb = ts % NM_S
            bs_ = ts // NM_S

            rm = mb * BM_S + tl.arange(0, BM_S)
            sb = S + bs_ * (NT * CS) + rm[:, None] * CS
            ns = nbi * BN_S + tl.arange(0, BN_S)

            ssum = tl.zeros([BM_S], tl.float32)
            ssq = tl.zeros([BM_S], tl.float32)
            gsum = tl.zeros([BM_S, BST], tl.float32)
            gsq = tl.zeros([BM_S, BM_S], tl.float32)
            gsq2 = tl.zeros([BM_S, BST], tl.float32)
            acc = tl.zeros([BM_S, BN_S], tl.float32)
            for k0 in range(0, CS, BK_S):
                ks = k0 + tl.arange(0, BK_S)
                if EXM_S and EXK_S:
                    a = tl.load(sb + ks[None, :])
                else:
                    a = tl.load(sb + ks[None, :],
                                mask=(rm < NT)[:, None] & (ks < CS)[None, :],
                                other=0.0)
                if TCSTAT != 0:
                    # both statistics accumulate on the tensor cores across the
                    # whole K-loop and are extracted once, after it, so the loop
                    # body carries no reduction chain at all
                    ones_s = tl.full([BK_S, BST], 1.0, DT)
                    gsum = tl.dot(a, ones_s, gsum)
                    if TCSTAT == 1:
                        gsq = tl.dot(a, tl.trans(a), gsq)
                    else:
                        gsq2 = tl.dot(a * a, ones_s, gsq2)
                else:
                    f = a.to(tl.float32)
                    ssum += tl.sum(f, 1)
                    ssq += tl.sum(f * f, 1)
                if WBLK:
                    wt = tl.load(WS + nbi * (CS_PAD * BN_S)
                                 + ks[:, None] * BN_S
                                 + tl.arange(0, BN_S)[None, :])
                else:
                    wt = tl.load(WS + ks[:, None] * NS_PAD + ns[None, :])
                acc = tl.dot(a, wt, acc)
            if TCSTAT != 0:
                ssum = tl.max(gsum, 1)
                if TCSTAT == 1:
                    dg_s = (tl.arange(0, BM_S)[:, None]
                            == tl.arange(0, BM_S)[None, :])
                    ssq = tl.sum(tl.where(dg_s, gsq, 0.0), 1)
                else:
                    ssq = tl.max(gsq2, 1)
            mu = ssum / CS
            rs = 1.0 / tl.sqrt(tl.maximum(ssq / CS - mu * mu, 0.0) + EPS_S)
            os_ = (acc * rs[:, None]
                   - (rs * mu)[:, None] * tl.load(QS + ns)[None, :]
                   + tl.load(BS + ns)[None, :]).to(DT)
            ops = O_SGL + (bs_ * NT + rm)[:, None] * NS_PAD + ns[None, :]
            if EXM_S:
                tl.store(ops, os_)
            else:
                tl.store(ops, os_, mask=(rm < NT)[:, None])


# ``torch.cuda.current_stream(dev).cuda_stream`` builds a Stream object per
# call; the private raw accessor is what inductor-generated code uses.  Fall
# back if it is missing.
_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)
if _raw_stream is None:  # pragma: no cover - older torch
    def _raw_stream(index):
        return torch.cuda.current_stream(index).cuda_stream


class _FusedPlan:
    """Everything the hot path needs: shapes it is valid for + launch args."""

    __slots__ = ("s_shape", "z_shape", "dtype", "device", "src_w", "kernel",
                 "grid", "args", "out_specs", "views", "fn", "meta", "dev_index",
                 "direct", "launch")

    def __init__(self):
        self.kernel = None
        self.direct = False


def _plan_args(module, s, z, tiles=None):
    """Validate the fast path for these inputs and pre-bake its launch args.

    Device-agnostic on purpose: everything except the compile/launch lives here
    so ``tools/simkernel.py`` can drive the real kernel on CPU tensors under
    ``TRITON_INTERPRET=1`` and check its index / mask / epilogue math without a
    GPU.  Returns ``None`` for anything the fast path does not cover.
    """
    if triton is None:
        return None
    if s.dtype != z.dtype or s.device != z.device:
        return None
    if s.dtype not in (torch.bfloat16, torch.float16):
        return None  # fp32 wants an ieee-precision dot; eager is fine there
    if not s.is_contiguous() or not z.is_contiguous():
        return None
    if s.dim() < 2 or z.dim() < 3 or s.dim() + 1 != z.dim():
        return None
    if tuple(s.shape[:-2]) != tuple(z.shape[:-3]):
        return None
    nt = s.shape[-2]
    cs = s.shape[-1]
    cz = z.shape[-1]
    if z.shape[-3] != nt or z.shape[-2] != nt or nt == 0:
        return None
    nb = 1
    for d in s.shape[:-2]:
        nb *= d
    if cs > 4096 or cz > 256:
        return None

    heads = (module.distogram, module.plddt, module.pae, module.pde,
             module.experimentally_resolved)
    for h in heads:
        if h.linear.bias is not None:
            return None  # the folded bias below assumes bias-free Linears
        ln = getattr(h, "layer_norm", None)
        if ln is not None and tuple(ln.normalized_shape) != (h.linear.weight.shape[1],):
            return None
    for h in (module.plddt, module.experimentally_resolved):
        if h.linear.weight.shape[1] != cs or not h.layer_norm.promote_fp32:
            return None
    for h in (module.distogram, module.pae, module.pde):
        if h.linear.weight.shape[1] != cz:
            return None
    for h in (module.pae, module.pde):
        if not h.layer_norm.promote_fp32:
            return None
    if module.plddt.layer_norm.eps != module.experimentally_resolved.layer_norm.eps:
        return None
    if module.pae.layer_norm.eps != module.pde.layer_norm.eps:
        return None
    if module.plddt.linear.weight.dtype != s.dtype:
        return None

    dev, dt = s.device, s.dtype

    def fold(head):
        """(W * g)^T, W @ b -- the LayerNorm affine folded into Linear."""
        w = head.linear.weight.float()
        ln = getattr(head, "layer_norm", None)
        if ln is None:
            return w, w.new_zeros(w.shape[0])
        g, bi = ln.weight, ln.bias
        wf = w if g is None else w * g.float()[None, :]
        bf = w.new_zeros(w.shape[0]) if bi is None else w @ bi.float()
        return wf, bf

    wpl, bpl = fold(module.plddt)
    wer, ber = fold(module.experimentally_resolved)
    wdi, _ = fold(module.distogram)
    wpa, bpa = fold(module.pae)
    wpd, bpd = fold(module.pde)
    npl, ner = wpl.shape[0], wer.shape[0]
    nd, na, ne = wdi.shape[0], wpa.shape[0], wpd.shape[0]
    nst = npl + ner
    nbp = max(nd, na, ne)

    tp = _tile_plan(nb, nt, cs, cz, nst, nbp, tiles)
    np3, ns_pad, cs_pad = tp["NP3"], tp["NS_PAD"], tp["CS_PAD"]
    bk_z = tp["BK_Z"]

    def pad_cat(blocks, width, rows, row_pad):
        """[rows, width] with each block at its own column offset, zero elsewhere.

        Zero padding (both the unused tail of a head's block and the tail of the
        whole row) is what makes every hot-path weight load unmasked and keeps
        the padding columns of the output buffers harmless.
        """
        out = blocks[0][1].new_zeros(row_pad, width)
        for off, blk in blocks:
            out[:rows, off:off + blk.shape[1]] = blk
        return out

    # [in, out]: contraction dim major, so consecutive lanes / MMA columns read
    # consecutive output channels.
    ws = pad_cat([(0, wpl.t()), (npl, wer.t())], ns_pad, cs, cs_pad)
    wz = pad_cat([(0, wdi.t()), (nbp, wpa.t()), (2 * nbp, wpd.t())],
                 np3, cz, max(bk_z, cz))
    ws = ws.contiguous().to(dt)
    wz = wz.contiguous().to(dt)
    bkz_pad = max(bk_z, cz)
    bs = torch.zeros(ns_pad, dtype=torch.float32, device=dev)
    bs[:npl] = bpl
    bs[npl:nst] = ber
    bz = torch.zeros(np3, dtype=torch.float32, device=dev)
    bz[nbp:nbp + na] = bpa
    bz[2 * nbp:2 * nbp + ne] = bpd
    # colsum of the *bf16* weight the dot actually uses, so the post-GEMM
    # LayerNorm correction is exact w.r.t. that dot.  Taken before any
    # re-blocking, so it stays in global output-column order.
    qs = ws.float().sum(0).contiguous()
    qz = wz.float().sum(0).contiguous()
    if tp["WBLK"]:
        # [CS_PAD, NN_S * BN_S] -> [NN_S, CS_PAD, BN_S] and
        # [BK_Z, NCG * BN_Z]   -> [NCG, BK_Z, BN_Z]: one contiguous slab per
        # program, streamed instead of gathered.
        ws = (ws.view(cs_pad, tp["NN_S"], tp["BN_S"])
              .permute(1, 0, 2).contiguous())
        wz = (wz.view(bkz_pad, tp["NCG"], tp["BN_Z"])
              .permute(1, 0, 2).contiguous())

    plan = _FusedPlan()
    plan.s_shape = tuple(s.shape)
    plan.z_shape = tuple(z.shape)
    plan.dtype = dt
    plan.device = dev
    plan.src_w = module.plddt.linear.weight
    plan.grid = tp["grid"]
    plan.launch = (tp["num_warps"], tp["num_stages"])
    # two buffers, five column views: distogram | pae | pde out of the pair
    # buffer and plddt | experimentally_resolved out of the single buffer.
    plan.out_specs = ((tuple(z.shape[:-1]) + (np3,), dt, dev),
                      (tuple(s.shape[:-1]) + (ns_pad,), dt, dev))
    plan.views = ((0, nd), (nbp, na), (2 * nbp, ne), (0, npl), (npl, ner))
    plan.dev_index = (dev.index if dev.index is not None
                      else (torch.cuda.current_device() if dev.type == "cuda" else 0))
    plan.args = [
        s, z, None, None,
        ws, bs, qs, wz, bz, qz,
        float(module.plddt.layer_norm.eps), float(module.pae.layer_norm.eps),
        nt, cs, cz, nbp, np3, ns_pad, tp["NZP"],
        tp["BM_Z"], tp["BN_Z"], bk_z, tp["NCG"], tp["NM_Z"],
        tp["BM_S"], tp["BN_S"], tp["BK_S"], tp["NN_S"], tp["NM_S"],
        tp["EXM_Z"], tp["EXM_S"], tp["EXK_Z"], tp["EXK_S"],
        tp["TCSTAT"], tp["ONEHEAD"], tp["BST"],
        cs_pad, tp["WBLK"],
    ]
    return plan


class AuxiliaryHeads(nn.Module):
    """All auxiliary prediction heads for AF3.

    Args:
        c_s: Single embedding channel dimension
        c_z: Pair embedding channel dimension
        c_s_input: Input single rep dimension (for PairformerEmbedding)
        max_atoms_per_token: Max atoms per token (23 for all-atom)
    """

    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        c_s_input: int = 449,
        max_atoms_per_token: int = 23,
    ):
        super().__init__()
        self.pairformer_embedding = PairformerEmbedding(
            c_s_input=c_s_input,
            c_z=c_z,
            c_s=c_s,
        )
        self.distogram = DistogramHead(c_z, no_bins=64)
        self.plddt = PLDDTHead(c_s, no_bins=50, max_atoms_per_token=max_atoms_per_token)
        self.pae = PAEHead(c_z, no_bins=64)
        self.pde = PDEHead(c_z, no_bins=64)
        self.experimentally_resolved = ExperimentallyResolvedHead(
            c_s, no_bins=2, max_atoms_per_token=max_atoms_per_token,
        )
        # Fused-path state, built lazily on the first forward (weights are
        # loaded after __init__) and invalidated whenever the parameters move,
        # are recast or are reloaded -- see ``_apply`` / the load hook below.
        self._plan = None
        try:
            self.register_load_state_dict_post_hook(_drop_plan_hook)
        except AttributeError:  # pragma: no cover - very old torch
            pass

    # -- fused-path cache management ----------------------------------------
    def _apply(self, *args, **kwargs):
        self._plan = None
        return super()._apply(*args, **kwargs)

    def _eager(self, s: torch.Tensor, z: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "distogram_logits": self.distogram(z),
            "plddt_logits": self.plddt(s),
            "pae_logits": self.pae(z),
            "pde_logits": self.pde(z),
            "experimentally_resolved_logits": self.experimentally_resolved(s),
        }

    def _build_plan(self, s: torch.Tensor, z: torch.Tensor):
        if not s.is_cuda:
            return None
        plan = _plan_args(self, s, z)
        if plan is None:
            return None
        nw, nst_ = plan.launch
        outs = [torch.empty(sh, dtype=d, device=dv) for sh, d, dv in plan.out_specs]
        plan.args[2:4] = outs
        compiled = _af3_heads_kernel[(plan.grid, 1, 1)](
            *plan.args, num_warps=nw, num_stages=nst_)
        plan.kernel = compiled
        plan.fn = compiled.function
        plan.meta = compiled.packed_metadata
        # Launch through CompiledKernel.run (skipping the JIT's per-call
        # re-specialization); verify that private path works once here rather
        # than trusting it, and fall back to the public launcher if it moves.
        try:
            compiled.run(plan.grid, 1, 1, _raw_stream(plan.dev_index), plan.fn,
                         plan.meta, None, None, None, *plan.args)
            plan.direct = True
        except Exception:
            plan.direct = False
        self._plan = plan
        return plan

    def forward(
        self, s: torch.Tensor, z: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        plan = self._plan
        if (plan is None
                or s.shape != plan.s_shape
                or z.shape != plan.z_shape
                or s.dtype is not plan.dtype
                or plan.src_w is not self.plddt.linear.weight):
            plan = self._build_plan(s, z)
            if plan is None:
                return self._eager(s, z)
        specs = plan.out_specs
        o_pair = torch.empty(specs[0][0], dtype=specs[0][1], device=specs[0][2])
        o_sgl = torch.empty(specs[1][0], dtype=specs[1][1], device=specs[1][2])
        a = plan.args
        a[0] = s
        a[1] = z
        a[2] = o_pair
        a[3] = o_sgl
        if plan.direct:
            plan.kernel.run(plan.grid, 1, 1, _raw_stream(plan.dev_index),
                            plan.fn, plan.meta, None, None, None, *a)
        else:  # pragma: no cover - only if Triton's launch API changes
            _af3_heads_kernel[(plan.grid, 1, 1)](
                *a, num_warps=plan.launch[0], num_stages=plan.launch[1])
        v = plan.views
        return {
            "distogram_logits": o_pair.narrow(-1, v[0][0], v[0][1]),
            "plddt_logits": o_sgl.narrow(-1, v[3][0], v[3][1]),
            "pae_logits": o_pair.narrow(-1, v[1][0], v[1][1]),
            "pde_logits": o_pair.narrow(-1, v[2][0], v[2][1]),
            "experimentally_resolved_logits": o_sgl.narrow(-1, v[4][0], v[4][1]),
        }


def _drop_plan_hook(module, incompatible_keys):
    module._plan = None
