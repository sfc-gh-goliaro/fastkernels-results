"""Diffusion transformer for AlphaFold3.

24-block transformer used inside the diffusion module. Each block:
AttentionPairBias + ConditionedTransitionBlock (AdaLN-Zero).

Reference: openfold3/core/model/layers/diffusion_transformer.py

Optimization notes
------------------
At the captured shapes this operator is *entirely* CPU-issue bound: the 24-block
token stack pushes N=16 tokens through ~137 aten ops per block (~3300 per call)
and the GPU finishes long before Python is done issuing them.  What matters is
therefore the number and cost of operations issued per call, not FLOPs -- the
whole token stack reads ~312 MB of weights, i.e. tens of microseconds of B200
bandwidth.

``s``, ``z`` and ``mask`` are passed *unchanged* to every block, so everything
derived from them alone is loop-invariant.  All of it is hoisted out of the loop
and batched across blocks:

* the pair bias (``layer_norm_z`` + ``linear_z`` + permute) becomes one
  normalization of ``z`` plus one stacked GEMM producing every block's
  ``[*, H, Q, K]`` bias, with the (also loop-invariant, and block-independent)
  ``mask_bias`` folded in by a single broadcast add.  In the self-attention
  stack each block has its own ``layer_norm_z``; the LayerNorm *statistics* are
  block-independent and the norm is weight-only, so each block's weight folds
  into its ``linear_z`` weight and one normalization serves all of them.
* every AdaLN shares one ``layer_norm_s(s)`` reduction, and its gate and
  ``linear_s`` shift -- plus ``linear_ada_out(s)`` and the transition's output
  gate, which are functions of raw ``s`` -- are computed for all blocks and all
  AdaLN sites in two stacked GEMMs before the loop.  ``layer_norm_s`` is also
  weight-only, so it folds into the consumer weights the same way.
* in the cross-attention stack the block gather indices, the block mask and the
  gathered/blocked ``s`` are computed once for the whole stack instead of twice
  per block.

What remains per block is collapsed into three fused CUDA kernels and four
GEMMs (``kernel.cu``), and the block loop itself runs in C++ (``run_self`` /
``run_cross``) so the whole stack costs one Python/pybind crossing per call and
every per-block slice -- each block's gate / scale / shift columns of the
stacked conditioning GEMMs, its bias plane, its weights -- becomes pointer
arithmetic off a small int64 config block built once per shape.

On top of that the *entire call* is captured in a CUDA graph, keyed on the input
signature, so a repeat call is one graph launch (~2 us of host time regardless of
how many kernels it contains) instead of ~200-750 dispatches.  That is what
finally removes the host side: the atom stack was spending 0.79 ms issuing 0.39
ms of GPU work.  Nothing in the restructured path depends on a tensor *value* on
the host -- the block-index arithmetic keeps its ``mask.sum()`` comparison on
device -- so the captured region is the same op sequence in the same order, and
``_capture`` refuses to install a graph whose replay is not bit-identical to an
eager call.  The bench hands a different ``data_ptr`` to every iteration, so the
four inputs are staged into static buffers and the captured output is copied out
before it can be overwritten.

The kernels:

* ``res_adaln`` -- the previous block's residual add, the affine-free LayerNorm
  (fp32 reduction) and the precomputed scale/shift, in one pass over ``a``.
  Folding the residual into the *next* AdaLN is what gets the block down to one
  read-modify-write of the residual stream.
* ``gather_adaln`` -- the cross stack's key side: gather the shared normalized
  ``a`` and apply the key AdaLN's own scale/shift.
* ``attn`` -- QK^T, the pair bias, the softmax, PV and the output gate's
  sigmoid and multiply, emitted directly in ``[row, head*d]`` layout.
* the GEMM fuses q/k/v/gate into one projection (with the ``1/sqrt(c_hidden)``
  query scale folded into its weight) and ``linear_a``/``linear_b`` plus the
  SiLU into one SwiGLU call.  It reads the weight in its natural ``nn.Linear``
  ``[N, K]`` layout, one warp per CTA, with the k-step prefetch depth pinned by
  ``__launch_bounds__`` -- see kernel.cu, where each of those is worth 1.15-2x.
* ``lnz`` -- the pair representation's LayerNorm, which ``F.layer_norm`` runs at
  48 GB/s on the atom stack's 49152 rows of 16.
* ``blkidx`` / ``maskbias`` -- the atom stack's key-block gather indices, block
  mask and mask bias, reproducing the reference's deliberately-bf16 index
  arithmetic (see kernel.cu) in place of ~25 tiny eager ops.

Every kernel keeps the reference's rounding sequence -- fp32 LayerNorm
statistics, and a bf16 round exactly where the eager reference materializes a
bf16 tensor.  That is not cosmetic: with the rounding order changed instead,
one block already lands 1.8% of elements outside the 1%-relative tolerance and
24 blocks of accumulation take it to 57%.  The only place the order is *not*
matched is the hand-written GEMM's K accumulation, which cannot match cuBLAS's
by construction; on the 24-block stack with re-randomized N(0, 0.05) weights
that leaves ~4% of elements outside a 1% relative tolerance -- against ~47% for
an fp32 evaluation of the same reference, i.e. well inside the operator's own
conditioning (``prof/chaos.py``).

All stacked/folded weights, scratch buffers, the config block and the captured
graphs are built lazily on the first forward and dropped whenever the parameters
are replaced or moved (``_apply``, ``_load_from_state_dict``) -- a graph holds
raw pointers to the folded weights, so it cannot outlive them.  If the extension
cannot be built
the same restructuring runs on eager ops -- that path is bit-exact against the
reference -- and anything the fast path does not cover (no AdaLN,
``_mask_trans=False``, unexpected shapes, reduction dims that are not multiples
of 16, an attention tile too large for shared memory) falls back to the
reference block loop.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L2.alphafold3_attention_pair_bias import AttentionPairBias, CrossAttentionPairBias
from ..L2.alphafold3_swiglu_transition import ConditionedTransitionBlock


__targets__ = ["DiffusionTransformer"]

_EXT = None
_EXT_TRIED = False

# Whole-call CUDA-graph replay (kill switch for A/B measurement only).
_GRAPHS = os.environ.get("AF3_DT_GRAPH", "1") != "0"
_GRAPH_WARMUP = 3


def _ext():
    """Compile / return the fused-kernel extension (None if unavailable).

    Built on first use, which is a correctness forward -- never inside a timed
    region.  Any build failure falls back to the eager fast path, which is a
    correct (just slower) implementation of the same restructuring.
    """
    global _EXT, _EXT_TRIED
    if _EXT_TRIED:
        return _EXT
    _EXT_TRIED = True
    try:
        import os
        from torch.utils.cpp_extension import load
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel.cu")
        if not os.path.isfile(src):
            return None
        # Build only for the GPU actually present.  This *overrides* any
        # inherited TORCH_CUDA_ARCH_LIST: the ambient one here spans sm_75
        # upwards, and the bf16 `mma.m16n8k16` in the GEMM needs sm_80+, so a
        # wide list fails ptxas outright (and costs minutes of nvcc time).
        prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
        cc = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{cc[0]}.{cc[1]}"
        try:
            _EXT = load(name=f"af3_diff_transformer_fused_sm{cc[0]}{cc[1]}",
                        sources=[src], extra_cuda_cflags=["-O3"], verbose=False)
        finally:
            if prev is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = prev
    except Exception:
        _EXT = None
    return _EXT


# Shared-memory budget of the fused attention kernel (q, k, v tiles in bf16
# plus the fp32 score tile).  Blackwell allows ~227 KB of dynamic shared memory
# per CTA; stay well inside it and fall back to the eager loop otherwise.
_SMEM_MAX = 160 * 1024
_SMEM_PAD = 2          # must match SMEM_PAD in kernel.cu


def _attn_smem(q: int, k: int, d: int) -> int:
    """Shared memory of the fused attention kernel, at the *largest* query tile
    it may pick (the launcher only ever splits Q further, which shrinks this)."""
    return 2 * ((q + 2 * k) * (d + _SMEM_PAD)) + 4 * q * k


def _lnz_ok(rows: int, c: int) -> bool:
    """Is the one-thread-per-row fused LayerNorm the right tool here?

    It wins exactly when torch loses: many short rows.  With few long rows
    (the self stack normalizes 256 rows of 128) one thread per row leaves a
    single CTA and measures 10x *worse* than ``F.layer_norm``.
    """
    return c <= 64 and rows >= 4096


def _get_block_key_indices(atom_mask, n_query: int, n_key: int):
    """Key-block gather indices -- verbatim from the reference sequence-local
    atom attention helper.  Reproduced here (rather than imported and called
    once per block) because it is loop-invariant: it depends only on ``mask``.
    The bf16 arithmetic on ``n_real`` is load-bearing (indices past 256 are
    not representable in bf16 and round to even), so it is kept exactly."""
    batch_dims = atom_mask.shape[:-1]
    n_atom = atom_mask.shape[-1]
    num_blocks = math.ceil(n_atom / n_query)
    device = atom_mask.device
    offset = n_query // 2

    subset_centers = offset + torch.arange(num_blocks, device=device) * n_query
    subset_centers = subset_centers.reshape(*(1,) * len(batch_dims), num_blocks)
    subset_centers = subset_centers.expand(*batch_dims, num_blocks)

    n_real = atom_mask.sum(dim=-1, keepdim=True).expand(*batch_dims, num_blocks)

    initial = (
        subset_centers.unsqueeze(-1)
        + torch.arange(-n_key // 2, n_key // 2, device=device)
    ).int()

    underflow = torch.relu(-initial[..., 0])
    overflow = torch.relu(initial[..., -1] - (n_real - 1))
    total_shift = torch.where(underflow > 0, underflow, -overflow)
    final = initial + total_shift.unsqueeze(-1)

    n_real_exp = n_real.unsqueeze(-1)
    invalid = (final < 0) | (final >= n_real_exp)
    safe = torch.clamp(final, torch.zeros_like(n_real_exp), (n_real_exp - 1).clamp(min=0))

    return safe.long(), invalid


class DiffusionTransformerBlock(nn.Module):
    """AF3 Algorithm 23: Diffusion transformer block.

    Args:
        c_a: Token activation channel dimension
        c_s: Single activation channel dimension
        c_z: Pair activation channel dimension
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        n_transition: Transition layer scale
        use_ada_layer_norm: Whether to use AdaLN-Zero
        inf: Large masking constant
    """

    def __init__(
        self,
        c_a: int,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        n_transition: int,
        use_ada_layer_norm: bool = True,
        n_query: int | None = None,
        n_key: int | None = None,
        inf: float = 1e9,
    ):
        super().__init__()
        self.use_cross_attention = n_query is not None

        if not self.use_cross_attention:
            self.attention_pair_bias = AttentionPairBias(
                c_q=c_a, c_k=c_a, c_v=c_a,
                c_s=c_s, c_z=c_z,
                c_hidden=c_hidden,
                no_heads=no_heads,
                use_ada_layer_norm=use_ada_layer_norm,
                gating=True,
                inf=inf,
            )
        else:
            self.attention_pair_bias = CrossAttentionPairBias(
                c_q=c_a, c_k=c_a, c_v=c_a,
                c_s=c_s, c_z=c_z,
                c_hidden=c_hidden,
                no_heads=no_heads,
                use_ada_layer_norm=use_ada_layer_norm,
                n_query=n_query,
                n_key=n_key,
                gating=True,
                inf=inf,
            )

        self.conditioned_transition = ConditionedTransitionBlock(
            c_a=c_a, c_s=c_s, n=n_transition,
        )

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        use_high_precision_attention: bool = False,
        _mask_trans: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            a:    [*, N, C_token] token-level embedding
            s:    [*, N, C_s] single embedding
            z:    [*, N, N, C_z] pair embedding
            mask: [*, N] mask

        Returns:
            [*, N, C_token] updated token embedding
        """
        a = a + self.attention_pair_bias(a=a, z=z, s=s, mask=mask)

        trans_mask = mask if _mask_trans else None
        a = a + self.conditioned_transition(a=a, s=s, mask=trans_mask)

        return a


class _Plan:
    """Block-invariant stacked / folded weights and scratch, built once.

    Everything here is a function of the parameters only, so it survives across
    calls; it is dropped whenever the parameters are replaced or moved (see
    ``DiffusionTransformer._apply`` / ``_load_from_state_dict``).
    """

    __slots__ = (
        "c_a", "c_s", "c_z", "n_heads", "c_hidden", "c_hid_all", "c_ff", "eps",
        "W1", "b1", "W1k", "b1k", "W2", "b2", "Wz",
        "Wqkvg", "bqkvg", "Wkv", "Wo", "Wab", "Wout",
        "ext", "empty", "rt",
    )


class _Graph:
    """One captured whole-call graph plus the buffers its launches point at."""

    __slots__ = ("graph", "dst", "flat", "out")


class DiffusionTransformer(nn.Module):
    """AF3 Algorithm 23: Diffusion transformer stack.

    Args:
        c_a: Token activation channel dimension
        c_s: Single activation channel dimension
        c_z: Pair activation channel dimension
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        no_blocks: Number of transformer blocks
        n_transition: Transition layer scale
        use_ada_layer_norm: Whether to use AdaLN-Zero
        inf: Large masking constant
    """

    def __init__(
        self,
        c_a: int,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_blocks: int,
        n_transition: int,
        use_ada_layer_norm: bool = True,
        n_query: int | None = None,
        n_key: int | None = None,
        inf: float = 1e9,
        blocks_per_ckpt: int | None = None,
        **kwargs,
    ):
        super().__init__()
        from ..L1.layer_norm import LayerNorm

        self.use_cross_attention = n_query is not None
        if self.use_cross_attention:
            self.layer_norm_z = LayerNorm(c_z, create_offset=False)

        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(
                c_a=c_a, c_s=c_s, c_z=c_z,
                c_hidden=c_hidden, no_heads=no_heads,
                n_transition=n_transition,
                use_ada_layer_norm=use_ada_layer_norm,
                n_query=n_query,
                n_key=n_key,
                inf=inf,
            )
            for _ in range(no_blocks)
        ])

        # Fast-path geometry (plain ints, not module state).
        self._n_query = n_query
        self._n_key = n_key
        self._inf = inf
        self._use_ada = use_ada_layer_norm
        self._plan: _Plan | None = None
        # Per-signature captured graphs, and the signatures capture gave up on.
        self._graphs: dict = {}
        self._nocap: set = set()

    # -- cache lifetime -----------------------------------------------------
    # The stacked / folded weights are derived from the parameters, so they must
    # be dropped whenever a parameter is replaced (``load_state_dict``) or moved
    # / re-typed (``.to()``, ``.cuda()``, ``.float()`` all route through
    # ``_apply``).  Both hooks fire before any forward, which leaves the
    # per-call cost at one ``is None`` test rather than a per-parameter check.
    # A captured graph holds *raw pointers* to the folded weights and the
    # scratch, so it dies with the plan that owns them.
    def _drop_caches(self):
        self._plan = None
        self._graphs = {}
        self._nocap = set()

    def _apply(self, *args, **kwargs):
        self._drop_caches()
        return super()._apply(*args, **kwargs)

    def _load_from_state_dict(self, *args, **kwargs):
        self._drop_caches()
        return super()._load_from_state_dict(*args, **kwargs)

    # -- plan ---------------------------------------------------------------
    def _build_plan(self) -> _Plan:
        p = _Plan()
        blocks = self.blocks
        apb0 = blocks[0].attention_pair_bias
        mha0 = apb0.mha
        p.c_a = apb0.c_q
        p.c_s = apb0.c_s
        p.c_z = apb0.c_z
        p.n_heads = mha0.no_heads
        p.c_hidden = mha0.c_hidden
        p.c_hid_all = mha0.c_hidden * mha0.no_heads
        p.c_ff = blocks[0].conditioned_transition.linear_out.weight.shape[1]
        p.eps = blocks[0].conditioned_transition.layer_norm.layer_norm_a.eps
        scale = 1.0 / math.sqrt(p.c_hidden)

        cross = self.use_cross_attention
        ada_main = []   # AdaLN sites fed by layer_norm_s(s) over the full rows
        ada_key = []    # cross only: key-side AdaLN, fed by the gathered s
        gates_raw = []  # (linear_ada_out, transition linear_g), fed by raw s
        zws = []
        Wqkvg, bqkvg, Wkv, Wo, Wab, Wout = [], [], [], [], [], []

        # The per-block projection weights stay in their natural ``nn.Linear``
        # [N, K] layout: k contiguous is what makes the mma B fragment two
        # aligned 32-bit loads per k-step instead of four 16-bit ones (see
        # kernel.cu), and it is also exactly what ``F.linear`` -- which is what
        # the reference itself calls -- wants on the eager fallback path.  Only
        # the conditioning GEMMs below stay [K, N], because those go to cuBLAS.
        for blk in blocks:
            apb = blk.attention_pair_bias
            ct = blk.conditioned_transition
            mha = apb.mha
            if cross:
                ada_main.append((apb.layer_norm_a_q, ct.layer_norm))
                ada_key.append((apb.layer_norm_a_k,))
                zws.append(apb.linear_z.weight)
            else:
                ada_main.append((apb.layer_norm_a, ct.layer_norm))
                # layer_norm_z is weight-only -> folds into linear_z.
                zws.append(apb.linear_z.weight * apb.layer_norm_z.weight.unsqueeze(0))
            gates_raw.append((apb.linear_ada_out, ct.linear_g))

            # The 1/sqrt(c_hidden) query scale folds into the query weight and
            # bias; it only feeds the softmax, which the mask bias dominates.
            wq = mha.linear_q.weight * scale
            bq = mha.linear_q.bias * scale
            zero = bq.new_zeros(p.c_hid_all)
            if cross:
                Wqkvg.append(torch.cat([wq, mha.linear_g.weight], 0))
                bqkvg.append(torch.cat([bq, zero], 0).contiguous())
                Wkv.append(torch.cat([mha.linear_k.weight, mha.linear_v.weight], 0))
            else:
                Wqkvg.append(torch.cat([wq, mha.linear_k.weight, mha.linear_v.weight,
                                        mha.linear_g.weight], 0))
                bqkvg.append(torch.cat([bq, zero, zero, zero], 0).contiguous())
            # `linear_o` / `linear_out` need no restacking at all, so they are
            # used in place rather than copied (85 MB per token stack).
            Wo.append(mha.linear_o.weight.detach())
            Wab.append(torch.cat([ct.swiglu.linear_a.weight,
                                  ct.swiglu.linear_b.weight], 0))
            Wout.append(ct.linear_out.weight.detach())

        p.Wqkvg, p.bqkvg, p.Wkv = Wqkvg, bqkvg, Wkv
        p.Wo, p.Wab, p.Wout = Wo, Wab, Wout
        p.Wz = torch.cat(zws, 0).t().contiguous()

        p.W1, p.b1 = self._stack_adaln(ada_main, p.c_a)
        p.W1k, p.b1k = self._stack_adaln(ada_key, p.c_a) if cross else (None, None)

        # Gates fed by raw s: linear_ada_out for every block, then the
        # transition output gate for every block.
        gw = [row[0].weight for row in gates_raw] + [row[1].weight for row in gates_raw]
        gb = [row[0].bias for row in gates_raw] + [row[1].bias for row in gates_raw]
        p.W2 = torch.cat(gw, 0).t().contiguous()
        p.b2 = torch.cat(gb, 0).contiguous()

        p.ext = _ext()
        # The mma GEMM steps K in 16 and loads A fragments as aligned 32-bit
        # pairs; every reduction dim here (c_a, c_hidden*no_heads, n*c_a) is a
        # multiple of 16 at the captured configs, but guard rather than assume.
        # RES_MAX * BLK in kernel.cu bounds the AdaLN row width held in
        # registers; c_a is 768 / 128 at the captured configs.
        if any(d % 16 for d in (p.c_a, p.c_hid_all, p.c_ff)) or p.c_a > 8 * 256:
            p.ext = None
        p.empty = p.b2.new_empty(0)
        p.rt = {}
        return p

    @staticmethod
    def _stack_adaln(sites, c_a: int):
        """One stacked GEMM for a set of AdaLN sites sharing ``layer_norm_s(s)``.

        ``sites`` is a per-block tuple of AdaLN modules.  ``layer_norm_s`` is
        weight-only, so its weight folds into ``linear_g`` / ``linear_s`` and a
        single affine-free normalization of ``s`` feeds every block.  Output
        columns are laid out
        ``[g(site0, all blocks) | g(site1, ...) | shift(site0, ...) | ...]`` so
        one in-place sigmoid covers every gate and the shift half needs no
        post-processing at all.
        """
        g_w, g_b, s_w = [], [], []
        for site in range(len(sites[0])):
            for row in sites:
                ada = row[site]
                fold = ada.layer_norm_s.weight.unsqueeze(0)
                g_w.append(ada.linear_g.weight * fold)
                g_b.append(ada.linear_g.bias)
                s_w.append(ada.linear_s.weight * fold)
        zeros = g_b[0].new_zeros(len(s_w) * c_a)
        return (torch.cat(g_w + s_w, 0).t().contiguous(),
                torch.cat(g_b + [zeros], 0).contiguous())

    def _bufs(self, plan, rows, keys, zrows, q_len, k_len, nblk):
        """Persistent per-shape scratch for the fused path.

        Every buffer is written by a kernel before it is read and consumed
        inside the block that produces it, so one set serves all blocks.  The
        q/k/v/gate column views of the fused projection are cut once here
        instead of once per block.
        """
        rt = plan.rt.get(rows)
        if rt is not None:
            return rt
        c_a, c_hid, c_ff = plan.c_a, plan.c_hid_all, plan.c_ff
        new = plan.b2.new_empty
        rt = {
            "ax": new((rows, c_a)), "tx": new((rows, c_a)),
            "og": new((rows, c_hid)), "ao": new((rows, c_a)),
            "act": new((rows, c_ff)), "o2": new((rows, c_a)),
        }
        rt["o2"].zero_()          # padded tail rows must stay zero, see below
        # Normalized z, for the fused LayerNorm (see kernel.cu: k_lnz).  Only
        # worth it when there are many short rows; otherwise torch is faster and
        # this buffer is not allocated.
        rt["zn"] = new((zrows, plan.c_z)) if _lnz_ok(zrows, plan.c_z) else None
        # Self stack only: the contiguous [block, H, Q, K] bias plane.  An eager
        # add inherits its operand's stride order, and the operand here is a
        # permuted view of the stacked GEMM output, so the sum has to be written
        # into a contiguous buffer with ``out=`` for the attention kernel.
        # Cross stack: the fused block-index / mask-bias preamble's outputs.
        # ``n_key`` must be even for ``-n_key // 2`` (a Python floor division in
        # the reference) to equal ``-(n_key / 2)`` in the kernel.
        if keys is not None and k_len % 2 == 0:
            rt["idx"] = plan.b2.new_empty((keys,), dtype=torch.int64)
            rt["valid"] = new((keys,))
            rt["mk"] = new((keys,))
            rt["vcol"] = rt["valid"].view(keys, 1)
            rt["mbias"] = new((nblk, q_len, k_len))
        else:
            rt["idx"] = None
        if keys is None:
            qkvg = new((rows, 4 * c_hid))
            rt["qkvg"] = qkvg
            for i, nm in enumerate(("q", "k", "v", "g")):
                rt[nm] = qkvg.narrow(1, i * c_hid, c_hid)
            rt["ld"] = 4 * c_hid
        else:
            qg, kv = new((rows, 2 * c_hid)), new((keys, 2 * c_hid))
            rt["qg"], rt["kv"] = qg, kv
            rt["q"] = qg.narrow(1, 0, c_hid)
            rt["g"] = qg.narrow(1, c_hid, c_hid)
            rt["k"] = kv.narrow(1, 0, c_hid)
            rt["v"] = kv.narrow(1, c_hid, c_hid)
            rt["ld"] = 2 * c_hid
            rt["nrm"] = new((rows, c_a))
            rt["ak"] = new((keys, c_a))
        # Geometry + stable pointers for the whole-stack driver (see kernel.cu:
        # run_self / run_cross).  Built once per shape; the driver reads it on
        # the host, so the per-call Python cost is a single extension call.
        nb = len(self.blocks)
        cfg = [nb, rows, keys or 0, c_a, c_hid, c_ff, plan.n_heads, plan.c_hidden,
               q_len, k_len, nblk, rt["ld"],
               rt["ax"].data_ptr(), rt["tx"].data_ptr(),
               rt["qkvg" if keys is None else "qg"].data_ptr(),
               0 if keys is None else rt["kv"].data_ptr(),
               rt["og"].data_ptr(), rt["ao"].data_ptr(), rt["act"].data_ptr(),
               rt["o2"].data_ptr(),
               0 if keys is None else rt["nrm"].data_ptr(),
               0 if keys is None else rt["ak"].data_ptr()]
        for b in range(nb):
            cfg += [plan.Wqkvg[b].data_ptr(), plan.bqkvg[b].data_ptr(),
                    plan.Wkv[b].data_ptr() if plan.Wkv else 0,
                    plan.Wo[b].data_ptr(), plan.Wab[b].data_ptr(),
                    plan.Wout[b].data_ptr()]
        rt["cfg"] = torch.tensor(cfg, dtype=torch.int64)
        plan.rt[rows] = rt
        return rt

    # -- reference fallback -------------------------------------------------
    def _forward_ref(self, a, s, z, mask):
        if self.use_cross_attention:
            z = self.layer_norm_z(z)
        for block in self.blocks:
            a = block(a=a, s=s, z=z, mask=mask)
        return a

    @staticmethod
    def _split_cond(g, shift, n_sites, nb, c_a):
        """Stacked [(rows, n_sites*nb*c_a)] x2 -> per-(site, block) views."""
        gv = g.view(g.shape[0], n_sites, nb, c_a)
        sv = shift.view(shift.shape[0], n_sites, nb, c_a)
        return ([gv.select(1, i).unbind(1) for i in range(n_sites)],
                [sv.select(1, i).unbind(1) for i in range(n_sites)])

    # -- forward ------------------------------------------------------------
    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        use_high_precision_attention: bool = False,
        _mask_trans: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            a:    [*, N, C_token] token-level embedding
            s:    [*, N, C_s] single embedding
            z:    [*, N, N, C_z] pair embedding
            mask: [*, N] mask

        Returns:
            [*, N, C_token] updated token embedding
        """
        plan = self._plan
        if plan is None:
            plan = self._plan = self._build_plan()

        # Whole-call graph replay.  What is left after the restructuring is a
        # fixed sequence of ~180 (cross) to ~750 (token) launches that depends
        # only on the *shapes*, so one capture per signature turns the entire
        # host side -- the conditioning GEMMs, the block-mask / index preamble
        # and the block loop -- into a single ~2 us replay.
        key = (a.shape, s.shape, z.shape, None if mask is None else mask.shape,
               a.dtype, s.dtype, z.dtype, a.device, _mask_trans)
        ent = self._graphs.get(key)
        if ent is not None:
            ins = (a, s, z) if mask is None else (a, s, z, mask)
            # The bench hands a different ``data_ptr`` to every iteration and a
            # capture bakes in addresses, so the inputs are staged into the
            # buffers the graph actually reads.
            if ent.flat is not None:
                torch.cat([t.reshape(-1) for t in ins], out=ent.flat)
            else:
                torch._foreach_copy_(ent.dst, list(ins))
            ent.graph.replay()
            # The captured output lives in the graph's private pool and the next
            # replay overwrites it, so the caller gets its own copy.
            return ent.out.clone()

        if not self._restructured_ok(a, s, z, mask, plan, _mask_trans):
            return self._forward_ref(a, s, z, mask)
        out = self._forward_fast(a, s, z, mask, plan)
        self._capture(key, a, s, z, mask, plan)
        return out

    # -- fast path ----------------------------------------------------------
    def _restructured_ok(self, a, s, z, mask, plan, _mask_trans) -> bool:
        """Does the restructured (hoisted) path cover this call?

        Every test is on shapes and flags -- nothing reads a tensor *value* --
        which is what lets it gate a graph capture: a host-side branch on data
        would bake one call's data into the graph.
        """
        if not self._use_ada or not _mask_trans:
            return False
        n = a.shape[-2]
        if a.numel() != n * plan.c_a or s.numel() != n * plan.c_s:
            return False
        if mask is not None and mask.numel() != n:
            return False
        if self.use_cross_attention:
            nq, nk = self._n_query, self._n_key
            return z.numel() == -(-n // nq) * nk * nq * plan.c_z
        return z.numel() == n * n * plan.c_z

    def _forward_fast(self, a, s, z, mask, plan):
        if mask is None:
            mask = a.new_ones(a.shape[-2])
        if self.use_cross_attention:
            return self._forward_cross(a, s, z, mask, plan)
        return self._forward_self(a, s, z, mask, plan)

    # -- graph capture ------------------------------------------------------
    def _capture(self, key, a, s, z, mask, plan):
        """Capture one whole call, or give up on this signature for good.

        The restructured path is capturable as it stands: nothing in it reads a
        tensor value on the host.  ``_get_block_key_indices`` keeps its
        deliberately-bf16 index arithmetic and its ``n_real = mask.sum()``
        comparison entirely on device (``torch.where`` / ``clamp``, no
        ``.item()`` and no data-dependent shape), and the block mask and mask
        bias are plain broadcasts.  So the captured region is the *same op
        sequence in the same order* as the eager path -- the numerics do not
        move, which the bit-exactness check below enforces rather than assumes.

        Capture needs the lazy state materialized first: the caller has already
        run one eager call, which builds the folded/stacked weights, the scratch
        and the int64 cfg block, so none of them is allocated during capture.
        """
        if not _GRAPHS or key in self._nocap or not a.is_cuda:
            return
        if torch.is_grad_enabled() or torch.cuda.is_current_stream_capturing():
            return
        self._nocap.add(key)     # removed again only once capture has succeeded
        ins = [a, s, z] if mask is None else [a, s, z, mask]
        try:
            # Static inputs.  One flat buffer when the dtypes agree, so staging
            # is a single ``cat(out=)`` rather than four copies; otherwise one
            # buffer each and a ``_foreach_copy_``.
            flat, dst = None, []
            esz = ins[0].element_size()
            offs, off = [], 0
            for t in ins:
                offs.append(off)
                off += t.numel()
            if (len({t.dtype for t in ins}) == 1
                    and all(o * esz % 16 == 0 for o in offs)):
                flat = ins[0].new_empty((off,))
                dst = [flat.narrow(0, o, t.numel()).view(t.shape)
                       for o, t in zip(offs, ins)]
            else:
                # Mixed dtypes, or a slice that would land unaligned inside the
                # flat buffer: one buffer each (each 512 B aligned) instead.
                dst = [torch.empty(t.shape, dtype=t.dtype, device=t.device)
                       for t in ins]
            for d, t in zip(dst, ins):
                d.copy_(t)
            args = dst if mask is not None else dst + [None]

            # Warm up on a side stream: this forces every lazy allocation, any
            # cuBLAS workspace and the attention kernel's shared-memory opt-in
            # to happen *before* capture.
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(_GRAPH_WARMUP):
                    self._forward_fast(*args, plan)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                out = self._forward_fast(*args, plan)

            # End-to-end guard: a replay must reproduce the eager result *bit
            # for bit*.  Any drift means the preamble was restructured rather
            # than merely captured, so the graph is thrown away.
            graph.replay()
            ref = self._forward_fast(*args, plan)
            if (ref.shape != out.shape or ref.dtype != out.dtype
                    or not torch.equal(ref.reshape(-1).view(torch.uint8),
                                       out.reshape(-1).view(torch.uint8))):
                raise RuntimeError("graph replay is not bit-exact")
        except Exception:
            # A failed capture leaves nothing behind but a synchronize; the
            # eager driver stays the correctness reference for this signature.
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            return
        ent = _Graph()
        ent.graph, ent.dst, ent.flat, ent.out = graph, dst, flat, out
        self._nocap.discard(key)
        self._graphs[key] = ent

    # -- self-attention stack ----------------------------------------------
    def _forward_self(self, a, s, z, mask, plan):
        c_a, c_s, c_z = plan.c_a, plan.c_s, plan.c_z
        nh, dh, c_hid, c_ff = plan.n_heads, plan.c_hidden, plan.c_hid_all, plan.c_ff
        nb = len(self.blocks)
        n = a.shape[-2]
        eps, ext = plan.eps, plan.ext
        if ext is not None and _attn_smem(n, n, dh) > _SMEM_MAX:
            ext = None
        out_shape = a.shape

        a2 = a.reshape(n, c_a).clone()      # written in place below
        s2 = s.reshape(n, c_s)
        mcol = mask.reshape(n, 1)

        # --- loop-invariant conditioning --------------------------------
        # One layer_norm_s(s) and one GEMM produce every block's AdaLN gate and
        # shift for both AdaLN sites; a second GEMM every block's
        # linear_ada_out(s) gate and transition output gate.
        sh = F.layer_norm(s2, (c_s,), None, None, eps)
        g1 = torch.addmm(plan.b1, sh, plan.W1)
        torch.sigmoid_(g1.narrow(1, 0, 2 * nb * c_a))
        g2 = torch.addmm(plan.b2, s2, plan.W2)
        torch.sigmoid_(g2)

        # One normalization of z (each block's weight-only layer_norm_z folded
        # into its linear_z) and one GEMM produce all blocks' [H, Q, K] biases;
        # the mask bias is identical for every block, so it is added in once.
        rt = None if ext is None else self._bufs(plan, n, None, n * n, n, n, 1)
        zh = F.layer_norm(z.reshape(-1, c_z), (c_z,), None, None, eps)
        # Left in the layout the stacked GEMM produces, [row, block * H + h]: the
        # attention kernel indexes it directly and adds the mask bias itself.
        zb = torch.mm(zh, plan.Wz)
        mb = mask.reshape(1, 1, 1, n).sub(1).mul_(self._inf)

        if ext is None:
            half = 2 * nb * c_a
            g_l, sh_l = self._split_cond(g1.narrow(1, 0, half),
                                         g1.narrow(1, half, half), 2, nb, c_a)
            g2v = g2.view(n, 2, nb, c_a)
            bias_l = (zb.view(n, n, nb, nh).permute(2, 3, 0, 1) + mb).unbind(0)
            return self._loop_self_torch(a2, mcol, bias_l, g_l[0], sh_l[0], g_l[1],
                                         sh_l[1], g2v.select(1, 0).unbind(1),
                                         g2v.select(1, 1).unbind(1), plan, out_shape)

        # One crossing runs the whole stack: 7 launches per block from C++.
        ext.run_self(rt["cfg"], a2, g1, g2, zb, mb, mcol, eps)
        return a2.view(out_shape)

    def _loop_self_torch(self, a2, mcol, bias_l, g_attn, sh_attn, g_tr, sh_tr,
                         gate_ao, gate_ct, plan, out_shape):
        """Eager fallback for the self-attention block loop (no extension).

        Kept operation-for-operation faithful to the reference's rounding: the
        AdaLN is ``g * (norm + shift)`` rather than an ``addcmul`` on a
        premultiplied ``g * shift``, and each residual contribution is
        materialized in bf16 before being added, because over 24 blocks the
        extra roundings otherwise walk onto the 1%-relative tolerance.
        """
        c_a, nh, dh, c_hid = plan.c_a, plan.n_heads, plan.c_hidden, plan.c_hid_all
        c_ff, eps = plan.c_ff, plan.eps
        n = a2.shape[0]
        for b in range(len(self.blocks)):
            an = F.layer_norm(a2, (c_a,), None, None, eps)
            ax = torch.add(an, sh_attn[b]).mul_(g_attn[b])
            qkvg = F.linear(ax, plan.Wqkvg[b], plan.bqkvg[b])
            torch.sigmoid_(qkvg.narrow(1, 3 * c_hid, c_hid))
            q, k, v, gt = qkvg.view(n, 4, nh, dh).permute(1, 2, 0, 3).unbind(0)
            sc = torch.matmul(q, k.transpose(-1, -2)) + bias_l[b]
            o = torch.matmul(torch.softmax(sc, -1), v) * gt
            ao = F.linear(o.permute(1, 0, 2).reshape(n, c_hid), plan.Wo[b])
            a2 = a2 + gate_ao[b] * ao

            tn = F.layer_norm(a2, (c_a,), None, None, eps)
            tx = torch.add(tn, sh_tr[b]).mul_(g_tr[b])
            hh = F.linear(tx, plan.Wab[b])
            act = F.silu(hh.narrow(1, 0, c_ff)) * hh.narrow(1, c_ff, c_ff)
            upd = gate_ct[b] * F.linear(act, plan.Wout[b])
            a2 = a2 + upd.mul_(mcol)
        return a2.view(out_shape)

    # -- cross-attention stack ---------------------------------------------
    def _forward_cross(self, a, s, z, mask, plan):
        c_a, c_s, c_z = plan.c_a, plan.c_s, plan.c_z
        nh, dh, c_hid, c_ff = plan.n_heads, plan.c_hidden, plan.c_hid_all, plan.c_ff
        nb = len(self.blocks)
        nq, nk = self._n_query, self._n_key
        n = a.shape[-2]
        nblk = math.ceil(n / nq)
        pad = (-n) % nq
        npad = n + pad
        nkeys = nblk * nk
        eps, ext = plan.eps, plan.ext
        if ext is not None and _attn_smem(nq, nk, dh) > _SMEM_MAX:
            ext = None
        out_shape = a.shape

        # ``a`` itself must not be touched: the fused path updates the residual
        # in place, and F.pad with a zero pad width would hand back a view.
        a_pad = (F.pad(a.reshape(n, c_a), (0, 0, 0, pad)) if pad
                 else a.reshape(n, c_a).clone())
        s_pad = F.pad(s.reshape(n, c_s), (0, 0, 0, pad))
        m_pad = F.pad(mask.reshape(n), (0, pad))
        mcol = m_pad.view(npad, 1)

        # --- loop-invariant block gather / mask -------------------------
        # Indices, block mask and the gathered s depend only on mask / s, so the
        # reference's two _convert_single_rep_to_blocks calls per block collapse
        # to one index computation for the whole stack.
        zrows = nkeys * nq
        rt = (None if ext is None else
              self._bufs(plan, npad, nkeys, zrows, nq, nk, nblk))
        if rt is not None and rt["idx"] is not None:
            # Two kernels instead of ~25 tiny ones (57 us per call); see
            # kernel.cu: k_blkidx / k_maskbias.  ``n_real`` stays a torch
            # reduction because its fp32 reduction order is what the reference's
            # bf16 result depends on.
            idx, vcol, mbias = rt["idx"], rt["vcol"], rt["mbias"]
            n_real = m_pad.sum(dim=-1, keepdim=True)
            ext.blkidx(m_pad, n_real, idx, rt["valid"], rt["mk"], mbias,
                       nblk, nq, nk, self._inf)
        else:
            idx, invalid = _get_block_key_indices(m_pad.view(1, npad), nq, nk)
            idx = idx.reshape(-1)
            valid = (~invalid).to(a.dtype)
            vcol = valid.reshape(-1, 1)
            mk = (valid.reshape(1, nblk, nk)
                  * m_pad.index_select(0, idx).view(1, nblk, nk))
            mbias = (m_pad.view(1, nblk, nq, 1) * mk.unsqueeze(-2))
            mbias = mbias.sub_(1).mul_(self._inf)

        # --- loop-invariant conditioning --------------------------------
        zf = z.reshape(zrows, c_z)
        if rt is not None and rt["zn"] is not None:
            # ``F.layer_norm`` is 62 us on these 49152 rows of 16 against 8 us
            # for one thread per row (kernel.cu: k_lnz).  Its fp32 promotion is
            # dropped with it: a bf16 F.layer_norm already reduces in fp32 and
            # the two were verified bit-identical.
            zh = rt["zn"]
            ext.lnz(zf, self.layer_norm_z.weight, zh, eps)
        else:
            zh = F.layer_norm(zf.float(), (c_z,),
                              self.layer_norm_z.weight.float(), None, eps).to(a.dtype)
        zb = torch.mm(zh, plan.Wz)

        sh = F.layer_norm(s_pad, (c_s,), None, None, eps)
        g1 = torch.addmm(plan.b1, sh, plan.W1)
        torch.sigmoid_(g1.narrow(1, 0, 2 * nb * c_a))

        # Key side: the reference zeroes invalid key rows *before* normalizing,
        # so the gathered s must be masked before its own layer_norm_s.
        s_k = s_pad.index_select(0, idx) * vcol
        shk = F.layer_norm(s_k, (c_s,), None, None, eps)
        gk = torch.addmm(plan.b1k, shk, plan.W1k)
        torch.sigmoid_(gk.narrow(1, 0, nb * c_a))

        g2 = torch.addmm(plan.b2, s_pad, plan.W2)
        torch.sigmoid_(g2)
        if pad:
            # Padded query rows are zero and must stay zero: zeroing their
            # linear_ada_out gate lets every residual run over all npad rows
            # (their transition gate is already killed by the zero mask).
            g2.narrow(0, n, pad).zero_()
        if ext is None:
            half = 2 * nb * c_a
            g_l, sh_l = self._split_cond(g1.narrow(1, 0, half),
                                         g1.narrow(1, half, half), 2, nb, c_a)
            hk = nb * c_a
            g_k = gk.narrow(1, 0, hk).view(nkeys, nb, c_a).unbind(1)
            sh_k = gk.narrow(1, hk, hk).view(nkeys, nb, c_a).unbind(1)
            g2v = g2.view(npad, 2, nb, c_a)
            bias = zb.view(nblk, nq, nk, nb, nh).permute(3, 0, 4, 1, 2)
            bias_l = (bias + mbias.unsqueeze(-3)).unbind(0)
            return self._loop_cross_torch(a_pad, n, nblk, nq, nk, mcol, idx, vcol,
                                          bias_l, g_l[0], sh_l[0], g_k, sh_k,
                                          g_l[1], sh_l[1],
                                          g2v.select(1, 0).unbind(1),
                                          g2v.select(1, 1).unbind(1), plan,
                                          out_shape)

        ext.run_cross(rt["cfg"], a_pad, g1, g2, gk, zb, mbias, mcol, idx, vcol, eps)
        return a_pad.narrow(0, 0, n).view(out_shape)

    def _loop_cross_torch(self, a_pad, n, nblk, nq, nk, mcol, idx, vcol, bias_l,
                          g_q, sh_q, g_k, sh_k, g_tr, sh_tr, gate_ao, gate_ct,
                          plan, out_shape):
        """Eager fallback for the cross-attention block loop (no extension)."""
        c_a, nh, dh, c_hid = plan.c_a, plan.n_heads, plan.c_hidden, plan.c_hid_all
        c_ff, eps = plan.c_ff, plan.eps
        npad = a_pad.shape[0]
        for b in range(len(self.blocks)):
            an = F.layer_norm(a_pad, (c_a,), None, None, eps)
            aq = torch.add(an, sh_q[b]).mul_(g_q[b])
            ank = an.index_select(0, idx).mul_(vcol)
            ak = torch.add(ank, sh_k[b]).mul_(g_k[b])
            qg = F.linear(aq, plan.Wqkvg[b], plan.bqkvg[b])
            torch.sigmoid_(qg.narrow(1, c_hid, c_hid))
            kv = F.linear(ak, plan.Wkv[b])
            q, gt = qg.view(nblk, nq, 2, nh, dh).permute(2, 0, 3, 1, 4).unbind(0)
            k, v = kv.view(nblk, nk, 2, nh, dh).permute(2, 0, 3, 1, 4).unbind(0)
            sc = torch.matmul(q, k.transpose(-1, -2)) + bias_l[b]
            o = torch.matmul(torch.softmax(sc, -1), v) * gt
            ao = F.linear(o.permute(0, 2, 1, 3).reshape(npad, c_hid), plan.Wo[b])
            a_pad = a_pad + gate_ao[b] * ao

            tn = F.layer_norm(a_pad, (c_a,), None, None, eps)
            tx = torch.add(tn, sh_tr[b]).mul_(g_tr[b])
            hh = F.linear(tx, plan.Wab[b])
            act = F.silu(hh.narrow(1, 0, c_ff)) * hh.narrow(1, c_ff, c_ff)
            upd = gate_ct[b] * F.linear(act, plan.Wout[b])
            a_pad = a_pad + upd.mul_(mcol)
        return a_pad.narrow(0, 0, n).view(out_shape)
