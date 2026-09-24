"""PairBlock for AlphaFold3 -- whole-block CUDA graph over an inlined block.

Shared pair-representation update block used by PairFormer, MSA module,
and template embedder. Sequence: TriMulOut -> TriMulIn -> TriAttStart ->
TriAttEnd -> SwiGLUTransition.

The captured workload is tiny -- z:bf16[1, 16, 16, 128], i.e. 256 rows of 128
channels, 64 KiB of activations, called 832 times. Profiling the eager block
gives 109 CUDA kernels and 271 us of GPU time per forward: ~2.5 us apiece, which
at this size is per-kernel fixed cost rather than arithmetic. Two things follow,
and this file does both:

1. Replay the block from a CUDA graph instead of launching it. That removes the
   ~135 us of launch overhead (the eager forward measures 406 us against 271 us
   of GPU time). Replay needs fixed addresses and the benchmark's shifting
   memory pool moves ``z`` / ``pair_mask`` every call, so the graph owns private
   static input buffers: copy in, replay, clone the static output on the way out.
2. Cut the node count inside the graph. Post-graph the cost is ~2.3 us per node
   no matter how little each node does, so the block is inlined here and written
   to reach the same values in fewer kernels. Every fusion below is exact
   algebra on the baseline expression, with the LayerNorm reductions left in
   fp32.

Reference: openfold3/core/model/latent/base_blocks.py PairBlock
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .alphafold3_triangle_multiplication import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from .alphafold3_triangle_attention import TriangleAttention
from .alphafold3_swiglu_transition import SwiGLUTransition


# Adjacent bf16 pointwise nodes are merged into single Inductor kernels. Each
# node costs ~2.7us of CUDA-graph replay latency regardless of how little it
# does, and after the GEMM fusions most of what is left is a serial pointwise
# chain, so merging neighbours is the main remaining lever.
#
# ``emulate_precision_casts`` is what makes this admissible. By default Inductor
# keeps a fused bf16 chain in fp32 registers and rounds once at the end, while
# eager rounds after every op; this block amplifies a single bf16 ulp past the
# benchmark's tolerance once the weights are O(1) (measured -- see
# ITERATIONS.md), so that is not a free reassociation here. The option
# re-inserts the intermediate casts, and every expression below was checked
# bit-identical to its eager form.
#
# Only pointwise chains are compiled -- never a region containing a matmul, and
# never a reduction. Compiling across the GEMMs lets Inductor re-lay-out their
# operands, which changes which cuBLAS kernel runs and therefore the
# accumulation order. Reductions are worse: a compiled ``layer_norm`` agrees
# with ATen's on most inputs but not all (measured 4.9e-4 on a [256, 128] bf16
# draw), because Inductor builds its own reduction tree. Folding the two
# transposed LayerNorms' staging copies into a compiled norm would have saved
# 3 nodes and was rejected for exactly this.
_PW_OPTIONS = {"emulate_precision_casts": True}


def _pw(fn):
    """Compile one bf16 pointwise chain, preserving eager's per-op rounding."""
    try:
        return torch.compile(fn, dynamic=False, options=_PW_OPTIONS)
    except Exception:  # no triton / unsupported build -- eager is still correct
        return fn


@_pw
def _pw_qkv(t: torch.Tensor, scale: float) -> torch.Tensor:
    """Stage q, k, v contiguous and apply the query scale in the same kernel."""
    return torch.cat((t[0:1] / scale, t[1:3]), 0)


@_pw
def _pw_gate(gates: torch.Tensor, mask: torch.Tensor,
             proj: torch.Tensor) -> torch.Tensor:
    # baseline: (mask * sigmoid(gate)) * proj -- two-operand products are
    # bitwise order-independent, so sigmoid * mask first is the same value
    return torch.sigmoid(gates) * mask * proj


@_pw
def _pw_res_gate(z: torch.Tensor, x: torch.Tensor,
                 gg: torch.Tensor) -> torch.Tensor:
    return z + x * torch.sigmoid(gg).t()


@_pw
def _pw_bias(scores: torch.Tensor, mask: torch.Tensor, inf: float,
             triangle_bias: torch.Tensor) -> torch.Tensor:
    # inf * (mask - 1) is recomputed inside each of the two bias kernels rather
    # than materialized once: as its own node it costs a full replay slot, and
    # inlined it costs nothing.
    return scores + (inf * (mask - 1))[:, None, None, :] + triangle_bias


@_pw
def _pw_gate_att(o: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    return o * torch.sigmoid(g)


@_pw
def _pw_swiglu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.silu(a) * b


@_pw
def _pw_res_mask(z: torch.Tensor, x: torch.Tensor,
                 mask: torch.Tensor) -> torch.Tensor:
    return z + x * mask


def _norm(ln: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """``ln(x)`` in one kernel instead of three.

    The baseline LayerNorm promotes to fp32 around the reduction:
    ``F.layer_norm(x.float(), ...).to(bf16)`` -- a cast, an fp32 norm and an
    uncast. PyTorch's native bf16 ``layer_norm`` already carries the Welford
    reduction *and* the affine in ``acc_type<BFloat16> == float``, so the
    reduction stays in fp32 and both casts disappear. On a non-contiguous input
    the cast was also serving as the ``contiguous()`` copy, so those norms go
    3 -> 2 nodes rather than 3 -> 1.
    """
    return F.layer_norm(x, ln.normalized_shape, ln.weight, ln.bias, ln.eps)


class PairBlock(nn.Module):
    """Shared pair stack block for AF3 PairFormer / MSA module / template.

    Args:
        c_z: Pair embedding channel dimension
        c_hidden_mul: Hidden dim for triangle multiplication
        c_hidden_pair_att: Per-head hidden dim for triangle attention
        no_heads_pair: Number of heads in triangle attention
        transition_n: Scale of pair transition hidden dimension
        pair_dropout: Dropout rate (unused in inference baseline)
        inf: Large constant for masking
    """

    def __init__(
        self,
        c_z: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_pair: int,
        transition_n: int,
        pair_dropout: float = 0.0,
        fuse_projection_weights: bool = False,
        inf: float = 1e9,
    ):
        super().__init__()

        self.tri_mul_out = TriangleMultiplicationOutgoing(c_z, c_hidden_mul)
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z, c_hidden_mul)

        self.tri_att_start = TriangleAttention(
            c_z, c_hidden_pair_att, no_heads_pair, starting=True, inf=inf,
        )
        self.tri_att_end = TriangleAttention(
            c_z, c_hidden_pair_att, no_heads_pair, starting=False, inf=inf,
        )

        self.pair_transition = SwiGLUTransition(c_in=c_z, n=transition_n)

        self.c_z = c_z
        self.inf = inf

        # Concatenated weights, built on the first forward from the *loaded*
        # parameters. Keyed by slot, guarded by identity against the source
        # Parameter objects -- see ``_fused_weight``.
        self._fused: dict[str, tuple] = {}

        # Lazily-captured whole-block CUDA graph, keyed on input shape+dtype.
        # Plain attributes, not buffers: ``module.to(device)`` must not try to
        # move a graph, and a device change invalidates the capture anyway
        # (the key carries the device).
        self._graph_key = None
        self._graph = None
        self._g_z = None
        self._g_mask = None
        self._g_out = None

    # -----------------------------------------------------------------------
    # Lazy concatenated weights
    # -----------------------------------------------------------------------
    def _fused_weight(self, slot: str, srcs: tuple, build) -> torch.Tensor:
        """The concatenation of *srcs*, built once and cached.

        The fused copy is a derived tensor, never an ``nn.Parameter``: the bench
        shares weights with ``load_state_dict(baseline.state_dict())``, so the
        parameter names and shapes must stay exactly the baseline's. Registering
        a pre-fused weight would leave it out of the state_dict and silently
        feed the block uninitialized memory.

        Cached behind an identity compare on the source Parameters, the same way
        the baseline LayerNorm caches its fp32 weight copy, plus a dtype/device
        compare: ``module.to()`` replaces ``param.data`` while keeping the
        Parameter object, so identity alone would not notice a cast or a device
        move. (Values mutated *in place* after the build are not seen --
        inherent to any weight pre-fusion, and to CUDA-graph capture generally,
        since a replayed graph holds the addresses it was captured with. Weight
        loading finishes before the first forward, so the cache is safe here.)
        """
        ent = self._fused.get(slot)
        if (ent is not None and len(ent[0]) == len(srcs)
                and all(a is b for a, b in zip(ent[0], srcs))
                and ent[1].dtype == srcs[0].dtype
                and ent[1].device == srcs[0].device):
            return ent[1]
        w = build()
        self._fused[slot] = (tuple(srcs), w)
        return w

    # -----------------------------------------------------------------------
    # Fused block. Operates on the batch-1 pair representation as [N, N, C].
    # -----------------------------------------------------------------------
    def _tri_mul(self, m: nn.Module, z2: torch.Tensor, mask_1r: torch.Tensor,
                 N: int, slot: str, outgoing: bool) -> torch.Tensor:
        """TriangleMultiplicativeUpdate. z2 is [R, C] contiguous (R = N*N),
        mask_1r is [1, R]. Returns ``z2 + update``, residual add included.

        Two exact rewrites of the baseline expression:

        * ``linear_a_p / linear_b_p / linear_a_g / linear_b_g / linear_g`` all
          read the same ``layer_norm_in`` output, so they are one GEMM against a
          row-concatenated weight (5 nodes -> 1), sliced afterwards.
        * That GEMM is computed **transposed** (``W @ z_ln^T``, giving
          ``[features, R]``) rather than ``z_ln @ W^T``. In this layout a slice
          is ``[c_hidden, R]`` with unit stride along R, so ``view(M, N, N)`` is
          already the ``[c, i, j]`` operand the triangle contraction needs and
          the batched matmul runs with zero staging copies -- in the natural
          layout ``permute(2, 0, 1)`` leaves no unit-stride matrix dimension and
          cuBLAS forces two ``contiguous()`` copies.

        Laying ``a_p | b_p`` adjacent and ``a_g | b_g`` adjacent also lets the
        two gate chains collapse into one: ``sigmoid(gates) * proj * mask``
        computes both ``a`` and ``b`` in 3 nodes instead of 6.
        """
        M, R = m.c_hidden, z2.shape[0]

        z_ln = _norm(m.layer_norm_in, z2)

        w = self._fused_weight(
            slot,
            (m.linear_a_p.weight, m.linear_b_p.weight, m.linear_a_g.weight,
             m.linear_b_g.weight, m.linear_g.weight),
            lambda: torch.cat([m.linear_a_p.weight, m.linear_b_p.weight,
                               m.linear_a_g.weight, m.linear_b_g.weight,
                               m.linear_g.weight], 0).contiguous(),
        )
        ft = torch.mm(w, z_ln.t())                       # [4M + C, R]

        # Multiply order matters in bf16: the baseline evaluates
        # ``(mask * sigmoid(gate)) * proj``, and every bf16 product rounds, so
        # folding the mask in last instead of first shifts ``a`` and ``b`` by a
        # full ulp (~0.4%). That survives the 16-term triangle contraction and
        # shows up as ~0.5% on the block output -- enough to drop the matched
        # ratio to 0.92 once the weights are O(1). Two-operand products are
        # order-independent bitwise, so ``sigmoid * mask`` first reproduces the
        # baseline exactly.
        ab = _pw_gate(ft[2 * M:4 * M], mask_1r, ft[:2 * M])
        a3 = ab[:M].view(M, N, N)                        # [c, i, j]
        b3 = ab[M:].view(M, N, N)                        # [c, k, j]

        # p[i, k, c] = sum_j a[i, j, c] * b[k, j, c]   (outgoing)
        # p[i, k, c] = sum_j a[j, i, c] * b[j, k, c]   (incoming)
        p = torch.bmm(a3, b3.transpose(1, 2)) if outgoing else \
            torch.bmm(a3.transpose(1, 2), b3)            # [c, i, k]

        x = _norm(m.layer_norm_out, p.view(M, R).t())    # [R, M]
        x = F.linear(x, m.linear_z.weight)
        # output gate and the block's residual add in one kernel; sigmoid reads
        # the contiguous [C, R] slice and the transpose is a free view
        return _pw_res_gate(z2, x, ft[4 * M:])

    def _tri_att(self, m: nn.Module, x3: torch.Tensor, mask2: torch.Tensor,
                 N: int, slot: str) -> torch.Tensor:
        """TriangleAttention on x3 [N, N, C] -- the caller has already oriented
        x3 and mask2 for the starting vs ending node.

        ``linear_q / linear_k / linear_v / linear_g`` and the triangle bias'
        ``linear_z`` all read the same ``layer_norm`` output, so they are one
        GEMM against a row-concatenated weight, sliced afterwards.

        The slices are then re-strided rather than copied one at a time.
        ``torch.matmul`` wants a *collapsible* batch, and `[i, j, h, d]`
        transposed to `[i, h, j, d]` has batch strides `(N*F, D)` which do not
        collapse -- so it silently stages q, k and v through three separate
        `contiguous()` copies. One ``as_strided`` over the fused output covers
        all three at once (`[3, i, h, j, d]`), so a single copy leaves both
        matmuls copy-free. The gate and the triangle bias are pure views: their
        trailing dimensions are already unit-stride inside a row.

        The scale, the two bias adds and the gate multiply stay exactly where the
        baseline has them -- moving any of them changes a bf16 rounding.
        """
        C = x3.shape[-1]
        mha = m.mha
        H, D = mha.no_heads, mha.c_hidden
        HD, R = H * D, N * N

        x_ln = _norm(m.layer_norm, x3).reshape(R, C)

        w = self._fused_weight(
            slot,
            (mha.linear_q.weight, mha.linear_k.weight, mha.linear_v.weight,
             mha.linear_g.weight, m.linear_z.weight),
            lambda: torch.cat([mha.linear_q.weight, mha.linear_k.weight,
                               mha.linear_v.weight, mha.linear_g.weight,
                               m.linear_z.weight], 0).contiguous(),
        )
        ft = F.linear(x_ln, w)                           # [R, 4*HD + H]
        W = ft.shape[1]
        off = ft.storage_offset()

        qkv = _pw_qkv(
            ft.as_strided((3, N, H, N, D), (HD, N * W, D, W, 1), off),
            math.sqrt(D))

        scores = torch.matmul(qkv[0], qkv[1].transpose(-1, -2))
        # triangle_bias[h, q, k] = linear_z(x_ln)[q, k, h], broadcast over i
        scores = _pw_bias(
            scores, mask2, m.inf,
            ft.as_strided((H, N, N), (1, N * W, W), off + 4 * HD)[None])
        scores = F.softmax(scores, dim=-1)

        o = torch.matmul(scores, qkv[2]).transpose(-2, -3)
        o = _pw_gate_att(
            o, ft.as_strided((N, N, H, D), (N * W, W, D, 1), off + 3 * HD))
        return F.linear(o.reshape(R, HD), mha.linear_o.weight).view(N, N, C)

    def _transition(self, m: nn.Module, z3: torch.Tensor,
                    mask_r: torch.Tensor | None) -> torch.Tensor:
        """SwiGLUTransition on z3 [N, N, C]. Returns ``z + update`` as [R, C],
        with the mask multiply and the residual add fused into one kernel."""
        C, R = z3.shape[-1], z3.shape[0] * z3.shape[1]
        x = _norm(m.layer_norm, z3).reshape(R, C)
        w = self._fused_weight(
            "trans",
            (m.swiglu.linear_a.weight, m.swiglu.linear_b.weight),
            lambda: torch.cat([m.swiglu.linear_a.weight,
                               m.swiglu.linear_b.weight], 0).contiguous(),
        )
        ft = F.linear(x, w)                              # [R, 2*n*C]
        h = ft.shape[1] // 2
        x = _pw_swiglu(ft[:, :h], ft[:, h:])
        x = F.linear(x, m.linear_out.weight)
        z2 = z3.reshape(R, C)
        if mask_r is None:
            return z2 + x
        return _pw_res_mask(z2, x, mask_r)

    def _block(self, z: torch.Tensor, pair_mask: torch.Tensor,
               mask_trans: bool) -> torch.Tensor:
        """The captured region: batch-1 ``z`` [1, N, N, C], ``pair_mask``
        [1, N, N]. Run eagerly for warmup, then traced once into the graph."""
        N, C = z.shape[-2], z.shape[-1]
        R = N * N
        mask2 = pair_mask.reshape(N, N)
        mask_r = mask2.reshape(R, 1)
        mask_1r = mask2.reshape(1, R)

        z2 = z.reshape(R, C)
        z2 = self._tri_mul(self.tri_mul_out, z2, mask_1r, N, "mul_out", True)
        z2 = self._tri_mul(self.tri_mul_in, z2, mask_1r, N, "mul_in", False)
        z3 = z2.view(N, N, C)

        z3 = z3 + self._tri_att(self.tri_att_start, z3, mask2, N, "att_start")
        z3 = z3 + self._tri_att(self.tri_att_end, z3.transpose(0, 1),
                                mask2.t(), N, "att_end").transpose(0, 1)

        z2 = self._transition(self.pair_transition, z3,
                              mask_r if mask_trans else None)
        return z2.view(1, N, N, C)

    # -- eager fallback for anything the fused/graph path does not cover -----
    def _block_ref(self, z: torch.Tensor, pair_mask: torch.Tensor,
                   mask_trans: bool) -> torch.Tensor:
        z = z + self.tri_mul_out(z, mask=pair_mask)
        z = z + self.tri_mul_in(z, mask=pair_mask)
        z = z + self.tri_att_start(z, mask=pair_mask)
        z = z + self.tri_att_end(z, mask=pair_mask)
        z = z + self.pair_transition(
            z, mask=pair_mask if mask_trans else None)
        return z

    # -- graph capture -------------------------------------------------------
    def _capture(self, z: torch.Tensor, pair_mask: torch.Tensor,
                 mask_trans: bool, key) -> None:
        # Drop any previous capture first so a re-key does not hold two graphs
        # (and their private memory pools) alive at once.
        self._graph_key = None
        self._graph = self._g_z = self._g_mask = self._g_out = None

        g_z = torch.empty(z.shape, dtype=z.dtype, device=z.device)
        g_mask = torch.empty(pair_mask.shape, dtype=pair_mask.dtype,
                             device=pair_mask.device)
        g_z.copy_(z)
        g_mask.copy_(pair_mask)

        # Build every lazy cache (concatenated weights, cuBLAS workspaces) on
        # the *current* stream first, so no one-shot setup lands inside the
        # capture and no cached tensor is allocated against the side stream.
        self._block(g_z, g_mask, mask_trans)

        # Warm up on a side stream before capturing: the first calls allocate
        # workspaces and pick cuBLAS heuristics. Doing that inside the capture
        # would bake one-shot setup into the replayed graph (or fail capture).
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                warm = self._block(g_z, g_mask, mask_trans)
            del warm
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            g_out = self._block(g_z, g_mask, mask_trans)

        self._g_z, self._g_mask, self._g_out = g_z, g_mask, g_out
        self._graph = graph
        self._graph_key = key

    def forward(
        self,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            z:         [*, N, N, C_z] pair embedding
            pair_mask: [*, N, N] pair mask

        Returns:
            [*, N, N, C_z] updated pair embedding
        """
        # The fused block assumes a single batch element and a contiguous z;
        # the graph additionally needs no autograd, a CUDA tensor and a stream
        # that is not already capturing. Anything else takes the eager path.
        if (pair_mask is None or not z.is_cuda or torch.is_grad_enabled()
                or z.dim() != 4 or z.shape[0] != 1 or z.shape[-2] != z.shape[-3]
                or not z.is_contiguous() or not pair_mask.is_contiguous()
                or torch.cuda.is_current_stream_capturing()):
            return self._block_ref(z, pair_mask, _mask_trans)

        key = (tuple(z.shape), z.dtype, tuple(pair_mask.shape),
               pair_mask.dtype, z.device, bool(_mask_trans))
        if self._graph_key != key:
            self._capture(z, pair_mask, bool(_mask_trans), key)

        # Never bake the caller's pointers into the graph: copy in, replay,
        # clone out. The benchmark's shifting pool moves z/pair_mask every call.
        self._g_z.copy_(z)
        self._g_mask.copy_(pair_mask)
        self._graph.replay()
        return self._g_out.clone()
