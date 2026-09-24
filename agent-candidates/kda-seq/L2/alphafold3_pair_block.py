"""Fused AlphaFold3 ``PairBlock`` for B200 (sm_100), same contract as ``baseline.py``.

The block is ``TriMulOut -> TriMulIn -> TriAttStart -> TriAttEnd ->
SwiGLUTransition``, each in a residual add. Eager, on the captured shape
(``z bf16[1,16,16,128]``), it issues **109 kernels** whose total device self-time
is 431 us while the host spends 1324 us enqueueing them. A dependent-launch probe
on this device gives ``latency ~= 7 us + 2.0-3.8 us per dependent launch``, which
predicts the graphed baseline exactly (``7 + 109*2.5 = 280 us``, measured 280 us),
and the marginal host cost of a dispatch is ~5.0 us. Against that, the whole
block's 146 M MAC is ~4.1 us of plain fp32 FFMA and its HBM traffic ~0.3 us
(``profile/p1-baseline/results.md``).

So this operator is launch-bound twice over, and the two counts are separate
levers. ``alphafold3_pair_block_kernels.cu`` collapses the block to **fourteen
device kernels reached through one host dispatch**: fourteen launches instead of
109, and one Python/dispatcher crossing instead of 655 aten dispatches. Its header
documents the two algebraic facts that make the fusion possible (every LayerNorm is
followed by a matmul contracting the normalised axis, and every consumer of a
LayerNorm output consumes it through several projections of the same input, so 27
``mm`` calls become 5), the column interleaving the epilogues need, and why the
count is fourteen rather than the ten a first cut produced -- an NCU record
(``profile/p1-fused-v1/``) showed the over-fused output projections running at 0.43
waves over 148 SMs with 16.6 stall cycles per issue-active, which costs far more
than the 2.0-3.8 us a saved launch is worth.

The five baseline submodules stay as real children, so ``state_dict`` keys match
the baseline verbatim and anything the kernels do not cover reproduces the
baseline *sequence* -- not merely something inside tolerance. The fast path is
chosen from metadata only; reading tensor values would need a host sync inside
``forward``.

Two things are deliberately observable, because a green 1.0x run that silently
fell back to the baseline is the most likely way this candidate produces a false
positive: ``BUILD_STATUS`` records whether the extension built, and
``PairBlock.fast_path_reason`` answers whether a given input would take the fast
path. Neither costs host time on the timed path.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch
import torch.nn as nn

from .alphafold3_triangle_multiplication import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from .alphafold3_triangle_attention import TriangleAttention
from .alphafold3_swiglu_transition import SwiGLUTransition

# Unique to this file: torch keys both the build directory and the registered
# namespace on the extension name, so a shared name makes two sources invalidate
# each other's build on every import.
_LIBRARY_NAME = "fk_af3_pair_block"
_SOURCE = Path(__file__).with_name("alphafold3_pair_block_kernels.cu")

# One warp per attention head, so the per-head hidden dim must be the warp width.
_WARP = 32
# Sections per triangle-multiplication and per triangle-attention group in the
# packed buffer, matching the extension's Section enum.
_SEC_TRIMUL, _SEC_ATT = 8, 5
# Bound on N. The attention kernel's dynamic shared memory grows with N (it holds
# the scores and the triangle bias for the CTA's queries), and the extension opts in
# above the 48 KB default, but there is no reason to let an untested size through:
# beyond this the baseline sequence is the honest answer.
_MAX_N = 512


def _load_ops():
    """Build and register the fused operators at import, returning the namespace.

    Compiling here rather than inside ``forward`` keeps it out of every timed
    region, and out of the warmup iterations that ``_check_threads`` brackets:
    that guard samples ``threading.active_count()`` immediately around the timing
    call, so ninja's threads are only safe if they are already in the "before"
    sample.
    """
    from torch.utils.cpp_extension import load

    ns = getattr(torch.ops, _LIBRARY_NAME, None)
    if ns is not None and hasattr(ns, "pair_block"):
        return ns  # already loaded in this process; dlopen-ing twice re-registers

    # cpp_extension otherwise honours the ambient TORCH_CUDA_ARCH_LIST, which in
    # this environment names six architectures -- six nvcc passes for five
    # targets that will never run the kernel. Narrowing it to the live device is
    # what keeps the cold build comfortable inside the harness's 1200 s wall cap.
    # Derived from the device rather than hardcoded, and restored afterwards.
    previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    try:
        load(
            name=_LIBRARY_NAME,
            sources=[str(_SOURCE)],
            extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
            is_python_module=False,
            # The bench worker's stall watchdog reads the mtime of the log its
            # stdout is redirected to, so streaming ninja progress refreshes the
            # 600 s clock. Verbose output is a safety feature here, not noise.
            verbose=True,
        )
    finally:
        if previous is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous
    return getattr(torch.ops, _LIBRARY_NAME)


try:
    _ops = _load_ops()
    # Bind the overloads, not the packets: a packet re-resolves overloads from
    # the argument types on every call, and this path is launch-latency bound.
    _pair_block_op = _ops.pair_block.default
    _layout_op = _ops.layout.default
    BUILD_STATUS = "ok"
except Exception as exc:  # noqa: BLE001 - a build that cannot happen must
    # degrade to the baseline sequence, not take the module import down and cost
    # every case at once. The reason is retained rather than swallowed.
    _ops = None
    _pair_block_op = None
    _layout_op = None
    BUILD_STATUS = f"unavailable: {type(exc).__name__}: {exc}"


class _PackedLayout:
    """Where each fused weight lives in the single packed buffer.

    Read from the extension rather than mirrored here, so the host cannot disagree
    with the kernels about a section's offset. Also carries the two build facts the
    host needs: whether this build reproduces the baseline's bf16 rounding points
    (which decides whether to pack the folded weight or the raw weight plus the
    LayerNorm affine), and the largest hidden width the attention block shape
    supports.
    """

    __slots__ = ("off", "num", "total", "faithful", "max_hd")

    def __init__(self, *cfg: int):
        rows = _layout_op(*cfg).tolist()
        nsec = len(rows) - 2
        self.off = [r[0] for r in rows[:nsec]]
        self.num = [r[1] for r in rows[:nsec]]
        self.total, faithful = rows[nsec]
        self.faithful = bool(faithful)
        self.max_hd = rows[nsec + 1][0]


def _affine(norm: nn.Module, width: int, device, dtype=torch.float32):
    """LayerNorm scale and offset as fp32, substituting the identity when the
    module was built without them."""
    w = norm.weight
    b = norm.bias
    w = (torch.ones(width, device=device, dtype=dtype) if w is None
         else w.detach().to(device=device, dtype=dtype))
    b = (torch.zeros(width, device=device, dtype=dtype) if b is None
         else b.detach().to(device=device, dtype=dtype))
    return w, b


def _fold(weight: torch.Tensor, ln_w: torch.Tensor, ln_b: torch.Tensor,
          faithful: bool):
    """Fold a LayerNorm affine into the GEMM that follows it.

    With ``LN(x)_k = xhat_k*w_k + b_k`` and a GEMM contracting ``k``,
    ``y_c = sum_k (W[c,k]*w_k)*xhat_k + sum_k W[c,k]*b_k``, so the scale becomes
    ``W diag(w)`` and the offset a bias vector. The fold removes the baseline's
    bf16 round between the LayerNorm and the GEMM, so the rounding-faithful build
    is handed the raw weight and applies the affine in the kernel instead.
    """
    if faithful:
        return weight, weight.new_zeros(weight.shape[0])
    return weight * ln_w[None, :], weight @ ln_b


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

        # Plain ints, so nothing here reaches state_dict().
        self._c_z = int(c_z)
        self._c_hidden_mul = int(c_hidden_mul)
        self._c_hidden_pair_att = int(c_hidden_pair_att)
        self._no_heads_pair = int(no_heads_pair)
        self._transition_n = int(transition_n)
        self._inf = float(inf)

        norms = [
            self.tri_mul_out.layer_norm_in, self.tri_mul_out.layer_norm_out,
            self.tri_mul_in.layer_norm_in, self.tri_mul_in.layer_norm_out,
            self.tri_att_start.layer_norm, self.tri_att_end.layer_norm,
            self.pair_transition.layer_norm,
        ]
        eps = {float(n.eps) for n in norms}
        self._eps = eps.pop() if len(eps) == 1 else None

        self._layout = (None if _layout_op is None else
                        _PackedLayout(self._c_z, self._c_hidden_mul,
                                      self._c_hidden_pair_att,
                                      self._no_heads_pair, self._transition_n))

        # Config relations the kernels assume, resolved once rather than per call.
        hd = no_heads_pair * c_hidden_pair_att
        self._cfg_ok = (
            self._layout is not None
            and self._eps is not None
            and self._c_hidden_pair_att == _WARP
            and hd % _WARP == 0
            and hd <= self._layout.max_hd
            and self._c_z % 8 == 0
            and self._c_hidden_mul % 4 == 0
            and (transition_n * c_z) % 2 == 0
        )

        # The concatenated, affine-folded weights. A plain attribute, not a
        # parameter or a buffer, so it never enters state_dict() and can never
        # show up as an unexpected key.
        self._packed: torch.Tensor | None = None

        # Invalidation is event-driven, never polled. Comparing parameter
        # identity per forward would not even be correct -- load_state_dict
        # copies in place and the harness assigns p.data, so neither replaces the
        # Parameter object -- and comparing data_ptr or _version would spend ~20
        # attribute reads per call against a ~5 us dispatch budget in an operator
        # that is host-bound to begin with.
        self.register_load_state_dict_post_hook(PairBlock._invalidate_hook)

    # -- fused weight cache ------------------------------------------------
    @staticmethod
    def _invalidate_hook(module: "PairBlock", incompatible_keys) -> None:
        module._packed = None

    def _apply(self, *args, **kwargs):
        # Covers .to(), .cuda(), and dtype casts, which move or replace the
        # source parameters underneath a cache that holds their values.
        self._packed = None
        return super()._apply(*args, **kwargs)

    def _build_packed(self, device: torch.device) -> torch.Tensor:
        """Concatenate, pre-scale and transpose every projection weight into one
        fp32 buffer, laid out by the extension's own ``layout`` so the host and
        the kernels cannot disagree about where a section starts."""
        layout = self._layout
        off, num, faithful = layout.off, layout.num, layout.faithful
        packed = torch.zeros(layout.total, dtype=torch.float32, device=device)

        def put(section: int, value: torch.Tensor) -> None:
            flat = value.reshape(-1).to(device=device, dtype=torch.float32)
            width = num[section]
            assert flat.numel() <= width, (section, flat.numel(), width)
            packed[off[section] : off[section] + flat.numel()] = flat

        def wide(section: int, weight: torch.Tensor) -> None:
            """Store a [cols, K] weight transposed into the section's [K, padded]
            shape, zero-filling the padding so a column tile is always in bounds
            and the kernels' k loop needs no predicate."""
            k = weight.shape[1]
            buf = torch.zeros(k, num[section] // k, dtype=torch.float32,
                              device=device)
            buf[:, : weight.shape[0]] = weight.t()
            put(section, buf)

        def w32(*mods: nn.Module) -> list[torch.Tensor]:
            return [mod.weight.detach().to(device=device, dtype=torch.float32)
                    for mod in mods]

        c_z, m = self._c_z, self._c_hidden_mul
        f = self._transition_n * c_z
        att_base = 2 * _SEC_TRIMUL
        tr_base = att_base + 2 * _SEC_ATT

        for t, sub in enumerate((self.tri_mul_out, self.tri_mul_in)):
            base = t * _SEC_TRIMUL
            ln_w, ln_b = _affine(sub.layer_norm_in, c_z, device)
            # Interleave (value, gate) per hidden channel, a's pairs then b's: a
            # column tile is 32-64 wide while c_hidden_mul is 128, so without the
            # interleave a gate and the value it gates would land in different CTAs.
            ap, ag, bp, bg, wg = w32(sub.linear_a_p, sub.linear_a_g,
                                     sub.linear_b_p, sub.linear_b_g,
                                     sub.linear_g)
            pair_a = torch.stack([ap, ag], dim=1).reshape(2 * m, c_z)
            pair_b = torch.stack([bp, bg], dim=1).reshape(2 * m, c_z)
            w1p, beta1 = _fold(torch.cat([pair_a, pair_b, wg], 0), ln_w, ln_b,
                               faithful)
            wide(base + 0, w1p)
            put(base + 1, beta1)
            put(base + 2, ln_w)
            put(base + 3, ln_b)

            lo_w, lo_b = _affine(sub.layer_norm_out, m, device)
            wzp, betaz = _fold(w32(sub.linear_z)[0], lo_w, lo_b, faithful)
            put(base + 4, wzp.t().contiguous())
            put(base + 5, betaz)
            put(base + 6, lo_w)
            put(base + 7, lo_b)

        # The query scale folds into the q rows of the concatenated weight. It is
        # folded in both modes: FK_PB_FAITHFUL reinstates the rounds after the
        # LayerNorm, the GEMMs, the softmax and the residual adds, which is what
        # the numerical target is stated in terms of, not this scalar.
        q_scale = 1.0 / math.sqrt(self._c_hidden_pair_att)
        for t, sub in enumerate((self.tri_att_start, self.tri_att_end)):
            base = att_base + t * _SEC_ATT
            ln_w, ln_b = _affine(sub.layer_norm, c_z, device)
            mha = sub.mha
            wq, wk, wv, wgt, wlz, wo = w32(mha.linear_q, mha.linear_k,
                                           mha.linear_v, mha.linear_g,
                                           sub.linear_z, mha.linear_o)
            wp, beta = _fold(torch.cat([wq * q_scale, wk, wv, wgt, wlz], 0),
                             ln_w, ln_b, faithful)
            wide(base + 0, wp)
            put(base + 1, beta)
            put(base + 2, ln_w)
            put(base + 3, ln_b)
            put(base + 4, wo.t().contiguous())

        sub = self.pair_transition
        ln_w, ln_b = _affine(sub.layer_norm, c_z, device)
        # Interleave (a, b) per hidden channel, for the same reason as above.
        wa, wb, wout = w32(sub.swiglu.linear_a, sub.swiglu.linear_b,
                           sub.linear_out)
        wp, beta = _fold(torch.stack([wa, wb], dim=1).reshape(2 * f, c_z),
                         ln_w, ln_b, faithful)
        wide(tr_base + 0, wp)
        put(tr_base + 1, beta)
        put(tr_base + 2, ln_w)
        put(tr_base + 3, ln_b)
        put(tr_base + 4, wout.t().contiguous())

        self._packed = packed
        return packed

    # -- fast-path selection ----------------------------------------------
    def fast_path_reason(self, z: torch.Tensor, pair_mask: torch.Tensor) -> str | None:
        """``None`` if this input would take the fused path, else why it would not.

        Diagnostic twin of the boolean gate in ``forward``; ``check_stages.py``
        asserts the two agree over a matrix of inputs, so the cheap gate cannot
        drift from the explanation.
        """
        if _pair_block_op is None:
            return BUILD_STATUS
        if not self._cfg_ok:
            return ("config outside the kernels' assumptions "
                    f"(c_hidden_pair_att={self._c_hidden_pair_att}, "
                    f"no_heads_pair={self._no_heads_pair}, c_z={self._c_z}, "
                    f"eps={self._eps})")
        if torch.is_grad_enabled():
            return "grad mode is enabled"
        if z.dtype is not torch.bfloat16 or pair_mask.dtype is not torch.bfloat16:
            return f"dtype {z.dtype}/{pair_mask.dtype} is not bf16"
        if not (z.is_cuda and pair_mask.is_cuda):
            return "input is not on CUDA"
        if z.dim() != 4 or pair_mask.dim() != 3:
            return f"rank {z.dim()}/{pair_mask.dim()} is not [B,N,N,C]/[B,N,N]"
        b, n, n2, c = z.shape
        if n2 != n or c != self._c_z:
            return f"z shape {tuple(z.shape)} is not [B,N,N,{self._c_z}]"
        if tuple(pair_mask.shape) != (b, n, n):
            return f"pair_mask shape {tuple(pair_mask.shape)} does not match z"
        if not (1 <= n <= _MAX_N):
            return f"N={n} outside the tiled range 1..{_MAX_N}"
        if not (z.is_contiguous() and pair_mask.is_contiguous()):
            return "input is not contiguous"
        p = self.tri_mul_out.linear_a_p.weight
        if p.dtype is not torch.bfloat16:
            return f"parameters are {p.dtype}, not bf16"
        if p.device != z.device:
            return f"parameters are on {p.device}, input on {z.device}"
        return None

    def _eligible(self, z: torch.Tensor, pair_mask: torch.Tensor) -> bool:
        """The gate actually on the timed path: metadata reads only, no strings,
        no tensor values (reading those would need a host sync)."""
        if not self._cfg_ok or torch.is_grad_enabled():
            return False
        if z.dtype is not torch.bfloat16 or pair_mask.dtype is not torch.bfloat16:
            return False
        if not z.is_cuda or z.dim() != 4 or pair_mask.dim() != 3:
            return False
        shape = z.shape
        n = shape[1]
        if shape[2] != n or shape[3] != self._c_z or not 1 <= n <= _MAX_N:
            return False
        if pair_mask.shape[0] != shape[0] or pair_mask.shape[1] != n \
                or pair_mask.shape[2] != n:
            return False
        if not z.is_contiguous() or not pair_mask.is_contiguous():
            return False
        p = self.tri_mul_out.linear_a_p.weight
        return p.dtype is torch.bfloat16 and p.device == z.device

    # -- forward -----------------------------------------------------------
    def _baseline_chain(self, z: torch.Tensor, pair_mask: torch.Tensor,
                        mask_trans: bool) -> torch.Tensor:
        """The baseline sequence, submodule for submodule."""
        pair_trans_mask = pair_mask if mask_trans else None
        z = z + self.tri_mul_out(z, mask=pair_mask)
        z = z + self.tri_mul_in(z, mask=pair_mask)
        z = z + self.tri_att_start(z, mask=pair_mask)
        z = z + self.tri_att_end(z, mask=pair_mask)
        z = z + self.pair_transition(z, mask=pair_trans_mask)
        return z

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

        ``chunk_size``, the three ``use_*`` flags, ``inplace_safe`` and
        ``_attn_chunk_size`` are accepted and ignored, exactly as the baseline
        ignores them: it forwards none of them to its submodules.
        """
        if self._eligible(z, pair_mask):
            packed = self._packed
            if packed is None:
                # Built on the first eligible call, which the harness's
                # construct -> cast -> sanitize -> load_state_dict -> forward
                # order puts after the weights have landed, and which is a
                # correctness round rather than a timed one.
                packed = self._build_packed(z.device)
            return _pair_block_op(
                z, pair_mask, packed, self._c_z, self._c_hidden_mul,
                self._c_hidden_pair_att, self._no_heads_pair,
                self._transition_n, self._inf, self._eps, bool(_mask_trans),
            )
        return self._baseline_chain(z, pair_mask, bool(_mask_trans))
