"""SwiGLU MLP for GLA / RetNet decoder layers: two projections into one buffer, fused.

Same contract as the baseline -- ``__init__(hidden_size, intermediate_size)``,
``forward(x)``, and the FLA parameter names ``gate_proj.weight`` /
``up_proj.weight`` / ``down_proj.weight``. The baseline evaluates

    down_proj(silu(gate_proj(x)) * up_proj(x))

as five kernels: two GEMMs, a SiLU pass, a multiply pass, a GEMM. This file writes the
two projections into disjoint column halves of **one** ``[.., 2I]`` buffer and hands
that buffer to the frozen L1 ``SiluAndMul`` kernel, which computes
``silu(in[..., :I]) * in[..., I:]`` in a single streaming pass:

    gu = empty(M, 2I);  mm(x, Wg.T, out=gu[:, :I]);  mm(x, Wu.T, out=gu[:, I:])
    down_proj(SiluAndMul()(gu))

The two ``mm`` calls write straight into their column slices -- confirmed copy-free by
kernel-level profiling, see below -- so the fused kernel's contiguity contract is met
without a ``cat``, without repacking, and without a second copy of either weight. The
activation and the multiply, which in the baseline read and write four ``[M, I]``
tensors, become one pass over ``[M, 2I]`` producing ``[M, I]``.

Measured, not assumed. Five variants were timed through the benchmark's own
``bench._time_module`` -- so the shifting pool's per-iteration input copy sits inside the
CUDA-event window exactly as it does when the score is taken -- as (candidate, then
baseline) pairs in the harness's order, three repeats, median, each ratio against the
baseline readings interleaved with *that* variant. Chain D is this module, imported rather
than reimplemented, so the row cannot drift from what ships. B200, 148 SMs, 126 MiB L2,
torch 2.11.0+cu130. `profile/p1_chain_probe/`:

    M          B merge      C merge      D THIS FILE   E this file,
                no fuse    and fuse    (exact act)     default act
    1            1.015        1.129          1.089           1.129
    64           1.015        1.221          1.081           1.117
    116          1.000        1.176          1.063           1.098
    256          0.963        1.224          1.056           1.115
    195 661      0.749        1.130          1.076           1.126
    geomean      0.942        1.175          1.073           1.117

Four readings that shape the design:

* **Fusing the activation is the lever; merging the projections is not.** B merges the two
  projections into one ``N = 13824`` GEMM but leaves the activation and multiply to torch,
  over *strided* halves of the merged output. It is neutral in the decode window and
  0.749x at prefill -- worse than the baseline. Merging without fusing is not a smaller
  win, it is a loss.
* **Merging on top of fusing is worth ~5 %** (C 1.175 against E 1.117, same activation).
  That is what this file gives up, and why, below.
* **The reference-exact activation costs ~4 %** (E 1.117 against D 1.073, same
  composition). E exists in that table for exactly this reason: without it the C-vs-D gap
  would conflate composition with activation.
* **The scored run agrees with the probe.** ``validate.py``: 5/5 ``PASSED`` at
  1.0722 / 1.0645 / 1.1072 / 1.0975 / 1.0507, geomean 1.0782x (lower median of
  three stable runs), and ``max_abs_error`` **exactly 0.0** on every case -- see the
  activation note below.

**Numerics: bitwise identity, deliberately bought.** The fused kernel's default activation
for bf16 is ``kTanhRound``, ``0.5x(1 + tanh.approx(0.5x))``; the baseline uses ``expf``
plus an IEEE divide. With the default, all five scored cases still pass at
``matched_ratio`` exactly 1.000000, but ``max_abs_error`` reaches 1.562e-02 and
``max_rel_error`` 7.8e-03 -- real margin against a 1e-2 + 1e-2*|ref| bound, and two orders
worse than the error-propagation argument for this composition predicts. So this file asks
the frozen extension for its reference-exact variant instead, through the entry point it
already exports, and every scored case becomes **bitwise identical to the baseline**
(``max_abs_error`` 0.0, kernel name ``flat_kernel<__nv_bfloat16, 32, 0>`` rather than
``..., 1``). It costs ~4 % of speedup and it makes every present and future tolerance
question unconditional; the arithmetic, not just the result, now matches. ``_EXACT_ACT_KIND``
is the single switch if that trade is ever the wrong one.

The split GEMM contributes *no* numerical difference at all: with the exact activation the
whole chain is bit-for-bit the baseline, so ``mm`` into column halves and ``F.linear`` on a
merged weight agree exactly.

**Why C is measured faster and still not shipped.** C requires a cached
``torch.cat([Wg, Wu], 0)``, since rebuilding it per call costs 135 MiB of traffic
against a 54 us call. A cache must be invalidated, and the only host-side signals
available are ``data_ptr`` and ``_version``. Those cover both mechanisms the harness
itself performs -- ``p.data = p.data.to(bf16)`` replaces the storage and moves
``data_ptr``; ``load_state_dict`` and ``_sanitize_float_params`` mutate in place and
bump ``_version`` -- but they do not cover the ``.data`` family. Measured on this
build: ``p.data.copy_(y)``, ``p.data[...] = y`` and ``p.data.add_(1)`` all change the
values while leaving ``data_ptr`` identical and ``_version`` untouched, because
``p.data`` hands out a fresh tensor carrying its own version counter (``p.detach()``
shares the parameter's counter; ``p.data`` does not). A load-state-dict post-hook or an
``_apply`` override does not see them either. So a cached merge can silently answer
from stale weights after ``gate_proj.weight.data.copy_(...)``, which is an ordinary
weight-loader idiom -- and this design's first obligation is that no lifecycle event
turns into a wrong answer. D holds no derived state at all, so the question does not
arise: every call reads the live parameters.

This design's other properties, all measured rather than argued:

* **No duplicated weight.** C holds 67.5 MiB of packed copy for the module's lifetime, per
  instance -- a model with 64 of these pays ~4.2 GiB. This holds none.
* **Exactly the baseline's memory, on both metrics.** Profiling the imported module against
  the baseline at prefill: peak allocated 9180.1 MiB and peak reserved 9240.0 MiB for
  *both*, transient 7740.0 MiB for both, 173 391.5 MiB of headroom on a 182 632 MiB device.
  This is why ``gate_up`` is released before the down projection -- holding that reference
  across the last GEMM adds one ``[M, hidden]`` tensor, measured at +956.0 MiB.
* **No hidden copies.** The imported module's profiled kernel list is three GEMMs and the
  fused kernel at ``M = 256``, and two distinct GEMMs plus the fused kernel at
  ``M = 195 661``: no ``cat``, no ``copy``, no ``contiguous``. The control that
  concatenates two separate GEMM outputs openly does show the copy kernel, and costs
  6.7 ms at prefill.
* **Almost no launch cost left to recover.** Replaying the chain from a CUDA graph, which
  collapses the inter-kernel submission gaps, saves 2.0 us of a 53.2 us call at
  ``M = 256`` and nothing at prefill -- against 5.2 us for the baseline's five launches.

Nothing is precomputed in ``__init__``: the harness constructs the module, reassigns
``p.data`` to cast fp32 to bf16, rewrites the values in place, and only then copies the
reference weights in, so anything derived at construction time would be stale three
times over. Holding no derived state also means the first timed call has no build to
perform and no thread to spawn.

The fast path is admitted by an allow-list, not by a set of equalities, and everything
outside it evaluates the baseline expression verbatim. ``x.dtype is torch.bfloat16``
rather than ``x.dtype is Wg.dtype`` because equality also admits float64 -- which the
fused kernel rejects outright, turning an answered case into an error -- and fp32,
which is unmeasured here. Two of the preconditions guard measured wrong answers rather
than merely unmeasured ones: active CUDA autocast, because ``mm`` with an ``out=``
argument cannot be autocast while the baseline's ``F.linear`` can, so the two would
compute in different dtypes; and a zero reduction dimension, because ``reshape(-1, 0)``
raises where ``F.linear`` answers with zeros. Contiguity is deliberately *not* a
precondition: ``reshape``
and ``mm`` accept a leading dimension, and the fused kernel materialises a contiguous
copy for anything it cannot read linearly, which is the same work ``F.linear`` would do
in the baseline.

``FUSED_STATUS`` is the empty string exactly when the fused kernel is live, and carries
the import error otherwise. The L1 import is guarded because it builds its extension at
import time: unguarded, a build failure would make this module unimportable, and the
bench reports that as a single ``cannot import`` row for the whole operator rather than
five answered cases.
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn

from ..L1.linear import Linear
from ..L1.silu import SiLU

# Guarded: ``..L1.silu_and_mul`` calls ``load_op`` at import, which JIT-builds a CUDA
# extension. One stderr line on failure, in the shape ``L1/linear.py`` uses, so a
# swallowed build error stays visible instead of hiding behind a silent 1.00x.
FUSED_STATUS = ""
try:
    from ..L1.silu_and_mul import SiluAndMul, _C as _FUSED_EXT
except Exception as exc:  # noqa: BLE001 - a build failure must stay importable
    SiluAndMul = None
    _FUSED_EXT = None
    FUSED_STATUS = f"{type(exc).__name__}: {exc}"
    print(f"[candidate L2/gla_mlp] fused silu_and_mul unavailable, delegating to the "
          f"baseline expression: {FUSED_STATUS}", file=sys.stderr, flush=True)

# The reference-exact activation: ``expf`` plus an IEEE divide, with the activated half
# rounded to the storage dtype before the multiply -- the arithmetic the baseline's
# ``F.silu`` performs, bit for bit. ``silu_and_mul_tuned`` reaches it through the frozen
# extension's own exported entry point; with every other argument left at its default it
# is the same ``run(...)`` call as ``silu_and_mul``, differing only in ``act_kind``.
#
# It is only *instantiated* for the 32-byte vector path, though: ``FK_DISPATCH_ACT``
# compiles the non-default variants under ``BYTES_CONST == 32`` and ``TORCH_CHECK``s
# every narrower width and the scalar path. For bf16 a 32-byte access is 16 elements, and
# ``max_vector_bytes`` admits it when ``d % 16 == 0`` and both buffers are 32-byte
# aligned -- the caching allocator's 512-byte blocks make the alignment automatic for the
# freshly allocated buffers involved here, so the divisibility is the real condition.
# ``_EXACT_VEC_ELEMS`` is that condition; anything failing it uses the module's shipped
# default activation instead, which still passes the harness gate.
_EXACT_ACT_KIND = _FUSED_EXT.ACT_EXP_ROUND if _FUSED_EXT is not None else None
_EXACT_VEC_ELEMS = 16

# Fused-path and fallback-path entries per ``(hidden_size, intermediate_size)``. Plain
# ints incremented on the host: no threads, no device sync, nothing the integrity
# guards watch. This is what distinguishes "the fused path ran and tied" from "the
# fused path never ran".
_FUSED_HITS: dict[tuple[int, int], int] = {}
_BASELINE_HITS: dict[tuple[int, int], int] = {}


class GLAMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)
        self.act = SiLU()
        # A submodule, so the fused kernel is used exactly as L1 ships it, including
        # its own compiled-mode dispatch. It holds no parameters or buffers, so it
        # changes neither ``state_dict()`` nor what the harness casts.
        self.fused_act = SiluAndMul() if SiluAndMul is not None else None
        # Whether the reference-exact activation is instantiated for this width. A plain
        # int decided from ``intermediate_size`` alone, so no derived tensor state and
        # nothing to invalidate; ``None`` means use the module's default activation.
        self._exact_act = (_EXACT_ACT_KIND
                           if (_EXACT_ACT_KIND is not None
                               and intermediate_size % _EXACT_VEC_ELEMS == 0)
                           else None)
        # Not registered, so it stays out of ``state_dict()``: a plain tuple used only
        # to key the host-side call counters.
        self._hit_key = (hidden_size, intermediate_size)

    def _use_fused(self, x: torch.Tensor) -> bool:
        """Whether the fused chain's stated preconditions hold for this call.

        Pure and cheap: dtype, device, rank and integer checks only, no CUDA call and
        no sync. Every predicate guards something the composition genuinely relies on.
        """
        if self.fused_act is None:
            return False
        # The fused kernel builds no autograd graph, and writing two GEMMs into slices
        # of one buffer is hostile to tracing; the baseline expression handles both,
        # and ``SiluAndMul`` has its own pure-PyTorch path under compilation.
        if torch.is_grad_enabled() or torch.compiler.is_compiling():
            return False
        # Autocast is a correctness precondition, not a performance one. ``mm`` with an
        # ``out=`` argument cannot be autocast -- the destination dtype is already
        # fixed -- while the baseline's ``F.linear`` is on autocast's cast list. Under
        # fp16 CUDA autocast with bf16 operands the baseline therefore computes in fp16
        # and this path would compute in bf16: measured 0.9846 matched ratio against the
        # harness's 0.99 gate on a small case, and 0.8527 at a larger weight scale.
        if torch.is_autocast_enabled("cuda"):
            return False
        gate_weight = self.gate_proj.weight
        up_weight = self.up_proj.weight
        down_weight = self.down_proj.weight
        # An allow-list, not an equality: see the module docstring on float64 and fp32.
        bf16 = torch.bfloat16
        if (x.dtype is not bf16 or gate_weight.dtype is not bf16
                or up_weight.dtype is not bf16 or down_weight.dtype is not bf16):
            return False
        # Three launches, one device.
        if not x.is_cuda:
            return False
        device = x.device
        if (gate_weight.device != device or up_weight.device != device
                or down_weight.device != device):
            return False
        # The fused kernel requires at least one dimension; two projections share one
        # buffer only if they have the same output width.
        if x.dim() < 1 or gate_weight.dim() != 2 or down_weight.dim() != 2:
            return False
        if gate_weight.shape != up_weight.shape:
            return False
        # A zero reduction dimension is excluded because ``x.reshape(-1, 0)`` raises --
        # the leading product is ambiguous when the total element count is also zero --
        # where ``F.linear`` answers a zero-K problem with zeros. Excluding it keeps a
        # degenerate shape answered rather than errored. ``M == 0`` needs no such guard:
        # ``reshape(-1, K)`` with K > 0 resolves to zero rows.
        if gate_weight.shape[1] == 0:
            return False
        return x.shape[-1] == gate_weight.shape[1]

    def _activate(self, packed: torch.Tensor) -> torch.Tensor:
        """``silu(packed[..., :I]) * packed[..., I:]``, reference-exact where available."""
        kind = self._exact_act
        if kind is None:
            return self.fused_act(packed)
        try:
            return _FUSED_EXT.silu_and_mul_tuned(packed, act_kind=kind)
        except RuntimeError as exc:
            # Only a proved dispatch-availability failure downgrades the activation. The
            # extension's own checks for it read "is only built for the 32-byte vector
            # path" and "is not built for the scalar path"; anything else -- an allocation
            # failure, a device-side error -- has to propagate rather than silently change
            # this module's numerics for the rest of the process on an unrelated fault.
            if "built for" not in str(exc):
                raise
            # The width is gated in ``__init__`` and the allocator supplies the alignment,
            # so arriving here means one of those assumptions did not hold for this input.
            # Answer with the kernel's shipped default, which still passes the harness
            # gate -- never with an exception.
            self._exact_act = None
            print(f"[candidate L2/gla_mlp] reference-exact activation not available for "
                  f"this input, using the fused kernel's default variant",
                  file=sys.stderr, flush=True)
            return self.fused_act(packed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._use_fused(x):
            key = self._hit_key
            _BASELINE_HITS[key] = _BASELINE_HITS.get(key, 0) + 1
            return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))

        key = self._hit_key
        _FUSED_HITS[key] = _FUSED_HITS.get(key, 0) + 1
        gate_weight = self.gate_proj.weight
        up_weight = self.up_proj.weight
        inter = gate_weight.shape[0]
        rows = x.reshape(-1, gate_weight.shape[1])
        gate_up = torch.empty(rows.shape[0], 2 * inter, device=x.device, dtype=x.dtype)
        # Disjoint column halves of one allocation. ``out=<strided view>`` is written
        # in place by cuBLAS here -- profiled at both a decode and the prefill shape,
        # no copy or cat kernel appears -- so this reaches the fused kernel's
        # contiguous ``[.., 2I]`` contract without materialising anything extra.
        torch.mm(rows, gate_weight.t(), out=gate_up[:, :inter])
        torch.mm(rows, up_weight.t(), out=gate_up[:, inter:])
        # Rebound rather than assigned to a second name, so that clearing this one name
        # below drops the only remaining reference to the buffer's storage.
        gate_up = gate_up.view(*x.shape[:-1], 2 * inter)
        hidden = self._activate(gate_up)
        # Released before the down projection: carrying this reference across the last
        # GEMM measured +956.0 MiB of peak allocation on the prefill case, and buys
        # nothing since the fused kernel has already consumed it.
        gate_up = None
        return self.down_proj(hidden)
