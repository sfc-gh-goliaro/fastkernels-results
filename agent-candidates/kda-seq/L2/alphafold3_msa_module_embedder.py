"""MSA module embedder for AlphaFold3 (Algorithm 8, lines 1-4), fused into one kernel.

Reference: openfold3/core/model/feature_embedders/input_embedders.py
           MSAModuleEmbedder

The baseline is five host operations -- ``torch.cat`` of three tensors, two
bias-free ``Linear``s, an ``unsqueeze`` and a broadcast add -- producing four
kernel launches and two intermediates. None of that is arithmetic: the captured
shape is 740 K MACs over ~100 KB, which a B200 does in noise. What it costs is
host issue time.

Measured inside a faithful reproduction of the harness's own timed window
(``profile/measure_costs.py``, which drives ``bench.py``'s ``_collect_cases`` /
``_make_call`` / ``_time_module``):

===================================================  ==============
configuration                                        median window
===================================================  ==============
floor: 9 pool copies, module does nothing            34.1 us
floor + 1 / 2 / 3 / 4 elementwise launches           38.4 / 42.0 / 45.6 / 49.9
floor + ``torch.cat`` of 3                           39.9 us
floor + one ``F.linear`` [16,449]x[449,64]           48.1 us
baseline module                                      85.8 us
===================================================  ==============

So a launch costs ~4 us of window and an ``F.linear`` ~14 us, the extra ~10 us
being cuBLAS's host-side heuristic selection -- which two GEMMs this small should
never see. The 34-41 us floor is the harness's ``_ShiftingPool`` re-copying the
forward's nine tensor leaves, including the four ``batch`` entries this forward
never reads; both arms pay it and it is not addressable from here. The
addressable part is the module's 51.7 us, and collapsing it to one kernel behind
one pybind crossing is the entire optimization. Tensor cores, tiling and
occupancy are not the lever at this size.

Every rule on the fast path below follows from that ~4 us: the extension
allocates its own output so one crossing does allocate-and-launch, there is no
Python-side ``unsqueeze`` / ``view`` / ``reshape`` / ``contiguous``, nothing is
cached across calls, and the build happens at import.

Layouts the kernel does not claim are reported in-band -- the entry point returns
an undefined tensor, which arrives here as ``None`` -- and routed to
``_forward_ref``, which is the baseline body verbatim over the frozen
``..L1.linear.Linear``. The same path serves a failed build. Nothing catches an
exception around a launch: the fallback is a decision, not a recovery.
"""

from __future__ import annotations

import hashlib
import os
import sys

import torch
import torch.nn as nn

from fastkernels.infra.cuda_ext import load_op

from ..L1.linear import Linear

# The extension name is the sole key for both the ninja build directory and the
# resulting ``.so``, and that cache is shared across operator workspaces and
# runs; torch also rebuilds whenever a source is newer than the ``.so``. So the
# name is operator-specific *and* content-addressed: any edit to the sidecar
# moves the build directory, and a stale or concurrently written ``.so`` can
# never be picked up. Sharing a name with the vendored ``linear`` extension --
# live in this same process, because the fallback goes through it -- would make
# the two sources invalidate each other's build on every import.
_SOURCE = "msa_module_embed_kernels.cu"
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), _SOURCE), "rb") as _f:
    _SOURCE_DIGEST = hashlib.sha256(_f.read()).hexdigest()[:12]

_EXT = None
# "built", "disabled:no-cuda-device", or "failed:<Type>: <msg>". Exposed because a
# swallowed build failure would present as a plausible 1.00x rather than as
# breakage, and ``bench.py`` cannot tell the two apart.
_KERNEL_STATUS = "disabled:no-cuda-device"

if torch.cuda.is_available():
    try:
        # Eager, not lazy: the harness times eager forward calls, so a first
        # build inside ``forward`` would land inside the timed region. ``load_op``
        # resolves sources relative to the *calling module's* directory, so it has
        # to be called from this file rather than from a helper elsewhere.
        _EXT = load_op(f"fk_cand_l2_msa_embed_{_SOURCE_DIGEST}", _SOURCE)
        _KERNEL_STATUS = "built"
    except Exception as exc:  # compiler, driver, or architecture rejected the kernel
        _KERNEL_STATUS = f"failed:{type(exc).__name__}: {exc}"
        # One line, at import, on stderr: the bench worker routes this to the
        # per-operator log, so the failure stays visible instead of hiding behind
        # a silent 1.00x.
        print(f"[candidate L2/alphafold3_msa_module_embedder] kernel unavailable, "
              f"using the reference forward: {_KERNEL_STATUS}", file=sys.stderr, flush=True)

# Bound at import. The whole module is host-bound, so each attribute lookup
# avoided on the way to the kernel is a real fraction of what it costs.
_embed = _EXT.msa_module_embed if _EXT is not None else None
_is_compiling = torch.compiler.is_compiling

# Plain module-level integers, incremented on the host. Nothing here spawns a
# thread or synchronizes, so it trips none of the harness's integrity guards --
# ``_check_threads`` snapshots ``threading.active_count()`` immediately before
# candidate timing and rejects any increase. This is the only thing that
# distinguishes a run that exercised the kernel from a run that passed entirely
# through the reference forward at 1.00x.
_FASTPATH_HITS = 0
_REFERENCE_HITS = 0


def kernel_status() -> str:
    """``"built"``, ``"disabled:no-cuda-device"``, or ``"failed:<Type>: <msg>"``."""
    return _KERNEL_STATUS


def path_counts() -> tuple[int, int]:
    """``(fast_path_calls, reference_calls)`` since import."""
    return _FASTPATH_HITS, _REFERENCE_HITS


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
        # Exactly the baseline's submodules, under exactly the baseline's names,
        # and nothing derived from their weights. The harness shares weights with
        # ``load_state_dict(..., strict=False)`` inside a bare ``try/except:
        # pass``, so a renamed submodule is a *silent* failure -- the candidate
        # would be timed and compared against its own random initialization. And
        # it casts parameters and only *then* loads the state dict, so any
        # pre-transposed or packed copy made here would be stale (the trap
        # ``candidate/L1/linear.py`` documents). The kernel is written so no
        # derived copy is wanted: it transposes ``Wm`` into shared memory itself
        # and reads ``Ws`` as it lies.
        self.linear_m = Linear(c_m_feats, c_m, bias=False)
        self.linear_s_input = Linear(c_s_input, c_m, bias=False)

    def _forward_ref(
        self,
        batch: dict,
        s_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The baseline body, verbatim, over the frozen ``..L1.linear.Linear``.

        Reached for a layout the kernel declines, for a failed build, and under
        tracing. Kept identical to the baseline rather than reimplemented so
        divergence from it is impossible.
        """
        msa_feat = torch.cat(
            [
                batch["msa"],
                batch["has_deletion"].unsqueeze(-1),
                batch["deletion_value"].unsqueeze(-1),
            ],
            dim=-1,
        )
        msa_mask = batch["msa_mask"]

        m = self.linear_m(msa_feat)
        m = m + self.linear_s_input(s_input).unsqueeze(-3)

        return m, msa_mask

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
                   msa_mask [*, N_msa, N_token]
            s_input: [*, N_token, c_s_input]

        Returns:
            m: [*, N_seq, N_token, c_m]
            msa_mask: [*, N_seq, N_token]
        """
        global _FASTPATH_HITS, _REFERENCE_HITS
        # Under tracing the kernel must not be reached at all. Beyond Inductor
        # wanting a graph it can fuse, ``validate/bench_openfold3.py`` can wrap
        # the model in ``torch.compile(mode="reduce-overhead")``, and a
        # CUDA-graph replay would be outright *wrong* under this harness: the
        # shifting pool hands out a fresh ``data_ptr`` every iteration, so a
        # captured graph reads the previous iteration's buffers. That guarantee
        # is also what licenses the allocator shortcut inside the extension.
        if _embed is None or _is_compiling():
            _REFERENCE_HITS += 1
            return self._forward_ref(batch, s_input)

        # One crossing, which decides *and* executes: the extension checks every
        # predicate, allocates the output and launches. ``msa_mask`` never
        # reaches the kernel -- it is a pass-through, not work.
        m = _embed(batch["msa"], batch["has_deletion"], batch["deletion_value"],
                   s_input, self.linear_m.weight, self.linear_s_input.weight)
        if m is None:
            # An unclaimed layout: an unsupported dtype or device, a
            # non-contiguous operand, a shape outside the checked family, active
            # autograd, or a staging budget over the per-block limit. The
            # predicates all ran before any allocation, so nothing was written.
            _REFERENCE_HITS += 1
            return self._forward_ref(batch, s_input)

        _FASTPATH_HITS += 1
        return m, batch["msa_mask"]
