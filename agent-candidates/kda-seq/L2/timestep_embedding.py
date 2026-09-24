"""Timestep and text projection embeddings for B200 / sm_100: fewer launches.

The baseline is not compute-bound. Every scored case has batch M = 1, so every
matmul is a GEMV and there is no tensor-core opportunity; what the harness
measures is dominated by host-side dispatch. Measured on this workspace
(``profile/p1_calibration/baseline_probe.log``):

    Timesteps            [1]                78.06 us window, 11 launches, 15.4 us device
    TimestepEmbedding    [1,256]            21.50 us window,  3 launches, 13.7 us device
    TimestepEmbedding    [1,768]            23.55 us window,  3 launches, 15.1 us device
    ...GuidanceTextProj  [1],[1],[1,768]   434.54 us window, 35 launches, 78.3 us device

The lever is therefore kernel count. This module collapses each case into a
single ``load_inline`` extension call:

    Timesteps                  1 kernel   -- fp32 sinusoid, both halves per thread
    TimestepEmbedding          2 kernels  -- GEMV+bias+SiLU, then GEMV+bias
    CombinedTimestep*          2 kernels  -- the same two, over 2 or 3 branches

The three ``Combined*`` shapes and the plain MLP are the same computation with a
different branch count: each branch is ``h_b = silu(W1_b . x_b + b1_b)`` followed
by ``W2_b . h_b + b2_b``, and the branch outputs are summed. A sinusoid branch
computes its own input inside the first kernel from the frequency table, so no
projection tensor is ever materialised.

Numerics track the baseline term for term rather than approximately. The
sinusoid argument is evaluated as ``(t * f) * scale`` in that association,
because folding ``scale`` into the frequency table would change the rounding.
``F.linear``'s bf16 store after its fp32 bias epilogue is reproduced, so SiLU
reads a rounded value exactly as it does in the baseline, and the branch sum is
left-associative in bf16 -- not fp32, which would be arithmetically better and
numerically wrong; ``tests/test_numerics.py`` measures both departures rather than
asserting them. The activation is the exact ``x / (1 + exp(-x))`` ATen evaluates, not the
frozen ``candidate/L1/silu.py`` approximation, which differs from ``F.silu`` on three
bfloat16 encodings.

Admission is decided by a host-side predicate that performs no CUDA call and no
synchronisation. Anything it declines runs ``_baseline_forward``, a verbatim copy
of the baseline body that also routes its children through *their*
``_baseline_forward``, so a declined call is bitwise equal to an unmodified
baseline module rather than merely close to one. A build failure leaves
``_EXT is None``, which declines everything.

The domain is general ``M = 1`` bf16 with ``K % 8 == 0``, not an exact-shape
allow-list, because this file is imported by higher-level candidates:
``L2/sdxl_time_embedding.py`` and ``L2/hunyuan_video_embeddings.py`` use
``Timesteps``/``TimestepEmbedding``, ``L3/hunyuan_video_token_refiner_block.py``
and ``L4/flux.py`` use the composites, and ``L4/sdxl.py`` builds
``Timesteps(320, ...)`` with ``TimestepEmbedding(320, 1280)``.
"""

from __future__ import annotations

import contextlib
import math
import os
import sys
import threading
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

from ..L1.linear import Linear

# ``candidate/L1/silu.py`` is deliberately *not* imported for ``self.act``. Its
# ``__fdividef(x, 1 + __expf(-x))`` differs from ``F.silu`` at x in
# {-87.5, -88.0, -88.5}, where ``__expf(-x)`` overflows to inf and the result is -0
# against ATen's ~-1e-37 -- and AC-5 requires a declined call to be bitwise equal to an
# unmodified baseline module, whose ``act`` is ``F.silu``. ``tests/test_fallback.py``
# still measures that frozen file's behaviour, so the difference stays on the record;
# this module simply does not depend on it. ``Linear`` above is imported and used: it
# delegates every call to ``F.linear``, which is asserted rather than assumed.

_EXTENSION_NAME = "fk_timestep_embedding_sm100_v1"
_CUDA_ARCH = "10.0"
_KERNEL_SOURCE = Path(__file__).with_name("timestep_embedding_kernels.cu")

# ``Timesteps`` never forwards ``max_period``, so the frequency table is always
# built at the free function's default.
_MAX_PERIOD = 10000

# Trig forms and numerical departures, mirrored from the .cu file. The shipped path is
# (kTrigSinCos, no flags); every other combination exists so ``tests/test_numerics.py``
# can measure it against the baseline instead of the design resting on an assumption.
# FLAG_APPROX_SILU selects the frozen ``candidate/L1/silu.py`` form, which is a *rejected*
# variant here rather than the shipped one, and FLAG_FP32_H is the inter-layer experiment
# AC-9 names -- distinct from FLAG_NO_PRE_ROUND, which changes the rounding before SiLU.
TRIG_SINCOS = 0
TRIG_SEPARATE = 1
TRIG_APPROX = 2
FLAG_NO_PRE_ROUND = 1
FLAG_FP32_REDUCE = 2
FLAG_APPROX_SILU = 4
FLAG_FP32_H = 8

# ---------------------------------------------------------------------------
# Kernel geometry.
#
# The measured table lives in ``timestep_embedding_kernels.cu``
# (``choose_geometry``), keyed on (k, n, branch count), because picking it there
# costs nothing while doing it here would mean reading ``weight.shape`` on the
# host to index a dict -- and on the DRAM-bound TimestepEmbedding cases the whole
# margin over the baseline is a few microseconds of host time. ``_GEOMETRY`` is
# an override the sweep in ``profile/p2_geometry/`` sets; a non-positive component
# means "use the measured table", which is what ships. The five components are
# ``(warps_per_row, rows_per_block)`` for the first kernel, the same for the second,
# and the second kernel's per-lane unroll depth.
# ---------------------------------------------------------------------------
_SINUSOID_BLOCK = 128
_GEOMETRY = (-1, -1, -1, -1, -1)

# Whether a sinusoid branch's input is materialised by a prologue kernel and staged as
# a dense row, or recomputed inside every block of the first kernel. False -- per-block
# recomputation -- is the measured winner; see
# profile/p2_geometry/prologue_vs_recompute.log and the task23 section of
# profile/p2_geometry/REPORT.md.
USE_PROLOGUE = False

# Fast-path entries per configuration. Plain host-side ints: no device state, no
# thread, no allocation, and nothing a reader of them has to synchronise for.
# This is what distinguishes "the fused path ran and matched" from "the guard
# declined and we measured the baseline".
_FASTPATH_HITS: dict[tuple, int] = {}

# A mirror of ``kMaxStagedK`` in the .cu: the widest branch input the first kernel
# will stage in shared memory. 8192 bf16 is 16 KiB, which leaves room for the
# reduction scratch inside the 48 KiB a kernel gets without opting in to more, and
# covers every K these classes are constructed with (256, 320, 768, 1280, 2816,
# 3072). Nothing here reads it -- the predicate is on the other side -- but
# ``tests/test_fallback.py`` needs it to construct a shape deliberately outside the
# kernel's domain, and a mirror with a stated source beats a magic number in a test.
_MAX_STAGED_K = 8192


# Set to any non-empty value to make the loader below raise instead of building. This
# exists so ``tests/test_loader.py`` can exercise the degradation path -- import still
# succeeds, every class falls back, ``validate.py`` still reports PASSED -- while
# leaving the frozen ``candidate/L1`` extensions alone. Monkey-patching
# ``cpp_extension.load_inline`` instead breaks L1's build too, which round 0 did: it
# made the test pass for the wrong reason and put two degradation lines on stderr where
# the AC asks for one. Never set on any scored path.
_FORCE_BUILD_FAILURE_ENV = "FK_TE_FORCE_BUILD_FAILURE"

# Test-only overrides for the extension name and build directory, read at *import* so a
# fresh interpreter can be pointed at a never-used directory. `tests/test_loader.py` uses
# them to prove a genuinely cold build followed by a warm import in a second interpreter --
# calling `_load_extension` twice in one process proves nothing, because cpp_extension's
# in-process module cache can satisfy the second call. Unset on every scored path.
_EXT_NAME_ENV = "FK_TE_EXT_NAME"
_BUILD_DIR_ENV = "FK_TE_BUILD_DIR"


def _build_directory(name: str = "") -> str | None:
    """Persistent per-workspace build cache, so a warm import never calls nvcc."""
    try:
        path = Path(__file__).resolve().parents[2] / ".torch_extensions" / (
            name or _EXTENSION_NAME)
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None  # fall back to cpp_extension's own default location
    return str(path)


def _load_extension(arch: str | None = _CUDA_ARCH, name: str = _EXTENSION_NAME,
                    build_dir: str | None = None):
    # Pinning the arch list does two things, and only the first is conditional on
    # the environment:
    #   * it keeps the build single-arch. This workspace's shell exports
    #     TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0 10.0 12.0+PTX", so without
    #     pinning the same source compiles for six architectures instead of one.
    #   * it is outright required wherever the variable is unset or "native":
    #     that branch of _get_cuda_arch_flags iterates torch.cuda.device_count()
    #     and then indexes the result, raising IndexError with no visible GPU.
    #     The failure only surfaces when nvcc actually runs, so a warm build
    #     cache hides it -- which is why a fresh workspace needs the pin.
    # ``arch=None`` skips the pin, which is how that path is tested.
    if os.environ.get(_FORCE_BUILD_FAILURE_ENV):
        raise RuntimeError(
            f"{_FORCE_BUILD_FAILURE_ENV} is set: refusing to build, so the fallback path "
            f"can be exercised")
    previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if arch is None:
        os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
    else:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    try:
        return load_inline(
            name=name,
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_KERNEL_SOURCE.read_text(),
            functions=[
                "sinusoid_forward",
                "fused_mlp_forward",
                "fused_combined2_forward",
                "fused_combined3_forward",
                "sinusoid_admits",
                "mlp_admits",
                "combined2_admits",
                "combined3_admits",
                "silu_probe",
            ],
            # -lineinfo lets ncu attribute SASS to source. --use_fast_math is
            # deliberately absent: the one approximation here is an explicit
            # intrinsic in the activation, inherited from the frozen L1 winner.
            extra_cuda_cflags=["-O3", "-lineinfo"],
            build_directory=build_dir or _build_directory(name),
            verbose=False,
        )
    finally:
        if previous is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous


_CPP_SOURCE = r"""
at::Tensor sinusoid_forward(const at::Tensor& t, const at::Tensor& freq, int64_t num_channels,
                            bool flip, double scale, int64_t block, int64_t trig_mode);
at::Tensor silu_probe(const at::Tensor& x, int64_t numeric_flags);
bool sinusoid_admits(const at::Tensor& t, const at::Tensor& freq, int64_t num_channels);
bool mlp_admits(const at::Tensor& x, const at::Tensor& w1, const at::Tensor& b1,
                const at::Tensor& w2, const at::Tensor& b2);
bool combined2_admits(const at::Tensor& t, const at::Tensor& p, const at::Tensor& freq,
                      int64_t num_channels, const at::Tensor& tw1, const at::Tensor& tb1,
                      const at::Tensor& tw2, const at::Tensor& tb2, const at::Tensor& pw1,
                      const at::Tensor& pb1, const at::Tensor& pw2, const at::Tensor& pb2);
bool combined3_admits(const at::Tensor& t, const at::Tensor& g, const at::Tensor& p,
                      const at::Tensor& freq, int64_t num_channels, const at::Tensor& tw1,
                      const at::Tensor& tb1, const at::Tensor& tw2, const at::Tensor& tb2,
                      const at::Tensor& gw1, const at::Tensor& gb1, const at::Tensor& gw2,
                      const at::Tensor& gb2, const at::Tensor& pw1, const at::Tensor& pb1,
                      const at::Tensor& pw2, const at::Tensor& pb2);
at::Tensor fused_mlp_forward(const at::Tensor& x, const at::Tensor& w1, const at::Tensor& b1,
                             const at::Tensor& w2, const at::Tensor& b2, int64_t wpr1,
                             int64_t rpb1, int64_t wpr2, int64_t rpb2, int64_t unroll2,
                             int64_t numeric_flags);
at::Tensor fused_combined2_forward(const at::Tensor& t, const at::Tensor& p,
                                   const at::Tensor& freq, int64_t num_channels, bool flip,
                                   double scale, const at::Tensor& tw1, const at::Tensor& tb1,
                                   const at::Tensor& tw2, const at::Tensor& tb2,
                                   const at::Tensor& pw1, const at::Tensor& pb1,
                                   const at::Tensor& pw2, const at::Tensor& pb2, int64_t wpr1,
                                   int64_t rpb1, int64_t wpr2, int64_t rpb2, int64_t unroll2,
                                   int64_t trig_mode, int64_t numeric_flags,
                                   bool use_prologue);
at::Tensor fused_combined3_forward(const at::Tensor& t, const at::Tensor& g,
                                   const at::Tensor& p, const at::Tensor& freq,
                                   int64_t num_channels, bool flip, double scale,
                                   const at::Tensor& tw1, const at::Tensor& tb1,
                                   const at::Tensor& tw2, const at::Tensor& tb2,
                                   const at::Tensor& gw1, const at::Tensor& gb1,
                                   const at::Tensor& gw2, const at::Tensor& gb2,
                                   const at::Tensor& pw1, const at::Tensor& pb1,
                                   const at::Tensor& pw2, const at::Tensor& pb2, int64_t wpr1,
                                   int64_t rpb1, int64_t wpr2, int64_t rpb2, int64_t unroll2,
                                   int64_t trig_mode, int64_t numeric_flags,
                                   bool use_prologue);
"""

# Built at import time, never lazily inside forward. ninja spawns subprocesses,
# and while the harness's no-new-threads guard brackets only the timing window
# and so would not actually catch a first-call build, building at import keeps
# compilation out of the timed region entirely and happens while the worker is
# still producing output, clear of the no-output stall watchdog.
EXTENSION_STATUS = ""
try:
    _EXT = _load_extension(name=os.environ.get(_EXT_NAME_ENV) or _EXTENSION_NAME,
                           build_dir=os.environ.get(_BUILD_DIR_ENV) or None)
except Exception as exc:  # noqa: BLE001 - a build failure must stay importable
    _EXT = None
    EXTENSION_STATUS = f"{type(exc).__name__}: {exc}"
    print(
        f"[candidate L2/timestep_embedding] CUDA extension unavailable, "
        f"delegating to the baseline body: {EXTENSION_STATUS}",
        file=sys.stderr,
        flush=True,
    )


# ---------------------------------------------------------------------------
# Declining has to be a faithful module substitution, not just a numeric one.
# ---------------------------------------------------------------------------
# A declined call must be indistinguishable from the baseline in every way a caller can
# observe, and output bits are only part of that. The baseline's `TimestepEmbedding`
# calls `self.linear_1(x)`, `self.act(x)`, `self.linear_2(x)`, so anyone who registers a
# forward pre-hook, a forward hook or a full backward hook on those public submodules
# sees it fire. Round 1 replaced those calls with `F.linear`/`F.silu` to get the
# arithmetic exact, which silently stopped the hooks firing: with the same
# value-changing forward hook on `baseline.linear_1` and `candidate.linear_1`, a declined
# call returned different bits.
#
# So the fallback calls the child modules, exactly as the baseline does -- and `self.act`
# becomes the module below rather than the frozen L1 one, because calling a module whose
# arithmetic is not the baseline's would trade one half of the guarantee for the other.
class SiLU(nn.Module):
    """`F.silu`, as a module, so `self.act(x)` is the baseline's call and its value.

    The baseline's `act` is `fastkernels.tasks.baseline.L1.silu.SiLU`, whose `forward`
    is exactly this. Parameterless, so `state_dict()` is unchanged and AC-1's key
    equality is untouched.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)


_FORCE_BASELINE = threading.local()


def _forcing_baseline() -> bool:
    return getattr(_FORCE_BASELINE, "depth", 0) > 0


@contextlib.contextmanager
def _force_baseline():
    """Make every candidate module in *this thread* take its baseline body.

    A declined composite has to run its verbatim baseline body through the ordinary
    `self.time_proj(...)` and `self.timestep_embedder(...)` calls, so hooks on those
    public submodules fire as they do on the baseline -- but those children must not then
    take their own fused paths, or the composite's "declined" result would be assembled
    from fused parts.

    Thread-local and re-entrant rather than a module-level boolean: a process-global flag
    would be visible to concurrent forwards on other threads, and nesting (a composite
    whose child is itself asked to force) would clear it too early.
    """
    depth = getattr(_FORCE_BASELINE, "depth", 0)
    _FORCE_BASELINE.depth = depth + 1
    try:
        yield
    finally:
        _FORCE_BASELINE.depth = depth


# ---------------------------------------------------------------------------
# Baseline reference implementation, kept verbatim.
# ---------------------------------------------------------------------------
def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """Sinusoidal timestep embedding (DDPM-style)."""
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device,
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


# ---------------------------------------------------------------------------
# Predicates.
#
# The MLP and composite predicates live in the .cu file, not here, and every
# entry point reports a decline *in band* by returning an undefined tensor --
# ``None`` on this side -- so ``forward`` is one call plus a ``is None`` test.
#
# That is not premature. With the fast path in place the TimestepEmbedding cases
# are DRAM-bound: 26.6 us of cold device time against the baseline's 28.8
# (``profile/p2_geometry/cold_probe.log``), so the candidate already streams
# faster than cuBLAS and the entire margin is the host-visible part of the harness
# window -- 0.93 us of it, for the baseline. An equivalent predicate written in
# Python measured 3.3 us on its own (``profile/p2_geometry/host_probe.log``),
# which is the whole result several times over. The same checks in C++ are free,
# and there is then exactly one copy of them: ``_admits`` below asks the
# extension, so the tested predicate and the live one cannot drift apart.
#
# Only ``Timesteps`` keeps a Python pre-filter, and only for the two things the
# extension cannot be asked: whether it exists at all, and whether to build the
# frequency table before handing it over.
# ---------------------------------------------------------------------------


def _record(key: tuple) -> None:
    _FASTPATH_HITS[key] = _FASTPATH_HITS.get(key, 0) + 1


def fastpath_hits() -> dict[tuple, int]:
    """A copy of the host-side hit counters. Reading these syncs nothing."""
    return dict(_FASTPATH_HITS)


def reset_fastpath_hits() -> None:
    _FASTPATH_HITS.clear()


# ---------------------------------------------------------------------------
# Modules.
# ---------------------------------------------------------------------------
class Timesteps(nn.Module):
    """Wraps get_timestep_embedding as an nn.Module."""

    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        # Non-persistent so it stays out of state_dict(), and a buffer rather
        # than an nn.Parameter so _sanitize_float_params -- which rewrites any
        # high-precision parameter whose magnitude leaves (1e-6, 1e4) -- cannot
        # reach it. exp(exponent) for num_channels 256 spans 1.0 down to 1.1e-4,
        # so as a parameter it would be rewritten to noise.
        #
        # Left None here and filled on first use because the table has to be
        # built on the device it will be read from: the harness constructs on CPU
        # and only then calls .to(device), and a CPU-built-then-moved table is
        # not guaranteed to be the same fp32 bits as the baseline's own
        # device-side torch.exp.
        self.register_buffer("_freq_table", None, persistent=False)
        # What the cached table was built for. Plain attributes, not buffers: the
        # cache is only valid while the configuration it was derived from is
        # unchanged, and the baseline recomputes from these attributes on every
        # call. Raised by the task25 review (docs/reviews/task25_codex_review.md),
        # which measured a 0.802 maximum error after mutating
        # downscale_freq_shift on a module that had already run a forward.
        self._freq_half = -1
        self._freq_shift = None
        half_dim = num_channels // 2
        self._static_ok = (
            num_channels > 0
            and num_channels % 2 == 0                        # odd widths get an F.pad column
            and isinstance(scale, (int, float))
            and not isinstance(scale, bool)
            and math.isfinite(float(scale))
            and isinstance(flip_sin_to_cos, bool)
            and isinstance(downscale_freq_shift, (int, float))
            and (half_dim - downscale_freq_shift) != 0       # the baseline divides by zero here
        )

    def _apply(self, *args, **kwargs):
        """Invalidate the cached table before any device or dtype transform.

        ``Module._apply`` is what ``.to()``, ``.cuda()``, ``.cpu()`` and ``.float()``
        all go through, and it *migrates registered buffers* -- so after
        ``mod.to("cuda:1")`` the cached table's ``.device`` matches the new device
        while the table itself is a copy of one built on the old one. The device check
        in ``frequencies`` cannot tell those apart, so the cache is dropped here and
        the next forward rebuilds from scalars on the device it will be read from.
        Raised by the round-0 review.
        """
        self._freq_table = None
        self._freq_half = -1
        self._freq_shift = None
        return super()._apply(*args, **kwargs)

    def frequencies(self, device: torch.device) -> torch.Tensor | None:
        """The fp32 frequency table for *device*, built with the baseline's own
        operation sequence and cached. ``None`` where the baseline itself divides
        by zero, which the caller must treat as a decline.

        Derived from ``__init__`` scalars only. Nothing here reads a weight: the
        harness rebinds every parameter's ``.data`` and then loads the
        baseline's state_dict, so anything precomputed from a parameter in
        ``__init__`` would be stale by the first forward.

        The cache is keyed on the device, the dtype, *and* the two scalars the
        table is derived from. Device and dtype because ``Module.to`` can change
        either after the table exists -- a cast would leave the extension
        declining every call instead of the table healing itself. The scalars
        because the baseline recomputes the table from the current attribute
        values on every call, so a module whose ``downscale_freq_shift`` is
        mutated after its first forward must not keep answering from the old one.
        """
        half_dim = self.num_channels // 2
        shift = self.downscale_freq_shift
        table = self._freq_table
        if (table is not None and table.device == device
                and table.dtype is torch.float32
                and self._freq_half == half_dim and self._freq_shift == shift):
            return table
        if half_dim - shift == 0:
            return None  # the baseline divides by zero here; let it own that
        exponent = -math.log(_MAX_PERIOD) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=device,
        )
        exponent = exponent / (half_dim - shift)
        table = torch.exp(exponent)
        self._freq_table = table
        self._freq_half = half_dim
        self._freq_shift = shift
        return table

    def _admits(self, timesteps: torch.Tensor) -> bool:
        """Would the fused path claim this call?

        The cheap part is here because it also decides whether to build the
        frequency table; the rest is the extension's own predicate, asked
        directly so the tested guard and the live one are the same code.
        """
        if _EXT is None or not self._static_ok or not timesteps.is_cuda:
            return False
        freq = self.frequencies(timesteps.device)
        if freq is None:
            return False
        return _EXT.sinusoid_admits(timesteps, freq, self.num_channels)

    def _baseline_forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return get_timestep_embedding(
            timesteps, self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if (_EXT is not None and not _forcing_baseline() and self._static_ok
                and timesteps.is_cuda):
            freq = self.frequencies(timesteps.device)
            if freq is not None:
                out = _EXT.sinusoid_forward(
                    timesteps, freq, self.num_channels, self.flip_sin_to_cos,
                    float(self.scale), _SINUSOID_BLOCK, TRIG_SINCOS,
                )
                if out is not None:
                    _record(("Timesteps", self.num_channels, timesteps.numel()))
                    return out
        return self._baseline_forward(timesteps)


class TimestepEmbedding(nn.Module):
    """Two-layer MLP that projects sinusoidal timestep encodings."""

    def __init__(self, in_channels: int, time_embed_dim: int, act_fn: str = "silu"):
        super().__init__()
        self.linear_1 = Linear(in_channels, time_embed_dim, bias=True)
        self.act = SiLU()
        self.linear_2 = Linear(time_embed_dim, time_embed_dim, bias=True)

    def _admits(self, sample: torch.Tensor) -> bool:
        """Would the fused path claim this call? Asks the extension's predicate."""
        if _EXT is None:
            return False
        lin1, lin2 = self.linear_1, self.linear_2
        b1, b2 = lin1.bias, lin2.bias
        if b1 is None or b2 is None:
            return False
        return _EXT.mlp_admits(sample, lin1.weight, b1, lin2.weight, b2)

    def _baseline_forward(self, sample: torch.Tensor) -> torch.Tensor:
        """The baseline body, verbatim -- three child module calls in its order.

        Calling the modules rather than the functions is the point: it is what makes a
        declined call a faithful substitution, hooks included. ``self.act`` is the exact
        ``SiLU`` defined above, and ``self.linear_1``/``self.linear_2`` are the frozen
        ``candidate/L1`` ``Linear``, which delegates every call to ``F.linear`` --
        asserted over a dense sample in ``tests/test_fallback.py`` rather than assumed.
        """
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        if _EXT is not None and not _forcing_baseline():
            lin1, lin2 = self.linear_1, self.linear_2
            b1, b2 = lin1.bias, lin2.bias
            # bias=False is the one thing the extension cannot be handed: pybind
            # has no None to bind to a Tensor reference.
            if b1 is not None and b2 is not None:
                w1 = lin1.weight
                out = _EXT.fused_mlp_forward(sample, w1, b1, lin2.weight, b2, *_GEOMETRY, 0)
                if out is not None:
                    shape = w1.shape
                    _record(("TimestepEmbedding", shape[1], shape[0]))
                    return out
        return self._baseline_forward(sample)


class _CombinedBase(nn.Module):
    """Shared helper for the two ``Combined*`` composites.

    Both are the same fused computation over a different branch count, so both
    resolve their branch parameters the same way and hand them to the extension
    in the baseline's summation order, sinusoid branches first.
    """

    def _branch_weights(self, embedders):
        groups = []
        for emb in embedders:
            lin1, lin2 = emb.linear_1, emb.linear_2
            b1, b2 = lin1.bias, lin2.bias
            if b1 is None or b2 is None:
                return None
            groups.append((lin1.weight, b1, lin2.weight, b2))
        return groups


class CombinedTimestepTextProjEmbeddings(_CombinedBase):
    """Combines sinusoidal timestep encoding with pooled text projection.

    Produces ``timestep_embedder`` + ``text_embedder`` weight names matching
    the diffusers checkpoint layout.
    """

    def __init__(self, embedding_dim: int, pooled_projection_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = TimestepEmbedding(in_channels=pooled_projection_dim, time_embed_dim=embedding_dim)

    def _admits(self, timestep: torch.Tensor, pooled_projection: torch.Tensor) -> bool:
        """Would the fused path claim this call? Asks the extension's predicate."""
        groups = self._branch_weights((self.timestep_embedder, self.text_embedder))
        proj = self.time_proj
        if _EXT is None or groups is None or not proj._static_ok \
                or not pooled_projection.is_cuda:
            return False
        freq = proj.frequencies(pooled_projection.device)
        if freq is None:
            return False
        return _EXT.combined2_admits(timestep, pooled_projection, freq,
                                     proj.num_channels, *groups[0], *groups[1])

    def _baseline_forward(self, timestep: torch.Tensor, pooled_projection: torch.Tensor) -> torch.Tensor:
        # The baseline's body, through the ordinary child calls, with the children
        # pinned to their own baseline bodies for the duration.
        with _force_baseline():
            timesteps_proj = self.time_proj(timestep)
            timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))
            pooled_projections = self.text_embedder(pooled_projection)
            return timesteps_emb + pooled_projections

    def forward(self, timestep: torch.Tensor, pooled_projection: torch.Tensor) -> torch.Tensor:
        proj = self.time_proj
        if (_EXT is not None and not _forcing_baseline() and proj._static_ok
                and pooled_projection.is_cuda):
            groups = self._branch_weights((self.timestep_embedder, self.text_embedder))
            freq = proj.frequencies(pooled_projection.device)
            if groups is not None and freq is not None:
                out = _EXT.fused_combined2_forward(
                    timestep, pooled_projection, freq,
                    proj.num_channels, proj.flip_sin_to_cos, float(proj.scale),
                    *groups[0], *groups[1], *_GEOMETRY, TRIG_SINCOS, 0, USE_PROLOGUE,
                )
                if out is not None:
                    shape = groups[1][0].shape
                    _record(("CombinedTimestepTextProjEmbeddings", proj.num_channels,
                             shape[1], shape[0]))
                    return out
        return self._baseline_forward(timestep, pooled_projection)


class CombinedTimestepGuidanceTextProjEmbeddings(_CombinedBase):
    """Combines sinusoidal timestep + guidance encoding with pooled text projection.

    Adds a ``guidance_embedder`` on top of
    :class:`CombinedTimestepTextProjEmbeddings`.  Weight names match the
    diffusers checkpoint layout.
    """

    def __init__(self, embedding_dim: int, pooled_projection_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.guidance_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = TimestepEmbedding(in_channels=pooled_projection_dim, time_embed_dim=embedding_dim)

    def _admits(self, timestep: torch.Tensor, guidance: torch.Tensor,
                pooled_projection: torch.Tensor) -> bool:
        """Would the fused path claim this call? Asks the extension's predicate."""
        groups = self._branch_weights((self.timestep_embedder, self.guidance_embedder,
                                       self.text_embedder))
        proj = self.time_proj
        if _EXT is None or groups is None or not proj._static_ok \
                or not pooled_projection.is_cuda:
            return False
        freq = proj.frequencies(pooled_projection.device)
        if freq is None:
            return False
        return _EXT.combined3_admits(timestep, guidance, pooled_projection, freq,
                                     proj.num_channels, *groups[0], *groups[1], *groups[2])

    def _baseline_forward(self, timestep: torch.Tensor, guidance: torch.Tensor, pooled_projection: torch.Tensor) -> torch.Tensor:
        with _force_baseline():
            timesteps_proj = self.time_proj(timestep)
            timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))
            guidance_proj = self.time_proj(guidance)
            guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=pooled_projection.dtype))
            pooled_projections = self.text_embedder(pooled_projection)
            return timesteps_emb + guidance_emb + pooled_projections

    def forward(self, timestep: torch.Tensor, guidance: torch.Tensor, pooled_projection: torch.Tensor) -> torch.Tensor:
        proj = self.time_proj
        if (_EXT is not None and not _forcing_baseline() and proj._static_ok
                and pooled_projection.is_cuda):
            groups = self._branch_weights((self.timestep_embedder, self.guidance_embedder,
                                           self.text_embedder))
            freq = proj.frequencies(pooled_projection.device)
            if groups is not None and freq is not None:
                out = _EXT.fused_combined3_forward(
                    timestep, guidance, pooled_projection, freq, proj.num_channels,
                    proj.flip_sin_to_cos, float(proj.scale),
                    *groups[0], *groups[1], *groups[2], *_GEOMETRY, TRIG_SINCOS, 0,
                    USE_PROLOGUE,
                )
                if out is not None:
                    shape = groups[2][0].shape
                    _record(("CombinedTimestepGuidanceTextProjEmbeddings",
                             proj.num_channels, shape[1], shape[0]))
                    return out
        return self._baseline_forward(timestep, guidance, pooled_projection)
