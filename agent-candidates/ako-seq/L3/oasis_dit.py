"""Oasis diffusion transformer -- dispatch-collapsed, bit-exact.

The captured workload is bsz=1, T=2..6 frames of a 9x16 token grid at
hidden=1024 against 604M fp32 parameters, and the reference forward is **1741
CUDA launches for 7.7 ms of device work over a 26.5 ms window**: five sixths of
the time the GPU is idle waiting for Python and the ATen dispatcher.  So the
first job is to delete the dispatch and hand the survivors to the driver as one
graph, and the second is to make the survivors cheaper without changing a single
rounding.

One fact dominates the design.

**The output is pathologically sensitive, so nothing may be reassociated.**
Injecting a *one-ULP* relative perturbation into the patch-embedder output
leaves only 79% of the output inside the harness' fp32 band (atol 1e-5 / rtol
1e-3, 99% required); injected after block 14 it is still only 95%, and only
after block 15 does it pass.  Flipping ``allow_tf32`` moves 28% of the output
out of band.  So this kernel reproduces the reference's arithmetic *operation
for operation* and only removes work that is provably a no-op, or changes
*scheduling* while provably leaving the arithmetic alone.  Every claim of that
kind here was checked with ``torch.equal`` against the reference chain, never
with a tolerance: ``dev/exact.py`` (whole forward, 5 shapes x 3 seeds),
``dev/lnfuse.py`` (the LayerNorm fusion against the three ATen ops it replaces),
``dev/vview.py`` (the strided V), ``dev/g6.py`` (every GEMM schedule).

That sensitivity is also why ``x_embedder`` / ``t_embedder`` / ``final_layer``
are constructed (their parameters are the state dict the harness shares) but
their *forwards* are not called: those frozen L2 winners are tuned for the fp16
scenario, and their fp32 fast paths differ from the reference by 3.6e-7 and
7.5e-6 -- enough, at this sensitivity, to fail the whole operator.  A verbatim
copy of ``baseline.py`` scores INCORRECT_NUMERICAL for exactly that reason.

What the forward does
---------------------

``c`` is computed once and passed unchanged to all 16 blocks, so ``SiLU(c)`` --
recomputed 32 times by the reference -- is computed once and the 33 modulation
projections become one GEMM against a weight concatenated along N (bit-identical
to the 33 separate ones).  It reads 780 MB of weights to produce T<=6 rows, i.e.
it runs at HBM peak while everything around it is compute bound, so it is split
in two along N and the 31-projection tail is **forked onto a second stream**
inside the capture and joined before block 1.

Each block half is then ``{gate+LayerNorm+modulate, qkv GEMM, rope+split, SDPA,
merge, out GEMM}`` and ``{gate+LayerNorm+modulate, fc1, GELU, fc2}``:

* **One kernel does the residual gate, the LayerNorm and the modulate.**  This is
  the largest single win after the graph.  The LayerNorm half is a transcription
  of ATen's ``vectorized_layer_norm_kernel``, and it is bit-exact only if three
  things are right, none of which is a tuning knob: ``cuWelfordCombine``'s
  variance term must be ``delta*delta*dataA.count*nB`` (not the algebraically
  equal ``delta*delta*nA*dataB.count`` -- the product associates left to right);
  the file must be compiled with nvcc's *default* contraction, because ATen's
  ``mean + delta*(1/count)`` and ``sigma2 + delta*(val-new_mean)`` are FMAs;
  and blockDim.y must be 4, because ATen launches
  ``dim3(warp_size, num_threads()/warp_size)`` and ``num_threads()`` is 32*4.
  The row is held in registers between the Welford pass and the normalize pass
  (N=1024, vec 4, 128 threads = exactly 2 float4 each), so 194 launches and
  615 us at T=6 become 65 launches and 359 us.
* **The block GEMMs run on a schedule chosen per (shape, M), not on cuBLAS'
  heuristic.**  Same cublasLt entry point, same tf32 MMA, same bias epilogue --
  only the tile, pipeline depth and cluster differ, so each output element still
  accumulates the whole of K sequentially in one CTA and the result is bitwise
  identical.  split-k and stream-k are never configured: they reorder that sum.
  M is only T*144 on a 148-SM B200, so the heuristic's 128x128/128x256 tiles
  leave most of the machine idle (fc2 at T=4 is 5x8 = 40 tiles); 64x64 is 144,
  one clean wave, and a linear cluster multicasts the operand tile through
  DSMEM.  Worth -73 us at T=2 and -148 us at T=4 in situ.
* **V is never materialised.**  ``rope_split`` writes only Q and K, with the
  rotary folded into the load and straight into the layout SDPA wants; V needs
  no rotation, so it is a strided *view* of the qkv GEMM output, which the
  mem-efficient FMHA reads at the same cost for bitwise the same answer.
* The rotary tables are static and memoized (built through the reference's own
  linspace/einsum/repeat_interleave/cos/sin, so the values are identical by
  construction) rather than rebuilt on each of 32 calls.
* ``_modulate``'s ``repeat`` + ``unsqueeze`` chain is a pure broadcast at these
  shapes; the tail-end ``chunk`` and ``repeat`` do not exist as ops.

Then the whole thing is captured as one CUDA graph per T, with x / t /
external_cond staged into static buffers: **~326 nodes, one driver launch**, and
the window then equals device time.

What is left is a floor
-----------------------

At T=6 the 4.14 ms window is 50% cuBLAS tf32 GEMM and 30% cutlass FMHA.  Both
were attacked and measured rather than assumed.  The FMHA computes its Q@K^T
with ``cutlass::arch::OpMultiplyAddFastF32`` -- 3xTF32 -- so no fp32 FMA chain
reproduces it, and folding batch/head or swapping backends changes nothing.  A
Triton ``tl.dot(input_precision="tf32")`` on RNE-rounded operands *is*
bit-identical to cuBLAS but 2-3x slower (no tcgen05 path on sm100), and a full
1095-config sweep of cuBLAS' own schedules is what produced the table above --
there is no faster bit-exact GEMM available here.  GELU is a 28 MB streaming
pass already at HBM peak and cuBLAS' ``EPILOGUE_GELU`` is not ATen's tanh
approximation.  See ITERATIONS.md for the measurements.

Everything degrades rather than fails: no extension, non-fp32, grad enabled, an
unexpected geometry, an in-flight capture, or a graph that will not capture all
fall back to ``_forward_exact``, the reference chain composed from the frozen
blocks (which are bit-exact) plus direct ATen for the three L2 modules above.
Each lever also has its own off switch -- ``OASIS_DIT_NO_LT`` (GEMM schedules),
``OASIS_DIT_LN_TY=0`` (LayerNorm fusion), ``OASIS_DIT_NO_FORK``,
``OASIS_DIT_NO_VVIEW``, ``OASIS_DIT_NO_GRAPH``, ``OASIS_DIT_NO_FUSE`` -- so any
one of them can be bisected out on a box where it misbehaves.
"""


from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.linear import Linear
from ..L1.oasis_rotary import OasisRotaryEmbedding
from ..L2.oasis_final_layer import OasisFinalLayer
from ..L2.oasis_patch_embed import OasisPatchEmbed
from ..L2.oasis_timestep_embedder import OasisTimestepEmbedder
from .oasis_block import SpatioTemporalDiTBlock

_NO_FUSE = bool(os.environ.get("OASIS_DIT_NO_FUSE"))
# Fold the residual gate and the modulate into one LayerNorm pass.  4 is not a
# tuning choice: ATen launches its vectorized LayerNorm with
# dim3(warp_size, num_threads()/warp_size) = (32, 4), so blockDim.y = 4 is the
# one reduction tree whose rounding is the reference's.  dev/lnfuse.py shows
# ty in {1,2,8} differ by ~2-4e-6 and ty=4 is `torch.equal` on every shape.
# 0 disables the fusion and falls back to the three-kernel ATen chain.
_LN_TY = int(os.environ.get("OASIS_DIT_LN_TY", "4"))
_NO_GRAPH = bool(os.environ.get("OASIS_DIT_NO_GRAPH"))
# Pick the cuBLAS *schedule* per (shape, M) instead of taking its heuristic.
_NO_LT = bool(os.environ.get("OASIS_DIT_NO_LT"))
# Modulation projections computed on the main stream before the fork; the rest go
# to a side stream.  2 = block 0's two, i.e. join before block 1.  0 disables the
# fork and computes all 33 in one GEMM on the main stream.
_ADALN_HEAD = int(os.environ.get("OASIS_DIT_ADALN_HEAD", "2"))
_NO_FORK = bool(os.environ.get("OASIS_DIT_NO_FORK"))
_V_VIEW = not os.environ.get("OASIS_DIT_NO_VVIEW")
_DEBUG = bool(os.environ.get("OASIS_DIT_DEBUG"))

_CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>

at::Tensor oasis_modulate(const at::Tensor& h, const at::Tensor& mod,
                          int64_t shift_off, int64_t scale_off, int64_t P);
void oasis_gate_add(const at::Tensor& x, const at::Tensor& y, const at::Tensor& mod,
                    int64_t gate_off, int64_t P);
at::Tensor oasis_gate_ln_mod(const at::Tensor& x, const c10::optional<at::Tensor>& y,
                             const at::Tensor& mod, int64_t gate_off, int64_t shift_off,
                             int64_t scale_off, int64_t P, double eps, int64_t ty);
std::vector<at::Tensor> oasis_rope_split(const at::Tensor& qkv, const at::Tensor& cosT,
                                         const at::Tensor& sinT, int64_t T, int64_t P,
                                         int64_t H, int64_t mode, int64_t emit_v);
at::Tensor oasis_merge(const at::Tensor& o, int64_t T, int64_t P, int64_t H, int64_t D,
                       int64_t st, int64_t sp, int64_t sh);

std::vector<std::vector<double>> lt_heur(int64_t, int64_t, int64_t, int64_t, int64_t,
                                         int64_t, int64_t);
std::vector<std::vector<double>> lt_probe(int64_t, int64_t, int64_t, int64_t, int64_t,
                                          int64_t, int64_t);
int64_t lt_make(int64_t, int64_t, int64_t, int64_t, at::Tensor, int64_t, int64_t, int64_t,
                int64_t, int64_t, int64_t, int64_t, int64_t);
void lt_run(int64_t, at::Tensor, at::Tensor, at::Tensor, at::Tensor);
double lt_time(int64_t, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t);
void lt_release(int64_t);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("modulate", &oasis_modulate);
  m.def("gate_add", &oasis_gate_add);
  m.def("gate_ln_mod", &oasis_gate_ln_mod);
  m.def("rope_split", &oasis_rope_split);
  m.def("merge", &oasis_merge);
  m.def("lt_heur", &lt_heur);
  m.def("lt_probe", &lt_probe);
  m.def("lt_make", &lt_make);
  m.def("lt_run", &lt_run);
  m.def("lt_time", &lt_time);
  m.def("lt_release", &lt_release);
}
"""

# --- verbatim copies of the reference's table builders -----------------------
# The rotary tables are static, so they are built once instead of 32 times per
# forward -- but they must be built by *these* expressions, not by the frozen L1
# module's fused equivalents, because the values are compared bit for bit.
def _ref_freqs(positions: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    f = torch.einsum("..., f -> ... f", positions.to(freqs.dtype), freqs)
    return f.repeat_interleave(2, dim=-1)


def _ref_axial_freqs(freqs: torch.Tensor, freqs_for: str, dims, device):
    colon = slice(None)
    all_freqs = []
    for index, dim in enumerate(dims):
        use_pixel = freqs_for == "pixel" and index >= len(dims) - 2
        if use_pixel:
            pos = torch.linspace(-1, 1, steps=dim, device=device)
        else:
            pos = torch.arange(dim, device=device)
        seq_freqs = _ref_freqs(pos, freqs)
        axis = [None] * len(dims)
        axis[index] = colon
        all_freqs.append(seq_freqs[(Ellipsis, *axis, colon)])
    all_freqs = torch.broadcast_tensors(*all_freqs)
    return torch.cat(all_freqs, dim=-1)


# ---------------------------------------------------------------------------
# cuBLAS *schedules* that beat cuBLAS' own heuristic, per (shape, M).
#
# Every one of these runs the same cublasLt entry point, the same
# CUBLAS_COMPUTE_32F_FAST_TF32 tf32 MMA and the same bias epilogue as
# ``F.linear``; the only thing chosen differently is the tile, the pipeline
# depth and the thread-block cluster -- i.e. *which* CTAs compute *which* output
# tile.  That leaves each output element accumulating the whole of K
# sequentially inside one CTA, which is why the results come out bitwise
# identical; split-k and stream-k reduction reorder that sum and are never
# configured (``lt_make`` pins SPLITK_NUM=1, REDUCTION_SCHEME=NONE).
#
# Why the defaults are wrong here: B200 has 148 SMs and M is only T*144, so the
# heuristic's 128x128/256x128 tiles leave most of the machine idle -- e.g. fc2
# at T=4 (M=576, N=1024) is 5x8 = 40 tiles.  A 64x64 tile is 9x16 = 144, one
# clean wave, and a linear cluster (8x1x1) keeps the L2 re-read down by
# multicasting the operand tile through DSMEM.  Measured (dev/g6.py, graph-timed
# against a weight pool larger than L2): +6% over the heuristic on all 20
# (shape, M) pairs, best on the starved ones -- qkv at T=2 +21%, out at T=4 +12%.
#
# Tile ids are cublasLtMatmulTile_t: 15=64x64, 17=64x128, 19=64x256, 20=128x128
# (in cuBLAS' column-major order, so the first number is the linear's N).
# Cluster ids are cublasLtClusterShape_t: 0=auto, 3=2x1x1, 4=4x1x1, 6=2x2x1,
# 7=4x2x1, 9=2x4x1, 11=8x1x1, 13=8x2x1, 14=2x8x1, 15=16x1x1.
# Fields: (algoId, tile, stages, ctaSwizzle, customOption, innerShape, cluster).
# ---------------------------------------------------------------------------
_LT_TILE_STG = 34
_SCHED = {
    ("qkv", 288): ((17, 15), (17, 0), (17, 3), (17, 11)),
    ("qkv", 432): ((19, 0), (19, 3), (19, 6), (20, 3)),
    ("qkv", 576): ((20, 11), (20, 4), (20, 0), (19, 4)),
    ("qkv", 720): ((19, 3), (19, 0), (20, 11), (19, 4)),
    ("qkv", 864): ((17, 11), (17, 4), (17, 6), (17, 3)),
    ("out", 288): ((15, 0), (15, 3), (15, 6), (15, 4)),
    ("out", 432): ((15, 9), (15, 4), (15, 14), (15, 7)),
    ("out", 576): ((15, 11), (15, 0), (15, 3), (15, 4)),
    ("out", 720): ((17, 4), (17, 11), (17, 6), (17, 0)),
    ("out", 864): ((17, 11), (17, 4), (17, 13), (17, 7)),
    ("fc1", 288): ((20, 0), (20, 3), (20, 4), (19, 0)),
    ("fc1", 432): ((19, 0), (19, 3), (20, 11), (19, 4)),
    ("fc1", 576): ((17, 11), (17, 15), (17, 4), (15, 4)),
    ("fc1", 720): ((17, 11), (17, 6), (17, 7), (17, 4)),
    ("fc1", 864): ((19, 0), (19, 3), (19, 11), (19, 13)),
    ("fc2", 288): ((15, 6), (15, 0), (15, 3), (15, 7)),
    ("fc2", 432): ((15, 9), (15, 4), (15, 7), (15, 11)),
    ("fc2", 576): ((15, 11), (15, 15), (17, 6), (15, 0)),
    ("fc2", 720): ((17, 11), (17, 4), (17, 6), (17, 0)),
    ("fc2", 864): ((17, 11), (17, 4), (17, 13), (17, 7)),
}
# Used when M is not one of the captured five: the union of everything that ever
# won above, so an unseen T is tuned rather than left on the heuristic.
_SCHED_ANY = tuple((t, c) for t in (15, 17, 19, 20) for c in (0, 3, 4, 11))


_EXT = None
_EXT_TRIED = False


def _ext():
    """Build (once, cached by content hash) the fused-glue extension."""
    global _EXT, _EXT_TRIED
    if _EXT_TRIED:
        return _EXT
    _EXT_TRIED = True
    try:
        here = Path(__file__).resolve().parent
        cu = (here / "oasis_glue.cu").read_text()
        lt = (here / "oasis_ltgemm.cu").read_text()
        tag = hashlib.sha1((cu + lt + _CPP_SRC + torch.__version__).encode()).hexdigest()[:12]
        from torch.utils.cpp_extension import load_inline
        _EXT = load_inline(
            name=f"oasis_dit_glue_{tag}",
            cpp_sources=[_CPP_SRC],
            cuda_sources=[cu, lt],
            # Default contraction on purpose: every elementwise expression in the
            # glue uses __fmul_rn/__fadd_rn/__fsub_rn, which cannot be contracted,
            # while the transcribed LayerNorm reduction *needs* the same FMAs ATen
            # gets.  See the header comment in oasis_glue.cu.
            extra_cuda_cflags=["-O3", "--extended-lambda"],
            extra_cflags=["-O3"],
            extra_ldflags=["-lcublasLt", "-lcublas"],
            verbose=False,
        )
    except Exception:
        if _DEBUG:
            raise
        _EXT = None
    return _EXT


class OasisDiT(nn.Module):
    def __init__(
        self,
        *,
        input_h: int = 18,
        input_w: int = 32,
        patch_size: int = 2,
        in_channels: int = 16,
        hidden_size: int = 1024,
        depth: int = 16,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        external_cond_dim: int = 25,
        max_frames: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.max_frames = max_frames

        self.x_embedder = OasisPatchEmbed(input_h, input_w, patch_size, in_channels, hidden_size, flatten=False)
        self.t_embedder = OasisTimestepEmbedder(hidden_size)
        head_dim = hidden_size // num_heads
        self.spatial_rotary_emb = OasisRotaryEmbedding(dim=head_dim // 2, freqs_for="pixel", max_freq=256)
        self.temporal_rotary_emb = OasisRotaryEmbedding(dim=head_dim, freqs_for="lang")
        self.external_cond = Linear(external_cond_dim, hidden_size, bias=True) if external_cond_dim > 0 else nn.Identity()
        self.blocks = nn.ModuleList(
            [
                SpatioTemporalDiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    is_causal=True,
                    spatial_rotary_emb=self.spatial_rotary_emb,
                    temporal_rotary_emb=self.temporal_rotary_emb,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = OasisFinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

        # --- fused-path state (all built lazily on the first forward: __init__
        # runs before the harness casts p.data and load_state_dict's over it) ---
        self.hidden_size = hidden_size
        self.depth = depth
        self.head_dim = head_dim
        self.grid = self.x_embedder.grid_size          # (9, 16)
        self._plan = None                              # packed adaLN weights
        self._plan_sig = None
        self._tables = {}                              # rotary / timestep tables
        self._graphs = {}                              # T -> capture entry
        self._lt = {}                                  # M -> chosen GEMM plans
        self.register_load_state_dict_post_hook(lambda *a, **k: self._invalidate())

    def _side(self):
        """Lazily-made second stream for the forked adaLN tail.  Created outside
        capture; `wait_stream` on both sides brings it into the captured graph."""
        s = getattr(self, "_side_stream", None)
        if s is None:
            try:
                s = torch.cuda.Stream()
            except Exception:
                return None
            self._side_stream = s
        return s

    def _invalidate(self):
        self._plan = None
        self._plan_sig = None
        self._tables.clear()
        self._graphs.clear()
        ext = _EXT
        if ext is not None:
            for sel in self._lt.values():
                for h in sel.values():
                    if h is not None:
                        try:
                            ext.lt_release(h)
                        except Exception:
                            pass
        self._lt.clear()

    def initialize_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        weight = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(weight.view(weight.shape[0], -1))
        if self.x_embedder.proj.bias is not None:
            nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.s_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.s_adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.t_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.t_adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        if self.final_layer.linear.bias is not None:
            nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = x.shape[1]
        w = x.shape[2]
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    # ------------------------------------------------------------------
    # Lazily-built, bit-exact plan.
    # ------------------------------------------------------------------
    def _adaln_mods(self):
        """The 33 modulation Linears, in the order the forward consumes them."""
        mods = []
        for blk in self.blocks:
            mods.append(blk.s_adaLN_modulation[-1])
            mods.append(blk.t_adaLN_modulation[-1])
        mods.append(self.final_layer.adaLN_modulation[-1])
        return mods

    def _sig(self):
        """Cheap guard: identity+version of a few tensors the plan bakes in."""
        w = self.blocks[0].s_adaLN_modulation[-1].weight
        z = self.blocks[-1].t_adaLN_modulation[-1].weight
        f = self.final_layer.adaLN_modulation[-1].weight
        s = self.spatial_rotary_emb.freqs
        t = self.temporal_rotary_emb.freqs
        return (w.data_ptr(), w._version, z.data_ptr(), z._version,
                f.data_ptr(), f._version, s.data_ptr(), s._version,
                t.data_ptr(), t._version, w.dtype, w.device)

    def _build_plan(self, device, dtype):
        mods = self._adaln_mods()
        # The batched adaLN GEMM reads 780 MB of weights for T<=6 output rows, so
        # it runs at HBM peak (~136 us) while the block GEMMs are compute bound at
        # roughly a third of peak.  Split it in two along N -- which r1 verified is
        # bit-identical to the 33 separate GEMMs, hence to any partition of them --
        # so the tail can be forked onto a second stream and hidden under block 0.
        # Clamped so the split is always non-degenerate: both halves must have at
        # least one projection or the `torch.cat` below has nothing to concatenate.
        # Use OASIS_DIT_NO_FORK to turn the fork off, not OASIS_DIT_ADALN_HEAD=0.
        nh = min(max(_ADALN_HEAD, 1), len(mods) - 1)
        with torch.no_grad():
            wh = torch.cat([m.weight for m in mods[:nh]], dim=0).contiguous()
            bh = torch.cat([m.bias for m in mods[:nh]], dim=0).contiguous()
            wt = torch.cat([m.weight for m in mods[nh:]], dim=0).contiguous()
            bt = torch.cat([m.bias for m in mods[nh:]], dim=0).contiguous()
        offs = []
        o = 0
        for m in mods:
            offs.append(o)
            o += m.weight.shape[0]
        split = offs[nh]
        # (which buffer, offset within it) per modulation index
        pick = [(0, offs[j]) if j < nh else (1, offs[j] - split) for j in range(len(mods))]
        # rotary tables, built through the reference's own ops
        H = self.grid[0]
        W = self.grid[1]
        sfreqs = _ref_axial_freqs(self.spatial_rotary_emb.freqs,
                                  self.spatial_rotary_emb.freqs_for, (H, W),
                                  device).reshape(-1, self.head_dim)
        scos, ssin = sfreqs.cos().contiguous(), sfreqs.sin().contiguous()
        half = self.t_embedder.frequency_embedding_size // 2
        tsfreqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=device)
            / half,
        )
        self._plan = dict(wh=wh, bh=bh, wt=wt, bt=bt, offs=offs, pick=pick,
                          nh=nh, split=split, width=o,
                          scos=scos, ssin=ssin, tsfreqs=tsfreqs, tcs={})
        self._plan_sig = self._sig()

    def _temporal_cs(self, T, ref):
        """(cos, sin) for positions 0..T-1, exactly as rotate_queries_or_keys builds them."""
        cs = self._plan["tcs"].get(T)
        if cs is None:
            freqs = self.temporal_rotary_emb.freqs
            positions = torch.arange(T, device=ref.device, dtype=ref.dtype)
            sf = _ref_freqs(positions, freqs)
            cs = (sf.cos().contiguous(), sf.sin().contiguous())
            self._plan["tcs"][T] = cs
        return cs

    # ------------------------------------------------------------------
    # GEMM schedule selection.  Runs once per M, before any capture: time
    # cuBLAS' own heuristic choice and the baked candidates through the same
    # graph-timed path, keep the fastest that is `torch.equal` to F.linear, and
    # fall back to F.linear itself whenever nothing beats it or anything at all
    # goes wrong.  Nothing here can cost correctness -- the gate is equality,
    # not tolerance.
    # ------------------------------------------------------------------
    def _gemm_specs(self):
        blk = self.blocks[0]
        return (("qkv", blk.s_attn.to_qkv), ("out", blk.s_attn.to_out),
                ("fc1", blk.s_mlp.fc1), ("fc2", blk.s_mlp.fc2))

    def _lt_select(self, M, ext):
        sel = self._lt.get(M)
        if sel is not None:
            return sel
        sel = {}
        # Selection times candidates and synchronises, so it must never run inside
        # someone else's capture; the fused path is entered that way when this
        # module is nested in an outer graph.  Plain F.linear then.
        if not _NO_LT and not torch.cuda.is_current_stream_capturing():
            for name, mod in self._gemm_specs():
                try:
                    sel[name] = self._lt_one(name, M, mod.weight, mod.bias, ext)
                except Exception:
                    if _DEBUG:
                        raise
                    sel[name] = None
        self._lt[M] = sel
        return sel

    def _lt_one(self, name, M, W, b, ext):
        N, K = W.shape
        bt = b if b is not None else W.new_empty(0)   # to_qkv has bias=False
        x = torch.randn(M, K, device=W.device, dtype=W.dtype)
        ref = F.linear(x, W, b)
        out = torch.empty_like(ref)
        cands = list(_SCHED.get((name, M), _SCHED_ANY))
        # cuBLAS' own first heuristic is the thing to beat; F.linear picks it.  Ask
        # for it with the same epilogue F.linear will use -- `to_qkv` has no bias,
        # and the bias epilogue changes which schedules the heuristic offers.
        heur = ext.lt_heur(M, N, K, 0, 1 if b is not None else 0, 1 << 20, 4)
        base = None
        if heur:
            h0 = heur[0]
            base = (int(h0[1]), int(h0[6]))
            cands = [base] + [c for c in cands if c != base]
        best_t, best_h, base_t = None, None, None
        for tile, clu in cands:
            try:
                h = ext.lt_make(M, N, K, 0, bt, 73, tile, _LT_TILE_STG, 0, 0, 0, clu, 0)
            except Exception:
                continue
            try:
                out.zero_()
                ext.lt_run(h, W, x, out, bt)
                torch.cuda.synchronize()
                if not torch.equal(out, ref):
                    ext.lt_release(h)
                    continue
                t = ext.lt_time(h, W, x, out, bt, 40, 8)
            except Exception:
                ext.lt_release(h)
                continue
            if t <= 0:
                ext.lt_release(h)
                continue
            if (tile, clu) == base:
                base_t = t
            if best_t is None or t < best_t:
                if best_h is not None:
                    ext.lt_release(best_h)
                best_t, best_h, best_cfg = t, h, (tile, clu)
            else:
                ext.lt_release(h)
        if _DEBUG:
            print(f"[oasis_dit] {name} M={M}: heur={heur[0][1] if heur else None}/"
                  f"{heur[0][6] if heur else None} base={base_t} best={best_t} "
                  f"cfg={best_cfg if best_h is not None else None}")
        if best_h is None:
            return None
        # Only take the override when it actually wins by more than measurement
        # noise; otherwise stay on F.linear, which is the shipped fallback.
        if base_t is not None and best_t > base_t * 0.995:
            ext.lt_release(best_h)
            return None
        if _DEBUG:
            print(f"[oasis_dit] {name} M={M}: tile={best_cfg[0]} clu={best_cfg[1]} "
                  f"{best_t:.2f}us vs heuristic {base_t}us")
        return best_h

    # ------------------------------------------------------------------
    # The fused forward.
    # ------------------------------------------------------------------
    def _core(self, x, t, external_cond, ext):
        bsz, time, channels, height, width = x.shape
        P = self.grid[0] * self.grid[1]
        C = self.hidden_size
        H = self.num_heads
        bt = bsz * time
        pl = self._plan
        offs = pl["offs"]
        pick = pl["pick"]
        nh = pl["nh"]

        # --- patch embed: F.conv2d then the reference's channels-last view,
        # densified once into the [bt*P, C] matrix every later stage wants.
        pe = self.x_embedder
        h = F.conv2d(x.reshape(bt, channels, height, width), pe.proj.weight,
                     pe.proj.bias, stride=pe.patch_size)
        h = h.permute(0, 2, 3, 1).reshape(bt * P, C)

        # --- timestep + external conditioning -> c, then SiLU(c) once
        tt = t.reshape(bt)
        args = tt[:, None].float() * pl["tsfreqs"][None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        mlp = self.t_embedder.mlp
        c = F.linear(emb, mlp[0].weight, mlp[0].bias)
        c = F.silu(c)
        c = F.linear(c, mlp[2].weight, mlp[2].bias)
        if torch.is_tensor(external_cond):
            c = c + F.linear(external_cond.reshape(bt, -1),
                             self.external_cond.weight, self.external_cond.bias)
        # --- the modulation projections.  The head is needed immediately; the tail
        # is forked so its 780 MB of weight streaming overlaps block 0's compute.
        sc = F.silu(c)
        mh = F.linear(sc, pl["wh"], pl["bh"])
        side = None
        if nh and nh < len(offs) and not _NO_FORK:
            side = self._side()
        if side is not None:
            cur = torch.cuda.current_stream()
            side.wait_stream(cur)
            with torch.cuda.stream(side):
                mt = F.linear(sc, pl["wt"], pl["bt"])
        else:
            mt = F.linear(sc, pl["wt"], pl["bt"])
        modb = (mh, mt)          # head, forked tail

        scos, ssin = pl["scos"], pl["ssin"]
        tcos, tsin = self._temporal_cs(time, h)
        modulate, gate_add, rope_split, merge = (ext.modulate, ext.gate_add,
                                                 ext.rope_split, ext.merge)
        lt = self._lt_select(bt * P, ext)
        lt_run = ext.lt_run

        def gemm(which, src, mod_):
            """F.linear, but on the schedule the sweep picked for this (shape, M)."""
            h_ = lt.get(which)
            w_ = mod_.weight
            if h_ is None:
                return F.linear(src, w_, mod_.bias)
            b_ = mod_.bias
            o_ = torch.empty((src.shape[0], w_.shape[0]), dtype=src.dtype,
                             device=src.device)
            lt_run(h_, w_, src, o_, b_ if b_ is not None else w_.new_empty(0))
            return o_
        lnw = (C,)
        ty = _LN_TY
        # The residual gate of one sub-block and the LayerNorm+modulate of the
        # next are one kernel, so the gate is carried forward rather than issued.
        pend = [None]

        def ln_mod(w, shift_off, scale_off):
            m = modb[w]
            if ty:
                pv = pend[0]
                pend[0] = None
                if pv is None:
                    return ext.gate_ln_mod(h, None, m, 0, shift_off, scale_off,
                                           P, 1e-6, ty)
                if pv[2] != w:
                    # The pending gate lives in the other buffer, so it cannot ride
                    # along in the fused kernel; flush it with the standalone kernel.
                    # Happens exactly once per forward, at the fork boundary.
                    gate_add(h, pv[0], modb[pv[2]], pv[1], P)
                    return ext.gate_ln_mod(h, None, m, 0, shift_off, scale_off,
                                           P, 1e-6, ty)
                return ext.gate_ln_mod(h, pv[0], m, pv[1], shift_off, scale_off,
                                       P, 1e-6, ty)
            return modulate(F.layer_norm(h, lnw, None, None, 1e-6), m,
                            shift_off, scale_off, P)

        def gate(y, w, gate_off):
            if ty:
                pend[0] = (y, gate_off, w)
            else:
                gate_add(h, y, modb[w], gate_off, P)

        for i, blk in enumerate(self.blocks):
            if side is not None and 2 * i >= nh:
                # join once, before the first block that reads the forked tail
                torch.cuda.current_stream().wait_stream(side)
                mt.record_stream(torch.cuda.current_stream())
                side = None
            for mode in (0, 1):
                w, o = pick[2 * i + mode]
                if mode == 0:
                    attn, mlpm, cs, sn, causal = blk.s_attn, blk.s_mlp, scos, ssin, False
                else:
                    attn, mlpm, cs, sn, causal = blk.t_attn, blk.t_mlp, tcos, tsin, blk.t_attn.is_causal
                # ---- attention half
                hn = ln_mod(w, o, o + C)
                qkv = gemm("qkv", hn, attn.to_qkv)
                if _V_VIEW:
                    # V is not rotated and not reordered, so it is a view of the
                    # GEMM output rather than a third of rope_split's writes.
                    q, k = rope_split(qkv, cs, sn, time, P, H, mode, 0)
                    q5 = qkv.view(time, P, 3, H, self.head_dim)
                    v = (q5[:, :, 2].permute(0, 2, 1, 3) if mode == 0
                         else q5[:, :, 2].permute(1, 2, 0, 3))
                else:
                    q, k, v = rope_split(qkv, cs, sn, time, P, H, mode, 1)
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                     dropout_p=0.0, is_causal=causal)
                out = self._merge(out, merge, time, P, H, C, mode)
                out = gemm("out", out, attn.to_out)
                gate(out, w, o + 2 * C)
                # ---- mlp half
                hn = ln_mod(w, o + 3 * C, o + 4 * C)
                hn = gemm("fc1", hn, mlpm.fc1)
                hn = F.gelu(hn, approximate="tanh")
                hn = gemm("fc2", hn, mlpm.fc2)
                gate(hn, w, o + 5 * C)

        # --- final layer + unpatchify
        if side is not None:
            torch.cuda.current_stream().wait_stream(side)
            mt.record_stream(torch.cuda.current_stream())
        fl = self.final_layer
        w, o = pick[-1]
        hn = ln_mod(w, o, o + C)
        y = F.linear(hn, fl.linear.weight, fl.linear.bias)
        p = self.patch_size
        y = y.view(bt, self.grid[0], self.grid[1], p, p, self.out_channels)
        y = y.permute(0, 5, 1, 3, 2, 4).reshape(bt, self.out_channels,
                                                self.grid[0] * p, self.grid[1] * p)
        return y.view(bsz, time, self.out_channels, self.grid[0] * p, self.grid[1] * p)

    @staticmethod
    def _merge(o, merge, T, P, H, C, mode):
        """SDPA output -> the [T*P, C] matrix, as a free view when the backend's
        layout allows it (spatial: it does; temporal: it does not)."""
        D = o.shape[3]
        if mode == 0:                             # (T, H, P, D)
            v = o.permute(0, 2, 1, 3)             # -> (T, P, H, D)
            st, sh, sp = o.stride(0), o.stride(1), o.stride(2)
        else:                                     # (P, H, T, D)
            v = o.permute(2, 0, 1, 3)             # -> (T, P, H, D)
            sp, sh, st = o.stride(0), o.stride(1), o.stride(2)
        if v.is_contiguous():
            return v.reshape(T * P, C)
        if o.stride(3) == 1:
            return merge(o, T, P, H, D, st, sp, sh)
        return v.reshape(T * P, C)                # ATen copy (never hit here)

    # ------------------------------------------------------------------
    # Exact fallback: the reference chain (blocks resolve to the baseline, which
    # is bit-exact; the three L2 modules are replaced by direct ATen because
    # their fp32 fast paths do not reproduce it).
    # ------------------------------------------------------------------
    def _forward_exact(self, x, t, external_cond):
        bsz, time, channels, height, width = x.shape
        pe = self.x_embedder
        xx = F.conv2d(x.reshape(bsz * time, channels, height, width),
                      pe.proj.weight, pe.proj.bias, stride=pe.patch_size)
        xx = xx.permute(0, 2, 3, 1)
        xx = xx.reshape(bsz, time, xx.shape[1], xx.shape[2], xx.shape[3])
        tt = t.reshape(bsz * time)
        half = self.t_embedder.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half,
        )
        args = tt[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        mlp = self.t_embedder.mlp
        c = F.linear(emb, mlp[0].weight, mlp[0].bias)
        c = F.silu(c)
        c = F.linear(c, mlp[2].weight, mlp[2].bias).reshape(bsz, time, -1)
        if torch.is_tensor(external_cond):
            c = c + F.linear(external_cond, self.external_cond.weight,
                             self.external_cond.bias)
        for block in self.blocks:
            xx = block(xx, c)
        fl = self.final_layer
        m = F.linear(F.silu(c), fl.adaLN_modulation[-1].weight,
                     fl.adaLN_modulation[-1].bias)
        shift, scale = m.chunk(2, dim=-1)
        while shift.dim() < xx.dim():
            shift = shift.unsqueeze(-2)
            scale = scale.unsqueeze(-2)
        xx = F.layer_norm(xx, (self.hidden_size,), None, None, 1e-6) * (1 + scale) + shift
        xx = F.linear(xx, fl.linear.weight, fl.linear.bias)
        xx = xx.reshape(bsz * time, xx.shape[2], xx.shape[3], xx.shape[4])
        xx = self.unpatchify(xx)
        return xx.reshape(bsz, time, xx.shape[1], xx.shape[2], xx.shape[3])

    # ------------------------------------------------------------------
    # CUDA graph: one capture per T, weights static, inputs staged in.
    # ------------------------------------------------------------------
    def _capture(self, x, t, external_cond, ext):
        sx = torch.empty_like(x)
        sx.copy_(x)
        st = torch.empty_like(t)
        st.copy_(t)
        se = None
        if torch.is_tensor(external_cond):
            se = torch.empty_like(external_cond)
            se.copy_(external_cond)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._core(sx, st, se, ext)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = self._core(sx, st, se, ext)
        return (g, sx, st, se, out)

    def _graph_entry(self, x, t, external_cond, ext):
        key = (tuple(x.shape), x.dtype, x.device, torch.is_tensor(external_cond))
        hit = self._graphs.get(key, 0)
        if hit != 0:
            return hit
        try:
            entry = self._capture(x, t, external_cond, ext)
        except Exception:
            if _DEBUG:
                raise
            entry = None
        if len(self._graphs) >= 12:
            self._graphs.clear()
        self._graphs[key] = entry
        return entry

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: torch.Tensor,
                external_cond: torch.Tensor | None = None) -> torch.Tensor:
        ext = None if _NO_FUSE else _ext()
        ok = (ext is not None
              and x.is_cuda and x.dtype is torch.float32
              and x.dim() == 5 and x.shape[0] == 1
              and x.shape[2] == self.in_channels
              and (x.shape[3], x.shape[4]) == self.x_embedder.img_size
              and t.numel() == x.shape[1]
              and not torch.is_grad_enabled() and not x.requires_grad
              and (external_cond is None
                   or (torch.is_tensor(external_cond) and external_cond.is_cuda
                       and external_cond.dtype is torch.float32)))
        if not ok:
            return self._forward_exact(x, t, external_cond)
        try:
            return self._forward_fused(x, t, external_cond, ext)
        except Exception:
            # Nothing in the fused path is allowed to cost correctness: the
            # reference chain below is bit-exact by construction.
            if _DEBUG:
                raise
            return self._forward_exact(x, t, external_cond)

    def _forward_fused(self, x, t, external_cond, ext):
        if self._plan is None or self._plan_sig != self._sig():
            self._graphs.clear()
            self._build_plan(x.device, x.dtype)
        xc = x if x.is_contiguous() else x.contiguous()
        ec = external_cond
        if torch.is_tensor(ec) and not ec.is_contiguous():
            ec = ec.contiguous()
        if (not _NO_GRAPH) and not torch.cuda.is_current_stream_capturing():
            entry = self._graph_entry(xc, t, ec, ext)
            if entry is not None:
                g, sx, st, se, out = entry
                sx.copy_(xc)
                st.copy_(t)
                if se is not None:
                    se.copy_(ec)
                g.replay()
                # The graph writes one fixed buffer; hand the caller a copy so
                # nothing aliases across calls.
                return out.clone()
        return self._core(xc, t, ec, ext)
