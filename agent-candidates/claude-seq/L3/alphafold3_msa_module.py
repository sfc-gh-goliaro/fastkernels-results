"""MSA module for AlphaFold3 -- the whole 4-block stack as one graph replay.

4-block MSA module: each block runs OPM -> MSA row attention -> transition ->
PairBlock.

Reference: openfold3/core/model/latent/msa_module.py MSAModuleStack

Where the time goes
-------------------
Every sub-layer here is already a single fused kernel (the frozen L2 winners), so
what is left to remove at this level is *between* the kernels, not inside them.
The eager stack issues 24 launches carrying ~220 us of GPU work, but measures
~410 us end to end: at N_res=16 / N_seq=8 no kernel runs long enough to cover the
next launch, so nearly half the latency is dispatch and the stream bubbles it
leaves.  Three changes address that, none of them touching the arithmetic.

**One launch instead of 24.**  The stack is captured into a CUDA graph and
replayed, which submits the whole dependency chain at once and lets the driver
stay ahead of the GPU.  Replay needs fixed addresses, so the four inputs are
staged into static buffers and the two outputs copied back out -- both by
``multicopy`` in ``alphafold3_msa_module.cu``, one launch per direction, because
at 80 KB total the copies are pure launch overhead (``_foreach_copy_`` measured
4.6 us a side, the fused kernel ~1.3 us).  Copying the outputs rather than
handing back the graph's private buffers means a caller's result cannot be
mutated by the next call.

**Two streams.**  With ``opm_first`` the block body is
``z += opm(m); m += att(m, z); m += trans(m); z = pair(z)`` -- so once ``z`` has
absorbed the outer-product mean, the MSA update and the pair stack read it but
never each other.  The pair stack is a 16-CTA cluster on a 148-SM device and the
MSA side is three small kernels, so the MSA path (plus the *next* block's
outer-product mean, which needs only the fresh ``m``) runs on a side stream and
disappears inside the pair stack: ~20 us hidden under ~38 us, per block.

**Submission order.**  Once forked, the pair stack has to be enqueued *before*
the side branch it is concurrent with.  It is a cluster launch needing 179 KB of
shared memory on each of 16 SMs, and a trace showed it waiting ~7 us per block
behind side-branch kernels that had been queued first.  Recording the fork event
before the pair stack but enqueueing the branch after it keeps the dependency
identical and removes that wait.

Together these take the captured case from ~410 us to ~225 us with bit-identical
output.  The pair stacks are then ~90% of the critical path and the rest of the
graph runs gap-free; going further means beating the L2 pair-block kernel, which
resisted distributed shared memory for its cross-CTA exchange, a cluster-scope
fence, ``ldmatrix`` A-fragments, a deeper weight pipeline, two deeper-ILP GEMM
schedules, and 1024 threads with a k-split (all within +-13%, none a win).

Anything the graph cannot serve -- grad enabled, a CPU or non-contiguous input,
an already-capturing stream, a shape whose sub-kernels fall back to their eager
reference path, or a capture that fails -- runs the unchanged eager sequence
instead, and the block sequence, submodule names and numerics are the baseline's
either way.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L2.alphafold3_msa_attention import MSARowAttentionWithPairBias
from ..L2.alphafold3_outer_product_mean import OuterProductMean
from ..L2.alphafold3_pair_block import PairBlock
from ..L2.alphafold3_swiglu_transition import SwiGLUTransition

from ....infra.cuda_ext import lazy_op

_C = lazy_op("fk_l3_af3_msa_module_io", "alphafold3_msa_module.cu",
             extra_cuda_cflags=["-arch=sm_100a", "--use_fast_math"])


__targets__ = ["MSAModuleStack"]


def _copy16(srcs, dsts):
    """Stage ``srcs`` into ``dsts`` in one launch.

    ``dsts`` is always allocator memory (512-byte aligned), but ``srcs`` may be a
    caller-supplied view, so the 16-byte requirement is re-checked per call --
    four ``data_ptr()`` reads, against a kernel that is entirely launch bound.
    """
    for t in srcs:
        if t.data_ptr() % 16:
            torch._foreach_copy_(dsts, srcs)
            return
    _C.multicopy(srcs, dsts)


def _pick_copier(tensors):
    """``_copy16``, or ``_foreach_copy_`` when the fast path cannot serve.

    ``multicopy`` moves whole 16-byte words, so every buffer's byte count has to
    be a multiple of 16 -- true of the captured shapes, but a property of the
    shape, so it is settled once when the graph is built. A missing nvcc or an
    unexpected build failure also falls back rather than raising.
    """
    if any(t.numel() * t.element_size() % 16 for t in tensors):
        return torch._foreach_copy_
    try:
        _C.multicopy([tensors[0]], [tensors[0].clone()])
    except Exception:
        return torch._foreach_copy_
    return _copy16


class MSAModuleBlock(nn.Module):
    """Single block of AF3 Algorithm 8.

    Args:
        c_m: MSA channel dimension
        c_z: Pair channel dimension
        c_hidden_msa_att: Hidden dim in MSA attention
        c_hidden_opm: Hidden dim in outer product mean
        c_hidden_mul: Hidden dim in triangle multiplication
        c_hidden_pair_att: Hidden dim in triangle attention
        no_heads_msa: Heads for MSA attention
        no_heads_pair: Heads for triangle attention
        transition_n: Transition layer scale
        msa_dropout: MSA dropout rate
        pair_dropout: Pair dropout rate
        opm_first: Whether OPM comes before MSA attention
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        transition_n: int,
        msa_dropout: float = 0.0,
        pair_dropout: float = 0.0,
        opm_first: bool = True,
        fuse_projection_weights: bool = False,
        inf: float = 1e9,
        eps: float = 1e-3,
        last_block: bool = False,
    ):
        super().__init__()
        self.opm_first = opm_first
        self.skip_msa_update = last_block and opm_first

        if not self.skip_msa_update:
            self.msa_att_row = MSARowAttentionWithPairBias(
                c_m=c_m, c_z=c_z,
                c_hidden=c_hidden_msa_att,
                no_heads=no_heads_msa,
                inf=inf,
            )

            self.msa_transition = SwiGLUTransition(c_in=c_m, n=transition_n)

        self.outer_product_mean = OuterProductMean(
            c_m=c_m, c_z=c_z, c_hidden=c_hidden_opm, eps=eps,
        )

        self.pair_stack = PairBlock(
            c_z=c_z,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_pair=no_heads_pair,
            transition_n=transition_n,
            pair_dropout=pair_dropout,
            inf=inf,
        )

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m:        [*, N_seq, N_res, C_m] MSA embedding
            z:        [*, N_res, N_res, C_z] pair embedding
            msa_mask: [*, N_seq, N_res] MSA mask
            pair_mask:[*, N_res, N_res] pair mask

        Returns:
            (m, z): updated MSA and pair embeddings
        """
        if self.opm_first:
            z = z + self.outer_product_mean(m, mask=msa_mask)

        if not self.skip_msa_update:
            m = m + self.msa_att_row(m, z=z, mask=pair_mask)
            m = m + self.msa_transition(m)

        if not self.opm_first:
            z = z + self.outer_product_mean(m, mask=msa_mask)

        z = self.pair_stack(z=z, pair_mask=pair_mask)

        return m, z


class MSAModuleStack(nn.Module):
    """AF3 Algorithm 8: MSA module stack.

    Args:
        c_m: MSA channel dimension
        c_z: Pair channel dimension
        c_hidden_msa_att: Hidden dim in MSA attention
        c_hidden_opm: Hidden dim in outer product mean
        c_hidden_mul: Hidden dim in triangle multiplication
        c_hidden_pair_att: Hidden dim in triangle attention
        no_heads_msa: Heads for MSA attention
        no_heads_pair: Heads for triangle attention
        no_blocks: Number of MSA module blocks
        transition_n: Transition scale
        opm_first: Whether OPM comes before MSA attention
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        no_blocks: int,
        transition_n: int,
        msa_dropout: float = 0.0,
        pair_dropout: float = 0.0,
        opm_first: bool = True,
        fuse_projection_weights: bool = False,
        blocks_per_ckpt: int | None = None,
        inf: float = 1e9,
        eps: float = 1e-3,
        **kwargs,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            MSAModuleBlock(
                c_m=c_m, c_z=c_z,
                c_hidden_msa_att=c_hidden_msa_att,
                c_hidden_opm=c_hidden_opm,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att,
                no_heads_msa=no_heads_msa,
                no_heads_pair=no_heads_pair,
                transition_n=transition_n,
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                opm_first=opm_first,
                inf=inf,
                eps=eps,
                last_block=(i == no_blocks - 1),
            )
            for i in range(no_blocks)
        ])

        # Captured graph state: (graph, static inputs, graph outputs) plus the
        # (shape, dtype) key it was built for. Built on the first eligible call,
        # after weights are in place; dropped whenever they could have changed,
        # i.e. on a state-dict load or any ``_apply`` (``to``, ``cuda``, ...).
        self._graph = None
        self._graph_key = None
        self._graph_off = False   # capture failed once -> stop retrying
        self._fork = None         # side stream the captured graph forks onto
        self.register_load_state_dict_post_hook(MSAModuleStack._drop_graph)

    @staticmethod
    def _drop_graph(module, incompatible_keys=None):
        module._graph = None
        module._graph_key = None

    def _apply(self, *args, **kwargs):
        self._graph = None
        self._graph_key = None
        return super()._apply(*args, **kwargs)

    # -- the unchanged eager sequence --------------------------------------
    def _blocks(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            m, z = block(m=m, z=z, msa_mask=msa_mask, pair_mask=pair_mask)

        return m, z

    def _fused_ready(self) -> bool:
        """Is every pair stack on its fused kernel for the shape just run?

        ``PairBlock`` builds its packed weights the first time it takes the fused
        path and leaves ``_packed`` unset otherwise, so this reads whether the
        warmup pass actually used it.  Only then is a graph worth building: on the
        eager reference path the stack expands to several hundred ops whose
        temporaries all have to be threaded through the graph's private pool,
        which costs far more to capture than the shape -- unsupported by the
        frozen kernels, so not the one this module is tuned for -- can win back.
        """
        return all(getattr(b.pair_stack, "_packed", None) is not None
                   for b in self.blocks)

    def _forkable(self) -> bool:
        """Is every block the ``opm_first`` shape the two-stream order assumes?"""
        n = len(self.blocks)
        return n > 0 and all(
            b.opm_first and b.skip_msa_update == (i == n - 1)
            for i, b in enumerate(self.blocks)
        )

    def _blocks_forked(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        side: torch.cuda.Stream,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same arithmetic as ``_blocks``, on two streams (see the module docstring).

        Every add keeps the baseline's operand order -- the pair output is the
        ``z`` the baseline adds the next outer-product mean to -- so the result is
        bit-identical, not merely within tolerance.
        """
        main = torch.cuda.current_stream()
        blocks = self.blocks
        last = len(blocks) - 1

        z = z + blocks[0].outer_product_mean(m, mask=msa_mask)

        for i, block in enumerate(blocks):
            work = not block.skip_msa_update
            if work:
                # fork at this z, but enqueue the branch after the pair stack so
                # the cluster launch is not stuck behind it
                side.wait_stream(main)
            z_next = block.pair_stack(z=z, pair_mask=pair_mask)
            opm_next = None
            if work:
                with torch.cuda.stream(side):
                    m = m + block.msa_att_row(m, z=z, mask=pair_mask)
                    m = m + block.msa_transition(m)
                    if i < last:
                        opm_next = blocks[i + 1].outer_product_mean(
                            m, mask=msa_mask)
                # the side branch's tensors outlive its stream
                m.record_stream(main)
                if opm_next is not None:
                    opm_next.record_stream(main)
                main.wait_stream(side)
            z = z_next if opm_next is None else z_next + opm_next

        return m, z

    @torch.no_grad()
    def _build_graph(self, inputs, key):
        """Warm up, decide whether a graph is worth it, then capture one."""
        static = [t.detach().clone() for t in inputs]
        orig = torch.cuda.current_stream()

        # First pass plain and on the caller's stream: it is exactly the eager
        # fallback, so a shape the frozen kernels do not fuse costs one ordinary
        # forward here and nothing more. It also settles which path each
        # sub-kernel took.
        self._blocks(*static)
        if not self._fused_ready():
            # Remember the key so this shape is answered eagerly from now on
            # rather than re-deciding on every call.
            self._graph, self._graph_key = None, key
            return

        # Two more passes, now in the shape that will be captured, so every lazy
        # init the sub-kernels do -- JIT compile, weight packing, allocator
        # growth -- happens before the capture, which cannot tolerate any of it.
        fork = torch.cuda.Stream() if self._forkable() else None
        warm = torch.cuda.Stream()
        warm.wait_stream(orig)
        with torch.cuda.stream(warm):
            for _ in range(2):
                self._body(static, fork)
        orig.wait_stream(warm)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out_m, out_z = self._body(static, fork)
        outs = [out_m, out_z]
        self._graph = (graph, static, outs, _pick_copier(static + outs))
        self._graph_key = key
        self._fork = fork   # keep the capture stream alive alongside its graph

    def _body(self, static, fork):
        if fork is None:
            return self._blocks(*static)
        return self._blocks_forked(*static, fork)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m:        [*, N_seq, N_res, C_m] MSA embedding
            z:        [*, N_res, N_res, C_z] pair embedding
            msa_mask: [*, N_seq, N_res] MSA mask
            pair_mask:[*, N_res, N_res] pair mask

        Returns:
            (m, z): updated MSA and pair embeddings
        """
        inputs = (m, z, msa_mask, pair_mask)
        if (not self._graph_off
                and not torch.is_grad_enabled()
                and all(isinstance(t, torch.Tensor) and t.is_cuda
                        and t.is_contiguous() for t in inputs)
                and not torch.cuda.is_current_stream_capturing()):
            key = tuple((tuple(t.shape), t.dtype) for t in inputs)
            if self._graph_key != key:
                try:
                    self._build_graph(inputs, key)
                except Exception:
                    self._graph = None
                    self._graph_key = None
                    self._graph_off = True
            entry = self._graph
            if entry is not None:
                graph, static, outs, copy = entry
                copy(list(inputs), static)
                graph.replay()
                fresh = [torch.empty_like(t) for t in outs]
                copy(outs, fresh)
                return fresh[0], fresh[1]

        return self._blocks(m, z, msa_mask, pair_mask)
