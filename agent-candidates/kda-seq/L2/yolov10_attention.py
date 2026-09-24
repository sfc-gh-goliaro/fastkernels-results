"""YOLOv10 spatial attention as three Triton kernels over BatchNorm-folded weights.

The baseline composes this block out of ~18 separate CUDA launches (three
Conv2d + BatchNorm2d pairs, a reshaping copy, two batched GEMMs, a softmax and a
residual add). At the captured configuration it issues only ~44 us of GPU work
inside a ~150 us window, so for the *baseline* the cost is the launch count, not
the arithmetic: ~407 MFLOP is a fraction of a microsecond of B200 tensor-core
time, and ~2.5 MB of traffic is a fraction of a microsecond of HBM time.

Collapsing each Conv2d+BatchNorm2d pair into one weight/bias pair -- an eval-mode
identity -- lets the whole dataflow fit in three kernels:

    qkv_proj          1x1 projection as a channel-contraction GEMM
    flash_attn        attention with online softmax; the n x n matrix is never
                      materialized
    pe_residual_proj  depthwise 3x3 + its bias + the residual add + the output
                      1x1 projection, in one pass

Cutting the launch count is where the speedup starts, but it is not where it
ends. Once there are only three launches, this operator stops being
dispatch-bound and becomes bound by the kernels themselves plus the host cost of
issuing them -- measured, at that point, at ~44 us of GPU time against ~41 us of
Python, overlapping inside a ~53 us window. So the tile sizes and warp counts
below are the ones an offline sweep picked (`scripts/tune_tiles.py`), the
addressing in the epilogue is written the way that lets Triton vectorize it, and
the two internal buffers are reused rather than reallocated per call. Profiling
records for both are under `profile/`.

The fold cannot happen in ``__init__``: the benchmark constructs the module,
*then* casts and overwrites its parameters, so at construction time the weights
are still uninitialized. It therefore happens on the first forward and is
invalidated whenever the weights, dtype, device or training mode change.

Correctness never depends on the kernels. Anything the fast path does not
support -- training mode, CPU tensors, float32, autograd, a non-3x3 positional
encoding, a missing or non-compiling Triton -- falls through to a reference
forward that reproduces the baseline's dataflow with the baseline's own
submodules.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn

from ..L1.softmax import Softmax
from .yolov10_conv import YOLOConv

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - Triton ships with CUDA torch builds
    triton = None
    tl = None

# tl.dot contracts over at least 16 elements and produces tiles at least 16
# wide, so every block size that feeds a dot is padded up to this and masked.
_MIN_DOT = 16

_LOG2E = math.log2(math.e)

# Triton specializes a pointer argument on whether it is 16-byte divisible, and
# bakes that decision into the compiled binary. Any launch that bypasses the
# usual argument binding has to re-establish it rather than assume it.
_POINTER_ALIGNMENT = 16


def _pad_grid(grid: tuple) -> tuple[int, int, int]:
    return (grid[0], grid[1] if len(grid) > 1 else 1,
            grid[2] if len(grid) > 2 else 1)


def _pad_for_dot(value: int) -> int:
    """Smallest power of two >= max(value, 16), for a masked tl.dot tile."""
    return max(_MIN_DOT, triton.next_power_of_2(value))


if triton is not None:

    @triton.jit
    def _qkv_proj_kernel(
        x_ptr,            # fp16/bf16 [batch, c_in, n]
        w_ptr,            # fp16/bf16 [c_out, c_in], rows already permuted to q|k|v
        b_ptr,            # fp32 [c_out]
        out_ptr,          # fp16/bf16 [batch, c_out, n]
        n,
        c_in,
        c_out,
        BLOCK_C: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """out[batch, co, i] = sum_ci w[co, ci] * x[batch, ci, i] + b[co]."""
        pid_batch = tl.program_id(0)
        pid_pix = tl.program_id(1)
        pid_chan = tl.program_id(2)

        offs_n = pid_pix * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_c = pid_chan * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_n = offs_n < n
        mask_c = offs_c < c_out

        x_base = x_ptr + pid_batch * c_in * n
        acc = tl.zeros((BLOCK_C, BLOCK_N), dtype=tl.float32)
        for k0 in tl.range(0, c_in, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < c_in
            w = tl.load(
                w_ptr + offs_c[:, None] * c_in + offs_k[None, :],
                mask=mask_c[:, None] & mask_k[None, :],
                other=0.0,
            )
            # Pixel-innermost on both operands, so every access is coalesced
            # along n.
            xt = tl.load(
                x_base + offs_k[:, None] * n + offs_n[None, :],
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
            )
            acc += tl.dot(w, xt)

        acc += tl.load(b_ptr + offs_c, mask=mask_c, other=0.0)[:, None]
        tl.store(
            out_ptr + pid_batch * c_out * n + offs_c[:, None] * n + offs_n[None, :],
            acc.to(out_ptr.dtype.element_ty),
            mask=mask_c[:, None] & mask_n[None, :],
        )

    @triton.jit
    def _flash_attn_kernel(
        qkv_ptr,          # fp16/bf16 [batch, c_qkv, n], laid out q | k | v
        out_ptr,          # fp16/bf16 [batch, num_heads * head_dim, n]
        n,
        num_heads,
        key_dim,
        head_dim,
        c_qkv,
        k_offset,         # first k row  = num_heads * key_dim
        v_offset,         # first v row  = 2 * num_heads * key_dim
        scale_log2e,      # key_dim ** -0.5 * log2(e), so exp2 replaces exp
        BLOCK_M: tl.constexpr,
        BLOCK_KEY: tl.constexpr,
        BLOCK_DK: tl.constexpr,
        BLOCK_DV: tl.constexpr,
    ):
        """Softmax attention over pixels, per (batch, head), fp32 accumulators.

        Queries and keys are stored channel-major (``[channel, pixel]``), which
        is the transpose of what the first dot wants, so the query tile is
        loaded coalesced along pixels and transposed in registers once.
        """
        pid_bh = tl.program_id(0)
        pid_m = tl.program_id(1)
        batch = pid_bh // num_heads
        head = pid_bh % num_heads

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_dk = tl.arange(0, BLOCK_DK)
        offs_dv = tl.arange(0, BLOCK_DV)
        mask_m = offs_m < n
        mask_dk = offs_dk < key_dim
        mask_dv = offs_dv < head_dim

        qkv_base = qkv_ptr + batch * c_qkv * n
        qt = tl.load(
            qkv_base + (head * key_dim + offs_dk)[:, None] * n + offs_m[None, :],
            mask=mask_dk[:, None] & mask_m[None, :],
            other=0.0,
        )
        q = tl.trans(qt)

        k_rows = k_offset + head * key_dim + offs_dk
        v_rows = v_offset + head * head_dim + offs_dv

        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)

        for j0 in tl.range(0, n, BLOCK_KEY):
            offs_j = j0 + tl.arange(0, BLOCK_KEY)
            mask_j = offs_j < n
            kt = tl.load(
                qkv_base + k_rows[:, None] * n + offs_j[None, :],
                mask=mask_dk[:, None] & mask_j[None, :],
                other=0.0,
            )
            logits = tl.dot(q, kt) * scale_log2e
            logits = tl.where(mask_j[None, :], logits, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(logits, 1))
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(logits - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]

            vt = tl.load(
                qkv_base + v_rows[:, None] * n + offs_j[None, :],
                mask=mask_dv[:, None] & mask_j[None, :],
                other=0.0,
            )
            acc += tl.dot(p.to(kt.dtype), tl.trans(vt))
            m_i = m_new

        acc = acc / l_i[:, None]
        tl.store(
            out_ptr
            + batch * (num_heads * head_dim) * n
            + (head * head_dim + offs_dv)[:, None] * n
            + offs_m[None, :],
            tl.trans(acc).to(out_ptr.dtype.element_ty),
            mask=mask_dv[:, None] & mask_m[None, :],
        )

    @triton.jit
    def _pe_residual_proj_kernel(
        qkv_ptr,          # fp16/bf16 [batch, c_qkv, n]; v starts at v_offset
        attn_ptr,         # fp16/bf16 [batch, dim, n], the attention output
        pe_w_ptr,         # fp32 [dim, 9]
        pe_b_ptr,         # fp32 [dim]
        proj_w_ptr,       # fp16/bf16 [dim, dim]
        proj_b_ptr,       # fp32 [dim]
        out_ptr,          # fp16/bf16 [batch, dim, n]
        n,
        img_h,
        img_w,
        dim,
        c_qkv,
        v_offset,
        BLOCK_DIM: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_OC: tl.constexpr,
    ):
        """The whole epilogue: depthwise 3x3 on v, its bias, the residual add,
        then the output 1x1 projection.

        This one kernel replaces six baseline launches -- the reshaping copy that
        materializes v as an image, the depthwise convolution, its BatchNorm, the
        residual add, the projection convolution and its BatchNorm.
        """
        pid_batch = tl.program_id(0)
        pid_pix = tl.program_id(1)

        offs_n = pid_pix * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_c = tl.arange(0, BLOCK_DIM)
        mask_n = offs_n < n
        mask_c = offs_c < dim

        row = offs_n // img_w
        col = offs_n % img_w

        v_base = qkv_ptr + pid_batch * c_qkv * n + (v_offset + offs_c)[:, None] * n
        acc = tl.zeros((BLOCK_DIM, BLOCK_N), dtype=tl.float32)
        for tap in tl.static_range(9):
            d_row = tap // 3 - 1
            d_col = tap % 3 - 1
            # The neighbour's flat pixel index is `offs_n + d_row * img_w + d_col`
            # -- affine in the pixel index, so this load vectorizes. Spelling the
            # same number as `(row + d_row) * img_w + (col + d_col)` hides that
            # affinity behind the `//` and `%`, and Triton then falls back to one
            # scalar load per element (~141 global load instructions per thread
            # instead of a handful of vector loads).
            shifted = offs_n + (d_row * img_w + d_col)
            # Zero padding: a tap that leaves the image contributes nothing. The
            # row/column decomposition is only needed for this mask, never for
            # the address. Masked lanes are not accessed, so the out-of-range
            # arithmetic at the image corners is never dereferenced.
            in_image = (
                mask_n
                & (row + d_row >= 0)
                & (row + d_row < img_h)
                & (col + d_col >= 0)
                & (col + d_col < img_w)
            )
            v = tl.load(
                v_base + shifted[None, :],
                mask=mask_c[:, None] & in_image[None, :],
                other=0.0,
            )
            tap_w = tl.load(pe_w_ptr + offs_c * 9 + tap, mask=mask_c, other=0.0)
            acc += tap_w[:, None] * v.to(tl.float32)

        acc += tl.load(pe_b_ptr + offs_c, mask=mask_c, other=0.0)[:, None]
        attn = tl.load(
            attn_ptr + pid_batch * dim * n + offs_c[:, None] * n + offs_n[None, :],
            mask=mask_c[:, None] & mask_n[None, :],
            other=0.0,
        )
        # Rows past `dim` stay zero, so they contribute nothing to the
        # contraction below and no separate masking of the dot is needed.
        y = (acc + attn.to(tl.float32)).to(proj_w_ptr.dtype.element_ty)

        out_base = out_ptr + pid_batch * dim * n
        for oc0 in tl.range(0, dim, BLOCK_OC):
            offs_oc = oc0 + tl.arange(0, BLOCK_OC)
            mask_oc = offs_oc < dim
            proj_w = tl.load(
                proj_w_ptr + offs_oc[:, None] * dim + offs_c[None, :],
                mask=mask_oc[:, None] & mask_c[None, :],
                other=0.0,
            )
            out = tl.dot(proj_w, y)
            out += tl.load(proj_b_ptr + offs_oc, mask=mask_oc, other=0.0)[:, None]
            tl.store(
                out_base + offs_oc[:, None] * n + offs_n[None, :],
                out.to(out_ptr.dtype.element_ty),
                mask=mask_oc[:, None] & mask_n[None, :],
            )


class _Launcher(NamedTuple):
    """A compiled kernel's launch entry point, hoisted out of ``JITFunction.run``.

    Calling ``kernel[grid](...)`` re-does per-call work that is identical every
    time once the cache is warm: binding arguments, building a cache key, looking
    the binary up, canonicalizing the grid, allocating a launch-metadata object
    and calling two empty hook chains. Going straight to the compiled kernel's
    ``run`` skips all of it and reaches the same binary with the same arguments --
    measured at ~6-7.5 us less host time per launch, which on a ~40 us operator
    with three launches is the single largest remaining lever.

    The correctness catch is that a compiled binary carries the specialization
    decided when it was compiled, including each pointer's 16-byte-divisibility
    flag and each integer's by-value class. So a descriptor is only ever used for
    the exact key it was built under, and the pointer alignment it assumed is
    re-checked on every launch rather than trusted.
    """

    run: object
    function: object
    metadata: object
    grid: tuple[int, int, int]


class _FoldedPack(NamedTuple):
    """Conv+BatchNorm folded into plain weights, with q/k/v made contiguous.

    Held as a plain attribute, never a registered buffer, so ``state_dict``
    stays exactly the baseline's.
    """

    qkv_w: torch.Tensor    # [c_qkv, dim], rows permuted so q | k | v
    qkv_b: torch.Tensor    # [c_qkv] fp32
    pe_w: torch.Tensor     # [dim, 9] fp32
    pe_b: torch.Tensor     # [dim] fp32
    proj_w: torch.Tensor   # [dim, dim]
    proj_b: torch.Tensor   # [dim] fp32
    dtype: torch.dtype
    device: torch.device


class _LaunchPlan(NamedTuple):
    """Grids and block sizes for one (batch, height, width). Cached per shape so
    the hot path does no arithmetic beyond a dict lookup."""

    grid_qkv: tuple[int, int, int]
    grid_attn: tuple[int, int]
    grid_epilogue: tuple[int, int]
    block_c: int
    block_n_qkv: int
    block_k: int
    block_m: int
    block_key: int
    block_dk: int
    block_dv: int
    block_dim: int
    block_n_epilogue: int
    block_oc: int
    warps_qkv: int
    warps_attn: int
    warps_epilogue: int


def _pair(value) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    first, second = tuple(value)
    return (int(first), int(second))


def _conv_geometry(conv: nn.Module):
    """``(kernel, stride, padding, dilation, groups)``, or None if unreadable.

    The Conv2d these blocks are built from stores ``stride``, ``padding``,
    ``dilation`` and ``groups`` but no ``kernel_size`` -- that has to come from
    the weight shape. Which Conv2d it actually is depends on whether a frozen
    lower-level winner is on the path, so nothing here is assumed: anything that
    cannot be read routes to the reference path instead of being guessed at.
    """
    try:
        shape = conv.weight.shape
        return (
            (int(shape[2]), int(shape[3])),
            _pair(conv.stride),
            _pair(conv.padding),
            _pair(conv.dilation),
            int(conv.groups),
        )
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


def _tensor_state(tensor: torch.Tensor | None):
    """(identity, version) of a tensor the fold depends on, or None.

    `_version` bumps on in-place writes, so this distinguishes "the same tensor,
    edited" from "the same tensor, untouched" -- which identity alone cannot.
    """
    if tensor is None:
        return None
    return (id(tensor), tensor._version)


def _bn_foldable(bn: nn.Module) -> bool:
    """Whether this BatchNorm has everything the eval-mode fold needs."""
    return (
        getattr(bn, "weight", None) is not None
        and getattr(bn, "bias", None) is not None
        and getattr(bn, "running_mean", None) is not None
        and getattr(bn, "running_var", None) is not None
        and isinstance(getattr(bn, "eps", None), float)
        and bool(getattr(bn, "track_running_stats", True))
    )


def _fold_conv_bn(conv: nn.Module, bn: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the eval-mode equivalent (weight, bias) of ``bn(conv(x))``, fp32.

    ``eps`` is read from the module rather than assumed: YOLOConv builds its
    BatchNorm with ``eps=1e-3``, not the torch default.
    """
    weight = conv.weight.detach().float()
    scale = bn.weight.detach().float() / torch.sqrt(
        bn.running_var.detach().float() + bn.eps
    )
    bias = bn.bias.detach().float() - bn.running_mean.detach().float() * scale
    if conv.bias is not None:
        bias = bias + conv.bias.detach().float() * scale
    return weight * scale.view(-1, *([1] * (weight.dim() - 1))), bias


class YOLOAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = YOLOConv(dim, h, 1, act=False)
        self.proj = YOLOConv(dim, dim, 1, act=False)
        self.pe = YOLOConv(dim, dim, 3, 1, g=dim, act=False)
        self._softmax = Softmax(dim=-1)

        self.dim = dim
        self.c_qkv = h
        self.k_offset = nh_kd
        self.v_offset = 2 * nh_kd

        # Nothing here touches a weight *value*: the benchmark casts and
        # overwrites parameters after construction, so anything derived from
        # them has to wait for the first forward.
        self._pack: _FoldedPack | None = None
        # The pack is built with ordinary torch ops on whatever stream is
        # current. Any *other* stream that later consumes it has to wait for
        # those ops to retire, so the creating stream's completion is recorded
        # and each consuming stream is synchronized against it once.
        self._pack_ready: torch.cuda.Event | None = None
        self._pack_streams: set = set()
        self._plans: dict[tuple, _LaunchPlan] = {}
        self._scratch: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self._launchers: dict[tuple, tuple[_Launcher, ...]] = {}
        # The direct launch path is implemented, proven byte-identical and
        # measured -- and switched off, on the evidence.
        #
        # It does what it claims: skipping `JITFunction.run` saves 5.7-7.6 us of
        # host time per launch, ~19 us across the three. End to end that buys
        # nothing, because the host work overlaps GPU execution and the GPU is
        # the critical path at this operating point: three back-to-back
        # measurements put the difference at -0.02 to +0.03 us
        # (`scripts/measure_launch_paths.py`). A cache keyed on compile-time
        # specialization is a real risk surface, so it is not worth carrying for
        # a gain that is not there.
        #
        # It stays because it becomes worth switching on the moment the kernels
        # get faster than the host -- flip this to True, and
        # `scripts/test_direct_launch.py` is the proof that it is safe to.
        self._direct_launch = False
        self._qkv_row_perm: torch.Tensor | None = None
        # Fingerprint of the module structure the support decision was made
        # against; None forces a probe on the first forward.
        self._structure_key: tuple | None = None
        self._fast_path_supported = False
        # Set when a kernel raises and the module permanently falls back, so a
        # silent degradation is still diagnosable after the fact.
        self._fast_path_error: BaseException | None = None
        self.register_load_state_dict_post_hook(_invalidate_on_load)

    # -- folded-weight cache lifetime -------------------------------------
    #
    # The pack is derived state, so every path that can change what it was
    # derived from has to drop it. The benchmark only mutates weights before the
    # first forward, so these exist for callers outside it and for the tests.
    #
    # Residual holes that no hook in this torch version can observe, and that
    # are therefore out of scope: ``param.data.copy_(...)``, ``param.data = ...``,
    # calling ``.eval()`` or ``.to()`` on a *child* module directly, and in-place
    # BatchNorm running-stat updates (which do not bump ``_version``).

    def _invalidate_pack(self) -> None:
        self._pack = None
        self._pack_ready = None
        self._pack_streams.clear()
        # Descriptor keys carry the pack's identity, so they can never match
        # again; drop them rather than leaving dead entries behind.
        self._launchers.clear()
        # These two are derived from the head split, not from the input shape, but
        # they are keyed only by shape and by device. Re-laying-out the heads --
        # say from (num_heads=2, head_dim=64, key_dim=32) to (4, 32, 16), which
        # keeps c_qkv at 256 and passes support revalidation -- leaves a plan with
        # the wrong attention grid and a permutation for the wrong layout. Both
        # produce finite, wrong output.
        self._plans.clear()
        self._qkv_row_perm = None
        # And the scratch, whose size depends on `c_qkv` and `dim`. Those are not
        # part of its cache key, so a structural change that grows the channel
        # count would otherwise hand back a buffer that is too small and the
        # projection kernel would write past it.
        self._scratch.clear()

    def _apply(self, fn, recurse: bool = True):
        # `fn` may move the module to another device or dtype, which invalidates
        # the reused scratch as well as the pack.
        self._invalidate_pack()
        self._scratch.clear()
        self._launchers.clear()
        return super()._apply(fn, recurse=recurse)

    def train(self, mode: bool = True):
        # eval() delegates here, so one override covers both directions.
        self._invalidate_pack()
        return super().train(mode)

    def _row_permutation(self, device: torch.device) -> torch.Tensor:
        """Rows of the folded qkv weight, reordered to q | k | v.

        The baseline's qkv channels interleave per head as
        ``[q0 k0 v0 q1 k1 v1 ...]``. Applying the reordering once to the folded
        weight -- instead of gathering activations on every forward -- puts q, k
        and v each in one contiguous block, which is what lets the attention and
        epilogue kernels address them with no index arithmetic.
        """
        cached = self._qkv_row_perm
        if cached is not None and cached.device == device:
            return cached
        nh, kd, hd = self.num_heads, self.key_dim, self.head_dim
        inner = 2 * kd + hd
        heads = torch.arange(nh, device=device).unsqueeze(1) * inner
        perm = torch.cat(
            (
                (heads + torch.arange(kd, device=device)).reshape(-1),
                (heads + kd + torch.arange(kd, device=device)).reshape(-1),
                (heads + 2 * kd + torch.arange(hd, device=device)).reshape(-1),
            )
        )
        self._qkv_row_perm = perm
        return perm

    def _folded_pack(self, dtype: torch.dtype, device: torch.device,
                     stream: torch.cuda.Stream | None = None) -> _FoldedPack:
        if stream is None:
            stream = torch.cuda.current_stream(device)
        pack = self._pack
        if pack is not None and pack.dtype == dtype and pack.device == device:
            if stream not in self._pack_streams:
                # First time this stream reads a pack built elsewhere: wait for
                # the building work, and tell the allocator this stream now holds
                # a reference. Without `record_stream` the blocks could be freed
                # (by an `eval()` or a weight load) and handed out again on the
                # creating stream while this stream's kernels are still queued
                # against them.
                stream.wait_event(self._pack_ready)
                self._record_pack_stream(pack, stream)
            return pack
        with torch.no_grad():
            qkv_w, qkv_b = _fold_conv_bn(self.qkv.conv, self.qkv.bn)
            pe_w, pe_b = _fold_conv_bn(self.pe.conv, self.pe.bn)
            proj_w, proj_b = _fold_conv_bn(self.proj.conv, self.proj.bn)
            perm = self._row_permutation(qkv_w.device)
            pack = _FoldedPack(
                qkv_w=qkv_w.reshape(self.c_qkv, self.dim)[perm].to(dtype).contiguous(),
                qkv_b=qkv_b[perm].contiguous(),
                # Only the tensor-core contractions need the activation dtype;
                # the depthwise taps are a plain multiply-accumulate, so keeping
                # them fp32 costs nothing and drops a rounding step.
                pe_w=pe_w.reshape(self.dim, 9).contiguous(),
                pe_b=pe_b.contiguous(),
                proj_w=proj_w.reshape(self.dim, self.dim).to(dtype).contiguous(),
                proj_b=proj_b.contiguous(),
                dtype=dtype,
                device=device,
            )
        self._pack = pack
        self._pack_ready = torch.cuda.Event()
        self._pack_ready.record(stream)
        self._pack_streams = set()
        self._record_pack_stream(pack, stream)
        return pack

    @staticmethod
    def _pack_tensors(pack: _FoldedPack):
        return (pack.qkv_w, pack.qkv_b, pack.pe_w, pack.pe_b, pack.proj_w,
                pack.proj_b)

    def _record_pack_stream(self, pack: _FoldedPack,
                            stream: torch.cuda.Stream) -> None:
        for tensor in self._pack_tensors(pack):
            tensor.record_stream(stream)
        self._pack_streams.add(stream)

    # -- routing ----------------------------------------------------------

    def _structure(self) -> tuple:
        """A fingerprint of everything ``_probe_static_support`` decides on.

        The support decision cannot simply be cached once. These blocks are
        public attributes and so is everything inside them, so a caller can fuse
        a block, swap ``pe`` for a different geometry, or replace an activation
        *after* a successful fast forward. A latched "supported" answer would
        then keep running the kernels -- against a folded pack built from the old
        weights -- which is silent wrong output rather than a loud failure.

        So the fingerprint is recomputed every forward and compared. It covers
        the identity of each block and its children (which catches replacement)
        and every property the probe reads (which catches in-place mutation of a
        child that was not itself replaced, such as ``block.fuse()``).

        It deliberately does not try to detect ``param.data.copy_(...)`` or
        ``param.data = ...``; those are the documented residual holes that no
        mechanism in this torch version catches, and the benchmark performs all
        of its weight mutation before the first forward.
        """
        parts = [
            self.dim, self.num_heads, self.key_dim, self.head_dim, self.c_qkv,
            self.k_offset, self.v_offset, self.scale,
        ]
        for block in (self.qkv, self.proj, self.pe):
            conv = getattr(block, "conv", None)
            bn = getattr(block, "bn", None)
            act = getattr(block, "act", None)
            weight = getattr(conv, "weight", None)
            parts.append((
                id(block),
                id(conv),
                id(bn),
                id(act),
                type(act),
                bool(getattr(block, "_is_fused", False)),
                getattr(bn, "eps", None),
                bool(getattr(bn, "track_running_stats", True)),
                _conv_geometry(conv),
                # Every tensor the fold reads, by identity *and* version. Identity
                # catches a replaced Parameter or a swapped submodule; the version
                # counter catches an in-place edit of one that was not replaced,
                # such as `bn.bias.data.fill_(...)`. Leaving `conv.bias`,
                # `bn.bias` or `bn.running_mean` out of this tuple is enough to
                # serve a stale fold: each of them shifts the folded bias.
                _tensor_state(weight),
                _tensor_state(getattr(conv, "bias", None)),
                _tensor_state(getattr(bn, "weight", None)),
                _tensor_state(getattr(bn, "bias", None)),
                _tensor_state(getattr(bn, "running_mean", None)),
                _tensor_state(getattr(bn, "running_var", None)),
            ))
        return tuple(parts)

    def _static_support(self) -> bool:
        """Whether the fast path applies, re-checked against the fingerprint."""
        structure = self._structure()
        if structure != self._structure_key:
            # Anything the probe looks at has moved, so the folded pack derived
            # from it is stale too.
            self._invalidate_pack()
            self._structure_key = structure
            self._fast_path_supported = self._probe_static_support()
        return self._fast_path_supported

    def _probe_static_support(self) -> bool:
        if triton is None:
            return False
        if not (self.num_heads >= 1 and self.key_dim >= 1 and self.head_dim >= 1):
            return False
        if self.num_heads * self.head_dim != self.dim:
            return False
        if self.c_qkv != self.dim + 2 * self.num_heads * self.key_dim:
            return False
        # Every tile that feeds a tl.dot is padded up to 16 and masked, so a
        # small dim or key_dim is fine; a dim that is not a power of two is fine
        # too. What is not fine is a fused block (its BatchNorm is gone) or a
        # convolution whose geometry these kernels do not implement.
        for block in (self.qkv, self.proj, self.pe):
            if getattr(block, "_is_fused", False):
                return False
            if not isinstance(getattr(block, "act", None), nn.Identity):
                return False
            if not hasattr(block, "bn") or not _bn_foldable(block.bn):
                return False
        geometry = [_conv_geometry(b.conv) for b in (self.qkv, self.proj, self.pe)]
        if any(g is None for g in geometry):
            return False
        qkv_g, proj_g, pe_g = geometry
        point_wise = ((1, 1), (1, 1), (0, 0), (1, 1), 1)
        return (
            qkv_g == point_wise
            and proj_g == point_wise
            and pe_g == ((3, 3), (1, 1), (1, 1), (1, 1), self.dim)
        )

    def _launch_plan(self, batch: int, img_h: int, img_w: int) -> _LaunchPlan:
        key = (batch, img_h, img_w)
        plan = self._plans.get(key)
        if plan is not None:
            return plan
        n = img_h * img_w
        nh, kd, hd = self.num_heads, self.key_dim, self.head_dim

        # Latency here is one program's critical path, not throughput: at these
        # sizes every program is resident in a single wave on 148 SMs, so there
        # is no tail to amortize. The caps below were picked by sweeping tiles and
        # warp counts offline (`scripts/tune_tiles.py`) and are pinned rather than
        # autotuned, because autotuning would benchmark inside the timed region.
        #
        # The sweep's shape: narrow pixel tiles win everywhere (the spread across
        # configurations is 7 us to 500 us, so this is not a marginal choice), but
        # the attention key tile wants to be *wide* -- 256 rather than 64 -- since
        # each key tile costs a dot, an online-softmax rescale and another dot in
        # sequence, and fewer iterations means a shorter dependency chain.
        block_n_qkv = min(32, _pad_for_dot(n))
        block_c = min(32, _pad_for_dot(self.c_qkv))
        block_k = min(128, _pad_for_dot(self.dim))
        block_m = min(16, _pad_for_dot(n))
        block_key = min(256, _pad_for_dot(n))
        block_dk = _pad_for_dot(kd)
        block_dv = _pad_for_dot(hd)
        block_dim = _pad_for_dot(self.dim)
        block_n_epi = min(16, _pad_for_dot(n))
        block_oc = min(128, _pad_for_dot(self.dim))

        plan = _LaunchPlan(
            grid_qkv=(
                batch,
                triton.cdiv(n, block_n_qkv),
                triton.cdiv(self.c_qkv, block_c),
            ),
            grid_attn=(batch * nh, triton.cdiv(n, block_m)),
            grid_epilogue=(batch, triton.cdiv(n, block_n_epi)),
            block_c=block_c,
            block_n_qkv=block_n_qkv,
            block_k=block_k,
            block_m=block_m,
            block_key=block_key,
            block_dk=block_dk,
            block_dv=block_dv,
            block_dim=block_dim,
            block_n_epilogue=block_n_epi,
            block_oc=block_oc,
            warps_qkv=8,
            warps_attn=4,
            warps_epilogue=8,
        )
        self._plans[key] = plan
        return plan

    def _use_fast_path(self, x: torch.Tensor) -> bool:
        return (
            not self.training
            and x.is_cuda
            and x.dim() == 4
            and x.dtype in (torch.float16, torch.bfloat16)
            and x.shape[1] == self.dim
            and x.shape[2] * x.shape[3] > 0
            and not x.requires_grad
            and not torch.is_grad_enabled()
            and self._static_support()
        )

    # -- forward ----------------------------------------------------------

    def _workspace(self, batch: int, n: int, dtype: torch.dtype,
                   device: torch.device,
                   stream: torch.cuda.Stream | None = None,
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        """The two intermediates, reused across calls on the same stream.

        `qkv` and the attention output are internal scratch: each is written in
        full before it is read and neither escapes the forward, so keeping them
        instead of reallocating saves a measured ~2.6 us of host time per
        allocation -- real money against a ~40 us operator. The returned output
        tensor is always freshly allocated, since that one does escape.

        Reuse is per stream. Sharing one buffer across streams would let two
        concurrent forwards of the same module instance overwrite each other's
        intermediates, and the baseline it has to match allocates independent
        intermediates every call. Keying on the stream keeps the saving while
        restoring that contract: work enqueued on one stream is ordered against
        itself, so a buffer is only ever reused after the previous forward that
        used it has retired.

        The key also carries `c_qkv` and `dim`, which decide the buffer sizes.
        """
        if stream is None:
            stream = torch.cuda.current_stream(device)
        # `c_qkv` and `dim` are in the key, not just in the sizes: they are what
        # the buffers are shaped by. Keying only on the input shape means a
        # structural change to the head split silently reuses a mis-sized buffer,
        # and clearing on invalidation alone would leave that one bug away from
        # an out-of-bounds write.
        key = (batch, n, self.c_qkv, self.dim, dtype, device, stream)
        buffers = self._scratch.get(key)
        if buffers is None:
            buffers = (
                torch.empty((batch, self.c_qkv, n), dtype=dtype, device=device),
                torch.empty((batch, self.dim, n), dtype=dtype, device=device),
            )
            self._scratch[key] = buffers
        return buffers

    def _reference_forward(self, x: torch.Tensor) -> torch.Tensor:
        """The baseline's dataflow, using the baseline's own submodules."""
        b, c, h, w = x.shape
        n = h * w
        qkv = self.qkv(x)
        q, k, v = qkv.view(
            b, self.num_heads, self.key_dim * 2 + self.head_dim, n
        ).split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = self._softmax(attn)
        x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.pe(
            v.reshape(b, c, h, w)
        )
        return self.proj(x)

    def _fast_forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, img_h, img_w = x.shape
        n = img_h * img_w
        dim, c_qkv = self.dim, self.c_qkv
        stream = torch.cuda.current_stream(x.device)
        pack = self._folded_pack(x.dtype, x.device, stream)
        plan = self._launch_plan(b, img_h, img_w)
        x = x.contiguous()

        qkv, attn_out = self._workspace(b, n, x.dtype, x.device, stream)
        out = torch.empty((b, dim, img_h, img_w), dtype=x.dtype, device=x.device)

        # The full bound-argument sequences, in declaration order with the
        # constexpr values in place. This is exactly what `JITFunction.run`
        # forwards to the launcher, so the same tuples serve both launch paths.
        qkv_args = (
            x, pack.qkv_w, pack.qkv_b, qkv, n, dim, c_qkv,
            plan.block_c, plan.block_n_qkv, plan.block_k,
        )
        attn_args = (
            qkv, attn_out, n, self.num_heads, self.key_dim, self.head_dim,
            c_qkv, self.k_offset, self.v_offset, self.scale * _LOG2E,
            plan.block_m, plan.block_key, plan.block_dk, plan.block_dv,
        )
        epilogue_args = (
            qkv, attn_out, pack.pe_w, pack.pe_b, pack.proj_w, pack.proj_b, out,
            n, img_h, img_w, dim, c_qkv, self.v_offset,
            plan.block_dim, plan.block_n_epilogue, plan.block_oc,
        )

        key = (b, img_h, img_w, x.dtype, x.device, stream, id(pack), id(qkv),
               id(attn_out), plan)
        launchers = self._launchers.get(key) if self._direct_launch else None
        # Every pointer the compiled binaries were specialized against must still
        # have the alignment they were specialized for. The cached buffers cannot
        # move while `key` holds their identity, so only the two externally-owned
        # pointers need re-checking: the caller's input and this call's output.
        if launchers is not None and (
                (x.data_ptr() | out.data_ptr()) % _POINTER_ALIGNMENT == 0):
            handle = stream.cuda_stream
            for launcher, args in zip(launchers, (qkv_args, attn_args,
                                                  epilogue_args)):
                grid = launcher.grid
                launcher.run(grid[0], grid[1], grid[2], handle,
                             launcher.function, launcher.metadata,
                             None, None, None, *args)
            return out

        # Standard path. It is the fallback for an alignment or key miss, and it
        # is also how a descriptor gets built: `JITFunction.run` returns the
        # compiled kernel it just launched, so the miss does the real work and
        # yields the exact binary for these arguments at the same time.
        compiled = (
            _qkv_proj_kernel[plan.grid_qkv](
                x, pack.qkv_w, pack.qkv_b, qkv, n, dim, c_qkv,
                BLOCK_C=plan.block_c, BLOCK_N=plan.block_n_qkv,
                BLOCK_K=plan.block_k, num_warps=plan.warps_qkv),
            _flash_attn_kernel[plan.grid_attn](
                qkv, attn_out, n, self.num_heads, self.key_dim, self.head_dim,
                c_qkv, self.k_offset, self.v_offset, self.scale * _LOG2E,
                BLOCK_M=plan.block_m, BLOCK_KEY=plan.block_key,
                BLOCK_DK=plan.block_dk, BLOCK_DV=plan.block_dv,
                num_warps=plan.warps_attn),
            _pe_residual_proj_kernel[plan.grid_epilogue](
                qkv, attn_out, pack.pe_w, pack.pe_b, pack.proj_w, pack.proj_b,
                out, n, img_h, img_w, dim, c_qkv, self.v_offset,
                BLOCK_DIM=plan.block_dim, BLOCK_N=plan.block_n_epilogue,
                BLOCK_OC=plan.block_oc, num_warps=plan.warps_epilogue),
        )
        if self._direct_launch and key not in self._launchers and all(
                getattr(k, "function", None) is not None for k in compiled):
            aligned = all(
                t.data_ptr() % _POINTER_ALIGNMENT == 0
                for t in (x, qkv, attn_out, out, pack.qkv_w, pack.qkv_b,
                          pack.pe_w, pack.pe_b, pack.proj_w, pack.proj_b))
            if aligned:
                self._launchers[key] = tuple(
                    _Launcher(run=k.run, function=k.function,
                              metadata=k.packed_metadata, grid=grid)
                    for k, grid in zip(compiled, (
                        _pad_grid(plan.grid_qkv), _pad_grid(plan.grid_attn),
                        _pad_grid(plan.grid_epilogue))))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._use_fast_path(x):
            return self._reference_forward(x)
        try:
            return self._fast_forward(x)
        except Exception as exc:  # noqa: BLE001 - a kernel that will not build
            # must never cost correctness; fall back and stop retrying.
            self._fast_path_supported = False
            self._fast_path_error = exc
            return self._reference_forward(x)


def _invalidate_on_load(module: nn.Module, incompatible_keys) -> None:
    """load_state_dict post-hook: new weights mean the folded pack is stale.

    Fires on the top-level module even when only a submodule's keys were
    present, and fires under ``strict=False`` with populated ``missing_keys``.
    """
    del incompatible_keys
    module._invalidate_pack()
