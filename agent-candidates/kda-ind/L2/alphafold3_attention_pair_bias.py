"""Launch-bound fusion of AttentionPairBias / CrossAttentionPairBias.

Both operators are tiny -- 16 tokens for ``AttentionPairBias``, 368 atoms in 12
blocks for ``CrossAttentionPairBias`` -- so the GPU sits idle for most of the
forward and the cost is the host walking 76-225 ATen calls at ~5 us each.
Measured on B200 under the bench harness's own timing recipe: an empty forward
costs 19.4 us (that is the harness's ``_ShiftingPool.next()``, not ours) and
each extra launch costs ~4.5 us, so latency is ``19.4 us + n_launches * 4.5 us``
and the only lever that matters is the launch count, not arithmetic.

Each class therefore runs three Triton launches:

1. ``prep``  -- every row-wise normalization, the ``s``-consumer projections,
                the AdaLN combine, and (``AttentionPairBias``) the pair-bias
                projection with ``layer_norm_z``'s affine applied inline.
2. ``attn``  -- the ``q/k/v/g`` projection as a per-head prologue, key-index
                derivation in closed form, a predicated key/value gather, pair
                bias, mask bias, fp32 softmax, output gate.
3. ``out``   -- ``linear_o`` with the ``sigmoid(linear_ada_out(s))`` epilogue.

The restructurings are exact algebra over the baseline: fold each projection
into the kernel that produces its input, and move row-wise work in front of the
key gather (legal because every step from the gather to K/V is row-wise, so
``f(gather(x)) == gather(f(x))``). Nothing is approximated and no weight is
rewritten -- the kernels read the registered parameters directly, so there is no
derived-weight cache to build after ``load_state_dict`` or to invalidate later.

Inputs the fused path does not cover -- other devices or dtypes, several batch
elements, channel counts past the tile limits, no ``n_query``/``n_key`` -- go to
a reference path that is the baseline composition. It is chosen by an explicit
precondition check and never by catching an exception from the fused path, so a
numerical or launch bug surfaces instead of being papered over.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import triton
import triton.language as tl

try:
    from triton.runtime.errors import OutOfResources as _OutOfResources
except ImportError:  # pragma: no cover - older Triton reports it differently
    class _OutOfResources(Exception):
        pass

from ..L1.sigmoid import Sigmoid
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from .alphafold3_swiglu import AdaLN
from .alphafold3_of3_attention import OF3Attention

_EPS = 1e-5


def _permute_final_dims(tensor: torch.Tensor, inds: list[int]) -> torch.Tensor:
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


# ---------------------------------------------------------------------------
# Fused-path plumbing
# ---------------------------------------------------------------------------


def _np2(n: int) -> int:
    return 1 << max(0, (int(n) - 1).bit_length())


def _tile(n: int, cap: int) -> int:
    """Power-of-two tile covering *n*, at least 16 (``tl.dot``'s minimum)."""
    return min(cap, max(16, _np2(n)))


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _one_batch(t: torch.Tensor, n_keep: int) -> bool:
    """True when every dim left of the last *n_keep* axes is 1.

    The kernels index rows with a single flat stride, so they cover one logical
    batch element. Necessary but not sufficient -- see :func:`_leading_ok`.
    """
    n = 1
    for d in t.shape[:t.dim() - n_keep]:
        n *= d
    return n == 1


def _leading_ok(lead, *others) -> bool:
    """True when the baseline's own broadcast would produce *lead*.

    The fused path writes into ``torch.empty_like(a)``, so it may only run when
    that is the shape the baseline would return. Every leading shape here
    multiplies to 1, but they can still differ in *rank*: ``a[1,16,768]`` with
    ``s[1,1,16,384]`` is legal and the baseline returns ``[1,1,16,768]``, because
    AdaLN broadcasts ``a_norm`` against ``linear_s(s_norm)``. A fused path that
    only checked the products would return ``[1,16,768]``.
    """
    try:
        return tuple(torch.broadcast_shapes(lead, *others)) == tuple(lead)
    except RuntimeError:
        return False


def _sources_ok(srcs, expected, device) -> bool:
    """Every tensor the kernels read must be laid out the way they read it.

    The kernels take a bare pointer and do row-major arithmetic from the shapes
    in the launch plan, so a same-shape *transposed* weight -- a perfectly valid
    ``Parameter`` that the baseline handles -- would be read as though it were
    contiguous. Shape, contiguity, dtype and device therefore all have to be
    checked before launching, and the ones that can change without changing the
    parameter's identity also have to be in the invalidation key.

    A ``None`` source is an optional parameter the kernel has a flag for;
    requiredness is enforced separately by each class's ``_fused_ok``.
    """
    for t, shape in zip(srcs, expected):
        if t is None:
            continue
        if (t.dtype is not torch.bfloat16 or tuple(t.shape) != shape
                or not t.is_contiguous() or t.device != device):
            return False
    return True


def _source_key(srcs) -> tuple:
    """What must invalidate a cached plan, for every tensor the kernels read.

    Values are deliberately absent: the launcher reads each ``data_ptr`` at
    launch, so an in-place update or a ``.data`` swap is picked up on its own --
    which is what makes ``load_state_dict`` work. What must invalidate is a
    *replaced* parameter (the cached argument tuple would keep launching against
    the detached old tensor) and anything that changes how the bytes are read:
    dtype, shape, contiguity, device.
    """
    return tuple(None if t is None else
                 (id(t), t.dtype, t.shape, t.is_contiguous(), t.device)
                 for t in srcs)


def _needs_autograd(srcs, *inputs) -> bool:
    """True when the fused path would silently drop an autograd graph.

    The launches build no backward and the output comes from
    ``torch.empty_like``, so anything the baseline would have made
    differentiable has to go to the reference path instead. Free on the
    benchmark's path: the mode check short-circuits before any tensor is
    touched, and the harness runs every forward under ``no_grad``.
    """
    if not torch.is_grad_enabled():
        return False
    return (any(t is not None and t.requires_grad for t in inputs)
            or any(t is not None and t.requires_grad for t in srcs))


def _contig(*tensors) -> bool:
    """Contiguity cannot be cached: a caller may pass a transposed view with the
    same shape the plan was built for."""
    return all(t is None or t.is_contiguous() for t in tensors)


def _raw_stream(device) -> int:
    """Current CUDA stream as an integer handle, for use as a cache key."""
    try:
        return torch._C._cuda_getCurrentRawStream(device.index or 0)
    except AttributeError:  # pragma: no cover - very old torch
        return torch.cuda.current_stream(device).cuda_stream


class _Launcher:
    """Launch a compiled Triton kernel without re-binding its arguments.

    ``JITFunction.run`` re-binds and re-specializes every argument on every
    call. For these kernels' 25-35 arguments that measured 12-24 us per launch,
    more than the kernels take on the GPU. So the first call goes through the
    JIT -- which is what compiles the kernel, and doing it on the first forward
    keeps compilation out of the timed region -- and keeps the
    ``CompiledKernel`` it returns. Later calls go through that kernel's own
    runner, which takes every parameter positionally in declaration order,
    constexprs included, for about 6 us.

    Argument order comes from ``fn.arg_names`` rather than being hardcoded, and
    the constexpr tail is checked against it once. If a Triton build does not
    expose the runner, ``_runner`` stays None and every call keeps going through
    the JIT path: slower, identical results.
    """

    # Pipeline depth and K tile to try, in order. How much shared memory a tile
    # shape needs is not predictable from the source -- a wide token axis with
    # three staged weight tiles overflows where the benched shapes fit
    # comfortably -- so back off along both rather than pinning a depth
    # conservative enough for every shape and slowing the ones that fit.
    _BACKOFF = ((None, 1), (2, 1), (1, 1), (1, 2), (1, 4), (1, 8))

    __slots__ = ("_fn", "_grid", "_names", "_tail", "_kw", "_runner")

    def __init__(self, fn, grid, const, opts=None):
        names = tuple(fn.arg_names)
        n_pos = len(names) - len(const)
        if names[n_pos:] != tuple(const):
            raise ValueError(f"{fn.__name__}: constexpr args must come last, "
                             f"in declaration order; got {names[n_pos:]} "
                             f"vs {tuple(const)}")
        self._fn = fn
        self._grid = tuple(grid) + (1,) * (3 - len(grid))
        self._names = names[n_pos:]
        self._tail = tuple(const.values())
        self._kw = {**const, **(opts or {})}
        self._runner = None

    def __call__(self, *pos):
        runner = self._runner
        if runner is not None:
            runner(*pos, *self._tail)
            return
        self._prime(pos)

    def _prime(self, pos) -> None:
        """Compile, then keep the compiled kernel's fast runner.

        Runs on the first forward, so nothing is compiled inside a timed region.
        """
        pinned = "num_stages" in self._kw
        failure = None
        for stages, shrink in self._BACKOFF:
            kw = dict(self._kw)
            if not pinned and stages is not None:
                kw["num_stages"] = stages
            if shrink > 1:
                if "BK" not in kw or kw["BK"] <= 16:
                    break
                kw["BK"] = max(16, kw["BK"] // shrink)
            try:
                kernel = self._fn.run(*pos, grid=self._grid, warmup=False, **kw)
            except _OutOfResources as exc:
                failure = exc
                continue
            self._tail = tuple(kw[nm] for nm in self._names)
            try:
                self._runner = kernel[self._grid]
            except (AttributeError, TypeError):  # other Triton launch APIs
                self._runner = None
            return
        raise failure


class _Plan:
    """Everything about a launch that depends only on the module's structure and
    the input shapes, resolved once.

    Rebuilding tile sizes, precondition checks and 30-element argument tuples on
    every call cost 30 us of the 82 us this forward once took -- more than the
    kernels themselves. What is left per call is building the signature and
    comparing it.

    The plan owns its workspace, so evicting a plan frees the buffers with it.
    That matters because the workspace is per stream: an application that creates
    streams over time would otherwise accumulate one set forever (measured 192
    buffers and 24 MB after 40 transient streams).

    ``args`` holds the arguments that do not vary between calls -- the workspace
    and the registered parameters. The input tensors are passed separately,
    because the harness hands the forward a different ``data_ptr`` every
    iteration.
    """

    __slots__ = ("ok", "launch", "args", "nil", "buf")

    def __init__(self, ok, launch=(), args=(), nil=None, buf=None):
        self.ok = ok
        self.launch = launch
        self.args = args
        self.nil = nil
        self.buf = buf  # keeps the workspace alive for exactly this plan's life


class _PlanCache:
    """Bounded least-recently-used plan cache.

    Keyed by the full signature, so alternating CUDA streams reuses each stream's
    plan instead of re-priming every launcher on each switch, and bounded so the
    workspaces it keeps alive cannot grow without limit. Workspaces are only ever
    used on the stream they were allocated on, so dropping the least recently
    used plan is safe: the caching allocator will not hand a freed block to a
    different stream without synchronizing first.
    """

    __slots__ = ("_d",)
    MAX = 8

    def __init__(self):
        self._d: "OrderedDict[tuple, _Plan]" = OrderedDict()

    def get(self, sig):
        plan = self._d.get(sig)
        if plan is not None:
            self._d.move_to_end(sig)
        return plan

    def put(self, sig, plan):
        self._d[sig] = plan
        self._d.move_to_end(sig)
        while len(self._d) > self.MAX:
            self._d.popitem(last=False)
        return plan


class _Workspace:
    """The scratch buffers one plan owns, allocated on that plan's stream.

    An allocation is a host-side call like any other, and at ~4 us each these
    would cost as much as a kernel launch, so they are allocated once per plan
    and reused. They never escape the forward, so the reuse is invisible to
    callers.
    """

    __slots__ = ("_d", "_dev")

    def __init__(self, device):
        self._d: dict = {}
        self._dev = device

    def get(self, key, shape) -> torch.Tensor:
        t = self._d.get(key)
        if t is None:
            t = torch.empty(shape, dtype=torch.bfloat16, device=self._dev)
            self._d[key] = t
        return t


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------
# Every rounding the baseline performs is reproduced. The baseline's
# intermediates are bf16 tensors, so scores, biases, softmax probabilities and
# gates are rounded to bf16 between steps while reductions and dot accumulations
# run in fp32, exactly as ATen's opmath does. ``_rb`` marks each of those
# rounding points; dropping one is not merely less accurate, it changes
# behaviour -- a bf16 score plus a bf16 ``-1e9`` mask bias collapses to exactly
# ``-1e9``, so a fully masked query row softmaxes to uniform weights, which an
# fp32 chain would not reproduce.


@triton.jit
def _rb(x):
    """Round to bf16 and widen back: one ATen bf16 intermediate."""
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _sig(x):
    return 1.0 / (1.0 + tl.exp(-x))


@triton.jit
def _mask_sum(mask_ptr, n_atom, HAS_MASK: tl.constexpr, BMS: tl.constexpr):
    """``atom_mask.sum(-1)`` over the padded axis, rounded to bf16 like ATen.

    The pad entries the baseline appends are zero, so summing the real atoms is
    the same value. The bf16 rounding of the result is not cosmetic: it is what
    makes the key indices below land on duplicates.
    """
    acc = 0.0
    for i0 in range(0, n_atom, BMS):
        ii = i0 + tl.arange(0, BMS)
        if HAS_MASK:
            mv = tl.load(mask_ptr + ii, mask=ii < n_atom, other=0.0).to(tl.float32)
        else:
            mv = tl.where(ii < n_atom, 1.0, 0.0)
        acc += tl.sum(mv, 0)
    return _rb(acc)


@triton.jit
def _block_key_indices(n_real, first, n_key, BKN: tl.constexpr):
    """``_get_block_key_indices`` for one query block, in closed form.

    Reproduces the dtype promotions, which are load-bearing rather than
    incidental. ``n_real`` comes from a bf16 ``sum``, so PyTorch promotes to
    bf16 at three points -- ``overflow`` (``int32 - bf16``), ``total_shift``
    (``where(int32, bf16)``) and ``final`` (``int32 + bf16``). bf16 spacing is 2
    above 256, so with 368 real atoms ``n_real - 1`` rounds *up* to 368, index
    367 rounds to 368, and most indices in the trailing blocks land on a
    duplicate: block 11 gathers ``[240, 240, 242, 244, ...]`` and its last slot
    reads a zero pad row. Doing this in int32 would silently attend to
    different atoms.

    Returns ``(safe_index, invalid)`` exactly as the reference does: the index is
    the reference's own clamped value, with no extra bound of our own, so a test
    can compare every element against it. Keeping a gather in range is the
    caller's job -- see the sentinel redirect in :func:`_capb_attn`.
    ``invalid`` is derived from the *unclamped* value, as in the baseline.
    """
    kk = tl.arange(0, BKN)
    underflow = tl.maximum(-first, 0)
    nr1 = _rb(n_real - 1.0)
    overflow = tl.maximum(_rb(_rb((first + n_key - 1).to(tl.float32)) - nr1), 0.0)
    shift = tl.where(underflow > 0, _rb(underflow.to(tl.float32)), -overflow)
    final = _rb(_rb((first + kk).to(tl.float32)) + shift)
    invalid = (final < 0.0) | (final >= n_real)
    safe = tl.minimum(tl.maximum(final, 0.0), tl.maximum(nr1, 0.0))
    return safe.to(tl.int32), invalid  # ``.long()`` truncates toward zero


@triton.jit
def _apb_prep(
    a_ptr, s_ptr, z_ptr, an_ptr, ada_ptr, zb_ptr,
    lna_w, lna_b, lns_w, wg_ptr, bg_ptr, ws_ptr, wao_ptr, bao_ptr,
    lnz_w, lnz_b, wz_ptr,
    N, c_q, c_s, c_z, H, n_qt, eps,
    ADA: tl.constexpr, LNA_W: tl.constexpr, LNA_B: tl.constexpr,
    LNZ_B: tl.constexpr,
    BN: tl.constexpr, BQ: tl.constexpr, BK: tl.constexpr,
    BZ: tl.constexpr, BCZ: tl.constexpr, BH: tl.constexpr,
):
    """Normalizations, the AdaLN combine, and the pair-bias projection.

    Two disjoint program ranges share one launch: ``pid < n_qt`` owns a column
    tile of the ``[N, c_q]`` token work, the rest own a row tile of the
    ``[N*N, c_z]`` pair work. Merging them costs a branch and saves a launch,
    which at ~4.5 us per launch is the whole point of this file.
    """
    pid = tl.program_id(0)
    if pid < n_qt:
        rows = tl.arange(0, BN)
        rok = rows < N
        qc = pid * BQ + tl.arange(0, BQ)
        qok = qc < c_q

        # Row statistics streamed over the channel axis, so a wide c_q never
        # has to sit in registers all at once.
        s1 = tl.zeros([BN], tl.float32)
        s2 = tl.zeros([BN], tl.float32)
        for c0 in range(0, c_q, BK):
            cc = c0 + tl.arange(0, BK)
            x = tl.load(a_ptr + rows[:, None] * c_q + cc[None, :],
                        mask=rok[:, None] & (cc < c_q)[None, :], other=0.0).to(tl.float32)
            s1 += tl.sum(x, 1)
            s2 += tl.sum(x * x, 1)
        mu = s1 / c_q
        rstd = 1.0 / tl.sqrt(tl.maximum(s2 / c_q - mu * mu, 0.0) + eps)

        A = tl.load(a_ptr + rows[:, None] * c_q + qc[None, :],
                    mask=rok[:, None] & qok[None, :], other=0.0).to(tl.float32)
        a_hat = (A - mu[:, None]) * rstd[:, None]

        if ADA:
            t1 = tl.zeros([BN], tl.float32)
            t2 = tl.zeros([BN], tl.float32)
            for c0 in range(0, c_s, BK):
                cc = c0 + tl.arange(0, BK)
                x = tl.load(s_ptr + rows[:, None] * c_s + cc[None, :],
                            mask=rok[:, None] & (cc < c_s)[None, :], other=0.0).to(tl.float32)
                t1 += tl.sum(x, 1)
                t2 += tl.sum(x * x, 1)
            mus = t1 / c_s
            rstds = 1.0 / tl.sqrt(tl.maximum(t2 / c_s - mus * mus, 0.0) + eps)

            acc_g = tl.zeros([BN, BQ], tl.float32)
            acc_s = tl.zeros([BN, BQ], tl.float32)
            acc_o = tl.zeros([BN, BQ], tl.float32)
            for c0 in range(0, c_s, BK):
                cc = c0 + tl.arange(0, BK)
                cok = cc < c_s
                S = tl.load(s_ptr + rows[:, None] * c_s + cc[None, :],
                            mask=rok[:, None] & cok[None, :], other=0.0).to(tl.float32)
                wv = tl.load(lns_w + cc, mask=cok, other=0.0).to(tl.float32)
                u = tl.where(cok[None, :], (S - mus[:, None]) * rstds[:, None], 0.0)
                sn = (u * wv[None, :]).to(tl.bfloat16)
                sr = tl.where(cok[None, :], S, 0.0).to(tl.bfloat16)
                wm = qok[:, None] & cok[None, :]
                off = qc[:, None] * c_s + cc[None, :]
                acc_g += tl.dot(sn, tl.trans(tl.load(wg_ptr + off, mask=wm, other=0.0)))
                acc_s += tl.dot(sn, tl.trans(tl.load(ws_ptr + off, mask=wm, other=0.0)))
                acc_o += tl.dot(sr, tl.trans(tl.load(wao_ptr + off, mask=wm, other=0.0)))

            bg = tl.load(bg_ptr + qc, mask=qok, other=0.0).to(tl.float32)
            g = _rb(_sig(_rb(acc_g + bg[None, :])))
            a_n = _rb(g * _rb(_rb(a_hat) + _rb(acc_s)))
            bao = tl.load(bao_ptr + qc, mask=qok, other=0.0).to(tl.float32)
            tl.store(ada_ptr + rows[:, None] * c_q + qc[None, :],
                     _rb(acc_o + bao[None, :]).to(tl.bfloat16),
                     mask=rok[:, None] & qok[None, :])
        else:
            a_n = a_hat
            if LNA_W:
                a_n = a_n * tl.load(lna_w + qc, mask=qok, other=0.0).to(tl.float32)[None, :]
            if LNA_B:
                a_n = a_n + tl.load(lna_b + qc, mask=qok, other=0.0).to(tl.float32)[None, :]
            a_n = _rb(a_n)

        tl.store(an_ptr + rows[:, None] * c_q + qc[None, :], a_n.to(tl.bfloat16),
                 mask=rok[:, None] & qok[None, :])
    else:
        nz = N * N
        zr = (pid - n_qt) * BZ + tl.arange(0, BZ)
        zok = zr < nz
        zc = tl.arange(0, BCZ)
        zcok = zc < c_z
        Z = tl.load(z_ptr + zr[:, None] * c_z + zc[None, :],
                    mask=zok[:, None] & zcok[None, :], other=0.0).to(tl.float32)
        zmu = tl.sum(Z, 1) / c_z
        zd = tl.where(zcok[None, :], Z - zmu[:, None], 0.0)
        zvar = tl.sum(zd * zd, 1) / c_z
        zn = zd * (1.0 / tl.sqrt(zvar + eps))[:, None]
        zn = zn * tl.load(lnz_w + zc, mask=zcok, other=0.0).to(tl.float32)[None, :]
        if LNZ_B:
            zn = zn + tl.load(lnz_b + zc, mask=zcok, other=0.0).to(tl.float32)[None, :]
        hh = tl.arange(0, BH)
        hok = hh < H
        wzt = tl.load(wz_ptr + hh[:, None] * c_z + zc[None, :],
                      mask=hok[:, None] & zcok[None, :], other=0.0)
        zb = tl.dot(zn.to(tl.bfloat16), tl.trans(wzt))
        tl.store(zb_ptr + hh[None, :] * nz + zr[:, None], zb.to(tl.bfloat16),
                 mask=zok[:, None] & hok[None, :])


@triton.jit
def _apb_attn(
    mask_ptr, an_ptr, zb_ptr, og_ptr, wq, bq, wk, wv, wgt,
    N, c_q, D, HD, qdiv, inf,
    HAS_MASK: tl.constexpr, Q_BIAS: tl.constexpr,
    BN: tl.constexpr, BD: tl.constexpr, BK: tl.constexpr,
):
    """One program per head: q/k/v/g projection, biased softmax, output gate.

    The projection rides along here rather than in its own GEMM because each
    head owns a disjoint slice of the projection output, so folding it in adds
    no redundant work -- it only removes a launch.
    """
    h = tl.program_id(0)
    rows = tl.arange(0, BN)
    rok = rows < N
    dd = tl.arange(0, BD)
    dok = dd < D
    oc = h * D + dd

    aq = tl.zeros([BN, BD], tl.float32)
    ak = tl.zeros([BN, BD], tl.float32)
    av = tl.zeros([BN, BD], tl.float32)
    ag = tl.zeros([BN, BD], tl.float32)
    for c0 in range(0, c_q, BK):
        cc = c0 + tl.arange(0, BK)
        cok = cc < c_q
        An = tl.load(an_ptr + rows[:, None] * c_q + cc[None, :],
                     mask=rok[:, None] & cok[None, :], other=0.0)
        wm = dok[:, None] & cok[None, :]
        off = oc[:, None] * c_q + cc[None, :]
        aq += tl.dot(An, tl.trans(tl.load(wq + off, mask=wm, other=0.0)))
        ak += tl.dot(An, tl.trans(tl.load(wk + off, mask=wm, other=0.0)))
        av += tl.dot(An, tl.trans(tl.load(wv + off, mask=wm, other=0.0)))
        ag += tl.dot(An, tl.trans(tl.load(wgt + off, mask=wm, other=0.0)))

    if Q_BIAS:
        aq = aq + tl.load(bq + oc, mask=dok, other=0.0).to(tl.float32)[None, :]
    q = _rb(_rb(aq) / qdiv)
    g = _rb(_sig(_rb(ag)))

    sc = _rb(tl.dot(q.to(tl.bfloat16), tl.trans(_rb(ak).to(tl.bfloat16))))
    if HAS_MASK:
        mv = tl.load(mask_ptr + rows, mask=rok, other=1.0).to(tl.float32)
        sc = _rb(sc + _rb(inf * _rb(mv - 1.0))[None, :])
    zb = tl.load(zb_ptr + h * N * N + rows[:, None] * N + rows[None, :],
                 mask=rok[:, None] & rok[None, :], other=0.0).to(tl.float32)
    sc = _rb(sc + zb)

    sc = tl.where(rok[None, :], sc, float("-inf"))
    p = tl.exp(sc - tl.max(sc, 1)[:, None])
    p = (p / tl.sum(p, 1)[:, None]).to(tl.bfloat16)
    o = _rb(_rb(tl.dot(p, _rb(av).to(tl.bfloat16))) * g)
    tl.store(og_ptr + rows[:, None] * HD + oc[None, :], o.to(tl.bfloat16),
             mask=rok[:, None] & dok[None, :])


@triton.jit
def _gated_out(
    out_ptr, og_ptr, ada_ptr, wo, n_row, c_q, HD,
    ADA: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr,
):
    """``linear_o`` with the ``sigmoid(linear_ada_out(s))`` gate as its epilogue."""
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    rok = rows < n_row
    nc = tl.program_id(1) * BC + tl.arange(0, BC)
    nok = nc < c_q
    acc = tl.zeros([BR, BC], tl.float32)
    for c0 in range(0, HD, BK):
        cc = c0 + tl.arange(0, BK)
        cok = cc < HD
        OG = tl.load(og_ptr + rows[:, None] * HD + cc[None, :],
                     mask=rok[:, None] & cok[None, :], other=0.0)
        W = tl.load(wo + nc[:, None] * HD + cc[None, :],
                    mask=nok[:, None] & cok[None, :], other=0.0)
        acc += tl.dot(OG, tl.trans(W))
    o = _rb(acc)
    if ADA:
        ap = tl.load(ada_ptr + rows[:, None] * c_q + nc[None, :],
                     mask=rok[:, None] & nok[None, :], other=0.0).to(tl.float32)
        o = _rb(o * _rb(_sig(ap)))
    tl.store(out_ptr + rows[:, None] * c_q + nc[None, :], o.to(tl.bfloat16),
             mask=rok[:, None] & nok[None, :])


@triton.jit
def _capb_prep(
    a_ptr, s_ptr, z_ptr, aq_ptr, ak_ptr, ada_ptr, zb_ptr, wz,
    lnsq_w, lnsk_w, wgq, bgq, wsq, wgk, bgk, wsk, wao, bao,
    lnq_w, lnq_b, lnk_w, lnk_b,
    n_atom, n_pad, c_q, c_s, c_z, H, n_zrow, n_rt, eps,
    ADA: tl.constexpr, LN_W: tl.constexpr, LN_B: tl.constexpr,
    BR: tl.constexpr, BCQ: tl.constexpr, BK: tl.constexpr,
    BZ: tl.constexpr, BCZ: tl.constexpr, BH: tl.constexpr,
):
    """Query- and key-side normalized rows, computed flat on the padded axis.

    Every step from the key gather to K/V is row-wise, so ``f(gather(x)) ==
    gather(f(x))`` and this runs once over ``n_pad`` rows instead of once over
    ``n_blocks * n_key`` gathered rows -- 384 instead of 1536 -- with the gather
    deferred into the attention kernel.

    Rows at or past ``n_atom`` are the zero pad the baseline creates, and are
    loaded as zeros. One extra row past the pad, index ``n_pad``, is written the
    same way and exists so the attention kernel has somewhere to point an
    invalid key slot. The baseline zeroes a gathered row *before* the norms, so
    an invalid slot is worth whatever this pipeline maps an all-zero row to --
    exactly zero under AdaLN, where ``layer_norm_a`` has neither scale nor
    offset and ``layer_norm_s`` has no offset so ``LayerNorm(0) == 0`` and
    ``linear_s(0) == 0``, but the LayerNorm *bias* when ``use_ada_layer_norm``
    is False. Substituting zero in that second case would drop the bias and
    silently change every fully masked query row.

    A second program range projects the pair embedding, ``linear_z``, into
    ``[H, n_blocks*n_query*n_key]``. It belongs here rather than in the
    attention kernel: there, each program would have to walk ``z`` with a
    ``c_z``-element stride to assemble its own ``[n_query, n_key]`` bias tile,
    which turned one coalesced read into a transaction per element and cost
    68 us of the 78 us that kernel was spending.
    """
    pid = tl.program_id(0)
    if pid >= n_rt:
        zrow = (pid - n_rt) * BZ + tl.arange(0, BZ)
        zrok = zrow < n_zrow
        zcc = tl.arange(0, BCZ)
        zcok = zcc < c_z
        Zt = tl.load(z_ptr + zrow[:, None] * c_z + zcc[None, :],
                     mask=zrok[:, None] & zcok[None, :], other=0.0)
        hh = tl.arange(0, BH)
        hok = hh < H
        wzt = tl.load(wz + hh[:, None] * c_z + zcc[None, :],
                      mask=hok[:, None] & zcok[None, :], other=0.0)
        tl.store(zb_ptr + hh[None, :] * n_zrow + zrow[:, None],
                 tl.dot(Zt, tl.trans(wzt)).to(tl.bfloat16),
                 mask=zrok[:, None] & hok[None, :])
        return

    rows = pid * BR + tl.arange(0, BR)
    real = rows < n_atom
    qc = tl.arange(0, BCQ)
    qok = qc < c_q
    store_ok = (rows <= n_pad)[:, None] & qok[None, :]

    A = tl.load(a_ptr + rows[:, None] * c_q + qc[None, :],
                mask=real[:, None] & qok[None, :], other=0.0).to(tl.float32)
    mu = tl.sum(A, 1) / c_q
    d = tl.where(qok[None, :], A - mu[:, None], 0.0)
    var = tl.sum(d * d, 1) / c_q
    a_hat = d * (1.0 / tl.sqrt(var + eps))[:, None]

    if ADA:
        t1 = tl.zeros([BR], tl.float32)
        t2 = tl.zeros([BR], tl.float32)
        for c0 in range(0, c_s, BK):
            cc = c0 + tl.arange(0, BK)
            x = tl.load(s_ptr + rows[:, None] * c_s + cc[None, :],
                        mask=real[:, None] & (cc < c_s)[None, :], other=0.0).to(tl.float32)
            t1 += tl.sum(x, 1)
            t2 += tl.sum(x * x, 1)
        mus = t1 / c_s
        rstds = 1.0 / tl.sqrt(tl.maximum(t2 / c_s - mus * mus, 0.0) + eps)

        acc_gq = tl.zeros([BR, BCQ], tl.float32)
        acc_sq = tl.zeros([BR, BCQ], tl.float32)
        acc_gk = tl.zeros([BR, BCQ], tl.float32)
        acc_sk = tl.zeros([BR, BCQ], tl.float32)
        acc_o = tl.zeros([BR, BCQ], tl.float32)
        for c0 in range(0, c_s, BK):
            cc = c0 + tl.arange(0, BK)
            cok = cc < c_s
            S = tl.load(s_ptr + rows[:, None] * c_s + cc[None, :],
                        mask=real[:, None] & cok[None, :], other=0.0).to(tl.float32)
            u = tl.where(cok[None, :], (S - mus[:, None]) * rstds[:, None], 0.0)
            snq = (u * tl.load(lnsq_w + cc, mask=cok, other=0.0).to(tl.float32)[None, :]
                   ).to(tl.bfloat16)
            snk = (u * tl.load(lnsk_w + cc, mask=cok, other=0.0).to(tl.float32)[None, :]
                   ).to(tl.bfloat16)
            sr = tl.where(cok[None, :], S, 0.0).to(tl.bfloat16)
            wm = qok[:, None] & cok[None, :]
            off = qc[:, None] * c_s + cc[None, :]
            acc_gq += tl.dot(snq, tl.trans(tl.load(wgq + off, mask=wm, other=0.0)))
            acc_sq += tl.dot(snq, tl.trans(tl.load(wsq + off, mask=wm, other=0.0)))
            acc_gk += tl.dot(snk, tl.trans(tl.load(wgk + off, mask=wm, other=0.0)))
            acc_sk += tl.dot(snk, tl.trans(tl.load(wsk + off, mask=wm, other=0.0)))
            acc_o += tl.dot(sr, tl.trans(tl.load(wao + off, mask=wm, other=0.0)))

        ah = _rb(a_hat)
        bq_ = tl.load(bgq + qc, mask=qok, other=0.0).to(tl.float32)[None, :]
        bk_ = tl.load(bgk + qc, mask=qok, other=0.0).to(tl.float32)[None, :]
        aqn = _rb(_rb(_sig(_rb(acc_gq + bq_))) * _rb(ah + _rb(acc_sq)))
        akn = _rb(_rb(_sig(_rb(acc_gk + bk_))) * _rb(ah + _rb(acc_sk)))
        tl.store(ada_ptr + rows[:, None] * c_q + qc[None, :],
                 _rb(acc_o + tl.load(bao + qc, mask=qok, other=0.0).to(tl.float32)[None, :]
                     ).to(tl.bfloat16), mask=store_ok)
    else:
        aqn = a_hat
        akn = a_hat
        if LN_W:
            aqn = aqn * tl.load(lnq_w + qc, mask=qok, other=0.0).to(tl.float32)[None, :]
            akn = akn * tl.load(lnk_w + qc, mask=qok, other=0.0).to(tl.float32)[None, :]
        if LN_B:
            aqn = aqn + tl.load(lnq_b + qc, mask=qok, other=0.0).to(tl.float32)[None, :]
            akn = akn + tl.load(lnk_b + qc, mask=qok, other=0.0).to(tl.float32)[None, :]
        aqn = _rb(aqn)
        akn = _rb(akn)

    tl.store(aq_ptr + rows[:, None] * c_q + qc[None, :], aqn.to(tl.bfloat16), mask=store_ok)
    tl.store(ak_ptr + rows[:, None] * c_q + qc[None, :], akn.to(tl.bfloat16), mask=store_ok)


@triton.jit
def _capb_attn(
    mask_ptr, aq_ptr, ak_ptr, zb_ptr, og_ptr, wq, bq, wk, wv, wgt,
    n_atom, n_pad, n_query, n_key, c_q, n_zrow, D, HD, first0, qdiv, inf,
    HAS_MASK: tl.constexpr, Q_BIAS: tl.constexpr,
    BQ: tl.constexpr, BKN: tl.constexpr, BD: tl.constexpr,
    BK: tl.constexpr, BMS: tl.constexpr,
):
    """Sequence-local attention: key indices in closed form, keys and values
    gathered with a predicated load, pair bias folded into the prologue.

    See :func:`_block_key_indices` for the index arithmetic and
    :func:`_mask_sum` for the ``n_real`` it depends on.
    """
    b = tl.program_id(0)
    h = tl.program_id(1)

    n_real = _mask_sum(mask_ptr, n_atom, HAS_MASK, BMS)
    ref_idx, invalid = _block_key_indices(
        n_real, first0 + b * n_query, n_key, BKN)
    kk = tl.arange(0, BKN)
    kok = kk < n_key
    # Row n_pad holds what the key pipeline maps an all-zero input row to, which
    # is what the baseline's pre-norm zero-fill leaves in an invalid slot. The
    # second clamp keeps the gather in range where the baseline's own is not:
    # bf16 can round its clamp bound up to n_real, which equals n_pad when
    # nothing needed padding, and the baseline then indexes one row past its
    # tensor. A slot that lands there reads the sentinel instead.
    idx = tl.minimum(tl.where(invalid, n_pad, ref_idx), n_pad)

    qr = b * n_query + tl.arange(0, BQ)
    qok = tl.arange(0, BQ) < n_query

    dd = tl.arange(0, BD)
    dok = dd < D
    oc = h * D + dd
    accq = tl.zeros([BQ, BD], tl.float32)
    accg = tl.zeros([BQ, BD], tl.float32)
    acck = tl.zeros([BKN, BD], tl.float32)
    accv = tl.zeros([BKN, BD], tl.float32)
    for c0 in range(0, c_q, BK):
        cc = c0 + tl.arange(0, BK)
        cok = cc < c_q
        wm = dok[:, None] & cok[None, :]
        off = oc[:, None] * c_q + cc[None, :]
        Aq = tl.load(aq_ptr + qr[:, None] * c_q + cc[None, :],
                     mask=qok[:, None] & cok[None, :], other=0.0)
        accq += tl.dot(Aq, tl.trans(tl.load(wq + off, mask=wm, other=0.0)))
        accg += tl.dot(Aq, tl.trans(tl.load(wgt + off, mask=wm, other=0.0)))
        Ak = tl.load(ak_ptr + idx[:, None] * c_q + cc[None, :],
                     mask=kok[:, None] & cok[None, :], other=0.0)
        acck += tl.dot(Ak, tl.trans(tl.load(wk + off, mask=wm, other=0.0)))
        accv += tl.dot(Ak, tl.trans(tl.load(wv + off, mask=wm, other=0.0)))

    if Q_BIAS:
        accq = accq + tl.load(bq + oc, mask=dok, other=0.0).to(tl.float32)[None, :]
    q = _rb(_rb(accq) / qdiv)
    g = _rb(_sig(_rb(accg)))
    sc = _rb(tl.dot(q.to(tl.bfloat16), tl.trans(_rb(acck).to(tl.bfloat16))))

    if HAS_MASK:
        mq = tl.load(mask_ptr + qr, mask=(qr < n_atom) & qok, other=0.0).to(tl.float32)
        mk = tl.load(mask_ptr + idx, mask=(idx < n_atom) & kok, other=0.0).to(tl.float32)
    else:
        mq = tl.where((qr < n_atom) & qok, 1.0, 0.0)
        mk = tl.where((idx < n_atom) & kok, 1.0, 0.0)
    mkv = _rb(_rb(tl.where(invalid, 0.0, 1.0)) * mk)
    sc = _rb(sc + _rb(inf * _rb(_rb(mq[:, None] * mkv[None, :]) - 1.0)))

    zb = tl.load(zb_ptr + h * n_zrow + qr[:, None] * n_key + kk[None, :],
                 mask=qok[:, None] & kok[None, :], other=0.0).to(tl.float32)
    sc = _rb(sc + zb)

    sc = tl.where(kok[None, :], sc, float("-inf"))
    p = tl.exp(sc - tl.max(sc, 1)[:, None])
    p = (p / tl.sum(p, 1)[:, None]).to(tl.bfloat16)
    o = _rb(_rb(tl.dot(p, _rb(accv).to(tl.bfloat16))) * g)
    tl.store(og_ptr + qr[:, None] * HD + oc[None, :], o.to(tl.bfloat16),
             mask=((qr < n_pad) & qok)[:, None] & dok[None, :])


class AttentionPairBias(nn.Module):
    """AF3 Algorithm 24: Attention with pair bias.

    Same ``__init__`` / ``forward`` contract and the same submodule and parameter
    names as the baseline, so ``load_state_dict`` transfers every weight. The
    harness wraps that call in a bare ``try/except``, so a renamed or reshaped
    parameter would silently transfer nothing and show up only as a numerical
    mismatch -- hence nothing here is renamed, reshaped or pre-fused.

    Reference: openfold3/core/model/layers/attention_pair_bias.py

    Args:
        c_q: Input dimension of query/key/value
        c_s: Single activation channel dimension (for AdaLN)
        c_z: Pair activation channel dimension
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        use_ada_layer_norm: Whether to use AdaLN-Zero conditioning
        gating: Whether to gate output
        inf: Large constant for masking
    """

    def __init__(
        self,
        c_q: int,
        c_k: int = 0,
        c_v: int = 0,
        c_s: int = 0,
        c_z: int = 128,
        c_hidden: int = 32,
        no_heads: int = 4,
        use_ada_layer_norm: bool = False,
        gating: bool = True,
        inf: float = 1e9,
    ):
        super().__init__()
        c_k = c_k or c_q
        c_v = c_v or c_q

        self.c_q = c_q
        self.c_s = c_s
        self.c_z = c_z
        self.inf = inf
        self.use_ada_layer_norm = use_ada_layer_norm

        if use_ada_layer_norm:
            self.layer_norm_a = AdaLN(c_a=c_q, c_s=c_s)
            self.linear_ada_out = Linear(c_s, c_q, bias=True)
        else:
            self.layer_norm_a = LayerNorm(c_q)

        self.layer_norm_z = LayerNorm(
            c_z, create_offset=not use_ada_layer_norm,
        )
        self.linear_z = Linear(c_z, no_heads, bias=False)

        self.sigmoid = Sigmoid()

        self.mha = OF3Attention(
            c_q=c_q, c_k=c_k, c_v=c_v,
            c_hidden=c_hidden, no_heads=no_heads, gating=gating,
            q_bias=True,
        )

        self._plans = _PlanCache()

    # -- reference path: the baseline composition, unchanged -----------------
    def _prep_bias(
        self, a: torch.Tensor, z: torch.Tensor, mask: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        if mask is None:
            mask = a.new_ones(a.shape[:-1])

        batch_dims = a.shape[:-2]
        mask = mask.expand((*batch_dims, -1))

        mask_bias = (self.inf * (mask - 1))[..., None, None, :]
        biases = [mask_bias]

        z = self.layer_norm_z(z)
        z = self.linear_z(z)
        z = _permute_final_dims(z, [2, 0, 1])
        biases.append(z)

        return biases

    def _reference(
        self,
        a: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        biases = self._prep_bias(a=a, z=z, mask=mask)
        a = self.layer_norm_a(a, s) if self.use_ada_layer_norm else self.layer_norm_a(a)
        a = self.mha(q_x=a, kv_x=a, biases=biases)
        if self.use_ada_layer_norm:
            a = self.sigmoid(self.linear_ada_out(s)) * a
        return a

    # -- fused path ---------------------------------------------------------
    # Order of the tensors the fused kernels read. `_build_plan` unpacks this by
    # name, so the invalidation key and the launch arguments cannot describe
    # different tensors. Unused slots are None, which keeps the tuple's shape
    # stable across configurations.
    _SOURCES = (
        "layer_norm_a.weight", "layer_norm_a.bias",
        "layer_norm_a.layer_norm_s.weight", "layer_norm_a.linear_g.weight",
        "layer_norm_a.linear_g.bias", "layer_norm_a.linear_s.weight",
        "linear_ada_out.weight", "linear_ada_out.bias",
        "layer_norm_z.weight", "layer_norm_z.bias", "linear_z.weight",
        "mha.linear_q.weight", "mha.linear_q.bias", "mha.linear_k.weight",
        "mha.linear_v.weight", "mha.linear_g.weight", "mha.linear_o.weight",
    )

    def _sources(self) -> tuple:
        ada = self.use_ada_layer_norm
        aln = self.layer_norm_a
        lnz = self.layer_norm_z
        mha = self.mha
        return (
            None if ada else aln.weight,
            None if ada else aln.bias,
            aln.layer_norm_s.weight if ada else None,
            aln.linear_g.weight if ada else None,
            aln.linear_g.bias if ada else None,
            aln.linear_s.weight if ada else None,
            self.linear_ada_out.weight if ada else None,
            self.linear_ada_out.bias if ada else None,
            lnz.weight, lnz.bias, self.linear_z.weight,
            mha.linear_q.weight, mha.linear_q.bias, mha.linear_k.weight,
            mha.linear_v.weight,
            None if mha.linear_g is None else mha.linear_g.weight,
            mha.linear_o.weight,
        )

    def _expected(self) -> tuple:
        """Shape each source must have, derived from the module's own config.

        Parallel to `_SOURCES`. Checked for every source that is present, which
        is what stops the kernels reading a wrong-extent or transposed weight.
        """
        cq, cs, cz = self.c_q, self.c_s, self.c_z
        hd = self.mha.no_heads * self.mha.c_hidden
        return ((cq,), (cq,), (cs,), (cq, cs), (cq,), (cq, cs), (cq, cs), (cq,),
                (cz,), (cz,), (self.mha.no_heads, cz),
                (hd, cq), (hd,), (hd, cq), (hd, cq), (hd, cq), (cq, hd))

    def _struct(self) -> tuple:
        """Every value the kernels bake into a grid, a constexpr or a constant.

        These are public attributes the baseline reads on each forward, so they
        are mutable between calls -- changing `inf` after a plan was cached left
        the fused path using the old value. Anything that reaches a launch
        argument belongs here.
        """
        mha = self.mha
        return (self.inf, self.c_q, self.c_s, self.c_z, self.use_ada_layer_norm,
                mha.no_heads, mha.c_hidden, mha.linear_g is not None)

    def _fused_ok(self, srcs, a, z, s, mask) -> bool:
        """Preconditions for the fused path. Everything it rejects is handled by
        the reference path, so this narrows the optimization, never the contract."""
        mha = self.mha
        if mha.linear_g is None or not a.is_cuda or a.dim() < 2:
            return False
        n, c_q = a.shape[-2:]
        if c_q != self.c_q or z.dim() < 3:
            return False
        # The score matrix and one pair-embedding row each live in registers.
        if not (_np2(n) <= 128 and _np2(self.c_z) <= 256
                and _np2(mha.c_hidden) <= 256):
            return False
        if tuple(z.shape[-3:]) != (n, n, self.c_z):
            return False
        if not (_one_batch(a, 2) and _one_batch(z, 3)):
            return False
        if mask is not None and not (mask.dim() >= 1 and mask.shape[-1] == n
                                     and _one_batch(mask, 1)):
            return False
        if self.layer_norm_z.weight is None:
            return False
        ada = self.use_ada_layer_norm
        if ada:
            aln = self.layer_norm_a
            if s is None or s.dim() < 2 or tuple(s.shape[-2:]) != (n, self.c_s):
                return False
            if not _one_batch(s, 2):
                return False
            if aln.layer_norm_s.weight is None or aln.layer_norm_a.weight is not None:
                return False
        # The output is `empty_like(a)`, so the baseline's broadcast must agree.
        lead = [z.shape[:-3]]
        if ada:
            lead.append(s.shape[:-2])
        if mask is not None:
            lead.append(mask.shape[:-1])
        if not _leading_ok(a.shape[:-2], *lead):
            return False
        inputs = (a, z, mask) + ((s,) if ada else ())
        if any(t is not None and t.dtype is not torch.bfloat16 for t in inputs):
            return False
        return _sources_ok(srcs, self._expected(), a.device)

    def _build_plan(self, srcs, a, z, s, mask) -> _Plan:
        if not self._fused_ok(srcs, a, z, s, mask):
            return _Plan(False)
        (lna_w, lna_b, lns_w, adg_w, adg_b, ads_w, ado_w, ado_b,
         lnz_w, lnz_b, lz_w, q_w, q_b, k_w, v_w, g_w, o_w) = srcs
        mha = self.mha
        n, c_q = a.shape[-2:]
        c_s, c_z = self.c_s, self.c_z
        h, d = mha.no_heads, mha.c_hidden
        hd = h * d
        ada = self.use_ada_layer_norm
        buf = _Workspace(a.device)
        nil = buf.get("nil", (1,))
        an = buf.get("an", (n, c_q))
        zb = buf.get("zb", (h * n * n,))
        og = buf.get("og", (n, hd))
        ada_pre = buf.get("ada", (n, c_q)) if ada else nil

        bn = max(16, _np2(n))
        bq = _tile(c_q, 128)
        bc = _tile(c_q, 128)
        bz = _tile(n * n, 128)
        n_qt = _cdiv(c_q, bq)
        return _Plan(
            True,
            launch=(
                _Launcher(
                    _apb_prep, (n_qt + _cdiv(n * n, bz),),
                    {"ADA": ada, "LNA_W": lna_w is not None,
                     "LNA_B": lna_b is not None,
                     "LNZ_B": lnz_b is not None, "BN": bn, "BQ": bq,
                     "BK": _tile(min(c_q, c_s) if ada else c_q, 128), "BZ": bz,
                     "BCZ": _np2(c_z), "BH": max(16, _np2(h))}),
                _Launcher(
                    _apb_attn, (h,),
                    {"HAS_MASK": mask is not None, "Q_BIAS": q_b is not None,
                     "BN": bn, "BD": max(16, _np2(d)), "BK": _tile(c_q, 128)}),
                _Launcher(
                    _gated_out, (1, _cdiv(c_q, bc)),
                    {"ADA": ada, "BR": bn, "BC": bc, "BK": _tile(hd, 128)}),
            ),
            args=(
                (an, ada_pre, zb,
                 nil if lna_w is None else lna_w,
                 nil if lna_b is None else lna_b,
                 nil if lns_w is None else lns_w,
                 nil if adg_w is None else adg_w,
                 nil if adg_b is None else adg_b,
                 nil if ads_w is None else ads_w,
                 nil if ado_w is None else ado_w,
                 nil if ado_b is None else ado_b,
                 lnz_w, nil if lnz_b is None else lnz_b, lz_w,
                 n, c_q, c_s, c_z, h, n_qt, _EPS),
                (an, zb, og, q_w, nil if q_b is None else q_b, k_w, v_w, g_w,
                 n, c_q, d, hd, math.sqrt(d), self.inf),
                (og, ada_pre, o_w, n, c_q, hd),
            ),
            nil=nil, buf=buf,
        )

    def _signature(self, srcs, a, z, s, mask) -> tuple:
        return (a.shape, z.shape, a.dtype, a.device, mask is None,
                None if mask is None else mask.shape,
                None if s is None else s.shape,
                _raw_stream(a.device) if a.is_cuda else 0,
                self._struct(), _source_key(srcs))

    def _fused_available(self, a, z, s=None, mask=None) -> bool:
        """Whether a forward with these arguments would take the fused path.

        Exactly the condition ``forward`` applies, so a test that asserts on this
        is asserting on the real dispatch rather than on a subset of it.
        """
        srcs = self._sources()
        return (not _needs_autograd(srcs, a, z, s, mask)
                and _contig(a, z, s, mask)
                and self._build_plan(srcs, a, z, s, mask).ok)

    def forward(
        self,
        a: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        use_high_precision_attention: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            a:    [*, N, C_q] token/atom-level embedding
            z:    [*, N, N, C_z] pair embedding
            s:    [*, N, C_s] single embedding (for AdaLN)
            mask: [*, N] mask

        Returns:
            [*, N, C_q] attention update
        """
        srcs = self._sources()
        if _needs_autograd(srcs, a, z, s, mask) or not _contig(a, z, s, mask):
            return self._reference(a, z, s, mask)
        sig = self._signature(srcs, a, z, s, mask)
        plan = self._plans.get(sig)
        if plan is None:
            plan = self._plans.put(sig, self._build_plan(srcs, a, z, s, mask))
        if not plan.ok:
            return self._reference(a, z, s, mask)

        prep, attn, gate = plan.launch
        args, nil = plan.args, plan.nil
        out = torch.empty_like(a)
        prep(a, s if s is not None else nil, z, *args[0])
        attn(mask if mask is not None else nil, *args[1])
        gate(out, *args[2])
        return out


class CrossAttentionPairBias(nn.Module):
    """AF3 Algorithm 24: Cross-attention with pair bias for atom attention.

    Sequence-local: queries are ``n_query``-row blocks of the atom axis and keys
    are ``n_key`` gathered rows per block. Separate ``layer_norm_a_q`` /
    ``layer_norm_a_k`` AdaLNs, and no ``layer_norm_z``.

    Reference: openfold3/core/model/layers/attention_pair_bias.py

    Args:
        c_q: Input dimension of query/key/value
        c_s: Single activation channel dimension (for AdaLN)
        c_z: Pair activation channel dimension
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        use_ada_layer_norm: Whether to use AdaLN-Zero conditioning
        n_query: Block size for queries
        n_key: Block size for keys
        gating: Whether to gate output
        inf: Large constant for masking
    """

    def __init__(
        self,
        c_q: int,
        c_k: int = 0,
        c_v: int = 0,
        c_s: int = 0,
        c_z: int = 16,
        c_hidden: int = 16,
        no_heads: int = 4,
        use_ada_layer_norm: bool = True,
        n_query: int | None = None,
        n_key: int | None = None,
        gating: bool = True,
        inf: float = 1e9,
    ):
        super().__init__()
        c_k = c_k or c_q
        c_v = c_v or c_q

        self.c_q = c_q
        self.c_s = c_s
        self.c_z = c_z
        self.inf = inf
        self.use_ada_layer_norm = use_ada_layer_norm
        self.n_query = n_query
        self.n_key = n_key

        if use_ada_layer_norm:
            self.layer_norm_a_q = AdaLN(c_a=c_q, c_s=c_s)
            self.layer_norm_a_k = AdaLN(c_a=c_q, c_s=c_s)
            self.linear_ada_out = Linear(c_s, c_q, bias=True)
        else:
            self.layer_norm_a_q = LayerNorm(c_q)
            self.layer_norm_a_k = LayerNorm(c_q)

        self.linear_z = Linear(c_z, no_heads, bias=False)

        self.sigmoid = Sigmoid()

        self.mha = OF3Attention(
            c_q=c_q, c_k=c_k, c_v=c_v,
            c_hidden=c_hidden, no_heads=no_heads, gating=gating,
            q_bias=True,
        )

        self._plans = _PlanCache()

    # -- reference path: the baseline composition, unchanged -----------------
    def _reference(
        self,
        a: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        from .alphafold3_atom_attention import (
            _convert_single_rep_to_blocks, _apply_block_indices,
        )

        batch_dims = a.shape[:-2]
        n_atom, n_dim = a.shape[-2:]

        if mask is None:
            mask = a.new_ones(a.shape[:-1])

        a_query, a_key, block_mask = _convert_single_rep_to_blocks(
            ql=a, n_query=self.n_query, n_key=self.n_key, atom_mask=mask,
        )

        mask_bias = (self.inf * (block_mask - 1))[..., None, :, :]
        biases = [mask_bias]

        z_bias = self.linear_z(z)
        z_bias = _permute_final_dims(z_bias, [2, 0, 1])
        biases.append(z_bias)

        if self.use_ada_layer_norm:
            s_q, s_k, _ = _apply_block_indices(
                ql=s, n_query=self.n_query, n_key=self.n_key, atom_mask=mask,
            )
            a_q = self.layer_norm_a_q(a_query, s_q)
            a_k = self.layer_norm_a_k(a_key, s_k)
        else:
            a_q = self.layer_norm_a_q(a_query)
            a_k = self.layer_norm_a_k(a_key)

        a_out = self.mha(q_x=a_q, kv_x=a_k, biases=biases)
        a_out = a_out.reshape((*batch_dims, -1, n_dim))[..., :n_atom, :]

        if self.use_ada_layer_norm:
            a_out = self.sigmoid(self.linear_ada_out(s)) * a_out

        return a_out

    # -- fused path ---------------------------------------------------------
    # Order of the tensors the fused kernels read; see AttentionPairBias._SOURCES.
    _SOURCES = (
        "layer_norm_a_q.weight", "layer_norm_a_q.bias",
        "layer_norm_a_k.weight", "layer_norm_a_k.bias",
        "layer_norm_a_q.layer_norm_s.weight", "layer_norm_a_q.linear_g.weight",
        "layer_norm_a_q.linear_g.bias", "layer_norm_a_q.linear_s.weight",
        "layer_norm_a_k.layer_norm_s.weight", "layer_norm_a_k.linear_g.weight",
        "layer_norm_a_k.linear_g.bias", "layer_norm_a_k.linear_s.weight",
        "linear_ada_out.weight", "linear_ada_out.bias", "linear_z.weight",
        "mha.linear_q.weight", "mha.linear_q.bias", "mha.linear_k.weight",
        "mha.linear_v.weight", "mha.linear_g.weight", "mha.linear_o.weight",
    )

    def _sources(self) -> tuple:
        ada = self.use_ada_layer_norm
        lq, lk, mha = self.layer_norm_a_q, self.layer_norm_a_k, self.mha
        return (
            None if ada else lq.weight, None if ada else lq.bias,
            None if ada else lk.weight, None if ada else lk.bias,
            lq.layer_norm_s.weight if ada else None,
            lq.linear_g.weight if ada else None,
            lq.linear_g.bias if ada else None,
            lq.linear_s.weight if ada else None,
            lk.layer_norm_s.weight if ada else None,
            lk.linear_g.weight if ada else None,
            lk.linear_g.bias if ada else None,
            lk.linear_s.weight if ada else None,
            self.linear_ada_out.weight if ada else None,
            self.linear_ada_out.bias if ada else None,
            self.linear_z.weight,
            mha.linear_q.weight, mha.linear_q.bias, mha.linear_k.weight,
            mha.linear_v.weight,
            None if mha.linear_g is None else mha.linear_g.weight,
            mha.linear_o.weight,
        )

    def _expected(self) -> tuple:
        """Shape each source must have, derived from the module's own config."""
        cq, cs, cz = self.c_q, self.c_s, self.c_z
        h = self.mha.no_heads
        hd = h * self.mha.c_hidden
        return ((cq,), (cq,), (cq,), (cq,),
                (cs,), (cq, cs), (cq,), (cq, cs),
                (cs,), (cq, cs), (cq,), (cq, cs),
                (cq, cs), (cq,), (h, cz),
                (hd, cq), (hd,), (hd, cq), (hd, cq), (hd, cq), (cq, hd))

    def _struct(self) -> tuple:
        """Every value the kernels bake into a grid, a constexpr or a constant."""
        mha = self.mha
        return (self.inf, self.c_q, self.c_s, self.c_z, self.use_ada_layer_norm,
                self.n_query, self.n_key, mha.no_heads, mha.c_hidden,
                mha.linear_g is not None)

    def _fused_ok(self, srcs, a, z, s, mask) -> bool:
        """Preconditions for the fused path. Everything it rejects is handled by
        the reference path, so this narrows the optimization, never the contract."""
        mha = self.mha
        nq, nk = self.n_query, self.n_key
        if mha.linear_g is None or not a.is_cuda or a.dim() < 2:
            return False
        if not (isinstance(nq, int) and isinstance(nk, int) and nq > 0 and nk > 0):
            return False
        n_atom, c_q = a.shape[-2:]
        if c_q != self.c_q or z.dim() < 4:
            return False
        # One query block's scores, and a whole channel row, live in registers.
        if not (_np2(c_q) <= 256 and _np2(nk) <= 256 and _np2(nq) <= 128
                and _np2(mha.c_hidden) <= 128 and _np2(self.c_z) <= 128):
            return False
        n_blocks = _cdiv(n_atom, nq)
        if tuple(z.shape[-4:]) != (n_blocks, nq, nk, self.c_z):
            return False
        if not (_one_batch(a, 2) and _one_batch(z, 4)):
            return False
        if mask is not None and not (mask.dim() >= 1 and mask.shape[-1] == n_atom
                                     and _one_batch(mask, 1)):
            return False
        ada = self.use_ada_layer_norm
        if ada:
            aq, ak = self.layer_norm_a_q, self.layer_norm_a_k
            if s is None or s.dim() < 2 or tuple(s.shape[-2:]) != (n_atom, self.c_s):
                return False
            if not _one_batch(s, 2):
                return False
            if (aq.layer_norm_s.weight is None or ak.layer_norm_s.weight is None
                    or aq.layer_norm_a.weight is not None):
                return False
            # With mask=None the baseline builds a rank-(a.dim()-1) mask and then
            # tries to expand it for `s`, which raises when `a` outranks `s`.
            # Matching that means declining the case, not computing through it.
            if mask is None and a.dim() > s.dim():
                return False
        lead = [z.shape[:-4]]
        if ada:
            lead.append(s.shape[:-2])
        if mask is not None:
            lead.append(mask.shape[:-1])
        if not _leading_ok(a.shape[:-2], *lead):
            return False
        inputs = (a, z, mask) + ((s,) if ada else ())
        if any(t is not None and t.dtype is not torch.bfloat16 for t in inputs):
            return False
        return _sources_ok(srcs, self._expected(), a.device)

    def _build_plan(self, srcs, a, z, s, mask) -> _Plan:
        if not self._fused_ok(srcs, a, z, s, mask):
            return _Plan(False)
        (lq_w, lq_b, lk_w, lk_b, lnsq_w, gq_w, gq_b, sq_w,
         lnsk_w, gk_w, gk_b, sk_w, ado_w, ado_b, lz_w,
         q_w, q_b, k_w, v_w, g_w, o_w) = srcs
        mha = self.mha
        n_atom, c_q = a.shape[-2:]
        c_s, c_z = self.c_s, self.c_z
        nq, nk = self.n_query, self.n_key
        h, d = mha.no_heads, mha.c_hidden
        hd = h * d
        ada = self.use_ada_layer_norm
        n_blocks = _cdiv(n_atom, nq)
        n_pad = n_blocks * nq
        n_zrow = n_blocks * nq * nk
        buf = _Workspace(a.device)
        nil = buf.get("nil", (1,))
        # n_pad + 1 rows, not n_pad: the prep kernel stores a_q_n, a_k_n and
        # ada_pre under one predicate that covers the sentinel row, so a shorter
        # buffer would be written one row past its end.
        aqn = buf.get("aqn", (n_pad + 1, c_q))
        akn = buf.get("akn", (n_pad + 1, c_q))
        og = buf.get("og", (n_pad, hd))
        zb = buf.get("zb", (h * n_zrow,))
        ada_pre = buf.get("ada", (n_pad + 1, c_q)) if ada else nil

        br = _tile(n_pad, 64)
        bcq = max(16, _np2(c_q))
        bz = _tile(n_zrow, 128)
        bro = _tile(n_atom, 64)
        n_rt = _cdiv(n_pad + 1, br)
        return _Plan(
            True,
            launch=(
                _Launcher(
                    _capb_prep, (n_rt + _cdiv(n_zrow, bz),),
                    {"ADA": ada, "LN_W": lq_w is not None,
                     "LN_B": lq_b is not None, "BR": br,
                     "BCQ": bcq, "BK": _tile(c_s, 128) if ada else 16, "BZ": bz,
                     "BCZ": max(16, _np2(c_z)), "BH": max(16, _np2(h))},
                    # The c_s loop is usually a single iteration, so pipelining
                    # only inflates shared memory for the five weight tiles.
                    {"num_stages": 1, "num_warps": 8}),
                _Launcher(
                    _capb_attn, (n_blocks, h),
                    {"HAS_MASK": mask is not None, "Q_BIAS": q_b is not None,
                     "BQ": max(16, _np2(nq)), "BKN": max(16, _np2(nk)),
                     "BD": max(16, _np2(d)), "BK": _tile(c_q, 128),
                     "BMS": _tile(n_atom, 1024)},
                    {"num_stages": 1, "num_warps": 8}),
                _Launcher(
                    _gated_out, (_cdiv(n_atom, bro), 1),
                    {"ADA": ada, "BR": bro, "BC": bcq, "BK": _tile(hd, 128)}),
            ),
            args=(
                (aqn, akn, ada_pre, zb, lz_w,
                 nil if lnsq_w is None else lnsq_w,
                 nil if lnsk_w is None else lnsk_w,
                 nil if gq_w is None else gq_w, nil if gq_b is None else gq_b,
                 nil if sq_w is None else sq_w,
                 nil if gk_w is None else gk_w, nil if gk_b is None else gk_b,
                 nil if sk_w is None else sk_w,
                 nil if ado_w is None else ado_w, nil if ado_b is None else ado_b,
                 nil if lq_w is None else lq_w, nil if lq_b is None else lq_b,
                 nil if lk_w is None else lk_w, nil if lk_b is None else lk_b,
                 n_atom, n_pad, c_q, c_s, c_z, h, n_zrow, n_rt, _EPS),
                (aqn, akn, zb, og, q_w, nil if q_b is None else q_b, k_w, v_w,
                 g_w, n_atom, n_pad, nq, nk, c_q, n_zrow, d, hd,
                 nq // 2 + (-nk // 2), math.sqrt(d), self.inf),
                (og, ada_pre, o_w, n_atom, c_q, hd),
            ),
            nil=nil, buf=buf,
        )

    def _signature(self, srcs, a, z, s, mask) -> tuple:
        return (a.shape, z.shape, a.dtype, a.device, mask is None,
                None if mask is None else mask.shape,
                None if s is None else s.shape,
                _raw_stream(a.device) if a.is_cuda else 0,
                self._struct(), _source_key(srcs))

    def _fused_available(self, a, z, s=None, mask=None) -> bool:
        """Whether a forward with these arguments would take the fused path."""
        srcs = self._sources()
        return (not _needs_autograd(srcs, a, z, s, mask)
                and _contig(a, z, s, mask)
                and self._build_plan(srcs, a, z, s, mask).ok)

    def forward(
        self,
        a: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            a:    [*, N_atom, C_q] atom-level embedding
            z:    [*, N_blocks, n_key, n_key, C_z] blocked pair embedding
            s:    [*, N_atom, C_s] single embedding (for AdaLN)
            mask: [*, N_atom] mask

        Returns:
            [*, N_atom, C_q] attention update
        """
        srcs = self._sources()
        if _needs_autograd(srcs, a, z, s, mask) or not _contig(a, z, s, mask):
            return self._reference(a, z, s, mask)
        sig = self._signature(srcs, a, z, s, mask)
        plan = self._plans.get(sig)
        if plan is None:
            plan = self._plans.put(sig, self._build_plan(srcs, a, z, s, mask))
        if not plan.ok:
            return self._reference(a, z, s, mask)

        prep, attn, gate = plan.launch
        args, nil = plan.args, plan.nil
        out = torch.empty_like(a)
        prep(a, s if s is not None else nil, z, *args[0])
        attn(mask if mask is not None else nil, *args[1])
        gate(out, *args[2])
        return out
