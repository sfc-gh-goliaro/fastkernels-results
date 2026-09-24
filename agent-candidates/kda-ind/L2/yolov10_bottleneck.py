"""YOLOv10 bottleneck block fused into two Triton convolutions.

The reference block runs ``conv -> batch_norm -> silu`` twice and then adds the residual, which
costs fifteen GPU kernels: two convolutions doing the arithmetic, and thirteen more for
BatchNorm, the elementwise activations and add, and the NCHW<->NHWC layout conversions cuDNN
needs. Every benchmarked shape has well under ten microseconds of intrinsic arithmetic, so the
measured latency is almost entirely per-kernel overhead and kernel count is the lever.

Two things collapse it to two kernels. In eval mode BatchNorm is an affine map with fixed
statistics, so it folds into the preceding convolution's weight and bias exactly. And a 3x3
same-padded stride-1 convolution over NCHW is an implicit GEMM whose activation tap is just a
shift of the flat pixel index, so it needs no im2col and no layout change -- which lets the bias,
SiLU and residual ride along in the epilogue.

The fold is derived lazily on the first eval forward, never in ``__init__``: the weights are
loaded after construction, so anything derived from their values in ``__init__`` would be stale.
It is invalidated whenever the parameters are reloaded, the module is moved or recast, or it
changes between training and eval.

"Two kernels" describes a forward once the fold exists. The forward that builds it also runs the
handful of elementwise operations the fold itself needs, and any forward that declines the fast
path runs the reference block instead.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .yolov10_conv import YOLOConv

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - the fallback path covers this
    _HAS_TRITON = False

_FAST_DTYPES = (torch.float16, torch.bfloat16)

# The epilogue is this activation and nothing else. Matched by exact type rather than isinstance:
# a subclass may override forward, and the kernel would silently apply SiLU regardless.
_SUPPORTED_ACT = type(YOLOConv.default_act)

# Block count the heuristic aims for. Measured, not derived: with this target the heuristic lands on
# the fastest configuration found by profile/sweep_blocks.py for all five benchmarked shapes, across
# a 40-160 configuration sweep per shape. Nsight Compute explains why it wants to be well above the
# 148 SMs -- achieved occupancy is only 8.8-10.8%, so several blocks per SM are needed before the
# machine has enough work in flight to hide the load latency it is bound by.
_TARGET_BLOCKS = 400

# tl.dot needs each tile dimension to be a power of two and at least 16 (the float16 MMA
# minimum), so channel counts below 16 are handled by padding the tile and masking rather
# than by shrinking it.
_MIN_BLOCK = 16
# 32 output channels per tile measured fastest everywhere the channel count allows it; a wider
# channel tile mostly costs parallelism on these shapes.
_MAX_BLOCK_N = 32
_MAX_BLOCK_K = 128
_MAX_BLOCK_M = 128


def _fold_conv_bn(conv: nn.Module, bn: nn.Module | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the eval-mode ``(weight, bias)`` of ``conv`` with ``bn`` folded in, in float32.

    ``scale = gamma / sqrt(running_var + eps)`` scales each output channel, so
    ``weight = conv.weight * scale`` and ``bias = (conv.bias - running_mean) * scale + beta``.
    Everything is computed in float32: ``running_mean`` and ``running_var`` are float32 buffers
    even after the module's parameters are cast to float16, and doing the divide and square root
    in half precision loses accuracy for no gain. The caller casts to the activation dtype.

    ``bn is None`` means the block was already fused by ``YOLOConv.fuse()``, which deletes ``bn``
    and installs a real convolution bias; folding again would apply the scale twice.
    """
    weight = conv.weight.detach().float()
    conv_bias = conv.bias.detach().float() if conv.bias is not None else None
    if bn is None:
        return weight, conv_bias if conv_bias is not None else weight.new_zeros(weight.shape[0])
    scale = bn.weight.detach().float() / torch.sqrt(bn.running_var.detach().float() + bn.eps)
    if conv_bias is None:
        conv_bias = torch.zeros_like(scale)
    bias = (conv_bias - bn.running_mean.detach().float()) * scale + bn.bias.detach().float()
    return weight * scale.view(-1, 1, 1, 1), bias


def _tensor_state(tensor: torch.Tensor | None) -> tuple | None:
    """Identity, in-place revision, dtype and device of one tensor.

    Identity matters as much as the version counter: replacing a buffer with a *different* tensor
    that happens to carry the same version would otherwise be invisible to a version-only
    signature, and the fold built from the old buffer would be reused.
    """
    if tensor is None:
        return None
    return (id(tensor), tensor._version, tensor.dtype, tensor.device)


def _next_pow2(n: int) -> int:
    return 1 << max(0, n - 1).bit_length()


def _pixel_channel_blocks(pixels: int, cout: int, batch: int) -> tuple[int, int, int, int]:
    """Pick ``(BLOCK_M, BLOCK_N, num_warps)`` for a ``pixels x cout`` output per image.

    Splitting the channel tile below ``cout`` is what makes the small shapes reachable at all:
    ``[1, 128, 20, 20]`` has only 400 pixels, so a tile covering all 128 channels is four blocks
    against roughly 148 SMs. With the channel tile fixed at 32 (16 when there are only 16
    channels), the pixel tile is taken as large as it can be while the grid still reaches
    ``_TARGET_BLOCKS``, which trades the least parallelism for the most operand reuse.
    """
    block_n = min(_MAX_BLOCK_N, max(_MIN_BLOCK, _next_pow2(cout)))
    channel_tiles = -(-cout // block_n)
    block_m = _MIN_BLOCK
    candidate = _MAX_BLOCK_M
    while candidate >= _MIN_BLOCK:
        if -(-pixels // candidate) * channel_tiles * batch >= _TARGET_BLOCKS:
            block_m = candidate
            break
        candidate //= 2
    # Four warps won every configuration in the sweep, including the large tiles where eight
    # looked plausible: more warps split the same accumulator without adding resident blocks, and
    # this kernel is short of blocks rather than short of threads.
    return block_m, block_n, _NUM_WARPS, _NUM_STAGES


# Software-pipelining depth. Nsight Compute showed the pipelining buffers, not anything the kernel
# allocates itself, are what cap occupancy on the small shapes: 76.8 KB of shared memory per block
# held [1, 64, 40, 40] to 3 blocks per SM at an achieved occupancy of 8.8%. Since the kernel is
# latency-bound and nowhere near a bandwidth roof, trading pipeline depth for resident blocks is
# the lever that profile pointed at. Swept in profile/sweep_blocks.py.
_NUM_STAGES = 2
_NUM_WARPS = 4


# Measured launch configurations, keyed by (N, H, W, Cin, Cout, add, dtype) -- the full key, so a
# configuration is only reused for a call it was actually measured on. Populated from the sweep in
# profile/sweep_blocks.py, which times whole-module forwards through the harness's own _time_module
# and rejects any configuration that does not reproduce the reference exactly. Both convolution
# invocations of a benchmarked shape get their own entry, because they differ in Cin and Cout and in
# whether the residual is added.
#
# The heuristic below covers everything not in this table, so an unmeasured shape still runs fast.
_MEASURED_PLANS: dict[tuple, tuple[int, int, int, int, int]] = {
    # (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages), from profile/attribution/
    # sweep_blockk_run.txt. Both invocations of a shape share an entry: they differ only in whether
    # the residual is added, and the whole-module timing that selected them cannot attribute a
    # difference to one invocation or the other without a quadratic sweep.
    #
    # The interesting column is BLOCK_K, and it is why this table exists rather than a formula. A
    # reduction width *wider* than Cin is legal -- the ci >= Cin lanes mask off -- and on two shapes
    # it beats the natural next_pow2(Cin) choice, because masked lanes cost less than extra trips
    # around the reduction loop. No closed form the heuristic could compute would have found that.
    (1, 20, 20, 128, 128, False, torch.float16): (16, 32, 128, 4, 2),
    (1, 20, 20, 128, 128, True, torch.float16): (16, 32, 128, 4, 2),
    # [4, 16, 160, 160]: BLOCK_K 32 against a natural 16, worth 56.3 -> 52.2 us on the weakest case.
    (4, 160, 160, 16, 16, False, torch.float16): (128, 16, 32, 4, 2),
    (4, 160, 160, 16, 16, True, torch.float16): (128, 16, 32, 4, 2),
    (4, 40, 40, 64, 64, False, torch.float16): (32, 32, 64, 4, 2),
    (4, 40, 40, 64, 64, True, torch.float16): (32, 32, 64, 4, 2),
    (1, 40, 40, 64, 64, False, torch.float16): (16, 32, 64, 4, 2),
    (1, 40, 40, 64, 64, True, torch.float16): (16, 32, 64, 4, 2),
    # [1, 32, 80, 80]: BLOCK_K 64 against a natural 32, worth 24.4 -> 23.6 us.
    (1, 80, 80, 32, 32, False, torch.float16): (16, 32, 64, 4, 2),
    (1, 80, 80, 32, 32, True, torch.float16): (16, 32, 64, 4, 2),
}


def _channel_reduction_block(cin: int) -> int:
    return min(_MAX_BLOCK_K, max(_MIN_BLOCK, _next_pow2(cin)))


if _HAS_TRITON:

    @triton.jit
    def _conv3x3_silu_kernel(
        x_ptr, w_ptr, bias_ptr, res_ptr, out_ptr,
        H, W, HW, Cin, Cout,
        sx_n, sx_c, sx_h, sx_w,
        sr_n, sr_c, sr_h, sr_w,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        ADD_RESIDUAL: tl.constexpr,
    ):
        """3x3 same-padded stride-1 NCHW convolution with a bias + SiLU + residual epilogue.

        One program computes a ``BLOCK_M`` x ``BLOCK_N`` tile of (flat pixel, output channel)
        for one image. ``w_ptr`` is the folded weight pre-permuted to ``[9, Cin, Cout]`` so the
        tap's slice is already a GEMM operand; ``bias_ptr`` stays float32.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        n = tl.program_id(2)

        p = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        in_p = p < HW
        # Recovering the row and column per element is what makes a BLOCK_M spanning several
        # image rows safe: the column is then bounds-checked independently of the row, so a
        # left-edge tap cannot wrap onto the previous row's right edge.
        h = p // W
        w = p % W
        co = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        in_c = co < Cout

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        x_image = x_ptr + n * sx_n
        for k0 in range(0, Cin, BLOCK_K):
            ci = k0 + tl.arange(0, BLOCK_K)
            in_k = ci < Cin
            for kh in tl.static_range(3):
                hi = h + kh - 1
                ok_row = in_p & (hi >= 0) & (hi < H)
                for kw in tl.static_range(3):
                    wi = w + kw - 1
                    ok = ok_row & (wi >= 0) & (wi < W)
                    # Clamp the spatial offset where the mask is false, so no lane forms an
                    # address below the tensor base -- the unclamped top-left tap would be
                    # -(W + 1). Masked-off channel and tail lanes still form addresses past the
                    # end; those are never dereferenced, which is what tl.load's mask guarantees.
                    idx = tl.where(ok, hi * sx_h + wi * sx_w, 0)
                    a = tl.load(x_image + ci[None, :] * sx_c + idx[:, None],
                                mask=ok[:, None] & in_k[None, :], other=0.0)
                    b = tl.load(w_ptr + (kh * 3 + kw) * Cin * Cout
                                + ci[:, None] * Cout + co[None, :],
                                mask=in_k[:, None] & in_c[None, :], other=0.0)
                    acc = tl.dot(a, b, acc)

        acc += tl.load(bias_ptr + co, mask=in_c, other=0.0)[None, :]
        y = acc * tl.sigmoid(acc)
        keep = in_p[:, None] & in_c[None, :]
        if ADD_RESIDUAL:
            # The residual is added after SiLU, matching x + silu(bn2(conv2(...))).
            res = res_ptr + n * sr_n + co[None, :] * sr_c + (h * sr_h + w * sr_w)[:, None]
            y += tl.load(res, mask=keep, other=0.0).to(tl.float32)
        tl.store(out_ptr + n * Cout * HW + co[None, :] * HW + p[:, None],
                 y.to(out_ptr.dtype.element_ty), mask=keep)


class _FoldedConv:
    """One convolution's folded weight, packed for the kernel, plus its float32 bias."""

    __slots__ = ("weight", "bias", "block_k")

    def __init__(self, block: YOLOConv, dtype: torch.dtype):
        weight, bias = _fold_conv_bn(block.conv, getattr(block, "bn", None))
        cout, cin = weight.shape[0], weight.shape[1]
        # [Cout, Cin, kh, kw] -> [kh, kw, Cin, Cout] -> [9, Cin, Cout], so tap (kh, kw) lives
        # at index kh * 3 + kw. This is a plain re-index with no spatial flip: F.conv2d is a
        # cross-correlation, and the kernel reads x[h + kh - 1, w + kw - 1] to match.
        self.weight = weight.permute(2, 3, 1, 0).contiguous().reshape(9, cin, cout).to(dtype)
        self.bias = bias
        self.block_k = _channel_reduction_block(cin)


class YOLOBottleneck(nn.Module):
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k=(3, 3), e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = YOLOConv(c1, c_, k[0], 1)
        self.cv2 = YOLOConv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self._folded: tuple[_FoldedConv, _FoldedConv] | None = None
        self._folded_dtype: torch.dtype | None = None
        self._blocks: dict[tuple, tuple[int, int, int, int, int]] = {}
        self._scratch: torch.Tensor | None = None
        self._signature: tuple | None = None
        # Loading into a submodule directly -- cv1.load_state_dict(...), or deeper still
        # cv1.bn.load_state_dict(...) -- never reaches this module's own _load_from_state_dict,
        # so hook every descendant that owns state. Resetting more than once per load is
        # harmless, and the hook costs nothing on the forward path.
        for module in self.modules():
            if module is not self:
                module.register_load_state_dict_post_hook(self._on_descendant_load)

    def _on_descendant_load(self, module: nn.Module, incompatible_keys) -> None:  # noqa: ARG002
        self.reset_plan()

    def _describe(self, dtype: torch.dtype, device: torch.device) -> tuple | None:
        """A structural signature of everything the fold and the dispatch decision depend on.

        Returned fresh on every forward and compared against the signature the cached fold was
        built from, so nothing decided earlier is trusted. ``None`` means the current state is not
        one the kernel implements -- or not one the reference itself would accept -- and the caller
        must use the reference so that equivalent results, or equivalent errors, come out.

        This exists because the alternative -- deciding once and reusing forever -- is wrong in
        both directions. A convolution whose padding or activation was reassigned after
        construction would keep a fast path it no longer qualifies for, and a BatchNorm whose
        statistics moved would keep a fold that no longer matches. Both are reachable through
        ordinary module APIs and neither is reachable through the benchmark.
        """
        parts: list = [self.add]
        for block in (self.cv1, self.cv2):
            conv = block.conv
            if (tuple(conv.weight.shape[2:]) != (3, 3) or tuple(conv.stride) != (1, 1)
                    or tuple(conv.padding) != (1, 1) or tuple(conv.dilation) != (1, 1)
                    or conv.groups != 1
                    or type(block.act) is not _SUPPORTED_ACT):
                return None
            bn = getattr(block, "bn", None)
            fused = getattr(block, "_is_fused", False)
            # Only two states are coherent: unfused means a BatchNorm and no convolution bias,
            # fused means a convolution bias and no BatchNorm. Anything else is a half-applied
            # fuse() that the reference cannot run either, so it must not be read as one or the
            # other -- deleting `bn` without setting the flag used to look exactly like "fused".
            if fused != (bn is None) or fused != (conv.bias is not None):
                return None
            # A training-mode BatchNorm uses batch statistics, which no offline fold reproduces.
            if bn is not None and bn.training:
                return None
            # The reference would refuse a dtype or device the input does not share, so the fast
            # path must refuse it too rather than quietly casting and returning an answer.
            affine = [conv.weight, conv.bias]
            if bn is not None:
                affine += [bn.weight, bn.bias]
            for tensor in affine:
                if tensor is not None and (tensor.dtype is not dtype or tensor.device != device):
                    return None
            parts.append((_tensor_state(conv.weight), _tensor_state(conv.bias), fused))
            if bn is None:
                parts.append(None)
                continue
            # The running statistics stay float32 under the harness's parameter-only cast and the
            # fold promotes them anyway, so their dtype is deliberately unconstrained -- but they
            # still have to live on the same device.
            for tensor in (bn.running_mean, bn.running_var):
                if tensor.device != device:
                    return None
            parts.append((id(bn), bn.eps,
                          _tensor_state(bn.weight), _tensor_state(bn.bias),
                          _tensor_state(bn.running_mean), _tensor_state(bn.running_var)))
        return tuple(parts)

    def reset_plan(self) -> None:
        """Discard the folded weights so the next eval forward re-derives them.

        Call this after mutating a parameter in place by any route other than
        ``load_state_dict`` or ``_apply`` (``.to()``, ``.half()``, ``.cuda()``, ...), which are
        already hooked below.
        """
        self._folded = None
        self._folded_dtype = None
        self._signature = None
        self._scratch = None

    def _apply(self, *args, **kwargs):
        self.reset_plan()
        return super()._apply(*args, **kwargs)

    def _load_from_state_dict(self, *args, **kwargs):
        self.reset_plan()
        return super()._load_from_state_dict(*args, **kwargs)

    def train(self, mode: bool = True):
        # A training forward updates the BatchNorm running statistics, so a fold built before it
        # is stale once the module returns to eval.
        self.reset_plan()
        return super().train(mode)

    def __getstate__(self):
        # Never carry the derived tensors through a pickle or a deepcopy: they would either
        # resurrect a stale fold or drag a GPU scratch buffer along, neither of which the
        # reference module does.
        state = dict(super().__getstate__())
        state.update(_folded=None, _folded_dtype=None, _signature=None, _scratch=None)
        return state

    def _folded_convs(self, dtype: torch.dtype, signature: tuple) -> tuple[_FoldedConv, _FoldedConv]:
        """The folded weights, cached independently of the spatial shape.

        Rebuilt whenever the activation dtype or the structural signature changes, so a cached
        fold is only ever used for the state it was derived from.
        """
        if self._folded is None or self._folded_dtype is not dtype or self._signature != signature:
            self._folded = (_FoldedConv(self.cv1, dtype), _FoldedConv(self.cv2, dtype))
            self._folded_dtype = dtype
            self._signature = signature
        return self._folded

    def _blocking(self, key: tuple, pixels: int, cout: int,
                  batch: int) -> tuple[int, int, int, int, int]:
        """The launch configuration for one convolution invocation.

        ``key`` is the full ``(N, H, W, Cin, Cout, add, dtype)`` tuple, so a measured entry is only
        reused for a call it was measured on. Anything absent from the table falls back to the
        heuristic. Cached per key because both are pure functions of the key and this sits on the
        hot path of an operator whose cost is dominated by host work.
        """
        cfg = self._blocks.get(key)
        if cfg is None:
            measured = _MEASURED_PLANS.get(key)
            if measured is None:
                block_m, block_n, warps, stages = _pixel_channel_blocks(pixels, cout, batch)
                measured = (block_m, block_n, _channel_reduction_block(key[3]), warps, stages)
            cfg = self._blocks[key] = measured
        return cfg

    def _reference(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y

    def _run(self, x: torch.Tensor, folded: _FoldedConv, out: torch.Tensor,
             residual: torch.Tensor | None) -> None:
        n, cin, h, w = x.shape
        cout = out.shape[1]
        pixels = h * w
        key = (n, h, w, cin, cout, residual is not None, x.dtype)
        block_m, block_n, block_k, num_warps, num_stages = self._blocking(key, pixels, cout, n)
        grid = (-(-pixels // block_m), -(-cout // block_n), n)
        # With no residual the pointer is never read, so pass the output and zero strides
        # rather than paying for another stride() call.
        source = residual if residual is not None else out
        source_stride = residual.stride() if residual is not None else (0, 0, 0, 0)
        _conv3x3_silu_kernel[grid](
            x, folded.weight, folded.bias, source, out,
            h, w, pixels, cin, cout,
            *x.stride(), *source_stride,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            ADD_RESIDUAL=residual is not None,
            num_warps=num_warps, num_stages=num_stages,
        )

    def _intermediate(self, shape: torch.Size, x: torch.Tensor) -> torch.Tensor:
        """A reusable buffer for the first convolution's output.

        Safe to reuse because it is never returned: a caller such as YOLOv10's ``C2f``
        concatenates bottleneck outputs, so the *output* must stay freshly allocated, but
        nothing outside this module ever observes the intermediate. Not reentrant -- one module
        instance must not be driven from two threads at once.
        """
        buf = self._scratch
        if buf is None or buf.shape != shape or buf.dtype != x.dtype or buf.device != x.device:
            buf = self._scratch = torch.empty(shape, dtype=x.dtype, device=x.device)
        return buf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not _HAS_TRITON or self.training
                or torch.is_grad_enabled() or x.dim() != 4 or not x.is_cuda
                or x.dtype not in _FAST_DTYPES
                or x.shape[1] != self.cv1.conv.weight.shape[1]
                # Clamping a masked tap's spatial offset to 0 assumes the base pointer is the
                # tensor's lowest address. PyTorch does not currently produce negative strides,
                # so this only pins the invariant the clamp relies on.
                or min(x.stride()) < 0):
            self.reset_plan()
            return self._reference(x)
        # Describing the state and folding are inside the fallback boundary too: a module in a
        # shape this code did not anticipate should reach the reference, not raise from here.
        try:
            signature = self._describe(x.dtype, x.device)
            plan = None if signature is None else self._folded_convs(x.dtype, signature)
        except Exception:  # noqa: BLE001 - an undescribable module falls back
            plan = None
        if plan is None:
            # An unsupported state must not leave a fold behind for a later forward that qualifies.
            self.reset_plan()
            return self._reference(x)
        try:
            f1, f2 = plan
            n, _, h, w = x.shape
            mid = self._intermediate(torch.Size((n, f1.weight.shape[2], h, w)), x)
            out = torch.empty((n, f2.weight.shape[2], h, w), dtype=x.dtype, device=x.device)
            self._run(x, f1, mid, None)
            self._run(mid, f2, out, x if self.add else None)
        except Exception:  # noqa: BLE001 - a planning or launch failure falls back
            # Synchronous failures only. A fault raised asynchronously by the device surfaces at
            # the next synchronization, past this handler.
            return self._reference(x)
        return out
