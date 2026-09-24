"""Gated RMSNorm with element-wise multiplicative gate (L1).

Implements ``RMSNorm(x) * activation(z)`` in a single fused Triton kernel.
Hosts two vendored vLLM variants (kernel bodies unchanged):

- ``rmsnorm_fn`` / ``RMSNormGated`` (from FLA ``layernorm_guard``), used by
  Qwen3-Next's GDN output gating (``norm_before_gate=True``, ``swish``).
- ``rms_norm_gated`` / ``FusedRMSNormGated`` (from FLA ``layernorm_gated``),
  used by Kimi-Delta attention's output ``o_norm``.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable

import torch
import torch.nn as nn
import triton
import triton.language as tl


def cdiv(a: int, b: int) -> int:
    return -(a // -b)


def next_power_of_2(n: int) -> int:
    return 1 if n < 1 else 1 << (n - 1).bit_length()


def num_compute_units(device_id: int = 0) -> int:
    return torch.cuda.get_device_properties(device_id).multi_processor_count


def input_guard(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Ensure input tensors are contiguous and set the device from them."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_args = (
            i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args
        )
        contiguous_kwargs = {
            k: (v if not isinstance(v, torch.Tensor) else v.contiguous())
            for k, v in kwargs.items()
        }

        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        if tensor is not None:
            ctx = torch.accelerator.device_index(tensor.device.index)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["B"] is not None,
        "HAS_Z": lambda args: args["Z"] is not None,
    }
)
@triton.jit
def layer_norm_fwd_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Z,  # pointer to the other branch
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_z_row,
    M,  # number of rows in X
    N: tl.constexpr,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    # Map the program id to the starting row of X and Y it should compute.
    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    group = tl.program_id(1)

    # Create 2D tile: [ROWS_PER_BLOCK, BLOCK_N]
    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, BLOCK_N)

    # Compute offsets for 2D tile
    row_offsets = rows[:, None] * stride_x_row
    col_offsets = cols[None, :] + group * N

    # Base pointers
    X_base = X + row_offsets + col_offsets
    Y_base = Y + rows[:, None] * stride_y_row + col_offsets

    # Create mask for valid rows and columns
    row_mask = rows[:, None] < M
    col_mask = cols[None, :] < N
    mask = row_mask & col_mask

    # Load input data with 2D tile
    x = tl.load(X_base, mask=mask, other=0.0).to(tl.float32)

    if HAS_Z and not NORM_BEFORE_GATE:
        Z_base = Z + rows[:, None] * stride_z_row + col_offsets
        z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            x *= z * tl.sigmoid(z)
        elif ACTIVATION == "sigmoid":
            x *= tl.sigmoid(z)

    # Compute mean and variance per row (reduce along axis 1)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=1) / N  # Shape: [ROWS_PER_BLOCK]
        # Store mean for each row
        mean_offsets = group * M + rows
        mean_mask = rows < M
        tl.store(Mean + mean_offsets, mean, mask=mean_mask)
        # Broadcast mean back to 2D for subtraction
        xbar = tl.where(mask, x - mean[:, None], 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N  # Shape: [ROWS_PER_BLOCK]
    else:
        xbar = tl.where(mask, x, 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N  # Shape: [ROWS_PER_BLOCK]
        mean = 0.0  # Placeholder for RMS norm

    rstd = tl.rsqrt(var + eps)  # Shape: [ROWS_PER_BLOCK]

    # Store rstd for each row
    rstd_offsets = group * M + rows
    rstd_mask = rows < M
    tl.store(Rstd + rstd_offsets, rstd, mask=rstd_mask)

    # Load weights and biases (broadcast across rows)
    w_offsets = cols + group * N
    w_mask = cols < N
    w = tl.load(W + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

    if HAS_BIAS:
        b = tl.load(B + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

    # Normalize and apply linear transformation
    if not IS_RMS_NORM:
        x_hat = (x - mean[:, None]) * rstd[:, None]
    else:
        x_hat = x * rstd[:, None]

    y = x_hat * w[None, :] + b[None, :] if HAS_BIAS else x_hat * w[None, :]

    if HAS_Z and NORM_BEFORE_GATE:
        Z_base = Z + rows[:, None] * stride_z_row + col_offsets
        z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            y *= z * tl.sigmoid(z)
        elif ACTIVATION == "sigmoid":
            y *= tl.sigmoid(z)

    # Write output
    tl.store(Y_base, y, mask=mask)


def calc_rows_per_block(M: int, device: torch.device) -> int:
    sm_count = num_compute_units(device.index)
    rows_per_block = next_power_of_2(cdiv(M, 2 * sm_count))
    rows_per_block = min(rows_per_block, 4)
    return rows_per_block


def layer_norm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    z: torch.Tensor = None,
    out: torch.Tensor = None,
    group_size: int = None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = False,
    activation: str = "swish",
):
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    # allocate output
    if out is not None:
        assert out.shape == x.shape
    else:
        out = torch.empty_like(x)
    assert out.stride(-1) == 1
    mean = (
        torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
        if not is_rms_norm
        else None
    )
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    # Calculate rows per block based on SM count
    rows_per_block = calc_rows_per_block(M, x.device)
    # Update grid to use rows_per_block
    grid = (cdiv(M, rows_per_block), ngroups)
    layer_norm_fwd_kernel[grid](
        x,
        out,
        weight,
        bias,
        z,
        mean,
        rstd,
        x.stride(0),
        out.stride(0),
        z.stride(0) if z is not None else 0,
        M,
        group_size,
        eps,
        BLOCK_N=BLOCK_N,
        ROWS_PER_BLOCK=rows_per_block,
        HAS_BIAS=bias is not None,
        HAS_Z=z is not None,
        NORM_BEFORE_GATE=norm_before_gate,
        IS_RMS_NORM=is_rms_norm,
        num_warps=num_warps,
        ACTIVATION=activation,
    )
    return out, mean, rstd


def _layer_norm_fn_impl(
    x,
    weight,
    bias,
    z=None,
    eps=1e-6,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
    activation: str = "swish",
):
    """Triton layer/RMS norm with optional gating.

    If z is not None, computes norm(x) * silu(z) when norm_before_gate,
    else norm(x * silu(z)).

    This calls the triton kernel directly. The original code wrapped this
    in a torch.autograd.Function (LayerNormFn) to save tensors for a
    backward pass, but vLLM is inference-only so there is no backward pass.
    The autograd wrapper also prevented torch.compile/dynamo from tracing
    through the function due to its @staticmethod forward.
    """
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    if x.stride(-1) != 1:
        x = x.contiguous()
    if z is not None:
        assert z.shape == x_shape_og
        z = z.reshape(-1, z.shape[-1])
        if z.stride(-1) != 1:
            z = z.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    y, _, _ = layer_norm_fwd(
        x,
        weight,
        bias,
        eps,
        z=z,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
        activation=activation,
    )
    return y.reshape(x_shape_og)


@input_guard
def rmsnorm_fn(
    x,
    weight,
    bias,
    z=None,
    eps=1e-6,
    group_size=None,
    norm_before_gate=True,
    activation: str = "swish",
):
    return _layer_norm_fn_impl(
        x, weight, bias, z, eps, group_size, norm_before_gate, True, activation
    )


# ---------------------------------------------------------------------------
# Fast path (this file's optimization): single-pass streaming gated RMSNorm
# specialized for the only shape family this operator ever sees --
# ``x, z : bfloat16[M, 128]``, ``norm_before_gate=True``, ``swish``.
#
# Design notes (measured on B200, see ITERATIONS.md):
#   * N is a constexpr equal to BLOCK_N, so there is no column mask and the
#     row reduction is a pure intra-warp shuffle over 16/32 lanes.
#   * ``mean``/``rstd`` are never read by anyone, so they are neither
#     allocated nor stored (the vendored path allocates + writes rstd).
#   * x, z and w are all loaded up front, before the reduction; the gate is
#     computed off the reduction's critical path.
#   * sigmoid is built from ``exp2`` (one ``ex2.approx.f32``) instead of
#     ``exp``.
#   * Two tile shapes, both 128 threads/CTA: 4 rows (4 elems/thread) when the
#     launch is the bottleneck, 16 rows (16 elems/thread, 128-bit accesses)
#     when DRAM bandwidth is.
#   * The Python wrapper is a straight line: no ``input_guard``, no reshape,
#     no asserts, no ``get_device_properties``. At small M the harness window
#     is host-bound, so this is worth as much as the kernel itself.
#   * Two entry points rather than one ``EVEN_M`` constexpr: when BLOCK_M
#     divides M -- which is every shape this operator is ever given -- the
#     launch does not carry ``M`` at all.
#
# Why the tile split is where it is, and why there is nothing below it:
# the benchmark's timed window is ``copy(x); copy(z); module(x, z)`` behind a
# 265 MB L2 flush, and each operation on that stream is charged a whole ~2.05 us
# front-end slot. A kernel fits in one slot iff its own span (measured with
# in-kernel ``%globaltimer``) is under ~0.48 us. At M=16 four CTAs moving 4 KB
# span ~0.45 us and the window is 13.3 us; from M=416 up, a bare ``y = x`` copy
# already spans 0.61 us at the best of 15 tiles, so two slots (15.36 us) is the
# floor there for *any* correct kernel. Hence: pick the tile that keeps the
# small end's span minimal (4 rows, 4 CTAs at M=16) and the large end's
# bandwidth maximal (16 rows), and do not expect the middle to move.
# ---------------------------------------------------------------------------

_LOG2E = tl.constexpr(1.4426950408889634)

# Below this many rows the launch, not DRAM, sets the time: use the narrow tile
# so more CTAs are in flight (and, at M=16, so the kernel's span stays inside one
# ~2.05 us front-end slot). Above it, stream with 16 elems/thread; that also
# keeps M=7120 at 445 CTAs, since 3560+ CTAs cost the front-end enough extra
# dispatch time to push that shape from two slots to three (15.33 -> 17.41 us).
_WIDE_TILE_MIN_ROWS = 4096


@triton.jit
def _rcp_approx(v):
    """``rcp.approx.ftz.f32``: one MUFU op instead of the ~10-instruction
    ``div.rn.f32`` Triton emits for ``a / b``. Worth 4 us at M=262144
    (87.1 -> 83.0 us); with that gone the gate is free and the kernel sits on
    the DRAM floor. Its ~1 ulp error is far inside a bf16 mantissa, and the
    argument here is ``1 + exp2(...) >= 1``, so FTZ can only bite where the true
    gate has already underflowed bf16."""
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;", "=r,r", [v],
        dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _rmsg_row_span(BLOCK_M: tl.constexpr):
    """Descending CTA -> row mapping.

    Whoever produced x and z wrote them most recently at their *high* rows, so at
    M=262144 (where x + z = 134 MB just about fills the 132 MB L2) walking rows
    downward reads the still-resident lines before our own output stores evict
    them. Measured 83.1 -> 81.0 us, i.e. 0.1 us off the 80.9 us 2R+1W floor.
    Free at every other M."""
    return (tl.num_programs(0) - 1 - tl.program_id(0)) * BLOCK_M + tl.arange(0, BLOCK_M)


@triton.jit
def _rmsg_math(x, z, w, EPS: tl.constexpr, N: tl.constexpr):
    """swish-gated RMSNorm on an already-loaded fp32 tile."""
    # swish(z) = z * sigmoid(z) = z * rcp(1 + exp2(-z * log2(e)))
    gate = z * _rcp_approx(1.0 + tl.math.exp2(-z * _LOG2E))
    rstd = tl.rsqrt(tl.sum(x * x, axis=1) * (1.0 / N) + EPS)
    return (x * rstd[:, None]) * w[None, :] * gate


@triton.jit
def _rmsg_fwd_even(
    X,  # *bf16 [M, N]      input
    Z,  # *bf16 [M, N]      gate
    W,  # *bf16 [N]         weight
    Y,  # *bf16 [M, N]      output
    EPS: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """BLOCK_M divides M, so there is no row mask and ``M`` is not needed at all.

    Every captured shape takes this path (all captured M are multiples of 16), so
    the launch carries four pointers and nothing else. The mask-free body is
    instruction-for-instruction what the previous ``EVEN_M=True`` specialization
    compiled to; dropping the dead ``M`` parameter only shrinks the launch
    payload. It is kept because it is strictly less work, not because it shows
    up: an interleaved A/B against the parent measured a tie, and a dedicated
    probe over 6/7/8/11 PTX parameters found argument count does not move the
    window at any benched shape (see ITERATIONS.md).
    """
    cols = tl.arange(0, N)
    offs = _rmsg_row_span(BLOCK_M)[:, None] * N + cols[None, :]
    x = tl.load(X + offs).to(tl.float32)
    z = tl.load(Z + offs).to(tl.float32)
    w = tl.load(W + cols).to(tl.float32)
    y = _rmsg_math(x, z, w, EPS, N)
    tl.store(Y + offs, y.to(Y.dtype.element_ty))


@triton.jit
def _rmsg_fwd(
    X,  # *bf16 [M, N]      input
    Z,  # *bf16 [M, N]      gate
    W,  # *bf16 [N]         weight
    Y,  # *bf16 [M, N]      output
    M,
    EPS: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Row-masked variant, for an M that BLOCK_M does not divide."""
    cols = tl.arange(0, N)
    rows = _rmsg_row_span(BLOCK_M)
    offs = rows[:, None] * N + cols[None, :]
    keep = rows[:, None] < M
    x = tl.load(X + offs, mask=keep, other=0.0).to(tl.float32)
    z = tl.load(Z + offs, mask=keep, other=0.0).to(tl.float32)
    w = tl.load(W + cols).to(tl.float32)
    y = _rmsg_math(x, z, w, EPS, N)
    tl.store(Y + offs, y.to(Y.dtype.element_ty), mask=keep)


def _rmsg_plan(M: int) -> tuple:
    """``(grid, BLOCK_M, unmasked)`` for M rows. Cached per module instance."""
    block = 16 if M >= _WIDE_TILE_MIN_ROWS else 4
    while block > 1 and M < block:
        block //= 2
    return ((cdiv(M, block),), block, M % block == 0)


__targets__ = ["RMSNormGated", "FusedRMSNormGated"]


class RMSNormGated(nn.Module):
    """Fused gated RMSNorm: ``out = activation(z) * RMSNorm(x, weight)``."""

    def __init__(self, hidden_size: int, eps: float = 1e-6,
                 norm_before_gate: bool = True, activation: str = "swish"):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.norm_before_gate = norm_before_gate
        self.activation = activation
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # Straight-line fast-path state, kept out of ``_parameters`` /
        # ``_buffers`` so ``forward`` never pays ``nn.Module.__getattr__``.
        object.__setattr__(self, "_fast", hidden_size == 128 and norm_before_gate
                           and activation in ("swish", "silu"))
        object.__setattr__(self, "_w", self.weight)
        object.__setattr__(self, "_plans", {})

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if (self._fast and x.dtype == torch.bfloat16 and x.ndim == 2
                and x.shape[1] == 128 and z.shape == x.shape
                and z.dtype == x.dtype
                and x.is_contiguous() and z.is_contiguous()):
            # ``_fast`` already pins hidden_size == 128, so ``x.shape[1] == 128``
            # is what guarantees the weight load stays in bounds.
            M = x.shape[0]
            plan = self._plans.get(M)
            if plan is None:
                plan = self._plans[M] = _rmsg_plan(M)
            y = torch.empty_like(x)
            if plan[2]:
                _rmsg_fwd_even[plan[0]](
                    x, z, self._w, y,
                    EPS=self.eps, N=128, BLOCK_M=plan[1],
                    num_warps=4, num_stages=1,
                )
            else:
                _rmsg_fwd[plan[0]](
                    x, z, self._w, y, M,
                    EPS=self.eps, N=128, BLOCK_M=plan[1],
                    num_warps=4, num_stages=1,
                )
            return y
        return rmsnorm_fn(
            x, self.weight, bias=None,
            z=z, eps=self.eps,
            norm_before_gate=self.norm_before_gate,
            activation=self.activation,
        )


# --- FLA layernorm_gated variant (used by Kimi-Delta o_norm) ---

class CustomOp(nn.Module):
    def __init__(self, *, enforce_enable: bool = False, compile_native: bool = False):
        super().__init__()
        self._forward_method = (
            self.forward_hip if torch.version.hip is not None else self.forward_cuda
        )

    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs):
        raise NotImplementedError

    def forward_hip(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)

    @classmethod
    def register(cls, name: str, dynamic_arg_dims=None):
        def decorator(op_cls):
            op_cls.name = name
            op_cls._dynamic_arg_dims = dynamic_arg_dims
            return op_cls

        return decorator


@triton.heuristics(
    {
        "STORE_RESIDUAL_OUT": lambda args: args["residual_out"] is not None,
        "HAS_RESIDUAL": lambda args: args["residual"] is not None,
        "HAS_WEIGHT": lambda args: args["w"] is not None,
        "HAS_BIAS": lambda args: args["b"] is not None,
    }
)
@triton.jit
def layer_norm_gated_fwd_kernel(
    x,  # pointer to the input
    g,  # pointer to the gate
    y,  # pointer to the output
    w,  # pointer to the weights
    b,  # pointer to the biases
    residual,  # pointer to the residual
    residual_out,  # pointer to the residual
    mean,  # pointer to the mean
    rstd,  # pointer to the 1/std
    eps,  # epsilon to avoid division by zero
    T,  # number of rows in x
    D: tl.constexpr,  # number of columns in x
    BT: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_t = tl.program_id(0)

    o_d = tl.arange(0, BD)
    m_d = o_d < D

    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    if HAS_RESIDUAL:
        p_res = tl.make_block_ptr(
            residual, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0)
        )
        b_x += tl.load(p_res, boundary_check=(0, 1)).to(tl.float32)
    if STORE_RESIDUAL_OUT:
        p_res_out = tl.make_block_ptr(
            residual_out, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0)
        )
        tl.store(p_res_out, b_x.to(p_res_out.dtype.element_ty), boundary_check=(0, 1))
    if not IS_RMS_NORM:
        b_mean = tl.sum(b_x, axis=1) / D
        p_mean = tl.make_block_ptr(mean, (T,), (1,), (i_t * BT,), (BT,), (0,))
        tl.store(p_mean, b_mean.to(p_mean.dtype.element_ty), boundary_check=(0,))
        b_xbar = tl.where(m_d[None, :], b_x - b_mean[:, None], 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
    else:
        b_xbar = tl.where(m_d[None, :], b_x, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)

    p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (i_t * BT,), (BT,), (0,))
    tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), boundary_check=(0,))

    if HAS_WEIGHT:
        b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + o_d, mask=m_d).to(tl.float32)
    b_x_hat = (
        (b_x - b_mean[:, None]) * b_rstd[:, None]
        if not IS_RMS_NORM
        else b_x * b_rstd[:, None]
    )
    b_y = b_x_hat * b_w[None, :] if HAS_WEIGHT else b_x_hat
    if HAS_BIAS:
        b_y = b_y + b_b[None, :]

    # swish/sigmoid output gate
    p_g = tl.make_block_ptr(g, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    if ACTIVATION == "swish" or ACTIVATION == "silu":
        b_y = b_y * b_g * tl.sigmoid(b_g)
    elif ACTIVATION == "sigmoid":
        b_y = b_y * tl.sigmoid(b_g)

    # Write output
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "STORE_RESIDUAL_OUT": lambda args: args["residual_out"] is not None,
        "HAS_RESIDUAL": lambda args: args["residual"] is not None,
        "HAS_WEIGHT": lambda args: args["w"] is not None,
        "HAS_BIAS": lambda args: args["b"] is not None,
    }
)
@triton.jit
def layer_norm_gated_fwd_kernel1(
    x,  # pointer to the input
    g,  # pointer to the gate
    y,  # pointer to the output
    w,  # pointer to the weights
    b,  # pointer to the biases
    residual,  # pointer to the residual
    residual_out,  # pointer to the residual
    mean,  # pointer to the mean
    rstd,  # pointer to the 1/std
    eps,  # epsilon to avoid division by zero
    D: tl.constexpr,  # number of columns in x
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    g += i_t * D
    if HAS_RESIDUAL:
        residual += i_t * D
    if STORE_RESIDUAL_OUT:
        residual_out += i_t * D

    o_d = tl.arange(0, BD)
    m_d = o_d < D
    b_x = tl.load(x + o_d, mask=m_d, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        b_x += tl.load(residual + o_d, mask=m_d, other=0.0).to(tl.float32)
    if STORE_RESIDUAL_OUT:
        tl.store(residual_out + o_d, b_x, mask=m_d)
    if not IS_RMS_NORM:
        b_mean = tl.sum(b_x, axis=0) / D
        tl.store(mean + i_t, b_mean)
        b_xbar = tl.where(m_d, b_x - b_mean, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    else:
        b_xbar = tl.where(m_d, b_x, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)
    tl.store(rstd + i_t, b_rstd)

    if HAS_WEIGHT:
        b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + o_d, mask=m_d).to(tl.float32)
    b_x_hat = (b_x - b_mean) * b_rstd if not IS_RMS_NORM else b_x * b_rstd
    b_y = b_x_hat * b_w if HAS_WEIGHT else b_x_hat
    if HAS_BIAS:
        b_y = b_y + b_b

    # swish/sigmoid output gate
    b_g = tl.load(g + o_d, mask=m_d, other=0.0).to(tl.float32)
    if ACTIVATION == "swish" or ACTIVATION == "silu":
        b_y = b_y * b_g * tl.sigmoid(b_g)
    elif ACTIVATION == "sigmoid":
        b_y = b_y * tl.sigmoid(b_g)

    # Write output
    tl.store(y + o_d, b_y, mask=m_d)


def layer_norm_gated_fwd(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = "swish",
    eps: float = 1e-5,
    residual: torch.Tensor = None,
    out_dtype: torch.dtype = None,
    residual_dtype: torch.dtype = None,
    is_rms_norm: bool = False,
):
    if residual is not None:
        residual_dtype = residual.dtype
    T, D = x.shape
    if residual is not None:
        assert residual.shape == (T, D)
    if weight is not None:
        assert weight.shape == (D,)
    if bias is not None:
        assert bias.shape == (D,)
    # allocate output
    y = x if out_dtype is None else torch.empty_like(x, dtype=out_dtype)
    if residual is not None or (
        residual_dtype is not None and residual_dtype != x.dtype
    ):
        residual_out = torch.empty(T, D, device=x.device, dtype=residual_dtype)
    else:
        residual_out = None
    mean = (
        torch.empty((T,), dtype=torch.float, device=x.device)
        if not is_rms_norm
        else None
    )
    rstd = torch.empty((T,), dtype=torch.float, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps

    if D <= 512:
        BT = 32
        layer_norm_gated_fwd_kernel[(cdiv(T, BT),)](
            x=x,
            g=g,
            y=y,
            w=weight,
            b=bias,
            residual=residual,
            residual_out=residual_out,
            mean=mean,
            rstd=rstd,
            eps=eps,
            T=T,
            D=D,
            BD=BD,
            BT=BT,
            ACTIVATION=activation,
            IS_RMS_NORM=is_rms_norm,
            num_warps=4,
        )
    else:
        layer_norm_gated_fwd_kernel1[(T,)](
            x=x,
            g=g,
            y=y,
            w=weight,
            b=bias,
            residual=residual,
            residual_out=residual_out,
            mean=mean,
            rstd=rstd,
            eps=eps,
            D=D,
            BD=BD,
            ACTIVATION=activation,
            IS_RMS_NORM=is_rms_norm,
            num_warps=4,
        )
    # residual_out is None if residual is None and residual_dtype == input_dtype
    return y, mean, rstd, residual_out if residual_out is not None else x


def rms_norm_gated(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = "swish",
    residual: torch.Tensor | None = None,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    eps: float = 1e-6,
):
    x_shape_og = x.shape
    # reshape input data into 2D tensor
    x = x.contiguous().reshape(-1, x.shape[-1])
    g = g.contiguous().reshape(-1, g.shape[-1])
    if residual is not None:
        assert residual.shape == x_shape_og
        residual = residual.contiguous().reshape(-1, residual.shape[-1])
    residual_dtype = (
        residual.dtype
        if residual is not None
        else (torch.float if residual_in_fp32 else None)
    )
    y, _, _, residual_out = layer_norm_gated_fwd(
        x=x,
        g=g,
        weight=weight,
        bias=bias,
        activation=activation,
        eps=eps,
        residual=residual,
        residual_dtype=residual_dtype,
        is_rms_norm=True,
    )
    y = y.reshape(x_shape_og)
    return y if not prenorm else (y, residual_out.reshape(x_shape_og))


@CustomOp.register("fused_rms_norm_gated")
class FusedRMSNormGated(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        activation: str = "swish",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps
        self.activation = activation

        if self.activation not in ["swish", "silu", "sigmoid"]:
            raise ValueError(f"Unsupported activation: {self.activation}")

        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def forward_native(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        residual: torch.Tensor | None = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ) -> torch.Tensor:
        """Decomposed PyTorch ops for torch.compile/inductor fusion."""
        # TODO(https://github.com/vllm-project/vllm/issues/36175): implement
        # native residual/prenorm path and unify with RMSNormGated.
        # For now, fall back to the triton kernel.
        if residual is not None or prenorm:
            return self.forward_cuda(x, g, residual, prenorm, residual_in_fp32)
        x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_float * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            x_normed = x_normed * self.weight.float()
        g_float = g.float()
        if self.activation in ("swish", "silu"):
            out = x_normed * g_float * torch.sigmoid(g_float)
        else:  # sigmoid
            out = x_normed * torch.sigmoid(g_float)
        return out.to(x.dtype)

    def forward_cuda(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        residual: torch.Tensor | None = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ) -> torch.Tensor:
        return rms_norm_gated(
            x,
            g,
            self.weight,
            self.bias,
            self.activation,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )
