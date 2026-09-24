"""Oasis temporal axial attention: two cuBLAS GEMMs around one fused Triton kernel.

The operator is a pair of skinny GEMMs with a data shuffle between them. At the
captured shapes the `to_qkv` projection is 5.44 GFLOP and `to_out` is 1.81 GFLOP,
while the attention itself is ~21 MFLOP -- 0.3% of the arithmetic. Yet the
baseline's largest single kernel is an SDPA flash-attention kernel serving
`(2304, 16, 6, 64)`: at sequence length 6 every SDPA backend is far off its
roofline, because the shape is 2304 independent 6-token sequences.

So the projections stay on cuBLAS (it shares both operands across its tile grid,
which is worth more than the round trip through HBM it costs) and *everything
between them* collapses into one kernel: the rotary embedding, the causal
attention over time, and the relayout back to token-major order. Writing the
context directly in the order `to_out` consumes is what removes the baseline's
two `permute(...).reshape(...)` copies and the `torch.cat` inside
`oasis_apply_rotary_emb`, taking the kernel count per forward from ~25 to 3.

Fast path (CUDA, fp16/bf16, grad disabled, time <= 64); everything else runs a
reference forward built from the same ops as the baseline, so the module stays a
drop-in for the DiT blocks that share one rotary embedding across all of them.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import triton
import triton.language as tl

from ..L1.dense_attention import DenseAttention
from ..L1.linear import Linear
from ..L1.oasis_rotary import OasisRotaryEmbedding

# Above this the single-tile softmax below stops being reasonable (the score
# block is `time x time` in fp32), so hand longer sequences to the reference
# path. The real workloads reach 32 frames.
_MAX_FUSED_SEQ = 64

# Tiny tiles (a `[16,64]` fp16 operand is 2 KB), so one warp per program keeps
# the launch cheap: measured against 2, 4 and 8 at the captured shapes, one warp
# won at every `time`. Longer sequences carry a `time x time` fp32 score block, so
# they get proportionally more lanes. A pinned function of the tile, not a runtime
# search -- `triton.autotune` would spawn threads during the timing window.
_NUM_WARPS_BY_TILE = {16: 1, 32: 4, 64: 8}


@triton.jit
def _rope_attn_relayout_kernel(
    qkv_ptr, cos_ptr, sin_ptr, ctx_ptr,
    stride_qkv_row, stride_ctx_row, stride_table_row,
    seq_len, hw, head_dim, n_pairs, qkv_group, scale,
    BT: tl.constexpr, BJ: tl.constexpr, BD: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """One program per (batch*spatial position, head): rotary, attention, relayout.

    `qkv` is token-major `[bsz*time*hw, 3*heads*head_dim]`, the natural output of
    the projection GEMM; time is the *slowest* varying index within a batch
    element, so the `time` rows this program needs are `hw` apart. Reading them
    strided and writing the context back token-major is exactly the shuffle the
    baseline pays two copies for.
    """
    pid_pos = tl.program_id(0)
    pid_head = tl.program_id(1)
    batch = pid_pos // hw
    spatial = pid_pos % hw

    offs_t = tl.arange(0, BT)
    offs_j = tl.arange(0, BJ)
    offs_d = tl.arange(0, BD)
    mask_t = offs_t < seq_len
    mask_j = offs_j < n_pairs
    mask_d = offs_d < head_dim

    # Row of the first time step for this spatial position, then stride by `hw`.
    row0 = (batch * seq_len) * hw + spatial
    rows = (row0 + offs_t * hw) * stride_qkv_row
    head_off = pid_head * head_dim

    # Read q and k whole and *then* separate the even and odd lanes of each
    # rotary pair. Addressing the lanes directly (`2j`, `2j+1`) also works and
    # needs no reshape, but it touches every 32-byte sector while using only half
    # the bytes in each. Profiled at T=6: the strided form issues 995,328 global
    # load sectors, this one 221,184 for the same data, at the full 32 bytes per
    # sector -- and measures 17.6 us against 21.6 us in isolation.
    #
    # The interleave is never materialized either way: the score only needs
    # `sum_d q_d k_d = sum_j (q_even k_even + q_odd k_odd)`, so it is the sum of
    # two `[BT,BJ] x [BJ,BT]` products. That also covers a rotary width narrower
    # than the head for free -- untouched pairs just contribute unrotated terms.
    q_base = qkv_ptr + rows[:, None] + head_off
    k_base = q_base + qkv_group
    wide_mask = mask_t[:, None] & mask_d[None, :]
    q_all = tl.load(q_base + offs_d[None, :], mask=wide_mask, other=0.0)
    k_all = tl.load(k_base + offs_d[None, :], mask=wide_mask, other=0.0)
    q_even, q_odd = tl.split(tl.reshape(q_all, (BT, BJ, 2)))
    k_even, k_odd = tl.split(tl.reshape(k_all, (BT, BJ, 2)))

    # `cos[t, j]` is the value the baseline's `repeat_interleave(2)` puts at both
    # lanes of pair j, so one table entry serves both.
    table = offs_t[:, None] * stride_table_row + offs_j[None, :]
    pair_mask = mask_t[:, None] & mask_j[None, :]
    cos = tl.load(cos_ptr + table, mask=pair_mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + table, mask=pair_mask, other=0.0).to(tl.float32)

    in_dtype = q_even.dtype
    qe, qo = q_even.to(tl.float32), q_odd.to(tl.float32)
    ke, ko = k_even.to(tl.float32), k_odd.to(tl.float32)
    # Round back to the input dtype, mirroring the baseline handing fp16 q/k to
    # SDPA rather than keeping the rotated values in higher precision.
    q_even = (qe * cos - qo * sin).to(in_dtype)
    q_odd = (qo * cos + qe * sin).to(in_dtype)
    k_even = (ke * cos - ko * sin).to(in_dtype)
    k_odd = (ko * cos + ke * sin).to(in_dtype)

    scores = tl.dot(q_even, tl.trans(k_even), out_dtype=tl.float32)
    scores = tl.dot(q_odd, tl.trans(k_odd), scores, out_dtype=tl.float32)
    scores = scores * scale

    # The mask deliberately has no `t < seq_len` term. That is what guarantees no
    # row is ever fully masked for seq_len >= 1: row t keeps keys
    # 0..min(t, seq_len-1) when causal and 0..seq_len-1 otherwise, both
    # non-empty. With max subtraction the denominator therefore always contains
    # the exp(0) = 1 term from the row maximum and cannot be zero -- including
    # for the padding rows that exist because `tl.dot` needs M, N >= 16 while
    # time is 6. Adding `t < seq_len` here would reintroduce the NaN.
    valid = offs_t[None, :] < seq_len
    if IS_CAUSAL:
        valid = valid & (offs_t[None, :] <= offs_t[:, None])
    # A *finite* fp32 sentinel: -inf would give NaN for a fully masked row
    # (`-inf - -inf`), and in fp16 any large negative value saturates to -inf
    # anyway. Held in fp32 and subtracted from the row max, it underflows to a
    # clean zero.
    #
    # This keeps the softmax finite for finite scores, which is as far as the
    # guarantee goes: if the projection itself overflows -- its fp32 accumulator
    # is cast back to fp16, so finite inputs can still produce Inf -- then
    # `Inf - Inf` is NaN and nothing here repairs it. The baseline's SDPA has the
    # same exposure, and the harness rejects a non-finite reference too, so
    # clamping would buy nothing and would make this differ from the baseline.
    scores = tl.where(valid, scores, -1e30)
    row_max = tl.max(scores, 1)
    probs = tl.exp(scores - row_max[:, None])
    probs = tl.where(valid, probs, 0.0)
    denom = tl.sum(probs, 1)
    probs = probs / tl.maximum(denom, 1e-30)[:, None]

    v = tl.load(k_base + qkv_group + offs_d[None, :], mask=wide_mask, other=0.0)
    ctx = tl.dot(probs.to(v.dtype), v, out_dtype=tl.float32)

    # Token-major store: row is the same token index the projection produced, so
    # `to_out` consumes this directly.
    ctx_rows = (row0 + offs_t * hw) * stride_ctx_row
    tl.store(ctx_ptr + ctx_rows[:, None] + head_off + offs_d[None, :],
             ctx.to(ctx_ptr.dtype.element_ty), mask=wide_mask)


def _table_stamp(freqs: torch.Tensor) -> tuple:
    """Everything cheap that identifies the *contents* of a `freqs` tensor.

    `_version` catches ordinary in-place mutation and the `copy_` inside
    `load_state_dict`, but it raises on inference tensors, and assigning through
    `.data` swaps the storage without incrementing it. So pair it with the
    storage identity and layout, which `.data =` does change. (Mutating
    *through* `.data` -- `freqs.data.mul_(2)` -- defeats both, because `.data`
    hands out a fresh version counter over the same storage. That is unsafe by
    PyTorch's own contract and is not detectable without reading the values.)
    """
    version = None if torch.is_inference(freqs) else freqs._version
    return (version, freqs.data_ptr(), tuple(freqs.shape), tuple(freqs.stride()))


class OasisTemporalAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: OasisRotaryEmbedding,
        *,
        is_causal: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.to_qkv = Linear(dim, dim_head * heads * 3, bias=False)
        self.to_out = Linear(dim_head * heads, dim, bias=True)
        self.rotary_emb = rotary_emb
        self.is_causal = is_causal
        self.attn = DenseAttention(backend="sdpa")
        # Plain attribute, not a buffer: registering the table would add
        # `unexpected_keys` to the weight-sharing `load_state_dict`.
        self._rotary_tables: dict = {}

    # -- rotary table ------------------------------------------------------
    def _rotary_table(self, seq_len: int, dtype: torch.dtype, device: torch.device):
        """cos/sin for `seq_len` positions, `[seq_len, dim_head // 2]`.

        Built through the same chain as the baseline -- an `arange` in the
        query's dtype, the outer product in `freqs.dtype`, then `cos`/`sin` -- so
        the values match bit for bit. That matters because the harness casts
        every high-precision parameter to the case dtype, `freqs` included, so
        the baseline's whole rotary path runs in fp16; deriving the angle in
        fp32 would be *more* accurate than the reference and drift from it.
        """
        freqs = self.rotary_emb.freqs
        inference = torch.is_inference_mode_enabled()
        # `is_causal` is in the key only so that a table can never outlive a
        # change of mode; the values themselves do not depend on it.
        key = (seq_len, dtype, device.type, device.index, self.is_causal, inference)
        cached = self._rotary_tables.get(key)
        if cached is not None:
            ref, ref_dtype, ref_stamp, cos, sin = cached
            # Identity by `is` against a strongly held reference, so a freed
            # `freqs` whose address the allocator reused cannot produce a false
            # hit -- the storage cannot be freed while this entry holds it.
            if (ref is freqs and ref_dtype == freqs.dtype
                    and ref_stamp == _table_stamp(freqs)):
                return cos, sin

        n_pairs = self.dim_head // 2
        pos = torch.arange(seq_len, device=device, dtype=dtype)
        angle = torch.einsum("..., f -> ... f", pos.to(freqs.dtype), freqs)
        cos, sin = angle.cos(), angle.sin()
        n_rotated = angle.shape[-1]
        if n_rotated < n_pairs:
            # Pairs past the rotary width rotate by angle zero, which is the
            # identity -- the same thing the baseline does by passing its
            # `t_right` slice through untouched.
            pad = (0, n_pairs - n_rotated)
            cos = torch.nn.functional.pad(cos, pad, value=1.0)
            sin = torch.nn.functional.pad(sin, pad, value=0.0)
        cos, sin = cos.contiguous(), sin.contiguous()

        # An inference `freqs` has no readable version counter, and in-place
        # mutation inside inference mode is both legal and untracked -- so a
        # cached entry could never be invalidated. Rebuild every call instead.
        # This costs nothing on the benched path, where `freqs` is an ordinary
        # parameter created outside inference mode.
        if not torch.is_inference(freqs):
            if len(self._rotary_tables) > 64:  # a handful of `time` values occur
                self._rotary_tables.clear()
            self._rotary_tables[key] = (
                freqs, freqs.dtype, _table_stamp(freqs), cos, sin)
        return cos, sin

    # -- dispatch ----------------------------------------------------------
    def _fused_path_applies(self, x: torch.Tensor) -> bool:
        """Checked before any reshape, so an unsupported input never raises."""
        if not x.is_cuda or x.dim() != 5:
            return False
        if x.dtype not in (torch.float16, torch.bfloat16):
            return False
        # The kernel has no backward; anything recording grad takes the
        # reference path, which is differentiable.
        if torch.is_grad_enabled():
            return False
        seq_len = x.shape[1]
        if not 1 <= seq_len <= _MAX_FUSED_SEQ:
            return False
        head_dim, heads = self.dim_head, self.heads
        if heads < 1 or head_dim < 2 or head_dim % 2:
            return False
        qkv_w, out_w = self.to_qkv.weight, self.to_out.weight
        if self.to_qkv.bias is not None:
            return False
        if qkv_w.dtype != x.dtype or out_w.dtype != x.dtype:
            return False
        if qkv_w.device != x.device or out_w.device != x.device:
            return False
        if x.shape[-1] != qkv_w.shape[1]:
            return False
        if qkv_w.shape[0] != 3 * heads * head_dim:
            return False
        if out_w.shape[1] != heads * head_dim:
            return False
        bias = self.to_out.bias
        if bias is not None and (bias.dtype != x.dtype or bias.device != x.device):
            return False
        freqs = self.rotary_emb.freqs
        if freqs.dim() != 1 or freqs.numel() < 1 or freqs.device != x.device:
            return False
        if 2 * freqs.shape[-1] > head_dim:  # the baseline itself cannot broadcast this
            return False
        return True

    def _reference_forward(self, x: torch.Tensor) -> torch.Tensor:
        """The baseline's own operations, for every input the fused path declines."""
        bsz, time, height, width, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.reshape(bsz, time, height, width, self.heads, -1).permute(0, 2, 3, 4, 1, 5)
        k = k.reshape(bsz, time, height, width, self.heads, -1).permute(0, 2, 3, 4, 1, 5)
        v = v.reshape(bsz, time, height, width, self.heads, -1).permute(0, 2, 3, 4, 1, 5)

        q = q.reshape(bsz * height * width, self.heads, time, -1)
        k = k.reshape(bsz * height * width, self.heads, time, -1)
        v = v.reshape(bsz * height * width, self.heads, time, -1)

        q = self.rotary_emb.rotate_queries_or_keys(q, self.rotary_emb.freqs)
        k = self.rotary_emb.rotate_queries_or_keys(k, self.rotary_emb.freqs)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = self.attn(q, k, v, causal=self.is_causal)
        out = out.reshape(bsz, height, width, time, self.heads, -1)
        out = out.permute(0, 3, 1, 2, 4, 5).reshape(bsz, time, height, width, -1)
        return self.to_out(out.to(q.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._fused_path_applies(x):
            return self._reference_forward(x)

        bsz, time, height, width, _ = x.shape
        heads, head_dim = self.heads, self.dim_head
        hw = height * width
        n_tokens = bsz * time * hw
        qkv_group = heads * head_dim

        # `reshape`, not `view`: a non-contiguous input (a slice along time, say)
        # needs the copy.
        qkv = torch.mm(x.reshape(n_tokens, x.shape[-1]), self.to_qkv.weight.t())
        cos, sin = self._rotary_table(time, x.dtype, x.device)
        ctx = torch.empty((n_tokens, qkv_group), dtype=x.dtype, device=x.device)

        # `tl.dot` needs M, N, K >= 16, so short sequences and narrow heads are
        # padded; the padded lanes load as zero and are masked out of the store.
        bt = triton.next_power_of_2(max(time, 16))
        # `BJ = BD // 2` so the pair axis falls straight out of `tl.split`, and
        # BD >= 32 keeps that axis at the 16 `tl.dot` needs for its K dimension.
        bd = max(32, triton.next_power_of_2(head_dim))
        bj = bd // 2
        _rope_attn_relayout_kernel[(bsz * hw, heads)](
            qkv, cos, sin, ctx,
            qkv.stride(0), ctx.stride(0), cos.stride(0),
            time, hw, head_dim, head_dim // 2, qkv_group, head_dim ** -0.5,
            BT=bt, BJ=bj, BD=bd, IS_CAUSAL=self.is_causal,
            num_warps=_NUM_WARPS_BY_TILE[bt],
        )

        bias = self.to_out.bias
        weight_t = self.to_out.weight.t()
        out = (torch.mm(ctx, weight_t) if bias is None
               else torch.addmm(bias, ctx, weight_t))
        return out.view(bsz, time, height, width, -1)
