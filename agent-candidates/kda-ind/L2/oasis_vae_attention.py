"""Oasis VAE self-attention with a fused rotary + QKV-split kernel.

The eager baseline spends 23 of its 26 kernels -- and 75% of its GPU time --
materializing the rotary embedding and the layout shuffles around it: two
``oasis_apply_rotary_emb`` calls, each of which builds fp32 ``cos``/``sin``, an
fp32 product, an fp16 ``neg``, a ``stack``, a second fp32 product, an fp32 add,
and a ``cat`` that promotes the untouched pass-through half to fp32 before a
closing downcast. None of that is arithmetic; it is memory traffic in the wrong
dtype, plus a per-op CPU launch cost the GPU cannot hide.

One Triton kernel reads the contiguous ``qkv`` GEMM output and writes a
``(3, B, S, H, D)`` buffer whose planes are ``rotary(q)``, ``rotary(k)``, and
``v`` verbatim, which reduces the forward to four kernels: the ``qkv`` GEMM, the
rearrangement, attention, and the ``proj`` GEMM. The rewrite is an algebraic
restatement rather than an approximation -- same expression, same fp32 promotion,
same single closing rounding -- so it agrees with the eager path to within one
half-precision ulp, differing only where the eager path's fp32 result lands on a
rounding tie (measured: 0.005 % of elements, 0.30 of one fp16 quantum). The eager
path stays available behind a gate for anything the kernel does not cover.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.dense_attention import DenseAttention
from ..L1.linear import Linear
from ..L1.oasis_rotary import OasisRotaryEmbedding, oasis_apply_rotary_emb

try:
    import triton
    import triton.language as tl
except ImportError:  # without Triton the eager path below is the only path
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _rope_qkv_split_kernel(
        qkv_ptr,          # f16/bf16 [B, S, 3 * H * D], contiguous
        out_ptr,          # f16/bf16 [3, B, S, H, D], contiguous
        cos_ptr,          # fp32 [S, ROT]
        sin_ptr,          # fp32 [S, ROT]
        plane_stride,     # B * S * H * D, the gap between the q/k/v planes
        S: tl.constexpr,
        HD: tl.constexpr,
        D: tl.constexpr,
        ROT: tl.constexpr,
    ):
        # One program per (b, s) row. A 2-D grid gives ``s`` directly, so the
        # row -> (b, s) division and modulo never happen at runtime. ``s`` is the
        # fast-varying axis so consecutive programs read consecutive qkv rows,
        # and it is axis 0 because only that axis is not capped at 65535.
        s = tl.program_id(0)
        row = tl.program_id(1) * S + s

        # ``lane`` walks one [H, D] plane as a single contiguous 1024-element
        # run, exactly as it is laid out in the qkv row it comes from -- so the
        # stores are as contiguous as the loads.
        lane = tl.arange(0, HD)
        d = lane % D
        rotated = d < ROT

        # ``cos``/``sin`` are [S, ROT] fp32, reused by every program in the
        # batch, so they live in L2 rather than being re-derived.
        cos = tl.load(cos_ptr + s * ROT + d, mask=rotated, other=0.0)
        sin = tl.load(sin_ptr + s * ROT + d, mask=rotated, other=0.0)
        negate = (d % 2) == 0

        src = qkv_ptr + row * (3 * HD)
        dst = out_ptr + row * HD + lane
        store_dtype = out_ptr.dtype.element_ty

        for plane in tl.static_range(2):  # q, then k
            x = tl.load(src + plane * HD + lane).to(tl.float32)
            # ``rotate_half`` flattens to rh[d] = (d even ? -1 : +1) * x[d ^ 1].
            # The partner lane sits in the same 16-byte run this thread already
            # holds, so this second load is expected to be served from L1 or
            # registers rather than DRAM -- unmeasured, since no cache-traffic
            # counters were collected, so it is a rationale for the layout and
            # not an established fact.
            partner = tl.load(src + plane * HD + (lane ^ 1)).to(tl.float32)
            rh = tl.where(negate, -partner, partner)
            # Products and sum in fp32 with one closing rounding, matching the
            # baseline's fp32 promotion against the fp32 freqs buffer. Whether
            # the two products contract into an FMA is not controllable here
            # (enable_fp_fusion=False measurably changes nothing), so the fp32
            # sum can sit one fp32 ulp off the baseline's; where that lands on a
            # half-precision rounding tie the stored value differs by one ulp.
            y = tl.where(rotated, x * cos + rh * sin, x)
            tl.store(dst + plane * plane_stride, y.to(store_dtype))

        # v is passed through untouched, in its own dtype.
        tl.store(dst + 2 * plane_stride, tl.load(src + 2 * HD + lane))


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _cacheable_version(t: torch.Tensor) -> int | None:
    """``t``'s version counter, or None when it does not have one.

    Inference tensors -- which is what a module constructed inside
    ``torch.inference_mode()`` holds -- do not track a version counter and raise
    on ``_version`` rather than returning anything. Returning None lets the
    caller derive fresh values instead of making a correct forward depend on a
    counter that does not exist.
    """
    if t.is_inference():
        return None
    try:
        return t._version
    except RuntimeError:
        return None


_FUSED_DTYPES = (torch.float16, torch.bfloat16)

# The launch grid is (seq_len, batch). CUDA caps grid axes 1 and 2 at 65535 and
# axis 0 at 2**31-1, so the batch is the dimension that needs a bound.
_MAX_GRID_DIM_YZ = 65535
_MAX_GRID_DIM_X = 2 ** 31 - 1


class OasisVAEAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        frame_height: int,
        frame_width: int,
        *,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = Linear(dim, dim, bias=True)
        self.rotary = OasisRotaryEmbedding(
            dim=(dim // num_heads) // 4,
            freqs_for="pixel",
            max_freq=frame_height * frame_width,
        )
        self.register_buffer(
            "rotary_freqs",
            self.rotary.get_axial_freqs(frame_height, frame_width),
            persistent=False,
        )
        self.attn = DenseAttention(backend="sdpa")

        self.dim = dim
        self.head_dim = dim // num_heads
        self.seq_len = frame_height * frame_width
        self.rot_dim = self.rotary_freqs.shape[-1]
        self.num_warps = 4

        # ``tl.arange`` bounds must be powers of two, so merely-even num_heads or
        # head_dim (num_heads=12, head_dim=80) would reach a compile error rather
        # than this gate. Nothing here reads the weights: they only arrive after
        # __init__, whereas rotary_freqs is deterministic.
        self._fused_ok = (
            triton is not None
            and num_heads > 0
            and dim == num_heads * self.head_dim
            and _is_pow2(num_heads)
            and _is_pow2(self.head_dim)
            and self.head_dim >= 2
            and self.rot_dim >= 2
            and self.rot_dim % 2 == 0
            and self.rot_dim <= self.head_dim
            and tuple(self.rotary_freqs.shape)
            == (frame_height, frame_width, self.rot_dim)
            and 0 < self.seq_len <= _MAX_GRID_DIM_X
        )
        self._freqs_shape = self.rotary_freqs.shape
        # The kernel forms its offsets in int32; the largest one it touches is
        # 3 * x.numel() - 1, so anything above this many input elements is left
        # to the eager path rather than silently addressing the wrong storage.
        self._max_numel = (2 ** 31 - 1) // 3
        # Per-instance, so two modules never share a derived table; keyed on the
        # source buffer's object identity rather than its data_ptr, which the
        # caching allocator can recycle.
        self._cos_sin_cache = None

    def _rotary_cos_sin(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``cos``/``sin`` of rotary_freqs, derived on the GPU and cached.

        Deriving them on the device rather than on the CPU in __init__ keeps them
        bit-identical to the baseline's own ``freqs.cos()``/``freqs.sin()``; a CPU
        derivation can differ by up to one fp32 ulp. The two extra kernels are
        paid once on the first forward, or on every forward for a buffer the
        cache cannot be keyed on -- which is still no worse than the baseline,
        since it re-derives them every call.
        """
        freqs = self.rotary_freqs
        version = _cacheable_version(freqs)
        cache = self._cos_sin_cache
        if (
            version is not None
            and cache is not None
            and cache[0] is freqs
            and cache[1] == freqs.device
            and cache[2] == freqs.dtype
            and cache[3] == version
        ):
            return cache[4], cache[5]
        flat = freqs.reshape(self.seq_len, self.rot_dim)
        cos = flat.cos().contiguous()
        sin = flat.sin().contiguous()
        # The version counter catches an in-place rewrite of the same buffer
        # object -- freqs.copy_(...), as a DDP buffer broadcast does -- which
        # identity, device, and dtype alone cannot see. Without one there is
        # nothing safe to key on, so the values are derived fresh each call,
        # which is what the baseline does anyway. Values derived under
        # inference mode are likewise not kept, since reusing them outside it
        # is what raises.
        if version is not None and not cos.is_inference():
            self._cos_sin_cache = (
                freqs, freqs.device, freqs.dtype, version, cos, sin
            )
        return cos, sin

    def _fused_input_ok(self, x: torch.Tensor) -> bool:
        # fp32 freqs are what makes the fused rotation promote exactly like the
        # baseline's -- an fp16 freqs buffer (as module.half() leaves behind)
        # must degrade identically to the baseline instead, which only the eager
        # path does. The shape is re-checked here and not just in __init__
        # because a buffer replaced with a different but same-sized shape would
        # broadcast differently in the baseline than it flattens here.
        return (
            x.is_cuda
            and x.dim() == 3
            and x.shape[1] == self.seq_len
            and x.shape[2] == self.dim
            and x.numel() > 0
            and x.numel() <= self._max_numel
            # the batch rides grid axis 1, which CUDA caps at 65535
            and x.shape[0] <= _MAX_GRID_DIM_YZ
            and x.is_contiguous()
            and x.dtype in _FUSED_DTYPES
            and self.qkv.weight.dtype == x.dtype
            and self.rotary_freqs.dtype == torch.float32
            and self.rotary_freqs.shape == self._freqs_shape
            # the kernel is handed raw pointers, which carry no device, so a
            # cross-device buffer would read whatever sits at that address
            and self.rotary_freqs.device == x.device
            and not (torch.is_grad_enabled() and self._needs_grad(x))
        )

    def _needs_grad(self, x: torch.Tensor) -> bool:
        """Whether anything flowing through the kernel would need a gradient.

        The kernel has no backward, so it would silently cut every one of these
        out of the graph -- which the harness never sees, because it runs under
        ``torch.no_grad()``, but training would. Every differentiable input the
        kernel consumes has to be listed: the activation, the ``qkv`` parameters
        that produce its input, and the ``rotary_freqs`` buffer that ``cos``/
        ``sin`` are derived from. ``proj`` is applied after the kernel, so its
        parameters are not at risk.
        """
        bias = self.qkv.bias
        return (
            x.requires_grad
            or self.qkv.weight.requires_grad
            or (bias is not None and bias.requires_grad)
            or self.rotary_freqs.requires_grad
        )

    def _eager_forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(bsz, self.frame_height, self.frame_width, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        k = k.reshape(bsz, self.frame_height, self.frame_width, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        v = v.reshape(bsz, self.frame_height, self.frame_width, self.num_heads, -1).permute(0, 3, 1, 2, 4)

        q = oasis_apply_rotary_emb(self.rotary_freqs, q)
        k = oasis_apply_rotary_emb(self.rotary_freqs, k)

        seq_len = self.frame_height * self.frame_width
        q = q.reshape(bsz, self.num_heads, seq_len, -1).transpose(1, 2)
        k = k.reshape(bsz, self.num_heads, seq_len, -1).transpose(1, 2)
        v = v.reshape(bsz, self.num_heads, seq_len, -1).transpose(1, 2)
        out = self.attn(q, k, v)
        out = out.reshape(bsz, seq_len, -1)
        return self.proj(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (self._fused_ok and self._fused_input_ok(x)):
            return self._eager_forward(x)

        bsz = x.shape[0]
        heads, head_dim, seq_len = self.num_heads, self.head_dim, self.seq_len
        qkv = self.qkv(x)
        cos, sin = self._rotary_cos_sin()

        # (3, B, S, H, D): one allocation, and stores as contiguous as the loads.
        # (B, H, S, D) would scatter each row into H separate 128-byte runs.
        buf = torch.empty(
            (3, bsz, seq_len, heads, head_dim), dtype=qkv.dtype, device=qkv.device
        )
        _rope_qkv_split_kernel[(seq_len, bsz)](
            qkv,
            buf,
            cos,
            sin,
            bsz * seq_len * heads * head_dim,
            S=seq_len,
            HD=heads * head_dim,
            D=head_dim,
            ROT=self.rot_dim,
            num_warps=self.num_warps,
        )

        # DenseAttention(backend="sdpa") permutes these to (B, H, S, D); cuDNN on
        # sm100 takes the non-contiguous views directly and returns an output that
        # is physically (B, S, H, D), so the flatten below aliases its storage and
        # the baseline's output-transpose copy disappears. If a future PyTorch
        # returns (B, H, S, D)-contiguous, reshape falls back to a copy: one more
        # kernel, still correct.
        out = self.attn(buf[0], buf[1], buf[2])
        return self.proj(out.reshape(bsz, seq_len, self.dim))
