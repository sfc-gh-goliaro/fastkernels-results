import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _rotate_middle_kernel(
    x_ptr,              # *f16/*bf16
    y_ptr,              # *f16/*bf16
    cos_ptr,            # *f32
    sin_ptr,            # *f32
    N: tl.constexpr,    # rows
    T: tl.constexpr,    # time
    D: tl.constexpr,    # dim
    rot_pairs: tl.constexpr,  # number of (cos,sin) pairs = len(freqs)
    skip: tl.constexpr,       # start column for rotated region
    stride_n: tl.constexpr,
    stride_t: tl.constexpr,
    stride_d: tl.constexpr,
):
    pid_n = tl.program_id(0)  # row id
    pid_t = tl.program_id(1)  # time id
    if pid_n >= N or pid_t >= T:
        return

    row_x = x_ptr + pid_n * stride_n + pid_t * stride_t
    row_y = y_ptr + pid_n * stride_n + pid_t * stride_t

    # Process even indices within the rotated region: jj = 2*i in [0, 2*rot_pairs)
    for i in range(0, rot_pairs):
        jj = 2 * i
        col = skip + jj  # absolute feature index

        # Load x[col], x[col+1]
        x_j  = tl.load(row_x + col        * stride_d)
        x_j1 = tl.load(row_x + (col + 1)  * stride_d)

        # Load cos[i], sin[i] as f32
        c = tl.load(cos_ptr + i)
        s = tl.load(sin_ptr + i)

        # Compute in f32
        xj  = x_j.to(tl.float32)
        xj1 = x_j1.to(tl.float32)
        c32 = c.to(tl.float32)
        s32 = s.to(tl.float32)

        # even = x[j] * cos + x[j+1] * sin * (-1)
        # odd  = x[j+1] * cos + x[j] * sin
        even = xj  * c32 + xj1 * s32 * (-1.0)
        odd  = xj1 * c32 + xj  * s32

        # Store to output at positions (col=skip+2*i) and (col+1=skip+2*i+1)
        tl.store(row_y + (skip + 2 * i)       * stride_d, even.to(x_j.dtype))
        tl.store(row_y + (skip + 2 * i + 1)   * stride_d, odd.to(x_j.dtype))

def _rotate_middle_triton(x: torch.Tensor, freqs: torch.Tensor, skip: int = 0) -> torch.Tensor:
    """
    x: [N, T, D] CUDA tensor (fp16/bf16), contiguous
    freqs: 1D tensor length rot_pairs (used to compute cos/sin)
    Returns y with same shape/dtype.
    Rotates only the middle block starting at column 'skip' with width = 2*len(freqs).
    """
    assert x.is_cuda, "Triton kernel requires CUDA tensor"
    assert x.dtype in (torch.float16, torch.bfloat16), "Expected fp16/bf16"
    x = x.contiguous()
    N, T, D = x.shape
    rot_pairs = int(freqs.numel())   # number of cosine/sine pairs
    rot_width = 2 * rot_pairs        # width of rotated region

    # Compute cos/sin on device in f32
    device = x.device
    freqs_f32 = freqs.to(dtype=torch.float32, device=device)
    cos = torch.cos(freqs_f32)
    sin = torch.sin(freqs_f32)

    y = torch.empty_like(x)

    stride_n = x.stride(0)
    stride_t = x.stride(1)
    stride_d = x.stride(2)

    grid = (N, T)
    _rotate_middle_kernel[grid](
        x, y, cos, sin,
        N, T, D,
        rot_pairs=rot_pairs,
        skip=skip,
        stride_n=stride_n,
        stride_t=stride_t,
        stride_d=stride_d,
        num_warps=1,
        num_stages=1,
    )
    return y

class OasisRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        freqs_for: str = "lang",
        theta: float = 10000.0,
        max_freq: float = 10.0,
    ):
        super().__init__()
        self.dim = dim
        self.freqs_for = freqs_for
        if freqs_for == "lang":
            # freqs length = dim/2 (pairs)
            arange = torch.arange(0, dim, 2).float()
            freqs = 1.0 / (theta ** (arange / dim))
        elif freqs_for == "pixel":
            # freqs length = dim/2 (pairs)
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * math.pi
        else:
            raise ValueError(f"unsupported rotary mode: {freqs_for}")
        self.freqs = nn.Parameter(freqs, requires_grad=False)
        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    @property
    def device(self) -> torch.device:
        return self.dummy.device

    def rotate_queries_or_keys(self, t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        # t shape: [N, T, D]
        if not t.is_cuda:
            raise RuntimeError("CPU rotate not supported in Triton version")
        # Rotated region starts at skip=0 and has width = 2*len(freqs)
        return _rotate_middle_triton(t, freqs, skip=0)

class Matmul(nn.Module):
    def forward(self, input, weight, bias=None):
        return torch.nn.functional.linear(input, weight, bias)

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.matmul = Matmul()

    def forward(self, input):
        return self.matmul(input, self.weight, self.bias)

_CUDNN_MAX_HEAD_DIM = 128

class DenseAttention(nn.Module):
    def __init__(self, backend: Literal["auto", "sdpa", "flash_attn", "cudnn", "flex"] = "auto"):
        super().__init__()
        self.fa_func = None
        self.use_cudnn_kernel = False
        self.use_flex_kernel = False
        self._flex_fn = None

        if backend == "sdpa":
            return
        if backend == "cudnn":
            self.use_cudnn_kernel = True
            return
        if backend == "flex":
            from torch.nn.attention.flex_attention import flex_attention
            self.use_flex_kernel = True
            self._flex_fn = torch.compile(flex_attention, dynamic=False)
            return
        if backend == "flash_attn":
            self.fa_func = _resolve_flash_attn_func()
            return

        cc = (torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1])
        if 80 <= cc < 100:
            self.fa_func = _resolve_flash_attn_func()
        elif cc >= 100:
            self.use_cudnn_kernel = True

    def forward(self, query, key, value, softmax_scale=None, causal=False, attn_mask: torch.Tensor | None = None):
        if self.fa_func is not None and attn_mask is None and query.dtype != torch.float32:
            out = self.fa_func(query, key, value, softmax_scale=softmax_scale, causal=causal)
            if isinstance(out, tuple):
                out = out[0]
            return out

        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)

        if self.use_flex_kernel:
            q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
            out = self._flex_fn(q, k, v, block_mask=attn_mask, scale=softmax_scale)
        elif self.use_cudnn_kernel:
            if attn_mask is not None and causal:
                raise ValueError("Pass either attn_mask or causal=True, not both.")
            if attn_mask is not None and not attn_mask.is_contiguous():
                attn_mask = attn_mask.contiguous()
            if q.shape[-1] > _CUDNN_MAX_HEAD_DIM:
                if attn_mask is not None and attn_mask.dtype not in (torch.bool, q.dtype):
                    attn_mask = attn_mask.to(dtype=q.dtype)
                with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION, torch.nn.attention.SDPBackend.MATH]):
                    out = torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=causal, scale=softmax_scale
                    )
            else:
                try:
                    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.CUDNN_ATTENTION]):
                        out = torch.nn.functional.scaled_dot_product_attention(
                            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=causal, scale=softmax_scale
                        )
                except RuntimeError:
                    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION, torch.nn.attention.SDPBackend.MATH]):
                        out = torch.nn.functional.scaled_dot_product_attention(
                            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=causal, scale=softmax_scale
                        )
        else:
            if attn_mask is not None and attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.to(dtype=q.dtype)
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False if attn_mask is not None else causal, scale=softmax_scale
            )
        return out.permute(0, 2, 1, 3)

class ModelNew(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, rotary_emb: OasisRotaryEmbedding, *, is_causal: bool = True):
        super().__init__()
        self.heads = heads
        self.to_qkv = Linear(dim, dim_head * heads * 3, bias=False)
        self.to_out = Linear(dim_head * heads, dim, bias=True)
        self.rotary_emb = rotary_emb
        self.is_causal = is_causal
        self.attn = DenseAttention(backend="sdpa")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H, W, D]
        bsz, time, height, width, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # Layout to [N, T, D]: N = B*H*W*heads, D = dim_head
        q = q.reshape(bsz, time, height, width, self.heads, -1).permute(0, 2, 3, 4, 1, 5).reshape(bsz * height * width * self.heads, time, -1)
        k = k.reshape(bsz, time, height, width, self.heads, -1).permute(0, 2, 3, 4, 1, 5).reshape(bsz * height * width * self.heads, time, -1)
        v = v.reshape(bsz, time, height, width, self.heads, -1).permute(0, 2, 3, 4, 1, 5).reshape(bsz * height * width * self.heads, time, -1)

        # Rotate using Triton
        q = self.rotary_emb.rotate_queries_or_keys(q, self.rotary_emb.freqs)
        k = self.rotary_emb.rotate_queries_or_keys(k, self.rotary_emb.freqs)

        # Transpose to [N, D, T] for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Call SDPA
        out = self.attn(q, k, v, causal=self.is_causal)

        # Restore layout
        out = out.reshape(bsz, height, width, time, self.heads, -1).permute(0, 3, 1, 2, 4, 5).reshape(bsz, time, height, width, -1)
        return self.to_out(out.to(q.dtype))

OasisTemporalAxialAttention = ModelNew
