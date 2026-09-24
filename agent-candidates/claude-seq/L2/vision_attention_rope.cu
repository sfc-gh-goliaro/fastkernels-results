// Fused qkv-split + rotary for the Qwen vision encoder's attention layer.
//
// The baseline makes two full passes over the q|k half of the fused qkv
// activation: a ``.contiguous()`` that re-lays it out as (2, seq, heads, dim)
// and an in-place ``apply_rotary`` over the result.  This kernel does both in
// one pass, and writes q/k at a padded head_dim (zero channels cannot change
// q @ k^T) because FlashAttention's SM100 forward has tuned tiles at multiples
// of 32 and runs a markedly slower path at head_dim 72.
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

template <typename T>
struct __align__(8) Vec4 { T v[4]; };

__device__ __forceinline__ float to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float to_f(__half x) { return __half2float(x); }
__device__ __forceinline__ void from_f(__nv_bfloat16& d, float s) { d = __float2bfloat16(s); }
__device__ __forceinline__ void from_f(__half& d, float s) { d = __float2half(s); }

// One thread owns 4 rotary pairs: it reads the 4 low and 4 high channels of one
// head (two 8B loads) and writes the rotated pair (two 8B stores).  The block
// shape is (half/4, heads, 2), so one block covers the q and k halves of
// ROWS_PER_BLOCK tokens and every index is a thread coordinate -- no integer
// division in the inner loop.  cos/sin are shared by every head of a token, so
// they are staged in shared memory once per token instead of re-read 2*heads
// times.
template <typename T, int ROWS_PER_BLOCK>
__global__ void qk_rope_kernel(
    const Vec4<T>* __restrict__ qkv, const Vec4<T>* __restrict__ cosp,
    const Vec4<T>* __restrict__ sinp, Vec4<T>* __restrict__ out_q,
    Vec4<T>* __restrict__ out_k, long n, int qkv_row4, int qsize4, int hd4,
    int half4, int pad4, int out_row4)
{
    __shared__ Vec4<T> sm[2 * ROWS_PER_BLOCK * 32];   // [2][ROWS][half4], half4<=32
    const int g = threadIdx.x, h = threadIdx.y, qk = threadIdx.z;
    const int lane = h * blockDim.z + qk;
    const int nlane = blockDim.y * blockDim.z;

    for (long row0 = (long)blockIdx.x * ROWS_PER_BLOCK; row0 < n;
         row0 += (long)gridDim.x * ROWS_PER_BLOCK) {
        const int nr = (int)min((long)ROWS_PER_BLOCK, n - row0);
        for (int i = lane; i < nr * 2; i += nlane) {
            const int r = i >> 1;
            const Vec4<T>* p = (i & 1) ? sinp : cosp;
            sm[(i & 1) * ROWS_PER_BLOCK * half4 + r * half4 + g] =
                p[(row0 + r) * (long)half4 + g];
        }
        __syncthreads();
        for (int r = 0; r < nr; ++r) {
            const long row = row0 + r;
            const Vec4<T>* src = qkv + row * (long)qkv_row4 + qk * qsize4 + h * hd4;
            const Vec4<T> lo = src[g];
            const Vec4<T> hi = src[half4 + g];
            const Vec4<T> c = sm[r * half4 + g];
            const Vec4<T> s = sm[ROWS_PER_BLOCK * half4 + r * half4 + g];
            Vec4<T> olo, ohi;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float x0 = to_f(lo.v[i]), x1 = to_f(hi.v[i]);
                const float cf = to_f(c.v[i]), sf = to_f(s.v[i]);
                from_f(olo.v[i], x0 * cf - x1 * sf);
                from_f(ohi.v[i], x0 * sf + x1 * cf);
            }
            Vec4<T>* dst = (qk ? out_k : out_q) + row * (long)out_row4 + h * pad4;
            dst[g] = olo;
            dst[half4 + g] = ohi;
            // The padding channels carry no information, but writing them is
            // cheaper than leaving partially-dirty cache lines behind, which
            // cost a read-for-ownership on eviction: 0.062ms vs 0.077ms per
            // layer at 20680 tokens.  It also means the caller can hand us an
            // uninitialized buffer.
            for (int i = 2 * half4 + g; i < pad4; i += blockDim.x)
                dst[i] = Vec4<T>{};
        }
        __syncthreads();
    }
}

template <typename T>
void launch(const at::Tensor& qkv, const at::Tensor& cos, const at::Tensor& sin,
            at::Tensor& q, at::Tensor& k, long rows_per_block, int nheads,
            int pad_dim, int head_dim)
{
    const long n = qkv.size(0);
    const int half4 = head_dim / 8;
    const dim3 block(half4, nheads, 2);
    const dim3 grid((unsigned)((n + rows_per_block - 1) / rows_per_block));
    const auto st = at::cuda::getCurrentCUDAStream();
    auto P = [](at::Tensor& t) { return reinterpret_cast<Vec4<T>*>(t.data_ptr()); };
    auto CP = [](const at::Tensor& t) {
        return reinterpret_cast<const Vec4<T>*>(t.data_ptr());
    };
#define FK_LAUNCH(R)                                                              \
    qk_rope_kernel<T, R><<<grid, block, 0, st>>>(                                 \
        CP(qkv), CP(cos), CP(sin), P(q), P(k), n, (int)(qkv.stride(0) / 4),        \
        nheads * head_dim / 4, head_dim / 4, half4, pad_dim / 4,                   \
        nheads * pad_dim / 4)
    switch (rows_per_block) {
        case 1: FK_LAUNCH(1); break;
        case 2: FK_LAUNCH(2); break;
        case 4: FK_LAUNCH(4); break;
        case 8: FK_LAUNCH(8); break;
        default: TORCH_CHECK(false, "unsupported rows_per_block");
    }
#undef FK_LAUNCH
}

}  // namespace

// qkv: [n, 3 * nheads * head_dim] (row stride may exceed the width)
// cos/sin: [n, head_dim / 2], contiguous
// q/k: [n, nheads, pad_dim], contiguous, may be uninitialized
void fk_va_qk_rope(at::Tensor qkv, at::Tensor cos, at::Tensor sin,
                   at::Tensor q, at::Tensor k, long rows_per_block)
{
    const int nheads = (int)q.size(1), pad_dim = (int)q.size(2);
    const int head_dim = (int)cos.size(1) * 2;
    TORCH_CHECK(head_dim % 8 == 0 && pad_dim % 4 == 0 && pad_dim >= head_dim,
                "qk_rope: head_dim must be a multiple of 8 and fit pad_dim");
    TORCH_CHECK(head_dim / 8 <= 32, "qk_rope: head_dim too large");
    TORCH_CHECK(qkv.stride(0) % 4 == 0 && qkv.stride(1) == 1 &&
                qkv.size(1) >= 3 * nheads * head_dim,
                "qk_rope: qkv must be row-major with 3*nheads*head_dim channels");
    TORCH_CHECK(cos.is_contiguous() && sin.is_contiguous() &&
                cos.sizes() == sin.sizes() && cos.size(0) >= qkv.size(0),
                "qk_rope: cos/sin must be contiguous [>=n, head_dim/2]");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && q.sizes() == k.sizes() &&
                q.size(0) == qkv.size(0), "qk_rope: bad q/k output layout");
    TORCH_CHECK(qkv.scalar_type() == cos.scalar_type() &&
                qkv.scalar_type() == sin.scalar_type() &&
                qkv.scalar_type() == q.scalar_type() &&
                qkv.scalar_type() == k.scalar_type(),
                "qk_rope: all tensors must share a dtype");
    const at::cuda::OptionalCUDAGuard guard(device_of(qkv));
    if (qkv.scalar_type() == at::kBFloat16) {
        launch<__nv_bfloat16>(qkv, cos, sin, q, k, rows_per_block, nheads, pad_dim, head_dim);
    } else if (qkv.scalar_type() == at::kHalf) {
        launch<__half>(qkv, cos, sin, q, k, rows_per_block, nheads, pad_dim, head_dim);
    } else {
        TORCH_CHECK(false, "qk_rope: only float16/bfloat16 supported");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qk_rope", &fk_va_qk_rope,
          "fused qkv split + rotary at a padded head_dim (CUDA)");
}
