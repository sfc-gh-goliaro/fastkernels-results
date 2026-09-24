// Hand-written kernels for the Kimi-MLA dense-prefill path (sm_100a / B200).
//
//   rmsnorm_latent  RMSNorm over the latent columns of the fused q/kv_a GEMM
//                   output, copying the rope columns through into the same
//                   buffer.  Replaces the reference's contiguous() copy plus
//                   separate norm launch.
//   mla_attention   causal FlashAttention (mma.sync m16n8k16 over cp.async
//                   tiles) that reads K_nope and V straight out of the
//                   kv_b_proj output and K_pe out of the kv_a buffer, so
//                   [k_nope | k_pe] is never materialised.  Faster than FA4 up
//                   to ~64 tokens, where FA4's fixed cost outweighs its
//                   tcgen05 inner loop.
//
// Shared-memory addressing is 32-bit and hoisted out of the inner loops --
// 64-bit address math there costs more issue slots than the mma it feeds.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define DEVI __device__ __forceinline__
using bf16 = __nv_bfloat16;

namespace {

DEVI uint32_t sm_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

DEVI void mma16816(float (&d)[4], const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ldmatrix.x4 fills the four m16n8k16 A-operand registers when the 32 lane
// addresses are (row = lane % 16, col = 8 * (lane / 16)); the same instruction
// fills two B-operand fragments when they are (row = n, col = k).
DEVI void ldm4(uint32_t* d, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a));
}

// Transposing variant, used to read V (stored token-major) as a B operand.
DEVI void ldm4t(uint32_t* d, uint32_t a) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a));
}

DEVI void cpa16(uint32_t dst, const void* src, bool pred) {
  int n = pred ? 16 : 0;  // src-size 0 zero-fills instead of reading
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst),
               "l"(src), "r"(n));
}
DEVI void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
DEVI void cp_wait_all() { asm volatile("cp.async.wait_group 0;\n"); }

DEVI uint32_t pk(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

// ===========================================================================
// RMSNorm over the first L columns of a row; the trailing `tail` columns are
// copied verbatim.  One warp per row, so the reduction is pure shuffles and a
// 512-wide latent is two 16 B vectors per lane.  L and tail are multiples of 8.
// ===========================================================================
constexpr int NORM_ROWS = 8;  // rows (warps) per CTA

__global__ __launch_bounds__(32 * NORM_ROWS) void rmsnorm_copy_kernel(
    const bf16* __restrict__ in, int ld, bf16* __restrict__ out, int ldo,
    const bf16* __restrict__ gamma, float eps, int L, int tail, int M) {
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * NORM_ROWS + (threadIdx.x >> 5);
  if (row >= M) return;
  const bf16* src = in + (size_t)row * ld;
  bf16* dst = out + (size_t)row * ldo;
  const int nvec = L >> 3;

  float part = 0.f;
  for (int v = lane; v < nvec; v += 32) {
    float4 x = *reinterpret_cast<const float4*>(src + v * 8);
    const bf16* h = reinterpret_cast<const bf16*>(&x);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float f = __bfloat162float(h[i]);
      part += f * f;
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) part += __shfl_xor_sync(0xffffffff, part, o);
  const float rs = rsqrtf(part / (float)L + eps);

  for (int v = lane; v < nvec; v += 32) {
    float4 x = *reinterpret_cast<const float4*>(src + v * 8);
    float4 g = *reinterpret_cast<const float4*>(gamma + v * 8);
    const bf16* h = reinterpret_cast<const bf16*>(&x);
    const bf16* gh = reinterpret_cast<const bf16*>(&g);
    bf16 o[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
      o[i] = __float2bfloat16(__bfloat162float(h[i]) * rs * __bfloat162float(gh[i]));
    *reinterpret_cast<float4*>(dst + v * 8) = *reinterpret_cast<float4*>(o);
  }
  for (int v = lane * 8; v < tail; v += 32 * 8)
    *reinterpret_cast<float4*>(dst + L + v) =
        *reinterpret_cast<const float4*>(src + L + v);
}

// ===========================================================================
// Causal MLA attention: AM queries x AN keys per CTA, 4 warps of 16 rows.
// Single-buffered K/V keeps shared memory at 67 KB so three CTAs fit per SM.
// ===========================================================================
constexpr int DQK = 192, DV = 128, AM = 64, AN = 64;
constexpr int QLD = DQK + 8;  // 200 halves = 400 B rows, conflict-free ldmatrix
constexpr int VLD = DV + 8;   // 136 halves = 272 B

__global__ __launch_bounds__(128) void mla_attn_kernel(
    const bf16* __restrict__ Qb, int ldq, const bf16* __restrict__ KVb, int ldkv,
    const bf16* __restrict__ PEb, int ldpe, bf16* __restrict__ Ob, int ldo,
    int M, float scale) {
  extern __shared__ bf16 smem[];
  const uint32_t sQ = sm_addr(smem);
  const uint32_t sK = sQ + AM * QLD * 2;
  const uint32_t sV = sK + AN * QLD * 2;

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  // Causal attention gives m-block j exactly j+1 key blocks of work, so run the
  // heaviest blocks first: CTAs are issued in x-major order, and
  // longest-processing-time-first packs the SMs better than the natural order.
  const int m0 = (gridDim.x - 1 - blockIdx.x) * AM, head = blockIdx.y;
  const int lr = tid >> 1, lh = tid & 1;

  const bf16* kvrow = KVb + head * (DV + DV);
  const int nblk = min((m0 + AM + AN - 1) / AN, (M + AN - 1) / AN);

  {  // Q tile: 24 chunks of 16 B per row, 12 per thread
    const bf16* p = Qb + (size_t)(m0 + lr) * ldq + head * DQK + lh * 96;
    const uint32_t d = sQ + (uint32_t)(lr * QLD + lh * 96) * 2;
    const bool ok = (m0 + lr) < M;
#pragma unroll
    for (int i = 0; i < 12; ++i) cpa16(d + i * 16, p + i * 8, ok);
  }

  auto load_kv = [&](int j) {
    const bool ok = (j * AN + lr) < M;
    const bf16* pn = kvrow + (size_t)(j * AN + lr) * ldkv + lh * 64;
    const uint32_t dn = sK + (uint32_t)(lr * QLD + lh * 64) * 2;
#pragma unroll
    for (int i = 0; i < 8; ++i) cpa16(dn + i * 16, pn + i * 8, ok);
    const bf16* pp = PEb + (size_t)(j * AN + lr) * ldpe + lh * 32;
    const uint32_t dp = sK + (uint32_t)(lr * QLD + DV + lh * 32) * 2;
#pragma unroll
    for (int i = 0; i < 4; ++i) cpa16(dp + i * 16, pp + i * 8, ok);
    const bf16* pv = kvrow + (size_t)(j * AN + lr) * ldkv + DV + lh * 64;
    const uint32_t dv = sV + (uint32_t)(lr * VLD + lh * 64) * 2;
#pragma unroll
    for (int i = 0; i < 8; ++i) cpa16(dv + i * 16, pv + i * 8, ok);
  };
  load_kv(0);
  cp_commit();

  float o[16][4];
#pragma unroll
  for (int t = 0; t < 16; ++t)
#pragma unroll
    for (int e = 0; e < 4; ++e) o[t][e] = 0.f;
  float mi[2] = {-1e30f, -1e30f}, li[2] = {0.f, 0.f};

  const float ls = scale * 1.4426950408889634f;  // fold log2(e) into the scale
  const uint32_t rq = sQ + (uint32_t)((warp * 16 + (lane & 15)) * QLD) * 2 +
                      (uint32_t)((lane >> 4) * 8) * 2;
  const uint32_t rk = sK + (uint32_t)((8 * (lane >> 4) + (lane & 7)) * QLD) * 2 +
                      (uint32_t)(8 * ((lane >> 3) & 1)) * 2;
  const uint32_t rv = sV + (uint32_t)((8 * ((lane >> 3) & 1) + (lane & 7)) * VLD) * 2 +
                      (uint32_t)(8 * (lane >> 4)) * 2;
  const int r_lo = m0 + warp * 16 + (lane >> 2);
  const int c_off = (lane & 3) * 2;

  for (int j = 0; j < nblk; ++j) {
    cp_wait_all();
    __syncthreads();

    // S = Q K^T : 16 rows x 64 keys per warp, K of width 192
    float s[8][4];
#pragma unroll
    for (int t = 0; t < 8; ++t)
#pragma unroll
      for (int e = 0; e < 4; ++e) s[t][e] = 0.f;
#pragma unroll
    for (int ks = 0; ks < DQK / 16; ++ks) {
      uint32_t af[4], bfr[4][4];
      ldm4(af, rq + ks * 32);
#pragma unroll
      for (int p = 0; p < 4; ++p) ldm4(bfr[p], rk + ks * 32 + p * 16 * QLD * 2);
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        mma16816(s[p * 2 + 0], af, &bfr[p][0]);
        mma16816(s[p * 2 + 1], af, &bfr[p][2]);
      }
    }

    const int k0 = j * AN;
    const int c_lo = k0 + c_off;
    const bool need_mask = (k0 + AN - 1) >= r_lo || (k0 + AN) > M;
    float rmax[2] = {-1e30f, -1e30f};
#pragma unroll
    for (int t = 0; t < 8; ++t)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        float v = s[t][e] * ls;
        if (need_mask) {
          int r = r_lo + (e >> 1) * 8;
          int c = c_lo + t * 8 + (e & 1);
          if (c > r || c >= M) v = -1e30f;
        }
        s[t][e] = v;
        rmax[e >> 1] = fmaxf(rmax[e >> 1], v);
      }
    // Each row of the accumulator lives in 4 lanes of the same quad.
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      rmax[h] = fmaxf(rmax[h], __shfl_xor_sync(0xffffffff, rmax[h], 1));
      rmax[h] = fmaxf(rmax[h], __shfl_xor_sync(0xffffffff, rmax[h], 2));
    }
    float corr[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      float mn = fmaxf(mi[h], rmax[h]);
      corr[h] = exp2f(mi[h] - mn);
      mi[h] = mn;
    }
    float rsum[2] = {0.f, 0.f};
#pragma unroll
    for (int t = 0; t < 8; ++t)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        float p = exp2f(s[t][e] - mi[e >> 1]);
        s[t][e] = p;
        rsum[e >> 1] += p;
      }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      rsum[h] += __shfl_xor_sync(0xffffffff, rsum[h], 1);
      rsum[h] += __shfl_xor_sync(0xffffffff, rsum[h], 2);
      li[h] = li[h] * corr[h] + rsum[h];
    }
#pragma unroll
    for (int t = 0; t < 16; ++t)
#pragma unroll
      for (int e = 0; e < 4; ++e) o[t][e] *= corr[e >> 1];

    // O += P V.  Two neighbouring S column tiles already sit in exactly the
    // register layout an m16n8k16 A operand wants, so P needs no shuffling.
#pragma unroll
    for (int kb = 0; kb < 4; ++kb) {
      uint32_t pf[4] = {pk(s[kb * 2][0], s[kb * 2][1]),
                        pk(s[kb * 2][2], s[kb * 2][3]),
                        pk(s[kb * 2 + 1][0], s[kb * 2 + 1][1]),
                        pk(s[kb * 2 + 1][2], s[kb * 2 + 1][3])};
      const uint32_t vb = rv + kb * 16 * VLD * 2;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        uint32_t d[4];
        ldm4t(d, vb + nt * 32);
        mma16816(o[nt * 2 + 0], pf, &d[0]);
        mma16816(o[nt * 2 + 1], pf, &d[2]);
      }
    }

    if (j + 1 < nblk) {
      __syncthreads();
      load_kv(j + 1);
      cp_commit();
    }
  }

  const float i0 = 1.f / (li[0] == 0.f ? 1.f : li[0]);
  const float i1 = 1.f / (li[1] == 0.f ? 1.f : li[1]);
  bf16* dst = Ob + head * DV + c_off;
  if (r_lo < M) {
    bf16* d0 = dst + (size_t)r_lo * ldo;
#pragma unroll
    for (int t = 0; t < 16; ++t)
      *reinterpret_cast<uint32_t*>(d0 + t * 8) = pk(o[t][0] * i0, o[t][1] * i0);
  }
  if (r_lo + 8 < M) {
    bf16* d1 = dst + (size_t)(r_lo + 8) * ldo;
#pragma unroll
    for (int t = 0; t < 16; ++t)
      *reinterpret_cast<uint32_t*>(d1 + t * 8) = pk(o[t][2] * i1, o[t][3] * i1);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Host entry points
// ---------------------------------------------------------------------------

// RMSNorm the ``latent`` columns of ``y`` starting at ``off`` and copy the
// ``rope`` columns that follow into one fresh [M, latent + rope] buffer.
// Returns (buffer, latent view, rope view) so the caller needs no slicing.
std::vector<torch::Tensor> rmsnorm_latent(torch::Tensor y, torch::Tensor gamma,
                                          double eps, int64_t off,
                                          int64_t latent, int64_t rope) {
  const int M = (int)y.size(0);
  auto out = torch::empty({M, latent + rope}, y.options());
  const int blocks = (M + NORM_ROWS - 1) / NORM_ROWS;
  rmsnorm_copy_kernel<<<blocks, 32 * NORM_ROWS, 0,
                        c10::cuda::getCurrentCUDAStream()>>>(
      (const bf16*)y.data_ptr() + off, (int)y.stride(0), (bf16*)out.data_ptr(),
      (int)(latent + rope), (const bf16*)gamma.data_ptr(), (float)eps,
      (int)latent, (int)rope, M);
  return {out, out.narrow(1, 0, latent), out.narrow(1, latent, rope)};
}

// Causal attention read straight off the projection buffers: q from ``y``
// (head h at h*192 of each row), K_nope and V from ``kv`` (head h at h*256),
// K_pe from ``pe``.
torch::Tensor mla_attention(torch::Tensor y, torch::Tensor kv,
                            torch::Tensor pe, double scale, int64_t H) {
  const int M = (int)y.size(0);
  auto o = torch::empty({M, H * DV}, y.options());
  constexpr size_t smem = ((size_t)AM * QLD + AN * QLD + AN * VLD) * sizeof(bf16);
  static bool init = false;
  if (!init) {
    cudaFuncSetAttribute(mla_attn_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem);
    init = true;
  }
  dim3 grid((M + AM - 1) / AM, (int)H);
  mla_attn_kernel<<<grid, 128, smem, c10::cuda::getCurrentCUDAStream()>>>(
      (const bf16*)y.data_ptr(), (int)y.stride(0), (const bf16*)kv.data_ptr(),
      (int)kv.stride(0), (const bf16*)pe.data_ptr(), (int)pe.stride(0),
      (bf16*)o.data_ptr(), (int)H * DV, M, (float)scale);
  return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm_latent", &rmsnorm_latent, "RMSNorm latent cols, copy rope");
  m.def("mla_attention", &mla_attention, "causal MLA attention");
}
