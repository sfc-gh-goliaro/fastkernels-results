// Fused helpers for the Kimi MLA prefill path.
//
// The baseline builds FlashAttention's K/V out of the ``kv_b_proj`` output with
// three ATen copies: a strided ``k[..., :nope] = k_nope`` slice-assign, a
// broadcasting ``k[..., nope:] = k_pe`` slice-assign (64 bf16 = 128 B per
// segment, so every store covers a third of a sector) and FlashAttention's own
// ``maybe_contiguous`` clone.  At 16k tokens those three move ~600 MB and cost
// more than the GEMM that produced the data.
//
// ``build_kv`` replaces all three with one pass of 16-byte vector loads and
// stores: each (token, head) slot is a whole ``uint4`` and the shared RoPE tail
// is read straight out of the fused QKV buffer.  V is never copied at all --
// FlashAttention only requires a dense last dimension, so the value half of the
// ``kv_b_proj`` output is handed over as a strided view.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

namespace mlafast {

using vec_t = uint4;                 // 16 B == 8 bf16 lanes
constexpr int VEC = 8;

// ---------------------------------------------------------------------------
//   k[t, h, 0:DN]     = kvb[t, h, 0:DN]        (NoPE half of kv_b_proj)
//   k[t, h, DN:DN+DR] = kpe[t, 0:DR]           (RoPE tail, shared by all heads)
//
// One thread per 16-byte output slot.  ``grid.y`` carries the token index so
// the only division a thread does is by the compile-time slot count.
// ---------------------------------------------------------------------------
template <int DN, int DR, int DV>
__global__ void build_kv_kernel(const vec_t* __restrict__ kvb,
                                const vec_t* __restrict__ kpe,
                                vec_t* __restrict__ kout,
                                int slots_per_token,
                                long s_kvb_row, long s_kpe_row) {
  constexpr int NOPEV = DN / VEC;
  constexpr int SRCV  = (DN + DV) / VEC;
  constexpr int KVECS = (DN + DR) / VEC;

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= slots_per_token) return;
  const long t = blockIdx.y;
  const int h = idx / KVECS;
  const int j = idx - h * KVECS;
  const int nheads = slots_per_token / KVECS;

  const vec_t val = (j < NOPEV)
      ? kvb[t * s_kvb_row + (long)h * SRCV + j]
      : kpe[t * s_kpe_row + (j - NOPEV)];
  kout[t * (long)nheads * KVECS + (long)h * KVECS + j] = val;
}

at::Tensor build_kv(const at::Tensor& kvb, const at::Tensor& kpe,
                    int64_t num_heads, int64_t d_nope, int64_t d_rope,
                    int64_t d_v) {
  TORCH_CHECK(kvb.is_cuda() && kpe.is_cuda(), "build_kv: cuda tensors");
  TORCH_CHECK(kvb.scalar_type() == at::kBFloat16, "build_kv: bf16 only");
  TORCH_CHECK(kpe.scalar_type() == at::kBFloat16, "build_kv: bf16 only");
  TORCH_CHECK(kvb.dim() == 2 && kpe.dim() == 2, "build_kv: 2-D inputs");
  TORCH_CHECK(kvb.stride(1) == 1 && kpe.stride(1) == 1, "build_kv: dense rows");
  TORCH_CHECK(d_nope == 128 && d_rope == 64 && d_v == 128,
              "build_kv: head dims must be (128, 64, 128)");
  const long n = kvb.size(0);
  TORCH_CHECK(kpe.size(0) == n, "build_kv: token count mismatch");
  TORCH_CHECK(kvb.size(1) == num_heads * (d_nope + d_v), "build_kv: kvb width");
  TORCH_CHECK(kpe.size(1) == d_rope, "build_kv: kpe width");
  TORCH_CHECK(kvb.stride(0) % VEC == 0 && kpe.stride(0) % VEC == 0,
              "build_kv: row strides must be a multiple of 8");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(kvb.data_ptr()) % 16) == 0 &&
              (reinterpret_cast<uintptr_t>(kpe.data_ptr()) % 16) == 0,
              "build_kv: inputs must be 16-byte aligned");

  const c10::cuda::CUDAGuard guard(kvb.device());
  auto opts = kvb.options();
  at::Tensor k = at::empty({n, num_heads * (d_nope + d_rope)}, opts);
  if (n == 0) return k;

  constexpr int SLOTS = (128 + 64) / VEC;               // 24 uint4 per head
  const int slots_per_token = (int)num_heads * SLOTS;
  const int threads = slots_per_token < 256 ? ((slots_per_token + 31) / 32) * 32
                                            : 256;
  const dim3 grid((slots_per_token + threads - 1) / threads, (unsigned)n);
  auto stream = at::cuda::getCurrentCUDAStream();
  build_kv_kernel<128, 64, 128><<<grid, threads, 0, stream>>>(
      reinterpret_cast<const vec_t*>(kvb.data_ptr()),
      reinterpret_cast<const vec_t*>(kpe.data_ptr()),
      reinterpret_cast<vec_t*>(k.data_ptr()),
      slots_per_token, kvb.stride(0) / VEC, kpe.stride(0) / VEC);
  return k;
}

// ===========================================================================
// Short-sequence MLA attention, straight off the ``kv_b_proj`` output.
//
// FlashAttention-4 is the right kernel for a 16k-token prefill, but at 1-443
// tokens its CuTeDSL launcher costs ~20-40 us of host time to enqueue ~7 us of
// GPU work -- and this operator is host-bound at those lengths, so the launcher
// alone is a third of the measured latency.  Up to ~160 tokens this kernel is
// faster end to end (38 us vs 73 us of measured forward at 1 token, 50 vs 72 at
// 64, 67 vs 71 at 128); past that FA4's tile pipeline pulls ahead and the caller
// switches back.  It also skips the K/V assembly entirely: the tiles it
// stages are gathered
// straight from the ``kv_b_proj`` output (NoPE half for K, value half for V) and
// from the shared RoPE tail of the fused QKV buffer, so neither K nor V is ever
// materialized.  It only handles the dense causal case the captures exercise:
// one query per key position, ``cu_seqlens_q == cu_seqlens_k``.
//
// One warp owns 16 queries of one head end to end, so both softmax reductions
// stay inside the warp; eight warps share a block (and therefore the staged K/V
// tiles).  Q lives in registers for the whole k-loop as 12 ``m16n8k16`` A
// fragments, and the score accumulators *are* the next ``mma``'s A fragments --
// the QK output layout (row = lane/4, col = 2*(lane%4)) is exactly the A-operand
// layout with the key index as k, so P goes into the PV ``mma`` with a bf16
// pack and no shuffles.  V is the one operand that needs the other orientation
// (``mma`` wants its B operand k-contiguous, i.e. key-contiguous), so the value
// tile is transposed on the way into shared memory.
// ===========================================================================

__device__ __forceinline__ void mma_m16n8k16(float (&d)[4], const uint32_t (&a)[4],
                                             const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b[0]), "r"(b[1]));
}

constexpr int AT_QT = 16;                 // queries per warp (one mma m-tile)
constexpr int AT_BK = 32;                 // keys per staged tile

// ``WARPS`` trades SM coverage for tile sharing: eight warps amortize one staged
// K/V tile over 128 queries but leave a 64-token forward at one block per head.
// Sharing wins anyway -- see the numbers in ``attn_small``.
template <int DN, int DR, int DV, int WARPS>
__global__ __launch_bounds__(WARPS * 32) void attn_small_kernel(
    const __nv_bfloat16* __restrict__ q, long lq,
    const __nv_bfloat16* __restrict__ kvb, long lkvb,
    const __nv_bfloat16* __restrict__ kpe, long lkpe,
    __nv_bfloat16* __restrict__ out, long lo,
    const int* __restrict__ cu_seqlens, float scale) {
  constexpr int DQ = DN + DR;
  constexpr int AT_THREADS = WARPS * 32;
  constexpr int AT_BM = AT_QT * WARPS;
  constexpr int KSTEPS = DQ / 16;          // QK k-steps
  constexpr int NT_S = AT_BK / 8;          // score n-tiles
  constexpr int KT_P = AT_BK / 16;         // PV k-tiles
  constexpr int NT_O = DV / 8;             // output n-tiles
  constexpr int SK = DQ + 8;               // K row pitch (bf16 lanes)
  constexpr int SV = AT_BK + 2;            // V^T row pitch
  constexpr int K32 = DQ / 2;
  constexpr int V32 = DV / 2;
  constexpr int N32 = DN / 2;

  __shared__ __align__(16) __nv_bfloat16 Ks[AT_BK * SK];
  __shared__ __align__(16) __nv_bfloat16 Vt[DV * SV];

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, tg = lane & 3;
  const int h = blockIdx.y;
  const int base = cu_seqlens[blockIdx.z];
  const int len = cu_seqlens[blockIdx.z + 1] - base;
  const int m0 = blockIdx.x * AT_BM;
  if (m0 >= len) return;
  const long hkv = (long)h * (DN + DV);

  // Q as 12 A fragments, resident for the whole k-loop.
  const int r0 = m0 + warp * AT_QT + g;    // this lane's two query rows
  const int r1 = r0 + 8;
  uint32_t qf[KSTEPS][4];
  {
    const __nv_bfloat16* qa = q + (long)(base + r0) * lq + (long)h * DQ;
    const __nv_bfloat16* qb = q + (long)(base + r1) * lq + (long)h * DQ;
    const bool oka = r0 < len, okb = r1 < len;
#pragma unroll
    for (int kk = 0; kk < KSTEPS; ++kk) {
      const int d0 = kk * 16 + tg * 2;
      qf[kk][0] = oka ? *(const uint32_t*)(qa + d0) : 0u;
      qf[kk][1] = okb ? *(const uint32_t*)(qb + d0) : 0u;
      qf[kk][2] = oka ? *(const uint32_t*)(qa + d0 + 8) : 0u;
      qf[kk][3] = okb ? *(const uint32_t*)(qb + d0 + 8) : 0u;
    }
  }

  float oacc[NT_O][4];
#pragma unroll
  for (int nt = 0; nt < NT_O; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) oacc[nt][e] = 0.f;
  float mi[2] = {-1e30f, -1e30f}, li[2] = {0.f, 0.f};

  const int kmax = min(len, m0 + AT_BM);
  for (int j0 = 0; j0 < kmax; j0 += AT_BK) {
    const int krows = min(AT_BK, len - j0);
    __syncthreads();
    // K tile: NoPE half from kv_b_proj, RoPE tail shared by every head.
    for (int idx = tid; idx < AT_BK * K32; idx += AT_THREADS) {
      const int r = idx / K32, c = idx - r * K32;
      uint32_t val = 0u;
      if (r < krows) {
        const long t = base + j0 + r;
        val = c < N32
            ? *(const uint32_t*)(kvb + t * lkvb + hkv + c * 2)
            : *(const uint32_t*)(kpe + t * lkpe + (c - N32) * 2);
      }
      *(uint32_t*)(&Ks[r * SK + c * 2]) = val;
    }
    // V tile, transposed into [dim][key] so mma's B operand is key-contiguous.
    for (int idx = tid; idx < AT_BK * V32; idx += AT_THREADS) {
      const int r = idx / V32, c = idx - r * V32;
      uint32_t val = 0u;
      if (r < krows)
        val = *(const uint32_t*)(kvb + (long)(base + j0 + r) * lkvb + hkv + DN +
                                 c * 2);
      Vt[(c * 2) * SV + r] = __ushort_as_bfloat16((unsigned short)(val & 0xffffu));
      Vt[(c * 2 + 1) * SV + r] = __ushort_as_bfloat16((unsigned short)(val >> 16));
    }
    __syncthreads();

    float sacc[NT_S][4];
#pragma unroll
    for (int nt = 0; nt < NT_S; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) sacc[nt][e] = 0.f;
#pragma unroll
    for (int kk = 0; kk < KSTEPS; ++kk) {
#pragma unroll
      for (int nt = 0; nt < NT_S; ++nt) {
        const __nv_bfloat16* kb = &Ks[(nt * 8 + g) * SK + kk * 16 + tg * 2];
        const uint32_t bf[2] = {*(const uint32_t*)(kb),
                                *(const uint32_t*)(kb + 8)};
        mma_m16n8k16(sacc[nt], qf[kk], bf);
      }
    }

    // scale + causal mask, then the row max over this lane's 8 columns
    float rmax[2] = {-1e30f, -1e30f};
#pragma unroll
    for (int nt = 0; nt < NT_S; ++nt) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int qrow = half ? r1 : r0;
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int key = nt * 8 + tg * 2 + e;
          const bool ok = (qrow < len) && (key < krows) && (j0 + key <= qrow);
          float v = ok ? sacc[nt][half * 2 + e] * scale : -1e30f;
          sacc[nt][half * 2 + e] = v;
          rmax[half] = fmaxf(rmax[half], v);
        }
      }
    }
    // the other 24 columns of these rows live in the 3 sibling lanes g*4+*
#pragma unroll
    for (int off = 1; off < 4; off <<= 1) {
#pragma unroll
      for (int half = 0; half < 2; ++half)
        rmax[half] = fmaxf(rmax[half], __shfl_xor_sync(0xffffffff, rmax[half], off));
    }

    float corr[2], rsum[2] = {0.f, 0.f};
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const float nm = fmaxf(mi[half], rmax[half]);
      corr[half] = __expf(mi[half] - nm);
      mi[half] = nm;
    }
#pragma unroll
    for (int nt = 0; nt < NT_S; ++nt)
#pragma unroll
      for (int half = 0; half < 2; ++half)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float p = __expf(sacc[nt][half * 2 + e] - mi[half]);
          sacc[nt][half * 2 + e] = p;
          rsum[half] += p;
        }
#pragma unroll
    for (int off = 1; off < 4; off <<= 1) {
#pragma unroll
      for (int half = 0; half < 2; ++half)
        rsum[half] += __shfl_xor_sync(0xffffffff, rsum[half], off);
    }
#pragma unroll
    for (int half = 0; half < 2; ++half)
      li[half] = li[half] * corr[half] + rsum[half];
#pragma unroll
    for (int nt = 0; nt < NT_O; ++nt) {
      oacc[nt][0] *= corr[0];
      oacc[nt][1] *= corr[0];
      oacc[nt][2] *= corr[1];
      oacc[nt][3] *= corr[1];
    }

    // O += P @ V.  sacc holds P in exactly the A-fragment layout.
#pragma unroll
    for (int kt = 0; kt < KT_P; ++kt) {
      const __nv_bfloat162 a0 =
          __floats2bfloat162_rn(sacc[kt * 2][0], sacc[kt * 2][1]);
      const __nv_bfloat162 a1 =
          __floats2bfloat162_rn(sacc[kt * 2][2], sacc[kt * 2][3]);
      const __nv_bfloat162 a2 =
          __floats2bfloat162_rn(sacc[kt * 2 + 1][0], sacc[kt * 2 + 1][1]);
      const __nv_bfloat162 a3 =
          __floats2bfloat162_rn(sacc[kt * 2 + 1][2], sacc[kt * 2 + 1][3]);
      const uint32_t pa[4] = {*(const uint32_t*)&a0, *(const uint32_t*)&a1,
                              *(const uint32_t*)&a2, *(const uint32_t*)&a3};
#pragma unroll
      for (int nt = 0; nt < NT_O; ++nt) {
        const __nv_bfloat16* vb = &Vt[(nt * 8 + g) * SV + kt * 16 + tg * 2];
        const uint32_t vf[2] = {*(const uint32_t*)(vb),
                                *(const uint32_t*)(vb + 8)};
        mma_m16n8k16(oacc[nt], pa, vf);
      }
    }
  }

#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int qrow = half ? r1 : r0;
    if (qrow >= len) continue;
    const float inv = 1.f / li[half];
    __nv_bfloat16* dst = out + (long)(base + qrow) * lo + (long)h * DV + tg * 2;
#pragma unroll
    for (int nt = 0; nt < NT_O; ++nt)
      *(__nv_bfloat162*)(dst + nt * 8) = __floats2bfloat162_rn(
          oacc[nt][half * 2] * inv, oacc[nt][half * 2 + 1] * inv);
  }
}

at::Tensor attn_small(const at::Tensor& q, const at::Tensor& kvb,
                      const at::Tensor& kpe, const at::Tensor& cu_seqlens,
                      int64_t num_heads, int64_t d_nope, int64_t d_rope,
                      int64_t d_v, int64_t max_seqlen, double scale,
                      int64_t force_warps) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kBFloat16, "attn_small: q");
  TORCH_CHECK(kvb.scalar_type() == at::kBFloat16 &&
              kpe.scalar_type() == at::kBFloat16, "attn_small: bf16 only");
  TORCH_CHECK(q.dim() == 2 && kvb.dim() == 2 && kpe.dim() == 2,
              "attn_small: 2-D inputs");
  TORCH_CHECK(q.stride(1) == 1 && kvb.stride(1) == 1 && kpe.stride(1) == 1,
              "attn_small: dense rows");
  TORCH_CHECK(d_nope == 128 && d_rope == 64 && d_v == 128,
              "attn_small: head dims must be (128, 64, 128)");
  TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt && cu_seqlens.is_cuda(),
              "attn_small: cu_seqlens must be int32 cuda");
  TORCH_CHECK(q.size(1) >= num_heads * (d_nope + d_rope), "attn_small: q width");
  TORCH_CHECK(kvb.size(1) == num_heads * (d_nope + d_v), "attn_small: kvb width");
  TORCH_CHECK(kpe.size(1) >= d_rope, "attn_small: kpe width");
  const long m = q.size(0);
  TORCH_CHECK(kvb.size(0) == m && kpe.size(0) == m,
              "attn_small: token count mismatch");
  at::Tensor out = at::empty({m, num_heads * d_v}, q.options());
  if (m == 0) return out;

  const c10::cuda::CUDAGuard guard(q.device());
  const int nseq = (int)cu_seqlens.numel() - 1;
  // Eight warps -- 128 queries per block -- measured best at every length in
  // range (1 token: 38 us of forward either way; 64: 50 us vs 60 at four warps;
  // 128: 67 vs 83).  Sharing one staged K/V tile across more queries beats
  // spreading narrower blocks over more SMs, even at 26 tokens where the grid is
  // one block per head.  Sixteen warps would not fit the 165-register frame.
  const int warps = force_warps > 0 ? (int)force_warps : 8;
  auto stream = at::cuda::getCurrentCUDAStream();
#define ATTN_CASE(w)                                                          \
  case (w): {                                                                 \
    const dim3 grid(((int)max_seqlen + AT_QT * (w) - 1) / (AT_QT * (w)),       \
                    (unsigned)num_heads, (unsigned)nseq);                     \
    attn_small_kernel<128, 64, 128, (w)><<<grid, (w) * 32, 0, stream>>>(       \
        (const __nv_bfloat16*)q.data_ptr(), q.stride(0),                      \
        (const __nv_bfloat16*)kvb.data_ptr(), kvb.stride(0),                  \
        (const __nv_bfloat16*)kpe.data_ptr(), kpe.stride(0),                  \
        (__nv_bfloat16*)out.data_ptr(), num_heads * d_v,                      \
        cu_seqlens.data_ptr<int>(), (float)scale);                            \
    break;                                                                    \
  }
  switch (warps) {
    ATTN_CASE(1) ATTN_CASE(2) ATTN_CASE(4) ATTN_CASE(8)
    default: TORCH_CHECK(false, "attn_small: bad warp count ", warps);
  }
#undef ATTN_CASE
  return out;
}

} // namespace mlafast

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("build_kv", &mlafast::build_kv,
        "Assemble contiguous MLA K from kv_b_proj output + shared RoPE tail");
  m.def("attn_small", &mlafast::attn_small,
        "Causal MLA attention for short sequences, gathered from kv_b_proj");
}
