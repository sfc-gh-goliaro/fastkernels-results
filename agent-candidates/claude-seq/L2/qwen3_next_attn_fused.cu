// Fused Qwen3-Next full-attention layer (L2 candidate).
//
// Two kernels plus the two projections, launched from a single host call so the
// whole layer costs one Python round trip:
//
//   1) qk_norm_rope_store  -- per-head QK-RMSNorm + partial NeoX RoPE for Q,
//      and for K straight into the paged HND cache; V is copied into the cache.
//      The gate half of the QKV buffer is left in place; the attention epilogue
//      reads it from there, so no gate copy is needed.
//
//   2) attn  -- paged causal GQA flash attention (bf16 WMMA) whose epilogue
//      divides by the softmax denominator and multiplies by sigmoid(gate).
//      Two passes over the keys (scores first, then P*V) instead of an online
//      softmax: at the captured prefill lengths the whole score row fits in
//      shared memory, which removes the per-tile accumulator rescale.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace wmma = nvcuda::wmma;

using bf16 = __nv_bfloat16;

#define CDIV(a, b) (((a) + (b) - 1) / (b))

// ---------------------------------------------------------------------------
// 1) QK-RMSNorm + partial RoPE + paged K/V store.
//    grid  = (ceil(N / WARPS), HQ + 2 * HKV)
//    block = 32 * WARPS   (one warp per token, one head slot per block row)
// ---------------------------------------------------------------------------
template <int D, int ROT, int PAGE, int WARPS, bool SLOT64>
__global__ __launch_bounds__(32 * WARPS) void qk_norm_rope_store_kernel(
    const bf16* __restrict__ qkv,
    bf16* __restrict__ q_out,
    bf16* __restrict__ k_cache,
    bf16* __restrict__ v_cache,
    const void* __restrict__ slot_ptr,
    const float* __restrict__ q_gain,
    const float* __restrict__ k_gain,
    const float* __restrict__ cos_sin,
    const int64_t* __restrict__ positions,
    int n_tokens,
    int64_t qkv_stride,
    int cos_stride,
    int HQ,
    int HKV,
    float eps) {
  constexpr int E = D / 32;            // elements per lane
  constexpr int HALF = ROT / 2;
  constexpr int RLANES = ROT / E;      // lanes covering the rotary half-pair
  constexpr int HLANES = RLANES / 2;

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int t = blockIdx.x * WARPS + warp;
  if (t >= n_tokens) return;
  const int hs = blockIdx.y;

  float x[E];
  const bf16* src;
  const float* gain;
  bool is_v = hs >= HQ + HKV;
  bool is_k = !is_v && hs >= HQ;
  int kvh = is_v ? (hs - HQ - HKV) : (is_k ? hs - HQ : 0);

  if (is_v) {
    src = qkv + t * qkv_stride + (int64_t)(HQ * 2 * D + HKV * D + kvh * D) + lane * E;
    gain = nullptr;
  } else if (is_k) {
    src = qkv + t * qkv_stride + (int64_t)(HQ * 2 * D + kvh * D) + lane * E;
    gain = k_gain;
  } else {
    src = qkv + t * qkv_stride + (int64_t)(hs * 2 * D) + lane * E;
    gain = q_gain;
  }

  bf16 xb[E];
  *reinterpret_cast<uint4*>(xb) = *reinterpret_cast<const uint4*>(src);
  // Start the position -> cos/sin dependent chain now; the RMSNorm reduction
  // below covers its latency.
  const float* cs = nullptr;
  if (!is_v && lane < RLANES) {
    cs = cos_sin + positions[t] * (int64_t)cos_stride + (lane % HLANES) * E;
  }

  int64_t slot = 0;
  if (is_v || is_k) {
    slot = SLOT64 ? ((const int64_t*)slot_ptr)[t] : (int64_t)((const int*)slot_ptr)[t];
  }

  if (is_v) {
    if (slot >= 0) {
      bf16* dst = v_cache + (slot / PAGE) * (int64_t)(HKV * PAGE * D)
                  + (int64_t)kvh * (PAGE * D) + (slot % PAGE) * D + lane * E;
      *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(xb);
    }
    return;
  }

  // ---- RMSNorm over the full head ----
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < E; ++i) {
    x[i] = __bfloat162float(xb[i]);
    ss += x[i] * x[i];
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
  const float inv = rsqrtf(ss / (float)D + eps);

  float w[E];
#pragma unroll
  for (int i = 0; i < E / 4; ++i) {
    *reinterpret_cast<float4*>(w + 4 * i) =
        *reinterpret_cast<const float4*>(gain + lane * E + 4 * i);
  }

  // Round-trip through bf16 so the RoPE input matches the reference's
  // (rmsnorm -> memory -> rope) behaviour.
  float xn[E];
#pragma unroll
  for (int i = 0; i < E; ++i) xn[i] = __bfloat162float(__float2bfloat16(x[i] * inv * w[i]));

  // ---- partial NeoX RoPE on [0, ROT) ----
  float partner[E];
#pragma unroll
  for (int i = 0; i < E; ++i)
    partner[i] = __shfl_xor_sync(0xffffffffu, xn[i], HLANES);

  float y[E];
  if (lane < RLANES) {
    float cv[E], sv[E];
#pragma unroll
    for (int i = 0; i < E / 4; ++i) {
      *reinterpret_cast<float4*>(cv + 4 * i) = *reinterpret_cast<const float4*>(cs + 4 * i);
      *reinterpret_cast<float4*>(sv + 4 * i) =
          *reinterpret_cast<const float4*>(cs + HALF + 4 * i);
    }
    const float sign = (lane < HLANES) ? -1.f : 1.f;
#pragma unroll
    for (int i = 0; i < E; ++i) y[i] = xn[i] * cv[i] + sign * partner[i] * sv[i];
  } else {
#pragma unroll
    for (int i = 0; i < E; ++i) y[i] = xn[i];
  }

  bf16 yb[E];
#pragma unroll
  for (int i = 0; i < E; ++i) yb[i] = __float2bfloat16(y[i]);

  bf16* dst;
  if (is_k) {
    if (slot < 0) return;
    dst = k_cache + (slot / PAGE) * (int64_t)(HKV * PAGE * D)
          + (int64_t)kvh * (PAGE * D) + (slot % PAGE) * D + lane * E;
  } else {
    dst = q_out + (int64_t)t * (HQ * D) + hs * D + lane * E;
  }
  *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(yb);
}

// ---------------------------------------------------------------------------
// 2) Paged causal GQA attention with fused sigmoid gate.
//    grid  = (ceil(max_q / BM), HQ, num_seqs)
//    block = 32 * NWARPS
//
//  * K/V tiles are staged into shared memory with cp.async so all 16 128-bit
//    copies of a tile are in flight at once; a plain load/store loop serialises
//    on the global latency of each copy, which at these tiny grids is the whole
//    kernel (46% long-scoreboard stall before the change).
//  * The page ids for a tile are read once per tile, not once per 8 elements.
//  * Scores live in shared memory for the whole key range (SMAX), so the
//    P*V accumulators never need the online-softmax rescale; the output
//    accumulator aliases the K/V staging buffer.
// ---------------------------------------------------------------------------
template <int BM, int BN, int D, int PAGE, int NWARPS, int SMAX>
__global__ __launch_bounds__(32 * NWARPS) void attn_kernel(
    const bf16* __restrict__ q,
    const bf16* __restrict__ k_cache,
    const bf16* __restrict__ v_cache,
    const bf16* __restrict__ qkv,
    bf16* __restrict__ out,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    const int* __restrict__ cu_q,
    int bt_stride,
    int64_t qkv_stride,
    int HQ,
    int HKV,
    int GQA,
    float scale) {
  // WMMA requires ldm to be a multiple of 16B / element size; padding the
  // rows also breaks the 16-way bank conflict a stride of exactly D would give.
  constexpr int SS = SMAX + 8;                 // padded score stride
  constexpr int LDB = D + 8;                   // padded Q / K / V stride
  constexpr int LDA = D + 4;                   // padded accumulator stride
  constexpr int NFRAG_N = BN / 16;             // score subtiles per key tile
  constexpr int DW = D / NWARPS;               // output columns per warp
  constexpr int NT = 32 * NWARPS;
  constexpr int VEC = BN * (D / 8) / NT;       // 128-bit copies per thread
  constexpr int QVEC = BM * (D / 8) / NT;
  constexpr int PPT = BN / PAGE;               // pages per key tile

  extern __shared__ char smem_raw[];
  bf16* qs = reinterpret_cast<bf16*>(smem_raw);            // [BM][LDB]
  bf16* kv = qs + BM * LDB;                                // [BN][LDB]
  float* sf = reinterpret_cast<float*>(kv + BN * LDB);     // [BM][SS]
  bf16* ps = reinterpret_cast<bf16*>(sf + BM * SS);        // [BM][SS]
  float* rl = reinterpret_cast<float*>(ps + BM * SS);      // [BM]
  float* ac = reinterpret_cast<float*>(kv);                // [BM][LDA] (aliased)

  const int tid = threadIdx.x;
  const int warp = tid >> 5;

  const int m0 = blockIdx.x * BM;
  const int h = blockIdx.y;
  const int s = blockIdx.z;

  const int q_start = cu_q[s];
  const int q_len = cu_q[s + 1] - q_start;
  if (m0 >= q_len) return;
  const int seq_len = seq_lens[s];
  const int ctx = seq_len - q_len;
  const int kv_end = min(seq_len, ctx + m0 + BM);
  const int kvh = h / GQA;

  // ---- stage Q, and prefetch the gate this thread needs in the epilogue ----
  uint4 gate_pf[QVEC];
#pragma unroll
  for (int i = 0; i < QVEC; ++i) {
    const int idx = tid + i * NT;
    const int m = idx / (D / 8);
    const int c8 = idx % (D / 8);
    if (m0 + m < q_len) {
      __pipeline_memcpy_async(
          qs + m * LDB + c8 * 8,
          q + (int64_t)(q_start + m0 + m) * (HQ * D) + h * D + c8 * 8, 16);
      gate_pf[i] = *reinterpret_cast<const uint4*>(
          qkv + (int64_t)(q_start + m0 + m) * qkv_stride + (int64_t)(h * 2 * D + D)
          + c8 * 8);
    } else {
      *reinterpret_cast<uint4*>(qs + m * LDB + c8 * 8) = make_uint4(0, 0, 0, 0);
    }
  }
  __pipeline_commit();

  const int ntiles = CDIV(kv_end, BN);
  const int64_t cache_page = (int64_t)HKV * PAGE * D;
  const int64_t head_off = (int64_t)kvh * (PAGE * D);

  // ---- pass 1: scores ----
  for (int tile = 0; tile < ntiles; ++tile) {
    const int kv0 = tile * BN;
    const int rows16 = min(BN, ((kv_end - kv0) + 15) & ~15);
    int pages[PPT];
#pragma unroll
    for (int i = 0; i < PPT; ++i) {
      const int key = kv0 + i * PAGE;
      pages[i] = (key < kv_end) ? block_table[s * bt_stride + (key / PAGE)] : -1;
    }
    if (tile == 0) __pipeline_wait_prior(0);
    __syncthreads();
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      const int idx = tid + i * NT;
      const int r = idx / (D / 8);
      const int c8 = idx % (D / 8);
      const int key = kv0 + r;
      if (key < kv_end) {
        __pipeline_memcpy_async(
            kv + r * LDB + c8 * 8,
            k_cache + pages[r / PAGE] * cache_page + head_off + (key % PAGE) * D + c8 * 8,
            16);
      } else if (r < rows16) {
        // only the 16-row tail the WMMA tiles actually read needs zeroing
        *reinterpret_cast<uint4*>(kv + r * LDB + c8 * 8) = make_uint4(0, 0, 0, 0);
      }
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    // subtiles entirely past the causal end are never read by the softmax
    if ((warp % NFRAG_N) * 16 < rows16) {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      wmma::fill_fragment(acc, 0.0f);
      wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> af[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> bfr[2];
      const bf16* kbase = kv + (warp % NFRAG_N) * 16 * LDB;
#pragma unroll 2
      for (int k = 0; k < D; k += 32) {
        wmma::load_matrix_sync(af[0], qs + k, LDB);
        wmma::load_matrix_sync(bfr[0], kbase + k, LDB);
        wmma::load_matrix_sync(af[1], qs + k + 16, LDB);
        wmma::load_matrix_sync(bfr[1], kbase + k + 16, LDB);
        wmma::mma_sync(acc, af[0], bfr[0], acc);
        wmma::mma_sync(acc, af[1], bfr[1], acc);
      }
      wmma::store_matrix_sync(sf + kv0 + (warp % NFRAG_N) * 16, acc, SS,
                              wmma::mem_row_major);
    }
  }
  __syncthreads();

  // ---- pass 2: causal softmax, 8 threads per row, float4 at a time ----
  {
    constexpr int TPR = 8;
    const int m = tid / TPR;
    const int cg = tid % TPR;
    if (m < BM) {
      const int qpos = ctx + m0 + m;
      const bool live = (m0 + m) < q_len;
      const int n4 = CDIV(kv_end, 4);
      float mx = -INFINITY;
      for (int j4 = cg; j4 < n4; j4 += TPR) {
        float4 v = *reinterpret_cast<const float4*>(sf + m * SS + j4 * 4);
        float* vv = reinterpret_cast<float*>(&v);
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          const int j = j4 * 4 + t;
          vv[t] = (live && j <= qpos && j < kv_end) ? vv[t] * scale : -INFINITY;
          mx = fmaxf(mx, vv[t]);
        }
        *reinterpret_cast<float4*>(sf + m * SS + j4 * 4) = v;
      }
#pragma unroll
      for (int off = 1; off < TPR; off <<= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
      if (!live) mx = 0.f;
      float sum = 0.f;
      const int npad = (kv_end + 15) & ~15;
      for (int j4 = cg; j4 < npad / 4; j4 += TPR) {
        float4 v = *reinterpret_cast<const float4*>(sf + m * SS + j4 * 4);
        const float* vv = reinterpret_cast<const float*>(&v);
        bf16 pb[4];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          const float p = (j4 * 4 + t < kv_end) ? __expf(vv[t] - mx) : 0.f;
          sum += p;
          pb[t] = __float2bfloat16(p);
        }
        *reinterpret_cast<uint2*>(ps + m * SS + j4 * 4) =
            *reinterpret_cast<const uint2*>(pb);
      }
#pragma unroll
      for (int off = 1; off < TPR; off <<= 1)
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
      if (cg == 0) rl[m] = (sum > 0.f) ? 1.f / sum : 0.f;
    }
  }

  // ---- pass 3: P * V ----
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> oacc[DW / 16];
#pragma unroll
  for (int i = 0; i < DW / 16; ++i) wmma::fill_fragment(oacc[i], 0.0f);

  for (int tile = 0; tile < ntiles; ++tile) {
    const int kv0 = tile * BN;
    const int rows16 = min(BN, ((kv_end - kv0) + 15) & ~15);
    int pages[PPT];
#pragma unroll
    for (int i = 0; i < PPT; ++i) {
      const int key = kv0 + i * PAGE;
      pages[i] = (key < kv_end) ? block_table[s * bt_stride + (key / PAGE)] : -1;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      const int idx = tid + i * NT;
      const int r = idx / (D / 8);
      const int c8 = idx % (D / 8);
      const int key = kv0 + r;
      if (key < kv_end) {
        __pipeline_memcpy_async(
            kv + r * LDB + c8 * 8,
            v_cache + pages[r / PAGE] * cache_page + head_off + (key % PAGE) * D + c8 * 8,
            16);
      } else if (r < rows16) {
        *reinterpret_cast<uint4*>(kv + r * LDB + c8 * 8) = make_uint4(0, 0, 0, 0);
      }
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    const int kmax = min(BN, ((kv_end - kv0) + 15) & ~15);
    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> bfr;
#pragma unroll 1
    for (int k = 0; k < kmax; k += 16) {
      wmma::load_matrix_sync(af, ps + kv0 + k, SS);
#pragma unroll
      for (int i = 0; i < DW / 16; ++i) {
        wmma::load_matrix_sync(bfr, kv + k * LDB + warp * DW + i * 16, LDB);
        wmma::mma_sync(oacc[i], af, bfr, oacc[i]);
      }
    }
  }

  __syncthreads();
#pragma unroll
  for (int i = 0; i < DW / 16; ++i)
    wmma::store_matrix_sync(ac + warp * DW + i * 16, oacc[i], LDA, wmma::mem_row_major);
  __syncthreads();

  // ---- epilogue: / l, * sigmoid(gate), store ----
#pragma unroll
  for (int i = 0; i < QVEC; ++i) {
    const int idx = tid + i * NT;
    const int m = idx / (D / 8);
    const int c8 = idx % (D / 8);
    if (m0 + m >= q_len) continue;
    const int64_t row = q_start + m0 + m;
    const float r = rl[m];
    bf16 gb[8];
    *reinterpret_cast<uint4*>(gb) = gate_pf[i];
    bf16 ob[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float g = __bfloat162float(gb[j]);
      ob[j] = __float2bfloat16(ac[m * LDA + c8 * 8 + j] * r / (1.f + __expf(-g)));
    }
    *reinterpret_cast<uint4*>(out + row * (HQ * D) + h * D + c8 * 8) =
        *reinterpret_cast<const uint4*>(ob);
  }
}


// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

// C[M, NO] = A[M, K] * W[NO, K]^T -- the projections, same GEMM the baseline's
// ``F.linear`` runs, called from here so the whole layer is one host call.
static at::Tensor linear_nt(const at::Tensor& a, const at::Tensor& w) {
  return at::mm(a, w.t());
}

constexpr int kD = 256;
constexpr int kROT = 64;
constexpr int kPAGE = 16;
constexpr int kBM = 16;
constexpr int kBN = 64;
constexpr int kNWARPS = 4;
constexpr int kSMAX = 512;

template <int SMAX>
static size_t attn_smem_bytes() {
  const int SS = SMAX + 8;
  const int LDB = kD + 8;
  return sizeof(bf16) * (kBM * LDB) + sizeof(bf16) * (kBN * LDB)
         + sizeof(float) * (kBM * SS) + sizeof(bf16) * (kBM * SS)
         + sizeof(float) * kBM;
}

static void check_prep(const at::Tensor& qkv, const at::Tensor& k_cache,
                       const at::Tensor& slot_mapping, const at::Tensor& cos_sin,
                       const at::Tensor& positions, int HQ, int HKV) {
  TORCH_CHECK(qkv.scalar_type() == at::kBFloat16 && k_cache.scalar_type() == at::kBFloat16,
              "qwen3_next_attn: bf16 only");
  TORCH_CHECK(qkv.size(1) == (HQ * 2 + 2 * HKV) * kD, "qwen3_next_attn: bad qkv width");
  TORCH_CHECK(qkv.stride(1) == 1, "qwen3_next_attn: qkv rows must be contiguous");
  TORCH_CHECK(k_cache.is_contiguous() && k_cache.size(1) == HKV
                  && k_cache.size(2) == kPAGE && k_cache.size(3) == kD,
              "qwen3_next_attn: unexpected KV cache layout");
  TORCH_CHECK(slot_mapping.scalar_type() == at::kInt
                  || slot_mapping.scalar_type() == at::kLong,
              "qwen3_next_attn: slot_mapping must be int32/int64");
  TORCH_CHECK(cos_sin.scalar_type() == at::kFloat && cos_sin.size(1) == kROT,
              "qwen3_next_attn: cos_sin_cache must be fp32 [P, 64]");
  TORCH_CHECK(positions.scalar_type() == at::kLong, "qwen3_next_attn: positions int64");
}

static void check_attn(const at::Tensor& block_table, const at::Tensor& seq_lens,
                       const at::Tensor& cu_q, int max_seq_len) {
  TORCH_CHECK(block_table.scalar_type() == at::kInt && block_table.stride(1) == 1,
              "qwen3_next_attn: block_table must be int32, row-contiguous");
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt && cu_q.scalar_type() == at::kInt,
              "qwen3_next_attn: seq_lens / cu_seqlens must be int32");
  TORCH_CHECK(max_seq_len <= kSMAX, "qwen3_next_attn: seq_len above kernel limit");
}

void launch_qk_norm_rope_store(const at::Tensor& qkv, at::Tensor& q_out,
                               at::Tensor& k_cache, at::Tensor& v_cache,
                               const at::Tensor& slot_mapping,
                               const at::Tensor& q_gain, const at::Tensor& k_gain,
                               const at::Tensor& cos_sin, const at::Tensor& positions,
                               int HQ, int HKV, double eps, cudaStream_t stream) {
  check_prep(qkv, k_cache, slot_mapping, cos_sin, positions, HQ, HKV);
  constexpr int WARPS = 4;
  const int n = (int)qkv.size(0);
  dim3 grid(CDIV(n, WARPS), HQ + 2 * HKV);
  dim3 block(32 * WARPS);
  const bool slot64 = slot_mapping.scalar_type() == at::kLong;
  auto run = [&](auto tag) {
    constexpr bool S64 = decltype(tag)::value;
    qk_norm_rope_store_kernel<kD, kROT, kPAGE, WARPS, S64><<<grid, block, 0, stream>>>(
        (const bf16*)qkv.data_ptr(), (bf16*)q_out.data_ptr(),
        (bf16*)k_cache.data_ptr(), (bf16*)v_cache.data_ptr(),
        slot_mapping.data_ptr(), q_gain.data_ptr<float>(), k_gain.data_ptr<float>(),
        cos_sin.data_ptr<float>(), positions.data_ptr<int64_t>(), n,
        qkv.stride(0), (int)cos_sin.stride(0), HQ, HKV, (float)eps);
  };
  if (slot64) run(std::true_type{});
  else run(std::false_type{});
}

template <int SMAX>
static void launch_attn_tpl(const at::Tensor& q, const at::Tensor& k_cache,
                            const at::Tensor& v_cache, const at::Tensor& qkv,
                            at::Tensor& out, const at::Tensor& block_table,
                            const at::Tensor& seq_lens, const at::Tensor& cu_q,
                            int num_seqs, int max_query_len, int HQ, int HKV,
                            double scale, cudaStream_t stream) {
  dim3 grid(CDIV(max_query_len, kBM), HQ, num_seqs);
  dim3 block(32 * kNWARPS);
  const size_t smem = attn_smem_bytes<SMAX>();
  auto kern = attn_kernel<kBM, kBN, kD, kPAGE, kNWARPS, SMAX>;
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr_set = true;
  }
  kern<<<grid, block, smem, stream>>>(
      (const bf16*)q.data_ptr(), (const bf16*)k_cache.data_ptr(),
      (const bf16*)v_cache.data_ptr(), (const bf16*)qkv.data_ptr(),
      (bf16*)out.data_ptr(), block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
      cu_q.data_ptr<int>(), (int)block_table.stride(0), qkv.stride(0), HQ, HKV,
      HQ / HKV, (float)scale);
}

void launch_attn(const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
                 const at::Tensor& qkv, at::Tensor& out, const at::Tensor& block_table,
                 const at::Tensor& seq_lens, const at::Tensor& cu_q, int num_seqs,
                 int max_query_len, int max_seq_len, int HQ, int HKV, double scale,
                 cudaStream_t stream) {
  check_attn(block_table, seq_lens, cu_q, max_seq_len);
  TORCH_CHECK(q.is_contiguous() && out.is_contiguous(),
              "qwen3_next_attn: q / out must be contiguous");
  if (max_seq_len <= 64) {
    launch_attn_tpl<64>(q, k_cache, v_cache, qkv, out, block_table, seq_lens, cu_q,
                        num_seqs, max_query_len, HQ, HKV, scale, stream);
  } else if (max_seq_len <= 128) {
    launch_attn_tpl<128>(q, k_cache, v_cache, qkv, out, block_table, seq_lens, cu_q,
                         num_seqs, max_query_len, HQ, HKV, scale, stream);
  } else if (max_seq_len <= 256) {
    launch_attn_tpl<256>(q, k_cache, v_cache, qkv, out, block_table, seq_lens, cu_q,
                         num_seqs, max_query_len, HQ, HKV, scale, stream);
  } else {
    launch_attn_tpl<kSMAX>(q, k_cache, v_cache, qkv, out, block_table, seq_lens, cu_q,
                           num_seqs, max_query_len, HQ, HKV, scale, stream);
  }
}

int attn_max_seq_len() { return kSMAX; }

// Standalone entry points (testing / mixed paths).
void qk_norm_rope_store(at::Tensor qkv, at::Tensor q_out, at::Tensor k_cache,
                        at::Tensor v_cache, at::Tensor slot_mapping, at::Tensor q_gain,
                        at::Tensor k_gain, at::Tensor cos_sin, at::Tensor positions,
                        int64_t num_heads, int64_t num_kv_heads, double eps) {
  launch_qk_norm_rope_store(qkv, q_out, k_cache, v_cache, slot_mapping, q_gain, k_gain,
                            cos_sin, positions, (int)num_heads, (int)num_kv_heads, eps,
                            at::cuda::getCurrentCUDAStream());
}

void attn(at::Tensor q, at::Tensor k_cache, at::Tensor v_cache, at::Tensor qkv,
          at::Tensor out, at::Tensor block_table, at::Tensor seq_lens, at::Tensor cu_q,
          int64_t num_seqs, int64_t max_query_len, int64_t max_seq_len,
          int64_t num_heads, int64_t num_kv_heads, double scale) {
  launch_attn(q, k_cache, v_cache, qkv, out, block_table, seq_lens, cu_q, (int)num_seqs,
              (int)max_query_len, (int)max_seq_len, (int)num_heads, (int)num_kv_heads,
              scale, at::cuda::getCurrentCUDAStream());
}

// Split entry points: the projection + prep half (one call), and the output
// projection, so a mid-length step can put a different attention kernel in
// between and still pay only three Python round trips.
std::vector<at::Tensor> prep(at::Tensor x, at::Tensor w_qkv, at::Tensor q_gain,
                             at::Tensor k_gain, at::Tensor cos_sin, at::Tensor positions,
                             at::Tensor k_cache, at::Tensor v_cache,
                             at::Tensor slot_mapping, int64_t num_heads,
                             int64_t num_kv_heads, double eps) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const int n = (int)x.size(0);
  at::Tensor qkv = linear_nt(x, w_qkv);
  at::Tensor q = at::empty({n, (int64_t)(num_heads * kD)}, x.options());
  launch_qk_norm_rope_store(qkv, q, k_cache, v_cache, slot_mapping, q_gain, k_gain,
                            cos_sin, positions, (int)num_heads, (int)num_kv_heads, eps,
                            stream);
  return {qkv, q};
}

at::Tensor proj(at::Tensor o, at::Tensor w_o) { return linear_nt(o, w_o); }


// One-call fast path: qkv GEMM -> norm/rope/store -> attention+gate -> o GEMM.
at::Tensor fused_forward(at::Tensor x, at::Tensor w_qkv, at::Tensor w_o,
                         at::Tensor q_gain, at::Tensor k_gain, at::Tensor cos_sin,
                         at::Tensor positions, at::Tensor k_cache, at::Tensor v_cache,
                         at::Tensor slot_mapping, at::Tensor block_table,
                         at::Tensor seq_lens, at::Tensor cu_q, int64_t num_seqs,
                         int64_t max_query_len, int64_t max_seq_len, int64_t num_heads,
                         int64_t num_kv_heads, double eps, double scale) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const int n = (int)x.size(0);
  at::Tensor qkv = linear_nt(x, w_qkv);
  auto opts = x.options();
  at::Tensor q = at::empty({n, (int64_t)(num_heads * kD)}, opts);
  launch_qk_norm_rope_store(qkv, q, k_cache, v_cache, slot_mapping, q_gain, k_gain,
                            cos_sin, positions, (int)num_heads, (int)num_kv_heads, eps,
                            stream);
  at::Tensor o = at::empty({n, (int64_t)(num_heads * kD)}, opts);
  launch_attn(q, k_cache, v_cache, qkv, o, block_table, seq_lens, cu_q, (int)num_seqs,
              (int)max_query_len, (int)max_seq_len, (int)num_heads, (int)num_kv_heads,
              scale, stream);
  return linear_nt(o, w_o);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qk_norm_rope_store", &qk_norm_rope_store, "QK norm + RoPE + paged KV store");
  m.def("attn", &attn, "paged causal GQA attention with sigmoid gate");
  m.def("fused_forward", &fused_forward, "whole attention layer, one call");
  m.def("prep", &prep, "qkv projection + QK norm/RoPE + paged KV store");
  m.def("proj", &proj, "output projection");
  m.def("attn_max_seq_len", &attn_max_seq_len, "max seq_len the attn kernel supports");
}
