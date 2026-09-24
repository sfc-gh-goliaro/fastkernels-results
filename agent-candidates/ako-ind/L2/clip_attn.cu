// Fused single-pass CLIP self-attention over a packed QKV buffer.
//
// One launch replaces the baseline's QK^T -> *scale -> +mask -> softmax -> PV
// -> transpose -> contiguous -> reshape chain (7 launches, 2 full-tensor
// copies).  The whole score row of one query fits in registers spread across a
// warp, so no online (flash) rescaling is needed: the block computes the global
// row max and row sum first, then normalizes once.
//
// Layout contract
//   qkv  [B, S, 3E]  contiguous, E = NH*HD, Q at 0, K at E, V at 2E,
//                    head h of each at + h*HD.  ``scale`` is pre-folded into
//                    the Q projection weights, so no scaling happens here.
//   mask [*, *, S, S] any 4-D broadcastable mask (strides passed in, 0 for
//                    broadcast dims), or nullptr.
//   out  [B, S, E]   head-concatenated -- exactly the layout out_proj wants,
//                    so no transpose/contiguous is needed afterwards.
//
// Shared-memory traffic is what this kernel is shaped around.  ncu on the
// one-row-per-warp version: L1/TEX throughput 51% (the top metric -- compute
// 12.7%, DRAM 0.9%), 1152 instructions/warp, 8.2 achieved warps/SM.  The cause
// is that a warp owning one query row reads the *entire* K tile (S*64 floats,
// 19.7KB at S=77) and the entire V tile from shared, so the S*NH = 924 row-warps
// pull ~36MB through L1 for 380 MFLOP of work.
//
// ``RPW`` query rows per warp divides that traffic by RPW: the K row and V row
// loaded for one key are reused by all RPW rows of the warp, and only the score
// registers and the P shuffles scale with RPW.  Instruction count per row falls
// only ~10% (the FFMAs are irreducible) but shared bytes per row fall ~1/RPW,
// which is the metric that was saturated.  ROWS (rows per block) is independent
// and sets the grid: grid.x = ceil(S/ROWS), warps/block = ROWS/RPW.
//
// Two things that do *not* work, both measured: splitting each row's *key* range
// across warps to raise occupancy (5.92 -> 6.71 -> 7.39us for 1/2/4-way) -- the
// extra block reductions and the 32-key padding waste cost more than the added
// warps buy; and splitting the *head* dim across 2 warps per row (round 1:
// 5.7 -> 7.2us), which duplicates the whole QK^T.
//
// Precision: the benchmark environment sets TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1,
// so the *baseline*'s matmuls round their inputs to TF32 on tensor cores (its
// own error vs fp64 is ~6.6e-4 relative, i.e. right at the harness's fp32 rtol
// of 1e-3).  An exact-fp32 kernel is therefore *too far from the baseline* to
// pass.  ``TF32`` templates the two data paths: round-to-TF32 inputs with fp32
// accumulate (matches the baseline; selected from torch.backends.cuda.matmul.
// allow_tf32) or plain fp32 (matches a non-TF32 baseline).  The row max and row
// sum are computed over the whole row before normalizing, so the value that
// gets rounded is the normalized probability -- what cuBLAS rounds on its way
// into the P@V matmul -- independently of WPR.
//
// Shared memory holds K and V for the whole sequence of one head, padded to
// PAD=68 floats per row: 68 = 64 + 4 keeps every row float4-aligned while
// making the 8-lane phases of a 128-bit shared load cover all 32 banks exactly
// once (68 % 32 = 4), i.e. conflict-free vector loads and conflict-free
// column-wise float loads in the PV pass.
//
// PDL: the kernel is launched with programmatic stream serialization, so its
// prologue -- grid setup plus an L2 prefetch of the *next* GEMM's weights --
// overlaps the producer QKV GEMM's tail (36 of 148 SMs, so there is room), and
// it waits on cudaGridDependencySynchronize() immediately before its first read
// of qkv.  Measured -2.2us of harness span.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include "clip_tf32.cuh"

#define HD 64             // head_dim of this operator family
#define PAD 68            // shared-memory row pitch (floats)
#define NV (HD / 4)       // float4 per head-dim row
#define MAX_NJ 10         // ceil(S/32) supported => S <= 320
#define STAGE 5           // global->register loads issued per thread per pass

template <int NJ, int ROWS, int RPW, bool TF32, bool PDL>
__global__ __launch_bounds__((ROWS / RPW) * 32) void clip_fused_attn_kernel(
    const float* __restrict__ qkv,
    const float* __restrict__ mask,
    float* __restrict__ out,
    const char* __restrict__ pref, long pref_lines, int pref_mode,
    int S, int NH,
    long mask_bs, long mask_hs, long mask_rs, long mask_es) {
  constexpr int WARPS = ROWS / RPW;
  constexpr int NT = WARPS * 32;
  extern __shared__ float smem[];
  float* __restrict__ ksh = smem;                     // [S][PAD]
  float* __restrict__ vsh = smem + (long)S * PAD;     // [S][PAD]
  float* __restrict__ qsh = vsh + (long)S * PAD;      // [ROWS][PAD]

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int row0 = blockIdx.x * ROWS;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const int E = NH * HD;
  const int stride = 3 * E;

  // Prologue, deliberately before the grid-dependency wait: pull the next GEMM's
  // weights into L2 while the producer is still draining.
  if (pref != nullptr) {
    const long nblk = (long)gridDim.x * gridDim.y * gridDim.z;
    const long bid = (long)blockIdx.x + gridDim.x * (blockIdx.y + (long)gridDim.y * blockIdx.z);
    if (pref_mode == 1) {
      for (long i = bid * NT + tid; i < pref_lines; i += nblk * NT)
        asm volatile("prefetch.global.L2 [%0];" :: "l"(pref + (i << 7)));
    } else {
      float acc = 0.f;
      for (long i = bid * NT + tid; i < pref_lines; i += nblk * NT) {
        const float4 v = *(const float4*)(pref + (i << 4));
        acc += v.x + v.y + v.z + v.w;
      }
      if (acc == 12345.678f) out[0] = acc;
    }
  }
#if __CUDA_ARCH__ >= 900
  if (PDL) cudaGridDependencySynchronize();   // first touch of qkv is below
#endif

  const float* __restrict__ base = qkv + ((long)b * S) * stride + h * HD;

  // Q rows of this block, one float4 per slot.
  float4 qreg[(ROWS * NV + NT - 1) / NT];
#pragma unroll
  for (int u = 0; u < (ROWS * NV + NT - 1) / NT; ++u) {
    const int idx = tid + u * NT;
    if (idx < ROWS * NV && row0 + idx / NV < S)
      qreg[u] = *((const float4*)(base + (long)(row0 + idx / NV) * stride) + idx % NV);
  }

  // K and V for the whole sequence of this head.
  const int total = S * NV;
  for (int pass = 0; pass < total; pass += STAGE * NT) {
    float4 kr[STAGE], vr[STAGE];
#pragma unroll
    for (int u = 0; u < STAGE; ++u) {
      const int idx = pass + tid + u * NT;
      if (idx < total) {
        const float* r = base + (long)(idx / NV) * stride;
        const int c = idx % NV;
        kr[u] = *((const float4*)(r + E) + c);
        vr[u] = *((const float4*)(r + 2 * E) + c);
      }
    }
#pragma unroll
    for (int u = 0; u < STAGE; ++u) {
      const int idx = pass + tid + u * NT;
      if (idx < total) {
        const int j = idx / NV, c = idx % NV;
        *((float4*)(ksh + (long)j * PAD) + c) = round4<TF32>(kr[u]);
        *((float4*)(vsh + (long)j * PAD) + c) = round4<TF32>(vr[u]);
      }
    }
  }
#pragma unroll
  for (int u = 0; u < (ROWS * NV + NT - 1) / NT; ++u) {
    const int idx = tid + u * NT;
    if (idx < ROWS * NV && row0 + idx / NV < S)
      *((float4*)(qsh + (long)(idx / NV) * PAD) + idx % NV) = round4<TF32>(qreg[u]);
  }
  __syncthreads();

  // This warp's RPW query rows.  Rows past S are computed anyway (clamped to
  // row 0, which is always valid) and simply not stored, so the whole warp stays
  // on one control path and the shuffles below keep a full mask.
  int row[RPW];
  bool live[RPW];
#pragma unroll
  for (int r = 0; r < RPW; ++r) {
    const int g = row0 + warp * RPW + r;
    live[r] = g < S;
    row[r] = live[r] ? g : 0;
  }

  // ---- scores: this lane owns keys j = lane, lane+32, ... ------------------
  const float4* q4[RPW];
#pragma unroll
  for (int r = 0; r < RPW; ++r)
    q4[r] = (const float4*)(qsh + (long)(row[r] - row0) * PAD);
  const float4* k4[NJ];
  bool valid[NJ];
#pragma unroll
  for (int jj = 0; jj < NJ; ++jj) {
    const int j = (jj << 5) + lane;
    valid[jj] = j < S;
    // Out-of-range lanes read row 0 and are discarded below, so the inner loop
    // stays branch-free.
    k4[jj] = (const float4*)(ksh + (long)(valid[jj] ? j : 0) * PAD);
  }
  float acc[RPW][NJ][2];
#pragma unroll
  for (int r = 0; r < RPW; ++r)
#pragma unroll
    for (int jj = 0; jj < NJ; ++jj) { acc[r][jj][0] = 0.f; acc[r][jj][1] = 0.f; }
  // One K float4 read serves all RPW rows -- this reuse is the whole point of RPW.
#pragma unroll
  for (int c = 0; c < NV; ++c) {
    float4 a[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r) a[r] = q4[r][c];
#pragma unroll
    for (int jj = 0; jj < NJ; ++jj) {
      const float4 k = k4[jj][c];
#pragma unroll
      for (int r = 0; r < RPW; ++r) {
        acc[r][jj][0] = fmaf(a[r].x, k.x, fmaf(a[r].y, k.y, acc[r][jj][0]));
        acc[r][jj][1] = fmaf(a[r].z, k.z, fmaf(a[r].w, k.w, acc[r][jj][1]));
      }
    }
  }
  float s[RPW][NJ];
#pragma unroll
  for (int r = 0; r < RPW; ++r)
#pragma unroll
    for (int jj = 0; jj < NJ; ++jj)
      s[r][jj] = valid[jj] ? (acc[r][jj][0] + acc[r][jj][1]) : -INFINITY;

  // ---- additive mask ----------------------------------------------------
  if (mask != nullptr) {
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      const float* mrow = mask + (long)b * mask_bs + (long)h * mask_hs +
                          (long)row[r] * mask_rs;
#pragma unroll
      for (int jj = 0; jj < NJ; ++jj) {
        const int j = (jj << 5) + lane;
        if (j < S) s[r][jj] += mrow[(long)j * mask_es];
      }
    }
  }

  // ---- single-pass row softmax (warp-wide max / sum) ---------------------
#pragma unroll
  for (int r = 0; r < RPW; ++r) {
    float m = -INFINITY;
#pragma unroll
    for (int jj = 0; jj < NJ; ++jj) m = fmaxf(m, s[r][jj]);
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float sum = 0.f;
#pragma unroll
    for (int jj = 0; jj < NJ; ++jj) {
      s[r][jj] = expf(s[r][jj] - m);
      sum += s[r][jj];
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    // Normalize *before* rounding: the baseline rounds the normalized softmax
    // output on its way into the P@V tensor-core matmul.
    const float inv = 1.f / sum;
#pragma unroll
    for (int jj = 0; jj < NJ; ++jj) {
      const float pn = s[r][jj] * inv;
      s[r][jj] = TF32 ? to_tf32(pn) : pn;
    }
  }

  // ---- P @ V: this lane owns the dim pair (2*lane, 2*lane+1) --------------
  // One V float2 read serves all RPW rows; two accumulator pairs per row keep
  // the FMA chain 2-way.
  const float2* v2 = (const float2*)(vsh) + lane;      // stride PAD/2 float2
  float o0[RPW], o1[RPW], o2[RPW], o3[RPW];
#pragma unroll
  for (int r = 0; r < RPW; ++r) { o0[r] = 0.f; o1[r] = 0.f; o2[r] = 0.f; o3[r] = 0.f; }
#pragma unroll
  for (int jj = 0; jj < NJ; ++jj) {
    const int jbase = jj << 5;
    const int n = min(32, S - jbase);
    if (n == 32) {
#pragma unroll
      for (int src = 0; src < 32; src += 2) {
        const float2 a = v2[(long)(jbase + src) * (PAD / 2)];
        const float2 bb = v2[(long)(jbase + src + 1) * (PAD / 2)];
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
          const float p0 = __shfl_sync(0xffffffffu, s[r][jj], src);
          const float p1 = __shfl_sync(0xffffffffu, s[r][jj], src + 1);
          o0[r] = fmaf(p0, a.x, o0[r]);
          o1[r] = fmaf(p0, a.y, o1[r]);
          o2[r] = fmaf(p1, bb.x, o2[r]);
          o3[r] = fmaf(p1, bb.y, o3[r]);
        }
      }
    } else {
      for (int src = 0; src < n; ++src) {
        const float2 a = v2[(long)(jbase + src) * (PAD / 2)];
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
          const float p0 = __shfl_sync(0xffffffffu, s[r][jj], src);
          o0[r] = fmaf(p0, a.x, o0[r]);
          o1[r] = fmaf(p0, a.y, o1[r]);
        }
      }
    }
  }

#pragma unroll
  for (int r = 0; r < RPW; ++r)
    if (live[r])
      *((float2*)(out + ((long)b * S + row[r]) * E + h * HD) + lane) =
          make_float2(o0[r] + o2[r], o1[r] + o3[r]);
}

template <int NJ, int ROWS, int RPW, bool TF32, bool PDL>
static void launch_attn(const float* qkv, const float* mask, float* out,
                        const char* pref, long pref_lines, int pref_mode,
                        int B, int S, int NH, size_t smem, long mbs, long mhs,
                        long mrs, long mes, cudaStream_t stream) {
  auto kern = clip_fused_attn_kernel<NJ, ROWS, RPW, TF32, PDL>;
  static size_t configured = 0;
  if (smem > configured) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    configured = smem;
  }
  const dim3 grid((S + ROWS - 1) / ROWS, NH, B);
  const int nthr = (ROWS / RPW) * 32;
  bool pdl_ok = false;
  if (PDL) {
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid; cfg.blockDim = dim3(nthr, 1, 1);
    cfg.dynamicSmemBytes = smem; cfg.stream = stream;
    cfg.attrs = &attr; cfg.numAttrs = 1;
    pdl_ok = cudaLaunchKernelEx(&cfg, kern, qkv, mask, out, pref, pref_lines,
                                pref_mode, S, NH, mbs, mhs, mrs, mes) == cudaSuccess;
  }
  if (!pdl_ok)
    kern<<<grid, nthr, smem, stream>>>(qkv, mask, out, pref, pref_lines,
                                       pref_mode, S, NH, mbs, mhs, mrs, mes);
}

// qkv: [B, S, 3E] contiguous.  Returns attention output [B, S, E].
at::Tensor clip_fused_attn(const at::Tensor& qkv,
                           const std::optional<at::Tensor>& mask_opt,
                           int64_t num_heads, bool tf32,
                           const std::optional<at::Tensor>& prefetch,
                           int64_t pref_mode, bool pdl, int64_t variant) {
  TORCH_CHECK(qkv.dim() == 3 && qkv.is_contiguous() && qkv.scalar_type() == at::kFloat,
              "qkv must be a contiguous fp32 [B, S, 3E] tensor");
  const int B = (int)qkv.size(0), S = (int)qkv.size(1);
  const int E = (int)(qkv.size(2) / 3), NH = (int)num_heads;
  TORCH_CHECK(E == NH * HD, "head_dim must be 64");
  TORCH_CHECK(S <= 32 * MAX_NJ, "seq_len too long for the fused path");

  // PDL needs sm_90+; below that the device-side wait compiles out, so never
  // route a launch through it or the kernel would race its producer.
  if (pdl) {
    int major = 0, dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    pdl = major >= 9;
  }

  const float* mask_ptr = nullptr;
  long mbs = 0, mhs = 0, mrs = 0, mes = 0;
  if (mask_opt.has_value()) {
    const at::Tensor& m = *mask_opt;
    TORCH_CHECK(m.dim() == 4 && m.scalar_type() == at::kFloat, "mask must be fp32 4-D");
    TORCH_CHECK(m.size(2) == S && m.size(3) == S, "mask must be [*, *, S, S]");
    TORCH_CHECK(m.size(0) == B || m.size(0) == 1, "bad mask batch dim");
    TORCH_CHECK(m.size(1) == NH || m.size(1) == 1, "bad mask head dim");
    mask_ptr = m.data_ptr<float>();
    mbs = m.size(0) == 1 ? 0 : m.stride(0);
    mhs = m.size(1) == 1 ? 0 : m.stride(1);
    mrs = m.stride(2);
    mes = m.stride(3);
  }

  const char* pref = nullptr;
  long pref_lines = 0;
  if (prefetch.has_value() && prefetch->numel() > 0) {
    pref = (const char*)prefetch->data_ptr();
    pref_lines = (prefetch->numel() * prefetch->element_size()) >> 7;
    if (pref_mode == 2) pref_lines <<= 3;   // real loads walk float4, not 128B lines
  }

  at::Tensor out = at::empty({qkv.size(0), qkv.size(1), (long)E}, qkv.options());
  const float* qp = qkv.data_ptr<float>();
  float* op = out.data_ptr<float>();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#define ARGS qp, mask_ptr, op, pref, pref_lines, (int)pref_mode, B, S, NH, \
             sm, mbs, mhs, mrs, mes, stream
#define LAUNCH(NJ, ROWS, RPW)                                                  \
  do {                                                                         \
    const size_t sm = ((size_t)2 * S + (ROWS)) * PAD * sizeof(float);           \
    if (tf32 && pdl)  launch_attn<NJ, ROWS, RPW, true, true>(ARGS);             \
    else if (tf32)    launch_attn<NJ, ROWS, RPW, true, false>(ARGS);            \
    else if (pdl)     launch_attn<NJ, ROWS, RPW, false, true>(ARGS);            \
    else              launch_attn<NJ, ROWS, RPW, false, false>(ARGS);           \
  } while (0)
#define DISPATCH(ROWS, RPW)                                                    \
  switch ((S + 31) / 32) {                                                      \
    case 1: LAUNCH(1, ROWS, RPW); break;                                        \
    case 2: LAUNCH(2, ROWS, RPW); break;                                        \
    case 3: LAUNCH(3, ROWS, RPW); break;                                        \
    case 4: LAUNCH(4, ROWS, RPW); break;                                        \
    case 5: LAUNCH(5, ROWS, RPW); break;                                        \
    default: LAUNCH(MAX_NJ, ROWS, RPW); break;                                  \
  }
  // (ROWS/RPW)*32 <= 1024 threads, and RPW must divide ROWS.
  switch (variant) {
    case 1:  DISPATCH(8, 1);  break;
    case 2:  DISPATCH(8, 2);  break;
    case 3:  DISPATCH(16, 2); break;
    case 4:  DISPATCH(16, 4); break;
    case 5:  DISPATCH(32, 4); break;
    case 6:  DISPATCH(24, 2); break;
    case 7:  DISPATCH(32, 2); break;
    case 8:  DISPATCH(8, 4);  break;
    case 9:  DISPATCH(16, 1); break;
    case 10: DISPATCH(64, 4); break;
    case 11: DISPATCH(32, 8); break;
    default: DISPATCH(16, 2); break;
  }
#undef DISPATCH
#undef LAUNCH
#undef ARGS
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("clip_fused_attn", &clip_fused_attn, "fused CLIP attention over packed QKV",
        py::arg("qkv"), py::arg("mask"), py::arg("num_heads"), py::arg("tf32"),
        py::arg("prefetch") = std::nullopt, py::arg("pref_mode") = 1,
        py::arg("pdl") = false, py::arg("variant") = 0);
}
