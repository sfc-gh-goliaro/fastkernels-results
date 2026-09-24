// Plan-based cublasLt fp32/tf32 GEMM with a bias epilogue.
//
// The arithmetic here is exactly what F.linear already issues -- same cublasLt
// entry point, same CUBLAS_COMPUTE_32F_FAST_TF32 compute type, same bias
// epilogue.  The only thing this file adds is the ability to *name the
// schedule*: tile, pipeline stages, cluster shape, CTA swizzle and custom
// option are chosen by an offline sweep instead of by cuBLAS' heuristic, and
// every candidate is gated on bitwise equality with the default before it is
// allowed to run.  split-k / stream-k reduction is never configured, because
// reordering the k accumulation is exactly what this operator cannot absorb.
#include <torch/extension.h>
#include <cublasLt.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <map>
#include <vector>

#define LTCHK(x) do { cublasStatus_t s_ = (x); TORCH_CHECK(s_ == CUBLAS_STATUS_SUCCESS, \
  "cublasLt error ", (int)s_, " line ", __LINE__); } while (0)

static cublasLtHandle_t g_lt = nullptr;
static cublasLtHandle_t lt() {
  if (!g_lt) LTCHK(cublasLtCreate(&g_lt));
  return g_lt;
}

struct Plan {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t La = nullptr, Lb = nullptr, Lc = nullptr;
  cublasLtMatmulAlgo_t algo{};
  at::Tensor ws;
  size_t wsBytes = 0;
  int64_t M = 0, N = 0, K = 0, nt = 0;
  bool bias = false;
};
static std::map<int64_t, Plan> g_plans;
static int64_t g_next = 1;

// Build desc/layouts for out[M,N] = x[M,K] @ (nt ? Wt[K,N] : W[N,K]^T) + bias.
static void make_desc(int64_t M, int64_t N, int64_t K, int64_t nt, bool bias,
                      const void* biasPtr, cublasLtMatmulDesc_t* op,
                      cublasLtMatrixLayout_t* La, cublasLtMatrixLayout_t* Lb,
                      cublasLtMatrixLayout_t* Lc) {
  LTCHK(cublasLtMatmulDescCreate(op, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));
  cublasOperation_t ta = nt ? CUBLAS_OP_N : CUBLAS_OP_T, tb = CUBLAS_OP_N;
  LTCHK(cublasLtMatmulDescSetAttribute(*op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  LTCHK(cublasLtMatmulDescSetAttribute(*op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
  if (bias) {
    cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
    LTCHK(cublasLtMatmulDescSetAttribute(*op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
    LTCHK(cublasLtMatmulDescSetAttribute(*op, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                         &biasPtr, sizeof(biasPtr)));
  }
  if (nt) LTCHK(cublasLtMatrixLayoutCreate(La, CUDA_R_32F, N, K, N));
  else    LTCHK(cublasLtMatrixLayoutCreate(La, CUDA_R_32F, K, N, K));
  LTCHK(cublasLtMatrixLayoutCreate(Lb, CUDA_R_32F, K, M, K));
  LTCHK(cublasLtMatrixLayoutCreate(Lc, CUDA_R_32F, N, M, N));
}

static std::vector<double> cfg_row(const cublasLtMatmulAlgo_t& a, float waves, size_t wsn) {
  int32_t id = 0, splitk = 1;
  uint32_t tile = 0, stages = 0, swz = 0, custom = 0, red = 0;
  uint16_t inner = 0, clu = 0;
  size_t w;
  cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_ID, &id, 4, &w);
  cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, 4, &w);
  cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, 4, &w);
  cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, 4, &w);
  cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, 4, &w);
  cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swz, 4, &w);
  cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &custom, 4, &w);
  cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &inner, 2, &w);
  cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &clu, 2, &w);
  return {(double)id, (double)tile, (double)stages, (double)swz, (double)custom,
          (double)inner, (double)clu, (double)splitk, (double)red, (double)waves,
          (double)wsn};
}

// ---- what cuBLAS' own heuristic would pick, in the same row format ----------
std::vector<std::vector<double>> lt_heur(int64_t M, int64_t N, int64_t K, int64_t nt,
                                         int64_t bias, int64_t wsmax, int64_t count) {
  cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t La, Lb, Lc;
  void* dummy = (void*)0x1000;
  make_desc(M, N, K, nt, bias != 0, dummy, &op, &La, &Lb, &Lc);
  cublasLtMatmulPreference_t pref;
  LTCHK(cublasLtMatmulPreferenceCreate(&pref));
  size_t wm = (size_t)wsmax;
  LTCHK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                             &wm, sizeof(wm)));
  std::vector<cublasLtMatmulHeuristicResult_t> res(count);
  int found = 0;
  cublasLtMatmulAlgoGetHeuristic(lt(), op, La, Lb, Lc, Lc, pref, (int)count,
                                 res.data(), &found);
  std::vector<std::vector<double>> rows;
  for (int i = 0; i < found; ++i)
    rows.push_back(cfg_row(res[i].algo, res[i].wavesCount, res[i].workspaceSize));
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(La); cublasLtMatrixLayoutDestroy(Lb);
  cublasLtMatrixLayoutDestroy(Lc); cublasLtMatmulDescDestroy(op);
  return rows;
}

// ---- every schedule the hardware will accept, split-k excluded --------------
std::vector<std::vector<double>> lt_probe(int64_t M, int64_t N, int64_t K, int64_t nt,
                                          int64_t bias, int64_t wsmax, int64_t maxcand) {
  cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t La, Lb, Lc;
  void* dummy = (void*)0x1000;
  make_desc(M, N, K, nt, bias != 0, dummy, &op, &La, &Lb, &Lc);

  int nids = 0;
  std::vector<int> ids(128);
  LTCHK(cublasLtMatmulAlgoGetIds(lt(), CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F, CUDA_R_32F,
                                 CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 128, ids.data(), &nids));
  static const uint16_t CLUSTERS[] = {0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<std::vector<double>> rows;
  for (int ii = 0; ii < nids && (int64_t)rows.size() < maxcand; ++ii) {
    cublasLtMatmulAlgo_t algo;
    if (cublasLtMatmulAlgoInit(lt(), CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F, CUDA_R_32F,
                               CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, ids[ii], &algo)
        != CUBLAS_STATUS_SUCCESS) continue;
    size_t w = 0;
    uint32_t tiles[64] = {0}, stages[64] = {0};
    int ntile = 0, nstage = 0;
    if (cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, tiles,
                                          sizeof(tiles), &w) == CUBLAS_STATUS_SUCCESS && w)
      ntile = (int)(w / 4);
    if (cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_STAGES_IDS, stages,
                                          sizeof(stages), &w) == CUBLAS_STATUS_SUCCESS && w)
      nstage = (int)(w / 4);
    if (!ntile) { tiles[0] = 0; ntile = 1; }
    if (!nstage) { stages[0] = 0; nstage = 1; }
    int swzmax = 0, custmax = 0;
    uint32_t v32 = 0;
    if (cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT,
                                          &v32, 4, &w) == CUBLAS_STATUS_SUCCESS) swzmax = (int)v32;
    if (cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX,
                                          &v32, 4, &w) == CUBLAS_STATUS_SUCCESS)
      custmax = std::min<int>((int)v32, 3);
    for (int t = 0; t < ntile && (int64_t)rows.size() < maxcand; ++t)
    for (int s = 0; s < nstage && (int64_t)rows.size() < maxcand; ++s)
    for (uint16_t clu : CLUSTERS) {
      for (int swz = 0; swz <= swzmax; ++swz)
      for (int cust = 0; cust <= custmax; ++cust) {
        int32_t one = 1; uint32_t none = 0;
        uint32_t utile = tiles[t], ustage = stages[s], uswz = (uint32_t)swz,
                 ucust = (uint32_t)cust;
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &utile, 4);
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &ustage, 4);
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &one, 4);
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &none, 4);
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &uswz, 4);
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &ucust, 4);
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &clu, 2);
        cublasLtMatmulHeuristicResult_t hr{};
        if (cublasLtMatmulAlgoCheck(lt(), op, La, Lb, Lc, Lc, &algo, &hr)
            != CUBLAS_STATUS_SUCCESS) continue;
        if (hr.state != CUBLAS_STATUS_SUCCESS) continue;
        if (hr.workspaceSize > (size_t)wsmax) continue;
        rows.push_back(cfg_row(algo, hr.wavesCount, hr.workspaceSize));
      }
    }
  }
  cublasLtMatrixLayoutDestroy(La); cublasLtMatrixLayoutDestroy(Lb);
  cublasLtMatrixLayoutDestroy(Lc); cublasLtMatmulDescDestroy(op);
  return rows;
}

// ---- bind one named schedule into a reusable, graph-capturable plan ---------
int64_t lt_make(int64_t M, int64_t N, int64_t K, int64_t nt, at::Tensor biasT,
                int64_t id, int64_t tile, int64_t stages, int64_t swz, int64_t cust,
                int64_t inner, int64_t clu, int64_t wsBytes) {
  Plan p;
  p.M = M; p.N = N; p.K = K; p.nt = nt; p.bias = biasT.numel() > 0;
  make_desc(M, N, K, nt, p.bias, p.bias ? biasT.data_ptr() : nullptr,
            &p.op, &p.La, &p.Lb, &p.Lc);
  LTCHK(cublasLtMatmulAlgoInit(lt(), CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F, CUDA_R_32F,
                               CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, (int)id, &p.algo));
  int32_t one = 1;
  uint32_t utile = (uint32_t)tile, ustage = (uint32_t)stages, uswz = (uint32_t)swz,
           ucust = (uint32_t)cust, none = 0;
  uint16_t uclu = (uint16_t)clu, uinner = (uint16_t)inner;
  cublasLtMatmulAlgoConfigSetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &utile, 4);
  cublasLtMatmulAlgoConfigSetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &ustage, 4);
  cublasLtMatmulAlgoConfigSetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &one, 4);
  cublasLtMatmulAlgoConfigSetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &none, 4);
  cublasLtMatmulAlgoConfigSetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &uswz, 4);
  cublasLtMatmulAlgoConfigSetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &ucust, 4);
  cublasLtMatmulAlgoConfigSetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &uinner, 2);
  cublasLtMatmulAlgoConfigSetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &uclu, 2);
  cublasLtMatmulHeuristicResult_t hr{};
  LTCHK(cublasLtMatmulAlgoCheck(lt(), p.op, p.La, p.Lb, p.Lc, p.Lc, &p.algo, &hr));
  TORCH_CHECK(hr.state == CUBLAS_STATUS_SUCCESS, "algo rejected: ", (int)hr.state);
  p.wsBytes = std::max<size_t>(hr.workspaceSize, (size_t)wsBytes);
  if (p.wsBytes) p.ws = at::empty({(int64_t)p.wsBytes},
                                  at::TensorOptions().dtype(at::kByte).device(at::kCUDA));
  int64_t h = g_next++;
  g_plans[h] = std::move(p);
  return h;
}

static void set_bias(Plan& p, const at::Tensor& biasT) {
  if (!p.bias) return;
  const void* bp = biasT.data_ptr();
  LTCHK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                       &bp, sizeof(bp)));
}

// The bias pointer lives in the descriptor, so it is rebound before every call.
// That is a host-side write: under stream capture each call still becomes its own
// graph node with its own baked-in pointers, which is what lets the 32 blocks
// share one plan.
void lt_run(int64_t h, at::Tensor Wm, at::Tensor x, at::Tensor out, at::Tensor biasT) {
  auto it = g_plans.find(h);
  TORCH_CHECK(it != g_plans.end(), "bad plan handle");
  Plan& p = it->second;
  TORCH_CHECK(x.size(0) == p.M && x.size(1) == p.K && out.size(0) == p.M
              && out.size(1) == p.N, "plan shape mismatch");
  set_bias(p, biasT);
  float one = 1.0f, zero = 0.0f;
  LTCHK(cublasLtMatmul(lt(), p.op, &one, Wm.data_ptr(), p.La, x.data_ptr(), p.Lb,
                       &zero, out.data_ptr(), p.Lc, out.data_ptr(), p.Lc, &p.algo,
                       p.wsBytes ? p.ws.data_ptr() : nullptr, p.wsBytes,
                       at::cuda::getCurrentCUDAStream()));
}

// Time a plan the way the shipped kernel will run it: inside a captured graph,
// on a private stream, with no host work in the loop.
double lt_time(int64_t h, at::Tensor Wm, at::Tensor x, at::Tensor out,
               at::Tensor biasT, int64_t inner, int64_t reps) {
  auto it = g_plans.find(h);
  TORCH_CHECK(it != g_plans.end(), "bad plan handle");
  Plan& p = it->second;
  set_bias(p, biasT);
  cudaStream_t s;
  cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  float one = 1.0f, zero = 0.0f;
  auto call = [&]() {
    cublasLtMatmul(lt(), p.op, &one, Wm.data_ptr(), p.La, x.data_ptr(), p.Lb, &zero,
                   out.data_ptr(), p.Lc, out.data_ptr(), p.Lc, &p.algo,
                   p.wsBytes ? p.ws.data_ptr() : nullptr, p.wsBytes, s);
  };
  for (int i = 0; i < 3; ++i) call();
  cudaStreamSynchronize(s);
  cudaGraph_t g; cudaGraphExec_t ge;
  cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed);
  for (int i = 0; i < inner; ++i) call();
  if (cudaStreamEndCapture(s, &g) != cudaSuccess) { cudaStreamDestroy(s); return -1.0; }
  if (cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0) != cudaSuccess) {
    cudaGraphDestroy(g); cudaStreamDestroy(s); return -1.0;
  }
  cudaGraphLaunch(ge, s); cudaStreamSynchronize(s);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    cudaEventRecord(e0, s);
    cudaGraphLaunch(ge, s);
    cudaEventRecord(e1, s);
    cudaStreamSynchronize(s);
    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
    best = std::min(best, (double)ms * 1000.0 / (double)inner);
  }
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  cudaGraphExecDestroy(ge); cudaGraphDestroy(g); cudaStreamDestroy(s);
  return best;
}

void lt_release(int64_t h) { g_plans.erase(h); }
int64_t lt_nplans() { return (int64_t)g_plans.size(); }
