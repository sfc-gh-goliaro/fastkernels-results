/*
 * A state-caching launcher for trtllm-gen's MXFP4 fused MoE.
 *
 * The compute is trtllm-gen's, byte for byte: this file calls the same
 * ``Routing::Runner::run`` and ``MoE::Runner::run`` with the same arguments as
 * flashinfer's ``trtllm_fused_moe_kernel_launcher.cu``. What changes is *setup*.
 * flashinfer's launcher rebuilds, on every single invocation:
 *
 *   * four ``FP4BlockScaleLauncher`` objects (one per supported tile_N) plus
 *     four ``MoERunnerArgs``, of which three are thrown away;
 *   * a ``MoE::Runner`` -- which constructs two ``TrtllmGenBatchedGemmRunner``
 *     objects that each enumerate and filter the whole trtllm-gen cubin table,
 *     then forms the cartesian product of the surviving gemm1 x gemm2 configs;
 *   * ``getValidConfigIndices`` and a linear search through it;
 *   * ``getWorkspaceSizeInBytes``;
 *   * ~15 device allocations for routing scratch, histograms, CTA index maps,
 *     the gemm1/gemm2 outputs and the two bmm workspaces.
 *
 * Measured at one token that is 0.33 ms of host time against 21 us of GPU work,
 * and ~0.26 ms of it is the Runner construction alone (four of them, via
 * ``trtllm_get_valid_moe_configs``, cost 1.05 ms). None of it depends on
 * anything that changes between calls with the same shape, so all of it is
 * hoisted into a process-static plan keyed by the invariants. What is left per
 * call is: hash a small key, write ~14 pointers into the cached args struct,
 * and make the two launches.
 *
 * Specialised to the one case this kernel needs -- bfloat16 activations x
 * MXFP4 (MxE2m1 + E8M0) weights, SwiGLU, shuffled MajorK weights, routing from
 * logits, do_finalize, no per-token / per-channel scaling, no LoRA bias, no
 * routing replay -- so every branch flashinfer's launcher takes at run time is
 * resolved at compile time here.
 *
 * Buffer reuse is safe: flashinfer's launcher never zeroes any of these buffers
 * either, and torch's caching allocator hands the same (dirty) blocks back on
 * every iteration, so a persistent buffer is exactly as initialised as a freshly
 * "allocated" one. The two places that would care check out: the routing kernel
 * zeroes ``mPtrExpertCounts`` itself (``launchInitExpertCounts`` runs before the
 * atomicAdd in both the coop and the multi-kernel path), and the bmm workspace
 * is only non-empty for DeepSeek-FP8 fused activation (not this path) or a
 * 4-byte PersistentSm90 tile counter that ``BatchedGemmInterface::run``
 * re-initialises from pinned host memory on every call.
 */
#include <cuda_runtime.h>
#include <flashinfer/exception.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/trtllm/gen/DtypeDecl.h"
#include "flashinfer/trtllm/fused_moe/RoutingKernel.h"
#include "flashinfer/trtllm/fused_moe/runner.h"
#include "tvm_ffi_utils.h"

namespace fastkernels {

namespace btg = batchedGemm::trtllm::gen;
namespace tgmoe = tensorrt_llm::kernels::trtllmgen_moe;

using tvm::ffi::Tensor;
using tvm::ffi::TensorView;

// Everything this launcher is specialised for.
constexpr btg::Dtype kDtypeAct = btg::Dtype::Bfloat16;
constexpr btg::Dtype kDtypeWeights = btg::Dtype::MxE2m1;
constexpr btg::Dtype kDtypeOut = btg::Dtype::Bfloat16;
constexpr tgmoe::MoE::ActivationType kActivation = tgmoe::MoE::ActivationType::Swiglu;
constexpr batchedGemm::gemm::MatrixLayout kWeightLayout = batchedGemm::gemm::MatrixLayout::MajorK;
constexpr batchedGemm::gemm::BiasType kBiasType = batchedGemm::gemm::BiasType::None;
// FP4BlockScaleLauncher::mBaseSupportedTileNums for a bfloat16 activation.
constexpr int32_t kSupportedTileNums[] = {8, 16, 32, 64};
constexpr int kNumSupportedTileNums = 4;

// A plan pins two [num_tokens, top_k] scratch tensors plus the gemm1/gemm2
// outputs, so the cache cannot be allowed to grow without bound. Serving sees a
// bounded set of token counts; dropping the whole cache on overflow costs one
// re-plan per surviving token count, which is the same thing flashinfer's
// launcher pays on every call anyway.
constexpr size_t kMaxPlans = 64;

// ---------------------------------------------------------------------------
// Tile selection -- copied from flashinfer's launcher so a cached tactic lands
// on the same tile_N it was profiled for.
// ---------------------------------------------------------------------------
inline int32_t next_power_of_two(float value) {
  int32_t n = static_cast<int32_t>(std::ceil(value));
  if (n <= 1) return 1;
  if ((n & (n - 1)) == 0) return n;
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return n + 1;
}

// ``computeSelectedTileN(...).begin()``, i.e. flashinfer's default tile_N: the
// heuristic centre of the candidate set, which is the smallest element of it.
inline int32_t default_tile_n(int32_t num_tokens, int32_t top_k, int32_t num_local_experts) {
  float const avg_tokens_per_expert =
      static_cast<float>(static_cast<int64_t>(num_tokens) * top_k) / num_local_experts;
  int32_t centre = std::clamp(next_power_of_two(avg_tokens_per_expert),
                              kSupportedTileNums[0], kSupportedTileNums[kNumSupportedTileNums - 1]);
  // The candidate set is {centre-1, centre, centre+1, centre+2} clamped to the
  // ladder, and std::set orders it, so the first element is the predecessor of
  // the centre when one exists.
  for (int i = 0; i < kNumSupportedTileNums; ++i) {
    if (kSupportedTileNums[i] == centre) return i > 0 ? kSupportedTileNums[i - 1] : centre;
  }
  FLASHINFER_CHECK(false, "tile_N ", centre, " is not on the supported ladder");
  return centre;
}

// ---------------------------------------------------------------------------
// The MoE::Runner cache. Construction filters the whole trtllm-gen cubin table
// twice (gemm1 and gemm2) and then builds the cartesian product of the passing
// configs, which is the single most expensive thing flashinfer's launcher redoes
// per call. Keyed by (device, tile_N) because everything else is fixed above.
// ---------------------------------------------------------------------------
tgmoe::MoE::Runner* runner_for(int device, int32_t tile_n) {
  static std::mutex mu;
  static std::vector<std::pair<uint64_t, std::unique_ptr<tgmoe::MoE::Runner>>> cache;
  uint64_t const key = (static_cast<uint64_t>(static_cast<uint32_t>(device)) << 32) |
                       static_cast<uint32_t>(tile_n);
  std::lock_guard<std::mutex> guard(mu);
  for (auto const& entry : cache) {
    if (entry.first == key) return entry.second.get();
  }
  cache.emplace_back(key, std::make_unique<tgmoe::MoE::Runner>(
                              kDtypeAct, kDtypeWeights, /*useDeepSeekFp8=*/false, tile_n,
                              kActivation, /*useShuffledMatrix=*/true, kWeightLayout, kBiasType,
                              /*usePerTokenScalingGemm1=*/false,
                              /*usePerTokenScalingGemm2=*/false));
  return cache.back().second.get();
}

// ---------------------------------------------------------------------------
// One plan per (shape, tactic, device).
// ---------------------------------------------------------------------------
struct PlanKey {
  int32_t num_tokens;
  int32_t num_experts;
  int32_t top_k;
  int32_t hidden_size;
  int32_t hidden_size_output;
  int32_t intermediate_size;
  int32_t routing_method_type;
  int32_t tile_n_in;
  int64_t config_in;
  int32_t device;

  bool operator==(PlanKey const& o) const {
    return num_tokens == o.num_tokens && num_experts == o.num_experts && top_k == o.top_k &&
           hidden_size == o.hidden_size && hidden_size_output == o.hidden_size_output &&
           intermediate_size == o.intermediate_size &&
           routing_method_type == o.routing_method_type && tile_n_in == o.tile_n_in &&
           config_in == o.config_in && device == o.device;
  }
};

struct PlanKeyHash {
  size_t operator()(PlanKey const& k) const {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t v) {
      h = (h ^ v) * 1099511628211ull;
    };
    mix(static_cast<uint32_t>(k.num_tokens));
    mix(static_cast<uint32_t>(k.num_experts));
    mix((static_cast<uint64_t>(static_cast<uint32_t>(k.top_k)) << 32) |
        static_cast<uint32_t>(k.hidden_size));
    mix((static_cast<uint64_t>(static_cast<uint32_t>(k.hidden_size_output)) << 32) |
        static_cast<uint32_t>(k.intermediate_size));
    mix((static_cast<uint64_t>(static_cast<uint32_t>(k.routing_method_type)) << 32) |
        static_cast<uint32_t>(k.tile_n_in));
    mix(static_cast<uint64_t>(k.config_in));
    mix(static_cast<uint32_t>(k.device));
    return static_cast<size_t>(h);
  }
};

struct Plan {
  tgmoe::MoE::Runner* runner{nullptr};
  int32_t tile_n{0};
  int64_t config{-1};
  tgmoe::MoE::MoERunnerArgs args;
  tgmoe::MoE::MoEWorkspace workspace;

  // Persistent scratch. Held so the plan owns the memory the workspace points at.
  Tensor num_tokens_per_expert;
  Tensor total_num_padded_tokens;
  Tensor expanded_idx_to_permuted_idx;
  Tensor permuted_idx_to_token_idx;
  Tensor expert_count_histogram;
  Tensor cta_idx_xy_to_batch_idx;
  Tensor cta_idx_xy_to_mn_limit;
  Tensor num_non_exiting_ctas;
  Tensor topk_ids;
  Tensor topk_weights;
  Tensor gemm1_output;
  Tensor gemm2_output;
  Tensor workspace_fc1;
  Tensor workspace_fc2;
};

std::shared_ptr<Plan> build_plan(PlanKey const& k, DLDevice device) {
  auto plan = std::make_shared<Plan>();

  int64_t tile_n = k.tile_n_in;
  int64_t config = k.config_in;
  if (tile_n == -1 || config == -1) {
    tile_n = default_tile_n(k.num_tokens, k.top_k, k.num_experts);
    config = -1;
  }
  plan->tile_n = static_cast<int32_t>(tile_n);
  plan->runner = runner_for(k.device, plan->tile_n);

  // ---- args: everything that does not depend on a data pointer -------------
  auto& args = plan->args;
  args.activation_type = kActivation;
  args.gemm1_bias_type = kBiasType;
  args.num_tokens = k.num_tokens;
  args.num_experts = k.num_experts;
  args.hidden_size = k.hidden_size;
  args.hidden_size_output = k.hidden_size_output;
  args.top_k = k.top_k;
  args.n_group = 0;
  args.topk_group = 0;
  args.local_expert_offset = 0;
  args.local_num_experts = k.num_experts;
  args.intermediate_size = k.intermediate_size;
  args.routed_scaling_factor = 1.0f;
  args.mDtypeElt = kDtypeAct;
  args.mDtypeOut = kDtypeOut;
  args.mUseRoutingScalesOnInput = false;
  args.mUseDeepSeekFp8 = false;
  args.do_finalize = true;
  args.output_scale = nullptr;

  if (config == -1) {
    config = plan->runner->getDefaultValidConfigIndex(args.top_k, args.hidden_size,
                                                      args.intermediate_size,
                                                      args.local_num_experts, args.num_tokens);
  }
  auto const valid_cfgs =
      plan->runner->getValidConfigIndices(args.top_k, args.hidden_size, args.intermediate_size,
                                          args.local_num_experts, args.num_tokens);
  FLASHINFER_CHECK(std::find(valid_cfgs.begin(), valid_cfgs.end(), config) != valid_cfgs.end(),
                   "Invalid MoE tactic ", config, " for tile_N=", plan->tile_n,
                   ". Number of valid tactics for this tile is ", valid_cfgs.size(), ".");
  plan->config = config;

  // ---- routing scratch (FP4BlockScaleLauncher::prepare_routing) ------------
  int32_t const max_num_padded_tokens = tgmoe::Routing::getMaxPermutedPaddedCount(
      args.num_tokens, args.top_k, args.num_experts, plan->tile_n);
  int32_t const max_num_ctas = tgmoe::Routing::getMaxNumCtasInBatchDim(
      args.num_tokens, args.top_k, args.num_experts, plan->tile_n);
  int64_t const histogram_size = std::max<int64_t>(args.num_experts * 2, 256 * 2);

  plan->num_tokens_per_expert = alloc_tensor({args.num_experts}, dl_int32, device);
  plan->total_num_padded_tokens = alloc_tensor({1}, dl_int32, device);
  plan->expanded_idx_to_permuted_idx =
      alloc_tensor({static_cast<int64_t>(args.num_tokens) * args.top_k}, dl_int32, device);
  plan->permuted_idx_to_token_idx = alloc_tensor({max_num_padded_tokens}, dl_int32, device);
  plan->expert_count_histogram = alloc_tensor({histogram_size}, dl_int32, device);
  plan->cta_idx_xy_to_batch_idx = alloc_tensor({max_num_ctas}, dl_int32, device);
  plan->cta_idx_xy_to_mn_limit = alloc_tensor({max_num_ctas}, dl_int32, device);
  plan->num_non_exiting_ctas = alloc_tensor({1}, dl_int32, device);
  plan->topk_ids = alloc_tensor({args.num_tokens, args.top_k}, dl_int32, device);
  plan->topk_weights = alloc_tensor({args.num_tokens, args.top_k}, dl_bfloat16, device);

  auto& ws = plan->workspace;
  ws.total_num_padded_tokens = static_cast<int32_t*>(plan->total_num_padded_tokens.data_ptr());
  ws.total_max_padded_tokens = max_num_padded_tokens;
  ws.ProjUpTileN = plan->tile_n;
  ws.routing_expert_indexes = static_cast<int32_t*>(plan->topk_ids.data_ptr());
  ws.expert_weights = plan->topk_weights.data_ptr();
  ws.permuted_idx_size = static_cast<int32_t*>(plan->total_num_padded_tokens.data_ptr());
  ws.expanded_idx_to_permuted_idx =
      static_cast<int32_t*>(plan->expanded_idx_to_permuted_idx.data_ptr());
  ws.permuted_idx_to_token_idx =
      static_cast<int32_t*>(plan->permuted_idx_to_token_idx.data_ptr());
  ws.permuted_idx_to_expanded_idx = nullptr;
  ws.cta_idx_xy_to_batch_idx = static_cast<int32_t*>(plan->cta_idx_xy_to_batch_idx.data_ptr());
  ws.cta_idx_xy_to_mn_limit = static_cast<int32_t*>(plan->cta_idx_xy_to_mn_limit.data_ptr());
  ws.num_non_exiting_ctas = static_cast<int32_t*>(plan->num_non_exiting_ctas.data_ptr());

  // ---- moe scratch (FP4BlockScaleLauncher::prepare_moe) -------------------
  auto const workspace_sizes = plan->runner->getWorkspaceSizeInBytes(args, plan->config);
  plan->workspace_fc1 = alloc_tensor({std::get<0>(workspace_sizes)}, dl_int8, device);
  plan->workspace_fc2 = alloc_tensor({std::get<1>(workspace_sizes)}, dl_int8, device);
  ws.bmm1_workspace = plan->workspace_fc1.data_ptr();
  ws.bmm2_workspace = plan->workspace_fc2.data_ptr();

  int32_t const padded_gemm1 = tgmoe::Routing::maybeGetMinTokenCount(
      max_num_padded_tokens, args.intermediate_size, btg::dtypeGetNumBits(kDtypeAct));
  int32_t const padded_gemm2 = tgmoe::Routing::maybeGetMinTokenCount(
      max_num_padded_tokens, args.hidden_size, btg::dtypeGetNumBits(btg::Dtype::Bfloat16));
  plan->gemm1_output =
      alloc_tensor({padded_gemm1, args.intermediate_size}, dl_bfloat16, device);
  plan->gemm2_output = alloc_tensor({padded_gemm2, args.hidden_size}, dl_bfloat16, device);

  ws.hidden_states_scale_linear = nullptr;
  ws.gemm1_output = plan->gemm1_output.data_ptr();
  ws.gemm1_output_scale = nullptr;
  ws.gemm2_output = plan->gemm2_output.data_ptr();
  ws.gemm2_output_scale = nullptr;
  return plan;
}

// Returned by value: a plan is shared, so evicting the cache (or clearing it on
// overflow) cannot pull the scratch out from under a call that is still using it.
//
// Note the scratch inside a plan is *not* re-entrant -- two concurrent calls with
// the same key would share one gemm1/gemm2 buffer. That is the same contract
// vLLM's single MoE-per-layer forward already satisfies, and it is the price of
// not allocating per call; flashinfer's launcher allocates fresh buffers instead.
std::shared_ptr<Plan> plan_for(PlanKey const& key, DLDevice device) {
  static std::mutex mu;
  static std::unordered_map<PlanKey, std::shared_ptr<Plan>, PlanKeyHash> cache;
  std::lock_guard<std::mutex> guard(mu);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  if (cache.size() >= kMaxPlans) cache.clear();
  auto plan = build_plan(key, device);
  cache.emplace(key, plan);
  return plan;
}

// ---------------------------------------------------------------------------
// Entry point.
// ---------------------------------------------------------------------------
void mxfp4_moe(TensorView routing_logits, TensorView hidden_states, TensorView gemm1_weights,
               TensorView gemm1_weights_scale, TensorView gemm1_bias, TensorView gemm1_alpha,
               TensorView gemm1_beta, TensorView gemm1_clamp_limit, TensorView gemm2_weights,
               TensorView gemm2_weights_scale, TensorView gemm2_bias, TensorView output,
               int64_t num_experts, int64_t top_k, int64_t intermediate_size,
               int64_t routing_method_type, int64_t tile_n_in, int64_t config_in,
               bool enable_pdl) {
  TVM_FFI_ICHECK(hidden_states.dtype() == dl_bfloat16)
      << "hidden_states must be bfloat16 for this launcher.";
  TVM_FFI_ICHECK(output.dtype() == dl_bfloat16) << "output must be bfloat16.";
  TVM_FFI_ICHECK(gemm1_weights.dtype() == dl_uint8 && gemm2_weights.dtype() == dl_uint8)
      << "weights must be fp4 packed in uint8.";
  TVM_FFI_ICHECK(gemm1_weights_scale.dtype() == dl_float8_e4m3fn &&
                 gemm2_weights_scale.dtype() == dl_float8_e4m3fn)
      << "weight scales must be fp8_e4m3 (E8M0 bytes viewed as fp8).";
  TVM_FFI_ICHECK(hidden_states.ndim() == 2 && output.ndim() == 2)
      << "hidden_states and output must be 2D.";

  DLDevice const device = hidden_states.device();
  PlanKey key{static_cast<int32_t>(hidden_states.size(0)),
              static_cast<int32_t>(num_experts),
              static_cast<int32_t>(top_k),
              static_cast<int32_t>(hidden_states.size(1)),
              static_cast<int32_t>(output.size(1)),
              static_cast<int32_t>(intermediate_size),
              static_cast<int32_t>(routing_method_type),
              static_cast<int32_t>(tile_n_in),
              config_in,
              device.device_id};
  auto const plan = plan_for(key, device);

  auto& args = plan->args;
  args.routing_logits = routing_logits.data_ptr();
  args.routing_bias = nullptr;
  args.hidden_states = hidden_states.data_ptr();
  args.hidden_states_scale = nullptr;
  args.gemm1_weights = gemm1_weights.data_ptr();
  args.gemm1_weights_scale = gemm1_weights_scale.data_ptr();
  args.gemm1_bias = gemm1_bias.data_ptr();
  args.gemm1_alpha = static_cast<float*>(gemm1_alpha.data_ptr());
  args.gemm1_beta = static_cast<float*>(gemm1_beta.data_ptr());
  args.gemm1_clamp_limit = static_cast<float*>(gemm1_clamp_limit.data_ptr());
  args.gemm2_weights = gemm2_weights.data_ptr();
  args.gemm2_weights_scale = gemm2_weights_scale.data_ptr();
  args.gemm2_bias = static_cast<float*>(gemm2_bias.data_ptr());
  args.output1_scales_scalar = nullptr;
  args.output1_scales_gate_scalar = nullptr;
  args.output2_scales_scalar = nullptr;
  args.output = output.data_ptr();

  cudaStream_t const stream = get_stream(device);
  btg::Dtype const logits_dtype =
      routing_logits.dtype() == dl_float32 ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;

  tgmoe::Routing::Runner routing_runner(plan->tile_n);
  routing_runner.run(
      args.routing_logits, /*routingBias=*/nullptr, args.num_tokens, args.num_experts, args.top_k,
      args.n_group, args.topk_group, args.local_expert_offset, args.local_num_experts,
      args.routed_scaling_factor, static_cast<int32_t*>(plan->topk_ids.data_ptr()),
      static_cast<int32_t*>(plan->expert_count_histogram.data_ptr()),
      static_cast<int32_t*>(plan->total_num_padded_tokens.data_ptr()),
      static_cast<int32_t*>(plan->expanded_idx_to_permuted_idx.data_ptr()),
      /*permutedIdxToExpandedIdx=*/nullptr,
      static_cast<int32_t*>(plan->permuted_idx_to_token_idx.data_ptr()),
      /*expertIds=*/nullptr, plan->topk_weights.data_ptr(),
      static_cast<int32_t*>(plan->num_tokens_per_expert.data_ptr()),
      static_cast<int32_t*>(plan->cta_idx_xy_to_batch_idx.data_ptr()),
      static_cast<int32_t*>(plan->cta_idx_xy_to_mn_limit.data_ptr()),
      static_cast<int32_t*>(plan->num_non_exiting_ctas.data_ptr()), args.mDtypeElt,
      btg::Dtype::Bfloat16, /*useRoutingScalesOnInput=*/false, /*useDeepSeekFp8=*/false,
      static_cast<tgmoe::Routing::RoutingMethodType>(routing_method_type), stream, logits_dtype,
      /*normTopkProb=*/true, /*routing_replay_out=*/nullptr, enable_pdl);

  plan->runner->run(args, plan->workspace, device.device_id, stream, plan->config, enable_pdl);
}

}  // namespace fastkernels

// The trtllm-gen kernels are loaded from cubins at run time through
// ``flashinfer::trtllm_cubin_loader::getCubin``, which flashinfer defines by
// including this header inside that namespace in whichever launcher TU it
// builds. Since we replace that TU, we have to provide the definition (and the
// ``FlashInferSetCubinCallback`` / ``FlashInferSetCurrentCubin`` hooks the
// Python side calls to hand cubins down) ourselves.
namespace flashinfer {
namespace trtllm_cubin_loader {
#include <flashinfer/cubin_loader.h>
}
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fastkernels_mxfp4_moe, fastkernels::mxfp4_moe);
