// CUTLASS 4.x Sm100 (Blackwell) dense-GEMM instantiations for the FLUX FFN.
//
// GEMM1  x[M,3072] @ W1[12288,3072]^T -> h[M,12288],  epilogue: + bias[N], tanh-GELU
// GEMM2  h[M,12288] @ W2[3072,12288]^T -> y[M,3072],  epilogue: + bias[N]
//
// Both are the native TN layout for tcgen05: A row-major [M,K], B column-major
// [K,N] (which is exactly how torch stores a Linear weight, [N,K] row-major),
// D row-major [M,N].  bf16 in / bf16 out with fp32 accumulation.
//
// A 256-wide MMA tile with an even cluster-M makes the CollectiveBuilder select
// the 2-SM UMMA atom, i.e. tcgen05.mma.cta_group::2 plus the cta_group::2 TMA
// load.  See ITERATIONS.md for the PTX evidence.
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "ako_knobs.h"

namespace ako {

using namespace cute;

using EA = cutlass::bfloat16_t;
using LA = cutlass::layout::RowMajor;
using EB = cutlass::bfloat16_t;
using LB = cutlass::layout::ColumnMajor;
using ED = cutlass::bfloat16_t;
using LD = cutlass::layout::RowMajor;
using EAcc = float;

static constexpr int AlignA = 128 / cutlass::sizeof_bits<EA>::value;  // 8
static constexpr int AlignB = 128 / cutlass::sizeof_bits<EB>::value;  // 8
static constexpr int AlignD = 128 / cutlass::sizeof_bits<ED>::value;  // 8

// D = tanh-GELU(acc + bias[n]) -- GELU_taylor is the tanh approximation,
// 0.5*x*(1+tanh(0.7978845608*(x + 0.044715 x^3))), evaluated in fp32.
using FusionBiasGelu = cutlass::epilogue::fusion::LinCombPerColBiasEltAct<
    cutlass::epilogue::thread::GELU_taylor, ED, EAcc, EB, void, EAcc, AlignD>;

// D = acc + bias[n]
using FusionBias = cutlass::epilogue::fusion::LinCombPerColBias<
    ED, EAcc, EB, void, EAcc, AlignD>;

using Sched2Sm = cutlass::gemm::KernelTmaWarpSpecialized2SmSm100;
using TileSchedCLC = void;                              // dynamic persistent (CLC)
using TileSchedStreamK = cutlass::gemm::StreamKScheduler;

template <class MmaTileShape, class ClusterShape, class Fusion, class TileSched>
struct Cfg {
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      EAcc, EAcc,
      void, LD, AlignD,
      ED, LD, AlignD,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      Fusion>::CollectiveOp;

  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      EA, LA, AlignA,
      EB, LB, AlignB,
      EAcc,
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      Sched2Sm>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, Mainloop, Epilogue, TileSched>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// The Stream-K scheduler's Arguments carry `splits`; the default CLC one does not.
template <class T, class = void>
struct has_splits : std::false_type {};
template <class T>
struct has_splits<T, std::void_t<decltype(T::splits)>> : std::true_type {};

template <class Gemm>
typename Gemm::Arguments make_args(const void* A, const void* B, const void* bias, void* D,
                                   int M, int N, int K, Knobs const& kn) {
  using Kern = typename Gemm::GemmKernel;
  auto sA = cutlass::make_cute_packed_stride(typename Kern::StrideA{}, cute::make_shape(M, K, 1));
  auto sB = cutlass::make_cute_packed_stride(typename Kern::StrideB{}, cute::make_shape(N, K, 1));
  auto sD = cutlass::make_cute_packed_stride(typename Kern::StrideD{}, cute::make_shape(M, N, 1));
  auto sC = cutlass::make_cute_packed_stride(typename Kern::StrideC{}, cute::make_shape(M, N, 1));

  typename Gemm::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape = {M, N, K, 1};
  args.mainloop = {static_cast<const EA*>(A), sA, static_cast<const EB*>(B), sB};
  args.epilogue.thread.alpha = 1.0f;
  args.epilogue.thread.beta = 0.0f;
  args.epilogue.thread.bias_ptr = static_cast<const EB*>(bias);
  args.epilogue.ptr_C = nullptr;
  args.epilogue.dC = sC;
  args.epilogue.ptr_D = static_cast<ED*>(D);
  args.epilogue.dD = sD;

  using RO = cutlass::gemm::kernel::detail::RasterOrderOptions;
  args.scheduler.max_swizzle_size = kn.swizzle;
  args.scheduler.raster_order = (kn.raster == 1) ? RO::AlongM
                              : (kn.raster == 2) ? RO::AlongN
                                                 : RO::Heuristic;

  using SchedArgs = std::decay_t<decltype(args.scheduler)>;
  if constexpr (has_splits<SchedArgs>::value) {
    using SKP = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams;
    args.scheduler.splits = kn.splits;
    args.scheduler.decomposition_mode =
        (kn.decomp == 1) ? SKP::DecompositionMode::SplitK
      : (kn.decomp == 2) ? SKP::DecompositionMode::StreamK
      : (kn.decomp == 3) ? SKP::DecompositionMode::DataParallel
                         : SKP::DecompositionMode::Heuristic;
    // Deterministic keeps run-to-run bit-reproducibility of the fp32 fixup.
    args.scheduler.reduction_mode = SKP::ReductionMode::Deterministic;
  }
  return args;
}

template <class C>
int launch(const void* A, const void* B, const void* bias, void* D,
           int M, int N, int K, void* ws, cudaStream_t stream, Knobs const& kn) {
  using Gemm = typename C::Gemm;
  auto args = make_args<Gemm>(A, B, bias, D, M, N, K, kn);
  Gemm op;
  return static_cast<int>(op.run(args, ws, stream));
}

template <class C>
size_t ws_bytes(int M, int N, int K, Knobs const& kn) {
  using Gemm = typename C::Gemm;
  const void* p = reinterpret_cast<const void*>(0x100);
  auto args = make_args<Gemm>(p, p, p, const_cast<void*>(p), M, N, K, kn);
  return Gemm::get_workspace_size(args);
}

}  // namespace ako

// Emit the extern "C" entry points for one config.  NAME must be unique; MMA and
// CLUSTER are type aliases (they contain commas, so they cannot be macro args).
#define AKO_DEFINE(NAME, MMA, CLUSTER, FUSION, TILESCHED)                            \
  namespace ako {                                                                     \
  using Cfg_##NAME = Cfg<MMA, CLUSTER, FUSION, TILESCHED>;                            \
  }                                                                                   \
  extern "C" int ako_run_##NAME(const void* A, const void* B, const void* bias,        \
                               void* D, int M, int N, int K, void* ws,                 \
                               cudaStream_t stream, const ako::Knobs* kn) {             \
    return ako::launch<ako::Cfg_##NAME>(A, B, bias, D, M, N, K, ws, stream, *kn);        \
  }                                                                                    \
  extern "C" size_t ako_ws_##NAME(int M, int N, int K, const ako::Knobs* kn) {           \
    return ako::ws_bytes<ako::Cfg_##NAME>(M, N, K, *kn);                                \
  }
