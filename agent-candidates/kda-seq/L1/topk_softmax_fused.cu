// Fused top-k + softmax routing for Mixture-of-Experts, specialized for the one
// configuration the captured workload actually uses: bf16 logits, 128 experts,
// top_k = 8, renormalize = true.
//
// The reference kernel computes a full-row softmax over all 128 experts, runs k
// rounds of butterfly argmax over the resulting fp32 probabilities, writes each
// winner to global memory from a single lane, and then reloads those k values to
// renormalize them.  Two algebraic facts let this kernel skip most of that work:
//
//   1. With renormalize = true the softmax denominator cancels exactly:
//
//          w_i = (e_i / Z) / sum_{j in topk} (e_j / Z) = e_i / sum_{j in topk} e_j
//
//      with e_i = exp(x_i - max).  So only top_k exponentials are needed, not
//      128, and there is no full-row sum reduction.  Because the max is itself
//      the top-1 winner, the max reduction is free and e_0 is exactly 1.
//
//   2. Selection does not need the softmax at all.  exp is strictly monotonic,
//      so the argmax order over probabilities is the argmax order over the raw
//      bf16 logits, and an order-preserving integer key lets one unsigned
//      integer max do the comparison and the tie-break together.
//
// Where this is *not* bit-equivalent to the reference kernel, and why it does not
// matter for the workload:
//
//   * +/-0 ordering.  The packed key orders +0.0 strictly above -0.0, where IEEE
//     comparison calls them equal, so a row containing both near the top-k cut
//     picks the +0.0 slot where the reference kernel picks the lower expert
//     index.  Enumerating all 65,536 bf16 patterns shows this is the *only* pair
//     in the whole domain where the key order and fp32 order disagree.
//   * Probability collapse.  The reference kernel argmaxes over fp32
//     probabilities; two distinct bf16 logits that exponentiate to the same fp32
//     probability tie there but not here.  That needs a row span beyond ~87.34
//     nats (both underflow) or an eighth-largest logit below about 2^-16 (both
//     round together).  Every such divergent slot carries weight exactly 0.
//   * NaN ordering.  A NaN logit compares false against everything in the
//     reference kernel but has a defined key position here.  A row containing
//     +inf makes the reference kernel emit NaN weights, so neither
//     implementation is usable on such rows.
//
// See ../../docs/divergence.md for the measured margins on the benchmark's input
// distribution (torch.randn bf16 rows of 128), which are four to five orders of
// magnitude away from every one of these conditions.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <climits>
#include <cstdint>

namespace {

constexpr int kWarpSize = 32;
constexpr int kBytesPerLdg = 16;
constexpr int kWarpsPerCta = 4;

// A lane whose slot has been consumed by an earlier selection round is set to
// this.  Reaching it from a real logit needs bit pattern 0xFFFF -- a negative
// NaN -- at the last expert; the smallest key any non-NaN logit can produce is
// 0x007F0000 (-inf at the last expert).  So a blanked slot can never win.
constexpr uint32_t kBlankedKey = 0u;

// Lanes that never own a winner still take part in the butterfly sum, so they
// need a defined key.  This one decodes to +0.0, keeping every intermediate
// finite; their contribution is masked out before the sum regardless.
constexpr uint32_t kIdleLaneKey = 0x80000000u;

// Map a bf16 bit pattern to an order-preserving u16 in the high half, and pack
// NUM_EXPERTS - 1 - expert into the low half, so that a single unsigned max both
// picks the larger logit and breaks ties toward the *lower* expert index -- the
// reference kernel's rule.
//
// `flip` is all ones for a negative pattern and 0x80000000 otherwise, so the xor
// computes the complement of negatives and sets the top bit of positives in one
// step.  Masking to 0xFFFF0000 is not optional: for a negative pattern the
// complement sets all 16 low bits, which would otherwise overwrite the tie field
// and make two different experts holding the same negative logit produce the
// *same* key -- destroying the uniqueness that blanking by key equality relies on.
__device__ __forceinline__ uint32_t packed_key(uint32_t bf16_bits, uint32_t tie) {
  const uint32_t hi = bf16_bits << 16;
  const uint32_t flip =
      static_cast<uint32_t>(static_cast<int32_t>(hi) >> 31) | 0x80000000u;
  return ((hi ^ flip) & 0xFFFF0000u) | tie;
}

// Inverse of the high half, widened to fp32.  bf16 -> fp32 is just 16 appended
// zero bits, so this recovers the original logit exactly.
__device__ __forceinline__ float logit_from_key(uint32_t key) {
  const uint32_t bits =
      (key & 0x80000000u) ? (key & 0x7FFF0000u) : ((~key) & 0xFFFF0000u);
  return __uint_as_float(bits);
}

// How a consumed winner is retired from a lane's local candidates.
enum class Retire {
  // Clear the matching key by equality.  Keys are unique within a row, so this
  // touches exactly one register in exactly one lane and needs no index maths.
  ByKeyEquality = 0,
  // Keep each lane's keys in a descending sorted list, reduce only the heads, and
  // shift the winning lane's list down by one.  Trades a one-off sorting network
  // for cheaper rounds.
  BySortedShift = 1,
};

// Below this the reference's expf underflows to a denormal, so two distinct logits
// can collapse to the same fp32 probability and the raw-key order stops agreeing
// with the reference's probability order.  logf(FLT_MIN).
constexpr float kCollapseThreshold = -87.3365479f;

__device__ __forceinline__ void compare_exchange_desc(uint32_t& a, uint32_t& b) {
  const uint32_t hi = ::max(a, b);
  b = ::min(a, b);
  a = hi;
}

// Batcher odd-even mergesort for eight elements: 19 comparators in 6 layers,
// descending.  Every index is a compile-time constant, so the keys stay in scalar
// registers and nothing is addressed dynamically.
__device__ __forceinline__ void sort8_desc(uint32_t* k) {
  compare_exchange_desc(k[0], k[1]); compare_exchange_desc(k[2], k[3]);
  compare_exchange_desc(k[4], k[5]); compare_exchange_desc(k[6], k[7]);
  compare_exchange_desc(k[0], k[2]); compare_exchange_desc(k[1], k[3]);
  compare_exchange_desc(k[4], k[6]); compare_exchange_desc(k[5], k[7]);
  compare_exchange_desc(k[1], k[2]); compare_exchange_desc(k[5], k[6]);
  compare_exchange_desc(k[0], k[4]); compare_exchange_desc(k[1], k[5]);
  compare_exchange_desc(k[2], k[6]); compare_exchange_desc(k[3], k[7]);
  compare_exchange_desc(k[2], k[4]); compare_exchange_desc(k[3], k[5]);
  compare_exchange_desc(k[1], k[2]); compare_exchange_desc(k[3], k[4]);
  compare_exchange_desc(k[5], k[6]);
}

// Values per lane is kNumExperts / kLanesPerRow, so the caller picks the
// decomposition.  kLanesPerRow must be at least kTopK, because winner k is kept
// only in lane k.
template <int kNumExperts, int kTopK, int kLanesPerRow,
          Retire kRetire = Retire::ByKeyEquality, bool kGuardCollapse = true>
__launch_bounds__(kWarpsPerCta* kWarpSize) __global__
    void topkSoftmaxFusedKernel(const __nv_bfloat16* __restrict__ router_logits,
                                float* __restrict__ topk_weights,
                                int* __restrict__ topk_ids,
                                int num_rows) {
  constexpr int kValuesPerLane = kNumExperts / kLanesPerRow;
  constexpr int kWordsPerLane = kValuesPerLane / 2;  // two bf16 per 32-bit word
  constexpr int kRowsPerWarp = kWarpSize / kLanesPerRow;
  constexpr int kRowsPerCta = kWarpsPerCta * kRowsPerWarp;

  static_assert(kNumExperts % kLanesPerRow == 0, "lanes must divide the row evenly");
  static_assert(kValuesPerLane >= 2 && kValuesPerLane % 2 == 0,
                "a lane must hold an even number of bf16 so loads stay vectorized");
  static_assert(kLanesPerRow > 0 && kLanesPerRow <= kWarpSize, "row must fit in a warp");
  static_assert(kWarpSize % kLanesPerRow == 0, "row groups must tile the warp");
  static_assert((kLanesPerRow & (kLanesPerRow - 1)) == 0, "butterfly needs a power of two");
  static_assert(kTopK <= kLanesPerRow, "lane k carries winner k");
  static_assert(kTopK <= kNumExperts, "cannot select more experts than exist");
  static_assert(kWordsPerLane == 1 || kWordsPerLane == 2 || kWordsPerLane == 4 ||
                    kWordsPerLane % 4 == 0,
                "lane slice must be loadable as uint32 / uint2 / uint4 chunks");
  static_assert(kRetire == Retire::ByKeyEquality || kValuesPerLane == 8,
                "the sorted-shift strategy only has an 8-element network compiled");

  const int lane = static_cast<int>(threadIdx.x);
  const int lane_in_row = lane % kLanesPerRow;
  const int row_in_warp = lane / kLanesPerRow;
  const int row = static_cast<int>(blockIdx.x) * kRowsPerCta +
                  static_cast<int>(threadIdx.y) * kRowsPerWarp + row_in_warp;

  // A row group is a contiguous run of kLanesPerRow lanes and owns exactly one
  // row, so the whole group either stays or returns together.  Scoping every
  // shuffle to the group means no shuffle ever names a lane that has exited --
  // unlike a blanket 0xffffffff after an early return.
  constexpr uint32_t kGroupBits =
      (kLanesPerRow == kWarpSize) ? 0xFFFFFFFFu : ((1u << kLanesPerRow) - 1u);
  const unsigned row_group_mask = kGroupBits << (row_in_warp * kLanesPerRow);

  if (row >= num_rows) {
    return;
  }

  // Vector load of the lane's slice: the widest aligned type that fits, so a warp
  // still covers kRowsPerWarp contiguous rows with fully coalesced traffic.
  const uint32_t* slice = reinterpret_cast<const uint32_t*>(
      router_logits + row * kNumExperts + lane_in_row * kValuesPerLane);
  uint32_t pairs[kWordsPerLane];
  if constexpr (kWordsPerLane % 4 == 0) {
#pragma unroll
    for (int chunk = 0; chunk < kWordsPerLane / 4; ++chunk) {
      const uint4 raw = reinterpret_cast<const uint4*>(slice)[chunk];
      pairs[chunk * 4 + 0] = raw.x;
      pairs[chunk * 4 + 1] = raw.y;
      pairs[chunk * 4 + 2] = raw.z;
      pairs[chunk * 4 + 3] = raw.w;
    }
  } else if constexpr (kWordsPerLane == 2) {
    const uint2 raw = *reinterpret_cast<const uint2*>(slice);
    pairs[0] = raw.x;
    pairs[1] = raw.y;
  } else {
    pairs[0] = slice[0];
  }

  // The tie field is a compile-time offset from a loop-invariant base, so no
  // per-element index arithmetic survives into the selection rounds.
  const uint32_t tie_base =
      static_cast<uint32_t>(kNumExperts - 1 - lane_in_row * kValuesPerLane);
  uint32_t key[kValuesPerLane];
#pragma unroll
  for (int j = 0; j < kValuesPerLane; ++j) {
    const uint32_t bf16_bits =
        (j & 1) ? (pairs[j >> 1] >> 16) : (pairs[j >> 1] & 0xFFFFu);
    key[j] = packed_key(bf16_bits, tie_base - static_cast<uint32_t>(j));
  }

  // Selection.  Every lane ends each round holding the round's winning key, but
  // only lane k keeps winner k -- one live register instead of kTopK.
  //
  // Two ways to retire a consumed winner are compiled.  ByKeyEquality clears the
  // matching register (kValuesPerLane compares plus selects per round, but no
  // set-up cost).  BySortedShift sorts each lane's keys once with a compile-time
  // network, then reduces only the heads and shifts the winning lane's list down.
  uint32_t row_max_key = kBlankedKey;
  uint32_t my_winner_key = kIdleLaneKey;

  if constexpr (kRetire == Retire::BySortedShift) {
    sort8_desc(key);  // descending, so key[0] is this lane's best candidate
  }

#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    uint32_t best;
    if constexpr (kRetire == Retire::BySortedShift) {
      best = key[0];  // the list head is already this lane's maximum
    } else {
      best = key[0];
#pragma unroll
      for (int j = 1; j < kValuesPerLane; ++j) {
        best = ::max(best, key[j]);
      }
    }
#pragma unroll
    for (int offset = kLanesPerRow / 2; offset > 0; offset >>= 1) {
      best = ::max(best,
                   __shfl_xor_sync(row_group_mask, best, offset, kLanesPerRow));
    }
    if (k == 0) {
      row_max_key = best;  // the top-1 winner is the row max, so it is free
    }
    if (lane_in_row == k) {
      my_winner_key = best;
    }
    if (k + 1 < kTopK) {
      if constexpr (kRetire == Retire::BySortedShift) {
        // Exactly one lane holds the winner, and it is that lane's head, so the
        // whole retire is a predicated shift of compile-time-indexed scalars.
        if (key[0] == best) {
#pragma unroll
          for (int j = 0; j < kValuesPerLane - 1; ++j) {
            key[j] = key[j + 1];
          }
          key[kValuesPerLane - 1] = kBlankedKey;
        }
      } else {
        // Keys are unique within a row, so this clears exactly one register in
        // exactly one lane, with no lane or slot index arithmetic.
#pragma unroll
        for (int j = 0; j < kValuesPerLane; ++j) {
          if (key[j] == best) {
            key[j] = kBlankedKey;
          }
        }
      }
    }
  }

  // Weights.  One exponential per lane at kTopK-of-kLanesPerRow occupancy rather
  // than kTopK exponentials on one lane, then a butterfly sum over the group.
  const bool owns_slot = lane_in_row < kTopK;
  const float row_max = logit_from_key(row_max_key);
  float numerator = 0.0f;
  if (owns_slot) {
    numerator = __expf(logit_from_key(my_winner_key) - row_max);
  }
  float denominator = numerator;
#pragma unroll
  for (int offset = kLanesPerRow / 2; offset > 0; offset >>= 1) {
    denominator +=
        __shfl_xor_sync(row_group_mask, denominator, offset, kLanesPerRow);
  }

  float weight = numerator * (1.0f / denominator);
  int winner = static_cast<int>(static_cast<uint32_t>(kNumExperts - 1) -
                               (my_winner_key & 0xFFFFu));

  // Probability collapse.  Selection above ordered the raw logits; the reference
  // orders fp32 probabilities.  Those agree unless two distinct logits exponentiate
  // to the same fp32 value, which needs the kTopK-th selected logit to sit more than
  // ~87.34 nats below the row max.  The predicate is row-group uniform (it is read
  // from the lane owning the last slot and broadcast), so the branch is coherent and
  // the common path pays one shuffle and one compare.
  if constexpr (kGuardCollapse) {
    const float last_selected = __shfl_sync(row_group_mask, numerator, kTopK - 1,
                                            kLanesPerRow);
    // numerator is exp(v - row_max); it underflows exactly when the gap does.
    if (last_selected <= 0.0f ||
        __shfl_sync(row_group_mask, logit_from_key(my_winner_key), kTopK - 1,
                    kLanesPerRow) - row_max <= kCollapseThreshold) {
      // Redo the row the way the reference does: full-row fp32 softmax, then kTopK
      // rounds of argmax over probabilities with lower-index-wins ties.  Same
      // launch, no allocation; the logits are reloaded rather than kept live so the
      // common path's register budget is untouched.
      float probability[kValuesPerLane];
#pragma unroll
      for (int j = 0; j < kValuesPerLane; ++j) {
        const uint32_t bf16_bits =
            (j & 1) ? (pairs[j >> 1] >> 16) : (pairs[j >> 1] & 0xFFFFu);
        probability[j] = __uint_as_float(bf16_bits << 16);
      }
      // The reference's row max is the fp32 max of the row, which is exactly the
      // top-1 logit already in hand.
      const float reference_max = row_max;
#pragma unroll
      for (int j = 0; j < kValuesPerLane; ++j) {
        probability[j] = expf(probability[j] - reference_max);
      }
      // Reproduce the reference's addition order.  It sums a contiguous eight-value
      // slice per lane and then butterflies over 16 lanes with offsets 8, 4, 2, 1.
      // A lane here owns kValuesPerLane / 8 of those slices, so each slice is summed
      // separately, the cross-lane offsets are applied to every sub-accumulator, and
      // the sub-accumulators are folded last -- which is the same tree.
      constexpr int kSlices = kValuesPerLane / 8 > 0 ? kValuesPerLane / 8 : 1;
      constexpr int kSliceWidth = kValuesPerLane / kSlices;
      float slice_sum[kSlices];
#pragma unroll
      for (int s = 0; s < kSlices; ++s) {
        float acc = probability[s * kSliceWidth];
#pragma unroll
        for (int j = 1; j < kSliceWidth; ++j) {
          acc += probability[s * kSliceWidth + j];
        }
        slice_sum[s] = acc;
      }
#pragma unroll
      for (int offset = kLanesPerRow / 2; offset > 0; offset >>= 1) {
#pragma unroll
        for (int s = 0; s < kSlices; ++s) {
          slice_sum[s] += __shfl_xor_sync(row_group_mask, slice_sum[s], offset,
                                          kLanesPerRow);
        }
      }
      float row_sum = slice_sum[0];
#pragma unroll
      for (int s = 1; s < kSlices; ++s) {
        row_sum += slice_sum[s];
      }
      const float reciprocal_row_sum = 1.0f / row_sum;
#pragma unroll
      for (int j = 0; j < kValuesPerLane; ++j) {
        probability[j] = probability[j] * reciprocal_row_sum;
      }

      const int expert_base = lane_in_row * kValuesPerLane;
      float selected_weight = 0.0f;
      int selected_expert = kNumExperts;
      float renormalize_sum = 0.0f;
#pragma unroll
      for (int k = 0; k < kTopK; ++k) {
        // Local argmax with the reference's strict `>`, so the lowest index wins.
        float best_value = probability[0];
        int best_expert = expert_base;
#pragma unroll
        for (int j = 1; j < kValuesPerLane; ++j) {
          if (probability[j] > best_value) {
            best_value = probability[j];
            best_expert = expert_base + j;
          }
        }
        // Butterfly argmax, breaking ties toward the lower expert index.
#pragma unroll
        for (int offset = kLanesPerRow / 2; offset > 0; offset >>= 1) {
          const float other_value =
              __shfl_xor_sync(row_group_mask, best_value, offset, kLanesPerRow);
          const int other_expert =
              __shfl_xor_sync(row_group_mask, best_expert, offset, kLanesPerRow);
          if (other_value > best_value ||
              (other_value == best_value && other_expert < best_expert)) {
            best_value = other_value;
            best_expert = other_expert;
          }
        }
        renormalize_sum += best_value;
        if (lane_in_row == k) {
          selected_weight = best_value;
          selected_expert = best_expert;
        }
        if (k + 1 < kTopK) {
          // Blank the winner wherever it lives.  Unlike the packed key there is no
          // uniqueness guarantee here, so match on the expert index as the
          // reference does.
#pragma unroll
          for (int j = 0; j < kValuesPerLane; ++j) {
            if (expert_base + j == best_expert) {
              probability[j] = -10000.0f;
            }
          }
        }
      }
      weight = selected_weight * (1.0f / renormalize_sum);
      winner = selected_expert;
    }
  }

  // Output.  Lanes 0..kTopK-1 of each row group store their own slot: the row groups
  // of a warp hold adjacent rows, so the active lanes of a warp cover a contiguous
  // run of bytes -- no alignment precondition at all.
  if (owns_slot) {
    const int slot = row * kTopK + lane_in_row;
    topk_weights[slot] = weight;
    topk_ids[slot] = winner;
  }
}

// Values per lane, chosen so that (a) at least kTopK lanes exist, since winner k
// lives in lane k, and (b) a lane's slice is as wide as possible for the load.
// The default prefers a 16 B slice and widens the lane count only when kTopK
// demands it -- E = 32 with kTopK = 8, for instance, must use eight lanes of four
// values (an 8 B load) rather than four lanes of eight.
constexpr int preferred_lanes_per_row(int num_experts, int top_k) {
  int lanes = num_experts / 8;  // 8 bf16 = 16 B
  if (lanes < 1) {
    lanes = 1;
  }
  while (lanes < top_k && lanes < kWarpSize && (num_experts % (lanes * 2)) == 0) {
    lanes *= 2;
  }
  return lanes;
}

// The decomposition that measured fastest, which is not always the structural
// default.  At (128 experts, top_k 8) halving the lane count to eight -- 16 values
// per lane, a 32 B slice, and a *three*-step butterfly instead of four -- measured
// 1.100x faster at M = 16384 and within noise (0.999x-1.006x) at M <= 1000, on
// paired ABBA timing over nine repeats (tests/decomposition.py).  That it wins while
// doing *more* per-lane blanking work is the clearest evidence that this kernel is
// limited by the length of the serial shuffle chain, not by ALU throughput.
constexpr int tuned_lanes_per_row(int num_experts, int top_k) {
  const int structural = preferred_lanes_per_row(num_experts, top_k);
  if (num_experts == 128 && top_k == 8) {
    return structural / 2;  // 8 lanes x 16 values
  }
  return structural;
}

template <int kNumExperts, int kTopK, int kLanesPerRow,
          Retire kRetire = Retire::ByKeyEquality, bool kGuardCollapse = true>
void launch_fused(const __nv_bfloat16* router_logits,
                  float* topk_weights,
                  int* topk_ids,
                  int num_rows,
                  cudaStream_t stream) {
  constexpr int kRowsPerCta = kWarpsPerCta * (kWarpSize / kLanesPerRow);
  const int blocks = (num_rows + kRowsPerCta - 1) / kRowsPerCta;
  topkSoftmaxFusedKernel<kNumExperts, kTopK, kLanesPerRow, kRetire, kGuardCollapse>
      <<<blocks, dim3(kWarpSize, kWarpsPerCta), 0, stream>>>(
          router_logits, topk_weights, topk_ids, num_rows);
}

// One instantiation per (experts, top_k).  `lanes_override` exists so the
// decomposition can be compared by measurement rather than assumed; 0 means take
// the default.
template <int kNumExperts, int kTopK>
bool dispatch_lanes(const __nv_bfloat16* router_logits,
                    float* topk_weights,
                    int* topk_ids,
                    int num_rows,
                    int lanes_override,
                    int retire_override,
                    bool guard_collapse,
                    cudaStream_t stream) {
  // The experimental geometries and strategies are compiled only for the benched
  // configuration; production never asks for them.
  if constexpr (kNumExperts == 128 && kTopK == 8) {
    if (retire_override == static_cast<int>(Retire::BySortedShift)) {
      if (lanes_override == 16) {
        launch_fused<128, 8, 16, Retire::BySortedShift, true>(
            router_logits, topk_weights, topk_ids, num_rows, stream);
        return true;
      }
      return false;
    }
    if (!guard_collapse) {
      if (lanes_override == 8 || lanes_override == 0) {
        launch_fused<128, 8, 8, Retire::ByKeyEquality, false>(
            router_logits, topk_weights, topk_ids, num_rows, stream);
        return true;
      }
      if (lanes_override == 16) {
        launch_fused<128, 8, 16, Retire::ByKeyEquality, false>(
            router_logits, topk_weights, topk_ids, num_rows, stream);
        return true;
      }
      return false;
    }
  } else {
    if (retire_override != static_cast<int>(Retire::ByKeyEquality) || !guard_collapse) {
      return false;
    }
  }
  constexpr int kStructuralLanes = preferred_lanes_per_row(kNumExperts, kTopK);
  constexpr int kTunedLanes = tuned_lanes_per_row(kNumExperts, kTopK);
  const int lanes = lanes_override ? lanes_override : kTunedLanes;
  if (lanes == kStructuralLanes) {
    launch_fused<kNumExperts, kTopK, kStructuralLanes>(
        router_logits, topk_weights, topk_ids, num_rows, stream);
    return true;
  }
  // Half the lanes and twice the values per lane: one fewer butterfly step at the
  // cost of more per-lane scan work.  Only legal while it still leaves kTopK lanes.
  constexpr int kHalfLanes = kStructuralLanes / 2;
  if constexpr (kHalfLanes >= kTopK && kHalfLanes >= 1 &&
                (kNumExperts % kHalfLanes) == 0 &&
                ((kNumExperts / kHalfLanes) % 2) == 0 &&
                (kWarpSize % kHalfLanes) == 0) {
    if (lanes == kHalfLanes) {
      launch_fused<kNumExperts, kTopK, kHalfLanes>(
          router_logits, topk_weights, topk_ids, num_rows, stream);
      return true;
    }
  }
  return false;
}

}  // namespace

// Specialized entry point.  Unlike the reference kernel this takes no softcapping
// scalar and no correction bias -- the Python contract never supplies them -- and
// allocates nothing: the power-of-two path needs no softmax workspace, so the
// reference kernel's unconditional zero-size `torch::empty` is pure host overhead.
//
// `lanes_per_row` is normally 0, meaning "use the default decomposition"; a
// non-zero value selects a compiled alternative so the two can be compared by
// measurement.  It does not change results, only the lane geometry.
static void topk_softmax_fused_impl(torch::Tensor& topk_weights,
                                    torch::Tensor& topk_ids,
                                    torch::Tensor& router_logits,
                                    const bool renormalize,
                                    const int64_t lanes_per_row,
                                    const int64_t retire_strategy,
                                    const bool guard_collapse) {
  TORCH_CHECK(router_logits.is_cuda(), "router_logits must be a CUDA tensor");
  TORCH_CHECK(topk_weights.is_cuda(), "topk_weights must be a CUDA tensor");
  TORCH_CHECK(topk_ids.is_cuda(), "topk_ids must be a CUDA tensor");
  TORCH_CHECK(router_logits.is_contiguous(), "router_logits must be contiguous");
  TORCH_CHECK(topk_weights.is_contiguous(), "topk_weights must be contiguous");
  TORCH_CHECK(topk_ids.is_contiguous(), "topk_ids must be contiguous");
  TORCH_CHECK(router_logits.dim() == 2,
              "router_logits must be 2D [num_tokens, num_experts]");
  TORCH_CHECK(topk_weights.dim() == 2, "topk_weights must be 2D [num_tokens, top_k]");
  TORCH_CHECK(topk_ids.dim() == 2, "topk_ids must be 2D [num_tokens, top_k]");
  TORCH_CHECK(router_logits.scalar_type() == at::ScalarType::BFloat16,
              "this kernel is specialized for bfloat16 router_logits, got ",
              router_logits.scalar_type());
  TORCH_CHECK(topk_weights.scalar_type() == at::ScalarType::Float,
              "topk_weights must be float32");
  TORCH_CHECK(topk_ids.scalar_type() == at::ScalarType::Int,
              "topk_ids must be int32");
  TORCH_CHECK(topk_weights.size(1) == topk_ids.size(1),
              "topk_weights and topk_ids must agree on top_k");
  TORCH_CHECK(topk_weights.size(0) == router_logits.size(0) &&
                  topk_ids.size(0) == router_logits.size(0),
              "output rows must match router_logits rows");
  // The fused form derives the weights from the top_k winners alone, which is
  // only equal to the reference kernel when the denominator cancels.  Without
  // renormalization the full-row sum is needed and the caller must take the
  // reference path instead.
  TORCH_CHECK(renormalize,
              "the fused path requires renormalize=True; route renormalize=False "
              "to the reference implementation");
  // The kernel forms element offsets as `row * num_experts + ...` in int32, so the
  // bound is on the element count, not the row count.  A fixed row cutoff is wrong:
  // at 256 experts it would admit twice as many rows as int32 can address.
  TORCH_CHECK(router_logits.numel() <= static_cast<int64_t>(INT32_MAX),
              "router_logits must hold at most INT32_MAX elements so int32 row "
              "offsets cannot overflow; got ", router_logits.numel());
  TORCH_CHECK(topk_weights.numel() <= static_cast<int64_t>(INT32_MAX),
              "topk_weights must hold at most INT32_MAX elements; got ",
              topk_weights.numel());
  // The device guard below follows router_logits, so the outputs have to live on
  // the same device or the kernel would write across devices.  The Python wrapper
  // guarantees this, but this entry point is callable on its own.
  TORCH_CHECK(topk_weights.get_device() == router_logits.get_device() &&
                  topk_ids.get_device() == router_logits.get_device(),
              "topk_weights, topk_ids and router_logits must be on the same device, got ",
              topk_weights.get_device(), ", ", topk_ids.get_device(), ", ",
              router_logits.get_device());

  const int num_experts = static_cast<int>(router_logits.size(1));
  const int top_k = static_cast<int>(topk_weights.size(1));
  const int num_rows = static_cast<int>(router_logits.size(0));
  const int lanes = static_cast<int>(lanes_per_row);
  const int retire = static_cast<int>(retire_strategy);

  // A lane loads num_experts / lanes contiguous bf16, so that slice's width is the
  // alignment the vector load needs.  Checked against the widest slice any
  // compiled decomposition can use for this expert count.
  const int64_t widest_slice_bytes =
      std::min<int64_t>(kBytesPerLdg, static_cast<int64_t>(num_experts) * 2);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(router_logits.data_ptr()) %
                      static_cast<uintptr_t>(widest_slice_bytes) == 0,
              "router_logits must be ", widest_slice_bytes,
              " B aligned for the vector load");

  if (num_rows == 0) {
    return;
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(router_logits));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const __nv_bfloat16* logits = reinterpret_cast<const __nv_bfloat16*>(
      router_logits.data_ptr<at::BFloat16>());
  float* weights = topk_weights.data_ptr<float>();
  int* ids = topk_ids.data_ptr<int>();

  bool launched = false;
#define TOPK_SOFTMAX_DISPATCH(EXPERTS, K)                                       \
  if (num_experts == (EXPERTS) && top_k == (K)) {                              \
    launched = dispatch_lanes<(EXPERTS), (K)>(logits, weights, ids, num_rows,   \
                                              lanes, retire, guard_collapse,   \
                                              stream);                         \
  }
#define TOPK_SOFTMAX_DISPATCH_K(EXPERTS)   \
  TOPK_SOFTMAX_DISPATCH(EXPERTS, 1)        \
  TOPK_SOFTMAX_DISPATCH(EXPERTS, 2)        \
  TOPK_SOFTMAX_DISPATCH(EXPERTS, 4)        \
  TOPK_SOFTMAX_DISPATCH(EXPERTS, 8)
  TOPK_SOFTMAX_DISPATCH_K(32)
  TOPK_SOFTMAX_DISPATCH_K(64)
  TOPK_SOFTMAX_DISPATCH_K(128)
  TOPK_SOFTMAX_DISPATCH_K(256)
#undef TOPK_SOFTMAX_DISPATCH_K
#undef TOPK_SOFTMAX_DISPATCH

  TORCH_CHECK(launched,
              "no fused instantiation for num_experts=", num_experts,
              ", top_k=", top_k, ", lanes_per_row=", lanes,
              ", retire=", retire, ", guard_collapse=", guard_collapse,
              "; the caller must take the reference path");
}

// Production entry point.  Four arguments, matching the contract; the accepted
// decomposition and retire strategy are chosen internally and the collapse guard is
// always on.
void topk_softmax_fused(torch::Tensor& topk_weights,
                        torch::Tensor& topk_ids,
                        torch::Tensor& router_logits,
                        const bool renormalize) {
  topk_softmax_fused_impl(topk_weights, topk_ids, router_logits, renormalize,
                          /*lanes_per_row=*/0,
                          /*retire_strategy=*/static_cast<int64_t>(Retire::ByKeyEquality),
                          /*guard_collapse=*/true);
}

// Test-only entry point.  Selects a specific lane geometry, retire strategy, and
// collapse-guard setting so the alternatives can be compared and the unguarded
// behaviour characterized.  Not called on any production path.
void topk_softmax_fused_variant(torch::Tensor& topk_weights,
                                torch::Tensor& topk_ids,
                                torch::Tensor& router_logits,
                                const bool renormalize,
                                const int64_t lanes_per_row,
                                const int64_t retire_strategy,
                                const bool guard_collapse) {
  topk_softmax_fused_impl(topk_weights, topk_ids, router_logits, renormalize,
                          lanes_per_row, retire_strategy, guard_collapse);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("topk_softmax_fused", &topk_softmax_fused,
        "Fused top-k softmax routing, bf16 / 32-256 experts / top_k 1-8 (CUDA)");
  m.def("topk_softmax_fused_variant", &topk_softmax_fused_variant,
        "Test-only: fused top-k softmax with an explicit lane geometry, retire "
        "strategy and collapse-guard setting (CUDA)");
}
