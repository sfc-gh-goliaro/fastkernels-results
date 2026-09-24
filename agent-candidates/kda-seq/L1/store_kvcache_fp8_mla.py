"""MLA KV cache store and gather, with custom BF16 scatter/gather kernels.

Same two-class contract as the baseline. On the ``"auto"`` cache layout -- the
one ``kv_cache_dtype="auto"`` selects, and the only one this operator is
benchmarked on -- both directions reduce to a 1152 byte-per-token row copy into
or out of a paged ``[num_blocks, block_size, 576]`` BF16 cache. The vendored
vLLM kernels leave most of the machine idle doing it:

* ``concat_and_cache_mla`` copies **one BF16 element per thread**, so each
  thread issues a single 2 byte access and retires. That is ~4 KB in flight per
  SM where ~33 KB is needed to cover HBM latency, and it measures 0.74 TB/s on
  a 37.7 MB copy -- roughly 9x off roofline, with 80 % of warp stalls on
  ``long_scoreboard``.
* ``gather_and_maybe_dequant_cache`` gives each token its own 64-thread CTA, so
  the dependent ``token_to_seq -> cu_seq_lens -> block_table`` chain is walked
  again for every row *and* duplicated across the CTA's two warps: 10 of the 46
  load sectors per token are metadata.

Both kernels here instead give one **warp** a whole token and move the row as 72
sixteen-byte vectors, which turns the copy into three warp-wide accesses with
all the loads in flight before the first store, and resolves the gather's
metadata once per row on warp-uniform loads.

Only layouts the vectorized path can prove safe take it. Everything else --
``fp8_ds_mla``, ``fp8_e4m3``, fp16/fp32 caches, an unexpected width, an
innermost stride that is not 1, a row stride that is not a multiple of a
16 byte vector, an unaligned base pointer -- is handed to the vendored kernel,
which is imported lazily so an ``"auto"``-only process never builds it.

Supported cache layouts, argument shapes, and the meaning of ``_k_scale`` are
unchanged from the baseline; see its docstrings.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Importing this pins ``TORCH_CUDA_ARCH_LIST`` to the local compute capability
# (``10.0a`` on this B200) the way every baseline extension in the repo is
# built, so our JIT compile targets the same architecture. It compiles nothing
# by itself, and it is the only reason the import exists -- ``lazy_op`` cannot
# be reused here because it resolves sources relative to the *calling* module's
# directory, which for a candidate is not where the source lives.
import fastkernels.infra.cuda_ext  # noqa: F401

_KV_C_DIM = 512
_K_PE_DIM = 64
_ENTRY_DIM = _KV_C_DIM + _K_PE_DIM      # 576 BF16 elements = 1152 B per token
_VEC_DIM = 8                            # BF16 elements in one 16 byte access

# Route tally for the workspace's own bytewise checker: ``fastkernels bench``
# compares zero tensors for this operator (both forwards return ``None``), so
# the checker needs to see for itself that the benchmarked shapes took the
# vectorized path and the odd layouts did not. Gated on an environment variable
# read once at import, so the benchmarked path pays nothing for it.
_TRACE = os.environ.get("FASTKERNELS_MLA_TRACE") == "1"
_PATH_COUNTS = {"store_fast": 0, "store_fallback": 0, "store_empty": 0,
                "gather_fast": 0, "gather_fallback": 0, "gather_empty": 0}


def path_counts() -> dict[str, int]:
    """Snapshot of how many calls took each route (see ``_TRACE``)."""
    return dict(_PATH_COUNTS)


# Launch-configuration override, so the sweep in tools/mb_baseline.py can time
# every geometry without editing this file between measurements. 0 means "use the
# measured production dispatch", which is what every real call uses; the encoding
# of a non-zero value is documented on the wrappers in the CUDA source.
_STORE_CONFIG = 0
_GATHER_CONFIG = 0


def set_launch_config(store: int = 0, gather: int = 0) -> None:
    """Pin the launch geometry. ``0`` restores the production dispatch.

    ``store`` is a ``TOKENS_PER_CTA``; ``gather`` is
    ``WARPS_PER_CTA * 100 + ROWS_PER_WARP * 10 + page_run``.
    """
    global _STORE_CONFIG, _GATHER_CONFIG
    _STORE_CONFIG, _GATHER_CONFIG = store, gather


def launch_config() -> tuple[int, int]:
    """The currently pinned ``(store, gather)`` configuration."""
    return _STORE_CONFIG, _GATHER_CONFIG


_CPP_SOURCE = """
void store_mla_bf16(at::Tensor kv_c_normed, at::Tensor k_pe,
                    at::Tensor kv_cache, at::Tensor slot_mapping,
                    int64_t config);
void gather_mla_bf16(at::Tensor kv_cache, at::Tensor workspace,
                     at::Tensor block_table, at::Tensor cu_seq_lens,
                     at::Tensor token_to_seq, int64_t total_tokens,
                     at::Tensor workspace_starts, int64_t config);
"""

_CUDA_SOURCE = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <limits>

namespace {

// A 576-element BF16 token row is 72 sixteen-byte vectors: 64 carrying the
// compressed KV half and 8 carrying the RoPE half. One warp owns a whole row,
// so the 32 lanes cover it in three warp-wide accesses -- two full 512 B ones
// and one eight-lane 128 B one -- and every access is warp-uniform in its row.
constexpr int kKvCVecs = 64;
constexpr int kPeVecs = 8;
constexpr int kEntryVecs = kKvCVecs + kPeVecs;   // 72
constexpr int kVecDim = 8;   // BF16 elements per 16 byte vector

// The page-run gather works a whole page at a time, so its tile is the captured
// page size; a tile that is not exactly one page cannot take that path.
constexpr int kPageTile = 64;
constexpr int kPageRunThreads = 256;

// All of the following were selected by the sweep in tools/mb_baseline.py over
// every geometry certified by tools/check_correctness.py; the full matrix is in
// profile/launch_sweep.json.
//
// Store: two tokens per CTA is fastest or tied at every captured size (13.33 us
// at N=16384 against 15.36 for one token per CTA, and <= 7.17 us on every small
// case), so there is no threshold to pick.
constexpr int kStoreTokensPerCta = 2;
// Gather: one row per warp wins below ~1024 tokens (7.20 us at T=3 against 9.15
// for two rows, which pays for a second chain nothing uses), and two rows per
// warp wins above it (31.74 us at T=65536 against 33.68 for one row). Two rows
// costs 36 registers against 24, capping occupancy near 87 %, and still wins:
// the memory-level parallelism is worth more than the resident warps.
constexpr int kGatherSmallBatch = 1024;
constexpr int kGatherSmallWarps = 4;
constexpr int kGatherLargeWarps = 8;

template <int TOKENS_PER_CTA>
__global__ void store_mla_bf16_kernel(
    const uint4* __restrict__ kv_c,
    const uint4* __restrict__ k_pe,
    uint4* __restrict__ kv_cache,
    const int64_t* __restrict__ slot_mapping,
    const int num_tokens,
    const int64_t kv_c_row_vecs,
    const int64_t k_pe_row_vecs,
    const int block_size,
    const int64_t block_vecs,
    const int64_t entry_vecs) {
  const int lane = threadIdx.x & 31;
  const int warp = TOKENS_PER_CTA == 1 ? 0 : static_cast<int>(threadIdx.x >> 5);
  const int token = blockIdx.x * TOKENS_PER_CTA + warp;
  if (token >= num_tokens) return;

  // Issue the slot load first -- it starts the longest dependency chain, since
  // the destination address hangs off it -- and then the three payload loads,
  // which do not depend on it at all. Waiting for slot_mapping before issuing
  // the payload serialises two full memory round trips for no reason. The rows
  // are in bounds for every token < num_tokens because the host guard proved
  // both sources have at least that many rows, so reading the row of a token
  // that turns out to be padded is safe; it is a discarded read with no side
  // effect, and padded tokens are rare (vLLM V1 sizes slot_mapping to the
  // unpadded count).
  const int64_t slot = slot_mapping[token];
  const uint4* src_kv = kv_c + static_cast<int64_t>(token) * kv_c_row_vecs;
  const uint4* src_pe = k_pe + static_cast<int64_t>(token) * k_pe_row_vecs;
  const bool carries_pe = lane < kPeVecs;
  const uint4 lo = src_kv[lane];
  const uint4 hi = src_kv[lane + 32];
  uint4 pe = make_uint4(0u, 0u, 0u, 0u);
  if (carries_pe) pe = src_pe[lane];

  // vLLM marks a CUDA-graph padded token with -1. Predicating the *stores* on
  // that, instead of returning before them, is deliberate: an early return here
  // gives the compiler a reason to sink the payload loads below the branch, and
  // it does -- with a `return` the SASS issues the k_pe load, then EXIT, then
  // the two kv_c loads, which puts a full round trip on slot_mapping in front of
  // most of the payload. The cost is that a padded token performs three loads it
  // discards, which is cheap and rare (vLLM V1 sizes slot_mapping to the
  // unpadded count, so negatives do not appear on this workload at all).
  const bool live = slot >= 0;
  // Clamped so the address arithmetic below is well defined even for a padded
  // token whose pointer is never dereferenced.
  const int64_t safe_slot = live ? slot : 0;

  // Decomposed and widened exactly as the vendored kernel does it. The
  // decomposition has to stay -- stride(0) is not necessarily
  // block_size * stride(1), so addressing the cache as slot * entry_stride
  // would break on a cache that is a slice of a wider allocation -- and the
  // arithmetic has to be 64-bit, because the captured cache holds
  // 182699 * 64 * 576 = 6.7e9 elements, past what 32 bits can index.
  const int64_t block_idx = safe_slot / block_size;
  const int64_t block_off = safe_slot % block_size;
  uint4* dst = kv_cache + block_idx * block_vecs + block_off * entry_vecs;

  if (live) {
    dst[lane] = lo;
    dst[lane + 32] = hi;
    if (carries_pe) dst[kKvCVecs + lane] = pe;
  }
}

// Resolve one token's metadata chain. Returns false when the vendored kernel
// would skip the token: past the end of its sequence, or past num_tokens.
__device__ __forceinline__ bool resolve_row(
    const int32_t* __restrict__ cu_seq_lens,
    const int32_t* __restrict__ token_to_seq,
    const int32_t* __restrict__ workspace_starts,
    int token, int num_tokens, int& seq, int32_t& off) {
  if (token >= num_tokens) return false;
  seq = token_to_seq[token];
  // The end test precedes the offset arithmetic, the block-table read, and the
  // cache read. A token whose sequence has already ended can have an offset
  // past the end of its block-table row, so resolving the page first would
  // read out of bounds as well as write a row the vendored kernel leaves alone.
  if (token >= cu_seq_lens[seq + 1]) return false;
  off = token - cu_seq_lens[seq];
  if (workspace_starts != nullptr) off += workspace_starts[seq];
  return true;
}

__device__ __forceinline__ void copy_row(
    const uint4* __restrict__ src, uint4* __restrict__ dst, int lane) {
  const bool carries_pe = lane < kPeVecs;
  const uint4 lo = src[lane];
  const uint4 hi = src[lane + 32];
  uint4 pe = make_uint4(0u, 0u, 0u, 0u);
  if (carries_pe) pe = src[kKvCVecs + lane];
  dst[lane] = lo;
  dst[lane + 32] = hi;
  if (carries_pe) dst[kKvCVecs + lane] = pe;
}

// Deliberately *not* register-budgeted. KernelWiki's technique-vectorized-loads
// recommends `__launch_bounds__`/`-maxrregcount` for memory-bound kernels, and it
// was tried here: forcing this kernel from 36 registers down to 32 (to lift the
// resident-warp cap from ~87 % to 100 %) makes ptxas spill 8 bytes, and the spill
// destroys exactly the overlap the second row buys. Measured in one session,
// ROWS_PER_WARP=2 against ROWS_PER_WARP=1 goes from 0.942x to 1.001x -- the whole
// advantage gone. Registers here are cheaper than resident warps.
template <int WARPS_PER_CTA, int ROWS_PER_WARP>
__global__ void gather_mla_bf16_kernel(
    const uint4* __restrict__ kv_cache,
    uint4* __restrict__ workspace,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ cu_seq_lens,
    const int32_t* __restrict__ token_to_seq,
    const int32_t* __restrict__ workspace_starts,
    const int num_tokens,
    const int block_size,
    const int64_t block_table_row,
    const int64_t block_vecs,
    const int64_t entry_vecs,
    const int64_t dst_row_vecs) {
  const int lane = threadIdx.x & 31;
  const int warp = WARPS_PER_CTA == 1 ? 0 : static_cast<int>(threadIdx.x >> 5);
  const int base = (blockIdx.x * WARPS_PER_CTA + warp) * ROWS_PER_WARP;

  if constexpr (ROWS_PER_WARP == 1) {
    // One row per warp: every lane resolves the same warp-uniform chain, which
    // is one sector per level, so there is nothing to broadcast.
    int seq = 0;
    int32_t off = 0;
    if (!resolve_row(cu_seq_lens, token_to_seq, workspace_starts, base,
                     num_tokens, seq, off))
      return;
    const int32_t page =
        block_table[static_cast<int64_t>(seq) * block_table_row + off / block_size];
    copy_row(kv_cache + static_cast<int64_t>(page) * block_vecs +
                 static_cast<int64_t>(off % block_size) * entry_vecs,
             workspace + static_cast<int64_t>(base) * dst_row_vecs, lane);
    return;
  }

  // Several rows per warp: resolve their chains *in parallel across lanes* and
  // broadcast the results, rather than serially in one lane, which would
  // rebuild exactly the dependent-latency problem this kernel exists to remove.
  // No lane may return before the shuffles: they need the full warp.
  int my_page = 0, my_slot = 0, my_live = 0;
  if (lane < ROWS_PER_WARP) {
    int seq = 0;
    int32_t off = 0;
    if (resolve_row(cu_seq_lens, token_to_seq, workspace_starts, base + lane,
                    num_tokens, seq, off)) {
      my_page = block_table[static_cast<int64_t>(seq) * block_table_row +
                            off / block_size];
      my_slot = off % block_size;
      my_live = 1;
    }
  }

  const uint4* src[ROWS_PER_WARP];
  int live[ROWS_PER_WARP];
#pragma unroll
  for (int r = 0; r < ROWS_PER_WARP; ++r) {
    live[r] = __shfl_sync(0xffffffffu, my_live, r);
    const int page = __shfl_sync(0xffffffffu, my_page, r);
    const int slot = __shfl_sync(0xffffffffu, my_slot, r);
    src[r] = kv_cache + static_cast<int64_t>(page) * block_vecs +
             static_cast<int64_t>(slot) * entry_vecs;
  }

  // All the rows' loads are issued before any store, so several rows' payloads
  // are outstanding together -- the point of ROWS_PER_WARP > 1.
  const bool carries_pe = lane < kPeVecs;
  uint4 lo[ROWS_PER_WARP], hi[ROWS_PER_WARP], pe[ROWS_PER_WARP];
#pragma unroll
  for (int r = 0; r < ROWS_PER_WARP; ++r) {
    if (live[r]) {
      lo[r] = src[r][lane];
      hi[r] = src[r][lane + 32];
      if (carries_pe) pe[r] = src[r][kKvCVecs + lane];
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS_PER_WARP; ++r) {
    if (live[r]) {
      uint4* dst = workspace + static_cast<int64_t>(base + r) * dst_row_vecs;
      dst[lane] = lo[r];
      dst[lane + 32] = hi[r];
      if (carries_pe) dst[kKvCVecs + lane] = pe[r];
    }
  }
}

// One CTA per 64-token tile. Where the tile happens to be exactly one page of
// one sequence -- which is what workspace_starts == cu_seq_lens[:-1] produces,
// and therefore what the captured workload produces almost everywhere -- the 64
// dependent metadata chains collapse into a single block_table lookup and the
// copy becomes 72 KB contiguous on both sides. Any tile that does not qualify
// falls back to the per-row algorithm, in eight-warp waves, on a CTA-uniform
// branch.
__global__ void gather_page_run_kernel(
    const uint4* __restrict__ kv_cache,
    uint4* __restrict__ workspace,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ cu_seq_lens,
    const int32_t* __restrict__ token_to_seq,
    const int32_t* __restrict__ workspace_starts,
    const int num_tokens,
    const int block_size,
    const int64_t block_table_row,
    const int64_t block_vecs,
    const int64_t entry_vecs,
    const int64_t dst_row_vecs) {
  __shared__ int s_seq[kPageTile];
  __shared__ int s_off[kPageTile];
  __shared__ int s_page;
  __shared__ int s_disqualified;

  const int t0 = blockIdx.x * kPageTile;
  if (threadIdx.x == 0) s_disqualified = 0;
  __syncthreads();

  // The first 64 lanes resolve and validate one token each. A token the
  // vendored kernel would skip is marked with seq = -1 and disqualifies the
  // cooperative path; the fallback below then leaves its row untouched.
  if (threadIdx.x < kPageTile) {
    int seq = -1;
    int32_t off = -1;
    if (!resolve_row(cu_seq_lens, token_to_seq, workspace_starts,
                     t0 + threadIdx.x, num_tokens, seq, off)) {
      seq = -1;
      atomicOr(&s_disqualified, 1);
    }
    s_seq[threadIdx.x] = seq;
    s_off[threadIdx.x] = off;
  }
  __syncthreads();

  // Qualify only an exact page run: all 64 tokens live, one sequence,
  // consecutive offsets, starting at slot 0, filling exactly one page.
  if (threadIdx.x < kPageTile) {
    if (s_seq[threadIdx.x] != s_seq[0] ||
        s_off[threadIdx.x] != s_off[0] + static_cast<int>(threadIdx.x))
      atomicOr(&s_disqualified, 1);
    if (threadIdx.x == 0 && (s_seq[0] < 0 || s_off[0] % block_size != 0))
      atomicOr(&s_disqualified, 1);
  }
  __syncthreads();

  if (s_disqualified == 0) {
    if (threadIdx.x == 0)
      s_page = block_table[static_cast<int64_t>(s_seq[0]) * block_table_row +
                           s_off[0] / block_size];
    __syncthreads();
    // The launcher only selects this kernel when a page's rows and the
    // destination rows are both exactly kEntryVecs apart, so source and
    // destination are each one contiguous run of kPageTile * kEntryVecs
    // vectors and consecutive threads touch consecutive addresses.
    const uint4* src = kv_cache + static_cast<int64_t>(s_page) * block_vecs;
    uint4* dst = workspace + static_cast<int64_t>(t0) * dst_row_vecs;
    constexpr int kTileVecs = kPageTile * kEntryVecs;
#pragma unroll 4
    for (int i = threadIdx.x; i < kTileVecs; i += kPageRunThreads)
      dst[i] = src[i];
    return;
  }

  // CTA-uniform fallback: the per-row algorithm, reusing the metadata the first
  // 64 lanes already resolved. s_seq[r] < 0 marks the rows to leave alone.
  const int lane = threadIdx.x & 31;
  const int warp = static_cast<int>(threadIdx.x >> 5);
  constexpr int kWaveWarps = kPageRunThreads / 32;
#pragma unroll
  for (int wave = 0; wave < kPageTile / kWaveWarps; ++wave) {
    const int r = wave * kWaveWarps + warp;
    const int seq = s_seq[r];
    if (seq < 0) continue;
    const int32_t off = s_off[r];
    const int32_t page =
        block_table[static_cast<int64_t>(seq) * block_table_row + off / block_size];
    copy_row(kv_cache + static_cast<int64_t>(page) * block_vecs +
                 static_cast<int64_t>(off % block_size) * entry_vecs,
             workspace + static_cast<int64_t>(t0 + r) * dst_row_vecs, lane);
  }
}

}  // namespace

void store_mla_bf16(at::Tensor kv_c_normed, at::Tensor k_pe,
                    at::Tensor kv_cache, at::Tensor slot_mapping,
                    int64_t config) {
  const c10::cuda::CUDAGuard device_guard(kv_cache.device());
  // slot_mapping, not kv_c_normed: in vLLM V1 the source rows carry CUDA-graph
  // padding that slot_mapping does not, and the vendored launcher takes its
  // token count from here for the same reason.
  const int64_t token_count = slot_mapping.size(0);
  if (token_count == 0) return;
  TORCH_CHECK(token_count <= std::numeric_limits<int>::max(),
              "store_mla_bf16: num_tokens exceeds int32");
  const int num_tokens = static_cast<int>(token_count);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int block_size = static_cast<int>(kv_cache.size(1));
  const int64_t kv_c_row_vecs = kv_c_normed.stride(0) / kVecDim;
  const int64_t k_pe_row_vecs = k_pe.stride(0) / kVecDim;
  const int64_t block_vecs = kv_cache.stride(0) / kVecDim;
  const int64_t entry_vecs = kv_cache.stride(1) / kVecDim;

  const uint4* src_kv = reinterpret_cast<const uint4*>(kv_c_normed.data_ptr());
  const uint4* src_pe = reinterpret_cast<const uint4*>(k_pe.data_ptr());
  uint4* dst = reinterpret_cast<uint4*>(kv_cache.data_ptr());
  const int64_t* slots = slot_mapping.data_ptr<int64_t>();

  // config 0 selects the measured production dispatch; any other value is a
  // TOKENS_PER_CTA the sweep in tools/mb_baseline.py wants timed directly.
  const int tokens_per_cta =
      config != 0 ? static_cast<int>(config) : kStoreTokensPerCta;

#define KDA_LAUNCH_STORE(T)                                                   \
  store_mla_bf16_kernel<T><<<1 + (num_tokens - 1) / (T), 32 * (T), 0,         \
                             stream>>>(                                       \
      src_kv, src_pe, dst, slots, num_tokens, kv_c_row_vecs, k_pe_row_vecs,   \
      block_size, block_vecs, entry_vecs)

  switch (tokens_per_cta) {
    case 1: KDA_LAUNCH_STORE(1); break;
    case 2: KDA_LAUNCH_STORE(2); break;
    case 4: KDA_LAUNCH_STORE(4); break;
    case 8: KDA_LAUNCH_STORE(8); break;
    default:
      TORCH_CHECK(false, "store_mla_bf16: unsupported tokens_per_cta ",
                  tokens_per_cta);
  }
#undef KDA_LAUNCH_STORE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_mla_bf16(at::Tensor kv_cache, at::Tensor workspace,
                     at::Tensor block_table, at::Tensor cu_seq_lens,
                     at::Tensor token_to_seq, int64_t total_tokens,
                     at::Tensor workspace_starts, int64_t config) {
  const c10::cuda::CUDAGuard device_guard(kv_cache.device());
  if (total_tokens == 0) return;
  TORCH_CHECK(total_tokens > 0, "gather_mla_bf16: total_tokens must be "
                                "non-negative, got ", total_tokens);
  TORCH_CHECK(total_tokens <= std::numeric_limits<int>::max(),
              "gather_mla_bf16: total_tokens exceeds int32");
  const int num_tokens = static_cast<int>(total_tokens);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int block_size = static_cast<int>(kv_cache.size(1));
  const int64_t block_vecs = kv_cache.stride(0) / kVecDim;
  const int64_t entry_vecs = kv_cache.stride(1) / kVecDim;
  const int64_t dst_row_vecs = workspace.stride(0) / kVecDim;

  const uint4* src = reinterpret_cast<const uint4*>(kv_cache.data_ptr());
  uint4* dst = reinterpret_cast<uint4*>(workspace.data_ptr());
  const int32_t* bt = block_table.data_ptr<int32_t>();
  const int32_t* cu = cu_seq_lens.data_ptr<int32_t>();
  const int32_t* t2s = token_to_seq.data_ptr<int32_t>();
  const int32_t* starts = workspace_starts.data_ptr<int32_t>();
  const int64_t block_table_row = block_table.stride(0);

  // The page-run kernel copies a whole page as one contiguous run, so it needs
  // a page to be exactly kPageTile rows and both the cache rows within a page
  // and the destination rows to be exactly kEntryVecs apart. Anything else
  // stays on the row kernel.
  const bool page_run_possible = block_size == kPageTile &&
                                 entry_vecs == kEntryVecs &&
                                 dst_row_vecs == kEntryVecs;

  // config 0 selects the measured production dispatch. Otherwise the sweep
  // encodes one configuration as warps * 100 + rows * 10 + page_run.
  int warps_per_cta, rows_per_warp;
  bool page_run;
  if (config != 0) {
    page_run = (config % 10) != 0;
    rows_per_warp = static_cast<int>((config / 10) % 10);
    warps_per_cta = static_cast<int>(config / 100);
  } else {
    // The page-run kernel is never selected automatically. It is implemented,
    // certified bytewise correct, and reachable through `config` -- but it
    // measures slower than the row kernel at every captured size (33.79 us
    // against 31.74 at T=65536; 17.26 against 13.31 at T=14588). Collapsing 64
    // dependent metadata chains into one lookup does not pay for the two
    // __syncthreads and the shared-memory round trip its validation needs,
    // whereas ROWS_PER_WARP=2 buys the same overlap for free.
    page_run = false;
    const bool small = num_tokens <= kGatherSmallBatch;
    rows_per_warp = small ? 1 : 2;
    warps_per_cta = small ? kGatherSmallWarps : kGatherLargeWarps;
  }

  if (page_run && page_run_possible) {
    gather_page_run_kernel<<<1 + (num_tokens - 1) / kPageTile, kPageRunThreads,
                             0, stream>>>(
        src, dst, bt, cu, t2s, starts, num_tokens, block_size, block_table_row,
        block_vecs, entry_vecs, dst_row_vecs);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

#define KDA_LAUNCH_GATHER(W, R)                                               \
  gather_mla_bf16_kernel<W, R><<<1 + (num_tokens - 1) / ((W) * (R)),          \
                                 32 * (W), 0, stream>>>(                      \
      src, dst, bt, cu, t2s, starts, num_tokens, block_size, block_table_row, \
      block_vecs, entry_vecs, dst_row_vecs)
#define KDA_GATHER_ROWS(W)                                                    \
  switch (rows_per_warp) {                                                    \
    case 1: KDA_LAUNCH_GATHER(W, 1); break;                                   \
    case 2: KDA_LAUNCH_GATHER(W, 2); break;                                   \
    case 4: KDA_LAUNCH_GATHER(W, 4); break;                                   \
    default:                                                                  \
      TORCH_CHECK(false, "gather_mla_bf16: unsupported rows_per_warp ",        \
                  rows_per_warp);                                             \
  }

  switch (warps_per_cta) {
    case 1: KDA_GATHER_ROWS(1); break;
    case 2: KDA_GATHER_ROWS(2); break;
    case 4: KDA_GATHER_ROWS(4); break;
    case 8: KDA_GATHER_ROWS(8); break;
    default:
      TORCH_CHECK(false, "gather_mla_bf16: unsupported warps_per_cta ",
                  warps_per_cta);
  }
#undef KDA_GATHER_ROWS
#undef KDA_LAUNCH_GATHER
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_EXT = None
_VENDORED = None


def _ext():
    """Our two kernels, JIT-compiled on the first call that needs them.

    Deferred rather than built at import so that importing this module -- which
    the bench harness does while collecting operators, and which the correctness
    checker does before it has a GPU in hand -- never invokes nvcc.
    """
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name="kda_mla_kvcache_bf16",
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["store_mla_bf16", "gather_mla_bf16"],
            extra_cflags=["-O3"],
            # -lineinfo so Nsight Compute can attribute counters back to the
            # source lines above; the kernels carry no relaxed-constexpr or
            # BF16-conversion needs because they move raw 16 byte vectors.
            extra_cuda_cflags=["-O3", "-lineinfo"],
        )
    return _EXT


def _vendored():
    """The vendored vLLM extension, for the layouts we do not accelerate.

    Imported here rather than at module scope so an ``"auto"``-only process --
    which is every benchmarked call -- never pays its compile or load cost. The
    baseline module is *not* held as a submodule: that would add a child to this
    module's tree and change what the harness' weight sharing and module walk
    see, so the entry point is called directly with our own ``_k_scale``.
    """
    global _VENDORED
    if _VENDORED is None:
        from fastkernels.tasks.baseline.L1 import store_kvcache_fp8_mla as vendored
        _VENDORED = vendored._C
    return _VENDORED


def _paged_cache_ok(kv_cache: torch.Tensor) -> bool:
    """Whether the paged cache is BF16, 576 wide, and vector-addressable with
    non-overlapping rows *and* non-overlapping pages.

    Both overlap tests matter. ``stride(1) >= 576`` stops the entries inside one
    page overlapping each other, but a cache whose ``stride(0)`` is smaller than
    ``size(1) * stride(1)`` has one page reaching into the next, so two logically
    distinct rows alias. The vendored store launches one CTA per token while this
    one packs several token warps into a CTA, so the two would not necessarily
    resolve such a conflict the same way -- there is no defined result to match,
    and the layout is handed to the vendored kernel instead.
    """
    shape, stride = kv_cache.shape, kv_cache.stride()
    return (len(shape) == 3
            and shape[2] == _ENTRY_DIM
            and stride[2] == 1
            and stride[1] >= _ENTRY_DIM
            and stride[0] >= shape[1] * stride[1]
            and stride[0] % _VEC_DIM == 0
            and stride[1] % _VEC_DIM == 0)


def _rows_and_stride(t: torch.Tensor, width: int):
    """``(row count, row stride in elements)`` if every row of *t* is exactly
    *width* contiguous elements, else ``None``.

    This is what lets the wrapper read ``stride(0)`` instead of reshaping in
    Python. It has to be exact: ``k_pe`` arrives as ``[N, 1, 64]`` with
    ``stride(0) = 576`` when it is a view into a fused projection output, but as
    ``stride(0) = 64`` when N is 1 and the tensor happens to be contiguous, and
    a ``[N, 2, 32]`` row split across a non-contiguous middle dimension is not a
    row we can move as whole vectors at all.

    Shape and strides are pulled as tuples in one call each -- fetching the
    whole tuple costs less than a single ``t.stride(i)``, and this runs on every
    call.
    """
    shape, stride = t.shape, t.stride()
    rank = len(shape)
    if rank == 2:
        return (shape[0], stride[0]) if stride[1] == 1 and shape[1] == width else None
    if rank == 3:
        if (stride[2] == 1 and stride[1] == shape[2]
                and shape[1] * shape[2] == width):
            return shape[0], stride[0]
        return None
    if rank < 2:
        return None
    span = 1
    for d in range(rank - 1, 0, -1):
        if stride[d] != span:
            return None
        span *= shape[d]
    return (shape[0], stride[0]) if span == width else None


def _store_takes_fast_path(kv_cache_dtype, kv_c_normed, k_pe, kv_cache,
                           slot_mapping) -> bool:
    """Whether the vectorized scatter can reproduce this call exactly."""
    if kv_cache_dtype != "auto":
        return False
    bf16 = torch.bfloat16
    if (kv_cache.dtype is not bf16 or kv_c_normed.dtype is not bf16
            or k_pe.dtype is not bf16
            or slot_mapping.dtype is not torch.int64
            or not _paged_cache_ok(kv_cache)):
        return False

    slot_shape, slot_stride = slot_mapping.shape, slot_mapping.stride()
    if len(slot_shape) != 1 or slot_stride[0] != 1:
        return False
    num_tokens = slot_shape[0]

    # Rank 2 exactly. The vendored launcher takes kv_lora_rank from
    # kv_c.size(1) and then requires kv_cache.size(2) == kv_lora_rank + pe_dim,
    # so it *rejects* a contiguous [N, 2, 256] source against a 576-wide cache;
    # a flatten-tolerant test here would accelerate a call the baseline refuses.
    # k_pe stays flatten-tolerant because the baseline reshapes it.
    kv_c_shape, kv_c_stride = kv_c_normed.shape, kv_c_normed.stride()
    if (len(kv_c_shape) != 2 or kv_c_shape[1] != _KV_C_DIM
            or kv_c_stride[1] != 1):
        return False
    k_pe_row = _rows_and_stride(k_pe, _K_PE_DIM)
    if k_pe_row is None:
        return False
    # Every *derived* row has to be 16 byte aligned, not just the base pointer,
    # which is what makes the row strides part of the test and not only the
    # innermost one.
    if kv_c_stride[0] % _VEC_DIM or k_pe_row[1] % _VEC_DIM:
        return False
    # slot_mapping, not the sources, sets the token count: in vLLM V1 the source
    # rows carry CUDA-graph padding that slot_mapping does not.
    if kv_c_shape[0] < num_tokens or k_pe_row[0] < num_tokens:
        return False
    if (kv_cache.data_ptr() | kv_c_normed.data_ptr() | k_pe.data_ptr()) % 16:
        return False

    device = kv_cache.get_device()          # -1 for a CPU tensor
    return (device >= 0 and kv_c_normed.get_device() == device
            and k_pe.get_device() == device
            and slot_mapping.get_device() == device)


def _gather_takes_fast_path(kv_cache_dtype, kv_cache, workspace, block_table,
                            cu_seq_lens, token_to_seq, total_tokens,
                            workspace_starts) -> bool:
    """Whether the row-tiled gather can reproduce this call exactly."""
    if kv_cache_dtype != "auto" or not isinstance(total_tokens, int):
        return False
    bf16, i32 = torch.bfloat16, torch.int32
    if (kv_cache.dtype is not bf16 or workspace.dtype is not bf16
            or block_table.dtype is not i32 or cu_seq_lens.dtype is not i32
            or token_to_seq.dtype is not i32):
        return False
    # workspace_starts is a required argument of this contract, so anything else
    # here is out of contract rather than a layout worth accelerating.
    if (not isinstance(workspace_starts, torch.Tensor)
            or workspace_starts.dtype is not i32):
        return False

    if not _paged_cache_ok(kv_cache):
        return False

    # An output row stride below the row width would make the written rows
    # overlap, so which warp wrote last would decide the result.
    dst = _rows_and_stride(workspace, _ENTRY_DIM)
    if (dst is None or dst[1] < _ENTRY_DIM or dst[1] % _VEC_DIM
            or dst[0] < total_tokens):
        return False

    bt_shape, bt_stride = block_table.shape, block_table.stride()
    if len(bt_shape) != 2 or bt_stride[1] != 1:
        return False
    num_seqs = bt_shape[0]

    t2s_shape, t2s_stride = token_to_seq.shape, token_to_seq.stride()
    if len(t2s_shape) != 1 or t2s_stride[0] != 1 or t2s_shape[0] < total_tokens:
        return False
    cu_shape, cu_stride = cu_seq_lens.shape, cu_seq_lens.stride()
    # cu_seq_lens is read at seq and seq + 1, workspace_starts at seq.
    if len(cu_shape) != 1 or cu_stride[0] != 1 or cu_shape[0] < num_seqs + 1:
        return False
    start_shape, start_stride = workspace_starts.shape, workspace_starts.stride()
    if len(start_shape) != 1 or start_stride[0] != 1 or start_shape[0] < num_seqs:
        return False

    if (kv_cache.data_ptr() | workspace.data_ptr()) % 16:
        return False

    device = kv_cache.get_device()          # -1 for a CPU tensor
    return (device >= 0 and workspace.get_device() == device
            and block_table.get_device() == device
            and cu_seq_lens.get_device() == device
            and token_to_seq.get_device() == device
            and workspace_starts.get_device() == device)


class StoreKVCacheFP8MLA(nn.Module):
    """Store ``kv_c_normed`` and ``k_pe`` into MLA paged cache.

    On the ``"auto"`` BF16 layout this runs our warp-per-token 128-bit scatter:
    72 sixteen-byte vectors per token, one warp per token, three warp-wide
    stores instead of the vendored kernel's eighteen. ``"fp8_ds_mla"`` and
    ``"fp8_e4m3"``, and any BF16 layout the vectorized path cannot prove safe,
    go to the vendored ``concat_and_cache_mla``.

    Args:
        kv_c_normed: ``[N, 512]`` BF16 -- compressed KV after layernorm.
        k_pe: ``[N, 1, 64]`` or ``[N, 64]`` BF16 -- RoPE key component. Its row
            stride is read as-is; the captured layout is a view into a fused
            allocation with ``stride(0) = 576``, not 64.
        kv_cache: ``[num_blocks, block_size, 576|656]`` (BF16 / fp8 / uint8).
        slot_mapping: ``[N]`` int64 -- linear slot index per token (``-1`` skips).
    """

    def __init__(self, kv_cache_dtype: str = "auto"):
        super().__init__()
        assert kv_cache_dtype in ("auto", "fp8_ds_mla", "fp8_e4m3"), (
            f"StoreKVCacheFP8MLA: unsupported kv_cache_dtype={kv_cache_dtype!r}"
        )
        self.kv_cache_dtype = kv_cache_dtype
        # ``k_scale`` is the per-tensor dequant scale the kernel divides by on
        # the ``fp8_e4m3`` path (it is ignored for ``auto`` and for the
        # block-scaled ``fp8_ds_mla`` layout). vLLM initialises ``layer._k_scale``
        # to 1.0 and only overwrites it from a checkpoint's calibration scales,
        # which nvidia/GLM-5.2-NVFP4 does not ship -- so ONE, not zero. A zero
        # here silently turned every stored KV element into inf/nan. Kept
        # non-persistent and fp32: the harness casts parameters, never buffers.
        self.register_buffer(
            "_k_scale", torch.ones(1, dtype=torch.float32), persistent=False,
        )

    def forward(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        # Before choosing a route: a zero-token call has nothing to do, and the
        # vendored launcher would build dim3 grid(0), which is not a valid launch
        # configuration. Deciding this first makes the no-op independent of
        # whether the layout would have been accelerated, and means a zero-token
        # call never triggers the JIT build either.
        if slot_mapping.numel() == 0:
            if _TRACE:
                _PATH_COUNTS["store_empty"] += 1
            return
        if _store_takes_fast_path(self.kv_cache_dtype, kv_c_normed, k_pe,
                                  kv_cache, slot_mapping):
            if _TRACE:
                _PATH_COUNTS["store_fast"] += 1
            _ext().store_mla_bf16(kv_c_normed, k_pe, kv_cache, slot_mapping,
                                  _STORE_CONFIG)
            return
        if _TRACE:
            _PATH_COUNTS["store_fallback"] += 1
        k_pe_2d = k_pe.reshape(k_pe.shape[0], -1)
        _vendored().concat_and_cache_mla(
            kv_c_normed, k_pe_2d, kv_cache, slot_mapping,
            self.kv_cache_dtype, self._k_scale,
        )


class GatherKVCacheFP8MLA(nn.Module):
    """Gather and upconvert KV from FP8 MLA paged cache to BF16.

    Unchanged from the baseline: this entry point is FP8-only, so there is no
    BF16 row copy to vectorize and it delegates to the vendored kernel.

    Returns:
        ``workspace``: ``[total_tokens, 576]`` BF16 -- dequantized kv_c_normed
        (512 dims) concatenated with k_pe (64 dims).
    """

    def forward(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        workspace_starts: torch.Tensor,
        num_seqs: int,
        workspace: torch.Tensor,
    ) -> None:
        # ``seq_lens`` is unused by the kernel; lengths are implied by
        # ``workspace_starts`` + ``workspace.size(0)``. Kept in the Module API
        # for call-site compatibility.
        del seq_lens
        _vendored().cp_gather_and_upconvert_fp8_kv_cache(
            kv_cache, workspace, block_table, workspace_starts, num_seqs, None,
        )


class GatherAndDequantKVCacheMLA(nn.Module):
    """Gather MLA KV cache into a BF16 workspace.

    On the ``"auto"`` BF16 layout this runs our row-tiled gather: one warp owns
    a whole 576-element row, the ``token_to_seq -> cu_seq_lens -> block_table``
    chain is resolved once per row on warp-uniform loads instead of once per
    warp per row, and the skip test for a token whose sequence has already ended
    happens before the block-table read. Other layouts go to the vendored
    ``gather_and_maybe_dequant_cache``.

    Required arguments match the kernel's signature:
        ``kv_cache``: ``[num_blocks, block_size, 576]`` BF16 (``"auto"``) or
                      ``[num_blocks, block_size, 656]`` uint8 (``fp8_ds_mla``).
        ``workspace``: ``[total_tokens, 576]`` BF16 output buffer. It may have
                      more rows than ``total_tokens``; the extra rows are left
                      untouched, and the row count comes from ``total_tokens``.
        ``block_table``: ``[num_seqs, max_blocks]`` int32.
        ``cu_seq_lens``: ``[num_seqs+1]`` int32 cumulative sequence lengths.
        ``token_to_seq``: ``[total_tokens]`` int32 mapping.
        ``total_tokens``: scalar int.
        ``workspace_starts``: ``[num_seqs]`` int32 -- starting workspace row
                             per sequence (for chunked context gathers). It is
                             not assumed to equal ``cu_seq_lens[:-1]``.

    ``kv_cache_dtype`` selects the source layout and must match the cache the
    owning ``MLAAttention`` allocated; vLLM likewise forwards its own
    ``self.kv_cache_dtype`` here, and passing ``"fp8_ds_mla"`` for a BF16
    cache reinterprets the bytes and silently corrupts the gathered context.
    """

    def __init__(self, kv_cache_dtype: str = "fp8_ds_mla"):
        super().__init__()
        assert kv_cache_dtype in ("auto", "fp8_ds_mla", "fp8_e4m3"), (
            f"GatherAndDequantKVCacheMLA: unsupported "
            f"kv_cache_dtype={kv_cache_dtype!r}"
        )
        self.kv_cache_dtype = kv_cache_dtype
        # ONE, not zero: on the ``fp8_e4m3`` path the kernel MULTIPLIES the
        # gathered fp8 values by this scale to dequantize (vLLM passes
        # ``layer._k_scale``, default 1.0). Zero would blank the gathered
        # context. Ignored for ``auto`` and for the block-scaled ``fp8_ds_mla``.
        self.register_buffer(
            "_k_scale", torch.ones(1, dtype=torch.float32), persistent=False,
        )

    def forward(
        self,
        kv_cache: torch.Tensor,
        workspace: torch.Tensor,
        block_table: torch.Tensor,
        cu_seq_lens: torch.Tensor,
        token_to_seq: torch.Tensor,
        total_tokens: int,
        workspace_starts: torch.Tensor,
    ) -> None:
        # Same as the store: settle a zero-token call before route selection,
        # since the vendored launcher's dim3 grid(total_tokens) is invalid at 0.
        # A negative count is rejected outright rather than silently ignored --
        # it is not a layout to delegate, it is a caller error, and the vendored
        # launcher would turn it into an enormous grid.
        if isinstance(total_tokens, int):
            if total_tokens < 0:
                raise ValueError(
                    "GatherAndDequantKVCacheMLA: total_tokens must be "
                    f"non-negative, got {total_tokens}"
                )
            if total_tokens == 0:
                if _TRACE:
                    _PATH_COUNTS["gather_empty"] += 1
                return
        if _gather_takes_fast_path(self.kv_cache_dtype, kv_cache, workspace,
                                   block_table, cu_seq_lens, token_to_seq,
                                   total_tokens, workspace_starts):
            if _TRACE:
                _PATH_COUNTS["gather_fast"] += 1
            _ext().gather_mla_bf16(
                kv_cache, workspace, block_table, cu_seq_lens, token_to_seq,
                total_tokens, workspace_starts, _GATHER_CONFIG,
            )
            return
        if _TRACE:
            _PATH_COUNTS["gather_fallback"] += 1
        _vendored().gather_and_maybe_dequant_cache(
            kv_cache, workspace,
            block_table, cu_seq_lens, token_to_seq,
            total_tokens,
            self.kv_cache_dtype,
            self._k_scale,
            workspace_starts,
        )
