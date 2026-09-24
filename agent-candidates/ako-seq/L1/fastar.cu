// Custom intra-node IPC all-reduce for the FastKernels L1 AllReduce op.
//
// Structure follows the vLLM CustomAllreduce design (IPC-shared staging buffer
// + per-block cross-device flag barriers), with three changes aimed at B200:
//
//  1. The device->device staging copy that the eager path used to issue as a
//     separate cudaMemcpyAsync is folded into the reduce kernel, so a
//     non-captured call is ONE launch instead of two. The copy is laid out with
//     the same block->index mapping the reduce phase reads with, so the cheap
//     per-block barrier still suffices (see ar_twostage_fused for the
//     per-segment version of that argument).
//  2. The one-shot (latency) path alternates between two halves of the staging
//     buffer, which removes the need for a closing cross-device barrier: a
//     rank cannot reach call n+2's copy phase until every peer has released
//     call n+1's start flag, which happens only after their call-n kernel
//     retired. That leaves a single barrier on the latency path.
//  3. Launch geometry (blocks / threads) and the one-shot <-> two-stage
//     crossover are tunable at runtime instead of hardcoded to vLLM's
//     A100/H100 constants, and kMaxBlocks is raised well past 36.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#define CUDACHECK(cmd)                                              \
  do {                                                              \
    cudaError_t e = cmd;                                            \
    if (e != cudaSuccess) {                                         \
      throw std::runtime_error(                                     \
          std::string("CUDA error: ") + cudaGetErrorString(e) +     \
          " at " + __FILE__ + ":" + std::to_string(__LINE__));      \
    }                                                               \
  } while (0)

// Grid-size ceiling. B200 has 148 SMs; vLLM's 36 (an A100/H100 inheritance)
// leaves three quarters of the device idle on the multi-MB shapes.
constexpr int kMaxBlocks = 160;
// FlagType slots per block row. 32 slots = 128 B, i.e. one cache line per
// block, so peer flag writes from different blocks never share a line.
constexpr int kFlagPad = 32;

using FlagType = uint32_t;
using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

struct Signal {
  alignas(128) FlagType start[kMaxBlocks][kFlagPad];
  alignas(128) FlagType end[kMaxBlocks][kFlagPad];
  alignas(128) FlagType _flag[kMaxBlocks];
};

struct __align__(16) RankData {
  const void* ptrs[8];
};

struct __align__(16) RankSignals {
  Signal* signals[8];
};

template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};

template <typename T>
struct packed_t {
  using P = array_t<T, 16 / sizeof(T)>;
  using A = array_t<float, 16 / sizeof(T)>;
};

#define DINLINE __device__ __forceinline__

DINLINE float upcast_s(half val) { return __half2float(val); }
DINLINE float upcast_s(nv_bfloat16 val) { return __bfloat162float(val); }

template <typename T>
DINLINE T downcast_s(float val);
template <>
DINLINE half downcast_s(float val) { return __float2half(val); }
template <>
DINLINE nv_bfloat16 downcast_s(float val) { return __float2bfloat16(val); }

DINLINE half& assign_add(half& a, half b) { a = __hadd(a, b); return a; }
DINLINE nv_bfloat16& assign_add(nv_bfloat16& a, nv_bfloat16 b) {
  a = __hadd(a, b); return a;
}
DINLINE float& assign_add(float& a, float b) { return a += b; }

template <typename T, int N>
DINLINE array_t<T, N>& packed_assign_add(array_t<T, N>& a, array_t<T, N> b) {
#pragma unroll
  for (int i = 0; i < N; i++) assign_add(a.data[i], b.data[i]);
  return a;
}

template <typename T, int N>
DINLINE array_t<float, N> upcast(array_t<T, N> val) {
  if constexpr (std::is_same<T, float>::value) {
    return val;
  } else {
    array_t<float, N> out;
#pragma unroll
    for (int i = 0; i < N; i++) out.data[i] = upcast_s(val.data[i]);
    return out;
  }
}

template <typename O>
DINLINE O downcast(array_t<float, O::size> val) {
  if constexpr (std::is_same<typename O::type, float>::value) {
    return val;
  } else {
    O out;
#pragma unroll
    for (int i = 0; i < O::size; i++)
      out.data[i] = downcast_s<typename O::type>(val.data[i]);
    return out;
  }
}

// ---------------------------------------------------------------------------
// Flag operations
// ---------------------------------------------------------------------------
static DINLINE void st_flag_release(FlagType* flag_addr, FlagType flag) {
  asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(flag),
               "l"(flag_addr));
}

static DINLINE FlagType ld_flag_acquire(FlagType* flag_addr) {
  FlagType flag;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(flag)
               : "l"(flag_addr));
  return flag;
}

static DINLINE void st_flag_volatile(FlagType* flag_addr, FlagType flag) {
  asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
}

static DINLINE FlagType ld_flag_volatile(FlagType* flag_addr) {
  FlagType flag;
  asm volatile("ld.volatile.global.u32 %0, [%1];"
               : "=r"(flag)
               : "l"(flag_addr));
  return flag;
}

// Spin until our own copy of the peer flag has REACHED OR PASSED `flag`.
//
// The ">= flag" (rather than "== flag") test is what makes dropping the closing
// barrier viable. Without a closing barrier a rank can enter call k+1 while a
// peer is still in call k, and its call-k+1 flag store overwrites the value the
// peer was waiting on; an equality spin then hangs forever. A monotone test
// instead lets the lagging rank see the newer value and proceed, and run-ahead
// stays bounded at exactly one call (call k+2's barrier cannot pass until the
// lagging rank publishes call k+1), which is precisely what the two alternating
// staging slots cover. The subtraction is done in signed 32-bit so the test
// stays correct across the counter wrapping.
//
// bmode 0: one ld.acquire.sys per poll (vLLM's barrier_at_end shape).
// bmode 1: poll with plain volatile loads and pay a single fence.acq_rel.sys
//          once the flag is observed. Same acquire semantics (load-then-fence),
//          but the acquire cost is paid once instead of once per poll.
static DINLINE void spin_flag(FlagType* self, FlagType flag, int bmode) {
  if (bmode == 0) {
    while ((int32_t)(ld_flag_acquire(self) - flag) < 0);
  } else {
    while ((int32_t)(ld_flag_volatile(self) - flag) < 0);
    asm volatile("fence.acq_rel.sys;" ::: "memory");
  }
}

// ---------------------------------------------------------------------------
// Barriers. RELEASE=true is required whenever the data a peer is about to read
// was written by *this* kernel (the fused-copy paths): the release store plus
// the preceding __syncthreads() is what makes the block's copy visible to the
// peer that observes the flag. RELEASE=false is the vLLM behaviour, valid only
// when the data was written by an earlier kernel on the same stream.
// ---------------------------------------------------------------------------
template <int ngpus, bool RELEASE>
DINLINE void bar_start(const RankSignals& sg, Signal* self_sg, int rank,
                       int bmode) {
  __syncthreads();
  FlagType flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    FlagType* peer = &sg.signals[threadIdx.x]->start[blockIdx.x][rank];
    FlagType* self = &self_sg->start[blockIdx.x][threadIdx.x];
    if constexpr (RELEASE) st_flag_release(peer, flag);
    else st_flag_volatile(peer, flag);
    spin_flag(self, flag, bmode);
  }
  __syncthreads();
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

// FINAL=true skips the trailing __syncthreads() (nothing follows in the
// kernel), matching vLLM's final_sync.
template <int ngpus, bool FINAL>
DINLINE void bar_end(const RankSignals& sg, Signal* self_sg, int rank,
                     int bmode) {
  __syncthreads();
  FlagType flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    FlagType* peer = &sg.signals[threadIdx.x]->end[blockIdx.x][rank];
    FlagType* self = &self_sg->end[blockIdx.x][threadIdx.x];
    if constexpr (!FINAL) {
      st_flag_release(peer, flag);
      spin_flag(self, flag, bmode);
    } else {
      // Closing barrier of the one-shot path: nothing in this kernel reads
      // peer memory afterwards, so plain volatile visibility is enough.
      st_flag_volatile(peer, flag);
      while ((int32_t)(ld_flag_volatile(self) - flag) < 0);
    }
  }
  if constexpr (!FINAL) __syncthreads();
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

template <typename P, int ngpus, typename A>
DINLINE P packed_reduce(const P* const* ptrs, int idx) {
  A tmp = upcast(ptrs[0][idx]);
#pragma unroll
  for (int i = 1; i < ngpus; i++) {
    packed_assign_add(tmp, upcast(ptrs[i][idx]));
  }
  return downcast<P>(tmp);
}

template <typename P>
DINLINE P* get_tmp_buf(Signal* sg) {
  return (P*)(((Signal*)sg) + 1);
}

// ---------------------------------------------------------------------------
// One-shot (every rank reads every rank's copy). Latency path.
//
// FUSED: stage `inp` into our own IPC buffer inside this kernel instead of
//        relying on a preceding cudaMemcpyAsync.
// ENDBAR: emit the closing barrier. Not needed when the host alternates `off`
//        between two staging halves (see the file header).
// ---------------------------------------------------------------------------
// REG: the message fits in one packed unit per thread, so keep our own
// contribution in a register across the barrier. Post-barrier the thread then
// issues ngpus-1 peer loads instead of ngpus loads, one of which was a
// redundant read-back of the line we had just written. On the latency shapes
// that is a quarter of the post-barrier traffic and the only local load on the
// critical path.
template <typename T, int ngpus, bool FUSED, bool ENDBAR, bool REG>
__global__ void __launch_bounds__(1024, 1)
ar_oneshot(RankData* _dp, RankSignals sg, Signal* self_sg,
           const T* __restrict__ inp, T* __restrict__ result, int rank,
           int size, int64_t off, int bmode) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const P* ptrs[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    int t = (rank + i) % ngpus;
    ptrs[i] = ((const P*)_dp->ptrs[t]) + off;
  }
  if constexpr (REG) {
    const bool active = tid < size;
    P mine{};
    if constexpr (FUSED) {
      if (active) {
        mine = ((const P*)inp)[tid];
        ((P*)ptrs[0])[tid] = mine;  // (rank + 0) % ngpus == rank
      }
    }
    bar_start<ngpus, FUSED>(sg, self_sg, rank, bmode);
    if (active) {
      A acc;
      if constexpr (FUSED) acc = upcast(mine);
      else acc = upcast(ptrs[0][tid]);
#pragma unroll
      for (int i = 1; i < ngpus; i++) packed_assign_add(acc, upcast(ptrs[i][tid]));
      ((P*)result)[tid] = downcast<P>(acc);
    }
  } else {
    if constexpr (FUSED) {
      P* self_buf = (P*)ptrs[0];
      const P* src = (const P*)inp;
      for (int idx = tid; idx < size; idx += stride) self_buf[idx] = src[idx];
    }
    bar_start<ngpus, FUSED>(sg, self_sg, rank, bmode);
    for (int idx = tid; idx < size; idx += stride)
      ((P*)result)[idx] = packed_reduce<P, ngpus, A>(ptrs, idx);
  }
  if constexpr (ENDBAR) bar_end<ngpus, true>(sg, self_sg, rank, bmode);
}

// ---------------------------------------------------------------------------
// Push one-shot. Same shape as ar_oneshot (stage, one barrier, reduce) with the
// data movement reversed: instead of every rank reading every peer's staged
// copy, every rank *writes* its input into every peer's slot and then reduces
// ngpus slots that are all local. Remote writes are posted, so they pipeline
// without the per-request round trip that caps peer reads.
//
// Per-block barrier still suffices: block b of rank q writes exactly the index
// slice that block b of rank r reads, because both use the same grid stride
// under identical geometry.
// ---------------------------------------------------------------------------
template <typename T, int ngpus, bool ENDBAR>
__global__ void __launch_bounds__(1024, 1)
ar_oneshot_push(RankData* _dp, RankSignals sg, Signal* self_sg,
                const T* __restrict__ inp, T* __restrict__ result, int rank,
                int size, int64_t off, int64_t slot, int bmode) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  // Where our contribution goes in each rank's buffer, ours included.
  P* dst[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    int t = (rank + i) % ngpus;
    dst[i] = ((P*)_dp->ptrs[t]) + off + (int64_t)rank * slot;
  }
  const P* src = (const P*)inp;
  for (int idx = tid; idx < size; idx += stride) {
    P v = src[idx];
#pragma unroll
    for (int i = 0; i < ngpus; i++) dst[i][idx] = v;
  }
  bar_start<ngpus, true>(sg, self_sg, rank, bmode);
  // Every contribution is now in our own buffer; the reduce is all-local.
  const P* mine = ((const P*)_dp->ptrs[rank]) + off;
  for (int idx = tid; idx < size; idx += stride) {
    A acc = upcast(mine[idx]);
#pragma unroll
    for (int i = 1; i < ngpus; i++)
      packed_assign_add(acc, upcast(mine[(int64_t)i * slot + idx]));
    ((P*)result)[idx] = downcast<P>(acc);
  }
  if constexpr (ENDBAR) bar_end<ngpus, true>(sg, self_sg, rank, bmode);
}

// ---------------------------------------------------------------------------
// Two-stage (reduce-scatter + all-gather). Bandwidth path: moves 1.5x the
// message in remote bytes per rank instead of one-shot's 3x.
//
// The fused copy walks the ngpus rank segments in turn, each with the same
// `tid`-based grid stride that stage 1 reads its own segment with. That is what
// keeps the *per-block* barrier sufficient: block b of every rank writes
// exactly the indices block b of every rank later reads.
// ---------------------------------------------------------------------------
template <typename T, int ngpus, bool FUSED>
__global__ void __launch_bounds__(1024, 1)
ar_twostage(RankData* _dp, RankSignals sg, Signal* self_sg,
            const T* __restrict__ inp, T* __restrict__ result, int rank,
            int size, int bmode) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const int part = size / ngpus;
  const int seg_start = rank * part;
  const int seg_end = rank == ngpus - 1 ? size : seg_start + part;
  const int largest_part = part + size % ngpus;
  const P* ptrs[ngpus];
  P* tmps[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    int target = (rank + i) % ngpus;
    ptrs[i] = (const P*)_dp->ptrs[target];
    tmps[i] = get_tmp_buf<P>(sg.signals[target]);
  }
  if constexpr (FUSED) {
    P* self_buf = (P*)ptrs[0];
    const P* src = (const P*)inp;
#pragma unroll
    for (int r = 0; r < ngpus; r++) {
      int s = r * part;
      int e = (r == ngpus - 1) ? size : s + part;
      for (int idx = s + tid; idx < e; idx += stride) self_buf[idx] = src[idx];
    }
  }
  auto tmp_out = tmps[0];
  bar_start<ngpus, FUSED>(sg, self_sg, rank, bmode);
  for (int idx = seg_start + tid; idx < seg_end; idx += stride)
    tmp_out[idx - seg_start] = packed_reduce<P, ngpus, A>(ptrs, idx);
  bar_end<ngpus, false>(sg, self_sg, rank, bmode);
  for (int idx = tid; idx < largest_part; idx += stride) {
#pragma unroll
    for (int i = 0; i < ngpus; i++) {
      int gather_from_rank = ((rank + i) % ngpus);
      if (gather_from_rank == ngpus - 1 || idx < part) {
        int dst_idx = gather_from_rank * part + idx;
        ((P*)result)[dst_idx] = tmps[i][idx];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// IPC handle type
// ---------------------------------------------------------------------------
using IPC_KEY = std::array<uint8_t, sizeof(cudaIpcMemHandle_t)>;
static_assert(sizeof(IPC_KEY) == sizeof(cudaIpcMemHandle_t));
static_assert(alignof(IPC_KEY) == alignof(cudaIpcMemHandle_t));

CUpointer_attribute rangeStartAddrAttr = CU_POINTER_ATTRIBUTE_RANGE_START_ADDR;

// Runtime-tunable geometry / algorithm selection. Defaults are the values the
// sweeps in ITERATIONS.md settled on; `set_tune` lets a sweep override them.
struct Tuning {
  // one-shot <-> two-stage crossover, in bytes, for the fused (eager) path.
  int64_t oneshot_max_bytes = 512 * 1024;
  // same crossover for the registered (CUDA-graph) path.
  int64_t oneshot_max_bytes_reg = 512 * 1024;
  int oneshot_threads = 512;
  int oneshot_blocks = 36;
  int twostage_threads = 512;
  int twostage_blocks = 36;
  // 0 = auto, 1 = force one-shot, 2 = force two-stage
  int force_algo = 0;
  // Alternate between two dedicated staging slots so the one-shot path needs no
  // closing barrier (see file header). The slots live in their own tail region
  // of the staging allocation so they can never overlap the offset-0 region the
  // two-stage path (which keeps its own closing barrier) writes.
  int double_buffer = 1;
  int64_t oneshot_region_bytes = 4 * 1024 * 1024;
  // 1 = use the register-held one-shot when one unit per thread covers the
  // message; 0 = always take the grid-stride form.
  int oneshot_regmode = 1;
  // Barrier spin form; see spin_flag.
  int barrier_mode = 0;
  // 1 = use the push one-shot instead of the pull one-shot.
  int push = 0;
};

// ---------------------------------------------------------------------------
// C++ CustomAllreduce class
// ---------------------------------------------------------------------------
class CustomAllreduce {
 public:
  int rank_;
  int world_size_;
  bool fully_connected_;

  RankSignals sg_;
  std::unordered_map<void*, RankData*> buffers_;
  Signal* self_sg_;

  RankData *d_rank_data_base_, *d_rank_data_end_;
  std::vector<void*> graph_unreg_buffers_;
  std::map<IPC_KEY, char*> ipc_handles_;

  Tuning tune_;
  // Bytes at the start of the staging allocation that every non-double-buffered
  // path may write (i.e. the caller's max_size). The one-shot slots are carved
  // out of whatever is above it, so a retune of oneshot_region_bytes can never
  // make them overlap. Set once from Python; deliberately not part of Tuning.
  int64_t reserved_bytes_ = 0;
  // Base of the push region and per-(rank,parity) slot stride,
  // both in bytes. Set once from Python; sized so the region can
  // never overlap anything below it.
  int64_t push_base_ = 0;
  int64_t push_slot_ = 0;
  // Counts only the double-buffered one-shot calls; its parity picks the
  // staging slot. All ranks issue the same call sequence (it is a collective)
  // and make the same size-based decisions, so the parities agree.
  uint64_t db_seq_ = 0;

  CustomAllreduce(Signal** signals, void* rank_data, size_t rank_data_sz,
                  int rank, int world_size, bool fully_connected = true)
      : rank_(rank),
        world_size_(world_size),
        fully_connected_(fully_connected),
        self_sg_(signals[rank]),
        d_rank_data_base_(reinterpret_cast<RankData*>(rank_data)),
        d_rank_data_end_(d_rank_data_base_ + rank_data_sz / sizeof(RankData)) {
    for (int i = 0; i < world_size_; i++) {
      sg_.signals[i] = signals[i];
    }
  }

  char* open_ipc_handle(const void* ipc_handle) {
    auto [it, new_handle] =
        ipc_handles_.insert({*((IPC_KEY*)ipc_handle), nullptr});
    if (new_handle) {
      char* ipc_ptr;
      CUDACHECK(cudaIpcOpenMemHandle((void**)&ipc_ptr,
                                     *((const cudaIpcMemHandle_t*)ipc_handle),
                                     cudaIpcMemLazyEnablePeerAccess));
      it->second = ipc_ptr;
    }
    return it->second;
  }

  std::pair<std::string, std::vector<int64_t>> get_graph_buffer_ipc_meta() {
    auto num_buffers = graph_unreg_buffers_.size();
    auto handle_sz = sizeof(cudaIpcMemHandle_t);
    std::string handles(handle_sz * num_buffers, static_cast<char>(0));
    std::vector<int64_t> offsets(num_buffers);
    for (size_t i = 0; i < num_buffers; i++) {
      auto ptr = graph_unreg_buffers_[i];
      void* base_ptr;
      if (cuPointerGetAttribute(&base_ptr, rangeStartAddrAttr,
                                (CUdeviceptr)ptr) != CUDA_SUCCESS)
        throw std::runtime_error("failed to get pointer attr");
      CUDACHECK(cudaIpcGetMemHandle(
          (cudaIpcMemHandle_t*)&handles[i * handle_sz], base_ptr));
      offsets[i] = ((char*)ptr) - ((char*)base_ptr);
    }
    return std::make_pair(handles, offsets);
  }

  void check_rank_data_capacity(size_t num = 1) {
    if (d_rank_data_base_ + num > d_rank_data_end_)
      throw std::runtime_error(
          "Rank data buffer overflow by " +
          std::to_string(d_rank_data_base_ + num - d_rank_data_end_));
  }

  void register_buffer(void** ptrs) {
    check_rank_data_capacity();
    RankData data;
    for (int i = 0; i < world_size_; i++) {
      data.ptrs[i] = ptrs[i];
    }
    auto d_data = d_rank_data_base_++;
    CUDACHECK(
        cudaMemcpy(d_data, &data, sizeof(RankData), cudaMemcpyHostToDevice));
    buffers_[ptrs[rank_]] = d_data;
  }

  void register_graph_buffers(
      const std::vector<std::string>& handles,
      const std::vector<std::vector<int64_t>>& offsets) {
    auto num_buffers = graph_unreg_buffers_.size();
    check_rank_data_capacity(num_buffers);
    std::vector<RankData> rank_data(num_buffers);
    for (size_t i = 0; i < num_buffers; i++) {
      auto self_ptr = graph_unreg_buffers_[i];
      auto& rd = rank_data[i];
      for (int j = 0; j < world_size_; j++) {
        if (j != rank_) {
          char* handle =
              open_ipc_handle(&handles[j][i * sizeof(cudaIpcMemHandle_t)]);
          handle += offsets[j][i];
          rd.ptrs[j] = handle;
        } else {
          rd.ptrs[j] = self_ptr;
        }
      }
    }
    CUDACHECK(cudaMemcpy(d_rank_data_base_, rank_data.data(),
                         sizeof(RankData) * num_buffers,
                         cudaMemcpyHostToDevice));
    d_rank_data_base_ += num_buffers;
    graph_unreg_buffers_.clear();
  }

  // `staged` is the IPC staging base for the eager path (input is an arbitrary
  // torch tensor that peers cannot see); pass nullptr on the registered
  // CUDA-graph path, where `input` itself is the registered buffer.
  template <typename T>
  void allreduce(cudaStream_t stream, const T* input, T* output, int size,
                 void* staged, int64_t staged_capacity_bytes) {
    constexpr int d = packed_t<T>::P::size;
    if (size % d != 0)
      throw std::runtime_error(
          "custom allreduce requires input length to be multiple of " +
          std::to_string(d));

    bool fused = staged != nullptr;

    cudaStreamCaptureStatus status;
    CUDACHECK(cudaStreamIsCapturing(stream, &status));
    const bool capturing = status == cudaStreamCaptureStatusActive;
    if (capturing && fused) {
      // A captured call cannot use the fused form: the staging half is picked
      // by a host counter, which would be frozen into the graph and reused by
      // every replay. Fall back to the pre-fusion shape -- an explicit staging
      // copy, then a kernel that reads the registered buffers -- so the graph
      // records both nodes and stays correct. (The engine's graph path uses
      // capture()/registered=True and never lands here.)
      CUDACHECK(cudaMemcpyAsync(staged, input, (size_t)size * sizeof(T),
                                cudaMemcpyDeviceToDevice, stream));
      input = reinterpret_cast<const T*>(staged);
      fused = false;
    }

    RankData* ptrs;
    if (capturing) {
      ptrs = d_rank_data_base_ + graph_unreg_buffers_.size();
      graph_unreg_buffers_.push_back((void*)input);
    } else {
      const void* rank_key = fused ? staged : (const void*)input;
      auto it = buffers_.find((void*)rank_key);
      if (it == buffers_.end())
        throw std::runtime_error(
            "buffer address " +
            std::to_string(reinterpret_cast<uint64_t>(rank_key)) +
            " is not registered!");
      ptrs = it->second;
    }

    const int64_t bytes = (int64_t)size * sizeof(T);
    size /= d;  // now in 16-byte packed units

    bool one_shot;
    if (tune_.force_algo == 1) {
      one_shot = true;
    } else if (tune_.force_algo == 2) {
      one_shot = false;
    } else if (world_size_ == 2) {
      one_shot = true;
    } else {
      one_shot = bytes < (fused ? tune_.oneshot_max_bytes
                                : tune_.oneshot_max_bytes_reg);
    }

    // Two dedicated staging slots let the one-shot path skip its closing
    // barrier. They sit in the tail of the allocation, disjoint from the
    // offset-0 region used by every other path.
    int64_t off = 0;
    bool endbar = true;
    if (fused && one_shot && tune_.double_buffer && !tune_.push) {
      int64_t region = tune_.oneshot_region_bytes;
      const int64_t avail = (staged_capacity_bytes - reserved_bytes_) / 2;
      if (region > avail) region = (avail / 16) * 16;
      const int64_t db_base = staged_capacity_bytes - 2 * region;
      if (region > 0 && bytes <= region && db_base >= reserved_bytes_) {
        off = (db_base + ((db_seq_ & 1ULL) ? region : 0)) / 16;
        endbar = false;
        db_seq_++;
      }
    }

    // Push one-shot: every rank's contribution lands in every rank's buffer, so
    // the region needs ngpus slots (times two for the alternating parity that
    // lets the closing barrier go away).
    bool push = false;
    int64_t push_slot_units = 0;
    if (fused && one_shot && tune_.push && push_slot_ > 0 &&
        bytes <= push_slot_ && push_base_ >= reserved_bytes_) {
      push = true;
      push_slot_units = push_slot_ / 16;
      const int64_t parity_stride = (int64_t)world_size_ * push_slot_;
      off = (push_base_ + ((db_seq_ & 1ULL) ? parity_stride : 0)) / 16;
      endbar = false;
      db_seq_++;
    }

    const int threads = one_shot ? tune_.oneshot_threads : tune_.twostage_threads;
    const int limit = std::min(one_shot ? tune_.oneshot_blocks
                                        : tune_.twostage_blocks, kMaxBlocks);
    int blocks = std::min(limit, (size + threads - 1) / threads);
    if (blocks < 1) blocks = 1;
    const bool regmode =
        tune_.oneshot_regmode && (int64_t)blocks * threads >= size;
    const int bmode = tune_.barrier_mode;

#define LAUNCH_ONESHOT_R(ngpus, FUSED, ENDBAR, REG)                       \
  ar_oneshot<T, ngpus, FUSED, ENDBAR, REG><<<blocks, threads, 0, stream>>>( \
      ptrs, sg_, self_sg_, input, output, rank_, size, off, bmode)
#define LAUNCH_ONESHOT_PUSH(ngpus, ENDBAR)                                \
  ar_oneshot_push<T, ngpus, ENDBAR><<<blocks, threads, 0, stream>>>(      \
      ptrs, sg_, self_sg_, input, output, rank_, size, off,               \
      push_slot_units, bmode)
#define LAUNCH_ONESHOT(ngpus, FUSED, ENDBAR)                              \
  do {                                                                    \
    if (FUSED && push) {                                                   \
      if (ENDBAR) LAUNCH_ONESHOT_PUSH(ngpus, true);                        \
      else        LAUNCH_ONESHOT_PUSH(ngpus, false);                       \
    } else if (regmode) LAUNCH_ONESHOT_R(ngpus, FUSED, ENDBAR, true);     \
    else         LAUNCH_ONESHOT_R(ngpus, FUSED, ENDBAR, false);           \
  } while (0)
#define LAUNCH_TWOSTAGE(ngpus, FUSED)                                     \
  ar_twostage<T, ngpus, FUSED><<<blocks, threads, 0, stream>>>(           \
      ptrs, sg_, self_sg_, input, output, rank_, size, bmode)

#define REDUCE_CASE(ngpus)                          \
  case ngpus: {                                     \
    if (one_shot) {                                 \
      if (fused) {                                  \
        if (endbar) LAUNCH_ONESHOT(ngpus, true, true);   \
        else        LAUNCH_ONESHOT(ngpus, true, false);  \
      } else {                                      \
        LAUNCH_ONESHOT(ngpus, false, true);         \
      }                                             \
    } else {                                        \
      if (fused) LAUNCH_TWOSTAGE(ngpus, true);      \
      else       LAUNCH_TWOSTAGE(ngpus, false);     \
    }                                               \
    break;                                          \
  }

    switch (world_size_) {
      REDUCE_CASE(2)
      REDUCE_CASE(4)
      REDUCE_CASE(6)
      REDUCE_CASE(8)
      default:
        throw std::runtime_error(
            "custom allreduce only supports world_size in {2,4,6,8}, got " +
            std::to_string(world_size_));
    }
    // A rejected launch (e.g. a thread count whose register demand does not
    // fit) must not pass silently: the output buffer would be returned
    // uninitialized, and worse, no rank would enter the barrier. Surfacing it
    // lets the Python layer fall back to NCCL, which stays collective because
    // every rank rejects the same launch.
    CUDACHECK(cudaGetLastError());
#undef REDUCE_CASE
#undef LAUNCH_TWOSTAGE
#undef LAUNCH_ONESHOT
#undef LAUNCH_ONESHOT_R
#undef LAUNCH_ONESHOT_PUSH
  }

  ~CustomAllreduce() {
    for (auto [_, ptr] : ipc_handles_) {
      cudaIpcCloseMemHandle(ptr);
    }
  }
};

// ---------------------------------------------------------------------------
// Python-facing functions
// ---------------------------------------------------------------------------
fptr_t init_custom_ar(const std::vector<fptr_t>& fake_ipc_ptrs,
                      torch::Tensor& rank_data, int64_t rank,
                      bool fully_connected) {
  int world_size = fake_ipc_ptrs.size();
  if (world_size > 8)
    throw std::invalid_argument("world size > 8 is not supported");
  if (world_size % 2 != 0)
    throw std::invalid_argument("odd num gpus is not supported");
  if (rank < 0 || rank >= world_size)
    throw std::invalid_argument("invalid rank");

  Signal* ipc_ptrs[8];
  for (int i = 0; i < world_size; i++) {
    ipc_ptrs[i] = reinterpret_cast<Signal*>(fake_ipc_ptrs[i]);
  }
  return (fptr_t) new CustomAllreduce(ipc_ptrs, rank_data.data_ptr(),
                                      rank_data.numel(), rank, world_size,
                                      fully_connected);
}

void set_tune(fptr_t _fa, int64_t oneshot_max_bytes,
              int64_t oneshot_max_bytes_reg, int64_t oneshot_threads,
              int64_t oneshot_blocks, int64_t twostage_threads,
              int64_t twostage_blocks, int64_t force_algo,
              int64_t double_buffer, int64_t oneshot_region_bytes,
              int64_t oneshot_regmode, int64_t barrier_mode,
              int64_t push) {
  auto fa = reinterpret_cast<CustomAllreduce*>(_fa);
  fa->tune_.oneshot_max_bytes = oneshot_max_bytes;
  fa->tune_.oneshot_max_bytes_reg = oneshot_max_bytes_reg;
  fa->tune_.oneshot_threads = (int)oneshot_threads;
  fa->tune_.oneshot_blocks = (int)oneshot_blocks;
  fa->tune_.twostage_threads = (int)twostage_threads;
  fa->tune_.twostage_blocks = (int)twostage_blocks;
  fa->tune_.force_algo = (int)force_algo;
  fa->tune_.double_buffer = (int)double_buffer;
  fa->tune_.oneshot_region_bytes = oneshot_region_bytes;
  fa->tune_.oneshot_regmode = (int)oneshot_regmode;
  fa->tune_.barrier_mode = (int)barrier_mode;
  fa->tune_.push = (int)push;
}

int64_t max_blocks() { return kMaxBlocks; }

void set_layout(fptr_t _fa, int64_t reserved_bytes,
                int64_t push_base, int64_t push_slot) {
  auto fa = reinterpret_cast<CustomAllreduce*>(_fa);
  fa->reserved_bytes_ = reserved_bytes;
  fa->push_base_ = push_base;
  fa->push_slot_ = push_slot;
}

void all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out,
                fptr_t _reg_buffer, int64_t reg_buffer_sz_bytes) {
  auto fa = reinterpret_cast<CustomAllreduce*>(_fa);
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  auto input_size = inp.numel() * inp.element_size();
  auto reg_buffer = reinterpret_cast<void*>(_reg_buffer);
  if (reg_buffer) {
    TORCH_CHECK_LE(input_size, reg_buffer_sz_bytes);
  }
  switch (out.scalar_type()) {
    case at::ScalarType::Float:
      fa->allreduce<float>(stream,
                           reinterpret_cast<const float*>(inp.data_ptr()),
                           reinterpret_cast<float*>(out.data_ptr()),
                           out.numel(), reg_buffer, reg_buffer_sz_bytes);
      break;
    case at::ScalarType::Half:
      fa->allreduce<half>(stream,
                          reinterpret_cast<const half*>(inp.data_ptr()),
                          reinterpret_cast<half*>(out.data_ptr()),
                          out.numel(), reg_buffer, reg_buffer_sz_bytes);
      break;
    case at::ScalarType::BFloat16:
      fa->allreduce<nv_bfloat16>(
          stream, reinterpret_cast<const nv_bfloat16*>(inp.data_ptr()),
          reinterpret_cast<nv_bfloat16*>(out.data_ptr()), out.numel(),
          reg_buffer, reg_buffer_sz_bytes);
      break;
    default:
      throw std::runtime_error(
          "custom allreduce only supports float32, float16 and bfloat16");
  }
}

void dispose(fptr_t _fa) {
  delete reinterpret_cast<CustomAllreduce*>(_fa);
}

int64_t meta_size() { return sizeof(Signal); }

void register_buffer(fptr_t _fa, const std::vector<fptr_t>& fake_ipc_ptrs) {
  auto fa = reinterpret_cast<CustomAllreduce*>(_fa);
  TORCH_CHECK(static_cast<int>(fake_ipc_ptrs.size()) == fa->world_size_);
  void* ipc_ptrs[8];
  for (size_t i = 0; i < fake_ipc_ptrs.size(); i++) {
    ipc_ptrs[i] = reinterpret_cast<void*>(fake_ipc_ptrs[i]);
  }
  fa->register_buffer(ipc_ptrs);
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>>
get_graph_buffer_ipc_meta(fptr_t _fa) {
  auto fa = reinterpret_cast<CustomAllreduce*>(_fa);
  auto [handle, offsets] = fa->get_graph_buffer_ipc_meta();
  std::vector<int64_t> bytes(handle.begin(), handle.end());
  return std::make_tuple(bytes, offsets);
}

void register_graph_buffers(fptr_t _fa,
                            const std::vector<std::vector<int64_t>>& handles,
                            const std::vector<std::vector<int64_t>>& offsets) {
  auto fa = reinterpret_cast<CustomAllreduce*>(_fa);
  std::vector<std::string> bytes;
  bytes.reserve(handles.size());
  for (size_t i = 0; i < handles.size(); i++) {
    bytes.emplace_back(handles[i].begin(), handles[i].end());
  }
  fa->register_graph_buffers(bytes, offsets);
}

std::tuple<fptr_t, torch::Tensor> allocate_shared_buffer_and_handle(
    int64_t size) {
  auto device_index = c10::cuda::current_device();
  at::DeviceGuard device_guard(at::Device(at::DeviceType::CUDA, device_index));
  void* buffer;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  CUDACHECK(cudaMalloc((void**)&buffer, size));
  CUDACHECK(cudaMemsetAsync(buffer, 0, size, stream));
  CUDACHECK(cudaStreamSynchronize(stream));
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  auto options =
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
  auto handle =
      torch::empty({static_cast<int64_t>(sizeof(cudaIpcMemHandle_t))}, options);
  CUDACHECK(
      cudaIpcGetMemHandle((cudaIpcMemHandle_t*)handle.data_ptr(), buffer));

  return std::make_tuple(reinterpret_cast<fptr_t>(buffer), handle);
}

fptr_t open_mem_handle(torch::Tensor& mem_handle) {
  void* ipc_ptr;
  CUDACHECK(cudaIpcOpenMemHandle(
      (void**)&ipc_ptr, *((const cudaIpcMemHandle_t*)mem_handle.data_ptr()),
      cudaIpcMemLazyEnablePeerAccess));
  return reinterpret_cast<fptr_t>(ipc_ptr);
}

void free_shared_buffer(fptr_t buffer) {
  CUDACHECK(cudaFree(reinterpret_cast<void*>(buffer)));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init_custom_ar", &init_custom_ar);
  m.def("all_reduce", &all_reduce);
  m.def("dispose", &dispose);
  m.def("meta_size", &meta_size);
  m.def("max_blocks", &max_blocks);
  m.def("set_tune", &set_tune);
  m.def("set_layout", &set_layout);
  m.def("register_buffer", &register_buffer);
  m.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta);
  m.def("register_graph_buffers", &register_graph_buffers);
  m.def("allocate_shared_buffer_and_handle",
        &allocate_shared_buffer_and_handle);
  m.def("open_mem_handle", &open_mem_handle);
  m.def("free_shared_buffer", &free_shared_buffer);
}
