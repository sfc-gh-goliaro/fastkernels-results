// Two-tensor dim=1 concatenation for contiguous NCHW fp16, in ONE launch.
//
// What the op is, mechanically.  For contiguous NCHW inputs a[N,Ca,H,W] and
// b[N,Cb,H,W] concatenated on dim 1, the output plane for sample n is
// [a-slab(n) | b-slab(n)] and every slab is itself contiguous.  So the whole
// operator is a batched pair of contiguous runs -- there is no per-element
// index decomposition to do, which is exactly what aten's generic
// CatArrayBatchedCopy pays for (it divmods a flat output index against the
// output plane extent for every element it moves).
//
// The mapping used here is division-free:
//
//     blockIdx.y  = n                       (sample)
//     j           = vector within the output plane, [0, VPS)
//     dst         = out + n*VPS + j
//     src         = j < VA ? a + n*VA + j  :  b + n*VB + (j - VA)
//
// with all extents counted in fixed-width vectors.  Every captured shape has a
// per-sample a-slab that is a whole number of vectors *and* a multiple of the
// block's vector count, so the a/b boundary always falls on a block boundary:
// the `j < VA` select is warp-uniform, and there is no ragged tail (the EXACT
// specialisation drops the bounds test entirely).
//
// Why the launch, not the bytes, is the target.  The bench times a window that
// opens on a 265 MB L2 flush and contains the shifting pool's two input D2D
// memcpys before the op runs.  nsys on [4,128,20,20]+[4,256,20,20]:
//
//   memset 68.1us | gap 1.54 | D2D 1.98 | gap 1.76 | D2D 2.14 | concat
//
// Consecutive stream operations are separated by a ~1.8 us GPU-side gap, and
// that gap is larger than the entire byte cost of these shapes (0.3-9.8 MB is
// 0.09-2.8 us of HBM).  `torch.cat` costs floor + 4.1 us: its own gap plus
// ~2.0 us of CatArrayBatchedCopy.  So:
//
//   1. One launch for both inputs.  Not two copies, not memcpys -- a second
//      stream op would cost more than the copy it performs.
//   2. Programmatic Dependent Launch, which is where essentially the whole win
//      comes from.  `cudaLaunchAttributeProgrammaticStreamSerialization` gets
//      this kernel's CTAs dispatched onto idle SMs during the preceding
//      operation's tail, deleting that ~1.8 us gap outright; measured with PDL
//      off, a hand-written single-launch vectorized concat is a wash with
//      `torch.cat` (geomean 0.97-1.02x).  The
//      `cudaGridDependencySynchronize()` sits after all address arithmetic and
//      before the first load of producer-written data: that wait is what makes
//      reading the producer's output safe, so it is not optional.
//
// What is left is the load side.  Isolating the two halves (store-only vs
// load-only variants of this kernel) shows the stores are absorbed -- they land
// dirty in L2 and are written back after the window -- while the loads must
// come from HBM in competition with the flush's writeback drain, at ~0.6 TB/s
// on the smallest shapes rising to ~2.4 TB/s on the 9.8 MB one.  That read cost
// is the residual, it is the same read cost `torch.cat` pays, and no block
// shape, vector width or cache policy moves it (all swept, all neutral).
//
// Block shape, vector width and cache policy are chosen on the host and pinned
// by `set_cfg` for sweeps.  Note that on sm_100 the `.L2::evict_first`
// eviction-priority modifier is only legal on 32-byte (`.v8.b32`) accesses,
// which is why the wide path exists at all.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>

namespace {

// ---------------------------------------------------------------------------
// PDL plumbing.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void cat2_pdl_wait() {
#if __CUDA_ARCH__ >= 900
  cudaGridDependencySynchronize();
#endif
}

int env_int(const char* name, int dflt) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0') return dflt;
  return std::atoi(v);
}

bool pdl_supported() {
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return false;
  int major = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev)
      != cudaSuccess)
    return false;
  return major >= 9;
}

int sm_count() {
  int dev = 0, n = 148;
  if (cudaGetDevice(&dev) != cudaSuccess) return n;
  cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
  return n;
}

// Resolved once at module load so the hot path carries no guards. The pins are
// mutable only so `set_cfg` can sweep them; the shipped path leaves them at
// their env defaults.
int kPdl = env_int("FK_CAT2_PDL", 1) && pdl_supported();
const int kSMs = sm_count();
int kBlockPin = env_int("FK_CAT2_BLOCK", 0);  // 0 = auto
int kVptPin = env_int("FK_CAT2_VPT", 0);      // 0 = auto
int kWidthPin = env_int("FK_CAT2_W", 0);      // 0 = auto, else 2/16/32
// Cache policy: `L2::evict_first` on both sides. Neither the inputs (the
// producer just wrote them) nor the output is re-read, and on the one
// bandwidth-bound shape it is worth 6% of kernel time (3.36 -> 3.14 us);
// elsewhere it measures neutral. Legal only on the 32 B path on sm_100, so the
// 16 B and 2 B paths silently fall back to plain accesses.
int kHint = env_int("FK_CAT2_HINT", 4);

// ---------------------------------------------------------------------------
// The move.  W is the payload width in bytes; HINT selects the cache policy.
// HINT 0 plain, 1 non-coherent load, 2 write-through store, 3 both,
// 4 evict-first on both sides (32 B only -- sm_100 rejects the modifier on
// narrower accesses).
// ---------------------------------------------------------------------------
template <int W, int HINT>
struct Move;

template <int HINT>
struct Move<2, HINT> {
  __device__ __forceinline__ static void go(const void* s, void* d) {
    *static_cast<ushort*>(d) = *static_cast<const ushort*>(s);
  }
};

template <int HINT>
struct Move<16, HINT> {
  __device__ __forceinline__ static void go(const void* s, void* d) {
    uint4 v;
    if (HINT == 1 || HINT == 3) {
      asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(s));
    } else {
      v = *static_cast<const uint4*>(s);
    }
    if (HINT == 2 || HINT == 3) {
      asm volatile("st.global.wt.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(d), "r"(v.x),
                   "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
    } else {
      *static_cast<uint4*>(d) = v;
    }
  }
};

// 32-byte accesses: sm_100 `.v8.b32`. Halves the instruction count of the move
// and is the only width on which the L2 eviction-priority hints are legal.
template <int HINT>
struct Move<32, HINT> {
  __device__ __forceinline__ static void go(const void* s, void* d) {
    uint4 lo, hi;
    if (HINT == 4) {
      asm volatile(
          "ld.global.nc.L2::evict_first.v8.b32 "
          "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
          : "=r"(lo.x), "=r"(lo.y), "=r"(lo.z), "=r"(lo.w), "=r"(hi.x),
            "=r"(hi.y), "=r"(hi.z), "=r"(hi.w) : "l"(s));
      asm volatile(
          "st.global.L2::evict_first.v8.b32 [%0], "
          "{%1,%2,%3,%4,%5,%6,%7,%8};" ::"l"(d),
          "r"(lo.x), "r"(lo.y), "r"(lo.z), "r"(lo.w), "r"(hi.x), "r"(hi.y),
          "r"(hi.z), "r"(hi.w) : "memory");
      return;
    }
    if (HINT == 1 || HINT == 3) {
      asm volatile("ld.global.nc.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                   : "=r"(lo.x), "=r"(lo.y), "=r"(lo.z), "=r"(lo.w),
                     "=r"(hi.x), "=r"(hi.y), "=r"(hi.z), "=r"(hi.w) : "l"(s));
    } else {
      asm volatile("ld.global.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                   : "=r"(lo.x), "=r"(lo.y), "=r"(lo.z), "=r"(lo.w),
                     "=r"(hi.x), "=r"(hi.y), "=r"(hi.z), "=r"(hi.w) : "l"(s));
    }
    if (HINT == 2 || HINT == 3) {
      asm volatile("st.global.wt.v8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::"l"(
                       d),
                   "r"(lo.x), "r"(lo.y), "r"(lo.z), "r"(lo.w), "r"(hi.x),
                   "r"(hi.y), "r"(hi.z), "r"(hi.w) : "memory");
    } else {
      asm volatile("st.global.v8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::"l"(d),
                   "r"(lo.x), "r"(lo.y), "r"(lo.z), "r"(lo.w), "r"(hi.x),
                   "r"(hi.y), "r"(hi.z), "r"(hi.w) : "memory");
    }
  }
};

// ---------------------------------------------------------------------------
// The kernel. All extents are counted in W-byte vectors.
// ---------------------------------------------------------------------------
template <int W, int BLOCK, int VPT, bool EXACT, int HINT>
__global__ __launch_bounds__(BLOCK) void cat2_kernel(
    const char* __restrict__ a, const char* __restrict__ b,
    char* __restrict__ out, int va, int vb, int vps) {
  const int n = blockIdx.y;
  const int j0 = blockIdx.x * (BLOCK * VPT) + threadIdx.x;
  // Base vector offsets: one multiply each, hoisted out of the unrolled body.
  const int oa = n * va;
  const int ob = n * vb - va;
  char* __restrict__ dst = out + (int64_t)n * vps * W;

  cat2_pdl_wait();

#pragma unroll
  for (int k = 0; k < VPT; ++k) {
    const int j = j0 + k * BLOCK;
    if (EXACT || j < vps) {
      const char* __restrict__ s = (j < va) ? a : b;
      const int o = (j < va) ? (oa + j) : (ob + j);
      Move<W, HINT>::go(s + (int64_t)o * W, dst + (int64_t)j * W);
    }
  }
}

template <typename K, typename... Args>
inline void launch(K kern, dim3 grid, int block, cudaStream_t stream,
                   Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(block);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  if (kPdl) {
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
  }
  cudaLaunchKernelEx(&cfg, kern, args...);
}

struct Job {
  const char* a;
  const char* b;
  char* out;
  int va, vb, vps, n;      // extents in vectors
  cudaStream_t stream;
};

template <int W, int BLOCK, int VPT, int HINT>
void launch_4(const Job& j) {
  const int per = BLOCK * VPT;
  const dim3 grid((j.vps + per - 1) / per, j.n);
  if (j.vps % per == 0)
    launch(cat2_kernel<W, BLOCK, VPT, true, HINT>, grid, BLOCK, j.stream, j.a,
           j.b, j.out, j.va, j.vb, j.vps);
  else
    launch(cat2_kernel<W, BLOCK, VPT, false, HINT>, grid, BLOCK, j.stream, j.a,
           j.b, j.out, j.va, j.vb, j.vps);
}

template <int W, int BLOCK, int VPT>
void launch_3(const Job& j) {
  switch (kHint) {
    case 1: launch_4<W, BLOCK, VPT, 1>(j); return;
    case 2: launch_4<W, BLOCK, VPT, 2>(j); return;
    case 3: launch_4<W, BLOCK, VPT, 3>(j); return;
    case 4:
      if (W == 32) { launch_4<W, BLOCK, VPT, 4>(j); return; }
      break;
    default: break;
  }
  launch_4<W, BLOCK, VPT, 0>(j);
}

template <int W>
void launch_2(const Job& j, int block, int vpt) {
#define CAT2_CASE(B, P)                            \
  if (block == (B) && vpt == (P)) {                \
    launch_3<W, B, P>(j);                          \
    return;                                        \
  }
  CAT2_CASE(256, 1)
  CAT2_CASE(256, 2)
  CAT2_CASE(256, 4)
  CAT2_CASE(128, 1)
  CAT2_CASE(128, 2)
  CAT2_CASE(128, 4)
  CAT2_CASE(64, 1)
  CAT2_CASE(64, 2)
  CAT2_CASE(64, 4)
  CAT2_CASE(512, 1)
  CAT2_CASE(512, 2)
  CAT2_CASE(512, 4)
  CAT2_CASE(1024, 1)
  CAT2_CASE(1024, 2)
#undef CAT2_CASE
  launch_3<W, 256, 1>(j);
}

// Grid shape.  The kernel is one load + one store per vector, so its duration
// is memory latency plus whatever share of HBM it can win; what matters is
// having enough blocks resident to cover that latency.  Sweeping block x vpt
// showed the choice is flat *except* when the grid falls under the SM count --
// e.g. 256 threads x 2 vectors on the 0.3 MB shape is 38 blocks on 148 SMs and
// costs a whole plateau.  So: fix a 128-thread block (>= 150 blocks for every
// captured shape) and only raise the per-thread vector count once the grid is
// several waves deep anyway.
void pick_grid(int vps, int n, int* block, int* vpt) {
  const int b = kBlockPin ? kBlockPin : 128;
  int v = 1;
  if (kVptPin) {
    v = kVptPin;
  } else {
    // One vector per thread maximises memory-level parallelism, so only fold
    // work into threads while doing so still leaves ~8 waves resident. Profiled
    // kernel time on the 9.8 MB shape: 2400 blocks 3.42 us -> 1200 blocks
    // 3.14 us -> 600 blocks 3.38 us; on the 1.2 MB shape 300 blocks is 1.79 us
    // and folding to 152 costs 2.18 us.
    while (v < 4 &&
           (int64_t)((vps + b * v * 2 - 1) / (b * v * 2)) * n >= (int64_t)kSMs * 8)
      v *= 2;
  }
  *block = b;
  *vpt = v;
}

}  // namespace

// Concatenate two tensors along dim 1. Falls back to at::cat for anything the
// fast path does not cover, so the Python gate can stay at two cheap checks.
at::Tensor cat2(const at::Tensor& a, const at::Tensor& b) {
  const int64_t nd = a.dim();
  if (!(a.is_cuda() && b.is_cuda()) || a.scalar_type() != b.scalar_type() ||
      a.element_size() != 2 || nd < 2 || b.dim() != nd || !a.is_contiguous() ||
      !b.is_contiguous() || a.requires_grad() || b.requires_grad()) {
    return at::cat({a, b}, 1);
  }
  for (int64_t d = 0; d < nd; ++d) {
    if (d != 1 && a.size(d) != b.size(d)) return at::cat({a, b}, 1);
  }

  const int64_t n = a.size(0);
  // Stack buffer, not a std::vector: this runs on every call and a heap
  // allocation per concat is pure host overhead. Ranks above 8 are not a thing
  // for this op, but fall back rather than assume it.
  if (nd > 8) return at::cat({a, b}, 1);
  int64_t shape[8];
  for (int64_t d = 0; d < nd; ++d) shape[d] = a.size(d);
  shape[1] = a.size(1) + b.size(1);
  at::Tensor out = at::empty(at::IntArrayRef(shape, nd), a.options());
  if (out.numel() == 0 || n == 0) return out;

  Job j;
  j.a = static_cast<const char*>(a.const_data_ptr());
  j.b = static_cast<const char*>(b.const_data_ptr());
  j.out = static_cast<char*>(out.data_ptr());
  j.n = (int)n;
  j.stream = at::cuda::getCurrentCUDAStream();

  // Per-sample slab sizes in bytes.
  const int64_t ba = (a.numel() / n) * 2;
  const int64_t bb = (b.numel() / n) * 2;
  const uintptr_t bits = reinterpret_cast<uintptr_t>(j.a) |
                         reinterpret_cast<uintptr_t>(j.b) |
                         reinterpret_cast<uintptr_t>(j.out);

  int width = 2;
  if ((bits & 31) == 0 && (ba % 32) == 0 && (bb % 32) == 0) width = 32;
  else if ((bits & 15) == 0 && (ba % 16) == 0 && (bb % 16) == 0) width = 16;
  if (kWidthPin == 2 || kWidthPin == 16 || kWidthPin == 32) {
    // A pin is only honoured when it is actually legal for these operands.
    if (kWidthPin <= width || (kWidthPin == 16 && width == 32)) width = kWidthPin;
  }

  if (n > 65535 || (ba + bb) / width * n >= 0x7fffffffLL) {
    return at::cat({a, b}, 1);
  }
  j.va = (int)(ba / width);
  j.vb = (int)(bb / width);
  j.vps = j.va + j.vb;

  int block, vpt;
  pick_grid(j.vps, j.n, &block, &vpt);
  if (width == 32) launch_2<32>(j, block, vpt);
  else if (width == 16) launch_2<16>(j, block, vpt);
  else launch_2<2>(j, block, vpt);
  return out;
}

// Sweep hook: pin block / vpt / PDL / payload width / cache hint. Zero on
// block, vpt or width restores the automatic choice.
void set_cfg(int block, int vpt, int pdl, int width, int hint) {
  kBlockPin = block;
  kVptPin = vpt;
  kPdl = pdl && pdl_supported();
  kWidthPin = width;
  kHint = hint;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cat2", &cat2, "Two-input dim=1 concat (2-byte dtypes, CUDA)");
  m.def("set_cfg", &set_cfg, "Pin (block, vpt, pdl, width, hint) for sweeps");
}
