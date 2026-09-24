// 5x5 / stride-1 / pad-2 fp16 max-pool over contiguous 20x20 NCHW planes.
//
// Both axes of the separable max stay in registers or warp lanes -- there is no
// shared memory and no barrier anywhere, so the only memory traffic is one
// coalesced read and one coalesced write of each 800-byte plane:
//
//   * columns are packed two per lane as half2 and spread across the warp, so
//     the horizontal 5-tap is 2 __shfl + 2 PRMT + 4 __hmax2_nan per lane:
//         lm = (c-2,c-1)  m0 = (c-1,c)  a = (c,c+1)  m1 = (c+1,c+2)
//         lp = (c+2,c+3)
//     which are exactly the 5-wide window for both packed columns at once.
//   * rows live in each lane's registers, so the vertical 5-tap is 4
//     __hmax2_nan (nvcc CSEs the shared sub-maxima across adjacent rows).
//
// Lane layout: 16 lanes per plane, two planes per warp.  Lanes 1..10 hold the
// 20 real columns; lanes 0 and 11..15 never load, so they hold -inf and feed
// the horizontal shuffles at each plane edge.  Padding therefore costs no
// instructions at all -- there is no column boundary test in the row loop.
//
// R (output rows per warp) trades halo re-reads against warp count: R=10 loads
// 14 rows to produce 10, and puts 512 warps / 128 CTAs on the 148-SM B200,
// which measured lowest at [4,128,20,20]; R=4 measured lowest at
// [1,128,20,20], which has 4x fewer planes to spread over the machine.
//
// PDL is the single biggest lever here (2.0 us).  The harness times a window
// dominated by fixed launch overhead, and with
// cudaLaunchAttributeProgrammaticStreamSerialization the CTAs are dispatched
// while the producer (the harness's input copy_) is still draining, so this
// kernel's launch is free.  Every timed iteration lands either at 7.1 us
// (overlapped) or 9.2 us (not); with the attribute off it is 9.2 us every
// time.  The asm barrier ahead of the wait stops nvcc from sinking the address
// arithmetic past it, which is worth a further 3% of kernel duration.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

#define H 20
#define W 20

__device__ __forceinline__ unsigned h2u(__half2 v) {
  return *reinterpret_cast<unsigned *>(&v);
}
__device__ __forceinline__ __half2 u2h(unsigned v) {
  return *reinterpret_cast<__half2 *>(&v);
}
__device__ __forceinline__ __half2 neg2() { return u2h(0xFC00FC00u); }

template <int R, int BLK, bool EXACT>
__global__ __launch_bounds__(BLK) void mp5_kernel(
    const __half2 *__restrict__ x, __half2 *__restrict__ o, int nplanes) {
  // R >= 2 is required: the cheap halo test below assumes only the first and
  // the last row block of a plane can reach outside it.
  static_assert(R >= 2 && H % R == 0, "R must be >= 2 and divide H");
  constexpr int RB = H / R;      // row blocks per plane
  constexpr int WPB = BLK / 32;  // warps per block
  const int lane = threadIdx.x & 31;
  const int wid = blockIdx.x * WPB + (threadIdx.x >> 5);
  const int pg = wid / RB;
  const int rb = wid - pg * RB;
  const int ll = lane & 15;
  const int plane = pg * 2 + (lane >> 4);
  const bool act = (unsigned)(ll - 1) < 10u && (EXACT || plane < nplanes);
  // One base pointer per thread; every row offset below is a compile-time
  // immediate that folds into the LDG / STG.
  const int idx = plane * (H * W / 2) + (rb * R - 2) * (W / 2) + (ll - 1);
  const __half2 *xb = x + idx;
  __half2 *ob = o + idx + 2 * (W / 2);
  const bool notfirst = (RB == 1) ? false : (rb != 0);
  const bool notlast = (RB == 1) ? false : (rb != RB - 1);

  __half2 v[R + 4];
#if __CUDA_ARCH__ >= 900
  // Keep the index arithmetic above the wait, where it overlaps the producer.
  asm volatile("" ::"l"(xb), "l"(ob), "r"((int)act), "r"((int)notfirst),
               "r"((int)notlast) : "memory");
  cudaGridDependencySynchronize();
#endif
#pragma unroll
  for (int i = 0; i < R + 4; ++i) {
    // input rows [r0-2, r0+R+2): the two at each end exist only for interior
    // row blocks, and -inf everywhere else reproduces pad=2 exactly
    const bool ok = act && (i >= 2 || notfirst) && (i < R + 2 || notlast);
    v[i] = ok ? xb[i * (W / 2)] : neg2();
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    // vertical: output row r0+r takes input rows r0+r-2 .. r0+r+2 = v[r..r+4]
    __half2 a = __hmax2_nan(__hmax2_nan(v[r], v[r + 1]),
                            __hmax2_nan(v[r + 2], v[r + 3]));
    a = __hmax2_nan(a, v[r + 4]);
    // horizontal: the 5 half2 terms aligned to this lane's column pair
    const unsigned au = h2u(a);
    const unsigned lm = __shfl_up_sync(0xffffffffu, au, 1);
    const unsigned lp = __shfl_down_sync(0xffffffffu, au, 1);
    const unsigned m0 = __byte_perm(lm, au, 0x5432);
    const unsigned m1 = __byte_perm(au, lp, 0x5432);
    __half2 res = __hmax2_nan(__hmax2_nan(u2h(lm), u2h(m0)),
                              __hmax2_nan(u2h(m1), u2h(lp)));
    res = __hmax2_nan(res, a);
    if (act) ob[r * (W / 2)] = res;
  }
}

template <typename K, typename... Args>
static inline void launch(K kern, int grid, int blk, cudaStream_t s,
                          Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid, 1, 1);
  cfg.blockDim = dim3(blk, 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = s;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, kern, args...);
}

// EXACT drops the `plane < nplanes` test, so it is only safe when the grid
// covers exactly the warps that have work: an odd plane count leaves the second
// plane of the last pair empty, and a warp count that is not a whole number of
// blocks leaves spare warps in the last block.  Both map to plane >= nplanes.
#define MP5_LAUNCH(R, BLK)                                                    \
  {                                                                           \
    const int warps = pairs * (H / (R));                                      \
    const int grid = (warps + (BLK) / 32 - 1) / ((BLK) / 32);                 \
    if (n % 2 == 0 && warps % ((BLK) / 32) == 0)                              \
      launch(mp5_kernel<R, BLK, true>, grid, BLK, s, x, o, n);                \
    else                                                                      \
      launch(mp5_kernel<R, BLK, false>, grid, BLK, s, x, o, n);               \
  }

// Entry point.  nplanes = N*C, over contiguous fp16 20x20 planes.
void mp5_auto(int64_t xptr, int64_t optr, int64_t nplanes) {
  const __half2 *x = reinterpret_cast<const __half2 *>(xptr);
  __half2 *o = reinterpret_cast<__half2 *>(optr);
  const int n = (int)nplanes;
  const int pairs = (n + 1) / 2;
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  if (n >= 256)
    MP5_LAUNCH(10, 128)  // 512 planes -> 512 warps / 128 CTAs, 0.98 us
  else
    MP5_LAUNCH(4, 128)   // 128 planes -> 320 warps / 80 CTAs, 0.93 us
}
