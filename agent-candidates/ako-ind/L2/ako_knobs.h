// Per-call tuning knobs, shared between ako_bind.cpp (which pulls in no CUTLASS
// headers, so it compiles in seconds) and the CUTLASS config translation units.
#pragma once

namespace ako {

struct Knobs {
  int swizzle = 0;   // scheduler max_swizzle_size (0 = default)
  int raster = 0;    // 0 Heuristic, 1 AlongM, 2 AlongN
  int splits = 1;    // Split-K count (Stream-K scheduler only)
  int decomp = 0;    // 0 Heuristic, 1 SplitK, 2 StreamK, 3 DataParallel
};

}  // namespace ako
