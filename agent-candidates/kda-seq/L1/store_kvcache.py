"""Vectorised store of key/value rows into an HND paged KV cache.

Layout: ``[num_blocks, num_kv_heads, page_size, head_dim]``.  Pure scatter-copy,
no arithmetic, so the only thing that decides latency is how many bytes each lane
moves and how much of the launch is overhead.

Three things are different from the Triton baseline:

* **16 bytes per lane instead of 2.**  The baseline launches ``(N, H)`` programs,
  each covering one ``head_dim`` row; with four warps that is one bfloat16 per
  lane per access.  Here a row is moved as ``head_dim / 8`` explicit ``uint4``
  units, so every load and store is 128 bits wide.  ``uint4`` is spelled out
  rather than left to a compiler because the destination address is
  data-dependent, and vectorising a scattered store is not something to infer
  from source that does not say it.
* **One kernel instead of two.**  The baseline casts ``slot_mapping`` whenever it
  is not already int64, which allocates and launches a second kernel; every
  captured call passes int32, so it pays this every time, and on the launch-bound
  shapes it is about half of the operator's GPU work.  The kernel here is
  templated on the mapping dtype and widens in register.
* **No division on the hot path.**  ``blockDim = (units_per_row, num_kv_heads,
  rows_per_block)`` takes the chunk, head and token indices straight from
  ``threadIdx``, leaving only the page divide, which is a shift whenever
  ``page_size`` is a power of two -- it is 16 in every captured call.

Keeping the head index inside the block, rather than on the grid as the baseline
does, is deliberate: a block then reads one contiguous run of
``num_kv_heads * head_dim`` source elements per token, and all of a token's head
rows land in one page's region, instead of reading ``head_dim``-sized runs and
scattering every token across ``num_kv_heads`` unrelated pages.  (A block may
still hold several tokens, whose slots are unrelated, so its writes are not
confined to one page overall.)

Preconditions the caller owns, none of which the baseline checks either:

* every non-negative slot is below ``num_blocks * page_size``.  The baseline does
  no upper-bound check and neither does this, so a positive out-of-range slot is
  an out-of-bounds device write here exactly as it is there.  The torch fallback
  will usually raise instead, so the behaviour is path-dependent -- that is
  inherited undefined behaviour, not a defined contract.
* slots are distinct.  Duplicates are a data race in both implementations.
* ``value.shape == key.shape``, and the two caches do not alias.

Inputs outside what the fast path can promise -- an odd head stride, a
non-contiguous cache, a head dim whose row is not a multiple of 16 bytes, a CPU
tensor -- fall through to a generic device kernel or to a layout-general torch
store, and the module records which one ran so a silent fallback cannot pass for
a result.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

_WORKSPACE = Path(__file__).resolve().parents[2]

#: Where the inline extension is built.  Workspace-local, so sibling operator
#: workspaces building their own extensions never share a directory; concurrent
#: bench workers inside this workspace are serialised by load_inline's own lock.
EXTENSION_DIR = _WORKSPACE / ".torch_extensions"

#: ``"cuda"`` once the extension is loaded, ``"torch"`` if it could not be built.
BACKEND = "torch"
#: Why the build failed, if it did.  A silent fallback would score about 1.00x
#: and look like an honest result, so the reason is kept and asserted on.
BUILD_ERROR: str | None = None
#: The ``TORCH_CUDA_ARCH_LIST`` the build actually saw.
ARCH_LIST_USED: str | None = None
#: The ambient value, restored after the build.  The environment here names six
#: architectures, which would be compiled six times over on every build.
AMBIENT_ARCH_LIST = os.environ.get("TORCH_CUDA_ARCH_LIST", "__unset__")
#: Which path served the most recent call: vectorized, generic, reference, empty.
LAST_PATH = "none"

_EXT_MODULE_NAME = "store_kvcache_hnd_ext"
_ext = None

_ELEMENTS_PER_UNIT = 8  # a 16-byte unit holds eight 2-byte elements


_CPP_SOURCE = r"""
#include <torch/extension.h>

int64_t store_kv_hnd(const at::Tensor& key, const at::Tensor& value,
                     at::Tensor k_cache, at::Tensor v_cache,
                     const at::Tensor& slot_mapping, int64_t page_size);
"""


_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <limits>

namespace {

using Unit = uint4;                     // one 128-bit access
constexpr int kElementsPerUnit = 8;     // ... of 2-byte elements
constexpr int kTargetThreads = 256;
constexpr int kMaxThreads = 1024;
constexpr int kMaxBlockZ = 64;   // CUDA caps blockDim.z at 64, not at kMaxThreads
constexpr int kMaxFlatBlocks = 1 << 20;

// One thread per (token, head, 16-byte chunk).  The token comes from
// threadIdx.z + blockIdx.x, so no division is needed to decompose the work.
template <typename SlotT, bool PageIsPowerOfTwo>
__global__ void store_rows(const Unit* __restrict__ key,
                           const Unit* __restrict__ value,
                           Unit* __restrict__ k_cache,
                           Unit* __restrict__ v_cache,
                           const SlotT* __restrict__ slot_mapping,
                           const int64_t key_row_units,
                           const int64_t value_row_units,
                           const int units_per_row,
                           const int num_heads,
                           const int rows_per_block,
                           const int num_tokens,
                           const int page_size,
                           const int page_shift) {
  const int token = blockIdx.x * rows_per_block + threadIdx.z;
  if (token >= num_tokens) return;
  // Widened here rather than by a host-side cast, which would cost a whole
  // extra kernel launch per call.
  const int64_t slot = static_cast<int64_t>(slot_mapping[token]);
  if (slot < 0) return;  // padded or unscheduled token

  int64_t block, sibling;
  if (PageIsPowerOfTwo) {
    block = slot >> page_shift;
    sibling = slot & (static_cast<int64_t>(page_size) - 1);
  } else {
    block = slot / page_size;
    sibling = slot - block * page_size;
  }

  const int head = threadIdx.y;
  const int chunk = threadIdx.x;
  // 64-bit throughout: num_blocks * H * page_size * head_dim passes 2^31 at
  // 131072 blocks for the H=8, page=16, head_dim=128 geometry, and nothing in
  // the contract bounds num_blocks.
  const int64_t src_within = static_cast<int64_t>(head) * units_per_row + chunk;
  const int64_t dst = ((block * num_heads + head) * page_size + sibling) *
                          static_cast<int64_t>(units_per_row) + chunk;
  k_cache[dst] = key[static_cast<int64_t>(token) * key_row_units + src_within];
  v_cache[dst] = value[static_cast<int64_t>(token) * value_row_units + src_within];
}

// Fallback shape for rows too wide to fit one block: a flat grid-stride loop
// that pays real divisions to decompose the unit index.
template <typename SlotT>
__global__ void store_units(const Unit* __restrict__ key,
                            const Unit* __restrict__ value,
                            Unit* __restrict__ k_cache,
                            Unit* __restrict__ v_cache,
                            const SlotT* __restrict__ slot_mapping,
                            const int64_t key_row_units,
                            const int64_t value_row_units,
                            const int units_per_row,
                            const int num_heads,
                            const int page_size,
                            const int64_t total_units) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total_units; i += step) {
    const int64_t row = i / units_per_row;
    const int chunk = static_cast<int>(i - row * units_per_row);
    const int64_t token = row / num_heads;
    const int head = static_cast<int>(row - token * num_heads);
    const int64_t slot = static_cast<int64_t>(slot_mapping[token]);
    if (slot < 0) continue;
    const int64_t block = slot / page_size;
    const int64_t sibling = slot - block * page_size;
    const int64_t src_within = static_cast<int64_t>(head) * units_per_row + chunk;
    const int64_t dst = ((block * num_heads + head) * page_size + sibling) *
                            static_cast<int64_t>(units_per_row) + chunk;
    k_cache[dst] = key[token * key_row_units + src_within];
    v_cache[dst] = value[token * value_row_units + src_within];
  }
}

}  // namespace

#define LAUNCH_ROWS(SlotT, POW2)                                              \
  store_rows<SlotT, POW2><<<grid, block, 0, stream>>>(                        \
      reinterpret_cast<const Unit*>(key.data_ptr()),                          \
      reinterpret_cast<const Unit*>(value.data_ptr()),                        \
      reinterpret_cast<Unit*>(k_cache.data_ptr()),                            \
      reinterpret_cast<Unit*>(v_cache.data_ptr()),                            \
      slot_mapping.data_ptr<SlotT>(), key_row_units, value_row_units,         \
      static_cast<int>(units_per_row), static_cast<int>(num_heads),           \
      rows_per_block, static_cast<int>(num_tokens),                           \
      static_cast<int>(page_size), page_shift)

#define LAUNCH_UNITS(SlotT)                                                   \
  store_units<SlotT><<<grid, block, 0, stream>>>(                             \
      reinterpret_cast<const Unit*>(key.data_ptr()),                          \
      reinterpret_cast<const Unit*>(value.data_ptr()),                        \
      reinterpret_cast<Unit*>(k_cache.data_ptr()),                            \
      reinterpret_cast<Unit*>(v_cache.data_ptr()),                            \
      slot_mapping.data_ptr<SlotT>(), key_row_units, value_row_units,         \
      static_cast<int>(units_per_row), static_cast<int>(num_heads),           \
      static_cast<int>(page_size), total_units)

// Returns 0 when the division-free shape ran, 1 when a generic shape did.
int64_t store_kv_hnd(const at::Tensor& key, const at::Tensor& value,
                     at::Tensor k_cache, at::Tensor v_cache,
                     const at::Tensor& slot_mapping, int64_t page_size) {
  const int64_t num_tokens = key.size(0);
  const int64_t num_heads = key.size(1);
  const int64_t head_dim = key.size(2);
  TORCH_CHECK(num_tokens > 0, "no tokens to store");
  TORCH_CHECK(page_size >= 1 && page_size <= std::numeric_limits<int>::max(),
              "page_size out of range: ", page_size);
  TORCH_CHECK(num_heads >= 1, "num_kv_heads must be positive, got ", num_heads);
  TORCH_CHECK(head_dim >= kElementsPerUnit && head_dim % kElementsPerUnit == 0,
              "head_dim ", head_dim, " is not a positive multiple of ",
              kElementsPerUnit);
  TORCH_CHECK(slot_mapping.numel() >= num_tokens, "slot_mapping is shorter than the "
              "token count");

  const at::cuda::CUDAGuard guard(key.device());
  const auto stream = at::cuda::getCurrentCUDAStream();

  const int64_t units_per_row = head_dim / kElementsPerUnit;
  const int64_t key_row_units = key.stride(0) / kElementsPerUnit;
  const int64_t value_row_units = value.stride(0) / kElementsPerUnit;
  const auto slot_dtype = slot_mapping.scalar_type();
  TORCH_CHECK(slot_dtype == at::kInt || slot_dtype == at::kLong,
              "slot_mapping dtype must be int32 or int64, got ", slot_dtype);

  int page_shift = -1;
  if ((page_size & (page_size - 1)) == 0) {
    page_shift = 0;
    for (int64_t p = page_size; p > 1; p >>= 1) ++page_shift;
  }

  const int64_t units_per_token = units_per_row * num_heads;
  int64_t code = 0;
  if (units_per_token <= kMaxThreads &&
      num_tokens <= std::numeric_limits<int>::max()) {
    // Narrow rows put many tokens in a block, and blockDim.z stops at 64 however
    // many threads are left over: head_dim 8 with one head would otherwise ask
    // for 256 and fail the launch outright.
    int rows_per_block =
        static_cast<int>(std::max<int64_t>(1, kTargetThreads / units_per_token));
    rows_per_block = static_cast<int>(
        std::min<int64_t>({static_cast<int64_t>(rows_per_block), num_tokens,
                           static_cast<int64_t>(kMaxBlockZ)}));
    const dim3 block(static_cast<unsigned>(units_per_row),
                     static_cast<unsigned>(num_heads),
                     static_cast<unsigned>(rows_per_block));
    const dim3 grid(static_cast<unsigned>(
        (num_tokens + rows_per_block - 1) / rows_per_block));
    if (page_shift >= 0) {
      if (slot_dtype == at::kInt) { LAUNCH_ROWS(int32_t, true); }
      else                        { LAUNCH_ROWS(int64_t, true); }
      code = 0;
    } else {
      if (slot_dtype == at::kInt) { LAUNCH_ROWS(int32_t, false); }
      else                        { LAUNCH_ROWS(int64_t, false); }
      code = 1;
    }
  } else {
    const int64_t total_units = num_tokens * units_per_token;
    const dim3 block(kTargetThreads);
    const dim3 grid(static_cast<unsigned>(std::min<int64_t>(
        (total_units + kTargetThreads - 1) / kTargetThreads, kMaxFlatBlocks)));
    if (slot_dtype == at::kInt) { LAUNCH_UNITS(int32_t); }
    else                        { LAUNCH_UNITS(int64_t); }
    code = 1;
  }
  AT_CUDA_CHECK(cudaGetLastError());
  return code;
}
"""


def _target_arch() -> str:
    """The single architecture to compile for.

    Asks the visible device so the extension is never built for an architecture
    it will not run on; falls back to Blackwell when no device is visible, which
    is the case for a plain import outside a GPU lease.
    """
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return f"{major}.{minor}"
    except Exception:  # noqa: BLE001 -- a probe failure must not stop the build
        pass
    return "10.0"


def _build_extension():
    """Compile at import so ``forward`` has no build branch to take."""
    global BACKEND, BUILD_ERROR, ARCH_LIST_USED

    cached = sys.modules.get(_EXT_MODULE_NAME)
    if cached is not None:
        BACKEND = "cuda"
        ARCH_LIST_USED = getattr(cached, "_arch_list_used", _target_arch())
        return cached

    from torch.utils.cpp_extension import load_inline

    arch = _target_arch()
    previous_extensions_dir = os.environ.get("TORCH_EXTENSIONS_DIR")
    EXTENSION_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(EXTENSION_DIR)
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    try:
        module = load_inline(
            name=_EXT_MODULE_NAME,
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["store_kv_hnd"],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001 -- recorded, then asserted on by the gate
        BUILD_ERROR = f"{type(exc).__name__}: {exc}"
        return None
    else:
        BACKEND = "cuda"
        ARCH_LIST_USED = arch
        module._arch_list_used = arch
        sys.modules[_EXT_MODULE_NAME] = module
        return module
    finally:
        if AMBIENT_ARCH_LIST == "__unset__":
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = AMBIENT_ARCH_LIST
        if previous_extensions_dir is None:
            os.environ.pop("TORCH_EXTENSIONS_DIR", None)
        else:
            os.environ["TORCH_EXTENSIONS_DIR"] = previous_extensions_dir


_ext = _build_extension()


def _device_path_applies(key, value, k_cache, v_cache, slot_mapping,
                         page_size: int) -> bool:
    """Everything the device kernels promise, checked rather than assumed.

    Each clause is a precondition of the 16-byte access, of the index arithmetic,
    or of the launch geometry; failing any one of them diverts the store to a
    path that can express it.  This says nothing about slot values: an
    out-of-range positive slot still writes out of bounds, as it does in the
    baseline.
    """
    if not (key.is_cuda and value.is_cuda and k_cache.is_cuda and v_cache.is_cuda
            and slot_mapping.is_cuda):
        return False
    device = key.device
    if any(t.device != device for t in (value, k_cache, v_cache, slot_mapping)):
        return False
    if key.dim() != 3 or k_cache.dim() != 4 or v_cache.dim() != 4:
        return False
    if value.shape != key.shape or v_cache.shape != k_cache.shape:
        return False
    if key.element_size() != 2:
        return False
    if not (value.dtype == k_cache.dtype == v_cache.dtype == key.dtype):
        return False

    num_heads, head_dim = int(key.shape[1]), int(key.shape[2])
    # A degenerate head count or head dim would divide by zero when the launch
    # geometry is chosen.
    if num_heads < 1 or head_dim < _ELEMENTS_PER_UNIT:
        return False
    if head_dim % _ELEMENTS_PER_UNIT != 0:
        return False
    if (int(k_cache.shape[1]) != num_heads or int(k_cache.shape[2]) != page_size
            or int(k_cache.shape[3]) != head_dim):
        return False
    if not (k_cache.is_contiguous() and v_cache.is_contiguous()):
        return False

    # A row must be a contiguous head_dim run, and consecutive heads must sit
    # head_dim apart -- that is what makes head*head_dim the source offset.
    for source in (key, value):
        if source.stride(1) != head_dim or source.stride(2) != 1:
            return False
        if source.stride(0) % _ELEMENTS_PER_UNIT != 0:
            return False

    if slot_mapping.dim() != 1 or not slot_mapping.is_contiguous():
        return False
    if slot_mapping.dtype not in (torch.int32, torch.int64):
        return False
    if page_size < 1:
        return False
    # The kernels mark both caches __restrict__, so they must not be the same
    # tensor.  (Partially overlapping caches are out of contract either way.)
    if k_cache.data_ptr() == v_cache.data_ptr():
        return False
    return all(t.data_ptr() % 16 == 0 for t in (key, value, k_cache, v_cache))


def _store_by_index(key, value, k_cache, v_cache, slot_mapping,
                    page_size: int) -> None:
    """Layout-general store in terms of logical indices only.

    Deliberately not written as ``k_cache.view(-1, head_dim)``: that raises on a
    cache which is a non-contiguous view, and absorbing exactly those layouts is
    this path's whole purpose.  Negative slots are dropped, the token count comes
    from ``key``, both source row strides are respected because the source is
    indexed rather than offset, and untouched cache bytes are never rewritten.
    """
    num_tokens = int(key.shape[0])
    num_heads = int(key.shape[1])
    # Widened before the sign test, in that order, because the baseline casts the
    # whole mapping to int64 first: a fractional -0.5 becomes 0 and is stored,
    # not skipped.
    slots = slot_mapping.reshape(-1)[:num_tokens].to(torch.int64)

    rows = None
    if not bool((slots >= 0).all()):
        rows = torch.nonzero(slots >= 0, as_tuple=False).reshape(-1)
        slots = slots[rows]
    if slots.numel() == 0:
        return

    slots = slots.to(device=k_cache.device)
    block = torch.div(slots, page_size, rounding_mode="floor")
    sibling = (slots - block * page_size).unsqueeze(1)
    block = block.unsqueeze(1)
    heads = torch.arange(num_heads, device=k_cache.device)

    src_key = key if rows is None else key.index_select(0, rows.to(key.device))
    src_value = value if rows is None else value.index_select(0, rows.to(value.device))
    k_cache[block, heads, sibling] = src_key.to(k_cache.dtype)
    v_cache[block, heads, sibling] = src_value.to(v_cache.dtype)


class StoreKVCacheHND(nn.Module):
    """HND layout store: [num_blocks, num_kv_heads, page_size, head_dim]."""

    def __init__(self, page_size: int):
        super().__init__()
        self.page_size = page_size

    def forward(self, key, value, k_cache, v_cache, slot_mapping):
        global LAST_PATH
        num_tokens = int(key.shape[0])
        if slot_mapping.numel() < num_tokens:
            raise ValueError(
                f"slot_mapping has {slot_mapping.numel()} entries for {num_tokens} "
                "tokens; the token count comes from key.shape[0]")
        if value.shape != key.shape:
            # The baseline reads value at key-derived offsets, so a different
            # value shape is already meaningless there.  Rejected up front rather
            # than half-applied: the alternative fallback path can mutate k_cache
            # and only then fail on v_cache.
            raise ValueError(
                f"value shape {tuple(value.shape)} does not match key "
                f"{tuple(key.shape)}")
        if num_tokens == 0:
            LAST_PATH = "empty"
            return None

        page_size = int(self.page_size)
        if _ext is not None and _device_path_applies(
                key, value, k_cache, v_cache, slot_mapping, page_size):
            code = _ext.store_kv_hnd(key, value, k_cache, v_cache, slot_mapping,
                                     page_size)
            LAST_PATH = "vectorized" if code == 0 else "generic"
            return None

        LAST_PATH = "reference"
        _store_by_index(key, value, k_cache, v_cache, slot_mapping, page_size)
        return None
