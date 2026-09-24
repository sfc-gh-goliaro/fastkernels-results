"""MSA module embedder for AlphaFold3 (Algorithm 8, lines 1-4), fused.

Embeds MSA features and adds projected s_input. The reference implementation
spends four CUDA kernels and four eager dispatches on this (a concatenation, two
GEMMs and a broadcast add) for well under a megabyte of arithmetic, so it is
bound by launch and dispatch overhead rather than by compute or bandwidth. This
implementation computes both projections and their sum in a single CUDA kernel
behind a single host-side call.

Reference: openfold3/core/model/feature_embedders/input_embedders.py
           MSAModuleEmbedder
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

from ..L1.linear import Linear

# The kernel is specialized on the MSA channel count at compile time; every other
# extent is a runtime argument bounded by the shared-memory budget.
_FUSED_C_M = 64

# How many blocks share one token's output channels. 1 gives one block per token;
# larger values spread the same work over proportionally more blocks, trading a
# repeated read of the token's s_input row for more of the device. Every value
# here is measured against the others by tools/latency_rig.py, and the default is
# whichever won.
_CHANNEL_SPLITS = (1, 2, 4, 8)
_DEFAULT_CHANNEL_SPLIT = 8

# Weight loads the s_input reduction issues before consuming any. Both forms are
# compiled so the choice can be made on a paired measurement rather than on the
# usual assumption that more outstanding loads must help a latency-bound loop.
_LOADS_IN_FLIGHT = (1, 4)
_DEFAULT_LOADS_IN_FLIGHT = 1

# Resource limits the fused path has to stay inside: the default dynamic
# shared-memory allowance per block, and the number of blocks one grid dimension
# can index, which bounds the flattened leading extent.
_MAX_SHARED_BYTES = 48 * 1024
_MAX_GRID_DIMENSION = 65535

# Build isolation. torch's FileBaton has no timeout, no PID check and no staleness
# recovery: a builder killed mid-build leaves its lock file behind, and every
# later import of that extension name then spins on it forever. A workspace-local
# build directory under a project-unique name keeps that failure class out of the
# shared cache, and the outer lock below makes the remaining race recoverable.
_EXTENSION_NAME = "fk_l2_alphafold3_msa_module_embedder"
_BUILD_ROOT = Path(__file__).resolve().parents[2] / ".torch_extensions"

# The name of torch's own lock file inside the build directory, and the name of
# the outer lock that guards it.
_INNER_LOCK_NAME = "lock"
_OUTER_LOCK_NAME = "build.flock"

# Architecture to compile for. Deliberately a constant rather than a capability
# query: reading the capability would initialize CUDA at import, which is the very
# thing setting this variable exists to prevent.
_CUDA_ARCH = "10.0"

_ENTRY_SIGNATURE = """(const at::Tensor& msa, const at::Tensor& has_deletion,
    const at::Tensor& deletion_value, const at::Tensor& s_input,
    const at::Tensor& w_msa, const at::Tensor& w_s_input)"""

_KERNEL_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

namespace {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarps = kThreads / kWarpSize;
constexpr int kChannels = 64;
constexpr int kMaxSharedBytes = 48 * 1024;

// One block owns one (batch, token) pair and one group of `kChannels / SPLIT`
// output channels; it produces m[b, :, t, c_base : c_base + channels_per_block].
//
// Both projections are computed here. The s_input projection reduces over
// c_s_input and is warp-reduced; the MSA projection reduces over n_feat and is
// laid out so that no cross-thread reduction is needed at all.
//
// MATCH_REFERENCE_ROUNDING reproduces the reference's rounding structure: the
// reference rounds each projection to bfloat16 before adding them, and its
// bfloat16 add rounds the float sum in turn. The alternative keeps the two
// accumulators in float all the way to the single final rounding.
template <int SPLIT, bool MATCH_REFERENCE_ROUNDING, int LOADS_IN_FLIGHT>
__global__ void msa_embed_kernel(
    const __nv_bfloat16* __restrict__ msa,
    const __nv_bfloat16* __restrict__ has_deletion,
    const __nv_bfloat16* __restrict__ deletion_value,
    const __nv_bfloat16* __restrict__ s_input,
    const __nv_bfloat16* __restrict__ w_msa,
    const __nv_bfloat16* __restrict__ w_s_input,
    __nv_bfloat16* __restrict__ out,
    int n_seq, int n_token, int n_feat, int c_s_input) {
  static_assert(kChannels % SPLIT == 0, "the channel split must divide the channels");
  constexpr int kChannelsPerBlock = kChannels / SPLIT;
  constexpr int kWeightStride = kChannelsPerBlock + 1;

  const int token = blockIdx.x;
  const int sample = blockIdx.y;
  const int channel_base = blockIdx.z * kChannelsPerBlock;
  const int tid = threadIdx.x;

  // The two trailing feature channels are has_deletion and deletion_value; the
  // rest come from msa.
  const int n_msa_channels = n_feat - 2;
  const int feature_stride = n_feat + 1;

  extern __shared__ float shared[];
  float* s_input_tile = shared;                                 // [c_s_input]
  float* feature_tile = s_input_tile + c_s_input;               // [n_seq][feature_stride]
  float* w_msa_tile = feature_tile + n_seq * feature_stride;    // [n_feat][kWeightStride]
  float* s_projection = w_msa_tile + n_feat * kWeightStride;    // [kChannelsPerBlock]

  const __nv_bfloat16* s_input_row =
      s_input + (static_cast<long>(sample) * n_token + token) * c_s_input;
  for (int j = tid; j < c_s_input; j += kThreads) {
    s_input_tile[j] = __bfloat162float(s_input_row[j]);
  }

  // Assemble the feature rows this block reduces over. For a fixed token,
  // msa[b, s, t, :] is n_msa_channels contiguous values.
  for (int idx = tid; idx < n_seq * n_feat; idx += kThreads) {
    const int seq = idx / n_feat;
    const int k = idx - seq * n_feat;
    const long row = (static_cast<long>(sample) * n_seq + seq) * n_token + token;
    float value;
    if (k < n_msa_channels) {
      value = __bfloat162float(msa[row * n_msa_channels + k]);
    } else if (k == n_msa_channels) {
      value = __bfloat162float(has_deletion[row]);
    } else {
      value = __bfloat162float(deletion_value[row]);
    }
    feature_tile[seq * feature_stride + k] = value;
  }

  // Stage the MSA weight transposed. Reading w_msa[c, k] straight from global
  // with c varying across threads would stride by a row and touch one sector per
  // lane; instead the block reads it linearly and transposes through shared
  // memory. The row pad keeps both the scatter here and the strided read below
  // free of bank conflicts.
  for (int idx = tid; idx < kChannelsPerBlock * n_feat; idx += kThreads) {
    const int channel = idx / n_feat;
    const int k = idx - channel * n_feat;
    w_msa_tile[k * kWeightStride + channel] =
        __bfloat162float(w_msa[static_cast<long>(channel_base + channel) * n_feat + k]);
  }
  __syncthreads();

  // s_input projection: one warp per channel, lanes strided over c_s_input so
  // each warp load is 32 consecutive values of one weight row.
  //
  // LOADS_IN_FLIGHT controls how many of those loads are issued into separate
  // accumulators before any is consumed. 1 is the plain reduction; larger values
  // trade registers for outstanding memory requests. Both forms are compiled so
  // they can be compared paired in one process.
  const int warp = tid / kWarpSize;
  const int lane = tid % kWarpSize;
  for (int channel = warp; channel < kChannelsPerBlock; channel += kWarps) {
    const __nv_bfloat16* w_row =
        w_s_input + static_cast<long>(channel_base + channel) * c_s_input;
    float partial[LOADS_IN_FLIGHT] = {};
    int j = lane;
    for (; j + (LOADS_IN_FLIGHT - 1) * kWarpSize < c_s_input;
         j += LOADS_IN_FLIGHT * kWarpSize) {
      float weight[LOADS_IN_FLIGHT];
#pragma unroll
      for (int u = 0; u < LOADS_IN_FLIGHT; ++u) {
        weight[u] = __bfloat162float(w_row[j + u * kWarpSize]);
      }
#pragma unroll
      for (int u = 0; u < LOADS_IN_FLIGHT; ++u) {
        partial[u] += weight[u] * s_input_tile[j + u * kWarpSize];
      }
    }
    for (; j < c_s_input; j += kWarpSize) {
      partial[0] += __bfloat162float(w_row[j]) * s_input_tile[j];
    }
    float acc = partial[0];
#pragma unroll
    for (int u = 1; u < LOADS_IN_FLIGHT; ++u) {
      acc += partial[u];
    }
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
      s_projection[channel] = acc;
    }
  }
  __syncthreads();

  // MSA projection, sum and store. Each thread owns one (sequence, channel)
  // output, so the reduction over n_feat is thread-local: the feature value is
  // warp-uniform and broadcasts, and consecutive lanes read consecutive
  // channels of the staged weight and write consecutive outputs.
  const int n_out = n_seq * kChannelsPerBlock;
  for (int idx = tid; idx < n_out; idx += kThreads) {
    const int seq = idx / kChannelsPerBlock;
    const int channel = idx - seq * kChannelsPerBlock;
    const float* feature_row = feature_tile + seq * feature_stride;
    const float* weight_column = w_msa_tile + channel;
    float acc = 0.0f;
    for (int k = 0; k < n_feat; ++k) {
      acc += feature_row[k] * weight_column[k * kWeightStride];
    }
    float total;
    if (MATCH_REFERENCE_ROUNDING) {
      total = __bfloat162float(__float2bfloat16(acc)) +
              __bfloat162float(__float2bfloat16(s_projection[channel]));
    } else {
      total = acc + s_projection[channel];
    }
    const long dst =
        ((static_cast<long>(sample) * n_seq + seq) * n_token + token) * kChannels +
        channel_base + channel;
    out[dst] = __float2bfloat16(total);
  }
}

const __nv_bfloat16* bf16_ptr(const at::Tensor& t) {
  return reinterpret_cast<const __nv_bfloat16*>(t.const_data_ptr<at::BFloat16>());
}

template <int SPLIT, bool MATCH_REFERENCE_ROUNDING, int LOADS_IN_FLIGHT>
at::Tensor msa_embed_impl(const at::Tensor& msa, const at::Tensor& has_deletion,
                          const at::Tensor& deletion_value, const at::Tensor& s_input,
                          const at::Tensor& w_msa, const at::Tensor& w_s_input) {
  constexpr int kChannelsPerBlock = kChannels / SPLIT;

  TORCH_CHECK(msa.is_cuda() && has_deletion.is_cuda() && deletion_value.is_cuda() &&
                  s_input.is_cuda() && w_msa.is_cuda() && w_s_input.is_cuda(),
              "msa_embed: every input must be a CUDA tensor");
  // One device guard and one stream cover the launch, so a tensor on a different
  // device than msa would be read through a pointer this kernel cannot resolve.
  TORCH_CHECK(has_deletion.device() == msa.device() &&
                  deletion_value.device() == msa.device() &&
                  s_input.device() == msa.device() && w_msa.device() == msa.device() &&
                  w_s_input.device() == msa.device(),
              "msa_embed: every input must be on the same device as msa (",
              msa.device(), ")");
  TORCH_CHECK(msa.scalar_type() == at::kBFloat16 &&
                  has_deletion.scalar_type() == at::kBFloat16 &&
                  deletion_value.scalar_type() == at::kBFloat16 &&
                  s_input.scalar_type() == at::kBFloat16 &&
                  w_msa.scalar_type() == at::kBFloat16 &&
                  w_s_input.scalar_type() == at::kBFloat16,
              "msa_embed: every input must be bfloat16");
  TORCH_CHECK(msa.is_contiguous() && has_deletion.is_contiguous() &&
                  deletion_value.is_contiguous() && s_input.is_contiguous() &&
                  w_msa.is_contiguous() && w_s_input.is_contiguous(),
              "msa_embed: every input must be contiguous");
  // The reference accepts any number of leading dimensions: msa is
  // [*, N_msa, N_token, K] and s_input is [*, N_token, J]. Those leading
  // dimensions are contiguous and only ever indexed as a whole, so they flatten
  // into a single axis and the kernel needs no notion of them.
  TORCH_CHECK(msa.dim() >= 3 && w_msa.dim() == 2 && w_s_input.dim() == 2,
              "msa_embed: expected msa [*, N_msa, N_token, K] and both weights 2-D, "
              "got dims ", msa.dim(), "/", w_msa.dim(), "/", w_s_input.dim());
  const int64_t leading = msa.dim() - 3;
  TORCH_CHECK(has_deletion.dim() == msa.dim() - 1 &&
                  deletion_value.dim() == msa.dim() - 1 &&
                  s_input.dim() == leading + 2,
              "msa_embed: for msa ", msa.sizes(), " expected has_deletion and "
              "deletion_value with ", msa.dim() - 1, " dims and s_input with ",
              leading + 2, ", got ", has_deletion.dim(), "/", deletion_value.dim(),
              "/", s_input.dim());

  const int64_t n_seq = msa.size(leading);
  const int64_t n_token = msa.size(leading + 1);
  const int64_t n_msa_channels = msa.size(leading + 2);
  const int64_t n_feat = w_msa.size(1);
  const int64_t c_s_input = w_s_input.size(1);
  int64_t n_sample = 1;
  for (int64_t d = 0; d < leading; ++d) {
    TORCH_CHECK(s_input.size(d) == msa.size(d),
                "msa_embed: s_input ", s_input.sizes(), " and msa ", msa.sizes(),
                " disagree in leading dimension ", d);
    n_sample *= msa.size(d);
  }

  TORCH_CHECK(has_deletion.sizes() == msa.sizes().slice(0, msa.dim() - 1),
              "msa_embed: has_deletion ", has_deletion.sizes(),
              " must match msa ", msa.sizes(), " without its channel dimension");
  TORCH_CHECK(deletion_value.sizes() == has_deletion.sizes(),
              "msa_embed: has_deletion ", has_deletion.sizes(), " and deletion_value ",
              deletion_value.sizes(), " must agree");
  TORCH_CHECK(s_input.size(leading) == n_token,
              "msa_embed: s_input ", s_input.sizes(), " and msa ", msa.sizes(),
              " disagree in token count");
  TORCH_CHECK(s_input.size(leading + 1) == c_s_input,
              "msa_embed: s_input trailing dim ", s_input.size(leading + 1),
              " disagrees with the s_input weight ", w_s_input.sizes());
  TORCH_CHECK(n_sample <= 65535,
              "msa_embed: ", n_sample, " leading elements exceed the 65535 blocks "
              "one grid dimension can hold");
  TORCH_CHECK(n_feat == n_msa_channels + 2,
              "msa_embed: the MSA weight expects ", n_feat,
              " features but msa contributes ", n_msa_channels, " plus 2");
  TORCH_CHECK(w_msa.size(0) == kChannels && w_s_input.size(0) == kChannels,
              "msa_embed: this kernel is specialized on ", kChannels,
              " output channels, got ", w_msa.size(0), " and ", w_s_input.size(0));

  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(msa));
  at::DimVector out_sizes(msa.sizes());
  out_sizes.back() = kChannels;
  at::Tensor out = at::empty(out_sizes, msa.options());
  if (out.numel() == 0) {
    return out;
  }

  const int64_t shared_floats = c_s_input + n_seq * (n_feat + 1) +
                                n_feat * (kChannelsPerBlock + 1) + kChannelsPerBlock;
  const int64_t shared_bytes = shared_floats * static_cast<int64_t>(sizeof(float));
  TORCH_CHECK(shared_bytes <= kMaxSharedBytes,
              "msa_embed: these extents need ", shared_bytes,
              " bytes of shared memory, above the ", kMaxSharedBytes, " byte budget");

  const dim3 grid(static_cast<unsigned>(n_token), static_cast<unsigned>(n_sample), SPLIT);
  msa_embed_kernel<SPLIT, MATCH_REFERENCE_ROUNDING, LOADS_IN_FLIGHT>
      <<<grid, kThreads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
          bf16_ptr(msa), bf16_ptr(has_deletion), bf16_ptr(deletion_value),
          bf16_ptr(s_input), bf16_ptr(w_msa), bf16_ptr(w_s_input),
          reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr<at::BFloat16>()),
          static_cast<int>(n_seq), static_cast<int>(n_token), static_cast<int>(n_feat),
          static_cast<int>(c_s_input));
  AT_CUDA_CHECK(cudaGetLastError());
  return out;
}

}  // namespace
"""


def _entry_name(channel_split: int, match_reference_rounding: bool,
                loads_in_flight: int = 1) -> str:
    suffix = "" if match_reference_rounding else "_wide_add"
    if loads_in_flight != 1:
        suffix += f"_inflight{loads_in_flight}"
    return f"msa_embed_split{channel_split}{suffix}"


# One entry point per (decomposition, rounding structure, loads-in-flight) triple,
# so that the choice costs nothing at call time: the module resolves its entry
# point once and then passes only the six tensors the kernel needs.
_ENTRY_VARIANTS = tuple(
    (_entry_name(split, matched), split, matched, 1)
    for matched in (True, False)
    for split in _CHANNEL_SPLITS
) + tuple(
    (_entry_name(split, True, 4), split, True, 4) for split in _CHANNEL_SPLITS
)
_ENTRY_NAMES = tuple(name for name, _, _, _ in _ENTRY_VARIANTS)

_CPP_SOURCE = "#include <ATen/ATen.h>\n\n" + "".join(
    f"at::Tensor {name}{_ENTRY_SIGNATURE};\n" for name in _ENTRY_NAMES
)

_CUDA_SOURCE = _KERNEL_SOURCE + "".join(
    f"at::Tensor {name}{_ENTRY_SIGNATURE} {{\n"
    f"  return msa_embed_impl<{split}, {'true' if matched else 'false'}, {inflight}>(\n"
    f"      msa, has_deletion, deletion_value, s_input, w_msa, w_s_input);\n"
    f"}}\n"
    for name, split, matched, inflight in _ENTRY_VARIANTS
)


@contextlib.contextmanager
def _pinned_cuda_arch():
    """Build for one architecture, without touching CUDA to find out which.

    Left to itself the extension builder either fans out over whatever
    TORCH_CUDA_ARCH_LIST happens to hold -- six architectures in this
    environment, so a cold build several times longer than it needs to be -- or,
    when that is unset, initializes CUDA purely to probe the device capability.
    Setting the variable from a constant avoids both. The override is scoped to
    the build so nothing else in the process sees it.
    """
    previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = _CUDA_ARCH
    try:
        yield
    finally:
        if previous is None:
            del os.environ["TORCH_CUDA_ARCH_LIST"]
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous


@contextlib.contextmanager
def _exclusive_build(build_dir: Path):
    """Hold an outer lock for the whole build, then clear torch's inner lock.

    torch guards a build with a lock *file*, which a killed builder leaves behind
    and later importers then wait on forever. Deleting it on a timer cannot fix
    that: a process already inside torch's wait never re-checks, so a builder
    killed after we looked wedges us regardless of any age threshold.

    An `flock` fixes it properly, because the kernel releases it when the holder
    dies. Every builder of this extension takes this lock first, so once it is
    held no live builder owns the inner lock, and any inner lock still present is
    debris that can be removed unconditionally. A waiter therefore makes progress
    as soon as the holder exits, however it exits.
    """
    build_dir.mkdir(parents=True, exist_ok=True)
    outer = build_dir / _OUTER_LOCK_NAME
    handle = os.open(outer, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"[{_EXTENSION_NAME}] another process is building; waiting for it",
                  file=sys.stderr, flush=True)
            fcntl.flock(handle, fcntl.LOCK_EX)
        inner = build_dir / _INNER_LOCK_NAME
        if inner.exists():
            try:
                inner.unlink()
                print(f"[{_EXTENSION_NAME}] cleared an abandoned build lock at {inner}",
                      file=sys.stderr, flush=True)
            except FileNotFoundError:
                pass
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        os.close(handle)


def _build_extension():
    """Compile the fused kernel, or report why it could not be compiled.

    Returns (extension, error). A build failure is not fatal: the module keeps
    working through the reference path, but it says so on stderr -- which lands
    in the benchmark worker log -- and reports it through `backend`, so a run
    that silently fell back cannot be mistaken for a fused one.
    """
    from torch.utils.cpp_extension import load_inline

    build_dir = _BUILD_ROOT / _EXTENSION_NAME
    try:
        with _exclusive_build(build_dir), _pinned_cuda_arch():
            extension = load_inline(
                name=_EXTENSION_NAME,
                cpp_sources=_CPP_SOURCE,
                cuda_sources=_CUDA_SOURCE,
                functions=list(_ENTRY_NAMES),
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "-lineinfo", "--expt-relaxed-constexpr"],
                build_directory=str(build_dir),
            )
    except Exception as exc:  # noqa: BLE001 - any build failure degrades, loudly
        print(f"[{_EXTENSION_NAME}] build failed, falling back to the reference "
              f"implementation, which is correct but unoptimized: {exc!r}",
              file=sys.stderr, flush=True)
        return None, exc
    print(f"[{_EXTENSION_NAME}] fused kernel ready from {build_dir}",
          file=sys.stderr, flush=True)
    return extension, None


# Built once, at import, so that no compilation, subprocess or CUDA
# initialization can happen inside a timed forward. Which way the build went is
# announced either way: the benchmark worker captures this module's stderr into
# its per-operator log, so the log of a recorded run carries positive evidence of
# which implementation was actually measured, rather than only saying something
# when the build failed.
_EXTENSION, _BUILD_ERROR = _build_extension()


def build_error() -> BaseException | None:
    """The exception that stopped the fused kernel from building, if any."""
    return _BUILD_ERROR


def _fused_eligible(msa: torch.Tensor, has_deletion: torch.Tensor,
                    deletion_value: torch.Tensor, s_input: torch.Tensor,
                    max_sequences: int) -> bool:
    """Whether these inputs are within the fused kernel's resource limits.

    This draws the line between two different kinds of "the kernel cannot do
    this". A call the reference handles perfectly well must keep working, so the
    cases below -- a non-contiguous input, more sequences than fit in shared
    memory, or a leading product past what one grid dimension can index -- are
    routed to the reference path and produce a correct answer, just unfused.

    Genuinely malformed input is *not* filtered here. A CPU tensor, a
    non-bfloat16 tensor or mutually inconsistent shapes fall through to the
    extension, which rejects them with a specific `TORCH_CHECK` rather than
    silently computing something. Falling back on those would turn a caller's
    mistake into a quiet slowdown.
    """
    if not (msa.is_contiguous() and has_deletion.is_contiguous()
            and deletion_value.is_contiguous() and s_input.is_contiguous()):
        return False
    shape = msa.shape
    if len(shape) < 3:
        # Too few dimensions to be a valid msa at all. Hand it to the extension so
        # the caller gets the dimensional TORCH_CHECK rather than a silent detour
        # through a reference path that would fail its own way.
        return True
    if shape[-3] > max_sequences:
        return False
    leading = 1
    for extent in shape[:-3]:
        leading *= extent
    return leading <= _MAX_GRID_DIMENSION


class MSAModuleEmbedder(nn.Module):
    """AF3 Algorithm 8, lines 1-4: MSA feature embedding.

    Args:
        c_m_feats: MSA input features channel dimension (34 = 32 msa + has_deletion + deletion_value)
        c_m: MSA channel dimension
        c_s_input: Single (s_input) channel dimension
    """

    def __init__(
        self,
        c_m_feats: int = 34,
        c_m: int = 64,
        c_s_input: int = 449,
    ):
        super().__init__()
        self.linear_m = Linear(c_m_feats, c_m, bias=False)
        self.linear_s_input = Linear(c_s_input, c_m, bias=False)
        self._fused_entry = None
        self._match_reference_rounding = True
        self._channel_split = _DEFAULT_CHANNEL_SPLIT
        self._loads_in_flight = _DEFAULT_LOADS_IN_FLIGHT
        self._max_sequences = 0
        if self._fusable(c_m_feats, c_m, c_s_input):
            self._fused_entry = getattr(_EXTENSION, _entry_name(
                self._channel_split, True, self._loads_in_flight))

    def _fusable(self, c_m_feats: int, c_m: int, c_s_input: int) -> bool:
        """Whether the fused kernel covers this configuration.

        The kernel is compiled for one output channel count and needs at least one
        MSA channel alongside the two deletion channels. It also stages the
        s_input row and the transposed MSA weight in shared memory, which is what
        bounds how many sequences a call may carry; that bound depends only on the
        constructor arguments, so it is turned into a per-call sequence limit here
        rather than recomputed on every forward.
        """
        if _EXTENSION is None or c_m != _FUSED_C_M or c_m_feats < 3:
            return False
        # The staged weight tile shrinks as the channel split grows, so this uses
        # the widest tile of any compiled decomposition. That makes the sequence
        # limit the same whichever decomposition is selected, which is worth more
        # than the few extra sequences a per-split bound would admit: whether a
        # call is eligible should not change when the kernel is retuned.
        staged_floats = c_s_input + c_m_feats * (_FUSED_C_M + 1) + _FUSED_C_M
        remaining = _MAX_SHARED_BYTES // 4 - staged_floats
        # Each staged sequence costs one padded feature row.
        self._max_sequences = remaining // (c_m_feats + 1)
        return self._max_sequences >= 1

    @property
    def backend(self) -> str:
        """Which implementation this instance runs: `"fused"` or `"reference"`."""
        return "fused" if self._fused_entry is not None else "reference"

    def select_kernel(self, *, channel_split: int | None = None,
                      match_reference_rounding: bool | None = None,
                      loads_in_flight: int | None = None) -> None:
        """Pick the kernel variant to run.

        `channel_split` is how many blocks share one token's channels: 1 gives one
        block per token, larger values spread the same work over proportionally
        more blocks. `loads_in_flight` is how many weight loads the s_input
        reduction issues before consuming any. Raises if this instance is on the
        reference path, so a tuning sweep cannot silently measure the fallback.
        """
        if self._fused_entry is None:
            raise RuntimeError(
                "the fused kernel is unavailable for this configuration"
                + (f": {_BUILD_ERROR!r}" if _BUILD_ERROR is not None else ""))
        if channel_split is not None:
            if channel_split not in _CHANNEL_SPLITS:
                raise ValueError(f"channel_split must be one of {_CHANNEL_SPLITS}")
            self._channel_split = channel_split
        if match_reference_rounding is not None:
            self._match_reference_rounding = bool(match_reference_rounding)
        if loads_in_flight is not None:
            if loads_in_flight not in _LOADS_IN_FLIGHT:
                raise ValueError(f"loads_in_flight must be one of {_LOADS_IN_FLIGHT}")
            self._loads_in_flight = loads_in_flight
        if self._loads_in_flight != 1 and not self._match_reference_rounding:
            raise ValueError("only the reference rounding structure is compiled with "
                             "more than one load in flight")
        self._fused_entry = getattr(_EXTENSION, _entry_name(
            self._channel_split, self._match_reference_rounding,
            self._loads_in_flight))

    def forward(
        self,
        batch: dict,
        s_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch: needs msa [*, N_msa, N_token, 32],
                   has_deletion [*, N_msa, N_token],
                   deletion_value [*, N_msa, N_token],
                   msa_mask [*, N_msa, N_token]
            s_input: [*, N_token, c_s_input]

        Returns:
            m: [*, N_seq, N_token, c_m]
            msa_mask: [*, N_seq, N_token]
        """
        if self._fused_entry is not None:
            msa = batch["msa"]
            has_deletion = batch["has_deletion"]
            deletion_value = batch["deletion_value"]
            if _fused_eligible(msa, has_deletion, deletion_value, s_input,
                               self._max_sequences):
                m = self._fused_entry(msa, has_deletion, deletion_value, s_input,
                                      self.linear_m.weight,
                                      self.linear_s_input.weight)
                return m, batch["msa_mask"]
        return self._reference(batch, s_input)

    def _reference(
        self,
        batch: dict,
        s_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The unfused computation, for configurations the kernel does not cover."""
        msa_feat = torch.cat(
            [
                batch["msa"],
                batch["has_deletion"].unsqueeze(-1),
                batch["deletion_value"].unsqueeze(-1),
            ],
            dim=-1,
        )
        m = self.linear_m(msa_feat)
        m = m + self.linear_s_input(s_input).unsqueeze(-3)
        return m, batch["msa_mask"]
