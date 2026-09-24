"""Oasis spatial axial attention — fused Triton implementation.

The captured workload is tiny (M = time*144 <= 864 rows, 16 heads, head_dim 64,
spatial sequence 9*16 = 144), so the eager baseline is dominated by per-op
launch overhead: it rebuilds the axial rotary table from scratch on every call
(linspace / einsum / repeat_interleave / broadcast / cat / cos / sin), runs four
elementwise rope kernels, several permute-copies, and SDPA.

Here the whole forward is three Triton launches:

  1. ``_gemm_tma``  -- QKV projection, ``x[M,1024] @ Wqkv[3072,1024]^T``, with the
                   axial rotary folded into the epilogue: the q and k halves of
                   the accumulator are rotated in registers before they are
                   written, so the attention kernel reads plain q/k and the key
                   rotation is done once instead of once per query tile.
  2. ``_attn``      -- non-causal attention straight out of the packed
                   projection, writing the result already laid out as
                   [M, heads*dim_head].
  3. ``_gemm_tma``  -- output projection + bias.

(``_gemm`` is the same GEMM with plain pointer loads, used if TMA descriptors
are unavailable.)

Two further tricks matter at this size:

* the cos/sin table is built once and cached -- the spatial extent is fixed for
  a run.  Only the even rotary lanes are stored: ``get_axial_freqs`` ends in
  ``repeat_interleave(2)``, so ``freqs[..., 2i] == freqs[..., 2i+1]`` and one
  table entry serves a whole lane pair.
* the three launches are captured into a CUDA graph (one per input shape), so a
  call costs a single ``cudaGraphLaunch`` rather than three Triton launches --
  worth ~45us of host time per call here.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from ..L1.dense_attention import DenseAttention
from ..L1.linear import Linear
from ..L1.oasis_rotary import OasisRotaryEmbedding, oasis_apply_rotary_emb

try:
    import triton
    import triton.language as tl
    from triton.tools.tensor_descriptor import TensorDescriptor

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - triton ships with torch on this target
    _HAS_TRITON = False


# "graph" = capture the three launches once per shape, "eager" = launch per call.
_MODE = os.environ.get("FK_OASIS_MODE", "graph")
_USE_TMA = os.environ.get("FK_OASIS_TMA", "1") == "1"

# GEMM tile configs tuned on B200 (sm100); keyed by (min_M, N), largest
# tabulated M that fits wins.  K is 1024 for both GEMMs here.
_GEMM_CFGS = {
    (0, 3072): dict(BLOCK_M=128, BLOCK_N=64, BLOCK_K=128, GROUP_M=8,
                    num_warps=4, num_stages=3),
    (432, 3072): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, GROUP_M=8,
                      num_warps=8, num_stages=3),
    (864, 3072): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, GROUP_M=8,
                      num_warps=4, num_stages=3),
    (0, 1024): dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=256, GROUP_M=8,
                    num_warps=4, num_stages=3),
    (720, 1024): dict(BLOCK_M=128, BLOCK_N=64, BLOCK_K=128, GROUP_M=8,
                      num_warps=4, num_stages=3),
}
# Attention tiles, keyed by min_M: with few rows the query axis is split finely
# so there are enough CTAs to hide the load latency; with many rows bigger tiles
# win because the key/value tiles are then re-read fewer times.
_ATTN_CFGS = {
    0: dict(BLOCK_M=16, num_warps=2, num_stages=1),
    720: dict(BLOCK_M=64, num_warps=4, num_stages=2),
}


def _pick_by_m(table, M: int):
    best, best_m = None, -1
    for m, cfg in table.items():
        if m <= M and m > best_m:
            best, best_m = cfg, m
    return dict(best)


def _pick(M: int, N: int) -> dict:
    best, best_m = None, -1
    for (m, n), cfg in _GEMM_CFGS.items():
        if n == N and m <= M and m > best_m:
            best, best_m = cfg, m
    if best is None:
        return dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, GROUP_M=1,
                    num_warps=4, num_stages=3)
    return dict(best)


if _HAS_TRITON:

    @triton.jit
    def _slot_ptr(HOLD, slot):
        """Read a run-time base address out of the indirection slot.

        A captured CUDA graph bakes its kernel arguments, but the output tensor
        is freshly allocated on every call.  Rather than writing into a static
        buffer and cloning it out (~2.4us), the final store takes its base
        address from a device slot that the graph refreshes from page-locked
        host memory, which the CPU can poke with a plain store (~0.6us).
        ``multiple_of`` restores the alignment fact the bitcast loses so the
        store stays vectorised; a torch allocation is always 16-byte aligned.

        Only the *store* is indirected.  Feeding a loop-carried load off a
        bitcast pointer costs far more than the staging copy it would save --
        Triton can no longer prove the access pattern and falls back to scalar
        loads.
        """
        return (tl.multiple_of(tl.load(HOLD + slot), 16)
                .to(tl.pointer_type(tl.float16), bitcast=True))

    @triton.jit
    def _tile_id(pid, M, BLOCK_M: tl.constexpr, num_pid_n: tl.constexpr,
                 GROUP_M: tl.constexpr):
        """Group-ordered (pid_m, pid_n) for L2 reuse of the weight tiles."""
        num_pid_m = tl.cdiv(M, BLOCK_M)
        per_group = GROUP_M * num_pid_n
        first_m = (pid // per_group) * GROUP_M
        rows = min(num_pid_m - first_m, GROUP_M)
        return first_m + ((pid % per_group) % rows), (pid % per_group) // rows

    @triton.jit
    def _rope_epilogue(acc, offs_m, COS, SIN,
                       S: tl.constexpr, DH: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        """Rotate the lane pairs of a q/k accumulator tile.

        ``BLOCK_N`` is a multiple of the head dim, so ``on0`` is too and the
        even-lane index within the tile is just ``arange % DH``.  The tile is
        narrowed to fp16 *before* the lane split: splitting is a layout change
        that goes through shared memory, so halving the element width halves
        that traffic -- and it is what the reference does anyway (it rotates the
        fp16 projection output, not an fp32 accumulator).
        """
        jj = tl.arange(0, BLOCK_N // 2) % DH
        srow = offs_m % S
        c = tl.load(COS + srow[:, None] * DH + jj[None, :])
        s = tl.load(SIN + srow[:, None] * DH + jj[None, :])
        e, o = tl.split(tl.reshape(acc.to(tl.float16), (BLOCK_M, BLOCK_N // 2, 2)))
        ef = e.to(tl.float32)
        of = o.to(tl.float32)
        return tl.reshape(tl.join((ef * c - of * s).to(tl.float16),
                                  (of * c + ef * s).to(tl.float16)),
                          (BLOCK_M, BLOCK_N))

    @triton.jit
    def _gemm(A, B, C, Bias, COS, SIN, HOLD, M, K,
              N: tl.constexpr, S: tl.constexpr, DH: tl.constexpr,
              ROPE_END: tl.constexpr, C_SLOT: tl.constexpr,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
              BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
              HAS_BIAS: tl.constexpr):
        """C[M,N] = A[M,K] @ B[N,K]^T (+ Bias[N]), rope on columns < ROPE_END."""
        num_pid_n: tl.constexpr = N // BLOCK_N
        pid_m, pid_n = _tile_id(tl.program_id(0), M, BLOCK_M, num_pid_n, GROUP_M)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        on0 = pid_n * BLOCK_N
        offs_n = on0 + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        m_mask = offs_m < M

        a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
        b_ptrs = B + offs_n[None, :] * K + offs_k[:, None]
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
            acc = tl.dot(a, tl.load(b_ptrs), acc)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K

        if HAS_BIAS:
            acc += tl.load(Bias + offs_n)[None, :].to(tl.float32)
        out = acc.to(tl.float16)
        if ROPE_END > 0 and on0 < ROPE_END:
            out = _rope_epilogue(acc, offs_m, COS, SIN, S, DH, BLOCK_M, BLOCK_N)
        if C_SLOT >= 0:
            C = _slot_ptr(HOLD, C_SLOT)
        tl.store(C + offs_m[:, None] * N + offs_n[None, :], out,
                 mask=m_mask[:, None])

    @triton.jit
    def _gemm_tma(DA, DB, C, Bias, COS, SIN, HOLD, M, K,
                  N: tl.constexpr, S: tl.constexpr, DH: tl.constexpr,
                  ROPE_END: tl.constexpr, C_SLOT: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
                  HAS_BIAS: tl.constexpr):
        """:func:`_gemm` with the operands fetched through TMA descriptors."""
        num_pid_n: tl.constexpr = N // BLOCK_N
        pid_m, pid_n = _tile_id(tl.program_id(0), M, BLOCK_M, num_pid_n, GROUP_M)
        om = pid_m * BLOCK_M
        on0 = pid_n * BLOCK_N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in tl.range(0, K, BLOCK_K):
            acc = tl.dot(DA.load([om, k]), DB.load([on0, k]).T, acc)

        offs_m = om + tl.arange(0, BLOCK_M)
        offs_n = on0 + tl.arange(0, BLOCK_N)
        if HAS_BIAS:
            acc += tl.load(Bias + offs_n)[None, :].to(tl.float32)
        out = acc.to(tl.float16)
        if ROPE_END > 0 and on0 < ROPE_END:
            out = _rope_epilogue(acc, offs_m, COS, SIN, S, DH, BLOCK_M, BLOCK_N)
        if C_SLOT >= 0:
            C = _slot_ptr(HOLD, C_SLOT)
        tl.store(C + offs_m[:, None] * N + offs_n[None, :], out,
                 mask=(offs_m < M)[:, None])

    @triton.jit
    def _attn(QKV, O,
              S: tl.constexpr, D: tl.constexpr, HD: tl.constexpr,
              SCALE: tl.constexpr, BLOCK_M: tl.constexpr,
              BN: tl.constexpr, BT: tl.constexpr):
        """Non-causal attention over the packed projection [B*S, 3*HD].

        Head ``h`` lives at column offsets ``h*D`` / ``HD + h*D`` /
        ``2*HD + h*D``.  The key axis is covered by two ``BN`` tiles plus a
        ``BT`` tail (2*BN + BT == S), so no tensor-core work is spent on
        masked-off keys and every load is a contiguous vector load.
        """
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)
        ST: tl.constexpr = 3 * HD
        qbase = QKV + pid_b * S * ST + pid_h * D
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = offs_m < S
        # Clamp instead of masking the load: the tail rows are thrown away by
        # the store mask, and an unmasked load keeps the vectorised path.
        mrow = tl.minimum(offs_m, S - 1)
        offs_d = tl.arange(0, D)
        n1 = tl.arange(0, BN)
        n2 = BN + tl.arange(0, BN)
        n3 = 2 * BN + tl.arange(0, BT)

        q = tl.load(qbase + mrow[:, None] * ST + offs_d[None, :])
        # All three key tiles are issued before the first dot so their misses
        # overlap; the CTAs here are small and there is little else to hide them.
        kb = qbase + HD
        k1 = tl.load(kb + n1[:, None] * ST + offs_d[None, :])
        k2 = tl.load(kb + n2[:, None] * ST + offs_d[None, :])
        k3 = tl.load(kb + n3[:, None] * ST + offs_d[None, :])
        s1 = tl.dot(q, k1.T) * SCALE
        s2 = tl.dot(q, k2.T) * SCALE
        s3 = tl.dot(q, k3.T) * SCALE

        mx = tl.maximum(tl.maximum(tl.max(s1, 1), tl.max(s2, 1)), tl.max(s3, 1))
        p1 = tl.exp(s1 - mx[:, None])
        p2 = tl.exp(s2 - mx[:, None])
        p3 = tl.exp(s3 - mx[:, None])
        lse = tl.sum(p1, 1) + tl.sum(p2, 1) + tl.sum(p3, 1)

        vb = qbase + 2 * HD
        v1 = tl.load(vb + n1[:, None] * ST + offs_d[None, :])
        v2 = tl.load(vb + n2[:, None] * ST + offs_d[None, :])
        v3 = tl.load(vb + n3[:, None] * ST + offs_d[None, :])
        acc = tl.dot(p1.to(tl.float16), v1)
        acc = tl.dot(p2.to(tl.float16), v2, acc)
        acc = tl.dot(p3.to(tl.float16), v3, acc)
        acc = acc / lse[:, None]
        tl.store(O + (pid_b * S + offs_m[:, None]) * HD + pid_h * D + offs_d[None, :],
                 acc.to(tl.float16), mask=m_mask[:, None])


class _Plan:
    """Per-shape launch plan, optionally backed by a captured CUDA graph."""

    graph = None
    body = None


class OasisSpatialAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: OasisRotaryEmbedding,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.to_qkv = Linear(dim, dim_head * heads * 3, bias=False)
        self.to_out = Linear(dim_head * heads, dim, bias=True)
        self.rotary_emb = rotary_emb
        self.attn = DenseAttention(backend="sdpa")
        self._rope_cache: dict = {}
        self._plans: dict = {}

    # ---------------------------------------------------------------- rotary
    def _cos_sin(self, height: int, width: int):
        """Cached ``(cos, sin)`` for the even rotary lanes, shape [H*W, D/2]."""
        key = (height, width)
        hit = self._rope_cache.get(key)
        if hit is None:
            freqs = self.rotary_emb.get_axial_freqs(height, width)
            flat = freqs.reshape(height * width, -1)
            hit = (flat.cos()[:, 0::2].float().contiguous(),
                   flat.sin()[:, 0::2].float().contiguous())
            self._rope_cache[key] = hit
        return hit

    # ------------------------------------------------------------- reference
    def _forward_ref(self, x: torch.Tensor) -> torch.Tensor:
        bsz, time, height, width, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.reshape(bsz * time, height, width, self.heads, -1).permute(0, 3, 1, 2, 4)
        k = k.reshape(bsz * time, height, width, self.heads, -1).permute(0, 3, 1, 2, 4)
        v = v.reshape(bsz * time, height, width, self.heads, -1).permute(0, 3, 1, 2, 4)
        freqs = self.rotary_emb.get_axial_freqs(height, width)
        q = oasis_apply_rotary_emb(freqs, q)
        k = oasis_apply_rotary_emb(freqs, k)
        q = q.reshape(bsz * time, self.heads, height * width, -1).transpose(1, 2)
        k = k.reshape(bsz * time, self.heads, height * width, -1).transpose(1, 2)
        v = v.reshape(bsz * time, self.heads, height * width, -1).transpose(1, 2)
        out = self.attn(q, k, v, causal=False)
        out = out.reshape(bsz, time, height, width, self.heads, -1).reshape(
            bsz, time, height, width, -1)
        return self.to_out(out.to(q.dtype))

    # ------------------------------------------------------------------ plan
    def _build_plan(self, x: torch.Tensor):
        bsz, time, height, width, dim = x.shape
        D, S = self.dim_head, height * width
        DH = D // 2
        HD = self.heads * D
        B, M = bsz * time, bsz * time * S
        N_QKV = self.to_qkv.weight.shape[0]
        N_OUT = self.to_out.weight.shape[0]

        # Key axis as two power-of-two tiles plus a power-of-two tail.
        BN = triton.next_power_of_2(S) // 4
        BT = S - 2 * BN
        if BN < 16 or BT <= 0 or (BT & (BT - 1)) != 0:
            return None
        qcfg = _pick(M, N_QKV)
        ocfg = _pick(M, N_OUT)
        if qcfg["BLOCK_N"] % D or HD % qcfg["BLOCK_N"]:
            return None

        cos, sin = self._cos_sin(height, width)
        wq, wo, bo = self.to_qkv.weight, self.to_out.weight, self.to_out.bias
        dev = x.device
        qkv = torch.empty((M, N_QKV), device=dev, dtype=torch.float16)
        o = torch.empty((M, HD), device=dev, dtype=torch.float16)

        acfg = _pick_by_m(_ATTN_CFGS, M)
        abm = acfg.pop("BLOCK_M")
        g_attn = (triton.cdiv(S, abm), self.heads, B)
        g_qkv = (triton.cdiv(M, qcfg["BLOCK_M"]) * (N_QKV // qcfg["BLOCK_N"]),)
        g_out = (triton.cdiv(M, ocfg["BLOCK_M"]) * (N_OUT // ocfg["BLOCK_N"]),)

        plan = _Plan()
        plan.out_shape = (bsz, time, height, width, N_OUT)
        plan.out_dtype = x.dtype
        plan.out_device = dev
        plan.M, plan.dim, plan.N_OUT = M, dim, N_OUT
        plan.keep = [qkv, o, cos, sin]

        def body(xf, y, tma, hold, c_slot):
            if tma is not None:
                _gemm_tma[g_qkv](tma[0], tma[1], qkv, wq, cos, sin, hold, M, dim,
                                 N_QKV, S=S, DH=DH, ROPE_END=2 * HD, C_SLOT=-1,
                                 HAS_BIAS=False, **qcfg)
            else:
                _gemm[g_qkv](xf, wq, qkv, wq, cos, sin, hold, M, dim, N_QKV,
                             S=S, DH=DH, ROPE_END=2 * HD, C_SLOT=-1,
                             HAS_BIAS=False, **qcfg)
            _attn[g_attn](qkv, o, S=S, D=D, HD=HD, SCALE=D ** -0.5,
                          BLOCK_M=abm, BN=BN, BT=BT, **acfg)
            if tma is not None:
                _gemm_tma[g_out](tma[2], tma[3], y, bo, cos, sin, hold, M, HD,
                                 N_OUT, S=S, DH=DH, ROPE_END=0, C_SLOT=c_slot,
                                 HAS_BIAS=True, **ocfg)
            else:
                _gemm[g_out](o, wo, y, bo, cos, sin, hold, M, HD, N_OUT,
                             S=S, DH=DH, ROPE_END=0, C_SLOT=c_slot,
                             HAS_BIAS=True, **ocfg)

        hold_dev = torch.zeros(1, dtype=torch.int64, device=dev)
        plan.hold_dev = hold_dev
        plan.body = body
        plan.tma = None
        plan.keep.append(hold_dev)
        if _MODE != "graph":
            return plan

        try:
            hold_host = torch.zeros(1, dtype=torch.int64, pin_memory=True)
            # The input is staged through a fixed buffer: a TMA descriptor bakes
            # its base address, and the copy is cheaper than the alternatives.
            # ``ys`` only has to be a valid address for capture -- every replay
            # stores through the slot instead.
            xs = torch.empty((M, dim), device=dev, dtype=torch.float16)
            ys = torch.empty((M, N_OUT), device=dev, dtype=torch.float16)
            hold_host[0] = ys.data_ptr()
            hold_dev.copy_(hold_host)
            tma = None
            if _USE_TMA:
                try:
                    tma = (TensorDescriptor.from_tensor(xs, [qcfg["BLOCK_M"], qcfg["BLOCK_K"]]),
                           TensorDescriptor.from_tensor(wq, [qcfg["BLOCK_N"], qcfg["BLOCK_K"]]),
                           TensorDescriptor.from_tensor(o, [ocfg["BLOCK_M"], ocfg["BLOCK_K"]]),
                           TensorDescriptor.from_tensor(wo, [ocfg["BLOCK_N"], ocfg["BLOCK_K"]]))
                    body(xs, ys, tma, hold_dev, 0)
                    torch.cuda.synchronize()
                except Exception:
                    tma = None
            for _ in range(2):
                body(xs, ys, tma, hold_dev, 0)
            torch.cuda.synchronize()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                body(xs, ys, tma, hold_dev, 0)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                hold_dev.copy_(hold_host, non_blocking=True)
                body(xs, ys, tma, hold_dev, 0)
            torch.cuda.synchronize()
            plan.graph = graph
            plan.stage_in = xs
            plan.hold_np = hold_host.numpy()
            plan.tma = tma
            plan.keep += [graph, hold_host, xs, ys]
        except Exception:
            plan.graph = None
        return plan

    # --------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, time, height, width, dim = x.shape
        D = self.dim_head
        S = height * width
        if (not _HAS_TRITON or not x.is_cuda or x.dtype != torch.float16
                or D % 2 or D > 128 or S > 1024 or self.heads * D != dim
                or not x.is_contiguous()
                or self.to_qkv.weight.dtype != torch.float16):
            return self._forward_ref(x)

        key = (bsz, time, height, width, dim)
        plan = self._plans.get(key)
        if plan is None:
            try:
                plan = self._build_plan(x)
            except Exception:
                plan = False
            self._plans[key] = plan
        if not plan:
            return self._forward_ref(x)

        y = torch.empty(plan.out_shape, device=plan.out_device, dtype=plan.out_dtype)
        if plan.graph is not None:
            plan.stage_in.copy_(x.view(plan.M, plan.dim))
            plan.hold_np[0] = y.data_ptr()
            plan.graph.replay()
            return y
        plan.body(x.view(plan.M, plan.dim), y.view(plan.M, plan.N_OUT), None,
                  plan.hold_dev, -1)
        return y
