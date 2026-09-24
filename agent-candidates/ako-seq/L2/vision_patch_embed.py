"""Vision patch embedding for Qwen VL models.

Flattens 3D video/image patches via Conv3d weight reshaped into a linear projection.

Unified across Qwen2-VL and Qwen3-VL:
  - bias: Qwen2-VL uses bias=False, Qwen3-VL uses bias=True.

The operator is one compute-bound bf16 GEMM with two of three dimensions pinned
by the captured init (``embed_dim=1152``, ``in_channels*temporal_patch_size*
patch_size**2 = 1536``):  ``y[M, 1152] = x[M, 1536] @ W[1152, 1536].T + b``.
``Conv3d`` is a parameter holder only -- it is never invoked for compute.

`_gemm` below is a hand-written Blackwell (sm_100) tcgen05 GEMM in the CUTLASS
Python DSL: warp-specialised (TMA / MMA / epilogue warps), 2-SM MMA over a
(2,1,1) cluster, TMA-pipelined A/B loads, fp32 accumulation in TMEM, bias fused
into the epilogue and a TMA store of the bf16 result.  The frozen L1 `Matmul`
cannot help here (its `_MM_CFG` holds only tf32 *batched* entries, so every
captured shape falls through to `F.linear`), and its docstring records why
Triton is the wrong tool: it cannot express the Blackwell-native tile shapes and
2-CTA clusters that `nvjet_sm100_*` uses.  tcgen05 via CuTe DSL can.

Two structural edges over a generic library kernel:

* **N is 1152 = 6x192**, so an N tile of 192 tiles the output panel exactly --
  no masked MMA work at the N edge, unlike the 256-wide tiles a general kernel
  prefers (1152/256 = 4.5).
* **The tile walk is N-fastest**, so the 6 clusters sharing an M block run
  concurrently and read that A block from L2 instead of DRAM, and the whole
  3.4 MB B panel stays L2-resident for the life of the kernel.  This is the
  single biggest lever in the kernel, not a detail: M-fastest costs 1.6x at
  M=64680 (223 us vs 138), because A is 199 MB there and every one of its 6
  re-reads then comes from DRAM.

`_CFG` is the dispatch gate: a shape reaches `_gemm` only through a tuned entry,
i.e. only where this kernel was *measured faster than the reference on this GPU*
and verified against it.  Everything else defers to the L1 `Matmul`.
See ITERATIONS.md for the per-shape measurement table.
"""

# NB: no `from __future__ import annotations` -- `@cute.struct` below reads the
# annotations of SharedStorage eagerly and PEP-563 strings break it.
import os

import torch
import torch.nn as nn

from ..L1.conv3d import Conv3d
from ..L1.linear import Matmul

_K = 1536          # in_channels * temporal_patch_size * patch_size**2
_N = 1152          # embed_dim

try:
    import cutlass
    import cutlass.cute as cute
    import cutlass.pipeline as pipeline
    import cutlass.utils as utils
    import cutlass.utils.blackwell_helpers as sm100_utils
    from cutlass.cute.nvgpu import cpasync, tcgen05
    from cutlass.cute.runtime import make_ptr
    from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

    _CUTE_ERR = None
except Exception as _e:                                    # pragma: no cover
    _CUTE_ERR = _e


# ---------------------------------------------------------------------------
# Kernel factory.  Every tile/pipeline parameter is a Python constant captured
# by the closure, so it reaches the DSL as a compile-time constant.
# ---------------------------------------------------------------------------
def _build(tile_m, tile_n, tile_k, a_stages, b_stages, epi_stages, acc_stages,
           use_2cta, cluster_m, raster_n_first, use_pdl, max_active_clusters):
    """Return a `@cute.jit` host launcher for one (tile, pipeline) config.

    A and B ride *separate* TMA->MMA pipelines so their depths can differ.  The
    tuned config keeps them equal: what the mainloop actually needs is k tiles
    *in flight*, which is `min(a_stages, b_stages)`, so trading B depth for A
    depth only moves the stall (measured: 8/6 and 10/4 are 4% and 20% slower
    than 7/7 even though they buy more total SMEM).  7/7 is the deepest equal
    pair that fits, since one stage costs (128 + 96) x 64 x 2 B per CTA.

    `cluster_m` > 1 puts `cluster_m` M tiles in one cluster so their shared B
    tile is TMA-multicast instead of fetched once per CTA, cutting L2->SMEM
    traffic per flop.  Measured slower here (the B panel is already L2-resident,
    so the multicast saves traffic that was not costing anything, and the wider
    cluster costs scheduling granularity) -- kept as a knob, not used."""

    io_dtype = cutlass.BFloat16
    acc_dtype = cutlass.Float32

    cluster_v = 2 if use_2cta else 1
    cluster_size = cluster_v * cluster_m
    cluster_shape_mnk = (cluster_size, 1, 1)
    cta_group = tcgen05.CtaGroup.TWO if use_2cta else tcgen05.CtaGroup.ONE
    cta_tile_shape_mnk = (tile_m // cluster_v, tile_n, tile_k)

    mma_inst_shape_mnk = (tile_m, tile_n, 16)
    mma_tiler_mnk = (tile_m, tile_n, tile_k)

    n_tiles_n = _N // tile_n
    k_tiles = _K // tile_k

    epi_warps = (0, 1, 2, 3)
    mma_warp_id = 4
    tma_warp_id = 5
    threads_in_epilogue = 32 * len(epi_warps)
    threads_per_cta = 32 * 6

    epilog_sync_bar_id = 1
    tmem_alloc_sync_bar_id = 2

    n_tmem_cols = 1 << max(5, (acc_stages * tile_n - 1).bit_length())
    assert n_tmem_cols <= 512, "accumulator does not fit in TMEM"

    def _decode(work, n_tiles_m_grp):
        """Work id -> (m tile group, n tile).

        N-fastest is the default: the `n_tiles_n` clusters that share an M block
        are then co-resident, so that A block is read from DRAM once and served
        from L2 for the other `n_tiles_n - 1` tiles.  M-fastest is kept as a
        knob because it is the better order once A no longer fits in L2.
        """
        if raster_n_first:
            return work // n_tiles_n, work % n_tiles_n
        return work % n_tiles_m_grp, work // n_tiles_m_grp

    @cute.struct
    class SharedStorage:
        a_mbar_ptr: cute.struct.MemRange[cutlass.Int64, a_stages * 2]
        b_mbar_ptr: cute.struct.MemRange[cutlass.Int64, b_stages * 2]
        acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stages * 2]
        tmem_dealloc_mbar: cutlass.Int64
        tmem_holding_buffer: cutlass.Int32

    @cute.kernel
    def kernel(
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC: cute.Tensor,
        mBias: cute.Tensor,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        c_smem_layout_kind: cutlass.Constexpr,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        cta_layout_vmnk: cute.Layout,
        n_tiles_m_grp: cutlass.Int32,
        n_clusters: cutlass.Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
        cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

        mma_tile_coord_v = bidx % cluster_v
        is_leader_cta = mma_tile_coord_v == 0
        cluster_id = bidx // cluster_size
        # rank of this CTA's M tile within the cluster's group of `cluster_m`
        m_rank = cta_in_cluster_coord_vmnk[1]

        # Total work and this cluster's share.  The walk is N-fastest so the
        # `n_tiles_n` clusters that share an M block are co-resident and hit L2
        # on the same A tile.
        n_work = n_tiles_m_grp * n_tiles_n
        n_rounds = (n_work + n_clusters - 1) // n_clusters

        if warp_idx == tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        num_mcast_participants = (
            cute.size(cta_layout_vmnk, mode=[1])
            + cute.size(cta_layout_vmnk, mode=[2]) - 1
        )
        tma_mcast_mask_a = cpasync.create_tma_multicast_mask(
            cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=2)
        tma_mcast_mask_b = cpasync.create_tma_multicast_mask(
            cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=1)

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        epilogue_sync_barrier = pipeline.NamedBarrier(
            barrier_id=epilog_sync_bar_id, num_threads=threads_in_epilogue)
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=tmem_alloc_sync_bar_id,
            num_threads=32 * (1 + len(epi_warps)))
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buffer.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=epi_warps[0],
            is_two_cta=use_2cta,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        v_size = cute.size(cta_layout_vmnk, mode=[0])
        a_copy_bytes = cute.size_in_bytes(
            io_dtype, cute.select(a_smem_layout, mode=[0, 1, 2])) * v_size
        b_copy_bytes = cute.size_in_bytes(
            io_dtype, cute.select(b_smem_layout, mode=[0, 1, 2])) * v_size

        a_producer, a_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.a_mbar_ptr.data_ptr(),
            num_stages=a_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, size=num_mcast_participants),
            tx_count=a_copy_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
        ).make_participants()
        b_producer, b_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.b_mbar_ptr.data_ptr(),
            num_stages=b_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, size=num_mcast_participants),
            tx_count=b_copy_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
        ).make_participants()

        acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=acc_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                size=cute.size(cta_layout_vmnk, mode=[0]) * len(epi_warps)),
            cta_layout_vmnk=cta_layout_vmnk,
        ).make_participants()

        pipeline_init_arrive(cluster_shape_mn=cluster_shape_mnk, is_relaxed=True)

        sA = smem.allocate_tensor(io_dtype, a_smem_layout.outer,
                                  byte_alignment=128, swizzle=a_smem_layout.inner)
        sB = smem.allocate_tensor(io_dtype, b_smem_layout.outer,
                                  byte_alignment=128, swizzle=b_smem_layout.inner)
        sC = smem.allocate_tensor(io_dtype, epi_smem_layout_staged.outer,
                                  byte_alignment=128,
                                  swizzle=epi_smem_layout_staged.inner)

        # (bM, bK, RestM, RestK) / (bN, bK, RestN, RestK) / (bM, bN, RestM, RestN)
        gA = cute.local_tile(mA, cute.slice_(mma_tiler_mnk, (None, 0, None)),
                             (None, None))
        gB = cute.local_tile(mB, cute.slice_(mma_tiler_mnk, (0, None, None)),
                             (None, None))
        gC = cute.local_tile(mC, cute.slice_(mma_tiler_mnk, (None, None, 0)),
                             (None, None))
        gBias = cute.local_tile(mBias, cute.slice_(mma_tiler_mnk, (None, None, 0)),
                                (None, None))

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)
        tCgC = thr_mma.partition_C(gC)
        tCgBias = thr_mma.partition_C(gBias)

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, acc_stages))

        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a, cta_in_cluster_coord_vmnk[2],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
            cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3))
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b, cta_in_cluster_coord_vmnk[1],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
            cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3))

        gC_epi = cute.flat_divide(tCgC[((None, None), 0, 0, None, None)], epi_tile)
        tCsC, tCgC_tma = cpasync.tma_partition(
            tma_atom_c, 0, cute.make_layout(1),
            cute.group_modes(sC, 0, 2), cute.group_modes(gC_epi, 0, 2))

        pipeline_init_wait(cluster_shape_mn=cluster_shape_mnk)

        # -------------------------------------------------------------- TMA
        if warp_idx == tma_warp_id:
            if cutlass.const_expr(use_pdl):
                # A is produced by whatever kernel ran before us; block until it
                # is visible.  Everything above this line -- descriptor
                # prefetch, TMEM allocation, pipeline/cluster barrier init --
                # has already overlapped with that kernel's tail.
                cute.arch.griddepcontrol_wait()
            for r in cutlass.range(n_rounds):
                work = cluster_id + r * n_clusters
                if work < n_work:
                    m_grp, tile_n_idx = _decode(work, n_tiles_m_grp)
                    tile_m_idx = m_grp * cluster_m + m_rank
                    tAgA_slice = tAgA[(None, tile_m_idx, None)]
                    tBgB_slice = tBgB[(None, tile_n_idx, None)]
                    for k in cutlass.range_constexpr(k_tiles):
                        ha = a_producer.acquire_and_advance()
                        cute.copy(tma_atom_a, tAgA_slice[(None, k)],
                                  tAsA[(None, ha.index)],
                                  tma_bar_ptr=ha.barrier,
                                  mcast_mask=tma_mcast_mask_a)
                        hb = b_producer.acquire_and_advance()
                        cute.copy(tma_atom_b, tBgB_slice[(None, k)],
                                  tBsB[(None, hb.index)],
                                  tma_bar_ptr=hb.barrier,
                                  mcast_mask=tma_mcast_mask_b)
            a_producer.tail()
            b_producer.tail()

        # -------------------------------------------------------------- MMA
        elif warp_idx == mma_warp_id:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            for r in cutlass.range(n_rounds):
                work = cluster_id + r * n_clusters
                if work < n_work and is_leader_cta:
                    acc_empty = acc_producer.acquire_and_advance()
                    tCtAcc = tCtAcc_base[(None, None, None, acc_empty.index)]
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for k in cutlass.range_constexpr(k_tiles):
                        ha = a_consumer.wait_and_advance()
                        hb = b_consumer.wait_and_advance()
                        for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                            cute.gemm(tiled_mma, tCtAcc,
                                      tCrA[(None, None, kb, ha.index)],
                                      tCrB[(None, None, kb, hb.index)], tCtAcc)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        ha.release()
                        hb.release()
                    acc_empty.commit()
            acc_producer.tail()

        # --------------------------------------------------------- epilogue
        elif warp_idx < mma_warp_id:
            tmem.allocate(n_tmem_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            epilogue_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=epi_stages,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, size=threads_in_epilogue),
            )
            copy_atom_t2r = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition.x32, tcgen05.Pack.NONE),
                acc_dtype)
            if cutlass.const_expr(use_pdl):
                cute.arch.griddepcontrol_wait()

            for r in cutlass.range(n_rounds):
                work = cluster_id + r * n_clusters
                if work < n_work:
                    m_grp, tile_n_idx = _decode(work, n_tiles_m_grp)
                    tile_m_idx = m_grp * cluster_m + m_rank

                    acc_full = acc_consumer.wait_and_advance()
                    tCtAcc = tCtAcc_base[(None, None, None, acc_full.index)]
                    tCtAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0)],
                                                  epi_tile)
                    tCgC_epi = cute.flat_divide(
                        tCgC[((None, None), 0, 0, tile_m_idx, tile_n_idx)], epi_tile)
                    tCgBias_epi = cute.flat_divide(
                        tCgBias[((None, None), 0, 0, tile_m_idx, tile_n_idx)],
                        epi_tile)
                    tCgC_tma_cur = tCgC_tma[(None, None, None, tile_m_idx,
                                             tile_n_idx)]

                    tiled_copy_t2r = tcgen05.make_tmem_copy(
                        copy_atom_t2r, tCtAcc_epi[(None, None, 0, 0)])
                    thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
                    tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc_epi)
                    tTR_gC = thr_copy_t2r.partition_D(tCgC_epi)
                    tTR_gBias = thr_copy_t2r.partition_D(tCgBias_epi)
                    frag_shape = tTR_gC[(None, None, None, 0, 0)].shape
                    tTR_rAcc = cute.make_rmem_tensor(frag_shape, acc_dtype)
                    tTR_rBias = cute.make_rmem_tensor(frag_shape, io_dtype)
                    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                    tTR_gBias = cute.group_modes(tTR_gBias, 3,
                                                 cute.rank(tTR_gBias))

                    copy_atom_r2s = sm100_utils.get_smem_store_op(
                        c_smem_layout_kind, acc_dtype, acc_dtype, tiled_copy_t2r)
                    tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s,
                                                            tiled_copy_t2r)
                    thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                    tRS_sC = thr_copy_r2s.partition_D(sC)
                    tRS_rAcc = tiled_copy_r2s.retile(tTR_rAcc)
                    tRS_rBias = tiled_copy_r2s.retile(tTR_rBias)
                    tRS_rC = cute.make_rmem_tensor(tRS_rAcc.shape, io_dtype)
                    tCgC_grouped = cute.group_modes(tCgC_tma_cur, 1,
                                                    cute.rank(tCgC_tma_cur))

                    subtiles = cute.size(tTR_tAcc.shape, mode=[3])
                    for st in cutlass.range_constexpr(subtiles):
                        cute.copy(tiled_copy_t2r,
                                  tTR_tAcc[(None, None, None, st)], tTR_rAcc)
                        c_buf = st % epi_stages
                        cute.autovec_copy(tTR_gBias[(None, None, None, st)],
                                          tTR_rBias)
                        # fp32 accumulate + bias, then one narrowing cast
                        tRS_rC.store(
                            (tRS_rAcc.load() + tRS_rBias.load().to(acc_dtype))
                            .to(io_dtype))
                        cute.copy(tiled_copy_r2s, tRS_rC,
                                  tRS_sC[(None, None, None, c_buf)])
                        cute.arch.fence_view_async_shared()
                        epilogue_sync_barrier.arrive_and_wait()
                        if warp_idx == epi_warps[0]:
                            cute.copy(tma_atom_c, tCsC[(None, c_buf)],
                                      tCgC_grouped[(None, st)])
                            epilogue_pipeline.producer_commit()
                            epilogue_pipeline.producer_acquire()
                        epilogue_sync_barrier.arrive_and_wait()

                    with cute.arch.elect_one():
                        acc_full.release()

            epilogue_pipeline.producer_tail()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

    @cute.jit
    def host(a_ptr, b_ptr, bias_ptr, c_ptr, m: cutlass.Int32):
        mA = cute.make_tensor(a_ptr, cute.make_layout((m, _K), stride=(_K, 1)))
        mB = cute.make_tensor(b_ptr, cute.make_layout((_N, _K), stride=(_K, 1)))
        mC = cute.make_tensor(c_ptr, cute.make_layout((m, _N), stride=(_N, 1)))
        # Bias broadcast over M costs nothing: stride 0 on the M mode lets the
        # epilogue partition it exactly like C.
        mBias = cute.make_tensor(bias_ptr,
                                 cute.make_layout((m, _N), stride=(0, 1)))

        op = tcgen05.MmaF16BF16Op(
            io_dtype, acc_dtype, mma_inst_shape_mnk, cta_group,
            tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K)
        tiled_mma = cute.make_tiled_mma(op)

        a_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, mma_tiler_mnk, io_dtype, a_stages)
        b_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, mma_tiler_mnk, io_dtype, b_stages)
        c_smem_layout_kind = utils.LayoutEnum.from_tensor(mC)

        cta_layout_vmnk = cute.tiled_divide(cute.make_layout(cluster_shape_mnk),
                                            (tiled_mma.thr_id,))

        tma_op = cpasync.CopyBulkTensorTileG2SMulticastOp(cta_group)
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            tma_op, mA, cute.slice_(a_smem_layout, (None, None, None, 0)),
            mma_tiler_mnk, tiled_mma, cta_layout_vmnk.shape)
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            tma_op, mB, cute.slice_(b_smem_layout, (None, None, None, 0)),
            mma_tiler_mnk, tiled_mma, cta_layout_vmnk.shape)

        epi_tile = utils.compute_epilogue_tile_shape(
            cta_tile_shape_mnk, use_2cta, c_smem_layout_kind, io_dtype)
        epi_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            io_dtype, c_smem_layout_kind, epi_tile, epi_stages)
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mC,
            cute.slice_(epi_smem_layout_staged, (None, None, 0)), epi_tile)

        n_tiles_m = (m + tile_m - 1) // tile_m
        n_tiles_m_grp = (n_tiles_m + cluster_m - 1) // cluster_m
        n_work = n_tiles_m_grp * n_tiles_n
        n_clusters = cutlass.min(n_work, cutlass.Int32(max_active_clusters))

        kernel(
            tiled_mma, tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b,
            tma_atom_c, tma_tensor_c, mBias,
            a_smem_layout, b_smem_layout, c_smem_layout_kind,
            epi_smem_layout_staged, epi_tile, cta_layout_vmnk,
            n_tiles_m_grp, n_clusters,
        ).launch(
            grid=(n_clusters * cluster_size, 1, 1),
            block=(threads_per_cta, 1, 1),
            cluster=cluster_shape_mnk,
            use_pdl=use_pdl,
        )

    return host


# ---------------------------------------------------------------------------
# Per-shape configs: M-range -> (tile_m, tile_n, tile_k, a_stages, b_stages,
#                                epi_stages, acc_stages, use_2cta, cluster_m,
#                                raster_n_first, use_pdl)
#
# `_CFG` is the dispatch gate: only M values it covers run on `_gemm`; anything
# else uses the L1 `Matmul` (which itself defers to the reference for these
# shapes).  Entries were picked by a sweep measured with the benchmark's own
# timing loop -- see ITERATIONS.md.
# ---------------------------------------------------------------------------
_BIG = (256, 192, 64, 7, 7, 2, 2, True, 1, True, True)

# Measured over the captured M spread (26 shapes, see ITERATIONS.md).  This
# kernel's time is linear in M -- 9.4 us + 2.08 us per 1000 rows -- so its
# advantage grows with M as the fixed cost amortises: it reaches ~1.19x by
# M=48000 but is 0.77x at M=3072, where the fixed cost is half the runtime and
# the tile grid cannot fill the 74 clusters (M=1760 uses 42 of them).  16000 is
# where it turns over for good; the reference wins below that.
# The upper bound only keeps `m * 1536` inside Int32 for the layouts below;
# no captured shape comes near it.
_CFG = ((16000, 1_000_000, _BIG),)

_COMPILED: dict = {}


def _cfg_for(m):
    env = os.environ.get("VPE_CFG")           # dev sweeps only
    if env:
        v = [int(t) for t in env.split(",")]
        return (*v[:7], bool(v[7]), v[8], bool(v[9]), bool(v[10]))
    for lo, hi, cfg in _CFG:
        if lo <= m <= hi:
            return cfg
    return None


def _compiled(cfg):
    fn = _COMPILED.get(cfg)
    if fn is None:
        mac = utils.HardwareInfo().get_max_active_clusters(
            (2 if cfg[7] else 1) * cfg[8])
        host = _build(*cfg, max_active_clusters=mac)
        a = make_ptr(cutlass.BFloat16, 0, cute.AddressSpace.gmem, assumed_align=16)
        b = make_ptr(cutlass.BFloat16, 0, cute.AddressSpace.gmem, assumed_align=16)
        bs = make_ptr(cutlass.BFloat16, 0, cute.AddressSpace.gmem, assumed_align=16)
        c = make_ptr(cutlass.BFloat16, 0, cute.AddressSpace.gmem, assumed_align=16)
        fn = cute.compile(host, a, b, bs, c, cutlass.Int32(1))
        _COMPILED[cfg] = fn
    return fn


def _gemm(x, w, bias, cfg):
    m = x.shape[0]
    y = torch.empty(m, _N, dtype=x.dtype, device=x.device)
    fn = _compiled(cfg)
    fn(make_ptr(cutlass.BFloat16, x.data_ptr(), cute.AddressSpace.gmem,
                assumed_align=16),
       make_ptr(cutlass.BFloat16, w.data_ptr(), cute.AddressSpace.gmem,
                assumed_align=16),
       make_ptr(cutlass.BFloat16, bias.data_ptr(), cute.AddressSpace.gmem,
                assumed_align=16),
       make_ptr(cutlass.BFloat16, y.data_ptr(), cute.AddressSpace.gmem,
                assumed_align=16),
       cutlass.Int32(m))
    return y


class VisionPatchEmbed(nn.Module):
    def __init__(self, patch_size: int, temporal_patch_size: int,
                 in_channels: int, embed_dim: int, bias: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_size = in_channels * temporal_patch_size * patch_size * patch_size
        kernel = (temporal_patch_size, patch_size, patch_size)
        self.proj = Conv3d(in_channels, embed_dim, kernel, bias=bias)
        self.linear = Matmul()
        self._w2d = None
        self._w_src = None

    def _weight2d(self):
        """`Conv3d.weight` reshaped to [embed_dim, input_size], cached.

        The reshape is pure metadata but it is not free in Python, and it ran
        once per forward on a kernel whose smallest captured case is ~13 us.
        Keyed on the parameter object, so `load_state_dict` (which copies in
        place) keeps the cached view while rebinding the parameter drops it.
        """
        w = self.proj.weight
        if self._w_src is not w:
            self._w2d = w.view(self.embed_dim, self.input_size)
            self._w_src = w
        return self._w2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._weight2d()
        bias = self.proj.bias
        if (_CUTE_ERR is None and x.ndim == 2 and x.shape[1] == _K
                and self.embed_dim == _N and bias is not None
                and x.dtype is torch.bfloat16 and w.dtype is torch.bfloat16
                and bias.dtype is torch.bfloat16 and x.is_contiguous()
                and w.is_contiguous() and bias.is_contiguous()):
            cfg = _cfg_for(x.shape[0])
            if cfg is not None:
                return _gemm(x, w, bias, cfg)
        if x.ndim != 2 or x.shape[1] != self.input_size:
            # same reshape the baseline always does; elided when it is a no-op,
            # which is every captured case
            x = x.view(x.shape[0], self.input_size)
        return self.linear(x, w, bias)
