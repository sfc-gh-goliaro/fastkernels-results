"""Vision patch embedding for Qwen VL models.

Flattens 3D video/image patches via Conv3d weight reshaped into a linear projection.

Unified across Qwen2-VL and Qwen3-VL:
  - bias: Qwen2-VL uses bias=False, Qwen3-VL uses bias=True.

The projection is a single bf16 GEMM ``x[M, K] @ W[N, K]^T (+ bias)`` with a
small, fixed ``N``/``K`` (e.g. 1152 x 1536) and a large, varying ``M``. cuBLAS
leaves ~20% of the tensor cores idle on that aspect ratio, so the forward runs a
hand-written SM100 kernel instead: a warp-specialized (TMA / MMA / epilogue)
persistent GEMM using ``tcgen05`` 2-SM MMA with TMEM accumulators, written in
CuTe DSL. The N tile is chosen to divide ``embed_dim`` exactly (192 for 1152) so
no MMA work is wasted, tiles are rasterized along N so each row-block of ``x``
is fetched from HBM once, and the bias is staged in SMEM and folded into the
epilogue. Anything the kernel does not cover (small M, non-bf16, non-SM100)
falls back to ``F.linear``.
"""

from __future__ import annotations

import functools

import torch
import torch.nn as nn

from ..L1.conv3d import Conv3d
from ..L1.linear import Matmul

try:
    import cutlass
    import cutlass.cute as cute
    import cutlass.pipeline as pipeline
    import cutlass.utils as utils
    import cutlass.utils.blackwell_helpers as sm100_utils
    from cutlass.cute.nvgpu import cpasync, tcgen05
    from cutlass.cute.runtime import from_dlpack
    from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

    _HAVE_CUTE = True
except Exception:  # pragma: no cover - kernel simply unavailable
    _HAVE_CUTE = False


if _HAVE_CUTE:
    IO = cutlass.BFloat16
    ACC = cutlass.Float32

    # Tile / pipeline configuration (tuned on B200 for N=1152, K=1536).
    #   * MMA tile 256x<N>x64 with a 2-CTA cluster: 128 accumulator rows per SM,
    #     B multicast across the pair, and the largest N tile that divides
    #     embed_dim exactly (192 for 1152) so no MMA column is wasted.
    #   * 7 A/B stages is the deepest pipeline that still fits in 227 KB of SMEM.
    #   * 2 accumulator stages (2 x 192 of 512 TMEM columns) let the epilogue of
    #     one tile overlap the mainloop of the next.
    _MMA_TILER_N = (192, 128, 64)
    _STAGES = (7, 2, 2)  # (A/B, accumulator, epilogue) pipeline depths
    _CLUSTER = (2, 1, 1)
    _K_UNROLL = 2
    # Below this many rows the 256-row cluster tiles cannot fill the GPU (and the
    # fixed prologue/epilogue dominates), so cuBLAS' finer tiles win; defer to it.
    _MIN_ROWS = 12000
    # A persistent kernel runs ceil(tiles / clusters) waves, so a shape whose tile
    # count leaves a badly under-filled last wave loses more to the tail than the
    # mainloop gains. Defer those to cuBLAS too (measured crossover ~0.88).
    _MIN_WAVE_FILL = 0.88


    @cute.kernel()
    def kernel(
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC: cute.Tensor,
        mBias: cute.Tensor,
        N_TILE: cutlass.Constexpr,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        c_smem_layout_kind: cutlass.Constexpr,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        cta_layout_vmnk: cute.Layout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        mma_tiler_mnk: cutlass.Constexpr,
        ab_stages: cutlass.Constexpr,
        acc_stages: cutlass.Constexpr,
        epi_stages: cutlass.Constexpr,
        cluster_shape_mnk: cutlass.Constexpr,
        HAS_BIAS: cutlass.Constexpr,
        SharedStorage: cutlass.Constexpr,
    ):
        threads_in_epilogue = 128
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
        cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        mma_tile_coord_v = bidx % cute.size(cta_layout_vmnk, mode=[0])
        is_leader_cta = mma_tile_coord_v == 0

        epilogue_warp_ids = (0, 1, 2, 3)
        mma_warp_id = 4
        tma_warp_id = 5

        if warp_idx == tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        num_mcast_participants = (
            cute.size(cta_layout_vmnk, mode=[1]) + cute.size(cta_layout_vmnk, mode=[2]) - 1
        )
        tma_mcast_mask_a = cpasync.create_tma_multicast_mask(
            cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=2
        )
        tma_mcast_mask_b = cpasync.create_tma_multicast_mask(
            cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=1
        )

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        epilogue_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=threads_in_epilogue
        )
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=32 * len((mma_warp_id, *epilogue_warp_ids))
        )
        two_cta = cluster_shape_mnk[0] == 2
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buffer,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=epilogue_warp_ids[0],
            is_two_cta=two_cta,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar,
        )

        num_tma_copy_bytes = (
            cute.size_in_bytes(IO, cute.select(a_smem_layout, mode=[0, 1, 2]))
            + cute.size_in_bytes(IO, cute.select(b_smem_layout, mode=[0, 1, 2]))
        ) * cute.size(cta_layout_vmnk, mode=[0])

        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=ab_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, size=num_mcast_participants
            ),
            tx_count=num_tma_copy_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
        ).make_participants()

        acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=acc_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                size=cute.size(cta_layout_vmnk, mode=[0]) * len(epilogue_warp_ids),
            ),
            cta_layout_vmnk=cta_layout_vmnk,
        ).make_participants()

        pipeline_init_arrive(cluster_shape_mn=cluster_shape_mnk, is_relaxed=True)

        sA = smem.allocate_tensor(
            element_type=IO, layout=a_smem_layout.outer,
            byte_alignment=128, swizzle=a_smem_layout.inner,
        )
        sB = smem.allocate_tensor(
            element_type=IO, layout=b_smem_layout.outer,
            byte_alignment=128, swizzle=b_smem_layout.inner,
        )
        sC = smem.allocate_tensor(
            element_type=IO, layout=epi_smem_layout_staged.outer,
            byte_alignment=128, swizzle=epi_smem_layout_staged.inner,
        )

        if cutlass.const_expr(HAS_BIAS):
            sBias = smem.allocate_tensor(
                element_type=IO, layout=cute.make_layout(N_TILE), byte_alignment=128,
            )

        gA = cute.local_tile(mA, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None))
        gB = cute.local_tile(mB, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None))
        gC = cute.local_tile(mC, cute.slice_(mma_tiler_mnk, (None, None, 0)), (None, None))

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)
        tCgC = thr_mma.partition_C(gC)

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, acc_stages))

        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            cta_in_cluster_coord_vmnk[2],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            cta_in_cluster_coord_vmnk[1],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        gC_epi = cute.flat_divide(tCgC[((None, None), 0, 0, None, None)], epi_tile)
        tCsC, tCgC_tma = cpasync.tma_partition(
            tma_atom_c, 0, cute.make_layout(1),
            cute.group_modes(sC, 0, 2),
            cute.group_modes(gC_epi, 0, 2),
        )

        pipeline_init_wait(cluster_shape_mn=cluster_shape_mnk)

        # Programmatic dependent launch: the prologue above (descriptor
        # prefetch, barrier / TMEM setup) needs nothing from the producer of A,
        # so let it overlap with the preceding kernel and only block here.
        cute.arch.griddepcontrol_wait()

        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        num_k_tiles = cute.size(gA, mode=[3])

        if warp_idx == tma_warp_id:
            while work_tile.is_valid_tile:
                c = work_tile.tile_idx
                m_idx = c[0] // cute.size(tiled_mma.thr_id.shape)
                tAgA_slice = tAgA[(None, m_idx, None)]
                tBgB_slice = tBgB[(None, c[1], None)]
                for k in cutlass.range(num_k_tiles, unroll=_K_UNROLL):
                    handle = ab_producer.acquire_and_advance()
                    cute.copy(tma_atom_a, tAgA_slice[(None, k)],
                              tAsA[(None, handle.index)],
                              tma_bar_ptr=handle.barrier, mcast_mask=tma_mcast_mask_a)
                    cute.copy(tma_atom_b, tBgB_slice[(None, k)],
                              tBsB[(None, handle.index)],
                              tma_bar_ptr=handle.barrier, mcast_mask=tma_mcast_mask_b)
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            ab_producer.tail()

        elif warp_idx == mma_warp_id:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(ACC)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            while work_tile.is_valid_tile:
                if is_leader_cta:
                    acc_empty = acc_producer.acquire_and_advance()
                    tCtAcc = tCtAcc_base[(None, None, None, acc_empty.index)]
                    for k in cutlass.range(num_k_tiles, unroll=_K_UNROLL):
                        handle = ab_consumer.wait_and_advance()
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, k != 0)
                        crd = (None, None, None, handle.index)
                        cute.gemm(tiled_mma, tCtAcc, tCrA[crd], tCrB[crd], tCtAcc)
                        handle.release()
                    acc_empty.commit()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            acc_producer.tail()

        elif warp_idx < mma_warp_id:
            tmem.allocate(512)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(ACC)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            epilogue_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=epi_stages,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, size=128),
            )
            copy_atom_t2r = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition.x32, tcgen05.Pack.NONE), ACC
            )

            while work_tile.is_valid_tile:
                c = work_tile.tile_idx
                m_idx = c[0] // cute.size(tiled_mma.thr_id.shape)
                acc_full = acc_consumer.wait_and_advance()
                tCtAcc = tCtAcc_base[(None, None, None, acc_full.index)]
                tCtAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0)], epi_tile)
                tCgC_epi = cute.flat_divide(
                    tCgC[((None, None), 0, 0, m_idx, c[1])], epi_tile
                )
                tCgC_tma_cur = tCgC_tma[(None, None, None, m_idx, c[1])]

                tiled_copy_t2r = tcgen05.make_tmem_copy(
                    copy_atom_t2r, tCtAcc_epi[(None, None, 0, 0)]
                )
                thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
                tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc_epi)
                tTR_gC = thr_copy_t2r.partition_D(tCgC_epi)
                tTR_rAcc = cute.make_rmem_tensor(
                    tTR_gC[(None, None, None, 0, 0)].shape, ACC
                )
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                if cutlass.const_expr(HAS_BIAS):
                    n0 = c[1] * N_TILE
                    for i in cutlass.range_constexpr((N_TILE + 127) // 128):
                        j = i * 128 + tidx
                        if j < N_TILE:
                            g = n0 + j
                            sBias[j] = mBias[g] if g < cute.size(mBias, mode=[0]) else IO(0.0)
                    epilogue_sync_barrier.arrive_and_wait()
                    sBiasB = cute.make_tensor(
                        sBias.iterator,
                        cute.make_layout(tCgC[((None, None), 0, 0, 0, 0)].shape, stride=(0, 1)),
                    )
                    tTR_sBias = thr_copy_t2r.partition_D(cute.flat_divide(sBiasB, epi_tile))
                    tTR_sBias = cute.group_modes(tTR_sBias, 3, cute.rank(tTR_sBias))
                    tTR_rBias = cute.make_rmem_tensor(
                        tTR_gC[(None, None, None, 0, 0)].shape, IO
                    )

                copy_atom_r2s = sm100_utils.get_smem_store_op(
                    c_smem_layout_kind, ACC, ACC, tiled_copy_t2r
                )
                tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
                thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                tRS_sC = thr_copy_r2s.partition_D(sC)
                tRS_rAcc = tiled_copy_r2s.retile(tTR_rAcc)
                tRS_rC = cute.make_rmem_tensor(tRS_rAcc.shape, IO)
                tCgC_grouped = cute.group_modes(tCgC_tma_cur, 1, cute.rank(tCgC_tma_cur))
                if cutlass.const_expr(HAS_BIAS):
                    tRS_rBias = tiled_copy_r2s.retile(tTR_rBias)

                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                for st in cutlass.range(subtile_cnt):
                    cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, st)], tTR_rAcc)
                    cbuf = st % epi_stages
                    if cutlass.const_expr(HAS_BIAS):
                        cute.autovec_copy(tTR_sBias[(None, None, None, st)], tTR_rBias)
                        tRS_rC.store(
                            (tRS_rAcc.load() + tRS_rBias.load().to(ACC)).to(IO)
                        )
                    else:
                        tRS_rC.store(tRS_rAcc.load().to(IO))
                    cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, cbuf)])
                    cute.arch.fence_view_async_shared()
                    epilogue_sync_barrier.arrive_and_wait()
                    if warp_idx == epilogue_warp_ids[0]:
                        cute.copy(tma_atom_c, tCsC[(None, cbuf)],
                                  tCgC_grouped[(None, st)])
                        epilogue_pipeline.producer_commit()
                        epilogue_pipeline.producer_acquire()
                    epilogue_sync_barrier.arrive_and_wait()

                with cute.arch.elect_one():
                    acc_full.release()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            epilogue_pipeline.producer_tail()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)


    @cute.jit
    def host_fn(
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        bias: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        mma_tiler_mnk: cutlass.Constexpr,
        cluster_shape_mnk: cutlass.Constexpr,
        ab_stages: cutlass.Constexpr,
        acc_stages: cutlass.Constexpr,
        epi_stages: cutlass.Constexpr,
        HAS_BIAS: cutlass.Constexpr,
        stream,
    ):
        use_2cta = cluster_shape_mnk[0] == 2
        mma_inst_shape = (mma_tiler_mnk[0], mma_tiler_mnk[1], 16)
        op = tcgen05.MmaF16BF16Op(
            IO, ACC, mma_inst_shape,
            tcgen05.CtaGroup.TWO if use_2cta else tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)

        a_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, mma_tiler_mnk, a.element_type, ab_stages
        )
        b_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, mma_tiler_mnk, b.element_type, ab_stages
        )
        c_smem_layout_kind = utils.LayoutEnum.from_tensor(c)

        cta_layout_mnk = cute.make_layout(cluster_shape_mnk)
        cta_layout_vmnk = cute.tiled_divide(cta_layout_mnk, (tiled_mma.thr_id,))

        tma_op = cpasync.CopyBulkTensorTileG2SMulticastOp(
            tcgen05.CtaGroup.TWO if use_2cta else tcgen05.CtaGroup.ONE
        )
        a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
            tma_op, a, cute.slice_(a_smem_layout, (None, None, None, 0)),
            mma_tiler_mnk, tiled_mma, cta_layout_vmnk.shape,
        )
        b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
            tma_op, b, cute.slice_(b_smem_layout, (None, None, None, 0)),
            mma_tiler_mnk, tiled_mma, cta_layout_vmnk.shape,
        )

        cta_tile_shape_mnk = (
            mma_tiler_mnk[0] // cute.size(tiled_mma.thr_id),
            mma_tiler_mnk[1],
            mma_tiler_mnk[2],
        )
        epi_tile = utils.compute_epilogue_tile_shape(
            cta_tile_shape_mnk, use_2cta, c_smem_layout_kind, IO
        )
        epi_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            IO, c_smem_layout_kind, epi_tile, epi_stages
        )
        c_tma_atom, c_tma_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c,
            cute.slice_(epi_smem_layout_staged, (None, None, 0)), epi_tile,
        )

        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ab_stages * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stages * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buffer: cutlass.Int32

        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mn = gc[(0, (None, None))].shape
        # Rasterize along N: a wave of clusters then spans only ~ntiles_n
        # row-blocks of A, so A stays resident in L2 across its N reuses.
        tile_sched_params = utils.PersistentTileSchedulerParams(
            (*num_ctas_mn, 1), cluster_shape_mnk, raster_along_m=False,
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        kernel(
            tiled_mma, a_tma_atom, a_tma_tensor, b_tma_atom, b_tma_tensor,
            c_tma_atom, c_tma_tensor, bias, mma_tiler_mnk[1],
            a_smem_layout, b_smem_layout, c_smem_layout_kind,
            epi_smem_layout_staged, epi_tile, cta_layout_vmnk, tile_sched_params,
            mma_tiler_mnk, ab_stages, acc_stages, epi_stages, cluster_shape_mnk,
            HAS_BIAS, SharedStorage,
        ).launch(grid=grid, block=[192, 1, 1], cluster=cluster_shape_mnk,
                 stream=stream, use_pdl=True)


    # ---------------------------------------------------------------------------
    # Host-side driver: compile once per (N, K, bias) and dispatch on M.
    # ---------------------------------------------------------------------------
    class _Gemm:
        """Compiled bf16 linear: out = x @ w.T (+ bias), x [M,K] / w [N,K] K-major."""

        def __init__(self, n: int, k: int, has_bias: bool):
            import cutlass.torch as cutlass_torch

            tiler_n = next((t for t in _MMA_TILER_N if n % t == 0), _MMA_TILER_N[-1])
            self.mma_tiler = (256, tiler_n, 64)
            self.tiles_n = -(-n // tiler_n)
            self._cur_stream = cutlass_torch.current_stream
            self._const = {}
            self.zero_bias = torch.zeros(n, dtype=torch.bfloat16, device="cuda")
            mac = utils.HardwareInfo().get_max_active_clusters(_CLUSTER[0] * _CLUSTER[1])
            self.clusters = mac

            a = torch.empty(256, k, dtype=torch.bfloat16, device="cuda")
            b = torch.empty(n, k, dtype=torch.bfloat16, device="cuda")
            c = torch.empty(256, n, dtype=torch.bfloat16, device="cuda")
            bias = torch.empty(n, dtype=torch.bfloat16, device="cuda")
            self._compiled = cute.compile(
                host_fn,
                self._t(a), self._t(b), self._t(c), self._b(bias), mac,
                self.mma_tiler, _CLUSTER, *_STAGES, has_bias, self._cur_stream(),
            )

        def wave_fill(self, m):
            """Fraction of the last persistent wave that is occupied, in [0, 1]."""
            tiles = -(-m // self.mma_tiler[0]) * self.tiles_n
            waves = -(-tiles // self.clusters)
            return tiles / (waves * self.clusters)

        @staticmethod
        def _t(t):
            return from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(
                leading_dim=1)

        @staticmethod
        def _b(t):
            return from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(
                leading_dim=0)

        def _const_arg(self, t, maker):
            # w / bias live in module parameters: same storage on every call, so wrap
            # them once instead of per forward (the dlpack capsule keeps them alive).
            arg = self._const.get(t.data_ptr())
            if arg is None:
                arg = maker(t)
                self._const[t.data_ptr()] = arg
            return arg

        def __call__(self, x, w, bias, out):
            if bias is None:
                bias = self.zero_bias
            self._compiled(self._t(x), self._const_arg(w, self._t), self._t(out),
                           self._const_arg(bias, self._b), self._cur_stream())
            return out


    # (n, k, has_bias) -> compiled kernel, or None once that shape has proven
    # unsupported. Failures are isolated per shape so one unusable embed_dim
    # cannot disable the shapes that do work.
    _CACHE = {}


    def _get_gemm(key):
        if key not in _CACHE:
            try:
                _CACHE[key] = _Gemm(*key)
            except Exception:
                _CACHE[key] = None
        return _CACHE[key]


    @functools.cache
    def _sm100() -> bool:
        try:
            return torch.cuda.get_device_capability() == (10, 0)
        except Exception:
            return False


    def fast_linear(x, w, bias):
        """out = F.linear(x, w, bias) via the SM100 kernel, or None if unsupported."""
        m, k = x.shape
        n = w.shape[0]
        if (m < _MIN_ROWS or n % 8 or k % 8
                or x.dtype is not torch.bfloat16 or w.dtype is not torch.bfloat16
                or not x.is_cuda or not x.is_contiguous() or not w.is_contiguous()
                or x.data_ptr() % 16 or w.data_ptr() % 16 or not _sm100()
                or (bias is not None and (bias.dtype is not torch.bfloat16
                                          or not bias.is_contiguous()))):
            return None
        key = (n, k, bias is not None)
        gemm = _get_gemm(key)
        if gemm is None or gemm.wave_fill(m) < _MIN_WAVE_FILL:
            return None
        try:
            out = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
            return gemm(x, w, bias, out)
        except Exception:
            _CACHE[key] = None
            return None

else:  # pragma: no cover - no CuTe DSL available

    def fast_linear(x, w, bias):
        return None


class VisionPatchEmbed(nn.Module):
    def __init__(self, patch_size: int, temporal_patch_size: int,
                 in_channels: int, embed_dim: int, bias: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_size = in_channels * temporal_patch_size * patch_size * patch_size
        kernel = (temporal_patch_size, patch_size, patch_size)
        self.proj = Conv3d(in_channels, embed_dim, kernel, bias=bias)
        self.linear = Matmul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], self.input_size)
        weight = self.proj.weight.view(self.embed_dim, self.input_size)
        bias = self.proj.bias
        out = fast_linear(x, weight, bias)
        if out is not None:
            return out
        return self.linear(x, weight, bias)
