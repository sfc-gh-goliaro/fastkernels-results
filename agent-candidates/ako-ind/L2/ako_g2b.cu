// GEMM2 (bias only): 2-SM MMA tile 256x256x64, cluster 2x1x1, Stream-K scheduler.
// GEMM2 has K=12288 but only 3072 output columns, so at M=4096 a data-parallel
// decomposition leaves a large partial wave; Stream-K recovers ~20us there.
#include "ako_ffn.cuh"

namespace ako {
using MmaT_g2b = cute::Shape<cute::_256, cute::_256, cute::_64>;
using ClT_g2b = cute::Shape<cute::_2, cute::_1, cute::_1>;
}

AKO_DEFINE(g2b, ako::MmaT_g2b, ako::ClT_g2b, ako::FusionBias, ako::TileSchedStreamK)
