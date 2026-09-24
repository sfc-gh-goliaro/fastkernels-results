// GEMM2 (bias only): 2-SM MMA tile 256x192x64, cluster 2x1x1, CLC.
// Fastest measured on the GEMM2 shape (K=12288, N=3072) around M=1024.
#include "ako_ffn.cuh"

namespace ako {
using MmaT_g2a = cute::Shape<cute::_256, cute::_192, cute::_64>;
using ClT_g2a = cute::Shape<cute::_2, cute::_1, cute::_1>;
}

AKO_DEFINE(g2a, ako::MmaT_g2a, ako::ClT_g2a, ako::FusionBias, ako::TileSchedCLC)
