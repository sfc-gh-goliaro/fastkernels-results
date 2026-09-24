// GEMM1 fused (bias + tanh-GELU): 2-SM MMA tile 256x256x64, cluster 2x2x1, CLC.
// Fastest measured on the GEMM1 shape around M=1024.
#include "ako_ffn.cuh"

namespace ako {
using MmaT_g1b = cute::Shape<cute::_256, cute::_256, cute::_64>;
using ClT_g1b = cute::Shape<cute::_2, cute::_2, cute::_1>;
}

AKO_DEFINE(g1b, ako::MmaT_g1b, ako::ClT_g1b, ako::FusionBiasGelu, ako::TileSchedCLC)
