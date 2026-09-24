// GEMM1 fused (bias + tanh-GELU): 2-SM MMA tile 256x192x64, cluster 2x1x1, CLC.
// Fastest measured on the GEMM1 shape at M=512 and at M=4096.
#include "ako_ffn.cuh"

namespace ako {
using MmaT_g1a = cute::Shape<cute::_256, cute::_192, cute::_64>;
using ClT_g1a = cute::Shape<cute::_2, cute::_1, cute::_1>;
}

AKO_DEFINE(g1a, ako::MmaT_g1a, ako::ClT_g1a, ako::FusionBiasGelu, ako::TileSchedCLC)
