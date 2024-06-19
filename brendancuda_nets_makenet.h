#pragma once

#include "brendancuda_nets_net.h"
#include "brendancuda_points.cuh"
#include "brendancuda_random_anyrng.cuh"

namespace BrendanCUDA {
    namespace Nets {
        Net MakeNet_3D(size_t NodeCount, float ConnectionRange, BrendanCUDA::Random::AnyRNG<uint64_t> RNG, thrust::device_vector<float_3>** NodePoints);
    }
}