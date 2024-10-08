#pragma once

#include "nets_net.h"
#include "points.h"
#include "rand_anyrng.h"

namespace bcuda {
    namespace nets {
        //Creates a bcuda::nets::Net by creating NodeCount points in a rectangular prism such that all coordinates are in the range of [0, 1], and then connecting up the nodes cooresponding to any points within a distance of ConnectionRange from each other. If NodePoints is non-null, the value it points to will be assigned the vector of points -- otherwise, the vector of points will be disposed of.
        Net MakeNet_3D(size_t NodeCount, float ConnectionRange, bcuda::rand::AnyRNG<uint64_t> RNG, thrust::device_vector<float_3>** NodePoints);
    }
}