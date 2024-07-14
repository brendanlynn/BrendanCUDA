#pragma once

namespace BrendanCUDA {
#ifdef __CUDA_ARCH__
    constexpr bool isCuda = true;
#else
    constexpr bool isCuda = false;
#endif
}