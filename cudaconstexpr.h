#pragma once

namespace bcuda {
#ifdef __CUDA_ARCH__
    constexpr bool isCuda = true;
#else
    constexpr bool isCuda = false;
#endif
}