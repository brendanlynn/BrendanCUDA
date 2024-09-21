#pragma once

#include <concepts>
#include <curand_kernel.h>

namespace brendancuda {
    template <typename _T>
    concept KernelCurandState =
        std::same_as<_T, curandStateMRG32k3a_t> ||
        std::same_as<_T, curandStateMtgp32_t> ||
        std::same_as<_T, curandStatePhilox4_32_10_t> ||
        std::same_as<_T, curandStateScrambledSobol32_t> ||
        std::same_as<_T, curandStateScrambledSobol64_t> ||
        std::same_as<_T, curandStateSobol32_t> ||
        std::same_as<_T, curandStateSobol64_t> ||
        std::same_as<_T, curandStateXORWOW_t>;
}