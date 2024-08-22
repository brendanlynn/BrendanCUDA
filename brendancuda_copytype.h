#pragma once

#include <cstdint>

namespace BrendanCUDA {
    enum CopyType : uint32_t {
        copyTypeMemcpy = 0,
        copyTypeMoveAssignment = 1,
        copyTypeCopyAssignment = 2
    };
}