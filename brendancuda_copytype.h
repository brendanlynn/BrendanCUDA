#pragma once

#include <cstdint>

namespace BrendanCUDA {
    enum CopyType : uint32_t {
        copyTypeMemcpy,
        copyTypeCopyAssignment
    };
}