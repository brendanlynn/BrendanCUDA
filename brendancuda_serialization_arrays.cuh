#pragma once

#include <cstdint>
#include <stdexcept>
#include <cuda_runtime.h>
#include <iostream>

namespace BrendanCUDA {
    namespace Serialization {
        void SerializeArrayNoLengthWrite(std::basic_ostream<char>& Stream, uint8_t* Array, size_t Length);
    }
}