#pragma once

#include <concepts>
#include <type_traits>

namespace bcuda {
    template <typename _T>
        requires (std::is_enum_v<_T> || std::integral<_T>)
    static inline void ThrowIfBad(_T e) {
        if (e) throw e;
    }
}