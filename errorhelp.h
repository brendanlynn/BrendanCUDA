#pragma once

namespace bcuda {
    template <typename _T>
        requires (std::is_enum_v<_T> || std::integral<_T>)
    void ThrowIfBad(_T e) {
        if (e) throw e;
    }
}