#pragma once

#include <memory>
#include <type_traits>

namespace BrendanCUDA {
    template <typename _T>
    class CopyPtr {
        std::unique_ptr<_T> ptr;
    public:
        using element_t = _T;
        static constexpr bool copyable = std::copy_constructible<_T>;

        __forceinline explicit CopyPtr(std::unique_ptr<_T> UniPtr)
            : ptr(std::move(UniPtr)) { }
        __forceinline CopyPtr(const CopyPtr<_T>& Other) requires copyable
            : ptr(Other.ptr ? std::make_unique<_T>(*Other.ptr) : std::unique_ptr<_T>(nullptr)) { }
        __forceinline CopyPtr(CopyPtr<_T>&& Other)
            : ptr(std::move(Other.ptr)) { }
        __forceinline explicit CopyPtr(element_t* Val)
            : ptr(Val) { }
        __forceinline explicit CopyPtr(std::nullptr_t)
            : ptr(nullptr) { }

        __forceinline CopyPtr<_T>& operator=(CopyPtr<_T> Other) {
            std::swap(ptr, Other.ptr);
            return *this;
        }

        __forceinline _T* Get() const {
            return (_T*)ptr.get();
        }

        __forceinline operator _T*() const {
            return Get();
        }
    };

    template <typename _T>
    CopyPtr<_T> MakeCopyPtr(_T Val) {
        return CopyPtr<_T>(new _T(Val));
    }
}