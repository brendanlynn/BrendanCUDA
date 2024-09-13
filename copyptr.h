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

        explicit CopyPtr(std::unique_ptr<_T> UniPtr)
            : ptr(std::move(UniPtr)) { }
        CopyPtr(const CopyPtr<_T>& Other) requires copyable
            : ptr(Other.ptr ? std::make_unique<_T>(*Other.ptr) : std::unique_ptr<_T>(0)) { }
        CopyPtr(CopyPtr<_T>&& Other)
            : ptr(std::move(Other.ptr)) { }
        CopyPtr(_T Val)
            : ptr(std::make_unique<_T>(Val)) { }

        CopyPtr<_T>& operator=(CopyPtr<_T> Other) {
            std::swap(ptr, Other.ptr);
        }

        _T* Get() const {
            return (_T*)ptr.get();
        }

        _T* operator->() const {
            return ptr.get();
        }
        _T& operator*() const {
            return *ptr;
        }
        _T& operator[](size_t Idx) {
            return ptr[Idx];
        }

        operator _T*() const {
            return ptr;
        }
    };
}