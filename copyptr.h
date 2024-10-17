#pragma once

#include <memory>
#include <type_traits>

namespace bcuda {
    template <typename _T>
    class CopyPtr {
        std::unique_ptr<_T> ptr;
    public:
        using element_t = _T;
        static constexpr bool copyable = std::copy_constructible<_T>;

        inline explicit CopyPtr(std::unique_ptr<_T> UniPtr)
            : ptr(std::move(UniPtr)) { }
        inline CopyPtr(const CopyPtr<_T>& Other) requires copyable
            : ptr(Other.ptr ? std::make_unique<_T>(*Other.ptr) : std::unique_ptr<_T>(nullptr)) { }
        inline CopyPtr(CopyPtr<_T>&& Other)
            : ptr(std::move(Other.ptr)) { }
        inline explicit CopyPtr(element_t* Val)
            : ptr(Val) { }
        inline explicit CopyPtr(std::nullptr_t = nullptr)
            : ptr(nullptr) { }

        inline CopyPtr<_T>& operator=(CopyPtr<_T> Other) {
            std::swap(ptr, Other.ptr);
            return *this;
        }

        inline _T* Get() const {
            return (_T*)ptr.get();
        }

        inline operator _T*() const {
            return Get();
        }
        inline _T* operator->() const {
            return Get();
        }

        inline _T* Release() {
            return (_T*)ptr.release();
        }

        inline void Reset(_T* newPtr = 0) {
            ptr.reset(newPtr);
        }
    };

    template <typename _T>
    static inline CopyPtr<_T> MakeCopyPtr() requires std::is_default_constructible_v<_T> {
        return CopyPtr<_T>(new _T());
    }
    template <typename _T>
    static inline CopyPtr<_T> MakeCopyPtr(_T Val) {
        return CopyPtr<_T>(new _T(Val));
    }
}