#pragma once

#include <algorithm>
#include <cuda_runtime.h>
#include <vector>

namespace BrendanCUDA {
    template <typename _T>
    struct Span;

    template <typename _T>
        requires (!std::is_const_v<_T>)
    class ArrayV {
        _T* ptr;
        size_t size;
    public:
        using element_t = _T;

        __forceinline ArrayV() = default;
        __host__ __device__ __forceinline ArrayV(_T* Ptr, size_t Size)
            : ptr(Ptr), size(Size) { }
        __host__ __device__ __forceinline ArrayV(size_t Size)
            : ptr(new _T[Size]), size(Size) { }
        __host__ __device__ __forceinline ArrayV(const ArrayV<_T>& Array)
            : ArrayV(Array.size) {
            std::copy_n(Array.ptr, Array.size, ptr);
        }
        __host__ __device__ __forceinline ArrayV(ArrayV<_T>&& Array)
            : ArrayV(Array.ptr, Array.size) {
            Array.ptr = 0;
        }
        __host__ __device__ __forceinline ArrayV(std::initializer_list<_T> InitList)
            : size(InitList.size()), ptr(new _T[InitList.size()]) {
            std::copy(InitList.begin(), InitList.end(), ptr);
        }

        __host__ __device__ __forceinline ~ArrayV() {
            delete[] ptr;
        }

        __host__ __device__ __forceinline ArrayV<_T>& operator=(ArrayV<_T> Array) {
            std::swap(ptr, Array.ptr);
            std::swap(size, Array.size);
            return *this;
        }

        __host__ __device__ _T* Data() {
            return ptr;
        }
        __host__ __device__ const _T* Data() const {
            return ptr;
        }
        __host__ __device__ size_t Size() const {
            return size;
        }

        __host__ __device__ __forceinline _T& operator[](size_t Idx) {
            return ptr[Idx];
        }
        __host__ __device__ __forceinline const _T& operator[](size_t Idx) const {
            return ptr[Idx];
        }

        __host__ __device__ __forceinline Span<_T> Split(size_t Start, size_t NewSize) {
            return Span<_T>(*this).Split(Start, NewSize);
        }

        __host__ __device__ __forceinline Span<const _T> Split(size_t Start, size_t NewSize) const {
            return Span<const _T>(*this).Split(Start, NewSize);
        }
    };

    template <typename _T>
    struct Span {
        using element_t = _T;

        _T* ptr;
        size_t size;

        __host__ __device__ __forceinline Span(_T* Ptr, size_t Size)
            : ptr(Ptr), size(Size) { }

        template <size_t _Size>
        __host__ __device__ __forceinline Span(std::array<std::remove_const_t<_T>, _Size>& Array)
            : ptr(Array.data()), size(Array.size()) { }
        template <size_t _Size>
        __host__ __device__ __forceinline Span(const std::array<std::remove_const_t<_T>, _Size>& Array) requires (std::is_const_v<_T>)
            : ptr(Array.data()), size(Array.size()) { }
        __host__ __device__ __forceinline Span(std::vector<std::remove_const_t<_T>>& Vector)
            : ptr(Vector.data()), size(Vector.size()) { }
        __host__ __device__ __forceinline Span(const std::vector<std::remove_const_t<_T>>& Vector) requires (std::is_const_v<_T>)
            : ptr(Vector.data()), size(Vector.size()) { }
        __host__ __device__ __forceinline Span(ArrayV<std::remove_const_t<_T>>& Array)
            : ptr(Array.Data()), size(Array.Size()) { }
        __host__ __device__ __forceinline Span(const ArrayV<std::remove_const_t<_T>>& Array)
            : ptr(Array.Data()), size(Array.Size()) { }
        __host__ __device__ __forceinline Span(const Span<_T>& Span)
            : ptr(Span.ptr), size(Span.size) { }
        __host__ __device__ __forceinline Span(const Span<std::remove_const_t<_T>>& Span) requires std::is_const_v<_T>
            : ptr(Span.ptr), size(Span.size) { }

        __host__ __device__ __forceinline _T& operator[](size_t Idx) const {
            return ptr[Idx];
        }

        __host__ __device__ __forceinline Span<_T> Split(size_t Start, size_t NewSize) const {
            return Span<_T>(ptr + Start, NewSize);
        }
    };
}