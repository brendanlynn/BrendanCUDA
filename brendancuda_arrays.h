#pragma once

#include <cuda_runtime.h>

namespace BrendanCUDA {
    template <typename _T>
    struct Span;

    template <typename _T>
    struct SpanConst;

    template <typename _T>
    class ArrayV {
        _T* ptr;
        size_t size;
    public:
        __forceinline ArrayV() = default;
        __host__ __device__ __forceinline ArrayV(_T* Ptr, size_t Size)
            : ptr(Ptr), size(Size) { }
        __host__ __device__ __forceinline ArrayV(size_t Size)
            : ptr(new _T[Size]), size(Size) { }
        __host__ __device__ __forceinline ArrayV(const ArrayV<_T>& Array)
            : ArrayV(Array.size) {
            std::copy(Array.ptr, Array.ptr + Array.size, ptr);
        }
        __host__ __device__ __forceinline ArrayV(ArrayV<_T>&& Array)
            : ArrayV(Array.ptr, Array.size) {
            Array.ptr = 0;
        }

        __host__ __device__ __forceinline ~ArrayV() {
            delete[] ptr;
        }

        __host__ __device__ __forceinline ArrayV<_T>& operator=(const ArrayV<_T>& Array) {
            this->~ArrayV();
            new (this) ArrayV<_T>(Array);
            return *this;
        }
        __host__ __device__ __forceinline ArrayV<_T>& operator=(ArrayV<_T>&& Array) {
            this->~ArrayV();
            new (this) ArrayV<_T>(Array);
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

        __host__ __device__ __forceinline operator Span<_T>() {
            return Span<_T>(*this);
        }
        __host__ __device__ __forceinline operator SpanConst<_T>() const {
            return SpanConst<_T>(*this);
        }
        __host__ __device__ __forceinline Span<_T> Split(size_t Start, size_t NewSize) {
            return Span<_T>(*this);
        }
        __host__ __device__ __forceinline SpanConst<_T> Split(size_t Start, size_t NewSize) const {
            return SpanConst<_T>(*this);
        }
    };

    template <typename _T>
    struct Span {
        _T* ptr;
        size_t size;
        template <size_t _Size>
        __host__ __device__ __forceinline Span(_T* Ptr, size_t Size)
            : ptr(Ptr), size(Size) { }

        __host__ __device__ __forceinline Span(std::array<_T, _Size>& Array)
            : ptr(Array.data()), size(Array.size()) { }
        __host__ __device__ __forceinline Span(std::vector<_T>& Vector)
            : ptr(Vector.data()), size(Vector.size()) { }
        __host__ __device__ __forceinline Span(ArrayV<_T>& Array)
            : ptr(Array.Data()), size(Array.Size()) { }

        __host__ __device__ __forceinline _T& operator[](size_t Idx) {
            return ptr[Idx];
        }
        __host__ __device__ __forceinline const _T& operator[](size_t Idx) const {
            return ptr[Idx];
        }

        __host__ __device__ __forceinline Span<_T> Split(size_t Start, size_t NewSize) const {
            return Span<_T>(ptr + Start, NewSize);
        }

        __host__ __device__ __forceinline operator SpanConst<_T>() const {
            return SpanConst<_T>(*this);
        }
    };

    template <typename _T>
    struct SpanConst {
        const _T* ptr;
        size_t size;
        __host__ __device__ __forceinline SpanConst(const _T* Ptr, size_t Size)
            : ptr(Ptr), size(Size) { }

        template <size_t _Size>
        __host__ __device__ __forceinline SpanConst(const std::array<_T, _Size>& Array)
            : ptr(Array.data()), size(Array.size()) { }
        template <typename _TAlloc>
        __host__ __device__ __forceinline SpanConst(const std::vector<_T, _TAlloc>& Vector)
            : ptr(Vector.data()), size(Vector.size()) { }
        __host__ __device__ __forceinline SpanConst(const ArrayV<_T>& Array)
            : ptr(Array.Data()), size(Array.Size()) { }
        __host__ __device__ __forceinline SpanConst(const Span<_T>& Span)
            : ptr(Span.ptr), size(Span.size) { }

        __host__ __device__ __forceinline const _T& operator[](size_t Idx) const {
            return ptr[Idx];
        }

        __host__ __device__ __forceinline SpanConst<_T> Split(size_t Start, size_t NewSize) const {
            return SpanConst<_T>(ptr + Start, NewSize);
        }
    };
}