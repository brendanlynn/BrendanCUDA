#pragma once

#include <cuda_runtime.h>

namespace BrendanCUDA {
    template <typename _T>
    struct Span;

    template <typename _T>
    struct Span_ReadOnly;

    template <typename _T, size_t _Size>
    struct ArrayF final {
        _T data[_Size];
        __host__ __device__ inline ArrayF() = default;
        __host__ __device__ inline _T& operator[](size_t Index);
        __host__ __device__ inline const _T& operator[](size_t Index) const;
        __host__ __device__ inline operator Span<_T>() const;
        __host__ __device__ inline operator Span_ReadOnly<_T>() const;
        __host__ __device__ inline Span<_T> Split(size_t Start, size_t NewSize) const;
    };

    template <typename _T>
    struct ArrayV final {
        _T* ptr;
        size_t size;
        __host__ __device__ inline ArrayV(_T* Pointer, size_t Size);
        __host__ __device__ inline ArrayV(size_t Size);
        __host__ __device__ inline _T& operator[](size_t Index);
        __host__ __device__ inline const _T& operator[](size_t Index) const;
        __host__ __device__ inline void Dispose();
        __host__ __device__ inline operator Span<_T>() const;
        __host__ __device__ inline operator Span_ReadOnly<_T>() const;
        __host__ __device__ inline Span<_T> Split(size_t Start, size_t NewSize) const;
    };

    template <typename _T>
    struct Span final {
        _T* ptr;
        size_t size;
        template <size_t _Size>
        __host__ __device__ inline Span(ArrayF<_T, _Size>& Array);
        __host__ __device__ inline Span(ArrayV<_T> Array);
        __host__ __device__ inline Span(_T* Pointer, size_t Size);
        __host__ __device__ inline _T& operator[](size_t Index);
        __host__ __device__ inline const _T& operator[](size_t Index) const;
        __host__ __device__ inline Span<_T> Split(size_t Start, size_t NewSize) const;
        __host__ __device__ inline operator Span_ReadOnly<_T>() const;
    };

    template <typename _T>
    struct Span_ReadOnly final {
        const _T* ptr;
        size_t size;
        template <size_t _Size>
        __host__ __device__ inline Span_ReadOnly(ArrayF<_T, _Size>& Array);
        __host__ __device__ inline Span_ReadOnly(ArrayV<_T> Array);
        __host__ __device__ inline Span_ReadOnly(Span<_T> Span);
        __host__ __device__ inline Span_ReadOnly(_T* Pointer, size_t Size);
        __host__ __device__ inline const _T& operator[](size_t Index) const;
        __host__ __device__ inline Span_ReadOnly<_T> Split(size_t Start, size_t NewSize) const;
    };
}

template <typename _T, size_t _Size>
__host__ __device__ inline _T& BrendanCUDA::ArrayF<_T, _Size>::operator[](size_t Index) {
    return data[Index];
}

template <typename _T, size_t _Size>
__host__ __device__ inline const _T& BrendanCUDA::ArrayF<_T, _Size>::operator[](size_t Index) const {
    return data[Index];
}

template <typename _T, size_t _Size>
__host__ __device__ inline BrendanCUDA::ArrayF<_T, _Size>::operator BrendanCUDA::Span<_T>() const {
    return Span<_T>((_T*)this, _Size);
}

template <typename _T, size_t _Size>
__host__ __device__ inline BrendanCUDA::ArrayF<_T, _Size>::operator BrendanCUDA::Span_ReadOnly<_T>() const {
    return Span_ReadOnly<_T>((_T*)this, _Size);
}

template <typename _T, size_t _Size>
__host__ __device__ inline BrendanCUDA::Span<_T> BrendanCUDA::ArrayF<_T, _Size>::Split(size_t Start, size_t NewSize) const {
    return Span<_T>((_T*)this + Start, NewSize);
}

template <typename _T>
__host__ __device__ inline BrendanCUDA::ArrayV<_T>::ArrayV(_T* Pointer, size_t Size) {
    ptr = Pointer;
    size = Size;
}

template <typename _T>
__host__ __device__ inline BrendanCUDA::ArrayV<_T>::ArrayV(size_t Size) {
    ptr = new _T[Size];
    size = Size;
}

template <typename _T>
__host__ __device__ inline _T& BrendanCUDA::ArrayV<_T>::operator[](size_t Index) {
    return ptr[Index];
}

template <typename _T>
__host__ __device__ inline const _T& BrendanCUDA::ArrayV<_T>::operator[](size_t Index) const {
    return ptr[Index];
}

template <typename _T>
__host__ __device__ inline void BrendanCUDA::ArrayV<_T>::Dispose() {
    delete[] ptr;
}

template <typename _T>
__host__ __device__ inline BrendanCUDA::ArrayV<_T>::operator BrendanCUDA::Span<_T>() const {
    return Span<_T>(ptr, size);
}

template <typename _T>
__host__ __device__ inline BrendanCUDA::ArrayV<_T>::operator BrendanCUDA::Span_ReadOnly<_T>() const {
    return Span_ReadOnly<_T>(ptr, size);
}

template <typename _T>
__host__ __device__ inline BrendanCUDA::Span<_T> BrendanCUDA::ArrayV<_T>::Split(size_t Start, size_t NewSize) const {
    return Span<_T>(ptr + Start, NewSize);
}

template <typename _T>
template <size_t _Size>
__host__ __device__ inline BrendanCUDA::Span<_T>::Span(ArrayF<_T, _Size>& Array) {
    ptr = (_T*)&Array;
    size = _Size;
}

template <typename _T>
__host__ __device__ inline BrendanCUDA::Span<_T>::Span(ArrayV<_T> Array) {
    ptr = Array.ptr;
    size = Array.size;
}

template <typename _T>
__host__ __device__ inline BrendanCUDA::Span<_T>::Span(_T* Pointer, size_t Size) {
    ptr = Pointer;
    size = Size;
}

template <typename _T>
__host__ __device__ inline _T& BrendanCUDA::Span<_T>::operator[](size_t Index) {
    return ptr[Index];
}

template <typename _T>
__host__ __device__ inline const _T& BrendanCUDA::Span<_T>::operator[](size_t Index) const {
    return ptr[Index];
}

template <typename _T>
__host__ __device__ inline BrendanCUDA::Span<_T> BrendanCUDA::Span<_T>::Split(size_t Start, size_t NewSize) const {
    return Span<_T>(ptr + Start, NewSize);
}

template <typename _T>
__host__ __device__ inline BrendanCUDA::Span<_T>::operator BrendanCUDA::Span_ReadOnly<_T>() const {
    return Span_ReadOnly<_T>(ptr, size);
}

template <typename _T>
template <size_t _Size>
__host__ __device__ inline BrendanCUDA::Span_ReadOnly<_T>::Span_ReadOnly(ArrayF<_T, _Size>& Array) {
    ptr = (_T*)&Array;
    size = _Size;
}

template <typename _T>
__host__ __device__ inline BrendanCUDA::Span_ReadOnly<_T>::Span_ReadOnly(ArrayV<_T> Array) {
    ptr = Array.ptr;
    size = Array.size;
}

template <typename _T>
__host__ __device__ inline BrendanCUDA::Span_ReadOnly<_T>::Span_ReadOnly(Span<_T> Span) {
    ptr = Span.ptr;
    size = Span.size;
}

template <typename _T>
__host__ __device__ inline BrendanCUDA::Span_ReadOnly<_T>::Span_ReadOnly(_T* Pointer, size_t Size) {
    ptr = Pointer;
    size = Size;
}

template <typename _T>
__host__ __device__ inline const _T& BrendanCUDA::Span_ReadOnly<_T>::operator[](size_t Index) const {
    return ptr[Index];
}

template <typename _T>
__host__ __device__ inline BrendanCUDA::Span_ReadOnly<_T> BrendanCUDA::Span_ReadOnly<_T>::Split(size_t Start, size_t NewSize) const {
    return Span_ReadOnly<_T>(ptr + Start, NewSize);
}