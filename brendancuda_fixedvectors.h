#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cstdarg>
#include <concepts>
#include <cmath>
#include "brendancuda_mathfuncs.h"
#include "BSerializer/Serializer.h"

namespace BrendanCUDA {
    template <typename _T, size_t _Size>
        requires std::is_arithmetic_v<_T>
    struct FixedVector final {
        static_assert(_Size, "_Size must be positive.");

        _T v[_Size];

        __host__ __device__ __forceinline constexpr FixedVector();
        template <std::convertible_to<uint32_t>... _Ts>
            requires (sizeof...(_Ts) == _Size)
        __host__ __device__ __forceinline constexpr FixedVector(_Ts... V);
        __host__ __device__ __forceinline constexpr FixedVector(const _T V[_Size]);

        __host__ __device__ __forceinline constexpr _T& operator[](size_t Index);
        __host__ __device__ __forceinline constexpr const _T& operator[](size_t Index) const;

        __host__ __device__ __forceinline constexpr FixedVector<_T, _Size> operator+(FixedVector<_T, _Size> Other) const;
        __host__ __device__ __forceinline void operator+=(FixedVector<_T, _Size> Other);
        __host__ __device__ __forceinline constexpr FixedVector<_T, _Size> operator-(FixedVector<_T, _Size> Other) const;
        __host__ __device__ __forceinline void operator-=(FixedVector<_T, _Size> Other);
        __host__ __device__ __forceinline constexpr FixedVector<_T, _Size> operator*(_T Other) const;
        __host__ __device__ __forceinline void operator*=(_T Other);
        __host__ __device__ __forceinline constexpr FixedVector<_T, _Size> operator/(_T Other) const;
        __host__ __device__ __forceinline void operator/=(_T Other);

        __host__ __device__ static __forceinline constexpr _T Dot(FixedVector<_T, _Size> Left, FixedVector<_T, _Size> Right);
        
        __host__ __device__ __forceinline constexpr _T MagnatudeSquared() const;
        __host__ __device__ __forceinline _T MagnatudeI() const requires std::integral<_T>;
        __host__ __device__ __forceinline float MagnatudeF() const requires std::integral<_T>;
        __host__ __device__ __forceinline std::conditional_t<std::floating_point<_T>, _T, double> Magnatude() const;

        __forceinline size_t SerializedSize() const;
        __forceinline void Serialize(void*& Data) const;
        static __forceinline FixedVector<_T, _Size> Deserialize(const void*& Data);
        static __forceinline void Deserialize(const void*& Data, void* Value);
    };
    template <typename _T>
        requires std::is_arithmetic_v<_T>
    struct FixedVector<_T, 1> final {
        union {
            _T x;
            _T v[1];
        };

        __host__ __device__ __forceinline constexpr FixedVector();
        __host__ __device__ __forceinline constexpr FixedVector(_T X);
        __host__ __device__ __forceinline constexpr FixedVector(const _T V[1]);

        __host__ __device__ __forceinline constexpr _T& operator[](size_t Index);
        __host__ __device__ __forceinline constexpr const _T& operator[](size_t Index) const;

        __host__ __device__ __forceinline constexpr FixedVector<_T, 1> operator+(FixedVector<_T, 1> Other) const;
        __host__ __device__ __forceinline void operator+=(FixedVector<_T, 1> Other);
        __host__ __device__ __forceinline constexpr FixedVector<_T, 1> operator-(FixedVector<_T, 1> Other) const;
        __host__ __device__ __forceinline void operator-=(FixedVector<_T, 1> Other);
        __host__ __device__ __forceinline constexpr FixedVector<_T, 1> operator*(_T Other) const;
        __host__ __device__ __forceinline void operator*=(_T Other);
        __host__ __device__ __forceinline constexpr FixedVector<_T, 1> operator/(_T Other) const;
        __host__ __device__ __forceinline void operator/=(_T Other);

        __host__ __device__ static __forceinline constexpr _T Dot(FixedVector<_T, 1> Left, FixedVector<_T, 1> Right);
        
        __host__ __device__ __forceinline constexpr _T MagnatudeSquared() const;
        __host__ __device__ __forceinline _T MagnatudeI() const requires std::integral<_T>;
        __host__ __device__ __forceinline float MagnatudeF() const requires std::integral<_T>;
        __host__ __device__ __forceinline std::conditional_t<std::floating_point<_T>, _T, double> Magnatude() const;

        __forceinline size_t SerializedSize() const;
        __forceinline void Serialize(void*& Data) const;
        static __forceinline FixedVector<_T, 1> Deserialize(const void*& Data);
        static __forceinline void Deserialize(const void*& Data, void* Value);
    };
    template <typename _T>
        requires std::is_arithmetic_v<_T>
    struct FixedVector<_T, 2> final {
        union {
            struct { _T x, y; };
            _T v[2];
        };

        __host__ __device__ __forceinline constexpr FixedVector();
        __host__ __device__ __forceinline constexpr FixedVector(_T X, _T Y);
        __host__ __device__ __forceinline constexpr FixedVector(const _T V[2]);

        __host__ __device__ __forceinline constexpr _T& operator[](size_t Index);
        __host__ __device__ __forceinline constexpr const _T& operator[](size_t Index) const;

        __host__ __device__ __forceinline constexpr FixedVector<_T, 2> operator+(FixedVector<_T, 2> Other) const;
        __host__ __device__ __forceinline void operator+=(FixedVector<_T, 2> Other);
        __host__ __device__ __forceinline constexpr FixedVector<_T, 2> operator-(FixedVector<_T, 2> Other) const;
        __host__ __device__ __forceinline void operator-=(FixedVector<_T, 2> Other);
        __host__ __device__ __forceinline constexpr FixedVector<_T, 2> operator*(_T Other) const;
        __host__ __device__ __forceinline void operator*=(_T Other);
        __host__ __device__ __forceinline constexpr FixedVector<_T, 2> operator/(_T Other) const;
        __host__ __device__ __forceinline void operator/=(_T Other);

        __host__ __device__ static __forceinline constexpr _T Dot(FixedVector<_T, 2> Left, FixedVector<_T, 2> Right);
        __host__ __device__ __forceinline constexpr FixedVector<_T, 2> Cross() const;
        __host__ __device__ static __forceinline constexpr FixedVector<_T, 2> Cross(FixedVector<_T, 2> Value);
        __host__ __device__ static __forceinline constexpr _T Cross(FixedVector<_T, 2> Left, FixedVector<_T, 2> Right);
        
        __host__ __device__ __forceinline constexpr _T MagnatudeSquared() const;
        __host__ __device__ __forceinline _T MagnatudeI() const requires std::integral<_T>;
        __host__ __device__ __forceinline float MagnatudeF() const requires std::integral<_T>;
        __host__ __device__ __forceinline std::conditional_t<std::floating_point<_T>, _T, double> Magnatude() const;

        __forceinline size_t SerializedSize() const;
        __forceinline void Serialize(void*& Data) const;
        static __forceinline FixedVector<_T, 2> Deserialize(const void*& Data);
        static __forceinline void Deserialize(const void*& Data, void* Value);
    };
    template <typename _T>
        requires std::is_arithmetic_v<_T>
    struct FixedVector<_T, 3> final {
        union {
            struct { _T x, y, z; };
            _T v[3];
        };

        __host__ __device__ __forceinline constexpr FixedVector();
        __host__ __device__ __forceinline constexpr FixedVector(_T X, _T Y, _T Z);
        __host__ __device__ __forceinline constexpr FixedVector(const _T V[3]);

        __host__ __device__ __forceinline constexpr _T& operator[](size_t Index);
        __host__ __device__ __forceinline constexpr const _T& operator[](size_t Index) const;

        __host__ __device__ __forceinline constexpr FixedVector<_T, 3> operator+(FixedVector<_T, 3> Other) const;
        __host__ __device__ __forceinline void operator+=(FixedVector<_T, 3> Other);
        __host__ __device__ __forceinline constexpr FixedVector<_T, 3> operator-(FixedVector<_T, 3> Other) const;
        __host__ __device__ __forceinline void operator-=(FixedVector<_T, 3> Other);
        __host__ __device__ __forceinline constexpr FixedVector<_T, 3> operator*(_T Other) const;
        __host__ __device__ __forceinline void operator*=(_T Other);
        __host__ __device__ __forceinline constexpr FixedVector<_T, 3> operator/(_T Other) const;
        __host__ __device__ __forceinline void operator/=(_T Other);

        __host__ __device__ static __forceinline constexpr _T Dot(FixedVector<_T, 3> Left, FixedVector<_T, 3> Right);
        __host__ __device__ static __forceinline constexpr FixedVector<_T, 3> Cross(FixedVector<_T, 3> Left, FixedVector<_T, 3> Right);
        
        __host__ __device__ __forceinline constexpr _T MagnatudeSquared() const;
        __host__ __device__ __forceinline _T MagnatudeI() const requires std::integral<_T>;
        __host__ __device__ __forceinline float MagnatudeF() const requires std::integral<_T>;
        __host__ __device__ __forceinline std::conditional_t<std::floating_point<_T>, _T, double> Magnatude() const;

        __forceinline size_t SerializedSize() const;
        __forceinline void Serialize(void*& Data) const;
        static __forceinline FixedVector<_T, 3> Deserialize(const void*& Data);
        static __forceinline void Deserialize(const void*& Data, void* Value);
    };
    template <typename _T>
        requires std::is_arithmetic_v<_T>
    struct FixedVector<_T, 4> final {
        union {
            struct { _T x, y, z, w; };
            _T v[4];
        };

        __host__ __device__ __forceinline constexpr FixedVector();
        __host__ __device__ __forceinline constexpr FixedVector(_T X, _T Y, _T Z, _T W);
        __host__ __device__ __forceinline constexpr FixedVector(const _T V[4]);

        __host__ __device__ __forceinline constexpr _T& operator[](size_t Index);
        __host__ __device__ __forceinline constexpr const _T& operator[](size_t Index) const;

        __host__ __device__ __forceinline constexpr FixedVector<_T, 4> operator+(FixedVector<_T, 4> Other) const;
        __host__ __device__ __forceinline void operator+=(FixedVector<_T, 4> Other);
        __host__ __device__ __forceinline constexpr FixedVector<_T, 4> operator-(FixedVector<_T, 4> Other) const;
        __host__ __device__ __forceinline void operator-=(FixedVector<_T, 4> Other);
        __host__ __device__ __forceinline constexpr FixedVector<_T, 4> operator*(_T Other) const;
        __host__ __device__ __forceinline void operator*=(_T Other);
        __host__ __device__ __forceinline constexpr FixedVector<_T, 4> operator/(_T Other) const;
        __host__ __device__ __forceinline void operator/=(_T Other);

        __host__ __device__ static __forceinline constexpr _T Dot(FixedVector<_T, 4> Left, FixedVector<_T, 4> Right);
        
        __host__ __device__ __forceinline constexpr _T MagnatudeSquared() const;
        __host__ __device__ __forceinline _T MagnatudeI() const requires std::integral<_T>;
        __host__ __device__ __forceinline float MagnatudeF() const requires std::integral<_T>;
        __host__ __device__ __forceinline std::conditional_t<std::floating_point<_T>, _T, double> Magnatude() const;

        __forceinline size_t SerializedSize() const;
        __forceinline void Serialize(void*& Data) const;
        static __forceinline FixedVector<_T, 4> Deserialize(const void*& Data);
        static __forceinline void Deserialize(const void*& Data, void* Value);
    };

    using float_1 = FixedVector<float, 1>;
    using float_2 = FixedVector<float, 2>;
    using float_3 = FixedVector<float, 3>;
    using float_4 = FixedVector<float, 4>;
    using double_1 = FixedVector<double, 1>;
    using double_2 = FixedVector<double, 2>;
    using double_3 = FixedVector<double, 3>;
    using double_4 = FixedVector<double, 4>;
    using int8_1 = FixedVector<int8_t, 1>;
    using int8_2 = FixedVector<int8_t, 2>;
    using int8_3 = FixedVector<int8_t, 3>;
    using int8_4 = FixedVector<int8_t, 4>;
    using uint8_1 = FixedVector<uint8_t, 1>;
    using uint8_2 = FixedVector<uint8_t, 2>;
    using uint8_3 = FixedVector<uint8_t, 3>;
    using uint8_4 = FixedVector<uint8_t, 4>;
    using int16_1 = FixedVector<int16_t, 1>;
    using int16_2 = FixedVector<int16_t, 2>;
    using int16_3 = FixedVector<int16_t, 3>;
    using int16_4 = FixedVector<int16_t, 4>;
    using uint16_1 = FixedVector<uint16_t, 1>;
    using uint16_2 = FixedVector<uint16_t, 2>;
    using uint16_3 = FixedVector<uint16_t, 3>;
    using uint16_4 = FixedVector<uint16_t, 4>;
    using int32_1 = FixedVector<int32_t, 1>;
    using int32_2 = FixedVector<int32_t, 2>;
    using int32_3 = FixedVector<int32_t, 3>;
    using int32_4 = FixedVector<int32_t, 4>;
    using uint32_1 = FixedVector<uint32_t, 1>;
    using uint32_2 = FixedVector<uint32_t, 2>;
    using uint32_3 = FixedVector<uint32_t, 3>;
    using uint32_4 = FixedVector<uint32_t, 4>;
    using int64_1 = FixedVector<int64_t, 1>;
    using int64_2 = FixedVector<int64_t, 2>;
    using int64_3 = FixedVector<int64_t, 3>;
    using int64_4 = FixedVector<int64_t, 4>;
    using uint64_1 = FixedVector<uint64_t, 1>;
    using uint64_2 = FixedVector<uint64_t, 2>;
    using uint64_3 = FixedVector<uint64_t, 3>;
    using uint64_4 = FixedVector<uint64_t, 4>;
}

template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr BrendanCUDA::FixedVector<_T, _Size>::FixedVector() {
    for (size_t i = 0; i < _Size; ++i) {
        v[i] = 0;
    }
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
template <std::convertible_to<uint32_t>... _Ts>
    requires (sizeof...(_Ts) == _Size)
__host__ __device__ __forceinline constexpr BrendanCUDA::FixedVector<_T, _Size>::FixedVector(_Ts... V) {
    _T tempVs[_Size] = { V... };
    for (size_t i = 0; i < _Size; ++i)
        v[i] = tempVs[i];
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr BrendanCUDA::FixedVector<_T, _Size>::FixedVector(const _T V[_Size]) {
    for (size_t i = 0; i < _Size; ++i) {
        v[i] = V[i];
    }
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr _T& BrendanCUDA::FixedVector<_T, _Size>::operator[](size_t Index) {
    return v[Index];
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr const _T& BrendanCUDA::FixedVector<_T, _Size>::operator[](size_t Index) const {
    return v[Index];
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr auto BrendanCUDA::FixedVector<_T, _Size>::operator+(FixedVector<_T, _Size> Other) const -> FixedVector<_T, _Size> {
    FixedVector<_T, _Size> r;
    for (size_t i = 0; i < _Size; ++i) {
        r[i] = v[i] + Other[i];
    }
    return r;
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline void BrendanCUDA::FixedVector<_T, _Size>::operator+=(FixedVector<_T, _Size> Other) {
    for (size_t i = 0; i < _Size; ++i) {
        v[i] += Other[i];
    }
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr auto BrendanCUDA::FixedVector<_T, _Size>::operator-(FixedVector<_T, _Size> Other) const -> FixedVector<_T, _Size> {
    FixedVector<_T, _Size> r;
    for (size_t i = 0; i < _Size; ++i) {
        r[i] = v[i] - Other[i];
    }
    return r;
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline void BrendanCUDA::FixedVector<_T, _Size>::operator-=(FixedVector<_T, _Size> Other) {
    for (size_t i = 0; i < _Size; ++i) {
        v[i] -= Other[i];
    }
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr auto BrendanCUDA::FixedVector<_T, _Size>::operator*(_T Other) const -> FixedVector<_T, _Size> {
    FixedVector<_T, _Size> r;
    for (size_t i = 0; i < _Size; ++i) {
        r[i] = v[i] * Other;
    }
    return r;
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline void BrendanCUDA::FixedVector<_T, _Size>::operator*=(_T Other) {
    for (size_t i = 0; i < _Size; ++i) {
        v[i] *= Other;
    }
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr auto BrendanCUDA::FixedVector<_T, _Size>::operator/(_T Other) const -> FixedVector<_T, _Size> {
    FixedVector<_T, _Size> r;
    for (size_t i = 0; i < _Size; ++i) {
        r[i] = v[i] / Other;
    }
    return r;
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline void BrendanCUDA::FixedVector<_T, _Size>::operator/=(_T Other) {
    for (size_t i = 0; i < _Size; ++i) {
        v[i] /= Other;
    }
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr _T BrendanCUDA::FixedVector<_T, _Size>::Dot(FixedVector<_T, _Size> Left, FixedVector<_T, _Size> Right) {
    _T t = 0;
    for (size_t i = 0; i < _Size; ++i) {
        t += Left[i] * Right[i];
    }
    return t;
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr _T BrendanCUDA::FixedVector<_T, _Size>::MagnatudeSquared() const {
    _T t = 0;
    for (size_t i = 0; i < _Size; ++i) {
        t += v[i] * v[i];
    }
    return t;
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline _T BrendanCUDA::FixedVector<_T, _Size>::MagnatudeI() const requires std::integral<_T> {
    return Math::sqrt(MagnatudeSquared());
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline float BrendanCUDA::FixedVector<_T, _Size>::MagnatudeF() const requires std::integral<_T> {
    return sqrt((float)MagnatudeSquared());
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline std::conditional_t<std::floating_point<_T>, _T, double> BrendanCUDA::FixedVector<_T, _Size>::Magnatude() const {
    if constexpr (std::floating_point<_T>) {
        return sqrt(MagnatudeSquared());
    }
    else {
        return sqrt((double)MagnatudeSquared());
    }
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__forceinline size_t BrendanCUDA::FixedVector<_T, _Size>::SerializedSize() const {
    return sizeof(_T) * _Size;
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__forceinline void BrendanCUDA::FixedVector<_T, _Size>::Serialize(void*& Data) const {
    for (size_t i = 0; i < _Size; ++i)
        BSerializer::Serialize(Data, v[i]);
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__forceinline BrendanCUDA::FixedVector<_T, _Size> BrendanCUDA::FixedVector<_T, _Size>::Deserialize(const void*& Data) {
    BrendanCUDA::FixedVector<_T, _Size> vec;
    for (size_t i = 0; i < _Size; ++i)
        vec[i] = BSerializer::Deserialize<_T>(Data);
    return vec;
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__forceinline void BrendanCUDA::FixedVector<_T, _Size>::Deserialize(const void*& Data, void* Value) {
    FixedVector<_T, _Size>* p_vec = new (Value) FixedVector<_T, _Size>;
    FixedVector<_T, _Size>& vec = *p_vec;
    for (size_t i = 0; i < _Size; ++i)
        vec[i] = BSerializer::Deserialize<_T>(Data);
}

template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr BrendanCUDA::FixedVector<_T, 1>::FixedVector()
    : x(0) { }
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr BrendanCUDA::FixedVector<_T, 1>::FixedVector(_T x)
    : x(x) { }
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr BrendanCUDA::FixedVector<_T, 1>::FixedVector(const _T v[1])
    : x(v[0]) { }
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr _T& BrendanCUDA::FixedVector<_T, 1>::operator[](size_t Index) {
    return v[Index];
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr const _T& BrendanCUDA::FixedVector<_T, 1>::operator[](size_t Index) const {
    return v[Index];
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr auto BrendanCUDA::FixedVector<_T, 1>::operator+(FixedVector<_T, 1> Other) const -> BrendanCUDA::FixedVector<_T, 1> {
    return FixedVector<_T, 1>(x + Other.x);
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline void BrendanCUDA::FixedVector<_T, 1>::operator+=(FixedVector<_T, 1> Other) {
    x += Other.x;
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr auto BrendanCUDA::FixedVector<_T, 1>::operator-(FixedVector<_T, 1> Other) const -> BrendanCUDA::FixedVector<_T, 1> {
    return FixedVector<_T, 1>(x - Other.x);
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline void BrendanCUDA::FixedVector<_T, 1>::operator-=(FixedVector<_T, 1> Other) {
    x -= Other.x;
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr auto BrendanCUDA::FixedVector<_T, 1>::operator*(_T Other) const -> BrendanCUDA::FixedVector<_T, 1> {
    return FixedVector<_T, 1>(x * Other);
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline void BrendanCUDA::FixedVector<_T, 1>::operator*=(_T Other) {
    x *= Other;
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr auto BrendanCUDA::FixedVector<_T, 1>::operator/(_T Other) const -> BrendanCUDA::FixedVector<_T, 1> {
    return FixedVector<_T, 1>(x / Other);
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline void BrendanCUDA::FixedVector<_T, 1>::operator/=(_T Other) {
    x /= Other;
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr _T BrendanCUDA::FixedVector<_T, 1>::Dot(FixedVector<_T, 1> Left, FixedVector<_T, 1> Right) {
    return Left.x * Right.x;
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr _T BrendanCUDA::FixedVector<_T, 1>::MagnatudeSquared() const {
    return x * x;
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__forceinline size_t BrendanCUDA::FixedVector<_T, 1>::SerializedSize() const {
    return sizeof(_T);
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__forceinline void BrendanCUDA::FixedVector<_T, 1>::Serialize(void*& Data) const {
    BSerializer::Serialize(Data, x);
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__forceinline BrendanCUDA::FixedVector<_T, 1> BrendanCUDA::FixedVector<_T, 1>::Deserialize(const void*& Data) {
    _T x = BSerializer::Deserialize<_T>(Data);
    return BrendanCUDA::FixedVector<_T, 1>(x);
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__forceinline void BrendanCUDA::FixedVector<_T, 1>::Deserialize(const void*& Data, void* Value) {
    _T x = BSerializer::Deserialize<_T>(Data);
    new (Value) FixedVector<_T, 1>(x);
}

template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr BrendanCUDA::FixedVector<_T, 2>::FixedVector()
    : x(0),
      y(0) { }
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr BrendanCUDA::FixedVector<_T, 2>::FixedVector(_T X, _T Y)
    : x(X),
      y(Y) { }
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr BrendanCUDA::FixedVector<_T, 2>::FixedVector(const _T V[2])
    : x(V[0]),
      y(V[1]) { }
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr _T& BrendanCUDA::FixedVector<_T, 2>::operator[](size_t Index) {
    return v[Index];
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr const _T& BrendanCUDA::FixedVector<_T, 2>::operator[](size_t Index) const {
    return v[Index];
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr auto BrendanCUDA::FixedVector<_T, 2>::operator+(FixedVector<_T, 2> Other) const -> BrendanCUDA::FixedVector<_T, 2> {
    return FixedVector<_T, 2>(x + Other.x, y + Other.y);
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline void BrendanCUDA::FixedVector<_T, 2>::operator+=(FixedVector<_T, 2> Other) {
    x += Other.x;
    y += Other.y;
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr auto BrendanCUDA::FixedVector<_T, 2>::operator-(FixedVector<_T, 2> Other) const -> BrendanCUDA::FixedVector<_T, 2> {
    return FixedVector<_T, 2>(x - Other.x, y - Other.y);
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline void BrendanCUDA::FixedVector<_T, 2>::operator-=(FixedVector<_T, 2> Other) {
    x -= Other.x;
    y -= Other.y;
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr auto BrendanCUDA::FixedVector<_T, 2>::operator*(_T Other) const -> BrendanCUDA::FixedVector<_T, 2> {
    return FixedVector<_T, 2>(x * Other, y * Other);
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline void BrendanCUDA::FixedVector<_T, 2>::operator*=(_T Other) {
    x *= Other;
    y *= Other;
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr auto BrendanCUDA::FixedVector<_T, 2>::operator/(_T Other) const -> BrendanCUDA::FixedVector<_T, 2> {
    return FixedVector<_T, 2>(x / Other, y / Other);
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline void BrendanCUDA::FixedVector<_T, 2>::operator/=(_T Other) {
    x /= Other;
    y /= Other;
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr _T BrendanCUDA::FixedVector<_T, 2>::Dot(FixedVector<_T, 2> Left, FixedVector<_T, 2> Right) {
    return Left.x * Right.x + Left.y * Right.y;
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr BrendanCUDA::FixedVector<_T, 2> BrendanCUDA::FixedVector<_T, 2>::Cross() const {
    return FixedVector<_T, 2>(-y, x);
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr BrendanCUDA::FixedVector<_T, 2> BrendanCUDA::FixedVector<_T, 2>::Cross(FixedVector<_T, 2> Value) {
    return Value.Cross();
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr _T BrendanCUDA::FixedVector<_T, 2>::Cross(FixedVector<_T, 2> Left, FixedVector<_T, 2> Right) {
    return Left.x * Right.y - Left.y * Right.x;
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr _T BrendanCUDA::FixedVector<_T, 2>::MagnatudeSquared() const {
    return x * x + y * y;
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__forceinline size_t BrendanCUDA::FixedVector<_T, 2>::SerializedSize() const {
    return sizeof(_T) << 1;
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__forceinline void BrendanCUDA::FixedVector<_T, 2>::Serialize(void*& Data) const {
    BSerializer::Serialize(Data, x);
    BSerializer::Serialize(Data, y);
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__forceinline BrendanCUDA::FixedVector<_T, 2> BrendanCUDA::FixedVector<_T, 2>::Deserialize(const void*& Data) {
    _T x = BSerializer::Deserialize<_T>(Data);
    _T y = BSerializer::Deserialize<_T>(Data);
    return BrendanCUDA::FixedVector<_T, 2>(x, y);
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__forceinline void BrendanCUDA::FixedVector<_T, 2>::Deserialize(const void*& Data, void* Value) {
    _T x = BSerializer::Deserialize<_T>(Data);
    _T y = BSerializer::Deserialize<_T>(Data);
    new (Value) FixedVector<_T, 2>(x, y);
}

template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr BrendanCUDA::FixedVector<_T, 3>::FixedVector()
    : x(0),
      y(0),
      z(0) { }
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr BrendanCUDA::FixedVector<_T, 3>::FixedVector(_T X, _T Y, _T Z)
    : x(X),
      y(Y),
      z(Z) { }
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr BrendanCUDA::FixedVector<_T, 3>::FixedVector(const _T V[3])
    : x(V[0]),
      y(V[1]),
      z(V[2]) { }
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr _T& BrendanCUDA::FixedVector<_T, 3>::operator[](size_t Index) {
    return v[Index];
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr const _T& BrendanCUDA::FixedVector<_T, 3>::operator[](size_t Index) const {
    return v[Index];
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr BrendanCUDA::FixedVector<_T, 3> BrendanCUDA::FixedVector<_T, 3>::operator+(FixedVector<_T, 3> Other) const {
    return FixedVector<_T, 3>(x + Other.x, y + Other.y, z + Other.z);
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline void BrendanCUDA::FixedVector<_T, 3>::operator+=(FixedVector<_T, 3> Other) {
    x += Other.x;
    y += Other.y;
    z += Other.z;
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr BrendanCUDA::FixedVector<_T, 3> BrendanCUDA::FixedVector<_T, 3>::operator-(FixedVector<_T, 3> Other) const {
    return FixedVector<_T, 3>(x - Other.x, y - Other.y, z - Other.z);
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline void BrendanCUDA::FixedVector<_T, 3>::operator-=(FixedVector<_T, 3> Other) {
    x -= Other.x;
    y -= Other.y;
    z -= Other.z;
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr BrendanCUDA::FixedVector<_T, 3> BrendanCUDA::FixedVector<_T, 3>::operator*(_T Other) const {
    return FixedVector<_T, 3>(x * Other, y * Other, z * Other);
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline void BrendanCUDA::FixedVector<_T, 3>::operator*=(_T Other) {
    x *= Other;
    y *= Other;
    z *= Other;
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr BrendanCUDA::FixedVector<_T, 3> BrendanCUDA::FixedVector<_T, 3>::operator/(_T Other) const {
    return FixedVector<_T, 3>(x / Other, y / Other, z / Other);
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline void BrendanCUDA::FixedVector<_T, 3>::operator/=(_T Other) {
    x /= Other;
    y /= Other;
    z /= Other;
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr _T BrendanCUDA::FixedVector<_T, 3>::Dot(FixedVector<_T, 3> Left, FixedVector<_T, 3> Right) {
    return Left.x * Right.x + Left.y * Right.y + Left.z * Right.z;
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr BrendanCUDA::FixedVector<_T, 3> BrendanCUDA::FixedVector<_T, 3>::Cross(FixedVector<_T, 3> Left, FixedVector<_T, 3> Right) {
    return FixedVector<_T, 3>(
        Left.y * Right.z - Left.z * Right.y,
        Left.z * Right.x - Left.x * Right.z,
        Left.x * Right.y - Left.y * Right.x
    );
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr _T BrendanCUDA::FixedVector<_T, 3>::MagnatudeSquared() const {
    return x * x + y * y + z * z;
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__forceinline size_t BrendanCUDA::FixedVector<_T, 3>::SerializedSize() const {
    return sizeof(_T) * 3;
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__forceinline void BrendanCUDA::FixedVector<_T, 3>::Serialize(void*& Data) const {
    BSerializer::Serialize(Data, x);
    BSerializer::Serialize(Data, y);
    BSerializer::Serialize(Data, z);
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__forceinline BrendanCUDA::FixedVector<_T, 3> BrendanCUDA::FixedVector<_T, 3>::Deserialize(const void*& Data) {
    _T x = BSerializer::Deserialize<_T>(Data);
    _T y = BSerializer::Deserialize<_T>(Data);
    _T z = BSerializer::Deserialize<_T>(Data);
    return BrendanCUDA::FixedVector<_T, 3>(x, y, z);
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__forceinline void BrendanCUDA::FixedVector<_T, 3>::Deserialize(const void*& Data, void* Value) {
    _T x = BSerializer::Deserialize<_T>(Data);
    _T y = BSerializer::Deserialize<_T>(Data);
    _T z = BSerializer::Deserialize<_T>(Data);
    new (Value) FixedVector<_T, 3>(x, y, z);
}

template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr BrendanCUDA::FixedVector<_T, 4>::FixedVector()
    : x(0),
      y(0),
      z(0),
      w(0) { }
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr BrendanCUDA::FixedVector<_T, 4>::FixedVector(_T X, _T Y, _T Z, _T W)
    : x(X),
      y(Y),
      z(Z),
      w(W) { }
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr BrendanCUDA::FixedVector<_T, 4>::FixedVector(const _T V[4])
    : x(V[0]),
      y(V[1]),
      z(V[2]),
      w(V[3]) { }
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr _T& BrendanCUDA::FixedVector<_T, 4>::operator[](size_t Index) {
    return v[Index];
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr const _T& BrendanCUDA::FixedVector<_T, 4>::operator[](size_t Index) const {
    return v[Index];
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr BrendanCUDA::FixedVector<_T, 4> BrendanCUDA::FixedVector<_T, 4>::operator+(FixedVector<_T, 4> Other) const {
    return FixedVector<_T, 4>(x + Other.x, y + Other.y, z + Other.z, w + Other.w);
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline void BrendanCUDA::FixedVector<_T, 4>::operator+=(FixedVector<_T, 4> Other) {
    x += Other.x;
    y += Other.y;
    z += Other.z;
    w += Other.w;
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr BrendanCUDA::FixedVector<_T, 4> BrendanCUDA::FixedVector<_T, 4>::operator-(FixedVector<_T, 4> Other) const {
    return FixedVector<_T, 4>(x - Other.x, y - Other.y, z - Other.z, w - Other.w);
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline void BrendanCUDA::FixedVector<_T, 4>::operator-=(FixedVector<_T, 4> Other) {
    x -= Other.x;
    y -= Other.y;
    z -= Other.z;
    w -= Other.w;
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr BrendanCUDA::FixedVector<_T, 4> BrendanCUDA::FixedVector<_T, 4>::operator*(_T Other) const {
    return FixedVector<_T, 4>(x * Other, y * Other, z * Other, w * Other);
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline void BrendanCUDA::FixedVector<_T, 4>::operator*=(_T Other) {
    x *= Other;
    y *= Other;
    z *= Other;
    w *= Other;
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr BrendanCUDA::FixedVector<_T, 4> BrendanCUDA::FixedVector<_T, 4>::operator/(_T Other) const {
    return FixedVector<_T, 4>(x / Other, y / Other, z / Other, w / Other);
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline void BrendanCUDA::FixedVector<_T, 4>::operator/=(_T Other) {
    x /= Other;
    y /= Other;
    z /= Other;
    w /= Other;
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr _T BrendanCUDA::FixedVector<_T, 4>::Dot(FixedVector<_T, 4> Left, FixedVector<_T, 4> Right) {
    return Left.x * Right.x + Left.y * Right.y + Left.z * Right.z + Left.w * Right.w;
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr _T BrendanCUDA::FixedVector<_T, 4>::MagnatudeSquared() const {
    return x * x + y * y + z * z + w * w;
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__forceinline size_t BrendanCUDA::FixedVector<_T, 4>::SerializedSize() const {
    return sizeof(_T) << 2;
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__forceinline void BrendanCUDA::FixedVector<_T, 4>::Serialize(void*& Data) const {
    BSerializer::Serialize(Data, x);
    BSerializer::Serialize(Data, y);
    BSerializer::Serialize(Data, z);
    BSerializer::Serialize(Data, w);
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__forceinline BrendanCUDA::FixedVector<_T, 4> BrendanCUDA::FixedVector<_T, 4>::Deserialize(const void*& Data) {
    _T x = BSerializer::Deserialize<_T>(Data);
    _T y = BSerializer::Deserialize<_T>(Data);
    _T z = BSerializer::Deserialize<_T>(Data);
    _T w = BSerializer::Deserialize<_T>(Data);
    return BrendanCUDA::FixedVector<_T, 4>(x, y, z, w);
}
template <typename _T>
    requires std::is_arithmetic_v<_T>
__forceinline void BrendanCUDA::FixedVector<_T, 4>::Deserialize(const void*& Data, void* Value) {
    _T x = BSerializer::Deserialize<_T>(Data);
    _T y = BSerializer::Deserialize<_T>(Data);
    _T z = BSerializer::Deserialize<_T>(Data);
    _T w = BSerializer::Deserialize<_T>(Data);
    new (Value) FixedVector<_T, 4>(x, y, z, w);
}