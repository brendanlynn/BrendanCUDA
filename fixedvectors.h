#pragma once

#include "BSerializer/Serializer.h"
#include "mathfuncs.h"
#include <cmath>
#include <concepts>
#include <cstdarg>
#include <cstdint>
#include <cuda_runtime.h>

namespace bcuda {
    namespace details {
        template <typename _T, size_t _Size>
            requires std::is_arithmetic_v<_T>
        struct FixedVectorBase {
            static_assert(_Size, "_Size must be positive.");

            _T v[_Size];
        };
        template <typename _T>
        struct FixedVectorBase<_T, 1> {
            union {
                _T x;
                _T v[1];
            };
        };
        template <typename _T>
        struct FixedVectorBase<_T, 2> {
            union {
                struct { _T x; _T y; };
                _T v[2];
            };
        };
        template <typename _T>
        struct FixedVectorBase<_T, 3> {
            union {
                struct { _T x; _T y; _T z; };
                _T v[3];
            };
        };
        template <typename _T>
        struct FixedVectorBase<_T, 4> {
            union {
                struct { _T x; _T y; _T z; _T w; };
                _T v[4];
            };
        };
    }
    template <typename _T, size_t _Size>
        requires std::is_arithmetic_v<_T>
    struct FixedVector
        : public details::FixedVectorBase<_T, _Size> {
        __host__ __device__ __forceinline constexpr FixedVector();
        template <std::convertible_to<uint32_t>... _Ts>
            requires (sizeof...(_Ts) == _Size)
        __host__ __device__ __forceinline constexpr FixedVector(_Ts... V) {
            _T tempVs[_Size] = { V... };
            for (size_t i = 0; i < _Size; ++i)
                this->v[i] = tempVs[i];
        }
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

        __host__ __device__ __forceinline constexpr FixedVector<_T, 2> Cross() const requires (_Size == 2) {
            return FixedVector<_T, 2>(-this->y, this->x);
        }
        __host__ __device__ static __forceinline constexpr FixedVector<_T, 2> Cross(FixedVector<_T, 2> Value) requires (_Size == 2);
        __host__ __device__ static __forceinline constexpr _T Cross(FixedVector<_T, 2> Left, FixedVector<_T, 2> Right) requires (_Size == 2);

        __host__ __device__ static __forceinline constexpr FixedVector<_T, 3> Cross(FixedVector<_T, 3> Left, FixedVector<_T, 3> Right) requires (_Size == 3);
        
        __host__ __device__ __forceinline constexpr _T MagnatudeSquared() const;
        __host__ __device__ __forceinline _T MagnatudeI() const requires std::integral<_T>;
        __host__ __device__ __forceinline float MagnatudeF() const requires std::integral<_T>;
        __host__ __device__ __forceinline std::conditional_t<std::floating_point<_T>, _T, double> Magnatude() const;

        __forceinline size_t SerializedSize() const;
        __forceinline void Serialize(void*& Data) const;
        static __forceinline FixedVector<_T, _Size> Deserialize(const void*& Data);
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
__host__ __device__ __forceinline constexpr bcuda::FixedVector<_T, _Size>::FixedVector() {
    for (size_t i = 0; i < _Size; ++i) {
        this->v[i] = 0;
    }
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr bcuda::FixedVector<_T, _Size>::FixedVector(const _T V[_Size]) {
    for (size_t i = 0; i < _Size; ++i) {
        this->v[i] = V[i];
    }
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr _T& bcuda::FixedVector<_T, _Size>::operator[](size_t Index) {
    return this->v[Index];
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr const _T& bcuda::FixedVector<_T, _Size>::operator[](size_t Index) const {
    return this->v[Index];
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr auto bcuda::FixedVector<_T, _Size>::operator+(FixedVector<_T, _Size> Other) const -> FixedVector<_T, _Size> {
    FixedVector<_T, _Size> r;
    for (size_t i = 0; i < _Size; ++i) {
        r[i] = this->v[i] + Other[i];
    }
    return r;
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline void bcuda::FixedVector<_T, _Size>::operator+=(FixedVector<_T, _Size> Other) {
    for (size_t i = 0; i < _Size; ++i) {
        this->v[i] += Other[i];
    }
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr auto bcuda::FixedVector<_T, _Size>::operator-(FixedVector<_T, _Size> Other) const -> FixedVector<_T, _Size> {
    FixedVector<_T, _Size> r;
    for (size_t i = 0; i < _Size; ++i) {
        r[i] = this->v[i] - Other[i];
    }
    return r;
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline void bcuda::FixedVector<_T, _Size>::operator-=(FixedVector<_T, _Size> Other) {
    for (size_t i = 0; i < _Size; ++i) {
        this->v[i] -= Other[i];
    }
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr auto bcuda::FixedVector<_T, _Size>::operator*(_T Other) const -> FixedVector<_T, _Size> {
    FixedVector<_T, _Size> r;
    for (size_t i = 0; i < _Size; ++i) {
        r[i] = this->v[i] * Other;
    }
    return r;
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline void bcuda::FixedVector<_T, _Size>::operator*=(_T Other) {
    for (size_t i = 0; i < _Size; ++i) {
        this->v[i] *= Other;
    }
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr auto bcuda::FixedVector<_T, _Size>::operator/(_T Other) const -> FixedVector<_T, _Size> {
    FixedVector<_T, _Size> r;
    for (size_t i = 0; i < _Size; ++i) {
        r[i] = this->v[i] / Other;
    }
    return r;
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline void bcuda::FixedVector<_T, _Size>::operator/=(_T Other) {
    for (size_t i = 0; i < _Size; ++i) {
        this->v[i] /= Other;
    }
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr _T bcuda::FixedVector<_T, _Size>::Dot(FixedVector<_T, _Size> Left, FixedVector<_T, _Size> Right) {
    _T t = 0;
    for (size_t i = 0; i < _Size; ++i) {
        t += Left[i] * Right[i];
    }
    return t;
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr _T bcuda::FixedVector<_T, _Size>::MagnatudeSquared() const {
    _T t = 0;
    for (size_t i = 0; i < _Size; ++i) {
        _T thisV = this->v[i];
        t += thisV * thisV;
    }
    return t;
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline _T bcuda::FixedVector<_T, _Size>::MagnatudeI() const requires std::integral<_T> {
    return math::sqrt(MagnatudeSquared());
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline float bcuda::FixedVector<_T, _Size>::MagnatudeF() const requires std::integral<_T> {
    return sqrt((float)MagnatudeSquared());
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline std::conditional_t<std::floating_point<_T>, _T, double> bcuda::FixedVector<_T, _Size>::Magnatude() const {
    if constexpr (std::floating_point<_T>) {
        return sqrt(MagnatudeSquared());
    }
    else {
        return sqrt((double)MagnatudeSquared());
    }
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__forceinline size_t bcuda::FixedVector<_T, _Size>::SerializedSize() const {
    return sizeof(_T) * _Size;
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__forceinline void bcuda::FixedVector<_T, _Size>::Serialize(void*& Data) const {
    for (size_t i = 0; i < _Size; ++i)
        BSerializer::Serialize(Data, this->v[i]);
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__forceinline bcuda::FixedVector<_T, _Size> bcuda::FixedVector<_T, _Size>::Deserialize(const void*& Data) {
    bcuda::FixedVector<_T, _Size> vec;
    for (size_t i = 0; i < _Size; ++i)
        vec[i] = BSerializer::Deserialize<_T>(Data);
    return vec;
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__forceinline void bcuda::FixedVector<_T, _Size>::Deserialize(const void*& Data, void* Value) {
    FixedVector<_T, _Size>* p_vec = new (Value) FixedVector<_T, _Size>;
    FixedVector<_T, _Size>& vec = *p_vec;
    for (size_t i = 0; i < _Size; ++i)
        vec[i] = BSerializer::Deserialize<_T>(Data);
}

template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr bcuda::FixedVector<_T, 2> bcuda::FixedVector<_T, _Size>::Cross(FixedVector<_T, 2> Value) requires (_Size == 2) {
    return Value.Cross();
}
template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr _T bcuda::FixedVector<_T, _Size>::Cross(FixedVector<_T, 2> Left, FixedVector<_T, 2> Right) requires (_Size == 2) {
    return Left.x * Right.y - Left.y * Right.x;
}

template <typename _T, size_t _Size>
    requires std::is_arithmetic_v<_T>
__host__ __device__ __forceinline constexpr bcuda::FixedVector<_T, 3> bcuda::FixedVector<_T, _Size>::Cross(FixedVector<_T, 3> Left, FixedVector<_T, 3> Right) requires (_Size == 3) {
    return FixedVector<_T, 3>(
        Left.y * Right.z - Left.z * Right.y,
        Left.z * Right.x - Left.x * Right.z,
        Left.x * Right.y - Left.y * Right.x
    );
}