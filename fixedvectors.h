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
        __host__ __device__ inline constexpr FixedVector() {
            for (size_t i = 0; i < _Size; ++i) {
                this->v[i] = 0;
            }
        }
        template <std::convertible_to<uint32_t>... _Ts>
            requires (sizeof...(_Ts) == _Size)
        __host__ __device__ inline constexpr FixedVector(_Ts... V) {
            _T tempVs[_Size] = { V... };
            for (size_t i = 0; i < _Size; ++i)
                this->v[i] = tempVs[i];
        }
        __host__ __device__ inline constexpr FixedVector(const _T V[_Size]) {
            for (size_t i = 0; i < _Size; ++i) {
                this->v[i] = V[i];
            }
        }

        __host__ __device__ inline constexpr _T& operator[](size_t Index) {
            return this->v[Index];
        }
        __host__ __device__ inline constexpr const _T& operator[](size_t Index) const {
            return this->v[Index];
        }

        __host__ __device__ inline constexpr FixedVector<_T, _Size> operator+(FixedVector<_T, _Size> Other) const {
            FixedVector<_T, _Size> r;
            for (size_t i = 0; i < _Size; ++i) {
                r[i] = this->v[i] + Other[i];
            }
            return r;
        }
        __host__ __device__ inline void operator+=(FixedVector<_T, _Size> Other) {
            for (size_t i = 0; i < _Size; ++i) {
                this->v[i] += Other[i];
            }
        }
        __host__ __device__ inline constexpr FixedVector<_T, _Size> operator-(FixedVector<_T, _Size> Other) const {
            FixedVector<_T, _Size> r;
            for (size_t i = 0; i < _Size; ++i) {
                r[i] = this->v[i] - Other[i];
            }
            return r;
        }
        __host__ __device__ inline void operator-=(FixedVector<_T, _Size> Other) {
            for (size_t i = 0; i < _Size; ++i) {
                this->v[i] -= Other[i];
            }
        }
        __host__ __device__ inline constexpr FixedVector<_T, _Size> operator*(_T Other) const {
            FixedVector<_T, _Size> r;
            for (size_t i = 0; i < _Size; ++i) {
                r[i] = this->v[i] * Other;
            }
            return r;
        }
        __host__ __device__ inline void operator*=(_T Other) {
            for (size_t i = 0; i < _Size; ++i) {
                this->v[i] *= Other;
            }
        }
        __host__ __device__ inline constexpr FixedVector<_T, _Size> operator/(_T Other) const {
            FixedVector<_T, _Size> r;
            for (size_t i = 0; i < _Size; ++i) {
                r[i] = this->v[i] / Other;
            }
            return r;
        }
        __host__ __device__ inline void operator/=(_T Other) {
            for (size_t i = 0; i < _Size; ++i) {
                this->v[i] /= Other;
            }
        }

        __host__ __device__ static inline constexpr _T Dot(FixedVector<_T, _Size> Left, FixedVector<_T, _Size> Right) {
            _T t = 0;
            for (size_t i = 0; i < _Size; ++i) {
                t += Left[i] * Right[i];
            }
            return t;
        }

        __host__ __device__ inline constexpr FixedVector<_T, 2> Cross() const requires (_Size == 2) {
            return FixedVector<_T, 2>(-this->y, this->x);
        }
        __host__ __device__ static inline constexpr FixedVector<_T, 2> Cross(FixedVector<_T, 2> Value) requires (_Size == 2) {
            return Value.Cross();
        }
        __host__ __device__ static inline constexpr _T Cross(FixedVector<_T, 2> Left, FixedVector<_T, 2> Right) requires (_Size == 2) {
            return Left.x * Right.y - Left.y * Right.x;
        }

        __host__ __device__ static inline constexpr FixedVector<_T, 3> Cross(FixedVector<_T, 3> Left, FixedVector<_T, 3> Right) requires (_Size == 3) {
            return FixedVector<_T, 3>(
                Left.y * Right.z - Left.z * Right.y,
                Left.z * Right.x - Left.x * Right.z,
                Left.x * Right.y - Left.y * Right.x
            );
        }
        
        __host__ __device__ inline constexpr _T MagnatudeSquared() const {
            _T t = 0;
            for (size_t i = 0; i < _Size; ++i) {
                _T thisV = this->v[i];
                t += thisV * thisV;
            }
            return t;
        }
        __host__ __device__ inline _T MagnatudeI() const requires std::integral<_T> {
            return math::sqrt(MagnatudeSquared());
        }
        __host__ __device__ inline float MagnatudeF() const requires std::integral<_T> {
            return sqrt((float)MagnatudeSquared());
        }
        __host__ __device__ inline std::conditional_t<std::floating_point<_T>, _T, double> Magnatude() const {
            if constexpr (std::floating_point<_T>) {
                return sqrt(MagnatudeSquared());
            }
            else {
                return sqrt((double)MagnatudeSquared());
            }
        }

        inline size_t SerializedSize() const {
            return sizeof(_T) * _Size;
        }
        inline void Serialize(void*& Data) const {
            for (size_t i = 0; i < _Size; ++i)
                BSerializer::Serialize(Data, this->v[i]);
        }
        static inline FixedVector<_T, _Size> Deserialize(const void*& Data) {
            bcuda::FixedVector<_T, _Size> vec;
            for (size_t i = 0; i < _Size; ++i)
                vec[i] = BSerializer::Deserialize<_T>(Data);
            return vec;
        }
        static inline void Deserialize(const void*& Data, void* Value) {
            FixedVector<_T, _Size>* p_vec = new (Value) FixedVector<_T, _Size>;
            FixedVector<_T, _Size>& vec = *p_vec;
            for (size_t i = 0; i < _Size; ++i)
                vec[i] = BSerializer::Deserialize<_T>(Data);
        }
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