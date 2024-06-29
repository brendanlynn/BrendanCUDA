#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include "brendancuda_math.cuh"

namespace BrendanCUDA {
    class float_2 final {
    public:
        union {
            struct { float x, y; };
            float v[2];
        };
        __host__ __device__ inline constexpr float_2();
        __host__ __device__ inline constexpr float_2(float x, float y);
        __host__ __device__ inline constexpr float_2(const float v[2]);
        __host__ __device__ inline constexpr float_2 operator+(float_2 other) const;
        __host__ __device__ inline constexpr float_2 operator-(float_2 other) const;
        __host__ __device__ inline constexpr float_2 operator*(float other) const;
        __host__ __device__ inline constexpr float_2 operator/(float other) const;
        __host__ __device__ static inline constexpr float Dot(float_2 left, float_2 right);
        __host__ __device__ inline constexpr float MagnatudeSquared() const;
        __host__ __device__ inline float Magnatude() const;
    };
    class float_3 final {
    public:
        union {
            struct { float x, y, z; };
            float v[3];
        };
        __host__ __device__ inline constexpr float_3();
        __host__ __device__ inline constexpr float_3(float x, float y, float z);
        __host__ __device__ inline constexpr float_3(const float v[3]);
        __host__ __device__ inline constexpr float_3 operator+(float_3 other) const;
        __host__ __device__ inline constexpr float_3 operator-(float_3 other) const;
        __host__ __device__ inline constexpr float_3 operator*(float other) const;
        __host__ __device__ inline constexpr float_3 operator/(float other) const;
        __host__ __device__ static inline constexpr float Dot(float_3 left, float_3 right);
        __host__ __device__ inline constexpr float MagnatudeSquared() const;
        __host__ __device__ inline float Magnatude() const;
    };
    class float_4 final {
    public:
        union {
            struct { float x, y, z, w; };
            float v[4];
        };
        __host__ __device__ inline constexpr float_4();
        __host__ __device__ inline constexpr float_4(float x, float y, float z, float w);
        __host__ __device__ inline constexpr float_4(const float v[4]);
        __host__ __device__ inline constexpr float_4 operator+(float_4 other) const;
        __host__ __device__ inline constexpr float_4 operator-(float_4 other) const;
        __host__ __device__ inline constexpr float_4 operator*(float other) const;
        __host__ __device__ inline constexpr float_4 operator/(float other) const;
        __host__ __device__ static inline constexpr float Dot(float_4 left, float_4 right);
        __host__ __device__ inline constexpr float MagnatudeSquared() const;
        __host__ __device__ inline float Magnatude() const;
    };
    class double_2 final {
    public:
        union {
            struct { double x, y; };
            double v[2];
        };
        __host__ __device__ inline constexpr double_2();
        __host__ __device__ inline constexpr double_2(double x, double y);
        __host__ __device__ inline constexpr double_2(const double v[2]);
        __host__ __device__ inline constexpr double_2 operator+(double_2 other) const;
        __host__ __device__ inline constexpr double_2 operator-(double_2 other) const;
        __host__ __device__ inline constexpr double_2 operator*(double other) const;
        __host__ __device__ inline constexpr double_2 operator/(double other) const;
        __host__ __device__ static inline constexpr double Dot(double_2 left, double_2 right);
        __host__ __device__ inline constexpr double MagnatudeSquared() const;
        __host__ __device__ inline double Magnatude() const;
    };
    class double_3 final {
    public:
        union {
            struct { double x, y, z; };
            double v[3];
        };
        __host__ __device__ inline constexpr double_3();
        __host__ __device__ inline constexpr double_3(double x, double y, double z);
        __host__ __device__ inline constexpr double_3(const double v[3]);
        __host__ __device__ inline constexpr double_3 operator+(double_3 other) const;
        __host__ __device__ inline constexpr double_3 operator-(double_3 other) const;
        __host__ __device__ inline constexpr double_3 operator*(double other) const;
        __host__ __device__ inline constexpr double_3 operator/(double other) const;
        __host__ __device__ static inline constexpr double Dot(double_3 left, double_3 right);
        __host__ __device__ inline constexpr double MagnatudeSquared() const;
        __host__ __device__ inline double Magnatude() const;
    };
    class double_4 final {
    public:
        union {
            struct { double x, y, z, w; };
            double v[4];
        };
        __host__ __device__ inline constexpr double_4();
        __host__ __device__ inline constexpr double_4(double x, double y, double z, double w);
        __host__ __device__ inline constexpr double_4(const double v[4]);
        __host__ __device__ inline constexpr double_4 operator+(double_4 other) const;
        __host__ __device__ inline constexpr double_4 operator-(double_4 other) const;
        __host__ __device__ inline constexpr double_4 operator*(double other) const;
        __host__ __device__ inline constexpr double_4 operator/(double other) const;
        __host__ __device__ static inline constexpr double Dot(double_4 left, double_4 right);
        __host__ __device__ inline constexpr double MagnatudeSquared() const;
        __host__ __device__ inline double Magnatude() const;
    };
    class int8_2 final {
    public:
        union {
            struct { int8_t x, y; };
            int8_t v[2];
        };
        __host__ __device__ inline constexpr int8_2();
        __host__ __device__ inline constexpr int8_2(int8_t x, int8_t y);
        __host__ __device__ inline constexpr int8_2(const int8_t v[2]);
        __host__ __device__ inline constexpr int8_2 operator+(int8_2 other) const;
        __host__ __device__ inline constexpr int8_2 operator-(int8_2 other) const;
        __host__ __device__ inline constexpr int8_2 operator*(int8_t other) const;
        __host__ __device__ inline constexpr int8_2 operator/(int8_t other) const;
        __host__ __device__ static inline constexpr int8_t Dot(int8_2 left, int8_2 right);
        __host__ __device__ inline constexpr int8_t MagnatudeSquared() const;
        __host__ __device__ inline int8_t Magnatude() const;
    };
    class int8_3 final {
    public:
        union {
            struct { int8_t x, y, z; };
            int8_t v[3];
        };
        __host__ __device__ inline constexpr int8_3();
        __host__ __device__ inline constexpr int8_3(int8_t x, int8_t y, int8_t z);
        __host__ __device__ inline constexpr int8_3(const int8_t v[3]);
        __host__ __device__ inline constexpr int8_3 operator+(int8_3 other) const;
        __host__ __device__ inline constexpr int8_3 operator-(int8_3 other) const;
        __host__ __device__ inline constexpr int8_3 operator*(int8_t other) const;
        __host__ __device__ inline constexpr int8_3 operator/(int8_t other) const;
        __host__ __device__ static inline constexpr int8_t Dot(int8_3 left, int8_3 right);
        __host__ __device__ inline constexpr int8_t MagnatudeSquared() const;
        __host__ __device__ inline int8_t Magnatude() const;
    };
    class int8_4 final {
    public:
        union {
            struct { int8_t x, y, z, w; };
            int8_t v[4];
        };
        __host__ __device__ inline constexpr int8_4();
        __host__ __device__ inline constexpr int8_4(int8_t x, int8_t y, int8_t z, int8_t w);
        __host__ __device__ inline constexpr int8_4(const int8_t v[4]);
        __host__ __device__ inline constexpr int8_4 operator+(int8_4 other) const;
        __host__ __device__ inline constexpr int8_4 operator-(int8_4 other) const;
        __host__ __device__ inline constexpr int8_4 operator*(int8_t other) const;
        __host__ __device__ inline constexpr int8_4 operator/(int8_t other) const;
        __host__ __device__ static inline constexpr int8_t Dot(int8_4 left, int8_4 right);
        __host__ __device__ inline constexpr int8_t MagnatudeSquared() const;
        __host__ __device__ inline int8_t Magnatude() const;
    };
    class uint8_2 final {
    public:
        union {
            struct { uint8_t x, y; };
            uint8_t v[2];
        };
        __host__ __device__ inline constexpr uint8_2();
        __host__ __device__ inline constexpr uint8_2(uint8_t x, uint8_t y);
        __host__ __device__ inline constexpr uint8_2(const uint8_t v[2]);
        __host__ __device__ inline constexpr uint8_2 operator+(uint8_2 other) const;
        __host__ __device__ inline constexpr uint8_2 operator-(uint8_2 other) const;
        __host__ __device__ inline constexpr uint8_2 operator*(uint8_t other) const;
        __host__ __device__ inline constexpr uint8_2 operator/(uint8_t other) const;
        __host__ __device__ static inline constexpr uint8_t Dot(uint8_2 left, uint8_2 right);
        __host__ __device__ inline constexpr uint8_t MagnatudeSquared() const;
        __host__ __device__ inline uint8_t Magnatude() const;
    };
    class uint8_3 final {
    public:
        union {
            struct { uint8_t x, y, z; };
            uint8_t v[3];
        };
        __host__ __device__ inline constexpr uint8_3();
        __host__ __device__ inline constexpr uint8_3(uint8_t x, uint8_t y, uint8_t z);
        __host__ __device__ inline constexpr uint8_3(const uint8_t v[3]);
        __host__ __device__ inline constexpr uint8_3 operator+(uint8_3 other) const;
        __host__ __device__ inline constexpr uint8_3 operator-(uint8_3 other) const;
        __host__ __device__ inline constexpr uint8_3 operator*(uint8_t other) const;
        __host__ __device__ inline constexpr uint8_3 operator/(uint8_t other) const;
        __host__ __device__ static inline constexpr uint8_t Dot(uint8_3 left, uint8_3 right);
        __host__ __device__ inline constexpr uint8_t MagnatudeSquared() const;
        __host__ __device__ inline uint8_t Magnatude() const;
    };
    class uint8_4 final {
    public:
        union {
            struct { uint8_t x, y, z, w; };
            uint8_t v[4];
        };
        __host__ __device__ inline constexpr uint8_4();
        __host__ __device__ inline constexpr uint8_4(uint8_t x, uint8_t y, uint8_t z, uint8_t w);
        __host__ __device__ inline constexpr uint8_4(const uint8_t v[4]);
        __host__ __device__ inline constexpr uint8_4 operator+(uint8_4 other) const;
        __host__ __device__ inline constexpr uint8_4 operator-(uint8_4 other) const;
        __host__ __device__ inline constexpr uint8_4 operator*(uint8_t other) const;
        __host__ __device__ inline constexpr uint8_4 operator/(uint8_t other) const;
        __host__ __device__ static inline constexpr uint8_t Dot(uint8_4 left, uint8_4 right);
        __host__ __device__ inline constexpr uint8_t MagnatudeSquared() const;
        __host__ __device__ inline uint8_t Magnatude() const;
    };
    class int16_2 final {
    public:
        union {
            struct { int16_t x, y; };
            int16_t v[2];
        };
        __host__ __device__ inline constexpr int16_2();
        __host__ __device__ inline constexpr int16_2(int16_t x, int16_t y);
        __host__ __device__ inline constexpr int16_2(const int16_t v[2]);
        __host__ __device__ inline constexpr int16_2 operator+(int16_2 other) const;
        __host__ __device__ inline constexpr int16_2 operator-(int16_2 other) const;
        __host__ __device__ inline constexpr int16_2 operator*(int16_t other) const;
        __host__ __device__ inline constexpr int16_2 operator/(int16_t other) const;
        __host__ __device__ static inline constexpr int16_t Dot(int16_2 left, int16_2 right);
        __host__ __device__ inline constexpr int16_t MagnatudeSquared() const;
        __host__ __device__ inline int16_t Magnatude() const;
    };
    class int16_3 final {
    public:
        union {
            struct { int16_t x, y, z; };
            int16_t v[3];
        };
        __host__ __device__ inline constexpr int16_3();
        __host__ __device__ inline constexpr int16_3(int16_t x, int16_t y, int16_t z);
        __host__ __device__ inline constexpr int16_3(const int16_t v[3]);
        __host__ __device__ inline constexpr int16_3 operator+(int16_3 other) const;
        __host__ __device__ inline constexpr int16_3 operator-(int16_3 other) const;
        __host__ __device__ inline constexpr int16_3 operator*(int16_t other) const;
        __host__ __device__ inline constexpr int16_3 operator/(int16_t other) const;
        __host__ __device__ static inline constexpr int16_t Dot(int16_3 left, int16_3 right);
        __host__ __device__ inline constexpr int16_t MagnatudeSquared() const;
        __host__ __device__ inline int16_t Magnatude() const;
    };
    class int16_4 final {
    public:
        union {
            struct { int16_t x, y, z, w; };
            int16_t v[4];
        };
        __host__ __device__ inline constexpr int16_4();
        __host__ __device__ inline constexpr int16_4(int16_t x, int16_t y, int16_t z, int16_t w);
        __host__ __device__ inline constexpr int16_4(const int16_t v[4]);
        __host__ __device__ inline constexpr int16_4 operator+(int16_4 other) const;
        __host__ __device__ inline constexpr int16_4 operator-(int16_4 other) const;
        __host__ __device__ inline constexpr int16_4 operator*(int16_t other) const;
        __host__ __device__ inline constexpr int16_4 operator/(int16_t other) const;
        __host__ __device__ static inline constexpr int16_t Dot(int16_4 left, int16_4 right);
        __host__ __device__ inline constexpr int16_t MagnatudeSquared() const;
        __host__ __device__ inline int16_t Magnatude() const;
    };
    class uint16_2 final {
    public:
        union {
            struct { uint16_t x, y; };
            uint16_t v[2];
        };
        __host__ __device__ inline constexpr uint16_2();
        __host__ __device__ inline constexpr uint16_2(uint16_t x, uint16_t y);
        __host__ __device__ inline constexpr uint16_2(const uint16_t v[2]);
        __host__ __device__ inline constexpr uint16_2 operator+(uint16_2 other) const;
        __host__ __device__ inline constexpr uint16_2 operator-(uint16_2 other) const;
        __host__ __device__ inline constexpr uint16_2 operator*(uint16_t other) const;
        __host__ __device__ inline constexpr uint16_2 operator/(uint16_t other) const;
        __host__ __device__ static inline constexpr uint16_t Dot(uint16_2 left, uint16_2 right);
        __host__ __device__ inline constexpr uint16_t MagnatudeSquared() const;
        __host__ __device__ inline uint16_t Magnatude() const;
    };
    class uint16_3 final {
    public:
        union {
            struct { uint16_t x, y, z; };
            uint16_t v[3];
        };
        __host__ __device__ inline constexpr uint16_3();
        __host__ __device__ inline constexpr uint16_3(uint16_t x, uint16_t y, uint16_t z);
        __host__ __device__ inline constexpr uint16_3(const uint16_t v[3]);
        __host__ __device__ inline constexpr uint16_3 operator+(uint16_3 other) const;
        __host__ __device__ inline constexpr uint16_3 operator-(uint16_3 other) const;
        __host__ __device__ inline constexpr uint16_3 operator*(uint16_t other) const;
        __host__ __device__ inline constexpr uint16_3 operator/(uint16_t other) const;
        __host__ __device__ static inline constexpr uint16_t Dot(uint16_3 left, uint16_3 right);
        __host__ __device__ inline constexpr uint16_t MagnatudeSquared() const;
        __host__ __device__ inline uint16_t Magnatude() const;
    };
    class uint16_4 final {
    public:
        union {
            struct { uint16_t x, y, z, w; };
            uint16_t v[4];
        };
        __host__ __device__ inline constexpr uint16_4();
        __host__ __device__ inline constexpr uint16_4(uint16_t x, uint16_t y, uint16_t z, uint16_t w);
        __host__ __device__ inline constexpr uint16_4(const uint16_t v[4]);
        __host__ __device__ inline constexpr uint16_4 operator+(uint16_4 other) const;
        __host__ __device__ inline constexpr uint16_4 operator-(uint16_4 other) const;
        __host__ __device__ inline constexpr uint16_4 operator*(uint16_t other) const;
        __host__ __device__ inline constexpr uint16_4 operator/(uint16_t other) const;
        __host__ __device__ static inline constexpr uint16_t Dot(uint16_4 left, uint16_4 right);
        __host__ __device__ inline constexpr uint16_t MagnatudeSquared() const;
        __host__ __device__ inline uint16_t Magnatude() const;
    };
    class int32_2 final {
    public:
        union {
            struct { int32_t x, y; };
            int32_t v[2];
        };
        __host__ __device__ inline constexpr int32_2();
        __host__ __device__ inline constexpr int32_2(int32_t x, int32_t y);
        __host__ __device__ inline constexpr int32_2(const int32_t v[2]);
        __host__ __device__ inline constexpr int32_2 operator+(int32_2 other) const;
        __host__ __device__ inline constexpr int32_2 operator-(int32_2 other) const;
        __host__ __device__ inline constexpr int32_2 operator*(int32_t other) const;
        __host__ __device__ inline constexpr int32_2 operator/(int32_t other) const;
        __host__ __device__ static inline constexpr int32_t Dot(int32_2 left, int32_2 right);
        __host__ __device__ inline constexpr int32_t MagnatudeSquared() const;
        __host__ __device__ inline int32_t Magnatude() const;
    };
    class int32_3 final {
    public:
        union {
            struct { int32_t x, y, z; };
            int32_t v[3];
        };
        __host__ __device__ inline constexpr int32_3();
        __host__ __device__ inline constexpr int32_3(int32_t x, int32_t y, int32_t z);
        __host__ __device__ inline constexpr int32_3(const int32_t v[3]);
        __host__ __device__ inline constexpr int32_3 operator+(int32_3 other) const;
        __host__ __device__ inline constexpr int32_3 operator-(int32_3 other) const;
        __host__ __device__ inline constexpr int32_3 operator*(int32_t other) const;
        __host__ __device__ inline constexpr int32_3 operator/(int32_t other) const;
        __host__ __device__ static inline constexpr int32_t Dot(int32_3 left, int32_3 right);
        __host__ __device__ inline constexpr int32_t MagnatudeSquared() const;
        __host__ __device__ inline int32_t Magnatude() const;
    };
    class int32_4 final {
    public:
        union {
            struct { int32_t x, y, z, w; };
            int32_t v[4];
        };
        __host__ __device__ inline constexpr int32_4();
        __host__ __device__ inline constexpr int32_4(int32_t x, int32_t y, int32_t z, int32_t w);
        __host__ __device__ inline constexpr int32_4(const int32_t v[4]);
        __host__ __device__ inline constexpr int32_4 operator+(int32_4 other) const;
        __host__ __device__ inline constexpr int32_4 operator-(int32_4 other) const;
        __host__ __device__ inline constexpr int32_4 operator*(int32_t other) const;
        __host__ __device__ inline constexpr int32_4 operator/(int32_t other) const;
        __host__ __device__ static inline constexpr int32_t Dot(int32_4 left, int32_4 right);
        __host__ __device__ inline constexpr int32_t MagnatudeSquared() const;
        __host__ __device__ inline int32_t Magnatude() const;
    };
    class uint32_2 final {
    public:
        union {
            struct { uint32_t x, y; };
            uint32_t v[2];
        };
        __host__ __device__ inline constexpr uint32_2();
        __host__ __device__ inline constexpr uint32_2(uint32_t x, uint32_t y);
        __host__ __device__ inline constexpr uint32_2(const uint32_t v[2]);
        __host__ __device__ inline constexpr uint32_2 operator+(uint32_2 other) const;
        __host__ __device__ inline constexpr uint32_2 operator-(uint32_2 other) const;
        __host__ __device__ inline constexpr uint32_2 operator*(uint32_t other) const;
        __host__ __device__ inline constexpr uint32_2 operator/(uint32_t other) const;
        __host__ __device__ static inline constexpr uint32_t Dot(uint32_2 left, uint32_2 right);
        __host__ __device__ inline constexpr uint32_t MagnatudeSquared() const;
        __host__ __device__ inline uint32_t Magnatude() const;
    };
    class uint32_3 final {
    public:
        union {
            struct { uint32_t x, y, z; };
            uint32_t v[3];
        };
        __host__ __device__ inline constexpr uint32_3();
        __host__ __device__ inline constexpr uint32_3(uint32_t x, uint32_t y, uint32_t z);
        __host__ __device__ inline constexpr uint32_3(const uint32_t v[3]);
        __host__ __device__ inline constexpr uint32_3 operator+(uint32_3 other) const;
        __host__ __device__ inline constexpr uint32_3 operator-(uint32_3 other) const;
        __host__ __device__ inline constexpr uint32_3 operator*(uint32_t other) const;
        __host__ __device__ inline constexpr uint32_3 operator/(uint32_t other) const;
        __host__ __device__ static inline constexpr uint32_t Dot(uint32_3 left, uint32_3 right);
        __host__ __device__ inline constexpr uint32_t MagnatudeSquared() const;
        __host__ __device__ inline uint32_t Magnatude() const;
    };
    class uint32_4 final {
    public:
        union {
            struct { uint32_t x, y, z, w; };
            uint32_t v[4];
        };
        __host__ __device__ inline constexpr uint32_4();
        __host__ __device__ inline constexpr uint32_4(uint32_t x, uint32_t y, uint32_t z, uint32_t w);
        __host__ __device__ inline constexpr uint32_4(const uint32_t v[4]);
        __host__ __device__ inline constexpr uint32_4 operator+(uint32_4 other) const;
        __host__ __device__ inline constexpr uint32_4 operator-(uint32_4 other) const;
        __host__ __device__ inline constexpr uint32_4 operator*(uint32_t other) const;
        __host__ __device__ inline constexpr uint32_4 operator/(uint32_t other) const;
        __host__ __device__ static inline constexpr uint32_t Dot(uint32_4 left, uint32_4 right);
        __host__ __device__ inline constexpr uint32_t MagnatudeSquared() const;
        __host__ __device__ inline uint32_t Magnatude() const;
    };
    class int64_2 final {
    public:
        union {
            struct { int64_t x, y; };
            int64_t v[2];
        };
        __host__ __device__ inline constexpr int64_2();
        __host__ __device__ inline constexpr int64_2(int64_t x, int64_t y);
        __host__ __device__ inline constexpr int64_2(const int64_t v[2]);
        __host__ __device__ inline constexpr int64_2 operator+(int64_2 other) const;
        __host__ __device__ inline constexpr int64_2 operator-(int64_2 other) const;
        __host__ __device__ inline constexpr int64_2 operator*(int64_t other) const;
        __host__ __device__ inline constexpr int64_2 operator/(int64_t other) const;
        __host__ __device__ static inline constexpr int64_t Dot(int64_2 left, int64_2 right);
        __host__ __device__ inline constexpr int64_t MagnatudeSquared() const;
        __host__ __device__ inline int64_t Magnatude() const;
    };
    class int64_3 final {
    public:
        union {
            struct { int64_t x, y, z; };
            int64_t v[3];
        };
        __host__ __device__ inline constexpr int64_3();
        __host__ __device__ inline constexpr int64_3(int64_t x, int64_t y, int64_t z);
        __host__ __device__ inline constexpr int64_3(const int64_t v[3]);
        __host__ __device__ inline constexpr int64_3 operator+(int64_3 other) const;
        __host__ __device__ inline constexpr int64_3 operator-(int64_3 other) const;
        __host__ __device__ inline constexpr int64_3 operator*(int64_t other) const;
        __host__ __device__ inline constexpr int64_3 operator/(int64_t other) const;
        __host__ __device__ static inline constexpr int64_t Dot(int64_3 left, int64_3 right);
        __host__ __device__ inline constexpr int64_t MagnatudeSquared() const;
        __host__ __device__ inline int64_t Magnatude() const;
    };
    class int64_4 final {
    public:
        union {
            struct { int64_t x, y, z, w; };
            int64_t v[4];
        };
        __host__ __device__ inline constexpr int64_4();
        __host__ __device__ inline constexpr int64_4(int64_t x, int64_t y, int64_t z, int64_t w);
        __host__ __device__ inline constexpr int64_4(const int64_t v[4]);
        __host__ __device__ inline constexpr int64_4 operator+(int64_4 other) const;
        __host__ __device__ inline constexpr int64_4 operator-(int64_4 other) const;
        __host__ __device__ inline constexpr int64_4 operator*(int64_t other) const;
        __host__ __device__ inline constexpr int64_4 operator/(int64_t other) const;
        __host__ __device__ static inline constexpr int64_t Dot(int64_4 left, int64_4 right);
        __host__ __device__ inline constexpr int64_t MagnatudeSquared() const;
        __host__ __device__ inline int64_t Magnatude() const;
    };
    class uint64_2 final {
    public:
        union {
            struct { uint64_t x, y; };
            uint64_t v[2];
        };
        __host__ __device__ inline constexpr uint64_2();
        __host__ __device__ inline constexpr uint64_2(uint64_t x, uint64_t y);
        __host__ __device__ inline constexpr uint64_2(const uint64_t v[2]);
        __host__ __device__ inline constexpr uint64_2 operator+(uint64_2 other) const;
        __host__ __device__ inline constexpr uint64_2 operator-(uint64_2 other) const;
        __host__ __device__ inline constexpr uint64_2 operator*(uint64_t other) const;
        __host__ __device__ inline constexpr uint64_2 operator/(uint64_t other) const;
        __host__ __device__ static inline constexpr uint64_t Dot(uint64_2 left, uint64_2 right);
        __host__ __device__ inline constexpr uint64_t MagnatudeSquared() const;
        __host__ __device__ inline uint64_t Magnatude() const;
    };
    class uint64_3 final {
    public:
        union {
            struct { uint64_t x, y, z; };
            uint64_t v[3];
        };
        __host__ __device__ inline constexpr uint64_3();
        __host__ __device__ inline constexpr uint64_3(uint64_t x, uint64_t y, uint64_t z);
        __host__ __device__ inline constexpr uint64_3(const uint64_t v[3]);
        __host__ __device__ inline constexpr uint64_3 operator+(uint64_3 other) const;
        __host__ __device__ inline constexpr uint64_3 operator-(uint64_3 other) const;
        __host__ __device__ inline constexpr uint64_3 operator*(uint64_t other) const;
        __host__ __device__ inline constexpr uint64_3 operator/(uint64_t other) const;
        __host__ __device__ static inline constexpr uint64_t Dot(uint64_3 left, uint64_3 right);
        __host__ __device__ inline constexpr uint64_t MagnatudeSquared() const;
        __host__ __device__ inline uint64_t Magnatude() const;
    };
    class uint64_4 final {
    public:
        union {
            struct { uint64_t x, y, z, w; };
            uint64_t v[4];
        };
        __host__ __device__ inline constexpr uint64_4();
        __host__ __device__ inline constexpr uint64_4(uint64_t x, uint64_t y, uint64_t z, uint64_t w);
        __host__ __device__ inline constexpr uint64_4(const uint64_t v[4]);
        __host__ __device__ inline constexpr uint64_4 operator+(uint64_4 other) const;
        __host__ __device__ inline constexpr uint64_4 operator-(uint64_4 other) const;
        __host__ __device__ inline constexpr uint64_4 operator*(uint64_t other) const;
        __host__ __device__ inline constexpr uint64_4 operator/(uint64_t other) const;
        __host__ __device__ static inline constexpr uint64_t Dot(uint64_4 left, uint64_4 right);
        __host__ __device__ inline constexpr uint64_t MagnatudeSquared() const;
        __host__ __device__ inline uint64_t Magnatude() const;
    };
}

__host__ __device__ inline constexpr BrendanCUDA::float_2::float_2() : x(0), y(0) { }
__host__ __device__ inline constexpr BrendanCUDA::float_2::float_2(const float x, const float y) : x(x), y(y) { }
__host__ __device__ inline constexpr BrendanCUDA::float_2::float_2(const float v[2]) : v{ v[0], v[1] } { }
__host__ __device__ inline constexpr BrendanCUDA::float_2 BrendanCUDA::float_2::operator+(float_2 other) const {
    return float_2(x + other.x, y + other.y);
}
__host__ __device__ inline constexpr BrendanCUDA::float_2 BrendanCUDA::float_2::operator-(float_2 other) const {
    return float_2(x - other.x, y - other.y);
}
__host__ __device__ inline constexpr BrendanCUDA::float_2 BrendanCUDA::float_2::operator*(float other) const {
    return float_2(x * other, y * other);
}
__host__ __device__ inline constexpr BrendanCUDA::float_2 BrendanCUDA::float_2::operator/(float other) const {
    return float_2(x / other, y / other);
}
__host__ __device__ inline constexpr float BrendanCUDA::float_2::Dot(float_2 left, float_2 right) {
    return left.x * right.x + left.y * right.y;
}
__host__ __device__ inline constexpr float BrendanCUDA::float_2::MagnatudeSquared() const {
    return x * x + y * y;
}
__host__ __device__ inline float BrendanCUDA::float_2::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ inline constexpr BrendanCUDA::float_3::float_3() : x(0), y(0), z(0) { }
__host__ __device__ inline constexpr BrendanCUDA::float_3::float_3(const float x, const float y, const float z) : x(x), y(y), z(z) { }
__host__ __device__ inline constexpr BrendanCUDA::float_3::float_3(const float v[3]) : v{ v[0], v[1], v[2] } { }
__host__ __device__ inline constexpr BrendanCUDA::float_3 BrendanCUDA::float_3::operator+(float_3 other) const {
    return float_3(x + other.x, y + other.y, z + other.z);
}
__host__ __device__ inline constexpr BrendanCUDA::float_3 BrendanCUDA::float_3::operator-(float_3 other) const {
    return float_3(x - other.x, y - other.y, z - other.z);
}
__host__ __device__ inline constexpr BrendanCUDA::float_3 BrendanCUDA::float_3::operator*(float other) const {
    return float_3(x * other, y * other, z * other);
}
__host__ __device__ inline constexpr BrendanCUDA::float_3 BrendanCUDA::float_3::operator/(float other) const {
    return float_3(x / other, y / other, z / other);
}
__host__ __device__ inline constexpr float BrendanCUDA::float_3::Dot(float_3 left, float_3 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}
__host__ __device__ inline constexpr float BrendanCUDA::float_3::MagnatudeSquared() const {
    return x * x + y * y + z * z;
}
__host__ __device__ inline float BrendanCUDA::float_3::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ inline constexpr BrendanCUDA::float_4::float_4() : x(0), y(0), z(0), w(0) { }
__host__ __device__ inline constexpr BrendanCUDA::float_4::float_4(const float x, const float y, const float z, const float w) : x(x), y(y), z(z), w(w) { }
__host__ __device__ inline constexpr BrendanCUDA::float_4::float_4(const float v[4]) : v{ v[0], v[1], v[2], v[3] } { }
__host__ __device__ inline constexpr BrendanCUDA::float_4 BrendanCUDA::float_4::operator+(float_4 other) const {
    return float_4(x + other.x, y + other.y, z + other.z, w + other.w);
}
__host__ __device__ inline constexpr BrendanCUDA::float_4 BrendanCUDA::float_4::operator-(float_4 other) const {
    return float_4(x - other.x, y - other.y, z - other.z, w - other.w);
}
__host__ __device__ inline constexpr BrendanCUDA::float_4 BrendanCUDA::float_4::operator*(float other) const {
    return float_4(x * other, y * other, z * other, w * other);
}
__host__ __device__ inline constexpr BrendanCUDA::float_4 BrendanCUDA::float_4::operator/(float other) const {
    return float_4(x / other, y / other, z / other, w / other);
}
__host__ __device__ inline constexpr float BrendanCUDA::float_4::Dot(float_4 left, float_4 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w;
}
__host__ __device__ inline constexpr float BrendanCUDA::float_4::MagnatudeSquared() const {
    return x * x + y * y + z * z + w * w;
}
__host__ __device__ inline float BrendanCUDA::float_4::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ inline constexpr BrendanCUDA::double_2::double_2() : x(0), y(0) { }
__host__ __device__ inline constexpr BrendanCUDA::double_2::double_2(const double x, const double y) : x(x), y(y) { }
__host__ __device__ inline constexpr BrendanCUDA::double_2::double_2(const double v[2]) : v{ v[0], v[1] } { }
__host__ __device__ inline constexpr BrendanCUDA::double_2 BrendanCUDA::double_2::operator+(double_2 other) const {
    return double_2(x + other.x, y + other.y);
}
__host__ __device__ inline constexpr BrendanCUDA::double_2 BrendanCUDA::double_2::operator-(double_2 other) const {
    return double_2(x - other.x, y - other.y);
}
__host__ __device__ inline constexpr BrendanCUDA::double_2 BrendanCUDA::double_2::operator*(double other) const {
    return double_2(x * other, y * other);
}
__host__ __device__ inline constexpr BrendanCUDA::double_2 BrendanCUDA::double_2::operator/(double other) const {
    return double_2(x / other, y / other);
}
__host__ __device__ inline constexpr double BrendanCUDA::double_2::Dot(double_2 left, double_2 right) {
    return left.x * right.x + left.y * right.y;
}
__host__ __device__ inline constexpr double BrendanCUDA::double_2::MagnatudeSquared() const {
    return x * x + y * y;
}
__host__ __device__ inline double BrendanCUDA::double_2::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ inline constexpr BrendanCUDA::double_3::double_3() : x(0), y(0), z(0) { }
__host__ __device__ inline constexpr BrendanCUDA::double_3::double_3(const double x, const double y, const double z) : x(x), y(y), z(z) { }
__host__ __device__ inline constexpr BrendanCUDA::double_3::double_3(const double v[3]) : v{ v[0], v[1], v[2] } { }
__host__ __device__ inline constexpr BrendanCUDA::double_3 BrendanCUDA::double_3::operator+(double_3 other) const {
    return double_3(x + other.x, y + other.y, z + other.z);
}
__host__ __device__ inline constexpr BrendanCUDA::double_3 BrendanCUDA::double_3::operator-(double_3 other) const {
    return double_3(x - other.x, y - other.y, z - other.z);
}
__host__ __device__ inline constexpr BrendanCUDA::double_3 BrendanCUDA::double_3::operator*(double other) const {
    return double_3(x * other, y * other, z * other);
}
__host__ __device__ inline constexpr BrendanCUDA::double_3 BrendanCUDA::double_3::operator/(double other) const {
    return double_3(x / other, y / other, z / other);
}
__host__ __device__ inline constexpr double BrendanCUDA::double_3::Dot(double_3 left, double_3 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}
__host__ __device__ inline constexpr double BrendanCUDA::double_3::MagnatudeSquared() const {
    return x * x + y * y + z * z;
}
__host__ __device__ inline double BrendanCUDA::double_3::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ inline constexpr BrendanCUDA::double_4::double_4() : x(0), y(0), z(0), w(0) { }
__host__ __device__ inline constexpr BrendanCUDA::double_4::double_4(const double x, const double y, const double z, const double w) : x(x), y(y), z(z), w(w) { }
__host__ __device__ inline constexpr BrendanCUDA::double_4::double_4(const double v[4]) : v{ v[0], v[1], v[2], v[3] } { }
__host__ __device__ inline constexpr BrendanCUDA::double_4 BrendanCUDA::double_4::operator+(double_4 other) const {
    return double_4(x + other.x, y + other.y, z + other.z, w + other.w);
}
__host__ __device__ inline constexpr BrendanCUDA::double_4 BrendanCUDA::double_4::operator-(double_4 other) const {
    return double_4(x - other.x, y - other.y, z - other.z, w - other.w);
}
__host__ __device__ inline constexpr BrendanCUDA::double_4 BrendanCUDA::double_4::operator*(double other) const {
    return double_4(x * other, y * other, z * other, w * other);
}
__host__ __device__ inline constexpr BrendanCUDA::double_4 BrendanCUDA::double_4::operator/(double other) const {
    return double_4(x / other, y / other, z / other, w / other);
}
__host__ __device__ inline constexpr double BrendanCUDA::double_4::Dot(double_4 left, double_4 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w;
}
__host__ __device__ inline constexpr double BrendanCUDA::double_4::MagnatudeSquared() const {
    return x * x + y * y + z * z + w * w;
}
__host__ __device__ inline double BrendanCUDA::double_4::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ inline constexpr BrendanCUDA::int8_2::int8_2() : x(0), y(0) { }
__host__ __device__ inline constexpr BrendanCUDA::int8_2::int8_2(const int8_t x, const int8_t y) : x(x), y(y) { }
__host__ __device__ inline constexpr BrendanCUDA::int8_2::int8_2(const int8_t v[2]) : v{ v[0], v[1] } { }
__host__ __device__ inline constexpr BrendanCUDA::int8_2 BrendanCUDA::int8_2::operator+(int8_2 other) const {
    return int8_2(x + other.x, y + other.y);
}
__host__ __device__ inline constexpr BrendanCUDA::int8_2 BrendanCUDA::int8_2::operator-(int8_2 other) const {
    return int8_2(x - other.x, y - other.y);
}
__host__ __device__ inline constexpr BrendanCUDA::int8_2 BrendanCUDA::int8_2::operator*(int8_t other) const {
    return int8_2(x * other, y * other);
}
__host__ __device__ inline constexpr BrendanCUDA::int8_2 BrendanCUDA::int8_2::operator/(int8_t other) const {
    return int8_2(x / other, y / other);
}
__host__ __device__ inline constexpr int8_t BrendanCUDA::int8_2::Dot(int8_2 left, int8_2 right) {
    return left.x * right.x + left.y * right.y;
}
__host__ __device__ inline constexpr int8_t BrendanCUDA::int8_2::MagnatudeSquared() const {
    return x * x + y * y;
}
__host__ __device__ inline int8_t BrendanCUDA::int8_2::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ inline constexpr BrendanCUDA::int8_3::int8_3() : x(0), y(0), z(0) { }
__host__ __device__ inline constexpr BrendanCUDA::int8_3::int8_3(const int8_t x, const int8_t y, const int8_t z) : x(x), y(y), z(z) { }
__host__ __device__ inline constexpr BrendanCUDA::int8_3::int8_3(const int8_t v[3]) : v{ v[0], v[1], v[2] } { }
__host__ __device__ inline constexpr BrendanCUDA::int8_3 BrendanCUDA::int8_3::operator+(int8_3 other) const {
    return int8_3(x + other.x, y + other.y, z + other.z);
}
__host__ __device__ inline constexpr BrendanCUDA::int8_3 BrendanCUDA::int8_3::operator-(int8_3 other) const {
    return int8_3(x - other.x, y - other.y, z - other.z);
}
__host__ __device__ inline constexpr BrendanCUDA::int8_3 BrendanCUDA::int8_3::operator*(int8_t other) const {
    return int8_3(x * other, y * other, z * other);
}
__host__ __device__ inline constexpr BrendanCUDA::int8_3 BrendanCUDA::int8_3::operator/(int8_t other) const {
    return int8_3(x / other, y / other, z / other);
}
__host__ __device__ inline constexpr int8_t BrendanCUDA::int8_3::Dot(int8_3 left, int8_3 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}
__host__ __device__ inline constexpr int8_t BrendanCUDA::int8_3::MagnatudeSquared() const {
    return x * x + y * y + z * z;
}
__host__ __device__ inline int8_t BrendanCUDA::int8_3::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ inline constexpr BrendanCUDA::int8_4::int8_4() : x(0), y(0), z(0), w(0) { }
__host__ __device__ inline constexpr BrendanCUDA::int8_4::int8_4(const int8_t x, const int8_t y, const int8_t z, const int8_t w) : x(x), y(y), z(z), w(w) { }
__host__ __device__ inline constexpr BrendanCUDA::int8_4::int8_4(const int8_t v[4]) : v{ v[0], v[1], v[2], v[3] } { }
__host__ __device__ inline constexpr BrendanCUDA::int8_4 BrendanCUDA::int8_4::operator+(int8_4 other) const {
    return int8_4(x + other.x, y + other.y, z + other.z, w + other.w);
}
__host__ __device__ inline constexpr BrendanCUDA::int8_4 BrendanCUDA::int8_4::operator-(int8_4 other) const {
    return int8_4(x - other.x, y - other.y, z - other.z, w - other.w);
}
__host__ __device__ inline constexpr BrendanCUDA::int8_4 BrendanCUDA::int8_4::operator*(int8_t other) const {
    return int8_4(x * other, y * other, z * other, w * other);
}
__host__ __device__ inline constexpr BrendanCUDA::int8_4 BrendanCUDA::int8_4::operator/(int8_t other) const {
    return int8_4(x / other, y / other, z / other, w / other);
}
__host__ __device__ inline constexpr int8_t BrendanCUDA::int8_4::Dot(int8_4 left, int8_4 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w;
}
__host__ __device__ inline constexpr int8_t BrendanCUDA::int8_4::MagnatudeSquared() const {
    return x * x + y * y + z * z + w * w;
}
__host__ __device__ inline int8_t BrendanCUDA::int8_4::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ inline constexpr BrendanCUDA::uint8_2::uint8_2() : x(0), y(0) { }
__host__ __device__ inline constexpr BrendanCUDA::uint8_2::uint8_2(const uint8_t x, const uint8_t y) : x(x), y(y) { }
__host__ __device__ inline constexpr BrendanCUDA::uint8_2::uint8_2(const uint8_t v[2]) : v{ v[0], v[1] } { }
__host__ __device__ inline constexpr BrendanCUDA::uint8_2 BrendanCUDA::uint8_2::operator+(uint8_2 other) const {
    return uint8_2(x + other.x, y + other.y);
}
__host__ __device__ inline constexpr BrendanCUDA::uint8_2 BrendanCUDA::uint8_2::operator-(uint8_2 other) const {
    return uint8_2(x - other.x, y - other.y);
}
__host__ __device__ inline constexpr BrendanCUDA::uint8_2 BrendanCUDA::uint8_2::operator*(uint8_t other) const {
    return uint8_2(x * other, y * other);
}
__host__ __device__ inline constexpr BrendanCUDA::uint8_2 BrendanCUDA::uint8_2::operator/(uint8_t other) const {
    return uint8_2(x / other, y / other);
}
__host__ __device__ inline constexpr uint8_t BrendanCUDA::uint8_2::Dot(uint8_2 left, uint8_2 right) {
    return left.x * right.x + left.y * right.y;
}
__host__ __device__ inline constexpr uint8_t BrendanCUDA::uint8_2::MagnatudeSquared() const {
    return x * x + y * y;
}
__host__ __device__ inline uint8_t BrendanCUDA::uint8_2::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ inline constexpr BrendanCUDA::uint8_3::uint8_3() : x(0), y(0), z(0) { }
__host__ __device__ inline constexpr BrendanCUDA::uint8_3::uint8_3(const uint8_t x, const uint8_t y, const uint8_t z) : x(x), y(y), z(z) { }
__host__ __device__ inline constexpr BrendanCUDA::uint8_3::uint8_3(const uint8_t v[3]) : v{ v[0], v[1], v[2] } { }
__host__ __device__ inline constexpr BrendanCUDA::uint8_3 BrendanCUDA::uint8_3::operator+(uint8_3 other) const {
    return uint8_3(x + other.x, y + other.y, z + other.z);
}
__host__ __device__ inline constexpr BrendanCUDA::uint8_3 BrendanCUDA::uint8_3::operator-(uint8_3 other) const {
    return uint8_3(x - other.x, y - other.y, z - other.z);
}
__host__ __device__ inline constexpr BrendanCUDA::uint8_3 BrendanCUDA::uint8_3::operator*(uint8_t other) const {
    return uint8_3(x * other, y * other, z * other);
}
__host__ __device__ inline constexpr BrendanCUDA::uint8_3 BrendanCUDA::uint8_3::operator/(uint8_t other) const {
    return uint8_3(x / other, y / other, z / other);
}
__host__ __device__ inline constexpr uint8_t BrendanCUDA::uint8_3::Dot(uint8_3 left, uint8_3 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}
__host__ __device__ inline constexpr uint8_t BrendanCUDA::uint8_3::MagnatudeSquared() const {
    return x * x + y * y + z * z;
}
__host__ __device__ inline uint8_t BrendanCUDA::uint8_3::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ inline constexpr BrendanCUDA::uint8_4::uint8_4() : x(0), y(0), z(0), w(0) { }
__host__ __device__ inline constexpr BrendanCUDA::uint8_4::uint8_4(const uint8_t x, const uint8_t y, const uint8_t z, const uint8_t w) : x(x), y(y), z(z), w(w) { }
__host__ __device__ inline constexpr BrendanCUDA::uint8_4::uint8_4(const uint8_t v[4]) : v{ v[0], v[1], v[2], v[3] } { }
__host__ __device__ inline constexpr BrendanCUDA::uint8_4 BrendanCUDA::uint8_4::operator+(uint8_4 other) const {
    return uint8_4(x + other.x, y + other.y, z + other.z, w + other.w);
}
__host__ __device__ inline constexpr BrendanCUDA::uint8_4 BrendanCUDA::uint8_4::operator-(uint8_4 other) const {
    return uint8_4(x - other.x, y - other.y, z - other.z, w - other.w);
}
__host__ __device__ inline constexpr BrendanCUDA::uint8_4 BrendanCUDA::uint8_4::operator*(uint8_t other) const {
    return uint8_4(x * other, y * other, z * other, w * other);
}
__host__ __device__ inline constexpr BrendanCUDA::uint8_4 BrendanCUDA::uint8_4::operator/(uint8_t other) const {
    return uint8_4(x / other, y / other, z / other, w / other);
}
__host__ __device__ inline constexpr uint8_t BrendanCUDA::uint8_4::Dot(uint8_4 left, uint8_4 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w;
}
__host__ __device__ inline constexpr uint8_t BrendanCUDA::uint8_4::MagnatudeSquared() const {
    return x * x + y * y + z * z + w * w;
}
__host__ __device__ inline uint8_t BrendanCUDA::uint8_4::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ inline constexpr BrendanCUDA::int16_2::int16_2() : x(0), y(0) { }
__host__ __device__ inline constexpr BrendanCUDA::int16_2::int16_2(const int16_t x, const int16_t y) : x(x), y(y) { }
__host__ __device__ inline constexpr BrendanCUDA::int16_2::int16_2(const int16_t v[2]) : v{ v[0], v[1] } { }
__host__ __device__ inline constexpr BrendanCUDA::int16_2 BrendanCUDA::int16_2::operator+(int16_2 other) const {
    return int16_2(x + other.x, y + other.y);
}
__host__ __device__ inline constexpr BrendanCUDA::int16_2 BrendanCUDA::int16_2::operator-(int16_2 other) const {
    return int16_2(x - other.x, y - other.y);
}
__host__ __device__ inline constexpr BrendanCUDA::int16_2 BrendanCUDA::int16_2::operator*(int16_t other) const {
    return int16_2(x * other, y * other);
}
__host__ __device__ inline constexpr BrendanCUDA::int16_2 BrendanCUDA::int16_2::operator/(int16_t other) const {
    return int16_2(x / other, y / other);
}
__host__ __device__ inline constexpr int16_t BrendanCUDA::int16_2::Dot(int16_2 left, int16_2 right) {
    return left.x * right.x + left.y * right.y;
}
__host__ __device__ inline constexpr int16_t BrendanCUDA::int16_2::MagnatudeSquared() const {
    return x * x + y * y;
}
__host__ __device__ inline int16_t BrendanCUDA::int16_2::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ inline constexpr BrendanCUDA::int16_3::int16_3() : x(0), y(0), z(0) { }
__host__ __device__ inline constexpr BrendanCUDA::int16_3::int16_3(const int16_t x, const int16_t y, const int16_t z) : x(x), y(y), z(z) { }
__host__ __device__ inline constexpr BrendanCUDA::int16_3::int16_3(const int16_t v[3]) : v{ v[0], v[1], v[2] } { }
__host__ __device__ inline constexpr BrendanCUDA::int16_3 BrendanCUDA::int16_3::operator+(int16_3 other) const {
    return int16_3(x + other.x, y + other.y, z + other.z);
}
__host__ __device__ inline constexpr BrendanCUDA::int16_3 BrendanCUDA::int16_3::operator-(int16_3 other) const {
    return int16_3(x - other.x, y - other.y, z - other.z);
}
__host__ __device__ inline constexpr BrendanCUDA::int16_3 BrendanCUDA::int16_3::operator*(int16_t other) const {
    return int16_3(x * other, y * other, z * other);
}
__host__ __device__ inline constexpr BrendanCUDA::int16_3 BrendanCUDA::int16_3::operator/(int16_t other) const {
    return int16_3(x / other, y / other, z / other);
}
__host__ __device__ inline constexpr int16_t BrendanCUDA::int16_3::Dot(int16_3 left, int16_3 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}
__host__ __device__ inline constexpr int16_t BrendanCUDA::int16_3::MagnatudeSquared() const {
    return x * x + y * y + z * z;
}
__host__ __device__ inline int16_t BrendanCUDA::int16_3::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ inline constexpr BrendanCUDA::int16_4::int16_4() : x(0), y(0), z(0), w(0) { }
__host__ __device__ inline constexpr BrendanCUDA::int16_4::int16_4(const int16_t x, const int16_t y, const int16_t z, const int16_t w) : x(x), y(y), z(z), w(w) { }
__host__ __device__ inline constexpr BrendanCUDA::int16_4::int16_4(const int16_t v[4]) : v{ v[0], v[1], v[2], v[3] } { }
__host__ __device__ inline constexpr BrendanCUDA::int16_4 BrendanCUDA::int16_4::operator+(int16_4 other) const {
    return int16_4(x + other.x, y + other.y, z + other.z, w + other.w);
}
__host__ __device__ inline constexpr BrendanCUDA::int16_4 BrendanCUDA::int16_4::operator-(int16_4 other) const {
    return int16_4(x - other.x, y - other.y, z - other.z, w - other.w);
}
__host__ __device__ inline constexpr BrendanCUDA::int16_4 BrendanCUDA::int16_4::operator*(int16_t other) const {
    return int16_4(x * other, y * other, z * other, w * other);
}
__host__ __device__ inline constexpr BrendanCUDA::int16_4 BrendanCUDA::int16_4::operator/(int16_t other) const {
    return int16_4(x / other, y / other, z / other, w / other);
}
__host__ __device__ inline constexpr int16_t BrendanCUDA::int16_4::Dot(int16_4 left, int16_4 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w;
}
__host__ __device__ inline constexpr int16_t BrendanCUDA::int16_4::MagnatudeSquared() const {
    return x * x + y * y + z * z + w * w;
}
__host__ __device__ inline int16_t BrendanCUDA::int16_4::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ inline constexpr BrendanCUDA::uint16_2::uint16_2() : x(0), y(0) { }
__host__ __device__ inline constexpr BrendanCUDA::uint16_2::uint16_2(const uint16_t x, const uint16_t y) : x(x), y(y) { }
__host__ __device__ inline constexpr BrendanCUDA::uint16_2::uint16_2(const uint16_t v[2]) : v{ v[0], v[1] } { }
__host__ __device__ inline constexpr BrendanCUDA::uint16_2 BrendanCUDA::uint16_2::operator+(uint16_2 other) const {
    return uint16_2(x + other.x, y + other.y);
}
__host__ __device__ inline constexpr BrendanCUDA::uint16_2 BrendanCUDA::uint16_2::operator-(uint16_2 other) const {
    return uint16_2(x - other.x, y - other.y);
}
__host__ __device__ inline constexpr BrendanCUDA::uint16_2 BrendanCUDA::uint16_2::operator*(uint16_t other) const {
    return uint16_2(x * other, y * other);
}
__host__ __device__ inline constexpr BrendanCUDA::uint16_2 BrendanCUDA::uint16_2::operator/(uint16_t other) const {
    return uint16_2(x / other, y / other);
}
__host__ __device__ inline constexpr uint16_t BrendanCUDA::uint16_2::Dot(uint16_2 left, uint16_2 right) {
    return left.x * right.x + left.y * right.y;
}
__host__ __device__ inline constexpr uint16_t BrendanCUDA::uint16_2::MagnatudeSquared() const {
    return x * x + y * y;
}
__host__ __device__ inline uint16_t BrendanCUDA::uint16_2::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ inline constexpr BrendanCUDA::uint16_3::uint16_3() : x(0), y(0), z(0) { }
__host__ __device__ inline constexpr BrendanCUDA::uint16_3::uint16_3(const uint16_t x, const uint16_t y, const uint16_t z) : x(x), y(y), z(z) { }
__host__ __device__ inline constexpr BrendanCUDA::uint16_3::uint16_3(const uint16_t v[3]) : v{ v[0], v[1], v[2] } { }
__host__ __device__ inline constexpr BrendanCUDA::uint16_3 BrendanCUDA::uint16_3::operator+(uint16_3 other) const {
    return uint16_3(x + other.x, y + other.y, z + other.z);
}
__host__ __device__ inline constexpr BrendanCUDA::uint16_3 BrendanCUDA::uint16_3::operator-(uint16_3 other) const {
    return uint16_3(x - other.x, y - other.y, z - other.z);
}
__host__ __device__ inline constexpr BrendanCUDA::uint16_3 BrendanCUDA::uint16_3::operator*(uint16_t other) const {
    return uint16_3(x * other, y * other, z * other);
}
__host__ __device__ inline constexpr BrendanCUDA::uint16_3 BrendanCUDA::uint16_3::operator/(uint16_t other) const {
    return uint16_3(x / other, y / other, z / other);
}
__host__ __device__ inline constexpr uint16_t BrendanCUDA::uint16_3::Dot(uint16_3 left, uint16_3 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}
__host__ __device__ inline constexpr uint16_t BrendanCUDA::uint16_3::MagnatudeSquared() const {
    return x * x + y * y + z * z;
}
__host__ __device__ inline uint16_t BrendanCUDA::uint16_3::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ inline constexpr BrendanCUDA::uint16_4::uint16_4() : x(0), y(0), z(0), w(0) { }
__host__ __device__ inline constexpr BrendanCUDA::uint16_4::uint16_4(const uint16_t x, const uint16_t y, const uint16_t z, const uint16_t w) : x(x), y(y), z(z), w(w) { }
__host__ __device__ inline constexpr BrendanCUDA::uint16_4::uint16_4(const uint16_t v[4]) : v{ v[0], v[1], v[2], v[3] } { }
__host__ __device__ inline constexpr BrendanCUDA::uint16_4 BrendanCUDA::uint16_4::operator+(uint16_4 other) const {
    return uint16_4(x + other.x, y + other.y, z + other.z, w + other.w);
}
__host__ __device__ inline constexpr BrendanCUDA::uint16_4 BrendanCUDA::uint16_4::operator-(uint16_4 other) const {
    return uint16_4(x - other.x, y - other.y, z - other.z, w - other.w);
}
__host__ __device__ inline constexpr BrendanCUDA::uint16_4 BrendanCUDA::uint16_4::operator*(uint16_t other) const {
    return uint16_4(x * other, y * other, z * other, w * other);
}
__host__ __device__ inline constexpr BrendanCUDA::uint16_4 BrendanCUDA::uint16_4::operator/(uint16_t other) const {
    return uint16_4(x / other, y / other, z / other, w / other);
}
__host__ __device__ inline constexpr uint16_t BrendanCUDA::uint16_4::Dot(uint16_4 left, uint16_4 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w;
}
__host__ __device__ inline constexpr uint16_t BrendanCUDA::uint16_4::MagnatudeSquared() const {
    return x * x + y * y + z * z + w * w;
}
__host__ __device__ inline uint16_t BrendanCUDA::uint16_4::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ inline constexpr BrendanCUDA::int32_2::int32_2() : x(0), y(0) { }
__host__ __device__ inline constexpr BrendanCUDA::int32_2::int32_2(const int32_t x, const int32_t y) : x(x), y(y) { }
__host__ __device__ inline constexpr BrendanCUDA::int32_2::int32_2(const int32_t v[2]) : v{ v[0], v[1] } { }
__host__ __device__ inline constexpr BrendanCUDA::int32_2 BrendanCUDA::int32_2::operator+(int32_2 other) const {
    return int32_2(x + other.x, y + other.y);
}
__host__ __device__ inline constexpr BrendanCUDA::int32_2 BrendanCUDA::int32_2::operator-(int32_2 other) const {
    return int32_2(x - other.x, y - other.y);
}
__host__ __device__ inline constexpr BrendanCUDA::int32_2 BrendanCUDA::int32_2::operator*(int32_t other) const {
    return int32_2(x * other, y * other);
}
__host__ __device__ inline constexpr BrendanCUDA::int32_2 BrendanCUDA::int32_2::operator/(int32_t other) const {
    return int32_2(x / other, y / other);
}
__host__ __device__ inline constexpr int32_t BrendanCUDA::int32_2::Dot(int32_2 left, int32_2 right) {
    return left.x * right.x + left.y * right.y;
}
__host__ __device__ inline constexpr int32_t BrendanCUDA::int32_2::MagnatudeSquared() const {
    return x * x + y * y;
}
__host__ __device__ inline int32_t BrendanCUDA::int32_2::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ inline constexpr BrendanCUDA::int32_3::int32_3() : x(0), y(0), z(0) { }
__host__ __device__ inline constexpr BrendanCUDA::int32_3::int32_3(const int32_t x, const int32_t y, const int32_t z) : x(x), y(y), z(z) { }
__host__ __device__ inline constexpr BrendanCUDA::int32_3::int32_3(const int32_t v[3]) : v{ v[0], v[1], v[2] } { }
__host__ __device__ inline constexpr BrendanCUDA::int32_3 BrendanCUDA::int32_3::operator+(int32_3 other) const {
    return int32_3(x + other.x, y + other.y, z + other.z);
}
__host__ __device__ inline constexpr BrendanCUDA::int32_3 BrendanCUDA::int32_3::operator-(int32_3 other) const {
    return int32_3(x - other.x, y - other.y, z - other.z);
}
__host__ __device__ inline constexpr BrendanCUDA::int32_3 BrendanCUDA::int32_3::operator*(int32_t other) const {
    return int32_3(x * other, y * other, z * other);
}
__host__ __device__ inline constexpr BrendanCUDA::int32_3 BrendanCUDA::int32_3::operator/(int32_t other) const {
    return int32_3(x / other, y / other, z / other);
}
__host__ __device__ inline constexpr int32_t BrendanCUDA::int32_3::Dot(int32_3 left, int32_3 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}
__host__ __device__ inline constexpr int32_t BrendanCUDA::int32_3::MagnatudeSquared() const {
    return x * x + y * y + z * z;
}
__host__ __device__ inline int32_t BrendanCUDA::int32_3::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ inline constexpr BrendanCUDA::int32_4::int32_4() : x(0), y(0), z(0), w(0) { }
__host__ __device__ inline constexpr BrendanCUDA::int32_4::int32_4(const int32_t x, const int32_t y, const int32_t z, const int32_t w) : x(x), y(y), z(z), w(w) { }
__host__ __device__ inline constexpr BrendanCUDA::int32_4::int32_4(const int32_t v[4]) : v{ v[0], v[1], v[2], v[3] } { }
__host__ __device__ inline constexpr BrendanCUDA::int32_4 BrendanCUDA::int32_4::operator+(int32_4 other) const {
    return int32_4(x + other.x, y + other.y, z + other.z, w + other.w);
}
__host__ __device__ inline constexpr BrendanCUDA::int32_4 BrendanCUDA::int32_4::operator-(int32_4 other) const {
    return int32_4(x - other.x, y - other.y, z - other.z, w - other.w);
}
__host__ __device__ inline constexpr BrendanCUDA::int32_4 BrendanCUDA::int32_4::operator*(int32_t other) const {
    return int32_4(x * other, y * other, z * other, w * other);
}
__host__ __device__ inline constexpr BrendanCUDA::int32_4 BrendanCUDA::int32_4::operator/(int32_t other) const {
    return int32_4(x / other, y / other, z / other, w / other);
}
__host__ __device__ inline constexpr int32_t BrendanCUDA::int32_4::Dot(int32_4 left, int32_4 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w;
}
__host__ __device__ inline constexpr int32_t BrendanCUDA::int32_4::MagnatudeSquared() const {
    return x * x + y * y + z * z + w * w;
}
__host__ __device__ inline int32_t BrendanCUDA::int32_4::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ inline constexpr BrendanCUDA::uint32_2::uint32_2() : x(0), y(0) { }
__host__ __device__ inline constexpr BrendanCUDA::uint32_2::uint32_2(const uint32_t x, const uint32_t y) : x(x), y(y) { }
__host__ __device__ inline constexpr BrendanCUDA::uint32_2::uint32_2(const uint32_t v[2]) : v{ v[0], v[1] } { }
__host__ __device__ inline constexpr BrendanCUDA::uint32_2 BrendanCUDA::uint32_2::operator+(uint32_2 other) const {
    return uint32_2(x + other.x, y + other.y);
}
__host__ __device__ inline constexpr BrendanCUDA::uint32_2 BrendanCUDA::uint32_2::operator-(uint32_2 other) const {
    return uint32_2(x - other.x, y - other.y);
}
__host__ __device__ inline constexpr BrendanCUDA::uint32_2 BrendanCUDA::uint32_2::operator*(uint32_t other) const {
    return uint32_2(x * other, y * other);
}
__host__ __device__ inline constexpr BrendanCUDA::uint32_2 BrendanCUDA::uint32_2::operator/(uint32_t other) const {
    return uint32_2(x / other, y / other);
}
__host__ __device__ inline constexpr uint32_t BrendanCUDA::uint32_2::Dot(uint32_2 left, uint32_2 right) {
    return left.x * right.x + left.y * right.y;
}
__host__ __device__ inline constexpr uint32_t BrendanCUDA::uint32_2::MagnatudeSquared() const {
    return x * x + y * y;
}
__host__ __device__ inline uint32_t BrendanCUDA::uint32_2::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ inline constexpr BrendanCUDA::uint32_3::uint32_3() : x(0), y(0), z(0) { }
__host__ __device__ inline constexpr BrendanCUDA::uint32_3::uint32_3(const uint32_t x, const uint32_t y, const uint32_t z) : x(x), y(y), z(z) { }
__host__ __device__ inline constexpr BrendanCUDA::uint32_3::uint32_3(const uint32_t v[3]) : v{ v[0], v[1], v[2] } { }
__host__ __device__ inline constexpr BrendanCUDA::uint32_3 BrendanCUDA::uint32_3::operator+(uint32_3 other) const {
    return uint32_3(x + other.x, y + other.y, z + other.z);
}
__host__ __device__ inline constexpr BrendanCUDA::uint32_3 BrendanCUDA::uint32_3::operator-(uint32_3 other) const {
    return uint32_3(x - other.x, y - other.y, z - other.z);
}
__host__ __device__ inline constexpr BrendanCUDA::uint32_3 BrendanCUDA::uint32_3::operator*(uint32_t other) const {
    return uint32_3(x * other, y * other, z * other);
}
__host__ __device__ inline constexpr BrendanCUDA::uint32_3 BrendanCUDA::uint32_3::operator/(uint32_t other) const {
    return uint32_3(x / other, y / other, z / other);
}
__host__ __device__ inline constexpr uint32_t BrendanCUDA::uint32_3::Dot(uint32_3 left, uint32_3 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}
__host__ __device__ inline constexpr uint32_t BrendanCUDA::uint32_3::MagnatudeSquared() const {
    return x * x + y * y + z * z;
}
__host__ __device__ inline uint32_t BrendanCUDA::uint32_3::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ inline constexpr BrendanCUDA::uint32_4::uint32_4() : x(0), y(0), z(0), w(0) { }
__host__ __device__ inline constexpr BrendanCUDA::uint32_4::uint32_4(const uint32_t x, const uint32_t y, const uint32_t z, const uint32_t w) : x(x), y(y), z(z), w(w) { }
__host__ __device__ inline constexpr BrendanCUDA::uint32_4::uint32_4(const uint32_t v[4]) : v{ v[0], v[1], v[2], v[3] } { }
__host__ __device__ inline constexpr BrendanCUDA::uint32_4 BrendanCUDA::uint32_4::operator+(uint32_4 other) const {
    return uint32_4(x + other.x, y + other.y, z + other.z, w + other.w);
}
__host__ __device__ inline constexpr BrendanCUDA::uint32_4 BrendanCUDA::uint32_4::operator-(uint32_4 other) const {
    return uint32_4(x - other.x, y - other.y, z - other.z, w - other.w);
}
__host__ __device__ inline constexpr BrendanCUDA::uint32_4 BrendanCUDA::uint32_4::operator*(uint32_t other) const {
    return uint32_4(x * other, y * other, z * other, w * other);
}
__host__ __device__ inline constexpr BrendanCUDA::uint32_4 BrendanCUDA::uint32_4::operator/(uint32_t other) const {
    return uint32_4(x / other, y / other, z / other, w / other);
}
__host__ __device__ inline constexpr uint32_t BrendanCUDA::uint32_4::Dot(uint32_4 left, uint32_4 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w;
}
__host__ __device__ inline constexpr uint32_t BrendanCUDA::uint32_4::MagnatudeSquared() const {
    return x * x + y * y + z * z + w * w;
}
__host__ __device__ inline uint32_t BrendanCUDA::uint32_4::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ inline constexpr BrendanCUDA::int64_2::int64_2() : x(0), y(0) { }
__host__ __device__ inline constexpr BrendanCUDA::int64_2::int64_2(const int64_t x, const int64_t y) : x(x), y(y) { }
__host__ __device__ inline constexpr BrendanCUDA::int64_2::int64_2(const int64_t v[2]) : v{ v[0], v[1] } { }
__host__ __device__ inline constexpr BrendanCUDA::int64_2 BrendanCUDA::int64_2::operator+(int64_2 other) const {
    return int64_2(x + other.x, y + other.y);
}
__host__ __device__ inline constexpr BrendanCUDA::int64_2 BrendanCUDA::int64_2::operator-(int64_2 other) const {
    return int64_2(x - other.x, y - other.y);
}
__host__ __device__ inline constexpr BrendanCUDA::int64_2 BrendanCUDA::int64_2::operator*(int64_t other) const {
    return int64_2(x * other, y * other);
}
__host__ __device__ inline constexpr BrendanCUDA::int64_2 BrendanCUDA::int64_2::operator/(int64_t other) const {
    return int64_2(x / other, y / other);
}
__host__ __device__ inline constexpr int64_t BrendanCUDA::int64_2::Dot(int64_2 left, int64_2 right) {
    return left.x * right.x + left.y * right.y;
}
__host__ __device__ inline constexpr int64_t BrendanCUDA::int64_2::MagnatudeSquared() const {
    return x * x + y * y;
}
__host__ __device__ inline int64_t BrendanCUDA::int64_2::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ inline constexpr BrendanCUDA::int64_3::int64_3() : x(0), y(0), z(0) { }
__host__ __device__ inline constexpr BrendanCUDA::int64_3::int64_3(const int64_t x, const int64_t y, const int64_t z) : x(x), y(y), z(z) { }
__host__ __device__ inline constexpr BrendanCUDA::int64_3::int64_3(const int64_t v[3]) : v{ v[0], v[1], v[2] } { }
__host__ __device__ inline constexpr BrendanCUDA::int64_3 BrendanCUDA::int64_3::operator+(int64_3 other) const {
    return int64_3(x + other.x, y + other.y, z + other.z);
}
__host__ __device__ inline constexpr BrendanCUDA::int64_3 BrendanCUDA::int64_3::operator-(int64_3 other) const {
    return int64_3(x - other.x, y - other.y, z - other.z);
}
__host__ __device__ inline constexpr BrendanCUDA::int64_3 BrendanCUDA::int64_3::operator*(int64_t other) const {
    return int64_3(x * other, y * other, z * other);
}
__host__ __device__ inline constexpr BrendanCUDA::int64_3 BrendanCUDA::int64_3::operator/(int64_t other) const {
    return int64_3(x / other, y / other, z / other);
}
__host__ __device__ inline constexpr int64_t BrendanCUDA::int64_3::Dot(int64_3 left, int64_3 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}
__host__ __device__ inline constexpr int64_t BrendanCUDA::int64_3::MagnatudeSquared() const {
    return x * x + y * y + z * z;
}
__host__ __device__ inline int64_t BrendanCUDA::int64_3::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ inline constexpr BrendanCUDA::int64_4::int64_4() : x(0), y(0), z(0), w(0) { }
__host__ __device__ inline constexpr BrendanCUDA::int64_4::int64_4(const int64_t x, const int64_t y, const int64_t z, const int64_t w) : x(x), y(y), z(z), w(w) { }
__host__ __device__ inline constexpr BrendanCUDA::int64_4::int64_4(const int64_t v[4]) : v{ v[0], v[1], v[2], v[3] } { }
__host__ __device__ inline constexpr BrendanCUDA::int64_4 BrendanCUDA::int64_4::operator+(int64_4 other) const {
    return int64_4(x + other.x, y + other.y, z + other.z, w + other.w);
}
__host__ __device__ inline constexpr BrendanCUDA::int64_4 BrendanCUDA::int64_4::operator-(int64_4 other) const {
    return int64_4(x - other.x, y - other.y, z - other.z, w - other.w);
}
__host__ __device__ inline constexpr BrendanCUDA::int64_4 BrendanCUDA::int64_4::operator*(int64_t other) const {
    return int64_4(x * other, y * other, z * other, w * other);
}
__host__ __device__ inline constexpr BrendanCUDA::int64_4 BrendanCUDA::int64_4::operator/(int64_t other) const {
    return int64_4(x / other, y / other, z / other, w / other);
}
__host__ __device__ inline constexpr int64_t BrendanCUDA::int64_4::Dot(int64_4 left, int64_4 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w;
}
__host__ __device__ inline constexpr int64_t BrendanCUDA::int64_4::MagnatudeSquared() const {
    return x * x + y * y + z * z + w * w;
}
__host__ __device__ inline int64_t BrendanCUDA::int64_4::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ inline constexpr BrendanCUDA::uint64_2::uint64_2() : x(0), y(0) { }
__host__ __device__ inline constexpr BrendanCUDA::uint64_2::uint64_2(const uint64_t x, const uint64_t y) : x(x), y(y) { }
__host__ __device__ inline constexpr BrendanCUDA::uint64_2::uint64_2(const uint64_t v[2]) : v{ v[0], v[1] } { }
__host__ __device__ inline constexpr BrendanCUDA::uint64_2 BrendanCUDA::uint64_2::operator+(uint64_2 other) const {
    return uint64_2(x + other.x, y + other.y);
}
__host__ __device__ inline constexpr BrendanCUDA::uint64_2 BrendanCUDA::uint64_2::operator-(uint64_2 other) const {
    return uint64_2(x - other.x, y - other.y);
}
__host__ __device__ inline constexpr BrendanCUDA::uint64_2 BrendanCUDA::uint64_2::operator*(uint64_t other) const {
    return uint64_2(x * other, y * other);
}
__host__ __device__ inline constexpr BrendanCUDA::uint64_2 BrendanCUDA::uint64_2::operator/(uint64_t other) const {
    return uint64_2(x / other, y / other);
}
__host__ __device__ inline constexpr uint64_t BrendanCUDA::uint64_2::Dot(uint64_2 left, uint64_2 right) {
    return left.x * right.x + left.y * right.y;
}
__host__ __device__ inline constexpr uint64_t BrendanCUDA::uint64_2::MagnatudeSquared() const {
    return x * x + y * y;
}
__host__ __device__ inline uint64_t BrendanCUDA::uint64_2::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ inline constexpr BrendanCUDA::uint64_3::uint64_3() : x(0), y(0), z(0) { }
__host__ __device__ inline constexpr BrendanCUDA::uint64_3::uint64_3(const uint64_t x, const uint64_t y, const uint64_t z) : x(x), y(y), z(z) { }
__host__ __device__ inline constexpr BrendanCUDA::uint64_3::uint64_3(const uint64_t v[3]) : v{ v[0], v[1], v[2] } { }
__host__ __device__ inline constexpr BrendanCUDA::uint64_3 BrendanCUDA::uint64_3::operator+(uint64_3 other) const {
    return uint64_3(x + other.x, y + other.y, z + other.z);
}
__host__ __device__ inline constexpr BrendanCUDA::uint64_3 BrendanCUDA::uint64_3::operator-(uint64_3 other) const {
    return uint64_3(x - other.x, y - other.y, z - other.z);
}
__host__ __device__ inline constexpr BrendanCUDA::uint64_3 BrendanCUDA::uint64_3::operator*(uint64_t other) const {
    return uint64_3(x * other, y * other, z * other);
}
__host__ __device__ inline constexpr BrendanCUDA::uint64_3 BrendanCUDA::uint64_3::operator/(uint64_t other) const {
    return uint64_3(x / other, y / other, z / other);
}
__host__ __device__ inline constexpr uint64_t BrendanCUDA::uint64_3::Dot(uint64_3 left, uint64_3 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}
__host__ __device__ inline constexpr uint64_t BrendanCUDA::uint64_3::MagnatudeSquared() const {
    return x * x + y * y + z * z;
}
__host__ __device__ inline uint64_t BrendanCUDA::uint64_3::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ inline constexpr BrendanCUDA::uint64_4::uint64_4() : x(0), y(0), z(0), w(0) { }
__host__ __device__ inline constexpr BrendanCUDA::uint64_4::uint64_4(const uint64_t x, const uint64_t y, const uint64_t z, const uint64_t w) : x(x), y(y), z(z), w(w) { }
__host__ __device__ inline constexpr BrendanCUDA::uint64_4::uint64_4(const uint64_t v[4]) : v{ v[0], v[1], v[2], v[3] } { }
__host__ __device__ inline constexpr BrendanCUDA::uint64_4 BrendanCUDA::uint64_4::operator+(uint64_4 other) const {
    return uint64_4(x + other.x, y + other.y, z + other.z, w + other.w);
}
__host__ __device__ inline constexpr BrendanCUDA::uint64_4 BrendanCUDA::uint64_4::operator-(uint64_4 other) const {
    return uint64_4(x - other.x, y - other.y, z - other.z, w - other.w);
}
__host__ __device__ inline constexpr BrendanCUDA::uint64_4 BrendanCUDA::uint64_4::operator*(uint64_t other) const {
    return uint64_4(x * other, y * other, z * other, w * other);
}
__host__ __device__ inline constexpr BrendanCUDA::uint64_4 BrendanCUDA::uint64_4::operator/(uint64_t other) const {
    return uint64_4(x / other, y / other, z / other, w / other);
}
__host__ __device__ inline constexpr uint64_t BrendanCUDA::uint64_4::Dot(uint64_4 left, uint64_4 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w;
}
__host__ __device__ inline constexpr uint64_t BrendanCUDA::uint64_4::MagnatudeSquared() const {
    return x * x + y * y + z * z + w * w;
}
__host__ __device__ inline uint64_t BrendanCUDA::uint64_4::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
