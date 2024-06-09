#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace BrendanCUDA {
    class float_2 final {
    public:
        union {
            struct { float x, y; };
            float v[2];
        };
        __host__ __device__ float_2(float x, float y);
        __host__ __device__ float_2(float v[2]);
        __host__ __device__ float_2 operator+(float_2 other);
        __host__ __device__ float_2 operator-(float_2 other);
        __host__ __device__ float_2 operator*(float other);
        __host__ __device__ float_2 operator/(float other);
        __host__ __device__ static float Dot(float_2 left, float_2 right);
        __host__ __device__ float MagnatudeSquared() const;
        __host__ __device__ float Magnatude() const;
    };
    class float_3 final {
    public:
        union {
            struct { float x, y, z; };
            float v[3];
        };
        __host__ __device__ float_3(float x, float y, float z);
        __host__ __device__ float_3(float v[3]);
        __host__ __device__ float_3 operator+(float_3 other);
        __host__ __device__ float_3 operator-(float_3 other);
        __host__ __device__ float_3 operator*(float other);
        __host__ __device__ float_3 operator/(float other);
        __host__ __device__ static float Dot(float_3 left, float_3 right);
        __host__ __device__ float MagnatudeSquared() const;
        __host__ __device__ float Magnatude() const;
    };
    class float_4 final {
    public:
        union {
            struct { float x, y, z, w; };
            float v[4];
        };
        __host__ __device__ float_4(float x, float y, float z, float w);
        __host__ __device__ float_4(float v[4]);
        __host__ __device__ float_4 operator+(float_4 other);
        __host__ __device__ float_4 operator-(float_4 other);
        __host__ __device__ float_4 operator*(float other);
        __host__ __device__ float_4 operator/(float other);
        __host__ __device__ static float Dot(float_4 left, float_4 right);
        __host__ __device__ float MagnatudeSquared() const;
        __host__ __device__ float Magnatude() const;
    };
    class double_2 final {
    public:
        union {
            struct { double x, y; };
            double v[2];
        };
        __host__ __device__ double_2(double x, double y);
        __host__ __device__ double_2(double v[2]);
        __host__ __device__ double_2 operator+(double_2 other);
        __host__ __device__ double_2 operator-(double_2 other);
        __host__ __device__ double_2 operator*(double other);
        __host__ __device__ double_2 operator/(double other);
        __host__ __device__ static double Dot(double_2 left, double_2 right);
        __host__ __device__ double MagnatudeSquared() const;
        __host__ __device__ double Magnatude() const;
    };
    class double_3 final {
    public:
        union {
            struct { double x, y, z; };
            double v[3];
        };
        __host__ __device__ double_3(double x, double y, double z);
        __host__ __device__ double_3(double v[3]);
        __host__ __device__ double_3 operator+(double_3 other);
        __host__ __device__ double_3 operator-(double_3 other);
        __host__ __device__ double_3 operator*(double other);
        __host__ __device__ double_3 operator/(double other);
        __host__ __device__ static double Dot(double_3 left, double_3 right);
        __host__ __device__ double MagnatudeSquared() const;
        __host__ __device__ double Magnatude() const;
    };
    class double_4 final {
    public:
        union {
            struct { double x, y, z, w; };
            double v[4];
        };
        __host__ __device__ double_4(double x, double y, double z, double w);
        __host__ __device__ double_4(double v[4]);
        __host__ __device__ double_4 operator+(double_4 other);
        __host__ __device__ double_4 operator-(double_4 other);
        __host__ __device__ double_4 operator*(double other);
        __host__ __device__ double_4 operator/(double other);
        __host__ __device__ static double Dot(double_4 left, double_4 right);
        __host__ __device__ double MagnatudeSquared() const;
        __host__ __device__ double Magnatude() const;
    };
    class int8_2 final {
    public:
        union {
            struct { int8_t x, y; };
            int8_t v[2];
        };
        __host__ __device__ int8_2(int8_t x, int8_t y);
        __host__ __device__ int8_2(int8_t v[2]);
        __host__ __device__ int8_2 operator+(int8_2 other);
        __host__ __device__ int8_2 operator-(int8_2 other);
        __host__ __device__ int8_2 operator*(int8_t other);
        __host__ __device__ int8_2 operator/(int8_t other);
        __host__ __device__ static int8_t Dot(int8_2 left, int8_2 right);
        __host__ __device__ int8_t MagnatudeSquared() const;
        __host__ __device__ int8_t Magnatude() const;
    };
    class int8_3 final {
    public:
        union {
            struct { int8_t x, y, z; };
            int8_t v[3];
        };
        __host__ __device__ int8_3(int8_t x, int8_t y, int8_t z);
        __host__ __device__ int8_3(int8_t v[3]);
        __host__ __device__ int8_3 operator+(int8_3 other);
        __host__ __device__ int8_3 operator-(int8_3 other);
        __host__ __device__ int8_3 operator*(int8_t other);
        __host__ __device__ int8_3 operator/(int8_t other);
        __host__ __device__ static int8_t Dot(int8_3 left, int8_3 right);
        __host__ __device__ int8_t MagnatudeSquared() const;
        __host__ __device__ int8_t Magnatude() const;
    };
    class int8_4 final {
    public:
        union {
            struct { int8_t x, y, z, w; };
            int8_t v[4];
        };
        __host__ __device__ int8_4(int8_t x, int8_t y, int8_t z, int8_t w);
        __host__ __device__ int8_4(int8_t v[4]);
        __host__ __device__ int8_4 operator+(int8_4 other);
        __host__ __device__ int8_4 operator-(int8_4 other);
        __host__ __device__ int8_4 operator*(int8_t other);
        __host__ __device__ int8_4 operator/(int8_t other);
        __host__ __device__ static int8_t Dot(int8_4 left, int8_4 right);
        __host__ __device__ int8_t MagnatudeSquared() const;
        __host__ __device__ int8_t Magnatude() const;
    };
    class uint8_2 final {
    public:
        union {
            struct { uint8_t x, y; };
            uint8_t v[2];
        };
        __host__ __device__ uint8_2(uint8_t x, uint8_t y);
        __host__ __device__ uint8_2(uint8_t v[2]);
        __host__ __device__ uint8_2 operator+(uint8_2 other);
        __host__ __device__ uint8_2 operator-(uint8_2 other);
        __host__ __device__ uint8_2 operator*(uint8_t other);
        __host__ __device__ uint8_2 operator/(uint8_t other);
        __host__ __device__ static uint8_t Dot(uint8_2 left, uint8_2 right);
        __host__ __device__ uint8_t MagnatudeSquared() const;
        __host__ __device__ uint8_t Magnatude() const;
    };
    class uint8_3 final {
    public:
        union {
            struct { uint8_t x, y, z; };
            uint8_t v[3];
        };
        __host__ __device__ uint8_3(uint8_t x, uint8_t y, uint8_t z);
        __host__ __device__ uint8_3(uint8_t v[3]);
        __host__ __device__ uint8_3 operator+(uint8_3 other);
        __host__ __device__ uint8_3 operator-(uint8_3 other);
        __host__ __device__ uint8_3 operator*(uint8_t other);
        __host__ __device__ uint8_3 operator/(uint8_t other);
        __host__ __device__ static uint8_t Dot(uint8_3 left, uint8_3 right);
        __host__ __device__ uint8_t MagnatudeSquared() const;
        __host__ __device__ uint8_t Magnatude() const;
    };
    class uint8_4 final {
    public:
        union {
            struct { uint8_t x, y, z, w; };
            uint8_t v[4];
        };
        __host__ __device__ uint8_4(uint8_t x, uint8_t y, uint8_t z, uint8_t w);
        __host__ __device__ uint8_4(uint8_t v[4]);
        __host__ __device__ uint8_4 operator+(uint8_4 other);
        __host__ __device__ uint8_4 operator-(uint8_4 other);
        __host__ __device__ uint8_4 operator*(uint8_t other);
        __host__ __device__ uint8_4 operator/(uint8_t other);
        __host__ __device__ static uint8_t Dot(uint8_4 left, uint8_4 right);
        __host__ __device__ uint8_t MagnatudeSquared() const;
        __host__ __device__ uint8_t Magnatude() const;
    };
    class int16_2 final {
    public:
        union {
            struct { int16_t x, y; };
            int16_t v[2];
        };
        __host__ __device__ int16_2(int16_t x, int16_t y);
        __host__ __device__ int16_2(int16_t v[2]);
        __host__ __device__ int16_2 operator+(int16_2 other);
        __host__ __device__ int16_2 operator-(int16_2 other);
        __host__ __device__ int16_2 operator*(int16_t other);
        __host__ __device__ int16_2 operator/(int16_t other);
        __host__ __device__ static int16_t Dot(int16_2 left, int16_2 right);
        __host__ __device__ int16_t MagnatudeSquared() const;
        __host__ __device__ int16_t Magnatude() const;
    };
    class int16_3 final {
    public:
        union {
            struct { int16_t x, y, z; };
            int16_t v[3];
        };
        __host__ __device__ int16_3(int16_t x, int16_t y, int16_t z);
        __host__ __device__ int16_3(int16_t v[3]);
        __host__ __device__ int16_3 operator+(int16_3 other);
        __host__ __device__ int16_3 operator-(int16_3 other);
        __host__ __device__ int16_3 operator*(int16_t other);
        __host__ __device__ int16_3 operator/(int16_t other);
        __host__ __device__ static int16_t Dot(int16_3 left, int16_3 right);
        __host__ __device__ int16_t MagnatudeSquared() const;
        __host__ __device__ int16_t Magnatude() const;
    };
    class int16_4 final {
    public:
        union {
            struct { int16_t x, y, z, w; };
            int16_t v[4];
        };
        __host__ __device__ int16_4(int16_t x, int16_t y, int16_t z, int16_t w);
        __host__ __device__ int16_4(int16_t v[4]);
        __host__ __device__ int16_4 operator+(int16_4 other);
        __host__ __device__ int16_4 operator-(int16_4 other);
        __host__ __device__ int16_4 operator*(int16_t other);
        __host__ __device__ int16_4 operator/(int16_t other);
        __host__ __device__ static int16_t Dot(int16_4 left, int16_4 right);
        __host__ __device__ int16_t MagnatudeSquared() const;
        __host__ __device__ int16_t Magnatude() const;
    };
    class uint16_2 final {
    public:
        union {
            struct { uint16_t x, y; };
            uint16_t v[2];
        };
        __host__ __device__ uint16_2(uint16_t x, uint16_t y);
        __host__ __device__ uint16_2(uint16_t v[2]);
        __host__ __device__ uint16_2 operator+(uint16_2 other);
        __host__ __device__ uint16_2 operator-(uint16_2 other);
        __host__ __device__ uint16_2 operator*(uint16_t other);
        __host__ __device__ uint16_2 operator/(uint16_t other);
        __host__ __device__ static uint16_t Dot(uint16_2 left, uint16_2 right);
        __host__ __device__ uint16_t MagnatudeSquared() const;
        __host__ __device__ uint16_t Magnatude() const;
    };
    class uint16_3 final {
    public:
        union {
            struct { uint16_t x, y, z; };
            uint16_t v[3];
        };
        __host__ __device__ uint16_3(uint16_t x, uint16_t y, uint16_t z);
        __host__ __device__ uint16_3(uint16_t v[3]);
        __host__ __device__ uint16_3 operator+(uint16_3 other);
        __host__ __device__ uint16_3 operator-(uint16_3 other);
        __host__ __device__ uint16_3 operator*(uint16_t other);
        __host__ __device__ uint16_3 operator/(uint16_t other);
        __host__ __device__ static uint16_t Dot(uint16_3 left, uint16_3 right);
        __host__ __device__ uint16_t MagnatudeSquared() const;
        __host__ __device__ uint16_t Magnatude() const;
    };
    class uint16_4 final {
    public:
        union {
            struct { uint16_t x, y, z, w; };
            uint16_t v[4];
        };
        __host__ __device__ uint16_4(uint16_t x, uint16_t y, uint16_t z, uint16_t w);
        __host__ __device__ uint16_4(uint16_t v[4]);
        __host__ __device__ uint16_4 operator+(uint16_4 other);
        __host__ __device__ uint16_4 operator-(uint16_4 other);
        __host__ __device__ uint16_4 operator*(uint16_t other);
        __host__ __device__ uint16_4 operator/(uint16_t other);
        __host__ __device__ static uint16_t Dot(uint16_4 left, uint16_4 right);
        __host__ __device__ uint16_t MagnatudeSquared() const;
        __host__ __device__ uint16_t Magnatude() const;
    };
    class int32_2 final {
    public:
        union {
            struct { int32_t x, y; };
            int32_t v[2];
        };
        __host__ __device__ int32_2(int32_t x, int32_t y);
        __host__ __device__ int32_2(int32_t v[2]);
        __host__ __device__ int32_2 operator+(int32_2 other);
        __host__ __device__ int32_2 operator-(int32_2 other);
        __host__ __device__ int32_2 operator*(int32_t other);
        __host__ __device__ int32_2 operator/(int32_t other);
        __host__ __device__ static int32_t Dot(int32_2 left, int32_2 right);
        __host__ __device__ int32_t MagnatudeSquared() const;
        __host__ __device__ int32_t Magnatude() const;
    };
    class int32_3 final {
    public:
        union {
            struct { int32_t x, y, z; };
            int32_t v[3];
        };
        __host__ __device__ int32_3(int32_t x, int32_t y, int32_t z);
        __host__ __device__ int32_3(int32_t v[3]);
        __host__ __device__ int32_3 operator+(int32_3 other);
        __host__ __device__ int32_3 operator-(int32_3 other);
        __host__ __device__ int32_3 operator*(int32_t other);
        __host__ __device__ int32_3 operator/(int32_t other);
        __host__ __device__ static int32_t Dot(int32_3 left, int32_3 right);
        __host__ __device__ int32_t MagnatudeSquared() const;
        __host__ __device__ int32_t Magnatude() const;
    };
    class int32_4 final {
    public:
        union {
            struct { int32_t x, y, z, w; };
            int32_t v[4];
        };
        __host__ __device__ int32_4(int32_t x, int32_t y, int32_t z, int32_t w);
        __host__ __device__ int32_4(int32_t v[4]);
        __host__ __device__ int32_4 operator+(int32_4 other);
        __host__ __device__ int32_4 operator-(int32_4 other);
        __host__ __device__ int32_4 operator*(int32_t other);
        __host__ __device__ int32_4 operator/(int32_t other);
        __host__ __device__ static int32_t Dot(int32_4 left, int32_4 right);
        __host__ __device__ int32_t MagnatudeSquared() const;
        __host__ __device__ int32_t Magnatude() const;
    };
    class uint32_2 final {
    public:
        union {
            struct { uint32_t x, y; };
            uint32_t v[2];
        };
        __host__ __device__ uint32_2(uint32_t x, uint32_t y);
        __host__ __device__ uint32_2(uint32_t v[2]);
        __host__ __device__ uint32_2 operator+(uint32_2 other);
        __host__ __device__ uint32_2 operator-(uint32_2 other);
        __host__ __device__ uint32_2 operator*(uint32_t other);
        __host__ __device__ uint32_2 operator/(uint32_t other);
        __host__ __device__ static uint32_t Dot(uint32_2 left, uint32_2 right);
        __host__ __device__ uint32_t MagnatudeSquared() const;
        __host__ __device__ uint32_t Magnatude() const;
    };
    class uint32_3 final {
    public:
        union {
            struct { uint32_t x, y, z; };
            uint32_t v[3];
        };
        __host__ __device__ uint32_3(uint32_t x, uint32_t y, uint32_t z);
        __host__ __device__ uint32_3(uint32_t v[3]);
        __host__ __device__ uint32_3 operator+(uint32_3 other);
        __host__ __device__ uint32_3 operator-(uint32_3 other);
        __host__ __device__ uint32_3 operator*(uint32_t other);
        __host__ __device__ uint32_3 operator/(uint32_t other);
        __host__ __device__ static uint32_t Dot(uint32_3 left, uint32_3 right);
        __host__ __device__ uint32_t MagnatudeSquared() const;
        __host__ __device__ uint32_t Magnatude() const;
    };
    class uint32_4 final {
    public:
        union {
            struct { uint32_t x, y, z, w; };
            uint32_t v[4];
        };
        __host__ __device__ uint32_4(uint32_t x, uint32_t y, uint32_t z, uint32_t w);
        __host__ __device__ uint32_4(uint32_t v[4]);
        __host__ __device__ uint32_4 operator+(uint32_4 other);
        __host__ __device__ uint32_4 operator-(uint32_4 other);
        __host__ __device__ uint32_4 operator*(uint32_t other);
        __host__ __device__ uint32_4 operator/(uint32_t other);
        __host__ __device__ static uint32_t Dot(uint32_4 left, uint32_4 right);
        __host__ __device__ uint32_t MagnatudeSquared() const;
        __host__ __device__ uint32_t Magnatude() const;
    };
    class int64_2 final {
    public:
        union {
            struct { int64_t x, y; };
            int64_t v[2];
        };
        __host__ __device__ int64_2(int64_t x, int64_t y);
        __host__ __device__ int64_2(int64_t v[2]);
        __host__ __device__ int64_2 operator+(int64_2 other);
        __host__ __device__ int64_2 operator-(int64_2 other);
        __host__ __device__ int64_2 operator*(int64_t other);
        __host__ __device__ int64_2 operator/(int64_t other);
        __host__ __device__ static int64_t Dot(int64_2 left, int64_2 right);
        __host__ __device__ int64_t MagnatudeSquared() const;
        __host__ __device__ int64_t Magnatude() const;
    };
    class int64_3 final {
    public:
        union {
            struct { int64_t x, y, z; };
            int64_t v[3];
        };
        __host__ __device__ int64_3(int64_t x, int64_t y, int64_t z);
        __host__ __device__ int64_3(int64_t v[3]);
        __host__ __device__ int64_3 operator+(int64_3 other);
        __host__ __device__ int64_3 operator-(int64_3 other);
        __host__ __device__ int64_3 operator*(int64_t other);
        __host__ __device__ int64_3 operator/(int64_t other);
        __host__ __device__ static int64_t Dot(int64_3 left, int64_3 right);
        __host__ __device__ int64_t MagnatudeSquared() const;
        __host__ __device__ int64_t Magnatude() const;
    };
    class int64_4 final {
    public:
        union {
            struct { int64_t x, y, z, w; };
            int64_t v[4];
        };
        __host__ __device__ int64_4(int64_t x, int64_t y, int64_t z, int64_t w);
        __host__ __device__ int64_4(int64_t v[4]);
        __host__ __device__ int64_4 operator+(int64_4 other);
        __host__ __device__ int64_4 operator-(int64_4 other);
        __host__ __device__ int64_4 operator*(int64_t other);
        __host__ __device__ int64_4 operator/(int64_t other);
        __host__ __device__ static int64_t Dot(int64_4 left, int64_4 right);
        __host__ __device__ int64_t MagnatudeSquared() const;
        __host__ __device__ int64_t Magnatude() const;
    };
    class uint64_2 final {
    public:
        union {
            struct { uint64_t x, y; };
            uint64_t v[2];
        };
        __host__ __device__ uint64_2(uint64_t x, uint64_t y);
        __host__ __device__ uint64_2(uint64_t v[2]);
        __host__ __device__ uint64_2 operator+(uint64_2 other);
        __host__ __device__ uint64_2 operator-(uint64_2 other);
        __host__ __device__ uint64_2 operator*(uint64_t other);
        __host__ __device__ uint64_2 operator/(uint64_t other);
        __host__ __device__ static uint64_t Dot(uint64_2 left, uint64_2 right);
        __host__ __device__ uint64_t MagnatudeSquared() const;
        __host__ __device__ uint64_t Magnatude() const;
    };
    class uint64_3 final {
    public:
        union {
            struct { uint64_t x, y, z; };
            uint64_t v[3];
        };
        __host__ __device__ uint64_3(uint64_t x, uint64_t y, uint64_t z);
        __host__ __device__ uint64_3(uint64_t v[3]);
        __host__ __device__ uint64_3 operator+(uint64_3 other);
        __host__ __device__ uint64_3 operator-(uint64_3 other);
        __host__ __device__ uint64_3 operator*(uint64_t other);
        __host__ __device__ uint64_3 operator/(uint64_t other);
        __host__ __device__ static uint64_t Dot(uint64_3 left, uint64_3 right);
        __host__ __device__ uint64_t MagnatudeSquared() const;
        __host__ __device__ uint64_t Magnatude() const;
    };
    class uint64_4 final {
    public:
        union {
            struct { uint64_t x, y, z, w; };
            uint64_t v[4];
        };
        __host__ __device__ uint64_4(uint64_t x, uint64_t y, uint64_t z, uint64_t w);
        __host__ __device__ uint64_4(uint64_t v[4]);
        __host__ __device__ uint64_4 operator+(uint64_4 other);
        __host__ __device__ uint64_4 operator-(uint64_4 other);
        __host__ __device__ uint64_4 operator*(uint64_t other);
        __host__ __device__ uint64_4 operator/(uint64_t other);
        __host__ __device__ static uint64_t Dot(uint64_4 left, uint64_4 right);
        __host__ __device__ uint64_t MagnatudeSquared() const;
        __host__ __device__ uint64_t Magnatude() const;
    };
}