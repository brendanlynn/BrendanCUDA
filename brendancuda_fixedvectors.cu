#include "brendancuda_fixedvectors.cuh"
#include "brendancuda_math.cuh"

__host__ __device__ BrendanCUDA::float_2::float_2(float x, float y) {
    this->x = x;
    this->y = y;
}
__host__ __device__ BrendanCUDA::float_2::float_2(float v[2]) {
    this->v[0] = v[0];
    this->v[1] = v[1];
}
__host__ __device__ BrendanCUDA::float_2 BrendanCUDA::float_2::operator+(float_2 other) {
    return float_2(x + other.x, y + other.y);
}
__host__ __device__ BrendanCUDA::float_2 BrendanCUDA::float_2::operator-(float_2 other) {
    return float_2(x - other.x, y - other.y);
}
__host__ __device__ BrendanCUDA::float_2 BrendanCUDA::float_2::operator*(float other) {
    return float_2(x * other, y * other);
}
__host__ __device__ BrendanCUDA::float_2 BrendanCUDA::float_2::operator/(float other) {
    return float_2(x / other, y / other);
}
__host__ __device__ float BrendanCUDA::float_2::Dot(float_2 left, float_2 right) {
    return left.x * right.x + left.y * right.y;
}
__host__ __device__ float BrendanCUDA::float_2::MagnatudeSquared() const {
    return x * x + y * y;
}
__host__ __device__ float BrendanCUDA::float_2::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ BrendanCUDA::float_3::float_3(float x, float y, float z) {
    this->x = x;
    this->y = y;
    this->z = z;
}
__host__ __device__ BrendanCUDA::float_3::float_3(float v[3]) {
    this->v[0] = v[0];
    this->v[1] = v[1];
    this->v[2] = v[2];
}
__host__ __device__ BrendanCUDA::float_3 BrendanCUDA::float_3::operator+(float_3 other) {
    return float_3(x + other.x, y + other.y, z + other.z);
}
__host__ __device__ BrendanCUDA::float_3 BrendanCUDA::float_3::operator-(float_3 other) {
    return float_3(x - other.x, y - other.y, z - other.z);
}
__host__ __device__ BrendanCUDA::float_3 BrendanCUDA::float_3::operator*(float other) {
    return float_3(x * other, y * other, z * other);
}
__host__ __device__ BrendanCUDA::float_3 BrendanCUDA::float_3::operator/(float other) {
    return float_3(x / other, y / other, z / other);
}
__host__ __device__ float BrendanCUDA::float_3::Dot(float_3 left, float_3 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}
__host__ __device__ float BrendanCUDA::float_3::MagnatudeSquared() const {
    return x * x + y * y + z * z;
}
__host__ __device__ float BrendanCUDA::float_3::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ BrendanCUDA::float_4::float_4(float x, float y, float z, float w) {
    this->x = x;
    this->y = y;
    this->z = z;
    this->w = w;
}
__host__ __device__ BrendanCUDA::float_4::float_4(float v[4]) {
    this->v[0] = v[0];
    this->v[1] = v[1];
    this->v[2] = v[2];
    this->v[3] = v[3];
}
__host__ __device__ BrendanCUDA::float_4 BrendanCUDA::float_4::operator+(float_4 other) {
    return float_4(x + other.x, y + other.y, z + other.z, w + other.w);
}
__host__ __device__ BrendanCUDA::float_4 BrendanCUDA::float_4::operator-(float_4 other) {
    return float_4(x - other.x, y - other.y, z - other.z, w - other.w);
}
__host__ __device__ BrendanCUDA::float_4 BrendanCUDA::float_4::operator*(float other) {
    return float_4(x * other, y * other, z * other, w * other);
}
__host__ __device__ BrendanCUDA::float_4 BrendanCUDA::float_4::operator/(float other) {
    return float_4(x / other, y / other, z / other, w / other);
}
__host__ __device__ float BrendanCUDA::float_4::Dot(float_4 left, float_4 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w;
}
__host__ __device__ float BrendanCUDA::float_4::MagnatudeSquared() const {
    return x * x + y * y + z * z + w * w;
}
__host__ __device__ float BrendanCUDA::float_4::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ BrendanCUDA::double_2::double_2(double x, double y) {
    this->x = x;
    this->y = y;
}
__host__ __device__ BrendanCUDA::double_2::double_2(double v[2]) {
    this->v[0] = v[0];
    this->v[1] = v[1];
}
__host__ __device__ BrendanCUDA::double_2 BrendanCUDA::double_2::operator+(double_2 other) {
    return double_2(x + other.x, y + other.y);
}
__host__ __device__ BrendanCUDA::double_2 BrendanCUDA::double_2::operator-(double_2 other) {
    return double_2(x - other.x, y - other.y);
}
__host__ __device__ BrendanCUDA::double_2 BrendanCUDA::double_2::operator*(double other) {
    return double_2(x * other, y * other);
}
__host__ __device__ BrendanCUDA::double_2 BrendanCUDA::double_2::operator/(double other) {
    return double_2(x / other, y / other);
}
__host__ __device__ double BrendanCUDA::double_2::Dot(double_2 left, double_2 right) {
    return left.x * right.x + left.y * right.y;
}
__host__ __device__ double BrendanCUDA::double_2::MagnatudeSquared() const {
    return x * x + y * y;
}
__host__ __device__ double BrendanCUDA::double_2::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ BrendanCUDA::double_3::double_3(double x, double y, double z) {
    this->x = x;
    this->y = y;
    this->z = z;
}
__host__ __device__ BrendanCUDA::double_3::double_3(double v[3]) {
    this->v[0] = v[0];
    this->v[1] = v[1];
    this->v[2] = v[2];
}
__host__ __device__ BrendanCUDA::double_3 BrendanCUDA::double_3::operator+(double_3 other) {
    return double_3(x + other.x, y + other.y, z + other.z);
}
__host__ __device__ BrendanCUDA::double_3 BrendanCUDA::double_3::operator-(double_3 other) {
    return double_3(x - other.x, y - other.y, z - other.z);
}
__host__ __device__ BrendanCUDA::double_3 BrendanCUDA::double_3::operator*(double other) {
    return double_3(x * other, y * other, z * other);
}
__host__ __device__ BrendanCUDA::double_3 BrendanCUDA::double_3::operator/(double other) {
    return double_3(x / other, y / other, z / other);
}
__host__ __device__ double BrendanCUDA::double_3::Dot(double_3 left, double_3 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}
__host__ __device__ double BrendanCUDA::double_3::MagnatudeSquared() const {
    return x * x + y * y + z * z;
}
__host__ __device__ double BrendanCUDA::double_3::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ BrendanCUDA::double_4::double_4(double x, double y, double z, double w) {
    this->x = x;
    this->y = y;
    this->z = z;
    this->w = w;
}
__host__ __device__ BrendanCUDA::double_4::double_4(double v[4]) {
    this->v[0] = v[0];
    this->v[1] = v[1];
    this->v[2] = v[2];
    this->v[3] = v[3];
}
__host__ __device__ BrendanCUDA::double_4 BrendanCUDA::double_4::operator+(double_4 other) {
    return double_4(x + other.x, y + other.y, z + other.z, w + other.w);
}
__host__ __device__ BrendanCUDA::double_4 BrendanCUDA::double_4::operator-(double_4 other) {
    return double_4(x - other.x, y - other.y, z - other.z, w - other.w);
}
__host__ __device__ BrendanCUDA::double_4 BrendanCUDA::double_4::operator*(double other) {
    return double_4(x * other, y * other, z * other, w * other);
}
__host__ __device__ BrendanCUDA::double_4 BrendanCUDA::double_4::operator/(double other) {
    return double_4(x / other, y / other, z / other, w / other);
}
__host__ __device__ double BrendanCUDA::double_4::Dot(double_4 left, double_4 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w;
}
__host__ __device__ double BrendanCUDA::double_4::MagnatudeSquared() const {
    return x * x + y * y + z * z + w * w;
}
__host__ __device__ double BrendanCUDA::double_4::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ BrendanCUDA::int8_2::int8_2(int8_t x, int8_t y) {
    this->x = x;
    this->y = y;
}
__host__ __device__ BrendanCUDA::int8_2::int8_2(int8_t v[2]) {
    this->v[0] = v[0];
    this->v[1] = v[1];
}
__host__ __device__ BrendanCUDA::int8_2 BrendanCUDA::int8_2::operator+(int8_2 other) {
    return int8_2(x + other.x, y + other.y);
}
__host__ __device__ BrendanCUDA::int8_2 BrendanCUDA::int8_2::operator-(int8_2 other) {
    return int8_2(x - other.x, y - other.y);
}
__host__ __device__ BrendanCUDA::int8_2 BrendanCUDA::int8_2::operator*(int8_t other) {
    return int8_2(x * other, y * other);
}
__host__ __device__ BrendanCUDA::int8_2 BrendanCUDA::int8_2::operator/(int8_t other) {
    return int8_2(x / other, y / other);
}
__host__ __device__ int8_t BrendanCUDA::int8_2::Dot(int8_2 left, int8_2 right) {
    return left.x * right.x + left.y * right.y;
}
__host__ __device__ int8_t BrendanCUDA::int8_2::MagnatudeSquared() const {
    return x * x + y * y;
}
__host__ __device__ int8_t BrendanCUDA::int8_2::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ BrendanCUDA::int8_3::int8_3(int8_t x, int8_t y, int8_t z) {
    this->x = x;
    this->y = y;
    this->z = z;
}
__host__ __device__ BrendanCUDA::int8_3::int8_3(int8_t v[3]) {
    this->v[0] = v[0];
    this->v[1] = v[1];
    this->v[2] = v[2];
}
__host__ __device__ BrendanCUDA::int8_3 BrendanCUDA::int8_3::operator+(int8_3 other) {
    return int8_3(x + other.x, y + other.y, z + other.z);
}
__host__ __device__ BrendanCUDA::int8_3 BrendanCUDA::int8_3::operator-(int8_3 other) {
    return int8_3(x - other.x, y - other.y, z - other.z);
}
__host__ __device__ BrendanCUDA::int8_3 BrendanCUDA::int8_3::operator*(int8_t other) {
    return int8_3(x * other, y * other, z * other);
}
__host__ __device__ BrendanCUDA::int8_3 BrendanCUDA::int8_3::operator/(int8_t other) {
    return int8_3(x / other, y / other, z / other);
}
__host__ __device__ int8_t BrendanCUDA::int8_3::Dot(int8_3 left, int8_3 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}
__host__ __device__ int8_t BrendanCUDA::int8_3::MagnatudeSquared() const {
    return x * x + y * y + z * z;
}
__host__ __device__ int8_t BrendanCUDA::int8_3::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ BrendanCUDA::int8_4::int8_4(int8_t x, int8_t y, int8_t z, int8_t w) {
    this->x = x;
    this->y = y;
    this->z = z;
    this->w = w;
}
__host__ __device__ BrendanCUDA::int8_4::int8_4(int8_t v[4]) {
    this->v[0] = v[0];
    this->v[1] = v[1];
    this->v[2] = v[2];
    this->v[3] = v[3];
}
__host__ __device__ BrendanCUDA::int8_4 BrendanCUDA::int8_4::operator+(int8_4 other) {
    return int8_4(x + other.x, y + other.y, z + other.z, w + other.w);
}
__host__ __device__ BrendanCUDA::int8_4 BrendanCUDA::int8_4::operator-(int8_4 other) {
    return int8_4(x - other.x, y - other.y, z - other.z, w - other.w);
}
__host__ __device__ BrendanCUDA::int8_4 BrendanCUDA::int8_4::operator*(int8_t other) {
    return int8_4(x * other, y * other, z * other, w * other);
}
__host__ __device__ BrendanCUDA::int8_4 BrendanCUDA::int8_4::operator/(int8_t other) {
    return int8_4(x / other, y / other, z / other, w / other);
}
__host__ __device__ int8_t BrendanCUDA::int8_4::Dot(int8_4 left, int8_4 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w;
}
__host__ __device__ int8_t BrendanCUDA::int8_4::MagnatudeSquared() const {
    return x * x + y * y + z * z + w * w;
}
__host__ __device__ int8_t BrendanCUDA::int8_4::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ BrendanCUDA::uint8_2::uint8_2(uint8_t x, uint8_t y) {
    this->x = x;
    this->y = y;
}
__host__ __device__ BrendanCUDA::uint8_2::uint8_2(uint8_t v[2]) {
    this->v[0] = v[0];
    this->v[1] = v[1];
}
__host__ __device__ BrendanCUDA::uint8_2 BrendanCUDA::uint8_2::operator+(uint8_2 other) {
    return uint8_2(x + other.x, y + other.y);
}
__host__ __device__ BrendanCUDA::uint8_2 BrendanCUDA::uint8_2::operator-(uint8_2 other) {
    return uint8_2(x - other.x, y - other.y);
}
__host__ __device__ BrendanCUDA::uint8_2 BrendanCUDA::uint8_2::operator*(uint8_t other) {
    return uint8_2(x * other, y * other);
}
__host__ __device__ BrendanCUDA::uint8_2 BrendanCUDA::uint8_2::operator/(uint8_t other) {
    return uint8_2(x / other, y / other);
}
__host__ __device__ uint8_t BrendanCUDA::uint8_2::Dot(uint8_2 left, uint8_2 right) {
    return left.x * right.x + left.y * right.y;
}
__host__ __device__ uint8_t BrendanCUDA::uint8_2::MagnatudeSquared() const {
    return x * x + y * y;
}
__host__ __device__ uint8_t BrendanCUDA::uint8_2::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ BrendanCUDA::uint8_3::uint8_3(uint8_t x, uint8_t y, uint8_t z) {
    this->x = x;
    this->y = y;
    this->z = z;
}
__host__ __device__ BrendanCUDA::uint8_3::uint8_3(uint8_t v[3]) {
    this->v[0] = v[0];
    this->v[1] = v[1];
    this->v[2] = v[2];
}
__host__ __device__ BrendanCUDA::uint8_3 BrendanCUDA::uint8_3::operator+(uint8_3 other) {
    return uint8_3(x + other.x, y + other.y, z + other.z);
}
__host__ __device__ BrendanCUDA::uint8_3 BrendanCUDA::uint8_3::operator-(uint8_3 other) {
    return uint8_3(x - other.x, y - other.y, z - other.z);
}
__host__ __device__ BrendanCUDA::uint8_3 BrendanCUDA::uint8_3::operator*(uint8_t other) {
    return uint8_3(x * other, y * other, z * other);
}
__host__ __device__ BrendanCUDA::uint8_3 BrendanCUDA::uint8_3::operator/(uint8_t other) {
    return uint8_3(x / other, y / other, z / other);
}
__host__ __device__ uint8_t BrendanCUDA::uint8_3::Dot(uint8_3 left, uint8_3 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}
__host__ __device__ uint8_t BrendanCUDA::uint8_3::MagnatudeSquared() const {
    return x * x + y * y + z * z;
}
__host__ __device__ uint8_t BrendanCUDA::uint8_3::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ BrendanCUDA::uint8_4::uint8_4(uint8_t x, uint8_t y, uint8_t z, uint8_t w) {
    this->x = x;
    this->y = y;
    this->z = z;
    this->w = w;
}
__host__ __device__ BrendanCUDA::uint8_4::uint8_4(uint8_t v[4]) {
    this->v[0] = v[0];
    this->v[1] = v[1];
    this->v[2] = v[2];
    this->v[3] = v[3];
}
__host__ __device__ BrendanCUDA::uint8_4 BrendanCUDA::uint8_4::operator+(uint8_4 other) {
    return uint8_4(x + other.x, y + other.y, z + other.z, w + other.w);
}
__host__ __device__ BrendanCUDA::uint8_4 BrendanCUDA::uint8_4::operator-(uint8_4 other) {
    return uint8_4(x - other.x, y - other.y, z - other.z, w - other.w);
}
__host__ __device__ BrendanCUDA::uint8_4 BrendanCUDA::uint8_4::operator*(uint8_t other) {
    return uint8_4(x * other, y * other, z * other, w * other);
}
__host__ __device__ BrendanCUDA::uint8_4 BrendanCUDA::uint8_4::operator/(uint8_t other) {
    return uint8_4(x / other, y / other, z / other, w / other);
}
__host__ __device__ uint8_t BrendanCUDA::uint8_4::Dot(uint8_4 left, uint8_4 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w;
}
__host__ __device__ uint8_t BrendanCUDA::uint8_4::MagnatudeSquared() const {
    return x * x + y * y + z * z + w * w;
}
__host__ __device__ uint8_t BrendanCUDA::uint8_4::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ BrendanCUDA::int16_2::int16_2(int16_t x, int16_t y) {
    this->x = x;
    this->y = y;
}
__host__ __device__ BrendanCUDA::int16_2::int16_2(int16_t v[2]) {
    this->v[0] = v[0];
    this->v[1] = v[1];
}
__host__ __device__ BrendanCUDA::int16_2 BrendanCUDA::int16_2::operator+(int16_2 other) {
    return int16_2(x + other.x, y + other.y);
}
__host__ __device__ BrendanCUDA::int16_2 BrendanCUDA::int16_2::operator-(int16_2 other) {
    return int16_2(x - other.x, y - other.y);
}
__host__ __device__ BrendanCUDA::int16_2 BrendanCUDA::int16_2::operator*(int16_t other) {
    return int16_2(x * other, y * other);
}
__host__ __device__ BrendanCUDA::int16_2 BrendanCUDA::int16_2::operator/(int16_t other) {
    return int16_2(x / other, y / other);
}
__host__ __device__ int16_t BrendanCUDA::int16_2::Dot(int16_2 left, int16_2 right) {
    return left.x * right.x + left.y * right.y;
}
__host__ __device__ int16_t BrendanCUDA::int16_2::MagnatudeSquared() const {
    return x * x + y * y;
}
__host__ __device__ int16_t BrendanCUDA::int16_2::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ BrendanCUDA::int16_3::int16_3(int16_t x, int16_t y, int16_t z) {
    this->x = x;
    this->y = y;
    this->z = z;
}
__host__ __device__ BrendanCUDA::int16_3::int16_3(int16_t v[3]) {
    this->v[0] = v[0];
    this->v[1] = v[1];
    this->v[2] = v[2];
}
__host__ __device__ BrendanCUDA::int16_3 BrendanCUDA::int16_3::operator+(int16_3 other) {
    return int16_3(x + other.x, y + other.y, z + other.z);
}
__host__ __device__ BrendanCUDA::int16_3 BrendanCUDA::int16_3::operator-(int16_3 other) {
    return int16_3(x - other.x, y - other.y, z - other.z);
}
__host__ __device__ BrendanCUDA::int16_3 BrendanCUDA::int16_3::operator*(int16_t other) {
    return int16_3(x * other, y * other, z * other);
}
__host__ __device__ BrendanCUDA::int16_3 BrendanCUDA::int16_3::operator/(int16_t other) {
    return int16_3(x / other, y / other, z / other);
}
__host__ __device__ int16_t BrendanCUDA::int16_3::Dot(int16_3 left, int16_3 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}
__host__ __device__ int16_t BrendanCUDA::int16_3::MagnatudeSquared() const {
    return x * x + y * y + z * z;
}
__host__ __device__ int16_t BrendanCUDA::int16_3::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ BrendanCUDA::int16_4::int16_4(int16_t x, int16_t y, int16_t z, int16_t w) {
    this->x = x;
    this->y = y;
    this->z = z;
    this->w = w;
}
__host__ __device__ BrendanCUDA::int16_4::int16_4(int16_t v[4]) {
    this->v[0] = v[0];
    this->v[1] = v[1];
    this->v[2] = v[2];
    this->v[3] = v[3];
}
__host__ __device__ BrendanCUDA::int16_4 BrendanCUDA::int16_4::operator+(int16_4 other) {
    return int16_4(x + other.x, y + other.y, z + other.z, w + other.w);
}
__host__ __device__ BrendanCUDA::int16_4 BrendanCUDA::int16_4::operator-(int16_4 other) {
    return int16_4(x - other.x, y - other.y, z - other.z, w - other.w);
}
__host__ __device__ BrendanCUDA::int16_4 BrendanCUDA::int16_4::operator*(int16_t other) {
    return int16_4(x * other, y * other, z * other, w * other);
}
__host__ __device__ BrendanCUDA::int16_4 BrendanCUDA::int16_4::operator/(int16_t other) {
    return int16_4(x / other, y / other, z / other, w / other);
}
__host__ __device__ int16_t BrendanCUDA::int16_4::Dot(int16_4 left, int16_4 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w;
}
__host__ __device__ int16_t BrendanCUDA::int16_4::MagnatudeSquared() const {
    return x * x + y * y + z * z + w * w;
}
__host__ __device__ int16_t BrendanCUDA::int16_4::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ BrendanCUDA::uint16_2::uint16_2(uint16_t x, uint16_t y) {
    this->x = x;
    this->y = y;
}
__host__ __device__ BrendanCUDA::uint16_2::uint16_2(uint16_t v[2]) {
    this->v[0] = v[0];
    this->v[1] = v[1];
}
__host__ __device__ BrendanCUDA::uint16_2 BrendanCUDA::uint16_2::operator+(uint16_2 other) {
    return uint16_2(x + other.x, y + other.y);
}
__host__ __device__ BrendanCUDA::uint16_2 BrendanCUDA::uint16_2::operator-(uint16_2 other) {
    return uint16_2(x - other.x, y - other.y);
}
__host__ __device__ BrendanCUDA::uint16_2 BrendanCUDA::uint16_2::operator*(uint16_t other) {
    return uint16_2(x * other, y * other);
}
__host__ __device__ BrendanCUDA::uint16_2 BrendanCUDA::uint16_2::operator/(uint16_t other) {
    return uint16_2(x / other, y / other);
}
__host__ __device__ uint16_t BrendanCUDA::uint16_2::Dot(uint16_2 left, uint16_2 right) {
    return left.x * right.x + left.y * right.y;
}
__host__ __device__ uint16_t BrendanCUDA::uint16_2::MagnatudeSquared() const {
    return x * x + y * y;
}
__host__ __device__ uint16_t BrendanCUDA::uint16_2::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ BrendanCUDA::uint16_3::uint16_3(uint16_t x, uint16_t y, uint16_t z) {
    this->x = x;
    this->y = y;
    this->z = z;
}
__host__ __device__ BrendanCUDA::uint16_3::uint16_3(uint16_t v[3]) {
    this->v[0] = v[0];
    this->v[1] = v[1];
    this->v[2] = v[2];
}
__host__ __device__ BrendanCUDA::uint16_3 BrendanCUDA::uint16_3::operator+(uint16_3 other) {
    return uint16_3(x + other.x, y + other.y, z + other.z);
}
__host__ __device__ BrendanCUDA::uint16_3 BrendanCUDA::uint16_3::operator-(uint16_3 other) {
    return uint16_3(x - other.x, y - other.y, z - other.z);
}
__host__ __device__ BrendanCUDA::uint16_3 BrendanCUDA::uint16_3::operator*(uint16_t other) {
    return uint16_3(x * other, y * other, z * other);
}
__host__ __device__ BrendanCUDA::uint16_3 BrendanCUDA::uint16_3::operator/(uint16_t other) {
    return uint16_3(x / other, y / other, z / other);
}
__host__ __device__ uint16_t BrendanCUDA::uint16_3::Dot(uint16_3 left, uint16_3 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}
__host__ __device__ uint16_t BrendanCUDA::uint16_3::MagnatudeSquared() const {
    return x * x + y * y + z * z;
}
__host__ __device__ uint16_t BrendanCUDA::uint16_3::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ BrendanCUDA::uint16_4::uint16_4(uint16_t x, uint16_t y, uint16_t z, uint16_t w) {
    this->x = x;
    this->y = y;
    this->z = z;
    this->w = w;
}
__host__ __device__ BrendanCUDA::uint16_4::uint16_4(uint16_t v[4]) {
    this->v[0] = v[0];
    this->v[1] = v[1];
    this->v[2] = v[2];
    this->v[3] = v[3];
}
__host__ __device__ BrendanCUDA::uint16_4 BrendanCUDA::uint16_4::operator+(uint16_4 other) {
    return uint16_4(x + other.x, y + other.y, z + other.z, w + other.w);
}
__host__ __device__ BrendanCUDA::uint16_4 BrendanCUDA::uint16_4::operator-(uint16_4 other) {
    return uint16_4(x - other.x, y - other.y, z - other.z, w - other.w);
}
__host__ __device__ BrendanCUDA::uint16_4 BrendanCUDA::uint16_4::operator*(uint16_t other) {
    return uint16_4(x * other, y * other, z * other, w * other);
}
__host__ __device__ BrendanCUDA::uint16_4 BrendanCUDA::uint16_4::operator/(uint16_t other) {
    return uint16_4(x / other, y / other, z / other, w / other);
}
__host__ __device__ uint16_t BrendanCUDA::uint16_4::Dot(uint16_4 left, uint16_4 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w;
}
__host__ __device__ uint16_t BrendanCUDA::uint16_4::MagnatudeSquared() const {
    return x * x + y * y + z * z + w * w;
}
__host__ __device__ uint16_t BrendanCUDA::uint16_4::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ BrendanCUDA::int32_2::int32_2(int32_t x, int32_t y) {
    this->x = x;
    this->y = y;
}
__host__ __device__ BrendanCUDA::int32_2::int32_2(int32_t v[2]) {
    this->v[0] = v[0];
    this->v[1] = v[1];
}
__host__ __device__ BrendanCUDA::int32_2 BrendanCUDA::int32_2::operator+(int32_2 other) {
    return int32_2(x + other.x, y + other.y);
}
__host__ __device__ BrendanCUDA::int32_2 BrendanCUDA::int32_2::operator-(int32_2 other) {
    return int32_2(x - other.x, y - other.y);
}
__host__ __device__ BrendanCUDA::int32_2 BrendanCUDA::int32_2::operator*(int32_t other) {
    return int32_2(x * other, y * other);
}
__host__ __device__ BrendanCUDA::int32_2 BrendanCUDA::int32_2::operator/(int32_t other) {
    return int32_2(x / other, y / other);
}
__host__ __device__ int32_t BrendanCUDA::int32_2::Dot(int32_2 left, int32_2 right) {
    return left.x * right.x + left.y * right.y;
}
__host__ __device__ int32_t BrendanCUDA::int32_2::MagnatudeSquared() const {
    return x * x + y * y;
}
__host__ __device__ int32_t BrendanCUDA::int32_2::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ BrendanCUDA::int32_3::int32_3(int32_t x, int32_t y, int32_t z) {
    this->x = x;
    this->y = y;
    this->z = z;
}
__host__ __device__ BrendanCUDA::int32_3::int32_3(int32_t v[3]) {
    this->v[0] = v[0];
    this->v[1] = v[1];
    this->v[2] = v[2];
}
__host__ __device__ BrendanCUDA::int32_3 BrendanCUDA::int32_3::operator+(int32_3 other) {
    return int32_3(x + other.x, y + other.y, z + other.z);
}
__host__ __device__ BrendanCUDA::int32_3 BrendanCUDA::int32_3::operator-(int32_3 other) {
    return int32_3(x - other.x, y - other.y, z - other.z);
}
__host__ __device__ BrendanCUDA::int32_3 BrendanCUDA::int32_3::operator*(int32_t other) {
    return int32_3(x * other, y * other, z * other);
}
__host__ __device__ BrendanCUDA::int32_3 BrendanCUDA::int32_3::operator/(int32_t other) {
    return int32_3(x / other, y / other, z / other);
}
__host__ __device__ int32_t BrendanCUDA::int32_3::Dot(int32_3 left, int32_3 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}
__host__ __device__ int32_t BrendanCUDA::int32_3::MagnatudeSquared() const {
    return x * x + y * y + z * z;
}
__host__ __device__ int32_t BrendanCUDA::int32_3::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ BrendanCUDA::int32_4::int32_4(int32_t x, int32_t y, int32_t z, int32_t w) {
    this->x = x;
    this->y = y;
    this->z = z;
    this->w = w;
}
__host__ __device__ BrendanCUDA::int32_4::int32_4(int32_t v[4]) {
    this->v[0] = v[0];
    this->v[1] = v[1];
    this->v[2] = v[2];
    this->v[3] = v[3];
}
__host__ __device__ BrendanCUDA::int32_4 BrendanCUDA::int32_4::operator+(int32_4 other) {
    return int32_4(x + other.x, y + other.y, z + other.z, w + other.w);
}
__host__ __device__ BrendanCUDA::int32_4 BrendanCUDA::int32_4::operator-(int32_4 other) {
    return int32_4(x - other.x, y - other.y, z - other.z, w - other.w);
}
__host__ __device__ BrendanCUDA::int32_4 BrendanCUDA::int32_4::operator*(int32_t other) {
    return int32_4(x * other, y * other, z * other, w * other);
}
__host__ __device__ BrendanCUDA::int32_4 BrendanCUDA::int32_4::operator/(int32_t other) {
    return int32_4(x / other, y / other, z / other, w / other);
}
__host__ __device__ int32_t BrendanCUDA::int32_4::Dot(int32_4 left, int32_4 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w;
}
__host__ __device__ int32_t BrendanCUDA::int32_4::MagnatudeSquared() const {
    return x * x + y * y + z * z + w * w;
}
__host__ __device__ int32_t BrendanCUDA::int32_4::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ BrendanCUDA::uint32_2::uint32_2(uint32_t x, uint32_t y) {
    this->x = x;
    this->y = y;
}
__host__ __device__ BrendanCUDA::uint32_2::uint32_2(uint32_t v[2]) {
    this->v[0] = v[0];
    this->v[1] = v[1];
}
__host__ __device__ BrendanCUDA::uint32_2 BrendanCUDA::uint32_2::operator+(uint32_2 other) {
    return uint32_2(x + other.x, y + other.y);
}
__host__ __device__ BrendanCUDA::uint32_2 BrendanCUDA::uint32_2::operator-(uint32_2 other) {
    return uint32_2(x - other.x, y - other.y);
}
__host__ __device__ BrendanCUDA::uint32_2 BrendanCUDA::uint32_2::operator*(uint32_t other) {
    return uint32_2(x * other, y * other);
}
__host__ __device__ BrendanCUDA::uint32_2 BrendanCUDA::uint32_2::operator/(uint32_t other) {
    return uint32_2(x / other, y / other);
}
__host__ __device__ uint32_t BrendanCUDA::uint32_2::Dot(uint32_2 left, uint32_2 right) {
    return left.x * right.x + left.y * right.y;
}
__host__ __device__ uint32_t BrendanCUDA::uint32_2::MagnatudeSquared() const {
    return x * x + y * y;
}
__host__ __device__ uint32_t BrendanCUDA::uint32_2::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ BrendanCUDA::uint32_3::uint32_3(uint32_t x, uint32_t y, uint32_t z) {
    this->x = x;
    this->y = y;
    this->z = z;
}
__host__ __device__ BrendanCUDA::uint32_3::uint32_3(uint32_t v[3]) {
    this->v[0] = v[0];
    this->v[1] = v[1];
    this->v[2] = v[2];
}
__host__ __device__ BrendanCUDA::uint32_3 BrendanCUDA::uint32_3::operator+(uint32_3 other) {
    return uint32_3(x + other.x, y + other.y, z + other.z);
}
__host__ __device__ BrendanCUDA::uint32_3 BrendanCUDA::uint32_3::operator-(uint32_3 other) {
    return uint32_3(x - other.x, y - other.y, z - other.z);
}
__host__ __device__ BrendanCUDA::uint32_3 BrendanCUDA::uint32_3::operator*(uint32_t other) {
    return uint32_3(x * other, y * other, z * other);
}
__host__ __device__ BrendanCUDA::uint32_3 BrendanCUDA::uint32_3::operator/(uint32_t other) {
    return uint32_3(x / other, y / other, z / other);
}
__host__ __device__ uint32_t BrendanCUDA::uint32_3::Dot(uint32_3 left, uint32_3 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}
__host__ __device__ uint32_t BrendanCUDA::uint32_3::MagnatudeSquared() const {
    return x * x + y * y + z * z;
}
__host__ __device__ uint32_t BrendanCUDA::uint32_3::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ BrendanCUDA::uint32_4::uint32_4(uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
    this->x = x;
    this->y = y;
    this->z = z;
    this->w = w;
}
__host__ __device__ BrendanCUDA::uint32_4::uint32_4(uint32_t v[4]) {
    this->v[0] = v[0];
    this->v[1] = v[1];
    this->v[2] = v[2];
    this->v[3] = v[3];
}
__host__ __device__ BrendanCUDA::uint32_4 BrendanCUDA::uint32_4::operator+(uint32_4 other) {
    return uint32_4(x + other.x, y + other.y, z + other.z, w + other.w);
}
__host__ __device__ BrendanCUDA::uint32_4 BrendanCUDA::uint32_4::operator-(uint32_4 other) {
    return uint32_4(x - other.x, y - other.y, z - other.z, w - other.w);
}
__host__ __device__ BrendanCUDA::uint32_4 BrendanCUDA::uint32_4::operator*(uint32_t other) {
    return uint32_4(x * other, y * other, z * other, w * other);
}
__host__ __device__ BrendanCUDA::uint32_4 BrendanCUDA::uint32_4::operator/(uint32_t other) {
    return uint32_4(x / other, y / other, z / other, w / other);
}
__host__ __device__ uint32_t BrendanCUDA::uint32_4::Dot(uint32_4 left, uint32_4 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w;
}
__host__ __device__ uint32_t BrendanCUDA::uint32_4::MagnatudeSquared() const {
    return x * x + y * y + z * z + w * w;
}
__host__ __device__ uint32_t BrendanCUDA::uint32_4::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ BrendanCUDA::int64_2::int64_2(int64_t x, int64_t y) {
    this->x = x;
    this->y = y;
}
__host__ __device__ BrendanCUDA::int64_2::int64_2(int64_t v[2]) {
    this->v[0] = v[0];
    this->v[1] = v[1];
}
__host__ __device__ BrendanCUDA::int64_2 BrendanCUDA::int64_2::operator+(int64_2 other) {
    return int64_2(x + other.x, y + other.y);
}
__host__ __device__ BrendanCUDA::int64_2 BrendanCUDA::int64_2::operator-(int64_2 other) {
    return int64_2(x - other.x, y - other.y);
}
__host__ __device__ BrendanCUDA::int64_2 BrendanCUDA::int64_2::operator*(int64_t other) {
    return int64_2(x * other, y * other);
}
__host__ __device__ BrendanCUDA::int64_2 BrendanCUDA::int64_2::operator/(int64_t other) {
    return int64_2(x / other, y / other);
}
__host__ __device__ int64_t BrendanCUDA::int64_2::Dot(int64_2 left, int64_2 right) {
    return left.x * right.x + left.y * right.y;
}
__host__ __device__ int64_t BrendanCUDA::int64_2::MagnatudeSquared() const {
    return x * x + y * y;
}
__host__ __device__ int64_t BrendanCUDA::int64_2::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ BrendanCUDA::int64_3::int64_3(int64_t x, int64_t y, int64_t z) {
    this->x = x;
    this->y = y;
    this->z = z;
}
__host__ __device__ BrendanCUDA::int64_3::int64_3(int64_t v[3]) {
    this->v[0] = v[0];
    this->v[1] = v[1];
    this->v[2] = v[2];
}
__host__ __device__ BrendanCUDA::int64_3 BrendanCUDA::int64_3::operator+(int64_3 other) {
    return int64_3(x + other.x, y + other.y, z + other.z);
}
__host__ __device__ BrendanCUDA::int64_3 BrendanCUDA::int64_3::operator-(int64_3 other) {
    return int64_3(x - other.x, y - other.y, z - other.z);
}
__host__ __device__ BrendanCUDA::int64_3 BrendanCUDA::int64_3::operator*(int64_t other) {
    return int64_3(x * other, y * other, z * other);
}
__host__ __device__ BrendanCUDA::int64_3 BrendanCUDA::int64_3::operator/(int64_t other) {
    return int64_3(x / other, y / other, z / other);
}
__host__ __device__ int64_t BrendanCUDA::int64_3::Dot(int64_3 left, int64_3 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}
__host__ __device__ int64_t BrendanCUDA::int64_3::MagnatudeSquared() const {
    return x * x + y * y + z * z;
}
__host__ __device__ int64_t BrendanCUDA::int64_3::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ BrendanCUDA::int64_4::int64_4(int64_t x, int64_t y, int64_t z, int64_t w) {
    this->x = x;
    this->y = y;
    this->z = z;
    this->w = w;
}
__host__ __device__ BrendanCUDA::int64_4::int64_4(int64_t v[4]) {
    this->v[0] = v[0];
    this->v[1] = v[1];
    this->v[2] = v[2];
    this->v[3] = v[3];
}
__host__ __device__ BrendanCUDA::int64_4 BrendanCUDA::int64_4::operator+(int64_4 other) {
    return int64_4(x + other.x, y + other.y, z + other.z, w + other.w);
}
__host__ __device__ BrendanCUDA::int64_4 BrendanCUDA::int64_4::operator-(int64_4 other) {
    return int64_4(x - other.x, y - other.y, z - other.z, w - other.w);
}
__host__ __device__ BrendanCUDA::int64_4 BrendanCUDA::int64_4::operator*(int64_t other) {
    return int64_4(x * other, y * other, z * other, w * other);
}
__host__ __device__ BrendanCUDA::int64_4 BrendanCUDA::int64_4::operator/(int64_t other) {
    return int64_4(x / other, y / other, z / other, w / other);
}
__host__ __device__ int64_t BrendanCUDA::int64_4::Dot(int64_4 left, int64_4 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w;
}
__host__ __device__ int64_t BrendanCUDA::int64_4::MagnatudeSquared() const {
    return x * x + y * y + z * z + w * w;
}
__host__ __device__ int64_t BrendanCUDA::int64_4::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ BrendanCUDA::uint64_2::uint64_2(uint64_t x, uint64_t y) {
    this->x = x;
    this->y = y;
}
__host__ __device__ BrendanCUDA::uint64_2::uint64_2(uint64_t v[2]) {
    this->v[0] = v[0];
    this->v[1] = v[1];
}
__host__ __device__ BrendanCUDA::uint64_2 BrendanCUDA::uint64_2::operator+(uint64_2 other) {
    return uint64_2(x + other.x, y + other.y);
}
__host__ __device__ BrendanCUDA::uint64_2 BrendanCUDA::uint64_2::operator-(uint64_2 other) {
    return uint64_2(x - other.x, y - other.y);
}
__host__ __device__ BrendanCUDA::uint64_2 BrendanCUDA::uint64_2::operator*(uint64_t other) {
    return uint64_2(x * other, y * other);
}
__host__ __device__ BrendanCUDA::uint64_2 BrendanCUDA::uint64_2::operator/(uint64_t other) {
    return uint64_2(x / other, y / other);
}
__host__ __device__ uint64_t BrendanCUDA::uint64_2::Dot(uint64_2 left, uint64_2 right) {
    return left.x * right.x + left.y * right.y;
}
__host__ __device__ uint64_t BrendanCUDA::uint64_2::MagnatudeSquared() const {
    return x * x + y * y;
}
__host__ __device__ uint64_t BrendanCUDA::uint64_2::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ BrendanCUDA::uint64_3::uint64_3(uint64_t x, uint64_t y, uint64_t z) {
    this->x = x;
    this->y = y;
    this->z = z;
}
__host__ __device__ BrendanCUDA::uint64_3::uint64_3(uint64_t v[3]) {
    this->v[0] = v[0];
    this->v[1] = v[1];
    this->v[2] = v[2];
}
__host__ __device__ BrendanCUDA::uint64_3 BrendanCUDA::uint64_3::operator+(uint64_3 other) {
    return uint64_3(x + other.x, y + other.y, z + other.z);
}
__host__ __device__ BrendanCUDA::uint64_3 BrendanCUDA::uint64_3::operator-(uint64_3 other) {
    return uint64_3(x - other.x, y - other.y, z - other.z);
}
__host__ __device__ BrendanCUDA::uint64_3 BrendanCUDA::uint64_3::operator*(uint64_t other) {
    return uint64_3(x * other, y * other, z * other);
}
__host__ __device__ BrendanCUDA::uint64_3 BrendanCUDA::uint64_3::operator/(uint64_t other) {
    return uint64_3(x / other, y / other, z / other);
}
__host__ __device__ uint64_t BrendanCUDA::uint64_3::Dot(uint64_3 left, uint64_3 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}
__host__ __device__ uint64_t BrendanCUDA::uint64_3::MagnatudeSquared() const {
    return x * x + y * y + z * z;
}
__host__ __device__ uint64_t BrendanCUDA::uint64_3::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
__host__ __device__ BrendanCUDA::uint64_4::uint64_4(uint64_t x, uint64_t y, uint64_t z, uint64_t w) {
    this->x = x;
    this->y = y;
    this->z = z;
    this->w = w;
}
__host__ __device__ BrendanCUDA::uint64_4::uint64_4(uint64_t v[4]) {
    this->v[0] = v[0];
    this->v[1] = v[1];
    this->v[2] = v[2];
    this->v[3] = v[3];
}
__host__ __device__ BrendanCUDA::uint64_4 BrendanCUDA::uint64_4::operator+(uint64_4 other) {
    return uint64_4(x + other.x, y + other.y, z + other.z, w + other.w);
}
__host__ __device__ BrendanCUDA::uint64_4 BrendanCUDA::uint64_4::operator-(uint64_4 other) {
    return uint64_4(x - other.x, y - other.y, z - other.z, w - other.w);
}
__host__ __device__ BrendanCUDA::uint64_4 BrendanCUDA::uint64_4::operator*(uint64_t other) {
    return uint64_4(x * other, y * other, z * other, w * other);
}
__host__ __device__ BrendanCUDA::uint64_4 BrendanCUDA::uint64_4::operator/(uint64_t other) {
    return uint64_4(x / other, y / other, z / other, w / other);
}
__host__ __device__ uint64_t BrendanCUDA::uint64_4::Dot(uint64_4 left, uint64_4 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w;
}
__host__ __device__ uint64_t BrendanCUDA::uint64_4::MagnatudeSquared() const {
    return x * x + y * y + z * z + w * w;
}
__host__ __device__ uint64_t BrendanCUDA::uint64_4::Magnatude() const {
    return Math::sqrt(MagnatudeSquared());
}
