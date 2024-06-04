#include "brendancuda_random_rngfunc.cuh"

template <typename T>
__host__ __device__ BrendanCUDA::Random::rngWState<T>::rngWState(rngFunc_t<T> func, void* state) {
    this->func = func;
    this->state = state;
}
template <typename T>
__host__ __device__ T BrendanCUDA::Random::rngWState<T>::Run() const {
    return func(state);
}

__device__ uint64_t getI64FromDeviceRandom(void* state) {
    return ((BrendanCUDA::DeviceRandom*)state)->GetI64();
}
__device__ BrendanCUDA::Random::rngWState<uint64_t> BrendanCUDA::Random::rngWState64_FromDeviceRandom(DeviceRandom* dr) {
    return rngWState<uint64_t>(getI64FromDeviceRandom, dr);
}
__device__ uint32_t getI32FromDeviceRandom(void* state) {
    return ((BrendanCUDA::DeviceRandom*)state)->GetI32();
}
__device__ BrendanCUDA::Random::rngWState<uint32_t> BrendanCUDA::Random::rngWState32_FromDeviceRandom(DeviceRandom* dr) {
    return rngWState<uint32_t>(getI32FromDeviceRandom, dr);
}
uint64_t getI64From_mt19937(void* state) {
    std::uniform_int_distribution<uint64_t> dis(0, std::numeric_limits<uint64_t>::max());
    return dis(*(std::mt19937*)state);
}
BrendanCUDA::Random::rngWState<uint64_t> BrendanCUDA::Random::rngWState64_From_mt19937(std::mt19937* dr) {
    return rngWState<uint64_t>(getI64From_mt19937, dr);
}
uint32_t getI32From_mt19937(void* state) {
    std::uniform_int_distribution<uint32_t> dis(0, std::numeric_limits<uint32_t>::max());
    return dis(*(std::mt19937*)state);
}
BrendanCUDA::Random::rngWState<uint32_t> BrendanCUDA::Random::rngWState32_From_mt19937(std::mt19937* dr) {
    return rngWState<uint32_t>(getI32From_mt19937, dr);
}
uint64_t getI64From_mt19937_64(void* state) {
    std::uniform_int_distribution<uint64_t> dis(0, std::numeric_limits<uint64_t>::max());
    return dis(*(std::mt19937_64*)state);
}
BrendanCUDA::Random::rngWState<uint64_t> BrendanCUDA::Random::rngWState64_From_mt19937_64(std::mt19937_64* dr) {
    return rngWState<uint64_t>(getI64From_mt19937_64, dr);
}
uint32_t getI32From_mt19937_64(void* state) {
    std::uniform_int_distribution<uint32_t> dis(0, std::numeric_limits<uint32_t>::max());
    return dis(*(std::mt19937_64*)state);
}
BrendanCUDA::Random::rngWState<uint32_t> BrendanCUDA::Random::rngWState32_From_mt19937_64(std::mt19937_64* dr) {
    return rngWState<uint32_t>(getI32From_mt19937_64, dr);
}

template BrendanCUDA::Random::rngWState<float>;
template BrendanCUDA::Random::rngWState<double>;
template BrendanCUDA::Random::rngWState<int8_t>;
template BrendanCUDA::Random::rngWState<uint8_t>;
template BrendanCUDA::Random::rngWState<int16_t>;
template BrendanCUDA::Random::rngWState<uint16_t>;
template BrendanCUDA::Random::rngWState<int32_t>;
template BrendanCUDA::Random::rngWState<uint32_t>;
template BrendanCUDA::Random::rngWState<int64_t>;
template BrendanCUDA::Random::rngWState<uint64_t>;

__host__ __device__ BrendanCUDA::Random::rngWStateA::rngWStateA(rngFunc_t<float> funcF, rngFunc_t<double> funcD, rngFunc_t<uint8_t> func8, rngFunc_t<uint16_t> func16, rngFunc_t<uint32_t> func32, rngFunc_t<uint64_t> func64, void* state) {
    this->funcF = funcF;
    this->funcD = funcD;
    this->func8 = func8;
    this->func16 = func16;
    this->func32 = func32;
    this->func64 = func64;
    this->state = state;
}
__host__ __device__ float BrendanCUDA::Random::rngWStateA::RunF() {
    return funcF(state);
}
__host__ __device__ double BrendanCUDA::Random::rngWStateA::RunD() {
    return funcD(state);
}
__host__ __device__ uint8_t BrendanCUDA::Random::rngWStateA::Run8() {
    return func8(state);
}
__host__ __device__ uint16_t BrendanCUDA::Random::rngWStateA::Run16() {
    return func16(state);
}
__host__ __device__ uint32_t BrendanCUDA::Random::rngWStateA::Run32() {
    return func32(state);
}
__host__ __device__ uint64_t BrendanCUDA::Random::rngWStateA::Run64() {
    return func64(state);
}