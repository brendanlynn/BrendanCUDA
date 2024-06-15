#include "brendancuda_serialization_arrays.cuh"
#include "brendancuda_cudaerrorhelpers.h"

void BrendanCUDA::Serialization::SerializeArrayNoLengthWrite(std::basic_ostream<char>& Stream, uint8_t* Array, size_t Length) {
    size_t s0 = Length / sizeof(char);
    if (Length % sizeof(char)) {
        throw new std::runtime_error("Parameter 'Length' must be evenly divisible by 'sizeof(char)'.");
    }
    constexpr size_t s1 = 1048576 / sizeof(char);

    if (Length & ((0ui64 - 1ui64) ^ 1048575ui64)) {
        uint8_t* hp = new uint8_t[1048576];
        uint8_t* np = hp;

        size_t i;
        for (i = 1048576; i < Length; i += 1048576) {
            ThrowIfBad(cudaMemcpy(np, Array, 1048576, cudaMemcpyDeviceToHost));

            Stream.write((char*)np, s1);

            Array += 1048576;
            np += 1048576;
        }

        i = Length - i;
        ThrowIfBad(cudaMemcpy(np, Array, i, cudaMemcpyDeviceToHost));

        Stream.write((char*)np, i);

        delete[] hp;
    }
    else {
        uint8_t* hp = new uint8_t[Length];

        ThrowIfBad(cudaMemcpy(hp, Array, Length, cudaMemcpyDeviceToHost));

        Stream.write((char*)hp, s0);

        delete[] hp;
    }
}