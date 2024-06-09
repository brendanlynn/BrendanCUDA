#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "brendancuda_fixedvectors.cuh"

namespace BrendanCUDA {
    namespace Fields {
        __host__ __device__ uint32_t Coordinates32_2ToIndex32_RM(uint32_2 Dimensions, uint32_2 Coordinates);
        __host__ __device__ uint32_2 Index32ToCoordinates32_2_RM(uint32_2 Dimensions, uint32_t Index);
        __host__ __device__ uint32_t Coordinates32_2ToIndex32_CM(uint32_2 Dimensions, uint32_2 Coordinates);
        __host__ __device__ uint32_2 Index32ToCoordinates32_2_CM(uint32_2 Dimensions, uint32_t Index);
        __host__ __device__ uint32_t Coordinates32_3ToIndex32_RM(uint32_3 Dimensions, uint32_3 Coordinates);
        __host__ __device__ uint32_3 Index32ToCoordinates32_3_RM(uint32_3 Dimensions, uint32_t Index);
        __host__ __device__ uint32_t Coordinates32_3ToIndex32_CM(uint32_3 Dimensions, uint32_3 Coordinates);
        __host__ __device__ uint32_3 Index32ToCoordinates32_3_CM(uint32_3 Dimensions, uint32_t Index);
        __host__ __device__ uint32_t Coordinates32_4ToIndex32_RM(uint32_4 Dimensions, uint32_4 Coordinates);
        __host__ __device__ uint32_4 Index32ToCoordinates32_4_RM(uint32_4 Dimensions, uint32_t Index);
        __host__ __device__ uint32_t Coordinates32_4ToIndex32_CM(uint32_4 Dimensions, uint32_4 Coordinates);
        __host__ __device__ uint32_4 Index32ToCoordinates32_4_CM(uint32_4 Dimensions, uint32_t Index);
        __host__ __device__ uint64_t Coordinates32_2ToIndex64_RM(uint32_2 Dimensions, uint32_2 Coordinates);
        __host__ __device__ uint32_2 Index64ToCoordinates32_2_RM(uint32_2 Dimensions, uint64_t Index);
        __host__ __device__ uint64_t Coordinates32_2ToIndex64_CM(uint32_2 Dimensions, uint32_2 Coordinates);
        __host__ __device__ uint32_2 Index64ToCoordinates32_2_CM(uint32_2 Dimensions, uint64_t Index);
        __host__ __device__ uint64_t Coordinates32_3ToIndex64_RM(uint32_3 Dimensions, uint32_3 Coordinates);
        __host__ __device__ uint32_3 Index64ToCoordinates32_3_RM(uint32_3 Dimensions, uint64_t Index);
        __host__ __device__ uint64_t Coordinates32_3ToIndex64_CM(uint32_3 Dimensions, uint32_3 Coordinates);
        __host__ __device__ uint32_3 Index64ToCoordinates32_3_CM(uint32_3 Dimensions, uint64_t Index);
        __host__ __device__ uint64_t Coordinates32_4ToIndex64_RM(uint32_4 Dimensions, uint32_4 Coordinates);
        __host__ __device__ uint32_4 Index64ToCoordinates32_4_RM(uint32_4 Dimensions, uint64_t Index);
        __host__ __device__ uint64_t Coordinates32_4ToIndex64_CM(uint32_4 Dimensions, uint32_4 Coordinates);
        __host__ __device__ uint32_4 Index64ToCoordinates32_4_CM(uint32_4 Dimensions, uint64_t Index);
        __host__ __device__ uint32_t Coordinates64_2ToIndex32_RM(uint64_2 Dimensions, uint64_2 Coordinates);
        __host__ __device__ uint64_2 Index32ToCoordinates64_2_RM(uint64_2 Dimensions, uint32_t Index);
        __host__ __device__ uint32_t Coordinates64_2ToIndex32_CM(uint64_2 Dimensions, uint64_2 Coordinates);
        __host__ __device__ uint64_2 Index32ToCoordinates64_2_CM(uint64_2 Dimensions, uint32_t Index);
        __host__ __device__ uint32_t Coordinates64_3ToIndex32_RM(uint64_3 Dimensions, uint64_3 Coordinates);
        __host__ __device__ uint64_3 Index32ToCoordinates64_3_RM(uint64_3 Dimensions, uint32_t Index);
        __host__ __device__ uint32_t Coordinates64_3ToIndex32_CM(uint64_3 Dimensions, uint64_3 Coordinates);
        __host__ __device__ uint64_3 Index32ToCoordinates64_3_CM(uint64_3 Dimensions, uint32_t Index);
        __host__ __device__ uint32_t Coordinates64_4ToIndex32_RM(uint64_4 Dimensions, uint64_4 Coordinates);
        __host__ __device__ uint64_4 Index32ToCoordinates64_4_RM(uint64_4 Dimensions, uint32_t Index);
        __host__ __device__ uint32_t Coordinates64_4ToIndex32_CM(uint64_4 Dimensions, uint64_4 Coordinates);
        __host__ __device__ uint64_4 Index32ToCoordinates64_4_CM(uint64_4 Dimensions, uint32_t Index);
        __host__ __device__ uint64_t Coordinates64_2ToIndex64_RM(uint64_2 Dimensions, uint64_2 Coordinates);
        __host__ __device__ uint64_2 Index64ToCoordinates64_2_RM(uint64_2 Dimensions, uint64_t Index);
        __host__ __device__ uint64_t Coordinates64_2ToIndex64_CM(uint64_2 Dimensions, uint64_2 Coordinates);
        __host__ __device__ uint64_2 Index64ToCoordinates64_2_CM(uint64_2 Dimensions, uint64_t Index);
        __host__ __device__ uint64_t Coordinates64_3ToIndex64_RM(uint64_3 Dimensions, uint64_3 Coordinates);
        __host__ __device__ uint64_3 Index64ToCoordinates64_3_RM(uint64_3 Dimensions, uint64_t Index);
        __host__ __device__ uint64_t Coordinates64_3ToIndex64_CM(uint64_3 Dimensions, uint64_3 Coordinates);
        __host__ __device__ uint64_3 Index64ToCoordinates64_3_CM(uint64_3 Dimensions, uint64_t Index);
        __host__ __device__ uint64_t Coordinates64_4ToIndex64_RM(uint64_4 Dimensions, uint64_4 Coordinates);
        __host__ __device__ uint64_4 Index64ToCoordinates64_4_RM(uint64_4 Dimensions, uint64_t Index);
        __host__ __device__ uint64_t Coordinates64_4ToIndex64_CM(uint64_4 Dimensions, uint64_4 Coordinates);
        __host__ __device__ uint64_4 Index64ToCoordinates64_4_CM(uint64_4 Dimensions, uint64_t Index);
        __host__ __device__ void GetConsecutives(uint32_3 Dimensions, uint32_t Index, uint32_t& POO, uint32_t& NOO, uint32_t& OPO, uint32_t& ONO, uint32_t& OOP, uint32_t& OON);
        __host__ __device__ void GetConsecutives(uint32_3 Dimensions, uint32_t Index, uint32_t& PPP, uint32_t& OPP, uint32_t& NPP, uint32_t& POP, uint32_t& OOP, uint32_t& NOP, uint32_t& PNP, uint32_t& ONP, uint32_t& NNP, uint32_t& PPO, uint32_t& OPO, uint32_t& NPO, uint32_t& POO, uint32_t& NOO, uint32_t& PNO, uint32_t& ONO, uint32_t& NNO, uint32_t& PPN, uint32_t& OPN, uint32_t& NPN, uint32_t& PON, uint32_t& OON, uint32_t& NON, uint32_t& PNN, uint32_t& ONN, uint32_t& NNN);
        __host__ __device__ void GetConsecutives(uint32_3 Dimensions, uint32_3 Coordinates, uint32_3& POO, uint32_3& NOO, uint32_3& OPO, uint32_3& ONO, uint32_3& OOP, uint32_3& OON);
        __host__ __device__ void GetConsecutives(uint32_3 Dimensions, uint32_3 Coordinates, uint32_3& PPP, uint32_3& OPP, uint32_3& NPP, uint32_3& POP, uint32_3& OOP, uint32_3& NOP, uint32_3& PNP, uint32_3& ONP, uint32_3& NNP, uint32_3& PPO, uint32_3& OPO, uint32_3& NPO, uint32_3& POO, uint32_3& NOO, uint32_3& PNO, uint32_3& ONO, uint32_3& NNO, uint32_3& PPN, uint32_3& OPN, uint32_3& NPN, uint32_3& PON, uint32_3& OON, uint32_3& NON, uint32_3& PNN, uint32_3& ONN, uint32_3& NNN);
        __host__ __device__ void GetConsecutives(uint32_3 Dimensions, uint32_t Index, uint32_3& POO, uint32_3& NOO, uint32_3& OPO, uint32_3& ONO, uint32_3& OOP, uint32_3& OON);
        __host__ __device__ void GetConsecutives(uint32_3 Dimensions, uint32_t Index, uint32_3& PPP, uint32_3& OPP, uint32_3& NPP, uint32_3& POP, uint32_3& OOP, uint32_3& NOP, uint32_3& PNP, uint32_3& ONP, uint32_3& NNP, uint32_3& PPO, uint32_3& OPO, uint32_3& NPO, uint32_3& POO, uint32_3& NOO, uint32_3& PNO, uint32_3& ONO, uint32_3& NNO, uint32_3& PPN, uint32_3& OPN, uint32_3& NPN, uint32_3& PON, uint32_3& OON, uint32_3& NON, uint32_3& PNN, uint32_3& ONN, uint32_3& NNN);
        __host__ __device__ void GetConsecutives(uint32_3 Dimensions, uint32_3 Coordinates, uint32_t& POO, uint32_t& NOO, uint32_t& OPO, uint32_t& ONO, uint32_t& OOP, uint32_t& OON);
        __host__ __device__ void GetConsecutives(uint32_3 Dimensions, uint32_3 Coordinates, uint32_t& PPP, uint32_t& OPP, uint32_t& NPP, uint32_t& POP, uint32_t& OOP, uint32_t& NOP, uint32_t& PNP, uint32_t& ONP, uint32_t& NNP, uint32_t& PPO, uint32_t& OPO, uint32_t& NPO, uint32_t& POO, uint32_t& NOO, uint32_t& PNO, uint32_t& ONO, uint32_t& NNO, uint32_t& PPN, uint32_t& OPN, uint32_t& NPN, uint32_t& PON, uint32_t& OON, uint32_t& NON, uint32_t& PNN, uint32_t& ONN, uint32_t& NNN);
        __host__ __device__ void GetConsecutives(uint64_3 Dimensions, uint32_t Index, uint64_3& POO, uint64_3& NOO, uint64_3& OPO, uint64_3& ONO, uint64_3& OOP, uint64_3& OON);
        __host__ __device__ void GetConsecutives(uint64_3 Dimensions, uint32_t Index, uint64_3& PPP, uint64_3& OPP, uint64_3& NPP, uint64_3& POP, uint64_3& OOP, uint64_3& NOP, uint64_3& PNP, uint64_3& ONP, uint64_3& NNP, uint64_3& PPO, uint64_3& OPO, uint64_3& NPO, uint64_3& POO, uint64_3& NOO, uint64_3& PNO, uint64_3& ONO, uint64_3& NNO, uint64_3& PPN, uint64_3& OPN, uint64_3& NPN, uint64_3& PON, uint64_3& OON, uint64_3& NON, uint64_3& PNN, uint64_3& ONN, uint64_3& NNN);
        __host__ __device__ void GetConsecutives(uint64_3 Dimensions, uint64_3 Coordinates, uint32_t& POO, uint32_t& NOO, uint32_t& OPO, uint32_t& ONO, uint32_t& OOP, uint32_t& OON);
        __host__ __device__ void GetConsecutives(uint64_3 Dimensions, uint64_3 Coordinates, uint32_t& PPP, uint32_t& OPP, uint32_t& NPP, uint32_t& POP, uint32_t& OOP, uint32_t& NOP, uint32_t& PNP, uint32_t& ONP, uint32_t& NNP, uint32_t& PPO, uint32_t& OPO, uint32_t& NPO, uint32_t& POO, uint32_t& NOO, uint32_t& PNO, uint32_t& ONO, uint32_t& NNO, uint32_t& PPN, uint32_t& OPN, uint32_t& NPN, uint32_t& PON, uint32_t& OON, uint32_t& NON, uint32_t& PNN, uint32_t& ONN, uint32_t& NNN);
        __host__ __device__ void GetConsecutives(uint64_3 Dimensions, uint64_t Index, uint64_t& POO, uint64_t& NOO, uint64_t& OPO, uint64_t& ONO, uint64_t& OOP, uint64_t& OON);
        __host__ __device__ void GetConsecutives(uint64_3 Dimensions, uint64_t Index, uint64_t& PPP, uint64_t& OPP, uint64_t& NPP, uint64_t& POP, uint64_t& OOP, uint64_t& NOP, uint64_t& PNP, uint64_t& ONP, uint64_t& NNP, uint64_t& PPO, uint64_t& OPO, uint64_t& NPO, uint64_t& POO, uint64_t& NOO, uint64_t& PNO, uint64_t& ONO, uint64_t& NNO, uint64_t& PPN, uint64_t& OPN, uint64_t& NPN, uint64_t& PON, uint64_t& OON, uint64_t& NON, uint64_t& PNN, uint64_t& ONN, uint64_t& NNN);
        __host__ __device__ void GetConsecutives(uint64_3 Dimensions, uint64_3 Coordinates, uint64_3& POO, uint64_3& NOO, uint64_3& OPO, uint64_3& ONO, uint64_3& OOP, uint64_3& OON);
        __host__ __device__ void GetConsecutives(uint64_3 Dimensions, uint64_3 Coordinates, uint64_3& PPP, uint64_3& OPP, uint64_3& NPP, uint64_3& POP, uint64_3& OOP, uint64_3& NOP, uint64_3& PNP, uint64_3& ONP, uint64_3& NNP, uint64_3& PPO, uint64_3& OPO, uint64_3& NPO, uint64_3& POO, uint64_3& NOO, uint64_3& PNO, uint64_3& ONO, uint64_3& NNO, uint64_3& PPN, uint64_3& OPN, uint64_3& NPN, uint64_3& PON, uint64_3& OON, uint64_3& NON, uint64_3& PNN, uint64_3& ONN, uint64_3& NNN);
        __host__ __device__ void GetConsecutives(uint32_3 Dimensions, uint64_t Index, uint32_3& POO, uint32_3& NOO, uint32_3& OPO, uint32_3& ONO, uint32_3& OOP, uint32_3& OON);
        __host__ __device__ void GetConsecutives(uint32_3 Dimensions, uint64_t Index, uint32_3& PPP, uint32_3& OPP, uint32_3& NPP, uint32_3& POP, uint32_3& OOP, uint32_3& NOP, uint32_3& PNP, uint32_3& ONP, uint32_3& NNP, uint32_3& PPO, uint32_3& OPO, uint32_3& NPO, uint32_3& POO, uint32_3& NOO, uint32_3& PNO, uint32_3& ONO, uint32_3& NNO, uint32_3& PPN, uint32_3& OPN, uint32_3& NPN, uint32_3& PON, uint32_3& OON, uint32_3& NON, uint32_3& PNN, uint32_3& ONN, uint32_3& NNN);
        __host__ __device__ void GetConsecutives(uint32_3 Dimensions, uint32_3 Coordinates, uint64_t& POO, uint64_t& NOO, uint64_t& OPO, uint64_t& ONO, uint64_t& OOP, uint64_t& OON);
        __host__ __device__ void GetConsecutives(uint32_3 Dimensions, uint32_3 Coordinates, uint64_t& PPP, uint64_t& OPP, uint64_t& NPP, uint64_t& POP, uint64_t& OOP, uint64_t& NOP, uint64_t& PNP, uint64_t& ONP, uint64_t& NNP, uint64_t& PPO, uint64_t& OPO, uint64_t& NPO, uint64_t& POO, uint64_t& NOO, uint64_t& PNO, uint64_t& ONO, uint64_t& NNO, uint64_t& PPN, uint64_t& OPN, uint64_t& NPN, uint64_t& PON, uint64_t& OON, uint64_t& NON, uint64_t& PNN, uint64_t& ONN, uint64_t& NNN);
        __host__ __device__ void GetConsecutives(uint64_3 Dimensions, uint64_t Index, uint64_3& POO, uint64_3& NOO, uint64_3& OPO, uint64_3& ONO, uint64_3& OOP, uint64_3& OON);
        __host__ __device__ void GetConsecutives(uint64_3 Dimensions, uint64_t Index, uint64_3& PPP, uint64_3& OPP, uint64_3& NPP, uint64_3& POP, uint64_3& OOP, uint64_3& NOP, uint64_3& PNP, uint64_3& ONP, uint64_3& NNP, uint64_3& PPO, uint64_3& OPO, uint64_3& NPO, uint64_3& POO, uint64_3& NOO, uint64_3& PNO, uint64_3& ONO, uint64_3& NNO, uint64_3& PPN, uint64_3& OPN, uint64_3& NPN, uint64_3& PON, uint64_3& OON, uint64_3& NON, uint64_3& PNN, uint64_3& ONN, uint64_3& NNN);
        __host__ __device__ void GetConsecutives(uint64_3 Dimensions, uint64_3 Coordinates, uint64_t& POO, uint64_t& NOO, uint64_t& OPO, uint64_t& ONO, uint64_t& OOP, uint64_t& OON);
        __host__ __device__ void GetConsecutives(uint64_3 Dimensions, uint64_3 Coordinates, uint64_t& PPP, uint64_t& OPP, uint64_t& NPP, uint64_t& POP, uint64_t& OOP, uint64_t& NOP, uint64_t& PNP, uint64_t& ONP, uint64_t& NNP, uint64_t& PPO, uint64_t& OPO, uint64_t& NPO, uint64_t& POO, uint64_t& NOO, uint64_t& PNO, uint64_t& ONO, uint64_t& NNO, uint64_t& PPN, uint64_t& OPN, uint64_t& NPN, uint64_t& PON, uint64_t& OON, uint64_t& NON, uint64_t& PNN, uint64_t& ONN, uint64_t& NNN);
    }
}