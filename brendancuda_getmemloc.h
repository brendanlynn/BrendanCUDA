#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "brendancuda_errorhelp.h"

namespace BrendanCUDA {
    enum MemoryLocationBasic {
        MemoryLocationNull,
        MemoryLocationHost,
        MemoryLocationDevice,
        MemoryLocationBoth,
        MemoryLocationUnknown
    };
    __forceinline MemoryLocationBasic GetMemLocAssumeHostIfNotDevice(const void* Pointer) {
        return GetMemLocAssumeHostIfNotDevice(const_cast<void*>(Pointer));
    }
    __forceinline MemoryLocationBasic GetMemLocAssumeHostIfNotDevice(void* Pointer) {
        if (!Pointer) return MemoryLocationNull;
        ThrowIfBad(cudaGetLastError());
        uint8_t x;
        cudaError_t e = cudaMemcpy(&x, Pointer, 1, cudaMemcpyDeviceToHost);
        if (e) {
            cudaGetLastError();
            return MemoryLocationHost;
        }
        return MemoryLocationDevice;
    }
    __forceinline MemoryLocationBasic GetMemLocUnknownIfNotDevice(const void* Pointer) {
        return GetMemLocUnknownIfNotDevice(const_cast<void*>(Pointer));
    }
    __forceinline MemoryLocationBasic GetMemLocUnknownIfNotDevice(void* Pointer) {
        if (!Pointer) return MemoryLocationNull;
        ThrowIfBad(cudaGetLastError());
        uint8_t x;
        cudaError_t e = cudaMemcpy(&x, Pointer, 1, cudaMemcpyDeviceToHost);
        if (e) {
            cudaGetLastError();
            return MemoryLocationUnknown;
        }
        return MemoryLocationDevice;
    }
    __forceinline MemoryLocationBasic GetMemLocAssumeDeviceIfNotHost(const void* Pointer) {
        return GetMemLocAssumeDeviceIfNotHost(const_cast<void*>(Pointer));
    }
    __forceinline MemoryLocationBasic GetMemLocAssumeDeviceIfNotHost(void* Pointer) {
        if (!Pointer) return MemoryLocationNull;
        ThrowIfBad(cudaGetLastError());
        cudaError_t e = cudaMemcpy(Pointer, Pointer, 1, cudaMemcpyHostToHost);
        if (e) {
            cudaGetLastError();
            return MemoryLocationDevice;
        }
        return MemoryLocationHost;
    }
    __forceinline MemoryLocationBasic GetMemLocUnknownIfNotHost(const void* Pointer) {
        return GetMemLocUnknownIfNotHost(const_cast<void*>(Pointer));
    }
    __forceinline MemoryLocationBasic GetMemLocUnknownIfNotHost(void* Pointer) {
        if (!Pointer) return MemoryLocationNull;
        ThrowIfBad(cudaGetLastError());
        uint8_t x;
        cudaError_t e = cudaMemcpy(&x, Pointer, 1, cudaMemcpyHostToHost);
        if (e) {
            cudaGetLastError();
            return MemoryLocationUnknown;
        }
        return MemoryLocationHost;
    }
    __forceinline MemoryLocationBasic GetMemLoc(const void* Pointer) {
        return GetMemLoc(const_cast<void*>(Pointer));
    }
    __forceinline MemoryLocationBasic GetMemLoc(void* Pointer) {
        if (!Pointer) return MemoryLocationNull;
        ThrowIfBad(cudaGetLastError());

        uint8_t x;

        bool host;
        cudaError_t e = cudaMemcpy(&x, Pointer, 1, cudaMemcpyHostToHost);
        if (e) {
            cudaGetLastError();
            host = false;
        }
        host = true;

        bool device;
        e = cudaMemcpy(&x, Pointer, 1, cudaMemcpyDeviceToHost);
        if (e) {
            cudaGetLastError();
            device = false;
        }
        device = true;

        if (host)
            if (device) return MemoryLocationBoth;
            else return MemoryLocationHost;
        else
            if (device) return MemoryLocationDevice;
            else return MemoryLocationUnknown;
    }
}