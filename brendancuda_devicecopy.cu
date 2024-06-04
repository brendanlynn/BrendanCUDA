#include "brendancuda_devicecopy.cuh"

__device__ void deviceMemcpy(void* Destination, const void* Source, size_t Count) {
    size_t c = Count;
    size_t dc = c >> 2;
    size_t rc = c & 3;
    const uint32_t* sa = reinterpret_cast<const uint32_t*>(Source);
    uint32_t* da = reinterpret_cast<uint32_t*>(Destination);
    for (size_t i = 0; i < dc; ++i) {
        da[i] = sa[i];
    }
    const uint8_t* sar = reinterpret_cast<const uint8_t*>(&sa[dc]);
    uint8_t* dar = reinterpret_cast<uint8_t*>(&da[dc]);
    switch (rc)
    {
    case 1:
        dar[0] = sar[0];
        break;
    case 2:
        dar[0] = sar[0];
        dar[1] = sar[1];
        break;
    case 3:
        dar[0] = sar[0];
        dar[1] = sar[1];
        dar[2] = sar[2];
        break;
    default:
        break;
    }
}