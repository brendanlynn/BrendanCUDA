#include "ai_mlpb_mlpblw.h"

__host__ __device__ uint64_t BrendanCUDA::AI::MLPB::MLPBLW::Run(uint64_t Input) const {
    switch (type) {
    case eMLPBL8T8:
        return d8t8.RunG(Input);
    case eMLPBL8T16:
        return d8t16.RunG(Input);
    case eMLPBL8T32:
        return d8t32.RunG(Input);
    case eMLPBL8T64:
        return d8t64.RunG(Input);
    case eMLPBL16T8:
        return d16t8.RunG(Input);
    case eMLPBL16T16:
        return d16t16.RunG(Input);
    case eMLPBL16T32:
        return d16t32.RunG(Input);
    case eMLPBL16T64:
        return d16t64.RunG(Input);
    case eMLPBL32T8:
        return d32t8.RunG(Input);
    case eMLPBL32T16:
        return d32t16.RunG(Input);
    case eMLPBL32T32:
        return d32t32.RunG(Input);
    case eMLPBL32T64:
        return d32t64.RunG(Input);
    case eMLPBL64T8:
        return d64t8.RunG(Input);
    case eMLPBL64T16:
        return d64t16.RunG(Input);
    case eMLPBL64T32:
        return d64t32.RunG(Input);
    case eMLPBL64T64:
        return d64t64.RunG(Input);
    default:
        return 0l;
    }
}
__host__ __device__ void BrendanCUDA::AI::MLPB::MLPBLW::Dispose() {
    switch (type) {
    case eMLPBL8T8:
        return d8t8.Dispose();
    case eMLPBL8T16:
        return d8t16.Dispose();
    case eMLPBL8T32:
        return d8t32.Dispose();
    case eMLPBL8T64:
        return d8t64.Dispose();
    case eMLPBL16T8:
        return d16t8.Dispose();
    case eMLPBL16T16:
        return d16t16.Dispose();
    case eMLPBL16T32:
        return d16t32.Dispose();
    case eMLPBL16T64:
        return d16t64.Dispose();
    case eMLPBL32T8:
        return d32t8.Dispose();
    case eMLPBL32T16:
        return d32t16.Dispose();
    case eMLPBL32T32:
        return d32t32.Dispose();
    case eMLPBL32T64:
        return d32t64.Dispose();
    case eMLPBL64T8:
        return d64t8.Dispose();
    case eMLPBL64T16:
        return d64t16.Dispose();
    case eMLPBL64T32:
        return d64t32.Dispose();
    case eMLPBL64T64:
        return d64t64.Dispose();
    default:
        return;
    }
}
__host__ __device__ auto BrendanCUDA::AI::MLPB::MLPBLW::Clone() const -> MLPBLW {
    switch (type) {
    case eMLPBL8T8:
        return MLPBLW(d8t8.Clone());
    case eMLPBL8T16:
        return MLPBLW(d8t16.Clone());
    case eMLPBL8T32:
        return MLPBLW(d8t32.Clone());
    case eMLPBL8T64:
        return MLPBLW(d8t64.Clone());
    case eMLPBL16T8:
        return MLPBLW(d16t8.Clone());
    case eMLPBL16T16:
        return MLPBLW(d16t16.Clone());
    case eMLPBL16T32:
        return MLPBLW(d16t32.Clone());
    case eMLPBL16T64:
        return MLPBLW(d16t64.Clone());
    case eMLPBL32T8:
        return MLPBLW(d32t8.Clone());
    case eMLPBL32T16:
        return MLPBLW(d32t16.Clone());
    case eMLPBL32T32:
        return MLPBLW(d32t32.Clone());
    case eMLPBL32T64:
        return MLPBLW(d32t64.Clone());
    case eMLPBL64T8:
        return MLPBLW(d64t8.Clone());
    case eMLPBL64T16:
        return MLPBLW(d64t16.Clone());
    case eMLPBL64T32:
        return MLPBLW(d64t32.Clone());
    case eMLPBL64T64:
        return MLPBLW(d64t64.Clone());
    default:
        return MLPBLW();
    }
}
size_t BrendanCUDA::AI::MLPB::MLPBLW::SerializedSize() const {
    switch (type) {
    case eMLPBL8T8:
        return sizeof(networkType_t) + d8t8.SerializedSize();
    case eMLPBL8T16:
        return sizeof(networkType_t) + d8t16.SerializedSize();
    case eMLPBL8T32:
        return sizeof(networkType_t) + d8t32.SerializedSize();
    case eMLPBL8T64:
        return sizeof(networkType_t) + d8t64.SerializedSize();
    case eMLPBL16T8:
        return sizeof(networkType_t) + d16t8.SerializedSize();
    case eMLPBL16T16:
        return sizeof(networkType_t) + d16t16.SerializedSize();
    case eMLPBL16T32:
        return sizeof(networkType_t) + d16t32.SerializedSize();
    case eMLPBL16T64:
        return sizeof(networkType_t) + d16t64.SerializedSize();
    case eMLPBL32T8:
        return sizeof(networkType_t) + d32t8.SerializedSize();
    case eMLPBL32T16:
        return sizeof(networkType_t) + d32t16.SerializedSize();
    case eMLPBL32T32:
        return sizeof(networkType_t) + d32t32.SerializedSize();
    case eMLPBL32T64:
        return sizeof(networkType_t) + d32t64.SerializedSize();
    case eMLPBL64T8:
        return sizeof(networkType_t) + d64t8.SerializedSize();
    case eMLPBL64T16:
        return sizeof(networkType_t) + d64t16.SerializedSize();
    case eMLPBL64T32:
        return sizeof(networkType_t) + d64t32.SerializedSize();
    case eMLPBL64T64:
        return sizeof(networkType_t) + d64t64.SerializedSize();
    default:
        return sizeof(networkType_t);
    }
}
void BrendanCUDA::AI::MLPB::MLPBLW::Serialize(void*& Data) const {
    *(networkType_t*)Data = type;
    Data = ((networkType_t*)Data) + 1;
    switch (type) {
    case eMLPBL8T8:
        d8t8.Serialize(Data);
        break;
    case eMLPBL8T16:
        d8t16.Serialize(Data);
        break;
    case eMLPBL8T32:
        d8t32.Serialize(Data);
        break;
    case eMLPBL8T64:
        d8t64.Serialize(Data);
        break;
    case eMLPBL16T8:
        d16t8.Serialize(Data);
        break;
    case eMLPBL16T16:
        d16t16.Serialize(Data);
        break;
    case eMLPBL16T32:
        d16t32.Serialize(Data);
        break;
    case eMLPBL16T64:
        d16t64.Serialize(Data);
        break;
    case eMLPBL32T8:
        d32t8.Serialize(Data);
        break;
    case eMLPBL32T16:
        d32t16.Serialize(Data);
        break;
    case eMLPBL32T32:
        d32t32.Serialize(Data);
        break;
    case eMLPBL32T64:
        d32t64.Serialize(Data);
        break;
    case eMLPBL64T8:
        d64t8.Serialize(Data);
        break;
    case eMLPBL64T16:
        d64t16.Serialize(Data);
        break;
    case eMLPBL64T32:
        d64t32.Serialize(Data);
        break;
    case eMLPBL64T64:
        d64t64.Serialize(Data);
        break;
    }
}
auto BrendanCUDA::AI::MLPB::MLPBLW::Deserialize(const void*& Data) -> MLPBLW {
    networkType_t type = *(networkType_t*)Data;
    Data = ((networkType_t*)Data) + 1;
    switch (type) {
    case eMLPBL8T8:
        return MLPBLW(mlpbl8T8_t::Deserialize(Data));
    case eMLPBL8T16:
        return MLPBLW(mlpbl8T16_t::Deserialize(Data));
    case eMLPBL8T32:
        return MLPBLW(mlpbl8T32_t::Deserialize(Data));
    case eMLPBL8T64:
        return MLPBLW(mlpbl8T64_t::Deserialize(Data));
    case eMLPBL16T8:
        return MLPBLW(mlpbl16T8_t::Deserialize(Data));
    case eMLPBL16T16:
        return MLPBLW(mlpbl16T16_t::Deserialize(Data));
    case eMLPBL16T32:
        return MLPBLW(mlpbl16T32_t::Deserialize(Data));
    case eMLPBL16T64:
        return MLPBLW(mlpbl16T64_t::Deserialize(Data));
    case eMLPBL32T8:
        return MLPBLW(mlpbl32T8_t::Deserialize(Data));
    case eMLPBL32T16:
        return MLPBLW(mlpbl32T16_t::Deserialize(Data));
    case eMLPBL32T32:
        return MLPBLW(mlpbl32T32_t::Deserialize(Data));
    case eMLPBL32T64:
        return MLPBLW(mlpbl32T64_t::Deserialize(Data));
    case eMLPBL64T8:
        return MLPBLW(mlpbl64T8_t::Deserialize(Data));
    case eMLPBL64T16:
        return MLPBLW(mlpbl64T16_t::Deserialize(Data));
    case eMLPBL64T32:
        return MLPBLW(mlpbl64T32_t::Deserialize(Data));
    case eMLPBL64T64:
        return MLPBLW(mlpbl64T64_t::Deserialize(Data));
    default:
        return MLPBLW();
    }
}