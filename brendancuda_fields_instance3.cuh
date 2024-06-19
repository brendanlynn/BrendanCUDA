#pragma once

#include "brendancuda_fields_dfield3.cuh"
#include "brendancuda_random_anyrng.cuh"
#include <tuple>
#include "brendancuda_random_sseed.cuh"

namespace BrendanCUDA {
    namespace details {
        template <typename _T>
        struct FieldInstance3_CurrentInstance final {
            Fields::DField3<_T>& field;
            uint32_3* inputs;
            uint32_3* outputs;
            void* obj;
            void* objectRunner_sharedData;
            Random::AnyRNG<uint32_t> rng;
        };
    }
    namespace Fields {
        template <typename _T>
        using fieldInstance3_objectRunner_t = void(*)(void* Object, DField3<_T> Field, void* SharedData);
        template <typename _T>
        using fieldInstance3_createField_t = DField3<_T>*(*)(void* SharedData);
        template <typename _T>
        struct FieldInstance3_Construct_Settings final {
            Random::AnyRNG<uint32_t> rng;
            void* objectRunner_sharedData;
            void* createField_sharedData;
        };
        template <typename _T, fieldInstance3_createField_t<_T> _CreateField>
        void* FieldInstance3_Construct(void* Object, void* Settings);
        template <typename _T, fieldInstance3_objectRunner_t<_T> _ObjRunner>
        _T* FieldInstance3_Iterate(void* CurrentInstance, _T* Inputs);
        template <typename _T>
        void FieldInstance3_Destruct(void* CurrentInstance);
    }
}

template <typename _T, BrendanCUDA::Fields::fieldInstance3_createField_t<_T> _CreateField>
void* BrendanCUDA::Fields::FieldInstance3_Construct(void* Object, void* Settings) {
    FieldInstance3_Construct_Settings<_T> settings = *(FieldInstance3_Construct_Settings<_T>*)Settings;
    Random::AnyRNG<uint32_t>& rng = settings.rng;

    DField3<_T>& f = *_CreateField(settings.createField_sharedData);
    details::FieldInstance3_CurrentInstance<_T>* p_rv = new details::FieldInstance3_CurrentInstance<_T>{ f, 0, 0, Object, settings.objectRunner_sharedData, rng };
    uint32_3*& il = p_rv->inputs;
    uint32_3*& ol = p_rv->outputs;

    std::uniform_int_distribution<uint64_t> disU64(0, std::numeric_limits<uint64_t>::max());

    std::uniform_int_distribution<uint32_t> disX(0, f.LengthX() - 1);
    std::uniform_int_distribution<uint32_t> disY(0, f.LengthY() - 1);
    std::uniform_int_distribution<uint32_t> disZ(0, f.LengthZ() - 1);

    il = new uint32_3[fieldOutputLength];
    for (size_t i = 0; i < fieldOutputLength; ++i) {
        il[i] = uint32_3{ disX(rng), disY(rng), disZ(rng) };
    }

    ol = new uint32_3[fieldOutputLength];
    for (size_t i = 0; i < fieldOutputLength; ++i) {
        ol[i] = uint32_3{ disX(rng), disY(rng), disZ(rng) };
    }

    return p_rv;
};
template <typename _T, BrendanCUDA::Fields::fieldInstance3_objectRunner_t<_T> _ObjRunner>
_T* BrendanCUDA::Fields::FieldInstance3_Iterate(void* CurrentInstance, _T* Inputs) {
    details::FieldInstance3_CurrentInstance<_T> c = *(details::FieldInstance3_CurrentInstance<_T>*)CurrentInstance;
    DField3<uint64_t>& f = c.field;
    uint32_3* il = c.inputs;
    uint32_3* ol = c.outputs;

    if (Inputs) {
        for (size_t i = 0; i < fieldInputLength; ++i) {
            f.FFront().SetValueAt(il[i], Inputs[i]);
        }
    }
    _ObjRunner(c.obj, f, c.objectRunner_sharedData);
    _T* opts = new _T[fieldOutputLength];
    for (size_t i = 0; i < fieldOutputLength; ++i) {
        opts[i] = f.FFront().GetValueAt(ol[i]);
    }
    return opts;
};
template <typename _T>
void BrendanCUDA::Fields::FieldInstance3_Destruct(void* CurrentInstance) {
    details::FieldInstance3_CurrentInstance<_T>* p_c = (details::FieldInstance3_CurrentInstance<_T>*)CurrentInstance;
    details::FieldInstance3_CurrentInstance<_T> c = *p_c;
    DField3<_T>& f = c.field;

    f.Dispose();
    delete (&f);
    delete[] c.inputs;
    delete[] c.outputs;
    delete p_c;
};