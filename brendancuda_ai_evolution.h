#pragma once

namespace BrendanCUDA {
    namespace AI {
        namespace Evolution {
            using creationFunction_t = void* (*)(void* CreationSharedData);
            using evaluationFunction_t = float (*)(void* Object, void* EvaluationSharedData);
            using reproductionFunction_t = void* (*)(void* Object, void* ReproductionSharedData);
            using disposeFunction_t = void (*)(void* Object, void* DisposeSharedData);
        }
    }
}