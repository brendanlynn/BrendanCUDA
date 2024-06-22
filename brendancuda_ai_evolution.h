#pragma once

namespace BrendanCUDA {
    namespace AI {
        namespace Evolution {
            //A function that creates an object.
            using creationFunction_t = void* (*)(void* CreationSharedData);
            //A function that evaluates an object.
            using evaluationFunction_t = float (*)(void* Object, void* EvaluationSharedData);
            //A function that reproduces an object.
            using reproductionFunction_t = void* (*)(void* Object, void* ReproductionSharedData);
            //A function that disposes of an object.
            using disposeFunction_t = void (*)(void* Object, void* DisposeSharedData);
        }
    }
}