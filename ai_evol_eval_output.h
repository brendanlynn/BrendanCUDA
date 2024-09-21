#pragma once

#include <cstdint>
#include <utility>

namespace bcuda {
    namespace ai {
        namespace evol {
            namespace eval {
                namespace output {
                    using constructInstance_t = void*(*)(void* Object, void* ConstructInstanceSharedData);
                    
                    template <typename _TInput, typename _TOutput>
                    using iterateInstance_t = _TOutput(*)(void* CurrentInstance, _TInput Input);

                    using destructInstance_t = void(*)(void* CurrentInstance);

                    template <typename _TInput, typename _TOutput>
                    struct InstanceFunctions final {
                        constructInstance_t constructInstance;
                        iterateInstance_t<_TInput, _TOutput> iterateInstance;
                        destructInstance_t destructInstance;
                        constexpr InstanceFunctions() = default;
                    };

                    template <typename _TInput, typename _TOutput, constructInstance_t _CI, iterateInstance_t<_TInput, _TOutput> _II, destructInstance_t _DI>
                    class Instance_C final {
                    public:
                        Instance_C(void* Object, void* ConstructInstanceSharedData) {
                            is = _CI(Object, ConstructInstanceSharedData);
                        }
                        _TOutput IterateInstance(_TInput Input) {
                            return _II(is, Input);
                        }
                        void DestroyInstance() {
                            _DI(is);
                        }
                    private:
                        void* is;
                    };

                    template <typename _TInput, typename _TOutput>
                    class Instance_V final {
                    public:
                        Instance_V(constructInstance_t CI, iterateInstance_t<_TInput, _TOutput> II, destructInstance_t DI, void* Object, void* ConstructInstanceSharedData) {
                            this->ci = CI;
                            this->ii = II;
                            this->di = DI;
                            
                            is = CI(Object, ConstructInstanceSharedData);
                        }
                        Instance_V(InstanceFunctions<_TInput, _TOutput> InstanceFunctions, void* Object, void* ConstructInstanceSharedData) {
                            this->ci = InstanceFunctions.constructInstance;
                            this->ii = InstanceFunctions.iterateInstance;
                            this->di = InstanceFunctions.destructInstance;

                            is = ci(Object, ConstructInstanceSharedData);
                        }
                        _TOutput IterateInstance(_TInput Input) {
                            return ii(is, Input);
                        }
                        void DestroyInstance() {
                            di(is);
                        }
                    private:
                        constructInstance_t ci;
                        iterateInstance_t<_TInput, _TOutput> ii;
                        destructInstance_t di;
                        void* is;
                    };
                }
            }
        }
    }
}