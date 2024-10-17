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
                        inline constexpr InstanceFunctions()
                            : constructInstance(0), iterateInstance(0), destructInstance(0) { }
                    };

                    template <typename _TInput, typename _TOutput, constructInstance_t _CI, iterateInstance_t<_TInput, _TOutput> _II, destructInstance_t _DI>
                    class Instance_C final {
                    public:
                        inline Instance_C(void* Object, void* ConstructInstanceSharedData)
                            : is(_CI(Object, ConstructInstanceSharedData)) { }
                        inline _TOutput IterateInstance(_TInput Input) {
                            return _II(is, Input);
                        }
                        inline void DestroyInstance() {
                            _DI(is);
                        }
                    private:
                        void* is;
                    };

                    template <typename _TInput, typename _TOutput>
                    class Instance_V final {
                    public:
                        inline Instance_V(constructInstance_t CI, iterateInstance_t<_TInput, _TOutput> II, destructInstance_t DI, void* Object, void* ConstructInstanceSharedData)
                            : ci(CI), ii(II), di(DI), is(CI(Object, ConstructInstanceSharedData)) { }
                        inline Instance_V(InstanceFunctions<_TInput, _TOutput> InstanceFunctions, void* Object, void* ConstructInstanceSharedData)
                            : ci(InstanceFunctions.constructInstance), ii(InstanceFunctions.iterateInstance), di(InstanceFunctions.destructInstance), is(InstanceFunctions.constructInstance(Object, ConstructInstanceSharedData)) { }
                        inline _TOutput IterateInstance(_TInput Input) {
                            return ii(is, Input);
                        }
                        inline void DestroyInstance() {
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