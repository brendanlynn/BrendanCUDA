#pragma once

#include <cstdint>
#include <utility>

namespace BrendanCUDA {
    namespace AI {
        namespace Evolution {
            namespace Evaluation {
                namespace Output {
                    using constructInstance_t = void*(*)(void* Object, void* ConstructInstanceSharedData);
                    
                    template <typename _T>
                    using iterateInstance_t = _T*(*)(void* CurrentInstance, _T* Inputs);

                    using destructInstance_t = void(*)(void* CurrentInstance);

                    template <typename _T>
                    struct InstanceFunctions final {
                        constructInstance_t constructInstance;
                        iterateInstance_t<_T> iterateInstance;
                        destructInstance_t destructInstance;
                        constexpr InstanceFunctions() = default;
                    };

                    template <typename _T, constructInstance_t _CI, iterateInstance_t<_T> _II, destructInstance_t _DI>
                    class Instance_C final {
                    public:
                        Instance_C(void* Object, void* ConstructInstanceSharedData) {
                            is = _CI(Object, ConstructInstanceSharedData);
                        }
                        _T* IterateInstance(_T* Inputs) {
                            return _II(is, Inputs);
                        }
                        void DestroyInstance() {
                            _DI(is);
                        }
                    private:
                        void* is;
                    };

                    template <typename _T>
                    class Instance_V final {
                    public:
                        Instance_V(constructInstance_t CI, iterateInstance_t<_T> II, destructInstance_t DI, void* Object, void* ConstructInstanceSharedData) {
                            this->ci = CI;
                            this->ii = II;
                            this->di = DI;
                            
                            is = CI(Object, ConstructInstanceSharedData);
                        }
                        Instance_V(InstanceFunctions<_T> InstanceFunctions, void* Object, void* ConstructInstanceSharedData) {
                            this->ci = InstanceFunctions.constructInstance;
                            this->ii = InstanceFunctions.iterateInstance;
                            this->di = InstanceFunctions.destructInstance;

                            is = ci(Object, ConstructInstanceSharedData);
                        }
                        _T* IterateInstance(_T* Inputs) {
                            return ii(is, Inputs);
                        }
                        void DestroyInstance() {
                            di(is);
                        }
                    private:
                        constructInstance_t ci;
                        iterateInstance_t<_T> ii;
                        destructInstance_t di;
                        void* is;
                    };
                }
            }
        }
    }
}