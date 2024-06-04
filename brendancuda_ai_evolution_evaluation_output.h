#pragma once

#include <cstdint>
#include <utility>

namespace BrendanCUDA {
    namespace AI {
        namespace Evolution {
            namespace Evaluation {
                namespace Output {
                    using constructInstance_t = void*(*)(void* Object, void* ConstructInstanceSharedData);
                    
                    template <typename T>
                    using iterateInstance_t = T*(*)(void* CurrentInstance, T* Inputs);

                    using destructInstance_t = void(*)(void* CurrentInstance);

                    template <typename T>
                    struct instanceFunctions_t final {
                        constructInstance_t constructInstance;
                        iterateInstance_t<T> iterateInstance;
                        destructInstance_t destructInstance;
                    };

                    template <typename T, constructInstance_t ci, iterateInstance_t<T> ii, destructInstance_t di>
                    class instance_c_t final {
                    public:
                        instance_c_t(void* Object, void* ConstructInstanceSharedData) {
                            is = ci(Object, ConstructInstanceSharedData);
                        }
                        T* IterateInstance(T* Inputs) {
                            return ii(is, Inputs);
                        }
                        void DestroyInstance() {
                            di(is);
                        }
                    private:
                        void* is;
                    };

                    template <typename T>
                    class instance_v_t final {
                    public:
                        instance_v_t(constructInstance_t ci, iterateInstance_t<T> ii, destructInstance_t di, void* Object, void* ConstructInstanceSharedData) {
                            this->ci = ci;
                            this->ii = ii;
                            this->di = di;
                            
                            is = ci(Object, ConstructInstanceSharedData);
                        }
                        instance_v_t(instanceFunctions_t<T> instanceFunctions, void* Object, void* ConstructInstanceSharedData) {
                            this->ci = instanceFunctions.constructInstance;
                            this->ii = instanceFunctions.iterateInstance;
                            this->di = instanceFunctions.destructInstance;

                            is = ci(Object, ConstructInstanceSharedData);
                        }
                        T* IterateInstance(T* Inputs) {
                            return ii(is, Inputs);
                        }
                        void DestroyInstance() {
                            di(is);
                        }
                    private:
                        constructInstance_t ci;
                        iterateInstance_t<T> ii;
                        destructInstance_t di;
                        void* is;
                    };
                }
            }
        }
    }
}