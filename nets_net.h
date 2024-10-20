#pragma once

#include "errorhelp.h"
#include <cuda_runtime.h>
#include <ostream>
#include <thrust/device_vector.h>

namespace bcuda {
    namespace nets {
        struct NetNode;
        //Destroys the NetNode::data field, provided context.
        using dataDestructor_t = void(*)(NetNode);
        //Clones the NetNode::data field, provided context.
        using dataCloner_t = void*(*)(NetNode);
        //A node of a bcuda::nets::Net.
        struct NetNode {
            //A pointer to the data attached to the node.
            void* data;
            //The inputs to the node. All input claims must be reflected in the list of outputs for each referenced node, or undefined behavior will result. No duplicate elements may exist in the list, or undefined behavior will result.
            NetNode** inputs;
            //The count of the input connections.
            size_t inputCount;
            //The outputs to the node. All output claims must be reflected in the list of inputs for each referenced node, or undefined behavior will result. No duplicate elements may exist in the list, or undefined behavior will result.
            NetNode** outputs;
            //The count of the output connections.
            size_t outputCount;

            //Constructs a bcuda::nets::NetNode object.
            __host__ __device__ inline constexpr NetNode()
                : data(0), inputs(0), inputCount(0), outputs(0), outputCount(0) { }

            //Disposes of a bcuda::nets::NetNode object.
            inline void Dispose(dataDestructor_t DataDestructor) const {
                if (DataDestructor)
                    DataDestructor(*this);
#ifdef __CUDA_ARCH__
                delete[] inputs;
                delete[] outputs;
#else
                ThrowIfBad(cudaFree(inputs));
                ThrowIfBad(cudaFree(outputs));
#endif
            }
        };
        //A directed graph.
        class Net {
        public:
            //Creates a bcuda::nets::Net object.
            inline Net()
                : nodes(*(new thrust::device_vector<NetNode>())) { }
            //Creates a bcuda::nets::Net object, using Data as its vector of nodes without copying it.
            inline Net(thrust::device_vector<NetNode>& Data)
                : nodes(Data) { }
            //Disposes of a bcuda::nets::Net object.
            inline void Dispose(dataDestructor_t DataDestructor) {
                for (size_t i = 0; i < nodes.size(); ++i)
                    ((NetNode)nodes[i]).Dispose(DataDestructor);
                delete (&nodes);
            }
            //Returns the vector of nodes, for external manipulation at the user's risk.
            inline thrust::device_vector<NetNode>& DataVec() {
                return nodes;
            }
            //Returns the vector of nodes, for external manipulation at the user's risk.
            inline const thrust::device_vector<NetNode>& DataVec() const {
                return nodes;
            }
            //Returns a pointer to the first node in the vector of nodes, for external manipulation at the user's risk.
            inline thrust::device_ptr<NetNode> DataPtr() {
                return nodes.data();
            }
            //Returns a pointer to the first node in the vector of nodes, for external manipulation at the user's risk.
            inline thrust::device_ptr<const NetNode> DataPtr() const {
                return nodes.data();
            }
            inline thrust::device_reference<NetNode> operator[](size_t i) {
                return nodes[i];
            }
            inline thrust::device_reference<const NetNode> operator[](size_t i) const {
                return nodes[i];
            }
            //Prints a list of nodes, their identifiers, and their inputs and outputs to the Output stream. IndentPre is the amount of spaces (not indents) before the left of the printout, and IndentSize is the amount of spaces in each indent afterward.
            void PrintTo(std::ostream& Output, size_t IndentPre = 0, size_t IndentSize = 4) const;
            //Makes a deep-copy of the bcuda::nets::Net object.
            Net Clone(dataCloner_t DataCloner) const;

            //Adds a connection between InputNode and OutputNode that goes from InputNode to OutputNode, but only changes InputNode. Use at your own risk.
            static bool AddConnection_OnlyInput(NetNode* InputNode, NetNode* OutputNode, bool CheckForPreexistence, bool CheckForAvailableExcess);
            //Adds a connection between InputNode and OutputNode that goes from InputNode to OutputNode, but only changes OutputNode. Use at your own risk.
            static bool AddConnection_OnlyOutput(NetNode* InputNode, NetNode* OutputNode, bool CheckForPreexistence, bool CheckForAvailableExcess);
            //Adds a connection between InputNode and OutputNode that goes from InputNode to OutputNode.
            static bool AddConnection(NetNode* InputNode, NetNode* OutputNode, bool CheckForPreexistence, bool CheckForAvailableExcess);
            //Removes the connection between InputNode and OutputNode that goes from InputNode to OutputNode, but only changes InputNode. Use at your own risk.
            static bool RemoveConnection_OnlyInput(NetNode* InputNode, NetNode* OutputNode, bool RemoveExcess);
            //Removes the connection between InputNode and OutputNode that goes from InputNode to OutputNode, but only changes OutputNode. Use at your own risk.
            static bool RemoveConnection_OnlyOutput(NetNode* InputNode, NetNode* OutputNode, bool RemoveExcess);
            //Removes the connection between InputNode and OutputNode that goes from InputNode to OutputNode.
            static bool RemoveConnection(NetNode* InputNode, NetNode* OutputNode, bool RemoveExcess);
            //Removes all connections, to and from Node.
            static void RemoveAllConnections(NetNode* Node, bool RemoveExcess);
        private:
            thrust::device_vector<NetNode>& nodes;
        };
    }
}