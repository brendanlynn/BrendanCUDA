#pragma once

#include <thrust/device_vector.h>
#include <ostream>

namespace BrendanCUDA {
    namespace Nets {
        struct NetNode;
        //Destroys the NetNode::data field, provided context.
        using dataDestructor_t = void(*)(NetNode);
        //Clones the NetNode::data field, provided context.
        using dataCloner_t = void*(*)(NetNode);
        //A node of a BrendanCUDA::Nets::Net.
        struct NetNode final {
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

            //Constructs a BrendanCUDA::Nets::NetNode object.
            __host__ __device__ NetNode();

            //Disposes of a BrendanCUDA::Nets::NetNode object.
            void Dispose(dataDestructor_t DataDestructor) const;
        };
        //A directed graph.
        class Net final {
        public:
            //Creates a BrendanCUDA::Nets::Net object.
            Net();
            //Creates a BrendanCUDA::Nets::Net object, using Data as its vector of nodes without copying it.
            Net(thrust::device_vector<NetNode>& Data);
            //Disposes of a BrendanCUDA::Nets::Net object.
            void Dispose(dataDestructor_t DataDestructor);
            //Returns the vector of nodes, for external manipulation at the user's risk.
            thrust::device_vector<NetNode>& DataVec() const;
            //Returns a pointer to the first node in the vector of nodes, for external manipulation at the user's risk.
            thrust::device_ptr<NetNode> DataPtr() const;
            thrust::device_reference<NetNode> operator[](size_t i) const;
            //Prints a list of nodes, their identifiers, and their inputs and outputs to the Output stream. IndentPre is the amount of spaces (not indents) before the left of the printout, and IndentSize is the amount of spaces in each indent afterward.
            void PrintTo(std::ostream& Output, size_t IndentPre = 0, size_t IndentSize = 4) const;
            //Makes a deep-copy of the BrendanCUDA::Nets::Net object.
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