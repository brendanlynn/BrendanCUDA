#include "brendancuda_nets_net.cuh"
#include <device_launch_parameters.h>
#include "brendancuda_cudaerrorhelpers.h"
#include "brendancuda_crossassignment.h"

__host__ __device__ BrendanCUDA::Nets::NetNode::NetNode() {
    data = 0;
    inputs = 0;
    inputCount = 0;
    outputs = 0;
    outputCount = 0;
}

__host__ __device__ void BrendanCUDA::Nets::NetNode::Dispose(dataDestructor_t DataDestructor) {
    if (DataDestructor) {
        DataDestructor(data);
    }
#if __CUDA_ARCH__
    delete[] inputs;
    delete[] outputs;
#else
    ThrowIfBad(cudaFree(inputs));
    ThrowIfBad(cudaFree(outputs));
#endif
}

BrendanCUDA::Nets::Net::Net() 
    : nodes(*(new thrust::device_vector<NetNode>())) {}

__global__ void disposeKernel(BrendanCUDA::Nets::NetNode* nodes, BrendanCUDA::Nets::dataDestructor_t DataDestructor) {
    nodes[blockIdx.x].Dispose(DataDestructor);
}
void BrendanCUDA::Nets::Net::Dispose(dataDestructor_t DataDestructor) {
    disposeKernel<<<nodes.size(), 1>>>(nodes.data().get(), DataDestructor);
    delete (&nodes);
}

thrust::device_vector<BrendanCUDA::Nets::NetNode>& BrendanCUDA::Nets::Net::DataVec() const {
    return nodes;
}
thrust::device_ptr<BrendanCUDA::Nets::NetNode> BrendanCUDA::Nets::Net::DataPtr() const {
    return nodes.data();
}

thrust::device_reference<BrendanCUDA::Nets::NetNode> BrendanCUDA::Nets::Net::operator[](size_t i) const {
    return nodes[i];
}

__global__ void net_addConnection_checkForPreexistence(BrendanCUDA::Nets::NetNode** arr, BrendanCUDA::Nets::NetNode* v, bool* opt) {
    if (arr[blockIdx.x] == v) {
        *opt = true;
    }
}

bool BrendanCUDA::Nets::Net::AddConnection_OnlyInput(NetNode* InputNode, NetNode* OutputNode, bool CheckForPreexistence, bool CheckForAvailableExcess) {
    NetNode in = GetVR(InputNode);

    if (in.outputs) {
        if (CheckForPreexistence) {
            bool f = false;
            bool* opt;
            ThrowIfBad(cudaMalloc(&opt, sizeof(bool)));
            ThrowIfBad(cudaMemcpy(opt, &f, sizeof(bool), cudaMemcpyHostToDevice));
            net_addConnection_checkForPreexistence<<<in.outputCount, 1>>>(in.outputs, OutputNode, opt);
            ThrowIfBad(cudaMemcpy(&f, opt, sizeof(bool), cudaMemcpyDeviceToHost));
            ThrowIfBad(cudaFree(opt));
            if (f) {
                return false;
            }
        }

        bool in_o_e;
        if (CheckForAvailableExcess) {
            size_t s;
            ThrowIfBad(cudaGetSymbolSize(&s, in.outputs));
            in_o_e = (s >= (in.outputCount + 1) * sizeof(NetNode*));
        }
        else {
            in_o_e = false;
        }

        if (in_o_e) {
            ThrowIfBad(cudaMemcpy(in.outputs + in.outputCount, &OutputNode, sizeof(NetNode*), cudaMemcpyHostToDevice));
            in.outputCount++;
        }
        else {
            NetNode** n;
            size_t noc = in.outputCount + 1;
            ThrowIfBad(cudaMalloc(&n, sizeof(NetNode*) * noc));
            ThrowIfBad(cudaMemcpy(n, in.outputs, sizeof(NetNode*) * in.outputCount, cudaMemcpyDeviceToDevice));
            ThrowIfBad(cudaMemcpy(n + in.outputCount, &OutputNode, sizeof(NetNode*), cudaMemcpyHostToDevice));
            ThrowIfBad(cudaFree(in.outputs));
            in.outputs = n;
            in.outputCount = noc;
        }
    }
    else {
        ThrowIfBad(cudaMalloc(&in.outputs, sizeof(NetNode*)));
        SetVR(in.outputs, OutputNode);
        in.outputCount = 1;
    }

    SetVR(InputNode, in);
    return true;
}
bool BrendanCUDA::Nets::Net::AddConnection_OnlyOutput(NetNode* InputNode, NetNode* OutputNode, bool CheckForPreexistence, bool CheckForAvailableExcess) {
    NetNode on = GetVR(InputNode);

    if (on.inputs) {
        if (CheckForPreexistence) {
            bool f = false;
            bool* opt;
            ThrowIfBad(cudaMalloc(&opt, sizeof(bool)));
            ThrowIfBad(cudaMemcpy(opt, &f, sizeof(bool), cudaMemcpyHostToDevice));
            net_addConnection_checkForPreexistence<<<on.inputCount, 1>>>(on.inputs, InputNode, opt);
            ThrowIfBad(cudaMemcpy(&f, opt, sizeof(bool), cudaMemcpyDeviceToHost));
            ThrowIfBad(cudaFree(opt));
            if (f) {
                return false;
            }
        }

        bool on_i_e;
        if (CheckForAvailableExcess) {
            size_t s;
            ThrowIfBad(cudaGetSymbolSize(&s, on.inputs));
            on_i_e = (s >= (on.inputCount + 1) * sizeof(NetNode*));
        }
        else {
            on_i_e = false;
        }

        if (on_i_e) {
            ThrowIfBad(cudaMemcpy(on.inputs + on.inputCount, &InputNode, sizeof(NetNode*), cudaMemcpyHostToDevice));
            on.inputCount++;
        }
        else {
            NetNode** n;
            size_t nic = on.inputCount + 1;
            ThrowIfBad(cudaMalloc(&n, sizeof(NetNode*) * nic));
            ThrowIfBad(cudaMemcpy(n, on.inputs, sizeof(NetNode*) * on.inputCount, cudaMemcpyDeviceToDevice));
            ThrowIfBad(cudaMemcpy(n + on.inputCount, &InputNode, sizeof(NetNode*), cudaMemcpyHostToDevice));
            ThrowIfBad(cudaFree(on.inputs));
            on.inputs = n;
            on.inputCount = nic;
        }
    }
    else {
        ThrowIfBad(cudaMalloc(&on.inputs, sizeof(NetNode*)));
        SetVR(on.inputs, InputNode);
        on.inputCount = 1;
    }

    SetVR(OutputNode, on);
    return true;
}
bool BrendanCUDA::Nets::Net::AddConnection(NetNode* InputNode, NetNode* OutputNode, bool CheckForPreexistence, bool CheckForAvailableExcess) {
    NetNode in = GetVR(InputNode);
    NetNode on = GetVR(OutputNode);

    if (CheckForPreexistence && in.outputs) {
        bool f = false;
        bool* opt;
        ThrowIfBad(cudaMalloc(&opt, sizeof(bool)));
        ThrowIfBad(cudaMemcpy(opt, &f, sizeof(bool), cudaMemcpyHostToDevice));
        net_addConnection_checkForPreexistence<<<in.outputCount, 1>>>(in.outputs, OutputNode, opt);
        ThrowIfBad(cudaMemcpy(&f, opt, sizeof(bool), cudaMemcpyDeviceToHost));
        ThrowIfBad(cudaFree(opt));
        if (f) {
            return false;
        }
    }

    bool in_o_e;
    bool on_i_e;
    if (CheckForAvailableExcess) {
        size_t s;
        if (in.outputs) {
            ThrowIfBad(cudaGetSymbolSize(&s, in.outputs));
            in_o_e = (s >= (in.outputCount + 1) * sizeof(NetNode*));
        }
        else {
            in_o_e = false;
        }
        if (on.inputs) {
            ThrowIfBad(cudaGetSymbolSize(&s, on.inputs));
            on_i_e = (s >= (on.inputCount + 1) * sizeof(NetNode*));
        }
        else {
            on_i_e = false;
        }
    }
    else {
        in_o_e = false;
        on_i_e = false;
    }

    if (in.outputs) {
        ThrowIfBad(cudaMalloc(&in.outputs, sizeof(NetNode*)));
        SetVR(in.outputs, OutputNode);
        in.outputCount = 1;
    }
    else if (in_o_e) {
        ThrowIfBad(cudaMemcpy(in.outputs + in.outputCount, &OutputNode, sizeof(NetNode*), cudaMemcpyHostToDevice));
        in.outputCount++;
    }
    else {
        NetNode** n;
        size_t noc = in.outputCount + 1;
        ThrowIfBad(cudaMalloc(&n, sizeof(NetNode*) * noc));
        if (in.outputs) {
            ThrowIfBad(cudaMemcpy(n, in.outputs, sizeof(NetNode*) * in.outputCount, cudaMemcpyDeviceToDevice));
        }
        ThrowIfBad(cudaMemcpy(n + in.outputCount, &OutputNode, sizeof(NetNode*), cudaMemcpyHostToDevice));
        ThrowIfBad(cudaFree(in.outputs));
        in.outputs = n;
        in.outputCount = noc;
    }

    if (on.inputs) {
        ThrowIfBad(cudaMalloc(&on.inputs, sizeof(NetNode*)));
        SetVR(on.inputs, InputNode);
        on.inputCount = 1;
    }
    else if (on_i_e) {
        ThrowIfBad(cudaMemcpy(on.inputs + on.inputCount, &InputNode, sizeof(NetNode*), cudaMemcpyHostToDevice));
        on.inputCount++;
    }
    else {
        NetNode** n;
        size_t nic = on.inputCount + 1;
        ThrowIfBad(cudaMalloc(&n, sizeof(NetNode*) * nic));
        if (on.inputs) {
            ThrowIfBad(cudaMemcpy(n, on.inputs, sizeof(NetNode*) * on.inputCount, cudaMemcpyDeviceToDevice));
        }
        ThrowIfBad(cudaMemcpy(n + on.inputCount, &InputNode, sizeof(NetNode*), cudaMemcpyHostToDevice));
        ThrowIfBad(cudaFree(on.inputs));
        on.inputs = n;
        on.inputCount = nic;
    }

    SetVR(InputNode, in);
    SetVR(OutputNode, on);
    return true;
}

bool BrendanCUDA::Nets::Net::RemoveConnection_OnlyInput(NetNode* InputNode, NetNode* OutputNode, bool RemoveExcess) {
    NetNode in = GetVR(InputNode);

    if (in.outputs) {
        NetNode** in_o = new NetNode * [in.outputCount];

        ThrowIfBad(cudaMemcpy(in_o, in.outputs, sizeof(NetNode*) * in.outputCount, cudaMemcpyDeviceToHost));

        if (RemoveExcess) {
            for (size_t i = 0; i < in.outputCount; ++i) {
                if (in_o[i] == OutputNode) {
                    in_o[i] = in_o[in.outputCount - 1];
                    goto ExitA;
                }
            }
            return false;

        ExitA:
            in.outputCount--;

            NetNode** in_o_n;

            ThrowIfBad(cudaMalloc(&in_o_n, sizeof(NetNode*) * in.outputCount));

            ThrowIfBad(cudaMemcpy(in_o_n, in_o, sizeof(NetNode*) * in.outputCount, cudaMemcpyHostToDevice));

            delete[] in_o;

            ThrowIfBad(cudaFree(in.outputs));

            in.outputs = in_o_n;

            SetVR(InputNode, in);
        }
        else {
            for (size_t i = 0; i < in.outputCount; ++i) {
                if (in_o[i] == OutputNode) {
                    ThrowIfBad(cudaMemcpy(in.outputs + i, in.outputs + (in.outputCount - 1), sizeof(NetNode*), cudaMemcpyDeviceToDevice));
                    in_o[i] = in_o[in.outputCount - 1];
                    goto ExitB;
                }
            }
            return false;

        ExitB:
            in.outputCount--;
            SetVR(InputNode, in);
        }
        return true;
    }
    else {
        return false;
    }
}
bool BrendanCUDA::Nets::Net::RemoveConnection_OnlyOutput(NetNode* InputNode, NetNode* OutputNode, bool RemoveExcess) {
    NetNode on = GetVR(OutputNode);

    if (on.inputs) {
        NetNode** on_i = new NetNode * [on.inputCount];

        ThrowIfBad(cudaMemcpy(on_i, on.inputs, sizeof(NetNode*) * on.inputCount, cudaMemcpyDeviceToHost));

        if (RemoveExcess) {
            for (size_t i = 0; i < on.inputCount; ++i) {
                if (on_i[i] == InputNode) {
                    on_i[i] = on_i[on.inputCount - 1];
                    goto ExitA;
                }
            }
            return false;

        ExitA:
            on.inputCount--;

            NetNode** on_i_n;

            ThrowIfBad(cudaMalloc(&on_i_n, sizeof(NetNode*) * on.inputCount));

            ThrowIfBad(cudaMemcpy(on_i_n, on_i, sizeof(NetNode*) * on.inputCount, cudaMemcpyHostToDevice));

            delete[] on_i;

            ThrowIfBad(cudaFree(on.inputs));

            on.inputs = on_i_n;

            SetVR(OutputNode, on);
        }
        else {
            for (size_t i = 0; i < on.inputCount; ++i) {
                if (on_i[i] == InputNode) {
                    ThrowIfBad(cudaMemcpy(on.inputs + i, on.inputs + (on.inputCount - 1), sizeof(NetNode*), cudaMemcpyDeviceToDevice));
                    on_i[i] = on_i[on.inputCount - 1];
                    goto ExitB;
                }
            }
            return false;

        ExitB:
            on.inputCount--;
            SetVR(OutputNode, on);
        }
        return true;
    }
    else {
        return false;
    }
}
bool BrendanCUDA::Nets::Net::RemoveConnection(NetNode* InputNode, NetNode* OutputNode, bool RemoveExcess) {
    NetNode in = GetVR(InputNode);
    NetNode on = GetVR(OutputNode);

    if (in.outputs || on.inputs) {
        NetNode** in_o = new NetNode * [in.outputCount];
        NetNode** on_i = new NetNode * [on.inputCount];

        ThrowIfBad(cudaMemcpy(in_o, in.outputs, sizeof(NetNode*) * in.outputCount, cudaMemcpyDeviceToHost));
        ThrowIfBad(cudaMemcpy(on_i, on.inputs, sizeof(NetNode*) * on.inputCount, cudaMemcpyDeviceToHost));

        if (RemoveExcess) {
            for (size_t i = 0; i < in.outputCount; ++i) {
                if (in_o[i] == OutputNode) {
                    in_o[i] = in_o[in.outputCount - 1];
                    goto Exit0A;
                }
            }
            return false;

        Exit0A:
            for (size_t i = 0; i < on.inputCount; ++i) {
                if (on_i[i] == InputNode) {
                    on_i[i] = on_i[on.inputCount - 1];
                    goto Exit1A;
                }
            }
            throw std::exception();

        Exit1A:
            in.outputCount--;
            on.inputCount--;

            NetNode** in_o_n;
            NetNode** on_i_n;

            ThrowIfBad(cudaMalloc(&in_o_n, sizeof(NetNode*) * in.outputCount));
            ThrowIfBad(cudaMalloc(&on_i_n, sizeof(NetNode*) * on.inputCount));

            ThrowIfBad(cudaMemcpy(in_o_n, in_o, sizeof(NetNode*) * in.outputCount, cudaMemcpyHostToDevice));
            ThrowIfBad(cudaMemcpy(on_i_n, on_i, sizeof(NetNode*) * on.inputCount, cudaMemcpyHostToDevice));

            delete[] in_o;
            delete[] on_i;

            ThrowIfBad(cudaFree(in.outputs));
            ThrowIfBad(cudaFree(on.inputs));

            in.outputs = in_o_n;
            on.inputs = on_i_n;

            SetVR(InputNode, in);
            SetVR(OutputNode, on);
        }
        else {
            for (size_t i = 0; i < in.outputCount; ++i) {
                if (in_o[i] == OutputNode) {
                    ThrowIfBad(cudaMemcpy(in.outputs + i, in.outputs + (in.outputCount - 1), sizeof(NetNode*), cudaMemcpyDeviceToDevice));
                    in_o[i] = in_o[in.outputCount - 1];
                    goto Exit0B;
                }
            }
            return false;

        Exit0B:
            for (size_t i = 0; i < on.inputCount; ++i) {
                if (on_i[i] == InputNode) {
                    ThrowIfBad(cudaMemcpy(on.inputs + i, on.inputs + (on.inputCount - 1), sizeof(NetNode*), cudaMemcpyDeviceToDevice));
                    on_i[i] = on_i[on.inputCount - 1];
                    goto Exit1B;
                }
            }
            throw std::exception();

        Exit1B:
            in.outputCount--;
            on.inputCount--;
            SetVR(InputNode, in);
            SetVR(OutputNode, on);
        }
        return true;
    }
    else {
        return false;
    }
}

void BrendanCUDA::Nets::Net::RemoveAllConnections(NetNode* Node) {
    NetNode nn = GetVR(Node);
    NetNode** inputs = new NetNode*[nn.inputCount];
    NetNode** outputs = new NetNode*[nn.outputCount];
    ThrowIfBad(cudaMemcpy(inputs, nn.inputs, sizeof(NetNode*) * nn.inputCount, cudaMemcpyDeviceToHost));
    ThrowIfBad(cudaMemcpy(outputs, nn.outputs, sizeof(NetNode*) * nn.outputCount, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < nn.inputCount; ++i) {
        NetNode* o = inputs[i];
        if (o != Node) {
            RemoveConnection_OnlyInput(o, Node, false);
        }
    }
    for (size_t i = 0; i < nn.outputCount; ++i) {
        NetNode* o = outputs[i];
        if (o != Node) {
            RemoveConnection_OnlyOutput(Node, o, false);
        }
    }
    delete[] inputs;
    delete[] outputs;
    ThrowIfBad(cudaFree(nn.inputs));
    ThrowIfBad(cudaFree(nn.outputs));
    nn.inputCount = 0;
    nn.inputs = 0;
    nn.outputCount = 0;
    nn.outputs = 0;
}