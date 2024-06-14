#include "brendancuda_nets_net.cuh"
#include <device_launch_parameters.h>
#include "brendancuda_cudaerrorhelpers.h"

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
    cudaFree(inputs);
    cudaFree(outputs);
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

thrust::device_ptr<BrendanCUDA::Nets::NetNode> BrendanCUDA::Nets::Net::Data() {
    return nodes.data();
}

thrust::device_reference<BrendanCUDA::Nets::NetNode> BrendanCUDA::Nets::Net::operator[](size_t i) {
    return nodes[i];
}

__global__ void addConnection_checkForPreexistence(size_t* arr, size_t v, bool* opt) {
    if (arr[blockIdx.x] == v) {
        *opt = true;
    }
}

bool BrendanCUDA::Nets::Net::AddConnection_OnlyInput(size_t InputIndex, size_t OutputIndex, bool CheckForPreexistence, bool CheckForAvailableExcess) {
    NetNode in = nodes[InputIndex];

    if (CheckForPreexistence) {
        bool f = false;
        bool* opt;
        ThrowIfBad(cudaMalloc(&opt, sizeof(bool)));
        ThrowIfBad(cudaMemcpy(opt, &f, sizeof(bool), cudaMemcpyHostToDevice));
        addConnection_checkForPreexistence<<<in.outputCount, 1>>>(in.outputs, OutputIndex, opt);
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
        in_o_e = (s >= (in.outputCount + 1) * sizeof(size_t));
    }
    else {
        in_o_e = false;
    }

    if (in_o_e) {
        ThrowIfBad(cudaMemcpy(in.outputs + in.outputCount, &OutputIndex, sizeof(size_t), cudaMemcpyHostToDevice));
        in.outputCount++;
    }
    else {
        size_t* n;
        size_t noc = in.outputCount + 1;
        ThrowIfBad(cudaMalloc(&n, sizeof(size_t) * noc));
        ThrowIfBad(cudaMemcpy(n, in.outputs, sizeof(size_t) * in.outputCount, cudaMemcpyDeviceToDevice));
        ThrowIfBad(cudaMemcpy(n + in.outputCount, &OutputIndex, sizeof(size_t), cudaMemcpyHostToDevice));
        ThrowIfBad(cudaFree(in.outputs));
        in.outputs = n;
        in.outputCount = noc;
    }

    nodes[InputIndex] = in;
}
bool BrendanCUDA::Nets::Net::AddConnection_OnlyOutput(size_t InputIndex, size_t OutputIndex, bool CheckForPreexistence, bool CheckForAvailableExcess) {
    NetNode on = nodes[OutputIndex];

    if (CheckForPreexistence) {
        bool f = false;
        bool* opt;
        ThrowIfBad(cudaMalloc(&opt, sizeof(bool)));
        ThrowIfBad(cudaMemcpy(opt, &f, sizeof(bool), cudaMemcpyHostToDevice));
        addConnection_checkForPreexistence<<<on.inputCount, 1>>>(on.inputs, InputIndex, opt);
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
        on_i_e = (s >= (on.inputCount + 1) * sizeof(size_t));
    }
    else {
        on_i_e = false;
    }

    if (on_i_e) {
        ThrowIfBad(cudaMemcpy(on.inputs + on.inputCount, &InputIndex, sizeof(size_t), cudaMemcpyHostToDevice));
        on.inputCount++;
    }
    else {
        size_t* n;
        size_t nic = on.inputCount + 1;
        ThrowIfBad(cudaMalloc(&n, sizeof(size_t) * nic));
        ThrowIfBad(cudaMemcpy(n, on.inputs, sizeof(size_t) * on.inputCount, cudaMemcpyDeviceToDevice));
        ThrowIfBad(cudaMemcpy(n + on.inputCount, &InputIndex, sizeof(size_t), cudaMemcpyHostToDevice));
        ThrowIfBad(cudaFree(on.inputs));
        on.inputs = n;
        on.inputCount = nic;
    }

    nodes[OutputIndex] = on;
}
bool BrendanCUDA::Nets::Net::AddConnection(size_t InputIndex, size_t OutputIndex, bool CheckForPreexistence, bool CheckForAvailableExcess) {
    NetNode in = nodes[InputIndex];
    NetNode on = nodes[OutputIndex];

    if (CheckForPreexistence) {
        bool f = false;
        bool* opt;
        ThrowIfBad(cudaMalloc(&opt, sizeof(bool)));
        ThrowIfBad(cudaMemcpy(opt, &f, sizeof(bool), cudaMemcpyHostToDevice));
        addConnection_checkForPreexistence<<<in.outputCount, 1>>>(in.outputs, OutputIndex, opt);
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
        ThrowIfBad(cudaGetSymbolSize(&s, in.outputs));
        in_o_e = (s >= (in.outputCount + 1) * sizeof(size_t));
        ThrowIfBad(cudaGetSymbolSize(&s, on.inputs));
        on_i_e = (s >= (on.inputCount + 1) * sizeof(size_t));
    }
    else {
        in_o_e = false;
        on_i_e = false;
    }

    if (in_o_e) {
        ThrowIfBad(cudaMemcpy(in.outputs + in.outputCount, &OutputIndex, sizeof(size_t), cudaMemcpyHostToDevice));
        in.outputCount++;
    }
    else {
        size_t* n;
        size_t noc = in.outputCount + 1;
        ThrowIfBad(cudaMalloc(&n, sizeof(size_t) * noc));
        ThrowIfBad(cudaMemcpy(n, in.outputs, sizeof(size_t) * in.outputCount, cudaMemcpyDeviceToDevice));
        ThrowIfBad(cudaMemcpy(n + in.outputCount, &OutputIndex, sizeof(size_t), cudaMemcpyHostToDevice));
        ThrowIfBad(cudaFree(in.outputs));
        in.outputs = n;
        in.outputCount = noc;
    }

    if (on_i_e) {
        ThrowIfBad(cudaMemcpy(on.inputs + on.inputCount, &InputIndex, sizeof(size_t), cudaMemcpyHostToDevice));
        on.inputCount++;
    }
    else {
        size_t* n;
        size_t nic = on.inputCount + 1;
        ThrowIfBad(cudaMalloc(&n, sizeof(size_t) * nic));
        ThrowIfBad(cudaMemcpy(n, on.inputs, sizeof(size_t) * on.inputCount, cudaMemcpyDeviceToDevice));
        ThrowIfBad(cudaMemcpy(n + on.inputCount, &InputIndex, sizeof(size_t), cudaMemcpyHostToDevice));
        ThrowIfBad(cudaFree(on.inputs));
        on.inputs = n;
        on.inputCount = nic;
    }

    nodes[InputIndex] = in;
    nodes[OutputIndex] = on;
}

bool BrendanCUDA::Nets::Net::RemoveConnection_OnlyInput(size_t InputIndex, size_t OutputIndex, bool RemoveExcess) {
    NetNode in = nodes[InputIndex];

    size_t* in_o = new size_t[in.outputCount];

    ThrowIfBad(cudaMemcpy(in_o, in.outputs, sizeof(size_t) * in.outputCount, cudaMemcpyDeviceToHost));

    if (RemoveExcess) {
        for (size_t i = 0; i < in.outputCount; ++i) {
            if (in_o[i] == OutputIndex) {
                in_o[i] = in_o[in.outputCount - 1];
                goto ExitA;
            }
        }
        return false;

    ExitA:
        in.outputCount--;

        size_t* in_o_n;

        ThrowIfBad(cudaMalloc(&in_o_n, sizeof(size_t) * in.outputCount));

        ThrowIfBad(cudaMemcpy(in_o_n, in_o, sizeof(size_t) * in.outputCount, cudaMemcpyHostToDevice));

        delete[] in_o;

        ThrowIfBad(cudaFree(in.outputs));

        in.outputs = in_o_n;

        nodes[InputIndex] = in;
    }
    else {
        for (size_t i = 0; i < in.outputCount; ++i) {
            if (in_o[i] == OutputIndex) {
                cudaMemcpy(in.outputs + i, in.outputs + (in.outputCount - 1), sizeof(size_t), cudaMemcpyDeviceToDevice);
                in_o[i] = in_o[in.outputCount - 1];
                goto ExitB;
            }
        }
        return false;

    ExitB:
        in.outputCount--;
        nodes[InputIndex] = in;
    }
}
bool BrendanCUDA::Nets::Net::RemoveConnection_OnlyOutput(size_t InputIndex, size_t OutputIndex, bool RemoveExcess) {
    NetNode on = nodes[OutputIndex];

    size_t* on_i = new size_t[on.inputCount];

    ThrowIfBad(cudaMemcpy(on_i, on.inputs, sizeof(size_t) * on.inputCount, cudaMemcpyDeviceToHost));

    if (RemoveExcess) {
        for (size_t i = 0; i < on.inputCount; ++i) {
            if (on_i[i] == InputIndex) {
                on_i[i] = on_i[on.inputCount - 1];
                goto ExitA;
            }
        }
        return false;

    ExitA:
        on.inputCount--;

        size_t* on_i_n;

        ThrowIfBad(cudaMalloc(&on_i_n, sizeof(size_t) * on.inputCount));

        ThrowIfBad(cudaMemcpy(on_i_n, on_i, sizeof(size_t) * on.inputCount, cudaMemcpyHostToDevice));

        delete[] on_i;

        ThrowIfBad(cudaFree(on.inputs));

        on.inputs = on_i_n;

        nodes[OutputIndex] = on;
    }
    else {
        for (size_t i = 0; i < on.inputCount; ++i) {
            if (on_i[i] == InputIndex) {
                cudaMemcpy(on.inputs + i, on.inputs + (on.inputCount - 1), sizeof(size_t), cudaMemcpyDeviceToDevice);
                on_i[i] = on_i[on.inputCount - 1];
                goto ExitB;
            }
        }
        return false;

    ExitB:
        on.inputCount--;
        nodes[OutputIndex] = on;
    }
}
bool BrendanCUDA::Nets::Net::RemoveConnection(size_t InputIndex, size_t OutputIndex, bool RemoveExcess) {
    NetNode in = nodes[InputIndex];
    NetNode on = nodes[OutputIndex];

    size_t* in_o = new size_t[in.outputCount];
    size_t* on_i = new size_t[on.inputCount];

    ThrowIfBad(cudaMemcpy(in_o, in.outputs, sizeof(size_t) * in.outputCount, cudaMemcpyDeviceToHost));
    ThrowIfBad(cudaMemcpy(on_i, on.inputs, sizeof(size_t) * on.inputCount, cudaMemcpyDeviceToHost));

    if (RemoveExcess) {
        for (size_t i = 0; i < in.outputCount; ++i) {
            if (in_o[i] == OutputIndex) {
                in_o[i] = in_o[in.outputCount - 1];
                goto Exit0A;
            }
        }
        return false;

Exit0A:
        for (size_t i = 0; i < on.inputCount; ++i) {
            if (on_i[i] == InputIndex) {
                on_i[i] = on_i[on.inputCount - 1];
                goto Exit1A;
            }
        }
        throw std::exception();

Exit1A:
        in.outputCount--;
        on.inputCount--;

        size_t* in_o_n;
        size_t* on_i_n;

        ThrowIfBad(cudaMalloc(&in_o_n, sizeof(size_t) * in.outputCount));
        ThrowIfBad(cudaMalloc(&on_i_n, sizeof(size_t) * on.inputCount));

        ThrowIfBad(cudaMemcpy(in_o_n, in_o, sizeof(size_t) * in.outputCount, cudaMemcpyHostToDevice));
        ThrowIfBad(cudaMemcpy(on_i_n, on_i, sizeof(size_t) * on.inputCount, cudaMemcpyHostToDevice));

        delete[] in_o;
        delete[] on_i;

        ThrowIfBad(cudaFree(in.outputs));
        ThrowIfBad(cudaFree(on.inputs));

        in.outputs = in_o_n;
        on.inputs = on_i_n;

        nodes[InputIndex] = in;
        nodes[OutputIndex] = on;
    }
    else {
        for (size_t i = 0; i < in.outputCount; ++i) {
            if (in_o[i] == OutputIndex) {
                cudaMemcpy(in.outputs + i, in.outputs + (in.outputCount - 1), sizeof(size_t), cudaMemcpyDeviceToDevice);
                in_o[i] = in_o[in.outputCount - 1];
                goto Exit0B;
            }
        }
        return false;

Exit0B:
        for (size_t i = 0; i < on.inputCount; ++i) {
            if (on_i[i] == InputIndex) {
                cudaMemcpy(on.inputs + i, on.inputs + (on.inputCount - 1), sizeof(size_t), cudaMemcpyDeviceToDevice);
                on_i[i] = on_i[on.inputCount - 1];
                goto Exit1B;
            }
        }
        throw std::exception();

Exit1B:
        in.outputCount--;
        on.inputCount--;
        nodes[InputIndex] = in;
        nodes[OutputIndex] = on;
    }
}

void BrendanCUDA::Nets::Net::RemoveAt(size_t Index, dataDestructor_t DataDestructor) {
    NetNode nn = nodes[Index];
    size_t* inputs = new size_t[nn.inputCount];
    size_t* outputs = new size_t[nn.outputCount];
    if (cudaMemcpy(inputs, nn.inputs, sizeof(size_t) * nn.inputCount, cudaMemcpyDeviceToHost)) {
        throw std::exception();
    }
    if (cudaMemcpy(outputs, nn.outputs, sizeof(size_t) * nn.outputCount, cudaMemcpyDeviceToHost)) {
        throw std::exception();
    }
    for (size_t i = 0; i < nn.inputCount; ++i) {
        NetNode nn2 = nodes[inputs[i]];
        size_t* outputs2 = new size_t[nn2.outputCount];
        size_t ocm1 = nn2.outputCount - 1;
        if (cudaMemcpy(outputs2, nn2.outputs, sizeof(size_t) * nn2.outputCount, cudaMemcpyDeviceToHost)) {
            throw std::exception();
        }
        for (size_t j = 0; j < nn2.outputCount; ++j) {
            if (outputs2[j] == Index) {
                outputs2[j] = outputs[ocm1];
                break;
            }
        }
        size_t* outputs3;
        if (cudaMalloc(&outputs3, sizeof(size_t) * ocm1)) {
            throw std::exception();
        }
        if (cudaMemcpy(outputs3, outputs2, sizeof(size_t) * ocm1, cudaMemcpyDeviceToHost)) {
            throw std::exception();
        }
        cudaFree(nn2.outputs);
        nn2.outputs = outputs3;
        nn2.outputCount = ocm1;
        nodes[inputs[i]] = nn2;
        delete[] outputs2;
    }
    for (size_t i = 0; i < nn.outputCount; ++i) {
        NetNode nn2 = nodes[outputs[i]];
        size_t* inputs2 = new size_t[nn2.inputCount];
        size_t icm1 = nn2.inputCount - 1;
        if (cudaMemcpy(inputs2, nn2.inputs, sizeof(size_t) * nn2.inputCount, cudaMemcpyDeviceToHost)) {
            throw std::exception();
        }
        for (size_t j = 0; j < nn2.inputCount; ++j) {
            if (inputs2[j] == Index) {
                inputs2[j] = inputs[icm1];
                break;
            }
        }
        size_t* inputs3;
        if (cudaMalloc(&inputs3, sizeof(size_t) * icm1)) {
            throw std::exception();
        }
        if (cudaMemcpy(inputs3, inputs2, sizeof(size_t) * icm1, cudaMemcpyDeviceToHost)) {
            throw std::exception();
        }
        cudaFree(nn2.inputs);
        nn2.inputs = inputs3;
        nn2.inputCount = icm1;
        nodes[outputs[i]] = nn2;
        delete[] inputs2;
    }
    delete[] inputs;
    delete[] outputs;
    nn.Dispose(DataDestructor);
    nodes[Index] = nodes[nodes.size() - 1];
    nodes.pop_back();
}