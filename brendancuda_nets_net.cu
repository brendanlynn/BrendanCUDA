#include "brendancuda_nets_net.h"
#include <device_launch_parameters.h>
#include "brendancuda_errorhelp.h"
#include "brendancuda_crossassigns.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

__global__ void net_addConnection_checkForPreexistence(BrendanCUDA::Nets::NetNode** arr, BrendanCUDA::Nets::NetNode* v, bool* opt) {
    if (arr[blockIdx.x] == v) {
        *opt = true;
    }
}

__global__ void replaceBase(BrendanCUDA::Nets::NetNode* oldBase, BrendanCUDA::Nets::NetNode** oldNodes, BrendanCUDA::Nets::NetNode* newBase, BrendanCUDA::Nets::NetNode** newNodes) {
    BrendanCUDA::Nets::NetNode* oldNode = oldNodes[blockIdx.x];
    BrendanCUDA::Nets::NetNode*& newNode = newNodes[blockIdx.x];

    newNode = oldNode - oldBase + newBase;
}

BrendanCUDA::Nets::Net BrendanCUDA::Nets::Net::Clone(dataCloner_t DataCloner) const {
    thrust::device_vector<NetNode>* p_newNodes = new thrust::device_vector<NetNode>(nodes.size());
    thrust::device_vector<NetNode>& newNodes = *p_newNodes;
    NetNode* oldBaseNN = nodes.data().get();
    NetNode* newBaseNN = newNodes.data().get();
    for (size_t i = 0; i < nodes.size(); ++i) {
        NetNode oldNN = nodes[i];
        NetNode newNN;
        newNN.data = DataCloner(oldNN);
        newNN.inputCount = oldNN.inputCount;
        newNN.outputCount = oldNN.outputCount;
        cudaMalloc(&newNN.inputs, sizeof(NetNode*) * oldNN.inputCount);
        cudaMalloc(&newNN.outputs, sizeof(NetNode*) * oldNN.outputCount);
        replaceBase<<<oldNN.inputCount, 1>>>(oldBaseNN, oldNN.inputs, newBaseNN, newNN.inputs);
        replaceBase<<<oldNN.outputCount, 1>>>(oldBaseNN, oldNN.outputs, newBaseNN, newNN.outputs);
        newNodes[i] = newNN;
    }
    return Net(newNodes);
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
            ThrowIfBad(cuMemGetAddressRange_v2(0, &s, (CUdeviceptr_v2)in.outputs));
            in_o_e = (s >= (in.outputCount + 1) * sizeof(NetNode*));
        }
        else {
            in_o_e = false;
        }

        if (!in.outputs) {
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
            ThrowIfBad(cuMemGetAddressRange_v2(0, &s, (CUdeviceptr_v2)on.inputs));
            on_i_e = (s >= (on.inputCount + 1) * sizeof(NetNode*));
        }
        else {
            on_i_e = false;
        }

        if (!on.inputs) {
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
    if (InputNode != OutputNode) {
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
                ThrowIfBad(cuMemGetAddressRange_v2(0, &s, (CUdeviceptr_v2)in.outputs));
                in_o_e = (s >= (in.outputCount + 1) * sizeof(NetNode*));
            }
            else {
                in_o_e = false;
            }
            if (on.inputs) {
                ThrowIfBad(cuMemGetAddressRange_v2(0, &s, (CUdeviceptr_v2)on.inputs));
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

        if (!in.outputs) {
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

        if (!on.inputs) {
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
    else {
        NetNode n = GetVR(InputNode);

        if (CheckForPreexistence && n.inputs) {
            bool f = false;
            bool* opt;
            ThrowIfBad(cudaMalloc(&opt, sizeof(bool)));
            ThrowIfBad(cudaMemcpy(opt, &f, sizeof(bool), cudaMemcpyHostToDevice));
            net_addConnection_checkForPreexistence<<<n.inputCount, 1>>>(n.inputs, InputNode, opt);
            ThrowIfBad(cudaMemcpy(&f, opt, sizeof(bool), cudaMemcpyDeviceToHost));
            ThrowIfBad(cudaFree(opt));
            if (f) {
                return false;
            }
        }

        bool n_o_e;
        bool n_i_e;
        if (CheckForAvailableExcess) {
            size_t s;
            if (n.outputs) {
                ThrowIfBad(cuMemGetAddressRange_v2(0, &s, (CUdeviceptr_v2)n.outputs));
                n_o_e = (s >= (n.outputCount + 1) * sizeof(NetNode*));
            }
            else {
                n_o_e = false;
            }
            if (n.inputs) {
                ThrowIfBad(cuMemGetAddressRange_v2(0, &s, (CUdeviceptr_v2)n.inputs));
                n_i_e = (s >= (n.inputCount + 1) * sizeof(NetNode*));
            }
            else {
                n_i_e = false;
            }
        }
        else {
            n_o_e = false;
            n_i_e = false;
        }

        if (!n.outputs) {
            ThrowIfBad(cudaMalloc(&n.outputs, sizeof(NetNode*)));
            SetVR(n.outputs, OutputNode);
            n.outputCount = 1;
        }
        else if (n_o_e) {
            ThrowIfBad(cudaMemcpy(n.outputs + n.outputCount, &OutputNode, sizeof(NetNode*), cudaMemcpyHostToDevice));
            n.outputCount++;
        }
        else {
            NetNode** ns;
            size_t noc = n.outputCount + 1;
            ThrowIfBad(cudaMalloc(&ns, sizeof(NetNode*) * noc));
            if (n.outputs) {
                ThrowIfBad(cudaMemcpy(ns, n.outputs, sizeof(NetNode*) * n.outputCount, cudaMemcpyDeviceToDevice));
            }
            ThrowIfBad(cudaMemcpy(ns + n.outputCount, &OutputNode, sizeof(NetNode*), cudaMemcpyHostToDevice));
            ThrowIfBad(cudaFree(n.outputs));
            n.outputs = ns;
            n.outputCount = noc;
        }

        if (!n.inputs) {
            ThrowIfBad(cudaMalloc(&n.inputs, sizeof(NetNode*)));
            SetVR(n.inputs, InputNode);
            n.inputCount = 1;
        }
        else if (n_i_e) {
            ThrowIfBad(cudaMemcpy(n.inputs + n.inputCount, &InputNode, sizeof(NetNode*), cudaMemcpyHostToDevice));
            n.inputCount++;
        }
        else {
            NetNode** ns;
            size_t nic = n.inputCount + 1;
            ThrowIfBad(cudaMalloc(&ns, sizeof(NetNode*) * nic));
            if (n.inputs) {
                ThrowIfBad(cudaMemcpy(ns, n.inputs, sizeof(NetNode*) * n.inputCount, cudaMemcpyDeviceToDevice));
            }
            ThrowIfBad(cudaMemcpy(ns + n.inputCount, &InputNode, sizeof(NetNode*), cudaMemcpyHostToDevice));
            ThrowIfBad(cudaFree(n.inputs));
            n.inputs = ns;
            n.inputCount = nic;
        }

        SetVR(InputNode, n);
        return true;
    }
}

bool BrendanCUDA::Nets::Net::RemoveConnection_OnlyInput(NetNode* InputNode, NetNode* OutputNode, bool RemoveExcess) {
    NetNode in = GetVR(InputNode);

    if (in.outputs) {
        NetNode** in_o = new NetNode*[in.outputCount];

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

            if (in.outputCount) {
                ThrowIfBad(cudaMalloc(&in_o_n, sizeof(NetNode*) * in.outputCount));

                ThrowIfBad(cudaMemcpy(in_o_n, in_o, sizeof(NetNode*) * in.outputCount, cudaMemcpyHostToDevice));
            }
            else {
                in_o_n = 0;
            }

            delete[] in_o;

            ThrowIfBad(cudaFree(in.outputs));

            in.outputs = in_o_n;
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
        }
        SetVR(InputNode, in);
        return true;
    }
    else {
        return false;
    }
}
bool BrendanCUDA::Nets::Net::RemoveConnection_OnlyOutput(NetNode* InputNode, NetNode* OutputNode, bool RemoveExcess) {
    NetNode on = GetVR(OutputNode);

    if (on.inputs) {
        NetNode** on_i = new NetNode*[on.inputCount];

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

            if (on.inputCount) {
                ThrowIfBad(cudaMalloc(&on_i_n, sizeof(NetNode*) * on.inputCount));

                ThrowIfBad(cudaMemcpy(on_i_n, on_i, sizeof(NetNode*) * on.inputCount, cudaMemcpyHostToDevice));
            }
            else {
                on_i_n = 0;
            }

            delete[] on_i;

            ThrowIfBad(cudaFree(on.inputs));

            on.inputs = on_i_n;
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
        }
        SetVR(OutputNode, on);
        return true;
    }
    else {
        return false;
    }
}
bool BrendanCUDA::Nets::Net::RemoveConnection(NetNode* InputNode, NetNode* OutputNode, bool RemoveExcess) {
    if (InputNode != OutputNode) {
        NetNode in = GetVR(InputNode);
        NetNode on = GetVR(OutputNode);

        if (in.outputs || on.inputs) {
            NetNode** in_o = new NetNode*[in.outputCount];
            NetNode** on_i = new NetNode*[on.inputCount];

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

                if (in.outputCount) {
                    ThrowIfBad(cudaMalloc(&in_o_n, sizeof(NetNode*) * in.outputCount));
                    ThrowIfBad(cudaMemcpy(in_o_n, in_o, sizeof(NetNode*) * in.outputCount, cudaMemcpyHostToDevice));
                }
                else {
                    in_o_n = 0;
                }

                if (in.outputCount) {
                    ThrowIfBad(cudaMalloc(&on_i_n, sizeof(NetNode*) * on.inputCount));
                    ThrowIfBad(cudaMemcpy(on_i_n, on_i, sizeof(NetNode*) * on.inputCount, cudaMemcpyHostToDevice));
                }
                else {
                    on_i_n = 0;
                }

                delete[] in_o;
                delete[] on_i;

                ThrowIfBad(cudaFree(in.outputs));
                ThrowIfBad(cudaFree(on.inputs));

                in.outputs = in_o_n;
                on.inputs = on_i_n;
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
            }
            SetVR(InputNode, in);
            SetVR(OutputNode, on);
            return true;
        }
        else {
            return false;
        }
    }
    else {
        NetNode n = GetVR(InputNode);

        if (n.outputs || n.inputs) {
            NetNode** n_o = new NetNode*[n.outputCount];
            NetNode** n_i = new NetNode*[n.inputCount];

            ThrowIfBad(cudaMemcpy(n_o, n.outputs, sizeof(NetNode*) * n.outputCount, cudaMemcpyDeviceToHost));
            ThrowIfBad(cudaMemcpy(n_i, n.inputs, sizeof(NetNode*) * n.inputCount, cudaMemcpyDeviceToHost));

            if (RemoveExcess) {
                for (size_t i = 0; i < n.outputCount; ++i) {
                    if (n_o[i] == OutputNode) {
                        n_o[i] = n_o[n.outputCount - 1];
                        goto Exit0C;
                    }
                }
                return false;

Exit0C:
                for (size_t i = 0; i < n.inputCount; ++i) {
                    if (n_i[i] == InputNode) {
                        n_i[i] = n_i[n.inputCount - 1];
                        goto Exit1C;
                    }
                }
                throw std::exception();

Exit1C:
                n.outputCount--;
                n.inputCount--;

                NetNode** n_o_n;
                NetNode** n_i_n;

                if (n.outputCount) {
                    ThrowIfBad(cudaMalloc(&n_o_n, sizeof(NetNode*) * n.outputCount));
                    ThrowIfBad(cudaMemcpy(n_o_n, n_o, sizeof(NetNode*) * n.outputCount, cudaMemcpyHostToDevice));
                }
                else {
                    n_o_n = 0;
                }

                if (n.outputCount) {
                    ThrowIfBad(cudaMalloc(&n_i_n, sizeof(NetNode*) * n.inputCount));
                    ThrowIfBad(cudaMemcpy(n_i_n, n_i, sizeof(NetNode*) * n.inputCount, cudaMemcpyHostToDevice));
                }
                else {
                    n_i_n = 0;
                }

                delete[] n_o;
                delete[] n_i;

                ThrowIfBad(cudaFree(n.outputs));
                ThrowIfBad(cudaFree(n.inputs));

                n.outputs = n_o_n;
                n.inputs = n_i_n;
            }
            else {
                for (size_t i = 0; i < n.outputCount; ++i) {
                    if (n_o[i] == OutputNode) {
                        ThrowIfBad(cudaMemcpy(n.outputs + i, n.outputs + (n.outputCount - 1), sizeof(NetNode*), cudaMemcpyDeviceToDevice));
                        n_o[i] = n_o[n.outputCount - 1];
                        goto Exit0D;
                    }
                }
                return false;

Exit0D:
                for (size_t i = 0; i < n.inputCount; ++i) {
                    if (n_i[i] == InputNode) {
                        ThrowIfBad(cudaMemcpy(n.inputs + i, n.inputs + (n.inputCount - 1), sizeof(NetNode*), cudaMemcpyDeviceToDevice));
                        n_i[i] = n_i[n.inputCount - 1];
                        goto Exit1D;
                    }
                }
                throw std::exception();

Exit1D:
                n.outputCount--;
                n.inputCount--;
            }
            SetVR(InputNode, n);
            return true;
        }
        else {
            return false;
        }
    }
}

void BrendanCUDA::Nets::Net::RemoveAllConnections(NetNode* Node, bool RemoveExcess) {
    NetNode nn = GetVR(Node);
    NetNode** inputs = new NetNode*[nn.inputCount];
    NetNode** outputs = new NetNode*[nn.outputCount];
    ThrowIfBad(cudaMemcpy(inputs, nn.inputs, sizeof(NetNode*) * nn.inputCount, cudaMemcpyDeviceToHost));
    ThrowIfBad(cudaMemcpy(outputs, nn.outputs, sizeof(NetNode*) * nn.outputCount, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < nn.inputCount; ++i) {
        NetNode* o = inputs[i];
        if (o != Node) {
            RemoveConnection_OnlyInput(o, Node, RemoveExcess);
        }
    }
    for (size_t i = 0; i < nn.outputCount; ++i) {
        NetNode* o = outputs[i];
        if (o != Node) {
            RemoveConnection_OnlyOutput(Node, o, RemoveExcess);
        }
    }
    delete[] inputs;
    delete[] outputs;
    if (RemoveExcess) {
        ThrowIfBad(cudaFree(nn.inputs));
        ThrowIfBad(cudaFree(nn.outputs));
        nn.inputs = 0;
        nn.outputs = 0;
    }
    nn.inputCount = 0;
    nn.outputCount = 0;
    SetVR(Node, nn);
}
void BrendanCUDA::Nets::Net::PrintTo(std::ostream& Output, size_t IndentPre, size_t IndentSize) const {
    std::string pi(IndentPre, ' ');
    std::string si(IndentSize, ' ');
    thrust::device_vector<NetNode>& vec = DataVec();
    for (size_t i = 0; i < vec.size(); ++i) {
        thrust::device_ptr<NetNode> dp = vec.data() + i;
        NetNode* p = dp.get();
        NetNode v = *dp;
        Output << pi << p << ":" << std::endl;
        Output << pi << si << "Data: " << v.data << std::endl;
        Output << pi << si << "Input Count: " << v.inputCount << std::endl;
        Output << pi << si << "Inputs: " << v.inputs << std::endl;
        for (size_t j = 0; j < v.inputCount; ++j) {
            Output << pi << si << si << j << ": " << GetVR(v.inputs + j) << std::endl;
        }
        Output << pi << si << "Output Count: " << v.outputCount << std::endl;
        Output << pi << si << "Outputs: " << v.outputs << std::endl;
        for (size_t j = 0; j < v.outputCount; ++j) {
            Output << pi << si << si << j << ": " << GetVR(v.outputs + j) << std::endl;
        }
    }
}