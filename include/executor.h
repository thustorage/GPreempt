#pragma once

#include "model.h"
#include "workloads/graphcompute.h"
#include "workloads/scicomput.h"
#include <memory>
#include <atomic>
#include <unistd.h>

namespace foo {

// An Executor is used to execute a model.
// It is an abstract class.
class Executor {
public:

    Status set_graph_path(const std::string& path);

    virtual Status load_model(std::string model_path);

    virtual Status load_param(std::string model_path);

    // Execute the model asynchronously.
    virtual Status execute(GPUstream stream);

    virtual Status launch_kernel(size_t kernel_offset, GPUstream stream = 0);

    virtual Status init(std::string workload);
    virtual Status set_input(const std::string &key, void *value, size_t len, GPUstream stream);
    virtual Status set_input(const std::string &key, std::vector<float> &value, GPUstream stream);
    virtual Status get_output(const std::string &key, void *value, size_t len, GPUstream stream);
    virtual Status get_output(const std::string &key, std::vector<float> &value, GPUstream stream);
    virtual size_t get_data_size(const std::string &key);

    virtual void clear();
    size_t get_kernel_num() const;
    virtual void capture_graph(GPUstream stream);
    virtual ~Executor() {}

protected:
    std::shared_ptr<Model> model;
    std::shared_ptr<GraphComputeRaw::GraphCompute> graph_compute;
    std::string type;
    bool use_cuda_graph = false;
    GPUgraph graph;
    GPUgraphExec graph_exec;
    std::string graph_path;
    double mass0, te0;
};

class BaseExecutor : public Executor {
// methods inherited from Executor

public: 
    BaseExecutor();
    virtual ~BaseExecutor();

};

class BLPExecutor : public Executor {
public:
    BLPExecutor();
    virtual ~BLPExecutor();
    Status getStopPoint(GPUstream stream);
    Status resume(GPUstream stream);
    virtual Status init(std::string workload) override;
    virtual Status execute(GPUstream stream) override;
    virtual void capture_graph(GPUstream stream) override;
    virtual void clear() override;
    void set_lc_be_tasks(std::atomic<int> &lc_tasks, std::atomic<int> &be_tasks){
        this->lc_tasks = &lc_tasks;
        this->be_tasks = &be_tasks;
    }
public:
    GPUdeviceptr dpStop;         // allocated by user to share among all clients
    std::atomic<bool> running;
private:
    GPUdeviceptr dpStopIndex;    // allocated by BLPExecutor
    GPUdeviceptr executed; // allocated by BLPExecutor
    int  stopIndex;
    std::shared_ptr<GraphComputeBlp::GraphCompute> graph_compute;
    std::atomic<int> *lc_tasks;
    std::atomic<int> *be_tasks;
};

} // namespace reef