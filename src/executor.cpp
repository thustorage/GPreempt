#include "executor.h"
#include "model.h"

#include "util/gpu_util.h"
#include "util/common.h"
#include <sstream>
#include <chrono>
#include <algorithm>
#include <set>

#define PATH_OF(name) MODEL_PATH "/" #name
#define GRAPH_BASE_PATH "/home/frw/workdir/dataset/"
const std::set<std::string> graphCompute = {"cc", "bfs", "pagerank", "sssp"};
const std::set<std::string> sciCompute = {"miniweather"};

namespace foo {

Status Executor::set_graph_path(const std::string &path) {
    graph_path = path;
    return Status::Succ;
}

Status Executor::init(std::string workload) {
    if(graphCompute.count(workload)) {
        type = "graph";
        if(workload == "cc") {
            graph_compute = std::make_shared<GraphComputeRaw::cc::CC>(std::string(GRAPH_BASE_PATH) + graph_path);
        } else if(workload == "bfs") {
            graph_compute = std::make_shared<GraphComputeRaw::bfs::BFS>(std::string(GRAPH_BASE_PATH) + graph_path);
        } else if(workload == "pagerank") {
            graph_compute = std::make_shared<GraphComputeRaw::pagerank::PageRank>(std::string(GRAPH_BASE_PATH) + graph_path);
        } else if(workload == "sssp") {
            graph_compute = std::make_shared<GraphComputeRaw::sssp::SSSP>(std::string(GRAPH_BASE_PATH) + graph_path);
        }
    } else if(sciCompute.count(workload)) {
        type = "sci";
        SciComputeRaw::sci_init();
        SciComputeRaw::reductions(this->mass0, this->te0);
    } else {
        type = "dnn";
        RETURN_STATUS(this->load_model(std::string(MODEL_PATH) + "/" + workload));
        RETURN_STATUS(this->load_param(std::string(MODEL_PATH) + "/" + workload));
    }
    return Status::Succ;
}

Status Executor::set_input(const std::string &key, void *value, size_t len, GPUstream stream) {
    if(type == "dnn") {
        size_t storage_idx;
        if (model->find_storage_idx(key, storage_idx) != Status::Succ) {
            return Status::NotFound;
        }
        return model->set(storage_idx, value, len, stream);
    } else if(type == "graph") {
        return Status::Succ;
    } else {
        return Status::Succ;
    }
}
Status Executor::set_input(const std::string &key, std::vector<float> &value, GPUstream stream) {
    return set_input(key, value.data(), value.size() * sizeof(float), stream);
}
Status Executor::get_output(const std::string &key, void *value, size_t len, GPUstream stream) {
    if(type == "dnn") {
        size_t storage_idx;
        if (model->find_storage_idx(key, storage_idx) != Status::Succ) {
            return Status::NotFound;
        }
        return model->get(storage_idx, value, len, stream);
    } else if(type == "graph") {
        return Status::Succ;
    } else {
        // double mass, te;
        // SciComputeBlp::reductions(mass, te);
        // printf("mass: %le, te: %le\n", (mass - mass0)/mass0, (te - te0)/te0);
        return Status::Succ;
    }
}
Status Executor::get_output(const std::string &key, std::vector<float> &value, GPUstream stream) {
    return get_output(key, value.data(), value.size() * sizeof(float), stream);
}

size_t Executor::get_data_size(const std::string &key) {
    if(type == "dnn") {
        size_t storage_idx;
        if (model->find_storage_idx(key, storage_idx) != Status::Succ) {
            return 0;
        }
        return model->get_storage_size(storage_idx);
    } else if(type == "graph") {
        return 0;
    } else {
        return 0;
    }
}

void Executor::clear() {
    if(type == "dnn") {
        model->clear();
    } else if(type == "graph") {
        graph_compute.reset();
    } else {
        double mass, te;
        SciComputeRaw::reductions(mass, te);
        // printf("mass: %le, te: %le\n", (mass - mass0)/mass0, (te - te0)/te0);
        SciComputeRaw::finalize();
    }
}

Status Executor::load_model(std::string model_path) {
    model = std::make_shared<Model>();
    return model->load_model(model_path + "/mod.json", model_path + "/host.json", model_path + "/mod.cubin");
}

Status Executor::load_param(std::string model_path) {
    return model->load_param((model_path + "/mod.params").c_str());
}

Status Executor::execute(GPUstream stream) {
    if(use_cuda_graph) {
        CUDA_RETURN_STATUS(GPUGraphLaunch(graph_exec, stream));
        return Status::Succ;
    }
    if(type == "dnn") {
        for (size_t i = 0; i < get_kernel_num(); i++) {
            RETURN_STATUS(launch_kernel(i, stream));
        }
    } else if(type == "graph") {
        graph_compute->compute(stream);
    } else {
        SciComputeRaw::perform_timestep(stream);
    }
    return Status::Succ;
}

Status Executor::launch_kernel(size_t kernel_offset, GPUstream stream) {
    auto& kernel_info = this->model->get_kernel_info(kernel_offset);
    std::string& func_name = kernel_info.name;
    if(func_name == "nop") {
        size_t size = model->get_storage_size(kernel_info.args_index[0]);
        // LOG(INFO) << "Copying data from " << (void*)kernel_info.args[0] << " to " << (void*)kernel_info.args[1] << ", size: " << size << ", storage size: " << model->get_storage_size(kernel_info.args_index[0]) << ", storage size: " << model->get_storage_size(kernel_info.args_index[1]);
        CUDA_RETURN_STATUS(GPUMemcpyDtoDAsync(
            kernel_info.args[1], kernel_info.args[0], size, stream
        ));
        return Status::Succ;
    }
    GPUfunction func = kernel_info.handler;
    auto& launch_params = kernel_info.launch_params;
    // std::stringstream ss;
    // ss << func_name << " <<<(" << launch_params[0] << ", " << launch_params[1] << ", " << launch_params[2] << "), ("
    //    << launch_params[3] << ", " << launch_params[4] << ", " << launch_params[5] << ")>>>";
    // for(auto arg_ptr: kernel_info.args_ptr) {
    //     ss << " " << (void*)*arg_ptr;
    // }
    // LOG(INFO) << "Launching kernel: " << ss.str();

    CUDA_RETURN_STATUS(GPULaunchKernel(
        func,
        launch_params[0], launch_params[1], launch_params[2],
        launch_params[3], launch_params[4], launch_params[5],
        0, stream, (void**)(kernel_info.args_ptr.data()), nullptr
    ));
    return Status::Succ;
}

size_t Executor::get_kernel_num() const {
    if(type == "dnn") {
        return model->get_kernel_num();
    } else if(type == "graph") {
        return 1;
    } else {
        return 0;
    }
}

void Executor::capture_graph(GPUstream stream){
    use_cuda_graph = true;
    GPUGraphCreate(&graph, 0);
    GPUStreamBeginCapture(stream, GPU_STREAM_CAPTURE_MODE_GLOBAL);
    if(type == "dnn") {
        for (size_t i = 0; i < get_kernel_num(); i++) {
            launch_kernel(i, stream);
        }
    } else if(type == "graph") {
        use_cuda_graph = false;
        return;
    } else {
        SciComputeRaw::perform_timestep(stream);
    }
    GPUStreamEndCapture(stream, &graph);
#ifdef CUDA
    GPUGraphInstantiate(&graph_exec, graph, 0);
#else
    GPUGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
#endif
}


BaseExecutor::BaseExecutor() {}

BaseExecutor::~BaseExecutor() {}

BLPExecutor::BLPExecutor() {}

BLPExecutor::~BLPExecutor() {
    if(dpStopIndex){
        ASSERT_CUDA_ERROR(GPUMemFree(dpStopIndex));
    }
}

Status BLPExecutor::execute(GPUstream stream) {
    if(type == "dnn") {
        CUDA_RETURN_STATUS(GPUMemsetAsync(executed, 0, sizeof(int) * model->get_kernel_num(), stream));
    }
    else if(type == "sci") {
        CUDA_RETURN_STATUS(GPUMemsetAsync(executed, 0, sizeof(int) * 3 * 12, stream));
    }
    if(use_cuda_graph) {
        CUDA_RETURN_STATUS(GPUGraphLaunch(graph_exec, stream));
        return Status::Succ;
    }
    if(type == "dnn") {
        for (size_t i = 0; i < get_kernel_num(); i++) {
            RETURN_STATUS(launch_kernel(i, stream));
        }
    } else if(type == "graph") {
        graph_compute->compute(stream, (int*)dpStop, (int*)this->executed, *lc_tasks, *be_tasks);
    } else {
        SciComputeBlp::perform_timestep(stream, (int*)dpStop, (int*)dpStopIndex, (int*)executed, 0);
    }
    return Status::Succ;
}

void BLPExecutor::capture_graph(GPUstream stream) {
    use_cuda_graph = true;
    GPUGraphCreate(&graph, 0);
    GPUStreamBeginCapture(stream, GPU_STREAM_CAPTURE_MODE_GLOBAL);
    if(type == "dnn") {
        for (size_t i = 0; i < get_kernel_num(); i++) {
            launch_kernel(i, stream);
        }
    } else if(type == "graph") {
        use_cuda_graph = false;
        return;
    } else {
        SciComputeBlp::perform_timestep(stream, (int*)dpStop, (int*)dpStopIndex, (int*)executed, 0);
    }
    GPUStreamEndCapture(stream, &graph);
    #ifdef CUDA
    GPUGraphInstantiate(&graph_exec, graph, 0);
    #else
    GPUGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
    #endif
}

void BLPExecutor::clear() {
    if(type == "dnn") {
        model->clear();
    } else if(type == "graph") {
        graph_compute.reset();
    } else {
        double mass, te;
        SciComputeBlp::reductions(mass, te);
        // printf("mass: %le, te: %le\n", (mass - mass0)/mass0, (te - te0)/te0);
        SciComputeBlp::finalize();
    }
}

Status BLPExecutor::getStopPoint(GPUstream stream) {
    CUDA_RETURN_STATUS(GPUMemcpyDtoHAsync(&stopIndex, dpStopIndex, sizeof(int), stream));
    CUDA_RETURN_STATUS(GPUStreamSynchronize(stream));
    if(stopIndex == -1) {
        return Status::Succ;
    }
    static int tmp = -1;
    CUDA_RETURN_STATUS(GPUMemcpyHtoDAsync(dpStopIndex, &tmp, sizeof(int), stream));
    return Status::Succ;
}

Status BLPExecutor::resume(GPUstream stream) {
    if(!running || type == "graph") {
        return Status::Succ;
    }
    getStopPoint(stream);
    // printf("Resuming from kernel No. %d\n", stopIndex);
    if(stopIndex == -1) {
        return Status::Succ;
    }
    if(type == "dnn") {
        for(int i = stopIndex; i < get_kernel_num(); i++) {
            RETURN_STATUS(launch_kernel(i, stream));
        }
    } else {
        SciComputeBlp::perform_timestep(stream, (int*)dpStop, (int*)dpStopIndex, (int*)this->executed, stopIndex);
    }
    // LOG(INFO) << "Resuming from kernel No. " << stopIndex << "/" << get_kernel_num();
    return Status::Succ;
}

Status BLPExecutor::init(std::string workload) {
    dpStopIndex = 0;
    if(graphCompute.count(workload)) {
        type = "graph";
        if(workload == "cc") {
            graph_compute = std::make_shared<GraphComputeBlp::cc::CC>(std::string(GRAPH_BASE_PATH) + graph_path);
        } else if(workload == "bfs") {
            graph_compute = std::make_shared<GraphComputeBlp::bfs::BFS>(std::string(GRAPH_BASE_PATH) + graph_path);
        } else if(workload == "pagerank") {
            graph_compute = std::make_shared<GraphComputeBlp::pagerank::PageRank>(std::string(GRAPH_BASE_PATH) + graph_path);
        } else if(workload == "sssp") {
            graph_compute = std::make_shared<GraphComputeBlp::sssp::SSSP>(std::string(GRAPH_BASE_PATH) + graph_path);
        }
        CUDA_RETURN_STATUS(GPUMemAlloc(&executed, sizeof(int) * 3));
        return Status::Succ;
    } else if(sciCompute.count(workload)) {
        type = "sci";
        int tmp = -1;
        CUDA_RETURN_STATUS(GPUMemAlloc(&dpStopIndex, sizeof(int)));
        CUDA_RETURN_STATUS(GPUMemcpyHtoD(dpStopIndex, &tmp, sizeof(int)));
        CUDA_RETURN_STATUS(GPUMemAlloc(&executed, sizeof(int) * 3 * 12));
        SciComputeBlp::sci_init();
        SciComputeBlp::reductions(mass0, te0);
    } else {
        type = "dnn";
        model = std::make_shared<Model>();
        auto model_path = std::string(MODEL_PATH) + "/" + workload;
        RETURN_STATUS(model->load_model(model_path + "/mod.json", model_path + "/host.json", model_path + "/mod.tr.cubin"));
        int tmp = -1;
        CUDA_RETURN_STATUS(GPUMemAlloc(&dpStopIndex, sizeof(int)));
        CUDA_RETURN_STATUS(GPUMemcpyHtoD(dpStopIndex, &tmp, sizeof(int)));
        model->batchAddParam(&this->dpStop);
        model->batchAddParam(&this->dpStopIndex);
        CUDA_RETURN_STATUS(GPUMemAlloc(&executed, sizeof(int) * model->get_kernel_num()));
        model->add_block_preemption_param(executed);
        RETURN_STATUS(this->load_param(model_path));
    }
    stopIndex = -1;
    GPUMemcpyHtoD(dpStopIndex, &stopIndex, sizeof(int));

    return Status::Succ;
    
}


} // namespace foo
