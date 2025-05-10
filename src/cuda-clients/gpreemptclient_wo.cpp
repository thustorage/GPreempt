#include "disb.h"

#include <string>
#include <memory>
#include <chrono>
#include <thread>
#include <assert.h>
#include <iostream>
#include <mutex>
#include <json/json.h>
#include <gdrapi.h>
#include "executor.h"
#include "gpreempt.h"
#include "block.h"

#define PATH_OF(name) MODEL_PATH "/" #name

// Simply modify time slices of different priority

std::mutex mtx;
GPUcontext g_ctx[2];
std::atomic<int> cores(0);

class FooClient: public DISB::DependentClient
{
public:
    FooClient(const Json::Value &config) {
        model_name = config["model_name"].asString();
        priority = config["priority"].asInt();
        preprocess_time = config["preprocess_time"].asUInt64();
        setName(config["name"].asString());
        if(config.isMember("use_cuda_graph")) {
            use_cuda_graph = config["use_cuda_graph"].asBool();
        }
        if(config.isMember("graph_path")) {
            graph_path = config["graph_path"].asString();
        }
    }
    
    ~FooClient() {
        executor->clear();
        GPUMemFreeHost(input);
        GPUMemFreeHost(output);
    }

    virtual void init() override
    {
        foo::util::init_cuda();
        executor = std::make_shared<foo::BaseExecutor>();
        if(!graph_path.empty()) {
            CHECK_STATUS(executor->set_graph_path(graph_path));
        }
        CHECK_STATUS(executor->init(model_name));
        input_size = executor->get_data_size("data");
        output_size = executor->get_data_size("heads");
        GPUStreamCreate(&stream, 0);
        ASSERT_CUDA_ERROR(GPUMemHostAlloc(&input, input_size, CU_MEMHOSTALLOC_PORTABLE));
        ASSERT_CUDA_ERROR(GPUMemHostAlloc(&output, output_size, CU_MEMHOSTALLOC_PORTABLE));

        if(use_cuda_graph){
            executor->capture_graph(stream);
        }

        LOG(INFO) << "Init " << getName() << " done" << std::endl;
    }

    virtual void initInThread() override 
    {
        bind_core(cores.fetch_add(1));
        executor->clear();
        GPUMemFreeHost(input);
        GPUMemFreeHost(output);
        GPUStreamDestroy(stream);
        GPUdevice dev;
        GPUDeviceGet(&dev, 0);
        mtx.lock();
        if(g_ctx[priority] == nullptr) {
            GPUCtxCreate(&g_ctx[priority], 0, dev);
            NvContext nvctx;
            nvctx.hClient = util_gettid();
            NVRMCHECK(NvRmQuery(&nvctx));
            if(set_priority(nvctx, priority)){
                LOG(ERROR) << "Failed to set priority";
                exit(-1);
            }
        }
        GPUCtxSetCurrent(g_ctx[priority]);
        GPUStreamCreate(&stream, 0);
        CHECK_STATUS(executor->init(model_name));
        ASSERT_CUDA_ERROR(GPUMemHostAlloc(&input, input_size, CU_MEMHOSTALLOC_PORTABLE));
        ASSERT_CUDA_ERROR(GPUMemHostAlloc(&output, output_size, CU_MEMHOSTALLOC_PORTABLE)); 

        if(use_cuda_graph){
            executor->capture_graph(stream);
        }

        mtx.unlock();
        LOG(INFO) << "Init in thread " << getName() << " done" << std::endl;
    }
    
    virtual void prepareInput() override {}

    virtual void preprocess() override {
        usleep(preprocess_time);
    }

    virtual void copyInput() override
    {
        CHECK_STATUS(executor->set_input("data", input, input_size, stream));
    }

    virtual void infer() override
    {
        executor->execute(stream);
        GPUStreamSynchronize(stream);
    }

    virtual void copyOutput() override
    {
        CHECK_STATUS(executor->get_output("heads", output, output_size, stream));
        GPUStreamSynchronize(stream);
    }

    virtual void postprocess() override {}
private:
    std::string model_name;
    int priority;
    std::shared_ptr<foo::BaseExecutor> executor;
    GPUstream stream;
    size_t input_size;
    size_t output_size;
    void *input;
    void *output;
    bool use_cuda_graph = false;
    uint64_t preprocess_time;
    std::string graph_path;
};

std::shared_ptr<DISB::Client> FooClientFactory(const Json::Value &config)
{
    return std::make_shared<FooClient>(config);
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " config.json" << std::endl;
        return -1;
    }

    DISB::BenchmarkSuite benchmark;
    std::string jsonStr = readStringFromFile(argv[1]);
    benchmark.init(jsonStr, FooClientFactory);
    benchmark.run();
    Json::Value report = benchmark.generateReport();
    std::cout << report << std::endl;

    return 0;
}