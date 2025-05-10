#include "disb.h"

#include <string>
#include <memory>
#include <chrono>
#include <thread>
#include <assert.h>
#include <iostream>
#include <json/json.h>
#include "executor.h"
#include <mutex>
#include <atomic>

std::mutex mtx;
std::atomic<int> lc_tasks(0);
std::mutex lock;
std::atomic<int> cores(0);
#define PATH_OF(name) MODEL_PATH "/" #name

// Stream priority version
// All the threads share the same context but have different streams with different priorities

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
        GPUMemFreeHost(input);
        GPUMemFreeHost(output);
    }

    virtual void initInThread() override {
        bind_core(cores.fetch_add(1));
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

        ASSERT_CUDA_ERROR(GPUStreamCreate(&stream, 0));

        if(use_cuda_graph){
            executor->capture_graph(stream);
        }
        LOG(INFO) << "Init " << getName() << " done" << std::endl;
    }
    
    virtual void prepareInput() override {}

    virtual void preprocess() override {
        usleep(preprocess_time);
    }

    virtual void copyInput() override
    {
        CHECK_STATUS(executor->set_input("data", input, input_size, stream));
        GPUStreamSynchronize(stream);
    }

    virtual void infer() override
    {
        if(priority == 0){
            lc_tasks.fetch_add(1);
            lock.lock();
            executor->execute(stream);
            GPUStreamSynchronize(stream);
            lc_tasks.fetch_sub(1);
            lock.unlock();
        } else {
            while(1){
                while(lc_tasks.load() > 0){}
                if(lock.try_lock()){
                    executor->execute(stream);
                    GPUStreamSynchronize(stream);
                    lock.unlock();
                    break;
                }
            }
        }
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
