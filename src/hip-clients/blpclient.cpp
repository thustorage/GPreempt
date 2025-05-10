#include "disb.h"

#include <string>
#include <memory>
#include <chrono>
#include <thread>
#include <assert.h>
#include <iostream>
#include <json/json.h>
#include <atomic>
#include <vector>
#include "executor.h"
#include <set>

#define PATH_OF(name) MODEL_PATH "/" #name

// blp - block level preemption

using std::chrono::system_clock;
using std::chrono::seconds;
using std::chrono::microseconds;
using std::chrono::nanoseconds;

std::mutex mtx;
std::vector<GPUstream> beStreams;
std::vector<std::shared_ptr<foo::BLPExecutor>> beExecutors;
int cnt = 0;

std::atomic<int> be_suspended = false;
std::atomic<int> lc_tasks(0);
std::atomic<int> be_tasks(0);
bool initOnce = false;

std::atomic<int> cores(0);

int* g_stop;
int64_t sum_time = 0;
int cnt_time = 0;
void suspend_be() {
    be_suspended.store(1);
    // auto t0 = system_clock::now();
    *g_stop = 1;
    // auto t1 = system_clock::now();
    // sum_time += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    // cnt_time++;
}

void resume_be() {
    *g_stop = 0;
    for(int i = 0; i < beStreams.size(); i++) {
        beExecutors[i]->resume(beStreams[i]);
    }
    be_suspended.store(0);
}

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
        mtx.lock();
        for(auto beexecutor : beExecutors) {
            beexecutor = nullptr;
        }
        beStreams.clear();
        if(g_stop) {
            GPUMemFree(g_stop);
            g_stop = nullptr;
        }
        mtx.unlock();
    }

    virtual void init() override
    {
        foo::util::init_cuda();
        if(priority) 
            executor = std::make_shared<foo::BLPExecutor>();
        else 
            executor = std::make_shared<foo::BaseExecutor>();
        if(!graph_path.empty()) {
            CHECK_STATUS(executor->set_graph_path(graph_path));
        }
        CHECK_STATUS(executor->init(model_name));
        input_size = executor->get_data_size("data");
        output_size = executor->get_data_size("heads");
        if(priority == 1) {
            foo::BLPExecutor *blp_executor = (foo::BLPExecutor*)(executor.get());
            blp_executor->set_lc_be_tasks(lc_tasks, be_tasks);
        }
        ASSERT_CUDA_ERROR(GPUMemHostAlloc(&input, input_size, CU_MEMHOSTALLOC_PORTABLE));
        ASSERT_CUDA_ERROR(GPUMemHostAlloc(&output, output_size, CU_MEMHOSTALLOC_PORTABLE));
        if(!initOnce) {
            GPUMallocManaged((void**)&g_stop, sizeof(int), GPU_MEM_ATTACH_GLOBAL);
            *g_stop = 0;
            initOnce = true;
        }
        GPUStreamCreate(&stream, 0);
        if(priority != 0) {
            beStreams.push_back(stream);
            auto blp_executor = std::dynamic_pointer_cast<foo::BLPExecutor>(executor);
            beExecutors.push_back(blp_executor);
            blp_executor->dpStop = g_stop;
        }

        if(use_cuda_graph){
            executor->capture_graph(stream);
        }

        LOG(INFO) << "Init " << getName() << " done" << std::endl;
    }

    virtual void initInThread() override {
        bind_core(cores.fetch_add(1));
        foo::util::init_cuda();
        initialized = true;
        LOG(INFO) << "Init " << getName() << " in thread done" << std::endl;
    }
    
    virtual void prepareInput() override {}

    virtual void preprocess() override {
        usleep(preprocess_time);
    }

    virtual void copyInput() override
    {
        if(priority == 0) {
            lc_tasks.fetch_add(1);
        }
        CHECK_STATUS(executor->set_input("data", input, input_size, stream));
        GPUStreamSynchronize(stream);
    }

    virtual void infer() override
    {
        if(priority == 1){
            while(lc_tasks.load() != 0) {
            }
            foo::BLPExecutor *blp_executor = (foo::BLPExecutor*)(executor.get());
            blp_executor->running = true;
        }
        executor->execute(stream);
        GPUStreamSynchronize(stream);
        if(priority > 0) {
            do {
                GPUStreamSynchronize(stream);
            } while(be_suspended.load());
        }
        if(priority == 0) {
            lc_tasks.fetch_sub(1);
        }
        if(priority == 1) {
            foo::BLPExecutor *blp_executor = (foo::BLPExecutor*)(executor.get());
            blp_executor->running = false;
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
    std::shared_ptr<foo::Executor> executor;
    GPUstream stream;
    size_t input_size;
    size_t output_size;
    void *input;
    void *output;
    bool use_cuda_graph = false;
    bool initialized = false;
    uint64_t preprocess_time;
    std::string graph_path;
};

void fooDaemon(std::atomic<bool> &stopFlag, int task_cnt)
{
    foo::util::init_cuda();
    int cnt = 0;
    while (!stopFlag.load()) {
        if(lc_tasks.load() == 0 && be_suspended.load() == 1) {
            resume_be();
        }
        if(lc_tasks.load() != 0 && be_suspended.load() == 0) {
            suspend_be();
        }
    }
}

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
    DISB::setDaemonThread(fooDaemon);
    benchmark.init(jsonStr, FooClientFactory);
    benchmark.run();
    Json::Value report = benchmark.generateReport();
    std::cout << report << std::endl;
    return 0;
}