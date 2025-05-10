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
#include <gdrapi.h>
#include "executor.h"

#define PATH_OF(name) MODEL_PATH "/" #name

// blp - block level preemption

using std::chrono::seconds;
using std::chrono::microseconds;
using std::chrono::nanoseconds;

std::mutex mtx;
std::vector<GPUstream> beStreams;
std::vector<std::shared_ptr<foo::BLPExecutor>> beExecutors;
int cnt = 0;

bool initOnce = false;

GPUdeviceptr d_pool;
int *h_pool;
int pool_top;
gdr_mh_t g_mh;
struct GdrEntry {
    GPUdeviceptr d;
    GPUdeviceptr* d_ptr;
    void *cpu_map;
}g_stop;

std::atomic<int> running_be(0);
std::atomic<int> running_lc(0);

std::atomic<int> cores(0);

int get_gdr_map(GdrEntry *entry) {
    if(h_pool == nullptr) {
        gdr_t g = gdr_open();
        if(g == nullptr) {
            LOG(ERROR) << "Failed to open gdr";
            return -1;
        }
        ASSERT_CUDA_ERROR(GPUMemAlloc(&d_pool, GPU_PAGE_SIZE * 2));
        d_pool = (d_pool + (GPU_PAGE_SIZE - 1)) & ~(GPU_PAGE_SIZE - 1);
        gdr_mh_t mh;
        if(gdr_pin_buffer(g, (unsigned long)d_pool, GPU_PAGE_SIZE , 0, 0, &g_mh) != 0) {
            LOG(ERROR) << "Failed to pin input buffer";
            return -1;
        }
        if (gdr_map(g, g_mh, (void**)&h_pool, GPU_PAGE_SIZE ) != 0) {
            LOG(ERROR) << "Failed to map input GPU buffer";
            return -1;
        }
        gdr_info_t info;
        if(gdr_get_info(g, g_mh, &info) != 0) {
            LOG(ERROR) << "Failed to get info";
            return -1;
        }
        int off = info.va - d_pool;
        h_pool = (int*)((char*)h_pool + off);
    }
    entry->d = d_pool + pool_top * sizeof(int);
    entry->d_ptr = &entry->d;
    entry->cpu_map = h_pool + pool_top;
    pool_top++;
    return 0;
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
        for(auto &beexecutor : beExecutors) {
            beexecutor = nullptr;
        }
        beStreams.clear();
        if(g_stop.d) {
            GPUMemFree(g_stop.d);
            g_stop.d = 0;
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
        ASSERT_CUDA_ERROR(GPUMemHostAlloc(&input, input_size, CU_MEMHOSTALLOC_PORTABLE));
        ASSERT_CUDA_ERROR(GPUMemHostAlloc(&output, output_size, CU_MEMHOSTALLOC_PORTABLE));
        if(!initOnce) {
            int res = get_gdr_map(&g_stop);
            assert(0 == res);
            *(int*)g_stop.cpu_map = 0;
            initOnce = true;
        }
        GPUStreamCreate(&stream, 0);
        if(priority != 0) {
            beStreams.push_back(stream);
            auto blp_executor = std::dynamic_pointer_cast<foo::BLPExecutor>(executor);
            beExecutors.push_back(blp_executor);
            blp_executor->dpStop = g_stop.d;
            blp_executor->set_lc_be_tasks(running_lc, running_be);
        }

        if(use_cuda_graph){
            executor->capture_graph(stream);
        }

        LOG(INFO) << "Init " << getName() << " done" << std::endl;
    }

    virtual void initInThread() override {
        bind_core(cores.fetch_add(1));
        foo::util::init_cuda();
        LOG(INFO) << "Init " << getName() << " in thread done" << std::endl;
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
        if(priority != 0){
            foo::BLPExecutor *blp_executor = (foo::BLPExecutor*)(executor.get());
            while(running_lc.load() > 0) {}
            running_be.fetch_add(1);
            blp_executor->running = true;
            blp_executor->execute(stream);
            while(1){
                GPUStreamSynchronize(stream);
                if(running_lc.load() > 0){
                    running_be.fetch_sub(1);
                } else {
                    running_be.fetch_sub(1);
                    blp_executor->running = false;
                    return ;
                }
                while(running_lc.load() > 0) {}
                blp_executor->resume(stream);
                running_be.fetch_add(1);
            }
        } else {
            int cnt = running_lc.fetch_add(1);
            if(cnt == 0) {
                *(int*)g_stop.cpu_map = 1;
            }
            while(running_be.load() > 0) {}
            executor->execute(stream);
            GPUStreamSynchronize(stream);
            cnt = running_lc.fetch_sub(1);
            if(cnt == 1) {
                *(int*)g_stop.cpu_map = 0;
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
    std::shared_ptr<foo::Executor> executor;
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
