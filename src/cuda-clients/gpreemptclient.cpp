#include <string>
#include <memory>
#include <chrono>
#include <set>
#include <thread>
#include <assert.h>
#include <iostream>
#include <mutex>
#include <json/json.h>
#include <atomic>
#include <gdrapi.h>

#include "disb.h"
#include "executor.h"
#include "gpreempt.h"
#include "block.h"


#define PATH_OF(name) MODEL_PATH "/" #name

#define SWITCH_TIME 100

using std::chrono::system_clock;
using std::chrono::seconds;
using std::chrono::microseconds;
using std::chrono::nanoseconds;

// tsp - Time Slice with Preprocess
// Base on time slice version, add optimizations during preprocess
bool reserve_preempt = true;
GPUdeviceptr d_pool;
int *h_pool;
int pool_top;
gdr_mh_t g_mh;
struct GdrEntry {
    GPUdeviceptr d;
    GPUdeviceptr* d_ptr;
    void *cpu_map;
};

std::atomic<std::chrono::_V2::system_clock::rep> avg_time;
std::atomic<int> cnt;
void record(std::chrono::high_resolution_clock::time_point t0, std::chrono::high_resolution_clock::time_point t1) {
    avg_time.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    cnt.fetch_add(1);
}

std::mutex mtx;
GPUcontext g_ctx[2];

struct DeamonHint {
    system_clock::time_point t;
    GPUstream stream;
    GdrEntry *signal;
};
std::atomic<DeamonHint*> g_hint = nullptr;

bool operator < (const DeamonHint &a, const DeamonHint &b) {
    return a.t < b.t;
}

GPUmodule block_module;
GPUfunction block_function;

std::atomic<int> cores(0);

void load_block() {
    std::string cubin_path = BUILD_PATH "/block.cubin";
    std::string kernel_name = "gpu_block";
    ASSERT_CUDA_ERROR(GPUModuleLoad(&block_module, cubin_path.c_str()));
    ASSERT_CUDA_ERROR(GPUModuleGetFunction(&block_function, block_module, kernel_name.c_str()));
}

void start_blocking_stream(GPUstream stream, GdrEntry *signal) {
    ASSERT_CUDA_ERROR(GPULaunchKernel(block_function, 1, 1, 1, 1, 1, 1, 0, stream, (void**)&signal->d_ptr, nullptr));
    ASSERT_CUDA_ERROR(GPULaunchKernel(block_function, 1, 1, 1, 1, 1, 1, 0, stream, (void**)&signal->d_ptr, nullptr));
}

void end_blocking_stream(GdrEntry *signal) {
    *(int*)signal->cpu_map = 1;
}

int get_gdr_map(GdrEntry *entry) {
    if(h_pool == nullptr) {
        gdr_t g = gdr_open();
        if(g == nullptr) {
            LOG(ERROR) << "Failed to open gdr";
            return -1;
        }
        ASSERT_CUDA_ERROR(GPUMemAlloc(&d_pool, GPU_PAGE_SIZE  * 2));
        d_pool = (d_pool + GPU_PAGE_SIZE ) & ~(GPU_PAGE_SIZE - 1);
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
        if(!priority) {
            int res = get_gdr_map(&gdr_stop);
            assert(res == 0);
        }
        GPUStreamCreate(&stream, 0);
        GPUStreamCreate(&copy_stream, 0);
        CHECK_STATUS(executor->init(model_name));
        ASSERT_CUDA_ERROR(GPUMemHostAlloc(&input, input_size, CU_MEMHOSTALLOC_PORTABLE));
        ASSERT_CUDA_ERROR(GPUMemHostAlloc(&output, output_size, CU_MEMHOSTALLOC_PORTABLE)); 
        if(priority == 0 && block_module == nullptr)
            load_block();

        if(use_cuda_graph){
            executor->capture_graph(stream);
        }

        mtx.unlock();
        initialized = 1;
        LOG(INFO) << "Init in thread " << getName() << " done" << std::endl;
    }
    
    virtual void prepareInput() override {}

    virtual void preprocess() override {
        if(priority == 0 && initialized) {
            DeamonHint *hint;
            *(int*)(gdr_stop.cpu_map) = 0;
            if(reserve_preempt){
                hint = new DeamonHint({system_clock::now() + microseconds(preprocess_time - SWITCH_TIME), stream, &gdr_stop});
                DeamonHint *nu = nullptr;
                while(!g_hint.compare_exchange_weak(nu, hint)) {
                    nu = nullptr;
                }
            } else {
                start_blocking_stream(this->stream, &gdr_stop);
            }
        }
        usleep(preprocess_time);;
        // usleep(preprocess_time);
    }

    virtual void copyInput() override
    {
        if(initialized) {
            CHECK_STATUS(executor->set_input("data", input, input_size, copy_stream));
            GPUStreamSynchronize(copy_stream);
        } else {
            CHECK_STATUS(executor->set_input("data", input, input_size, stream));
            GPUStreamSynchronize(stream);
        }
    }

    virtual void infer() override
    {
        if(!priority && initialized) {
            executor->execute(stream);
            end_blocking_stream(&gdr_stop);
        } else {
            executor->execute(stream);
        }
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
    GPUstream copy_stream;
    size_t input_size;
    size_t output_size;
    void *input;
    void *output;
    GdrEntry gdr_stop;
    int initialized = 0;
    bool use_cuda_graph = false;
    uint64_t preprocess_time;
    std::string graph_path;
};

void fooDaemon(std::atomic<bool> &stopFlag, int task_cnt)
{
    bind_core(cores.fetch_add(1));

    while(g_ctx[0] == nullptr) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    GPUCtxSetCurrent(g_ctx[0]);

    std::set<DeamonHint> hints;
    DeamonHint *nu = nullptr;
    DeamonHint *hint = nullptr;
    while (!stopFlag.load()) {
        hint = nullptr;
        if(!g_hint.compare_exchange_weak(hint, nu)) {
            g_hint.store(nu);
            hints.insert(*hint);
            delete hint;
        }
        while(hints.size() && system_clock::now() > hints.begin()->t) {
            start_blocking_stream(hints.begin()->stream, hints.begin()->signal);
            hints.erase(hints.begin());
        }
    }
}

std::shared_ptr<DISB::Client> FooClientFactory(const Json::Value &config)
{
    return std::make_shared<FooClient>(config);
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " config.json <reserve_preempt=true>" << std::endl;
        return -1;
    }
    if (argc == 3) {
        std::string arg = argv[2];
        if (arg == "true") {
            reserve_preempt = true;
        } else if (arg == "false") {
            reserve_preempt = false;
        } else {
            std::cout << "Invalid argument: " << arg << std::endl;
            return -1;
        }
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
