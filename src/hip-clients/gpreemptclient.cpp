#include "disb.h"

#include <string>
#include <memory>
#include <chrono>
#include <thread>
#include <assert.h>
#include <iostream>
#include <mutex>
#include <set>
#include <atomic>
#include <json/json.h>
#include "executor.h"
#include "gpreempt.h"

#define PATH_OF(name) MODEL_PATH "/" #name
#define ASSERT_SUCC(expr) assert(Status::Succ == expr)

#define SWITCH_TIME 200
#define MAX_QUEUE_NUM 12

// Disable and Resume version

using std::chrono::system_clock;
using std::chrono::seconds;
using std::chrono::microseconds;
using std::chrono::nanoseconds;

bool initOnce = false;
std::atomic<int> lc_cnt;

std::mutex stream_op_mutex;
std::vector<int> queue_ids;
std::atomic<int> cores(0);
std::atomic<int> be_tasks[32];
GPUstream stream_pool[20];
int stream_cnt = 0;

hipStream_t getStream(int priority, int &stream_id) {
    std::lock_guard<std::mutex> lock(stream_op_mutex);
    hipStream_t ret = stream_pool[stream_cnt++];
    if(priority == 1) {
        int sid;
        hipStreamQueryId(ret, &sid);
        stream_id = queue_ids.size();
        queue_ids.push_back(sid);
    }
    return ret;
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
        GPUMemFreeHost(input);
        GPUMemFreeHost(output);
    }

    virtual void init() override
    {
        foo::util::init_cuda();
        if(!initOnce) {
            for(int i = 0; i < MAX_QUEUE_NUM; i++) {
                GPUStreamCreate(&stream_pool[i], 0);
            }
            initOnce = true;
        }
        stream = getStream(priority, be_cnt);

        executor = std::make_shared<foo::BaseExecutor>();
        if(!graph_path.empty()) {
            CHECK_STATUS(executor->set_graph_path(graph_path));
        }
        ASSERT_SUCC(executor->init(model_name));
        input_size = executor->get_data_size("data");
        output_size = executor->get_data_size("heads");
        ASSERT_CUDA_ERROR(GPUMemHostAlloc(&input, input_size, CU_MEMHOSTALLOC_PORTABLE));
        ASSERT_CUDA_ERROR(GPUMemHostAlloc(&output, output_size, CU_MEMHOSTALLOC_PORTABLE));
        if(use_cuda_graph){
            executor->capture_graph(stream);
        }
        LOG(INFO) << "Init " << getName() << " done" << std::endl;
    }

    virtual void initInThread() override {
        initialized = 1;
        bind_core(cores.fetch_add(1));
        LOG(INFO) << "Init in thread " << getName() << " done" << std::endl;
    }
    
    virtual void prepareInput() override {}

    virtual void preprocess() override {
        if(priority == 0){
            lc_cnt.fetch_add(1);
        }
        usleep(preprocess_time);

    }

    virtual void copyInput() override
    {
        ASSERT_SUCC(executor->set_input("data", input, input_size, stream));
        GPUStreamSynchronize(stream);
    }

    virtual void infer() override
    {
        if(priority == 1){
            while(lc_cnt.load() != 0) {
            }
            be_tasks[be_cnt].fetch_add(1);
        }
        executor->execute(stream);
        GPUStreamSynchronize(stream);
        if(priority == 1){
            be_tasks[be_cnt].fetch_sub(1);
        } else {
            lc_cnt.fetch_sub(1);
        }
    }

    virtual void copyOutput() override
    {
        ASSERT_SUCC(executor->get_output("heads", output, output_size, stream));
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
    int initialized = 0;
    uint64_t preprocess_time;
    std::string graph_path;
    int be_cnt;
};

void fooDaemon(std::atomic<bool> &stopFlag, int task_cnt) {
    int fd = hipGetFd();
    std::vector<int> queues;
    bool suspended = false;

    while (!stopFlag.load()) {
        if(lc_cnt.load() != 0 && suspended == false) {
            queues.clear();
            for(int j = 0; j < queue_ids.size(); j++) {
                if(be_tasks[j].load() != 0) {
                    queues.push_back(queue_ids[j]);
                }
            }
            suspended = true;
            hipSuspendStreams(fd, queues);
        }
        if(lc_cnt.load() == 0 && suspended == true) {
            suspended = false;
            hipResumeStreams(fd, queues);
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
    setenv("GPU_MAX_HW_QUEUES", "20", 1);

    DISB::BenchmarkSuite benchmark;
    std::string jsonStr = readStringFromFile(argv[1]);
    DISB::setDaemonThread(fooDaemon);
    benchmark.init(jsonStr, FooClientFactory);
    benchmark.run();
    Json::Value report = benchmark.generateReport();
    std::cout << report << std::endl;

    return 0;
}
