#pragma once

#include <string>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <unordered_map>
#include <thread>
#include "threadsafe_queue.h"
#include "util/common.h"
#include "util/gpu_util.h"
#include "hybrid_executor.h"

namespace reef {
namespace executor {

class REEFScheduler {
    class Model;
public:
    typedef uint32_t ModelID;
    typedef uint32_t QueueID;
    typedef uint32_t TaskID;

    enum TaskQueueType {
        RealTimeQueue,
        BestEffortQueue,
    };

    enum TaskState {
        Init,
        Waiting,
        Executing,
        Finish    
    };

    struct Task {
        friend REEFScheduler;
    private:
        std::shared_ptr<Model> model;
        QueueID qid;
        TaskID id;
        volatile TaskState state;
        int launch_offset; // the kernel idx that has been launched to host queue
        int kernel_offset; // the kernel idx that has been executed

        std::mutex mtx;
        std::condition_variable cv;
        std::chrono::system_clock::time_point submit; // when this task is created
        std::chrono::system_clock::time_point start; // when this task is scheduled
        std::chrono::system_clock::time_point end; // when this task is completed
        bool preempted;
        bool padding;
        bool padding_to_finish;
    public:
        bool is_preempted() const;
        
        std::vector<std::chrono::system_clock::time_point> get_timestamp() const;
    };

public:
    REEFScheduler();
    ~REEFScheduler();

    Status load_model(
        const std::string& model_dir,
        ModelID& mid
    );


    Status create_queue(
        const TaskQueueType& qtp,
        QueueID& qid
    );

    Status bind_model_queue(
        const QueueID& qid,
        const ModelID& mid
    );

    Status get_data_size(ModelID mid, const std::string& name, size_t& size);

    Status set_input(ModelID mid, const std::string& name, void* data, size_t len);

    Status get_output(ModelID mid, const std::string& name, void* data, size_t len);

    Status new_task(
        const ModelID& mid,
        TaskID& tid
    );

    Status wait_task(
        TaskID tid
    );

    Status get_task(
        TaskID tid,
        std::shared_ptr<Task>& t
    );

    void set_wait_sync(bool value);

    void set_be_stream_cap(int value);
    Status run();
    Status shutdown();

    int64_t avg_preempt_latency() const;
    
    int64_t avg_kernel_sel_latency() const;
private:
    const size_t model_pool_capacity = 1024;
    std::atomic_uint32_t model_pool_size;
    struct Model {
        executor::HybridExecutor executor;
        QueueID qid;
    };
    std::vector<std::shared_ptr<Model>> model_pool;


    std::atomic_uint32_t task_idx_pool;
    std::unordered_map<TaskID, std::shared_ptr<Task>> task_pool;
    std::mutex task_pool_mtx;

    struct TaskQueue {
        ThreadSafeQueue<std::shared_ptr<Task>> task_queue;
        GPUstream stream;
    };

    const size_t max_num_be_queues = 32;
    const QueueID rt_queue_id = 32; // the same with be queue num
    std::mutex be_queues_mtx;
    std::vector<std::shared_ptr<TaskQueue>> be_queues;
    volatile uint32_t be_queue_cnt;
    std::shared_ptr<TaskQueue> rt_queue;
    std::mutex task_cnt_mtx;
    std::condition_variable task_cnt_cv; // To wake up the scheduler
    volatile uint32_t task_cnt;
    bool wait_sync;

    std::unique_ptr<std::thread> scheduler;
    GPUstream execute_stream, preempt_stream;
    GPUdeviceptr preempt_flag;
    bool preempted;
    int be_stream_device_queue_cap;
    std::atomic_bool _shutdown;

    uint64_t preempt_count;
    uint64_t preempt_latency_sum;

    uint64_t kernel_sel_count;
    uint64_t kernel_sel_latency_sum;
private:
    Status create_task_queue(std::shared_ptr<TaskQueue>& ret, bool rt);
    void loop_body();
    void execute_be_task(std::shared_ptr<Task>& task, std::shared_ptr<TaskQueue>& tqueue);
    void execute_rt_task(std::shared_ptr<Task>& task);
    void preempt_be_tasks();
    void reset_preempt_flag_async();
    void preempt_reset();
    void preempt_wait();
};

} // namespace executor
} // namespace reef