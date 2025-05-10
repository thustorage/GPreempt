// # Original Copyright 2023 SJTU-IPADS
// #
// # Licensed under the Apache License, Version 2.0 (the "License");
// # you may not use this file except in compliance with the License.
// # You may obtain a copy of the License at
// #
// #     http://www.apache.org/licenses/LICENSE-2.0
// #
// # Unless required by applicable law or agreed to in writing, software
// # distributed under the License is distributed on an "AS IS" BASIS,
// # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// # See the License for the specific language governing permissions and
// # limitations under the License.
// #
// # ------------------------------------------------------------------------------
// # MODIFICATIONS:
// # This file has been modified in 2025.
// // #
// # The following changes were made to the original file:
// # - Extract the reef scheduler from the original file.
// # - Remove dynamic kernel padding.
// # 
// # All modifications are also licensed under the Apache License, Version 2.0.
// # ------------------------------------------------------------------------------


#include "reef_scheduler.h"
#include "util/common.h"
#include "util/gpu_util.h"

#define ENABLE_TASK_CV 
namespace reef {
namespace executor {

REEFScheduler::REEFScheduler() : 
    model_pool(this->model_pool_capacity),
    model_pool_size(0),
    task_idx_pool(0),
    be_queues(max_num_be_queues),
    be_queue_cnt(0),
    _shutdown(false),
    preempted(false),
    wait_sync(false),
    preempt_count(0),
    preempt_latency_sum(0),
    kernel_sel_latency_sum(0),
    kernel_sel_count(0)
{
    ASSERT_CUDA_ERROR(GPUStreamCreate(&execute_stream, CU_STREAM_DEFAULT));
    ASSERT_CUDA_ERROR(GPUStreamCreate(&preempt_stream, CU_STREAM_DEFAULT));
    ASSERT_CUDA_ERROR(GPUMemAlloc((void**)&preempt_flag, 4));
    ASSERT_CUDA_ERROR(GPUWriteValue32Async(preempt_stream, preempt_flag, 0, 0));
    ASSERT_CUDA_ERROR(GPUStreamSynchronize(preempt_stream));
    be_stream_device_queue_cap = 2;
}

REEFScheduler::~REEFScheduler() {
    
}


Status REEFScheduler::create_task_queue(std::shared_ptr<TaskQueue>& ret, bool rt) {
    GPUstream stream;
    if (rt) {
        LOG(INFO) << "create rt stream";
        CUDA_RETURN_STATUS(hipStreamCreateWithWindowSize(&stream, 1024));
    } else {
        LOG(INFO) << "create be stream";
        CUDA_RETURN_STATUS(hipStreamCreateWithWindowSize(
            &stream, be_stream_device_queue_cap
        ));
    }
    ret = std::make_shared<TaskQueue>();
    ret->stream = stream;
    return Status::Succ;
}

Status REEFScheduler::load_model(
    const std::string& model_path,
    ModelID& mid
) {
    std::shared_ptr<Model> model(new Model);
    model->executor.set_preempt_flag(preempt_flag);
    RETURN_STATUS(model->executor.load_model(model_path));
    RETURN_STATUS(model->executor.load_param(model_path));
    model->qid = rt_queue_id; // rt queue as default

    auto idx = model_pool_size.fetch_add(1);
    if (idx >= model_pool_capacity) {
        LOG(ERROR) << "model pool is full";
        RETURN_STATUS(Status::Fail);
    }
    model_pool[idx] = std::move(model);
    LOG(INFO) << "load model from " << model_path << ", idx: " << idx;
    mid = idx;
    return Status::Succ;
}

Status REEFScheduler::create_queue(
    const TaskQueueType& qtp,
    QueueID& qid
) {
    if (qtp == TaskQueueType::RealTimeQueue) {
        qid = rt_queue_id;
        if (rt_queue.get() == nullptr) {
            CHECK_STATUS(create_task_queue(rt_queue, true));
        }
        return Status::Succ;
    }
    std::shared_ptr<TaskQueue> q;
    RETURN_STATUS(create_task_queue(q, false));
    {
        // writer lock
        std::unique_lock<std::mutex> lock(be_queues_mtx);
        auto idx = be_queue_cnt;
        if (idx >= max_num_be_queues) RETURN_STATUS(Status::Full);
        be_queues[idx] = std::move(q);
        be_queue_cnt++;
        qid = idx;
    }
    return Status::Succ;
}

Status REEFScheduler::bind_model_queue(
    const QueueID& qid,
    const ModelID& mid
) {
    if (model_pool_size.load() <= mid) RETURN_STATUS(Status::OutOfRange);
    if (be_queue_cnt <= qid && qid != rt_queue_id) RETURN_STATUS(Status::OutOfRange);
    model_pool[mid]->qid = qid;
    return Status::Succ;
}

Status REEFScheduler::new_task(
    const ModelID& mid,
    TaskID& tid
) {
    if (model_pool_size.load() <= mid) RETURN_STATUS(Status::OutOfRange);
    auto &model = model_pool[mid];
    std::shared_ptr<Task> task(new Task);
    task->model = model;
    task->id = task_idx_pool.fetch_add(1);
    task->qid = model->qid;
    task->kernel_offset = 0;
    task->launch_offset = 0;
    task->state = TaskState::Init;
    task->submit = std::chrono::system_clock::now();
    task->preempted = false;
    task->padding = false;
    task->padding_to_finish = false;
    if (model->qid == rt_queue_id) {
        rt_queue->task_queue.push(task);
    } else {
        be_queues[model->qid]->task_queue.push(task);
    }
    tid = task->id;
    {
        std::unique_lock<std::mutex> lock(task_cnt_mtx);
        if (task_cnt == 0) 
            task_cnt_cv.notify_all();
        task_cnt++;
    }
    {
        std::unique_lock<std::mutex> lock(task_pool_mtx);
        task_pool.insert({tid, task});
    }
    return Status::Succ;
}

Status REEFScheduler::get_task(TaskID tid, std::shared_ptr<Task>& t) {
    std::shared_ptr<Task> task;
    {
        std::unique_lock<std::mutex> lock(task_pool_mtx);
        auto res = task_pool.find(tid);
        if (res == task_pool.end()) RETURN_STATUS(Status::NotFound);
        task = res->second;
    }
    t = task;
    return Status::Succ;
}

Status REEFScheduler::wait_task(TaskID tid) {
    std::shared_ptr<Task> task;
    RETURN_STATUS(get_task(tid, task));
#ifdef ENABLE_TASK_CV
    {
        std::unique_lock<std::mutex> lock(task->mtx);
        while (task->state != TaskState::Finish) {
            task->cv.wait(lock);
        }
    }
#else
    while (task->state != TaskState::Finish) {
        usleep(10);
    }
#endif
    // {
    //     std::unique_lock<std::mutex> lock(task_pool_mtx);
    //     auto res = task_pool.find(tid);
    //     if (res == task_pool.end()) RETURN_STATUS(Status::NotFound);
    //     task_pool.erase(res);
    // }
    return Status::Succ;
}

void REEFScheduler::set_wait_sync(bool value) {
    wait_sync = value;
}

Status REEFScheduler::get_data_size(ModelID mid, const std::string& name, size_t& size) {
    if (mid >= model_pool_size.load()) RETURN_STATUS(Status::OutOfRange);
    auto &model = model_pool[mid];
    size = model->executor.get_data_size(name);
    return Status::Succ;
}


Status REEFScheduler::set_input(ModelID mid, const std::string& name, void* data, size_t len) {
    if (mid >= model_pool_size.load()) RETURN_STATUS(Status::OutOfRange);
    GPUstream stream;
    if (model_pool[mid]->qid == rt_queue_id) {
        stream = rt_queue->stream;
    } else {
        stream = be_queues[model_pool[mid]->qid]->stream;
    }
    auto &model = model_pool[mid];
    model->executor.set_input(name, data, len, stream);
    ASSERT_CUDA_ERROR(GPUStreamSynchronize(stream));
    return Status::Succ;
}

Status REEFScheduler::get_output(ModelID mid, const std::string& name, void* data, size_t len) {
    if (mid >= model_pool_size.load()) RETURN_STATUS(Status::OutOfRange);
    GPUstream stream;
    if (model_pool[mid]->qid == rt_queue_id) {
        stream = rt_queue->stream;
    } else {
        stream = be_queues[model_pool[mid]->qid]->stream;
    }
    auto &model = model_pool[mid];
    model->executor.get_output(name, data, len, stream);
    ASSERT_CUDA_ERROR(GPUStreamSynchronize(stream));
    return Status::Succ;
}

Status REEFScheduler::run() {
    if (scheduler.get() != nullptr) RETURN_STATUS(Status::Fail);
    if (rt_queue.get() == nullptr) {
        CHECK_STATUS(create_task_queue(rt_queue, true));
    }
    
    scheduler.reset(new std::thread([this]{
        // make sure the real_time queue is created for convenience.
        while (true) {
            this->loop_body();
            if (this->_shutdown.load()) return;
        }
    }));
    return Status::Succ;
}

Status REEFScheduler::shutdown() {
    _shutdown.store(true);
    scheduler->join();
    return Status::Succ;
}


void REEFScheduler::set_be_stream_cap(int value) {
    be_stream_device_queue_cap = value;
}

void REEFScheduler::loop_body() {
    // Real-time Mode:
    rtmode:
    while (true) {
        if (rt_queue->task_queue.empty()) goto bemode;

        preempt_be_tasks();

        auto rt_task = rt_queue->task_queue.front();
        rt_queue->task_queue.pop();
        execute_rt_task(rt_task);
    }


    // Best-effort Mode:
    bemode:
    auto be_queue_num = be_queue_cnt;
    for (int i = 0; i < be_queue_cnt; i++) {
        auto &be_queue = be_queues[i];
        while (!be_queue->task_queue.empty()) {
            if (!rt_queue->task_queue.empty()) {
                goto rtmode;
            }
            auto be_task = be_queue->task_queue.front();
            execute_be_task(be_task, be_queue);
            if (!rt_queue->task_queue.empty()) {
                goto rtmode;
            }
            if (be_task->state == TaskState::Finish) {
                be_queue->task_queue.pop();
#ifdef ENABLE_TASK_CV
                {
                    std::unique_lock<std::mutex> lock(be_task->mtx);
                    be_task->cv.notify_all();
                }
#endif
                continue;
            }
            break;
        }
    }
}

void REEFScheduler::preempt_reset() {
    // step 1: reset device queue
    // actually, this step should be the second one,
    // but we can overlap this with the host queue reset.
    ASSERT_CUDA_ERROR(GPUWriteValue32Async(preempt_stream, preempt_flag, 1, 0));
    auto num_be_queues = be_queue_cnt;
    
    // step 2: reset host queue
    for (int i = 0; i < num_be_queues; i++) {
        uint64_t temp;
        ASSERT_CUDA_ERROR(GPUClearHostQueue(&temp, be_queues[i]->stream));
        if (!be_queues[i]->task_queue.empty()) {
            auto task = be_queues[i]->task_queue.front();
            if (task->state == TaskState::Executing) {
                LOG(INFO) << task->kernel_offset << ", " << task->launch_offset;
                task->kernel_offset = std::max(
                    task->launch_offset - (int)temp - be_stream_device_queue_cap,
                    task->kernel_offset
                );
                LOG(INFO) << "new kernel_offset " << task->kernel_offset;
                task->state = TaskState::Waiting;
                task->preempted = true;
            }
        }
    }

    // step 3: reset CUs
    for (int i = 0; i < be_stream_device_queue_cap + 1; i++)
        ASSERT_CUDA_ERROR(GPUResetCU());

    ASSERT_CUDA_ERROR(GPUStreamSynchronize(preempt_stream));
    // if (wait_sync) {
    //     // GPUDeviceSynchronize();
    //     for (int i = 0; i < num_be_queues; i++) {
    //         while (GPUStreamQuery(be_queues[i]->stream) != GPU_SUCCESS) {
    //         // if (GPUStreamQuery(preempt_stream) != GPUStatusOK) {
    //             ASSERT_CUDA_ERROR(GPUResetCU());
    //             sched_yield();
    //         }
    //     }
    // }
}

void REEFScheduler::preempt_wait() {
    auto start = std::chrono::system_clock::now();
    int value = 1;
    ASSERT_CUDA_ERROR(hipStreamWriteValue32(preempt_stream, preempt_flag, 1, 0));

    ASSERT_CUDA_ERROR(hipMemcpyHtoDAsync(preempt_flag, &value, 4, preempt_stream));
    ASSERT_CUDA_ERROR(hipStreamSynchronize(preempt_stream));
    auto set_flag = std::chrono::system_clock::now();
    ASSERT_CUDA_ERROR(hipDeviceSynchronize());
    auto end = std::chrono::system_clock::now();
    auto set_flag_duration = std::chrono::duration_cast<std::chrono::microseconds>(set_flag-start).count();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    for (int i = 0; i < be_queue_cnt; i++) {
        auto &be_queue = be_queues[i]->task_queue;
        if (!be_queue.empty()) {
            auto be_task = be_queue.front();
            ASSERT_CUDA_ERROR(hipStreamSynchronize(preempt_stream));
        }
    }
    // LOG(INFO) << "preempt latency: " << duration << "us, set flag: " << set_flag_duration;
}

void REEFScheduler::preempt_be_tasks() {
    if (preempted) return;
    preempted = true;
    LOG(INFO) << "preempt";
    auto start = std::chrono::system_clock::now();

    preempt_reset();

    auto end = std::chrono::system_clock::now();
    preempt_count++;
    auto latency = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    preempt_latency_sum += latency;
    LOG(INFO) << "preempt latency: " << latency << " us";

    for (int i = 0; i < be_queue_cnt; i++) {
        auto &be_queue = be_queues[i]->task_queue;
        if (!be_queue.empty()) {
            auto be_task = be_queue.front();
            if (be_task->state == TaskState::Executing) {
//                 if (be_task->kernel_offset >= be_task->model->executor.num_kernels()) {
//                     be_task->end = std::chrono::system_clock::now();
// #ifdef ENABLE_TASK_CV
//                     {
//                         std::unique_lock<std::mutex> lock(be_task->mtx);
//                         be_task->cv.notify_all();
//                     }
// #endif
//                     be_task->state = TaskState::Finish;
//                     be_queue.pop();
//                 } else {
//                     LOG(INFO) << be_task->kernel_offset;
                    be_task->state = TaskState::Waiting;
            }
        }
    
    }
}

int64_t REEFScheduler::avg_preempt_latency() const {
    if (preempt_count == 0) return 0;
    return preempt_latency_sum / preempt_count;
}

int64_t REEFScheduler::avg_kernel_sel_latency() const {
    if (kernel_sel_count == 0) return 0;
    return kernel_sel_latency_sum / kernel_sel_count;
}

void REEFScheduler::reset_preempt_flag_async() {
    int preempt_value_false = 0;
    hipStreamWriteValue32(preempt_stream, preempt_flag, 0, 0);
}

void REEFScheduler::execute_rt_task(std::shared_ptr<Task> &task) {
    LOG(INFO) << "start rt task";
    task->state = TaskState::Executing;
    task->start = std::chrono::system_clock::now();
    // auto &exe = task->model->executor;
    // exe.execute(rt_queue->stream);

    task->model->executor.execute(rt_queue->stream);
    ASSERT_CUDA_ERROR(GPUStreamSynchronize(rt_queue->stream));
    task->end = std::chrono::system_clock::now();
    task->state = TaskState::Finish;
#ifdef ENABLE_TASK_CV
    {
        std::unique_lock<std::mutex> lock(task->mtx);
        task->cv.notify_all();
    }
#endif
    LOG(INFO) << "rt task finish";
    return;
}

bool GPUStreamEmpty(GPUstream s) {
    hipError_t res = GPUStreamQuery(s);
    return hipSuccess == res;
}

void REEFScheduler::execute_be_task(std::shared_ptr<Task>& task, std::shared_ptr<TaskQueue>& tqueue) {
    switch (task->state) {
    case TaskState::Finish:
        return;
    case TaskState::Executing: {
        // check if the task is finished
        bool finished = GPUStreamEmpty(tqueue->stream);
        if (finished) {
            // TODO: use GPUEvent timestamp
            // LOG(INFO) << "be task finished";
            task->end = std::chrono::system_clock::now();
            task->state = TaskState::Finish;
            task->kernel_offset = 0;
        }
        return;
    }
    case TaskState::Init:
    case TaskState::Waiting: {
        int num_kernels = task->model->executor.get_kernel_num();
        if (task->state == TaskState::Init) {
            task->start = std::chrono::system_clock::now();
        }
        if (preempted) {
            reset_preempt_flag_async();
            preempted = false;
            ASSERT_CUDA_ERROR(GPUStreamSynchronize(preempt_stream));
            // LOG(INFO) << "reset preempt flag";
        }
        int num_launched = 0;
        auto& exe = task->model->executor;
        if (task->kernel_offset >= num_kernels) {
            task->kernel_offset = 0;
            task->end = std::chrono::system_clock::now();
            // LOG(INFO) << "best-effort task done " << be_task->id;
            task->state = TaskState::Finish;
            return;
        }
        task->state = TaskState::Executing;
        for (int i = task->kernel_offset; i < exe.get_kernel_num(); i++) {
            CHECK_STATUS(exe.launch_preempt_kernel(i, tqueue->stream));
            task->launch_offset = i;
            if (!rt_queue->task_queue.empty()) {
                return; // preempt
            }
        }
        // LOG(INFO) << "launch be task";
    }
    }

    return;
}

std::vector<std::chrono::system_clock::time_point> 
    REEFScheduler::Task::get_timestamp() const {
    return std::vector<std::chrono::system_clock::time_point>({
        submit, start, end
    });
}


bool REEFScheduler::Task::is_preempted() const {
    return preempted;
}


} // namespace executor
} // namespace reef