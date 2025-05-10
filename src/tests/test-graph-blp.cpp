#include <stdio.h>
#include <string>
#include "util/gpu_util.h"
#include <chrono>
#include "workloads/graphcompute.h"

int main(int argc, char **argv){
    auto t1 = std::chrono::high_resolution_clock::now();
    static bool initialized = false;
    if (!initialized) {
        ASSERT_CUDA_ERROR(GPUInit(0));
        initialized = true;
    }
    GPUdevice device;
    ASSERT_CUDA_ERROR(GPUDeviceGet(&device, 0));
    GPUcontext context;
    ASSERT_CUDA_ERROR(GPUDevicePrimaryCtxRetain(&context, device));
    ASSERT_CUDA_ERROR(GPUCtxSetCurrent(context));
    auto t2 = std::chrono::high_resolution_clock::now();
    std::string dataset("/home/frw/workdir/dataset/crankseg_1/crankseg_1.bel");
    GraphComputeBlp::GraphCompute *pr;
    if(std::string(argv[1]) == "cc") pr = new GraphComputeBlp::cc::CC(dataset);
    else if(std::string(argv[1]) == "bfs") pr = new GraphComputeBlp::bfs::BFS(dataset);
    else if(std::string(argv[1]) == "pagerank") pr = new GraphComputeBlp::pagerank::PageRank(dataset);
    else if(std::string(argv[1]) == "sssp") pr = new GraphComputeBlp::sssp::SSSP(dataset);
    auto t3 = std::chrono::high_resolution_clock::now();
    GPUstream stream;
    GPUStreamCreate(&stream, 0);
    std::atomic<int> lc_tasks(0);
    std::atomic<int> be_tasks(0);
    int *preempted;
    int *executed;
    GPUMemAlloc((GPUdeviceptr*)&preempted, sizeof(int));
    GPUMemAlloc((GPUdeviceptr*)&executed, sizeof(int) * 3);
    GPUMemsetAsync((GPUdeviceptr)preempted, 0, sizeof(int), stream);
    GPUMemsetAsync((GPUdeviceptr)executed, 0, sizeof(int) * 3, stream);
    pr->compute(stream, preempted, executed, lc_tasks, be_tasks);
    t3 = std::chrono::high_resolution_clock::now();
    pr->compute(stream, preempted, executed, lc_tasks, be_tasks);

    auto t4 = std::chrono::high_resolution_clock::now();
    auto diff1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    auto diff2 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2);
    auto diff3 = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3);
    printf("GPU Init: %ldus\n", diff1.count());
    printf("BFS Init: %ldus\n", diff2.count());
    printf("BFS Compute: %ldus\n", diff3.count());
    delete pr;
    return 0;
}