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
    GraphComputeRaw::GraphCompute *pr;
    if(std::string(argv[1]) == "cc") pr = new GraphComputeRaw::cc::CC(dataset);
    else if(std::string(argv[1]) == "bfs") pr = new GraphComputeRaw::bfs::BFS(dataset);
    else if(std::string(argv[1]) == "pagerank") pr = new GraphComputeRaw::pagerank::PageRank(dataset);
    else if(std::string(argv[1]) == "sssp") pr = new GraphComputeRaw::sssp::SSSP(dataset);
    else {
        printf("Invalid argument. Please use 'cc', 'bfs', 'pagerank', or 'sssp'.\n");
        return 1;
    }
    
    auto t3 = std::chrono::high_resolution_clock::now();
    GPUstream stream;
    GPUStreamCreate(&stream, 0);
    pr->compute(stream);
    t3 = std::chrono::high_resolution_clock::now();
    pr->compute(stream);
    auto t4 = std::chrono::high_resolution_clock::now();
    auto diff1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    auto diff2 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2);
    auto diff3 = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3);
    printf("GPU Init: %ldus\n", diff1.count());
    printf("BFS Init: %ldus\n", diff2.count());
    printf("BFS Compute: %ldus\n", diff3.count());
    
    return 0;
}