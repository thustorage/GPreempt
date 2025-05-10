#include "executor.h"
#include <vector>
#include <assert.h>
#include <chrono>

#define PATH_OF(name) MODEL_PATH "/" #name

int main() {
    foo::util::init_cuda();
    foo::BLPExecutor executor;
    GPUstream stream;
    GPUStreamCreate(&stream, 0);
    std::string model_name = "miniweather";
    GPUdeviceptr p1, p2;
    CHECK_STATUS(executor.init(model_name));
    ASSERT_CUDA_ERROR(GPUMallocManaged(&p1, 4, GPU_MEM_ATTACH_GLOBAL));
    GPUMemset(p1, 0, 4);
    executor.dpStop = p1;
    auto t0 = std::chrono::high_resolution_clock::now();
    CHECK_STATUS(executor.execute(stream));
    // usleep(100);
    *(int*)p1 = 1;
    GPUStreamSynchronize(stream);
    *(int*)p1 = 0;
    executor.resume(stream);
    GPUStreamSynchronize(stream);
    executor.clear();
    auto t1 = std::chrono::high_resolution_clock::now();
    
    return 0;
}