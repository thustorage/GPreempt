#include "executor.h"
#include <vector>
#include <assert.h>
#include <chrono>

#define PATH_OF(name) MODEL_PATH "/" #name

int main(int argc, char **argv) {
    if(argc < 2) {
        printf("Usage: %s <model>\n", argv[0]);
        return 1;
    }
    foo::util::init_cuda();
    foo::BLPExecutor executor;
    GPUstream stream;
    GPUStreamCreate(&stream, 0);
    std::string model_name = argv[1];
    GPUdeviceptr p1, p2;
    CHECK_STATUS(executor.init(model_name));
    ASSERT_CUDA_ERROR(GPUMallocManaged(&p1, 4, GPU_MEM_ATTACH_GLOBAL));
    GPUMemset(p1, 0, 4);
    executor.dpStop = p1;
    size_t input_size = executor.get_data_size("data");;
    size_t output_size = executor.get_data_size("heads");
    if(model_name == "bert") {
        std::vector<int64_t> input({101, 7592, 1010, 2088,  999,  102});
        CHECK_STATUS(executor.set_input("data", input.data(), input_size, stream));
    } else {
        std::vector<float> input(input_size / sizeof(float));
        for(size_t i = 0; i < input.size(); i++) {
            input[i] = 10.0;
        }
        CHECK_STATUS(executor.set_input("data", input.data(), input_size, stream));
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    CHECK_STATUS(executor.execute(stream));
    executor.running = true;
    // usleep(100);
    *(int*)p1 = 1;
    GPUStreamSynchronize(stream);
    *(int*)p1 = 0;
    executor.resume(stream);
    GPUStreamSynchronize(stream);

    auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<float> output(output_size / sizeof(float));
    CHECK_STATUS(executor.get_output("heads", output, stream));
    GPUStreamSynchronize(stream);
    printf("Time: %ldus\n", std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    for(size_t i = 0; i < 10; i++) {
        printf("%f\n", output[i]);
    }
    return 0;
}