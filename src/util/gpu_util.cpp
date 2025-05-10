#include "util/gpu_util.h"

static const char *err_str;
#if defined(CUDA)
char *cuGetErrorStringCompat(CUresult error) {
    cuGetErrorString(error, &err_str);
    return (char *)err_str;
}
#endif

namespace foo {
namespace util {

void init_cuda() {
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
}



SizedGPUBuffer::SizedGPUBuffer(size_t size) {
    ASSERT_CUDA_ERROR(GPUMemAlloc(&ptr_, size));
    size_ = size;
}

SizedGPUBuffer::~SizedGPUBuffer() {
    GPUMemFree(ptr_);
}

GPUdeviceptr SizedGPUBuffer::ptr() const {
    return ptr_;
}

size_t SizedGPUBuffer::size() const {
    return size_;
}

void SizedGPUBuffer::from_vector(std::vector<float>& vec) {
    size_t size = std::min(size_, vec.size() * sizeof(float));
    GPUMemcpyHtoD(ptr_, vec.data(), size);
}

void SizedGPUBuffer::to_vector(std::vector<float>& vec) {
    vec.resize(size_ / sizeof(float));
    GPUMemcpyDtoH(vec.data(), ptr_, size_);
}

} // namespace util
} 