#pragma once

#include <glog/logging.h>
#include <sys/syscall.h>

#define util_gettid() ((pid_t)syscall(SYS_gettid))

#ifndef SM_COUNT
#define SM_COUNT 80
#endif

#if defined(CUDA)

#include <cuda.h>

typedef CUdeviceptr         GPUdeviceptr;
typedef CUcontext           GPUcontext;
typedef CUfunction          GPUfunction;
typedef CUdevice            GPUdevice;
typedef CUmodule            GPUmodule;
typedef CUresult            GPUresult;
typedef CUstream            GPUstream;
typedef CUgraph             GPUgraph;
typedef CUgraphExec         GPUgraphExec;

#define GPUInit                      cuInit
#define GPUDeviceGet                 cuDeviceGet
#define GPUCtxCreate                 cuCtxCreate
#define GPUDevicePrimaryCtxRetain    cuDevicePrimaryCtxRetain
#define GPUCtxSetCurrent             cuCtxSetCurrent
#define GPUCtxDestroy                cuCtxDestroy
#define GPUMemAlloc                  cuMemAlloc
#define GPUMemHostAlloc              cuMemHostAlloc
#define GPUMallocManaged             cuMemAllocManaged
#define GPU_MEM_ATTACH_GLOBAL        CU_MEM_ATTACH_GLOBAL
#define GPUMemFree                   cuMemFree
#define GPUMemFreeHost               cuMemFreeHost
#define GPUMemset                    cuMemsetD8
#define GPUMemsetAsync               cuMemsetD8Async
#define GPUMemcpyHtoD                cuMemcpyHtoD
#define GPUMemcpyDtoH                cuMemcpyDtoH
#define GPUMemcpyDtoD                cuMemcpyDtoD
#define GPUModuleLoad                cuModuleLoad
#define GPUModuleUnload              cuModuleUnload
#define GPUModuleGetFunction         cuModuleGetFunction
#define GPUModuleGetGlobal           cuModuleGetGlobal
#define GPULaunchKernel              cuLaunchKernel    // Runtime API
#define GPUMemcpyHtoDAsync           cuMemcpyHtoDAsync
#define GPUMemcpyDtoHAsync           cuMemcpyDtoHAsync
#define GPUMemcpyDtoDAsync           cuMemcpyDtoDAsync
#define GPUMemcpyToSymbol            cudaMemcpyToSymbol
#define GPUMemcpyAsync               cuMemcpyAsync
#define GPUStreamCreate              cuStreamCreate
#define GPUStreamDestroy             cuStreamDestroy
#define GPUStreamSynchronize         cuStreamSynchronize
#define GPUStreamQuery               cuStreamQuery
#define GPUStreamDefault             cudaStreamDefault
#define GPUCtxGetStreamPriorityRange cuCtxGetStreamPriorityRange
#define GPUStreamCreateWithPriority  cuStreamCreateWithPriority
#define GPUStreamBeginCapture        cuStreamBeginCapture
#define GPUStreamEndCapture          cuStreamEndCapture
#define GPUGraphCreate               cuGraphCreate
#define GPUGraphInstantiate          cuGraphInstantiate
#define GPUGraphLaunch               cuGraphLaunch
char *cuGetErrorStringCompat(CUresult error);
// put cuGetErrorStringCompat into a macro
#define GPUGetErrorString            cuGetErrorStringCompat

#define GPU_STREAM_CAPTURE_MODE_GLOBAL  CU_STREAM_CAPTURE_MODE_GLOBAL
#define GPU_SUCCESS                     CUDA_SUCCESS
#define GPUOccupancyMaxPotentialBlockSizeWithFlags cudaOccupancyMaxPotentialBlockSizeWithFlags
#define GPUModuleOccupancyMaxPotentialBlockSizeWithFlags cuOccupancyMaxPotentialBlockSizeWithFlags

#elif defined(HIP)

#include <hip/hip_runtime.h>

typedef hipDeviceptr_t      GPUdeviceptr;
typedef hipCtx_t            GPUcontext;
typedef hipFunction_t       GPUfunction;
typedef hipDevice_t         GPUdevice;
typedef hipModule_t         GPUmodule;
typedef hipError_t          GPUresult;
typedef hipStream_t         GPUstream;
typedef hipGraph_t          GPUgraph;
typedef hipGraphExec_t      GPUgraphExec;

#define GPUInit                      hipInit
#define GPUDeviceGet                 hipDeviceGet
#define GPUCtxCreate                 hipCtxCreate
#define GPUDevicePrimaryCtxRetain    hipDevicePrimaryCtxRetain
#define GPUCtxSetCurrent             hipCtxSetCurrent
#define GPUCtxDestroy                hipCtxDestroy
#define GPUMemAlloc                  hipMalloc                  // hipMemAlloc
#define GPUMemAllocAsync             hipMallocAsync
#define GPUMallocManaged             hipMallocManaged
#define GPU_MEM_ATTACH_GLOBAL        hipMemAttachGlobal
#define GPUMemHostAlloc              hipHostMalloc
#define CU_MEMHOSTALLOC_PORTABLE     hipHostMallocPortable
#define GPUMemFree                   hipFree                    // hipMemFree
#define GPUMemFreeHost               hipHostFree
#define GPUMemcpyHtoD                hipMemcpyHtoD
#define GPUMemcpyDtoH                hipMemcpyDtoH
#define GPUMemcpyDtoD                hipMemcpyDtoD
#define GPUMemcpyToSymbol            hipMemcpyToSymbol
#define GPUMemset                    hipMemsetD8
#define GPUMemsetAsync               hipMemsetD8Async
#define GPUModuleLoad                hipModuleLoad
#define GPUModuleUnload              hipModuleUnload
#define GPUModuleGetFunction         hipModuleGetFunction
#define GPUModuleGetGlobal           hipModuleGetGlobal
#define GPULaunchKernel              hipModuleLaunchKernel      // hipLaunchKernel
#define GPUMemcpyHtoDAsync           hipMemcpyHtoDAsync
#define GPUMemcpyDtoHAsync           hipMemcpyDtoHAsync
#define GPUMemcpyDtoDAsync           hipMemcpyDtoDAsync
#define GPUMemcpyAsync               hipMemcpyAsync
#define GPUGetErrorString            hipGetErrorString
#define GPUStreamCreate              hipStreamCreateWithFlags   // hipStreamCreate
#define GPUStreamDestroy             hipStreamDestroy
#define GPUStreamSynchronize         hipStreamSynchronize
#define GPUStreamQuery               hipStreamQuery
#define GPUCtxGetStreamPriorityRange hipDeviceGetStreamPriorityRange
#define GPUStreamCreateWithPriority  hipStreamCreateWithPriority
#define GPUStreamBeginCapture        hipStreamBeginCapture
#define GPUStreamEndCapture          hipStreamEndCapture
#define GPUGraphCreate               hipGraphCreate
#define GPUGraphInstantiate          hipGraphInstantiate
#define GPUGraphLaunch               hipGraphLaunch
#define GPUGetErrorString            hipGetErrorString
#define GPUStreamDefault             hipStreamDefault
#define GPU_STREAM_CAPTURE_MODE_GLOBAL  hipStreamCaptureModeGlobal
#define GPU_SUCCESS                     hipSuccess
#define GPUStreamEmpty                 hipStreamIsLazy
#define GPUResetCU hipResetWavefronts
#define GPUClearHostQueue hipStreamClearQueue
#define GPUWriteValue32Async hipStreamWriteValue32
#define CU_STREAM_DEFAULT hipStreamDefault
#define GPUModuleOccupancyMaxPotentialBlockSizeWithFlags hipModuleOccupancyMaxPotentialBlockSizeWithFlags
#define GPUOccupancyMaxPotentialBlockSizeWithFlags hipOccupancyMaxPotentialBlockSizeWithFlags

#endif

#define ASSERT_CUDA_ERROR(cmd)\
{\
    GPUresult error = cmd;\
    if (error != GPU_SUCCESS) {\
        const char* str = GPUGetErrorString(error);\
        LOG(ERROR) << "GPU error: " << error << " " << str << " at " << __FILE__ << ":" << __LINE__; \
        exit(EXIT_FAILURE);\
    }\
}

#define CUDA_RETURN_STATUS(cmd) \
{\
    GPUresult error = cmd;\
    if (error != GPU_SUCCESS) {\
        const char* str = GPUGetErrorString(error);\
        std::string err_str(str);\
        std::cout << "GPU error: " << error << " " << err_str << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return Status::Fail;\
    }\
}

namespace foo {
namespace util {

// Initialize CUDA once globally
// This function is idempotent.
void init_cuda(); 

class SizedGPUBuffer {
public:
    SizedGPUBuffer(size_t size);
    ~SizedGPUBuffer();

    GPUdeviceptr ptr() const;
    size_t size() const;

    void from_vector(std::vector<float>& vec);
    void to_vector(std::vector<float>& vec);
private:
    GPUdeviceptr ptr_;
    size_t size_;
};

}
}