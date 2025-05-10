#include <stdio.h>
#include "gpreempt.h"
#include <gdrapi.h>
#include "util/gpu_util.h"

int main(){
    foo::util::init_cuda();
    GPUcontext g_ctx;
    GPUdevice dev;
    GPUDeviceGet(&dev, 0);
    GPUCtxCreate(&g_ctx, 0, dev);
    NvContext nvctx;
    nvctx.hClient = util_gettid();
    NVRMCHECK(NvRmQuery(&nvctx));
    if(set_priority(nvctx, 1)){
        LOG(ERROR) << "Failed to set priority";
        return -1;
    }
    printf("PASS set priority\n");

    // Test GDRcopy
    gdr_t g = gdr_open();
    if(g == nullptr) {
        LOG(ERROR) << "Failed to open gdr";
        return -1;
    }
    GPUdeviceptr mem_pool;
    ASSERT_CUDA_ERROR(GPUMemAlloc(&mem_pool, GPU_PAGE_SIZE  * 2));
    mem_pool = (mem_pool + GPU_PAGE_SIZE ) & ~(GPU_PAGE_SIZE - 1);
    gdr_mh_t g_mh;
    if(gdr_pin_buffer(g, (unsigned long)mem_pool, GPU_PAGE_SIZE , 0, 0, &g_mh) != 0) {
        LOG(ERROR) << "GDRError Failed to pin input buffer";
        return -1;
    }
    int *h_pool;
    if (gdr_map(g, g_mh, (void**)&h_pool, GPU_PAGE_SIZE ) != 0) {
        LOG(ERROR) << "GDRError Failed to map input GPU buffer";
        return -1;
    }
    gdr_info_t info;
    if(gdr_get_info(g, g_mh, &info) != 0) {
        LOG(ERROR) << "GDRError Failed to get info";
        return -1;
    }
    printf("PASS GDRcopy\n");

    // Release GDRcopy
    gdr_unpin_buffer(g, g_mh);
    gdr_close(g);
    
    GPUMemFree(mem_pool);
    GPUMemFreeHost(h_pool);

    printf("PASS all\n");
    return 0;
}