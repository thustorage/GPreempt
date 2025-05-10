/*
Modified from https://github.com/illinois-impact/EMOGI
Only keep the whole graph in GPU version 
Add blocklevel preemption mechanism
*/

#include <stdio.h>
#include "util/gpu_util.h"
#include <chrono>
#include "workloads/graphcompute.h"
#include <iostream>
#include <atomic>

namespace GraphComputeBlp {

void read_graph(const std::string &filename, uint64_t &vertex_cnt, uint64_t &edge_cnt, uint64_t *&vertexList, EdgeT *&edgeList) {
    auto vertex_file = filename + ".col";
    auto edge_file = filename + ".dst";

    FILE *vertex_fp = fopen(vertex_file.c_str(), "rb");
    FILE *edge_fp = fopen(edge_file.c_str(), "rb");

    if(vertex_fp == NULL || edge_fp == NULL) {
        printf("Error: file open failed\n");
        throw std::runtime_error("Error: file open failed");
    }

    uint64_t typeT;

    fread(&vertex_cnt, sizeof(uint64_t), 1, vertex_fp);
    fread(&typeT, sizeof(uint64_t), 1, vertex_fp);

    vertex_cnt--;

    fread(&edge_cnt, sizeof(uint64_t), 1, edge_fp);
    fread(&typeT, sizeof(uint64_t), 1, edge_fp);

    vertexList = new uint64_t[vertex_cnt + 1];
    edgeList = new EdgeT[edge_cnt];

    fread(vertexList, sizeof(uint64_t), vertex_cnt + 1, vertex_fp);
    fread(edgeList, sizeof(EdgeT), edge_cnt, edge_fp);

    fclose(vertex_fp);
    fclose(edge_fp);
}

void copy_gpu_graph(uint64_t vertex_cnt, uint64_t edge_cnt, uint64_t *vertexList_h, EdgeT *edgeList_h, uint64_t *&vertexList_d, EdgeT *&edgeList_d) {
    ASSERT_CUDA_ERROR(GPUMemAlloc((GPUdeviceptr*)&vertexList_d, (vertex_cnt + 1) * sizeof(uint64_t)));
    ASSERT_CUDA_ERROR(GPUMemcpyHtoD((GPUdeviceptr)vertexList_d, vertexList_h, (vertex_cnt + 1) * sizeof(uint64_t)));
    
    ASSERT_CUDA_ERROR(GPUMemAlloc((GPUdeviceptr*)&edgeList_d, edge_cnt * sizeof(EdgeT)));
    ASSERT_CUDA_ERROR(GPUMemcpyHtoD((GPUdeviceptr)edgeList_d, edgeList_h, edge_cnt * sizeof(EdgeT)));
}

namespace cc {
__device__ void cc_kernel_coalesce__blp(dim3 taskIdx, bool *curr_visit, bool *next_visit, uint64_t vertex_count, uint64_t *vertexList, EdgeT *edgeList, unsigned long long *comp, bool *changed);

__global__ void cc_kernel_coalesce(bool *curr_visit, bool *next_visit, uint64_t vertex_count, uint64_t *vertexList, EdgeT *edgeList, unsigned long long *comp, bool *changed, int* preempted, int total_block, int* executed, dim3 shape)
{
    while (true) {
        __shared__ bool stop;
        __shared__ unsigned int x,y,z;
        dim3 taskIdx;

        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
            stop = *preempted;
            if(stop == 0) {
                int exec_index = atomicAdd(executed, 1);
                if(exec_index >= total_block) {
                    stop = 1;
                }
                x = exec_index % shape.x;
                y = (exec_index / shape.x) % shape.y;
                z = exec_index / (shape.x * shape.y);
            }
        }
        __syncthreads();
        if (stop == 1) {
            return;
        }
        taskIdx.x = x;
        taskIdx.y = y;
        taskIdx.z = z;
        cc_kernel_coalesce__blp(taskIdx, curr_visit, next_visit, vertex_count, vertexList, edgeList, comp, changed);
    }
}
__device__ void cc_kernel_coalesce__blp(dim3 taskIdx, bool *curr_visit, bool *next_visit, uint64_t vertex_count, uint64_t *vertexList, EdgeT *edgeList, unsigned long long *comp, bool *changed){
    const uint64_t tid = blockDim.x * BLOCK_SIZE * taskIdx.y + blockDim.x * taskIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);

    if (warpIdx < vertex_count && curr_visit[warpIdx] == true) {
        const uint64_t start = vertexList[warpIdx];
        const uint64_t shift_start = start & MEM_ALIGN;
        const uint64_t end = vertexList[warpIdx+1];

        for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE) {
            if (i >= start) {
                unsigned long long comp_src = comp[warpIdx];
                const EdgeT next = edgeList[i];

                unsigned long long comp_next = comp[next];
                unsigned long long comp_target;
                EdgeT next_target;

                if (comp_next != comp_src) {
                    if (comp_src < comp_next) {
                        next_target = next;
                        comp_target = comp_src;
                    }
                    else {
                        next_target = warpIdx;
                        comp_target = comp_next;
                    }

                    atomicMin(&comp[next_target], comp_target);
                    next_visit[next_target] = true;
                    *changed = true;
                }
            }
        }
    }
}

CC::CC(const std::string &filename) {
    read_graph(filename, vertex_count, edge_count, vertexList_h, edgeList_h);
    copy_gpu_graph(vertex_count, edge_count, vertexList_h, edgeList_h, vertexList_d, edgeList_d);
    ASSERT_CUDA_ERROR(GPUMemAlloc((GPUdeviceptr*)&curr_visit_d, vertex_count * sizeof(bool)));
    ASSERT_CUDA_ERROR(GPUMemAlloc((GPUdeviceptr*)&changed_d, sizeof(bool)));
    int numblocks = ((vertex_count * WARP_SIZE + numthreads) / numthreads);
    ASSERT_CUDA_ERROR(GPUMemAlloc((GPUdeviceptr*)&comp_d, vertex_count * sizeof(unsigned long long)));
    ASSERT_CUDA_ERROR(GPUMemAlloc((GPUdeviceptr*)&next_visit_d, vertex_count * sizeof(bool)));

    comp_h = new unsigned long long[vertex_count];
    blockDim = dim3(BLOCK_SIZE, (numblocks+BLOCK_SIZE)/BLOCK_SIZE);
    int t;
    GPUOccupancyMaxPotentialBlockSizeWithFlags(&launch_block_dim, &t, cc_kernel_coalesce, 0, numthreads, 0);
    launch_block_dim = std::min(launch_block_dim, (int)(blockDim.x * blockDim.y * blockDim.z));
}

void CC::compute(GPUstream stream, int* preempted, int* executed, std::atomic<int> &lc_tasks, std::atomic<int> &be_tasks){
    for (uint64_t i = 0; i < vertex_count; i++){
        comp_h[i] = i;
    }
    ASSERT_CUDA_ERROR(GPUMemcpyHtoDAsync((GPUdeviceptr)comp_d, comp_h, vertex_count * sizeof(unsigned long long), stream));
    ASSERT_CUDA_ERROR(GPUMemsetAsync((GPUdeviceptr)curr_visit_d, 0x01, vertex_count * sizeof(bool), stream));
    ASSERT_CUDA_ERROR(GPUMemsetAsync((GPUdeviceptr)next_visit_d, 0x00, vertex_count * sizeof(bool), stream));
    int iter = 0;
    bool changed_h = false;
    int expected = blockDim.x * blockDim.y * blockDim.z;
    do {
        int executed_h = 0;
        changed_h = false;
        ASSERT_CUDA_ERROR(GPUMemcpyHtoDAsync((GPUdeviceptr)executed, &executed_h, sizeof(int), stream));
        ASSERT_CUDA_ERROR(GPUMemcpyHtoDAsync((GPUdeviceptr)changed_d, &changed_h, sizeof(bool), stream));
        resume:
        cc_kernel_coalesce<<<launch_block_dim, numthreads, 0, stream>>>(curr_visit_d, next_visit_d, vertex_count, vertexList_d, edgeList_d, comp_d, changed_d, preempted, expected, executed, blockDim);
        ASSERT_CUDA_ERROR(GPUMemcpyDtoHAsync(&executed_h, (GPUdeviceptr)executed, sizeof(int), stream));
        ASSERT_CUDA_ERROR(GPUMemcpyDtoHAsync(&changed_h, (GPUdeviceptr)changed_d, sizeof(bool), stream));
        ASSERT_CUDA_ERROR(GPUStreamSynchronize(stream));
        if(executed_h < expected){
            be_tasks.fetch_sub(1);
            while(lc_tasks.load() != 0);
            be_tasks.fetch_add(1);
            goto resume;
        }
        ASSERT_CUDA_ERROR(GPUMemsetAsync((GPUdeviceptr)curr_visit_d, 0x00, vertex_count * sizeof(bool), stream));
        bool *temp = curr_visit_d;
        curr_visit_d = next_visit_d;
        next_visit_d = temp;
        iter++;
    } while(changed_h);
    // printf("CC iter: %d\n", iter);
}

CC::~CC() {
    ASSERT_CUDA_ERROR(GPUMemFree((GPUdeviceptr)changed_d));
    ASSERT_CUDA_ERROR(GPUMemFree((GPUdeviceptr)curr_visit_d));
    ASSERT_CUDA_ERROR(GPUMemFree((GPUdeviceptr)next_visit_d));
    ASSERT_CUDA_ERROR(GPUMemFree((GPUdeviceptr)comp_d));
    delete[] comp_h;
}

}

namespace bfs {
__device__ void bfs_kernel_coalesce__blp(dim3 taskIdx, uint32_t *label, const uint32_t level, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, bool *changed);

__global__ void bfs_kernel_coalesce(uint32_t *label, const uint32_t level, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, bool *changed, int* preempted, int total_block, int* executed, dim3 shape)
{
    while (true) {
        __shared__ bool stop;
        __shared__ unsigned int x,y,z;
        dim3 taskIdx;

        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
            stop = *preempted;
            if(stop == 0) {
                int exec_index = atomicAdd(executed, 1);
                if(exec_index >= total_block) {
                    stop = 1;
                }
                x = exec_index % shape.x;
                y = (exec_index / shape.x) % shape.y;
                z = exec_index / (shape.x * shape.y);
            }
        }
        __syncthreads();
        if (stop == 1) {
            return;
        }
        taskIdx.x = x;
        taskIdx.y = y;
        taskIdx.z = z;
        bfs_kernel_coalesce__blp(taskIdx, label, level, vertex_count, vertexList, edgeList, changed);
    }
}
__device__ void bfs_kernel_coalesce__blp(dim3 taskIdx, uint32_t *label, const uint32_t level, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, bool *changed){
    const uint64_t tid = blockDim.x * BLOCK_SIZE * taskIdx.y + blockDim.x * taskIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);

    if(warpIdx < vertex_count && label[warpIdx] == level) {
        const uint64_t start = vertexList[warpIdx];
        const uint64_t shift_start = start & MEM_ALIGN;
        const uint64_t end = vertexList[warpIdx+1];

        for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE) {
            if (i >= start) {
                const EdgeT next = edgeList[i];

                if(label[next] == MYINFINITY) {
                    label[next] = level + 1;
                    *changed = true;
                }
            }
        }
    }
}

BFS::BFS(const std::string &filename) {
    read_graph(filename, vertex_count, edge_count, vertexList_h, edgeList_h);
    copy_gpu_graph(vertex_count, edge_count, vertexList_h, edgeList_h, vertexList_d, edgeList_d);
    ASSERT_CUDA_ERROR(GPUMemAlloc((GPUdeviceptr*)&label_d, vertex_count * sizeof(uint32_t)));
    ASSERT_CUDA_ERROR(GPUMemAlloc((GPUdeviceptr*)&changed_d, sizeof(bool)));
    int numblocks_update = ((vertex_count + numthreads) / numthreads);
    int numblocks = ((vertex_count * WARP_SIZE + numthreads) / numthreads);
    
    blockDim = dim3(BLOCK_SIZE, (numblocks+BLOCK_SIZE)/BLOCK_SIZE);
    int t;
    GPUOccupancyMaxPotentialBlockSizeWithFlags(&launch_block_dim, &t, bfs_kernel_coalesce, 0, numthreads, 0);
    launch_block_dim = std::min(launch_block_dim, (int)(blockDim.x * blockDim.y * blockDim.z));
    src = 0;
}

void BFS::compute(GPUstream stream, int* preempted, int* executed, std::atomic<int> &lc_tasks, std::atomic<int> &be_tasks){
    int level = 0;
    int iter = 0;
    int zero = 0;
    int expected = blockDim.x * blockDim.y * blockDim.z;
    ASSERT_CUDA_ERROR(GPUMemsetAsync((GPUdeviceptr)label_d, 0xFF, vertex_count * sizeof(uint32_t), stream));
    ASSERT_CUDA_ERROR(GPUMemcpyHtoDAsync((GPUdeviceptr)&label_d[src], &zero, sizeof(uint32_t), stream));
    do {
        int h_executed = 0;
        changed_h = false;
        ASSERT_CUDA_ERROR(GPUMemcpyHtoDAsync((GPUdeviceptr)executed, &h_executed, sizeof(int), stream));
        ASSERT_CUDA_ERROR(GPUMemcpyHtoDAsync((GPUdeviceptr)changed_d, &changed_h, sizeof(bool), stream));
        resume:
        bfs_kernel_coalesce<<<launch_block_dim, numthreads, 0, stream>>>(label_d, level, vertex_count, vertexList_d, edgeList_d, changed_d, preempted, expected, executed, blockDim);

        ASSERT_CUDA_ERROR(GPUMemcpyDtoHAsync(&changed_h, (GPUdeviceptr)changed_d, sizeof(bool), stream));
        ASSERT_CUDA_ERROR(GPUMemcpyDtoHAsync(&h_executed, (GPUdeviceptr)executed, sizeof(int), stream));
        ASSERT_CUDA_ERROR(GPUStreamSynchronize(stream));
        if(h_executed < expected){
            be_tasks.fetch_sub(1);
            while(lc_tasks.load() != 0);
            be_tasks.fetch_add(1);
            goto resume;
        }
        ASSERT_CUDA_ERROR(GPUMemsetAsync((GPUdeviceptr)executed, 0, sizeof(int), stream));
        iter++;
        level++;
    } while(changed_h);
    // printf("BFS iter: %d\n", iter);
}

BFS::~BFS() {
    ASSERT_CUDA_ERROR(GPUMemFree((GPUdeviceptr)label_d));
    ASSERT_CUDA_ERROR(GPUMemFree((GPUdeviceptr)changed_d));
}

}

namespace pagerank {
__device__ void pr_initialize__blp(dim3 taskIdx, bool *label, ValueT *delta, ValueT *residual, ValueT *value, const uint64_t vertex_count, const uint64_t *vertexList, ValueT alpha);

__global__ void pr_initialize(bool *label, ValueT *delta, ValueT *residual, ValueT *value, const uint64_t vertex_count, const uint64_t *vertexList, ValueT alpha, int* preempted, int total_block, int* executed, dim3 shape)
{
    while (true) {
        __shared__ bool stop;
        __shared__ unsigned int x,y,z;
        dim3 taskIdx;

        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
            stop = *preempted;
            if(stop == 0) {
                int exec_index = atomicAdd(executed, 1);
                if(exec_index >= total_block) {
                    stop = 1;
                }
                x = exec_index % shape.x;
                y = (exec_index / shape.x) % shape.y;
                z = exec_index / (shape.x * shape.y);
            }
        }
        __syncthreads();
        if (stop == 1) {
            return;
        }
        taskIdx.x = x;
        taskIdx.y = y;
        taskIdx.z = z;
        pr_initialize__blp(taskIdx, label, delta, residual, value, vertex_count, vertexList, alpha);
    }
}
__device__ void pr_initialize__blp(dim3 taskIdx, bool *label, ValueT *delta, ValueT *residual, ValueT *value, const uint64_t vertex_count, const uint64_t *vertexList, ValueT alpha){
    const uint64_t tid = blockDim.x * BLOCK_SIZE * taskIdx.y + blockDim.x * taskIdx.x + threadIdx.x;
    if (tid < vertex_count) {
        value[tid] = 1.0f - alpha;
        delta[tid] = (1.0f - alpha) * alpha / (vertexList[tid+1] - vertexList[tid]);
        residual[tid] = 0.0f;
        label[tid] = true;
	}
}
__device__ void pr_kernel_coalesce__blp(dim3 taskIdx, bool* label, ValueT *delta, ValueT *residual, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList);

__global__ void pr_kernel_coalesce(bool* label, ValueT *delta, ValueT *residual, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, int* preempted, int total_block, int* executed, dim3 shape)
{
    while (true) {
        __shared__ bool stop;
        __shared__ unsigned int x,y,z;
        dim3 taskIdx;

        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
            stop = *preempted;
            if(stop == 0) {
                int exec_index = atomicAdd(executed, 1);
                if(exec_index >= total_block) {
                    stop = 1;
                }
                x = exec_index % shape.x;
                y = (exec_index / shape.x) % shape.y;
                z = exec_index / (shape.x * shape.y);
            }
        }
        __syncthreads();
        if (stop == 1) {
            return;
        }
        taskIdx.x = x;
        taskIdx.y = y;
        taskIdx.z = z;
        pr_kernel_coalesce__blp(taskIdx, label, delta, residual, vertex_count, vertexList, edgeList);
    }
}
__device__ void pr_kernel_coalesce__blp(dim3 taskIdx, bool* label, ValueT *delta, ValueT *residual, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList){
    const uint64_t tid = blockDim.x * BLOCK_SIZE * taskIdx.y + blockDim.x * taskIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);

    if(warpIdx < vertex_count && label[warpIdx]) {
        const uint64_t start = vertexList[warpIdx];
        const uint64_t shift_start = start & MEM_ALIGN;
        const uint64_t end = vertexList[warpIdx+1];

        for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE)
            if (i >= start)
                atomicAdd(&residual[edgeList[i]], delta[warpIdx]);

        label[warpIdx] = false;
    }
}
__device__ void pr_update__blp(dim3 taskIdx, bool *label, ValueT *delta, ValueT *residual, ValueT *value, const uint64_t vertex_count, const uint64_t *vertexList, ValueT tolerance, ValueT alpha, bool *changed);

__global__ void pr_update(bool *label, ValueT *delta, ValueT *residual, ValueT *value, const uint64_t vertex_count, const uint64_t *vertexList, ValueT tolerance, ValueT alpha, bool *changed, int* preempted, int total_block, int* executed, dim3 shape)
{
    while (true) {
        __shared__ bool stop;
        __shared__ unsigned int x,y,z;
        dim3 taskIdx;

        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
            stop = *preempted;
            if(stop == 0) {
                int exec_index = atomicAdd(executed, 1);
                if(exec_index >= total_block) {
                    stop = 1;
                }
                x = exec_index % shape.x;
                y = (exec_index / shape.x) % shape.y;
                z = exec_index / (shape.x * shape.y);
            }
        }
        __syncthreads();
        if (stop == 1) {
            return;
        }
        taskIdx.x = x;
        taskIdx.y = y;
        taskIdx.z = z;
        pr_update__blp(taskIdx, label, delta, residual, value, vertex_count, vertexList, tolerance, alpha, changed);
    }
}
__device__ void pr_update__blp(dim3 taskIdx, bool *label, ValueT *delta, ValueT *residual, ValueT *value, const uint64_t vertex_count, const uint64_t *vertexList, ValueT tolerance, ValueT alpha, bool *changed){
    const uint64_t tid = blockDim.x * BLOCK_SIZE * taskIdx.y + blockDim.x * taskIdx.x + threadIdx.x;
    if (tid < vertex_count && residual[tid] > tolerance) {
        value[tid] += residual[tid];
        delta[tid] = residual[tid] * alpha / (vertexList[tid+1] - vertexList[tid]);
        residual[tid] = 0.0f;
        label[tid] = true;
        *changed = true;
	}
}

PageRank::PageRank(const std::string &filename) {
    read_graph(filename, vertex_count, edge_count, vertexList_h, edgeList_h);
    copy_gpu_graph(vertex_count, edge_count, vertexList_h, edgeList_h, vertexList_d, edgeList_d);
    ASSERT_CUDA_ERROR(GPUMemAlloc((GPUdeviceptr*)&label_d, vertex_count * sizeof(bool)));
    ASSERT_CUDA_ERROR(GPUMemAlloc((GPUdeviceptr*)&changed_d, sizeof(bool)));
    ASSERT_CUDA_ERROR(GPUMemAlloc((GPUdeviceptr*)&delta_d, vertex_count * sizeof(ValueT)));
    ASSERT_CUDA_ERROR(GPUMemAlloc((GPUdeviceptr*)&residual_d, vertex_count * sizeof(ValueT)));
    ASSERT_CUDA_ERROR(GPUMemAlloc((GPUdeviceptr*)&value_d, vertex_count * sizeof(ValueT)));
    int numblocks_update = ((vertex_count + numthreads) / numthreads);
    int numblocks = ((vertex_count * WARP_SIZE + numthreads) / numthreads);

    blockDim = dim3(BLOCK_SIZE, (numblocks+BLOCK_SIZE)/BLOCK_SIZE);
    blockDim_update = dim3(BLOCK_SIZE, (numblocks_update+BLOCK_SIZE)/BLOCK_SIZE);

    int t;
    GPUOccupancyMaxPotentialBlockSizeWithFlags(&launch_block_dim1, &t, pr_initialize, 0, numthreads, 0);
    GPUOccupancyMaxPotentialBlockSizeWithFlags(&launch_block_dim2, &t, pr_kernel_coalesce, 0, numthreads, 0);
    GPUOccupancyMaxPotentialBlockSizeWithFlags(&launch_block_dim3, &t, pr_update, 0, numthreads, 0);
    launch_block_dim1 = std::min(launch_block_dim1, (int)(blockDim_update.x * blockDim_update.y * blockDim_update.z));
    launch_block_dim2 = std::min(launch_block_dim2, (int)(blockDim.x * blockDim.y * blockDim.z));
    launch_block_dim3 = std::min(launch_block_dim3, (int)(blockDim_update.x * blockDim_update.y * blockDim_update.z));

}

void PageRank::compute(GPUstream stream, int* preempted, int* executed, std::atomic<int> & lc_tasks, std::atomic<int> &be_tasks){
    int iter = 0;
    int executed_h[3] = {0};
    int expected1 = blockDim.x * blockDim.y * blockDim.z;
    int expected2 = blockDim_update.x * blockDim_update.y * blockDim_update.z;
    GPUMemsetAsync((GPUdeviceptr)executed, 0, 3 * sizeof(int), stream);
    while(1){
        pr_initialize<<<launch_block_dim1, numthreads, 0, stream>>>(label_d, delta_d, residual_d, value_d, vertex_count, vertexList_d, alpha, preempted, expected2, executed, blockDim_update);
        ASSERT_CUDA_ERROR(GPUMemcpyDtoHAsync(executed_h, (GPUdeviceptr)executed, sizeof(int) * 3, stream));
        GPUStreamSynchronize(stream);
        if(executed_h[0] < expected2){
            be_tasks.fetch_sub(1);
            while(lc_tasks.load() != 0);
            be_tasks.fetch_add(1);
        } else {
            break;
        }
    }
    do {
        executed_h[1] = 0;
        executed_h[2] = 0;
        changed_h = false;
        ASSERT_CUDA_ERROR(GPUMemcpyHtoDAsync((GPUdeviceptr)changed_d, &changed_h, sizeof(bool), stream));
        ASSERT_CUDA_ERROR(GPUMemcpyHtoDAsync((GPUdeviceptr)executed, &executed_h, sizeof(int) * 3, stream));
        resume1:
        pr_kernel_coalesce<<<launch_block_dim2, numthreads, 0, stream>>>(label_d, delta_d, residual_d, vertex_count, vertexList_d, edgeList_d, preempted, expected1, executed + 1, blockDim);
        resume2:
        pr_update<<<launch_block_dim3, numthreads, 0, stream>>>(label_d, delta_d, residual_d, value_d, vertex_count, vertexList_d, tolerance, alpha, changed_d, preempted, expected2, executed + 2, blockDim_update);
        ASSERT_CUDA_ERROR(GPUMemcpyDtoHAsync(executed_h, (GPUdeviceptr)executed, sizeof(int) * 3, stream));
        ASSERT_CUDA_ERROR(GPUMemcpyDtoHAsync(&changed_h, (GPUdeviceptr)changed_d, sizeof(bool), stream));
        GPUStreamSynchronize(stream);
        if(executed_h[1] < expected1){
            be_tasks.fetch_sub(1);
            while(lc_tasks.load() != 0);
            be_tasks.fetch_add(1);
            goto resume1;
        }
        if(executed_h[2] < expected2){
            be_tasks.fetch_sub(1);
            while(lc_tasks.load() != 0);
            be_tasks.fetch_add(1);
            goto resume2;
        }
        iter++;
    } while(changed_h && iter < max_iter);
    // printf("PageRank iter: %d\n", iter);
}

PageRank::~PageRank() {
    ASSERT_CUDA_ERROR(GPUMemFree((GPUdeviceptr)label_d));
    ASSERT_CUDA_ERROR(GPUMemFree((GPUdeviceptr)delta_d));
    ASSERT_CUDA_ERROR(GPUMemFree((GPUdeviceptr)residual_d));
    ASSERT_CUDA_ERROR(GPUMemFree((GPUdeviceptr)value_d));
    ASSERT_CUDA_ERROR(GPUMemFree((GPUdeviceptr)changed_d));
}

}

namespace sssp {
__device__ void sssp_kernel_coalesce__blp(dim3 taskIdx, bool *label, const WeightT *costList, WeightT *newCostList, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, const WeightT *weightList);

__global__ void sssp_kernel_coalesce(bool *label, const WeightT *costList, WeightT *newCostList, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, const WeightT *weightList, int* preempted, int total_block, int* executed, dim3 shape)
{
    while (true) {
        __shared__ bool stop;
        __shared__ unsigned int x,y,z;
        dim3 taskIdx;

        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
            stop = *preempted;
            if(stop == 0) {
                int exec_index = atomicAdd(executed, 1);
                if(exec_index >= total_block) {
                    stop = 1;
                }
                x = exec_index % shape.x;
                y = (exec_index / shape.x) % shape.y;
                z = exec_index / (shape.x * shape.y);
            }
        }
        __syncthreads();
        if (stop == 1) {
            return;
        }
        taskIdx.x = x;
        taskIdx.y = y;
        taskIdx.z = z;
        sssp_kernel_coalesce__blp(taskIdx, label, costList, newCostList, vertex_count, vertexList, edgeList, weightList);
    }
}
__device__ void sssp_kernel_coalesce__blp(dim3 taskIdx, bool *label, const WeightT *costList, WeightT *newCostList, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, const WeightT *weightList){
    const uint64_t tid = blockDim.x * BLOCK_SIZE * taskIdx.y + blockDim.x * taskIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);

    if (warpIdx < vertex_count && label[warpIdx]) {
        uint64_t start = vertexList[warpIdx];
        const uint64_t shift_start = start & MEM_ALIGN;
        uint64_t end = vertexList[warpIdx+1];

        WeightT cost = newCostList[warpIdx];

        for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE) {
            if (newCostList[warpIdx] != cost)
                break;
            if (newCostList[edgeList[i]] > cost + weightList[i] && i >= start)
                atomicMin(&(newCostList[edgeList[i]]), cost + weightList[i]);
        }

        label[warpIdx] = false;
    }
}
__device__ void sssp_update__blp(dim3 taskIdx, bool *label, WeightT *costList, WeightT *newCostList, const uint32_t vertex_count, bool *changed);

__global__ void sssp_update(bool *label, WeightT *costList, WeightT *newCostList, const uint32_t vertex_count, bool *changed, int* preempted, int total_block, int* executed, dim3 shape)
{
    while (true) {
        __shared__ bool stop;
        __shared__ unsigned int x,y,z;
        dim3 taskIdx;

        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
            stop = *preempted;
            if(stop == 0) {
                int exec_index = atomicAdd(executed, 1);
                if(exec_index >= total_block) {
                    stop = 1;
                }
                x = exec_index % shape.x;
                y = (exec_index / shape.x) % shape.y;
                z = exec_index / (shape.x * shape.y);
            }
        }
        __syncthreads();
        if (stop == 1) {
            return;
        }
        taskIdx.x = x;
        taskIdx.y = y;
        taskIdx.z = z;
        sssp_update__blp(taskIdx, label, costList, newCostList, vertex_count, changed);
    }
}
__device__ void sssp_update__blp(dim3 taskIdx, bool *label, WeightT *costList, WeightT *newCostList, const uint32_t vertex_count, bool *changed){
	uint64_t tid = blockDim.x * BLOCK_SIZE * taskIdx.y + blockDim.x * taskIdx.x + threadIdx.x;

    if (tid < vertex_count) {
        if (newCostList[tid] < costList[tid]) {
            costList[tid] = newCostList[tid];
            label[tid] = true;
            *changed = true;
        }
    }
}

SSSP::SSSP(const std::string &filename) {
    read_graph(filename, vertex_count, edge_count, vertexList_h, edgeList_h);
    copy_gpu_graph(vertex_count, edge_count, vertexList_h, edgeList_h, vertexList_d, edgeList_d);
    ASSERT_CUDA_ERROR(GPUMemAlloc((GPUdeviceptr*)&label_d, vertex_count * sizeof(bool)));
    ASSERT_CUDA_ERROR(GPUMemAlloc((GPUdeviceptr*)&changed_d, sizeof(bool)));
    ASSERT_CUDA_ERROR(GPUMemAlloc((GPUdeviceptr*)&costList_d, vertex_count * sizeof(WeightT)));
    ASSERT_CUDA_ERROR(GPUMemAlloc((GPUdeviceptr*)&newCostList_d, vertex_count * sizeof(WeightT)));
    int numblocks_update = ((vertex_count + numthreads) / numthreads);
    int numblocks = ((vertex_count * WARP_SIZE + numthreads) / numthreads);

    blockDim = dim3(BLOCK_SIZE, (numblocks+BLOCK_SIZE)/BLOCK_SIZE);
    blockDim_update = dim3(BLOCK_SIZE, (numblocks_update+BLOCK_SIZE)/BLOCK_SIZE);
    src = 0;
    uint64_t weight_count, dummy;
    FILE *fp = fopen((filename + ".val").c_str(), "rb");
    fread(&weight_count, sizeof(uint64_t), 1, fp);
    fread(&dummy, sizeof(uint64_t), 1, fp);
    uint64_t weight_size = weight_count * sizeof(WeightT);
    weightList_h = (WeightT*)malloc(weight_size);
    fread((char*)weightList_h, sizeof(WeightT), weight_count, fp);
    ASSERT_CUDA_ERROR(GPUMemAlloc((GPUdeviceptr*)&weightList_d, weight_size));
    ASSERT_CUDA_ERROR(GPUMemcpyHtoD((GPUdeviceptr)weightList_d, weightList_h, weight_size));
    fclose(fp);
    int t;
    GPUOccupancyMaxPotentialBlockSizeWithFlags(&launch_block_dim1, &t, sssp_kernel_coalesce, 0, numthreads, 0);
    GPUOccupancyMaxPotentialBlockSizeWithFlags(&launch_block_dim2, &t, sssp_update, 0, numthreads, 0);
    launch_block_dim1 = std::min(launch_block_dim1, (int)(blockDim.x * blockDim.y * blockDim.z));
    launch_block_dim2 = std::min(launch_block_dim2, (int)(blockDim_update.x * blockDim_update.y * blockDim_update.z));
}

void SSSP::compute(GPUstream stream, int* preempted, int* executed, std::atomic<int> &lc_tasks, std::atomic<int> &be_tasks){
    int iter = 0;
    int zero = 0;
    int one = 1;
    ASSERT_CUDA_ERROR(GPUMemsetAsync((GPUdeviceptr)costList_d, 0xFF, vertex_count * sizeof(WeightT), stream));
    ASSERT_CUDA_ERROR(GPUMemsetAsync((GPUdeviceptr)newCostList_d, 0xFF, vertex_count * sizeof(WeightT), stream));
    ASSERT_CUDA_ERROR(GPUMemsetAsync((GPUdeviceptr)label_d, 0x0, vertex_count * sizeof(bool), stream));
    ASSERT_CUDA_ERROR(GPUMemcpyHtoDAsync((GPUdeviceptr)&label_d[src], &one, sizeof(bool), stream));
    ASSERT_CUDA_ERROR(GPUMemcpyHtoDAsync((GPUdeviceptr)&costList_d[src], &zero, sizeof(WeightT), stream));
    ASSERT_CUDA_ERROR(GPUMemcpyHtoDAsync((GPUdeviceptr)&newCostList_d[src], &zero, sizeof(WeightT), stream));
    int expected1 = blockDim.x * blockDim.y * blockDim.z;
    int expected2 = blockDim_update.x * blockDim_update.y * blockDim_update.z;
    do {
        changed_h = false;
        int executed_h[2] = {0, 0};
        ASSERT_CUDA_ERROR(GPUMemcpyHtoDAsync((GPUdeviceptr)executed, executed_h, sizeof(int) * 2, stream));
        ASSERT_CUDA_ERROR(GPUMemcpyHtoDAsync((GPUdeviceptr)changed_d, &changed_h, sizeof(bool), stream));
        resume1:
        sssp_kernel_coalesce<<<launch_block_dim1, numthreads, 0, stream>>>(label_d, costList_d, newCostList_d, vertex_count, vertexList_d, edgeList_d, weightList_d, preempted, expected1, executed, blockDim);
        resume2:
        sssp_update<<<launch_block_dim2, numthreads, 0, stream>>>(label_d, costList_d, newCostList_d, vertex_count, changed_d, preempted, expected2, executed + 1, blockDim_update);
        ASSERT_CUDA_ERROR(GPUMemcpyDtoHAsync(executed_h, (GPUdeviceptr)executed, sizeof(int) * 2, stream));

        ASSERT_CUDA_ERROR(GPUMemcpyDtoHAsync(&changed_h, (GPUdeviceptr)changed_d, sizeof(bool), stream));
        ASSERT_CUDA_ERROR(GPUStreamSynchronize(stream));
        if(executed_h[0] < expected1){
            be_tasks.fetch_sub(1);
            while(lc_tasks.load() != 0);
            be_tasks.fetch_add(1);
            goto resume1;
        }
        if(executed_h[1] < expected2){
            be_tasks.fetch_sub(1);
            while(lc_tasks.load() != 0);
            be_tasks.fetch_add(1);
            goto resume2;
        }
        
        iter++;
    } while(changed_h);
    // printf("SSSP iter: %d\n", iter);

}

SSSP::~SSSP() {
    ASSERT_CUDA_ERROR(GPUMemFree((GPUdeviceptr)label_d));
    ASSERT_CUDA_ERROR(GPUMemFree((GPUdeviceptr)costList_d));
    ASSERT_CUDA_ERROR(GPUMemFree((GPUdeviceptr)newCostList_d));
    ASSERT_CUDA_ERROR(GPUMemFree((GPUdeviceptr)changed_d));
    ASSERT_CUDA_ERROR(GPUMemFree((GPUdeviceptr)weightList_d));
    free(weightList_h);
}

}

} // namespace GraphComputeBlp