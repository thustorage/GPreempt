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

namespace GraphComputeRaw {

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

__global__ void cc_kernel_coalesce(bool *curr_visit, bool *next_visit, uint64_t vertex_count, uint64_t *vertexList, EdgeT *edgeList, unsigned long long *comp, bool *changed) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
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
}

void CC::compute(GPUstream stream){
    for (uint64_t i = 0; i < vertex_count; i++){
        comp_h[i] = i;
    }
    ASSERT_CUDA_ERROR(GPUMemcpyHtoDAsync((GPUdeviceptr)comp_d, comp_h, vertex_count * sizeof(unsigned long long), stream));
    ASSERT_CUDA_ERROR(GPUMemsetAsync((GPUdeviceptr)curr_visit_d, 0x01, vertex_count * sizeof(bool), stream));
    ASSERT_CUDA_ERROR(GPUMemsetAsync((GPUdeviceptr)next_visit_d, 0x00, vertex_count * sizeof(bool), stream));
    int iter = 0;
    bool changed_h = false;
    do {
        changed_h = false;
        ASSERT_CUDA_ERROR(GPUMemcpyHtoDAsync((GPUdeviceptr)changed_d, &changed_h, sizeof(bool), stream));

        cc_kernel_coalesce<<<blockDim, numthreads, 0, stream>>>(curr_visit_d, next_visit_d, vertex_count, vertexList_d, edgeList_d, comp_d, changed_d);

        ASSERT_CUDA_ERROR(GPUMemsetAsync((GPUdeviceptr)curr_visit_d, 0x00, vertex_count * sizeof(bool), stream));

        bool *temp = curr_visit_d;
        curr_visit_d = next_visit_d;
        next_visit_d = temp;

        iter++;

        ASSERT_CUDA_ERROR(GPUMemcpyDtoHAsync(&changed_h, (GPUdeviceptr)changed_d, sizeof(bool), stream));
        ASSERT_CUDA_ERROR(GPUStreamSynchronize(stream));
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

__global__ void bfs_kernel_coalesce(uint32_t *label, const uint32_t level, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, bool *changed) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
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
    src = 0;
}

void BFS::compute(GPUstream stream){
    int level = 0;
    int iter = 0;
    int zero = 0;
    ASSERT_CUDA_ERROR(GPUMemsetAsync((GPUdeviceptr)label_d, 0xFF, vertex_count * sizeof(uint32_t), stream));
    ASSERT_CUDA_ERROR(GPUMemcpyHtoDAsync((GPUdeviceptr)&label_d[src], &zero, sizeof(uint32_t), stream));
    do {
        changed_h = false;
        ASSERT_CUDA_ERROR(GPUMemcpyHtoDAsync((GPUdeviceptr)changed_d, &changed_h, sizeof(bool), stream));

        bfs_kernel_coalesce<<<blockDim, numthreads, 0, stream>>>(label_d, level, vertex_count, vertexList_d, edgeList_d, changed_d);

        iter++;
        level++;
        ASSERT_CUDA_ERROR(GPUMemcpyDtoHAsync(&changed_h, (GPUdeviceptr)changed_d, sizeof(bool), stream));
        ASSERT_CUDA_ERROR(GPUStreamSynchronize(stream));
    } while(changed_h);
    // printf("BFS iter: %d\n", iter);
}

BFS::~BFS() {
    ASSERT_CUDA_ERROR(GPUMemFree((GPUdeviceptr)label_d));
    ASSERT_CUDA_ERROR(GPUMemFree((GPUdeviceptr)changed_d));
}

}

namespace pagerank {

__global__ void pr_initialize(bool *label, ValueT *delta, ValueT *residual, ValueT *value, const uint64_t vertex_count, const uint64_t *vertexList, ValueT alpha) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < vertex_count) {
        value[tid] = 1.0f - alpha;
        delta[tid] = (1.0f - alpha) * alpha / (vertexList[tid+1] - vertexList[tid]);
        residual[tid] = 0.0f;
        label[tid] = true;
	}
}

__global__ void pr_kernel_coalesce(bool* label, ValueT *delta, ValueT *residual, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
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

__global__ void pr_update(bool *label, ValueT *delta, ValueT *residual, ValueT *value, const uint64_t vertex_count, const uint64_t *vertexList, ValueT tolerance, ValueT alpha, bool *changed) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
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
}

void PageRank::compute(GPUstream stream){
    int iter = 0;
    pr_initialize<<<blockDim_update, numthreads, 0, stream>>>(label_d, delta_d, residual_d, value_d, vertex_count, vertexList_d, alpha);
    do {
        changed_h = false;
        ASSERT_CUDA_ERROR(GPUMemcpyHtoDAsync((GPUdeviceptr)changed_d, &changed_h, sizeof(bool), stream));
        
        pr_kernel_coalesce<<<blockDim, numthreads, 0, stream>>>(label_d, delta_d, residual_d, vertex_count, vertexList_d, edgeList_d);

        pr_update<<<blockDim_update, numthreads, 0, stream>>>(label_d, delta_d, residual_d, value_d, vertex_count, vertexList_d, tolerance, alpha, changed_d);

        ASSERT_CUDA_ERROR(GPUMemcpyDtoHAsync(&changed_h, (GPUdeviceptr)changed_d, sizeof(bool), stream));
        GPUStreamSynchronize(stream);
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

__global__ void sssp_kernel_coalesce(bool *label, const WeightT *costList, WeightT *newCostList, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, const WeightT *weightList) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
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

__global__ void sssp_update(bool *label, WeightT *costList, WeightT *newCostList, const uint32_t vertex_count, bool *changed) {
	uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;

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
}

void SSSP::compute(GPUstream stream){
    int iter = 0;
    int zero = 0;
    int one = 1;
    ASSERT_CUDA_ERROR(GPUMemsetAsync((GPUdeviceptr)costList_d, 0xFF, vertex_count * sizeof(WeightT), stream));
    ASSERT_CUDA_ERROR(GPUMemsetAsync((GPUdeviceptr)newCostList_d, 0xFF, vertex_count * sizeof(WeightT), stream));
    ASSERT_CUDA_ERROR(GPUMemsetAsync((GPUdeviceptr)label_d, 0x0, vertex_count * sizeof(bool), stream));
    ASSERT_CUDA_ERROR(GPUMemcpyHtoDAsync((GPUdeviceptr)&label_d[src], &one, sizeof(bool), stream));
    ASSERT_CUDA_ERROR(GPUMemcpyHtoDAsync((GPUdeviceptr)&costList_d[src], &zero, sizeof(WeightT), stream));
    ASSERT_CUDA_ERROR(GPUMemcpyHtoDAsync((GPUdeviceptr)&newCostList_d[src], &zero, sizeof(WeightT), stream));

    do {
        changed_h = false;
        ASSERT_CUDA_ERROR(GPUMemcpyHtoDAsync((GPUdeviceptr)changed_d, &changed_h, sizeof(bool), stream));

        sssp_kernel_coalesce<<<blockDim, numthreads, 0, stream>>>(label_d, costList_d, newCostList_d, vertex_count, vertexList_d, edgeList_d, weightList_d);

        sssp_update<<<blockDim_update, numthreads, 0, stream>>>(label_d, costList_d, newCostList_d, vertex_count, changed_d);

        iter++;

        ASSERT_CUDA_ERROR(GPUMemcpyDtoHAsync(&changed_h, (GPUdeviceptr)changed_d, sizeof(bool), stream));
        ASSERT_CUDA_ERROR(GPUStreamSynchronize(stream));
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

} // namespace GraphComputeRaw