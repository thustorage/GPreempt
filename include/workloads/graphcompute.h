#ifndef GRAPHCOMPUTE_H
#define GRAPHCOMPUTE_H

#include <string>
#include "util/gpu_util.h"
#include <atomic>
namespace GraphComputeRaw {

#define BLOCK_SIZE 1024
#define WARP_SHIFT 5
#define WARP_SIZE 32
#define MEM_ALIGN_64 (~(0xfULL))
#define MEM_ALIGN MEM_ALIGN_64
#define MYINFINITY 0xFFFFFFFF

typedef uint64_t EdgeT;

void read_graph(const std::string &filename, uint64_t &vertex_cnt, uint64_t &edge_cnt, uint64_t *&vertexList, EdgeT *&edgeList);
void copy_gpu_graph(uint64_t vertex_cnt, uint64_t edge_cnt, uint64_t *vertexList_h, EdgeT *edgeList_h, uint64_t *&vertexList_d, EdgeT *&edgeList_d);

class GraphCompute {
public:
    virtual void compute(GPUstream stream) = 0;
    virtual ~GraphCompute() {}
protected:
    uint64_t vertex_count;
    uint64_t edge_count;
    uint64_t *vertexList_d;
    uint64_t *vertexList_h;
    EdgeT *edgeList_d;
    EdgeT *edgeList_h;
};

namespace cc {

class CC : public GraphCompute {
private:
    dim3 blockDim;
    int numthreads = BLOCK_SIZE;
    bool *changed_d;
    bool *curr_visit_d;
    bool *next_visit_d;
    unsigned long long *comp_d;
    unsigned long long *comp_h;
public:
    CC(const std::string &filename);
    void compute(GPUstream stream) override;
    virtual ~CC();
};

}

namespace bfs {

class BFS : public GraphCompute {
private:
    uint32_t *label_d;
    bool *changed_d;
    bool changed_h;
    uint64_t iter;
    const uint64_t max_iter = 100;
    dim3 blockDim;
    int numthreads = BLOCK_SIZE;
    int src;
public:
    BFS(const std::string &filename);
    void compute(GPUstream stream) override;
    virtual ~BFS();
};

}

namespace pagerank {
typedef float ValueT;

class PageRank : public GraphCompute {
private:
    bool *label_d;
    ValueT *delta_d;
    ValueT *residual_d;
    ValueT *value_d;
    bool *changed_d;
    bool changed_h;
    const uint64_t max_iter = 100;
    const ValueT tolerance = 0.001;
    const ValueT alpha = 0.85;
    dim3 blockDim;
    dim3 blockDim_update;
    int numthreads = BLOCK_SIZE;
public:
    PageRank(const std::string &filename);
    void compute(GPUstream stream) override;
    virtual ~PageRank();
};

}

namespace sssp {
typedef uint32_t WeightT;

class SSSP : public GraphCompute {
private:
    bool *label_d;
    WeightT *costList_d;
    WeightT *newCostList_d;
    bool *changed_d;
    bool changed_h;
    const uint64_t max_iter = 100;
    dim3 blockDim;
    dim3 blockDim_update;
    int numthreads = BLOCK_SIZE;
    int src;
    WeightT *weightList_h;
    WeightT *weightList_d;
public:
    SSSP(const std::string &filename);
    void compute(GPUstream stream) override;
    virtual ~SSSP();
};

}

} // namespace GraphComputeRaw


namespace GraphComputeBlp {

#define BLOCK_SIZE 1024
#define WARP_SHIFT 5
#define WARP_SIZE 32
#define MEM_ALIGN_64 (~(0xfULL))
#define MEM_ALIGN MEM_ALIGN_64
#define MYINFINITY 0xFFFFFFFF

typedef uint64_t EdgeT;

void read_graph(const std::string &filename, uint64_t &vertex_cnt, uint64_t &edge_cnt, uint64_t *&vertexList, EdgeT *&edgeList);
void copy_gpu_graph(uint64_t vertex_cnt, uint64_t edge_cnt, uint64_t *vertexList_h, EdgeT *edgeList_h, uint64_t *&vertexList_d, EdgeT *&edgeList_d);

class GraphCompute {
public:
    virtual void compute(GPUstream stream, int* preempted, int* executed, std::atomic<int> &lc_tasks, std::atomic<int> &be_tasks) = 0;
    virtual ~GraphCompute() {}
protected:
    uint64_t vertex_count;
    uint64_t edge_count;
    uint64_t *vertexList_d;
    uint64_t *vertexList_h;
    EdgeT *edgeList_d;
    EdgeT *edgeList_h;
};

namespace cc {

class CC : public GraphCompute {
private:
    dim3 blockDim;
    int numthreads = BLOCK_SIZE;
    bool *changed_d;
    bool *curr_visit_d;
    bool *next_visit_d;
    unsigned long long *comp_d;
    unsigned long long *comp_h;
    int launch_block_dim;
public:
    CC(const std::string &filename);
    virtual void compute(GPUstream stream, int* preempted, int* executed, std::atomic<int> &lc_tasks, std::atomic<int> &be_tasks) override;
    virtual ~CC();
};

}

namespace bfs {

class BFS : public GraphCompute {
private:
    uint32_t *label_d;
    bool *changed_d;
    bool changed_h;
    uint64_t iter;
    const uint64_t max_iter = 100;
    dim3 blockDim;
    int numthreads = BLOCK_SIZE;
    int src;
    int launch_block_dim;
public:
    BFS(const std::string &filename);
    virtual void compute(GPUstream stream, int* preempted, int* executed, std::atomic<int> &lc_tasks, std::atomic<int> &be_tasks) override;
    virtual ~BFS();
};

}

namespace pagerank {
typedef float ValueT;

class PageRank : public GraphCompute {
private:
    bool *label_d;
    ValueT *delta_d;
    ValueT *residual_d;
    ValueT *value_d;
    bool *changed_d;
    bool changed_h;
    const uint64_t max_iter = 100;
    const ValueT tolerance = 0.001;
    const ValueT alpha = 0.85;
    dim3 blockDim;
    dim3 blockDim_update;
    int numthreads = BLOCK_SIZE;
    int launch_block_dim1;
    int launch_block_dim2;
    int launch_block_dim3;
public:
    PageRank(const std::string &filename);
    virtual void compute(GPUstream stream, int* preempted, int* executed, std::atomic<int> &lc_tasks, std::atomic<int> &be_tasks) override;
    virtual ~PageRank();
};

}

namespace sssp {
typedef uint32_t WeightT;

class SSSP : public GraphCompute {
private:
    bool *label_d;
    WeightT *costList_d;
    WeightT *newCostList_d;
    bool *changed_d;
    bool changed_h;
    const uint64_t max_iter = 100;
    dim3 blockDim;
    dim3 blockDim_update;
    int numthreads = BLOCK_SIZE;
    int src;
    WeightT *weightList_h;
    WeightT *weightList_d;
    int launch_block_dim1;
    int launch_block_dim2;
public:
    SSSP(const std::string &filename);
    virtual void compute(GPUstream stream, int* preempted, int* executed, std::atomic<int> &lc_tasks, std::atomic<int> &be_tasks) override;
    virtual ~SSSP();
};

}

} // namespace GraphComputeBlp



#endif // GRAPHCOMPUTE_H