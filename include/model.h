#pragma once

#include "util/common.h"
#include "util/gpu_util.h"
#include <unordered_map>

namespace foo {

using namespace foo::util;

// The information of a model parameter.
class StorageInfo {
public:
    std::string name;
    size_t size;
    std::string stype;
};

// The information of a GPU kernel.
class KernelInfo {
public:
    std::string name;
    uint32_t launch_params[6]; // gridDims & blockDims
    GPUfunction handler; // The GPUfunction handle.

    std::vector<size_t> args_index;
    std::vector<GPUdeviceptr> args;  // store device ptr for each kernel's args
    std::vector<GPUdeviceptr *> args_ptr;  // store the **host ptr** of device ptr for each kernel's args, it is used to launch the kernel
};

// The model class for a REEF model.
class Model {
public:
    Model();

    Model(const std::string &device_json, const std::string &host_json, const std::string &cuda_module);

    ~Model();

    Status load_model(const std::string &device_json, const std::string &host_json, const std::string &cuda_module);

    Status load_param(const char *param_file_path);

    bool is_loaded();

    size_t get_kernel_num() const;

    size_t get_storage_num() const;

    size_t get_storage_size(size_t idx);

    StorageInfo &get_storage_info(size_t idx);

    KernelInfo &get_kernel_info(size_t idx);

    GPUdeviceptr get_storage_ptr(size_t idx);

    /// get the storage data by index
    Status get(size_t idx, void *out, size_t len, GPUstream stream);

    /// get the storage data by name
    Status get(const std::string &key, void *out, size_t len, GPUstream stream);

    /// set the storage data by index
    Status set(size_t idx, void *value, size_t len, GPUstream stream);

    /// set the storage data by name
    Status set(const std::string &key, void *value, size_t len, GPUstream stream);

    static size_t get_stype_size(std::string &stype);

    Status find_storage_idx(const std::string &name, size_t &idx);

    Status get_global_ptr(const std::string &name, GPUdeviceptr *ptr);

    Status clear();

    Status batchAddParam(GPUdeviceptr *ptr);
    
    Status add_block_preemption_param(GPUdeviceptr ptr);

private:
    GPUmodule module{};

    std::vector<StorageInfo> storage_infos;
    std::vector<KernelInfo> kernel_infos;

    std::vector<GPUdeviceptr> storages;  // store the device ptr of input, output and param
    std::unordered_map<std::string, size_t> storages_idx;  // store the index of storages
    std::vector<dim3> shapes;

    Status load_from_json(const char *device_json, const char *host_json);
};

typedef std::unordered_map<std::string, std::vector<float>> ModelParam;

class ModelParamParser {
public:
    static ModelParam *parse_from_file(const char *param_file);
};

} // namespace fii

