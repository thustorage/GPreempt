
#include <cassert>
#include <sstream>
#include <fstream>
#include <memory>
#include <chrono>
#include <json/json.h>

#include "model.h"
#include "util/common.h"


namespace foo {

#define PARAM_MAGIC "TVM_MODEL_PARAMS"

ModelParam *ModelParamParser::parse_from_file(const char *param_file) {
    FILE *fp;
    fp = fopen(param_file, "rb");
    char magic[sizeof(PARAM_MAGIC)];
    size_t res = fread(magic, sizeof(char), sizeof(PARAM_MAGIC), fp);
    ASSERT(res == sizeof(PARAM_MAGIC));
    ASSERT(std::string(magic) == PARAM_MAGIC);

    uint64_t params_size;
    res = fread(&params_size, sizeof(uint64_t), 1, fp);
    ASSERT(res == 1);
    ASSERT(params_size != 0);

    ModelParam *params = new ModelParam(params_size);
    for (uint64_t i = 0; i < params_size; i++) {
        char key_buf[256];
        uint64_t key_len = 0;
        while (true) {
            char c;
            res = fread(&c, sizeof(char), 1, fp);
            assert(res == 1);
            key_buf[key_len] = c;
            key_len++;
            if (c == '\0') break;
        }
        std::string key(key_buf);
        uint64_t array_size;
        res = fread(&array_size, sizeof(uint64_t), 1, fp);
        ASSERT(res == 1);
        ASSERT(array_size != 0);
        std::vector<float> array(array_size);
        array.resize(array_size);
        res = fread(array.data(), sizeof(float), array_size, fp);
        ASSERT(res == array_size);
        params->insert({key, array});
        // LOG(INFO) << "Find param: " << key << ", size: " << array_size;
    }
    return params;
}

Model::Model() {
    module = nullptr;
}

Model::Model(const std::string &device_json, const std::string &host_json, const std::string &cuda_module) {
    load_model(device_json, host_json, cuda_module);
}

Model::~Model() {
    clear();
}

size_t Model::get_stype_size(std::string &stype) {
    if (stype == "float32") return 4;
    if (stype == "int64") return 8;
    if (stype == "byte") return 1;
    if (stype == "uint1") return 1;
    if (stype == "int32") return 4;
    LOG(ERROR) << "Unknown stype: " << stype;
    std::terminate();
}

Status Model::load_model(const std::string &device_json, const std::string &host_json, const std::string &cuda_module) {

    if(is_loaded()) {
        LOG(ERROR) << "The model is already is_loaded";
        return Status::Fail;
    }

    // LOG(INFO) << "Load model from device json: " << device_json << ", host json: " << host_json << ", cuda module: " << cuda_module;
    CUDA_RETURN_STATUS(GPUModuleLoad(&module, cuda_module.c_str()));
    // LOG(INFO) << "Load GPU module: " << module << ", from: " << cuda_module;
    
    RETURN_STATUS(load_from_json(device_json.c_str(), host_json.c_str()));
    // LOG(INFO) << "Load device json: " << device_json << ", host json: " << host_json;

    // Fill kernel handler (GPUfunction) for each kernel
    for (KernelInfo &kernel_info: kernel_infos) {
        if(kernel_info.name == "nop") continue;
        CUDA_RETURN_STATUS(GPUModuleGetFunction(&kernel_info.handler, module, kernel_info.name.c_str()));
        // LOG(INFO) << "Load GPU kernel: " << kernel_info.name;
    }
    // LOG(INFO) << "Load GPU kernel handlers num: " << kernel_infos.size();
    // Allocate storage
    storages.resize(storage_infos.size());
    for(size_t i = 0; i < storage_infos.size(); i++) {
        auto &storage_info = storage_infos[i];
        size_t size = storage_info.size;
        GPUdeviceptr device_ptr;
        std::vector<char> temp;
        temp.resize(size, 0);
        CUDA_RETURN_STATUS(GPUMemAlloc(&device_ptr, size));
        CUDA_RETURN_STATUS(GPUMemcpyHtoD(device_ptr, temp.data(), size));
        // LOG(INFO) << "Allocate GPU memory at: " << (void *) device_ptr << ", size: " << size << ", name: " << storage_info.name;
        storages[i] = device_ptr;
        storages_idx.emplace(storage_info.name, i);
    }
    // LOG(INFO) << "Allocate GPU memory num: " << storage_infos.size();

    // Fill kernel arg pointers
    for (KernelInfo &kernel_info: kernel_infos) {
        kernel_info.args.reserve(kernel_info.args_index.size() + 10);
        for (size_t arg_idx: kernel_info.args_index) {
            kernel_info.args.push_back(storages[arg_idx]);
            kernel_info.args_ptr.push_back(&storages[arg_idx]);
        }
    }
    return Status::Succ;
}

/// load the model info from the json file
Status Model::load_from_json(const char *device_json, const char *host_json) {
    // 0. load the json file
    std::ifstream difs(device_json);
    std::ifstream hifs(host_json);
    if (!difs.is_open()) {
        LOG(ERROR) << "Cannot open the json file: " << device_json;
        return Status::NotFound;
    }
    if (!hifs.is_open()) {
        LOG(ERROR) << "Cannot open the json file: " << host_json;
        return Status::NotFound;
    }
    std::string deviceJson((std::istreambuf_iterator<char>(difs)), std::istreambuf_iterator<char>());
    std::string hostJson((std::istreambuf_iterator<char>(hifs)), std::istreambuf_iterator<char>());
    difs.close();
    hifs.close();

    Json::Value dDoc;
    Json::Value hDoc;
    Json::Reader reader;
    if (!reader.parse(deviceJson, dDoc)) {
        LOG(ERROR) << "Failed to parse the json file: " << device_json;
        return Status::Fail;
    }
    if (!reader.parse(hostJson, hDoc)) {
        LOG(ERROR) << "Failed to parse the json file: " << host_json;
        return Status::Fail;
    }
    
    auto attrs = dDoc["attrs"];
    auto dltypes = attrs["dltype"][1];
    auto shapes = attrs["shape"][1];
    size_t global_len = dltypes.size();
    // Global storage
    for(int i = 0; i < global_len; i++) {
        auto node = dDoc["nodes"][i];
        StorageInfo storage_info;
        storage_info.name = node["name"].asString().data();
        storage_info.stype = dltypes[i].asString().data();
        size_t size = 1;
        for(auto shape : shapes[i]) {
            size *= shape.asUInt64();
        }
        storage_info.size = size * get_stype_size(storage_info.stype);
        this->storage_infos.push_back(storage_info);
    }
    size_t temp_len = 0;
    // Temp storage
    for(auto temp_size : hDoc["temp_args"]) {
        StorageInfo storage_info;
        storage_info.name = "temp";
        storage_info.stype = "float32";
        storage_info.size = temp_size.asUInt64();
        this->storage_infos.push_back(storage_info);
        temp_len++;
    }
    // LOG(INFO) << "Load storage num: " << global_len + temp_len;
    // Func args
    std::vector<std::pair<std::string, std::vector<size_t>>> func_args;
    for(int i = 0; i < global_len; i++) {
        auto node = dDoc["nodes"][i];
        if(node["inputs"].size()) {
            std::vector<size_t> args;
            for(auto input : node["inputs"]) {
                args.push_back(input[0].asUInt64());
            }
            args.push_back(i);
            func_args.push_back(std::make_pair(node["attrs"]["func_name"].asString().data(), args));
        }
    }
    // Kernel infos
    int i = 0;
    for(auto func : hDoc["funcs"]) {
        while(func_args[i].first == "__nop") {
            KernelInfo kernel_info;
            kernel_info.name = "nop";
            kernel_info.args_index.assign(func_args[i].second.begin(), func_args[i].second.end());
            kernel_infos.push_back(kernel_info);
            i++;
        }
        if(func["name"].asString() != func_args[i].first) {
            LOG(ERROR) << "Function name mismatch: " << func["name"].asString() << " vs " << func_args[i].first;
            return Status::Fail;
        }
        std::vector<size_t> args = func_args[i++].second;
        for(auto kernel : func["kernels"]) {
            KernelInfo kernel_info;
            kernel_info.name = kernel["name"].asString().data();
            for(auto arg : kernel["args"]) {
                size_t idx = arg.asInt64();
                if(idx < global_len) {
                    kernel_info.args_index.push_back(args[idx]);
                }  else {
                    kernel_info.args_index.push_back(global_len - idx - 1);
                }
            }
            for(int i = 0; i < 6; ++i) {
                kernel_info.launch_params[i] = kernel["launch_params"][i].asUInt64();
            }
            kernel_infos.push_back(kernel_info);
        }
    }
    // Heads (output)
    storage_infos[dDoc["heads"][0][0].asUInt64()].name = "heads";
    // LOG(INFO) << "Load heads: " << dDoc["heads"][0][0].asUInt64();

    return Status::Succ;
}

/// clear the model info is_loaded
Status Model::clear() {
    if (module == nullptr) {
        return Status::Succ;
    }

    // free the device memory
    for (GPUdeviceptr ptr: storages) {
        CUDA_RETURN_STATUS(GPUMemFree(ptr));
    }
    // unload the gpu module
    CUDA_RETURN_STATUS(GPUModuleUnload(module));

    // clear the model info
    storage_infos.clear();
    kernel_infos.clear();
    // args.clear();
    storages.clear();
    storages_idx.clear();

    module = nullptr;

    return Status::Succ; 
}

/// load the params from the param file
Status Model::load_param(const char *param_file_path) {
    // 1. load the param file
    std::unique_ptr<ModelParam> params(ModelParamParser::parse_from_file(param_file_path));

    // 2. copy the params from host to device
    for (size_t i = 0; i < storages.size(); i++) {
        // 2.1 for each storage info
        StorageInfo &storage_info = this->storage_infos[i];
        // 2.2 check if the storage info is in the params
        if (params->find(storage_info.name) == params->end()) continue;
        // 2.3 copy the params from host to device
        auto &array = params->at(storage_info.name);
        // LOG(INFO) << "Load param: " << storage_info.name << ", size: " << array.size();
        CUDA_RETURN_STATUS(GPUMemcpyHtoD(
                (GPUdeviceptr) storages[i], array.data(),
                storage_info.size));
    }
    // LOG(INFO) << "Load params from file: " << param_file_path;
    return Status::Succ;
}

bool Model::is_loaded() {
    return module != nullptr;
}

Status Model::find_storage_idx(const std::string &name, size_t &idx) {
    // LOG(INFO) << "Find storage: " << name;
    auto it = this->storages_idx.find(name);
    if (it == this->storages_idx.end()) RETURN_STATUS(Status::NotFound);
    idx = it->second;
    return Status::Succ;
}

Status Model::get(size_t idx, void *out, size_t len, GPUstream stream) {
    if (idx >= this->storages.size()) RETURN_STATUS(Status::OutOfRange);
    StorageInfo &storage_info = this->storage_infos[idx];
    size_t size = storage_info.size;
    if (len < size) RETURN_STATUS(Status::Fail);
    // LOG(INFO) << "this->storage[" << idx << "]: " << (void*)this->storages[idx];
    // LOG(INFO) << "Get storage: " << idx << ", storage size: " << size << ", len: " << len;
    CUDA_RETURN_STATUS(GPUMemcpyDtoHAsync(out, this->storages[idx], size, stream));
    return Status::Succ;
}

Status Model::get(const std::string &key, void *out, size_t len, GPUstream stream) {
    size_t storage_idx;
    if (find_storage_idx(key, storage_idx) != Status::Succ) RETURN_STATUS(Status::NotFound);
    return get(storage_idx, out, len, stream);
}

Status Model::set(size_t idx, void *value, size_t len, GPUstream stream) {
    if (idx >= storages.size()) RETURN_STATUS(Status::OutOfRange);
    StorageInfo &storage_info = this->storage_infos[idx];
    size_t size = storage_info.size;
    if (len < size) RETURN_STATUS(Status::OutOfRange);
    CUDA_RETURN_STATUS(GPUMemcpyHtoDAsync(this->storages[idx], value, size, stream));
    // LOG(INFO) << "Set storage: " << idx << ", storage size: " << size << ", len: " << len;
    return Status::Succ;
}

Status Model::set(const std::string &key, void *value, size_t len, GPUstream stream) {
    size_t storage_idx;
    if (find_storage_idx(key, storage_idx) != Status::Succ) RETURN_STATUS(Status::NotFound);
    return set(storage_idx, value, len, stream);
}

size_t Model::get_kernel_num() const {
    return kernel_infos.size();
}

size_t Model::get_storage_num() const {
    return storage_infos.size();
}

StorageInfo &Model::get_storage_info(size_t idx) {
    return storage_infos[idx];
}

GPUdeviceptr Model::get_storage_ptr(size_t idx) {
    return storages[idx];
}

size_t Model::get_storage_size(size_t idx) {
    return storage_infos[idx].size;
}

KernelInfo &Model::get_kernel_info(size_t idx) {
    return kernel_infos[idx];
}

Status Model::get_global_ptr(const std::string &name, GPUdeviceptr *ptr) {
    CUDA_RETURN_STATUS(GPUModuleGetGlobal(ptr, NULL, module, name.c_str()));
    return Status::Succ;
}

Status Model::batchAddParam(GPUdeviceptr *ptr) {
    for(auto &kernel_info : kernel_infos) {
        kernel_info.args.push_back(*ptr);
        kernel_info.args_ptr.push_back(ptr);
    }
    return Status::Succ;
}

Status Model::add_block_preemption_param(GPUdeviceptr ptr) {
    shapes.reserve(kernel_infos.size());
    for(int i = 0; i < kernel_infos.size(); i++){
        auto &kernel_info = kernel_infos[i];
        if(kernel_info.name == "nop"){
            continue;
        }
        int total_blocks = kernel_info.launch_params[0] * kernel_info.launch_params[1] * kernel_info.launch_params[2];
        int raw_block_size = kernel_info.launch_params[3] * kernel_info.launch_params[4] * kernel_info.launch_params[5];
        int gridsize, blocksize;
#ifdef CUDA
        GPUModuleOccupancyMaxPotentialBlockSizeWithFlags(&gridsize, &blocksize, kernel_info.handler, 0, 0, raw_block_size, 0);
#else
        ASSERT_CUDA_ERROR(GPUModuleOccupancyMaxPotentialBlockSizeWithFlags(&gridsize, &blocksize, kernel_info.handler, 0, raw_block_size, 0));
#endif
        // printf("Kernel name: %s, gridsize: %d, blocksize: %d\n", kernel_info.name.c_str(), gridsize, blocksize);
        dim3 raw_launch_params = dim3(kernel_info.launch_params[0], kernel_info.launch_params[1], kernel_info.launch_params[2]);
        kernel_info.launch_params[0] = std::min(total_blocks, gridsize);
        kernel_info.launch_params[1] = 1;
        kernel_info.launch_params[2] = 1;
        kernel_info.args.push_back((GPUdeviceptr)total_blocks);
        kernel_info.args_ptr.push_back((GPUdeviceptr *)&kernel_info.args.back());
        kernel_info.args.push_back((GPUdeviceptr)((uint64_t)ptr + i * sizeof(int)));
        kernel_info.args_ptr.push_back((GPUdeviceptr *)&kernel_info.args.back());
        shapes.push_back(raw_launch_params);
        kernel_info.args_ptr.push_back((GPUdeviceptr *)&shapes.back());
        kernel_info.args.push_back((GPUdeviceptr)i);
        kernel_info.args_ptr.push_back((GPUdeviceptr *)&kernel_info.args.back());
    }
    return Status::Succ;
}

} // namespace Executor::Model