// # Original Copyright 2023 SJTU-IPADS
// #
// # Licensed under the Apache License, Version 2.0 (the "License");
// # you may not use this file except in compliance with the License.
// # You may obtain a copy of the License at
// #
// #     http://www.apache.org/licenses/LICENSE-2.0
// #
// # Unless required by applicable law or agreed to in writing, software
// # distributed under the License is distributed on an "AS IS" BASIS,
// # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// # See the License for the specific language governing permissions and
// # limitations under the License.
// #
// # ------------------------------------------------------------------------------
// # MODIFICATIONS:
// # This file has been modified in 2025.
// // #
// # The following changes were made to the original file:
// # - Extract the reef scheduler from the original file.
// # - Remove dynamic kernel padding.
// # 
// # All modifications are also licensed under the Apache License, Version 2.0.
// # ------------------------------------------------------------------------------

#include "hybrid_executor.h"

namespace reef {
namespace executor {

HybridExecutor::HybridExecutor() {

}

HybridExecutor::~HybridExecutor() {

}

Status HybridExecutor::load_model(std::string model_path) {
    model = std::make_shared<Model>();
    model->load_model(model_path + "/mod.json", model_path + "/host.json", model_path + "/mod.cubin");
    CUDA_RETURN_STATUS(GPUModuleLoad(&preempt_mod, (model_path + "/mod.cubin").c_str()));
    for (size_t i = 0; i < model->get_kernel_num(); i++) {
        KernelInfo kernel_info = model->get_kernel_info(i);
        if(kernel_info.name == "nop") {
            preempt_kernel_infos.push_back(kernel_info);
            continue;
        }
        GPUfunction func;
        CUDA_RETURN_STATUS(GPUModuleGetFunction(&func, preempt_mod, kernel_info.name.c_str()));
        kernel_info.handler = func;
        kernel_info.args.push_back(preempt_flag);
        kernel_info.args_ptr.push_back(&preempt_flag);
        preempt_kernel_infos.push_back(kernel_info);
    }
    type = "dnn";
    return Status::Succ;
}

Status HybridExecutor::set_preempt_flag(GPUdeviceptr flag) {
    preempt_flag = flag;
    return Status::Succ;
}

Status HybridExecutor::execute_preemptale(GPUstream stream) {
    for (int i = 0; i < this->model->get_kernel_num(); i++) {
        Status ret = launch_preempt_kernel(i, stream);
        if (ret != Status::Succ) return ret;
    }
    CUDA_RETURN_STATUS(GPUStreamSynchronize(stream));
    return Status::Succ;
}

Status HybridExecutor::launch_preempt_kernel(int kernel_offset, GPUstream stream) {
    auto& kernel_info = preempt_kernel_infos[kernel_offset];
    std::string& func_name = kernel_info.name;
    if(func_name == "nop") {
        size_t size = model->get_storage_size(kernel_info.args_index[0]);
        // LOG(INFO) << "Copying data from " << (void*)kernel_info.args[0] << " to " << (void*)kernel_info.args[1] << ", size: " << size << ", storage size: " << model->get_storage_size(kernel_info.args_index[0]) << ", storage size: " << model->get_storage_size(kernel_info.args_index[1]);
        CUDA_RETURN_STATUS(GPUMemcpyDtoDAsync(
            kernel_info.args[1], kernel_info.args[0], size, stream
        ));
        return Status::Succ;
    }
    GPUfunction func = kernel_info.handler;
    auto& launch_params = kernel_info.launch_params;
    // std::stringstream ss;
    // ss << func_name << " <<<(" << launch_params[0] << ", " << launch_params[1] << ", " << launch_params[2] << "), ("
    //    << launch_params[3] << ", " << launch_params[4] << ", " << launch_params[5] << ")>>>";
    // for(auto arg_ptr: kernel_info.args_ptr) {
    //     ss << " " << (void*)*arg_ptr;
    // }
    // LOG(INFO) << "Launching kernel: " << ss.str();

    CUDA_RETURN_STATUS(GPULaunchKernel(
        func,
        launch_params[0], launch_params[1], launch_params[2],
        launch_params[3], launch_params[4], launch_params[5],
        0, stream, (void**)(kernel_info.args_ptr.data()), nullptr
    ));
    return Status::Succ;
}


Status HybridExecutor::get_reset_kernel_idx(int start_inx, int& ret) {
    return Status::Succ; // TODO:
}

} // namespace executor
} // namespace reef