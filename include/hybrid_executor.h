#pragma once

#include "executor.h"
#include "util/gpu_util.h"
#include "util/common.h"

namespace reef {
namespace server {
    class REEFScheduler;
} // namespace server
namespace executor {

using namespace foo;
// HybridExecutor contains two version of the model
//   (1) transformed version, which is used to perform dynamic kernel padding
//   (2) preemptable version, which is used to perform reset-based preemption (for best-effort tasks).
//
// The transformed version is inherited from TransExecutor.
// 
// The preemptable version adds preemption flag based on the raw model.
class HybridExecutor : public BaseExecutor {

friend class server::REEFScheduler;

public:
    HybridExecutor();
    virtual ~HybridExecutor();
    
    virtual Status load_model(std::string model_path) override;

    Status execute_preemptale(GPUstream stream = GPUStreamDefault);

    Status launch_preempt_kernel(int kernel_offset, GPUstream stream);

    Status set_preempt_flag(GPUdeviceptr flag);
    
    Status get_reset_kernel_idx(int start_inx, int& ret);

protected:
    Status init_hybrid_executor(std::string model_path);

protected:
    GPUdeviceptr preempt_flag;
    GPUmodule preempt_mod;
    std::vector<KernelInfo> preempt_kernel_infos;

}; 

} // namespace executor
} // namespace reef 



