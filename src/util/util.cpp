#include "util/common.h"

void bind_core(int core_id) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(core_id, &mask);
    int ret = sched_setaffinity(0, sizeof(mask), &mask);
    if (ret != 0) {
        LOG(ERROR) << "Failed to bind core " << core_id;
    }
}