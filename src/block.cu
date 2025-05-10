
extern "C" __global__ void gpu_block(volatile int *stop) {
    while(!*stop);
}

extern "C" __global__ void gpu_noop() {
    // Do nothing
}