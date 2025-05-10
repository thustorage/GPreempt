#include "workloads/scicomput.h"
#include "util/gpu_util.h"

using namespace SciComputeRaw;

int main(){
    foo::util::init_cuda();
    GPUstream stream;
    GPUStreamCreate(&stream, 0);
    sci_init();
    double mass0, te0;
    reductions(mass0, te0);
    perform_timestep(0);
    finalize();

    sci_init();
    perform_timestep(0);
    double mass, te;
    reductions(mass, te);
    printf( "d_mass: %le\n" , (mass - mass0)/mass0 );
    printf( "d_te:   %le\n" , (te   - te0  )/te0   );
    finalize();
    return 0;
}