#pragma once
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <ctime>
#include <iostream>
#include <chrono>
#include "util/gpu_util.h"

namespace SciComputeRaw {

void sci_init();
void reductions(double &mass , double &te);

void perform_timestep(GPUstream stream);
void finalize();

} // namespace SciComputeRaw

namespace SciComputeBlp {

void sci_init();
void reductions(double &mass , double &te);

void perform_timestep(GPUstream stream, int* preempted, int* pStopIndex, int* executed, int start_from);
void finalize();

} // namespace SciComputeBlp