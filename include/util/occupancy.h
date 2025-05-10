#pragma once

#include <string>
#include <unordered_map>

namespace foo {
namespace util {

int get_max_block_per_SM(int threadsPerBlock, int registersPerThread, int sharedMemoryPerBlock);

} // namespace util
} // namespace reef
