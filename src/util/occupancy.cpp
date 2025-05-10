#define ARCH 70
#define ARCH_RUNTIME_SHARED_MEMORY 0

#include "util/occupancy.h"
#include "util/common.h"

namespace foo {
namespace util {

template <typename Arch>
struct Config;

template <>
struct Config<std::integral_constant<int, 20>> {
    static constexpr int threadsPerWarp = 32;
    static constexpr int warpsPerMultiprocessor = 48;
    static constexpr int threadBlocksPerMultiprocessor = 8;
    static constexpr int sharedMemoryPerMultiprocessor = 49152;
    static constexpr int registerFileSize = 32768;
    static constexpr int registerAllocationUnitSize = 64;
    static constexpr int maxRegistersPerThread = 63;
    static constexpr int maxRegistersPerBlock = 32768;
    static constexpr int sharedMemoryAllocationUnitSize = 128;
    static constexpr int warpAllocationGranularity = 2;
};

template <>
struct Config<std::integral_constant<int, 21>> {
    static constexpr int threadsPerWarp = 32;
    static constexpr int warpsPerMultiprocessor = 48;
    static constexpr int threadBlocksPerMultiprocessor = 8;
    static constexpr int sharedMemoryPerMultiprocessor = 49152;
    static constexpr int registerFileSize = 32768;
    static constexpr int registerAllocationUnitSize = 64;
    static constexpr int maxRegistersPerThread = 63;
    static constexpr int maxRegistersPerBlock = 32768;
    static constexpr int sharedMemoryAllocationUnitSize = 128;
    static constexpr int warpAllocationGranularity = 2;
};

template <>
struct Config<std::integral_constant<int, 30>> {
    static constexpr int threadsPerWarp = 32;
    static constexpr int warpsPerMultiprocessor = 64;
    static constexpr int threadBlocksPerMultiprocessor = 16;
    static constexpr int sharedMemoryPerMultiprocessor = 49152;
    static constexpr int registerFileSize = 65536;
    static constexpr int registerAllocationUnitSize = 256;
    static constexpr int maxRegistersPerThread = 63;
    static constexpr int maxRegistersPerBlock = 65536;
    static constexpr int sharedMemoryAllocationUnitSize = 256;
    static constexpr int warpAllocationGranularity = 4;
};

template <>
struct Config<std::integral_constant<int, 32>> {
    static constexpr int threadsPerWarp = 32;
    static constexpr int warpsPerMultiprocessor = 64;
    static constexpr int threadBlocksPerMultiprocessor = 16;
    static constexpr int sharedMemoryPerMultiprocessor = 49152;
    static constexpr int registerFileSize = 65536;
    static constexpr int registerAllocationUnitSize = 256;
    static constexpr int maxRegistersPerThread = 255;
    static constexpr int maxRegistersPerBlock = 65536;
    static constexpr int sharedMemoryAllocationUnitSize = 256;
    static constexpr int warpAllocationGranularity = 4;
};

template <>
struct Config<std::integral_constant<int, 35>> {
    static constexpr int threadsPerWarp = 32;
    static constexpr int warpsPerMultiprocessor = 64;
    static constexpr int threadBlocksPerMultiprocessor = 16;
    static constexpr int sharedMemoryPerMultiprocessor = 49152;
    static constexpr int registerFileSize = 65536;
    static constexpr int registerAllocationUnitSize = 256;
    static constexpr int maxRegistersPerThread = 255;
    static constexpr int maxRegistersPerBlock = 65536;
    static constexpr int sharedMemoryAllocationUnitSize = 256;
    static constexpr int warpAllocationGranularity = 4;
};

template <>
struct Config<std::integral_constant<int, 37>> {
    static constexpr int threadsPerWarp = 32;
    static constexpr int warpsPerMultiprocessor = 64;
    static constexpr int threadBlocksPerMultiprocessor = 16;
    static constexpr int sharedMemoryPerMultiprocessor = 114688;
    static constexpr int registerFileSize = 131072;
    static constexpr int registerAllocationUnitSize = 256;
    static constexpr int maxRegistersPerThread = 255;
    static constexpr int maxRegistersPerBlock = 65536;
    static constexpr int sharedMemoryAllocationUnitSize = 256;
    static constexpr int warpAllocationGranularity = 4;
};

template <>
struct Config<std::integral_constant<int, 50>> {
    static constexpr int threadsPerWarp = 32;
    static constexpr int warpsPerMultiprocessor = 64;
    static constexpr int threadBlocksPerMultiprocessor = 32;
    static constexpr int sharedMemoryPerMultiprocessor = 65536;
    static constexpr int registerFileSize = 65536;
    static constexpr int registerAllocationUnitSize = 256;
    static constexpr int maxRegistersPerThread = 255;
    static constexpr int maxRegistersPerBlock = 65536;
    static constexpr int sharedMemoryAllocationUnitSize = 256;
    static constexpr int warpAllocationGranularity = 4;
};

template <>
struct Config<std::integral_constant<int, 52>> {
    static constexpr int threadsPerWarp = 32;
    static constexpr int warpsPerMultiprocessor = 64;
    static constexpr int threadBlocksPerMultiprocessor = 32;
    static constexpr int sharedMemoryPerMultiprocessor = 98304;
    static constexpr int registerFileSize = 65536;
    static constexpr int registerAllocationUnitSize = 256;
    static constexpr int maxRegistersPerThread = 255;
    static constexpr int maxRegistersPerBlock = 32768;
    static constexpr int sharedMemoryAllocationUnitSize = 256;
    static constexpr int warpAllocationGranularity = 4;
};

template <>
struct Config<std::integral_constant<int, 53>> {
    static constexpr int threadsPerWarp = 32;
    static constexpr int warpsPerMultiprocessor = 64;
    static constexpr int threadBlocksPerMultiprocessor = 32;
    static constexpr int sharedMemoryPerMultiprocessor = 65536;
    static constexpr int registerFileSize = 65536;
    static constexpr int registerAllocationUnitSize = 256;
    static constexpr int maxRegistersPerThread = 255;
    static constexpr int maxRegistersPerBlock = 32768;
    static constexpr int sharedMemoryAllocationUnitSize = 256;
    static constexpr int warpAllocationGranularity = 4;
};

template <>
struct Config<std::integral_constant<int, 60>> {
    static constexpr int threadsPerWarp = 32;
    static constexpr int warpsPerMultiprocessor = 64;
    static constexpr int threadBlocksPerMultiprocessor = 32;
    static constexpr int sharedMemoryPerMultiprocessor = 65536;
    static constexpr int registerFileSize = 65536;
    static constexpr int registerAllocationUnitSize = 256;
    static constexpr int maxRegistersPerThread = 255;
    static constexpr int maxRegistersPerBlock = 65536;
    static constexpr int sharedMemoryAllocationUnitSize = 256;
    static constexpr int warpAllocationGranularity = 2;
};

template <>
struct Config<std::integral_constant<int, 61>> {
    static constexpr int threadsPerWarp = 32;
    static constexpr int warpsPerMultiprocessor = 64;
    static constexpr int threadBlocksPerMultiprocessor = 32;
    static constexpr int sharedMemoryPerMultiprocessor = 98304;
    static constexpr int registerFileSize = 65536;
    static constexpr int registerAllocationUnitSize = 256;
    static constexpr int maxRegistersPerThread = 255;
    static constexpr int maxRegistersPerBlock = 65536;
    static constexpr int sharedMemoryAllocationUnitSize = 256;
    static constexpr int warpAllocationGranularity = 4;
};

template <>
struct Config<std::integral_constant<int, 62>> {
    static constexpr int threadsPerWarp = 32;
    static constexpr int warpsPerMultiprocessor = 64;
    static constexpr int threadBlocksPerMultiprocessor = 32;
    static constexpr int sharedMemoryPerMultiprocessor = 65536;
    static constexpr int registerFileSize = 65536;
    static constexpr int registerAllocationUnitSize = 256;
    static constexpr int maxRegistersPerThread = 255;
    static constexpr int maxRegistersPerBlock = 65536;
    static constexpr int sharedMemoryAllocationUnitSize = 256;
    static constexpr int warpAllocationGranularity = 4;
};

template <>
struct Config<std::integral_constant<int, 70>> {
    static constexpr int threadsPerWarp = 32;
    static constexpr int warpsPerMultiprocessor = 64;
    static constexpr int threadBlocksPerMultiprocessor = 32;
    static constexpr int sharedMemoryPerMultiprocessor = 98304;
    static constexpr int registerFileSize = 65536;
    static constexpr int registerAllocationUnitSize = 256;
    static constexpr int maxRegistersPerThread = 255;
    static constexpr int maxRegistersPerBlock = 65536;
    static constexpr int sharedMemoryAllocationUnitSize = 256;
    static constexpr int warpAllocationGranularity = 4;
};

template <>
struct Config<std::integral_constant<int, 75>> {
    static constexpr int threadsPerWarp = 32;
    static constexpr int warpsPerMultiprocessor = 32;
    static constexpr int threadBlocksPerMultiprocessor = 16;
    static constexpr int sharedMemoryPerMultiprocessor = 65536;
    static constexpr int registerFileSize = 65536;
    static constexpr int registerAllocationUnitSize = 256;
    static constexpr int maxRegistersPerThread = 255;
    static constexpr int maxRegistersPerBlock = 65536;
    static constexpr int sharedMemoryAllocationUnitSize = 256;
    static constexpr int warpAllocationGranularity = 4;
};

template <>
struct Config<std::integral_constant<int, 80>> {
    static constexpr int threadsPerWarp = 32;
    static constexpr int warpsPerMultiprocessor = 64;
    static constexpr int threadBlocksPerMultiprocessor = 32;
    static constexpr int sharedMemoryPerMultiprocessor = 167936;
    static constexpr int registerFileSize = 65536;
    static constexpr int registerAllocationUnitSize = 256;
    static constexpr int maxRegistersPerThread = 255;
    static constexpr int maxRegistersPerBlock = 65536;
    static constexpr int sharedMemoryAllocationUnitSize = 128;
    static constexpr int warpAllocationGranularity = 4;
};

template <>
struct Config<std::integral_constant<int, 86>> {
    static constexpr int threadsPerWarp = 32;
    static constexpr int warpsPerMultiprocessor = 48;
    static constexpr int threadBlocksPerMultiprocessor = 16;
    static constexpr int sharedMemoryPerMultiprocessor = 102400;
    static constexpr int registerFileSize = 65536;
    static constexpr int registerAllocationUnitSize = 256;
    static constexpr int maxRegistersPerThread = 255;
    static constexpr int maxRegistersPerBlock = 65536;
    static constexpr int sharedMemoryAllocationUnitSize = 128;
    static constexpr int warpAllocationGranularity = 4;
};

typedef Config<std::integral_constant<int, ARCH>> CONFIG;

int get_max_block_per_SM(int threadsPerBlock, int registersPerThread, int sharedMemoryPerBlock) {
    auto blockWraps = div_ceil(threadsPerBlock, CONFIG::threadsPerWarp);
    auto registersPerWarp = align_up(registersPerThread * CONFIG::threadsPerWarp, CONFIG::registerAllocationUnitSize);
    auto warpsPerMultiprocessorLimitedByRegisters = align_down(CONFIG::maxRegistersPerBlock / registersPerWarp, CONFIG::warpAllocationGranularity);
    auto blockSharedMemory = align_up(sharedMemoryPerBlock + ARCH_RUNTIME_SHARED_MEMORY, CONFIG::sharedMemoryAllocationUnitSize);
    auto threadBlocksPerMultiprocessorLimitedByWarpsOrBlocksPerMultiprocessor = std::min(
            CONFIG::threadBlocksPerMultiprocessor,
            CONFIG::warpsPerMultiprocessor / blockWraps
    );
    int threadBlocksPerMultiprocessorLimitedByRegistersPerMultiprocessor;
    if (registersPerThread > CONFIG::maxRegistersPerThread) {
        threadBlocksPerMultiprocessorLimitedByRegistersPerMultiprocessor = 0;
    } else if (registersPerThread > 0) {
        threadBlocksPerMultiprocessorLimitedByRegistersPerMultiprocessor = (warpsPerMultiprocessorLimitedByRegisters / blockWraps) * (CONFIG::registerFileSize / CONFIG::maxRegistersPerBlock);
    } else {
        threadBlocksPerMultiprocessorLimitedByRegistersPerMultiprocessor = CONFIG::threadBlocksPerMultiprocessor;
    }

    int threadBlocksPerMultiprocessorLimitedBySharedMemoryPerMultiprocessor;
    if (blockSharedMemory > 0) {
        threadBlocksPerMultiprocessorLimitedBySharedMemoryPerMultiprocessor = CONFIG::sharedMemoryPerMultiprocessor / blockSharedMemory;
    } else {
        threadBlocksPerMultiprocessorLimitedBySharedMemoryPerMultiprocessor = CONFIG::threadBlocksPerMultiprocessor;
    }

    return std::min(
        threadBlocksPerMultiprocessorLimitedByWarpsOrBlocksPerMultiprocessor,
        std::min(
            threadBlocksPerMultiprocessorLimitedByRegistersPerMultiprocessor,
            threadBlocksPerMultiprocessorLimitedBySharedMemoryPerMultiprocessor
        )
    );
}

} // namespace util
} // namespace reef