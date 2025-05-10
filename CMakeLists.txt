project(GPreempt)
cmake_minimum_required(VERSION 3.10)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS}")
# set(CMAKE_BUILD_TYPE "Debug")

#set pic
set_property(GLOBAL PROPERTY POSITION_INDEPENDENT_CODE ON)

link_directories("/usr/local/lib/")
include_directories("/usr/local/include/")

if(NOT USE_HIP)
    find_package(CUDA)
    message(STATUS "CUDA found")
    set(USE_CUDA ON)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 80)
    set(CMAKE_CUDA_FLAGS "-O3 ${CMAKE_CUDA_FLAGS}")
    add_definitions(-DCUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
    set(GPU_LIBRARIES cuda gdrapi ${CUDA_LIBRARIES})
else()
    message(STATUS "CUDA not found, use HIP instead")
    list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")
    enable_language(HIP)
    # find_package(hip REQUIRED)
    set(CMAKE_HIP_ARCHITECTURES gfx908)
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3")
    add_definitions(-DHIP)
    add_definitions(-D__HIP_PLATFORM_AMD__)
    include_directories("/opt/rocm/include")
    link_directories("/opt/rocm/lib")
    set(GPU_LIBRARIES ${HIP_LIBRARIES})
    set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
endif()

# find_package(glog REQUIRED)
# find_package(jsoncpp REQUIRED)

add_subdirectory(third_party/glog)
add_subdirectory(third_party/jsoncpp)

include_directories("${PROJECT_SOURCE_DIR}/third_party/jsoncpp/include")
link_directories(${CMAKE_BINARY_DIR}/third_party/jsoncpp/src/lib_json)

add_subdirectory(third_party/disb)
include_directories("${PROJECT_SOURCE_DIR}/third_party/disb/include")

add_definitions(-DMODEL_PATH="${CMAKE_CURRENT_LIST_DIR}/model")
add_definitions(-DBUILD_PATH="${CMAKE_CURRENT_LIST_DIR}/build")

include_directories("${PROJECT_SOURCE_DIR}/include")

aux_source_directory("${PROJECT_SOURCE_DIR}/src/util" util)

file(GLOB WORKLOADS_SRC "${PROJECT_SOURCE_DIR}/src/workloads/*.cu")
add_library(workloads ${WORKLOADS_SRC} ${util})
target_link_libraries(workloads glog::glog)
if(USE_CUDA)
    set_source_files_properties(${WORKLOADS_SRC} PROPERTIES LANGUAGE CUDA)
else()
    set_source_files_properties(${WORKLOADS_SRC} PROPERTIES LANGUAGE HIP)
endif()

if(USE_CUDA) 
    set_source_files_properties(${PROJECT_SOURCE_DIR}/src/model.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${PROJECT_SOURCE_DIR}/src/executor.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${util} PROPERTIES LANGUAGE CUDA)
else()
    set_source_files_properties(${PROJECT_SOURCE_DIR}/src/model.cpp PROPERTIES LANGUAGE HIP)
    set_source_files_properties(${PROJECT_SOURCE_DIR}/src/executor.cpp PROPERTIES LANGUAGE HIP)
    set_source_files_properties(${util} PROPERTIES LANGUAGE HIP)
endif()

add_library(executor
    ${PROJECT_SOURCE_DIR}/src/model.cpp
    ${PROJECT_SOURCE_DIR}/src/executor.cpp
    ${util}
)

add_library(gpreempt
    ${PROJECT_SOURCE_DIR}/src/gpreempt.cpp
)

target_link_libraries(executor
    disb
    workloads
    ${GPU_LIBRARIES}
    glog::glog
    jsoncpp_lib
    gpreempt
)

set(CUBIN_OUTPUT "block.cubin")
if (USE_CUDA)
    add_custom_command(
        OUTPUT ${CUBIN_OUTPUT}
        COMMAND ${CMAKE_CUDA_COMPILER} -arch=sm_80 ${PROJECT_SOURCE_DIR}/src/block.cu -o ${CUBIN_OUTPUT} -cubin
        DEPENDS ${PROJECT_SOURCE_DIR}/src/block.cu
        COMMENT "Compiling block.cu to CUBIN"
    )
else()
    add_custom_command(
        OUTPUT ${CUBIN_OUTPUT}
        COMMAND ${HIP_HIPCC_EXECUTABLE} --offload-arch=gfx908 ${PROJECT_SOURCE_DIR}/src/block.cu -o ${CUBIN_OUTPUT} --genco
        DEPENDS ${PROJECT_SOURCE_DIR}/src/block.cu
        COMMENT "Compiling block.cu to CUBIN"
    )
endif()
add_custom_target(block_cubin ALL DEPENDS ${CUBIN_OUTPUT})

file(GLOB COMMON_SOURCES "${PROJECT_SOURCE_DIR}/src/common-clients/*.cpp")

if(USE_CUDA)
    file(GLOB CLIENT_SOURCES "${PROJECT_SOURCE_DIR}/src/cuda-clients/*.cpp")
    set(CLIENT_SOURCES ${CLIENT_SOURCES} ${COMMON_SOURCES})
    set_source_files_properties(${CLIENT_SOURCES} PROPERTIES LANGUAGE CUDA)
else()
    file(GLOB CLIENT_SOURCES "${PROJECT_SOURCE_DIR}/src/hip-clients/*.cpp")
    set(CLIENT_SOURCES ${CLIENT_SOURCES} ${COMMON_SOURCES})
    set_source_files_properties(${CLIENT_SOURCES} PROPERTIES LANGUAGE HIP)
endif()

foreach(CLIENT_SOURCE ${CLIENT_SOURCES})
    get_filename_component(CLIENT_NAME ${CLIENT_SOURCE} NAME_WE)
    add_executable(${CLIENT_NAME} ${CLIENT_SOURCE})
    target_link_libraries(${CLIENT_NAME}
        disb 
        executor
        ${GPU_LIBRARIES}
        glog::glog
        jsoncpp_lib
    )
endforeach()

if(USE_HIP)
    file(GLOB REEF_CLIENT_SOURCES "${PROJECT_SOURCE_DIR}/src/reef-client/*.cpp")
    set_source_files_properties(${REEF_CLIENT_SOURCES} PROPERTIES LANGUAGE HIP)
    add_executable(reef_client ${REEF_CLIENT_SOURCES})
    target_link_libraries(reef_client
        disb
        executor
        ${GPU_LIBRARIES}
        glog::glog
        jsoncpp_lib
    )
endif()

file(GLOB TEST_SOURCES "${PROJECT_SOURCE_DIR}/src/tests/test-*.cpp")

if(USE_CUDA)
    set_source_files_properties(${TEST_SOURCES} PROPERTIES LANGUAGE CUDA)
else()
    set_source_files_properties(${TEST_SOURCES} PROPERTIES LANGUAGE HIP)
endif()

foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME}
        executor
        ${GPU_LIBRARIES}
        glog::glog
        jsoncpp_lib
    )
endforeach()

