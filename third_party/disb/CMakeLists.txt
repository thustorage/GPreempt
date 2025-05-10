# Original Copyright 2023 SJTU-IPADS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ------------------------------------------------------------------------------
# MODIFICATIONS:
# This file has been modified in 2025.
#
# The following changes were made to the original file:
# - Removed 'add_subdirectory(samples)'
# - Removed 'add_subdirectory(benchmarks/frameworks)'
#
# All modifications are also licensed under the Apache License, Version 2.0.
# ------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.5)
project(DISB)

option(BUILD_TENSORRT "whether to build tensorrt" OFF)
option(BUILD_TRITON "whether to build triton" OFF)
option(BUILD_TFSERVING "whether to build tfserving" OFF)
option(BUILD_REEF "whether to build reef" OFF)

include_directories("${PROJECT_SOURCE_DIR}/include")
aux_source_directory("${PROJECT_SOURCE_DIR}/src" disb_src)

add_library(disb STATIC ${disb_src})

target_link_libraries(disb jsoncpp pthread)

file(GLOB_RECURSE include_src
    ${CMAKE_CURRENT_LIST_DIR}/include/*.h
)

install(TARGETS disb ARCHIVE DESTINATION lib)
install(FILES ${include_src} DESTINATION include)