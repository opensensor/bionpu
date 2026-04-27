# CMakeLists.txt — alternate host-runner build.
#
# The Makefile invokes g++-15 directly (matching /conv_stem). This
# CMakeLists.txt is provided as an alternative for users who prefer the
# mlir-aie programming_examples CMake flow. Both produce equivalent host
# binaries; the vendored artifact is built via the Makefile.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.30)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

project(crispr_match_multitile_runner LANGUAGES C CXX)

if(NOT DEFINED CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER gcc-15)
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER g++-15)
endif()

set(TARGET_NAME crispr_match_multitile CACHE STRING "Target to be built")

# Locate XRT.
if(DEFINED ENV{XILINX_XRT})
    set(XRT_ROOT $ENV{XILINX_XRT})
else()
    set(XRT_ROOT /opt/xilinx/xrt)
endif()

add_executable(${TARGET_NAME} runner.cpp)

target_include_directories(${TARGET_NAME} PUBLIC ${XRT_ROOT}/include)
target_link_directories(${TARGET_NAME} PUBLIC ${XRT_ROOT}/lib)
target_link_libraries(${TARGET_NAME} PUBLIC xrt_coreutil uuid)
set_target_properties(${TARGET_NAME} PROPERTIES INSTALL_RPATH "${XRT_ROOT}/lib")
