# CMakeLists.txt — host runner for crispr_match_singletile.
#
# Mirrors the mlir-aie programming_examples pattern. We delegate XRT and
# test_utils discovery to the repo's `common.cmake` (path computed via
# MLIR_AIE_DIR env var; populated by the bring-up env).
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.30)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

project(_pre_xrt_init LANGUAGES C CXX)

# Pull in mlir-aie's common.cmake so we get the same XRT detection +
# test_utils target the bringup-tested examples use.
if(DEFINED ENV{MLIR_AIE_DIR})
    include($ENV{MLIR_AIE_DIR}/programming_examples/common.cmake)
else()
    message(FATAL_ERROR "MLIR_AIE_DIR not set; source the bringup env.")
endif()

if(NOT DEFINED CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER gcc-13)
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER g++-13)
endif()

set(TARGET_NAME crispr_match_singletile CACHE STRING "Target to be built")
project(${TARGET_NAME})

add_executable(${TARGET_NAME} test.cpp)

target_include_directories(${TARGET_NAME} PUBLIC ${XRT_INC_DIR})
target_link_directories(${TARGET_NAME} PUBLIC ${XRT_LIB_DIR})
target_link_libraries(${TARGET_NAME} PUBLIC xrt_coreutil)

target_link_test_utils(${TARGET_NAME})
