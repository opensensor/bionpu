# Optional CMake helper for the bf16 LSTM cell host runner.
# Kept minimal — the canonical build path is `make NPU2=1 all`. CMake
# is shipped per the task's manifest list so IDE / clangd can pick up
# include paths without parsing the Makefile. No xclbin generation here;
# only the host runner C++ binary.

cmake_minimum_required(VERSION 3.20)
project(dorado_fast_lstm_cell_bf16 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED ENV{XILINX_XRT})
    message(FATAL_ERROR "XILINX_XRT not set; source /opt/xilinx/xrt/setup.sh first")
endif()

set(XRT_ROOT $ENV{XILINX_XRT})

add_executable(dorado_fast_lstm_cell_bf16_host runner.cpp)
target_include_directories(dorado_fast_lstm_cell_bf16_host PRIVATE
    ${XRT_ROOT}/include
)
target_link_directories(dorado_fast_lstm_cell_bf16_host PRIVATE
    ${XRT_ROOT}/lib
)
target_link_libraries(dorado_fast_lstm_cell_bf16_host PRIVATE
    xrt_coreutil
    uuid
)
target_link_options(dorado_fast_lstm_cell_bf16_host PRIVATE
    -Wl,-rpath,${XRT_ROOT}/lib
)
target_compile_options(dorado_fast_lstm_cell_bf16_host PRIVATE -O2)
