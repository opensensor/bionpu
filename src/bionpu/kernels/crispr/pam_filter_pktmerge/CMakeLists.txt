# Optional CMake target for the host runner (mirrors pam_filter/).
# Default build path is the Makefile (matches / ); CMake here is
# kept for IDE integration only.
cmake_minimum_required(VERSION 3.20)
project(crispr_pam_filter_pktmerge CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ENV{XILINX_XRT})
    message(FATAL_ERROR "XILINX_XRT not set; source /opt/xilinx/xrt/setup.sh first")
endif()

set(XRT_ROOT $ENV{XILINX_XRT})

add_executable(crispr_pam_filter_pktmerge runner.cpp)
target_include_directories(crispr_pam_filter_pktmerge PRIVATE ${XRT_ROOT}/include)
target_link_directories(crispr_pam_filter_pktmerge PRIVATE ${XRT_ROOT}/lib)
target_link_libraries(crispr_pam_filter_pktmerge PRIVATE xrt_coreutil uuid)
set_target_properties(crispr_pam_filter_pktmerge PROPERTIES
    BUILD_RPATH ${XRT_ROOT}/lib
    INSTALL_RPATH ${XRT_ROOT}/lib
)
