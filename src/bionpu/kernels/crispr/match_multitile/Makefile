##===- Makefile -----------------------------------------------------------===##
# — multi-tile streaming CRISPR match kernel (C-M4).
#
# Builds:
#   - build/match_kernel.o          (AIE2P object — both `match` and `join`
#                                     entry points; aiecc links into all 5
#                                     compute tiles via the per-Kernel
#                                     `link_with` attribute).
#   - build/aie.mlir                 (IRON Python lowering output)
#   - build/final.xclbin             (xclbin for AIE2P — 5 compute tiles,
#                                     1 column).
#   - build/insts.bin                (NPU instruction stream)
#   - crispr_match_multitile         (host C++ runner; via direct g++-15)
#
# Invocation (from this directory):
#   source /opt/xilinx/xrt/setup.sh
#   source <ironenv>/bin/activate
#   export MLIR_AIE_DIR=<xdna-bringup>/mlir-aie
#   export PEANO_INSTALL_DIR=<ironenv>/lib/python3.11/site-packages/llvm-aie
#   make NPU2=1 all
#
# `NPU2=1` is required (AIE2P). Without it the toolchain silently builds
# for AIE2 (older NPU1) and produces garbage at runtime — the same trap
# T0.1's bring-up canary documents and 's Makefile guards.
#
# Why g++-15 for the host runner: XRT's libxrt_coreutil.so depends on
# CXXABI_1.3.15, which g++-13's bundled libstdc++ does not export. 's
# Makefile already documented this; 's runner uses the same CXX.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
mlir_aie_dir ?= $(MLIR_AIE_DIR)

ifeq ($(strip $(mlir_aie_dir)),)
$(error MLIR_AIE_DIR is unset; source the bring-up env first)
endif

include ${mlir_aie_dir}/programming_examples/makefile-common

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
targetname = crispr_match_multitile
aie_py_src = multitile.py
aie_cc_src = match_kernel.cc
runner_cc_src = runner.cpp

all: build/final.xclbin build/insts.bin ${targetname}

# AIE compute-kernel object — peano clang++ targeting aie2p.
# The same .o exposes BOTH `crispr_match_multitile_match` and
# `crispr_match_multitile_join` symbols; aiecc picks the right entry point
# per-tile from the IRON-generated MLIR.
build/match_kernel.o: ${aie_cc_src}
	mkdir -p ${@D}
ifeq ($(devicename),npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -c $(srcdir)/${aie_cc_src} -o ${@F}
else ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -c $(srcdir)/${aie_cc_src} -o ${@F}
else
	@echo "Device type $(devicename) not supported; pass NPU2=1"
	@false
endif

# IRON lowering output → MLIR.
build/aie.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< -d ${devicename} > $@

# xclbin + instruction stream from aiecc.
build/final.xclbin: build/aie.mlir build/match_kernel.o
	mkdir -p ${@D}
	cd ${@D} && aiecc --aie-generate-xclbin --no-compile-host \
	    --xclbin-name=$(@F) \
	    --no-xchesscc --no-xbridge \
	    --aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%)

build/insts.bin: build/final.xclbin
	@test -f $@ || (echo "ERROR: aiecc did not produce $@"; exit 1)

# Host runner — direct g++-15 build (matches 's pattern, not 's
# CMake/test_utils path; we don't need the test_utils helpers since this
# runner is purpose-built for caller-driven file I/O).
HOST_CXX ?= g++-15

${targetname}: ${runner_cc_src} build/final.xclbin
	@if [ -z "$$XILINX_XRT" ]; then \
	  echo "XILINX_XRT not set; source /opt/xilinx/xrt/setup.sh first" >&2; \
	  exit 1; \
	fi
	${HOST_CXX} -std=c++17 -O2 \
	  -I${XILINX_XRT}/include \
	  ${runner_cc_src} \
	  -L${XILINX_XRT}/lib -Wl,-rpath,${XILINX_XRT}/lib \
	  -lxrt_coreutil -luuid \
	  -o ${targetname}

run: ${targetname} build/final.xclbin build/insts.bin
	@echo "(Set --guides / --windows / --out paths as needed; see runner.cpp.)"

clean:
	rm -rf build ${targetname}

.PHONY: all clean run
