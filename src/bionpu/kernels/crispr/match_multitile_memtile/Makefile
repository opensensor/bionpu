##===- Makefile -----------------------------------------------------------===##
# — memtile-aggregated 4-into-1 multi-tile CRISPR match kernel.
#
# **AM020 cross-walk follow-up to .** Recovers the original PRD §4.2
# 4-tile fan-in sketch via memtile-mediated aggregation (AM020 Ch. 5 p. 74).
#
# Builds:
#   - build/match_kernel.o          (AIE2P object — single
#                                     `crispr_match_memtile_match` symbol;
#                                     no joiner kernel needed because the
#                                     join is fabric-side via memtile DMA).
#   - build/aie.mlir                 (IRON Python lowering output)
#   - build/final.xclbin             (xclbin for AIE2P — 4 match tiles +
#                                     1 memtile, 1 column).
#   - build/insts.bin                (NPU instruction stream)
#   - crispr_match_multitile_memtile (host C++ runner; via direct g++-15)
#
# Invocation (from this directory):
#   source /opt/xilinx/xrt/setup.sh
#   source ~/xdna-bringup/ironenv/bin/activate
#   export MLIR_AIE_DIR=~/xdna-bringup/mlir-aie
#   export PEANO_INSTALL_DIR=~/xdna-bringup/ironenv/lib/python3.11/site-packages/llvm-aie
#   make NPU2=1 all
#
# `NPU2=1` is required (AIE2P). Without it the toolchain silently builds
# for AIE2 (older NPU1) and produces garbage at runtime — the same trap
# T0.1's bring-up canary documents and / 's Makefiles guard.
#
# Why g++-15 for the host runner: XRT's libxrt_coreutil.so depends on
# CXXABI_1.3.15, which g++-13's bundled libstdc++ does not export.
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
targetname = crispr_match_multitile_memtile
aie_py_src = multitile_memtile.py
aie_cc_src = match_kernel.cc
runner_cc_src = runner.cpp

all: build/final.xclbin build/insts.bin ${targetname}

# AIE compute-kernel object — peano clang++ targeting aie2p.
# Single symbol (`crispr_match_memtile_match`) shared across all 4 match
# tiles; aiecc links the same .o into each compute tile via the
# `link_with = "match_kernel.o"` attribute on the Kernel handle in
# multitile_memtile.py.
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

# Host runner — direct g++-15 build.
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
