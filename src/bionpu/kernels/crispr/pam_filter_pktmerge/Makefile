##===- Makefile -----------------------------------------------------------===##
# — PacketFifo PAM-filter retrofit (C-M5-pktmerge).
#
# Builds the pktMerge variant xclbin from a single Python source +
# C++ object. Same shape as 's Makefile but only one variant
# (the pktmerge retrofit) per the task spec.
#
# Targets:
#   build/final.xclbin               pktMerge xclbin
#   build/insts.bin                  instruction stream
#   crispr_pam_filter_pktmerge       host C++ runner
#
# Invocation (from this directory):
#   source /opt/xilinx/xrt/setup.sh
#   source ~/xdna-bringup/ironenv/bin/activate
#   export MLIR_AIE_DIR=~/xdna-bringup/mlir-aie
#   export PEANO_INSTALL_DIR=~/xdna-bringup/ironenv/lib/python3.14/site-packages/llvm-aie
#   make NPU2=1 all
#
# `NPU2=1` is required (AIE2P).
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
targetname = crispr_pam_filter_pktmerge
aie_py_src = pam_filter_pktmerge.py
aie_cc_src = tile_a_pktmerge.cc
runner_cc_src = runner.cpp
peano_extra_flags ?= $(PEANO_EXTRA_FLAGS)
direct_stream_arg = $(if $(filter 1,$(DIRECT_STREAM)),--direct-stream,)
compact_packets_arg = $(if $(filter 1,$(COMPACT_PACKETS)),--compact-packets,)
fork_python = ${mlir_aie_dir}/install/python
repo_root := $(realpath ${srcdir}/../../../../)
PYTHON ?= python3
AIECC ?= ${mlir_aie_dir}/install/bin/aiecc

all: build/final.xclbin build/insts.bin ${targetname}

# AIE compute-kernel object — peano clang++ targeting aie2p.
build/tile_a_pktmerge.o: ${aie_cc_src}
	mkdir -p ${@D}
ifeq ($(devicename),npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} ${peano_extra_flags} -c $(srcdir)/${aie_cc_src} -o ${@F}
else ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} ${peano_extra_flags} -c $(srcdir)/${aie_cc_src} -o ${@F}
else
	@echo "Device type $(devicename) not supported; pass NPU2=1"
	@false
endif

# IRON lowering output → MLIR.
build/aie.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	PYTHONPATH=${fork_python}:${repo_root}:$${PYTHONPATH} ${PYTHON} $< -d ${devicename} ${direct_stream_arg} ${compact_packets_arg} > $@

# xclbin + instruction stream from aiecc.
build/final.xclbin: build/aie.mlir build/tile_a_pktmerge.o
	mkdir -p ${@D}
	cd ${@D} && PATH=${PEANO_INSTALL_DIR}/bin:$${PATH} ${AIECC} \
	    --aie-generate-xclbin --no-compile-host \
	    --xclbin-name=$(@F) \
	    --no-xchesscc --no-xbridge \
	    --aie-generate-npu-insts --npu-insts-name=insts.bin aie.mlir

build/insts.bin: build/final.xclbin
	@test -f $@ || (echo "ERROR: aiecc did not produce $@"; exit 1)

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
