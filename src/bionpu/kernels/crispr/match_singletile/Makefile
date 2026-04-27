##===- Makefile -----------------------------------------------------------===##
# — single-tile CRISPR match kernel (C-M3).
#
# Builds:
#   - build/match_kernel.o          (AIE2P object compiled with peano clang++)
#   - build/aie.mlir                 (IRON Python lowering output)
#   - build/final.xclbin             (xclbin for AIE2P)
#   - build/insts.bin                (NPU instruction stream)
#   - crispr_match_singletile        (host C++ binary; via _build/cmake)
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
# documented in T0.1's bring-up canary scripts.
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Pull in the same flags + helpers the mlir-aie examples use. Requires
# MLIR_AIE_DIR to be set (bring-up env).
include ${MLIR_AIE_DIR}/programming_examples/makefile-common

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
targetname = crispr_match_singletile
aie_py_src = match_singletile.py

all: build/final.xclbin build/insts.bin ${targetname}

# AIE compute-kernel object — peano clang++ targeting aie2p (or aie2 on NPU1).
build/match_kernel.o: match_kernel.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -c $(srcdir)/match_kernel.cc -o ${@F}
else ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -c $(srcdir)/match_kernel.cc -o ${@F}
else
	@echo "Device type not supported"
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

# Host C++ binary.
${targetname}: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && cmake $(srcdir) -DTARGET_NAME=${targetname}
	cd _build && cmake --build . --config Release
	cp _build/${targetname} $@

run: ${targetname} build/final.xclbin build/insts.bin
	@echo "(Set --guides / --windows / --out paths as needed; see test.cpp.)"

clean:
	rm -rf build _build ${targetname}
