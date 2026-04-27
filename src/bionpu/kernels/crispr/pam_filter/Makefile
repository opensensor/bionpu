##===- Makefile -----------------------------------------------------------===##
# — PAM filter + threshold + sparse-emit CRISPR kernel (C-M5).
#
# Builds TWO xclbin variants (filter-early + filter-late) from the same
# Python source + same C++ object. The variants share `tile_a_filter.o`
# but pull in different Tile-A / Tile-Z symbols via the `Kernel(symbol,…)`
# references in pam_filter.py.
#
# Targets:
#   build/early/final.xclbin         filter-early xclbin (production)
#   build/early/insts.bin            instruction stream for early
#   build/late/final.xclbin          filter-late xclbin (comparison)
#   build/late/insts.bin             instruction stream for late
#   crispr_pam_filter                host C++ runner (one binary; takes
#                                     either xclbin via --xclbin)
#
# Invocation (from this directory):
#   source /opt/xilinx/xrt/setup.sh
#   source ~/xdna-bringup/ironenv/bin/activate
#   export MLIR_AIE_DIR=~/xdna-bringup/mlir-aie
#   export PEANO_INSTALL_DIR=~/xdna-bringup/ironenv/lib/python3.11/site-packages/llvm-aie
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
targetname = crispr_pam_filter
aie_py_src = pam_filter.py
aie_cc_src = tile_a_filter.cc
runner_cc_src = runner.cpp

all: build/early/final.xclbin build/early/insts.bin \
     build/late/final.xclbin build/late/insts.bin \
     ${targetname}

# AIE compute-kernel object — peano clang++ targeting aie2p. The same .o
# exposes ALL Tile-A and Tile-Z symbols (early + late) plus the verbatim
# match-tile entry; aiecc resolves per-tile.
#
# (T3, vectorisation half): the match-tile body now uses
# ::aie::vector intrinsics (#include <aie_api/aie.hpp>). The aie_api
# headers ship at <mlir-aie>/third_party/aie_api/include — not always
# symlinked into ${MLIR_AIE_DIR}/include — so we add an explicit
# fallback include path. Both forms (install/include/aie_api and
# third_party/aie_api/include/aie_api) are searched; the first one
# present wins.
AIE_API_INCLUDE ?= -I${MLIR_AIE_DIR}/install/include -I${MLIR_AIE_DIR}/third_party/aie_api/include
# Optional: add `BIONPU_FORCE_SCALAR=1` on the make line to compile the
# scalar reference variant of the match kernel ( head-to-head
# baseline). Empty by default.
EXTRA_KERNEL_FLAGS ?=
ifeq ($(BIONPU_FORCE_SCALAR),1)
EXTRA_KERNEL_FLAGS += -DBIONPU_FORCE_SCALAR
endif

build/tile_a_filter.o: ${aie_cc_src}
	mkdir -p ${@D}
ifeq ($(devicename),npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} ${AIE_API_INCLUDE} ${EXTRA_KERNEL_FLAGS} -c $(srcdir)/${aie_cc_src} -o ${@F}
else ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} ${AIE_API_INCLUDE} ${EXTRA_KERNEL_FLAGS} -c $(srcdir)/${aie_cc_src} -o ${@F}
else
	@echo "Device type $(devicename) not supported; pass NPU2=1"
	@false
endif

# IRON lowering output → MLIR (per variant).
build/early/aie.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< -d ${devicename} --mode early > $@

build/late/aie.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< -d ${devicename} --mode late > $@

# xclbin + instruction stream from aiecc (per variant).
# Each variant needs its own copy of tile_a_filter.o in its build dir
# because aiecc reads link inputs from the cwd.
build/early/final.xclbin: build/early/aie.mlir build/tile_a_filter.o
	mkdir -p ${@D}
	cp build/tile_a_filter.o build/early/tile_a_filter.o
	cd ${@D} && aiecc --aie-generate-xclbin --no-compile-host \
	    --xclbin-name=$(@F) \
	    --no-xchesscc --no-xbridge \
	    --aie-generate-npu-insts --npu-insts-name=insts.bin aie.mlir

build/late/final.xclbin: build/late/aie.mlir build/tile_a_filter.o
	mkdir -p ${@D}
	cp build/tile_a_filter.o build/late/tile_a_filter.o
	cd ${@D} && aiecc --aie-generate-xclbin --no-compile-host \
	    --xclbin-name=$(@F) \
	    --no-xchesscc --no-xbridge \
	    --aie-generate-npu-insts --npu-insts-name=insts.bin aie.mlir

build/early/insts.bin: build/early/final.xclbin
	@test -f $@ || (echo "ERROR: aiecc did not produce $@"; exit 1)

build/late/insts.bin: build/late/final.xclbin
	@test -f $@ || (echo "ERROR: aiecc did not produce $@"; exit 1)

HOST_CXX ?= g++-15

${targetname}: ${runner_cc_src} build/early/final.xclbin
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

run: ${targetname} build/early/final.xclbin build/early/insts.bin
	@echo "(Set --guides / --windows / --out paths as needed; see runner.cpp.)"

clean:
	rm -rf build ${targetname}

.PHONY: all clean run
