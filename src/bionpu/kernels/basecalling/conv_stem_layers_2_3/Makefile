##===- Makefile -----------------------------------------------------------===##
#
# Build AIE2P xclbins for Dorado fast Conv stem layers 2 + 3.
#
# Mirrors the structure of 's `conv_stem/Makefile`. NPU2=1 is required.
# Two targets, parameterised on layer number:
#
#   make NPU2=1 layer=2   ->  build/final_T2000_l2.xclbin, runner_l2
#   make NPU2=1 layer=3   ->  build/final_T2000_l3.xclbin, runner_l3
#
# Required env (from CLAUDE.md / task brief):
#   source /opt/xilinx/xrt/setup.sh
#   source ~/xdna-bringup/ironenv/bin/activate
#   export MLIR_AIE_DIR=~/xdna-bringup/mlir-aie
#   export PEANO_INSTALL_DIR=~/xdna-bringup/ironenv/lib/python3.11/site-packages/llvm-aie
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
layer ?= 2
time ?= 2000
chunk ?= 200
in_time ?= 2018
out_time ?= 334

ifeq ($(layer),2)
  targetname = dorado_fast_conv_stem_layer2
  aie_py_src = conv1d_layer2.py
  aie_cc_src = conv1d_layer2.cc
  obj_name   = conv1d_layer2.o
  py_args    = -t ${time} -c ${chunk}
  build_tag  = T${time}_l2
else ifeq ($(layer),3)
  targetname = dorado_fast_conv_stem_layer3
  aie_py_src = conv1d_layer3.py
  aie_cc_src = conv1d_layer3.cc
  obj_name   = conv1d_layer3.o
  py_args    = --in-time ${in_time} --out-time ${out_time}
  build_tag  = T${time}_l3
else
  $(error layer must be 2 or 3)
endif

runner_cc_src = runner.cpp

all: build/final_${build_tag}.xclbin build/insts_${build_tag}.bin ${targetname}_${build_tag}

# AIE-tile kernel object
build/${obj_name}: ${aie_cc_src}
	mkdir -p $(@D)
ifeq ($(devicename),npu2)
	cd $(@D) && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} \
	  -I${mlir_aie_dir}/aie_kernels \
	  -c ${srcdir}/${aie_cc_src} -o ${obj_name}
else ifeq ($(devicename),npu)
	cd $(@D) && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} \
	  -I${mlir_aie_dir}/aie_kernels \
	  -c ${srcdir}/${aie_cc_src} -o ${obj_name}
else
	$(error devicename=$(devicename) not supported; use NPU2=1)
endif

# IRON Python -> MLIR module
build/aie_${build_tag}.mlir: ${aie_py_src}
	mkdir -p $(@D)
	python3 ${srcdir}/${aie_py_src} -d ${devicename} ${py_args} > $@

# Final xclbin + insts via aiecc.
build/final_${build_tag}.xclbin: build/aie_${build_tag}.mlir build/${obj_name}
	mkdir -p $(@D)
	cd $(@D) && aiecc --aie-generate-xclbin --no-compile-host \
	  --xclbin-name=$(@F) \
	  --no-xchesscc --no-xbridge \
	  --aie-generate-npu-insts --npu-insts-name=insts_${build_tag}.bin \
	  $(<:%=../%)

build/insts_${build_tag}.bin: build/final_${build_tag}.xclbin
	@test -f $@ || (echo "expected $@ as a side effect of the xclbin build" && exit 1)

HOST_CXX ?= g++-15

${targetname}_${build_tag}: ${runner_cc_src} build/final_${build_tag}.xclbin
	@if [ -z "$$XILINX_XRT" ]; then \
	  echo "XILINX_XRT not set; source /opt/xilinx/xrt/setup.sh first" >&2; \
	  exit 1; \
	fi
	${HOST_CXX} -std=c++17 -O2 \
	  -I${XILINX_XRT}/include \
	  ${runner_cc_src} \
	  -L${XILINX_XRT}/lib -Wl,-rpath,${XILINX_XRT}/lib \
	  -lxrt_coreutil -luuid \
	  -o ${targetname}_${build_tag}

clean:
	rm -rf build dorado_fast_conv_stem_layer*_T*

.PHONY: all clean
