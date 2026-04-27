##===- Makefile -----------------------------------------------------------===##
#
# Build the AIE2P xclbin for the bf16 Dorado fast LSTM cell.
#
# Mirrors 's lstm_cell/Makefile. NPU2=1 is required (bf16 vector
# tanh is AIE2P-only).
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
targetname = dorado_fast_lstm_cell_bf16
seq ?= 334

aie_py_src = lstm_cell_bf16.py
aie_cc_src = lstm_cell_bf16.cc
runner_cc_src = runner.cpp

all: build/final_L${seq}.xclbin build/insts_L${seq}.bin ${targetname}_L${seq}

build/lstm_cell_bf16.o: ${aie_cc_src}
	mkdir -p $(@D)
ifeq ($(devicename),npu2)
	cd $(@D) && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} \
	  -I${mlir_aie_dir}/aie_kernels \
	  -c ${srcdir}/${aie_cc_src} -o lstm_cell_bf16.o
else
	$(error devicename=$(devicename) not supported; bf16 hardware tanh is AIE2P-only — pass NPU2=1)
endif

build/aie_L${seq}.mlir: ${aie_py_src}
	mkdir -p $(@D)
	python3 ${srcdir}/${aie_py_src} -d ${devicename} -L ${seq} > $@

build/final_L${seq}.xclbin: build/aie_L${seq}.mlir build/lstm_cell_bf16.o
	mkdir -p $(@D)
	cd $(@D) && aiecc --aie-generate-xclbin --no-compile-host \
	  --xclbin-name=$(@F) \
	  --no-xchesscc --no-xbridge \
	  --aie-generate-npu-insts --npu-insts-name=insts_L${seq}.bin \
	  $(<:%=../%)

build/insts_L${seq}.bin: build/final_L${seq}.xclbin
	@test -f $@ || (echo "expected $@ as a side effect of the xclbin build" && exit 1)

HOST_CXX ?= g++-15

${targetname}_L${seq}: ${runner_cc_src} build/final_L${seq}.xclbin
	@if [ -z "$$XILINX_XRT" ]; then \
	  echo "XILINX_XRT not set; source /opt/xilinx/xrt/setup.sh first" >&2; \
	  exit 1; \
	fi
	${HOST_CXX} -std=c++17 -O2 \
	  -I${XILINX_XRT}/include \
	  ${runner_cc_src} \
	  -L${XILINX_XRT}/lib -Wl,-rpath,${XILINX_XRT}/lib \
	  -lxrt_coreutil -luuid \
	  -o ${targetname}_L${seq}

clean:
	rm -rf build ${targetname}_L*

.PHONY: all clean
