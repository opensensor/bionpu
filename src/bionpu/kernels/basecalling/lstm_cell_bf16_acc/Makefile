##===- Makefile -----------------------------------------------------------===##
#
# Build the AIE2P xclbin for the mixed-precision Dorado fast LSTM cell
#.
#
# Mirrors 's lstm_cell_bf16/Makefile. NPU2=1 is required (bf16
# vector tanh + accfloat<->bfloat16 narrowing are AIE2P-only).
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
targetname = dorado_fast_lstm_cell_bf16_acc
seq ?= 334

aie_py_src = lstm_cell_bf16_acc.py
aie_cc_src = lstm_cell_bf16_acc.cc
runner_cc_src = runner.cpp
stack_runner_cc_src = stack_runner.cpp
PYTHON ?= python3
AIE_API_INCLUDE ?= -I${MLIR_AIE_DIR}/install/include -I${MLIR_AIE_DIR}/third_party/aie_api/include
AIECC ?= ${MLIR_AIE_DIR}/install/bin/aiecc

all: build/final_L${seq}.xclbin build/insts_L${seq}.bin ${targetname}_L${seq} ${targetname}_stack_L${seq}

build/lstm_cell_bf16_acc.o: ${aie_cc_src}
	mkdir -p $(@D)
ifeq ($(devicename),npu2)
	cd $(@D) && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} \
	  -I${mlir_aie_dir}/aie_kernels \
	  ${AIE_API_INCLUDE} \
	  -c ${srcdir}/${aie_cc_src} -o lstm_cell_bf16_acc.o
else
	$(error devicename=$(devicename) not supported; bf16 hardware tanh is AIE2P-only — pass NPU2=1)
endif

build/aie_L${seq}.mlir: ${aie_py_src}
	mkdir -p $(@D)
	${PYTHON} ${srcdir}/${aie_py_src} -d ${devicename} -L ${seq} > $@

build/final_L${seq}.xclbin: build/aie_L${seq}.mlir build/lstm_cell_bf16_acc.o
	mkdir -p $(@D)
	cd $(@D) && PATH=${PEANO_INSTALL_DIR}/bin:$${PATH} ${AIECC} \
	  --aie-generate-xclbin --no-compile-host \
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

${targetname}_stack_L${seq}: ${stack_runner_cc_src} build/final_L${seq}.xclbin
	@if [ -z "$$XILINX_XRT" ]; then \
	  echo "XILINX_XRT not set; source /opt/xilinx/xrt/setup.sh first" >&2; \
	  exit 1; \
	fi
	${HOST_CXX} -std=c++17 -O2 \
	  -I${XILINX_XRT}/include \
	  ${stack_runner_cc_src} \
	  -L${XILINX_XRT}/lib -Wl,-rpath,${XILINX_XRT}/lib \
	  -lxrt_coreutil -luuid \
	  -o ${targetname}_stack_L${seq}

install-artifact: all
	mkdir -p ${srcdir}/../../../dispatch/_npu_artifacts/${targetname}_L${seq}
	cp build/final_L${seq}.xclbin \
	  ${srcdir}/../../../dispatch/_npu_artifacts/${targetname}_L${seq}/final.xclbin
	cp build/insts_L${seq}.bin \
	  ${srcdir}/../../../dispatch/_npu_artifacts/${targetname}_L${seq}/insts.bin
	cp ${targetname}_L${seq} \
	  ${srcdir}/../../../dispatch/_npu_artifacts/${targetname}_L${seq}/host_runner
	cp ${targetname}_stack_L${seq} \
	  ${srcdir}/../../../dispatch/_npu_artifacts/${targetname}_L${seq}/stack_host_runner
	printf '%s\n' \
	  '# ${targetname}_L${seq}' \
	  '' \
	  'Production-shape Dorado fast mixed-state LSTM cell artifact.' \
	  'Includes stack_host_runner for one-process five-layer LSTM batching.' \
	  '' \
	  '- seq: ${seq}' \
	  '- input shape: (${seq}, 1, 96)' \
	  '- output shape: (${seq}, 1, 96)' \
	  '- build: make NPU2=1 seq=${seq} install-artifact' \
	  > ${srcdir}/../../../dispatch/_npu_artifacts/${targetname}_L${seq}/MANIFEST.md

clean:
	rm -rf build ${targetname}_L* ${targetname}_stack_L*

.PHONY: all clean install-artifact
