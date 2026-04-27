##===- Makefile -----------------------------------------------------------===##
#
# Build the AIE2P xclbin for the first Dorado fast stem Conv1d.
#
# Mirrors the structure of mlir-aie/programming_examples/basic/
# vector_scalar_mul/Makefile. NPU2=1 is
# required: the kernel targets AIE2P (`-t aie2p` in the Peano flags).
# Without NPU2=1 the build picks aie2 flags and silently produces an
# xclbin for the wrong ISA (see T0.1 recheck script).
#
# Required env (from CLAUDE.md / task brief):
#   source /opt/xilinx/xrt/setup.sh
#   source <ironenv>/bin/activate
#   export MLIR_AIE_DIR=<xdna-bringup>/mlir-aie
#   export PEANO_INSTALL_DIR=<ironenv>/lib/python3.11/site-packages/llvm-aie
#
# Targets:
#   build/final_T${time}.xclbin   - vendored to bionpu/dispatch/_npu_artifacts
#   build/insts_T${time}.bin      - vendored to bionpu/dispatch/_npu_artifacts
#   conv_stem_runner_T${time}     - host runner (built from runner.cpp)
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
targetname = dorado_fast_conv_stem_layer1
time ?= 2000
chunk ?= 200

aie_py_src = conv1d_layer1.py
aie_cc_src = conv1d_layer1.cc
runner_cc_src = runner.cpp

all: build/final_T${time}.xclbin build/insts_T${time}.bin ${targetname}_T${time}

# AIE-tile kernel object. Match the vector_scalar_mul pattern: per-arch
# Peano flags chosen by `devicename`.
build/conv1d_layer1.o: ${aie_cc_src}
	mkdir -p $(@D)
ifeq ($(devicename),npu2)
	cd $(@D) && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} \
	  -I${mlir_aie_dir}/aie_kernels \
	  -c ${srcdir}/${aie_cc_src} -o conv1d_layer1.o
else ifeq ($(devicename),npu)
	cd $(@D) && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} \
	  -I${mlir_aie_dir}/aie_kernels \
	  -c ${srcdir}/${aie_cc_src} -o conv1d_layer1.o
else
	$(error devicename=$(devicename) not supported; use NPU2=1)
endif

# IRON Python -> MLIR module
build/aie_T${time}.mlir: ${aie_py_src}
	mkdir -p $(@D)
	python3 ${srcdir}/${aie_py_src} -d ${devicename} -t ${time} -c ${chunk} > $@

# Final xclbin + insts via aiecc.
build/final_T${time}.xclbin: build/aie_T${time}.mlir build/conv1d_layer1.o
	mkdir -p $(@D)
	cd $(@D) && aiecc --aie-generate-xclbin --no-compile-host \
	  --xclbin-name=$(@F) \
	  --no-xchesscc --no-xbridge \
	  --aie-generate-npu-insts --npu-insts-name=insts_T${time}.bin \
	  $(<:%=../%)

build/insts_T${time}.bin: build/final_T${time}.xclbin
	@test -f $@ || (echo "expected $@ as a side effect of the xclbin build" && exit 1)

# Host runner: a thin wrapper around the XRT C++ API. Data is passed via
# files (binary float32) so the Python backend can drive caller-supplied
# weights + signals without needing pyxrt (which only ships as a 3.14
# binding on this host while ironenv is 3.11).
#
# We require g++-15 because XRT's libxrt_coreutil.so depends on
# CXXABI_1.3.15, which g++-13's bundled libstdc++ does not export
# (verified on this host: `readelf -V /usr/lib/gcc/x86_64-linux-gnu/13/
# libstdc++.so` tops out at CXXABI_1.3.14, while the system
# `/usr/lib/x86_64-linux-gnu/libstdc++.so.6` has 1.3.15+ — g++-15 picks
# the latter at link time).
HOST_CXX ?= g++-15

${targetname}_T${time}: ${runner_cc_src} build/final_T${time}.xclbin
	@if [ -z "$$XILINX_XRT" ]; then \
	  echo "XILINX_XRT not set; source /opt/xilinx/xrt/setup.sh first" >&2; \
	  exit 1; \
	fi
	${HOST_CXX} -std=c++17 -O2 \
	  -I${XILINX_XRT}/include \
	  ${runner_cc_src} \
	  -L${XILINX_XRT}/lib -Wl,-rpath,${XILINX_XRT}/lib \
	  -lxrt_coreutil -luuid \
	  -o ${targetname}_T${time}

clean:
	rm -rf build ${targetname}_T*

.PHONY: all clean
