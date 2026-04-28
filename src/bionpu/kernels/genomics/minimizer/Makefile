##===- Makefile -----------------------------------------------------------===##
# Build the AIE2P xclbin for the bionpu (w, k) minimizer kernel (v0).
#
# v0 BUILD-TIME parameter list:
#   K, W       — pinned (k=15, w=10) or (k=21, w=11)
#   experiment ∈ {production, wide2, wide4, wide8} → n_tiles ∈ {1, 2, 4, 8}
#   seq        — fixture length tag (artifact suffix only)
#
# v0 produces ONE xclbin/insts pair per (k, w, n_tiles) cell — single pass.
# Build invocations (from this directory):
#
#   source /opt/xilinx/xrt/setup.sh
#   source /home/matteius/xdna-bringup/ironenv/bin/activate
#   export MLIR_AIE_DIR=/home/matteius/genetics/third_party/mlir-aie
#   export PEANO_INSTALL_DIR=/home/matteius/xdna-bringup/ironenv/lib/python3.14/site-packages/llvm-aie
#
#   make NPU2=1 K=15 W=10 experiment=wide4 seq=10000 all
#   make NPU2=1 K=21 W=11 experiment=wide4 seq=10000 all
#
# `NPU2=1` is required (AIE2P).
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
mlir_aie_dir ?= $(MLIR_AIE_DIR)

ifneq ($(MAKECMDGOALS),clean)

ifeq ($(strip $(mlir_aie_dir)),)
$(error MLIR_AIE_DIR is unset; source the bring-up env first)
endif

include ${mlir_aie_dir}/programming_examples/makefile-common

ifneq ($(NPU2),1)
$(error NPU2=1 is required (AIE2P); pass NPU2=1 on the make line)
endif

endif  # MAKECMDGOALS != clean

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
seq ?= 10000
experiment ?= production
K ?= 15
W ?= 10

# Sanity-check (K, W).
ifeq ($(K)_$(W),15_10)
  KW_ACTIVE = 1510
else ifeq ($(K)_$(W),21_11)
  KW_ACTIVE = 2111
else
  $(error unsupported (K=$(K), W=$(W)); pinned: (15,10) or (21,11))
endif

ifeq ($(experiment),production)
  n_tiles = 1
  launch_chunks = 1
else ifeq ($(experiment),wide2)
  n_tiles = 2
  launch_chunks = 2
else ifeq ($(experiment),wide4)
  n_tiles = 4
  launch_chunks = 4
else ifeq ($(experiment),wide8)
  n_tiles = 8
  launch_chunks = 8
else
  $(error unknown experiment=$(experiment); supported: production, wide2, wide4, wide8)
endif

targetname = bionpu_minimizer

artifact_suffix = k${K}_w${W}_n${n_tiles}_L${seq}
kw_obj_suffix = k${K}_w${W}

aie_py_src        = minimizer.py
aie_tile_cc_src   = minimizer_tile.cc
aie_tile_obj      = minimizer_tile_${kw_obj_suffix}.o
runner_cc_src     = runner.cpp

PYTHON ?= python3
AIE_API_INCLUDE ?= -I${MLIR_AIE_DIR}/install/include -I${MLIR_AIE_DIR}/third_party/aie_api/include
AIECC ?= $(shell ${PYTHON} -c "import pathlib, mlir_aie; print(pathlib.Path(mlir_aie.__file__).parent / 'bin' / 'aiecc')" 2>/dev/null || command -v aiecc 2>/dev/null || echo aiecc)

all: build/final_${artifact_suffix}.xclbin build/insts_${artifact_suffix}.bin ${targetname}

# Per-(K, W) compute-tile .o.
build/${aie_tile_obj}: ${aie_tile_cc_src} ${srcdir}/minimizer_constants.h
	mkdir -p $(@D)
ifeq ($(devicename),npu2)
	cd $(@D) && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} \
	  -DKW_ACTIVE=${KW_ACTIVE} \
	  -I${mlir_aie_dir}/aie_kernels \
	  ${AIE_API_INCLUDE} \
	  -c ${srcdir}/${aie_tile_cc_src} -o ${aie_tile_obj}
else
	$(error devicename=$(devicename) not supported; use NPU2=1)
endif

# IRON Python lowering → MLIR.
build/aie_${artifact_suffix}.mlir: ${aie_py_src}
	mkdir -p $(@D)
	BIONPU_MINIMIZER_LAUNCH_CHUNKS=${launch_chunks} \
	  BIONPU_MINIMIZER_K=${K} \
	  BIONPU_MINIMIZER_W=${W} \
	  ${PYTHON} ${srcdir}/${aie_py_src} \
	    -d ${devicename} \
	    --k ${K} \
	    --w ${W} \
	    --launch-chunks ${launch_chunks} \
	    > $@.tmp
	mv $@.tmp $@

# Per-(K, W, n_tiles) xclbin. The per-(K,W) tile object is aliased
# under the canonical name `minimizer_tile.o` so aiecc finds it.
build/final_${artifact_suffix}.xclbin: build/aie_${artifact_suffix}.mlir build/${aie_tile_obj}
	mkdir -p $(@D)
	cp build/${aie_tile_obj} build/minimizer_tile.o
	cd $(@D) && PATH=${PEANO_INSTALL_DIR}/bin:$${PATH} ${AIECC} \
	  --aie-generate-xclbin --no-compile-host \
	  --xclbin-name=$(@F) \
	  --no-xchesscc --no-xbridge \
	  --aie-generate-npu-insts --npu-insts-name=insts_${artifact_suffix}.bin \
	  aie_${artifact_suffix}.mlir

build/insts_${artifact_suffix}.bin: build/final_${artifact_suffix}.xclbin
	@test -f $@ || (echo "expected $@ as a side effect of the xclbin build" && exit 1)

HOST_CXX ?= g++-15

${targetname}: ${runner_cc_src}
	@if [ -z "$$XILINX_XRT" ]; then \
	  echo "XILINX_XRT not set; source /opt/xilinx/xrt/setup.sh first" >&2; \
	  exit 1; \
	fi
	${HOST_CXX} -std=c++17 -O2 \
	  -I${XILINX_XRT}/include \
	  ${runner_cc_src} \
	  -L${XILINX_XRT}/lib -Wl,-rpath,${XILINX_XRT}/lib \
	  -lxrt_coreutil -luuid -ltbb \
	  -o ${targetname}

clean:
	rm -rf build ${targetname}

.PHONY: all clean
