##===- Makefile -----------------------------------------------------------===##
#
# Build the AIE2P xclbin for the Dorado fast linear projection.
#
# Three experiments are supported (additive — the default keeps the
# existing per-group fp32 artifact byte-identical):
#
#   experiment=production       (default) — per-group fp32, 1336 calls/dispatch
#   experiment=fused-perts      — DESIGN-fusion.md stage 3, bf16 weights,
#                                 334 calls/dispatch (one per timestep)
#   experiment=fused-dispatch   — DESIGN-fusion.md stage 4, bf16 weights,
#                                 single-kernel-call-per-dispatch entry
#                                 symbol (collapses to stage-3 MLIR shape
#                                 due to IRON Kernel ABI; see
#                                 linear_projection_fused_dispatch.py
#                                 fallback rationale)
#   experiment=fused-batched    — Batched stage-3: process N_CHUNKS chunks
#                                 per silicon dispatch (single xclbin load,
#                                 single weight slab, IRON loops chunks).
#                                 Knob: BIONPU_LINEAR_PROJECTION_BATCH_CHUNKS
#                                 (default 1; supported {1, 2, 4, 8}).
#                                 Reuses linear_projection_fused.cc kernel.
#
# Build invocations:
#
#   make NPU2=1 seq=334 all                                                # production (default)
#   make NPU2=1 experiment=fused-perts seq=334 all                         # stage 3 fused
#   make NPU2=1 experiment=fused-dispatch seq=334 all                      # stage 4 fused-dispatch
#   BIONPU_LINEAR_PROJECTION_BATCH_CHUNKS=8 make NPU2=1 experiment=fused-batched seq=334 all
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
seq ?= 334
experiment ?= production

# experiment-aware naming + source selection. Default (production)
# preserves the existing artifact paths byte-for-byte.
ifeq ($(experiment),fused-perts)
  targetname = dorado_fast_linear_projection_fused_perts
  artifact_suffix = fused_perts_L${seq}
  aie_py_src = linear_projection_fused.py
  aie_cc_src = linear_projection_fused.cc
  aie_obj    = linear_projection_fused.o
  runner_cc_src = runner_fused.cpp
else ifeq ($(experiment),fused-dispatch)
  targetname = dorado_fast_linear_projection_fused_dispatch
  artifact_suffix = fused_dispatch_L${seq}
  aie_py_src = linear_projection_fused_dispatch.py
  aie_cc_src = linear_projection_fused_dispatch.cc
  aie_obj    = linear_projection_fused_dispatch.o
  # Wire format identical to stage 3 (bf16 input + bf16 weight slab +
  # bf16 output); reuse the stage-3 host runner.
  runner_cc_src = runner_fused.cpp
else ifeq ($(experiment),fused-batched)
  # Batched stage-3: process n_chunks chunks per silicon dispatch.
  # Reuses the stage-3 C++ kernel byte-for-byte (linear_projection_fused.cc
  # / linear_projection_fused.o); chunk-outer batching lives in IRON.
  # The artifact suffix bakes the batch size in so n=1, 2, 4, 8 builds
  # coexist side-by-side without clobbering.
  n_chunks ?= $(if $(BIONPU_LINEAR_PROJECTION_BATCH_CHUNKS),$(BIONPU_LINEAR_PROJECTION_BATCH_CHUNKS),1)
  targetname = dorado_fast_linear_projection_fused_batched
  artifact_suffix = fused_batched_n${n_chunks}_L${seq}
  aie_py_src = linear_projection_fused_batched.py
  aie_cc_src = linear_projection_fused.cc
  aie_obj    = linear_projection_fused.o
  runner_cc_src = runner_fused_batched.cpp
else ifeq ($(experiment),production)
  targetname = dorado_fast_linear_projection
  artifact_suffix = L${seq}
  aie_py_src = linear_projection.py
  aie_cc_src = linear_projection.cc
  aie_obj    = linear_projection.o
  runner_cc_src = runner.cpp
else
  $(error unknown experiment=$(experiment); supported: production, fused-perts, fused-dispatch, fused-batched)
endif

PYTHON ?= python3
AIE_API_INCLUDE ?= -I${MLIR_AIE_DIR}/install/include -I${MLIR_AIE_DIR}/third_party/aie_api/include
AIECC ?= $(shell ${PYTHON} -c "import pathlib, mlir_aie; print(pathlib.Path(mlir_aie.__file__).parent / 'bin' / 'aiecc')" 2>/dev/null || command -v aiecc 2>/dev/null || echo aiecc)

all: build/final_${artifact_suffix}.xclbin build/insts_${artifact_suffix}.bin ${targetname}_${artifact_suffix}

build/${aie_obj}: ${aie_cc_src}
	mkdir -p $(@D)
ifeq ($(devicename),npu2)
	cd $(@D) && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} \
	  -I${mlir_aie_dir}/aie_kernels \
	  ${AIE_API_INCLUDE} \
	  -c ${srcdir}/${aie_cc_src} -o ${aie_obj}
else ifeq ($(devicename),npu)
	cd $(@D) && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} \
	  -I${mlir_aie_dir}/aie_kernels \
	  ${AIE_API_INCLUDE} \
	  -c ${srcdir}/${aie_cc_src} -o ${aie_obj}
else
	$(error devicename=$(devicename) not supported; use NPU2=1)
endif

# Per-experiment extra args to the IRON-emit Python. Default empty;
# fused-batched needs the explicit chunk count so the env var is not
# the sole source of truth for the artifact's MLIR shape.
ifeq ($(experiment),fused-batched)
  py_extra_args = --n-chunks ${n_chunks}
else
  py_extra_args =
endif

build/aie_${artifact_suffix}.mlir: ${aie_py_src}
	mkdir -p $(@D)
	${PYTHON} ${srcdir}/${aie_py_src} -d ${devicename} -L ${seq} ${py_extra_args} > $@.tmp
	mv $@.tmp $@

build/final_${artifact_suffix}.xclbin: build/aie_${artifact_suffix}.mlir build/${aie_obj}
	mkdir -p $(@D)
	cd $(@D) && PATH=${PEANO_INSTALL_DIR}/bin:$${PATH} ${AIECC} \
	  --aie-generate-xclbin --no-compile-host \
	  --xclbin-name=$(@F) \
	  --no-xchesscc --no-xbridge \
	  --aie-generate-npu-insts --npu-insts-name=insts_${artifact_suffix}.bin \
	  $(<:%=../%)

build/insts_${artifact_suffix}.bin: build/final_${artifact_suffix}.xclbin
	@test -f $@ || (echo "expected $@ as a side effect of the xclbin build" && exit 1)

HOST_CXX ?= g++-15

${targetname}_${artifact_suffix}: ${runner_cc_src} build/final_${artifact_suffix}.xclbin
	@if [ -z "$$XILINX_XRT" ]; then \
	  echo "XILINX_XRT not set; source /opt/xilinx/xrt/setup.sh first" >&2; \
	  exit 1; \
	fi
	${HOST_CXX} -std=c++17 -O2 \
	  -I${XILINX_XRT}/include \
	  ${runner_cc_src} \
	  -L${XILINX_XRT}/lib -Wl,-rpath,${XILINX_XRT}/lib \
	  -lxrt_coreutil -luuid \
	  -o ${targetname}_${artifact_suffix}

clean:
	rm -rf build \
	  dorado_fast_linear_projection_L* \
	  dorado_fast_linear_projection_fused_perts_L* \
	  dorado_fast_linear_projection_fused_dispatch_L* \
	  dorado_fast_linear_projection_fused_batched_fused_batched_n*_L*

.PHONY: all clean
