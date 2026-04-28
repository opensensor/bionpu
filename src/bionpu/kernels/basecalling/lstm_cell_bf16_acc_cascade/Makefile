##===- Makefile -----------------------------------------------------------===##
#
# Build the AIE2P xclbin for the 5-layer cascade-stream LSTM stack
#.
#
# Mirrors 's lstm_cell_bf16_acc/Makefile; only differences:
#   - Single C++ source produces 3 extern "C" entry points (one per
#     cascade role: put_only, put_get, get_only).
#   - IRON Python emit uses CascadeFifo + AccumFifo.
#   - Multi-tile xclbin (5 CoreTile workers in column 0, rows 2..6).
#
# NPU2=1 is required (bf16 vector tanh + accfloat<->bfloat16 narrowing
# are AIE2P-only; cascade-stream first-class IRON is the fork's surface).
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
# Post bionpu-public-submodule swap: kernel sits at
#   genetics/bionpu-public/src/bionpu/kernels/basecalling/lstm_cell_bf16_acc_cascade
# (6 levels into genetics). third_party/ remains at the genetics root.
repo_root ?= $(realpath ${srcdir}/../../../../../..)
fork_python ?= ${repo_root}/third_party/mlir-aie/install/python
mlir_aie_dir ?= $(MLIR_AIE_DIR)

ifeq ($(strip $(mlir_aie_dir)),)
$(error MLIR_AIE_DIR is unset; source the bring-up env first)
endif

include ${mlir_aie_dir}/programming_examples/makefile-common

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
targetname = dorado_fast_lstm_cell_bf16_acc_cascade
seq ?= 334
experiment ?= production
artifact_suffix = $(if $(filter hidden-state-only,$(experiment)),hidden_state_only_L${seq},L${seq})
artifact_suffix := $(if $(filter hidden-state-math,$(experiment)),hidden_state_math_L${seq},${artifact_suffix})
artifact_suffix := $(if $(filter hidden-state-weight,$(experiment)),hidden_state_weight_L${seq},${artifact_suffix})
artifact_suffix := $(if $(filter hidden-state-weight-chain,$(experiment)),hidden_state_weight_chain_L${seq},${artifact_suffix})
artifact_suffix := $(if $(filter production-weight-cache-chain,$(experiment)),production_weight_cache_chain_L${seq},${artifact_suffix})
artifact_suffix := $(if $(filter production-weight-stream-chain,$(experiment)),production_weight_stream_chain_L${seq},${artifact_suffix})
artifact_suffix := $(if $(filter production-weight-direct-stream,$(experiment)),production_weight_direct_stream_L${seq},${artifact_suffix})
artifact_suffix := $(if $(filter production-weight-direct-stream-probe,$(experiment)),production_weight_direct_stream_probe_L${seq},${artifact_suffix})
artifact_suffix := $(if $(filter production-weight-direct-layer-probe,$(experiment)),production_weight_direct_layer_probe_L${seq},${artifact_suffix})
artifact_suffix := $(if $(filter production-weight-direct-group-probe,$(experiment)),production_weight_direct_group_probe_L${seq},${artifact_suffix})
artifact_suffix := $(if $(filter production-weight-direct-group-math-probe,$(experiment)),production_weight_direct_group_math_probe_L${seq},${artifact_suffix})
artifact_suffix := $(if $(filter production-weight-direct-group-cascade-probe,$(experiment)),production_weight_direct_group_cascade_probe_L${seq},${artifact_suffix})
artifact_suffix := $(if $(filter production-weight-direct-group-vertical-chain-probe,$(experiment)),production_weight_direct_group_vertical_chain_probe_L${seq},${artifact_suffix})
artifact_suffix := $(if $(filter production-weight-consolidated-vertical-chain-probe,$(experiment)),production_weight_consolidated_vertical_chain_probe_L${seq},${artifact_suffix})
artifact_suffix := $(if $(filter production-weight-consolidated-horizontal-probe,$(experiment)),production_weight_consolidated_horizontal_probe_L${seq},${artifact_suffix})
artifact_suffix := $(if $(filter production-weight-consolidated-full-chain-probe,$(experiment)),production_weight_consolidated_full_chain_probe_L${seq},${artifact_suffix})
artifact_suffix := $(if $(filter production-weight-consolidated-full-chain-lstm-probe,$(experiment)),production_weight_consolidated_full_chain_lstm_probe_L${seq},${artifact_suffix})
experiment_arg = $(if $(filter hidden-state-only,$(experiment)),--experiment hidden-state-only,)
experiment_arg := $(if $(filter hidden-state-math,$(experiment)),--experiment hidden-state-math,${experiment_arg})
experiment_arg := $(if $(filter hidden-state-weight,$(experiment)),--experiment hidden-state-weight,${experiment_arg})
experiment_arg := $(if $(filter hidden-state-weight-chain,$(experiment)),--experiment hidden-state-weight-chain,${experiment_arg})
experiment_arg := $(if $(filter production-weight-cache-chain,$(experiment)),--experiment production-weight-cache-chain,${experiment_arg})
experiment_arg := $(if $(filter production-weight-stream-chain,$(experiment)),--experiment production-weight-stream-chain,${experiment_arg})
experiment_arg := $(if $(filter production-weight-direct-stream,$(experiment)),--experiment production-weight-direct-stream,${experiment_arg})
experiment_arg := $(if $(filter production-weight-direct-stream-probe,$(experiment)),--experiment production-weight-direct-stream-probe,${experiment_arg})
experiment_arg := $(if $(filter production-weight-direct-layer-probe,$(experiment)),--experiment production-weight-direct-layer-probe,${experiment_arg})
experiment_arg := $(if $(filter production-weight-direct-group-probe,$(experiment)),--experiment production-weight-direct-group-probe,${experiment_arg})
experiment_arg := $(if $(filter production-weight-direct-group-math-probe,$(experiment)),--experiment production-weight-direct-group-math-probe,${experiment_arg})
experiment_arg := $(if $(filter production-weight-direct-group-cascade-probe,$(experiment)),--experiment production-weight-direct-group-cascade-probe,${experiment_arg})
experiment_arg := $(if $(filter production-weight-direct-group-vertical-chain-probe,$(experiment)),--experiment production-weight-direct-group-vertical-chain-probe,${experiment_arg})
experiment_arg := $(if $(filter production-weight-consolidated-vertical-chain-probe,$(experiment)),--experiment production-weight-consolidated-vertical-chain-probe,${experiment_arg})
experiment_arg := $(if $(filter production-weight-consolidated-horizontal-probe,$(experiment)),--experiment production-weight-consolidated-horizontal-probe,${experiment_arg})
experiment_arg := $(if $(filter production-weight-consolidated-full-chain-probe,$(experiment)),--experiment production-weight-consolidated-full-chain-probe,${experiment_arg})
experiment_arg := $(if $(filter production-weight-consolidated-full-chain-lstm-probe,$(experiment)),--experiment production-weight-consolidated-full-chain-lstm-probe,${experiment_arg})
PYTHON ?= python3
AIECC ?= $(shell ${PYTHON} -c "import pathlib, mlir_aie; print(pathlib.Path(mlir_aie.__file__).parent / 'bin' / 'aiecc')" 2>/dev/null || command -v aiecc 2>/dev/null || echo aiecc)

aie_py_src = lstm_cell_bf16_acc_cascade.py
aie_cc_src = lstm_layer_cascade.cc
runner_cc_src = runner.cpp

all: build/final_${artifact_suffix}.xclbin build/insts_${artifact_suffix}.bin ${targetname}_${artifact_suffix}

build/lstm_layer_cascade.o: ${aie_cc_src}
	mkdir -p $(@D)
ifeq ($(devicename),npu2)
	cd $(@D) && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} \
	  -I${mlir_aie_dir}/aie_kernels \
	  -I${mlir_aie_dir}/install/include \
	  -c ${srcdir}/${aie_cc_src} -o lstm_layer_cascade.o
else
	$(error devicename=$(devicename) not supported; cascade kernel is AIE2P-only — pass NPU2=1)
endif

build/aie_${artifact_suffix}.mlir: ${aie_py_src}
	mkdir -p $(@D)
	PYTHONPATH=${fork_python}:${repo_root}:$${PYTHONPATH} ${PYTHON} ${srcdir}/${aie_py_src} -d ${devicename} -L ${seq} ${experiment_arg} > $@.tmp
	mv $@.tmp $@

build/final_${artifact_suffix}.xclbin: build/aie_${artifact_suffix}.mlir build/lstm_layer_cascade.o
	mkdir -p $(@D)
	cd $(@D) && PYTHONPATH= ${AIECC} --aie-generate-xclbin --no-compile-host \
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
	rm -rf build ${targetname}_L* ${targetname}_hidden_state_only_L* ${targetname}_hidden_state_math_L* ${targetname}_hidden_state_weight_L* ${targetname}_hidden_state_weight_chain_L* ${targetname}_production_weight_cache_chain_L* ${targetname}_production_weight_stream_chain_L* ${targetname}_production_weight_direct_stream_L* ${targetname}_production_weight_direct_stream_probe_L* ${targetname}_production_weight_direct_layer_probe_L* ${targetname}_production_weight_direct_group_probe_L* ${targetname}_production_weight_direct_group_math_probe_L* ${targetname}_production_weight_direct_group_cascade_probe_L* ${targetname}_production_weight_direct_group_vertical_chain_probe_L* ${targetname}_production_weight_consolidated_vertical_chain_probe_L* ${targetname}_production_weight_consolidated_horizontal_probe_L* ${targetname}_production_weight_consolidated_full_chain_probe_L* ${targetname}_production_weight_consolidated_full_chain_lstm_probe_L*

.PHONY: all clean
