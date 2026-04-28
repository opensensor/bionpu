##===- Makefile -----------------------------------------------------------===##
# bionpu — bert_int8_matmul (DNABERT-Epi AIE2P scorer port, v0.4 alpha).
#
# v0 ships the head specialization (M=47, K=768, N=2 — classifier head).
# qkvo and ffn specializations rebuild this same source with different
# M/K/N preprocessor flags; see MANIFEST.md. FFN builds are 768-output
# group kernels; host dispatch streams the full FFN weight from DDR in
# row-major groups.
#
# Targets:
#   build/head/final.xclbin    head xclbin (M=47 K=768 N=2)
#   build/head/insts.bin       head instruction stream
#   make qkvo                  qkvo group xclbin (M=47 K=768 N=768)
#   make ffn1                  ffn1 group xclbin (M=47 K=768 N=768)
#   make ffn2                  ffn2 group xclbin (M=47 K=3072 N=768)
#   bert_int8_matmul           host C++ runner (one binary; see runner.cpp)
#
# Invocation (from this directory):
#   source /opt/xilinx/xrt/setup.sh
#   source <ironenv>/bin/activate
#   export MLIR_AIE_DIR=<xdna-bringup>/mlir-aie
#   export PEANO_INSTALL_DIR=<ironenv>/lib/python3.11/site-packages/llvm-aie
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
targetname  = bert_int8_matmul
aie_py_src  = bert_int8_matmul.py
aie_cc_src  = bert_int8_matmul.cc
runner_cc_src = runner.cpp

# v0 head specialization. Override on the command line for other shapes:
#   make NPU2=1 M=47 K=768 N=768 V=qkvo all
M ?= 47
K ?= 768
N ?= 2
V ?= head
# qkvo/ffn K-axis chunk size. Row-major memtile-join topology keeps
# all N=768 outputs per tile row slab; K_CHUNK=8 leaves enough tile DM
# for the full scale vector, accumulator, output slab, and stack.
K_CHUNK ?= 8
N_TILES ?= 4

AIE_API_INCLUDE ?= -I${MLIR_AIE_DIR}/install/include -I${MLIR_AIE_DIR}/third_party/aie_api/include
SHAPE_FLAGS = -DBIONPU_BERT_M=$(M) -DBIONPU_BERT_K=$(K) -DBIONPU_BERT_N=$(N) \
              -DBIONPU_BERT_N_TILES=$(N_TILES) -DBIONPU_BERT_K_CHUNK=$(K_CHUNK)
# Large contiguous ObjectFifo/Buffer allocations (head x, qkvo/ffn acc)
# intentionally span 16 KiB tile-memory banks. Request sequential
# allocation directly instead of the aiecc default "bank-aware then
# fallback" path, which emits noisy warnings before producing the same
# valid layout.
AIE_ALLOC_SCHEME ?= basic-sequential

all: build/$(V)/final.xclbin build/$(V)/insts.bin ${targetname}

# AIE compute-kernel object — peano clang++ targeting aie2p.
build/$(V)/bert_int8_matmul.o: ${aie_cc_src}
	mkdir -p ${@D}
ifeq ($(devicename),npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} ${AIE_API_INCLUDE} ${SHAPE_FLAGS} -c $(srcdir)/${aie_cc_src} -o ${@F}
else ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} ${AIE_API_INCLUDE} ${SHAPE_FLAGS} -c $(srcdir)/${aie_cc_src} -o ${@F}
else
	@echo "Device type $(devicename) not supported; pass NPU2=1"
	@false
endif

# IRON lowering output → MLIR.
build/$(V)/aie.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< -d ${devicename} --M $(M) --K $(K) --N $(N) \
	    --variant $(V) --K-chunk $(K_CHUNK) > $@

# xclbin + instruction stream from aiecc.
build/$(V)/final.xclbin: build/$(V)/aie.mlir build/$(V)/bert_int8_matmul.o
	mkdir -p ${@D}
	cd ${@D} && aiecc --aie-generate-xclbin --no-compile-host \
	    --xclbin-name=$(@F) \
	    --alloc-scheme=$(AIE_ALLOC_SCHEME) \
	    --no-xchesscc --no-xbridge \
	    --aie-generate-npu-insts --npu-insts-name=insts.bin aie.mlir

build/$(V)/insts.bin: build/$(V)/final.xclbin
	@test -f $@ || (echo "ERROR: aiecc did not produce $@"; exit 1)

HOST_CXX ?= g++-15

${targetname}: ${runner_cc_src} build/$(V)/final.xclbin
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

clean:
	rm -rf build ${targetname}

qkvo:
	$(MAKE) NPU2=$(NPU2) M=47 K=768 N=768 V=qkvo all

ffn1:
	$(MAKE) NPU2=$(NPU2) M=47 K=768 N=768 V=ffn1 all

ffn2:
	$(MAKE) NPU2=$(NPU2) M=47 K=3072 N=768 V=ffn2 all

# Regenerate qkvo silicon-byte-equal reference fixtures (xs.bin, w.bin,
# y_qkvo_ref.bin) from the canonical Python packers + emulator. Use after
# K_CHUNK or shape changes invalidate the on-disk fixtures (cf. commit
# 8f74723 which flipped K_CHUNK 64→8 without regenerating refs).
SEED ?= 42

qkvo-refs:
	python3 $(srcdir)/regen_qkvo_refs.py --seed $(SEED) --output-dir $(srcdir)

.PHONY: all clean qkvo ffn1 ffn2 qkvo-refs
