##===- Makefile -----------------------------------------------------------===##
# bionpu — bert_mini_block (DNABERT-Epi BERT-mini step 0.3).
#
# Builds two AIE2P kernels for one BERT transformer block at hidden=256:
#
#   build/softmax/final.xclbin    bert_mini_attention_softmax (per-row, M=47, PAD=64)
#   build/layer_norm/final.xclbin bert_mini_layer_norm        (per-row, HIDDEN=256)
#
# Both kernels link the same bert_mini_block.o (different IRON-Python
# lowerings; same C++ source). Following the bert_int8_matmul Makefile
# pattern. NPU2=1 is required (AIE2P bf16 hardware exp2/invsqrt/tanh).
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
aie_cc_src   = bert_mini_block.cc

# Per-kernel IRON-Python lowering files.
softmax_py_src    = bert_mini_softmax.py
layer_norm_py_src = bert_mini_layer_norm.py

M ?= 47
HIDDEN ?= 256
SOFTMAX_PAD ?= 64
NUM_HEADS ?= 4

AIE_API_INCLUDE ?= -I${MLIR_AIE_DIR}/install/include -I${MLIR_AIE_DIR}/third_party/aie_api/include
SHAPE_FLAGS = -DBIONPU_BERT_MINI_M=$(M) \
              -DBIONPU_BERT_MINI_HIDDEN=$(HIDDEN) \
              -DBIONPU_SOFTMAX_PAD=$(SOFTMAX_PAD) \
              -DBIONPU_LN_PAD=$(HIDDEN)

AIE_ALLOC_SCHEME ?= basic-sequential

all: build/softmax/final.xclbin build/softmax/insts.bin \
     build/layer_norm/final.xclbin build/layer_norm/insts.bin

# ─── shared object — bert_mini_block.cc ────────────────────────────
build/softmax/bert_mini_block.o: ${aie_cc_src}
	mkdir -p ${@D}
ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} \
	    ${AIE_API_INCLUDE} ${SHAPE_FLAGS} -c $(srcdir)/${aie_cc_src} -o ${@F}
else
	@echo "Device $(devicename) not supported; pass NPU2=1"
	@false
endif

build/layer_norm/bert_mini_block.o: ${aie_cc_src}
	mkdir -p ${@D}
ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} \
	    ${AIE_API_INCLUDE} ${SHAPE_FLAGS} -c $(srcdir)/${aie_cc_src} -o ${@F}
else
	@false
endif

# ─── softmax variant ───────────────────────────────────────────────
build/softmax/aie.mlir: ${srcdir}/${softmax_py_src}
	mkdir -p ${@D}
	python3 $< -d ${devicename} --M $(M) --PAD $(SOFTMAX_PAD) --NUM_HEADS $(NUM_HEADS) > $@

build/softmax/final.xclbin: build/softmax/aie.mlir build/softmax/bert_mini_block.o
	mkdir -p ${@D}
	cd ${@D} && aiecc --aie-generate-xclbin --no-compile-host \
	    --xclbin-name=$(@F) \
	    --alloc-scheme=$(AIE_ALLOC_SCHEME) \
	    --no-xchesscc --no-xbridge \
	    --aie-generate-npu-insts --npu-insts-name=insts.bin aie.mlir

build/softmax/insts.bin: build/softmax/final.xclbin
	@test -f $@ || (echo "ERROR: aiecc did not produce $@"; exit 1)

# ─── layer_norm variant ────────────────────────────────────────────
build/layer_norm/aie.mlir: ${srcdir}/${layer_norm_py_src}
	mkdir -p ${@D}
	python3 $< -d ${devicename} --M $(M) --HIDDEN $(HIDDEN) > $@

build/layer_norm/final.xclbin: build/layer_norm/aie.mlir build/layer_norm/bert_mini_block.o
	mkdir -p ${@D}
	cd ${@D} && aiecc --aie-generate-xclbin --no-compile-host \
	    --xclbin-name=$(@F) \
	    --alloc-scheme=$(AIE_ALLOC_SCHEME) \
	    --no-xchesscc --no-xbridge \
	    --aie-generate-npu-insts --npu-insts-name=insts.bin aie.mlir

build/layer_norm/insts.bin: build/layer_norm/final.xclbin
	@test -f $@ || (echo "ERROR: aiecc did not produce $@"; exit 1)

softmax: build/softmax/final.xclbin build/softmax/insts.bin

layer_norm: build/layer_norm/final.xclbin build/layer_norm/insts.bin

clean:
	rm -rf build

.PHONY: all clean softmax layer_norm
