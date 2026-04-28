##===- Makefile -----------------------------------------------------------===##
# Build the AIE2P xclbin(s) for the bionpu canonical k-mer counter (v0.5).
#
# Per state/kmer_count_interface_contract.md (T1) v0.5 — symbols, ObjectFifo
# names, constants, and the experiment switch are pinned there.
#
# v0.5 BUILD-TIME parameter list:
#   K         ∈ {15, 21, 31}   — k-mer width
#   experiment ∈ {production, wide2, wide4, wide8}
#                              → n_tiles ∈ {1, 2, 4, 8}
#   n_passes  ∈ {1, 4, 16}     — hash-slice partition count
#   pass_idx  ∈ {0..n_passes-1} — which slice this xclbin handles
#   seq                        — host-side fixture length tag (artifact suffix only)
#
# v0.5 produces N_PASSES xclbin/insts pairs per (k, n_tiles, n_passes)
# install. The single-xclbin target `final_${artifact_suffix}.xclbin`
# builds ONE specific (pass_idx) xclbin; the `all-passes` target loops
# pass_idx ∈ [0, n_passes) and builds them all.
#
# Build invocations (from this directory):
#
#   source /opt/xilinx/xrt/setup.sh
#   source /home/matteius/xdna-bringup/ironenv/bin/activate
#   export MLIR_AIE_DIR=/home/matteius/genetics/third_party/mlir-aie
#   export PEANO_INSTALL_DIR=/home/matteius/xdna-bringup/ironenv/lib/python3.14/site-packages/llvm-aie
#
#   # Single (k=21, n_tiles=4, n_passes=4) build (4 xclbins + 1 host_runner):
#   make NPU2=1 K=21 experiment=wide4 n_passes=4 seq=10000 all-passes
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
K ?= 21
n_passes ?= 4
pass_idx ?= 0
# v1.2 (b): batched-dispatch knob. n_chunks_per_launch ∈ {1, 2, 4, 8}.
# Default 1 keeps backward-compat smoke artifacts byte-equal. Set to 8
# for chr22 throughput cells (closes kmer-runner-per-dispatch-overhead).
n_chunks_per_launch ?= 1

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

targetname = bionpu_kmer_count

# Per-(k, n_tiles, n_passes, pass_idx, n_batch, seq) artifact suffix so
# all builds coexist under build/ without clobber. Pre-v1.2 (b) builds
# with n_chunks_per_launch=1 use the legacy suffix without `_b1`; the
# new batched cell appends `_b{N_BATCH}` for N_BATCH > 1.
ifeq ($(n_chunks_per_launch),1)
  artifact_suffix = k${K}_n${n_tiles}_np${n_passes}_p${pass_idx}_L${seq}
else
  artifact_suffix = k${K}_n${n_tiles}_np${n_passes}_b${n_chunks_per_launch}_p${pass_idx}_L${seq}
endif

# K_only suffix used by the per-K tile object (one .o per K).
k_obj_suffix = k${K}

aie_py_src        = kmer_count.py
aie_tile_cc_src   = kmer_count_tile.cc
aie_tile_obj      = kmer_count_tile_${k_obj_suffix}.o
runner_cc_src     = runner.cpp

PYTHON ?= python3
AIE_API_INCLUDE ?= -I${MLIR_AIE_DIR}/install/include -I${MLIR_AIE_DIR}/third_party/aie_api/include
AIECC ?= $(shell ${PYTHON} -c "import pathlib, mlir_aie; print(pathlib.Path(mlir_aie.__file__).parent / 'bin' / 'aiecc')" 2>/dev/null || command -v aiecc 2>/dev/null || echo aiecc)

# Single-pass build target: produces ONE xclbin/insts pair for the
# given (K, n_tiles, n_passes, pass_idx) tuple.
all: build/final_${artifact_suffix}.xclbin build/insts_${artifact_suffix}.bin ${targetname}

# Per-tile compute kernel object — peano clang++ targeting aie2p.
# One object per K (the C++ source instantiates ONE template specialisation
# gated on -DKMER_K_ACTIVE=$(K) to stay within the 64 KiB CoreTile DM cap).
# Note: this object is independent of n_passes / pass_idx / n_tiles —
# pass_idx / n_passes_log2 are kernel int32 args.
build/${aie_tile_obj}: ${aie_tile_cc_src} ${srcdir}/kmer_count_constants.h
	mkdir -p $(@D)
ifeq ($(devicename),npu2)
	cd $(@D) && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} \
	  -DKMER_K_ACTIVE=${K} \
	  -I${mlir_aie_dir}/aie_kernels \
	  ${AIE_API_INCLUDE} \
	  -c ${srcdir}/${aie_tile_cc_src} -o ${aie_tile_obj}
else ifeq ($(devicename),npu)
	cd $(@D) && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} \
	  -DKMER_K_ACTIVE=${K} \
	  -I${mlir_aie_dir}/aie_kernels \
	  ${AIE_API_INCLUDE} \
	  -c ${srcdir}/${aie_tile_cc_src} -o ${aie_tile_obj}
else
	$(error devicename=$(devicename) not supported; use NPU2=1)
endif

# IRON Python lowering → MLIR. n_passes + pass_idx are baked in at IRON-emit
# time as compile-time arith.constants (so each pass's xclbin filters on
# its own slice).
build/aie_${artifact_suffix}.mlir: ${aie_py_src}
	mkdir -p $(@D)
	BIONPU_KMER_COUNT_LAUNCH_CHUNKS=${launch_chunks} \
	  BIONPU_KMER_COUNT_N_PASSES=${n_passes} \
	  BIONPU_KMER_COUNT_PASS_IDX=${pass_idx} \
	  BIONPU_KMER_COUNT_N_CHUNKS_PER_LAUNCH=${n_chunks_per_launch} \
	  ${PYTHON} ${srcdir}/${aie_py_src} \
	    -d ${devicename} \
	    --k ${K} \
	    --launch-chunks ${launch_chunks} \
	    --n-passes ${n_passes} \
	    --pass-idx ${pass_idx} \
	    --n-chunks-per-launch ${n_chunks_per_launch} \
	    > $@.tmp
	mv $@.tmp $@

# Per-(K, n_tiles, n_passes, pass_idx) xclbin. The per-K tile object is
# copied / aliased into the build cwd so aiecc finds it under the
# canonical name `kmer_count_tile.o` regardless of K.
build/final_${artifact_suffix}.xclbin: build/aie_${artifact_suffix}.mlir build/${aie_tile_obj}
	mkdir -p $(@D)
	cp build/${aie_tile_obj} build/kmer_count_tile.o
	cd $(@D) && PATH=${PEANO_INSTALL_DIR}/bin:$${PATH} ${AIECC} \
	  --aie-generate-xclbin --no-compile-host \
	  --xclbin-name=$(@F) \
	  --no-xchesscc --no-xbridge \
	  --aie-generate-npu-insts --npu-insts-name=insts_${artifact_suffix}.bin \
	  aie_${artifact_suffix}.mlir

build/insts_${artifact_suffix}.bin: build/final_${artifact_suffix}.xclbin
	@test -f $@ || (echo "expected $@ as a side effect of the xclbin build" && exit 1)

HOST_CXX ?= g++-15

# Host runner — single binary at the kernel root.
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

# all-passes: build N_PASSES xclbin/insts pairs for the active
# (K, n_tiles, n_passes, seq) tuple, plus the host_runner.
#
# Iterates pass_idx ∈ [0, n_passes) by recursively invoking $(MAKE)
# with pass_idx overridden. Each recursive build produces one xclbin/
# insts pair.
all-passes: ${targetname}
	@for p in $$(seq 0 $$(( ${n_passes} - 1 ))); do \
	  echo "==> building pass_idx=$$p (n_chunks_per_launch=${n_chunks_per_launch})"; \
	  $(MAKE) -f $(firstword $(MAKEFILE_LIST)) \
	    NPU2=${NPU2} K=${K} experiment=${experiment} \
	    n_passes=${n_passes} pass_idx=$$p seq=${seq} \
	    n_chunks_per_launch=${n_chunks_per_launch} \
	    || exit 1; \
	done

clean:
	rm -rf build ${targetname}

.PHONY: all all-passes clean
