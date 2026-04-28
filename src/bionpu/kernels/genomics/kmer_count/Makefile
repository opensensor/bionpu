##===- Makefile -----------------------------------------------------------===##
# Build the AIE2P xclbin for the bionpu canonical k-mer counter.
#
# Per state/kmer_count_interface_contract.md (T1) — symbols, ObjectFifo
# names, constants, and the experiment switch are pinned there.
#
# Mirrors `bionpu-public/src/bionpu/kernels/crispr/pam_filter/Makefile`
# (the 133-line reference). Does NOT use the cascade pattern's
# `repo_root` / `fork_python` — kmer_count uses standard
# `aie.iron.ObjectFifo` only and does not need the fork python path.
#
# Four experiments — each maps to a fixed n_tiles fan-out and the
# corresponding BIONPU_KMER_COUNT_LAUNCH_CHUNKS env override consumed
# by kmer_count.py at IRON-emit time:
#
#   experiment=production    n_tiles=1   LAUNCH_CHUNKS=1
#   experiment=wide2         n_tiles=2   LAUNCH_CHUNKS=2
#   experiment=wide4         n_tiles=4   LAUNCH_CHUNKS=4
#   experiment=wide8         n_tiles=8   LAUNCH_CHUNKS=8
#
# `seq` parameterises the input length in bases (default a smoke
# 10 Kbp; production runs scale up via the host runner CLI). It does
# not change the xclbin shape — only the host-side dispatch chunking —
# but is baked into the artifact_suffix so multi-seq builds coexist.
#
# `K` parameterises the k-mer width ∈ {15, 21, 31} for the per-tile
# kernel symbol (`kmer_count_tile_k{K}`). Default 21.
#
# Build invocations (from this directory):
#
#   source /opt/xilinx/xrt/setup.sh
#   source /home/matteius/xdna-bringup/ironenv/bin/activate
#   export MLIR_AIE_DIR=/home/matteius/genetics/third_party/mlir-aie
#   # PEANO_INSTALL_DIR uses the ironenv-bundled peano per the
#   # `peano-ironenv-canonical` memory:
#   export PEANO_INSTALL_DIR=/home/matteius/xdna-bringup/ironenv/lib/python3.14/site-packages/llvm-aie
#   make NPU2=1 experiment=production seq=10000 all
#   make NPU2=1 experiment=wide4      seq=10000 all
#
# `NPU2=1` is required (AIE2P).
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
mlir_aie_dir ?= $(MLIR_AIE_DIR)

# Skip the env-required checks when the only goal is `clean` so
# housekeeping does not require sourcing the bring-up env. All build
# rules still require MLIR_AIE_DIR + NPU2=1 because make-common +
# PEANOWRAP2P_FLAGS are referenced unconditionally below.
ifneq ($(MAKECMDGOALS),clean)

ifeq ($(strip $(mlir_aie_dir)),)
$(error MLIR_AIE_DIR is unset; source the bring-up env first)
endif

include ${mlir_aie_dir}/programming_examples/makefile-common

# NPU2=1 enforcement. AIE2P (Strix / XDNA 2) is the only target.
ifneq ($(NPU2),1)
$(error NPU2=1 is required (AIE2P); pass NPU2=1 on the make line)
endif

endif  # MAKECMDGOALS != clean

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
seq ?= 10000
experiment ?= production
K ?= 21

# Per-experiment switch — n_tiles fan-out + LAUNCH_CHUNKS env override.
# The artifact_suffix encodes both n_tiles and seq so all four
# experiments × multiple seq values coexist under build/.
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
artifact_suffix = n${n_tiles}_L${seq}

aie_py_src        = kmer_count.py
aie_tile_cc_src   = kmer_count_tile.cc
aie_agg_cc_src    = kmer_count_aggregator.cc
aie_tile_obj      = kmer_count_tile.o
aie_agg_obj       = kmer_count_aggregator.o
runner_cc_src     = runner.cpp

PYTHON ?= python3
AIE_API_INCLUDE ?= -I${MLIR_AIE_DIR}/install/include -I${MLIR_AIE_DIR}/third_party/aie_api/include
AIECC ?= $(shell ${PYTHON} -c "import pathlib, mlir_aie; print(pathlib.Path(mlir_aie.__file__).parent / 'bin' / 'aiecc')" 2>/dev/null || command -v aiecc 2>/dev/null || echo aiecc)

all: build/final_${artifact_suffix}.xclbin build/insts_${artifact_suffix}.bin ${targetname}

# Per-tile compute kernel object — peano clang++ targeting aie2p.
# Exposes the three k-specialised symbols (kmer_count_tile_k15,
# kmer_count_tile_k21, kmer_count_tile_k31); aiecc resolves the one
# referenced by the IRON Kernel(...) declaration at lower time.
build/${aie_tile_obj}: ${aie_tile_cc_src}
	mkdir -p $(@D)
ifeq ($(devicename),npu2)
	cd $(@D) && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} \
	  -I${mlir_aie_dir}/aie_kernels \
	  ${AIE_API_INCLUDE} \
	  -c ${srcdir}/${aie_tile_cc_src} -o ${aie_tile_obj}
else ifeq ($(devicename),npu)
	cd $(@D) && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} \
	  -I${mlir_aie_dir}/aie_kernels \
	  ${AIE_API_INCLUDE} \
	  -c ${srcdir}/${aie_tile_cc_src} -o ${aie_tile_obj}
else
	$(error devicename=$(devicename) not supported; use NPU2=1)
endif

# Aggregator kernel object — single k-independent symbol
# `kmer_count_aggregator`. Built separately so aiecc reads both
# objects from the per-experiment cwd at link time.
build/${aie_agg_obj}: ${aie_agg_cc_src}
	mkdir -p $(@D)
ifeq ($(devicename),npu2)
	cd $(@D) && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} \
	  -I${mlir_aie_dir}/aie_kernels \
	  ${AIE_API_INCLUDE} \
	  -c ${srcdir}/${aie_agg_cc_src} -o ${aie_agg_obj}
else ifeq ($(devicename),npu)
	cd $(@D) && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} \
	  -I${mlir_aie_dir}/aie_kernels \
	  ${AIE_API_INCLUDE} \
	  -c ${srcdir}/${aie_agg_cc_src} -o ${aie_agg_obj}
else
	$(error devicename=$(devicename) not supported; use NPU2=1)
endif

# IRON Python lowering → MLIR. The BIONPU_KMER_COUNT_LAUNCH_CHUNKS env
# override is read by kmer_count.py at module-load time per the T1
# contract; we set it on the python3 invocation so the emitted MLIR
# carries the correct n_tiles fan-out for this experiment.
build/aie_${artifact_suffix}.mlir: ${aie_py_src}
	mkdir -p $(@D)
	BIONPU_KMER_COUNT_LAUNCH_CHUNKS=${launch_chunks} \
	  ${PYTHON} ${srcdir}/${aie_py_src} \
	    -d ${devicename} \
	    --k ${K} \
	    --launch-chunks ${launch_chunks} \
	    > $@.tmp
	mv $@.tmp $@

# xclbin + instruction stream from aiecc. Both kernel objects are
# copied into the xclbin build cwd so aiecc's link step finds them.
build/final_${artifact_suffix}.xclbin: build/aie_${artifact_suffix}.mlir build/${aie_tile_obj} build/${aie_agg_obj}
	mkdir -p $(@D)
	cd $(@D) && PATH=${PEANO_INSTALL_DIR}/bin:$${PATH} ${AIECC} \
	  --aie-generate-xclbin --no-compile-host \
	  --xclbin-name=$(@F) \
	  --no-xchesscc --no-xbridge \
	  --aie-generate-npu-insts --npu-insts-name=insts_${artifact_suffix}.bin \
	  aie_${artifact_suffix}.mlir

build/insts_${artifact_suffix}.bin: build/final_${artifact_suffix}.xclbin
	@test -f $@ || (echo "expected $@ as a side effect of the xclbin build" && exit 1)

HOST_CXX ?= g++-15

# Host runner — single binary at the kernel root that dispatches
# whichever xclbin the user passes via --xclbin. Mirrors the CRISPR
# pam_filter pattern (one host_runner binary per kernel directory).
${targetname}: ${runner_cc_src} build/final_${artifact_suffix}.xclbin
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

# T11 performs the install step — copies the (xclbin, insts.bin,
# host_runner) triple into
# `bionpu-public/src/bionpu/dispatch/_npu_artifacts/bionpu_kmer_count_k${K}_n${n_tiles}/`
# along with a per-artifact MANIFEST.md. T10 only sets up the build.

clean:
	rm -rf build ${targetname}

.PHONY: all clean
