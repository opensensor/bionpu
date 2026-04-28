#!/usr/bin/env bash
# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
# SPDX-License-Identifier: GPL-3.0-only
#
# T15 — End-to-end k-mer counting benchmark wrapper (NPU silicon vs
# Jellyfish 1T/8T). Sources the build env, invokes the python bench
# under one wrapping ``npu_silicon_lock``, and copies the results
# JSON to results/kmer/v1-bench-{iso}/.
#
# Per state/kmer_count_interface_contract.md (T1) v0.5 — the bench
# uses the v0.5 streaming + multi-pass build (n_tiles=4, n_passes=4).
#
# Usage:
#   benchmarks/genomics/run_kmer_bench.sh              # full sweep (smoke + synthetic + chr22)
#   benchmarks/genomics/run_kmer_bench.sh --skip-chr22 # smoke + synthetic only (fast)
#
# Environment overrides:
#   KMER_BENCH_OUT_DIR    pre-set output directory (default auto-iso)
#   KMER_BENCH_KS         override k sweep (default "15,21,31")
#   KMER_BENCH_FIXTURES   override fixture sweep (default "smoke,synthetic,chr22")
#   KMER_BENCH_N_ITERS    iters per cell (default 3)
#   KMER_BENCH_WARMUP     warmup iters (default 1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="/home/matteius/genetics"
BIONPU_PUBLIC="${REPO_ROOT}/bionpu-public"

# ---- Activate XRT + ironenv ------------------------------------------------- #
# CLAUDE.md "Build env" section: source /opt/xilinx/xrt/setup.sh +
# /home/matteius/xdna-bringup/ironenv/bin/activate. PYTHONPATH gates
# both the in-process pyxrt and bionpu sources.
source /opt/xilinx/xrt/setup.sh > /dev/null 2>&1
# shellcheck disable=SC1091
source /home/matteius/xdna-bringup/ironenv/bin/activate
export PYTHONPATH="/opt/xilinx/xrt/python:${BIONPU_PUBLIC}/src:${PYTHONPATH:-}"

# ---- Resolve sweep knobs ---------------------------------------------------- #
KS="${KMER_BENCH_KS:-15,21,31}"
FIXTURES="${KMER_BENCH_FIXTURES:-smoke,synthetic,chr22}"
N_ITERS="${KMER_BENCH_N_ITERS:-3}"
WARMUP="${KMER_BENCH_WARMUP:-1}"

# Forward extra args (e.g. --skip-chr22) verbatim.
EXTRA=("$@")

ISO="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${KMER_BENCH_OUT_DIR:-${REPO_ROOT}/results/kmer/v1-bench-${ISO}}"
mkdir -p "${OUT_DIR}"

echo "==> $0"
echo "    KS=${KS}"
echo "    FIXTURES=${FIXTURES}"
echo "    N_ITERS=${N_ITERS} WARMUP=${WARMUP}"
echo "    OUT_DIR=${OUT_DIR}"

# ---- Pre-flight: jellyfish + ironenv -------------------------------------- #
if ! command -v jellyfish > /dev/null 2>&1; then
    echo "ERROR: jellyfish not on PATH; install jellyfish or sudo apt install jellyfish" >&2
    exit 2
fi

# ---- Run ------------------------------------------------------------------ #
python3 "${SCRIPT_DIR}/bench_kmer_count.py" \
    --out "${OUT_DIR}" \
    --ks "${KS}" \
    --fixtures "${FIXTURES}" \
    --n-iters "${N_ITERS}" \
    --warmup "${WARMUP}" \
    "${EXTRA[@]}"

echo "==> wrote ${OUT_DIR}/measurements.json"
