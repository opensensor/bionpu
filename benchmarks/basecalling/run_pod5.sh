#!/usr/bin/env bash
# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
# SPDX-License-Identifier: GPL-3.0-only
#
# Basecall a pod5 read set on the NPU and verify byte-equality against
# a Dorado reference FASTQ.
#
# Usage:
#   benchmarks/basecalling/run_pod5.sh <pod5_path>
#
# Status: v0.1 ships a skeleton — the NPU basecalling pipeline lives in
# the bionpu source tree (src/bionpu/kernels/basecalling/) but the CLI
# wrapper that drives the full streaming pipeline is v0.2 scope. For
# v0.1 the supported flow is to run the per-kernel make targets
# manually and then verify byte-equality with `bionpu verify
# basecalling`.

set -euo pipefail

POD5="${1:-}"
OUT_DIR="benchmarks/results/basecalling/$(basename "${POD5%.pod5}" 2>/dev/null || echo unknown)"

if [[ -z "${POD5}" ]]; then
    cat <<EOF
Usage: $0 <pod5_path>

  pod5_path   Path to a pod5 read set (Nanopore raw signal).

Pre-computed reference FASTQs are at reference/basecalling/.
EOF
    exit 1
fi

REF_FASTQ="reference/basecalling/dorado-reference.fastq"
NPU_FASTQ="${OUT_DIR}/npu.fastq"

mkdir -p "${OUT_DIR}"

echo "==> $0 ${POD5}"
echo "    output dir:    ${OUT_DIR}"
echo "    reference:     ${REF_FASTQ}"

cat <<EOF

[v0.1 placeholder]
The end-to-end driver is v0.2 scope. For v0.1, run the kernels
manually and then call:

    bionpu verify basecalling "${NPU_FASTQ}" "${REF_FASTQ}"

The kernels live at:
    src/bionpu/kernels/basecalling/{conv_stem,lstm_cell_*,linear_projection,...}

The Dorado reference FASTQ is committed at reference/basecalling/ when
it has been generated on a host with Dorado available (the build is
not redistributable; see Dorado's license).
EOF
