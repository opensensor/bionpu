#!/usr/bin/env bash
# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
# SPDX-License-Identifier: GPL-3.0-only
#
# Run a CRISPR off-target scan against a target chromosome and verify
# byte-equality against a Cas-OFFinder reference.
#
# Usage:
#   benchmarks/crispr/run_chr.sh <chr> [<guides_file>]
#
# Example:
#   benchmarks/crispr/run_chr.sh chr22 reference/crispr/guides_chr22.txt
#
# Status: v0.1 ships a skeleton — the NPU scan invocation lives in the
# bionpu source tree (src/bionpu/kernels/crispr/) but the CLI wrapper
# that drives a full chromosome scan from the command line is v0.2
# scope. For v0.1 the supported flow is to run the per-kernel make
# target manually and then verify byte-equality with `bionpu verify
# crispr`.

set -euo pipefail

CHR="${1:-}"
GUIDES="${2:-reference/crispr/guides_${CHR}.txt}"
OUT_DIR="benchmarks/results/crispr/${CHR}"

if [[ -z "${CHR}" ]]; then
    cat <<EOF
Usage: $0 <chr> [<guides_file>]

  chr           Target chromosome (chr1 ... chr22, chrX, chrY).
  guides_file   Newline-separated list of 20-nt guide spacers.
                Defaults to reference/crispr/guides_<chr>.txt.

Example:
    $0 chr22

End-to-end pipeline (v0.2 scope; v0.1 ships a manual workflow):
    1. Scan ${CHR} with the NPU PAM filter + match kernel.
    2. Run cas-offinder on the same input as the CPU reference.
    3. bionpu verify crispr <npu.tsv> <ref.tsv>
EOF
    exit 1
fi

REF_TSV="reference/crispr/casoffinder-${CHR}-canonical.tsv"
NPU_TSV="${OUT_DIR}/npu.tsv"

mkdir -p "${OUT_DIR}"

echo "==> $0 ${CHR}"
echo "    guides:        ${GUIDES}"
echo "    reference TSV: ${REF_TSV}"
echo "    output dir:    ${OUT_DIR}"

cat <<EOF

[v0.1 placeholder]
The end-to-end driver is v0.2 scope. For v0.1, run the kernels
manually (one-time, ~30 s build) and then call:

    bionpu verify crispr "${NPU_TSV}" "${REF_TSV}"

The verify command exits 0 on byte-equality and 1 on divergence.

The kernels live at:
    src/bionpu/kernels/crispr/{pam_filter,match_multitile_memtile,...}

The CPU reference is built from cas-offinder; pre-computed canonical
TSVs are at reference/crispr/.

When the v0.2 driver lands, this script will:
    1. Build (or use cached) NPU artifacts for the kernels.
    2. Dispatch the scan against \${CHR} via bionpu.dispatch.
    3. Run cas-offinder on the same input.
    4. Call bionpu verify crispr ... and exit with its return code.
EOF
