#!/usr/bin/env bash
# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
# GPL-3.0-or-later. See ../LICENSE.
#
# Drive the 14-fold leave-one-sgRNA-group-out CV sweep that lets us
# claim a paper-comparable mean-AUROC. Sequential — one fold at a
# time, ~4-5 hours each, ~2.5-3 days end to end. After completion,
# scoop the per-fold ROC-AUC + PR-AUC into a summary CSV and compute
# mean ± stddev across folds.
#
# Survives reboot: every per-fold pipeline is idempotent
# (see tools/run-fold-protocol.sh — skip-if-output-exists). Re-run
# this script after a reboot and it picks up where it left off.

set -euo pipefail

PROTOCOL="$(cd "$(dirname "$0")" && pwd)/run-fold-protocol.sh"
DATA="${BIONPU_DNABERT_DATA:-$HOME/dnabert-epi-data}"
ITER="${BIONPU_ITER:-0}"
FOLDS="${BIONPU_FOLDS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13}"

SUMMARY="$DATA/results/14fold-sweep-iter${ITER}.csv"
mkdir -p "$(dirname "$SUMMARY")"
[[ -f "$SUMMARY" ]] || echo "fold,iter,roc_auc,pr_auc,accuracy,f1,mcc" > "$SUMMARY"

echo "14-fold sweep — folds: $FOLDS, iter: $ITER"
echo "summary CSV: $SUMMARY"
echo

for FOLD in $FOLDS; do
    echo "================================="
    echo "==  fold $FOLD / iter $ITER  ==="
    echo "================================="
    "$PROTOCOL" "$FOLD" "$ITER"

    # Scrape final test metrics from the GUIDE-seq transfer log.
    LOG="/tmp/run-fold${FOLD}-iter${ITER}-stage3b-train-guideseq.log"
    if [[ -f "$LOG" ]]; then
        # Upstream prints e.g.:
        #   Accuracy: 0.9981, Precision: 0.6063, Recall: 0.4824, F1 Score: 0.5373, MCC: 0.5399, ROC AUC: 0.9824, PR AUC: 0.5448
        line=$(grep -E "ROC AUC" "$LOG" | tail -1)
        if [[ -n "$line" ]]; then
            roc=$(echo "$line" | sed -E 's/.*ROC AUC: ([0-9.]+).*/\1/')
            pr=$(echo "$line"  | sed -E 's/.*PR AUC: ([0-9.]+).*/\1/')
            acc=$(echo "$line" | sed -E 's/.*Accuracy: ([0-9.]+).*/\1/')
            f1=$(echo "$line"  | sed -E 's/.*F1 Score: ([0-9.]+).*/\1/')
            mcc=$(echo "$line" | sed -E 's/.*MCC: ([0-9.]+).*/\1/')
            echo "$FOLD,$ITER,$roc,$pr,$acc,$f1,$mcc" >> "$SUMMARY"
            echo "  → recorded: ROC=$roc, PR=$pr, Acc=$acc, F1=$f1, MCC=$mcc"
        else
            echo "  → no metrics line found in $LOG (fold may have errored)"
        fi
    fi
    echo
done

echo "================================="
echo "==  sweep complete             ==="
echo "================================="
echo "summary: $SUMMARY"
echo
column -ts, "$SUMMARY"

# Mean / stddev across folds — Python one-liner because awk is fiddly here.
python3 - <<PY
import csv, statistics
with open("$SUMMARY") as f:
    rows = list(csv.DictReader(f))
roc = [float(r["roc_auc"]) for r in rows if r.get("roc_auc")]
pr  = [float(r["pr_auc"])  for r in rows if r.get("pr_auc")]
if roc:
    print(f"\nROC-AUC:  mean={statistics.mean(roc):.4f}  stddev={statistics.stdev(roc):.4f}  (n={len(roc)})")
if pr:
    print(f"PR-AUC:   mean={statistics.mean(pr):.4f}  stddev={statistics.stdev(pr):.4f}  (n={len(pr)})")
PY
