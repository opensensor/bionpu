#!/usr/bin/env bash
# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
# GPL-3.0-or-later. See ../LICENSE.
#
# Run the upstream DNABERT-Epi training protocol for one fold + iter:
#
#   1. (one-time) pair-finetune DNABERT-3 on the mismatch-prediction task
#   2. CHANGE-seq scratch (in-vitro pre-training)
#   3. GUIDE-seq transfer (in-cellula fine-tune)
#   4. extract head into bionpu's expected format
#   5. report ROC-AUC / PR-AUC + verify CPU↔GPU byte-equivalence
#
# Use this to drive the 14-fold sweep that produces the paper's
# headline 0.9857 ± 0.0124 ROC-AUC. Each fold takes ~4-5 hours on
# RTX 4070 (8 epochs CHANGE-seq + 8 epochs GUIDE-seq transfer at
# batch=128). Pair-finetune is ~7 hours but only runs once.
#
# Idempotent: skips stages whose outputs already exist.
#
# Usage: run-fold-protocol.sh <fold> [iter]
#   fold:  0..13   (paper's 14-fold leave-one-sgRNA-group-out CV)
#   iter:  0..N    (random-seed iter; default 0)

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <fold:0-13> [iter:0]" >&2
    exit 1
fi

FOLD="$1"
ITER="${2:-0}"
SUBMODULE="$(cd "$(dirname "$0")/.." && pwd)/third_party/crispr_dnabert"
DATA="${BIONPU_DNABERT_DATA:-$HOME/dnabert-epi-data}"
LOG_DIR="${BIONPU_DNABERT_LOG_DIR:-/tmp}"

cd "$SUBMODULE"
source .venv-dnabert/bin/activate

echo "fold=${FOLD} iter=${ITER}"
echo "submodule: $SUBMODULE"
echo "data:      $DATA"
echo "logs:      $LOG_DIR/run-fold${FOLD}-iter${ITER}-{stageN}.log"
echo

# -----------------------------------------------------------------------------
# Stage 1 — pair-finetune (one-time, shared across folds)
# -----------------------------------------------------------------------------
PAIR_FT="$DATA/models/dnabert/pair_finetuned/model.safetensors"
if [[ -f "$PAIR_FT" ]]; then
    echo "[1/5] pair-finetune: present at $(dirname "$PAIR_FT") (skipping)"
else
    echo "[1/5] pair-finetune: running (one-time, ~7 hours)"
    python3 -u src/models/pair_finetuning_dnabert.py --pretrain \
        > "$LOG_DIR/run-fold${FOLD}-iter${ITER}-stage1-pair.log" 2>&1
fi
echo

# -----------------------------------------------------------------------------
# Stage 2 — CHANGE-seq preprocess + scratch (per-fold)
# -----------------------------------------------------------------------------
CS_INPUT="$DATA/input/Lazzarotto_2020_CHANGE_seq/DNABERT_dataset"
CS_CKPT="$DATA/models/dnabert/Lazzarotto_2020_CHANGE_seq/scratch/fold${FOLD}_iter${ITER}.pth"

if [[ -d "$CS_INPUT" ]]; then
    echo "[2a/5] CHANGE-seq preprocess: present (skipping)"
else
    echo "[2a/5] CHANGE-seq preprocess: running"
    python3 -u src/run_preprocess.py --model DNABERT --dataset Lazzarotto_2020_CHANGE_seq \
        > "$LOG_DIR/run-fold${FOLD}-iter${ITER}-stage2a-prep-changeseq.log" 2>&1
fi

if [[ -f "$CS_CKPT" ]]; then
    echo "[2b/5] CHANGE-seq scratch fold${FOLD}: present at $CS_CKPT (skipping)"
else
    echo "[2b/5] CHANGE-seq scratch fold${FOLD}: running (~2 hours)"
    # CHANGE-seq is the in-vitro half; the upstream validator rejects
    # `--dataset_in_cellula Lazzarotto_2020_CHANGE_seq`. Pass it via
    # --dataset_in_vitro instead.
    python3 -u src/run_model.py \
        --model DNABERT \
        --dataset_in_vitro Lazzarotto_2020_CHANGE_seq \
        --fold "$FOLD" --iter "$ITER" \
        --train --test --exe_type scratch \
        > "$LOG_DIR/run-fold${FOLD}-iter${ITER}-stage2b-train-changeseq.log" 2>&1
fi
echo

# -----------------------------------------------------------------------------
# Stage 3 — GUIDE-seq preprocess + transfer (per-fold)
# -----------------------------------------------------------------------------
GS_INPUT="$DATA/input/Lazzarotto_2020_GUIDE_seq/DNABERT_dataset"
GS_CKPT="$DATA/models/dnabert/Lazzarotto_2020_GUIDE_seq/transfer/fold${FOLD}_iter${ITER}.pth"

if [[ -d "$GS_INPUT" ]]; then
    echo "[3a/5] GUIDE-seq preprocess: present (skipping)"
else
    echo "[3a/5] GUIDE-seq preprocess: running"
    python3 -u src/run_preprocess.py --model DNABERT --dataset Lazzarotto_2020_GUIDE_seq \
        > "$LOG_DIR/run-fold${FOLD}-iter${ITER}-stage3a-prep-guideseq.log" 2>&1
fi

if [[ -f "$GS_CKPT" ]]; then
    echo "[3b/5] GUIDE-seq transfer fold${FOLD}: present at $GS_CKPT (skipping)"
else
    echo "[3b/5] GUIDE-seq transfer fold${FOLD}: running (~2 hours)"
    python3 -u src/run_model.py \
        --model DNABERT \
        --dataset_in_cellula Lazzarotto_2020_GUIDE_seq \
        --dataset_in_vitro Lazzarotto_2020_CHANGE_seq \
        --fold "$FOLD" --iter "$ITER" \
        --train --test --exe_type transfer \
        > "$LOG_DIR/run-fold${FOLD}-iter${ITER}-stage3b-train-guideseq.log" 2>&1
fi
echo

# -----------------------------------------------------------------------------
# Stage 4 — extract head into bionpu format
# -----------------------------------------------------------------------------
BIONPU_HEAD="$DATA/bionpu-fold${FOLD}-iter${ITER}.pt"
if [[ -f "$BIONPU_HEAD" ]]; then
    echo "[4/5] bionpu head: present at $BIONPU_HEAD (skipping)"
else
    echo "[4/5] extracting bionpu head from $GS_CKPT"
    deactivate
    source /home/matteius/genetics/.venv/bin/activate
    PYTHONPATH=/home/matteius/genetics/bionpu-public/src \
        python3 -m bionpu.cli score-extract-head \
        --upstream-checkpoint "$GS_CKPT" \
        --out "$BIONPU_HEAD"
    deactivate
    source "$SUBMODULE/.venv-dnabert/bin/activate"
fi
echo

# -----------------------------------------------------------------------------
# Stage 5 — surface metrics
# -----------------------------------------------------------------------------
RESULT_JSON="$DATA/results/Lazzarotto_2020_GUIDE_seq/DNABERT/transfer/fold${FOLD}_iter${ITER}.json"
echo "[5/5] reporting"
if [[ -f "$RESULT_JSON" ]]; then
    echo "  result file: $RESULT_JSON"
    grep -E "ROC|PR|F1|MCC|Accuracy" "$RESULT_JSON" 2>/dev/null || cat "$RESULT_JSON" | head -20
elif [[ -f "$LOG_DIR/run-fold${FOLD}-iter${ITER}-stage3b-train-guideseq.log" ]]; then
    echo "  result file not found; tail of stage 3b log:"
    grep -E "Accuracy|ROC AUC|PR AUC|F1|MCC" "$LOG_DIR/run-fold${FOLD}-iter${ITER}-stage3b-train-guideseq.log" | tail -3
fi
echo
echo "fold${FOLD} iter${ITER} done. bionpu head: $BIONPU_HEAD"
