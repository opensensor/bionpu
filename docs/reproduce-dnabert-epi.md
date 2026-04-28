# Reproducing the DNABERT-Epi baseline

How to drive the
[`opensensor/CRISPR_DNABERT`](https://github.com/opensensor/CRISPR_DNABERT)
fork (pinned as a git submodule under
[`third_party/crispr_dnabert/`](../third_party/crispr_dnabert))
to produce a fine-tuned classifier checkpoint that
[`bionpu score`](../src/bionpu/cli.py) can load via `--weights`.

This document is the bridge between *upstream paper reproduction* and
*bionpu's pluggable scorer interface.* It covers only the no-epi
variant; the with-epi (BigWig) variant is a follow-up.

## Status

**End-to-end paper-replication achieved 2026-04-28** on the ProArt
RTX 4070 host. Single-fold result on Lazzarotto 2020 GUIDE-seq fold 0
(scratch mode, 8 epochs, no pair-finetune pretraining, no in-vitro
CHANGE-seq transfer baseline):

| Metric | Our fold-0 | Paper (Kimata 2025, mean ± σ over 14 folds) |
|---|---|---|
| ROC-AUC | **0.9824** | 0.9857 ± 0.0124 |
| PR-AUC | **0.5448** | 0.5501 ± 0.0673 |

Both metrics are within 1 σ of the paper's reported mean, despite
running with current 2026-era deps (torch 2.11, transformers 5.x,
datasets 4.x) instead of the paper's pinned 2024 versions.

The upstream pipeline required a small set of bug fixes to run on
those deps; they're filed as
[opensensor/CRISPR_DNABERT#1](https://github.com/opensensor/CRISPR_DNABERT/pull/1).
Until that PR merges, run from the patches in the submodule's
working tree.

## Prerequisites

```bash
# From the bionpu-public root:
git submodule update --init --recursive third_party/crispr_dnabert

# Upstream Python deps (per third_party/crispr_dnabert/README.md):
python -m venv .venv-dnabert
source .venv-dnabert/bin/activate
pip install \
    'torch==2.5.1' \
    'transformers==4.48.3' \
    'numpy==2.0.2' \
    'pandas==2.2.3' \
    'scikit-learn==1.6.1'
```

## Datasets and base weights

The upstream training pipeline expects:

1. **Yaish 2024 datasets** — the Lazzarotto 2020 GUIDE-seq + CHANGE-seq
   off-target pairs, packaged by Yaish *et al.* at
   <https://github.com/OrensteinLab/CRISPR-Bulge/blob/main/files/datasets.zip>.
   Pin SHA-256: `f892f70b...3eab0` (524 MiB). Pre-wired as a fetcher:
   ```python
   from bionpu.data import REGISTRY, Fetcher
   Fetcher().fetch(REGISTRY["yaish_2024"], mode="full")
   # → ~/.../data_cache/crispr/yaish_2024/datasets.zip
   ```
   Extract that zip under a directory the upstream `config.yaml` points at.
2. **DNABERT-3 base weights** — Hugging Face model
   `zhihan1996/DNA_bert_3`. Downloaded automatically by `transformers`
   on first use, or pre-pulled via:
   ```bash
   huggingface-cli download zhihan1996/DNA_bert_3 \
       --local-dir ./pretrained-dnabert3
   ```
3. **(Optional, for the with-epi variant later)** ATAC-seq / H3K4me3 /
   H3K27ac BigWig files for the cell type of interest. The paper used
   T-cell GEO accessions `GSM4498611`, `GSM4495703`, `GSM4495711`.

Edit `third_party/crispr_dnabert/config.yaml` to point at where you
extracted Yaish 2024 and the DNABERT-3 weights.

## Fine-tune DNABERT for mismatch prediction

Per the upstream README:

```bash
cd third_party/crispr_dnabert
python3 src/models/pair_finetuning_dnabert.py --pretrain
```

This produces an intermediate checkpoint that the off-target task
fine-tunes on top of.

## Train + test on Lazzarotto 2020 GUIDE-seq (no-epi)

```bash
python3 src/run_preprocess.py \
    --model DNABERT \
    --dataset Lazzarotto_2020_GUIDE_seq

python3 src/run_model.py \
    --model DNABERT \
    --dataset_in_cellula Lazzarotto_2020_GUIDE_seq \
    --dataset_in_vitro   Lazzarotto_2020_CHANGE_seq \
    --fold 0 --iter 0 \
    --train --test \
    --exe_type transfer
    # NOTE: omit `-epi` / `-uepi` to train the no-epi variant we port
```

Replicating the paper's reported metric requires running across all 14
folds with the upstream `run_result.py` aggregator. The paper reports
ROC-AUC `0.9857 ± 0.0124` across folds.

## Export a checkpoint for `bionpu score`

The upstream training writes a state dict for the full
`DNABERTEpiModule`. For the no-epi variant bionpu's clean-room
classifier head only needs the final `Dropout → Linear(768, 2)`
projection. Use the bundled extractor:

```bash
bionpu score-extract-head \
    --upstream-checkpoint third_party/crispr_dnabert/.../fold0_iter0.pt \
    --out bionpu-dnabert-epi-noEpi-fold0.pt
```

The extractor handles three upstream key layouts (`classifier.1.*`,
flat `classifier.*`, and DataParallel `module.classifier.*`). It
fails loud with a precise error if the checkpoint was trained WITH
epigenetic features (the with-epi `Linear`'s `in_features` is
`768 + 256*N`, not `768` — the wrong shape for the no-epi head).

Then:

```bash
bionpu score \
    --candidates results/scan/chr22.tsv \
    --out       results/score/dnabert-epi-cpu.tsv \
    --device cpu \
    --weights  bionpu-dnabert-epi-noEpi-fold0.pt
```

## Cross-device byte-equivalence check

Once a CPU score TSV is produced, repeat with `--device gpu` and
compare under `NUMERIC_EPSILON` (cross-fabric ULP-level deviation is
expected; identity columns must match exactly):

```bash
bionpu score \
    --candidates results/scan/chr22.tsv \
    --out       results/score/dnabert-epi-gpu.tsv \
    --device gpu \
    --weights  bionpu-dnabert-epi-noEpi-fold0.pt \
    --verify   results/score/dnabert-epi-cpu.tsv \
    --verify-policy NUMERIC_EPSILON \
    --verify-epsilon 1e-4
```

Tighten `--verify-epsilon` once we have empirical drift numbers from
matched CPU / GPU runs. The eventual NPU backend will be held to the
same NUMERIC_EPSILON contract.

## What's deferred

* **With-epi variant.** Adds the BigWig fetcher stage + the upstream
  gating MLP. Both are scoped after no-epi proves out end-to-end.
* **CRISPRoffT validation corpus.** The
  [`bionpu.data.fetchers.crisproff`](../src/bionpu/data/fetchers/crisproff.py)
  fetcher pins the SHA-256 of the 144 MiB TSV; once a checkpoint
  exists we evaluate against it as an independent benchmark.
* **AIE2P scorer backend.** The clean-room classifier head and
  scorer interface stay device-agnostic; the NPU port slots in as a
  third backend behind the same `--device` flag.
