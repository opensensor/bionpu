#!/usr/bin/env bash
# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
# GPL-3.0-or-later. See ../LICENSE.
#
# One-shot host setup for replicating the upstream DNABERT-Epi paper
# (Kimata 2025) via third_party/crispr_dnabert/. Takes a clean host
# from "git submodule update --init" to "ready to run
# pair_finetuning_dnabert.py".
#
# Idempotent: skips work that's already done. Re-runnable.
#
# Surfaces every implicit requirement we discovered the hard way:
#   - the venv (Python 3.11+, with the upstream's transitive deps
#     INCLUDING the ones not in the upstream's pinned requirements)
#   - the system libs pysam needs to build htslib (libbz2-dev,
#     liblzma-dev, libcurl4-openssl-dev — the upstream README never
#     mentions these)
#   - the implicit config.yaml gaps (dataset_name: {}, model_info: {}
#     — top-level dicts the upstream code writes into without
#     creating)
#   - the hg38.fa reference (data_loader.py opens it unconditionally
#     in __init__, even for the no-epi pair-finetune path)
#
# Skips the actual training run; that's user-driven (multi-hour GPU job).

set -euo pipefail

DATA_DIR="${BIONPU_DNABERT_DATA:-$HOME/dnabert-epi-data}"
SUBMODULE="$(cd "$(dirname "$0")/.." && pwd)/third_party/crispr_dnabert"
VENV="$SUBMODULE/.venv-dnabert"

echo "setup-dnabert-host: data root  = $DATA_DIR"
echo "setup-dnabert-host: submodule  = $SUBMODULE"
echo "setup-dnabert-host: venv       = $VENV"
echo

# -----------------------------------------------------------------------------
# 1. System packages (sudo)
# -----------------------------------------------------------------------------
echo "[1/5] system packages (htslib build deps for pysam)"
need_pkgs=()
for pkg in libbz2-dev liblzma-dev libcurl4-openssl-dev libssl-dev zlib1g-dev; do
    if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"; then
        need_pkgs+=("$pkg")
    fi
done
if (( ${#need_pkgs[@]} > 0 )); then
    echo "      missing: ${need_pkgs[*]}"
    echo "      run:    sudo apt install -y ${need_pkgs[*]}"
    echo "      (script does not call sudo for you)"
    exit 1
fi
echo "      all required system packages present"
echo

# -----------------------------------------------------------------------------
# 2. venv (Python + upstream deps)
# -----------------------------------------------------------------------------
echo "[2/5] Python venv at $VENV"
if [[ ! -d "$VENV" ]]; then
    python3 -m venv "$VENV"
    echo "      created"
else
    echo "      already present"
fi

# The upstream's `requirements`-style README pins old versions; we
# install the latest 2026-era versions (torch 2.11+, transformers 5+)
# and then patch the tail of the import graph that the README forgot.
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet \
    torch transformers numpy pandas scikit-learn pyyaml tqdm pyarrow \
    huggingface-hub datasets matplotlib seaborn statsmodels \
    biopython shap captum umap-learn pysam
echo "      deps installed"
echo

# -----------------------------------------------------------------------------
# 3. config.yaml implicit gaps
# -----------------------------------------------------------------------------
echo "[3/5] config.yaml — checking for implicit-dict gaps"
config="$SUBMODULE/config.yaml"
if ! grep -qE '^dataset_name:' "$config"; then
    echo "      adding missing 'dataset_name: {}' top-level key"
    sed -i '1i dataset_name: {}\nmodel_info: {}\n' "$config"
fi
echo "      config.yaml has dataset_name + model_info"
echo

# -----------------------------------------------------------------------------
# 4. data + reference layout
# -----------------------------------------------------------------------------
echo "[4/5] data + reference layout under $DATA_DIR"
mkdir -p "$DATA_DIR/Yaish_2024" "$DATA_DIR/reference" "$DATA_DIR/models/dnabert/pretrained"

# Yaish 2024
yaish_csv="$DATA_DIR/Yaish_2024/datasets/FullGUIDEseq/include_on_targets/FullGUIDEseq_CR_Lazzarotto_2020_dataset.csv"
if [[ ! -f "$yaish_csv" ]]; then
    echo "      Yaish 2024 datasets not extracted yet. Run:"
    echo "        cd /home/matteius/genetics && source .venv/bin/activate"
    echo "        PYTHONPATH=bionpu-public/src python3 -c \"from bionpu.data import REGISTRY, Fetcher; print(Fetcher().fetch(REGISTRY['yaish_2024'], mode='full'))\""
    echo "        cd $DATA_DIR/Yaish_2024 && unzip <cached>/datasets.zip"
else
    echo "      Yaish 2024 datasets present"
fi

# DNABERT-3 base
dnabert_cfg="$DATA_DIR/models/dnabert/pretrained/config.json"
if [[ ! -f "$dnabert_cfg" ]]; then
    echo "      DNABERT-3 base weights not present. Run:"
    echo "        $VENV/bin/huggingface-cli download zhihan1996/DNA_bert_3 \\"
    echo "            --local-dir $DATA_DIR/models/dnabert/pretrained"
else
    echo "      DNABERT-3 base weights present"
fi

# hg38.fa reference (data_loader.py opens this unconditionally)
hg38="$DATA_DIR/reference/hg38.fa"
if [[ ! -f "$hg38" ]]; then
    echo "      hg38.fa NOT present at $hg38. data_loader.py will fail."
    echo "      Run:"
    echo "        curl -L -o /tmp/hg38.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
    echo "        gunzip -c /tmp/hg38.fa.gz > $hg38"
else
    echo "      hg38.fa present ($(du -h "$hg38" | cut -f1))"
fi
echo

# -----------------------------------------------------------------------------
# 5. ready-to-run smoke test
# -----------------------------------------------------------------------------
echo "[5/5] ready-to-run smoke check"
"$VENV/bin/python3" - <<'PY' || { echo "      smoke import failed — see error above"; exit 1; }
import os, sys
import yaml
sys.path.insert(0, os.path.expanduser("/home/matteius/genetics/bionpu-public/third_party/crispr_dnabert/src"))
os.chdir(os.path.expanduser("/home/matteius/genetics/bionpu-public/third_party/crispr_dnabert"))

cfg = yaml.safe_load(open("config.yaml"))
assert "dataset_name" in cfg, "config.yaml missing dataset_name dict"
assert "model_info" in cfg, "config.yaml missing model_info dict"

import models.dnabert_module           # noqa: F401
import models.pair_finetuning_dnabert  # noqa: F401
import models.data_loader              # noqa: F401
print("      all critical imports OK")
PY

echo
echo "setup-dnabert-host: ready. To kick off pair-finetuning:"
echo
echo "    cd $SUBMODULE"
echo "    source .venv-dnabert/bin/activate"
echo "    python3 src/models/pair_finetuning_dnabert.py --pretrain"
echo
