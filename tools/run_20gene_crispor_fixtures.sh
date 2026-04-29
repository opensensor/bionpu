#!/usr/bin/env bash
# bionpu — 20-gene CRISPOR fixture generator.
# Generates `tests/fixtures/crispor_reference/{gene}.tsv` and
# `{gene}_offtargets.tsv` for each of the 20 pinned genes by running
# CRISPOR locally at the pinned commit.
#
# Pre-requisites (per `tests/fixtures/crispor_reference/DELTAS.md`):
#  1. CRISPOR cloned and checked out at commit
#     ed47b7e856010ad0f9f1660872563ef9f736e76c. Suggested location:
#     /home/$USER/crispor-install/crisporWebsite/
#  2. CRISPOR's hg38 genome layout populated under
#     <crispor>/genomes/hg38/ with: hg38.fa + BWA index (.amb/.ann/
#     .bwt/.pac/.sa) + hg38.2bit + hg38.sizes.
#  3. CRISPOR Python deps (Python 3.11; biopython, twobitreader,
#     pytabix, pandas, scipy, matplotlib, lmdbm, scikit-learn, xlwt).
# Optional:
#   CRISPOR_GENES=AAVS1,EMX1   # generate only selected symbols
#   CRISPOR_NO_EFF_SCORES=0    # attempt Azimuth/RS2 efficiency scoring
#
# UCSC academic license; do NOT vendor CRISPOR source into bionpu-public.
# Per `restrictive-license-model-policy.md` memory note (2026-04-28).

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
CRISPOR_DIR="${CRISPOR_DIR:-/home/$USER/crispor-install/crisporWebsite}"
CRISPOR_PYTHON="${CRISPOR_PYTHON:-/home/$USER/crispor-install/venv/bin/python}"
CRISPOR_NO_EFF_SCORES="${CRISPOR_NO_EFF_SCORES:-1}"
FIXTURES="$REPO/bionpu-public/tests/fixtures/crispor_reference"
PINNED_JSON="$FIXTURES/genes_pinned.json"

if [[ ! -f "$PINNED_JSON" ]]; then
  echo "missing $PINNED_JSON" >&2
  exit 2
fi
if [[ ! -d "$CRISPOR_DIR" ]]; then
  echo "CRISPOR install not found at $CRISPOR_DIR" >&2
  echo "see DELTAS.md for installation steps" >&2
  exit 2
fi
if [[ ! -d "$CRISPOR_DIR/genomes/hg38" ]]; then
  echo "CRISPOR genome layout missing at $CRISPOR_DIR/genomes/hg38" >&2
  exit 2
fi
if [[ ! -x "$CRISPOR_PYTHON" ]]; then
  echo "CRISPOR python interpreter not found: $CRISPOR_PYTHON" >&2
  exit 2
fi

# Slice each pinned gene's locus from the GRCh38 FASTA, run CRISPOR.
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

GENOME_FASTA="$REPO/data_cache/genomes/grch38/hg38.fa"

# Pull the gene list from the JSON via Python (jq may not be everywhere).
mapfile -t GENES < <("$CRISPOR_PYTHON" -c "
import json, sys
with open(sys.argv[1]) as fh:
    meta = json.load(fh)
selected = {s.strip() for s in sys.argv[2].split(',') if s.strip()}
for e in meta['genes']:
    if selected and e['symbol'] not in selected:
        continue
    print(f\"{e['symbol']}\\t{e['chrom']}\\t{e['start_1b']}\\t{e['end_1b']}\")
" "$PINNED_JSON" "${CRISPOR_GENES:-}")

CRISPOR_FLAGS=()
if [[ "$CRISPOR_NO_EFF_SCORES" != "0" ]]; then
  CRISPOR_FLAGS+=(--noEffScores)
fi

for gene_line in "${GENES[@]}"; do
  IFS=$'\t' read -r sym chrom start1 end1 <<< "$gene_line"
  echo "==> $sym ($chrom:$start1-$end1)" >&2
  out_tsv="$FIXTURES/${sym}.tsv"
  out_offt="$FIXTURES/${sym}_offtargets.tsv"
  if [[ -f "$out_tsv" && -f "$out_offt" ]]; then
    echo "    skip (already present)" >&2
    continue
  fi

  # Slice the locus to a FASTA. CRISPOR accepts FASTA input directly.
  locus_fa="$TMP_DIR/${sym}.fa"
  "$CRISPOR_PYTHON" - "$GENOME_FASTA" "$chrom" "$start1" "$end1" "$locus_fa" "$sym" <<'PY'
import sys
fasta, chrom, start1, end1, out, sym = sys.argv[1:]
start = int(start1) - 1  # 1-based inclusive -> 0-based
end = int(end1)
in_target = False
buf = []
chrom_pos = 0
need = end - start
with open(fasta) as fh:
    for line in fh:
        if line.startswith(">"):
            if in_target:
                break
            tok = line[1:].strip().split(None, 1)[0]
            if tok == chrom:
                in_target = True
                chrom_pos = 0
            continue
        if not in_target:
            continue
        s = line.strip()
        ln = len(s)
        ls, le = chrom_pos, chrom_pos + ln
        chrom_pos = le
        ovs = max(ls, start)
        ove = min(le, end)
        if ovs < ove:
            buf.append(s[ovs - ls:ove - ls])
        if chrom_pos >= end:
            break
seq = "".join(buf)
assert len(seq) == need, (len(seq), need)
with open(out, "w") as fh:
    fh.write(f">{sym}_{chrom}_{start1}_{end1}\n")
    for i in range(0, len(seq), 80):
        fh.write(seq[i:i+80] + "\n")
PY

  pushd "$CRISPOR_DIR" > /dev/null
  set +e
  "$CRISPOR_PYTHON" crispor.py \
    "${CRISPOR_FLAGS[@]}" \
    --pam=NGG \
    --maxOcc=60000 \
    --mm=4 \
    --guideLen=20 \
    hg38 \
    "$locus_fa" \
    "$out_tsv" \
    -o "$out_offt" 2>"$TMP_DIR/${sym}.log"
  rc=$?
  set -e
  popd > /dev/null

  if [[ $rc -ne 0 ]]; then
    echo "    CRISPOR failed (rc=$rc); see $TMP_DIR/${sym}.log" >&2
    cp "$TMP_DIR/${sym}.log" "$FIXTURES/${sym}.crispor.err.log" || true
  else
    echo "    OK -> $out_tsv ($(wc -l < "$out_tsv") rows)" >&2
  fi
done

echo "fixture generation complete; see $FIXTURES" >&2
