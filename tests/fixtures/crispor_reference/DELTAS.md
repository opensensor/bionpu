# 20-gene CRISPOR Validation Harness — Deltas Log

**PRD reference:** `PRDs/PRD-guide-design-on-xdna.md` v0.2 §4.3 hard gates.
**Authored:** 2026-04-28 (Wave 1 / Tier 1+ harness landing).
**Pinned commits:**
- CRISPOR: `ed47b7e856010ad0f9f1660872563ef9f736e76c` (per
  `state/spike/composite_scoring/phase0_audit.json`).
- bionpu repo: see `state/wave1/twenty_gene_validation_status.json`
  field `git_commit_repo` for the run-time pin.

This file captures every measurable disagreement between bionpu's
`bionpu crispr design` output and CRISPOR's reference output, plus the
context needed to interpret each delta as a *known-honest gap* versus a
*regression*. Empty sections under "per-gene deltas" mean no
disagreement is currently observed (or the CRISPOR fixture is not yet
generated; see "CRISPOR install status" below).

## CRISPOR install status

CRISPOR is consumed as an **external runtime dependency**, not vendored
into `bionpu-public/`. This is mandated by:

- `state/spike/composite_scoring/phase0_audit.json` ➜
  `restrictive_license_findings[CRISPOR]`: "academic-only since v4 …
  do not vendor / fork the source."
- Memory note `restrictive-license-model-policy.md`.

### Recommended local install

```bash
# 1. Clone and pin (Python 3.11 — CRISPOR uses cgi/pipes which Py3.13+ removed)
git clone https://github.com/maximilianh/crisporWebsite.git /home/$USER/crispor-install/crisporWebsite
cd /home/$USER/crispor-install/crisporWebsite
git checkout ed47b7e856010ad0f9f1660872563ef9f736e76c

# 2. Python 3.11 venv with deps
python3.11 -m venv /home/$USER/crispor-install/venv
source /home/$USER/crispor-install/venv/bin/activate
pip install biopython twobitreader pytabix pandas scipy matplotlib lmdbm scikit-learn xlwt

# 3. Set up genome layout
mkdir -p genomes/hg38
ln -s /path/to/hg38.fa genomes/hg38/hg38.fa
./bin/Linux-x86_64/bwa index genomes/hg38/hg38.fa
./bin/Linux-x86_64/faToTwoBit genomes/hg38/hg38.fa genomes/hg38/hg38.2bit
./bin/Linux-x86_64/twoBitInfo genomes/hg38/hg38.2bit genomes/hg38/hg38.sizes
# genomeInfo.tab: a 2-col tab-separated file with `name\thg38` and `org\thg38`.
printf "name\thg38\norg\thg38\n" > genomes/hg38/genomeInfo.tab
touch genomes/hg38/hg38.segments.bed  # empty is OK; CRISPOR no-ops if absent

# 4. Generate fixtures
cd /path/to/genetics
bionpu-public/tools/run_20gene_crispor_fixtures.sh
```

### Known install gotchas

1. **Python version**: CRISPOR's `crispor.py` imports `cgi` and `pipes`
   at top of file (lines 11-12). These were removed in Python 3.13. Use
   Python 3.11 (3.10 also works). The repo's primary `ironenv` is on
   Python 3.14 and CANNOT run CRISPOR directly — keep CRISPOR in a
   separate Py3.11 venv.

2. **BWA index**: `bwa index` of GRCh38 takes 1-2 hours single-threaded
   on a developer laptop and writes ~5 GB across `.bwt/.ann/.amb/.pac/.sa`.
   Run once; reuse forever.

3. **Azimuth / Doench RS2**: CRISPOR's effscores rely on
   `bin/Azimuth-2.0/`. The pickled regressor was trained on Python 3.6 /
   sklearn 0.18 era. On Py3.11 + modern sklearn, the pickle either loads
   with warnings or fails outright. The harness sidesteps this via
   `--noEffScores` if needed; the per-guide-Doench gate is a SOFT gate
   per PRD §4.3 v0.2 status note (RS1 ≠ RS2 by construction).

4. **Some genes vs CRISPOR's annotation**: `AAVS1` is not a HGNC
   gene symbol (it's a locus name = PPP1R12C intron 1). CRISPOR's gene
   resolver may not accept it directly; the fixture-generation script
   (`run_20gene_crispor_fixtures.sh`) feeds CRISPOR a *FASTA slice of
   the locus*, not a gene symbol, so this is a non-issue.

## Hard-gate interpretation

PRD §4.3 hard gates per the harness:

| # | Gate | Bar | Interpretation |
|---|---|---|---|
| 1 | Top-20 agreement | ≥ 18/20 shared | Hard. |
| 2 | Top-50 Spearman ρ | ≥ 0.85 | Hard. |
| 3 | Off-target site set superset | bionpu ⊇ CRISPOR | Hard but **provisional** — see Tier 1 inheritance below. |
| 4 | Per-site CFD diff | < 0.01 | Hard. |
| 5 | Per-guide Doench rank corr | Spearman ρ ≥ 0.85 | **Soft** until PR-D Azimuth lands. |

## Tier 1 limitations propagated into the harness

The Tier 1 just-shipped status (`state/wave1/tier1_status.json` ➜
`limitations_documented`) inherits into this harness verbatim. The
following limitations are EXPECTED to bite some hard gates; they are
**known-limitation soft-XFAILs**, NOT regressions.

### L-1. Off-target scan is locus-scoped, not full-genome

**Source:** Tier 1 finding 4. The scan runs against the target locus
only; CRISPOR scans the entire reference. Therefore the bionpu off-
target site set is a **strict subset** of CRISPOR's, **failing** the
gate-3 superset check on every gene.

**Harness behaviour:** `test_off_target_site_set_superset[<gene>]` is
xfailed with the marker "off-target superset gate FAILS because Tier 1
off-target scan is locus-scoped". The status JSON's
`off_target_superset_pass_count` will be very small (bounded above by
the count of CRISPOR off-target sites that happen to land within the
target locus, typically ~1-5 per gene from the on-target site itself).

**Ungate when:** the full-genome off-target scan path lands. This
depends on
`bionpu/dispatch/_npu_artifacts/crispr_match_multitile_memtile/final.xclbin`
being vendored into `bionpu-public` (already silicon-validated under
`PRD-crispr-scan-on-xdna` C-M6, blocked on the parallel vendoring agent
finishing).

### L-2. Doench RS1 30-mer context truncation at locus boundaries

**Source:** Tier 1 finding 6. Guides whose 30-mer Doench context window
extends past the locus slice get an `on_target_score = 0.0`. For BRCA1
this is ~24 guides per run; for compact genes (HBB ~4 kbp; EMX1 ~1
kbp) the affected fraction is much higher (any guide within ~30 nt of
either end is dropped to 0).

**Impact on gates:** Pulls the per-guide-Doench rank correlation down
when truncated guides leak into the top-50 set. Compact genes (EMX1,
HBB, MYC, AAVS1) are the worst-affected.

**Harness behaviour:** `test_zzz_emit_status_json` records a
`doench_rs1_boundary_truncation:N_guides` flag in
`per_gene[<gene>].limitations_in_play` whenever any top-50 guide has
on-target score 0.0.

**Ungate when:** Tier 2 widens the locus slice (or N-pads the FASTA
slice) before the Doench RS1 context lookup. Tracked as a Tier 1
follow-up in `state/wave1/tier1_status.json`.

### L-3. Doench RS1 ≠ CRISPOR's Doench RS2 / Azimuth

**Source:** PRD §4.3 v0.2 status note. CRISPOR uses the Azimuth
gradient-boosted regressor (Doench Rule Set 2); bionpu ships RS1
(logistic regression).  Per-guide raw scores diverge by O(0.05–0.15).

**Harness behaviour:** `test_per_guide_doench_rank_correlation_soft`
is interpreted as a SOFT gate; failure xfails with
"PR-D Azimuth lands the hard gate." Status JSON records the
rank-correlation value separately under
`hard_gate_summary.doench_rank_corr_soft_pass_count`.

**Ungate when:** PR-D lands per `restrictive-license-model-policy.md`
(consume Azimuth as external runtime dep; long-term retrain).

### L-4. NPU device path multi-batch dispatch is future work

**Source:** Tier 1 finding 5 (artifact gate) + harness landing
finding (multi-batch gate).

The Tier 1 status JSON listed two NPU-side gates:
1. xclbins not vendored under `_npu_artifacts/crispr_*`.
2. `xrt-smi examine` reports 0 devices (NPU runtime suspended).

The harness landing surfaced a third:

3. **N_GUIDES per-launch hard cap.** `bionpu.scan.npu_scan` raises
   `NotImplementedError("v0.2 supports at most 128 guides per launch;
   got <N>. Multi-batch dispatch is future work.")` whenever a locus
   has > 128 candidate guides. Empirically every locus in the 20-gene
   set has 260 — 8968 candidates (BRCA1: 8968, BRCA2: 7482, EMX1: 260,
   smallest in the set). **No gene in the 20-gene set fits the v0.2
   single-launch NPU path.**

   Verified locally that `_npu_artifacts/crispr_pam_filter_early/
   final.xclbin` IS present in the bionpu-public checkout (contradicting
   the Tier 1 status JSON's "xclbins not vendored" note); the gate is
   the multi-batch dispatch missing piece, not the artifact.

**Harness behaviour:**
`test_npu_device_path_artifact_gated[<gene>]` is an XFAIL with
`NotImplementedError` exception captured for every gene (and for
artifact-missing / xrt-smi-suspended cases too).

**Ungate when:** any of:
- multi-batch dispatch lands (`bionpu.scan.npu_scan` chunks the guide
  set into N_GUIDES-sized batches and merges results); OR
- the per-locus candidate count is brought below N_GUIDES (e.g. by
  pre-filtering by GC%, but that destroys the comparison shape vs
  CRISPOR).

The CPU dispatch path is the canonical hard-gate measurement until
multi-batch lands.

## Per-gene deltas

(empty until CRISPOR fixtures are generated.)

For each gene where a CRISPOR fixture is later generated and the
harness reports a delta, append a section here keyed by gene + gate:

```
### <gene> — <gate>

- **Observed:** <value>
- **Expected:** <bar>
- **Hypothesised cause:** <one-liner>
- **Action:** <leave-as-known-limitation | file-followup-issue | …>
```

Genes with EXPECTED known-limitation deltas (no per-gene entry needed
unless the delta is in addition to the inherited limitation):

- All 20 genes: gate 3 (off-target superset) — L-1.
- Compact-gene cluster (EMX1, HBB, MYC, AAVS1, KRAS): gate 2 (Spearman),
  gate 5 (Doench rank corr) — L-2.
- All 20 genes: gate 5 (per-guide Doench) — L-3.

## Coordinate / annotation notes

The 20-gene fixture (`genes_pinned.json`) uses NCBI RefSeq Annotation
Release 110 / GRCh38.p14 coordinates as the canonical source. CRISPOR's
default RefSeq pin is **109**. Genes with known coordinate-version
sensitivity:

- `RUNX1` and `TET2` are very large (~150 kbp and 1.27 Mb respectively);
  exact-end coordinates differ between RefSeq 109 and 110 by tens of
  kbp at the 3′ end. Top-20 agreement should be unaffected (canonical
  guides sit in the body of the gene); off-target counts can drift.
- `AAVS1` is not a HGNC gene; CRISPOR may resolve it via UCSC's table-
  browser knowledge of the locus.

If a per-gene boundary mismatch is observed, append a `### <gene> —
coordinate annotation` section to the per-gene deltas above.

## Provenance

- Fixture-author commit: see `state/wave1/twenty_gene_validation_status.json`
  field `git_commit_repo` at run time.
- bionpu output regenerator: `bionpu-public/tools/run_20gene_validation.py`.
- CRISPOR fixture regenerator: `bionpu-public/tools/run_20gene_crispor_fixtures.sh`.
- Status emitter: `bionpu-public/tests/test_crispr_design_e2e_validation.py::test_zzz_emit_status_json`.
