---
# Structured front-matter for mechanical lint checks ( harness will validate
# that every metric below has unit + formula + scope + percentile_method where
# applicable). Do not remove keys; add new metrics by appending entries.
spec_version: "1.0.0"
spec_purpose: "Lock measurement units before any track measures anything (umbrella PRD §4.2; risk row 'apples-to-oranges measurements')."
generated: "2026-04-25"
owner: ""
consumed_by: ["", "", "", "", "", ""]
metrics:
  - name: throughput_samples_per_sec
    unit: "samples/sec"
    scope: ["basecalling"]
    formula: "raw_signal_samples_processed / wall_clock_seconds_in_timed_window"
    notes: "Numerator counted at the chunker input boundary (POD5 read → fixed-size signal samples). Denominator is the timed-run wrapper window (see harness)."
  - name: throughput_bp_per_sec
    unit: "bp/sec"
    scope: ["basecalling"]
    formula: "total_emitted_bases_in_FASTQ / wall_clock_seconds_in_timed_window"
    notes: "Numerator counted at the FASTQ writer; counts only basecalled letters (ACGTN), not quality-score chars."
  - name: throughput_guides_per_sec
    unit: "guides/sec"
    scope: ["crispr"]
    formula: "candidate_guides_scanned / wall_clock_seconds_in_timed_window"
    notes: "A guide is fully scanned only when every window in the configured genome (or genome subset) has been compared against it."
  - name: throughput_sites_per_sec
    unit: "sites/sec"
    scope: ["crispr"]
    formula: "candidate_sites_scored / wall_clock_seconds_in_timed_window"
    notes: "A site is one (guide, genome_position, strand) tuple that survived the match filter and reached the scoring path."
  - name: accuracy_modal
    unit: "percentage_points"
    scope: ["basecalling"]
    formula: "modal_accuracy = mode of (per-read alignment identity), per Dorado/ONT convention; computed via minimap2 alignment to GRCh38 + paftools.js stat (samstats-style)"
    notes: "Cite minimap2 commit + paftools version in measurement passport. PRD basecalling §3.2 within-0.5pp gate evaluated against this metric."
  - name: accuracy_indel_rate
    unit: "ratio"
    scope: ["basecalling"]
    formula: "indel_rate = (sum CIGAR I + sum CIGAR D) / sum CIGAR M"
    notes: "PRD basecalling §3.2 within-10%-relative gate evaluated against this metric."
  - name: accuracy_crispr_set_equality
    unit: "boolean"
    scope: ["crispr"]
    formula: "set(scan_output_after_normalize) == set(reference_output_after_normalize)"
    notes: "Normalizer: bionpu.data.canonical_sites.normalize from (sort by chrom, start, mismatch_count, guide_id, strand). Equality is byte-for-byte after normalization."
  - name: energy_per_megabase
    unit: "J/Mbp"
    scope: ["basecalling"]
    formula: "(end_energy_joules - start_energy_joules) / (total_emitted_bases_in_FASTQ / 1e6)"
    notes: "Energy reader documented in POWER_DOMAINS.md per device. Integration window = harness timed-run wrapper boundaries (see §Energy below)."
  - name: energy_per_guide_genome
    unit: "J/(guide·genome)"
    scope: ["crispr"]
    formula: "(end_energy_joules - start_energy_joules) / (n_guides * n_haploid_basepairs_scanned)"
    notes: "n_haploid_basepairs_scanned is documented in the run config (e.g. GRCh38 primary chromosomes = 3.0e9). PRD CRISPR §3.4 energy claim evaluated here."
  - name: latency_p50
    unit: "ms"
    scope: ["both"]
    formula: "nearest-rank percentile of per-chunk wall-clock; p50 = value at index ceil(0.50 * N) - 1 in 0-indexed sorted samples"
    notes: "Nearest-rank, NOT linear interpolation. See §Latency."
  - name: latency_p95
    unit: "ms"
    scope: ["both"]
    formula: "nearest-rank: value at index ceil(0.95 * N) - 1"
  - name: latency_p99
    unit: "ms"
    scope: ["both"]
    formula: "nearest-rank: value at index ceil(0.99 * N) - 1"
  - name: latency_max
    unit: "ms"
    scope: ["both"]
    formula: "max(per_chunk_wall_clock_ms)"
  - name: memory_rss
    unit: "bytes"
    scope: ["both"]
    formula: "max sampled value of /proc/<pid>/status:VmRSS over the timed window (Linux); polled at >= 10 Hz"
  - name: memory_vram_peak
    unit: "bytes"
    scope: ["both"]
    formula: "torch.cuda.max_memory_allocated(device) at end_of_window minus baseline at start_of_window; reset via torch.cuda.reset_peak_memory_stats() before window starts"
  - name: memory_tile_used_bytes
    unit: "bytes"
    scope: ["both"]
    formula: "static AIE2P tile-local memory footprint reported by the IRON/Peano build artifact for the lowered kernel; sum across active tiles"
    notes: "Static footprint, not runtime peak. Records what the compiled kernel statically allocates per tile."
---

# UNITS — Locked Benchmarking Metric Definitions

This document fixes every metric unit and computation rule that the bench
harness, the energy readers, the basecalling track, and the
CRISPR track must conform to. It exists to discharge umbrella PRD §4.2's
requirement that cross-track numbers be comparable, and to neutralize the
"apples-to-oranges measurements" risk row in the umbrella PRD risk table.

If a number is reported in this repo without conforming to this file, the
writeup pipeline MUST refuse to render it.

Companion document: [`POWER_DOMAINS.md`](./POWER_DOMAINS.md) — what each energy
reading covers, per device.

---

## Quick reference table

| Metric                   | Unit               | Formula (numerator / denominator)                                                                 | Basecalling | CRISPR |
|--------------------------|--------------------|---------------------------------------------------------------------------------------------------|:-----------:|:------:|
| Throughput (signal)      | `samples/sec`      | raw_signal_samples_processed / wall_clock_seconds                                                 | yes         |        |
| Throughput (bases)       | `bp/sec`           | total_emitted_bases_in_FASTQ / wall_clock_seconds                                                 | yes         |        |
| Throughput (guides)      | `guides/sec`       | candidate_guides_fully_scanned / wall_clock_seconds                                               |             | yes    |
| Throughput (sites)       | `sites/sec`        | candidate_sites_scored / wall_clock_seconds                                                       |             | yes    |
| Accuracy (basecall)      | `pp` (modal acc.)  | mode of per-read alignment identity (minimap2 → GRCh38, paftools samstats)                        | yes         |        |
| Indel rate               | ratio              | (CIGAR I + CIGAR D) / CIGAR M                                                                     | yes         |        |
| Accuracy (CRISPR)        | boolean            | set-equality after `bionpu.data.canonical_sites.normalize`                                        |             | yes    |
| Energy (basecalling)     | `J/Mbp`            | ΔJoules / (bases_emitted / 1e6)                                                                   | yes         |        |
| Energy (CRISPR)          | `J/(guide·genome)` | ΔJoules / (n_guides × n_haploid_bp_scanned)                                                       |             | yes    |
| Latency p50/p95/p99/max  | `ms`               | nearest-rank percentile of per-chunk wall-clock                                                   | yes         | yes    |
| Memory RSS               | `bytes`            | max VmRSS sampled over window                                                                     | yes         | yes    |
| VRAM peak                | `bytes`            | `torch.cuda.max_memory_allocated` at end - reset baseline at start                                | yes         | yes    |
| Tile memory used         | `bytes`            | static IRON/Peano tile allocation, summed across active tiles                                     | yes         | yes    |

---

## 1. Throughput

Throughput is always **work-units per wall-clock second** measured inside the
harness's timed-run wrapper. The wrapper records two timestamps —
`t_start` and `t_end` — using `time.monotonic_ns()` and persists both into
`measurements.json` so any consumer can recompute the rate.

### 1.1 `samples/sec` (basecalling)

- **Numerator:** raw nanopore signal samples (16-bit ints from POD5) that
  entered the chunker during the timed window.
- **Denominator:** `t_end - t_start` in seconds.
- **Where measured:** at the chunker input boundary (so chunk overlap does NOT
  inflate the count — overlap regions are not double-counted).

### 1.2 `bp/sec` (basecalling)

- **Numerator:** total emitted bases in the FASTQ output. Counts ACGT and N;
  does **not** count quality scores, line breaks, or read-name characters.
- **Denominator:** `t_end - t_start` in seconds.
- **Where measured:** at the FASTQ writer.

### 1.3 `guides/sec` (CRISPR)

- **Numerator:** candidate guides fully scanned. A guide is "fully scanned"
  when every window in the configured genome (or genome subset declared in the
  run config) has been compared against it. Partial scans are NOT credited.
- **Denominator:** `t_end - t_start` in seconds.

### 1.4 `sites/sec` (CRISPR)

- **Numerator:** candidate sites that survived the match filter and reached
  the scoring path. A site is a `(guide_id, chrom, start, strand)` tuple.
- **Denominator:** `t_end - t_start` in seconds.
- **Note:** if the scoring path is not invoked (scan-only run), this metric is
  emitted as `null`, never `0`.

---

## 2. Accuracy

### 2.1 Basecalling: modal accuracy

PRD basecalling §3.2 names two gates: modal accuracy within 0.5 pp of the GPU
reference, and indel rate within 10% relative. Both are computed from a
minimap2 alignment of the FASTQ to GRCh38 (or whatever reference the run
config names; default GRCh38).

- **Tool:** `minimap2 -ax map-ont` followed by `paftools.js stat` (or
  `samtools stats` parsed for `error rate`/`average length`; the harness
  records which tool ran with its version SHA).
- **Modal accuracy formula:** mode (most-common value) of per-read alignment
  identity, where per-read identity = `matches / (matches + mismatches +
  insertions + deletions)` from the alignment record. This matches Dorado's
  reporting convention (and ONT's documentation around `dorado summary`).
- **Indel rate formula:** sum across all alignments of `CIGAR I + CIGAR D`,
  divided by sum of `CIGAR M`. Reported as a unitless ratio.
- **PRD §3.2 evaluation:** chosen production config (FP16 or INT8, ratified at
  ) MUST be within 0.5 pp absolute of the GPU FP16 reference modal
  accuracy, AND its indel rate MUST be within 10% relative of the GPU FP16
  reference indel rate. Failing either is a documented finding, not a silent
  pass.

### 2.2 CRISPR: exact set-equality after canonical normalization

PRD CRISPR §3.2 requires "matches Cas-OFFinder's output exactly. No false
negatives. False positives only if explainable."

- **Comparison:** byte-for-byte equality of the **normalized** output file
  against `tracks/crispr/reference/casoffinder-canonical.normalized.tsv` from
  . Normalization is performed by `bionpu.data.canonical_sites.normalize`
; see also `tracks/crispr/fixtures/FIXTURE-A.md` for the canonical
  fixture parameters.
- **Sort key (canonical):** `(chrom, start, mismatch_count, guide_id,
  strand)`.
- **Idempotence:** normalize ∘ normalize == normalize. asserts this.
- **Equality semantics:** set-equality of records, but because the normalizer
  produces a stable total order, byte-for-byte equality of the serialized TSV
  is the implementation form.
- **No-false-negatives clause:** if the NPU output is a strict superset of the
  reference, the run is reported as **fail with diagnostic** (extra rows
  printed in the report); it is not silently passed.

---

## 3. Energy

Energy is computed as `(end_energy_joules - start_energy_joules) /
work_units`. The Joules reading comes from the device-specific reader in
`bionpu/bench/energy/`. What each domain
covers — and where readings might be unavailable on this kernel — lives in
[`POWER_DOMAINS.md`](./POWER_DOMAINS.md).

### 3.1 Integration window

The energy integration window is the **timed-run wrapper window from the
harness**:

- **Start (`t_start_e`):** captured *immediately before* the first work-unit
  enters the device under test. For basecalling, this is the first chunker
  push; for CRISPR, the first window streamed to the match kernel.
- **End (`t_end_e`):** captured *immediately after* the last work-unit's
  output is committed (FASTQ flushed for basecalling; emit-buffer drained for
  CRISPR).
- **Warmup is excluded.** Each run includes a warmup phase (default: 10% of
  total work or 5 seconds, whichever is greater) which is timed but NOT
  counted toward the energy or throughput integration window.

The harness writes both `t_start_e` and `t_end_e` into `measurements.json` so
energy can be recomputed from the raw counter samples post-hoc.

### 3.2 `J/Mbp` (basecalling)

- **Numerator:** ΔJoules over the integration window for the device under
  test (CPU package, GPU board, NPU subdomain — see `POWER_DOMAINS.md`).
- **Denominator:** total emitted bases in FASTQ during the window, divided by
  `1e6`.
- **Comparison rule:** energy comparisons across devices are only valid with
  the caveats in `POWER_DOMAINS.md` (different devices measure different
  rails). enforces this in figure captions.

### 3.3 `J/(guide·genome)` (CRISPR)

- **Numerator:** ΔJoules over the integration window.
- **Denominator:** `n_guides * n_haploid_basepairs_scanned`. The
  `n_haploid_basepairs_scanned` is read from the run config; for GRCh38
  primary autosomes + chrX + chrY this is 3.0e9. The harness records the
  exact value used so a reviewer can reproduce.
- **PRD CRISPR §3.4 evaluation:** the NPU figure must be lower than the GPU
  figure to claim the energy thesis. Otherwise the negative result is
  reported per umbrella PRD §3.4.

---

## 4. Latency

Latency is per **chunk** (basecalling) or per **batch** (CRISPR), measured
from chunk-enter to chunk-exit on the timed device. Stored as a sorted array
of millisecond floats per run.

### 4.1 Percentile interpolation method

**Nearest-rank.** Given N samples sorted ascending and percentile p ∈ (0, 1):

```
index = ceil(p * N) - 1     # 0-indexed
percentile_value = sorted_samples[index]
```

NOT linear interpolation. Rationale: nearest-rank is what `numpy.percentile(...,
method='nearest')` returns and what tail-latency dashboards (Grafana, p99
SLOs) typically display. We pick one and stick with it; mixing methods across
tracks is exactly the apples-to-oranges trap this document closes.

The harness's percentile helper (`bionpu/bench/units.py` to land in ) MUST
hard-code `method='nearest'` and refuse a `method=` override.

### 4.2 Reported percentiles

Always report **p50, p95, p99, max**. p99 + max together expose tail
behavior; p50 + p95 cover the common case. Reporting only mean is
disallowed — means hide bimodal tail behavior.

### 4.3 `max`

`max = sorted_samples[-1]`. Reported alongside the percentile triple. Useful
because nearest-rank p99 on N < 100 samples == max, which is misleading; max
is reported separately so the consumer notices.

---

## 5. Memory

### 5.1 `RSS` (process resident set size)

- **Source:** `/proc/<pid>/status`, field `VmRSS`, polled at ≥10 Hz during the
  timed window.
- **Reported value:** maximum sampled value (peak) in bytes.
- **Includes:** the bench harness, any forked workers, and the PyTorch /
  ONNX-Runtime / IRON runtime allocations.

### 5.2 `VRAM_peak`

- **Source:** `torch.cuda.max_memory_allocated(device)`.
- **Reset:** `torch.cuda.reset_peak_memory_stats()` is called at `t_start`.
- **Reported value:** the value at `t_end` minus the value at `t_start`,
  in bytes.
- **Caveat:** this counts only PyTorch-managed allocations. Non-Torch CUDA
  allocations (e.g. raw CuPy buffers) are NOT included; the harness emits a
  warning when it detects non-Torch CUDA contexts.

### 5.3 `tile_memory_used_bytes` (NPU)

- **Source:** static analysis of the IRON/Peano-compiled kernel artifact for
  AIE2P tile-local memory.
- **Reported value:** sum of static tile-local allocations across active
  tiles, in bytes. This is the **static footprint**, not a runtime peak.
- **Why static:** AIE2P tile memory is statically partitioned at compile
  time; runtime peak == static footprint for correctly-compiled kernels.
- **Per-tile breakdown:** the harness optionally records `tile_memory_per_tile`
  (a `dict[tile_id, bytes]`) for kernels that occupy >1 tile.

---

## 6. Cross-references

- `bionpu.data.canonical_sites.normalize` — . CRISPR accuracy comparison
  uses this normalizer; do not re-implement.
- `bionpu/bench/POWER_DOMAINS.md` — companion document. Required reading
  before interpreting any energy figure.
- `tracks/basecalling/fixtures/` — smoke fixtures for basecalling.
- `tracks/crispr/fixtures/FIXTURE-A.md` — canonical CRISPR fixture.
- Umbrella PRD `PRDs/PRD-bio-on-xdna.md` §4.2 — "every measurement uses the
  same harness so cross-track numbers are comparable."
- Basecalling PRD `PRDs/PRD-basecalling-on-xdna.md` §3 — success criteria.
- CRISPR PRD `PRDs/PRD-crispr-scan-on-xdna.md` §3 — success criteria.

## 7. Compliance

- (bench harness): `bionpu/bench/units.py` exports the formulas above as
  pure functions; the harness writes only metric names that appear in the
  front-matter `metrics:` list of this file. Any metric not listed here is a
  CI failure.
- (writeup pipeline): figure captions for energy MUST cite
  `POWER_DOMAINS.md` (not this file) for cross-domain caveats. Throughput,
  accuracy, latency, and memory captions cite this file.
- Both tracks (basecalling, CRISPR) must conform; deviations are documented in
  the run's measurement passport, not silently allowed.
