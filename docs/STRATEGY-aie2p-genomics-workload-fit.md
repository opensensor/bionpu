# Strategic finding: AIE2P fits streaming genomics primitives

**Date:** 2026-04-28

## Summary

AIE2P is a poor fit for recurrent FP32 genomics pipelines such as
Dorado-style basecalling, but it is a strong fit for streaming genomics
primitives built from byte-level state, local windows, threshold tests,
and sparse emits.

The evidence is now broader than a single CRISPR scanner. Seven
independent primitives have exercised the same programming toolkit
across four algorithmic shapes, plus one product-facing composition.

| Primitive | Algorithmic shape | chr22 validation | chr22 e2e |
| --- | --- | --- | ---: |
| `kmer_count` v1.x | counting + canonical 2-bit math | top-1000 byte-equal; canonical=0 exact after v1.2 fix | 10.18 s |
| `minimizer` v0/v1 | sliding-window comparator | partial: 14.26 M / 20.01 M emits (71%); cap-fire filed | 50-365 s |
| `seed_extend` v0 | host hash lookup over silicon minimizers | 100 Kbp chr22 self-mapping smoke | 0.3 s lookup |
| `primer_scan` v0 | substring / adapter exact match | smoke 0/0; synthetic 2/2 byte-equal; chr22 path built | 2.05 s |
| `cpg_island` v0 | windowed multi-counter stats + threshold | 1,352 islands byte-equal; 1,373,708 candidates | 3.80 s |
| `tandem_repeat` v0 | period detection / autocorrelation | chr22 silicon run; 289,548 records with 3 host-merge edge diffs | 9.4 s |
| `methylation_context` v0/v1 | local base-context classifier | chr22 record-equal with c1024; 18.41 M records | 154.77 s |
| `bionpu trim` v0/v1 | toolkit-to-tool composition | cutadapt byte-equal on 4/4 cross-checks | workload-dependent |

The common shape is:

1. packed 2-bit DNA streamed through AIE2P tiles;
2. small rolling state in tile memory;
3. integer or byte-level arithmetic;
4. per-position/window thresholding;
5. sparse length-prefixed emits;
6. host post-pass for aggregation, sorting, merging, or lookup.

This is the design center to keep using.

## Why This Fits AIE2P

AIE2P has a 6x8 grid of CoreTiles, explicit ObjectFifo dataflow, 64 KiB
per-tile data memory, memtile fan-in/fan-out, and strong integer/bf16
vector support. These features line up with genomics kernels that are:

- **streaming:** scan once over packed DNA;
- **local-state:** keep only rolling counters, packed k-mers, or a small
  comparator ring;
- **independent by position/window:** no recurrent dependency chain;
- **integer-exact:** no FP32 sensitivity or accumulated numerical drift;
- **sparse-output:** emit only hits/candidates, then let the host merge.

They do not line up with production basecaller LSTM stacks: those are
recurrent, FP32-trained, precision-sensitive, and hard to parallelize
across timesteps. Isolated bf16-friendly blocks can be useful, but the
end-to-end basecaller shape fights the hardware.

## Primitive Evidence

### `kmer_count`

`kmer_count` established the canonical 2-bit math, chunk ownership,
multi-pass partitioning, sparse emit, and host merge pattern. The v1.x
line reached chr22 top-1000 byte equality and fixed the canonical=0
centromere case exactly. The latest chr22 result is 10.18 s after
batched dispatch, sort/RLE merge, and pipelined host runner work.

The remaining ceiling is dispatch/DMA overhead rather than algorithmic
correctness.

### `minimizer`

`minimizer` extended the toolkit from counting to sliding-window
selection. It validates rolling forward/reverse canonical math plus a
window comparator/ring state, but full chr22 remains cap-limited:
14.26 M / 20.01 M emits at the primary chr22 setting. This is a useful
negative result: the shape fits, but dense real-DNA chunks need either a
wider sparse output slot or smaller chunks.

The cap-fire is filed and bounded in
`src/bionpu/kernels/genomics/minimizer/gaps.yaml`.

### `seed_extend`

`seed_extend` demonstrates composition: reuse the silicon minimizer to
produce query seeds, then do the reference-index lookup on host. That is
the right division of labor. AIE2P does the streaming per-byte
canonical computation; the CPU owns the large reference hash table and
candidate lookup.

The current v0 validates 100 Kbp chr22 self-mapping with a 0.3 s lookup
and includes a frequency cutoff to avoid repeat-driven seed explosions.

### `primer_scan`

`primer_scan` is the minimal substring-match primitive. It reuses the
same packed-DNA stream and sparse emit geometry, but changes the tile
state to a rolling primer-length equality check. Runtime primer
canonicals live in the chunk header, so one xclbin per primer length can
scan arbitrary primers/adapters.

This primitive is the natural basis for product-facing adapter trimming
or restriction-site scanning.

### `cpg_island`

`cpg_island` validates a windowed statistics shape: rolling `C`, `G`,
and `CG` counters with fixed-point threshold tests. The tile emits raw
candidate window starts; the host merges contiguous candidate runs into
CpG island intervals.

Full chr22 validation:

- fixture: `tracks/genomics/fixtures/chr22.2bit.bin`
- bases: 50,818,468
- CPU oracle: 1,352 islands in 15.008 s
- NPU: 1,352 islands in 3.800 s
- candidate starts: 1,373,708
- byte-equal: true

Measurement file:
`results/cpg/v0-chr22/measurements.json`.

### `tandem_repeat`

`tandem_repeat` validates the fourth algorithmic shape: exact
periodicity detection. The tile checks short periods and emits STR
records for host-side sorting/dedup. This is the first primitive in the
cascade whose core logic is not just canonical math, substring compare,
or rolling counters.

Full chr22 validation produced 289,548 records in 9.4 s with three
documented host-merge edge differences. That makes the shape useful as
research evidence even though v1 should tighten chunk-boundary merge
semantics before it becomes a production caller.

### `methylation_context`

`methylation_context` validates a local base-context classifier:
`CG`, `CHG`, and `CHH` contexts are emitted on both strands, with
minus-strand cytosines represented as forward-reference `G` positions.
The silicon path is byte-equal to the CPU oracle on mixed synthetic
smoke tests and all-A negative controls.

Full chr22 validation first exposed a deliberate boundary test:

- fixture: `tracks/genomics/fixtures/chr22.2bit.bin`
- bases: 50,818,468
- CPU oracle: 18,406,838 context records in 21.318 s
- NPU c4096: 9,840,462 context records in 23.277 s wall
- host-reported dispatch: 813.408 us average over 3,108 chunks
- record recovery: 53.46%
- byte-equal: false, due to `MC_MAX_EMIT_IDX=4094` cap-fire

Context counts:

| context | NPU | CPU oracle |
| --- | ---: | ---: |
| `CG` | 664,817 | 1,269,292 |
| `CHG` | 2,291,284 | 4,336,761 |
| `CHH` | 6,884,361 | 12,800,785 |

Measurement file:
`results/methylation_context/v0-chr22/measurements.json`.

This is an important correction to the naive "sparse emit" assumption:
per-base methylation contexts are dense in real DNA. The classifier
fits AIE2P, but the output contract needs a v1 mode before full-genome
record emission can be byte-equal. Good v1 options are a wider output
slot, smaller chunks, context-filtered scans, or count/window aggregate
mode.

The record-safe fix is smaller chunks, not changed semantics. A c2048
variant recovered 17,839,396 / 18,406,838 records (96.9%) but still
cap-fired in dense chunks. The c1024 artifact removes cap-fire on chr22:

- NPU c1024: 18,406,838 context records in 154.771 s wall
- chunks: 12,505
- host-reported dispatch: 386.744 us average
- record recovery: 100.00%
- byte-equal: true, record-by-record

Measurement file:
`results/methylation_context/v1-c1024-chr22/measurements.json`.

Batched c1024 variants preserve record equality but do not materially
change wall time:

| artifact | batches | avg dispatch | wall |
| --- | ---: | ---: | ---: |
| `n4_c1024` | 12,505 | 386.744 us | 154.771 s |
| `n4_b4_c1024` | 3,127 | 1,057.99 us | 150.818 s |
| `n4_b8_c1024` | 1,564 | 1,778.95 us | 150.795 s |

The `n4_b8_c1024` host runner can now filter chunk overlaps into strict
owned intervals and skip global sort/unique once records are monotonic.
That preserved full chr22 record equality but did not materially move
wall time, localizing the remaining bottleneck to output
materialization/write of 18.4 M dense records.

## Toolkit Pattern

The recurring implementation pattern is now stable:

| Layer | Reused pattern |
| --- | --- |
| Input | MSB-first packed 2-bit DNA |
| Chunking | 8-byte or small in-band header with actual bytes + ownership offset |
| Tile math | rolling canonical/counter/comparator state |
| Parallelism | multi-tile broadcast or fan-out by chunk/work slice |
| Output | 32 KiB sparse output slot with count prefix |
| Host post-pass | sort, dedup, RLE, interval merge, or hash lookup |
| Validation | CPU oracle first, silicon byte-equality second |

The host/silicon split is important. Silicon should do the per-byte
streaming math. Host code should own large dictionaries, interval
composition, dynamic programming, and product I/O.

## What To Build Next

The easy cascade primitives are useful but mostly not shape-novel:

- restriction-site finder: `primer_scan` shape;
- GC% windowed scanner: `cpg_island` counter subset;
- N-base detector: trivial threshold / sparse emit.

The most valuable next work is no longer another trivial primitive. It
is either:

- fix dense-output primitives (`methylation_context`, `minimizer`) with
  wider output or alternate aggregate modes;
- turn `bionpu trim` batching into a production throughput result; or
- write up the seven-primitive/four-shape empirical foundation.

## Boundary Conditions

Do not use these results to claim that all genomics workloads fit
AIE2P. The fit is shape-specific.

Good fits:

- packed DNA streaming;
- local counters/comparators;
- exact integer decisions;
- sparse emits;
- host aggregation.

Poor fits:

- recurrent FP32 models;
- large mutable state per timestep;
- precision-sensitive CTC-style decoding;
- giant resident indexes that belong in host memory.

The current evidence supports a focused claim: AIE2P is effective for
streaming, exact, sparse-output genomics primitives, and the
implementation toolkit is reusable across multiple biologically useful
tasks.
