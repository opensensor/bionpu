# Strategic finding: AIE2P fits streaming genomics primitives

**Date:** 2026-04-28

## Summary

AIE2P is a poor fit for recurrent FP32 genomics pipelines such as
Dorado-style basecalling, but it is a strong fit for streaming genomics
primitives built from byte-level state, local windows, threshold tests,
and sparse emits.

The evidence is now broader than a single CRISPR scanner. Five
independent primitives have silicon-validated the same programming
toolkit across three algorithmic shapes:

| Primitive | Algorithmic shape | chr22 validation | chr22 e2e |
| --- | --- | --- | ---: |
| `kmer_count` v1.x | counting + canonical 2-bit math | top-1000 byte-equal; canonical=0 exact after v1.2 fix | 10.18 s |
| `minimizer` v0/v1 | sliding-window comparator | partial: 14.26 M / 20.01 M emits (71%); cap-fire filed | 50-365 s |
| `seed_extend` v0 | host hash lookup over silicon minimizers | 100 Kbp chr22 self-mapping smoke | 0.3 s lookup |
| `primer_scan` v0 | substring / adapter exact match | smoke 0/0; synthetic 2/2 byte-equal; chr22 path built | 2.05 s |
| `cpg_island` v0 | windowed multi-counter stats + threshold | 1,352 islands byte-equal; 1,373,708 candidates | 3.80 s |

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

The easy cascade primitives are useful but not shape-novel:

- restriction-site finder: `primer_scan` shape;
- methylation context `CG/CHH/CHG`: local base-context classifier;
- GC% windowed scanner: `cpg_island` counter subset;
- N-base detector: trivial threshold / sparse emit.

The most valuable next research primitive is a tandem-repeat caller.
It would validate a fourth shape: periodicity detection. A conservative
v0 should avoid FFT and start with exact short tandem repeats:

- motif length 1..6;
- emit `(start, end, period, repeat_count, motif)`;
- synthetic disease-style fixtures: `CAG`, `CGG`, `GAA`, `ATTTC`;
- chr22 scan against the CPU oracle after synthetic byte equality.

This extends the empirical foundation from five primitives / three
shapes to six primitives / four shapes.

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
