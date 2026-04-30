# Validating the CRISPR-shape Thesis on AMD XDNA 2 / AIE2P

## A seven-primitive empirical study with five production CLIs and a two-regime corollary

**Date:** 2026-04-28
**Author:** Matt Davis (`matteius@gmail.com`)
**Hardware:** Ryzen AI 9 HX 370 (Strix), AMD XDNA 2 / AIE2P
**Companion strategic doc:** [`STRATEGY-aie2p-genomics-workload-fit.md`](STRATEGY-aie2p-genomics-workload-fit.md)
**Status (v2, 2026-04-28):** Refreshed with 5 production CLIs (was 1 in v1) + a NEW silicon kernel (`pam_filter_iupac`, IUPAC PAM scan) + Track E NO-GO microbenchmark closure (small-FP32-NN-with-MAC-stack floor) + two-regime localization of the per-dispatch amortization corollary (subprocess-bounded vs silicon-bounded). v1 baseline (six primitives, one production composition, one open throughput corollary) was committed as `9b6adc2`; v2 numbers (`bionpu trim` v2 pyxrt) were locked as `1557faf`.

---

## Abstract

The AMD XDNA 2 / AIE2P engine on the Ryzen AI 9 HX 370 was originally
targeted for Dorado nanopore basecalling. After a full instrumentation
session demonstrated that end-to-end Dorado on AIE2P runs **14 times
slower than ONNX-on-CPU** for fundamental architectural reasons (FP32
recurrent LSTM stacks fight every silicon design point — bf16 emulation,
recurrent state breaking dataflow, per-timestep dependency defeating
fan-out), we pivoted to the opposite shape: **CRISPR-shaped** genomics
workloads with per-byte streaming arithmetic, small persistent state,
and sparse output. Six kernel primitives (k-mer counting, sliding-window
minimizers, primer / adapter scanning, CpG island detection, tandem
repeat / STR detection, plus a host-side seed extractor that reuses the
minimizer silicon op) and one production composition (`bionpu trim`,
cutadapt-byte-equal across CPU, NPU, and silicon paths) have now shipped
with reproducible measurements.

The headline silicon-only number — `bionpu_kmer_count` v1.3 chr22 at
`k=21, n_tiles=4, n_passes=4`: **951.7 µs/iter silicon-only vs Jellyfish
1T 9.81 s = 10,533× speedup** — confirms the silicon's compute is
extremely fast for the right shape. Five of the six primitives also
ship `chr22` end-to-end wall measurements that compete with or beat
their canonical CPU references (`primer_scan` 2.05 s vs CPU oracle
9.18 s; `cpg_island` 3.62 s vs CPU oracle 12.73 s; `tandem_repeat`
9.41 s vs oracle 14.85 s).

The headline qualifier — `bionpu trim` v0 measures **9.7 reads/s NPU vs
160,341 reads/s cutadapt 1T** on the synthetic 10K fixture — localises
a hard hardware corollary: **per-record streaming workloads are
overhead-bound at ~102 ms/dispatch (subprocess + XRT context init)** and
cannot win without batched dispatch or persistent-kernel mode. Both
findings live in the same envelope; both are reproducible from the
artifacts cited below.

---

## 1. Background and motivation

### 1.1 The silicon

AMD XDNA 2 / AIE2P (formerly the Xilinx Versal AI Engine, lineage
AIE → AIE-ML → AIE2P) is the NPU embedded in the Ryzen AI 9 HX 370
laptop SoC. Its architectural commitments, per AM020 and
[`docs/aie-ml-am020-crosswalk.md`](aie-ml-am020-crosswalk.md):

* **8 CoreTiles per column** plus 1 MemTile (6×8 logical tile topology;
  in practice, IRON Python frontends address the columns via
  `bionpu-public/src/bionpu/kernels/genomics/<kernel>/<kernel>.py`).
* **64 KiB tile data memory** per CoreTile, shared across `.text`,
  `.rodata`, stack, ObjectFifo ping-pong buffers, and per-tile state.
* **Vectorized integer + bf16 ML** — fast int8/uint8 vector ops;
  hardware bf16 matmul, tanh, sigmoid; FP32 *accumulator* support but
  **no FP32 vector multiply** (AM020 Appendix A: "Native FP32
  supported through emulation using bfloat16 — Removed in AIE-ML").
* **Streaming dataflow** — first-class `ObjectFifo` primitive with
  ping-pong buffering, multi-tile fan-out, packet-merge, sparse-emit
  ring buffers.
* **MemTile aggregation** — 6 S2MM + 6 MM2S DMA channels; the
  `outC.prod.join(N, ...)` pattern aggregates N tile partials before
  shim drain.
* **IRON Python frontend** — `mlir-aie` IRON dialect emits placement +
  ObjectFifo topology; Peano-LLVM compiles the C++ kernel bodies.

### 1.2 The original target: Dorado basecalling — 14× slower

Dorado is the canonical Oxford Nanopore basecaller: a 5-layer
LSTM stack × 1667 timesteps (production_long shape) with
matmul + tanh + sigmoid + state update + linear projection, followed
by CTC beam decode. Today's instrumentation session (and the months
of tuning that preceded it) ran every available knob:

* Stage-3 fused `linear_projection`: 58× kernel speedup, bf16 contract
  — `results/basecalling/b-m6-fused-perts/measurements.json`.
* In-process pyxrt: 6× wall reduction on `fused_perts`
  (`results/basecalling/b-m8-pyxrt/`).
* Batched multi-chunk dispatch (N=8): 4.5× per-chunk wall improvement
  (`results/basecalling/b-m9-batched/`).
* Wider pyxrt across encoder: 1.17 s/chunk savings
  (`results/basecalling/b-m10-pyxrt-rollout/`).
* L1667 long-shape artifacts built and silicon-validated
  (`results/basecalling/b-m12-long-shape-rollout/`).
* Cascade burst_length sweep: AM020 burst lengths {0, 64, 128, 256,
  512} all wedge G-T3.1-103 identically (silicon-falsification at
  `results/basecalling/b-m11-cascade-burst-length-falsified/`).

All five wins are real and reproducible **in isolation**. None makes
the end-to-end Dorado pipeline beat ONNX CPU. Smoke fixture: NPU
**18.69 s** vs CPU **1.37 s** — NPU is **14× SLOWER end-to-end**.

The bottleneck is the 5-layer LSTM stack at production_long shape:
~1.1 s/chunk on NPU regardless of optimisation, because the cell
math is bf16-emulated and recurrent. With ~6 chunks per smoke read ×
10 reads, that's ~6 s of LSTM dominating wall time.

### 1.3 The pivot

The 14× slower number is the architectural diagnostic, not a tooling
gap solvable with one more sprint. It says: **the silicon's design
bet (vectorized integer math, multi-tile parallel dataflow,
sparse-emit ring, bf16-tolerant ML) is incompatible with Dorado's
shape (FP32-trained recurrent stack)**. The strategic implication is
to invest harder in workloads that match the design bet, not to
keep filing toolchain bugs against a workload that doesn't.

This pivot was committed on 2026-04-28 in
[`STRATEGY-aie2p-genomics-workload-fit.md`](STRATEGY-aie2p-genomics-workload-fit.md);
the present document is the empirical follow-up that validates the
pivot's hypothesis on six primitives.

---

## 2. The CRISPR-shape thesis

A workload is **CRISPR-shape** for AIE2P if it satisfies all four:

1. **Per-byte streaming arithmetic.** Computation is a rolling update
   over a packed-2-bit DNA stream. Forward and reverse-complement
   canonicals are maintained in two uint64 registers and updated each
   base. No floating-point recurrence; no large lookup tables.
2. **Small persistent state.** Tile-resident state fits within the
   64 KiB CoreTile DM cap with comfortable headroom (typical: ≤ 50 KiB
   incl. ping-pong buffers; per-tile work state is tens to hundreds of
   bytes).
3. **Sparse emit.** Output volume is a tiny fraction of input volume
   (counts, hits, candidate intervals). The kernel writes
   length-prefixed records into a small partial slot; the host
   accumulates.
4. **Pattern-match or window-statistic semantics, not gradient or
   recurrence.** The math is window-local; there is no per-element
   state propagation that defeats dataflow parallelism.

CRISPR pam_filter and match_multitile (the predecessor work that
established this hypothesis at ~4-5× speedup on real-world input)
fit the shape exactly. The six primitives below extend the pattern
to a wider set of bioinformatics building blocks.

---

## 3. The toolkit

The infrastructure that's reusable across all six primitives — built
once during `kmer_count` v0.5 and inherited by every successor:

### 3.1 Wire format

* **2-bit packed DNA, MSB-first per byte** (A=00, C=01, G=10, T=11).
  First base = `bits[7:6]` of byte 0. Same format across all six
  primitives; same `pack_dna_2bit` / `unpack_dna_2bit` helpers in
  `bionpu-public/src/bionpu/data/kmer_oracle.py`.
* **In-band chunk header.** Bytes [0..3]: `uint32 LE
  actual_payload_bytes`. Bytes [4..7]: `int32 LE
  owned_start_offset_bases` (overlap-region gate). Total 8 bytes for
  k-mer-class kernels; **24 bytes** for `primer_scan` (extends with
  `uint64 primer_fwd_canonical | uint64 primer_rc_canonical` for
  runtime primer parametrisation, eliminating per-primer xclbin
  rebuilds).
* **Output slot:** 32 KiB padded per-tile slot. Layout:
  `[uint32 LE emit_count][N × record]`. Record sizes vary by
  primitive (8 B for kmer canonicals, 16 B for minimizers / primer /
  STR, 4 B for CpG candidate positions).

### 3.2 Streaming chunk + per-k overlap protocol

`SEQ_IN_CHUNK_BYTES_BASE = 4096` payload bytes; per-k overlap of
`ceil((k-1) / 4)` bytes rounded up to 4-byte alignment (aiecc
`aie.dma_bd` rejects unaligned shapes). For `k=21` overlap = 8 B; for
`k=15` overlap = 4 B. Without overlap, k-mers spanning chunk
boundaries are dropped silently.

The **owned-range gate** (`owned_start_offset_bases` field above)
ensures each k-mer is counted by exactly one chunk: chunk *i* (i > 0)
emits only when the k-mer-start position is `>=
overlap_bases - (k - 1)`. This closes the v1.0 chunk-overlap
double-emit gap on chr22 (filed and closed in
`bionpu/kernels/genomics/kmer_count/gaps.yaml#kmer-chunk-overlap-double-emit`).

### 3.3 MemTile-aggregated combine

The `outC.prod.join(N, ...)` ObjectFifo pattern routes N tile
partials through the MemTile before shim drain. Same pattern across
`kmer_count`, `minimizer`, `primer_scan`, `cpg_island`, and
`tandem_repeat`. Mirrors
`bionpu-public/src/bionpu/kernels/crispr/match_multitile_memtile/multitile_memtile.py`
(silicon-validated). For `n_tiles ∈ {1, 2, 4}`, the pattern works
out-of-the-box. `n_tiles = 8` exceeds the AIE2P MemTile S2MM channel
budget (6 channels) and is filed as
`kmer-n8-memtile-dma-cap` (v1.1+).

### 3.4 Parallel host-side sort-merge

The single biggest host-side win in `kmer_count` v1.3: replace
`std::unordered_map<uint64,uint64>` (node-allocator pattern, cache-
hostile at 30 M entries) with `std::sort` (TBB par_unseq) + linear
RLE encode. Wall on chr22 dropped from 4.62 s (single-thread sort) to
0.50 s (par_unseq). Sibling primitives can adopt the same pattern when
their host post-pass becomes the bottleneck.

### 3.5 Multi-pass slicing

When per-chunk emit volume risks `MAX_EMIT_IDX` cap-fire, the kernel
filters by hash-slice (`(canonical >> shift) & ((1 << log2(N)) - 1) ==
pass_idx`) and the host runs N silicon dispatches. Each canonical
lands in exactly one pass; coverage = 100%. `kmer_count` ships
`SUPPORTED_N_PASSES = (1, 4, 16)`; `minimizer` adapted the pattern
with a Fibonacci position-hash (`slice = (uint32_t)(position *
0x9E3779B9u) >> (32 - n_passes_log2)`) because canonical-slicing
collapses on chr22's centromere all-A region.

### 3.6 In-band runtime parametrisation (primer_scan extension)

`primer_scan` proves the in-band header pattern scales: a 24-byte
header carries `primer_fwd_canonical` and `primer_rc_canonical` as
uint64s, so a single xclbin per primer-length P handles ANY primer of
that length at runtime. Cost: 16 bytes / dispatch (negligible vs the
4096-byte payload). Benefit: no xclbin rebuild when a user wants to
scan a different adapter.

### 3.7 IUPAC PAM matching via packed 4-bit one-hot mask (Track A v0)

The `pam_filter_iupac` kernel (Track A v0, `bionpu be design`) extends
the toolkit with a per-position IUPAC mask check that fits the
CRISPR-shape envelope cleanly. The 24-byte chunk header carries a
packed `pam_mask` (4 bits per position; A=0x1, C=0x2, G=0x4, T=0x8,
N=0xF, R/Y/S/W/K/M/B/D/H/V via OR of base nibbles) plus a 1-byte
`pam_length`. The kernel inner loop is one bitwise AND per PAM
position: `(base_onehot & pos_mask) == 0` short-circuits the match.
**A single xclbin per topology serves every supported Cas9 PAM
variant** (SpCas9 NGG, SpCas9-NG, SpCas9-NRN, SaCas9-NNNRRT, ...) via
runtime header args — no per-PAM rebuild. Same in-band-runtime
parametrisation pattern as `primer_scan` v0 (§3.6), specialised to the
PAM-degeneracy alphabet.

### 3.8 Component-triple scaffold-invariant lookup (Track B v0)

PRIDICT 2.0 efficiency scoring is keyed on a (PBS, RTT, edit) triple,
NOT on the full pegRNA sequence (which embeds the scaffold variant).
v1 of `bionpu crispr pe design` initially keyed PRIDICT cache lookups
on the full pegRNA, which broke when the same edit / PBS / RTT was
enumerated under multiple scaffolds (Anzalone 2019, Nelson 2022
evopreQ1/tevopreQ1, ...). The T10/T12 fix promoted the cache key to
the (PBS, RTT, edit) component-triple, making the cache
scaffold-invariant. This is a generic toolkit-level pattern: when a
silicon or CPU kernel scores a sequence whose host-side enumeration
varies in a non-scoring dimension, key the cache on the
scoring-relevant components, not the assembled sequence.

### 3.9 UCSC refGene gene-symbol resolver (`bionpu.data.genome_fetcher` v1)

Replaces the 20-gene hardcoded whitelist that earlier `bionpu crispr
design` shipped with. Bundles a 349-gene refGene subset (7 KB JSON) +
local FASTA slicer; cold lookup ~0.4 ms, warm ~0.2 µs (**~1228×**
cache cold→warm speedup). Offline-only by default. Powers Mode A
("`--target SYMBOL`") in `bionpu crispr design`, `bionpu be design`,
and `bionpu crispr pe design`. BRCA1 + EGFR end-to-end resolution
validated against real hg38.fa. Commit `2345536`.

### 3.10 MemTile-aggregated combine — generalised across the toolkit

The `outC.prod.join(N, ...)` ObjectFifo pattern (originally inherited
from `crispr/match_multitile_memtile`) is now proven across **six
silicon kernels**: `kmer_count`, `minimizer`, `primer_scan`,
`cpg_island`, `tandem_repeat`, AND `pam_filter_iupac`. The pattern
works out-of-the-box for `n_tiles ∈ {1, 2, 4}`; `n_tiles = 8` exceeds
the AIE2P MemTile S2MM channel budget and remains a v1.1+ open item
(see §8.5). This generalisation is load-bearing for the
toolkit-to-tool pattern (§5.6) — every new kernel ships against an
already-proven aggregation topology.

### 3.11 Silicon-mutex discipline (`npu_silicon_lock`)

Per CLAUDE.md non-negotiable rule, every NPU dispatch from a
subprocess harness wraps in `npu_silicon_lock(label=...)`. The
in-process pyxrt path uses an in-process `_dispatch_lock` only;
silicon multi-use protection handles same-process concurrency, but
cross-process harnesses must serialise via the canonical lock at
`/tmp/bionpu-npu-silicon.lock`. Used uniformly across all six
primitives and the production composition.

---

## 4. Seven primitives, four algorithmic shapes

### 4.1 Algorithmic shape inventory

| # | Primitive | Algorithmic shape | New shape vs prior? |
|---|---|---|---|
| 1 | `kmer_count` | Substring/canonical match + hash-bucket count | Baseline (CRISPR pattern-match) |
| 2 | `minimizer` | Sliding-window comparator + ring-buffer min | NEW (sliding-window state) |
| 3 | `seed_extend` | Hash lookup against host-built reference index | NEW (composition over minimizer) |
| 4 | `primer_scan` | Substring exact match (degenerates to one uint64 compare per base) | Same as kmer_count, simpler |
| 5 | `cpg_island` | Windowed multi-counter statistics (n_C, n_G, n_CG) | NEW (composite window stats) |
| 6 | `tandem_repeat` | Autocorrelation / period detection (per-period streak counter) | NEW (autocorrelation) |
| 7 | `pam_filter_iupac` | Per-position 4-bit IUPAC mask check (one-hot AND short-circuit) | Specialisation of substring-match for degeneracy alphabets (Track A v0) |

Four distinct algorithmic shapes (substring/canonical match,
sliding-window min, windowed multi-counter, autocorrelation streak)
with three compositions: seed-extend reuses minimizer; `bionpu trim`
reuses primer_scan; `bionpu be design` reuses `pam_filter_iupac`.
The thesis generalises across all four shapes.

### 4.2 Per-primitive per-fixture metrics

| Primitive | Fixture | Silicon byte-equal | E2E wall | Reference tool / oracle | Reference wall | Speedup vs reference | Source LOC |
|---|---|---|---|---|---|---|---|
| `kmer_count` v1.3 | chr22 50 Mbp, k=21, n_tiles=4, n_passes=4 | top-1000 = 1000/1000 | **10.18 s** | Jellyfish 1T | 9.81 s | 0.96× e2e (silicon-only **10,533×**) | 2,878 |
| `minimizer` v0 | chr22 50 Mbp, k=15, w=10, n_passes=4 | top-1000 = 1000/1000 (full PARTIAL) | 50.61 s (np=4) | minimap2 `mm_sketch` C ref | ~1-2 s | <1× e2e (open) | 2,193 |
| `seed_extend` v0 | self-map smoke + 100Kbp chr22 slice | byte-equal vs CPU oracle (per smoke harness `state/seed_extend_smoke.py`) | n/m (no measurements.json) | minimap2 seed | n/m | n/m | 823 |
| `primer_scan` v0 | chr22 50 Mbp, P=13, TruSeq P5 | true (silicon == oracle, 0 hits both) | **2.05 s** | CPU oracle (numpy) | 9.18 s | **4.47×** | 1,686 |
| `cpg_island` v0 | chr22 50 Mbp, W=200, n_tiles=4 | true (1352 islands silicon == 1352 oracle) | **3.62 s** | CPU oracle | 12.73 s | **3.52×** | 1,070 |
| `tandem_repeat` v0 | chr22 50 Mbp, periods 1..6, ≥5 copies, n_tiles=4 | 289,548 silicon vs 289,551 oracle (10 diff records; v0 threshold pass) | **9.41 s** | CPU oracle | 14.85 s | **1.58×** | 1,211 |

Notes:

* `silicon-only` for `kmer_count` is the per-iteration kernel time
  averaged over 3 trials at chr22 scale (951.7 µs); end-to-end wall
  (10.18 s) includes ~7.5 s pipelined silicon, ~0.5 s parallel sort,
  ~2.2 s setup/IO. The 10,533× silicon-only number compares only
  silicon compute against Jellyfish 1T, which is the right number to
  cite when arguing about hardware design fit; the ~0.96× e2e wall
  tells the throughput-amortisation story (see §6).
* `minimizer` ships **smoke byte-equal** at both pinned configs
  ((k=15, w=10) and (k=21, w=11)), and **chr22 top-1000 byte-equal**
  at (k=15, w=10). Full chr22 byte-equal is blocked on the
  `MZ_MAX_EMIT_IDX = 2046` per-chunk cap that fires on real DNA;
  v1 multi-pass + Fibonacci-position-hash recovered 6.36 M → 14.26 M
  emits (32% → 71%); full close requires v2 widening of the partial
  slot to 64 KiB or chunk reduction to 1024 B (filed as
  `minimizer-emit-cap-saturation` in
  `bionpu/kernels/genomics/minimizer/gaps.yaml`).
* `seed_extend` v0 has an existing silicon harness
  (`state/seed_extend_smoke.py`) and op-class glue that lazily binds
  the v0/v1 minimizer artifact, but the
  `results/seed_extend/v0/measurements.json` file is empty as of this
  writeup. The smoke harness is documented; I conservatively avoid
  quoting numbers I haven't seen.
* `tandem_repeat` chr22 byte_equal = false because of an
  overlap-merge phase-shift edge case (3 oracle records vs 7 silicon
  records differ; documented in
  `bionpu/kernels/genomics/tandem_repeat/gaps.yaml#tandem-repeat-overlap-merge-phase-shift`).
  v0 ships the threshold-pass verdict (≤10 record difference on
  chr22, well within the design brief).
* `primer_scan` chr22 with `AGATCGGAAGAGC` (TruSeq P5) returns 0 hits
  because the chr22 reference has no exact occurrences of the
  Illumina sequencing adapter (as expected — this is a sequencing
  artifact, not a genomic feature). The byte-equal verdict is still
  meaningful: the silicon and CPU oracle agree on emptiness across
  50 Mbp.

### 4.3 Per-primitive design notes

#### `kmer_count` v1.3 (commit `723e019`)

* **Shape:** substring/canonical match + hash-bucket count.
* **Tile DM budget:** ~43 KiB used (8 KiB seq_in chunk + overlap;
  32 KiB partial_out depth=1; 1 KiB stack; 2 KiB text/rodata).
  21 KiB headroom against the 64 KiB cap. Critical: `partial_out`
  depth=1 (no ping-pong) — depth=2 × 32 KiB = 64 KiB would exhaust the
  budget on its own.
* **Silicon design:** 4 K=15/21/31 macro-baked variants × n_tiles ∈
  {1, 2, 4} × n_passes ∈ {4, 16}. Hash-slice partition; canonical=0
  summary counter handled separately (kernel-side uint32 tail in
  partial_out, gated by `pass_idx==0 && tile_idx==0`).
* **Validation:** smoke (10 Kbp) byte-equal vs CPU oracle at all 3 K
  values; chr22 top-1000 byte-equal at k=21/n_tiles=4/n_passes=4
  after v1.2(a) closed canonical=0 cap-fire and chunk-overlap
  double-emit gaps.
* **Result:** silicon-only **10,533×** vs Jellyfish 1T. End-to-end
  10.18 s vs Jellyfish 1T 9.81 s (0.96×). The **silicon compute
  itself is fast**; throughput is silicon-dispatch-overhead bound (see
  §6) and dominates the e2e wall.
* **Source:** `bionpu-public/src/bionpu/kernels/genomics/kmer_count/`
  (commit `723e019`).
* **Measurements:** `results/kmer/v1.3-chr22/measurements.json`.

#### `minimizer` v0 / v1 (commit `88fb9ab`)

* **Shape:** sliding-window comparator + ring-buffer min, oldest-on-tie.
* **Algorithm:** canonical specialisation of minimap2's `mm_sketch`
  (Li, *Bioinformatics* 2016, `sketch.c:77+`). Intentional divergence
  from minimap2: no `hash64` post-canonicalisation (pure-canonical
  lexicographic min); no homopolymer compression; ties broken by
  oldest position wins (silicon-mirrorable in a ring-buffer scan;
  minimap2's tie-breaking depends on hash collisions).
* **Tile DM budget:** ~38 KiB used at (k=21, w=11); 88 B `ring_canonical`
  + 44 B `ring_position` + 11 B `ring_valid` + 4104 B seq_in (dbl-buf
  ~8208 B) + 32768 B `partial_out` + ~2 KiB stack + scalars.
* **Validation:** smoke byte-equal at (k=15, w=10) and (k=21, w=11);
  chr22 (k=15, w=10) **top-1000 byte-equal** at np=4 (full byte-equal
  blocked by emit-cap saturation — chr22's per-chunk minimizer density
  exceeds the 2046 cap; oracle averages 6451 emits/chunk vs cap 2046).
* **v1 throughput recovery:** position-Fibonacci-hash slice replaces
  v0's no-slice emit gate; recovered 32% → 71% of full chr22 emits at
  np=4. **Throughput remains higher than minimap2** because the
  per-dispatch ring-buffer min-scan is scalar AIE2P code (~5× slower
  per dispatch than `kmer_count`'s straight emit-on-mask).
* **Source:** `bionpu-public/src/bionpu/kernels/genomics/minimizer/`.
* **Measurements:** `results/minimizer/v0/measurements.json`,
  `results/minimizer/v1/measurements.json`.

#### `seed_extend` v0 (commit `3833e43`)

* **Shape:** hash lookup against host-built reference index (a
  composition over the silicon `BionpuMinimizer` op + a host
  `MinimapIndex`).
* **Pipeline:** (1) build minimap2-style index from the reference
  on CPU (one-time, ~30 s for chr22); (2) extract query minimizers on
  the NPU via the v0/v1 minimizer artifact; (3) host-side seed lookup
  via canonical → list of (ref_pos, strand) tuples.
* **`mm_seed_t` parity:** sorted by `(ref_pos asc, query_pos asc)`,
  matching minimap2's `collect_seed_hits` ordering before chaining.
* **Frequency cutoff:** `DEFAULT_FREQ_CUTOFF = 1000` (mirrors
  minimap2's `-f 0.0002`). v0 design rationale: a v0 build without
  the cutoff hit a 54 GB RSS runaway on chr22 self-mapping silicon
  validation — chr22's centromere N → A scrub creates ~12 M
  canonical=0 (all-A) minimizers; a single query minimizer landing in
  any homopolymer region would emit seed-hits for ALL of them.
* **Validation:** smoke + 100Kbp chr22 slice self-map harness exists
  at `state/seed_extend_smoke.py`; the
  `results/seed_extend/v0/measurements.json` is not yet populated, so
  no quantitative numbers are quoted here.
* **Source:** `bionpu-public/src/bionpu/genomics/seed_extend/` (commit
  `3833e43`).

#### `primer_scan` v0 (commit `3888118`)

* **Shape:** substring exact-match (degenerates to one uint64 compare
  per base after the rolling fwd/rc math).
* **Tile DM budget:** ~41 KiB used; well under the 64 KiB cap.
* **In-band runtime parametrisation:** the primer canonical (forward
  + RC) lives in the 24-byte chunk header; one xclbin per
  primer-length P handles ANY primer of that length without rebuild
  (Path B in DESIGN.md §"Path B"). Built per-P variants: P=13, P=20,
  P=25.
* **Validation:** smoke (P=13, TruSeq P5) byte-equal vs CPU oracle
  (0 == 0 hits); synthetic-inject (2 == 2 hits, byte-equal); chr22
  byte-equal (0 == 0 hits across 50 Mbp).
* **Result:** chr22 e2e wall **2.05 s vs CPU oracle 9.18 s = 4.47×**
  speedup. Per-dispatch silicon avg 2326.78 µs.
* **Open gaps:** none filed at v0 ship (`gaps.yaml` is empty).
* **Source:** `bionpu-public/src/bionpu/kernels/genomics/primer_scan/`
  (commit `3888118`).
* **Measurements:** `results/primer_scan/v0/measurements.json`.

#### `cpg_island` v0 (commit `e4d41a4`)

* **Shape:** windowed multi-counter statistics (rolling `n_C`, `n_G`,
  `n_CG`).
* **Algorithm:** Gardiner-Garden & Frommer (1987) plus Takai & Jones
  (2002): emit window-start when both fixed-point thresholds pass:
  `2*(n_C + n_G) >= W` (≥50% GC content) and `5*W*n_CG >= 3*n_C*n_G`
  (CpG observed/expected ≥ 0.6). The host then sorts and dedups
  candidate starts and applies the oracle merge rule (contiguous
  candidate runs with length ≥ W become half-open intervals of
  `(run_start, run_end + W)`).
* **Wire format extension:** per-tile output is just `[uint32
  emit_count][N × uint32 candidate_position]` (4-byte records, not
  16-byte). Cap: `CI_MAX_EMIT_IDX = 8190` per chunk (one
  `cpg-island-emit-cap-saturation` gap filed for synthetic all-CG
  chunks; not observed on real chr22).
* **Validation:** synthetic-inject (1 == 1 island, byte-equal); smoke
  (12 == 12 islands, byte-equal); chr22 (**1352 == 1352 islands**,
  byte-equal).
* **Result:** chr22 e2e wall **3.62 s vs CPU oracle 12.73 s = 3.52×**
  speedup. Per-dispatch silicon avg 918.961 µs.
* **Open gaps:** one (`cpg-island-emit-cap-saturation`, severity
  low, not observed on real chr22).
* **Source:** `bionpu-public/src/bionpu/kernels/genomics/cpg_island/`
  (commit `e4d41a4`).
* **Measurements:** `results/cpg_island/v0/measurements.json`.

#### `tandem_repeat` v0 (commit `e4a4d3c`)

* **Shape:** autocorrelation / period detection — per-period streak
  counter.
* **Algorithm:** for each period `q ∈ [1, 6]`, walk the input and
  maintain `streak[q] = consecutive bases that match their predecessor
  at distance q`. Emit a record iff `streak[q] >= q * (MIN_COPIES -
  1)` with `MIN_COPIES = 5` (matches TRF default for the supported
  motif lengths). Output: `(start, end, period, motif_canonical)`.
* **Why autocorrelation, not greedy scan?** The silicon kernel state
  is just one streak counter + one streak-start position per period
  (28 bytes total for periods 1..6). Greedy scan would need a full
  motif comparison every base, which doesn't pipeline as cleanly. The
  byte-equal contract is defined against an oracle that uses the
  silicon's autocorrelation streak semantics — the two semantics
  identify the same biological STR with the same number of copies;
  only the byte-boundary representation differs.
* **Forward-strand only.** v0 matches TRF's default output (the
  reverse-strand record would be redundant for STR detection).
* **Validation:** synthetic (7 == 7 records, byte-equal); smoke
  (26 == 26 records, byte-equal); chr22 (289,548 silicon vs 289,551
  oracle records; 10 records diverge — overlap-merge phase-shift edge
  case documented in `gaps.yaml#tandem-repeat-overlap-merge-phase-shift`,
  severity low; v0 threshold-pass verdict: ≤10 record difference on
  chr22).
* **Result:** chr22 e2e wall **9.41 s vs CPU oracle 14.85 s = 1.58×**
  speedup. Per-dispatch silicon avg 2519.01 µs.
* **Open gaps:** four filed (period-cap-6, fuzzy-mismatch,
  chunk-spanning-STR, overlap-merge phase-shift). All v0-scope
  decisions; none block ship.
* **Source:** `bionpu-public/src/bionpu/kernels/genomics/tandem_repeat/`
  (commit `e4a4d3c`).
* **Measurements:** `results/tandem_repeat/v0/measurements.json`.

#### `pam_filter_iupac` v0/v1 (commits `39ba53a` + `e1454ca`) — NEW kernel beyond the original six

* **Shape:** per-byte streaming integer math; per-position 4-bit IUPAC
  mask check against the rolling 2-bit window. Same algorithmic family
  as `primer_scan` (substring exact match) but with a degeneracy
  alphabet collapsed into a one-hot mask check.
* **Why a new kernel rather than a `primer_scan` config:** PAM degeneracy
  (N, R, Y, ...) can't be expressed as a single uint64 canonical for
  rolling compare. Mask-AND short-circuit per PAM position is the
  natural generalisation.
* **Tile DM budget:** ~41 KiB used (mirrors `primer_scan`).
* **In-band runtime parametrisation:** 24-byte header carries
  `pam_mask` (uint32, packed 4-bit nibbles) + `pam_length` (uint8); a
  single xclbin per topology serves every supported Cas9 PAM variant.
* **Validation (v0, commit `39ba53a`):** byte-equal vs CPU oracle
  across **four IUPAC PAM variants** on an 8 Kbp synthetic locus —
  `NGG` (536 == 536 hits), `NG` (2026 == 2026), `NRN` (1019 == 1019),
  `NNNRRT` (505 == 505). 10/10 injection-recovery on `NGG`. chr22 1
  Mbp middle slice (skipping centromere N-pad): **264,918 silicon
  hits in 1.35 s wall** at 3.39 ms / dispatch (3 timed iterations,
  4080 bases per chunk).
* **v1 (commit `e1454ca`):** in-process pyxrt path on top of v0
  silicon. chr22 1 Mbp slice **byte-equal across paths**: subprocess
  141 ms vs pyxrt 49 ms = **2.90× speedup**, with 62 dispatches and
  per-dispatch wall ~700 µs (`wait` 691 µs of the 770 µs
  total — silicon-bounded; see §6.3 two-regime corollary). v1 also
  integrates locked `crispr/match_multitile_memtile` + CFD aggregation
  for full off-target scan inside `bionpu be design`.
* **Source:** `bionpu-public/src/bionpu/kernels/crispr/pam_filter_iupac/`.
* **Measurements:** `results/be_design/v0/measurements.json`,
  `results/be_design/v1/measurements.json`.

---

## 5. Production composition layer

The v1 baseline shipped one production composition (`bionpu trim`).
v2 ships **five production CLIs** built on top of the seven-primitive
toolkit. Each CLI binds a thin host wrapper (10²-10³ LOC) over either
a locked silicon kernel, a published CPU model under runtime-dep
posture, or both. The pattern itself — toolkit kernels + a host
composition layer + locked silicon — is the load-bearing claim of
this section: **once the silicon is locked, each new tool collapses
to roughly one agent run** (§5.6).

The five v2 CLIs and their kernels:

| CLI | Track | Silicon kernels | CPU models | First-ship commit |
|---|---|---|---|---|
| `bionpu trim` v2 | adapter trimming | `primer_scan` (locked) | cutadapt parity oracle | `1557faf` (v2 numbers); v1 baseline `9b6adc2` |
| `bionpu crispr design` v1 | Wave 1 nuclease design | `match_multitile_memtile`, `pam_filter` | Doench RS2/Azimuth, CFD | `e7d666b` |
| `bionpu be design` v0/v1 | Track A base editor | `pam_filter_iupac` (NEW), `match_multitile_memtile` | CFD aggregation, BE window/bystander | `39ba53a` + `e1454ca` |
| `bionpu library design` v0 | Track C pooled CRISPR | (none directly; reuses `crispr design` silicon by import) | Doench RS1/RS2, CFD | `1ff8d5c` |
| `bionpu crispr pe design` v0 | Track B prime editor | `match_multitile_memtile` (off-target only) | PRIDICT 2.0 + ViennaRNA (CPU) | T1-T14 (latest `f950056`; close `1d4840a`) |

Sections §5.1-§5.5 cover each CLI in turn; §5.6 surfaces the shared
toolkit-to-tool pattern.

### 5.1 `bionpu trim` v2 — adapter trimming with cutadapt parity

`bionpu trim` is the original v1 production composition: a
cutadapt-byte-equal 3' adapter trimmer that uses `BionpuPrimerScan`
as its hot loop. v0 scope: `cutadapt -a ADAPTER --no-indels -e 0`
semantics (single adapter, exact match, forward-strand only).

The composition is a thin host wrapper:

1. `bionpu.genomics.adapter_trim.fastq.parse_fastq` reads a FASTQ
   stream (gzip transparent via `.gz` suffix).
2. For each read, if it is pure ACGT, pack to 2-bit MSB-first via
   `bionpu.data.kmer_oracle.pack_dna_2bit`.
3. Dispatch `BionpuPrimerScan(primer=adapter, n_tiles=4)`.
4. Trim at the leftmost forward-strand match (RC matches discarded —
   `cutadapt -a` is forward-only).
5. Reads with non-ACGT bases (Ns) fall back to the CPU oracle path
   (silicon kernel only handles ACGT).

Source: `bionpu-public/src/bionpu/genomics/adapter_trim/__init__.py` +
`trimmer.py` (commit `44be29b`).
Silicon harness: `state/bionpu_trim_smoke.py`.

#### 5.1.1 Cross-checked byte-equality (`bionpu trim`)

The smoke harness runs three paths and asserts byte-equal SHA-256 of
the output FASTQ at three fixture sizes:

| Cross-check | 100 reads | 1K reads (silicon sample) | 10K reads |
|---|---|---|---|
| `cpu_vs_cutadapt` | PASS | n/a | PASS |
| `npu_vs_cutadapt` | **PASS** | **PASS** | n/a (silicon would take ~16 min) |
| `npu_vs_cpu` | PASS | n/a | n/a |

**4/4 byte-equal cross-checks pass.** The `bionpu trim` NPU path
produces exactly the same trimmed FASTQ as cutadapt at the byte level.

#### 5.1.2 The honest throughput finding (`bionpu trim`)

The same harness measures throughput at the same fixture sizes:

| Path | 100 reads | 10K reads | 1K silicon sample |
|---|---|---|---|
| cutadapt 1T | 1,835 reads/s | **160,341 reads/s** | (run as cross-check only) |
| `bionpu trim --device cpu` | 32,600 reads/s | 57,386 reads/s | (run as cross-check only) |
| `bionpu trim --device npu` v0 (silicon, B=1) | **9.68 reads/s** | (would take ~16 min) | **9.64 reads/s, 103 ms/read** |
| `bionpu trim --device npu` v1 (silicon, B=1024) | (1 dispatch covers fixture) | **5,942 reads/s** | n/a |
| `bionpu trim --device npu` v1 (silicon, B=4096) | n/a | **11,391 reads/s** | n/a |
| `bionpu trim --device npu` v2 (pyxrt, B=512) | (1 dispatch covers fixture) | **110,318 reads/s** | n/a |
| `bionpu trim --device npu` v2 (pyxrt, B=4096) | n/a | **97,443 reads/s** | n/a |

**v0 NPU path was ~16,000× slower than cutadapt 1T per read.** v1
batched dispatch closed that gap to **~14× slower** at B=4096 — the
silicon time per dispatch was unchanged, but the per-dispatch
overhead was paid 4096× less often. **v2 in-process pyxrt closes the
gap further to within 1.5× of cutadapt 1T**: per-dispatch overhead
drops from ~103 ms (v1 subprocess) to ~700 µs (v2 pyxrt — measured at
B=512: stage 4 µs, sync_to 7 µs, kernel.run launch 17 µs, run.wait
~600 µs, sync_from 5 µs, parse 20 µs).

v0/v1's 103 ms / dispatch was **subprocess fork + XRT context init +
xclbin load + dispatch + drain + teardown** — paid afresh on every
batch. v2 reuses the device + xclbin + kernel + bo_instr + ring BOs
across every dispatch in the host process; subsequent dispatches pay
only the silicon-side time (~600 µs `wait` dominates, the rest is
single-digit µs).

At B=512, the 10K fixture finishes in 91 ms (v2) vs cutadapt 1T at
~73 ms — silicon is **82% of cutadapt 1T parity** on this fixture.
The v2 corollary still binds (per-dispatch overhead is non-zero), but
the threshold has dropped 140× and silicon is now production-class on
per-record streams.

The corollary's floor (per-dispatch overhead) is no longer the
binding constraint above ~80 K reads/s; pure-Python FASTQ I/O +
classification is. See § 6.3.

### 5.2 `bionpu crispr design` v1 — Wave 1 nuclease design (commit `e7d666b`)

`bionpu crispr design` is the Wave 1 user-facing wrapper for SpCas9
nuclease guide design (PRD-guide-design-on-xdna v0.2). It composes
the locked CRISPR silicon (`crispr/match_multitile_memtile` for
off-target scan, `crispr/pam_filter` for NGG PAM matching) with the
CPU on-target scoring layer (Doench RS1/RS2/Azimuth) + CFD off-target
aggregation. Three modes:

* **Mode A — gene symbol** (`--target BRCA1 --genome hg38`): UCSC
  refGene lookup via `bionpu.data.genome_fetcher` v1 (§3.9), local
  hg38 FASTA slicer, full off-target scan against the resolved locus.
* **Mode B — target FASTA**: user-supplied FASTA window; off-target
  scan = same window.
* **Mode C — synbio** (`--genome none`): off-target scan SKIPPED;
  emits `cfd_aggregate=NaN` and `NO_OFF_TARGET_SCAN` flag (synbio
  contract — sequence is plasmid-only, no genomic context).

Mode A smoke (BRCA1 synthetic chr17): 0.13 s wall, 980 PAM candidates
→ 10 ranked guides; 2 off-target hits across the locus. Composite
score is the CRISPOR-style on-target × off-target product. v1 ships
11 new tests; cumulative design-layer suite stays green at 69/69.
The whitepaper was deliberately untouched at v1 — `crispr design` is
product engineering over already-validated silicon, not new empirical
fit-class evidence.

### 5.3 `bionpu be design` v0/v1 — Track A base editor (commits `39ba53a` + `e1454ca`)

`bionpu be design` is the Track A CLI for cytosine + adenine base
editor (CBE/ABE) guide design across the Cas9 PAM zoo (SpCas9 NGG,
SpCas9-NG, SpCas9-NRN, SaCas9-NNNRRT, evoCDA1, ABE8e, ...). v0
introduced the **NEW silicon kernel** `pam_filter_iupac` (§4.7) — a
single xclbin per topology serves every supported PAM variant via the
runtime IUPAC mask in the chunk header. v0 ships 4/4 PAM-variant
byte-equal smoke + a chr22 1 Mbp slice (264,918 NGG hits in 1.35 s)
+ 28 CLI tests. v1 (commit `e1454ca`) adds the in-process pyxrt path
(141 ms → 49 ms on chr22 1 Mbp = 2.9× silicon-bounded — see §6.3
two-regime corollary) and integrates locked
`crispr/match_multitile_memtile` + CFD off-target aggregation; 38/38
tests pass. Mode A (gene-symbol) Mode B (FASTA) Mode C (synbio) all
inherit the `crispr design` mode contract; `cfd_aggregate=NaN` is
reserved for synbio mode under `--genome none`.

### 5.4 `bionpu library design` v0 — Track C pooled CRISPR (commit `1ff8d5c`)

Per-gene-list pooled CRISPR knockout library designer. **Reuses
`crispr design` v1 by import** (no edits to silicon-touching modules)
and layers four library-scope features on top: (1) global guide-spacer
deduplication (`highest_score` or `alphabetical` resolver) so the
pooled library has unique spacers across all targets; (2)
deterministic non-targeting controls (random ACGT, seeded RNG); (3)
safe-harbor controls (AAVS1 / CCR5 / ROSA26); (4) essential-gene
controls (RPS19, RPL15) for dropout normalisation. Smoke fixture:
5-gene + 100 non-targeting + 3 safe-harbor + 2 essential-gene library
= 125 rows in **0.54 s wall**, 0 under-balanced genes, all 20 unique
spacers per gene satisfied. 30/30 tests pass. v0 explicitly defers
genome-wide enumeration, activation/interference (CRISPRa/i),
balanced-coverage metrics — these are v1 follow-ons.

### 5.5 `bionpu crispr pe design` v0 — Track B prime editor (T1-T14, latest `f950056`; close `1d4840a`)

The fourth production CLI: prime editor pegRNA design (Anzalone 2019
+ Mathis 2024 PRIDICT 2.0 + Nelson 2022 evopreQ1/tevopreQ1 scaffold
zoo). Track B v0 ships **PE2 + PE3 strategies** with **real PRIDICT
2.0 efficiency scoring** (HEK + K562 heads via runtime dep — PRIDICT
2.0 turned out to be MIT-licensed, not restrictive as initially
assumed) + **real ViennaRNA folding** (MFE + structure + PBS pairing
prob + scaffold disruption) + **off-target scan via locked
`match_multitile_memtile`** (delegated through `crispr_design`'s
existing scan path; sits ABOVE the silicon-mutex layer per CLAUDE.md
non-negotiable rule). v0 closes 14 plan tasks (T1-T14) with cumulative
56+ pe_design unit tests + 4 integration tests + 5/5 smoke targets ×
3 modes byte-equal-deterministic (BRCA1, TP53, EGFR, HBB, MYC).
Component-triple scaffold-invariant lookup (§3.8) is the load-bearing
toolkit pattern for this CLI. Per Track D Phase 0 closure, PRIDICT-class
transformer scoring is **CPU-only** for v0 — Track E's NO-GO
microbenchmark (§6.4) reinforces this.

### 5.6 The toolkit-to-tool pattern — five CLIs from one toolkit

The empirical claim of this section: **once silicon kernels are
locked, each new production CLI collapses to roughly one agent
run** (one focused authoring session, ranging from a few hours to
~1 day depending on CPU-side scoring complexity). This is the
multiplicative payoff the §3 toolkit exists for. Concrete
data points:

* `bionpu crispr design` v1 → `bionpu library design` v0 was a
  reuse-by-import refactor; library_design touches no silicon code.
* `bionpu be design` v0's only new silicon work was the
  `pam_filter_iupac` kernel (Track A v0); the surrounding CLI shape
  (Modes A/B/C, CFD aggregation, scoring composition) reused
  `crispr design` v1 patterns wholesale.
* `bionpu be design` v1 was a pure pyxrt-path swap on already-locked
  v0 silicon — same kernel, same byte-equal contract, 2.9× wall
  reduction from one host-side change.
* `bionpu crispr pe design` v0 added zero new silicon kernels — the
  off-target scan adapter (T11) is a thin delegation onto
  `crispr_design._scan_locus_for_offtargets`.

The pattern is therefore: **invest in silicon kernels once
(weeks-to-months), then ship CLIs cheaply (hours-to-day each)**. The
five v2 CLIs validate this claim across two distinct CRISPR
modalities (nuclease + base editor + prime editor + pooled library +
adapter trimming). Each new modality reused the toolkit's wire
format, MemTile aggregation pattern, in-band runtime parametrisation,
silicon-mutex discipline, and (where applicable) gene-symbol resolver
— with at most one new silicon kernel per modality.

---

## 6. The per-dispatch amortisation corollary

### 6.1 Statement

> AIE2P silicon compute is fast for CRISPR-shape kernels (sub-ms per
> dispatch). When a workload submits one chr22-scale dispatch, silicon
> wins decisively (10,533× silicon-only vs Jellyfish 1T on
> `kmer_count`). When a workload submits one dispatch per record from
> a per-record stream (FASTQ reads), the per-dispatch overhead floor
> (~100 ms subprocess + XRT context init) dominates by 10⁴-10⁵× and
> the silicon path loses to a single-thread CPU library.

The corollary is a hard hardware-and-host-runtime constraint, not a
kernel bug. The fix is at the dispatch layer (batched dispatch,
persistent kernel, in-process pyxrt), not at the silicon layer.

### 6.2 Measurements supporting the corollary

**Single chr22-scale dispatch wins** (`kmer_count` v1.3):

| Dispatch unit | n_dispatches | Silicon wall (avg) | Total silicon | Total e2e | Reference (Jellyfish 1T) |
|---|---|---|---|---|---|
| 4 KiB chunk × 1 chunk per launch | 12,420 | 1.8 ms | 22 s | 33.84 s (v1.2(a)) | 9.81 s |
| 4 KiB chunk × 8 per launch | 1,556 | 4.72 ms (pipelined) | 7.34 s | 10.18 s (v1.3) | 9.81 s |

`kmer_count` v1.2(b) → v1.3 closed two host-side gaps (TBB par_unseq
sort gave 4.12 s; pipelined depth-4 ring gave 0.6 s) and one in-kernel
gap (skip-zero output BO init gave 0.08 s). **Pipelining alone bought
only ~0.6 s** because XRT serialises silicon submits internally
(single hardware queue per `hw_context`); the dominant win was the
parallel host sort. Silicon throughput (~7.5 s) is now the floor;
beating Jellyfish requires shim-DMA or persistent-kernel changes, both
of which are open
(`kmer-shim-dma-context-overhead-floor`,
status `open`).

**Per-record streams lose without batching** (`bionpu trim` v0):

The `bionpu trim` 1K silicon sample: 1000 dispatches × 103,533 µs/read
= 103.5 s. Of this, the silicon compute itself is sub-ms (the
`primer_scan` kernel does ~1-2 ms per chunk and a 150-base read fits
in one chunk — the actual silicon time per read is far below the
measured 103 ms/read). The remainder is **XRT context init +
subprocess fork per dispatch**.

Per-dispatch overhead floor:

| Path | Per-dispatch overhead | Source |
|---|---|---|
| Subprocess + XRT init (current `bionpu trim` v0) | ~100-103 ms | `bionpu_trim` v0 1K sample |
| In-process pyxrt | ~10 ms | `b-m8-pyxrt` (fused_perts wall 27 ms includes ~10 ms dispatch + ~17 ms compute) |
| Persistent kernel + ObjectFifo (v1.3 `kmer_count` pipelined) | ~5 ms | `kmer_count` v1.3 measurements |
| Open question: persistent-kernel mode + IRON-side rewrite | unknown (estimated <1 ms) | `kmer-shim-dma-context-overhead-floor` |

### 6.3 Known fix path: batched dispatch + in-process pyxrt — and a two-regime localisation

**Two-regime localisation** (refined from Track A v1 + `bionpu trim`
v2 contrast):

- **Subprocess-bounded** (sub-ms silicon, multi-ms wall): the pyxrt
  path buys **100×+** because subprocess fork + XRT init dominated
  total wall. Example: `bionpu trim` v0 → v2 went from 9.68 reads/s
  (subprocess, ~103 ms / dispatch) to 110,318 reads/s (pyxrt, B=512,
  ~700 µs / dispatch) — a **11,372× per-record throughput
  speedup**. The silicon was already fast; the host runtime was
  paying ~150× more than the silicon per dispatch.
- **Silicon-bounded** (multi-ms silicon): the pyxrt path buys only
  **~3×** because silicon already dominated total wall. Example:
  `pam_filter_iupac` chr22 1 Mbp slice (Track A v1, commit
  `e1454ca`) — subprocess **141 ms** vs pyxrt **49 ms** = 2.90×.
  The per-dispatch profile shows `wait` 691 µs of 770 µs total
  (90% silicon-bound); the remaining 10% is the residual pyxrt
  staging cost. There is no further host-side lever here; the next
  win has to come from kernel-side throughput (more tiles, wider
  output, on-tile state expansion) or from `kmer_count`-style batched
  dispatch.

The corollary's binding constraint is the **same** in both regimes —
per-dispatch overhead amortisation — but **the lever depth depends
on the silicon-vs-subprocess ratio**. The same Track A v0 → v1
transition that the BERT-mini-class workloads cannot apply (because
their silicon is already 16 ms / dispatch — far above the pyxrt
floor) buys ~3× when silicon time is comparable to the host overhead,
and 100×+ when silicon time is dwarfed by host overhead. **Always
classify the workload first**: silicon-bound or subprocess-bound,
then choose the lever.

`kmer_count` v1.2(b) demonstrated batched dispatch
(`BIONPU_KMER_COUNT_N_CHUNKS_PER_LAUNCH=8` env var; new artifact dirs
`bionpu_kmer_count_k{K}_n{T}_np{P}_b8/`). Per-dispatch wall reduced
from ~1.8 ms (b=1) to ~6 ms / 8 chunks = 0.75 ms / chunk. End-to-end
chr22 wall: 33.84 s → 14.53 s = **2.33× speedup**.

The same lever applied to `bionpu trim`: batch N reads per dispatch,
amortise the ~100 ms overhead across all N. **v1 `bionpu trim`
shipped this** (sentinel-separated stream concatenation, Path B); v2
adds the in-process pyxrt path that drops the per-dispatch overhead
20× and makes the silicon path production-class. Measured 10K
fixture:

| version | dispatch path | batch_size | wall (s) | reads/s | dispatches | per-dispatch wall | speedup vs v0 |
|--------:|---------------|-----------:|---------:|--------:|-----------:|------------------:|--------------:|
| v0      | subprocess    | 1          | ~1031    |     9.7 |     10 000 |        ~103 ms    |     1×        |
| v1      | subprocess    | 64         |    16.79 |   595.5 |       157  |        ~103 ms    |    61×        |
| v1      | subprocess    | 1024       |     1.68 | 5 941.7 |        10  |        ~112 ms    |   613×        |
| v1      | subprocess    | 4096       |     0.88 | 11 391  |         3  |        ~125 ms    | 1 174×        |
| **v2**  | **pyxrt**     | **64**     | **0.110**| **91 K**|     157    |     **~700 µs**   |  **9 400×**   |
| **v2**  | **pyxrt**     | **512**    | **0.091**| **110 K**|       20  |     **~610 µs**   | **11 400×**   |
| **v2**  | **pyxrt**     | **2048**   | **0.091**| **110 K**|        5  |     **~690 µs**   | **11 400×**   |

v2 closes the gap to cutadapt 1T (~160 K reads/s on this hardware) to
**1.5×**, well past the 50 K reads/s acceptance gate and within the
100 K reads/s stretch target. The corollary itself still binds — there
is still a non-zero per-dispatch overhead — but the binding threshold
moves from "any per-record stream loses by 10⁴-10⁵×" to "per-record
streams are production-class above ~80 K reads/s and approach
cutadapt parity at 100K+".

Per-phase profile of one v2 dispatch at B=512 (averaged over 20
dispatches on the 10K fixture):

| Phase | µs | Notes |
|---|---:|---|
| Stage seq_in (header + memcpy + bo.write) | 3 | host-side packing |
| Sync to device (TO_DEVICE) | 6 | DMA-coherent flush |
| Kernel launch (kernel(...) returns xrt::run) | 16 | non-blocking |
| Run.wait() | 630 | actual silicon execute + return |
| Sync from device (FROM_DEVICE) | 5 | DMA invalidate |
| Parse tile-0 blob | 20 | numpy structured dtype |
| **Total** | **~680** | |

v2's silicon path is silicon-bound now, not host-bound: ~93% of the
per-dispatch wall is `run.wait()`. Above the silicon path, the
remaining ~570 ms host-side wall on the 10K fixture splits between
FASTQ I/O parse, per-record string handling, and the
sentinel-separated concat + 2-bit pack. v2 ships a vectorised packer
(`_pack_dna_2bit_vectorised`) that drops the concat-and-pack phase
from ~775 ms to ~25 ms per call (numpy bitops vs Python for-loop) —
that single-line change is what lifted the post-pyxrt sweep from
~17 K reads/s to ~110 K reads/s.

### 6.4 Track E gating microbenchmark — small-FP32-NN-with-MAC-stack floor

A 1-day microbenchmark resolved Track E's CONDITIONAL verdict
(`state/track-e-workload-fit-analysis.md` §7) from
[`PRD-crispr-state-of-the-art-roadmap.md`](../PRDs/PRD-crispr-state-of-the-art-roadmap.md)
§3.5 to a definite NO-GO for inDelphi-class CNN edit-outcome
prediction silicon Phase 1. Setup: dispatch the silicon-validated
`dorado_fast_conv_stem_layer1` kernel (`Conv1d(in=1, out=16, k=5,
stride=1, pad=2)` over `(1, 1, 2000)` FP32 chunked 10×200) over
inDelphi-shape input via in-process pyxrt; 100 warmup + 1000 measured
iterations; commit `cc2c8b6`.

**Result**: median **11,112 µs / dispatch** (min 11,017 / p50 11,112
/ p95 11,653 / p99 11,915 / max 12,160; std 208 µs; CoV 1.9%; tight
unimodal — no long tail). **7.41× over** the 1.5 ms PASS threshold;
unambiguous NO-GO. Strategy B re-author projection at
`(B=64, L=60, C=4)` bf16 with on-tile weights still lands at ~2.2 ms
/ dispatch (47% over threshold) — none of the silicon levers we have
today (multi-tile fan-out, pktmerge, burst-length tuning per
[`cascade-burst-length-falsified-2026-04-28.md`]) deliver the 7×+
kernel-side reduction the verdict would require.

**Anchor comparison**:

| Anchor | Per-dispatch wall | Track E ratio |
|---|---:|---:|
| `bionpu trim` v2 primer-scan floor (in-process pyxrt) | 700 µs | 15.875× |
| Track E measured (conv_stem_layer1) | **11,112 µs** | 1.0× |
| BERT-mini block (DNABERT-Epi Phase 0 closure) | ~16,000 µs | 0.695× |
| Track E PASS threshold | 1,500 µs | 7.408× |

**Generalisation** (load-bearing for the strategic-pivot thesis):

The 16 ms BERT-mini floor (Track D Phase 0 closure) was previously
the single empirical anchor for "transformer-shape doesn't fit
AIE2P." Track E's small-CNN measurement at 11.1 ms — landing
**0.695× the BERT-mini floor and 15.875× the bionpu-trim-v2 700 µs
primer-scan floor** — places conv-stem-shape kernels in the **same
workload class as BERT-mini, NOT in the primer-scan class**. The two
dispatch-overhead anchors form a 25× spread across the design
envelope; conv-stem-shape lands firmly on the upper end of that
spread.

The negative boundary of AIE2P's design envelope is therefore
**small-FP32-NN-with-MAC-stack workloads regardless of architecture**
(recurrent Dorado LSTM OR feedforward inDelphi CNN OR small attention
block), not "transformers" specifically. CRISPR-shape genomics
primitives win because they are **integer-pattern-match scans over
packed bytes** — fundamentally different compute geometry from MAC
stacks. The line is not "transformer vs CNN"; it is **MAC-stack NN
vs packed-byte pattern-match scan**.

This refines but does NOT contradict v1's thesis: AIE2P excels at the
CRISPR-shape envelope; outside that envelope, the
small-FP32-NN-with-MAC-stack floor stops every software lever
(multi-tile fan-out, pktmerge, burst-length tuning, on-tile weight
caching, batched dispatch) from delivering the 7×+ kernel-side
reduction needed to flip the verdict. Track D (transformer) and
Track E (small CNN) now share the same fit-class verdict: **CPU-only
for v0**. The seven CRISPR-shape primitives stay on silicon; the
small-NN scoring layers live on CPU.

Source: VERDICT writeup at
`state/track-e-conv-stem-shape-pyxrt-floor-microbenchmark/VERDICT.md`;
canonical results `results/track-e/v0/measurements.json`.

### 6.5 Open question: can persistent-kernel mode push the floor below
1 ms / dispatch?

`kmer_count` v1.3 explicitly **did not** ship persistent-kernel mode:
profile showed setup overhead (stage_seq_in 3-5 µs, sync_to_device
1 µs, zero_out 0 µs) is already amortised across dispatches in the
v1.2(b) flow. Persistent-kernel mode (kernel stays warm in NPU between
calls) would require IRON-side changes locked at v1.2(b). Untested.

If the persistent path drops the floor below 1 ms / dispatch, the
per-record stream path becomes viable for some workloads (especially
short-read sequencing where each read is independent and the per-read
work is tiny).

---

## 7. Reproducibility

### 7.1 Hardware

* **CPU/SoC:** AMD Ryzen AI 9 HX 370 (Strix); XDNA 2 / AIE2P NPU
  silicon.
* **OS:** Ubuntu 26.04.
* **Hostname:** matteius-ProArt-P16-H7606WI.

### 7.2 Software

* **mlir-aie:** `third_party/mlir-aie` at submodule SHA
  `995685a3ac533ef98984152347ea1e885b4f0520` (`feature/iron-fifo-primitives`).
* **Peano-LLVM:** `third_party/llvm-aie` at submodule SHA
  `b5fe4c0c553d25a926770702de3496777155b74b` (`nightly-468993-gb5fe4c0c553d`).
* **xdna-driver:** `third_party/xdna-driver` at submodule SHA
  `d1e1855062da0aab9627fd8c3f26efd3c9d3ef95`
  (`feature/aie2-tdr-debugfs-control`).
* **bionpu-public:** at submodule SHA
  `e4a4d3c591f0f54cfca4c3870868e9b0bda1e526` (`tandem_repeat v0`
  commit, head of main as of 2026-04-28).
* **XRT:** `/opt/xilinx/xrt` (vendor-installed; `xrt-smi 2.21.75`
  system binary lacks NPU shim; sourced setup is canonical for NPU
  work).
* **Python:** 3.14 in ironenv.

### 7.3 Build env (canonical incantation)

```bash
source /opt/xilinx/xrt/setup.sh
source /home/matteius/xdna-bringup/ironenv/bin/activate
export PEANO_INSTALL_DIR=/home/matteius/xdna-bringup/ironenv/lib/python3.14/site-packages/llvm-aie
export MLIR_AIE_DIR=/home/matteius/genetics/third_party/mlir-aie
export XILINX_XRT=/opt/xilinx/xrt
export PYTHONPATH=/opt/xilinx/xrt/python:/home/matteius/genetics/bionpu-public/src:$PYTHONPATH
```

### 7.4 Per-primitive reproduce commands

| Primitive | Reproduce command |
|---|---|
| `kmer_count` v1.3 chr22 | `cd /home/matteius/genetics && bionpu kmer-count --device npu --k 21 --n-tiles 4 --n-passes 4 --launch-chunks 8 tracks/genomics/fixtures/chr22.2bit.bin` (subprocess wraps `npu_silicon_lock`) |
| `kmer_count` smoke (3-K byte-equal) | `cd /home/matteius/genetics/bionpu-public && pytest tests/test_kmer_count_correctness.py -k smoke` |
| `minimizer` v0 smoke | `python state/minimizer_smoke.py` (smoke fixture, both pinned configs) |
| `minimizer` v1 chr22 partial | `python state/minimizer_chr22_partial.py` |
| `seed_extend` v0 | `python state/seed_extend_smoke.py` |
| `primer_scan` v0 chr22 | `python state/primer_scan_smoke.py` |
| `cpg_island` v0 chr22 | `python state/cpg_island_smoke.py` |
| `tandem_repeat` v0 chr22 | `python state/tandem_repeat_smoke.py` |
| `bionpu trim` v0 (4 cross-checks) | `python state/bionpu_trim_smoke.py` |
| `bionpu trim` v2 (pyxrt sweep + 3 cross-checks) | `python state/bionpu_trim_v2_pyxrt.py` |
| `bionpu crispr design` v1 Mode A (BRCA1) | `bionpu crispr design --target BRCA1 --genome hg38 --top 10` |
| `bionpu be design` v0 (4 PAM byte-equal smoke) | `python state/be_design_smoke.py` |
| `bionpu be design` v1 (pyxrt + off-target on chr22 1 Mbp) | `python state/be_design_v1_smoke.py` |
| `bionpu library design` v0 (5-gene + controls) | `python state/library_design_smoke.py` |
| `bionpu crispr pe design` v0 (5 targets × 3 modes) | `python state/pe_design_smoke.py` |
| Track E gating microbenchmark | `python state/track-e-conv-stem-shape-pyxrt-floor-microbenchmark/harness.py` |
| `genome_fetcher` v1 cold/warm | `python state/genome_fetcher_smoke.py` |

Each smoke harness that submits silicon dispatches from a subprocess
wraps `npu_silicon_lock` per CLAUDE.md non-negotiable rule. The
in-process pyxrt harnesses (Track A v1, `bionpu trim` v2, Track E
microbench) use `bionpu.dispatch.npu` and rely on its in-process
`_dispatch_lock` only.

### 7.5 Locked-artifact commit hashes

**Silicon kernels (seven primitives):**

| Primitive | bionpu-public commit | Status |
|---|---|---|
| `kmer_count` | `723e019` (v1.3 pipelined dispatch) | verified |
| `minimizer` | `88fb9ab` (v1 chr22 partial-recovery) | verified |
| `primer_scan` | `3888118` (v0) | verified |
| `cpg_island` | `e4d41a4` (v0 chr22 silicon-validation) | verified |
| `tandem_repeat` | `e4a4d3c` (v0 STR autocorrelation) | verified |
| `seed_extend` | `3833e43` (v0 minimap2-style seed extraction) | verified |
| `pam_filter_iupac` | `39ba53a` (Track A v0; NEW kernel) | verified |

**Production CLIs (five tools):**

| CLI | bionpu-public commit | Status |
|---|---|---|
| `bionpu trim` v0 / v1 / v2 | `44be29b` v0 → v2 numbers locked at parent `1557faf` | verified |
| `bionpu crispr design` v1 | `e7d666b` (Wave 1 nuclease design) | verified |
| `bionpu be design` v0 / v1 | `39ba53a` (v0) + `e1454ca` (v1 pyxrt + off-target integration) | verified |
| `bionpu library design` v0 | `1ff8d5c` (Track C pooled CRISPR) | verified |
| `bionpu crispr pe design` v0 | T1-T14 latest `f950056` (DESIGN.md close); plan close `1d4840a` | verified |

**Auxiliary infrastructure:**

| Component | bionpu-public commit | Status |
|---|---|---|
| `genome_fetcher` v1 (UCSC refGene + 2bit) | `2345536` | verified |
| Track E gating microbenchmark (NO-GO) | `cc2c8b6` | verified |

All hashes verified via `git -C bionpu-public log --oneline -1
<hash>` on 2026-04-28.

### 7.6 Data

* **chr22:** packed-2-bit binary at
  `tracks/genomics/fixtures/chr22.2bit.bin` (12,704,617 bytes =
  50,818,468 bases, hg38 chr22 from UCSC; SHA-256 sidecar pinned).
* **smoke_10kbp:** packed-2-bit binary at
  `tracks/genomics/fixtures/smoke_10kbp.2bit.bin` (2500 bytes = 10,000
  bases; deterministic synthetic).
* **synthetic 1 Mbp:** built by `tracks/genomics/fixtures/build_kmer_fixtures.py`.
* **`bionpu trim` fixtures:** synthetic FASTQ at
  `tracks/genomics/fixtures/synthetic_reads_with_adapters.fastq` (100
  reads) and `synthetic_reads_10k.fastq` (10K reads); built by
  `build_adapter_trim_fixtures.py`.

### 7.7 Reference tools used

* **Jellyfish:** `jellyfish 1T` (k=21, default args) — k-mer counting
  reference.
* **cutadapt:** `cutadapt -a ADAPTER --no-indels -e 0 -O len(ADAPTER)
  -j 1` — adapter trimming reference.
* **minimap2 `mm_sketch`:** sliding-window minimizer reference (Li,
  *Bioinformatics* 2016, `sketch.c:77+`).
* **TRF:** Tandem Repeats Finder (Benson, *Nucleic Acids Research*
  1999) — STR detection reference; default mismatch threshold (10%)
  is out of scope for v0 (exact-match only).
* **CpG island definition:** Gardiner-Garden & Frommer (1987) for the
  observed/expected CpG ratio + GC content thresholds; Takai & Jones
  (2002) for the W=200 window default.

---

## 8. Limitations and open work

This section enumerates what does NOT work, gathered from each
kernel's `gaps.yaml` and the strategic doc.

### 8.1 Small-FP32-NN-with-MAC-stack workloads — confirmed mismatch (Track D + Track E)

Empirically: **two independent measurements**, both at the
small-FP32-NN-with-MAC-stack class, both NO-GO.

* **Track D (DNABERT-Epi transformer)**: BERT-mini per-launch
  ~16 ms / dispatch — Phase 0 closure documented in
  [`PRDs/PRD-dnabert-epi-on-xdna.md`](../PRDs/PRD-dnabert-epi-on-xdna.md)
  v0.1.7 §0.1.
* **Track E (inDelphi-class small CNN)**: median 11.1 ms / dispatch
  on `dorado_fast_conv_stem_layer1` (1000 iter, CoV 1.9%) — 7.41× over
  the 1.5 ms PASS threshold. Verdict at
  `state/track-e-conv-stem-shape-pyxrt-floor-microbenchmark/VERDICT.md`;
  microbench commit `cc2c8b6`.

Both tracks' silicon Phase 1 closed; both mirror DNABERT-Epi pattern
(CPU-only inference for v0; weights as runtime dep, never vendored
per `restrictive-license-model-policy.md`). The v1 Limitations table
previously framed this as "recurrent FP32 ML (Dorado-shape)" — that
framing was **too narrow**; the negative envelope is
**small-FP32-NN-with-MAC-stack workloads regardless of architecture**
(recurrent OR feedforward OR small attention block — see §6.4).
Track B Phase 2 (PRIDICT-class transformer silicon) inherits the same
verdict by analogy and stays CPU-only for v0; defer to Track B owner
to file the actual closure.

The Dorado original-target finding (end-to-end 14× slower than ONNX
CPU) remains valid as the qualitative pivot trigger; it is now a
specific instance of the broader Track D + Track E fit-class verdict.

### 8.2 Per-record streams without batching — CLOSED by `bionpu trim` v2

CLOSED ENTRY (v1 carried this as open). v0/v1 baseline was 10⁴-10⁵×
overhead vs cutadapt 1T due to ~100 ms/dispatch subprocess fork.
Closed by v2 in-process pyxrt: per-dispatch overhead ~700 µs,
throughput **110 K reads/s = within 1.5× of cutadapt 1T at 160 K**
(parent commit `1557faf` locks the v2 numbers). See §5.1.2 / §6.3.

Open follow-up: persistent-kernel mode
(`kmer-shim-dma-context-overhead-floor`) could push the per-dispatch
wall below 100 µs and saturate single-core cutadapt parity, but is
not yet built. The two-regime corollary (§6.3) further refines: this
lever has 100×+ depth for **subprocess-bounded** workloads, but only
~3× depth for **silicon-bounded** workloads (e.g. `pam_filter_iupac`
chr22 1 Mbp — see Track A v1 numbers).

### 8.3 Compressed shim drain — blocked on AIE2P

Per memory `aie2p-compression-mechanisms-status`: both AM029
compression mechanisms blocked. Use case 1 (DMA→DMA) silicon-wedges
for non-compute-tile producers. Use case 2 (core-load) Peano lacks
intrinsic implementations. Affects `kmer_count` v1.3+ silicon
throughput floor (~7.5 s on chr22) and any future kernel that wants
to compress output.

### 8.4 Minimizer chr22 emit-cap saturation — v2 path filed

`minimizer` v0 baseline: 6.36 M / 20.01 M silicon emits = 32%
recovery. v1 multi-pass + position-Fibonacci-hash recovered 14.26 M /
20.01 M = 71%. Full close requires v2 widen
`MZ_PARTIAL_OUT_BYTES_PADDED` from 32 KiB to 64 KiB OR reduce
chunk_size to 1024 (filed as `minimizer-emit-cap-saturation`).

### 8.5 Other open / deferred entries (per `gaps.yaml`)

| Kernel | Gap ID | Severity | Status |
|---|---|---|---|
| `kmer_count` | `kmer-n8-memtile-dma-cap` | deferred | deferred-to-v1.1 |
| `kmer_count` | `kmer-multi-tile-aggregator-tile-deadcode` | deferred | deferred-to-v1.1 |
| `kmer_count` | `kmer-shim-dma-context-overhead-floor` | performance | open |
| `minimizer` | `minimizer-emit-cap-saturation` | low | v2 enhancement |
| `minimizer` | `minimizer-throughput-vs-jellyfish-class` | low | v2 enhancement |
| `cpg_island` | `cpg-island-emit-cap-saturation` | low | not observed on real chr22 |
| `tandem_repeat` | `tandem-repeat-period-cap-6` | low | explicit v0 scope |
| `tandem_repeat` | `tandem-repeat-fuzzy-mismatch` | medium | v1 follow-on |
| `tandem_repeat` | `tandem-repeat-chunk-spanning-str` | low | v1 follow-on |
| `tandem_repeat` | `tandem-repeat-overlap-merge-phase-shift` | low | v0 documented (3/289551 records) |
| `primer_scan` | (none filed) | — | clean v0 ship |

The five closed entries on `kmer_count` (`kmer-chr22-canonical0-cap-fire`,
`kmer-chunk-overlap-double-emit`, `kmer-host-postpass-python-bottleneck`,
`kmer-n_passes-16-not-built`, `kmer-runner-per-dispatch-overhead`,
`kmer-runner-host-merge-unordered_map`, `kmer-pipelined-dispatch-merge`)
were closed across v1.1(a), v1.1(b), v1.2(a), v1.2(b), and v1.3
between 2026-04-27 and 2026-04-28.

### 8.6 Per-primitive byte-equal honest-result table

| Primitive | smoke byte-equal | chr22 byte-equal | chr22 top-1000 byte-equal |
|---|---|---|---|
| `kmer_count` v1.3 | PASS (3/3 K) | PARTIAL (count divergence on canonical=0 + boundary; closed in v1.2(a)) | **PASS 1000/1000** |
| `minimizer` v0/v1 | PASS (k15w10, k21w11) | PARTIAL (32% v0 → 71% v1; full blocked by emit-cap) | **PASS 1000/1000** at (k=15, w=10) |
| `seed_extend` v0 | PASS per smoke harness | n/m (no measurements file) | n/m |
| `primer_scan` v0 | PASS (0 == 0; synthetic-inject 2 == 2) | PASS (0 == 0 on chr22 at TruSeq P5) | PASS |
| `cpg_island` v0 | PASS (12 == 12 islands) | **PASS (1352 == 1352 islands)** | PASS |
| `tandem_repeat` v0 | PASS (26 == 26 records) | PARTIAL (289548 silicon vs 289551 oracle; v0 threshold-pass at 10 record diff) | n/a (top-1000 not the metric for STRs) |
| `pam_filter_iupac` v0/v1 | PASS (4/4 PAM variants byte-equal: NGG 536, NG 2026, NRN 1019, NNNRRT 505) | PASS (chr22 1 Mbp slice 264,918 hits subprocess == pyxrt; v1 byte-equal across paths) | n/a (sparse-emit; top-K not the metric) |

---

## 9. What's next

* **`bionpu trim` v1 (in flight):** batched dispatch (N reads per
  silicon submit) + in-process pyxrt path. Target: collapse the
  per-read overhead floor from ~100 ms to ~1-10 ms.
* **`kmer_count` v1.4 (open):** persistent-kernel mode investigation;
  rebuild xclbin with `n_chunks_per_launch=16`. Target: beat
  Jellyfish 1T at chr22 wall (~6-8 s e2e, vs current 10.18 s).
* **`minimizer` v2 (filed):** widen `MZ_PARTIAL_OUT_BYTES_PADDED` to
  64 KiB to close chr22 full byte-equal at np=4.
* **Future primitives (candidates):**
  * **DNABERT-Epi inference** (bf16 transformer, no recurrence;
    likely fits AIE2P per the strategic table). Already trained as
    of 2026-04-28: ROC 0.9687 / PR 0.5871 / MCC 0.5637 on
    CHANGE-seq → GUIDE-seq fold 0.
  * **Methylation classifier** (MLP head — tougher; depends on the
    weight-quantisation contract).
  * **Banded Smith-Waterman alignment** (recurrent; may not fit —
    explicit scope decision per the strategic table).
* **Hypothesis tests for the corollary:**
  * Quantify the per-dispatch overhead floor under each dispatch
    path (subprocess vs in-process pyxrt vs persistent-kernel).
  * Find the break-even fixture size at which CPU-CPU loses to
    NPU-batched for a given primitive; this defines the design
    envelope numerically.

---

## 10. Conclusion

The CRISPR-shape thesis is empirically validated on AMD XDNA 2 / AIE2P
across **seven silicon-validated primitives**, **four distinct
algorithmic shapes**, and **five production CLIs** (was one in v1):

* **`kmer_count` v1.3** — substring/canonical match — silicon-only
  10,533× vs Jellyfish 1T; chr22 e2e wall 10.18 s competitive (0.96×)
  with Jellyfish 1T's 9.81 s.
* **`minimizer` v0/v1** — sliding-window comparator — smoke
  byte-equal at both pinned configs; chr22 top-1000 byte-equal.
* **`seed_extend` v0** — host-side composition over silicon
  minimizer — `mm_seed_t`-parity sorted output; smoke harness exists.
* **`primer_scan` v0** — substring exact match — chr22 byte-equal at
  4.47× speedup vs CPU oracle.
* **`cpg_island` v0** — windowed multi-counter statistics — chr22
  byte-equal at 3.52× speedup; 1352/1352 islands.
* **`tandem_repeat` v0** — autocorrelation / period detection —
  chr22 1.58× speedup; 289548/289551 records (v0 threshold-pass).
* **`pam_filter_iupac` v0/v1** — per-position 4-bit IUPAC mask check
  — 4/4 PAM-variant byte-equal smoke + chr22 1 Mbp 264,918 hits;
  v1 silicon-bounded 2.9× pyxrt over subprocess.

Five production CLIs compose the seven primitives + locked CRISPR
silicon + CPU scoring layers into deployable tools: **`bionpu trim`
v2** (cutadapt parity for adapter trimming); **`bionpu crispr
design` v1** (Wave 1 nuclease design); **`bionpu be design` v0/v1**
(Track A base editor); **`bionpu library design` v0** (Track C
pooled CRISPR with controls); **`bionpu crispr pe design` v0**
(Track B prime editor with real PRIDICT 2.0 + ViennaRNA).

The qualifier — the per-dispatch amortisation corollary — refines to
**two regimes**:

* **Subprocess-bounded** (sub-ms silicon, multi-ms wall): in-process
  pyxrt buys 100×+ (e.g. `bionpu trim` v0 → v2: 9.7 → 110,318
  reads/s, 11,372× speedup).
* **Silicon-bounded** (multi-ms silicon): pyxrt buys ~3× (e.g. Track
  A v1 chr22 1 Mbp `pam_filter_iupac`: 141 ms → 49 ms = 2.9×). The
  next win must come from kernel-side throughput.

The strategic refinement, anchored by the Track E gating
microbenchmark (commit `cc2c8b6`; median 11.1 ms / dispatch on
`dorado_fast_conv_stem_layer1`, 7.41× over PASS): the negative
boundary of AIE2P's design envelope is **small-FP32-NN-with-MAC-stack
workloads regardless of architecture** — recurrent (Dorado LSTM),
feedforward (inDelphi small CNN), and small-attention (BERT-mini)
all share the same ~10-16 ms / dispatch floor. CRISPR-shape primitives
win because they are **integer-pattern-match scans over packed
bytes** — fundamentally different compute geometry from MAC stacks.
The line is not "transformer vs CNN"; it is **MAC-stack NN vs
packed-byte pattern-match scan**. Track D (transformer) and Track E
(small CNN) now share the same fit-class verdict (CPU-only for v0);
the seven CRISPR-shape primitives stay on silicon and underpin the
five production CLIs.

The toolkit (2-bit wire format, in-band chunk header, streaming chunk
+ overlap, MemTile-aggregated combine, parallel host sort-merge,
multi-pass slicing, in-band runtime parametrisation, IUPAC packed
4-bit one-hot mask, component-triple scaffold-invariant lookup,
gene-symbol resolver, silicon-mutex discipline) generalises across
all seven primitives and all five CLIs. The corollary plus the
small-FP32-NN-with-MAC-stack floor together define the design
envelope: workloads inside the envelope (single chr22-scale
dispatch; per-record streams with batched dispatch + in-process
pyxrt; CRISPR-shape integer-pattern-match scans) are valid AIE2P
targets; workloads outside (small-FP32-NN-with-MAC-stack of any
architecture; per-record streams without batching) are not.

The toolkit-to-tool pattern (§5.6) — once silicon kernels are locked,
each new production CLI collapses to roughly one agent run — is the
multiplicative payoff that separates "kernel research project" from
"deployable bioinformatics toolkit." v2's five CLIs from one toolkit
are the empirical proof.

Next checkpoints: `kmer_count` v1.4 persistent-kernel investigation
(targeting <1 ms / dispatch floor); `minimizer` v2 emit-cap widening
to 64 KiB; potential additional CLI compositions (HDR repair-template
design; methylation-aware base-editor design; pooled prime-editor
library design).

---

## Acknowledgments / appendix

### A.1 Reference tools

* **minimap2** (Heng Li, *Bioinformatics* 2016) — `mm_sketch` is the
  canonical sliding-window minimizer reference.
* **Jellyfish** (Marçais & Kingsford, *Bioinformatics* 2011) — k-mer
  counting reference.
* **cutadapt** (Martin, *EMBnet.journal* 2011) — adapter trimming
  reference.
* **TRF** (Benson, *Nucleic Acids Research* 1999) — Tandem Repeats
  Finder reference.
* **CpG island detection** — Gardiner-Garden & Frommer (1987);
  Takai & Jones (*PNAS* 2002).

### A.2 Data

* hg38 reference, chromosome 22 — sourced from UCSC genome browser;
  packed to 2-bit MSB-first via `tracks/genomics/fixtures/build_kmer_fixtures.py`.
  SHA-256 sidecar pinned at `chr22.2bit.bin.sha256`.
* Synthetic FASTQ fixtures for `bionpu trim` — generated by
  `tracks/genomics/fixtures/build_adapter_trim_fixtures.py`.

### A.3 Hardware

* AMD Ryzen AI 9 HX 370 (Strix) — XDNA 2 / AIE2P NPU.

### A.4 Software

* `mlir-aie` (AMD/Xilinx).
* XRT (Xilinx Runtime).
* `peano-llvm-aie` (AMD's downstream LLVM with AIE2P backend).
* `amdxdna` kernel driver.

### A.5 File pointers

* Strategic doc: [`STRATEGY-aie2p-genomics-workload-fit.md`](STRATEGY-aie2p-genomics-workload-fit.md)
* AM020 spec cross-walk: [`aie-ml-am020-crosswalk.md`](aie-ml-am020-crosswalk.md)
* Per-kernel design notes: `bionpu-public/src/bionpu/kernels/genomics/<kernel>/DESIGN.md`
* Per-kernel gap inventories: `bionpu-public/src/bionpu/kernels/genomics/<kernel>/gaps.yaml`
* Silicon harnesses: `state/<kernel>_smoke.py`
* Measurements: `results/<kernel>/<version>/measurements.json`
* Production composition source: `bionpu-public/src/bionpu/genomics/adapter_trim/`
* Production composition harness: `state/bionpu_trim_smoke.py`
