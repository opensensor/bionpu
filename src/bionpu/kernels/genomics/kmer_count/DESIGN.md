# bionpu_kmer_count — k-mer counting on AIE2P (DESIGN, v0.5)

Per `state/kmer_count_interface_contract.md` (T1) v0.5 — symbols,
ObjectFifo names, constants, the streaming chunk + overlap protocol,
and the multi-pass hash-slice partition are pinned there. This document
expands PRD §4 + the v0.5 redesign into a kernel-local design rationale,
captures the silicon-validated numbers from T13 / T14 / T15, and pins
the v1.0 ship boundary against the v1.1 follow-on backlog enumerated in
`gaps.yaml`.

---

## §1 — Strategic context

See `docs/STRATEGY-aie2p-genomics-workload-fit.md` for the umbrella
thesis. K-mer counting is the v1.0 ship target post-pivot: it shares
the CRISPR pam-filter / match-multitile shape (per-byte streaming
canonical math + memtile-aggregated fan-in + sparse host emit) and
empirically validates the strategic claim that AIE2P silicon's
sweet-spot is **dense per-byte arithmetic**, NOT large stateful tables.
The v1.0 silicon-only result of 10,533× vs Jellyfish 1T on chr22
(silicon-only avg 951.7 µs/iter; Jellyfish 1T 9.81 s) is the gate met
for PRD §3.3.

---

## §2 — v0.5 architecture (streaming + multi-pass)

The v0.5 build pinned by T1's "REDESIGN" section is the shipped
kernel. It supersedes the original in-tile counting design (T11
first round, retained in source files behind unreferenced symbols
for v1.1+ research; not linked into the v0.5 xclbins).

### Topology

```
shim ─seq_in─ broadcast ──▶ tile_0 .. tile_{N_TILES-1}
                                  │
                            (compute canonical;
                             emit if slice == pass_idx)
                                  │
                                  ▼
                           partial_count_<i>
                                  │
                      memtile .join(N_TILES, ...)
                                  │
                                  ▼
                              shim drain
```

### Per-tile kernel body

```cpp
extern "C" void kmer_count_tile_k{K}(
    uint8_t*  __restrict packed_in,    // chunk + per-k overlap
    uint8_t*  __restrict partial_out,  // 32 KiB pass-slot (one per chunk × tile)
    int32_t   n_input_bytes,
    int32_t   pass_idx,                // baked at IRON-emit time
    int32_t   n_passes_log2);          // baked at IRON-emit time
```

The kernel walks the packed-2-bit stream MSB-first per byte, rolls
two uint64 registers (forward + reverse-complement), recomputes
`canonical = min(fwd, rc)` per base, and emits only when
`(canonical >> SLICE_HASH_SHIFT) & ((1 << n_passes_log2) - 1) == pass_idx`.

Output blob layout per (chunk × tile): `[uint32 emit_idx LE][emit_idx
× uint64 canonical]` packed into a 32 KiB padded slot.

### No in-tile counting

The original T11 design carried a per-tile open-addressed hash table
with emit-on-evict overflow. T13 first-round silicon validation
showed:

* 512 buckets/tile × 4 tiles = 2048 total slots
* chr22 has ~30 M unique canonicals
* observed chain-overflow rate ≈ 99.99 % — the table was effectively
  bypassed at the centromere
* PRD §3.2 byte-equal vs Jellyfish was unachievable

The v0.5 redesign moves all counting to the host. Silicon's job is
just per-base byte-math (rolling fwd/rc canonicalisation + hash-slice
filter); the host accumulates `std::unordered_map<uint64_t,
uint64_t>` after all passes.

### Multi-pass partition

To stay inside the per-pass shim DMA budget at chr22 scale, the host
runs N silicon dispatches per fixture, one per `pass_idx ∈ [0, N)`.
The kernel's hash-slice filter ensures each canonical lands in
**exactly one** pass — no double-counting, no dropped k-mers.
Coverage = 100 %.

`SLICE_HASH_SHIFT = 0` — the slice index uses the canonical's low
bits. With `SUPPORTED_N_PASSES = (1, 4, 16)` and `n_passes_log2 ∈
{0, 2, 4}`, per-pass output volume scales as `1 / N_PASSES`.

### Memtile-aggregated combine

The N tiles' partials fan into the memtile via `outC.prod.join(N, ...)`
— same pattern as
`bionpu/kernels/crispr/match_multitile_memtile/multitile_memtile.py`
(silicon-validated). For v1.0, only `n_tiles ∈ {1, 2, 4}` are built;
`n_tiles = 8` blows the AIE2P MemTile S2MM channel budget (6
channels) and is filed as `kmer-n8-memtile-dma-cap` (deferred to
v1.1).

---

## §3 — Tile DM budget (post-v0.5)

AIE2P CoreTile: 64 KiB DM total, shared across stack, ObjectFifo
ping-pong buffers, .text, .rodata, and per-tile state. The v0.5
budget is:

| Buffer | Depth | Bytes per element | Total |
|---|---:|---:|---:|
| `seq_in_buff` (chunk + overlap) | 2 | ~4 100 | ~8 KiB |
| `partial_out_buff` (pass-slot) | **1** | 32 768 | 32 KiB |
| stack | — | — | ~1 KiB |
| .text + .rodata | — | — | ~2 KiB |
| **Total tile DM** | | | **~43 KiB** |
| Headroom (vs 64 KiB cap) | | | **21 KiB** |

**Critical: `partial_out` depth=1 (no ping-pong).** Depth=2 × 32 KiB
= 64 KiB would exhaust the budget on its own. Depth=1 means the
kernel must finish writing before the next DMA submit; this
serialises slightly but is acceptable at v0.5 — the alternative
(halving chunk size to 2 KiB) doubles dispatch count and was
empirically slower in T11 sweeps.

The original T1 contract specified 4096 buckets × 12 byte CountRecord
= 48 KiB primary table plus depth=2 partial ping-pong (= 96 KiB
tile-side). T11 first round failed to build because depth=2 ×
partial blew 64 KiB. The intermediate fix (1024 buckets × 12 = 12
KiB) shipped a build but T13 then surfaced the capacity ceiling
above. v0.5 drops the in-tile table entirely.

---

## §4 — Per-k mask discipline

Three per-k canonical bit masks pinned in `kmer_count_constants.h`:

```cpp
KMER_MASK_K15 = (1ULL << 30) - 1ULL;   // 30-bit canonical
KMER_MASK_K21 = (1ULL << 42) - 1ULL;   // 42-bit canonical
KMER_MASK_K31 = (1ULL << 62) - 1ULL;   // 62-bit canonical
```

Apply the mask on **every** rolling update — both the forward and
the reverse-complement registers. Specifically:

```cpp
forward = ((forward << 2) | new_base) & KMER_MASK_K{K};
rc      = (rc >> 2) | (((uint64_t)(new_base ^ 0x3)) << (2 * (k-1)));
rc      = rc & KMER_MASK_K{K};
canonical = forward < rc ? forward : rc;
```

K=31 is **load-bearing** (62-bit canonical width on a 64-bit register
type): without the mask, the high bits leak across rolls and
canonical = min(fwd, rc) silently corrupts. The T2 numpy oracle
applies the same mask discipline; the smoke fixture's 3 k values ×
byte-equal gate validates the contract.

---

## §5 — Streaming chunk + 4-byte alignment

```cpp
SEQ_IN_CHUNK_BYTES_BASE = 4096
```

Per-k overlap = `ceil((k-1) / 4)` bytes, rounded up to keep the
chunk total 4-byte-aligned (aiecc `aie.dma_bd` rejects unaligned
shapes). Pinned values:

| k | overlap (bytes) | chunk + overlap |
|---:|---:|---:|
| 15 | 4 | 4 100 |
| 21 | 8 | 4 104 |
| 31 | 8 | 4 104 |

(K=21 needs only 5 bytes for the (k-1)=20 base k-mer span; rounded
up to the nearest 4-byte multiple = 8 bytes.)

Without overlap, k-mers spanning chunk boundaries are dropped
silently — a correctness fail vs Jellyfish. The runner's overlap
emission policy is the one open issue at chr22 scale (see §8 below
+ `gaps.yaml:kmer-chunk-overlap-double-emit`).

---

## §6 — Hash-slice partition correctness

```cpp
const uint64_t shift_drop = SLICE_HASH_SHIFT;   // 0 = low bits
const uint64_t slice_mask = (1ULL << n_passes_log2) - 1ULL;
const uint64_t slice = (canonical >> shift_drop) & slice_mask;
if ((int32_t)slice != pass_idx) continue;
```

Coverage proof:

* Every canonical k-mer maps to a single `slice` value via the
  bit-AND. Domain = `[0, 2^(2k))`. Range = `[0, n_passes)`.
* For each canonical, `slice == pass_idx` is true for exactly one
  pass index (by definition of bit-AND with a power-of-2 modulus).
* The host runs all `n_passes` dispatches, accumulating
  `counts[canonical] += 1` per emitted canonical; the union of N
  partitions equals the full canonical multiset.

**No double-counting, no dropped k-mers** at the slice-filter level.
(Boundary double-counting at the chunk-overlap level is the runner-
side issue filed as `kmer-chunk-overlap-double-emit`.)

`SLICE_HASH_SHIFT = 0` (low bits) is a deliberate choice: the low
2 bits of canonical encode the last base of the lexicographically-
smaller orientation, which is reasonably well-distributed for ACGT
input. Higher shifts (e.g. shift = 2k - log2(N) per the original
contract sketch) distribute equally well for synthetic data but
break down at the centromere where canonical=0 (all-A) is wildly
over-represented; the all-A run lands in **pass 0** regardless of
shift, and pass 0 is then over-represented by ~12 M canonicals on
chr22. The cap-fire at 4095 emits per chunk × 4 passes × 3102
chunks then truncates the count for canonical=0 to ~2.98 M instead
of the oracle's ~11.67 M. Filed as `kmer-chr22-canonical0-cap-fire`.

---

## §7 — Silicon-validated throughput

Cited from `results/kmer/v1-silicon-validation/measurements.json`
(T13), `results/kmer/v1-chr22/measurements.json` (T14), and
`results/kmer/v1-bench-{iso}/measurements.json` (T15).

### Smoke fixture (10 Kbp, k=15/21/31)

* NPU silicon-only avg: **~700–865 µs/iter** across the 3 k values
  (T15 v1-bench measurements).
* CPU oracle (Jellyfish 1T): **~0.5–0.6 s** per k value.
* Silicon-only speedup vs Jellyfish 1T: **~700×** to ~880× per k.
* End-to-end (incl. Python host post-pass): ~5× faster than
  Jellyfish 1T at this scale (host post-pass overhead is amortised
  by the small ~10K canonical set).
* Byte-equal vs CPU oracle: **PASS** at all 3 k values (T13 +
  T16 retest log).

### Synthetic 1 Mbp fixture (k=15/21/31)

* NPU silicon-only avg: **~900–940 µs/iter** across 3 k values.
* CPU oracle (Jellyfish 1T): ~0.79–0.89 s per k value.
* Silicon-only speedup: **~870×** to ~945×.
* End-to-end: ~1.2–1.4× faster than Jellyfish 1T (Python host post-
  pass over ~1 M canonicals starts to dominate).

### chr22 (50 Mbp, k=21, n_tiles=4, n_passes=4) — T14 result

* NPU silicon-only avg: **951.7 µs/iter** (3 iters, 1 warmup).
* End-to-end wall (incl. Python host): **32.7 s/iter**.
* Jellyfish 1T: **9.81 s** total.
* Jellyfish 8T: **1.75 s** total.
* **Silicon-only speedup vs Jellyfish 1T: 10 533×** (PRD §3.3 gate
  met).
* Silicon-only speedup vs Jellyfish 8T: **1 842×**.
* End-to-end speedup vs Jellyfish 1T: **0.31×** (Python `dict` over
  30 M canonicals = bottleneck; filed as
  `kmer-host-postpass-python-bottleneck`).
* Top-1000 canonical Jaccard vs Jellyfish: **0.988** — high
  agreement on which canonicals are popular, count distribution
  diverges at canonical=0 (cap-fire) + ~3-8 excess on each canonical
  at the chunk boundary (overlap double-emit).

---

## §8 — Known issues / v1.1 follow-on

All items in this section are filed in `gaps.yaml`; the v1.0 build
ships with them deferred. The headline result (silicon-only
10 533× vs Jellyfish 1T + smoke byte-equal) is the v1.0 ship.

### chr22 + n_passes=4 — canonical=0 cap-fire

At `n_passes=4`, the all-A canonical = 0 falls in slice 0 (since
`0 & (4-1) == 0`). chr22's centromere has ~12 M all-A k-mers
(N → A scrub artefact during 2bit packing). With chr22 split into
~3 102 4 KiB chunks, pass 0 sees ~12 M / 3 102 ≈ 3 870 all-A k-mers
per chunk — close to the per-chunk emit cap of `MAX_EMIT_IDX_V05 =
4095`. After the cap fires + chunk-overlap double-emits, the
host-side count for canonical=0 is ~2.98 M instead of the oracle's
~11.67 M. Workaround: **n_passes=16** (untested in v1.0 — only
n_passes=4 was built; v1.1 builds + benchmarks the missing
variants). Filed as `kmer-chr22-canonical0-cap-fire`.

### Chunk-overlap edge double-emit

The runner's chunked dispatch counts boundary k-mers in two adjacent
chunks because the overlap region is processed by both. Excess on
chr22: 3-8 per canonical, cumulated across ~3 102 chunks. This is
the second-order error explaining the 6-9 unit count divergence
between NPU and Jellyfish at the top-1000 canonicals (T14 diffs at
positions 1–47). Fix in v1.1: emit only when start-index ∈ chunk's
"owned" range (not the overlap region). Filed as
`kmer-chunk-overlap-double-emit`.

### n_tiles = 8 deferred

`outC.prod.join(N=8, ...)` exceeds the AIE2P memtile S2MM channel
budget (6 channels). The two-stage memtile fan-in path or the
multi-memtile-column path is v1.1 work. Filed as
`kmer-n8-memtile-dma-cap`.

### Host post-pass Python bottleneck

End-to-end at chr22 scale is dominated by the Python `dict`
accumulation across N passes × N tiles × ~3 102 chunks. The C++
runner already uses `std::unordered_map`, but the op-class
`__call__` parses Jellyfish-FASTA output through Python, which is
~33 s of pure-Python overhead per chr22 iter. Fix in v1.1: native
ndarray return path or in-process pyxrt reading raw blobs. Filed
as `kmer-host-postpass-python-bottleneck`.

### n_passes ∈ {1, 16} not built

`SUPPORTED_N_PASSES = (1, 4, 16)` is kernel-supported, but only
`n_passes=4` artifacts exist in `_npu_artifacts/`. v1.1 builds and
benches all 3 variants. Filed as `kmer-n_passes-16-not-built`.

### Aggregator CoreTile dead source

`kmer_count_aggregator.cc` is unused in the v0.5 build (memtile-
join replaces the aggregator CoreTile path). The file is retained
for v1.1 research (n_tiles ≥ 8 may revive a hierarchical
CoreTile-then-memtile fan-in). Filed as
`kmer-multi-tile-aggregator-tile-deadcode`.

---

## §9 — Validation protocol

* **T2 numpy oracle**: `bionpu/data/kmer_oracle.py` —
  `pack_dna_2bit`, `unpack_dna_2bit`, `canonical_kmer_2bit`,
  `count_kmers_canonical`. The reference for byte-equal gating.
* **T3 fixtures**: `tracks/genomics/fixtures/build_kmer_fixtures.py`
  produces deterministic `smoke_10kbp.2bit.bin`,
  `synthetic_1mbp.2bit.bin`, `chr22.2bit.bin` with sha256 sidecars
  + per-k expected JSON for the smoke gate.
* **T12 correctness tests**: `bionpu-public/tests/test_kmer_count_correctness.py`
  — smoke byte-equal at all 3 k values vs the T2 oracle.
* **T13 silicon validation**: `results/kmer/v1-silicon-validation/
  measurements.json` — chr22 silicon dispatch + jaccard against the
  oracle + dmesg pre/post wedge probe. Confirmed PASS for top-100
  canonical Jaccard = 1.0 at k=21 / n=4 / n_passes=4 (count
  divergence is the v1.1 cap-fire / chunk-overlap issues, not a
  jaccard fail).
* **T14 Jellyfish ground-truth**: `results/kmer/v1-chr22/measurements.json`
  — chr22 vs Jellyfish 1T / 8T at k=21 / n=4 / n_passes=4. Top-1000
  Jaccard = 0.988; silicon-only speedup 10 533× vs Jellyfish 1T;
  byte-equal verdict = FAIL (v1.1 follow-on).
* **T15 end-to-end bench**: `bionpu-public/benchmarks/genomics/
  bench_kmer_count.py` + `run_kmer_bench.sh`. Sweeps
  (smoke / synthetic / chr22) × {k=15, k=21, k=31} × {n_tiles=4,
  n_passes=4} under one wrapping `npu_silicon_lock`. Captures
  per-cell NPU avg/min/max wall + Jellyfish 1T/8T wall + speedups.
  Latest result tree: `results/kmer/v1-bench-{iso}/measurements.json`.
* **T16 CLI**: `bionpu kmer-count --device {cpu,npu} --k {15,21,31}
  --launch-chunks {1,2,4} ...` — silicon-mutex wrapped per CLAUDE.md
  non-negotiable rule. Smoke fixture byte-equal CPU vs NPU
  confirmed (T16 v0.5 retest log line 1064).

---

## §10 — Cross-reference to gaps.yaml

This DESIGN.md surfaces six items; all six live in the kernel-local
`gaps.yaml`:

1. `kmer-n8-memtile-dma-cap` — deferred (existing).
2. `kmer-chr22-canonical0-cap-fire` — performance/correctness, v1.1.
3. `kmer-chunk-overlap-double-emit` — correctness, v1.1.
4. `kmer-host-postpass-python-bottleneck` — performance, v1.1.
5. `kmer-n_passes-16-not-built` — build-side / completeness, v1.1.
6. `kmer-multi-tile-aggregator-tile-deadcode` — research deferred.

Each entry has `attempted`, `failed_with`, `root_cause`,
`workaround`, `severity`, `status` per the T1-pinned schema.
