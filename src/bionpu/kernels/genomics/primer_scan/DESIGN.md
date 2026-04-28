# primer_scan v0 — DESIGN

Fourth CRISPR-shape genomics primitive on AIE2P silicon (after
`kmer_count`, `minimizer`, and `seed_extend`). Per-byte primer /
adapter exact-match scan against a packed-2-bit query. Reuses ~80% of
the kmer_count + minimizer infrastructure.

## Goal

Given a packed-2-bit DNA query buffer and a primer (5–30 bp ASCII
ACGT, default Illumina TruSeq P5 adapter `AGATCGGAAGAGC`), emit every
position where the primer occurs as an exact match on either strand
(forward + reverse-complement).

## Why this fits AIE2P even better than minimizer

* Same per-byte 2-bit math (rolling fwd / rc registers).
* No sliding-window state — just compare current k-mer (k = primer
  length) to a small set of pinned canonicals.
* Emit volume is LOW: typical adapter occurs 0–2 times per read; even
  chr22 has only a few thousand TruSeq P5 sites.
* No cap-fire risk (`PS_MAX_EMIT_IDX = 2046` per slot is many orders
  of magnitude above the realistic emit density).
* Memtile-aggregated combine pattern: same as kmer_count.

## Wire format

### Input (host → kernel)

* Packed-2-bit DNA, MSB-first: A=00, C=01, G=10, T=11. Same wire
  format as `kmer_count` and `minimizer`.

### In-band chunk header (24 bytes)

| offset  | type     | field                               |
|---------|----------|-------------------------------------|
| 0..3    | uint32 LE| `actual_payload_bytes`              |
| 4..7    | int32 LE | `owned_start_offset_bases`          |
| 8..15   | uint64 LE| `primer_fwd_canonical`              |
| 16..23  | uint64 LE| `primer_rc_canonical`               |

Payload starts at byte 24. The 8-byte prefix matches kmer_count v1.2(a)
+ minimizer; the 16-byte primer canonical pair is the v0 extension.

`owned_start_offset_bases` for chunk *i*:
* `i = 0`: 0 (the whole payload is owned).
* `i > 0`: `(overlap_bytes * 4) - (P - 1)`. The kernel's owned-range
  gate drops emits whose k-mer-start position falls below this offset
  so duplicate emits in the chunk overlap region are suppressed.

### Output (kernel → host)

Per-tile pass-slot (32 KiB). Layout per slot:

| offset                          | type     | field                       |
|---------------------------------|----------|-----------------------------|
| 0..3                            | uint32 LE| `emit_count`                |
| 4 + 16·i + 0..3                 | uint32 LE| record `i` `query_pos`      |
| 4 + 16·i + 4                    | uint8    | record `i` `strand` (0/1)   |
| 4 + 16·i + 5                    | uint8    | record `i` `primer_idx` (0) |
| 4 + 16·i + 6..7                 | uint16   | reserved (zero)             |
| 4 + 16·i + 8..15                | uint64   | reserved (zero)             |

`strand = 0` means the forward k-mer at `query_pos` equalled
`primer_fwd_canonical`; `strand = 1` means the rc k-mer did, i.e. the
query's substring at `query_pos` is the RC of the primer.

Palindromic primers (where `primer_fwd_canonical == primer_rc_canonical`)
emit only the forward record at any matched position to avoid double
emit. The CPU oracle has the same gate.

### Host-side aggregated blob

The runner's `--output-format binary` emits:

```
[u64 n_records LE][n × 16-byte records as above]
```

`record.query_pos` in the host-side blob is GLOBAL (chunk local
position + chunk's `src_offset * 4`). Records are sorted by
`(query_pos asc, strand asc)` and de-duplicated.

## Algorithm — per-tile

```
For each new ACGT base entering at index i:
  fwd = ((fwd << 2) | base) & MASK
  rc  = (rc >> 2) | ((base ^ 3) << (2*(P-1)))
  valid_run += 1
  If valid_run < P: continue.
  kmer_start = i - (P - 1)
  Owned-range gate: kmer_start >= owned_start_offset_bases
  fwd_match = (fwd == primer_fwd)
  rc_match  = (rc == primer_fwd) && (primer_fwd != primer_rc)
  If fwd_match: emit (kmer_start, strand=0)
  If rc_match:  emit (kmer_start, strand=1)
```

The match check is one uint64 compare + branch (after the rolling
math). Trivial.

## Per-P build-time constants

| P  | mask                  | RC_HIGH_SHIFT | symbol                   |
|----|-----------------------|---------------|--------------------------|
| 13 | `(1 << 26) - 1`       | 24            | `primer_scan_tile_p13`   |
| 20 | `(1 << 40) - 1`       | 38            | `primer_scan_tile_p20`   |
| 25 | `(1 << 50) - 1`       | 48            | `primer_scan_tile_p25`   |

## Path B (runtime primer canonical)

The primer canonical lives in the chunk header (16 bytes) rather than
as a build-time arith.constant. This means a single xclbin per
primer-length P handles ANY primer of that length at runtime — no
xclbin rebuild when a user wants to scan a different adapter. The
runtime cost is 16 bytes/dispatch (negligible vs the 4096-byte
payload).

Cost vs Path A (per-primer xclbin):

* Build artifact count: 3 (one per P) vs 3 × N_primers.
* Per-dispatch DMA payload: +16 bytes vs unchanged.
* Kernel code size: +1 64-bit memcpy from header into a register-
  backed local at function entry, plus the comparison branches that
  Path A would also need. Net: indistinguishable.

Path B was chosen for v0 because real adapter-trimming pipelines use
many primers (16-primer barcoded sets are common in nanopore reads),
and rebuilding an xclbin per primer is hostile to that workflow.

## Topology

Same broadcast topology as `minimizer` v0 / v1:

```
shim ─seq_in─ broadcast ──▶ tile_0 .. tile_{N_TILES-1}
                                  │
                            (compute fwd + rc;
                             compare to primer_fwd in header;
                             emit on match.)
                                  │
                            partial_primer_<i>
                                  │
                      memtile .join(N_TILES, ...)
                                  │
                                  ▼
                              shim drain
```

Every tile receives the SAME input chunk (broadcast), so every tile
produces the SAME hit list. The host runner reads tile_0's slot only
and ignores duplicates from siblings. (This is the simplest topology
to ship; per-tile partition is a future optimisation.)

## Tile DM budget

| component                           | bytes |
|-------------------------------------|-------|
| seq_in element (4096 + 8 overlap + 24 header — but Python doesn't model header explicitly; we widen the chunk) | 4104 |
| seq_in dbl-buffered                 | ~8208 |
| partial_out element                 | 32768 |
| partial_out single-buffered (depth=1)| 32768 |
| **per-tile total**                  | **~41 KiB** |

Well within the 64 KiB AIE2P CoreTile DM cap.

NOTE: the IRON Python's `seq_chunk_ty` is sized at `SEQ_CHUNK_BYTES +
SEQ_OVERLAP_BYTES = 4104` bytes. The 24-byte header occupies bytes
[0..23] of that buffer; the kernel reads `n_input_bytes - 24` bytes
of payload starting at byte 24. This is the same accounting trick
kmer_count + minimizer use.

## Acceptance gates (v0)

1. **CPU oracle TDD ≥4 tests pass** — DONE (7 tests in
   `tests/test_primer_oracle.py`).
2. **Smoke (P=13, TruSeq P5) silicon byte-equal vs CPU oracle** —
   gated on artifact build.
3. **chr22 (P=13) silicon byte-equal vs CPU oracle** — gated on
   artifact build.
4. **chr22 e2e wall ≤ 2 s** — performance target.
5. **Build clean for at least one (P, primer) artifact** — gated on
   silicon build env.

## Out of scope for v0 (deferred to v1+)

* Multi-primer set (16-primer barcoded sets common in nanopore).
* ≤1 mismatch tolerance (Hamming distance).
* Quality-aware soft matching.
* Per-tile work partition (e.g. each tile scans 1/N_TILES of the
  primer set in a multi-primer build).
* Hash-slice multi-pass (not needed — emit volume is far below cap).

## Comparison to sibling primitives

| primitive   | per-byte work | output volume          | sliding-window state | hash table | ship status |
|-------------|---------------|------------------------|----------------------|------------|-------------|
| kmer_count  | rolling fwd/rc, slice filter | one record per emit; ~bases/2 unique | None | host-side | shipped v1.x |
| minimizer   | rolling fwd/rc, ring + min  | ~2/(w+1) bases | w-deep ring | None | shipped v0/v1 |
| seed_extend | re-uses minimizer artifact | seed-extension chains | None (uses minimizer)| None | shipped v0 |
| **primer_scan** | rolling fwd/rc, single uint64 compare | ~few bases per primer occurrence (very low) | None | None | this PR |

primer_scan is the simplest of the four: one fewer state element
than minimizer (no ring buffer), no hash table, low emit volume, no
slice / pass machinery.
