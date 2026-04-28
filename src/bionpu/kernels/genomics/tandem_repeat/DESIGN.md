# bionpu_tandem_repeat — Design v0

`tandem_repeat` is the sixth CRISPR-shape genomics primitive on AIE2P
silicon. It validates a NEW algorithmic shape — autocorrelation / period
detection — that none of the prior five primitives (kmer_count,
minimizer, seed_extend, primer_scan, cpg_island) exercise.

## Algorithm

For each period `q` in `[MIN_PERIOD, MAX_PERIOD]`, walk the input and
maintain a streak counter:

```text
streak[q] = consecutive bases that match their predecessor at distance q
```

When `streak[q]` breaks (or end-of-chunk is reached), emit a record iff
`streak[q] >= q * (MIN_COPIES - 1)`. The threshold derives from: a
tandem repeat with `MIN_COPIES` copies of period `q` has length
`q * MIN_COPIES`, of which `q * (MIN_COPIES - 1)` bases match their
predecessor at distance `q` (the first `q` bases of the run can't match
yet — there is no prior copy).

The emitted record:

```text
start  = streak_start                          // first motif's first base
n_copies = (streak + q) / q                    // floored
end    = streak_start + n_copies * q           // q-aligned exclusive end
period = q
motif  = seq[streak_start : streak_start + q]
```

This is **autocorrelation streak** semantics — byte-equal between the
oracle (`bionpu.data.tandem_repeat_oracle.find_tandem_repeats`) and the
silicon kernel.

The host-side post-pass de-duplicates overlapping records using the
same greedy "longer wins; smaller period wins on ties" pass as the
oracle.

## Pinned constants

| symbol           | value | meaning                                               |
| ---------------- | ----: | ----------------------------------------------------- |
| `TR_MIN_PERIOD`  |     1 | smallest period scanned (mono-nucleotide repeats)     |
| `TR_MAX_PERIOD`  |     6 | largest period scanned (hexa-nucleotide repeats)      |
| `TR_MIN_COPIES`  |     5 | minimum copy count to emit (matches TRF default)      |
| `TR_HEADER_BYTES` |    8 | in-band chunk header size                              |
| `TR_SEQ_IN_CHUNK_BYTES_BASE` | 4096 | streaming chunk payload                       |
| `TR_SEQ_IN_OVERLAP` | 12 | chunk-overlap bytes (covers 36 + skid bases)           |
| `TR_RECORD_BYTES` |   16 | record size: u32 start \| u32 end \| u32 period \| u32 motif |
| `TR_PARTIAL_OUT_BYTES_PADDED` | 32768 | per-tile slot                              |
| `TR_MAX_EMIT_IDX` | 2046 | per-chunk emit cap                                     |

## Wire format

* Input: MSB-first packed-2-bit DNA (A=00, C=01, G=10, T=11). Same
  format as kmer_count + siblings.
* 8-byte chunk header: `[u32 actual_payload_bytes][i32
  owned_start_offset_bases]`. The kernel does not gate on the
  `owned_start_offset_bases` field; the host uses overlap-merge to
  dedup. The field is reserved for future use.
* Per-tile output slot: 32 KiB. Layout: `[u32 emit_count][N × 16-byte
  records]`.
* Per-record: `u32 start | u32 end | u32 period | u32 motif_canonical`,
  all little-endian. `motif_canonical` is the motif (period bases)
  packed MSB-first as 2-bit codes, occupying `period * 2` bits with the
  remaining high bits zero.

## Topology

Mirrors cpg_island/primer_scan: shim → broadcast → N tiles → memtile
join → shim. Every tile receives the same input chunk; tile 0's output
slot is authoritative (siblings produce identical duplicates that are
ignored). v0 ships `n_tiles=4`.

## Host post-processing

1. **Per-(period, motif) overlap-merge.** All records emitted by silicon
   are grouped by `(period, motif_canon)`. Within each group, intervals
   are sorted by `start` and merged when `cur.start <= back.end`. This
   collapses chunk-overlap duplicates and chunk-spanning streaks (where
   one chunk ends at position `e` and the next chunk starts the same
   streak at position `s <= e`) into a single record covering the union
   range.

2. **Cross-period dedup.** After per-(period, motif) merge, sort all
   records by `(start asc, length desc, period asc)` and apply a greedy
   first-fit pass: keep a record only if its start is `>= last_end`.
   This ensures each base position lives in at most one record, with
   longer repeats winning over shorter ones (and equal-length ties
   resolved by smaller period — i.e., more copies — first).

The result is byte-equal with `find_tandem_repeats` on the same input.

## Scope

v0 supports periods 1..6 with 5+ copies on the forward strand only. v1+
defers:

* Periods 7..100 (heavy state for the longer periods).
* Mismatch tolerance / fuzzy matching (TRF supports up to 10% mismatch).
* Quality-aware variant calling.
* Indel allowance.
* STRs that span > one chunk payload (~16384 bp) — covers the rare
  disease loci on chr22.

## Forward-strand only

Tandem repeats are conventionally reported on the reference strand
(motif on the forward strand). v0 does NOT compute reverse-complement
matches — this matches TRF's default output and the oracle's contract.

## Why the per-period scan is byte-equal with the oracle

For period `q`, the silicon scans `seq[p] == seq[p - q]` and the oracle
scans `seq[i:i+q] == seq[j:j+q]` (greedy). When the motif transitions
at a non-q-aligned boundary (e.g., "AGCT…AGCTTTTT…"), the silicon
**autocorrelation** start can be up to `q-1` bases earlier than the
oracle's greedy start at the same period. To make the silicon and
oracle produce byte-equal output, the **oracle is defined to use the
silicon's autocorrelation streak semantics** (rather than greedy scan).

The two semantics identify the same biological STR with the same number
of copies — only the byte-boundary representation differs. Pinning the
autocorrelation semantics keeps the silicon kernel as a single
per-period streak counter (28 bytes of state for periods 1..6), which
is the simplest possible silicon realization.
