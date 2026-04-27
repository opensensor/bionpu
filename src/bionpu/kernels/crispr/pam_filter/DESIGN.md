# — PAM filter + threshold + sparse-emit (C-M5)

CRISPR PRD §4.3 + plan . Builds on 's multi-tile dataflow by
adding **on-tile NGG PAM check at Tile A** (the filter-early path) and
**on-tile threshold + sparse-emit at Tile Z** (replaces 's host-side
filter). Output bytes are byte-equal to Cas-OFFinder's normalized
canonical TSV (after `bionpu.data.canonical_sites.normalize`) for
chr22 × 10 guides.

## Topology

```
                      shim DMA in  ──guides──→ broadcast
                          │
                          ↓
                      Tile A (PAM filter / dispatcher)
                          │
                          ├─→ windows_out (broadcast to match tiles)
                          │
                          └─→ pam_meta (1 byte/window) ──→ Tile Z
                          │
                          ├─→ match_tile_0: guides[0:64]
                          │     ↓ partial_0 (64 windows × 64 guides)
                          │
                          ├─→ match_tile_1: guides[64:128]
                          │     ↓ partial_1
                          │
                          └─→ Tile Z (threshold + sparse-emit)
                                ↓ sparse_out (length-prefixed records)
                              shim DMA out
```

Plus an `windows_in` ObjectFifo (shim → Tile A) carrying 6 bytes per
window record (5 spacer bytes + 1 PAM byte).

## §1 — Filter-early vs filter-late

Both variants ship from the same Python source and the same C++ object
(`tile_a_filter.o`). They produce **byte-identical sparse hit-list
output** (after normalization) on the same input. They differ in:

| step                  | filter-early                    | filter-late                       |
|-----------------------|---------------------------------|-----------------------------------|
| Tile A PAM check      | yes — drops 7/8 of windows      | no — passes everything through    |
| match-tile work       | full (every window — see below) | full (every window)               |
| Tile Z PAM check      | no (Tile A already did it)      | yes — re-checks via pam_meta byte |
| Tile Z threshold      | yes                             | yes                               |
| sparse output bytes   | identical                       | identical                         |
| match-tile *useful* cycles | ~12.5% of windows         | 100% of windows (7/8 wasted)      |

**Important v1 caveat (also documented in `gaps.yaml`)**: the
match-tile kernel does NOT yet branch on the pam_meta byte. Both
variants therefore pay the same match-tile *cycle* cost; what changes
is which windows produce *meaningful* match-count output. The
sparse-emit at Tile Z drops the PAM-failing windows BEFORE they reach
the output ring buffer, so DMA-out volume is what actually shrinks
8× in filter-early. / may fold the PAM check into the match
kernel itself (producing real cycle savings), but for v1 we keep the
match kernel byte-equal to so the byte-equality gate is direct.

The headline metric reported in `results/crispr/c-m5/measurements.json`
is therefore the **sparse-output volume ratio** + the
**Tile-Z-emit-cycle ratio**, NOT the wall-clock ratio (which is
dominated by shim-DMA setup + the unchanged match-tile cycles).

## §2 — Window record byte layout

Per Tile-A input record (6 bytes total):

| offset | bytes | content                                     |
|--------|-------|---------------------------------------------|
| 0 | 5 | 20-nt spacer, 2-bit packed (A=00,C=01,G=10,T=11) — same as / |
| 5      | 1     | 3-nt PAM context: bits 1:0=pam[0], 3:2=pam[1], 5:4=pam[2], 7:6=padding=0 |

The spacer encoding is **identical** to / — the match-tile
kernel reads the first 5 bytes of each record exactly as it did before.
The 6th byte is consumed by Tile A and the corresponding pam_meta byte
is forwarded to Tile Z.

## §3 — Strand handling (host-driven, not in-tile)

Two passes through Tile A: forward strand + reverse-complement.
Host-side glue:

1. Build forward window stream: for each genomic position, pack
   `seq[s : s+20]` into 5 spacer bytes and `seq[s+20 : s+23]` into 1
   PAM byte. Dispatch as Pass 1.
2. Build reverse-strand stream: for each genomic position, pack
   `RC(seq[s : s+23])` into the same 6-byte layout (the spacer is the
   RC of `seq[s+3 : s+23]` and the PAM is the RC of `seq[s : s+3]`).
   Dispatch as Pass 2.
3. Tile A treats both passes identically — the strand label is host-
   side bookkeeping attached to each chunk's window-index range.

**Why not in-tile RC?** Two reasons:

1. **Program memory budget**: RC requires per-base 2-bit complement
   (bitwise NOT then byte-reverse) which roughly doubles Tile A's
   PM budget without reducing DMA volume.
2. **Sparse-emit clarity**: the strand stamp on each emit record is
   host-side metadata; mixing strands inside one Tile A run would
   require per-window strand bits in the record and bloat the DMA
   payload.

Cost of two passes: 2× the shim-DMA volume on the windows_in stream.
That's amortized by the existing PAM filter at Tile A — only ~12.5%
of windows survive to the match tiles, so the *match-tile* DMA volume
is unchanged from a single-pass run on the same density.

## §4 — Sparse emit record layout

8 bytes per surviving (guide, window) pair:

| offset | bytes | content                              |
|--------|-------|--------------------------------------|
| 0      | 2     | window_idx (uint16, little-endian)   |
| 2      | 1     | guide_idx (uint8, 0..127)            |
| 3      | 1     | mismatches (uint8, 0..20)            |
| 4      | 4     | reserved (zero — host stamps strand + chrom from window_idx) |

Tile Z prefixes each ring slot with a **uint32 record count** at
offset 0. The host reads this count after each chunk-DMA-out and
appends the records to its growing sparse hit list.

## §5 — Tile memory budget

Per `results/crispr/c-m5/measurements.json`:

| tile          | resident / dbl-buf in / dbl-buf out      | total bytes | % of 64 KiB |
|---------------|-------------------------------------------|-------------|-------------|
| Tile A        | — / 2*64*6 = 768 / 2*64*5 = 640 + 2*64*1 = 128 | **1536** | 2.3%   |
| match_0/1 | 640 / 640 / 8192 (unchanged from ) | **9472** | 14.4% |
| Tile Z        | — / 2 partials × 16384 / sparse 4096 / pam_meta 256 | **20736** | 31.6%   |

Peak (Tile Z) is **20.7 KiB**, *down* from 's 32 KiB joiner peak
because we no longer hold the dense window-major output buffer —
sparse emit replaces it.

## §6 — Strand pass throughput accounting

Per launch pair (forward + reverse):

* `windows_in` shim-DMA: 2 × 4096 × 6 B = 49152 B
* `sparse_out` shim-DMA per pass: ~4 + 8 × N_records bytes
  (N_records on chr22 random ACGT ≈ 3-5 K per 4096-window chunk
  at threshold 4 — measured in the c-m5 run summary).

For the chr22 × 10 guides byte-equality fixture, each strand pass
produces ~15 K records out of a worst-case 4096 × 128 = 524 288 cells.

## §7 — Toolchain gaps surfaced

See `gaps.yaml`. Key entries:

* — IRON has no first-class "stream-conditional forward"
  pattern. We work around by always forwarding a fixed-size window
  out of Tile A (with PAM-failing windows zero-filled) and gating at
  Tile Z via the pam_meta byte. Real cycle savings on the match
  tiles need a kernel-level branch.
* — The 2-input-DMA-channel constraint inherited from
   still applies. Memtile-aggregated fan-in to recover the
  4-tile match parallelism is deferred to (it's not strictly
  required for filter-early to be a win on its own terms).

## §8 — What this design does NOT do (and why)

* No in-tile reverse-complement (host pre-flips; see §3).
* No in-kernel match-tile PAM branch (preserves byte-equality
  guarantee; deferred to / ).
* No memtile-aggregated 4-match-tile fan-in.
* No IUPAC ambiguity codes in the PAM template (NGG is hardcoded;
  IUPAC would be a separate kernel variant — out of scope for v1).
* No ring-buffer overrun protection at Tile Z beyond the per-slot
  256-record cap (host pre-allocates worst-case-sized BO; ring
  pressure under full-genome scans is 's surface).

These are explicit follow-on items, listed here so the writeup
can cite them.
