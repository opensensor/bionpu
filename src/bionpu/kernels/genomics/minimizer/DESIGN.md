# bionpu_minimizer — sliding-window (w, k) minimizers on AIE2P (DESIGN, v0)

Per `bionpu/data/minimizer_oracle.py` and `minimizer_constants.h` —
symbols, ObjectFifo names, constants, the streaming chunk + overlap
protocol, and the wire-format records are pinned here. v0 ships a
single-pass single-tile-output kernel covering both pinned
`(k, w) ∈ {(15, 10), (21, 11)}` configurations.

---

## §1 — Strategic context

Minimizers are the second silicon-validated genomics workload (kmer_count
v1.3 was the first). The strategic claim — AIE2P silicon's sweet-spot
is **dense per-byte arithmetic** (not large stateful tables) — is
re-tested here on a **stateless** sliding-window operator that emits
sparsely. A minimizer kernel that lands silicon byte-equal vs a CPU
oracle confirms the streaming-canonical math + small ring-buffer state
+ sparse-emit topology generalises beyond k-mer counting.

---

## §2 — Sliding-window math (canonical specialisation)

For each length-k substring of the input string we compute its
**canonical** 2-bit uint64 representation:

    canonical = min(forward, reverse_complement)

Within each window of `w` consecutive k-mers, we emit the smallest
canonical (oldest-on-tie). This is the canonical specialisation of
minimap2's `mm_sketch` (Li, *Bioinformatics* 2016, `sketch.c:77`+) but
**without** the secondary `hash64` ordering that minimap2 applies. We
intentionally simplify so the silicon-byte-equal contract is tight and
implementable in scalar AIE2P code with a tiny ring buffer.

Divergences from minimap2 `mm_sketch` (intentional, v0):

* No `hash64` post-canonicalisation. Pure-canonical lexicographic min.
* No HPC (homopolymer compression).
* No skipping of `forward == reverse_complement` symmetric k-mers.
* Ties broken by **oldest position wins** — silicon-mirrorable in a
  ring-buffer scan; minimap2's tie-breaking depends on hash collisions.

---

## §3 — Wire format

* **Input**: packed-2-bit DNA, MSB-first per byte (A=00, C=01, G=10, T=11).
  First base = `bits[7:6]` of byte 0.

* **Per-chunk in-band header (8 B)**:
  ```
  [0..3] uint32 LE actual_payload_bytes
  [4..7] int32  LE owned_start_offset_bases
  ```
  Payload starts at byte 8. Total chunk bytes = `4096 + 8 (overlap)`
  for both pinned `(k, w)`.

* **Per-tile output slot (32 KiB)**:
  ```
  [0..3]:                 uint32 LE emit_count
  [4 .. 4+16*emit_count]: emit_count × {
                             uint64 LE canonical,
                             uint32 LE position,
                             uint32 _pad
                          }
  ```
  Position is chunk-local (in bases). Host translates to global by
  adding `chunk.src_offset * 4`.

* **Final output (`--output-format binary`)**:
  ```
  [uint64 n_records LE][n_records × 16-byte record]
  ```
  Records sorted by `(position asc, canonical asc)` and de-duplicated.

---

## §4 — Tile DM budget

Per-tile state (compute tile, `(k=21, w=11)` worst case):

| structure                  | size            |
|----------------------------|-----------------|
| `ring_canonical[w]`        | 11 × 8 = 88 B   |
| `ring_position[w]`         | 11 × 4 = 44 B   |
| `ring_valid[w]`            | 11 × 1 = 11 B   |
| `seq_in` chunk (input BO)  | 4 KiB + 8 B     |
| `partial_out` (output BO)  | 32 KiB          |
| stack frame + scalars      | <2 KiB          |

Total ~38 KiB — well under the 64 KiB CoreTile DM cap.

---

## §5 — Per-(k, w) configurations

| (k, w)    | mask              | bases/window | byte chunks |
|-----------|-------------------|--------------|-------------|
| (15, 10)  | `(1<<30) - 1`     | 24           | 4096 + 8    |
| (21, 11)  | `(1<<42) - 1`     | 31           | 4096 + 8    |

Both round to 8-byte overlap (4-byte-aligned for aiecc).

---

## §6 — Topology (v0 broadcast → tile_0 authoritative)

```
shim ─seq_in─▶ broadcast ──▶ tile_0 .. tile_{N-1}
                                   │
                       (compute canonical;
                        ring-buffer min;
                        emit on min change/slide)
                                   │
                       partial_minimizer_<i>
                                   │
                  memtile.join(N, …) (16 B records, 32 KiB/slot)
                                   │
                                   ▼
                             shim drain
```

v0 broadcasts the same chunk to every tile so every tile produces the
SAME minimizers; the host runner reads tile_0's slot only and ignores
the others. v1 will partition work across tiles (e.g. by chunk
round-robin) for real per-tile parallelism.

---

## §7 — Host runner streaming protocol

1. Read the packed-2-bit input.
2. Plan chunks: each chunk covers 4096 payload bytes plus a
   `w + k - 1` base overlap with its predecessor (rounded to the
   nearest 4-byte boundary, both pinned configs hit 8 B).
3. For each chunk, write the in-band header, copy payload, dispatch
   the kernel (opcode 3), parse tile_0's slot, translate positions to
   global, append to the merged record list.
4. After all chunks: sort by `(position asc, canonical asc)`,
   de-duplicate adjacent records (defensive — the owned-range gate
   should already prevent overlap-region duplicates), apply `--top` if
   requested, emit the binary blob.

The owned-range gate (kernel-side): a record is OWNED by the chunk iff
the minimizer-k-mer's start position is `>= owned_start_offset_bases`
(chunk 0 = 0; chunk i>0 = `overlap_bases - (k+w-2)`). Without this gate,
adjacent chunks would emit duplicate minimizers in the overlap region.

---

## §8 — v0 ship boundary

v0 ships:

* (k=15, w=10) silicon byte-equal vs CPU oracle on `smoke_10kbp` and
  `chr22` slice.
* (k=21, w=11) silicon byte-equal vs CPU oracle on `smoke_10kbp`.
* TDD oracle test suite (4 tests).

v0 explicitly does NOT ship:

* per-tile work partitioning (broadcast to all N tiles; tile_0 slot
  authoritative).
* minimap2 `hash64` ordering parity.
* HPC.
* pyxrt in-process dispatch (subprocess only).

These are tracked in `gaps.yaml` once v0 lands.
