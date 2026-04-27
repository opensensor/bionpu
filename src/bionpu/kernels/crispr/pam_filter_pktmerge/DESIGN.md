# — PacketFifo PAM-filter retrofit (C-M5-pktmerge)

This is the kernel-author DESIGN doc for the retrofit of 's
filter-early variant using the fork's PacketFifo. Read this
together with `results/crispr/c-m5-pktmerge/verdict.md` for the
ratification outcome.

## Mission

Close the OTHER HALF of :

* **Phase 1** documented the wall: IRON's ObjectFifo had no
  skip-element semantics, so filter-early always forwarded full
  chunks. The match tiles paid the cycle cost on every window,
  including the ~93.6% (strict NGG on random ACGT) that fail PAM.

* **Phase 2 — API surface** landed the fork's PacketFifo
  primitive: variable-rate pktMerge / finish-on-TLAST / out-of-order
  BD. AM020 Ch. 2 Figure 17 (pktMerge N:1) + Ch. 2 p. 27 (S2MM
  finish-on-TLAST) + Ch. 5 p. 74 (out-of-order BD). The IRON-level
  class accepts the canonical filter-early construction.

* **Phase 2 — silicon** ratifies the silicon-level
  pktMerge N:1 shape in two forms. The `pktmerge_topology()` helper
  constructs the canonical PacketFifo at API-surface level, and the
  opt-in `DIRECT_STREAM=1` path replaces the `windows_out` payload
  edge with compact core-to-core streams. The default dispatch artifact
  remains the byte-equal filter-early twin until the packetized
  stream path is promoted through silicon performance runs.

## Topology (target)

The canonical replacement topology, declared via
`pktmerge_topology()`:

```
shim ──windows_in (ObjectFifo)── Tile A
                                   │ PacketFifo / core stream
                                   │   valid windows only:
                                   │   u32 window_idx + 5 spacer bytes
                                   │   + 3 pad bytes
                                   ▼
                          match tiles (variable-rate consumers;
                                       only see PAM-passing windows)
                                   │ partials ObjectFifo
                                   ▼
                          Tile Z (threshold + sparse-emit) ──→ shim
```

Per-tile placement (matches ):
* Tile A: `(0, 2)`
* Match tile 0: `(0, 3)`
* Match tile 1: `(0, 4)`
* Tile Z: `(0, 5)`

## Topology (live)

Per the live IRON Program keeps the filter-early
ObjectFifo wiring (Tile A → match tiles via a fixed-rate `windows_out`
ObjectFifo). Same xclbin arithmetic; same chunk geometry; same byte-
equal output. The Kernel-symbol bindings are renamed
(`crispr_pam_filter_tile_a_pktmerge` /
`crispr_pam_filter_tile_z_pktmerge`) so the `crispr_pam_filter_pktmerge`
op surfaces in NPU_OPS as a separate measurable artifact.

## Topology (direct stream prototype)

The buildable packetized-stream replacement is opt-in:

```bash
make NPU2=1 DIRECT_STREAM=1 build/final.xclbin
```

This path bypasses ADF C++ stream pointers entirely. The Peano install
used here does not ship `<adf.h>`, and the direct stream ABI therefore
uses lower-level AIE2P Peano intrinsics from `<aie2p/aie2p_streams.h>`:

* Tile A calls `crispr_pam_filter_tile_a_pktmerge_stream_i32`.
  It scans a 64-window chunk, emits only NGG-passing windows, and writes
  three 32-bit stream words per valid packet:
  `uint32 window_idx`, packed spacer bytes 0..3, packed spacer byte 4.
  Empty chunks emit one sentinel packet (`window_idx == UINT32_MAX`) so
  both stream consumers execute the same per-chunk receive protocol; Tile
  Z skips the sentinel and emits no sparse records for that chunk.
* Each match tile calls
  `crispr_match_multitile_match_packetized_stream_i32`. It reads the
  per-chunk valid count from a sideband ObjectFifo, receives exactly that
  many three-word packets with `get_ss_int()`, computes 64 guide
  mismatches, and writes compact partial records keyed by original
  `window_idx`.
* IRON emits only compact `scf.for` chunk loops plus two explicit
  `aie.flow(%logical_core, Core : 0, %logical_core_N, Core : 0)` routes.
  The generated direct-stream MLIR is 166 lines and contains no
  `aie.put_stream` or `aie.get_stream` operations.

The abandoned pure-MLIR stream-op route is documented here because it
looked attractive but did not scale:

* `aie.put_stream` / `aie.get_stream` on `i32` verified only when the ops
  were emitted directly inside `aie.core`, which forced per-packet
  unrolling. The 4096-window program generated about 153k lines of MLIR
  and tens of thousands of stream ops.
* `i64` stream words are rejected by the AIE dialect verifier.
* `i128` stream words verify and route, but AIE2P lowering calls
  `llvm.aie2p.put.wms` / `llvm.aie2p.get.wss`, which this backend does
  not provide. The backend has the 32-bit `put.ms` / `get.ss` path, so
  the C++ intrinsic route is the buildable compact form.

## Math

**Verbatim from / / ** — XOR + 2-bit-pair popcount +
5-byte sum spacer mismatch arithmetic; NGG PAM check (bits 3:2 == G
&& bits 5:4 == G); threshold + sparse-emit at Tile Z.

The match-tile entry (`crispr_match_multitile_match`) is unchanged
from . Tile A's filter and Tile Z's emit kernels are renamed to
disambiguate from 's symbols, but their bodies match 's
filter-early variants byte-for-byte.

## Per-tile memory

| tile             | bytes | budget hit                                    |
|------------------|------:|-----------------------------------------------|
| `tile_a_pktmerge`| 1664  | windows_in DBL + windows_out DBL + pam_meta DBL + 128 B header DBL |
| `match_tile`     | 9472  | guides resident + windows DBL + partial DBL  |
| `tile_z_emit`    | 20608 | partials DBL + pam_meta DBL + sparse ring DBL |
| **peak**         | 20608 | (Tile Z; ~32% of AIE2P 64 KiB DM cap)         |

Default Δ vs filter-early: +128 B at Tile A for the per-chunk
packet header buffer; otherwise identical. The `DIRECT_STREAM=1`
prototype removes the fixed-rate `windows_out` payload buffering from
the match-tile path and replaces it with stream traffic plus sideband
valid counts and compact partial packets. Both forms remain under cap.

## Wire format (unchanged from )

* **Window record** (Tile A input): 5 spacer bytes + 1 PAM byte = 6 B
  per window. PAM byte layout: bits 1:0 = pam[0] (N, ignored),
  3:2 = pam[1] (must be G), 5:4 = pam[2] (must be G), 7:6 = padding.
* **Packetized valid-window payload** (`DIRECT_STREAM=1`): three 32-bit
  words per valid window: `uint32 window_idx`, spacer bytes 0..3 packed
  little-endian, spacer byte 4 in the low byte of the third word.
* **Sparse emit record** (Tile Z output): 8 B per surviving (window,
  guide) pair: u16 window_idx (LE) | u8 guide_idx | u8 mismatches |
  4 B reserved.
* **Sparse buffer**: u32 length prefix + N×8 records.

## Symbols emitted (`tile_a_pktmerge.cc`)

* `crispr_pam_filter_tile_a_pktmerge` — Tile A. Same body as 's
  `_tile_a_early` symbol (NGG check; copy spacer if pass; zero-fill
  if fail; pam_meta carries the 1-bit valid header per window).
* `crispr_match_multitile_match` — match tile. Verbatim from .
* `crispr_pam_filter_tile_z_pktmerge` — Tile Z. Same body as 's
  `_tile_z_early` symbol (threshold + sparse-emit; per-window
  pam_meta gate skips PAM-failing windows).
* `crispr_pam_filter_tile_a_pktmerge_stream_i32` — direct-stream Tile A
  producer. Uses `put_ms` to emit three i32 words for each PAM-passing
  window. No ADF stream pointer ABI.
* `crispr_match_multitile_match_packetized_stream_i32` — direct-stream
  match consumer. Uses `get_ss_int()` to receive packet words and emits
  compact partial mismatch records.
* `crispr_pam_filter_tile_z_pktmerge_packetized` — packetized Tile Z
  reducer for compact partial records.

## Silicon-level cycle-saving path

The earlier "sidecar PacketFifo" attempt was a dead end: it emitted
packetflow MLIR beside the existing `windows_out` ObjectFifo, but the
payload still moved through ObjectFifo and the sidecar consumed the same
limited switch/DMA resources. The correct lift is replacement, not
coexistence:

1. Keep 's fork fix: `PacketFifo.resolve` can emit packet
   flows from live Worker endpoint tiles, and the minimal fork-level
   PacketFifo tests cover that.

2. Use the `DIRECT_STREAM=1` path as the current replacement prototype.
   It replaces `of_windows_out` with explicit core stream routes and
   compact packet kernels while leaving the default byte-equal artifact
   unchanged.

3. Use the corrected packet ABI: PacketFifo packet IDs are assigned per
   producer by the fork's lowering, not per packet. Tile A must emit only
   PAM-passing windows; invalid windows are skipped, not emitted with an
   invalid/drop header. Each valid packet carries a 12-byte payload:
   `uint32 window_idx`, 5 spacer bytes, and 3 pad bytes.

4. Keep the stream transfer loops in Peano C++ for now. Pure MLIR
   `aie.put_stream` / `aie.get_stream` lowering either explodes the IR
   (`i32`), fails verification (`i64`), or reaches missing AIE2P wide
   intrinsics (`i128`).

5. The match tiles then consume only PAM-passing windows; the
   match-tile window-work count drops by ~15/16 (strict NGG on random
   ACGT). Re-run `test_throughput_delta_vs_t62_filter_early_in_envelope`
   and update `results/crispr/c-m5-pktmerge/measurements.json`.

The current C++ kernel + IRON-Python topology + host harness remain
useful for byte-equality and for shaping the packet ABI. The speedup
work is now explicitly a data-path replacement task, with a buildable
direct-stream xclbin available for silicon timing and promotion work.

## Direct-Stream Silicon Status

The direct-stream xclbin is vendored separately under:

```text
bionpu/dispatch/_npu_artifacts/crispr_pam_filter_pktmerge_direct_stream/
```

and registered as `crispr_pam_filter_pktmerge_direct_stream`, leaving the
default `crispr_pam_filter_pktmerge` artifact untouched as the rollback
ObjectFifo path.

Silicon canaries run on 2026-04-26:

* random 4096-window fixture, 262 PAM-pass windows, zero host hits:
  byte-equal to host, ~0.83 ms kernel time after persistent-worker fix;
* one valid window total, all later chunks empty: byte-equal to host
  after the sentinel fix, 128 hits, ~1.3 ms kernel time;
* one valid window per chunk: byte-equal to host, 8192 hits, ~40.9 ms
  before the persistent-worker fix and ~0.52 ms after it.

The direct-stream path is therefore correctness-positive on targeted
canaries, including the zero-valid-chunk case that previously produced a
TDR. was closed by making direct-stream workers persistent
instead of `while_true=False`, so repeated `n_iters=3 warmup=1` timing now
works.

Serial comparisons against filter-early:

* random empty-hit fixture: direct ~0.83 ms, filter-early ~0.75 ms,
  byte-equal, ratio early/direct ~0.91x;
* one-valid-window-per-chunk non-empty fixture: direct ~0.52 ms,
  filter-early ~0.75 ms, byte-equal, ratio early/direct ~1.45x.

This is a positive sparse-fixture timing signal, but it remains below the
original [4x, 8x] envelope and needs broader fixture coverage before
default-artifact promotion.

Valid-density matrix (`n_iters=3`, `warmup=1`, no-hit fixtures):

| valid windows / chunk | direct-stream | filter-early | early/direct | verdict |
|----------------------:|--------------:|-------------:|-------------:|---------|
| 0                     |      ~292 us  |      ~687 us |        2.35x | pass    |
| 1                     |      ~326 us  |      ~670 us |        2.06x | pass    |
| 4                     |      ~769 us  |      ~690 us |        0.90x | pass but slower |
| 16                    |           TDR |            - |            - | fail    |

The knee is therefore between 1 and 4 valid windows per 64-window chunk.
The current Peano AIE2P stream helper exposes only the single
`put_ms`/`get_ss_int` stream, so replacing the Core:0 fanout with two
explicit independent streams is blocked at the C++ ABI layer. See
.

## Compact-Packet ObjectFifo Status

The alternate compact-packet xclbin is vendored separately under:

```text
bionpu/dispatch/_npu_artifacts/crispr_pam_filter_pktmerge_compact_packets/
```

and registered as `crispr_pam_filter_pktmerge_compact_packets`. It stores
the per-chunk valid count in logical word 0 of a compact `memref<65xi64>`
ObjectFifo. Tile A produces one counted packet buffer, both match tiles
consume that buffer, and Tile Z receives the valid-count sideband plus
compact partial records. The local window index is packed into the high
byte of each packet's second spacer word and carried as one byte in each
partial record. This avoids the single Core:0 direct-stream fanout that
TDRs under dense valid-window traffic without adding another Tile A output
FIFO.

Build/routing status:

* `COMPACT_PACKETS=1` generated 154 lines of MLIR.
* `cp_packet_words` lowers as `memref<65xi64>`.
* compact partial records lower as `memref<4160xui8>` (65 bytes x 64).
* `aie-opt --verify-diagnostics` and pathfinder routing pass.
* `make NPU2=1 COMPACT_PACKETS=1 build/final.xclbin build/insts.bin
  crispr_pam_filter_pktmerge` passes.

Silicon canaries run on 2026-04-26:

* one valid matching window: 128 NPU hits == 128 host hits;
* no-hit density matrix below: every row byte-equal to host, including
  the 16-valid/chunk row that TDRs direct-stream.

Valid-density matrix (`n_iters=10`, `warmup=2`, no-hit fixtures):

| valid windows / chunk | compact packets | filter-early | early/compact | verdict |
|----------------------:|----------------:|-------------:|--------------:|---------|
| 0                     |         171 us  |      724 us  |         4.25x | pass    |
| 1                     |         240 us  |      680 us  |         2.84x | pass    |
| 4                     |         704 us  |      683 us  |         0.97x | pass but slightly slower |
| 16                    |        2553 us  |     1857 us  |         0.73x | pass but slower |

This is better than direct-stream for robustness: the dense-valid TDR is
gone. Shrinking the packet ABI improved the sparse end enough to hit the
lower edge of the original 4x envelope for empty chunks, but it did not
move the dense knee. It is not a default-promotion result yet because the
performance knee remains workload-dependent. Sparse strict-NGG chunks see
the intended speedup, while dense PAM-pass chunks pay for compact packet
materialization plus multicast ObjectFifo traffic.

## Provenance

This file is licensed under the Apache License v2.0 with LLVM Exceptions.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

(c) Copyright 2026 Matt Davis. Math kernels derived verbatim from
`bionpu/kernels/crispr/pam_filter/tile_a_filter.cc`. Topology
adapted via the fork's PacketFifo primitive ( — fork commit
`236059d6`).
