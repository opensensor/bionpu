# bionpu_pam_filter_iupac — source manifest (kernel-dir level)

Multi-PAM IUPAC scan on AIE2P. Track A v0 — base editor design.

A SINGLE xclbin serves every Cas9 PAM variant via runtime IUPAC mask
in the per-chunk header. No per-PAM rebuild required.

## Files (kernel directory)

| name                              | role                                                          |
|-----------------------------------|---------------------------------------------------------------|
| `pam_filter_iupac.py`             | IRON Python lowering — single entry point.                    |
| `pam_filter_iupac_tile.cc`        | AIE2P C++ per-tile kernel — single `extern "C"` symbol.       |
| `pam_filter_iupac_constants.h`    | Header pinning IUPAC nibble layout, record geometry.          |
| `runner.cpp`                      | Host C++ XRT runner — chunked DMA + dedup-merge + binary out. |
| `Makefile`                        | Build rules — n_tiles ∈ {1, 2, 4, 8}.                          |
| `__init__.py`                     | Python `NpuOp` registration — single `register_npu_op` call.  |
| `DESIGN.md`                       | Topology, byte layouts, scan math, ship boundary.             |
| `gaps.yaml`                       | Toolchain-gap report.                                         |
| `PASSPORT.json`                   | Build provenance.                                             |
| `MANIFEST.md`                     | This file.                                                    |

## Pinned shape

| field                        | value                                                                           |
|------------------------------|---------------------------------------------------------------------------------|
| Supported PAM length         | 1..8 (PFI_PAM_LEN_MAX = 8; covers SaCas9-KKH NNNRRT)                            |
| Supported n_tiles            | `{1, 2, 4, 8}` (constructor arg on shared op class)                             |
| `PFI_PARTIAL_OUT_BYTES_PADDED` | 32768 (32 KiB per tile slot)                                                  |
| `PFI_RECORD_BYTES`           | 16 (uint32 query_pos + uint8 strand + uint8 + 16b + 64b pad)                    |
| `PFI_MAX_EMIT_IDX`           | 2046 (per slot)                                                                 |
| `PFI_HEADER_BYTES`           | 24 (4 mask + 1 length + 11 pad + 4 actual_bytes + 4 owned_offset)               |
| Overlap                      | 8 bytes (covers `PAM_LEN_MAX - 1` bases for all v0 PAMs)                        |
| Streaming chunk              | 4096 payload + 24 header + 8 overlap-headroom = 4128 B                          |
| Output sort                  | `(query_pos asc, strand asc)` — host-side merge across chunks                   |
| Path                         | Runtime IUPAC mask (single xclbin per n_tiles cell)                             |

## Per-cell artifact directories

Built artifacts live under
`bionpu/dispatch/_npu_artifacts/bionpu_pam_filter_iupac_n{n_tiles}/`:

| n_tiles | dir | status |
|---------|-----|--------|
| 4 | `bionpu_pam_filter_iupac_n4/` | scaffold; xclbin pending build (see `state/track-a-be-design-plan.md`) |

## Cross-validation

Byte-equal vs the locked `crispr/pam_filter` kernel on the SAME NGG
fixture is a v0 acceptance criterion: when the IUPAC kernel runs with
PAM=NGG it must emit the same forward-strand records as the locked
NGG-only kernel (positions; the locked kernel emits with guide-aware
records and the IUPAC kernel is PAM-only, so the comparison is
position-set equality).

## Phase relationship

* Phase 1 (this v0): SpCas9 wt + NG; BE4max + ABE7.10. Silicon kernel
  ships; off-target scan deferred.
* Phase 2 (v1+): SpRY, SaCas9-KKH multi-variant ranker; locked
  `crispr/match_multitile_memtile` wired for off-target.
* Phase 3 (v2+): BE-Hive-class neural scorer (CPU per Track D's
  lessons).
