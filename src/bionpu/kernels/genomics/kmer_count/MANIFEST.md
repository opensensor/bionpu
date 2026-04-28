# bionpu_kmer_count — source manifest (kernel-dir level)

Per `state/kmer_count_interface_contract.md` (T1) — symbols, ObjectFifo
names, constants, streaming chunk + overlap protocol, and overflow
policy are pinned there. This file is the kernel-directory-level
manifest; per-artifact MANIFEST.md files (one per
`bionpu_kmer_count_k{k}_n{n_tiles}/` cell, 12 total) are written by
T11 under
`bionpu-public/src/bionpu/dispatch/_npu_artifacts/bionpu_kmer_count_k*/`.

K-mer counting on the AIE2P NPU. Streams packed-2-bit DNA through a
shim → broadcast → N tiles → aggregator → shim topology, with on-tile
canonical (forward + reverse-complement) k-mer counting via
open-addressed linear-probe hash tables and emit-on-evict overflow to
preserve counts byte-equal to Jellyfish.

## Files (kernel directory)

| name                       | role                                                                  |
|----------------------------|-----------------------------------------------------------------------|
| `kmer_count.py`            | IRON Python lowering (T8) — 3 (k) × 4 (n_tiles) variants.             |
| `kmer_count_tile.cc`       | AIE2P C++ per-tile kernel (T5) — 3 `extern "C"` symbols (one per k).  |
| `kmer_count_aggregator.cc` | AIE2P C++ aggregator kernel (T6) — single `kmer_count_aggregator`.    |
| `kmer_count_constants.h`   | Header pinned by T1 contract — masks, EmitRecord layout, geometry.    |
| `runner.cpp`               | Host C++ XRT runner (T7) — 4 K chunked DMA + host-side dedup-merge.   |
| `Makefile`                 | Build rules (T10) — 12 (k, n_tiles) cells.                            |
| `__init__.py`              | Python `NpuOp` registration (T9) — 3 `register_npu_op` calls.         |
| `DESIGN.md`                | Topology, byte layouts, RC handling, overflow policy (T17 fills).    |
| `gaps.yaml`                | Toolchain-gap report (T18 populates).                                 |
| `PASSPORT.json`            | Build provenance (this file's sibling).                              |
| `MANIFEST.md`              | This file.                                                            |

## Pinned shape (per T1 contract)

| field                       | value                                                                |
|-----------------------------|----------------------------------------------------------------------|
| Supported k                 | `{15, 21, 31}` (3 registry entries)                                  |
| Supported n_tiles           | `{1, 2, 4, 8}` (constructor arg on shared op class)                  |
| `HASH_BUCKETS_PER_TILE`     | 4096 (× 12 B/record = 48 KiB tile-DM, ≤64 KiB cap)                   |
| `OVERFLOW_THRESHOLD`        | 8 (linear-probe chain length triggering emit-on-evict)               |
| `EMIT_RECORD_BYTES`         | 16 (`canonical_u64 + count_u32 + flags_u32`; `EVICT_FLAG` = bit 0)   |
| `EMIT_SLOT_RECORDS`         | 1024 (sparse-emit ring slot capacity)                                |
| `SEQ_IN_CHUNK_BYTES_BASE`   | 4096 + per-k overlap (`ceil((k-1)/4)` bytes)                         |
| 2-bit base codes            | A=0x0, C=0x1, G=0x2, T=0x3 (MSB-first within byte)                   |
| Output format               | length-prefixed sparse `EmitRecord` blobs; host dedup-merges by canonical |

## Artifact matrix (3 × 4 = 12, written by T11)

```
bionpu_kmer_count_k15_n{1,2,4,8}/
bionpu_kmer_count_k21_n{1,2,4,8}/
bionpu_kmer_count_k31_n{1,2,4,8}/
```

Each leaf contains `final.xclbin`, `insts.bin`, `host_runner`,
`MANIFEST.md` (per-artifact, T11). All four leaves are gated by
`BionpuKmerCount.artifacts_present()` (T9).
