# crispr_pam_filter — source manifest (C-M5)

PAM filter + threshold + sparse-emit on the AIE2P NPU. Builds on 's
multi-tile dataflow by adding (a) on-tile NGG PAM check at Tile A and
(b) on-tile threshold + sparse-emit at Tile Z. Two xclbin variants
ship from this same source (filter-early + filter-late) so the
C-M5 writeup can compare work distribution head-to-head.

## Files

| name              | role                                                                                       |
|-------------------|--------------------------------------------------------------------------------------------|
| `pam_filter.py`   | IRON lowering (Tile A + match tiles + Tile Z; selects filter-early vs filter-late). |
| `tile_a_filter.cc`| AIE2P C++ kernel — Tile A (early/late variants), match (verbatim ), Tile Z (early/late variants). |
| `runner.cpp` | Host C++ XRT runner — same file-backed I/O as with a sparse-emit output blob. |
| `Makefile`        | Builds both xclbin variants + the host runner.                                             |
| `CMakeLists.txt`  | Optional CMake target for IDE integration.                                                 |
| `__init__.py`     | Python `NpuOp` registration: `crispr_pam_filter_early` + `crispr_pam_filter_late`.         |
| `DESIGN.md`       | Topology, byte layouts, strand handling, gaps surfaced.                                    |
| `gaps.yaml` | toolchain-gap report for . |
| `MANIFEST.md`     | This file.                                                                                  |

## Pinned shape

| field                | value                                            |
|----------------------|--------------------------------------------------|
| `N_GUIDES`           | 128                                              |
| `N_WINDOWS`          | 4096                                             |
| `SPACER_LEN`         | 20 nt                                            |
| `SPACER_BYTES`       | 5 (2-bit packed: A=00, C=01, G=10, T=11)         |
| `PAM_LEN`            | 3 nt (NGG)                                       |
| `PAM_BYTES`          | 1 (3 nt × 2 bits / 8, padded)                    |
| `WINDOW_BYTES_IN`    | 6 (5 spacer + 1 PAM)                             |
| `WINDOWS_PER_CHUNK`  | 64                                               |
| `N_MATCH_TILES`      | 2 |
| `GUIDES_PER_TILE`    | 64                                               |
| `EMIT_RECORD_BYTES`  | 8                                                |
| Output layout        | length-prefixed sparse records (host concatenates 64 chunks). |

## Tile topology

```
shim ──guides── (broadcast) ──→ match_0, match_1
shim ──windows_in (6 B/record)── Tile A (PAM filter)
                                    ├──windows_out (5 B/record)── (broadcast) ──→ match_0, match_1
                                    └──pam_meta (1 B/window) ──→ Tile Z
match_0 ──partial_0── Tile Z
match_1 ──partial_1── Tile Z
Tile Z (threshold + sparse-emit) ──sparse_out (length-prefixed records)── shim
```

## Tile-memory budget (DESIGN.md §5)

AIE2P: ~64 KiB DM per tile, ~16 KiB program memory per tile.

| tile          | resident                | dbl-buf in              | dbl-buf out          | total bytes | % of 64 KiB |
|---------------|-------------------------|-------------------------|----------------------|-------------|-------------|
| Tile A filter | —                       | 768 (windows_in × 2)    | 640 + 128 (windows_out + pam_meta × 2) | **1536**    | **2.3%**    |
| match_0/1     | 640 (full guides)       | 640 (windows × 2)       | 8192 (partial × 2)   | **9472**    | **14.4%**   |
| Tile Z emit   | —                       | 16384 + 128 (2 partials × 2 + pam × 2) | 4096 (sparse × 2) | **20608**   | **31.6%**   |

Peak (Tile Z) is 20.6 KiB — *down* from 's 32 KiB joiner peak
because we replaced the dense window-major output buffer with the
smaller sparse-emit ring slot.

## Build provenance

- **Sources**: `pam_filter.py` (IRON), `tile_a_filter.cc` (kernels),
  `runner.cpp` (host), `Makefile`, `CMakeLists.txt`.
- **NPU2 flag**: `1` (REQUIRED — AIE2P).
- **Bring-up env**:
  - XRT 2.23.0 at `/opt/xilinx/xrt/`
  - amdxdna 2.23.0
  - NPU firmware 1.1.2.64
  - Architecture `aie2p` (6×8 = 48 tiles)
  - `~/xdna-bringup/ironenv` with Python 3.11.15, mlir-aie, llvm-aie

## How to rebuild

```bash
source /opt/xilinx/xrt/setup.sh
source ~/xdna-bringup/ironenv/bin/activate
export MLIR_AIE_DIR=~/xdna-bringup/mlir-aie
export PEANO_INSTALL_DIR=~/xdna-bringup/ironenv/lib/python3.11/site-packages/llvm-aie

cd <repo>/bionpu/kernels/crispr/pam_filter
make NPU2=1 clean
make NPU2=1 all

cp build/early/final.xclbin <repo>/bionpu/dispatch/_npu_artifacts/crispr_pam_filter_early/final.xclbin
cp build/early/insts.bin    <repo>/bionpu/dispatch/_npu_artifacts/crispr_pam_filter_early/insts.bin
cp crispr_pam_filter        <repo>/bionpu/dispatch/_npu_artifacts/crispr_pam_filter_early/host_runner

cp build/late/final.xclbin  <repo>/bionpu/dispatch/_npu_artifacts/crispr_pam_filter_late/final.xclbin
cp build/late/insts.bin     <repo>/bionpu/dispatch/_npu_artifacts/crispr_pam_filter_late/insts.bin
cp crispr_pam_filter        <repo>/bionpu/dispatch/_npu_artifacts/crispr_pam_filter_late/host_runner
```

Then update the per-variant MANIFEST.md sha256 rows.

## What the host binary does

The single host runner binary takes either xclbin via `--xclbin`. For
each launch it:

1. Reads `--guides PATH` (640 bytes, 128 guides × 5 packed bytes).
2. Reads `--windows PATH` (24576 bytes, 4096 windows × 6 bytes:
   5 spacer + 1 PAM).
3. Pushes both onto the shim DMA-in. Tile A reads the windows stream,
   filters (early) or passes (late), and forwards to the match tiles
   + Tile Z.
4. Match tiles compute mismatch counts (verbatim arithmetic).
   Tile Z applies threshold (and PAM check in the late variant) and
   emits surviving records via shim DMA-out.
5. Host writes the sparse blob to `--out PATH`. The on-disk layout
   is **per-slot length-prefixed**: the runtime sequence drains
   `N_CHUNKS = 64` ring slots back-to-back into the host buffer
   (each slot = 2048 B = 4-byte uint32 count + up to 256 records ×
   8 B + zero pad). The Python wrapper
   (`decode_per_slot_sparse_buffer`) iterates the slots and applies
   the per-slot `WINDOWS_PER_CHUNK` offset to the kernel-side window
   index (which is hardcoded to 0..63 inside each slot, since
   `chunk_base_window_idx` is passed as 0 from the IRON lowering).

   ** (2026-04-25)**: the previous monolithic decoder only
   saw slot-0's records (~64 vs the kernel's actual ~64×N output).
   Fixed by introducing the per-slot decoder; the precompiled xclbin
   needs no rebuild.

A `PASS!` marker is printed on a clean run; the wrapper hard-fails if
that marker is absent. Byte-equality verification lives in
`tests/test_t62_pam_filter_npu.py` and `tracks/crispr/cli/scan.py`.
