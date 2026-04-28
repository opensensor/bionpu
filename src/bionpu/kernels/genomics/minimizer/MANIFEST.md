# bionpu_minimizer — source manifest (kernel-dir level)

Sliding-window (w, k) minimizer extraction on AIE2P. v0 ships two
silicon-validated `(k, w)` cells: `(15, 10)` and `(21, 11)`. See
`DESIGN.md` for the topology, byte layouts, and ship boundary.

## Files (kernel directory)

| name                        | role                                                                  |
|-----------------------------|-----------------------------------------------------------------------|
| `minimizer.py`              | IRON Python lowering — single-pass per-(k, w) variants.               |
| `minimizer_tile.cc`         | AIE2P C++ per-tile kernel — 2 `extern "C"` symbols (one per (k, w)). |
| `minimizer_constants.h`     | Header pinning masks, record layout, geometry, header bytes.          |
| `runner.cpp`                | Host C++ XRT runner — chunked DMA + dedup-merge + binary blob output. |
| `Makefile`                  | Build rules — 2 (k, w) × 4 n_tiles cells (8 cells; v0 ships 2).       |
| `__init__.py`               | Python `NpuOp` registration — 2 `register_npu_op` calls.              |
| `DESIGN.md`                 | Topology, byte layouts, sliding-window math, ship boundary.           |
| `gaps.yaml`                 | Toolchain-gap report (populated post-v0).                             |
| `PASSPORT.json`             | Build provenance.                                                     |
| `MANIFEST.md`               | This file.                                                            |

## Pinned shape

| field                       | value                                                                |
|-----------------------------|----------------------------------------------------------------------|
| Supported (k, w)            | `{(15, 10), (21, 11)}` (2 registry entries)                          |
| Supported n_tiles           | `{1, 2, 4, 8}` (constructor arg on shared op class)                  |
| `MZ_PARTIAL_OUT_BYTES_PADDED` | 32768 (32 KiB per tile slot)                                       |
| `MZ_RECORD_BYTES`           | 16 (uint64 canonical + uint32 position + uint32 pad)                 |
| `MZ_MAX_EMIT_IDX`           | 2046 (per slot; ~10× headroom over expected emit density 2/(w+1))    |
| `MZ_HEADER_BYTES`           | 8 (uint32 actual_bytes + int32 owned_start_offset_bases)             |
| Per-(k, w) overlap          | 8 bytes (covers `w + k - 1` bases for both pinned configs)           |
| Streaming chunk             | 4096 payload + 8 header + 8 overlap = 4112 B (4-byte aligned)        |
| Output sort                 | `(position asc, canonical asc)` — kernel-local within-chunk; host    |
|                             | merges across chunks with the same key.                              |

## Per-cell artifact directories

Built artifacts live under
`bionpu-public/src/bionpu/dispatch/_npu_artifacts/bionpu_minimizer_k{k}_w{w}_n{n_tiles}/`:

* `final.xclbin` — single-pass xclbin for this (k, w, n_tiles) cell.
* `insts.bin` — NPU instructions binary.
* `host_runner` — host-side XRT runner.

v0 silicon-validated cells:

* `bionpu_minimizer_k15_w10_n4` — short-read default
* `bionpu_minimizer_k21_w11_n4` — long-read default

(SHA-256 sums of the artifact files appear in each cell's per-artifact
`MANIFEST.md` — written at install time by the build script.)

## Build invocations (from this directory)

```bash
source /opt/xilinx/xrt/setup.sh
source /home/$USER/xdna-bringup/ironenv/bin/activate
export MLIR_AIE_DIR=/home/$USER/genetics/third_party/mlir-aie
export PEANO_INSTALL_DIR=/home/$USER/xdna-bringup/ironenv/lib/python3.14/site-packages/llvm-aie

make NPU2=1 K=15 W=10 experiment=wide4 seq=10000 all
make NPU2=1 K=21 W=11 experiment=wide4 seq=10000 all
```
