# bionpu_primer_scan — source manifest (kernel-dir level)

Primer / adapter exact-match scan on AIE2P. v0 ships three pinned
primer-length cells: `P=13` (Illumina TruSeq P5 adapter; default
smoke target), `P=20` (typical PCR primer), `P=25` (qPCR primer). See
`DESIGN.md` for the topology, byte layouts, and ship boundary.

## Files (kernel directory)

| name                        | role                                                                  |
|-----------------------------|-----------------------------------------------------------------------|
| `primer_scan.py`            | IRON Python lowering — single-pass per-P variants.                    |
| `primer_scan_tile.cc`       | AIE2P C++ per-tile kernel — 3 `extern "C"` symbols (one per P).       |
| `primer_scan_constants.h`   | Header pinning masks, record layout, geometry, header bytes.          |
| `runner.cpp`                | Host C++ XRT runner — chunked DMA + dedup-merge + binary blob output. |
| `Makefile`                  | Build rules — 3 P × 4 n_tiles cells (12 cells; v0 ships 1).           |
| `__init__.py`               | Python `NpuOp` registration — 3 `register_npu_op` calls.              |
| `DESIGN.md`                 | Topology, byte layouts, scan math, ship boundary.                     |
| `gaps.yaml`                 | Toolchain-gap report (populated post-v0).                             |
| `PASSPORT.json`             | Build provenance.                                                     |
| `MANIFEST.md`               | This file.                                                            |

## Pinned shape

| field                       | value                                                                |
|-----------------------------|----------------------------------------------------------------------|
| Supported P                 | `{13, 20, 25}` (3 registry entries)                                  |
| Supported n_tiles           | `{1, 2, 4, 8}` (constructor arg on shared op class)                  |
| `PS_PARTIAL_OUT_BYTES_PADDED` | 32768 (32 KiB per tile slot)                                       |
| `PS_RECORD_BYTES`           | 16 (uint32 query_pos + uint8 strand + uint8 primer_idx + 16 bits + 64 bits pad) |
| `PS_MAX_EMIT_IDX`           | 2046 (per slot; ample headroom — typical adapter occurs 0-2x/read)   |
| `PS_HEADER_BYTES`           | 24 (8 prefix + 16 primer canonical pair)                             |
| Overlap                     | 8 bytes (covers `P - 1` bases for all v0 P values)                   |
| Streaming chunk             | 4096 payload + 24 header + 8 overlap-headroom = 4128 B               |
| Output sort                 | `(query_pos asc, strand asc)` — host-side merge across chunks        |
| Path                        | B (runtime primer canonical in chunk header)                         |

## Per-cell artifact directories

Built artifacts live under
`bionpu-public/src/bionpu/dispatch/_npu_artifacts/bionpu_primer_scan_p{P}_n{n_tiles}/`:

* `final.xclbin` — single-pass xclbin for this (P, n_tiles) cell.
* `insts.bin` — NPU instructions binary.
* `host_runner` — host-side XRT runner.

v0 silicon-validated cells:

* `bionpu_primer_scan_p13_n4` — TruSeq P5 default.

Additional cells (`p20_n4`, `p25_n4`) are configured but only built
on-demand.

## Build invocations (from this directory)

```bash
source /opt/xilinx/xrt/setup.sh
source /home/$USER/xdna-bringup/ironenv/bin/activate
export MLIR_AIE_DIR=/home/$USER/genetics/third_party/mlir-aie
export PEANO_INSTALL_DIR=/home/$USER/xdna-bringup/ironenv/lib/python3.14/site-packages/llvm-aie

make NPU2=1 P=13 experiment=wide4 seq=10000 all
make NPU2=1 P=20 experiment=wide4 seq=10000 all   # optional
make NPU2=1 P=25 experiment=wide4 seq=10000 all   # optional
```
