# crispr_match_singletile — pre-built T4.3 artifacts (C-M3)

These four files are the complete on-disk surface that
`bionpu.kernels.crispr.match_singletile.CrisprMatchSingleTile` needs to
run the single-tile CRISPR mismatch-count kernel on the AIE2P NPU.
They are the v1-thin "pre-built xclbin + lookup table" approach mandated
by umbrella PRD §4.1 — no on-the-fly compile.

This is the **first non-ML compute-on-ML-accelerator** artifact for the
CRISPR track. Per CRISPR PRD §4.1–4.2 + plan task T4.3 / C-M3: the
question is whether the AIE2P MAC array can do approximate string
matching at competitive throughput. The answer (T5.3 multi-tile is the
real perf race) is the headline number recorded under
`results/crispr/c-m3/measurements.json`.

## Files

| name           | sha256                                                             | role                                                                            |
|----------------|--------------------------------------------------------------------|---------------------------------------------------------------------------------|
| `final.xclbin` | `aef447480c7b5d1094db6c4a7dbf1d4b030816f60cacf80397eabbccd521bfae` | XCLBIN: AIE2P bitstream + control program for the single-tile match kernel.     |
| `insts.bin`    | `3f82a922b45015b02f48a14c7a82418dc02d8bfcff15d1df8ad1d6632d7f560f` | Userspace instruction stream (`uint32` PDI fragment).                           |
| `host_runner`  | `974ddd80f24bee2c2cdf07e9d7e1356c7ad4b122f3980cc206e1471d1c8024b5` | C++ host binary built from `bionpu/kernels/crispr/match_singletile/test.cpp`.   |
| `MANIFEST.md`  | (this file)                                                        | Provenance + rebuild instructions.                                              |

## Pinned shape

The xclbin is **shape-pinned** to T4.3's correctness fixture:

| field           | value                                |
|-----------------|--------------------------------------|
| `N_GUIDES`      | 128                                  |
| `N_WINDOWS`     | 4096                                 |
| `SPACER_LEN`    | 20 nt                                |
| `SPACER_BYTES`  | 5 (2-bit packed: A=00, C=01, G=10, T=11) |
| `WINDOWS_PER_CHUNK` | 64 (DM double-buffered)          |
| Output layout   | window-major `(N_WINDOWS, N_GUIDES)` `uint8`; the Python wrapper transposes to `(N_GUIDES, N_WINDOWS)` so the registered-op contract matches the oracle's natural shape. |

T5.3 will lift the shape pin (true streaming + multi-tile fan-out).

## Tile-memory budget

AIE2P: ~64 KiB DM per tile, ~16 KiB program memory per tile.

This kernel uses, per tile:

| region                      | bytes  |
|-----------------------------|--------|
| guides (resident)           |   640  |
| windows chunk × 2 (double)  |   640  |
| out chunk × 2 (double)      | 16384  |
| **total tile DM used**      | **~17.7 KiB**, well under the 64 KiB cap |

Program memory (the compiled `match_kernel.o` + AIE control) is
small — the kernel is a scalar nested loop of integer popcount; no
template-heavy aie::vector intrinsics. The 16 KiB program memory budget
is comfortably unused.

## Build provenance

- **Source**:
  - `bionpu/kernels/crispr/match_singletile/match_kernel.cc`
  - `bionpu/kernels/crispr/match_singletile/match_singletile.py`
  - `bionpu/kernels/crispr/match_singletile/test.cpp`
  - `bionpu/kernels/crispr/match_singletile/Makefile`
  - `bionpu/kernels/crispr/match_singletile/CMakeLists.txt`
- **NPU2 flag**: `1` (REQUIRED — AIE2P. Without `NPU2=1` the kernel
  silently compiles for AIE2 and produces garbage. See
  `tools/canaries/iron-vector-add.sh` and the T0.1 recheck script.)
- **Build date (UTC)**: `2026-04-25T07:37Z`
- **Host**: `matteius-ProArt-P16-H7606WI`
- **Bring-up env**:
  - XRT 2.23.0 at `/opt/xilinx/xrt/`
  - amdxdna 2.23.0
  - NPU firmware 1.1.2.64
  - Architecture `aie2p` (6×8 = 48 tiles), BDF `0000:67:00.1`
  - `~/xdna-bringup/ironenv` with Python 3.11.15, mlir-aie, llvm-aie

## How to rebuild

```bash
source /opt/xilinx/xrt/setup.sh
source ~/xdna-bringup/ironenv/bin/activate
export MLIR_AIE_DIR=~/xdna-bringup/mlir-aie
export PEANO_INSTALL_DIR=~/xdna-bringup/ironenv/lib/python3.11/site-packages/llvm-aie

cd <repo>/bionpu/kernels/crispr/match_singletile
make NPU2=1 clean
make NPU2=1 all

cp build/final.xclbin              <repo>/bionpu/dispatch/_npu_artifacts/crispr_match_singletile/final.xclbin
cp build/insts.bin                  <repo>/bionpu/dispatch/_npu_artifacts/crispr_match_singletile/insts.bin
cp crispr_match_singletile          <repo>/bionpu/dispatch/_npu_artifacts/crispr_match_singletile/host_runner
```

Then update the sha256 rows above. The C++ binary is the path used
in v1 because pyxrt ships only as a Python 3.14 binding on this host
and ironenv is Python 3.11 (T4.1 documents the same constraint).

## What the binary does

The host binary:

1. Reads a `--guides PATH` blob (640 bytes, packed `uint8`) into the
   tile-side guides ObjectFifo.
2. Reads a `--windows PATH` blob (20480 bytes, packed `uint8`) into the
   windows ObjectFifo. The IRON program walks 64 chunks of 64 windows,
   feeding the kernel one chunk at a time.
3. The AIE C++ kernel `crispr_match_singletile` (in `match_kernel.cc`)
   computes mismatch counts via XOR + 2-bit-pair popcount + 5-byte sum,
   producing a `(N_WINDOWS, N_GUIDES)` `uint8` matrix back through
   `of_out`.
4. The host writes the output to `--out PATH` and prints
   `Avg / Min / Max NPU time: ... us.` lines on stdout. The Python
   wrapper parses these and transposes window-major → guide-major.

A `PASS!` marker is printed on a clean run; the wrapper hard-fails if
that marker is absent. There is **no** in-binary verify (unlike T4.1's
`vector_scalar_mul`) — the byte-equality verify lives in the Python
runner (`tracks/crispr/npu/match_singletile.py`) which compares against
the NumPy oracle (T2.7) on the same fixture.

## Recorded T4.3 baseline (C-M3)

Per `results/crispr/c-m3/measurements.json`:

| metric                            | value                          |
|-----------------------------------|--------------------------------|
| byte-equality (NPU vs oracle)     | **PASS** (`np.array_equal == True`) |
| byte-equality (canonical TSV)     | **PASS** (`bytes == bytes`)    |
| windows / sec                     | ~33.6 K (one launch overhead-dominated; throughput-limited by host I/O round-trip in v1) |
| guide × window / sec              | ~4.3 M                         |
| `tile_memory_used_bytes`          | 17 664                         |
| `energy_source`                   | `real-xrt-smi` (T4.4 wired)    |

These are the C-M3 numbers to beat in C-M4 (T5.3, multi-tile streaming).
