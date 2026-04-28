# crispr_match_multitile — pre-built T5.3 artifacts (C-M4)

These three files are the complete on-disk surface that
`bionpu.kernels.crispr.match_multitile.CrisprMatchMultiTile` needs to run
the multi-tile streaming CRISPR mismatch-count kernel on the AIE2P NPU.
Same v1-thin "pre-built xclbin + lookup table" approach umbrella PRD §4.1
mandates and that T4.1 / T4.2 / T4.3 already follow.

This is the **multi-tile dataflow** milestone for the CRISPR track. Per
CRISPR PRD §4.2 + plan task T5.3 / C-M4 ("the hard part"): does the AIE2P
fan-out pattern (broadcast inputs, parallel match tiles, joiner)
accelerate the mismatch-count race against single-tile? The headline
number lives in `results/crispr/c-m4/measurements.json`.

## Files

| name           | sha256                                                             | role                                                                            |
|----------------|--------------------------------------------------------------------|---------------------------------------------------------------------------------|
| `final.xclbin` | `95c9c98167c05b34cd70cb0e539d0f1bb6e993b75b81333706f1b841ec2123b9` | XCLBIN: AIE2P bitstream + control program for the 3-tile (2 match + 1 join) topology. |
| `insts.bin`    | `3f82a922b45015b02f48a14c7a82418dc02d8bfcff15d1df8ad1d6632d7f560f` | Userspace instruction stream (`uint32` PDI fragment).                           |
| `host_runner`  | `e7e7d0f54b4d72940bf29c55114c88fa673430d78f483def0666b863d4ef70fd` | C++ host binary built from `bionpu/kernels/crispr/match_multitile/runner.cpp`.   |
| `MANIFEST.md`  | (this file)                                                        | Provenance + rebuild instructions.                                              |

## Pinned shape

The xclbin is **shape-pinned** to T5.3's correctness fixture (same as T4.3
so byte-equality comparison is direct):

| field                | value                                |
|----------------------|--------------------------------------|
| `N_GUIDES`           | 128                                  |
| `N_WINDOWS`          | 4096                                 |
| `SPACER_LEN`         | 20 nt                                |
| `SPACER_BYTES`       | 5 (2-bit packed: A=00, C=01, G=10, T=11) |
| `WINDOWS_PER_CHUNK`  | 64 (DM double-buffered)              |
| `N_MATCH_TILES`      | 2                                    |
| `GUIDES_PER_TILE`    | 64                                   |
| Output layout        | window-major `(N_WINDOWS, N_GUIDES)` `uint8`; the Python wrapper transposes to `(N_GUIDES, N_WINDOWS)` so the registered-op contract matches T4.3 / oracle. |

T6.2 (PAM filter + threshold + emit pipeline) replaces dense output with a
sparse hit-list ring buffer; T6.3 adds genome-scale scan.

## Tile topology

```
                      shim DMA in  ──guides──→ broadcast
                          │
                          ├─→ match_tile_0 (col=0, row=2): guides[0:64]
                          │     ↓ partial_0 (64 windows × 64 guides)
                          │
                          ├─→ match_tile_1 (col=0, row=3): guides[64:128]
                          │     ↓ partial_1
                          │
                          └─→ joiner_tile (col=1, row=2 — placed by aiecc)
                                ↓ out (64 windows × 128 guides)
                              shim DMA out
```

Plus an independent `windows` ObjectFifo broadcast from shim → both match
tiles (same depth=2 double-buffering as T4.3).

## Tile-memory budget

AIE2P: ~64 KiB DM per tile, ~16 KiB program memory per tile.

Per tile:

| tile          | resident                 | dbl-buf in        | dbl-buf out                | total bytes | % of 64 KiB |
|---------------|--------------------------|-------------------|----------------------------|-------------|-------------|
| match_0/1     | 640 (full guides)        | 640 (windows × 2) | 8192 (partial × 2)         | **9472**    | **14.4%**   |
| joiner        | —                        | 16384 (2 partials × 2) | 16384 (out × 2)       | **32768**   | **50.0%**   |

Peak (joiner) is well under the 64 KiB cap. Program memory (the
compiled `match_kernel.o` containing both `match` and `join` extern "C"
entry points; aiecc picks the right symbol per-tile) is small — the
algorithm is scalar nested-loop popcount; no template-heavy `aie::vector`
intrinsics. The 16 KiB program memory budget is comfortably unused.

## Dataflow ObjectFifo summary

| FIFO          | producer            | consumer(s)                    | depth | per-elem bytes |
|---------------|---------------------|--------------------------------|-------|----------------|
| `guides`      | shim                | match_0, match_1               | 1     | 640            |
| `windows`     | shim                | match_0, match_1               | 2     | 320            |
| `partial_0`   | match_0             | joiner                         | 2     | 4096           |
| `partial_1`   | match_1             | joiner                         | 2     | 4096           |
| `out`         | joiner              | shim                           | 2     | 8192           |

## Build provenance

- **Source**:
  - `bionpu/kernels/crispr/match_multitile/match_kernel.cc` (both
    `crispr_match_multitile_match` and `crispr_match_multitile_join`)
  - `bionpu/kernels/crispr/match_multitile/multitile.py` (IRON lowering)
  - `bionpu/kernels/crispr/match_multitile/runner.cpp` (host C++)
  - `bionpu/kernels/crispr/match_multitile/Makefile`
- **NPU2 flag**: `1` (REQUIRED — AIE2P. Without `NPU2=1` the kernel
  silently compiles for AIE2 and produces garbage. T0.1's recheck script
  enforces this.)
- **Build date (UTC)**: `2026-04-25T11:15Z`
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

cd <repo>/bionpu/kernels/crispr/match_multitile
make NPU2=1 clean
make NPU2=1 all

cp build/final.xclbin              <repo>/bionpu/dispatch/_npu_artifacts/crispr_match_multitile/final.xclbin
cp build/insts.bin                  <repo>/bionpu/dispatch/_npu_artifacts/crispr_match_multitile/insts.bin
cp crispr_match_multitile           <repo>/bionpu/dispatch/_npu_artifacts/crispr_match_multitile/host_runner
```

Then update the sha256 rows above. The C++ binary path is the same v1
trade-off as T4.1 / T4.2 / T4.3: pyxrt ships only as a Python 3.14 binding
on this host while ironenv is Python 3.11.

## What the binary does

The host binary:

1. Reads `--guides PATH` (640 bytes, packed `uint8`) and pushes it onto
   the `guides` ObjectFifo. IRON broadcasts it to both match tiles.
2. Reads `--windows PATH` (20480 bytes, packed `uint8`) and pushes it
   onto the `windows` ObjectFifo. IRON walks 64 chunks of 64 windows,
   broadcasting each chunk to both match tiles.
3. Each match tile (col=0, rows 2/3) computes `(64 windows, 64 guides)`
   partial mismatch matrices using `crispr_match_multitile_match` (XOR +
   2-bit-pair popcount + 5-byte sum — bit-equal to T4.3's algorithm).
4. The joiner tile (placed by aiecc — typically col=1, row=2) reads both
   partials per chunk and concatenates along the guide axis to produce
   `(64 windows, 128 guides)`. Output is forwarded via shim DMA out.
5. The host writes the output to `--out PATH` and prints
   `Avg / Min / Max NPU time: ... us.` on stdout. The Python wrapper
   parses these and transposes window-major → guide-major.

A `PASS!` marker is printed on a clean run; the wrapper hard-fails if
that marker is absent. There is **no** in-binary verify (mirrors T4.3) —
byte-equality verify lives in
`tracks/crispr/npu/match_multitile.py`.

## Recorded T5.3 baseline (C-M4)

Per `results/crispr/c-m4/measurements.json` and the run-summary alongside.
The two key numbers:

| metric                            | value (target / actual)                        |
|-----------------------------------|------------------------------------------------|
| byte-equality (NPU vs T4.3 oracle)| **PASS** (`np.array_equal == True`)            |
| windows / sec                     | recorded from the chr22 fixture run            |
| `tile_memory_used_bytes` (peak)   | **32768** (joiner; 50% of 64 KiB DM cap)       |
| `energy_source`                   | `real-xrt-smi` (T4.4 wired)                    |

T6.2 layers PAM filter + on-tile threshold + sparse-emit ring buffer on
this multi-tile foundation. T6.3 measures full GRCh38 scan throughput.

## Toolchain gaps encountered (filed in `gaps.yaml`)

The original PRD §4.2 sketch called for **4** match tiles; we landed on 2
because AIE2P compute tiles have only 2 input DMA channels each and a
4-into-1 joiner blew the budget (`error: 'aie.tile' op number of input
DMA channel exceeded!`). See `bionpu/kernels/crispr/match_multitile/gaps.yaml`
entry G-T5.3-001 for the verbatim diagnostic + the workaround
(2-match-tile fan-out, future memtile aggregation path documented).
