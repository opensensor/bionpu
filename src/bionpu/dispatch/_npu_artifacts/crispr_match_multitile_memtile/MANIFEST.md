# crispr_match_multitile_memtile — pre-built T5.3-memtile artifacts

These three files are the complete on-disk surface that
`bionpu.kernels.crispr.match_multitile_memtile.CrisprMatchMultiTileMemtile`
needs to run the **memtile-aggregated 4-into-1 multi-tile** CRISPR
mismatch-count kernel on the AIE2P NPU. Same v1-thin "pre-built xclbin
+ lookup table" approach the umbrella PRD §4.1 mandates and that T4.1 /
T4.2 / T4.3 / T5.3 already follow.

This is the **AM020 cross-walk follow-up** to T5.3. T5.3 had to reduce
PRD §4.2's original 4-match-tile sketch to 2 match tiles because compute
tiles only have 2 input DMA channels (G-T5.3-001). AM020 Ch. 5 p. 74
documents the canonical AIE-ML fix: aggregate via the memtile (6 MM2S +
6 S2MM channels — 3× the compute-tile fan-in budget). This kernel
implements that path.

## Files

| name           | sha256                                                             | role                                                                            |
|----------------|--------------------------------------------------------------------|---------------------------------------------------------------------------------|
| `final.xclbin` | `cb17162fcf13d84f3fa2b1ff05d012bc841c6ba46bfe8525b221994806b1bc3f` | XCLBIN: AIE2P bitstream + control program — 4 match tiles + 1 memtile (no joiner compute tile). |
| `insts.bin`    | `3f82a922b45015b02f48a14c7a82418dc02d8bfcff15d1df8ad1d6632d7f560f` | Userspace instruction stream (`uint32` PDI fragment).                           |
| `host_runner`  | `e7e7d0f54b4d72940bf29c55114c88fa673430d78f483def0666b863d4ef70fd` | C++ host binary built from `bionpu/kernels/crispr/match_multitile_memtile/runner.cpp`.   |
| `MANIFEST.md`  | (this file)                                                        | Provenance + rebuild instructions.                                              |

## Pinned shape

The xclbin is **shape-pinned** to the same correctness fixture as T4.3 +
T5.3 so byte-equality comparison is direct:

| field                | value                                |
|----------------------|--------------------------------------|
| `N_GUIDES`           | 128                                  |
| `N_WINDOWS`          | 4096                                 |
| `SPACER_LEN`         | 20 nt                                |
| `SPACER_BYTES`       | 5 (2-bit packed: A=00, C=01, G=10, T=11) |
| `WINDOWS_PER_CHUNK`  | 64 (DM double-buffered)              |
| `N_MATCH_TILES`      | **4** (recovers PRD §4.2 sketch)     |
| `GUIDES_PER_TILE`    | **32**                               |
| Output layout        | window-major `(N_WINDOWS, N_GUIDES)` `uint8`; the Python wrapper transposes to `(N_GUIDES, N_WINDOWS)` so the registered-op contract matches T4.3 / T5.3 / oracle. |

## Tile topology

```
                           shim DMA in
                                │
                                │ guides (broadcast, depth=1, fired once)
                                │ windows (broadcast, depth=2, fired N_CHUNKS)
                ┌────────┬──────┼──────┬────────┐
                ↓        ↓      ↓      ↓        ↓
             match_0  match_1 match_2 match_3
             g[0:32]  g[32:64] g[64:96] g[96:128]
                │        │      │        │
                │ memC0  │memC1 │memC2  │memC3
                │ (2048B)│(2048B)│(2048B)│(2048B)
                ↓        ↓      ↓        ↓
                          [memtile]
                  (aie.objectfifo.link with offsets [0, 32, 64, 96])
                          5D address gen reorganises 4 partials
                          → joined (64 windows × 128 guides)
                              │
                              ↓ out (depth=2)
                          shim DMA out
```

**No joiner compute tile** — the join is fabric-side via memtile DMA
(`aie.objectfifo.link [@memC0, @memC1, @memC2, @memC3] -> [@out]([0, 32, 64, 96] [])`
in the generated MLIR).

## Tile-memory budget

AIE2P compute tile: 64 KiB DM (verified empirically via T5.3); memtile:
nominal 512 KiB per AM020 (AIE-ML; AIE2P unverified at this layer).

| tile          | resident                 | dbl-buf in        | dbl-buf out                | total bytes | % of cap |
|---------------|--------------------------|-------------------|----------------------------|-------------|----------|
| match_0..3    | 640 (full guides)        | 640 (windows × 2) | 4096 (partial × 2)         | **5376**    | **8.2%** of 64 KiB |
| memtile       | —                        | 16384 (4 partials × 2) | 16384 (out × 2)       | **32768**   | **6.25%** of 512 KiB |

Per-match-tile footprint is **~half** T5.3's 9472 B — narrower 32-guide
slice. Memtile footprint matches T5.3's joiner-tile footprint (the work
moved location, not size).

## Dataflow ObjectFifo summary

| FIFO       | producer     | consumer(s)                           | depth | per-elem bytes |
|------------|--------------|---------------------------------------|-------|----------------|
| `guides`   | shim         | match_0, match_1, match_2, match_3    | 1     | 640            |
| `windows`  | shim         | match_0, match_1, match_2, match_3    | 2     | 320            |
| `memC0..3` | match_<i>    | memtile (per-tile slot of joined out) | 2     | 2048           |
| `out`      | memtile      | shim                                  | 2     | 8192 (joined)  |

`outC.prod().join([0, 32, 64, 96], obj_types=[partial_chunk_ty]*4, ...)`
in the IRON lowering creates the 4-into-1 memtile-aggregated
`out` FIFO. IRON emits an `aie.objectfifo.link` with stride offsets in
the generated MLIR.

## Build provenance

- **Source**:
  - `bionpu/kernels/crispr/match_multitile_memtile/match_kernel.cc` (single
    `crispr_match_memtile_match` symbol — no joiner kernel)
  - `bionpu/kernels/crispr/match_multitile_memtile/multitile_memtile.py` (IRON lowering)
  - `bionpu/kernels/crispr/match_multitile_memtile/runner.cpp` (host C++)
  - `bionpu/kernels/crispr/match_multitile_memtile/Makefile`
- **NPU2 flag**: `1` (REQUIRED — AIE2P. Without `NPU2=1` the kernel
  silently compiles for AIE2 and produces garbage. T0.1's recheck
  enforces this.)
- **Build date (UTC)**: `2026-04-25T17:30Z`
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

cd <repo>/bionpu/kernels/crispr/match_multitile_memtile
make NPU2=1 clean
make NPU2=1 all

cp build/final.xclbin              <repo>/bionpu/dispatch/_npu_artifacts/crispr_match_multitile_memtile/final.xclbin
cp build/insts.bin                  <repo>/bionpu/dispatch/_npu_artifacts/crispr_match_multitile_memtile/insts.bin
cp crispr_match_multitile_memtile   <repo>/bionpu/dispatch/_npu_artifacts/crispr_match_multitile_memtile/host_runner
```

Then update the sha256 rows above.

## What the binary does

The host binary:

1. Reads `--guides PATH` (640 bytes, packed `uint8`) and pushes it onto
   the `guides` ObjectFifo. IRON broadcasts it to all 4 match tiles.
2. Reads `--windows PATH` (20480 bytes, packed `uint8`) and pushes it
   onto the `windows` ObjectFifo. IRON walks 64 chunks of 64 windows,
   broadcasting each chunk to all 4 match tiles.
3. Each match tile (col=0, rows 2..5) computes `(64 windows × 32 guides)`
   partial mismatch matrices using `crispr_match_memtile_match` (XOR +
   2-bit-pair popcount + 5-byte sum — bit-equal to T4.3 / T5.3).
4. The memtile (col=0, MemTile row) reorganises the 4 per-chunk partials
   `(64 × 32 each)` into a single `(64 × 128)` joined buffer via 5D DMA
   address generation with offsets `[0, 32, 64, 96]`. Memtile MM2S →
   shim closes the path. **No joiner compute tile**.
5. The host writes the output to `--out PATH` and prints
   `Avg / Min / Max NPU time: ... us.` on stdout. The Python wrapper
   parses these and transposes window-major → guide-major.

A `PASS!` marker is printed on a clean run; the wrapper hard-fails if
that marker is absent.

## T5.3-memtile measurements

Per `results/crispr/c-m4-memtile/measurements.json` and `run-summary.json`:

| metric                            | T5.3 (ref)        | T5.3-memtile      | speedup |
|-----------------------------------|-------------------|-------------------|---------|
| byte-equality vs T4.3 oracle      | PASS              | PASS              | —       |
| byte-equality vs T5.3 (this op)   | —                 | **PASS** (load-bearing) | — |
| windows / sec (kernel-only)       | 394 616           | **748 142**       | **1.90×** |
| wall-clock per launch (chr22×128) | 240.18 ms         | **176.4 ms**      | **1.36×** |
| avg NPU µs / iter                 | 10379.7           | 5474.9            | **1.90×** |
| per-match-tile bytes              | 9472              | 5376              | 0.57× (smaller) |
| memtile / joiner-tile bytes       | 32768 (joiner CT) | 32768 (memtile)   | same    |
| match tiles                       | 2                 | **4**             | 2× parallelism |
| guides per tile                   | 64                | **32**            | 0.5× per tile |
| `energy_source`                   | `real-xrt-smi`    | `real-xrt-smi`    | —       |

The kernel-only throughput approaches the predicted 2× match-tile
parallelism ceiling (1.90× actual vs 2× theoretical). Wall-clock ratio
is lower (1.36×) because the per-launch fixed overhead (XRT setup,
subprocess fork, file I/O) is unchanged.

## Toolchain gaps encountered

See `bionpu/kernels/crispr/match_multitile_memtile/gaps.yaml`. Empty
`gaps: []` if the build went clean; any IRON pattern absences
(e.g. memtile-S2MM-east/west exposure, 5D vs 4D address generation
visibility) get filed there.

## Relationship to T5.3

This is an **additive sibling**, not a replacement. T5.3's
`crispr_match_multitile` op stays registered and shipped as the
2-tile reference; this op (`crispr_match_multitile_memtile`) is the
4-tile recovery via memtile aggregation. Both ops produce byte-equal
output on the same fixture by construction.
