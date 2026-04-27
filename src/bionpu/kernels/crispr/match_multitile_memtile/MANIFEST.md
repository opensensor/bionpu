# crispr_match_multitile_memtile — kernel sources

**AM020 cross-walk follow-up to .** Recovers the original CRISPR PRD
§4.2 sketch (4 match tiles × 32 guides each) via memtile-mediated fan-in
aggregation. The 2-into-1 reduction was forced into by 
(compute-tile 2-input-DMA-channel ceiling) is bypassed here because the
fan-in target is the memtile (6 MM2S + 6 S2MM channels per AM020 Ch. 5
p. 74), not a compute tile.

## Files

| name                  | role |
|-----------------------|------|
| `__init__.py`         | Public Python surface (`CrisprMatchMultiTileMemtile`). |
| `multitile_memtile.py`| IRON lowering (4-into-1 via `outC.prod().join(...)`). |
| `match_kernel.cc`     | AIE2P compute kernel (single `crispr_match_memtile_match` symbol). |
| `runner.cpp`          | C++ host runner (CXXABI 1.3.15, g++-15). |
| `Makefile`            | Build orchestration (`NPU2=1` required for AIE2P). |
| `CMakeLists.txt`      | Alternate host-runner build (CMake; same output). |
| `MANIFEST.md`         | This file. |
| `DESIGN.md`           | Full AM020 cross-walk rationale. |
| ``           | Toolchain-gap report. |

## Vendored artifacts

The xclbin/insts/host-runner are vendored under
`bionpu/dispatch/_npu_artifacts/crispr_match_multitile_memtile/` per the
v1-thin "pre-built artifacts" pattern. See
that directory's `MANIFEST.md` for the sha256s and rebuild instructions.

## Topology (per AM020 Figure 22 + canonical AIE-ML memtile aggregation)

```
                                 shim DMA
                                    │
                                    │ guides (broadcast)
                                    │ windows (broadcast)
                  ┌─────────┬───────┼───────┬─────────┐
                  ↓         ↓       ↓       ↓         ↓
              match_0   match_1  match_2  match_3
              (32 g.)   (32 g.)  (32 g.)  (32 g.)
                  │         │       │         │
                  │         │       │         │
                  └─────────┴──┬────┴─────────┘
                               ↓
                  [memtile: outC = outC.prod().join(offsets, ...)]
                               │
                               │ joined (n_windows × N_GUIDES)
                               ↓
                            shim DMA
```

Each match tile MM2S streams its `(64 windows × 32 guides) = 2048 B`
partial into a separate memtile S2MM channel (channels 0..3 — the
east/west-capable ones per AM020). Memtile reorganizes the 4 partials
into a single contiguous `(64 windows × 128 guides) = 8192 B` window-
major buffer using 5D address generation (AM020 Ch. 5 p. 71). Memtile
MM2S → shim closes the path. **No joiner compute tile** — the join is
fabric-side.

## Tile-memory budget (per match tile and memtile)

AIE2P compute tile: 64 KiB DM cap (per AM020 Table 14, AIE-ML number
verified empirically on AIE2P via ).

Per match tile (Tiles match_0..match_3, all identical):

| component                          | bytes     |
|------------------------------------|-----------|
| guides full batch (resident)       | 640       |
| windows chunk dbl-buf              | 640       |
| partial out dbl-buf (64 × 32 × 2)  | 4096      |
| **subtotal**                       | **5376**  |
| % of 64 KiB cap                    | **8.2%**  |

Per memtile (the new "joiner" — fabric-side, 512 KiB cap per AM020 Ch. 5):

| component                            | bytes      |
|--------------------------------------|------------|
| 4 partials × 2 dbl-buf × 64 × 32     | 16384      |
| 1 joined × 2 dbl-buf × 64 × 128      | 16384      |
| **subtotal**                         | **32768**  |
| % of 512 KiB cap                     | **6.25%**  |

The per-match-tile footprint is ~half 's (5376 B vs 9472 B) because
each tile carries a narrower 32-guide slice. The memtile footprint is
identical to 's joiner-tile footprint (the work is the same; only
the location moved from compute tile to memtile).

## ObjectFifo summary

| FIFO       | producer     | consumer(s)                           | depth | per-elem bytes |
|------------|--------------|---------------------------------------|-------|----------------|
| `guides`   | shim         | match_0, match_1, match_2, match_3    | 1     | 640            |
| `windows`  | shim         | match_0, match_1, match_2, match_3    | 2     | 320            |
| `outC`     | match_0..3   | memtile (joined) → shim               | 2     | 8192 (joined)  |
| `memC0..3` | match_<i>    | memtile (per-tile slot of `outC`)     | 2     | 2048           |

`outC.prod().join([0, 32, 64, 96], obj_types=[partial_chunk_ty]*4, ...)`
declares the memtile-aggregated FIFO; IRON lowers each per-tile producer
(`memC<i>`) into a memtile S2MM channel with an offset into the joined
buffer.

## How to rebuild

```bash
source /opt/xilinx/xrt/setup.sh
source <ironenv>/bin/activate
export MLIR_AIE_DIR=<xdna-bringup>/mlir-aie
export PEANO_INSTALL_DIR=<ironenv>/lib/python3.11/site-packages/llvm-aie

cd <repo>/bionpu/kernels/crispr/match_multitile_memtile
make NPU2=1 clean
make NPU2=1 all

# Vendor:
cp build/final.xclbin              <repo>/bionpu/dispatch/_npu_artifacts/crispr_match_multitile_memtile/final.xclbin
cp build/insts.bin                  <repo>/bionpu/dispatch/_npu_artifacts/crispr_match_multitile_memtile/insts.bin
cp crispr_match_multitile_memtile   <repo>/bionpu/dispatch/_npu_artifacts/crispr_match_multitile_memtile/host_runner
```

Then update sha256 rows in the vendored MANIFEST.md.

## Byte-equality contract

Output dense `(N_GUIDES, N_WINDOWS) uint8` matrix is **byte-equal** to
's 2-into-1 output (and 's single-tile output, and the NumPy
oracle) on the same fixture. The architectural change is fan-in width
only; the math is identical to / (verbatim XOR + 2-bit-pair
popcount + 5-byte sum on the 2-bit packed encoding).

## What's deferred

* On-tile PAM filtering — 's territory.
* On-tile sparse-emit + ring buffer — 's territory.
* Genome-scale walk — retests with this retrofit.
* Variable-rate stream filter-early — needs IRON below the
  ObjectFifo abstraction.
