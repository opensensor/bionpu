# crispr_pam_filter_early — pre-built T6.2 artifacts (C-M5, production path)

These three files are the on-disk surface that
`bionpu.kernels.crispr.pam_filter.CrisprPamFilterEarly` needs to run
the filter-early variant of the PAM-filter + threshold + sparse-emit
CRISPR kernel on the AIE2P NPU. Same v1-thin "pre-built xclbin +
lookup table" approach as T4.1 / T4.2 / T4.3 / T5.3.

This is the **production** path: Tile A drops PAM-failing windows
before the match tiles compute, and Tile Z applies the threshold +
emits sparse records.

## Files

| name           | sha256                                                             | role                                                                             |
|----------------|--------------------------------------------------------------------|----------------------------------------------------------------------------------|
| `final.xclbin` | `7057f84845577c64877c6ab782a3e0cb82b8471b6efb3e6b15977c2c2e1cda73` | XCLBIN: AIE2P bitstream + control program for the 4-tile (Tile A + 2 match + Tile Z) topology, filter-early variant. (G-T6.3-004 dynamic-fifo half by T9 swarm Wave 2, 2026-04-25 — windows_out ObjectFifo replaced with VariableRateFifo; lowered MLIR carries `aie.variable_rate = true` discardable attribute on the windows_out ObjectFifoCreateOp. Composed on top of T3 vectorisation half from same date — match-tile inner loop uses ::aie::vector<uint8,64> intrinsics; 22 vector ops vs scalar 0. Per-launch wall measurement see state/g-t6.3-004/dynamic-fifo-20260425T234647Z.json. Previous SHA `049eb39…` (T3 vectorisation), `d6c039c…` (T6.2 baseline) — see backup directories. **Followup-A experimental builds (2026-04-26, NOT vendored)**: tested Item 3's burst_length-annotation hypothesis. SHA `648fee63…` (burst_length=512 on shim BDs; -0.005% wall delta — falsified Item 3's BD-cliff hypothesis at the shim layer for this kernel; per-launch wall stays 6.144 s) and SHA `02b3e475…` (burst_length=64 on shim BDs; WEDGED firmware on first launch — see state/followup-a/dmesg-wedge-from-burst64-20260426T013800Z.log for dmesg signature). Vendored sha unchanged from T9 baseline; the env-var override that produced these experiments lives in `pam_filter.py::_maybe_install_burst_length_override` and is dormant unless `BIONPU_PAM_FILTER_SHIM_BURST_LENGTH` is set. Full followup-A evidence: state/followup-a/per-launch-wall-20260426T013500Z.json + state/followup-a/bd-audit-20260426T011703Z.json.) |
| `insts.bin`    | `38ae4b75f01471ad724ab3de741bd7bf050e17b48765ea29bd9fe8a5e195ef99` | Userspace instruction stream (`uint32` PDI fragment) — unchanged from T6.2 baseline (host runtime sequence is independent of the match-tile vectorisation).                            |
| `host_runner`  | `4235d235f6c36b14c2d8ea00f51728751884fc4edac3016e20f7486d3f785a52` | C++ host binary built from `bionpu/kernels/crispr/pam_filter/runner.cpp` — unchanged.        |
| `MANIFEST.md`  | (this file)                                                        | Provenance + rebuild instructions.                                                |

## Pinned shape

| field                | value                                |
|----------------------|--------------------------------------|
| `N_GUIDES`           | 128                                  |
| `N_WINDOWS`          | 4096                                 |
| `SPACER_LEN`         | 20 nt                                |
| `SPACER_BYTES`       | 5 (2-bit packed: A=00, C=01, G=10, T=11) |
| `PAM_LEN`            | 3 nt (NGG)                           |
| `WINDOW_BYTES_IN`    | 6 (5 spacer + 1 PAM)                 |
| `WINDOWS_PER_CHUNK`  | 64 (DM double-buffered)              |
| `N_MATCH_TILES`      | 2 (G-T5.3-001 / G-T6.2-002 inherited) |
| `GUIDES_PER_TILE`    | 64                                   |
| `EMIT_RECORD_BYTES`  | 8                                    |
| Output layout        | length-prefixed sparse records (host concatenates 64 chunks). |

## Build provenance

- **Source**: `bionpu/kernels/crispr/pam_filter/{pam_filter.py, tile_a_filter.cc, runner.cpp, Makefile}`
- **NPU2 flag**: `1` (REQUIRED — AIE2P)
- **Build date (UTC)**: `2026-04-25T13:00Z` (see git log for the actual commit timestamp)
- **Bring-up env**:
  - XRT 2.23.0 at `/opt/xilinx/xrt/`
  - amdxdna 2.23.0
  - NPU firmware 1.1.2.64
  - Architecture `aie2p` (6×8 = 48 tiles)
  - `~/xdna-bringup/ironenv` with Python 3.11.15, mlir-aie, llvm-aie

## How to rebuild

See the kernel directory's `MANIFEST.md`:

```bash
cd <repo>/bionpu/kernels/crispr/pam_filter
make NPU2=1 clean
make NPU2=1 all
cp build/early/final.xclbin <repo>/bionpu/dispatch/_npu_artifacts/crispr_pam_filter_early/final.xclbin
cp build/early/insts.bin    <repo>/bionpu/dispatch/_npu_artifacts/crispr_pam_filter_early/insts.bin
cp crispr_pam_filter        <repo>/bionpu/dispatch/_npu_artifacts/crispr_pam_filter_early/host_runner
```

Then update the sha256 rows above.

## What the binary does

1. Reads `--guides PATH` (640 bytes) and pushes onto the `guides`
   ObjectFifo. IRON broadcasts to both match tiles.
2. Reads `--windows PATH` (24576 bytes — 4096 records × 6 bytes:
   5 spacer + 1 PAM) and pushes onto the `windows_in` ObjectFifo.
   Tile A reads each chunk, checks NGG, and forwards only PAM-passing
   windows to the match tiles + the corresponding pam_meta byte to
   Tile Z.
3. Match tiles compute (window × guide) mismatch counts (verbatim
   T5.3 arithmetic).
4. Tile Z applies threshold (≤ max_mm) and emits surviving (window,
   guide, mm) records via shim DMA-out into a length-prefixed sparse
   buffer.
5. Host writes the sparse blob to `--out PATH`. The Python wrapper
   parses the per-slot length-prefixed layout — the kernel emits ONE
   length-prefixed slot per sub-chunk (`N_CHUNKS = 64` slots × 2048 B
   each), with `chunk_base_window_idx` hardcoded to 0; the Python
   wrapper (`bionpu.kernels.crispr.pam_filter.decode_per_slot_sparse_buffer`)
   reapplies the per-slot offset so the decoded `window_idx` is
   launch-relative in `[0, N_WINDOWS)`. (G-T6.3-005, 2026-04-25:
   the original `decode_sparse_buffer` was monolithic and only saw
   slot-0's records — see commit closing G-T6.3-005 for the full
   diagnosis.)

A `PASS!` marker is printed on a clean run.

## Filter-early vs filter-late

Both variants ship from the same source. Differences:

* This (early) variant: Tile A's `crispr_pam_filter_tile_a_early`
  symbol drops PAM-failing windows; Tile Z's
  `crispr_pam_filter_tile_z_early` skips them based on the pam_meta
  byte.
* The late variant (`crispr_pam_filter_late/`): Tile A's
  `crispr_pam_filter_tile_a_late` passes everything through; Tile Z's
  `crispr_pam_filter_tile_z_late` re-checks PAM on every window
  before applying the threshold.

Output bytes are identical between the two variants (after canonical
normalization). The comparison is the C-M5 headline finding —
documented in `results/crispr/c-m5/filter-early-vs-late.md`.
