# crispr_pam_filter_pktmerge — pre-built T3.3 artifacts (C-M5-pktmerge)

Three on-disk artifacts that
`bionpu.kernels.crispr.pam_filter_pktmerge.CrisprPamFilterPktmerge`
needs to run the T3.3 PacketFifo retrofit of the T6.2 PAM-filter +
threshold + sparse-emit CRISPR kernel on the AIE2P NPU.

T3.3 closes the OTHER HALF of G-T6.2-001 (Phase 1 documented; Phase 2
closes via the fork's PacketFifo primitive — T2.2). Per
`results/crispr/c-m5-pktmerge/verdict.md`: API-surface ratified +
xclbin builds + runs on silicon with byte-equal output to T6.2's
filter-early. The silicon-level cycle saving from pktMerge fabric
filtering is BLOCKED at the live-program path by G-T3.3-001 (fork-
internal Program/PacketFifo endpoint shape mismatch — out of T3.3's
"OUTER REPO ONLY" scope).

## Files

| name           | sha256                                                             | role                                                                                                |
|----------------|--------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| `final.xclbin` | `52eb675219cb9d21ae48c62eb803f6fecf4c3a6bbe52c3de8e252fee1aec646b` | XCLBIN: AIE2P bitstream + control program for the 4-tile T3.3 topology (Tile A + 2 match + Tile Z). |
| `insts.bin`    | `38ae4b75f01471ad724ab3de741bd7bf050e17b48765ea29bd9fe8a5e195ef99` | Userspace instruction stream (`uint32` PDI fragment). Same I/O contract as T6.2 filter-early.       |
| `host_runner`  | `4235d235f6c36b14c2d8ea00f51728751884fc4edac3016e20f7486d3f785a52` | C++ host binary built from `bionpu/kernels/crispr/pam_filter_pktmerge/runner.cpp`.                  |
| `MANIFEST.md`  | (this file)                                                        | Provenance + rebuild instructions.                                                                  |

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
| `HEADER_BYTES`       | 1 (uint8 packet header_dtype)        |
| `PACKET_ID_VALID`    | 1 (PAM-passing windows)              |
| `PACKET_ID_INVALID`  | 0 (PAM-failing windows)              |
| Output layout        | length-prefixed sparse records (host concatenates 64 chunks). |

## Build provenance

- **Source**: `bionpu/kernels/crispr/pam_filter_pktmerge/{pam_filter_pktmerge.py, tile_a_pktmerge.cc, runner.cpp, Makefile, gaps.yaml}`
- **NPU2 flag**: `1` (REQUIRED — AIE2P)
- **Build date (UTC)**: `2026-04-25T16:21Z`
- **Bring-up env**:
  - XRT 2.23.0 at `/opt/xilinx/xrt/`
  - amdxdna 2.23.0_20260425
  - NPU firmware 1.1.2.64
  - Architecture `aie2p` (6×8 = 48 tiles)
  - `~/xdna-bringup/ironenv` with Python 3.14.4, mlir-aie fork (T2.2 PacketFifo
    landed at fork-internal commit `236059d6`)

## How to rebuild

```bash
cd <repo>/bionpu/kernels/crispr/pam_filter_pktmerge
source /opt/xilinx/xrt/setup.sh
source ~/xdna-bringup/ironenv/bin/activate
export MLIR_AIE_DIR=~/xdna-bringup/mlir-aie
export PEANO_INSTALL_DIR=~/xdna-bringup/ironenv/lib/python3.14/site-packages/llvm-aie
export PATH=<repo>/third_party/mlir-aie/install/bin:$PATH
make NPU2=1 clean
make NPU2=1 all
cp build/final.xclbin <repo>/bionpu/dispatch/_npu_artifacts/crispr_pam_filter_pktmerge/final.xclbin
cp build/insts.bin    <repo>/bionpu/dispatch/_npu_artifacts/crispr_pam_filter_pktmerge/insts.bin
cp crispr_pam_filter_pktmerge <repo>/bionpu/dispatch/_npu_artifacts/crispr_pam_filter_pktmerge/host_runner
```

Then update the sha256 rows above.

## What the binary does

1. Reads `--guides PATH` (640 bytes) and pushes onto the `guides`
   ObjectFifo. IRON broadcasts to both match tiles.
2. Reads `--windows PATH` (24576 bytes — 4096 records × 6 bytes:
   5 spacer + 1 PAM) and pushes onto the `windows_in` ObjectFifo.
   Tile A reads each chunk, runs the NGG check, writes the per-window
   1-bit valid header into `pam_meta` and a compact filtered chunk
   into `windows_out`.
3. Match tiles compute (window × guide) mismatch counts (verbatim
   T6.2 / T5.3 arithmetic — byte-equality by construction).
4. Tile Z gates emits on the per-window `pam_meta` byte (PAM-failing
   windows are dropped at this stage on the live program; on the
   silicon-level pktMerge path post-G-T3.3-001 they are dropped in
   fabric before reaching the match tiles), applies threshold (≤ max_mm),
   and emits surviving (window, guide, mm) records via shim DMA-out.
5. Host writes the sparse blob to `--out PATH`.

A `PASS!` marker is printed on a clean run.

## Difference vs T6.2 filter-early

The xclbin differs at the IRON-level Kernel-symbol bindings (T3.3
binds `crispr_pam_filter_tile_a_pktmerge` /
`crispr_pam_filter_tile_z_pktmerge`; T6.2 filter-early binds
`crispr_pam_filter_tile_a_early` / `crispr_pam_filter_tile_z_early`)
but the C++ math is byte-equal. The IRON Program also includes a
PacketFifo topology *declaration* (`pktmerge_topology()`) for API-
surface ratification of T2.2's PacketFifo + the canonical filter-
early construction; per G-T3.3-001 the PacketFifo is not yet wired
into the live Program flow (needs upstream fork edits beyond T3.3's
scope).

## Related

* Verdict: `results/crispr/c-m5-pktmerge/verdict.md`
* Measurements: `results/crispr/c-m5-pktmerge/measurements.json`
* Test: `tests/test_pam_filter_pktmerge_npu.py`
* Fork PacketFifo: `third_party/mlir-aie/python/iron/packet.py` (T2.2)
* T6.2 reference (byte-equal twin): `bionpu/kernels/crispr/pam_filter/`
