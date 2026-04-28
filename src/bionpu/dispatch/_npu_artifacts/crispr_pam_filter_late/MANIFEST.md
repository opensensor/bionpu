# crispr_pam_filter_late — pre-built T6.2 artifacts (C-M5, comparison path)

These three files are the on-disk surface that
`bionpu.kernels.crispr.pam_filter.CrisprPamFilterLate` needs to run
the filter-late variant of the PAM-filter + threshold + sparse-emit
CRISPR kernel. **This variant ships as a comparison artifact**, not
as the production path: it computes match scores on every window
(including the 7/8 that fail the PAM check) and only filters at
Tile Z.

The point of shipping it is to make the filter-early-vs-filter-late
trade-off measurable — both variants produce byte-identical output
on the same input, but the work distribution is different. See
`results/crispr/c-m5/filter-early-vs-late.md` for the headline
comparison.

## Files

| name           | sha256                                                             | role                                                                            |
|----------------|--------------------------------------------------------------------|---------------------------------------------------------------------------------|
| `final.xclbin` | `5467b042cd15ba0b11b315a190fe057252f33e88ba01b37bf9392a5fedb68784` | XCLBIN: AIE2P bitstream + control program for the 4-tile topology, filter-late variant. |
| `insts.bin`    | `38ae4b75f01471ad724ab3de741bd7bf050e17b48765ea29bd9fe8a5e195ef99` | Userspace instruction stream (`uint32` PDI fragment).                            |
| `host_runner`  | `4235d235f6c36b14c2d8ea00f51728751884fc4edac3016e20f7486d3f785a52` | C++ host binary (identical to the early variant — same runner.cpp).             |
| `MANIFEST.md`  | (this file)                                                        | Provenance + rebuild instructions.                                                |

The host_runner is **byte-identical** to the early variant's binary
(it's compiled from the same `runner.cpp`); the difference is
encoded in the xclbin (which selects the `_late` symbols on Tile A
and Tile Z).

## Pinned shape

Identical to the early variant — see
`bionpu/dispatch/_npu_artifacts/crispr_pam_filter_early/MANIFEST.md`.

## Build provenance

- **Source**: same as the early variant — `bionpu/kernels/crispr/pam_filter/`.
- **Build date (UTC)**: `2026-04-25T13:00Z`.
- **Bring-up env**: same XRT 2.23.0 + ironenv + AIE2P.

## How to rebuild

See the kernel directory's `MANIFEST.md`:

```bash
cd <repo>/bionpu/kernels/crispr/pam_filter
make NPU2=1 clean
make NPU2=1 all
cp build/late/final.xclbin <repo>/bionpu/dispatch/_npu_artifacts/crispr_pam_filter_late/final.xclbin
cp build/late/insts.bin    <repo>/bionpu/dispatch/_npu_artifacts/crispr_pam_filter_late/insts.bin
cp crispr_pam_filter       <repo>/bionpu/dispatch/_npu_artifacts/crispr_pam_filter_late/host_runner
```

## What's different from the early variant

Same xclbin shape, different Tile-A / Tile-Z symbols:

* Tile A: `crispr_pam_filter_tile_a_late` — passes every window through;
  records the PAM-pass byte in the `pam_meta` ObjectFifo for Tile Z.
* Tile Z: `crispr_pam_filter_tile_z_late` — re-checks the PAM byte
  before the threshold + emit. Otherwise identical to the early
  variant.

Match tiles are unchanged across the two variants (verbatim T5.3
mismatch arithmetic).

The kernel object `tile_a_filter.o` exposes BOTH variants' symbols;
aiecc selects per-tile based on the IRON lowering's
`Kernel(symbol, …)` references in `pam_filter.py`'s
`crispr_pam_filter(..., mode='late')` path.
