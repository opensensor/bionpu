# Changelog

## v0.1.0.dev0 — Initial public release (in progress)

The first public release of `bionpu` — extracted from the internal
genetics development tree, re-licensed to GPL-3.0, scrubbed of
project-internal task IDs, and structured as a defensible asset.

### Added

- **`bionpu.verify`** — byte-equality harness with documented public
  API, frozen and tested before extraction:
  - `compare_against_cas_offinder` for CRISPR off-target hits TSVs
    (sorts by canonical key, normalises line endings, computes
    SHA-256 over the canonical wire form, reports first-N
    divergences with chrom/position/strand).
  - `compare_against_dorado` for Nanopore basecaller FASTQs
    (sorts by `read_id`, normalises CRLF, field-level divergence
    diagnostics distinguishing `read_id` / sequence / quality /
    header mismatches).
  - 18 tests covering equality, sort-order invariance, CRLF
    handling, divergence reporting, max-divergences cap, and
    pinned SHA-256 regression guards.

- **`bionpu.dispatch`** — NPU dispatch infrastructure:
  - `npu_silicon_lock` — process-level mutex on
    `/tmp/bionpu-npu-silicon.lock` with pre-flight wedge detection
    (scans `dmesg` for `aie2_dump_ctx` / `Firmware timeout` /
    `aie2_tdr_work` signatures BEFORE submitting work).
  - `NpuBackend` abstraction — `pyxrt` in-process path (default) +
    `subprocess` fallback for environments where `pyxrt` cannot be
    imported.

- **`bionpu.kernels`** — 18 AIE2P kernels:
  - 6 CRISPR: PAM filter (filter-early + filter-late variants),
    match (singletile / multitile / multitile-memtile),
    `crispr_net`, pktmerge variant.
  - 12 basecalling: conv stem, linear projection, LSTM cell
    (bf16 / int8 / acc / acc-cascade / compressed variants),
    LSTM stack (3 variants).

- **`bionpu.bench`** — energy + timing measurement:
  - Three per-device readers: AMD RAPL (CPU), `nvidia-smi` (GPU),
    `xrt-smi` (NPU AIE-partition firmware estimate).
  - Three-phase sustained-load measurement shape (pre-warmup,
    measurement window, drift-detection window).
  - `POWER_DOMAINS.md` per-device specification (includes,
    excludes, sampling rates, sources, known issues).

- **`bionpu.data`** — Cas-OFFinder canonical TSV normaliser, public
  dataset fetchers (Doench 2016, GUIDE-seq, HG002 pod5, reference
  genomes), in-repo smoke fixture loaders.

- **`bionpu.quant`** — quantization passport schema, `onnxruntime`
  calibration driver, Peano IR export hook.

- **`bionpu.iron_extensions.cascade_stream`** — back-compat shim
  for cascade-chain topology helpers. **Superseded** by upstream
  `aie.iron.CascadeFifo` ([Xilinx/mlir-aie #3039]); will be removed
  once that PR's `mlir-aie` floor is set in `pyproject.toml`.

- **`bionpu.cli`** — `bionpu verify {crispr,basecalling}` end-to-end
  CLI; exits 0 on byte-equality, 1 on divergence.

- **`reference/`** — Cas-OFFinder canonical reference (422 rows),
  chr22 ten-guide fixture, basecalling smoke FASTQ.

- **`docs/`** — energy methodology, reproduction recipe, per-subsystem
  v0.1 status.

[Xilinx/mlir-aie #3039]: https://github.com/Xilinx/mlir-aie/pull/3039

### Deferred to v0.2

- End-to-end driver scripts for `bionpu scan` / `bionpu basecall`
  (kernels are extracted and individually buildable; what's missing
  is the top-level Python that ties scan input → NPU dispatch →
  output → verify into one CLI invocation).

- Pre-computed `benchmarks/results/{crispr,basecalling}/*.json`
  snapshots for chr1 / chr19 / chr22 / a representative pod5.

- Conversion of per-kernel `DESIGN.md` files from
  internal-investigation logs into upstream-style architectural
  reference docs (one done so far —
  `lstm_cell_bf16_acc_cascade/DESIGN.md`).

- Removal of the `bionpu.iron_extensions` back-compat shim once
  `aie.iron.CascadeFifo` reaches a tagged `mlir-aie` release.
