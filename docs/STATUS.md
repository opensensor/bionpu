# `bionpu` v0.1 status

What works end-to-end vs what's an extracted module that still needs
a driver script before it's usable on hardware.

## Subsystem status

| Subsystem | Status | Notes |
|---|---|---|
| `bionpu.verify.crispr` — byte-equality vs Cas-OFFinder TSV | ✅ working | 18 tests in `tests/test_verify_crispr.py`. Smoke-tested on the 422-row canonical reference at `reference/crispr/casoffinder-canonical.tsv` via `bionpu verify crispr`. |
| `bionpu.verify.basecalling` — byte-equality vs Dorado FASTQ | ✅ working | 10 tests in `tests/test_verify_basecalling.py`. Field-level divergence diagnostics (`read_id` / sequence / quality / header). |
| `bionpu.dispatch` — silicon-serialisation lock + NPU backend | ✅ extracted | The `npu_silicon_lock` mutex with pre-flight wedge detection is the canonical way to serialise `/dev/accel/accel0`. The `NpuBackend` abstraction covers `pyxrt` (in-process) and `subprocess` (fallback) paths. |
| `bionpu.kernels.{crispr,basecalling}` — AIE2P MLIR-AIE kernels | ✅ extracted, build via `make` | Six CRISPR kernels (PAM filter, match singletile, match multitile, match multitile memtile, crispr_net, pktmerge variant) and twelve basecalling kernels (conv stem, LSTM cell variants, linear projection, LSTM stack). Each ships `Makefile`, Python topology, kernel C++, and a `MANIFEST.md` describing the inputs/outputs. Pre-built `host_runner` binaries are gitignored — users `make` to rebuild. |
| `bionpu.bench.energy` — RAPL / nvsmi / xrt energy readers | ✅ extracted | Three per-device readers + the measurement harness. `bionpu/bench/POWER_DOMAINS.md` is the per-device specification. `bionpu/bench/energy/SANITY-LOG.md` is the calibration evidence log. |
| `bionpu.data.canonical_sites` — Cas-OFFinder TSV normaliser | ✅ working | The canonical home for the `(chrom, start, mismatch_count, guide_id, strand)` sort + LF-newline canonicalisation that the `verify` module depends on. |
| `bionpu.data.fetchers` — public dataset fetchers | ✅ extracted | SHA-pinned fetchers for Doench 2016, GUIDE-seq, HG002 pod5, reference genomes. The fetcher framework documents what every dataset entry must satisfy. |
| `bionpu.quant` — quantization passport / calibration | ✅ extracted | Calibration driver around `onnxruntime.quantization`, plus the passport schema that every quantized model in the repo carries (calibration source + op recipe + reproducibility hash). |
| `bionpu.iron_extensions.cascade_stream` | ⚠️ deprecated; superseded by upstream `aie.iron.CascadeFifo` ([Xilinx/mlir-aie #3039](https://github.com/Xilinx/mlir-aie/pull/3039)) | Kept as a back-compat shim for designs written against the pre-upstream API. Will be removed once that PR's `mlir-aie` floor is set in `pyproject.toml`. |
| `bionpu.cli` — `bionpu verify {crispr,basecalling}` | ✅ working | Returns exit code 0 on byte-equality, 1 on divergence. |
| `bionpu.cli` — `bionpu scan` / `bionpu basecall` / `bionpu bench` | ⚠️ v0.2 scope | Stub subcommands that print a "v0.2 scope" message. The kernels they would invoke are extracted (and individually buildable via `make`); what's missing is the top-level Python that ties scan input → NPU dispatch → output → verify into one command. Expected to land in v0.2. |
| Pre-computed `benchmarks/results/` snapshots | ⚠️ v0.2 scope | Empty until the `bionpu scan` / `bionpu basecall` driver is wired (above). The reproduction scripts at `benchmarks/{crispr,basecalling}/run_*.sh` are skeletons that document the v0.2 driver shape. |

## What's in v0.1 but you should know about it

- **Per-kernel `DESIGN.md` files** carry historical design notes from the
  internal investigations that produced each kernel. They're useful
  background for someone reading the kernel source, but they're written
  in the voice of an investigation log rather than reference docs. A
  cleanup pass to convert these into upstream-style architectural
  references is on the v0.2 list.

- **`bionpu.iron_extensions`** ships, but new code should use
  `aie.iron.CascadeFifo` from upstream `mlir-aie` instead.

- **`bionpu.cli`'s argparse skeleton** has the v0.2 subcommands listed
  with stub implementations. They print to stderr with a clear "v0.2
  scope" message rather than silently failing.

## v0.2 plan

1. Wire `bionpu scan` and `bionpu basecall` into a real driver pipeline.
2. Run `benchmarks/{crispr,basecalling}/run_*.sh` against chr1 / chr19 /
   chr22 (and a small representative pod5) to populate
   `benchmarks/results/`.
3. Tag `v0.2` once the headline numbers are reproducible from the
   committed scripts on a clean machine.
4. Remove the `bionpu.iron_extensions` shim if upstream `mlir-aie` ships
   a release containing the `CascadeFifo` PR.
5. Convert kernel `DESIGN.md` investigation logs into architectural
   reference docs.
