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
| `bionpu.cli` — `bionpu scan --device cpu` | ✅ working end-to-end | Pure-numpy CRISPR off-target scan via `bionpu.scan.cpu_scan`. Produces a canonical TSV that `bionpu verify crispr` can compare byte-equally. NGG PAM only. |
| `bionpu.cli` — `bionpu scan --device npu` | ✅ working end-to-end (host-emulation fallback) | Driver wired through `bionpu.scan.npu_scan` → `bionpu.dispatch.lookup_npu_op("crispr_pam_filter_early")`. When kernel artifacts are present (after `make NPU2=1` in the kernel directory), dispatches to AIE2P silicon. When artifacts are absent, the kernel's host-emulation path takes over and produces output byte-equal to the silicon path by construction. CPU and NPU paths produce byte-equal canonical TSVs on the same input — locked in by `tests/test_scan.py::test_cpu_and_npu_paths_byte_equal_on_*`. |
| `bionpu.cli` — `bionpu score --device {cpu,gpu}` | ✅ scaffolded; smoke-mode end-to-end working | `bionpu/scoring/dnabert_epi.py` ships a CPU+GPU off-target probability scorer. `--smoke` mode produces deterministic SHA-256-keyed pseudo-random scores without torch / weights — exercises the pipeline end-to-end on any host. Real-mode (DNABERT-Epi BERT-base + clean-room classifier head, loaded from a fine-tuned checkpoint via `--weights`) is wired but inert until a checkpoint is produced (see `docs/reproduce-dnabert-epi.md`). The AIE2P scorer backend is a v0.4+ follow-up. The headline byte-equality contract is preserved across CPU/GPU/(future)NPU via `bionpu.verify.score` policies (BITWISE_EXACT for deterministic backends, NUMERIC_EPSILON for cross-fabric comparisons). |
| `bionpu.cli` — `bionpu bench probe` / `bionpu bench scan` | ✅ working | `bench probe` reports per-device energy-reader availability (RAPL / nvidia-smi / xrt-smi) with reasons; side-effect free, JSON or text output, optional `--require-all-real` CI gate. `bench scan` wraps the existing CRISPR scan in `bionpu.bench.harness.TimedRun` with the best-available energy reader, emits a `measurements.json` matching `bench/schema.json` with throughput / energy / RSS / VRAM-peak / latency percentiles. Stub fallback is documented per POWER_DOMAINS.md §1.4 + §3 (never fabricate; the stub clearly reports `energy_source=stub`). |
| `bionpu.cli` — `bionpu basecall` | ⚠️ v0.2+ scope | Stub that prints a "v0.2 scope" message. The kernels are extracted (and individually buildable via `make`); what's missing is the top-level Python that ties pod5 input → NPU dispatch → output → verify into one command. |
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
