# bionpu

**AIE2P-accelerated genomics with reference-equivalence verification.**

`bionpu` is a public reference implementation of two genomics workloads
running on AMD's XDNA NPU (AIE2P silicon, Ryzen AI 9 HX / Strix family),
together with the byte-equality harness that proves the NPU output is
*exactly* the same as the canonical CPU reference — not "approximately
equal," not "within tolerance." Bit-for-bit identical.

| Workload | NPU kernel | Reference | What gets compared |
|---|---|---|---|
| CRISPR off-target scan | `bionpu/kernels/crispr/` | [cas-offinder](https://github.com/snugel/cas-offinder) | Hit TSVs (chrom, position, strand, mismatches) |
| Nanopore basecalling | `bionpu/kernels/basecalling/` | [Dorado](https://github.com/nanoporetech/dorado) | FASTQ reads (sequence + quality scores) |

This repository is the public release of work that was developed
internally; the public version is GPLv3, ships the kernels + dispatch
+ verification harness + reproducible benchmarks, and excludes the
internal planning / research-tracking infrastructure.

## What's interesting about this

1. **Reference-equivalence verification.** Most "X on the GPU/NPU"
   genomics demos compare against an approximate reference and report
   "good enough." For clinical / GLP / regulated workloads, "good
   enough" is not a defensible claim. `bionpu/verify/` exposes a
   documented public API that takes an NPU output and a reference
   output and proves byte-equality (or surfaces the exact divergence
   if not). The harness is reusable across both workloads and is the
   load-bearing contribution.

2. **Reproducible energy methodology.** All performance numbers in
   this repo include sustained-load energy in joules, measured via
   AMD RAPL counters with documented spec-bracketing assumptions.
   See [`docs/ENERGY_METHODOLOGY.md`](docs/ENERGY_METHODOLOGY.md).

3. **Reproducible on any chromosome / any pod5.** Benchmarks ship as
   shell scripts that take a target chromosome (CRISPR) or a pod5
   read set (basecalling) as input. Pre-computed results across
   chromosomes are v0.2 scope — see [`docs/STATUS.md`](docs/STATUS.md)
   for the v0.1-vs-v0.2 split.

## What ships in v0.1

Concrete inventory; every number here is reproducible from the
committed source on a clean clone.

| Component | Count / size | Notes |
|---|---|---|
| AIE2P kernels (CRISPR) | **6** | PAM filter (filter-early + filter-late variants), match (singletile / multitile / multitile-memtile), `crispr_net`, pktmerge variant. |
| AIE2P kernels (basecalling) | **12** | conv stem, linear projection, LSTM cell (7 precision / cascade / int8 / compressed variants), LSTM stack (3 variants). |
| Verify-harness tests | **18 / 18 passing** | 8 CRISPR + 10 basecalling. SHA-256 regression-pinned. Run: `pytest tests/`. |
| Cas-OFFinder canonical reference | **422 records, 30 877 canonical bytes** | SHA-256 `0765660ed275d7516937029c05e2b113f45d5e5fa94c17cbb4a9057d3a60432b` — pinned by `tests/test_verify_crispr.py::test_sha256_known_value_for_canonical_fixture`. |
| chr22 ten-guide fixture | **31 524 records, 2.30 MB canonical bytes** | SHA-256 `c2104e5c446b8d93889f5e9244e08f74745127f9319f4190b47cc6ed2661271d`. |
| Energy methodology spec | **480 lines** in `bench/POWER_DOMAINS.md` | Per-device rail spec (CPU / GPU / NPU) with includes, excludes, sampling rates, sources, known issues, cross-compare caveats. Mechanically lint-checked front-matter. |
| Calibration log | **408 lines** in `bench/energy/SANITY-LOG.md` | Per-host probe results — which counters are AVAILABLE / UNAVAILABLE on this host. Append, never overwrite. |

## Quick start

```sh
# Install (editable):
pip install -e .

# CRISPR scan on the CPU path (pure numpy; no NPU required):
bionpu scan --target chr22.fa --guides GUIDES.txt \
            --out /tmp/cpu_hits.tsv --device cpu

# Same scan on the NPU path. When the kernel artifacts are not built,
# the kernel's host-emulation fallback produces output byte-equal to
# the silicon path. When the artifacts are built (make NPU2=1 in the
# kernel directory), dispatches to AIE2P silicon.
bionpu scan --target chr22.fa --guides GUIDES.txt \
            --out /tmp/npu_hits.tsv --device npu \
            --verify reference/crispr/casoffinder-chr22-canonical.tsv
# → result EQUAL on byte-equality (exit 0) or DIVERGENT (exit 1).

# Or compare the two paths against each other to demonstrate they
# produce bit-identical output:
bionpu verify crispr /tmp/cpu_hits.tsv /tmp/npu_hits.tsv

# Basecall a pod5 file with byte-equality vs Dorado reference:
# (basecall is v0.2+ scope; the kernels are extracted but the
# streaming-pipeline driver is not yet wired)
bionpu basecall --pod5 reads.pod5 --verify
```

See [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for full reproduction
instructions including environment setup (XDNA driver, mlir-aie
build, Peano toolchain).

## Layout

```
src/bionpu/
├── kernels/             AIE2P MLIR-AIE kernels
│   ├── crispr/          CRISPR off-target scan (PAM filter, match, …)
│   └── basecalling/     Nanopore basecalling (LSTM cell, conv stem, …)
├── dispatch/            NPU silicon-dispatch + serialisation lock
├── verify/              Byte-equality harness (public API; +score policies)
├── scoring/             Off-target probability scorers (CPU/GPU; NPU TBD)
├── bench/               Energy + timing measurement (RAPL)
├── data/                Reference / fixture fetchers
└── quant/               Quantisation helpers (FP32 → bf16/int8)
docs/                    Methodology + reproduction docs
reference/               Canonical CPU-reference outputs (TSVs, FASTQs)
benchmarks/              Run scripts + pre-computed results
third_party/             Pinned upstream research repos (git submodules)
tests/                   pytest suite
```

## Status

This repository is the public release of work that's developed
internally; the public version ships the kernels, dispatch infrastructure,
verification harness, energy methodology, and the bench module. See
[`docs/STATUS.md`](docs/STATUS.md) for the per-subsystem state — what
runs end-to-end vs what's an extracted module that needs a driver
script to drive it on hardware.

The headline contribution as of v0.1 is the byte-equality harness
(`bionpu.verify`) — see [`src/bionpu/verify/README.md`](src/bionpu/verify/README.md).
The energy methodology that backs every `J/Mbp` number we publish is
in [`docs/ENERGY_METHODOLOGY.md`](docs/ENERGY_METHODOLOGY.md).

Pre-computed benchmark results across multiple chromosomes are
v0.2 scope and not yet committed; the methodology and reproduction
scripts are in place so other hosts can produce comparable numbers.

## License

GPL-3.0. See [LICENSE](LICENSE). All code in this repository — kernels,
dispatch, verification harness, CLI, benchmarks — is GPL-3.0.

`bionpu` consumes external projects as build / runtime dependencies;
those carry their own licenses and are not redistributed in this repo:

- [`mlir-aie`](https://github.com/Xilinx/mlir-aie) — Apache-2.0 with
  LLVM exception (the framework the kernels target).
- [`xdna-driver`](https://github.com/amd/xdna-driver) — Apache-2.0.
- [cas-offinder](https://github.com/snugel/cas-offinder) — BSD-2-Clause
  (CRISPR CPU reference).
- [Dorado](https://github.com/nanoporetech/dorado) — Oxford Nanopore
  Technologies Public License (basecalling CPU/GPU reference).

If you need a permissively-licensed version of any specific kernel for
non-GPL downstream use, reach out — we can re-issue under Apache-2.0 +
LLVM exception against the upstream `mlir-aie` PR provenance.
