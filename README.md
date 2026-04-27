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
   read set (basecalling) as input. Pre-computed results for chr1,
   chr19, chr22 are in `benchmarks/results/`.

## Quick start

```sh
# Install (editable):
pip install -e .

# CRISPR scan against chr22 with byte-equality vs cas-offinder:
bionpu scan --target chr22 --guides GUIDES.txt --verify

# Basecall a pod5 file with byte-equality vs Dorado reference:
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
├── verify/              Byte-equality harness (public API)
├── bench/               Energy + timing measurement (RAPL)
├── data/                Reference / fixture fetchers
└── quant/               Quantisation helpers (FP32 → bf16/int8)
docs/                   Methodology + reproduction docs
reference/              Canonical CPU-reference outputs (TSVs, FASTQs)
benchmarks/             Run scripts + pre-computed results
tests/                  pytest suite
```

## Status

This is the v0.1 release of `bionpu` as a public asset. Active
development happens upstream in opensensor's wider AMD-XDNA stack;
this repo is a reproducible snapshot suitable for evaluation,
verification, and external collaboration.

See the [v0.1 release notes](https://github.com/opensensor/bionpu/releases/tag/v0.1)
for the headline benchmark numbers.

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
