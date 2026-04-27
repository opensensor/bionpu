# Reproducing `bionpu` results

> Status: shell — full reproduction notes will land during the v0.1
> extraction. This document tracks what's required end-to-end so a
> reader can run the benchmarks on their own machine.

## Hardware

- AMD Ryzen AI 9 HX (Strix family) or other AIE2P-equipped silicon.
- Linux (kernel ≥ 6.10 with `amdxdna` available).

## Software prerequisites

- `xdna-driver` built and `amdxdna.ko` loaded.
- XRT (Xilinx Runtime) installed (`/opt/xilinx/xrt`).
- `mlir-aie` built; `aiecc` on `$PATH`.
- Peano (LLVM-AIE) installed; `$PEANO_INSTALL_DIR` set.
- Python ≥ 3.11.

See `bionpu`'s upstream documentation in
[opensensor/genetics](https://github.com/opensensor/genetics) for the
NPU bring-up steps if the above are not yet on your system.

## Install bionpu

```sh
git clone https://github.com/opensensor/bionpu.git
cd bionpu
pip install -e ".[test]"
```

## Run a single benchmark

CRISPR off-target scan against chr22, with byte-equality vs cas-offinder:

```sh
benchmarks/crispr/run_chr.sh chr22
```

Basecalling against a small pod5 fixture, with byte-equality vs Dorado:

```sh
benchmarks/basecalling/run_pod5.sh reference/basecalling/smoke.pod5
```

Pre-computed results for chr1, chr19, chr22 are checked into
`benchmarks/results/`. To regenerate, run the same scripts and they
will overwrite the JSON snapshots in place.
