# Reproducing `bionpu` results

End-to-end reproduction recipe for everything in `bionpu` that's
runnable today. v0.1 ships the byte-equality harness end-to-end; the
full scan / basecall pipelines are v0.2 scope (see
[`STATUS.md`](STATUS.md)) — this document covers what works now and
documents the v0.2 driver shape so you can drive the kernels manually
in the meantime.

## 1. Hardware

- AMD Ryzen AI 9 HX (Strix family) or other AIE2P-equipped silicon
  with the `amdxdna` accelerator exposed at `/dev/accel/accel0`.
- Linux kernel ≥ 6.10 with `amdxdna.ko` loaded.

## 2. Software prerequisites

- **`xdna-driver`** built and `amdxdna.ko` loaded. Verify:
  ```sh
  lsmod | grep amdxdna
  ls -l /dev/accel/accel0
  ```
- **XRT** installed at `/opt/xilinx/xrt`:
  ```sh
  source /opt/xilinx/xrt/setup.sh
  xrt-smi examine    # must list the NPU device
  ```
- **`mlir-aie`** built; `aiecc` on `$PATH`. The recommended `mlir-aie`
  is the wheel-built `ironenv` — see the [opensensor/genetics
  bring-up
  guide](https://github.com/opensensor/genetics/blob/main/docs/xdna-driver-build.md)
  for the canonical setup.
- **Peano (LLVM-AIE)** installed; `$PEANO_INSTALL_DIR` points at the
  install tree.
- **Python ≥ 3.11**.

## 3. Install bionpu

```sh
git clone https://github.com/opensensor/bionpu.git
cd bionpu
pip install -e ".[test]"
```

This installs the `bionpu` CLI on `$PATH` and exposes the
`bionpu.{verify,kernels,dispatch,bench,data,quant}` modules to your
Python interpreter.

## 4. Reproduce the byte-equality smoke check

This is the simplest end-to-end demonstration of the verify harness.
No NPU required — just the committed reference data.

```sh
# Compare the canonical reference TSV against itself; expect EQUAL.
bionpu verify crispr \
    reference/crispr/casoffinder-canonical.tsv \
    reference/crispr/casoffinder-canonical.tsv
# Expected: result EQUAL, 422 records, matching SHA-256s. Exit code 0.

# Negative control: mutate the input and expect DIVERGENT.
sed -i.bak 's/^chr/chr_DIRTY_/' /tmp/dirty.tsv 2>/dev/null
sed 's/^chr/chr_DIRTY_/' reference/crispr/casoffinder-canonical.tsv > /tmp/dirty.tsv
bionpu verify crispr /tmp/dirty.tsv reference/crispr/casoffinder-canonical.tsv
# Expected: result DIVERGENT, exit code 1, first divergence reported.
```

## 5. Reproduce a kernel build (any one)

Pick any kernel under `src/bionpu/kernels/`. For the CRISPR PAM filter:

```sh
cd src/bionpu/kernels/crispr/pam_filter
export MLIR_AIE_DIR=<path/to/mlir-aie>
export PEANO_INSTALL_DIR=<path/to/llvm-aie>
make NPU2=1
# Expected: build/final.xclbin + build/insts.bin produced.
```

Each kernel directory ships `MANIFEST.md` describing the inputs,
outputs, expected on-tile placement, and any kernel-specific
`make` flags.

## 6. Run a kernel against silicon (manual driver, v0.2-scope)

The v0.2 `bionpu scan` / `bionpu basecall` drivers are not yet wired,
so end-to-end pipeline runs go through the per-kernel host runner:

```sh
cd src/bionpu/kernels/crispr/pam_filter
# After 'make NPU2=1':
./host_runner --xclbin build/final.xclbin \
              --insts  build/insts.bin \
              --in     <input.bin> \
              --out    /tmp/npu_hits.tsv
bionpu verify crispr /tmp/npu_hits.tsv reference/crispr/casoffinder-chr22-10guides.tsv
```

The `host_runner` argv shape varies per kernel — see each kernel's
`MANIFEST.md`.

## 7. Energy methodology

For the per-device energy figures the bench harness produces, read
[`ENERGY_METHODOLOGY.md`](ENERGY_METHODOLOGY.md) first. The TL;DR:

- CPU rail = AMD RAPL package counter, package-only, no DRAM.
- GPU rail = `nvidia-smi` total board (compute + memory + VRMs).
- NPU rail = `xrt-smi` AIE-partition firmware-internal estimate.

A figure caption that compares any two of these without listing the
includes / excludes is not an honest comparison.

## 8. Sanity-log discipline

Calibration evidence — which counters are AVAILABLE / UNAVAILABLE on
your host, what the probe path returned, what the resolution path
was — is recorded in
[`src/bionpu/bench/energy/SANITY-LOG.md`](../src/bionpu/bench/energy/SANITY-LOG.md).
**Append, never overwrite.** A run on a different host (different
kernel / driver / governor) is a different measurement; record it
as a new entry rather than editing in place.

## What's deferred to v0.2

- `bionpu scan` and `bionpu basecall` end-to-end drivers (the kernels
  are migrated and buildable; what's missing is the Python that
  drives them as a single CLI invocation).
- Pre-computed `benchmarks/results/{crispr,basecalling}/*.json`
  snapshots for chr1 / chr19 / chr22 / a representative pod5.
- A tagged `v0.1` GitHub release with the headline numbers.
