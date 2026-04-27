# Energy methodology

This document is the public-facing methodology for the energy figures
reported in `benchmarks/results/`. It exists because cross-device
energy comparisons (CPU vs GPU vs NPU joules-per-Mbp / joules-per-scan)
are easy to misuse — every device's "energy" is a different rail with
different includes, different sampling rates, and different known
instrumentation gaps. We document those explicitly so a reader can
decide whether the comparison is honest.

## TL;DR

| Device | Counter source | Includes | Excludes | Sampling |
|---|---|---|---|---|
| **CPU** | `/sys/class/powercap/{intel-rapl,amd-rapl-msr}:0/energy_uj` (RAPL) | All P-cores + E-cores + L3 / uncore on Zen 5 package | DRAM (no separate AMD RAPL DRAM domain), discrete GPU, NPU subdomain, platform IO | ≥10 Hz, monotonic counter integrated start-to-end |
| **GPU** | `nvidia-smi --query-gpu=power.draw,total_energy_consumption` | Compute cores + GDDR/HBM memory + PCIe interface (board side) + VRMs | Host CPU, host DRAM, NPU | ~1 Hz (driver-reported); prefer driver-integrated `total_energy_consumption` over trapezoidal-integrated `power.draw` |
| **NPU** | `xrt-smi examine -r platform` (firmware-internal estimate) | AIE compute tiles in the active hardware-context partition | Host SoC package (CPU; on the RAPL rail), host DRAM, Radeon iGPU on the same package, platform IO outside the AIE partition | 10 Hz (capped by ~40 ms `xrt-smi` invocation cost); trapezoidal-integrated to joules |

A figure caption that compares any two of these without listing the
includes / excludes is not honest enough to publish.

## Reference documents

The full methodology lives in three places in this repo:

1. **[`src/bionpu/bench/POWER_DOMAINS.md`](../src/bionpu/bench/POWER_DOMAINS.md)**
   — exhaustive per-device specification: rail name, target hardware,
   includes / excludes, sampling rate, source path, fallback source,
   known issues, cross-compare caveats. Front-matter is mechanically
   lint-checked so every device entry is fully populated.

2. **[`src/bionpu/bench/energy/SANITY-LOG.md`](../src/bionpu/bench/energy/SANITY-LOG.md)**
   — the calibration log. Records the host system the numbers were
   measured on, the kernel + module versions, the probe results
   (which counters are AVAILABLE / UNAVAILABLE on this host), and
   the resolution paths for the UNAVAILABLE cases. Future calibration
   runs append; never overwrite.

3. **[`src/bionpu/bench/UNITS.md`](../src/bionpu/bench/UNITS.md)** —
   units convention (J vs Wh vs J/Mbp), measurement passport schema,
   and the rules for combining same-rail / cross-rail figures.

## Sustained-load measurement

Every benchmark in `benchmarks/` measures energy across three
windows, in this order:

1. **Pre-warmup** (`pre_warmup_seconds`, default 10 s): host runs the
   workload at full duty cycle to bring caches, governors, NPU
   firmware, and GPU clocks to their steady-state. Energy in this
   window is **not** counted.

2. **Measurement** (`measurement_seconds`, default 30 s): the actual
   integration window. The energy counter is sampled at the start
   boundary, sampled again at the end boundary, and sampled at
   least once mid-window to detect counter wraparound.

3. **Drift-detection** (`drift_seconds`, default 5 s): a
   final-window sample taken `drift_seconds` after the measurement
   window ends. If the per-second power in the drift window deviates
   from the measurement window by > drift threshold (default 5 %),
   the measurement passport flags the run as `drift_detected: true`
   and the published number is the measurement-window value with a
   drift-warning annotation.

This three-phase shape distinguishes steady-state energy from
cold-start spikes; almost all "the NPU uses X joules" claims that
report a single-shot wall-time figure are conflating the warmup
transient with the steady-state, sometimes by a factor of 2-3×.

## Spec-bracketing assumptions

The published energy numbers are reported alongside the
manufacturer-spec TDP envelope of the device's silicon, so a reader
can check whether the measurement falls in a plausible range:

- **CPU** — Ryzen AI 9 HX nominal 28-54 W TDP envelope.
- **GPU** — per-board TGP from the OEM, recorded per run in the
  measurement passport.
- **NPU** — AIE2P partition at sustained load typically falls in
  the 1.5-3.5 W range; published measurements outside this band
  are flagged as out-of-spec and require a calibration entry in
  `SANITY-LOG.md` before publication.

The spec envelope is a sanity gate, not a target. A measurement that
falls in-band is not automatically valid; a measurement that falls
out-of-band is not automatically wrong (silicon binning,
firmware-state, or governor changes can move the steady-state
envelope by ±20 %). The bracket is published so readers can
challenge the number.

## When a counter is UNAVAILABLE

Per the rules in `POWER_DOMAINS.md`, if any counter probe fails
(permission denied, sysfs path missing, driver too old) the harness
emits a measurements record with that device's energy field set to
`null` and a `reason_unavailable` string. The harness MUST NEVER
fabricate a reading. A run with an UNAVAILABLE counter still records
wall-time; the published comparison just drops that device from the
energy column with a footnote pointing at the sanity-log entry that
explains why.

## Reproducibility envelope

The numbers in `benchmarks/results/` are valid for the host
configuration recorded at the head of `SANITY-LOG.md`. A different
host (different kernel, different driver, different governor) is a
different measurement. We do not ship "expected energy" thresholds
that other hosts must hit; we ship the reproducible measurement
**method** so other hosts can produce their own numbers.
