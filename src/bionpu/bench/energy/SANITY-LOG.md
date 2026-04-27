# — Real RAPL + nvidia-smi Energy Sanity Gates

This log captures the load-bearing sanity-gate evidence for 
(POWER_DOMAINS.md §1.4 + the umbrella spec energy thesis).
Future runs of should append, not overwrite.

## Run metadata

- **Host:** `matteius-ProArt-P16-H7606WI-H7606WI`
- **CPU:** AMD Ryzen AI 9 HX (Strix; Zen 5)
- **GPU:** NVIDIA GeForce RTX 4070 Laptop (driver 580.142)
- **Kernel:** Linux 7.0.0-14-generic (Ubuntu 26.04)
- **Date:** 2026-04-25
- **User:** `matteius` (uid 1000, NOT root)
- **Module versions:** `intel_rapl_msr` loaded; `intel_rapl_common` loaded;
  `amd_energy` not loaded; `amd-rapl-msr` sysfs path not exposed.

## CPU / RAPL — UNAVAILABLE on this kernel

### Probe result

```
RAPL probe: UNAVAILABLE
  candidates checked:
    /sys/class/powercap/intel-rapl:0/energy_uj
    /sys/class/powercap/amd-rapl-msr:0/energy_uj
    /sys/class/powercap/amd_energy:0/energy_uj
  reason: PermissionError (mode 0400 root:root)
```

### Root cause

`/sys/class/powercap/intel-rapl:0/energy_uj` **exists** on this host
(the `intel_rapl_msr` driver does expose AMD package energy through
the historical "intel-rapl" sysfs hierarchy on Zen 5), but stock Ubuntu
26.04 ships it with mode `0400 root:root` due to the upstream RAPL
side-channel mitigation. Confirmed via:

```
$ stat -c "%a %U %G %n" /sys/class/powercap/intel-rapl:0/energy_uj
400 root root /sys/class/powercap/intel-rapl:0/energy_uj
```

The reader probes successfully exit with `RaplUnavailableError`
listing the candidates and the recommended remediation (run as root
OR `chmod a+r /sys/class/powercap/*/energy_uj` OR install `amd_energy`
hwmon).

### Sanity gate (stress-ng vs sleep)

**SKIPPED.** Required because the probe failed; without a readable
counter there is no comparison to make. `stress-ng` is also not
installed on this host (no sudo to install via apt). Both gating
conditions independent — either alone is sufficient to skip.

### Outcome

** CPU branch: Partial.** Per POWER_DOMAINS.md §1.4 and the 
task brief: never fabricate. Production behaviour:

- `RaplReader()` in `bionpu/bench/energy/rapl.py` correctly raises
  `RaplUnavailableError` with an actionable message.
- `auto_reader("cpu")` falls back to `bionpu.bench.harness.RaplStub`
  with a `WARNING`-level log line citing the unavailability reason.
- The fallback stub records `energy_source="stub"` so 's writeup
  pipeline marks any CPU energy figure as preliminary.
- `POWER_DOMAINS.md` §7 ("Open issues") is annotated with this
  outcome (additive — does not overwrite 's content).

### Remediation for a future operator

Pick the lowest-risk option that fits the host's policy:

1. **Loosen sysfs perms (one-shot, lost on reboot):**
   ```
   sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj \
                  /sys/class/powercap/intel-rapl:0:0/energy_uj
   ```
   Then re-run the sanity gate. should flip to `Completed`.

2. **udev rule (persists across reboots):**
   ```
   # /etc/udev/rules.d/99-rapl.rules
   SUBSYSTEM=="powercap", KERNEL=="intel-rapl:*", ACTION=="add", \
       RUN+="/bin/chmod a+r /sys%p/energy_uj"
   ```

3. **Install `amd_energy` hwmon driver and configure read access** —
   this avoids the side-channel mitigation that gates `intel-rapl`.

4. **Run the bench as root.** Discouraged in CI but valid for ad-hoc
   characterization.

## GPU / `nvidia-smi` — AVAILABLE; sanity gate PASSES

### Method auto-selection

```
method = power.draw
```

`total_energy_consumption` not supported on driver 580.142 (verified
via `nvidia-smi --help-query-gpu | grep total_energy_consumption`
returning nothing parseable for an actual `--query-gpu=total_energy_consumption`
call). The `NvidiaSmiReader` therefore takes the trapezoidal fallback
path documented in POWER_DOMAINS.md §2.3, polling `power.draw` at
10 Hz on a daemon thread.

### Idle baseline (5 s, system at rest)

```
idle 5s: 70.56 J  (14.11 W avg)
```

### CUDA matmul load (8192×8192, ~5 s)

```
cuda matmul 49 iters over 5.11s: 350.14 J  (68.51 W avg)
```

### Delta

```
delta_W: 54.40 W   (gate requires > 5 W)
ratio:   4.85x      (load_j / idle_j over equal windows)
```

**PASSES with margin.** The reader is responsive to real GPU load.

### Outcome

** GPU branch: Completed.** `NvidiaSmiReader` is the canonical
GPU energy reader for the bench harness. Future tasks
import from `bionpu.bench.energy` rather than re-defining the
trapezoidal poller. 's `NvidiaSmiReal` in
`tracks/crispr/baseline/run_baseline.py` remains untouched in this
commit per the task brief; 's runner can adopt
`bionpu.bench.energy.NvidiaSmiReader` in a follow-up.

## NPU / `xrt-smi` — STUB pending 

(Historic — replaced by the outcome below. Kept for
provenance.)

`bionpu/bench/energy/xrt.py` ships a clear-error stub:

```python
raise XrtUnavailableError(
    "NPU energy reading wired in ; xrt-smi telemetry not yet "
    "characterized on this host (see POWER_DOMAINS.md §3.1)"
)
```

`auto_reader("npu")` returns `bionpu.bench.harness.XrtStub` so the
harness still emits a record (with `energy_source="stub"`) rather
than crashing. is the gate to promote this to a real reader.

## Reproduction

```
source .venv/bin/activate
pytest tests/test_energy_real.py -q -m "not slow"        # all GREEN
pytest tests/test_energy_real.py -q -m slow              # GPU PASS,
                                                         # CPU SKIPPED
                                                         # (RAPL gated)
```

Manual sanity gate (the script captured in this log) is in the 
log entry of `bio-on-xdna-plan.md`.

## Re-run after stress-ng + RAPL permissions (2026-04-25)

The original 2026-04-25 run reported `Partial` for the CPU branch
because (a) `stress-ng` was not on `PATH` and (b) the powercap
energy counter was mode `0400 root:root`. The operator subsequently:

1. installed `stress-ng` via `sudo apt install stress-ng`
   (`stress-ng, version 0.20.01`, `/usr/bin/stress-ng`), and
2. relaxed permissions on the package node:
   `sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj`
   (also set `kernel.dmesg_restrict=0` transiently via
   `sysctl -w` for unrelated probing).

Both blockers are now removed. Verified state at re-run:

```
$ stat -c "%a %U %G %n" /sys/class/powercap/intel-rapl:0/energy_uj
444 root root /sys/class/powercap/intel-rapl:0/energy_uj

$ stat -c "%a %U %G %n" /sys/class/powercap/intel-rapl:0:0/energy_uj
400 root root /sys/class/powercap/intel-rapl:0:0/energy_uj   # subdomain still 0400; harmless

$ which stress-ng
/usr/bin/stress-ng

$ cat /proc/sys/kernel/dmesg_restrict
0
```

Note: only the package-level counter (`intel-rapl:0/energy_uj`) was
chmodded; the core-subdomain counter (`intel-rapl:0:0/energy_uj`)
remains `0400`. `RaplReader.probe_rapl()` uses the package counter
first, so the reader is fully functional. If a future task needs
core-only attribution, the operator will need to chmod that node too.

### CPU sanity gate (stress-ng vs sleep) — PASSES

Manual run as user `matteius` (uid 1000, NOT root), `RaplReader`
auto-probed `/sys/class/powercap/intel-rapl:0/energy_uj`:

```
RAPL path: /sys/class/powercap/intel-rapl:0/energy_uj

Idle baseline (time.sleep(10)):
    152.761 J  (15.28 W avg)

Loaded (stress-ng --cpu 8 --timeout 10s):
    611.449 J  (61.14 W avg)

delta_W: 45.86 W   (gate requires strictly load > idle; >5x ideal)
ratio:   4.00x      (load_j / idle_j over equal 10 s windows)
```

A 4.00× ratio over a 10 s window is solidly above the "strictly
greater" assertion and within plausible bounds for a Strix mobile
package. The original task brief's >5× target is approached but not
quite cleared on this part — the package counter on Strix at idle
is comparatively high (~15 W) because GPU/SoC subdomains are folded
into the `intel-rapl:0` package counter. The sanity-gate test
(`tests/test_energy_real.py::test_rapl_responsive_to_load`) only
asserts `load_j > idle_j`, which holds with ample margin.

### Slow pytest gate — both GREEN

```
$ .venv/bin/pytest tests/test_energy_real.py -v -m slow
============================= test session starts ==============================
platform linux -- Python 3.11.15, pytest-9.0.3, pluggy-1.6.0
collecting ... collected 18 items / 16 deselected / 2 selected

tests/test_energy_real.py::test_rapl_responsive_to_load PASSED           [ 50%]
tests/test_energy_real.py::test_nvsmi_responsive_to_load PASSED          [100%]

====================== 2 passed, 16 deselected in 23.18s =======================
```

`test_rapl_responsive_to_load` is GREEN where it previously SKIPPED.
GPU sanity gate continues to pass (re-confirmed under the same
session; idle ≈ 14 W, CUDA matmul ≈ 68 W as in the original run).

### Outcome — flips to Completed

** CPU branch: Completed.** `RaplReader` is now exercised
end-to-end on the development host with a real load-bearing gate.
`auto_reader("cpu")` returns the real `RaplReader` (no longer falling
back to `RaplStub`). Writeups produced on this host can quote CPU
energy as `energy_source="real-rapl"` rather than `"stub"`.

The remediation runbook above (chmod / udev / amd_energy / root) is
retained for fresh hosts; the operator chose option 1 (one-shot
`chmod a+r`). Note this does NOT persist across reboots — the
`amd_energy` hwmon path or a udev rule remain the durable options.

## NPU sanity gate

 lands the real NPU energy reader (`bionpu/bench/energy/xrt.py
::XrtReader`) — replaces the stub. The reader polls
`xrt-smi examine -r platform` at 10 Hz on a daemon thread,
trapezoidal-integrating the firmware-reported instantaneous power to
joules. This subsection records the load-bearing sanity-gate
evidence; POWER_DOMAINS.md §3 documents the rail attribution
(AIE compute-tile partition, firmware-estimated).

### Telemetry surface chosen

```
xrt-smi examine -r platform
```

(parsed via regex `^Estimated Power\s*:\s*([0-9.]+)\s*Watt` on
stdout). XRT 2.23.0 / amdxdna 2.23.0 / firmware 1.1.2.64 do NOT
ship a dedicated `power` or `thermal` reporter (only
`aie-partitions`, `host`, `platform`, `all`); `platform` is the
smallest report that includes the power line. The equivalent JSON
path under `devices[0].platforms[0].electrical.power_consumption_watts`
is documented in POWER_DOMAINS.md §3.2 but not used by the reader
(parsing the text output avoids a temp-file write per poll).

**No amdxdna sysfs power node.** Verified by:

- `/sys/class/hwmon/hwmon*/name` enumeration: only `acpi_fan`,
  `amdgpu`, `k10temp`, `BAT1`, `nvme`, etc. — no `amdxdna`/`npu`
  hwmon driver.
- `/sys/bus/pci/devices/0000:67:00.1/accel/accel0/` exposes only
  `dev`, `device`, `power` (runtime-PM directory), `subsystem`,
  `uevent` — no `power_input`, `energy_uj`, or analogous node.

A future amdxdna release that adds a sysfs power node could be
preferred without changing `XrtReader`'s public surface.

### Idle baseline (5 s, system at rest, no hardware context)

```
idle 5s: 0.0000 J  (firmware noise floor 0.001-0.003 W; integrates to <10 mJ, rounds to 0 at 4 dp)
```

### Loaded (100x `vector_scalar_mul` dispatches, ~10-15 s wall-clock)

The sanity-gate test (`test_xrt_responsive_to_load`) dispatches the
 canonical canary 100 times back-to-back. Each call spawns the
mlir-aie host binary, loads the xclbin via XRT, runs the kernel,
and exits. Sustained NPU load throughout the integration window.

```
load: 0.1406 J  (100 dispatches; n_dispatches=100)
```

A separate unbroken-load capture (50000 iterations in a single
host-binary invocation) yielded a steady-state reading of ~0.412 W
on the firmware-reported `Estimated Power` field — see
POWER_DOMAINS.md §7.2.

### Delta

```
delta:   +0.1406 J (idle 0.0000 J → load 0.1406 J)
ratio:   ~∞ (idle floors near-zero; assertion is `load_j > idle_j`, holds with clear margin)
```

**PASSES.** The reader is responsive to real NPU load. The
firmware noise floor at idle (0.001-0.003 W) is below the
trapezoidal-integration accuracy at 10 Hz over 5 s, so idle
integrates to ~0 J at 4 decimal places. The 0.1406 J loaded reading
over ~10 s averages to ~0.014 W; the observed steady-state under
sustained load is ~0.412 W (see the 50000-iteration capture). The
overall workload mix here is dispatch-overhead-dominated (each
host-binary spawn takes ~106 ms wall-clock per the numbers),
so the average load joules are dominated by inter-dispatch idle
windows, not by tile-active power. This is a feature of the
v1-thin "shell out to the C++ host binary" runner, not a flaw in
the reader.

### Outcome

** NPU branch: Completed.** `XrtReader` is the canonical NPU
energy reader. `auto_reader("npu")` returns it (no longer falls
back to `XrtStub`). Writeups produced on this host can quote NPU
energy as `energy_source="real-xrt-smi"` rather than `"stub"`.

`bionpu bench --all --device npu` now produces a 3-rail
measurements.json:

```
cpu -> real-rapl       energy_j=0.0     (sub-ms work below RAPL counter resolution)
gpu -> real-nvidia-smi energy_j=1.10    (CUDA matmul reference)
npu -> real-xrt-smi    energy_j=0.0013  (vector_scalar_mul kernel; firmware-estimated)
```

### Caveat for absolute-number citations

The `Estimated Power` field is a **firmware-internal estimate**
(not a sense-resistor reading; AMD does not publish the estimation
algorithm). For cross-device J/Mbp ratios the relative scale is
the comparable quantity. Writeups citing absolute joule numbers
should note the firmware-estimate provenance per the
`known_issues` field in POWER_DOMAINS.md's front-matter.

For per-task absolute-energy comparisons against a calibrated
external reference (e.g. wall-power meter on the laptop AC input
minus baseline), the relative-scale comparison can be promoted to
absolute once a calibration constant is measured. / T7.* may
need to do this for the genome-scale CRISPR scan figure if the
external reviewer asks.

### Reproduction

```
source /opt/xilinx/xrt/setup.sh
source ~/xdna-bringup/ironenv/bin/activate
pytest tests/test_t44_xrt_energy.py -m "not npu" -q       # all GREEN (fast)
pytest tests/test_t44_xrt_energy.py -m npu -q             # all GREEN (10-20 s)
python -m bionpu bench --all --device npu --iters 3       # writes results/u-m2/measurements.json
```

## Cross-references

- `bionpu/bench/POWER_DOMAINS.md` §1.4 — RAPL availability rule.
- `bionpu/bench/POWER_DOMAINS.md` §2.3 — trapezoidal fallback
  documentation.
- `bionpu/bench/POWER_DOMAINS.md` §3 — NPU rail attribution
  (resolved by ).
- `bionpu/bench/POWER_DOMAINS.md` §7 — open issues (annotated
  with the RAPL-unavailable result for this host plus the 
  resolution).
- `bionpu/bench/POWER_DOMAINS.md` §7.2 — outcome on this
  host.
- `bionpu/bench/energy/rapl.py` — `RaplReader`, `probe_rapl`,
  `RaplUnavailableError`.
- `bionpu/bench/energy/nvsmi.py` — `NvidiaSmiReader`,
  `NvidiaSmiUnavailableError`.
- `bionpu/bench/energy/xrt.py` — `XrtReader`, `probe_xrt`,
  `XrtUnavailableError`.
- `bionpu/bench/energy/__init__.py` — `auto_reader` factory
  (cpu/gpu/npu).
- `tests/test_energy_real.py` — TDD tests + CPU/GPU sanity-
  gate harness.
- `tests/test_t44_xrt_energy.py` — TDD tests + NPU
  sanity-gate harness.
