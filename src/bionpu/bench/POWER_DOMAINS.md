---
# Structured front-matter for mechanical lint checks. (writeup pipeline)
# auto-cites this file for every energy figure caption; the lint asserts that
# every device entry below has includes/excludes/sampling_rate/source/known_issues
# populated. NPU rail attribution stays "TBD pending " until characterized
# (per task brief).
spec_version: "1.0.0"
spec_purpose: "Document exactly what each energy reading covers, per device, so cross-domain comparisons are honest. Companion to UNITS.md."
generated: "2026-04-25"
owner: ""
consumed_by: ["", "", "", "", ""]
required_caption_citation: true
devices:
  cpu:
    rail_name: "CPU package (RAPL)"
    target_hardware: "AMD Ryzen AI 9 HX (Strix, Zen 5)"
    includes: ["all P-cores", "all E-cores (if present)", "L3 cache / uncore on Zen 5 package"]
    excludes: ["DRAM (no separate amd-rapl-msr DRAM domain on Strix as of writing)", "discrete GPU", "NPU subdomain", "platform IO / PCIe rails outside the SoC package"]
    sampling_rate_hz: ">=10 (energy counter is monotonic; harness samples at start and end of integration window plus one mid-run sample for monotonicity check)"
    source: "/sys/class/powercap/amd-rapl-msr:0/energy_uj (or /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj on systems where amd-rapl exposes the same hierarchy)"
    fallback_source: "/sys/class/hwmon/hwmon*/energy*_input from amd_energy if amd-rapl-msr unavailable"
    units_native: "microjoules (uJ); harness divides by 1e6 for joules"
    counter_wraparound: "64-bit on Zen 5; harness still records both endpoints to detect wraparound for safety"
    known_issues: ["AMD RAPL on Strix may require a kernel patch to expose energy counters; if the sysfs path returns 0 / -EIO / does not exist, must mark CPU energy as UNAVAILABLE rather than report nonsense (risk row 'apples-to-oranges')", "no DRAM subdomain on AMD as of mainline 6.x; energy figures exclude DRAM contribution"]
    cross_compare_caveat: "CPU rail is package-only; does not include DRAM, so basecalling J/Mbp on CPU underestimates total system energy compared with a 'wall-power' measurement."
  gpu:
    rail_name: "GPU board (whole-board)"
    target_hardware: "NVIDIA discrete GPU on the same laptop (model recorded per run)"
    includes: ["compute cores", "GDDR/HBM memory", "PCIe interface on the board side", "VRMs on the board"]
    excludes: ["host CPU", "host DRAM", "NPU"]
    sampling_rate_hz: "~1 (driver-reported; nvidia-smi typical update rate ~1 Hz)"
    source: "nvidia-smi --query-gpu=power.draw,total_energy_consumption --format=csv,noheader,nounits"
    units_native: "power.draw=W; total_energy_consumption=mJ on supported drivers"
    integration_method: "Prefer total_energy_consumption (driver-integrated counter, monotonic). If unsupported, fall back to trapezoidal integration of power.draw at the polling rate; record the method used in the measurement passport."
    known_issues: ["total_energy_consumption requires driver >= 535 on most GeForce; harness probes capability and falls back to trapezoidal integration with an explicit annotation", "1 Hz sampling under-resolves bursty workloads; integrated counter is preferred", "thermal throttling can change power.draw mid-run; harness records min/max/mean over the window for diagnostics"]
    cross_compare_caveat: "GPU rail is whole-board including memory, so it is NOT comparable to the CPU package-only rail without explicit caveat."
  npu:
    rail_name: "NPU AIE-partition compute-tile rail (xrt-smi firmware estimate)"
    target_hardware: "AMD XDNA 2 / AIE2P on Ryzen AI 9 HX (Strix), BDF 0000:67:00.1, 6x8 = 48 tiles"
    includes: ["AIE compute tiles in the active hardware-context partition (firmware-internal estimate; not a sense-resistor reading)"]
    excludes: ["host SoC package (CPU; covered by RAPL rail)", "host DRAM", "Radeon iGPU on the same package", "platform IO / PCIe lanes outside the AIE partition"]
    sampling_rate_hz: "10 (XrtReader default; xrt-smi invocation costs ~40 ms wall-clock per call which caps polling at ~25 Hz before xrt-smi overhead dominates)"
    source: "xrt-smi examine -r platform (parsed via regex `^Estimated Power\\s*:\\s*([0-9.]+)\\s*Watt` on the text-format output; identical figure available in JSON at devices[0].platforms[0].electrical.power_consumption_watts)"
    fallback_source: "no driver-integrated cumulative-energy counter on amdxdna 2.23.0; no amdxdna sysfs hwmon power node — verified by enumerating /sys/class/hwmon/hwmon*/name and the PCI accel0 device. Trapezoidal integration of xrt-smi instantaneous power is the only path."
    units_native: "watts (instantaneous power; harness trapezoidal-integrates over [start, stop) window to joules)"
    known_issues: ["Estimated Power is a firmware-internal estimate, NOT a sense-resistor reading; AMD does not publish the estimation algorithm. Calibration against a system-wall-power measurement (e.g. hardware power-meter on the laptop AC input minus baseline) is recommended for any externally-cited absolute number, but for cross-device J/Mbp comparison the relative scale is the comparable quantity.", "When no hardware context is loaded, the firmware floors the reading near 0 W (typically 0.001-0.003 W noise floor). Idle-baseline measurements taken with no hardware context loaded therefore under-represent leakage power; sanity-gate idle baselines should keep an XRT context open if absolute idle leakage matters.", "Reading is per-AIE-partition; on a host that runs multiple concurrent NPU workloads (currently impossible — amdxdna 2.23.0 doesn't ship multi-context scheduling on Strix) the per-partition figures would need to be summed."]
    cross_compare_caveat: "NPU rail is AIE compute tiles only; does NOT include host SoC (CPU) or DRAM. Comparable to GPU compute-cores energy MINUS GPU on-board memory (which the GPU reading includes); comparable to CPU package energy MINUS CPU package overhead (cores+uncore are in the CPU figure but not in the NPU figure). Cross-domain captions must list these per-rail boundaries."
comparison_rules:
  - "Cross-device energy comparison is only valid when each domain's includes/excludes are stated in the figure caption."
  - "If any of the three readers is UNAVAILABLE on the host, the harness MUST emit measurements with that device's energy field set to null and a `reason_unavailable` string, NEVER a fabricated number."
  - "Throughput numbers are comparable across devices without a power caveat (they measure work, not power), but their J/work derivatives are NOT — derivatives require this file's caveats."
caption_template: "Energy is reported per `bionpu/bench/POWER_DOMAINS.md` v{spec_version}. CPU = {cpu_rail_name} ({cpu_includes_short}). GPU = {gpu_rail_name} ({gpu_includes_short}). NPU = {npu_rail_name} ({npu_includes_short}). Cross-domain comparisons inherit the caveats listed in that document; in particular {npu_preliminary_caveat_if_applicable}."
---

# POWER_DOMAINS — What Each Energy Reading Covers

This file is the reference for every cross-device energy comparison in the
repo. (writeup pipeline) auto-cites this file in every energy figure
caption; the citation is a hard requirement, not a recommendation. Without
this document, the umbrella spec energy claims are unfalsifiable.

Companion: [`UNITS.md`](./UNITS.md) for the metric formulas themselves.

---

## Side-by-side: CPU vs GPU vs NPU

| Field | CPU (RAPL on Zen 5 / Strix) | GPU (`nvidia-smi`) | NPU (`xrt-smi`) |
|-------------------|----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| Rail | Package energy | Whole-board | AIE compute-tile partition rail (§3, characterized in ) |
| Cores | yes (P-cores + E-cores) | yes (compute cores) | yes (AIE compute tiles in active partition) |
| Uncore / fabric | yes (Zen 5 package includes L3 + uncore) | n/a (board-level reading) | no (host SoC fabric is on a different rail; reading is per-AIE-partition) |
| On-device memory | **excludes** DRAM (no AMD DRAM domain in mainline) | **includes** GDDR/HBM | partial — AIE tile-local memory is on-die in the partition; system DRAM/host pools not included |
| IO / PCIe | excludes platform IO outside the package | includes board-side PCIe; excludes host-side | excludes (AIE partition only) |
| Sampling rate | ≥10 Hz (counter is monotonic; harness reads start + mid + end) | ~1 Hz driver-reported (10 Hz harness poll fallback) | 10 Hz harness poll (xrt-smi invocation ~40 ms; trapezoidal integration) |
| Sysfs / cmd path | `/sys/class/powercap/amd-rapl-msr:0/energy_uj` | `nvidia-smi --query-gpu=power.draw,total_energy_consumption --format=csv` | `xrt-smi examine -r platform` (no amdxdna sysfs power node on driver 2.23.0) |
| Native unit | microjoules | watts (power.draw); millijoules (total_energy_consumption) | watts (firmware-estimated instantaneous power) |
| Known issue | AMD RAPL may require kernel patch on Strix; if absent, mark UNAVAILABLE — never fabricate | 1 Hz under-resolves bursty workloads; prefer driver-integrated counter | "Estimated Power" is a firmware estimate (not sense-resistor); absolute number not calibrated against wall-power |

---

## 1. CPU — RAPL on Zen 5 / Strix

### 1.1 What the package domain includes

On AMD Zen 5 (Ryzen AI 9 HX / Strix), the RAPL **package** energy domain
covers the entire CPU package: all cores (P + E), the L3 cache, and the
uncore / fabric inside the SoC die. It is reported as a monotonically
increasing 64-bit microjoule counter.

### 1.2 What it excludes

- **DRAM.** AMD's mainline RAPL implementation does not expose a separate
  `dram` energy domain (Intel does; AMD does not, as of kernel 6.x). DRAM
  energy is therefore **not** included in the CPU figure.
- **Discrete GPU.** Separate rail.
- **NPU subdomain.** Separate rail (the AMD XDNA NPU is on-package but is
  read via xrt-smi, not RAPL).
- **Platform IO.** PCIe lanes, USB controllers, display engines outside the
  SoC package are not in the package domain.

### 1.3 Sysfs path the harness reads

```
/sys/class/powercap/amd-rapl-msr:0/energy_uj
```

Fallback path if `amd-rapl-msr` is unavailable (kernel without the AMD MSR
RAPL driver):

```
/sys/class/hwmon/hwmon*/energy*_input # via amd_energy module
```

The harness probes `amd-rapl-msr` first, falls back to `amd_energy`, and if
neither exists emits a clear `RAPL_UNAVAILABLE` error rather than reporting
zero. **Failing loud is a hard requirement** — see the project risk register
("apples-to-oranges measurements") and 's validation gate, which
explicitly says energy comparison is **skipped, not faked**, when RAPL is
unavailable.

### 1.4 Known issue: AMD RAPL on Strix may require a kernel patch

AMD RAPL surfaces have been gated behind specific kernel commits and may
return zero, `-EIO`, or simply not exist on stock Ubuntu 26.04 kernels. The
bring-up agent has not yet characterized this surface on the target laptop
(see `state/`). validation gate uses a `stress-ng --cpu 8 --timeout
10s` workload as a sanity check: if CPU energy under load is not strictly
greater than CPU energy under `sleep 10`, the reader is broken and 
blocks. In that case, this document grows a note in the
`comparison_rules` front-matter that CPU energy is unavailable on this
host, and the writeup says so.

---

## 2. GPU — `nvidia-smi` whole-board telemetry

### 2.1 What it includes

- **Compute cores.**
- **On-board GDDR/HBM memory.** This is the major reason the GPU figure is
  not comparable to the CPU package figure — the CPU number excludes DRAM,
  the GPU number includes its on-board memory.
- **PCIe interface on the board side.**
- **Board VRMs (voltage regulators).**

### 2.2 What it excludes

- Host CPU.
- Host DRAM.
- NPU.

### 2.3 Exact query string the harness uses

```
nvidia-smi --query-gpu=power.draw,total_energy_consumption --format=csv,noheader,nounits
```

`power.draw` is in watts; `total_energy_consumption` is in millijoules.
Driver support for `total_energy_consumption` requires NVIDIA driver ≥535 on
GeForce-class cards. The harness probes capability:

- If `total_energy_consumption` is supported → use the driver-integrated
  counter (preferred; monotonic; immune to sampling-rate aliasing).
- Else → poll `power.draw` at the harness's max safe rate (~1 Hz on most
  drivers) and trapezoidal-integrate. Annotate the run with
  `gpu_energy_method=trapezoidal`.

### 2.4 Sampling-rate caveat

Driver-reported `power.draw` updates at roughly 1 Hz. For bursty workloads
shorter than ~10 seconds, trapezoidal integration of 1 Hz samples
under-resolves the energy. The driver-integrated counter (`total_energy_-
consumption`) does not have this problem and is the strongly preferred path.

---

## 3. NPU — `xrt-smi` telemetry

### 3.1 What it covers — AIE compute-tile rail (firmware-estimated)

 characterized the `xrt-smi` power surface on this host. The
answer to each of the five "TBD" questions previously listed in this
section:

1. **Which exact rail?** The **AIE compute-tile rail of the active
   hardware-context partition** — i.e. the AIE tiles the running
   xclbin occupies. NOT the full SoC, NOT the CPU package, NOT the
   Radeon iGPU. The amdxdna firmware exposes a single
   `Estimated Power` line per partition; on this host there is one
   partition per NPU and one NPU.
2. **Native unit?** Watts (instantaneous; firmware-internal estimate,
   not a sense-resistor reading). xrt-smi 2.23 prints
   `Estimated Power : 0.412 Watts` in the human-readable
   format and `power_consumption_watts: "0.412"` in the JSON
   variant under `devices[0].platforms[0].electrical`.
3. **Update rate?** The reading appears to update at ~5-10 Hz at the
   firmware level. The bottleneck for the harness is xrt-smi
   invocation overhead (~40 ms per call), so the harness polls at
   10 Hz, matching `NvidiaSmiReader` trapezoidal-fallback rate.
4. **Driver-integrated counter?** No. `amdxdna` 2.23.0 does NOT
   expose a cumulative joule counter analogous to NVML's
   `total_energy_consumption`. The harness trapezoidal-integrates the
   instantaneous-power samples — the same path
   `NvidiaSmiReader.METHOD_TRAPEZOIDAL` takes when the GPU driver
   doesn't support the cumulative counter.
5. **Documented rail name?** `xrt-smi` labels the line literally as
   `Estimated Power`. The JSON key is `power_consumption_watts`. No
   AMD-published documentation enumerates the firmware estimation
   algorithm; treat the absolute number as a **firmware-internal
   estimate** and the **relative** scale (idle vs loaded) as the
   directly-comparable quantity for cross-device J/Mbp figures.

**Empirical characterization on this host
(matteius-ProArt-P16-H7606WI; XRT 2.23.0; amdxdna 2.23.0; firmware
1.1.2.64; AIE2P 6x8; BDF `0000:67:00.1`):**

- **Idle (no hardware context loaded):** 0.001-0.003 W. The AIE
  tiles are clock-gated; this is the firmware noise floor.
- **Loaded (sustained `vector_scalar_mul` kernel running, ~50000
  iterations):** 0.412 W steady-state.
- **Idle → load delta:** ~0.41 W; ratio ~412×. Well above any
  sanity-gate threshold; the reader is plainly responsive to load.

### 3.2 Source path the harness reads

Primary:

```
xrt-smi examine -r platform
```

The harness parses the text output via the regex
`^Estimated Power\s*:\s*([0-9.]+)\s*Watt`. The `platform` reporter is
chosen over `all` because it's the smallest report that includes the
power line (`aie-partitions` and `host` reporters do not).

Equivalent JSON path (not used; the regex on text output avoids a
temp-file write per poll):

```
xrt-smi examine -r platform -f JSON -o <tempfile>
# devices[0].platforms[0].electrical.power_consumption_watts
```

**No sysfs path.** verified that `amdxdna` 2.23.0 does not
expose a power node:

- `/sys/class/hwmon/hwmon*/name` enumeration shows only `acpi_fan`,
  `amdgpu`, `k10temp`, etc. — no `amdxdna` or `npu` hwmon driver.
- `/sys/bus/pci/devices/0000:67:00.1/accel/accel0/` has only `dev`,
  `device`, `power` (runtime-PM), `subsystem`, `uevent` — no
  `power_input`, `energy_uj`, or analogous node.

If a future amdxdna release adds a sysfs power node, the reader can
be extended to prefer it (analogous to RaplReader's powercap-then-
hwmon probe order) without changing `XrtReader`'s public surface.

### 3.3 Reader implementation

`bionpu/bench/energy/xrt.py::XrtReader` mirrors 's
`NvidiaSmiReader.METHOD_TRAPEZOIDAL` path:

- `start` takes one eager sample and spawns a daemon thread that
  polls every 100 ms (10 Hz default).
- `stop` joins the thread, takes one final sample, then
  trapezoidal-integrates the (t_monotonic, watts) samples to joules.
- Constructor probes for xrt-smi + a parseable `Estimated Power`
  line; raises `XrtUnavailableError` on failure.
  `bionpu.bench.energy.auto_reader("npu")` catches that error and
  falls back to `bionpu.bench.harness.XrtStub` (logged at WARNING).

---

## 4. Comparison rules

1. **Caption-cite this file.** Every energy figure produced by `bionpu writeup`
 must cite `bionpu/bench/POWER_DOMAINS.md` in its caption. 's
   template lint enforces this.
2. **Per-domain caveat.** Cross-domain comparisons must list each domain's
   `includes` / `excludes` in the caption, in compact form. Use the caption
   template below.
3. **Never fabricate.** If a reader is UNAVAILABLE on the host (RAPL kernel
   patch missing; xrt-smi not installed; nvidia-smi missing — i.e. CPU-only
   laptop), the harness emits `null` for that device's energy and a non-null
   `reason_unavailable` string. renders the figure with a "—" cell,
   not a zero.
4. **Throughput is comparable; energy derivatives are not, without caveat.**
   `samples/sec`, `bp/sec`, `guides/sec`, `sites/sec` are device-agnostic
   work measurements — they cross-compare cleanly. `J/Mbp` and
   `J/(guide·genome)` cross-compare ONLY with the rail caveats above.
5. **NPU is preliminary until .** Until lands and characterizes the
   xrt-smi rail, every NPU energy figure carries the
   `npu_preliminary_caveat`.

---

## 5. Recommended caption template

's writeup template should auto-populate the caption below for every
energy figure. The bracketed slots are filled from the run config + this
document's front-matter:

```markdown
**Figure N. Energy per [Mbp | guide·genome] across CPU, GPU, and NPU.**
Energy is reported per [`bionpu/bench/POWER_DOMAINS.md`](../../bionpu/bench/POWER_DOMAINS.md)
v{spec_version}.
CPU = package rail (cores + uncore; **excludes** DRAM and platform IO);
sampled from `/sys/class/powercap/amd-rapl-msr:0/energy_uj`.
GPU = whole-board ({gpu_model}, includes GDDR/HBM and on-board PCIe;
**excludes** host CPU/DRAM); sampled via
`nvidia-smi --query-gpu=power.draw,total_energy_consumption`
({gpu_energy_method: driver-integrated | trapezoidal-1Hz}).
NPU = xrt-smi subdomain ({npu_rail_attribution_or "TBD pending "}); sampled
via `xrt-smi examine -r`. **{npu_preliminary_caveat_if_applicable}**
Cross-domain comparison inherits all rail-attribution caveats in the cited
document. {if any reader UNAVAILABLE: "Note: {device} energy was unavailable
on this host ({reason}); the corresponding bar is omitted, not zeroed."}
```

A run that has NPU energy characterized and all three readers available
collapses cleanly:

```markdown
**Figure 3. Energy per Mbp across CPU, GPU, and NPU on Ryzen AI 9 HX.**
Energy reported per `bionpu/bench/POWER_DOMAINS.md` v1.0.0.
CPU = package rail (cores + uncore; excludes DRAM); RAPL via
`/sys/class/powercap/amd-rapl-msr:0/energy_uj`.
GPU = whole-board (RTX 4070 Mobile, includes GDDR6 + PCIe; excludes host);
`nvidia-smi total_energy_consumption` (driver-integrated).
NPU = AIE2P compute-tile rail (excludes host SoC); `xrt-smi examine -r`.
Cross-domain caveats apply per the cited document.
```

---

## 6. Cross-references

- [`UNITS.md`](./UNITS.md) — metric definitions (`J/Mbp`,
  `J/(guide·genome)`, integration window).
- `bionpu/bench/energy/rapl.py` — CPU reader; lands in .
- `bionpu/bench/energy/nvsmi.py` — GPU reader; lands in .
- `bionpu/bench/energy/xrt.py` — NPU reader; lands in .
- `state/bringup-recheck-*.json` — current bring-up status (xrt-smi not
  yet on PATH).
- `bringup-report.md` — bring-up agent run report.
- the umbrella spec, §4.2
  (harness conformance).

## 7. Open issues

- **AMD RAPL on Strix availability:** TBD. sanity gate (`stress-ng`)
  will flip this from open to closed-with-result. If RAPL is unavailable,
  this document gains a `cpu_unavailable` annotation and writeups omit the
  CPU bar (never zero it).
- **NPU rail attribution:** RESOLVED by (2026-04-25). Rail is the
  AIE compute-tile partition; reader is `xrt-smi examine -r platform`
  parsed via regex; trapezoidal-integrated at 10 Hz. See §3 above.
  The "Estimated Power" caveat (firmware estimate, not
  sense-resistor) remains a documented constraint for absolute-number
  citations; relative scale is fine for cross-device J/Mbp.
- **GPU `total_energy_consumption` capability:** detected at runtime per
  laptop's installed driver; the harness records the method actually used
  per run.

### 7.1 outcome on the development host (2026-04-25, additive)

This subsection is appended by and is additive to the open-issues
list above. Do not rewrite §7's bullets; they document the planned
gates. This subsection records the actual gate result on the laptop
where was characterized.

**CPU / RAPL: UNAVAILABLE on this host (writeups must omit CPU
bar — never zero it).**

The kernel does expose `/sys/class/powercap/intel-rapl:0/energy_uj`
(via the `intel_rapl_msr` driver, which on AMD Zen 5 routes the AMD
package counter through the historical "intel-rapl" path), but stock
Ubuntu 26.04 ships it mode `0400 root:root` due to the upstream RAPL
side-channel mitigation. `RaplReader` correctly raises
`RaplUnavailableError`; `auto_reader("cpu")` falls back to
`RaplStub` and the harness records `energy_source="stub"`. Sanity
gate (`stress-ng`) was therefore not run — there is no readable
counter to compare. Remediation options (in increasing order of
operator cost): `chmod a+r` on the sysfs node, a udev rule, the
`amd_energy` hwmon driver, or running the bench as root. See
`bionpu/bench/energy/SANITY-LOG.md` for the full evidence + the
remediation runbook.

**GPU / `nvidia-smi`: PASS** (driver 580.142, RTX 4070 Mobile).
`total_energy_consumption` is not supported on this driver, so the
reader auto-selects the `power.draw` 10-Hz trapezoidal-integration
path documented in §2.3. Idle 5 s reported 14.1 W; CUDA matmul load
reported 68.5 W — delta 54 W, roughly 5× ratio, well above the 5-W
gate. `NvidiaSmiReader` is recorded as `energy_source=real-nvidia-smi`.

**NPU / `xrt-smi`: STUB** until lands. `bionpu/bench/energy/xrt.py`
raises `XrtUnavailableError("…wired in …")`; `auto_reader("npu")`
returns `XrtStub` so the harness emits a record without crashing.

#### Resolution status (post-operator, 2026-04-25)

The two CPU-side blockers from the initial 2026-04-25 entry are now
resolved:

- **`stress-ng` missing from PATH:** RESOLVED. Operator ran
  `sudo apt install stress-ng`; `/usr/bin/stress-ng` (v0.20.01) is now
  available.
- **RAPL `energy_uj` mode `0400 root:root`:** RESOLVED for the package
  counter. Operator ran
  `sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj`; the
  node is now `0444 root:root`. The core subdomain
  (`intel-rapl:0:0/energy_uj`) remains `0400`, which is harmless for
  the package-level reader. The chmod does NOT persist across reboot;
  durable options (udev rule, `amd_energy` hwmon driver, or running
  the bench as root) remain documented in the SANITY-LOG runbook.

CPU sanity gate (`stress-ng --cpu 8 --timeout 10s` vs `time.sleep(10)`
under `RaplReader`) now passes with idle 15.28 W → load 61.14 W
(ratio 4.00×, delta 45.86 W). `tests/test_energy_real.py -m slow`
shows both `test_rapl_responsive_to_load` and
`test_nvsmi_responsive_to_load` GREEN. flips from `Partial` to
`Completed` on this host. See `bionpu/bench/energy/SANITY-LOG.md`
("Re-run after stress-ng + RAPL permissions") for the captured
evidence.

### 7.2 outcome on the development host (2026-04-25, additive)

This subsection is appended by and is additive to §7's open
issues + §7.1's outcome. closes the "NPU rail attribution
TBD" item by characterizing `xrt-smi`'s telemetry surface on this
laptop.

**NPU / `xrt-smi`: AVAILABLE; sanity gate PASSES.**

The XRT install (`/opt/xilinx/xrt/bin/xrt-smi`, version 2.23.0) does
NOT ship a dedicated `power` or `thermal` reporter — only
`aie-partitions`, `host`, `platform`, `all`. The `platform` reporter
is the smallest one that includes a power line. chose
`xrt-smi examine -r platform` parsed via the regex
`^Estimated Power\s*:\s*([0-9.]+)\s*Watt` on stdout. JSON output
under `devices[0].platforms[0].electrical.power_consumption_watts`
is equivalent and slightly heavier (requires a temp file).

**No amdxdna sysfs power node** on this driver (verified by
enumerating `/sys/class/hwmon/hwmon*/name` — no `amdxdna` /
`npu` entry — and inspecting the PCI accel0 device, which exposes
only `dev`, `device`, `power` (runtime-PM), `subsystem`, `uevent`).

**Empirical readings on this host:**

- Idle baseline (no hardware context): 0.001-0.003 W (firmware
  noise floor).
- Loaded (sustained `vector_scalar_mul` kernel, ~50000 iterations
  back-to-back): 0.412 W steady-state.
- Idle → load delta ~0.41 W; ratio ~412×. Far above the
  "load > idle" sanity-gate requirement.

**Reader implementation:**
`bionpu/bench/energy/xrt.py::XrtReader` mirrors 's
`NvidiaSmiReader.METHOD_TRAPEZOIDAL`: 10 Hz daemon-thread poller of
`xrt-smi examine -r platform`, trapezoidal-integrating the (t, watts)
samples to joules at `stop`. There is no driver-integrated
cumulative-energy counter on amdxdna 2.23.0 (no NVML
`total_energy_consumption` analogue), so the trapezoidal path is the
only one offered.

**`auto_reader("npu")`** now returns `XrtReader` (was: `XrtStub`).
Harness records `energy_source="real-xrt-smi"` rather than `"stub"`.
 flips the NPU branch from STUB to Completed on this host. See
`bionpu/bench/energy/SANITY-LOG.md` ("NPU sanity gate") for
the full capture + sanity-gate output.

**Caveat for absolute-number citations:** the `Estimated Power`
field is a **firmware-internal estimate** (not a sense-resistor
reading; AMD does not publish the estimation algorithm). For
cross-device J/Mbp ratios the relative scale is the comparable
quantity. Writeups citing absolute joule numbers should note this
firmware-estimate provenance per the `known_issues` field in this
file's front-matter.
