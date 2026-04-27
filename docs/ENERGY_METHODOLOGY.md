# Energy methodology

> Status: shell — populated from `bionpu/bench/POWER_DOMAINS.md` and
> `bionpu/bench/energy/SANITY-LOG.md` during the v0.1 extraction. Until
> filled, the v0.1 `bench` numbers in this repo should be treated as
> wall-clock only.

This document will cover:

- AMD RAPL counter access path (`/sys/class/powercap/intel-rapl:*` on
  Ryzen-AI HX systems; the AMD-specific `package-0` / `package-1`
  domain layout).
- Sustained-load measurement — pre-warmup window, measurement window,
  drift-detection window — the three-phase shape that distinguishes
  steady-state energy from cold-start spikes.
- Spec-bracketing assumptions — what TDP range we assume the package
  is in, how we cross-check against the documented Ryzen-AI 9 HX SKU
  TDP envelope, where the assumption fails.
- NPU-specific power accounting — what's measurable today vs what is
  inferred from the package counter delta with NPU idle vs NPU active.
- Reproducibility envelope — what hardware revisions / firmware
  versions / governor settings the documented numbers are valid for.
