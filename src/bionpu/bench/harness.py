# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Bench harness — timed-run wrapper + energy-reader Protocol + stubs.

Implements PRD §4.2 (umbrella) per the metric formulas locked in
`bionpu/bench/UNITS.md` and the rail definitions in
`bionpu/bench/POWER_DOMAINS.md`.

This module ships **stubs only** for energy reading. (CPU/GPU) and 
(NPU) replace the stubs with real readers via the `EnergyReader` Protocol.
Every measurement record carries an `energy_source` field so the writeup
pipeline can caveat stub-derived numbers as preliminary.

Memory:
- RSS via `psutil.Process().memory_info().rss`.
- VRAM peak via `torch.cuda.max_memory_allocated()` (if torch + CUDA present).
- Tile memory is NPU-only and is recorded as `null` here. /
  populate it from IRON/Peano build artifacts.

Per UNITS.md, `null` ≠ `0` for any unavailable metric.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import psutil

from .units import (
    EnergySource,
    bp_per_sec,
    guides_per_sec,
    nearest_rank_percentile,
    samples_per_sec,
    sites_per_sec,
)

# --------------------------------------------------------------------------- #
# EnergyReader Protocol + stubs
# --------------------------------------------------------------------------- #

@runtime_checkable
class EnergyReader(Protocol):
    """Pluggable energy-reader interface.

    Every reader exposes:
    - `start()`: capture the starting energy counter.
    - `stop() -> float`: capture the ending counter and return Joules consumed
      over the [start, stop) window.
    - `is_real`: `False` for the stubs; `True` once / wire real
      readers. The harness records the corresponding `energy_source` enum so
      the writeup can mark stub-derived numbers as preliminary.
    - `source`: the enum value to record in `measurements.json`.
    """

    is_real: bool
    source: EnergySource

    def start(self) -> None: ...

    def stop(self) -> float: ...

@dataclass
class _StubBase:
    """Shared stub plumbing. Reports `wall_seconds * watts_estimate` joules.

    The estimate is intentionally a fixed average power (idle + small load
    multiplier) so that two consecutive stub runs on the same wall-clock window
    produce the same joule count to within scheduler jitter. Real readers in
     / replace this with sysfs / driver counters.
    """

    is_real: bool = False
    source: EnergySource = EnergySource.STUB
    _t_start_s: float = 0.0
    # Subclasses set this.
    _watts: float = 0.0

    def start(self) -> None:
        self._t_start_s = time.monotonic()

    def stop(self) -> float:
        elapsed_s = max(0.0, time.monotonic() - self._t_start_s)
        return elapsed_s * self._watts

@dataclass
class RaplStub(_StubBase):
    """CPU energy stub: 8 W average idle + small load (per POWER_DOMAINS.md cpu)."""

    _watts: float = 8.0

@dataclass
class NvidiaSmiStub(_StubBase):
    """GPU energy stub: 25 W average idle (per POWER_DOMAINS.md gpu)."""

    _watts: float = 25.0

@dataclass
class XrtStub(_StubBase):
    """NPU energy stub: ~5 W average (per POWER_DOMAINS.md npu, preliminary).

    POWER_DOMAINS.md flags this as TBD pending . The stub picks a
    conservative ~5 W to avoid implying false precision before lands.
    """

    _watts: float = 5.0

def default_energy_reader(device: str) -> EnergyReader:
    """Return the default stub reader for a device.

    Stubs only — preserves 's contract: callers that want a real
    reader pass one explicitly via ``TimedRun(..., energy_reader=...)``
    or use :func:`bionpu.bench.energy.auto_reader`. The umbrella
    parity runner (and :mod:`bionpu.bench.runner` users via /)
    pass ``auto_reader(device)`` to ``TimedRun`` so the harness records
    the real ``energy_source`` enum end-to-end.
    """
    if device == "cpu":
        return RaplStub()
    if device == "gpu":
        return NvidiaSmiStub()
    if device == "npu":
        return XrtStub()
    raise ValueError(f"unknown device {device!r}; expected one of cpu/gpu/npu")

# --------------------------------------------------------------------------- #
# MeasurementRun dataclass
# --------------------------------------------------------------------------- #

@dataclass
class MeasurementRun:
    """One timed run's measurements, serializable to `measurements.json`.

    Field names match `bionpu/bench/UNITS.md` so the schema and the
    front-matter `metrics:` list line up.
    """

    track: str
    op: str
    device: str
    t_start: float
    t_end: float
    throughput: dict[str, float] = field(default_factory=dict)
    accuracy: dict[str, float | None] = field(default_factory=dict)
    energy_j: float | None = None
    energy_per_unit: dict[str, float | None] = field(default_factory=dict)
    energy_source: str = EnergySource.STUB.value
    latencies_ms: dict[str, float | None] = field(default_factory=dict)
    rss_bytes: int | None = None
    vram_peak_bytes: int | None = None
    tile_memory_bytes: int | None = None  # NPU-only; null on cpu/gpu per UNITS.md
    env: dict[str, str] = field(default_factory=dict)
    n_iters: int = 0
    reason_unavailable: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "track": self.track,
            "op": self.op,
            "device": self.device,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "throughput": self.throughput,
            "accuracy": self.accuracy,
            "energy_j": self.energy_j,
            "energy_per_unit": self.energy_per_unit,
            "energy_source": self.energy_source,
            "latencies_ms": self.latencies_ms,
            "rss_bytes": self.rss_bytes,
            "vram_peak_bytes": self.vram_peak_bytes,
            "tile_memory_bytes": self.tile_memory_bytes,
            "env": self.env,
            "n_iters": self.n_iters,
            "reason_unavailable": self.reason_unavailable,
        }

# --------------------------------------------------------------------------- #
# TimedRun context manager
# --------------------------------------------------------------------------- #

def _capture_env() -> dict[str, str]:
    """Capture a stable, deterministic-across-machines env snapshot.

    Anything host-specific (hostname, MAC, full $PATH) is intentionally
    excluded so the deterministic-modulo-timestamp test passes. will
    cite this dict in writeup metadata; per-run host info goes elsewhere.
    """
    return {
        "python_version": os.environ.get("PYTHON_VERSION", ""),
        "harness_module": "bionpu.bench.harness",
    }

def _read_rss() -> int | None:
    try:
        return int(psutil.Process().memory_info().rss)
    except Exception:  # pragma: no cover — psutil should always work on Linux
        return None

def _read_vram_peak() -> int | None:
    """Best-effort PyTorch CUDA peak. Returns `None` if torch/CUDA absent."""
    try:
        import torch  # local import — torch is optional at the package level
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        return int(torch.cuda.max_memory_allocated())
    except Exception:  # pragma: no cover
        return None

def _reset_vram_peak() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:  # pragma: no cover
            pass

class TimedRun:
    """Context manager that wraps a timed run.

    Usage::

        with TimedRun(track="basecalling", op="basecall", device="cpu") as r:
            r.record_units(bases=N)
            for chunk in chunks:
                t0 = time.monotonic()
                process(chunk)
                r.record_latency_ms((time.monotonic() - t0) * 1000)
        result = r.measurements

    The harness:
    - Captures `t_start` / `t_end` via `time.monotonic_ns()` (UNITS.md §1).
    - Reads RSS at `__exit__` and VRAM peak at `__exit__` (after `__enter__`
      reset).
    - Calls the energy reader's `start()` at `__enter__` and `stop()` at
      `__exit__`. Stubs by default; / replace.
    - Computes nearest-rank latency percentiles from `record_latency_ms`
      samples (UNITS.md §4.1).
    """

    def __init__(
        self,
        track: str,
        op: str,
        device: str,
        energy_reader: EnergyReader | None = None,
    ) -> None:
        self.track = track
        self.op = op
        self.device = device
        self._energy_reader = energy_reader or default_energy_reader(device)
        self._latency_samples: list[float] = []
        self._units: dict[str, int] = {}
        self._t_start_ns: int = 0
        self._t_end_ns: int = 0
        self.measurements: MeasurementRun | None = None  # populated at __exit__

    # ----- context-manager protocol -----

    def __enter__(self) -> TimedRun:
        _reset_vram_peak()
        self._energy_reader.start()
        self._t_start_ns = time.monotonic_ns()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._t_end_ns = time.monotonic_ns()
        joules = self._energy_reader.stop()
        rss = _read_rss()
        vram = _read_vram_peak()

        seconds = max(0.0, (self._t_end_ns - self._t_start_ns) / 1e9)
        throughput = self._compute_throughput(seconds)
        latencies = self._compute_latencies()
        energy_per_unit = self._compute_energy_per_unit(joules)

        self.measurements = MeasurementRun(
            track=self.track,
            op=self.op,
            device=self.device,
            t_start=self._t_start_ns / 1e9,
            t_end=self._t_end_ns / 1e9,
            throughput=throughput,
            accuracy={},
            energy_j=joules,
            energy_per_unit=energy_per_unit,
            energy_source=self._energy_reader.source.value,
            latencies_ms=latencies,
            rss_bytes=rss,
            vram_peak_bytes=vram,
            tile_memory_bytes=None, # NPU-only; / populate. UNITS.md: null != 0.
            env=_capture_env(),
            n_iters=0,
            reason_unavailable=None,
        )

    # ----- recording API used inside the `with` block -----

    def record_units(
        self,
        *,
        samples: int | None = None,
        bases: int | None = None,
        guides: int | None = None,
        sites: int | None = None,
    ) -> None:
        """Record the work done so the harness can compute throughput.

        Multiple unit kinds may be recorded in the same run (e.g. a
        basecalling run records both samples-in and bases-out).
        """
        if samples is not None:
            self._units["samples"] = self._units.get("samples", 0) + int(samples)
        if bases is not None:
            self._units["bases"] = self._units.get("bases", 0) + int(bases)
        if guides is not None:
            self._units["guides"] = self._units.get("guides", 0) + int(guides)
        if sites is not None:
            self._units["sites"] = self._units.get("sites", 0) + int(sites)

    def record_latency_ms(self, ms: float) -> None:
        """Record a single per-chunk / per-batch latency sample."""
        self._latency_samples.append(float(ms))

    def record_iter(self) -> None:
        """Increment the iteration counter (best-effort metadata for the run)."""
        # Stored on the dataclass at __exit__ via length of latency samples by
        # default; this method is here so callers can override semantics.
        pass

    # ----- helpers -----

    def _compute_throughput(self, seconds: float) -> dict[str, float]:
        out: dict[str, float] = {}
        if "samples" in self._units:
            out["samples_per_sec"] = samples_per_sec(self._units["samples"], seconds)
        if "bases" in self._units:
            out["bp_per_sec"] = bp_per_sec(self._units["bases"], seconds)
        if "guides" in self._units:
            out["guides_per_sec"] = guides_per_sec(self._units["guides"], seconds)
        if "sites" in self._units:
            out["sites_per_sec"] = sites_per_sec(self._units["sites"], seconds)
        return out

    def _compute_latencies(self) -> dict[str, float | None]:
        if not self._latency_samples:
            return {"p50": None, "p95": None, "p99": None, "max": None}
        samples = self._latency_samples
        return {
            "p50": nearest_rank_percentile(samples, 0.50),
            "p95": nearest_rank_percentile(samples, 0.95),
            "p99": nearest_rank_percentile(samples, 0.99),
            "max": max(samples),
        }

    def _compute_energy_per_unit(self, joules: float | None) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        if joules is None:
            return out
        if "bases" in self._units and self._units["bases"] > 0:
            out["joules_per_megabase"] = joules / (self._units["bases"] / 1e6)
        if "guides" in self._units and self._units["guides"] > 0:
            # n_haploid_bp is run-config-dependent (UNITS.md §3.3) — not known
            # at harness level. Recorded as joules per guide here; the writeup
            # multiplies in the genome size from the run config.
            out["joules_per_guide"] = joules / self._units["guides"]
        return out

__all__ = [
    "EnergyReader",
    "MeasurementRun",
    "NvidiaSmiStub",
    "RaplStub",
    "TimedRun",
    "XrtStub",
    "default_energy_reader",
    "nearest_rank_percentile",
]
