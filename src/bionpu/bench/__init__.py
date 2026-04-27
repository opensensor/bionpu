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

"""Benchmarking harness.

Implements PRD §4.2 (umbrella) per the metric formulas locked in
`bionpu/bench/UNITS.md` and rail definitions in `POWER_DOMAINS.md`.

Public surface:

- :class:`TimedRun` — context manager wrapping a timed run.
- :class:`MeasurementRun` — dataclass serialized into ``measurements.json``.
- :func:`bench_all` — runs every (op, device) pair, writes ``results/<track>/
  <run_id>/measurements.json`` matching ``schema.json``.

Stubs (replaced by real readers in / ):

- :class:`RaplStub`, :class:`NvidiaSmiStub`, :class:`XrtStub` — implement the
  :class:`EnergyReader` Protocol with plausible joule figures so downstream
  tasks can integration-test the harness without a real device.
"""

from .harness import (
    EnergyReader,
    MeasurementRun,
    NvidiaSmiStub,
    RaplStub,
    TimedRun,
    XrtStub,
    default_energy_reader,
    nearest_rank_percentile,
)
from .runner import SCHEMA_VERSION, bench_all

__all__ = [
    "EnergyReader",
    "MeasurementRun",
    "NvidiaSmiStub",
    "RaplStub",
    "SCHEMA_VERSION",
    "TimedRun",
    "XrtStub",
    "bench_all",
    "default_energy_reader",
    "nearest_rank_percentile",
]
