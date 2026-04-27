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

"""Metric formulas + enums for the bench harness.

Implements the formulas locked in `bionpu/bench/UNITS.md`. Every
metric name used in the harness output appears in this module so that a
mechanical lint can cross-check the front-matter `metrics:` list of
`UNITS.md`.

Pure functions. No I/O. No global state.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from enum import StrEnum
from typing import Final

# --------------------------------------------------------------------------- #
# Enums (string-valued so they round-trip through JSON without a custom encoder)
# --------------------------------------------------------------------------- #

class Track(StrEnum):
    BASECALLING = "basecalling"
    CRISPR = "crispr"
    SMOKE = "_smoke"  # used by `bionpu bench --all` for the default workload
    UMBRELLA = "umbrella"

class Device(StrEnum):
    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"

class ThroughputUnit(StrEnum):
    """Throughput unit names from UNITS.md §1."""

    SAMPLES_PER_SEC = "samples_per_sec"  # basecalling chunker input
    BP_PER_SEC = "bp_per_sec"  # basecalling FASTQ output
    GUIDES_PER_SEC = "guides_per_sec"  # CRISPR scan
    SITES_PER_SEC = "sites_per_sec"  # CRISPR scoring

class AccuracyType(StrEnum):
    """Accuracy metric names from UNITS.md §2."""

    MODAL_ACCURACY = "modal_accuracy"  # basecalling, percentage points
    INDEL_RATE = "indel_rate"  # basecalling, ratio
    SET_EQUALITY_PCT = "set_equality_pct"  # CRISPR

class EnergyUnit(StrEnum):
    """Energy-derivative unit names from UNITS.md §3."""

    JOULES_PER_MEGABASE = "joules_per_megabase"  # basecalling J/Mbp
    JOULES_PER_GUIDE_GENOME = "joules_per_guide_genome"  # CRISPR J/(guide·genome)

class EnergySource(StrEnum):
    """How the joules figure was obtained.

    `stub` is what ships; / replace these with real readers.
    Every record in `measurements.json` carries this so the writeup pipeline
 can mark stub-derived numbers as preliminary.
    """

    STUB = "stub"
    REAL_RAPL = "real-rapl"
    REAL_NVIDIA_SMI = "real-nvidia-smi"
    REAL_XRT_SMI = "real-xrt-smi"

# --------------------------------------------------------------------------- #
# Latency: nearest-rank percentile (UNITS.md §4.1)
# --------------------------------------------------------------------------- #

PERCENTILE_METHOD: Final[str] = "nearest-rank"

def nearest_rank_percentile(samples: Iterable[float], p: float) -> float:
    """Nearest-rank percentile per UNITS.md §4.1.

    Given N samples sorted ascending and percentile p in (0, 1]:

        index = ceil(p * N) - 1     # 0-indexed
        value = sorted_samples[index]

    NOT linear interpolation. UNITS.md hard-codes this method; the harness
    refuses any other.

    Raises:
        ValueError: if `samples` is empty or `p` is outside (0, 1].
    """
    if not 0.0 < p <= 1.0:
        raise ValueError(f"percentile must be in (0, 1], got {p}")
    sorted_samples = sorted(float(x) for x in samples)
    n = len(sorted_samples)
    if n == 0:
        raise ValueError("nearest_rank_percentile requires at least one sample")
    index = max(0, math.ceil(p * n) - 1)
    return sorted_samples[index]

# --------------------------------------------------------------------------- #
# Throughput formulas (UNITS.md §1)
# --------------------------------------------------------------------------- #

def samples_per_sec(samples: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return samples / seconds

def bp_per_sec(bases: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return bases / seconds

def guides_per_sec(guides: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return guides / seconds

def sites_per_sec(sites: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return sites / seconds

# --------------------------------------------------------------------------- #
# Energy-derivative formulas (UNITS.md §3)
# --------------------------------------------------------------------------- #

def joules_per_megabase(joules: float, bases: int) -> float | None:
    if bases <= 0:
        return None
    return joules / (bases / 1e6)

def joules_per_guide_genome(
    joules: float, n_guides: int, n_haploid_bp: int
) -> float | None:
    if n_guides <= 0 or n_haploid_bp <= 0:
        return None
    return joules / (n_guides * n_haploid_bp)

__all__ = [
    "AccuracyType",
    "Device",
    "EnergySource",
    "EnergyUnit",
    "PERCENTILE_METHOD",
    "ThroughputUnit",
    "Track",
    "bp_per_sec",
    "guides_per_sec",
    "joules_per_guide_genome",
    "joules_per_megabase",
    "nearest_rank_percentile",
    "samples_per_sec",
    "sites_per_sec",
]
