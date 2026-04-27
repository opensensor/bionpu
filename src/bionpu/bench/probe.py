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

"""Pre-flight probe for the bench harness's energy readers.

Probes the host for the three per-device energy sources documented in
:doc:`POWER_DOMAINS.md` and reports which are AVAILABLE vs which fall
back to the documented :class:`bionpu.bench.harness._StubBase` stubs.
The output is a fixed-shape dict so callers (CLI, CI, the bench
runner) can switch on it without grepping log lines.

POWER_DOMAINS.md §1.4 / §3 are load-bearing here: when a reader is
unavailable, we record the *reason* rather than fabricating a
counter. Stub fallback is a documented operating mode, not an error.
"""

from __future__ import annotations

import platform
import socket
from dataclasses import asdict, dataclass

from .energy.nvsmi import NvidiaSmiReader, NvidiaSmiUnavailableError
from .energy.rapl import probe_rapl, RaplUnavailableError
from .energy.xrt import probe_xrt, XrtUnavailableError

__all__ = [
    "ReaderStatus",
    "ProbeReport",
    "probe_readers",
]


@dataclass(frozen=True)
class ReaderStatus:
    """Status of one device's energy reader.

    Attributes
    ----------
    device:
        ``"cpu"``, ``"gpu"``, or ``"npu"``.
    available:
        ``True`` iff the real reader can run on this host.
    source:
        Which counter family will be used. ``"rapl_sysfs"`` /
        ``"nvidia_smi_total_energy_consumption"`` /
        ``"nvidia_smi_power_draw"`` / ``"xrt_smi_platform"`` for real
        readers; ``"stub"`` when falling back.
    detail:
        Free-form human-readable note: the path / binary / firmware
        capability that made the reader available, or the reason it
        was unavailable.
    """

    device: str
    available: bool
    source: str
    detail: str


@dataclass(frozen=True)
class ProbeReport:
    """Full report from :func:`probe_readers`."""

    hostname: str
    platform_str: str
    readers: tuple[ReaderStatus, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "hostname": self.hostname,
            "platform": self.platform_str,
            "readers": [asdict(r) for r in self.readers],
        }

    def all_real(self) -> bool:
        """True iff every device has a real reader available."""
        return all(r.available for r in self.readers)


def _probe_cpu() -> ReaderStatus:
    try:
        path = probe_rapl()
        return ReaderStatus(
            device="cpu",
            available=True,
            source="rapl_sysfs",
            detail=f"powercap path: {path}",
        )
    except RaplUnavailableError as exc:
        return ReaderStatus(
            device="cpu",
            available=False,
            source="stub",
            detail=str(exc),
        )


def _probe_gpu() -> ReaderStatus:
    # NvidiaSmiReader has its probe in __init__; we instantiate to
    # discover whether nvidia-smi is on PATH and which integration
    # method (driver-integrated vs trapezoidal) the host supports.
    try:
        reader = NvidiaSmiReader()
        return ReaderStatus(
            device="gpu",
            available=True,
            source=f"nvidia_smi_{reader.method}",
            detail=f"nvidia-smi method: {reader.method}",
        )
    except NvidiaSmiUnavailableError as exc:
        return ReaderStatus(
            device="gpu",
            available=False,
            source="stub",
            detail=str(exc),
        )


def _probe_npu() -> ReaderStatus:
    try:
        bin_path = probe_xrt()
        return ReaderStatus(
            device="npu",
            available=True,
            source="xrt_smi_platform",
            detail=f"xrt-smi binary: {bin_path}",
        )
    except XrtUnavailableError as exc:
        return ReaderStatus(
            device="npu",
            available=False,
            source="stub",
            detail=str(exc),
        )


def probe_readers() -> ProbeReport:
    """Probe every per-device energy reader on this host.

    Cheap (no integration windows; just capability checks) and
    side-effect free. Always succeeds — unavailability is reported
    in the per-reader status, not raised.
    """
    return ProbeReport(
        hostname=socket.gethostname(),
        platform_str=platform.platform(),
        readers=(_probe_cpu(), _probe_gpu(), _probe_npu()),
    )
