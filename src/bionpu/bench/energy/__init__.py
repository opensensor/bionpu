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

"""Real energy readers for the bench harness.

Public surface:

- :class:`RaplReader` — CPU package energy via ``/sys/class/powercap/``
. Raises :class:`RaplUnavailableError` when the surface is not
  exposed on this kernel (POWER_DOMAINS.md §1.4 — never fabricate).
- :class:`NvidiaSmiReader` — GPU whole-board energy via ``nvidia-smi``
. Auto-selects ``total_energy_consumption`` (driver-integrated)
  vs ``power.draw`` (trapezoidal). Raises
  :class:`NvidiaSmiUnavailableError` when nvidia-smi is missing.
- :class:`XrtReader` — NPU subdomain energy via
  ``xrt-smi examine -r platform``. Trapezoidal integration of
  the firmware-reported instantaneous power at ~10 Hz. Raises
  :class:`XrtUnavailableError` when xrt-smi is missing /
  non-functional. :class:`XrtStub` (re-exported from
  :mod:`bionpu.bench.harness`) is the documented fallback.
- :func:`auto_reader` — factory that returns the best-available reader
  for a device, falling back to the matching stub from
  :mod:`bionpu.bench.harness` with a logged reason.
- :func:`probe_rapl`, :func:`probe_xrt` — exposed so callers can probe
  without instantiating the reader.

Per POWER_DOMAINS.md §1.4 + §3: real readers MUST fail loud. Stub
fallback happens only at the factory boundary (auto_reader), which
records the fallback reason in
:attr:`bionpu.bench.MeasurementRun.reason_unavailable` via call sites.
"""

from __future__ import annotations

import logging

from ..harness import NvidiaSmiStub, RaplStub, XrtStub
from .nvsmi import (
    NvidiaSmiReader,
    NvidiaSmiUnavailableError,
)
from .rapl import (
    RaplReader,
    RaplUnavailableError,
    probe_rapl,
)
from .xrt import (
    XrtReader,
    XrtUnavailableError,
    probe_xrt,
    read_xrt_energy,
)

_log = logging.getLogger(__name__)

def auto_reader(device: str):
    """Return the best-available energy reader for ``device``.

    Behaviour per device:

    - ``"cpu"`` → :class:`RaplReader` if ``probe_rapl()`` succeeds; else
      :class:`bionpu.bench.harness.RaplStub` with a log line citing the
      RAPL unavailability reason (POWER_DOMAINS.md §1.4: never
      fabricate; the stub clearly reports ``energy_source=stub``).
    - ``"gpu"`` → :class:`NvidiaSmiReader` if probe succeeds; else
      :class:`bionpu.bench.harness.NvidiaSmiStub`.
    - ``"npu"`` → :class:`XrtReader` if ``probe_xrt()`` succeeds; else
      :class:`bionpu.bench.harness.XrtStub` with a log line citing the
      xrt-smi unavailability reason (POWER_DOMAINS.md §3 — never
      fabricate; the stub clearly reports ``energy_source=stub``).

    The fallback path is intentional, not a silent failure: every stub
    fallback is logged at WARNING level and the returned reader's
    ``source`` field tells the harness to record the right
    ``energy_source`` enum.

    Args:
        device: ``"cpu"``, ``"gpu"``, or ``"npu"``.

    Raises:
        ValueError: ``device`` is not one of the three accepted names.
    """
    if device == "cpu":
        try:
            return RaplReader()
        except RaplUnavailableError as exc:
            _log.warning(
                "RAPL unavailable; falling back to RaplStub. reason=%s", exc
            )
            return RaplStub()
    if device == "gpu":
        try:
            return NvidiaSmiReader()
        except NvidiaSmiUnavailableError as exc:
            _log.warning(
                "nvidia-smi unavailable; falling back to NvidiaSmiStub. reason=%s",
                exc,
            )
            return NvidiaSmiStub()
    if device == "npu":
        try:
            return XrtReader()
        except XrtUnavailableError as exc:
            _log.warning(
                "xrt-smi unavailable; falling back to XrtStub. reason=%s",
                exc,
            )
            return XrtStub()
    raise ValueError(
        f"unknown device {device!r}; expected one of 'cpu', 'gpu', 'npu'"
    )

__all__ = [
    "NvidiaSmiReader",
    "NvidiaSmiStub",
    "NvidiaSmiUnavailableError",
    "RaplReader",
    "RaplStub",
    "RaplUnavailableError",
    "XrtReader",
    "XrtStub",
    "XrtUnavailableError",
    "auto_reader",
    "probe_rapl",
    "probe_xrt",
    "read_xrt_energy",
]
