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

"""RAPL CPU energy reader.

Implements the CPU package-energy reader contract from
``bionpu/bench/POWER_DOMAINS.md`` §1 and the joule-integration window
from ``bionpu/bench/UNITS.md`` §3.1. Replaces the stub in
``bionpu/bench/harness.py`` (``RaplStub``) with a real
``/sys/class/powercap/`` reader.

Probe order (first readable wins):

1. ``/sys/class/powercap/intel-rapl:0/energy_uj`` — on AMD Zen 5 with
   the ``intel_rapl_msr`` driver loaded, this hierarchy exposes the
   AMD package energy counter under the historical "intel-rapl" name.
2. ``/sys/class/powercap/amd-rapl-msr:0/energy_uj`` — explicit AMD MSR
   exposure on kernels that route AMD RAPL to a distinct sysfs path.
3. ``/sys/class/powercap/amd_energy:0/energy_uj`` — legacy
   ``amd_energy`` driver path.
4. ``/sys/class/hwmon/hwmon*/energy*_input`` — from the ``amd_energy``
   hwmon driver on older kernels.

If none of those candidates is readable by the current process, raise
:class:`RaplUnavailableError` with the candidate list and a remediation
hint. **Never fabricate** — POWER_DOMAINS.md §1.4 and the umbrella
PRD §4.2 risk row "apples-to-oranges measurements" make this load-bearing.

Counter-wrap handling
---------------------

``energy_uj`` is documented as a monotonically-increasing microjoule
counter. The kernel exposes the wrap point via ``max_energy_range_uj``
in the same directory. The reader records both ``start_uj`` and
``max_uj`` at :meth:`RaplReader.start` and computes:

    delta_uj = end_uj - start_uj                    # normal case
    delta_uj = (max_uj - start_uj) + end_uj         # wrap detected

A wrap is detected when ``end_uj < start_uj``. On Zen 5 the counter is
64-bit (max ≈ 1.8e19 µJ ≈ 575 years at 1 W) so wrap is theoretical.
Older Intel parts use a 32-bit counter that wraps every ~60 s at 70 W;
the harness handles that case for portability.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ..units import EnergySource

# Sysfs candidates probed in order. First readable + parseable wins.
_POWERCAP_CANDIDATES: tuple[str, ...] = (
    "/sys/class/powercap/intel-rapl:0/energy_uj",
    "/sys/class/powercap/amd-rapl-msr:0/energy_uj",
    "/sys/class/powercap/amd_energy:0/energy_uj",
)

# Hwmon glob pattern as the final fallback (amd_energy module).
_HWMON_GLOB = "/sys/class/hwmon/hwmon*/energy*_input"

# Default max if max_energy_range_uj is missing for a candidate path: the
# 32-bit microjoule wrap point (~4.3e9 µJ ≈ 4295 J). Modern AMD/Intel
# expose 64-bit counters and the kernel reports the actual wrap point in
# max_energy_range_uj; this default is only used as a last-resort safety.
_DEFAULT_MAX_UJ = 2**32

class RaplUnavailableError(RuntimeError):
    """Raised when no RAPL surface is readable on this host.

    Per POWER_DOMAINS.md §1.4 the harness MUST fail loud rather than
    return zero or a fabricated number. Catch sites either fall back to
    the matching stub from ``bionpu.bench.harness`` (writeup pipeline
    marks the figure as preliminary) or skip the energy comparison
    entirely.
    """

def _hwmon_candidates() -> Iterable[Path]:
    """Yield candidate hwmon energy paths.

    Globs ``/sys/class/hwmon/hwmon*/energy*_input``. Order is sysfs
    enumeration order (typically by hwmon index); we don't sort to keep
    the system's own ordering preference.
    """
    root = Path("/sys/class/hwmon")
    if not root.is_dir():
        return
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        # Only consider amd_energy hwmon nodes; other hwmons (k10temp,
        # acpitz, amdgpu, batteries) expose temp/power but their
        # energy*_input fields are not the CPU package counter.
        try:
            name = (entry / "name").read_text().strip()
        except OSError:
            continue
        if name != "amd_energy":
            continue
        yield from sorted(entry.glob("energy*_input"))

def _try_read_uj(path: Path) -> int | None:
    """Attempt to read a microjoule counter; return None on any failure.

    Failure modes covered:
    - ``PermissionError`` (mode 0400 on stock kernels with the RAPL
      side-channel mitigation).
    - ``FileNotFoundError`` (path does not exist).
    - ``ValueError`` (non-numeric body).
    - ``OSError`` (e.g. ``-EIO``).
    """
    try:
        body = path.read_text()
    except (PermissionError, FileNotFoundError, OSError):
        return None
    try:
        return int(body.strip())
    except ValueError:
        return None

def _max_uj_for(path: Path) -> int:
    """Return ``max_energy_range_uj`` for the same powercap dir as ``path``.

    Falls back to ``_DEFAULT_MAX_UJ`` if the sibling node is missing /
    unreadable. ``path`` is the ``energy_uj`` file or an
    ``energy*_input`` hwmon file; for hwmon the wrap point is not
    documented so we use the 32-bit fallback (safe; over-reports wraps
    on a true 64-bit counter only as a corner case).
    """
    parent = path.parent
    sibling = parent / "max_energy_range_uj"
    if sibling.is_file():
        body = _try_read_uj(sibling)
        if body is not None and body > 0:
            return body
    return _DEFAULT_MAX_UJ

def probe_rapl() -> Path:
    """Return the first readable RAPL energy_uj path.

    Probes in order:
    1. Each path in :data:`_POWERCAP_CANDIDATES`.
    2. Each ``amd_energy`` hwmon node under ``/sys/class/hwmon``.

    Raises:
        RaplUnavailableError: if no candidate is readable. The error
            message lists every path tried plus a remediation hint per
            POWER_DOMAINS.md §1.4.
    """
    checked: list[str] = []

    for cand in _POWERCAP_CANDIDATES:
        p = Path(cand)
        checked.append(cand)
        if not p.is_file():
            continue
        if _try_read_uj(p) is not None:
            return p

    # hwmon fallback (amd_energy)
    for p in _hwmon_candidates():
        checked.append(str(p))
        if _try_read_uj(p) is not None:
            return p

    # All candidates either missing, mode 0400 (root-only), or returned
    # garbage. POWER_DOMAINS.md §1.4 says fail loud.
    is_root = os.geteuid() == 0
    hint = (
        "AMD RAPL on Strix typically requires `intel_rapl_msr` loaded AND "
        "either running as root OR `chmod a+r /sys/class/powercap/*/energy_uj` "
        "(stock Ubuntu ships these mode 0400 due to a side-channel mitigation). "
        "Alternative: install `amd_energy` and read via hwmon. If neither path "
        "works on this kernel, document CPU energy as UNAVAILABLE per "
        "POWER_DOMAINS.md §1.4 and skip the cross-domain CPU energy comparison."
    )
    if is_root:
        hint = (
            "Running as root and still no RAPL surface — kernel likely lacks "
            "`intel_rapl_msr` / `amd_energy` modules. " + hint
        )
    raise RaplUnavailableError(
        f"RAPL surface not exposed on this kernel; checked: {checked!r}; "
        f"suggested fix: {hint}"
    )

@dataclass
class RaplReader:
    """Real CPU package energy reader.

    Probes ``/sys/class/powercap/`` (and the ``amd_energy`` hwmon
    fallback) at construction. If no candidate is readable, the
    constructor raises :class:`RaplUnavailableError` so the caller can
    fall back to :class:`bionpu.bench.harness.RaplStub` deliberately
    (never silently).

    Wrap handling: if ``end_uj < start_uj``, the reader assumes the
    counter wrapped exactly once and computes delta as
    ``(max_uj - start_uj) + end_uj``. The wrap point is read from
    ``max_energy_range_uj`` in the same powercap directory; for hwmon
    fallback, defaults to ``2**32`` µJ.

    Implements the :class:`bionpu.bench.harness.EnergyReader` Protocol.
    """

    is_real: bool = True
    source: EnergySource = EnergySource.REAL_RAPL
    path: Path = field(default_factory=lambda: Path("/dev/null"))
    _start_uj: int = 0
    _max_uj: int = _DEFAULT_MAX_UJ

    def __init__(self, path: Path | None = None) -> None:
        # If the caller passed an explicit path (testing), honor it; else
        # probe. Probe failure raises RaplUnavailableError — by design.
        self.is_real = True
        self.source = EnergySource.REAL_RAPL
        self.path = path if path is not None else probe_rapl()
        if _try_read_uj(self.path) is None:
            # Caller supplied an unreadable path explicitly.
            raise RaplUnavailableError(
                f"RAPL path {self.path!s} is not readable by this process"
            )
        self._max_uj = _max_uj_for(self.path)
        self._start_uj = 0

    def _read_uj(self) -> int:
        """Read the current counter; raises RaplUnavailableError on failure
        AFTER successful construction (e.g. permissions changed mid-run).
        """
        v = _try_read_uj(self.path)
        if v is None:
            raise RaplUnavailableError(
                f"RAPL path {self.path!s} became unreadable after start()"
            )
        return v

    def start(self) -> None:
        """Capture the starting microjoule counter."""
        self._start_uj = self._read_uj()

    def stop(self) -> float:
        """Return joules consumed over the [start, stop) window.

        Handles single-wrap: if the counter has wrapped exactly once
        between start and stop (``end < start``), delta is computed as
        ``(max_uj - start) + end``.

        Multi-wrap is not handled; on a 32-bit µJ counter at 70 W
        (~60 s wrap), a window longer than ~60 s could double-wrap. The
        harness records the start/end timestamps so post-hoc analysis
        can detect that case. POWER_DOMAINS.md §1.1 documents Zen 5 as
        64-bit so multi-wrap is theoretical there.
        """
        end_uj = self._read_uj()
        if end_uj >= self._start_uj:
            delta_uj = end_uj - self._start_uj
        else:
            # Single wrap; reconstruct the missing prefix from max.
            delta_uj = (self._max_uj - self._start_uj) + end_uj
        return delta_uj / 1e6

__all__ = [
    "RaplReader",
    "RaplUnavailableError",
    "probe_rapl",
]
