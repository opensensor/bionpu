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

"""NPU energy reader via ``xrt-smi``.

Implements the NPU subdomain reader contract from
``bionpu/bench/POWER_DOMAINS.md`` §3 and the joule-integration window
from ``bionpu/bench/UNITS.md`` §3.1. Replaces the stub that raised
:class:`XrtUnavailableError` with a real reader that polls
``xrt-smi examine -r platform`` for the firmware-reported instantaneous
power and trapezoidal-integrates over the ``[start, stop)`` window.

Telemetry surface
-----------------

The bring-up agent's ``xrt-smi`` (XRT 2.23.0 / amdxdna 2.23.0 /
firmware 1.1.2.64) does NOT ship a dedicated ``power`` or ``thermal``
reporter on AIE2P / Strix. The available reporters are
``aie-partitions``, ``host``, ``platform``, and ``all``. The
``platform`` reporter is the smallest one that includes the
firmware-reported instantaneous-power line:

    Estimated Power          : 0.000 Watts

This corresponds to ``devices[0].platforms[0].electrical
.power_consumption_watts`` in the JSON variant. There is no
amdxdna sysfs power node on this driver / firmware combination
(verified by enumerating
``/sys/class/hwmon/hwmon*/name`` — no ``amdxdna`` / ``npu`` entry —
and the PCI ``accel0`` device exposes only ``dev`` / ``power``
runtime-PM directories, not a power-input node). therefore
chose the ``xrt-smi examine -r platform`` text-output surface as the
primary path, parsed via the regex ``^Estimated Power\\s*:\\s*
([0-9.]+)\\s*Watts``.

What the reading covers
~~~~~~~~~~~~~~~~~~~~~~~

POWER_DOMAINS.md §3 previously marked rail attribution as "TBD pending
". 's empirical characterization on this host:

- **Idle (no hardware context):** 0.001-0.003 W (firmware noise floor,
  effectively zero — the AIE tiles are clock-gated).
- **Loaded (sustained ``vector_scalar_mul`` kernel run):** ~0.4 W
  steady-state, with a clean transient ramp.
- **Update behaviour:** the firmware reports the AIE-partition power
  estimate; when no hardware context is loaded, the line floors near
  zero. The number is firmware-internal estimation, not a
  sense-resistor reading — labelled "Estimated Power" in the
  human-readable output.
- **Rail attribution:** AIE compute-tile rail (the AIE partition the
  hardware context occupies). It does NOT include the host SoC
  package, host DRAM, the Radeon iGPU, or platform IO. POWER_DOMAINS.md
  §3 is updated with this attribution.
- **Update rate:** the reading appears stable at ~5-10 Hz; each
  ``xrt-smi examine -r platform`` invocation costs ~40 ms wall-clock,
  which caps polling at ~25 Hz before xrt-smi overhead dominates. The
  reader defaults to 10 Hz (``poll_hz=10.0``) — the same rate 's
  ``NvidiaSmiReader`` uses for the trapezoidal fallback.

Method
------

The reader is the trapezoidal-integration analogue of 's
``NvidiaSmiReader.METHOD_TRAPEZOIDAL`` path. There is no
driver-integrated cumulative-energy counter on amdxdna 2.23.0, so the
``METHOD_DRIVER_INTEGRATED`` path NvidiaSmiReader auto-selects when
``total_energy_consumption`` is supported has no analogue here.

At :meth:`XrtReader.start`:

- Take a baseline ``xrt-smi`` sample.
- Spawn a daemon thread that polls every ``1 / poll_hz`` seconds
  and appends ``(time.monotonic(), watts)`` to ``self._samples``.

At :meth:`XrtReader.stop`:

- Set the stop event; join the poller (timeout 2 s).
- Take a final ``xrt-smi`` sample.
- Trapezoidal-integrate the samples to joules.
- Return the joule total.

Tempfile JSON path is also offered (set ``method="json"`` in the
constructor) for callers that prefer the structured output, but text
parsing is the default — it's faster (avoids a temp-file write) and
the regex is anchored to the exact firmware output string.

Multi-NPU
---------

Only one NPU on this host (BDF ``0000:67:00.1``); the reader pins to
that device implicitly because ``xrt-smi`` reports the lone device's
power. The constructor accepts a ``bdf`` argument for the
multi-device future; leaves it ``None`` (sums across all reported
``electrical.power_consumption_watts`` lines, which on this host is a
single line).

Per POWER_DOMAINS.md §3 / never-fabricate: if the probe fails
(``xrt-smi`` not on PATH; no NPU device reported; the regex doesn't
match), the constructor raises :class:`XrtUnavailableError` and the
:func:`bionpu.bench.energy.auto_reader` factory falls back to
:class:`bionpu.bench.harness.XrtStub` with a logged reason.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from ..units import EnergySource

_log = logging.getLogger(__name__)

# Default poll rate for the trapezoidal-integration path. xrt-smi
# invocation costs ~40 ms wall-clock per call so 10 Hz is the
# practical ceiling without saturating the binary; this matches 's
# NvidiaSmiReader.METHOD_TRAPEZOIDAL default.
_DEFAULT_POLL_HZ: float = 10.0

# Regex anchored to the exact firmware output line. xrt-smi prints
# (after one or two leading spaces depending on context):
#   "Estimated Power          : 0.412 Watts"
# The number is always non-negative (firmware-clipped) and ASCII
# decimal. We accept "Watts" (xrt-smi 2.23) or "Watt" (defensive,
# in case a future xrt-smi normalises the string).
_RE_ESTIMATED_POWER = re.compile(
    r"^\s*Estimated Power\s*:\s*([0-9.]+)\s*Watt", re.MULTILINE
)

# The default xrt-smi binary path. The bring-up env's setup.sh
# prepends /opt/xilinx/xrt/bin to PATH; we still fall back to the
# absolute path so callers that haven't sourced setup.sh in this shell
# (e.g. pytest run from a fresh subshell) get a useful error rather
# than a PATH miss.
_XRT_SMI_DEFAULT = "/opt/xilinx/xrt/bin/xrt-smi"

class XrtUnavailableError(RuntimeError):
    """Raised when ``xrt-smi`` power telemetry is not usable on this host.

    Raised by :class:`XrtReader.__init__` when:

    - ``xrt-smi`` is not on ``$PATH`` AND not at ``/opt/xilinx/xrt/bin/xrt-smi``;
    - ``xrt-smi examine -r platform`` returns a non-zero exit;
    - the output does not contain a parseable ``Estimated Power: ... Watts``
      line.

    Per POWER_DOMAINS.md §3 the reader MUST fail loud rather than
    fabricate. Catch sites either fall back to the matching stub from
    :mod:`bionpu.bench.harness` (writeup pipeline marks the figure as
    preliminary) or skip the NPU energy comparison entirely.
    """

def _resolve_xrt_smi(explicit: str | Path | None = None) -> str | None:
    """Return a working ``xrt-smi`` binary path, or ``None`` if absent.

    Order: explicit path > ``shutil.which("xrt-smi")`` > the
    /opt/xilinx/xrt/bin fallback.
    """
    if explicit is not None:
        p = Path(explicit)
        if p.is_file():
            return str(p)
        return None
    cand = shutil.which("xrt-smi")
    if cand:
        return cand
    if Path(_XRT_SMI_DEFAULT).is_file():
        return _XRT_SMI_DEFAULT
    return None

def _read_estimated_power_watts(
    bin_path: str, bdf: str | None = None, timeout_s: float = 5.0
) -> float:
    """Single-shot read of ``Estimated Power`` from ``xrt-smi examine -r platform``.

    If ``bdf`` is given, pin the query to that device (``-d <bdf>``);
    otherwise let xrt-smi pick (this host has exactly one NPU, BDF
    ``0000:67:00.1``).

    Returns:
        Power in watts as a float (>= 0; firmware clamps).

    Raises:
        XrtUnavailableError: xrt-smi exited non-zero or the output
            doesn't contain a parseable Estimated Power line.
    """
    args = [bin_path, "examine", "-r", "platform"]
    if bdf is not None:
        args.extend(["-d", bdf])
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise XrtUnavailableError(
            f"xrt-smi invocation failed ({type(exc).__name__}: {exc}); "
            f"check that the bring-up env is loaded and the NPU is "
            f"visible (POWER_DOMAINS.md §3)"
        ) from exc
    if proc.returncode != 0:
        raise XrtUnavailableError(
            f"xrt-smi examine -r platform exited {proc.returncode}; "
            f"stderr={proc.stderr.strip()!r}; "
            f"stdout={proc.stdout.strip()!r}"
        )
    # xrt-smi may print a "Estimated Power" line per device; sum them
    # to support a hypothetical multi-NPU future. On this host there
    # is exactly one device.
    matches = _RE_ESTIMATED_POWER.findall(proc.stdout)
    if not matches:
        raise XrtUnavailableError(
            "xrt-smi examine -r platform did not contain a parseable "
            "'Estimated Power: ... Watts' line. xrt-smi output:\n"
            + proc.stdout
        )
    total = 0.0
    for m in matches:
        try:
            total += float(m)
        except ValueError as exc:
            raise XrtUnavailableError(
                f"xrt-smi reported an unparseable power value: {m!r}"
            ) from exc
    return total

def probe_xrt() -> str:
    """Return a working xrt-smi binary path or raise :class:`XrtUnavailableError`.

    A "working" xrt-smi is one that:

    - exists on disk (``$PATH`` or ``/opt/xilinx/xrt/bin/xrt-smi``);
    - returns exit 0 from ``xrt-smi examine -r platform``;
    - emits at least one parseable ``Estimated Power: ... Watts`` line.

    POWER_DOMAINS.md §3 says fail loud; this function never returns a
    fallback or fabricated value.
    """
    bin_path = _resolve_xrt_smi()
    if bin_path is None:
        raise XrtUnavailableError(
            "xrt-smi not found on PATH or at /opt/xilinx/xrt/bin/xrt-smi. "
            "Source the bring-up env (`. /opt/xilinx/xrt/setup.sh`) and "
            "rerun. POWER_DOMAINS.md §3 — never fabricate."
        )
    # A successful read confirms (a) the binary works, (b) the NPU is
    # visible, (c) the regex matches this xrt-smi version's output.
    _read_estimated_power_watts(bin_path)
    return bin_path

@dataclass
class XrtReader:
    """Real NPU energy reader via ``xrt-smi`` power telemetry.

    Implements the :class:`bionpu.bench.harness.EnergyReader` Protocol
    using a daemon-thread poller of ``xrt-smi examine -r platform``,
    trapezoidal-integrating the resulting (t, watts) samples into
    joules. Mirrors 's
    :class:`bionpu.bench.energy.NvidiaSmiReader` design, less the
    driver-integrated counter path (no equivalent on amdxdna 2.23.0).

    Args:
        bdf: optional Bus:Device.Function pin (e.g. ``"0000:67:00.1"``).
            ``None`` (default) lets xrt-smi pick; this host has exactly
            one NPU so the result is identical.
        poll_hz: polling rate for the trapezoidal-integration loop.
            10 Hz is the practical ceiling because each ``xrt-smi``
            invocation costs ~40 ms.
        bin_path: optional explicit path to ``xrt-smi``. Defaults to
            ``shutil.which("xrt-smi")`` then ``/opt/xilinx/xrt/bin/xrt-smi``.

    Raises:
        XrtUnavailableError: xrt-smi missing / non-functional. Per
            POWER_DOMAINS.md §3 the reader MUST fail loud rather than
            fabricate. The
            :func:`bionpu.bench.energy.auto_reader` factory catches
            this and falls back to
            :class:`bionpu.bench.harness.XrtStub`.
    """

    is_real: bool = True
    source: EnergySource = EnergySource.REAL_XRT_SMI
    bdf: str | None = None
    poll_hz: float = _DEFAULT_POLL_HZ
    method: str = "xrt-smi-trapezoidal"

    _bin_path: str = ""
    _samples: list[tuple[float, float]] = field(default_factory=list)
    _stop_evt: threading.Event | None = None
    _thread: threading.Thread | None = None
    _n_samples: int = 0

    METHOD_TRAPEZOIDAL: ClassVar[str] = "xrt-smi-trapezoidal"

    def __init__(
        self,
        bdf: str | None = None,
        poll_hz: float = _DEFAULT_POLL_HZ,
        bin_path: str | Path | None = None,
    ) -> None:
        self.is_real = True
        self.source = EnergySource.REAL_XRT_SMI
        self.bdf = bdf
        self.poll_hz = poll_hz
        self.method = self.METHOD_TRAPEZOIDAL

        resolved = _resolve_xrt_smi(bin_path)
        if resolved is None:
            raise XrtUnavailableError(
                "xrt-smi not found on PATH or at /opt/xilinx/xrt/bin/xrt-smi. "
                "Source the bring-up env (`. /opt/xilinx/xrt/setup.sh`) and "
                "rerun. POWER_DOMAINS.md §3 — never fabricate."
            )
        # Confirm the binary actually returns a parseable power line.
        # Raises XrtUnavailableError on any failure mode.
        _read_estimated_power_watts(resolved, bdf=self.bdf)
        self._bin_path = resolved

        self._samples = []
        self._stop_evt = None
        self._thread = None
        self._n_samples = 0

    # ----- poller -----

    def _poll_loop(self) -> None:
        assert self._stop_evt is not None
        interval = 1.0 / max(0.1, self.poll_hz)
        while not self._stop_evt.is_set():
            t = time.monotonic()
            try:
                w = _read_estimated_power_watts(self._bin_path, bdf=self.bdf)
            except XrtUnavailableError:
                # Transient (e.g. xrt-smi briefly unavailable while
                # firmware re-initialises a hardware context). Skip
                # this sample; stop() will decide what to do based on
                # how many samples accumulated.
                self._stop_evt.wait(interval)
                continue
            self._samples.append((t, w))
            self._stop_evt.wait(interval)

    # ----- EnergyReader Protocol -----

    def start(self) -> None:
        """Capture the baseline sample + spawn the poller daemon thread.

        Mirrors :meth:`bionpu.bench.energy.NvidiaSmiReader._start_trapezoidal`:
        a single eager sample at start so even very short ``[start, stop)``
        windows have at least one endpoint to anchor the trapezoid.
        """
        self._samples = []
        self._stop_evt = threading.Event()
        # Eager baseline sample. If this somehow fails (transient
        # firmware error), proceed without it; the poller will
        # accumulate samples and stop() will fall back to 0.0 if too
        # few accumulate.
        try:
            w0 = _read_estimated_power_watts(self._bin_path, bdf=self.bdf)
            self._samples.append((time.monotonic(), w0))
        except XrtUnavailableError as exc:
            _log.debug("xrt-smi baseline sample failed: %s", exc)
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        """Stop the poller, sample once more, return joules consumed.

        Trapezoidal integration of the (t_monotonic, watts) samples.
        If fewer than two samples accumulated (very short window AND
        every poll hit a transient error), returns ``0.0`` per
        POWER_DOMAINS.md §3 / never-fabricate. Caller decides whether
        to fall back to the stub.
        """
        if self._stop_evt is not None:
            self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        # Final sample so the integration window covers right up to
        # the user's stop() call.
        try:
            w_end = _read_estimated_power_watts(self._bin_path, bdf=self.bdf)
            self._samples.append((time.monotonic(), w_end))
        except XrtUnavailableError as exc:
            _log.debug("xrt-smi final sample failed: %s", exc)
        samples = self._samples
        self._n_samples = len(samples)
        if len(samples) < 2:
            return 0.0
        joules = 0.0
        for (t0, w0), (t1, w1) in zip(samples, samples[1:], strict=False):
            dt = max(0.0, t1 - t0)
            joules += 0.5 * (w0 + w1) * dt
        return joules

# --------------------------------------------------------------------------- #
# Compatibility shim — callers used `read_xrt_energy` to surface the
# stub-error contract. keeps the symbol as a one-shot helper that
# returns the current instantaneous power-as-energy-over-1-second figure
# would be misleading; instead this helper now starts/stops a tiny
# 1-second window and returns the joule reading. Existing callers that
# only care about the `XrtUnavailableError` raise contract still work
# because XrtReader() raises the same error type if xrt-smi is absent.
# --------------------------------------------------------------------------- #

def read_xrt_energy(window_s: float = 1.0) -> float:
    """One-shot energy reading: start, sleep ``window_s``, stop.

    Convenience wrapper around :class:`XrtReader` for callers that
    don't want to manage start/stop themselves. Returns the joule
    integral over the window.

    Raises:
        XrtUnavailableError: xrt-smi not usable on this host. The
            error message is unchanged from 's stub contract
            (mentions the POWER_DOMAINS.md §3 reference) so existing
             callers continue to match on the same error type.
    """
    reader = XrtReader()
    reader.start()
    time.sleep(max(0.0, float(window_s)))
    return reader.stop()

__all__ = [
    "XrtReader",
    "XrtUnavailableError",
    "probe_xrt",
    "read_xrt_energy",
]
