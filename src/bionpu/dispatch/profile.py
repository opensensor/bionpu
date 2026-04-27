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

"""Measured-latency profile table.

Per umbrella PRD §4.1: "A profile table populated by the benchmarking
harness — measured latency for each (op, shape, device) tuple — that the
dispatcher consults at compile time."

Latency unit is milliseconds (`ms`), per `bionpu/bench/UNITS.md` §4. The
JSON schema is intentionally a flat list of records so it's
diff-friendly and easy to merge across hosts. Each record carries a
`measured_at` ISO-8601 timestamp and a `host` identifier so the
provenance of a number is obvious six months from now.

"""

from __future__ import annotations

import json
import socket
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from bionpu.dispatch.devices import DEVICES, Device

# Tiebreak preference: when two devices have identical latency, this
# order picks the winner. `gpu` first because PyTorch GPU is the
# best-supported "fast" path; `cpu` second because it's always
# available; `npu` last because in v1 it's a stub anyway. Documented
# here AND in `ProfileTable.best_device.__doc__`.
_TIEBREAK_ORDER: tuple[Device, ...] = ("gpu", "cpu", "npu")

def _shape_key(shape: Sequence[int]) -> tuple[int, ...]:
    """Coerce any shape-like (list, tuple, torch.Size) to a `tuple[int, ...]`."""
    return tuple(int(d) for d in shape)

@dataclass(frozen=True)
class _Key:
    """Internal hashable key for the latency dict."""

    op: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    device: Device

@dataclass
class _Record:
    """Internal record: latency + provenance."""

    latency_ms: float
    measured_at: str  # ISO-8601 UTC
    host: str

class ProfileTable:
    """In-memory measured-latency lookup, persisted as JSON.

    Schema on disk (`save` / `load`):

        {
          "schema_version": 1,
          "records": [
            {
              "op": "conv1d",
              "input_shape": [1, 64, 1024],
              "output_shape": [1, 96, 1024],
              "device": "cpu",
              "latency_ms": 12.5,
              "measured_at": "2026-04-25T01:23:45+00:00",
              "host": "ryzen-ai-laptop"
            },
            ...
          ]
        }

    Lookups are exact-match on `(op, input_shape, output_shape, device)`.
    There is deliberately no fuzzy matching; if shapes differ even by
    one dim, treat it as "no measurement, fall back to CPU". v1 keeps
    it thin (umbrella PRD §7).
    """

    SCHEMA_VERSION = 1

    def __init__(self) -> None:
        self._data: dict[_Key, _Record] = {}

    # ------------------------------------------------------------------
    # mutation
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        op: str,
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        device: Device,
        latency_ms: float,
        measured_at: str | None = None,
        host: str | None = None,
    ) -> None:
        """Record a measured latency in milliseconds.

        Re-recording the same key overwrites the previous value (latest
        wins). `measured_at` defaults to "now" in ISO-8601 UTC. `host`
        defaults to `socket.gethostname()`.
        """
        if device not in DEVICES:
            raise ValueError(
                f"unknown device {device!r}; expected one of {list(DEVICES)}"
            )
        if not isinstance(latency_ms, (int, float)):
            raise TypeError(f"latency_ms must be a number, got {type(latency_ms).__name__}")
        if latency_ms < 0:
            raise ValueError(f"latency_ms must be non-negative, got {latency_ms}")

        key = _Key(
            op=op,
            input_shape=_shape_key(input_shape),
            output_shape=_shape_key(output_shape),
            device=device,
        )
        rec = _Record(
            latency_ms=float(latency_ms),
            measured_at=measured_at or datetime.now(UTC).isoformat(),
            host=host or socket.gethostname(),
        )
        self._data[key] = rec

    # ------------------------------------------------------------------
    # query
    # ------------------------------------------------------------------

    def lookup(
        self,
        *,
        op: str,
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        device: Device,
    ) -> float | None:
        """Return the recorded latency in ms, or `None` if not measured."""
        key = _Key(
            op=op,
            input_shape=_shape_key(input_shape),
            output_shape=_shape_key(output_shape),
            device=device,
        )
        rec = self._data.get(key)
        return rec.latency_ms if rec is not None else None

    def best_device(
        self,
        *,
        op: str,
        input_shape: Sequence[int],
        output_shape: Sequence[int],
    ) -> Device:
        """Return the device with the lowest measured latency for `(op, shapes)`.

        Tiebreak: when two or more devices are tied to within float
        equality, prefer ``gpu`` over ``cpu`` over ``npu``. This is a
        deliberate, deterministic ordering — see `_TIEBREAK_ORDER`.

        Fallback: if the table has no record for `op` at the given
        shapes, returns ``"cpu"``. CPU is always available and is the
        only device guaranteed to satisfy the dispatch contract on a
        host where bring-up hasn't completed.
        """
        in_shape = _shape_key(input_shape)
        out_shape = _shape_key(output_shape)

        candidates: list[tuple[float, int, Device]] = []
        for key, rec in self._data.items():
            if key.op != op or key.input_shape != in_shape or key.output_shape != out_shape:
                continue
            tiebreak_rank = _TIEBREAK_ORDER.index(key.device)
            candidates.append((rec.latency_ms, tiebreak_rank, key.device))

        if not candidates:
            return "cpu"

        # Sort by (latency asc, tiebreak rank asc). Ties on latency
        # resolve via the explicit preference order.
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][2]

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write the table to `path` as JSON. Overwrites any existing file."""
        records = []
        for key, rec in self._data.items():
            records.append(
                {
                    "op": key.op,
                    "input_shape": list(key.input_shape),
                    "output_shape": list(key.output_shape),
                    "device": key.device,
                    "latency_ms": rec.latency_ms,
                    "measured_at": rec.measured_at,
                    "host": rec.host,
                }
            )
        payload = {"schema_version": self.SCHEMA_VERSION, "records": records}
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> ProfileTable:
        """Load a table from a JSON file written by `save()`."""
        raw = json.loads(Path(path).read_text())
        if not isinstance(raw, dict) or "records" not in raw:
            raise ValueError(
                f"{path}: missing top-level 'records' key — not a ProfileTable file"
            )
        version = raw.get("schema_version")
        if version != cls.SCHEMA_VERSION:
            raise ValueError(
                f"{path}: schema_version {version!r}, expected {cls.SCHEMA_VERSION}"
            )

        table = cls()
        for entry in raw["records"]:
            table.record(
                op=entry["op"],
                input_shape=entry["input_shape"],
                output_shape=entry["output_shape"],
                device=entry["device"],
                latency_ms=entry["latency_ms"],
                measured_at=entry.get("measured_at"),
                host=entry.get("host"),
            )
        return table

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"ProfileTable(records={len(self._data)})"

__all__ = ["ProfileTable"]
