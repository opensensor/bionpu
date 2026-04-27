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

"""bench_all runner — orchestrates timed runs across (op, device) pairs.

Writes `results/<track>/<run_id>/measurements.json` per the schema in
`bionpu/bench/schema.json`. Used by the `bionpu bench --all` CLI subcommand
and by track-specific bench scripts.
"""

from __future__ import annotations

import json
import secrets
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .harness import MeasurementRun, TimedRun

SCHEMA_VERSION = "1.0.0"

def _make_run_id() -> str:
    """ISO-8601 timestamp (microsecond precision) + 4-char hex suffix.

    Microsecond precision plus the hex suffix means two `bench_all`
    invocations always produce distinct run_ids. The hex suffix alone is the
    uniqueness guarantee; the timestamp prefix is for sortability.
    """
    now = datetime.now(UTC)
    ts = now.strftime("%Y%m%dT%H%M%S") + f"{now.microsecond:06d}Z"
    suffix = secrets.token_hex(2)  # 4 hex chars
    return f"{ts}-{suffix}"

def _make_timestamp() -> str:
    """ISO-8601 wall-clock timestamp recorded in `measurements.json`.

    Microsecond precision so two runs within the same millisecond still
    produce distinct timestamps. (Two runs within the same microsecond would
    collide; the run_id's hex suffix is the actual uniqueness guarantee.)
    """
    now = datetime.now(UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond:06d}Z"

def _run_one(
    track: str,
    op_name: str,
    op_callable: Callable[[], Any],
    device: str,
    n_iters: int,
) -> MeasurementRun:
    """Run one (op, device) pair n_iters times under a TimedRun wrapper."""
    with TimedRun(track=track, op=op_name, device=device) as run:
        for _ in range(n_iters):
            t0 = time.monotonic_ns()
            op_callable()
            t1 = time.monotonic_ns()
            run.record_latency_ms((t1 - t0) / 1e6)
        # Default unit accounting: each iteration counts as one "sample" so
        # the harness emits a non-empty throughput field even for opaque
        # workloads. Track-specific scripts call `record_units(...)` with the
        # right kind.
        run.record_units(samples=n_iters)
    assert run.measurements is not None
    run.measurements.n_iters = n_iters
    return run.measurements

def bench_all(
    track: str,
    ops: list[tuple[str, Callable[[], Any]]],
    devices: list[str],
    n_iters: int,
    results_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Run every (op, device) pair and write `measurements.json`.

    Args:
        track: track name, e.g. ``"basecalling"``, ``"crispr"``, ``"_smoke"``.
        ops: list of ``(op_name, callable)`` pairs. Each callable takes no
            arguments and is called ``n_iters`` times.
        devices: list of device strings (subset of ``{"cpu", "gpu", "npu"}``).
             ships with stub readers regardless of which devices are
            actually present; / wire real-device probes.
        n_iters: iterations per (op, device).
        results_dir: parent directory (``results/`` by default). The file
            lands at ``<results_dir>/<track>/<run_id>/measurements.json``.

    Returns:
        A dict with ``run_id``, ``measurements_path``, and the in-memory
        ``payload`` that was written.
    """
    if results_dir is None:
        results_dir = Path("results")
    results_dir = Path(results_dir)

    run_id = _make_run_id()
    out_dir = results_dir / track / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "measurements.json"

    records: list[dict[str, Any]] = []
    for op_name, op_callable in ops:
        for device in devices:
            record = _run_one(track, op_name, op_callable, device, n_iters)
            records.append(record.to_json())

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp": _make_timestamp(),
        "track": track,
        "n_iters": n_iters,
        "records": records,
    }

    with json_path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")

    return {
        "run_id": run_id,
        "timestamp": payload["timestamp"],
        "measurements_path": str(json_path),
        "payload": payload,
    }

__all__ = ["bench_all", "SCHEMA_VERSION"]
