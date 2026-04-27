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

"""Quantization passport.

Every quantized model in this repo carries a passport: calibration data
ID, accuracy degradation versus FP32, host, hardware signature, and the
SHA-256 of both the FP32 source and the quantized output. The schema
lives at ``bionpu/quant/passport.schema.json`` and is consumed verbatim
by 's basecalling INT8 sweep and 's CRISPR scoring quant.

Mirrors the ``x-deterministic: false`` annotation convention introduced
by ``bionpu/bench/schema.json``: :func:`nondeterministic_fields`
parses the schema and returns the names of fields that legitimately
differ run-to-run, so callers can diff passports modulo non-determinism.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Schema version is hard-coded here so that producers stamp the file with
# the version of the schema they were compiled against. Bump in lockstep
# with schema-shape changes; pins to a major.
SCHEMA_VERSION = "1.0.0"

_SCHEMA_PATH = Path(__file__).resolve().parent / "passport.schema.json"

@dataclass
class Passport:
    """Per-model quantization passport.

    Field semantics (and which are non-deterministic) are defined by
    ``passport.schema.json``. Keep this dataclass and the schema in
    lockstep — the round-trip ``write`` / ``read`` path validates against
    the schema.
    """

    model_name: str
    model_sha256_fp32: str
    model_sha256_quant: str
    strategy: str
    precision: str
    calibration_data_id: str
    n_samples: int
    accuracy_metrics: dict[str, float]
    degradation_vs_fp32: dict[str, float]
    host: str
    generated_at_iso8601: str
    hardware_signature: str
    schema_version: str = field(default=SCHEMA_VERSION)

# --------------------------------------------------------------------------- #
# Schema helpers
# --------------------------------------------------------------------------- #

def _load_schema() -> dict[str, Any]:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))

def nondeterministic_fields() -> set[str]:
    """Return the set of field names marked ``x-deterministic: false``.

    Mirrors the helper convention used by the bench harness: callers can
    drop these before comparing two passports for byte-equality of the
    deterministic content.
    """
    schema = _load_schema()
    out: set[str] = set()
    for name, prop in schema.get("properties", {}).items():
        if isinstance(prop, dict) and prop.get("x-deterministic") is False:
            out.add(name)
    return out

# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #

def sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    """Hex SHA-256 of a file. Used to fill ``model_sha256_*`` fields."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #

def write(passport: Passport, path: Path) -> None:
    """Serialize a passport to JSON and validate against the schema.

    Pretty-printed with ``sort_keys=True`` so two passports with
    identical content produce byte-identical files (modulo
    non-deterministic field values, which are part of the content).
    """
    data = asdict(passport)
    schema = _load_schema()
    # Local import: schema validation is the only jsonschema use here.
    import jsonschema

    jsonschema.validate(data, schema)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

def read(path: Path) -> Passport:
    """Load + validate a passport from JSON."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = _load_schema()
    import jsonschema

    jsonschema.validate(raw, schema)
    return Passport(**raw)

# --------------------------------------------------------------------------- #
# Diffing
# --------------------------------------------------------------------------- #

def _dict_diff(
    a: dict[str, float], b: dict[str, float]
) -> dict[str, float]:
    """Return entries where ``b - a`` is non-zero, keyed by metric name.

    Keys present in only one side appear with the lone value (positive
    if added in B, negative if removed in B).
    """
    out: dict[str, float] = {}
    keys = set(a) | set(b)
    for k in keys:
        if k in a and k in b:
            delta = float(b[k]) - float(a[k])
            if delta != 0.0:
                out[k] = delta
        elif k in b:
            out[k] = float(b[k])
        else:
            out[k] = -float(a[k])
    return out

def diff(passport_a: Passport, passport_b: Passport) -> dict[str, Any]:
    """Compare two passports; return per-metric deltas plus scalar deltas.

    Used by 's sweep to summarize how a config change moves accuracy.
    Numeric fields produce a single delta; nested ``accuracy_metrics`` /
    ``degradation_vs_fp32`` produce dicts of per-key deltas. Fields that
    are byte-equal are omitted.
    """
    out: dict[str, Any] = {}

    am = _dict_diff(passport_a.accuracy_metrics, passport_b.accuracy_metrics)
    if am:
        out["accuracy_metrics"] = am

    dm = _dict_diff(
        passport_a.degradation_vs_fp32, passport_b.degradation_vs_fp32
    )
    if dm:
        out["degradation_vs_fp32"] = dm

    if passport_a.n_samples != passport_b.n_samples:
        out["n_samples"] = passport_b.n_samples - passport_a.n_samples

    # Categorical/string fields: report the pair on mismatch so a human
    # reviewer can see what flipped.
    for fld in (
        "model_name",
        "model_sha256_fp32",
        "model_sha256_quant",
        "strategy",
        "precision",
        "calibration_data_id",
        "hardware_signature",
        "schema_version",
    ):
        va = getattr(passport_a, fld)
        vb = getattr(passport_b, fld)
        if va != vb:
            out[fld] = {"a": va, "b": vb}

    return out
