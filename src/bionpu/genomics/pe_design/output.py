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

"""Track B v0 — TSV/JSON output formatter for ranked pegRNA candidates
(Task T9).

The 31-column TSV schema mirrors PrimeDesign / PRIDICT 2.0 conventions
and is the canonical machine-readable output of the pegRNA design
pipeline (T8 ranker -> T9 emit -> T10 CLI write). The companion JSON
emits the same payload as a single-document list-of-dicts (mirroring
:func:`bionpu.genomics.crispr_design.format_result_json`'s
``json.dumps(..., indent=2, sort_keys=True)`` pattern; choice: SINGLE
JSON DOCUMENT, not NDJSON).

NaN-vs-None disambiguation rule
-------------------------------
The dataclass distinguishes ``float`` (possibly ``NaN``) from
``float | None`` (None for PE2 candidates lacking PE3 nicking metrics).
JSON has neither NaN nor a None-vs-NaN distinction; TSV is text-only.
We resolve via:

* TSV emit: ``None`` -> ``""``; ``NaN`` -> ``"NA"``; numeric float ->
  ``f"{x:.6g}"`` (compact roundtrip-stable representation).
* JSON emit: ``None`` -> ``null``; ``NaN`` -> ``null`` (JSON has no
  NaN); numeric float -> the float.
* JSON read-back disambiguation (per-field):
    - ``cfd_aggregate_pegrna``, ``pridict_efficiency``,
      ``pridict_edit_rate``, ``pridict_confidence``, ``mfe_kcal``,
      ``scaffold_disruption``, ``pbs_pairing_prob``, ``composite_pridict``
      are always numeric (NaN possible) -> ``null`` decodes to ``NaN``.
    - ``cfd_aggregate_nicking``, ``off_target_count_nicking``,
      ``nicking_spacer``, ``nicking_pam``, ``nicking_distance`` may be
      ``None`` for PE2 candidates and numeric (or NaN, in the case of
      ``cfd_aggregate_nicking``) for PE3. The disambiguation keys on
      ``pe_strategy``: ``PE2`` -> ``None``; ``PE3`` -> ``NaN`` for the
      float field, integer for ``off_target_count_nicking``,
      string for ``nicking_spacer`` / ``nicking_pam`` / ``nicking_distance``.
* TSV read-back: ``""`` -> ``None``; ``"NA"`` -> ``NaN``; otherwise
  parse via the dataclass field type.

Roundtrip determinism
---------------------
Float emission uses ``f"{x:.6g}"``. Reading-then-writing produces
byte-identical output as long as the original float can be represented
exactly in 6 significant digits (the test fixtures use 1-2 decimal
floats so this holds; production callers should be aware that very
high-precision floats may lose precision after one TSV roundtrip but
become stable on subsequent roundtrips because the second emit reads
and re-emits the already-truncated value).
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import IO, Iterable

from .types import RankedPegRNA

__all__ = [
    "TSV_HEADER",
    "FLOAT_FORMAT",
    "write_tsv",
    "write_json",
    "read_tsv",
    "read_json",
]


# ---------------------------------------------------------------------------
# Schema — column ordering is the canonical 31-column TSV from plan §T9.
# Sourcing this from a literal list (NOT from ``dataclasses.fields()``)
# matches the plan's documented column order one-way; the dataclass
# field order happens to match but the literal is the source of truth.
# ---------------------------------------------------------------------------

TSV_HEADER: tuple[str, ...] = (
    "pegrna_id",
    "edit_notation",
    "edit_position",
    "edit_type",
    "spacer_strand",
    "spacer_seq",
    "pam_seq",
    "scaffold_variant",
    "pbs_seq",
    "pbs_length",
    "rtt_seq",
    "rtt_length",
    "rt_product_seq",
    "nick_site",
    "full_pegrna_rna_seq",
    "pe_strategy",
    "nicking_spacer",
    "nicking_pam",
    "nicking_distance",
    "pridict_efficiency",
    "pridict_edit_rate",
    "pridict_confidence",
    "mfe_kcal",
    "scaffold_disruption",
    "pbs_pairing_prob",
    "cfd_aggregate_pegrna",
    "off_target_count_pegrna",
    "cfd_aggregate_nicking",
    "off_target_count_nicking",
    "composite_pridict",
    "rank",
    "notes",
    # v0.1 — paralog-aware split aggregate (appended at end so the
    # first 32 columns remain stable for v0 readers).
    "paralog_hit_count_pegrna",
    "cfd_aggregate_paralog_pegrna",
)

# Plan §T9 prose calls this the "31-column TSV"; the enumerated columns
# in the same section actually total 32 (note that ``rt_product_seq``
# was added per the reviewer-gap update). v0.1 appends two paralog-aware
# columns at the end so the count is now 34. The dataclass field set
# matches; we lock the count here so a future schema drift trips a
# loud assertion rather than silently emitting the wrong shape.
assert len(TSV_HEADER) == 34

FLOAT_FORMAT = ".6g"


# ---------------------------------------------------------------------------
# Per-field type registry — drives both emit and parse.
#
# Categories:
#   * "str"           : plain string (pegrna_id, edit_notation, ...).
#   * "int"           : plain int.
#   * "float"         : float; NaN possible -> "NA"/null.
#   * "opt_str"       : str | None  (None -> "" / null).
#   * "opt_int"       : int | None.
#   * "opt_float_pe3" : float | None where None means "PE2 candidate"
#                       and NaN means "PE3 with no off-target data".
#                       Disambiguated on read via ``pe_strategy``.
#   * "notes"         : tuple[str, ...] -> "," joined / JSON list.
# ---------------------------------------------------------------------------

_FIELD_TYPES: dict[str, str] = {
    "pegrna_id": "str",
    "edit_notation": "str",
    "edit_position": "int",
    "edit_type": "str",
    "spacer_strand": "str",
    "spacer_seq": "str",
    "pam_seq": "str",
    "scaffold_variant": "str",
    "pbs_seq": "str",
    "pbs_length": "int",
    "rtt_seq": "str",
    "rtt_length": "int",
    "rt_product_seq": "str",
    "nick_site": "int",
    "full_pegrna_rna_seq": "str",
    "pe_strategy": "str",
    "nicking_spacer": "opt_str",
    "nicking_pam": "opt_str",
    "nicking_distance": "opt_int",
    "pridict_efficiency": "float",
    "pridict_edit_rate": "float",
    "pridict_confidence": "float",
    "mfe_kcal": "float",
    "scaffold_disruption": "float",
    "pbs_pairing_prob": "float",
    "cfd_aggregate_pegrna": "float",
    "off_target_count_pegrna": "int",
    "cfd_aggregate_nicking": "opt_float_pe3",
    "off_target_count_nicking": "opt_int",
    "composite_pridict": "float",
    "rank": "int",
    "notes": "notes",
    # v0.1 paralog-aware split aggregate columns.
    "paralog_hit_count_pegrna": "int",
    "cfd_aggregate_paralog_pegrna": "float",
}

assert set(_FIELD_TYPES.keys()) == set(TSV_HEADER), (
    "field-type registry must cover every TSV column"
)


# ---------------------------------------------------------------------------
# Emit helpers
# ---------------------------------------------------------------------------


def _emit_tsv_cell(value: object, field_type: str) -> str:
    """Format one cell for TSV emission.

    NaN -> ``"NA"``; None -> ``""``; floats use ``FLOAT_FORMAT``.
    """
    if field_type == "notes":
        # tuple[str, ...] -> comma-joined; empty tuple -> "".
        assert isinstance(value, tuple)
        return ",".join(value)

    if value is None:
        return ""

    if field_type in ("float", "opt_float_pe3"):
        assert isinstance(value, float)
        if math.isnan(value):
            return "NA"
        return format(value, FLOAT_FORMAT)

    if field_type in ("int", "opt_int"):
        return str(value)

    # str / opt_str.
    return str(value)


def _emit_json_value(value: object, field_type: str) -> object:
    """Convert one field to a JSON-serialisable value.

    NaN -> None (JSON ``null``); tuple notes -> list. None passes
    through.
    """
    if field_type == "notes":
        assert isinstance(value, tuple)
        return list(value)

    if value is None:
        return None

    if field_type in ("float", "opt_float_pe3"):
        assert isinstance(value, float)
        if math.isnan(value):
            return None
        return value

    if field_type in ("int", "opt_int"):
        return int(value)  # type: ignore[arg-type]

    return value


# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------


def _parse_tsv_cell(
    cell: str,
    field_type: str,
    *,
    pe_strategy: str | None = None,
) -> object:
    """Parse one TSV cell into the appropriate Python value."""
    if field_type == "notes":
        return tuple(s for s in cell.split(",") if s) if cell else ()

    if cell == "":
        if field_type in ("opt_str", "opt_int", "opt_float_pe3"):
            return None
        # Empty cell on a non-optional column shouldn't happen; pass the
        # empty string through for plain strings, raise for numerics.
        if field_type == "str":
            return ""
        raise ValueError(f"empty TSV cell for non-optional field of type {field_type}")

    if cell == "NA":
        if field_type in ("float", "opt_float_pe3"):
            return float("nan")
        # "NA" appearing in a non-float column is a corruption signal.
        raise ValueError(f"unexpected NA for field type {field_type}")

    if field_type == "float":
        return float(cell)
    if field_type == "opt_float_pe3":
        # PE3 with numeric value -> float; PE3 with "NA" already
        # handled above.
        return float(cell)
    if field_type in ("int", "opt_int"):
        return int(cell)
    # str / opt_str.
    return cell


def _parse_json_value(
    value: object,
    field_type: str,
    *,
    pe_strategy: str | None = None,
) -> object:
    """Parse one JSON value into the appropriate Python value.

    JSON ``null`` is ambiguous between None and NaN; the field-level
    rule (see module docstring) keys on ``pe_strategy`` for the
    PE3-specific optional fields.
    """
    if field_type == "notes":
        if value is None:
            return ()
        assert isinstance(value, list)
        return tuple(value)

    if value is None:
        if field_type == "float":
            # Always-numeric field (NaN possible) -> NaN.
            return float("nan")
        if field_type == "opt_float_pe3":
            # PE2 -> None; PE3 -> NaN.
            if pe_strategy == "PE3":
                return float("nan")
            return None
        if field_type in ("opt_str", "opt_int"):
            # PE2 -> None always (no NaN equivalent for str/int).
            return None
        # Required str / int field with null is a corruption signal.
        raise ValueError(f"null for required field of type {field_type}")

    # Non-null path.
    if field_type in ("float", "opt_float_pe3"):
        return float(value)  # type: ignore[arg-type]
    if field_type in ("int", "opt_int"):
        return int(value)  # type: ignore[arg-type]
    return value


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _open_for_write(path_or_handle):
    """Return ``(handle, should_close)`` for path-or-handle."""
    if isinstance(path_or_handle, (str, Path)):
        return open(path_or_handle, "w", encoding="utf-8", newline=""), True
    return path_or_handle, False


def _open_for_read(path_or_handle):
    """Return ``(handle, should_close)`` for path-or-handle."""
    if isinstance(path_or_handle, (str, Path)):
        return open(path_or_handle, "r", encoding="utf-8", newline=""), True
    return path_or_handle, False


# ---------------------------------------------------------------------------
# Public API — TSV
# ---------------------------------------------------------------------------


def write_tsv(records: Iterable[RankedPegRNA], path) -> None:
    """Emit a header-led tab-separated file.

    Cells are joined by ``\\t``; rows by ``\\n``; trailing ``\\n``.
    """
    handle, should_close = _open_for_write(path)
    try:
        handle.write("\t".join(TSV_HEADER) + "\n")
        for rec in records:
            cells = [
                _emit_tsv_cell(getattr(rec, col), _FIELD_TYPES[col])
                for col in TSV_HEADER
            ]
            handle.write("\t".join(cells) + "\n")
    finally:
        if should_close:
            handle.close()


def read_tsv(path) -> list[RankedPegRNA]:
    """Read a TSV produced by :func:`write_tsv` back into RankedPegRNA."""
    handle, should_close = _open_for_read(path)
    try:
        text = handle.read()
    finally:
        if should_close:
            handle.close()

    lines = text.rstrip("\n").split("\n")
    if not lines:
        return []
    header = tuple(lines[0].split("\t"))
    # v0.1 backward-compat: v0 TSVs (32 cols, no paralog columns) read
    # through the same path; missing paralog cells default to the
    # v0-equivalent values (0 hits / 100.0 specificity). v1+ readers
    # MUST emit the full 34-col header to avoid this fallback.
    n_actual = len(header)
    n_canonical = len(TSV_HEADER)
    if header == TSV_HEADER:
        compat_v0 = False
    elif header == TSV_HEADER[:32] and n_actual == 32:
        compat_v0 = True
    else:
        raise ValueError(
            f"unexpected TSV header (got {n_actual} cols, expected "
            f"{n_canonical} or v0-compat 32); first mismatch column: "
            f"{next((a for a, b in zip(header, TSV_HEADER) if a != b), '?')}"
        )

    out: list[RankedPegRNA] = []
    for line in lines[1:]:
        cells = line.split("\t")
        if compat_v0:
            # Pad missing v0.1 paralog columns with their v0-equivalent
            # defaults: count=0, specificity=100.0.
            if len(cells) != 32:
                raise ValueError(
                    f"v0-compat row has {len(cells)} cells, expected 32"
                )
            cells = cells + ["0", "100"]
        if len(cells) != n_canonical:
            raise ValueError(
                f"row has {len(cells)} cells, expected {n_canonical}"
            )
        cell_map = dict(zip(TSV_HEADER, cells, strict=True))
        pe_strategy = cell_map["pe_strategy"]
        kwargs: dict = {}
        for col in TSV_HEADER:
            kwargs[col] = _parse_tsv_cell(
                cell_map[col],
                _FIELD_TYPES[col],
                pe_strategy=pe_strategy,
            )
        out.append(RankedPegRNA(**kwargs))
    return out


# ---------------------------------------------------------------------------
# Public API — JSON
# ---------------------------------------------------------------------------


def write_json(records: Iterable[RankedPegRNA], path) -> None:
    """Emit a single JSON document: a list of dicts (one per record).

    Mirrors :func:`bionpu.genomics.crispr_design.format_result_json`'s
    indent=2 + sort_keys=False shape (we keep ``sort_keys=False`` so the
    on-disk key order matches the documented TSV column order — this
    makes diff'ing TSV<->JSON reasonably visual). Trailing newline.
    """
    payload: list[dict] = []
    for rec in records:
        d = {
            col: _emit_json_value(getattr(rec, col), _FIELD_TYPES[col])
            for col in TSV_HEADER
        }
        payload.append(d)

    handle, should_close = _open_for_write(path)
    try:
        handle.write(json.dumps(payload, indent=2) + "\n")
    finally:
        if should_close:
            handle.close()


def read_json(path) -> list[RankedPegRNA]:
    """Read a JSON document produced by :func:`write_json`."""
    handle, should_close = _open_for_read(path)
    try:
        payload = json.load(handle)
    finally:
        if should_close:
            handle.close()

    if not isinstance(payload, list):
        raise ValueError("expected a JSON list at the document root")

    out: list[RankedPegRNA] = []
    # v0.1 backward-compat: when the JSON payload lacks the new paralog
    # columns, fall back to the v0-equivalent defaults (count=0,
    # specificity=100.0). Missing-key sentinel chosen so we can
    # distinguish "key absent (v0)" from "key present with null".
    _MISSING = object()
    for d in payload:
        if not isinstance(d, dict):
            raise ValueError("expected each list element to be a JSON object")
        pe_strategy = d.get("pe_strategy")
        if not isinstance(pe_strategy, str):
            raise ValueError("missing required string field `pe_strategy`")
        kwargs: dict = {}
        for col in TSV_HEADER:
            if col == "notes":
                raw = d.get("notes", [])
            elif col in ("paralog_hit_count_pegrna", "cfd_aggregate_paralog_pegrna"):
                raw = d.get(col, _MISSING)
                if raw is _MISSING:
                    # v0 JSON document — supply v0-equivalent defaults.
                    kwargs[col] = 0 if col == "paralog_hit_count_pegrna" else 100.0
                    continue
            else:
                raw = d.get(col, None)
            kwargs[col] = _parse_json_value(
                raw,
                _FIELD_TYPES[col],
                pe_strategy=pe_strategy,
            )
        out.append(RankedPegRNA(**kwargs))
    return out


# Silence the unused-import warning on dataclasses while keeping the
# import available for downstream type checkers / future helpers.
_ = dataclasses
_IO = IO  # re-export for type-checkers
