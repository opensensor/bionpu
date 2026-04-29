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

"""Track B v0 — Tests for the pegRNA TSV/JSON output formatter (Task T9).

Acceptance criteria (per ``track-b-pegrna-design-plan.md`` §T9):

1. TSV roundtrip: write 5 ``RankedPegRNA`` records, read back, byte-equal
   field-by-field after re-emit.
2. JSON roundtrip: write 5 ``RankedPegRNA`` records, read back, byte-equal
   after re-emit.
3. NaN handling: a record with ``cfd_aggregate_pegrna=float('nan')``
   serialises to ``"NA"`` in TSV and ``null`` in JSON; on read-back the
   value restores to ``float('nan')`` so subsequent re-emits match.

NaN-vs-None disambiguation rule (documented in ``output.py``)
-------------------------------------------------------------
* TSV: ``None`` -> empty string; ``NaN`` -> ``"NA"``; numeric ->
  ``f"{x:.6g}"``.
* JSON: ``None`` -> ``null``; ``NaN`` -> ``null`` (JSON has no NaN);
  numeric -> the float.
* JSON read-back: ``null`` is ambiguous between ``None`` and ``NaN``.
  Resolution is field-level:
    - ``cfd_aggregate_pegrna`` is always numeric (NaN possible) ->
      ``null`` decodes to ``NaN``.
    - ``cfd_aggregate_nicking`` / ``off_target_count_nicking`` are
      ``None`` for PE2 candidates; for PE3 they may be ``NaN``. The
      field-level rule keys on ``pe_strategy``: ``PE2`` -> ``None``;
      ``PE3`` + ``null`` -> ``NaN``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from bionpu.genomics.pe_design.output import (
    read_json,
    read_tsv,
    write_json,
    write_tsv,
)
from bionpu.genomics.pe_design.types import RankedPegRNA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_record(
    *,
    pegrna_id: str,
    pe_strategy: str = "PE2",
    cfd_pegrna: float = 95.0,
    notes: tuple[str, ...] = (),
    rank: int = 1,
) -> RankedPegRNA:
    """Build a synthetic ``RankedPegRNA``; PE3 fields are populated when
    ``pe_strategy == "PE3"`` and ``None`` otherwise."""
    is_pe3 = pe_strategy == "PE3"
    return RankedPegRNA(
        pegrna_id=pegrna_id,
        edit_notation="C>T at 100",
        edit_position=100,
        edit_type="substitution",
        spacer_strand="+",
        spacer_seq="ACGTACGTACGTACGTACGT",
        pam_seq="AGG",
        scaffold_variant="sgRNA_canonical",
        pbs_seq="GCAUGCAUGC",
        pbs_length=10,
        rtt_seq="ACGUACGUACGUAC",
        rtt_length=14,
        rt_product_seq="ACGTACGTACGTAC",
        nick_site=97,
        full_pegrna_rna_seq="A" * 90,
        pe_strategy=pe_strategy,
        nicking_spacer="GGGGTTTTAAAACCCCAAAA" if is_pe3 else None,
        nicking_pam="TGG" if is_pe3 else None,
        nicking_distance=60 if is_pe3 else None,
        pridict_efficiency=72.5,
        pridict_edit_rate=0.31,
        pridict_confidence=0.88,
        mfe_kcal=-32.4,
        scaffold_disruption=0.05,
        pbs_pairing_prob=0.71,
        cfd_aggregate_pegrna=cfd_pegrna,
        off_target_count_pegrna=4,
        cfd_aggregate_nicking=22.0 if is_pe3 else None,
        off_target_count_nicking=2 if is_pe3 else None,
        composite_pridict=68.4,
        rank=rank,
        notes=notes,
    )


@pytest.fixture
def five_records() -> list[RankedPegRNA]:
    """5 records with a mix of PE2 / PE3 + populated/empty notes."""
    return [
        _make_record(pegrna_id="rec1", pe_strategy="PE2", rank=1),
        _make_record(
            pegrna_id="rec2",
            pe_strategy="PE3",
            rank=2,
            notes=("SCAFFOLD_OUT_OF_DISTRIBUTION",),
        ),
        _make_record(
            pegrna_id="rec3",
            pe_strategy="PE2",
            cfd_pegrna=88.0,
            notes=("POLY_T_RUN", "SCAFFOLD_OUT_OF_DISTRIBUTION"),
            rank=3,
        ),
        _make_record(pegrna_id="rec4", pe_strategy="PE3", rank=4),
        _make_record(pegrna_id="rec5", pe_strategy="PE2", rank=5),
    ]


def _records_equal(a: RankedPegRNA, b: RankedPegRNA) -> bool:
    """Field-by-field equality with NaN-aware float comparison."""
    from dataclasses import fields

    for f in fields(a):
        va = getattr(a, f.name)
        vb = getattr(b, f.name)
        if isinstance(va, float) and isinstance(vb, float):
            if math.isnan(va) and math.isnan(vb):
                continue
            if va != vb:
                return False
        else:
            if va != vb:
                return False
    return True


# ---------------------------------------------------------------------------
# Acceptance test 1 — TSV roundtrip
# ---------------------------------------------------------------------------


def test_tsv_roundtrip(tmp_path: Path, five_records: list[RankedPegRNA]) -> None:
    """Write 5 records, read back, re-emit -> byte-equal blob."""
    p1 = tmp_path / "out1.tsv"
    write_tsv(five_records, p1)
    blob1 = p1.read_bytes()

    parsed = read_tsv(p1)
    assert len(parsed) == len(five_records)
    for orig, got in zip(five_records, parsed, strict=True):
        assert _records_equal(orig, got), f"mismatch on {orig.pegrna_id}"

    # Re-emit determinism: writing parsed records produces identical bytes.
    p2 = tmp_path / "out2.tsv"
    write_tsv(parsed, p2)
    assert p2.read_bytes() == blob1


# ---------------------------------------------------------------------------
# Acceptance test 2 — JSON roundtrip
# ---------------------------------------------------------------------------


def test_json_roundtrip(tmp_path: Path, five_records: list[RankedPegRNA]) -> None:
    """Write 5 records to JSON, read back, re-emit -> byte-equal blob."""
    p1 = tmp_path / "out1.json"
    write_json(five_records, p1)
    blob1 = p1.read_bytes()

    parsed = read_json(p1)
    assert len(parsed) == len(five_records)
    for orig, got in zip(five_records, parsed, strict=True):
        assert _records_equal(orig, got), f"mismatch on {orig.pegrna_id}"

    p2 = tmp_path / "out2.json"
    write_json(parsed, p2)
    assert p2.read_bytes() == blob1


# ---------------------------------------------------------------------------
# Acceptance test 3 — NaN handling
# ---------------------------------------------------------------------------


def test_nan_handling(tmp_path: Path) -> None:
    """``cfd_aggregate_pegrna=NaN`` -> ``"NA"`` in TSV / ``null`` in JSON;
    read-back restores ``float('nan')``; PE3 ``cfd_aggregate_nicking=NaN``
    likewise; PE2 ``cfd_aggregate_nicking=None`` stays ``None``."""
    pe2_nan = _make_record(
        pegrna_id="pe2_nan",
        pe_strategy="PE2",
        cfd_pegrna=float("nan"),
        notes=(),
    )
    pe3_nan = _make_record(
        pegrna_id="pe3_nan",
        pe_strategy="PE3",
        cfd_pegrna=float("nan"),
        notes=(),
    )
    # PE3 record with NaN nicking aggregate (rare but defined in the rule):
    pe3_nan = RankedPegRNA(
        **{
            **{
                k: v
                for k, v in pe3_nan.__dict__.items()
                if k != "cfd_aggregate_nicking"
            },
            "cfd_aggregate_nicking": float("nan"),
        }
    )

    records = [pe2_nan, pe3_nan]

    # ----- TSV -----
    tsv_path = tmp_path / "nan.tsv"
    write_tsv(records, tsv_path)
    text = tsv_path.read_text()
    lines = text.rstrip("\n").split("\n")
    header = lines[0].split("\t")
    rows = [dict(zip(header, line.split("\t"), strict=True)) for line in lines[1:]]

    assert rows[0]["cfd_aggregate_pegrna"] == "NA"
    # PE2 -> nicking columns are empty strings.
    assert rows[0]["cfd_aggregate_nicking"] == ""
    assert rows[0]["off_target_count_nicking"] == ""

    assert rows[1]["cfd_aggregate_pegrna"] == "NA"
    # PE3 with NaN nicking aggregate -> "NA".
    assert rows[1]["cfd_aggregate_nicking"] == "NA"

    # Read-back restores NaN / None correctly.
    tsv_parsed = read_tsv(tsv_path)
    assert math.isnan(tsv_parsed[0].cfd_aggregate_pegrna)
    assert tsv_parsed[0].cfd_aggregate_nicking is None
    assert tsv_parsed[0].off_target_count_nicking is None
    assert math.isnan(tsv_parsed[1].cfd_aggregate_pegrna)
    assert tsv_parsed[1].cfd_aggregate_nicking is not None
    assert math.isnan(tsv_parsed[1].cfd_aggregate_nicking)

    # ----- JSON -----
    json_path = tmp_path / "nan.json"
    write_json(records, json_path)
    payload = json.loads(json_path.read_text())
    # JSON shape: list of dicts (mirrors crispr_design.py's single-document
    # JSON; one dict per RankedPegRNA).
    assert isinstance(payload, list)
    assert payload[0]["cfd_aggregate_pegrna"] is None  # NaN -> null
    assert payload[0]["cfd_aggregate_nicking"] is None  # PE2 None -> null
    assert payload[1]["cfd_aggregate_pegrna"] is None
    assert payload[1]["cfd_aggregate_nicking"] is None  # PE3 NaN -> null

    json_parsed = read_json(json_path)
    # Field-level disambiguation:
    #   cfd_aggregate_pegrna is always numeric (NaN possible) -> NaN.
    #   cfd_aggregate_nicking on PE2 -> None; on PE3 -> NaN.
    assert math.isnan(json_parsed[0].cfd_aggregate_pegrna)
    assert json_parsed[0].cfd_aggregate_nicking is None
    assert json_parsed[0].off_target_count_nicking is None
    assert math.isnan(json_parsed[1].cfd_aggregate_pegrna)
    assert json_parsed[1].cfd_aggregate_nicking is not None
    assert math.isnan(json_parsed[1].cfd_aggregate_nicking)
