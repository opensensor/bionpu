"""Byte-equality harness — CRISPR comparator tests.

GPL-3.0. (c) 2026 OpenSensor.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from bionpu.verify.crispr import compare_against_cas_offinder


# ---------------------------------------------------------------------------
# Fixtures: minimal Cas-OFFinder-shaped TSVs
# ---------------------------------------------------------------------------

LEGACY_TSV_A = (
    "AAAAAAAAAAAAAAAAAAAANGG\tchr22\t1000\tAAAAAAAAAAAAAAAAAAAACGG\t+\t1\n"
    "AAAAAAAAAAAAAAAAAAAANGG\tchr22\t2000\tAAAAAAAAAAAAAAAAAAAATGG\t-\t2\n"
    "GGGGGGGGGGGGGGGGGGGGNGG\tchr22\t1500\tGGGGGGGGGGGGGGGGGGGGAGG\t+\t0\n"
)
# Same matches, different row order (Cas-OFFinder OpenCL backend can reorder)
LEGACY_TSV_A_REORDERED = (
    "GGGGGGGGGGGGGGGGGGGGNGG\tchr22\t1500\tGGGGGGGGGGGGGGGGGGGGAGG\t+\t0\n"
    "AAAAAAAAAAAAAAAAAAAANGG\tchr22\t2000\tAAAAAAAAAAAAAAAAAAAATGG\t-\t2\n"
    "AAAAAAAAAAAAAAAAAAAANGG\tchr22\t1000\tAAAAAAAAAAAAAAAAAAAACGG\t+\t1\n"
)
# One mismatch — different row at index 1 (different position).
LEGACY_TSV_B_DIFFERENT = (
    "AAAAAAAAAAAAAAAAAAAANGG\tchr22\t1000\tAAAAAAAAAAAAAAAAAAAACGG\t+\t1\n"
    "AAAAAAAAAAAAAAAAAAAANGG\tchr22\t2001\tAAAAAAAAAAAAAAAAAAAATGG\t-\t2\n"
    "GGGGGGGGGGGGGGGGGGGGNGG\tchr22\t1500\tGGGGGGGGGGGGGGGGGGGGAGG\t+\t0\n"
)
LEGACY_HEADER_LINE = (
    "#crRNA\tChromosome\tPosition\tDNA\tDirection\tMismatches\n"
)


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(LEGACY_HEADER_LINE + content)
    return p


# ---------------------------------------------------------------------------
# Equality tests
# ---------------------------------------------------------------------------

def test_byte_equal_when_identical_input(tmp_path: Path) -> None:
    a = _write(tmp_path, "a.tsv", LEGACY_TSV_A)
    b = _write(tmp_path, "b.tsv", LEGACY_TSV_A)
    result = compare_against_cas_offinder(a, b)
    assert result.equal is True
    assert bool(result) is True
    assert result.npu_sha256 == result.ref_sha256
    assert result.record_count == 3
    assert result.divergences == ()


def test_byte_equal_after_normalising_row_order(tmp_path: Path) -> None:
    """Same matches, different OpenCL-emitted row order, should byte-equal."""
    a = _write(tmp_path, "a.tsv", LEGACY_TSV_A)
    b = _write(tmp_path, "b.tsv", LEGACY_TSV_A_REORDERED)
    result = compare_against_cas_offinder(a, b)
    assert result.equal is True
    assert result.npu_sha256 == result.ref_sha256


def test_byte_equal_idempotent_sha256(tmp_path: Path) -> None:
    """SHA-256 of the canonical form is independent of input row order."""
    a = _write(tmp_path, "a.tsv", LEGACY_TSV_A)
    b = _write(tmp_path, "b.tsv", LEGACY_TSV_A_REORDERED)
    res1 = compare_against_cas_offinder(a, b)
    res2 = compare_against_cas_offinder(b, a)
    assert res1.npu_sha256 == res2.ref_sha256
    assert res1.ref_sha256 == res2.npu_sha256


# ---------------------------------------------------------------------------
# Divergence tests
# ---------------------------------------------------------------------------

def test_divergence_reports_first_differing_row(tmp_path: Path) -> None:
    a = _write(tmp_path, "a.tsv", LEGACY_TSV_A)
    b = _write(tmp_path, "b.tsv", LEGACY_TSV_B_DIFFERENT)
    result = compare_against_cas_offinder(a, b)
    assert result.equal is False
    assert bool(result) is False
    assert result.npu_sha256 != result.ref_sha256
    # The differing row is the chr22:2000 vs chr22:2001 entry; in
    # canonical sort by (chrom, start, mismatches, guide_id, strand)
    # it appears at row index 2.
    assert len(result.divergences) >= 1
    div = result.divergences[0]
    assert div.record_index == 2
    assert b"2000" in div.npu_bytes
    assert b"2001" in div.ref_bytes
    assert "chr22:2000" in div.message
    assert "chr22:2001" in div.message


def test_divergence_caps_at_max_divergences(tmp_path: Path) -> None:
    """Many-mismatch case: divergences are capped at the requested limit."""
    a = _write(tmp_path, "a.tsv", LEGACY_TSV_A)
    # Three rows, ALL different from a
    big_diff = (
        "TTTTTTTTTTTTTTTTTTTTNGG\tchr22\t1000\tTTTTTTTTTTTTTTTTTTTTCGG\t+\t1\n"
        "TTTTTTTTTTTTTTTTTTTTNGG\tchr22\t2000\tTTTTTTTTTTTTTTTTTTTTTGG\t-\t2\n"
        "CCCCCCCCCCCCCCCCCCCCNGG\tchr22\t1500\tCCCCCCCCCCCCCCCCCCCCAGG\t+\t0\n"
    )
    b = _write(tmp_path, "b.tsv", big_diff)
    result = compare_against_cas_offinder(a, b, max_divergences=2)
    assert result.equal is False
    assert len(result.divergences) == 2


def test_max_divergences_zero_skips_collection(tmp_path: Path) -> None:
    a = _write(tmp_path, "a.tsv", LEGACY_TSV_A)
    b = _write(tmp_path, "b.tsv", LEGACY_TSV_B_DIFFERENT)
    result = compare_against_cas_offinder(a, b, max_divergences=0)
    assert result.equal is False
    assert result.divergences == ()


def test_extra_npu_row_reported_as_divergence(tmp_path: Path) -> None:
    """NPU emits a row the reference doesn't have — that's a divergence."""
    a_extra = LEGACY_TSV_A + (
        "GGGGGGGGGGGGGGGGGGGGNGG\tchr22\t9999\tGGGGGGGGGGGGGGGGGGGGAGG\t+\t3\n"
    )
    a = _write(tmp_path, "a.tsv", a_extra)
    b = _write(tmp_path, "b.tsv", LEGACY_TSV_A)
    result = compare_against_cas_offinder(a, b)
    assert result.equal is False
    # The extra NPU row reports as "reference ends before NPU output"
    assert any(
        "reference ends before NPU" in d.message for d in result.divergences
    )


# ---------------------------------------------------------------------------
# Hash determinism
# ---------------------------------------------------------------------------

def test_sha256_known_value_for_canonical_fixture(tmp_path: Path) -> None:
    """Pin the canonical SHA-256 of LEGACY_TSV_A so a regression in the
    canonicaliser's wire format trips this test loudly.
    """
    a = _write(tmp_path, "a.tsv", LEGACY_TSV_A)
    b = _write(tmp_path, "b.tsv", LEGACY_TSV_A)
    result = compare_against_cas_offinder(a, b)
    # Expected SHA computed by serializing LEGACY_TSV_A through the canonicaliser
    # at the time this test was authored. If the canonical wire format changes,
    # update the comparator's docs and this constant in lockstep.
    expected_canonical_blob = (
        "guide_id\tbulge_type\tcrrna\tdna\tchrom\tstart\tstrand\tmismatches\tbulge_size\n"
        "AAAAAAAAAAAAAAAAAAAANGG\tX\tAAAAAAAAAAAAAAAAAAAANGG\tAAAAAAAAAAAAAAAAAAAACGG\tchr22\t1000\t+\t1\t0\n"
        "GGGGGGGGGGGGGGGGGGGGNGG\tX\tGGGGGGGGGGGGGGGGGGGGNGG\tGGGGGGGGGGGGGGGGGGGGAGG\tchr22\t1500\t+\t0\t0\n"
        "AAAAAAAAAAAAAAAAAAAANGG\tX\tAAAAAAAAAAAAAAAAAAAANGG\tAAAAAAAAAAAAAAAAAAAATGG\tchr22\t2000\t-\t2\t0\n"
    ).encode("utf-8")
    expected_sha = hashlib.sha256(expected_canonical_blob).hexdigest()
    assert result.npu_sha256 == expected_sha
