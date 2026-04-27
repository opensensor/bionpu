"""Byte-equality harness — basecalling FASTQ comparator tests.

GPL-3.0. (c) 2026 OpenSensor.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from bionpu.verify.basecalling import compare_against_dorado


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FASTQ_A = (
    "@read_001 runid=abc\n"
    "ACGTACGTACGT\n"
    "+\n"
    "############\n"
    "@read_002 runid=abc\n"
    "TTTTAAAAGGGG\n"
    "+\n"
    "############\n"
)
FASTQ_A_REORDERED = (
    "@read_002 runid=abc\n"
    "TTTTAAAAGGGG\n"
    "+\n"
    "############\n"
    "@read_001 runid=abc\n"
    "ACGTACGTACGT\n"
    "+\n"
    "############\n"
)
FASTQ_A_CRLF = FASTQ_A.replace("\n", "\r\n")
FASTQ_B_SEQUENCE_DIFFERS = (
    "@read_001 runid=abc\n"
    "ACGTACGTACGT\n"
    "+\n"
    "############\n"
    "@read_002 runid=abc\n"
    "TTTTAAAACGGG\n"   # sequence differs at position 8
    "+\n"
    "############\n"
)
FASTQ_B_QUALITY_DIFFERS = (
    "@read_001 runid=abc\n"
    "ACGTACGTACGT\n"
    "+\n"
    "$$$$$$$$$$$$\n"   # quality differs
    "@read_002 runid=abc\n"
    "TTTTAAAAGGGG\n"
    "+\n"
    "############\n"
)
FASTQ_B_HEADER_DIFFERS = (
    "@read_001 runid=def\n"   # runid differs (read_id same)
    "ACGTACGTACGT\n"
    "+\n"
    "############\n"
    "@read_002 runid=abc\n"
    "TTTTAAAAGGGG\n"
    "+\n"
    "############\n"
)


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# Equality tests
# ---------------------------------------------------------------------------

def test_byte_equal_when_identical_input(tmp_path: Path) -> None:
    a = _write(tmp_path, "a.fastq", FASTQ_A)
    b = _write(tmp_path, "b.fastq", FASTQ_A)
    result = compare_against_dorado(a, b)
    assert result.equal is True
    assert bool(result) is True
    assert result.npu_sha256 == result.ref_sha256
    assert result.record_count == 2


def test_byte_equal_after_normalising_read_order(tmp_path: Path) -> None:
    """Read order in the input doesn't affect byte-equality."""
    a = _write(tmp_path, "a.fastq", FASTQ_A)
    b = _write(tmp_path, "b.fastq", FASTQ_A_REORDERED)
    result = compare_against_dorado(a, b)
    assert result.equal is True


def test_byte_equal_normalises_crlf(tmp_path: Path) -> None:
    """CRLF-terminated FASTQs compare equal to LF-terminated FASTQs."""
    a = _write(tmp_path, "a.fastq", FASTQ_A)
    b = _write(tmp_path, "b.fastq", FASTQ_A_CRLF)
    result = compare_against_dorado(a, b)
    assert result.equal is True


# ---------------------------------------------------------------------------
# Divergence tests
# ---------------------------------------------------------------------------

def test_sequence_difference_reports_actionable_message(tmp_path: Path) -> None:
    a = _write(tmp_path, "a.fastq", FASTQ_A)
    b = _write(tmp_path, "b.fastq", FASTQ_B_SEQUENCE_DIFFERS)
    result = compare_against_dorado(a, b)
    assert result.equal is False
    assert len(result.divergences) >= 1
    div = result.divergences[0]
    assert div.record_index == 1  # read_002 sorts after read_001
    assert "sequence differs" in div.message
    assert "read_002" in div.message


def test_quality_difference_reports_actionable_message(tmp_path: Path) -> None:
    a = _write(tmp_path, "a.fastq", FASTQ_A)
    b = _write(tmp_path, "b.fastq", FASTQ_B_QUALITY_DIFFERS)
    result = compare_against_dorado(a, b)
    assert result.equal is False
    div = result.divergences[0]
    assert "quality string differs" in div.message


def test_header_difference_reports_actionable_message(tmp_path: Path) -> None:
    a = _write(tmp_path, "a.fastq", FASTQ_A)
    b = _write(tmp_path, "b.fastq", FASTQ_B_HEADER_DIFFERS)
    result = compare_against_dorado(a, b)
    assert result.equal is False
    div = result.divergences[0]
    assert "header line differs" in div.message
    assert "runid=abc" in div.message
    assert "runid=def" in div.message


def test_max_divergences_caps_collection(tmp_path: Path) -> None:
    big_diff = (
        "@read_001\nAAAA\n+\n!!!!\n"
        "@read_002\nAAAA\n+\n!!!!\n"
        "@read_003\nAAAA\n+\n!!!!\n"
    )
    a_diff = (
        "@read_001\nGGGG\n+\n!!!!\n"
        "@read_002\nGGGG\n+\n!!!!\n"
        "@read_003\nGGGG\n+\n!!!!\n"
    )
    a = _write(tmp_path, "a.fastq", a_diff)
    b = _write(tmp_path, "b.fastq", big_diff)
    result = compare_against_dorado(a, b, max_divergences=2)
    assert result.equal is False
    assert len(result.divergences) == 2


def test_max_divergences_zero_skips_collection(tmp_path: Path) -> None:
    a = _write(tmp_path, "a.fastq", FASTQ_A)
    b = _write(tmp_path, "b.fastq", FASTQ_B_SEQUENCE_DIFFERS)
    result = compare_against_dorado(a, b, max_divergences=0)
    assert result.equal is False
    assert result.divergences == ()


# ---------------------------------------------------------------------------
# Malformed input
# ---------------------------------------------------------------------------

def test_malformed_fastq_raises_value_error(tmp_path: Path) -> None:
    bad = _write(tmp_path, "bad.fastq", "@read_001\nACGT\n")  # 2 lines, not 4
    good = _write(tmp_path, "good.fastq", FASTQ_A)
    with pytest.raises(ValueError, match="not a multiple of 4"):
        compare_against_dorado(bad, good)


def test_seq_qual_length_mismatch_raises(tmp_path: Path) -> None:
    bad = _write(
        tmp_path,
        "bad.fastq",
        "@read_001\nACGT\n+\n!!!\n",  # 4 quals for 4 bases would be ok, this is 3
    )
    good = _write(tmp_path, "good.fastq", FASTQ_A)
    with pytest.raises(ValueError, match="seq len .* != qual len"):
        compare_against_dorado(bad, good)
