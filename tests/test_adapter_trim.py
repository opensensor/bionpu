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

"""Unit + silicon tests for ``bionpu trim`` (adapter_trim v0).

Coverage:

* FASTQ parser/writer round-trip and edge cases.
* Trim-loop semantics (CPU oracle path) vs cutadapt -a parity.
* Silicon byte-equal vs CPU oracle (skipped if no NPU artifacts).
"""

from __future__ import annotations

import gzip
import io
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from bionpu.genomics.adapter_trim import (
    FastqRecord,
    TrimStats,
    open_fastq,
    parse_fastq,
    trim_fastq,
    write_fastq,
)
from bionpu.genomics.adapter_trim.fastq import FastqError
from bionpu.genomics.adapter_trim.trimmer import trim_records


ADAPTER = "AGATCGGAAGAGC"  # TruSeq P5; len=13 (P=13)


_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[2] / "tracks" / "genomics" / "fixtures"
)


def _find_cutadapt() -> str | None:
    """Locate cutadapt either on PATH or alongside the running interpreter.

    bionpu's CI runs in the xdna-bringup ironenv where cutadapt is
    installed in the venv's ``bin/`` but not necessarily on ``$PATH``
    when pytest is invoked outside the venv's activation script.
    """
    on_path = shutil.which("cutadapt")
    if on_path is not None:
        return on_path
    sibling = Path(sys.executable).parent / "cutadapt"
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    return None


# --------------------------------------------------------------------------- #
# FASTQ parser tests
# --------------------------------------------------------------------------- #


def test_fastq_round_trip_basic():
    """parse_fastq + write_fastq round-trip preserves all 4 lines."""
    src = "@read1\nACGT\n+\nIIII\n@read2\nGGGGGGGG\n+\nJJJJJJJJ\n"
    recs = list(parse_fastq(io.StringIO(src)))
    assert len(recs) == 2
    assert recs[0] == FastqRecord("read1", "ACGT", "IIII")
    assert recs[1] == FastqRecord("read2", "GGGGGGGG", "JJJJJJJJ")

    buf = io.StringIO()
    for r in recs:
        write_fastq(buf, r)
    assert buf.getvalue() == src


def test_fastq_parser_rejects_length_mismatch():
    """Sequence/quality length mismatch is a hard error."""
    src = "@read1\nACGT\n+\nIII\n"  # qual is 3 chars, seq is 4
    with pytest.raises(FastqError, match="sequence length 4 != quality length 3"):
        list(parse_fastq(io.StringIO(src)))


def test_fastq_parser_rejects_missing_header_at():
    """Header line without leading '@' is a hard error."""
    src = "read1\nACGT\n+\nIIII\n"
    with pytest.raises(FastqError, match="expected FASTQ header"):
        list(parse_fastq(io.StringIO(src)))


def test_fastq_parser_truncated_record():
    """Truncated record (no quality line) is a hard error."""
    src = "@read1\nACGT\n+\n"
    with pytest.raises(FastqError, match="quality line missing"):
        list(parse_fastq(io.StringIO(src)))


def test_fastq_parser_handles_empty_input():
    """Empty input yields no records (no error)."""
    recs = list(parse_fastq(io.StringIO("")))
    assert recs == []


def test_fastq_parser_preserves_header_text():
    """Multi-token headers are passed through verbatim."""
    src = "@read1 description goes here\nACGT\n+\nIIII\n"
    recs = list(parse_fastq(io.StringIO(src)))
    assert recs[0].header == "read1 description goes here"


def test_fastq_gzip_round_trip(tmp_path: Path):
    """open_fastq transparently handles .gz suffix."""
    rec = FastqRecord("read1", "ACGTACGT", "IIIIIIII")
    path = tmp_path / "test.fastq.gz"
    with open_fastq(path, "w") as f:
        write_fastq(f, rec)
    # Verify it's actually gzipped on disk.
    with gzip.open(path, "rt", encoding="ascii") as f:
        text = f.read()
    assert text == "@read1\nACGTACGT\n+\nIIIIIIII\n"
    # Round-trip through open_fastq.
    with open_fastq(path, "r") as f:
        round = list(parse_fastq(f))
    assert round == [rec]


# --------------------------------------------------------------------------- #
# Trim-loop CPU semantics tests
# --------------------------------------------------------------------------- #


def test_trim_records_cpu_no_match_pass_through():
    """Reads with no adapter are passed through unchanged."""
    recs = [FastqRecord("r1", "AAAAAAAA", "IIIIIIII")]
    out = list(trim_records(iter(recs), adapter=ADAPTER))
    assert out == recs


def test_trim_records_cpu_full_match_drop_to_empty():
    """Read starting with adapter trims to empty."""
    seq = ADAPTER + "GGGGGGGGGG"
    qual = "I" * len(seq)
    recs = [FastqRecord("r1", seq, qual)]
    out = list(trim_records(iter(recs), adapter=ADAPTER))
    assert out == [FastqRecord("r1", "", "")]


def test_trim_records_cpu_inserted_adapter():
    """Adapter at known position trims at that position."""
    seq = "AAAAAAA" + ADAPTER + "GGGGGGGG"  # adapter at position 7
    qual = "I" * len(seq)
    recs = [FastqRecord("r1", seq, qual)]
    out = list(trim_records(iter(recs), adapter=ADAPTER))
    assert out == [FastqRecord("r1", "AAAAAAA", "IIIIIII")]


def test_trim_records_cpu_short_read_pass_through():
    """Read shorter than adapter cannot match; pass through."""
    short = "ACGT"  # 4 < 13
    recs = [FastqRecord("r1", short, "IIII")]
    out = list(trim_records(iter(recs), adapter=ADAPTER))
    assert out == recs


def test_trim_records_cpu_n_bases_no_match_through_n():
    """Adapter cannot straddle an N base — read with N inside adapter
    region is NOT trimmed (cutadapt -a parity)."""
    # Place an N in the middle of where the adapter would otherwise match.
    seq = "AAAAAAA" + "AGATCGGNAGAGC" + "GGGGGGGG"  # N at position 14
    qual = "I" * len(seq)
    recs = [FastqRecord("r1", seq, qual)]
    out = list(trim_records(iter(recs), adapter=ADAPTER))
    # The N breaks the match — read should be untrimmed.
    assert out[0].seq == seq
    assert out[0].qual == qual


def test_trim_records_first_match_wins():
    """When the adapter appears multiple times, the leftmost wins."""
    seq = "AA" + ADAPTER + "TT" + ADAPTER + "GG"
    qual = "I" * len(seq)
    recs = [FastqRecord("r1", seq, qual)]
    out = list(trim_records(iter(recs), adapter=ADAPTER))
    assert out[0].seq == "AA"  # trim at first match (pos 2)


def test_trim_stats_counters():
    """TrimStats accumulate correctly."""
    recs = [
        FastqRecord("r1", "AAAAAAAA", "IIIIIIII"),  # untrimmed
        FastqRecord("r2", ADAPTER + "GG", "I" * (len(ADAPTER) + 2)),  # trim to ""
        FastqRecord(
            "r3",
            "TTTT" + ADAPTER + "AAAA",
            "I" * (4 + len(ADAPTER) + 4),
        ),  # trim to TTTT
    ]
    stats = TrimStats()
    list(trim_records(iter(recs), adapter=ADAPTER, stats=stats))
    assert stats.n_reads == 3
    assert stats.n_trimmed == 2
    assert stats.n_untrimmed == 1
    assert stats.total_bases_in == 8 + 15 + 21
    assert stats.total_bases_out == 8 + 0 + 4
    assert stats.total_bases_removed == 15 + 17


def test_trim_records_invalid_adapter():
    """Non-ACGT adapters are rejected."""
    with pytest.raises(ValueError, match="non-ACGT"):
        list(trim_records(iter([]), adapter="ACGN"))


def test_trim_records_empty_adapter():
    """Empty adapter is rejected."""
    with pytest.raises(ValueError, match="adapter cannot be empty"):
        list(trim_records(iter([]), adapter=""))


# --------------------------------------------------------------------------- #
# trim_fastq end-to-end (CPU)
# --------------------------------------------------------------------------- #


def test_trim_fastq_synthetic_fixture_cpu_byte_equal_cutadapt(tmp_path: Path):
    """Synthetic 100-read fixture: bionpu trim CPU == cutadapt -a -O 13."""
    in_path = _FIXTURE_ROOT / "synthetic_reads_with_adapters.fastq"
    if not in_path.exists():
        pytest.skip(f"fixture missing: {in_path}")

    cutadapt = _find_cutadapt()
    if cutadapt is None:
        pytest.skip("cutadapt not on PATH; skipping cross-check")

    bionpu_out = tmp_path / "bionpu.fastq"
    cutadapt_out = tmp_path / "cutadapt.fastq"

    # bionpu trim (CPU oracle path: op=None).
    stats = trim_fastq(in_path, bionpu_out, adapter=ADAPTER, op=None)
    assert stats.n_reads == 100
    assert stats.n_trimmed == 50  # 50 reads have the adapter injected

    # cutadapt with full-overlap exact-match (parity for our exact-only kernel).
    proc = subprocess.run(
        [
            cutadapt,
            "-a", ADAPTER,
            "--no-indels", "-e", "0", "-O", str(len(ADAPTER)),
            "-j", "1",
            "-o", str(cutadapt_out),
            str(in_path),
        ],
        capture_output=True, text=True, check=True,
    )
    assert "no errors" in proc.stderr.lower() or proc.returncode == 0

    assert bionpu_out.read_bytes() == cutadapt_out.read_bytes(), (
        "bionpu trim CPU output differs from cutadapt -a -O 13 reference"
    )


def test_trim_fastq_gzip_round_trip(tmp_path: Path):
    """trim_fastq handles gzip input + gzip output."""
    in_gz = tmp_path / "in.fastq.gz"
    with gzip.open(in_gz, "wt", encoding="ascii") as f:
        f.write(f"@r1\n{'A' * 20}{ADAPTER}TTTT\n+\n{'I' * (20 + len(ADAPTER) + 4)}\n")

    out_gz = tmp_path / "out.fastq.gz"
    stats = trim_fastq(in_gz, out_gz, adapter=ADAPTER, op=None)
    assert stats.n_reads == 1
    assert stats.n_trimmed == 1

    with gzip.open(out_gz, "rt", encoding="ascii") as f:
        text = f.read()
    assert text == f"@r1\n{'A' * 20}\n+\n{'I' * 20}\n"


# --------------------------------------------------------------------------- #
# Silicon integration tests (skipped if no NPU artifacts).
# --------------------------------------------------------------------------- #


def _maybe_skip_if_no_npu():
    """Skip silicon tests if the primer_scan_p13_n4 artifact is missing."""
    try:
        from bionpu.kernels.genomics.primer_scan import BionpuPrimerScan
    except ImportError as exc:
        pytest.skip(f"primer_scan import failed: {exc}")
    op = BionpuPrimerScan(primer=ADAPTER, n_tiles=4)
    if not op.artifacts_present():
        pytest.skip(
            f"NPU artifacts missing for {op.name} at {op.artifact_dir}"
        )
    return op


def test_trim_fastq_silicon_byte_equal_cpu(tmp_path: Path):
    """Silicon path produces identical output to CPU oracle on the
    100-read synthetic fixture.

    This is the v0 silicon-vs-oracle byte-equal acceptance check —
    skipped automatically if the NPU artifact is absent.
    """
    in_path = _FIXTURE_ROOT / "synthetic_reads_with_adapters.fastq"
    if not in_path.exists():
        pytest.skip(f"fixture missing: {in_path}")
    op = _maybe_skip_if_no_npu()

    cpu_out = tmp_path / "cpu.fastq"
    npu_out = tmp_path / "npu.fastq"

    trim_fastq(in_path, cpu_out, adapter=ADAPTER, op=None)
    trim_fastq(in_path, npu_out, adapter=ADAPTER, op=op)

    assert cpu_out.read_bytes() == npu_out.read_bytes(), (
        "silicon (BionpuPrimerScan) output differs from CPU oracle on "
        "synthetic_reads_with_adapters.fastq"
    )


def test_trim_fastq_silicon_byte_equal_cutadapt(tmp_path: Path):
    """Silicon path matches cutadapt -a -O 13 byte-equally on the
    100-read synthetic fixture (full e2e validation)."""
    in_path = _FIXTURE_ROOT / "synthetic_reads_with_adapters.fastq"
    if not in_path.exists():
        pytest.skip(f"fixture missing: {in_path}")
    cutadapt = _find_cutadapt()
    if cutadapt is None:
        pytest.skip("cutadapt not on PATH; skipping cross-check")
    op = _maybe_skip_if_no_npu()

    npu_out = tmp_path / "npu.fastq"
    cutadapt_out = tmp_path / "cutadapt.fastq"

    trim_fastq(in_path, npu_out, adapter=ADAPTER, op=op)

    subprocess.run(
        [
            cutadapt,
            "-a", ADAPTER,
            "--no-indels", "-e", "0", "-O", str(len(ADAPTER)),
            "-j", "1",
            "-o", str(cutadapt_out),
            str(in_path),
        ],
        capture_output=True, check=True,
    )

    assert npu_out.read_bytes() == cutadapt_out.read_bytes()
