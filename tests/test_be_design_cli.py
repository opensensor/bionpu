# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# CLI integration tests for `bionpu be design` (Track A v0).

from __future__ import annotations

import subprocess
import sys

import pytest

from bionpu.cli import main


def _write_fasta(path, name: str, seq: str) -> None:
    path.write_text(f">{name}\n{seq}\n")


def test_be_design_cli_produces_tsv(tmp_path) -> None:
    """End-to-end smoke: a synthetic target with known NGG sites + Cs."""
    # 20-nt protospacer with C at position 5 + AGG PAM 3' of it.
    # Add some flanking ACGT so the PAM oracle has a non-trivial run.
    proto = "AAAAACAAAAAAAAAAAAAA"  # C at position 5 (in CBE window)
    pam = "AGG"
    flank_5 = "ACGTACGT"  # 8 bases of flank 5'
    flank_3 = "ACGTACGT"  # 8 bases of flank 3'
    target_seq = flank_5 + proto + pam + flank_3
    target = tmp_path / "target.fa"
    _write_fasta(target, "target", target_seq)
    out = tmp_path / "guides.tsv"

    rc = main(
        [
            "be", "design",
            "--target-fasta", str(target),
            "--be-variant", "BE4max",
            "--cas9-variant", "wt",
            "--genome", "none",
            "--top", "10",
            "--output", str(out),
        ]
    )

    assert rc == 0
    text = out.read_text()
    lines = text.strip().split("\n")
    # Header + at least one guide row.
    assert lines[0].startswith("rank\tguide_seq\tpam_seq")
    assert len(lines) >= 2
    # First guide row should contain the planted protospacer.
    rows = [line.split("\t") for line in lines[1:]]
    found = False
    for row in rows:
        if row[1] == proto and row[2] == pam:
            found = True
            assert row[4] == "C"  # target_base for BE4max
            assert row[6] == "1"  # in_activity_window (C at pos 5)
            break
    assert found, f"planted guide {proto} {pam} not in output rows: {rows}"


def test_be_design_cli_abe_variant(tmp_path) -> None:
    """Same target with ABE7.10 picks As as targets, not Cs."""
    # 20-nt protospacer with A at position 4 (inside ABE7.10 window [3..6]).
    proto = "CCCCACCCCCCCCCCCCCCC"  # A at position 4
    pam = "AGG"
    flank_5 = "ACGTACGT"
    target_seq = flank_5 + proto + pam + "ACGT"
    target = tmp_path / "target.fa"
    _write_fasta(target, "target", target_seq)
    out = tmp_path / "guides.tsv"

    rc = main(
        [
            "be", "design",
            "--target-fasta", str(target),
            "--be-variant", "ABE7.10",
            "--cas9-variant", "wt",
            "--genome", "none",
            "--top", "20",
            "--output", str(out),
        ]
    )

    assert rc == 0
    text = out.read_text()
    rows = [line.split("\t") for line in text.strip().split("\n")[1:]]
    # Find the planted protospacer.
    matches = [r for r in rows if r[1] == proto and r[2] == pam]
    assert matches, f"planted guide not in output: {rows}"
    row = matches[0]
    assert row[4] == "A"  # target_base = A for ABE7.10
    assert row[6] == "1"  # in_activity_window


def test_be_design_cli_rejects_unsupported_genome(tmp_path) -> None:
    """Only --genome none is supported in v0 (synbio mode)."""
    target = tmp_path / "target.fa"
    _write_fasta(target, "target", "ACGT" * 10)
    rc = main(
        [
            "be", "design",
            "--target-fasta", str(target),
            "--be-variant", "BE4max",
            "--cas9-variant", "wt",
            "--genome", "GRCh38",
            "--output", "-",
        ]
    )
    # Should fail with rc=2 (deferred-to-v1 message).
    assert rc == 2


def test_be_design_cli_module_invocation(tmp_path) -> None:
    """Exercise the CLI via `python -m bionpu.cli` so argv routing is wired."""
    target = tmp_path / "target.fa"
    _write_fasta(target, "target", "AAAAACAAAAAAAAAAAAAA" + "AGG" + "ACGT")
    out = tmp_path / "guides.tsv"
    proc = subprocess.run(
        [
            sys.executable, "-m", "bionpu.cli",
            "be", "design",
            "--target-fasta", str(target),
            "--be-variant", "BE4max",
            "--cas9-variant", "wt",
            "--genome", "none",
            "--top", "5",
            "--output", str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert proc.returncode == 0, (
        f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    )
    text = out.read_text()
    assert text.startswith("rank\tguide_seq\tpam_seq")
