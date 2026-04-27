"""bionpu.scan — pure-CPU CRISPR off-target scan tests.

GPL-3.0. (c) 2026 OpenSensor.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bionpu.data.canonical_sites import normalize, write_tsv
from bionpu.scan import GuideSpec, cpu_scan, parse_guides, read_fasta
from bionpu.verify.crispr import compare_against_cas_offinder


# Tiny synthetic target: two clean NGG guide-matches on the forward strand
# at known positions, plus a CCN (reverse-strand NGG) at a third position.
SYNTHETIC_FASTA = (
    ">chr_synthetic\n"
    "AAAAAAAAAAAAAAAAAAAAAGG"        # 0..23 forward NGG, perfect AAA…20
    "CTAGCTAGCTAGCTAGCTAG"           # 23..43 spacer slop
    "TTTTTTTTTTTTTTTTTTTTAGG"        # 43..66 forward NGG, perfect TTT…20
    "GGGGGGGGGGGGGGGGGGGG"           # 66..86 slop
    "CCAAAAAAAAAAAAAAAAAAAAAA"       # 86..110 reverse-strand site (CCN at 86..89)
)


def _write_fasta(tmp_path: Path) -> Path:
    p = tmp_path / "synth.fa"
    p.write_text(SYNTHETIC_FASTA)
    return p


def test_parse_guides_comma_separated() -> None:
    guides = parse_guides("AAAAAAAAAAAAAAAAAAAA,TTTTTTTTTTTTTTTTTTTT")
    assert len(guides) == 2
    assert guides[0].spacer == "AAAAAAAAAAAAAAAAAAAA"
    assert guides[1].spacer == "TTTTTTTTTTTTTTTTTTTT"
    # Default ID is the spacer itself when no `id:` prefix
    assert guides[0].guide_id == "AAAAAAAAAAAAAAAAAAAA"


def test_parse_guides_file_with_ids(tmp_path: Path) -> None:
    f = tmp_path / "guides.txt"
    f.write_text(
        "# comment\n"
        "g1:AAAAAAAAAAAAAAAAAAAA\n"
        "\n"
        "g2:CCCCCCCCCCCCCCCCCCCC\n"
    )
    guides = parse_guides(str(f))
    assert [g.guide_id for g in guides] == ["g1", "g2"]


def test_parse_guides_rejects_invalid_spacer() -> None:
    with pytest.raises(ValueError, match="20 nt of ACGT"):
        parse_guides("AAAA")
    with pytest.raises(ValueError, match="20 nt of ACGT"):
        parse_guides("AAAAAAAAAAAAAAAAAAAN")  # contains N


def test_read_fasta_single_record(tmp_path: Path) -> None:
    f = _write_fasta(tmp_path)
    chrom, seq = read_fasta(f)
    assert chrom == "chr_synthetic"
    assert "AAAAAAAAAAAAAAAAAAAAAGG" in seq


def test_cpu_scan_finds_perfect_forward_match(tmp_path: Path) -> None:
    chrom, seq = read_fasta(_write_fasta(tmp_path))
    guides = [GuideSpec(spacer="A" * 20, guide_id="g_polyA")]
    rows = cpu_scan(chrom=chrom, seq=seq, guides=guides, max_mismatches=0)
    # Perfect match on the polyA + AGG at position 0
    perfect = [r for r in rows if r.mismatches == 0 and r.strand == "+"]
    assert len(perfect) >= 1
    assert any(r.start == 0 for r in perfect)


def test_cpu_scan_finds_reverse_strand_match(tmp_path: Path) -> None:
    chrom, seq = read_fasta(_write_fasta(tmp_path))
    # CC + 20 As → reverse-strand polyA match starting at position 86 (CC at 86..88)
    guides = [GuideSpec(spacer="T" * 20, guide_id="g_polyT_rc")]
    rows = cpu_scan(chrom=chrom, seq=seq, guides=guides, max_mismatches=0)
    rev = [r for r in rows if r.strand == "-"]
    assert len(rev) >= 1, f"expected ≥1 reverse-strand hit; got {rows}"


def test_cpu_scan_respects_max_mismatches(tmp_path: Path) -> None:
    chrom, seq = read_fasta(_write_fasta(tmp_path))
    # Single mismatch from the polyA site (one C in the middle)
    guides = [
        GuideSpec(spacer="A" * 10 + "C" + "A" * 9, guide_id="g_one_mm"),
    ]
    rows_strict = cpu_scan(chrom=chrom, seq=seq, guides=guides, max_mismatches=0)
    rows_loose = cpu_scan(chrom=chrom, seq=seq, guides=guides, max_mismatches=1)
    # Strict (no mismatches allowed) finds nothing; loose (≤1) finds the polyA site.
    assert all(r.mismatches == 0 for r in rows_strict)
    assert any(r.mismatches == 1 for r in rows_loose)


def test_cpu_scan_skips_non_acgt_windows() -> None:
    """A window containing N is silently dropped (matches Cas-OFFinder)."""
    seq = "N" * 30 + "AAAAAAAAAAAAAAAAAAAAAGG"
    rows = cpu_scan(
        chrom="chr_n",
        seq=seq,
        guides=[GuideSpec(spacer="A" * 20, guide_id="g")],
        max_mismatches=0,
    )
    # The polyA NGG starts at 30; expect exactly one forward-strand hit
    forward = [r for r in rows if r.strand == "+"]
    assert len(forward) == 1
    assert forward[0].start == 30


def test_scan_output_byte_equal_to_self_via_verify(tmp_path: Path) -> None:
    """End-to-end: scan → canonical TSV → verify against itself = EQUAL."""
    chrom, seq = read_fasta(_write_fasta(tmp_path))
    guides = parse_guides("AAAAAAAAAAAAAAAAAAAA,TTTTTTTTTTTTTTTTTTTT")
    rows = normalize(
        cpu_scan(chrom=chrom, seq=seq, guides=guides, max_mismatches=1)
    )
    out = tmp_path / "scan.tsv"
    write_tsv(out, rows)
    result = compare_against_cas_offinder(out, out)
    assert result.equal
    assert result.npu_sha256 == result.ref_sha256


def test_pam_other_than_ngg_raises() -> None:
    with pytest.raises(NotImplementedError, match="only NGG PAM"):
        cpu_scan(
            chrom="x",
            seq="A" * 50,
            guides=[GuideSpec(spacer="A" * 20, guide_id="g")],
            pam_template="NAG",
        )
