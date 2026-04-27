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


# ---------------------------------------------------------------------------
# CPU ↔ NPU cross-equivalence — locks in the byte-equality contract.
#
# When the kernel artefacts are not present, the NPU op falls back to the
# kernel's host-emulation path, which produces output byte-equal to the
# real NPU dispatch by construction (same arithmetic). The CPU and NPU
# paths therefore must produce byte-equal canonical TSVs on the same
# input — this is the load-bearing v0.2 claim.
# ---------------------------------------------------------------------------

import random


def _seeded_target_with_known_matches() -> tuple[str, list[GuideSpec]]:
    """Build a synthetic target seeded with one perfect-match and one
    1-mismatch hit on the forward strand for a single guide."""
    random.seed(2026)
    bases = "ACGT"
    guide = "AAACCCGGGTTTACGTACGT"
    prefix = "".join(random.choice(bases) for _ in range(100))
    mid = "".join(random.choice(bases) for _ in range(200))
    suffix = "".join(random.choice(bases) for _ in range(2000))
    seq = (
        prefix
        + guide + "AGG"  # exact match + AGG (NGG) at position 100
        + mid
        + "AAACCCGGGTTTACGTACGA" + "CGG"  # 1 mismatch at last spacer position
        + suffix
    )
    return seq, [GuideSpec(spacer=guide, guide_id="g_seeded")]


def test_cpu_and_npu_paths_byte_equal_on_seeded_target() -> None:
    """The headline contract: cpu_scan and npu_scan produce byte-equal output."""
    from bionpu.scan import npu_scan
    from bionpu.data.canonical_sites import normalize, serialize_canonical

    seq, guides = _seeded_target_with_known_matches()
    cpu_rows = normalize(cpu_scan(
        chrom="chr_seeded", seq=seq, guides=guides, max_mismatches=4,
    ))
    npu_rows = normalize(npu_scan(
        chrom="chr_seeded", seq=seq, guides=guides, max_mismatches=4,
    ))
    cpu_blob = serialize_canonical(cpu_rows)
    npu_blob = serialize_canonical(npu_rows)
    assert cpu_blob == npu_blob, (
        f"CPU and NPU paths diverged.\n"
        f"CPU bytes: {len(cpu_blob)}, NPU bytes: {len(npu_blob)}\n"
        f"--- CPU ---\n{cpu_blob.decode()}\n--- NPU ---\n{npu_blob.decode()}"
    )
    # Should have at least the seeded perfect + 1-mismatch match.
    assert len(cpu_rows) >= 2


def test_cpu_and_npu_paths_byte_equal_on_random_target() -> None:
    """Same byte-equality contract on a random sequence (no seeded matches).

    Looser threshold so we get hits in random data; this exercises the
    code path with many small mismatches per row.
    """
    from bionpu.scan import npu_scan
    from bionpu.data.canonical_sites import normalize, serialize_canonical

    random.seed(1234)
    bases = "ACGT"
    seq = "".join(random.choice(bases) for _ in range(5000))
    guides = [GuideSpec(spacer="A" * 20, guide_id="g_polyA")]
    cpu_rows = normalize(cpu_scan(
        chrom="chr_rand", seq=seq, guides=guides, max_mismatches=10,
    ))
    npu_rows = normalize(npu_scan(
        chrom="chr_rand", seq=seq, guides=guides, max_mismatches=10,
    ))
    assert serialize_canonical(cpu_rows) == serialize_canonical(npu_rows)
