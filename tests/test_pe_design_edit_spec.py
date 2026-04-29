# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

"""Track B v0 — Tests for the T2 edit-spec parser.

Covers the acceptance criteria for Task T2 of
``track-b-pegrna-design-plan.md``:

* Simple notation: substitution, multi-base substitution, insertion, deletion.
* HGVS notation: substitution, delins, dup, deletion.
* Transcript-relative HGVS resolution (``NM_xxx:c.NNNN...``) via the
  shipped ``bionpu.data.genome_fetcher`` refGene subset.
* Minus-strand HGVS test (``NM_007294.4:c.5266dupC`` on BRCA1) — the
  resulting ``EditSpec.alt_seq`` must be the genome ``+``-strand
  complement of the HGVS alt allele, and ``strand`` must be ``"-"``.
* Error paths: ``RefMismatchError``, ``EditTooLargeError``,
  ``UnsupportedHGVSVariant``.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# --------------------------------------------------------------------- #
# Fixtures: synthetic FASTA so ref-allele validation can be exercised.
# --------------------------------------------------------------------- #


def _write_fasta(path: Path, chrom: str, seq: str) -> None:
    """Write a multi-line FASTA the slice helper can stream through."""
    with path.open("w", encoding="ascii") as fh:
        fh.write(f">{chrom}\n")
        for i in range(0, len(seq), 80):
            fh.write(seq[i : i + 80] + "\n")


@pytest.fixture
def synthetic_chr1_fasta(tmp_path: Path) -> Path:
    """A small synthetic chr1 with a known A-block + ``CGTA`` marker
    at position 100 (1-based) so simple-notation tests can validate
    ref alleles against a real FASTA."""
    bases = ["N"] * 200
    # 1-based position 100 -> 0-based index 99; deposit "CGTA..." here.
    marker = "CGTACGTACGTAC"  # 13 bp at positions 100-112 (1-based)
    for i, b in enumerate(marker):
        bases[99 + i] = b
    # Surround with A's so flanks are well-defined.
    for i in range(len(bases)):
        if bases[i] == "N":
            bases[i] = "A"
    fasta = tmp_path / "synth_chr1.fa"
    _write_fasta(fasta, "chr1", "".join(bases))
    return fasta


# --------------------------------------------------------------------- #
# Module surface
# --------------------------------------------------------------------- #


def test_module_exports_parse_edit_spec_and_errors() -> None:
    """Public surface check: parse_edit_spec + the three exception types
    must be importable from ``bionpu.genomics.pe_design.edit_spec``."""
    from bionpu.genomics.pe_design import edit_spec as es

    assert callable(es.parse_edit_spec)
    assert issubclass(es.RefMismatchError, Exception)
    assert issubclass(es.EditTooLargeError, Exception)
    assert issubclass(es.UnsupportedHGVSVariant, Exception)


# --------------------------------------------------------------------- #
# Simple notation tests
# --------------------------------------------------------------------- #


def test_simple_substitution_single_base(synthetic_chr1_fasta: Path) -> None:
    """``"C>T at 100"`` resolves to a 1-bp substitution at 1-based pos
    100 with strand ``+`` and validates against the FASTA's base at that
    position (``C``)."""
    from bionpu.genomics.pe_design.edit_spec import parse_edit_spec

    spec = parse_edit_spec(
        "C>T at chr1:100",
        genome_path=synthetic_chr1_fasta,
    )
    assert spec.chrom == "chr1"
    # 0-indexed inclusive start, exclusive end.
    assert spec.start == 99
    assert spec.end == 100
    assert spec.ref_seq == "C"
    assert spec.alt_seq == "T"
    assert spec.edit_type == "substitution"
    assert spec.strand == "+"
    assert "C>T" in spec.notation_used


def test_simple_multi_base_substitution(synthetic_chr1_fasta: Path) -> None:
    """``"CGT>AAA at chr1:100..102"`` resolves to a 3-bp delins."""
    from bionpu.genomics.pe_design.edit_spec import parse_edit_spec

    spec = parse_edit_spec(
        "CGT>AAA at chr1:100..102",
        genome_path=synthetic_chr1_fasta,
    )
    assert spec.start == 99
    assert spec.end == 102
    assert spec.ref_seq == "CGT"
    assert spec.alt_seq == "AAA"
    assert spec.edit_type == "substitution"


def test_simple_insertion(synthetic_chr1_fasta: Path) -> None:
    """``"insAGT at chr1:100"`` is a pure insertion BEFORE position 100."""
    from bionpu.genomics.pe_design.edit_spec import parse_edit_spec

    spec = parse_edit_spec(
        "insAGT at chr1:100",
        genome_path=synthetic_chr1_fasta,
    )
    assert spec.edit_type == "insertion"
    assert spec.ref_seq == ""
    assert spec.alt_seq == "AGT"
    # Pure insertion: end == start (zero-width range).
    assert spec.start == 99
    assert spec.end == 99


def test_simple_deletion(synthetic_chr1_fasta: Path) -> None:
    """``"del chr1:100..105"`` is a pure 6-bp deletion."""
    from bionpu.genomics.pe_design.edit_spec import parse_edit_spec

    spec = parse_edit_spec(
        "del chr1:100..105",
        genome_path=synthetic_chr1_fasta,
    )
    assert spec.edit_type == "deletion"
    assert spec.alt_seq == ""
    assert spec.ref_seq == "CGTACG"  # 6 bp from the marker
    assert spec.start == 99
    assert spec.end == 105


# --------------------------------------------------------------------- #
# HGVS notation tests
# --------------------------------------------------------------------- #


def test_hgvs_substitution_genomic_g_dot(synthetic_chr1_fasta: Path) -> None:
    """``"chr1:g.100C>T"`` is the genomic flavor — strand is ``+`` and
    no transcript resolution is required."""
    from bionpu.genomics.pe_design.edit_spec import parse_edit_spec

    spec = parse_edit_spec(
        "chr1:g.100C>T",
        genome_path=synthetic_chr1_fasta,
    )
    assert spec.chrom == "chr1"
    assert spec.start == 99
    assert spec.end == 100
    assert spec.ref_seq == "C"
    assert spec.alt_seq == "T"
    assert spec.edit_type == "substitution"
    assert spec.strand == "+"


def test_hgvs_delins(synthetic_chr1_fasta: Path) -> None:
    """``"chr1:g.100_102delCGTinsAAA"`` is a delins on +/genomic."""
    from bionpu.genomics.pe_design.edit_spec import parse_edit_spec

    spec = parse_edit_spec(
        "chr1:g.100_102delCGTinsAAA",
        genome_path=synthetic_chr1_fasta,
    )
    assert spec.start == 99
    assert spec.end == 102
    assert spec.ref_seq == "CGT"
    assert spec.alt_seq == "AAA"
    assert spec.edit_type == "substitution"


def test_hgvs_dup_genomic(synthetic_chr1_fasta: Path) -> None:
    """``"chr1:g.100dupC"`` duplicates the C at position 100 — represented
    as an insertion of ``C`` immediately after position 100."""
    from bionpu.genomics.pe_design.edit_spec import parse_edit_spec

    spec = parse_edit_spec(
        "chr1:g.100dupC",
        genome_path=synthetic_chr1_fasta,
    )
    assert spec.edit_type == "insertion"
    assert spec.ref_seq == ""
    assert spec.alt_seq == "C"
    # The insertion is positioned AFTER pos 100 (1-based) -> 0-based 100..100.
    assert spec.start == 100
    assert spec.end == 100


# --------------------------------------------------------------------- #
# Transcript-relative HGVS — the strand-aware case
# --------------------------------------------------------------------- #


def test_hgvs_transcript_relative_minus_strand_brca1_complements_alt() -> None:
    """``"NM_007294.4:c.5266dupC"`` resolves to BRCA1 (chr17, ``-``).

    The HGVS alt allele (``C``) is described in transcript orientation,
    which equals the minus strand. To populate ``EditSpec`` in genome-
    ``+``-strand orientation we MUST complement the alt to ``G`` AND
    record ``strand == "-"`` so downstream tools can re-orient if they
    need to.

    This test does NOT validate ``ref_seq`` against a real FASTA — it
    asserts the parser ran the strand-flip + complement correctly. The
    parser's ``c.NNN`` -> genomic-coord resolution is best-effort given
    the bundled refGene has only transcript span (no exon/CDS map);
    that gap is filed as a v1 follow-on in the parser docstring.
    """
    from bionpu.genomics.pe_design.edit_spec import parse_edit_spec

    spec = parse_edit_spec("NM_007294.4:c.5266dupC")
    assert spec.chrom == "chr17"
    assert spec.strand == "-"
    assert spec.edit_type == "insertion"
    assert spec.ref_seq == ""
    # Complement of HGVS-alt 'C' on the minus strand -> 'G' on genome +.
    assert spec.alt_seq == "G", (
        f"minus-strand HGVS dupC must be complemented to G on genome +; "
        f"got {spec.alt_seq!r}"
    )
    # The position must fall within the BRCA1 transcript span on chr17.
    # NM_007294 covers 43044295..43125483 (1-based incl).
    assert 43044295 - 1 <= spec.start <= 43125483


# --------------------------------------------------------------------- #
# Error paths
# --------------------------------------------------------------------- #


def test_ref_mismatch_raises_refmismatcherror(synthetic_chr1_fasta: Path) -> None:
    """If the ref allele in the notation doesn't match the FASTA at
    the given coords, ``RefMismatchError`` fires."""
    from bionpu.genomics.pe_design import edit_spec as es

    with pytest.raises(es.RefMismatchError):
        # FASTA at pos 100 is 'C'; we claim 'A' here.
        es.parse_edit_spec(
            "A>T at chr1:100",
            genome_path=synthetic_chr1_fasta,
        )


def test_edit_too_large_raises() -> None:
    """An alt allele exceeding ``MAX_EDIT_LENGTH_BP`` raises
    ``EditTooLargeError``."""
    from bionpu.genomics.pe_design import edit_spec as es
    from bionpu.genomics.pe_design.pegrna_constants import MAX_EDIT_LENGTH_BP

    big_insert = "A" * (MAX_EDIT_LENGTH_BP + 1)
    with pytest.raises(es.EditTooLargeError):
        es.parse_edit_spec(f"ins{big_insert} at chr1:100")


def test_unsupported_hgvs_inversion_raises() -> None:
    """HGVS inversion (``inv``) raises ``UnsupportedHGVSVariant``."""
    from bionpu.genomics.pe_design import edit_spec as es

    with pytest.raises(es.UnsupportedHGVSVariant):
        es.parse_edit_spec("chr1:g.100_105inv")


def test_unparseable_notation_raises_value_error() -> None:
    """Junk notation strings raise ``ValueError`` (not a silent no-op)."""
    from bionpu.genomics.pe_design.edit_spec import parse_edit_spec

    with pytest.raises(ValueError):
        parse_edit_spec("this is not an edit specifier")
