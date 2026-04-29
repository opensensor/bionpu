# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

"""Track B v0.1 — Tests for the paralog gene-family mapper.

Covers:

1. ``get_paralog_spans("HBB")`` returns the hemoglobin-cluster siblings
   (HBD, HBG1, HBG2, HBE1, HBZ) and excludes HBB itself.
2. Unknown gene symbol returns an empty tuple (no exception).
3. ``is_in_any_paralog`` correctly hit-tests an HBD coordinate against
   the HBB paralog span list (positive) and a chr1 coordinate against
   the same list (negative).
4. The reverse-index lookup is case-insensitive on symbols.
5. Every span carries a strictly positive width and a chr-prefixed
   contig (sanity check for the hardcoded coordinate table).
"""

from __future__ import annotations

import pytest

from bionpu.data.paralog_mapper import (
    PARALOG_FAMILIES,
    GeneSpan,
    get_paralog_spans,
    is_in_any_paralog,
)


def test_hbb_paralog_spans_cover_hemoglobin_cluster() -> None:
    """HBB's siblings are the hemoglobin family minus HBB itself."""
    spans = get_paralog_spans("HBB")
    symbols = sorted(s.symbol for s in spans)
    assert symbols == ["HBD", "HBE1", "HBG1", "HBG2", "HBZ"]
    # HBB itself MUST NOT appear in its own paralog list (the on-target
    # gene is never a "paralog hit").
    assert all(s.symbol != "HBB" for s in spans)


def test_unknown_gene_symbol_returns_empty_tuple() -> None:
    """Symbols not in the v0.1 hardcoded map yield an empty tuple."""
    assert get_paralog_spans("UNKNOWN_GENE_XYZ") == ()
    # And callers passing arbitrary case still get the empty result
    # (case-insensitive resolution).
    assert get_paralog_spans("definitely_not_a_gene") == ()


def test_is_in_any_paralog_hit_test_hbb_cluster() -> None:
    """A coordinate inside HBD's span hits the HBB paralog list;
    one in chr1 doesn't."""
    spans = get_paralog_spans("HBB")
    # HBD: chr11:5232708..5235762 (1-based inclusive) per v0.1 map.
    # 0-based 5234000 == 1-based 5234001 — inside HBD.
    assert is_in_any_paralog(
        chrom="chr11", pos_0b=5234000, paralog_spans=spans
    )
    # Same chrom, far away from any hemoglobin gene -> miss.
    assert not is_in_any_paralog(
        chrom="chr11", pos_0b=5_500_000, paralog_spans=spans
    )
    # Different chrom entirely -> miss (even though numerically the
    # position would land inside an HBB paralog if the chrom matched).
    assert not is_in_any_paralog(
        chrom="chr1", pos_0b=5234000, paralog_spans=spans
    )


def test_lookup_is_case_insensitive_on_symbol() -> None:
    """Symbol normalisation is case-insensitive."""
    upper = get_paralog_spans("BRCA1")
    lower = get_paralog_spans("brca1")
    mixed = get_paralog_spans("BrCa1")
    assert {s.symbol for s in upper} == {s.symbol for s in lower}
    assert {s.symbol for s in upper} == {s.symbol for s in mixed}
    # And BRCA1 itself is filtered.
    assert all(s.symbol.upper() != "BRCA1" for s in upper)


def test_all_hardcoded_spans_have_positive_width_and_chr_prefix() -> None:
    """Sanity-check the hardcoded GRCh38 coordinate table."""
    seen: set[str] = set()
    for family_key, spans in PARALOG_FAMILIES.items():
        assert spans, f"family {family_key} is empty"
        for s in spans:
            assert isinstance(s, GeneSpan)
            assert s.chrom.startswith("chr"), (
                f"family {family_key}: {s.symbol} chrom={s.chrom!r} "
                f"missing 'chr' prefix"
            )
            assert s.start > 0
            assert s.end > s.start, (
                f"family {family_key}: {s.symbol} has end={s.end} <= "
                f"start={s.start}"
            )
            seen.add(s.symbol.upper())
    # Coverage check: the v0.1 smoke targets must all have entries.
    for required in ("BRCA1", "TP53", "EGFR", "HBB", "MYC"):
        assert required in seen, (
            f"v0.1 smoke target {required} missing from PARALOG_FAMILIES"
        )


def test_egfr_family_includes_erbb_paralogs() -> None:
    """Spot-check EGFR's siblings are the canonical ERBB paralogs."""
    symbols = {s.symbol for s in get_paralog_spans("EGFR")}
    assert {"ERBB2", "ERBB3", "ERBB4"} <= symbols
    assert "EGFR" not in symbols


def test_genespan_contains_uses_1_based_inclusive_test() -> None:
    """The 0-based -> 1-based conversion in ``contains`` is correct
    on both span boundaries."""
    span = GeneSpan(symbol="TEST", chrom="chrT", start=100, end=200)
    # 1-based 100 == 0-based 99 (inclusive lower edge).
    assert span.contains(chrom="chrT", pos_0b=99)
    # 1-based 200 == 0-based 199 (inclusive upper edge).
    assert span.contains(chrom="chrT", pos_0b=199)
    # 1-based 99 == 0-based 98 (just below).
    assert not span.contains(chrom="chrT", pos_0b=98)
    # 1-based 201 == 0-based 200 (just above).
    assert not span.contains(chrom="chrT", pos_0b=200)
    # Wrong chrom -> never contains.
    assert not span.contains(chrom="chrU", pos_0b=150)
