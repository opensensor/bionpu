"""bionpu.scoring.cfd — CFD off-target scorer smoke tests.

GPL-3.0. (c) 2026 OpenSensor.

These tests exercise the CFD scorer on small fixture inputs without
requiring the NPU silicon path. The scorer is a pure-CPU stage that
consumes ``CasOFFinderRow``s (the canonical NPU-scan output) and
emits ``ScoreRow``s with per-row CFD scores.
"""

from __future__ import annotations

import pytest

from bionpu.data.canonical_sites import CasOFFinderRow
from bionpu.scoring.cfd import (
    DOENCH_2016_MM_MATRIX,
    DOENCH_2016_PAM_MATRIX,
    HSU_2013_POSITION_WEIGHTS,
    CFDScorer,
    aggregate_cfd,
    cfd_score_pair,
)
from bionpu.scoring.types import ScoreRow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PERFECT_GUIDE = "GAGTCCGAGCAGAAGAAGAA"  # canonical EMX1 sgRNA #1


def _row(
    *,
    crrna_spacer: str = PERFECT_GUIDE,
    dna_spacer: str | None = None,
    pam: str = "TGG",
    chrom: str = "chr1",
    start: int = 1000,
    strand: str = "+",
    mismatches: int = 0,
    guide_id: str = "g1",
) -> CasOFFinderRow:
    """Build a 23-nt CasOFFinder row from a 20-nt spacer + 3-nt PAM."""
    if dna_spacer is None:
        dna_spacer = crrna_spacer
    # Cas-OFFinder convention: lowercases mismatches in the spacer.
    dna_with_case = "".join(
        b if b == g else b.lower()
        for g, b in zip(crrna_spacer, dna_spacer, strict=True)
    )
    return CasOFFinderRow(
        guide_id=guide_id,
        bulge_type="X",
        crrna=crrna_spacer + "NGG",
        dna=dna_with_case + pam,
        chrom=chrom,
        start=start,
        strand=strand,
        mismatches=mismatches,
        bulge_size=0,
    )


# ---------------------------------------------------------------------------
# pair-level CFD
# ---------------------------------------------------------------------------


def test_perfect_match_scores_one() -> None:
    """A perfect match has CFD == 1.0 by construction."""
    s = cfd_score_pair(PERFECT_GUIDE, PERFECT_GUIDE)
    assert s == 1.0


def test_perfect_match_with_pam_scores_pam_weight() -> None:
    """Perfect spacer match + AGG PAM scales by the PAM matrix."""
    s = cfd_score_pair(PERFECT_GUIDE, PERFECT_GUIDE, pam="GG")
    # GG is the canonical NGG -> weight 1.0; total 1.0 * 1.0 = 1.0
    assert s == 1.0


def test_pam_penalty_for_non_ngg() -> None:
    """A non-NGG PAM scales the score by the matrix entry."""
    s = cfd_score_pair(PERFECT_GUIDE, PERFECT_GUIDE, pam="GT")
    assert s == DOENCH_2016_PAM_MATRIX["GT"]


def test_t_to_u_match_is_not_a_mismatch() -> None:
    """The CFD matrix keys use rU; a DNA-encoded T spacer must match T."""
    crrna_dna = PERFECT_GUIDE  # T-encoded (DNA form of the sgRNA)
    crrna_rna = crrna_dna.replace("T", "U")
    a = cfd_score_pair(crrna_dna, PERFECT_GUIDE)
    b = cfd_score_pair(crrna_rna, PERFECT_GUIDE)
    assert a == 1.0
    assert b == 1.0


def test_single_seed_mismatch_drives_score_down() -> None:
    """A single mismatch at PAM-proximal position 20 (seed) should
    drive the score well below 1.0.

    This is a property test, not a value test: any reasonable CFD
    matrix gives weight < 1 at a seed mismatch (otherwise the seed
    wouldn't be load-bearing). We pick a flip that we know is in
    the matrix and assert weight < 1.
    """
    # Position 20 (last spacer base, immediately 5' of PAM).
    # PERFECT_GUIDE[19] is 'A'. Flip to 'C'.
    off = list(PERFECT_GUIDE)
    off[19] = "C"
    off_str = "".join(off)
    score = cfd_score_pair(PERFECT_GUIDE, off_str)
    # rA position-20 match swap: rA:dC at position 20 has weight 0.0.
    # Confirm we're below 1 -- exact value depends on the matrix.
    assert 0.0 <= score < 1.0


def test_pam_distal_mismatch_scores_higher_than_seed() -> None:
    """Property: a mismatch at the PAM-distal end (position 1) costs
    less than a mismatch at the seed (position 20). This is the
    headline finding in Hsu 2013 / Doench 2016."""
    # Use rA -> dC mismatch at position 1 vs at position 20, where the
    # base context is the same.
    crrna = "A" + PERFECT_GUIDE[1:]  # ensure pos-1 is A
    crrna = crrna[:19] + "A"          # ensure pos-20 is A
    distal_off = "C" + crrna[1:]      # mismatch at pos 1
    seed_off = crrna[:19] + "C"       # mismatch at pos 20

    distal_score = cfd_score_pair(crrna, distal_off)
    seed_score = cfd_score_pair(crrna, seed_off)
    # The PAM-distal mismatch (pos 1) preserves more activity.
    assert distal_score > seed_score


def test_score_is_product_over_positions() -> None:
    """Two independent mismatches: score == w1 * w2 * 1^18.

    This is a sanity check that the implementation is *just* the
    product over per-position weights, with matches contributing 1.0.
    The CFD matrix is keyed by ``(sgRNA_base, WC-complement-of-DNA-base,
    position)``: spacer A vs off-target C corresponds to matrix key
    (A, G, ...) since complement(C) = G.
    """
    crrna = "A" * 20
    # Mismatches at positions 1 and 5 only: spacer-A vs off-target-C
    # (matrix key (A, G, 1)) and spacer-A vs off-target-G (matrix key
    # (A, C, 5)).
    off = list(crrna)
    off[0] = "C"   # position 1 (PAM-distal)
    off[4] = "G"   # position 5
    off_str = "".join(off)

    # Expected matrix keys are (sgRNA_base, complement-of-off-base, pos):
    w1 = DOENCH_2016_MM_MATRIX[("A", "G", 1)]
    w2 = DOENCH_2016_MM_MATRIX[("A", "C", 5)]
    expected = w1 * w2

    actual = cfd_score_pair(crrna, off_str)
    assert abs(actual - expected) < 1e-12


def test_invalid_spacer_length_rejected() -> None:
    with pytest.raises(ValueError, match="must be 20 nt"):
        cfd_score_pair("AAAA", "AAAA")


def test_unknown_matrix_rejected() -> None:
    with pytest.raises(ValueError, match="unknown matrix"):
        cfd_score_pair(PERFECT_GUIDE, PERFECT_GUIDE, matrix="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Hsu 2013 matrix
# ---------------------------------------------------------------------------


def test_hsu_2013_matrix_perfect_match_is_one() -> None:
    s = cfd_score_pair(PERFECT_GUIDE, PERFECT_GUIDE, matrix="hsu_2013")
    assert s == 1.0


def test_hsu_2013_seed_mismatch_zeroes_score() -> None:
    """Hsu 2013 weights at PAM-proximal positions 1, 2, 4, 5, 8 are 0.0
    (the most strict seed positions). A single mismatch in the seed
    therefore drives the score to 0."""
    # The "PAM-proximal position 1" in Hsu's convention is position 20
    # in the 5'->3' indexing used by ``cfd_score_pair``. So flip the
    # last spacer base.
    off = PERFECT_GUIDE[:19] + ("C" if PERFECT_GUIDE[19] != "C" else "T")
    s = cfd_score_pair(PERFECT_GUIDE, off, matrix="hsu_2013")
    # Hsu position 1 weight = 0.0 (per HSU_2013_POSITION_WEIGHTS).
    assert HSU_2013_POSITION_WEIGHTS[1] == 0.0
    assert s == 0.0


def test_hsu_2013_distal_mismatch_preserves_score() -> None:
    """Position-20 PAM-distal in Hsu's convention has weight 0.583;
    a single mismatch at the very 5' end of the spacer should keep
    most of the score."""
    off = ("C" if PERFECT_GUIDE[0] != "C" else "T") + PERFECT_GUIDE[1:]
    s = cfd_score_pair(PERFECT_GUIDE, off, matrix="hsu_2013")
    assert abs(s - HSU_2013_POSITION_WEIGHTS[20]) < 1e-12


# ---------------------------------------------------------------------------
# Scorer (row-stream) interface
# ---------------------------------------------------------------------------


def test_cfdscorer_yields_one_row_per_input() -> None:
    rows = [
        _row(start=100, mismatches=0),
        _row(dna_spacer="C" + PERFECT_GUIDE[1:], start=200, mismatches=1),
    ]
    out = list(CFDScorer().score(rows))
    assert len(out) == 2
    for r_in, r_out in zip(rows, out):
        assert isinstance(r_out, ScoreRow)
        assert r_out.identity_key()[:5] == (
            r_in.guide_id,
            r_in.bulge_type,
            r_in.crrna,
            r_in.dna,
            r_in.chrom,
        )


def test_cfdscorer_perfect_row_scores_one() -> None:
    out = list(CFDScorer().score([_row(mismatches=0)]))
    assert out[0].score == 1.0


def test_cfdscorer_pam_penalty_off_by_default() -> None:
    """A non-NGG PAM in a perfect-spacer row scores 1.0 by default
    (PAM filter at scan time is presumed)."""
    out = list(CFDScorer().score([_row(pam="TGT")]))
    # spacer is perfect; PAM penalty disabled -> 1.0
    assert out[0].score == 1.0


def test_cfdscorer_pam_penalty_on_applies_matrix() -> None:
    out = list(CFDScorer(apply_pam_penalty=True).score([_row(pam="TGT")]))
    # leading 'T' of 'TGT' is the N; matrix key is 'GT'.
    expected = DOENCH_2016_PAM_MATRIX["GT"]
    assert abs(out[0].score - expected) < 1e-12


def test_cfdscorer_preserves_input_order() -> None:
    """Order is load-bearing -- the verify-score policy assumes
    identity-aligned rows. The scorer must NOT re-sort."""
    rows = [
        _row(start=300, guide_id="zg"),
        _row(start=100, guide_id="ag"),
        _row(start=200, guide_id="mg"),
    ]
    out = list(CFDScorer().score(rows))
    assert [r.start for r in out] == [300, 100, 200]


def test_cfdscorer_invalid_matrix_rejected() -> None:
    with pytest.raises(ValueError, match="matrix must be"):
        CFDScorer(matrix="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Aggregate (specificity) score
# ---------------------------------------------------------------------------


def test_aggregate_cfd_no_offtargets_is_perfect() -> None:
    """A guide whose only hit is the on-target gets specificity == 100."""
    rows = [_row(mismatches=0, guide_id="g1")]
    scored = list(CFDScorer().score(rows))
    agg = aggregate_cfd(scored)
    assert agg == {"g1": 100.0}


def test_aggregate_cfd_includes_offtargets() -> None:
    """Off-targets contribute to the denominator."""
    off_spacer = list(PERFECT_GUIDE)
    off_spacer[5] = "T"  # one mismatch
    off_str = "".join(off_spacer)
    rows = [
        _row(mismatches=0, guide_id="g1", start=100),
        _row(dna_spacer=off_str, mismatches=1, guide_id="g1", start=200),
    ]
    scored = list(CFDScorer().score(rows))
    agg = aggregate_cfd(scored)
    # Off-target CFD is in (0, 1]; spec = 100 * (100 / (100 + 100 * cfd))
    # so spec is in [50, 100). The on-target row is excluded by default.
    assert "g1" in agg
    assert 50.0 <= agg["g1"] < 100.0


def test_aggregate_cfd_excludes_on_target_by_default() -> None:
    """The on-target row (mismatches=0) is excluded from the off-
    target sum by default. Re-including it via the flag drops the
    aggregate score because on-target CFD is 1.0."""
    rows = [_row(mismatches=0, guide_id="g1")]
    scored = list(CFDScorer().score(rows))

    excluded = aggregate_cfd(scored, exclude_on_target=True)
    included = aggregate_cfd(scored, exclude_on_target=False)
    assert excluded["g1"] == 100.0
    # On-target CFD = 1.0 enters the sum as 100 (CRISPOR convention),
    # so spec = 100 * (100 / (100 + 100)) = 50.0
    assert abs(included["g1"] - 50.0) < 1e-9


def test_aggregate_cfd_per_guide_is_independent() -> None:
    """Different guides aggregate separately."""
    off1 = list(PERFECT_GUIDE)
    off1[5] = "T"
    rows = [
        _row(guide_id="gA", dna_spacer="".join(off1), mismatches=1),
        _row(guide_id="gB", dna_spacer="".join(off1), mismatches=1),
    ]
    scored = list(CFDScorer().score(rows))
    agg = aggregate_cfd(scored)
    assert set(agg.keys()) == {"gA", "gB"}
    # Same off-target sequence and same guide -> identical specificity.
    assert agg["gA"] == agg["gB"]


def test_aggregate_cfd_empty_input() -> None:
    assert aggregate_cfd([]) == {}
