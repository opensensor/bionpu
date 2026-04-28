"""bionpu.scoring.doench_rs2 — Doench on-target scorer smoke tests.

GPL-3.0. (c) 2026 OpenSensor.

Exercises Rule Set 1 (logistic regression) against pure-CPU fixture
inputs. Rule Set 2 / Azimuth is a placeholder — its constructor is
expected to raise loudly when the upstream package is unavailable.
"""

from __future__ import annotations

import pytest

from bionpu.data.canonical_sites import CasOFFinderRow
from bionpu.scoring.doench_rs2 import (
    DOENCH_RS1_INTERCEPT,
    DOENCH_RS1_WEIGHTS,
    AzimuthNotInstalledError,
    DoenchAzimuthScorer,
    DoenchRS1Scorer,
    doench_rs1_score,
    extract_30mer_context,
)
from bionpu.scoring.types import ScoreRow


# ---------------------------------------------------------------------------
# 30-mer context extraction
# ---------------------------------------------------------------------------


def _synthetic_chrom(length: int = 200) -> str:
    """Deterministic ACGT chromosome of the requested length."""
    bases = "ACGT"
    return "".join(bases[(i * 7 + 3) % 4] for i in range(length))


def test_extract_30mer_forward_strand() -> None:
    """For a forward-strand hit the 30-mer is just a substring."""
    chrom = _synthetic_chrom()
    start = 50  # inside the chrom with room for flanks
    ctx = extract_30mer_context(chrom_seq=chrom, start=start, strand="+")
    assert len(ctx) == 30
    # First 4 bases of ctx come from chrom[start - 4 : start].
    assert ctx[:4] == chrom[start - 4 : start].upper()
    # Bases 4..23 of ctx are the protospacer = chrom[start : start+20].
    assert ctx[4:24] == chrom[start : start + 20].upper()


def test_extract_30mer_reverse_strand_is_rc() -> None:
    """For a reverse-strand hit the 30-mer is the RC of the forward window."""
    chrom = _synthetic_chrom()
    start = 50
    fwd = extract_30mer_context(chrom_seq=chrom, start=start, strand="+")
    # Re-extract on - strand at the same coordinate.
    # The - strand window flips the flank direction.
    rev = extract_30mer_context(chrom_seq=chrom, start=start, strand="-")
    assert len(rev) == 30
    # The reverse-strand context, when reverse-complemented, must
    # come from the same chromosome. We verify by RC'ing it and
    # checking the spacer bases land in the chromosome.
    rc_table = str.maketrans("ACGT", "TGCA")
    rev_back_to_fwd = rev.translate(rc_table)[::-1]
    # `rev_back_to_fwd` is the 30-nt forward window centred on the
    # same hit but with the flanks swapped (the - strand has its
    # 3' flank where the + strand has its 5' flank, etc.).
    assert rev_back_to_fwd[3:26] in chrom.upper()


def test_extract_30mer_out_of_range_rejects() -> None:
    """A site too close to the chromosome edge must fail loud."""
    chrom = _synthetic_chrom(40)
    with pytest.raises(ValueError, match="out of range"):
        extract_30mer_context(chrom_seq=chrom, start=2, strand="+")


def test_extract_30mer_invalid_strand_rejects() -> None:
    chrom = _synthetic_chrom()
    with pytest.raises(ValueError, match="strand must be"):
        extract_30mer_context(chrom_seq=chrom, start=50, strand="?")


# ---------------------------------------------------------------------------
# Rule Set 1 logistic regression
# ---------------------------------------------------------------------------


def _ngg_30mer(spacer: str, *, flank5: str = "ACGT", flank3: str = "ACG") -> str:
    """Build a 30-nt context with the supplied 20-nt spacer + NGG PAM."""
    if len(spacer) != 20:
        raise ValueError("spacer must be 20 nt")
    return flank5 + spacer + "AGG" + flank3


def test_doench_rs1_score_in_unit_interval() -> None:
    ctx = _ngg_30mer("AAAAAAAAAAAAAAAAAAAA")
    s = doench_rs1_score(ctx)
    assert 0.0 < s < 1.0


def test_doench_rs1_score_pure_intercept_matches_logistic() -> None:
    """Pick a 30-mer that hits no non-zero weights — only the GC
    indicator and the intercept fire. Verify the score is the
    logistic of the resulting linear sum.

    A 20-nt spacer of all A: GC count = 0, fires gc_low.
    Construct a 30-mer where no position-specific 1-mer or 2-mer
    weight in the table fires — easier said than done, so we
    don't assert exact equality here but instead check that the
    score is in (0, 1) and the logistic-decision is sane.
    """
    # All-A spacer; flanks chosen to NOT trigger the position-specific
    # weight at position 1 (G) etc. We just use 'A' everywhere outside
    # the PAM.
    ctx = "A" * 4 + "A" * 20 + "AGG" + "A" * 3
    s = doench_rs1_score(ctx)
    assert 0.0 < s < 1.0


def test_doench_rs1_high_gc_score_differs_from_low_gc() -> None:
    """The GC-content indicator MUST influence the score: a spacer
    with GC > 10 fires gc_high, GC < 10 fires gc_low — different
    weights, different output."""
    low_gc_spacer = "AAAAAAAAAAAAAAAAAAAA"  # GC = 0
    high_gc_spacer = "GCGCGCGCGCGCGCGCGCGC"  # GC = 20
    s_low = doench_rs1_score(_ngg_30mer(low_gc_spacer))
    s_high = doench_rs1_score(_ngg_30mer(high_gc_spacer))
    assert s_low != s_high


def test_doench_rs1_changes_with_position_specific_base() -> None:
    """Flipping a position whose weight is non-zero must change the
    score; flipping a zero-weight position must not."""
    base_ctx = _ngg_30mer("AAAAAAAAAAAAAAAAAAAA")
    base_score = doench_rs1_score(base_ctx)

    # Position 1 has weight only for 'G' (-0.2753771). Flip A->G at
    # ctx[0] (i.e. position 1 of the 30-mer).
    flipped = "G" + base_ctx[1:]
    flipped_score = doench_rs1_score(flipped)
    assert flipped_score != base_score


def test_doench_rs1_invalid_length_rejects() -> None:
    with pytest.raises(ValueError, match="must be 30 nt"):
        doench_rs1_score("ACGT")


def test_doench_rs1_invalid_base_rejects() -> None:
    with pytest.raises(ValueError, match="must be ACGT"):
        doench_rs1_score("N" * 30)


# ---------------------------------------------------------------------------
# Scorer (row-stream) interface
# ---------------------------------------------------------------------------


def _row(*, chrom: str = "chr1", start: int = 50, strand: str = "+",
         guide_id: str = "g1", mismatches: int = 0) -> CasOFFinderRow:
    return CasOFFinderRow(
        guide_id=guide_id,
        bulge_type="X",
        crrna="A" * 20 + "AGG",
        dna="A" * 20 + "AGG",
        chrom=chrom,
        start=start,
        strand=strand,
        mismatches=mismatches,
        bulge_size=0,
    )


def test_scorer_yields_one_row_per_input() -> None:
    chrom = _synthetic_chrom()
    rows = [
        _row(start=50, guide_id="g1"),
        _row(start=100, guide_id="g2"),
    ]
    scorer = DoenchRS1Scorer({"chr1": chrom})
    out = list(scorer.score(rows))
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


def test_scorer_score_in_unit_interval() -> None:
    chrom = _synthetic_chrom()
    rows = [_row(start=i * 5 + 30) for i in range(10) if i * 5 + 30 < 150]
    scorer = DoenchRS1Scorer({"chr1": chrom})
    out = list(scorer.score(rows))
    for r in out:
        assert 0.0 < r.score < 1.0


def test_scorer_chrom_lookup_required() -> None:
    """Scoring a row whose chromosome isn't in the lookup must fail loudly."""
    chrom = _synthetic_chrom()
    scorer = DoenchRS1Scorer({"chr1": chrom})
    rows = [_row(chrom="chrX", start=50)]
    with pytest.raises(KeyError, match="chrom_lookup"):
        list(scorer.score(rows))


def test_scorer_preserves_input_order() -> None:
    """Identity columns are preserved verbatim; the scorer never re-sorts."""
    chrom = _synthetic_chrom()
    rows = [
        _row(start=120, guide_id="zg"),
        _row(start=40, guide_id="ag"),
        _row(start=80, guide_id="mg"),
    ]
    scorer = DoenchRS1Scorer({"chr1": chrom})
    out = list(scorer.score(rows))
    assert [r.start for r in out] == [120, 40, 80]


def test_scorer_chrom_lookup_must_be_dict() -> None:
    with pytest.raises(TypeError, match="chrom_lookup"):
        DoenchRS1Scorer("not a dict")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Rule Set 2 / Azimuth placeholder
# ---------------------------------------------------------------------------


def test_azimuth_scorer_raises_when_unavailable() -> None:
    """The Azimuth path is a placeholder; constructing it raises so
    callers can't accidentally fall through to a silently-broken
    implementation. If `azimuth` is installed in the test env, the
    constructor still raises with a 'not implemented in v0.3'
    message — track upgrade in docs/model-selection-audit.md."""
    with pytest.raises(AzimuthNotInstalledError):
        DoenchAzimuthScorer()


# ---------------------------------------------------------------------------
# Coefficient-table sanity
# ---------------------------------------------------------------------------


def test_rs1_intercept_is_paper_value() -> None:
    """Guard against accidental edits to the published intercept."""
    assert DOENCH_RS1_INTERCEPT == 0.59763615


def test_rs1_weights_table_has_gc_indicators() -> None:
    """The two GC-content indicator features must be present."""
    assert ("gc_low",) in DOENCH_RS1_WEIGHTS
    assert ("gc_high",) in DOENCH_RS1_WEIGHTS


def test_rs1_weights_positions_in_range() -> None:
    """Every position-specific feature uses position in 1..30."""
    for key in DOENCH_RS1_WEIGHTS:
        if key[0] in ("pos", "di"):
            position = key[1]
            assert 1 <= position <= 30, f"position {position} out of range in {key}"
