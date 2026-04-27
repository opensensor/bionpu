"""bionpu.scoring — smoke-mode (torch-free) tests.

GPL-3.0. (c) 2026 OpenSensor.

These tests exercise the scoring pipeline end-to-end without
torch / transformers / weights, by invoking
:class:`DNABERTEpiScorer` in `smoke=True` mode. They verify the
canonical TSV format, score-row alignment, determinism, and the
identity-key contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bionpu.data.canonical_sites import CasOFFinderRow, normalize, write_tsv
from bionpu.scoring.dnabert_epi import DNABERTEpiScorer
from bionpu.scoring.types import (
    ScoreRow,
    parse_score_tsv,
    serialize_canonical_score,
    write_score_tsv,
)


def _row(chrom="chr1", start=100, strand="+", mismatches=0, guide_id="g1") -> CasOFFinderRow:
    return CasOFFinderRow(
        guide_id=guide_id,
        bulge_type="X",
        crrna="AAAAAAAAAAAAAAAAAAAANGG",
        dna="AAAAAAAAAAAAAAAAAAAAAGG",
        chrom=chrom,
        start=start,
        strand=strand,
        mismatches=mismatches,
        bulge_size=0,
    )


def test_smoke_scorer_yields_one_row_per_input() -> None:
    rows = [_row(start=i * 10) for i in range(5)]
    scorer = DNABERTEpiScorer(smoke=True)
    out = list(scorer.score(rows))
    assert len(out) == 5
    for r_in, r_out in zip(rows, out):
        assert isinstance(r_out, ScoreRow)
        assert r_out.identity_key()[:5] == (
            r_in.guide_id,
            r_in.bulge_type,
            r_in.crrna,
            r_in.dna,
            r_in.chrom,
        )


def test_smoke_scorer_is_deterministic() -> None:
    """Same inputs + same seed → byte-identical output."""
    rows = [_row(start=i * 10) for i in range(5)]
    a = list(DNABERTEpiScorer(smoke=True, seed=42).score(rows))
    b = list(DNABERTEpiScorer(smoke=True, seed=42).score(rows))
    assert serialize_canonical_score(a) == serialize_canonical_score(b)


def test_smoke_scorer_seed_changes_score() -> None:
    """Different seed → different scores (with overwhelming probability)."""
    rows = [_row(start=i * 10) for i in range(5)]
    a = list(DNABERTEpiScorer(smoke=True, seed=1).score(rows))
    b = list(DNABERTEpiScorer(smoke=True, seed=2).score(rows))
    # Identity columns identical, scores differ.
    assert all(ra.identity_key() == rb.identity_key() for ra, rb in zip(a, b))
    assert any(ra.score != rb.score for ra, rb in zip(a, b))


def test_smoke_scorer_scores_in_unit_interval() -> None:
    rows = [_row(start=i * 10) for i in range(20)]
    out = list(DNABERTEpiScorer(smoke=True).score(rows))
    assert all(0.0 <= r.score < 1.0 for r in out)


def test_smoke_score_tsv_round_trip(tmp_path: Path) -> None:
    rows = normalize([_row(start=i * 10, guide_id=f"g{i}") for i in range(5)])
    scored = list(DNABERTEpiScorer(smoke=True).score(rows))
    out = tmp_path / "scored.tsv"
    write_score_tsv(out, scored)
    parsed = parse_score_tsv(out)
    assert len(parsed) == len(scored)
    for s, p in zip(scored, parsed):
        assert s.identity_key() == p.identity_key()
        # Score precision is %.6f → tolerate <= 0.5e-6 round-trip error.
        assert abs(s.score - p.score) < 1e-6


def test_score_row_from_row_preserves_identity() -> None:
    base = _row(chrom="chrX", start=1234, strand="-", mismatches=2)
    sr = ScoreRow.from_row(base, score=0.5)
    # Every identity field is verbatim.
    assert sr.guide_id == base.guide_id
    assert sr.crrna == base.crrna
    assert sr.dna == base.dna
    assert sr.chrom == base.chrom
    assert sr.start == base.start
    assert sr.strand == base.strand
    assert sr.mismatches == base.mismatches
    assert sr.bulge_size == base.bulge_size
    assert sr.score == 0.5


def test_smoke_scorer_invalid_device_rejected() -> None:
    with pytest.raises(ValueError, match="device must be 'cpu' or 'gpu'"):
        DNABERTEpiScorer(device="tpu", smoke=True)
