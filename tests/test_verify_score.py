"""bionpu.verify.score — equivalence policy tests.

GPL-3.0. (c) 2026 OpenSensor.

Tests the BITWISE_EXACT and NUMERIC_EPSILON policies. The smoke-mode
scorer provides deterministic test fixtures so we can drive the
policies under known divergence patterns without torch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bionpu.data.canonical_sites import CasOFFinderRow, normalize
from bionpu.scoring.dnabert_epi import DNABERTEpiScorer
from bionpu.scoring.types import write_score_tsv
from bionpu.verify.score import compare_score_outputs


def _row(start: int) -> CasOFFinderRow:
    return CasOFFinderRow(
        guide_id="g1",
        bulge_type="X",
        crrna="AAAAAAAAAAAAAAAAAAAANGG",
        dna="AAAAAAAAAAAAAAAAAAAAAGG",
        chrom="chr1",
        start=start,
        strand="+",
        mismatches=0,
        bulge_size=0,
    )


def _write_smoke_tsv(tmp_path: Path, name: str, *, seed: int = 0, n: int = 5) -> Path:
    rows = normalize([_row(start=i * 10) for i in range(n)])
    scored = list(DNABERTEpiScorer(smoke=True, seed=seed).score(rows))
    out = tmp_path / name
    write_score_tsv(out, scored)
    return out


# ---------- BITWISE_EXACT ----------

def test_bitwise_equal_when_inputs_byte_identical(tmp_path: Path) -> None:
    a = _write_smoke_tsv(tmp_path, "a.tsv", seed=42)
    b = _write_smoke_tsv(tmp_path, "b.tsv", seed=42)
    r = compare_score_outputs(a, b, policy="BITWISE_EXACT")
    assert r.equal
    assert r.a_sha256 == r.b_sha256
    assert r.divergences == ()


def test_bitwise_diverges_on_score_difference(tmp_path: Path) -> None:
    a = _write_smoke_tsv(tmp_path, "a.tsv", seed=42)
    b = _write_smoke_tsv(tmp_path, "b.tsv", seed=43)
    r = compare_score_outputs(a, b, policy="BITWISE_EXACT")
    assert not r.equal
    assert r.a_sha256 != r.b_sha256
    assert any("sha256 mismatch" in d.message for d in r.divergences)


def test_bitwise_rejects_epsilon_arg(tmp_path: Path) -> None:
    a = _write_smoke_tsv(tmp_path, "a.tsv")
    b = _write_smoke_tsv(tmp_path, "b.tsv")
    with pytest.raises(ValueError, match="BITWISE_EXACT policy does not accept epsilon"):
        compare_score_outputs(a, b, policy="BITWISE_EXACT", epsilon=1e-6)


# ---------- NUMERIC_EPSILON ----------

def test_numeric_eps_equal_on_identical_input(tmp_path: Path) -> None:
    a = _write_smoke_tsv(tmp_path, "a.tsv", seed=42)
    b = _write_smoke_tsv(tmp_path, "b.tsv", seed=42)
    r = compare_score_outputs(a, b, policy="NUMERIC_EPSILON", epsilon=1e-6)
    assert r.equal
    assert r.max_abs_diff == 0.0


def test_numeric_eps_equal_when_diff_below_tolerance(tmp_path: Path) -> None:
    """Different scorer seed produces large score diffs; eps=1.0 always passes."""
    a = _write_smoke_tsv(tmp_path, "a.tsv", seed=1)
    b = _write_smoke_tsv(tmp_path, "b.tsv", seed=2)
    r = compare_score_outputs(a, b, policy="NUMERIC_EPSILON", epsilon=1.0)
    assert r.equal
    # max_abs_diff is reported regardless — useful for drift monitoring.
    assert r.max_abs_diff is not None
    assert 0.0 < r.max_abs_diff <= 1.0


def test_numeric_eps_diverges_when_diff_above_tolerance(tmp_path: Path) -> None:
    a = _write_smoke_tsv(tmp_path, "a.tsv", seed=1)
    b = _write_smoke_tsv(tmp_path, "b.tsv", seed=2)
    r = compare_score_outputs(a, b, policy="NUMERIC_EPSILON", epsilon=1e-9)
    assert not r.equal
    # All five rows should diverge (smoke seeds produce uniform-random scores).
    assert all("score deviation" in d.message for d in r.divergences)


def test_numeric_eps_rejects_negative_epsilon(tmp_path: Path) -> None:
    a = _write_smoke_tsv(tmp_path, "a.tsv")
    b = _write_smoke_tsv(tmp_path, "b.tsv")
    with pytest.raises(ValueError, match="NUMERIC_EPSILON policy requires epsilon"):
        compare_score_outputs(a, b, policy="NUMERIC_EPSILON", epsilon=-0.1)


def test_numeric_eps_requires_epsilon(tmp_path: Path) -> None:
    a = _write_smoke_tsv(tmp_path, "a.tsv")
    b = _write_smoke_tsv(tmp_path, "b.tsv")
    with pytest.raises(ValueError, match="NUMERIC_EPSILON policy requires epsilon"):
        compare_score_outputs(a, b, policy="NUMERIC_EPSILON")


def test_numeric_eps_detects_row_count_mismatch(tmp_path: Path) -> None:
    a = _write_smoke_tsv(tmp_path, "a.tsv", n=5)
    b = _write_smoke_tsv(tmp_path, "b.tsv", n=3)
    r = compare_score_outputs(a, b, policy="NUMERIC_EPSILON", epsilon=1e-6)
    assert not r.equal
    assert any("row count mismatch" in d.message for d in r.divergences)


def test_numeric_eps_detects_identity_drift(tmp_path: Path) -> None:
    """Two TSVs with the same row count but different chrom values diverge."""
    a_rows = normalize([_row(start=i * 10) for i in range(3)])
    b_rows = normalize([
        CasOFFinderRow(
            guide_id="g1", bulge_type="X",
            crrna="AAAAAAAAAAAAAAAAAAAANGG",
            dna="AAAAAAAAAAAAAAAAAAAAAGG",
            chrom="chr2",  # NOT chr1 — identity drift
            start=i * 10, strand="+", mismatches=0, bulge_size=0,
        )
        for i in range(3)
    ])
    a = tmp_path / "a.tsv"
    b = tmp_path / "b.tsv"
    write_score_tsv(a, list(DNABERTEpiScorer(smoke=True, seed=0).score(a_rows)))
    write_score_tsv(b, list(DNABERTEpiScorer(smoke=True, seed=0).score(b_rows)))
    r = compare_score_outputs(a, b, policy="NUMERIC_EPSILON", epsilon=1.0)
    assert not r.equal
    assert any("identity-column mismatch" in d.message for d in r.divergences)


def test_unknown_policy_rejected(tmp_path: Path) -> None:
    a = _write_smoke_tsv(tmp_path, "a.tsv")
    b = _write_smoke_tsv(tmp_path, "b.tsv")
    with pytest.raises(ValueError, match="unknown policy"):
        compare_score_outputs(a, b, policy="OUR_OWN_POLICY")  # type: ignore[arg-type]
