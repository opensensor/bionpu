# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

"""Track B v0.1 — Tests for the paralog-aware T8 ranker (split CFD
aggregates + PRIDICT G-prefix retry).

Closes the HBB NaN regression surfaced by Track B v0 T13's smoke run.

Coverage
--------
1. ``split_cfd_aggregates`` correctly partitions a hand-built site list
   into paralog vs non-paralog CRISPOR aggregates.
2. The composite score is computed from the NON-paralog CFD aggregate
   only — adding in-paralog hits (with ``in_paralog=True``) does NOT
   degrade the composite.
3. PRIDICT G-prefix retry: when the scorer returns
   ``PEGRNA_NOT_ENUMERATED_BY_PRIDICT`` for a non-G-prefix spacer and
   succeeds for the ``G + spacer[1:]`` retry, the ranker adopts the
   retry's efficiency and tags the row with ``PRIDICT_GPREFIX_RETRY``.
4. The G-prefix retry is NOT triggered when the spacer already starts
   with ``G`` (no spurious retry calls).
5. The T8 ranker populates ``paralog_hit_count_pegrna`` and
   ``cfd_aggregate_paralog_pegrna`` on the resulting RankedPegRNA.
"""

from __future__ import annotations

import math
from typing import Callable

import pytest

from bionpu.genomics.pe_design.types import (
    EditSpec,
    OffTargetSite,
    PegRNACandidate,
    PegRNAFoldingFeatures,
    PRIDICTScore,
)


# ---------------------------------------------------------------------------
# Helpers (mirroring test_pe_design_ranker.py minus paralog-specific bits)
# ---------------------------------------------------------------------------


def _make_substitution_spec() -> EditSpec:
    return EditSpec(
        chrom="chrSyn",
        start=100,
        end=101,
        ref_seq="C",
        alt_seq="T",
        edit_type="substitution",
        notation_used="C>T at chrSyn:101",
        strand="+",
    )


_PBS_SOURCE = "GCACGUGCUCAGUCGAUCGAUCG"
_RTT_SOURCE = "UCACGUGCUCAGUCGAUCGAUCGA"


def _make_pe2_candidate(
    *,
    nick_site: int,
    spacer_seq: str = "GGCCCAGACTGAGCACGTGA",
    pbs_length: int = 13,
    rtt_length: int = 14,
) -> PegRNACandidate:
    pbs_seq = _PBS_SOURCE[:pbs_length]
    rtt_seq = _RTT_SOURCE[:rtt_length]
    return PegRNACandidate(
        spacer_seq=spacer_seq,
        pam_seq="TGG",
        scaffold_variant="sgRNA_canonical",
        pbs_seq=pbs_seq,
        pbs_length=pbs_length,
        rtt_seq=rtt_seq,
        rtt_length=rtt_length,
        nick_site=nick_site,
        full_pegrna_rna_seq="GGCCCAGACUGAGCACGUGA" + "GUUU" + rtt_seq + pbs_seq,
        edit_position_in_rtt=0,
        strategy="PE2",
        strand="+",
        rt_product_seq="N/A",
        chrom="chrSyn",
    )


def _folding_features_factory(mfe_kcal: float = -10.0):
    def _extract(spacer, scaffold, rtt, pbs, *, scaffold_variant="sgRNA_canonical"):
        return PegRNAFoldingFeatures(
            mfe_kcal=mfe_kcal,
            mfe_structure="." * (len(spacer) + len(scaffold) + len(rtt) + len(pbs)),
            pbs_pairing_prob=0.5,
            scaffold_disruption=0.05,
        )

    return _extract


# ---------------------------------------------------------------------------
# Test 1 — split_cfd_aggregates partitioning
# ---------------------------------------------------------------------------


def test_split_cfd_aggregates_partitions_correctly() -> None:
    """A site list with mixed in_paralog flags splits into the right
    per-bucket CRISPOR aggregates and counts."""
    from bionpu.genomics.pe_design.ranker import split_cfd_aggregates

    sites = [
        OffTargetSite(chrom="chrA", pos=10, strand="+", mismatches=0, cfd_score=1.0),  # on-target
        OffTargetSite(chrom="chrA", pos=200, strand="+", mismatches=2, cfd_score=0.5, in_paralog=False),
        OffTargetSite(chrom="chrA", pos=500, strand="-", mismatches=3, cfd_score=0.3, in_paralog=False),
        OffTargetSite(chrom="chrB", pos=100, strand="+", mismatches=1, cfd_score=0.9, in_paralog=True),
        OffTargetSite(chrom="chrB", pos=400, strand="+", mismatches=2, cfd_score=0.4, in_paralog=True),
    ]
    cfd_non, cfd_par, count_non, count_par = split_cfd_aggregates(sites)
    # Non-paralog: sum=0.5+0.3=0.8 -> spec = 100 / (1+0.8) = 55.555...
    assert count_non == 2
    assert cfd_non == pytest.approx(100.0 / (1.0 + 0.8), abs=1e-9)
    # Paralog: sum=0.9+0.4=1.3 -> spec = 100 / (1+1.3)
    assert count_par == 2
    assert cfd_par == pytest.approx(100.0 / (1.0 + 1.3), abs=1e-9)


def test_split_cfd_aggregates_empty_buckets_yield_perfect_score() -> None:
    """No off-targets in either bucket -> both aggregates 100.0."""
    from bionpu.genomics.pe_design.ranker import split_cfd_aggregates

    cfd_non, cfd_par, count_non, count_par = split_cfd_aggregates([])
    assert cfd_non == 100.0
    assert cfd_par == 100.0
    assert count_non == 0
    assert count_par == 0


# ---------------------------------------------------------------------------
# Test 2 — Composite uses non-paralog aggregate only
# ---------------------------------------------------------------------------


def test_composite_uses_non_paralog_aggregate_only() -> None:
    """Adding in-paralog off-targets MUST NOT degrade the composite
    (they're informational, not a safety penalty)."""
    from bionpu.genomics.pe_design.ranker import rank_candidates

    edit_spec = _make_substitution_spec()
    cand = _make_pe2_candidate(nick_site=49, pbs_length=13)

    # Two off-target site lists exposing the same NON-paralog CFD
    # aggregate; one has an additional huge paralog cluster.
    base_sites = [
        OffTargetSite(chrom="chrSyn", pos=100, strand="+", mismatches=0, cfd_score=1.0),
        OffTargetSite(chrom="chrSyn", pos=500, strand="+", mismatches=3, cfd_score=0.2, in_paralog=False),
    ]
    paralog_sites = base_sites + [
        OffTargetSite(chrom="chrSyn", pos=1000 + i, strand="+", mismatches=2, cfd_score=0.8, in_paralog=True)
        for i in range(50)  # huge paralog cluster
    ]

    canned = PRIDICTScore(efficiency=70.0, edit_rate=0.7, confidence=1.0, notes=())

    class _Stub:
        def __init__(self, sites):
            self.sites = sites

        def score(self, *_a, **_kw):
            return canned

    def _scan_factory(sites):
        def _scan(spacer, *, max_mismatches=4, **_kw):
            # Aggregate computed from the sites list per CRISPOR.
            non_par = [s.cfd_score for s in sites if not s.in_paralog and s.mismatches > 0]
            agg = 100.0 * 100.0 / (100.0 + 100.0 * sum(non_par)) if non_par else 100.0
            return (sites, agg, len(sites))
        return _scan

    rows_no_par = rank_candidates(
        candidates=[cand],
        edit_spec=edit_spec,
        target_context="GCC(C/T)GTG",
        off_target_scan_fn=_scan_factory(base_sites),
        scorer=_Stub(base_sites),
        folding_extractor=_folding_features_factory(),
    )
    rows_with_par = rank_candidates(
        candidates=[cand],
        edit_spec=edit_spec,
        target_context="GCC(C/T)GTG",
        off_target_scan_fn=_scan_factory(paralog_sites),
        scorer=_Stub(paralog_sites),
        folding_extractor=_folding_features_factory(),
    )

    # The composite must be IDENTICAL in both cases — paralog hits do
    # not penalise the composite.
    assert rows_no_par[0].composite_pridict == pytest.approx(
        rows_with_par[0].composite_pridict, abs=1e-9
    )
    # The paralog version reports a non-zero paralog count.
    assert rows_no_par[0].paralog_hit_count_pegrna == 0
    assert rows_with_par[0].paralog_hit_count_pegrna == 50
    # And reports a paralog-bucket aggregate distinctly different from
    # the non-paralog aggregate.
    assert rows_with_par[0].cfd_aggregate_paralog_pegrna < 100.0
    assert rows_with_par[0].cfd_aggregate_paralog_pegrna != pytest.approx(
        rows_with_par[0].cfd_aggregate_pegrna
    )


# ---------------------------------------------------------------------------
# Test 3 — PRIDICT G-prefix retry recovers HBB-shaped non-G spacer
# ---------------------------------------------------------------------------


def test_pridict_gprefix_retry_recovers_non_g_spacer() -> None:
    """When PRIDICT returns PEGRNA_NOT_ENUMERATED_BY_PRIDICT for a
    non-G-prefix spacer, the ranker retries with G + spacer[1:] and
    adopts the retry's efficiency."""
    from bionpu.genomics.pe_design.ranker import rank_candidates

    edit_spec = _make_substitution_spec()
    # Non-G-prefix HBB-shaped spacer (matches the 2026-04-29 smoke
    # regression surfaced HBB top-1 spacer pattern: starts with A).
    non_g_spacer = "ATATCCCCCAGTTTAGTAGT"
    cand = _make_pe2_candidate(nick_site=49, spacer_seq=non_g_spacer)

    nan_score = PRIDICTScore(
        efficiency=float("nan"),
        edit_rate=float("nan"),
        confidence=float("nan"),
        notes=("PRIDICT_FAILED", "PEGRNA_NOT_ENUMERATED_BY_PRIDICT"),
    )
    real_score = PRIDICTScore(
        efficiency=63.5, edit_rate=0.635, confidence=1.0, notes=()
    )

    seen_spacers: list[str | None] = []

    class _GPrefixStub:
        def score(
            self,
            pegrna_seq,
            *,
            scaffold_variant="sgRNA_canonical",
            target_context="",
            folding_features=None,
            spacer_dna=None,
            pbs_dna=None,
            rtt_dna=None,
        ):
            seen_spacers.append(spacer_dna)
            if spacer_dna is None:
                return nan_score
            if spacer_dna[0].upper() == "G":
                return real_score
            return nan_score

    rows = rank_candidates(
        candidates=[cand],
        edit_spec=edit_spec,
        target_context="GCC(C/T)GTG",
        off_target_scan_fn=lambda spacer, *, max_mismatches=4, **_kw: ([], 100.0, 0),
        scorer=_GPrefixStub(),
        folding_extractor=_folding_features_factory(),
    )

    assert len(rows) == 1
    row = rows[0]
    # First call uses the original (non-G) spacer; retry uses G-prefix.
    assert seen_spacers[0] == non_g_spacer
    assert seen_spacers[1] == "G" + non_g_spacer[1:]
    # Composite reflects the retry's efficiency, NOT NaN.
    assert not math.isnan(row.composite_pridict)
    assert row.pridict_efficiency == 63.5
    # The retry tag is propagated to notes.
    assert "PRIDICT_GPREFIX_RETRY" in row.notes
    # PEGRNA_NOT_ENUMERATED_BY_PRIDICT is NOT present on the row (the
    # retry succeeded, so no failure flag).
    assert "PEGRNA_NOT_ENUMERATED_BY_PRIDICT" not in row.notes
    assert "PRIDICT_FAILED" not in row.notes


# ---------------------------------------------------------------------------
# Test 4 — G-prefix retry NOT triggered when spacer already starts with G
# ---------------------------------------------------------------------------


def test_no_gprefix_retry_when_spacer_already_starts_with_g() -> None:
    """A G-prefix spacer that fails enumeration MUST NOT trigger a
    retry (would call back with the same spacer; pointless work)."""
    from bionpu.genomics.pe_design.ranker import rank_candidates

    edit_spec = _make_substitution_spec()
    g_spacer = "GACTGTTTATAGCTGTTGGA"  # canonical G-prefix
    cand = _make_pe2_candidate(nick_site=49, spacer_seq=g_spacer)

    nan_score = PRIDICTScore(
        efficiency=float("nan"),
        edit_rate=float("nan"),
        confidence=float("nan"),
        notes=("PRIDICT_FAILED", "PEGRNA_NOT_ENUMERATED_BY_PRIDICT"),
    )

    call_count = {"n": 0}

    class _CountingStub:
        def score(self, *_a, **kwargs):
            call_count["n"] += 1
            return nan_score

    rows = rank_candidates(
        candidates=[cand],
        edit_spec=edit_spec,
        target_context="GCC(C/T)GTG",
        off_target_scan_fn=lambda spacer, *, max_mismatches=4, **_kw: ([], 100.0, 0),
        scorer=_CountingStub(),
        folding_extractor=_folding_features_factory(),
    )

    # Exactly one PRIDICT call (no retry).
    assert call_count["n"] == 1
    # Row carries PEGRNA_NOT_ENUMERATED_BY_PRIDICT but NOT the retry tag.
    assert "PRIDICT_GPREFIX_RETRY" not in rows[0].notes
    assert "PEGRNA_NOT_ENUMERATED_BY_PRIDICT" in rows[0].notes


# ---------------------------------------------------------------------------
# Test 5 — RankedPegRNA carries paralog fields populated from sites
# ---------------------------------------------------------------------------


def test_ranked_pegrna_populates_paralog_aggregate_fields() -> None:
    """The RankedPegRNA carries the new v0.1 paralog-bucket fields and
    they reflect the partition of in_paralog hits."""
    from bionpu.genomics.pe_design.ranker import rank_candidates

    edit_spec = _make_substitution_spec()
    cand = _make_pe2_candidate(nick_site=49, pbs_length=13)

    sites = [
        OffTargetSite(chrom="chrSyn", pos=100, strand="+", mismatches=0, cfd_score=1.0),
        OffTargetSite(chrom="chrSyn", pos=200, strand="+", mismatches=2, cfd_score=0.5, in_paralog=False),
        OffTargetSite(chrom="chrSyn", pos=300, strand="+", mismatches=2, cfd_score=0.7, in_paralog=True),
        OffTargetSite(chrom="chrSyn", pos=400, strand="+", mismatches=3, cfd_score=0.3, in_paralog=True),
    ]

    canned = PRIDICTScore(efficiency=70.0, edit_rate=0.7, confidence=1.0, notes=())

    class _Stub:
        def score(self, *_a, **_kw):
            return canned

    def _scan(spacer, *, max_mismatches=4, **_kw):
        non_par = [s.cfd_score for s in sites if not s.in_paralog and s.mismatches > 0]
        agg = 100.0 * 100.0 / (100.0 + 100.0 * sum(non_par)) if non_par else 100.0
        return (sites, agg, len(sites))

    rows = rank_candidates(
        candidates=[cand],
        edit_spec=edit_spec,
        target_context="GCC(C/T)GTG",
        off_target_scan_fn=_scan,
        scorer=_Stub(),
        folding_extractor=_folding_features_factory(),
    )

    row = rows[0]
    assert row.paralog_hit_count_pegrna == 2
    # Paralog bucket: sum=0.7+0.3=1.0 -> spec = 100/(1+1) = 50
    assert row.cfd_aggregate_paralog_pegrna == pytest.approx(50.0, abs=1e-6)
    # Non-paralog bucket: sum=0.5 -> spec = 100/(1+0.5) = 66.66...
    assert row.cfd_aggregate_pegrna == pytest.approx(100.0 / 1.5, abs=1e-6)
    assert row.off_target_count_pegrna == 1  # one non-paralog off-target


# ---------------------------------------------------------------------------
# Test 6 — Top-K bound preserves aggregate when count is below the bound
# ---------------------------------------------------------------------------


def test_topk_bound_preserves_aggregate_below_bound() -> None:
    """When the per-bucket hit count is below TOPK_CFD_AGGREGATE_BOUND,
    the aggregate equals the all-rows aggregate (Top-K is a no-op)."""
    from bionpu.genomics.pe_design.ranker import (
        TOPK_CFD_AGGREGATE_BOUND,
        split_cfd_aggregates,
    )

    n = TOPK_CFD_AGGREGATE_BOUND - 1
    sites = [
        OffTargetSite(
            chrom="chrA", pos=100 + i, strand="+", mismatches=2, cfd_score=0.01
        )
        for i in range(n)
    ]
    cfd_non, _, count_non, _ = split_cfd_aggregates(sites)
    expected = 100.0 * 100.0 / (100.0 + 100.0 * 0.01 * n)
    assert cfd_non == pytest.approx(expected, abs=1e-9)
    assert count_non == n
