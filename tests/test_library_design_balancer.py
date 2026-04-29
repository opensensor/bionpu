# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

"""Unit tests for ``bionpu.genomics.library_design.library_balancer``.

These tests are pure-unit (no FASTA, no design-pipeline run) — we
build :class:`PerGenePool` records by hand to exercise dedup +
balance logic in isolation.
"""

from __future__ import annotations

import pytest

from bionpu.genomics.crispr_design import RankedGuide
from bionpu.genomics.library_design.library_balancer import (
    DEDUP_STRATEGIES,
    BalanceReport,
    balance_library,
    global_dedup,
)
from bionpu.genomics.library_design.per_gene_designer import PerGenePool


class _FakeRun:
    """Stand-in for DesignRunResult.stage_timings_s consumers in tests."""

    stage_timings_s: dict[str, float] = {"target_resolve": 0.0}


def _mkguide(
    spacer: str,
    *,
    composite_crispor: float = 0.5,
    composite_bionpu: float | None = None,
    pos: int = 100,
    chrom: str = "syn",
) -> RankedGuide:
    return RankedGuide(
        rank=1,
        guide_id=f"g_{spacer}",
        guide_seq=spacer,
        pam_seq="TGG",
        strand="+",
        target_chrom=chrom,
        target_pos=pos,
        gc_pct=50.0,
        on_target_score=0.5,
        cfd_aggregate=80.0,
        off_target_count=0,
        top_off_targets="",
        composite_crispor=composite_crispor,
        composite_bionpu=(
            composite_bionpu
            if composite_bionpu is not None
            else composite_crispor
        ),
        ranked_by="crispor",
        predicted_indel="",
        notes="",
    )


def _mkpool(gene: str, guides: list[RankedGuide], guides_per_gene: int = 2) -> PerGenePool:
    return PerGenePool(
        gene=gene,
        guides_per_gene=guides_per_gene,
        pool_oversample=2,
        ranked_full=tuple(guides),
        run_result=_FakeRun(),
    )


def test_dedup_strategies_set_matches_module_doc():
    assert DEDUP_STRATEGIES == frozenset({"highest_score", "alphabetical"})


def test_global_dedup_no_collisions_returns_top_n_per_gene():
    """When pools share no spacers, every gene gets its top guides_per_gene."""
    pool_a = _mkpool(
        "A",
        [
            _mkguide("AAAACCCCGGGGTTTTACGT", composite_crispor=0.9),
            _mkguide("CCCCGGGGTTTTAAAACGTA", composite_crispor=0.8),
            _mkguide("GGGGTTTTAAAACCCCGTAC", composite_crispor=0.7),
        ],
    )
    pool_b = _mkpool(
        "B",
        [
            _mkguide("TTTAAACCCAGGGCCCAGGT", composite_crispor=0.85),
            _mkguide("GAGAGAGAGAGAGAGAACGT", composite_crispor=0.75),
        ],
    )
    chosen = global_dedup([pool_a, pool_b], strategy="highest_score")
    assert len(chosen["A"]) == 2
    assert len(chosen["B"]) == 2


def test_global_dedup_highest_score_keeps_better_gene():
    """When 2 genes claim the same spacer, the higher-composite wins."""
    shared = "AAAACCCCGGGGTTTTACGT"
    pool_a = _mkpool(
        "A",
        [
            _mkguide(shared, composite_crispor=0.95),  # higher
            _mkguide("CCCCGGGGTTTTAAAACGTA", composite_crispor=0.5),
        ],
    )
    pool_b = _mkpool(
        "B",
        [
            _mkguide(shared, composite_crispor=0.4),  # lower; loses
            _mkguide("GGGGTTTTAAAACCCCGTAC", composite_crispor=0.3),
        ],
    )
    chosen = global_dedup([pool_a, pool_b], strategy="highest_score")
    a_spacers = {g.guide_seq for g in chosen["A"]}
    b_spacers = {g.guide_seq for g in chosen["B"]}
    assert shared in a_spacers
    assert shared not in b_spacers


def test_global_dedup_alphabetical_keeps_lex_first_gene():
    """``alphabetical`` strategy is determinism-only: lex-first gene wins."""
    shared = "AAAACCCCGGGGTTTTACGT"
    pool_a = _mkpool(
        "Z_GENE",
        [
            _mkguide(shared, composite_crispor=0.99),  # higher score, but ...
            _mkguide("CCCCGGGGTTTTAAAACGTA", composite_crispor=0.5),
        ],
    )
    pool_b = _mkpool(
        "A_GENE",  # ... lex-first
        [
            _mkguide(shared, composite_crispor=0.1),
            _mkguide("GGGGTTTTAAAACCCCGTAC", composite_crispor=0.3),
        ],
    )
    chosen = global_dedup([pool_a, pool_b], strategy="alphabetical")
    a_spacers = {g.guide_seq for g in chosen["A_GENE"]}
    z_spacers = {g.guide_seq for g in chosen["Z_GENE"]}
    assert shared in a_spacers  # alphabetical winner
    assert shared not in z_spacers


def test_global_dedup_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="strategy must be"):
        global_dedup([_mkpool("A", [_mkguide("AAAACCCCGGGGTTTTACGT")])],
                     strategy="random")


def test_balance_library_flags_under_balanced_genes():
    """A gene whose chosen list is short of guides_per_gene shows under_balanced."""
    pool_a = _mkpool(
        "A",
        [_mkguide("AAAACCCCGGGGTTTTACGT")],
        guides_per_gene=4,
    )
    chosen = {"A": [pool_a.ranked_full[0]]}
    reports = balance_library([pool_a], chosen)
    assert len(reports) == 1
    assert isinstance(reports[0], BalanceReport)
    assert reports[0].under_balanced is True
    assert reports[0].deficit == 3
    assert reports[0].requested == 4
    assert len(reports[0].selected) == 1


def test_balance_library_healthy_gene_has_zero_deficit():
    pool_a = _mkpool(
        "A",
        [
            _mkguide("AAAACCCCGGGGTTTTACGT"),
            _mkguide("CCCCGGGGTTTTAAAACGTA"),
        ],
        guides_per_gene=2,
    )
    chosen = {"A": list(pool_a.ranked_full)}
    reports = balance_library([pool_a], chosen)
    assert reports[0].under_balanced is False
    assert reports[0].deficit == 0


def test_global_dedup_uses_bionpu_composite_when_rank_by_bionpu():
    """rank_by='bionpu' should drive the highest_score tiebreak via composite_bionpu."""
    shared = "AAAACCCCGGGGTTTTACGT"
    pool_a = _mkpool(
        "A",
        [
            # crispor lower, bionpu higher
            _mkguide(shared, composite_crispor=0.3, composite_bionpu=0.9),
            _mkguide("CCCCGGGGTTTTAAAACGTA", composite_crispor=0.5, composite_bionpu=0.5),
        ],
    )
    pool_b = _mkpool(
        "B",
        [
            # crispor higher, bionpu lower
            _mkguide(shared, composite_crispor=0.9, composite_bionpu=0.3),
            _mkguide("GGGGTTTTAAAACCCCGTAC", composite_crispor=0.4, composite_bionpu=0.4),
        ],
    )
    chosen = global_dedup([pool_a, pool_b], strategy="highest_score", rank_by="bionpu")
    a_spacers = {g.guide_seq for g in chosen["A"]}
    b_spacers = {g.guide_seq for g in chosen["B"]}
    # A wins the shared spacer because its bionpu composite is higher
    # (0.9 vs 0.3) — even though its crispor composite is lower.
    assert shared in a_spacers
    assert shared not in b_spacers
