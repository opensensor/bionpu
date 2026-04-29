# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

"""Library-wide deduplication + balance enforcement (Track C v0).

Pooled CRISPR libraries cannot have duplicate guide spacers across
genes (every guide must be uniquely attributable in a screen
read-out). This module implements:

* :func:`global_dedup` — when guide X appears in >1 gene's ranked
  pool, keep it only in the pool where it scores highest (default) or
  in lexicographic-first gene (``"alphabetical"``, deterministic
  test-friendly).
* :func:`balance_library` — verify the post-dedup pools all hit the
  ``guides_per_gene`` quota; emit a structured warning per
  under-balanced gene with the chosen-count delta.

Strategy choice rationale: the published Brunello library uses a
top-quality dedup ("highest-quality-keeps-the-guide") because the
secondary collision is rare (genome-wide) and biological-relevance
trumps determinism. The ``alphabetical`` strategy exists for tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from bionpu.genomics.crispr_design import RankedGuide

from .per_gene_designer import PerGenePool, select_top_guides_for_gene

__all__ = [
    "BalanceReport",
    "balance_library",
    "global_dedup",
]

# Allowed dedup strategies for :func:`global_dedup`.
DEDUP_STRATEGIES: frozenset[str] = frozenset({"highest_score", "alphabetical"})


@dataclass(frozen=True, slots=True)
class BalanceReport:
    """Per-gene balance verdict.

    ``selected`` is the post-dedup top-N for the gene (length should
    equal ``pool.guides_per_gene`` for healthy genes; less for genes
    that lost too many to dedup).

    ``under_balanced`` is True iff ``len(selected) < pool.guides_per_gene``.
    The library writer continues anyway — under-balanced genes still
    appear in the library with whatever guides survived. Callers can
    inspect ``balance_reports`` and flag for post-hoc review.
    """

    gene: str
    selected: tuple[RankedGuide, ...]
    pool_size: int
    requested: int
    under_balanced: bool

    @property
    def deficit(self) -> int:
        return max(0, self.requested - len(self.selected))


def _composite_for(g: RankedGuide, rank_by: str = "crispor") -> float:
    """Read the composite score in the same direction as the design pipeline."""
    if rank_by == "bionpu":
        return float(g.composite_bionpu)
    return float(g.composite_crispor)


def global_dedup(
    pools: Sequence[PerGenePool],
    *,
    strategy: str = "highest_score",
    rank_by: str = "crispor",
) -> dict[str, list[RankedGuide]]:
    """Resolve cross-pool guide-sequence collisions.

    Returns a mapping ``gene -> [chosen RankedGuide, ...]`` such that
    no spacer string appears in more than one gene's chosen list.

    The chosen list per gene is the top ``pool.guides_per_gene`` guides
    in pool order *minus* spacers that were claimed by another gene
    under the chosen strategy.

    Parameters
    ----------
    pools:
        Per-gene pools produced by
        :func:`bionpu.genomics.library_design.per_gene_designer.design_per_gene_pools`.
    strategy:
        ``"highest_score"`` (default) or ``"alphabetical"``. See module
        docstring.
    rank_by:
        Which composite drives the ``highest_score`` tiebreak. Mirrors
        the per-gene design ``rank_by`` argument.
    """
    if strategy not in DEDUP_STRATEGIES:
        raise ValueError(
            f"strategy must be one of {sorted(DEDUP_STRATEGIES)}; "
            f"got {strategy!r}"
        )

    # Build the spacer -> [(gene, RankedGuide), ...] index over the
    # *full* ranked pool of every gene — not just the top-N — so the
    # dedup decision is based on the entire candidate set.
    spacer_to_owners: dict[str, list[tuple[str, RankedGuide]]] = {}
    for pool in pools:
        seen_in_this_gene: set[str] = set()
        for g in pool.ranked_full:
            spacer = g.guide_seq.upper()
            if spacer in seen_in_this_gene:
                # Same gene listing the same guide twice (paralog
                # placement): only count once per gene for collision
                # purposes; the per-gene ranker already collapses
                # multi-row identical-spacer entries via guide_id.
                continue
            seen_in_this_gene.add(spacer)
            spacer_to_owners.setdefault(spacer, []).append((pool.gene, g))

    # For each spacer with >1 owner, pick exactly one keeper per
    # strategy; the others get added to that gene's "forbidden"
    # set.
    forbidden_per_gene: dict[str, set[str]] = {p.gene: set() for p in pools}
    for spacer, owners in spacer_to_owners.items():
        if len(owners) <= 1:
            continue
        if strategy == "highest_score":
            # Keep the (gene, guide) with the highest composite. Ties
            # broken by alphabetical gene name for determinism.
            owners_sorted = sorted(
                owners,
                key=lambda go: (-_composite_for(go[1], rank_by=rank_by), go[0]),
            )
        else:  # "alphabetical"
            owners_sorted = sorted(owners, key=lambda go: go[0])
        keeper_gene = owners_sorted[0][0]
        for gene, _g in owners_sorted[1:]:
            forbidden_per_gene[gene].add(spacer)
        # Keeper gene is unrestricted on this spacer (no forbidden entry).
        # The keeper_gene is referenced in the loop but not bound; the
        # set logic above already excludes it.
        del keeper_gene  # silence linter

    # Now apply the forbidden sets per gene to pick the top-N.
    chosen: dict[str, list[RankedGuide]] = {}
    for pool in pools:
        chosen[pool.gene] = select_top_guides_for_gene(
            pool, forbidden_spacers=forbidden_per_gene[pool.gene]
        )
    return chosen


def balance_library(
    pools: Sequence[PerGenePool],
    chosen: dict[str, list[RankedGuide]],
) -> list[BalanceReport]:
    """Build per-gene :class:`BalanceReport` records.

    The returned list mirrors the input pool order. The library writer
    consumes this for the ``notes`` column (under-balanced genes get a
    ``"UNDER_BALANCED:deficit=N"`` note appended).
    """
    reports: list[BalanceReport] = []
    for pool in pools:
        sel = chosen.get(pool.gene, [])
        reports.append(
            BalanceReport(
                gene=pool.gene,
                selected=tuple(sel),
                pool_size=len(pool.ranked_full),
                requested=pool.guides_per_gene,
                under_balanced=len(sel) < pool.guides_per_gene,
            )
        )
    return reports
