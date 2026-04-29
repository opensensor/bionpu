# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

"""Per-gene CRISPR guide pool generator (Track C v0).

Wraps :func:`bionpu.genomics.crispr_design.design_guides_for_target`
in a multi-target loop. The single-target wrapper is reused by
import (no duplication, no edit) per the Track C build brief.

The pool produced by this module has these properties:

* Each gene is run independently; failures (e.g. unknown gene symbol)
  raise :class:`bionpu.genomics.crispr_design.GeneNotFoundError`.
* The per-gene pool is oversized vs. the library target
  (``top_n = guides_per_gene * pool_oversample``) so the global dedup
  pass can drop colliding guides without dropping the gene below the
  library balance threshold.
* Returned objects are :class:`PerGenePool` records carrying both the
  full ranked-guide list (for fallback) and the chosen top-N slice.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from bionpu.genomics.crispr_design import (
    DEFAULT_GC_MAX,
    DEFAULT_GC_MIN,
    DEFAULT_MAX_MISMATCHES,
    DesignRunResult,
    RankedGuide,
    design_guides_for_target,
)

__all__ = [
    "PerGenePool",
    "design_per_gene_pools",
    "select_top_guides_for_gene",
]


@dataclass(frozen=True, slots=True)
class PerGenePool:
    """A per-gene guide pool: full ranked-guide list + the top-N slice.

    ``ranked_full`` is the full top_n from
    :func:`bionpu.genomics.crispr_design.design_guides_for_target`; we
    keep the entire list so the global dedup pass has fallbacks if a
    high-ranked guide collides with another gene's top guide.

    ``run_result`` is the underlying :class:`DesignRunResult` (target
    metadata + stage timings + TSV bytes); callers can drop it into
    measurement JSON.
    """

    gene: str
    guides_per_gene: int
    pool_oversample: int
    ranked_full: tuple[RankedGuide, ...]
    run_result: DesignRunResult


def design_per_gene_pools(
    targets: Sequence[str],
    *,
    genome: str,
    fasta_path: Path | str,
    guides_per_gene: int,
    pool_oversample: int = 4,
    max_mismatches: int = DEFAULT_MAX_MISMATCHES,
    gc_min: float = DEFAULT_GC_MIN,
    gc_max: float = DEFAULT_GC_MAX,
    device: str = "cpu",
    rank_by: str = "crispor",
    silicon_lock_label: str | None = None,
) -> list[PerGenePool]:
    """Run :func:`design_guides_for_target` once per target gene.

    Returns a list of :class:`PerGenePool` records in input order.

    The per-gene wall and total wall accumulated by this function are
    captured in the per-pool ``run_result.stage_timings_s`` field; the
    pipeline aggregator (see ``pipeline.py``) sums them for the
    measurements JSON.
    """
    if guides_per_gene <= 0:
        raise ValueError(
            f"guides_per_gene must be >= 1; got {guides_per_gene}"
        )
    if pool_oversample <= 0:
        raise ValueError(
            f"pool_oversample must be >= 1; got {pool_oversample}"
        )

    pools: list[PerGenePool] = []
    top_n = guides_per_gene * pool_oversample
    for gene in targets:
        # Each gene runs its full single-target pipeline; this is the
        # composition pattern called out in the Track C build brief.
        run_label = (
            silicon_lock_label
            or f"bionpu_library_design:{gene}"
        )
        run = design_guides_for_target(
            target=gene,
            genome=genome,
            fasta_path=fasta_path,
            top_n=top_n,
            max_mismatches=max_mismatches,
            gc_min=gc_min,
            gc_max=gc_max,
            device=device,
            rank_by=rank_by,
            silicon_lock_label=run_label,
        )
        pools.append(
            PerGenePool(
                gene=gene,
                guides_per_gene=guides_per_gene,
                pool_oversample=pool_oversample,
                ranked_full=tuple(run.ranked),
                run_result=run,
            )
        )
    return pools


def select_top_guides_for_gene(
    pool: PerGenePool,
    *,
    forbidden_spacers: Iterable[str] = (),
) -> list[RankedGuide]:
    """Pick the top ``pool.guides_per_gene`` guides, skipping forbidden spacers.

    ``forbidden_spacers`` is the set of guide-sequences already claimed
    by sibling genes in the dedup pass. The top-N traversal skips those
    and falls back to lower-ranked guides until either the requested
    quota is satisfied or the full ranked list is exhausted (in which
    case the gene is under-balanced — the library balancer flags this).
    """
    forbidden = frozenset(s.upper() for s in forbidden_spacers)
    chosen: list[RankedGuide] = []
    for g in pool.ranked_full:
        if g.guide_seq.upper() in forbidden:
            continue
        chosen.append(g)
        if len(chosen) >= pool.guides_per_gene:
            break
    return chosen
