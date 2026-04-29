# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

"""End-to-end pipeline orchestrator (Track C v0).

Glues per-gene design + dedup + balance + control generation +
library assembly into a single function. Lives in its own module so
the public API surface in ``__init__.py`` stays compact.

Library type handling
---------------------
v0 supports only ``library_type="knockout"``. ``activation`` and
``interference`` raise :class:`NotImplementedError` with a v1
deferral note (different PAM rules + Cas9-VPR / dCas9-KRAB scoring,
both out of scope per the build brief).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

from .controls import generate_controls
from .library_balancer import balance_library, global_dedup
from .output import LibraryGuide, assemble_library
from .per_gene_designer import design_per_gene_pools

__all__ = [
    "PipelineMetrics",
    "run_library_pipeline",
]


SUPPORTED_LIBRARY_TYPES: frozenset[str] = frozenset({"knockout"})
DEFERRED_LIBRARY_TYPES: frozenset[str] = frozenset({"activation", "interference"})


class PipelineMetrics:
    """Wall-clock + counts for the run; consumed by measurements.json.

    Not a dataclass: we want `.to_json()` semantics + flexible
    accumulation in the orchestrator.
    """

    __slots__ = (
        "per_gene_walls_s",
        "total_wall_s",
        "dedup_collisions",
        "n_safe_harbor",
        "n_essential",
        "n_non_targeting",
        "n_per_gene_rows",
        "n_under_balanced_genes",
    )

    def __init__(self) -> None:
        self.per_gene_walls_s: dict[str, float] = {}
        self.total_wall_s: float = 0.0
        self.dedup_collisions: int = 0
        self.n_safe_harbor: int = 0
        self.n_essential: int = 0
        self.n_non_targeting: int = 0
        self.n_per_gene_rows: int = 0
        self.n_under_balanced_genes: int = 0

    def to_json(self) -> dict[str, object]:
        return {
            "per_gene_walls_s": dict(self.per_gene_walls_s),
            "total_wall_s": float(self.total_wall_s),
            "dedup_collisions": int(self.dedup_collisions),
            "n_safe_harbor": int(self.n_safe_harbor),
            "n_essential": int(self.n_essential),
            "n_non_targeting": int(self.n_non_targeting),
            "n_per_gene_rows": int(self.n_per_gene_rows),
            "n_under_balanced_genes": int(self.n_under_balanced_genes),
        }


def _count_dedup_collisions(pools) -> int:
    """How many spacers appear in 2+ pools' full ranked lists?"""
    seen: dict[str, set[str]] = {}
    for pool in pools:
        seen_in_this_gene: set[str] = set()
        for g in pool.ranked_full:
            spacer = g.guide_seq.upper()
            if spacer in seen_in_this_gene:
                continue
            seen_in_this_gene.add(spacer)
            seen.setdefault(spacer, set()).add(pool.gene)
    return sum(1 for owners in seen.values() if len(owners) >= 2)


def run_library_pipeline(
    *,
    targets: Sequence[str],
    library_type: str,
    guides_per_gene: int,
    genome: str,
    fasta_path: Path | str | None,
    n_controls: int,
    max_mismatches: int,
    gc_min: float,
    gc_max: float,
    device: str,
    rank_by: str,
    pool_oversample: int,
    dedup_strategy: str,
    rng_seed: int,
    silicon_lock_label: str | None,
    metrics: PipelineMetrics | None = None,
) -> list[LibraryGuide]:
    """Track C v0 pipeline. Returns the assembled :class:`LibraryGuide` list.

    The ``metrics`` argument lets the CLI receive metrics in-band
    without piggy-backing them on the return value (which is a flat
    list for ergonomics + easy schema-validation in tests). When
    ``metrics`` is provided, this function fills it in place; when
    ``None``, no metrics are recorded.
    """
    if library_type in DEFERRED_LIBRARY_TYPES:
        raise NotImplementedError(
            f"library_type={library_type!r} is deferred to v1 "
            "(different PAM rules + Cas9-VPR / dCas9-KRAB scoring); "
            "Track C v0 supports only 'knockout' (NGG SpCas9). "
            "See PRDs/PRD-crispr-state-of-the-art-roadmap.md §3.3 "
            "Phase 3 for the activation/interference roadmap."
        )
    if library_type not in SUPPORTED_LIBRARY_TYPES:
        raise ValueError(
            f"library_type must be one of {sorted(SUPPORTED_LIBRARY_TYPES)} "
            f"in v0; got {library_type!r}"
        )
    if not targets:
        raise ValueError("targets list is empty; nothing to design")
    if fasta_path is None:
        raise ValueError(
            "fasta_path is required for Track C v0 (per-gene designer "
            "needs a GRCh38 FASTA to slice loci from)"
        )

    metrics = metrics if metrics is not None else PipelineMetrics()

    # ---- Step 1: per-gene design pools --------------------------------------
    t_total_start = time.perf_counter()

    pools = design_per_gene_pools(
        targets=list(targets),
        genome=genome,
        fasta_path=fasta_path,
        guides_per_gene=guides_per_gene,
        pool_oversample=pool_oversample,
        max_mismatches=max_mismatches,
        gc_min=gc_min,
        gc_max=gc_max,
        device=device,
        rank_by=rank_by,
        silicon_lock_label=silicon_lock_label,
    )
    for pool in pools:
        timings = pool.run_result.stage_timings_s
        metrics.per_gene_walls_s[pool.gene] = float(sum(timings.values()))

    # ---- Step 2: cross-pool dedup ------------------------------------------
    metrics.dedup_collisions = _count_dedup_collisions(pools)
    chosen = global_dedup(pools, strategy=dedup_strategy, rank_by=rank_by)

    # ---- Step 3: balance verification --------------------------------------
    reports = balance_library(pools, chosen)
    metrics.n_under_balanced_genes = sum(
        1 for r in reports if r.under_balanced
    )
    metrics.n_per_gene_rows = sum(len(r.selected) for r in reports)

    # ---- Step 4: control generation ----------------------------------------
    # The control generator uses an in-memory genome dict for its
    # exact-match check. For Track C v0 we deliberately pass
    # ``genome_seq_lookup=None`` (the brief allows this fallback —
    # "if non-targeting controls are slow to verify, use a CPU-side
    # prefilter or skip"). Random 20-mers have ~1e-12 collision
    # probability with the human genome; the safe-harbor + essential
    # control sets are pinned to canonical sequences anyway.
    controls = generate_controls(
        n_non_targeting=n_controls,
        genome_seq_lookup=None,
        rng_seed=rng_seed,
        gc_min=gc_min,
        gc_max=gc_max,
    )
    metrics.n_non_targeting = sum(
        1 for c in controls if c.control_class == "non_targeting"
    )
    metrics.n_safe_harbor = sum(
        1 for c in controls if c.control_class == "safe_harbor"
    )
    metrics.n_essential = sum(
        1 for c in controls if c.control_class == "essential_gene"
    )

    # ---- Step 5: assemble + return -----------------------------------------
    rows = assemble_library(balance_reports=reports, controls=controls)
    metrics.total_wall_s = time.perf_counter() - t_total_start
    return rows
