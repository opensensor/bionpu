# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

"""CPU-only CRISPR guide-design prefilter workflow.

The workflow composes two host-side primitives:

1. :func:`bionpu.genomics.guide_design.enumerate_guides` enumerates
   PAM-valid candidate guides and applies cheap sequence filters.
2. :func:`bionpu.genomics.offtarget_seed.prefilter_offtargets` finds
   seed-compatible off-target candidates for passing guides.

This is intentionally pre-scoring: it produces bounded candidate sets
and per-guide counts that downstream DP/scoring can consume.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from bionpu.genomics.guide_design import (
    GuideCandidate,
    GuideFilter,
    enumerate_guides,
)
from bionpu.genomics.offtarget_seed import (
    GuideRecord,
    OffTargetSeedCandidate,
    prefilter_offtargets,
)

__all__ = [
    "GuideDesignResult",
    "GuideDesignRun",
    "design_guides",
]


@dataclass(frozen=True, slots=True)
class GuideDesignResult:
    """Per-guide seed-prefilter output."""

    guide: GuideCandidate
    off_targets: tuple[OffTargetSeedCandidate, ...]
    seed_hit_count: int
    exact_seed_hit_count: int
    mismatched_seed_hit_count: int
    reference_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GuideDesignRun:
    """Complete output of :func:`design_guides`."""

    candidates: tuple[GuideCandidate, ...]
    passing_guides: tuple[GuideCandidate, ...]
    rejected_guides: tuple[GuideCandidate, ...]
    results: tuple[GuideDesignResult, ...]
    seed_hit_count: int


def design_guides(
    target_seq: str,
    references: Mapping[str, str] | str,
    *,
    chrom: str = "target",
    offset: int = 0,
    pam_templates: Sequence[str] = ("NGG",),
    guide_filter: GuideFilter | None = None,
    seed_length: int = 12,
    max_seed_mismatches: int = 0,
    include_reverse: bool = True,
    pam_aware: bool = True,
) -> GuideDesignRun:
    """Enumerate guides and attach seed-compatible off-target candidates.

    Failed guide-filter candidates are retained in ``rejected_guides`` for
    reporting, but only passing guides are sent into the off-target seed
    prefilter.
    """
    base_filter = guide_filter or GuideFilter()
    report_filter = replace(base_filter, drop_failed=False)
    candidates = tuple(
        enumerate_guides(
            target_seq,
            chrom=chrom,
            offset=offset,
            pam_templates=pam_templates,
            guide_filter=report_filter,
            include_reverse=include_reverse,
        )
    )
    passing = tuple(c for c in candidates if c.passes_filters)
    rejected = tuple(c for c in candidates if not c.passes_filters)

    if not passing:
        return GuideDesignRun(
            candidates=candidates,
            passing_guides=(),
            rejected_guides=rejected,
            results=(),
            seed_hit_count=0,
        )

    guide_records = [
        GuideRecord(guide_id=c.guide_id, sequence=c.spacer) for c in passing
    ]
    off_targets = prefilter_offtargets(
        guide_records,
        references,
        seed_length=seed_length,
        max_seed_mismatches=max_seed_mismatches,
        pam=pam_templates,
        pam_aware=pam_aware,
        guide_length=base_filter.spacer_len,
    )

    by_guide: dict[str, list[OffTargetSeedCandidate]] = {
        c.guide_id: [] for c in passing
    }
    for hit in off_targets:
        by_guide.setdefault(hit.guide_id, []).append(hit)

    results: list[GuideDesignResult] = []
    for guide in passing:
        hits = tuple(by_guide.get(guide.guide_id, ()))
        results.append(
            GuideDesignResult(
                guide=guide,
                off_targets=hits,
                seed_hit_count=len(hits),
                exact_seed_hit_count=sum(h.seed_mismatches == 0 for h in hits),
                mismatched_seed_hit_count=sum(
                    h.seed_mismatches > 0 for h in hits
                ),
                reference_names=tuple(sorted({h.ref_name for h in hits})),
            )
        )

    return GuideDesignRun(
        candidates=candidates,
        passing_guides=passing,
        rejected_guides=rejected,
        results=tuple(results),
        seed_hit_count=sum(r.seed_hit_count for r in results),
    )
