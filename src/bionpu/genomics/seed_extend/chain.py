# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

"""Host-side seed chaining for minimizer seed hits.

The NPU-friendly part of seed_extend is seed extraction. Chaining is a
small dynamic-programming pass over seed anchors and stays on the host in
v1, matching the minimap2 division of labor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from bionpu.genomics.seed_extend.lookup import SeedHit

__all__ = [
    "SeedChain",
    "chain_seed_hits",
]


@dataclass(frozen=True, slots=True)
class SeedChain:
    """One colinear seed chain."""

    seeds: tuple[SeedHit, ...]
    score: int
    query_start: int
    query_end: int
    ref_start: int
    ref_end: int
    strand: int
    diagonal: int

    @property
    def n_seeds(self) -> int:
        return len(self.seeds)


def chain_seed_hits(
    seeds: Sequence[SeedHit],
    *,
    max_diag_gap: int = 32,
    max_query_gap: int = 10_000,
    max_ref_gap: int = 10_000,
    min_chain_score: int = 2,
    max_chains: int | None = None,
) -> list[SeedChain]:
    """Group seed hits into simple colinear chains.

    Two anchors can chain when both query and reference positions
    increase, their diagonal ``ref_pos - query_pos`` stays within
    ``max_diag_gap``, and both coordinate gaps stay bounded. The score is
    the number of anchors in the chain.
    """
    if max_diag_gap < 0:
        raise ValueError("max_diag_gap must be non-negative")
    if max_query_gap < 0 or max_ref_gap < 0:
        raise ValueError("max_query_gap and max_ref_gap must be non-negative")
    if min_chain_score <= 0:
        raise ValueError("min_chain_score must be positive")
    if max_chains is not None and max_chains <= 0:
        raise ValueError("max_chains must be positive when provided")

    anchors = tuple(
        sorted(
            seeds,
            key=lambda h: (int(h.strand), int(h.ref_pos), int(h.query_pos)),
        )
    )
    n = len(anchors)
    if n == 0:
        return []

    scores = [1] * n
    prev = [-1] * n
    for i in range(n):
        hi = anchors[i]
        diag_i = int(hi.ref_pos) - int(hi.query_pos)
        for j in range(i):
            hj = anchors[j]
            if int(hj.strand) != int(hi.strand):
                continue
            dq = int(hi.query_pos) - int(hj.query_pos)
            dr = int(hi.ref_pos) - int(hj.ref_pos)
            if dq <= 0 or dr <= 0:
                continue
            if dq > max_query_gap or dr > max_ref_gap:
                continue
            diag_j = int(hj.ref_pos) - int(hj.query_pos)
            if abs(diag_i - diag_j) > max_diag_gap:
                continue
            candidate = scores[j] + 1
            if candidate > scores[i]:
                scores[i] = candidate
                prev[i] = j

    used: set[int] = set()
    chains: list[SeedChain] = []
    ranked = sorted(
        range(n),
        key=lambda i: (
            -scores[i],
            int(anchors[i].ref_pos),
            int(anchors[i].query_pos),
        ),
    )
    for end_idx in ranked:
        if scores[end_idx] < min_chain_score:
            break
        idxs: list[int] = []
        cur = end_idx
        while cur >= 0 and cur not in used:
            idxs.append(cur)
            cur = prev[cur]
        if len(idxs) < min_chain_score:
            continue
        idxs.reverse()
        for idx in idxs:
            used.add(idx)
        chain_seeds = tuple(anchors[idx] for idx in idxs)
        chains.append(_make_chain(chain_seeds))
        if max_chains is not None and len(chains) >= max_chains:
            break

    chains.sort(
        key=lambda c: (-c.score, c.ref_start, c.query_start, c.strand)
    )
    return chains


def _make_chain(seeds: tuple[SeedHit, ...]) -> SeedChain:
    first = seeds[0]
    last = seeds[-1]
    diagonals = [int(h.ref_pos) - int(h.query_pos) for h in seeds]
    diagonal = round(sum(diagonals) / len(diagonals))
    return SeedChain(
        seeds=seeds,
        score=len(seeds),
        query_start=int(first.query_pos),
        query_end=int(last.query_pos),
        ref_start=int(first.ref_pos),
        ref_end=int(last.ref_pos),
        strand=int(first.strand),
        diagonal=int(diagonal),
    )
