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

"""Host-side seed lookup for seed-and-extend v0.

Given a list of query minimizers (computed on the NPU via the
``BionpuMinimizer`` op) and a reference :class:`MinimapIndex`, emit
``SeedHit`` tuples ``(query_pos, ref_pos, strand)`` for every (query
canonical) → (reference canonical) match.

Output ordering: sorted by ``ref_pos`` ascending, then by ``query_pos``
ascending — matches the ``mm_seed_t`` post-sort ordering minimap2 uses
before chaining.
"""

from __future__ import annotations

from dataclasses import dataclass

from bionpu.genomics.seed_extend.index import MinimapIndex


__all__ = [
    "DEFAULT_FREQ_CUTOFF",
    "SeedHit",
    "query_to_seeds",
    "query_to_seeds_from_minimizers",
]


@dataclass(frozen=True, slots=True)
class SeedHit:
    """A single seed hit.

    Mirrors the ``(query_pos, ref_pos, strand)`` triple emitted by
    minimap2's ``collect_seed_hits`` (``mm_seed_t``).

    * ``query_pos``: 0-indexed start position of the minimizer k-mer in
      the query.
    * ``ref_pos``: 0-indexed start position of the matching minimizer
      k-mer in the reference.
    * ``strand``: 0 placeholder for v0 (oracle is canonical-only).
    * ``canonical``: the canonical uint64 that matched (kept for debug
      / round-trip checks).
    """

    query_pos: int
    ref_pos: int
    strand: int
    canonical: int


DEFAULT_FREQ_CUTOFF = 1000
"""Per-canonical seed-hit cap.

Mirrors minimap2's ``-f`` filter discipline: minimizers that occur more
than this many times in the reference are uninformative for seeding
(they'd produce huge seed-hit fan-outs and dominate the chaining
budget). minimap2 default is ``-f 0.0002`` — top 0.02% by frequency;
for chr22 (~5 M minimizers) that's roughly 1 000 occurrences.

v0 design rationale: a v0 build without this cutoff hit a 54 GB RSS
runaway on the chr22 self-mapping silicon validation — chr22's
centromere N→A scrub creates ~12 M canonical=0 (all-A) minimizers in
the reference; a single query minimizer that lands in any homopolymer
region would emit seed-hits for ALL of them. Cutoff at 1 000 keeps the
emit volume bounded while preserving ≥99% of informative seeds (random
DNA per the minimap2 paper has very few minimizers above ~50
occurrences; the >1 000 tail is dominated by repeat / low-complexity
regions that aren't useful as anchor seeds anyway).
"""


def query_to_seeds_from_minimizers(
    query_minimizers: list[tuple[int, int]],
    index: MinimapIndex,
    *,
    freq_cutoff: int | None = DEFAULT_FREQ_CUTOFF,
) -> list[SeedHit]:
    """Emit seed hits for already-extracted query minimizers.

    Pure host-side helper — used by tests when a known minimizer list is
    available without a silicon round-trip. The full pipeline is in
    :class:`bionpu.genomics.seed_extend.SeedExtractor`.

    Args:
        query_minimizers: ``[(canonical, query_pos), ...]`` as returned
            by the silicon op or the CPU oracle.
        index: reference index built via
            :func:`bionpu.genomics.seed_extend.index.build_minimap2_index`.
        freq_cutoff: maximum reference occurrences per minimizer.
            Canonicals that map to MORE than ``freq_cutoff`` ref-
            positions are skipped (too uninformative for seeding;
            mirrors minimap2's ``-f`` filter). ``None`` disables the
            filter entirely (NOT recommended on chr22-scale references —
            see :data:`DEFAULT_FREQ_CUTOFF` for the rationale).

    Returns:
        ``[SeedHit(query_pos, ref_pos, strand, canonical), ...]`` sorted
        by ``(ref_pos asc, query_pos asc)``.
    """
    out: list[SeedHit] = []
    for canonical, query_pos in query_minimizers:
        hits = index.get(int(canonical))
        if not hits:
            continue
        if freq_cutoff is not None and len(hits) > freq_cutoff:
            continue
        for ref_pos, strand in hits:
            out.append(
                SeedHit(
                    query_pos=int(query_pos),
                    ref_pos=int(ref_pos),
                    strand=int(strand),
                    canonical=int(canonical),
                )
            )
    out.sort(key=lambda h: (h.ref_pos, h.query_pos))
    return out


def query_to_seeds(
    query_seq_or_packed,
    index: MinimapIndex,
    npu_op=None,
    *,
    n_bases: int | None = None,
    freq_cutoff: int | None = DEFAULT_FREQ_CUTOFF,
):
    """End-to-end NPU-backed query → seed hit lookup.

    Thin wrapper that routes to the silicon minimizer op if provided,
    otherwise falls back to the CPU oracle. Returns the same
    ``list[SeedHit]`` either way.

    Args:
        query_seq_or_packed: either an ACGT string (oracle path) or a
            uint8 packed-2-bit ``np.ndarray`` (NPU path).
        index: reference index.
        npu_op: optional :class:`bionpu.kernels.genomics.minimizer.BionpuMinimizer`
            instance. If ``None`` (or no artifacts), the CPU oracle is used.
        n_bases: required when ``query_seq_or_packed`` is a packed
            ndarray; ignored for strings.
    """
    import numpy as np

    from bionpu.data.minimizer_oracle import (
        extract_minimizers,
        extract_minimizers_packed,
    )

    if isinstance(query_seq_or_packed, str):
        if npu_op is not None:
            raise ValueError(
                "npu_op was provided but query is a str; pack via "
                "bionpu.data.kmer_oracle.pack_dna_2bit first."
            )
        mzs = extract_minimizers(
            query_seq_or_packed, k=index.k, w=index.w
        )
    elif isinstance(query_seq_or_packed, np.ndarray):
        if n_bases is None:
            raise ValueError(
                "n_bases is required for packed-ndarray queries"
            )
        if npu_op is not None and npu_op.artifacts_present():
            mzs = npu_op(packed_seq=query_seq_or_packed, top_n=0)
        else:
            mzs = extract_minimizers_packed(
                query_seq_or_packed,
                n_bases=int(n_bases),
                k=index.k,
                w=index.w,
            )
    else:
        raise TypeError(
            f"query must be str or np.ndarray; got {type(query_seq_or_packed)!r}"
        )

    return query_to_seeds_from_minimizers(mzs, index, freq_cutoff=freq_cutoff)
