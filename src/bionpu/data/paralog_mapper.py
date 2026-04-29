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

"""Track B v0.1 — Paralog gene-family mapper.

Maps a target gene symbol to its sibling paralog gene spans on the
reference genome. Used by the prime-editor design pipeline to mark
off-target hits that fall inside a paralog as "expected on-family"
hits and exclude them from the safety-penalty CFD aggregate (the
``cfd_aggregate_pegrna`` term in the composite score).

Why this exists
---------------
A pegRNA targeting HBB (β-globin, chr11) will produce dense off-target
hits in HBD / HBG1 / HBG2 / HBE1 / HBZ — every member of the hemoglobin
gene cluster shares substantial sequence identity. Summing the per-site
CFD scores honestly across the cluster yields very large aggregate
values; the design composite then either (a) under-ranks genuinely
specific pegRNAs because the paralog cluster swamps the safety penalty,
or (b) numerically NaN's once the aggregate exceeds the float-format
precision used downstream.

The Track B v0.1 fix ships TWO orthogonal mitigations: this paralog
mapper (mark in-paralog hits and exclude from the safety penalty) AND a
top-K CFD aggregator at the ranker level (numerical stability bound
even when the paralog map is empty). Both compose; either alone helps.

v0.1 paralog map source
-----------------------
v0.1 ships a hardcoded paralog map covering the five Track B v0 smoke
targets (BRCA1, TP53, EGFR, HBB, MYC) plus their canonical paralog
families. Production v1 ships full-coverage HGNC gene-group integration
via the ``https://www.genenames.org/cgi-bin/genegroup/download`` TSV
(gap entry: ``track-b-paralog-aware-composite-filter``).

The hardcoded map's coordinate values are GRCh38 / hg38; symbol -> spans
were taken from NCBI Gene + Ensembl GRCh38.p14. Coordinates are 1-based
inclusive (UCSC convention) to match :mod:`bionpu.data.genome_fetcher`'s
``resolve_gene_symbol`` output. The mapper exposes:

* :func:`get_paralog_spans` — symbol -> list of paralog ``GeneSpan``s
  (excludes the query symbol itself).
* :func:`is_in_any_paralog` — coordinate hit-test helper used by the
  T11 off-target adapter to tag :class:`OffTargetSite.in_paralog`.

Both helpers normalise gene symbols case-insensitively and return empty
results for unknown symbols (rather than raising) so callers without a
paralog hit stay on the v0 codepath transparently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

__all__ = [
    "GeneSpan",
    "get_paralog_spans",
    "is_in_any_paralog",
    "PARALOG_FAMILIES",
]


# ---------------------------------------------------------------------------
# Paralog gene span dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneSpan:
    """A single gene's reference span.

    Attributes
    ----------
    symbol:
        HGNC gene symbol (uppercase canonical form).
    chrom:
        Reference contig (e.g. ``"chr11"``).
    start:
        1-based inclusive start coordinate (matches
        :class:`bionpu.data.genome_fetcher.GeneCoord.start`).
    end:
        1-based inclusive end coordinate.
    """

    symbol: str
    chrom: str
    start: int
    end: int

    def contains(self, *, chrom: str, pos_0b: int) -> bool:
        """Return ``True`` when 0-based coordinate ``pos_0b`` on
        ``chrom`` falls inside this span.

        The off-target adapter emits 0-based positions per
        :class:`bionpu.genomics.pe_design.types.OffTargetSite.pos`; we
        convert to a 1-based test internally so callers don't have to
        know our internal coordinate convention.
        """
        if chrom != self.chrom:
            return False
        pos_1b = pos_0b + 1
        return self.start <= pos_1b <= self.end


# ---------------------------------------------------------------------------
# Hardcoded paralog families (v0.1 — Track B v0.1 smoke coverage)
#
# Each entry maps a "query gene symbol" -> tuple of paralog GeneSpans
# (canonically EXCLUDING the query gene itself; the query's own span is
# never a "paralog hit").
#
# Coverage:
#   * HBB family (the v0 smoke regression): hemoglobin cluster on chr11.
#   * BRCA1 family: BRCA1 has functional homology to BRCA2 (chr13) and
#     PALB2 (chr16) but no paralog-cluster off-target shadow; for
#     completeness we list BRCA2 as a candidate paralog.
#   * TP53 family: paralogs TP63 (chr3) and TP73 (chr1).
#   * EGFR family: paralogs ERBB2 (chr17), ERBB3 (chr12), ERBB4 (chr2).
#   * MYC family: paralogs MYCN (chr2) and MYCL (chr1).
#
# Coordinates: GRCh38.p14 (1-based inclusive) per NCBI Gene as of
# 2026-04-28.
# ---------------------------------------------------------------------------

# Each value is a tuple of GeneSpan; the canonical map is the union of
# all spans across all families. We construct family-keyed lookups
# below and a flat symbol -> family map for fast lookup.
PARALOG_FAMILIES: dict[str, tuple[GeneSpan, ...]] = {
    "HBB_FAMILY": (
        # GRCh38.p14 hemoglobin cluster on chr11p15.4 (NCBI Gene, 2026-04).
        # Coordinates are 1-based inclusive; spans cover the gene body
        # (mRNA boundaries, NOT just the CDS).
        GeneSpan(symbol="HBB", chrom="chr11", start=5225464, end=5229395),
        GeneSpan(symbol="HBD", chrom="chr11", start=5232708, end=5235762),
        GeneSpan(symbol="HBG1", chrom="chr11", start=5269502, end=5271089),
        GeneSpan(symbol="HBG2", chrom="chr11", start=5274420, end=5275999),
        # HBE1 is the most 5' (toward higher chr11 coordinates) member
        # of the cluster; tight body of ~1.6 kb.
        GeneSpan(symbol="HBE1", chrom="chr11", start=5289580, end=5291402),
        # HBZ on chr16 (alpha-globin cluster locus); tight ~1.4 kb body.
        GeneSpan(symbol="HBZ", chrom="chr16", start=172847, end=174262),
    ),
    "BRCA1_FAMILY": (
        GeneSpan(symbol="BRCA1", chrom="chr17", start=43044295, end=43125370),
        GeneSpan(symbol="BRCA2", chrom="chr13", start=32315474, end=32400266),
    ),
    "TP53_FAMILY": (
        GeneSpan(symbol="TP53", chrom="chr17", start=7668421, end=7687490),
        GeneSpan(symbol="TP63", chrom="chr3", start=189566712, end=189897276),
        GeneSpan(symbol="TP73", chrom="chr1", start=3652515, end=3736201),
    ),
    "EGFR_FAMILY": (
        GeneSpan(symbol="EGFR", chrom="chr7", start=55019017, end=55211628),
        GeneSpan(symbol="ERBB2", chrom="chr17", start=39688094, end=39730426),
        GeneSpan(symbol="ERBB3", chrom="chr12", start=56473627, end=56497291),
        GeneSpan(symbol="ERBB4", chrom="chr2", start=211375717, end=212543272),
    ),
    "MYC_FAMILY": (
        GeneSpan(symbol="MYC", chrom="chr8", start=127736231, end=127741434),
        GeneSpan(symbol="MYCN", chrom="chr2", start=15940550, end=15947007),
        GeneSpan(symbol="MYCL", chrom="chr1", start=39884329, end=39892996),
    ),
}


# Reverse index: gene_symbol (uppercase) -> family_key.
_SYMBOL_TO_FAMILY: dict[str, str] = {}
for _family_key, _spans in PARALOG_FAMILIES.items():
    for _span in _spans:
        _SYMBOL_TO_FAMILY[_span.symbol.upper()] = _family_key


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_paralog_spans(symbol: str) -> tuple[GeneSpan, ...]:
    """Return the paralog gene spans for ``symbol`` (excluding ``symbol``).

    Lookup is case-insensitive on ``symbol``. When the symbol is
    unknown to the v0.1 hardcoded map, returns an empty tuple — the
    pe_design pipeline then falls back to v0 behaviour (no in-paralog
    tagging).

    Parameters
    ----------
    symbol:
        HGNC gene symbol (e.g. ``"HBB"``).

    Returns
    -------
    tuple[GeneSpan, ...]
        Sibling paralog spans. Empty tuple when the symbol is unknown
        OR when the symbol is in a single-member family (no siblings
        to list).

    Examples
    --------
    >>> spans = get_paralog_spans("HBB")
    >>> sorted(s.symbol for s in spans)
    ['HBD', 'HBE1', 'HBG1', 'HBG2', 'HBZ']
    >>> get_paralog_spans("UNKNOWN_GENE")
    ()
    """
    norm = symbol.strip().upper()
    family_key = _SYMBOL_TO_FAMILY.get(norm)
    if family_key is None:
        return ()
    spans = PARALOG_FAMILIES[family_key]
    return tuple(s for s in spans if s.symbol.upper() != norm)


def is_in_any_paralog(
    *,
    chrom: str,
    pos_0b: int,
    paralog_spans: Iterable[GeneSpan],
) -> bool:
    """Return ``True`` when ``(chrom, pos_0b)`` falls inside any of the
    supplied paralog spans.

    The 0-based coordinate convention matches
    :class:`bionpu.genomics.pe_design.types.OffTargetSite.pos`. Spans
    are 1-based inclusive internally; the test handles the conversion.

    Hot path: per-off-target call from the T11 adapter. The
    implementation is a flat ``any()`` over the spans tuple. With the
    v0.1 hardcoded map at most 6 spans/family the cost is negligible;
    when the v1 HGNC integration lands the paralog list per family
    grows to ~50 spans worst-case which still completes in microseconds
    per call.

    Parameters
    ----------
    chrom:
        Reference contig of the off-target hit.
    pos_0b:
        0-based genomic position.
    paralog_spans:
        Iterable of :class:`GeneSpan` (typically the output of
        :func:`get_paralog_spans` cached at scan-start).

    Returns
    -------
    bool
        ``True`` when any span contains the coordinate.
    """
    return any(s.contains(chrom=chrom, pos_0b=pos_0b) for s in paralog_spans)
