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

"""Track B v0 — Off-target adapter for the pegRNA design pipeline (Task T11).

This module is a thin **adapter** over the existing
:mod:`bionpu.genomics.crispr_design` off-target scan path. It does NOT
reimplement scanning or CFD scoring; it composes the same primitives
that :func:`bionpu.genomics.crispr_design._scan_locus_for_offtargets`
+ :func:`bionpu.genomics.crispr_design._score_off_target_cfd` chain
internally, then re-shapes the output into the pe_design
:class:`bionpu.genomics.pe_design.types.OffTargetSite` type that T8's
ranker / T9's TSV formatter / T10's CLI consume.

Public surface
--------------
* :func:`off_target_scan_for_spacer` — per-spacer wrapper returning
  ``(list[OffTargetSite], cfd_aggregate, count)``. Both the pegRNA
  spacer and the PE3 nicking spacer are scored via independent calls
  from T8's ranker.

Lock-discipline (CLAUDE.md non-negotiable)
------------------------------------------
**T11 sits ABOVE the lock layer.** ``crispr_design``'s scan path is
already wrapped in :func:`bionpu.dispatch.npu_silicon_lock.npu_silicon_lock`
for the subprocess silicon path AND uses the in-process
``_dispatch_lock``-only discipline for the pyxrt path per ``CLAUDE.md``.
T11 is a *client* of those layers — it does NOT introduce its own lock
call. Subprocess / multi-process callers (e.g. the Track B smoke
harness) MUST hold ``npu_silicon_lock`` themselves around the entire
harness body; in-process callers (T10's CLI, T8's ranker) inherit the
``_dispatch_lock`` discipline transparently from the underlying
:mod:`bionpu.dispatch.npu` module.

Implementation note: the v0 adapter uses :func:`bionpu.scan.cpu_scan`
because the locked silicon ``crispr/match_multitile_memtile`` artifacts
target a different fixture shape than per-spacer pe_design scans
(per-locus enumeration vs single-spacer dispatch). The pe_design
:func:`off_target_scan_for_spacer` API is locked here so that the v1
silicon path can drop in behind the same surface without churning T8 /
T10. When the silicon path lands, the lock-layer assumption above
remains correct: the silicon dispatch will go through
:mod:`bionpu.dispatch.npu` which provides the in-process
``_dispatch_lock`` already.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from bionpu.data.canonical_sites import CasOFFinderRow
from bionpu.data.paralog_mapper import GeneSpan, is_in_any_paralog
from bionpu.genomics.pe_design.types import OffTargetSite
from bionpu.scan import GuideSpec, cpu_scan
from bionpu.scoring.cfd import CFDScorer, aggregate_cfd

__all__ = [
    "off_target_scan_for_spacer",
]


# ---------------------------------------------------------------------------
# FASTA streaming
# ---------------------------------------------------------------------------


def _iter_fasta_records(path: Path) -> Iterable[tuple[str, str]]:
    """Yield ``(chrom, sequence)`` tuples from a multi-record FASTA.

    Streams record-by-record so a chr-scale FASTA never fully
    materialises in memory at once. The sequence is upper-cased ACGT;
    non-ACGT bases (N, IUPAC) are preserved verbatim so
    :func:`bionpu.scan.cpu_scan`'s ``_BASE_TO_CODE`` sentinel handling
    works as documented.

    A "record" here is the canonical ``>name\\n<seq lines>`` block;
    the chrom name is the first whitespace-delimited token after ``>``.
    Empty FASTAs yield nothing rather than raising.
    """
    chrom: str | None = None
    pieces: list[str] = []
    with path.open("rt", encoding="ascii") as fh:
        for line in fh:
            if line.startswith(">"):
                if chrom is not None:
                    yield chrom, "".join(pieces).upper()
                chrom = line[1:].strip().split(None, 1)[0]
                pieces = []
                continue
            stripped = line.strip()
            if stripped and chrom is not None:
                pieces.append(stripped)
    if chrom is not None:
        yield chrom, "".join(pieces).upper()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def off_target_scan_for_spacer(
    spacer: str,
    genome_path: Path | str,
    *,
    max_mismatches: int = 4,
    paralog_spans: Sequence[GeneSpan] | None = None,
) -> tuple[list[OffTargetSite], float, int]:
    """Run an off-target scan for a single spacer; return adapter shape.

    Delegates to :func:`bionpu.scan.cpu_scan` (the same primitive
    :func:`bionpu.genomics.crispr_design._scan_locus_for_offtargets`
    uses internally for its CPU device path) and to
    :class:`bionpu.scoring.cfd.CFDScorer` /
    :func:`bionpu.scoring.cfd.aggregate_cfd` for per-site CFD scoring +
    CRISPOR-style aggregation.

    Parameters
    ----------
    spacer:
        20-nt ACGT spacer (PE2 pegRNA spacer or PE3 nicking-sgRNA
        spacer). Validated by :class:`bionpu.scan.GuideSpec` semantics
        via the underlying scan; non-20-mer or non-ACGT inputs raise
        ``ValueError``.
    genome_path:
        Path to a multi-record FASTA. For pe_design v0 this is the
        same chromosome FASTA T10's CLI passes (e.g. a chr22 slice or
        a full GRCh38 FASTA). Each record is scanned independently
        and results are aggregated.
    max_mismatches:
        Maximum allowed mismatches in the 20-nt spacer region.
        Default 4 (matches :data:`bionpu.genomics.crispr_design.DEFAULT_MAX_MISMATCHES`).
    paralog_spans:
        Optional list of :class:`bionpu.data.paralog_mapper.GeneSpan`
        for the on-target gene's paralogs. When supplied, each
        emitted :class:`OffTargetSite` carries ``in_paralog=True``
        when its ``(chrom, pos)`` falls inside any paralog span. The
        T8 ranker uses this flag to split the CFD aggregate into the
        ``cfd_aggregate_pegrna`` (non-paralog hits, the safety
        penalty) and ``cfd_aggregate_paralog_pegrna`` (in-paralog
        hits, informational) terms. v0 callers that don't pass this
        argument get every hit tagged ``in_paralog=False`` (v0
        codepath: every hit treated as a true off-target).

    Returns
    -------
    tuple[list[OffTargetSite], float, int]
        ``(sites, cfd_aggregate, count)``:

        * ``sites`` — every PAM-bearing 20-mer with at most
          ``max_mismatches`` against ``spacer`` re-shaped into
          :class:`bionpu.genomics.pe_design.types.OffTargetSite`.
          Includes the on-target row (``mismatches == 0``) so callers
          that want to check on-target presence have the data; the
          aggregate excludes ``mismatches == 0`` per CRISPOR
          convention.
        * ``cfd_aggregate`` — CRISPOR-style specificity in [0, 100];
          100 means "no off-targets at all". Computed via
          :func:`bionpu.scoring.cfd.aggregate_cfd` with
          ``exclude_on_target=True``.
        * ``count`` — total row count (always equal to ``len(sites)``).

    Raises
    ------
    ValueError
        Propagated from the underlying scan when ``spacer`` is not a
        20-nt ACGT string. ``FileNotFoundError`` when ``genome_path``
        does not exist.

    Notes
    -----
    Lock-discipline: this adapter does NOT acquire any silicon lock.
    See module docstring for the full lock-layer contract.
    """
    path = Path(genome_path)
    if not path.is_file():
        raise FileNotFoundError(f"genome FASTA not found: {path}")

    # Validate the spacer up-front via GuideSpec semantics (cpu_scan
    # accepts any length but mismatch counting only makes sense for
    # 20-mers; this matches the crispr_design + GuideSpec contract).
    spacer_norm = spacer.upper()
    if len(spacer_norm) != 20 or any(c not in "ACGT" for c in spacer_norm):
        raise ValueError(
            f"spacer must be 20 nt of ACGT; got {spacer!r} "
            f"(length={len(spacer_norm)})"
        )

    guide_spec = GuideSpec(spacer=spacer_norm, guide_id=spacer_norm)

    all_rows: list[CasOFFinderRow] = []
    for chrom, seq in _iter_fasta_records(path):
        if not seq or len(seq) < 23:
            continue
        rows = cpu_scan(
            chrom=chrom,
            seq=seq,
            guides=[guide_spec],
            pam_template="NGG",
            max_mismatches=max_mismatches,
        )
        all_rows.extend(rows)

    # Score each row through the locked CFDScorer; the scored rows
    # carry per-site cfd via ``ScoreRow.score``. We pair them 1:1 with
    # the canonical scan rows so the OffTargetSite re-shape preserves
    # both the position metadata (chrom/pos/strand/mismatches) and the
    # delegated CFD score.
    scorer = CFDScorer(matrix="doench_2016", apply_pam_penalty=False)
    scored_rows = list(scorer.score(all_rows))

    # Build the OffTargetSite list. Iterate in the canonical row order
    # (the order cpu_scan returned) so callers that hash the list get
    # deterministic content for a deterministic input.
    spans = list(paralog_spans) if paralog_spans else []
    sites: list[OffTargetSite] = []
    for raw, scored in zip(all_rows, scored_rows, strict=True):
        in_par = (
            is_in_any_paralog(
                chrom=raw.chrom, pos_0b=int(raw.start), paralog_spans=spans
            )
            if spans
            else False
        )
        sites.append(
            OffTargetSite(
                chrom=raw.chrom,
                pos=int(raw.start),
                strand=raw.strand,
                mismatches=int(raw.mismatches),
                cfd_score=float(scored.score),
                in_paralog=in_par,
            )
        )

    # Aggregate via the locked aggregate_cfd helper (CRISPOR
    # specificity convention: 100 / (100 + Σ_off CFD * 100), expressed
    # on a 0-100 scale).
    aggregate_map = aggregate_cfd(scored_rows, exclude_on_target=True)
    cfd_aggregate = float(aggregate_map.get(spacer_norm, 100.0))

    return sites, cfd_aggregate, len(sites)
