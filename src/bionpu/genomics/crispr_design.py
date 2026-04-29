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
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""End-to-end CRISPR guide design orchestrator (PRD-guide-design-on-xdna v0.2).

This module is the host-side glue that composes the silicon-validated
PAM-scan + match kernels with the landed CFD + Doench-RS1 scoring
modules into a single function call. The CLI surface
(``bionpu crispr design``) is a thin argparse wrapper around
:func:`design_guides_for_target`.

Tier 1 scope (this module's first cut)
--------------------------------------

This is the smallest defensible end-to-end first cut, deliberately
narrower than PRD §6 Phase 1's full deliverable:

* **Mode A/B/C target inputs** — gene symbol -> coordinates, raw
  ``chr:start-end`` coordinates, or local ``--target-fasta``. The
  `_RESOLVE_GENE_TO_LOCUS` table currently knows only ``BRCA1``; full
  RefSeq/Ensembl lookup is deferred to follow-up agents.
* **TSV + compact JSON output.** The full `off_targets_full` JSON
  variant is deferred.
* **Locus-scope off-target scan.** The off-target scan is run against
  the *target locus itself* (not the full GRCh38 reference). This
  bounds the smoke-test wall-clock to <30 s on CPU. PRD §6 Phase 1
  requires a full-genome off-target scan; that lives in a follow-up.
* **`composite_bionpu` v1 baseline** — linear re-weight of the
  Doench-RS1 on-target score and the CFD-aggregate specificity. The
  research-spike formula (PRD §7.1 Q5b) replaces this in a follow-up.
* **No CRISPOR comparison harness.** The 20-gene validation set
  (PRD §4.2 / §4.3) lands in a follow-up agent.
* **Default ``--rank-by crispor``** is the only ranking mode wired;
  ``--rank-by bionpu`` is a follow-up.

What is wired correctly today
-----------------------------

All five PRD §2.1 stages are connected end-to-end:

1. Stage 1: target resolution (`resolve_target` -- gene -> locus +
   chrom slice from the GRCh38 FASTA).
2. Stage 2: PAM scan via :func:`bionpu.scan.cpu_scan` /
   :func:`bionpu.scan.npu_scan`. NPU dispatch wraps the
   :func:`bionpu.dispatch.npu_silicon_lock.npu_silicon_lock` per
   CLAUDE.md non-negotiable.
3. Stage 3: off-target scan. Tier 1 reuses the Stage-2 scan against
   the locus itself; the full-genome scan is the follow-up.
4. Stage 4a: on-target Doench RS1 scoring via
   :class:`bionpu.scoring.doench_rs2.DoenchRS1Scorer`.
5. Stage 4b: per-site CFD + per-guide aggregate via
   :class:`bionpu.scoring.cfd.CFDScorer` and
   :func:`bionpu.scoring.cfd.aggregate_cfd`.
6. Stage 5: rank + emit.

Concretely: the CLI works end-to-end for ``bionpu crispr design
--target BRCA1 --genome GRCh38 --top 10``, the smoke test runs
in <30 s, and follow-on agents land the validation harness on top.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import dataclasses
import json
from pathlib import Path

from bionpu.data.canonical_sites import CasOFFinderRow

__all__ = [
    "DEFAULT_GC_MAX",
    "DEFAULT_GC_MIN",
    "DEFAULT_MAX_MISMATCHES",
    "DEFAULT_NPU_AUTO_MIN_COMPARISONS",
    "DEFAULT_TOP_N",
    "GeneNotFoundError",
    "RankedGuide",
    "ResolvedTarget",
    "TARGET_RESOLVER_TIER1_NOTE",
    "TSV_HEADER",
    "compute_composite_bionpu",
    "compute_composite_crispor",
    "design_guides_for_target",
    "format_guides_tsv",
    "format_result_json",
    "rank_guides",
    "resolve_coordinate_target",
    "resolve_scan_device",
    "resolve_target_fasta",
    "resolve_target",
    "select_locus_guides",
    "slice_chrom_from_fasta",
]


# ---------------------------------------------------------------------------
# Tier 1 gene -> coords resolution.
#
# Per PRD §3.1 the production resolver is RefSeq / Ensembl GTF backed
# (PR-C `score_from_fasta` adapter). For the first-cut we hardcode BRCA1
# and document the limitation. Future agents extend the table or wire
# `pyensembl` here (PRD §7.1 Q2 — RefSeq is the resolved default).
# Coordinates are GRCh38 NCBI RefSeq (1-based inclusive on the reference,
# stored here as 1-based for human readability; the slice helper
# converts to 0-based).
# ---------------------------------------------------------------------------

TARGET_RESOLVER_TIER1_NOTE: str = (
    "Tier 1+ (20-gene CRISPOR validation harness): hardcoded 20-gene "
    "table covering the genes listed in "
    "tests/fixtures/crispor_reference/genes_pinned.json. Other genes "
    "raise GeneNotFoundError. PRD §3.1 RefSeq/Ensembl GTF integration "
    "(pyensembl) is the long-term replacement; the hardcoded table "
    "was extended additively (BRCA1 pinning unchanged) to unblock the "
    "20-gene §4.3 hard-gate validation harness."
)

# 1-based inclusive coordinates on GRCh38. Source: NCBI RefSeq Annotation
# Release 110 / GRCh38.p14 for each gene's longest mRNA span, padded to
# whole-gene window. The 20-gene set + stratification rationale lives in
# `tests/fixtures/crispor_reference/genes_pinned.json`; entries below
# mirror that fixture verbatim.
#
# BRCA1 was the only Tier 1 entry; do NOT remove or alter its coords —
# the smoke test pins them. The remaining 19 entries are additive.
_RESOLVE_GENE_TO_LOCUS: dict[str, tuple[str, int, int]] = {
    # Tier A — canonical CRISPR validation loci (10 genes).
    "BRCA1": ("chr17", 43044295, 43125483),
    "BRCA2": ("chr13", 32315086, 32400266),
    "EMX1":  ("chr2",  72950915, 72952010),
    "FANCF": ("chr11", 22622352, 22647105),
    "RNF2":  ("chr1",  185179770, 185279091),
    "VEGFA": ("chr6",  43770209, 43795797),
    "HBB":   ("chr11", 5225464, 5229395),
    "TP53":  ("chr17", 7668421, 7687490),
    "ATM":   ("chr11", 108222484, 108369102),
    "MYC":   ("chr8",  127736231, 127741434),
    # Tier C — screen-essential / disease-relevant (8 genes).
    "CCR5":  ("chr3",  46370154, 46376206),
    "CD33":  ("chr19", 51225956, 51235170),
    "CXCR4": ("chr2",  136114349, 136118149),
    "KRAS":  ("chr12", 25205246, 25250929),
    "TET2":  ("chr4",  105146876, 105279816),
    "RUNX1": ("chr21", 34787800, 36054533),
    "DNMT1": ("chr19", 10133345, 10194977),
    "HPRT1": ("chrX",  134460164, 134500668),
    # Tier D — safe-harbor / utility loci (2 genes).
    "AAVS1": ("chr19", 55115750, 55117600),
    "CFTR":  ("chr7",  117480025, 117668665),
}


class GeneNotFoundError(KeyError):
    """Raised when a gene symbol is not in the Tier 1 resolver."""


@dataclass(frozen=True, slots=True)
class ResolvedTarget:
    """A resolved on-target locus."""

    gene: str
    chrom: str
    start: int  # 0-based inclusive
    end: int  # 0-based exclusive (Python-slice convention)
    sequence: str  # uppercase ACGT[N], len == end - start

    @property
    def length(self) -> int:
        return self.end - self.start


def resolve_target(
    *,
    target: str,
    genome: str,
    fasta_path: Path | str,
) -> ResolvedTarget:
    """Resolve a gene symbol or ``chr:start-end`` target to a chromosome slice.

    Parameters
    ----------
    target:
        The gene symbol, e.g. ``"BRCA1"`` (Mode A), or a coordinate
        target like ``"chr17:43044295-43125483"`` (Mode B). Coordinates
        are interpreted as 1-based inclusive, with commas allowed.
    genome:
        Reference build identifier. Tier 1: only ``"GRCh38"`` is
        accepted; other values raise ValueError.
    fasta_path:
        Path to the GRCh38 reference FASTA (a multi-record file with
        ``>chr17``, etc. records).

    Returns
    -------
    ResolvedTarget
        With the chromosome slice already loaded into memory (typically
        80-200 kbp for a single gene; bounded).
    """
    if genome != "GRCh38":
        raise ValueError(
            f"genome must be 'GRCh38' (Tier 1); got {genome!r}. "
            "Other builds land in a follow-up agent."
        )
    coord = _parse_coordinate_target(target)
    if coord is not None:
        chrom, one_start, one_end = coord
        return resolve_coordinate_target(
            chrom=chrom,
            one_based_start=one_start,
            one_based_end=one_end,
            fasta_path=fasta_path,
            label=target,
        )
    sym = target.upper()
    if sym in _RESOLVE_GENE_TO_LOCUS:
        chrom, one_start, one_end = _RESOLVE_GENE_TO_LOCUS[sym]
    else:
        # UCSC genome-fetch v1: delegate to the refGene-backed resolver.
        # The hardcoded dict above stays as a fast in-tree pin so the
        # 20-gene CRISPOR validation harness and the
        # `monkeypatch.setitem(cd._RESOLVE_GENE_TO_LOCUS, ...)` test
        # pattern keep working.
        from bionpu.data.genome_fetcher import (
            GeneSymbolNotFound,
            resolve_gene_symbol,
        )

        try:
            coord = resolve_gene_symbol(sym, genome="hg38")
        except (GeneSymbolNotFound, FileNotFoundError) as exc:
            known = ", ".join(sorted(_RESOLVE_GENE_TO_LOCUS))
            raise GeneNotFoundError(
                f"gene {target!r} not found via the Tier 1 hardcoded "
                f"table (known: {known}) nor the UCSC refGene "
                f"resolver: {exc}. {TARGET_RESOLVER_TIER1_NOTE}"
            ) from exc
        chrom, one_start, one_end = coord.chrom, coord.start, coord.end
    zero_start = one_start - 1
    zero_end = one_end  # inclusive 1-based -> exclusive 0-based: end stays
    seq = slice_chrom_from_fasta(
        fasta_path=Path(fasta_path),
        chrom=chrom,
        start=zero_start,
        end=zero_end,
    )
    return ResolvedTarget(
        gene=sym,
        chrom=chrom,
        start=zero_start,
        end=zero_end,
        sequence=seq,
    )


def resolve_coordinate_target(
    *,
    chrom: str,
    one_based_start: int,
    one_based_end: int,
    fasta_path: Path | str,
    label: str | None = None,
) -> ResolvedTarget:
    """Resolve Mode B coordinates to a chromosome slice.

    Coordinates are 1-based inclusive at the user boundary and converted
    to the internal 0-based half-open convention.
    """
    if one_based_start < 1 or one_based_end < one_based_start:
        raise ValueError(
            "coordinate target must satisfy 1 <= start <= end; got "
            f"{chrom}:{one_based_start}-{one_based_end}"
        )
    zero_start = one_based_start - 1
    zero_end = one_based_end
    seq = slice_chrom_from_fasta(
        fasta_path=Path(fasta_path),
        chrom=chrom,
        start=zero_start,
        end=zero_end,
    )
    return ResolvedTarget(
        gene=label or f"{chrom}:{one_based_start}-{one_based_end}",
        chrom=chrom,
        start=zero_start,
        end=zero_end,
        sequence=seq,
    )


def _parse_coordinate_target(target: str) -> tuple[str, int, int] | None:
    chrom, sep, region = target.partition(":")
    if not sep:
        return None
    start_s, dash, end_s = region.partition("-")
    if not dash or not chrom or not start_s or not end_s:
        raise ValueError(
            "coordinate target must be chr:start-end, e.g. "
            "chr17:43044295-43125483"
        )
    try:
        start = int(start_s.replace(",", ""))
        end = int(end_s.replace(",", ""))
    except ValueError as exc:
        raise ValueError(
            f"coordinate target has non-integer bounds: {target!r}"
        ) from exc
    return chrom, start, end


def resolve_target_fasta(
    *,
    target_fasta_path: Path | str,
    target_name: str | None = None,
) -> ResolvedTarget:
    """Resolve Mode C input: a single-record FASTA target sequence.

    ``--target-fasta`` is intended for synthetic biology and local
    construct design where there is no GRCh38 coordinate system. The
    returned target uses local 0-based coordinates and the FASTA record
    header as ``chrom``. If multiple records are present, the first
    record is used and the rest are ignored deliberately.
    """
    path = Path(target_fasta_path)
    if not path.is_file():
        raise ValueError(f"target FASTA not found: {path}")
    chrom: str | None = None
    pieces: list[str] = []
    with path.open("rt", encoding="ascii") as fh:
        for line in fh:
            if line.startswith(">"):
                if chrom is not None:
                    break
                chrom = line[1:].strip().split(None, 1)[0] or path.stem
                continue
            if chrom is None:
                continue
            stripped = line.strip()
            if stripped:
                pieces.append(stripped.upper())
    if chrom is None:
        raise ValueError(f"target FASTA has no header: {path}")
    seq = "".join(pieces)
    if not seq:
        raise ValueError(f"target FASTA has an empty first record: {path}")
    name = target_name or path.stem
    return ResolvedTarget(
        gene=name,
        chrom=chrom,
        start=0,
        end=len(seq),
        sequence=seq,
    )


def slice_chrom_from_fasta(
    *,
    fasta_path: Path,
    chrom: str,
    start: int,
    end: int,
) -> str:
    """Stream-read a multi-record FASTA and return ``[start, end)`` of ``chrom``.

    Memory-efficient: never materialises any chromosome other than
    ``chrom``. Reads line-by-line and stops as soon as the requested
    window is covered. The returned string is uppercase ACGT[N];
    non-ACGT bases are preserved (the scan stage's `_BASE_TO_CODE`
    table assigns them the sentinel value 4 which propagates through
    correctly).

    Parameters
    ----------
    fasta_path:
        Path to a multi-record FASTA. Header lines start with ``>``;
        the contig name is the first whitespace-delimited token.
    chrom:
        The contig name to slice from (e.g. ``"chr17"``).
    start, end:
        0-based half-open coordinates on the forward strand of ``chrom``.

    Returns
    -------
    str
        The uppercase substring. Length is ``end - start``.

    Raises
    ------
    ValueError
        If ``chrom`` is not in the FASTA, or the requested window
        falls off the contig.
    """
    if start < 0 or end <= start:
        raise ValueError(
            f"invalid window [{start}, {end}); start must be >= 0 and "
            "end > start"
        )
    if not fasta_path.is_file():
        raise ValueError(f"FASTA not found: {fasta_path}")

    target_header = chrom
    in_target = False
    chrom_pos = 0  # number of bases of `chrom` consumed so far
    pieces: list[str] = []
    pieces_len = 0  # cached len(''.join(pieces))
    needed = end - start

    with fasta_path.open("rt", encoding="ascii") as fh:
        for line in fh:
            if line.startswith(">"):
                # Hitting a new record header: if we were in the target
                # contig, we're done (either we satisfied the request
                # already or we ran out of contig).
                if in_target:
                    break
                # Otherwise check whether this is the requested header.
                tok = line[1:].strip().split(None, 1)[0]
                if tok == target_header:
                    in_target = True
                    chrom_pos = 0
                continue
            if not in_target:
                continue
            stripped = line.rstrip("\n").rstrip("\r")
            line_len = len(stripped)
            if line_len == 0:
                continue
            line_start = chrom_pos
            line_end = chrom_pos + line_len
            chrom_pos = line_end
            # Compute overlap of [line_start, line_end) with [start, end).
            ov_start = max(line_start, start)
            ov_end = min(line_end, end)
            if ov_start < ov_end:
                pieces.append(stripped[ov_start - line_start : ov_end - line_start])
                pieces_len += ov_end - ov_start
                if pieces_len >= needed:
                    break
            elif line_start >= end:
                break

    if not in_target:
        raise ValueError(f"chromosome {chrom!r} not found in {fasta_path}")
    if pieces_len < needed:
        raise ValueError(
            f"requested window [{start}, {end}) extends past the end of "
            f"contig {chrom!r} (only {pieces_len} of {needed} bases "
            "available)"
        )
    return "".join(pieces).upper()


# ---------------------------------------------------------------------------
# Stage 5 -- rank + emit.
# ---------------------------------------------------------------------------

DEFAULT_TOP_N: int = 10
DEFAULT_MAX_MISMATCHES: int = 4
DEFAULT_GC_MIN: float = 25.0
DEFAULT_GC_MAX: float = 75.0

# Conservative auto-routing threshold for CRISPR design's off-target scan.
# Tiny loci lose to XRT/host-runner overhead (AAVS1: CPU off-target scan
# 24 ms vs NPU 586 ms). Large candidate x locus products are where the
# AIE2P path amortizes dispatch overhead and attacks the dominant wall time
# (RUNX1 CPU off-target scan: 1,705 s).
DEFAULT_NPU_AUTO_MIN_COMPARISONS: int = 50_000_000_000


# Composite-scoring weights used by `composite_bionpu` (PRD §7.1 Q5
# resolution: v1 baseline = "linear re-weight of `on_target_score`
# (Doench RS1) + normalized `cfd_aggregate`, with weights documented
# inline"). The research-spike formula (Q5b) replaces this.
_COMPOSITE_BIONPU_W_ONTARGET: float = 0.5
_COMPOSITE_BIONPU_W_OFFTARGET: float = 0.5

# CRISPOR composite mimic — CRISPOR's default formula is (broadly)
# `priority_weight * doench + (1 - priority_weight) * specificity/100`.
# Their public docs use 0.5 for the priority weight on the standalone
# CLI; we mirror.
_COMPOSITE_CRISPOR_W_ONTARGET: float = 0.5


def compute_composite_crispor(
    *, on_target_score: float, cfd_aggregate: float
) -> float:
    """CRISPOR-mimic composite in [0, 1].

    on_target_score is in [0, 1] (Doench RS1); cfd_aggregate is in
    [0, 100] (CRISPOR specificity). The composite weights the two
    equally by default (the CRISPOR CLI's `priority_weight` knob has
    a default of 0.5).
    """
    spec01 = max(0.0, min(1.0, cfd_aggregate / 100.0))
    on01 = max(0.0, min(1.0, on_target_score))
    return _COMPOSITE_CRISPOR_W_ONTARGET * on01 + (
        1.0 - _COMPOSITE_CRISPOR_W_ONTARGET
    ) * spec01


def compute_composite_bionpu(
    *, on_target_score: float, cfd_aggregate: float
) -> float:
    """v1 baseline `composite_bionpu` -- linear re-weight in [0, 1].

    Per PRD §7.1 Q5: this is the "v1 baseline" placeholder; weights
    pinned at 0.5 / 0.5 until the research-spike formula lands. Same
    inputs and output range as `composite_crispor`; the only
    difference today is that the weights are exposed as named
    constants for the spike to mutate.
    """
    spec01 = max(0.0, min(1.0, cfd_aggregate / 100.0))
    on01 = max(0.0, min(1.0, on_target_score))
    return (
        _COMPOSITE_BIONPU_W_ONTARGET * on01
        + _COMPOSITE_BIONPU_W_OFFTARGET * spec01
    )


@dataclass(frozen=True, slots=True)
class RankedGuide:
    """One row of the final ranked TSV (PRD §3.2 schema, Tier 1 subset)."""

    rank: int
    guide_id: str
    guide_seq: str
    pam_seq: str
    strand: str
    target_chrom: str
    target_pos: int
    gc_pct: float
    on_target_score: float
    cfd_aggregate: float
    off_target_count: int
    top_off_targets: str
    composite_crispor: float
    composite_bionpu: float
    ranked_by: str
    predicted_indel: str  # blank for v1; reserved for future indel scorer
    notes: str  # ";"-joined flags ("LOW_GC", "HIGH_GC", "POLY_T_RUN", ...)


# PRD §3.2 column order. We match it exactly so downstream pipelines
# that scrape CRISPOR-style TSVs can ingest ours unmodified.
TSV_HEADER: tuple[str, ...] = (
    "rank",
    "guide_id",
    "guide_seq",
    "pam_seq",
    "strand",
    "target_chrom",
    "target_pos",
    "gc_pct",
    "on_target_score",
    "cfd_aggregate",
    "off_target_count",
    "top_off_targets",
    "composite_crispor",
    "composite_bionpu",
    "ranked_by",
    "predicted_indel",
    "notes",
)


def _gc_pct(spacer: str) -> float:
    if not spacer:
        return 0.0
    gc = spacer.count("G") + spacer.count("C")
    return 100.0 * gc / len(spacer)


def _format_top_off_targets(
    rows_for_guide: Sequence[CasOFFinderRow],
    cfd_per_row: Sequence[float],
    *,
    top_k: int = 5,
) -> str:
    """Format up to ``top_k`` off-target hits as ";"-joined records.

    PRD §3.2: ``"{chrom}:{pos}:{mm};{cfd}"`` for each entry.
    Entries are sorted by descending per-site CFD (the riskiest first)
    so the user reading the TSV sees the worst offenders at a glance.
    """
    if len(rows_for_guide) != len(cfd_per_row):
        raise ValueError(
            "rows_for_guide and cfd_per_row length mismatch: "
            f"{len(rows_for_guide)} vs {len(cfd_per_row)}"
        )
    paired = list(zip(rows_for_guide, cfd_per_row, strict=True))
    paired.sort(key=lambda x: -x[1])
    out: list[str] = []
    for r, cfd in paired[:top_k]:
        out.append(f"{r.chrom}:{r.start}:{r.mismatches};{cfd:.4f}")
    return ";".join(out)


def _classify_notes(spacer: str, gc_pct: float, *, gc_min: float, gc_max: float) -> str:
    """Emit ";"-joined PRD §3.2 advisory flags."""
    flags: list[str] = []
    if gc_pct < gc_min:
        flags.append("LOW_GC")
    if gc_pct > gc_max:
        flags.append("HIGH_GC")
    # Poly-T run >=4 is an RNA-Pol-III termination signal; the bench
    # biologist persona cares about this.
    longest_run = 0
    cur_run = 0
    cur_base = ""
    for b in spacer:
        if b == cur_base:
            cur_run += 1
        else:
            cur_run = 1
            cur_base = b
        if cur_base == "T":
            longest_run = max(longest_run, cur_run)
    if longest_run >= 4:
        flags.append("POLY_T_RUN")
    return ";".join(flags)


def select_locus_guides(
    rows: Iterable[CasOFFinderRow],
    *,
    locus_chrom: str,
    locus_start: int,
    locus_end: int,
) -> list[CasOFFinderRow]:
    """Pick the on-target rows: rows whose hit lies inside the target locus
    *and* has zero mismatches against the input guide.

    Tier 1: we treat any mismatches==0 hit inside the locus as an
    on-target candidate. Off-target candidates are everything else
    (different position OR mismatches>=1).
    """
    out: list[CasOFFinderRow] = []
    for r in rows:
        if r.mismatches != 0:
            continue
        if r.chrom != locus_chrom:
            continue
        if r.start < locus_start or r.start >= locus_end:
            continue
        out.append(r)
    return out


def rank_guides(
    *,
    on_target_rows: Sequence[CasOFFinderRow],
    on_target_scores: dict[str, float],
    cfd_aggregate_per_guide: dict[str, float],
    off_target_rows_by_guide: dict[str, list[CasOFFinderRow]],
    cfd_per_off_target_by_guide: dict[str, list[float]],
    target_chrom: str,
    rank_by: str,
    top_n: int,
    gc_min: float,
    gc_max: float,
    synbio_mode: bool = False,
) -> list[RankedGuide]:
    """Build ranked output rows for emission.

    Parameters
    ----------
    on_target_rows:
        One row per *unique* on-target hit -- the input guide bound at
        its intended site (``mismatches == 0`` inside the target
        locus). Multiple rows for the same guide_id (e.g. paralog hits
        inside the locus) are collapsed by guide_id at scoring time
        and the first row is kept for position reporting.
    on_target_scores:
        Mapping ``guide_id -> Doench RS1 score [0, 1]``.
    cfd_aggregate_per_guide:
        Mapping ``guide_id -> CFD aggregate (CRISPOR specificity) [0, 100]``.
    off_target_rows_by_guide, cfd_per_off_target_by_guide:
        Per-guide off-target hit lists and per-row CFD for the
        ``top_off_targets`` formatter.
    target_chrom:
        Recorded verbatim in the output rows.
    rank_by:
        ``"crispor"`` (Tier 1 default) or ``"bionpu"``. Drives sort.
    top_n:
        Keep this many rows after sort. PRD §3.1 default is 20 but the
        Tier 1 brief defaults to 10.
    gc_min, gc_max:
        Acceptable GC% band for the ``LOW_GC`` / ``HIGH_GC`` notes.
        Guides outside the band still appear in the output, with the
        flag set, so the user can see what was rejected.
    """
    if rank_by not in ("crispor", "bionpu"):
        raise ValueError(
            f"rank_by must be 'crispor' or 'bionpu'; got {rank_by!r}"
        )

    # Collapse multiple on-target rows per guide_id (keep first).
    by_guide: dict[str, CasOFFinderRow] = {}
    for r in on_target_rows:
        by_guide.setdefault(r.guide_id, r)

    candidates: list[RankedGuide] = []
    for gid, on_row in by_guide.items():
        spacer = on_row.crrna[:20].upper()
        pam = on_row.dna[20:23].upper() if len(on_row.dna) >= 23 else ""
        gc = _gc_pct(spacer)
        rs1 = float(on_target_scores.get(gid, 0.0))
        if synbio_mode:
            # PRD §7.1 Q3: Mode C / synbio off-target scan is skipped;
            # emit NaN sentinels and the NO_OFF_TARGET_SCAN flag so
            # downstream consumers can't silently mistake them for
            # "scanned, zero hits".
            cfd_agg = float("nan")
            off_target_count = -1  # sentinel; rendered as "NaN" in TSV
            top_off_str = ""
            # composite formulas treat NaN CFD as "neutral" (skip the
            # off-target term) so the on-target score still drives ranking.
            composite_crispor = max(0.0, min(1.0, rs1))
            composite_bionpu = max(0.0, min(1.0, rs1))
            note_flags = ["NO_OFF_TARGET_SCAN"]
            extra = _classify_notes(spacer, gc, gc_min=gc_min, gc_max=gc_max)
            if extra:
                note_flags.append(extra)
            notes = ";".join(note_flags)
        else:
            cfd_agg = float(cfd_aggregate_per_guide.get(gid, 100.0))
            off_rows = off_target_rows_by_guide.get(gid, [])
            off_cfd = cfd_per_off_target_by_guide.get(gid, [])
            off_target_count = len(off_rows)
            top_off_str = _format_top_off_targets(off_rows, off_cfd)
            composite_crispor = compute_composite_crispor(
                on_target_score=rs1, cfd_aggregate=cfd_agg
            )
            composite_bionpu = compute_composite_bionpu(
                on_target_score=rs1, cfd_aggregate=cfd_agg
            )
            notes = _classify_notes(spacer, gc, gc_min=gc_min, gc_max=gc_max)
        candidates.append(
            RankedGuide(
                rank=0,  # filled in after sort
                guide_id=gid,
                guide_seq=spacer,
                pam_seq=pam,
                strand=on_row.strand,
                target_chrom=target_chrom,
                target_pos=on_row.start,
                gc_pct=gc,
                on_target_score=rs1,
                cfd_aggregate=cfd_agg,
                off_target_count=off_target_count,
                top_off_targets=top_off_str,
                composite_crispor=composite_crispor,
                composite_bionpu=composite_bionpu,
                ranked_by=rank_by,
                predicted_indel="",
                notes=notes,
            )
        )

    # Sort: descending by the chosen composite, then by guide_id for
    # determinism on ties.
    sort_key = (
        (lambda g: (-g.composite_crispor, g.guide_id))
        if rank_by == "crispor"
        else (lambda g: (-g.composite_bionpu, g.guide_id))
    )
    candidates.sort(key=sort_key)
    top = candidates[:top_n]
    # Re-rank in place. ``RankedGuide`` is slots=True so ``__dict__``
    # isn't available; ``dataclasses.replace`` is the right tool.
    return [
        dataclasses.replace(g, rank=i + 1)
        for i, g in enumerate(top)
    ]


def _fmt_float(x: float, fmt: str = ".6f") -> str:
    """Format a float, rendering NaN as the string ``"NaN"``.

    Python's f-string default for NaN is ``"nan"`` lowercased; PRD §3.2
    + the v1 brief specify ``"NaN"`` so a downstream consumer can
    detect-and-skip with a literal-string compare.
    """
    import math

    if math.isnan(x):
        return "NaN"
    return format(x, fmt)


def _fmt_off_target_count(n: int) -> str:
    """Format off_target_count: ``-1`` sentinel renders as ``"NaN"`` (synbio)."""
    return "NaN" if n < 0 else str(n)


def format_guides_tsv(rows: Iterable[RankedGuide]) -> bytes:
    """Serialise to the canonical TSV (LF newlines, header included).

    Score formatting policy: floats use ``%.6f`` to match
    :mod:`bionpu.scoring.types` (six decimals). Ints are unformatted.
    Synbio mode (PRD §7.1 Q3): ``cfd_aggregate`` is NaN and
    ``off_target_count`` is -1 sentinel; both render as ``"NaN"``.
    """
    parts: list[str] = ["\t".join(TSV_HEADER)]
    for g in rows:
        parts.append(
            "\t".join(
                (
                    str(g.rank),
                    g.guide_id,
                    g.guide_seq,
                    g.pam_seq,
                    g.strand,
                    g.target_chrom,
                    str(g.target_pos),
                    f"{g.gc_pct:.2f}",
                    _fmt_float(g.on_target_score),
                    _fmt_float(g.cfd_aggregate),
                    _fmt_off_target_count(g.off_target_count),
                    g.top_off_targets,
                    _fmt_float(g.composite_crispor),
                    _fmt_float(g.composite_bionpu),
                    g.ranked_by,
                    g.predicted_indel,
                    g.notes,
                )
            )
        )
    blob = "\n".join(parts) + "\n"
    return blob.encode("utf-8")


def format_result_json(result: "DesignRunResult") -> bytes:
    """Serialise a design run to stable, machine-readable JSON.

    NaN values (synbio mode CFD + off_target_count sentinels) render as
    JSON ``null`` so downstream consumers using strict JSON parsers
    (PRD §3.2 — synthesis-vendor pipelines) still succeed.
    """
    import math

    def _scrub(value):
        if isinstance(value, float) and math.isnan(value):
            return None
        if isinstance(value, dict):
            return {k: _scrub(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_scrub(v) for v in value]
        return value

    ranked_dicts = []
    for g in result.ranked:
        d = dataclasses.asdict(g)
        # PRD §3.2: off_target_count -1 sentinel → null in JSON
        if d.get("off_target_count", 0) < 0:
            d["off_target_count"] = None
        ranked_dicts.append(_scrub(d))

    payload = {
        "target": dataclasses.asdict(result.target),
        "n_candidates_total": result.n_candidates_total,
        "n_off_target_hits": result.n_off_target_hits,
        "stage_timings_s": result.stage_timings_s,
        "ranked": ranked_dicts,
    }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


# ---------------------------------------------------------------------------
# End-to-end orchestrator.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DesignRunResult:
    """Bundle returned by :func:`design_guides_for_target`."""

    target: ResolvedTarget
    ranked: list[RankedGuide]
    tsv_bytes: bytes
    n_candidates_total: int
    n_off_target_hits: int
    stage_timings_s: dict[str, float]


def design_guides_for_target(
    *,
    target: str,
    genome: str,
    fasta_path: Path | str,
    top_n: int = DEFAULT_TOP_N,
    max_mismatches: int = DEFAULT_MAX_MISMATCHES,
    gc_min: float = DEFAULT_GC_MIN,
    gc_max: float = DEFAULT_GC_MAX,
    device: str = "cpu",
    rank_by: str = "crispor",
    silicon_lock_label: str | None = None,
    target_fasta_path: Path | str | None = None,
) -> DesignRunResult:
    """Run the full Tier 1 pipeline and return ranked guides + TSV bytes.

    Parameters
    ----------
    target, genome:
        Inputs to :func:`resolve_target` unless ``target_fasta_path`` is
        supplied.
    fasta_path:
        GRCh38 reference FASTA. Used both for the locus slice and for
        the Doench-RS1 30-mer context lookup.
    top_n, max_mismatches, gc_min, gc_max:
        See PRD §3.1 defaults; Tier 1 overrides ``top_n`` to 10.
    device:
        ``"cpu"`` (default; numpy-only path), ``"npu"`` (silicon path),
        or ``"auto"`` (route large scans to NPU, tiny scans to CPU).
        The NPU path requires the precompiled CRISPR xclbins and wraps
        every dispatch in
        :func:`bionpu.dispatch.npu_silicon_lock.npu_silicon_lock`.
    rank_by:
        ``"crispor"`` (Tier 1 default) or ``"bionpu"``.
    silicon_lock_label:
        Optional diagnostic label for the silicon lock PID sidecar.
    target_fasta_path:
        Mode C target FASTA. When supplied, ``target`` is used only as
        the output label and ``genome`` may be ``"none"``.

    Returns
    -------
    DesignRunResult
    """
    import time

    timings: dict[str, float] = {}

    # Mode C / synbio: when --genome none is set, off-target scan is
    # skipped entirely (PRD §7.1 Q3 + §1.1). Output guides emit NaN
    # sentinels + the NO_OFF_TARGET_SCAN flag.
    synbio_mode = (genome == "none")

    # Stage 1 — target resolution.
    t0 = time.perf_counter()
    if target_fasta_path is not None:
        resolved = resolve_target_fasta(
            target_fasta_path=target_fasta_path,
            target_name=target,
        )
    else:
        resolved = resolve_target(
            target=target, genome=genome, fasta_path=fasta_path
        )
    timings["target_resolve"] = time.perf_counter() - t0

    # Stage 2 — PAM scan over the locus to enumerate candidate guides.
    # We use the existing :func:`bionpu.genomics.guide_design.enumerate_guides`
    # to discover all NGG-PAM-bearing 20-mers in the locus (forward +
    # reverse strand). This gives us the set of guide spacers to feed
    # into the off-target scan.
    t0 = time.perf_counter()
    candidate_guides = _enumerate_locus_guides(resolved)
    timings["pam_scan"] = time.perf_counter() - t0

    # Stage 3 — off-target scan.
    # Tier 1: scan against the locus itself (not the full genome). This
    # bounds smoke-test wall-clock and surfaces the wiring without the
    # follow-up agent's full-GRCh38 dispatch. Off-targets here include
    # any non-on-target hit (different position or different sequence
    # at <=N mismatches).
    # Synbio mode: skip Stage 3 entirely; we still need a list of
    # on-target rows (one per candidate guide) so Stage 5 can emit
    # them — synthesise via cpu_scan but only resolve the on-target
    # site (no mismatch search needed).
    t0 = time.perf_counter()
    scan_device = resolve_scan_device(
        requested=device,
        locus_bp=resolved.length,
        n_candidate_guides=len(candidate_guides),
        synbio_mode=synbio_mode,
    )
    if synbio_mode:
        all_hits = _scan_locus_for_offtargets(
            resolved=resolved,
            candidate_guides=candidate_guides,
            max_mismatches=0,  # only on-target hits
            device="cpu",  # synbio is small, always CPU
            silicon_lock_label=silicon_lock_label,
        )
    else:
        all_hits = _scan_locus_for_offtargets(
            resolved=resolved,
            candidate_guides=candidate_guides,
            max_mismatches=max_mismatches,
            device=scan_device,
            silicon_lock_label=silicon_lock_label,
        )
    timings["off_target_scan"] = time.perf_counter() - t0

    # Identify on-targets vs off-targets.
    locus_start_abs = resolved.start
    locus_end_abs = resolved.end
    on_target_rows = select_locus_guides(
        all_hits,
        locus_chrom=resolved.chrom,
        locus_start=locus_start_abs,
        locus_end=locus_end_abs,
    )

    # Build per-guide off-target row lists.
    off_target_rows_by_guide: dict[str, list[CasOFFinderRow]] = {}
    on_target_keys: set[tuple[str, str, int, str]] = {
        (r.guide_id, r.chrom, r.start, r.strand) for r in on_target_rows
    }
    for r in all_hits:
        key = (r.guide_id, r.chrom, r.start, r.strand)
        if key in on_target_keys and r.mismatches == 0:
            continue  # on-target site, not an off-target
        off_target_rows_by_guide.setdefault(r.guide_id, []).append(r)

    # Stage 4a — on-target scoring (Doench RS1).
    t0 = time.perf_counter()
    scoring_chrom, scoring_offset, scoring_seq = _load_scoring_context(
        resolved=resolved,
        fasta_path=Path(fasta_path),
        target_fasta_path=Path(target_fasta_path) if target_fasta_path else None,
    )
    on_target_scores = _score_on_target(
        on_target_rows=on_target_rows,
        chrom=scoring_chrom,
        chrom_seq_offset=scoring_offset,
        chrom_seq=scoring_seq,
    )
    timings["on_target_score"] = time.perf_counter() - t0

    # Stage 4b — off-target CFD aggregation.
    t0 = time.perf_counter()
    if synbio_mode:
        cfd_per_guide_aggregate, cfd_per_off_target_by_guide = {}, {}
    else:
        cfd_per_guide_aggregate, cfd_per_off_target_by_guide = (
            _score_off_target_cfd(off_target_rows_by_guide)
        )
    timings["off_target_score"] = time.perf_counter() - t0

    # Stage 5 — rank + emit.
    t0 = time.perf_counter()
    ranked = rank_guides(
        on_target_rows=on_target_rows,
        on_target_scores=on_target_scores,
        cfd_aggregate_per_guide=cfd_per_guide_aggregate,
        off_target_rows_by_guide=off_target_rows_by_guide,
        cfd_per_off_target_by_guide=cfd_per_off_target_by_guide,
        target_chrom=resolved.chrom,
        rank_by=rank_by,
        top_n=top_n,
        gc_min=gc_min,
        gc_max=gc_max,
        synbio_mode=synbio_mode,
    )
    tsv_bytes = format_guides_tsv(ranked)
    timings["rank_emit"] = time.perf_counter() - t0

    return DesignRunResult(
        target=resolved,
        ranked=ranked,
        tsv_bytes=tsv_bytes,
        n_candidates_total=len(candidate_guides),
        n_off_target_hits=sum(len(v) for v in off_target_rows_by_guide.values()),
        stage_timings_s=timings,
    )


def resolve_scan_device(
    *,
    requested: str,
    locus_bp: int,
    n_candidate_guides: int,
    synbio_mode: bool = False,
    min_comparisons_for_npu: int = DEFAULT_NPU_AUTO_MIN_COMPARISONS,
) -> str:
    """Resolve ``cpu`` / ``npu`` / ``auto`` to the actual scan device.

    ``auto`` is deliberately conservative: it only chooses NPU when the
    guide-count x locus-length product is large enough to amortize XRT
    and host-runner overhead. Explicit ``npu`` still forces silicon for
    validation and benchmarking.
    """
    if requested not in ("cpu", "npu", "auto"):
        raise ValueError(
            f"device must be 'cpu', 'npu', or 'auto'; got {requested!r}"
        )
    if requested in ("cpu", "npu"):
        return requested
    if synbio_mode:
        return "cpu"
    comparisons = int(locus_bp) * int(n_candidate_guides)
    return "npu" if comparisons >= min_comparisons_for_npu else "cpu"


# ---------------------------------------------------------------------------
# Helpers (private)
# ---------------------------------------------------------------------------


def _enumerate_locus_guides(resolved: ResolvedTarget) -> list[tuple[str, str, int, str]]:
    """Return list of ``(guide_id, spacer, abs_start, strand)`` for every
    NGG-PAM-bearing 20-mer in the locus.

    Forward strand: ``[s, s+20)`` is the spacer; ``[s+20, s+23)`` is
    the PAM (NGG -> seq[s+21]==G & seq[s+22]==G).
    Reverse strand: PAM is CCN at ``[s, s+3)``; spacer is RC of
    ``[s+3, s+23)``.
    """
    seq = resolved.sequence
    n = len(seq)
    out: list[tuple[str, str, int, str]] = []
    for s in range(0, n - 23 + 1):
        # Forward
        if seq[s + 21] == "G" and seq[s + 22] == "G":
            spacer = seq[s : s + 20]
            if all(c in "ACGT" for c in spacer):
                abs_pos = resolved.start + s
                gid = f"{resolved.gene}_+_{abs_pos}_{spacer}"
                out.append((gid, spacer, abs_pos, "+"))
        # Reverse: forward seq has CC at [s, s+2), so genomic forward is
        # CCN..., and the spacer in guide-strand is RC(seq[s+3:s+23]).
        if seq[s] == "C" and seq[s + 1] == "C":
            window = seq[s + 3 : s + 23]
            if all(c in "ACGT" for c in window):
                spacer = _rc(window)
                abs_pos = resolved.start + s
                gid = f"{resolved.gene}_-_{abs_pos}_{spacer}"
                out.append((gid, spacer, abs_pos, "-"))
    return out


_RC_TABLE = str.maketrans("ACGTacgt", "TGCAtgca")


def _rc(seq: str) -> str:
    return seq.translate(_RC_TABLE)[::-1]


def _scan_locus_for_offtargets(
    *,
    resolved: ResolvedTarget,
    candidate_guides: Sequence[tuple[str, str, int, str]],
    max_mismatches: int,
    device: str,
    silicon_lock_label: str | None,
) -> list[CasOFFinderRow]:
    """Run the canonical off-target scan against the locus sequence.

    Tier 1: scope is the locus, not the full genome. Output rows have
    ``chrom`` set to the locus chromosome and ``start`` to the
    *absolute* genomic 0-based position (the scan internally uses
    locus-relative coords; we shift up by ``resolved.start``).
    """
    from bionpu.scan import GuideSpec, cpu_scan, npu_scan

    guide_specs = [
        GuideSpec(spacer=spacer, guide_id=gid)
        for gid, spacer, _abs_pos, _strand in candidate_guides
    ]
    if not guide_specs:
        return []

    if device == "npu":
        # Wrap *every* silicon submission in npu_silicon_lock per
        # CLAUDE.md non-negotiable. The lock is process-wide; re-entry
        # within this context is fine but cross-process / harness-style
        # callers must not see overlapping submissions.
        from bionpu.dispatch.npu_silicon_lock import npu_silicon_lock

        label = silicon_lock_label or f"bionpu_crispr_design:{resolved.gene}"
        with npu_silicon_lock(label=label):
            rows = npu_scan(
                chrom=resolved.chrom,
                seq=resolved.sequence,
                guides=guide_specs,
                pam_template="NGG",
                max_mismatches=max_mismatches,
            )
    elif device == "cpu":
        rows = cpu_scan(
            chrom=resolved.chrom,
            seq=resolved.sequence,
            guides=guide_specs,
            pam_template="NGG",
            max_mismatches=max_mismatches,
        )
    else:
        raise ValueError(f"device must be 'cpu' or 'npu'; got {device!r}")

    # The scan returns locus-relative `start` (0 = start of the slice).
    # Shift to absolute genomic coordinates for downstream reporting.
    shifted: list[CasOFFinderRow] = []
    abs_offset = resolved.start
    for r in rows:
        shifted.append(
            CasOFFinderRow(
                guide_id=r.guide_id,
                bulge_type=r.bulge_type,
                crrna=r.crrna,
                dna=r.dna,
                chrom=r.chrom,
                start=r.start + abs_offset,
                strand=r.strand,
                mismatches=r.mismatches,
                bulge_size=r.bulge_size,
            )
        )
    return shifted


def _load_scoring_context(
    *,
    resolved: ResolvedTarget,
    fasta_path: Path,
    target_fasta_path: Path | None,
) -> tuple[str, int, str]:
    """Return ``(chrom, offset, seq)`` for Doench 30-mer context lookup.

    Guide enumeration and off-target scanning stay constrained to the
    requested locus. On-target scoring, however, needs flanking context
    outside that locus. For Mode A/GRCh38 inputs we load a small widened
    slice from the reference FASTA so edge guides do not silently get
    score 0.0 just because the target slice stopped at the gene
    boundary.
    """
    flank = 4
    if target_fasta_path is not None:
        return resolved.chrom, resolved.start, resolved.sequence
    ctx_start = max(0, resolved.start - flank)
    ctx_end = resolved.end + flank
    for end in (ctx_end, resolved.end):
        try:
            seq = slice_chrom_from_fasta(
                fasta_path=fasta_path,
                chrom=resolved.chrom,
                start=ctx_start,
                end=end,
            )
            return resolved.chrom, ctx_start, seq
        except ValueError:
            continue
    # Keep the design run usable on truncated developer fixtures.
    # Rows that still lack true context remain unscored below.
    return resolved.chrom, resolved.start, resolved.sequence


def _score_on_target(
    *,
    on_target_rows: Sequence[CasOFFinderRow],
    chrom: str,
    chrom_seq_offset: int,
    chrom_seq: str,
) -> dict[str, float]:
    """Score on-target rows with Doench RS1.

    The scorer requires a chromosome-name -> forward-sequence dict;
    we feed it the locus slice plus a wrapper that translates absolute
    positions into locus-relative ones via a shim CasOFFinderRow.
    """
    from bionpu.scoring.doench_rs2 import DoenchRS1Scorer

    if not on_target_rows:
        return {}

    # Build a context-slice chrom_lookup. The scorer indexes by
    # row.chrom and slices ``chrom_seq[start:start+30mer_window]``.
    # We pass the sequence under the locus chrom name and rewrite
    # each row's start to be locus-relative for the scoring call only.
    locus_lookup = {chrom: chrom_seq}
    scorer = DoenchRS1Scorer(chrom_lookup=locus_lookup)

    # Shift abs->relative for the scorer. We don't mutate the input
    # rows; the relative-coord copies are scratch.
    scratch_rows: list[CasOFFinderRow] = []
    for r in on_target_rows:
        scratch_rows.append(
            CasOFFinderRow(
                guide_id=r.guide_id,
                bulge_type=r.bulge_type,
                crrna=r.crrna,
                dna=r.dna,
                chrom=r.chrom,
                start=r.start - chrom_seq_offset,
                strand=r.strand,
                mismatches=r.mismatches,
                bulge_size=r.bulge_size,
            )
        )

    # Filter rows whose 30-mer context window still falls off the
    # scoring slice. Mode A callers pass a widened slice, so gene-edge
    # guides score normally. This guard remains for target FASTA mode
    # and genuinely contig-edge cases.
    locus_len = len(chrom_seq)
    scratch_rows = [
        r for r in scratch_rows
        if r.start - 4 >= 0 and r.start + 27 <= locus_len
    ]
    scores: dict[str, float] = {}
    for srow in scorer.score(scratch_rows):
        scores[srow.guide_id] = float(srow.score)
    return scores


def _score_off_target_cfd(
    off_target_rows_by_guide: dict[str, list[CasOFFinderRow]],
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Score off-target hits with CFD; return (per-guide aggregate, per-row CFD).

    Returns
    -------
    aggregate:
        ``guide_id -> CFD aggregate (CRISPOR specificity) [0, 100]``.
        Guides with no off-target hits get ``100.0`` (perfect).
    per_row:
        ``guide_id -> list[float]`` aligned 1:1 with
        ``off_target_rows_by_guide[guide_id]``.
    """
    from bionpu.scoring.cfd import CFDScorer, aggregate_cfd

    if not off_target_rows_by_guide:
        return {}, {}

    scorer = CFDScorer(matrix="doench_2016", apply_pam_penalty=False)
    per_row: dict[str, list[float]] = {}
    flat_score_rows = []
    for gid, rows in off_target_rows_by_guide.items():
        per_row[gid] = []
        for srow in scorer.score(rows):
            per_row[gid].append(float(srow.score))
            flat_score_rows.append(srow)

    aggregate = aggregate_cfd(flat_score_rows)
    # Guides whose off-target list ended up empty after filtering still
    # exist in `off_target_rows_by_guide` keys; back-fill perfect specificity.
    for gid in off_target_rows_by_guide:
        aggregate.setdefault(gid, 100.0)
    return aggregate, per_row
