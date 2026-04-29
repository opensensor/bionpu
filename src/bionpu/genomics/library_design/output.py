# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

"""Library-output assembly + TSV writer (Track C v0).

A :class:`LibraryGuide` is a single library row: per-gene guides AND
controls share this shape. The library writer emits TSV columns per
the build brief schema:

    library_index, target_symbol, guide_index_within_gene, guide_seq,
    pam_seq, chrom, start, end, strand, doench_rs1, doench_rs2,
    cfd_aggregate, off_target_count, control_class, notes

For control rows, ``guide_index_within_gene`` is a class-counter
(``non_targeting=1..N``, ``safe_harbor=1..3``, ``essential_gene=1..2``).

Note on doench_rs1 vs doench_rs2: the v1 ``crispr_design`` pipeline
emits a single ``on_target_score`` (named "Doench RS1" historically
but the underlying scorer at
:class:`bionpu.scoring.doench_rs2.DoenchRS1Scorer` is the RS1 model).
Track C v0's column shape exposes both names for downstream consumer
compatibility; today they hold the same value (we write the rs1 score
into both columns). The library extension that lands a true RS2
scorer will diverge them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from bionpu.genomics.crispr_design import RankedGuide

from .controls import ControlGuide
from .library_balancer import BalanceReport

__all__ = [
    "LIBRARY_TSV_HEADER",
    "LibraryGuide",
    "assemble_library",
    "format_library_tsv",
]


@dataclass(frozen=True, slots=True)
class LibraryGuide:
    """One row of the pooled-library TSV (per-gene guides + controls)."""

    library_index: int
    target_symbol: str
    guide_index_within_gene: int
    guide_seq: str
    pam_seq: str
    chrom: str
    start: int
    end: int
    strand: str
    doench_rs1: float
    doench_rs2: float
    cfd_aggregate: float
    off_target_count: int
    control_class: str  # "" for per-gene rows; one of the control-class strings for controls
    notes: str


LIBRARY_TSV_HEADER: tuple[str, ...] = (
    "library_index",
    "target_symbol",
    "guide_index_within_gene",
    "guide_seq",
    "pam_seq",
    "chrom",
    "start",
    "end",
    "strand",
    "doench_rs1",
    "doench_rs2",
    "cfd_aggregate",
    "off_target_count",
    "control_class",
    "notes",
)


def _ranked_to_library_row(
    ranked: RankedGuide,
    *,
    library_index: int,
    target_symbol: str,
    guide_index_within_gene: int,
    extra_notes: str = "",
) -> LibraryGuide:
    note_parts: list[str] = []
    if ranked.notes:
        note_parts.append(ranked.notes)
    if extra_notes:
        note_parts.append(extra_notes)
    notes = ";".join(p for p in note_parts if p)
    pam = ranked.pam_seq
    end = ranked.target_pos + len(ranked.guide_seq)
    return LibraryGuide(
        library_index=library_index,
        target_symbol=target_symbol,
        guide_index_within_gene=guide_index_within_gene,
        guide_seq=ranked.guide_seq,
        pam_seq=pam,
        chrom=ranked.target_chrom,
        start=ranked.target_pos,
        end=end,
        strand=ranked.strand,
        doench_rs1=float(ranked.on_target_score),
        doench_rs2=float(ranked.on_target_score),  # see module docstring
        cfd_aggregate=float(ranked.cfd_aggregate),
        off_target_count=int(ranked.off_target_count),
        control_class="",  # per-gene row
        notes=notes,
    )


def _control_to_library_row(
    ctrl: ControlGuide,
    *,
    library_index: int,
    guide_index_within_class: int,
) -> LibraryGuide:
    return LibraryGuide(
        library_index=library_index,
        target_symbol=ctrl.target_label,
        guide_index_within_gene=guide_index_within_class,
        guide_seq=ctrl.guide_seq,
        pam_seq=ctrl.pam_seq,
        chrom=ctrl.chrom,
        start=ctrl.start,
        end=ctrl.end,
        strand=ctrl.strand,
        # Controls don't have on/off-target scores by construction. We
        # emit NaN (rendered as "NaN" by the formatter) so consumers
        # never silently mistake them for "computed but zero".
        doench_rs1=float("nan"),
        doench_rs2=float("nan"),
        cfd_aggregate=float("nan"),
        off_target_count=-1,  # sentinel; rendered as "NaN"
        control_class=ctrl.control_class,
        notes=ctrl.notes,
    )


def assemble_library(
    *,
    balance_reports: Sequence[BalanceReport],
    controls: Sequence[ControlGuide],
) -> list[LibraryGuide]:
    """Combine per-gene selections + controls into a single ordered list.

    Order:

    1. Per-gene rows in input order; within a gene, by rank.
       Under-balanced genes have a ``UNDER_BALANCED:deficit=N`` flag
       appended to ``notes``.
    2. Non-targeting controls (deterministic order from generator).
    3. Safe-harbor controls (canonical order).
    4. Essential-gene controls (canonical order).

    ``library_index`` is assigned in emission order starting at 1 so
    downstream cloning pipelines can use it as the pool position.
    """
    rows: list[LibraryGuide] = []
    library_index = 0

    # Per-gene rows.
    for report in balance_reports:
        deficit = report.deficit
        extra = (
            f"UNDER_BALANCED:deficit={deficit}"
            if report.under_balanced
            else ""
        )
        for j, g in enumerate(report.selected, start=1):
            library_index += 1
            rows.append(
                _ranked_to_library_row(
                    g,
                    library_index=library_index,
                    target_symbol=report.gene,
                    guide_index_within_gene=j,
                    extra_notes=extra,
                )
            )

    # Controls: maintain a per-class counter so guide_index_within_gene
    # is meaningful within each class.
    class_counters: dict[str, int] = {}
    for ctrl in controls:
        class_counters[ctrl.control_class] = (
            class_counters.get(ctrl.control_class, 0) + 1
        )
        library_index += 1
        rows.append(
            _control_to_library_row(
                ctrl,
                library_index=library_index,
                guide_index_within_class=class_counters[ctrl.control_class],
            )
        )

    return rows


def _fmt_float(x: float, fmt: str = ".6f") -> str:
    """Format a float, rendering NaN as the string ``"NaN"``.

    Mirrors the convention from ``crispr_design.format_guides_tsv`` so
    downstream pipelines that already ingest the single-target TSV
    can read the library TSV with the same NaN handling.
    """
    import math

    if math.isnan(x):
        return "NaN"
    return format(x, fmt)


def _fmt_off_target_count(n: int) -> str:
    """``-1`` sentinel renders as ``"NaN"`` (control rows)."""
    return "NaN" if n < 0 else str(n)


def _fmt_int_coord(n: int) -> str:
    """``-1`` sentinel renders as ``"NaN"`` (non-targeting controls)."""
    return "NaN" if n < 0 else str(n)


def format_library_tsv(rows: Sequence[LibraryGuide]) -> bytes:
    """Serialise the assembled library to canonical TSV bytes (LF newlines)."""
    parts: list[str] = ["\t".join(LIBRARY_TSV_HEADER)]
    for r in rows:
        parts.append(
            "\t".join(
                (
                    str(r.library_index),
                    r.target_symbol,
                    str(r.guide_index_within_gene),
                    r.guide_seq,
                    r.pam_seq,
                    r.chrom,
                    _fmt_int_coord(r.start),
                    _fmt_int_coord(r.end),
                    r.strand,
                    _fmt_float(r.doench_rs1),
                    _fmt_float(r.doench_rs2),
                    _fmt_float(r.cfd_aggregate),
                    _fmt_off_target_count(r.off_target_count),
                    r.control_class,
                    r.notes,
                )
            )
        )
    blob = "\n".join(parts) + "\n"
    return blob.encode("utf-8")
