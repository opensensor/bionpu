# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

"""Track A v0 — Base editor guide ranker.

Composes the IUPAC PAM filter (silicon, when artifacts present; CPU
oracle otherwise) with edit-window classification, bystander counting,
and (optional) off-target CFD scoring.

The output is a ranked list of :class:`BaseEditorGuide` records with
fields suitable for a CRISPOR-grade BE design TSV.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from bionpu.data.pam_iupac_oracle import find_pam_matches

from .bystander import bystander_count, enumerate_bystander_edits
from .pam_variants import (
    PHASE1_BE_VARIANTS,
    PHASE1_CAS9_VARIANTS,
    BaseEditorSpec,
    Cas9PamSpec,
    get_be_spec,
    get_cas9_spec,
)
from .window_score import (
    PROTOSPACER_LEN,
    activity_window_slice,
    activity_window_target_positions,
    target_in_window,
)

__all__ = [
    "BaseEditorGuide",
    "design_base_editor_guides",
    "composite_be",
]


@dataclass(frozen=True)
class BaseEditorGuide:
    """One ranked base-editor guide.

    Attributes:
        guide_seq: 20-nt protospacer (5' to 3').
        pam_seq: concrete IUPAC PAM bases at the genomic site (matches
            the BE variant's Cas9 PAM template).
        target_pos: 0-indexed position in the input target sequence
            where the protospacer starts.
        target_base: ``"C"`` (CBE) or ``"A"`` (ABE).
        target_pos_in_protospacer: 0-indexed position of the desired-edit
            base within the protospacer (e.g. ``5`` if the C/A landed at
            nt 6 from the 5' end).
        in_activity_window: True iff the target base position falls
            within the BE variant's activity window (CBE: 4-8;
            ABE7.10: 4-7 — 1-indexed from PAM-distal end).
        bystander_count: number of additional editable bases (Cs for
            CBE, As for ABE) within the activity window.
        bystander_positions: sorted list of 0-indexed bystander positions
            within the protospacer.
        off_target_count: number of off-target sites passing the PAM
            filter at the locus-level off-target scan (0 when no genome
            supplied; the v0 ranker does NOT perform a full Cas-OFFinder
            scan — that's deferred to v1+ via the locked
            ``crispr/match_multitile_memtile`` kernel hookup).
        cfd_aggregate: sum of CFD scores for off-target hits (0.0 when
            ``off_target_count == 0``).
        rank_score: composite ranking score; higher is better.
        notes: free-form provenance / advisory string.
    """

    guide_seq: str
    pam_seq: str
    target_pos: int
    target_base: str
    target_pos_in_protospacer: int
    in_activity_window: bool
    bystander_count: int
    bystander_positions: tuple[int, ...]
    off_target_count: int = 0
    cfd_aggregate: float = 0.0
    rank_score: float = 0.0
    notes: str = ""


def _enumerate_pam_hits(
    target_seq: str,
    pam: str,
    *,
    use_silicon: bool,
) -> list[tuple[int, int]]:
    """Enumerate forward-strand PAM hits in ``target_seq``.

    For v0, we use the CPU IUPAC oracle by default. The silicon path is
    available via :class:`bionpu.kernels.crispr.pam_filter_iupac.BionpuPamFilterIupac`
    when the xclbin artifacts are present; this composition layer
    dispatches to the silicon op-class when ``use_silicon=True`` AND
    the artifacts are present.

    Args:
        target_seq: ACGT target string. Non-ACGT bases reset the rolling
            state per the oracle's contract.
        pam: IUPAC PAM string.
        use_silicon: True to attempt silicon dispatch; False (default)
            uses the CPU oracle.

    Returns:
        ``[(query_pos, strand=0), ...]`` sorted by ``query_pos asc``.
    """
    if use_silicon:
        try:
            from bionpu.kernels.crispr.pam_filter_iupac import (
                BionpuPamFilterIupac,
            )

            op = BionpuPamFilterIupac(pam=pam, n_tiles=4)
            if op.artifacts_present():
                # Silicon path: pack target_seq, dispatch, return hits.
                from bionpu.data.kmer_oracle import pack_dna_2bit

                packed = pack_dna_2bit(target_seq)
                hits = op(packed_seq=packed, target_seq=target_seq)
                return [(int(p), int(s)) for p, s in hits]
        except (ImportError, Exception):  # pragma: no cover - best-effort silicon
            pass
    # CPU oracle fallback (also the default path).
    return find_pam_matches(target_seq, pam)


def _composite_rank_score(
    *,
    in_window: bool,
    bystander_count: int,
    cfd_aggregate: float,
) -> float:
    """Compute a simple composite rank score.

    Rules (heuristic; v0/v1 — Phase 2 will replace with BE-Hive scoring):

    * ``+1.0`` if target is in the activity window; ``0.0`` otherwise.
    * ``-0.2`` per bystander edit (penalty).
    * ``-0.5 * cfd_aggregate`` (off-target penalty; raw CFD sum from
      :func:`bionpu.genomics.be_design.off_target.off_target_scan_for_be_guide`).

    NaN handling: ``cfd_aggregate=NaN`` (synbio mode — no scan
    performed) contributes 0 to the off-target penalty, so guides
    in synbio mode are ranked purely by edit-window + bystander.

    Higher is better. The absolute scale isn't load-bearing — only the
    rank order is.
    """
    score = 0.0
    if in_window:
        score += 1.0
    score -= 0.2 * float(bystander_count)
    if not (cfd_aggregate is None or math.isnan(float(cfd_aggregate))):
        score -= 0.5 * float(cfd_aggregate)
    return score


def composite_be(g: "BaseEditorGuide") -> float:
    """Public composite BE ranker.

    Mirrors :func:`_composite_rank_score` but takes a
    :class:`BaseEditorGuide` and is the canonical entry point for the
    Track A v1 ranker. v0's :class:`BaseEditorGuide` exposes
    ``in_activity_window``, ``bystander_count``, and ``cfd_aggregate``;
    NaN ``cfd_aggregate`` (synbio mode) is treated as a 0 off-target
    penalty.
    """
    return _composite_rank_score(
        in_window=g.in_activity_window,
        bystander_count=g.bystander_count,
        cfd_aggregate=g.cfd_aggregate,
    )


def design_base_editor_guides(
    target_seq: str,
    *,
    be_variant: str,
    cas9_variant: str = "wt",
    genome_path: Optional[str | Path] = None,
    top_n: int = 20,
    use_silicon: bool = False,
    require_in_window: bool = False,
    notes_prefix: str = "",
) -> list[BaseEditorGuide]:
    """Design base editor guides against a target sequence.

    v0 scope (per ``PRDs/PRD-crispr-state-of-the-art-roadmap.md`` §3.1
    Phase 1):

    * SpCas9 wt (NGG) + SpCas9-NG (NG); Phase 2 Cas9 zoo (SpRY,
      SaCas9-KKH, ...) is silicon-supported but not in the v0 cleared
      acceptance gate.
    * BE4max (CBE) + ABE7.10 (ABE).

    The ranker is a stateless composition over:

    1. PAM filter (silicon or CPU oracle).
    2. Edit-window classification.
    3. Bystander enumeration.
    4. Off-target scan (deferred when ``genome_path`` is None — this is
       the documented "synbio" mode per PRD-guide-design v0.2 §1.1).

    Args:
        target_seq: ACGT (case-insensitive) target string. Should be at
            least ``20 + len(pam)`` bases for any guide to be enumerated.
        be_variant: BE variant name (``"BE4max"`` or ``"ABE7.10"``;
            broader zoo deferred to Phase 2).
        cas9_variant: Cas9 PAM variant name (``"wt"``, ``"NG"``,
            ``"SpRY"``, ``"SaCas9-KKH"``). v0 acceptance gate covers
            ``"wt"`` + ``"NG"``.
        genome_path: Optional path to a reference FASTA for off-target
            scanning. ``None`` (default) skips off-target scoring;
            output guides will have ``off_target_count=0``,
            ``cfd_aggregate=0.0``, and a ``NO_OFF_TARGET_SCAN`` note
            (mirrors PRD-guide-design v0.2 §1.1 synbio mode).
        top_n: Return the top-N ranked guides. ``0`` returns all.
        use_silicon: Attempt silicon PAM dispatch when artifacts are
            present. Defaults to False (CPU oracle).
        require_in_window: If True, drop guides whose target base is
            not in the activity window. Default False (all guides
            ranked; users can sort by ``in_activity_window`` first).
        notes_prefix: Prepended to every guide's ``notes`` field
            (e.g. for caller provenance tagging).

    Returns:
        A ranked list of :class:`BaseEditorGuide` records, sorted by
        ``rank_score`` descending. Length is bounded by ``top_n``
        (or the total candidate count if smaller).
    """
    cas9 = get_cas9_spec(cas9_variant)
    be = get_be_spec(be_variant)

    target_seq_u = target_seq.upper()
    pam_template = cas9.pam_iupac
    pam_len = len(pam_template)

    notes_acc: list[str] = []
    if notes_prefix:
        notes_acc.append(notes_prefix)
    if cas9_variant not in PHASE1_CAS9_VARIANTS:
        notes_acc.append(f"PHASE2_CAS9={cas9_variant}")
    if be_variant not in PHASE1_BE_VARIANTS:
        notes_acc.append(f"PHASE2_BE={be_variant}")
    synbio_mode = genome_path is None
    if synbio_mode:
        notes_acc.append("NO_OFF_TARGET_SCAN")

    # Optional pre-load: read the FASTA once and pass the
    # ``[(chrom, seq), ...]`` to every per-guide call so we avoid
    # re-reading on every iteration. This dominates the off-target
    # wall when the genome is multi-MB.
    genome_records: Optional[list[tuple[str, str]]] = None
    if not synbio_mode:
        from .off_target import _read_fasta_records  # noqa: PLC0415
        genome_records = list(_read_fasta_records(Path(genome_path)))

    guides: list[BaseEditorGuide] = []

    # ----- PAM enumeration (silicon when artifacts present, else CPU oracle) -----
    pam_hits = _enumerate_pam_hits(
        target_seq_u, pam_template, use_silicon=use_silicon
    )

    # For each PAM hit, the protospacer sits 5' of it (PAM is 3' of
    # spacer). Skip if there isn't a 20-nt protospacer 5' of the PAM.
    for pam_pos, _strand in pam_hits:
        proto_start = pam_pos - PROTOSPACER_LEN
        if proto_start < 0:
            continue
        protospacer = target_seq_u[proto_start : proto_start + PROTOSPACER_LEN]
        if len(protospacer) != PROTOSPACER_LEN:
            continue
        # Skip if protospacer has any non-ACGT character.
        if any(ch not in "ACGT" for ch in protospacer):
            continue
        pam_concrete = target_seq_u[pam_pos : pam_pos + pam_len]
        if any(ch not in "ACGT" for ch in pam_concrete):
            continue

        # Find target-base positions within the activity window.
        target_positions = activity_window_target_positions(
            protospacer, be
        )
        if not target_positions:
            # No editable base in window. We can still emit the guide
            # (so the user sees it) IF require_in_window is False, with
            # in_activity_window=False. Pick a placeholder target_pos in
            # window for the bystander API contract.
            if require_in_window:
                continue
            # For the placeholder record, we have no concrete target
            # position; use lo so bystander_count returns the count of
            # editable bases at all positions in the window (which is 0
            # in this branch by construction, since target_positions
            # was empty).
            lo, _hi = activity_window_slice(be)
            in_win = False
            target_pos_p = lo
            byst_count = 0
            byst_positions: tuple[int, ...] = ()
        else:
            # Use the FIRST target position in window (closest to
            # PAM-distal end) as the canonical edit. v1 will let users
            # pick which target by config.
            target_pos_p = target_positions[0]
            in_win = True
            byst_positions_list = enumerate_bystander_edits(
                protospacer, target_pos_p, be
            )
            byst_count = len(byst_positions_list)
            byst_positions = tuple(byst_positions_list)

        # v1: optional off-target scan via the locked
        # crispr/match_multitile_memtile (silicon) or cpu_scan (CPU
        # oracle). Synbio mode (`genome_path=None`) yields NaN
        # `cfd_aggregate` so the ranker downstream treats it as
        # "unknown" rather than "0 off-targets".
        if synbio_mode:
            off_target_count = 0
            cfd_aggregate = float("nan")
        else:
            from .off_target import (  # noqa: PLC0415
                off_target_scan_for_be_guide,
            )
            try:
                _sites, cfd_aggregate, off_target_count = (
                    off_target_scan_for_be_guide(
                        guide_protospacer=protospacer,
                        pam_seq=pam_concrete,
                        genome_path=genome_path or "",
                        max_mismatches=4,
                        device="cpu",  # v1 default; npu opt-in via config
                        genome_records=genome_records,
                    )
                )
            except (FileNotFoundError, ValueError) as exc:  # pragma: no cover
                # Defensive: degrade to NaN if the scan fails for any
                # reason (missing FASTA, malformed sequence, ...).
                cfd_aggregate = float("nan")
                off_target_count = 0
                if "OFF_TARGET_FAILED" not in notes_acc:
                    notes_acc.append(f"OFF_TARGET_FAILED={type(exc).__name__}")

        rank = _composite_rank_score(
            in_window=in_win,
            bystander_count=byst_count,
            cfd_aggregate=cfd_aggregate,
        )

        notes = ";".join(notes_acc) if notes_acc else ""

        guides.append(
            BaseEditorGuide(
                guide_seq=protospacer,
                pam_seq=pam_concrete,
                target_pos=proto_start,
                target_base=be.target_base,
                target_pos_in_protospacer=target_pos_p,
                in_activity_window=in_win,
                bystander_count=byst_count,
                bystander_positions=byst_positions,
                off_target_count=off_target_count,
                cfd_aggregate=cfd_aggregate,
                rank_score=rank,
                notes=notes,
            )
        )

    # Rank descending by rank_score, then ascending by target_pos for
    # stable tie-breaking.
    guides.sort(key=lambda g: (-g.rank_score, g.target_pos))
    if top_n > 0:
        guides = guides[:top_n]
    return guides
