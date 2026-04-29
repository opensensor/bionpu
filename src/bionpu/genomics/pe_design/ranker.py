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

"""Track B v0 — Composite pegRNA ranker (Task T8).

This module is the canonical "for each candidate, score it three ways
and combine" entry point for the pegRNA design pipeline. It composes:

* :func:`bionpu.scoring.pegrna_folding.compute_folding_features` (T4)
  for ViennaRNA-derived MFE / scaffold / PBS-pairing features,
* :class:`bionpu.scoring.pridict2.PRIDICT2Scorer` (T5) for editing
  efficiency,
* an injected ``off_target_scan_fn(spacer, *, max_mismatches=...)``
  callable (T11 produces the production adapter; tests inject a fake)
  for spacer-level off-target burden + CFD aggregate.

The composite score is::

    composite_pridict = pridict_efficiency
                        - 0.3 * cfd_aggregate_pegrna
                        - 0.15 * cfd_aggregate_nicking      (PE3 only; 0 for PE2)
                        - 0.1 * mfe_penalty(mfe_kcal)

where ``mfe_penalty(mfe_kcal) = max(0, -(mfe_kcal + 25))``. The penalty
is zero for ``mfe_kcal > -25`` kcal/mol and grows linearly for tighter
folds (very-negative MFE indicates strong intramolecular structure
that competes with reverse-transcription priming).

NaN handling
------------
If :class:`PRIDICT2Scorer` raises while scoring a candidate, the
ranker:

* sets ``pridict_efficiency / pridict_edit_rate / pridict_confidence``
  to NaN,
* leaves ``composite_pridict`` as NaN,
* appends ``"PRIDICT_FAILED"`` to ``notes``,
* sorts the candidate AFTER all valid (non-NaN) candidates.

Determinism
-----------
Sort key is ``(composite_is_nan, -composite_pridict, chrom, nick_site,
strand, pbs_length, rtt_length, scaffold_variant, pe_strategy)``
ascending. NaN composites sort last; ties between valid composites
break on candidate identity ascending so two runs over the same input
yield byte-identical ``pegrna_id`` ordering.

PE3 vs PE2
----------
:class:`PE3PegRNACandidate` carries ``nicking_spacer`` /
``nicking_pam`` / ``nicking_distance_from_pe2_nick`` fields; the
ranker scans the nicking sgRNA's spacer for off-targets in addition
to the pegRNA spacer. PE2 candidates have ``cfd_aggregate_nicking ==
None`` and ``off_target_count_nicking == None`` in the output
:class:`RankedPegRNA`; the composite formula treats their nicking
contribution as zero.
"""

from __future__ import annotations

import math
from dataclasses import fields as _dc_fields
from typing import Any, Callable, Iterable

from bionpu.genomics.pe_design.pegrna_constants import SCAFFOLD_VARIANTS
from bionpu.genomics.pe_design.types import (
    EditSpec,
    PE3PegRNACandidate,
    PegRNACandidate,
    PegRNAFoldingFeatures,
    PRIDICTScore,
    RankedPegRNA,
)

__all__ = [
    "rank_candidates",
    "mfe_penalty",
    "compute_composite",
]


# ---------------------------------------------------------------------------
# Scoring kernels
# ---------------------------------------------------------------------------


def mfe_penalty(mfe_kcal: float) -> float:
    """Return the v0 MFE penalty term.

    ``mfe_penalty(mfe_kcal) = max(0, -(mfe_kcal + 25))``

    * ``mfe_kcal == -50``: penalty 25 (very tight fold).
    * ``mfe_kcal == -25``: penalty 0 (boundary).
    * ``mfe_kcal == -10``: penalty 0 (loose fold).
    * NaN MFE: returns 0 (treat as non-penalising; the upstream folding
      extractor returns NaN-tagged features only on degenerate empty
      sequences which is itself a separate red-flag the caller can
      handle via ``notes``).

    This is a v0 heuristic; v1 may calibrate the threshold against
    bench data once we have wet-lab pegRNA outcomes paired with MFE.
    """
    if math.isnan(mfe_kcal):
        return 0.0
    return max(0.0, -(mfe_kcal + 25.0))


def compute_composite(
    *,
    pridict_efficiency: float,
    cfd_aggregate_pegrna: float,
    cfd_aggregate_nicking: float | None,
    mfe_kcal: float,
) -> float:
    """Compute the composite score from the four contributors.

    ``cfd_aggregate_nicking`` is ``None`` for PE2 candidates; the
    formula treats that as zero. NaN inputs propagate (any NaN input
    yields a NaN output) so the ranker can use the result as the
    PRIDICT-failed sentinel.
    """
    if math.isnan(pridict_efficiency):
        return float("nan")
    nicking_term = 0.0 if cfd_aggregate_nicking is None else cfd_aggregate_nicking
    return (
        pridict_efficiency
        - 0.3 * cfd_aggregate_pegrna
        - 0.15 * nicking_term
        - 0.1 * mfe_penalty(mfe_kcal)
    )


# ---------------------------------------------------------------------------
# pegrna_id construction
# ---------------------------------------------------------------------------


def _make_pegrna_id(candidate: PegRNACandidate) -> str:
    """Build a stable, human-readable pegRNA identifier.

    Format: ``{chrom}_{nick}_{strand_tag}_pbs{N}_rtt{M}_{scaffold}_{strategy}_{seq8}``

    The id is fully derived from the candidate's content — no input
    insertion order — so two ``rank_candidates`` runs over the same
    candidate set yield byte-identical ids. The 8-char suffix is the
    first 8 chars of the SHA1 of ``full_pegrna_rna_seq`` to keep the id
    unique when every dataclass field matches but the RNA differs (e.g.
    same nick_site / pbs_length but different RTT bases due to scaffold
    variant rewrite).
    """
    import hashlib

    strand_tag = "p" if candidate.strand == "+" else "m"
    suffix = hashlib.sha1(
        candidate.full_pegrna_rna_seq.encode("ascii")
    ).hexdigest()[:8]
    return (
        f"{candidate.chrom}_{candidate.nick_site}_{strand_tag}_"
        f"pbs{candidate.pbs_length}_rtt{candidate.rtt_length}_"
        f"{candidate.scaffold_variant}_{candidate.strategy}_{suffix}"
    )


# ---------------------------------------------------------------------------
# Per-candidate scoring (PRIDICT + folding + off-target)
# ---------------------------------------------------------------------------


def _safe_score_pridict(
    scorer: Any,
    *,
    pegrna_seq: str,
    scaffold_variant: str,
    target_context: str,
    folding_features: PegRNAFoldingFeatures | None,
    spacer_dna: str | None = None,
    pbs_dna: str | None = None,
    rtt_dna: str | None = None,
) -> tuple[PRIDICTScore | None, bool]:
    """Score one pegRNA, swallowing any exception.

    Returns ``(score, failed)`` — ``score`` is ``None`` when ``failed``.

    Component hints (``spacer_dna``/``pbs_dna``/``rtt_dna``) are
    forwarded to scorers that accept them (canonical
    :class:`bionpu.scoring.pridict2.PRIDICT2Scorer`); they enable the
    scaffold-invariant component-triple match against PRIDICT's
    enumeration cache when T6's assembled pegRNA string differs from
    PRIDICT's (the standard case — T6 ships the Anzalone-2019
    canonical scaffold body while PRIDICT's pegRNAfinder bakes in
    Chen 2013's F+E optimised scaffold). Test stubs that only accept
    the legacy 4-arg signature still work because the ranker drops
    the hints on TypeError-fallback.
    """
    try:
        score = scorer.score(
            pegrna_seq,
            scaffold_variant=scaffold_variant,
            target_context=target_context,
            folding_features=folding_features,
            spacer_dna=spacer_dna,
            pbs_dna=pbs_dna,
            rtt_dna=rtt_dna,
        )
    except TypeError:
        # Stub scorers (test fixtures) may not accept the hint kwargs.
        # Fall back to the legacy signature; component-triple lookup
        # is opportunistic, never required.
        try:
            score = scorer.score(
                pegrna_seq,
                scaffold_variant=scaffold_variant,
                target_context=target_context,
                folding_features=folding_features,
            )
        except Exception:  # noqa: BLE001 — explicit policy: NaN-and-flag
            return None, True
    except Exception:  # noqa: BLE001 — explicit policy: NaN-and-flag
        return None, True
    return score, False


def _scan_off_targets(
    off_target_scan_fn: Callable[..., tuple[list, float, int]],
    spacer: str,
    *,
    max_mismatches: int,
) -> tuple[float, int]:
    """Run the off-target callable and return ``(cfd_aggregate, count)``.

    The callable's full signature in production (T11) returns
    ``(sites_list, cfd_aggregate, count)``; the ranker only needs the
    aggregate + count for composite + reporting.
    """
    sites, cfd_agg, count = off_target_scan_fn(spacer, max_mismatches=max_mismatches)
    del sites  # T8 doesn't surface per-site rows; T9/T10 may.
    return float(cfd_agg), int(count)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def rank_candidates(
    candidates: Iterable[PegRNACandidate | PE3PegRNACandidate],
    *,
    edit_spec: EditSpec,
    target_context: str,
    off_target_scan_fn: Callable[..., tuple[list, float, int]],
    scorer: Any,
    folding_extractor: Callable[..., PegRNAFoldingFeatures],
    max_mismatches: int = 4,
    top_n: int = 20,
) -> list[RankedPegRNA]:
    """Score, combine, and rank pegRNA candidates.

    Parameters
    ----------
    candidates:
        Iterable of :class:`PegRNACandidate` or
        :class:`PE3PegRNACandidate`. Mixed PE2/PE3 lists are supported.
    edit_spec:
        The edit specifier the candidates were enumerated for. Used to
        populate :class:`RankedPegRNA`'s ``edit_notation`` /
        ``edit_position`` / ``edit_type`` columns.
    target_context:
        PRIDICT 2.0 target-context string (``XXX(orig/edit)YYY``)
        passed through to :class:`PRIDICT2Scorer.score`.
    off_target_scan_fn:
        Callable ``(spacer_seq, *, max_mismatches=...) -> (sites,
        cfd_aggregate, count)``. T11 provides the production adapter;
        unit tests inject a fake.
    scorer:
        Object exposing ``score(pegrna_seq, *, scaffold_variant,
        target_context, folding_features=None) -> PRIDICTScore``.
        :class:`bionpu.scoring.pridict2.PRIDICT2Scorer` is the
        canonical impl.
    folding_extractor:
        Callable matching
        :func:`bionpu.scoring.pegrna_folding.compute_folding_features`'s
        signature ``(spacer, scaffold, rtt, pbs, *, scaffold_variant)
        -> PegRNAFoldingFeatures``.
    max_mismatches:
        Forwarded to ``off_target_scan_fn`` for both the pegRNA spacer
        and the PE3 nicking spacer scans.
    top_n:
        Maximum number of :class:`RankedPegRNA` rows to return. The
        full sorted list is computed first and then truncated.

    Returns
    -------
    list[RankedPegRNA]
        Sorted by :func:`compute_composite` descending (NaN last) with
        deterministic tie-breaking. ``rank`` is 1-indexed and assigned
        AFTER truncation reflects the row's position in the returned
        list.
    """
    candidates_list = list(candidates)

    # Cache for the scaffold body's folding features per scaffold name
    # is implemented inside compute_folding_features (T4). We instead
    # cache full-pegRNA folding features here keyed on the candidate's
    # full RNA sequence so identical pegRNAs (rare but possible across
    # variants) share work.
    folding_cache: dict[str, PegRNAFoldingFeatures] = {}
    off_target_cache: dict[str, tuple[float, int]] = {}

    rows: list[RankedPegRNA] = []

    for cand in candidates_list:
        notes: list[str] = []
        is_pe3 = isinstance(cand, PE3PegRNACandidate)

        # ---- Folding features (T4) ---------------------------------- #
        folding_key = cand.full_pegrna_rna_seq
        if folding_key in folding_cache:
            folding = folding_cache[folding_key]
        else:
            scaffold_body = SCAFFOLD_VARIANTS.get(cand.scaffold_variant) or ""
            try:
                folding = folding_extractor(
                    cand.spacer_seq,
                    scaffold_body,
                    cand.rtt_seq,
                    cand.pbs_seq,
                    scaffold_variant=cand.scaffold_variant,
                )
            except Exception:  # noqa: BLE001
                # Folding failure is non-fatal — fall back to NaN
                # features; composite formula tolerates NaN MFE.
                folding = PegRNAFoldingFeatures(
                    mfe_kcal=float("nan"),
                    mfe_structure="",
                    pbs_pairing_prob=float("nan"),
                    scaffold_disruption=float("nan"),
                )
                notes.append("FOLDING_FAILED")
            folding_cache[folding_key] = folding

        # ---- PRIDICT efficiency (T5) -------------------------------- #
        # Pass component triple as hints so the PRIDICT wrapper can
        # fall back to a scaffold-invariant component-wise match
        # against PRIDICT's enumeration cache when T6's full
        # assembled pegRNA string differs from PRIDICT's (T10 surface).
        pridict_score, failed = _safe_score_pridict(
            scorer,
            pegrna_seq=cand.full_pegrna_rna_seq,
            scaffold_variant=cand.scaffold_variant,
            target_context=target_context,
            folding_features=folding,
            spacer_dna=cand.spacer_seq,
            pbs_dna=cand.pbs_seq,
            rtt_dna=cand.rtt_seq,
        )
        if failed:
            efficiency = float("nan")
            edit_rate = float("nan")
            confidence = float("nan")
            notes.append("PRIDICT_FAILED")
        else:
            efficiency = float(pridict_score.efficiency)
            edit_rate = float(pridict_score.edit_rate)
            confidence = float(pridict_score.confidence)
            # Propagate any flags PRIDICT emitted (e.g.
            # "SCAFFOLD_OUT_OF_DISTRIBUTION").
            for note in pridict_score.notes:
                if note not in notes:
                    notes.append(note)

        # ---- Off-target scan (pegRNA spacer) ------------------------ #
        if cand.spacer_seq in off_target_cache:
            cfd_pegrna, count_pegrna = off_target_cache[cand.spacer_seq]
        else:
            cfd_pegrna, count_pegrna = _scan_off_targets(
                off_target_scan_fn,
                cand.spacer_seq,
                max_mismatches=max_mismatches,
            )
            off_target_cache[cand.spacer_seq] = (cfd_pegrna, count_pegrna)

        # ---- Off-target scan (PE3 nicking spacer) ------------------- #
        if is_pe3:
            nick_spacer = cand.nicking_spacer
            if nick_spacer in off_target_cache:
                cfd_nicking, count_nicking = off_target_cache[nick_spacer]
            else:
                cfd_nicking, count_nicking = _scan_off_targets(
                    off_target_scan_fn,
                    nick_spacer,
                    max_mismatches=max_mismatches,
                )
                off_target_cache[nick_spacer] = (cfd_nicking, count_nicking)
            cfd_nicking_field: float | None = cfd_nicking
            count_nicking_field: int | None = count_nicking
            nicking_spacer_field: str | None = cand.nicking_spacer
            nicking_pam_field: str | None = cand.nicking_pam
            nicking_distance_field: int | None = cand.nicking_distance_from_pe2_nick
        else:
            cfd_nicking_field = None
            count_nicking_field = None
            nicking_spacer_field = None
            nicking_pam_field = None
            nicking_distance_field = None

        # ---- Composite score ---------------------------------------- #
        composite = compute_composite(
            pridict_efficiency=efficiency,
            cfd_aggregate_pegrna=cfd_pegrna,
            cfd_aggregate_nicking=cfd_nicking_field,
            mfe_kcal=folding.mfe_kcal,
        )

        rows.append(
            RankedPegRNA(
                pegrna_id=_make_pegrna_id(cand),
                edit_notation=edit_spec.notation_used,
                edit_position=edit_spec.start,
                edit_type=edit_spec.edit_type,
                spacer_strand=cand.strand,
                spacer_seq=cand.spacer_seq,
                pam_seq=cand.pam_seq,
                scaffold_variant=cand.scaffold_variant,
                pbs_seq=cand.pbs_seq,
                pbs_length=cand.pbs_length,
                rtt_seq=cand.rtt_seq,
                rtt_length=cand.rtt_length,
                rt_product_seq=cand.rt_product_seq,
                nick_site=cand.nick_site,
                full_pegrna_rna_seq=cand.full_pegrna_rna_seq,
                pe_strategy="PE3" if is_pe3 else "PE2",
                nicking_spacer=nicking_spacer_field,
                nicking_pam=nicking_pam_field,
                nicking_distance=nicking_distance_field,
                pridict_efficiency=efficiency,
                pridict_edit_rate=edit_rate,
                pridict_confidence=confidence,
                mfe_kcal=folding.mfe_kcal,
                scaffold_disruption=folding.scaffold_disruption,
                pbs_pairing_prob=folding.pbs_pairing_prob,
                cfd_aggregate_pegrna=cfd_pegrna,
                off_target_count_pegrna=count_pegrna,
                cfd_aggregate_nicking=cfd_nicking_field,
                off_target_count_nicking=count_nicking_field,
                composite_pridict=composite,
                rank=0,  # filled in after sort
                notes=tuple(notes),
            )
        )

    # ---- Sort: NaN-last + deterministic tie-break ----------------- #
    def _sort_key(row: RankedPegRNA) -> tuple:
        is_nan = math.isnan(row.composite_pridict)
        # NaN composites sort last (is_nan=True > is_nan=False).
        # Among non-NaN: composite descending → use -composite ascending.
        # Tie-break on identity tuple ascending for byte-identical
        # ordering across runs.
        primary = 0.0 if is_nan else -row.composite_pridict
        return (
            is_nan,
            primary,
            row.spacer_strand,  # "+" < "-" lexicographically — fine; deterministic
            row.nick_site,
            row.pbs_length,
            row.rtt_length,
            row.scaffold_variant,
            row.pe_strategy,
            row.pegrna_id,  # final unique fallback
        )

    rows.sort(key=_sort_key)

    # ---- Truncate + assign ranks --------------------------------- #
    rows = rows[: max(0, int(top_n))]
    for i, row in enumerate(rows, start=1):
        # ``RankedPegRNA`` is kw-only-immutable-ish; reassign via
        # __setattr__ since the dataclass isn't frozen.
        row.rank = i

    # Sanity: confirm all dataclass fields populated (cheap O(N) check
    # that catches any future schema drift; raises a clear error
    # rather than emitting a half-formed row).
    if rows:
        expected_field_names = {f.name for f in _dc_fields(RankedPegRNA)}
        actual = set(rows[0].__dict__.keys())
        missing = expected_field_names - actual
        if missing:
            raise RuntimeError(
                f"ranker emitted RankedPegRNA missing fields: {missing}"
            )

    return rows
