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

"""Track B v0 — Shared dataclasses + NamedTuples for the pegRNA design
pipeline.

T3 hosts these types ABOVE the modules that produce or consume them so
T2/T4/T5/T6/T7/T8/T9/T11 can each import without inter-task
dependencies (per the dependency-graph reviewer fix recorded in
``track-b-pegrna-design-plan.md`` §"Dependency Graph").

Type ownership matrix
---------------------
* :class:`EditSpec` — produced by T2 (edit-spec parser); consumed by
  T6 (enumerator) and T10 (CLI).
* :class:`PegRNAFoldingFeatures` — produced by T4 (ViennaRNA wrapper);
  consumed by T5 (PRIDICT) + T8 (ranker).
* :class:`PRIDICTScore` — produced by T5; consumed by T8. The ``notes``
  tuple carries free-form flags such as ``SCAFFOLD_OUT_OF_DISTRIBUTION``
  (when scoring a non-``sgRNA_canonical`` scaffold; see plan §T5).
* :class:`PegRNACandidate` — produced by T6; consumed by T7/T8/T9.
* :class:`PE3PegRNACandidate` — extends :class:`PegRNACandidate` (T7).
* :class:`OffTargetSite` — produced via ``crispr_design`` and re-shaped
  by T11; consumed by T8 / T9.
* :class:`RankedPegRNA` — produced by T8 (composite ranker); consumed
  by T9 (TSV/JSON output) + T10 (CLI). Field set is locked to T9's
  TSV schema so ``dataclasses.fields()`` drives the column ordering
  one-way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

__all__ = [
    "EditSpec",
    "PegRNAFoldingFeatures",
    "PRIDICTScore",
    "PegRNACandidate",
    "PE3PegRNACandidate",
    "OffTargetSite",
    "RankedPegRNA",
]


# ----------------------------------------------------------------------
# Edit specifier (T2 produces; T6/T10 consume)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class EditSpec:
    """Parsed edit specification.

    Produced by the T2 edit-spec parser from either simple notation
    (``"C>T at 100"``) or HGVS notation (``"c.123C>T"`` or
    ``"NM_007294.4:c.123C>T"``). For HGVS specs resolved against a
    minus-strand transcript, the ``alt_seq`` is complemented to genomic
    orientation and ``strand`` carries the transcript's strand for
    downstream debugging.

    Attributes
    ----------
    chrom:
        Reference contig (e.g. ``"chr17"``).
    start:
        0-indexed inclusive start coordinate on the reference.
    end:
        0-indexed exclusive end coordinate on the reference. For a
        single-base substitution, ``end == start + 1``.
    ref_seq:
        Reference allele as written in genomic-+ orientation. Empty
        string for pure insertions.
    alt_seq:
        Alternate allele in genomic-+ orientation. Empty string for
        pure deletions.
    edit_type:
        One of ``"substitution"``, ``"insertion"``, ``"deletion"``.
    notation_used:
        The original notation string (preserved for downstream
        debugging + TSV emission).
    strand:
        ``"+"`` for direct genomic-coordinate specs; transcript strand
        (``"+"`` or ``"-"``) for HGVS specs resolved through a
        transcript.
    """

    chrom: str
    start: int
    end: int
    ref_seq: str
    alt_seq: str
    edit_type: str  # "substitution" | "insertion" | "deletion"
    notation_used: str
    strand: str  # "+" or "-"


# ----------------------------------------------------------------------
# Folding features (T4 produces; T5/T8 consume)
# ----------------------------------------------------------------------


class PegRNAFoldingFeatures(NamedTuple):
    """ViennaRNA-derived secondary-structure features for a pegRNA.

    Computed by :mod:`bionpu.scoring.pegrna_folding` (T4) over the full
    composite RNA sequence (spacer + scaffold + RTT + PBS).

    Attributes
    ----------
    mfe_kcal:
        Minimum free energy in kcal/mol (negative; lower = more
        stable structure).
    mfe_structure:
        Dot-bracket secondary structure for the full pegRNA RNA at
        MFE.
    pbs_pairing_prob:
        Probability that the PBS region pairs with the target genome
        nicked-strand stub. Range [0, 1]. Derived from RNAplfold.
    scaffold_disruption:
        Fraction of canonical scaffold base-pairs disrupted vs the
        published reference structure for the chosen scaffold variant.
        Range [0, 1] where 0 = canonical structure preserved.
    """

    mfe_kcal: float
    mfe_structure: str
    pbs_pairing_prob: float
    scaffold_disruption: float


# ----------------------------------------------------------------------
# PRIDICT score (T5 produces; T8 consumes)
# ----------------------------------------------------------------------


class PRIDICTScore(NamedTuple):
    """PRIDICT 2.0 (Mathis 2024) per-pegRNA prediction.

    Attributes
    ----------
    efficiency:
        Predicted editing efficiency as a percentage (0-100). PRIDICT
        2.0's primary head.
    edit_rate:
        Predicted edit-rate fraction (0-1). PRIDICT 2.0 also exposes
        unedited-byproduct heads; ``edit_rate`` here is the desired-
        edit fraction.
    confidence:
        Model confidence proxy (typically derived from variance across
        ensemble members; range [0, 1]).
    notes:
        Free-form flags propagated to T9's TSV ``notes`` column.
        Currently includes ``"SCAFFOLD_OUT_OF_DISTRIBUTION"`` when
        scoring a non-``sgRNA_canonical`` scaffold (PRIDICT 2.0 was
        trained only on the canonical scaffold; see plan §T5
        reviewer edge-case #6) and ``"PRIDICT_FAILED"`` when scoring
        raises.
    """

    efficiency: float
    edit_rate: float
    confidence: float
    notes: tuple[str, ...]


# ----------------------------------------------------------------------
# pegRNA candidates (T6 produces; T7/T8/T9 consume)
# ----------------------------------------------------------------------


@dataclass(kw_only=True)
class PegRNACandidate:
    """A single PE2-strategy pegRNA candidate.

    Produced by the T6 enumerator. PE3-strategy candidates extend this
    with the nicking-sgRNA fields via :class:`PE3PegRNACandidate`.

    Attributes
    ----------
    spacer_seq:
        20-nt spacer (target-DNA-matching; emitted as DNA letters).
    pam_seq:
        Adjacent PAM (typically ``NGG`` for SpCas9; reported as the
        actual genomic 3-mer).
    scaffold_variant:
        Name of the scaffold variant from
        :data:`bionpu.genomics.pe_design.pegrna_constants.SCAFFOLD_VARIANTS`
        (e.g. ``"sgRNA_canonical"``, ``"evopreQ1"``).
    pbs_seq:
        Primer binding site (RNA letters; reverse-complement of the
        nicked-strand 5' stub).
    pbs_length:
        Length of ``pbs_seq`` in nt.
    rtt_seq:
        Reverse-transcriptase template (RNA letters; encodes the
        desired edit + flanking bases).
    rtt_length:
        Length of ``rtt_seq`` in nt.
    nick_site:
        Genomic coordinate of the Cas9-H840A nick (0-indexed, on the
        spacer's strand).
    full_pegrna_rna_seq:
        Complete pegRNA RNA sequence (spacer + scaffold + RTT + PBS),
        5' -> 3'.
    edit_position_in_rtt:
        0-indexed position within ``rtt_seq`` where the desired edit
        is encoded. Used for PRIDICT input prep + downstream debugging.
    strategy:
        ``"PE2"`` for this base class; ``"PE3"`` for the subclass.
    strand:
        ``"+"`` or ``"-"`` — which genomic strand the spacer matches.
    rt_product_seq:
        Post-edit genomic sequence the RT enzyme produces. Surfaced as
        a TSV column for downstream debugging + manual QC.
    chrom:
        Reference contig (mirrors :class:`EditSpec.chrom`).
    """

    spacer_seq: str
    pam_seq: str
    scaffold_variant: str
    pbs_seq: str
    pbs_length: int
    rtt_seq: str
    rtt_length: int
    nick_site: int
    full_pegrna_rna_seq: str
    edit_position_in_rtt: int
    strategy: str  # "PE2" by default; "PE3" via subclass
    strand: str  # "+" or "-"
    rt_product_seq: str
    chrom: str


@dataclass(kw_only=True)
class PE3PegRNACandidate(PegRNACandidate):
    """A PE2 pegRNA paired with a PE3 nicking sgRNA.

    Produced by T7 PE3 nicking selector. The base ``PegRNACandidate``
    fields describe the prime-editing pegRNA; the additional fields
    describe the OPPOSITE-strand nicking sgRNA whose nick stimulates
    DNA repair toward the edited strand.

    Attributes
    ----------
    nicking_spacer:
        20-nt spacer for the PE3 nicking sgRNA.
    nicking_pam:
        Adjacent NGG PAM for the nicking sgRNA.
    nicking_distance_from_pe2_nick:
        Absolute distance in bp between the PE2 nick site and the
        PE3 nicking sgRNA's nick site. PE3b prefers 40-90 bp.
    """

    nicking_spacer: str = ""
    nicking_pam: str = ""
    nicking_distance_from_pe2_nick: int = 0
    # Override the inherited default; PE3 candidates always carry the
    # "PE3" strategy tag.
    strategy: str = "PE3"


# ----------------------------------------------------------------------
# Off-target site (T11 re-shapes from crispr_design's scan output)
# ----------------------------------------------------------------------


class OffTargetSite(NamedTuple):
    """A single off-target hit for a spacer.

    Re-shaped by T11 from ``bionpu.genomics.crispr_design``'s existing
    off-target scan output. The shape mirrors the parent module's
    NamedTuple so the CFD aggregation pipeline composes cleanly.

    Attributes
    ----------
    chrom:
        Reference contig of the off-target hit.
    pos:
        0-indexed genomic position.
    strand:
        ``"+"`` or ``"-"``.
    mismatches:
        Number of mismatches between the spacer and the off-target
        site (within the seed + non-seed region).
    cfd_score:
        Per-site CFD (cutting frequency determination) score; range
        [0, 1] where 1 = perfect on-target predicted activity.
    in_paralog:
        ``True`` when the hit's ``(chrom, pos)`` falls inside a
        paralog gene span of the on-target gene (per
        :mod:`bionpu.data.paralog_mapper`). Track B v0.1 split
        aggregates: in-paralog hits are excluded from the safety-
        penalty CFD aggregate (the ``cfd_aggregate_pegrna`` term in
        the composite score) and reported separately as
        ``cfd_aggregate_paralog_pegrna``. Defaults to ``False`` so
        callers that don't supply paralog metadata stay on the v0
        codepath (every hit treated as a true off-target).
    """

    chrom: str
    pos: int
    strand: str
    mismatches: int
    cfd_score: float
    in_paralog: bool = False


# ----------------------------------------------------------------------
# RankedPegRNA (T8 produces; T9 emits; T10 returns from public API)
# ----------------------------------------------------------------------


@dataclass(kw_only=True)
class RankedPegRNA:
    """A ranked pegRNA carrying every field T9's TSV schema emits.

    Field ordering here is the canonical TSV column order; T9 reads
    ``dataclasses.fields()`` to drive emission so this dataclass is
    the single source of truth.

    Numeric fields use ``float('nan')`` to represent missing values
    (e.g. Mode-C synbio runs have ``cfd_aggregate_pegrna = NaN``).
    Optional string fields use ``None`` for missing PE3 fields when
    the candidate is PE2-only.
    """

    # ---- identity / edit ---- #
    pegrna_id: str
    edit_notation: str
    edit_position: int
    edit_type: str  # "substitution" | "insertion" | "deletion"

    # ---- spacer / pegRNA composition ---- #
    spacer_strand: str  # "+" or "-"
    spacer_seq: str
    pam_seq: str
    scaffold_variant: str
    pbs_seq: str
    pbs_length: int
    rtt_seq: str
    rtt_length: int
    rt_product_seq: str
    nick_site: int
    full_pegrna_rna_seq: str

    # ---- PE strategy + nicking sgRNA ---- #
    pe_strategy: str  # "PE2" or "PE3"
    nicking_spacer: str | None
    nicking_pam: str | None
    nicking_distance: int | None

    # ---- PRIDICT 2.0 prediction ---- #
    pridict_efficiency: float  # 0-100
    pridict_edit_rate: float  # 0-1
    pridict_confidence: float  # 0-1

    # ---- ViennaRNA folding features ---- #
    mfe_kcal: float
    scaffold_disruption: float
    pbs_pairing_prob: float

    # ---- off-target ---- #
    cfd_aggregate_pegrna: float
    off_target_count_pegrna: int
    cfd_aggregate_nicking: float | None
    off_target_count_nicking: int | None

    # ---- composite + ranking ---- #
    composite_pridict: float
    rank: int
    notes: tuple[str, ...] = field(default_factory=tuple)

    # ---- v0.1 paralog-aware off-target split (Track B v0.1) ---------- #
    # These columns extend the v0 schema at the END so the v0 TSV
    # column order remains stable for the first 32 columns; v0.1
    # readers see two new optional columns 33-34.
    #
    # paralog_hit_count_pegrna == count of off-target hits whose
    # OffTargetSite.in_paralog is True (excluded from
    # cfd_aggregate_pegrna and reported separately here).
    #
    # cfd_aggregate_paralog_pegrna == CRISPOR-style specificity computed
    # over the in-paralog hits ONLY. Range [0, 100]; 100 means "no
    # in-paralog hits at all". Always emitted when the on-target gene
    # has a known paralog family; 0.0 default + count 0 when the
    # paralog map didn't apply (unknown gene OR no in-paralog hits).
    paralog_hit_count_pegrna: int = 0
    cfd_aggregate_paralog_pegrna: float = 100.0
