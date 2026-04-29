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

"""Track B v0 — PE3 nicking sgRNA selector (Task T7 of
``track-b-pegrna-design-plan.md``).

Given a PE2 :class:`PegRNACandidate` (T6 enumerator output) and the
target + strand genome window, this module finds candidate PE3 nicking
sgRNAs on the OPPOSITE strand within a configurable distance band of
the PE2 nick site (default 40-100 bp; PRIDICT 2.0 default 40-90 bp).

PE3 strategy (Anzalone 2019, Nature)
------------------------------------

The PE2 pegRNA's spacer + scaffold + RTT + PBS edits one strand of the
target. The PE3 strategy adds a *second* sgRNA — a "nicking sgRNA" —
that nicks the OPPOSITE strand at a position 40-100 bp from the PE2
nick. The opposite-strand nick stimulates the cell's mismatch-repair
machinery to use the edited strand as the template, dramatically
boosting editing efficiency (3-10x in HEK293 reporters per Anzalone
2019 Fig 4).

Geometry
--------

* :attr:`PegRNACandidate.nick_site` is a 0-indexed *genomic + strand*
  coordinate of the PE2 phosphodiester break, regardless of which
  strand the PE2 spacer is on (mirrors the convention T6's enumerator
  established).
* The PE3 nicking guide must sit on the opposite strand of the PE2
  spacer:
    - PE2 spacer on ``+`` → nicking on ``-`` → on the + strand we see
      ``CCN`` at the nicking PAM location (= NGG on -).
    - PE2 spacer on ``-`` → nicking on ``+`` → on the + strand we see
      ``NGG`` at the nicking PAM location.
* The PE3 nick is 3 nt 5' of its PAM on the nicking sgRNA's strand;
  per the same convention as T6, we report ``pe3_nick_site_plus`` in +
  coordinates.
* Distance = ``abs(pe3_nick_site_plus - pe2.nick_site)`` (both in +
  coords).

Off-bystander filter
--------------------

The whole point of PE3 is to nick the *other* strand at a position
distinct from the edit window — if the nicking guide were to cleave
inside the edit site, both strands would be cut and PE3's repair
template would no longer be the edited strand. The optional
``edit_region`` parameter lets the caller (T8 ranker) supply the edit
window in + coords; nick sites within that window are filtered out.
By default ``edit_region=None`` and no off-bystander filter is applied
(keeps the function pure + composable; the T8 ranker is the canonical
caller that wires the edit window).

Sort
----

Output is sorted by ``nicking_distance_from_pe2_nick`` ascending —
deterministic, and aligns with PRIDICT 2.0's preference for nicks in
the 40-90 bp band.
"""

from __future__ import annotations

from bionpu.genomics.pe_design.types import PE3PegRNACandidate, PegRNACandidate

__all__ = ["select_nicking_guides"]


# --------------------------------------------------------------------- #
# Sequence utilities (kept local; mirror enumerator.py's conventions)
# --------------------------------------------------------------------- #


_RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _reverse_complement(seq: str) -> str:
    """Return the reverse complement of an A/C/G/T/N sequence (uppercased)."""
    return seq.translate(_RC_TABLE)[::-1].upper()


def _matches_ngg(pam: str) -> bool:
    """Return True iff ``pam`` is a 3-mer matching ``NGG`` (any base + GG)."""
    if len(pam) != 3:
        return False
    if pam[0] not in "ACGT":
        return False
    return pam[1] == "G" and pam[2] == "G"


# --------------------------------------------------------------------- #
# Strand utility
# --------------------------------------------------------------------- #


def _opposite_strand(strand: str) -> str:
    if strand == "+":
        return "-"
    if strand == "-":
        return "+"
    raise ValueError(f"strand must be '+' or '-'; got {strand!r}")


# --------------------------------------------------------------------- #
# Per-strand nicking-guide enumeration on the OPPOSITE strand
# --------------------------------------------------------------------- #


def _enumerate_nicking_guides_on_strand(
    *,
    target_genome_seq: str,
    nicking_strand: str,
    pe2_nick_site_plus: int,
    distance_range: range,
    edit_region: tuple[int, int] | None,
) -> list[tuple[str, str, int, int]]:
    """Return ``(spacer, pam, pe3_nick_site_plus, distance)`` tuples for
    every candidate nicking guide on ``nicking_strand``.

    Parameters
    ----------
    target_genome_seq:
        + strand DNA window covering the PE2 nick + the PE3 search radius.
    nicking_strand:
        ``"+"`` or ``"-"`` — the strand on which the nicking sgRNA's
        protospacer sits. (This is the OPPOSITE of the PE2 spacer's
        strand.)
    pe2_nick_site_plus:
        PE2 nick site in + coords.
    distance_range:
        Inclusive ``range`` of allowed distances (e.g. ``range(40, 101)``).
    edit_region:
        Optional ``(start, end)`` half-open window in + coords. PE3 nicks
        landing inside this window are filtered (off-bystander rule).
    """
    plus_seq = target_genome_seq.upper()
    n = len(plus_seq)

    # Sigma is the strand the nicking sgRNA's spacer reads 5' -> 3'. For
    # nicking on '+' that's the + strand directly; for nicking on '-' it's
    # the reverse complement of +.
    if nicking_strand == "+":
        sigma_seq = plus_seq
    elif nicking_strand == "-":
        sigma_seq = _reverse_complement(plus_seq)
    else:
        raise ValueError(
            f"nicking_strand must be '+' or '-'; got {nicking_strand!r}"
        )

    out: list[tuple[str, str, int, int]] = []

    # Walk every NGG candidate on sigma. The 20-nt protospacer must sit
    # immediately 5' of the PAM, so pam_start_sigma >= 20.
    for pam_start_sigma in range(20, n - 3 + 1):
        pam = sigma_seq[pam_start_sigma : pam_start_sigma + 3]
        if not _matches_ngg(pam):
            continue
        spacer = sigma_seq[pam_start_sigma - 20 : pam_start_sigma]
        # Reject spacers containing N — we don't model degenerate bases
        # at this layer.
        if "N" in spacer or len(spacer) < 20:
            continue

        # Sigma-local nick: 3 nt 5' of the PAM start on the sigma strand.
        nick_site_sigma = pam_start_sigma - 3
        # Translate to + coords (mirrors enumerator._sigma_nick_to_plus).
        if nicking_strand == "+":
            pe3_nick_site_plus = nick_site_sigma
        else:
            pe3_nick_site_plus = n - nick_site_sigma

        # Distance from PE2 nick (absolute, in + coords).
        distance = abs(pe3_nick_site_plus - pe2_nick_site_plus)
        if distance not in distance_range:
            continue

        # Off-bystander filter: drop if the PE3 nick falls inside the
        # caller-supplied edit window.
        if edit_region is not None:
            edit_start, edit_end = edit_region
            if edit_start <= pe3_nick_site_plus < edit_end:
                continue

        out.append((spacer, pam, pe3_nick_site_plus, distance))

    return out


# --------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------- #


def select_nicking_guides(
    pe2_candidate: PegRNACandidate,
    target_genome_seq: str,
    *,
    distance_range: range = range(40, 101),
    edit_region: tuple[int, int] | None = None,
) -> list[PE3PegRNACandidate]:
    """Find PE3 nicking sgRNA candidates on the strand opposite the PE2
    spacer.

    Parameters
    ----------
    pe2_candidate:
        The PE2 pegRNA from the T6 enumerator. ``nick_site`` is read in
        + strand coords; ``strand`` selects the OPPOSITE strand for the
        nicking guide search.
    target_genome_seq:
        + strand DNA window. Must extend at least
        ``max(distance_range) + 23`` bp on each side of the PE2 nick to
        cover the full search radius (PAM start + 23 nt slack for the
        upstream protospacer).
    distance_range:
        Inclusive band of allowed distances between the PE2 nick and the
        PE3 nick (both in + coords). PRIDICT 2.0 default is 40-90 bp; we
        widen to 40-100 bp per plan §T7. Pass ``range(40, 91)`` for the
        strict PRIDICT default.
    edit_region:
        Optional ``(start, end)`` half-open window in + coords. PE3
        nicks landing inside this window are filtered (off-bystander
        rule). Default ``None`` = no off-bystander filter (keeps the
        function pure for unit testing; T8 ranker is the canonical
        caller that wires the edit window from the source EditSpec).

    Returns
    -------
    list[PE3PegRNACandidate]
        One PE3 candidate per (opposite-strand NGG, in-band, off-
        bystander-clean) hit. Sorted by
        ``nicking_distance_from_pe2_nick`` ascending. Empty when no
        opposite-strand NGG is reachable.

    Raises
    ------
    ValueError
        If ``pe2_candidate.strand`` is neither ``"+"`` nor ``"-"``, or
        ``target_genome_seq`` is not a non-empty str.
    """
    if not isinstance(target_genome_seq, str) or not target_genome_seq:
        raise ValueError("target_genome_seq must be a non-empty str")

    nicking_strand = _opposite_strand(pe2_candidate.strand)

    hits = _enumerate_nicking_guides_on_strand(
        target_genome_seq=target_genome_seq,
        nicking_strand=nicking_strand,
        pe2_nick_site_plus=pe2_candidate.nick_site,
        distance_range=distance_range,
        edit_region=edit_region,
    )

    # Build PE3PegRNACandidate records carrying all PE2 fields unchanged.
    out: list[PE3PegRNACandidate] = []
    for spacer, pam, _pe3_nick_site_plus, distance in hits:
        out.append(
            PE3PegRNACandidate(
                # ---- PE2 fields propagated unchanged ---- #
                spacer_seq=pe2_candidate.spacer_seq,
                pam_seq=pe2_candidate.pam_seq,
                scaffold_variant=pe2_candidate.scaffold_variant,
                pbs_seq=pe2_candidate.pbs_seq,
                pbs_length=pe2_candidate.pbs_length,
                rtt_seq=pe2_candidate.rtt_seq,
                rtt_length=pe2_candidate.rtt_length,
                nick_site=pe2_candidate.nick_site,
                full_pegrna_rna_seq=pe2_candidate.full_pegrna_rna_seq,
                edit_position_in_rtt=pe2_candidate.edit_position_in_rtt,
                strand=pe2_candidate.strand,
                rt_product_seq=pe2_candidate.rt_product_seq,
                chrom=pe2_candidate.chrom,
                # ---- PE3-specific fields ---- #
                nicking_spacer=spacer,
                nicking_pam=pam,
                nicking_distance_from_pe2_nick=distance,
                # strategy default = "PE3" via PE3PegRNACandidate's class
                # default; explicit set here for clarity / determinism.
                strategy="PE3",
            )
        )

    # Deterministic sort: ascending distance, then nicking_spacer for
    # tie-break stability.
    out.sort(key=lambda c: (c.nicking_distance_from_pe2_nick, c.nicking_spacer))
    return out
