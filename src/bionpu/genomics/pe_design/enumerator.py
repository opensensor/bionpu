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

"""Track B v0 — PE2 pegRNA enumerator (Task T6 of
``track-b-pegrna-design-plan.md``).

This module enumerates all valid PE2 ``(spacer, PBS, RTT, scaffold)``
combinations on **both strands** for a given :class:`EditSpec`. The
result is the input to T7 (PE3 nicking selector) and T8 (composite
ranker).

Algorithmic geometry (PE2; Anzalone 2019, Nature)
-------------------------------------------------

Cas9-H840A nicks the **protospacer strand** (the strand the spacer
matches; the strand bearing the PAM) 3 nt 5' of the PAM. Counting from
the spacer's 5' end on its own strand, the nick is between protospacer
positions 17 and 18 (1-based). For our convention the *nick site* is the
0-indexed position immediately 3' of the cut on the spacer's strand —
i.e. ``nick_site_sigma = pam_start_sigma - 3`` where ``pam_start_sigma``
is the 0-indexed start of the PAM on the spacer's strand.

The PBS in the pegRNA RNA anneals to the nicked-strand 3' OH end. That
end is the 5' fragment of the cut; the 8-15 bp 5' of the nick on the
spacer's strand gets reverse-complemented (and ``T -> U``) to produce
the PBS.

The RTT in the pegRNA RNA is reverse-transcribed onto the nicked 3' OH
end. It encodes the desired edit + flanking. We compute it as the
reverse-complement (and ``T -> U``) of the *post-edit* spacer-strand
sequence over ``[nick_site_sigma, nick_site_sigma + rtt_length)``.

Canonical pegRNA assembly (5' to 3'):

    spacer_RNA  +  scaffold  +  RTT_RNA  +  PBS_RNA

Both strands
-------------

PE2 routinely uses opposite-strand spacers when the proximal strand
lacks a PAM near the edit (per plan §T6 reviewer pruning rule #1).
We enumerate spacers on both ``+`` and ``-`` of the supplied target
sequence; for the ``-`` strand we work in spacer-strand-local coords
on the reverse complement of the ``+`` strand window, and translate
the nick site back to a ``+`` strand genomic coordinate for sorting +
output.

Pruning rules (§T6)
-------------------

1. **Pol III termination**: drop candidates whose full pegRNA RNA
   contains ``UUUU`` (DNA: ``TTTT``) — that's a Pol III terminator.
2. **GC band**: drop candidates whose full pegRNA RNA GC% is outside
   ``[25, 75]``.
3. **PAM-recreation**: drop candidates whose post-edit spacer-strand
   sequence still presents an ``NGG`` PAM at the original PAM location
   (Cas9 would re-cut after editing).
4. **RTT length floor**: drop candidates with ``rtt_length <
   edit_size + 5`` (the RTT must extend past the edit by ≥5 nt for
   stable RT extension).

Sort
----

The output is sorted by
``(chrom, nick_site, strand, pbs_length, rtt_length, scaffold_variant)``
ascending. T13 asserts byte-identical output across runs.
"""

from __future__ import annotations

from typing import Iterable

from bionpu.genomics.pe_design.pegrna_constants import (
    PBS_LENGTH_MAX,
    PBS_LENGTH_MIN,
    RTT_LENGTH_MAX,
    RTT_LENGTH_MIN,
    SCAFFOLD_VARIANTS,
)
from bionpu.genomics.pe_design.types import EditSpec, PegRNACandidate

__all__ = ["enumerate_pe2_candidates"]


# --------------------------------------------------------------------- #
# Sequence utilities (kept local; mirror the simple translate-table
# convention used throughout `bionpu.genomics`).
# --------------------------------------------------------------------- #


_RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _reverse_complement(seq: str) -> str:
    """Return the reverse complement of an A/C/G/T/N sequence (uppercased)."""
    return seq.translate(_RC_TABLE)[::-1].upper()


def _to_rna(seq: str) -> str:
    """DNA -> RNA: uppercase + T -> U."""
    return seq.upper().replace("T", "U")


def _gc_fraction_rna(rna: str) -> float:
    """GC fraction of an RNA sequence in [0, 1]; empty -> 0."""
    if not rna:
        return 0.0
    gc = sum(1 for c in rna if c in "GC")
    return gc / len(rna)


# --------------------------------------------------------------------- #
# PAM matching
# --------------------------------------------------------------------- #


def _matches_ngg(pam: str) -> bool:
    """Return True iff ``pam`` is a 3-mer matching the IUPAC template
    ``NGG`` (any base + GG). Reject ``N`` in the data — we want a real
    DNA base in the first position."""
    if len(pam) != 3:
        return False
    if pam[0] not in "ACGT":
        return False
    return pam[1] == "G" and pam[2] == "G"


# --------------------------------------------------------------------- #
# Edit application on a sigma-strand (spacer-strand) sequence
# --------------------------------------------------------------------- #


def _spec_to_sigma_local(
    edit_spec: EditSpec,
    *,
    strand: str,
    target_genome_seq_len: int,
) -> tuple[int, int, str, str]:
    """Translate the EditSpec from + strand coords to sigma-strand-local
    coords, complementing alleles for the - strand.

    Returns
    -------
    (sigma_start, sigma_end_excl, ref_sigma, alt_sigma)
        Coordinates and alleles in the sigma-strand-local frame
        (sigma_start is 0-indexed inclusive; sigma_end_excl is
        0-indexed exclusive). For + strand these are identical to the
        EditSpec; for - strand they're reflected onto the reverse
        complement.
    """
    if strand == "+":
        return (
            edit_spec.start,
            edit_spec.end,
            edit_spec.ref_seq.upper(),
            edit_spec.alt_seq.upper(),
        )
    # - strand: reflect coords onto the reverse complement of the +
    # strand window. If the + range is [s, e) on a length-N + strand,
    # the - strand range is [N - e, N - s).
    n = target_genome_seq_len
    sigma_start = n - edit_spec.end
    sigma_end_excl = n - edit_spec.start
    ref_sigma = _reverse_complement(edit_spec.ref_seq) if edit_spec.ref_seq else ""
    alt_sigma = _reverse_complement(edit_spec.alt_seq) if edit_spec.alt_seq else ""
    return sigma_start, sigma_end_excl, ref_sigma, alt_sigma


def _apply_edit_sigma(
    sigma_seq: str,
    *,
    sigma_start: int,
    sigma_end_excl: int,
    alt_sigma: str,
) -> str:
    """Return ``sigma_seq`` with bases at ``[sigma_start, sigma_end_excl)``
    replaced by ``alt_sigma``. Used for computing the post-edit spacer
    strand sequence the RTT encodes.
    """
    return sigma_seq[:sigma_start] + alt_sigma + sigma_seq[sigma_end_excl:]


# --------------------------------------------------------------------- #
# Coordinate translation: sigma-local nick -> genomic + nick_site
# --------------------------------------------------------------------- #


def _sigma_nick_to_plus(
    nick_site_sigma: int,
    *,
    strand: str,
    target_genome_seq_len: int,
) -> int:
    """Translate a sigma-strand-local 0-indexed nick site to a genomic
    + strand 0-indexed coordinate.

    For ``strand == "+"`` this is the identity. For ``strand == "-"``
    we reflect: a position p on the - strand of a length-N + strand
    corresponds to + position (N - p). The "nick site" on - is the
    0-index of the base immediately 3' of the cut on -; that is the
    same physical location as + position ``N - nick_site_sigma`` on +.
    """
    if strand == "+":
        return nick_site_sigma
    return target_genome_seq_len - nick_site_sigma


# --------------------------------------------------------------------- #
# Edit-size for the rtt-floor pruning rule
# --------------------------------------------------------------------- #


def _edit_size(edit_spec: EditSpec) -> int:
    """Per-§T6 pruning rule 4: rtt_length must be >= edit_size + 5.

    For substitutions, edit_size = max(len(ref), len(alt)).
    For insertions, edit_size = len(alt).
    For deletions, edit_size = len(ref).
    """
    if edit_spec.edit_type == "insertion":
        return len(edit_spec.alt_seq)
    if edit_spec.edit_type == "deletion":
        return len(edit_spec.ref_seq)
    return max(len(edit_spec.ref_seq), len(edit_spec.alt_seq))


# --------------------------------------------------------------------- #
# Per-strand enumeration
# --------------------------------------------------------------------- #


def _enumerate_one_strand(
    *,
    edit_spec: EditSpec,
    target_genome_seq: str,
    sigma_seq: str,
    strand: str,
    scaffold_variant: str,
    scaffold_rna: str,
    pbs_lengths: range,
    rtt_lengths: range,
    edit_floor: int,
) -> list[PegRNACandidate]:
    """Enumerate PE2 candidates whose spacer is on ``strand``.

    ``sigma_seq`` is the sigma-strand-local (= same orientation as the
    spacer reads 5' -> 3') DNA. For ``strand == "+"`` it's
    ``target_genome_seq``; for ``strand == "-"`` it's
    ``reverse_complement(target_genome_seq)``.
    """
    out: list[PegRNACandidate] = []
    n = len(sigma_seq)

    # Sigma-local edit coords + alt allele.
    sigma_start, sigma_end_excl, _ref_sigma, alt_sigma = _spec_to_sigma_local(
        edit_spec, strand=strand, target_genome_seq_len=len(target_genome_seq)
    )

    # Apply the edit ONCE on sigma to compute the post-edit window the
    # RTT encodes. (Does not depend on the spacer / PAM / lengths.)
    edited_sigma_seq = _apply_edit_sigma(
        sigma_seq,
        sigma_start=sigma_start,
        sigma_end_excl=sigma_end_excl,
        alt_sigma=alt_sigma,
    )

    # Search radius for PAMs around the edit. The PE2 RTT spans
    # [nick, nick + RTT_LENGTH_MAX) on sigma; the edit must fall in
    # that range. The nick is at pam_start_sigma - 3 so the PAM lives
    # at [edit - RTT_LENGTH_MAX + 3 + 1, edit + 3) roughly. Use a
    # generous search window keyed to RTT_LENGTH_MAX to keep the inner
    # loops simple and scan-coupled.
    #
    # Concretely: PAM is at [pam_start_sigma, pam_start_sigma + 3); for
    # the edit to be inside the RTT window, sigma_start must lie in
    # [pam_start_sigma - 3, pam_start_sigma - 3 + RTT_LENGTH_MAX). So
    # pam_start_sigma is in
    # (sigma_start - RTT_LENGTH_MAX + 3, sigma_start + 3].
    pam_lo = max(0, sigma_start - RTT_LENGTH_MAX + 3)
    # +3 to allow the PAM to land just past the edit start (still
    # 5' of the right edge of the rtt window).
    pam_hi = min(n - 3, sigma_start + 3)
    # Spacers of length 20 sit immediately 5' of the PAM, so we also
    # require pam_start_sigma >= 20.
    pam_lo = max(pam_lo, 20)

    for pam_start_sigma in range(pam_lo, pam_hi + 1):
        pam = sigma_seq[pam_start_sigma : pam_start_sigma + 3]
        if not _matches_ngg(pam):
            continue
        spacer = sigma_seq[pam_start_sigma - 20 : pam_start_sigma]
        # Reject spacers containing N (we don't model degenerate bases).
        if "N" in spacer or len(spacer) < 20:
            continue

        nick_site_sigma = pam_start_sigma - 3

        # Edit must fall within the candidate RTT window. We check this
        # per (rtt_length) below; here we just guard the "edit entirely
        # 5' of nick" case which is invalid for ANY rtt_length.
        # Note: an insertion at exactly nick_site_sigma has
        # sigma_start == sigma_end_excl == nick_site_sigma; that's
        # valid (the inserted bases land at the 5'-most RTT position).
        if (
            sigma_end_excl < nick_site_sigma
            or (
                sigma_end_excl == nick_site_sigma
                and not (
                    edit_spec.edit_type == "insertion"
                    and sigma_start == nick_site_sigma
                )
            )
        ):
            continue

        # PAM-recreation check (rule #3, plan §T6 reviewer pruning #2):
        # if the edit OVERLAPS the PAM bases AND the post-edit PAM still
        # matches NGG, Cas9 will re-cut after editing -- drop the
        # candidate. We don't prune when the edit is entirely outside
        # the PAM region: that's the standard PE2 limitation (re-cutting
        # if no PAM-disrupting silent mutation is included), not the
        # "recreation" failure mode the plan calls out.
        #
        # The original PAM occupies sigma_seq[pam_start_sigma : pam_start_sigma + 3].
        # The post-edit PAM occupies edited_sigma_seq[pam_start_sigma : pam_start_sigma + 3]
        # (positions are unchanged because the nick lies 5' of the PAM
        # and we already require the edit to be at-or-3'-of-the-nick).
        edit_overlaps_pam = (
            sigma_start < pam_start_sigma + 3
            and sigma_end_excl > pam_start_sigma
        ) or (
            # Insertion at a PAM-internal position: half-open range may
            # be zero-width and must still count as overlapping.
            edit_spec.edit_type == "insertion"
            and pam_start_sigma <= sigma_start < pam_start_sigma + 3
        )
        if edit_overlaps_pam:
            post_edit_pam = edited_sigma_seq[
                pam_start_sigma : pam_start_sigma + 3
            ]
            if _matches_ngg(post_edit_pam):
                continue

        nick_site_plus = _sigma_nick_to_plus(
            nick_site_sigma,
            strand=strand,
            target_genome_seq_len=len(target_genome_seq),
        )

        for pbs_length in pbs_lengths:
            if pbs_length < 1:
                continue
            if nick_site_sigma - pbs_length < 0:
                # PBS would run off the 5' end of sigma_seq.
                continue
            pbs_dna = sigma_seq[
                nick_site_sigma - pbs_length : nick_site_sigma
            ]
            if "N" in pbs_dna:
                continue
            pbs_rna = _to_rna(_reverse_complement(pbs_dna))

            for rtt_length in rtt_lengths:
                if rtt_length < edit_floor:
                    continue
                # Edit must fall ENTIRELY within the RTT-encoded
                # post-edit window on sigma. The post-edit window
                # starts at nick_site_sigma; account for indel size
                # via the edited_sigma_seq directly.
                #
                # We also need rtt_length bp available on the post-edit
                # sigma starting at nick_site_sigma.
                if nick_site_sigma + rtt_length > len(edited_sigma_seq):
                    continue

                # Build rt_product_seq (post-edit sigma over rtt window).
                rt_product = edited_sigma_seq[
                    nick_site_sigma : nick_site_sigma + rtt_length
                ]
                if "N" in rt_product:
                    continue

                # Edit coverage check: the alt allele bases must lie
                # within the RTT window on the POST-edit sigma. For
                # substitutions/deletions/insertions alike, the edit
                # starts at sigma_start; require sigma_start >=
                # nick_site_sigma. For multi-bp alt, the entire alt
                # must fit in the RTT window. Equivalently:
                #   sigma_start - nick_site_sigma + len(alt_sigma) <= rtt_length
                # For pure deletions (len(alt) == 0), require
                #   sigma_start >= nick_site_sigma  (already the gate).
                if sigma_start < nick_site_sigma:
                    continue
                if (
                    sigma_start
                    - nick_site_sigma
                    + max(len(alt_sigma), 1)
                ) > rtt_length:
                    continue

                rtt_rna = _to_rna(_reverse_complement(rt_product))

                # Edit position within RTT (RNA). The RTT is
                # rev-comp(post-edit-sigma). The edit starts at
                # sigma_start in the post-edit sigma's own offset of
                # nick_site_sigma; in the RTT (which is reversed) it
                # lands at (rtt_length - 1 - (sigma_start - nick_site_sigma))
                # for the first base of the alt.
                edit_offset_in_post = sigma_start - nick_site_sigma
                edit_position_in_rtt = (
                    rtt_length - 1 - edit_offset_in_post
                )

                # Compose the full pegRNA RNA (5' -> 3'):
                #   spacer_RNA + scaffold + RTT + PBS
                spacer_rna = _to_rna(spacer)
                full_pegrna = spacer_rna + scaffold_rna + rtt_rna + pbs_rna

                # Pruning rule #1: Pol III TTTT/UUUU termination.
                # The canonical scaffold ends with the intended Pol III
                # terminator hairpin (...GCUUUU); we therefore scan ONLY
                # the user-controlled regions (spacer, RTT, PBS) and the
                # spacer/scaffold + scaffold/RTT junctions for spurious
                # poly-U runs of length 4. The scaffold's intentional
                # terminator does NOT count.
                pol3_check = (
                    spacer_rna[-3:] + scaffold_rna[:3]  # spacer/scaffold junction
                    + "|"  # delimiter so a run never spans non-junctions
                    + spacer_rna
                    + "|"
                    + rtt_rna
                    + "|"
                    + pbs_rna
                    + "|"
                    + scaffold_rna[-3:] + rtt_rna[:3]  # scaffold/RTT junction
                )
                if "UUUU" in pol3_check:
                    continue
                # Pruning rule #2: GC band [25, 75]%.
                gc_frac = _gc_fraction_rna(full_pegrna)
                if gc_frac < 0.25 or gc_frac > 0.75:
                    continue

                cand = PegRNACandidate(
                    spacer_seq=spacer,
                    pam_seq=pam,
                    scaffold_variant=scaffold_variant,
                    pbs_seq=pbs_rna,
                    pbs_length=pbs_length,
                    rtt_seq=rtt_rna,
                    rtt_length=rtt_length,
                    nick_site=nick_site_plus,
                    full_pegrna_rna_seq=full_pegrna,
                    edit_position_in_rtt=edit_position_in_rtt,
                    strategy="PE2",
                    strand=strand,
                    rt_product_seq=rt_product,
                    chrom=edit_spec.chrom,
                )
                out.append(cand)

    return out


# --------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------- #


def enumerate_pe2_candidates(
    edit_spec: EditSpec,
    *,
    target_genome_seq: str,
    scaffold_variant: str,
    pbs_lengths: range = range(PBS_LENGTH_MIN, PBS_LENGTH_MAX + 1),
    rtt_lengths: range = range(RTT_LENGTH_MIN, RTT_LENGTH_MAX + 1),
) -> list[PegRNACandidate]:
    """Enumerate all valid PE2 pegRNA candidates for ``edit_spec``.

    Parameters
    ----------
    edit_spec:
        Parsed edit specifier (T2). The genomic coordinates are
        + strand 0-based half-open ``[start, end)``.
    target_genome_seq:
        + strand DNA window covering the edit and its flanking. Must
        be long enough to enumerate at least one PAM within the
        reachable RTT window — typically ≥80 bp around the edit.
    scaffold_variant:
        Name from
        :data:`bionpu.genomics.pe_design.pegrna_constants.SCAFFOLD_VARIANTS`.
        v0 supports ``"sgRNA_canonical"``, ``"evopreQ1"``, ``"tevopreQ1"``;
        ``"cr772"`` is currently unsupported (see plan §T3).
    pbs_lengths:
        Inclusive PBS-length range to enumerate. Defaults to the
        :data:`PBS_LENGTH_MIN`-:data:`PBS_LENGTH_MAX` band from
        :mod:`pegrna_constants`.
    rtt_lengths:
        Inclusive RTT-length range. Defaults to the
        :data:`RTT_LENGTH_MIN`-:data:`RTT_LENGTH_MAX` band.

    Returns
    -------
    list[PegRNACandidate]
        Sorted by
        ``(chrom, nick_site, strand, pbs_length, rtt_length,
        scaffold_variant)``. Empty when no PAM is reachable from the
        edit on either strand (or all candidates were pruned).
    """
    if scaffold_variant not in SCAFFOLD_VARIANTS:
        raise KeyError(
            f"unknown scaffold_variant {scaffold_variant!r}; supported: "
            f"{sorted(SCAFFOLD_VARIANTS)}"
        )
    scaffold_rna = SCAFFOLD_VARIANTS[scaffold_variant]
    if scaffold_rna is None:
        raise ValueError(
            f"scaffold_variant {scaffold_variant!r} is None — its "
            f"canonical sequence has not been pinned in v0 (see "
            f"track-b-pegrna-design-plan.md §T3 for the v1 TODO)."
        )

    # Validate types + non-empty target sequence + sane edit spec.
    if not isinstance(target_genome_seq, str) or not target_genome_seq:
        raise ValueError("target_genome_seq must be a non-empty str")
    if edit_spec.start < 0 or edit_spec.end > len(target_genome_seq):
        raise ValueError(
            f"edit coords {edit_spec.start}..{edit_spec.end} out of "
            f"range for target_genome_seq of length "
            f"{len(target_genome_seq)}"
        )

    plus_seq = target_genome_seq.upper()
    minus_seq = _reverse_complement(plus_seq)

    edit_floor = _edit_size(edit_spec) + 5

    # Plus strand
    cands: list[PegRNACandidate] = []
    cands.extend(
        _enumerate_one_strand(
            edit_spec=edit_spec,
            target_genome_seq=plus_seq,
            sigma_seq=plus_seq,
            strand="+",
            scaffold_variant=scaffold_variant,
            scaffold_rna=scaffold_rna,
            pbs_lengths=pbs_lengths,
            rtt_lengths=rtt_lengths,
            edit_floor=edit_floor,
        )
    )
    # Minus strand
    cands.extend(
        _enumerate_one_strand(
            edit_spec=edit_spec,
            target_genome_seq=plus_seq,
            sigma_seq=minus_seq,
            strand="-",
            scaffold_variant=scaffold_variant,
            scaffold_rna=scaffold_rna,
            pbs_lengths=pbs_lengths,
            rtt_lengths=rtt_lengths,
            edit_floor=edit_floor,
        )
    )

    # Deterministic sort per §T6.
    cands.sort(
        key=lambda c: (
            c.chrom,
            c.nick_site,
            c.strand,
            c.pbs_length,
            c.rtt_length,
            c.scaffold_variant,
        )
    )
    return cands
