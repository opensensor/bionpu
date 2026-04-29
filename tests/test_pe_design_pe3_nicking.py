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

"""Track B v0 — Tests for the PE3 nicking sgRNA selector (Task T7).

Acceptance criteria (per ``track-b-pegrna-design-plan.md`` §T7):

1. Canonical PE3 setup from Anzalone 2019 produces correct nicking guide.
2. Distance range filter (no candidates outside [40, 100]).
3. Opposite-strand check (nicking guide is on the strand NOT the PE2
   spacer's strand).
4. Off-bystander check (nicking guide doesn't accidentally cleave the
   edit site itself).

Geometry recap (matches T6 enumerator's coordinate convention):

* :class:`PegRNACandidate.nick_site` is a 0-indexed *genomic + strand*
  coordinate of the phosphodiester break, regardless of which strand the
  PE2 spacer is on.
* The PE3 nicking sgRNA must sit on the OPPOSITE strand. Its NGG PAM is
  on the opposite strand; its nick is 3 nt 5' of the PAM on the opposite
  strand and is also reported in genomic + coordinates.
* Distance = ``abs(pe3_nick_site - pe2_nick_site)`` in + coords.
"""

from __future__ import annotations

import pytest

from bionpu.genomics.pe_design.enumerator import enumerate_pe2_candidates
from bionpu.genomics.pe_design.types import EditSpec, PE3PegRNACandidate, PegRNACandidate


# ---------------------------------------------------------------------------
# Helpers shared across tests (mirrors the helpers in
# test_pe_design_enumerator.py for consistency)
# ---------------------------------------------------------------------------


_RC = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _rc(seq: str) -> str:
    return seq.translate(_RC)[::-1].upper()


def _rna(seq: str) -> str:
    return seq.upper().replace("T", "U")


def _make_substitution_spec(*, chrom: str, pos_0b: int, ref: str, alt: str) -> EditSpec:
    return EditSpec(
        chrom=chrom,
        start=pos_0b,
        end=pos_0b + len(ref),
        ref_seq=ref,
        alt_seq=alt,
        edit_type="substitution",
        notation_used=f"{ref}>{alt} at {chrom}:{pos_0b + 1}",
        strand="+",
    )


# A canonical proximal-strand PE2 candidate fixture. The Anzalone 2019
# HEK3 site is the original PE3 paper exemplar; we reuse the same 20-nt
# spacer + TGG PAM the T6 enumerator tests use.
_PROXIMAL_SPACER = "GGCCCAGACTGAGCACGTGA"
_PROXIMAL_PAM = "TGG"
_PROXIMAL_PEGRNA_FIELDS = dict(
    spacer_seq=_PROXIMAL_SPACER,
    pam_seq=_PROXIMAL_PAM,
    scaffold_variant="sgRNA_canonical",
    pbs_seq=_rna(_rc("CAGACTGAGCACG")),  # 13 nt PBS
    pbs_length=13,
    rtt_seq="UCACGUGCUCAGUC",  # placeholder; not exercised
    rtt_length=14,
    full_pegrna_rna_seq="N/A",  # not exercised by the selector
    edit_position_in_rtt=0,
    strategy="PE2",
    rt_product_seq="N/A",  # not exercised by the selector
)


def _make_pe2_candidate(
    *,
    nick_site: int,
    strand: str,
    chrom: str = "chrSyn",
    spacer_seq: str = _PROXIMAL_SPACER,
    pam_seq: str = _PROXIMAL_PAM,
) -> PegRNACandidate:
    """Construct a minimal :class:`PegRNACandidate` for selector testing.

    The selector consumes only ``nick_site``, ``strand`` and ``chrom``;
    the other fields are propagated through to the
    :class:`PE3PegRNACandidate` output unchanged, so we set them to
    plausible placeholders.
    """
    return PegRNACandidate(
        spacer_seq=spacer_seq,
        pam_seq=pam_seq,
        scaffold_variant="sgRNA_canonical",
        pbs_seq=_rna(_rc("CAGACTGAGCACG")),
        pbs_length=13,
        rtt_seq="UCACGUGCUCAGUC",
        rtt_length=14,
        nick_site=nick_site,
        full_pegrna_rna_seq="N/A",
        edit_position_in_rtt=0,
        strategy="PE2",
        strand=strand,
        rt_product_seq="N/A",
        chrom=chrom,
    )


# ---------------------------------------------------------------------------
# Test 1 — Canonical Anzalone 2019 PE3 setup produces correct nicking guide
# ---------------------------------------------------------------------------


def test_anzalone_2019_pe3_setup_produces_correct_nicking_guide():
    """Anzalone 2019 PE3 cross-check (HEK3-style).

    Build a synthetic + strand genome where:
        * The PE2 spacer "GGCCCAGACTGAGCACGTGA" + PAM "TGG" sits on +
          starting at offset 32; PE2 nick lands at + position 49.
        * Place a single ``CCN`` on + (the - strand's NGG) at a known
          offset 65 bp 3' of the PE2 nick on +, so the - strand nicking
          guide nicks roughly 65 bp away from the PE2 nick — well inside
          the canonical 40-100 bp PE3 range.
        * No other NGG / CCN elsewhere on +.

    The selector must emit at least one PE3 candidate carrying the
    expected ``nicking_spacer`` (20 nt revcomp of the +-strand window
    immediately 3' of the CCN) and ``nicking_pam`` ("TGG" — the PAM read
    on the - strand 5'→3') and a distance in [40, 100].
    """
    from bionpu.genomics.pe_design.pe3_nicking import select_nicking_guides

    # Layout on +:
    #   [0:32)              upstream pad (no NGG / no CCN)
    #   [32:52)             PE2 spacer "GGCCCAGACTGAGCACGTGA"
    #   [52:55)             PE2 PAM "TGG" (NGG on +)
    #   [55:111)            56 bp pad (no NGG / no CCN). PE2 nick = 49,
    #                        so we want the - strand nicking PAM placed
    #                        such that PE3 nick lands ~65 bp 3' of 49.
    #   [111:114)           "CCA" (= NGG on -)
    #   [114:134)           20 bp = revcomp of - strand spacer (no NGG / CCN)
    #   [134:170)           downstream pad (no NGG / no CCN)
    upstream_pad = "ATATATATATATATATATATATATATATATAT"  # 32 bp pure AT
    middle_pad = "AT" * 28  # 56 bp pure AT (positions [55:111))
    minus_spacer = "TCACGTGCTCAGTCTGGGCC"  # arbitrary, balanced GC, no NGG/CCN inside
    plus_proto_paired_for_minus = _rc(minus_spacer)  # 20 bp on +
    minus_pam_on_plus = "CCA"  # NGG on -
    downstream_pad = "AT" * 18  # 36 bp pure AT
    genome = (
        upstream_pad
        + _PROXIMAL_SPACER
        + _PROXIMAL_PAM
        + middle_pad
        + minus_pam_on_plus
        + plus_proto_paired_for_minus
        + downstream_pad
    )

    # Sanity: confirmed offsets.
    assert genome[32:52] == _PROXIMAL_SPACER
    assert genome[52:55] == _PROXIMAL_PAM
    assert genome[111:114] == minus_pam_on_plus
    assert genome[114:134] == plus_proto_paired_for_minus

    # Build the PE2 candidate: + strand, nick_site = 49.
    pe2 = _make_pe2_candidate(nick_site=49, strand="+")

    pe3_candidates = select_nicking_guides(pe2, target_genome_seq=genome)
    assert pe3_candidates, "expected at least one PE3 nicking candidate"

    # Locate the candidate whose nicking spacer is exactly minus_spacer.
    matches = [c for c in pe3_candidates if c.nicking_spacer == minus_spacer]
    assert matches, (
        f"expected nicking_spacer={minus_spacer!r} in the candidate set; "
        f"got {[c.nicking_spacer for c in pe3_candidates]}"
    )
    c = matches[0]
    assert isinstance(c, PE3PegRNACandidate)
    assert c.strategy == "PE3"
    # The nicking PAM read 5'->3' on the - strand is the revcomp of "CCA"
    # on +, which is "TGG". (The plan describes nicking_pam as the actual
    # NGG read on the nicking sgRNA's strand.)
    assert c.nicking_pam == "TGG"
    # Distance = |pe3_nick - pe2_nick| with PE3 nick reported on + (the
    # phosphodiester break between + positions 116 and 117 on -, i.e. +
    # nick_site = 117 by the same convention T6 uses).
    expected_pe3_nick_site_plus = 117  # = N - (N - 117) for - strand cut
    expected_distance = abs(expected_pe3_nick_site_plus - 49)
    assert c.nicking_distance_from_pe2_nick == expected_distance
    assert 40 <= c.nicking_distance_from_pe2_nick <= 100

    # All PE2 fields propagated unchanged.
    assert c.spacer_seq == _PROXIMAL_SPACER
    assert c.pam_seq == _PROXIMAL_PAM
    assert c.strand == "+"
    assert c.nick_site == 49
    assert c.chrom == "chrSyn"


# ---------------------------------------------------------------------------
# Test 2 — Distance range filter
# ---------------------------------------------------------------------------


def test_distance_range_filter_excludes_out_of_band_candidates():
    """Build a genome with TWO opposite-strand NGG sites flanking the
    PE2 nick: one ~20 bp away (TOO CLOSE) and one ~70 bp away (in band).
    Only the in-band candidate should appear in the output.
    """
    from bionpu.genomics.pe_design.pe3_nicking import select_nicking_guides

    # PE2 nick at + position 200. Build a long (400 bp) AT genome and
    # splice in two minus-strand PAM+spacer sites: one too close, one
    # in-band.
    genome = list("AT" * 200)  # 400 bp pure AT

    too_close_minus_spacer = "TGAGCACGTGAGCACGTGAG"  # arbitrary; pure A/G/C/T, no NGG/CCN inside
    in_band_minus_spacer = "GACGTGAGCACGTGAGCACA"  # arbitrary; same constraints

    # Place "CCA" (= NGG on -) + 20-bp revcomp such that the PE3 nick
    # (= +position right of the - strand cut) lands at +220 (distance 20
    # from PE2 nick=200) for the too-close case, and at +270 (distance 70)
    # for the in-band case.
    #
    # Recall: for - strand sigma-local nick at sigma_nick, + nick_site
    # = N - sigma_nick. The CCA at + [a, a+3) corresponds to NGG on -
    # at sigma_pam_start = N - (a + 3). The sigma nick is sigma_pam_start
    # - 3 = N - a - 6. The + nick_site = N - sigma_nick = a + 6.
    #
    # So for + nick_site = 220, place CCA at + [214, 217).
    # For + nick_site = 270, place CCA at + [264, 267).
    too_close_a = 214
    in_band_a = 264

    # Place CCA + revcomp(spacer) for too-close.
    genome[too_close_a : too_close_a + 3] = list("CCA")
    genome[too_close_a + 3 : too_close_a + 23] = list(_rc(too_close_minus_spacer))
    # Place CCA + revcomp(spacer) for in-band.
    genome[in_band_a : in_band_a + 3] = list("CCA")
    genome[in_band_a + 3 : in_band_a + 23] = list(_rc(in_band_minus_spacer))

    genome_str = "".join(genome)

    pe2 = _make_pe2_candidate(nick_site=200, strand="+")

    pe3_candidates = select_nicking_guides(
        pe2,
        target_genome_seq=genome_str,
        distance_range=range(40, 101),
    )

    # Every emitted candidate must be in [40, 100].
    distances = [c.nicking_distance_from_pe2_nick for c in pe3_candidates]
    assert all(40 <= d <= 100 for d in distances), (
        f"out-of-band distance leaked: {distances}"
    )

    # The too-close candidate (distance 20) must NOT appear.
    assert too_close_minus_spacer not in [c.nicking_spacer for c in pe3_candidates]

    # The in-band candidate (distance 70) MUST appear.
    in_band_matches = [
        c for c in pe3_candidates if c.nicking_spacer == in_band_minus_spacer
    ]
    assert in_band_matches, "in-band PE3 candidate (distance 70) was not emitted"
    assert in_band_matches[0].nicking_distance_from_pe2_nick == 70


# ---------------------------------------------------------------------------
# Test 3 — Opposite-strand check
# ---------------------------------------------------------------------------


def test_nicking_guide_is_on_opposite_strand():
    """When the PE2 spacer is on '+', the nicking sgRNA's NGG must be on
    '-' (i.e. the + strand carries 'CCN' at the nicking PAM location).
    Symmetric assertion for PE2 on '-'.
    """
    from bionpu.genomics.pe_design.pe3_nicking import select_nicking_guides

    # ----- Case A: PE2 on '+', expect nicking on '-' ----- #
    upstream_pad = "AT" * 16  # 32 bp
    middle_pad = "AT" * 28  # 56 bp
    minus_spacer = "TCACGTGCTCAGTCTGGGCC"
    minus_pam_on_plus = "CCA"
    downstream_pad = "AT" * 18  # 36 bp
    genome_a = (
        upstream_pad
        + _PROXIMAL_SPACER
        + _PROXIMAL_PAM
        + middle_pad
        + minus_pam_on_plus
        + _rc(minus_spacer)
        + downstream_pad
    )
    pe2_plus = _make_pe2_candidate(nick_site=49, strand="+")
    out_a = select_nicking_guides(pe2_plus, target_genome_seq=genome_a)
    assert out_a, "expected at least one nicking candidate (case A)"
    # The nicking guide must be on '-', so the genomic-PAM bytes on + at
    # the candidate's nick-site context must read 'CCN' (= NGG on -).
    for c in out_a:
        # The + position immediately 3' of the nicking PAM (on +) is
        # nick_site_pe3 - 3, but the cleanest invariant is: there exists
        # a 'CCN' on the + strand within reach of the candidate. Stronger
        # invariant: re-derive the +/- PAM location from nick_site_pe3 =
        # a + 6 (per Test 2's geometry); CCA must be at +[nick - 6, nick - 3).
        nick_pe3 = c.nick_site + (
            c.nicking_distance_from_pe2_nick
            if c.nick_site + c.nicking_distance_from_pe2_nick == 49 + c.nicking_distance_from_pe2_nick
            else c.nicking_distance_from_pe2_nick
        )
        # Equivalent to: pe3 nick_site = pe2.nick_site +/- distance.
        # Because we tested case A and chose the candidate to be 3' of
        # the PE2 nick: nick_pe3 = pe2.nick_site + distance.
        nick_pe3 = pe2_plus.nick_site + c.nicking_distance_from_pe2_nick
        # CC + N at + [nick_pe3 - 6, nick_pe3 - 3).
        ccn = genome_a[nick_pe3 - 6 : nick_pe3 - 3]
        assert ccn[:2] == "CC", (
            f"expected 'CCN' on + for - strand nicking PAM near "
            f"+{nick_pe3 - 6}; got {ccn!r}"
        )

    # ----- Case B: PE2 on '-', expect nicking on '+' ----- #
    # Symmetric: PE2 spacer on - means we want NGG on + for the nicking
    # guide. Build a + strand genome with an NGG on + within 40-100 bp
    # of the PE2 nick site.
    #
    # Re-use the same protospacer + PAM placement on + (which is NGG on
    # +); construct a PE2 candidate that claims to be on '-' with nick
    # site placed appropriately. The selector should ignore the 'CCN'
    # candidates on + and seek 'NGG' on +.
    plus_pam_for_nicking = "TGG"
    plus_spacer_for_nicking = "GAACGTGAGCACGTGAGCAC"
    layout_b = (
        "AT" * 60  # 120 bp upstream pad
        + plus_spacer_for_nicking  # + [120, 140)
        + plus_pam_for_nicking  # + [140, 143) NGG
        + "AT" * 60  # 120 bp downstream pad
    )
    # NGG on + at +[140, 143): nick_site for that nicking guide = 137 (=
    # pam_start - 3, which is its + position because spacer is on '+').
    # We want PE2 (which is on '-') nick_site distance ~50-80 from 137.
    # Place PE2 nick at + position 80 → distance 137-80=57 (in band).
    pe2_minus = _make_pe2_candidate(nick_site=80, strand="-")
    out_b = select_nicking_guides(pe2_minus, target_genome_seq=layout_b)
    assert out_b, "expected at least one nicking candidate (case B)"
    # The nicking guide must be on '+'. So the genomic + strand must
    # contain 'NGG' at the candidate's PAM location, NOT 'CCN'.
    matches_b = [c for c in out_b if c.nicking_spacer == plus_spacer_for_nicking]
    assert matches_b, (
        f"expected + strand nicking spacer {plus_spacer_for_nicking!r}; "
        f"got {[c.nicking_spacer for c in out_b]}"
    )
    cb = matches_b[0]
    # The PAM read on the nicking sgRNA's strand is "TGG" (NGG on +).
    assert cb.nicking_pam == plus_pam_for_nicking
    # Distance = |137 - 80| = 57.
    assert cb.nicking_distance_from_pe2_nick == 57


# ---------------------------------------------------------------------------
# Test 4 — Off-bystander check (nicking nick must NOT fall AT the edit site)
# ---------------------------------------------------------------------------


def test_off_bystander_check_filters_nick_at_edit_site():
    """If the only opposite-strand NGG within [40, 100] bp would put the
    nicking nick AT the edit site itself, that candidate must be filtered
    out (PE3's whole point is to nick the OTHER strand at a position
    distinct from the edit window).

    We model this by constructing a PE2 candidate with a known edit
    region on '+' AND an opposite-strand nicking guide whose PE3 nick
    lands inside the edit region. The selector's off-bystander filter
    must drop it.

    Implementation contract: ``select_nicking_guides`` accepts an
    optional ``edit_region`` parameter (half-open ``(start, end)``) that
    specifies the edit window in + coords; nick sites within that
    window are filtered. When ``edit_region`` is None (the default),
    no off-bystander filtering occurs (keeps the function pure and
    composable; T8 ranker can wire the edit_region from the EditSpec).
    """
    from bionpu.genomics.pe_design.pe3_nicking import select_nicking_guides

    # PE2 nick at + position 200. Place a single '-' strand PAM such that
    # the PE3 nick lands at + position 270 (distance 70, in band).
    genome = list("AT" * 200)  # 400 bp
    in_band_minus_spacer = "GACGTGAGCACGTGAGCACA"
    a = 264  # CCA at +[264, 267) → PE3 nick on + = 270
    genome[a : a + 3] = list("CCA")
    genome[a + 3 : a + 23] = list(_rc(in_band_minus_spacer))
    genome_str = "".join(genome)

    pe2 = _make_pe2_candidate(nick_site=200, strand="+")

    # ----- Without edit_region: candidate kept (default behavior) ----- #
    out_no_filter = select_nicking_guides(pe2, target_genome_seq=genome_str)
    assert any(
        c.nicking_spacer == in_band_minus_spacer for c in out_no_filter
    ), "in-band candidate should be present without off-bystander filter"

    # ----- With edit_region overlapping PE3 nick: candidate dropped ---- #
    # Edit region [268, 273) in + coords brackets the PE3 nick site (270).
    out_filtered = select_nicking_guides(
        pe2,
        target_genome_seq=genome_str,
        edit_region=(268, 273),
    )
    assert all(
        c.nicking_spacer != in_band_minus_spacer for c in out_filtered
    ), (
        "off-bystander candidate (PE3 nick at edit site) MUST be dropped "
        "when edit_region is supplied"
    )

    # ----- With edit_region elsewhere: candidate retained ----- #
    out_safe_region = select_nicking_guides(
        pe2,
        target_genome_seq=genome_str,
        edit_region=(195, 205),  # the PE2 edit window itself, far from 270
    )
    assert any(
        c.nicking_spacer == in_band_minus_spacer for c in out_safe_region
    ), (
        "in-band candidate should be retained when edit_region does NOT "
        "overlap its PE3 nick site"
    )
