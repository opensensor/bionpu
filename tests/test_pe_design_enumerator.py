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

"""Track B v0 — Tests for the PE2 pegRNA enumerator (Task T6).

Acceptance criteria (per ``track-b-pegrna-design-plan.md`` §T6):

1. Single-base substitution near a known PAM produces a valid candidate.
2. Deletion produces correct RTT.
3. Insertion handled.
4. Edit too far from any PAM produces an empty list.
5. pegRNA RT product matches expected edited sequence byte-for-byte.
6. Opposite-strand spacer is enumerated when proximal-strand lacks a PAM.
7. PAM-recreation candidate is pruned.

Plus the cross-check (Anzalone 2019 HEK3-style) and the determinism
(sort-stability across 5 runs) checks the plan calls out.
"""

from __future__ import annotations

import pytest

from bionpu.genomics.pe_design.types import EditSpec


# ---------------------------------------------------------------------------
# Helpers shared across tests
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


def _make_deletion_spec(*, chrom: str, start_0b: int, ref: str) -> EditSpec:
    return EditSpec(
        chrom=chrom,
        start=start_0b,
        end=start_0b + len(ref),
        ref_seq=ref,
        alt_seq="",
        edit_type="deletion",
        notation_used=f"del {chrom}:{start_0b + 1}..{start_0b + len(ref)}",
        strand="+",
    )


def _make_insertion_spec(*, chrom: str, before_pos_0b: int, alt: str) -> EditSpec:
    return EditSpec(
        chrom=chrom,
        start=before_pos_0b,
        end=before_pos_0b,
        ref_seq="",
        alt_seq=alt,
        edit_type="insertion",
        notation_used=f"ins{alt} at {chrom}:{before_pos_0b + 1}",
        strand="+",
    )


def _build_plus_strand_genome_with_spacer(
    *,
    spacer: str,
    pam: str,
    upstream_pad: str,
    downstream_pad: str,
) -> str:
    """Build a synthetic + strand genome with a single PAM-bearing
    protospacer at a known offset.

    Layout (left to right on +): upstream_pad + spacer + pam + downstream_pad.
    The protospacer occupies indices [len(upstream_pad), len(upstream_pad)+20)
    and the PAM occupies [len(upstream_pad)+20, len(upstream_pad)+23).
    """
    return upstream_pad + spacer + pam + downstream_pad


# Concrete fixtures used across tests --------------------------------- #
#
# We pick a 20-nt spacer with balanced GC, a known PAM "TGG" (matches NGG),
# and place it inside ~80 nt of flanking context. The flanking is chosen
# so it does NOT contain stray NGG within ±30 nt of the edit (otherwise
# the enumerator would emit additional spurious candidates beyond the
# one(s) we're asserting against).
_PROXIMAL_SPACER = "GGCCCAGACTGAGCACGTGA"  # Anzalone 2019 HEK3-style spacer
_PROXIMAL_PAM = "TGG"
# Pick flanking that has no NGG and no CCN (so the proximal sequence
# admits ONE PAM on + only, not on -, and the flanks contribute none).
_UPSTREAM = "ATATATATATATATATATATATATATATATAT"  # 32 bp pure AT
_DOWNSTREAM = "ATATATATATATATATATATATATATATATAT"  # 32 bp pure AT


# ---------------------------------------------------------------------------
# Test 1 — single-base substitution near a known PAM
# ---------------------------------------------------------------------------


def test_substitution_near_pam_produces_valid_candidate():
    """A G>T substitution 1 bp downstream of the canonical nick site
    must produce at least one PE2 candidate with the expected spacer +
    pam, and the RTT must encode the ALT base at the right offset.
    """
    from bionpu.genomics.pe_design.enumerator import enumerate_pe2_candidates

    genome = _build_plus_strand_genome_with_spacer(
        spacer=_PROXIMAL_SPACER,
        pam=_PROXIMAL_PAM,
        upstream_pad=_UPSTREAM,
        downstream_pad=_DOWNSTREAM,
    )
    # nick site (+ strand) = upstream + 17 = 32 + 17 = 49.
    # Edit lives 1 bp 3' of nick = position 50 (0-based). The genomic
    # base at position 50 is the 18th base of the protospacer (= 'C' for
    # GGCCCAGACTGAGCACGTGA, offset 17? Let's compute: position 17 of
    # "GGCCCAGACTGAGCACGTGA" is 'T'. Use the actual base from the
    # spacer.)
    # We choose a clean substitution: change whatever base sits at
    # position 50 to a base that does NOT produce an NGG at the PAM
    # site post-edit (so PAM-recreation pruning does not kick in).
    edit_pos = 32 + 17  # nick site exactly; choose the first RTT base
    ref_base = genome[edit_pos]
    # Pick alt != ref so this is a real substitution.
    alt_base = "A" if ref_base != "A" else "T"

    spec = _make_substitution_spec(
        chrom="chrSyn",
        pos_0b=edit_pos,
        ref=ref_base,
        alt=alt_base,
    )

    candidates = enumerate_pe2_candidates(
        spec,
        target_genome_seq=genome,
        scaffold_variant="sgRNA_canonical",
    )

    assert candidates, "expected at least one PE2 candidate near the proximal PAM"
    # All emitted candidates must be PE2 and on the strand the spacer
    # comes from (here, +).
    # The canonical proximal spacer + PAM combo we constructed:
    plus_hits = [
        c
        for c in candidates
        if c.strand == "+" and c.spacer_seq == _PROXIMAL_SPACER
    ]
    assert plus_hits, (
        f"expected at least one + strand candidate with the canonical "
        f"spacer; got strands={[c.strand for c in candidates]}, "
        f"spacers={[c.spacer_seq for c in candidates][:3]}"
    )

    c = plus_hits[0]
    assert c.strategy == "PE2"
    assert c.spacer_seq == _PROXIMAL_SPACER
    assert c.pam_seq == _PROXIMAL_PAM
    assert c.scaffold_variant == "sgRNA_canonical"
    assert c.nick_site == 49
    assert c.chrom == "chrSyn"

    # RTT must encode the alt base at the appropriate offset within the
    # post-edit sequence; the rt_product_seq starts at the nick site on
    # +, so rt_product_seq[0] == alt_base for a substitution AT the nick.
    assert c.rt_product_seq[0] == alt_base, (
        f"rt_product_seq[0]={c.rt_product_seq[0]!r} but expected ALT "
        f"{alt_base!r} (RTT must encode the edit at offset 0)"
    )

    # Composite pegRNA: spacer (RNA) + scaffold + RTT + PBS, in that order.
    assert c.full_pegrna_rna_seq.startswith(_rna(_PROXIMAL_SPACER))
    # PBS is the suffix of the pegRNA RNA.
    assert c.full_pegrna_rna_seq.endswith(c.pbs_seq)
    # PBS reverse-complements the 8-15 nt of + strand 5' of the nick.
    pbs_window_dna = genome[c.nick_site - c.pbs_length : c.nick_site]
    assert c.pbs_seq == _rna(_rc(pbs_window_dna))


# ---------------------------------------------------------------------------
# Test 2 — deletion produces correct RTT
# ---------------------------------------------------------------------------


def test_deletion_produces_correct_rtt():
    """A 3-bp deletion within the RTT window must emit a candidate
    whose rt_product_seq omits the deleted bases byte-for-byte vs the
    pre-edit + strand."""
    from bionpu.genomics.pe_design.enumerator import enumerate_pe2_candidates

    genome = _build_plus_strand_genome_with_spacer(
        spacer=_PROXIMAL_SPACER,
        pam=_PROXIMAL_PAM,
        upstream_pad=_UPSTREAM,
        downstream_pad=_DOWNSTREAM,
    )
    # Delete 3 bp starting 5 bp 3' of the nick.
    nick = 32 + 17
    del_start = nick + 5
    ref = genome[del_start : del_start + 3]
    spec = _make_deletion_spec(chrom="chrSyn", start_0b=del_start, ref=ref)

    candidates = enumerate_pe2_candidates(
        spec,
        target_genome_seq=genome,
        scaffold_variant="sgRNA_canonical",
    )
    # At least one candidate with rtt long enough (edit_size + 5 = 8) must exist.
    plus_hits = [c for c in candidates if c.strand == "+"]
    assert plus_hits

    # Pick a candidate with rtt_length >= 10 (default range starts at 10).
    c = next(c for c in plus_hits if c.rtt_length >= 10)

    # rt_product_seq = + strand at [nick, nick + rtt_length_post_edit)
    # but with the 3-bp deletion applied. So byte-equal expectation:
    # take + strand pre-edit window, splice out the deleted 3 bp, the
    # remaining slice must equal rt_product_seq.
    pre_window = genome[c.nick_site : c.nick_site + c.rtt_length + 3]
    # Apply deletion at del_start within that window.
    rel_del = del_start - c.nick_site
    edited = pre_window[:rel_del] + pre_window[rel_del + 3 :]
    expected = edited[: c.rtt_length]
    assert c.rt_product_seq == expected, (
        f"rt_product_seq mismatch for deletion: got {c.rt_product_seq!r}, "
        f"expected {expected!r}"
    )


# ---------------------------------------------------------------------------
# Test 3 — insertion handled
# ---------------------------------------------------------------------------


def test_insertion_handled():
    """A 2-bp insertion just 3' of the nick must be encoded by the
    RTT; the rt_product_seq must contain the inserted bases at the
    expected offset."""
    from bionpu.genomics.pe_design.enumerator import enumerate_pe2_candidates

    genome = _build_plus_strand_genome_with_spacer(
        spacer=_PROXIMAL_SPACER,
        pam=_PROXIMAL_PAM,
        upstream_pad=_UPSTREAM,
        downstream_pad=_DOWNSTREAM,
    )
    nick = 32 + 17
    ins_pos = nick + 4
    spec = _make_insertion_spec(chrom="chrSyn", before_pos_0b=ins_pos, alt="GA")

    candidates = enumerate_pe2_candidates(
        spec,
        target_genome_seq=genome,
        scaffold_variant="sgRNA_canonical",
    )
    plus_hits = [c for c in candidates if c.strand == "+"]
    assert plus_hits

    c = plus_hits[0]
    # rt_product_seq must contain the inserted bases at offset 4 within
    # the post-edit window.
    rel = ins_pos - c.nick_site
    assert c.rt_product_seq[rel : rel + 2] == "GA", (
        f"insertion not encoded at expected offset; got "
        f"{c.rt_product_seq[rel : rel + 2]!r} at rel={rel}"
    )


# ---------------------------------------------------------------------------
# Test 4 — edit too far from any PAM produces empty list
# ---------------------------------------------------------------------------


def test_edit_far_from_pam_returns_empty():
    """An edit position with no NGG PAM (proximal OR opposite strand)
    within the reachable RTT window on EITHER strand must return an
    empty list."""
    from bionpu.genomics.pe_design.enumerator import enumerate_pe2_candidates

    # Build a genome with a long pure-AT region and a single edit position
    # in the middle. There are no NGG PAMs anywhere on either strand
    # (AT only -> no G on +, no C on - so no NGG / no CCN).
    genome = "AT" * 200  # 400 bp
    edit_pos = 200
    spec = _make_substitution_spec(
        chrom="chrAT",
        pos_0b=edit_pos,
        ref=genome[edit_pos],
        alt="C",  # different base
    )
    candidates = enumerate_pe2_candidates(
        spec,
        target_genome_seq=genome,
        scaffold_variant="sgRNA_canonical",
    )
    assert candidates == [], (
        f"expected zero candidates in pure-AT genome with no PAMs, "
        f"got {len(candidates)}"
    )


# ---------------------------------------------------------------------------
# Test 5 — RT product byte-equal vs expected edited sequence
# ---------------------------------------------------------------------------


def test_rt_product_byte_equal_expected_edited_sequence():
    """For a known substitution, rt_product_seq must byte-match the
    expected post-edit + strand window starting at the nick site."""
    from bionpu.genomics.pe_design.enumerator import enumerate_pe2_candidates

    genome = _build_plus_strand_genome_with_spacer(
        spacer=_PROXIMAL_SPACER,
        pam=_PROXIMAL_PAM,
        upstream_pad=_UPSTREAM,
        downstream_pad=_DOWNSTREAM,
    )
    nick = 32 + 17
    edit_pos = nick + 7  # well within the RTT window
    ref_base = genome[edit_pos]
    alt_base = "C" if ref_base != "C" else "A"

    spec = _make_substitution_spec(
        chrom="chrSyn",
        pos_0b=edit_pos,
        ref=ref_base,
        alt=alt_base,
    )

    candidates = enumerate_pe2_candidates(
        spec,
        target_genome_seq=genome,
        scaffold_variant="sgRNA_canonical",
    )
    plus_hits = [c for c in candidates if c.strand == "+"]
    assert plus_hits

    for c in plus_hits:
        # Manually compute the expected post-edit + strand window starting
        # at nick over rtt_length.
        pre = genome[c.nick_site : c.nick_site + c.rtt_length]
        rel = edit_pos - c.nick_site
        if rel < 0 or rel >= c.rtt_length:
            # Edit isn't covered by this RTT -- skip.
            continue
        expected = pre[:rel] + alt_base + pre[rel + 1 :]
        expected = expected[: c.rtt_length]
        assert c.rt_product_seq == expected, (
            f"rt_product_seq mismatch at rtt_len={c.rtt_length}: got "
            f"{c.rt_product_seq!r} vs expected {expected!r}"
        )


# ---------------------------------------------------------------------------
# Test 6 — opposite-strand spacer is enumerated when proximal lacks PAM
# ---------------------------------------------------------------------------


def test_opposite_strand_spacer_enumerated_when_proximal_lacks_pam():
    """Construct a genome where + has NO NGG within reachable distance
    of the edit but - DOES (CCN on +). Enumerator must still emit at
    least one candidate, with strand='-'.
    """
    from bionpu.genomics.pe_design.enumerator import enumerate_pe2_candidates

    # On + strand, place a "CCN" (which is the - strand's "NGG") near the
    # edit position. Use a spacer-shaped + strand with the protospacer
    # on - encoded as revcomp.
    minus_spacer = "TCACGTGCTCAGTCTGGGCC"  # arbitrary, GC-balanced
    # On +, the protospacer-paired region is revcomp(spacer) =
    # revcomp("TCACGTGCTCAGTCTGGGCC") = "GGCCCAGACTGAGCACGTGA"
    # The PAM on - is NGG, so on + we need CCN immediately 5' of the
    # protospacer-paired region.
    plus_proto_paired = _rc(minus_spacer)  # "GGCCCAGACTGAGCACGTGA"
    plus_pam_for_minus = "CCA"  # this is "TGG" reverse-complemented (NGG on -)
    # Layout on +:
    #   upstream_pad (no NGG, no CCN) + plus_pam_for_minus +
    #   plus_proto_paired + downstream_pad (no NGG, no CCN)
    upstream = "ATATATATATATATATATATATATATATATAT"  # 32 bp pure AT
    downstream = "ATATATATATATATATATATATATATATATAT"  # 32 bp pure AT
    genome = upstream + plus_pam_for_minus + plus_proto_paired + downstream

    # Edit position must lie within the - strand RTT window. On -,
    # the RTT extends 3' from the nick, which on + corresponds to the
    # 5' direction below the nick. With CCA at + [32, 35) (= NGG on -)
    # and the protospacer-paired region at + [35, 55), the nick on -
    # corresponds to the + cut between + 37 and + 38 (i.e. nick_site
    # reported on + = 38). The RTT-on-- window on + spans + positions
    # [38 - rtt_len, 38). So pick the edit at + position 37 (= the
    # last base 5' of the cut on +, = the first RTT base on -).
    edit_pos = 37
    ref_base = genome[edit_pos]
    # Pick alt with care: avoid recreating CCN at the - PAM region.
    alt_base = "A" if ref_base != "A" else "T"

    spec = _make_substitution_spec(
        chrom="chrSyn",
        pos_0b=edit_pos,
        ref=ref_base,
        alt=alt_base,
    )

    candidates = enumerate_pe2_candidates(
        spec,
        target_genome_seq=genome,
        scaffold_variant="sgRNA_canonical",
    )
    minus_hits = [c for c in candidates if c.strand == "-"]
    assert minus_hits, (
        f"expected at least one - strand candidate when proximal + has "
        f"no NGG; got candidates={[c.strand for c in candidates]}"
    )
    c = minus_hits[0]
    assert c.strategy == "PE2"
    assert c.spacer_seq == minus_spacer
    # PAM on - is the revcomp of "CCA" on + = "TGG" (matches NGG).
    assert c.pam_seq == "TGG"
    # nick_site (genomic + coord of the phosphodiester break) = 38.
    assert c.nick_site == 38


# ---------------------------------------------------------------------------
# Test 7 — PAM-recreation candidate is pruned
# ---------------------------------------------------------------------------


def test_pam_recreation_is_pruned():
    """If the edited RTT recreates the spacer's NGG PAM, the candidate
    must be DROPPED (Cas9 would re-cut after editing). Set up a
    substitution that lands on the PAM bases themselves and write the
    same NGG back.
    """
    from bionpu.genomics.pe_design.enumerator import enumerate_pe2_candidates

    genome = _build_plus_strand_genome_with_spacer(
        spacer=_PROXIMAL_SPACER,
        pam="AGG",  # NGG; first PAM base is A
        upstream_pad=_UPSTREAM,
        downstream_pad=_DOWNSTREAM,
    )
    # Edit: change the first PAM base from A to T (still NGG = TGG).
    # PAM lives at [52, 55) on +. nick at 49.
    pam_first_pos = 32 + 20  # 52
    spec = _make_substitution_spec(
        chrom="chrSyn",
        pos_0b=pam_first_pos,
        ref="A",
        alt="T",  # T != A, but TGG is still NGG -> PAM recreated
    )

    candidates = enumerate_pe2_candidates(
        spec,
        target_genome_seq=genome,
        scaffold_variant="sgRNA_canonical",
    )
    # The proximal + strand candidate must be pruned because the post-
    # edit PAM is still NGG.
    plus_hits = [
        c
        for c in candidates
        if c.strand == "+" and c.spacer_seq == _PROXIMAL_SPACER
    ]
    assert plus_hits == [], (
        f"PAM-recreation candidate must be pruned; got "
        f"{[c.spacer_seq for c in plus_hits]} for + strand "
        f"{_PROXIMAL_SPACER}"
    )

    # Sanity contrast: the SAME setup but the alt base disrupts the PAM
    # (e.g. A->C makes "CGG" which is also still NGG, hmm... pick A->A
    # we cannot, so use an alt that breaks NGG: change second PAM base).
    spec_break = _make_substitution_spec(
        chrom="chrSyn",
        pos_0b=pam_first_pos + 1,  # position 53, the first G of AGG
        ref="G",
        alt="A",  # AGG -> AAG: NOT NGG -> PAM broken
    )
    candidates_break = enumerate_pe2_candidates(
        spec_break,
        target_genome_seq=genome,
        scaffold_variant="sgRNA_canonical",
    )
    plus_break = [
        c
        for c in candidates_break
        if c.strand == "+" and c.spacer_seq == _PROXIMAL_SPACER
    ]
    assert plus_break, (
        "with PAM disrupted (AGG -> AAG) the proximal + candidate "
        "should NOT be pruned by PAM-recreation"
    )


# ---------------------------------------------------------------------------
# Cross-check (Anzalone 2019 HEK3-style): spacer + PBS + RTT byte-equal
# ---------------------------------------------------------------------------


def test_anzalone_2019_hek3_spacer_pbs_rtt_byte_equal():
    """Anzalone 2019 HEK3 +1 G->T pegRNA cross-check.

    The HEK3 protospacer + PAM as published is:
        GGCCCAGACTGAGCACGTGA TGG
    Nick site is between protospacer pos 17 and 18 (0-indexed: between
    16 and 17 from the spacer 5' end). For a +1 insertion of T (the
    canonical HEK3 +1 ins T pegRNA in Anzalone Fig 3), the canonical
    13-nt PBS is:
        CGUGCUCAGUCUG  (RNA; revcomp of CAGACTGAGCACG, the 13 bp 5' of
        the nick on the protospacer strand)
    The RTT (13 nt as published) for a +1 ins T is:
        revcomp(post-edit + strand starting at nick over 13 bp), where
        the +1 insertion adds a T at the nick.
    We verify the enumerator emits a candidate with this exact spacer,
    pbs (at pbs_length=13) and rtt that matches the byte-equal expected
    construction.
    """
    from bionpu.genomics.pe_design.enumerator import enumerate_pe2_candidates

    genome = _build_plus_strand_genome_with_spacer(
        spacer=_PROXIMAL_SPACER,
        pam=_PROXIMAL_PAM,  # "TGG"
        upstream_pad=_UPSTREAM,
        downstream_pad=_DOWNSTREAM,
    )
    nick = 32 + 17  # 49
    # Canonical HEK3 +1 ins T: insert "T" right at the nick (= before
    # position nick on +). EditSpec for an insertion at position N uses
    # start=end=N with alt_seq.
    spec = _make_insertion_spec(chrom="chrSyn", before_pos_0b=nick, alt="T")

    candidates = enumerate_pe2_candidates(
        spec,
        target_genome_seq=genome,
        scaffold_variant="sgRNA_canonical",
    )

    target = [
        c
        for c in candidates
        if c.strand == "+"
        and c.spacer_seq == _PROXIMAL_SPACER
        and c.pbs_length == 13
        and c.rtt_length == 13
    ]
    assert target, (
        f"expected the canonical HEK3 +1 ins T candidate (spacer="
        f"{_PROXIMAL_SPACER}, pbs_len=13, rtt_len=13); got "
        f"{len(candidates)} candidates total."
    )
    c = target[0]

    # Spacer (DNA) byte-equal.
    assert c.spacer_seq == "GGCCCAGACTGAGCACGTGA"
    # PAM byte-equal.
    assert c.pam_seq == "TGG"
    # PBS (RNA) byte-equal: revcomp of the 13 bp 5' of the nick on +.
    expected_pbs_dna = genome[nick - 13 : nick]  # 13 bp
    expected_pbs = _rna(_rc(expected_pbs_dna))
    # The 13 bp 5' of nick spans + positions 36..48 inclusive. With our
    # _UPSTREAM = "AT"*16 and the spacer "GGCCCAGACTGAGCACGTGA" starting
    # at position 32, positions 36..48 = spacer[4..16] = "CAGACTGAGCACG"
    # so revcomp -> "CGTGCTCAGTCTG" -> RNA "CGUGCUCAGUCUG".
    assert expected_pbs == "CGUGCUCAGUCUG", (
        f"expected_pbs computation invariant violated: {expected_pbs!r}"
    )
    assert c.pbs_seq == expected_pbs, (
        f"PBS byte-equal cross-check failed: got {c.pbs_seq!r}, "
        f"expected {expected_pbs!r}"
    )

    # RTT (RNA) byte-equal: revcomp of the post-edit + strand from nick
    # over rtt_length (13). For +1 ins T at nick, the post-edit + strand
    # at nick is 'T' followed by the original + strand from nick over
    # 12 more bp (since the RTT is fixed 13 nt long and we inserted 1
    # base, we keep 12 of the original).
    post_edit_plus = "T" + genome[nick : nick + 12]
    expected_rtt = _rna(_rc(post_edit_plus))
    assert c.rt_product_seq == post_edit_plus
    assert c.rtt_seq == expected_rtt, (
        f"RTT byte-equal cross-check failed: got {c.rtt_seq!r}, "
        f"expected {expected_rtt!r}"
    )


# ---------------------------------------------------------------------------
# Determinism: 5 runs against the same input produce byte-identical output
# ---------------------------------------------------------------------------


def test_deterministic_sort_5_runs_byte_identical():
    """Sort-stability: 5 invocations against the same input must yield
    byte-identical candidate ordering. T13 also asserts this on the
    full pipeline; we lock it at the enumerator boundary here.
    """
    from bionpu.genomics.pe_design.enumerator import enumerate_pe2_candidates

    genome = _build_plus_strand_genome_with_spacer(
        spacer=_PROXIMAL_SPACER,
        pam=_PROXIMAL_PAM,
        upstream_pad=_UPSTREAM,
        downstream_pad=_DOWNSTREAM,
    )
    nick = 32 + 17
    edit_pos = nick + 5
    spec = _make_substitution_spec(
        chrom="chrSyn",
        pos_0b=edit_pos,
        ref=genome[edit_pos],
        alt="A" if genome[edit_pos] != "A" else "T",
    )

    runs = []
    for _ in range(5):
        out = enumerate_pe2_candidates(
            spec,
            target_genome_seq=genome,
            scaffold_variant="sgRNA_canonical",
        )
        # Reduce to a tuple of sortable identity fields for byte-equal
        # comparison without relying on dataclass __eq__ subtleties on
        # repeated calls.
        sig = tuple(
            (
                c.chrom,
                c.nick_site,
                c.strand,
                c.pbs_length,
                c.rtt_length,
                c.scaffold_variant,
                c.spacer_seq,
                c.pbs_seq,
                c.rtt_seq,
                c.full_pegrna_rna_seq,
                c.rt_product_seq,
            )
            for c in out
        )
        runs.append(sig)

    first = runs[0]
    for i, r in enumerate(runs[1:], start=2):
        assert r == first, f"run #{i} diverged from run #1 (non-deterministic order)"

    # Verify the ordering itself matches (chrom, nick_site, strand, pbs_length,
    # rtt_length, scaffold_variant) ascending.
    keys = [
        (c[0], c[1], c[2], c[3], c[4], c[5]) for c in first
    ]
    assert keys == sorted(keys), "candidate list is not sorted by the canonical key"
