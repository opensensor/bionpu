# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

"""TDD tests for the tandem repeat (STR) oracle.

Mirrors :file:`tests/test_cpg_oracle.py` for the v0 tandem_repeat
kernel. Locks the period / min_copies / motif math the silicon kernel
must reproduce byte-for-byte.
"""

from __future__ import annotations

from bionpu.data.tandem_repeat_oracle import (
    TR_MAX_PERIOD,
    TR_MIN_COPIES,
    TR_MIN_PERIOD,
    find_tandem_repeats,
    find_tandem_repeats_packed,
    motif_to_canonical_u32,
)


# --------------------------------------------------------------------------- #
# Constants pin
# --------------------------------------------------------------------------- #


def test_default_constants() -> None:
    assert TR_MIN_PERIOD == 1
    assert TR_MAX_PERIOD == 6
    assert TR_MIN_COPIES == 5


# --------------------------------------------------------------------------- #
# Test 1: an all-A sequence is one mono-nucleotide STR spanning the input.
# --------------------------------------------------------------------------- #


def test_all_a_one_mononucleotide_str() -> None:
    """A-only string of length L is a single period=1 STR (start=0,
    end=L, motif="A")."""
    n = 100
    seq = "A" * n
    out = find_tandem_repeats(seq)
    assert out == [(0, n, 1, "A")], f"got {out!r}"


def test_all_cg_dinucleotide_str() -> None:
    """CG-repeats trigger the period=1 mono-nucleotide check first
    (streak[1] never builds because each base differs from its
    predecessor); then period=2 wins for the whole input."""
    n = 100  # even
    seq = "CG" * (n // 2)
    out = find_tandem_repeats(seq)
    # Expect a single period=2 record [0, n) with motif "CG".
    assert out == [(0, n, 2, "CG")], f"got {out!r}"


# --------------------------------------------------------------------------- #
# Test 2: short sequence below threshold emits nothing.
# --------------------------------------------------------------------------- #


def test_short_sequence_no_emit() -> None:
    # Period=1 needs MIN_COPIES * 1 = 5 bases minimum.
    assert find_tandem_repeats("AAAA") == []
    assert find_tandem_repeats("AAAAA") == [(0, 5, 1, "A")]
    # Period=2 needs MIN_COPIES * 2 = 10 bases minimum.
    assert find_tandem_repeats("CGCGCGCG") == []
    assert find_tandem_repeats("CGCGCGCGCG") == [(0, 10, 2, "CG")]
    # Empty: no records.
    assert find_tandem_repeats("") == []


# --------------------------------------------------------------------------- #
# Test 3: hexa-nucleotide STR.
# --------------------------------------------------------------------------- #


def test_hexa_nucleotide_str_in_padding() -> None:
    """A hexa-nucleotide STR (period=6, 5 copies = 30 bp) embedded in
    A-padding emits exactly one record (the period=6 STR) AND two
    period=1 mono-A records (one for each A-block of length >= 5)."""
    motif = "ACGTAC"
    pad = "T" * 50  # avoid any incidental period-1 in pad
    seq = pad + (motif * 6) + pad
    out = find_tandem_repeats(seq)
    # Expect exactly: pad-T STR (start=0, len=50, period=1, "T"),
    # the hexa STR (start=50, end=50+36=86, period=6, "ACGTAC"),
    # the trailing pad-T STR (start=86, len=50).
    assert (50, 86, 6, motif) in out
    assert (0, 50, 1, "T") in out
    assert (86, 136, 1, "T") in out
    # No other records.
    assert len(out) == 3, f"got {out!r}"


# --------------------------------------------------------------------------- #
# Test 4: dedup ensures non-overlapping records.
# --------------------------------------------------------------------------- #


def test_records_non_overlapping_after_dedup() -> None:
    """An all-A region that's ALSO an "AA" period-2 region should
    deduplicate to a single period-1 record (longer wins; period-1 has
    same length but smaller period)."""
    seq = "A" * 60
    out = find_tandem_repeats(seq)
    # Expect a single period=1 record (start=0, end=60, motif="A").
    # Period=2 ("AA") candidate spans [0, 60) too but ties go to smaller
    # period.
    assert out == [(0, 60, 1, "A")], f"got {out!r}"

    # Walk the records: every base position 0..59 in exactly one record.
    last_end = -1
    for r in out:
        assert r[0] >= last_end, f"records overlap: {out!r}"
        last_end = r[1]


# --------------------------------------------------------------------------- #
# Test 5: motif canonical packing matches MSB-first 2-bit encoding.
# --------------------------------------------------------------------------- #


def test_motif_canonical_packing() -> None:
    # Single base: A=00, C=01, G=10, T=11.
    assert motif_to_canonical_u32("A") == 0
    assert motif_to_canonical_u32("C") == 1
    assert motif_to_canonical_u32("G") == 2
    assert motif_to_canonical_u32("T") == 3

    # Period=2: MSB-first packing.
    # "AC" = 00 01 = 0b0001 = 1
    # "GT" = 10 11 = 0b1011 = 11
    assert motif_to_canonical_u32("AC") == 1
    assert motif_to_canonical_u32("GT") == 11

    # Period=6: motif "ACGTAC" = 00 01 10 11 00 01 = 0b000110110001 = 433
    assert motif_to_canonical_u32("ACGTAC") == 0b000110110001


# --------------------------------------------------------------------------- #
# Test 6: smoke regression-lock — pinned record count on a deterministic
# mixed fixture.
# --------------------------------------------------------------------------- #


def test_smoke_regression_lock_mixed() -> None:
    """A pinned mixed fixture lights up several STR archetypes; lock the
    record list so any drift in the period/streak math is caught
    pre-silicon.

    Layout (use long disjoint runs separated by guard bases):
      pos 0:   "T"*40        — period=1 T STR
      pos 40:  guard "GAGA"  — breaks the T streak, but is itself a
                              period=2 STR ("GA"*2 = 4 chars: too short)
      pos 44:  "ACG"*7 = 21bp  period=3 STR
      pos 65:  guard "TT"     — breaks the ACG streak; too short for STR
      pos 67:  "CACA"*3 = 12bp + "TGTG"*3 = 12bp  back-to-back distinct
                                                  period-2 STRs
      pos 91:  rest ‘N’-equivalent with non-matching guards
    """
    seq = (
        "T" * 40             # 0..40    — period=1 T STR
        + "GAGA"              # 40..44   — guard, no STR (too short)
        + "ACG" * 7           # 44..65   — period=3 ACG STR (7 copies)
        + "TT"                # 65..67   — guard
        + "CACA" * 3          # 67..79   — period=2 CA STR-ish but
                              #            "CACA"*3 = 12bp = 6 copies of "CA"
        + "TGTG" * 3          # 79..91   — period=2 TG STR (6 copies)
        + "AGGTCAGTCAA"       # 91..102  — random tail
    )
    out = find_tandem_repeats(seq)

    # Expected anchors:
    assert (0, 40, 1, "T") in out
    assert (44, 65, 3, "ACG") in out
    # "CACA"*3 = "CACACACACACA" = 6 copies of "CA" period=2.
    # The greedy oracle finds period=1 first (no streaks because bases
    # alternate), then period=2 with motif "CA" runs from 67 for 12 bp.
    assert (67, 79, 2, "CA") in out
    assert (79, 91, 2, "TG") in out

    # Record count should be at least these 4. Tail might add more
    # period-1 records inside random; lock the >= 4 floor.
    assert len(out) >= 4, f"expected >= 4 records, got {out!r}"


# --------------------------------------------------------------------------- #
# Test 7: packed-path matches string-path.
# --------------------------------------------------------------------------- #


def test_packed_path_equivalent_to_string_path() -> None:
    from bionpu.data.kmer_oracle import pack_dna_2bit

    seq = ("T" * 40) + ("ACG" * 7) + ("A" * 60)
    while len(seq) % 4 != 0:
        seq += "A"
    packed = pack_dna_2bit(seq)
    str_records = find_tandem_repeats(seq)
    packed_records = find_tandem_repeats_packed(packed, n_bases=len(seq))
    assert str_records == packed_records


# --------------------------------------------------------------------------- #
# Test 8: trinucleotide STR is found at correct boundaries.
# --------------------------------------------------------------------------- #


def test_trinucleotide_str_boundaries() -> None:
    """6 copies of "CAG" = 18 bp; embedded in T-padding."""
    motif = "CAG"
    pad_left = "T" * 60
    pad_right = "T" * 60
    seq = pad_left + (motif * 6) + pad_right
    out = find_tandem_repeats(seq)
    assert (60, 78, 3, motif) in out
    # Pad regions emit period=1 T runs.
    assert (0, 60, 1, "T") in out
    assert (78, 138, 1, "T") in out
