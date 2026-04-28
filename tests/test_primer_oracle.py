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

"""TDD tests for the primer/adapter scanning oracle.

Mirrors :file:`tests/test_minimizer_oracle.py` for the v0
primer_scan kernel. The RED→GREEN cycle: imports of
:func:`find_primer_matches` should fail before the oracle module
exists; once the module is in place, all tests pass and become the
regression lock for the silicon byte-equal harness.
"""

from __future__ import annotations

import numpy as np
import pytest

from bionpu.data.kmer_oracle import pack_dna_2bit, unpack_dna_2bit
from bionpu.data.primer_oracle import (
    TRUSEQ_P5_ADAPTER,
    encode_primer_canonical,
    find_primer_matches,
    find_primer_matches_packed,
)


# --------------------------------------------------------------------------- #
# Test 1: short hand-crafted fixture
# --------------------------------------------------------------------------- #


def test_simple_substring_match() -> None:
    """``AAAAGATCGGAAGAGCAAAA`` matches the TruSeq P5 adapter at pos 4."""
    primer = TRUSEQ_P5_ADAPTER  # "AGATCGGAAGAGC", 13 bp
    seq = "AAAA" + primer + "AAAA"
    hits = find_primer_matches(seq, primer, allow_rc=False)
    # Forward-strand only.
    assert hits == [(4, 0)], f"expected [(4, 0)], got {hits!r}"


# --------------------------------------------------------------------------- #
# Test 2: edge cases
# --------------------------------------------------------------------------- #


def test_empty_and_short_seq() -> None:
    """Sequences shorter than the primer return no matches."""
    primer = "ACGT"
    assert find_primer_matches("", primer) == []
    assert find_primer_matches("A", primer) == []
    assert find_primer_matches("ACG", primer) == []


def test_primer_equals_seq() -> None:
    """When the seq equals the primer, exactly one forward match at pos 0.

    With ``allow_rc=True``, a non-palindromic primer also produces zero
    additional RC matches (the seq IS the primer, not its RC).
    """
    primer = "ACGTACGT"  # not a palindrome (rc = ACGTACGT? let's check)
    # rc("ACGTACGT") = AC GT AC GT reversed-complemented = AC GT AC GT
    # actually: ACGTACGT -> complement TGCATGCA -> reverse ACGTACGT
    # so it IS a palindrome at length 8! Pick a clearly non-palindromic
    # primer instead.
    primer = "AAAACCCC"  # rc = GGGGTTTT
    hits_no_rc = find_primer_matches(primer, primer, allow_rc=False)
    assert hits_no_rc == [(0, 0)]
    hits_rc = find_primer_matches(primer, primer, allow_rc=True)
    # Only forward; no RC match because seq != rc(primer).
    assert hits_rc == [(0, 0)]


# --------------------------------------------------------------------------- #
# Test 3: reverse-complement match
# --------------------------------------------------------------------------- #


def test_rc_match() -> None:
    """The RC of a non-palindrome primer in the query produces a strand=1 hit."""
    primer = "AAAACCCC"  # rc = GGGGTTTT
    rc_str = "GGGGTTTT"
    seq = "AAAA" + rc_str + "AAAA"  # rc of primer at pos 4
    hits_no_rc = find_primer_matches(seq, primer, allow_rc=False)
    assert hits_no_rc == []
    hits_rc = find_primer_matches(seq, primer, allow_rc=True)
    assert hits_rc == [(4, 1)], f"expected [(4, 1)], got {hits_rc!r}"


# --------------------------------------------------------------------------- #
# Test 4: multiple forward + RC matches
# --------------------------------------------------------------------------- #


def test_multiple_matches_mixed_strand() -> None:
    """A query with two forward hits and one RC hit returns all three.

    Sorted by (position asc, strand asc).
    """
    primer = "AAAACCCC"  # rc = GGGGTTTT
    rc_str = "GGGGTTTT"
    seq = (
        primer       # pos 0 forward
        + "TT"
        + rc_str     # pos 10 RC (= len(primer) + 2)
        + "TT"
        + primer     # pos 20 forward
    )
    hits = find_primer_matches(seq, primer, allow_rc=True)
    assert hits == [(0, 0), (10, 1), (20, 0)], (
        f"expected [(0,0),(10,1),(20,0)], got {hits!r}"
    )


# --------------------------------------------------------------------------- #
# Test 5: encode_primer_canonical sanity
# --------------------------------------------------------------------------- #


def test_encode_primer_canonical_truseq_p5() -> None:
    """Encoding the TruSeq P5 adapter produces consistent fwd / rc / canonical."""
    fwd, rc, canon = encode_primer_canonical(TRUSEQ_P5_ADAPTER)
    # Sanity: canonical = min(fwd, rc).
    assert canon == min(fwd, rc)
    # Round-trip: rc of rc should equal fwd.
    fwd2, rc2, canon2 = encode_primer_canonical(_rc_str(TRUSEQ_P5_ADAPTER))
    assert fwd2 == rc, (
        f"encode(rc(p)).fwd ({fwd2}) should equal encode(p).rc ({rc})"
    )
    assert rc2 == fwd
    assert canon2 == canon


# --------------------------------------------------------------------------- #
# Test 6: packed-2-bit round-trip equivalence
# --------------------------------------------------------------------------- #


def test_packed_path_equivalent_to_string_path() -> None:
    """:func:`find_primer_matches_packed` produces the same hits as the string path."""
    seq = (
        "AAAA"
        + TRUSEQ_P5_ADAPTER
        + "TTTT"
        + _rc_str(TRUSEQ_P5_ADAPTER)
        + "GGGG"
    )
    # Pad to a multiple of 4 so pack_dna_2bit can pack cleanly.
    while len(seq) % 4 != 0:
        seq += "A"
    packed = pack_dna_2bit(seq)
    hits_packed = find_primer_matches_packed(
        packed, n_bases=len(seq), primer=TRUSEQ_P5_ADAPTER, allow_rc=True
    )
    hits_string = find_primer_matches(
        seq, TRUSEQ_P5_ADAPTER, allow_rc=True
    )
    assert hits_packed == hits_string


# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #


def _rc_str(s: str) -> str:
    comp = {"A": "T", "C": "G", "G": "C", "T": "A"}
    return "".join(comp[c] for c in reversed(s))
