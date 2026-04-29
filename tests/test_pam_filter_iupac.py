# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# Tests for the IUPAC PAM oracle (Track A v0). The silicon byte-equal
# tests against the live xclbin are in state/be_design_smoke.py (which
# requires a working NPU bring-up); these unit tests cover the CPU
# oracle independent of silicon.

from __future__ import annotations

from bionpu.data.pam_iupac_oracle import (
    IUPAC_NIBBLE,
    encode_pam_iupac,
    find_pam_matches,
    find_pam_matches_packed,
)
from bionpu.data.kmer_oracle import pack_dna_2bit


def test_iupac_nibbles_canonical() -> None:
    """The IUPAC nibble encoding must be self-consistent.

    R = A | G; Y = C | T; etc. The nibble values are checked against
    the bitwise OR of the constituent bases (A=1, C=2, G=4, T=8).
    """
    assert IUPAC_NIBBLE["A"] == 0x1
    assert IUPAC_NIBBLE["C"] == 0x2
    assert IUPAC_NIBBLE["G"] == 0x4
    assert IUPAC_NIBBLE["T"] == 0x8
    assert IUPAC_NIBBLE["R"] == IUPAC_NIBBLE["A"] | IUPAC_NIBBLE["G"]
    assert IUPAC_NIBBLE["Y"] == IUPAC_NIBBLE["C"] | IUPAC_NIBBLE["T"]
    assert IUPAC_NIBBLE["S"] == IUPAC_NIBBLE["G"] | IUPAC_NIBBLE["C"]
    assert IUPAC_NIBBLE["W"] == IUPAC_NIBBLE["A"] | IUPAC_NIBBLE["T"]
    assert IUPAC_NIBBLE["K"] == IUPAC_NIBBLE["G"] | IUPAC_NIBBLE["T"]
    assert IUPAC_NIBBLE["M"] == IUPAC_NIBBLE["A"] | IUPAC_NIBBLE["C"]
    assert IUPAC_NIBBLE["N"] == 0xF


def test_encode_pam_ngg() -> None:
    """NGG encodes to nibble 0xF (N) at pos 0, 0x4 (G) at 1, 0x4 (G) at 2."""
    mask, plen = encode_pam_iupac("NGG")
    assert plen == 3
    # bits: 0xF | (0x4 << 4) | (0x4 << 8) = 0x44F
    assert mask == 0x44F


def test_encode_pam_nrn_spry() -> None:
    """SpRY's NRN encodes to (N | R<<4 | N<<8) = 0xF5F."""
    mask, plen = encode_pam_iupac("NRN")
    assert plen == 3
    assert mask == 0xF5F


def test_encode_pam_nnnrrt_kkh() -> None:
    """SaCas9-KKH NNNRRT encodes correctly across 6 positions."""
    mask, plen = encode_pam_iupac("NNNRRT")
    assert plen == 6
    expected = (0xF | (0xF << 4) | (0xF << 8) |
                (0x5 << 12) | (0x5 << 16) | (0x8 << 20))
    assert mask == expected


def test_find_pam_simple_ngg() -> None:
    """A single NGG hit at a known position."""
    seq = "AAAA" + "AGG" + "AAAA"
    hits = find_pam_matches(seq, "NGG")
    assert hits == [(4, 0)]


def test_find_pam_multiple_ngg() -> None:
    """Multiple NGG hits in a small fixture."""
    seq = "AGGCGGTGG"
    hits = find_pam_matches(seq, "NGG")
    # pos 0: AGG, pos 3: CGG, pos 6: TGG
    assert hits == [(0, 0), (3, 0), (6, 0)]


def test_find_pam_no_match() -> None:
    """A sequence with no PAM hits returns an empty list."""
    seq = "AAAAAAAAAAAA"
    hits = find_pam_matches(seq, "NGG")
    assert hits == []


def test_find_pam_nrn_spry_density() -> None:
    """SpRY's NRN is much denser than NGG.

    For ACGT random sequence, NRN should match ~2/4 = 50% of positions
    (P(R=A or G)=0.5; N is wildcard). NGG should match ~1/16 = 6.25%.
    """
    # A deterministic ACGT fixture; we just check the relative densities.
    seq = "ACGT" * 100  # 400 bases
    hits_ngg = find_pam_matches(seq, "NGG")
    hits_nrn = find_pam_matches(seq, "NRN")
    # NRN should produce many more hits than NGG.
    assert len(hits_nrn) > 5 * len(hits_ngg), (
        f"NRN={len(hits_nrn)} should dominate NGG={len(hits_ngg)} "
        f"on random ACGT"
    )


def test_find_pam_nnnrrt_palindrome_safe() -> None:
    """NNNRRT (SaCas9-KKH) at a literal hit position."""
    # Inject CCCAAT at position 5 -- N=any C/A/T, R=A/G, R=A/G, T=T -- match
    seq = "GGGGG" + "CCCAAT" + "GGGGG"
    hits = find_pam_matches(seq, "NNNRRT")
    assert (5, 0) in hits, f"expected (5, 0) in {hits!r}"


def test_find_pam_packed_round_trip() -> None:
    """The packed-2-bit fixture API matches the ASCII API."""
    seq = "AGGAAAGGAAAAAAAAAAAA"
    n_bases = len(seq)
    packed = pack_dna_2bit(seq)
    hits_ascii = find_pam_matches(seq, "NGG")
    hits_packed = find_pam_matches_packed(packed, n_bases, "NGG")
    assert hits_packed == hits_ascii


def test_iupac_match_kernel_window_extraction() -> None:
    """Trace the kernel's window extraction logic on a tiny fixture.

    For PAM length 3 and bases [A, G, G] in order:
      - kernel updates rolling win = (win << 2 | base) & mask
      - after seeing A,G,G: win = (((0<<2|0)<<2|2)<<2|2) & 0x3F = 0b001010 = 10
      - position 0 (5'-most, A): win >> (2*(3-1-0)) & 3 = win >> 4 & 3 = 0 (A) ✓
      - position 1 (G):          win >> (2*(3-1-1)) & 3 = win >> 2 & 3 = 2 (G) ✓
      - position 2 (G):          win >> (2*(3-1-2)) & 3 = win & 3      = 2 (G) ✓
    Also re-verify via the oracle.
    """
    seq = "AGG"
    hits = find_pam_matches(seq, "NGG")
    assert hits == [(0, 0)]
    # And via the kernel's window math (manually):
    # After AGG: win = 0b001010 = 10
    win = 0
    pam_length = 3
    mask = (1 << (2 * pam_length)) - 1
    for b in (0, 2, 2):  # A, G, G
        win = ((win << 2) | b) & mask
    # Position 0: highest 2 bits.
    assert ((win >> (2 * (pam_length - 1 - 0))) & 0x3) == 0  # A
    assert ((win >> (2 * (pam_length - 1 - 1))) & 0x3) == 2  # G
    assert ((win >> (2 * (pam_length - 1 - 2))) & 0x3) == 2  # G
