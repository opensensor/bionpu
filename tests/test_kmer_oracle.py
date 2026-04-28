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

"""Tests for the canonical k-mer oracle + 2-bit packing helpers (T2).

This is the slow-but-correct CPU/numpy reference that downstream
silicon byte-equal tests (T12, T13, T14, T16) consume as ground truth.

Wire format pin (per T1's interface contract):
* A=00, C=01, G=10, T=11
* 4 bases per byte
* **MSB-first**: first base of sequence = bits[7:6] of byte 0,
  next = bits[5:4], etc.

Per-k masks (used in canonical RC math):
* KMER_MASK_K15 = (1 << 30) - 1
* KMER_MASK_K21 = (1 << 42) - 1
* KMER_MASK_K31 = (1 << 62) - 1
"""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
import pytest

from bionpu.data.kmer_oracle import (
    canonical_kmer_2bit,
    count_kmers_canonical,
    pack_dna_2bit,
    unpack_dna_2bit,
)

_BASES = "ACGT"
_BASE_TO_2BIT = {"A": 0, "C": 1, "G": 2, "T": 3}


def _random_seq(n: int, seed: int = 0) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice(_BASES) for _ in range(n))


def _kmer_to_int_msb(kmer: str) -> int:
    """Encode an ACGT k-mer string into a uint64.

    Uses the same convention used inside :func:`count_kmers_canonical`:
    the first base lands in the highest-order 2-bit lane, so that the
    rolling shift consumes top bits first (matches T1's MSB-first
    wire-format pin).
    """
    out = 0
    for c in kmer:
        out = (out << 2) | _BASE_TO_2BIT[c]
    return out


def _naive_canonical(kmer: str) -> int:
    """Reference canonical-encode of an ACGT k-mer string.

    Computes both forward and reverse-complement int representations
    independently from the strings, then returns ``min``. This is the
    naive ground truth the rolling counter must agree with.
    """
    rc = "".join({"A": "T", "C": "G", "G": "C", "T": "A"}[c] for c in kmer[::-1])
    fwd = _kmer_to_int_msb(kmer)
    rev = _kmer_to_int_msb(rc)
    return min(fwd, rev)


# --------------------------------------------------------------------------- #
# (a) pack/unpack round-trip
# --------------------------------------------------------------------------- #


def test_roundtrip_pack_unpack() -> None:
    """Random ACGT 1 Kbp sequence: pack -> unpack -> equal to input."""
    seq = _random_seq(1024, seed=42)
    packed = pack_dna_2bit(seq)
    assert isinstance(packed, np.ndarray)
    assert packed.dtype == np.uint8
    # 1024 bases / 4 bases-per-byte = 256 bytes
    assert packed.shape == (256,)
    rt = unpack_dna_2bit(packed, len(seq))
    assert rt == seq


# --------------------------------------------------------------------------- #
# (b) canonical idempotence
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("k", [15, 21, 31])
def test_canonical_idempotent(k: int) -> None:
    """canonical(canonical(x), k) == canonical(x, k) for 100 random inputs."""
    rng = random.Random(0xC0FFEE ^ k)
    mask = (1 << (2 * k)) - 1
    for _ in range(100):
        # Random uint64 within the per-k mask (per T1 contract: caller
        # must pre-mask before invoking canonical).
        x = rng.getrandbits(2 * k) & mask
        c1 = canonical_kmer_2bit(x, k)
        c2 = canonical_kmer_2bit(c1, k)
        assert c1 == c2, (
            f"non-idempotent at k={k}: x={x:#x} c1={c1:#x} c2={c2:#x}"
        )
        # And c1 must itself be within the mask.
        assert 0 <= c1 <= mask


# --------------------------------------------------------------------------- #
# (c) canonical <= min(forward, rc)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("k", [15, 21, 31])
def test_canonical_lt_or_eq_forward(k: int) -> None:
    """canonical(x, k) == min(x, rc(x)) — i.e., always returns the smaller."""
    rng = random.Random(0xBADBEEF ^ k)
    mask = (1 << (2 * k)) - 1
    for _ in range(200):
        x = rng.getrandbits(2 * k) & mask
        # Independent reverse-complement: complement = x ^ mask, then
        # reverse the 2-bit lanes manually.
        comp = x ^ mask
        rc = 0
        for i in range(k):
            lane = (comp >> (2 * i)) & 0x3
            rc = (rc << 2) | lane
        expected = min(x, rc)
        got = canonical_kmer_2bit(x, k)
        assert got == expected, (
            f"k={k} x={x:#x} rc={rc:#x} expected min={expected:#x} got={got:#x}"
        )
        # And canonical <= forward:
        assert got <= x


# --------------------------------------------------------------------------- #
# (d) rolling counter agrees with naive Counter on small input
# --------------------------------------------------------------------------- #


def test_count_kmers_naive_agreement() -> None:
    """At k=21 on a 100 bp random seq, dict matches a naive Counter."""
    k = 21
    seq = _random_seq(100, seed=7)
    got = count_kmers_canonical(seq, k)

    # Naive ground truth: enumerate every length-k substring, canonicalise,
    # count.
    naive: Counter[int] = Counter()
    for i in range(len(seq) - k + 1):
        kmer = seq[i : i + k]
        if any(c not in _BASE_TO_2BIT for c in kmer):
            continue
        naive[_naive_canonical(kmer)] += 1

    assert dict(got) == dict(naive), (
        f"got {len(got)} unique k-mers, naive {len(naive)} — diff:"
        f" {set(got.items()) ^ set(naive.items())}"
    )


def test_count_kmers_short_seq_returns_empty() -> None:
    """Sequences shorter than k must return an empty dict (not crash)."""
    assert count_kmers_canonical("ACGT", k=21) == {}
    assert count_kmers_canonical("", k=15) == {}
    # Exactly k-1 bases:
    assert count_kmers_canonical("A" * 20, k=21) == {}


def test_count_kmers_exact_k_length() -> None:
    """A length-k sequence must yield exactly one canonical k-mer."""
    k = 15
    seq = _random_seq(k, seed=11)
    got = count_kmers_canonical(seq, k)
    assert len(got) == 1
    assert sum(got.values()) == 1


# --------------------------------------------------------------------------- #
# (e) byte-equality vs CRISPR pam_filter spacer packer (skip if conventions
#     differ — they currently DO, so the body documents the divergence)
# --------------------------------------------------------------------------- #


def test_crispr_packer_byte_equal() -> None:
    """Pack 1 Kbp ACGT-only fixture and compare to CRISPR's encode_2bit.

    DIVERGENCE NOTE: CRISPR's `bionpu.kernels.crispr.match_singletile.
    encode_2bit` is a 20-nt-only spacer packer that uses
    **little-endian-within-byte** packing (bits 1:0 hold the first base,
    bits 3:2 the second, ...). T2's `pack_dna_2bit` follows T1's wire-
    format pin which is **MSB-first** (bits 7:6 hold the first base).

    The conventions are intentionally different:
    * encode_2bit was designed for a 20-nt fixed-width spacer kernel
      where the per-byte ordering matched the kernel's vload/vstore
      pattern.
    * pack_dna_2bit is the streaming whole-genome packer used by the
      kmer kernel (T1 wire-format pin).

    So byte-equal is NOT expected. We assert the expected mirror
    relationship (each pair of nibbles is bit-reversed within a byte)
    and skip the strict byte-equal assert with a documented reason.
    Downstream T3 (fixture builder) and T5 (per-tile kernel C++) MUST
    use the MSB-first wire format from this module — NOT encode_2bit.
    """
    try:
        from bionpu.kernels.crispr.match_singletile import (
            SPACER_BYTES,
            SPACER_LEN,
            encode_2bit,
        )
    except Exception as exc:  # noqa: BLE001 — broad on purpose for skip
        pytest.skip(f"CRISPR encode_2bit unavailable: {exc}")
        return

    # Build a ~1 Kbp ACGT-only fixture (truncated to a multiple of
    # SPACER_LEN=20 so we can pack chunk-wise with the CRISPR helper —
    # 1024 % 20 = 4, so we use 1020 bases / 51 spacers).
    raw = _random_seq(1024, seed=99)
    n_full = (len(raw) // SPACER_LEN) * SPACER_LEN
    seq = raw[:n_full]  # 1020 bases
    assert len(seq) % SPACER_LEN == 0
    assert len(seq) >= 1000  # still ~1 Kbp per the task spec

    # Pack with T2's whole-sequence MSB-first packer.
    t2_packed = pack_dna_2bit(seq)
    assert t2_packed.dtype == np.uint8

    # Pack with CRISPR's 20-nt LSB-first packer, chunk-wise.
    crispr_packed = np.concatenate(
        [
            encode_2bit(seq[i : i + SPACER_LEN])
            for i in range(0, len(seq), SPACER_LEN)
        ]
    )
    assert crispr_packed.dtype == np.uint8
    # Byte counts match (1020 / 4 = 255):
    assert crispr_packed.shape == t2_packed.shape

    # Now: are they byte-equal? On a non-trivial random fixture,
    # MSB-first vs LSB-first packing produces DIFFERENT bytes (each
    # byte is the bit-reversed-by-2-bit-pairs of the other for the
    # same 4-base window). Document this divergence and skip the
    # byte-equal claim — it is intentionally not byte-equal because
    # the wire formats differ.
    if not np.array_equal(t2_packed, crispr_packed):
        # Verify the documented mirror relationship: T2's byte ==
        # bit-pair-reversed CRISPR byte (i.e. swap nibble-pair lanes
        # within each byte).
        def _reverse_2bit_pairs(b: int) -> int:
            return (
                ((b >> 6) & 0x3)
                | (((b >> 4) & 0x3) << 2)
                | (((b >> 2) & 0x3) << 4)
                | ((b & 0x3) << 6)
            )

        mirrored = np.array(
            [_reverse_2bit_pairs(int(b)) for b in crispr_packed],
            dtype=np.uint8,
        )
        assert np.array_equal(t2_packed, mirrored), (
            "T2 packer is not even the bit-pair mirror of CRISPR packer"
            " — wire-format relationship broken"
        )
        pytest.skip(
            "DOCUMENTED DIVERGENCE: T2 uses MSB-first per T1 contract; "
            "CRISPR encode_2bit uses LSB-first for 20-nt spacers. "
            "Byte-equal mirror verified."
        )
