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

"""Slow-but-correct CPU/numpy reference for canonical k-mer counting (T2).

This is the **ground-truth oracle** that downstream silicon byte-equal
tests (T12, T13, T14, T16) consume. Correctness here is critical —
spend the cycles on tests rather than on optimising; the on-NPU kernel
will be fast.

Wire format (per T1's interface contract):

* Alphabet: A=00, C=01, G=10, T=11.
* 4 bases per byte.
* **MSB-first**: the first base of the sequence occupies bits[7:6] of
  byte 0, the second bits[5:4], the third bits[3:2], the fourth
  bits[1:0]. The fifth base lives in bits[7:6] of byte 1, etc.

Per-k mask constants (used in canonical-RC math):

* ``KMER_MASK_K15 = (1 << 30) - 1``
* ``KMER_MASK_K21 = (1 << 42) - 1``
* ``KMER_MASK_K31 = (1 << 62) - 1``

Note (divergence from CRISPR's ``encode_2bit``): the CRISPR-side
20-nt spacer packer (`bionpu.kernels.crispr.match_singletile.encode_2bit`)
uses LSB-first byte ordering — bits 1:0 hold the first base. This
module's ``pack_dna_2bit`` uses **MSB-first** because that is what
T1's wire-format pin specifies for the streaming whole-genome k-mer
kernel. The two formats are intentionally different and are NOT
byte-equal on the same input. See ``test_crispr_packer_byte_equal``
in tests for a documented assertion of the bit-pair-mirror
relationship.
"""

from __future__ import annotations

from typing import Final

import numpy as np

__all__ = [
    "KMER_MASK_K15",
    "KMER_MASK_K21",
    "KMER_MASK_K31",
    "canonical_kmer_2bit",
    "count_kmers_canonical",
    "kmer_mask",
    "pack_dna_2bit",
    "unpack_dna_2bit",
]

# Per-k masks, exact bit-widths (T1's interface contract).
KMER_MASK_K15: Final[int] = (1 << 30) - 1
KMER_MASK_K21: Final[int] = (1 << 42) - 1
KMER_MASK_K31: Final[int] = (1 << 62) - 1

# Internal lookup tables.
_BASE_TO_2BIT: Final[dict[str, int]] = {"A": 0, "C": 1, "G": 2, "T": 3}
_2BIT_TO_BASE: Final[str] = "ACGT"


def kmer_mask(k: int) -> int:
    """Return the per-k uint64 mask ``(1 << (2*k)) - 1``.

    Convenience accessor; equivalent to the pinned ``KMER_MASK_K{k}``
    constants for ``k in {15, 21, 31}`` and generalises for any
    ``1 <= k <= 32``.
    """
    if not 1 <= k <= 32:
        raise ValueError(f"k must be in 1..32 (uint64 capacity); got {k}")
    return (1 << (2 * k)) - 1


# --------------------------------------------------------------------------- #
# (a) pack_dna_2bit
# --------------------------------------------------------------------------- #


def pack_dna_2bit(seq: str) -> np.ndarray:
    """Pack an ACGT string into 2-bit MSB-first bytes (T1 wire format).

    Args:
        seq: ASCII ACGT string (uppercase). Length need not be a
            multiple of 4 — the trailing partial byte is zero-padded
            in its low-order bits.

    Returns:
        np.ndarray of dtype uint8 with shape ``(ceil(len(seq)/4),)``.
        Per T1's wire-format pin: byte 0 bits[7:6] = first base,
        bits[5:4] = second, bits[3:2] = third, bits[1:0] = fourth;
        byte 1 bits[7:6] = fifth, etc.

    Raises:
        ValueError: ``seq`` contains a non-ACGT character.
    """
    n = len(seq)
    n_bytes = (n + 3) // 4
    out = np.zeros(n_bytes, dtype=np.uint8)
    for i, c in enumerate(seq):
        v = _BASE_TO_2BIT.get(c)
        if v is None:
            raise ValueError(
                f"non-ACGT base {c!r} at position {i}; pack_dna_2bit "
                "requires ACGT-only input (callers must canonicalise N "
                "before packing — see T3 fixture builder for the "
                "documented v1 N-handling policy)"
            )
        # MSB-first: position-in-byte 0 lands at bits[7:6], 1 at [5:4],
        # 2 at [3:2], 3 at [1:0]. Equivalent shift = (3 - (i % 4)) * 2.
        byte_idx = i // 4
        shift = (3 - (i % 4)) * 2
        out[byte_idx] |= np.uint8((v & 0x3) << shift)
    return out


# --------------------------------------------------------------------------- #
# (b) unpack_dna_2bit
# --------------------------------------------------------------------------- #


def unpack_dna_2bit(buf: np.ndarray, n_bases: int) -> str:
    """Inverse of :func:`pack_dna_2bit`.

    Args:
        buf: uint8 ndarray as produced by :func:`pack_dna_2bit`.
        n_bases: number of bases originally packed. Must satisfy
            ``ceil(n_bases / 4) == buf.shape[0]``.

    Returns:
        ACGT string of length ``n_bases``.
    """
    if buf.dtype != np.uint8:
        raise ValueError(f"buf dtype must be uint8; got {buf.dtype}")
    if buf.ndim != 1:
        raise ValueError(f"buf must be 1-D; got shape {buf.shape}")
    expected = (n_bases + 3) // 4
    if buf.shape[0] != expected:
        raise ValueError(
            f"buf has {buf.shape[0]} bytes but n_bases={n_bases} requires "
            f"{expected}"
        )
    chars: list[str] = []
    for i in range(n_bases):
        byte_idx = i // 4
        shift = (3 - (i % 4)) * 2
        v = (int(buf[byte_idx]) >> shift) & 0x3
        chars.append(_2BIT_TO_BASE[v])
    return "".join(chars)


# --------------------------------------------------------------------------- #
# (c) canonical_kmer_2bit
# --------------------------------------------------------------------------- #


def canonical_kmer_2bit(kmer_int: int, k: int) -> int:
    """Return ``min(forward, reverse_complement)`` of a packed k-mer.

    The forward k-mer is encoded as a uint64 with the first base in
    the highest-order 2-bit lane (so a fresh base shifted in via
    ``(fwd << 2) | new_base & mask`` is consistent with T1's wire
    format pin and with the rolling counter in
    :func:`count_kmers_canonical`).

    Per T1's contract: callers MUST pass ``kmer_int`` already masked
    to ``kmer_mask(k)``. Bits above ``2*k`` are an error.

    Math:

    1. ``complement = kmer_int ^ mask`` — A↔T (00↔11) and C↔G (01↔10)
       are bitwise complements within their 2-bit lanes, so XOR with
       the all-ones-within-2k mask flips each base to its complement.
    2. Reverse the 2-bit lanes to get the actual reverse complement
       (the leftmost lane in the forward k-mer becomes the rightmost
       lane in RC, after complementing).

    Returns:
        ``min(forward, rc)`` as a non-negative Python int.
    """
    mask = (1 << (2 * k)) - 1
    if kmer_int < 0 or (kmer_int & ~mask) != 0:
        raise ValueError(
            f"kmer_int=0x{kmer_int:x} not in [0, 2^{2 * k}); caller must "
            f"pre-mask with kmer_mask({k})"
        )
    comp = kmer_int ^ mask
    # Reverse 2-bit lanes. Loop is bounded by k <= 32 so this is fine
    # for an oracle. (The on-NPU kernel uses a rolling forward+rc
    # register pair so it never needs to do this reversal — see T5.)
    rc = 0
    c = comp
    for _ in range(k):
        rc = (rc << 2) | (c & 0x3)
        c >>= 2
    if kmer_int <= rc:
        return kmer_int
    return rc


# --------------------------------------------------------------------------- #
# (d) count_kmers_canonical
# --------------------------------------------------------------------------- #


def count_kmers_canonical(seq: str, k: int) -> dict[int, int]:
    """Slow-but-correct rolling canonical k-mer counter.

    For each length-``k`` substring of ``seq`` consisting solely of
    ACGT characters, computes its canonical 2-bit-packed uint64
    representation (= ``min(forward, reverse_complement)``) and
    increments its count. Substrings containing non-ACGT characters
    are skipped (matches the v1 N-handling convention from the
    plan's T3 — N is converted to A in the packed binary, but the
    oracle here is meant to validate against the RAW string and
    therefore drops Ns rather than mis-counting).

    Args:
        seq: ACGT(+other) sequence as a Python string.
        k: k-mer length. Must be ``1 <= k <= 32`` so the packed k-mer
            fits in a uint64.

    Returns:
        dict mapping ``canonical_kmer_uint64 -> count``. Empty if
        ``len(seq) < k`` (does NOT raise).
    """
    if not 1 <= k <= 32:
        raise ValueError(f"k must be in 1..32; got {k}")
    n = len(seq)
    if n < k:
        return {}

    mask = (1 << (2 * k)) - 1
    counts: dict[int, int] = {}

    # Rolling forward + RC encoders.
    #
    # forward: first base sits at the top 2 bits, so shifting in a new
    # base is `((fwd << 2) | new) & mask`. After processing positions
    # 0..k-1 the forward register holds the first k-mer.
    #
    # rc: the running reverse-complement. For a streaming definition,
    # if we append base `b` to forward, we must prepend `comp(b)` to
    # rc. So `rc = (rc >> 2) | (comp(b) << (2*(k-1)))`. This gives
    # the rc of the same k-mer window without re-deriving it from
    # scratch each step. (Same trick as the AIE2P kernel will use in
    # T5.)
    high_lane_shift = 2 * (k - 1)
    fwd = 0
    rc = 0
    valid_run = 0  # how many consecutive ACGT bases we have seen

    for i in range(n):
        c = seq[i]
        v = _BASE_TO_2BIT.get(c)
        if v is None:
            # Non-ACGT base resets the rolling state. The next k-mer
            # window cannot start until we have seen k more ACGT
            # bases.
            valid_run = 0
            fwd = 0
            rc = 0
            continue

        fwd = ((fwd << 2) | v) & mask
        # comp(v) for 2-bit ACGT codes is `v ^ 0b11`:
        # A(00)↔T(11), C(01)↔G(10).
        rc = (rc >> 2) | ((v ^ 0x3) << high_lane_shift)
        valid_run += 1

        if valid_run >= k:
            canonical = fwd if fwd <= rc else rc
            counts[canonical] = counts.get(canonical, 0) + 1

    return counts
