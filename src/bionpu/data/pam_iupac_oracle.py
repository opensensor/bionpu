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

"""CPU/numpy reference for IUPAC PAM-filter scanning.

This oracle is the **ground-truth reference** that the
``pam_filter_iupac`` silicon byte-equal harness consumes. It enumerates
every position in the input where an IUPAC-encoded PAM occurs as an
exact match.

For Track A v0 we emit only the forward strand (mirrors the locked
``crispr/pam_filter`` kernel — host pre-flips for RC; see
``crispr/pam_filter`` DESIGN.md §3 "strand handling").

The silicon kernel's matching logic is deliberately mirrored here:

* For each position i where the rolling window has at least
  ``len(pam)`` valid bases, check the PAM against the IUPAC-encoded
  per-position 4-bit mask.
* Per-position mask: A=1, C=2, G=4, T=8 (one-hot 4-bit). N=0xF, R=0x5,
  Y=0xA, etc.
* Match iff for every position p, ``(1 << base_at_p) & pos_mask != 0``.

Records are returned sorted by position ascending.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from bionpu.data.kmer_oracle import pack_dna_2bit, unpack_dna_2bit

__all__ = [
    "IUPAC_NIBBLE",
    "PAM_RECORD_BYTES",
    "encode_pam_iupac",
    "find_pam_matches",
    "find_pam_matches_packed",
]

#: 16-byte wire record per emit (mirrors primer_scan):
#:   uint32 query_pos LE | uint8 strand | uint8 _pad | uint16 _pad2 |
#:   uint64 _pad3
PAM_RECORD_BYTES: Final[int] = 16

#: IUPAC 4-bit one-hot encoding (A=bit0, C=bit1, G=bit2, T=bit3).
IUPAC_NIBBLE: Final[dict[str, int]] = {
    "A": 0x1,
    "C": 0x2,
    "G": 0x4,
    "T": 0x8,
    "R": 0x5,  # A | G  (puRine)
    "Y": 0xA,  # C | T  (pYrimidine)
    "S": 0x6,  # G | C  (Strong)
    "W": 0x9,  # A | T  (Weak)
    "K": 0xC,  # G | T  (Keto)
    "M": 0x3,  # A | C  (aMino)
    "B": 0xE,  # C | G | T (not A)
    "D": 0xD,  # A | G | T (not C)
    "H": 0xB,  # A | C | T (not G)
    "V": 0x7,  # A | C | G (not T)
    "N": 0xF,  # any
}

_BASE_TO_2BIT: Final[dict[str, int]] = {"A": 0, "C": 1, "G": 2, "T": 3}


def encode_pam_iupac(pam: str) -> tuple[int, int]:
    """Encode an IUPAC PAM string as ``(pam_mask u32, pam_length u8)``.

    Args:
        pam: IUPAC PAM string (e.g. "NGG", "NRN", "NNNRRT"). Length 1..8.

    Returns:
        ``(mask, length)`` where ``mask`` is a uint32 packed nibbles
        (position 0 in bits [0..3], position 1 in [4..7], ...) and
        ``length`` is the number of active PAM positions.
    """
    p = len(pam)
    if not 1 <= p <= 8:
        raise ValueError(f"PAM length must be in 1..8; got {p}")
    mask = 0
    for i, ch in enumerate(pam):
        nib = IUPAC_NIBBLE.get(ch.upper())
        if nib is None:
            raise ValueError(f"non-IUPAC PAM base: {ch!r}")
        mask |= (nib & 0xF) << (4 * i)
    return mask & 0xFFFFFFFF, p


def find_pam_matches(seq: str, pam: str) -> list[tuple[int, int]]:
    """Find every forward-strand exact-match position of ``pam`` in ``seq``.

    Walks ``seq`` with a rolling window of length ``len(pam)``. Non-ACGT
    bases reset the rolling state. For each completed window, compare
    each base to the IUPAC mask at the same position.

    Args:
        seq: ACGT (case-sensitive) string. Non-ACGT bases reset state.
        pam: IUPAC PAM (e.g. "NGG").

    Returns:
        List of ``(query_pos, strand)`` sorted by ``query_pos asc``.
        Strand is always 0 (forward) for v0; the host runs RC as a
        separate pass (mirrors locked ``crispr/pam_filter``).
    """
    pam_mask, pam_length = encode_pam_iupac(pam)
    n = len(seq)
    if n < pam_length:
        return []

    # Per-position mask helper.
    def pos_nibble(p: int) -> int:
        return (pam_mask >> (4 * p)) & 0xF

    out: list[tuple[int, int]] = []
    valid_run = 0
    # Rolling buffer of last pam_length 2-bit values; we re-extract each
    # time. Simple ring buffer.
    win: list[int] = [0] * pam_length

    for i in range(n):
        c = seq[i]
        v = _BASE_TO_2BIT.get(c)
        if v is None:
            valid_run = 0
            continue
        # Slide window forward (FIFO).
        win[i % pam_length] = v
        valid_run += 1
        if valid_run < pam_length:
            continue

        pam_start = i - pam_length + 1
        # Check positions: position 0 is the FIRST PAM base (5'-most),
        # which is the OLDEST entry in the ring. Order them out.
        match = True
        for p in range(pam_length):
            base = win[(pam_start + p) % pam_length]
            base_onehot = 1 << base
            if (base_onehot & pos_nibble(p)) == 0:
                match = False
                break
        if match:
            out.append((pam_start, 0))

    return out


def find_pam_matches_packed(
    packed_seq: np.ndarray,
    n_bases: int,
    pam: str,
) -> list[tuple[int, int]]:
    """Run :func:`find_pam_matches` against a packed-2-bit fixture.

    Args:
        packed_seq: uint8 ndarray as produced by
            :func:`bionpu.data.kmer_oracle.pack_dna_2bit`.
        n_bases: number of bases packed.
        pam: IUPAC PAM string.

    Returns:
        Same as :func:`find_pam_matches`.
    """
    seq = unpack_dna_2bit(packed_seq, n_bases)
    return find_pam_matches(seq, pam)


# Re-export silencer for unused-import linters.
_ = pack_dna_2bit
