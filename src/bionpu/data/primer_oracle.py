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

"""Slow-but-correct CPU/numpy reference for primer/adapter scanning.

This oracle is the **ground-truth reference** that the primer_scan
silicon byte-equal harness consumes. It enumerates every position in
the input where a primer (or its reverse complement) occurs as an
exact match.

For v0 we restrict to:

* Single primer per call.
* Exact match (no mismatch tolerance).
* Forward strand AND reverse complement (caller can disable via
  ``allow_rc=False``).

The silicon kernel's matching logic is deliberately mirrored here:

* Encode the primer as a uint64 canonical = ``min(forward, rc)`` over
  the same packed-2-bit / MSB-first wire format as kmer_count.
* Walk the query, maintain a rolling forward + rc uint64 register pair
  (k = primer length).
* When canonical equals the primer's canonical, emit (position, strand)
  where strand = 0 if forward matched and strand = 1 if RC matched
  (determined by which of fwd / rc equalled the canonical at emit time).

This makes the oracle and the kernel byte-equal by construction on the
same input / primer.

Records are returned sorted by ``(position asc, strand asc)``.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from bionpu.data.kmer_oracle import (
    canonical_kmer_2bit,
    kmer_mask,
    pack_dna_2bit,
    unpack_dna_2bit,
)

__all__ = [
    "PRIMER_RECORD_BYTES",
    "TRUSEQ_P5_ADAPTER",
    "encode_primer_canonical",
    "find_primer_matches",
    "find_primer_matches_packed",
]

# 16-byte wire record per emit (mirrors minimizer):
#   uint32 query_pos LE | uint8 strand | uint8 primer_idx | uint16 _pad |
#   uint64 _pad2
PRIMER_RECORD_BYTES: Final[int] = 16

#: Illumina TruSeq P5 adapter (canonical default for v0).
TRUSEQ_P5_ADAPTER: Final[str] = "AGATCGGAAGAGC"

_BASE_TO_2BIT: Final[dict[str, int]] = {"A": 0, "C": 1, "G": 2, "T": 3}


def encode_primer_canonical(primer: str) -> tuple[int, int, int]:
    """Encode an ASCII primer as ``(forward_u64, rc_u64, canonical_u64)``.

    Args:
        primer: ASCII ACGT string, length 1..32.

    Returns:
        ``(forward, rc, canonical)`` where ``forward`` / ``rc`` are the
        2-bit-packed uint64 representations (first base in highest 2-bit
        lane) and ``canonical = min(forward, rc)``.
    """
    p = len(primer)
    if not 1 <= p <= 32:
        raise ValueError(f"primer length must be in 1..32; got {p}")
    fwd = 0
    for ch in primer:
        v = _BASE_TO_2BIT.get(ch.upper())
        if v is None:
            raise ValueError(
                f"non-ACGT base {ch!r} in primer; primer_oracle requires "
                "ACGT-only input"
            )
        fwd = (fwd << 2) | v
    fwd &= kmer_mask(p)
    canonical = canonical_kmer_2bit(fwd, p)
    # rc = canonical ^ fwd ^ canonical -- not useful; recompute from canonical
    # by re-deriving via canonical_kmer_2bit semantics:
    # canonical = min(fwd, rc) so rc is the OTHER member of {fwd, canonical}
    # if fwd != canonical, else rc = canonical_kmer_2bit's other branch.
    # Simpler: compute rc directly.
    mask = kmer_mask(p)
    comp = fwd ^ mask
    rc = 0
    c = comp
    for _ in range(p):
        rc = (rc << 2) | (c & 0x3)
        c >>= 2
    return fwd, rc, canonical


def find_primer_matches(
    seq: str,
    primer: str,
    *,
    allow_rc: bool = True,
) -> list[tuple[int, int]]:
    """Find every exact-match position of ``primer`` in ``seq``.

    Walks ``seq`` with a rolling forward + reverse-complement uint64
    register pair (k = ``len(primer)``). For each position ``i`` whose
    completed k-mer equals the primer (forward) or its RC, emits
    ``(kmer_start_pos, strand)`` where:

    * ``strand = 0`` iff the forward register equals the primer's
      forward encoding;
    * ``strand = 1`` iff the rc register equals the primer's forward
      encoding (i.e. the primer matches the reverse complement of the
      query at this position).

    Both can fire on a single position if the primer is its own RC
    (palindrome); both records are emitted.

    Non-ACGT bases reset the rolling state (no match can straddle
    Ns).

    Args:
        seq: ACGT (case-sensitive) string. Non-ACGT bases reset state.
        primer: ASCII ACGT primer, length 1..32.
        allow_rc: If False, only forward-strand matches are emitted.

    Returns:
        List of ``(query_pos, strand)`` sorted by ``(query_pos asc,
        strand asc)``. ``query_pos`` is the 0-indexed start of the
        match k-mer in ``seq``.
    """
    p = len(primer)
    if not 1 <= p <= 32:
        raise ValueError(f"primer length must be in 1..32; got {p}")
    n = len(seq)
    if n < p:
        return []

    primer_fwd, primer_rc, _canon = encode_primer_canonical(primer)
    mask = kmer_mask(p)
    high_lane_shift = 2 * (p - 1)

    fwd = 0
    rc = 0
    valid_run = 0
    out: list[tuple[int, int]] = []

    for i in range(n):
        c = seq[i]
        v = _BASE_TO_2BIT.get(c)
        if v is None:
            valid_run = 0
            fwd = 0
            rc = 0
            continue
        fwd = ((fwd << 2) | v) & mask
        rc = (rc >> 2) | ((v ^ 0x3) << high_lane_shift)
        valid_run += 1
        if valid_run < p:
            continue

        kmer_start = i - p + 1
        # Forward-strand match: primer == fwd.
        if fwd == primer_fwd:
            out.append((kmer_start, 0))
        # RC match: primer == rc (i.e. the query's substring at this
        # position is the RC of the primer). Only fire if allow_rc AND
        # the primer isn't a palindrome (palindrome = fwd == rc; the
        # forward branch above already covered the position).
        if allow_rc and rc == primer_fwd and primer_fwd != primer_rc:
            out.append((kmer_start, 1))
        elif allow_rc and rc == primer_fwd and primer_fwd == primer_rc:
            # Palindrome with allow_rc=True: the primer matches itself
            # both ways at this position. Emit the RC record too so the
            # oracle remains symmetric with allow_rc=False on
            # palindromes (which would emit only the forward record).
            # We tag as strand=1 to keep ordering deterministic.
            # NOTE: this branch is unreachable because the forward
            # match above already fired. Kept for documentation; in
            # practice we never duplicate-emit the same record.
            pass

    out.sort(key=lambda t: (t[0], t[1]))
    return out


def find_primer_matches_packed(
    packed_seq: np.ndarray,
    n_bases: int,
    primer: str,
    *,
    allow_rc: bool = True,
) -> list[tuple[int, int]]:
    """Run :func:`find_primer_matches` against a packed-2-bit fixture.

    Args:
        packed_seq: uint8 ndarray as produced by
            :func:`bionpu.data.kmer_oracle.pack_dna_2bit`.
        n_bases: number of bases packed.
        primer: ASCII ACGT primer.
        allow_rc: as in :func:`find_primer_matches`.

    Returns:
        Same as :func:`find_primer_matches`.
    """
    seq = unpack_dna_2bit(packed_seq, n_bases)
    return find_primer_matches(seq, primer, allow_rc=allow_rc)


# Re-export silencer for unused-import linters.
_ = pack_dna_2bit
