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

"""Slow-but-correct CPU/numpy reference for short tandem repeat detection.

The oracle uses an **autocorrelation-streak semantics** that is byte-
equal with the silicon kernel's per-period streak counters:

  For each period q in [MIN_PERIOD, MAX_PERIOD]:
    Walk p in [q, n_bases):
      If seq[p] == seq[p - q]:
        If streak == 0: streak_start = p - q  (first motif base)
        streak += 1
      Else:
        Maybe-emit the streak (see below) and reset.
    At end of sequence, maybe-emit any pending streak.

  A streak qualifies iff streak >= q * (MIN_COPIES - 1) — meaning the
  run contains at least MIN_COPIES copies of the period-q motif. The
  emitted record is:
    start  = streak_start
    end    = streak_start + n_copies * q  (q-aligned)
    period = q
    motif  = seq[streak_start : streak_start + q]
  where n_copies = (streak + q) // q.

This semantics differs slightly from TRF / the naive "greedy scan from
i" definition: when one motif transitions to another at a non-q-aligned
boundary, the autocorrelation start can be up to q-1 bases earlier than
the greedy-scan start. Both representations identify the same STR with
the same number of copies; the byte boundary just shifts. We pin
autocorrelation semantics here because they have a 1-to-1 silicon
realization (single per-period streak counter) and are unambiguous.

A host-side cross-period dedup pass picks the longest record at each
overlapping position (smaller period wins on ties). This matches what
TRF reports for chr22-class fixtures within the dedup convention's
tolerance.

Output record tuple: ``(start, end, period, motif)`` where ``end`` is
exclusive and ``motif`` is the ASCII motif string of length ``period``.
"""

from __future__ import annotations

from typing import Final

import numpy as np

__all__ = [
    "TR_MAX_PERIOD",
    "TR_MIN_COPIES",
    "TR_MIN_PERIOD",
    "find_tandem_repeats",
    "find_tandem_repeats_packed",
    "motif_to_canonical_u32",
]


# --------------------------------------------------------------------------- #
# Pinned constants (mirror tandem_repeat_constants.h byte-equal).
# --------------------------------------------------------------------------- #

TR_MIN_PERIOD: Final[int] = 1
TR_MAX_PERIOD: Final[int] = 6
TR_MIN_COPIES: Final[int] = 5

_BASE_TO_2BIT: Final[dict[str, int]] = {"A": 0, "C": 1, "G": 2, "T": 3}


# --------------------------------------------------------------------------- #
# Motif packing.
# --------------------------------------------------------------------------- #


def motif_to_canonical_u32(motif: str) -> int:
    """Pack an ASCII motif (length 1..6) as a uint32, MSB-first.

    The first base occupies the highest 2 bits of the period * 2 used
    bits. For ``period < 6`` the unused high bits are zero.

    Example: motif="ACGT" (period=4) -> bits = 00 01 10 11 = 0x1B = 27.
    """
    p = len(motif)
    if not 1 <= p <= TR_MAX_PERIOD:
        raise ValueError(
            f"motif length must be in 1..{TR_MAX_PERIOD}; got {p}"
        )
    v = 0
    for ch in motif:
        b = _BASE_TO_2BIT.get(ch.upper())
        if b is None:
            raise ValueError(f"non-ACGT base {ch!r} in motif {motif!r}")
        v = (v << 2) | b
    return v & 0xFFFFFFFF


# --------------------------------------------------------------------------- #
# Oracle: per-period autocorrelation streak detection + cross-period dedup.
# --------------------------------------------------------------------------- #


def _emit(streak: int, streak_start: int, q: int, seq: str,
          out: list[tuple[int, int, int, str]],
          threshold: int) -> None:
    if streak < threshold:
        return
    n_copies = (streak + q) // q
    end = streak_start + n_copies * q
    motif = seq[streak_start:streak_start + q]
    out.append((streak_start, end, q, motif))


def find_tandem_repeats(
    seq: str,
    *,
    min_period: int = TR_MIN_PERIOD,
    max_period: int = TR_MAX_PERIOD,
    min_copies: int = TR_MIN_COPIES,
) -> list[tuple[int, int, int, str]]:
    """Find tandem repeats in ``seq`` via per-period autocorrelation
    streak detection + cross-period dedup.

    Args:
        seq: ACGT (case-sensitive) string.
        min_period: smallest period to scan. Defaults to 1.
        max_period: largest period to scan. Defaults to 6.
        min_copies: minimum copy count required to emit a record.
            Defaults to 5.

    Returns:
        Sorted list of ``(start, end, period, motif)`` tuples, where
        ``end`` is the exclusive end position. Records are sorted by
        ``(start asc, length desc, period asc)`` and then deduplicated
        so each base position appears in at most one record (longer
        repeats win on ties; equal-length ties resolved by smaller
        period first).
    """
    if min_period < 1:
        raise ValueError(f"min_period must be >= 1; got {min_period}")
    if max_period < min_period:
        raise ValueError(
            f"max_period ({max_period}) < min_period ({min_period})"
        )
    if min_copies < 2:
        raise ValueError(f"min_copies must be >= 2; got {min_copies}")

    n = len(seq)
    candidates: list[tuple[int, int, int, str]] = []

    for period in range(min_period, max_period + 1):
        threshold = period * (min_copies - 1)
        streak = 0
        streak_start = 0
        for p in range(period, n):
            if seq[p] == seq[p - period]:
                if streak == 0:
                    streak_start = p - period
                streak += 1
            else:
                _emit(streak, streak_start, period, seq,
                      candidates, threshold)
                streak = 0
        # End-of-sequence flush.
        _emit(streak, streak_start, period, seq, candidates, threshold)

    # Cross-period dedup: longer wins; smaller period wins on ties.
    candidates.sort(key=lambda r: (r[0], -(r[1] - r[0]), r[2]))
    deduped: list[tuple[int, int, int, str]] = []
    last_end = -1
    for r in candidates:
        if r[0] >= last_end:
            deduped.append(r)
            last_end = r[1]
    return deduped


def find_tandem_repeats_packed(
    packed_seq: np.ndarray,
    n_bases: int,
    *,
    min_period: int = TR_MIN_PERIOD,
    max_period: int = TR_MAX_PERIOD,
    min_copies: int = TR_MIN_COPIES,
) -> list[tuple[int, int, int, str]]:
    """Run :func:`find_tandem_repeats` against a packed-2-bit fixture."""
    from bionpu.data.kmer_oracle import unpack_dna_2bit

    seq = unpack_dna_2bit(packed_seq, n_bases)
    return find_tandem_repeats(
        seq,
        min_period=min_period,
        max_period=max_period,
        min_copies=min_copies,
    )
