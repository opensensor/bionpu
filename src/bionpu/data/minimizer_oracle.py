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

"""Slow-but-correct CPU/numpy reference for canonical (w, k) minimizers.

This oracle is the **ground-truth reference** that downstream silicon
byte-equal tests consume. The algorithm is a deterministic specialisation
of the canonical sliding-window minimizer:

* For each length-k substring of the input, compute its canonical 2-bit
  uint64 representation = ``min(forward, reverse_complement)`` (same as
  :func:`bionpu.data.kmer_oracle.count_kmers_canonical`).
* Maintain a sliding window of the last w canonicals.
* Emit the smallest canonical in each window with its 0-indexed start
  position. Emission rule (deterministic, mirror-able in silicon):
  - When a new k-mer enters the window AND its canonical is STRICTLY
    less than the current minimum → emit the new k-mer (becomes the
    new min).
  - When the previous min slides out (its position falls below
    ``i - w + 1`` where ``i`` is the index of the newest k-mer) → scan
    the remaining w-1 entries for the new minimum. Emit it with its
    actual position.
  - Ties (equal canonicals): the OLDEST entry in the window wins (i.e.,
    on a fresh tie we don't replace; on slide-out scan, the leftmost
    matching entry is chosen). This is byte-equal mirrorable in the
    tile kernel.

This is NOT minimap2's exact mm_sketch behaviour: minimap2 uses a
secondary hash64 ordering, skips symmetric k-mers (where forward==rc),
emits identical-key duplicates, and does HPC. For v0 we use a SIMPLER
pure-canonical scheme so the silicon byte-equal contract is tight and
implementable in scalar AIE2P code with a tiny ring buffer.

Wire format (per kmer_oracle module + minimizer_constants.h):

* Alphabet: A=00, C=01, G=10, T=11.
* 4 bases per byte. **MSB-first**.
* ``KMER_MASK_K15 = (1 << 30) - 1``,
  ``KMER_MASK_K21 = (1 << 42) - 1``.

Output records:

* ``[(canonical_u64, position_int), ...]`` sorted by ``position`` asc.
* ``position`` is 0-indexed start offset (in BASES) of the minimizer
  k-mer within the input string.

Pinned configurations (matching the silicon artifacts):

* (k=15, w=10) — short-read default (minimap2's short-read default).
* (k=21, w=11) — long-read default.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from bionpu.data.kmer_oracle import (
    KMER_MASK_K15,
    KMER_MASK_K21,
    KMER_MASK_K31,
    pack_dna_2bit,
    unpack_dna_2bit,
)

__all__ = [
    "DEFAULT_W_FOR_K",
    "MINIMIZER_RECORD_BYTES",
    "SUPPORTED_KW",
    "extract_minimizers",
    "extract_minimizers_packed",
    "kmer_mask_for",
]

# 16-byte wire record per emit: (uint64 canonical, uint32 position, uint32 _pad).
MINIMIZER_RECORD_BYTES: Final[int] = 16

# Pinned (k, w) configurations matching the silicon artifacts.
SUPPORTED_KW: Final[tuple[tuple[int, int], ...]] = ((15, 10), (21, 11))

# Default w per k (used by the op-class constructor when w is omitted).
DEFAULT_W_FOR_K: Final[dict[int, int]] = {15: 10, 21: 11}

_KMER_MASKS: Final[dict[int, int]] = {
    15: KMER_MASK_K15,
    21: KMER_MASK_K21,
    31: KMER_MASK_K31,
}

_BASE_TO_2BIT: Final[dict[str, int]] = {"A": 0, "C": 1, "G": 2, "T": 3}


def kmer_mask_for(k: int) -> int:
    """Return the per-k uint64 canonical mask."""
    if int(k) not in _KMER_MASKS:
        valid = ", ".join(str(x) for x in _KMER_MASKS)
        raise ValueError(
            f"k={k!r} unsupported by minimizer oracle; expected one of {{{valid}}}"
        )
    return _KMER_MASKS[int(k)]


def _validate_params(k: int, w: int) -> None:
    if not 1 <= k <= 32:
        raise ValueError(f"k must be in 1..32 (uint64 capacity); got {k}")
    if not 1 <= w <= 256:
        raise ValueError(f"w must be in 1..256; got {w}")


def extract_minimizers(seq: str, k: int, w: int) -> list[tuple[int, int]]:
    """Extract canonical (w, k) minimizers from an ACGT string.

    Algorithm (deterministic, silicon-mirrorable):

    1. Walk the sequence, maintaining a rolling forward + reverse-complement
       uint64 pair. For each i ≥ k - 1 that completes a length-k window
       of ACGT bases, compute ``canonical = min(fwd, rc)``.
    2. Append ``(canonical, kmer_start_position)`` into a length-w ring
       buffer where ``kmer_start_position = i - k + 1``.
    3. Once the ring buffer holds w valid canonicals (i.e., w consecutive
       k-mers have entered the window), the ``current_min`` is the smallest
       canonical with the OLDEST position on tie.
    4. Emit ``current_min`` once when the window first fills. After that,
       on each newly-arriving canonical:
       a. If the new canonical is STRICTLY less than ``current_min`` →
          emit it; new ``current_min`` = new canonical.
       b. Else if the entry at the position currently held by
          ``current_min`` slides out (its position falls below the window
          start) → re-scan the surviving w-1 entries for the new min
          (oldest wins on tie); emit it.
       c. Otherwise: the current min is unchanged, no emit.
    5. Non-ACGT bases reset the rolling state and the window. The next
       window can fill only after w + k - 1 consecutive ACGT bases.

    Args:
        seq: ACGT (case-sensitive) string. Non-ACGT bases reset state.
        k: k-mer length (1..32).
        w: window length (1..256).

    Returns:
        List of ``(canonical_u64, kmer_start_position)`` sorted by
        ``kmer_start_position`` ascending. Per-position uniqueness is
        guaranteed by the algorithm (only one emit happens per
        new-base step).
    """
    _validate_params(k, w)
    n = len(seq)
    if n < k + w - 1:
        return []

    mask = kmer_mask_for(k) if k in _KMER_MASKS else (1 << (2 * k)) - 1
    high_lane_shift = 2 * (k - 1)

    # Rolling forward + reverse-complement registers.
    fwd = 0
    rc = 0
    valid_run = 0  # consecutive ACGT bases observed.

    # Ring buffer entries: (canonical_or_None, kmer_start_pos_or_None).
    # We use Python lists of fixed length w; positions index by
    # `slot = (kmer_index_in_run) % w`.
    UNSET = (None, None)  # noqa: N806
    ring: list[tuple[int | None, int | None]] = [UNSET] * w

    # Number of canonicals pushed into the ring since last reset.
    n_pushed = 0

    # Current min state — only valid once the window has fully filled
    # (n_pushed >= w). We track the canonical and its absolute position.
    cur_min_canonical: int | None = None
    cur_min_pos: int | None = None  # absolute base-position
    cur_min_slot: int | None = None  # ring index (so we can detect slide-out)

    out: list[tuple[int, int]] = []

    def _scan_ring_for_min() -> tuple[int, int, int]:
        """Scan the ring buffer for the oldest-on-tie minimum.

        Returns ``(min_canonical, min_pos, min_slot)``. Caller must only
        call this when n_pushed >= w (i.e., all w slots are populated).
        """
        # Walk slots in INSERTION ORDER (oldest first) so on a tie the
        # oldest entry wins. The slot containing the OLDEST entry is
        # ``n_pushed % w`` (about-to-overwrite slot if we pushed one
        # more); but during the scan we want to start at the oldest still
        # present, which is ``n_pushed % w`` (it's stale only when n_pushed
        # advances past it, which we model by always reading just-after
        # an overwrite).
        oldest_slot = n_pushed % w
        best_canon = -1  # sentinel; first valid entry replaces it
        best_pos = -1
        best_slot = -1
        for off in range(w):
            slot = (oldest_slot + off) % w
            entry_canon, entry_pos = ring[slot]
            if entry_canon is None:
                continue
            if best_canon < 0 or entry_canon < best_canon:
                best_canon = entry_canon
                best_pos = entry_pos
                best_slot = slot
        return best_canon, best_pos, best_slot

    for i in range(n):
        c = seq[i]
        v = _BASE_TO_2BIT.get(c)
        if v is None:
            # Non-ACGT base: reset rolling state AND the window.
            valid_run = 0
            fwd = 0
            rc = 0
            n_pushed = 0
            cur_min_canonical = None
            cur_min_pos = None
            cur_min_slot = None
            for s in range(w):
                ring[s] = UNSET
            continue

        fwd = ((fwd << 2) | v) & mask
        rc = (rc >> 2) | ((v ^ 0x3) << high_lane_shift)
        valid_run += 1

        if valid_run < k:
            continue

        canonical = fwd if fwd <= rc else rc
        kmer_start_pos = i - k + 1

        # Push into the ring at slot = n_pushed % w. The previous
        # occupant (if any) is OVERWRITTEN; this is the slide-out event.
        slot = n_pushed % w
        ring[slot] = (canonical, kmer_start_pos)
        n_pushed += 1

        if n_pushed < w:
            # Window not yet full; no emits possible.
            continue

        if n_pushed == w:
            # First time the window has w entries — emit the current min.
            cur_min_canonical, cur_min_pos, cur_min_slot = _scan_ring_for_min()
            if cur_min_canonical >= 0:
                out.append((cur_min_canonical, cur_min_pos))
            continue

        # n_pushed > w: window has slid by one. The slot we just
        # overwrote is `slot`. If it held the previous min, the min has
        # slid out; rescan. Otherwise check if the new canonical is a
        # strict improvement.
        if cur_min_slot == slot:
            # The previous min was overwritten by the new entry.
            # Re-scan the entire (now-full) ring for the new min.
            cur_min_canonical, cur_min_pos, cur_min_slot = _scan_ring_for_min()
            out.append((cur_min_canonical, cur_min_pos))
        elif (
            cur_min_canonical is not None
            and canonical < cur_min_canonical
        ):
            # Strict improvement: new canonical is the new min.
            cur_min_canonical = canonical
            cur_min_pos = kmer_start_pos
            cur_min_slot = slot
            out.append((cur_min_canonical, cur_min_pos))
        # else: no change; no emit.

    return out


def extract_minimizers_packed(
    packed_seq: np.ndarray,
    n_bases: int,
    k: int,
    w: int,
) -> list[tuple[int, int]]:
    """Run :func:`extract_minimizers` against a packed-2-bit fixture.

    Convenience wrapper used by silicon byte-equal harnesses.

    Args:
        packed_seq: uint8 ndarray as produced by
            :func:`bionpu.data.kmer_oracle.pack_dna_2bit`.
        n_bases: number of bases packed (length of the unpacked string).
        k: k-mer length.
        w: window length.

    Returns:
        Same as :func:`extract_minimizers`.
    """
    seq = unpack_dna_2bit(packed_seq, n_bases)
    return extract_minimizers(seq, k=k, w=w)


# Re-exports for convenience.
_ = pack_dna_2bit  # silence unused-import linters; users can import via this module
