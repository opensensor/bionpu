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

"""Slow-but-correct CPU/numpy reference for CpG island detection.

Implements the Gardiner-Garden & Frommer (1987) standard: for each
window of length ``W`` over the input, compute GC% and the
observed-vs-expected CpG ratio; a "candidate" position is one whose
length-``W`` window starting there satisfies BOTH thresholds. A CpG
island is a maximal run of ``>=W`` consecutive candidate positions.

For numerical exactness with the silicon kernel (which avoids float),
we phrase the thresholds as integer comparisons:

* ``GC_pct >= 0.5``     <=>  ``2 * (n_C + n_G) >= W``
* ``obs_exp_CG >= 0.6`` <=>  ``5 * W * n_CG >= 3 * n_C * n_G``

The ``obs_exp_CG`` reformulation uses the Gardiner-Garden definition
``obs_exp_CG = (n_CG * W) / (n_C * n_G)`` (when ``n_C * n_G > 0``;
else the ratio is undefined and we treat it as failing the threshold,
matching the typical convention).

Output records: ``[(island_start, island_end), ...]`` sorted by start
asc. ``island_end`` is EXCLUSIVE (the standard half-open convention).

This oracle is the **ground-truth reference** that the cpg_island
silicon byte-equal harness consumes. The silicon kernel's per-position
ON_STREAK output is host-side run-merged into ``(start, end)`` ranges
that match this oracle exactly.
"""

from __future__ import annotations

from typing import Final

import numpy as np

__all__ = [
    "CPG_DEFAULT_GC_NUM",
    "CPG_DEFAULT_GC_DEN",
    "CPG_DEFAULT_OBS_EXP_NUM",
    "CPG_DEFAULT_OBS_EXP_DEN",
    "CPG_DEFAULT_W",
    "find_cpg_island_streak_positions",
    "find_cpg_islands",
    "find_cpg_islands_packed",
    "merge_candidate_positions_to_islands",
    "merge_streak_positions_to_islands",
]


#: Gardiner-Garden window length.
CPG_DEFAULT_W: Final[int] = 200

#: GC% threshold = 0.5 expressed as a fraction.
CPG_DEFAULT_GC_NUM: Final[int] = 1
CPG_DEFAULT_GC_DEN: Final[int] = 2

#: obs/exp CpG threshold = 0.6 expressed as a fraction.
CPG_DEFAULT_OBS_EXP_NUM: Final[int] = 3
CPG_DEFAULT_OBS_EXP_DEN: Final[int] = 5


def _candidate_mask(
    seq: str,
    w: int,
    gc_num: int,
    gc_den: int,
    oe_num: int,
    oe_den: int,
) -> np.ndarray:
    """Per-position boolean mask: True if the length-w window starting
    at position p satisfies BOTH thresholds.

    Returns an ndarray of shape ``(max(0, len(seq) - w + 1),)``.
    """
    n = len(seq)
    if n < w:
        return np.zeros(0, dtype=bool)

    # Rolling counters over the window.
    n_c = 0
    n_g = 0
    n_cg = 0  # count of CpG dinucleotides FULLY inside the window

    # Initial fill: positions [0, w).
    for i in range(w):
        ch = seq[i]
        if ch == "C":
            n_c += 1
        elif ch == "G":
            n_g += 1
        # CpG dinucleotides spanning [i-1, i] for i >= 1, both inside [0, w).
        if i >= 1 and seq[i - 1] == "C" and ch == "G":
            n_cg += 1

    n_windows = n - w + 1
    mask = np.zeros(n_windows, dtype=bool)

    def _is_candidate(nc: int, ng: int, ncg: int) -> bool:
        # GC%: gc_den * (nc + ng) >= gc_num * w
        if gc_den * (nc + ng) < gc_num * w:
            return False
        # obs/exp: oe_den * w * ncg >= oe_num * nc * ng
        # Treat the n_c*n_g == 0 case as failing the threshold.
        if nc == 0 or ng == 0:
            return False
        if oe_den * w * ncg < oe_num * nc * ng:
            return False
        return True

    mask[0] = _is_candidate(n_c, n_g, n_cg)

    # Slide window by 1 base at each step, p in [1, n_windows).
    for p in range(1, n_windows):
        # Remove leaving base at position p-1.
        out_ch = seq[p - 1]
        if out_ch == "C":
            n_c -= 1
        elif out_ch == "G":
            n_g -= 1
        # If seq[p-1..p+1) was a CpG fully in the OLD window
        # (positions p-1 and p, both < p-1+w, i.e. p < w which always
        # held while p < w), removing the leaving base also removes
        # that dinuc. The OLD window covered [p-1, p-1+w), so
        # the dinuc seq[p-1] seq[p] was inside if p < p-1+w -> always.
        if seq[p - 1] == "C" and seq[p] == "G":
            n_cg -= 1

        # Add entering base at position p+w-1.
        in_idx = p + w - 1
        in_ch = seq[in_idx]
        if in_ch == "C":
            n_c += 1
        elif in_ch == "G":
            n_g += 1
        # New dinuc seq[in_idx-1] seq[in_idx] entering (both fully in
        # the NEW window [p, p+w) since in_idx-1 == p+w-2 >= p when
        # w >= 2).
        if seq[in_idx - 1] == "C" and seq[in_idx] == "G":
            n_cg += 1

        mask[p] = _is_candidate(n_c, n_g, n_cg)

    return mask


def find_cpg_island_streak_positions(
    seq: str,
    *,
    w: int = CPG_DEFAULT_W,
    gc_num: int = CPG_DEFAULT_GC_NUM,
    gc_den: int = CPG_DEFAULT_GC_DEN,
    oe_num: int = CPG_DEFAULT_OBS_EXP_NUM,
    oe_den: int = CPG_DEFAULT_OBS_EXP_DEN,
) -> list[int]:
    """Return the list of "ON_STREAK" window-start positions.

    A position ``p`` is "ON_STREAK" iff the length-``w`` window starting
    at ``p`` satisfies both thresholds AND ``p`` is part of a contiguous
    run of ``>= w`` such positions. (i.e. a position is ON_STREAK iff it
    will be included in some emitted island.)

    These are the silicon kernel's per-base emit semantics — the kernel
    emits a record per ON_STREAK window-start, and the host merges
    contiguous run-lengths into ``(island_start, island_end)`` tuples.
    """
    if w < 1:
        raise ValueError(f"w must be >= 1; got {w}")
    n = len(seq)
    if n < w:
        return []
    mask = _candidate_mask(seq, w, gc_num, gc_den, oe_num, oe_den)

    # A position is ON_STREAK iff it's part of a contiguous True-run of
    # length >= w in the mask. We find runs and, for each run with
    # length >= w, all positions in the run are ON_STREAK.
    out: list[int] = []
    i = 0
    n_windows = mask.shape[0]
    while i < n_windows:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n_windows and mask[j]:
            j += 1
        run_len = j - i
        if run_len >= w:
            out.extend(range(i, j))
        i = j
    return out


def merge_streak_positions_to_islands(
    positions: list[int],
    w: int = CPG_DEFAULT_W,
) -> list[tuple[int, int]]:
    """Merge a sorted list of ON_STREAK window-start positions into
    ``(island_start, island_end)`` half-open intervals.

    For a contiguous run of window-start positions ``[a, a+1, ..., b]``,
    the emitted island is ``(a, b + w)``: the first base of the first
    window through the last base of the last window (exclusive end).
    """
    if not positions:
        return []
    out: list[tuple[int, int]] = []
    run_start = positions[0]
    prev = positions[0]
    for p in positions[1:]:
        if p == prev + 1:
            prev = p
            continue
        out.append((run_start, prev + w))
        run_start = p
        prev = p
    out.append((run_start, prev + w))
    return out


def merge_candidate_positions_to_islands(
    positions: list[int],
    w: int = CPG_DEFAULT_W,
) -> list[tuple[int, int]]:
    """Merge raw candidate window-start positions into CpG islands.

    Unlike :func:`merge_streak_positions_to_islands`, this helper
    accepts every position whose length-``w`` window passes the
    thresholds. Only contiguous candidate runs with length ``>= w`` are
    emitted as islands.
    """
    if not positions:
        return []
    out: list[tuple[int, int]] = []
    run_start = positions[0]
    prev = positions[0]
    run_len = 1
    for p in positions[1:]:
        if p == prev + 1:
            prev = p
            run_len += 1
            continue
        if run_len >= w:
            out.append((run_start, prev + w))
        run_start = p
        prev = p
        run_len = 1
    if run_len >= w:
        out.append((run_start, prev + w))
    return out


def find_cpg_islands(
    seq: str,
    *,
    w: int = CPG_DEFAULT_W,
    gc_num: int = CPG_DEFAULT_GC_NUM,
    gc_den: int = CPG_DEFAULT_GC_DEN,
    oe_num: int = CPG_DEFAULT_OBS_EXP_NUM,
    oe_den: int = CPG_DEFAULT_OBS_EXP_DEN,
) -> list[tuple[int, int]]:
    """Gardiner-Garden & Frommer (1987) CpG island detector.

    Args:
        seq: ACGT (case-sensitive) string. Non-ACGT bases are treated
            as non-C/non-G (and they cannot participate in a CpG
            dinucleotide).
        w: window length. Default 200 (Gardiner-Garden / UCSC).
        gc_num, gc_den: GC% threshold expressed as a fraction
            ``gc_num / gc_den``. Default ``1/2`` (== 0.5).
        oe_num, oe_den: obs/exp CpG threshold as ``oe_num / oe_den``.
            Default ``3/5`` (== 0.6).

    Returns:
        Sorted list of ``(island_start, island_end)`` tuples
        (half-open) for each maximal run of ``>=w`` consecutive
        candidate positions.
    """
    positions = find_cpg_island_streak_positions(
        seq,
        w=w,
        gc_num=gc_num,
        gc_den=gc_den,
        oe_num=oe_num,
        oe_den=oe_den,
    )
    return merge_streak_positions_to_islands(positions, w=w)


def find_cpg_islands_packed(
    packed_seq: np.ndarray,
    n_bases: int,
    *,
    w: int = CPG_DEFAULT_W,
    gc_num: int = CPG_DEFAULT_GC_NUM,
    gc_den: int = CPG_DEFAULT_GC_DEN,
    oe_num: int = CPG_DEFAULT_OBS_EXP_NUM,
    oe_den: int = CPG_DEFAULT_OBS_EXP_DEN,
) -> list[tuple[int, int]]:
    """Run :func:`find_cpg_islands` against a packed-2-bit fixture.

    Args:
        packed_seq: uint8 ndarray as produced by
            :func:`bionpu.data.kmer_oracle.pack_dna_2bit`.
        n_bases: number of bases packed.
        w, gc_num, gc_den, oe_num, oe_den: as in
            :func:`find_cpg_islands`.

    Returns:
        Same as :func:`find_cpg_islands`.
    """
    from bionpu.data.kmer_oracle import unpack_dna_2bit

    seq = unpack_dna_2bit(packed_seq, n_bases)
    return find_cpg_islands(
        seq,
        w=w,
        gc_num=gc_num,
        gc_den=gc_den,
        oe_num=oe_num,
        oe_den=oe_den,
    )
