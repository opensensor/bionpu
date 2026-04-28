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

"""TDD tests for the CpG island oracle.

Mirrors :file:`tests/test_primer_oracle.py` for the v0 cpg_island
kernel. Imports of :func:`find_cpg_islands` should succeed once the
oracle module is in place; these tests then become the regression lock
for the silicon byte-equal harness.

The fixed-point integer formulation of the thresholds (used by the
silicon kernel) is exercised here so any divergence between the
oracle's float-equivalent statement and the fixed-point gate would
fail before silicon.
"""

from __future__ import annotations

import numpy as np
import pytest

from bionpu.data.cpg_oracle import (
    CPG_DEFAULT_GC_DEN,
    CPG_DEFAULT_GC_NUM,
    CPG_DEFAULT_OBS_EXP_DEN,
    CPG_DEFAULT_OBS_EXP_NUM,
    CPG_DEFAULT_W,
    find_cpg_island_streak_positions,
    find_cpg_islands,
    find_cpg_islands_packed,
    merge_candidate_positions_to_islands,
    merge_streak_positions_to_islands,
)


# --------------------------------------------------------------------------- #
# Test 1: an all-A sequence has no CpG islands.
# --------------------------------------------------------------------------- #


def test_all_a_no_islands() -> None:
    seq = "A" * 1000
    assert find_cpg_islands(seq) == []
    assert find_cpg_island_streak_positions(seq) == []


# --------------------------------------------------------------------------- #
# Test 2: a CG-repeat sequence is one big island.
# --------------------------------------------------------------------------- #


def test_all_cg_one_island() -> None:
    """CG repeats hit GC%=1.0 and obs/exp ~= 1.0 (well above thresholds).

    The whole prefix that produces full-w windows is on-streak; the
    emitted island spans positions [0, n).
    """
    n = 1000
    seq = "CG" * (n // 2)  # length n, n even
    islands = find_cpg_islands(seq)
    assert len(islands) == 1, f"expected 1 island; got {islands!r}"
    start, end = islands[0]
    # Every window-start position 0..(n-w) is on-streak. Merged island:
    # (0, (n - w) + w) == (0, n).
    assert start == 0
    assert end == n


# --------------------------------------------------------------------------- #
# Test 3: a known CpG-rich region embedded in A-padding.
# --------------------------------------------------------------------------- #


def test_mixed_sequence_island_position() -> None:
    """A CpG-rich block of length L > W embedded in A-padding produces
    exactly one island.

    With W=200 and a 400-bp CG-repeat block at position 500, the
    leftmost window that satisfies the thresholds starts before the
    block is fully entered (since once enough CGs are in the window the
    rolling counters cross the thresholds). The rightmost candidate
    window starts at position 800 (last fully-rich window), so the
    merged island spans [400, 1000) — first candidate's start through
    last candidate's end (= 800 + 200).
    """
    pad = "A" * 500
    rich = "CG" * 200  # 400 bp; >= W
    seq = pad + rich + pad
    islands = find_cpg_islands(seq)
    assert islands == [(400, 1000)], f"got {islands!r}"


# --------------------------------------------------------------------------- #
# Test 4: short rich block (length < W) emits NO island.
# --------------------------------------------------------------------------- #


def test_short_rich_block_no_island() -> None:
    """A CG-repeat block of length L < W cannot produce a single
    candidate window (no full-w window fits inside it).
    """
    rich = "CG" * 80  # 160 bp < 200
    seq = "A" * 500 + rich + "A" * 500
    assert find_cpg_islands(seq) == []


# --------------------------------------------------------------------------- #
# Test 5: empty / shorter-than-W input returns no islands.
# --------------------------------------------------------------------------- #


def test_empty_and_short_input() -> None:
    assert find_cpg_islands("") == []
    assert find_cpg_islands("A" * 50) == []
    assert find_cpg_islands("CG" * 99) == []  # 198 < 200


# --------------------------------------------------------------------------- #
# Test 6: streak -> islands merge round-trip.
# --------------------------------------------------------------------------- #


def test_streak_merge_round_trip() -> None:
    """``merge_streak_positions_to_islands`` over the streak positions
    equals the islands returned by ``find_cpg_islands``.
    """
    pad = "A" * 300
    rich = "CG" * 250  # 500 bp; well over W
    seq = pad + rich + pad + rich + pad
    streaks = find_cpg_island_streak_positions(seq)
    merged = merge_streak_positions_to_islands(streaks, w=CPG_DEFAULT_W)
    assert merged == find_cpg_islands(seq)
    # Two distinct rich blocks → two islands.
    assert len(merged) == 2


def test_candidate_merge_applies_min_run_length() -> None:
    candidates = list(range(10, 10 + CPG_DEFAULT_W - 1))
    assert merge_candidate_positions_to_islands(candidates) == []

    candidates.append(10 + CPG_DEFAULT_W - 1)
    assert merge_candidate_positions_to_islands(candidates) == [
        (10, 10 + CPG_DEFAULT_W - 1 + CPG_DEFAULT_W)
    ]


# --------------------------------------------------------------------------- #
# Test 7: packed-path equivalent to string-path.
# --------------------------------------------------------------------------- #


def test_packed_path_equivalent_to_string_path() -> None:
    from bionpu.data.kmer_oracle import pack_dna_2bit

    seq = (
        ("A" * 320)
        + ("CG" * 220)  # 440 bp rich
        + ("A" * 320)
    )
    while len(seq) % 4 != 0:
        seq += "A"
    packed = pack_dna_2bit(seq)
    str_islands = find_cpg_islands(seq)
    packed_islands = find_cpg_islands_packed(packed, n_bases=len(seq))
    assert str_islands == packed_islands
    assert len(str_islands) == 1


# --------------------------------------------------------------------------- #
# Test 8: smoke regression-lock — pinned ON_STREAK count for a
# deterministic mixed fixture (locks behaviour for the silicon byte-
# equal harness).
# --------------------------------------------------------------------------- #


def test_smoke_streak_count_pinned() -> None:
    """Locks the ON_STREAK count for a pinned mixed fixture so any
    drift in the threshold math (or window-edge handling) is caught
    by this regression test before silicon byte-equal runs.

    Two CG-rich blocks (200 bp + 400 bp) embedded in A-padding both
    produce candidate-window runs whose length >= W (the rolling
    thresholds clear well before the block is fully entered and
    persist past the trailing edge), so both blocks emit one island
    each. The pinned counts below are computed by the oracle on the
    fixture and are the regression contract for the silicon kernel.
    """
    block_a_len = CPG_DEFAULT_W            # 200
    block_b_len = 2 * CPG_DEFAULT_W        # 400
    seq = (
        ("A" * 500)
        + ("CG" * (block_a_len // 2))      # block A: 200 bp
        + ("A" * 500)
        + ("CG" * (block_b_len // 2))      # block B: 400 bp
        + ("A" * 500)
    )

    streaks = find_cpg_island_streak_positions(seq)
    assert len(streaks) == 602, (
        f"expected 602 ON_STREAK positions (block A: 201, block B: "
        f"401); got {len(streaks)}"
    )
    islands = find_cpg_islands(seq)
    # Block A run [400, 600] -> island (400, 800).
    # Block B run [1100, 1500] -> island (1100, 1700).
    assert islands == [(400, 800), (1100, 1700)], f"got {islands!r}"


# --------------------------------------------------------------------------- #
# Test 9: default thresholds match the documented Gardiner-Garden values.
# --------------------------------------------------------------------------- #


def test_threshold_defaults() -> None:
    assert CPG_DEFAULT_W == 200
    assert CPG_DEFAULT_GC_NUM / CPG_DEFAULT_GC_DEN == pytest.approx(0.5)
    assert CPG_DEFAULT_OBS_EXP_NUM / CPG_DEFAULT_OBS_EXP_DEN == pytest.approx(0.6)
