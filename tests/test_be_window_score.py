# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# Tests for the base editor activity-window scoring layer (Track A v0).

from __future__ import annotations

import pytest

from bionpu.genomics.be_design.pam_variants import get_be_spec
from bionpu.genomics.be_design.window_score import (
    AC_WINDOW_BOUNDS_0IDX,
    PROTOSPACER_LEN,
    activity_window_slice,
    activity_window_target_positions,
    target_in_window,
)


def test_window_bounds_be4max_cbe() -> None:
    """BE4max (CBE): nt 4-8 from PAM-distal => positions [3..7] (0-idx)."""
    spec = get_be_spec("BE4max")
    lo, hi = activity_window_slice(spec)
    assert (lo, hi) == (3, 7)
    assert (lo, hi) == AC_WINDOW_BOUNDS_0IDX["BE4max"]


def test_window_bounds_abe710_abe() -> None:
    """ABE7.10: nt 4-7 from PAM-distal => positions [3..6] (0-idx)."""
    spec = get_be_spec("ABE7.10")
    lo, hi = activity_window_slice(spec)
    assert (lo, hi) == (3, 6)
    assert (lo, hi) == AC_WINDOW_BOUNDS_0IDX["ABE7.10"]


def test_window_bounds_abe8e() -> None:
    """ABE8e (Phase 2): nt 4-8 wider window."""
    spec = get_be_spec("ABE8e")
    assert activity_window_slice(spec) == (3, 7)


def test_target_in_window_membership() -> None:
    """target_in_window is a pure positional check (not base-aware)."""
    be4 = get_be_spec("BE4max")
    proto = "A" * 20
    # Inside window (3..7)
    assert target_in_window(proto, 3, be4) is True
    assert target_in_window(proto, 5, be4) is True
    assert target_in_window(proto, 7, be4) is True
    # Outside window
    assert target_in_window(proto, 0, be4) is False
    assert target_in_window(proto, 2, be4) is False
    assert target_in_window(proto, 8, be4) is False
    assert target_in_window(proto, 19, be4) is False


def test_activity_window_targets_cbe() -> None:
    """CBE picks Cs in the activity window."""
    be4 = get_be_spec("BE4max")
    # Positions 5 and 7 are inside the CBE window [3..7]; position 0 is outside.
    proto = "C" + "AAAAA" + "C" + "A" + "C" + "AAAAAAAAAAA"  # pos 0=C, 6=C, 8=C
    assert len(proto) == PROTOSPACER_LEN
    targets = activity_window_target_positions(proto, be4)
    # Only the C at position 6 is inside [3..7].
    assert targets == [6]


def test_activity_window_targets_abe() -> None:
    """ABE7.10 picks As in nt 4-7 window."""
    abe = get_be_spec("ABE7.10")
    # Place A at positions 3 (in window), 5 (in window), 8 (outside).
    proto = "C" * 3 + "A" + "C" + "A" + "CC" + "A" + "C" * 11
    assert len(proto) == PROTOSPACER_LEN
    assert proto[3] == "A" and proto[5] == "A" and proto[8] == "A"
    targets = activity_window_target_positions(proto, abe)
    # Window is [3..6]; A at 3, 5; A at 8 is outside.
    assert targets == [3, 5]


def test_protospacer_len_validation() -> None:
    """Wrong-length protospacers raise ValueError."""
    be4 = get_be_spec("BE4max")
    with pytest.raises(ValueError, match="must be 20 nt"):
        activity_window_target_positions("AC" * 5, be4)
    with pytest.raises(ValueError, match="must be 20 nt"):
        target_in_window("AC" * 5, 3, be4)
