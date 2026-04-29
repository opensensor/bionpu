# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# Tests for the BE bystander-edit enumerator (Track A v0).

from __future__ import annotations

import pytest

from bionpu.genomics.be_design.bystander import (
    bystander_count,
    enumerate_bystander_edits,
)
from bionpu.genomics.be_design.pam_variants import get_be_spec


def test_no_bystanders_when_only_target_in_window() -> None:
    """A protospacer with exactly ONE C in the BE4max window has 0 bystanders."""
    be4 = get_be_spec("BE4max")
    # Window is [3..7]. Single C at position 5; rest are A.
    proto = "AAAAA" + "C" + "AAAAAAAAAAAAAA"
    assert len(proto) == 20
    assert proto[5] == "C"
    # The target is the C at position 5; no other C in [3..7].
    assert bystander_count(proto, 5, be4) == 0
    assert enumerate_bystander_edits(proto, 5, be4) == []


def test_two_bystanders_in_cbe_window() -> None:
    """Three Cs in window => 2 bystanders if target is one of them."""
    be4 = get_be_spec("BE4max")
    # Window [3..7]. Cs at 4, 5, 7.
    proto = "AAA" + "A" + "C" + "C" + "A" + "C" + "AAAAAAAAAAAA"
    assert len(proto) == 20
    assert proto[4] == "C" and proto[5] == "C" and proto[7] == "C"
    # If we pick C at pos 5 as the target, bystanders are at 4 and 7.
    assert bystander_count(proto, 5, be4) == 2
    assert enumerate_bystander_edits(proto, 5, be4) == [4, 7]
    # If we pick C at pos 4, bystanders are at 5 and 7.
    assert enumerate_bystander_edits(proto, 4, be4) == [5, 7]


def test_bystander_outside_window_does_not_count() -> None:
    """Cs outside the activity window are NOT bystanders."""
    be4 = get_be_spec("BE4max")
    # Window [3..7]. Cs at 0 (outside), 5 (in window, target), 10 (outside),
    # 15 (outside).
    chars = ["A"] * 20
    chars[0] = "C"
    chars[5] = "C"
    chars[10] = "C"
    chars[15] = "C"
    proto = "".join(chars)
    assert len(proto) == 20
    assert proto[0] == "C" and proto[5] == "C" and proto[10] == "C" and proto[15] == "C"
    # No other C in window [3..7] besides position 5.
    assert bystander_count(proto, 5, be4) == 0


def test_abe_bystander_count() -> None:
    """ABE7.10 (window [3..6]) picks As, not Cs."""
    abe = get_be_spec("ABE7.10")
    # Window [3..6]. As at 3, 4, 6; Cs ignored.
    proto = "CCC" + "A" + "A" + "C" + "A" + "CCCCCCCCCCCCC"
    assert len(proto) == 20
    assert proto[3] == "A" and proto[4] == "A" and proto[6] == "A"
    # Pick A at 3 as target; bystanders at 4 and 6.
    assert bystander_count(proto, 3, abe) == 2
    assert enumerate_bystander_edits(proto, 3, abe) == [4, 6]


def test_target_outside_window_raises() -> None:
    """target_pos_0idx must be inside the activity window."""
    be4 = get_be_spec("BE4max")
    proto = "A" * 20
    # Position 0 is outside window [3..7].
    with pytest.raises(ValueError, match="outside activity window"):
        bystander_count(proto, 0, be4)


def test_target_base_mismatch_raises() -> None:
    """If protospacer[target_pos] is not the editable target base, raise."""
    be4 = get_be_spec("BE4max")  # target_base = C
    # Position 5 is in window but the base at 5 is A, not C.
    proto = "A" * 20
    with pytest.raises(ValueError, match="not the target base"):
        bystander_count(proto, 5, be4)
