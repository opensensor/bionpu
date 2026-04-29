# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

"""Edit-window scoring for base editor design.

Per ``PRDs/PRD-crispr-state-of-the-art-roadmap.md`` §3.1: each base
editor has an activity window within the protospacer (typically nt
4-8 from the PAM-distal end for CBE; nt 4-7 for ABE). For high editing
efficiency, the desired edit must land in this window.

For a 20-nt protospacer where the PAM is 3' of it, "PAM-distal" means
the 5' end of the protospacer. Counting nt 4 (1-indexed) from the
PAM-distal end means the 4th base from the 5' end, i.e. position 3
(0-indexed). The CBE window nt 4-8 covers protospacer positions [3..7]
(0-indexed) inclusive.

This module is pure-Python; the v0 ranker uses the simple boolean
"target base in window?" + bystander count combination. A future
window-scoring NPU layer (per the PRD's §3.1 "be_window_score") would
replace this with a small bf16 MLP — but that's Phase 2 / 3 work
deferred per Track D's "bf16 transformers stay on CPU" lesson.
"""

from __future__ import annotations

from typing import Final

from .pam_variants import BaseEditorSpec

__all__ = [
    "AC_WINDOW_BOUNDS_0IDX",
    "PROTOSPACER_LEN",
    "activity_window_slice",
    "activity_window_target_positions",
    "target_in_window",
]

#: Pinned protospacer length (Cas9 standard).
PROTOSPACER_LEN: Final[int] = 20


#: 0-indexed inclusive activity-window bounds per BE variant for a
#: 20-nt protospacer.
#:
#:   CBE (BE4max, etc.) — nt 4-8 from PAM-distal -> [3..7]
#:   ABE7.10 (etc.)     — nt 4-7 from PAM-distal -> [3..6]
#:   ABE8e (wider)      — nt 4-8 from PAM-distal -> [3..7]
AC_WINDOW_BOUNDS_0IDX: Final[dict[str, tuple[int, int]]] = {
    "BE4max": (3, 7),
    "ABE7.10": (3, 6),
    "ABE8e": (3, 7),
}


def activity_window_slice(spec: BaseEditorSpec) -> tuple[int, int]:
    """Return ``(lo_0idx_inclusive, hi_0idx_inclusive)`` for a 20-nt protospacer.

    The PAM is 3' of the protospacer. "nt N from PAM-distal end"
    (1-indexed) maps to position ``N - 1`` (0-indexed) counted from the
    5' end.

    Args:
        spec: BE variant spec.

    Returns:
        ``(lo, hi)`` 0-indexed inclusive bounds within a 20-nt protospacer.
    """
    lo = spec.window_lo_pam_distal_1idx - 1
    hi = spec.window_hi_pam_distal_1idx - 1
    if not (0 <= lo <= hi < PROTOSPACER_LEN):
        raise ValueError(
            f"window bounds [{spec.window_lo_pam_distal_1idx}, "
            f"{spec.window_hi_pam_distal_1idx}] from BE spec {spec.name} "
            f"out of range for 20-nt protospacer"
        )
    return lo, hi


def activity_window_target_positions(
    protospacer: str,
    spec: BaseEditorSpec,
) -> list[int]:
    """Return the 0-indexed positions of the target base in the activity window.

    Args:
        protospacer: 20-nt ACGT string (5' to 3').
        spec: BE variant spec (target base + window bounds).

    Returns:
        Sorted list of 0-indexed positions where ``protospacer[pos] ==
        spec.target_base`` AND ``pos`` is within the activity window.
    """
    if len(protospacer) != PROTOSPACER_LEN:
        raise ValueError(
            f"protospacer must be {PROTOSPACER_LEN} nt; got "
            f"{len(protospacer)}"
        )
    lo, hi = activity_window_slice(spec)
    target = spec.target_base.upper()
    return [
        i
        for i in range(lo, hi + 1)
        if protospacer[i].upper() == target
    ]


def target_in_window(
    protospacer: str,
    target_pos_0idx: int,
    spec: BaseEditorSpec,
) -> bool:
    """True iff ``target_pos_0idx`` falls within the activity window.

    Args:
        protospacer: 20-nt ACGT string (5' to 3').
        target_pos_0idx: 0-indexed position to check.
        spec: BE variant spec.

    Returns:
        True iff ``target_pos_0idx`` is within ``[lo, hi]`` inclusive.
        Does NOT verify that the base at that position equals
        ``spec.target_base`` — that's the caller's job (this is a
        pure window-membership query).
    """
    if len(protospacer) != PROTOSPACER_LEN:
        raise ValueError(
            f"protospacer must be {PROTOSPACER_LEN} nt; got "
            f"{len(protospacer)}"
        )
    lo, hi = activity_window_slice(spec)
    return lo <= target_pos_0idx <= hi
