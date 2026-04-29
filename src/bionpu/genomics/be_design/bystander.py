# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

"""Bystander-edit enumerator for base editor design.

Per ``PRDs/PRD-crispr-state-of-the-art-roadmap.md`` §3.1: multiple Cs
(CBE) or As (ABE) within the activity window will all be edited. If
the on-target site has a "bystander" C/A within the window, the
resulting protein change is not what you wanted.

The bystander count is the simplest possible signal — it just tallies
the number of additional editable bases in the window beyond the
target. The v0 ranker uses ``bystander_count`` as a soft penalty
(prefer 0; tolerate >0 with a score reduction).

A more sophisticated v1+ would predict per-bystander edit rate
(BE-Hive, DeepBE) and compute the marginal codon-change probability —
but that's bf16-transformer territory and stays on CPU per Track D's
lessons.
"""

from __future__ import annotations

from .pam_variants import BaseEditorSpec
from .window_score import activity_window_slice, PROTOSPACER_LEN

__all__ = [
    "bystander_count",
    "enumerate_bystander_edits",
]


def enumerate_bystander_edits(
    protospacer: str,
    target_pos_0idx: int,
    spec: BaseEditorSpec,
) -> list[int]:
    """Return 0-indexed positions of editable bases in the activity window
    OTHER than ``target_pos_0idx``.

    Args:
        protospacer: 20-nt ACGT string (5' to 3').
        target_pos_0idx: 0-indexed position of the desired-edit base.
        spec: BE variant spec (target base + window bounds).

    Returns:
        Sorted list of positions ``p`` such that:

        * ``lo <= p <= hi`` (inside activity window),
        * ``protospacer[p] == spec.target_base``,
        * ``p != target_pos_0idx``.

    The length of this list IS :func:`bystander_count`.

    Raises:
        ValueError: protospacer length wrong, or ``target_pos_0idx``
            outside the activity window, or the base at
            ``target_pos_0idx`` is not the editable target base.
    """
    if len(protospacer) != PROTOSPACER_LEN:
        raise ValueError(
            f"protospacer must be {PROTOSPACER_LEN} nt; got "
            f"{len(protospacer)}"
        )
    lo, hi = activity_window_slice(spec)
    if not (lo <= target_pos_0idx <= hi):
        raise ValueError(
            f"target_pos_0idx={target_pos_0idx} outside activity window "
            f"[{lo}, {hi}] for BE variant {spec.name}"
        )
    target = spec.target_base.upper()
    if protospacer[target_pos_0idx].upper() != target:
        raise ValueError(
            f"protospacer[{target_pos_0idx}]="
            f"{protospacer[target_pos_0idx]!r} is not the target base "
            f"{target!r} for BE variant {spec.name}"
        )
    return [
        i
        for i in range(lo, hi + 1)
        if i != target_pos_0idx and protospacer[i].upper() == target
    ]


def bystander_count(
    protospacer: str,
    target_pos_0idx: int,
    spec: BaseEditorSpec,
) -> int:
    """Count editable bystander bases in the activity window.

    Convenience wrapper around :func:`enumerate_bystander_edits`. See
    its docstring for argument and error semantics.
    """
    return len(enumerate_bystander_edits(protospacer, target_pos_0idx, spec))
