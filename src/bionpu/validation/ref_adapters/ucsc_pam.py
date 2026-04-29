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

"""Naive Python NGG-finder oracle for the bionpu PAM-scan cross-check.

This is the in-tree "UCSC-grade" oracle: a deterministic regex-style
scan over an input sequence that emits every NGG position. Used as
the reference for ``bionpu validate crispr design --reference ucsc-pam``.

Always installable — pure Python, no external deps.
"""

from __future__ import annotations


__all__ = [
    "find_ngg_positions",
    "find_ngg_positions_both_strands",
    "ucsc_pam_installed",
]


def ucsc_pam_installed() -> bool:
    """Always True — pure-Python oracle has no external deps."""
    return True


def _revcomp(seq: str) -> str:
    table = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(table)[::-1]


def find_ngg_positions(seq: str) -> list[int]:
    """Return forward-strand NGG positions (0-based start of the 3-mer's first base).

    The convention matches :func:`bionpu.validation.agreement._scan_pam_with_bionpu_cpu`:
    a position ``i`` is a hit when ``seq[i+1:i+3] == "GG"`` and ``seq[i]``
    is in ``ACGT``. (i.e. position ``i`` is the start of a 3-mer
    [N, G, G] where N is the wildcard base.)
    """
    seq_u = seq.upper()
    out: list[int] = []
    for i in range(len(seq_u) - 2):
        if seq_u[i + 1] == "G" and seq_u[i + 2] == "G" and seq_u[i] in "ACGT":
            out.append(i)
    return out


def find_ngg_positions_both_strands(seq: str) -> dict[str, list[int]]:
    """Forward + reverse strand NGG positions for completeness."""
    fwd = find_ngg_positions(seq)
    rev_seq = _revcomp(seq)
    rev = find_ngg_positions(rev_seq)
    # Map reverse-strand positions back to forward-strand coords.
    L = len(seq)
    rev_on_fwd = [L - 3 - p for p in rev if 0 <= L - 3 - p < L]
    return {"forward": fwd, "reverse": sorted(rev_on_fwd)}
