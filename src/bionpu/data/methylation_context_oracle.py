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

"""Reference oracle for cytosine methylation-context scanning.

The primitive classifies methylatable cytosines into the standard
bisulfite/ONT-modbase contexts:

* ``CG``: C followed by G.
* ``CHG``: C followed by A/C/T then G.
* ``CHH``: C followed by A/C/T then A/C/T.

Both strands are emitted. A minus-strand cytosine appears as a ``G`` in
the forward reference, so the scanner uses the two preceding forward
bases to derive the cytosine-strand motif.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable

from bionpu.data.kmer_oracle import unpack_dna_2bit

__all__ = [
    "MethylationContextHit",
    "find_methylation_contexts",
    "find_methylation_contexts_packed",
]

_ACGT: Final[frozenset[str]] = frozenset("ACGT")
_H: Final[frozenset[str]] = frozenset("ACT")
_RC: Final[dict[int, int]] = str.maketrans("ACGTacgt", "TGCAtgca")


@dataclass(frozen=True, slots=True)
class MethylationContextHit:
    """Sparse methylation-context emit.

    Attributes:
        pos: 0-based reference position of the cytosine on ``+`` or the
            cytosine-equivalent ``G`` on ``-``.
        strand: ``"+"`` or ``"-"``.
        context: one of ``"CG"``, ``"CHG"``, or ``"CHH"``.
        motif: cytosine-strand motif, oriented 5' -> 3'. ``CG`` emits a
            2-base motif; ``CHG``/``CHH`` emit a 3-base motif.
    """

    pos: int
    strand: str
    context: str
    motif: str


def _classify_cytosine_motif(motif: str) -> str | None:
    """Return the methylation context for an oriented cytosine motif."""
    if len(motif) < 2 or motif[0] != "C":
        return None
    b1 = motif[1]
    if b1 == "G":
        return "CG"
    if b1 not in _H or len(motif) < 3:
        return None
    b2 = motif[2]
    if b2 == "G":
        return "CHG"
    if b2 in _H:
        return "CHH"
    return None


def find_methylation_contexts(seq: str) -> list[MethylationContextHit]:
    """Classify all CG/CHG/CHH cytosine contexts in ``seq``.

    Non-ACGT bases break local context. Hits that would need bases
    outside the sequence are skipped.
    """
    s = seq.upper()
    hits: list[MethylationContextHit] = []
    n = len(s)

    for i, base in enumerate(s):
        if base == "C":
            if i + 1 >= n or s[i + 1] not in _ACGT:
                continue
            if s[i + 1] == "G":
                hits.append(MethylationContextHit(i, "+", "CG", "CG"))
                continue
            if i + 2 >= n or s[i + 2] not in _ACGT:
                continue
            motif = s[i : i + 3]
            context = _classify_cytosine_motif(motif)
            if context is not None:
                hits.append(MethylationContextHit(i, "+", context, motif))
        elif base == "G":
            if i < 1 or s[i - 1] not in _ACGT:
                continue
            b1 = s[i - 1].translate(_RC)
            if b1 == "G":
                hits.append(MethylationContextHit(i, "-", "CG", "CG"))
                continue
            if i < 2 or s[i - 2] not in _ACGT:
                continue
            motif = "C" + b1 + s[i - 2].translate(_RC)
            context = _classify_cytosine_motif(motif)
            if context is not None:
                hits.append(MethylationContextHit(i, "-", context, motif))

    return hits


def find_methylation_contexts_packed(
    packed_seq: Iterable[int],
    *,
    n_bases: int,
) -> list[MethylationContextHit]:
    """Run :func:`find_methylation_contexts` against packed 2-bit DNA."""
    seq = unpack_dna_2bit(packed_seq, n_bases)
    return find_methylation_contexts(seq)
