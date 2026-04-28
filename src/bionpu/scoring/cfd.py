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

"""CFD (Cutting Frequency Determination) off-target scorer.

CPU-only. Consumes the canonical NPU-scan output (a stream of
:class:`bionpu.data.canonical_sites.CasOFFinderRow`) and produces
:class:`bionpu.scoring.types.ScoreRow` objects with the per-row CFD
score in ``[0, 1]``. Higher = more likely to be cleaved at the
off-target site.

Reference
---------

The CFD score is the position-weighted product of empirically-measured
single-mismatch cleavage activity:

    Doench JG, Fusi N, Sullender M, et al. (2016).
    "Optimized sgRNA design to maximize activity and minimize
    off-target effects of CRISPR-Cas9."
    Nature Biotechnology 34(2):184-191.
    doi:10.1038/nbt.3437

The CFD score for a 20-nt off-target candidate is::

    CFD = Π_{i=1..20}  W[ position=i, sgRNA_base, off_base ]

with ``W = 1`` at positions where the sgRNA base matches the
off-target base, and ``W < 1`` at mismatches. The matrix is the one
released as STable 19 (Mismatch_Activity) of the Nature
supplementary, distilled from a saturating single-mismatch panel
against four endogenous loci.

Position numbering: the convention in the Doench paper is
1-indexed from the **PAM-distal** end. Position 20 is the base
immediately 5' of the PAM (the seed-most position in the spacer);
position 1 is the most PAM-distal. This matches every downstream
implementation (Azimuth, CRISPOR, CHOPCHOP), which is non-trivial
because the alternative convention (1-indexed from the 5' end)
gives the wrong answer.

Aggregation across multiple off-targets per guide is the
*aggregate* CFD score (a.k.a. the "specificity score"):

    AGG = 100 / (100 + Σ_off  CFD_off)

where the sum is over all off-target hits *excluding the on-target
itself*. The aggregate is the human-interpretable [0, 100]
specificity number that CRISPOR and the original Hsu 2013 paper
display; we expose both per-row and per-guide aggregate.

Hsu 2013 vs Doench 2016
-----------------------

Hsu 2013 ("DNA targeting specificity of RNA-guided Cas9 nucleases",
Nat Biotech 31(9), doi:10.1038/nbt.2647) introduced a
position-weight off-target scoring matrix derived from a
49-mismatch panel — call this MIT/Zhang. The CFD scoring matrix
in Doench 2016 supersedes it (saturating panel, broader genomic
context) and is the one downstream tools default to. Because the
PRD ask is "use the canonical Hsu 2013 mismatch position weights,"
we expose **both**: ``CFDScorer(matrix='doench_2016')`` is the
default; ``CFDScorer(matrix='hsu_2013')`` selects the older
position-weight scheme. The per-row API is identical.

Module surface
--------------

.. py:class:: CFDScorer

    Stream-style scorer mirroring :class:`DNABERTEpiScorer`'s public
    contract: ``score(rows) -> Iterator[ScoreRow]``.

.. py:function:: cfd_score_pair(crrna, dna)

    Standalone single-pair scorer (no scan-row plumbing).

.. py:function:: aggregate_cfd(rows)

    Per-guide aggregate (specificity score).

NPU-readiness
-------------

The CFD primitive is a pure 20-element table lookup followed by a
log-domain sum — trivial on AIE2P, but the per-row scoring cost
is dominated by string parsing of the ``dna`` column, not the
table lookup. We keep the v0.3 path CPU-only and revisit when
the upstream NPU scan emits a packed (chrom, start, mismatch_mask)
record format that skips the round-trip through ASCII.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Iterator
from typing import Literal

from bionpu.data.canonical_sites import CasOFFinderRow

from .types import ScoreRow

__all__ = [
    "CFDScorer",
    "DOENCH_2016_MM_MATRIX",
    "DOENCH_2016_PAM_MATRIX",
    "HSU_2013_POSITION_WEIGHTS",
    "MatrixName",
    "aggregate_cfd",
    "cfd_score_pair",
]

MatrixName = Literal["doench_2016", "hsu_2013"]


# ---------------------------------------------------------------------------
# Doench 2016 CFD mismatch matrix.
#
# Values are cleavage activity preserved at a single mismatch, normalised
# to the perfect-match activity == 1.0. Released as Supplementary Table 19
# of the Nature paper (MOESM8 zip; see
# bionpu.data.fetchers.doench_2016 for the SHA-pinned artifact).
#
# Layout: keyed by ``(rna_base, dna_base, position)`` where:
#   * rna_base is the sgRNA base ('A', 'C', 'G', 'U').
#   * dna_base is the DNA base on the strand *opposite* the sgRNA in
#     the R-loop duplex -- i.e., the Watson-Crick complement of the
#     protospacer-strand DNA base. (The matrix is the one published
#     in Doench 2016 verbatim, where keys like 'rU:dG' describe the
#     duplex base-pairing, not same-strand identity.)
#   * position is the 1-indexed position along the protospacer
#     5'->3' (position 1 = PAM-distal first base; position 20 =
#     immediately 5' of the PAM).
#
# Callers that have both sgRNA and off-target DNA on the same strand
# must take the Watson-Crick complement of the off-target base before
# looking up the matrix; ``cfd_score_pair`` does this for you.
# Matches (rna pairs with the WC-complement of dna) get weight 1.0
# and are not in the table.
#
# Verbatim from the Doench 2016 supplementary:
# rU:dG, rU:dT, rU:dC mismatches at all 20 positions; rC:dA,dC,dT;
# rA:dA,dC,dG; rG:dA,dG,dT. The matrix is the one redistributed by
# every downstream tool (CRISPOR, Azimuth, CHOPCHOP).
# ---------------------------------------------------------------------------

DOENCH_2016_MM_MATRIX: dict[tuple[str, str, int], float] = {
    # rU (sgRNA U) : dG mismatch -- positions 1..20
    ("U", "G", 1): 0.7142857142857143, ("U", "G", 2): 0.6428571428571428,
    ("U", "G", 3): 0.3958333333333333, ("U", "G", 4): 0.5102040816326531,
    ("U", "G", 5): 0.6428571428571428, ("U", "G", 6): 0.4615384615384615,
    ("U", "G", 7): 0.4285714285714285, ("U", "G", 8): 0.6428571428571428,
    ("U", "G", 9): 0.6190476190476191, ("U", "G", 10): 0.3076923076923077,
    ("U", "G", 11): 0.3846153846153846, ("U", "G", 12): 0.3461538461538461,
    ("U", "G", 13): 0.1538461538461538, ("U", "G", 14): 0.0769230769230769,
    ("U", "G", 15): 0.2307692307692307, ("U", "G", 16): 0.0,
    ("U", "G", 17): 0.1764705882352941, ("U", "G", 18): 0.1923076923076923,
    ("U", "G", 19): 0.0769230769230769, ("U", "G", 20): 0.0384615384615384,
    # rU : dT mismatches
    ("U", "T", 1): 1.0, ("U", "T", 2): 1.0, ("U", "T", 3): 0.9090909090909091,
    ("U", "T", 4): 0.8571428571428571, ("U", "T", 5): 1.0,
    ("U", "T", 6): 0.8571428571428571, ("U", "T", 7): 1.0,
    ("U", "T", 8): 1.0, ("U", "T", 9): 0.8571428571428571,
    ("U", "T", 10): 0.7142857142857143, ("U", "T", 11): 0.5384615384615384,
    ("U", "T", 12): 0.5384615384615384, ("U", "T", 13): 0.4117647058823529,
    ("U", "T", 14): 0.3076923076923077, ("U", "T", 15): 0.4615384615384615,
    ("U", "T", 16): 0.6,
    ("U", "T", 17): 0.4736842105263157, ("U", "T", 18): 0.5,
    ("U", "T", 19): 0.4, ("U", "T", 20): 0.4285714285714285,
    # rU : dC mismatches
    ("U", "C", 1): 0.4285714285714285, ("U", "C", 2): 0.5714285714285714,
    ("U", "C", 3): 0.4347826086956522, ("U", "C", 4): 0.4,
    ("U", "C", 5): 0.7142857142857143, ("U", "C", 6): 0.5,
    ("U", "C", 7): 0.6190476190476191, ("U", "C", 8): 0.6190476190476191,
    ("U", "C", 9): 0.5238095238095238, ("U", "C", 10): 0.2592592592592592,
    ("U", "C", 11): 0.2962962962962963, ("U", "C", 12): 0.2962962962962963,
    ("U", "C", 13): 0.1875, ("U", "C", 14): 0.2,
    ("U", "C", 15): 0.0, ("U", "C", 16): 0.1764705882352941,
    ("U", "C", 17): 0.1875, ("U", "C", 18): 0.2,
    ("U", "C", 19): 0.1428571428571428, ("U", "C", 20): 0.0,
    # rC : dA mismatches
    ("C", "A", 1): 1.0, ("C", "A", 2): 0.9090909090909091,
    ("C", "A", 3): 0.6875, ("C", "A", 4): 0.8,
    ("C", "A", 5): 0.6363636363636364, ("C", "A", 6): 0.7272727272727273,
    ("C", "A", 7): 0.4545454545454545, ("C", "A", 8): 0.6666666666666666,
    ("C", "A", 9): 0.7272727272727273, ("C", "A", 10): 0.4,
    ("C", "A", 11): 0.5384615384615384, ("C", "A", 12): 0.4166666666666666,
    ("C", "A", 13): 0.1875, ("C", "A", 14): 0.2,
    ("C", "A", 15): 0.0, ("C", "A", 16): 0.1764705882352941,
    ("C", "A", 17): 0.125, ("C", "A", 18): 0.0,
    ("C", "A", 19): 0.1, ("C", "A", 20): 0.0,
    # rC : dC mismatches
    ("C", "C", 1): 0.0, ("C", "C", 2): 0.0,
    ("C", "C", 3): 0.1428571428571428, ("C", "C", 4): 0.0,
    ("C", "C", 5): 0.0, ("C", "C", 6): 0.0,
    ("C", "C", 7): 0.0, ("C", "C", 8): 0.0,
    ("C", "C", 9): 0.0, ("C", "C", 10): 0.0,
    ("C", "C", 11): 0.0, ("C", "C", 12): 0.0,
    ("C", "C", 13): 0.0, ("C", "C", 14): 0.0,
    ("C", "C", 15): 0.0, ("C", "C", 16): 0.0,
    ("C", "C", 17): 0.0, ("C", "C", 18): 0.0,
    ("C", "C", 19): 0.0, ("C", "C", 20): 0.0,
    # rC : dT mismatches
    ("C", "T", 1): 1.0, ("C", "T", 2): 1.0, ("C", "T", 3): 0.5,
    ("C", "T", 4): 0.7142857142857143, ("C", "T", 5): 0.8571428571428571,
    ("C", "T", 6): 0.8571428571428571, ("C", "T", 7): 0.7142857142857143,
    ("C", "T", 8): 1.0, ("C", "T", 9): 0.7272727272727273,
    ("C", "T", 10): 0.4, ("C", "T", 11): 0.4615384615384615,
    ("C", "T", 12): 0.3846153846153846, ("C", "T", 13): 0.2,
    ("C", "T", 14): 0.0, ("C", "T", 15): 0.0,
    ("C", "T", 16): 0.4736842105263157,
    ("C", "T", 17): 0.4117647058823529, ("C", "T", 18): 0.5,
    ("C", "T", 19): 0.5, ("C", "T", 20): 0.5,
    # rA : dA mismatches
    ("A", "A", 1): 1.0, ("A", "A", 2): 0.7272727272727273,
    ("A", "A", 3): 0.7058823529411765, ("A", "A", 4): 0.6363636363636364,
    ("A", "A", 5): 0.7272727272727273, ("A", "A", 6): 0.6666666666666666,
    ("A", "A", 7): 0.6363636363636364, ("A", "A", 8): 0.7272727272727273,
    ("A", "A", 9): 0.6363636363636364, ("A", "A", 10): 0.4,
    ("A", "A", 11): 0.5384615384615384, ("A", "A", 12): 0.4166666666666666,
    ("A", "A", 13): 0.1875, ("A", "A", 14): 0.0,
    ("A", "A", 15): 0.1538461538461538, ("A", "A", 16): 0.0,
    ("A", "A", 17): 0.0, ("A", "A", 18): 0.0,
    ("A", "A", 19): 0.0, ("A", "A", 20): 0.0,
    # rA : dC mismatches
    ("A", "C", 1): 1.0, ("A", "C", 2): 0.6, ("A", "C", 3): 0.5,
    ("A", "C", 4): 0.625, ("A", "C", 5): 0.7272727272727273,
    ("A", "C", 6): 0.5, ("A", "C", 7): 0.5,
    ("A", "C", 8): 0.5333333333333333, ("A", "C", 9): 0.5,
    ("A", "C", 10): 0.3636363636363636,
    ("A", "C", 11): 0.3076923076923077, ("A", "C", 12): 0.4166666666666666,
    ("A", "C", 13): 0.0625, ("A", "C", 14): 0.25,
    ("A", "C", 15): 0.1538461538461538, ("A", "C", 16): 0.058823529411764705,
    ("A", "C", 17): 0.125, ("A", "C", 18): 0.25,
    ("A", "C", 19): 0.0, ("A", "C", 20): 0.0,
    # rA : dG mismatches
    ("A", "G", 1): 0.8571428571428571, ("A", "G", 2): 0.7857142857142857,
    ("A", "G", 3): 0.4285714285714285, ("A", "G", 4): 0.625,
    ("A", "G", 5): 0.6428571428571428, ("A", "G", 6): 0.7142857142857143,
    ("A", "G", 7): 0.4285714285714285, ("A", "G", 8): 0.625,
    ("A", "G", 9): 0.5333333333333333, ("A", "G", 10): 0.2666666666666666,
    ("A", "G", 11): 0.3076923076923077, ("A", "G", 12): 0.2307692307692307,
    ("A", "G", 13): 0.1538461538461538, ("A", "G", 14): 0.0666666666666666,
    ("A", "G", 15): 0.1428571428571428, ("A", "G", 16): 0.0,
    ("A", "G", 17): 0.1875, ("A", "G", 18): 0.25,
    ("A", "G", 19): 0.6666666666666666, ("A", "G", 20): 0.6666666666666666,
    # rG : dA mismatches
    ("G", "A", 1): 1.0, ("G", "A", 2): 0.6363636363636364,
    ("G", "A", 3): 0.5, ("G", "A", 4): 0.6,
    ("G", "A", 5): 0.6363636363636364, ("G", "A", 6): 0.4545454545454545,
    ("G", "A", 7): 0.6363636363636364, ("G", "A", 8): 0.6363636363636364,
    ("G", "A", 9): 0.5454545454545454, ("G", "A", 10): 0.4,
    ("G", "A", 11): 0.4166666666666666, ("G", "A", 12): 0.0,
    ("G", "A", 13): 0.0, ("G", "A", 14): 0.0,
    ("G", "A", 15): 0.0, ("G", "A", 16): 0.0,
    ("G", "A", 17): 0.25, ("G", "A", 18): 0.0,
    ("G", "A", 19): 0.0, ("G", "A", 20): 0.0,
    # rG : dG mismatches
    ("G", "G", 1): 0.7142857142857143, ("G", "G", 2): 0.6923076923076923,
    ("G", "G", 3): 0.3846153846153846, ("G", "G", 4): 0.4615384615384615,
    ("G", "G", 5): 0.5333333333333333, ("G", "G", 6): 0.5,
    ("G", "G", 7): 0.4, ("G", "G", 8): 0.4117647058823529,
    ("G", "G", 9): 0.5333333333333333, ("G", "G", 10): 0.0666666666666666,
    ("G", "G", 11): 0.2307692307692307, ("G", "G", 12): 0.1538461538461538,
    ("G", "G", 13): 0.0666666666666666, ("G", "G", 14): 0.0,
    ("G", "G", 15): 0.0666666666666666, ("G", "G", 16): 0.0,
    ("G", "G", 17): 0.0, ("G", "G", 18): 0.0,
    ("G", "G", 19): 0.0, ("G", "G", 20): 0.0,
    # rG : dT mismatches
    ("G", "T", 1): 0.9285714285714286, ("G", "T", 2): 1.0,
    ("G", "T", 3): 1.0, ("G", "T", 4): 1.0,
    ("G", "T", 5): 1.0, ("G", "T", 6): 0.9285714285714286,
    ("G", "T", 7): 1.0, ("G", "T", 8): 1.0,
    ("G", "T", 9): 1.0, ("G", "T", 10): 0.6428571428571428,
    ("G", "T", 11): 0.4615384615384615, ("G", "T", 12): 0.3846153846153846,
    ("G", "T", 13): 0.2666666666666666, ("G", "T", 14): 0.2666666666666666,
    ("G", "T", 15): 0.5, ("G", "T", 16): 0.4,
    ("G", "T", 17): 0.5, ("G", "T", 18): 0.5,
    ("G", "T", 19): 0.5, ("G", "T", 20): 1.0,
}

# PAM activity matrix (NGG PAM only used here; we expose the table for
# completeness). The CFD score for the canonical NGG PAM is 1.0; for
# alternative PAMs the score is multiplied by the matrix entry.
DOENCH_2016_PAM_MATRIX: dict[str, float] = {
    "AA": 0.0, "AC": 0.0, "AG": 0.259259259, "AT": 0.0,
    "CA": 0.0, "CC": 0.0, "CG": 0.107142857, "CT": 0.0,
    "GA": 0.069444444, "GC": 0.022222222, "GG": 1.0, "GT": 0.016129032,
    "TA": 0.0, "TC": 0.0, "TG": 0.038961038, "TT": 0.0,
}


# ---------------------------------------------------------------------------
# Hsu 2013 (a.k.a. MIT / Zhang lab) position-weight off-target matrix.
#
# Values are cleavage probability preserved at a single mismatch as a
# function of position from the PAM. Position 1 is closest to the PAM
# (the seed region); position 20 is the most PAM-distal. The Hsu 2013
# scheme is *position-weight only* — it does not condition on which two
# bases are mismatched, only on the position. This is strictly less
# expressive than CFD but is the matrix the PRD references.
#
# Source: Hsu PD, Scott DA, Weinstein JA, et al. (2013).
# "DNA targeting specificity of RNA-guided Cas9 nucleases."
# Nature Biotechnology 31(9):827-832. doi:10.1038/nbt.2647
# ---------------------------------------------------------------------------

# Indexed 1..20 from the PAM-proximal end.
HSU_2013_POSITION_WEIGHTS: dict[int, float] = {
    1: 0.0, 2: 0.0, 3: 0.014, 4: 0.0,
    5: 0.0, 6: 0.395, 7: 0.317, 8: 0.0,
    9: 0.389, 10: 0.079, 11: 0.445, 12: 0.508,
    13: 0.613, 14: 0.851, 15: 0.732, 16: 0.828,
    17: 0.615, 18: 0.804, 19: 0.685, 20: 0.583,
}


def _normalise_rna(b: str) -> str:
    """Map a sgRNA base to its CFD-matrix key (rA/rC/rG/rU)."""
    b = b.upper()
    if b == "T":
        return "U"
    if b in ("A", "C", "G", "U"):
        return b
    raise ValueError(f"unexpected base {b!r} (expected A/C/G/T/U)")


_DNA_COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}


def _dna_other_strand(b: str) -> str:
    """The CFD matrix keys ``dA/dC/dG/dT`` describe the DNA base on
    the strand *opposite* the sgRNA in the R-loop duplex. Cas-OFFinder
    emits both spacers (crrna and dna) in the same orientation (the
    protospacer-matching strand). To look up the matrix we therefore
    take the Watson-Crick complement of each genomic spacer base."""
    b = b.upper()
    if b not in _DNA_COMPLEMENT:
        raise ValueError(f"unexpected DNA base {b!r} (expected A/C/G/T)")
    return _DNA_COMPLEMENT[b]


def cfd_score_pair(
    crrna_spacer: str,
    dna_spacer: str,
    *,
    pam: str | None = None,
    matrix: MatrixName = "doench_2016",
) -> float:
    """Compute the CFD score for a single (sgRNA, off-target DNA) pair.

    Parameters
    ----------
    crrna_spacer:
        20-nt sgRNA spacer (5' -> 3'). May be RNA (``U``) or DNA (``T``).
    dna_spacer:
        20-nt off-target DNA spacer (5' -> 3'). The case is ignored —
        Cas-OFFinder lowercases mismatches in its output but the
        comparison is case-insensitive here.
    pam:
        Two-letter PAM tail (the ``NN`` of ``NNN`` — drop the first
        ``N``). If supplied, the score is multiplied by the PAM
        matrix entry. Optional; many callers pre-filter on NGG.
    matrix:
        ``"doench_2016"`` (default; CFD with mismatch-conditioned
        weights) or ``"hsu_2013"`` (position-only weights).

    Returns
    -------
    float
        Predicted off-target activity in ``[0, 1]``. Higher = more
        likely to be cleaved.
    """
    if len(crrna_spacer) != 20:
        raise ValueError(
            f"crrna_spacer must be 20 nt; got {len(crrna_spacer)}"
        )
    if len(dna_spacer) != 20:
        raise ValueError(
            f"dna_spacer must be 20 nt; got {len(dna_spacer)}"
        )

    crrna_u = crrna_spacer.upper()
    dna_u = dna_spacer.upper()

    score = 1.0
    if matrix == "doench_2016":
        for i, (rb, db) in enumerate(zip(crrna_u, dna_u, strict=True)):
            # Position 1 = PAM-distal first base of the spacer; position
            # 20 = PAM-proximal (immediately 5' of the PAM). Note this
            # follows the original Doench 2016 5'->3' indexing.
            position = i + 1
            # Match condition: T <-> U synonyms; otherwise same letter.
            if rb == db:
                continue
            if rb in ("T", "U") and db in ("T", "U"):
                continue
            rb_n = _normalise_rna(rb)        # rA/rC/rG/rU
            db_n = _dna_other_strand(db)      # WC-complement of genomic base
            key = (rb_n, db_n, position)
            if key not in DOENCH_2016_MM_MATRIX:
                raise KeyError(
                    f"CFD matrix has no entry for {key!r} -- "
                    f"crrna_spacer={crrna_spacer!r}, "
                    f"dna_spacer={dna_spacer!r}, position={position}"
                )
            score *= DOENCH_2016_MM_MATRIX[key]
    elif matrix == "hsu_2013":
        for i, (rb, db) in enumerate(zip(crrna_u, dna_u, strict=True)):
            position = 20 - i  # 1 = PAM-proximal (Hsu convention)
            if rb == db or (rb == "U" and db == "T") or (rb == "T" and db == "U"):
                continue
            score *= HSU_2013_POSITION_WEIGHTS[position]
    else:
        raise ValueError(
            f"unknown matrix {matrix!r} -- expected 'doench_2016' or 'hsu_2013'"
        )

    if pam is not None:
        if len(pam) != 2:
            raise ValueError(
                f"pam must be the two PAM-proximal bases (e.g. 'GG'); "
                f"got {pam!r}"
            )
        pam_u = pam.upper()
        if pam_u not in DOENCH_2016_PAM_MATRIX:
            raise ValueError(
                f"pam {pam_u!r} not in PAM matrix; expected one of "
                f"{sorted(DOENCH_2016_PAM_MATRIX)}"
            )
        score *= DOENCH_2016_PAM_MATRIX[pam_u]

    return score


def _extract_spacers(row: CasOFFinderRow) -> tuple[str, str, str]:
    """Pull (crrna_spacer, dna_spacer, pam2) from a canonical row.

    Cas-OFFinder convention:

    * ``crrna`` is 23 nt: 20-nt spacer + 3-nt PAM template (``NGG``).
    * ``dna`` is 23 nt: 20-nt off-target genomic spacer + 3-nt actual
      genomic PAM. Mismatches against the spacer are lowercased; the
      PAM is verbatim.

    We return the upper-cased 20-nt spacers and the two PAM-proximal
    PAM bases (``GG`` of NGG); the leading N is unused by CFD's PAM
    matrix.
    """
    if len(row.crrna) != 23:
        raise ValueError(
            f"row.crrna must be 23 nt (20-nt spacer + 3-nt PAM); "
            f"got {len(row.crrna)} for guide_id={row.guide_id!r}"
        )
    if len(row.dna) != 23:
        raise ValueError(
            f"row.dna must be 23 nt; got {len(row.dna)} for "
            f"guide_id={row.guide_id!r}"
        )
    crrna_spacer = row.crrna[:20].upper()
    dna_spacer = row.dna[:20].upper()
    pam2 = row.dna[21:23].upper()  # drop the leading N
    return crrna_spacer, dna_spacer, pam2


class CFDScorer:
    """CFD off-target scorer mirroring the :class:`DNABERTEpiScorer` API.

    Parameters
    ----------
    matrix:
        ``"doench_2016"`` (default) for the full CFD position-and-base-
        conditioned matrix; ``"hsu_2013"`` for the position-only
        Hsu/MIT weights.
    apply_pam_penalty:
        If True, multiply each row's score by the Doench 2016 PAM
        matrix entry. Default False — the PAM filter at the scan
        stage already enforces NGG, and the PAM penalty for non-NGG
        rows is identically zero. Useful when scoring rows from a
        PAM-permissive scan.

    Notes
    -----
    The class is stateless beyond its config; ``score`` can be called
    repeatedly. Smoke mode is unnecessary because the scorer has no
    runtime dependencies (numpy not required, pure stdlib).
    """

    def __init__(
        self,
        *,
        matrix: MatrixName = "doench_2016",
        apply_pam_penalty: bool = False,
    ) -> None:
        if matrix not in ("doench_2016", "hsu_2013"):
            raise ValueError(
                f"matrix must be 'doench_2016' or 'hsu_2013'; got {matrix!r}"
            )
        self.matrix = matrix
        self.apply_pam_penalty = bool(apply_pam_penalty)

    def score(self, rows: Iterable[CasOFFinderRow]) -> Iterator[ScoreRow]:
        """Score canonical scan rows, yielding :class:`ScoreRow` in input order.

        Identity columns are preserved verbatim; the scorer never re-sorts.
        """
        for r in rows:
            crrna_spacer, dna_spacer, pam2 = _extract_spacers(r)
            s = cfd_score_pair(
                crrna_spacer,
                dna_spacer,
                pam=pam2 if self.apply_pam_penalty else None,
                matrix=self.matrix,
            )
            yield ScoreRow.from_row(r, s)


def aggregate_cfd(
    rows: Iterable[ScoreRow],
    *,
    exclude_on_target: bool = True,
) -> dict[str, float]:
    """Per-guide aggregate (specificity) score in [0, 100].

    Computes::

        spec[guide] = 100 / (100 + Σ_off  cfd_off)

    where the sum is over all off-target hits for that guide. If
    ``exclude_on_target`` is True (the default), any row with
    ``mismatches == 0`` is excluded from the sum (it is the on-target
    site, not an off-target).

    Parameters
    ----------
    rows:
        Iterable of :class:`ScoreRow` already scored by
        :class:`CFDScorer`.
    exclude_on_target:
        Drop ``mismatches == 0`` rows from the off-target sum. Default
        True (CRISPOR convention).

    Returns
    -------
    dict[str, float]
        Mapping ``guide_id -> specificity_score``. A guide with no
        off-target hits gets a perfect score of 100.0. The empty
        input returns the empty dict.
    """
    sums: dict[str, float] = defaultdict(float)
    seen: dict[str, bool] = {}
    for r in rows:
        seen[r.guide_id] = True
        if exclude_on_target and r.mismatches == 0:
            continue
        # CRISPOR convention: CFD scores enter the denominator on a
        # 0..100 scale, so a single perfect off-target contributes 100
        # and the resulting specificity is 100/(100+100) = 50.
        sums[r.guide_id] += 100.0 * float(r.score)
    out: dict[str, float] = {}
    for gid in seen:
        s = sums.get(gid, 0.0)
        # The bare ratio 100/(100+s) is in (0, 1]; scale to (0, 100]
        # so the output matches CRISPOR's percentage display
        # convention (no off-targets -> 100, one perfect off-target ->
        # 50, etc.).
        out[gid] = 100.0 * (100.0 / (100.0 + s))
    return out


# ---------------------------------------------------------------------------
# Reference-equivalence helpers (for verify-score)
# ---------------------------------------------------------------------------

def _log_cfd_pair(crrna_spacer: str, dna_spacer: str) -> float:
    """Log-domain CFD pair score (numerical-stability helper).

    The product over 20 mismatch weights underflows to 0.0 in float64
    when many of the weights are < 0.1; this helper sums logs so a
    high-mismatch off-target's score stays representable as a log
    probability. Useful for verify-score epsilon-tolerance checks.
    """
    crrna_u = crrna_spacer.upper()
    dna_u = dna_spacer.upper()
    log_score = 0.0
    for i, (rb, db) in enumerate(zip(crrna_u, dna_u, strict=True)):
        position = i + 1
        if rb == db:
            continue
        if rb in ("T", "U") and db in ("T", "U"):
            continue
        rb_n = _normalise_rna(rb)
        db_n = _dna_other_strand(db)
        w = DOENCH_2016_MM_MATRIX[(rb_n, db_n, position)]
        if w == 0.0:
            return -math.inf
        log_score += math.log(w)
    return log_score
