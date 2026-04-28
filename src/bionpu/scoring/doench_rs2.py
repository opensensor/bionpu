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

"""Doench on-target activity scorer (Rule Set 1, logistic regression).

CPU-only. Scores a 30-nt context window (4-nt 5' flank + 20-nt
protospacer + 3-nt PAM + 3-nt 3' flank) using the published
Doench 2014 logistic-regression coefficients.

Reference
---------

The "logistic regression" on-target activity model from:

    Doench JG, Hartenian E, Graham DB, Tothova Z, Hegde M, Smith I,
    Sullender M, Ebert BL, Xavier RJ, Root DE. (2014).
    "Rational design of highly active sgRNAs for CRISPR-Cas9-mediated
    gene inactivation."
    Nature Biotechnology 32(12):1262-1267.
    doi:10.1038/nbt.3026

This is the *Rule Set 1* model. Rule Set 2 (Doench 2016, the gradient-
boosted tree usually called "Azimuth") is a much larger ensemble that
requires the upstream ``azimuth`` package and a pickled scikit-learn
model file. For the v0.3 layer-on-top-of-the-NPU-scan ask, Rule Set 1
is the right fit:

* Pure logistic regression with 72 published feature weights — fully
  transparent, no pickle, no proprietary code.
* CPU-cheap (one dot product per guide) and CRISPR-track-friendly.
* The 30-nt context window matches Rule Set 2's input format, so
  upgrading the implementation later is a coefficient swap rather
  than a re-architecture.

Score interpretation: Rule Set 1's output is a logistic-regression
score in ``(0, 1)`` interpreted as the predicted percentile of
on-target activity. Higher = more active. Calibration is
empirical (see Doench 2014 Figure 1).

Inputs and conventions
----------------------

A 30-nt context window is required. Layout (1-indexed):

    1..4    : 5'-flanking genomic bases
    5..24   : 20-nt protospacer (the sgRNA spacer matches this, with U->T)
    25..27  : 3-nt PAM (canonically NGG)
    28..30  : 3'-flanking genomic bases

This matches the convention in every downstream Doench-family tool
(Azimuth, CRISPOR, sgRNA-Designer, sgrna_modeler).

Why this isn't Rule Set 2 / Azimuth
-----------------------------------

The Doench 2016 paper's "Rule Set 2" is a gradient-boosted regressor
fit on order-1 + order-2 position features, GC counts, and
thermodynamic features. It outperforms Rule Set 1 by ~5% Pearson r
on the held-out test set but has no closed-form coefficient table —
the only authoritative implementation is the Azimuth Python package
with a pickled sklearn ``GradientBoostingRegressor`` (~30 MB).
Re-implementing Rule Set 2 here without the pickle means either
re-training (data drift; rejects the "byte-equal to Doench 2016"
contract) or shipping the pickle (license / size / version-pinning
overhead disproportionate to the v0.3 scope).

The class :class:`DoenchAzimuthScorer` exists as a placeholder; it
raises :class:`AzimuthNotInstalledError` at construction time when
the upstream ``azimuth`` package is not importable. The PRD-1
"on-target accuracy" gate sets the bar at the Rule Set 1 logistic
regression for the v0.3 layer; Rule Set 2 / Azimuth is a follow-up
once a license-clean pickle distribution is settled.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator

from bionpu.data.canonical_sites import CasOFFinderRow

from .types import ScoreRow

__all__ = [
    "AzimuthNotInstalledError",
    "DOENCH_RS1_INTERCEPT",
    "DOENCH_RS1_WEIGHTS",
    "DoenchAzimuthScorer",
    "DoenchRS1Scorer",
    "doench_rs1_score",
    "extract_30mer_context",
]


# ---------------------------------------------------------------------------
# Doench 2014 Rule Set 1 logistic-regression coefficients.
#
# Weights are keyed by either:
#   - ('pos', position, base)         -- 1-mer position-specific feature
#   - ('di', position, dimer)         -- 2-mer position-specific feature
#                                        (position is the start index of
#                                        the dimer)
#   - ('gc_low',)                     -- indicator: GC count in spacer < 10
#   - ('gc_high',)                    -- indicator: GC count in spacer > 10
#
# All positions are 1-indexed across the 30-nt context window
# (4-nt flank + 20-nt protospacer + 3-nt PAM + 3-nt flank).
#
# Source: Doench 2014 Supplementary Table 1 (the published 72-feature
# logistic-regression model). Reproduced from the open-source
# `crispor` and `sgrna_modeler` redistributions of the paper's
# coefficient table; coefficients are paper-text-authoritative, not
# pickle-derived.
# ---------------------------------------------------------------------------

DOENCH_RS1_INTERCEPT: float = 0.59763615

# Indicator feature: GC content of the 20-nt protospacer.
# 'gc_low' fires when the spacer has fewer than 10 G+C bases.
# 'gc_high' fires when the spacer has more than 10 G+C bases.
# Exactly 10 G+C bases fires neither. The actual Doench 2014 paper
# coefficients for these indicators live in ``DOENCH_RS1_WEIGHTS``
# under the ``('gc_low',)`` / ``('gc_high',)`` keys.

DOENCH_RS1_WEIGHTS: dict[tuple, float] = {
    # ----- single-nucleotide position features -----
    # 1-indexed position 1..30 within the 30-mer context.
    # Format: ('pos', position, base) -> weight.
    # (Source table reproduced from Doench 2014 Sup. Table 1, columns
    # 'A', 'C', 'G', 'T' across 30 positions. Zero-weight cells are
    # omitted for compactness.)
    ("pos", 1, "G"): -0.2753771,
    ("pos", 2, "A"): -0.3238875,
    ("pos", 2, "C"): 0.17212887,
    ("pos", 3, "C"): -0.1006662,
    ("pos", 4, "C"): -0.2018029,
    ("pos", 4, "G"): 0.24595663,
    ("pos", 5, "A"): 0.03644004,
    ("pos", 5, "C"): 0.09837684,
    ("pos", 6, "C"): -0.7411813,
    ("pos", 6, "G"): -0.3932644,
    ("pos", 11, "A"): -0.466099,
    ("pos", 14, "A"): 0.08537695,
    ("pos", 14, "C"): -0.013814,
    ("pos", 15, "A"): 0.27262051,
    ("pos", 15, "C"): -0.1190226,
    ("pos", 15, "T"): -0.2859442,
    ("pos", 16, "A"): 0.09745459,
    ("pos", 16, "G"): -0.1755462,
    ("pos", 17, "C"): -0.3457955,
    ("pos", 17, "G"): -0.6780964,
    ("pos", 18, "A"): 0.22508903,
    ("pos", 18, "C"): -0.5077941,
    ("pos", 19, "G"): -0.4173736,
    ("pos", 19, "T"): -0.054307,
    ("pos", 20, "G"): 0.37989937,
    ("pos", 20, "T"): -0.0907126,
    ("pos", 21, "C"): 0.05782332,
    ("pos", 21, "T"): -0.5305673,
    ("pos", 22, "T"): -0.8770074,
    ("pos", 23, "C"): -0.8762358,
    ("pos", 23, "G"): 0.27891626,
    ("pos", 23, "T"): -0.4031022,
    ("pos", 24, "A"): -0.0773007,
    ("pos", 24, "C"): 0.28793562,
    ("pos", 24, "T"): -0.2216372,
    ("pos", 27, "G"): -0.6890167,
    ("pos", 27, "T"): 0.11787758,
    ("pos", 28, "C"): -0.1604453,
    ("pos", 29, "G"): 0.38634258,
    # ----- dinucleotide position features -----
    # ('di', position, dimer) -> weight, where 'position' is the 1-indexed
    # start of the dinucleotide in the 30-mer.
    ("di", 1, "GT"): -0.6257787,
    ("di", 4, "GC"): 0.30004332,
    ("di", 5, "AA"): -0.8348362,
    ("di", 5, "TA"): 0.76062777,
    ("di", 6, "GG"): -0.4908167,
    ("di", 11, "GG"): -1.5169074,
    ("di", 11, "TA"): 0.7092612,
    ("di", 11, "TC"): 0.49629861,
    ("di", 11, "TT"): -0.5868629,
    ("di", 12, "GG"): -0.3345637,
    ("di", 13, "GA"): 0.76384993,
    ("di", 13, "GC"): -0.5370252,
    ("di", 16, "TG"): -0.7981461,
    ("di", 18, "GG"): -0.6668087,
    ("di", 18, "TC"): 0.35318325,
    ("di", 19, "CC"): 0.74807209,
    ("di", 19, "TG"): -0.3672668,
    ("di", 20, "AC"): 0.56820913,
    ("di", 20, "CG"): 0.32907207,
    ("di", 20, "GA"): -0.8364568,
    ("di", 20, "GG"): -0.7822076,
    ("di", 21, "TC"): -1.029693,
    ("di", 22, "CG"): 0.85619782,
    ("di", 22, "CT"): -0.4632077,
    ("di", 23, "AA"): -0.5794924,
    ("di", 23, "AG"): 0.64907554,
    ("di", 24, "AG"): -0.0773007,
    ("di", 24, "CG"): 0.28793562,
    ("di", 24, "TG"): -0.2216372,
    ("di", 26, "GT"): 0.11787758,
    ("di", 28, "GG"): -0.69774,
    # ----- GC-content indicator features -----
    ("gc_low",): -0.2026259,    # spacer GC count < 10 (i.e., GC < 10 of 20)
    ("gc_high",): -0.1665878,   # spacer GC count > 10
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class AzimuthNotInstalledError(RuntimeError):
    """Raised when Doench Rule Set 2 / Azimuth is requested but the
    upstream ``azimuth`` package + model pickle are not available.
    Track upgrade in ``docs/model-selection-audit.md``."""


def extract_30mer_context(
    *,
    chrom_seq: str,
    start: int,
    strand: str,
    spacer_len: int = 20,
    pam_len: int = 3,
    flank_5p: int = 4,
    flank_3p: int = 3,
) -> str:
    """Extract the 30-nt context window for an on-target hit.

    Parameters
    ----------
    chrom_seq:
        Forward-strand chromosome sequence (ACGT-only; uppercase or
        lowercase). Must be long enough to include the flanks.
    start:
        0-based forward-strand start of the protospacer (the first
        spacer base).
    strand:
        ``"+"`` or ``"-"``.
    spacer_len, pam_len, flank_5p, flank_3p:
        Layout knobs. The defaults match the Doench 2014/2016 layout
        (4-nt 5' flank + 20-nt spacer + 3-nt PAM + 3-nt 3' flank =
        30 nt).

    Returns
    -------
    str
        30-nt uppercase context, in the *guide-strand* orientation.
        For ``+`` strand hits this is just the substring; for ``-``
        strand hits this is the reverse complement of the substring.

    Raises
    ------
    ValueError
        If the requested window falls off the end of the chromosome
        (e.g. the on-target site is too close to the contig boundary).
    """
    if strand not in ("+", "-"):
        raise ValueError(f"strand must be '+' or '-'; got {strand!r}")
    window_len = flank_5p + spacer_len + pam_len + flank_3p
    if strand == "+":
        win_start = start - flank_5p
        win_end = start + spacer_len + pam_len + flank_3p
    else:
        # On the - strand, the 'start' in canonical coordinates is the
        # 0-based position on the forward strand. The protospacer reads
        # 5'->3' on the *reverse* strand, so the 30-mer in guide-strand
        # orientation is the RC of the forward window.
        # The forward-strand window: [start - flank_3p, start + spacer + pam + flank_5p)
        win_start = start - flank_3p
        win_end = start + spacer_len + pam_len + flank_5p
    if win_start < 0 or win_end > len(chrom_seq):
        raise ValueError(
            f"30-mer window ({win_start}, {win_end}) is out of range "
            f"for chromosome of length {len(chrom_seq)}"
        )
    raw = chrom_seq[win_start:win_end].upper()
    if len(raw) != window_len:
        raise ValueError(
            f"extracted window has length {len(raw)}; expected {window_len}"
        )
    if strand == "-":
        raw = _reverse_complement(raw)
    return raw


_RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _reverse_complement(seq: str) -> str:
    return seq.translate(_RC_TABLE)[::-1]


def doench_rs1_score(context_30mer: str) -> float:
    """Compute the Doench 2014 Rule Set 1 on-target activity score.

    Parameters
    ----------
    context_30mer:
        30-nt uppercase ACGT string in the guide-strand orientation
        (4-nt 5' flank + 20-nt protospacer + 3-nt PAM + 3-nt 3' flank).

    Returns
    -------
    float
        Logistic-regression score in ``(0, 1)``. Higher = predicted
        more active.
    """
    if len(context_30mer) != 30:
        raise ValueError(
            f"context_30mer must be 30 nt; got {len(context_30mer)}"
        )
    seq = context_30mer.upper()
    if any(c not in "ACGT" for c in seq):
        raise ValueError(
            f"context_30mer must be ACGT; got {context_30mer!r}"
        )

    # Spacer occupies positions 5..24 (1-indexed).
    spacer = seq[4:24]
    gc_count = spacer.count("G") + spacer.count("C")

    score = DOENCH_RS1_INTERCEPT

    # Position-specific 1-mer features.
    for i, base in enumerate(seq, start=1):
        score += DOENCH_RS1_WEIGHTS.get(("pos", i, base), 0.0)

    # Position-specific 2-mer features.
    for i in range(1, 30):
        dimer = seq[i - 1 : i + 1]
        score += DOENCH_RS1_WEIGHTS.get(("di", i, dimer), 0.0)

    # GC content indicators.
    if gc_count < 10:
        score += DOENCH_RS1_WEIGHTS[("gc_low",)]
    elif gc_count > 10:
        score += DOENCH_RS1_WEIGHTS[("gc_high",)]

    # Logistic transform.
    return 1.0 / (1.0 + math.exp(-score))


class DoenchRS1Scorer:
    """Doench 2014 Rule Set 1 on-target scorer.

    Parameters
    ----------
    chrom_lookup:
        Mapping from chromosome name (e.g. ``"chr22"``) to its forward-
        strand sequence. Used to extract the 30-mer context flanking
        each on-target row. The sequence(s) must be long enough that
        the context window fits on each row's strand.

    score(rows)
        Yields :class:`ScoreRow` for every input row. The ``score``
        column is the Rule Set 1 logistic-regression activity score
        in ``(0, 1)``.

    Notes
    -----
    On-target scoring is meaningful only for the *intended* target
    site (mismatches == 0), but Rule Set 1 will score any 30-mer
    context. We do not gate on ``mismatches == 0`` here — the caller
    decides which rows to feed in. For a typical PRD-1 pipeline, the
    caller filters ``mismatches == 0`` before scoring (the on-target
    is the row matching the input guide; everything else is an
    off-target candidate scored by :class:`bionpu.scoring.cfd.CFDScorer`).

    The scorer is stateless beyond ``chrom_lookup``; ``score`` can be
    called repeatedly. Smoke mode is unnecessary because the scorer
    has no runtime dependencies (numpy not required, pure stdlib).
    """

    def __init__(self, chrom_lookup: dict[str, str]) -> None:
        if not isinstance(chrom_lookup, dict):
            raise TypeError(
                f"chrom_lookup must be a dict[str, str]; got "
                f"{type(chrom_lookup).__name__}"
            )
        self.chrom_lookup = {k: v.upper() for k, v in chrom_lookup.items()}

    def score(self, rows: Iterable[CasOFFinderRow]) -> Iterator[ScoreRow]:
        """Score rows, yielding :class:`ScoreRow` in input order.

        Identity columns are preserved verbatim; the scorer never re-sorts.
        """
        for r in rows:
            seq = self.chrom_lookup.get(r.chrom)
            if seq is None:
                raise KeyError(
                    f"chrom_lookup has no entry for {r.chrom!r}; "
                    f"have {sorted(self.chrom_lookup)!r}"
                )
            ctx = extract_30mer_context(
                chrom_seq=seq,
                start=r.start,
                strand=r.strand,
            )
            s = doench_rs1_score(ctx)
            yield ScoreRow.from_row(r, s)


class DoenchAzimuthScorer:
    """Doench 2016 Rule Set 2 (Azimuth) scorer — placeholder.

    Constructing this class without the upstream ``azimuth`` package
    installed raises :class:`AzimuthNotInstalledError`. The actual
    integration is gated on a license-clean redistribution of the
    Azimuth model pickle; tracked in
    ``docs/model-selection-audit.md`` § Rule-Set-2.
    """

    def __init__(self) -> None:
        try:
            import azimuth  # noqa: F401
        except ImportError as exc:
            raise AzimuthNotInstalledError(
                "Doench Rule Set 2 / Azimuth requires the upstream "
                "`azimuth` package and a pickled model. The bionpu "
                "v0.3 alpha layer ships Rule Set 1 only; use "
                "`DoenchRS1Scorer` for the in-tree path."
            ) from exc
        # Real Rule Set 2 wiring is intentionally not implemented:
        # the upstream API has shifted across azimuth releases and we
        # don't want to silently bind to whatever pickle is on disk.
        raise AzimuthNotInstalledError(
            "Doench Rule Set 2 wiring is not implemented in v0.3. "
            "Use DoenchRS1Scorer; tracked as a v0.4 follow-up."
        )

    def score(self, rows: Iterable[CasOFFinderRow]) -> Iterator[ScoreRow]:
        # Unreachable -- __init__ raises. Present so the class
        # surface mirrors :class:`DoenchRS1Scorer` for type-checkers.
        raise AzimuthNotInstalledError("DoenchAzimuthScorer not initialised")
        yield  # pragma: no cover  -- generator type only
