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

"""Track B v0 — ViennaRNA secondary-structure feature extractor.

This module is **Task T4** of ``track-b-pegrna-design-plan.md``. It
wraps the ViennaRNA Python bindings (``import RNA``; verified at
v2.7.2 in ``state/track-b-prereq-probe.md``) and exposes a single
public entry point :func:`compute_folding_features` that produces a
:class:`PegRNAFoldingFeatures` NamedTuple over the full composite
pegRNA RNA sequence (spacer + scaffold + RTT + PBS).

The features computed are:

``mfe_kcal``
    Minimum free energy of the full pegRNA in kcal/mol via
    ``RNA.fold_compound(seq).mfe()``.
``mfe_structure``
    Dot-bracket secondary structure for the same MFE fold; identical
    length to the RNA sequence; alphabet ``{ '.', '(', ')' }``.
``pbs_pairing_prob``
    Mean per-position probability that each base in the PBS region of
    the pegRNA is paired with **any** other base in the pegRNA, derived
    from the partition-function base-pair probability matrix
    (``RNA.fold_compound.pf()`` then ``bpp()``). Range [0, 1]. This is
    a v0 surrogate for "PBS pairs with target genome" — the genome
    stub is not folded into the same compound in v0; T8 may layer a
    cofold-based refinement later. The metric still ranks pegRNAs
    correctly because intramolecular PBS sequestration directly
    competes with target-genome priming.
``scaffold_disruption``
    Fraction of dot-bracket positions in the scaffold-region of the
    pegRNA that DIFFER from the cached canonical scaffold's MFE
    structure (a Hamming-style distance over the dot-bracket string,
    normalised by scaffold length). Range [0, 1]; 0 means the
    scaffold's structure is preserved exactly.

Caching
-------
The canonical scaffold's MFE structure is computed once per scaffold
variant via an ``lru_cache`` keyed on the scaffold's RNA sequence.

Edge cases
----------
* Empty / sub-fold-compound length pegRNA components — the wrapper
  returns a NaN-tagged features tuple rather than crashing or hanging
  ViennaRNA.
* Sub-pieces are normalised to RNA letters via ``T -> U`` replacement
  so the wrapper accepts both RNA-input (the canonical case for T6's
  enumerator) and DNA-input (convenient for ad-hoc calls).
"""

from __future__ import annotations

import math
from functools import lru_cache

import RNA  # ViennaRNA 2.x — installed by Track B T1

from bionpu.genomics.pe_design.types import PegRNAFoldingFeatures

__all__ = [
    "PegRNAFoldingFeatures",
    "compute_folding_features",
]


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _to_rna(seq: str) -> str:
    """Normalise an input nucleotide string to RNA (uppercase ACGU).

    Accepts DNA or RNA letters; ``T`` is replaced with ``U``. Other
    characters are passed through (ViennaRNA tolerates degenerate
    letters; downstream MFE just treats them as inert).
    """
    return seq.upper().replace("T", "U")


@lru_cache(maxsize=64)
def _scaffold_reference_structure(scaffold_rna: str) -> str:
    """Cached MFE dot-bracket structure for a scaffold RNA.

    We compute this exactly once per distinct scaffold sequence so the
    enumerator pipeline (T6 → T4 over thousands of candidates) does not
    re-fold the scaffold every call.
    """
    if not scaffold_rna:
        return ""
    fc = RNA.fold_compound(scaffold_rna)
    structure, _mfe = fc.mfe()
    return structure


def _pbs_paired_probability(
    bpp: tuple, n: int, pbs_start_1: int, pbs_end_1: int
) -> float:
    """Mean per-position pairing probability for the PBS span.

    Parameters
    ----------
    bpp:
        ViennaRNA's base-pair probability matrix (1-indexed, square,
        size ``n + 1``); ``bpp[i][j]`` is defined for ``i < j``.
    n:
        Length of the folded RNA sequence.
    pbs_start_1, pbs_end_1:
        1-indexed inclusive PBS span within the folded RNA.
    """
    if pbs_end_1 < pbs_start_1 or n == 0:
        return float("nan")

    total = 0.0
    count = 0
    for i in range(pbs_start_1, pbs_end_1 + 1):
        # Sum probability that base i is paired with anything.
        p = 0.0
        # Pairs (i, j) with j > i live at bpp[i][j].
        for j in range(i + 1, n + 1):
            p += bpp[i][j]
        # Pairs (k, i) with k < i live at bpp[k][i].
        for k in range(1, i):
            p += bpp[k][i]
        # Numerical guard: probabilities sum to <=1 in theory but can
        # drift fractionally above due to FP.
        if p > 1.0:
            p = 1.0
        total += p
        count += 1

    return total / count if count else float("nan")


def _scaffold_disruption(
    full_structure: str,
    reference_scaffold_structure: str,
    scaffold_start_0: int,
) -> float:
    """Fraction of scaffold-region dot-bracket positions that differ
    from the cached canonical reference.

    A simple position-wise mismatch ratio — symbols ``.``, ``(``, ``)``
    are compared at the same offset within the scaffold span. The
    metric is normalised by scaffold length.
    """
    scaffold_len = len(reference_scaffold_structure)
    if scaffold_len == 0:
        return float("nan")
    candidate_region = full_structure[
        scaffold_start_0 : scaffold_start_0 + scaffold_len
    ]
    if len(candidate_region) != scaffold_len:
        # Defensive guard: shouldn't happen unless caller passed
        # mismatched arguments.
        return float("nan")
    diffs = sum(
        1
        for a, b in zip(candidate_region, reference_scaffold_structure)
        if a != b
    )
    return diffs / scaffold_len


def _nan_features() -> PegRNAFoldingFeatures:
    """Return a NaN-tagged feature tuple for edge-case inputs."""
    nan = float("nan")
    return PegRNAFoldingFeatures(
        mfe_kcal=nan,
        mfe_structure="",
        pbs_pairing_prob=nan,
        scaffold_disruption=nan,
    )


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def compute_folding_features(
    spacer: str,
    scaffold: str,
    rtt: str,
    pbs: str,
    *,
    scaffold_variant: str = "sgRNA_canonical",
) -> PegRNAFoldingFeatures:
    """Compute ViennaRNA-derived secondary-structure features for a
    composite pegRNA.

    Parameters
    ----------
    spacer:
        20-nt spacer (DNA or RNA letters; auto-normalised to RNA).
    scaffold:
        Scaffold RNA (the canonical Anzalone 2019 scaffold or a
        published variant from
        :data:`bionpu.genomics.pe_design.pegrna_constants.SCAFFOLD_VARIANTS`).
    rtt:
        Reverse-transcriptase template (RNA letters; encodes the
        desired edit + flanking).
    pbs:
        Primer binding site (RNA letters; reverse-complement of the
        nicked-strand 5' stub).
    scaffold_variant:
        Name tag for the chosen scaffold; carried for diagnostic
        traceability but the actual reference structure is keyed on
        the scaffold sequence itself (so the cache is correct even if
        two variants map to the same canonical body, which is the v0
        situation for ``evopreQ1``/``tevopreQ1`` per
        :mod:`bionpu.genomics.pe_design.pegrna_constants`).

    Returns
    -------
    :class:`PegRNAFoldingFeatures`
        Folded features. NaN-tagged if the composite RNA is empty
        (length 0). Never raises for short-but-non-empty input;
        ViennaRNA tolerates ≥1 nt sequences.
    """
    # Normalise to RNA letters. The enumerator (T6) emits RNA already,
    # but the wrapper accepts DNA for ad-hoc use.
    spacer_rna = _to_rna(spacer)
    scaffold_rna = _to_rna(scaffold)
    rtt_rna = _to_rna(rtt)
    pbs_rna = _to_rna(pbs)

    full_seq = spacer_rna + scaffold_rna + rtt_rna + pbs_rna
    n = len(full_seq)
    if n == 0:
        return _nan_features()

    # Fold the full composite pegRNA.
    fc = RNA.fold_compound(full_seq)
    structure, mfe = fc.mfe()

    # Partition function + base-pair probability matrix.
    fc.pf()
    bpp = fc.bpp()

    # PBS span within the full pegRNA (1-indexed inclusive).
    pbs_len = len(pbs_rna)
    if pbs_len > 0:
        pbs_start_1 = (
            len(spacer_rna) + len(scaffold_rna) + len(rtt_rna) + 1
        )
        pbs_end_1 = pbs_start_1 + pbs_len - 1
        pbs_paired = _pbs_paired_probability(
            bpp, n, pbs_start_1, pbs_end_1
        )
    else:
        pbs_paired = float("nan")

    # Scaffold-region structure vs the cached canonical reference.
    if scaffold_rna:
        reference_struct = _scaffold_reference_structure(scaffold_rna)
        scaffold_start_0 = len(spacer_rna)
        disruption = _scaffold_disruption(
            structure, reference_struct, scaffold_start_0
        )
    else:
        disruption = float("nan")

    # Numeric defence: ViennaRNA returns numpy.float32-like objects
    # for MFE; cast to float for consistent NamedTuple typing.
    mfe_float = float(mfe)
    if math.isnan(mfe_float):
        # ViennaRNA shouldn't return NaN for a valid sequence, but
        # surface it cleanly if it ever does.
        pass

    return PegRNAFoldingFeatures(
        mfe_kcal=mfe_float,
        mfe_structure=structure,
        pbs_pairing_prob=float(pbs_paired),
        scaffold_disruption=float(disruption),
    )
