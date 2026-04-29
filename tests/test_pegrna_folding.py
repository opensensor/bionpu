# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

"""Track B v0 — Tests for the ViennaRNA folding-feature wrapper.

Covers Task T4 acceptance criteria from
``track-b-pegrna-design-plan.md``:

* A canonical Anzalone 2019 pegRNA (HEK3 +1) reproduces a published-
  reference MFE within ±0.5 kcal/mol. We pin the reference value via
  RNAfold itself (ViennaRNA 2.7.2 against the canonical spacer +
  scaffold + RTT + PBS) so the test acts as a regression lock against
  unintended changes to the wrapper's MFE plumbing.
* ``scaffold_disruption`` is exactly 0.0 when the candidate's scaffold
  region preserves the canonical scaffold's MFE structure (the trivial
  case: the scaffold by itself).
* ``pbs_pairing_prob`` is in [0, 1].
* The wrapper handles edge-case empty / short sequences gracefully
  (no crashes; NaN or a clean exception).
"""

from __future__ import annotations

import math

import pytest


# ---------- module-surface probe ---------- #


def test_module_exports_compute_folding_features():
    """The T4 wrapper exposes ``compute_folding_features`` as the
    canonical entry point and re-exports
    :class:`PegRNAFoldingFeatures` from T3's types module (must NOT
    redefine).
    """
    from bionpu.scoring import pegrna_folding
    from bionpu.genomics.pe_design.types import PegRNAFoldingFeatures

    assert hasattr(pegrna_folding, "compute_folding_features"), (
        "T4 must expose compute_folding_features"
    )
    # The wrapper must reuse T3's NamedTuple, NOT redefine it.
    assert pegrna_folding.PegRNAFoldingFeatures is PegRNAFoldingFeatures, (
        "T4 must import PegRNAFoldingFeatures from pe_design.types, "
        "not redefine it"
    )


# ---------- canonical pegRNA (Anzalone 2019, HEK3 +1) ---------- #
#
# This is the prototypical PE2 pegRNA from Anzalone et al. 2019,
# Nature 576, 149-157 (Search-and-replace genome editing) — HEK3 +1
# C->A edit. The canonical 13-nt PBS + 13-nt RTT geometry plus the
# Anzalone-scaffold gives a well-defined RNA of ~126 nt. The MFE is
# pinned via ViennaRNA 2.7.2 itself; the ±0.5 kcal/mol tolerance is
# the plan's published-tolerance for cross-checks.

# 20-nt spacer (RNA letters)
_HEK3_SPACER = "GGCCCAGACUGAGCACGUGA"
# 13-nt PBS (reverse-complement of the 13 nt 5' of the nick)
_HEK3_PBS = "CGUGCUCAGUCUG"
# 13-nt RTT encoding the +1 C->A edit + flanking
_HEK3_RTT = "CUCAUCACCCUCG"

# Canonical Anzalone 2019 SpCas9 sgRNA scaffold (80 nt RNA)
_SGRNA_CANONICAL = (
    "GUUUUAGAGCUAGAAAUAGCAAGUUAAAAUAAGGCUAGUCCGUUAUCAACUUGAAAAAGUGGCACCGAGUCGGUGCUUUU"
)

# Reference MFE from ViennaRNA 2.7.2 on the full HEK3 pegRNA above.
# This is a tight regression lock; published Anzalone figures don't
# report the per-pegRNA MFE so we use the tool itself as the oracle.
_HEK3_PEGRNA_MFE_REF = -49.1  # kcal/mol
_HEK3_PEGRNA_MFE_TOL = 0.5  # ±0.5 kcal/mol per plan §T4 acceptance


def test_anzalone_hek3_pegrna_mfe_within_tolerance():
    """Canonical PE2 pegRNA from Anzalone 2019 HEK3 +1 reproduces the
    pinned MFE within ±0.5 kcal/mol.
    """
    from bionpu.scoring.pegrna_folding import compute_folding_features

    feats = compute_folding_features(
        spacer=_HEK3_SPACER,
        scaffold=_SGRNA_CANONICAL,
        rtt=_HEK3_RTT,
        pbs=_HEK3_PBS,
        scaffold_variant="sgRNA_canonical",
    )

    assert isinstance(feats.mfe_kcal, float)
    assert math.isfinite(feats.mfe_kcal)
    assert abs(feats.mfe_kcal - _HEK3_PEGRNA_MFE_REF) <= _HEK3_PEGRNA_MFE_TOL, (
        f"Anzalone HEK3 pegRNA MFE {feats.mfe_kcal:.3f} drifts beyond "
        f"±{_HEK3_PEGRNA_MFE_TOL} kcal/mol from the regression-locked "
        f"reference {_HEK3_PEGRNA_MFE_REF}"
    )

    # mfe_structure must be a dot-bracket of the same length as the
    # full pegRNA RNA (spacer + scaffold + RTT + PBS).
    expected_len = (
        len(_HEK3_SPACER)
        + len(_SGRNA_CANONICAL)
        + len(_HEK3_RTT)
        + len(_HEK3_PBS)
    )
    assert isinstance(feats.mfe_structure, str)
    assert len(feats.mfe_structure) == expected_len
    assert set(feats.mfe_structure) <= {".", "(", ")"}


# ---------- scaffold disruption: canonical with itself ---------- #


def test_scaffold_disruption_zero_for_canonical_self_pair():
    """A pegRNA whose scaffold-region structure exactly matches the
    canonical scaffold's MFE structure (the trivial null pegRNA: empty
    spacer, empty RTT, empty PBS, scaffold-only) must have
    ``scaffold_disruption == 0.0``.

    This nails down the comparison semantic: scaffold_disruption is
    the FRACTION of canonical scaffold base-pair positions disrupted
    by the rest of the pegRNA, computed against the cached reference
    MFE structure of the chosen scaffold variant.
    """
    from bionpu.scoring.pegrna_folding import compute_folding_features

    feats = compute_folding_features(
        spacer="",
        scaffold=_SGRNA_CANONICAL,
        rtt="",
        pbs="",
        scaffold_variant="sgRNA_canonical",
    )

    # Range-bound by definition.
    assert 0.0 <= feats.scaffold_disruption <= 1.0
    # Trivial pegRNA = scaffold-only ⇒ structure is identical to the
    # cached reference ⇒ disruption is exactly 0.0.
    assert feats.scaffold_disruption == pytest.approx(0.0, abs=1e-9), (
        "Scaffold-only candidate must produce zero disruption against "
        "the canonical reference structure"
    )


# ---------- PBS pairing probability range ---------- #


def test_pbs_pairing_prob_in_unit_interval():
    """``pbs_pairing_prob`` is a probability and must live in [0, 1]
    for any non-trivial pegRNA.
    """
    from bionpu.scoring.pegrna_folding import compute_folding_features

    feats = compute_folding_features(
        spacer=_HEK3_SPACER,
        scaffold=_SGRNA_CANONICAL,
        rtt=_HEK3_RTT,
        pbs=_HEK3_PBS,
        scaffold_variant="sgRNA_canonical",
    )

    assert isinstance(feats.pbs_pairing_prob, float)
    assert math.isfinite(feats.pbs_pairing_prob)
    assert 0.0 <= feats.pbs_pairing_prob <= 1.0, (
        f"pbs_pairing_prob {feats.pbs_pairing_prob} out of [0, 1]"
    )


# ---------- edge case: empty / very short sequences ---------- #


def test_empty_or_short_sequence_handled_gracefully():
    """Edge-case input (empty pegRNA components, sub-fold-compound
    lengths) must not crash. The wrapper returns NaN-tagged features
    OR raises a clean ``ValueError``; either is acceptable per the
    plan's "graceful error or NaN, not crash" criterion.
    """
    from bionpu.scoring.pegrna_folding import compute_folding_features

    # All-empty composite (length 0) — this is meaningless biologically
    # but the wrapper must not segfault / hang ViennaRNA.
    try:
        feats = compute_folding_features(
            spacer="",
            scaffold="",
            rtt="",
            pbs="",
            scaffold_variant="sgRNA_canonical",
        )
    except ValueError:
        # Acceptable failure mode.
        return

    # If the wrapper chose to return a result instead of raising, all
    # numeric fields must be NaN-or-zero (not garbage); structure must
    # be empty.
    assert math.isnan(feats.mfe_kcal) or feats.mfe_kcal == 0.0
    assert feats.mfe_structure == ""
    assert math.isnan(feats.pbs_pairing_prob) or feats.pbs_pairing_prob == 0.0
    assert (
        math.isnan(feats.scaffold_disruption)
        or feats.scaffold_disruption == 0.0
    )
