# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

"""Unit tests for ``bionpu.genomics.library_design.controls``."""

from __future__ import annotations

import pytest

from bionpu.genomics.library_design.controls import (
    CANONICAL_ESSENTIAL_GENE_GUIDES,
    CANONICAL_SAFE_HARBOR_GUIDES,
    ControlGuide,
    generate_controls,
    generate_non_targeting_controls,
)


def test_canonical_safe_harbor_set_contains_aavs1_ccr5_rosa26():
    labels = {g.target_label for g in CANONICAL_SAFE_HARBOR_GUIDES}
    # Brief requires AAVS1, CCR5, ROSA26 canonical guides.
    assert any("AAVS1" in lbl for lbl in labels)
    assert any("CCR5" in lbl for lbl in labels)
    assert any("ROSA26" in lbl for lbl in labels)
    for g in CANONICAL_SAFE_HARBOR_GUIDES:
        assert g.control_class == "safe_harbor"
        assert len(g.guide_seq) == 20
        assert all(c in "ACGT" for c in g.guide_seq)
        assert len(g.pam_seq) == 3
        assert g.notes  # citation present


def test_canonical_essential_set_contains_rps19_rpl15():
    labels = {g.target_label for g in CANONICAL_ESSENTIAL_GENE_GUIDES}
    # Brief requires RPS19 + RPL15 essential-gene positive controls.
    assert any("RPS19" in lbl for lbl in labels)
    assert any("RPL15" in lbl for lbl in labels)
    for g in CANONICAL_ESSENTIAL_GENE_GUIDES:
        assert g.control_class == "essential_gene"
        assert len(g.guide_seq) == 20
        assert all(c in "ACGT" for c in g.guide_seq)
        assert g.notes


def test_generate_non_targeting_returns_n_unique_acgt_spacers():
    n = 50
    spacers = generate_non_targeting_controls(n=n, rng_seed=1)
    assert len(spacers) == n
    seqs = {s.guide_seq for s in spacers}
    # All ACGT 20-mers, all unique (50 is well under collision threshold).
    assert len(seqs) == n
    for s in spacers:
        assert len(s.guide_seq) == 20
        assert all(c in "ACGT" for c in s.guide_seq)
        assert s.control_class == "non_targeting"
        assert s.chrom == ""
        assert s.start == -1
        assert s.end == -1
        assert s.target_label.startswith("non_targeting_")


def test_generate_non_targeting_is_deterministic_across_runs():
    a = generate_non_targeting_controls(n=20, rng_seed=42)
    b = generate_non_targeting_controls(n=20, rng_seed=42)
    assert [s.guide_seq for s in a] == [s.guide_seq for s in b]


def test_generate_non_targeting_with_genome_skips_matches():
    """Non-targeting candidates must skip spacers found in the genome lookup."""
    # We'll seed a fake genome that contains the FIRST candidate that
    # the rng would otherwise emit. The generator should retry until
    # it finds a non-matching spacer.
    first_two = generate_non_targeting_controls(n=2, rng_seed=7)
    fake_genome = {"chr_fake": first_two[0].guide_seq}
    out = generate_non_targeting_controls(
        n=2, rng_seed=7, genome_seq_lookup=fake_genome
    )
    seqs = {s.guide_seq for s in out}
    assert first_two[0].guide_seq not in seqs


def test_generate_non_targeting_zero_returns_empty():
    assert generate_non_targeting_controls(n=0) == []


def test_generate_non_targeting_rejects_negative_n():
    with pytest.raises(ValueError, match="n must be >= 0"):
        generate_non_targeting_controls(n=-1)


def test_generate_non_targeting_filters_disqualifying_motifs():
    """Spacers with TTTT or homopolymer>=5 must not appear in output."""
    out = generate_non_targeting_controls(n=200, rng_seed=11)
    for s in out:
        assert "TTTT" not in s.guide_seq
        for base in "ACGT":
            assert base * 5 not in s.guide_seq


def test_generate_controls_combines_all_three_classes():
    out = generate_controls(n_non_targeting=5)
    classes = [c.control_class for c in out]
    # Order: non_targeting (5) + safe_harbor (3) + essential (2) = 10
    assert classes[:5] == ["non_targeting"] * 5
    assert classes[5:8] == ["safe_harbor"] * 3
    assert classes[8:10] == ["essential_gene"] * 2
    assert len(out) == 10


def test_generate_controls_can_omit_classes():
    out_nt_only = generate_controls(
        n_non_targeting=3,
        include_safe_harbor=False,
        include_essential_gene=False,
    )
    assert len(out_nt_only) == 3
    assert all(c.control_class == "non_targeting" for c in out_nt_only)


def test_control_guide_immutable_dataclass():
    """ControlGuide is frozen so it can't be mutated downstream."""
    g = CANONICAL_SAFE_HARBOR_GUIDES[0]
    with pytest.raises((AttributeError, TypeError)):
        g.guide_seq = "AAAAAAAAAAAAAAAAAAAA"  # type: ignore[misc]
    assert isinstance(g, ControlGuide)
