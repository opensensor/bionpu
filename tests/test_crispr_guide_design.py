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

from __future__ import annotations

import pytest

from bionpu.genomics.guide_design import (
    GuideFilter,
    canonical_guide_key,
    enumerate_guides,
    gc_fraction,
    homopolymer_run,
    matches_pam,
    reverse_complement,
)


def test_enumerate_guides_ngg_plus_strand() -> None:
    spacer = "ACGTACGTACGTACGTACGT"
    seq = spacer + "AGG"

    guides = enumerate_guides(seq, chrom="chrT", offset=100, include_reverse=False)

    assert len(guides) == 1
    g = guides[0]
    assert g.spacer == spacer
    assert g.pam == "AGG"
    assert g.chrom == "chrT"
    assert g.window_start == 100
    assert g.spacer_start == 100
    assert g.strand == "+"
    assert g.passes_filters is True
    assert g.rejection_reasons == ()
    assert g.canonical_key == canonical_guide_key(spacer)


def test_enumerate_guides_reverse_strand_coordinate() -> None:
    spacer = "TGCATGCATGCATGCATGCA"
    oriented = spacer + "TGG"
    forward_window = reverse_complement(oriented)

    guides = enumerate_guides(
        forward_window,
        chrom="chrR",
        offset=7,
        include_reverse=True,
    )

    reverse_hits = [g for g in guides if g.strand == "-"]
    assert len(reverse_hits) == 1
    g = reverse_hits[0]
    assert g.spacer == spacer
    assert g.pam == "TGG"
    assert g.window_start == 7
    assert g.spacer_start == 10


def test_filters_can_keep_failed_candidates_with_reasons() -> None:
    low_gc_spacer = "AAAAAAAAAAAAAAAAAAAC"
    seq = low_gc_spacer + "CGG"
    filt = GuideFilter(
        min_gc=0.40,
        max_gc=0.70,
        max_homopolymer=4,
        min_entropy_bits=1.2,
        drop_failed=False,
    )

    guides = enumerate_guides(
        seq,
        guide_filter=filt,
        include_reverse=False,
    )

    assert len(guides) == 1
    g = guides[0]
    assert g.passes_filters is False
    assert "low_gc" in g.rejection_reasons
    assert "homopolymer" in g.rejection_reasons
    assert "low_complexity" in g.rejection_reasons


def test_default_filters_drop_n_spacers() -> None:
    seq = "ACGTACGTACGTACGTACGNAGG"

    assert enumerate_guides(seq, include_reverse=False) == []


def test_multi_pam_templates_accept_nag() -> None:
    spacer = "ACGTACGTACGTACGTACGT"
    seq = spacer + "AAG"

    assert enumerate_guides(seq, include_reverse=False) == []
    guides = enumerate_guides(
        seq,
        pam_templates=("NGG", "NAG"),
        include_reverse=False,
    )
    assert len(guides) == 1
    assert guides[0].pam == "AAG"


def test_basic_sequence_metrics_and_pam_matching() -> None:
    assert gc_fraction("ACGTNN") == pytest.approx(0.5)
    assert homopolymer_run("AAACCCCG") == 4
    assert matches_pam("AGG", "NGG") is True
    assert matches_pam("AAG", "NGG") is False
    assert matches_pam("AAG", "NAG") is True
