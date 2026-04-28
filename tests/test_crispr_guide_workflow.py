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

from bionpu.genomics.guide_design import GuideFilter
from bionpu.genomics.guide_workflow import design_guides


def test_design_guides_composes_enumeration_and_seed_prefilter() -> None:
    spacer = "ACGTACGTACGTACGTACGT"
    target = spacer + "AGG"
    seed_mismatch_ref = spacer[:-1] + "A"
    reference = {
        "chrRef": "TT" + spacer + "AGG" + "CC" + seed_mismatch_ref + "TGG",
    }

    run = design_guides(
        target,
        reference,
        chrom="target",
        seed_length=12,
        max_seed_mismatches=1,
        include_reverse=False,
    )

    assert len(run.passing_guides) == 1
    assert run.rejected_guides == ()
    assert len(run.results) == 1

    result = run.results[0]
    assert result.guide.spacer == spacer
    assert result.seed_hit_count >= 2
    assert result.exact_seed_hit_count >= 1
    assert result.mismatched_seed_hit_count >= 1
    assert result.reference_names == ("chrRef",)
    assert run.seed_hit_count == result.seed_hit_count


def test_design_guides_reports_rejected_candidates_but_does_not_prefilter_them() -> None:
    rejected_spacer = "AAAAAAAAAAAAAAAAAAAC"
    passing_spacer = "ACGTACGTACGTACGTACGT"
    target = rejected_spacer + "AGG" + "TTTT" + passing_spacer + "AGG"
    filt = GuideFilter(drop_failed=True)

    run = design_guides(
        target,
        {"chrRef": passing_spacer + "AGG"},
        guide_filter=filt,
        include_reverse=False,
    )

    rejected = [g for g in run.rejected_guides if g.spacer == rejected_spacer]
    passing = [g for g in run.passing_guides if g.spacer == passing_spacer]

    assert rejected
    assert "low_gc" in rejected[0].rejection_reasons
    assert passing
    assert all(r.guide.spacer != rejected_spacer for r in run.results)


def test_design_guides_supports_multiple_pam_templates() -> None:
    spacer = "ACGTACGTACGTACGTACGT"

    run = design_guides(
        spacer + "AAG",
        {"chrRef": spacer + "AAG"},
        pam_templates=("NGG", "NAG"),
        include_reverse=False,
    )

    assert len(run.passing_guides) == 1
    assert run.passing_guides[0].pam == "AAG"
    assert run.seed_hit_count == 1
    assert run.results[0].off_targets[0].pam == "AAG"
