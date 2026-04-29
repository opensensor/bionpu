# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# Tests for the BE off-target scan integration (Track A v1).
#
# Coverage:
#   - off_target_scan_for_be_guide API (synthetic genome, end-to-end)
#   - design_base_editor_guides composer (genome path → cfd_aggregate
#     populated; genome=None → cfd_aggregate=NaN)
#   - ranker NaN handling (synbio mode contributes 0 off-target penalty)
#   - composite_be re-rank with off-target penalty (high CFD = lower rank)

from __future__ import annotations

import math
from pathlib import Path

import pytest

from bionpu.genomics.be_design import (
    BaseEditorGuide,
    OffTargetSite,
    composite_be,
    design_base_editor_guides,
    off_target_scan_for_be_guide,
)


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    """Write a multi-record FASTA. Each record is (name, seq)."""
    lines = []
    for name, seq in records:
        lines.append(f">{name}\n")
        # Simple 80-char wrap (FASTA convention; the reader handles both).
        for i in range(0, len(seq), 80):
            lines.append(seq[i : i + 80] + "\n")
    path.write_text("".join(lines))


# --------------------------------------------------------------------------- #
# 1. off_target_scan_for_be_guide — direct API tests.
# --------------------------------------------------------------------------- #


def test_off_target_scan_finds_perfect_on_target_only(tmp_path):
    """A genome with exactly one perfect match returns 0 off-targets.

    The on-target site (0 mismatches AND spacer == input) is filtered
    out; the function returns ``cfd_aggregate=0.0`` and ``count=0``.
    """
    proto = "AAAAACAAAAAAAAAAAAAA"  # 20 nt
    pam = "AGG"
    # Build a minimal genome where the protospacer + AGG appears once.
    genome_seq = "TTTTT" + proto + pam + "TTTTT"
    genome_fa = tmp_path / "genome.fa"
    _write_fasta(genome_fa, [("synthetic", genome_seq)])

    sites, cfd_agg, n_off = off_target_scan_for_be_guide(
        guide_protospacer=proto,
        pam_seq=pam,
        genome_path=str(genome_fa),
        max_mismatches=4,
        device="cpu",
    )

    assert n_off == 0, f"expected 0 off-targets; got {sites!r}"
    assert cfd_agg == 0.0
    assert sites == []


def test_off_target_scan_finds_mismatch_off_target(tmp_path):
    """A genome with one perfect on-target + one 1-mismatch off-target
    returns exactly one off-target site with 0 < CFD <= 1.
    """
    proto = "AAAAACAAAAAAAAAAAAAA"
    pam = "AGG"
    # Off-target: change protospacer position 0 (A → C).
    off_proto = "CAAAACAAAAAAAAAAAAAA"
    genome_seq = (
        "TTTTT"
        + proto + pam               # on-target site (protospacer + NGG)
        + "TTTTT"
        + off_proto + pam            # off-target with 1 mismatch at pos 0
        + "TTTTT"
    )
    genome_fa = tmp_path / "genome.fa"
    _write_fasta(genome_fa, [("synthetic", genome_seq)])

    sites, cfd_agg, n_off = off_target_scan_for_be_guide(
        guide_protospacer=proto,
        pam_seq=pam,
        genome_path=str(genome_fa),
        max_mismatches=4,
        device="cpu",
    )

    assert n_off == 1, f"expected 1 off-target; got {n_off} ({sites!r})"
    site = sites[0]
    assert isinstance(site, OffTargetSite)
    assert site.spacer_genome == off_proto
    assert site.pam_genome == pam
    assert site.mismatches == 1
    assert 0.0 < site.cfd <= 1.0
    assert cfd_agg == pytest.approx(site.cfd, abs=1e-9)


def test_off_target_scan_rejects_bad_protospacer(tmp_path):
    """20-nt-only ACGT-only protospacer; bad input → ValueError."""
    genome_fa = tmp_path / "genome.fa"
    _write_fasta(genome_fa, [("synthetic", "ACGT" * 20)])
    with pytest.raises(ValueError):
        off_target_scan_for_be_guide(
            guide_protospacer="ACG",  # too short
            pam_seq="AGG",
            genome_path=str(genome_fa),
        )
    with pytest.raises(ValueError):
        off_target_scan_for_be_guide(
            guide_protospacer="AAAAACAAAAAAAAAAAAA?",  # non-ACGT
            pam_seq="AGG",
            genome_path=str(genome_fa),
        )


def test_off_target_scan_genome_records_seam(tmp_path):
    """The ``genome_records`` test seam bypasses FASTA reads."""
    proto = "AAAAACAAAAAAAAAAAAAA"
    pam = "AGG"
    genome_seq = "TTTTT" + proto + pam + "TTTTT"
    sites, cfd_agg, n_off = off_target_scan_for_be_guide(
        guide_protospacer=proto,
        pam_seq=pam,
        genome_path="<unused>",
        device="cpu",
        genome_records=[("test", genome_seq)],
    )
    assert n_off == 0
    assert cfd_agg == 0.0
    assert sites == []


# --------------------------------------------------------------------------- #
# 2. design_base_editor_guides composer integration.
# --------------------------------------------------------------------------- #


def test_design_base_editor_guides_synbio_mode_yields_nan_cfd():
    """``genome_path=None`` => synbio mode; cfd_aggregate is NaN per
    guide and off_target_count is 0. The ``NO_OFF_TARGET_SCAN`` note
    is present.
    """
    proto = "AAAAACAAAAAAAAAAAAAA"
    pam = "AGG"
    target_seq = "ACGTACGT" + proto + pam + "ACGTACGT"
    guides = design_base_editor_guides(
        target_seq,
        be_variant="BE4max",
        cas9_variant="wt",
        genome_path=None,
        top_n=10,
    )
    assert guides, "expected at least one guide"
    for g in guides:
        assert math.isnan(float(g.cfd_aggregate)), (
            f"synbio guide should have NaN cfd_aggregate; got {g.cfd_aggregate!r}"
        )
        assert g.off_target_count == 0
        assert "NO_OFF_TARGET_SCAN" in g.notes


def test_design_base_editor_guides_genome_mode_populates_cfd(tmp_path):
    """When ``genome_path`` is supplied, every guide carries a real
    ``cfd_aggregate`` (>= 0.0; finite). The ``NO_OFF_TARGET_SCAN`` note
    is absent.
    """
    proto = "AAAAACAAAAAAAAAAAAAA"
    pam = "AGG"
    # Target sequence has the planted (proto + AGG); the genome FASTA
    # contains the same protospacer once (perfect on-target only).
    target_seq = "ACGTACGT" + proto + pam + "ACGTACGT"
    genome_fa = tmp_path / "genome.fa"
    _write_fasta(genome_fa, [("synthetic_chr", target_seq)])

    guides = design_base_editor_guides(
        target_seq,
        be_variant="BE4max",
        cas9_variant="wt",
        genome_path=genome_fa,
        top_n=10,
    )
    assert guides
    # Find the planted guide — it should have 0 off-targets in this
    # tiny single-locus genome.
    planted = [g for g in guides if g.guide_seq == proto]
    assert planted, f"planted guide not in output: {[g.guide_seq for g in guides]}"
    g = planted[0]
    assert not math.isnan(float(g.cfd_aggregate))
    assert g.cfd_aggregate >= 0.0
    assert g.off_target_count == 0
    assert "NO_OFF_TARGET_SCAN" not in g.notes


def test_design_base_editor_guides_high_offtarget_ranks_lower(tmp_path):
    """Two guides with identical edit-window/bystander signatures but
    different off-target loads should rank in the expected order:
    high-off-target guide ranks LOWER than low-off-target guide.

    Construction:
      - Guide A is planted in target + genome with NO close matches.
      - Guide B is planted in target + genome WITH 3 close (1-mm)
        matches injected.
    Both guides have C at protospacer position 5 (in BE4max window),
    so their in-window/bystander signatures are identical.
    """
    proto_A = "AAAAACAAAAAAAAAAAAAT"  # ends in T to differ from B
    proto_B = "TTTTTCTTTTTTTTTTTTTA"
    pam = "AGG"

    # Target sequence has BOTH planted protospacers, so both are
    # candidate guides for the design composer.
    target_seq = (
        "ACGTACGT"
        + proto_A + pam
        + "ACGTACGTACGT"
        + proto_B + pam
        + "ACGTACGT"
    )

    # Genome: Guide A appears once (on-target only). Guide B appears
    # once (on-target) PLUS 3 close-mismatch off-targets.
    proto_B_mm1 = "ATTTTCTTTTTTTTTTTTTA"  # T->A at pos 1
    proto_B_mm2 = "TTTATCTTTTTTTTTTTTTA"  # T->A at pos 3
    proto_B_mm3 = "TTTTTCTTATTTTTTTTTTA"  # T->A at pos 8
    genome_seq = (
        "GGGGGGGGGGGGGG"
        + proto_A + pam
        + "GGGGGGGGGGGGGG"
        + proto_B + pam
        + "GGGGGGGGGGGGGG"
        + proto_B_mm1 + pam
        + "GGGGGGGGGGGGGG"
        + proto_B_mm2 + pam
        + "GGGGGGGGGGGGGG"
        + proto_B_mm3 + pam
        + "GGGGGGGGGGGGGG"
    )
    genome_fa = tmp_path / "genome.fa"
    _write_fasta(genome_fa, [("synth_chr", genome_seq)])

    guides = design_base_editor_guides(
        target_seq,
        be_variant="BE4max",
        cas9_variant="wt",
        genome_path=genome_fa,
        top_n=20,
    )

    g_A = next((g for g in guides if g.guide_seq == proto_A), None)
    g_B = next((g for g in guides if g.guide_seq == proto_B), None)
    assert g_A is not None and g_B is not None, (
        f"both planted guides should be present; got "
        f"A={g_A!r} B={g_B!r}; all={[g.guide_seq for g in guides]}"
    )

    # Both guides should have C at position 5 (in BE4max window).
    assert g_A.in_activity_window
    assert g_B.in_activity_window

    # Guide B should have strictly more (close-mismatch) off-targets
    # than guide A. We don't pin to absolute counts because the
    # synthetic genome's flanking GGGGG runs can produce incidental
    # high-mismatch hits whose CFD is 0 (and which therefore don't
    # affect ranking).
    assert g_B.off_target_count > g_A.off_target_count, (
        f"expected guide B off_target_count > guide A; got "
        f"A={g_A.off_target_count} B={g_B.off_target_count}"
    )
    # Guide B's CFD aggregate should be strictly greater than A's
    # (the planted 1-mismatch off-targets contribute non-zero CFD).
    assert g_B.cfd_aggregate > g_A.cfd_aggregate, (
        f"expected guide B cfd_aggregate > A's; got "
        f"A={g_A.cfd_aggregate} B={g_B.cfd_aggregate}"
    )

    # Guide B should rank lower (smaller rank_score) than guide A.
    assert g_B.rank_score < g_A.rank_score, (
        f"high-off-target guide should rank lower; "
        f"A.rank_score={g_A.rank_score} B.rank_score={g_B.rank_score}"
    )

    # And in the sorted output, A appears before B.
    a_idx = guides.index(g_A)
    b_idx = guides.index(g_B)
    assert a_idx < b_idx


# --------------------------------------------------------------------------- #
# 3. composite_be / NaN handling.
# --------------------------------------------------------------------------- #


def _mk_guide(
    *,
    in_window: bool,
    bystander_count: int = 0,
    cfd_aggregate: float = 0.0,
    off_target_count: int = 0,
) -> BaseEditorGuide:
    return BaseEditorGuide(
        guide_seq="A" * 20,
        pam_seq="AGG",
        target_pos=0,
        target_base="C",
        target_pos_in_protospacer=5,
        in_activity_window=in_window,
        bystander_count=bystander_count,
        bystander_positions=(),
        off_target_count=off_target_count,
        cfd_aggregate=cfd_aggregate,
        rank_score=0.0,
        notes="",
    )


def test_composite_be_handles_nan_cfd_as_zero_penalty():
    """``cfd_aggregate=NaN`` (synbio) contributes 0 to the off-target
    penalty, so a synbio in-window guide scores +1.0 (window bonus).
    """
    g = _mk_guide(in_window=True, bystander_count=0, cfd_aggregate=float("nan"))
    assert composite_be(g) == pytest.approx(1.0, abs=1e-12)


def test_composite_be_off_target_penalty_subtracts_from_score():
    """A guide with a high CFD aggregate scores lower than the same
    guide with cfd_aggregate=0.
    """
    g_clean = _mk_guide(in_window=True, bystander_count=0, cfd_aggregate=0.0)
    g_dirty = _mk_guide(in_window=True, bystander_count=0, cfd_aggregate=2.0)
    s_clean = composite_be(g_clean)
    s_dirty = composite_be(g_dirty)
    # composite_be: in_window=+1.0; -0.2 * 0; -0.5 * cfd
    assert s_clean == pytest.approx(1.0, abs=1e-12)
    assert s_dirty == pytest.approx(1.0 - 0.5 * 2.0, abs=1e-12)
    assert s_dirty < s_clean


def test_composite_be_bystander_and_offtarget_compose():
    """Both bystander and off-target penalties stack."""
    g = _mk_guide(in_window=True, bystander_count=2, cfd_aggregate=1.0)
    # 1.0 (window) - 0.2 * 2 (bystander) - 0.5 * 1.0 (off-target) = -0.1 + 1 - 0.4 - 0.5 = 0.1
    assert composite_be(g) == pytest.approx(1.0 - 0.4 - 0.5, abs=1e-12)
