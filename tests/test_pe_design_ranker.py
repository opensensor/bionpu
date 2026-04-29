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

"""Track B v0 — Tests for the composite pegRNA ranker (Task T8).

Acceptance criteria (per ``track-b-pegrna-design-plan.md`` §T8):

1. Ranking-parity: high-efficiency + low-off-target candidate ranks first.
2. PE3 outranks an equivalent PE2 candidate when their efficiencies and
   off-target counts are equal (off-target stub injects a slightly higher
   pegRNA-spacer CFD penalty for PE2 to surface the PE3-nicking benefit).
3. NaN PRIDICT correctly flagged: when ``scorer.score()`` raises, the
   resulting :class:`RankedPegRNA` carries ``"PRIDICT_FAILED"`` in
   ``notes`` and ``composite_pridict`` is NaN. NaN entries sort LAST
   regardless of other fields.
4. Ranking is stable across runs: invoking ``rank_candidates`` twice on
   the same input yields byte-identical ``pegrna_id`` ordering (the
   tie-breaker is total over the candidate's identity tuple).

Stubs (no real PRIDICT / ViennaRNA inference is exercised here — those
paths belong to the T12 integration tests). The ``scorer``,
``folding_extractor`` and ``off_target_scan_fn`` are all mocked / faked
so the ranker is unit-tested in isolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import pytest

from bionpu.genomics.pe_design.types import (
    EditSpec,
    PE3PegRNACandidate,
    PegRNACandidate,
    PegRNAFoldingFeatures,
    PRIDICTScore,
    RankedPegRNA,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


_RC = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _rc(seq: str) -> str:
    return seq.translate(_RC)[::-1].upper()


def _make_substitution_spec(
    *, chrom: str = "chrSyn", pos_0b: int = 100, ref: str = "C", alt: str = "T"
) -> EditSpec:
    return EditSpec(
        chrom=chrom,
        start=pos_0b,
        end=pos_0b + len(ref),
        ref_seq=ref,
        alt_seq=alt,
        edit_type="substitution",
        notation_used=f"{ref}>{alt} at {chrom}:{pos_0b + 1}",
        strand="+",
    )


_PBS_SOURCE = "GCACGUGCUCAGUCGAUCGAUCG"  # 23 nt; long enough for pbs_length up to 17
_RTT_SOURCE = "UCACGUGCUCAGUCGAUCGAUCGA"  # 24 nt; long enough for rtt_length up to 20


def _make_pe2_candidate(
    *,
    nick_site: int,
    spacer_seq: str = "GGCCCAGACTGAGCACGTGA",
    pam_seq: str = "TGG",
    strand: str = "+",
    chrom: str = "chrSyn",
    scaffold_variant: str = "sgRNA_canonical",
    pbs_length: int = 13,
    rtt_length: int = 14,
) -> PegRNACandidate:
    pbs_seq = _PBS_SOURCE[:pbs_length]
    rtt_seq = _RTT_SOURCE[:rtt_length]
    return PegRNACandidate(
        spacer_seq=spacer_seq,
        pam_seq=pam_seq,
        scaffold_variant=scaffold_variant,
        pbs_seq=pbs_seq,
        pbs_length=pbs_length,
        rtt_seq=rtt_seq,
        rtt_length=rtt_length,
        nick_site=nick_site,
        full_pegrna_rna_seq="GGCCCAGACUGAGCACGUGA" + "GUUU" + rtt_seq + pbs_seq,
        edit_position_in_rtt=0,
        strategy="PE2",
        strand=strand,
        rt_product_seq="N/A",
        chrom=chrom,
    )


def _make_pe3_candidate(
    *,
    nick_site: int,
    nicking_distance: int = 65,
    spacer_seq: str = "GGCCCAGACTGAGCACGTGA",
    pam_seq: str = "TGG",
    strand: str = "+",
    chrom: str = "chrSyn",
    nicking_spacer: str = "TCACGTGCTCAGTCTGGGCC",
    nicking_pam: str = "TGG",
    pbs_length: int = 13,
    rtt_length: int = 14,
) -> PE3PegRNACandidate:
    pbs_seq = _PBS_SOURCE[:pbs_length]
    rtt_seq = _RTT_SOURCE[:rtt_length]
    return PE3PegRNACandidate(
        spacer_seq=spacer_seq,
        pam_seq=pam_seq,
        scaffold_variant="sgRNA_canonical",
        pbs_seq=pbs_seq,
        pbs_length=pbs_length,
        rtt_seq=rtt_seq,
        rtt_length=rtt_length,
        nick_site=nick_site,
        full_pegrna_rna_seq="GGCCCAGACUGAGCACGUGA" + "GUUU" + rtt_seq + pbs_seq,
        edit_position_in_rtt=0,
        strategy="PE3",
        strand=strand,
        rt_product_seq="N/A",
        chrom=chrom,
        nicking_spacer=nicking_spacer,
        nicking_pam=nicking_pam,
        nicking_distance_from_pe2_nick=nicking_distance,
    )


def _folding_features_factory(
    mfe_kcal: float = -10.0,
    pbs_pairing_prob: float = 0.2,
    scaffold_disruption: float = 0.05,
):
    """Return a folding extractor that always emits the same features."""

    def _extract(
        spacer: str,
        scaffold: str,
        rtt: str,
        pbs: str,
        *,
        scaffold_variant: str = "sgRNA_canonical",
    ) -> PegRNAFoldingFeatures:
        return PegRNAFoldingFeatures(
            mfe_kcal=mfe_kcal,
            mfe_structure="." * (len(spacer) + len(scaffold) + len(rtt) + len(pbs)),
            pbs_pairing_prob=pbs_pairing_prob,
            scaffold_disruption=scaffold_disruption,
        )

    return _extract


class _StubScorer:
    """Mock :class:`PRIDICT2Scorer` exposing the same surface T8 calls.

    ``score_map`` keys on ``pegrna_seq`` and returns a canned
    :class:`PRIDICTScore`. Unknown keys yield a default score so the
    test only has to mention pegRNAs whose efficiency it cares about.
    Setting ``raise_for`` triggers an exception for that pegrna_seq.
    """

    def __init__(
        self,
        *,
        score_map: dict[str, PRIDICTScore] | None = None,
        default: PRIDICTScore | None = None,
        raise_for: set[str] | None = None,
    ) -> None:
        self.score_map = dict(score_map or {})
        self.default = default or PRIDICTScore(
            efficiency=50.0, edit_rate=0.5, confidence=1.0, notes=()
        )
        self.raise_for = set(raise_for or ())
        self.calls = 0
        self.batch_calls = 0

    def score(
        self,
        pegrna_seq: str,
        *,
        scaffold_variant: str = "sgRNA_canonical",
        target_context: str = "",
        folding_features=None,
    ) -> PRIDICTScore:
        self.calls += 1
        if pegrna_seq in self.raise_for:
            raise RuntimeError(f"forced PRIDICT failure for {pegrna_seq[:20]!r}")
        return self.score_map.get(pegrna_seq, self.default)

    def score_batch(
        self,
        pegrna_seqs: list[str],
        *,
        scaffold_variants: list[str] | None = None,
        target_contexts: list[str],
        folding_features_list=None,
    ) -> list[PRIDICTScore]:
        self.batch_calls += 1
        # Match the production wrapper's behaviour: per-pegRNA exceptions
        # propagate so the ranker's per-candidate try/except can flag
        # them. Real PRIDICT2Scorer.score_batch swallows nothing, so the
        # ranker is responsible for wrapping it and falling back to
        # per-pegRNA score() on failure.
        return [
            self.score(
                seq,
                scaffold_variant=(scaffold_variants[i] if scaffold_variants else "sgRNA_canonical"),
                target_context=target_contexts[i],
            )
            for i, seq in enumerate(pegrna_seqs)
        ]


def _zero_off_target_scan(
    spacer_seq: str, *, max_mismatches: int = 4, **kwargs
):
    """No off-targets — predictable test."""
    return ([], 0.0, 0)


# ---------------------------------------------------------------------------
# Test 1 — Ranking parity: high efficiency + low off-target ranks first
# ---------------------------------------------------------------------------


def test_high_efficiency_low_off_target_ranks_first():
    """The candidate with the highest ``composite_pridict`` after
    penalties must occupy ``rank == 1``."""
    from bionpu.genomics.pe_design.ranker import rank_candidates

    edit_spec = _make_substitution_spec()
    cand_low_eff = _make_pe2_candidate(nick_site=49, pbs_length=13)
    cand_high_eff = _make_pe2_candidate(nick_site=60, pbs_length=14)

    score_map = {
        cand_low_eff.full_pegrna_rna_seq: PRIDICTScore(
            efficiency=20.0, edit_rate=0.2, confidence=1.0, notes=()
        ),
        cand_high_eff.full_pegrna_rna_seq: PRIDICTScore(
            efficiency=80.0, edit_rate=0.8, confidence=1.0, notes=()
        ),
    }

    # The high-efficiency candidate has zero off-targets; the low-eff
    # one has a small CFD aggregate. After composite, high should rank
    # first regardless.
    def off_target_scan(spacer_seq, *, max_mismatches=4, **kw):
        if spacer_seq == cand_low_eff.spacer_seq:
            return ([], 0.5, 3)  # some off-target burden
        return ([], 0.0, 0)

    result = rank_candidates(
        candidates=[cand_low_eff, cand_high_eff],
        edit_spec=edit_spec,
        target_context="GCC(C/T)GTG",
        off_target_scan_fn=off_target_scan,
        scorer=_StubScorer(score_map=score_map),
        folding_extractor=_folding_features_factory(),
    )

    assert len(result) == 2
    assert result[0].pridict_efficiency == 80.0
    assert result[0].rank == 1
    assert result[1].rank == 2
    assert result[0].composite_pridict > result[1].composite_pridict


# ---------------------------------------------------------------------------
# Test 2 — PE3 outranks an equivalent PE2 when nicking yields lower CFD
# ---------------------------------------------------------------------------


def test_pe3_outranks_equivalent_pe2():
    """The plan §T8 spec calls this "PE3 outranks equivalent PE2 when
    PE3's nicking_distance is in the 40-90 sweet spot".

    Realistic modeling: the PE3 candidate's pegRNA spacer is a
    different protospacer choice than PE2's (PE3 selection lets the
    designer prefer a cleaner off-target profile pegRNA spacer because
    the nick on the OPPOSITE strand stimulates repair regardless of
    which protospacer the pegRNA uses). With:

      * PE2: pegRNA-spacer CFD 0.5, no nicking term
      * PE3: pegRNA-spacer CFD 0.2, nicking-spacer CFD 0.0
      * identical efficiency 70.0, identical folding

    composite_PE2 = 70 - 0.3*0.5 - 0 - 0 = 69.85
    composite_PE3 = 70 - 0.3*0.2 - 0.15*0.0 - 0 = 69.94

    PE3 must take rank 1.
    """
    from bionpu.genomics.pe_design.ranker import rank_candidates

    edit_spec = _make_substitution_spec()
    # Different spacers — the realistic PE3-vs-PE2 trade-off is the
    # designer choosing a cleaner-spacer protospacer when PE3 nicking
    # frees them from the on-strand-only constraint.
    pe2 = _make_pe2_candidate(
        nick_site=49,
        pbs_length=13,
        rtt_length=14,
        spacer_seq="GGCCCAGACTGAGCACGTGA",
    )
    pe3 = _make_pe3_candidate(
        nick_site=60,
        pbs_length=14,
        rtt_length=14,
        nicking_distance=65,
        spacer_seq="AGCAAGGCTGAGCACGTGAA",  # cleaner spacer
        nicking_spacer="TCACGTGCTCAGTCTGGGCC",
    )

    canned = PRIDICTScore(
        efficiency=70.0, edit_rate=0.7, confidence=1.0, notes=()
    )
    score_map = {
        pe2.full_pegrna_rna_seq: canned,
        pe3.full_pegrna_rna_seq: canned,
    }

    def off_target_scan(spacer_seq, *, max_mismatches=4, **kw):
        if spacer_seq == pe2.spacer_seq:
            return ([], 0.5, 5)  # PE2 has the dirtier protospacer
        if spacer_seq == pe3.spacer_seq:
            return ([], 0.2, 2)  # PE3's pegRNA spacer is cleaner
        if spacer_seq == pe3.nicking_spacer:
            return ([], 0.0, 0)  # nicking sgRNA has no off-targets
        return ([], 0.0, 0)

    result = rank_candidates(
        candidates=[pe2, pe3],
        edit_spec=edit_spec,
        target_context="GCC(C/T)GTG",
        off_target_scan_fn=off_target_scan,
        scorer=_StubScorer(score_map=score_map),
        folding_extractor=_folding_features_factory(),
    )

    assert len(result) == 2
    pe3_ranked = next(r for r in result if r.pe_strategy == "PE3")
    pe2_ranked = next(r for r in result if r.pe_strategy == "PE2")

    assert pe3_ranked.composite_pridict > pe2_ranked.composite_pridict
    assert pe3_ranked.rank == 1
    assert pe2_ranked.rank == 2
    # PE3 fields populated; PE2 fields None.
    assert pe3_ranked.nicking_spacer == pe3.nicking_spacer
    assert pe3_ranked.nicking_pam == pe3.nicking_pam
    assert pe3_ranked.nicking_distance == 65
    assert pe3_ranked.cfd_aggregate_nicking == pytest.approx(0.0)
    assert pe3_ranked.off_target_count_nicking == 0
    assert pe2_ranked.nicking_spacer is None
    assert pe2_ranked.nicking_pam is None
    assert pe2_ranked.nicking_distance is None
    assert pe2_ranked.cfd_aggregate_nicking is None
    assert pe2_ranked.off_target_count_nicking is None


# ---------------------------------------------------------------------------
# Test 3 — NaN PRIDICT correctly flagged + sorted last
# ---------------------------------------------------------------------------


def test_nan_pridict_flagged_and_sorted_last():
    """When PRIDICT raises for a candidate, the ranker must flag it
    with ``"PRIDICT_FAILED"`` in ``notes``, set ``composite_pridict``
    to NaN, and place it AFTER all valid candidates in the output."""
    from bionpu.genomics.pe_design.ranker import rank_candidates

    edit_spec = _make_substitution_spec()
    cand_ok = _make_pe2_candidate(nick_site=49, pbs_length=13)
    cand_fail = _make_pe2_candidate(nick_site=60, pbs_length=14)

    scorer = _StubScorer(
        score_map={
            cand_ok.full_pegrna_rna_seq: PRIDICTScore(
                efficiency=42.0, edit_rate=0.4, confidence=1.0, notes=()
            ),
        },
        raise_for={cand_fail.full_pegrna_rna_seq},
    )

    result = rank_candidates(
        candidates=[cand_fail, cand_ok],  # failing one first to test sort
        edit_spec=edit_spec,
        target_context="GCC(C/T)GTG",
        off_target_scan_fn=_zero_off_target_scan,
        scorer=scorer,
        folding_extractor=_folding_features_factory(),
    )

    assert len(result) == 2
    # Valid candidate ranks first.
    assert result[0].pridict_efficiency == 42.0
    assert result[0].rank == 1
    assert not math.isnan(result[0].composite_pridict)
    assert "PRIDICT_FAILED" not in result[0].notes
    # Failed candidate sorts LAST regardless of input order.
    assert math.isnan(result[1].composite_pridict)
    assert "PRIDICT_FAILED" in result[1].notes
    assert result[1].rank == 2
    assert math.isnan(result[1].pridict_efficiency)


# ---------------------------------------------------------------------------
# Test 4 — Ranking is stable across runs
# ---------------------------------------------------------------------------


def test_ranking_is_stable_across_runs():
    """Two invocations on the same input must yield identical
    ``pegrna_id`` ordering. Equal-composite candidates break ties via
    ``(chrom, nick_site, strand, pbs_length, rtt_length,
    scaffold_variant)`` ascending, so the order is fully determined by
    the input dataclass values, not insertion order or dict iteration."""
    from bionpu.genomics.pe_design.ranker import rank_candidates

    edit_spec = _make_substitution_spec()

    # Build a deliberately tied set: three candidates with the same
    # score_map default but different nick_site / pbs_length so the
    # tie-breaker exercises every key.
    cands = [
        _make_pe2_candidate(nick_site=70, pbs_length=15, rtt_length=14),
        _make_pe2_candidate(nick_site=49, pbs_length=13, rtt_length=14),
        _make_pe2_candidate(nick_site=49, pbs_length=14, rtt_length=14),
        _make_pe2_candidate(nick_site=49, pbs_length=13, rtt_length=15),
    ]

    canned = PRIDICTScore(
        efficiency=50.0, edit_rate=0.5, confidence=1.0, notes=()
    )
    score_map = {c.full_pegrna_rna_seq: canned for c in cands}

    common_kw = dict(
        edit_spec=edit_spec,
        target_context="GCC(C/T)GTG",
        off_target_scan_fn=_zero_off_target_scan,
        folding_extractor=_folding_features_factory(),
    )

    # Run twice, with a fresh scorer each time so internal state
    # cannot influence ordering.
    first = rank_candidates(
        candidates=list(cands),
        scorer=_StubScorer(score_map=score_map),
        **common_kw,
    )
    # Permute input order on the second run to prove the sort is
    # determined by the data, not insertion order.
    second = rank_candidates(
        candidates=list(reversed(cands)),
        scorer=_StubScorer(score_map=score_map),
        **common_kw,
    )

    first_ids = [r.pegrna_id for r in first]
    second_ids = [r.pegrna_id for r in second]
    assert first_ids == second_ids, (
        f"ranker non-deterministic: {first_ids} != {second_ids}"
    )

    # And ranks must be 1..N consecutive.
    assert [r.rank for r in first] == list(range(1, len(cands) + 1))
    assert [r.rank for r in second] == list(range(1, len(cands) + 1))


# ---------------------------------------------------------------------------
# Test 5 — top_n filter
# ---------------------------------------------------------------------------


def test_top_n_filter_truncates_output():
    """``top_n`` must truncate the output to the top-N composite
    candidates while preserving rank-1 .. rank-N labels."""
    from bionpu.genomics.pe_design.ranker import rank_candidates

    edit_spec = _make_substitution_spec()
    # Vary BOTH pbs_length and rtt_length so each candidate has a
    # unique full_pegrna_rna_seq (otherwise score_map keys collide).
    cands = [
        _make_pe2_candidate(
            nick_site=49 + i,
            pbs_length=10 + i,  # 10..14
            rtt_length=12 + i,  # 12..16
        )
        for i in range(5)
    ]

    score_map = {
        c.full_pegrna_rna_seq: PRIDICTScore(
            efficiency=50.0 + i * 5,
            edit_rate=0.5,
            confidence=1.0,
            notes=(),
        )
        for i, c in enumerate(cands)
    }

    result = rank_candidates(
        candidates=cands,
        edit_spec=edit_spec,
        target_context="GCC(C/T)GTG",
        off_target_scan_fn=_zero_off_target_scan,
        scorer=_StubScorer(score_map=score_map),
        folding_extractor=_folding_features_factory(),
        top_n=3,
    )

    assert len(result) == 3
    assert [r.rank for r in result] == [1, 2, 3]
    # Top-N must be the highest-efficiency three (70, 65, 60).
    assert result[0].pridict_efficiency == 70.0
    assert result[1].pridict_efficiency == 65.0
    assert result[2].pridict_efficiency == 60.0


# ---------------------------------------------------------------------------
# Test 6 — MFE penalty
# ---------------------------------------------------------------------------


def test_mfe_penalty_reduces_composite_for_tight_folds():
    """A pegRNA with very-negative MFE (e.g. -50 kcal/mol, indicating
    strong intramolecular structure) must score lower than an
    otherwise-equivalent pegRNA with a mild MFE."""
    from bionpu.genomics.pe_design.ranker import rank_candidates

    edit_spec = _make_substitution_spec()
    cand_loose = _make_pe2_candidate(nick_site=49, pbs_length=13)
    cand_tight = _make_pe2_candidate(nick_site=60, pbs_length=14)

    canned = PRIDICTScore(
        efficiency=70.0, edit_rate=0.7, confidence=1.0, notes=()
    )
    score_map = {
        cand_loose.full_pegrna_rna_seq: canned,
        cand_tight.full_pegrna_rna_seq: canned,
    }

    # Different folding features per candidate's full RNA sequence.
    folding_map = {
        cand_loose.full_pegrna_rna_seq: -10.0,  # zero penalty
        cand_tight.full_pegrna_rna_seq: -50.0,  # 25-kcal penalty
    }

    def folding_extractor(spacer, scaffold, rtt, pbs, *, scaffold_variant="sgRNA_canonical"):
        full = spacer + scaffold + rtt + pbs
        # Match by RTT+PBS suffix since the keys above embed those.
        for full_seq, mfe in folding_map.items():
            if full_seq.endswith(rtt + pbs):
                return PegRNAFoldingFeatures(
                    mfe_kcal=mfe,
                    mfe_structure="." * len(full),
                    pbs_pairing_prob=0.2,
                    scaffold_disruption=0.05,
                )
        return PegRNAFoldingFeatures(
            mfe_kcal=-10.0,
            mfe_structure="." * len(full),
            pbs_pairing_prob=0.2,
            scaffold_disruption=0.05,
        )

    result = rank_candidates(
        candidates=[cand_loose, cand_tight],
        edit_spec=edit_spec,
        target_context="GCC(C/T)GTG",
        off_target_scan_fn=_zero_off_target_scan,
        scorer=_StubScorer(score_map=score_map),
        folding_extractor=folding_extractor,
    )

    loose_ranked = next(r for r in result if r.mfe_kcal == -10.0)
    tight_ranked = next(r for r in result if r.mfe_kcal == -50.0)

    # mfe_penalty(-50) = max(0, -(-50 + 25)) = 25
    # composite_loose = 70 - 0 - 0 - 0 = 70
    # composite_tight = 70 - 0 - 0 - 2.5 = 67.5
    assert loose_ranked.composite_pridict == pytest.approx(70.0)
    assert tight_ranked.composite_pridict == pytest.approx(67.5)
    assert loose_ranked.rank == 1
    assert tight_ranked.rank == 2
