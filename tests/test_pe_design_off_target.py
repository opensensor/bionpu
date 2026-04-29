# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

"""Track B v0 — Tests for the off-target adapter (Task T11).

Acceptance criteria (per ``track-b-pegrna-design-plan.md`` §T11):

1. Adapter delegates correctly: a known spacer + small synthetic genome
   yields ``(list[OffTargetSite], cfd_aggregate, count)`` of the right
   shape, with every emitted ``OffTargetSite`` carrying valid fields.
2. ``OffTargetSite`` re-shape preserves CFD scores: the per-site
   ``cfd_score`` we surface equals the value the locked
   :class:`bionpu.scoring.cfd.CFDScorer` produces for the same row
   (no re-implementation; we delegate).
3. Counts for a known guide match the same scan path
   :func:`bionpu.genomics.crispr_design.design_guides_for_target` uses.
   The adapter is a thin wrapper around
   :func:`bionpu.scan.cpu_scan` + :class:`bionpu.scoring.cfd.CFDScorer`
   — exactly the primitives ``crispr_design._scan_locus_for_offtargets``
   /``_score_off_target_cfd`` chain. We assert the off-target row count
   and per-row CFD scores agree byte-equal between the adapter and the
   crispr_design pipeline for an identical input spacer.

Lock-discipline reminder
------------------------
T11 sits ABOVE the lock layer. ``crispr_design``'s scan path is already
wrapped in :func:`bionpu.dispatch.npu_silicon_lock.npu_silicon_lock`
for the silicon path AND uses the in-process ``_dispatch_lock`` for
the pyxrt path per ``CLAUDE.md``. The adapter MUST NOT introduce its
own lock call; callers in subprocess harnesses are responsible for
holding the silicon lock around their entire harness.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest


SMOKE_CHROM = "chrSyn"
SMOKE_LOCUS_LEN = 6000
SMOKE_SEED = 0xC15B  # determinism: any non-zero seed works
# Hand-place a guaranteed forward-strand on-target site at the start of
# the locus so we have one perfect-match anchor. The 23-mer below is
# the spacer + NGG PAM. We use a non-trivial spacer (mix of all four
# bases) so any future scan-path change that drops e.g. polyG sites
# still leaves the smoke covered.
KNOWN_SPACER = "GCACTAGTACGCATCGTACA"  # 20 nt; mix of A/C/G/T
KNOWN_PAM = "TGG"  # NGG
KNOWN_SITE_OFFSET = 1000  # 0-indexed within the locus


def _build_synthetic_genome(tmp_path: Path) -> tuple[Path, str]:
    """Build a 6 kbp single-record synthetic chromosome.

    Returns ``(fasta_path, sequence_str)``. The sequence contains one
    hand-placed perfect-match site for ``KNOWN_SPACER`` at offset
    ``KNOWN_SITE_OFFSET`` so the adapter has a guaranteed on-target
    hit; off-targets emerge organically from random ACGT bases.
    """
    rng = random.Random(SMOKE_SEED)
    seq = list(rng.choices("ACGT", k=SMOKE_LOCUS_LEN))
    target_23mer = KNOWN_SPACER + KNOWN_PAM
    for i, c in enumerate(target_23mer):
        seq[KNOWN_SITE_OFFSET + i] = c
    body = "".join(seq)
    fasta = tmp_path / "synth.fa"
    with fasta.open("w", encoding="ascii") as fh:
        fh.write(f">{SMOKE_CHROM}\n")
        for i in range(0, len(body), 80):
            fh.write(body[i : i + 80] + "\n")
    return fasta, body


# ---------------------------------------------------------------------------
# Test 1 — Adapter delegates correctly + returns the documented shape
# ---------------------------------------------------------------------------


def test_off_target_scan_returns_correct_shape(tmp_path: Path) -> None:
    """The adapter returns ``(list[OffTargetSite], float, int)``.

    Exercises the public-API surface T8 ranker / T10 CLI consume.
    """
    from bionpu.genomics.pe_design.off_target import off_target_scan_for_spacer
    from bionpu.genomics.pe_design.types import OffTargetSite

    fasta, _seq = _build_synthetic_genome(tmp_path)
    sites, cfd_aggregate, count = off_target_scan_for_spacer(
        KNOWN_SPACER,
        fasta,
        max_mismatches=4,
    )

    assert isinstance(sites, list)
    assert isinstance(cfd_aggregate, float)
    assert isinstance(count, int)
    assert count == len(sites), (
        f"reported count={count} disagrees with len(sites)={len(sites)}"
    )
    # We planted one perfect-match on-target; allowing 4 mismatches over
    # ~6 kbp of random ACGT should surface at least the planted hit.
    assert count >= 1, "synthetic genome should contain at least the planted on-target"

    for s in sites:
        assert isinstance(s, OffTargetSite)
        assert s.chrom == SMOKE_CHROM
        assert isinstance(s.pos, int) and s.pos >= 0
        assert s.strand in ("+", "-")
        assert isinstance(s.mismatches, int) and 0 <= s.mismatches <= 4
        assert isinstance(s.cfd_score, float)
        assert 0.0 <= s.cfd_score <= 1.0, (
            f"per-site cfd_score {s.cfd_score} for {s} outside [0, 1]"
        )

    # The aggregate is CRISPOR-style specificity in [0, 100] (matching
    # crispr_design's _score_off_target_cfd convention).
    assert 0.0 <= cfd_aggregate <= 100.0


# ---------------------------------------------------------------------------
# Test 2 — re-shape preserves per-site CFD scores
# ---------------------------------------------------------------------------


def test_off_target_site_reshape_preserves_cfd_scores(tmp_path: Path) -> None:
    """The adapter's per-site ``cfd_score`` matches :class:`CFDScorer`.

    Re-runs the same scan via :func:`bionpu.scan.cpu_scan` +
    :class:`bionpu.scoring.cfd.CFDScorer` directly and asserts each
    adapter-emitted ``OffTargetSite.cfd_score`` is byte-equal to the
    underlying scorer's float score for the corresponding row. Proves
    the adapter delegates rather than re-implementing CFD math.
    """
    from bionpu.genomics.pe_design.off_target import off_target_scan_for_spacer
    from bionpu.scan import GuideSpec, cpu_scan
    from bionpu.scoring.cfd import CFDScorer

    fasta, seq = _build_synthetic_genome(tmp_path)

    # Adapter call.
    adapter_sites, _agg, _count = off_target_scan_for_spacer(
        KNOWN_SPACER,
        fasta,
        max_mismatches=4,
    )

    # Reference path — exactly what crispr_design._scan_locus_for_offtargets
    # + _score_off_target_cfd do internally.
    ref_rows = cpu_scan(
        chrom=SMOKE_CHROM,
        seq=seq,
        guides=[GuideSpec(spacer=KNOWN_SPACER, guide_id=KNOWN_SPACER)],
        pam_template="NGG",
        max_mismatches=4,
    )
    scorer = CFDScorer(matrix="doench_2016", apply_pam_penalty=False)
    ref_score_rows = list(scorer.score(ref_rows))

    # Build a position-keyed map of reference cfd scores so the adapter
    # can be insertion-order independent.
    ref_by_key = {
        (r.chrom, r.start, r.strand, r.mismatches): float(sr.score)
        for r, sr in zip(ref_rows, ref_score_rows, strict=True)
    }
    adapter_by_key = {
        (s.chrom, s.pos, s.strand, s.mismatches): s.cfd_score for s in adapter_sites
    }

    # Same set of keys (same site set).
    assert set(adapter_by_key.keys()) == set(ref_by_key.keys()), (
        f"adapter sites disagree with reference scan: "
        f"only-adapter={set(adapter_by_key) - set(ref_by_key)} "
        f"only-ref={set(ref_by_key) - set(adapter_by_key)}"
    )
    # Per-site CFD scores byte-equal (no float re-computation in the
    # adapter — pure pass-through).
    for key, ref_score in ref_by_key.items():
        assert adapter_by_key[key] == ref_score, (
            f"cfd_score mismatch at {key}: adapter={adapter_by_key[key]} "
            f"vs reference={ref_score}"
        )


# ---------------------------------------------------------------------------
# Test 3 — counts match the crispr_design scan path for the same spacer
# ---------------------------------------------------------------------------


def test_off_target_count_matches_crispr_design_scan_path(tmp_path: Path) -> None:
    """Adapter row count matches what crispr_design's scan path produces.

    This is the "delegation, not re-implementation" gate. The adapter
    ought to produce the exact set of off-target rows that
    ``crispr_design._scan_locus_for_offtargets`` -> ``cpu_scan``
    produces for the same spacer + same locus. We verify two things:

    1. The total off-target row count matches what
       :func:`bionpu.scan.cpu_scan` (the primitive crispr_design's
       path uses internally) returns for the same spacer + locus.
    2. The CFD aggregate value equals what
       :func:`bionpu.scoring.cfd.aggregate_cfd` returns when fed the
       same scored rows (delegation, not custom aggregation).
    """
    from bionpu.genomics.pe_design.off_target import off_target_scan_for_spacer
    from bionpu.scan import GuideSpec, cpu_scan
    from bionpu.scoring.cfd import CFDScorer, aggregate_cfd

    fasta, seq = _build_synthetic_genome(tmp_path)

    adapter_sites, adapter_cfd_agg, adapter_count = off_target_scan_for_spacer(
        KNOWN_SPACER,
        fasta,
        max_mismatches=4,
    )

    # Reference: same primitives crispr_design uses internally.
    ref_rows = cpu_scan(
        chrom=SMOKE_CHROM,
        seq=seq,
        guides=[GuideSpec(spacer=KNOWN_SPACER, guide_id=KNOWN_SPACER)],
        pam_template="NGG",
        max_mismatches=4,
    )
    scorer = CFDScorer(matrix="doench_2016", apply_pam_penalty=False)
    ref_score_rows = list(scorer.score(ref_rows))
    ref_aggregate = aggregate_cfd(ref_score_rows, exclude_on_target=True)

    # 1. Row count matches the underlying scan path byte-equal.
    assert adapter_count == len(ref_rows), (
        f"adapter count {adapter_count} disagrees with cpu_scan count "
        f"{len(ref_rows)} for spacer {KNOWN_SPACER!r}"
    )
    assert len(adapter_sites) == len(ref_rows)

    # 2. CFD aggregate matches the locked aggregate_cfd helper.
    expected_aggregate = ref_aggregate.get(KNOWN_SPACER, 100.0)
    assert adapter_cfd_agg == pytest.approx(expected_aggregate, abs=1e-9), (
        f"adapter cfd_aggregate {adapter_cfd_agg} disagrees with "
        f"aggregate_cfd output {expected_aggregate}"
    )


# ---------------------------------------------------------------------------
# Bonus — invalid spacer rejection (cheap input-validation guard)
# ---------------------------------------------------------------------------


def test_off_target_scan_rejects_non_20mer_spacer(tmp_path: Path) -> None:
    """Adapter rejects a non-20-nt spacer with a clear error.

    cpu_scan / GuideSpec validate this internally; the adapter forwards
    the failure with a descriptive message rather than silently
    returning an empty hit list.
    """
    from bionpu.genomics.pe_design.off_target import off_target_scan_for_spacer

    fasta, _seq = _build_synthetic_genome(tmp_path)

    with pytest.raises(ValueError):
        off_target_scan_for_spacer(
            "ACGT",  # 4 nt, not 20
            fasta,
            max_mismatches=4,
        )
