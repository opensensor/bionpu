# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

"""Unit tests for ``bionpu.genomics.library_design.per_gene_designer``.

Tests run against a *synthetic* multi-chromosome FASTA and patch the
single-target designer's gene resolver so the per-gene loop exercises
the full code path with no GRCh38 download.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from bionpu.genomics import crispr_design as cd
from bionpu.genomics.library_design.per_gene_designer import (
    PerGenePool,
    design_per_gene_pools,
    select_top_guides_for_gene,
)


SYN_LEN = 6000


def _build_synthetic_multichrom_fasta(tmp_path: Path) -> Path:
    """Build a 3-chromosome synthetic FASTA with NGG-bearing windows."""
    rng = random.Random(2026)
    fasta = tmp_path / "syn.fa"
    parts: list[str] = []
    chroms = ["syn_GENE_A", "syn_GENE_B", "syn_GENE_C"]
    planted = "AAGCTGAAACATTCATTAGGTGG"  # spacer + NGG
    for i, chrom in enumerate(chroms):
        seq = list(rng.choices("ACGT", k=SYN_LEN))
        # Stagger plant positions so different chroms get different sites.
        for j, c in enumerate(planted):
            seq[200 + i * 50 + j] = c
        body = "".join(seq)
        parts.append(f">{chrom}")
        for k in range(0, len(body), 80):
            parts.append(body[k : k + 80])
    fasta.write_text("\n".join(parts) + "\n", encoding="ascii")
    return fasta


def _patch_resolver(monkeypatch: pytest.MonkeyPatch, genes: list[str]) -> None:
    """Pin synthetic genes into the resolver table."""
    chroms = ["syn_GENE_A", "syn_GENE_B", "syn_GENE_C"]
    for i, gene in enumerate(genes):
        chrom = chroms[i % len(chroms)]
        monkeypatch.setitem(
            cd._RESOLVE_GENE_TO_LOCUS, gene, (chrom, 1, SYN_LEN)
        )

    real_resolve = cd.resolve_target

    def smoke_resolve(*, target, genome, fasta_path):
        if genome == "SYN":
            chrom, one_start, one_end = cd._RESOLVE_GENE_TO_LOCUS[target]
            zero_start = one_start - 1
            zero_end = one_end
            seq = cd.slice_chrom_from_fasta(
                fasta_path=Path(fasta_path),
                chrom=chrom,
                start=zero_start,
                end=zero_end,
            )
            return cd.ResolvedTarget(
                gene=target, chrom=chrom, start=zero_start, end=zero_end, sequence=seq,
            )
        return real_resolve(target=target, genome=genome, fasta_path=fasta_path)

    monkeypatch.setattr(cd, "resolve_target", smoke_resolve)


def test_design_per_gene_pools_returns_one_pool_per_gene(tmp_path, monkeypatch):
    fasta = _build_synthetic_multichrom_fasta(tmp_path)
    genes = ["SYNGENE1", "SYNGENE2", "SYNGENE3"]
    _patch_resolver(monkeypatch, genes)

    pools = design_per_gene_pools(
        targets=genes,
        genome="SYN",
        fasta_path=fasta,
        guides_per_gene=4,
        pool_oversample=4,
        device="cpu",
    )

    assert len(pools) == 3
    assert all(isinstance(p, PerGenePool) for p in pools)
    assert [p.gene for p in pools] == genes
    # top_n = 4*4 = 16; the synthetic locus produces hundreds of NGG
    # candidates, so each pool should saturate top_n (pool size = 16).
    for pool in pools:
        assert len(pool.ranked_full) == 16
        assert pool.guides_per_gene == 4
        assert pool.pool_oversample == 4


def test_design_per_gene_pools_rejects_zero_guides(tmp_path, monkeypatch):
    fasta = _build_synthetic_multichrom_fasta(tmp_path)
    _patch_resolver(monkeypatch, ["SYNGENE1"])
    with pytest.raises(ValueError, match="guides_per_gene must be >= 1"):
        design_per_gene_pools(
            targets=["SYNGENE1"],
            genome="SYN",
            fasta_path=fasta,
            guides_per_gene=0,
            device="cpu",
        )


def test_design_per_gene_pools_rejects_zero_oversample(tmp_path, monkeypatch):
    fasta = _build_synthetic_multichrom_fasta(tmp_path)
    _patch_resolver(monkeypatch, ["SYNGENE1"])
    with pytest.raises(ValueError, match="pool_oversample must be >= 1"):
        design_per_gene_pools(
            targets=["SYNGENE1"],
            genome="SYN",
            fasta_path=fasta,
            guides_per_gene=4,
            pool_oversample=0,
            device="cpu",
        )


def test_select_top_guides_skips_forbidden_spacers(tmp_path, monkeypatch):
    fasta = _build_synthetic_multichrom_fasta(tmp_path)
    _patch_resolver(monkeypatch, ["SYNGENE1"])
    pools = design_per_gene_pools(
        targets=["SYNGENE1"],
        genome="SYN",
        fasta_path=fasta,
        guides_per_gene=4,
        pool_oversample=4,
        device="cpu",
    )
    pool = pools[0]
    # Default: 4 chosen.
    chosen_default = select_top_guides_for_gene(pool)
    assert len(chosen_default) == 4
    # Forbid the top spacer; we should still get 4 (different ones).
    forbidden = {chosen_default[0].guide_seq}
    chosen_forbidden = select_top_guides_for_gene(
        pool, forbidden_spacers=forbidden
    )
    assert len(chosen_forbidden) == 4
    assert chosen_default[0].guide_seq not in {g.guide_seq for g in chosen_forbidden}


def test_select_top_guides_under_balances_when_pool_too_small(monkeypatch):
    """A small ranked pool returns < guides_per_gene chosen guides."""
    from bionpu.genomics.crispr_design import RankedGuide

    # Build a tiny synthetic pool of 2 guides and request 4: dedup
    # forbids 1 of them, leaving 1 -> under-balanced.
    rg1 = RankedGuide(
        rank=1, guide_id="g1", guide_seq="AAAACCCCGGGGTTTTACGT",
        pam_seq="TGG", strand="+", target_chrom="syn", target_pos=100,
        gc_pct=50.0, on_target_score=0.8, cfd_aggregate=99.0,
        off_target_count=0, top_off_targets="",
        composite_crispor=0.9, composite_bionpu=0.9,
        ranked_by="crispor", predicted_indel="", notes="",
    )
    rg2 = RankedGuide(
        rank=2, guide_id="g2", guide_seq="GGGGTTTTAAAACCCCGTAC",
        pam_seq="AGG", strand="+", target_chrom="syn", target_pos=200,
        gc_pct=50.0, on_target_score=0.6, cfd_aggregate=80.0,
        off_target_count=0, top_off_targets="",
        composite_crispor=0.7, composite_bionpu=0.7,
        ranked_by="crispor", predicted_indel="", notes="",
    )

    # Build a synthetic PerGenePool by hand.
    class FakeRunResult:
        stage_timings_s: dict[str, float] = {"x": 0.0}

    pool = PerGenePool(
        gene="SYN",
        guides_per_gene=4,
        pool_oversample=2,
        ranked_full=(rg1, rg2),
        run_result=FakeRunResult(),
    )

    forbidden = {rg1.guide_seq}
    chosen = select_top_guides_for_gene(pool, forbidden_spacers=forbidden)
    # Only rg2 survives; gene is under-balanced (2 < 4).
    assert len(chosen) == 1
    assert chosen[0].guide_seq == rg2.guide_seq
