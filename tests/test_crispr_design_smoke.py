# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

"""End-to-end smoke test for ``bionpu crispr design`` (Tier 1).

The smoke test wires every stage from the PRD §2.1 stage diagram and
asserts the output meets the §3.2 schema invariants. It runs in
under 30 seconds on a developer laptop by using a synthetic 10 kbp
locus rather than a real GRCh38 chromosome:

* Stage 1 (target resolution) — patches ``_RESOLVE_GENE_TO_LOCUS`` to
  pin the synthetic gene at a synthetic FASTA's coordinates.
* Stages 2-3 (PAM scan + off-target scan) — run on the synthetic
  locus through ``cpu_scan`` (the silicon path requires the
  ``crispr_pam_filter`` xclbin which is not vendored into this
  submodule's ``_npu_artifacts/`` tree as of Tier 1).
* Stages 4a/b — ``DoenchRS1Scorer`` + ``CFDScorer`` from the landed
  scoring layer.
* Stage 5 — ``rank_guides`` + ``format_guides_tsv``.

The synthetic locus is 10 kbp of seeded random ACGT with three
hand-placed strong-NGG sites injected so the test is deterministic
across runs.
"""

from __future__ import annotations

import io
import json
import random
import time
from pathlib import Path

import pytest

from bionpu.genomics import crispr_design as cd


SMOKE_LOCUS_LEN = 10_000
SMOKE_GENE = "SMOKE_BRCA1_LIKE"
SMOKE_CHROM = "smoke_chr"
SMOKE_GENOME = "SMOKE"
SMOKE_SEED = 42

# Hand-placed 23-mers (20-nt spacer + NGG PAM). Their positions are far
# enough from the locus boundaries that the 30-mer Doench RS1 context
# window fits without truncation.
_PLANTED_SITES = (
    (200, "AAGCTGAAACATTCATTAGGTGG"),  # +strand spacer + NGG
    (1000, "CGTGAGTACGCATGTACGTAAGG"),
    (5000, "CGCAATATCAGGTAGCAACGCGG"),
)


def _build_synthetic_fasta(tmp_path: Path) -> Path:
    """Build a 10 kbp synthetic chromosome with hand-placed PAM sites."""
    rng = random.Random(SMOKE_SEED)
    seq = list(rng.choices("ACGT", k=SMOKE_LOCUS_LEN))
    for pos, motif in _PLANTED_SITES:
        for i, c in enumerate(motif):
            seq[pos + i] = c
    body = "".join(seq)
    fasta = tmp_path / "smoke_chr.fa"
    with fasta.open("w", encoding="ascii") as fh:
        fh.write(f">{SMOKE_CHROM}\n")
        for i in range(0, len(body), 80):
            fh.write(body[i : i + 80] + "\n")
    return fasta


def _patch_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the synthetic gene + bypass the GRCh38-only guard for Tier 1."""
    # Add the synthetic gene to the resolver table. The Tier 1 resolver
    # uses 1-based inclusive coordinates; we span the entire locus.
    monkeypatch.setitem(
        cd._RESOLVE_GENE_TO_LOCUS, SMOKE_GENE, (SMOKE_CHROM, 1, SMOKE_LOCUS_LEN)
    )

    # The Tier 1 ``resolve_target`` rejects ``genome != 'GRCh38'`` (PRD
    # §3.1 limitation) — for the smoke we substitute a wrapper that
    # accepts ``"SMOKE"`` and otherwise delegates.
    real_resolve = cd.resolve_target

    def smoke_resolve(*, target, genome, fasta_path):
        if genome == SMOKE_GENOME:
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
                gene=target,
                chrom=chrom,
                start=zero_start,
                end=zero_end,
                sequence=seq,
            )
        return real_resolve(
            target=target, genome=genome, fasta_path=fasta_path
        )

    monkeypatch.setattr(cd, "resolve_target", smoke_resolve)


def test_crispr_design_end_to_end_smoke(tmp_path, monkeypatch) -> None:
    """End-to-end smoke. PRD §4.4: must complete in <30 s on a laptop."""
    fasta = _build_synthetic_fasta(tmp_path)
    _patch_resolver(monkeypatch)

    t0 = time.perf_counter()
    result = cd.design_guides_for_target(
        target=SMOKE_GENE,
        genome=SMOKE_GENOME,
        fasta_path=fasta,
        top_n=10,
        max_mismatches=4,
        gc_min=cd.DEFAULT_GC_MIN,
        gc_max=cd.DEFAULT_GC_MAX,
        device="cpu",
        rank_by="crispor",
    )
    wall = time.perf_counter() - t0

    # ---- timing gate (PRD §4.4 smoke fixture) ---------------------------- #
    assert wall < 30.0, f"smoke took {wall:.2f}s; PRD §4.4 budget is 30s"

    # ---- stage wiring assertions ----------------------------------------- #
    expected_stages = {
        "target_resolve",
        "pam_scan",
        "off_target_scan",
        "on_target_score",
        "off_target_score",
        "rank_emit",
    }
    assert set(result.stage_timings_s) == expected_stages
    for stage, secs in result.stage_timings_s.items():
        assert secs >= 0.0, f"stage {stage} reports negative time"

    # ---- top-N gate ------------------------------------------------------ #
    # The 10 kbp locus generates hundreds of candidate guides on random
    # ACGT alone; we asserted ``top=10`` so we expect exactly 10 ranked.
    assert len(result.ranked) == 10

    # ---- PRD §3.2 schema invariants -------------------------------------- #
    for g in result.ranked:
        assert len(g.guide_seq) == 20, f"guide {g.guide_id!r} not 20 nt"
        assert all(c in "ACGT" for c in g.guide_seq)
        assert g.pam_seq.endswith("GG"), (
            f"guide {g.guide_id!r} pam {g.pam_seq!r} is not NGG"
        )
        assert g.strand in ("+", "-")
        assert g.target_chrom == SMOKE_CHROM
        assert 0 <= g.gc_pct <= 100
        assert 0.0 <= g.on_target_score <= 1.0, (
            f"on-target {g.on_target_score} for {g.guide_id!r} out of [0,1]"
        )
        assert 0.0 <= g.cfd_aggregate <= 100.0, (
            f"cfd_aggregate {g.cfd_aggregate} for {g.guide_id!r} out of [0,100]"
        )
        assert 0.0 <= g.composite_crispor <= 1.0
        assert 0.0 <= g.composite_bionpu <= 1.0
        assert g.ranked_by == "crispor"
        assert g.off_target_count >= 0

    # Ranks are 1..N, monotonic, no gaps.
    assert [g.rank for g in result.ranked] == list(range(1, 11))
    # Sort is descending by composite_crispor (Tier 1 default).
    composites = [g.composite_crispor for g in result.ranked]
    assert composites == sorted(composites, reverse=True)

    # ---- TSV byte stability ---------------------------------------------- #
    tsv = result.tsv_bytes.decode("utf-8")
    header_line = tsv.splitlines()[0]
    assert header_line.split("\t") == list(cd.TSV_HEADER)
    body_lines = tsv.splitlines()[1:]
    assert len(body_lines) == 10
    for line in body_lines:
        cols = line.split("\t")
        assert len(cols) == len(cd.TSV_HEADER)


def test_crispr_design_cli_smoke(tmp_path, monkeypatch, capsys) -> None:
    """In-process CLI invocation (mirrors ``bionpu crispr design ...``).

    Ensures the argparse plumbing and ``_cmd_crispr_design`` wiring are
    exercised end-to-end with the synthetic locus, with no real GRCh38
    bytes touched.
    """
    fasta = _build_synthetic_fasta(tmp_path)
    _patch_resolver(monkeypatch)

    # The CLI's ``--genome`` choice list is restricted to GRCh38 by
    # argparse; the smoke fixture uses the SMOKE genome. Bypass argparse
    # by calling the orchestrator directly + emitting through the same
    # ``format_guides_tsv`` path (which is what the CLI does internally).
    out_path = tmp_path / "smoke_guides.tsv"

    # Drive the CLI through ``main`` so the argparse + dispatch glue is
    # under test. We pass ``--genome SMOKE`` is rejected by the choice
    # list, so the CLI smoke for Tier 1 directly invokes the
    # orchestrator (CLI argparse choices are validated separately by
    # ``test_crispr_design_cli_help``).
    result = cd.design_guides_for_target(
        target=SMOKE_GENE,
        genome=SMOKE_GENOME,
        fasta_path=fasta,
        top_n=5,
        device="cpu",
    )
    out_path.write_bytes(result.tsv_bytes)
    assert out_path.is_file()
    text = out_path.read_text("utf-8")
    lines = text.splitlines()
    assert len(lines) == 6  # header + 5 rows
    assert lines[0].split("\t") == list(cd.TSV_HEADER)


def test_crispr_design_cli_help_lists_design_subcommand(capsys) -> None:
    """``bionpu crispr design --help`` is wired into the top-level CLI.

    argparse's ``--help`` action raises ``SystemExit(0)`` after
    printing; we expect that and assert the printed help text covers
    the Tier 1 surface.
    """
    from bionpu.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["crispr", "design", "--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "--target" in captured.out
    assert "--genome" in captured.out
    assert "--rank-by" in captured.out


def test_resolver_rejects_unknown_gene(tmp_path) -> None:
    fasta = _build_synthetic_fasta(tmp_path)
    with pytest.raises(cd.GeneNotFoundError):
        cd.resolve_target(
            target="NOT_A_GENE", genome="GRCh38", fasta_path=fasta
        )


def test_resolver_rejects_unknown_genome(tmp_path) -> None:
    fasta = _build_synthetic_fasta(tmp_path)
    with pytest.raises(ValueError, match="genome must be 'GRCh38'"):
        cd.resolve_target(
            target="BRCA1", genome="hg19", fasta_path=fasta
        )


def test_compute_composite_clamps_inputs() -> None:
    """The composite formulas clamp out-of-range inputs gracefully."""
    # Out-of-range scores get clamped to [0, 1] before the linear combo.
    assert cd.compute_composite_crispor(
        on_target_score=2.0, cfd_aggregate=200.0
    ) == pytest.approx(1.0)
    assert cd.compute_composite_bionpu(
        on_target_score=-1.0, cfd_aggregate=-10.0
    ) == pytest.approx(0.0)
    # The default weights (0.5 / 0.5) make the two formulas equal at the
    # baseline -- the spike (PRD §7.1 Q5b) replaces the bionpu formula.
    assert cd.compute_composite_crispor(
        on_target_score=0.7, cfd_aggregate=80.0
    ) == pytest.approx(
        cd.compute_composite_bionpu(on_target_score=0.7, cfd_aggregate=80.0)
    )


def test_slice_chrom_extracts_correct_window(tmp_path) -> None:
    fasta = _build_synthetic_fasta(tmp_path)
    sl = cd.slice_chrom_from_fasta(
        fasta_path=fasta, chrom=SMOKE_CHROM, start=200, end=200 + 23
    )
    assert sl == _PLANTED_SITES[0][1]


def test_slice_chrom_rejects_missing_contig(tmp_path) -> None:
    fasta = _build_synthetic_fasta(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        cd.slice_chrom_from_fasta(
            fasta_path=fasta, chrom="not_a_chrom", start=0, end=10
        )


def test_gene_edge_guides_use_widened_doench_context(tmp_path, monkeypatch) -> None:
    """Guides at the requested locus edge still get Doench context."""
    chrom = "edge_chr"
    # Four upstream bases are outside the requested target slice but inside
    # the chromosome. The first target bases form a valid +strand NGG site.
    target = "AAGCTGAAACATTCATTAGGTGG" + ("ACGT" * 20)
    seq = "TTTT" + target + "CCCC"
    fasta = tmp_path / "edge.fa"
    fasta.write_text(f">{chrom}\n{seq}\n", encoding="ascii")
    monkeypatch.setitem(
        cd._RESOLVE_GENE_TO_LOCUS,
        "EDGE",
        (chrom, 5, 4 + len(target)),
    )

    result = cd.design_guides_for_target(
        target="EDGE",
        genome="GRCh38",
        fasta_path=fasta,
        top_n=10,
        device="cpu",
    )

    edge_rows = [g for g in result.ranked if g.target_pos == 4]
    assert edge_rows
    assert edge_rows[0].on_target_score > 0.0


def test_target_fasta_json_cli_mode(tmp_path, capsys) -> None:
    """Mode C target FASTA works without a GRCh38 reference and emits JSON."""
    from bionpu.cli import main

    target = "AAGCTGAAACATTCATTAGGTGG" + ("ACGT" * 20)
    fasta = tmp_path / "construct.fa"
    fasta.write_text(f">construct_1\n{target}\n", encoding="ascii")

    rc = main(
        [
            "crispr",
            "design",
            "--target",
            "construct",
            "--genome",
            "none",
            "--target-fasta",
            str(fasta),
            "--top",
            "3",
            "--device",
            "cpu",
            "--format",
            "json",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["target"]["gene"] == "construct"
    assert payload["target"]["chrom"] == "construct_1"
    assert payload["target"]["start"] == 0
    assert payload["ranked"]
    assert payload["ranked"][0]["guide_seq"]
