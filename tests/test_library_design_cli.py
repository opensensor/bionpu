# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

"""End-to-end CLI smoke test for ``bionpu library design`` (Track C v0).

Runs the full multi-target pipeline against a synthetic
multi-chromosome FASTA + patches the gene resolver. Asserts the
output TSV row counts, gene balance, controls present, and TSV
schema invariants.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from bionpu.genomics import crispr_design as cd
from bionpu.genomics.library_design import (
    CANONICAL_ESSENTIAL_GENE_GUIDES,
    CANONICAL_SAFE_HARBOR_GUIDES,
    LIBRARY_TSV_HEADER,
    LibraryGuide,
    design_pooled_library,
    format_library_tsv,
)


SYN_LEN = 6000
SYN_GENES = ["SYNGENE_1", "SYNGENE_2", "SYNGENE_3", "SYNGENE_4", "SYNGENE_5"]


def _build_synthetic_multichrom_fasta(tmp_path: Path) -> Path:
    fasta = tmp_path / "syn.fa"
    rng = random.Random(2026)
    parts: list[str] = []
    planted = "AAGCTGAAACATTCATTAGGTGG"
    for i, gene in enumerate(SYN_GENES):
        chrom = f"syn_{gene}"
        seq = list(rng.choices("ACGT", k=SYN_LEN))
        for j, c in enumerate(planted):
            seq[200 + i * 47 + j] = c
        body = "".join(seq)
        parts.append(f">{chrom}")
        for k in range(0, len(body), 80):
            parts.append(body[k : k + 80])
    fasta.write_text("\n".join(parts) + "\n", encoding="ascii")
    return fasta


def _patch_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin synthetic genes into the resolver and let the design pipeline run."""
    for gene in SYN_GENES:
        monkeypatch.setitem(
            cd._RESOLVE_GENE_TO_LOCUS, gene, (f"syn_{gene}", 1, SYN_LEN)
        )

    real_resolve = cd.resolve_target

    def smoke_resolve(*, target, genome, fasta_path):
        # Accept either the synthetic SYN genome or GRCh38 with our
        # patched table (the CLI restricts --genome to GRCh38).
        if target in cd._RESOLVE_GENE_TO_LOCUS and genome in ("SYN", "GRCh38"):
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
        return real_resolve(target=target, genome=genome, fasta_path=fasta_path)

    monkeypatch.setattr(cd, "resolve_target", smoke_resolve)


def test_design_pooled_library_end_to_end_5_genes(tmp_path, monkeypatch):
    """Brief acceptance criteria: 5 genes × 4 guides + 100 controls TSV."""
    fasta = _build_synthetic_multichrom_fasta(tmp_path)
    _patch_resolver(monkeypatch)

    rows = design_pooled_library(
        targets=SYN_GENES,
        library_type="knockout",
        guides_per_gene=4,
        genome="GRCh38",
        fasta_path=fasta,
        n_controls=100,
        device="cpu",
    )

    # ---- Brief acceptance #2: 5*4 + (100 + 3 + 2) = 125 rows -------------- #
    n_per_gene = sum(1 for r in rows if r.control_class == "")
    n_non_targeting = sum(1 for r in rows if r.control_class == "non_targeting")
    n_safe_harbor = sum(1 for r in rows if r.control_class == "safe_harbor")
    n_essential = sum(1 for r in rows if r.control_class == "essential_gene")
    assert n_per_gene == 5 * 4
    assert n_non_targeting == 100
    assert n_safe_harbor == len(CANONICAL_SAFE_HARBOR_GUIDES)
    assert n_essential == len(CANONICAL_ESSENTIAL_GENE_GUIDES)
    assert len(rows) == n_per_gene + n_non_targeting + n_safe_harbor + n_essential

    # ---- Brief acceptance #3: each gene has exactly 4; no inter-gene dup -- #
    by_gene: dict[str, list[LibraryGuide]] = {}
    for r in rows:
        if r.control_class == "":
            by_gene.setdefault(r.target_symbol, []).append(r)
    for gene, gene_rows in by_gene.items():
        assert len(gene_rows) == 4, (
            f"gene {gene!r} has {len(gene_rows)} guides, expected 4"
        )
    # Cross-gene spacer dedup: every per-gene spacer is unique across the library.
    per_gene_spacers = [r.guide_seq for r in rows if r.control_class == ""]
    assert len(set(per_gene_spacers)) == len(per_gene_spacers)

    # ---- Brief acceptance #4: control set fully present ------------------- #
    sh_labels = {r.target_symbol for r in rows if r.control_class == "safe_harbor"}
    assert any("AAVS1" in lbl for lbl in sh_labels)
    assert any("CCR5" in lbl for lbl in sh_labels)
    assert any("ROSA26" in lbl for lbl in sh_labels)
    ess_labels = {r.target_symbol for r in rows if r.control_class == "essential_gene"}
    assert any("RPS19" in lbl for lbl in ess_labels)
    assert any("RPL15" in lbl for lbl in ess_labels)

    # ---- Brief acceptance #5: top-1 per gene has reasonable composite ----- #
    # The single-target design pipeline returns guides sorted by composite.
    for gene_rows in by_gene.values():
        # First row (rank 1 in-gene) should have a valid Doench score.
        top = gene_rows[0]
        assert 0.0 <= top.doench_rs1 <= 1.0
        # cfd_aggregate must be a non-NaN finite float for per-gene rows.
        import math
        assert math.isfinite(top.cfd_aggregate)


def test_format_library_tsv_schema_validates(tmp_path, monkeypatch):
    """The emitted TSV has the brief-required header and column count."""
    fasta = _build_synthetic_multichrom_fasta(tmp_path)
    _patch_resolver(monkeypatch)

    rows = design_pooled_library(
        targets=SYN_GENES[:3],
        library_type="knockout",
        guides_per_gene=4,
        genome="GRCh38",
        fasta_path=fasta,
        n_controls=10,
        device="cpu",
    )
    tsv = format_library_tsv(rows).decode("utf-8")
    lines = tsv.splitlines()
    header = lines[0].split("\t")
    assert tuple(header) == LIBRARY_TSV_HEADER
    # Every body row has the same column count as header.
    for ln in lines[1:]:
        cols = ln.split("\t")
        assert len(cols) == len(LIBRARY_TSV_HEADER), (
            f"row has {len(cols)} cols, expected {len(LIBRARY_TSV_HEADER)}: "
            f"{ln!r}"
        )

    # Body has 3*4 + 10 + 3 + 2 = 27 rows.
    assert len(lines) - 1 == 3 * 4 + 10 + 3 + 2


def test_cli_wires_through_main(tmp_path, monkeypatch, capsys):
    """``python -m bionpu.cli library design ...`` runs end-to-end."""
    from bionpu.cli import main

    fasta = _build_synthetic_multichrom_fasta(tmp_path)
    _patch_resolver(monkeypatch)

    targets_file = tmp_path / "targets.txt"
    targets_file.write_text("\n".join(SYN_GENES[:3]) + "\n")

    out_path = tmp_path / "lib.tsv"
    measurements_path = tmp_path / "measurements.json"

    rc = main(
        [
            "library",
            "design",
            "--targets-file",
            str(targets_file),
            "--guides-per-gene",
            "4",
            "--genome",
            "GRCh38",
            "--fasta",
            str(fasta),
            "--controls",
            "10",
            "--output",
            str(out_path),
            "--measurements-out",
            str(measurements_path),
        ]
    )
    assert rc == 0
    assert out_path.is_file()
    assert measurements_path.is_file()

    text = out_path.read_text("utf-8")
    lines = text.splitlines()
    assert lines[0].split("\t") == list(LIBRARY_TSV_HEADER)
    body = lines[1:]
    # 3 genes * 4 guides + 10 NT + 3 SH + 2 ESS = 27.
    assert len(body) == 3 * 4 + 10 + 3 + 2

    metrics = json.loads(measurements_path.read_text())
    assert metrics["n_targets"] == 3
    assert metrics["library_type"] == "knockout"
    assert metrics["metrics"]["n_non_targeting"] == 10
    assert metrics["metrics"]["n_safe_harbor"] == 3
    assert metrics["metrics"]["n_essential"] == 2
    assert metrics["metrics"]["n_per_gene_rows"] == 12
    assert metrics["metrics"]["total_wall_s"] >= 0.0


def test_cli_rejects_activation_library_type(tmp_path, monkeypatch):
    """activation is a v1 deferral; CLI should exit 2 with an explanation."""
    from bionpu.cli import main

    fasta = _build_synthetic_multichrom_fasta(tmp_path)
    _patch_resolver(monkeypatch)

    rc = main(
        [
            "library",
            "design",
            "--targets",
            "SYNGENE_1",
            "--library-type",
            "activation",
            "--guides-per-gene",
            "4",
            "--genome",
            "GRCh38",
            "--fasta",
            str(fasta),
            "--controls",
            "0",
            "--output",
            "-",
        ]
    )
    assert rc == 2


def test_cli_rejects_missing_targets():
    """CLI should fail cleanly when neither --targets nor --targets-file is supplied."""
    from bionpu.cli import main

    rc = main(
        [
            "library",
            "design",
            "--guides-per-gene",
            "4",
            "--genome",
            "GRCh38",
            "--controls",
            "0",
            "--output",
            "-",
        ]
    )
    assert rc == 2


def test_cli_help_lists_design_subcommand(capsys):
    """``bionpu library design --help`` exits 0 with the expected flags."""
    from bionpu.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["library", "design", "--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    out = captured.out
    assert "--targets-file" in out
    assert "--library-type" in out
    assert "--guides-per-gene" in out
    assert "--controls" in out
