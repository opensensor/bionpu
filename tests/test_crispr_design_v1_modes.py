# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>

"""v1 acceptance tests for ``bionpu crispr design`` — three modes.

Per PRD-guide-design-on-xdna v0.2 §1.1 + the v1 brief, the CLI must
support three modes end-to-end:

* **Mode A** — gene symbol (``--target BRCA1 --genome GRCh38``).
* **Mode B** — coordinate range (``--target chr17:43044295-43125483``)
  or explicit ``--target-fasta`` against a real (sub-)reference.
* **Mode C** — synbio (``--target-fasta plasmid.fa --genome none``);
  off-target scan is skipped, ``cfd_aggregate`` renders as ``"NaN"``,
  ``off_target_count`` renders as ``"NaN"``, and ``notes`` contains
  ``NO_OFF_TARGET_SCAN``.

These tests do NOT require a real GRCh38 FASTA. Mode A is exercised by
patching the resolver to a synthetic locus; Mode B is exercised against
an explicit synthetic FASTA; Mode C is exercised with ``--genome none``
and a small construct FASTA.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from bionpu.cli import main as cli_main
from bionpu.genomics import crispr_design as cd


# --------------------------------------------------------------------- #
# Synthetic fixtures.
# --------------------------------------------------------------------- #


def _planted_locus(seed: int = 42, length: int = 8000) -> str:
    """Return an ACGT string with hand-placed strong-NGG sites.

    Planting positions are filtered to those that fit inside ``length``;
    short loci just receive fewer planted sites.
    """
    rng = random.Random(seed)
    bases = list(rng.choices("ACGT", k=length))
    planted = (
        (200, "AAGCTGAAACATTCATTAGGTGG"),
        (1000, "CGTGAGTACGCATGTACGTAAGG"),
        (3500, "CGCAATATCAGGTAGCAACGCGG"),
        (5500, "GGCATTGTCAGCATGCATGCTGG"),
    )
    for pos, motif in planted:
        if pos + len(motif) > length:
            continue
        for i, ch in enumerate(motif):
            bases[pos + i] = ch
    return "".join(bases)


def _write_fasta(path: Path, chrom: str, seq: str) -> None:
    with path.open("w", encoding="ascii") as fh:
        fh.write(f">{chrom}\n")
        for i in range(0, len(seq), 80):
            fh.write(seq[i : i + 80] + "\n")


# --------------------------------------------------------------------- #
# Mode A — gene symbol.
# --------------------------------------------------------------------- #


def test_mode_a_gene_symbol_resolves_and_designs(tmp_path, monkeypatch):
    """Mode A: ``--target BRCA1 --genome GRCh38`` runs end-to-end.

    We monkeypatch the resolver table so the synthetic locus is bound
    to the BRCA1 entry; this exercises the same gene-symbol code path
    as a real GRCh38 invocation without requiring the 3 GB reference.
    """
    seq = _planted_locus()
    fasta = tmp_path / "synthetic_chr17.fa"
    _write_fasta(fasta, "chr17", seq)

    # Pin the smoke locus over the BRCA1 entry for the duration of the test.
    monkeypatch.setitem(
        cd._RESOLVE_GENE_TO_LOCUS, "BRCA1", ("chr17", 1, len(seq))
    )

    result = cd.design_guides_for_target(
        target="BRCA1",
        genome="GRCh38",
        fasta_path=fasta,
        top_n=10,
        device="cpu",
    )

    assert len(result.ranked) >= 5  # at minimum the planted sites
    for g in result.ranked:
        assert len(g.guide_seq) == 20
        assert g.pam_seq.endswith("GG")
        assert g.target_chrom == "chr17"
        assert g.notes != "NO_OFF_TARGET_SCAN"  # Mode A scans
        # Mode A: cfd_aggregate is a real number (not NaN).
        assert g.cfd_aggregate == g.cfd_aggregate  # NaN-safe self-compare
        assert g.off_target_count >= 0


# --------------------------------------------------------------------- #
# Mode B — explicit target FASTA against a (sub-)reference.
# --------------------------------------------------------------------- #


def test_mode_b_target_fasta_with_genome_runs(tmp_path):
    """Mode B: ``--target-fasta brca1_exon11.fa`` with a sub-reference.

    Uses the synthetic locus directly via Mode C-style ``--target-fasta``
    (no GRCh38 lookup); the off-target scan still runs against the locus
    itself (Tier 1 locus-scope behaviour).
    """
    seq = _planted_locus(seed=43, length=6000)
    fasta = tmp_path / "brca1_exon.fa"
    _write_fasta(fasta, "synthetic_brca1", seq)

    result = cd.design_guides_for_target(
        target="brca1_exon",
        genome="GRCh38",  # not consulted; target_fasta_path overrides
        fasta_path=fasta,
        target_fasta_path=fasta,
        top_n=5,
        device="cpu",
    )

    assert len(result.ranked) >= 1
    for g in result.ranked:
        # Mode B with target_fasta_path → off-target scan still runs.
        assert g.notes != "NO_OFF_TARGET_SCAN"
        assert g.off_target_count >= 0
        assert g.cfd_aggregate == g.cfd_aggregate  # not NaN


# --------------------------------------------------------------------- #
# Mode C — synbio (--genome none).
# --------------------------------------------------------------------- #


def test_mode_c_synbio_emits_nan_and_no_off_target_scan_flag(tmp_path):
    """Mode C: ``--genome none`` skips off-target scan; emits NaN sentinels.

    PRD §3.2 + §7.1 Q3:
    * ``cfd_aggregate`` is NaN
    * ``off_target_count`` is the -1 sentinel (renders as ``"NaN"`` in
      TSV; ``null`` in JSON)
    * ``notes`` contains ``NO_OFF_TARGET_SCAN``
    """
    target = (
        "AAGCTGAAACATTCATTAGGTGG"
        + ("ACGT" * 25)
        + "CGTGAGTACGCATGTACGTAAGG"
        + ("ACGT" * 15)
    )
    fasta = tmp_path / "plasmid.fa"
    _write_fasta(fasta, "plasmid_1", target)

    result = cd.design_guides_for_target(
        target="plasmid",
        genome="none",
        fasta_path=fasta,
        target_fasta_path=fasta,
        top_n=5,
        device="cpu",
    )

    assert len(result.ranked) >= 1
    for g in result.ranked:
        # NaN-detection: NaN compares unequal to itself.
        assert g.cfd_aggregate != g.cfd_aggregate, (
            f"Mode C should emit NaN cfd_aggregate; got {g.cfd_aggregate!r}"
        )
        assert g.off_target_count == -1, (
            f"Mode C off_target_count sentinel should be -1; got "
            f"{g.off_target_count}"
        )
        assert "NO_OFF_TARGET_SCAN" in g.notes


def test_mode_c_synbio_tsv_renders_nan_strings(tmp_path):
    """Mode C TSV emits literal ``"NaN"`` strings for the unscored columns."""
    target = "AAGCTGAAACATTCATTAGGTGG" + ("ACGT" * 30)
    fasta = tmp_path / "plasmid.fa"
    _write_fasta(fasta, "plasmid_1", target)

    result = cd.design_guides_for_target(
        target="plasmid",
        genome="none",
        fasta_path=fasta,
        target_fasta_path=fasta,
        top_n=5,
        device="cpu",
    )
    tsv = result.tsv_bytes.decode("utf-8")
    body_lines = [line for line in tsv.splitlines()[1:] if line]
    assert body_lines, "Mode C produced no guides"
    for line in body_lines:
        cols = line.split("\t")
        # cfd_aggregate at index 9, off_target_count at 10.
        assert cols[9] == "NaN", f"row cfd_aggregate column not NaN: {cols!r}"
        assert cols[10] == "NaN", f"row off_target_count column not NaN: {cols!r}"
        assert "NO_OFF_TARGET_SCAN" in cols[-1], (
            f"row notes missing NO_OFF_TARGET_SCAN flag: {cols!r}"
        )


def test_mode_c_synbio_json_emits_null_for_nan(tmp_path):
    """Mode C JSON output renders NaN as ``null`` (strict-JSON-friendly)."""
    target = "AAGCTGAAACATTCATTAGGTGG" + ("ACGT" * 30)
    fasta = tmp_path / "construct.fa"
    _write_fasta(fasta, "construct_1", target)

    result = cd.design_guides_for_target(
        target="construct",
        genome="none",
        fasta_path=fasta,
        target_fasta_path=fasta,
        top_n=3,
        device="cpu",
    )
    payload_bytes = cd.format_result_json(result)
    payload = json.loads(payload_bytes.decode("utf-8"))
    assert payload["ranked"], "JSON ranked list empty"
    for g in payload["ranked"]:
        assert g["cfd_aggregate"] is None, (
            f"Mode C JSON should null-out cfd_aggregate; got {g['cfd_aggregate']!r}"
        )
        assert g["off_target_count"] is None, (
            f"Mode C JSON should null-out off_target_count; got "
            f"{g['off_target_count']!r}"
        )
        assert "NO_OFF_TARGET_SCAN" in g["notes"]


# --------------------------------------------------------------------- #
# CLI-level (argparse) end-to-end smoke for all three modes.
# --------------------------------------------------------------------- #


def test_cli_mode_b_target_fasta_tsv_round_trip(tmp_path):
    """``bionpu crispr design --target-fasta ... --genome GRCh38``."""
    seq = _planted_locus(seed=44, length=4000)
    fasta = tmp_path / "in.fa"
    _write_fasta(fasta, "synthetic", seq)
    out = tmp_path / "guides.tsv"

    rc = cli_main(
        [
            "crispr",
            "design",
            "--target",
            "synthetic",
            "--genome",
            "GRCh38",
            "--target-fasta",
            str(fasta),
            "--top",
            "5",
            "--device",
            "cpu",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    text = out.read_text("utf-8")
    lines = text.splitlines()
    # Header + at least 1 row.
    assert len(lines) >= 2
    assert lines[0].split("\t") == list(cd.TSV_HEADER)


def test_cli_mode_c_synbio_round_trip(tmp_path):
    """``bionpu crispr design --target-fasta plasmid.fa --genome none``."""
    target = (
        "AAGCTGAAACATTCATTAGGTGG"
        + ("ACGT" * 30)
        + "CGTGAGTACGCATGTACGTAAGG"
    )
    fasta = tmp_path / "plasmid.fa"
    _write_fasta(fasta, "plasmid_1", target)
    out = tmp_path / "guides.tsv"

    rc = cli_main(
        [
            "crispr",
            "design",
            "--target",
            "plasmid",
            "--genome",
            "none",
            "--target-fasta",
            str(fasta),
            "--top",
            "3",
            "--device",
            "cpu",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    text = out.read_text("utf-8")
    body = [line for line in text.splitlines()[1:] if line]
    assert body, "Mode C CLI produced no rows"
    for line in body:
        assert "NO_OFF_TARGET_SCAN" in line
        cols = line.split("\t")
        assert cols[9] == "NaN"  # cfd_aggregate
        assert cols[10] == "NaN"  # off_target_count


def test_cli_mode_a_gene_symbol_unknown_gene_errors(tmp_path):
    """Unknown gene symbols return non-zero exit with a clear message."""
    rc = cli_main(
        [
            "crispr",
            "design",
            "--target",
            "NOT_A_REAL_GENE_XYZ",
            "--genome",
            "GRCh38",
            "--fasta",
            str(tmp_path / "no_such.fa"),
            "--top",
            "3",
            "--device",
            "cpu",
        ]
    )
    assert rc == 2  # GeneNotFoundError / ValueError both exit 2


def test_cli_mode_c_default_genome_path_not_required(tmp_path):
    """Mode C does NOT require the ``--fasta`` reference flag."""
    target = "AAGCTGAAACATTCATTAGGTGG" + ("ACGT" * 30)
    fasta = tmp_path / "construct.fa"
    _write_fasta(fasta, "construct", target)

    # No --fasta supplied; relies on synbio path skipping ref lookup.
    rc = cli_main(
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


# --------------------------------------------------------------------- #
# Locked-asset reuse — surface checks (don't run silicon, just import).
# --------------------------------------------------------------------- #


def test_locked_silicon_kernels_importable():
    """v1 reuses locked silicon ops; the imports must remain stable.

    Per the v1 brief: pam_filter + match_multitile_memtile are
    reference-frozen. We don't dispatch silicon here; we only confirm
    the op-class import path stays intact so any regression to the
    locked artifact would surface in CI.
    """
    from bionpu.kernels.crispr import pam_filter  # noqa: F401
    from bionpu.kernels.crispr import match_multitile_memtile  # noqa: F401


def test_locked_scoring_modules_importable():
    """v1 reuses locked scoring; imports must remain stable."""
    from bionpu.scoring.cfd import CFDScorer, aggregate_cfd  # noqa: F401
    from bionpu.scoring.doench_rs2 import (  # noqa: F401
        DoenchRS1Scorer,
        doench_rs1_score,
    )
