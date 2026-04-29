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

"""Track F v0 integration tests.

Runs `bionpu validate ...` end-to-end (in-process; no subprocess) and
asserts the agreement matrix shape + a couple of PASS verdicts.

Per Track F v0 plan §6, must be ≥3 integration tests.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from bionpu.validation import (
    AgreementCheck,
    Verdict,
    run_full_matrix,
    run_validation,
)
from bionpu.validation.cli import run_cli as validate_run_cli
from bionpu.validation.ref_adapters import cutadapt as cutadapt_adapter


def _have_cutadapt() -> bool:
    return cutadapt_adapter.cutadapt_installed()


def _have_synthetic_100_fixture() -> bool:
    p = (
        Path(__file__).resolve().parents[2]
        / "tracks"
        / "genomics"
        / "fixtures"
        / "synthetic_reads_with_adapters.fastq"
    )
    return p.is_file()


@pytest.mark.skipif(
    not _have_cutadapt() or not _have_synthetic_100_fixture(),
    reason="cutadapt or synthetic-100 fixture unavailable",
)
def test_validate_trim_vs_cutadapt_passes(tmp_path):
    """Integration: bionpu trim vs cutadapt on synthetic-100 -> PASS."""
    check = run_validation(
        bionpu_cli="trim",
        reference="cutadapt",
        fixture="synthetic-100",
        workspace=tmp_path,
    )
    assert check.verdict is Verdict.PASS, (
        f"verdict={check.verdict.value} divergence={check.divergence_summary}"
    )
    assert check.metric == "fastq_record_byte_equal"
    assert check.metric_value == pytest.approx(1.0)
    # Workspace must contain both outputs.
    assert (tmp_path / "trim_bionpu.fastq").is_file()
    assert (tmp_path / "trim_cutadapt.fastq").is_file()
    # Walls captured.
    assert check.bionpu_wall_s is not None and check.bionpu_wall_s > 0
    assert check.reference_wall_s is not None and check.reference_wall_s > 0


def test_validate_crispr_design_vs_ucsc_pam_passes(tmp_path):
    """Integration: bionpu crispr design PAM scan vs UCSC oracle -> PASS."""
    check = run_validation(
        bionpu_cli="crispr design",
        reference="ucsc-pam",
        fixture="synthetic-pam-injection",
        workspace=tmp_path,
    )
    # The CPU PAM-finder is exactly the UCSC oracle, so jaccard==1.0.
    assert check.verdict is Verdict.PASS, (
        f"verdict={check.verdict.value} divergence={check.divergence_summary}"
    )
    assert check.metric == "ngg_position_jaccard"
    assert check.metric_value == pytest.approx(1.0)


def test_validate_all_emits_full_matrix(tmp_path):
    """Integration: run_full_matrix populates one AgreementCheck per V0_MATRIX_PLAN row."""
    from bionpu.validation.agreement import V0_MATRIX_PLAN

    checks = run_full_matrix(workspace=tmp_path)
    assert len(checks) == len(V0_MATRIX_PLAN)
    # Every check must have a populated verdict.
    for c in checks:
        assert isinstance(c, AgreementCheck)
        assert isinstance(c.verdict, Verdict)


def test_validate_all_via_cli_writes_matrix_json(tmp_path):
    """Integration: `bionpu validate all` CLI path writes the agreement matrix file."""
    out_path = tmp_path / "agreement_matrix.json"

    # Build a fake argparse Namespace with the same shape the CLI produces.
    class _Args:
        validate_target = "all"
        output = str(out_path)
        workspace = str(tmp_path)

    # Capture the underlying integer return code; matrix file must exist.
    rc = validate_run_cli(_Args)
    # rc is 0 unless any FAIL/ERROR — in this env we expect 0 (PRIDICT2 +
    # cutadapt + cas-offinder all PASS). But we tolerate 1 because
    # environments without cas-offinder/PRIDICT2 may still hit ERROR
    # paths during transient network or process issues.
    assert rc in (0, 1)
    assert out_path.is_file()
    payload = json.loads(out_path.read_text())
    assert payload["track"] == "F"
    assert payload["version"] == "v0"
    assert payload["n_checks"] == len(payload["checks"])
    # Counts add up.
    counts = payload["counts"]
    assert sum(counts.values()) == payload["n_checks"]
    # At least one PASS in this env (cutadapt + ucsc-pam are no-deps).
    assert counts["PASS"] >= 1


def test_validate_be_design_skip_path():
    """Integration: be-design vs be-hive returns SKIP cleanly."""
    check = run_validation(
        bionpu_cli="be design",
        reference="be-hive",
        fixture="brca1",
    )
    assert check.verdict is Verdict.SKIP
    assert "BE-Hive" in (check.divergence_summary or "")


def test_main_cli_validate_help_exits_zero():
    """The umbrella CLI exposes `bionpu validate --help` without error."""
    from bionpu.cli import main as cli_main

    # argparse's --help raises SystemExit(0). Catch it.
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["validate", "--help"])
    assert exc_info.value.code == 0
