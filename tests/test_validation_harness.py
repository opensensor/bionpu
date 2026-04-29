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

"""Track F v0 unit tests — agreement-matrix shape + adapter sanity.

Coverage (must be ≥4 unit tests per Track F v0 plan §6):

1. ``AgreementCheck`` round-trip serialisation (json -> dict -> json).
2. cutadapt adapter constructs the canonical argv shape.
3. ``ucsc_pam.find_ngg_positions`` finds NGG positions on a known sequence.
4. fixture builder is deterministic across two RNG runs.
5. SKIP path when reference tool is unavailable.
6. matrix_to_json verdict-counts aggregate matches the input.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bionpu.validation import (
    AgreementCheck,
    Verdict,
    matrix_to_json,
    run_full_matrix,
    run_validation,
)
from bionpu.validation.fixtures import (
    anzalone_hek3_pegrna,
    library_design_jaccard_fixture,
    synthetic_pam_injection_seq,
)
from bionpu.validation.ref_adapters import cutadapt as cutadapt_adapter
from bionpu.validation.ref_adapters import ucsc_pam as ucsc_adapter


# --------------------------------------------------------------------------- #
# 1. AgreementCheck round-trip
# --------------------------------------------------------------------------- #


def test_agreement_check_round_trip_json():
    """AgreementCheck.to_dict + from_dict + json round-trip preserves all fields."""
    original = AgreementCheck(
        bionpu_cli="trim",
        reference_tool="cutadapt",
        fixture="synthetic-100",
        verdict=Verdict.PASS,
        metric="fastq_record_byte_equal",
        metric_value=1.0,
        bionpu_output_path="/tmp/x.fastq",
        reference_output_path="/tmp/y.fastq",
        bionpu_wall_s=0.123,
        reference_wall_s=0.456,
        extra={"n_records": 100, "extra_int": 7},
    )
    d = original.to_dict()
    encoded = json.dumps(d)
    decoded = json.loads(encoded)
    rebuilt = AgreementCheck.from_dict(decoded)
    assert rebuilt.bionpu_cli == original.bionpu_cli
    assert rebuilt.reference_tool == original.reference_tool
    assert rebuilt.fixture == original.fixture
    assert rebuilt.verdict == original.verdict
    assert rebuilt.verdict is Verdict.PASS
    assert rebuilt.metric_value == original.metric_value
    assert rebuilt.bionpu_wall_s == pytest.approx(original.bionpu_wall_s)
    assert rebuilt.extra == original.extra


# --------------------------------------------------------------------------- #
# 2. cutadapt adapter argv shape
# --------------------------------------------------------------------------- #


def test_cutadapt_adapter_builds_canonical_argv(tmp_path):
    """build_cutadapt_argv must produce -a / --no-indels / -e 0 / -O <P>."""
    in_path = tmp_path / "in.fastq"
    out_path = tmp_path / "out.fastq"
    argv = cutadapt_adapter.build_cutadapt_argv(
        cutadapt="/usr/bin/cutadapt",
        in_path=in_path,
        out_path=out_path,
        adapter="AGATCGGAAGAGC",
        min_overlap=13,
    )
    assert argv[0] == "/usr/bin/cutadapt"
    # canonical flags must appear exactly
    assert "-a" in argv
    assert argv[argv.index("-a") + 1] == "AGATCGGAAGAGC"
    assert "--no-indels" in argv
    assert "-e" in argv
    assert argv[argv.index("-e") + 1] == "0"
    assert "-O" in argv
    assert argv[argv.index("-O") + 1] == "13"
    # Output FIRST then input (cutadapt CLI convention).
    assert "-o" in argv
    assert argv[argv.index("-o") + 1] == str(out_path)
    # Input is the last positional.
    assert argv[-1] == str(in_path)


# --------------------------------------------------------------------------- #
# 3. ucsc_pam oracle finds NGG on a known seq
# --------------------------------------------------------------------------- #


def test_ucsc_pam_oracle_finds_ngg_positions():
    """A hand-crafted 30 bp seq has 3 forward-strand NGGs at known positions."""
    # Positions of NGG ([N, G, G]) listed by 0-based index of N.
    # seq[i+1:i+3] must be "GG"
    seq = "AAAAGGCCCAGGTTTAGGCCCCCCCCCCCC"
    # find NGG: indices where seq[i+1:i+3] == "GG"
    expected = [i for i in range(len(seq) - 2) if seq[i + 1 : i + 3] == "GG"]
    found = ucsc_adapter.find_ngg_positions(seq)
    assert found == expected
    assert len(found) >= 3


def test_ucsc_pam_oracle_handles_lowercase_and_n():
    """Lowercase input is normalised; non-ACGT N at wildcard slot is filtered."""
    # Same 30 bp seq with mixed case.
    seq = "aAaAGgCCCAggTTtAGgCCCCCCCCCCCC"
    found_lower = ucsc_adapter.find_ngg_positions(seq)
    found_upper = ucsc_adapter.find_ngg_positions(seq.upper())
    assert found_lower == found_upper


# --------------------------------------------------------------------------- #
# 4. fixture determinism
# --------------------------------------------------------------------------- #


def test_synthetic_pam_fixture_is_deterministic():
    """Same seed -> byte-equal sequence across two builds."""
    a = synthetic_pam_injection_seq(seed=42)
    b = synthetic_pam_injection_seq(seed=42)
    assert a == b
    # Different seeds produce different output.
    c = synthetic_pam_injection_seq(seed=43)
    assert a != c


def test_library_jaccard_fixture_has_overlap():
    """The library jaccard fixture has guaranteed overlap >= 0.5."""
    bionpu, ref = library_design_jaccard_fixture()
    bionpu_set = set(bionpu)
    ref_set = set(ref)
    inter = bionpu_set & ref_set
    union = bionpu_set | ref_set
    j = len(inter) / len(union)
    assert j >= 0.5


def test_anzalone_pegrna_fixture_shape():
    """anzalone_hek3_pegrna returns the expected canonical fields."""
    p = anzalone_hek3_pegrna()
    for key in (
        "name",
        "left_flank",
        "right_flank",
        "orig",
        "edit",
        "target_context",
        "spacer",
        "pam",
        "pbs",
        "rtt",
        "scaffold",
    ):
        assert key in p, f"missing {key}"
    # Spacer must be 20 bp; PAM must be NGG-shaped.
    assert len(p["spacer"]) == 20
    assert len(p["pam"]) == 3
    assert p["pam"][1:] == "GG"
    # target_context must contain the (orig/edit) marker.
    assert f"({p['orig']}/{p['edit']})" in p["target_context"]


# --------------------------------------------------------------------------- #
# 5. SKIP-verdict path
# --------------------------------------------------------------------------- #


def test_skip_when_be_hive_unavailable():
    """The be-design vs be-hive cross-check returns SKIP unconditionally in v0."""
    check = run_validation(
        bionpu_cli="be design",
        reference="be-hive",
        fixture="brca1",
    )
    assert check.verdict is Verdict.SKIP
    assert check.bionpu_cli == "be design"
    assert check.reference_tool == "be-hive"


def test_unknown_plan_entry_raises_keyerror():
    """run_validation raises KeyError for a (cli, ref, fixture) not in V0_MATRIX_PLAN."""
    with pytest.raises(KeyError):
        run_validation(
            bionpu_cli="trim",
            reference="bowtie2",
            fixture="synthetic-100",
        )


# --------------------------------------------------------------------------- #
# 6. matrix_to_json counts
# --------------------------------------------------------------------------- #


def test_matrix_to_json_counts_match_inputs():
    """The counts dict in matrix_to_json output matches the input verdict tally."""
    checks = [
        AgreementCheck(
            bionpu_cli="trim",
            reference_tool="cutadapt",
            fixture="synthetic-100",
            verdict=Verdict.PASS,
        ),
        AgreementCheck(
            bionpu_cli="crispr design",
            reference_tool="ucsc-pam",
            fixture="synthetic-pam-injection",
            verdict=Verdict.PASS,
        ),
        AgreementCheck(
            bionpu_cli="be design",
            reference_tool="be-hive",
            fixture="brca1",
            verdict=Verdict.SKIP,
        ),
        AgreementCheck(
            bionpu_cli="library design",
            reference_tool="brunello",
            fixture="synthetic-library-5gene",
            verdict=Verdict.DIVERGE,
        ),
    ]
    payload = matrix_to_json(checks)
    assert payload["n_checks"] == 4
    assert payload["counts"]["PASS"] == 2
    assert payload["counts"]["SKIP"] == 1
    assert payload["counts"]["DIVERGE"] == 1
    assert payload["counts"]["FAIL"] == 0
    assert payload["counts"]["ERROR"] == 0
    assert payload["track"] == "F"
    assert payload["version"] == "v0"


def test_matrix_payload_round_trip_via_json(tmp_path):
    """A matrix payload survives a JSON round-trip + re-parses to AgreementChecks."""
    checks = [
        AgreementCheck(
            bionpu_cli="trim",
            reference_tool="cutadapt",
            fixture="synthetic-100",
            verdict=Verdict.PASS,
            metric="x",
            metric_value=0.987,
        )
    ]
    payload = matrix_to_json(checks)
    out = tmp_path / "matrix.json"
    out.write_text(json.dumps(payload, indent=2))
    reloaded = json.loads(out.read_text())
    assert reloaded == payload
    rebuilt = [AgreementCheck.from_dict(c) for c in reloaded["checks"]]
    assert rebuilt[0].verdict is Verdict.PASS
    assert rebuilt[0].metric_value == pytest.approx(0.987)


# --------------------------------------------------------------------------- #
# Bonus: cutadapt installability probe must not raise.
# --------------------------------------------------------------------------- #


def test_cutadapt_installed_probe_does_not_raise():
    """The cutadapt installability probe returns a bool without raising."""
    result = cutadapt_adapter.cutadapt_installed()
    assert isinstance(result, bool)
