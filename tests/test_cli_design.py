# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

from __future__ import annotations

import json

from bionpu.cli import main


def test_design_cli_writes_guide_seed_prefilter_json(tmp_path) -> None:
    spacer = "ACGTACGTACGTACGTACGT"
    target = tmp_path / "target.fa"
    reference = tmp_path / "reference.fa"
    out = tmp_path / "design.json"

    target.write_text(f">target\n{spacer}AGG\n")
    reference.write_text(
        f">ref\nTT{spacer}AGGCC{spacer[:-1]}AAGG\n"
    )

    rc = main(
        [
            "design",
            "--target",
            str(target),
            "--reference",
            str(reference),
            "--out",
            str(out),
            "--max-seed-mismatches",
            "1",
        ]
    )

    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["target"] == {"chrom": "target", "bases": 23}
    assert payload["reference"] == {"chrom": "ref", "bases": 50}
    assert payload["pam_templates"] == ["NGG"]
    assert payload["n_candidates"] == 1
    assert payload["n_passing"] == 1
    assert payload["n_rejected"] == 0
    assert payload["seed_hit_count"] == 2

    guide = payload["guides"][0]
    assert guide["spacer"] == spacer
    assert guide["pam"] == "AGG"
    assert guide["exact_seed_hit_count"] == 1
    assert guide["mismatched_seed_hit_count"] == 1
    assert guide["reference_names"] == ["ref"]
    assert [hit["seed_mismatches"] for hit in guide["off_targets"]] == [0, 1]
