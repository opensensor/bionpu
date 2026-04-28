# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

from __future__ import annotations

import pytest

from bionpu.data.cpg_oracle import find_cpg_islands_packed
from bionpu.data.kmer_oracle import pack_dna_2bit
from bionpu.kernels.genomics.cpg_island import BionpuCpgIsland


def _maybe_skip_if_no_artifacts() -> BionpuCpgIsland:
    op = BionpuCpgIsland(n_tiles=4)
    if not op.artifacts_present():
        pytest.skip(f"NPU artifacts missing for {op.name}: {op.artifact_dir}")
    return op


def test_cpg_island_silicon_byte_equal_all_cg_smoke() -> None:
    op = _maybe_skip_if_no_artifacts()
    seq = "CG" * 500
    packed = pack_dna_2bit(seq)

    expected = find_cpg_islands_packed(packed, n_bases=len(seq))
    got = op(packed_seq=packed, timeout_s=120.0)

    assert got == expected == [(0, 1000)]
    assert op.last_run is not None
    assert op.last_run.n_candidates == 801


def test_cpg_island_silicon_byte_equal_all_a_smoke() -> None:
    op = _maybe_skip_if_no_artifacts()
    seq = "A" * 1000
    packed = pack_dna_2bit(seq)

    expected = find_cpg_islands_packed(packed, n_bases=len(seq))
    got = op(packed_seq=packed, timeout_s=120.0)

    assert got == expected == []
