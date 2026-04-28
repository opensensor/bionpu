# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

from __future__ import annotations

import numpy as np
import pytest

from bionpu.data.kmer_oracle import pack_dna_2bit
from bionpu.dispatch.npu import NPU_OPS, NpuArtifactsMissingError
from bionpu.kernels.genomics import get_cpg_island_op
from bionpu.kernels.genomics.cpg_island import (
    BionpuCpgIsland,
    MAX_EMIT_IDX,
    PARTIAL_OUT_BYTES_PADDED,
    RECORD_BYTES,
)


def test_cpg_island_op_registered() -> None:
    assert "bionpu_cpg_island" in NPU_OPS
    op = get_cpg_island_op(n_tiles=4)
    assert isinstance(op, BionpuCpgIsland)
    assert op.n_tiles == 4


def test_cpg_island_constants_match_sparse_wire_budget() -> None:
    assert PARTIAL_OUT_BYTES_PADDED == 32768
    assert RECORD_BYTES == 4
    assert MAX_EMIT_IDX == 8190
    assert 4 + MAX_EMIT_IDX * RECORD_BYTES <= PARTIAL_OUT_BYTES_PADDED


def test_cpg_island_input_validation() -> None:
    op = BionpuCpgIsland(n_tiles=1)
    with pytest.raises(ValueError, match="dtype must be uint8"):
        op(packed_seq=np.zeros(64, dtype=np.int32))
    with pytest.raises(ValueError, match="must be 1-D"):
        op(packed_seq=np.zeros((64, 1), dtype=np.uint8))
    with pytest.raises(ValueError, match="too short"):
        op(packed_seq=np.zeros(10, dtype=np.uint8))


def test_cpg_island_missing_artifacts_reports_build_hint() -> None:
    op = BionpuCpgIsland(n_tiles=1)
    packed = pack_dna_2bit("A" * 200)
    if op.artifacts_present():
        pytest.skip("local cpg_island artifacts are present")
    with pytest.raises(NpuArtifactsMissingError, match="cpg_island"):
        op(packed_seq=packed)
