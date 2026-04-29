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
from bionpu.data.methylation_context_oracle import (
    find_methylation_contexts_packed,
)
from bionpu.kernels.genomics import get_methylation_context_op
from bionpu.kernels.genomics.methylation_context import (
    BionpuMethylationContext,
)


def _maybe_skip_if_no_artifacts() -> BionpuMethylationContext:
    op = BionpuMethylationContext(n_tiles=4)
    if not op.artifacts_present():
        pytest.skip(f"NPU artifacts missing for {op.name}: {op.artifact_dir}")
    return op


def test_methylation_context_op_validates_input_shape() -> None:
    op = BionpuMethylationContext(n_tiles=4)
    with pytest.raises(ValueError, match="dtype must be uint8"):
        op._validate_inputs(np.array([0], dtype=np.uint16), 4)
    with pytest.raises(ValueError, match="1-D"):
        op._validate_inputs(np.zeros((1, 1), dtype=np.uint8), 4)
    with pytest.raises(ValueError, match="requires 2"):
        op._validate_inputs(np.zeros(1, dtype=np.uint8), 5)


def test_get_methylation_context_op_returns_requested_tile_count() -> None:
    op = get_methylation_context_op(n_tiles=2, seq_chunk_bytes_base=2048)
    assert isinstance(op, BionpuMethylationContext)
    assert op.n_tiles == 2
    assert op.seq_chunk_bytes_base == 2048
    assert op.artifact_dir.name == "bionpu_methylation_context_n2_c2048"


def test_methylation_context_silicon_byte_equal_mixed_smoke() -> None:
    op = _maybe_skip_if_no_artifacts()
    seq = "ACGCCAGCTGG"
    packed = pack_dna_2bit(seq)

    expected = find_methylation_contexts_packed(packed, n_bases=len(seq))
    got = op(packed_seq=packed, n_bases=len(seq), timeout_s=120.0)

    assert got == expected
    assert op.last_run is not None
    assert op.last_run.n_records == len(expected)


def test_methylation_context_silicon_byte_equal_all_a_smoke() -> None:
    op = _maybe_skip_if_no_artifacts()
    seq = "A" * 128
    packed = pack_dna_2bit(seq)

    expected = find_methylation_contexts_packed(packed, n_bases=len(seq))
    got = op(packed_seq=packed, n_bases=len(seq), timeout_s=120.0)

    assert got == expected == []
