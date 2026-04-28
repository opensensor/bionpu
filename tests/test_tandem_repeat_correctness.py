# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

from __future__ import annotations

import pytest

from bionpu.data.kmer_oracle import pack_dna_2bit
from bionpu.data.tandem_repeat_oracle import find_tandem_repeats_packed
from bionpu.kernels.genomics.tandem_repeat import BionpuTandemRepeat


def _maybe_skip_if_no_artifacts() -> BionpuTandemRepeat:
    op = BionpuTandemRepeat(n_tiles=4)
    if not op.artifacts_present():
        pytest.skip(f"NPU artifacts missing for {op.name}: {op.artifact_dir}")
    return op


def test_tandem_repeat_silicon_byte_equal_all_a_smoke() -> None:
    op = _maybe_skip_if_no_artifacts()
    seq = "A" * 100
    packed = pack_dna_2bit(seq)

    expected = find_tandem_repeats_packed(packed, n_bases=len(seq))
    got = op(packed_seq=packed, timeout_s=120.0)

    assert got == expected
    assert (0, 100, 1, "A") in got


def test_tandem_repeat_silicon_byte_equal_all_cg_smoke() -> None:
    op = _maybe_skip_if_no_artifacts()
    seq = "CG" * 100  # 200 bp
    packed = pack_dna_2bit(seq)

    expected = find_tandem_repeats_packed(packed, n_bases=len(seq))
    got = op(packed_seq=packed, timeout_s=120.0)

    assert got == expected
    assert (0, 200, 2, "CG") in got


def test_tandem_repeat_silicon_byte_equal_mixed() -> None:
    op = _maybe_skip_if_no_artifacts()
    seq = (
        ("T" * 60)
        + ("ACG" * 8)        # 24 bp period=3 STR
        + ("T" * 60)
        + ("CACA" * 6)       # 24 bp period=2 STR ("CA"*12)
        + ("T" * 60)
    )
    while len(seq) % 4 != 0:
        seq += "A"
    packed = pack_dna_2bit(seq)

    expected = find_tandem_repeats_packed(packed, n_bases=len(seq))
    got = op(packed_seq=packed, timeout_s=120.0)

    assert got == expected
