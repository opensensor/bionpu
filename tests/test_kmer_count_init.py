# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

"""Registration + import smoke for ``bionpu.kernels.genomics.kmer_count``.

Per ``state/kmer_count_interface_contract.md`` (T1), the package must
register exactly three NPU ops (``bionpu_kmer_count_k{15,21,31}``) at
import time, and :class:`BionpuKmerCount` must accept ``(k, n_tiles)``
constructor args. Silicon validation lives in T13.
"""

from __future__ import annotations

import os

import numpy as np
import pytest


def test_kmer_count_registers_three_ops():
    """The three k-specific registry entries land at import time."""
    import bionpu.kernels.genomics.kmer_count  # noqa: F401  (registration import)

    from bionpu.dispatch.npu import NPU_OPS

    for k in (15, 21, 31):
        name = f"bionpu_kmer_count_k{k}"
        assert name in NPU_OPS, f"{name} not registered"


def test_bionpu_kmer_count_constructor_defaults():
    from bionpu.kernels.genomics.kmer_count import BionpuKmerCount

    op = BionpuKmerCount()
    assert op.k == 21
    assert op.n_tiles == 4
    assert op.name == "bionpu_kmer_count_k21"


def test_bionpu_kmer_count_artifact_dir_matrix():
    """Artifact dir matches the 3 x 4 = 12-cell on-disk matrix."""
    from bionpu.kernels.genomics.kmer_count import BionpuKmerCount

    for k in (15, 21, 31):
        for n_tiles in (1, 2, 4, 8):
            op = BionpuKmerCount(k=k, n_tiles=n_tiles)
            assert op.artifact_dir.name == f"bionpu_kmer_count_k{k}_n{n_tiles}"


def test_bionpu_kmer_count_artifacts_present_returns_bool():
    from bionpu.kernels.genomics.kmer_count import BionpuKmerCount

    op = BionpuKmerCount(k=21, n_tiles=4)
    # T11 vendors the artifacts; before T11 lands the answer is False.
    assert isinstance(op.artifacts_present(), bool)


@pytest.mark.parametrize("bad_k", [10, 16, 32, 64])
def test_bionpu_kmer_count_rejects_unsupported_k(bad_k):
    from bionpu.kernels.genomics.kmer_count import BionpuKmerCount

    with pytest.raises(ValueError, match="k="):
        BionpuKmerCount(k=bad_k)


@pytest.mark.parametrize("bad_n", [0, 3, 5, 16])
def test_bionpu_kmer_count_rejects_unsupported_n_tiles(bad_n):
    from bionpu.kernels.genomics.kmer_count import BionpuKmerCount

    with pytest.raises(ValueError, match="n_tiles="):
        BionpuKmerCount(k=21, n_tiles=bad_n)


def test_resolve_dispatch_impl_default():
    from bionpu.kernels.genomics.kmer_count import BionpuKmerCount

    op = BionpuKmerCount(k=21, n_tiles=4)
    # Strip env so the default kicks in.
    prev = os.environ.pop("BIONPU_KMER_COUNT_DISPATCH", None)
    try:
        assert op._resolve_dispatch_impl(None) == "subprocess"
    finally:
        if prev is not None:
            os.environ["BIONPU_KMER_COUNT_DISPATCH"] = prev


def test_resolve_dispatch_impl_pyxrt_via_env(monkeypatch):
    from bionpu.kernels.genomics.kmer_count import BionpuKmerCount

    monkeypatch.setenv("BIONPU_KMER_COUNT_DISPATCH", "pyxrt")
    op = BionpuKmerCount(k=21, n_tiles=4)
    assert op._resolve_dispatch_impl(None) == "pyxrt"


def test_resolve_dispatch_impl_explicit_overrides_env(monkeypatch):
    from bionpu.kernels.genomics.kmer_count import BionpuKmerCount

    monkeypatch.setenv("BIONPU_KMER_COUNT_DISPATCH", "pyxrt")
    op = BionpuKmerCount(k=21, n_tiles=4)
    assert op._resolve_dispatch_impl("subprocess") == "subprocess"


def test_resolve_dispatch_impl_rejects_garbage(monkeypatch):
    from bionpu.kernels.genomics.kmer_count import BionpuKmerCount

    op = BionpuKmerCount(k=21, n_tiles=4)
    with pytest.raises(ValueError):
        op._resolve_dispatch_impl("nonsense")


def test_input_validation_rejects_wrong_dtype():
    from bionpu.kernels.genomics.kmer_count import BionpuKmerCount

    op = BionpuKmerCount(k=21, n_tiles=4)
    with pytest.raises(ValueError, match="dtype"):
        op._validate_inputs(np.zeros(64, dtype=np.int32))


def test_input_validation_rejects_2d():
    from bionpu.kernels.genomics.kmer_count import BionpuKmerCount

    op = BionpuKmerCount(k=21, n_tiles=4)
    with pytest.raises(ValueError, match="1-D"):
        op._validate_inputs(np.zeros((4, 16), dtype=np.uint8))


def test_input_validation_rejects_too_short():
    from bionpu.kernels.genomics.kmer_count import BionpuKmerCount

    op = BionpuKmerCount(k=31, n_tiles=4)
    with pytest.raises(ValueError, match="too short"):
        # k=31 needs ceil(62/8) = 8 bytes; 4 is too few.
        op._validate_inputs(np.zeros(4, dtype=np.uint8))


def test_overlap_bytes_for():
    from bionpu.kernels.genomics.kmer_count import overlap_bytes_for

    assert overlap_bytes_for(15) == 4
    assert overlap_bytes_for(21) == 5
    assert overlap_bytes_for(31) == 8


def test_decode_packed_kmer_to_ascii_known_values():
    from bionpu.kernels.genomics.kmer_count import (
        _ascii_kmer_to_canonical_u64,
        decode_packed_kmer_to_ascii,
    )

    # All-A k-mer is canonical 0.
    assert decode_packed_kmer_to_ascii(0, 15) == "A" * 15
    # All-T k-mer is canonical = mask (every bit pair = 0b11).
    canonical_all_t = (1 << 30) - 1
    assert decode_packed_kmer_to_ascii(canonical_all_t, 15) == "T" * 15
    # Roundtrip a couple of arbitrary values.
    for k, val in ((15, 0x3F), (21, 0x12345), (31, 0xCAFEBABE)):
        ascii_kmer = decode_packed_kmer_to_ascii(val, k)
        assert _ascii_kmer_to_canonical_u64(ascii_kmer, k) == val


@pytest.mark.skip(reason="dispatch_class_with_npu_dependency — silicon test is T13")
def test_silicon_call_smoke():
    """Silicon-call smoke is T13 (kmer-silicon-validation.py)."""
