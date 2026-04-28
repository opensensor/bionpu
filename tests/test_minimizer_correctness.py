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

"""Silicon byte-equal correctness tests for the v0 minimizer kernel.

Skipped if NPU artifacts are not present. When artifacts are installed
under ``bionpu/dispatch/_npu_artifacts/bionpu_minimizer_k{k}_w{w}_n4/``
each test runs the full host_runner subprocess, parses the binary
blob, and asserts byte-equality vs the CPU oracle on the
``smoke_10kbp.2bit.bin`` fixture.

(k=15, w=10) is the smoke gate per acceptance criteria. (k=21, w=11)
is the long-read default. Per ``DESIGN.md`` §8 these two cells are the
v0 ship targets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bionpu.data.minimizer_oracle import extract_minimizers_packed


_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[2] / "tracks" / "genomics" / "fixtures"
)


def _smoke_buf_and_n_bases() -> tuple[np.ndarray, int]:
    bin_path = _FIXTURE_ROOT / "smoke_10kbp.2bit.bin"
    if not bin_path.exists():
        pytest.skip(f"smoke fixture missing: {bin_path}")
    buf = np.fromfile(bin_path, dtype=np.uint8)
    n_bases = buf.size * 4
    return buf, n_bases


def _maybe_skip_if_no_artifacts(k: int, w: int) -> None:
    from bionpu.kernels.genomics.minimizer import BionpuMinimizer

    op = BionpuMinimizer(k=k, w=w, n_tiles=4)
    if not op.artifacts_present():
        pytest.skip(
            f"NPU artifacts missing for {op.name} at {op.artifact_dir}; "
            f"build with `make NPU2=1 K={k} W={w} experiment=wide4 "
            f"seq=10000 all` and install per the runbook."
        )


@pytest.mark.parametrize("k,w", [(15, 10), (21, 11)])
def test_minimizer_silicon_byte_equal_smoke(k: int, w: int) -> None:
    """Silicon output for smoke_10kbp matches the CPU oracle byte-equal."""
    from bionpu.kernels.genomics.minimizer import BionpuMinimizer

    _maybe_skip_if_no_artifacts(k, w)

    buf, n_bases = _smoke_buf_and_n_bases()
    oracle = extract_minimizers_packed(buf, n_bases=n_bases, k=k, w=w)

    op = BionpuMinimizer(k=k, w=w, n_tiles=4)
    silicon = op(packed_seq=buf, top_n=0)

    assert len(silicon) == len(oracle), (
        f"(k={k}, w={w}): silicon emit count {len(silicon)} != oracle "
        f"emit count {len(oracle)}"
    )
    if silicon != oracle:
        # Surface the first divergence for easy debugging.
        for i, (s, o) in enumerate(zip(silicon, oracle)):
            if s != o:
                raise AssertionError(
                    f"(k={k}, w={w}): first diff at index {i}: "
                    f"silicon={s} oracle={o}"
                )
        raise AssertionError(
            f"(k={k}, w={w}): silicon vs oracle disagree (length-only)"
        )
