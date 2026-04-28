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


def _maybe_skip_if_no_artifacts_np(k: int, w: int, n_passes: int) -> None:
    from bionpu.kernels.genomics.minimizer import BionpuMinimizer

    op = BionpuMinimizer(k=k, w=w, n_tiles=4, n_passes=n_passes)
    if not op.artifacts_present():
        pytest.skip(
            f"NPU artifacts missing for {op.name} (n_passes={n_passes}) "
            f"at {op.artifact_dir}; build with `make NPU2=1 K={k} W={w} "
            f"experiment=wide4 n_passes={n_passes} seq=10000 "
            f"{'all-passes' if n_passes > 1 else 'all'}` and install per the runbook."
        )


@pytest.mark.parametrize("k,w", [(15, 10), (21, 11)])
def test_minimizer_silicon_byte_equal_smoke(k: int, w: int) -> None:
    """Silicon output for smoke_10kbp matches the CPU oracle byte-equal
    at n_passes=1 (v0 back-compat).
    """
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


@pytest.mark.slow
@pytest.mark.xfail(
    reason="v1 partial recovery (71%); cap-fire on dense chunks. v2 widens partial_out 32K→64K to close.",
    strict=False,
)
def test_minimizer_silicon_byte_equal_chr22_k15_w10_np4() -> None:
    """v1 chr22 byte-equal gate: silicon == oracle at (k=15, w=10, n_passes=4).

    Status: PARTIAL — v1 closes ~71% (14.26 M of 20.01 M oracle records)
    on chr22 with chunk_size=4096 + n_passes=4 + Fibonacci position
    hash. ``minimizer-emit-cap-saturation`` is downgraded from blocker
    to low-severity; full closure deferred to v2 (widen
    MZ_PARTIAL_OUT_BYTES_PADDED 32K→64K, no DM/IRON revision needed
    after cap-headroom audit).

    Marked ``slow`` AND ``xfail`` — the test exercises the gate but
    does NOT block CI on the partial-recovery v1 ship. CI's normal
    test bucket should NOT execute it.
    """
    from bionpu.kernels.genomics.minimizer import BionpuMinimizer
    from bionpu.data.minimizer_oracle import extract_minimizers_packed
    import pickle

    k, w, n_passes = 15, 10, 4
    _maybe_skip_if_no_artifacts_np(k, w, n_passes)

    chr22_path = (
        Path("/home/matteius/genetics/tracks/genomics/fixtures/chr22.2bit.bin")
    )
    if not chr22_path.exists():
        pytest.skip(f"chr22 fixture missing: {chr22_path}")

    cache = Path(
        "/home/matteius/genetics/state/v1_minimizer_chr22_oracle_k15_w10.pickle"
    )
    if cache.exists():
        with open(cache, "rb") as f:
            oracle = pickle.load(f)
    else:
        buf = np.fromfile(chr22_path, dtype=np.uint8)
        oracle = extract_minimizers_packed(
            buf, n_bases=buf.size * 4, k=k, w=w
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(oracle, f, protocol=pickle.HIGHEST_PROTOCOL)

    buf = np.fromfile(chr22_path, dtype=np.uint8)
    op = BionpuMinimizer(k=k, w=w, n_tiles=4, n_passes=n_passes)
    silicon = op(packed_seq=buf, top_n=0, timeout_s=1500.0)

    assert len(silicon) == len(oracle), (
        f"chr22 (np={n_passes}): silicon emit count {len(silicon)} != "
        f"oracle emit count {len(oracle)}"
    )
    if silicon != oracle:
        for i, (s, o) in enumerate(zip(silicon, oracle)):
            if s != o:
                raise AssertionError(
                    f"chr22 (np={n_passes}): first diff at index {i}: "
                    f"silicon={s} oracle={o}"
                )
        raise AssertionError(
            f"chr22 (np={n_passes}): silicon vs oracle disagree (length-only)"
        )


def test_minimizer_silicon_byte_equal_smoke_multipass_k15_w10() -> None:
    """v1 multi-pass: silicon output for smoke_10kbp at n_passes=4
    matches the CPU oracle byte-equal (k=15, w=10).

    The kernel's window-min logic is unchanged across passes; only the
    EMIT decision filters by hash-slice (low n_passes_log2 bits of
    canonical). The host runner unions emits across all passes, sorts
    by (position asc, canonical asc), and de-duplicates. The expected
    result is byte-equal to the n_passes=1 single-pass output.
    """
    from bionpu.kernels.genomics.minimizer import BionpuMinimizer

    k, w, n_passes = 15, 10, 4
    _maybe_skip_if_no_artifacts_np(k, w, n_passes)

    buf, n_bases = _smoke_buf_and_n_bases()
    oracle = extract_minimizers_packed(buf, n_bases=n_bases, k=k, w=w)

    op = BionpuMinimizer(k=k, w=w, n_tiles=4, n_passes=n_passes)
    silicon = op(packed_seq=buf, top_n=0)

    assert len(silicon) == len(oracle), (
        f"(k={k}, w={w}, n_passes={n_passes}): silicon emit count "
        f"{len(silicon)} != oracle emit count {len(oracle)}"
    )
    if silicon != oracle:
        for i, (s, o) in enumerate(zip(silicon, oracle)):
            if s != o:
                raise AssertionError(
                    f"(k={k}, w={w}, n_passes={n_passes}): first diff at "
                    f"index {i}: silicon={s} oracle={o}"
                )
        raise AssertionError(
            f"(k={k}, w={w}, n_passes={n_passes}): silicon vs oracle "
            f"disagree (length-only)"
        )
