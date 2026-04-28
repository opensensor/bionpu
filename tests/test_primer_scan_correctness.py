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

"""Silicon byte-equal correctness tests for the v0 primer_scan kernel.

Skipped if NPU artifacts are not present. When artifacts are installed
under ``bionpu/dispatch/_npu_artifacts/bionpu_primer_scan_p{P}_n4/``
each test runs the full host_runner subprocess, parses the binary
blob, and asserts byte-equality vs the CPU oracle on the
``smoke_10kbp.2bit.bin`` fixture.

(P=13, TruSeq P5 adapter) is the smoke gate per the v0 acceptance
criteria. (P=20, P=25) are tested with synthetic injected fixtures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bionpu.data.kmer_oracle import pack_dna_2bit, unpack_dna_2bit
from bionpu.data.primer_oracle import (
    TRUSEQ_P5_ADAPTER,
    find_primer_matches,
    find_primer_matches_packed,
)


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


def _maybe_skip_if_no_artifacts(p: int) -> None:
    from bionpu.kernels.genomics.primer_scan import BionpuPrimerScan

    primer = "A" * p  # placeholder; constructor only needs valid primer
    op = BionpuPrimerScan(primer=primer, n_tiles=4)
    if not op.artifacts_present():
        pytest.skip(
            f"NPU artifacts missing for {op.name} at {op.artifact_dir}; "
            f"build with `make NPU2=1 P={p} experiment=wide4 seq=10000 all` "
            f"and install per the runbook."
        )


# --------------------------------------------------------------------------- #
# Smoke (P=13 TruSeq P5 adapter) silicon byte-equal
# --------------------------------------------------------------------------- #


def test_primer_scan_silicon_byte_equal_smoke_truseq_p5() -> None:
    """Silicon output for smoke_10kbp scanning the TruSeq P5 adapter
    matches the CPU oracle byte-equal.

    Note: chr22 / random ACGT may genuinely have 0 hits for a 13-bp
    Illumina adapter; both oracle and silicon must emit 0 records in
    that case.
    """
    from bionpu.kernels.genomics.primer_scan import BionpuPrimerScan

    _maybe_skip_if_no_artifacts(13)

    buf, n_bases = _smoke_buf_and_n_bases()
    oracle = find_primer_matches_packed(
        buf, n_bases=n_bases, primer=TRUSEQ_P5_ADAPTER, allow_rc=True
    )

    op = BionpuPrimerScan(primer=TRUSEQ_P5_ADAPTER, n_tiles=4)
    silicon = op(packed_seq=buf, top_n=0)

    assert len(silicon) == len(oracle), (
        f"silicon emit count {len(silicon)} != oracle emit count "
        f"{len(oracle)}"
    )
    if silicon != oracle:
        for i, (s, o) in enumerate(zip(silicon, oracle)):
            if s != o:
                raise AssertionError(
                    f"first diff at index {i}: silicon={s} oracle={o}"
                )


# --------------------------------------------------------------------------- #
# Synthetic injected smoke (forces non-zero hits regardless of fixture)
# --------------------------------------------------------------------------- #


def _build_synthetic_fixture(
    primer: str, n_inject: int, gap_bases: int = 4096
) -> tuple[np.ndarray, int, list[tuple[int, int]]]:
    """Build a synthetic packed-2-bit query with primers injected at
    deterministic positions, separated by stretches of A. Returns the
    packed buffer, total n_bases, and the list of expected
    (position, strand) hits.
    """
    p = len(primer)
    seq_parts: list[str] = []
    expected: list[tuple[int, int]] = []
    cur = 0
    for i in range(n_inject):
        # Pad with A then forward primer.
        pad_a = "A" * gap_bases
        seq_parts.append(pad_a)
        cur += len(pad_a)
        seq_parts.append(primer)
        expected.append((cur, 0))  # forward
        cur += p
    # Pad to multiple of 4 for clean packing.
    while cur % 4 != 0:
        seq_parts.append("A")
        cur += 1
    seq = "".join(seq_parts)
    packed = pack_dna_2bit(seq)
    return packed, cur, expected


@pytest.mark.parametrize("p,primer", [
    (13, "AGATCGGAAGAGC"),
    (20, "ACGTACGTACGTACGTACGT"),  # P=20 synthetic
    (25, "ACGTACGTACGTACGTACGTACGTA"),  # P=25 synthetic
])
def test_primer_scan_silicon_byte_equal_synthetic(p: int, primer: str) -> None:
    """Inject 5 forward-strand primer matches into a synthetic A-stretch
    fixture; silicon must emit exactly those 5 records (forward strand
    only, since the all-A padding cannot accidentally match the RC).
    """
    from bionpu.kernels.genomics.primer_scan import BionpuPrimerScan

    _maybe_skip_if_no_artifacts(p)

    packed, n_bases, expected = _build_synthetic_fixture(primer, n_inject=5)
    seq = unpack_dna_2bit(packed, n_bases)
    oracle = find_primer_matches(seq, primer, allow_rc=True)
    # Sanity: oracle must include all 5 injected positions.
    for inj in expected:
        assert inj in oracle, (
            f"oracle missing injected hit {inj} (oracle has {oracle})"
        )

    op = BionpuPrimerScan(primer=primer, n_tiles=4)
    silicon = op(packed_seq=packed, top_n=0)

    assert silicon == oracle, (
        f"silicon != oracle for P={p}; silicon={silicon!r} "
        f"oracle={oracle!r}"
    )
