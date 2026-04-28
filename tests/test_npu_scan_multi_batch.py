# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
"""Regression: NPU scan with >128 guides via host-side multi-launch.

The PAM-filter kernel ABI is shape-pinned to ``N_GUIDES = 128`` guides
per launch. Pre-fix: ``encode_guide_batch`` raised
``NotImplementedError`` for any larger guide list, blocking every gene
in the pinned 20-gene CRISPOR validation set (BRCA1 alone produces 8968
candidates).

Fix: ``encode_guide_batches`` chunks the input list into 128-guide
batches; ``npu_scan`` issues one silicon launch per batch and
concatenates the hit lists. Output stays byte-equal to ``cpu_scan``.

These tests pin that contract.
"""

from __future__ import annotations

import random

import pytest

from bionpu.data.canonical_sites import normalize, serialize_canonical
from bionpu.scan import (
    GuideSpec,
    cpu_scan,
    encode_guide_batches,
    npu_scan,
)


def _random_guide(rng: random.Random) -> str:
    return "".join(rng.choice("ACGT") for _ in range(20))


def test_encode_guide_batches_splits_at_128() -> None:
    """encode_guide_batches must produce ceil(n/128) batches, each of the
    fixed kernel ABI shape."""
    from bionpu.kernels.crispr.match_singletile import SPACER_BYTES
    from bionpu.kernels.crispr.pam_filter import N_GUIDES

    rng = random.Random(0)
    n = 300  # 3 batches: 128 + 128 + 44
    guides = [
        GuideSpec(spacer=_random_guide(rng), guide_id=f"g_{i}")
        for i in range(n)
    ]
    batches = encode_guide_batches(guides)
    assert len(batches) == 3
    for b_2bit, pad in batches:
        assert b_2bit.shape == (N_GUIDES, SPACER_BYTES)
        assert b_2bit.dtype.kind == "u" and b_2bit.dtype.itemsize == 1
        assert len(pad) == N_GUIDES
    # First batch: all 128 real guides
    assert all(p is not None for p in batches[0][1])
    # Second batch: all 128 real guides
    assert all(p is not None for p in batches[1][1])
    # Third batch: 44 real, 84 padded
    real_count = sum(1 for p in batches[2][1] if p is not None)
    padded_count = sum(1 for p in batches[2][1] if p is None)
    assert real_count == 44
    assert padded_count == 84


def test_npu_scan_multi_batch_does_not_raise_above_128_guides() -> None:
    """The pre-fix block: ``encode_guide_batch`` raised
    NotImplementedError for >128 guides. Post-fix: ``npu_scan``
    transparently chunks and runs."""
    rng = random.Random(42)
    seq = "".join(rng.choice("ACGT") for _ in range(20000))
    # 200 distinct random guides — past the 128-guide cap
    guides = [
        GuideSpec(spacer=_random_guide(rng), guide_id=f"g_{i}")
        for i in range(200)
    ]
    rows = npu_scan(
        chrom="chr_multi", seq=seq, guides=guides, max_mismatches=4
    )
    # We don't assert a row count (random input may yield zero),
    # only that it didn't raise.
    assert isinstance(rows, list)


def test_npu_scan_multi_batch_byte_equal_cpu() -> None:
    """Headline contract: ``npu_scan`` over >128 guides matches
    ``cpu_scan`` row-for-row."""
    rng = random.Random(2026)
    bases = "ACGT"
    parts = [rng.choice(bases) for _ in range(20000)]
    # Inject a few hand-placed 0-mm hits at distinct positions so we
    # exercise the multi-batch path with non-empty results.
    seeded_guides = [
        ("AAACCCGGGTTTACGTACGT", 1000),
        ("CCCCGGGGTTTTAAAAACGT", 5000),
        ("TTTTAAAAACCCCGGGGACG", 10000),
        ("GGGGAAAATTTTCCCCAAGT", 15000),
    ]
    for spacer, pos in seeded_guides:
        for i, c in enumerate(spacer + "AGG"):
            parts[pos + i] = c
    seq = "".join(parts)

    # 200 guides total: the 4 seeded + 196 random distractors
    rng2 = random.Random(2027)
    guides: list[GuideSpec] = [
        GuideSpec(spacer=spacer, guide_id=f"seed_{i}")
        for i, (spacer, _) in enumerate(seeded_guides)
    ]
    for i in range(196):
        guides.append(
            GuideSpec(spacer=_random_guide(rng2), guide_id=f"d_{i}")
        )
    assert len(guides) == 200

    cpu_rows = normalize(
        cpu_scan(chrom="chr_multi", seq=seq, guides=guides, max_mismatches=4)
    )
    npu_rows = normalize(
        npu_scan(chrom="chr_multi", seq=seq, guides=guides, max_mismatches=4)
    )
    assert len(cpu_rows) >= 4, "expected at least the 4 seeded hits"
    assert serialize_canonical(cpu_rows) == serialize_canonical(npu_rows), (
        f"CPU/NPU byte-equality broken on multi-batch fixture.\n"
        f"CPU rows: {len(cpu_rows)}\nNPU rows: {len(npu_rows)}"
    )


@pytest.mark.parametrize("n_guides", [128, 129, 256, 300])
def test_npu_scan_multi_batch_at_boundary_sizes(n_guides: int) -> None:
    """Boundary cases: exactly N_GUIDES (single batch), N_GUIDES+1
    (forces a 2nd batch with only one real guide), and larger
    multi-batch sizes."""
    rng = random.Random(n_guides)
    parts = [rng.choice("ACGT") for _ in range(8000)]
    # Inject a hit for the LAST guide so multi-batch boundaries matter.
    last_spacer = _random_guide(rng)
    for i, c in enumerate(last_spacer + "AGG"):
        parts[2000 + i] = c
    seq = "".join(parts)

    guides = [
        GuideSpec(spacer=_random_guide(rng), guide_id=f"g_{i}")
        for i in range(n_guides - 1)
    ]
    guides.append(GuideSpec(spacer=last_spacer, guide_id=f"g_last"))
    cpu_rows = normalize(
        cpu_scan(chrom="chr_b", seq=seq, guides=guides, max_mismatches=0)
    )
    npu_rows = normalize(
        npu_scan(chrom="chr_b", seq=seq, guides=guides, max_mismatches=0)
    )
    assert serialize_canonical(cpu_rows) == serialize_canonical(npu_rows), (
        f"divergence at n_guides={n_guides}"
    )
    # The last guide's hit must show up (in either path).
    last_id_rows = [r for r in cpu_rows if r.guide_id == "g_last"]
    assert len(last_id_rows) >= 1, "last guide's seeded hit missing"
