# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
"""Regression: NPU vs CPU scan agreement on small synthetic fixtures.

Wave 1 vendoring agent observed `npu_scan` returning 0 rows where
`cpu_scan` returned 1 expected hit on a 10kb fixture (see
`state/wave1/npu_vendoring_status.json` "scan_npu_scan_correctness_gap"
section). Root cause: the vendored xclbin's per-slot byte size did not
match the host decoder's compiled-in ``EMIT_SLOT_BYTES`` constant — the
post-2026-04-26 source bumped the slot capacity from 256 -> 1024 records
(8x slot bytes), but the vendored xclbin still uses the 256-record /
2048-byte slot layout. Reading slots at the wrong stride caused the
decoder to silently drop every slot past slot 0.

Fix: ``decode_per_slot_sparse_buffer`` now infers ``slot_bytes`` from
``len(blob) // n_slots`` when not explicitly passed, so the host stays
compatible with both vendored generations.

These tests pin that fix.
"""

from __future__ import annotations

import random

import pytest

from bionpu.data.canonical_sites import normalize, serialize_canonical
from bionpu.scan import GuideSpec, cpu_scan, npu_scan


def _ten_kb_fixture_with_one_hit() -> tuple[str, list[GuideSpec]]:
    """Build a 10 kb random ACGT genome with one perfect 20-mer + AGG hit
    placed at position 1000. This is the exact fixture geometry the
    Wave 1 vendoring agent reproduced the divergence on."""
    random.seed(2026)
    bases = "ACGT"
    guide = "AAACCCGGGTTTACGTACGT"
    prefix = "".join(random.choice(bases) for _ in range(1000))
    suffix = "".join(random.choice(bases) for _ in range(10000 - 1000 - 23))
    seq = prefix + guide + "AGG" + suffix
    return seq, [GuideSpec(spacer=guide, guide_id="g_10kb")]


def test_npu_scan_finds_hit_at_pos_1000_on_10kb_fixture() -> None:
    """The Wave 1 reproducer: 10 kb random sequence, single 20-mer + AGG
    hit at pos 1000, single guide. Pre-fix: NPU returned 0 rows while
    CPU returned 1. Post-fix: byte-equal."""
    seq, guides = _ten_kb_fixture_with_one_hit()
    cpu_rows = normalize(
        cpu_scan(chrom="chr_10kb", seq=seq, guides=guides, max_mismatches=0)
    )
    npu_rows = normalize(
        npu_scan(chrom="chr_10kb", seq=seq, guides=guides, max_mismatches=0)
    )
    assert len(cpu_rows) == 1, "fixture should have exactly one CPU hit"
    assert serialize_canonical(cpu_rows) == serialize_canonical(npu_rows), (
        f"CPU/NPU divergence on 10kb single-hit fixture.\n"
        f"CPU: {[ (r.start, r.strand, r.mismatches) for r in cpu_rows ]}\n"
        f"NPU: {[ (r.start, r.strand, r.mismatches) for r in npu_rows ]}"
    )


@pytest.mark.parametrize("hit_pos", [0, 63, 64, 100, 500, 1000, 4095, 4096, 8000])
def test_npu_scan_finds_hit_at_arbitrary_position(hit_pos: int) -> None:
    """Hit at varying genomic positions exercises the per-slot decoder
    boundary handling. Pre-fix: only positions 0..63 (slot 0) returned
    hits. Post-fix: every position works.

    Coverage rationale:
    - 0, 63: slot-0 boundary (worked even before the fix)
    - 64: first slot-1 window (broken pre-fix)
    - 100, 500, 1000: mid-range slot offsets
    - 4095: last window of chunk-0
    - 4096: first window of chunk-1 (multi-chunk path)
    - 8000: deep into chunk-1
    """
    random.seed(2027)
    bases = "ACGT"
    guide = "AAACCCGGGTTTACGTACGT"
    seq_len = max(10000, hit_pos + 30)
    parts = [random.choice(bases) for _ in range(seq_len)]
    # Stamp the guide+AGG at hit_pos.
    for i, c in enumerate(guide + "AGG"):
        parts[hit_pos + i] = c
    seq = "".join(parts)
    guides = [GuideSpec(spacer=guide, guide_id="g")]
    cpu_rows = normalize(
        cpu_scan(chrom="chr_t", seq=seq, guides=guides, max_mismatches=0)
    )
    npu_rows = normalize(
        npu_scan(chrom="chr_t", seq=seq, guides=guides, max_mismatches=0)
    )
    assert serialize_canonical(cpu_rows) == serialize_canonical(npu_rows), (
        f"CPU/NPU divergence at hit_pos={hit_pos}: "
        f"CPU={len(cpu_rows)} NPU={len(npu_rows)}"
    )


def test_npu_scan_kernel_threshold_falls_back_to_host_above_4() -> None:
    """The vendored xclbin's IRON runtime sequence hardcodes
    ``max_mismatches = 4`` into the Tile-Z emit. Callers requesting
    ``max_mismatches > 4`` must transparently fall back to the host
    emulator (which implements the same arithmetic) to stay
    correctness-equivalent with ``cpu_scan``."""
    random.seed(1234)
    seq = "".join(random.choice("ACGT") for _ in range(5000))
    guides = [GuideSpec(spacer="A" * 20, guide_id="g_polyA")]
    cpu_rows = normalize(
        cpu_scan(chrom="chr_rand", seq=seq, guides=guides, max_mismatches=10)
    )
    npu_rows = normalize(
        npu_scan(chrom="chr_rand", seq=seq, guides=guides, max_mismatches=10)
    )
    assert len(cpu_rows) > 0, "polyA at mm<=10 should find some hits"
    assert serialize_canonical(cpu_rows) == serialize_canonical(npu_rows)


@pytest.mark.parametrize("mm_thresh", [0, 1, 2, 3, 4])
def test_npu_scan_thresholds_within_kernel_cap(mm_thresh: int) -> None:
    """Mismatch thresholds 0..4 (within the kernel's hardcoded cap) must
    agree byte-for-byte. Below the cap the host post-filters the silicon
    output; at the cap the silicon output is used unfiltered."""
    random.seed(2028)
    bases = "ACGT"
    guide = "AAACCCGGGTTTACGTACGT"
    parts = [random.choice(bases) for _ in range(5000)]
    # Inject one 0-mm hit + one 1-mm hit + one 2-mm hit.
    for i, c in enumerate(guide + "AGG"):
        parts[100 + i] = c
    one_mm = guide[:-1] + ("C" if guide[-1] != "C" else "G")
    for i, c in enumerate(one_mm + "AGG"):
        parts[1000 + i] = c
    two_mm = ("C" if guide[0] != "C" else "G") + guide[1:-1] + (
        "C" if guide[-1] != "C" else "G"
    )
    for i, c in enumerate(two_mm + "AGG"):
        parts[2000 + i] = c
    seq = "".join(parts)
    guides = [GuideSpec(spacer=guide, guide_id="g")]
    cpu_rows = normalize(
        cpu_scan(chrom="chr_t", seq=seq, guides=guides, max_mismatches=mm_thresh)
    )
    npu_rows = normalize(
        npu_scan(chrom="chr_t", seq=seq, guides=guides, max_mismatches=mm_thresh)
    )
    assert serialize_canonical(cpu_rows) == serialize_canonical(npu_rows), (
        f"divergence at mm_thresh={mm_thresh}: "
        f"cpu_rows={len(cpu_rows)} npu_rows={len(npu_rows)}"
    )
