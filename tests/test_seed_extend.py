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

"""TDD tests for the seed-and-extend v0 module
(:mod:`bionpu.genomics.seed_extend`).

Test surface (5 tests; ≥5 per the PRD's acceptance criterion):

1. ``test_index_build_synthetic_smoke`` — build a tiny synthetic index
   and assert each canonical maps to its known position.
2. ``test_index_save_load_roundtrip`` — pickle round-trip preserves the
   table.
3. ``test_query_to_seeds_self_mapping`` — feed the SAME sequence in as
   query and reference; every query minimizer must hit at the same
   ``ref_pos`` it was extracted from.
4. ``test_query_to_seeds_disjoint`` — query that shares zero canonicals
   with the reference produces zero seed hits.
5. ``test_seed_extractor_oracle_path`` — ``SeedExtractor.__call__``
   end-to-end via the oracle fallback (no NPU artifacts required).
6. ``test_seed_extractor_silicon_smoke`` — silicon path on the
   ``smoke_10kbp`` fixture; SKIP if NPU artifacts missing. When
   present, the silicon-extracted seeds equal the oracle-extracted
   seeds byte-equal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bionpu.data.kmer_oracle import pack_dna_2bit
from bionpu.data.minimizer_oracle import (
    extract_minimizers,
    extract_minimizers_packed,
)
from bionpu.genomics.seed_extend import (
    MinimapIndex,
    SeedExtractor,
    SeedHit,
    build_minimap2_index,
    build_minimap2_index_from_packed,
    load_index,
    query_to_seeds_from_minimizers,
    save_index,
)


_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[2] / "tracks" / "genomics" / "fixtures"
)


# Small deterministic ACGT pattern long enough for k=15, w=10 emits
# (need at least k + w - 1 = 24 bases). 200 bases of pseudo-random ACGT.
_RNG = np.random.default_rng(seed=42)
_ALPHABET = np.array(["A", "C", "G", "T"])
_TOY_SEQ = "".join(_ALPHABET[_RNG.integers(0, 4, size=200)])


# --------------------------------------------------------------------- #
# Synthetic-data tests (no fixtures, no NPU).
# --------------------------------------------------------------------- #


def test_index_build_synthetic_smoke() -> None:
    """A built index contains every minimizer from the reference."""
    k, w = 15, 10
    index = build_minimap2_index(_TOY_SEQ, k=k, w=w)

    # Every minimizer the oracle extracts from the reference must be
    # findable in the index at the right position.
    oracle_mzs = extract_minimizers(_TOY_SEQ, k=k, w=w)
    assert len(oracle_mzs) > 0, "toy sequence should yield some minimizers"

    assert isinstance(index, MinimapIndex)
    assert index.k == k
    assert index.w == w
    assert index.n_ref_bases == len(_TOY_SEQ)
    # n_entries == #oracle minimizers (one row per emit).
    assert index.n_entries() == len(oracle_mzs)

    # Each (canonical, pos) from the oracle must show up in the index.
    for canonical, pos in oracle_mzs:
        hits = index.get(canonical)
        assert hits, f"canonical {canonical:#x} missing from index"
        assert any(rp == pos for rp, _ in hits), (
            f"canonical {canonical:#x} missing pos {pos} in {hits}"
        )


def test_index_save_load_roundtrip(tmp_path) -> None:
    """Pickled index round-trips byte-equal."""
    k, w = 15, 10
    index = build_minimap2_index(_TOY_SEQ, k=k, w=w)
    p = tmp_path / "toy.pickle"
    save_index(index, p)
    assert p.exists()

    loaded = load_index(p)
    assert loaded.k == index.k
    assert loaded.w == index.w
    assert loaded.n_ref_bases == index.n_ref_bases
    assert loaded.n_entries() == index.n_entries()
    for canonical in index.keys():
        assert loaded.get(canonical) == index.get(canonical)


def test_query_to_seeds_self_mapping() -> None:
    """Self-mapping query must find every minimizer at its own position."""
    k, w = 15, 10
    index = build_minimap2_index(_TOY_SEQ, k=k, w=w)
    query_mzs = extract_minimizers(_TOY_SEQ, k=k, w=w)

    seeds = query_to_seeds_from_minimizers(query_mzs, index)
    assert seeds, "self-mapping should produce ≥1 seed hit"

    # For each query minimizer at position q, the index has at least
    # one ref_pos == q (because the reference IS the query).
    by_query_pos: dict[int, list[SeedHit]] = {}
    for h in seeds:
        by_query_pos.setdefault(h.query_pos, []).append(h)
    for canonical, pos in query_mzs:
        bucket = by_query_pos.get(pos, [])
        assert any(h.ref_pos == pos for h in bucket), (
            f"query minimizer at pos {pos} (canonical {canonical:#x}) "
            f"failed to self-map"
        )


def test_query_to_seeds_disjoint() -> None:
    """Disjoint query (all-A, all-T) shares no canonicals → no seeds."""
    k, w = 15, 10
    # Reference: all A repeats, query: all T repeats. The all-A and
    # all-T canonicals are reverse-complements of each other and so
    # collapse to the SAME canonical (= 0). To force disjointness we
    # use a constant + a single varying base (avoid all-A AND all-T
    # which are RC partners).
    ref_seq = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"  # 30 A's
    query_seq = "ACACACACACACACACACACACACACACAC"  # alternating AC

    index = build_minimap2_index(ref_seq, k=k, w=w)
    query_mzs = extract_minimizers(query_seq, k=k, w=w)
    seeds = query_to_seeds_from_minimizers(query_mzs, index)

    # The reference is all A's → its canonical is the all-A 30-bit
    # value (= 0). The query's canonicals are alternating-AC and never
    # equal 0. Expect zero seed hits.
    ref_canonicals = set(index.keys())
    query_canonicals = {c for c, _ in query_mzs}
    assert not (ref_canonicals & query_canonicals), (
        f"disjoint test setup wrong: {ref_canonicals & query_canonicals}"
    )
    assert seeds == [], f"expected zero seeds; got {len(seeds)}"


def test_seed_extractor_oracle_path() -> None:
    """End-to-end SeedExtractor via oracle fallback (no NPU artifacts)."""
    k, w = 15, 10
    index = build_minimap2_index(_TOY_SEQ, k=k, w=w)
    extractor = SeedExtractor(index, prefer_npu=False)

    packed = pack_dna_2bit(_TOY_SEQ)
    result = extractor(packed_query=packed, n_bases=len(_TOY_SEQ))

    assert result.used_npu is False
    assert result.n_query_minimizers > 0
    assert result.seeds, "self-mapping via oracle should emit ≥1 hit"
    assert result.lookup_wall_s >= 0.0
    assert result.npu_wall_s >= 0.0
    # Hits must be sorted by (ref_pos, query_pos) ascending.
    flat = [(h.ref_pos, h.query_pos) for h in result.seeds]
    assert flat == sorted(flat), "seeds must be sorted by (ref_pos, query_pos)"


def test_seed_extractor_constructor_validation() -> None:
    """Constructor enforces (k, w) consistency with attached index."""
    k, w = 15, 10
    index = build_minimap2_index(_TOY_SEQ, k=k, w=w)

    # Mismatched k
    with pytest.raises(ValueError, match="disagrees"):
        SeedExtractor(index, k=21, w=w)

    # Mismatched w
    with pytest.raises(ValueError, match="disagrees"):
        SeedExtractor(index, k=k, w=11)

    # No index AND no k/w
    with pytest.raises(ValueError, match="prebuilt index or both k\\+w"):
        SeedExtractor(None)


# --------------------------------------------------------------------- #
# Silicon byte-equal tests (skip if artifacts missing).
# --------------------------------------------------------------------- #


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
            f"NPU artifacts missing for {op.name}; v0 minimizer must "
            f"be installed at {op.artifact_dir}."
        )


@pytest.mark.parametrize("k,w", [(15, 10)])
def test_seed_extractor_silicon_smoke(k: int, w: int) -> None:
    """Silicon end-to-end on smoke_10kbp self-mapping.

    Builds a 10-Kbp reference index from the smoke fixture, then runs
    the same fixture as the query through the silicon NPU op + lookup.
    Asserts:

    1. NPU minimizer extraction byte-equals the oracle's minimizer list.
    2. Every query minimizer self-maps (i.e., the seeds include the
       identity hit at the same position).
    """
    _maybe_skip_if_no_artifacts(k, w)
    buf, n_bases = _smoke_buf_and_n_bases()
    index = build_minimap2_index_from_packed(
        buf, n_bases=n_bases, k=k, w=w
    )

    extractor = SeedExtractor(index, prefer_npu=True)
    result = extractor(packed_query=buf, n_bases=n_bases, timeout_s=120.0)

    # Silicon path must have been taken (artifacts present per the
    # skip guard above).
    assert result.used_npu is True

    # Cross-check: the NPU's minimizer list must equal the oracle's.
    oracle_mzs = extract_minimizers_packed(buf, n_bases=n_bases, k=k, w=w)
    # The extractor doesn't expose its raw mzs directly; round-trip via
    # the seeds: re-extract on the spot.
    from bionpu.kernels.genomics.minimizer import BionpuMinimizer
    silicon_mzs = BionpuMinimizer(k=k, w=w, n_tiles=4)(
        packed_seq=buf, top_n=0, timeout_s=120.0
    )
    assert silicon_mzs == oracle_mzs, (
        f"silicon vs oracle minimizer disagreement on smoke_10kbp: "
        f"silicon={len(silicon_mzs)} oracle={len(oracle_mzs)}"
    )

    # Self-mapping check: every oracle minimizer at pos P must have a
    # seed hit with query_pos == ref_pos == P.
    have = {(h.query_pos, h.ref_pos) for h in result.seeds}
    for _, pos in oracle_mzs:
        assert (pos, pos) in have, (
            f"self-mapping miss at pos {pos}: silicon seed hits did "
            f"not contain identity"
        )


def test_query_to_seeds_freq_cutoff_skips_high_freq_canonicals():
    """v0 freq-cutoff regression: minimizers with > cutoff ref-positions
    must be skipped (mirrors minimap2 -f filter; prevents the OOM
    runaway on chr22-style references with massive homopolymer-A
    centromere regions).
    """
    from bionpu.genomics.seed_extend.lookup import (
        DEFAULT_FREQ_CUTOFF,
        query_to_seeds_from_minimizers,
    )
    from bionpu.genomics.seed_extend.index import MinimapIndex

    # Synthetic index: one canonical at 5 ref positions ("low-freq"),
    # another at 5000 ref positions ("high-freq"; should be skipped).
    table = {
        0xAAAA: [(p, 0) for p in range(5)],
        0xBBBB: [(p, 0) for p in range(5000)],
    }
    idx = MinimapIndex(k=15, w=10, n_ref_bases=10000, table=table)

    # Query has both canonicals.
    query_mzs = [(0xAAAA, 100), (0xBBBB, 200)]

    # With default cutoff (1000): only the low-freq canonical contributes.
    seeds_default = query_to_seeds_from_minimizers(query_mzs, idx)
    assert len(seeds_default) == 5
    assert all(s.canonical == 0xAAAA for s in seeds_default)

    # With cutoff=None (disabled): both contribute.
    seeds_none = query_to_seeds_from_minimizers(query_mzs, idx, freq_cutoff=None)
    assert len(seeds_none) == 5005

    # With cutoff=10 (tighter): only low-freq still passes.
    seeds_tight = query_to_seeds_from_minimizers(query_mzs, idx, freq_cutoff=10)
    assert len(seeds_tight) == 5

    # DEFAULT_FREQ_CUTOFF must be a sane value (positive, between
    # minimap2's typical -f=0.0002 × 5M ≈ 1000 and a soft upper bound).
    assert 100 <= DEFAULT_FREQ_CUTOFF <= 100_000
