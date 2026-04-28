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
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""kmer_count correctness tests + smoke regression-lock (T12).

Per the k-mer counting plan (`/home/matteius/genetics/kmer-counting-plan.md`)
and `state/kmer_count_interface_contract.md` (T1) — symbol names,
ObjectFifo names, constants, byte-order, hash overflow policy, and the
streaming chunk + overlap protocol are pinned in T1. This test file
validates the on-NPU kernel against the slow-but-correct numpy oracle
in `bionpu/data/kmer_oracle.py` (T2's `count_kmers_canonical` /
`unpack_dna_2bit` / `pack_dna_2bit`) using the pre-built packed-2-bit
fixtures from `tracks/genomics/fixtures/` (T3).

Two test tiers:

* **Tier 1 (fast, no silicon)** — runs every CI invocation. Asserts
  the smoke fixture's sha256 matches its sidecar (guards against
  fixture corruption) and that T2's numpy oracle on the smoke fixture
  produces the committed top-100 JSON byte-equal (guards against
  silent oracle drift).

* **Tier 2 (silicon-required, marked ``@pytest.mark.npu``)** — runs
  the NPU op via `BionpuKmerCount.__call__` and asserts byte-equal
  against the numpy oracle on the smoke fixture (all k-mers, all
  three k values) and on chr22 (top-1000 only, since chr22 produces
  millions of unique k-mers). Skipped automatically when artifacts
  are absent. NPU dispatches are wrapped in
  :func:`bionpu.dispatch.npu_silicon_lock.npu_silicon_lock` per the
  CLAUDE.md silicon-mutex rule.

Reference patterns:
* `bionpu-public/tests/test_linear_projection_fused_correctness.py`
  for the silicon-skipped pattern.
* `bionpu-public/tests/test_kmer_oracle.py` (T2) for the numpy oracle
  + fixture-roundtrip pattern.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from bionpu.data.kmer_oracle import (
    count_kmers_canonical,
    unpack_dna_2bit,
)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

# Repo root: bionpu-public/tests/<file> -> bionpu-public/ -> repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FIXTURES = _REPO_ROOT / "tracks" / "genomics" / "fixtures"

_SMOKE_BIN = _FIXTURES / "smoke_10kbp.2bit.bin"
_SMOKE_SHA = _FIXTURES / "smoke_10kbp.2bit.bin.sha256"
_SMOKE_EXPECTED_K21_JSON = _FIXTURES / "smoke_10kbp_expected_k21.json"
_SMOKE_N_BASES = 10_000  # T3 fixture pinned size

_CHR22_BIN = _FIXTURES / "chr22.2bit.bin"
_CHR22_SHA = _FIXTURES / "chr22.2bit.bin.sha256"
_CHR22_N_BASES = 50_818_468  # per fixtures/README.md


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _expected_sha256_from_sidecar(sidecar: Path) -> str:
    """Parse the sha256sum-format sidecar and return the hex digest."""
    text = sidecar.read_text().strip()
    # sha256sum format: "<hex>  <basename>"
    digest = text.split()[0]
    return digest


def _load_packed_seq(path: Path, n_bases: int) -> tuple[np.ndarray, str]:
    """Read packed-2-bit binary and return (packed_bytes, ascii_seq).

    The ascii_seq is the unpacked ACGT string used to feed T2's
    ``count_kmers_canonical`` oracle.
    """
    buf = np.fromfile(path, dtype=np.uint8)
    seq = unpack_dna_2bit(buf, n_bases)
    return buf, seq


def _top_n_sorted(
    counts: dict[int, int], n: int
) -> list[tuple[int, int]]:
    """Sort by ``(count desc, canonical asc)`` and take top-N."""
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if n > 0:
        items = items[:n]
    return items


def _to_hex_dicts(
    pairs: list[tuple[int, int]],
) -> list[dict[str, object]]:
    """Convert ``[(canonical_u64, count), ...]`` into the JSON-shaped list."""
    return [
        {"canonical_u64_hex": f"0x{k:016x}", "count": int(v)} for k, v in pairs
    ]


# --------------------------------------------------------------------------- #
# Tier 1: fixture integrity + numpy-oracle regression lock (no silicon)
# --------------------------------------------------------------------------- #


def test_smoke_fixture_sha256() -> None:
    """Smoke fixture binary must match its checked-in sha256 sidecar.

    Guards against fixture corruption or accidental edits — if this
    test fails, regenerate via
    ``tracks/genomics/fixtures/build_kmer_fixtures.py``.
    """
    assert _SMOKE_BIN.exists(), f"missing smoke fixture: {_SMOKE_BIN}"
    assert _SMOKE_SHA.exists(), f"missing smoke sha256 sidecar: {_SMOKE_SHA}"
    actual = _sha256_of_file(_SMOKE_BIN)
    expected = _expected_sha256_from_sidecar(_SMOKE_SHA)
    assert actual == expected, (
        f"smoke fixture sha256 mismatch:\n"
        f"  actual:   {actual}\n"
        f"  expected: {expected} (from {_SMOKE_SHA.name})\n"
        f"Regenerate with tracks/genomics/fixtures/build_kmer_fixtures.py"
    )


def test_numpy_oracle_smoke_top100_locked() -> None:
    """T2's numpy oracle on the smoke fixture matches the committed top-100 JSON.

    This is the regression-guard: any future drift in the numpy oracle
    (`count_kmers_canonical`) or in the fixture would change the
    top-100 listing, which is then byte-comparable here. The committed
    JSON is generated as part of T12's setup by running the oracle on
    the smoke fixture and writing the top-100 entries sorted by
    ``(count desc, canonical asc)``.
    """
    assert _SMOKE_EXPECTED_K21_JSON.exists(), (
        f"missing committed expected-output JSON: "
        f"{_SMOKE_EXPECTED_K21_JSON}. T12 generates this by running "
        f"count_kmers_canonical on smoke_10kbp.2bit.bin at k=21 and "
        f"writing the top-100 by (count desc, canonical asc)."
    )
    expected = json.loads(_SMOKE_EXPECTED_K21_JSON.read_text())

    _, seq = _load_packed_seq(_SMOKE_BIN, _SMOKE_N_BASES)
    counts = count_kmers_canonical(seq, k=21)
    top100 = _top_n_sorted(counts, 100)
    actual = _to_hex_dicts(top100)

    assert actual == expected, (
        f"numpy oracle top-100 drift detected. "
        f"len(actual)={len(actual)}, len(expected)={len(expected)}. "
        f"First diff at: "
        f"{next((i, a, e) for i, (a, e) in enumerate(zip(actual, expected)) if a != e)}"
    )


# --------------------------------------------------------------------------- #
# Tier 2: silicon-byte-equal-vs-numpy-oracle (NPU-required)
# --------------------------------------------------------------------------- #


def _bionpu_kmer_count_op(k: int, n_tiles: int, n_passes: int = 4):
    """Construct a `BionpuKmerCount` op (kept lazy — import only when needed).

    Per ``state/kmer_count_interface_contract.md`` v0.5, the op constructor
    takes ``n_passes`` (hash-slice partition count) in addition to
    ``(k, n_tiles)``. The default ``n_passes=4`` matches the v0.5 build
    artifacts shipping in ``_npu_artifacts/...np4/``.
    """
    from bionpu.kernels.genomics.kmer_count import BionpuKmerCount

    return BionpuKmerCount(k=k, n_tiles=n_tiles, n_passes=n_passes)


def _skip_if_artifacts_missing(k: int, n_tiles: int, n_passes: int = 4) -> None:
    op = _bionpu_kmer_count_op(k, n_tiles, n_passes)
    if not op.artifacts_present():
        pytest.skip(
            f"NPU artifacts missing for "
            f"bionpu_kmer_count_k{k}_n{n_tiles}_np{n_passes}; "
            f"build via `make NPU2=1 K={k} n_passes={n_passes} "
            f"experiment={'production' if n_tiles == 1 else f'wide{n_tiles}'} all` "
            f"in `bionpu-public/src/bionpu/kernels/genomics/kmer_count/` "
            f"and copy outputs to {op.artifact_dir}/."
        )


@pytest.mark.npu
@pytest.mark.parametrize("k", [15, 21, 31])
def test_npu_smoke_byte_equal_to_numpy(k: int) -> None:
    """NPU full output on smoke fixture byte-equal to numpy oracle.

    For each k in {15, 21, 31} at n_tiles=4 the NPU dispatch must
    return EXACTLY the same ``(canonical_u64, count)`` listing as
    ``count_kmers_canonical``. No precision tolerance — this is the
    bit-exact gate from PRD section 3.2.

    Silicon dispatch is wrapped in ``npu_silicon_lock`` per the
    CLAUDE.md serialisation rule.
    """
    n_tiles = 4
    n_passes = 4
    _skip_if_artifacts_missing(k, n_tiles, n_passes)

    from bionpu.dispatch.npu_silicon_lock import npu_silicon_lock

    packed, seq = _load_packed_seq(_SMOKE_BIN, _SMOKE_N_BASES)

    # Numpy oracle reference — full mapping, all unique k-mers.
    ref_counts = count_kmers_canonical(seq, k=k)
    # Sort the reference the same way the NPU op + runner do
    # (count desc, canonical asc).
    ref_sorted = _top_n_sorted(ref_counts, 0)  # 0 = no top cap

    op = _bionpu_kmer_count_op(k, n_tiles, n_passes)
    with npu_silicon_lock(label=f"t12_kmer_test_k{k}"):
        npu_pairs = op(
            packed_seq=packed,
            top_n=0,         # no top cap; we want the FULL listing
            threshold=1,     # drop count-zero only (there shouldn't be any)
            n_iters=1,
            warmup=0,
        )

    if npu_pairs != ref_sorted:
        # Build a diagnostic message that does NOT crash on an empty
        # overlap (e.g. all-N silicon-output collapses to zero records,
        # in which case zip(...) yields no items).
        first_diff_or_none = next(
            (
                (i, a, b)
                for i, (a, b) in enumerate(zip(npu_pairs, ref_sorted))
                if a != b
            ),
            None,
        )
        first_diff = (
            "<no overlap-position diff; lengths differ at boundary>"
            if first_diff_or_none is None
            else first_diff_or_none
        )
        pytest.fail(
            f"k={k}: NPU output diverges from numpy oracle. "
            f"len(npu)={len(npu_pairs)}, len(ref)={len(ref_sorted)}. "
            f"First disagreement at: {first_diff}. "
            f"npu[:3]={npu_pairs[:3]}; ref[:3]={ref_sorted[:3]}."
        )


@pytest.mark.npu
def test_npu_chr22_top1000_byte_equal_at_k21() -> None:
    """NPU top-1000 on chr22 byte-equal to numpy oracle top-1000 at k=21.

    chr22 produces millions of unique k-mers, so we compare top-1000
    only — same gate the T14 jellyfish-ground-truth task uses. Skipped
    if the k=21, n_tiles=4 artifact is missing.
    """
    k = 21
    n_tiles = 4
    n_passes = 4
    _skip_if_artifacts_missing(k, n_tiles, n_passes)

    if not _CHR22_BIN.exists():
        pytest.skip(f"chr22 fixture missing: {_CHR22_BIN}")

    from bionpu.dispatch.npu_silicon_lock import npu_silicon_lock

    packed, seq = _load_packed_seq(_CHR22_BIN, _CHR22_N_BASES)

    # Numpy oracle on 50 Mbp is slow but tractable (~minutes).
    ref_counts = count_kmers_canonical(seq, k=k)
    ref_top1000 = _top_n_sorted(ref_counts, 1000)

    op = _bionpu_kmer_count_op(k, n_tiles, n_passes)
    with npu_silicon_lock(label="t12_kmer_test_chr22_k21"):
        npu_pairs = op(
            packed_seq=packed,
            top_n=1000,
            threshold=1,
            n_iters=1,
            warmup=0,
            timeout_s=300.0,  # 50 Mbp may take longer than the default
        )

    if npu_pairs != ref_top1000:
        first_diff_or_none = next(
            (
                (i, a, b)
                for i, (a, b) in enumerate(zip(npu_pairs, ref_top1000))
                if a != b
            ),
            None,
        )
        first_diff = (
            "<no overlap-position diff; lengths differ at boundary>"
            if first_diff_or_none is None
            else first_diff_or_none
        )
        pytest.fail(
            f"chr22 k=21 top-1000 NPU output diverges from numpy oracle. "
            f"len(npu)={len(npu_pairs)}, len(ref)={len(ref_top1000)}. "
            f"First disagreement at: {first_diff}. "
            f"npu[:3]={npu_pairs[:3]}; ref[:3]={ref_top1000[:3]}."
        )
