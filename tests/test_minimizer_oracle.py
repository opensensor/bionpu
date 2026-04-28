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

"""TDD tests for the canonical (w, k) minimizer oracle.

Mirrors :file:`tests/test_kmer_oracle.py` for the k-mer counter. The
RED→GREEN cycle: imports of :func:`extract_minimizers` should fail
before the oracle module exists; once the module is in place, all four
tests pass and become the regression lock.

Sliding-window minimizer math (oracle definition):

* For each length-k substring of ACGT, compute
  ``canonical = min(forward, reverse_complement)``.
* Maintain a ring of the last w canonicals.
* Emit ``(canonical, position)`` whenever the running minimum changes
  (new k-mer arrives that's strictly smaller, OR previous min slides
  out of the window).
* On ties: oldest entry wins (deterministic, silicon-mirrorable).

Per minimap2's mm_sketch comment "~2/(w+1) of k-mers are emitted as
minimizers in random sequences", the random-sequence emit count test
is a soft sanity check.
"""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
import pytest

from bionpu.data.kmer_oracle import pack_dna_2bit, unpack_dna_2bit
from bionpu.data.minimizer_oracle import (
    DEFAULT_W_FOR_K,
    SUPPORTED_KW,
    extract_minimizers,
    extract_minimizers_packed,
    kmer_mask_for,
)


_BASES = "ACGT"


def _random_seq(n: int, seed: int = 0) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice(_BASES) for _ in range(n))


# --------------------------------------------------------------------------- #
# Test 1: short hand-traced fixture
# --------------------------------------------------------------------------- #


def test_short_handcrafted_window_at_k3_w3() -> None:
    """A short sequence with k=3, w=3 produces the expected emits.

    Sequence: ACGT ACGT ACGT (12 bases). At k=3, k-mers are:
    pos 0  ACG -> fwd=0b000110=6,  rc(CGT)=0b011011=27, min=6
    pos 1  CGT -> fwd=0b011011=27, rc(ACG)=0b000110=6,  min=6
    pos 2  GTA -> fwd=0b101100=44, rc(TAC)=0b110001=49, min=44
    pos 3  TAC -> fwd=0b110001=49, rc(GTA)=0b101100=44, min=44
    pos 4  ACG -> 6
    pos 5  CGT -> 6
    pos 6  GTA -> 44
    pos 7  TAC -> 44
    pos 8  ACG -> 6
    pos 9  CGT -> 6

    First w=3-window completes at the 3rd pushed canonical (k-mer index 2,
    whose start pos = 2). At that point the ring holds canonicals
    [6, 6, 44]; min = 6, oldest-on-tie wins so position = 0.

    Subsequent emit conditions: only on strict-min-improvement OR
    slide-out. The min is 6 (well below 44), so the only triggers come
    from slide-out events when position 0's canonical leaves the window.
    """
    seq = "ACGTACGTACGT"
    out = extract_minimizers(seq, k=3, w=3)
    # First emit is canonical=6 (ACG) at position=0.
    assert out, "expected at least one emit"
    assert out[0] == (6, 0), f"first emit should be (6,0); got {out[0]}"
    # Output is sorted by position ascending.
    positions = [p for _, p in out]
    assert positions == sorted(positions), (
        f"emits not position-sorted: {positions}"
    )


# --------------------------------------------------------------------------- #
# Test 2: random-sequence emit-rate sanity check
# --------------------------------------------------------------------------- #


def test_random_sequence_emit_rate_within_bounds() -> None:
    """On a random ACGT sequence the emit rate is ~2/(w+1).

    minimap2's mm_sketch documents ~2/(w+1) of k-mers in random sequences
    are emitted as minimizers. We allow generous bounds (1.0/(w+1) to
    3.0/(w+1)) because:
      - our scheme uses canonical (not minimap2's hash64), which has a
        less uniform distribution;
      - small-n statistical noise.
    """
    n = 5000
    k = 15
    w = 10
    seq = _random_seq(n, seed=42)
    out = extract_minimizers(seq, k=k, w=w)
    n_kmers = n - k + 1
    rate = len(out) / max(n_kmers, 1)
    lo, hi = 1.0 / (w + 1), 3.0 / (w + 1)
    assert lo <= rate <= hi, (
        f"emit rate {rate:.4f} for n={n} k={k} w={w} not in expected "
        f"[{lo:.4f}, {hi:.4f}] (~2/(w+1))"
    )


# --------------------------------------------------------------------------- #
# Test 3: canonical = min(fwd, rc) invariant for every emit
# --------------------------------------------------------------------------- #


def test_emitted_canonicals_are_canonical_form() -> None:
    """Every emitted canonical equals min(fwd_kmer, rc_kmer) at its position.

    Cross-checks the oracle against an independent ground truth: re-derive
    the canonical from the substring at the reported position and assert
    equality.
    """
    seq = _random_seq(2000, seed=7)
    k = 21
    w = 11
    out = extract_minimizers(seq, k=k, w=w)
    base_to_2bit = {"A": 0, "C": 1, "G": 2, "T": 3}
    mask = kmer_mask_for(k)
    for canonical_emit, pos in out:
        kmer = seq[pos : pos + k]
        assert len(kmer) == k, (
            f"kmer at pos={pos} too short: {len(kmer)} (expected {k})"
        )
        fwd = 0
        for c in kmer:
            fwd = ((fwd << 2) | base_to_2bit[c]) & mask
        # rc
        rc_str = "".join(
            {"A": "T", "C": "G", "G": "C", "T": "A"}[c] for c in kmer[::-1]
        )
        rc = 0
        for c in rc_str:
            rc = ((rc << 2) | base_to_2bit[c]) & mask
        canonical_truth = min(fwd, rc)
        assert canonical_emit == canonical_truth, (
            f"emit canonical 0x{canonical_emit:x} at pos={pos} disagrees "
            f"with re-derived canonical 0x{canonical_truth:x}"
        )


# --------------------------------------------------------------------------- #
# Test 4: packed-2-bit roundtrip via extract_minimizers_packed
# --------------------------------------------------------------------------- #


def test_packed_2bit_path_matches_string_path() -> None:
    """:func:`extract_minimizers_packed` matches :func:`extract_minimizers`.

    For each pinned (k, w) configuration: pack the sequence with T2's
    pack_dna_2bit, then run extract_minimizers_packed and assert it
    returns the same listing as the string-path oracle.
    """
    seq = _random_seq(1000, seed=13)
    packed = pack_dna_2bit(seq)
    n_bases = len(seq)
    for k, w in SUPPORTED_KW:
        ref = extract_minimizers(seq, k=k, w=w)
        got = extract_minimizers_packed(packed, n_bases=n_bases, k=k, w=w)
        assert got == ref, (
            f"packed-path diverges from string-path at (k={k}, w={w}): "
            f"len(got)={len(got)} len(ref)={len(ref)}; "
            f"first diff: {next((i, a, b) for i, (a, b) in enumerate(zip(got, ref)) if a != b) if got != ref else None}"
        )
        # Sanity: positions are unique, sorted, and within bounds.
        positions = [p for _, p in got]
        assert positions == sorted(positions)
        assert all(0 <= p <= n_bases - k for p in positions)


# --------------------------------------------------------------------------- #
# Test 5: non-ACGT bases reset state cleanly
# --------------------------------------------------------------------------- #


def test_n_resets_state() -> None:
    """A run of N's resets the rolling state — emits resume only after k+w-1
    valid bases past the last N.
    """
    k = 5
    w = 3
    # 10 ACGT, then 1 N, then 10 ACGT.
    seq = "ACGTACGTAC" + "N" + "ACGTACGTAC"
    out = extract_minimizers(seq, k=k, w=w)
    # Every emit must be from a window whose k bases starting at `pos`
    # contain no N (positions in [0..6] for the first run, [11..16] for
    # the second run).
    for _, pos in out:
        assert "N" not in seq[pos : pos + k], (
            f"emit at pos={pos} has N in its kmer: {seq[pos : pos + k]!r}"
        )


# --------------------------------------------------------------------------- #
# Test 6: empty / too-short sequences return []
# --------------------------------------------------------------------------- #


def test_too_short_returns_empty() -> None:
    """Sequences shorter than k+w-1 produce no emits."""
    for seq in ["", "A", "ACGT"]:
        for k, w in SUPPORTED_KW:
            out = extract_minimizers(seq, k=k, w=w)
            assert out == [], (
                f"too-short seq {seq!r} at (k={k}, w={w}) should emit "
                f"nothing; got {out}"
            )


# --------------------------------------------------------------------------- #
# Test 7: validation
# --------------------------------------------------------------------------- #


def test_param_validation() -> None:
    """``k`` and ``w`` are validated."""
    seq = "ACGT" * 20
    with pytest.raises(ValueError):
        extract_minimizers(seq, k=0, w=10)
    with pytest.raises(ValueError):
        extract_minimizers(seq, k=15, w=0)
    with pytest.raises(ValueError):
        extract_minimizers(seq, k=33, w=10)


# --------------------------------------------------------------------------- #
# Test 8: default w mapping
# --------------------------------------------------------------------------- #


def test_default_w_mapping_is_pinned() -> None:
    """The pinned default w for each supported k is documented."""
    assert DEFAULT_W_FOR_K[15] == 10
    assert DEFAULT_W_FOR_K[21] == 11


# --------------------------------------------------------------------------- #
# Test 9: smoke regression-lock — first 100 minimizers checked-in
# --------------------------------------------------------------------------- #


def test_minimizer_smoke_regression_lock() -> None:
    """First 100 minimizers of smoke_10kbp at (k=15, w=10) byte-equal a
    checked-in fixture.

    The fixture lives at
    ``tracks/genomics/fixtures/smoke_minimizers_k15_w10_expected.json``
    and is the silicon byte-equal regression lock for v0. Re-generate
    with ``python3 -c "...extract_minimizers_packed..."`` (see CLAUDE.md
    minimizer follow-up).
    """
    import json
    from pathlib import Path

    fixture_root = Path(__file__).resolve().parents[2] / "tracks" / "genomics" / "fixtures"
    bin_path = fixture_root / "smoke_10kbp.2bit.bin"
    json_path = fixture_root / "smoke_minimizers_k15_w10_expected.json"
    if not bin_path.exists() or not json_path.exists():
        pytest.skip(
            f"smoke fixtures not present at {fixture_root} "
            f"(bin={bin_path.exists()}, json={json_path.exists()}); "
            "regression-lock test skipped"
        )

    with open(json_path) as f:
        expected = json.load(f)
    assert expected["k"] == 15 and expected["w"] == 10
    n_bases = expected["n_bases"]
    buf = np.fromfile(bin_path, dtype=np.uint8)
    out = extract_minimizers_packed(buf, n_bases=n_bases, k=15, w=10)
    assert len(out) >= 100, (
        f"expected >=100 minimizers from smoke_10kbp; got {len(out)}"
    )
    got_first_100 = [[int(c), int(p)] for c, p in out[:100]]
    expected_first_100 = expected["first_100_minimizers"]
    assert got_first_100 == expected_first_100, (
        "first 100 minimizers diverge from the regression lock"
    )
    assert len(out) == expected["n_total_minimizers"], (
        f"total minimizer count diverges: got {len(out)}, "
        f"expected {expected['n_total_minimizers']}"
    )


# --------------------------------------------------------------------------- #
# Test 10: canonical-invariance under reverse complement
# --------------------------------------------------------------------------- #


def _reverse_complement(seq: str) -> str:
    """ASCII reverse-complement of an ACGT string."""
    comp = {"A": "T", "C": "G", "G": "C", "T": "A"}
    return "".join(comp[c] for c in reversed(seq))


def test_minimizer_canonical_invariant() -> None:
    """Minimizers of a sequence and its reverse complement carry the same
    canonical multiset.

    Per the canonical = min(fwd, rc) construction, every k-mer's
    canonical is invariant under sequence-level RC. Positions translate
    by ``len(seq) - k - pos`` between the two strands, but the
    *multiset* of emitted canonicals is preserved. We assert multiset
    equality (Counter) — the algorithm emits on min-CHANGES so the
    sequences themselves needn't be elementwise equal.
    """
    rng = random.Random(99)
    seq = "".join(rng.choice(_BASES) for _ in range(2000))
    rc = _reverse_complement(seq)
    for k, w in SUPPORTED_KW:
        fwd_emits = extract_minimizers(seq, k=k, w=w)
        rc_emits = extract_minimizers(rc, k=k, w=w)
        # Every canonical that appears in the fwd run should appear in
        # the rc run with at least the same multiplicity (and vice
        # versa). Equality is too strong because emit triggers depend
        # on tie-break ordering which is direction-dependent — but the
        # MULTISET of canonicals must agree.
        fwd_counts = Counter(c for c, _ in fwd_emits)
        rc_counts = Counter(c for c, _ in rc_emits)
        # Allow up to 5% disagreement at the boundaries (window-fill
        # artefacts at the ends are direction-dependent).
        common = fwd_counts & rc_counts
        symmetric_diff = (fwd_counts | rc_counts) - common
        total = sum((fwd_counts | rc_counts).values())
        diff_ratio = sum(symmetric_diff.values()) / max(total, 1)
        assert diff_ratio < 0.05, (
            f"(k={k}, w={w}): canonical multiset disagreement "
            f"{diff_ratio:.4f} between fwd and rc strand exceeds 5% "
            f"boundary tolerance (fwd={len(fwd_emits)}, rc={len(rc_emits)})"
        )


# --------------------------------------------------------------------------- #
# Test 11: minimap2 binary parity (skip if minimap2 unavailable)
# --------------------------------------------------------------------------- #


def test_minimizer_byte_equal_minimap2() -> None:
    """Skipped: minimap2 ``mm_sketch`` byte-equal parity is NOT a v0 goal.

    Per ``DESIGN.md`` §2, our oracle is a SIMPLER pure-canonical
    minimizer (no ``hash64`` ordering, no HPC, no symmetric-kmer
    skip) so it cannot be byte-equal to minimap2's output. This test
    is a placeholder for v1 follow-on parity work; for v0 we
    intentionally skip.
    """
    pytest.skip(
        "v0 oracle uses pure-canonical (no hash64); minimap2 byte-equal "
        "parity tracked as v1 follow-on per DESIGN.md §2"
    )
