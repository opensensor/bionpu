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

"""Equivalence policies for scored off-target TSV outputs.

The byte-equality harness in :mod:`bionpu.verify.crispr` is the right
contract for the *scan* stage — chrom/start/strand are exact integer-
and string-valued and admit no tolerance. The *score* stage is
different: floating-point determinism across CPU vs GPU vs NPU is
real but not free (FMA fusion, GEMM order, mixed-precision policies)
and the right contract depends on the use case.

Two policies are exposed:

* **BITWISE_EXACT** — SHA-256 over the canonical TSV bytes; equivalent
  to the existing :func:`bionpu.verify.crispr.compare_against_cas_offinder`
  contract. Right answer when the producer is deterministic
  (e.g. CPU vs CPU, or NPU silicon-vs-host-emulation by construction).
  When passed two TSVs that came from different floating-point
  pipelines, expect divergence at the ULP.

* **NUMERIC_EPSILON** — identity columns must match byte-for-byte
  (chrom, start, strand, mismatches, sequence — the things the scan
  stage already underwrites). Score column tolerates absolute
  deviation ``|a - b| <= epsilon`` per row. Right answer when
  comparing across heterogeneous floating-point pipelines.

The policy is chosen by the caller; this module never picks a default
silently.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from bionpu.scoring.types import ScoreRow, parse_score_tsv, serialize_canonical_score

__all__ = [
    "Policy",
    "ScoreDivergence",
    "ScoreVerifyResult",
    "compare_score_outputs",
]


Policy = Literal["BITWISE_EXACT", "NUMERIC_EPSILON"]


@dataclass(frozen=True)
class ScoreDivergence:
    """A single divergence between two score TSVs.

    Identity-row mismatches (alignment failures — the row sets differ
    or sort order has drifted) are reported as record_index < 0 with
    a descriptive message; per-row score deviations carry the row
    index and the absolute difference.
    """

    record_index: int
    a_bytes: bytes
    b_bytes: bytes
    abs_diff: float | None
    message: str


@dataclass(frozen=True)
class ScoreVerifyResult:
    """Outcome of a score-TSV equivalence check.

    Attributes
    ----------
    equal:
        True iff the comparison passes under the chosen policy.
    policy:
        Which policy was applied.
    epsilon:
        Tolerance applied (``None`` for ``BITWISE_EXACT``).
    record_count:
        Number of rows compared.
    a_sha256, b_sha256:
        SHA-256 of each canonical TSV's bytes (filled regardless of
        policy — useful for triage even when NUMERIC_EPSILON passes).
    max_abs_diff:
        Largest ``|score_a - score_b|`` observed across rows. ``None``
        if no rows were compared. Reported regardless of policy so
        callers can monitor drift.
    divergences:
        First N divergences (empty if ``equal``). ``N`` defaults to
        16 — see ``compare_score_outputs(max_divergences=N)``.
    """

    equal: bool
    policy: Policy
    epsilon: float | None
    record_count: int
    a_sha256: str
    b_sha256: str
    max_abs_diff: float | None
    divergences: tuple[ScoreDivergence, ...] = field(default_factory=tuple)

    def __bool__(self) -> bool:
        return self.equal


def _sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _row_bytes(r: ScoreRow) -> bytes:
    """Serialise a single row to its canonical-TSV byte form (without header)."""
    blob = serialize_canonical_score([r])
    # Strip the header line — keep only the row line.
    _, _, body = blob.partition(b"\n")
    return body.rstrip(b"\n")


def _identity_match(a: ScoreRow, b: ScoreRow) -> bool:
    return a.identity_key() == b.identity_key()


def compare_score_outputs(
    a_tsv: Path,
    b_tsv: Path,
    *,
    policy: Policy,
    epsilon: float | None = None,
    max_divergences: int = 16,
) -> ScoreVerifyResult:
    """Compare two scored canonical TSVs under the named policy.

    Parameters
    ----------
    a_tsv, b_tsv:
        Paths to two score TSVs (canonical schema, see
        :mod:`bionpu.scoring.types`).
    policy:
        ``"BITWISE_EXACT"`` for SHA-256 byte equality; ``"NUMERIC_EPSILON"``
        for per-row identity equality plus score-column abs-tolerance.
    epsilon:
        Required when ``policy == "NUMERIC_EPSILON"``. Absolute
        per-row score tolerance. Must be ``>= 0``. Forbidden for
        ``BITWISE_EXACT``.
    max_divergences:
        Cap on reported divergences. The full count is implicit in the
        overall ``equal`` verdict.

    Returns
    -------
    :class:`ScoreVerifyResult`
        Boolean verdict, hashes, max observed score deviation, and the
        first N divergences if not equal.
    """
    if policy == "BITWISE_EXACT":
        if epsilon is not None:
            raise ValueError(
                "BITWISE_EXACT policy does not accept epsilon; "
                "did you mean NUMERIC_EPSILON?"
            )
    elif policy == "NUMERIC_EPSILON":
        if epsilon is None or epsilon < 0:
            raise ValueError(
                "NUMERIC_EPSILON policy requires epsilon >= 0"
            )
    else:
        raise ValueError(f"unknown policy {policy!r}")

    a_blob = Path(a_tsv).read_bytes()
    b_blob = Path(b_tsv).read_bytes()
    a_sha = _sha256_bytes(a_blob)
    b_sha = _sha256_bytes(b_blob)

    # BITWISE_EXACT: hash equality is the contract.
    if policy == "BITWISE_EXACT":
        equal = a_sha == b_sha
        return ScoreVerifyResult(
            equal=equal,
            policy=policy,
            epsilon=None,
            record_count=_row_count(a_blob),
            a_sha256=a_sha,
            b_sha256=b_sha,
            max_abs_diff=None,
            divergences=()
            if equal
            else (
                ScoreDivergence(
                    record_index=-1,
                    a_bytes=b"",
                    b_bytes=b"",
                    abs_diff=None,
                    message=(
                        f"sha256 mismatch: a={a_sha[:16]}..., "
                        f"b={b_sha[:16]}..."
                    ),
                ),
            ),
        )

    # NUMERIC_EPSILON: parse + compare row-aligned.
    a_rows = parse_score_tsv(Path(a_tsv))
    b_rows = parse_score_tsv(Path(b_tsv))
    return _compare_numeric_epsilon(
        a_rows, b_rows,
        a_sha=a_sha, b_sha=b_sha,
        epsilon=float(epsilon),
        max_divergences=max_divergences,
    )


def _row_count(blob: bytes) -> int:
    # Subtract 1 for the header line; clamp at 0.
    n = max(0, blob.count(b"\n") - 1)
    # If the file does not end with \n, the last row still counts.
    if blob and not blob.endswith(b"\n"):
        n += 1 if blob.count(b"\n") >= 1 else 0
    return n


def _compare_numeric_epsilon(
    a: Sequence[ScoreRow],
    b: Sequence[ScoreRow],
    *,
    a_sha: str,
    b_sha: str,
    epsilon: float,
    max_divergences: int,
) -> ScoreVerifyResult:
    divergences: list[ScoreDivergence] = []
    max_diff = None

    if len(a) != len(b):
        divergences.append(
            ScoreDivergence(
                record_index=-1,
                a_bytes=b"",
                b_bytes=b"",
                abs_diff=None,
                message=(
                    f"row count mismatch: a has {len(a)}, b has {len(b)}"
                ),
            )
        )
        return ScoreVerifyResult(
            equal=False,
            policy="NUMERIC_EPSILON",
            epsilon=epsilon,
            record_count=min(len(a), len(b)),
            a_sha256=a_sha,
            b_sha256=b_sha,
            max_abs_diff=max_diff,
            divergences=tuple(divergences),
        )

    n_records = len(a)
    for idx, (ra, rb) in enumerate(zip(a, b)):
        if not _identity_match(ra, rb):
            if len(divergences) < max_divergences:
                divergences.append(
                    ScoreDivergence(
                        record_index=idx,
                        a_bytes=_row_bytes(ra),
                        b_bytes=_row_bytes(rb),
                        abs_diff=None,
                        message=(
                            f"identity-column mismatch at row {idx}: "
                            f"a.key={ra.identity_key()!r} vs "
                            f"b.key={rb.identity_key()!r}"
                        ),
                    )
                )
            continue
        diff = abs(ra.score - rb.score)
        if max_diff is None or diff > max_diff:
            max_diff = diff
        if diff > epsilon and len(divergences) < max_divergences:
            divergences.append(
                ScoreDivergence(
                    record_index=idx,
                    a_bytes=_row_bytes(ra),
                    b_bytes=_row_bytes(rb),
                    abs_diff=diff,
                    message=(
                        f"score deviation at row {idx}: "
                        f"|{ra.score} - {rb.score}| = {diff} > eps={epsilon}"
                    ),
                )
            )

    equal = not divergences
    return ScoreVerifyResult(
        equal=equal,
        policy="NUMERIC_EPSILON",
        epsilon=epsilon,
        record_count=n_records,
        a_sha256=a_sha,
        b_sha256=b_sha,
        max_abs_diff=max_diff,
        divergences=tuple(divergences),
    )
