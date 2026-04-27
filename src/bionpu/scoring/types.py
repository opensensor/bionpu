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

"""Canonical TSV schema for scored off-target candidates.

Extends the canonical scan TSV (from :mod:`bionpu.data.canonical_sites`)
with a single ``score`` column appended at the end. Identity columns
(everything before ``score``) preserve the byte-equality contract from
the scan stage; the ``score`` column is float-formatted with a fixed
precision so BITWISE_EXACT comparison is meaningful for deterministic
backends. NUMERIC_EPSILON comparison parses the float and tolerates
the per-row absolute deviation policy (see
:mod:`bionpu.verify.score`).

Score formatting policy
-----------------------

Scores are written as ``%.6f`` (six decimal places, fixed-point) by
default. Six places is enough to distinguish well-trained binary
classifiers (which typically commit to within 1e-4 of either bound)
without baking in spurious LSB noise from the GPU-vs-CPU floating-point
order. Backends that produce sub-1e-6 reproducibility can override
the format via ``write_score_tsv(..., score_format=...)``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from bionpu.data.canonical_sites import CasOFFinderRow

__all__ = [
    "ScoreRow",
    "parse_score_tsv",
    "serialize_canonical_score",
    "write_score_tsv",
]

# Header for the scored canonical TSV. Matches the canonical scan
# header plus a trailing ``score`` column.
_SCORE_HEADER = (
    "guide_id",
    "bulge_type",
    "crrna",
    "dna",
    "chrom",
    "start",
    "strand",
    "mismatches",
    "bulge_size",
    "score",
)

_DEFAULT_SCORE_FORMAT = "%.6f"


@dataclass(frozen=True, slots=True)
class ScoreRow:
    """One scored off-target candidate.

    Identity fields match :class:`bionpu.data.canonical_sites.CasOFFinderRow`
    exactly. The ``score`` field is the scorer's posterior off-target
    probability in ``[0, 1]``.
    """

    guide_id: str
    bulge_type: str
    crrna: str
    dna: str
    chrom: str
    start: int
    strand: str
    mismatches: int
    bulge_size: int
    score: float

    @classmethod
    def from_row(cls, row: CasOFFinderRow, score: float) -> "ScoreRow":
        """Promote a canonical scan row to a scored row."""
        return cls(
            guide_id=row.guide_id,
            bulge_type=row.bulge_type,
            crrna=row.crrna,
            dna=row.dna,
            chrom=row.chrom,
            start=row.start,
            strand=row.strand,
            mismatches=row.mismatches,
            bulge_size=row.bulge_size,
            score=float(score),
        )

    def identity_key(self) -> tuple[str, str, str, str, str, int, str, int, int]:
        """Tuple of identity fields (everything except ``score``).

        Used by :mod:`bionpu.verify.score` to align rows across two
        score-TSVs before comparing the score column.
        """
        return (
            self.guide_id,
            self.bulge_type,
            self.crrna,
            self.dna,
            self.chrom,
            self.start,
            self.strand,
            self.mismatches,
            self.bulge_size,
        )


def _format_row(r: ScoreRow, score_format: str) -> str:
    return "\t".join(
        (
            r.guide_id,
            r.bulge_type,
            r.crrna,
            r.dna,
            r.chrom,
            str(r.start),
            r.strand,
            str(r.mismatches),
            str(r.bulge_size),
            score_format % r.score,
        )
    )


def write_score_tsv(
    path: Path,
    rows: Iterable[ScoreRow],
    *,
    score_format: str = _DEFAULT_SCORE_FORMAT,
) -> None:
    """Write scored rows to ``path`` using the canonical schema + LF newlines.

    The output format is byte-stable when the input rows are
    pre-sorted by the canonical scan key — this module does NOT
    re-sort, since the upstream scan stage has already canonicalised
    the row order and the score column is identity-aligned to that
    order.
    """
    path = Path(path)
    parts: list[str] = ["\t".join(_SCORE_HEADER)]
    for r in rows:
        parts.append(_format_row(r, score_format))
    blob = "\n".join(parts) + "\n"
    path.write_bytes(blob.encode("utf-8"))


def serialize_canonical_score(
    rows: Iterable[ScoreRow],
    *,
    score_format: str = _DEFAULT_SCORE_FORMAT,
) -> bytes:
    """Return the canonical TSV byte representation of ``rows``."""
    parts: list[str] = ["\t".join(_SCORE_HEADER)]
    for r in rows:
        parts.append(_format_row(r, score_format))
    return ("\n".join(parts) + "\n").encode("utf-8")


def parse_score_tsv(path: Path) -> list[ScoreRow]:
    """Parse a scored canonical TSV produced by :func:`write_score_tsv`.

    Tolerates either the canonical 10-column header or a no-header
    file in the same column order. Reject any other shape.
    """
    text = Path(path).read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("##")]
    if not lines:
        return []

    rows: list[ScoreRow] = []
    start_idx = 0
    if lines[0].startswith("guide_id\t") or lines[0].startswith("#"):
        start_idx = 1

    for line in lines[start_idx:]:
        cols = line.split("\t")
        if len(cols) != len(_SCORE_HEADER):
            raise ValueError(
                f"unrecognized score-TSV row ({len(cols)} cols, expected "
                f"{len(_SCORE_HEADER)}): {line!r}"
            )
        rows.append(
            ScoreRow(
                guide_id=cols[0],
                bulge_type=cols[1],
                crrna=cols[2],
                dna=cols[3],
                chrom=cols[4],
                start=int(cols[5]),
                strand=cols[6],
                mismatches=int(cols[7]),
                bulge_size=int(cols[8]),
                score=float(cols[9]),
            )
        )
    return rows
