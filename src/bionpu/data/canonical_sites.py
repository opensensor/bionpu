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

"""Canonical Cas-OFFinder normalizer.

Cas-OFFinder's row order for sites at identical mismatch counts is
implementation-defined: GPU runs of the same input on the same machine can
produce different row orderings (the row *set* is invariant; the order is
not). PRD §3.2 byte-equality is therefore asserted against a *normalized*
canonical TSV produced by this module.

The sort key is load-bearing for and byte-equality:

    (chrom, start, mismatch_count, guide_id, strand)

where:
- ``chrom`` is the contig name as emitted by Cas-OFFinder (e.g. ``"chr22"``).
- ``start`` is the 0-based site position (Cas-OFFinder calls this Location).
- ``mismatch_count`` is Cas-OFFinder's Mismatches column.
- ``guide_id`` is the stable ID assigned by FIXTURE-A — for v3 outputs we
  use Cas-OFFinder's leading Id column; for legacy outputs we fall back to
  the crRNA sequence as the ID surrogate (a stable string either way).
- ``strand`` is ``"+"`` or ``"-"`` (Cas-OFFinder's Direction column).

Sort is stable — calling ``normalize`` twice produces the same output (and
applying it to an already-normalized list is a no-op).

The TSV writer emits LF line endings and a single trailing newline so that
byte-equality holds independent of the producer's line-ending choice.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "CasOFFinderRow",
    "normalize",
    "normalize_file",
    "parse_tsv",
    "serialize_canonical",
    "write_tsv",
]

_VALID_STRANDS = frozenset({"+", "-"})

# Column header for the canonical normalized TSV. We pick a single fixed
# layout so byte-equality holds across producers (Cas-OFFinder v3, legacy
# Cas-OFFinder, NumPy oracle).
_HEADER = (
    "guide_id",
    "bulge_type",
    "crrna",
    "dna",
    "chrom",
    "start",
    "strand",
    "mismatches",
    "bulge_size",
)

@dataclass(frozen=True, slots=True)
class CasOFFinderRow:
    """One Cas-OFFinder match row, normalized into a fixed schema.

    Field names match the canonical TSV header. ``bulge_type`` is ``"X"`` for
    no-bulge runs (the FIXTURE-A regime); other values appear only when DNA
    or RNA bulges are enabled, which FIXTURE-A forbids.
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

    def sort_key(self) -> tuple[str, int, int, str, str]:
        if self.strand not in _VALID_STRANDS:
            raise ValueError(
                f"unknown strand {self.strand!r}; expected one of {sorted(_VALID_STRANDS)}"
            )
        return (self.chrom, self.start, self.mismatches, self.guide_id, self.strand)

def normalize(rows: Iterable[CasOFFinderRow]) -> list[CasOFFinderRow]:
    """Return rows sorted by the documented canonical key.

    Idempotent: ``normalize(normalize(rows)) == normalize(rows)``.
    Independent of input order: sorting is stable and total over the key.
    """
    materialized = list(rows)
    # Validate strands eagerly so a bad row fails the call rather than
    # silently being placed at an arbitrary position.
    for r in materialized:
        if r.strand not in _VALID_STRANDS:
            raise ValueError(
                f"unknown strand {r.strand!r}; expected one of {sorted(_VALID_STRANDS)}"
            )
    return sorted(materialized, key=CasOFFinderRow.sort_key)

def parse_tsv(path: Path) -> list[CasOFFinderRow]:
    """Parse a Cas-OFFinder TSV (v3 or legacy) into ``CasOFFinderRow`` objects.

    Header detection rules:
    - Lines starting with ``##`` are skipped (v3 generator banner).
    - A line starting with ``#`` is treated as the column header.
    - Otherwise the file is assumed to have no header (legacy form).
    - The canonical normalized TSV (this module's own output) starts with
      ``guide_id\\t...`` — that is recognized too.

    Column mappings (case-insensitive on header tokens):

        v3:        Id, Bulge Type, crRNA, DNA, Chromosome, Location, Direction,
                   Mismatches, Bulge Size
        legacy:    crRNA, Chromosome, Position, DNA, Direction, Mismatches
        canonical: guide_id, bulge_type, crrna, dna, chrom, start, strand,
                   mismatches, bulge_size
    """
    path = Path(path)
    raw_lines = path.read_text().splitlines()

    header: list[str] | None = None
    data_lines: list[str] = []
    for line in raw_lines:
        if not line:
            continue
        if line.startswith("##"):
            continue
        if line.startswith("#"):
            # Column header.
            header = line.lstrip("#").split("\t")
            continue
        if header is None and line.split("\t")[0].lower() == "guide_id":
            # Canonical normalized header (no leading '#').
            header = line.split("\t")
            continue
        data_lines.append(line)

    rows: list[CasOFFinderRow] = []

    if header is not None:
        idx = {h.strip().lower(): i for i, h in enumerate(header)}

        def has(*names: str) -> bool:
            return all(n in idx for n in names)

        v3_cols = (
            "id", "bulge type", "crrna", "dna", "chromosome",
            "location", "direction", "mismatches", "bulge size",
        )
        if has(*v3_cols):
            for line in data_lines:
                cols = line.split("\t")
                rows.append(
                    CasOFFinderRow(
                        guide_id=cols[idx["id"]],
                        bulge_type=cols[idx["bulge type"]],
                        crrna=cols[idx["crrna"]],
                        dna=cols[idx["dna"]],
                        chrom=cols[idx["chromosome"]],
                        start=int(cols[idx["location"]]),
                        strand=cols[idx["direction"]],
                        mismatches=int(cols[idx["mismatches"]]),
                        bulge_size=int(cols[idx["bulge size"]]),
                    )
                )
            return rows

        canonical_cols = (
            "guide_id", "bulge_type", "crrna", "dna", "chrom",
            "start", "strand", "mismatches", "bulge_size",
        )
        if has(*canonical_cols):
            for line in data_lines:
                cols = line.split("\t")
                rows.append(
                    CasOFFinderRow(
                        guide_id=cols[idx["guide_id"]],
                        bulge_type=cols[idx["bulge_type"]],
                        crrna=cols[idx["crrna"]],
                        dna=cols[idx["dna"]],
                        chrom=cols[idx["chrom"]],
                        start=int(cols[idx["start"]]),
                        strand=cols[idx["strand"]],
                        mismatches=int(cols[idx["mismatches"]]),
                        bulge_size=int(cols[idx["bulge_size"]]),
                    )
                )
            return rows

        if has("crrna", "chromosome", "position", "dna", "direction", "mismatches"):
            for line in data_lines:
                cols = line.split("\t")
                rows.append(
                    CasOFFinderRow(
                        guide_id=cols[idx["crrna"]],  # crRNA stands in for guide_id in legacy
                        bulge_type="X",
                        crrna=cols[idx["crrna"]],
                        dna=cols[idx["dna"]],
                        chrom=cols[idx["chromosome"]],
                        start=int(cols[idx["position"]]),
                        strand=cols[idx["direction"]],
                        mismatches=int(cols[idx["mismatches"]]),
                        bulge_size=0,
                    )
                )
            return rows

        raise ValueError(f"unrecognized Cas-OFFinder TSV header: {header!r}")

    # No header at all — treat as legacy 6-column.
    for line in data_lines:
        cols = line.split("\t")
        if len(cols) == 6:
            crrna, chrom, position, dna, direction, mismatches = cols
            rows.append(
                CasOFFinderRow(
                    guide_id=crrna,
                    bulge_type="X",
                    crrna=crrna,
                    dna=dna,
                    chrom=chrom,
                    start=int(position),
                    strand=direction,
                    mismatches=int(mismatches),
                    bulge_size=0,
                )
            )
        else:
            raise ValueError(
                f"unrecognized Cas-OFFinder TSV row (no header, {len(cols)} cols): {line!r}"
            )
    return rows

def write_tsv(path: Path, rows: Iterable[CasOFFinderRow]) -> None:
    """Write rows to ``path`` using the canonical schema with LF newlines.

    The resulting file is independent of producer line-ending choices,
    which matters for byte-equality across platforms.
    """
    path = Path(path)
    parts: list[str] = ["\t".join(_HEADER)]
    for r in rows:
        parts.append(
            "\t".join(
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
                )
            )
        )
    blob = "\n".join(parts) + "\n"
    path.write_bytes(blob.encode("utf-8"))

def normalize_file(input_tsv: Path, output_tsv: Path) -> None:
    """Read a Cas-OFFinder TSV (v3, legacy, or canonical), normalize, write.

    The output TSV is byte-stable: ``normalize_file(out, out2)`` produces
    ``out2`` byte-identical to ``out`` for any already-normalized ``out``.
    """
    rows = parse_tsv(Path(input_tsv))
    write_tsv(Path(output_tsv), normalize(rows))


def serialize_canonical(rows: Iterable[CasOFFinderRow]) -> bytes:
    """Return the canonical TSV byte representation of ``rows``.

    Equivalent to writing with :func:`write_tsv` and reading the result
    back as bytes; used by :mod:`bionpu.verify.crispr` to compute a
    SHA-256 over the canonical form without touching the filesystem.
    """
    parts: list[str] = ["\t".join(_HEADER)]
    for r in rows:
        parts.append(
            "\t".join(
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
                )
            )
        )
    blob = "\n".join(parts) + "\n"
    return blob.encode("utf-8")
