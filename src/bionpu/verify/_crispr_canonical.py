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

"""Canonical Cas-OFFinder TSV normalizer.

Internal helper for :mod:`bionpu.verify.crispr`. Cas-OFFinder's row order
for sites at identical mismatch counts is implementation-defined: the
GPU/OpenCL backends can produce different row orderings of the same
match set. Byte-equality is therefore asserted against a *normalized*
canonical TSV produced by this module, with the sort key:

    (chrom, start, mismatch_count, guide_id, strand)

The TSV writer emits LF line endings and a single trailing newline so
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
    "write_tsv",
    "CANONICAL_HEADER",
]


_VALID_STRANDS = frozenset({"+", "-"})

# The canonical normalized TSV column header. Picking a single fixed
# layout means byte-equality holds across producers (Cas-OFFinder v3,
# legacy Cas-OFFinder, NumPy oracle, NPU runner).
CANONICAL_HEADER: tuple[str, ...] = (
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

    Field names match the canonical TSV header. ``bulge_type`` is ``"X"``
    for no-bulge runs (the typical case); other values appear only when
    DNA or RNA bulges are enabled.
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
                f"unknown strand {self.strand!r}; "
                f"expected one of {sorted(_VALID_STRANDS)}"
            )
        return (self.chrom, self.start, self.mismatches, self.guide_id, self.strand)


def normalize(rows: Iterable[CasOFFinderRow]) -> list[CasOFFinderRow]:
    """Return rows sorted by the documented canonical key.

    Idempotent: ``normalize(normalize(rows)) == normalize(rows)``.
    Independent of input order: sorting is stable and total over the key.
    """
    materialized = list(rows)
    for r in materialized:
        if r.strand not in _VALID_STRANDS:
            raise ValueError(
                f"unknown strand {r.strand!r}; "
                f"expected one of {sorted(_VALID_STRANDS)}"
            )
    return sorted(materialized, key=CasOFFinderRow.sort_key)


def parse_tsv(path: Path) -> list[CasOFFinderRow]:
    """Parse a Cas-OFFinder TSV into ``CasOFFinderRow`` objects.

    Header detection rules:
    - Lines starting with ``##`` are skipped (v3 generator banner).
    - A line starting with ``#`` is treated as the column header.
    - Otherwise the file is assumed to have no header (legacy form).
    - The canonical normalized TSV (this module's own output) starts
      with ``guide_id\\t...`` and is recognized too.

    Column mappings (case-insensitive on header tokens):

    - v3:        Id, Bulge Type, crRNA, DNA, Chromosome, Location,
                 Direction, Mismatches, Bulge Size
    - legacy:    crRNA, Chromosome, Position, DNA, Direction, Mismatches
    - canonical: guide_id, bulge_type, crrna, dna, chrom, start,
                 strand, mismatches, bulge_size
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
            header = line.lstrip("#").split("\t")
            continue
        if header is None and line.split("\t")[0].lower() == "guide_id":
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

        legacy_cols = ("crrna", "chromosome", "position", "dna", "direction", "mismatches")
        if has(*legacy_cols):
            for line in data_lines:
                cols = line.split("\t")
                rows.append(
                    CasOFFinderRow(
                        guide_id=cols[idx["crrna"]],  # crRNA stands in for guide_id
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
                f"unrecognized Cas-OFFinder TSV row "
                f"(no header, {len(cols)} cols): {line!r}"
            )
    return rows


def write_tsv(path: Path, rows: Iterable[CasOFFinderRow]) -> None:
    """Write rows to ``path`` in canonical schema with LF newlines.

    The resulting file is independent of producer line-ending choices,
    which is what makes byte-equality robust across platforms.
    """
    path = Path(path)
    parts: list[str] = ["\t".join(CANONICAL_HEADER)]
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
    """Read a Cas-OFFinder TSV, normalize, write to ``output_tsv``.

    The output TSV is byte-stable: re-running ``normalize_file`` on the
    output produces a byte-identical file.
    """
    rows = parse_tsv(Path(input_tsv))
    write_tsv(Path(output_tsv), normalize(rows))


def serialize_canonical(rows: Iterable[CasOFFinderRow]) -> bytes:
    """Return the canonical TSV byte representation of ``rows``.

    Equivalent to writing with :func:`write_tsv` and reading the result;
    used by the comparator to compute a SHA-256 without touching the
    filesystem.
    """
    parts: list[str] = ["\t".join(CANONICAL_HEADER)]
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
