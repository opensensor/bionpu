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

"""FASTQ parsing and canonical byte-stream serialisation.

Internal helper for :mod:`bionpu.verify.basecalling`. The byte-equality
gate compares records sorted by ``read_id``, with CRLF normalised to LF
and a single trailing newline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "FastqRecord",
    "parse_fastq",
    "serialize_canonical",
]


@dataclass(frozen=True)
class FastqRecord:
    """One FASTQ record after parsing.

    The header is stored both raw (``header_line``, sans leading ``@``)
    and split (``read_id`` = first whitespace-delimited token).
    """

    read_id: str
    header_line: str  # full header after the '@', e.g. "uuid runid=..."
    sequence: str
    quality: str


def parse_fastq(path: Path) -> list[FastqRecord]:
    """Parse a 4-line-per-record FASTQ file.

    Tolerates trailing blank lines and CRLF line endings. Raises
    :class:`ValueError` if a record is malformed (sequence length must
    equal quality length; ``+`` separator must be present).
    """
    records: list[FastqRecord] = []
    text = Path(path).read_text(encoding="utf-8", errors="strict")
    lines = text.replace("\r\n", "\n").split("\n")
    while lines and lines[-1] == "":
        lines.pop()

    if len(lines) % 4 != 0:
        raise ValueError(
            f"FASTQ {path}: line count {len(lines)} not a multiple of 4"
        )

    for i in range(0, len(lines), 4):
        header, seq, sep, qual = lines[i : i + 4]
        if not header.startswith("@"):
            raise ValueError(
                f"FASTQ {path} record {i // 4}: header missing '@': {header!r}"
            )
        if not sep.startswith("+"):
            raise ValueError(
                f"FASTQ {path} record {i // 4}: separator missing '+': {sep!r}"
            )
        if len(seq) != len(qual):
            raise ValueError(
                f"FASTQ {path} record {i // 4}: "
                f"seq len {len(seq)} != qual len {len(qual)}"
            )
        header_line = header[1:]
        read_id = header_line.split(None, 1)[0] if header_line else ""
        records.append(
            FastqRecord(
                read_id=read_id,
                header_line=header_line,
                sequence=seq,
                quality=qual,
            )
        )
    return records


def serialize_canonical(records: list[FastqRecord]) -> bytes:
    """Canonicalise a list of records to a byte stream.

    Sorts by ``read_id`` (stable), normalises line endings to LF, and
    ends with a single trailing newline. Two FASTQ files are byte-equal
    iff their canonical serialisations are.
    """
    sorted_records = sorted(records, key=lambda r: r.read_id)
    out = []
    for r in sorted_records:
        out.append(f"@{r.header_line}\n{r.sequence}\n+\n{r.quality}\n")
    return "".join(out).encode("utf-8")
