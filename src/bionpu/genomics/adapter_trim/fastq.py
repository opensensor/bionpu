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

"""FASTQ parser/writer for ``bionpu trim``.

Strict 4-line-per-record FASTQ as understood by cutadapt and dnaio:

    @<header>
    <sequence>
    +[<optional repeat of header>]
    <quality string, same length as sequence>

* Gzip transparency via filename ``.gz`` suffix.
* Empty sequences (post-trim) are emitted as ``@h\\n\\n+\\n\\n`` —
  cutadapt does the same.
* Header text is preserved as-is (excluding the leading ``@``).
* Multi-line FASTQ is NOT supported. Multi-line sequences are
  exceedingly rare in modern FASTQ; cutadapt also requires 4-line.
"""

from __future__ import annotations

import gzip
import io
import os
from typing import IO, Iterator, NamedTuple

__all__ = [
    "FastqError",
    "FastqRecord",
    "open_fastq",
    "parse_fastq",
    "write_fastq",
]


class FastqError(ValueError):
    """Raised on malformed FASTQ input."""


class FastqRecord(NamedTuple):
    """Single FASTQ record.

    Attributes:
        header: Text after the leading ``@`` (no trailing newline).
        seq: Sequence string (no newlines).
        qual: Quality string (no newlines), same length as ``seq``.
    """

    header: str
    seq: str
    qual: str


def open_fastq(
    path: str | os.PathLike[str],
    mode: str = "r",
) -> IO[str]:
    """Open a FASTQ file for reading or writing.

    Gzip is auto-detected from the filename ``.gz`` suffix. Always
    opens in text mode; the caller iterates lines.

    Args:
        path: filename (str / os.PathLike).
        mode: ``"r"`` (read) or ``"w"`` (write).
    """
    if mode not in ("r", "w"):
        raise ValueError(f"open_fastq: mode must be 'r' or 'w'; got {mode!r}")
    p = os.fspath(path)
    if p.endswith(".gz"):
        # Open the gzip file in binary mode and wrap with a TextIOWrapper —
        # gzip.open's text mode is OK for our purposes; use it directly.
        return gzip.open(p, mode + "t", encoding="ascii", newline="\n")  # type: ignore[return-value]
    return open(p, mode, encoding="ascii", newline="\n")


def parse_fastq(stream: IO[str]) -> Iterator[FastqRecord]:
    """Iterate over FASTQ records from a text stream.

    Args:
        stream: Iterable of lines (e.g. an open file from
            :func:`open_fastq`).

    Yields:
        :class:`FastqRecord` per record.

    Raises:
        FastqError: malformed FASTQ (bad header, length mismatch,
        truncated record).
    """
    line_no = 0
    while True:
        # Header line (starts with @).
        h_line = stream.readline()
        if not h_line:
            return
        line_no += 1
        h = h_line.rstrip("\n").rstrip("\r")
        if not h:
            # Trailing empty lines are harmless; skip.
            continue
        if not h.startswith("@"):
            raise FastqError(
                f"line {line_no}: expected FASTQ header line starting with "
                f"'@'; got {h!r}"
            )
        header = h[1:]

        # Sequence line.
        s_line = stream.readline()
        if not s_line:
            raise FastqError(
                f"line {line_no + 1}: unexpected EOF after header {header!r} "
                f"(sequence line missing)"
            )
        line_no += 1
        seq = s_line.rstrip("\n").rstrip("\r")

        # Plus line.
        p_line = stream.readline()
        if not p_line:
            raise FastqError(
                f"line {line_no + 1}: unexpected EOF after sequence "
                f"(separator line '+' missing for header {header!r})"
            )
        line_no += 1
        plus = p_line.rstrip("\n").rstrip("\r")
        if not plus.startswith("+"):
            raise FastqError(
                f"line {line_no}: expected '+' separator; got {plus!r} "
                f"(record header {header!r})"
            )

        # Quality line.
        q_line = stream.readline()
        if not q_line:
            raise FastqError(
                f"line {line_no + 1}: unexpected EOF after separator "
                f"(quality line missing for header {header!r})"
            )
        line_no += 1
        qual = q_line.rstrip("\n").rstrip("\r")

        if len(seq) != len(qual):
            raise FastqError(
                f"record {header!r} (line {line_no}): sequence length "
                f"{len(seq)} != quality length {len(qual)}"
            )

        yield FastqRecord(header=header, seq=seq, qual=qual)


def write_fastq(stream: IO[str], record: FastqRecord) -> None:
    """Write a single FASTQ record to a text stream.

    Cutadapt-compatible 4-line format. Empty sequence/quality strings
    are emitted as empty lines (no special-casing).
    """
    if len(record.seq) != len(record.qual):
        raise FastqError(
            f"write_fastq: record {record.header!r} has seq length "
            f"{len(record.seq)} != qual length {len(record.qual)}"
        )
    stream.write(f"@{record.header}\n{record.seq}\n+\n{record.qual}\n")


def _resolve_text_stream(
    path_or_stream: str | os.PathLike[str] | IO[str],
    mode: str,
) -> tuple[IO[str], bool]:
    """Resolve a path to an opened text stream.

    Returns ``(stream, owned)`` — owned=True means we opened it and
    are responsible for closing.
    """
    if isinstance(path_or_stream, (str, os.PathLike)):
        return open_fastq(path_or_stream, mode), True
    if isinstance(path_or_stream, io.IOBase) or hasattr(path_or_stream, "read"):
        return path_or_stream, False  # type: ignore[return-value]
    raise TypeError(
        f"_resolve_text_stream: unsupported input type {type(path_or_stream)}"
    )
