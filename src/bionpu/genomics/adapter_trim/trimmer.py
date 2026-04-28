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

"""Core trim loop for ``bionpu trim`` — composition over primer_scan v0.

Algorithm (mirrors ``cutadapt -a ADAPTER --no-indels -e 0``):

1. For each FASTQ record:
   a. If the read contains any non-ACGT base (``N``, etc.), fall back
      to the CPU oracle path (silicon kernel only handles ACGT). The
      oracle's rolling-state semantics already match cutadapt's
      "non-ACGT resets the match" behaviour.
   b. Otherwise, pack to 2-bit MSB-first and dispatch
      :class:`BionpuPrimerScan`. Discard RC strand matches (cutadapt
      ``-a`` is forward-only).
2. If any forward-strand match exists, trim at the smallest match
   position (3'-most match in cutadapt parlance: the leftmost match
   in the read, since cutadapt -a removes from match start to end).
3. Emit the trimmed prefix (sequence and quality both truncated).

Reads SHORTER than the primer length cannot match — fast-path trivially.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Callable, Iterator

import numpy as np

from bionpu.data.kmer_oracle import pack_dna_2bit
from bionpu.data.primer_oracle import find_primer_matches

from .fastq import FastqRecord, open_fastq, parse_fastq, write_fastq

__all__ = [
    "TrimStats",
    "trim_fastq",
    "trim_records",
]


# Set used to test "is this read pure ACGT?"
_ACGT_SET = frozenset("ACGT")


@dataclass
class TrimStats:
    """Per-run trimming statistics (analogous to cutadapt's report)."""

    n_reads: int = 0
    n_trimmed: int = 0
    n_untrimmed: int = 0
    n_silicon_dispatches: int = 0
    n_cpu_fallback_reads: int = 0
    total_bases_in: int = 0
    total_bases_out: int = 0
    total_bases_removed: int = 0
    silicon_us: float = 0.0
    wall_s: float = 0.0
    trim_length_histogram: dict[int, int] = field(default_factory=dict)

    def record_trim(self, original_len: int, trimmed_len: int) -> None:
        self.n_reads += 1
        self.total_bases_in += original_len
        self.total_bases_out += trimmed_len
        removed = original_len - trimmed_len
        self.total_bases_removed += removed
        if removed > 0:
            self.n_trimmed += 1
            self.trim_length_histogram[removed] = (
                self.trim_length_histogram.get(removed, 0) + 1
            )
        else:
            self.n_untrimmed += 1


def _is_pure_acgt(seq: str) -> bool:
    """True iff every char of ``seq`` is one of A/C/G/T (uppercase)."""
    # frozenset membership beats a regex on small reads.
    return all(c in _ACGT_SET for c in seq)


def _trim_one_read_cpu(
    seq: str,
    primer: str,
) -> int | None:
    """CPU-oracle fallback. Returns the leftmost forward-strand match
    position, or ``None`` if no match."""
    matches = find_primer_matches(seq, primer, allow_rc=False)
    if not matches:
        return None
    # find_primer_matches returns sorted by (pos, strand). Forward is
    # strand=0; we already disabled RC. The first record is the leftmost.
    return matches[0][0]


def _trim_one_read_silicon(
    seq: str,
    primer: str,
    op,  # BionpuPrimerScan
) -> tuple[int | None, float]:
    """Silicon path. Returns ``(leftmost_fwd_match_pos | None, us_dispatch)``.

    Caller MUST ensure ``seq`` is pure ACGT.
    """
    # Pad the read to at least 1 base padding worth of length so the
    # silicon kernel doesn't reject for being too short.
    if len(seq) < op.p:
        # Short-circuit: cannot match by length.
        return None, 0.0
    packed = pack_dna_2bit(seq)
    t0 = time.perf_counter()
    matches = op(packed_seq=packed)
    dispatch_us = (time.perf_counter() - t0) * 1e6
    # Filter to forward strand (cutadapt -a semantic).
    fwd_positions = [pos for (pos, strand) in matches if strand == 0]
    if not fwd_positions:
        return None, dispatch_us
    return min(fwd_positions), dispatch_us


def trim_records(
    records: Iterator[FastqRecord],
    *,
    adapter: str,
    op=None,  # BionpuPrimerScan or None
    progress: Callable[[int], None] | None = None,
    stats: TrimStats | None = None,
) -> Iterator[FastqRecord]:
    """Trim 3' adapter from each FASTQ record.

    Args:
        records: iterable of input :class:`FastqRecord` (e.g. from
            :func:`parse_fastq`).
        adapter: ASCII ACGT adapter sequence. Length must match a
            supported P (see ``BionpuPrimerScan.SUPPORTED_P``).
        op: pre-instantiated :class:`BionpuPrimerScan` for silicon
            dispatch. If ``None``, all reads go through the CPU
            oracle (useful for testing or when no NPU is available).
        progress: optional callback invoked every 1000 reads with
            the running count.
        stats: optional :class:`TrimStats` to accumulate into.

    Yields:
        Trimmed :class:`FastqRecord` (sequence and quality truncated
        at the leftmost forward-strand match).
    """
    if not adapter:
        raise ValueError("trim_records: adapter cannot be empty")
    adapter_upper = adapter.upper()
    if not _is_pure_acgt(adapter_upper):
        raise ValueError(
            f"trim_records: adapter {adapter!r} contains non-ACGT bases; "
            f"only ACGT is supported in v0."
        )

    if stats is None:
        stats = TrimStats()

    for rec in records:
        seq = rec.seq.upper()
        # Cutadapt-compat: input is treated case-insensitive (ACGT only
        # in our v0 scope).
        original_len = len(seq)

        if original_len < len(adapter_upper):
            # No way to match: pass-through.
            stats.record_trim(original_len, original_len)
            yield rec
            if progress is not None and stats.n_reads % 1000 == 0:
                progress(stats.n_reads)
            continue

        if op is not None and _is_pure_acgt(seq):
            match_pos, dispatch_us = _trim_one_read_silicon(
                seq, adapter_upper, op
            )
            stats.n_silicon_dispatches += 1
            stats.silicon_us += dispatch_us
        else:
            match_pos = _trim_one_read_cpu(seq, adapter_upper)
            stats.n_cpu_fallback_reads += 1

        if match_pos is None:
            stats.record_trim(original_len, original_len)
            yield rec
        else:
            trimmed_seq = rec.seq[:match_pos]
            trimmed_qual = rec.qual[:match_pos]
            stats.record_trim(original_len, len(trimmed_seq))
            yield FastqRecord(
                header=rec.header,
                seq=trimmed_seq,
                qual=trimmed_qual,
            )

        if progress is not None and stats.n_reads % 1000 == 0:
            progress(stats.n_reads)


def trim_fastq(
    in_path: str | os.PathLike[str],
    out_path: str | os.PathLike[str],
    *,
    adapter: str,
    op=None,
    progress: Callable[[int], None] | None = None,
) -> TrimStats:
    """Process an entire FASTQ file: read, trim, write.

    Args:
        in_path: input FASTQ path (gzip auto-detected by ``.gz``).
        out_path: output FASTQ path (gzip auto-detected).
        adapter: ASCII ACGT adapter to trim.
        op: optional :class:`BionpuPrimerScan` instance. If ``None``,
            falls through to the CPU oracle for every read.
        progress: optional progress callback, invoked every 1000 reads.

    Returns:
        :class:`TrimStats` with per-run counters.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = TrimStats()
    t0 = time.perf_counter()
    with open_fastq(in_path, "r") as fin, open_fastq(out_path, "w") as fout:
        for trimmed in trim_records(
            parse_fastq(fin),
            adapter=adapter,
            op=op,
            progress=progress,
            stats=stats,
        ):
            write_fastq(fout, trimmed)
    stats.wall_s = time.perf_counter() - t0
    return stats
