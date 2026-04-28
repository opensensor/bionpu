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
    "trim_fastq_batched",
    "trim_records",
    "trim_records_batched",
]


# v1 batched-dispatch defaults.
# Sentinel = run of all-T bases between concatenated reads. The TruSeq P5
# adapter (and every other pinned 13/20/25-bp Illumina adapter we ship)
# contains zero T bases, so a 16-base run of T can never form an adapter
# match. For multi-adapter v2 work, the sentinel must be re-chosen to be
# disjoint from EVERY pinned adapter.
_SENTINEL_BASE = "T"
_SENTINEL_LEN = 16
_SENTINEL_STR = _SENTINEL_BASE * _SENTINEL_LEN

#: Default reads-per-dispatch for the batched silicon path.
#
# Sized to clear the v1 throughput gate (>=5K reads/s on the 10K fixture)
# given the measured ~102 ms per-dispatch floor. At B=1024:
#   * 10K-read sweep: ~5.9K reads/s (>= 5K gate; ~600x v0).
#   * 100-read sweep: 1 dispatch covers the whole input.
# B=4096 reaches ~11K reads/s but the silicon-time-per-dispatch starts
# climbing (124 us/dispatch); 1024 sits in the sweet spot. Override via
# the --batch-size CLI flag when tuning for stretch throughput.
DEFAULT_BATCH_SIZE: int = 1024


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
    """True iff every char of ``seq`` is one of A/C/G/T (uppercase).

    Uses ``str.translate`` to delete every ACGT — a non-empty result
    proves a non-ACGT base remains. Roughly an order of magnitude faster
    than a Python ``all(c in set ...)`` generator for ~150 bp reads
    (and significantly faster on multi-MB strings).
    """
    return not seq.translate(_NON_ACGT_DELETE_TABLE)


# Pre-built translate table that removes every A/C/G/T (upper). Built
# once at import time; thread-safe (immutable). Keys are unicode
# codepoints (matching ``str.translate``'s contract).
_NON_ACGT_DELETE_TABLE = str.maketrans("", "", "ACGT")


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


# --------------------------------------------------------------------------- #
# v1 — batched dispatch (Path B: sentinel-separated stream).
#
# Algorithm
# ---------
# 1. Buffer up to ``batch_size`` consecutive FASTQ records.
# 2. Partition the batch into:
#      * silicon batch: pure-ACGT reads of length >= P
#      * cpu batch: reads that fail either gate (CPU-oracle path)
#    Each input record's index is preserved so the output order matches
#    the input order verbatim.
# 3. For the silicon batch, build a single concatenated sequence:
#        read_0 + sentinel + read_1 + sentinel + ... + read_{K-1}
#    where the sentinel is ``_SENTINEL_STR`` (16 bp of all-T). Pack the
#    full string with ``pack_dna_2bit`` and dispatch ``op`` ONCE.
# 4. The kernel returns ``[(global_pos, strand), ...]``. Map each match
#    back to ``(read_idx, in_read_pos)`` using the per-read base offsets
#    we recorded in step 3. Drop:
#        * RC-strand matches (cutadapt -a is forward-only).
#        * Matches whose ``[global_pos, global_pos + P)`` interval is
#          NOT fully contained inside a single read body (i.e. they
#          straddle a sentinel — impossible-by-construction for an all-T
#          sentinel and a no-T adapter, but cheap to enforce as belt+
#          suspenders for v2).
# 5. For each silicon record, take the leftmost surviving match (if any)
#    as the trim position. CPU-batch reads use the existing oracle path.
# 6. Emit the trimmed records in original input order.
#
# Why Path B over Path A
# ----------------------
# Path A (per-read length-prefix header) requires kernel-side iteration
# over a K-read manifest, which would mean editing the locked
# primer_scan tile / IRON / runner. Path B is purely host-side: the
# kernel sees one big input and emits global-position records, which
# is exactly what it already does today. Zero kernel/IRON changes.
# --------------------------------------------------------------------------- #


# Vectorised ACGT -> 2-bit lookup.
#
# ``pack_dna_2bit`` from kmer_oracle is the canonical packer (Python loop;
# byte-equal wire-format reference). For batched silicon dispatch we hot-
# loop on the same op tens of times per second, so we ship a numpy-side
# vectorised reproduction here. Byte-equality is asserted by the v2 byte-
# equal regression tests; if those drift the test gates immediately.
_ACGT_LOOKUP = np.zeros(256, dtype=np.uint8)
_ACGT_LOOKUP[ord("A")] = 0
_ACGT_LOOKUP[ord("C")] = 1
_ACGT_LOOKUP[ord("G")] = 2
_ACGT_LOOKUP[ord("T")] = 3


def _pack_dna_2bit_vectorised(concat: str) -> np.ndarray:
    """Vectorised pack of an ACGT string to 2-bit MSB-first bytes.

    Byte-equal to :func:`bionpu.data.kmer_oracle.pack_dna_2bit`. ~30x
    faster on multi-MB strings (numpy bitops vs Python for-loop).

    Caller must guarantee ``concat`` is pure ACGT — non-ACGT inputs
    silently produce zero-padded output. The trimmer's pre-classification
    loop already filters to pure-ACGT before reaching this packer.
    """
    n = len(concat)
    if n == 0:
        return np.zeros(0, dtype=np.uint8)
    n_bytes = (n + 3) // 4
    # Pad to a multiple of 4 so we can reshape into rows of 4 bases.
    pad = (4 - (n % 4)) % 4
    if pad:
        # 'A' contributes a 0 nibble — same as the zero-padding the
        # pure-Python loop applies via np.zeros.
        concat = concat + ("A" * pad)
    arr = np.frombuffer(concat.encode("ascii"), dtype=np.uint8)
    codes = _ACGT_LOOKUP[arr]  # (n + pad,) uint8 with each value in 0..3
    rows = codes.reshape(-1, 4)
    # MSB-first: lane 0 -> bits[7:6], lane 1 -> [5:4], lane 2 -> [3:2], lane 3 -> [1:0].
    out = (
        (rows[:, 0] << 6)
        | (rows[:, 1] << 4)
        | (rows[:, 2] << 2)
        | (rows[:, 3])
    ).astype(np.uint8)
    return out[:n_bytes]


def _build_concat_silicon_input(
    seqs: list[str],
) -> tuple[np.ndarray, list[int], int]:
    """Build the concatenated 2-bit-packed input for a silicon batch.

    Args:
        seqs: list of pure-ACGT sequences. All entries must already be
            uppercase ACGT.

    Returns:
        ``(packed, read_base_offsets, total_concat_bases)`` where
        ``packed`` is the 2-bit MSB-first packed buffer for the
        concatenated sequence and ``read_base_offsets[i]`` is the global
        base offset (in the concatenated stream) at which read ``i``
        starts. ``total_concat_bases`` is the length of the concatenated
        sequence in bases; the post-last-read sentinel is omitted (the
        sentinel only sits BETWEEN reads).
    """
    parts: list[str] = []
    offsets: list[int] = []
    cursor = 0
    for i, s in enumerate(seqs):
        if i > 0:
            parts.append(_SENTINEL_STR)
            cursor += _SENTINEL_LEN
        offsets.append(cursor)
        parts.append(s)
        cursor += len(s)
    if not parts:
        return np.zeros(0, dtype=np.uint8), [], 0
    concat = "".join(parts)
    # Use the vectorised packer — byte-equal output to pack_dna_2bit but
    # ~30x faster. The byte-equal regression tests gate any drift.
    packed = _pack_dna_2bit_vectorised(concat)
    return packed, offsets, cursor


#: Per-call dispatch impl override for the silicon batched path.
#
# v2 default: ``"pyxrt"`` — in-process dispatch, ~3-5 ms per dispatch.
# Set ``BIONPU_TRIM_DISPATCH=subprocess`` (or pass ``_impl="subprocess"``
# in test fixtures) to revert to the v1 host_runner subprocess path
# (~100 ms per dispatch). The v2 in-process path is byte-equal to v1's
# subprocess path on the synthetic-100 fixture; see test
# ``test_v2_pyxrt_byte_equal_to_subprocess``.
_TRIM_DISPATCH_ENV: str = "BIONPU_TRIM_DISPATCH"


def _resolve_trim_impl() -> str:
    """Read the trim-side dispatch override from the env.

    Defaults to ``"pyxrt"`` (v2). Returns ``"subprocess"`` only if the
    env var is set explicitly to that value. Other values silently
    fall back to pyxrt (the safe default).
    """
    val = os.environ.get(_TRIM_DISPATCH_ENV, "").strip().lower()
    return "subprocess" if val == "subprocess" else "pyxrt"


def _silicon_batch_dispatch(
    seqs: list[str],
    primer: str,
    op,  # BionpuPrimerScan
) -> tuple[list[int | None], float]:
    """Dispatch ``op`` once on a concatenated silicon batch and return
    the per-read leftmost forward-strand match position (or ``None``).

    Default impl is the v2 in-process pyxrt path (BIONPU_TRIM_DISPATCH
    env var override; see :func:`_resolve_trim_impl`).

    Returns ``(match_positions, dispatch_us)`` where ``match_positions``
    is a list of length ``len(seqs)``.
    """
    if not seqs:
        return [], 0.0
    packed, offsets, _ = _build_concat_silicon_input(seqs)
    impl = _resolve_trim_impl()
    t0 = time.perf_counter()
    matches = op(packed_seq=packed, _impl=impl)
    dispatch_us = (time.perf_counter() - t0) * 1e6

    # Pre-compute per-read [start, end) intervals (inclusive of read body
    # only; sentinels are between consecutive reads). A match [pos, pos+P)
    # is valid for read i iff offsets[i] <= pos AND pos + P <= offsets[i] + len(seqs[i]).
    p = op.p
    read_starts = offsets
    read_ends = [offsets[i] + len(seqs[i]) for i in range(len(seqs))]

    # Initialise per-read leftmost match to None.
    leftmost: list[int | None] = [None] * len(seqs)

    # The kernel's matches are sorted (position asc, strand asc) per the
    # v0 wire-format pin. We can iterate them in order and binary-search
    # the read assignment, but linear bookkeeping is simpler and the
    # records list is bounded by MAX_EMIT_IDX (2046).
    n = len(seqs)
    cur_read = 0
    for pos, strand in matches:
        if strand != 0:
            # Forward-strand only (cutadapt -a parity).
            continue
        # Advance cur_read until pos falls within or after the cur_read body.
        while cur_read < n and pos >= read_ends[cur_read]:
            cur_read += 1
        if cur_read >= n:
            break
        if pos < read_starts[cur_read]:
            # Match starts inside a sentinel region — by-construction
            # impossible (sentinel is all-T, adapter has no T), but
            # filter anyway for v2 safety.
            continue
        # Match starts inside read cur_read; verify it ends inside too.
        if pos + p > read_ends[cur_read]:
            # Straddles into a sentinel; reject (also impossible for
            # T-only sentinel + T-free adapter, but kept as belt+brace).
            continue
        in_read_pos = pos - read_starts[cur_read]
        if leftmost[cur_read] is None or in_read_pos < leftmost[cur_read]:
            leftmost[cur_read] = in_read_pos
    return leftmost, dispatch_us


def trim_records_batched(
    records: Iterator[FastqRecord],
    *,
    adapter: str,
    op=None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress: Callable[[int], None] | None = None,
    stats: TrimStats | None = None,
) -> Iterator[FastqRecord]:
    """Batched-dispatch variant of :func:`trim_records`.

    Buffers ``batch_size`` records at a time, packs them into a single
    silicon dispatch via the sentinel-separated-stream strategy
    (Path B), and yields trimmed records in the original input order.

    When ``op is None`` (no NPU available), behaves identically to
    :func:`trim_records` (per-record CPU-oracle path; ``batch_size``
    is ignored).
    """
    if not adapter:
        raise ValueError("trim_records_batched: adapter cannot be empty")
    adapter_upper = adapter.upper()
    if not _is_pure_acgt(adapter_upper):
        raise ValueError(
            f"trim_records_batched: adapter {adapter!r} contains non-ACGT "
            f"bases; only ACGT is supported in v0."
        )
    if batch_size < 1:
        raise ValueError(
            f"trim_records_batched: batch_size={batch_size} must be >= 1"
        )

    if stats is None:
        stats = TrimStats()

    # If silicon is unavailable, fall straight through to the CPU
    # iterator (no batching benefit, and avoids double-bookkeeping).
    if op is None:
        yield from trim_records(
            records,
            adapter=adapter_upper,
            op=None,
            progress=progress,
            stats=stats,
        )
        return

    p_len = len(adapter_upper)
    batch: list[FastqRecord] = []

    def _flush() -> Iterator[FastqRecord]:
        # Per-batch processing. Two sub-batches (silicon vs cpu) keyed
        # by index in ``batch`` so we can re-order outputs to match the
        # input order.
        n = len(batch)
        if n == 0:
            return
        # Pre-classify and collect silicon inputs.
        silicon_idxs: list[int] = []
        silicon_seqs: list[str] = []
        # Per-record decision:
        #   match_pos[i] = leftmost forward-strand match position OR None
        #                  OR -1 sentinel for "pass-through (too short)".
        # We use Python None for "no match" and store an int otherwise.
        match_pos: list[int | None] = [None] * n
        too_short: list[bool] = [False] * n
        for i, rec in enumerate(batch):
            seq_upper = rec.seq.upper()
            if len(seq_upper) < p_len:
                too_short[i] = True
                continue
            if _is_pure_acgt(seq_upper):
                silicon_idxs.append(i)
                silicon_seqs.append(seq_upper)
            else:
                # CPU-oracle fallback (mirrors v0 per-record fallback).
                match_pos[i] = _trim_one_read_cpu(seq_upper, adapter_upper)
                stats.n_cpu_fallback_reads += 1

        # Single silicon dispatch for the whole sub-batch.
        if silicon_seqs:
            results, dispatch_us = _silicon_batch_dispatch(
                silicon_seqs, adapter_upper, op
            )
            stats.n_silicon_dispatches += 1
            stats.silicon_us += dispatch_us
            for sub_i, idx in enumerate(silicon_idxs):
                match_pos[idx] = results[sub_i]

        # Emit in input order.
        for i, rec in enumerate(batch):
            original_len = len(rec.seq)
            if too_short[i]:
                stats.record_trim(original_len, original_len)
                yield rec
            elif match_pos[i] is None:
                stats.record_trim(original_len, original_len)
                yield rec
            else:
                cut = match_pos[i]
                trimmed_seq = rec.seq[:cut]
                trimmed_qual = rec.qual[:cut]
                stats.record_trim(original_len, len(trimmed_seq))
                yield FastqRecord(
                    header=rec.header,
                    seq=trimmed_seq,
                    qual=trimmed_qual,
                )
            if progress is not None and stats.n_reads % 1000 == 0:
                progress(stats.n_reads)

    for rec in records:
        batch.append(rec)
        if len(batch) >= batch_size:
            yield from _flush()
            batch = []
    if batch:
        yield from _flush()


def trim_fastq_batched(
    in_path: str | os.PathLike[str],
    out_path: str | os.PathLike[str],
    *,
    adapter: str,
    op=None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress: Callable[[int], None] | None = None,
) -> TrimStats:
    """Batched-dispatch variant of :func:`trim_fastq`.

    Identical wire-format and trim semantics to :func:`trim_fastq`;
    differs only in that silicon dispatches process ``batch_size`` reads
    per ``op`` invocation. The output FASTQ is byte-equal to the
    unbatched path.

    See :func:`trim_records_batched` for algorithm details.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = TrimStats()
    t0 = time.perf_counter()
    with open_fastq(in_path, "r") as fin, open_fastq(out_path, "w") as fout:
        for trimmed in trim_records_batched(
            parse_fastq(fin),
            adapter=adapter,
            op=op,
            batch_size=batch_size,
            progress=progress,
            stats=stats,
        ):
            write_fastq(fout, trimmed)
    stats.wall_s = time.perf_counter() - t0
    return stats
