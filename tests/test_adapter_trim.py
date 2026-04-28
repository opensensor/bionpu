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

"""Unit + silicon tests for ``bionpu trim`` (adapter_trim v0).

Coverage:

* FASTQ parser/writer round-trip and edge cases.
* Trim-loop semantics (CPU oracle path) vs cutadapt -a parity.
* Silicon byte-equal vs CPU oracle (skipped if no NPU artifacts).
"""

from __future__ import annotations

import gzip
import io
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from bionpu.genomics.adapter_trim import (
    FastqRecord,
    TrimStats,
    open_fastq,
    parse_fastq,
    trim_fastq,
    trim_fastq_batched,
    write_fastq,
)
from bionpu.genomics.adapter_trim.fastq import FastqError
from bionpu.genomics.adapter_trim.trimmer import (
    trim_records,
    trim_records_batched,
)


ADAPTER = "AGATCGGAAGAGC"  # TruSeq P5; len=13 (P=13)


_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[2] / "tracks" / "genomics" / "fixtures"
)


def _find_cutadapt() -> str | None:
    """Locate cutadapt either on PATH or alongside the running interpreter.

    bionpu's CI runs in the xdna-bringup ironenv where cutadapt is
    installed in the venv's ``bin/`` but not necessarily on ``$PATH``
    when pytest is invoked outside the venv's activation script.
    """
    on_path = shutil.which("cutadapt")
    if on_path is not None:
        return on_path
    sibling = Path(sys.executable).parent / "cutadapt"
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    return None


# --------------------------------------------------------------------------- #
# FASTQ parser tests
# --------------------------------------------------------------------------- #


def test_fastq_round_trip_basic():
    """parse_fastq + write_fastq round-trip preserves all 4 lines."""
    src = "@read1\nACGT\n+\nIIII\n@read2\nGGGGGGGG\n+\nJJJJJJJJ\n"
    recs = list(parse_fastq(io.StringIO(src)))
    assert len(recs) == 2
    assert recs[0] == FastqRecord("read1", "ACGT", "IIII")
    assert recs[1] == FastqRecord("read2", "GGGGGGGG", "JJJJJJJJ")

    buf = io.StringIO()
    for r in recs:
        write_fastq(buf, r)
    assert buf.getvalue() == src


def test_fastq_parser_rejects_length_mismatch():
    """Sequence/quality length mismatch is a hard error."""
    src = "@read1\nACGT\n+\nIII\n"  # qual is 3 chars, seq is 4
    with pytest.raises(FastqError, match="sequence length 4 != quality length 3"):
        list(parse_fastq(io.StringIO(src)))


def test_fastq_parser_rejects_missing_header_at():
    """Header line without leading '@' is a hard error."""
    src = "read1\nACGT\n+\nIIII\n"
    with pytest.raises(FastqError, match="expected FASTQ header"):
        list(parse_fastq(io.StringIO(src)))


def test_fastq_parser_truncated_record():
    """Truncated record (no quality line) is a hard error."""
    src = "@read1\nACGT\n+\n"
    with pytest.raises(FastqError, match="quality line missing"):
        list(parse_fastq(io.StringIO(src)))


def test_fastq_parser_handles_empty_input():
    """Empty input yields no records (no error)."""
    recs = list(parse_fastq(io.StringIO("")))
    assert recs == []


def test_fastq_parser_preserves_header_text():
    """Multi-token headers are passed through verbatim."""
    src = "@read1 description goes here\nACGT\n+\nIIII\n"
    recs = list(parse_fastq(io.StringIO(src)))
    assert recs[0].header == "read1 description goes here"


def test_fastq_gzip_round_trip(tmp_path: Path):
    """open_fastq transparently handles .gz suffix."""
    rec = FastqRecord("read1", "ACGTACGT", "IIIIIIII")
    path = tmp_path / "test.fastq.gz"
    with open_fastq(path, "w") as f:
        write_fastq(f, rec)
    # Verify it's actually gzipped on disk.
    with gzip.open(path, "rt", encoding="ascii") as f:
        text = f.read()
    assert text == "@read1\nACGTACGT\n+\nIIIIIIII\n"
    # Round-trip through open_fastq.
    with open_fastq(path, "r") as f:
        round = list(parse_fastq(f))
    assert round == [rec]


# --------------------------------------------------------------------------- #
# Trim-loop CPU semantics tests
# --------------------------------------------------------------------------- #


def test_trim_records_cpu_no_match_pass_through():
    """Reads with no adapter are passed through unchanged."""
    recs = [FastqRecord("r1", "AAAAAAAA", "IIIIIIII")]
    out = list(trim_records(iter(recs), adapter=ADAPTER))
    assert out == recs


def test_trim_records_cpu_full_match_drop_to_empty():
    """Read starting with adapter trims to empty."""
    seq = ADAPTER + "GGGGGGGGGG"
    qual = "I" * len(seq)
    recs = [FastqRecord("r1", seq, qual)]
    out = list(trim_records(iter(recs), adapter=ADAPTER))
    assert out == [FastqRecord("r1", "", "")]


def test_trim_records_cpu_inserted_adapter():
    """Adapter at known position trims at that position."""
    seq = "AAAAAAA" + ADAPTER + "GGGGGGGG"  # adapter at position 7
    qual = "I" * len(seq)
    recs = [FastqRecord("r1", seq, qual)]
    out = list(trim_records(iter(recs), adapter=ADAPTER))
    assert out == [FastqRecord("r1", "AAAAAAA", "IIIIIII")]


def test_trim_records_cpu_short_read_pass_through():
    """Read shorter than adapter cannot match; pass through."""
    short = "ACGT"  # 4 < 13
    recs = [FastqRecord("r1", short, "IIII")]
    out = list(trim_records(iter(recs), adapter=ADAPTER))
    assert out == recs


def test_trim_records_cpu_n_bases_no_match_through_n():
    """Adapter cannot straddle an N base — read with N inside adapter
    region is NOT trimmed (cutadapt -a parity)."""
    # Place an N in the middle of where the adapter would otherwise match.
    seq = "AAAAAAA" + "AGATCGGNAGAGC" + "GGGGGGGG"  # N at position 14
    qual = "I" * len(seq)
    recs = [FastqRecord("r1", seq, qual)]
    out = list(trim_records(iter(recs), adapter=ADAPTER))
    # The N breaks the match — read should be untrimmed.
    assert out[0].seq == seq
    assert out[0].qual == qual


def test_trim_records_first_match_wins():
    """When the adapter appears multiple times, the leftmost wins."""
    seq = "AA" + ADAPTER + "TT" + ADAPTER + "GG"
    qual = "I" * len(seq)
    recs = [FastqRecord("r1", seq, qual)]
    out = list(trim_records(iter(recs), adapter=ADAPTER))
    assert out[0].seq == "AA"  # trim at first match (pos 2)


def test_trim_stats_counters():
    """TrimStats accumulate correctly."""
    recs = [
        FastqRecord("r1", "AAAAAAAA", "IIIIIIII"),  # untrimmed
        FastqRecord("r2", ADAPTER + "GG", "I" * (len(ADAPTER) + 2)),  # trim to ""
        FastqRecord(
            "r3",
            "TTTT" + ADAPTER + "AAAA",
            "I" * (4 + len(ADAPTER) + 4),
        ),  # trim to TTTT
    ]
    stats = TrimStats()
    list(trim_records(iter(recs), adapter=ADAPTER, stats=stats))
    assert stats.n_reads == 3
    assert stats.n_trimmed == 2
    assert stats.n_untrimmed == 1
    assert stats.total_bases_in == 8 + 15 + 21
    assert stats.total_bases_out == 8 + 0 + 4
    assert stats.total_bases_removed == 15 + 17


def test_trim_records_invalid_adapter():
    """Non-ACGT adapters are rejected."""
    with pytest.raises(ValueError, match="non-ACGT"):
        list(trim_records(iter([]), adapter="ACGN"))


def test_trim_records_empty_adapter():
    """Empty adapter is rejected."""
    with pytest.raises(ValueError, match="adapter cannot be empty"):
        list(trim_records(iter([]), adapter=""))


# --------------------------------------------------------------------------- #
# trim_fastq end-to-end (CPU)
# --------------------------------------------------------------------------- #


def test_trim_fastq_synthetic_fixture_cpu_byte_equal_cutadapt(tmp_path: Path):
    """Synthetic 100-read fixture: bionpu trim CPU == cutadapt -a -O 13."""
    in_path = _FIXTURE_ROOT / "synthetic_reads_with_adapters.fastq"
    if not in_path.exists():
        pytest.skip(f"fixture missing: {in_path}")

    cutadapt = _find_cutadapt()
    if cutadapt is None:
        pytest.skip("cutadapt not on PATH; skipping cross-check")

    bionpu_out = tmp_path / "bionpu.fastq"
    cutadapt_out = tmp_path / "cutadapt.fastq"

    # bionpu trim (CPU oracle path: op=None).
    stats = trim_fastq(in_path, bionpu_out, adapter=ADAPTER, op=None)
    assert stats.n_reads == 100
    assert stats.n_trimmed == 50  # 50 reads have the adapter injected

    # cutadapt with full-overlap exact-match (parity for our exact-only kernel).
    proc = subprocess.run(
        [
            cutadapt,
            "-a", ADAPTER,
            "--no-indels", "-e", "0", "-O", str(len(ADAPTER)),
            "-j", "1",
            "-o", str(cutadapt_out),
            str(in_path),
        ],
        capture_output=True, text=True, check=True,
    )
    assert "no errors" in proc.stderr.lower() or proc.returncode == 0

    assert bionpu_out.read_bytes() == cutadapt_out.read_bytes(), (
        "bionpu trim CPU output differs from cutadapt -a -O 13 reference"
    )


def test_trim_fastq_gzip_round_trip(tmp_path: Path):
    """trim_fastq handles gzip input + gzip output."""
    in_gz = tmp_path / "in.fastq.gz"
    with gzip.open(in_gz, "wt", encoding="ascii") as f:
        f.write(f"@r1\n{'A' * 20}{ADAPTER}TTTT\n+\n{'I' * (20 + len(ADAPTER) + 4)}\n")

    out_gz = tmp_path / "out.fastq.gz"
    stats = trim_fastq(in_gz, out_gz, adapter=ADAPTER, op=None)
    assert stats.n_reads == 1
    assert stats.n_trimmed == 1

    with gzip.open(out_gz, "rt", encoding="ascii") as f:
        text = f.read()
    assert text == f"@r1\n{'A' * 20}\n+\n{'I' * 20}\n"


# --------------------------------------------------------------------------- #
# Silicon integration tests (skipped if no NPU artifacts).
# --------------------------------------------------------------------------- #


def _maybe_skip_if_no_npu():
    """Skip silicon tests if the primer_scan_p13_n4 artifact is missing."""
    try:
        from bionpu.kernels.genomics.primer_scan import BionpuPrimerScan
    except ImportError as exc:
        pytest.skip(f"primer_scan import failed: {exc}")
    op = BionpuPrimerScan(primer=ADAPTER, n_tiles=4)
    if not op.artifacts_present():
        pytest.skip(
            f"NPU artifacts missing for {op.name} at {op.artifact_dir}"
        )
    return op


def test_trim_fastq_silicon_byte_equal_cpu(tmp_path: Path):
    """Silicon path produces identical output to CPU oracle on the
    100-read synthetic fixture.

    This is the v0 silicon-vs-oracle byte-equal acceptance check —
    skipped automatically if the NPU artifact is absent.
    """
    in_path = _FIXTURE_ROOT / "synthetic_reads_with_adapters.fastq"
    if not in_path.exists():
        pytest.skip(f"fixture missing: {in_path}")
    op = _maybe_skip_if_no_npu()

    cpu_out = tmp_path / "cpu.fastq"
    npu_out = tmp_path / "npu.fastq"

    trim_fastq(in_path, cpu_out, adapter=ADAPTER, op=None)
    trim_fastq(in_path, npu_out, adapter=ADAPTER, op=op)

    assert cpu_out.read_bytes() == npu_out.read_bytes(), (
        "silicon (BionpuPrimerScan) output differs from CPU oracle on "
        "synthetic_reads_with_adapters.fastq"
    )


# --------------------------------------------------------------------------- #
# v1 — batched dispatch byte-equal regression (CPU path; deterministic
# without silicon. With op=None the batched code path falls through to
# the unbatched iterator — but we exercise the *with-op* code path via a
# synthetic mock op-class that replays the CPU oracle on the
# concatenated stream so we still test the sentinel-separation logic.)
# --------------------------------------------------------------------------- #


class _OracleMockOp:
    """Mock BionpuPrimerScan that replays the CPU oracle on the
    concatenated 2-bit-packed input. Used to test the batched
    sentinel-separation logic deterministically without silicon.
    """

    def __init__(self, primer: str):
        self.primer = primer.upper()
        self.p = len(self.primer)

    def __call__(self, *, packed_seq, **_kw):
        from bionpu.data.kmer_oracle import unpack_dna_2bit
        from bionpu.data.primer_oracle import find_primer_matches

        n_bases = packed_seq.size * 4  # uint8 packed; 4 bases / byte
        seq = unpack_dna_2bit(packed_seq, n_bases)
        # find_primer_matches returns (pos, strand) sorted by (pos, strand).
        return find_primer_matches(seq, self.primer, allow_rc=False)


def test_trim_records_batched_byte_equal_cpu_unbatched():
    """Batched path with mock-oracle op produces identical output to
    the unbatched CPU oracle path on the same input stream.
    """
    in_path = _FIXTURE_ROOT / "synthetic_reads_with_adapters.fastq"
    if not in_path.exists():
        pytest.skip(f"fixture missing: {in_path}")
    op = _OracleMockOp(ADAPTER)
    with open_fastq(in_path, "r") as f:
        unbatched = list(trim_records(parse_fastq(f), adapter=ADAPTER, op=None))
    with open_fastq(in_path, "r") as f:
        batched = list(
            trim_records_batched(
                parse_fastq(f), adapter=ADAPTER, op=op, batch_size=64
            )
        )
    assert unbatched == batched, (
        "batched-dispatch mock-oracle output differs from unbatched CPU "
        "oracle on synthetic_reads_with_adapters.fastq"
    )


def test_trim_records_batched_no_op_is_unbatched_passthrough():
    """When op is None the batched API behaves exactly like the
    unbatched API (CPU oracle for every record).
    """
    recs = [
        FastqRecord("r1", "AAAAAAAA", "IIIIIIII"),
        FastqRecord("r2", ADAPTER + "GGGG", "I" * (len(ADAPTER) + 4)),
        FastqRecord("r3", "TTTT" + ADAPTER + "CC", "I" * (4 + len(ADAPTER) + 2)),
    ]
    a = list(trim_records(iter(recs), adapter=ADAPTER, op=None))
    b = list(
        trim_records_batched(iter(recs), adapter=ADAPTER, op=None, batch_size=8)
    )
    assert a == b


@pytest.mark.parametrize("batch_size", [1, 8, 32, 64, 128])
def test_trim_records_batched_various_batch_sizes_consistent(batch_size):
    """Different batch_size values produce identical output (no
    boundary effects from the sentinel-separation strategy).
    """
    in_path = _FIXTURE_ROOT / "synthetic_reads_with_adapters.fastq"
    if not in_path.exists():
        pytest.skip(f"fixture missing: {in_path}")
    op = _OracleMockOp(ADAPTER)
    with open_fastq(in_path, "r") as f:
        baseline = list(trim_records(parse_fastq(f), adapter=ADAPTER, op=None))
    with open_fastq(in_path, "r") as f:
        out = list(
            trim_records_batched(
                parse_fastq(f),
                adapter=ADAPTER,
                op=op,
                batch_size=batch_size,
            )
        )
    assert out == baseline, (
        f"batched output at batch_size={batch_size} diverged from CPU baseline"
    )


def test_trim_fastq_batched_byte_equal_unbatched_cpu(tmp_path: Path):
    """trim_fastq_batched(op=None) produces a FASTQ file byte-equal to
    trim_fastq(op=None) on the synthetic-100 fixture.
    """
    in_path = _FIXTURE_ROOT / "synthetic_reads_with_adapters.fastq"
    if not in_path.exists():
        pytest.skip(f"fixture missing: {in_path}")
    out_v0 = tmp_path / "v0.fastq"
    out_v1 = tmp_path / "v1.fastq"
    trim_fastq(in_path, out_v0, adapter=ADAPTER, op=None)
    trim_fastq_batched(
        in_path, out_v1, adapter=ADAPTER, op=None, batch_size=64
    )
    assert out_v0.read_bytes() == out_v1.read_bytes()


def test_trim_records_batched_sentinel_no_false_positives():
    """Sentinel design: a read ending in a partial T-run that
    concatenates with the next read's start cannot create a spurious
    forward-strand match for the all-T sentinel boundary.

    This is a belt+brace test for the Path-B sentinel choice. Build two
    reads that, naively concatenated, would create the adapter
    AGATCGGAAGAGC straddling a junction — but since the sentinel is 16
    bases of T, the adapter (which has no T) cannot survive
    sentinel-spanning at all. We assert that no spurious matches appear
    at boundary positions.
    """
    # Two reads, each 13+ bp pure ACGT, no internal adapter. The end of
    # read_a + start of read_b (without the sentinel) WOULD form the
    # adapter — but with the 16-T sentinel between them, the kernel can
    # only see [...end_of_a]TTTTTTTTTTTTTTTT[start_of_b...] and the
    # adapter has no T, so no match is possible across the boundary.
    read_a = "GGGG" + "AGATCGGAA"  # last 9 bp = first 9 of adapter
    read_b = "GAGC" + "GGGGGGGG"   # first 4 = last 4 of adapter
    op = _OracleMockOp(ADAPTER)
    qa = "I" * len(read_a)
    qb = "I" * len(read_b)
    recs = [
        FastqRecord("a", read_a, qa),
        FastqRecord("b", read_b, qb),
    ]
    out = list(
        trim_records_batched(
            iter(recs), adapter=ADAPTER, op=op, batch_size=8
        )
    )
    # Neither read contains the adapter; both pass through unchanged.
    assert out == recs


# --------------------------------------------------------------------------- #
# Silicon batched-dispatch tests (skipped if no NPU artifacts).
# --------------------------------------------------------------------------- #


def test_trim_fastq_batched_silicon_byte_equal_unbatched(tmp_path: Path):
    """v1 batched silicon dispatch produces identical output to v0
    unbatched silicon dispatch on the synthetic-100 fixture (HARD GATE).
    """
    in_path = _FIXTURE_ROOT / "synthetic_reads_with_adapters.fastq"
    if not in_path.exists():
        pytest.skip(f"fixture missing: {in_path}")
    op = _maybe_skip_if_no_npu()

    v0_out = tmp_path / "v0.fastq"
    v1_out = tmp_path / "v1.fastq"

    trim_fastq(in_path, v0_out, adapter=ADAPTER, op=op)
    trim_fastq_batched(
        in_path, v1_out, adapter=ADAPTER, op=op, batch_size=64
    )

    assert v0_out.read_bytes() == v1_out.read_bytes(), (
        "v1 batched silicon output differs from v0 unbatched silicon on "
        "synthetic_reads_with_adapters.fastq"
    )


def test_trim_fastq_batched_silicon_byte_equal_cutadapt(tmp_path: Path):
    """v1 batched silicon dispatch matches cutadapt -a -O 13 byte-equally."""
    in_path = _FIXTURE_ROOT / "synthetic_reads_with_adapters.fastq"
    if not in_path.exists():
        pytest.skip(f"fixture missing: {in_path}")
    cutadapt = _find_cutadapt()
    if cutadapt is None:
        pytest.skip("cutadapt not on PATH; skipping cross-check")
    op = _maybe_skip_if_no_npu()

    v1_out = tmp_path / "v1.fastq"
    cutadapt_out = tmp_path / "cutadapt.fastq"

    trim_fastq_batched(
        in_path, v1_out, adapter=ADAPTER, op=op, batch_size=64
    )

    subprocess.run(
        [
            cutadapt,
            "-a", ADAPTER,
            "--no-indels", "-e", "0", "-O", str(len(ADAPTER)),
            "-j", "1",
            "-o", str(cutadapt_out),
            str(in_path),
        ],
        capture_output=True, check=True,
    )

    assert v1_out.read_bytes() == cutadapt_out.read_bytes(), (
        "v1 batched silicon output differs from cutadapt -a -O 13"
    )


# --------------------------------------------------------------------------- #
# v2 — in-process pyxrt dispatch byte-equal regressions (skipped if no NPU).
# --------------------------------------------------------------------------- #


def test_v2_pyxrt_byte_equal_to_subprocess(tmp_path: Path):
    """v2 in-process pyxrt dispatch produces a FASTQ byte-equal to the
    v1 subprocess dispatch on the synthetic-100 fixture (HARD GATE).

    This is the v2 acceptance gate: if pyxrt diverges from subprocess by
    even a single byte, the new in-process BO staging or the parse logic
    is incorrect.

    Skipped when:
      * NPU artifacts are missing
      * the cross-process silicon lock is held by another process (so we
        can't run the subprocess path concurrently with pyxrt)
    """
    in_path = _FIXTURE_ROOT / "synthetic_reads_with_adapters.fastq"
    if not in_path.exists():
        pytest.skip(f"fixture missing: {in_path}")
    op = _maybe_skip_if_no_npu()

    pyxrt_out = tmp_path / "pyxrt.fastq"
    sub_out = tmp_path / "subprocess.fastq"

    # Force pyxrt path.
    monkey_env = os.environ.get("BIONPU_TRIM_DISPATCH")
    os.environ["BIONPU_TRIM_DISPATCH"] = "pyxrt"
    try:
        trim_fastq_batched(
            in_path, pyxrt_out, adapter=ADAPTER, op=op, batch_size=64,
        )
    finally:
        if monkey_env is None:
            os.environ.pop("BIONPU_TRIM_DISPATCH", None)
        else:
            os.environ["BIONPU_TRIM_DISPATCH"] = monkey_env

    # Subprocess path (a fresh op instance avoids reusing the cached
    # pyxrt BO state).
    from bionpu.kernels.genomics.primer_scan import BionpuPrimerScan
    sub_op = BionpuPrimerScan(primer=ADAPTER, n_tiles=4)
    monkey_env = os.environ.get("BIONPU_TRIM_DISPATCH")
    os.environ["BIONPU_TRIM_DISPATCH"] = "subprocess"
    try:
        try:
            trim_fastq_batched(
                in_path, sub_out, adapter=ADAPTER, op=sub_op,
                batch_size=64,
            )
        except Exception as exc:  # pragma: no cover — environment-dependent
            # The cross-process silicon lock can be held by an unrelated
            # job. Skip rather than fail in that case (same policy as
            # the v1 silicon test gates).
            msg = str(exc)
            if "lock" in msg.lower() or "LockTimeout" in msg:
                pytest.skip(
                    f"cross-process silicon lock unavailable: {exc}"
                )
            raise
    finally:
        if monkey_env is None:
            os.environ.pop("BIONPU_TRIM_DISPATCH", None)
        else:
            os.environ["BIONPU_TRIM_DISPATCH"] = monkey_env

    assert pyxrt_out.read_bytes() == sub_out.read_bytes(), (
        "v2 pyxrt output differs from v1 subprocess output on "
        "synthetic_reads_with_adapters.fastq"
    )


def test_v2_pyxrt_byte_equal_to_cutadapt(tmp_path: Path):
    """v2 pyxrt dispatch byte-equal to cutadapt -a -O 13 on the
    100-read synthetic fixture.

    Cross-check vs the production-engineering reference; closes the
    full silicon-vs-reference loop on the in-process path. Skipped when
    cutadapt is unavailable.
    """
    in_path = _FIXTURE_ROOT / "synthetic_reads_with_adapters.fastq"
    if not in_path.exists():
        pytest.skip(f"fixture missing: {in_path}")
    cutadapt = _find_cutadapt()
    if cutadapt is None:
        pytest.skip("cutadapt not on PATH; skipping cross-check")
    op = _maybe_skip_if_no_npu()

    pyxrt_out = tmp_path / "pyxrt.fastq"
    cutadapt_out = tmp_path / "cutadapt.fastq"

    monkey_env = os.environ.get("BIONPU_TRIM_DISPATCH")
    os.environ["BIONPU_TRIM_DISPATCH"] = "pyxrt"
    try:
        trim_fastq_batched(
            in_path, pyxrt_out, adapter=ADAPTER, op=op, batch_size=64,
        )
    finally:
        if monkey_env is None:
            os.environ.pop("BIONPU_TRIM_DISPATCH", None)
        else:
            os.environ["BIONPU_TRIM_DISPATCH"] = monkey_env

    subprocess.run(
        [
            cutadapt,
            "-a", ADAPTER,
            "--no-indels", "-e", "0", "-O", str(len(ADAPTER)),
            "-j", "1",
            "-o", str(cutadapt_out),
            str(in_path),
        ],
        capture_output=True, check=True,
    )

    assert pyxrt_out.read_bytes() == cutadapt_out.read_bytes(), (
        "v2 pyxrt output differs from cutadapt -a -O 13 on "
        "synthetic_reads_with_adapters.fastq"
    )


def test_v2_pyxrt_packer_byte_equal_to_pack_dna_2bit():
    """The v2 trimmer's vectorised packer must produce identical bytes
    to ``bionpu.data.kmer_oracle.pack_dna_2bit`` for every input.
    """
    from bionpu.data.kmer_oracle import pack_dna_2bit
    from bionpu.genomics.adapter_trim.trimmer import _pack_dna_2bit_vectorised

    cases = [
        "",
        "A", "C", "G", "T",
        "ACGT",
        "ACGTACGT",
        "ACGTAC",                # not aligned (6 chars -> 2 bytes)
        "AGATCGGAAGAGC",         # TruSeq P5
        "T" * 16,                # sentinel
        ("ACGT" * 100) + "A",    # 401 chars -> 101 bytes
        "AAAAA" + ("CGTA" * 50) + "GG",
    ]
    for case in cases:
        a = pack_dna_2bit(case)
        b = _pack_dna_2bit_vectorised(case)
        assert a.tobytes() == b.tobytes(), (
            f"vectorised packer drifted from pack_dna_2bit on input "
            f"len={len(case)}: {case[:20]!r}..."
        )


def test_trim_fastq_silicon_byte_equal_cutadapt(tmp_path: Path):
    """Silicon path matches cutadapt -a -O 13 byte-equally on the
    100-read synthetic fixture (full e2e validation)."""
    in_path = _FIXTURE_ROOT / "synthetic_reads_with_adapters.fastq"
    if not in_path.exists():
        pytest.skip(f"fixture missing: {in_path}")
    cutadapt = _find_cutadapt()
    if cutadapt is None:
        pytest.skip("cutadapt not on PATH; skipping cross-check")
    op = _maybe_skip_if_no_npu()

    npu_out = tmp_path / "npu.fastq"
    cutadapt_out = tmp_path / "cutadapt.fastq"

    trim_fastq(in_path, npu_out, adapter=ADAPTER, op=op)

    subprocess.run(
        [
            cutadapt,
            "-a", ADAPTER,
            "--no-indels", "-e", "0", "-O", str(len(ADAPTER)),
            "-j", "1",
            "-o", str(cutadapt_out),
            str(in_path),
        ],
        capture_output=True, check=True,
    )

    assert npu_out.read_bytes() == cutadapt_out.read_bytes()
