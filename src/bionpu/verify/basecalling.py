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

"""Byte-equality comparator for basecaller FASTQ output.

Compares an NPU-emitted FASTQ against a Dorado reference FASTQ. See
:mod:`bionpu.verify` for the public-API contract and the README in
this directory for the canonicalisation rules.
"""

from __future__ import annotations

import hashlib
from os import PathLike
from pathlib import Path

from . import _fastq
from .types import VerifyDivergence, VerifyResult

__all__ = ["compare_against_dorado"]


def _sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def compare_against_dorado(
    npu_fastq: str | bytes | PathLike,
    ref_fastq: str | bytes | PathLike,
    *,
    max_divergences: int = 16,
) -> VerifyResult:
    """Compare an NPU-emitted FASTQ against a Dorado reference FASTQ.

    Both files are parsed, sorted by ``read_id``, and serialised in the
    canonical wire format (LF line endings, single trailing newline).
    Two outputs are byte-equal iff their canonical serialisations are.

    Args:
        npu_fastq: Path to the NPU-emitted FASTQ.
        ref_fastq: Path to the Dorado reference FASTQ.
        max_divergences: How many divergences to capture if the
            comparison fails. Default 16. Set to 0 to skip divergence
            collection (faster on large mismatching files).

    Returns:
        A :class:`VerifyResult` with the boolean verdict and SHA-256s.
        On divergence, the first ``max_divergences`` differing reads
        are included with their canonical positions.

    Raises:
        FileNotFoundError: Either path does not exist.
        ValueError: A FASTQ cannot be parsed (line count not a multiple
            of 4, missing ``@`` / ``+`` markers, seq/qual length
            mismatch).
    """
    npu_path = Path(npu_fastq)
    ref_path = Path(ref_fastq)

    npu_records = _fastq.parse_fastq(npu_path)
    ref_records = _fastq.parse_fastq(ref_path)

    npu_blob = _fastq.serialize_canonical(npu_records)
    ref_blob = _fastq.serialize_canonical(ref_records)

    npu_sha = _sha256(npu_blob)
    ref_sha = _sha256(ref_blob)

    equal = npu_blob == ref_blob

    divergences: tuple[VerifyDivergence, ...]
    if equal or max_divergences <= 0:
        divergences = ()
    else:
        divergences = _collect_divergences(
            sorted(npu_records, key=lambda r: r.read_id),
            sorted(ref_records, key=lambda r: r.read_id),
            limit=max_divergences,
        )

    record_count = max(len(npu_records), len(ref_records))
    return VerifyResult(
        equal=equal,
        npu_sha256=npu_sha,
        ref_sha256=ref_sha,
        record_count=record_count,
        divergences=divergences,
    )


def _record_to_canonical_bytes(rec: _fastq.FastqRecord) -> bytes:
    return f"@{rec.header_line}\n{rec.sequence}\n+\n{rec.quality}\n".encode("utf-8")


def _collect_divergences(
    npu_records: list[_fastq.FastqRecord],
    ref_records: list[_fastq.FastqRecord],
    *,
    limit: int,
) -> tuple[VerifyDivergence, ...]:
    """Walk both record lists in canonical order and report the first
    ``limit`` differing positions.
    """
    out: list[VerifyDivergence] = []
    n = max(len(npu_records), len(ref_records))
    for i in range(n):
        if len(out) >= limit:
            break
        npu_r = npu_records[i] if i < len(npu_records) else None
        ref_r = ref_records[i] if i < len(ref_records) else None
        if npu_r is None:
            out.append(
                VerifyDivergence(
                    record_index=i,
                    npu_bytes=b"",
                    ref_bytes=_record_to_canonical_bytes(ref_r),  # type: ignore[arg-type]
                    message=f"read {i}: NPU output ends before reference",
                )
            )
            continue
        if ref_r is None:
            out.append(
                VerifyDivergence(
                    record_index=i,
                    npu_bytes=_record_to_canonical_bytes(npu_r),
                    ref_bytes=b"",
                    message=f"read {i}: reference ends before NPU output",
                )
            )
            continue
        npu_bytes = _record_to_canonical_bytes(npu_r)
        ref_bytes = _record_to_canonical_bytes(ref_r)
        if npu_bytes != ref_bytes:
            # Specific field-level diagnostic to make the divergence
            # actionable (read_id mismatch is a different kind of bug
            # from sequence mismatch).
            if npu_r.read_id != ref_r.read_id:
                msg = (
                    f"read {i}: read_id mismatch "
                    f"(NPU={npu_r.read_id!r} vs ref={ref_r.read_id!r})"
                )
            elif npu_r.sequence != ref_r.sequence:
                msg = (
                    f"read {i} ({npu_r.read_id}): sequence differs "
                    f"(NPU len={len(npu_r.sequence)}, "
                    f"ref len={len(ref_r.sequence)})"
                )
            elif npu_r.quality != ref_r.quality:
                msg = (
                    f"read {i} ({npu_r.read_id}): quality string differs"
                )
            else:
                msg = (
                    f"read {i} ({npu_r.read_id}): header line differs "
                    f"(NPU={npu_r.header_line!r} vs ref={ref_r.header_line!r})"
                )
            out.append(
                VerifyDivergence(
                    record_index=i,
                    npu_bytes=npu_bytes,
                    ref_bytes=ref_bytes,
                    message=msg,
                )
            )
    return tuple(out)
