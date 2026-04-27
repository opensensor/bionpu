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

"""Byte-equality comparator for CRISPR off-target scan output.

Compares an NPU-emitted hits TSV against a Cas-OFFinder reference TSV.
See :mod:`bionpu.verify` for the public-API contract and the README in
this directory for the canonicalisation rules.
"""

from __future__ import annotations

import hashlib
from os import PathLike
from pathlib import Path

from ..data import canonical_sites as _canon
from .types import VerifyDivergence, VerifyResult

__all__ = ["compare_against_cas_offinder"]


def _sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def compare_against_cas_offinder(
    npu_tsv: str | bytes | PathLike,
    ref_tsv: str | bytes | PathLike,
    *,
    max_divergences: int = 16,
) -> VerifyResult:
    """Compare an NPU off-target hits TSV against a Cas-OFFinder reference.

    Both inputs may be in any of the formats this module's parser
    accepts (Cas-OFFinder v3, legacy 6-column, or this module's own
    canonical format). Each is parsed, normalized via the canonical
    sort key, and serialized to the canonical wire format. Two outputs
    are byte-equal iff their canonical serializations are.

    Args:
        npu_tsv: Path to the NPU-emitted hits TSV.
        ref_tsv: Path to the Cas-OFFinder reference TSV.
        max_divergences: How many divergences to capture if the
            comparison fails. Default 16. Set to 0 to skip divergence
            collection (faster on large mismatching files).

    Returns:
        A :class:`VerifyResult` with the boolean verdict and SHA-256s.
        On divergence, the first ``max_divergences`` differing rows are
        included in ``divergences`` with their canonical sort positions.

    Raises:
        FileNotFoundError: Either path does not exist.
        ValueError: A TSV cannot be parsed (unrecognized header /
            unexpected column count / invalid strand).
    """
    npu_path = Path(npu_tsv)
    ref_path = Path(ref_tsv)

    npu_rows = _canon.normalize(_canon.parse_tsv(npu_path))
    ref_rows = _canon.normalize(_canon.parse_tsv(ref_path))

    npu_blob = _canon.serialize_canonical(npu_rows)
    ref_blob = _canon.serialize_canonical(ref_rows)

    npu_sha = _sha256(npu_blob)
    ref_sha = _sha256(ref_blob)

    equal = npu_blob == ref_blob

    divergences: tuple[VerifyDivergence, ...]
    if equal or max_divergences <= 0:
        divergences = ()
    else:
        divergences = _collect_divergences(
            npu_rows, ref_rows, limit=max_divergences
        )

    record_count = max(len(npu_rows), len(ref_rows))
    return VerifyResult(
        equal=equal,
        npu_sha256=npu_sha,
        ref_sha256=ref_sha,
        record_count=record_count,
        divergences=divergences,
    )


def _row_to_canonical_line(row: _canon.CasOFFinderRow) -> bytes:
    """Render one row in the canonical wire format (one TSV data line)."""
    return (
        "\t".join(
            (
                row.guide_id,
                row.bulge_type,
                row.crrna,
                row.dna,
                row.chrom,
                str(row.start),
                row.strand,
                str(row.mismatches),
                str(row.bulge_size),
            )
        )
        + "\n"
    ).encode("utf-8")


def _collect_divergences(
    npu_rows: list[_canon.CasOFFinderRow],
    ref_rows: list[_canon.CasOFFinderRow],
    *,
    limit: int,
) -> tuple[VerifyDivergence, ...]:
    """Walk both row lists in canonical order and report the first ``limit``
    differing positions.

    A divergence is one of:
    - extra row on the NPU side (no corresponding ref row at this index)
    - missing row on the NPU side (ref has a row this index doesn't)
    - both sides have a row at this index but the bytes differ
    """
    out: list[VerifyDivergence] = []
    n = max(len(npu_rows), len(ref_rows))
    for i in range(n):
        if len(out) >= limit:
            break
        npu_row = npu_rows[i] if i < len(npu_rows) else None
        ref_row = ref_rows[i] if i < len(ref_rows) else None
        if npu_row is None:
            out.append(
                VerifyDivergence(
                    record_index=i,
                    npu_bytes=b"",
                    ref_bytes=_row_to_canonical_line(ref_row),  # type: ignore[arg-type]
                    message=f"row {i}: NPU output ends before reference",
                )
            )
            continue
        if ref_row is None:
            out.append(
                VerifyDivergence(
                    record_index=i,
                    npu_bytes=_row_to_canonical_line(npu_row),
                    ref_bytes=b"",
                    message=f"row {i}: reference ends before NPU output",
                )
            )
            continue
        npu_bytes = _row_to_canonical_line(npu_row)
        ref_bytes = _row_to_canonical_line(ref_row)
        if npu_bytes != ref_bytes:
            out.append(
                VerifyDivergence(
                    record_index=i,
                    npu_bytes=npu_bytes,
                    ref_bytes=ref_bytes,
                    message=(
                        f"row {i}: differ at "
                        f"({npu_row.chrom}:{npu_row.start} {npu_row.strand}) "
                        f"vs ({ref_row.chrom}:{ref_row.start} {ref_row.strand})"
                    ),
                )
            )
    return tuple(out)
