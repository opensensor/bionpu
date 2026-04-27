"""Byte-equality comparator for CRISPR off-target scan output.

Compares an NPU-emitted hits TSV against a cas-offinder reference TSV.
"""

from __future__ import annotations

from .types import VerifyResult


def compare_against_cas_offinder(
    npu_tsv: str | bytes,
    ref_tsv: str | bytes,
    *,
    max_divergences: int = 16,
) -> VerifyResult:
    """Compare an NPU off-target hits TSV against a cas-offinder reference.

    Args:
        npu_tsv: Path to the NPU-emitted hits TSV.
        ref_tsv: Path to the cas-offinder reference TSV.
        max_divergences: How many divergences to capture if the
            comparison fails. Default 16. Set to 0 to skip divergence
            collection (faster on large mismatching files).

    Returns:
        A :class:`VerifyResult` whose ``equal`` field is True iff the
        canonicalised TSVs are byte-identical. See
        ``src/bionpu/verify/README.md`` for the canonicalisation rules.
    """
    raise NotImplementedError(
        "bionpu.verify.crispr.compare_against_cas_offinder: shell only; "
        "implementation lands during the v0.1 extraction from genetics. "
        "See bionpu/verify/README.md for the public API contract."
    )
