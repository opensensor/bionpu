"""Byte-equality comparator for basecaller FASTQ output.

Compares an NPU-emitted FASTQ against a Dorado reference FASTQ.
"""

from __future__ import annotations

from .types import VerifyResult


def compare_against_dorado(
    npu_fastq: str | bytes,
    ref_fastq: str | bytes,
    *,
    max_divergences: int = 16,
) -> VerifyResult:
    """Compare an NPU-emitted FASTQ against a Dorado reference FASTQ.

    Args:
        npu_fastq: Path to the NPU-emitted FASTQ.
        ref_fastq: Path to the Dorado reference FASTQ.
        max_divergences: How many divergences to capture if the
            comparison fails. Default 16. Set to 0 to skip divergence
            collection (faster on large mismatching files).

    Returns:
        A :class:`VerifyResult` whose ``equal`` field is True iff the
        canonicalised FASTQs are byte-identical. See
        ``src/bionpu/verify/README.md`` for the canonicalisation rules.
    """
    raise NotImplementedError(
        "bionpu.verify.basecalling.compare_against_dorado: shell only; "
        "implementation lands during the v0.1 extraction from genetics. "
        "See bionpu/verify/README.md for the public API contract."
    )
