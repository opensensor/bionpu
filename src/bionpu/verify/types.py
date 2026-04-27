"""Public types for the byte-equality harness."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VerifyDivergence:
    """A single divergence between NPU and reference output.

    Attributes:
        record_index: 0-based index of the differing record (line in
            TSV, or read in FASTQ) in the canonicalised sort order.
        npu_bytes: Bytes of the differing record on the NPU side.
        ref_bytes: Bytes of the differing record on the reference side.
        message: Human-readable description of the difference.
    """

    record_index: int
    npu_bytes: bytes
    ref_bytes: bytes
    message: str


@dataclass(frozen=True)
class VerifyResult:
    """The outcome of a byte-equality comparison.

    Attributes:
        equal: True iff the NPU output is bit-identical to the reference
            after canonical normalisation (sort order + whitespace +
            line endings). False otherwise.
        npu_sha256: SHA-256 of the canonicalised NPU output.
        ref_sha256: SHA-256 of the canonicalised reference output.
        record_count: Number of records compared (after normalisation).
        divergences: First N divergences (empty if ``equal == True``).
            ``N`` defaults to 16 — see ``compare_against_*(max_divergences=N)``.
    """

    equal: bool
    npu_sha256: str
    ref_sha256: str
    record_count: int
    divergences: tuple[VerifyDivergence, ...] = field(default_factory=tuple)

    def __bool__(self) -> bool:
        return self.equal
