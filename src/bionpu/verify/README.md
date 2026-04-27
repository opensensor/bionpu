# `bionpu.verify` — byte-equality harness

The load-bearing contribution of `bionpu`. This module proves the NPU
output is **bit-for-bit identical** to a canonical CPU reference, not
just "approximately equal" or "within tolerance."

## Why byte-equality matters

Most NPU/GPU genomics demos report agreement against a CPU reference
as "max error < 1e-3" or "F1 score 0.99." For clinical workloads,
GLP-compliant pipelines, regulatory submissions, and any setting where
the bit-for-bit hash of an output is a step in the audit trail, that
form of agreement is not defensible. A regulator does not accept "the
NPU usually produces the same answer as the reference."

What `bionpu.verify` does instead: every benchmark run produces an
NPU output and a reference output; the harness canonicalises both
(sort order, whitespace, line endings, internal field representation)
and computes a SHA-256 of each. If the hashes match, the run passes.
If they don't, the harness emits the *first divergence* with byte
offsets so the failure is actionable rather than statistical.

## Public API

```python
from bionpu.verify.crispr import compare_against_cas_offinder
from bionpu.verify.basecalling import compare_against_dorado

# CRISPR off-target hits TSV
result = compare_against_cas_offinder(
    npu_tsv="results/chr22_npu.tsv",
    ref_tsv="reference/crispr/casoffinder-chr22-canonical.tsv",
    max_divergences=16,
)
assert result, f"NPU output diverges: {result.divergences[0].message}"

# Basecaller FASTQ
result = compare_against_dorado(
    npu_fastq="results/reads_npu.fastq",
    ref_fastq="reference/basecalling/reads_dorado_reference.fastq",
)
print(f"NPU SHA-256: {result.npu_sha256}")
print(f"Reference SHA-256: {result.ref_sha256}")
print(f"Records compared: {result.record_count}")
print(f"Equal: {result.equal}")
```

Both comparators return a :class:`VerifyResult` (see
[`types.py`](types.py)) — a frozen dataclass that supports
`bool(result)` for `assert`-friendly use.

## Canonicalisation rules

Two outputs are considered byte-equal if their canonicalisations are.
Canonicalisation rules per format:

### CRISPR TSV (`compare_against_cas_offinder`)

Records sorted by `(chromosome, position, strand)`. Whitespace
collapsed to single tabs. Lines terminated by `\n`. Trailing newline
required. Mismatches column expressed as a non-negative integer
(no signed representation drift).

### FASTQ (`compare_against_dorado`)

Records sorted by `read_id`. Lines terminated by `\n`. Quality scores
expressed as the canonical phred+33 ASCII range. Read IDs compared
verbatim (no whitespace stripping in headers). Sequence bases must
match exactly.

## When this passes vs when it doesn't

A passing comparison is a strong claim: every emitted hit / every
emitted read / every quality score is exactly what the reference
would have produced. A failing comparison is a useful claim: the
harness tells you which record diverged first and what the byte-level
difference was.

Approximate-match modes (Hamming distance, edit distance, F1) are
intentionally not exposed by this module. If you want approximate
agreement metrics, use any of the existing tools (BLAST, edlib,
sklearn metrics) — `bionpu.verify` is the *byte-equality* claim
specifically.
