"""bionpu.verify: byte-equality harness for NPU output vs canonical reference.

The headline contribution of this package. Most "X on the GPU/NPU"
genomics demos compare against an approximate reference and report
"good enough." For clinical / GLP / regulated workloads, "good enough"
is not a defensible claim. This module exposes a documented public
API that takes an NPU output and a reference output and proves
*byte-equality* (or surfaces the exact divergence if not).

Public surface
--------------

The package exposes one comparator per workload:

- :func:`bionpu.verify.crispr.compare_against_cas_offinder` — CRISPR
  off-target hits TSV.
- :func:`bionpu.verify.basecalling.compare_against_dorado` — Nanopore
  basecaller FASTQ.

Both return a :class:`VerifyResult` with the boolean verdict, the
canonical normalised hash on each side, and (on divergence) the first
N differing records with byte offsets — so a divergence report is
actionable, not just a pass/fail.

Status
------

Shell. The comparator implementations land during the v0.1 extraction.
Public API is documented here first so it's frozen before migration.
"""

from .types import VerifyResult, VerifyDivergence

__all__ = ["VerifyResult", "VerifyDivergence"]
