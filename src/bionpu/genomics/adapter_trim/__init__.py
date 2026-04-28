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

"""bionpu trim v0 — production adapter trimmer composed on primer_scan v0.

Public API:

* :class:`FastqRecord` — record tuple (header, seq, qual).
* :func:`parse_fastq`, :func:`write_fastq`, :func:`open_fastq` —
  FASTQ I/O (gzip transparent via filename suffix).
* :func:`trim_fastq` — process a FASTQ file, emit trimmed FASTQ.
* :func:`bionpu_trim_main` — argparse CLI entry point.

This is the FIRST production composition built on top of the
silicon-validated CRISPR-shape primitives. ``primer_scan_v0`` is
locked; this module is a thin host wrapper that:

1. Parses FASTQ records.
2. Packs each read to 2-bit MSB-first via
   :func:`bionpu.data.kmer_oracle.pack_dna_2bit`.
3. Dispatches the silicon adapter scan
   (:class:`bionpu.kernels.genomics.primer_scan.BionpuPrimerScan`).
4. Trims at the FIRST forward-strand match (cutadapt -a semantics).

v0 scope: 3' adapter trimming with ``cutadapt -a`` semantics, single
adapter, exact match (no indels, no mismatches), forward-strand only.
RC matches from the silicon kernel are explicitly discarded — cutadapt
``-a`` is forward-only.
"""

from __future__ import annotations

from .fastq import FastqRecord, open_fastq, parse_fastq, write_fastq
from .trimmer import TrimStats, trim_fastq

__all__ = [
    "FastqRecord",
    "TrimStats",
    "bionpu_trim_main",
    "open_fastq",
    "parse_fastq",
    "trim_fastq",
    "write_fastq",
]


def bionpu_trim_main(argv: list[str] | None = None) -> int:
    """Console-script entry point delegating to :mod:`.cli`."""
    from .cli import main

    return main(argv)
